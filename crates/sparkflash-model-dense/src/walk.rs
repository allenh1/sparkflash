// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The dense arm as a Walk.
//!
//! The serving loop is not moved yet: DenseLanes still runs the fused engine, and this adapter
//! is the shape the arm supplies when it does. It routes every Pass to the same free walk the
//! existing loop calls, with the policy the existing loop uses for that pass: token rows pass no
//! wide policy (a row GEMV is the same sum at any batch), and a bulk or paired prompt pass
//! takes the arm's WidePolicy. Its head is the existing host logits path; the device head is
//! phase 4's.
//!
//! The layout and limits are the arm's: attention layers are positional BF16 key/value caches,
//! Gated DeltaNet layers are recurrent, the context is the served context, and a pass is
//! CHUNK_ROWS rows.

use std::sync::{Arc, Mutex};

use sparkflash_model::backend::{BackendError, StepOutput};
use sparkflash_model::ledger::State;
use sparkflash_ops::projection::WidePolicy;
use sparkflash_runtime::{
    Absent, ComputeDeclaration, HeadInput, HeadRow, Hidden, LaneMemory, LaneState, Limits,
    NoVision, Pass, PassKind, PassMask, Region, Segment, Segments, StateKind, StateLayout, Walk,
    WalkCapabilities,
};

use super::{
    CHUNK_ROWS, DEVICE, Mixer, Scratch, Sequence, Shared, head as dense_head, walk as dense_walk,
};

/// The rows of one walk, as offsets into the pass's residual buffer. The existing walk and this
/// adapter use this one layout, so a pass has the same rows whichever one runs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RowPlan {
    /// The total rows.
    pub rows: usize,
    /// The first row of each segment.
    pub offsets: Vec<usize>,
}

/// The row layout of a pass with one segment of each length.
pub(crate) fn row_plan(lengths: &[usize]) -> RowPlan {
    let mut offsets = Vec::with_capacity(lengths.len());
    let mut rows = 0;
    for &length in lengths {
        offsets.push(rows);
        rows += length;
    }
    RowPlan { rows, offsets }
}

/// The wide policy the existing loop uses for a pass shape.
pub(crate) fn policy_for_kind(kind: PassKind, policy: Option<WidePolicy>) -> Option<WidePolicy> {
    match kind {
        PassKind::TokenRows | PassKind::Verify { .. } => None,
        PassKind::Bulk { .. } | PassKind::Paired { .. } => policy,
    }
}

/// The dense arm, exposed to the shared runtime as a Walk.
pub struct DenseWalk {
    shared: Arc<Shared>,
    layout: StateLayout,
    limits: Limits,
    scratch: Mutex<Scratch>,
}

impl DenseWalk {
    /// A walk over shared's weights, with scratch for capacity rows.
    ///
    /// # Errors
    /// A device allocation the scratch cannot make.
    #[allow(dead_code)] // Phase 4's serving loop is the caller; the adapter tests use the row plan.
    pub(crate) fn new(shared: Arc<Shared>, capacity: usize) -> Result<Self, BackendError> {
        let rows = capacity.clamp(1, CHUNK_ROWS);
        let scratch = Scratch::new(Arc::clone(&shared), rows)
            .map_err(|error| BackendError::new(DEVICE, error))?;
        let layout = layout_of(&shared);
        let limits = Limits::new(CHUNK_ROWS, shared.context);
        Ok(Self {
            shared,
            layout,
            limits,
            scratch: Mutex::new(scratch),
        })
    }

    fn scratch(&self) -> Result<std::sync::MutexGuard<'_, Scratch>, BackendError> {
        self.scratch
            .lock()
            .map_err(|_| BackendError::new(DEVICE, "the walk's scratch is poisoned"))
    }
}

/// The arm's state, as the runtime needs to see it: one positional region a GQA layer, one
/// recurrent region for every Gated DeltaNet layer together.
#[allow(dead_code)] // Reached through DenseWalk::new, which phase 4's serving loop calls.
fn layout_of(shared: &Shared) -> StateLayout {
    let mut regions = Vec::new();
    let mut recurrent = 0usize;
    for layer in &shared.layers {
        if let Mixer::Gqa { .. } = layer.mixer {
            regions.push(Region::Positional {
                name: "kv",
                // Keys and values, BF16 a value.
                bytes_per_row: layer.shape.kv_width() * 2 * 2,
            });
        }
    }
    for size in &shared.linear_sizes {
        recurrent = recurrent.saturating_add(*size);
    }
    if recurrent > 0 {
        regions.push(Region::Recurrent {
            name: "gated-deltanet",
            bytes: recurrent,
        });
    }
    StateLayout::new(regions)
}

impl LaneState for Sequence {
    fn position(&self) -> usize {
        self.history.len()
    }

    fn checkpoint(&mut self, at: usize) -> Result<(), BackendError> {
        if at != self.history.len() {
            return Err(BackendError::new(
                DEVICE,
                format!(
                    "the checkpoint is asked for position {at}, the sequence is at {}",
                    self.history.len()
                ),
            ));
        }
        let live = self.linear_regions();
        if let Some(checkpoint) = self.checkpoint.as_mut() {
            checkpoint
                .save(&self.shared.cuda, None, &live, at)
                .map_err(|error| BackendError::new(DEVICE, format!("the rewind point: {error}")))?;
        }
        Ok(())
    }

    fn rewind(&mut self, to: usize) -> Result<(), BackendError> {
        if to > self.history.len() {
            return Err(BackendError::new(
                DEVICE,
                format!(
                    "the rewind is to position {to}, the sequence is at {}",
                    self.history.len()
                ),
            ));
        }
        if to == 0 {
            self.clear_linear()?;
        } else if let Some(checkpoint) = &self.checkpoint {
            let live = self.linear_regions();
            checkpoint
                .restore(&self.shared.cuda, None, &live, to)
                .map_err(|error| BackendError::new(DEVICE, format!("the rewind point: {error}")))?;
        }
        self.history.truncate(to);
        self.cached = 0;
        self.prompt = None;
        self.sampler.truncate(to);
        self.point = None;
        Ok(())
    }
}

impl Walk for DenseWalk {
    type Seq = Sequence;
    type Drafter = Absent<&'static str>;
    type Vision = NoVision;

    const CAPABILITIES: WalkCapabilities = WalkCapabilities {
        passes: PassMask::ALL,
        segments: Segments::Many,
        state: StateKind::Both,
        // The arm's own four rows, word for word the declaration in
        // `sparkflash_model::ARM_LEDGERS`; a test below holds them equal.
        compute: ComputeDeclaration {
            prompt: State::Native("prompt projections on the shared wide tile"),
            token: State::Off("row GEMV; decode.tensor_cores=true is refused"),
            stored: State::Native("BF16, FP8 E4M3 and NVFP4 parts run in their stored formats"),
            rewind: State::Native("a prompt that goes back to the last prompt's end"),
        },
    };

    fn layout(&self) -> &StateLayout {
        &self.layout
    }

    fn open(&self, memory: LaneMemory) -> Result<Self::Seq, BackendError> {
        let _ = memory;
        Sequence::open(Arc::clone(&self.shared)).map_err(|error| BackendError::new(DEVICE, error))
    }

    fn limits(&self) -> Limits {
        self.limits
    }

    fn walk(
        &self,
        segments: &mut [Segment<'_, Self::Seq>],
        pass: Pass<'_>,
    ) -> Result<Hidden, BackendError> {
        let Pass {
            kind,
            constraint: _,
            hook,
        } = pass;
        let lengths: Vec<usize> = segments.iter().map(|segment| segment.rows.len()).collect();
        let plan = row_plan(&lengths);
        let scratch = self.scratch()?;
        let policy = policy_for_kind(kind, self.shared.policy);
        let mut pairs: Vec<(&mut Sequence, &[u32])> = segments
            .iter_mut()
            .map(|segment| (&mut *segment.seq, segment.rows))
            .collect();
        dense_walk(&self.shared, &scratch.buffers, &mut pairs, policy, None)?;
        drop(pairs);
        let hidden = Hidden::new(plan.rows);
        if let Some(hook) = hook {
            hook.after_layers(&hidden)?;
        }
        Ok(hidden)
    }

    fn head(
        &self,
        hidden: &Hidden,
        rows: &mut [HeadRow<'_, Self::Seq>],
    ) -> Result<Vec<StepOutput>, BackendError> {
        if hidden.rows == 0 {
            return Ok(Vec::new());
        }
        let scratch = self.scratch()?;
        let contiguous = rows.iter().enumerate().all(|(index, row)| row.row == index);
        let logits: Vec<Vec<f32>> = if contiguous {
            dense_head(&self.shared, &scratch.buffers, 0, rows.len())?
                .chunks(self.shared.vocab)
                .map(<[f32]>::to_vec)
                .collect()
        } else {
            rows.iter()
                .map(|row| dense_head(&self.shared, &scratch.buffers, row.row, 1))
                .collect::<Result<Vec<_>, _>>()?
        };
        if logits.len() != rows.len() {
            return Err(BackendError::new(
                DEVICE,
                format!(
                    "the head was asked for {} rows and made {}",
                    rows.len(),
                    logits.len()
                ),
            ));
        }
        rows.iter_mut()
            .zip(logits)
            .map(|(row, logits)| {
                match &row.input {
                    HeadInput::Prompt(tokens) => {
                        row.seq.sampler.reset();
                        row.seq.sampler.record_prompt(tokens);
                    }
                    HeadInput::Token(token) => row.seq.sampler.record(*token),
                }
                let chosen = row.seq.sampler.choose(&logits, &row.seq.sampling);
                Ok(StepOutput {
                    logits,
                    next_token: Some(chosen),
                    draft: Vec::new(),
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pass_routes_to_the_policy_the_serving_loop_uses() {
        let policy = Some(WidePolicy::DENSE);
        assert_eq!(policy_for_kind(PassKind::TokenRows, policy), None);
        assert_eq!(policy_for_kind(PassKind::Verify { rows: 4 }, policy), None);
        assert_eq!(policy_for_kind(PassKind::Bulk { rows: 1 }, policy), policy);
        assert_eq!(
            policy_for_kind(PassKind::Paired { chunks: 2 }, policy),
            policy
        );
    }

    #[test]
    fn the_adapter_rows_are_the_existing_walks_rows() {
        // A token step: one row a member, the shape step builds.
        let step_rows = row_plan(&[1, 1, 1, 1]);
        assert_eq!(step_rows.rows, 4);
        assert_eq!(step_rows.offsets, [0, 1, 2, 3]);
        // A bulk prompt chunk: CHUNK_ROWS rows, the shape Sequence::part builds.
        let bulk_rows = row_plan(&[CHUNK_ROWS]);
        assert_eq!(bulk_rows.rows, CHUNK_ROWS);
        assert_eq!(bulk_rows.offsets, [0]);
        // Several prompts' chunks: each segment at its own offset, the shape the runtime asks
        // for when it pairs prompts.
        let paired = row_plan(&[CHUNK_ROWS, 1, 17]);
        assert_eq!(paired.rows, CHUNK_ROWS + 18);
        assert_eq!(paired.offsets, [0, CHUNK_ROWS, CHUNK_ROWS + 1]);
    }

    #[test]
    fn the_layout_is_positional_attention_and_recurrent_gated_deltanet() {
        let positional = StateLayout::new(vec![Region::Positional {
            name: "kv",
            bytes_per_row: 8,
        }]);
        assert_eq!(positional.kind(), StateKind::Positional);
        let both = StateLayout::new(vec![
            Region::Positional {
                name: "kv",
                bytes_per_row: 8,
            },
            Region::Recurrent {
                name: "gated-deltanet",
                bytes: 64,
            },
        ]);
        assert_eq!(both.kind(), StateKind::Both);
        assert!(both.needs_checkpoint());
    }

    /// The adapter's four declared rows are the arm's static declaration, so the two cannot
    /// drift while phase 1 moves the arm onto the runtime.
    #[test]
    fn the_declared_compute_rows_are_the_arm_ledger_rows() {
        use sparkflash_model::ledger::{Strategy, ledger_for_arm};
        let declared = ledger_for_arm("dense").expect("dense's declaration");
        let capabilities = <super::DenseWalk as Walk>::CAPABILITIES.compute;
        for (strategy, stated) in [
            (Strategy::TensorCorePrompt, capabilities.prompt),
            (Strategy::TensorCoreDecode, capabilities.token),
            (Strategy::StoredArithmetic, capabilities.stored),
            (Strategy::Rewind, capabilities.rewind),
        ] {
            let from_table = declared
                .rows
                .iter()
                .find(|(row, _)| *row == strategy)
                .map(|(_, state)| *state)
                .expect("the arm states every strategy");
            assert_eq!(stated, from_table, "{} drifted", strategy.name());
        }
    }

    sparkflash_runtime::conformance!(super::DenseWalk);
}

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
//! The runtime drives it: [runtime] opens a [`sparkflash_runtime::Runtime`] over this walk, and
//! the runtime owns the prompt parts, history, reuse, rewind, lane pool and ledger. This walk
//! routes every Pass to the same free walk the old serving loop called, with the arm's
//! [`WidePolicy::DENSE`] for every pass: a prompt chunk takes the wide tile, and a token step
//! takes the tile once it batches more than the row GEMV's eight rows (below that the policy still
//! picks the row GEMV, so a one-to-eight-row step is the kernel it always was). Its head turns
//! the final residual row into the token on the device for a greedy request, and only copies a
//! logits row when the request samples with temperature or penalties.
//!
//! The layout and limits are the arm's: attention layers are positional BF16 key/value caches,
//! Gated DeltaNet layers are recurrent, the context is the served context, and a pass is
//! CHUNK_ROWS rows.

use std::sync::{Arc, Mutex};

use sparkflash_model::backend::{BackendError, Sampling, StepOutput};
use sparkflash_model::ledger::State;
use sparkflash_ops::projection::WidePolicy;
use sparkflash_runtime::{
    Absent, ComputeDeclaration, HeadInput, HeadRow, Hidden, LaneMemory, LaneState, Limits,
    NoVision, Pass, PassKind, PassMask, Region, Runtime, Segment, Segments, StateKind, StateLayout,
    Walk, WalkCapabilities,
};

use super::{
    CHUNK_ROWS, DEVICE, Mixer, Scratch, Sequence, Shared, TOKEN_DECODE, head as dense_head,
    walk as dense_walk,
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

/// The wide policy a pass shape takes: the arm's policy for every pass.
///
/// [`WidePolicy::DENSE`] keeps the row GEMV at eight rows or fewer, so a one-row or four-lane
/// decode step still runs the kernel its bits were recorded under; a step that batches nine or
/// more rows takes the shared wide tile, as the policy says. Passing `None` here would hide the
/// tile from exactly those steps.
pub(crate) fn policy_for_kind(_kind: PassKind, policy: Option<WidePolicy>) -> Option<WidePolicy> {
    policy
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

/// The runtime over a dense walk for `shared`'s weights and `capacity` lanes: the serving arm the
/// dense backend opens. The runtime owns the prompt parts, history, reuse, rewind, lane pool and
/// ledger; the walk only runs rows and heads them.
///
/// # Errors
/// A device allocation the walk's scratch cannot make, or a `SPARKFLASH_REWIND` that is not a
/// switch.
pub(crate) fn runtime(
    shared: Arc<Shared>,
    capacity: usize,
) -> Result<Runtime<DenseWalk>, BackendError> {
    Runtime::new(DenseWalk::new(shared, capacity)?, 0)
}

/// The arm's state, as the runtime needs to see it: one positional region a GQA layer, one
/// recurrent region for every Gated DeltaNet layer together.
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
        self.sampler.truncate(to);
        Ok(())
    }

    fn room(&self, positions: usize, reset: bool) -> bool {
        self.lane.room(positions, reset)
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
        // `sparkflash_model::ARM_LEDGERS`; a test below holds them equal. The token row is the
        // same constant the profile check that refuses `decode.tensor_cores = true` reads.
        compute: ComputeDeclaration {
            prompt: State::Native("prompt projections on the shared wide tile"),
            token: TOKEN_DECODE,
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

    fn set_sampling(&self, seq: &mut Self::Seq, sampling: Sampling) {
        seq.sampling = sampling;
        seq.sampler.reset();
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
    use sparkflash_runtime::FusedArm;

    use super::*;

    #[test]
    fn every_pass_routes_to_the_dense_policy() {
        let policy = Some(WidePolicy::DENSE);
        for kind in [
            PassKind::TokenRows,
            PassKind::Verify { rows: 4 },
            PassKind::Bulk { rows: 1 },
            PassKind::Paired { chunks: 2 },
        ] {
            assert_eq!(policy_for_kind(kind, policy), policy, "{kind:?}");
        }
    }

    /// The policy's own threshold is dense's legacy input: eight rows or fewer keep the row
    /// GEMV, so a decode step at the served lane count does not move; nine rows take the wide
    /// tile. That boundary is where a bit delta can first appear.
    #[test]
    fn the_dense_policy_keeps_eight_rows_on_the_row_gemv_and_takes_the_tile_at_nine() {
        use sparkflash_ops::projection::{Route, route_bf16, route_fp8};

        let (columns, inner) = (256, 256);
        for rows in 1..=8 {
            assert_eq!(
                route_bf16(Some(WidePolicy::DENSE), columns, inner, rows),
                Route::RowGemv,
                "{rows} rows stay on the row GEMV"
            );
            assert_eq!(
                route_fp8(Some(WidePolicy::DENSE), columns, inner, rows),
                Route::RowGemv,
                "{rows} FP8 rows stay on the tiled E4M3 GEMV"
            );
        }
        assert_eq!(
            route_bf16(Some(WidePolicy::DENSE), columns, inner, 9),
            Route::Wide,
            "nine BF16 rows take dory_mma_wide"
        );
        assert_eq!(
            route_fp8(Some(WidePolicy::DENSE), columns, inner, 9),
            Route::Wide,
            "nine FP8 rows take q4p_dense_fp8"
        );
        assert_eq!(
            route_bf16(None, columns, inner, 9),
            Route::BatchedGemv,
            "without the policy the same rows stay on the row GEMV"
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

    /// One row the runtime handed a dense-shaped walk: the pass kind and each segment's rows.
    type Recorded = (PassKind, Vec<usize>);

    #[derive(Default)]
    struct RecordedSeq {
        rows: Vec<u32>,
    }

    impl LaneState for RecordedSeq {
        fn position(&self) -> usize {
            self.rows.len()
        }

        fn checkpoint(&mut self, _at: usize) -> Result<(), BackendError> {
            Ok(())
        }

        fn rewind(&mut self, to: usize) -> Result<(), BackendError> {
            self.rows.truncate(to);
            Ok(())
        }

        fn room(&self, _positions: usize, _reset: bool) -> bool {
            true
        }
    }

    /// A walk with dense's limits and no device, recording every pass the runtime gives it.
    struct RecordingWalk {
        layout: StateLayout,
        limits: Limits,
        passes: Mutex<Vec<Recorded>>,
    }

    impl Walk for RecordingWalk {
        type Seq = RecordedSeq;
        type Drafter = Absent<&'static str>;
        type Vision = NoVision;

        const CAPABILITIES: WalkCapabilities = WalkCapabilities::all(StateKind::Both);

        fn layout(&self) -> &StateLayout {
            &self.layout
        }

        fn open(&self, _memory: LaneMemory) -> Result<Self::Seq, BackendError> {
            Ok(RecordedSeq::default())
        }

        fn limits(&self) -> Limits {
            self.limits
        }

        fn set_sampling(&self, _seq: &mut Self::Seq, _sampling: Sampling) {}

        fn walk(
            &self,
            segments: &mut [Segment<'_, Self::Seq>],
            pass: Pass<'_>,
        ) -> Result<Hidden, BackendError> {
            let lengths: Vec<usize> = segments.iter().map(|segment| segment.rows.len()).collect();
            self.passes
                .lock()
                .expect("the pass record")
                .push((pass.kind, lengths));
            let mut rows = 0;
            for segment in segments {
                segment.seq.rows.extend_from_slice(segment.rows);
                rows += segment.rows.len();
            }
            Ok(Hidden::new(rows))
        }

        fn head(
            &self,
            hidden: &Hidden,
            rows: &mut [HeadRow<'_, Self::Seq>],
        ) -> Result<Vec<StepOutput>, BackendError> {
            let _ = hidden;
            Ok(rows
                .iter_mut()
                .map(|row| {
                    let token = row.seq.rows.last().copied().unwrap_or(0);
                    StepOutput {
                        logits: vec![0.0; 2],
                        next_token: Some(token),
                        draft: Vec::new(),
                    }
                })
                .collect())
        }
    }

    /// The runtime over a walk with dense's limits: CHUNK_ROWS a pass, a served context.
    fn dense_shaped_runtime() -> Runtime<RecordingWalk> {
        Runtime::new(
            RecordingWalk {
                layout: StateLayout::new(vec![
                    Region::Positional {
                        name: "kv",
                        bytes_per_row: 4,
                    },
                    Region::Recurrent {
                        name: "gated-deltanet",
                        bytes: 16,
                    },
                ]),
                limits: Limits::new(CHUNK_ROWS, 32_768),
                passes: Mutex::new(Vec::new()),
            },
            0,
        )
        .expect("the runtime starts")
    }

    fn prompt_passes(len: usize) -> Vec<Recorded> {
        let runtime = dense_shaped_runtime();
        assert_eq!(
            runtime.chunk_policy().rows(),
            CHUNK_ROWS,
            "dense's chunk size comes from its limits through ChunkPolicy"
        );
        let mut slot = runtime.open_slot().expect("a slot");
        runtime
            .prefill(&mut slot, &vec![7u32; len])
            .expect("the prefill");
        runtime
            .walk()
            .passes
            .lock()
            .expect("the pass record")
            .clone()
    }

    /// The runtime's prompt loop is dense's old one: 256-row chunks and the remainder, in order.
    #[test]
    fn the_runtime_hands_dense_256_row_bulk_chunks() {
        assert_eq!(
            prompt_passes(600),
            vec![
                (PassKind::Bulk { rows: 256 }, vec![256]),
                (PassKind::Bulk { rows: 256 }, vec![256]),
                (PassKind::Bulk { rows: 88 }, vec![88]),
            ],
            "a long prompt runs 256 rows a pass and its remainder last"
        );
        assert_eq!(
            prompt_passes(257),
            vec![
                (PassKind::Bulk { rows: 256 }, vec![256]),
                (PassKind::Bulk { rows: 1 }, vec![1]),
            ]
        );
    }

    /// A prompt that fits the chunk is one whole pass, as the old loop ran it.
    #[test]
    fn a_dense_prompt_that_fits_runs_whole() {
        for len in [1usize, 255, CHUNK_ROWS] {
            assert_eq!(
                prompt_passes(len),
                vec![(PassKind::Bulk { rows: len }, vec![len])],
                "a {len}-row prompt is one pass"
            );
        }
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

    /// The runtime's boot ledger states the token row the profile refusal reads, so a dense boot
    /// and the `decode.tensor_cores` check cannot disagree.
    #[test]
    fn the_runtime_ledger_states_the_refusal_row() {
        use sparkflash_runtime::{Mode, Strategy, ledger_for};
        let ledger = ledger_for::<DenseWalk>();
        assert_eq!(
            ledger.mode(Strategy::TensorCoreDecode),
            Mode::from(TOKEN_DECODE)
        );
        assert!(!ledger.line().contains("the arm does not state it"));
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

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

//! The Qwen4-Exp arm as a [Walk].
//!
//! The shared runtime owns the prompt parts, history, resume, rewind, lane pool and ledger.
//! This file supplies the arm's arithmetic: a prompt chunk runs through the arm's existing
//! prefill forwards (whose chunking stays [super::prefill::PrefillPlan]'s), a token pass runs
//! through the existing batch (one row a member, or a member's token and its drafts), and the
//! head is the arm's own sampler. The few-row tensor-core bodies are untouched.
//!
//! A prompt chunk is split by the prefill plan exactly as the old prompt loop split it, so a
//! prompt's chunk boundaries and therefore its bits do not move. The token pass keeps the
//! arm's speculative decode: a sequence whose last verify already holds the token answers from
//! it, and one that does not batches its token and drafts through the same kernels as before.
//!
//! The walk is kept for when the shared runtime can drive a drafting arm; Qwen serves on its own
//! fused arm ([super::fusion::QwenFusion]) until then, so much of this module is not constructed
//! on the serving path yet.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use sparkflash_model::backend::{BackendError, ModelSession, Sampling, StepOutput};
use sparkflash_model::ledger::State;
use sparkflash_runtime::{
    ChunkPolicy, ComputeDeclaration, Drafter, HeadInput, HeadRow, Hidden, LaneMemory, LaneState,
    Limits, Pass, PassKind, PassMask, Region, Runtime, Segment, Segments, StateKind, StateLayout,
    Walk, WalkCapabilities,
};

use sparkflash_kernels::cuda_driver::Deviceptr;

use super::batch::{BatchBuffers, MAX_BATCH};
use super::{DENSE_ROWS, DeviceSequence, SequenceShared};

/// The Qwen arm, exposed to the shared runtime as a [Walk].
pub(super) struct QwenWalk {
    shared: Arc<SequenceShared>,
    layout: StateLayout,
    limits: Limits,
    /// The token pass's combined row buffers, made on first use and reused. Joining a batch
    /// then costs no allocation.
    batch: Mutex<Option<BatchBuffers>>,
    /// The most members one token pass may hold: the driver's concurrency.
    members: usize,
    /// The steps a token or verify pass already resolved, in segment order, for the head.
    pending: Mutex<VecDeque<StepOutput>>,
}

impl QwenWalk {
    /// A walk over `shared`'s weights, serving sequences of up to `context_limit` tokens, with
    /// room for `capacity` members in one token pass.
    pub(super) fn new(
        shared: Arc<SequenceShared>,
        context_limit: usize,
        capacity: usize,
    ) -> Result<Self, BackendError> {
        let layout = layout_of(&shared.config);
        let limits = Limits::new(pass_rows(shared.prefill.chunk), context_limit);
        Ok(Self {
            shared,
            layout,
            limits,
            batch: Mutex::new(None),
            members: capacity.clamp(1, MAX_BATCH),
            pending: Mutex::new(VecDeque::new()),
        })
    }

    fn batch(
        &self,
        members: &mut [&mut DeviceSequence],
        drafts: &[Vec<u32>],
        confidence: &[Vec<f32>],
        tokens: &[u32],
    ) -> Result<Vec<StepOutput>, BackendError> {
        let mut batch = self
            .batch
            .lock()
            .map_err(|_| BackendError::new("qwen4_batch", "the batch buffers are poisoned"))?;
        if batch.is_none() {
            let score_stride = members.first().map_or(0, |session| session.score_stride);
            let rows = (self.members.max(members.len()) * self.shared.verify_rows.max(1)).max(1);
            *batch = Some(BatchBuffers::new(
                Arc::clone(&self.shared),
                score_stride,
                rows,
            )?);
        }
        let allocation = batch.as_mut().expect("the buffers were just made");
        let steps = allocation.step(members, drafts, confidence, tokens)?;
        if steps.len() != members.len() {
            return Err(BackendError::new(
                "qwen4_batch",
                format!(
                    "a fused step of {} member(s) answered {} step(s)",
                    members.len(),
                    steps.len()
                ),
            ));
        }
        Ok(steps)
    }

    /// A prompt chunk: the prefill plan's forwards, each on the arm's own prompt path.
    fn prompt_chunk(&self, seq: &mut DeviceSequence, rows: &[u32]) -> Result<(), BackendError> {
        seq.consumed.extend_from_slice(rows);
        for range in self.shared.prefill.forwards(rows.len()) {
            let start = seq.position;
            let end_of_chunk = range.end;
            let chunk = &rows[range];
            // The head of a prompt runs once, from [Walk::head], on the last chunk's row.
            seq.forward_batch(chunk, usize::MAX, 0)?;
            if seq.drafting() {
                let end = seq.position + rows.len().saturating_sub(end_of_chunk);
                let commits: Vec<_> = seq.backend.dflash.as_ref().map_or_else(Vec::new, |draft| {
                    draft
                        .config
                        .window_commits((start, chunk.len()), end, DENSE_ROWS)
                        .collect()
                });
                for (row, rows, position) in commits {
                    seq.dflash_commit(row, rows, position)?;
                }
            } else if seq.spec_enabled {
                seq.assemble_mtp_rows(chunk.len())?;
                seq.mtp_pass(chunk, start, false)?;
                seq.stash_last_hyper(chunk.len())?;
            }
        }
        Ok(())
    }

    /// One token a sequence, with the arm's drafts where it has them.
    fn token_pass(
        &self,
        segments: &mut [Segment<'_, DeviceSequence>],
    ) -> Result<Hidden, BackendError> {
        let mut steps: Vec<Option<StepOutput>> = (0..segments.len()).map(|_| None).collect();
        let mut forward: Vec<usize> = Vec::new();
        for index in 0..segments.len() {
            let segment = &mut segments[index];
            let token = segment.rows.first().copied().ok_or_else(|| {
                BackendError::new("qwen4_device", "a token pass holds a segment with no rows")
            })?;
            if segment.rows.len() != 1 {
                return Err(BackendError::new(
                    "qwen4_device",
                    "a token pass holds one row a segment",
                ));
            }
            if let Some(step) = segment.seq.feed(token)? {
                steps[index] = Some(step);
            } else {
                forward.push(index);
            }
        }
        if !forward.is_empty() {
            let mut drafts = Vec::with_capacity(forward.len());
            let mut confidence = Vec::with_capacity(forward.len());
            let mut tokens = Vec::with_capacity(forward.len());
            for &index in &forward {
                let seq = &mut *segments[index].seq;
                let draft = std::mem::take(&mut seq.spec_drafts);
                let sure = std::mem::take(&mut seq.spec_confidence);
                tokens.push(segments[index].rows[0]);
                drafts.push(draft);
                confidence.push(sure);
            }
            let mut needs = vec![false; segments.len()];
            for &index in &forward {
                needs[index] = true;
            }
            let mut members: Vec<&mut DeviceSequence> = Vec::with_capacity(forward.len());
            for (index, segment) in segments.iter_mut().enumerate() {
                if needs[index] {
                    members.push(&mut *segment.seq);
                }
            }
            let resolved = self.batch(&mut members, &drafts, &confidence, &tokens)?;
            for (&index, step) in forward.iter().zip(resolved) {
                steps[index] = Some(step);
            }
        }
        let mut pending = self
            .pending
            .lock()
            .map_err(|_| BackendError::new("qwen4_device", "the head queue is poisoned"))?;
        pending.clear();
        for step in steps {
            pending.push_back(step.ok_or_else(|| {
                BackendError::new("qwen4_device", "a token pass left a member without a step")
            })?);
        }
        Ok(Hidden::new(segments.len()))
    }

    /// A verify pass: every segment's anchor token and drafts, one batch.
    fn verify_pass(
        &self,
        segments: &mut [Segment<'_, DeviceSequence>],
    ) -> Result<Hidden, BackendError> {
        let mut drafts = Vec::with_capacity(segments.len());
        let mut confidence = Vec::with_capacity(segments.len());
        let mut tokens = Vec::with_capacity(segments.len());
        for segment in segments.iter() {
            let (anchor, rest) = segment.rows.split_first().ok_or_else(|| {
                BackendError::new("qwen4_device", "a verify segment holds no rows")
            })?;
            tokens.push(*anchor);
            drafts.push(rest.to_vec());
            confidence.push(vec![0.0; rest.len()]);
        }
        let mut members: Vec<&mut DeviceSequence> = segments
            .iter_mut()
            .map(|segment| &mut *segment.seq)
            .collect::<Vec<_>>();
        let steps = self.batch(&mut members, &drafts, &confidence, &tokens)?;
        let mut pending = self
            .pending
            .lock()
            .map_err(|_| BackendError::new("qwen4_device", "the head queue is poisoned"))?;
        pending.clear();
        pending.extend(steps);
        Ok(Hidden::new(segments.len()))
    }
}

/// The arm's state, as the runtime needs to see it.
fn layout_of(config: &crate::Qwen4ExpConfig) -> StateLayout {
    let sparse = config
        .layer_types
        .iter()
        .filter(|kind| matches!(kind, crate::config::LayerKind::Sparse))
        .count()
        + usize::from(config.serves_mtp());
    let kv = config.kv_proj_width() * 2 * 2;
    let index = config.indexer_head_dim * 2;
    let mut regions = Vec::new();
    if sparse > 0 {
        regions.push(Region::Positional {
            name: "qsa-kv",
            bytes_per_row: sparse * (kv + index),
        });
    }
    let linear = config
        .layer_types
        .iter()
        .filter(|kind| matches!(kind, crate::config::LayerKind::Linear))
        .count();
    let history = config.linear_conv_kernel_dim.saturating_sub(1).max(1);
    let conv = linear * history * config.linear_conv_width() * 4;
    let recurrent = linear
        * config.linear_num_value_heads
        * config.linear_key_head_dim
        * config.linear_value_head_dim
        * 4;
    let ple_layers = config.ple_layer_ids.len();
    let ple = ple_layers
        * config.ple_conv_kernel_size.saturating_sub(1).max(1)
        * config.ngram_size
        * config.hyper_width()
        * 4;
    if conv + recurrent + ple > 0 {
        regions.push(Region::Recurrent {
            name: "gated-deltanet",
            bytes: conv + recurrent + ple,
        });
    }
    StateLayout::new(regions)
}

/// The pass width the runtime's [ChunkPolicy] can return for the arm's own prefill chunk.
///
/// The policy quantises to its documented sizes, so a limit that is not one of them would move
/// the arm's chunk boundary. This picks the smallest documented width at or above the plan's
/// chunk that it divides, so the walk's prefill split lands on the plan's boundaries; a chunk
/// with no such width keeps its own value (the policy then rounds it down, and the arm's prompt
/// boundaries move with it).
fn pass_rows(chunk: usize) -> usize {
    let chunk = chunk.max(1);
    ChunkPolicy::sizes()
        .iter()
        .copied()
        .find(|size| *size >= chunk && size % chunk == 0)
        .or_else(|| {
            ChunkPolicy::sizes()
                .iter()
                .copied()
                .find(|size| *size >= chunk)
        })
        .unwrap_or(chunk)
}

/// The arm's runtime: the fused engine and the group mirror see this one type.
///
/// # Errors
/// A device allocation or a [sparkflash_runtime::REWIND_ENV] that is not a switch.
pub(super) fn runtime(
    shared: Arc<SequenceShared>,
    context_limit: usize,
    capacity: usize,
) -> Result<Runtime<QwenWalk>, BackendError> {
    Runtime::new(QwenWalk::new(shared, context_limit, capacity)?, 0)
}

/// The arm's draft source: the checkpoint's MTP stage, or a loaded DFlash 2 head.
///
/// The runtime's drafting methods are a later phase; the walk itself already drafts after each
/// prompt and each step, so the ledger states the source rather than `Absent`.
pub(super) struct QwenDrafter;

impl Drafter<QwenWalk> for QwenDrafter {
    const PRESENT: bool = true;

    fn describe(&self) -> String {
        "DFlash 2 or MTP".to_owned()
    }
}

impl Walk for QwenWalk {
    type Seq = DeviceSequence;
    type Drafter = QwenDrafter;
    type Vision = sparkflash_runtime::NoVision;

    const CAPABILITIES: WalkCapabilities = WalkCapabilities {
        passes: PassMask::TOKEN_ROWS
            .union(PassMask::BULK)
            .union(PassMask::VERIFY),
        segments: Segments::Many,
        state: StateKind::Both,
        // The four rows the arm states in `sparkflash_model::ARM_LEDGERS`; a test holds them
        // equal so the walk and the table cannot drift.
        compute: ComputeDeclaration {
            prompt: State::Native("prompt chunks on the wide tile and the grouped expert MMAs"),
            token: State::Native("few-row twins of the tile at every row count"),
            stored: State::Native("NVFP4 weights with BF16 activation pairs"),
            rewind: State::Native("a prompt that goes back to the last prompt's end"),
        },
    };

    fn layout(&self) -> &StateLayout {
        &self.layout
    }

    fn open(&self, memory: LaneMemory) -> Result<Self::Seq, BackendError> {
        let _ = memory;
        DeviceSequence::allocate(Arc::clone(&self.shared), self.limits.context)
    }

    fn limits(&self) -> Limits {
        self.limits
    }

    /// A token pass holds at most the batch's four members, though a prompt pass holds a whole
    /// prefill chunk.
    fn max_segments(&self) -> usize {
        self.members
    }

    fn set_sampling(&self, seq: &mut Self::Seq, sampling: Sampling) {
        seq.set_sampling(sampling);
    }

    fn walk(
        &self,
        segments: &mut [Segment<'_, Self::Seq>],
        pass: Pass<'_>,
    ) -> Result<Hidden, BackendError> {
        let hidden = match pass.kind {
            PassKind::TokenRows => self.token_pass(segments)?,
            PassKind::Verify { .. } => self.verify_pass(segments)?,
            PassKind::Bulk { .. } => {
                if segments.len() != 1 {
                    return Err(BackendError::new(
                        "qwen4_device",
                        "a bulk prompt pass holds one segment",
                    ));
                }
                let segment = &mut segments[0];
                let rows = segment.rows.len();
                self.pending
                    .lock()
                    .map_err(|_| BackendError::new("qwen4_device", "the head queue is poisoned"))?
                    .clear();
                let (seq, tokens) = (&mut *segment.seq, segment.rows);
                self.prompt_chunk(seq, tokens)?;
                Hidden::new(rows)
            }
            PassKind::Paired { .. } => {
                return Err(BackendError::new(
                    "qwen4_device",
                    "the Qwen arm does not pair several prompts in one pass",
                ));
            }
        };
        if let Some(hook) = pass.hook {
            hook.after_layers(&hidden)?;
        }
        Ok(hidden)
    }

    fn head(
        &self,
        hidden: &Hidden,
        rows: &mut [HeadRow<'_, Self::Seq>],
    ) -> Result<Vec<StepOutput>, BackendError> {
        let mut pending = self
            .pending
            .lock()
            .map_err(|_| BackendError::new("qwen4_device", "the head queue is poisoned"))?;
        if !pending.is_empty() {
            let steps: Vec<StepOutput> = pending.drain(..).collect();
            if steps.len() != rows.len() {
                return Err(BackendError::new(
                    "qwen4_device",
                    format!(
                        "the head was asked for {} row(s) and {} step(s) are pending",
                        rows.len(),
                        steps.len()
                    ),
                ));
            }
            return Ok(steps);
        }
        drop(pending);
        // A prompt head: the last chunk's deferred row, folded into logits here and chosen with
        // the prompt in the sampler.
        let logits = match rows.first_mut() {
            Some(row) => row.seq.deferred_logits()?,
            None => Vec::new(),
        };
        if logits.is_empty() {
            return Err(BackendError::new(
                "qwen4_device",
                "the head has no prompt logits",
            ));
        }
        let mut steps = Vec::with_capacity(rows.len());
        for row in rows.iter_mut() {
            let token = match &row.input {
                HeadInput::Prompt(tokens) => {
                    row.seq.sampler.record_prompt(tokens);
                    let position = row.seq.position;
                    // The point is the prompt's end, before any draft past it, exactly as the
                    // old prompt loop took it: the MTP stash the drafts rewrite is part of the
                    // point, so saving after them would restore a different state.
                    row.seq.keep_point()?;
                    let next = row.seq.next(&logits);
                    if row.seq.drafting() {
                        if let Some(token) = next {
                            row.seq.dflash_propose(token, position)?;
                        }
                    } else if row.seq.spec_enabled
                        && let Some(token) = next
                    {
                        row.seq.draft_chain(token, position)?;
                    }
                    next
                }
                HeadInput::Token(token) => {
                    row.seq.sampler.record(*token);
                    let next = row.seq.next(&logits);
                    if row.seq.spec_enabled
                        && let Some(token) = next
                    {
                        let position = row.seq.position;
                        row.seq.draft_chain(token, position)?;
                    }
                    next
                }
            };
            steps.push(StepOutput {
                next_token: token,
                logits: logits.clone(),
                draft: Vec::new(),
            });
        }
        let _ = hidden;
        Ok(steps)
    }
}

impl LaneState for DeviceSequence {
    fn position(&self) -> usize {
        self.position
    }

    fn checkpoint(&mut self, at: usize) -> Result<(), BackendError> {
        if self.rewind_point() == Some(at) {
            return Ok(());
        }
        if at != self.position {
            return Err(BackendError::new(
                "qwen4_device",
                format!(
                    "a checkpoint is asked for position {at}, the sequence is at {}",
                    self.position
                ),
            ));
        }
        self.keep_point()
    }

    fn rewind(&mut self, to: usize) -> Result<(), BackendError> {
        self.rewind_state(to)
    }

    fn room(&self, positions: usize, reset: bool) -> bool {
        DeviceSequence::room(self, positions, reset)
    }
}

impl DeviceSequence {
    /// The runtime's rewind: a prompt end restores the arm's point, and zero clears the state a
    /// fresh sequence would have.
    pub(super) fn rewind_state(&mut self, to: usize) -> Result<(), BackendError> {
        if to > self.position {
            return Err(BackendError::new(
                "qwen4_device",
                format!(
                    "the rewind is to position {to}, the sequence is at {}",
                    self.position
                ),
            ));
        }
        if to == self.position {
            return Ok(());
        }
        if to == 0 {
            return self.clear_state();
        }
        self.rewind_point()
            .is_some_and(|point| point == to)
            .then_some(())
            .ok_or_else(|| {
                BackendError::new(
                    "qwen4_device",
                    format!(
                        "the sequence holds no rewind point at position {to} (it holds {:?})",
                        self.rewind_point()
                    ),
                )
            })?;
        DeviceSequence::rewind(self, to)
    }

    /// Puts the sequence back to the state a fresh allocation holds: no positions, no state.
    fn clear_state(&mut self) -> Result<(), BackendError> {
        let cuda = &self.backend.cuda;
        cuda.make_current()
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        let config = &self.backend.config;
        let history = config.linear_conv_kernel_dim.saturating_sub(1).max(1);
        let conv_elems = history * config.linear_conv_width();
        let rec_elems = config.linear_num_value_heads
            * config.linear_key_head_dim
            * config.linear_value_head_dim;
        let ple_elems = (config.ple_conv_kernel_size.saturating_sub(1).max(1))
            * config.ngram_size
            * config.hyper_width();
        let zero_gpu = |pointer: Deviceptr, elements: usize| -> Result<(), BackendError> {
            let blocks = elements.div_ceil(256).max(1) as u32;
            cuda.launch_on(
                &self.stream,
                self.backend.kernels.zero,
                (blocks, 1, 1),
                (256, 1, 1),
                &[pointer.0, elements as u64],
            )
            .map_err(|error| BackendError::new("qwen4_device", error))
        };
        for &pointer in self.gdn_conv.iter().flatten() {
            zero_gpu(pointer, conv_elems)?;
        }
        for &pointer in self.gdn_recurrent.iter().flatten() {
            zero_gpu(pointer, rec_elems)?;
        }
        for &pointer in self.ple_conv.iter().flatten() {
            zero_gpu(pointer, ple_elems)?;
        }
        zero_gpu(self.mtp_stash, config.hyper_width())?;
        self.position = 0;
        self.consumed.clear();
        self.prefill_cached = 0;
        self.pending_head = None;
        self.spec_queue.clear();
        self.spec_served.clear();
        self.spec_drafts.clear();
        self.spec_confidence.clear();
        self.ple_hashers = fresh_hashers(config)?;
        self.point_hashers = fresh_hashers(config)?;
        if let Some(point) = self.point.as_mut() {
            point.forget();
        }
        Ok(())
    }
}

/// A fresh set of n-gram windows, one per PLE layer.
fn fresh_hashers(
    config: &crate::Qwen4ExpConfig,
) -> Result<Vec<Option<crate::ngram::PleHasher>>, BackendError> {
    let mut hashers = Vec::with_capacity(config.num_hidden_layers);
    for index in 0..config.num_hidden_layers {
        if let Some(ple_layer_index) = config.ple_layer_ids.iter().position(|id| *id == index + 1) {
            hashers.push(Some(
                crate::ngram::PleHasher::new(config, ple_layer_index)
                    .map_err(|error| BackendError::new("qwen4_device", error))?,
            ));
        } else {
            hashers.push(None);
        }
    }
    Ok(hashers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_model::ledger::{Strategy as LedgerStrategy, ledger_for_arm};
    use sparkflash_runtime::ChunkPolicy;

    use crate::config::Qwen4ExpConfig;
    use crate::device::prefill::{DEFAULT_PREFILL_CHUNK, PrefillPlan};

    /// The walk's four declared rows are the arm's static declaration, so the runtime's ledger
    /// and the `sparkflash arms --ledger` table cannot drift while Qwen moves onto the runtime.
    #[test]
    fn the_declared_compute_rows_are_the_arm_ledger_rows() {
        let declared = ledger_for_arm("Qwen").expect("Qwen's declaration");
        let capabilities = <QwenWalk as Walk>::CAPABILITIES.compute;
        for (strategy, stated) in [
            (LedgerStrategy::TensorCorePrompt, capabilities.prompt),
            (LedgerStrategy::TensorCoreDecode, capabilities.token),
            (LedgerStrategy::StoredArithmetic, capabilities.stored),
            (LedgerStrategy::Rewind, capabilities.rewind),
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

    /// The runtime's chunk policy is the arm's own prefill chunk: a prompt's boundaries do not
    /// move when the runtime drives it.
    #[test]
    fn the_runtime_chunk_is_the_prefill_plans_chunk() {
        for chunk in [DEFAULT_PREFILL_CHUNK, super::super::TOKEN_PREFILL_CHUNK] {
            let plan = PrefillPlan::new(chunk, false);
            let limits = Limits::new(pass_rows(plan.chunk), 4_096);
            let rows = ChunkPolicy::for_limits(limits).rows();
            assert!(rows >= plan.chunk, "{chunk} rows must be a whole pass");
            assert_eq!(
                rows % plan.chunk,
                0,
                "{chunk} rows: the runtime's pass splits on the plan's boundary"
            );
        }
        assert_eq!(pass_rows(DEFAULT_PREFILL_CHUNK), DEFAULT_PREFILL_CHUNK);
        assert_eq!(pass_rows(super::super::TOKEN_PREFILL_CHUNK), 256);
    }

    /// The arm's state is both kinds: the QSA caches are positional and the Gated DeltaNet
    /// recurrence is not, so the runtime must checkpoint and truncate.
    #[test]
    fn the_layout_is_positional_attention_and_recurrent_gated_deltanet() {
        let layout = layout_of(&Qwen4ExpConfig::tiny());
        assert_eq!(layout.kind(), StateKind::Both);
        assert!(layout.bytes_per_row() > 0, "the KV cache is positional");
        assert!(layout.needs_checkpoint(), "the recurrence needs a point");
    }

    /// The walk takes the passes the generated checks read: a prompt chunk and many segments.
    #[test]
    fn the_capabilities_support_every_generated_check() {
        let caps = <QwenWalk as Walk>::CAPABILITIES;
        assert!(caps.passes.contains(PassMask::BULK));
        assert!(caps.passes.contains(PassMask::TOKEN_ROWS));
        assert_eq!(caps.segments, Segments::Many);
        assert_eq!(caps.state, StateKind::Both);
        assert_eq!(pass_rows(DEFAULT_PREFILL_CHUNK), 4_096);
        let walk_batch = super::super::batch::MAX_BATCH;
        assert_eq!(walk_batch, 4);
        sparkflash_runtime::check_claims::<QwenWalk>(sparkflash_runtime::CONFORMANCE_CHECKS)
            .expect("every generated check is supported");
    }

    sparkflash_runtime::conformance!(QwenWalk);
}

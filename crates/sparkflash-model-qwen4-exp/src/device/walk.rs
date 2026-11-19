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
//! Qwen serves through this walk: `Qwen4ExpDeviceBackend::enable_fusion` and
//! `enable_group_fusion` start the runtime over it, so the fragment is the serving path.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use sparkflash_model::backend::{BackendError, ModelSession, Sampling, Speculation, StepOutput};
use sparkflash_model::ledger::State;
use sparkflash_runtime::{
    ChunkPolicy, ComputeDeclaration, Draft, Drafter, ExpertUnionWall, HeadInput, HeadRow, Hidden,
    LaneMemory, LaneState, Limits, Pass, PassKind, PassMask, Region, Runtime, Segment, Segments,
    StateKind, StateLayout, StepWall, Walk, WalkCapabilities,
};

use sparkflash_kernels::cuda_driver::Deviceptr;

use super::batch::{BatchBuffers, MAX_BATCH};
use super::{DENSE_ROWS, DeviceSequence, SequenceShared};

/// A shared walk's cost at TP=1 on a GB10-class node, from the block drafter's own step cost
/// ([super::DFLASH2_STEP_COST]: 59 ms a step and 8.4 ms a verified draft): about 10 new experts
/// a row at 0.8 ms each, the rest of a step fixed, and a few milliseconds a member past the first
/// for its own attention, recurrence and draft. The expert counts are the checkpoint's.
pub(super) const JOINT_WALL: ExpertUnionWall = ExpertUnionWall {
    fixed_ms: 50.4,
    expert_ms: 0.81,
    row_ms: 0.5,
    seq_ms: 6.0,
    experts: 0,
    top_k: 0,
};

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
    /// The arm's draft source, behind the runtime's round contract.
    drafter: QwenDrafter,
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
        let limits = walk_limits(shared.prefill.chunk, context_limit, capacity);
        let members = limits.members;
        let joint = sparkflash_runtime::joint_rounds()
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        let wall = ExpertUnionWall {
            experts: shared.config.num_experts,
            top_k: shared.config.num_experts_per_tok,
            ..JOINT_WALL
        };
        let drafter = QwenDrafter {
            depth: shared.verify_rows.saturating_sub(1),
            wall,
            joint,
        };
        Ok(Self {
            shared,
            layout,
            limits,
            batch: Mutex::new(None),
            members,
            pending: Mutex::new(VecDeque::new()),
            drafter,
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
    ///
    /// `prompt_end` is the pass's [Pass::prompt_end]: the rows of the prompt still to come.
    /// The drafter's commit window reaches the prompt's true end, as the old prompt loop's did,
    /// so a non-final pass does not stop the window at the chunk boundary.
    ///
    /// Only the prompt's last forward heads a row. The old loop passed `logit_rows=0` on every
    /// chunk but the last, which skipped each earlier chunk's head-row gather (a collective in a
    /// TP group); `usize::MAX` defers the last one's head to [Walk::head].
    fn prompt_chunk(
        &self,
        seq: &mut DeviceSequence,
        rows: &[u32],
        prompt_end: Option<usize>,
    ) -> Result<(), BackendError> {
        // A proposal from the previous round anchors on the old end, which this prompt may
        // replace; the fusion's begin_prompt cleared the same three lists. The final chunk's
        // head drafts afresh.
        seq.spec_drafts.clear();
        seq.spec_confidence.clear();
        seq.spec_served.clear();
        seq.consumed.extend_from_slice(rows);
        let end = seq.position + rows.len() + prompt_end.unwrap_or(0);
        let last_pass = prompt_end.is_none_or(|after| after == 0);
        for range in self.shared.prefill.forwards(rows.len()) {
            let start = seq.position;
            let last = last_pass && range.end == rows.len();
            let chunk = &rows[range];
            let logit_rows = if last { usize::MAX } else { 0 };
            seq.forward_batch(chunk, logit_rows, 0)?;
            if seq.drafting() {
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
            // The drafter's own confidence in these leading drafts, as the old fusion's round
            // passed it to the batch: a policy calibrated on it must not be fed zeroes.
            confidence.push(
                segment
                    .seq
                    .spec_confidence
                    .iter()
                    .take(rest.len())
                    .copied()
                    .collect(),
            );
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

/// The limits the runtime sees: the arm's own prefill chunk for a pass, the context, and the
/// batch's members.
///
/// The member count is the verify width, not the prompt chunk: a verify of `members` members is
/// `members x (1 + depth)` rows, and the batch buffers hold exactly that many. Capping it at
/// [MAX_BATCH] is the arm's own batch width.
fn walk_limits(chunk: usize, context: usize, capacity: usize) -> Limits {
    Limits::new(pass_rows(chunk), context).with_members(capacity.clamp(1, MAX_BATCH))
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
/// With `SPARKFLASH_JOINT_ROUNDS=off` the arm drafts nothing (the fusion's own policy): the
/// walk opens each sequence with speculation off, and the empty chain states it as well.
pub(super) struct QwenDrafter {
    /// The most drafts the verify batch holds, past the anchor.
    depth: usize,
    wall: ExpertUnionWall,
    /// Whether the arm drafts at all with the joint switch as set.
    joint: bool,
}

impl Drafter<QwenWalk> for QwenDrafter {
    const PRESENT: bool = true;
    /// The batch already folded each member's rows into its step.
    type Rows = StepOutput;

    fn wall(&self) -> &dyn StepWall {
        &self.wall
    }

    fn open_round(&self, seq: &mut DeviceSequence, token: u32) -> Result<Draft, BackendError> {
        seq.backend
            .cuda
            .make_current()
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        if let Some(step) = seq.feed(token)? {
            return Ok(Draft::Held(step));
        }
        let mut chain = seq.spec.chain(&seq.spec_confidence, seq.spec_drafts.len());
        chain.truncate(seq.spec_drafts.len());
        let drafts = std::mem::take(&mut seq.spec_drafts);
        // The confidence stays on the sequence: [QwenWalk::verify_pass] hands it back to the
        // batch, which records the round's calibration. Taking it here would make every verify
        // look like a drafter that offers no probabilities.
        if !self.joint {
            return Ok(Draft::Chain {
                confidence: Vec::new(),
                drafts: Vec::new(),
            });
        }
        Ok(Draft::Chain {
            confidence: chain,
            drafts,
        })
    }

    fn verify(
        &self,
        walk: &QwenWalk,
        segments: &mut [Segment<'_, DeviceSequence>],
        _depths: &[usize],
    ) -> Result<Vec<Self::Rows>, BackendError> {
        let members = segments.len();
        walk.verify_pass(segments)?;
        let mut pending = walk
            .pending
            .lock()
            .map_err(|_| BackendError::new("qwen4_device", "the head queue is poisoned"))?;
        let steps: Vec<StepOutput> = pending.drain(..).collect();
        if steps.len() != members {
            return Err(BackendError::new(
                "qwen4_device",
                format!(
                    "a verify of {members} member(s) answered {} step(s)",
                    steps.len()
                ),
            ));
        }
        Ok(steps)
    }

    fn settle(
        &self,
        _seq: &mut DeviceSequence,
        _chunk: &[u32],
        rows: Self::Rows,
    ) -> Result<StepOutput, BackendError> {
        // DeviceSequence::resolve_verify already settled the round inside the batch.
        Ok(rows)
    }

    /// The most rows one verify of `seqs` members may hold: every member's anchor and its
    /// drafts, which is the batch buffers' width. `share_rows` spends it on drafts only, so a
    /// chain's own depth still bounds what a member takes.
    fn budget(&self, seqs: usize) -> usize {
        seqs.saturating_mul(1 + self.depth)
    }

    fn ready(&self, seq: &DeviceSequence, token: u32) -> bool {
        seq.spec_queue
            .first()
            .is_some_and(|pending| pending.token == token)
    }

    fn give_back(&self, seq: &mut DeviceSequence, to: usize) -> Result<(), BackendError> {
        let count = seq.consumed.len().saturating_sub(to);
        if count == 0 {
            return Ok(());
        }
        let served = seq.spec_served.len();
        if count > served {
            return Err(BackendError::new(
                "qwen4_device",
                format!("{count} answers to give back, and the last verify served {served}"),
            ));
        }
        let pending = seq
            .spec_served
            .drain(served - count..)
            .next()
            .expect("the served answers were just counted");
        seq.spec_queue.clear();
        seq.unwind(&pending)?;
        seq.consumed.truncate(seq.consumed.len() - count);
        let history = seq.sampler.history().len();
        seq.sampler.truncate(history - count);
        Ok(())
    }

    fn speculation(&self, seq: &DeviceSequence) -> Speculation {
        seq.spec.stats()
    }

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
        let mut seq = DeviceSequence::allocate(Arc::clone(&self.shared), self.limits.context)?;
        // Without joint rounds the arm drafts nothing, exactly as the fusion's own policy did: a
        // member would otherwise run the MTP's prompt pass and then throw every chain away.
        seq.spec_enabled &= self.drafter.joint;
        Ok(seq)
    }

    fn limits(&self) -> Limits {
        self.limits
    }

    fn drafter(&self) -> &Self::Drafter {
        &self.drafter
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
                self.prompt_chunk(seq, tokens, pass.prompt_end)?;
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
                HeadInput::Verify { rows, row: index } => {
                    row.seq
                        .sampler
                        .record(rows.get(*index).copied().unwrap_or_default());
                    row.seq.next(&logits)
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
    use sparkflash_runtime::{ChunkPolicy, Fixture, WalkFixture};

    use crate::config::Qwen4ExpConfig;
    use crate::device::prefill::{DEFAULT_PREFILL_CHUNK, PrefillPlan};

    /// The stored bytes a token pass must read at this rank, and the floor the plan states.
    fn token_bytes(
        architecture: &sparkflash_model::Architecture,
        experts: usize,
        top_k: usize,
    ) -> Result<(u64, u64), BackendError> {
        let tp = sparkflash_model::TpPlan::new(1, 0)
            .map_err(|error| BackendError::new("fixture", error.detail))?;
        let tensors =
            sparkflash_ops::tensor_plan::ArchitectureTensorPlan::served_and_mtp(architecture, tp)
                .map_err(|error| BackendError::new("fixture", error.to_string()))?;
        let total = u64::try_from(tensors.rank_bytes()).unwrap_or(u64::MAX);
        let expert = u64::try_from(tensors.role_bytes(sparkflash_ops::tensor_plan::Role::Experts))
            .unwrap_or(u64::MAX);
        let floor = total.saturating_sub(expert);
        let share = if experts == 0 {
            0
        } else {
            expert.saturating_mul(top_k as u64) / experts as u64
        };
        Ok((floor.saturating_add(share), floor))
    }

    /// The stored format names the fixture's classes use.
    fn format_rule(
        format: sparkflash_model::architecture::WeightFormat,
    ) -> sparkflash_runtime::FormatRule {
        use sparkflash_model::architecture::WeightFormat;
        let name = match format {
            WeightFormat::Bf16 => "BF16",
            WeightFormat::Fp8Tile { .. } => "FP8 E4M3",
            WeightFormat::MxFp8 { .. } => "MX FP8",
            WeightFormat::NvFp4 { .. } | WeightFormat::MxFp4 { .. } => "NVFP4",
        };
        sparkflash_runtime::FormatRule::stored(name)
    }

    /// The Qwen fixture: the synthetic tiny checkpoint, opened through the arm's own TP=1
    /// constructor. It is a real checkpoint in the released storage formats, not a served image.
    impl WalkFixture for QwenWalk {
        fn fixture(world: usize) -> Result<Fixture<Self>, BackendError> {
            if world > 1 {
                return Err(BackendError::new(
                    "group",
                    "the Qwen fixture is one rank; TP 2 and TP 4 need a group",
                ));
            }
            // The Qwen3.5/3.6 spelling, not `tiny`: its family is not the canonical Qwen3.8
            // target, so the hub cache's released DFlash 2 drafter (2,560 wide) is not loaded
            // into a 128-wide fixture and refused. The tiny MTP stage still exercises the round.
            let config = Qwen4ExpConfig::tiny_qwen35_moe(crate::config::DenseQuant::Fp8Block);
            let directory = tempfile::tempdir()
                .map_err(|error| BackendError::new("fixture", error.to_string()))?;
            crate::write_synthetic_checkpoint(directory.path(), &config, 7)
                .map_err(|error| BackendError::new("fixture", error))?;
            let backend = crate::device::Qwen4ExpDeviceBackend::open(
                directory.path(),
                config.max_position_embeddings,
            )?;
            let walk = backend.walk_arm(4)?;
            let (bytes, floor) = token_bytes(
                &config.architecture,
                config.num_experts,
                config.num_experts_per_tok,
            )?;
            let formats = vec![
                format_rule(config.architecture.quantization.dense_format),
                format_rule(config.architecture.quantization.expert_format),
            ];
            Ok(Fixture::new(
                walk,
                "the Qwen3.5-MoE tiny synthetic checkpoint",
                bytes,
                floor,
                formats,
                Some(Box::new(directory)),
            ))
        }
    }

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

    /// The engine's member cap is the verify width, not the prompt chunk: a four-row batch
    /// steps at most four members, though its prompt pass holds the whole prefill chunk.
    #[test]
    fn the_member_cap_is_the_verify_width_not_the_prompt_chunk() {
        let limits = walk_limits(DEFAULT_PREFILL_CHUNK, 4_096, 9);
        assert_eq!(limits.rows, 4_096, "a prompt pass keeps the arm's chunk");
        assert_eq!(limits.members, 4, "a batch is the arm's verify width");
        assert_eq!(walk_limits(DEFAULT_PREFILL_CHUNK, 4_096, 0).members, 1);
        assert_eq!(walk_limits(DEFAULT_PREFILL_CHUNK, 4_096, 2).members, 2);
    }

    /// The drafter's budget is the verify's whole width: every member's anchor and its depth.
    /// The batch buffers hold this many rows, so a wider budget would claim memory the verify
    /// does not have and a narrower one would refuse a chain the runtime asked for.
    #[test]
    fn the_verify_budget_covers_every_anchor_and_draft() {
        for depth in [0, 1, 3] {
            let drafter = QwenDrafter {
                depth,
                wall: JOINT_WALL,
                joint: true,
            };
            assert_eq!(drafter.budget(0), 0);
            assert_eq!(drafter.budget(1), 1 + depth);
            assert_eq!(drafter.budget(4), 4 * (1 + depth));
        }
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

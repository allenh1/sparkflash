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

//! Sessions stepped together, so several sequences share one pass over the weights.
//!
//! A decode step reads about 10.9 GB of weights for one row, and two sessions on their own
//! streams read it twice. A batch puts one row of each member in a single forward: the stages that
//! depend only on the layer's weights (the projections, the hyper connection, the routed and
//! shared experts, the head) run once over every row, and the stages that touch a sequence's own
//! caches, position or recurrent state (the n-gram windows, the KV append, the indexer, the
//! attention, the Gated DeltaNet convolution and recurrence) run once per row on that row's slice.
//!
//! Each member keeps its own caches and state; only the per-row activation scratch moves. On entry
//! a member's scratch pointers are pointed at its row of the batch's combined buffers and saved, on
//! exit they are put back and its launches return to its own stream. The orderings are events on
//! the members' own streams, so no host synchronise is needed between steps.

use std::sync::Arc;

use sparkflash_kernels::cuda_driver::{Deviceptr, RunStream, StreamEvent};
use sparkflash_model::{BackendError, StepOutput};

use super::{DeviceSequence, GpuBranch, SCRATCH_SLOTS, SequenceShared};

/// The most sessions one batch steps together.
///
/// A row adds only its own routed experts, attention and recurrent state; the dense half of the
/// step is read once for all of them. Four is what this node's memory admits beside the 77.8 GiB
/// of weights at a 16k context, and it is where the measured curve still rises (1.75x the one-row
/// aggregate at four rows). The width is also capped by the engine's capacity, which is the
/// driver's concurrency, so a smaller deployment never allocates rows it cannot use.
pub(super) const MAX_BATCH: usize = 4;

/// The combined rows and the stream a batch steps its members on.
///
/// Allocated once for [MAX_BATCH] rows and reused by every step, so joining a batch costs no
/// allocation: a step points each member at its row and puts the pointers back afterwards.
pub(super) struct BatchBuffers {
    shared: Arc<SequenceShared>,
    /// One combined buffer a slot: [MAX_BATCH] rows of that slot's stride.
    combined: [Deviceptr; SCRATCH_SLOTS],
    /// Floats one row of each slot needs, in the same order as [DeviceSequence::scratch_pointers].
    strides: [usize; SCRATCH_SLOTS],
    /// A member's stream records this when the batch takes it over; the batch waits on each.
    enters: Vec<StreamEvent>,
    /// The batch's own completion, which every member's stream waits on when it leaves.
    done: StreamEvent,
    stream: RunStream,
    /// Rows the combined buffers hold: enough for every member to verify its drafts.
    rows: usize,
}

impl BatchBuffers {
    /// Allocate the combined rows for up to [MAX_BATCH] sessions.
    ///
    /// # Errors
    /// An allocation or stream the driver refuses.
    pub(super) fn new(
        shared: Arc<SequenceShared>,
        score_stride: usize,
        rows: usize,
    ) -> Result<Self, BackendError> {
        let cuda = &shared.cuda;
        cuda.make_current()
            .map_err(|error| BackendError::new("qwen4_batch", error))?;
        let strides = DeviceSequence::scratch_strides(&shared.config, score_stride);
        let mut total = 0usize;
        let mut combined = [Deviceptr(0); SCRATCH_SLOTS];
        for (slot, stride) in combined.iter_mut().zip(strides) {
            *slot = cuda
                .alloc((rows * stride * 4).max(4), &mut total)
                .map_err(|error| BackendError::new("qwen4_batch", error))?;
        }
        let stream = cuda
            .run_stream()
            .map_err(|error| BackendError::new("qwen4_batch", error))?;
        let mut enters = Vec::with_capacity(MAX_BATCH);
        for _ in 0..MAX_BATCH {
            enters.push(
                cuda.stream_event()
                    .map_err(|error| BackendError::new("qwen4_batch", error))?,
            );
        }
        let done = cuda
            .stream_event()
            .map_err(|error| BackendError::new("qwen4_batch", error))?;
        Ok(Self {
            shared,
            combined,
            strides,
            enters,
            done,
            stream,
            rows,
        })
    }

    /// Step `members` on their next tokens, in member order, and return every member's step.
    ///
    /// `drafts[m]` are member `m`'s speculative drafts for this step: its rows are its own token
    /// followed by them, and [DeviceSequence::resolve_verify] folds the logits back into the token
    /// the model chose, the drafts it agreed with and the state those rows left. A member with no
    /// drafts is one row and the same call.
    ///
    /// One row a member: member `r`'s launches read and write row `r` of the combined scratch, and
    /// the row-independent stages run once over all of them. A member whose stream had earlier work
    /// is ordered behind it, and every member is ordered behind the step before its own stream is
    /// handed back.
    ///
    /// # Errors
    /// A member or row count the batch cannot take, or the driver's failure code.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn step(
        &mut self,
        members: &mut [&mut DeviceSequence],
        drafts: &[Vec<u32>],
        confidence: &[Vec<f32>],
        tokens: &[u32],
    ) -> Result<Vec<StepOutput>, BackendError> {
        let count = members.len();
        if count == 0 || count > MAX_BATCH {
            return Err(BackendError::new(
                "qwen4_batch",
                format!("a batch steps 1..={MAX_BATCH} sessions, got {count}"),
            ));
        }
        if drafts.len() != count || confidence.len() != count || tokens.len() != count {
            return Err(BackendError::new(
                "qwen4_batch",
                format!(
                    "a batch of {count} sessions was given {} token(s), {} draft list(s) and {} confidence list(s)",
                    tokens.len(),
                    drafts.len(),
                    confidence.len()
                ),
            ));
        }
        let mut rows = Vec::with_capacity(count);
        let mut total = 0usize;
        for member in drafts {
            let width = member.len() + 1;
            rows.push(width);
            total += width;
        }
        if total > self.rows {
            return Err(BackendError::new(
                "qwen4_batch",
                format!(
                    "a batch of {count} sessions needs {total} rows, but this batch holds {}",
                    self.rows
                ),
            ));
        }
        let score_stride = members[0].score_stride;
        for member in members.iter() {
            if member.score_stride != score_stride {
                return Err(BackendError::new(
                    "qwen4_batch",
                    "a batch's sessions must share one context window",
                ));
            }
        }
        let cuda = &self.shared.cuda;
        cuda.make_current()
            .map_err(|error| BackendError::new("qwen4_batch", error))?;
        // Every member's earlier work first: the batch waits on each of their streams. Then each
        // member launches here for the length of the step, and its own scratch moves to its row.
        let mut saved = Vec::with_capacity(count);
        for (row, member) in members.iter().enumerate() {
            cuda.record_event_on(&member.stream, &self.enters[row])
                .map_err(|error| BackendError::new("qwen4_batch", error))?;
            cuda.stream_wait_event(&self.stream, &self.enters[row])
                .map_err(|error| BackendError::new("qwen4_batch", error))?;
        }
        // A member owns the rows its tokens occupy, so its scratch starts at the sum of the row
        // counts before it, not at its index: a verify's two rows are two rows of the combined
        // buffer whichever member they belong to.
        let mut offsets = Vec::with_capacity(count);
        let mut at = 0usize;
        for width in &rows {
            offsets.push(at);
            at += width;
        }
        for (member, start) in members.iter_mut().zip(&offsets) {
            let own = member.scratch_pointers();
            let mut pointers = own;
            for (index, slot) in pointers.iter_mut().enumerate() {
                *slot = self.combined[index].offset(start * self.strides[index] * 4);
            }
            saved.push(own);
            member.set_scratch_pointers(pointers);
            member.batch_stream = Some(self.stream.alias());
        }
        // A verify gives back the rows the model did not agree with, so each member's n-gram
        // windows are its own to restore: the state from before this step, and the rows it kept.
        let positions: Vec<usize> = members.iter().map(|member| member.position).collect();
        let hashers: Vec<_> = members
            .iter()
            .map(|member| std::sync::Arc::new(member.ple_hashers.clone()))
            .collect();
        let mut flat = Vec::with_capacity(total);
        for (member, token) in tokens.iter().enumerate() {
            flat.push(*token);
            flat.extend_from_slice(&drafts[member]);
        }
        let (lead, rest) = members.split_first_mut().expect("a batch is never empty");
        let outcome = lead.forward_rows(rest, &rows, &flat);
        // The exit runs whatever the step returned: a fault in the middle of the layer stack must
        // not leave a member launching on a stream that is about to be idle, or reading another
        // member's rows. The members are folded back into steps first: a draft the model agreed
        // with is kept by giving the rest of the chain's state back, and that needs the member's
        // own scratch still pointed at its rows.
        let folded = outcome.and_then(|logits| {
            let vocab = self.shared.config.vocab_size;
            let mut steps = Vec::with_capacity(count);
            let mut at = 0usize;
            for (member, width) in rows.iter().copied().enumerate() {
                let step = members[member].resolve_verify(
                    &flat[at..at + width],
                    &logits[at * vocab..(at + width) * vocab],
                    positions[member],
                    &drafts[member],
                    &confidence[member],
                    &hashers[member],
                )?;
                at += width;
                steps.push(step);
            }
            Ok(steps)
        });
        cuda.record_event_on(&self.stream, &self.done)
            .map_err(|error| BackendError::new("qwen4_batch", error))?;
        for (row, member) in members.iter_mut().enumerate() {
            cuda.stream_wait_event(&member.stream, &self.done)
                .map_err(|error| BackendError::new("qwen4_batch", error))?;
            member.set_scratch_pointers(saved[row]);
            member.batch_stream = None;
        }
        folded
    }
}

impl Drop for BatchBuffers {
    fn drop(&mut self) {
        let cuda = &self.shared.cuda;
        let _ = cuda.make_current();
        let _ = cuda.synchronize_stream(&self.stream);
        let pointers: Vec<&Deviceptr> = self.combined.iter().collect();
        cuda.free_buffers(&pointers);
        for event in &self.enters {
            cuda.destroy_event(event);
        }
        cuda.destroy_event(&self.done);
        cuda.destroy_run_stream(&self.stream);
    }
}

impl DeviceSequence {
    /// One forward for every member of a batch: `self` is the lead (member 0) and `rest` are the
    /// others, each with `rows[m]` rows of the concatenated `tokens`.
    ///
    /// A decode step gives every member one row; a speculative verify gives a member the token it is
    /// checking plus the drafts after it, so a two-member batch verifying two drafts each is a
    /// six-row pass. Rows keep member order: member `m` owns the slice that starts at the sum of the
    /// row counts before it, which is what lets a per-member stage run on that member's own caches,
    /// positions and recurrent state while the row-independent stages cover all of them.
    ///
    /// The stages that read a layer's weights - the hyper-connections, the query and key
    /// projections, the routed and shared experts, the output projections and the head - run once
    /// over every row; the stages that touch one sequence's own state - the PLE block's n-gram
    /// windows and convolution, the KV append, the indexer, the attention and the Gated DeltaNet
    /// convolution and recurrence - run once per member on its own slice.
    ///
    /// A member with more than one row is a speculative verify: its rows but the last have their
    /// convolution windows saved into its own snapshot, so the ones the model did not agree with
    /// can be given back by [DeviceSequence::restore_states].
    ///
    /// Returns every row's logits, concatenated in row order.
    ///
    /// # Errors
    /// The driver's failure code for a launch or a read-back.
    pub(super) fn forward_rows(
        &mut self,
        rest: &mut [&mut Self],
        rows: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<f32>, BackendError> {
        let count = rows.len();
        debug_assert_eq!(count, rest.len() + 1);
        debug_assert_eq!(rows.iter().sum::<usize>(), tokens.len());
        let total = tokens.len();
        // Where each member's rows start, in the combined scratch and in the token list.
        let mut offsets = Vec::with_capacity(count);
        let mut running = 0usize;
        for width in rows {
            offsets.push(running);
            running += width;
        }
        // The weights and kernels out of a handle of their own: a layer's PLE block takes the
        // session mutably, and the layer loop is reading the weights it sits in.
        let backend = Arc::clone(&self.backend);
        let config = Arc::clone(&backend.config);
        let kernels = &*backend.kernels;
        let weights = &backend.weights;
        let width = config.hyper_width();
        self.probe_rows = total;
        if super::prof::enabled() {
            super::prof::begin();
        }
        let embed_blocks = width.div_ceil(256).max(1) as u32;
        for (row, &token) in tokens.iter().enumerate() {
            self.launch(
                kernels.embed,
                (embed_blocks, 1, 1),
                (256, 1, 1),
                &[
                    weights.embed.pointer.0,
                    self.hyper.offset(row * width * 4).0,
                    u64::from(token),
                    config.hidden_size as u64,
                    width as u64,
                    0,
                ],
            )?;
        }
        self.probe_finite(0, "embed", self.hyper, total * width)?;
        self.prof_mark("embed")?;
        for (index, layer) in weights.layers.iter().enumerate() {
            for (member, width_rows) in rows.iter().copied().enumerate() {
                // A member's rows but the last save their windows: the state after its last row is
                // the state the next step would build anyway, so it is not worth a region.
                let verify = width_rows - 1;
                let at = offsets[member];
                let chunk = &tokens[at..at + width_rows];
                if member == 0 {
                    let (_, ple_slot) = self.snapshot_slots(index);
                    let ple_snapshot = match (verify > 0, ple_slot) {
                        (true, Some(offset)) => self.snapshot.offset(offset * 4).0,
                        _ => 0,
                    };
                    let stride = self.snapshot_stride as u64;
                    self.ple_step(
                        index,
                        layer,
                        chunk,
                        width_rows,
                        ple_snapshot,
                        verify,
                        stride,
                    )?;
                } else {
                    let session = &mut *rest[member - 1];
                    let (_, ple_slot) = session.snapshot_slots(index);
                    let ple_snapshot = match (verify > 0, ple_slot) {
                        (true, Some(offset)) => session.snapshot.offset(offset * 4).0,
                        _ => 0,
                    };
                    let stride = session.snapshot_stride as u64;
                    session.ple_step(
                        index,
                        layer,
                        chunk,
                        width_rows,
                        ple_snapshot,
                        verify,
                        stride,
                    )?;
                }
            }
            let (mixed, _) = self.hyper(&layer.attn_hyper, self.hyper, true, total)?;
            self.prof_mark("hyper_attn")?;
            match &layer.branch {
                GpuBranch::Linear { .. } => {
                    self.linear_project(&layer.branch, mixed, self.linear_b, self.linear_a, total)?;
                    for (member, width_rows) in rows.iter().copied().enumerate() {
                        let session: &Self = if member == 0 { self } else { rest[member - 1] };
                        let (conv_slot, _) = session.snapshot_slots(index);
                        let conv_snapshot = match (width_rows > 1, conv_slot) {
                            (true, Some(offset)) => session.snapshot.offset(offset * 4).0,
                            _ => 0,
                        };
                        // A verify keeps the rows a rejected draft would give back, and the replay
                        // that commits the kept ones reads them from the session's own inputs.
                        let (conv_out, b_out, a_out) =
                            match (width_rows > 1, session.gdn_replay[index]) {
                                (true, Some(inputs)) => session.replay_slices(inputs),
                                _ => (session.linear_conv, session.linear_b, session.linear_a),
                            };
                        if width_rows > 1 {
                            session.stash_gate_inputs(b_out, a_out, width_rows)?;
                        }
                        session.linear_state(
                            index,
                            &layer.branch,
                            conv_out,
                            b_out,
                            a_out,
                            width_rows,
                            false,
                            width_rows - 1,
                            conv_snapshot,
                            session.snapshot_stride as u64,
                        )?;
                    }
                    self.linear_output(index, &layer.branch, total)?;
                }
                GpuBranch::Sparse { .. } => {
                    self.sparse_project(&layer.branch, mixed, total)?;
                    for (member, width_rows) in rows.iter().copied().enumerate() {
                        let session: &Self = if member == 0 { self } else { rest[member - 1] };
                        session.sparse_attend(
                            &layer.branch,
                            index,
                            session.position,
                            width_rows,
                        )?;
                    }
                    self.sparse_output(&layer.branch, index, total)?;
                }
            }
            self.launch(
                kernels.inject,
                (width.div_ceil(256).max(1) as u32, total as u32, 1),
                (256, 1, 1),
                &[
                    self.hyper.0,
                    self.branch.0,
                    self.injection.0,
                    config.hc_count as u64,
                    config.hidden_size as u64,
                ],
            )?;
            self.probe_finite(index, "after_attn", self.hyper, total * width)?;
            self.prof_mark("attn_inject")?;
            let (mixed, injection) = self.hyper(&layer.mlp_hyper, self.hyper, true, total)?;
            let injection = injection.expect("the MLP hyper-connection injects");
            self.probe_finite(index, "mixed", mixed, total * config.hidden_size)?;
            self.prof_mark("hyper_mlp")?;
            // The routed experts do not share between different tokens, but the router, the shared
            // expert and both hyper-connections read their weights once for every row.
            self.mlp(
                &layer.moe, mixed, self.hyper, injection, index, false, total,
            )?;
            self.probe_finite(index, "after_mlp", self.hyper, total * width)?;
            // The drafter reads a few layers' widened rows; a member's own slice is the rows its
            // next block anchors on.
            for (member, width_rows) in rows.iter().copied().enumerate() {
                let session: &Self = if member == 0 { self } else { rest[member - 1] };
                session.capture_features(index, width_rows)?;
            }
            self.prof_mark("inject_mlp")?;
        }
        let (mixed, _) = self.hyper(&weights.mixer, self.hyper, false, total)?;
        self.gemv(&weights.lm_head, mixed, self.logits, total)?;
        self.probe_finite(48, "logits", self.logits, config.vocab_size * total)?;
        self.prof_mark("lm_head")?;
        let logits = self.read_logits(total)?;
        self.prof_mark("logits_dtoh")?;
        super::prof::report(self.position, total);
        for (member, width_rows) in rows.iter().copied().enumerate() {
            if member == 0 {
                self.position += width_rows;
            } else {
                rest[member - 1].position += width_rows;
            }
        }
        Ok(logits)
    }

    /// Copy this member's gate and decay rows out of the combined projections into the verify
    /// inputs the recurrent replay reads, for a member whose rows are a speculative verify.
    ///
    /// The projection itself ran once over every member's rows; only where those two vectors go
    /// differs between a plain step and a verify. Two small copies a linear layer buy that without
    /// a second pass over the projection's weights.
    ///
    /// # Errors
    /// The driver's failure code for the copies.
    fn stash_gate_inputs(
        &self,
        b_out: Deviceptr,
        a_out: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        let kernels = &*self.backend.kernels;
        let elems = rows * self.backend.config.linear_num_value_heads;
        let blocks = elems.div_ceil(256).max(1) as u32;
        self.launch(
            kernels.copy_flat,
            (blocks, 1, 1),
            (256, 1, 1),
            &[b_out.0, self.linear_b.0, elems as u64],
        )?;
        self.launch(
            kernels.copy_flat,
            (blocks, 1, 1),
            (256, 1, 1),
            &[a_out.0, self.linear_a.0, elems as u64],
        )?;
        Ok(())
    }
}

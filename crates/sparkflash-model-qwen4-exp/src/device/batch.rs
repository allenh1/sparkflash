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
use sparkflash_model::BackendError;

use super::{
    DeviceSequence, INJECTION_SLOT, PARTIALS_SLOT, SCRATCH_SLOTS, SequenceShared,
    fill_plain_injection, zero_split_tickets,
};

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

// SAFETY: the buffers hold device pointers and stream events of one CUDA context. They are made by
// the arm and used only from the serving engine's thread, which binds that context (`make_current`)
// before every launch; nothing else dereferences them, and the handles are not shared.
unsafe impl Send for BatchBuffers {}

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
        // A member's `gemm_partials` is its first row's region, tickets and all.
        zero_split_tickets(
            cuda,
            &stream,
            shared.kernels.zero.kernel,
            combined[PARTIALS_SLOT],
            rows,
        )
        .map_err(|error| BackendError::new("qwen4_batch", error))?;
        fill_plain_injection(cuda, &shared.config, combined[INJECTION_SLOT], rows)
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

    /// Take `members` over for one step of `rows[m]` rows each, in member order, and answer
    /// each member's own scratch pointers for [Self::exit].
    ///
    /// Member `m` owns the rows that start at the sum of the row counts before it: its launches
    /// read and write those rows of the combined scratch, and the row-independent stages run
    /// once over all of them. A member whose stream had earlier work is ordered behind it, and
    /// every launch until the exit is on the batch's stream.
    ///
    /// # Errors
    /// A member or row count the batch cannot take, or the driver's failure code.
    pub(super) fn enter(
        &self,
        members: &mut [&mut DeviceSequence],
        rows: &[usize],
    ) -> Result<Vec<[Deviceptr; SCRATCH_SLOTS]>, BackendError> {
        let count = members.len();
        if count == 0 || count > MAX_BATCH {
            return Err(BackendError::new(
                "qwen4_batch",
                format!("a batch steps 1..={MAX_BATCH} sessions, got {count}"),
            ));
        }
        let total: usize = rows.iter().sum();
        if rows.len() != count || total > self.rows {
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
        let mut at = 0usize;
        for (member, width) in members.iter_mut().zip(rows) {
            let own = member.scratch_pointers();
            let mut pointers = own;
            for (index, slot) in pointers.iter_mut().enumerate() {
                *slot = self.combined[index].offset(at * self.strides[index] * 4);
            }
            saved.push(own);
            member.set_scratch_pointers(pointers);
            member.batch_stream = Some(self.stream.alias());
            at += width;
        }
        Ok(saved)
    }

    /// Hand `members` back after a step: every member's stream is ordered behind the step, and
    /// its scratch pointers are the `saved` ones [Self::enter] answered. A fault in the middle
    /// of the layer stack exits too, so no member launches on a stream about to be idle or
    /// reads another member's rows.
    ///
    /// # Errors
    /// The driver's failure code.
    pub(super) fn exit(
        &self,
        members: &mut [&mut DeviceSequence],
        saved: &[[Deviceptr; SCRATCH_SLOTS]],
    ) -> Result<(), BackendError> {
        let cuda = &self.shared.cuda;
        cuda.record_event_on(&self.stream, &self.done)
            .map_err(|error| BackendError::new("qwen4_batch", error))?;
        for (row, member) in members.iter_mut().enumerate() {
            cuda.stream_wait_event(&member.stream, &self.done)
                .map_err(|error| BackendError::new("qwen4_batch", error))?;
            member.set_scratch_pointers(saved[row]);
            member.batch_stream = None;
        }
        Ok(())
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
    /// Copy this member's gate and decay rows out of the combined projections into the verify
    /// inputs the recurrent replay reads, for a member whose rows are a speculative verify.
    ///
    /// The projection itself ran once over every member's rows; only where those two vectors go
    /// differs between a plain step and a verify. Two small copies a linear layer buy that without
    /// a second pass over the projection's weights.
    ///
    /// # Errors
    /// The driver's failure code for the copies.
    pub(super) fn stash_gate_inputs(
        &self,
        b_out: Deviceptr,
        a_out: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        let kernels = &*self.backend.kernels;
        let elems = rows * self.backend.config.linear_num_value_heads;
        self.call(
            kernels.copy_flat,
            sparkflash_ops::calls::copy(b_out.0, self.linear_b.0, elems),
        )?;
        self.call(
            kernels.copy_flat,
            sparkflash_ops::calls::copy(a_out.0, self.linear_a.0, elems),
        )?;
        Ok(())
    }
}

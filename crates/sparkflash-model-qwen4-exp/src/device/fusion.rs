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

//! The Qwen4-Exp arm behind the shared fused engine.
//!
//! The engine ([sparkflash_model::FusedEngine]) owns the slots and the thread; this file is what
//! it calls. A slot is a whole [DeviceSequence] - its caches, its sampler and its history - and a
//! step is one batch over the members' rows through the pack's own [BatchBuffers].

use std::sync::{Arc, Mutex};

use sparkflash_model::{
    BackendError, FusedArm, ModelSession as _, Sampling, Speculation, StepOutput,
};

use super::batch::BatchBuffers;
use super::{DeviceSequence, SequenceShared};

/// The arm the engine steps.
pub(super) struct QwenFusion {
    shared: Arc<SequenceShared>,
    context_limit: usize,
    /// How many members one pass may hold: the combined buffers hold one row a member.
    rows: usize,
    /// The combined row buffers, made on the first step and reused: joining a batch costs no
    /// allocation.
    batch: Mutex<Option<BatchBuffers>>,
}

impl QwenFusion {
    /// The arm over `shared`, serving sequences of up to `context_limit` tokens, with room for
    /// `capacity` members in one pass.
    pub(super) fn new(shared: Arc<SequenceShared>, context_limit: usize, capacity: usize) -> Self {
        Self {
            shared,
            context_limit,
            rows: capacity.clamp(1, super::batch::MAX_BATCH),
            batch: Mutex::new(None),
        }
    }
}

impl FusedArm for QwenFusion {
    type Slot = DeviceSequence;

    fn open_slot(&self) -> Result<DeviceSequence, BackendError> {
        let mut session = DeviceSequence::allocate(Arc::clone(&self.shared), self.context_limit)?;
        // A fused member does not verify its drafts: the rows it adds cost more than the drafts
        // it keeps, which is the measurement the earlier fused engine made and this one keeps.
        session.spec_enabled = false;
        Ok(session)
    }

    fn holds_request(&self, slot: &DeviceSequence) -> bool {
        slot.position != 0 || !slot.consumed.is_empty()
    }

    fn renew(&self, slot: &mut DeviceSequence) -> Result<(), BackendError> {
        let sampling = slot.sampling;
        let mut fresh = self.open_slot()?;
        // The request's sampling reached the old sequence before its prompt replaced it.
        fresh.set_sampling(sampling);
        *slot = fresh;
        Ok(())
    }

    fn set_sampling(&self, slot: &mut DeviceSequence, sampling: Sampling) {
        slot.set_sampling(sampling);
    }

    fn prefill(
        &self,
        slot: &mut DeviceSequence,
        tokens: &[u32],
    ) -> Result<StepOutput, BackendError> {
        slot.prefill(tokens)
    }

    fn step(&self, members: &mut [(DeviceSequence, u32)]) -> Result<Vec<StepOutput>, BackendError> {
        let mut batch = self
            .batch
            .lock()
            .map_err(|_| BackendError::new("qwen4_batch", "the batch buffers are poisoned"))?;
        if batch.is_none() {
            let score_stride = members
                .first()
                .map_or(0, |(session, _)| session.score_stride);
            *batch = Some(BatchBuffers::new(
                Arc::clone(&self.shared),
                score_stride,
                self.rows.max(members.len()),
            )?);
        }
        let allocation = batch.as_mut().expect("the buffers were just made");
        let tokens: Vec<u32> = members.iter().map(|(_, token)| *token).collect();
        let drafts = vec![Vec::new(); members.len()];
        let confidence = vec![Vec::new(); members.len()];
        let mut refs: Vec<&mut DeviceSequence> =
            members.iter_mut().map(|(session, _)| session).collect();
        // One row a member, no drafts: the members' rows are the only rows in the pass.
        let steps = allocation.step(&mut refs, &drafts, &confidence, &tokens)?;
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

    fn speculation(&self, slot: &DeviceSequence) -> Speculation {
        slot.speculation()
    }

    fn max_batch(&self) -> usize {
        self.rows
    }
}

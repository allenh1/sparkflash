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
//! The engine ([sparkflash_runtime::FusedEngine]) owns the slots and the thread; this file is what
//! it calls. A slot is a whole [DeviceSequence] - its caches, its sampler and its history - and a
//! step is one batch over the members' rows through the pack's own [BatchBuffers].
//!
//! With a drafter loaded the members draft as they would alone and verify together
//! ([sparkflash_runtime::run_round]): each opens its round, the walk's rows are shared among the
//! chains, and one batch verifies every member's token and the drafts it was given. A lone member
//! decodes on its own, so its tokens and its depth are the unfused session's.

use std::sync::{Arc, Mutex};

use sparkflash_model::{BackendError, ModelSession as _, Sampling, Speculation, StepOutput};

use super::batch::BatchBuffers;
use super::{DeviceSequence, SequenceShared};
use sparkflash_runtime::{ExpertUnionWall, FusedArm, Opened, RoundArm, StepWall, run_round};

/// The arm the engine steps.
pub(super) struct QwenFusion {
    shared: Arc<SequenceShared>,
    context_limit: usize,
    /// How many members one pass may hold.
    rows: usize,
    /// Whether the members draft and verify their rounds together
    /// ([sparkflash_runtime::JOINT_ROUNDS_ENV]); off, a fused member drafts nothing and a step is
    /// one row a member.
    joint_rounds: bool,
    /// What a shared walk costs, for the row share.
    wall: ExpertUnionWall,
    /// The combined row buffers, made on the first step and reused: joining a batch costs no
    /// allocation. With joint rounds they hold every member's widest verify.
    batch: Mutex<Option<BatchBuffers>>,
}

/// A shared walk's cost at TP=1 on a GB10-class node, from the block drafter's own step cost
/// ([super::DFLASH2_STEP_COST]: 59 ms a step and 8.4 ms a verified draft): about 10 new experts a
/// row at 0.8 ms each, the rest of a step fixed, and a few milliseconds a member past the first
/// for its own attention, recurrence and draft. The expert counts are the checkpoint's.
const JOINT_WALL: ExpertUnionWall = ExpertUnionWall {
    fixed_ms: 50.4,
    expert_ms: 0.81,
    row_ms: 0.5,
    seq_ms: 6.0,
    experts: 0,
    top_k: 0,
};

impl QwenFusion {
    /// The arm over `shared`, serving sequences of up to `context_limit` tokens, with room for
    /// `capacity` members in one pass.
    ///
    /// # Errors
    /// A [sparkflash_runtime::JOINT_ROUNDS_ENV] that is neither on nor off.
    pub(super) fn new(
        shared: Arc<SequenceShared>,
        context_limit: usize,
        capacity: usize,
    ) -> Result<Self, BackendError> {
        let joint_rounds = sparkflash_runtime::joint_rounds()
            .map_err(|error| BackendError::new("config", error))?;
        let config = &shared.config;
        let wall = ExpertUnionWall {
            experts: config.num_experts,
            top_k: config.num_experts_per_tok,
            ..JOINT_WALL
        };
        Ok(Self {
            shared,
            context_limit,
            rows: capacity.clamp(1, super::batch::MAX_BATCH),
            joint_rounds,
            wall,
            batch: Mutex::new(None),
        })
    }

    /// The most drafts one member verifies in a round: the verify's rows past its token.
    fn depth(&self) -> usize {
        self.shared.verify_rows.saturating_sub(1)
    }

    /// One batch over `members`: each its token and `drafts[m]`.
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
            // A joint round's rows are every member's token and its drafts.
            let width = if self.joint_rounds {
                self.shared.verify_rows.max(1)
            } else {
                1
            };
            *batch = Some(BatchBuffers::new(
                Arc::clone(&self.shared),
                score_stride,
                self.rows.max(members.len()) * width,
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
}

impl RoundArm for QwenFusion {
    type Member = DeviceSequence;

    fn open(&self, member: &mut DeviceSequence, token: u32) -> Result<Opened, BackendError> {
        member.open_round(token)
    }

    /// One batch of every member's token and the drafts it was given: the batch's verify folds
    /// each member's rows back into its step and its next chain ([DeviceSequence::resolve_verify]).
    fn verify(
        &self,
        round: &mut [(&mut DeviceSequence, u32, usize)],
    ) -> Result<Vec<StepOutput>, BackendError> {
        let mut drafts = Vec::with_capacity(round.len());
        let mut confidence = Vec::with_capacity(round.len());
        let mut tokens = Vec::with_capacity(round.len());
        for (member, token, depth) in round.iter_mut() {
            let (chain, sure) = member.take_drafts(*depth);
            drafts.push(chain);
            confidence.push(sure);
            tokens.push(*token);
        }
        let mut members: Vec<&mut DeviceSequence> = round
            .iter_mut()
            .map(|(member, _, _)| &mut **member)
            .collect();
        self.batch(&mut members, &drafts, &confidence, &tokens)
    }

    fn wall(&self) -> &dyn StepWall {
        &self.wall
    }

    fn budget(&self, seqs: usize) -> usize {
        seqs * self.depth()
    }
}

impl FusedArm for QwenFusion {
    type Slot = DeviceSequence;

    fn open_slot(&self) -> Result<DeviceSequence, BackendError> {
        let mut session = DeviceSequence::allocate(Arc::clone(&self.shared), self.context_limit)?;
        // Without joint rounds a fused member drafts nothing: verifying every member's whole
        // chain cost more rows than its drafts kept (the earlier fused engine's measurement).
        // Joint rounds share the rows among the chains instead, so the member keeps its drafter.
        session.spec_enabled &= self.joint_rounds;
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

    /// With joint rounds, several members verify their rounds in one batch ([run_round]) and a
    /// lone member decodes as the unfused session does. Without, a step is one row a member.
    fn step(&self, members: &mut [(DeviceSequence, u32)]) -> Result<Vec<StepOutput>, BackendError> {
        if self.joint_rounds {
            if members.len() == 1 {
                return members
                    .iter_mut()
                    .map(|(session, token)| session.decode(*token))
                    .collect();
            }
            return run_round(self, members);
        }
        let tokens: Vec<u32> = members.iter().map(|(_, token)| *token).collect();
        for (session, token) in members.iter_mut() {
            session.feed_plain(*token);
        }
        let drafts = vec![Vec::new(); members.len()];
        let confidence = vec![Vec::new(); members.len()];
        let mut refs: Vec<&mut DeviceSequence> =
            members.iter_mut().map(|(session, _)| session).collect();
        // One row a member, no drafts: the members' rows are the only rows in the pass.
        self.batch(&mut refs, &drafts, &confidence, &tokens)
    }

    fn ready(&self, slot: &DeviceSequence, token: u32) -> bool {
        slot.holds(token)
    }

    fn give_back(&self, slot: &mut DeviceSequence, count: usize) -> Result<(), BackendError> {
        slot.give_back(count)
    }

    fn speculation(&self, slot: &DeviceSequence) -> Speculation {
        slot.speculation()
    }

    fn max_batch(&self) -> usize {
        self.rows
    }
}

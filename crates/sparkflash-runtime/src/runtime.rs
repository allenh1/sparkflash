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

//! The runtime's only `FusedArm`: every strategy over one [`Walk`].
//!
//! [`Runtime`] owns what no arm should write again: history, resume and the prefix match; the
//! prompt-part state machine, sized by [`ChunkPolicy`]; checkpoint and rewind, driven by
//! [`StateLayout`]; lane reservations from [`LanePool`]; give-back; the ledger; and, by
//! implementing [`FusedArm`], group mirroring and the fused engine's lanes. A model supplies
//! only the walk and the head.
//!
//! The engine itself is not replaced here: `FusedEngine` still schedules the lanes, and this
//! type is the one `FusedArm` a walk needs. The dense arm serves through it; the other arms
//! move onto it in the same phase, first dense as the proof.

use std::sync::Mutex;

use sparkflash_model::backend::{BackendError, Sampling, Speculation, StepOutput};
use sparkflash_model::vision::{ImageSpan, VisionImage};

use crate::chunk::ChunkPolicy;
use crate::fused::{FusedArm, Part, Resume, resume};
use crate::layout::{LaneState, Limits, StateKind, StateLayout};
use crate::ledger::{Ledger, Mode, Strategy};
use crate::pool::{LaneMemory, LanePool};
use crate::resume::{Restore, checkpoint_after, rewind_plan};
use crate::share::share_rows;
use crate::walk::{Draft, Drafter, HeadRow, Pass, PassMask, Segment, Vision, Walk};

/// A prompt the runtime is feeding a chunk at a time.
#[derive(Debug)]
struct PromptState {
    tokens: Vec<u32>,
    next: usize,
}

/// One request's state: the arm's sequence, the runtime's history, and the prompt in flight.
pub struct RuntimeSlot<W: Walk> {
    seq: W::Seq,
    lane: LaneMemory,
    /// The tokens the sequence holds, as the runtime tracks them; the arm's own history is its
    /// own business.
    history: Vec<u32>,
    /// The end of the last prompt, when rewind is on.
    point: Option<usize>,
    /// How many of the last prompt's tokens the sequence already held.
    cached: usize,
    prompt: Option<PromptState>,
    sampling: Sampling,
}

impl<W: Walk> RuntimeSlot<W> {
    /// The arm's sequence.
    #[must_use]
    pub fn seq(&self) -> &W::Seq {
        &self.seq
    }

    /// The arm's sequence, mutably.
    #[must_use]
    pub fn seq_mut(&mut self) -> &mut W::Seq {
        &mut self.seq
    }

    /// The lane reservation the sequence holds.
    #[must_use]
    pub fn lane(&self) -> LaneMemory {
        self.lane
    }

    /// The runtime's history for this sequence.
    #[must_use]
    pub fn history(&self) -> &[u32] {
        &self.history
    }
}

/// Every strategy over one [`Walk`].
pub struct Runtime<W: Walk> {
    walk: W,
    layout: StateLayout,
    limits: Limits,
    chunk: ChunkPolicy,
    lanes: Mutex<LanePool>,
    rewind: bool,
    /// Whether drafting members verify their rounds in one walk
    /// ([`crate::JOINT_ROUNDS_ENV`]); off, each lane's round walks alone.
    joint: bool,
}

impl<W: Walk> Runtime<W> {
    /// A runtime over `walk`, with `free_bytes` free for chunk sizing (zero: not measured).
    ///
    /// A walk whose drafter is present is driven through its round path; the refusal the first
    /// drafting phase carried is gone, because [`Drafter`] now requires every method the round
    /// needs.
    ///
    /// # Errors
    /// A [`crate::REWIND_ENV`] or [`crate::JOINT_ROUNDS_ENV`] that is not a switch, or a
    /// drafting walk that does not declare the verify pass its round runs.
    pub fn new(walk: W, free_bytes: u64) -> Result<Self, BackendError> {
        let joint =
            crate::rounds::joint_rounds().map_err(|error| BackendError::new("runtime", error))?;
        if W::Drafter::PRESENT && !W::CAPABILITIES.passes.contains(PassMask::VERIFY) {
            return Err(BackendError::new(
                "runtime",
                format!(
                    "{} declares a drafter ({}) and does not declare the verify pass its round                      runs; a drafting walk's rows must be true before it serves",
                    std::any::type_name::<W>(),
                    std::any::type_name::<W::Drafter>()
                ),
            ));
        }
        let layout = walk.layout().clone();
        let limits = walk.limits();
        let chunk = ChunkPolicy::new(limits, free_bytes, layout.bytes_per_row());
        let lane_bytes = layout.bytes_at(limits.context);
        let rewind = crate::fused::rewind().map_err(|error| BackendError::new("runtime", error))?;
        Ok(Self {
            walk,
            layout,
            limits,
            chunk,
            lanes: Mutex::new(LanePool::new(lane_bytes)),
            rewind,
            joint,
        })
    }

    /// Whether drafting members verify their rounds in one walk.
    #[must_use]
    pub const fn joint_rounds(&self) -> bool {
        self.joint
    }

    /// The same runtime with joint rounds set to `joint`.
    ///
    /// This is the switch [`crate::JOINT_ROUNDS_ENV`] writes, exposed by value so a check can
    /// run both policies without changing the process environment.
    #[must_use]
    pub fn with_joint_rounds(mut self, joint: bool) -> Self {
        self.joint = joint;
        self
    }

    /// The arm.
    #[must_use]
    pub fn walk(&self) -> &W {
        &self.walk
    }

    /// The layout the arm declared.
    #[must_use]
    pub fn layout(&self) -> &StateLayout {
        &self.layout
    }

    /// The runtime's chunk policy.
    #[must_use]
    pub fn chunk_policy(&self) -> ChunkPolicy {
        self.chunk
    }

    /// The limits the arm declared.
    #[must_use]
    pub fn limits(&self) -> Limits {
        self.limits
    }

    fn lanes(&self) -> Result<std::sync::MutexGuard<'_, LanePool>, BackendError> {
        self.lanes
            .lock()
            .map_err(|_| BackendError::new("runtime", "the lane pool is poisoned"))
    }

    fn release_lane(&self, lane: LaneMemory) {
        if let Ok(mut lanes) = self.lanes.lock() {
            lanes.release(lane);
        }
    }

    /// Runs the next chunk of `slot`'s prompt, then its head when the prompt ends.
    fn run_part(&self, slot: &mut RuntimeSlot<W>) -> Result<Part, BackendError> {
        let prompt = slot
            .prompt
            .take()
            .ok_or_else(|| BackendError::new("runtime", "the sequence has no prompt in parts"))?;
        let end = (prompt.next + self.chunk.rows()).min(prompt.tokens.len());
        let rows = &prompt.tokens[prompt.next..end];
        let hidden = {
            let mut segments = [Segment::new(&mut slot.seq, rows)];
            let pass = Pass::bulk(rows.len()).with_prompt_end(prompt.tokens.len() - end);
            self.walk.walk(&mut segments, pass)?
        };
        slot.history.extend_from_slice(rows);
        if end < prompt.tokens.len() {
            slot.prompt = Some(PromptState {
                tokens: prompt.tokens,
                next: end,
            });
            return Ok(Part::More);
        }
        let row = hidden.rows.checked_sub(1).ok_or_else(|| {
            BackendError::new("runtime", "a prompt walk left no residual row to head")
        })?;
        let mut head_rows = [HeadRow::prompt(&mut slot.seq, row, &prompt.tokens)];
        let mut steps = self.walk.head(&hidden, &mut head_rows)?;
        let step = steps
            .pop()
            .ok_or_else(|| BackendError::new("runtime", "the head answered no step"))?;
        if self.rewind {
            slot.point = Some(prompt.tokens.len());
            if checkpoint_after(&self.layout, prompt.tokens.len()).is_some() {
                slot.seq.checkpoint(prompt.tokens.len())?;
            }
        } else {
            slot.point = None;
        }
        Ok(Part::Done(step))
    }

    /// One real verify over the anchor and drafts of several members.
    ///
    /// One [`Pass::Verify`] walks every member's anchor and drafts, under the caller's output
    /// constraint. The members' rows are then headed in order: at each row the head answers the
    /// token the model chose, the runtime compares it with the draft after it, and the member's
    /// run stops at the first mismatch. Only the rows up to that stop (and within
    /// `constraint.max_rows`) reach the history, so a rejected draft is not appended. The
    /// answer is the anchor's step, one a member.
    ///
    /// A head past the first mismatch is never asked for: the check that a rejected draft costs
    /// no heading of the rows it invalidates is that the head's call count equals the kept rows.
    ///
    /// # Errors
    /// A backend failure, or a pass the walk does not accept.
    pub fn verify(
        &self,
        members: &mut [(RuntimeSlot<W>, u32, &[u32])],
        constraint: crate::OutputConstraint<'_>,
    ) -> Result<Vec<StepOutput>, BackendError> {
        let tokens: Vec<Vec<u32>> = members
            .iter()
            .map(|(_, anchor, drafts)| {
                let mut rows = Vec::with_capacity(1 + drafts.len());
                rows.push(*anchor);
                rows.extend_from_slice(drafts);
                rows
            })
            .collect();
        let hidden = {
            let mut segments: Vec<Segment<'_, W::Seq>> = members
                .iter_mut()
                .zip(tokens.iter())
                .map(|((slot, _, _), rows)| Segment::new(&mut slot.seq, rows))
                .collect();
            let rows = tokens.iter().map(Vec::len).sum();
            self.walk
                .walk(&mut segments, Pass::verify(rows, constraint))?
        };
        let mut offsets = Vec::with_capacity(tokens.len());
        let mut offset = 0;
        for rows in &tokens {
            offsets.push(offset);
            offset += rows.len();
        }
        let mut steps = Vec::with_capacity(members.len());
        for (index, ((slot, _, _), rows)) in members.iter_mut().zip(tokens.iter()).enumerate() {
            let base = offsets[index];
            let cap = constraint.max_rows.max(1).min(rows.len());
            let mut kept = 0;
            let mut anchor = None;
            for row in 0..rows.len() {
                if row >= cap {
                    break;
                }
                let mut head_rows = [HeadRow::verify(&mut slot.seq, base + row, rows, row)];
                let mut answered = self.walk.head(&hidden, &mut head_rows)?;
                let step = answered
                    .pop()
                    .ok_or_else(|| BackendError::new("runtime", "a verify row answered nothing"))?;
                if row == 0 {
                    anchor = Some(step.clone());
                }
                kept = row + 1;
                let matched = row + 1 == rows.len() || step.next_token == Some(rows[row + 1]);
                if !matched {
                    break;
                }
            }
            if kept == 0 {
                return Err(BackendError::new(
                    "runtime",
                    "a verify segment holds no rows",
                ));
            }
            slot.history.extend_from_slice(&rows[..kept]);
            steps.push(
                anchor.ok_or_else(|| {
                    BackendError::new("runtime", "a verify answered no anchor row")
                })?,
            );
        }
        Ok(steps)
    }

    /// One step with the arm's drafter: held answers are handed back with no device work; the
    /// opened chains are verified in one pass (a joint round) or one lane at a time (a lone
    /// member, or [`crate::JOINT_ROUNDS_ENV`] off), and settled.
    fn draft_step(
        &self,
        members: &mut [(RuntimeSlot<W>, u32)],
    ) -> Result<Vec<StepOutput>, BackendError> {
        struct Chain {
            confidence: Vec<f64>,
            drafts: Vec<u32>,
        }

        let drafter = self.walk.drafter();
        let mut answers: Vec<Option<StepOutput>> = (0..members.len()).map(|_| None).collect();
        let mut chains: Vec<Option<Chain>> = (0..members.len()).map(|_| None).collect();
        for (index, (slot, token)) in members.iter_mut().enumerate() {
            match drafter.open_round(&mut slot.seq, *token)? {
                Draft::Held(step) => answers[index] = Some(step),
                Draft::Chain { confidence, drafts } => {
                    chains[index] = Some(Chain { confidence, drafts });
                }
            }
        }
        let opened: Vec<usize> = chains
            .iter()
            .enumerate()
            .filter_map(|(index, chain)| chain.as_ref().map(|_| index))
            .collect();
        // A lone member keeps the drafter's own depth: there is no other chain to share with,
        // and trimming it through the row share would hand back a shorter run than the request
        // drafted for.
        let joint = self.joint && opened.len() > 1;
        let groups: Vec<Vec<usize>> = if joint {
            vec![opened]
        } else {
            opened.iter().map(|index| vec![*index]).collect()
        };
        for group in groups {
            let depths: Vec<usize> = if joint {
                let probabilities: Vec<Vec<f64>> = group
                    .iter()
                    .map(|&index| {
                        chains[index]
                            .as_ref()
                            .map(|chain| chain.confidence.clone())
                            .unwrap_or_default()
                    })
                    .collect();
                share_rows(
                    &probabilities,
                    drafter.budget(group.len()),
                    drafter.floor(),
                    drafter.wall(),
                )
            } else {
                group
                    .iter()
                    .map(|&index| chains[index].as_ref().map_or(0, |chain| chain.drafts.len()))
                    .collect()
            };
            let rows: Vec<Vec<u32>> = group
                .iter()
                .zip(&depths)
                .map(|(&index, &depth)| {
                    let chain = chains[index].as_ref().expect("the member opened a chain");
                    let mut rows = Vec::with_capacity(depth + 1);
                    rows.push(members[index].1);
                    rows.extend_from_slice(&chain.drafts[..depth]);
                    rows
                })
                .collect();
            let results = {
                let mut segments: Vec<Segment<'_, W::Seq>> = Vec::with_capacity(group.len());
                let mut next = 0;
                for (index, (slot, _)) in members.iter_mut().enumerate() {
                    if group.contains(&index) {
                        segments.push(Segment::new(&mut slot.seq, &rows[next]));
                        next += 1;
                    }
                }
                drafter.verify(&self.walk, &mut segments, &depths)?
            };
            if results.len() != group.len() {
                return Err(BackendError::new(
                    "runtime",
                    format!(
                        "a verify of {} member(s) answered {} row(s)",
                        group.len(),
                        results.len()
                    ),
                ));
            }
            for ((&index, rows), result) in group.iter().zip(rows.iter()).zip(results) {
                let step = drafter.settle(&mut members[index].0.seq, rows, result)?;
                answers[index] = Some(step);
            }
        }
        let mut steps = Vec::with_capacity(members.len());
        for (index, (slot, token)) in members.iter_mut().enumerate() {
            slot.history.push(*token);
            steps.push(answers[index].take().ok_or_else(|| {
                BackendError::new("runtime", "a round left a member without a step")
            })?);
        }
        Ok(steps)
    }

    /// A fresh slot, with no `Send` bound: [`FusedArm::open_slot`] delegates here, and so
    /// does a check's CPU walk.
    ///
    /// # Errors
    /// An exhausted lane pool, or a sequence the arm cannot open.
    pub fn open_slot(&self) -> Result<RuntimeSlot<W>, BackendError> {
        let lane = self.lanes()?.acquire();
        let seq = self.walk.open(lane)?;
        Ok(RuntimeSlot {
            seq,
            lane,
            history: Vec::new(),
            point: None,
            cached: 0,
            prompt: None,
            sampling: Sampling::default(),
        })
    }

    /// A prompt through the runtime, with no `Send` bound: the engine's [`FusedArm::prefill`]
    /// delegates here, and so does a checks' CPU walk.
    ///
    /// # Errors
    /// A backend failure, an empty prompt, or an image the arm does not take.
    pub fn prefill_slot(
        &self,
        slot: &mut RuntimeSlot<W>,
        tokens: &[u32],
    ) -> Result<StepOutput, BackendError> {
        let mut part = self.begin_prompt_slot(slot, tokens, &[])?;
        loop {
            match part {
                Part::Done(step) => return Ok(step),
                Part::More => part = self.next_part_slot(slot)?,
            }
        }
    }

    /// The first part of a prompt ([FusedArm::begin_prompt], without the join bounds).
    ///
    /// # Errors
    /// An empty prompt, an image the arm does not take, or a backend failure.
    pub fn begin_prompt_slot(
        &self,
        slot: &mut RuntimeSlot<W>,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<Part, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::new("runtime", "the prompt has no tokens"));
        }
        if !images.is_empty() {
            return Err(BackendError::new(
                "vision",
                if W::Vision::PRESENT {
                    "the runtime's vision path is not wired yet"
                } else {
                    "this arm has no vision path"
                },
            ));
        }
        // The history may lag the arm's position: a walk that verified drafts keeps their rows
        // in the sequence while the caller has only been given the anchor's token. A prompt may
        // extend the history only when the state ends exactly where the history does; otherwise
        // the rewind point is the resume, as an arm that owns the rule asks for.
        let current = slot.prompt.is_none() && slot.seq.position() == slot.history.len();
        let answer = resume(&slot.history, current, slot.point, tokens);
        let start = match answer {
            Some(Resume::Extend(held)) => held,
            Some(Resume::Rewind(point)) => {
                if !matches!(rewind_plan(&self.layout, point).restore, Restore::Nothing) {
                    slot.seq.rewind(point)?;
                }
                point
            }
            None => 0,
        };
        // A prompt from the start clears the arm's state even when the history was already
        // empty, as the arm's own prefill did: the lane may hold a rewound prefix.
        if start == 0 && !matches!(answer, Some(Resume::Rewind(_))) {
            slot.seq.rewind(0)?;
        }
        slot.history.truncate(start);
        slot.cached = start;
        if start == 0 {
            slot.point = None;
        }
        slot.prompt = Some(PromptState {
            tokens: tokens.to_vec(),
            next: start,
        });
        self.run_part(slot)
    }

    /// The next part of a prompt ([FusedArm::next_part], without the join bounds).
    ///
    /// # Errors
    /// A backend failure, or no prompt in parts.
    pub fn next_part_slot(&self, slot: &mut RuntimeSlot<W>) -> Result<Part, BackendError> {
        self.run_part(slot)
    }

    /// One step over the members ([FusedArm::step], without the join bounds): the round path
    /// when the drafter is present, a plain token row otherwise.
    ///
    /// # Errors
    /// A step before a prompt in parts finished, or a backend failure.
    pub fn step_slots(
        &self,
        members: &mut [(RuntimeSlot<W>, u32)],
    ) -> Result<Vec<StepOutput>, BackendError> {
        if members.iter().any(|(slot, _)| slot.prompt.is_some()) {
            return Err(BackendError::new(
                "runtime",
                "a prompt in parts was stepped before it finished",
            ));
        }
        if W::Drafter::PRESENT {
            self.draft_step(members)
        } else {
            self.plain_step(members)
        }
    }

    fn plain_step(
        &self,
        members: &mut [(RuntimeSlot<W>, u32)],
    ) -> Result<Vec<StepOutput>, BackendError> {
        let tokens: Vec<u32> = members.iter().map(|(_, token)| *token).collect();
        let hidden = {
            let mut segments: Vec<Segment<'_, W::Seq>> = members
                .iter_mut()
                .zip(tokens.iter())
                .map(|((slot, _), token)| Segment::new(&mut slot.seq, std::slice::from_ref(token)))
                .collect();
            self.walk.walk(&mut segments, Pass::token_rows())?
        };
        if hidden.rows != members.len() {
            return Err(BackendError::new(
                "runtime",
                format!(
                    "a token pass of {} rows left {} hidden rows",
                    members.len(),
                    hidden.rows
                ),
            ));
        }
        let mut head_rows: Vec<HeadRow<'_, W::Seq>> = members
            .iter_mut()
            .zip(tokens.iter())
            .enumerate()
            .map(|(row, ((slot, _), token))| HeadRow::token(&mut slot.seq, row, *token))
            .collect();
        let steps = self.walk.head(&hidden, &mut head_rows)?;
        for (slot, token) in members.iter_mut() {
            slot.history.push(*token);
        }
        Ok(steps)
    }
}

/// The ledger one [`Walk`] type states, derived from its types and its declared capabilities
/// with no device and no instance.
#[must_use]
pub fn ledger_for<W: Walk>() -> Ledger {
    let joint = crate::rounds::joint_rounds().unwrap_or(true);
    runtime_ledger::<W>(None, None, joint, None)
}

fn runtime_ledger<W: Walk>(
    chunk: Option<usize>,
    rewind: Option<bool>,
    joint: bool,
    drafter: Option<&str>,
) -> Ledger {
    let mut ledger = Ledger::new(std::any::type_name::<W>());
    let caps = W::CAPABILITIES;
    ledger.set(
        Strategy::PromptParts,
        if caps.passes.contains(crate::walk::PassMask::BULK) {
            Mode::native(match chunk {
                Some(rows) => format!("{rows}-row chunks from ChunkPolicy"),
                None => "chunks from ChunkPolicy".to_owned(),
            })
        } else {
            Mode::off("the walk does not take a bulk chunk")
        },
    );
    ledger.set(
        Strategy::Lanes,
        match caps.segments {
            crate::walk::Segments::Many => Mode::native("several segments a walk"),
            crate::walk::Segments::One => Mode::looped("one segment a walk"),
        },
    );
    // The three drafting rows follow from the drafter type and the round path the runtime now
    // drives, never from a hand-written value: a present drafter has every required method, so
    // its accepted runs reach the request; joint rounds follow the switch; an absent drafter
    // turns all three off with its reason.
    if W::Drafter::PRESENT {
        let name = drafter
            .map(str::to_owned)
            .unwrap_or_else(|| std::any::type_name::<W::Drafter>().to_owned());
        ledger.drafting(Some(name), joint, true);
    } else {
        ledger.drafting(None, joint, true);
    }
    ledger.set(
        Strategy::PrefixReuse,
        match caps.state {
            StateKind::None => Mode::off("the arm keeps no state"),
            StateKind::Positional => Mode::native("positional state, truncated to the prefix"),
            StateKind::Recurrent => Mode::native("recurrent state, restored from a checkpoint"),
            StateKind::Both => {
                Mode::native("positional state truncated and recurrent state checkpointed")
            }
        },
    );
    ledger.set(
        Strategy::Group,
        Mode::native("mirrored for any tensor-parallel group"),
    );
    // The four rows the arm's own kernels decide. The runtime cannot see inside a walk, so the
    // arm declares them on `WalkCapabilities`; a row left out is refused at boot.
    ledger.set(Strategy::TensorCorePrompt, caps.compute.prompt.into());
    ledger.set(Strategy::TensorCoreDecode, caps.compute.token.into());
    ledger.set(Strategy::StoredArithmetic, caps.compute.stored.into());
    // Rewind is declared by the arm and decided by the runtime's switch: when the switch is off
    // the ledger says so, and the same constant the walk declares cannot print native.
    ledger.set(
        Strategy::Rewind,
        match (rewind, caps.compute.rewind) {
            (Some(false), _) => {
                Mode::off(format!("{}=off keeps no rewind point", crate::REWIND_ENV))
            }
            (_, state) => state.into(),
        },
    );
    ledger
}

impl<W> FusedArm for Runtime<W>
where
    W: Walk + Send + Sync,
    W::Seq: Send,
{
    type Slot = RuntimeSlot<W>;

    fn open_slot(&self) -> Result<Self::Slot, BackendError> {
        Runtime::open_slot(self)
    }

    fn holds_request(&self, slot: &Self::Slot) -> bool {
        !slot.history.is_empty() || slot.prompt.is_some()
    }

    fn renew(&self, slot: &mut Self::Slot) -> Result<(), BackendError> {
        let fresh = self.open_slot()?;
        let sampling = slot.sampling;
        let old = std::mem::replace(slot, fresh);
        slot.sampling = sampling;
        self.walk.set_sampling(&mut slot.seq, sampling);
        self.release_lane(old.lane);
        Ok(())
    }

    fn set_sampling(&self, slot: &mut Self::Slot, sampling: Sampling) {
        slot.sampling = sampling;
        self.walk.set_sampling(&mut slot.seq, sampling);
    }

    fn prefill(&self, slot: &mut Self::Slot, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        self.prefill_slot(slot, tokens)
    }

    fn begin_prompt(
        &self,
        slot: &mut Self::Slot,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<Part, BackendError> {
        self.begin_prompt_slot(slot, tokens, images)
    }

    fn next_part(&self, slot: &mut Self::Slot) -> Result<Part, BackendError> {
        self.next_part_slot(slot)
    }

    fn step(&self, members: &mut [(Self::Slot, u32)]) -> Result<Vec<StepOutput>, BackendError> {
        self.step_slots(members)
    }

    fn ready(&self, slot: &Self::Slot, token: u32) -> bool {
        self.walk.drafter().ready(&slot.seq, token)
    }

    fn speculation(&self, slot: &Self::Slot) -> Speculation {
        self.walk.drafter().speculation(&slot.seq)
    }

    fn history<'s>(&self, slot: &'s Self::Slot) -> Option<&'s [u32]> {
        Some(&slot.history)
    }

    fn rewind_prefix<'s>(&self, slot: &'s Self::Slot) -> Option<&'s [u32]> {
        slot.point
            .filter(|point| *point <= slot.history.len())
            .map(|point| &slot.history[..point])
    }

    fn held_images<'s>(&self, slot: &'s Self::Slot) -> &'s [ImageSpan] {
        let _ = slot;
        &[]
    }

    fn cached_tokens(&self, slot: &Self::Slot) -> usize {
        slot.cached
    }

    fn room(&self, slot: &Self::Slot, prompt: Option<&[u32]>) -> bool {
        // A prompt's final position is its own length: it either extends the history or rewinds
        // to the point and forwards the rest. A step adds its anchor and, for a drafting walk,
        // the widest chain it may verify, so the reservation is the verify's rows and not one.
        let step_rows = if W::Drafter::PRESENT {
            1 + self.walk.drafter().budget(1)
        } else {
            1
        };
        let positions = match prompt {
            Some(prompt) => prompt.len(),
            None => slot.history.len().saturating_add(step_rows),
        };
        if positions > self.limits.context {
            return false;
        }
        // The arm answers the physical question: its lane memory may be narrower than the
        // context, and a prompt that starts over uses the chunks a fresh lane needs.
        match prompt {
            Some(prompt) => {
                let point = slot
                    .point
                    .filter(|point| *point <= slot.history.len())
                    .map(|point| &slot.history[..point]);
                let continues = crate::fused::extends(&slot.history, prompt)
                    || point.is_some_and(|point| crate::fused::extends(point, prompt));
                slot.seq.room(prompt.len(), !continues)
            }
            None => slot.seq.room(positions, false),
        }
    }

    fn release(&self, slot: &mut Self::Slot) -> Result<(), BackendError> {
        self.renew(slot)
    }

    fn give_back(&self, slot: &mut Self::Slot, count: usize) -> Result<(), BackendError> {
        if count == 0 {
            return Ok(());
        }
        let to = slot.history.len().checked_sub(count).ok_or_else(|| {
            BackendError::new(
                "runtime",
                format!(
                    "the slot holds {} tokens and was asked to give {count} back",
                    slot.history.len()
                ),
            )
        })?;
        self.walk.drafter().give_back(&mut slot.seq, to)?;
        slot.history.truncate(to);
        slot.point = slot.point.filter(|point| *point <= to);
        slot.cached = slot.cached.min(to);
        Ok(())
    }

    fn max_batch(&self) -> usize {
        if W::Drafter::PRESENT {
            // A verify refuses more rows than the rank's verify batch: a drafting walk's member
            // cap is the arm's verify width, not its prompt chunk.
            self.limits.members.max(1)
        } else {
            self.walk.max_segments().clamp(1, self.limits.rows)
        }
    }

    fn ledger(&self) -> Ledger {
        let drafter = W::Drafter::PRESENT.then(|| self.walk.drafter().describe());
        runtime_ledger::<W>(
            Some(self.chunk.rows()),
            Some(self.rewind),
            self.joint,
            drafter.as_deref(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use sparkflash_model::backend::BackendError;

    use std::collections::VecDeque;
    use std::sync::Arc;

    use sparkflash_model::verdict;

    use super::*;
    use crate::chunk::CHUNK_SIZES;
    use crate::layout::{LaneState, Region, StateKind};
    use crate::share::{BandWall, StepWall};
    use crate::walk::{
        Absent, Draft, HeadInput, Hidden, NoVision, OutputConstraint, Pass, PassKind, PassMask,
        Segment, WalkCapabilities,
    };

    /// The CPU reference walk: a sequence is the tokens it has forwarded, a walk appends rows,
    /// and a head picks the last token of the row it is asked about. It has no device, so it can
    /// prove the runtime's scheduling independently of any arm.
    #[derive(Debug, Default, PartialEq, Eq)]
    struct ToySeq {
        rows: Vec<u32>,
        checkpoint: Option<Vec<u32>>,
    }

    impl LaneState for ToySeq {
        fn position(&self) -> usize {
            self.rows.len()
        }

        fn checkpoint(&mut self, at: usize) -> Result<(), BackendError> {
            if at != self.rows.len() {
                return Err(BackendError::new(
                    "toy",
                    "a checkpoint must be at the position",
                ));
            }
            self.checkpoint = Some(self.rows.clone());
            Ok(())
        }

        fn rewind(&mut self, to: usize) -> Result<(), BackendError> {
            if to > self.rows.len() {
                return Err(BackendError::new("toy", "rewind past the position"));
            }
            match (to, self.checkpoint.take()) {
                (0, _) => self.rows.clear(),
                (_, Some(rows)) if rows.len() >= to => self.rows = rows[..to].to_vec(),
                _ => self.rows.truncate(to),
            }
            Ok(())
        }

        fn room(&self, _positions: usize, _reset: bool) -> bool {
            true
        }
    }

    struct ToyWalk {
        layout: StateLayout,
        limits: Limits,
        /// The most segments one walk takes, when narrower than the pass width.
        segments: usize,
        walks: Mutex<Vec<(PassKind, Vec<usize>)>>,
        sampling: Mutex<Option<Sampling>>,
        /// Rows the walk keeps past the ones the caller was given tokens for, as a verify keeps
        /// its drafts' rows. Zero is the plain walk.
        kept_past: usize,
        drafter: Absent<&'static str>,
    }

    impl ToyWalk {
        fn new(rows: usize, context: usize) -> Self {
            Self {
                layout: StateLayout::new(vec![Region::Positional {
                    name: "toy",
                    bytes_per_row: 4,
                }]),
                limits: Limits::new(rows, context),
                segments: rows,
                walks: Mutex::new(Vec::new()),
                sampling: Mutex::new(None),
                kept_past: 0,
                drafter: Absent("the toy walk drafts nothing"),
            }
        }

        /// The same walk, keeping `kept_past` rows past what the caller was given.
        fn keeping(mut self, kept_past: usize) -> Self {
            self.kept_past = kept_past;
            self
        }

        fn with_segments(mut self, segments: usize) -> Self {
            self.segments = segments;
            self
        }
    }

    impl Walk for ToyWalk {
        type Seq = ToySeq;
        type Drafter = Absent<&'static str>;
        type Vision = NoVision;

        const CAPABILITIES: WalkCapabilities = WalkCapabilities::all(StateKind::Positional);

        fn layout(&self) -> &StateLayout {
            &self.layout
        }

        fn open(&self, _memory: LaneMemory) -> Result<Self::Seq, BackendError> {
            Ok(ToySeq::default())
        }

        fn limits(&self) -> Limits {
            self.limits
        }

        fn drafter(&self) -> &Self::Drafter {
            &self.drafter
        }

        fn max_segments(&self) -> usize {
            self.segments
        }

        fn set_sampling(&self, _seq: &mut Self::Seq, sampling: Sampling) {
            *self.sampling.lock().expect("the toy sampler record") = Some(sampling);
        }

        fn walk(
            &self,
            segments: &mut [Segment<'_, Self::Seq>],
            pass: Pass<'_>,
        ) -> Result<Hidden, BackendError> {
            let kind = pass.kind;
            let lengths = segments
                .iter()
                .map(|segment| segment.rows.len())
                .collect::<Vec<_>>();
            self.walks
                .lock()
                .expect("the toy walk record")
                .push((kind, lengths));
            let mut rows = 0;
            for segment in segments {
                segment.seq.rows.extend_from_slice(segment.rows);
                rows += segment.rows.len();
                // A verify's rows stay in the state past the tokens the caller is given.
                segment.seq.rows.extend(vec![u32::MAX; self.kept_past]);
            }
            Ok(Hidden::new(rows))
        }

        fn head(
            &self,
            hidden: &Hidden,
            rows: &mut [HeadRow<'_, Self::Seq>],
        ) -> Result<Vec<StepOutput>, BackendError> {
            rows.iter_mut()
                .map(|row| {
                    debug_assert!(row.row < hidden.rows);
                    let token = row
                        .seq
                        .rows
                        .last()
                        .copied()
                        .ok_or_else(|| BackendError::new("toy", "no row to head"))?;
                    Ok(StepOutput {
                        logits: vec![f32::from(token as u16)],
                        next_token: Some(token.wrapping_add(1)),
                        draft: Vec::new(),
                    })
                })
                .collect()
        }
    }

    /// A sequence that also keeps the answers a verify already produced, oldest first.
    #[derive(Debug, Default)]
    struct DraftSeq {
        rows: Vec<u32>,
        checkpoint: Option<Vec<u32>>,
        pending: VecDeque<(u32, StepOutput)>,
        /// The positions the runtime asked its lane to hold room for.
        rooms: Mutex<Vec<usize>>,
    }

    impl LaneState for DraftSeq {
        fn position(&self) -> usize {
            self.rows.len()
        }

        fn checkpoint(&mut self, at: usize) -> Result<(), BackendError> {
            if at != self.rows.len() {
                return Err(BackendError::new(
                    "toy",
                    "a checkpoint must be at the position",
                ));
            }
            self.checkpoint = Some(self.rows.clone());
            Ok(())
        }

        fn rewind(&mut self, to: usize) -> Result<(), BackendError> {
            if to > self.rows.len() {
                return Err(BackendError::new("toy", "rewind past the position"));
            }
            self.rows.truncate(to);
            self.pending.clear();
            Ok(())
        }

        fn room(&self, positions: usize, _reset: bool) -> bool {
            self.rooms.lock().expect("the room record").push(positions);
            true
        }
    }

    /// What the toy drafter and its walk share: the chain it proposes, the passes it ran, and
    /// the counters a request reports.
    struct DraftShared {
        depth: usize,
        reject_at: Option<usize>,
        walks: Mutex<Vec<(PassKind, Vec<usize>)>>,
        headed: Mutex<Vec<usize>>,
        counters: Mutex<Speculation>,
    }

    /// A toy drafting walk: the model's next token is the last row plus one, the drafter
    /// proposes the same chain (or a wrong token at `reject_at`), and the verify settles with
    /// the shared verdict.
    struct DraftWalk {
        layout: StateLayout,
        limits: Limits,
        shared: Arc<DraftShared>,
        drafter: ToyDraftDrafter,
    }

    struct ToyDraftDrafter {
        shared: Arc<DraftShared>,
    }

    impl DraftWalk {
        fn new(depth: usize, reject_at: Option<usize>) -> Self {
            let shared = Arc::new(DraftShared {
                depth,
                reject_at,
                walks: Mutex::new(Vec::new()),
                headed: Mutex::new(Vec::new()),
                counters: Mutex::new(Speculation::default()),
            });
            Self {
                layout: StateLayout::new(vec![Region::Positional {
                    name: "toy",
                    bytes_per_row: 4,
                }]),
                limits: Limits::new(8, 4_096).with_members(4),
                drafter: ToyDraftDrafter {
                    shared: Arc::clone(&shared),
                },
                shared,
            }
        }

        fn passing() -> Self {
            Self::new(3, None)
        }

        fn rejecting() -> Self {
            Self::new(3, Some(1))
        }
    }

    impl Walk for DraftWalk {
        type Seq = DraftSeq;
        type Drafter = ToyDraftDrafter;
        type Vision = NoVision;

        const CAPABILITIES: WalkCapabilities = WalkCapabilities::all(StateKind::Positional);

        fn layout(&self) -> &StateLayout {
            &self.layout
        }

        fn open(&self, _memory: LaneMemory) -> Result<Self::Seq, BackendError> {
            Ok(DraftSeq::default())
        }

        fn limits(&self) -> Limits {
            self.limits
        }

        fn drafter(&self) -> &Self::Drafter {
            &self.drafter
        }

        fn cpu() -> Option<Self> {
            Some(Self::rejecting())
        }

        fn set_sampling(&self, _seq: &mut Self::Seq, _sampling: Sampling) {}

        fn walk(
            &self,
            segments: &mut [Segment<'_, Self::Seq>],
            pass: Pass<'_>,
        ) -> Result<Hidden, BackendError> {
            let lengths = segments
                .iter()
                .map(|segment| segment.rows.len())
                .collect::<Vec<_>>();
            self.shared
                .walks
                .lock()
                .expect("the toy walk record")
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
            _hidden: &Hidden,
            rows: &mut [HeadRow<'_, Self::Seq>],
        ) -> Result<Vec<StepOutput>, BackendError> {
            rows.iter_mut()
                .map(|row| {
                    let token = match &row.input {
                        HeadInput::Verify { rows, row: index } => {
                            self.shared
                                .headed
                                .lock()
                                .expect("the head record")
                                .push(*index);
                            rows.get(*index).copied().unwrap_or_default() + 1
                        }
                        HeadInput::Prompt(_) | HeadInput::Token(_) => row
                            .seq
                            .rows
                            .last()
                            .copied()
                            .ok_or_else(|| BackendError::new("toy", "no row to head"))?
                            .wrapping_add(1),
                    };
                    Ok(StepOutput {
                        logits: vec![f32::from(token as u16)],
                        next_token: Some(token),
                        draft: Vec::new(),
                    })
                })
                .collect()
        }
    }

    impl Drafter<DraftWalk> for ToyDraftDrafter {
        const PRESENT: bool = true;
        type Rows = ();

        fn wall(&self) -> &dyn StepWall {
            /// A flat wall: a draft row costs what an anchor row does, so every draft that can
            /// be accepted is worth its row.
            static WALL: BandWall = BandWall {
                a0: 1.0,
                a1: 0.0,
                b0: 1.0,
                b1: 0.0,
                seq_ms: 0.0,
                band_rows: 64,
            };
            &WALL
        }

        fn open_round(&self, seq: &mut DraftSeq, token: u32) -> Result<Draft, BackendError> {
            if let Some((front, _)) = seq.pending.front()
                && *front == token
            {
                let (_, step) = seq.pending.pop_front().expect("the front was just read");
                return Ok(Draft::Held(step));
            }
            let depth = self.shared.depth;
            let mut drafts = Vec::with_capacity(depth);
            let mut confidence = Vec::with_capacity(depth);
            for index in 0..depth {
                let correct = token.wrapping_add(1 + index as u32);
                drafts.push(if self.shared.reject_at == Some(index) {
                    token.wrapping_add(101 + index as u32)
                } else {
                    correct
                });
                confidence.push(0.9 - 0.05 * index as f64);
            }
            Ok(Draft::Chain { confidence, drafts })
        }

        fn verify(
            &self,
            walk: &DraftWalk,
            segments: &mut [Segment<'_, DraftSeq>],
            _depths: &[usize],
        ) -> Result<Vec<Self::Rows>, BackendError> {
            let rows = segments.iter().map(|segment| segment.rows.len()).sum();
            if rows > walk.limits.rows {
                return Err(BackendError::new(
                    "toy",
                    format!("a verify of {rows} rows is wider than the walk's pass"),
                ));
            }
            let constraint = OutputConstraint::new(&[], rows);
            walk.walk(segments, Pass::verify(rows, constraint))?;
            Ok(vec![(); segments.len()])
        }

        fn settle(
            &self,
            seq: &mut DraftSeq,
            chunk: &[u32],
            _rows: Self::Rows,
        ) -> Result<StepOutput, BackendError> {
            let picks: Vec<u32> = chunk.iter().map(|token| token.wrapping_add(1)).collect();
            let kept = verdict(chunk, &picks);
            let base = seq.rows.len().saturating_sub(chunk.len());
            seq.rows.truncate(base + kept);
            for &token in chunk.iter().take(kept).skip(1) {
                seq.pending.push_back((
                    token,
                    StepOutput {
                        logits: Vec::new(),
                        next_token: Some(token.wrapping_add(1)),
                        draft: Vec::new(),
                    },
                ));
            }
            let mut counters = self.shared.counters.lock().expect("the counters");
            counters.verify_steps += 1;
            counters.drafts_proposed += chunk.len() - 1;
            counters.drafts_accepted += kept - 1;
            Ok(StepOutput {
                logits: Vec::new(),
                next_token: Some(chunk[0].wrapping_add(1)),
                draft: Vec::new(),
            })
        }

        fn budget(&self, seqs: usize) -> usize {
            // One row a member, so a joint round shares a budget narrower than the chains and a
            // lone member still keeps its own depth.
            seqs
        }

        fn ready(&self, seq: &DraftSeq, token: u32) -> bool {
            seq.pending
                .front()
                .is_some_and(|(front, _)| *front == token)
        }

        fn give_back(&self, seq: &mut DraftSeq, to: usize) -> Result<(), BackendError> {
            seq.rows.truncate(to.min(seq.rows.len()));
            seq.pending.clear();
            Ok(())
        }

        fn speculation(&self, _seq: &DraftSeq) -> Speculation {
            *self.shared.counters.lock().expect("the counters")
        }
    }

    fn runtime(rows: usize) -> Runtime<ToyWalk> {
        Runtime::new(ToyWalk::new(rows, 4_096), 0).expect("the runtime starts")
    }

    fn prompt_of(runtime: &Runtime<ToyWalk>, tokens: &[u32]) -> Vec<u32> {
        let mut slot = runtime.open_slot().expect("a slot");
        runtime.prefill(&mut slot, tokens).expect("the prefill");
        slot.history.clone()
    }

    #[test]
    fn a_prompt_is_the_same_whole_chunked_or_a_token_at_a_time() {
        let whole = runtime(256);
        let chunked = runtime(8);
        let tokens = [3u32, 1, 4, 1, 5, 9, 2, 6, 5, 3];
        assert_eq!(prompt_of(&whole, &tokens), tokens);
        assert_eq!(prompt_of(&chunked, &tokens), tokens);
        let whole_kinds = whole
            .walk()
            .walks
            .lock()
            .expect("the toy walk record")
            .clone();
        let chunked_kinds = chunked
            .walk()
            .walks
            .lock()
            .expect("the toy walk record")
            .clone();
        assert!(
            chunked_kinds.len() > whole_kinds.len(),
            "the prompt was chunked"
        );
        assert_eq!(chunked_kinds[0].0, PassKind::Bulk { rows: 8 });
    }

    #[test]
    fn a_continued_prompt_forwards_only_the_rest() {
        let runtime = runtime(256);
        let mut slot = runtime.open_slot().expect("a slot");
        runtime
            .prefill(&mut slot, &[1, 2, 3])
            .expect("the first prompt");
        let before = slot.seq().rows.len();
        runtime
            .prefill(&mut slot, &[1, 2, 3, 4])
            .expect("the continued prompt");
        assert_eq!(slot.seq().rows, vec![1, 2, 3, 4]);
        assert_eq!(
            slot.seq().rows.len() - before,
            1,
            "the shared prefix was not forwarded again"
        );
    }

    /// A state that ends past the history (a verify kept its drafts' rows) cannot extend the
    /// history: the next prompt goes back to the rewind point instead of forwarding at a position
    /// the caller's text does not reach.
    #[test]
    fn a_prompt_extends_the_history_only_where_the_state_ends() {
        let runtime =
            Runtime::new(ToyWalk::new(256, 4_096).keeping(2), 0).expect("the runtime starts");
        let mut slot = runtime.open_slot().expect("a slot");
        runtime
            .prefill(&mut slot, &[1, 2, 3])
            .expect("the first prompt");
        // The walk kept two rows the caller was never given a token for.
        assert_eq!(slot.seq().rows, vec![1, 2, 3, u32::MAX, u32::MAX]);
        assert_eq!(slot.history(), &[1, 2, 3]);
        runtime
            .prefill(&mut slot, &[1, 2, 3, 9])
            .expect("the continuation");
        assert_eq!(
            slot.seq().rows,
            vec![1, 2, 3, 9, u32::MAX, u32::MAX],
            "the prompt went back to the point instead of forwarding from the kept rows"
        );
        assert_eq!(slot.history(), &[1, 2, 3, 9]);
    }

    /// The drafter's counters reach the serving layer through the runtime: the runtime cannot
    /// see inside a walk, so a drafter's report would otherwise read as zero.
    #[test]
    fn the_drafter_reports_a_sequence_s_speculation() {
        let runtime = Runtime::new(DraftWalk::new(3, None), 0).expect("the runtime starts");
        let mut slot = runtime.open_slot().expect("a slot");
        runtime.prefill(&mut slot, &[1]).expect("the prompt");
        assert_eq!(runtime.speculation(&slot), Speculation::default());
        runtime.step(&mut [(slot, 2)]).expect("the drafting step");
        let slot = runtime.open_slot().expect("a slot");
        assert_eq!(
            runtime.speculation(&slot),
            Speculation {
                verify_steps: 1,
                drafts_proposed: 3,
                drafts_accepted: 3,
            }
        );
    }

    /// The runtime asks the drafter what a slot already holds and what a give-back does, so an
    /// arm whose caches cannot go back (they consumed the unused answers) keeps its own rule.
    #[test]
    fn the_drafter_answers_a_run_and_a_give_back() {
        let runtime = Runtime::new(DraftWalk::new(3, None), 0).expect("the runtime starts");
        let mut slot = runtime.open_slot().expect("a slot");
        runtime.prefill(&mut slot, &[1]).expect("the prompt");
        let mut member = [(slot, 2u32)];
        let steps = runtime.step(&mut member).expect("the drafting step");
        let next = steps[0].next_token.expect("a token");
        assert!(
            runtime.ready(&member[0].0, next),
            "the verify holds the run"
        );
        assert!(!runtime.ready(&member[0].0, next.wrapping_add(100)));
        runtime
            .give_back(&mut member[0].0, 1)
            .expect("the drafter takes the give-back");
        assert_eq!(member[0].0.history(), &[1]);
        assert_eq!(member[0].0.seq().rows, vec![1]);
        assert!(!runtime.ready(&member[0].0, next));
    }

    #[test]
    fn a_rewound_prompt_equals_a_cold_one() {
        let runtime = runtime(256);
        let mut warm = runtime.open_slot().expect("a slot");
        runtime.prefill(&mut warm, &[1, 2, 3]).expect("the prompt");
        runtime
            .prefill(&mut warm, &[1, 2, 3, 4, 5])
            .expect("the reply");
        runtime
            .prefill(&mut warm, &[1, 2, 9, 9])
            .expect("the changed prompt");
        let mut cold = runtime.open_slot().expect("a slot");
        runtime
            .prefill(&mut cold, &[1, 2, 9, 9])
            .expect("the cold prompt");
        assert_eq!(warm.seq().rows, cold.seq().rows);
        assert_eq!(warm.history(), cold.history());
    }

    #[test]
    fn the_request_sampler_reaches_the_walk_on_a_fresh_and_a_renewed_slot() {
        let runtime = runtime(256);
        let sampling = Sampling {
            temperature: 0.7,
            repetition_penalty: 1.1,
            seed: 9,
            ..Sampling::default()
        };
        let mut slot = runtime.open_slot().expect("a slot");
        runtime.set_sampling(&mut slot, sampling);
        assert_eq!(
            *runtime
                .walk()
                .sampling
                .lock()
                .expect("the toy sampler record"),
            Some(sampling)
        );
        *runtime
            .walk()
            .sampling
            .lock()
            .expect("the toy sampler record") = None;
        runtime.renew(&mut slot).expect("the renewal");
        assert_eq!(
            *runtime
                .walk()
                .sampling
                .lock()
                .expect("the toy sampler record"),
            Some(sampling),
            "a renewed slot keeps the request's sampler"
        );
    }

    #[test]
    fn the_batch_cap_is_the_walks_and_not_the_pass_width() {
        // A prompt chunk may be wider than a token batch: the runtime caps members at the walk's
        // own answer, not at the prompt pass limit.
        let capped =
            Runtime::new(ToyWalk::new(64, 4_096).with_segments(3), 0).expect("the runtime starts");
        assert_eq!(capped.max_batch(), 3);
        assert_eq!(capped.limits().rows, 64);
        assert_eq!(runtime(64).max_batch(), 64, "the default is the pass width");
    }

    #[test]
    fn four_slots_step_in_one_walk_and_keep_their_own_rows() {
        let runtime = runtime(256);
        let mut slots = [
            runtime.open_slot().expect("a slot"),
            runtime.open_slot().expect("a slot"),
            runtime.open_slot().expect("a slot"),
            runtime.open_slot().expect("a slot"),
        ];
        for slot in &mut slots {
            runtime.prefill(slot, &[1]).expect("the prompt");
        }
        let mut members: Vec<(RuntimeSlot<ToyWalk>, u32)> = slots
            .into_iter()
            .map(|slot| {
                let token = slot.seq().rows[0] + 10;
                (slot, token)
            })
            .collect();
        let steps = runtime.step(&mut members).expect("one walk");
        assert_eq!(steps.len(), 4);
        let walks = runtime.walk().walks.lock().expect("the toy walk record");
        let last = walks.last().expect("the step walked");
        assert_eq!(last.0, PassKind::TokenRows);
        assert_eq!(last.1, vec![1, 1, 1, 1]);
    }

    /// `Runtime::verify` heads the rows in order, stops at the first draft the model did not
    /// choose, and appends only what the caller keeps. A verify that appended every draft (the
    /// shape this check exists to rule out) would leave `[1, 11, 12, 200, 201]` here and would
    /// have headed five rows.
    #[test]
    fn a_real_verify_stops_at_the_first_mismatch_and_appends_only_the_kept_rows() {
        let runtime = Runtime::new(DraftWalk::passing(), 0).expect("the runtime starts");
        let mut slot = runtime.open_slot().expect("a slot");
        runtime.prefill(&mut slot, &[1]).expect("the prompt");
        let mut members = [(slot, 11u32, &[12u32, 200, 201][..])];
        let constraint = OutputConstraint::new(&[], 8);
        let steps = runtime
            .verify(&mut members, constraint)
            .expect("the verify");
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].next_token, Some(12), "the anchor's own answer");
        assert_eq!(
            members[0].0.history(),
            &[1, 11, 12],
            "only the anchor and the accepted draft remain"
        );
        let walks = runtime
            .walk()
            .shared
            .walks
            .lock()
            .expect("the toy walk record");
        let last = walks.last().expect("the verify walked");
        assert_eq!(last.0, PassKind::Verify { rows: 4 });
        assert_eq!(last.1, vec![4]);
        let headed = runtime
            .walk()
            .shared
            .headed
            .lock()
            .expect("the head record");
        assert_eq!(&*headed, &[0, 1], "no row past the mismatch was headed");
    }

    /// The round path's first policy: one member verifies the drafter's own depth, with no row
    /// share even when the joint budget is narrower.
    #[test]
    fn one_member_verifies_the_drafters_own_depth() {
        let runtime = Runtime::new(DraftWalk::passing(), 0).expect("the runtime starts");
        assert_eq!(
            runtime.walk().drafter().budget(1),
            1,
            "the joint budget is narrow"
        );
        let mut slot = runtime.open_slot().expect("a slot");
        runtime.prefill(&mut slot, &[10]).expect("the prompt");
        runtime.step(&mut [(slot, 11)]).expect("the round");
        let walks = runtime
            .walk()
            .shared
            .walks
            .lock()
            .expect("the toy walk record");
        let verify = walks
            .iter()
            .find(|(kind, _)| matches!(kind, PassKind::Verify { .. }))
            .expect("the round verified");
        assert_eq!(verify.1, vec![4], "the anchor and all three own drafts");
    }

    /// The round path's joint policy: four members' chains share one verify pass and one row
    /// budget, instead of one pass each.
    #[test]
    fn four_members_share_one_verify_pass() {
        let runtime = Runtime::new(DraftWalk::passing(), 0).expect("the runtime starts");
        let mut slots = [
            runtime.open_slot().expect("a slot"),
            runtime.open_slot().expect("a slot"),
            runtime.open_slot().expect("a slot"),
            runtime.open_slot().expect("a slot"),
        ];
        for (index, slot) in slots.iter_mut().enumerate() {
            runtime
                .prefill(slot, &[index as u32 + 1])
                .expect("the prompt");
        }
        let mut members: Vec<(RuntimeSlot<DraftWalk>, u32)> = slots
            .into_iter()
            .enumerate()
            .map(|(index, slot)| (slot, index as u32 + 11))
            .collect();
        runtime.step(&mut members).expect("one joint round");
        let walks = runtime
            .walk()
            .shared
            .walks
            .lock()
            .expect("the toy walk record");
        let verifies: Vec<_> = walks
            .iter()
            .filter(|(kind, _)| matches!(kind, PassKind::Verify { .. }))
            .collect();
        assert_eq!(verifies.len(), 1, "one verify pass for the four members");
        assert_eq!(verifies[0].1, vec![2, 2, 2, 2], "one shared draft a member");
    }

    /// The round path's second policy: with joint rounds off, each lane's round runs alone.
    #[test]
    fn with_joint_off_each_lane_runs_alone() {
        let runtime = Runtime::new(DraftWalk::passing(), 0)
            .expect("the runtime starts")
            .with_joint_rounds(false);
        let mut slots = [
            runtime.open_slot().expect("a slot"),
            runtime.open_slot().expect("a slot"),
        ];
        for (index, slot) in slots.iter_mut().enumerate() {
            runtime
                .prefill(slot, &[index as u32 + 1])
                .expect("the prompt");
        }
        let mut members: Vec<(RuntimeSlot<DraftWalk>, u32)> = slots
            .into_iter()
            .enumerate()
            .map(|(index, slot)| (slot, index as u32 + 11))
            .collect();
        runtime.step(&mut members).expect("two lone rounds");
        let walks = runtime
            .walk()
            .shared
            .walks
            .lock()
            .expect("the toy walk record");
        let verifies: Vec<_> = walks
            .iter()
            .filter(|(kind, _)| matches!(kind, PassKind::Verify { .. }))
            .collect();
        assert_eq!(verifies.len(), 2, "each lane walked its own round");
        assert_eq!(verifies[0].1, vec![4]);
        assert_eq!(verifies[1].1, vec![4]);
    }

    /// A step reserves the verify's rows, not one: the lane is asked to hold the anchor and the
    /// widest chain the drafter may verify.
    #[test]
    fn a_step_reserves_the_verify_rows() {
        let runtime = Runtime::new(DraftWalk::passing(), 0).expect("the runtime starts");
        let slot = runtime.open_slot().expect("a slot");
        assert!(runtime.room(&slot, None));
        assert_eq!(
            *slot
                .seq()
                .rooms
                .lock()
                .expect("the room record")
                .last()
                .unwrap(),
            1 + runtime.walk().drafter().budget(1),
            "the anchor and the drafter's chain, not one row"
        );
    }

    /// A rejected draft gives its rows back: the state keeps the anchor and the draft the model
    /// chose, and the rows past the mismatch do not reach the next step.
    #[test]
    fn a_rejected_draft_gives_back_correctly() {
        let runtime = Runtime::new(DraftWalk::rejecting(), 0).expect("the runtime starts");
        let mut slot = runtime.open_slot().expect("a slot");
        runtime.prefill(&mut slot, &[1]).expect("the prompt");
        let mut member = [(slot, 2u32)];
        let steps = runtime.step(&mut member).expect("the round");
        assert_eq!(steps[0].next_token, Some(3));
        assert_eq!(
            member[0].0.seq().rows,
            vec![1, 2, 3],
            "the rejected draft's rows were given back"
        );
        assert_eq!(member[0].0.history(), &[1, 2]);
        let walks = runtime
            .walk()
            .shared
            .walks
            .lock()
            .expect("the toy walk record");
        let verifies = walks
            .iter()
            .filter(|(kind, _)| matches!(kind, PassKind::Verify { .. }))
            .count();
        drop(walks);
        // The accepted draft's answer is held, so feeding it costs no verify.
        member[0].1 = 3;
        let steps = runtime.step(&mut member).expect("the held step");
        assert_eq!(steps[0].next_token, Some(4));
        assert_eq!(member[0].0.history(), &[1, 2, 3]);
        let walks = runtime
            .walk()
            .shared
            .walks
            .lock()
            .expect("the toy walk record");
        assert_eq!(
            walks
                .iter()
                .filter(|(kind, _)| matches!(kind, PassKind::Verify { .. }))
                .count(),
            verifies,
            "a held answer costs no device rows"
        );
    }

    #[test]
    fn the_ledger_says_when_the_rewind_switch_is_off() {
        let ledger = runtime_ledger::<ToyWalk>(Some(256), Some(false), true, None);
        assert_eq!(
            ledger.mode(Strategy::Rewind),
            Mode::off(format!("{}=off keeps no rewind point", crate::REWIND_ENV))
        );
        // Without an instance (the preview path) the declaration is what the arm states.
        assert!(matches!(
            ledger_for::<ToyWalk>().mode(Strategy::Rewind),
            Mode::Native(_)
        ));
    }

    #[test]
    fn the_ledger_is_derived_from_the_types_and_capabilities() {
        let ledger = ledger_for::<ToyWalk>();
        assert!(matches!(
            ledger.mode(Strategy::PromptParts),
            Mode::Native(_)
        ));
        assert!(matches!(ledger.mode(Strategy::Lanes), Mode::Native(_)));
        assert_eq!(
            ledger.mode(Strategy::Drafting),
            Mode::off("no drafter configured")
        );
        assert!(matches!(
            ledger.mode(Strategy::PrefixReuse),
            Mode::Native(_)
        ));
        assert!(matches!(ledger.mode(Strategy::Group), Mode::Native(_)));
        assert!(!ledger.line().contains("the arm does not state it"));
    }

    #[test]
    fn the_capability_mask_names_every_pass_the_trait_defines() {
        for pass in [
            PassKind::TokenRows,
            PassKind::Bulk { rows: 1 },
            PassKind::Paired { chunks: 2 },
            PassKind::Verify { rows: 3 },
        ] {
            assert!(WalkCapabilities::all(StateKind::Positional).accepts(pass));
        }
        assert!(PassMask::ALL.contains(PassMask::TOKEN_ROWS.union(PassMask::VERIFY)));
    }

    #[test]
    fn chunk_policy_names_are_the_documented_ones() {
        assert!(CHUNK_SIZES.contains(&256));
        assert_eq!(CHUNK_SIZES.iter().copied().max(), Some(65_535));
    }

    /// A drafting walk is constructed and served: the round path exists, so the ledger states
    /// drafting and a run, and the joint row follows the switch rather than a constant.
    /// The seventh generated check runs on the toy: a run with drafts (and a rejected one)
    /// answers exactly the tokens the plain run does.
    #[test]
    fn the_toy_drafted_run_equals_a_plain_one() {
        crate::run_check::<DraftWalk>(crate::ConformanceCheck::DraftedEqualsPlain)
            .expect("drafted = plain on the toy");
    }

    #[test]
    fn a_drafting_walk_serves_and_its_ledger_rows_are_derived() {
        let runtime = Runtime::new(DraftWalk::passing(), 0)
            .expect("a drafting walk serves through the round path");
        assert!(matches!(
            runtime.ledger().mode(Strategy::Drafting),
            Mode::Native(_)
        ));
        assert!(matches!(
            runtime.ledger().mode(Strategy::Runs),
            Mode::Native(_)
        ));
        assert!(matches!(
            runtime.ledger().mode(Strategy::JointRounds),
            Mode::Native(_)
        ));
        let lone = runtime.with_joint_rounds(false);
        assert!(matches!(
            lone.ledger().mode(Strategy::JointRounds),
            Mode::Looped(_)
        ));
        let preview = ledger_for::<DraftWalk>();
        assert!(matches!(preview.mode(Strategy::Drafting), Mode::Native(_)));
        assert!(matches!(preview.mode(Strategy::Runs), Mode::Native(_)));
        assert!(!preview.line().contains("the arm does not state it"));
    }

    /// Drafted-equals-plain is claimed only by an arm that declares a drafter.
    #[test]
    fn the_seventh_check_requires_a_drafter() {
        use crate::ConformanceCheck;
        let drafted = ConformanceCheck::DraftedEqualsPlain;
        assert!(crate::checks_for::<DraftWalk>().contains(&drafted));
        assert!(!crate::checks_for::<ToyWalk>().contains(&drafted));
        assert!(crate::check_claims::<DraftWalk>(&[drafted]).is_ok());
        assert!(crate::check_claims::<ToyWalk>(&[drafted]).is_err());
        assert_eq!(ConformanceCheck::ALL.len(), 7);
    }

    /// The toy walk's fixture: the runtime's CPU walk, with no group on this host.
    impl crate::WalkFixture for ToyWalk {
        fn fixture(world: usize) -> Result<crate::Fixture<Self>, BackendError> {
            if world > 1 {
                return Err(BackendError::new(
                    "group",
                    "the toy walk has one rank; there is no TP group",
                ));
            }
            Ok(crate::Fixture::new(
                Self::new(8, 64),
                "the runtime's toy walk",
                4,
                4,
                Vec::new(),
                None,
            ))
        }
    }

    /// The drafting toy's fixture: the rejecting chain, so the check covers a rejected draft.
    impl crate::WalkFixture for DraftWalk {
        fn fixture(world: usize) -> Result<crate::Fixture<Self>, BackendError> {
            if world > 1 {
                return Err(BackendError::new(
                    "group",
                    "the drafting toy walk has one rank; there is no TP group",
                ));
            }
            Ok(crate::Fixture::new(
                Self::rejecting(),
                "the runtime's drafting toy walk (a rejected draft)",
                4,
                4,
                Vec::new(),
                None,
            ))
        }
    }

    /// Every generated check runs on the toy fixture; TP names its missing group rather than
    /// passing vacuously.
    #[test]
    fn the_generated_checks_run_on_the_toy_fixture() {
        use crate::{CheckOutcome, ConformanceCheck, run_check};
        for check in [
            ConformanceCheck::WholePromptEqualsChunked,
            ConformanceCheck::AloneEqualsFour,
            ConformanceCheck::ContinuedEqualsCold,
            ConformanceCheck::RewoundEqualsCold,
            ConformanceCheck::NoUnrecordedWidening,
        ] {
            let outcome = run_check::<ToyWalk>(check)
                .unwrap_or_else(|error| panic!("{}: {}", check.name(), error.message()));
            assert_eq!(outcome, CheckOutcome::Passed, "{}", check.name());
        }
        let tp = run_check::<ToyWalk>(ConformanceCheck::TpWithinExactness)
            .expect("the TP check skips, it does not fail");
        assert!(
            matches!(&tp, CheckOutcome::Skipped(reason) if reason.contains("no TP group")),
            "the TP check names its missing group: {tp:?}"
        );
        let drafted = run_check::<DraftWalk>(ConformanceCheck::DraftedEqualsPlain)
            .expect("the drafting toy fixture runs");
        assert_eq!(drafted, CheckOutcome::Passed);
    }

    crate::conformance!(ToyWalk);
}

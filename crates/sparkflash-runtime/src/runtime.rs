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
use crate::walk::{Drafter, HeadRow, Pass, Segment, Vision, Walk};

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
    speculation: Speculation,
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
}

impl<W: Walk> Runtime<W> {
    /// A runtime over `walk`, with `free_bytes` free for chunk sizing (zero: not measured).
    ///
    /// # Errors
    /// A [`crate::REWIND_ENV`] that is not a switch.
    pub fn new(walk: W, free_bytes: u64) -> Result<Self, BackendError> {
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
        })
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
            self.walk.walk(&mut segments, Pass::bulk(rows.len()))?
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

    /// One verify pass over the anchor and drafts of several members: the row-level half of
    /// the joint-round glue. The round loop decides each member's depth ([crate::run_round]);
    /// this turns every member's anchor plus its drafts into one [`Pass::Verify`], under the
    /// caller's output constraint, and heads each member's anchor row.
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
        let mut head_rows: Vec<HeadRow<'_, W::Seq>> = members
            .iter_mut()
            .zip(tokens.iter())
            .zip(offsets.iter())
            .map(|(((slot, anchor, _), _), row)| HeadRow::token(&mut slot.seq, *row, *anchor))
            .collect();
        let steps = self.walk.head(&hidden, &mut head_rows)?;
        for ((slot, _, _), rows) in members.iter_mut().zip(tokens.iter()) {
            slot.history.extend_from_slice(rows);
        }
        Ok(steps)
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
    runtime_ledger::<W>(None, None)
}

fn runtime_ledger<W: Walk>(chunk: Option<usize>, rewind: Option<bool>) -> Ledger {
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
    if W::Drafter::PRESENT {
        ledger.drafting(
            Some(std::any::type_name::<W::Drafter>().to_owned()),
            caps.passes.contains(crate::walk::PassMask::VERIFY)
                && caps.segments == crate::walk::Segments::Many,
            true,
        );
    } else {
        ledger.drafting(None, false, false);
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
            speculation: Speculation::default(),
        })
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
        let mut part = self.begin_prompt(slot, tokens, &[])?;
        loop {
            match part {
                Part::Done(step) => return Ok(step),
                Part::More => part = self.next_part(slot)?,
            }
        }
    }

    fn begin_prompt(
        &self,
        slot: &mut Self::Slot,
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

    fn next_part(&self, slot: &mut Self::Slot) -> Result<Part, BackendError> {
        self.run_part(slot)
    }

    fn step(&self, members: &mut [(Self::Slot, u32)]) -> Result<Vec<StepOutput>, BackendError> {
        if members.iter().any(|(slot, _)| slot.prompt.is_some()) {
            return Err(BackendError::new(
                "runtime",
                "a prompt in parts was stepped before it finished",
            ));
        }
        self.plain_step(members)
    }

    fn speculation(&self, slot: &Self::Slot) -> Speculation {
        // The counters are the drafter's, which only the arm can read; the slot's own field is
        // the fallback for a walk that states nothing.
        let measured = self.walk.speculation(&slot.seq);
        if measured == Speculation::default() {
            slot.speculation
        } else {
            measured
        }
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
        // to the point and forwards the rest. A step adds one row to the history.
        let positions = match prompt {
            Some(prompt) => prompt.len(),
            None => slot.history.len().saturating_add(1),
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
        slot.seq.rewind(to)?;
        slot.history.truncate(to);
        slot.point = slot.point.filter(|point| *point <= to);
        slot.cached = slot.cached.min(to);
        Ok(())
    }

    fn max_batch(&self) -> usize {
        self.limits.rows
    }

    fn ledger(&self) -> Ledger {
        runtime_ledger::<W>(Some(self.chunk.rows()), Some(self.rewind))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use sparkflash_model::backend::BackendError;

    use super::*;
    use crate::chunk::CHUNK_SIZES;
    use crate::layout::{LaneState, Region, StateKind};
    use crate::walk::{
        Absent, Hidden, NoVision, Pass, PassKind, PassMask, Segment, WalkCapabilities,
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
        walks: Mutex<Vec<(PassKind, Vec<usize>)>>,
        sampling: Mutex<Option<Sampling>>,
        /// Rows the walk keeps past the ones the caller was given tokens for, as a verify keeps
        /// its drafts' rows. Zero is the plain walk.
        kept_past: usize,
        /// What the walk reports for a sequence's speculative counters.
        counters: Speculation,
    }

    impl ToyWalk {
        fn new(rows: usize, context: usize) -> Self {
            Self {
                layout: StateLayout::new(vec![Region::Positional {
                    name: "toy",
                    bytes_per_row: 4,
                }]),
                limits: Limits::new(rows, context),
                walks: Mutex::new(Vec::new()),
                sampling: Mutex::new(None),
                kept_past: 0,
                counters: Speculation::default(),
            }
        }

        /// The same walk, keeping `kept_past` rows past what the caller was given.
        fn keeping(mut self, kept_past: usize) -> Self {
            self.kept_past = kept_past;
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

        fn set_sampling(&self, _seq: &mut Self::Seq, sampling: Sampling) {
            *self.sampling.lock().expect("the toy sampler record") = Some(sampling);
        }

        fn speculation(&self, _seq: &Self::Seq) -> Speculation {
            self.counters
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

    /// The arm's counters reach the serving layer through the runtime: the runtime cannot see
    /// inside a walk, so a drafter's report would otherwise read as zero.
    #[test]
    fn the_walk_reports_a_sequence_s_speculation() {
        let mut walk = ToyWalk::new(256, 4_096);
        walk.counters = Speculation {
            verify_steps: 3,
            drafts_proposed: 9,
            drafts_accepted: 4,
        };
        let runtime = Runtime::new(walk, 0).expect("the runtime starts");
        let slot = runtime.open_slot().expect("a slot");
        assert_eq!(
            runtime.speculation(&slot),
            Speculation {
                verify_steps: 3,
                drafts_proposed: 9,
                drafts_accepted: 4,
            }
        );
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

    #[test]
    fn the_joint_round_glue_sends_one_verify_pass_and_heads_each_anchor() {
        let runtime = runtime(256);
        let mut slot = runtime.open_slot().expect("a slot");
        runtime.prefill(&mut slot, &[1]).expect("the prompt");
        let mut members = [(slot, 11u32, &[12u32, 13][..])];
        let constraint = crate::OutputConstraint::new(&[11, 12, 13], 2);
        let steps = runtime
            .verify(&mut members, constraint)
            .expect("the verify");
        assert_eq!(steps.len(), 1);
        let walks = runtime.walk().walks.lock().expect("the toy walk record");
        let last = walks.last().expect("the verify walked");
        assert_eq!(last.0, PassKind::Verify { rows: 3 });
        assert_eq!(last.1, vec![3]);
        assert_eq!(members[0].0.history(), &[1, 11, 12, 13]);
    }

    #[test]
    fn the_ledger_says_when_the_rewind_switch_is_off() {
        let ledger = runtime_ledger::<ToyWalk>(Some(256), Some(false));
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

    crate::conformance!(ToyWalk);
}

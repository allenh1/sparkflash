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

//! Test support for model arms: the engine driven at an arm's widths, with no device.
//!
//! An arm's CPU tests ask what the runtime does at the arm's own numbers: which chunks a prompt
//! runs in, how wide a verify is, which decode plan several lanes take. None of that needs the
//! arm's arithmetic. [`ShapeWalk`] is a walk whose model is a counter ([`pick`]) and whose
//! drafter proposes a known chain, so an arm states only what is its own (its layout, its
//! limits, its verify budget, its step wall) and reads back the passes the engine handed the
//! walk ([`ShapeWalk::chunks`], [`ShapeWalk::verifies`]).
//!
//! The module is compiled into the library, as [`mod@crate::conformance`] is, so every arm's test
//! module can name it. Nothing here is on a serving path.

use std::collections::VecDeque;
use std::sync::Mutex;

use sparkflash_model::backend::{BackendError, Sampling, Speculation, StepOutput};
use sparkflash_model::verdict;

use crate::engine::{RequestId, TickEngine, TickEvent};
use crate::layout::{LaneState, Limits, StateKind, StateLayout, check_checkpoint, check_rewind};
use crate::share::{DraftBudget, StepWall};
use crate::walk::{
    Draft, Drafter, HeadInput, HeadRow, Hidden, NoVision, OutputConstraint, Pass, PassKind,
    Segment, Walk, WalkCapabilities,
};

/// The model's own pick after `token`: the plain path's next token.
#[must_use]
pub const fn pick(token: u32) -> u32 {
    token.wrapping_add(1)
}

/// A token the model never picks after `token`, so a draft of it is always rejected.
#[must_use]
pub const fn wrong(token: u32) -> u32 {
    token.wrapping_add(7)
}

/// The chain a [`ShapeDrafter`] proposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Chain {
    /// The model's own pick, then a wrong token: a round keeps one draft and stops at the next.
    /// The confidence falls 0.1 a draft.
    PickThenWrong,
    /// The model's own picks all the way: a round keeps every draft it was walked over. The
    /// confidence falls 0.05 a draft.
    Picks,
}

/// The default wall of a shared round: flat, so the row share spends the budget by confidence
/// alone.
struct FlatWall;

impl StepWall for FlatWall {
    fn ms(&self, _seqs: usize, _rows: usize) -> f64 {
        1.0
    }
}

/// A CPU walk at an arm's widths.
///
/// The arm gives its state layout and its limits; the walk records every prompt chunk and
/// every verify pass the engine hands it. Its sequence is the tokens it forwarded, its head
/// answers [`pick`] of the last row, and its drafter proposes [`pick`] followed by [`wrong`],
/// so a speculative round at a depth of two or more keeps one draft and rejects the next. It is
/// the shape the engine is driven at, not an arm's arithmetic.
///
/// It declares every pass ([`WalkCapabilities::all`]) with [`StateKind::Both`], takes one
/// segment in a bulk pass and refuses a paired pass.
pub struct ShapeWalk {
    layout: StateLayout,
    limits: Limits,
    /// The code of every error the walk and its sequences return.
    code: &'static str,
    drafter: ShapeDrafter,
    /// Every prompt chunk the walk was handed, in order.
    chunks: Mutex<Vec<usize>>,
    /// Every verify pass, one entry a member's rows.
    verifies: Mutex<Vec<Vec<usize>>>,
}

impl ShapeWalk {
    /// A walk with `layout` as its state and `limits` as its widths, whose drafter proposes
    /// at most `depth` drafts a member.
    ///
    /// The defaults are a flat wall of one modelled millisecond, the chain [`pick`] then
    /// [`wrong`], the verify budget [`DraftBudget::AnchorsAndDrafts`] and the error code `shape`.
    #[must_use]
    pub fn new(layout: StateLayout, limits: Limits, depth: usize) -> Self {
        Self {
            layout,
            limits,
            code: "shape",
            drafter: ShapeDrafter {
                depth,
                chain: Chain::PickThenWrong,
                budget: DraftBudget::AnchorsAndDrafts,
                wall: Box::new(FlatWall),
            },
            chunks: Mutex::new(Vec::new()),
            verifies: Mutex::new(Vec::new()),
        }
    }

    /// The same walk whose errors carry `code`, so a failure names the arm that drove it.
    #[must_use]
    pub fn with_code(mut self, code: &'static str) -> Self {
        self.code = code;
        self
    }

    /// The same walk with the arm's own [`Drafter::budget`] rule.
    #[must_use]
    pub fn with_budget(mut self, rule: DraftBudget) -> Self {
        self.drafter.budget = rule;
        self
    }

    /// The same walk with the arm's own step wall for the row share.
    #[must_use]
    pub fn with_wall(mut self, wall: Box<dyn StepWall + Send + Sync>) -> Self {
        self.drafter.wall = wall;
        self
    }

    /// The same walk whose drafter is always right: a chain is `depth` of the model's own
    /// picks, its confidence falling 0.05 a draft, so a round keeps every row it verified.
    #[must_use]
    pub fn with_right_picks(mut self) -> Self {
        self.drafter.chain = Chain::Picks;
        self
    }

    /// Every prompt chunk the walk was handed, in order.
    ///
    /// # Panics
    /// If a walk panicked while it held the record.
    #[must_use]
    pub fn chunks(&self) -> Vec<usize> {
        self.chunks.lock().expect("the chunk record").clone()
    }

    /// Every verify pass the walk was handed, in order: one entry a member, the rows of its
    /// anchor and the drafts verified with it.
    ///
    /// # Panics
    /// If a walk panicked while it held the record.
    #[must_use]
    pub fn verifies(&self) -> Vec<Vec<usize>> {
        self.verifies.lock().expect("the verify record").clone()
    }
}

/// A [`ShapeWalk`]'s sequence: the tokens it forwarded and still holds, plus the answers a
/// round queued.
#[derive(Debug)]
pub struct ShapeSeq {
    code: &'static str,
    rows: Vec<u32>,
    pending: VecDeque<(u32, StepOutput)>,
    point: Option<Vec<u32>>,
}

impl LaneState for ShapeSeq {
    fn position(&self) -> usize {
        self.rows.len()
    }

    fn checkpoint(&mut self, at: usize) -> Result<(), BackendError> {
        check_checkpoint(self.code, at, self.rows.len())?;
        self.point = Some(self.rows.clone());
        Ok(())
    }

    fn rewind(&mut self, to: usize) -> Result<(), BackendError> {
        check_rewind(self.code, to, self.rows.len())?;
        match (to, self.point.take()) {
            (0, _) => self.rows.clear(),
            (_, Some(rows)) if rows.len() >= to => self.rows = rows[..to].to_vec(),
            _ => self.rows.truncate(to),
        }
        self.pending.clear();
        Ok(())
    }

    fn room(&self, _positions: usize, _reset: bool) -> bool {
        true
    }
}

/// A [`ShapeWalk`]'s drafter: a known chain, the arm's budget and the arm's wall.
pub struct ShapeDrafter {
    /// The drafts a chain holds past the anchor.
    depth: usize,
    chain: Chain,
    /// The arm's rule for the most drafts a verify of `seqs` members holds at `depth`.
    budget: DraftBudget,
    wall: Box<dyn StepWall + Send + Sync>,
}

impl Walk for ShapeWalk {
    type Seq = ShapeSeq;
    type Hidden = Hidden;
    type Drafter = ShapeDrafter;
    type Vision = NoVision;

    const CAPABILITIES: WalkCapabilities = WalkCapabilities::all(StateKind::Both);

    fn layout(&self) -> &StateLayout {
        &self.layout
    }

    fn open(&self) -> Result<Self::Seq, BackendError> {
        Ok(ShapeSeq {
            code: self.code,
            rows: Vec::new(),
            pending: VecDeque::new(),
            point: None,
        })
    }

    fn limits(&self) -> Limits {
        self.limits
    }

    fn drafter(&self) -> &Self::Drafter {
        &self.drafter
    }

    fn set_sampling(&self, _seq: &mut Self::Seq, _sampling: Sampling) {}

    fn walk(
        &self,
        segments: &mut [Segment<'_, Self::Seq>],
        mut pass: Pass<'_>,
    ) -> Result<Hidden, BackendError> {
        match pass.kind {
            PassKind::TokenRows => {}
            PassKind::Bulk { .. } => {
                if segments.len() != 1 {
                    return Err(BackendError::new(
                        self.code,
                        "a bulk prompt pass holds one segment",
                    ));
                }
                let rows = segments[0].rows.len();
                self.chunks.lock().expect("the chunk record").push(rows);
            }
            PassKind::Verify { .. } => self
                .verifies
                .lock()
                .expect("the verify record")
                .push(segments.iter().map(|segment| segment.rows.len()).collect()),
            PassKind::Paired { .. } => {
                return Err(BackendError::new(
                    self.code,
                    "the shape walk does not pair several prompts in one pass",
                ));
            }
        }
        let mut rows = 0;
        for segment in segments {
            segment.seq.rows.extend_from_slice(segment.rows);
            rows += segment.rows.len();
        }
        let hidden = Hidden::new(rows);
        if let Some(hook) = pass.hook.take() {
            hook.after_layers(&hidden)?;
        }
        Ok(hidden)
    }

    fn head(
        &self,
        _hidden: &mut Hidden,
        rows: &mut [HeadRow<'_, Self::Seq>],
    ) -> Result<Vec<StepOutput>, BackendError> {
        Ok(rows
            .iter_mut()
            .map(|row| {
                let token = match &row.input {
                    HeadInput::Verify { rows, row: index } => {
                        pick(rows.get(*index).copied().unwrap_or_default())
                    }
                    HeadInput::Prompt(_) | HeadInput::Token(_) => {
                        row.seq.rows.last().copied().map_or(0, pick)
                    }
                };
                StepOutput {
                    logits: Vec::new(),
                    next_token: Some(token),
                    draft: Vec::new(),
                }
            })
            .collect())
    }
}

impl Drafter<ShapeWalk> for ShapeDrafter {
    const PRESENT: bool = true;
    type Rows = ();

    fn wall(&self) -> &dyn StepWall {
        &*self.wall
    }

    fn open_round(&self, seq: &mut ShapeSeq, token: u32) -> Result<Draft, BackendError> {
        if let Some((front, _)) = seq.pending.front()
            && *front == token
        {
            let (_, step) = seq.pending.pop_front().expect("the front was read");
            return Ok(Draft::Held(step));
        }
        let (drafts, fall): (Vec<u32>, f64) = match self.chain {
            Chain::PickThenWrong => (
                [pick(token), wrong(token)]
                    .into_iter()
                    .take(self.depth)
                    .collect(),
                0.1,
            ),
            Chain::Picks => (
                std::iter::successors(Some(pick(token)), |&last| Some(pick(last)))
                    .take(self.depth)
                    .collect(),
                0.05,
            ),
        };
        let confidence = (0..drafts.len()).map(|at| 0.9 - fall * at as f64).collect();
        Ok(Draft::Chain { confidence, drafts })
    }

    fn verify(
        &self,
        walk: &ShapeWalk,
        segments: &mut [Segment<'_, ShapeSeq>],
        _depths: &[usize],
    ) -> Result<Vec<Self::Rows>, BackendError> {
        let rows: usize = segments.iter().map(|segment| segment.rows.len()).sum();
        walk.walk(
            segments,
            Pass::verify(rows, OutputConstraint::new(&[], rows)),
        )?;
        Ok(vec![(); segments.len()])
    }

    fn settle(
        &self,
        seq: &mut ShapeSeq,
        chunk: &[u32],
        _rows: Self::Rows,
    ) -> Result<StepOutput, BackendError> {
        let picks: Vec<u32> = chunk.iter().map(|token| pick(*token)).collect();
        let kept = verdict(chunk, &picks);
        let base = seq.rows.len().saturating_sub(chunk.len());
        seq.rows.truncate(base + kept);
        for (index, &token) in chunk.iter().enumerate().take(kept).skip(1) {
            seq.pending.push_back((
                token,
                StepOutput {
                    logits: Vec::new(),
                    next_token: picks.get(index).copied(),
                    draft: Vec::new(),
                },
            ));
        }
        Ok(StepOutput {
            logits: Vec::new(),
            next_token: picks.first().copied(),
            draft: Vec::new(),
        })
    }

    fn budget(&self, seqs: usize) -> usize {
        self.budget.rows(seqs, self.depth)
    }

    fn ready(&self, seq: &ShapeSeq, token: u32) -> bool {
        seq.pending
            .front()
            .is_some_and(|(front, _)| *front == token)
    }

    fn give_back(&self, seq: &mut ShapeSeq, to: usize) -> Result<(), BackendError> {
        seq.rows.truncate(to.min(seq.rows.len()));
        seq.pending.clear();
        Ok(())
    }

    fn speculation(&self, _seq: &ShapeSeq) -> Speculation {
        Speculation::default()
    }
}

/// Admits a prompt and ticks until its first step.
///
/// # Panics
/// If the engine refuses the prompt or a tick fails.
pub fn prompt<W: Walk>(engine: &mut TickEngine<W>, tokens: &[u32]) -> (RequestId, StepOutput) {
    let id = engine
        .admit(tokens, &[], Sampling::default())
        .expect("the prompt is admitted");
    loop {
        for event in engine.tick().expect("the tick") {
            if let TickEvent::Prompt { id: got, step } = event
                && got == id
            {
                return (id, step);
            }
        }
    }
}

/// Feeds one token to a decoding request and ticks until its answer: the fed token's step and
/// every step the round accepted with it.
///
/// # Panics
/// If the engine refuses the token or a tick fails.
pub fn decode<W: Walk>(engine: &mut TickEngine<W>, id: RequestId, token: u32) -> Vec<StepOutput> {
    engine.feed(id, token).expect("the row is fed");
    loop {
        for event in engine.tick().expect("the tick") {
            if let TickEvent::Decode { id: got, steps } = event
                && got == id
            {
                return steps;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Region;

    fn walk(rows: usize, depth: usize) -> ShapeWalk {
        ShapeWalk::new(
            StateLayout::new(vec![Region::Positional {
                name: "shape",
                bytes_per_row: 4,
            }]),
            Limits::new(rows, 4_096).with_members(4),
            depth,
        )
    }

    /// A prompt longer than the limits' pass runs in the limits' chunks, and its first step is
    /// the model's pick after its last row.
    #[test]
    fn a_prompt_runs_in_the_limits_chunks() {
        let mut engine = TickEngine::new(walk(64, 0), 1).expect("the tick engine starts");
        assert_eq!(engine.config().chunk_rows, 64);
        let (id, first) = prompt(&mut engine, &[7u32; 150]);
        assert_eq!(engine.walk().chunks(), vec![64, 64, 22]);
        assert_eq!(first.next_token, Some(pick(7)));
        assert_eq!(engine.kept(id).and_then(|kept| kept.history), Some(150));
        assert!(
            engine.walk().verifies().is_empty(),
            "a prompt verifies nothing"
        );
    }

    /// A round at depth two verifies the anchor and both drafts, keeps the right one and
    /// rejects the wrong one.
    #[test]
    fn a_round_at_depth_two_keeps_one_draft() {
        let mut engine = TickEngine::new(walk(64, 2), 1).expect("the tick engine starts");
        let (id, first) = prompt(&mut engine, &[10, 11, 12, 13]);
        let anchor = first.next_token.expect("a token");
        let steps = decode(&mut engine, id, anchor);
        assert_eq!(steps.len(), 2, "the anchor and the accepted draft");
        assert_eq!(steps[0].next_token, Some(pick(anchor)));
        assert_eq!(steps[1].next_token, Some(pick(pick(anchor))));
        assert_eq!(engine.walk().verifies(), vec![vec![3]]);
        assert_eq!(
            engine.history(id).map(<[u32]>::to_vec),
            Some(vec![10, 11, 12, 13, anchor, pick(anchor)]),
            "the wrong draft never reaches the request"
        );
    }

    /// The arm's own chain, budget and wall replace the defaults: a drafter that is always
    /// right keeps every draft, and the budget rule sees the members and the depth.
    #[test]
    fn the_builders_state_an_arms_own_round() {
        struct Steep;
        impl StepWall for Steep {
            fn ms(&self, _seqs: usize, rows: usize) -> f64 {
                rows as f64
            }
        }

        let shaped = walk(64, 3)
            .with_code("arm")
            .with_right_picks()
            .with_budget(DraftBudget::OnePass(5))
            .with_wall(Box::new(Steep));
        assert_eq!(shaped.drafter().budget(1), 3, "the member's depth");
        assert_eq!(
            shaped.drafter().budget(3),
            2,
            "what three anchors leave of five rows"
        );
        assert!((shaped.drafter().wall().ms(1, 5) - 5.0).abs() < f64::EPSILON);
        assert_eq!(walk(64, 3).drafter().budget(3), 12, "anchors and drafts");

        let mut engine = TickEngine::new(shaped, 1).expect("the tick engine starts");
        let (id, first) = prompt(&mut engine, &[10, 11]);
        let anchor = first.next_token.expect("a token");
        let steps = decode(&mut engine, id, anchor);
        assert_eq!(steps.len(), 4, "the anchor and three right drafts");
        assert_eq!(engine.walk().verifies(), vec![vec![4]]);
    }
}

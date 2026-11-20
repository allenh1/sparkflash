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
//! The tick engine's drafting gate: a CPU walk with a drafter, so the engine's three decode
//! shapes - a plain row, a shared (multi-lane) round and a speculative round - are pinned
//! without a device.
//!
//! The walk is shaped like GLM's: one row a member, a verify of every member's anchor and
//! drafts, and a queue of the answers a round accepted. The drafter proposes `anchor + 1` and
//! then a deliberately wrong `anchor + 7`, so every speculative round has one accepted draft
//! and one rejected one; the model's own pick is always `token + 1`, so the accepted run is the
//! plain run and the rejected token never reaches the request.

use std::collections::VecDeque;
use std::sync::Mutex;

use sparkflash_model::backend::{BackendError, Sampling, StepOutput};
use sparkflash_model::verdict;
use sparkflash_runtime::{
    Draft, Drafter, HeadInput, HeadRow, Hidden, LaneState, Limits, NoVision, OutputConstraint,
    Pass, PassKind, Segment, StateKind, StateLayout, StepWall, TickConfig, TickEngine, TickEvent,
    Walk, WalkCapabilities, check_checkpoint, check_rewind,
};

/// The model's own pick after `token`: the plain path's next token.
fn pick(token: u32) -> u32 {
    token.wrapping_add(1)
}

/// One pass the walk was handed, for the shape assertions.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Shape {
    /// A plain token pass over this many members.
    TokenRows(usize),
    /// A prompt chunk of this many rows.
    Bulk(usize),
    /// A verify pass, one entry a member's rows.
    Verify(Vec<usize>),
}

/// A sequence is the tokens it forwarded and still holds, plus the answers a round queued.
#[derive(Debug, Default)]
struct Seq {
    rows: Vec<u32>,
    pending: VecDeque<(u32, StepOutput)>,
    checkpoint: Option<Vec<u32>>,
}

impl LaneState for Seq {
    fn position(&self) -> usize {
        self.rows.len()
    }

    fn checkpoint(&mut self, at: usize) -> Result<(), BackendError> {
        check_checkpoint("draft", at, self.rows.len())?;
        self.checkpoint = Some(self.rows.clone());
        Ok(())
    }

    fn rewind(&mut self, to: usize) -> Result<(), BackendError> {
        check_rewind("draft", to, self.rows.len())?;
        match (to, self.checkpoint.take()) {
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

/// The wall of a shared round: flat, so the row share gives the members their budget evenly.
struct Wall;

impl StepWall for Wall {
    fn ms(&self, _seqs: usize, _rows: usize) -> f64 {
        1.0
    }
}

static WALL: Wall = Wall;

/// The CPU walk the gate drives.
struct DraftWalk {
    layout: StateLayout,
    limits: Limits,
    drafter: DraftDrafter,
    /// Every pass the walk was handed, in order.
    shapes: Mutex<Vec<Shape>>,
}

impl DraftWalk {
    /// A walk whose prompt chunk is 64 rows, whose verify width is `members`, and whose
    /// drafter proposes `depth` drafts (`2` for the rejected-draft workload).
    fn new(members: usize, depth: usize) -> Self {
        Self {
            layout: StateLayout::new(vec![sparkflash_runtime::Region::Positional {
                name: "draft",
                bytes_per_row: 4,
            }]),
            limits: Limits::new(64, 4_096).with_members(members),
            drafter: DraftDrafter { depth, members },
            shapes: Mutex::new(Vec::new()),
        }
    }

    /// The passes so far.
    fn shapes(&self) -> Vec<Shape> {
        self.shapes.lock().expect("the pass record").clone()
    }

    fn clear(&self) {
        self.shapes.lock().expect("the pass record").clear();
    }
}

struct DraftDrafter {
    depth: usize,
    members: usize,
}

impl Walk for DraftWalk {
    type Seq = Seq;
    type Hidden = Hidden;
    type Drafter = DraftDrafter;
    type Vision = NoVision;

    const CAPABILITIES: WalkCapabilities = WalkCapabilities::all(StateKind::Positional);

    fn layout(&self) -> &StateLayout {
        &self.layout
    }

    fn open(&self) -> Result<Self::Seq, BackendError> {
        Ok(Seq::default())
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
        let shape = match pass.kind {
            PassKind::TokenRows => Shape::TokenRows(segments.len()),
            PassKind::Bulk { .. } | PassKind::Paired { .. } => {
                Shape::Bulk(segments.iter().map(|segment| segment.rows.len()).sum())
            }
            PassKind::Verify { .. } => {
                Shape::Verify(segments.iter().map(|segment| segment.rows.len()).collect())
            }
        };
        self.shapes.lock().expect("the pass record").push(shape);
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

impl Drafter<DraftWalk> for DraftDrafter {
    const PRESENT: bool = true;
    type Rows = ();

    fn wall(&self) -> &dyn StepWall {
        &WALL
    }

    fn open_round(&self, seq: &mut Seq, token: u32) -> Result<Draft, BackendError> {
        if let Some((front, _)) = seq.pending.front()
            && *front == token
        {
            let (_, step) = seq.pending.pop_front().expect("the front was read");
            return Ok(Draft::Held(step));
        }
        // The first draft is the model's own pick after the anchor; the second is not, so a
        // round always keeps one draft and stops at the next.
        let drafts: Vec<u32> = match self.depth {
            0 => Vec::new(),
            1 => vec![pick(token)],
            _ => vec![pick(token), token.wrapping_add(7)],
        };
        let confidence = (0..drafts.len()).map(|at| 0.9 - 0.1 * at as f64).collect();
        Ok(Draft::Chain { confidence, drafts })
    }

    fn verify(
        &self,
        walk: &DraftWalk,
        segments: &mut [Segment<'_, Seq>],
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
        seq: &mut Seq,
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

    /// One verify holds every member's anchor, and the rest of the width is shared among the
    /// members' chains.
    fn budget(&self, seqs: usize) -> usize {
        (seqs * self.depth).min(self.members.saturating_sub(seqs))
    }

    fn ready(&self, seq: &Seq, token: u32) -> bool {
        seq.pending
            .front()
            .is_some_and(|(front, _)| *front == token)
    }

    fn give_back(&self, seq: &mut Seq, to: usize) -> Result<(), BackendError> {
        seq.rows.truncate(to.min(seq.rows.len()));
        seq.pending.clear();
        Ok(())
    }

    fn speculation(&self, _seq: &Seq) -> sparkflash_model::backend::Speculation {
        sparkflash_model::backend::Speculation::default()
    }
}

/// Admits a prompt and ticks until its first step.
fn prompt(
    engine: &mut TickEngine<DraftWalk>,
    tokens: &[u32],
) -> (sparkflash_runtime::RequestId, StepOutput) {
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

/// Feeds one token and ticks until its answer, other events collected as they come.
fn decode(
    engine: &mut TickEngine<DraftWalk>,
    id: sparkflash_runtime::RequestId,
    token: u32,
) -> Vec<StepOutput> {
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

/// A drafting arm's config carries the verify width as the member cap, its drafter's depth,
/// and the shared shape.
#[test]
fn the_drafting_config_is_the_arms_profile() {
    let walk = DraftWalk::new(4, 2);
    let config = TickConfig::for_walk(&walk, 2);
    assert_eq!(config.rows_max, 64, "the prompt width is the arm's pass");
    assert_eq!(config.members, 4, "the verify width is the member cap");
    assert_eq!(config.chunk_rows, 64);
    assert_eq!(config.k, 2, "the drafter's depth");
    assert!(config.batch_spec.is_some());
    assert!(config.shared_rounds, "a drafting arm shares its verify");
    assert!(
        !config.decoder_skip,
        "only DeepSeek has decoder layers to skip"
    );
    assert!(!config.exchange, "a single process has no row exchange");
}

/// One plain row: a batch the engine plans `Plain` runs token rows, not a round.
#[test]
fn a_plain_batch_steps_one_row_a_member() {
    let walk = DraftWalk::new(4, 2);
    let mut config = TickConfig::for_walk(&walk, 2);
    config.shared_rounds = false;
    let mut engine = TickEngine::new(walk, 2)
        .expect("the engine starts")
        .with_config(config)
        .expect("the config")
        .with_trace();
    let (first, first_step) = prompt(&mut engine, &[10, 11, 12, 13]);
    let (second, second_step) = prompt(&mut engine, &[20, 21, 22, 23]);
    engine.walk().clear();
    engine
        .feed(first, first_step.next_token.expect("a token"))
        .expect("fed");
    engine
        .feed(second, second_step.next_token.expect("a token"))
        .expect("fed");
    let mut events = Vec::new();
    loop {
        events.extend(engine.tick().expect("the tick"));
        if events
            .iter()
            .filter(|event| matches!(event, TickEvent::Decode { .. }))
            .count()
            == 2
        {
            break;
        }
    }
    assert!(
        engine
            .trace()
            .expect("the trace")
            .render()
            .contains("plan=plain"),
        "two rows with joint rounds off run plain"
    );
    assert_eq!(
        engine.walk().shapes(),
        vec![Shape::TokenRows(2)],
        "a plain batch is one token pass over its members, with no verify"
    );
}

/// Several rows whose anchors fit the verify width run one shared round.
#[test]
fn a_fitting_batch_verifies_in_one_shared_round() {
    let walk = DraftWalk::new(4, 2);
    let mut engine = TickEngine::new(walk, 3)
        .expect("the engine starts")
        .with_trace();
    let mut ids = Vec::new();
    for (index, prompt_tokens) in [[10u32, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
        .into_iter()
        .enumerate()
    {
        let _ = index;
        ids.push(prompt(&mut engine, &prompt_tokens));
    }
    engine.walk().clear();
    for (id, step) in &ids {
        engine
            .feed(*id, step.next_token.expect("a token"))
            .expect("fed");
    }
    let mut events = Vec::new();
    loop {
        events.extend(engine.tick().expect("the tick"));
        if events
            .iter()
            .filter(|event| matches!(event, TickEvent::Decode { .. }))
            .count()
            == 3
        {
            break;
        }
    }
    assert!(
        engine
            .trace()
            .expect("the trace")
            .render()
            .contains("plan=shared"),
        "three rows in one verify are the shared plan"
    );
    let verifies: Vec<Vec<usize>> = engine
        .walk()
        .shapes()
        .into_iter()
        .filter_map(|shape| match shape {
            Shape::Verify(rows) => Some(rows),
            _ => None,
        })
        .collect();
    assert_eq!(verifies.len(), 1, "one shared verify, not one a lane");
    assert_eq!(verifies[0].len(), 3, "every member's chain is in the pass");
    assert!(!verifies[0].is_empty());
}

/// One row runs one speculative round: its draft is rejected, its accepted rows reach the
/// request and the rejected token never does.
#[test]
fn a_speculative_round_keeps_its_accepted_rows_only() {
    let walk = DraftWalk::new(4, 2);
    let mut engine = TickEngine::new(walk, 1)
        .expect("the engine starts")
        .with_trace();
    let (id, first) = prompt(&mut engine, &[10, 11, 12, 13]);
    let anchor = first.next_token.expect("a token");
    engine.walk().clear();
    let steps = decode(&mut engine, id, anchor);
    assert_eq!(steps.len(), 2, "the anchor and the accepted draft");
    assert_eq!(steps[0].next_token, Some(pick(anchor)));
    assert_eq!(steps[1].next_token, Some(pick(pick(anchor))));
    assert_eq!(
        engine.history(id).map(<[u32]>::to_vec),
        Some(vec![10, 11, 12, 13, anchor, pick(anchor)]),
        "the accepted draft reaches the history"
    );
    assert!(
        !engine
            .history(id)
            .expect("the history")
            .contains(&anchor.wrapping_add(7)),
        "the rejected draft never reaches the request"
    );
    let verifies: Vec<Vec<usize>> = engine
        .walk()
        .shapes()
        .into_iter()
        .filter_map(|shape| match shape {
            Shape::Verify(rows) => Some(rows),
            _ => None,
        })
        .collect();
    assert_eq!(
        verifies,
        vec![vec![3]],
        "the anchor and both drafts verified"
    );
    assert!(
        engine
            .trace()
            .expect("the trace")
            .render()
            .contains("plan=spec"),
        "one row takes the speculative plan"
    );
}

/// A tick-ahead plan is a decision, not an execution: a plan that names no round is not used
/// to skip the round the fed row needs.
#[test]
fn a_stale_tick_ahead_plan_does_not_skip_a_round() {
    let walk = DraftWalk::new(4, 2);
    let mut engine = TickEngine::new(walk, 1)
        .expect("the engine starts")
        .with_trace();
    let (id, first) = prompt(&mut engine, &[10, 11, 12, 13]);
    // The prompt's own hook left a plan for no decode rows; replace it with a stale one for the
    // one row the caller is about to feed. It names no round, and must not be executed as none.
    engine.with_ahead(1, sparkflash_runtime::tick::DecodePlan::None);
    engine.walk().clear();
    let steps = decode(&mut engine, id, first.next_token.expect("a token"));
    assert_eq!(
        steps.len(),
        2,
        "the round ran; the plan did not stand in for it"
    );
    assert!(
        engine
            .trace()
            .expect("the trace")
            .render()
            .contains("plan=spec"),
        "the engine re-decided the plan rather than executing none"
    );
    assert!(
        engine
            .walk()
            .shapes()
            .iter()
            .any(|shape| matches!(shape, Shape::Verify(_))),
        "the verify walked the member's chain"
    );
}

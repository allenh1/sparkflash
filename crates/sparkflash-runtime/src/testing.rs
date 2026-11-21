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
//! [`ArmProfile`] is what an arm states of itself for those tests, and the `check_*` functions
//! are the tests: [`crate::conformance!`] emits one `#[test]` a function, so an arm writes its
//! profile once and no test body.
//!
//! The module is compiled into the library, as [`mod@crate::conformance`] is, so every arm's test
//! module can name it. Nothing here is on a serving path.

use std::collections::VecDeque;
use std::sync::Mutex;

use sparkflash_model::backend::{BackendError, Sampling, Speculation, StepOutput};
use sparkflash_model::ledger::{Strategy, ledger_for_arm};
use sparkflash_model::verdict;

use crate::chunk::ChunkPolicy;
use crate::conformance::{ConformanceCheck, WalkFixture, checks_for};
use crate::engine::{RequestId, TickConfig, TickEngine, TickEvent};
use crate::layout::{LaneState, Limits, StateKind, StateLayout, check_checkpoint, check_rewind};
use crate::runtime::{Runtime, RuntimeSlot, ledger_for};
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
                if let PassKind::Bulk { rows: stated } = pass.kind
                    && stated != rows
                {
                    return Err(BackendError::new(
                        self.code,
                        format!("a bulk pass states {stated} rows and holds {rows}"),
                    ));
                }
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

/// One prompt an arm states: the limits its walk has while the prompt runs, the prompt's length
/// and the chunks it runs in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptCase {
    /// The walk's limits for this prompt: the arm's default, or what a profile switch gives.
    pub limits: Limits,
    /// The prompt's length in tokens.
    pub tokens: usize,
    /// The chunks the prompt runs in, in order.
    pub chunks: Vec<usize>,
}

/// What an arm states of itself for the generated profile tests: its ledger name, its state
/// layout, the limits its walk gives the runtime, its verify budget, its step wall and the
/// chunks its prompts run in.
///
/// Every `check_*` function of this module reads one, and [`crate::conformance!`] emits a test a
/// function. The profile holds the arm's numbers and no arithmetic: the checks drive a
/// [`ShapeWalk`] built from it ([`Self::walk`]).
pub struct ArmProfile {
    ledger: &'static str,
    layout: StateLayout,
    limits: Limits,
    code: &'static str,
    depth: usize,
    budget: DraftBudget,
    wall: Option<Box<dyn Fn() -> Box<dyn StepWall + Send + Sync>>>,
    prompts: Vec<PromptCase>,
}

impl ArmProfile {
    /// An arm named `ledger` in [`sparkflash_model::ledger::ARM_LEDGERS`], with `layout` as its
    /// sequence state and `limits` as the limits its walk states by default. It drafts nothing
    /// until [`Self::drafting`] says so.
    #[must_use]
    pub fn new(ledger: &'static str, layout: StateLayout, limits: Limits) -> Self {
        Self {
            ledger,
            layout,
            limits,
            code: "shape",
            depth: 0,
            budget: DraftBudget::AnchorsAndDrafts,
            wall: None,
            prompts: Vec::new(),
        }
    }

    /// The same profile whose shape walks return errors under `code`.
    #[must_use]
    pub fn with_code(mut self, code: &'static str) -> Self {
        self.code = code;
        self
    }

    /// The same profile for an arm whose drafter proposes `depth` drafts a member by default
    /// under the `budget` rule.
    #[must_use]
    pub fn drafting(mut self, depth: usize, budget: DraftBudget) -> Self {
        self.depth = depth;
        self.budget = budget;
        self
    }

    /// The same profile with the arm's step wall for the row share.
    #[must_use]
    pub fn with_wall<S>(mut self, wall: S) -> Self
    where
        S: StepWall + Copy + Send + Sync + 'static,
    {
        self.wall = Some(Box::new(move || Box::new(wall)));
        self
    }

    /// The same profile with one more stated prompt: under `limits`, `tokens` tokens run in
    /// `chunks`.
    #[must_use]
    pub fn prompt(mut self, limits: Limits, tokens: usize, chunks: &[usize]) -> Self {
        self.prompts.push(PromptCase {
            limits,
            tokens,
            chunks: chunks.to_vec(),
        });
        self
    }

    /// The limits the arm's walk states by default.
    #[must_use]
    pub fn limits(&self) -> Limits {
        self.limits
    }

    /// The prompts the arm states.
    #[must_use]
    pub fn prompts(&self) -> &[PromptCase] {
        &self.prompts
    }

    /// A shape walk at `limits` whose drafter proposes `depth` drafts a member, with the arm's
    /// layout, error code, budget rule and wall.
    #[must_use]
    pub fn walk(&self, limits: Limits, depth: usize) -> ShapeWalk {
        let walk = ShapeWalk::new(self.layout.clone(), limits, depth)
            .with_code(self.code)
            .with_budget(self.budget);
        match &self.wall {
            Some(wall) => walk.with_wall(wall()),
            None => walk,
        }
    }

    /// Every limits the profile states: the default and each prompt's, once each.
    fn stated_limits(&self) -> Vec<Limits> {
        let mut stated = vec![self.limits];
        for case in &self.prompts {
            if !stated.contains(&case.limits) {
                stated.push(case.limits);
            }
        }
        stated
    }

    /// The members a joint round of the checks steps: at most `most`, and no more than the
    /// arm's verify takes.
    fn round_members(&self, most: usize) -> usize {
        most.min(self.limits.members.max(1))
    }
}

/// The four compute rows the walk declares are the arm's static declaration, so the boot ledger
/// and `sparkflash arms --ledger` cannot drift.
///
/// # Panics
/// If the arm has no ledger entry, or a row the walk declares differs from the entry's.
pub fn check_compute_rows<W: Walk>(profile: &ArmProfile) {
    let arm = profile.ledger;
    let declared = ledger_for_arm(arm).unwrap_or_else(|| panic!("{arm} has no ledger entry"));
    let capabilities = W::CAPABILITIES.compute;
    for (strategy, stated) in [
        (Strategy::TensorCorePrompt, capabilities.prompt),
        (Strategy::TensorCoreDecode, capabilities.token),
        (Strategy::StoredArithmetic, capabilities.stored),
        (Strategy::Rewind, capabilities.rewind),
    ] {
        let from_table = declared
            .rows
            .iter()
            .find(|(row, _)| *row == strategy)
            .map(|(_, state)| *state)
            .unwrap_or_else(|| panic!("{arm} does not state {}", strategy.name()));
        assert_eq!(stated, from_table, "{arm}: {} drifted", strategy.name());
    }
}

/// The generated checks drive the path the arm serves on, the tick engine, and the form of
/// [`crate::conformance!`] the arm invoked is the one its drafter calls for: `rounds` for a walk
/// that declares a drafter, and only for one. A drafting arm claims drafted = plain.
///
/// # Panics
/// If the fixture is not on the tick engine, or the walk's drafter, the profile's depth and
/// the invoked form disagree.
pub fn check_serving_path<W: WalkFixture>(profile: &ArmProfile, rounds: bool) {
    let arm = profile.ledger;
    assert!(
        W::tick_engine(),
        "{arm}: a profiled arm serves on the tick engine, and its fixture says it does not"
    );
    assert_eq!(
        W::Drafter::PRESENT,
        rounds,
        "{arm}: a walk that declares a drafter invokes `conformance!` with `rounds`, and only \
         that walk does"
    );
    assert_eq!(
        profile.depth > 0,
        rounds,
        "{arm}: the profile states a draft depth exactly when the walk drafts"
    );
    assert_eq!(
        checks_for::<W>().contains(&ConformanceCheck::DraftedEqualsPlain),
        rounds,
        "{arm}: a drafting arm claims drafted = plain"
    );
}

/// The runtime's ledger states every row the walk declares: no row is left for the arm to
/// state, and the drafting rows follow the walk's drafter.
///
/// # Panics
/// If a row prints "does not state it", or the drafting row disagrees with the walk's drafter.
pub fn check_ledger<W: Walk>(profile: &ArmProfile) {
    let arm = profile.ledger;
    let ledger = ledger_for::<W>();
    let line = ledger.line();
    assert!(
        !line.contains("does not state it"),
        "{arm}: the runtime derives every row the walk states: {line}"
    );
    assert_eq!(
        matches!(ledger.mode(Strategy::Drafting), crate::Mode::Native(_)),
        W::Drafter::PRESENT,
        "{arm}: the drafting row follows the walk's drafter: {line}"
    );
    assert_eq!(
        matches!(ledger.mode(Strategy::Runs), crate::Mode::Native(_)),
        W::Drafter::PRESENT,
        "{arm}: an accepted run reaches the request exactly when the walk drafts: {line}"
    );
    let joint = crate::rounds::joint_rounds().unwrap_or(true);
    assert_eq!(
        matches!(ledger.mode(Strategy::JointRounds), crate::Mode::Native(_)),
        W::Drafter::PRESENT && joint,
        "{arm}: the joint row follows the drafter and the switch: {line}"
    );
}

/// The tick engine's config at the arm's limits is the arm's profile: its pass rows as the
/// prompt chunk on every prompt path, its member cap as the verify width, no decoder-layer skip
/// and no row exchange at one rank. With `rounds`, the drafter's own depth under the arm's
/// budget, shared rounds on, and the runtime's joint-round switch as the engine's.
///
/// # Panics
/// If a field of the config is not what the profile states.
pub fn check_tick_config(profile: &ArmProfile, rounds: bool) {
    let arm = profile.ledger;
    for limits in profile.stated_limits() {
        let walk = profile.walk(limits, profile.depth);
        let config = TickConfig::for_walk(&walk, limits.members.max(1));
        let chunk = ChunkPolicy::for_limits(limits).rows();
        assert_eq!(
            chunk, limits.rows,
            "{arm}: the walk's {} rows a pass must be a ChunkPolicy size, or the runtime moves \
             the arm's chunk",
            limits.rows
        );
        assert_eq!(config.rows_max, limits.rows, "{arm}: the widest pass");
        assert_eq!(config.chunk_rows, chunk, "{arm}: a prompt pass");
        assert_eq!(config.exact_tick_rows, chunk, "{arm}: the exact tick");
        assert_eq!(config.busy_chunk, chunk, "{arm}: a busy tick's chunk");
        assert_eq!(config.bulk_cap, Some(chunk), "{arm}: the bulk path's cap");
        assert_eq!(
            config.members,
            limits.members.max(1),
            "{arm}: the verify width is the member cap"
        );
        assert!(!config.decoder_skip, "{arm} has no decoder layers to skip");
        assert!(!config.exchange, "{arm}: one rank has no row exchange");
    }
    if !rounds {
        return;
    }
    let lanes = profile.limits.members.max(1);
    let walk = profile.walk(profile.limits, profile.depth);
    let config = TickConfig::for_walk(&walk, lanes);
    assert_eq!(
        config.k,
        profile.budget.rows(1, profile.depth),
        "{arm}: one member's budget at the drafter's own depth"
    );
    assert!(config.batch_spec.is_some(), "{arm} drafts");
    assert!(
        config.shared_rounds,
        "{arm}: joint rounds are on by default"
    );
    assert!(
        config.share_floor.abs() < f64::EPSILON,
        "{arm} states no confidence floor"
    );
    // The engine's shared-round switch is the runtime's, not a second rule.
    for joint in [true, false] {
        let runtime = Runtime::new(profile.walk(profile.limits, profile.depth), 0)
            .expect("the runtime")
            .with_joint_rounds(joint);
        assert_eq!(
            TickConfig::for_runtime(&runtime, lanes).shared_rounds,
            joint,
            "{arm}: joint rounds {joint} is the engine's shared verify"
        );
    }
}

/// The chunks one prompt runs in on each of the runtime's three prompt paths: the tick engine's
/// exact path, its bulk path under a group's row exchange, and the runtime's own prompt loop.
fn prompt_chunks(profile: &ArmProfile, limits: Limits, tokens: usize) -> [Vec<usize>; 3] {
    let prompt_tokens = vec![7u32; tokens];
    let mut engine = TickEngine::new(profile.walk(limits, 0), 1).expect("the tick engine starts");
    let _ = prompt(&mut engine, &prompt_tokens);
    let exact = engine.walk().chunks();

    let walk = profile.walk(limits, 0);
    let mut group = TickConfig::for_walk(&walk, 1);
    group.exchange = true;
    let mut engine = TickEngine::new(walk, 1)
        .expect("the tick engine starts")
        .with_config(group)
        .expect("the group config");
    let _ = prompt(&mut engine, &prompt_tokens);
    let bulk = engine.walk().chunks();

    let runtime = Runtime::new(profile.walk(limits, 0), 0).expect("the runtime starts");
    let mut slot = runtime.open_slot().expect("a slot");
    runtime
        .prefill_slot(&mut slot, &prompt_tokens)
        .expect("the prefill");
    [exact, bulk, runtime.walk().chunks()]
}

/// Every prompt the arm states runs in the chunks the arm states, on the tick engine's exact
/// path, on a group's bulk path and in the runtime's own prompt loop: the three are one split,
/// and it is the split the arm ran before the runtime drove it.
///
/// # Panics
/// If the profile states no prompt, or a path runs a stated prompt in other chunks.
pub fn check_prompt_chunks(profile: &ArmProfile) {
    let arm = profile.ledger;
    assert!(
        !profile.prompts.is_empty(),
        "{arm}: the profile states no prompt; state one longer than a pass"
    );
    for case in &profile.prompts {
        let [exact, bulk, looped] = prompt_chunks(profile, case.limits, case.tokens);
        assert_eq!(
            exact, case.chunks,
            "{arm}: a {}-row prompt on the tick engine at {} rows a pass",
            case.tokens, case.limits.rows
        );
        assert_eq!(
            bulk, case.chunks,
            "{arm}: a group's bulk chunks are the single process's chunks"
        );
        assert_eq!(
            looped, case.chunks,
            "{arm}: the runtime's prompt loop runs the engine's chunks"
        );
    }
}

/// A prompt that fits one pass runs whole: one row, one row short of the pass and the pass
/// itself are each one chunk, at every limits the arm states.
///
/// # Panics
/// If a prompt no longer than a pass is split.
pub fn check_whole_prompts(profile: &ArmProfile) {
    let arm = profile.ledger;
    for limits in profile.stated_limits() {
        for tokens in [1, limits.rows.saturating_sub(1), limits.rows] {
            if tokens == 0 || tokens > limits.context {
                continue;
            }
            for chunks in prompt_chunks(profile, limits, tokens) {
                assert_eq!(
                    chunks,
                    vec![tokens],
                    "{arm}: a {tokens}-row prompt is one pass of {} rows",
                    limits.rows
                );
            }
        }
    }
}

/// The budget fits the verify: for every member count the engine may step, the anchors and the
/// drafts the rule hands out are no more than one verify pass, and the member cap is no wider
/// than that pass.
///
/// # Panics
/// If the arm's rule can hand out more rows than its verify holds.
pub fn check_budget(profile: &ArmProfile) {
    let arm = profile.ledger;
    let members = profile.limits.members.max(1);
    match profile.budget {
        DraftBudget::AnchorsAndDrafts => {
            for seqs in 0..=members {
                for depth in 0..=profile.depth {
                    assert_eq!(
                        profile.budget.widest(seqs, depth),
                        seqs * (1 + depth),
                        "{arm}: every member's anchor and its depth"
                    );
                }
            }
        }
        DraftBudget::OnePass(rows) => {
            assert!(
                members <= rows,
                "{arm}: {members} members do not fit a verify of {rows} rows"
            );
            for seqs in 1..=rows {
                for depth in 0..rows.max(profile.depth + 1) {
                    let drafts = profile.budget.rows(seqs, depth);
                    assert!(
                        profile.budget.widest(seqs, depth) <= rows,
                        "{arm}: {seqs} member(s) at depth {depth} budget {drafts} drafts"
                    );
                    assert_eq!(drafts, (seqs * depth).min(rows.saturating_sub(seqs)));
                }
            }
        }
    }
}

/// Several decoding lanes verify in one shared round: three lanes take the engine's shared
/// plan, one verify holds every member's chain, the verify is no wider than the arm's budget
/// allows and the row share drafts at least one row. It holds whether the drafter is right
/// once and then wrong, or right all the way.
///
/// # Panics
/// If the lanes do not share one verify, or the verify is wider than the arm's.
pub fn check_joint_round(profile: &ArmProfile) {
    let arm = profile.ledger;
    let lanes = profile.round_members(3);
    assert!(
        lanes >= 2,
        "{arm}: a verify of one member has no joint round to keep"
    );
    for right in [false, true] {
        let walk = profile.walk(profile.limits, profile.depth);
        let walk = if right { walk.with_right_picks() } else { walk };
        let mut engine = TickEngine::new(walk, lanes)
            .expect("the tick engine starts")
            .with_trace();
        let mut fed = Vec::new();
        for at in 0..lanes {
            let at = u32::try_from(at).expect("a few lanes");
            let tokens: Vec<u32> = (0..7).map(|index| 1 + at * 13 + index).collect();
            let (id, step) = prompt(&mut engine, &tokens);
            assert_eq!(engine.kept(id).and_then(|kept| kept.history), Some(7));
            fed.push((id, step.next_token.expect("a token")));
        }
        for (id, token) in &fed {
            engine.feed(*id, *token).expect("the row is fed");
        }
        let mut answered = 0;
        while answered < fed.len() {
            for event in engine.tick().expect("the shared tick") {
                if matches!(event, TickEvent::Decode { .. }) {
                    answered += 1;
                }
            }
        }
        let trace = engine.trace().expect("the trace").render();
        assert!(
            trace.contains("plan=shared"),
            "{arm}: {lanes} lanes share one verify: {trace}"
        );
        let verifies = engine.walk().verifies();
        assert_eq!(
            verifies.len(),
            1,
            "{arm}: one shared verify, not one a lane: {verifies:?}"
        );
        assert_eq!(
            verifies[0].len(),
            lanes,
            "{arm}: every member's chain is in the pass: {verifies:?}"
        );
        let rows: usize = verifies[0].iter().sum();
        let widest = profile.budget.widest(lanes, profile.depth);
        assert!(
            rows <= widest,
            "{arm}: the anchors and the drafts they share fit {widest} rows: {verifies:?}"
        );
        if widest > lanes {
            assert!(
                rows > lanes,
                "{arm}: the row share drafts at least one row: {verifies:?}"
            );
        }
    }
}

/// One decoding row runs one speculative round at a depth of two: the anchor and both drafts
/// verify in one pass, the draft the drafter gets wrong is rejected, and only the accepted rows
/// reach the request.
///
/// # Panics
/// If the arm's budget has no room for two drafts of one member, or the round keeps a row it
/// rejected.
pub fn check_speculative_round(profile: &ArmProfile) {
    let arm = profile.ledger;
    assert!(
        profile.budget.rows(1, 2) >= 2,
        "{arm}: the budget has no room for one member's two drafts"
    );
    let mut engine = TickEngine::new(profile.walk(profile.limits, 2), 1)
        .expect("the tick engine starts")
        .with_trace();
    let (id, first) = prompt(&mut engine, &[10, 11, 12, 13]);
    let anchor = first.next_token.expect("a token");
    let steps = decode(&mut engine, id, anchor);
    assert_eq!(steps.len(), 2, "{arm}: the anchor and the accepted draft");
    assert_eq!(steps[0].next_token, Some(pick(anchor)));
    assert_eq!(steps[1].next_token, Some(pick(pick(anchor))));
    assert_eq!(
        engine.history(id).map(<[u32]>::to_vec),
        Some(vec![10, 11, 12, 13, anchor, pick(anchor)]),
        "{arm}: the accepted draft reaches the history"
    );
    assert!(
        !engine
            .history(id)
            .expect("the history")
            .contains(&wrong(anchor)),
        "{arm}: the rejected draft never reaches the request"
    );
    assert_eq!(
        engine.walk().verifies(),
        vec![vec![3]],
        "{arm}: the anchor and both drafts verified in one pass"
    );
    assert!(
        engine
            .trace()
            .expect("the trace")
            .render()
            .contains("plan=spec"),
        "{arm}: one row takes the speculative plan"
    );
}

/// A runtime over the profile's walk at the drafter's own depth, its drafter always right.
fn round_runtime(profile: &ArmProfile) -> Runtime<ShapeWalk> {
    Runtime::new(
        profile
            .walk(profile.limits, profile.depth)
            .with_right_picks(),
        0,
    )
    .expect("the shape runtime starts")
}

/// `count` slots of `runtime`, each past a one-token prompt, each with the token it decodes.
fn round_members(runtime: &Runtime<ShapeWalk>, count: usize) -> Vec<(RuntimeSlot<ShapeWalk>, u32)> {
    (0..u32::try_from(count).expect("a few members"))
        .map(|at| {
            let mut slot = runtime.open_slot().expect("a slot");
            runtime
                .prefill_slot(&mut slot, &[at + 1])
                .expect("the prompt");
            (slot, at + 11)
        })
        .collect()
}

/// The rows one member verifies alone at the drafter's own depth: its anchor and its chain, as
/// far as the budget lets one member draft.
fn lone_rows(profile: &ArmProfile) -> usize {
    1 + profile
        .depth
        .min(profile.budget.widest(1, profile.depth).saturating_sub(1))
}

/// One member keeps the drafter's own depth in one verify pass of the runtime's round.
///
/// # Panics
/// If the lone round verifies in more than one pass or at another depth.
pub fn check_lone_round(profile: &ArmProfile) {
    let runtime = round_runtime(profile);
    let mut members = round_members(&runtime, 1);
    runtime.step_slots(&mut members).expect("the round");
    assert_eq!(
        runtime.walk().verifies(),
        vec![vec![lone_rows(profile)]],
        "{}: one member, its anchor and its own depth",
        profile.ledger
    );
}

/// The members of one runtime round, four or the arm's cap, share one verify pass: every
/// member steps, and their anchors and drafts fit the arm's verify.
///
/// # Panics
/// If the members verify in more than one pass, one of them does not step, or the pass is
/// wider than the budget allows.
pub fn check_shared_round(profile: &ArmProfile) {
    let arm = profile.ledger;
    let count = profile.round_members(4);
    let runtime = round_runtime(profile);
    let mut members = round_members(&runtime, count);
    runtime.step_slots(&mut members).expect("the joint round");
    let passes = runtime.walk().verifies();
    assert_eq!(
        passes.len(),
        1,
        "{arm}: one verify pass for {count} members: {passes:?}"
    );
    assert_eq!(passes[0].len(), count);
    assert!(
        passes[0].iter().all(|rows| *rows >= 1),
        "{arm}: every member steps: {passes:?}"
    );
    let rows: usize = passes[0].iter().sum();
    let widest = profile.budget.widest(count, profile.depth);
    assert!(
        rows <= widest,
        "{arm}: the shared verify is no wider than {widest} rows: {passes:?}"
    );
}

/// With joint rounds off, each lane's round walks alone at the drafter's own depth.
///
/// # Panics
/// If two members share a pass with the switch off.
pub fn check_lone_rounds_with_joint_off(profile: &ArmProfile) {
    let runtime = round_runtime(profile).with_joint_rounds(false);
    let mut members = round_members(&runtime, 2);
    runtime.step_slots(&mut members).expect("two lone rounds");
    let alone = lone_rows(profile);
    assert_eq!(
        runtime.walk().verifies(),
        vec![vec![alone], vec![alone]],
        "{}: two lanes, one verify each",
        profile.ledger
    );
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

    /// A profile at toy widths: 64-row passes, a four-row verify.
    fn profile() -> ArmProfile {
        let limits = Limits::new(64, 4_096).with_members(4);
        ArmProfile::new(
            "toy",
            StateLayout::new(vec![Region::Positional {
                name: "shape",
                bytes_per_row: 4,
            }]),
            limits,
        )
        .with_code("toy")
        .prompt(limits, 150, &[64, 64, 22])
        .prompt(Limits::new(256, 4_096).with_members(4), 300, &[256, 44])
    }

    /// Every check that reads the profile alone holds for a profile that states the truth,
    /// whichever budget rule it states.
    #[test]
    fn a_true_profile_passes_the_checks_that_need_no_arm() {
        let plain = profile();
        check_tick_config(&plain, false);
        check_prompt_chunks(&plain);
        check_whole_prompts(&plain);
        for budget in [DraftBudget::AnchorsAndDrafts, DraftBudget::OnePass(4)] {
            let drafting = profile().drafting(2, budget);
            check_tick_config(&drafting, true);
            check_prompt_chunks(&drafting);
            check_budget(&drafting);
            check_joint_round(&drafting);
            check_speculative_round(&drafting);
            check_lone_round(&drafting);
            check_shared_round(&drafting);
            check_lone_rounds_with_joint_off(&drafting);
        }
    }

    /// A profile's walk carries the profile's code, budget and wall.
    #[test]
    fn a_profiles_walk_is_the_arms_shape() {
        #[derive(Clone, Copy)]
        struct Steep;
        impl StepWall for Steep {
            fn ms(&self, _seqs: usize, rows: usize) -> f64 {
                rows as f64
            }
        }

        let stated = profile()
            .drafting(3, DraftBudget::OnePass(4))
            .with_wall(Steep);
        let shaped = stated.walk(stated.limits(), 3);
        assert_eq!(shaped.limits(), stated.limits());
        assert_eq!(shaped.drafter().budget(2), 2, "what two anchors leave");
        assert!((shaped.drafter().wall().ms(1, 5) - 5.0).abs() < f64::EPSILON);
        assert_eq!(stated.prompts().len(), 2);
        let error = shaped
            .open()
            .expect("a sequence")
            .rewind(1)
            .expect_err("a rewind past the sequence is refused");
        assert_eq!(error.code(), "toy");
    }

    #[test]
    #[should_panic(expected = "a 150-row prompt on the tick engine at 64 rows a pass")]
    fn a_prompt_stated_in_other_chunks_fails() {
        let limits = Limits::new(64, 4_096);
        check_prompt_chunks(&profile().prompt(limits, 150, &[128, 22]));
    }

    #[test]
    #[should_panic(expected = "must be a ChunkPolicy size")]
    fn a_pass_the_policy_would_move_fails() {
        let limits = Limits::new(100, 4_096);
        check_tick_config(&ArmProfile::new("toy", StateLayout::empty(), limits), false);
    }

    #[test]
    #[should_panic(expected = "6 members do not fit a verify of 4 rows")]
    fn a_member_cap_wider_than_the_verify_fails() {
        let limits = Limits::new(64, 4_096).with_members(6);
        check_budget(
            &ArmProfile::new("toy", StateLayout::empty(), limits)
                .drafting(2, DraftBudget::OnePass(4)),
        );
    }

    #[test]
    #[should_panic(expected = "the profile states no prompt")]
    fn a_profile_with_no_prompt_fails_the_chunk_check() {
        check_prompt_chunks(&ArmProfile::new(
            "toy",
            StateLayout::empty(),
            Limits::new(64, 4_096),
        ));
    }
}

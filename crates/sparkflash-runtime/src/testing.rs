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

use sparkflash_model::architecture::WeightFormat;
use sparkflash_model::backend::{BackendError, Sampling, Speculation, StepOutput};
use sparkflash_model::ledger::{Strategy, ledger_for_arm};
use sparkflash_model::verdict;

use crate::chunk::ChunkPolicy;
use crate::conformance::FormatRule;
use crate::engine::{RequestId, TickConfig, TickEngine, TickEvent};
use crate::layout::{LaneState, Limits, StateKind, StateLayout, check_checkpoint, check_rewind};
use crate::runtime::Runtime;
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

    /// The same walk whose drafter proposes the model's pick and then a token it never picks: a
    /// round keeps the first draft and rejects the rest, which is the default.
    #[must_use]
    pub fn with_wrong_picks(mut self) -> Self {
        self.drafter.chain = Chain::PickThenWrong;
        self
    }

    /// The drafts the walk's drafter proposes past an anchor, before the budget shares them.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.drafter.depth
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

/// The stored format name a fixture's class uses.
#[must_use]
pub fn format_rule(format: WeightFormat) -> FormatRule {
    let name = match format {
        WeightFormat::Bf16 => "BF16",
        WeightFormat::Fp8Tile { .. } => "FP8 E4M3",
        WeightFormat::MxFp8 { .. } => "MX FP8",
        WeightFormat::NvFp4 { .. } | WeightFormat::MxFp4 { .. } => "NVFP4",
    };
    FormatRule::stored(name)
}

/// The bytes a token pass reads and the floor beneath them, for a fixture whose tensor plan
/// states `total` stored bytes and `expert` bytes of routed-expert weight: every non-expert
/// weight is read whatever a token routes, and the experts pay their `top_k` share.
#[must_use]
pub fn token_bytes(total: u64, expert: u64, experts: usize, top_k: usize) -> (u64, u64) {
    let floor = total.saturating_sub(expert);
    let share = if experts == 0 {
        0
    } else {
        expert.saturating_mul(top_k as u64) / experts as u64
    };
    (floor.saturating_add(share), floor)
}

/// Panics unless `arm`'s four declared compute rows are its arms-table row, so the boot ledger
/// and **sparkflash arms --ledger** cannot drift.
pub fn check_compute_rows<W: Walk + ?Sized>(arm: &str) {
    let declared = ledger_for_arm(arm).unwrap_or_else(|| panic!("{arm}'s declaration"));
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
            .unwrap_or_else(|| panic!("{arm} states no {}", strategy.name()));
        assert_eq!(stated, from_table, "{} drifted", strategy.name());
    }
}

/// Panics unless the tick config the runtime derives from `walk` is the arm's profile: its own
/// pass and verify widths, its drafter's depth and floor, and the runtime's joint-round
/// switch, with no decoder-layer skip and no row exchange.
pub fn check_tick_config<W: Walk>(walk: W, joint: bool) {
    let limits = walk.limits();
    let lanes = limits.members.max(1);
    let chunk = ChunkPolicy::for_limits(limits).rows();
    let drafting = W::Drafter::PRESENT && walk.drafter().drafts();
    let config = TickConfig::for_walk(&walk, lanes);
    assert_eq!(config.lanes, lanes, "the lanes the walk holds");
    assert_eq!(
        config.rows_max,
        limits.rows.max(1),
        "a prompt chunk is the walk's pass"
    );
    assert_eq!(
        config.chunk_rows,
        limits.rows.max(1),
        "the engine's chunk is the walk's pass"
    );
    assert_eq!(config.exact_tick_rows, chunk, "the exact tick is the pass");
    assert_eq!(config.busy_chunk, chunk, "a busy tick is the pass");
    assert_eq!(config.bulk_cap, Some(chunk), "the bulk cap is the pass");
    assert_eq!(
        config.members,
        limits.members.max(1),
        "the verify width is the member cap"
    );
    assert_eq!(
        config.k,
        if drafting {
            walk.drafter().budget(1)
        } else {
            0
        },
        "the chain cap is the drafter's own depth"
    );
    assert_eq!(
        config.batch_spec.is_some(),
        drafting,
        "only a drafting arm takes the speculative path"
    );
    assert_eq!(
        config.shared_rounds, drafting,
        "a drafting arm shares one verify"
    );
    assert_eq!(
        config.share_floor,
        if drafting {
            walk.drafter().floor()
        } else {
            0.2
        },
        "the floor is the drafter's"
    );
    assert!(!config.decoder_skip, "an arm states no decoder-layer skip");
    assert!(!config.exchange, "TP=1 has no row exchange");
    let runtime = Runtime::new(walk, 0)
        .expect("the runtime starts")
        .with_joint_rounds(joint);
    assert_eq!(
        TickConfig::for_runtime(&runtime, lanes).shared_rounds,
        drafting && joint,
        "the shared-round switch is the runtime's"
    );
}

/// The chunks `rows` rows split into at `width` rows a pass.
#[must_use]
pub fn chunks_at(rows: usize, width: usize) -> Vec<usize> {
    let width = width.max(1);
    let mut chunks = vec![width; rows / width];
    if !rows.is_multiple_of(width) {
        chunks.push(rows % width);
    }
    chunks
}

/// The prompt chunks the tick engine over `walk` hands a `rows`-row prompt, with the group's
/// row exchange on when `exchange`.
#[must_use]
pub fn prompt_chunks(walk: ShapeWalk, rows: usize, exchange: bool) -> Vec<usize> {
    let mut config = TickConfig::for_walk(&walk, 1);
    config.exchange = exchange;
    let mut engine = TickEngine::new(walk, 1)
        .expect("the tick engine starts")
        .with_config(config)
        .expect("the walk's own config");
    prompt(&mut engine, &vec![7u32; rows]);
    engine.walk().chunks()
}

/// Panics unless the tick engine over `walk` shares one verify across `lanes` members inside the
/// drafter's budget, or with `joint` off runs plain rows with no verify.
pub fn check_joint_round(walk: ShapeWalk, lanes: usize, joint: bool) {
    // The allocation is the property here: a drafter that is always right proposes its whole
    // depth, so a lone lane's verify is its anchor and the depth its budget allows.
    let walk = walk.with_right_picks();
    let mut config = TickConfig::for_walk(&walk, lanes);
    config.shared_rounds = joint && config.batch_spec.is_some();
    let mut engine = TickEngine::new(walk, lanes)
        .expect("the tick engine starts")
        .with_config(config)
        .expect("the walk's own config")
        .with_trace();
    let mut ids = Vec::with_capacity(lanes);
    let mut firsts = Vec::with_capacity(lanes);
    for at in 0..lanes as u32 {
        let rows: Vec<u32> = (0..7).map(|index| 1 + at * 13 + index).collect();
        let (id, step) = prompt(&mut engine, &rows);
        assert_eq!(
            engine.kept(id).and_then(|kept| kept.history),
            Some(7),
            "the prompt's history"
        );
        ids.push(id);
        firsts.push(step.next_token.expect("the prompt's token"));
    }
    for (id, token) in ids.iter().zip(&firsts) {
        engine.feed(*id, *token).expect("the row is fed");
    }
    let mut answered = 0;
    while answered < lanes {
        for event in engine.tick().expect("the shared tick") {
            if matches!(event, TickEvent::Decode { .. }) {
                answered += 1;
            }
        }
    }
    let trace = engine.trace().expect("the trace").render();
    let verifies = engine.walk().verifies();
    if joint {
        assert!(
            trace.contains("plan=shared"),
            "several lanes share one verify: {trace}"
        );
        assert_eq!(verifies.len(), 1, "one shared verify, not one a lane");
        assert_eq!(
            verifies[0].len(),
            lanes,
            "every member's chain is in the pass"
        );
        let rows: usize = verifies[0].iter().sum();
        assert!(
            verifies[0].iter().all(|rows| *rows >= 1),
            "every member steps: {verifies:?}"
        );
        // The row share gives out the budget or every chain, whichever is fewer, and a right
        // chain never stops a row: the pass's width is exact.
        let depth = engine.walk().depth();
        let budget = engine.walk().drafter().budget(lanes);
        let want = lanes + (lanes * depth).min(budget);
        assert_eq!(
            rows, want,
            "the shared verify holds the anchors and the drafts the budget allows"
        );
    } else {
        assert!(
            trace.contains("plan=plain"),
            "joint rounds off runs plain rows: {trace}"
        );
        assert!(
            verifies.is_empty(),
            "a plain row takes no verify: {verifies:?}"
        );
    }
}

/// Panics unless a decoding row's speculative round keeps only the rows the model chose: the
/// accepted draft reaches the request and the rejected one does not.
pub fn check_speculative_round(walk: ShapeWalk) {
    let mut engine = TickEngine::new(walk, 1)
        .expect("the tick engine starts")
        .with_trace();
    let (id, first) = prompt(&mut engine, &[10, 11, 12, 13]);
    let anchor = first.next_token.expect("a token");
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
            .contains(&wrong(anchor)),
        "the rejected draft never reaches the request"
    );
    assert_eq!(
        engine.walk().verifies(),
        vec![vec![3]],
        "the anchor and both drafts verified in one pass"
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

/// Emit the profile tests an arm that serves on the tick engine states.
///
/// The arm gives its walk's type, its row in the arms table, the depth its drafter proposes,
/// the prompt chunk widths it serves, and a constructor of the CPU [`ShapeWalk`] at one of
/// those widths and that depth. The macro emits, in a module named `profile`, a test a
/// statement:
///
/// * `the_declared_compute_rows_are_the_arm_ledger_rows`: the walk's declared compute rows
///   are the arms table's;
/// * `the_tick_config_is_the_arms_profile`: the engine's config is derived from the walk's
///   limits and drafter, and the joint-round switch is the runtime's;
/// * `the_tick_engine_hands_the_arms_chunks`: a prompt past two passes splits into the pass
///   width's chunks, on the single process and with a group's row exchange;
/// * `the_tick_engine_keeps_the_arms_joint_round`: several lanes share one verify, inside the
///   drafter's budget, and run plain rows with the joint switch off;
/// * `a_speculative_round_keeps_its_accepted_rows_only`: a rejected draft never reaches the
///   request;
/// * `the_generated_checks_drive_the_tick_engine`: the arm's generated checks take the path
///   it serves on.
///
/// ```ignore
/// sparkflash_runtime::profile_checks!(
///     MyWalk,
///     table: "My",
///     depth: 3,
///     chunks: [4096, 8192],
///     shape: |chunk, depth| shape_walk(depth, walk_limits(chunk, 1 << 20))
/// );
/// ```
#[macro_export]
macro_rules! profile_checks {
    (
        $arm:ty,
        table: $table:literal,
        depth: $depth:expr,
        chunks: [$($chunk:expr),* $(,)?],
        shape: |$chunk_var:ident, $depth_var:ident| $shape:expr $(,)?
    ) => {
        /// The profile checks this arm states.
        pub mod profile {
            use super::*;
            use $crate::Walk as _;

            /// The arm's row in the arms table.
            const TABLE: &str = $table;

            /// The depth the arm's drafter proposes a member.
            const DEPTH: usize = $depth;

            /// The prompt chunk widths the arm serves.
            const CHUNKS: &[usize] = &[$($chunk),*];

            /// A shape walk at `chunk` rows a pass, `depth` drafts a member.
            fn shape($chunk_var: usize, $depth_var: usize) -> $crate::testing::ShapeWalk {
                $shape
            }

            /// The widest prompt chunk the arm serves.
            fn widest() -> usize {
                *CHUNKS.iter().max().expect("an arm states a chunk width")
            }

            #[test]
            fn the_declared_compute_rows_are_the_arm_ledger_rows() {
                $crate::testing::check_compute_rows::<$arm>(TABLE);
            }

            #[test]
            fn the_tick_config_is_the_arms_profile() {
                $crate::testing::check_tick_config(shape(widest(), DEPTH), true);
                $crate::testing::check_tick_config(shape(widest(), DEPTH), false);
            }

            #[test]
            fn the_tick_engine_hands_the_arms_chunks() {
                for &width in CHUNKS {
                    let pass = shape(width, 0).limits().rows.max(1);
                    let rows = 2 * pass + pass / 3;
                    assert_eq!(
                        $crate::testing::prompt_chunks(shape(width, 0), rows, false),
                        $crate::testing::chunks_at(rows, pass),
                        "a {rows}-row prompt at {width} rows a pass"
                    );
                    assert_eq!(
                        $crate::testing::prompt_chunks(shape(width, 0), rows, true),
                        $crate::testing::chunks_at(rows, pass),
                        "the group's bulk chunks are the single process's chunks"
                    );
                }
            }

            #[test]
            fn the_tick_engine_keeps_the_arms_joint_round() {
                $crate::testing::check_joint_round(shape(widest(), DEPTH), 3, true);
                $crate::testing::check_joint_round(shape(widest(), DEPTH), 3, false);
            }

            #[test]
            fn a_speculative_round_keeps_its_accepted_rows_only() {
                $crate::testing::check_speculative_round(shape(widest(), 2).with_wrong_picks());
            }

            #[test]
            fn the_generated_checks_drive_the_tick_engine() {
                assert!(<$arm as $crate::WalkFixture>::tick_engine());
            }
        }
    };
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

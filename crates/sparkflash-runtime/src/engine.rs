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

//! DeepSeek's tick as the runtime's engine, over any Walk.
//!
//! crate::tick holds every decision without a device. This module is the engine that takes them:
//! it owns the queue and the lane order, the prefix lookup a lane is chosen by, the prompt order
//! and the chunk boundaries, the decode share and the tick-ahead hook, the Drafter round's
//! schedule, lane memory and the tick trace. It calls crate::tick for each decision and passes
//! the same inputs the old scheduler did; a decision is never re-derived here.
//!
//! Runtime remains the executor: it owns a sequence's history, its resume and rewind, the Pass a
//! chunk or a step walks, and the round's settle. The engine drives it a chunk or a step at a
//! time. That split is what lets one engine serve every arm: the engine knows when, the runtime
//! knows how.
//!
//! Two decisions are not DeepSeek's and are named rather than imitated:
//!
//! * Where a prompt resumes. DeepSeek can restart a lane at any shared prefix because its state
//!   is positional. A walk with a recurrent region can only go back to a checkpoint, so
//!   crate::resume's rule decides the start and the engine records what it chose. The engine
//!   still chooses the lane by crate::tick::pick_lane's shared prefix.
//! * When a decode row runs. The old Batch decodes every sequence in its decode phase each tick,
//!   because the device scheduler owns the token. Here the caller owns it: a sequence steps when
//!   its request has fed the next token. The plan therefore counts the rows that can run. A
//!   prompt that finishes its prefill in a tick is headed in that tick and decoded in the next,
//!   not both.
//!
//! The decode plan is decided ahead. A WalkHook between the last layer and the head of the
//! tick's prefill chunk computes it (Ahead), and that tick's decode uses it; when no chunk runs,
//! it is decided at the decode point.
//!
//! Neither changes a bit: the engine is the dense arm's serving path, and its own trace is
//! pinned in tests/tick_engine_trace.rs. The device-only decisions DeepSeek's Batch still makes
//! (lane memory and growth, adoption from another lane, paired prompt chunks, preemption, the
//! actual device step, the drafter's depth) are listed in docs/engine/04-onboarding-review.md as
//! unverified without a cluster.

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use sparkflash_comm::control::Tp4ControlHub;
use sparkflash_model::backend::{BackendError, ModelSession, Sampling, Speculation, StepOutput};
use sparkflash_model::vision::{ImageSpan, VisionImage};

use crate::chunk::ChunkPolicy;
use crate::group::{GroupWire, TickGroup, watched_hub};
use crate::ledger::{Ledger, Mode, Strategy};
use crate::runtime::Part;
use crate::runtime::{Runtime, RuntimeSlot, StepMode};
use crate::tick::{
    Clock, Decision, DecodePlan, DecodeShare, LaneCandidate, QUEUE, ShareClock, StepModel,
    TickHost, TickRows, TickTrace, TogetherHost, TraceSink, WallClock, bulk_budget, bulk_from,
    common_prefix, decode_plan, pick_lane, prefill_order, reusable_shared, tick_chunks,
    tick_prefill, tick_rows, whole_chunk_fits,
};
use crate::walk::{Drafter, Hidden, PassMask, Vision, Walk, WalkHook};

/// What the tick engine decides from, and the values a serving run reads from its profile and
/// the arm's crate::layout::Limits.
///
/// TickConfig::deepseek is DeepSeek's serving default, the values crate::tick::replay::Config
/// replays. TickConfig::for_walk is the same model narrowed to an arm's own limits: its
/// ChunkPolicy chunk, its Limits::rows, and no decoder-layer skip (only an arm with DeepSeek's
/// decoder stack has one). A drafting arm's profile adds its verify width as the member cap, its
/// drafter's depth and floor, and the Spec/Shared/Plain decode plan shapes; for_runtime narrows
/// the shared shape by the runtime's own joint-rounds switch.
#[derive(Clone, Copy, Debug)]
pub struct TickConfig {
    /// The lanes the pool holds.
    pub lanes: usize,
    /// The most rows one step may hold (R_MAX).
    pub rows_max: usize,
    /// The most members one batch may hold: the arm's verify width ([Limits::members]). A plain
    /// decode group never exceeds it, and a shared round verifies at most this many lanes.
    pub members: usize,
    /// The rows at a prompt's end that run the decoder layers.
    pub decoder_warm: usize,
    /// The rows one idle tick's exact prefill runs.
    pub exact_tick_rows: usize,
    /// The fewest rows a prefill takes the bulk path with.
    pub chunk_min: usize,
    /// The bulk chunk a busy tick runs.
    pub busy_chunk: usize,
    /// The exact rows steps a busy tick runs.
    pub busy_exact_steps: usize,
    /// The decoders' share of the device while a prompt prefills.
    pub share: DecodeShare,
    /// The fewest shared rows a long lane's prefix is kept for (crate::tick::reusable_shared).
    pub prefix_reuse_min_rows: usize,
    /// The most rows a bulk chunk may hold, when the arm has a bulk path.
    pub bulk_cap: Option<usize>,
    /// The chunks one tick may run at once: several consecutive chunks of one prompt
    /// (crate::tick::tick_chunks) or several prompts' whole rests in one paired pass
    /// (crate::tick::together). 1 runs one chunk a tick. An arm states it as [Limits::chunks]
    /// beside the paired pass; a group runs one, its wire having no paired frame yet.
    pub ways: usize,
    /// Whether a lane's held prefix is looked up at all.
    pub reuse: bool,
    /// Whether a prefill may take the bulk path, the exchange being present.
    pub exchange: bool,
    /// Whether the arm skips its decoder layers over all but a prompt's last rows.
    pub decoder_skip: bool,
    /// The batch-speculation switch: Some when this arm drafts, None when it does not.
    pub batch_spec: Option<f64>,
    /// Whether several decoding lanes verify in one shared pass: the runtime's joint rounds.
    /// With this off, or with more lanes than the verify width, the decode runs plain rows.
    pub shared_rounds: bool,
    /// The row share's least chance a draft is given a row.
    pub share_floor: f64,
    /// The shared step's cost model.
    pub model: StepModel,
    /// The drafter's chain cap.
    pub k: usize,
    /// The chunk the runtime's ChunkPolicy would run. The engine's chunk decisions are checked
    /// against it, so an arm whose chunk boundary its bits depend on keeps it.
    pub chunk_rows: usize,
}

impl TickConfig {
    /// DeepSeek's serving defaults at lanes lanes, as the profile and the device read them.
    #[must_use]
    pub fn deepseek(lanes: usize) -> Self {
        Self {
            lanes,
            rows_max: 64,
            members: 64,
            decoder_warm: 2541,
            exact_tick_rows: 4096,
            chunk_min: 96,
            busy_chunk: 2048,
            busy_exact_steps: 1,
            share: DecodeShare::new(0.5, 200_000.0),
            prefix_reuse_min_rows: crate::tick::prefix_reuse().1,
            bulk_cap: Some(4096),
            ways: 3,
            reuse: crate::tick::prefix_reuse().0,
            exchange: true,
            decoder_skip: true,
            batch_spec: Some(0.2),
            shared_rounds: true,
            share_floor: 0.2,
            model: (23.0, 4.0, 23.0, 4.0, 2.3),
            k: 5,
            chunk_rows: 4096,
        }
    }

    /// The tick an arm serves with: its own chunk, row width and verify width, the rest
    /// DeepSeek's.
    #[must_use]
    pub fn for_walk<W: Walk + ?Sized>(walk: &W, lanes: usize) -> Self {
        let limits = walk.limits();
        let chunk = ChunkPolicy::for_limits(limits).rows();
        let draft = W::Drafter::PRESENT && walk.drafter().drafts();
        Self {
            lanes,
            // A prompt chunk is as wide as the arm's pass; a verify is as wide as its batch.
            rows_max: limits.rows.max(1),
            members: limits.members.max(1),
            decoder_warm: 0,
            exact_tick_rows: chunk,
            chunk_min: chunk,
            busy_chunk: chunk,
            busy_exact_steps: 1,
            share: DecodeShare::new(0.5, 200_000.0),
            prefix_reuse_min_rows: crate::tick::prefix_reuse().1,
            bulk_cap: Some(chunk),
            // Several chunks a tick need the paired pass; an arm without it runs one.
            ways: if W::CAPABILITIES.passes.contains(PassMask::PAIRED) {
                limits.chunks.max(1)
            } else {
                1
            },
            reuse: crate::tick::prefix_reuse().0,
            // Dense (and every arm but DeepSeek) has no row exchange to pair chunks over.
            exchange: false,
            decoder_skip: false,
            batch_spec: draft.then_some(0.2),
            // A drafting arm's shapes: its drafter's own depths, shared across the members of one
            // verify when the runtime verifies them in one pass.
            shared_rounds: draft,
            share_floor: if draft { walk.drafter().floor() } else { 0.2 },
            model: (23.0, 4.0, 23.0, 4.0, 2.3),
            k: if draft { walk.drafter().budget(1) } else { 0 },
            chunk_rows: chunk,
        }
    }

    /// [Self::for_walk] with the runtime's own joint-rounds switch: a drafting arm whose joint
    /// rounds are off takes plain rows for the lanes that would have shared a verify, exactly as
    /// its runtime would.
    #[must_use]
    pub fn for_runtime<W: Walk>(runtime: &Runtime<W>, lanes: usize) -> Self {
        let mut config = Self::for_walk(runtime.walk(), lanes);
        config.shared_rounds = config.batch_spec.is_some() && runtime.joint_rounds();
        config
    }

    fn tick_rows(&self) -> TickRows {
        TickRows {
            busy_chunk: self.busy_chunk,
            busy_exact_steps: self.busy_exact_steps,
            exact_tick_rows: self.exact_tick_rows,
            rows_max: self.rows_max,
            chunk_min: self.chunk_min,
        }
    }
}

/// One request's identity in the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestId(pub u64);

/// A request's phase, as the scheduler models it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Admitted, waiting for a lane.
    Waiting,
    /// On a lane, with prompt rows left.
    Prefill { target: usize, bulk: bool },
    /// The prompt is in: rows step when the caller feeds the next token.
    Decode,
}

/// One request the engine is serving.
struct Request {
    id: RequestId,
    /// The prompt, then every token the caller has fed since.
    tokens: Vec<u32>,
    images: Vec<VisionImage>,
    prompt_len: usize,
    lane: Option<usize>,
    phase: Phase,
    admitted_ms: u64,
    done: bool,
    /// The request's sampler, applied to the sequence when its prompt is started.
    sampling: Sampling,
}

impl Request {
    fn emitted(&self) -> usize {
        self.tokens.len() - self.prompt_len
    }

    /// The position the prompt ends at, or the last fed token while it decodes.
    fn target(&self) -> usize {
        if self.emitted() == 0 {
            self.prompt_len
        } else {
            self.tokens.len() - 1
        }
    }
}

/// A lane's bookkeeping, as crate::tick::replay models it: the prompt tokens the slot holds, the
/// images those spans carry, the request it serves, when it was last given up and where its
/// decoder-layer skip ends.
#[derive(Default)]
struct Lane {
    resident: Vec<u32>,
    spans: Vec<ImageSpan>,
    seq: Option<RequestId>,
    used: u64,
    skip_end: usize,
}

/// One decode row this tick can run.
#[derive(Debug, Clone, Copy)]
struct Fed {
    id: RequestId,
    token: u32,
}

/// What one tick answered.
#[derive(Debug)]
pub enum TickEvent {
    /// A prompt finished: this is its first step.
    Prompt {
        /// The request.
        id: RequestId,
        /// Its first step.
        step: StepOutput,
    },
    /// A decode row ran: its step and, for a drafting arm, the held answers after it.
    Decode {
        /// The request.
        id: RequestId,
        /// The run, anchor first.
        steps: Vec<StepOutput>,
    },
    /// A request failed and is no longer served.
    Failed {
        /// The request.
        id: RequestId,
        /// The failure.
        error: BackendError,
    },
}

/// What a request holds after its last job, for a serving handle that cannot read its slot.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Kept {
    /// The length of the slot's history, when it has one.
    pub history: Option<usize>,
    /// The length of its rewind point, when it has one.
    pub rewind: Option<usize>,
    /// The images the history holds.
    pub images: Vec<ImageSpan>,
    /// How many of the last prompt's tokens the slot already held.
    pub cached: usize,
}

/// The tick-ahead hook: the plan for the tick's decode rows, decided between the last layer and
/// the head of the prefill walk. DeepSeek decides the next tick while the device works; this is
/// where the engine makes the same decision, through crate::tick::decode_plan.
struct Ahead {
    decoding: usize,
    batch_spec: Option<f64>,
    shared_rounds: bool,
    members: usize,
    drafter: bool,
    plan: Option<DecodePlan>,
}

impl Ahead {
    fn decide(&self) -> DecodePlan {
        plan_for(
            self.decoding,
            self.batch_spec,
            self.shared_rounds,
            self.members,
            self.drafter,
        )
    }
}

impl WalkHook for Ahead {
    fn after_layers(&mut self, _hidden: &Hidden) -> Result<(), BackendError> {
        self.plan = Some(self.decide());
        Ok(())
    }
}

/// The tick's decode plan: a drafting arm's Spec/Shared/Plain shapes, and crate::tick::decode_plan
/// for an arm which declares no drafter.
///
/// A drafting arm takes one of three shapes. One decoding row is `Spec`: that member's round,
/// drafted and verified alone. Several rows whose anchors fit the verify width are `Shared` when
/// the runtime verifies its lanes' rounds in one pass: every member's chain in one verify. Anything
/// wider than the verify width, or a rank whose joint rounds are off, runs `Plain`: one row a
/// member, in groups no wider than the verify width. That is the old runtime's own rule, which
/// chose the joint round for every batch it was handed.
///
/// An arm with no drafter has no speculative step to take. Its single sequence runs one plain row
/// and the trace says plain, because calling the row spec would name a row the arm never walked.
fn plan_for(
    rows: usize,
    batch_spec: Option<f64>,
    shared_rounds: bool,
    members: usize,
    drafter: bool,
) -> DecodePlan {
    let members = members.max(1);
    if drafter {
        if rows > 1 && shared_rounds && batch_spec.is_some() && rows <= members {
            return DecodePlan::Shared;
        }
        return decode_plan(rows, false, batch_spec, shared_rounds, members);
    }
    match decode_plan(rows, false, batch_spec, shared_rounds, members) {
        DecodePlan::None => DecodePlan::None,
        _ => DecodePlan::Plain {
            groups: rows.div_ceil(members),
        },
    }
}

/// Rank 0 gives up a lane: the release frame first (so a grouped worker gives up the same lane
/// before rank 0's half does), then the runtime's slot. A failure after the frame leaves the
/// workers past rank 0, so it ends the group ([TickGroup::out_of_step]).
fn release_lane<W: Walk>(
    group: Option<&TickGroup>,
    runtime: &Runtime<W>,
    lane: usize,
    slot: &mut RuntimeSlot<W>,
) -> bool {
    if let Some(group) = group {
        if let Err(error) = group.release(lane) {
            group.out_of_step(&error.to_string());
        }
        if let Err(error) = runtime.release_slot(slot) {
            group.out_of_step(&error.to_string());
        }
        true
    } else {
        runtime.release_slot(slot).is_ok()
    }
}

/// The fused engine's prompt-schedule switch. It is no longer read by any engine: the tick
/// engine always runs a prompt in chunks between the decoders' steps. A value that asks for
/// something else is refused rather than ignored.
pub const PROMPT_PARTS_ENV: &str = "SPARKFLASH_PROMPT_PARTS";

/// [PROMPT_PARTS_ENV] as set.
///
/// # Errors
/// A value that asks for a schedule this engine does not run, or a variable that is not UTF-8.
fn prompt_parts() -> Result<(), String> {
    match std::env::var(PROMPT_PARTS_ENV) {
        Err(std::env::VarError::NotPresent) => prompt_parts_request(None),
        Ok(value) => prompt_parts_request(Some(&value)),
        Err(error) => Err(format!("{PROMPT_PARTS_ENV}: {error}")),
    }
}

/// What a [PROMPT_PARTS_ENV] of `value` asks for. `on` (and unset) is the schedule the tick
/// engine runs, so it is accepted; `off` asked the fused engine to run every prompt whole as
/// it arrived, which this engine does not do, and is refused by name.
///
/// # Errors
/// `off`, or a value that is neither `on` nor `off`.
fn prompt_parts_request(value: Option<&str>) -> Result<(), String> {
    match value.map(str::trim) {
        None | Some("" | "on") => Ok(()),
        Some("off") => Err(format!(
            "{PROMPT_PARTS_ENV}=off ran every prompt whole in the fused engine, which is retired: \
             the tick engine chunks every prompt, so unset it"
        )),
        Some(other) => Err(format!("{PROMPT_PARTS_ENV}={other} is neither on nor off")),
    }
}

/// DeepSeek's tick over one Walk.
pub struct TickEngine<W: Walk> {
    runtime: Runtime<W>,
    config: TickConfig,
    /// The pool's slots, one a lane, indexed by lane. A slot stays here while its lane is free
    /// so its history can be reused; None only while a job holds it.
    slots: Vec<Option<RuntimeSlot<W>>>,
    /// The lanes no request holds.
    free: Vec<usize>,
    /// The lane bookkeeping, in the same order as slots.
    lanes: Vec<Lane>,
    /// Requests not yet finished, in admission order.
    requests: Vec<Request>,
    /// Prompts admitted and not yet placed.
    admissions: VecDeque<RequestId>,
    /// Decode rows fed by callers and not yet run.
    fed: Vec<Fed>,
    /// The next tick's decode plan, computed under the last prefill walk (the tick-ahead
    /// hook): the row count it was decided for, and the plan.
    ahead: Option<(usize, DecodePlan)>,
    /// Rank 0's group, when this engine serves a tensor-parallel group: every prompt, chunk,
    /// step, give-back and release is broadcast before it runs ([TickGroup]). The workers replay
    /// the rows and the step shape; the decisions stay here.
    group: Option<TickGroup>,
    /// The clock every decision is timed by.
    clock: Box<dyn Clock>,
    trace: Option<TraceSink>,
    tick: u64,
    last_ms: u64,
    share_clock: ShareClock,
    next_id: u64,
}

/// One tick of the engine as crate::tick::tick_prefill sees it: the engine, the tick's time and
/// the events it has made so far.
struct Ticking<'t, W: Walk> {
    engine: &'t mut TickEngine<W>,
    now: u64,
    events: &'t mut Vec<TickEvent>,
}

/// The engine's answers to crate::tick::together. It adopts nothing yet (a lane's prefix is
/// reused at placement, not copied between lanes), and a placed prompt's lane already has its
/// room ([TickEngine::ensure_room]), so nothing is copied and every lane is backed.
impl<W: Walk> TogetherHost for Ticking<'_, W> {
    type Error = std::convert::Infallible;

    fn order(&self) -> Vec<(usize, usize)> {
        self.engine.prefill_order()
    }

    fn adopt(&mut self, _index: usize) -> Result<(), Self::Error> {
        Ok(())
    }

    fn whole(&self, index: usize, budget: usize) -> Option<usize> {
        let request = &self.engine.requests[index];
        let (Phase::Prefill { bulk: true, .. }, Some(_)) = (request.phase, request.lane) else {
            return None;
        };
        let left = self.engine.left(index);
        (!request.done
            && request.images.is_empty()
            && whole_chunk_fits(
                left,
                budget,
                self.engine.config.decoder_skip,
                self.engine.config.decoder_warm,
                self.engine.config.rows_max,
            ))
        .then_some(left)
    }

    fn shared(&self, lead: usize, other: usize) -> usize {
        let other = &self.engine.requests[other];
        common_prefix(
            &self.engine.requests[lead].tokens,
            &[],
            &other.tokens,
            &other.images,
        )
    }

    fn pos(&self, index: usize) -> usize {
        let request = &self.engine.requests[index];
        request
            .lane
            .and_then(|lane| self.engine.slots[lane].as_ref())
            .map_or(0, |slot| slot.history().len())
    }

    fn back(&mut self, _index: usize) -> Result<bool, Self::Error> {
        Ok(true)
    }
}

impl<W: Walk> TickHost for Ticking<'_, W> {
    fn decoding(&self) -> bool {
        self.engine.decoding()
    }

    fn left(&self, index: usize) -> usize {
        self.engine.left(index)
    }

    fn budget(&self, decoding: bool) -> usize {
        let config = &self.engine.config;
        bulk_budget(&config.tick_rows(), config.bulk_cap, decoding).0
    }

    fn ways(&self) -> usize {
        self.engine.bulk_ways()
    }

    fn prefill_together(&mut self, picked: &[usize]) -> Result<usize, Self::Error> {
        Ok(self.engine.prefill_together(self.now, picked, self.events))
    }

    fn prefill_one(&mut self, index: usize, _decoding: bool) -> Result<usize, Self::Error> {
        Ok(self.engine.prefill(self.now, index, self.events))
    }
}

impl<W: Walk> TickEngine<W> {
    /// An engine over walk with capacity lanes.
    ///
    /// # Errors
    /// A runtime that refuses the walk, or a lane the arm cannot open.
    pub fn new(walk: W, capacity: usize) -> Result<Self, BackendError> {
        prompt_parts().map_err(|error| BackendError::new("tick", error))?;
        if capacity == 0 {
            return Err(BackendError::new(
                "tick",
                "a tick engine needs at least one lane",
            ));
        }
        let runtime = Runtime::new(walk, 0)?;
        if W::Drafter::PRESENT && !W::CAPABILITIES.passes.contains(PassMask::VERIFY) {
            return Err(BackendError::new(
                "tick",
                format!(
                    "{} declares a drafter ({}) without the verify pass its round runs",
                    std::any::type_name::<W>(),
                    std::any::type_name::<W::Drafter>()
                ),
            ));
        }
        let config = TickConfig::for_runtime(&runtime, capacity);
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(Some(runtime.open_slot()?));
        }
        let trace = trace_path().map(TraceSink::file);
        Ok(Self {
            runtime,
            config,
            slots,
            free: (0..capacity).collect(),
            lanes: (0..capacity).map(|_| Lane::default()).collect(),
            requests: Vec::new(),
            admissions: VecDeque::new(),
            fed: Vec::new(),
            ahead: None,
            group: None,
            clock: Box::new(WallClock::new()),
            trace,
            tick: 0,
            last_ms: 0,
            share_clock: ShareClock::default(),
            next_id: 0,
        })
    }

    /// The same engine with config (a gate's fixed workload, or DeepSeek's defaults).
    ///
    /// # Errors
    /// A configuration whose lane count differs, or whose width or chunks a tick the walk
    /// cannot hold.
    pub fn with_config(mut self, config: TickConfig) -> Result<Self, BackendError> {
        if config.lanes != self.config.lanes {
            return Err(BackendError::new(
                "tick",
                format!(
                    "the config names {} lanes, the engine holds {}",
                    config.lanes, self.config.lanes
                ),
            ));
        }
        if config.rows_max == 0 || config.members == 0 {
            return Err(BackendError::new(
                "tick",
                "a step needs at least one row and one member",
            ));
        }
        let limits = self.runtime.limits();
        if config.rows_max > limits.rows || config.bulk_cap.is_some_and(|cap| cap > limits.rows) {
            return Err(BackendError::new(
                "tick",
                format!(
                    "the config asks for {} rows and the walk takes {}",
                    config.rows_max.max(config.bulk_cap.unwrap_or(0)),
                    limits.rows
                ),
            ));
        }
        if config.members > limits.members {
            return Err(BackendError::new(
                "tick",
                format!(
                    "the config batches {} members and the walk verifies {}",
                    config.members, limits.members
                ),
            ));
        }
        let chunks = if W::CAPABILITIES.passes.contains(PassMask::PAIRED) {
            limits.chunks.max(1)
        } else {
            1
        };
        if config.ways == 0 || config.ways > chunks {
            return Err(BackendError::new(
                "tick",
                format!(
                    "the config runs {} chunks a tick and the walk's pass holds {chunks}",
                    config.ways
                ),
            ));
        }
        self.config = config;
        Ok(self)
    }

    /// The same engine timed by clock. The gate injects one so a replay's costs are scripted.
    #[must_use]
    pub fn with_clock(mut self, clock: Box<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// The same engine keeping its decisions, with no file. The CPU gate's trace.
    #[must_use]
    pub fn with_trace(mut self) -> Self {
        self.trace = Some(TraceSink::in_memory());
        self
    }

    /// Decides the tick-ahead plan as the [Ahead] hook would have, for `rows` decode rows.
    ///
    /// This is the state the hook leaves between a tick's prefill walk and its decode. The CPU
    /// gate uses it to show that a plan is not an execution: a plan that names no round still
    /// has one run, and the engine never lets a stale plan skip the rows the caller fed.
    pub fn with_ahead(&mut self, rows: usize, plan: DecodePlan) -> &mut Self {
        self.ahead = Some((rows, plan));
        self
    }

    /// The same engine as rank 0 of a [TickGroup]: every prompt, chunk, step, give-back and release is
    /// broadcast before it runs, and the row exchange is on, so the group's prompt path takes the
    /// arm's bulk chunks.
    ///
    /// The workers run the same tick on the same rows ([crate::mirror_tick_lanes]); what stays
    /// rank 0's is every decision - the lane, the hold, the chunk boundary, the step shape and
    /// the tick-ahead plan.
    #[must_use]
    pub fn with_group(mut self, group: TickGroup) -> Self {
        self.group = Some(group);
        self.config.exchange = true;
        // The wire carries a chunk's rows a lane ([TickGroup::rows]) and no paired frame, so a
        // worker could not mirror several prompts in one pass: a group runs one chunk a tick.
        self.config.ways = 1;
        self
    }

    /// Whether this engine is rank 0 of a group.
    #[must_use]
    pub fn grouped(&self) -> bool {
        self.group.is_some()
    }

    /// The arm.
    #[must_use]
    pub fn walk(&self) -> &W {
        self.runtime.walk()
    }

    /// The decisions the engine makes from.
    #[must_use]
    pub fn config(&self) -> &TickConfig {
        &self.config
    }

    /// The runtime the engine drives.
    #[must_use]
    pub fn runtime(&self) -> &Runtime<W> {
        &self.runtime
    }

    /// The trace, when the engine keeps one.
    #[must_use]
    pub fn trace(&self) -> Option<&TickTrace> {
        self.trace.as_ref().map(TraceSink::trace)
    }

    /// Writes the trace as it stands, when one is kept (crate::tick::TraceSink::flush).
    pub fn flush_trace(&mut self) {
        if let Some(trace) = self.trace.as_mut() {
            trace.flush();
        }
    }

    /// The boot ledger: the arm's rows, with the engine's lanes and prompt schedule.
    ///
    /// # Errors
    /// A row the arm does not state.
    pub fn ledger(&self) -> Result<Ledger, String> {
        let mut ledger = crate::ledger_for::<W>();
        ledger.set(
            Strategy::Lanes,
            Mode::native(format!(
                "{} lanes, one row a sequence; a prompt one chunk a tick",
                self.config.lanes
            )),
        );
        if !ledger.states(Strategy::PromptParts) {
            ledger.set(
                Strategy::PromptParts,
                Mode::off("the arm's walk takes no bulk chunk"),
            );
        }
        if let Some(group) = self.group.as_ref() {
            ledger.set(
                Strategy::Group,
                Mode::native(format!(
                    "{}: rank 0 decides the tick and the workers replay its rows and step shape",
                    group.name()
                )),
            );
        }
        ledger.require_complete()?;
        Ok(ledger)
    }

    /// Admits a prompt and returns its request.
    ///
    /// The request waits for a lane (tick places it) and answers its first step when the prompt's
    /// chunks are in, through TickEvent::Prompt.
    ///
    /// # Errors
    /// An empty prompt, an image on an arm with no vision path, or a context it cannot hold.
    pub fn admit(
        &mut self,
        tokens: &[u32],
        images: &[VisionImage],
        sampling: Sampling,
    ) -> Result<RequestId, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::new("tick", "the prompt has no tokens"));
        }
        if !images.is_empty() && !W::Vision::PRESENT {
            return Err(BackendError::new("vision", "this arm has no vision path"));
        }
        if tokens.len() > self.runtime.limits().context {
            return Err(BackendError::new(
                "context",
                format!(
                    "the prompt needs {} positions and the sequence holds {}",
                    tokens.len(),
                    self.runtime.limits().context
                ),
            ));
        }
        // DeepSeek's queue: the lanes plus crate::tick::QUEUE sequences may wait. A prompt past
        // that is refused here, before it becomes a frame the group has to abandon.
        let queued = self.requests.iter().filter(|request| !request.done).count();
        if queued >= self.slots.len() + QUEUE {
            return Err(BackendError::new(
                "queue",
                format!(
                    "the queue is full: {} lane(s) and {QUEUE} waiting",
                    self.slots.len()
                ),
            ));
        }
        // The group's own refusal runs before anything is queued, so a prompt no rank can take
        // never becomes a frame a worker has to abandon.
        if let Some(group) = self.group.as_ref() {
            group.admit(tokens, images)?;
        }
        let id = RequestId(self.next_id);
        self.next_id += 1;
        self.requests.push(Request {
            id,
            tokens: tokens.to_vec(),
            images: images.to_vec(),
            prompt_len: tokens.len(),
            lane: None,
            phase: Phase::Waiting,
            admitted_ms: 0,
            done: false,
            sampling: Sampling {
                prompt_len: tokens.len(),
                ..sampling
            },
        });
        self.admissions.push_back(id);
        Ok(id)
    }

    /// Feeds the next token of a decoding request, for the tick that runs it.
    ///
    /// # Errors
    /// A request that is unknown, finished, or still prefilling.
    pub fn feed(&mut self, id: RequestId, token: u32) -> Result<(), BackendError> {
        let Some(request) = self.requests.iter().find(|request| request.id == id) else {
            return Err(BackendError::new(
                "tick",
                format!("request {} is not in the engine", id.0),
            ));
        };
        if request.done || request.phase != Phase::Decode {
            return Err(BackendError::new(
                "tick",
                format!("request {} has no decode row to feed", id.0),
            ));
        }
        if self.fed.iter().any(|fed| fed.id == id) {
            return Err(BackendError::new(
                "tick",
                format!("request {} already fed this tick", id.0),
            ));
        }
        self.fed.push(Fed { id, token });
        Ok(())
    }

    /// Ends a request: its lane is given back for another request, and what the lane holds stays
    /// for a later prompt to reuse. A request the engine does not hold is ignored.
    pub fn finish(&mut self, id: RequestId) {
        let Some(index) = self.requests.iter().position(|request| request.id == id) else {
            return;
        };
        self.requests[index].done = true;
        self.fed.retain(|fed| fed.id != id);
        let Some(lane) = self.requests[index].lane else {
            return;
        };
        self.lanes[lane].seq = None;
        self.lanes[lane].used = self.tick;
        if !self.free.contains(&lane) {
            self.free.push(lane);
        }
    }

    /// Gives back count answers of id's run the caller never used.
    ///
    /// # Errors
    /// A count past the history, or an arm that cannot go back.
    pub fn give_back(&mut self, id: RequestId, count: usize) -> Result<(), BackendError> {
        let Some(lane) = self
            .requests
            .iter()
            .find(|request| request.id == id)
            .and_then(|request| request.lane)
        else {
            return Ok(());
        };
        let Some(slot) = self.slots[lane].as_mut() else {
            return Ok(());
        };
        // Rank 0's own half first: an arm that refuses the give-back refuses on every rank, so
        // a refusal must not leave the workers past rank 0 (as a lane mirror's does).
        self.runtime.give_back_slot(slot, count)?;
        if let Some(group) = self.group.as_ref()
            && let Err(error) = group.give_back(lane, count)
        {
            group.out_of_step(&error.to_string());
        }
        Ok(())
    }

    /// Whether the engine has work for a tick: a prompt to place or prefill, or a decode row to
    /// run.
    #[must_use]
    pub fn busy(&self) -> bool {
        // A waiting prompt can advance only while a lane is free; a decoding request only when
        // its caller has fed a row. When neither is true the engine has nothing to do, and the
        // serving thread waits for a job instead of ticking.
        !self.fed.is_empty()
            || self
                .requests
                .iter()
                .any(|request| !request.done && matches!(request.phase, Phase::Prefill { .. }))
            || (!self.free.is_empty()
                && self
                    .requests
                    .iter()
                    .any(|request| !request.done && request.phase == Phase::Waiting))
    }

    /// What id holds after its last job.
    #[must_use]
    pub fn kept(&self, id: RequestId) -> Option<Kept> {
        let request = self.requests.iter().find(|request| request.id == id)?;
        let lane = request.lane?;
        let slot = self.slots[lane].as_ref()?;
        Some(Kept {
            history: Some(slot.history().len()),
            rewind: slot.rewind_point(),
            images: slot.held_image_spans().to_vec(),
            cached: slot.cached(),
        })
    }

    /// The runtime's history for id, when it has a lane.
    #[must_use]
    pub fn history(&self, id: RequestId) -> Option<&[u32]> {
        let lane = self
            .requests
            .iter()
            .find(|request| request.id == id)?
            .lane?;
        self.slots[lane].as_ref().map(RuntimeSlot::history)
    }

    /// How many of the last prompt's tokens id's sequence already held.
    #[must_use]
    pub fn cached(&self, id: RequestId) -> Option<usize> {
        let lane = self
            .requests
            .iter()
            .find(|request| request.id == id)?
            .lane?;
        self.slots[lane].as_ref().map(RuntimeSlot::cached)
    }

    /// id's speculative counters, as the arm kept them.
    #[must_use]
    pub fn speculation(&self, id: RequestId) -> Speculation {
        let Some(lane) = self
            .requests
            .iter()
            .find(|request| request.id == id)
            .and_then(|request| request.lane)
        else {
            return Speculation::default();
        };
        self.slots[lane]
            .as_ref()
            .map_or_else(Speculation::default, |slot| {
                self.runtime.speculation_slot(slot)
            })
    }

    /// Runs one tick: places the waiting prompts, records the tick's hold, runs the first prefill
    /// in crate::tick::prefill_order and the decode plan, and answers what finished.
    ///
    /// # Errors
    /// Only a failure that leaves the engine unable to decide. A request's own failure is a
    /// TickEvent::Failed.
    pub fn tick(&mut self) -> Result<Vec<TickEvent>, BackendError> {
        let now = self.clock.now_ms();
        let cost_ms = now.saturating_sub(self.last_ms);
        self.last_ms = now;
        self.tick += 1;
        let mut events = Vec::new();
        // A prompt admitted before this tick is admitted at its start, as Replay::tick does: its
        // age is measured from the tick that first sees it.
        while let Some(id) = self.admissions.pop_front() {
            if let Some(request) = self.requests.iter_mut().find(|request| request.id == id) {
                request.admitted_ms = now;
            }
        }
        self.place(now, &mut events);
        let decoding = self.decoding();
        let order = self.prefill_order();
        let hold = self.share_clock.hold(
            &self.config.share,
            &order,
            |index| (now - self.requests[index].admitted_ms) as f64,
            decoding,
        );
        self.record(now, Decision::Hold { hold });
        let mut ticking = Ticking {
            engine: self,
            now,
            events: &mut events,
        };
        let long_chunk = match tick_prefill(&mut ticking, hold) {
            Ok(long_chunk) => long_chunk,
            Err(never) => match never {},
        };
        self.decode(now, &mut events);
        self.share_clock.settle(long_chunk, cost_ms as f64);
        self.requests.retain(|request| !request.done);
        self.flush_trace();
        Ok(events)
    }

    fn record(&mut self, now: u64, decision: Decision) {
        if let Some(trace) = self.trace.as_mut() {
            trace.record_at(self.tick, now, decision);
        }
    }

    fn decoding(&self) -> bool {
        self.requests
            .iter()
            .any(|request| !request.done && request.phase == Phase::Decode)
    }

    fn prefill_order(&self) -> Vec<(usize, usize)> {
        prefill_order(
            self.requests
                .iter()
                .enumerate()
                .filter_map(|(index, request)| match request.phase {
                    Phase::Prefill { target, .. } if !request.done => {
                        let lane = request.lane?;
                        let pos = self.slots[lane].as_ref()?.history().len();
                        Some((target.saturating_sub(pos), index))
                    }
                    _ => None,
                }),
        )
    }

    fn left(&self, index: usize) -> usize {
        let request = &self.requests[index];
        let Some(lane) = request.lane else {
            return 0;
        };
        let Phase::Prefill { target, .. } = request.phase else {
            return 0;
        };
        let pos = self.slots[lane]
            .as_ref()
            .map_or(target, |slot| slot.history().len());
        target.saturating_sub(pos)
    }

    /// Places every waiting request it can onto a free lane, in admission order.
    fn place(&mut self, now: u64, events: &mut Vec<TickEvent>) {
        loop {
            let Some(index) = self
                .requests
                .iter()
                .position(|request| !request.done && request.phase == Phase::Waiting)
            else {
                return;
            };
            let target = self.requests[index].target();
            let wanted = self.requests[index].tokens[..target].to_vec();
            let new_images = self.requests[index].images.clone();
            let candidate = {
                let lanes = &self.lanes;
                let config = &self.config;
                pick_lane(self.free.iter().copied().map(|lane| {
                    let held = &lanes[lane];
                    let raw_shared = if config.reuse {
                        common_prefix(&held.resident, &held.spans, &wanted, &[])
                    } else {
                        0
                    };
                    LaneCandidate {
                        lane,
                        shared: reusable_shared(
                            raw_shared,
                            held.resident.len(),
                            config.prefix_reuse_min_rows,
                        ),
                        used: held.used,
                    }
                }))
            };
            let Some(candidate) = candidate else {
                // Every lane serves a request: the rest wait.
                return;
            };
            let lane = candidate.lane;
            if !self.ensure_room(lane, &wanted) {
                self.fail(
                    index,
                    BackendError::new(
                        "context",
                        "no lane's memory can hold this prompt, and no idle lane holds any to give back",
                    ),
                    events,
                );
                continue;
            }
            let Some(slot) = self.slots[lane].as_mut() else {
                self.fail(
                    index,
                    BackendError::new("tick", "a free lane has no slot"),
                    events,
                );
                continue;
            };
            // The lane is claimed before the prompt starts, so a start that fails releases it
            // rather than leaving a half-rewound slot off the free list.
            self.requests[index].lane = Some(lane);
            let sampling = self.requests[index].sampling;
            self.runtime.walk().set_sampling(slot.seq_mut(), sampling);
            // A prompt whose start runs a collective (an image encode on every rank) has its
            // request frame in front of that start: the workers join the collective from the
            // frame, so rank 0 cannot wait for its own start first. The start frame follows
            // ([TickGroup::started]).
            let early = self.group.is_some() && !new_images.is_empty();
            if early {
                let group = self.group.as_ref().expect("a group");
                if let Err(error) = group.begin(lane, &wanted, &new_images, &sampling) {
                    group.out_of_step(&error.to_string());
                }
            }
            let start = match self.runtime.start_prompt_slot(slot, &wanted, &new_images) {
                Ok(start) => start,
                Err(error) => {
                    if early {
                        let group = self.group.as_ref().expect("a group");
                        group.out_of_step(&error.to_string());
                    }
                    self.fail(index, error, events);
                    continue;
                }
            };
            if start >= target {
                self.fail(
                    index,
                    BackendError::new(
                        "tick",
                        format!(
                            "a {target}-row prompt resumed at {start} with no row left to head"
                        ),
                    ),
                    events,
                );
                continue;
            }
            // The request frame carries the start rank 0 chose, so it follows rank 0's own
            // start_prompt_slot and precedes every row frame; the worker's own start is checked
            // against it there. A prompt that carried images already sent its request frame, so
            // its start travels alone. Nothing has run a row yet.
            if let Some(group) = self.group.as_ref() {
                let sent = if early {
                    group.started(lane, start)
                } else {
                    group.place(lane, &wanted, &new_images, start, &sampling)
                };
                if let Err(error) = sent {
                    group.out_of_step(&error.to_string());
                }
            }
            let bulk = bulk_from(
                target,
                start,
                self.config.bulk_cap.is_some(),
                self.config.exchange,
                false,
                self.config.chunk_min,
            );
            self.requests[index].phase = Phase::Prefill { target, bulk };
            self.free.retain(|free| *free != lane);
            let id = self.requests[index].id;
            self.lanes[lane].seq = Some(id);
            self.lanes[lane].used = now;
            self.lanes[lane].resident.truncate(start);
            self.lanes[lane].spans.retain(|span| span.start < start);
            self.record(
                now,
                Decision::Place {
                    seq: id.0,
                    lane,
                    shared: candidate.shared,
                    start,
                    target,
                    bulk,
                },
            );
        }
    }

    /// Gives up the least recently used free lane that still holds a request until lane has room
    /// for prompt. False when every idle lane has been given up and there is still none.
    fn ensure_room(&mut self, lane: usize, prompt: &[u32]) -> bool {
        loop {
            let Some(slot) = self.slots[lane].as_ref() else {
                return false;
            };
            if self.runtime.room_slot(slot, Some(prompt)) {
                return true;
            }
            let victim = self
                .free
                .iter()
                .copied()
                .filter(|&other| other != lane)
                .filter(|&other| {
                    self.slots[other]
                        .as_ref()
                        .is_some_and(|slot| !slot.history().is_empty())
                })
                .min_by_key(|&other| self.lanes[other].used);
            let Some(victim) = victim else {
                return false;
            };
            let Some(slot) = self.slots[victim].as_mut() else {
                return false;
            };
            // Every rank gives up the same lane before rank 0's own half does: a worker that
            // kept it would hold a lane rank 0 is about to reuse for a different prompt.
            if !release_lane(self.group.as_ref(), &self.runtime, victim, slot) {
                return false;
            }
            self.lanes[victim].resident.clear();
            self.lanes[victim].spans.clear();
            self.lanes[victim].skip_end = 0;
        }
    }

    fn fail(&mut self, index: usize, error: BackendError, events: &mut Vec<TickEvent>) {
        let id = self.requests[index].id;
        if let Some(lane) = self.requests[index].lane
            && let Some(slot) = self.slots[lane].as_mut()
        {
            // The prompt failed before any of its rows ran, so its request frame may never have
            // gone out; a release still keeps the lane the same on every rank (fresh on rank 0,
            // fresh on a worker that held an older sequence).
            release_lane(self.group.as_ref(), &self.runtime, lane, slot);
            self.lanes[lane].resident.clear();
            self.lanes[lane].spans.clear();
            self.lanes[lane].seq = None;
            self.lanes[lane].used = self.tick;
            if !self.free.contains(&lane) {
                self.free.push(lane);
            }
        }
        self.requests[index].done = true;
        events.push(TickEvent::Failed { id, error });
    }

    /// Runs the next prefill of index: one bulk chunk, or the exact rows steps a tick takes.
    /// Returns the rows it ran.
    fn prefill(&mut self, now: u64, index: usize, events: &mut Vec<TickEvent>) -> usize {
        let Phase::Prefill { target, bulk } = self.requests[index].phase else {
            return 0;
        };
        let Some(lane) = self.requests[index].lane else {
            return 0;
        };
        let Some(from) = self.slots[lane]
            .as_ref()
            .map(RuntimeSlot::history)
            .map(<[u32]>::len)
        else {
            return 0;
        };
        let id = self.requests[index].id;
        let mut ran = 0usize;
        let decoding = self.decoding();
        if bulk {
            let (budget, most) =
                bulk_budget(&self.config.tick_rows(), self.config.bulk_cap, decoding);
            // Several chunks at once only with nothing decoding (a tick with decoders runs one
            // short chunk) and without images (their tower rows are a single chunk's).
            let ways = if decoding || !self.requests[index].images.is_empty() {
                1
            } else {
                self.bulk_ways()
            };
            let split = self.config.decoder_skip.then_some(self.config.decoder_warm);
            let chunks = tick_chunks(
                from,
                target,
                budget,
                most,
                split,
                ways,
                self.config.rows_max,
                |_, _| None,
            );
            // The tick's chunks compute what they would one after the other, so they run that
            // way here; an arm that interleaves them does it inside its own pass.
            for (p, rows) in chunks {
                let rows = rows.min(target - p);
                self.record(
                    now,
                    Decision::Prefill {
                        seq: id.0,
                        from: p,
                        rows,
                        target,
                        bulk: true,
                    },
                );
                let decoder = !self.config.decoder_skip
                    || crate::tick::runs_decoder(p, rows, target, self.config.decoder_warm);
                self.lanes[lane].skip_end =
                    crate::tick::skip_end_after(self.lanes[lane].skip_end, p, rows, decoder);
                // The exact rows this chunk runs travel before it does; every rank runs this
                // many, not its own ChunkPolicy's chunk.
                if let Some(group) = self.group.as_ref()
                    && let Err(error) = group.rows(lane, rows)
                {
                    group.out_of_step(&error.to_string());
                }
                let mut ahead = self.ahead();
                let slot = self.slots[lane]
                    .as_mut()
                    .expect("a lane serving a prompt has its slot");
                match self
                    .runtime
                    .run_part_rows_hooked(slot, rows, Some(&mut ahead))
                {
                    Ok(part) => {
                        ran += rows;
                        let tokens = self.requests[index].tokens[p..p + rows].to_vec();
                        self.lanes[lane].resident.extend_from_slice(&tokens);
                        if let Some(plan) = ahead.plan {
                            self.ahead = Some((ahead.decoding, plan));
                        }
                        if let Part::Done(step) = part {
                            self.finish_prompt(index, step, events);
                            break;
                        }
                    }
                    Err(error) => {
                        // The frame went out before the rows ran: the worker is inside the same
                        // walk, so rank 0 cannot serve on past its failure.
                        if let Some(group) = self.group.as_ref() {
                            group.out_of_step(&error.to_string());
                        }
                        self.fail(index, error, events);
                        break;
                    }
                }
            }
        } else {
            let steps = tick_rows(
                &self.config.tick_rows(),
                self.config.bulk_cap,
                false,
                decoding,
            )
            .div_ceil(self.config.rows_max);
            let mut p = from;
            for _ in 0..steps {
                if p == target {
                    break;
                }
                let n = self.config.rows_max.min(target - p);
                self.record(
                    now,
                    Decision::Prefill {
                        seq: id.0,
                        from: p,
                        rows: n,
                        target,
                        bulk: false,
                    },
                );
                let decoder = !self.config.decoder_skip
                    || crate::tick::runs_decoder(p, n, target, self.config.decoder_warm);
                self.lanes[lane].skip_end =
                    crate::tick::skip_end_after(self.lanes[lane].skip_end, p, n, decoder);
                if let Some(group) = self.group.as_ref()
                    && let Err(error) = group.rows(lane, n)
                {
                    group.out_of_step(&error.to_string());
                }
                let mut ahead = self.ahead();
                let slot = self.slots[lane]
                    .as_mut()
                    .expect("a lane serving a prompt has its slot");
                let part = self.runtime.run_part_rows_hooked(slot, n, Some(&mut ahead));
                if let Some(plan) = ahead.plan {
                    self.ahead = Some((ahead.decoding, plan));
                }
                match part {
                    Ok(part) => {
                        p += n;
                        ran += n;
                        if let Part::Done(step) = part {
                            self.finish_prompt(index, step, events);
                            break;
                        }
                    }
                    Err(error) => {
                        if let Some(group) = self.group.as_ref() {
                            group.out_of_step(&error.to_string());
                        }
                        self.fail(index, error, events);
                        return ran;
                    }
                }
            }
        }
        ran
    }

    /// The chunks a bulk tick may run at once: [TickConfig::ways] with a bulk path, and one for
    /// a group.
    fn bulk_ways(&self) -> usize {
        if self.group.is_some() || self.config.bulk_cap.is_none() {
            1
        } else {
            self.config.ways.max(1)
        }
    }

    /// Ends the prompts of the `picked` requests (crate::tick::together) in one paired pass:
    /// each one's rest a whole chunk on its own lane, each sampling its first token. Returns the
    /// rows run.
    fn prefill_together(
        &mut self,
        now: u64,
        picked: &[usize],
        events: &mut Vec<TickEvent>,
    ) -> usize {
        let mut chunks = Vec::with_capacity(picked.len());
        for &index in picked {
            let Phase::Prefill { target, .. } = self.requests[index].phase else {
                continue;
            };
            let Some(lane) = self.requests[index].lane else {
                continue;
            };
            let Some(from) = self.slots[lane]
                .as_ref()
                .map(RuntimeSlot::history)
                .map(<[u32]>::len)
            else {
                continue;
            };
            chunks.push((index, lane, from, target));
        }
        for &(index, lane, from, target) in &chunks {
            let rows = target - from;
            self.record(
                now,
                Decision::Prefill {
                    seq: self.requests[index].id.0,
                    from,
                    rows,
                    target,
                    bulk: true,
                },
            );
            let decoder = !self.config.decoder_skip
                || crate::tick::runs_decoder(from, rows, target, self.config.decoder_warm);
            self.lanes[lane].skip_end =
                crate::tick::skip_end_after(self.lanes[lane].skip_end, from, rows, decoder);
        }
        let mut ahead = self.ahead();
        // The slots leave their lanes for the pass, in the picked order, and go straight back.
        let mut held: Vec<RuntimeSlot<W>> = chunks
            .iter()
            .filter_map(|&(_, lane, ..)| self.slots[lane].take())
            .collect();
        let ran = if held.len() == chunks.len() {
            let mut slots: Vec<&mut RuntimeSlot<W>> = held.iter_mut().collect();
            self.runtime.run_parts_paired(&mut slots, Some(&mut ahead))
        } else {
            Err(BackendError::new(
                "tick",
                "a lane ending its prompt has no slot",
            ))
        };
        for (slot, &(_, lane, ..)) in held.drain(..).zip(&chunks) {
            self.slots[lane] = Some(slot);
        }
        if let Some(plan) = ahead.plan {
            self.ahead = Some((ahead.decoding, plan));
        }
        match ran {
            Ok(steps) => {
                let mut rows = 0;
                for (step, &(index, lane, from, target)) in steps.into_iter().zip(&chunks) {
                    let tokens = self.requests[index].tokens[from..target].to_vec();
                    self.lanes[lane].resident.extend_from_slice(&tokens);
                    rows += target - from;
                    self.finish_prompt(index, step, events);
                }
                rows
            }
            Err(error) => {
                // The pass fails whole: every prompt in it fails with it.
                for &(index, ..) in &chunks {
                    self.fail(index, BackendError::new("tick", error.to_string()), events);
                }
                0
            }
        }
    }

    /// The tick-ahead hook for the next prefill: its decode-row count is fixed at the tick's
    /// start, and the plan it decides is used by the tick's decode.
    fn ahead(&self) -> Ahead {
        Ahead {
            decoding: self.fed.len(),
            batch_spec: self.config.batch_spec,
            shared_rounds: self.config.shared_rounds,
            members: self.config.members,
            drafter: W::Drafter::PRESENT && self.walk().drafter().drafts(),
            plan: None,
        }
    }

    fn finish_prompt(&mut self, index: usize, step: StepOutput, events: &mut Vec<TickEvent>) {
        let request = &mut self.requests[index];
        request.phase = Phase::Decode;
        let id = request.id;
        events.push(TickEvent::Prompt { id, step });
    }

    /// Runs this tick's decode rows: the plan from crate::tick::decode_plan, executed by the
    /// runtime's plain or round step.
    fn decode(&mut self, now: u64, events: &mut Vec<TickEvent>) {
        if self.fed.is_empty() {
            return;
        }
        let ids: Vec<RequestId> = self
            .requests
            .iter()
            .filter(|request| !request.done && request.phase == Phase::Decode)
            .map(|request| request.id)
            .collect();
        let mut rows: Vec<Fed> = Vec::with_capacity(self.fed.len());
        for id in ids {
            if let Some(position) = self.fed.iter().position(|fed| fed.id == id) {
                rows.push(self.fed.remove(position));
            }
        }
        self.fed.clear();
        if rows.is_empty() {
            return;
        }
        // The tick-ahead plan is a decision, not an execution: the hook may only name the
        // plan, so a plan that would run nothing is re-decided here and every nonempty
        // decode still runs its row.
        let plan = self
            .ahead
            .take()
            .filter(|(decided, _)| *decided == rows.len())
            .map(|(_, plan)| plan)
            .filter(|plan| !matches!(plan, DecodePlan::None))
            .unwrap_or_else(|| self.plan(rows.len()));
        match plan {
            DecodePlan::None => {}
            DecodePlan::Spec => {
                let (seqs, walked) = self.step_rows(rows, StepMode::Round { joint: false }, events);
                self.record(
                    now,
                    Decision::Decode {
                        seqs,
                        rows: walked,
                        plan: "spec",
                    },
                );
            }
            DecodePlan::Shared => {
                let (seqs, walked) = self.step_rows(rows, StepMode::Round { joint: true }, events);
                self.record(
                    now,
                    Decision::Decode {
                        seqs,
                        rows: walked,
                        plan: "shared",
                    },
                );
            }
            DecodePlan::Plain { .. } => {
                let mut seqs = 0;
                let mut walked = 0;
                let mut rest = rows;
                while !rest.is_empty() {
                    let take = self.config.members.min(rest.len());
                    let group: Vec<Fed> = rest.drain(..take).collect();
                    let (stepped, rows) = self.step_rows(group, StepMode::Plain, events);
                    seqs += stepped;
                    walked += rows;
                }
                self.record(
                    now,
                    Decision::Decode {
                        seqs,
                        rows: walked,
                        plan: "plain",
                    },
                );
            }
        }
    }

    /// The tick's decode plan: the one the prefill walk decided ahead when it fits the rows this
    /// tick can run, else crate::tick::decode_plan over them.
    fn plan(&self, rows: usize) -> DecodePlan {
        plan_for(
            rows,
            self.config.batch_spec,
            self.config.shared_rounds,
            self.config.members,
            W::Drafter::PRESENT && self.walk().drafter().drafts(),
        )
    }

    /// One runtime step over group, and every held answer after it. Returns the sequences and
    /// the rows the step (not the held answers) walked.
    fn step_rows(
        &mut self,
        group: Vec<Fed>,
        mode: StepMode,
        events: &mut Vec<TickEvent>,
    ) -> (usize, usize) {
        let mut members = Vec::with_capacity(group.len());
        let mut lanes = Vec::with_capacity(group.len());
        let mut stepping = Vec::with_capacity(group.len());
        for fed in &group {
            let Some(lane) = self
                .requests
                .iter()
                .find(|request| request.id == fed.id)
                .and_then(|request| request.lane)
            else {
                continue;
            };
            let Some(slot) = self.slots[lane].take() else {
                continue;
            };
            lanes.push(lane);
            stepping.push(*fed);
            members.push((slot, fed.token));
        }
        let seqs = members.len();
        // The shape rank 0's plan chose, and the members in this step's row order, travel before
        // the step: a worker runs the runtime's plain or round entry with the same shape instead
        // of deciding one of its own.
        if let Some(rank0) = self.group.as_ref() {
            let tokens: Vec<u32> = stepping.iter().map(|fed| fed.token).collect();
            if let Err(error) = rank0.step_mode(&lanes, &tokens, mode) {
                rank0.out_of_step(&error.to_string());
            }
        }
        let answer = self.runtime.step_mode_slots(&mut members, mode);
        // A step that ran appends the fed token to the history; the request's token list
        // follows it. A step that failed appended nothing.
        if answer.is_ok() {
            for (index, (_, token)) in members.iter().enumerate() {
                if let Some(fed) = stepping.get(index)
                    && let Some(request) = self
                        .requests
                        .iter_mut()
                        .find(|request| request.id == fed.id)
                {
                    request.tokens.push(*token);
                }
            }
        }
        let walked = match &answer {
            Ok((_, walked)) => *walked,
            Err(_) => 0,
        };
        // Put every slot back where it came from before the held answers run.
        for (lane, (slot, _)) in lanes.iter().copied().zip(members) {
            self.slots[lane] = Some(slot);
        }
        match answer {
            Ok((steps, _)) => {
                for (index, step) in steps.into_iter().enumerate() {
                    let Some(fed) = stepping.get(index).copied() else {
                        break;
                    };
                    let mut run = vec![step];
                    // The answers the arm already holds: they cost no row and reach the caller
                    // in the same reply (the old fused engine's run).
                    while let Some(token) = run.last().and_then(|step| step.next_token) {
                        let Some(lane) = self
                            .requests
                            .iter()
                            .find(|request| request.id == fed.id)
                            .and_then(|request| request.lane)
                        else {
                            break;
                        };
                        let ready = self.slots[lane]
                            .as_ref()
                            .is_some_and(|slot| self.runtime.ready_slot(slot, token));
                        if !ready {
                            break;
                        }
                        let Some(slot) = self.slots[lane].take() else {
                            break;
                        };
                        let mut member = [(slot, token)];
                        let held = self.runtime.step_slots(&mut member);
                        let [(slot, _)] = member;
                        self.slots[lane] = Some(slot);
                        match held.and_then(|mut steps| {
                            steps.pop().ok_or_else(|| {
                                BackendError::new("tick", "a held answer answered no step")
                            })
                        }) {
                            Ok(step) => run.push(step),
                            Err(_) => break,
                        }
                    }
                    events.push(TickEvent::Decode {
                        id: fed.id,
                        steps: run,
                    });
                }
            }
            Err(error) => {
                // The step frame went out: the workers are inside the same walk, so a failure
                // here ends the group rather than serving the other requests on.
                if let Some(rank0) = self.group.as_ref() {
                    rank0.out_of_step(&error.to_string());
                }
                for fed in group {
                    self.fail_by_id(fed.id, error.clone(), events);
                }
            }
        }
        (seqs, walked)
    }

    fn fail_by_id(&mut self, id: RequestId, error: BackendError, events: &mut Vec<TickEvent>) {
        if let Some(index) = self.requests.iter().position(|request| request.id == id) {
            self.fail(index, error, events);
        } else {
            events.push(TickEvent::Failed { id, error });
        }
    }
}

/// A serving engine: the tick engine on its own thread, with a handle a request blocks on.
///
/// The host driver opens one session a request; a session posts a prompt or a decode row to the
/// thread and waits for its answer, exactly as the fused engine's handles do. What differs is
/// what the thread does with a batch of them: the tick engine places the prompts by
/// crate::tick::pick_lane, orders them by crate::tick::prefill_order, sizes their chunks from
/// crate::tick::bulk_rows and runs the plan crate::tick::decode_plan picks.
pub struct TickServe<W: Walk> {
    jobs: Sender<ServeJob>,
    free: Mutex<Vec<usize>>,
    wake: Condvar,
    thread: Mutex<Option<JoinHandle<()>>>,
    stats: Vec<Arc<Mutex<Record>>>,
    capacity: usize,
    marker: PhantomData<fn() -> W>,
}

/// A request's record after its last job, refreshed by the engine thread.
#[derive(Debug, Clone, Default)]
struct Record {
    kept: Kept,
    speculation: Speculation,
}

/// Where a job's answer goes.
type Reply = Sender<Result<Vec<StepOutput>, BackendError>>;

/// What one job asks the engine for.
enum Work {
    /// Consume a prompt. sampling is applied to the sequence before its prompt runs.
    Prefill {
        tokens: Vec<u32>,
        images: Vec<VisionImage>,
        sampling: Sampling,
    },
    /// Consume one token of a request whose prompt is in.
    Decode { token: u32 },
    /// The request is over: its lane may serve another.
    Park,
    /// The handle never used the last count answers of its run.
    GiveBack { count: usize },
    /// End the engine thread.
    Shutdown,
}

/// One job and where its answer goes.
struct ServeJob {
    handle: usize,
    work: Work,
    reply: Reply,
}

/// The wait for a batch to fill, as the fused engine's thread waits (FILL_WINDOW).
const TICK_FILL_WINDOW: Duration = Duration::from_micros(300);

impl<W> TickServe<W>
where
    W: Walk + Send + Sync,
    W::Seq: Send,
{
    /// Allocates capacity lanes and starts the thread that ticks them.
    ///
    /// # Errors
    /// A lane the arm cannot open, an incomplete boot ledger, or a thread that cannot start.
    pub fn start(walk: W, capacity: usize) -> Result<Arc<Self>, BackendError> {
        Self::start_engine(TickEngine::new(walk, capacity)?, capacity)
    }

    /// Allocates capacity lanes and starts the thread as rank 0 of a tensor-parallel group.
    ///
    /// The engine is the tick engine; every prompt start, chunk, step, give-back and release is
    /// broadcast through `hub` before it runs, and each worker replays it
    /// ([crate::mirror_tick_lanes]) over its own `capacity` lanes, so lane `n` is the same sequence
    /// on every rank. The row
    /// exchange is on ([TickEngine::with_group]), so the prompt path takes the arm's bulk chunks;
    /// what stays rank 0's is every scheduler decision.
    ///
    /// # Errors
    /// As [Self::start], or a watch that cannot start.
    pub fn start_group(
        walk: W,
        capacity: usize,
        hub: Tp4ControlHub,
        wire: impl GroupWire,
    ) -> Result<Arc<Self>, BackendError> {
        let group = TickGroup::new(watched_hub(hub)?, Box::new(wire));
        Self::start_engine(TickEngine::new(walk, capacity)?.with_group(group), capacity)
    }

    /// [Self::start] with the engine already built: the serving thread and its handle table.
    fn start_engine(engine: TickEngine<W>, capacity: usize) -> Result<Arc<Self>, BackendError> {
        let ledger = engine
            .ledger()
            .map_err(|why| BackendError::new("tick", why))?;
        eprintln!("sparkflash serve: {ledger}");
        let stats: Vec<Arc<Mutex<Record>>> = (0..capacity)
            .map(|_| Arc::new(Mutex::new(Record::default())))
            .collect();
        let (jobs, inbox) = channel::<ServeJob>();
        let thread_stats = stats.clone();
        let thread = std::thread::Builder::new()
            .name("sparkflash-tick".to_owned())
            .spawn(move || serve(engine, inbox, thread_stats, capacity))
            .map_err(|error| BackendError::new("tick", error.to_string()))?;
        Ok(Arc::new(Self {
            jobs,
            free: Mutex::new((0..capacity).collect()),
            wake: Condvar::new(),
            thread: Mutex::new(Some(thread)),
            stats,
            capacity,
            marker: PhantomData,
        }))
    }

    /// Takes a handle for one request, blocking while every handle is in use.
    ///
    /// # Errors
    /// A poisoned handle list.
    pub fn open(self: &Arc<Self>) -> Result<TickSession<W>, BackendError> {
        let mut free = self
            .free
            .lock()
            .map_err(|_| BackendError::new("tick", "the handle list is poisoned"))?;
        while free.is_empty() {
            free = self
                .wake
                .wait(free)
                .map_err(|_| BackendError::new("tick", "the handle list is poisoned"))?;
        }
        let handle = free.remove(0);
        Ok(TickSession {
            handle,
            serve: Arc::clone(self),
            stats: Arc::clone(&self.stats[handle]),
            sampling: Sampling::default(),
            fresh: true,
            fed: Vec::new(),
            kept: Kept::default(),
            run: VecDeque::new(),
            next: None,
        })
    }

    /// How many requests may be in flight at once.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<W: Walk> Drop for TickServe<W> {
    fn drop(&mut self) {
        let (reply, _) = channel();
        let _ = self.jobs.send(ServeJob {
            handle: 0,
            work: Work::Shutdown,
            reply,
        });
        if let Ok(mut thread) = self.thread.lock()
            && let Some(thread) = thread.take()
        {
            let _ = thread.join();
        }
    }
}

/// A request's session: a handle that posts to the tick engine's thread.
pub struct TickSession<W: Walk> {
    handle: usize,
    serve: Arc<TickServe<W>>,
    stats: Arc<Mutex<Record>>,
    sampling: Sampling,
    /// Whether this handle has posted a prompt yet.
    fresh: bool,
    /// The ids this handle fed its request.
    fed: Vec<u32>,
    /// What the request held after this handle's last job.
    kept: Kept,
    /// The rest of the last decode's run, answered here while the caller feeds the token each
    /// answer chose.
    run: VecDeque<StepOutput>,
    next: Option<u32>,
}

impl<W: Walk> TickSession<W> {
    /// Gives the unused rest of the last run back before the next job, and says whether there
    /// was any.
    fn drop_run(&mut self) -> bool {
        let count = self.run.len();
        self.run.clear();
        self.next = None;
        if count > 0 {
            self.post(Work::GiveBack { count });
        }
        count > 0
    }

    fn ask(&mut self, work: Work) -> Result<Vec<StepOutput>, BackendError> {
        let (reply, answer) = channel();
        self.serve
            .jobs
            .send(ServeJob {
                handle: self.handle,
                work,
                reply,
            })
            .map_err(|_| BackendError::new("tick", "the engine thread has stopped"))?;
        let answer = answer
            .recv()
            .map_err(|_| BackendError::new("tick", "the engine thread has stopped"))?;
        if let Ok(cell) = self.stats.lock() {
            self.kept.clone_from(&cell.kept);
        }
        answer
    }

    fn post(&self, work: Work) {
        let (reply, _) = channel();
        let _ = self.serve.jobs.send(ServeJob {
            handle: self.handle,
            work,
            reply,
        });
    }

    fn prefix(&self, length: Option<usize>) -> Option<&[u32]> {
        if self.fresh {
            return None;
        }
        length
            .filter(|length| *length <= self.fed.len())
            .map(|length| &self.fed[..length])
    }

    fn live(&self) -> Record {
        self.stats
            .lock()
            .map(|cell| cell.clone())
            .unwrap_or_default()
    }
}

impl<W: Walk> Drop for TickSession<W> {
    fn drop(&mut self) {
        // Parked before the handle goes back: the jobs run in order, so the next request's
        // prompt finds the lane free and its history kept.
        self.drop_run();
        self.post(Work::Park);
        if let Ok(mut free) = self.serve.free.lock() {
            free.push(self.handle);
            self.serve.wake.notify_one();
        }
    }
}

impl<W> ModelSession for TickSession<W>
where
    W: Walk + Send + Sync,
    W::Seq: Send,
{
    fn set_sampling(&mut self, sampling: Sampling) {
        self.sampling = sampling;
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        self.prefill_images(tokens, &[])
    }

    fn prefill_images(
        &mut self,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<StepOutput, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::new("tick", "an empty prompt has no logits"));
        }
        self.fresh = false;
        self.drop_run();
        let answer = self
            .ask(Work::Prefill {
                tokens: tokens.to_vec(),
                images: images.to_vec(),
                sampling: self.sampling,
            })
            .and_then(first);
        if let Ok(step) = &answer {
            self.fed.clear();
            self.fed.extend_from_slice(tokens);
            self.next = step.next_token;
        }
        answer
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        if self.next == Some(token)
            && let Some(step) = self.run.pop_front()
        {
            self.fed.push(token);
            self.next = step.next_token;
            return Ok(step);
        }
        self.drop_run();
        let answer = self.ask(Work::Decode { token });
        match answer {
            Ok(steps) => {
                self.fed.push(token);
                let mut steps = VecDeque::from(steps);
                let step = steps.pop_front().unwrap_or_default();
                self.next = step.next_token;
                self.run = steps;
                Ok(step)
            }
            Err(error) => Err(error),
        }
    }

    fn speculation(&self) -> Speculation {
        self.stats
            .lock()
            .map(|record| record.speculation)
            .unwrap_or_default()
    }

    fn reusable_prefix(&self) -> Option<&[u32]> {
        self.prefix(
            self.live()
                .kept
                .history
                .and_then(|held| held.checked_sub(self.run.len())),
        )
    }

    fn rewind_prefix(&self) -> Option<&[u32]> {
        self.prefix(self.live().kept.rewind)
    }

    fn held_images(&self) -> &[ImageSpan] {
        &self.kept.images
    }

    fn cached_tokens(&self) -> usize {
        self.kept.cached
    }

    fn park(&mut self) {
        if self.drop_run() {
            let _ = self.ask(Work::Park);
        } else {
            self.post(Work::Park);
        }
    }
}

/// A prompt's reply is its one step.
fn first(steps: Vec<StepOutput>) -> Result<StepOutput, BackendError> {
    steps
        .into_iter()
        .next()
        .ok_or_else(|| BackendError::new("tick", "the engine answered no step"))
}

/// The engine thread: collect the waiting jobs, let the tick engine place and run them, and
/// answer every request the tick finished.
fn serve<W>(
    mut engine: TickEngine<W>,
    inbox: Receiver<ServeJob>,
    stats: Vec<Arc<Mutex<Record>>>,
    capacity: usize,
) where
    W: Walk + Send + Sync,
    W::Seq: Send,
{
    use std::collections::HashMap;

    let mut requests: Vec<Option<RequestId>> = vec![None; capacity];
    let mut waiting: HashMap<RequestId, Reply> = HashMap::new();
    loop {
        let first = if engine.busy() {
            match inbox.recv_timeout(TICK_FILL_WINDOW) {
                Ok(job) => Some(job),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => return,
            }
        } else {
            match inbox.recv() {
                Ok(job) => Some(job),
                Err(_) => return,
            }
        };
        let mut jobs: Vec<ServeJob> = first.into_iter().collect();
        // Everything already queued is one tick's batch.
        while let Ok(job) = inbox.try_recv() {
            jobs.push(job);
        }
        for job in jobs {
            match job.work {
                Work::Shutdown => return,
                Work::Prefill {
                    tokens,
                    images,
                    sampling,
                } => {
                    // A prompt on a handle ends the handle's last request, as a park does. Its
                    // lane keeps the history, so the prompt may continue it; left open, the old
                    // request would hold its lane for good, and a one-lane engine would wait
                    // for it forever.
                    if let Some(id) = requests[job.handle].take() {
                        engine.finish(id);
                    }
                    match engine.admit(&tokens, &images, sampling) {
                        Ok(id) => {
                            requests[job.handle] = Some(id);
                            waiting.insert(id, job.reply);
                        }
                        Err(error) => {
                            let _ = job.reply.send(Err(error));
                        }
                    }
                }
                Work::Decode { token } => {
                    let Some(id) = requests[job.handle] else {
                        let _ = job.reply.send(Err(BackendError::new(
                            "tick",
                            "the handle has no request to decode",
                        )));
                        continue;
                    };
                    match engine.feed(id, token) {
                        Ok(()) => {
                            waiting.insert(id, job.reply);
                        }
                        Err(error) => {
                            let _ = job.reply.send(Err(error));
                        }
                    }
                }
                Work::Park => {
                    if let Some(id) = requests[job.handle].take() {
                        engine.finish(id);
                    }
                    let _ = job.reply.send(Ok(Vec::new()));
                }
                Work::GiveBack { count } => {
                    if let Some(id) = requests[job.handle] {
                        let _ = engine.give_back(id, count);
                        refresh(&engine, id, Some(job.handle), &stats);
                    }
                    let _ = job.reply.send(Ok(Vec::new()));
                }
            }
        }
        if !engine.busy() {
            continue;
        }
        match engine.tick() {
            Ok(events) => {
                for event in events {
                    match event {
                        TickEvent::Prompt { id, step } => {
                            // The record is published before the reply: a waiting handle reads
                            // its kept lengths as soon as its answer arrives.
                            refresh(&engine, id, handle_of(id, &requests), &stats);
                            if let Some(reply) = waiting.remove(&id) {
                                let _ = reply.send(Ok(vec![step]));
                            }
                        }
                        TickEvent::Decode { id, steps } => {
                            refresh(&engine, id, handle_of(id, &requests), &stats);
                            if let Some(reply) = waiting.remove(&id) {
                                let _ = reply.send(Ok(steps));
                            }
                        }
                        TickEvent::Failed { id, error } => {
                            if let Some(reply) = waiting.remove(&id) {
                                let _ = reply.send(Err(error));
                            }
                            if let Some(handle) = handle_of(id, &requests) {
                                requests[handle] = None;
                            }
                        }
                    }
                }
            }
            Err(error) => {
                for (_, reply) in waiting.drain() {
                    let _ = reply.send(Err(error.clone()));
                }
                requests.fill(None);
            }
        }
    }
}

/// The handle a request belongs to, when one does.
fn handle_of(id: RequestId, requests: &[Option<RequestId>]) -> Option<usize> {
    requests.iter().position(|request| *request == Some(id))
}

/// Publishes what a request holds to its handle.
fn refresh<W: Walk>(
    engine: &TickEngine<W>,
    id: RequestId,
    handle: Option<usize>,
    stats: &[Arc<Mutex<Record>>],
) {
    let Some(handle) = handle else {
        return;
    };
    if let Some(record) = stats.get(handle)
        && let Ok(mut record) = record.lock()
    {
        record.kept = engine.kept(id).unwrap_or_default();
        record.speculation = engine.speculation(id);
    }
}

/// SPARKFLASH_TICK_TRACE's path, when it names one.
fn trace_path() -> Option<PathBuf> {
    let value = std::env::var_os("SPARKFLASH_TICK_TRACE")?;
    let path = PathBuf::from(value);
    (!path.as_os_str().is_empty()).then_some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fused engine's switch went with the engine. What it asked for that the tick engine
    /// runs is accepted; the rest is refused by name rather than ignored.
    #[test]
    fn the_retired_prompt_parts_switch_accepts_what_the_engine_runs_and_refuses_the_rest() {
        for on in [None, Some(""), Some("on")] {
            assert_eq!(prompt_parts_request(on), Ok(()), "{on:?}");
        }
        let off = prompt_parts_request(Some("off")).expect_err("off asked for whole prompts");
        assert!(off.contains(PROMPT_PARTS_ENV), "{off}");
        assert!(off.contains("retired"), "{off}");
        let other = prompt_parts_request(Some("0")).expect_err("not a switch value");
        assert!(other.contains(PROMPT_PARTS_ENV), "{other}");
    }
}

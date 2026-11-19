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

//! DeepSeek's tick, and the decisions it makes, moved here.
//!
//! [crate::share] already took the row share; this module takes the rest of the scheduler's
//! *decisions* out of SparkFlash-DeepSeek-V4.1-Flash's `tp4_forward`:
//!
//! * the queue and lane order ([pick_lane], [reusable_shared], [common_prefix]);
//! * the prompt order ([prefill_order]) and the decode share ([DecodeShare],
//!   [hold_prefill]);
//! * the chunk boundaries ([bulk_rows], [tick_rows], [bulk_budget], [bulk_from],
//!   [whole_chunk_fits]) and the decoder-layer skip ([runs_decoder], [skip_end_after],
//!   [decoder_resume]);
//! * the step's shape ([decode_plan]) and the band model it shares rows over ([StepModel],
//!   [band], [batch_step_ms]);
//! * the tick trace ([TickTrace], [InjectedClock], [replay]) that pins all of the above.
//!
//! The scheduler itself - `Batch::place`, `Batch::prefill`, `Batch::decode`,
//! `Batch::spec_step` and `Batch::plain_step` - still lives in the forward crate, because it
//! drives the device (`DeviceLayer`, `RowsState`, `DraftState`, Engram and the head). What is
//! here is every branch that can be decided without the device, unchanged: the callers pass the
//! values they used to read from their own fields, so the decisions and their order are the same.
//!
//! [replay] is the gate: a fixed workload, an injected clock and a committed trace. The
//! scheduler's time-based decisions are compared as decisions, never as wall time.

use sparkflash_model::vision::{ImageSpan, VisionImage};

/// Sequences that may wait for a lane, past the lanes themselves.
pub const QUEUE: usize = 64;
/// The fewest positions a sequence takes from another lane ([crate::tick::common_prefix]'s
/// caller, `Batch::adopt`).
pub const ADOPT_MIN: usize = 64;
/// A prefill with at most this many rows left never waits for the decoders.
pub const SHARE_MIN_ROWS: usize = 16384;
/// The fewest shared rows a match on a lane holding a long prefix must have to be reused
/// (SPARKFLASH_PREFIX_REUSE_MIN_ROWS).
pub const PREFIX_REUSE_MIN_ROWS: usize = 256;

/// The rows the shared step's cost model bends at: a step of up to this many rows is the
/// narrow body, past it the wide one (the bucket split the model is fit over).
pub const BAND_ROWS: usize = 16;
/// What a shared step costs for each sequence past the first, besides its anchor row (ms).
pub const SEQ_MS: f64 = 2.3;

/// A shared step's cost model: a0 + a1 * rows up to [BAND_ROWS] rows, b0 + b1 * rows above,
/// and seq_ms for each sequence past the first (ms).
pub type StepModel = (f64, f64, f64, f64, f64);

/// The modelled wall of a shared step over the given number of verified rows (ms), for the tests
/// that pin the model.
#[must_use]
pub fn batch_step_ms(model: StepModel, rows: usize) -> f64 {
    band(model).rows_ms(rows)
}

/// The step model as the shared row share's wall ([crate::share::BandWall]).
#[must_use]
pub fn band(model: StepModel) -> crate::share::BandWall {
    let (a0, a1, b0, b1, seq_ms) = model;
    crate::share::BandWall {
        a0,
        a1,
        b0,
        b1,
        seq_ms,
        band_rows: BAND_ROWS,
    }
}

/// The decoders' share of the device's time while long prompts prefill.
#[derive(Clone, Copy, Debug)]
pub struct DecodeShare {
    pub most: f64,
    pub deadline_ms: f64,
}

impl DecodeShare {
    /// The share named by `most` and `deadline_ms`, clamped as the profile's value was.
    #[must_use]
    pub fn new(most: f64, deadline_ms: f64) -> Self {
        Self {
            most: most.clamp(0.0, 0.9),
            deadline_ms,
        }
    }

    /// The share of time the decoders may take: `prefills` are the prompts still prefilling in
    /// the order they run, each as (rows left, ms since admission), at `row_ms` a row.
    #[must_use]
    pub fn allowed(&self, prefills: &[(usize, f64)], row_ms: f64) -> f64 {
        let mut need = 0.0;
        let mut share = self.most;
        for &(left, age_ms) in prefills {
            need += left as f64 * row_ms;
            let budget = self.deadline_ms - age_ms;
            if budget <= need {
                return 0.0;
            }
            share = share.min(1.0 - need / budget);
        }
        share
    }
}

/// Whether the decoders have had less than their share when `since_ms` passed since the last
/// long chunk. `order` is [prefill_order]'s result and `age_ms` reads a sequence index's
/// admission age.
#[must_use]
pub fn hold_prefill(
    share: &DecodeShare,
    order: &[(usize, usize)],
    age_ms: impl Fn(usize) -> f64,
    decoding: bool,
    row_ms: f64,
    chunk_ms: f64,
    since_ms: f64,
) -> bool {
    if row_ms == 0.0
        || order
            .first()
            .is_none_or(|&(left, _)| left <= SHARE_MIN_ROWS)
        || !decoding
    {
        return false;
    }
    let prefills: Vec<(usize, f64)> = order
        .iter()
        .map(|&(left, index)| (left, age_ms(index)))
        .collect();
    let share = share.allowed(&prefills, row_ms);
    share > 0.0 && since_ms < chunk_ms * share / (1.0 - share)
}

/// The prefilling sequences in the order their chunks run, each as (rows left, index): the
/// fewest rows left first (ties to the oldest), so a turn that reuses a cached prefix does not
/// wait behind a long cold prompt.
#[must_use]
pub fn prefill_order(prefills: impl IntoIterator<Item = (usize, usize)>) -> Vec<(usize, usize)> {
    let mut order: Vec<(usize, usize)> = prefills.into_iter().collect();
    order.sort_unstable();
    order
}

/// A lane [pick_lane] may place a sequence on: the rows it shares with the sequence and when it
/// was last given up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneCandidate {
    pub lane: usize,
    pub shared: usize,
    pub used: u64,
}

/// The free lane a sequence takes: the longest shared prefix, else the least recently used
/// (ties to the lowest lane).
#[must_use]
pub fn pick_lane(candidates: impl IntoIterator<Item = LaneCandidate>) -> Option<LaneCandidate> {
    candidates.into_iter().max_by(|a, b| {
        a.shared
            .cmp(&b.shared)
            .then_with(|| b.used.cmp(&a.used))
            .then_with(|| b.lane.cmp(&a.lane))
    })
}

/// The shared rows a placement credits a lane with: a match below `min_rows` against a lane
/// holding at least that many rows is worth more kept for a longer turn, so it counts as no
/// match. `0` credits every match.
#[must_use]
pub fn reusable_shared(raw_shared: usize, resident_len: usize, min_rows: usize) -> usize {
    if raw_shared < min_rows && resident_len >= min_rows {
        0
    } else {
        raw_shared
    }
}

/// How many leading positions of `wanted` a lane holding `held` (its image spans `spans`)
/// holds alike: the common prefix, cut at the first of the prompt's `images` the lane does not
/// hold alike (equal span tokens are not equal images).
#[must_use]
pub fn common_prefix(
    held: &[u32],
    spans: &[ImageSpan],
    wanted: &[u32],
    images: &[VisionImage],
) -> usize {
    let shared = held
        .iter()
        .zip(wanted)
        .take_while(|(held, want)| held == want)
        .count();
    images
        .iter()
        .map(|image| image.span)
        .find(|span| span.start < shared && !spans.contains(span))
        .map_or(shared, |span| span.start)
}

/// The rows of the next bulk chunk with `left` rows of the prompt to go: `budget`, or all of
/// them when they fit `most`, so no short tail takes a chunk of its own. With `split` (the
/// decoder's `warm` rows, when the tick runs another chunk after this one), the rows before the
/// prompt's last `split` take this chunk once they fit it. `rows_max` is R_MAX.
#[must_use]
pub fn bulk_rows(
    left: usize,
    budget: usize,
    most: usize,
    split: Option<usize>,
    rows_max: usize,
) -> usize {
    let early = split.map_or(0, |warm| left.saturating_sub(warm));
    if early > rows_max && early <= budget {
        early
    } else if left <= most {
        left
    } else {
        budget
    }
}

/// Whether prompt rows p..p+n of a prefill to `target` run the decoder layers: the chunks that
/// reach into its last `warm` rows.
#[must_use]
pub fn runs_decoder(p: usize, n: usize, target: usize, warm: usize) -> bool {
    p + n + warm > target
}

/// A lane's skip end once rows p..p+n ran the decoder layers (`decoder`) or skipped them. A
/// decoder row below the old end starts a run of its own.
#[must_use]
pub fn skip_end_after(skip_end: usize, p: usize, n: usize, decoder: bool) -> usize {
    if decoder { skip_end.min(p) } else { p + n }
}

/// Where a prefill to `target` resumes a lane whose prefix it shares up to `start`, given the
/// lane's skip end and the decoder `warm` rows.
#[must_use]
pub fn decoder_resume(skip_end: usize, start: usize, target: usize, warm: usize) -> usize {
    if skip_end == 0 || start >= skip_end + warm || target - start >= warm {
        start
    } else {
        target.saturating_sub(warm)
    }
}

/// Whether a prefill from `start` to `target` takes the bulk chunks.
#[must_use]
pub fn bulk_from(
    target: usize,
    start: usize,
    has_bulk: bool,
    has_exchange: bool,
    exact: bool,
    chunk_min: usize,
) -> bool {
    has_bulk && has_exchange && !exact && target - start >= chunk_min
}

/// The scheduler's row caps, which [tick_rows] and [bulk_budget] read.
#[derive(Clone, Copy, Debug)]
pub struct TickRows {
    pub busy_chunk: usize,
    pub busy_exact_steps: usize,
    pub exact_tick_rows: usize,
    pub rows_max: usize,
    pub chunk_min: usize,
}

/// The most rows one tick's prefill of a sequence runs (a bulk chunk, or exact rows steps).
#[must_use]
pub fn tick_rows(cfg: &TickRows, bulk_cap: Option<usize>, bulk: bool, decoding: bool) -> usize {
    match bulk_cap {
        Some(cap) if bulk && decoding => (cfg.busy_chunk + cfg.chunk_min).min(cap),
        Some(cap) if bulk => cap,
        _ if decoding => cfg.busy_exact_steps * cfg.rows_max,
        _ => cfg.exact_tick_rows,
    }
}

/// The bulk chunk a tick runs and the most it may stretch to ([bulk_rows]).
#[must_use]
pub fn bulk_budget(cfg: &TickRows, bulk_cap: Option<usize>, decoding: bool) -> (usize, usize) {
    let most = tick_rows(cfg, bulk_cap, true, decoding);
    match bulk_cap {
        Some(cap) if decoding => (cfg.busy_chunk.min(cap), most),
        _ => (most, most),
    }
}

/// Whether a prompt's rest fits one bulk chunk however its tick runs it: more than `rows_max`
/// rows, at most the budget, and too few to split off the decoder's.
#[must_use]
pub fn whole_chunk_fits(
    left: usize,
    budget: usize,
    decoder_skip: bool,
    decoder_warm: usize,
    rows_max: usize,
) -> bool {
    let most = if decoder_skip {
        budget.min(decoder_warm + rows_max)
    } else {
        budget
    };
    left > rows_max && left <= most
}

/// The shape of one decode step, from the sequences that are decoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodePlan {
    /// Nothing decodes.
    None,
    /// One sequence takes the speculative step.
    Spec,
    /// The group shares one speculative step ([crate::share_rows]).
    Shared,
    /// Plain rows steps, one row a sequence, in this many groups of at most `rows_max`.
    Plain { groups: usize },
}

/// Which decode step runs, from the number of decoding sequences and the switches that choose
/// it. Same order and same conditions as the scheduler always had.
#[must_use]
pub fn decode_plan(
    decoding: usize,
    constrained: bool,
    batch_spec: Option<f64>,
    exchange: bool,
    rows_max: usize,
) -> DecodePlan {
    match decoding {
        0 => DecodePlan::None,
        1 => DecodePlan::Spec,
        many if constrained => DecodePlan::Plain {
            groups: many.div_ceil(rows_max),
        },
        many if batch_spec.is_some() && exchange && 2 * many <= rows_max => DecodePlan::Shared,
        many => DecodePlan::Plain {
            groups: many.div_ceil(rows_max),
        },
    }
}

/// A monotonic millisecond clock. The tick trace takes one instead of reading wall time, so a
/// replay's boundary decisions are the same on every host.
pub trait Clock {
    fn now_ms(&self) -> u64;
}

/// A clock the replay advances by a scripted amount.
#[derive(Debug)]
pub struct InjectedClock {
    now: std::cell::Cell<u64>,
}

impl InjectedClock {
    #[must_use]
    pub fn new(start_ms: u64) -> Self {
        Self {
            now: std::cell::Cell::new(start_ms),
        }
    }

    pub fn advance(&self, ms: u64) {
        self.now.set(self.now.get() + ms);
    }
}

impl Clock for InjectedClock {
    fn now_ms(&self) -> u64 {
        self.now.get()
    }
}

/// One scheduler decision, as the trace records it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// A waiting sequence took a lane at `start` (its target and shared prefix included).
    Place {
        seq: u64,
        lane: usize,
        shared: usize,
        start: usize,
        target: usize,
        bulk: bool,
    },
    /// A prefill chunk ran.
    Prefill {
        seq: u64,
        from: usize,
        rows: usize,
        target: usize,
        bulk: bool,
    },
    /// The tick's decode step.
    Decode {
        seqs: usize,
        rows: usize,
        plan: &'static str,
    },
    /// Whether a long chunk was held back for the decoders.
    Hold { hold: bool },
}

impl Decision {
    fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        match *self {
            Self::Place {
                seq,
                lane,
                shared,
                start,
                target,
                bulk,
            } => {
                let _ = write!(
                    out,
                    "place seq={seq} lane={lane} shared={shared} start={start} target={target} path={}",
                    if bulk { "bulk" } else { "exact" }
                );
            }
            Self::Prefill {
                seq,
                from,
                rows,
                target,
                bulk,
            } => {
                let _ = write!(
                    out,
                    "prefill seq={seq} from={from} rows={rows} to={} target={target} path={}",
                    from + rows,
                    if bulk { "bulk" } else { "exact" }
                );
            }
            Self::Decode { seqs, rows, plan } => {
                let _ = write!(out, "decode seqs={seqs} rows={rows} plan={plan}");
            }
            Self::Hold { hold } => {
                let _ = write!(out, "hold {}", u8::from(hold));
            }
        }
        out
    }
}

/// One tick's decisions at one clock reading.
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub tick: u64,
    pub clock_ms: u64,
    pub decision: Decision,
}

/// The scheduler's decisions on one workload, in order, with the clock that produced them.
#[derive(Clone, Debug, Default)]
pub struct TickTrace {
    entries: Vec<Entry>,
}

impl TickTrace {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, tick: u64, clock_ms: u64, decision: Decision) {
        self.entries.push(Entry {
            tick,
            clock_ms,
            decision,
        });
    }

    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The trace as committed text: one decision a line, tick and injected clock first.
    #[must_use]
    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        for entry in &self.entries {
            let _ = writeln!(
                out,
                "t={} clock={} {}",
                entry.tick,
                entry.clock_ms,
                entry.decision.render()
            );
        }
        out
    }
}

/// A deterministic scheduler replay: the decisions [crate::tick] holds, over a scripted
/// workload and an injected clock. It models token state and lane residency, not the device, so
/// it runs in the CPU gate; the device scheduler (`Batch`) still calls the same functions.
pub mod replay {
    use std::collections::VecDeque;

    use sparkflash_model::speculation::{DepthPolicy, StepCost};
    use sparkflash_model::vision::ImageSpan;

    use super::{
        Clock, Decision, DecodePlan, DecodeShare, InjectedClock, LaneCandidate, SHARE_MIN_ROWS,
        StepModel, TickRows, TickTrace, band, bulk_budget, bulk_from, bulk_rows, common_prefix,
        decode_plan, decoder_resume, hold_prefill, pick_lane, prefill_order, reusable_shared,
        runs_decoder, skip_end_after, tick_rows, whole_chunk_fits,
    };
    use crate::share_rows;

    /// The constants and switches a replay models. The test fixes them; a serving run reads the
    /// same values from its profile and the device.
    #[derive(Clone, Copy, Debug)]
    pub struct Config {
        pub lanes: usize,
        pub rows_max: usize,
        pub decoder_warm: usize,
        pub exact_tick_rows: usize,
        pub chunk_min: usize,
        pub busy_chunk: usize,
        pub busy_exact_steps: usize,
        pub share: DecodeShare,
        pub prefix_reuse_min_rows: usize,
        pub bulk_cap: Option<usize>,
        pub reuse: bool,
        pub exchange: bool,
        pub decoder_skip: bool,
        pub batch_spec: Option<f64>,
        pub share_floor: f64,
        pub model: StepModel,
        pub k: usize,
    }

    impl Config {
        /// DeepSeek's serving defaults at eight lanes, as the profile and the device read them.
        #[must_use]
        pub fn deepseek(lanes: usize) -> Self {
            Self {
                lanes,
                rows_max: 64,
                decoder_warm: 2541,
                exact_tick_rows: 4096,
                chunk_min: 96,
                busy_chunk: 2048,
                busy_exact_steps: 1,
                share: DecodeShare::new(0.5, 200_000.0),
                prefix_reuse_min_rows: 256,
                bulk_cap: Some(4096),
                reuse: true,
                exchange: true,
                decoder_skip: true,
                batch_spec: Some(0.2),
                share_floor: 0.2,
                model: (23.0, 4.0, 23.0, 4.0, 2.3),
                k: 5,
            }
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

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Phase {
        Waiting,
        Prefill { target: usize, bulk: bool },
        Decode,
    }

    struct Seq {
        id: u64,
        tokens: Vec<u32>,
        prompt_len: usize,
        lane: Option<usize>,
        pos: usize,
        phase: Phase,
        admitted_ms: u64,
        done: bool,
    }

    impl Seq {
        fn emitted(&self) -> usize {
            self.tokens.len() - self.prompt_len
        }

        fn target(&self) -> usize {
            if self.emitted() == 0 {
                self.prompt_len
            } else {
                self.tokens.len() - 1
            }
        }
    }

    #[derive(Default)]
    struct Lane {
        resident: Vec<u32>,
        spans: Vec<ImageSpan>,
        seq: Option<u64>,
        used: u64,
        skip_end: usize,
    }

    /// A replayable scheduler state.
    pub struct Replay {
        cfg: Config,
        clock: InjectedClock,
        trace: TickTrace,
        tick: u64,
        lanes: Vec<Lane>,
        seqs: Vec<Seq>,
        /// Sequences to admit before a tick, oldest first.
        pending: VecDeque<(u64, u64, Vec<u32>)>,
        chunk_ms: f64,
        row_ms: f64,
        since_chunk_ms: f64,
    }

    impl Replay {
        #[must_use]
        pub fn new(cfg: Config) -> Self {
            Self {
                lanes: (0..cfg.lanes).map(|_| Lane::default()).collect(),
                cfg,
                clock: InjectedClock::new(0),
                trace: TickTrace::new(),
                tick: 0,
                seqs: Vec::new(),
                pending: VecDeque::new(),
                chunk_ms: 0.0,
                row_ms: 0.0,
                since_chunk_ms: 0.0,
            }
        }

        /// Admits `tokens` as sequence `id` before the next tick; `prompt_len` is how many of
        /// them are the prompt.
        pub fn admit(&mut self, id: u64, tokens: Vec<u32>, prompt_len: usize) {
            self.pending.push_back((id, prompt_len as u64, tokens));
        }

        /// Ends sequence `id` before the next tick.
        pub fn finish(&mut self, id: u64) {
            if let Some(index) = self.seqs.iter().position(|seq| seq.id == id) {
                self.seqs[index].done = true;
                if let Some(lane) = self.seqs[index].lane.take() {
                    self.lanes[lane].seq = None;
                    self.lanes[lane].used = self.tick;
                }
            }
        }

        #[must_use]
        pub fn trace(&self) -> &TickTrace {
            &self.trace
        }

        fn decoding(&self) -> bool {
            self.seqs
                .iter()
                .any(|seq| !seq.done && seq.phase == Phase::Decode)
        }

        fn prefill_order(&self) -> Vec<(usize, usize)> {
            prefill_order(
                self.seqs
                    .iter()
                    .enumerate()
                    .filter_map(|(index, seq)| match seq.phase {
                        Phase::Prefill { target, .. } if !seq.done => {
                            Some((target.saturating_sub(seq.pos), index))
                        }
                        _ => None,
                    }),
            )
        }

        /// Runs one tick whose whole cost is `cost_ms` (the injected clock advances by it).
        pub fn tick(&mut self, cost_ms: u64) {
            self.tick += 1;
            self.clock.advance(cost_ms);
            while let Some((id, prompt_len, tokens)) = self.pending.pop_front() {
                self.seqs.push(Seq {
                    id,
                    prompt_len: prompt_len as usize,
                    tokens,
                    lane: None,
                    pos: 0,
                    phase: Phase::Waiting,
                    admitted_ms: self.clock.now_ms(),
                    done: false,
                });
            }
            self.place();
            let decoding = self.decoding();
            let order = self.prefill_order();
            let now = self.clock.now_ms();
            let hold = hold_prefill(
                &self.cfg.share,
                &order,
                |index| (now - self.seqs[index].admitted_ms) as f64,
                decoding,
                self.row_ms,
                self.chunk_ms,
                self.since_chunk_ms,
            );
            self.trace
                .record(self.tick, self.clock.now_ms(), Decision::Hold { hold });
            let mut long_chunk = None;
            if let Some(&(_, index)) = order.first() {
                let left = match self.seqs[index].phase {
                    Phase::Prefill { target, .. } => target.saturating_sub(self.seqs[index].pos),
                    Phase::Waiting | Phase::Decode => 0,
                };
                if !(hold && decoding && left > SHARE_MIN_ROWS) {
                    long_chunk = self.prefill(index);
                }
            }
            self.decode();
            if let Some(ran) = long_chunk {
                self.chunk_ms = cost_ms as f64;
                self.row_ms = cost_ms as f64 / ran as f64;
                self.since_chunk_ms = 0.0;
            } else {
                self.since_chunk_ms += cost_ms as f64;
            }
            self.seqs.retain(|seq| !seq.done);
        }

        fn place(&mut self) {
            loop {
                let Some(index) = self
                    .seqs
                    .iter()
                    .position(|seq| !seq.done && seq.phase == Phase::Waiting)
                else {
                    return;
                };
                let target = self.seqs[index].target();
                let wanted_len = target;
                let free = (0..self.lanes.len()).filter(|&lane| self.lanes[lane].seq.is_none());
                let chosen = pick_lane(free.map(|lane| {
                    let held = &self.lanes[lane];
                    let raw_shared = if self.cfg.reuse {
                        let wanted = &self.seqs[index].tokens[..wanted_len];
                        common_prefix(&held.resident, &held.spans, wanted, &[])
                    } else {
                        0
                    };
                    LaneCandidate {
                        lane,
                        shared: reusable_shared(
                            raw_shared,
                            held.resident.len(),
                            self.cfg.prefix_reuse_min_rows,
                        ),
                        used: held.used,
                    }
                }));
                let Some(candidate) = chosen else {
                    return;
                };
                let (lane, shared) = (candidate.lane, candidate.shared);
                let cap = if self.seqs[index].emitted() == 0 {
                    target - 1
                } else {
                    target
                };
                let mut start = shared.min(cap);
                if self.cfg.decoder_skip {
                    start = decoder_resume(
                        self.lanes[lane].skip_end,
                        start,
                        target,
                        self.cfg.decoder_warm,
                    );
                }
                let bulk = bulk_from(
                    target,
                    start,
                    self.cfg.bulk_cap.is_some(),
                    self.cfg.exchange,
                    false,
                    self.cfg.chunk_min,
                );
                let seq = &mut self.seqs[index];
                let id = seq.id;
                seq.lane = Some(lane);
                seq.pos = start;
                seq.phase = if start == target {
                    Phase::Decode
                } else {
                    Phase::Prefill { target, bulk }
                };
                let used = self.tick;
                let lane_state = &mut self.lanes[lane];
                lane_state.seq = Some(id);
                lane_state.used = used;
                lane_state.resident.truncate(start);
                lane_state.spans.retain(|span| span.start < start);
                self.trace.record(
                    self.tick,
                    self.clock.now_ms(),
                    Decision::Place {
                        seq: id,
                        lane,
                        shared,
                        start,
                        target,
                        bulk,
                    },
                );
            }
        }

        fn prefill(&mut self, index: usize) -> Option<usize> {
            let Phase::Prefill { target, bulk } = self.seqs[index].phase else {
                return None;
            };
            let from = self.seqs[index].pos;
            let left = target - from;
            let lane = self.seqs[index].lane?;
            let id = self.seqs[index].id;
            let mut ran = 0;
            if bulk {
                let cap = self.cfg.bulk_cap;
                let (budget, most) = bulk_budget(&self.cfg.tick_rows(), cap, self.decoding());
                let fits = whole_chunk_fits(
                    left,
                    budget,
                    self.cfg.decoder_skip,
                    self.cfg.decoder_warm,
                    self.cfg.rows_max,
                );
                let rows = bulk_rows(
                    left,
                    if fits { most } else { budget },
                    most,
                    None,
                    self.cfg.rows_max,
                )
                .min(left);
                ran = rows;
                self.trace.record(
                    self.tick,
                    self.clock.now_ms(),
                    Decision::Prefill {
                        seq: id,
                        from,
                        rows,
                        target,
                        bulk: true,
                    },
                );
            } else {
                let steps = tick_rows(
                    &self.cfg.tick_rows(),
                    self.cfg.bulk_cap,
                    false,
                    self.decoding(),
                )
                .div_ceil(self.cfg.rows_max);
                let mut p = from;
                for _ in 0..steps {
                    if p == target {
                        break;
                    }
                    let n = self.cfg.rows_max.min(target - p);
                    self.trace.record(
                        self.tick,
                        self.clock.now_ms(),
                        Decision::Prefill {
                            seq: id,
                            from: p,
                            rows: n,
                            target,
                            bulk: false,
                        },
                    );
                    self.lanes[lane].skip_end = skip_end_after(
                        self.lanes[lane].skip_end,
                        p,
                        n,
                        !self.cfg.decoder_skip || runs_decoder(p, n, target, self.cfg.decoder_warm),
                    );
                    p += n;
                    ran += n;
                }
                self.seqs[index].pos = p;
            }
            if bulk {
                let end = from + ran;
                self.lanes[lane].skip_end = skip_end_after(
                    self.lanes[lane].skip_end,
                    from,
                    ran,
                    !self.cfg.decoder_skip
                        || runs_decoder(from, ran, target, self.cfg.decoder_warm),
                );
                self.seqs[index].pos = end;
                self.lanes[lane]
                    .resident
                    .extend_from_slice(&self.seqs[index].tokens[from..end]);
            }
            if self.seqs[index].pos >= target {
                self.seqs[index].phase = Phase::Decode;
            }
            Some(ran)
        }

        fn decode(&mut self) -> usize {
            let indices: Vec<usize> = self
                .seqs
                .iter()
                .enumerate()
                .filter(|(_, seq)| !seq.done && seq.phase == Phase::Decode)
                .map(|(index, _)| index)
                .collect();
            let plan = decode_plan(
                indices.len(),
                false,
                self.cfg.batch_spec,
                self.cfg.exchange,
                self.cfg.rows_max,
            );
            match plan {
                DecodePlan::None => 0,
                DecodePlan::Spec => {
                    let policy = DepthPolicy::adaptive(
                        self.cfg.k,
                        StepCost {
                            step_ms: 27.0,
                            row_ms: 4.0,
                            prior_tokens_per_ms: 0.06,
                        },
                    );
                    let probs: Vec<f32> = (0..self.cfg.k)
                        .map(|i| [0.9f32, 0.7, 0.5, 0.3, 0.1].get(i).copied().unwrap_or(0.1))
                        .collect();
                    let depth = policy.depth(&probs);
                    self.trace.record(
                        self.tick,
                        self.clock.now_ms(),
                        Decision::Decode {
                            seqs: 1,
                            rows: 1 + depth,
                            plan: "spec",
                        },
                    );
                    1 + depth
                }
                DecodePlan::Plain { groups } => {
                    self.trace.record(
                        self.tick,
                        self.clock.now_ms(),
                        Decision::Decode {
                            seqs: indices.len(),
                            rows: indices.len(),
                            plan: "plain",
                        },
                    );
                    let _ = groups;
                    indices.len()
                }
                DecodePlan::Shared => {
                    let policy = DepthPolicy::adaptive(
                        self.cfg.k,
                        StepCost {
                            step_ms: 27.0,
                            row_ms: 4.0,
                            prior_tokens_per_ms: 0.06,
                        },
                    );
                    let chains: Vec<Vec<f64>> = indices
                        .iter()
                        .enumerate()
                        .map(|(at, _)| {
                            let probs: Vec<f32> = (0..self.cfg.k)
                                .map(|i| {
                                    [0.95f32, 0.8, 0.6, 0.4, 0.2]
                                        .get((i + at) % 5)
                                        .copied()
                                        .unwrap_or(0.2)
                                })
                                .collect();
                            policy.chain(&probs)
                        })
                        .collect();
                    let budget = self.cfg.rows_max - indices.len();
                    let depths =
                        share_rows(&chains, budget, self.cfg.share_floor, &band(self.cfg.model));
                    let rows = indices.len() + depths.iter().sum::<usize>();
                    self.trace.record(
                        self.tick,
                        self.clock.now_ms(),
                        Decision::Decode {
                            seqs: indices.len(),
                            rows,
                            plan: "shared",
                        },
                    );
                    rows
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_queue_takes_the_longest_prefix_then_the_least_recent_lane() {
        let lanes = [
            LaneCandidate {
                lane: 0,
                shared: 10,
                used: 2,
            },
            LaneCandidate {
                lane: 1,
                shared: 20,
                used: 9,
            },
            LaneCandidate {
                lane: 2,
                shared: 20,
                used: 4,
            },
        ];
        assert_eq!(pick_lane(lanes).map(|lane| lane.lane), Some(2));
        let empty: [LaneCandidate; 0] = [];
        assert_eq!(pick_lane(empty), None);
    }

    #[test]
    fn a_short_match_does_not_spend_a_long_prefix() {
        assert_eq!(reusable_shared(255, 1200, PREFIX_REUSE_MIN_ROWS), 0);
        assert_eq!(reusable_shared(256, 1200, PREFIX_REUSE_MIN_ROWS), 256);
        assert_eq!(reusable_shared(255, 255, PREFIX_REUSE_MIN_ROWS), 255);
    }

    #[test]
    fn the_decode_plan_keeps_the_schedulers_order() {
        assert_eq!(decode_plan(0, false, Some(0.2), true, 64), DecodePlan::None);
        assert_eq!(decode_plan(1, false, Some(0.2), true, 64), DecodePlan::Spec);
        assert_eq!(
            decode_plan(2, false, Some(0.2), true, 64),
            DecodePlan::Shared
        );
        assert_eq!(
            decode_plan(33, false, Some(0.2), true, 64),
            DecodePlan::Plain { groups: 1 }
        );
        assert_eq!(
            decode_plan(2, true, Some(0.2), true, 64),
            DecodePlan::Plain { groups: 1 }
        );
        assert_eq!(
            decode_plan(2, false, None, true, 64),
            DecodePlan::Plain { groups: 1 }
        );
        assert_eq!(
            decode_plan(2, false, Some(0.2), false, 64),
            DecodePlan::Plain { groups: 1 }
        );
    }

    #[test]
    fn the_share_holds_the_chunk_until_the_decoders_have_had_their_time() {
        let share = DecodeShare::new(0.5, 200_000.0);
        let order = [(20_000, 0), (4_000, 1)];
        let ages = |_| 0.0;
        // The first chunk sets the per-row cost; until half a chunk's time has gone to the
        // decoders, a long prompt waits.
        assert!(hold_prefill(
            &share, &order, ages, true, 0.1, 2_000.0, 1_900.0
        ));
        assert!(!hold_prefill(
            &share, &order, ages, true, 0.1, 2_000.0, 2_100.0
        ));
        // A prompt that fits the share floor never waits, and neither does a tick with no
        // decoder.
        assert!(!hold_prefill(
            &share,
            &[(SHARE_MIN_ROWS, 0)],
            ages,
            true,
            0.1,
            2_000.0,
            0.0
        ));
        assert!(!hold_prefill(
            &share, &order, ages, false, 0.1, 2_000.0, 0.0
        ));
    }
}

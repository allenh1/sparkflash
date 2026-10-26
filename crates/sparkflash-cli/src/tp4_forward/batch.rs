//! Several sequences on the device at once: the scheduler behind [`super::spec::serve`] and
//! [`super::spec::generate`].
//!
//! A running sequence holds a lane (super::lanes: its own position-addressed history; without lanes
//! the run has one, the single sequence's allocation). The scheduler works in ticks, and every rank
//! runs the same ticks on the same decisions (the link's admissions and cancellations), so every
//! rank launches the same collectives. A tick:
//!
//! 1. gives free lanes to waiting sequences, oldest first; a sequence resumes from the longest
//!    prefix of its tokens a free lane still holds (an agent's turns resend the conversation);
//! 2. runs one prefill chunk of the sequence with the fewest rows left to prefill (an agent's
//!    cached turn does not wait behind a cold prompt): a whole bulk chunk when no sequence decodes,
//!    else SPARKFLASH_PREFILL_CHUNK_BUSY rows (or SPARKFLASH_PREFILL_EXACT_BUSY exact steps), so a
//!    new request does not stall the running ones for its whole prompt. Before its chunk it copies
//!    the state of the positions another lane holds more of than its own, so sequences admitted
//!    together with one system prompt prefill it once. A long prompt's chunks leave the decoders
//!    some ticks of their own in between ([`DecodeShare`]);
//! 3. runs one decode step over every decoding sequence. A lone sequence takes the speculative
//!    step (draft, verify, accept); up to eight (MAX_SEQS) take one speculative step together (each
//!    drafts, and the step's R_MAX rows go to the anchors and then to the drafts most likely
//!    accepted); more take one plain rows step together, one row each (R_MAX rows per step). Every
//!    row is under its own sampler at its own position. Every rows-step row is computed as if
//!    alone (bit-identical to M = 1 on the exact path) and the sampler's noise is a function of
//!    (seed, position, id), so a sequence emits the same tokens whatever it shares the device
//!    with.
//!
//! Lanes are backed as they grow. When the budget cannot back a step, free lanes give their memory
//! back (least recently used first), then the newest running sequence is preempted: it gives up its
//! lane and waits again, and recomputes its tokens when it gets one.
//!
//! A prompt's image spans (super::vision) are held as [`HASH_SENTINEL`] tokens, the reference's
//! Engram mask: no n-gram reads through them. Their embedding uploads carry the placeholder id,
//! and a lane's prefix is shared only up to the first image span the two prompts do not hold alike
//! (the same place, grid and canvas key).

use super::draft::{AnchorRow, BLOCK, DraftBlock, DraftState, MAX_SEQS};
use super::lanes::{SLIDE_STEP, Seg, slide_floor};
use super::rows::{R_MAX, RowsState};
use super::sample::Sampler;
use super::spec::{
    DepthPolicy, ENGRAM_AHEAD_THREADS, EngramGather, GatherTask, Keepalive, Reads, SpecInputs,
    SpecOutcome, Staging, bulk_min, cut_step, gather_rows,
};
use super::*;
use crate::device_link::{DeviceFinish, DeviceJob};
use sparkflash_api::VisionImage;
use sparkflash_engine::forward::engram::HASH_SENTINEL;
use sparkflash_kernels::cuda_driver::HostMapped;
use sparkflash_vision::{IMAGE_TOKEN_ID, ImageSpan};
use std::sync::Arc;
use std::time::Instant;

/// A tick's call to rank 0's link with the next tick's room and hold, made once the decode
/// step's device work is queued ([`crate::device_link::DeviceLink::ahead`]).
pub(super) type Ahead<'a> = &'a mut dyn FnMut(usize, bool) -> Result<(), String>;

/// Sequences that may wait for a lane, past the lanes themselves.
const QUEUE: usize = 64;
/// The fewest positions a sequence takes from another lane ([`Batch::adopt`]).
const ADOPT_MIN: usize = 64;

/// A bulk prefill chunk's rows while another sequence decodes (SPARKFLASH_PREFILL_CHUNK_BUSY).
/// Every chunk reads every expert's weights (about a quarter second), so a smaller one stalls the
/// decoders more often for less prefill: 2048 rows stall them about a second at a time.
fn busy_chunk() -> usize {
    std::env::var("SPARKFLASH_PREFILL_CHUNK_BUSY")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2048usize)
        .max(1)
}

/// Exact prefill steps (R_MAX rows each) per tick while another sequence decodes
/// (SPARKFLASH_PREFILL_EXACT_BUSY): two, 32 rows.
fn busy_exact_steps() -> usize {
    std::env::var("SPARKFLASH_PREFILL_EXACT_BUSY")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2usize)
        .max(1)
}

/// The least calibrated chain probability a draft needs for a verify row of a shared speculative
/// step (SPARKFLASH_BATCH_SPEC_FLOOR); SPARKFLASH_BATCH_SPEC=0 gives several sequences the plain
/// step instead.
fn batch_spec_floor() -> Option<f64> {
    if std::env::var("SPARKFLASH_BATCH_SPEC").as_deref() == Ok("0") {
        return None;
    }
    Some(
        std::env::var("SPARKFLASH_BATCH_SPEC_FLOOR")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.2),
    )
}

/// A prefill with at most this many rows left never waits for the decoders (an agent's turn after
/// its cached prefix): [`DecodeShare`] spaces out the chunks of long prompts only.
const SHARE_MIN_ROWS: usize = 16384;

/// The decoders' share of the device's time while long prompts prefill. A chunk stalls every
/// decoding sequence for its whole length (a second or more), so one decode step a chunk leaves
/// them at ~2 tokens/s for the minutes a cold 200K prompt takes. They get up to `most` of the time
/// (SPARKFLASH_DECODE_SHARE, default 0.5; 0 runs a chunk every tick), but no more than lets every
/// prompt still prefilling reach its first token `deadline_ms` after its admission
/// (SPARKFLASH_PREFILL_DEADLINE_MS, default 200 s: an agent client gives a request up after 300 s
/// without a token), at the last chunk's cost per row.
#[derive(Clone, Copy, Debug)]
struct DecodeShare {
    most: f64,
    deadline_ms: f64,
}

impl DecodeShare {
    fn from_env() -> Self {
        let read = |name: &str, default: f64| -> f64 {
            std::env::var(name)
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(default)
        };
        Self {
            most: read("SPARKFLASH_DECODE_SHARE", 0.5).clamp(0.0, 0.9),
            deadline_ms: read("SPARKFLASH_PREFILL_DEADLINE_MS", 200_000.0),
        }
    }

    /// The share of time the decoders may take: `prefills` are the prompts still prefilling in the
    /// order they run, each as (rows left, ms since admission), at `row_ms` a row.
    fn allowed(&self, prefills: &[(usize, f64)], row_ms: f64) -> f64 {
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

/// Shares `budget` verify rows among the sequences' drafts, the likeliest accepted first: each
/// row goes to the sequence whose next draft has the highest chain probability (the lowest
/// sequence on a tie), while that is at least `floor`. A sequence's chain falls with depth, so
/// this maximises the expected accepted drafts.
fn share_rows(chains: &[Vec<f64>], budget: usize, floor: f64) -> Vec<usize> {
    let mut depths = vec![0usize; chains.len()];
    for _ in 0..budget {
        let mut best: Option<(usize, f64)> = None;
        for (index, (chain, &depth)) in chains.iter().zip(&depths).enumerate() {
            if let Some(&p) = chain.get(depth)
                && p >= floor
                && best.is_none_or(|(_, q)| p > q)
            {
                best = Some((index, p));
            }
        }
        match best {
            Some((index, _)) => depths[index] += 1,
            None => break,
        }
    }
    depths
}

/// Exact prefill rows per tick while no sequence decodes: a long prompt runs over many ticks, so
/// its lane grows as it goes and the window rows behind it slide back to the pool.
const EXACT_TICK_ROWS: usize = SLIDE_STEP;

/// Where a sequence stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Waiting for a lane: new, or preempted (its tokens are recomputed).
    Waiting,
    /// Positions `pos..target` remain to prefill, in bulk chunks or exact rows steps.
    Prefill { target: usize, bulk: bool },
    /// Positions below `pos` are on the device, and the last token (`tokens[pos]`) is the anchor.
    Decode,
}

/// One sequence on the scheduler.
struct Seq {
    id: u64,
    /// The prompt, then every token emitted; an image span's positions hold [`HASH_SENTINEL`].
    tokens: Vec<u32>,
    /// The prompt's images, in prompt order (the canvases on the tower rank only).
    images: Vec<VisionImage>,
    prompt_len: usize,
    max_new: usize,
    sampler: Sampler,
    stop_ids: Vec<u32>,
    exact: bool,
    lane: Option<usize>,
    pos: usize,
    phase: Phase,
    policy: DepthPolicy,
    /// Ended this tick: dropped at its end.
    done: bool,
    admitted: Instant,
    prefill_ms: f64,
    decode_started: Option<Instant>,
    steps: usize,
    accepted: usize,
    verified: usize,
    draft_ms: f64,
    verify_ms: f64,
    phase_ms: [f64; 4],
    reused: Option<usize>,
}

impl Seq {
    fn emitted(&self) -> usize {
        self.tokens.len() - self.prompt_len
    }

    /// The positions a placement prefills to: the prompt (its last row samples the first token),
    /// or for a preempted sequence every token before its anchor.
    fn target(&self) -> usize {
        if self.emitted() == 0 {
            self.prompt_len
        } else {
            self.tokens.len() - 1
        }
    }
}

/// A lane as the scheduler sees it.
#[derive(Default)]
struct Lane {
    /// The tokens whose state the lane holds, position by position: a finished sequence's stay
    /// for a later request that shares their prefix.
    resident: Vec<u32>,
    /// The image spans that start in `resident`.
    spans: Vec<ImageSpan>,
    /// The sequence on the lane.
    seq: Option<u64>,
    /// The tick the lane was last given up (least recently used goes first).
    used: u64,
}

/// What one sequence did in a tick.
pub(super) struct Emission {
    pub(super) seq: u64,
    /// The tokens it emitted (none when it ended without a step).
    pub(super) tokens: Vec<u32>,
    /// Set when it ended: its outcome, or why it could not run (the group carries on).
    pub(super) done: Option<Result<SpecOutcome, String>>,
}

/// The Engram readers a scheduler's steps use, kept for its whole life (scoped threads).
pub(super) struct Pools<'r> {
    /// A decode step's reads (io_uring, or reader threads).
    decode: Option<Reads<'r>>,
    /// A bulk prefill chunk's reads (reader threads).
    prefill: Option<Reads<'r>>,
    /// Keeps the Engram drive awake while any sequence is on the scheduler.
    keepalive: Option<Keepalive>,
}

impl<'r> Pools<'r> {
    pub(super) fn start<'scope>(
        scope: &'scope std::thread::Scope<'scope, '_>,
        inputs: &SpecInputs<'r>,
    ) -> Self
    where
        'r: 'scope,
    {
        let bulk_staging = inputs
            .bulk
            .and_then(|state| state.bulk.as_ref())
            .is_some_and(|state| state.engram_staging.is_some());
        Self {
            decode: inputs.engram.map(|_| Reads::decode(scope)),
            prefill: (inputs.engram.is_some() && bulk_staging).then(|| Reads::prefill(scope)),
            keepalive: inputs
                .engram
                .and_then(|reader| Keepalive::start(scope, reader)),
        }
    }
}

/// A bulk chunk's reads, submitted under the chunk before it.
struct PrefillReads<'r> {
    seq: u64,
    p: usize,
    n: usize,
    task: Arc<GatherTask<'r>>,
    buffer: usize,
}

/// The scheduler.
pub(super) struct Batch<'b, 'a, 'r> {
    device: &'b mut DeviceLayer<'a>,
    rows: &'b RowsState,
    draft: &'b DraftState,
    inputs: &'b SpecInputs<'r>,
    pools: &'b Pools<'r>,
    k: usize,
    /// The decode steps' pinned Engram staging (R_MAX rows per slot).
    staging: Option<HostMapped>,
    lanes: Vec<Lane>,
    /// Whether the run has lanes attached (else its one lane is the single sequence allocation).
    attached: bool,
    /// Admission order: the oldest first.
    seqs: Vec<Seq>,
    /// The anchor row's Engram reads for a sequence's next speculative step: (sequence, anchor
    /// position, reads), started at the end of its last step to run under the draft.
    ahead: Option<(u64, usize, Arc<GatherTask<'r>>)>,
    prefill_next: Option<PrefillReads<'r>>,
    /// The later chunks' reads of the next bulk tick that runs several (each staged in its own
    /// rows state, SpecInputs::bulk_more).
    prefill_more: Vec<Option<PrefillReads<'r>>>,
    ticks: u64,
    exchange: Option<(usize, usize)>,
    counter: Cell<u64>,
    /// The longest prompt a sequence may bring: one verify block of room after it.
    room: usize,
    reuse: bool,
    early: bool,
    exact_prefill: bool,
    busy_chunk: usize,
    busy_exact_steps: usize,
    /// The shared speculative step's draft floor ([`batch_spec_floor`]); None: the plain step.
    spec_floor: Option<f64>,
    share: DecodeShare,
    /// This rank's clock for [`DecodeShare`]: the last long prefill chunk's tick (ms, and ms a
    /// row), and the ticks' time since it.
    chunk_ms: f64,
    row_ms: f64,
    since_chunk_ms: f64,
    /// The running tick's start and whether it ran a long chunk, for [`Self::hold_after`].
    tick_started: Instant,
    tick_chunk: bool,
    print: bool,
    trace: bool,
    preemptions: usize,
}

impl<'b, 'a, 'r> Batch<'b, 'a, 'r> {
    /// A scheduler over `device`'s lanes (or its one sequence).
    ///
    /// # Errors
    /// The staging allocation failed.
    pub(super) fn new(
        device: &'b mut DeviceLayer<'a>,
        rows: &'b RowsState,
        draft: &'b DraftState,
        inputs: &'b SpecInputs<'r>,
        pools: &'b Pools<'r>,
        k: usize,
    ) -> Result<Self, String> {
        let exchange = device.device_exchange();
        let attached = device.lane_count() > 0;
        // A step over several lanes gathers its rows' embeddings on the device.
        let count = if exchange.is_some() {
            device.lane_count().max(1)
        } else {
            1
        };
        let staging = match inputs.engram {
            Some(_) => device.engram_staging(R_MAX)?,
            None => None,
        };
        let room = device.max_positions.saturating_sub(k + BLOCK + 2);
        Ok(Self {
            device,
            rows,
            draft,
            inputs,
            pools,
            k,
            staging,
            lanes: (0..count).map(|_| Lane::default()).collect(),
            attached,
            seqs: Vec::new(),
            ahead: None,
            prefill_next: None,
            prefill_more: Vec::new(),
            ticks: 0,
            exchange,
            counter: Cell::new(0),
            room,
            reuse: std::env::var("SPARKFLASH_PREFIX_CACHE").as_deref() != Ok("0"),
            early: std::env::var("SPARKFLASH_ENGRAM_AHEAD").as_deref() != Ok("0"),
            exact_prefill: std::env::var("SPARKFLASH_PREFILL_EXACT").as_deref() == Ok("1"),
            busy_chunk: busy_chunk(),
            busy_exact_steps: busy_exact_steps(),
            spec_floor: batch_spec_floor(),
            share: DecodeShare::from_env(),
            chunk_ms: 0.0,
            row_ms: 0.0,
            since_chunk_ms: 0.0,
            tick_started: Instant::now(),
            tick_chunk: false,
            print: inputs.rank == 0 && std::env::var("SPARKFLASH_SPEC_TRACE").as_deref() == Ok("1"),
            trace: inputs.rank == 0
                && std::env::var("SPARKFLASH_BATCH_TRACE").as_deref() == Ok("1"),
            preemptions: 0,
        })
    }

    /// True when no sequence runs or waits.
    pub(super) fn idle(&self) -> bool {
        self.seqs.is_empty()
    }

    /// Keeps the Engram drive awake exactly while a sequence is on the scheduler.
    fn keep_awake(&self) {
        if let Some(keepalive) = &self.pools.keepalive {
            keepalive.set(!self.seqs.is_empty());
        }
    }

    /// How many more sequences the scheduler takes.
    pub(super) fn room(&self) -> usize {
        (self.lanes.len() + QUEUE).saturating_sub(self.seqs.len())
    }

    /// The sequences on the scheduler.
    pub(super) fn ids(&self) -> Vec<u64> {
        self.seqs.iter().map(|seq| seq.id).collect()
    }

    /// The lanes, the longest prompt and the lane memory, for the boot log.
    pub(super) fn describe(&self) -> String {
        let (_, budget) = self.device.lane_memory();
        format!(
            "{} lanes, prompts up to {} tokens, {} MiB lane memory",
            self.lanes.len(),
            self.room,
            budget >> 20
        )
    }

    /// Queues a request as sequence `id`. A request the device cannot serve is refused (the group
    /// carries on).
    ///
    /// # Errors
    /// The refusal.
    pub(super) fn admit(&mut self, id: u64, job: &DeviceJob) -> Result<(), String> {
        let sampler = Sampler {
            temperature: job.temperature,
            top_p: job.top_p,
            top_k: job.top_k,
            seed: job.seed,
        };
        self.admit_seq(
            id,
            &job.prompt,
            &job.images,
            job.max_tokens.max(1),
            sampler,
            &job.stop_ids,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn admit_seq(
        &mut self,
        id: u64,
        prompt: &[u32],
        images: &[VisionImage],
        max_new: usize,
        sampler: Sampler,
        stop_ids: &[u32],
        exact: bool,
    ) -> Result<(), String> {
        if prompt.is_empty() || prompt.len() > self.room {
            return Err(format!(
                "a prompt of {} tokens does not fit the {}-token device context",
                prompt.len(),
                self.room
            ));
        }
        let mut tokens = prompt.to_vec();
        if !images.is_empty() {
            if !self.device.has_vision() || self.exchange.is_none() {
                return Err(
                    "images need the vision tower and the device exchange, and this epoch has not both"
                        .to_owned(),
                );
            }
            let mut next = 0;
            for image in images {
                let span = &image.span;
                if span.start < next || span.end() > tokens.len() {
                    return Err(format!(
                        "an image span at {}..{} is out of order or past the {}-token prompt",
                        span.start,
                        span.end(),
                        tokens.len()
                    ));
                }
                tokens[span.start..span.end()].fill(HASH_SENTINEL);
                next = span.end();
            }
        }
        if !sampler.is_greedy() && self.exchange.is_none() {
            return Err(
                "sampling needs the device exchange (SPARKFLASH_DEVICE_GATHER=0 turns it off); only greedy requests are served"
                    .to_owned(),
            );
        }
        if self.seqs.iter().any(|seq| seq.id == id) {
            return Err(format!("sequence {id} is already on the device"));
        }
        let policy = DepthPolicy::from_env(self.k)?;
        self.seqs.push(Seq {
            id,
            tokens,
            images: images.to_vec(),
            prompt_len: prompt.len(),
            max_new,
            sampler,
            stop_ids: stop_ids.to_vec(),
            exact,
            lane: None,
            pos: 0,
            phase: Phase::Waiting,
            policy,
            done: false,
            admitted: Instant::now(),
            prefill_ms: 0.0,
            decode_started: None,
            steps: 0,
            accepted: 0,
            verified: 0,
            draft_ms: 0.0,
            verify_ms: 0.0,
            phase_ms: [0.0; 4],
            reused: None,
        });
        self.keep_awake();
        Ok(())
    }

    /// Ends sequence `id` before the tick (rank 0's stream ended); None when it is not here.
    ///
    /// # Errors
    /// A driver failure waiting for its reads.
    pub(super) fn cancel(&mut self, id: u64) -> Result<Option<SpecOutcome>, String> {
        let Some(index) = self.seqs.iter().position(|seq| seq.id == id) else {
            return Ok(None);
        };
        let valid = match self.seqs[index].phase {
            Phase::Waiting => 0,
            Phase::Prefill { .. } | Phase::Decode => self.seqs[index].pos,
        };
        let outcome = self.end(index, DeviceFinish::Aborted, valid)?;
        self.seqs.remove(index);
        self.keep_awake();
        Ok(Some(outcome))
    }

    /// Whether a sequence decodes.
    fn decoding(&self) -> bool {
        self.seqs
            .iter()
            .any(|seq| !seq.done && seq.phase == Phase::Decode)
    }

    /// The prefilling sequences in the order their chunks run, each as (rows left, index): the
    /// fewest rows left first (ties to the oldest), so a turn that reuses a cached prefix does not
    /// wait behind a long cold prompt.
    fn prefill_order(&self) -> Vec<(usize, usize)> {
        let mut order: Vec<(usize, usize)> = self
            .seqs
            .iter()
            .enumerate()
            .filter_map(|(index, seq)| match seq.phase {
                Phase::Prefill { target, .. } if !seq.done => {
                    Some((target.saturating_sub(seq.pos), index))
                }
                _ => None,
            })
            .collect();
        order.sort_unstable();
        order
    }

    /// This rank's answer for the next tick's [`DeviceTick::hold_prefill`] (rank 0's is the one
    /// every rank runs): whether the decoders have not yet had their [`DecodeShare`] of the time
    /// since the last long chunk.
    ///
    /// [`DeviceTick::hold_prefill`]: crate::device_link::DeviceTick::hold_prefill
    pub(super) fn hold_prefill(&self) -> bool {
        self.hold_since(self.since_chunk_ms)
    }

    /// [`Self::hold_prefill`] as it will stand when the running tick ends, `since_chunk_ms`
    /// projected to now: the answer rank 0 sends ahead under the tick's device work.
    fn hold_after(&self) -> bool {
        self.hold_since(if self.tick_chunk {
            0.0
        } else {
            self.since_chunk_ms + self.tick_started.elapsed().as_secs_f64() * 1e3
        })
    }

    /// Whether the decoders have had less than their share when `since_ms` passed since the
    /// last long chunk.
    fn hold_since(&self, since_ms: f64) -> bool {
        let order = self.prefill_order();
        if self.row_ms == 0.0
            || order
                .first()
                .is_none_or(|&(left, _)| left <= SHARE_MIN_ROWS)
            || !self.decoding()
        {
            return false;
        }
        let prefills: Vec<(usize, f64)> = order
            .iter()
            .map(|&(left, index)| {
                (
                    left,
                    self.seqs[index].admitted.elapsed().as_secs_f64() * 1e3,
                )
            })
            .collect();
        let share = self.share.allowed(&prefills, self.row_ms);
        share > 0.0 && since_ms < self.chunk_ms * share / (1.0 - share)
    }

    /// Hands rank 0's link the next tick's room and hold under this step's device work, so the
    /// next tick's decisions reach the workers before they finish this one. The room counts the
    /// sequences that end in this tick: the next tick only has more.
    fn send_ahead(&self, ahead: Ahead<'_>) -> Result<(), String> {
        ahead(self.room(), self.hold_after())
    }

    /// Runs one tick: placements, one prefill chunk (unless `hold`, rank 0's
    /// [`Self::hold_prefill`], keeps a long one back), one decode step. The decode step calls
    /// `ahead` with the next tick's room and hold once its device work is queued.
    ///
    /// # Errors
    /// A step failed: the ranks may have desynchronised, so the group cannot continue.
    pub(super) fn tick(&mut self, hold: bool, ahead: Ahead<'_>) -> Result<Vec<Emission>, String> {
        self.ticks += 1;
        let started = Instant::now();
        self.tick_started = started;
        self.tick_chunk = false;
        let mut out = Vec::new();
        self.place(&mut out)?;
        let decoding = self.decoding();
        let mut prefilled = None;
        let mut long_chunk = None;
        if let Some(&(_, index)) = self.prefill_order().first() {
            // The rows another lane holds are copied, not run (prefill's own adopt then finds
            // nothing more): the clock below counts only the rows the chunk computes.
            self.adopt(index)?;
            let seq = &self.seqs[index];
            let left = match seq.phase {
                Phase::Prefill { target, .. } => target.saturating_sub(seq.pos),
                Phase::Waiting | Phase::Decode => 0,
            };
            // Every rank takes the same branch: the rows left and `decoding` are the scheduler's
            // state, and `hold` is rank 0's.
            if !(hold && decoding && left > SHARE_MIN_ROWS) {
                let from = self.seqs[index].pos;
                self.prefill(index, decoding, &mut out)?;
                // A preempted sequence went back to position 0.
                let ran = self.seqs[index].pos.saturating_sub(from);
                prefilled = Some((self.seqs[index].id, from, ran));
                if left > SHARE_MIN_ROWS && ran > 0 {
                    long_chunk = Some(ran);
                    self.tick_chunk = true;
                }
            }
        }
        let prefill_ms = started.elapsed().as_secs_f64() * 1e3;
        let batch = self.decode(&mut out, ahead)?;
        // The decode step reads its tokens back, so the tick's wall time covers its device work.
        let tick_ms = started.elapsed().as_secs_f64() * 1e3;
        match long_chunk {
            Some(ran) => {
                self.chunk_ms = tick_ms;
                self.row_ms = tick_ms / ran as f64;
                self.since_chunk_ms = 0.0;
            }
            None => self.since_chunk_ms += tick_ms,
        }
        if self.trace {
            let (held, budget) = self.device.lane_memory();
            println!(
                "batch: tick {} seqs {} prefill {prefilled:?} {prefill_ms:.1} ms decode {batch} rows {:.1} ms lanes {} of {} MiB",
                self.ticks,
                self.seqs.len(),
                started.elapsed().as_secs_f64() * 1e3 - prefill_ms,
                held >> 20,
                budget >> 20,
            );
        }
        self.seqs.retain(|seq| !seq.done);
        self.keep_awake();
        self.slide()?;
        Ok(out)
    }

    /// Gives back the window ring rows a long running sequence no longer reads (SLIDE_KEEP behind
    /// its position, SLIDE_STEP rows at a time).
    fn slide(&mut self) -> Result<(), String> {
        if !self.attached {
            return Ok(());
        }
        for seq in &self.seqs {
            let Some(lane) = seq.lane else {
                continue;
            };
            let floor = slide_floor(seq.pos);
            if floor > self.device.lane_floor(lane) {
                self.device.slide_lane(lane, floor)?;
            }
        }
        Ok(())
    }

    /// Runs `request` alone to its end: `on_step(tokens, last)` gets each step's tokens, and a
    /// false answer on a step that is not the last ends it (Aborted) before the next.
    ///
    /// # Errors
    /// A step failed, or the device cannot serve the request.
    pub(super) fn run_one(
        &mut self,
        request: &super::spec::SpecRequest<'_>,
        on_step: &mut super::spec::OnStep<'_>,
    ) -> Result<SpecOutcome, String> {
        let id = u64::MAX;
        self.admit_seq(
            id,
            request.prompt,
            &[],
            request.max_new,
            request.sampler,
            request.stop_ids,
            request.exact,
        )?;
        loop {
            for emission in self.tick(false, &mut |_, _| Ok(()))? {
                let last = emission.done.is_some();
                let go = on_step(&emission.tokens, last)?;
                if let Some(done) = emission.done {
                    return done;
                }
                if !go {
                    return self
                        .cancel(id)?
                        .ok_or_else(|| "the request left the scheduler".to_owned());
                }
            }
        }
    }

    /// The boot warm-up: a bulk and an exact-path request together (so the plain batched step runs
    /// too), before the first client, so it does not pay the lazy module loads and first-use
    /// collective setup. Every rank runs the same ids; the lanes are given up after.
    ///
    /// # Errors
    /// A step failed.
    pub(super) fn warm_up(&mut self) -> Result<(), String> {
        let greedy = Sampler::GREEDY;
        for (index, len) in [bulk_min().max(1024), 40].into_iter().enumerate() {
            let len = len.min(self.room);
            let ids: Vec<u32> = (0..len as u32)
                .map(|i| 1000 + i.wrapping_mul(7919) % 90_000)
                .collect();
            self.admit_seq(u64::MAX - index as u64, &ids, &[], 8, greedy, &[], true)?;
        }
        while !self.idle() {
            for emission in self.tick(false, &mut |_, _| Ok(()))? {
                if let Some(Err(error)) = emission.done {
                    return Err(format!("warm-up: {error}"));
                }
            }
        }
        for lane in 0..self.lanes.len() {
            self.forget(lane)?;
        }
        Ok(())
    }

    /// Lane `lane` holds nothing any more (its memory goes back to the pool).
    fn forget(&mut self, lane: usize) -> Result<(), String> {
        self.lanes[lane].resident.clear();
        self.lanes[lane].spans.clear();
        if self.attached {
            self.device.release_lane(lane)?;
        }
        Ok(())
    }

    /// Points the device at lane `lane` for a step of its own.
    fn enter(&mut self, lane: usize) -> Result<(), String> {
        self.device.clear_segments();
        if self.attached {
            self.device.select_lane(lane)?;
        }
        Ok(())
    }

    /// Ends sequence `index` (it stays in `seqs`, marked done) and gives its lane up with the
    /// state of positions `..valid`.
    fn end(
        &mut self,
        index: usize,
        finish: DeviceFinish,
        valid: usize,
    ) -> Result<SpecOutcome, String> {
        let id = self.seqs[index].id;
        self.drop_reads(id)?;
        self.device.vision_release(id);
        let seq = &mut self.seqs[index];
        seq.done = true;
        if let Some(lane) = seq.lane.take() {
            let lane = &mut self.lanes[lane];
            lane.seq = None;
            lane.used = self.ticks;
            lane.resident.clear();
            let valid = valid.min(seq.tokens.len());
            lane.resident.extend_from_slice(&seq.tokens[..valid]);
            lane.spans.clear();
            lane.spans.extend(
                seq.images
                    .iter()
                    .map(|image| image.span)
                    .filter(|span| span.start < valid),
            );
        }
        Ok(SpecOutcome {
            emitted: seq.tokens[seq.prompt_len..].to_vec(),
            finish,
            prefill_ms: seq.prefill_ms,
            decode_ms: seq
                .decode_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1e3),
            steps: seq.steps,
            accepted_total: seq.accepted,
            verified_total: seq.verified,
            draft_ms: seq.draft_ms,
            verify_ms: seq.verify_ms,
            phase: seq.phase_ms,
            adaptive: seq.policy.adaptive,
            reused: seq.reused.unwrap_or(0),
        })
    }

    /// Waits for (and drops) any reads in flight for sequence `id`.
    fn drop_reads(&mut self, id: u64) -> Result<(), String> {
        if self.ahead.as_ref().is_some_and(|(seq, _, _)| *seq == id) {
            self.wait_ahead()?;
        }
        if self
            .prefill_next
            .iter()
            .chain(self.prefill_more.iter().flatten())
            .any(|next| next.seq == id)
        {
            self.wait_prefill_next()?;
        }
        Ok(())
    }

    /// Waits for the anchor reads in flight: they land in the decode staging.
    fn wait_ahead(&mut self) -> Result<(), String> {
        if let (Some((_, _, task)), Some(reads)) = (self.ahead.take(), self.pools.decode.as_ref()) {
            reads.wait_all(&task)?;
        }
        Ok(())
    }

    /// Waits for (and drops) the next bulk chunks' reads in flight.
    fn wait_prefill_next(&mut self) -> Result<(), String> {
        self.wait_prefill_more()?;
        if let (Some(next), Some(reads)) = (self.prefill_next.take(), self.pools.prefill.as_ref()) {
            reads.wait_all(&next.task)?;
        }
        Ok(())
    }

    fn wait_prefill_more(&mut self) -> Result<(), String> {
        for slot in 0..self.prefill_more.len() {
            self.wait_prefill_slot(slot)?;
        }
        Ok(())
    }

    fn wait_prefill_slot(&mut self, slot: usize) -> Result<(), String> {
        if let (Some(next), Some(reads)) = (
            self.prefill_more.get_mut(slot).and_then(Option::take),
            self.pools.prefill.as_ref(),
        ) {
            reads.wait_all(&next.task)?;
        }
        Ok(())
    }

    /// Whether sequence `index`'s bulk ticks run several chunks at once ([`Self::prefill_bulk`]):
    /// with more rows states, the collective stream and nothing decoding (a tick with decoders runs
    /// one short chunk), and without images (their tower rows are a single chunk's). Every rank
    /// answers alike.
    fn pairs(&self, index: usize, decoding: bool) -> bool {
        !decoding
            && !self.inputs.bulk_more.is_empty()
            && self.device.bulk_comm_open()
            && self.seqs[index].images.is_empty()
    }

    /// Sequence `index` ends here with `finish` and the state of `..valid`: its emission.
    fn finish(
        &mut self,
        index: usize,
        finish: DeviceFinish,
        valid: usize,
        tokens: Vec<u32>,
        out: &mut Vec<Emission>,
    ) -> Result<(), String> {
        let outcome = self.end(index, finish, valid)?;
        out.push(Emission {
            seq: self.seqs[index].id,
            tokens,
            done: Some(Ok(outcome)),
        });
        Ok(())
    }

    /// Gives free lanes to the waiting sequences, oldest first.
    fn place(&mut self, out: &mut Vec<Emission>) -> Result<(), String> {
        loop {
            let Some(index) = self
                .seqs
                .iter()
                .position(|seq| !seq.done && seq.phase == Phase::Waiting)
            else {
                return Ok(());
            };
            let target = self.seqs[index].target();
            let wanted = &self.seqs[index].tokens[..target];
            let images = &self.seqs[index].images;
            let free = (0..self.lanes.len()).filter(|&lane| self.lanes[lane].seq.is_none());
            let Some((lane, shared)) = free
                .map(|lane| {
                    let held = &self.lanes[lane];
                    (
                        lane,
                        if self.reuse {
                            common_prefix(&held.resident, &held.spans, wanted, images)
                        } else {
                            0
                        },
                    )
                })
                .max_by(|a, b| {
                    // The longest shared prefix, else the least recently used lane.
                    a.1.cmp(&b.1)
                        .then_with(|| self.lanes[b.0].used.cmp(&self.lanes[a.0].used))
                        .then_with(|| b.0.cmp(&a.0))
                })
            else {
                return Ok(());
            };
            // At least the prompt's last row runs, for the first token's logits.
            let cap = if self.seqs[index].emitted() == 0 {
                target - 1
            } else {
                target
            };
            let mut start = self.device.resume_boundary(shared.min(cap));
            // A lane whose window rings slid past the resume point cannot resume there (the rows
            // start reads need both windows, the layers' and the drafter's, behind it).
            let floor = self.device.lane_floor(lane);
            if floor > 0
                && start < floor + 2 * sparkflash_kernels::flash_mla_csa2_seq_launch::SWA_WINDOW
            {
                self.forget(lane)?;
                start = 0;
            }
            let bulk = self.bulk_from(target, start);
            // A long prompt's lane grows a tick's rows at a time as it prefills (its window rows
            // sliding back behind it), so placing it backs the first tick only; one that cannot
            // fit even alone is refused now, not at the end of its prefill.
            let end = target + BLOCK + self.k + 2;
            let ahead = self.tick_rows(bulk, false);
            let fits = self.device.lane_fits(end, ahead);
            if !fits || !self.grow(lane, end.min(start + ahead + BLOCK + self.k + 2))? {
                if fits && self.seqs.iter().any(|seq| !seq.done && seq.lane.is_some()) {
                    // It waits for a running sequence's memory.
                    return Ok(());
                }
                let id = self.seqs[index].id;
                self.seqs[index].done = true;
                out.push(Emission {
                    seq: id,
                    tokens: Vec::new(),
                    done: Some(Err(format!(
                        "a sequence of {target} tokens does not fit the device's lane memory (SPARKFLASH_LANE_MEMORY_MB)"
                    ))),
                });
                continue;
            }
            self.enter(lane)?;
            self.device.rewind_rows(start);
            let id = self.seqs[index].id;
            let lane_state = &mut self.lanes[lane];
            lane_state.seq = Some(id);
            lane_state.resident.truncate(start);
            lane_state.spans.retain(|span| span.start < start);
            let seq = &mut self.seqs[index];
            seq.lane = Some(lane);
            seq.pos = start;
            seq.reused.get_or_insert(start);
            seq.phase = if start == target {
                Phase::Decode
            } else {
                Phase::Prefill { target, bulk }
            };
        }
    }

    /// Whether a prefill from `start` to `target` takes the bulk chunks.
    fn bulk_from(&self, target: usize, start: usize) -> bool {
        self.inputs.bulk.is_some()
            && self.exchange.is_some()
            && !self.exact_prefill
            && target - start >= bulk_min()
    }

    /// Sequence `index`, about to prefill from its position, takes the state of the positions it
    /// shares with another lane when that lane holds more of them than its own (sequences that
    /// share a system prompt, admitted together, prefill it once): the rows are copied, not run
    /// again.
    fn adopt(&mut self, index: usize) -> Result<(), String> {
        let seq = &self.seqs[index];
        let (Phase::Prefill { target, .. }, Some(own)) = (seq.phase, seq.lane) else {
            return Ok(());
        };
        if !self.reuse || !self.attached {
            return Ok(());
        }
        let wanted = &seq.tokens[..target];
        let mut best: Option<(usize, usize)> = None;
        for (lane, held) in self.lanes.iter().enumerate() {
            if lane == own {
                continue;
            }
            let shared = match held.seq {
                // A running sequence's lane holds the positions below its own.
                Some(id) => {
                    let Some(other) = self.seqs.iter().find(|other| other.id == id) else {
                        continue;
                    };
                    let spans: Vec<ImageSpan> = other
                        .images
                        .iter()
                        .map(|image| image.span)
                        .filter(|span| span.start < other.pos)
                        .collect();
                    common_prefix(&other.tokens[..other.pos], &spans, wanted, &seq.images)
                }
                None => common_prefix(&held.resident, &held.spans, wanted, &seq.images),
            };
            if best.is_none_or(|(most, _)| shared > most) {
                best = Some((shared, lane));
            }
        }
        let Some((shared, src)) = best else {
            return Ok(());
        };
        // At least the prompt's last row runs, for the first token's logits.
        let cap = if seq.emitted() == 0 {
            target - 1
        } else {
            target
        };
        let start = self.device.resume_boundary(shared.min(cap));
        if start < seq.pos + ADOPT_MIN {
            return Ok(());
        }
        let floor = self.device.lane_floor(src).max(self.device.lane_floor(own));
        if floor > 0
            && start < floor + 2 * sparkflash_kernels::flash_mla_csa2_seq_launch::SWA_WINDOW
        {
            return Ok(());
        }
        if !self.device.copy_lane_prefix(own, src, start)? {
            return Ok(());
        }
        let bulk = self.bulk_from(target, start);
        let seq = &mut self.seqs[index];
        seq.pos = start;
        seq.phase = Phase::Prefill { target, bulk };
        seq.reused = Some(seq.reused.unwrap_or(0).max(start));
        Ok(())
    }

    /// The most rows one tick's prefill of a sequence runs (a bulk chunk, or exact rows steps):
    /// its lane is backed that far ahead. A busy chunk may take a tail too short for a chunk of
    /// its own ([`bulk_rows`]).
    fn tick_rows(&self, bulk: bool, decoding: bool) -> usize {
        match self.inputs.bulk {
            Some(state) if bulk && decoding => (self.busy_chunk + bulk_min()).min(state.cap),
            Some(state) if bulk => state.cap,
            _ if decoding => self.busy_exact_steps * R_MAX,
            _ => EXACT_TICK_ROWS,
        }
    }

    /// The bulk chunk a tick runs and the most it may stretch to ([`bulk_rows`]).
    fn bulk_budget(&self, decoding: bool) -> (usize, usize) {
        let most = self.tick_rows(true, decoding);
        match self.inputs.bulk {
            Some(state) if decoding => (self.busy_chunk.min(state.cap), most),
            _ => (most, most),
        }
    }

    /// Backs lane `lane` for `rows` positions, taking memory back from free lanes (least recently
    /// used first) when the budget is spent. False when even that is not enough.
    fn grow(&mut self, lane: usize, rows: usize) -> Result<bool, String> {
        if !self.attached {
            return Ok(true);
        }
        loop {
            if self.device.ensure_lane(lane, rows)? {
                return Ok(true);
            }
            let idle = (0..self.lanes.len())
                .filter(|&other| {
                    other != lane
                        && self.lanes[other].seq.is_none()
                        && self.device.lane_backed(other) > 0
                })
                .min_by_key(|&other| (self.lanes[other].used, other));
            let Some(idle) = idle else {
                return Ok(false);
            };
            self.forget(idle)?;
        }
    }

    /// Backs sequence `index`'s lane for its next step, preempting the newest running sequences
    /// while the memory is short. False when sequence `index` itself was preempted.
    fn make_room(&mut self, index: usize, rows: usize) -> Result<bool, String> {
        let Some(lane) = self.seqs[index].lane else {
            return Ok(false);
        };
        loop {
            if self.grow(lane, rows)? {
                return Ok(true);
            }
            let Some(victim) = self
                .seqs
                .iter()
                .rposition(|seq| !seq.done && seq.lane.is_some())
            else {
                return Ok(false);
            };
            self.preempt(victim)?;
            if victim == index {
                return Ok(false);
            }
        }
    }

    /// Sends sequence `index` back to wait: its lane and the lane's memory go, and its tokens are
    /// recomputed when it gets a lane again.
    fn preempt(&mut self, index: usize) -> Result<(), String> {
        let id = self.seqs[index].id;
        self.drop_reads(id)?;
        self.device.vision_release(id);
        let seq = &mut self.seqs[index];
        let Some(lane) = seq.lane.take() else {
            return Ok(());
        };
        seq.phase = Phase::Waiting;
        seq.pos = 0;
        self.lanes[lane].seq = None;
        self.lanes[lane].used = self.ticks;
        self.forget(lane)?;
        self.preemptions += 1;
        if self.inputs.rank == 0 {
            eprintln!(
                "batch: preempted sequence {id} at {} tokens to free lane memory ({} preemptions)",
                self.seqs[index].tokens.len(),
                self.preemptions
            );
        }
        Ok(())
    }

    /// One prefill chunk (or up to the tick's exact steps) of sequence `index`.
    fn prefill(
        &mut self,
        index: usize,
        decoding: bool,
        out: &mut Vec<Emission>,
    ) -> Result<(), String> {
        self.adopt(index)?;
        let Phase::Prefill { target, bulk } = self.seqs[index].phase else {
            return Ok(());
        };
        let lane = self.seqs[index].lane.ok_or("prefill: no lane")?;
        let p = self.seqs[index].pos;
        let mut ahead = self.tick_rows(bulk, decoding);
        if bulk && self.pairs(index, decoding) {
            ahead *= 1 + self.inputs.bulk_more.len();
        }
        if !self.make_room(index, target.min(p + ahead) + BLOCK + self.k + 2)? {
            // Preempted: it waits for a lane again.
            return Ok(());
        }
        self.enter(lane)?;
        let first = if bulk {
            self.prefill_bulk(index, target, decoding)?
        } else {
            self.prefill_exact(index, target, decoding)?
        };
        if self.seqs[index].pos < target {
            return Ok(());
        }
        self.device.cuda.synchronize()?;
        let seq = &mut self.seqs[index];
        seq.phase = Phase::Decode;
        let Some(first) = first else {
            // A recomputed sequence: its anchor is already known.
            return Ok(());
        };
        seq.prefill_ms = seq.admitted.elapsed().as_secs_f64() * 1e3;
        seq.decode_started = Some(Instant::now());
        let mut step = vec![first];
        let finish = cut_step(&mut step, 0, seq.max_new, &seq.stop_ids, seq.exact);
        seq.tokens.extend_from_slice(&step);
        match finish {
            Some(finish) => self.finish(index, finish, target, step, out),
            None => {
                out.push(Emission {
                    seq: seq.id,
                    tokens: step,
                    done: None,
                });
                Ok(())
            }
        }
    }

    /// Exact rows steps of R_MAX rows (bit-identical to M = 1); the last samples the first token.
    fn prefill_exact(
        &mut self,
        index: usize,
        target: usize,
        decoding: bool,
    ) -> Result<Option<u32>, String> {
        let (rows, inputs, draft) = (self.rows, self.inputs, self.draft);
        let steps = self.tick_rows(false, decoding).div_ceil(R_MAX);
        let mut first = None;
        for _ in 0..steps {
            let seq = &self.seqs[index];
            let p = seq.pos;
            if p == target {
                break;
            }
            let n = R_MAX.min(target - p);
            if let Some(reader) = inputs.engram {
                self.device
                    .engram_set_rows(inputs.config, reader, &seq.tokens, p, n)?;
            }
            match self.exchange {
                Some(exchange) => {
                    let ids = upload_ids(&seq.tokens[p..p + n]);
                    self.device
                        .cuda
                        .copy_htod(&Deviceptr(rows.b.row_ids), &ids)?;
                    let seq = &self.seqs[index];
                    self.device.vision_begin(seq.id, &seq.images, p, n)?;
                    self.device.begin_rows_device(
                        rows,
                        p,
                        inputs.embed.0,
                        rows.b.row_ids,
                        n,
                        exchange,
                    )?;
                }
                None => {
                    let hidden = gather_rows(
                        self.device.cuda,
                        inputs,
                        &self.counter,
                        &seq.tokens[p..p + n],
                    )?;
                    self.device.begin_rows(rows, p, &hidden)?;
                }
            }
            self.device.spec_layers(rows, inputs.staged, p, n)?;
            self.device.vision_end();
            self.device.draft_seed(rows, draft, p, n)?;
            let seq = &self.seqs[index];
            if p + n == target && seq.emitted() == 0 {
                // Row y (position p + y) samples the token at sequence index p + y + 1.
                let picks = self.device.head_rows_sampled(
                    rows,
                    inputs.tail,
                    inputs.transport,
                    n,
                    &seq.sampler,
                    p + 1,
                )?;
                first = Some(picks[n - 1].0);
            }
            self.seqs[index].pos = p + n;
        }
        Ok(first)
    }

    /// One bulk chunk (not bit-identical to M = 1; see super::bulk), or several (one per bulk rows
    /// state) when the sequence [pairs](Self::pairs) and has more chunks left of more than R_MAX
    /// rows each: their layers interleave so that each chunk's collectives run under the others'
    /// launches ([`DeviceLayer::prefill_layers_paired`]), with the same results as the chunks run
    /// one after the other. A chunk's Engram reads were submitted under the tick before it (or now, for a first
    /// chunk), and the next tick's go out under this one's device time. The last chunk's last row
    /// samples the first token.
    fn prefill_bulk(
        &mut self,
        index: usize,
        target: usize,
        decoding: bool,
    ) -> Result<Option<u32>, String> {
        let (inputs, draft) = (self.inputs, self.draft);
        let state = inputs.bulk.ok_or("prefill: no bulk rows state")?;
        let exchange = self
            .exchange
            .ok_or("prefill: the bulk chunks need the exchange")?;
        let staging = bulk_staging(state, inputs);
        let pools = self.pools;
        let reads = pools.prefill.as_ref().filter(|_| staging.is_some());
        let id = self.seqs[index].id;
        let p = self.seqs[index].pos;
        let (budget, most) = self.bulk_budget(decoding);
        let chunk = |p: usize| bulk_rows(target - p, budget, most);
        let pending = match self.prefill_next.take() {
            Some(next) if next.seq == id && next.p == p => Some(next),
            Some(next) => {
                self.prefill_next = Some(next);
                self.wait_prefill_next()?;
                None
            }
            None => None,
        };
        let submit = |device: &mut DeviceLayer<'_>,
                      tokens: &[u32],
                      p: usize,
                      n: usize,
                      staging: Option<&[HostMapped; 2]>,
                      buffer: usize|
         -> Result<Option<PrefillReads<'r>>, String> {
            let (Some(reader), Some(reads), Some(staging)) = (inputs.engram, reads, staging) else {
                return Ok(None);
            };
            let Some(mut gather) = device.engram_plan(inputs.config, reader, tokens, p, n, 0, n)?
            else {
                return Ok(None);
            };
            gather.cached = false;
            let task = reads.submit(
                gather,
                Staging::of(&staging[buffer]),
                super::spec::prefill_read_threads(),
            )?;
            Ok(Some(PrefillReads {
                seq: id,
                p,
                n,
                task,
                buffer,
            }))
        };
        let current = match pending {
            Some(next) => Some(next),
            None => submit(
                self.device,
                &self.seqs[index].tokens,
                p,
                chunk(p),
                staging,
                0,
            )?,
        };
        let n = current.as_ref().map_or_else(|| chunk(p), |next| next.n);
        let buffer = current.as_ref().map_or(0, |current| current.buffer);
        let several =
            self.pairs(index, decoding) && n > R_MAX && p + n < target && chunk(p + n) > R_MAX;
        if !several {
            // A later slot's reads are for a chunk this tick runs on the first rows state.
            self.wait_prefill_more()?;
            let ids = upload_ids(&self.seqs[index].tokens[p..p + n]);
            self.device
                .cuda
                .copy_htod(&Deviceptr(state.b.row_ids), &ids)?;
            let seq = &self.seqs[index];
            self.device.vision_begin(seq.id, &seq.images, p, n)?;
            let tokens = &self.seqs[index].tokens;
            self.device.begin_rows_device(
                state,
                p,
                inputs.embed.0,
                state.b.row_ids,
                n,
                exchange,
            )?;
            let after = p + n;
            let mut next = None;
            match reads {
                Some(reads) => {
                    self.device.prefill_layers_gathered(
                        state,
                        inputs.staged,
                        &inputs.config.engram_layer_ids,
                        reads,
                        chunk_reads(current.as_ref(), staging),
                        p,
                        n,
                        |device| {
                            if after < target {
                                next = submit(
                                    device,
                                    tokens,
                                    after,
                                    chunk(after),
                                    staging,
                                    buffer ^ 1,
                                )?;
                            }
                            Ok(())
                        },
                    )?;
                }
                None => self.device.spec_layers(state, inputs.staged, p, n)?,
            }
            self.device.vision_end();
            self.prefill_next = next;
            self.device.draft_seed(state, draft, p, n)?;
            self.seqs[index].pos = after;
            if after < target || self.seqs[index].emitted() > 0 {
                return Ok(None);
            }
            return self.bulk_first(index, state, n, target).map(Some);
        }
        // The chunks this tick runs together: the first, then one on each further rows state while
        // another chunk of more than R_MAX rows follows.
        let more = inputs.bulk_more;
        if self.prefill_more.len() < more.len() {
            self.prefill_more.resize_with(more.len(), || None);
        }
        let mut slots = vec![(state, p, n, current, staging, buffer)];
        let mut end = p + n;
        for (slot, state_i) in more.iter().enumerate() {
            if end >= target || chunk(end) <= R_MAX {
                break;
            }
            let staging_i = bulk_staging(state_i, inputs);
            let pending = match self.prefill_more[slot].take() {
                Some(next) if next.seq == id && next.p == end => Some(next),
                Some(next) => {
                    self.prefill_more[slot] = Some(next);
                    self.wait_prefill_slot(slot)?;
                    None
                }
                None => None,
            };
            let current_i = match pending {
                Some(next) => Some(next),
                None => submit(
                    self.device,
                    &self.seqs[index].tokens,
                    end,
                    chunk(end),
                    staging_i,
                    buffer,
                )?,
            };
            let n_i = current_i.as_ref().map_or_else(|| chunk(end), |next| next.n);
            let buffer_i = current_i.as_ref().map_or(buffer, |current| current.buffer);
            slots.push((state_i, end, n_i, current_i, staging_i, buffer_i));
            end += n_i;
        }
        for slot in slots.len() - 1..more.len() {
            self.wait_prefill_slot(slot)?;
        }
        for &(state, p, n, ..) in &slots {
            let ids = upload_ids(&self.seqs[index].tokens[p..p + n]);
            self.device
                .cuda
                .copy_htod(&Deviceptr(state.b.row_ids), &ids)?;
            self.device.begin_rows_device(
                state,
                p,
                inputs.embed.0,
                state.b.row_ids,
                n,
                exchange,
            )?;
        }
        let tokens = &self.seqs[index].tokens;
        let after = end;
        let chunks: Vec<_> = slots
            .iter()
            .map(|(state, p, n, current, staging, _)| {
                (*state, *p, *n, chunk_reads(current.as_ref(), *staging))
            })
            .collect();
        let mut next: Vec<Option<PrefillReads<'r>>> = slots.iter().map(|_| None).collect();
        self.device.prefill_layers_paired(
            &chunks,
            inputs.staged,
            &inputs.config.engram_layer_ids,
            reads,
            |device| {
                // The next tick's chunks, when it runs them as this one did.
                let mut at = after;
                for (slot, &(.., staging, buffer)) in slots.iter().enumerate() {
                    if at >= target {
                        break;
                    }
                    let n_next = chunk(at);
                    if slot > 0 && n_next <= R_MAX {
                        break;
                    }
                    next[slot] = submit(device, tokens, at, n_next, staging, buffer ^ 1)?;
                    if n_next <= R_MAX {
                        break;
                    }
                    at += n_next;
                }
                Ok(())
            },
        )?;
        drop(chunks);
        let mut next = next.into_iter();
        self.prefill_next = next.next().flatten();
        for (slot, reads) in next.enumerate() {
            self.prefill_more[slot] = reads;
        }
        for &(state, p, n, ..) in &slots {
            self.device.draft_seed(state, draft, p, n)?;
        }
        self.seqs[index].pos = after;
        if after < target || self.seqs[index].emitted() > 0 {
            return Ok(None);
        }
        let &(last, _, n_last, ..) = slots.last().ok_or("prefill: no bulk chunk")?;
        self.bulk_first(index, last, n_last, target).map(Some)
    }

    /// The first token of sequence `index`, from the last row of the bulk chunk of `n` rows that
    /// ended its prompt on `state`: the head reads that row only, so its stream and pre mix move
    /// to the decode rows.
    fn bulk_first(
        &mut self,
        index: usize,
        state: &RowsState,
        n: usize,
        target: usize,
    ) -> Result<u32, String> {
        let (rows, inputs) = (self.rows, self.inputs);
        let row = (n - 1) as u64;
        let hc = self.device.hc_mult;
        let stream_bytes = (hc * self.device.dim * 4) as u64;
        self.device.cuda.copy_dtod_async(
            &Deviceptr(rows.b.stream),
            &Deviceptr(state.b.stream + row * stream_bytes),
            stream_bytes as usize,
        )?;
        self.device.cuda.copy_dtod_async(
            &Deviceptr(rows.b.pre_mix),
            &Deviceptr(state.b.pre_mix + row * (hc * 4) as u64),
            hc * 4,
        )?;
        let sampler = self.seqs[index].sampler;
        let picks = self.device.head_rows_sampled(
            rows,
            inputs.tail,
            inputs.transport,
            1,
            &sampler,
            target,
        )?;
        Ok(picks[0].0)
    }

    /// One decode step over every decoding sequence; returns the rows it ran.
    fn decode(&mut self, out: &mut Vec<Emission>, ahead: Ahead<'_>) -> Result<usize, String> {
        let limit = self.device.max_positions;
        let decoding = |seqs: &[Seq]| -> Vec<usize> {
            (0..seqs.len())
                .filter(|&index| !seqs[index].done && seqs[index].phase == Phase::Decode)
                .collect()
        };
        for index in decoding(&self.seqs) {
            let pos = self.seqs[index].pos;
            if pos + self.k + BLOCK + 1 >= limit {
                self.finish(index, DeviceFinish::Length, pos, Vec::new(), out)?;
                continue;
            }
            // Older sequences get memory first: a step short of it preempts the newest.
            if !self.seqs[index].done {
                self.make_room(index, pos + self.k + BLOCK + 2)?;
            }
        }
        let decoding = decoding(&self.seqs);
        match decoding.as_slice() {
            [] => Ok(0),
            [index] => self.spec_step(*index, out, ahead),
            many if self.spec_floor.is_some()
                && self.exchange.is_some()
                && 2 * many.len() <= R_MAX
                && many.len() <= MAX_SEQS =>
            {
                self.spec_multi_step(many, out, ahead)
            }
            many => {
                let mut rows = 0;
                for group in many.chunks(R_MAX) {
                    rows += self.plain_step(group, out, &mut *ahead)?;
                }
                Ok(rows)
            }
        }
    }

    /// Starts the Engram reads of sequence `index`'s anchor row, to run under its draft.
    fn read_ahead(&mut self, index: usize) -> Result<(), String> {
        let (Some(reader), Some(reads), Some(staging), true) = (
            self.inputs.engram,
            self.pools.decode.as_ref(),
            self.staging.as_ref(),
            self.early,
        ) else {
            return Ok(());
        };
        let seq = &self.seqs[index];
        let Some(gather) = self.device.engram_plan(
            self.inputs.config,
            reader,
            &seq.tokens,
            seq.pos,
            1,
            0,
            BLOCK + 1,
        )?
        else {
            return Ok(());
        };
        let task = reads.submit(gather, Staging::of(staging), ENGRAM_AHEAD_THREADS)?;
        self.ahead = Some((seq.id, seq.pos, task));
        Ok(())
    }

    /// The speculative step of a lone sequence: draft from the anchor, verify [anchor, drafts]
    /// as one rows step, accept the drafts while each equals the verify pick before it, and emit
    /// the accepted drafts plus the pick after them.
    fn spec_step(
        &mut self,
        index: usize,
        out: &mut Vec<Emission>,
        ahead: Ahead<'_>,
    ) -> Result<usize, String> {
        let (rows, inputs, draft, k) = (self.rows, self.inputs, self.draft, self.k);
        let lane = self.seqs[index].lane.ok_or("decode: no lane")?;
        self.enter(lane)?;
        let (id, pos) = (self.seqs[index].id, self.seqs[index].pos);
        if self
            .ahead
            .as_ref()
            .is_some_and(|(seq, at, _)| *seq != id || *at != pos)
        {
            self.wait_ahead()?;
        }
        if self.ahead.is_none() {
            self.read_ahead(index)?;
        }
        let reads = self.ahead.take().map(|(_, _, task)| task);
        let sampler = self.seqs[index].sampler;
        let anchor = self.seqs[index].tokens[pos];
        let anchor_row = match self.exchange {
            Some(_) => Vec::new(),
            None => gather_rows(self.device.cuda, inputs, &self.counter, &[anchor])?,
        };
        // A fixed depth knows the verify's rows before the draft: their embedding is queued
        // behind the chain.
        let queued = match (self.exchange, self.seqs[index].policy.adaptive) {
            (Some(_), false) => k + 1,
            _ => 0,
        };
        if queued > 0 {
            self.device.check_begin(pos, queued, rows.cap)?;
        }
        let t0 = Instant::now();
        let (drafted, probs) = self.device.draft_block(
            rows,
            draft,
            inputs.tail,
            inputs.transport,
            pos,
            anchor,
            match self.exchange {
                Some(exchange) => AnchorRow::Device {
                    embed: inputs.embed.0,
                    exchange,
                },
                None => AnchorRow::Host(&anchor_row),
            },
            k,
            queued,
            &sampler,
        )?;
        let t1 = Instant::now();
        let seq = &mut self.seqs[index];
        let depth = seq.policy.depth(&probs);
        if queued > 0 && depth + 1 != queued {
            return Err(format!(
                "decode: {queued} verify rows queued for depth {depth}"
            ));
        }
        let drafts = drafted[..depth].to_vec();
        seq.verified += depth;
        let n = depth + 1;
        seq.tokens.truncate(pos + 1);
        seq.tokens.extend_from_slice(&drafts);
        let gather = match inputs.engram {
            Some(reader) if self.early => self.device.engram_plan(
                inputs.config,
                reader,
                &seq.tokens,
                pos + 1,
                depth,
                1,
                BLOCK + 1,
            )?,
            Some(reader) => {
                self.device
                    .engram_plan(inputs.config, reader, &seq.tokens, pos, n, 0, BLOCK + 1)?
            }
            None => None,
        };
        let te = Instant::now();
        match self.exchange {
            Some(_) if queued > 0 => {
                self.device.begin_rows_queued(rows, pos, n)?;
            }
            Some(exchange) => {
                self.device.begin_rows_device(
                    rows,
                    pos,
                    inputs.embed.0,
                    draft.ids(),
                    n,
                    exchange,
                )?;
            }
            None => {
                let mut hidden = anchor_row.clone();
                hidden.extend(gather_rows(
                    self.device.cuda,
                    inputs,
                    &self.counter,
                    &drafts,
                )?);
                self.device.begin_rows(rows, pos, &hidden)?;
            }
        }
        let tl = Instant::now();
        let waited = self.device.spec_layers_gathered(
            rows,
            inputs.staged,
            &inputs.config.engram_layer_ids,
            self.pools.decode.as_ref(),
            gather.zip(self.staging.as_ref().map(Staging::of)),
            self.staging.as_ref(),
            reads,
            pos,
            n,
        )?;
        self.send_ahead(ahead)?;
        let picks = self.device.head_rows_sampled(
            rows,
            inputs.tail,
            inputs.transport,
            n,
            &sampler,
            pos + 1,
        )?;
        let th = Instant::now();
        let seq = &mut self.seqs[index];
        seq.phase_ms[0] += (tl - te).as_secs_f64() * 1e3;
        seq.phase_ms[1] += (te - t1).as_secs_f64() * 1e3;
        seq.phase_ms[2] += (th - tl).as_secs_f64() * 1e3;
        seq.phase_ms[3] += waited;
        let mut m = 0usize;
        while m < depth && drafts[m] == picks[m].0 {
            m += 1;
        }
        let bonus = picks[m].0;
        seq.policy.observe(&probs, depth, m);
        seq.steps += 1;
        seq.accepted += m;
        let mut step = drafts[..m].to_vec();
        step.push(bonus);
        let emitted = seq.emitted() - drafts.len();
        let finish = cut_step(&mut step, emitted, seq.max_new, &seq.stop_ids, seq.exact);
        if self.print {
            println!(
                "spec step {}: pos {pos} drafts {drafted:?} probs {probs:.3?} depth {depth} verify {:?} accepted {m}",
                seq.steps,
                picks.iter().map(|pick| pick.0).collect::<Vec<_>>()
            );
        }
        seq.draft_ms += (t1 - t0).as_secs_f64() * 1e3;
        seq.tokens.truncate(pos + 1);
        seq.tokens.extend_from_slice(&step);
        if let Some(finish) = finish {
            seq.verify_ms += t1.elapsed().as_secs_f64() * 1e3;
            self.finish(index, finish, pos, step, out)?;
            return Ok(n);
        }
        self.device.rewind_rows(pos + m + 1);
        self.device.draft_seed(rows, draft, pos, m + 1)?;
        self.seqs[index].pos = pos + m + 1;
        self.read_ahead(index)?;
        let seq = &mut self.seqs[index];
        seq.verify_ms += t1.elapsed().as_secs_f64() * 1e3;
        out.push(Emission {
            seq: id,
            tokens: step,
            done: None,
        });
        Ok(n)
    }

    /// One plain rows step over the sequences `group` (up to R_MAX, each its own lane): each
    /// sequence's anchor is one row at its own position, sampled under its own sampler.
    fn plain_step(
        &mut self,
        group: &[usize],
        out: &mut Vec<Emission>,
        ahead: Ahead<'_>,
    ) -> Result<usize, String> {
        let (rows, inputs, draft) = (self.rows, self.inputs, self.draft);
        let exchange = self
            .exchange
            .ok_or("decode: several sequences need the exchange")?;
        // The step's reads land in the decode staging.
        self.wait_ahead()?;
        let mut segs = Vec::with_capacity(group.len());
        let mut parts = Vec::new();
        for (row, &index) in group.iter().enumerate() {
            let seq = &self.seqs[index];
            let lane = seq.lane.ok_or("decode: no lane")?;
            segs.push(Seg {
                lane,
                p0: seq.pos,
                n: 1,
                row0: row,
            });
            if let Some(reader) = inputs.engram
                && let Some(part) = self.device.engram_plan(
                    inputs.config,
                    reader,
                    &seq.tokens,
                    seq.pos,
                    1,
                    row,
                    R_MAX,
                )?
            {
                parts.push(part);
            }
        }
        let gather = EngramGather::merge(parts)?;
        let ids: Vec<u8> = group
            .iter()
            .flat_map(|&index| {
                let seq = &self.seqs[index];
                seq.tokens[seq.pos].to_le_bytes()
            })
            .collect();
        self.device
            .cuda
            .copy_htod(&Deviceptr(rows.b.row_ids), &ids)?;
        let n = self.device.begin_rows_segments(
            rows,
            &segs,
            inputs.embed.0,
            rows.b.row_ids,
            exchange,
        )?;
        let (room, hold) = (self.room(), self.hold_after());
        let result = (|| {
            self.device.spec_layers_gathered(
                rows,
                inputs.staged,
                &inputs.config.engram_layer_ids,
                self.pools.decode.as_ref(),
                gather.zip(self.staging.as_ref().map(Staging::of)),
                self.staging.as_ref(),
                None,
                0,
                n,
            )?;
            ahead(room, hold)?;
            let samplers: Vec<(Sampler, usize)> = group
                .iter()
                .map(|&index| (self.seqs[index].sampler, self.seqs[index].pos + 1))
                .collect();
            let picks =
                self.device
                    .head_rows_multi(rows, inputs.tail, inputs.transport, &samplers)?;
            self.device.draft_seed(rows, draft, 0, n)?;
            Ok::<_, String>(picks)
        })();
        self.device.clear_segments();
        let picks = result?;
        for (row, &index) in group.iter().enumerate() {
            let seq = &mut self.seqs[index];
            let pos = seq.pos;
            seq.steps += 1;
            let mut step = vec![picks[row].0];
            let finish = cut_step(
                &mut step,
                seq.emitted(),
                seq.max_new,
                &seq.stop_ids,
                seq.exact,
            );
            seq.tokens.extend_from_slice(&step);
            match finish {
                Some(finish) => self.finish(index, finish, pos + 1, step, out)?,
                None => {
                    seq.pos = pos + 1;
                    out.push(Emission {
                        seq: seq.id,
                        tokens: step,
                        done: None,
                    });
                }
            }
        }
        Ok(n)
    }
    /// One speculative step over the sequences `group` (each its own lane, at most R_MAX / 2 and
    /// MAX_SEQS): one draft pass drafts every sequence's block from its anchor
    /// ([DeviceLayer::draft_blocks]), the step's R_MAX rows go to the anchors and then to the drafts
    /// most likely accepted ([`share_rows`]), and one rows step verifies every sequence's
    /// [anchor, drafts] as its own segment. Each sequence then accepts as [Self::spec_step] does.
    fn spec_multi_step(
        &mut self,
        group: &[usize],
        out: &mut Vec<Emission>,
        ahead: Ahead<'_>,
    ) -> Result<usize, String> {
        let (rows, inputs, draft, k) = (self.rows, self.inputs, self.draft, self.k);
        let exchange = self
            .exchange
            .ok_or("decode: several sequences need the exchange")?;
        let floor = self.spec_floor.unwrap_or(0.0);
        // The step's reads land in the decode staging.
        self.wait_ahead()?;
        let t0 = Instant::now();
        let blocks = group
            .iter()
            .map(|&index| {
                let seq = &self.seqs[index];
                Ok(DraftBlock {
                    lane: seq.lane.ok_or("decode: no lane")?,
                    d: seq.pos,
                    anchor: seq.tokens[seq.pos],
                    sampler: seq.sampler,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        // No sequence can take more draft rows than the anchors leave, so none drafts deeper.
        let depth = k.min(R_MAX - group.len());
        let drafted =
            self.device
                .draft_blocks(rows, draft, inputs.tail, inputs.embed.0, &blocks, depth)?;
        let t1 = Instant::now();
        let chains: Vec<Vec<f64>> = group
            .iter()
            .zip(&drafted)
            .map(|(&index, (_, probs))| self.seqs[index].policy.chain(probs))
            .collect();
        let depths = share_rows(&chains, R_MAX - group.len(), floor);
        let mut segs = Vec::with_capacity(group.len());
        let mut ids = Vec::with_capacity(R_MAX * 4);
        let mut parts = Vec::new();
        let mut row0 = 0;
        for ((&index, (drafts, _)), &depth) in group.iter().zip(&drafted).zip(&depths) {
            let seq = &mut self.seqs[index];
            let (pos, n) = (seq.pos, depth + 1);
            seq.verified += depth;
            seq.tokens.truncate(pos + 1);
            seq.tokens.extend_from_slice(&drafts[..depth]);
            segs.push(Seg {
                lane: seq.lane.ok_or("decode: no lane")?,
                p0: pos,
                n,
                row0,
            });
            ids.extend(
                seq.tokens[pos..pos + n]
                    .iter()
                    .flat_map(|id| id.to_le_bytes()),
            );
            if let Some(reader) = inputs.engram
                && let Some(part) = self.device.engram_plan(
                    inputs.config,
                    reader,
                    &seq.tokens,
                    pos,
                    n,
                    row0,
                    R_MAX,
                )?
            {
                parts.push(part);
            }
            row0 += n;
        }
        let gather = EngramGather::merge(parts)?;
        self.device
            .cuda
            .copy_htod(&Deviceptr(rows.b.row_ids), &ids)?;
        let n = self.device.begin_rows_segments(
            rows,
            &segs,
            inputs.embed.0,
            rows.b.row_ids,
            exchange,
        )?;
        let (room, hold) = (self.room(), self.hold_after());
        let result = (|| {
            self.device.spec_layers_gathered(
                rows,
                inputs.staged,
                &inputs.config.engram_layer_ids,
                self.pools.decode.as_ref(),
                gather.zip(self.staging.as_ref().map(Staging::of)),
                self.staging.as_ref(),
                None,
                0,
                n,
            )?;
            ahead(room, hold)?;
            let samplers: Vec<(Sampler, usize)> = group
                .iter()
                .zip(&segs)
                .flat_map(|(&index, seg)| {
                    let sampler = self.seqs[index].sampler;
                    (0..seg.n).map(move |row| (sampler, seg.p0 + 1 + row))
                })
                .collect();
            let picks =
                self.device
                    .head_rows_multi(rows, inputs.tail, inputs.transport, &samplers)?;
            // Every verify row seeds its lane's drafter rings: a rejected draft's row sits where
            // the next draft block writes its own rows, before anything reads it.
            self.device.draft_seed(rows, draft, 0, n)?;
            Ok::<_, String>(picks)
        })();
        self.device.clear_segments();
        let picks = result?;
        let (draft_ms, verify_ms) = (
            (t1 - t0).as_secs_f64() * 1e3,
            t1.elapsed().as_secs_f64() * 1e3,
        );
        for ((&index, (drafts, probs)), seg) in group.iter().zip(&drafted).zip(&segs) {
            let depth = seg.n - 1;
            let seq = &mut self.seqs[index];
            let pos = seq.pos;
            let mut m = 0usize;
            while m < depth && drafts[m] == picks[seg.row0 + m].0 {
                m += 1;
            }
            seq.policy.observe(probs, depth, m);
            seq.steps += 1;
            seq.accepted += m;
            seq.draft_ms += draft_ms;
            seq.verify_ms += verify_ms;
            let mut step = drafts[..m].to_vec();
            step.push(picks[seg.row0 + m].0);
            let emitted = seq.emitted() - depth;
            let finish = cut_step(&mut step, emitted, seq.max_new, &seq.stop_ids, seq.exact);
            if self.print {
                println!(
                    "spec multi step {}: seq {} pos {pos} drafts {drafts:?} probs {probs:.3?} depth {depth} accepted {m}",
                    seq.steps, seq.id
                );
            }
            seq.tokens.truncate(pos + 1);
            seq.tokens.extend_from_slice(&step);
            if let Some(finish) = finish {
                self.finish(index, finish, pos, step, out)?;
                continue;
            }
            seq.pos = pos + m + 1;
            let id = seq.id;
            self.enter(seg.lane)?;
            self.device.rewind_rows(pos + m + 1);
            out.push(Emission {
                seq: id,
                tokens: step,
                done: None,
            });
        }
        Ok(n)
    }
}

/// A chunk's token ids as the embedding upload takes them: an image span's positions are the
/// placeholder's row (the tower rank overwrites them before the exchange).
/// The rows of the next bulk chunk with `left` rows of the prompt to go: `budget`, or all of them
/// when they fit `most`, so no short tail takes a chunk (and a read of every expert's weights) of
/// its own.
fn bulk_rows(left: usize, budget: usize, most: usize) -> usize {
    if left <= most { left } else { budget }
}

/// How many leading positions of `wanted` a lane holding `held` (its image spans `spans`) holds
/// alike: the common prefix, cut at the first of the prompt's `images` the lane does not hold alike
/// (equal span tokens are not equal images).
fn common_prefix(
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

/// A bulk chunk's reads in flight and the pinned buffer they land in.
fn chunk_reads<'c, 'r, 's>(
    current: Option<&'c PrefillReads<'r>>,
    staging: Option<&'s [HostMapped; 2]>,
) -> Option<(&'c GatherTask<'r>, &'s HostMapped)> {
    current
        .zip(staging)
        .map(|(current, staging)| (&*current.task, &staging[current.buffer]))
}

/// A bulk rows state's pinned Engram staging pair, when the run reads Engram.
fn bulk_staging<'s>(state: &'s RowsState, inputs: &SpecInputs<'_>) -> Option<&'s [HostMapped; 2]> {
    state
        .bulk
        .as_ref()
        .and_then(|bulk| bulk.engram_staging.as_ref())
        .filter(|_| inputs.engram.is_some())
}

fn upload_ids(tokens: &[u32]) -> Vec<u8> {
    tokens
        .iter()
        .map(|&id| {
            if id == HASH_SENTINEL {
                IMAGE_TOKEN_ID
            } else {
                id
            }
        })
        .flat_map(u32::to_le_bytes)
        .collect()
}

impl Drop for Batch<'_, '_, '_> {
    fn drop(&mut self) {
        // Reads still in flight land in the staging buffers: wait for them before they are freed.
        let _ = self.wait_ahead();
        let _ = self.wait_prefill_next();
        let _ = self.device.cuda.synchronize();
        self.device.clear_segments();
        if let Some(staging) = self.staging.take() {
            self.device.cuda.free_host_mapped(&staging);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DecodeShare, bulk_rows, common_prefix, share_rows};

    #[test]
    fn the_decoders_get_what_the_prefill_deadlines_leave() {
        let share = DecodeShare {
            most: 0.5,
            deadline_ms: 200_000.0,
        };
        let allowed = |prefills: &[(usize, f64)], row_ms| share.allowed(prefills, row_ms);
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        // 100K rows at 0.75 ms need 75 s of the 200 s: capped at half.
        assert!(close(allowed(&[(100_000, 0.0)], 0.75), 0.5));
        // 200K rows need 150 s: a quarter of the time is spare.
        assert!(close(allowed(&[(200_000, 0.0)], 0.75), 0.25));
        // The prompt after it waits for both: 50 s of its 100 s are spare, or 10 s of its 60 s.
        assert!(close(
            allowed(&[(40_000, 0.0), (40_000, 100_000.0)], 0.625),
            0.5
        ));
        assert!(close(
            allowed(&[(40_000, 0.0), (40_000, 140_000.0)], 0.625),
            1.0 / 6.0
        ));
        // Late: every tick prefills.
        assert!(close(allowed(&[(200_000, 60_000.0)], 0.75), 0.0));
        assert!(close(allowed(&[], 0.75), 0.5));
    }

    #[test]
    fn a_short_tail_rides_with_the_chunk_before_it() {
        assert_eq!(bulk_rows(540, 512, 768), 540);
        assert_eq!(bulk_rows(769, 512, 768), 512);
        assert_eq!(bulk_rows(5000, 4096, 4096), 4096);
        assert_eq!(bulk_rows(100, 4096, 4096), 100);
    }

    #[test]
    fn a_lane_shares_the_prompt_up_to_the_first_difference() {
        assert_eq!(common_prefix(&[1, 2, 3, 4], &[], &[1, 2, 3, 9, 9], &[]), 3);
        assert_eq!(common_prefix(&[1, 2], &[], &[1, 2, 3], &[]), 2);
        assert_eq!(common_prefix(&[], &[], &[1], &[]), 0);
    }

    #[test]
    fn rows_go_to_the_likeliest_drafts_down_to_the_floor() {
        let chains = [vec![0.9, 0.8, 0.3], vec![0.95, 0.5, 0.1], vec![0.15]];
        // 0.95 (b), 0.9 (a), 0.8 (a), 0.5 (b), 0.3 (a); 0.15 and 0.1 are under the floor.
        assert_eq!(share_rows(&chains, 8, 0.2), [3, 2, 0]);
        assert_eq!(share_rows(&chains, 3, 0.2), [2, 1, 0]);
        assert_eq!(share_rows(&chains, 0, 0.2), [0, 0, 0]);
        // A tie goes to the lower sequence.
        assert_eq!(share_rows(&[vec![0.5], vec![0.5]], 1, 0.0), [1, 0]);
    }
}

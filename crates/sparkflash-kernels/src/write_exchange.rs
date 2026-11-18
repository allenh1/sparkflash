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

//! The direct-RDMA exchange (the GB10 doorbell), opened once per process and shared by every
//! collective of the run.
//!
//! The DeepSeek arm's exchange, moved here so every arm that rings the doorbell runs one copy of
//! it: the open, the slab, the selection switch and the two launches. The device kernels are
//! kernels/opt_exchange_sm121.cuh, which the DeepSeek opt_misc image and the Inkling image both
//! compile; each arm hands [shared] the image it loads them from and the widest row it sends.
//!
//! It cannot be opened per step. Opening the exchange means a page-locked registration, a QP
//! mesh, a pump thread and a control-plane rendezvous with the other ranks - tens of milliseconds
//! of work that would swamp what a collective costs. So the run opens it on its first collective
//! (or at bring-up) and every later step borrows it.
//!
//! A collective is then host calls that never wait: [ExchangeState::stage] takes the round's
//! sequence number and launches the stage kernel, and the caller launches what reads the round's
//! row set behind it (a fold, or [ExchangeState::collect_slots]). The stage kernel copies the row
//! into the slab, rings the doorbell and waits on the device for every peer; the exchange's pump
//! thread posts the writes when the doorbell rings; the kernel behind it starts when the stage
//! grid completes. A round that cannot complete is reported by [check], which the forward pass
//! calls once the device has drained.
//!
//! Selection is one environment variable and it is never a silent fallback:
//!
//! - unset or empty selects the exchange, and stands down to the NCCL group with one printed line
//!   when the run was never told where its peers are.
//! - `SPARKFLASH_EXCHANGE=rdma` selects the exchange and refuses to start without it.
//! - `SPARKFLASH_EXCHANGE=nccl` leaves the NCCL group in place.
//! - anything else is refused by name.
//!
//! The address table is `SPARK_COMM_PEERS` (one address per rank, in rank order) and the control
//! port is `SPARK_COMM_CONTROL_PORT`; the collectives themselves run on the rails
//! `select_rails_from_live_fabric` resolves (`SPARK_COMM_RAIL_COUNT` of them, two unless told
//! otherwise), and the GIDs come from each rank's own record, so the two do not have to be the
//! same interface. A TP group's launcher passes its own rank table and a control port beside its
//! own ([shared_in], [GroupTable]), so a group opens the exchange by default without being told
//! twice; the variables still win when set. [ExchangeState::exchange_on] is one slot exchange on
//! any stream, the device leg every TP group's collective shares (sparkflash_ops::doorbell).

use std::sync::{Mutex, OnceLock};

use sparkflash_comm::rdma::{ExchangeStats, RdmaConfig, SlabLayout, WriteExchange};

use crate::cuda_driver::{Cuda, HostMapped, Kernel, Module, StreamEvent};

/// The switch that selects the exchange over the NCCL group.
pub const EXCHANGE_ENV: &str = "SPARKFLASH_EXCHANGE";

/// The stage kernel's symbol in an image that carries kernels/opt_exchange_sm121.cuh.
pub const STAGE_SYMBOL: &str = "opt_exchange_stage_bf16";

/// The collect kernel's symbol in the same image.
pub const COLLECT_SYMBOL: &str = "opt_exchange_collect";

/// The widest round a model arm's collectives send through the exchange by default, in bytes: a
/// decode step's residual sums and logit gather fit (Inkling's logit slice is 200,060 bytes a rank
/// at TP4), and so does a prefill chunk of a few rows; a wider payload stays on the NCCL group.
pub const DOORBELL_ROW_CAP: usize = 256 * 1024;

/// Arrival flags the exchange kernel can watch: one warp, a lane to a flag.
const KERNEL_FLAG_LANES: usize = 32;

/// The stage kernel's block: one block, so "every byte copied" is a `__syncthreads()`.
const STAGE_THREADS: u32 = 1024;

/// The collect kernel's block.
const COLLECT_THREADS: u32 = 256;

/// One submitted round: what the exchange kernel is launched with.
#[derive(Debug, Clone, Copy)]
pub struct Round {
    /// The round's sequence number, which the kernel rings the doorbell with and waits for.
    pub seq: u64,
    /// The device alias of the round's row set: one row per rank at `rank * row_bytes`, which is
    /// the fold's gather addressing.
    pub set: u64,
    /// The device alias of this rank's row inside that set, which the kernel fills.
    pub row: u64,
    /// The device alias of the control words: per rail an arrival flag per rank, then the
    /// doorbell, then the abort word.
    pub control: u64,
    /// Rails the row travels over, which is how many flags per rank the kernel waits for.
    pub rails: u64,
    /// The device alias of the published row lengths: one word per peer per round, which the
    /// kernel compares against this round's row size before the fold reads a byte.
    pub lengths: u64,
    /// Rounds of published length kept per peer, the modulus the kernel indexes with.
    pub length_ring: u64,
}

/// Everything one process needs to run the exchange, kept together so the registration outlives
/// the exchange that points at it.
pub struct ExchangeState {
    /// The exchange itself. Declared first so its pump stops and it deregisters before the slab.
    exchange: WriteExchange,
    /// The pinned slab the peers write into.
    slab: HostMapped,
    /// Where the rows and control words live inside the slab.
    layout: SlabLayout,
    /// `opt_exchange_stage_bf16`: stages the row, rings the doorbell, waits for the peers.
    kernel: Kernel,
    /// `opt_exchange_collect`: a slot exchange's copy of the peers' slots out of the slab.
    collect: Kernel,
    /// The modules the kernels were loaded from; they must outlive the kernels.
    _modules: [Module; 2],
    /// The raw stream the last [Self::exchange_on] round was launched on (0, the default stream,
    /// before the first).
    stream: u64,
    /// The event that orders a round on a new stream behind the rounds on the last one, made on
    /// the first switch.
    switch: Option<StreamEvent>,
}

// SAFETY: every field is plain data, a page-locked allocation the driver owns, or a CUDA function
// handle. The handle is only ever passed back to the driver by the thread holding the mutex, and
// the driver documents that a launched kernel and its handle may be used from one thread at a time,
// which the mutex guarantees. WriteExchange and HostMapped are Send in their own right.
unsafe impl Send for ExchangeState {}

impl ExchangeState {
    /// The exchange kernel.
    #[must_use]
    pub fn kernel(&self) -> Kernel {
        self.kernel
    }

    /// The kernel that copies a slot exchange's peers' slots out of the round's row set.
    #[must_use]
    pub fn collect(&self) -> Kernel {
        self.collect
    }

    /// The largest row a round can carry.
    #[must_use]
    pub fn row_cap(&self) -> usize {
        self.layout.row_cap()
    }

    /// Marks the exchange failed, for a caller that reserved a round and could not launch its
    /// kernel. Every later submit and check reports the reason instead of running on state this
    /// backend can no longer vouch for.
    pub fn fail(&self, reason: &str) {
        self.exchange.fail(reason);
    }

    /// Starts one exchange of `row_bytes` bytes and returns what its kernel is launched with. It
    /// does not wait for the device or the fabric; the kernels must be launched in submit order.
    ///
    /// # Errors
    /// The layout's refusal for a row that does not fit, or the first failure once any round has
    /// failed.
    pub fn submit(&mut self, row_bytes: usize) -> Result<Round, String> {
        let seq = self
            .exchange
            .submit(row_bytes)
            .map_err(|error| error.to_string())?;
        let parity = WriteExchange::parity(seq);
        let set = self
            .layout
            .set_offset(parity)
            .map_err(|error| error.to_string())?;
        let row = self
            .layout
            .row_offset(self.exchange.rank(), row_bytes, parity)
            .map_err(|error| error.to_string())?;
        let base = self.slab.device().0;
        Ok(Round {
            seq,
            set: base + set as u64,
            row: base + row as u64,
            control: base + self.layout.flags_offset() as u64,
            rails: self.layout.rails() as u64,
            lengths: base + self.layout.lengths_offset() as u64,
            length_ring: self.layout.length_ring() as u64,
        })
    }

    /// Submits one round of the `row_bytes` at the device address `row` and launches the stage
    /// kernel over it on the default stream. Whatever reads the round's row set (a fold, or
    /// [Self::collect_slots]) is launched behind it by the caller, still holding the lock, so
    /// rounds reach the stream in the order they were submitted.
    ///
    /// # Errors
    /// The submit's refusal, or a stage launch the driver refused; the latter marks the exchange
    /// failed first, because the round is reserved and its doorbell would never ring.
    pub fn stage(
        &mut self,
        cuda: &Cuda,
        row: u64,
        row_bytes: usize,
        rank: usize,
        world: usize,
        what: &dyn std::fmt::Display,
    ) -> Result<Round, String> {
        self.stage_on(cuda, 0, row, row_bytes, rank, world, what)
    }

    /// [Self::stage] on the raw stream `stream` (0 is the default stream, through [Cuda::launch]).
    /// The caller keeps rounds in submit order on the device; [Self::exchange_on] does that across
    /// streams.
    ///
    /// # Errors
    /// As [Self::stage].
    #[allow(clippy::too_many_arguments)]
    fn stage_on(
        &mut self,
        cuda: &Cuda,
        stream: u64,
        row: u64,
        row_bytes: usize,
        rank: usize,
        world: usize,
        what: &dyn std::fmt::Display,
    ) -> Result<Round, String> {
        let round = self
            .submit(row_bytes)
            .map_err(|error| format!("direct-RDMA exchange: {error}"))?;
        // A launch that fails here leaves the round reserved and its doorbell un-rung. Telling the
        // exchange is not optional: without it the pump posts the skipped round from whatever
        // bytes are in the row, and the peers fold them.
        if let Err(error) = cuda.launch_on_raw(
            stream,
            self.kernel,
            (1, 1, 1),
            (STAGE_THREADS, 1, 1),
            &[
                row,
                round.row,
                row_bytes as u64,
                round.control,
                rank as u64,
                world as u64,
                round.seq,
                round.rails,
                round.lengths,
                round.length_ring,
            ],
        ) {
            self.fail(&format!(
                "{what}: the exchange kernel for round {} did not launch: {error}",
                round.seq
            ));
            return Err(error);
        }
        Ok(round)
    }

    /// A slot exchange's second half: copies every peer's `bytes` out of `round`'s row set into
    /// `slots + peer * bytes`, leaving this rank's own slot. Bits are moved and nothing is summed.
    ///
    /// # Errors
    /// A launch the driver refused.
    pub fn collect_slots(
        &self,
        cuda: &Cuda,
        round: &Round,
        slots: u64,
        bytes: usize,
        rank: usize,
        world: usize,
    ) -> Result<(), String> {
        self.collect_on(cuda, 0, round, slots, bytes, rank, world)
    }

    /// [Self::collect_slots] on the raw stream `stream`, behind the round's stage kernel there.
    ///
    /// # Errors
    /// A launch the driver refused.
    #[allow(clippy::too_many_arguments)]
    fn collect_on(
        &self,
        cuda: &Cuda,
        stream: u64,
        round: &Round,
        slots: u64,
        bytes: usize,
        rank: usize,
        world: usize,
    ) -> Result<(), String> {
        // The sixteen-byte copy when both buffers and the slot are aligned for it.
        let unit = if (round.set | slots | bytes as u64) & 15 == 0 {
            16usize
        } else {
            1
        };
        let blocks = (world * bytes / unit).div_ceil(COLLECT_THREADS as usize) as u32;
        cuda.launch_on_raw(
            stream,
            self.collect,
            (blocks, 1, 1),
            (COLLECT_THREADS, 1, 1),
            &[
                round.set,
                slots,
                bytes as u64,
                rank as u64,
                world as u64,
                unit as u64,
            ],
        )
    }

    /// One slot exchange on the raw stream `stream`, the device leg of a TP collective's
    /// `exchange_on_device`: this rank's `bytes` at `send` reach every peer's
    /// `slots + rank * bytes`, and every peer's land at this rank's `slots + peer * bytes`, behind
    /// whatever `stream` already holds. Bits are moved and nothing is summed, so the fold the
    /// caller launches behind it gives the bits the NCCL group's exchange gives.
    ///
    /// Rounds must reach the device in the order they were submitted - the row sets alternate by
    /// the round's parity and the doorbell is one sequence word - so a round on a stream other
    /// than the last round's first orders that stream behind the last one with an event. A run
    /// that keeps its collectives on one stream never pays for it.
    ///
    /// # Errors
    /// A payload wider than a round, a refused submit, or a launch or ordering the driver refused;
    /// a collect that does not launch marks the exchange failed, since its round is already rung.
    #[allow(clippy::too_many_arguments)]
    pub fn exchange_on(
        &mut self,
        cuda: &Cuda,
        stream: u64,
        send: u64,
        slots: u64,
        bytes: usize,
        rank: usize,
        world: usize,
        what: &dyn std::fmt::Display,
    ) -> Result<(), String> {
        if bytes > self.row_cap() {
            return Err(format!(
                "{what}: {bytes} bytes are wider than the exchange's {} -byte round",
                self.row_cap()
            ));
        }
        if stream != self.stream {
            let event = match &self.switch {
                Some(event) => event,
                None => self.switch.insert(cuda.stream_event()?),
            };
            cuda.order_raw(event, self.stream, stream)?;
            self.stream = stream;
        }
        let round = self.stage_on(cuda, stream, send, bytes, rank, world, what)?;
        self.collect_on(cuda, stream, &round, slots, bytes, rank, world)
            .inspect_err(|error| {
                self.fail(&format!(
                    "{what}: the collect kernel for round {} did not launch: {error}",
                    round.seq
                ));
            })
    }
}

/// Set once when the default stood down, so the reason is printed by one rank-visible line.
static WARNED: OnceLock<()> = OnceLock::new();

/// The process-wide exchange, opened on first use.
static SHARED: OnceLock<Result<Mutex<ExchangeState>, String>> = OnceLock::new();

/// What the opening arm said the exchange replaces, for the fall-through note.
static REPLACES: OnceLock<&'static str> = OnceLock::new();

/// What the run asked for, and how hard it asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    /// The exchange was asked for by name. A failure to open is an error: a run that asked for this
    /// arm must never end up measuring the one it was replacing.
    Required,
    /// Nothing was said, so the exchange is the default. If it cannot be configured on this run
    /// (no peer table, no usable rail) the group is used and the reason is printed once - a default
    /// must not refuse to start a run that has never been told where the peers are.
    Default,
    /// The group was asked for by name.
    Off,
}

/// Reads the switch once and keeps it.
static SELECTED: OnceLock<Result<Selection, String>> = OnceLock::new();

fn select_from_env() -> Result<Selection, String> {
    match std::env::var(EXCHANGE_ENV) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "" => Ok(Selection::Default),
            "rdma" | "write" | "direct" | "1" | "on" => Ok(Selection::Required),
            "0" | "nccl" | "off" => Ok(Selection::Off),
            other => Err(format!(
                "{EXCHANGE_ENV}={other:?} is not a switch this build knows: use rdma for the direct-RDMA exchange, nccl for the collective group, or unset for the default (the exchange)"
            )),
        },
        Err(_) => Ok(Selection::Default),
    }
}

/// The arm this run selected.
///
/// # Errors
/// Returns a message naming the value when it is not one this switch accepts; a misspelled switch
/// must not leave a run measuring the arm it was trying to replace. It is read once and cached,
/// and [validate] calls it during bring-up so a misspelled switch fails at start-up rather than
/// whenever the first collective happens to reach this path.
pub fn selection() -> Result<Selection, String> {
    SELECTED.get_or_init(select_from_env).clone()
}

/// Refuses a switch this build does not understand, at bring-up.
///
/// # Errors
/// As [selection].
pub fn validate() -> Result<(), String> {
    selection().map(|_| ())
}

/// Reports a round that could not complete, once any has failed.
///
/// A kernel that was waiting for a peer when its round was given up leaves the wait without
/// folding, so the step that contained it produced nothing usable. The forward pass calls this
/// after the device has drained and before it reads anything the step wrote.
///
/// # Errors
/// The exchange's first failure.
pub fn check() -> Result<(), String> {
    let Some(Ok(state)) = SHARED.get() else {
        return Ok(());
    };
    let state = state
        .lock()
        .map_err(|_| "the direct-RDMA exchange is poisoned by an earlier panic".to_owned())?;
    state.exchange.check().map_err(|error| error.to_string())
}

/// The reason this arm must not stay on the group here, or `None`.
///
/// `SPARKFLASH_EXCHANGE=rdma` is a promise that the exchange is the collective. A path that
/// silently keeps the group breaks that promise quietly, and the run reports the arm it did not
/// take. This turns it into an error where the choice is made. A slot wider than a round does not
/// refuse: it is the group's by construction, and [note_fall_through] reports it instead.
#[must_use]
pub fn refusal_here() -> Option<String> {
    match selection() {
        Ok(Selection::Required) => Some(format!(
            "the direct-RDMA exchange is required ({EXCHANGE_ENV}=rdma) and this path would leave the collective to the NCCL group, so the run would be measuring neither arm"
        )),
        _ => None,
    }
}

/// Whether this process opened the exchange.
#[must_use]
pub fn is_open() -> bool {
    matches!(SHARED.get(), Some(Ok(_)))
}

/// The paths that have already said they fell through to the group.
static FELL_THROUGH: OnceLock<Mutex<std::collections::BTreeSet<String>>> = OnceLock::new();

/// Reports once, per path, that a collective the exchange did not carry stays on the group.
///
/// Falling through is what a run does with a collective wider than a round; doing it *silently*
/// is what a run that reported an arm must not do, so the caller notes it once per path. It
/// cannot desynchronise the ranks: the width is a property of the step's shape, which is the same
/// on every rank. A run that never opened the exchange says nothing, because the group is the arm
/// it reported.
pub fn note_fall_through(path: &str, why: &str) {
    let seen = FELL_THROUGH.get_or_init(|| Mutex::new(std::collections::BTreeSet::new()));
    let Ok(mut seen) = seen.lock() else {
        return;
    };
    // A wide slot falls through once a layer per prefill step, so the repeated case - a path that
    // has already spoken - must not allocate to find out that it has.
    if seen.contains(path) {
        return;
    }
    seen.insert(path.to_owned());
    let replaces = REPLACES
        .get()
        .copied()
        .unwrap_or("the collectives its arm named");
    eprintln!(
        "direct-RDMA exchange: {path} stays on the NCCL group ({why}); the exchange is replacing {replaces}, and this collective is not one of them on this run"
    );
}

/// What the exchange's pump has done so far, or `None` when this process never opened it.
///
/// Also `None` while a collective holds the exchange: a worker reports this as it leaves, from the
/// thread that watches rank 0, and it must not wait behind a device thread whose peers are gone.
#[must_use]
pub fn stats() -> Option<ExchangeStats> {
    let Some(Ok(state)) = SHARED.get() else {
        return None;
    };
    state.try_lock().ok().map(|state| state.exchange.stats())
}

/// What an arm opens the exchange with.
#[derive(Debug, Clone, Copy)]
pub struct Opening {
    /// The widest row a round carries, in bytes. A collective wider than this is the group's.
    pub row_cap: usize,
    /// Which collectives move to the exchange and which stay on the group, for the open banner:
    /// a reader of the logs must not conclude that every collective moved.
    pub replaces: &'static str,
}

/// The group's own address table, for a launcher that already knows where every rank is: a TP
/// group's `--tp4-peers` and a control port beside its own. It stands in for `SPARK_COMM_PEERS`
/// and `SPARK_COMM_CONTROL_PORT` when those are unset, so every group opens the exchange without
/// being told twice; a run that sets them still wins.
#[derive(Debug, Clone)]
pub struct GroupTable {
    /// One control-plane address per rank, in rank order.
    pub peers: Vec<String>,
    /// The port rank 0's control plane listens on.
    pub control_port: u16,
}

/// The shared exchange, opened on the first call.
///
/// `cubin` returns the image that carries [STAGE_SYMBOL] and [COLLECT_SYMBOL]; it is called once,
/// inside the open and before the rendezvous. Returns `Ok(None)` when the environment does not ask
/// for the exchange or the world is a single rank (there is nothing to exchange).
///
/// # Errors
/// Returns the reason the open failed: a refused switch, a missing address table, no usable rail,
/// a missing kernel, or a failed registration. A run that asked for the exchange does not silently
/// fall back.
pub fn shared<C: AsRef<[u8]>>(
    cuda: &Cuda,
    world: usize,
    rank: usize,
    opening: Opening,
    cubin: impl FnOnce() -> Result<C, String>,
) -> Result<Option<&'static Mutex<ExchangeState>>, String> {
    shared_in(cuda, world, rank, opening, None, cubin)
}

/// [shared], with `table` as the address table when the environment names none.
///
/// A table counts as configuration: the group told every rank where its peers are, so a failed
/// open refuses the run (naming `SPARKFLASH_EXCHANGE=nccl`, the opt-out) instead of leaving one
/// rank on the NCCL group while its peers wait at the exchange's rendezvous.
///
/// # Errors
/// As [shared].
pub fn shared_in<C: AsRef<[u8]>>(
    cuda: &Cuda,
    world: usize,
    rank: usize,
    opening: Opening,
    table: Option<GroupTable>,
    cubin: impl FnOnce() -> Result<C, String>,
) -> Result<Option<&'static Mutex<ExchangeState>>, String> {
    let selection = selection()?;
    if selection == Selection::Off || world < 2 {
        return Ok(None);
    }
    // Whether this run was *told* where the peers are. That is the line between the two kinds of
    // failure, and it is not the same as which arm was selected. A run with no table at all has
    // said nothing, so every rank of it stands down together and the group is what it gets. A run
    // that named a table has committed, even if that table is malformed: a rank whose variables are
    // wrong must be refused, not quietly parked on the other collective while its peers rendezvous.
    // So the test is the presence of the variable, not whether it parsed.
    let from_env = std::env::var("SPARK_COMM_PEERS")
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false);
    let derived = !from_env && table.is_some();
    let configured = from_env || derived;
    let opened = SHARED.get_or_init(|| open(cuda, world, rank, opening, table, cubin));
    match opened {
        Ok(state) => Ok(Some(state)),
        Err(error) => {
            if derived {
                return Err(format!(
                    "{error}; the exchange is the default for every TP group (its peer table is the group's own), so set {EXCHANGE_ENV}=nccl to run this one on the NCCL group"
                ));
            }
            if selection == Selection::Required || configured {
                return Err(error.clone());
            }
            // Nothing was configured, so nothing can have committed. Say so once and keep going on
            // the group: a default must not refuse to start a run that never mentioned peers.
            WARNED.get_or_init(|| {
                eprintln!(
                    "direct-RDMA exchange: not used on this run ({error}); the collectives stay on the NCCL group. Set {EXCHANGE_ENV}=nccl to say so, or fix the above to use the exchange"
                );
            });
            Ok(None)
        }
    }
}

/// Opens the exchange and loads its kernels.
fn open<C: AsRef<[u8]>>(
    cuda: &Cuda,
    world: usize,
    rank: usize,
    opening: Opening,
    table: Option<GroupTable>,
    cubin: impl FnOnce() -> Result<C, String>,
) -> Result<Mutex<ExchangeState>, String> {
    let row_cap = opening.row_cap;
    let peers_set = std::env::var("SPARK_COMM_PEERS").is_ok_and(|value| !value.trim().is_empty());
    let config = match table {
        Some(table) if !peers_set => {
            RdmaConfig::from_env_with_peers(world, rank, table.peers).map(|mut config| {
                if std::env::var_os("SPARK_COMM_CONTROL_PORT").is_none() {
                    config.control_port = table.control_port;
                }
                config
            })
        }
        _ => RdmaConfig::from_env(world, rank),
    }
    .map_err(|error| {
        format!(
            "direct-RDMA exchange: {error} (SPARK_COMM_PEERS must name one address per rank, and SPARK_COMM_CONTROL_PORT must not collide with the NCCL bootstrap)"
        )
    })?;
    // A row is split over the rails the fabric configuration opens (SPARK_COMM_RAIL_COUNT, two
    // unless told otherwise): at a step's row sizes a round costs throughput, not latency.
    let layout = SlabLayout::new(world, row_cap, config.rail_count).map_err(|error| {
        format!(
            "direct-RDMA exchange: a {world}-rank slab of {row_cap} -byte rows over {} rail(s): {error}",
            config.rail_count
        )
    })?;
    // The exchange kernel watches the arrival flags with one warp, a lane to a flag.
    if world * config.rail_count > KERNEL_FLAG_LANES {
        return Err(format!(
            "direct-RDMA exchange: {world} ranks over {} rail(s) is {} arrival flags, and the exchange kernel watches at most {KERNEL_FLAG_LANES}",
            config.rail_count,
            world * config.rail_count
        ));
    }
    let total = layout
        .total_bytes()
        .map_err(|error| format!("direct-RDMA exchange: {error}"))?;
    // Everything that can fail happens before the rendezvous inside [WriteExchange::open], which
    // is the run-wide commitment point. Loading the kernels here, first, is what lets that be true:
    // a rank that loaded them afterwards could fail at the kernel and leave its peers, already past
    // the barrier, waiting for a fourth that had quietly stood down to the NCCL group.
    let cubin = cubin().map_err(|error| format!("direct-RDMA exchange kernel: {error}"))?;
    let (module, kernel) = cuda
        .load_named(cubin.as_ref(), STAGE_SYMBOL)
        .map_err(|error| format!("direct-RDMA exchange kernel: {error}"))?;
    let (collect_module, collect) = cuda
        .load_named(cubin.as_ref(), COLLECT_SYMBOL)
        .map_err(|error| format!("direct-RDMA exchange kernel: {error}"))?;
    let mut allocated = 0usize;
    let slab = cuda
        .alloc_host_mapped(total, &mut allocated)
        .map_err(|error| {
            format!("direct-RDMA exchange: could not pin {total} bytes of host memory: {error}")
        })?;
    let exchange = WriteExchange::open(&config, slab.host_ptr(), layout)
        .map_err(|error| format!("direct-RDMA exchange: {error}"))?;
    // The slab is deliberately not freed on a failed open. The endpoint table - this rank's MR
    // address and key - is published inside the open and before its rendezvous, so a failure after
    // that point can leave a peer past the barrier with this address in hand and a write in flight
    // into it. Freeing it there is a use-after-free across the fabric; leaking one pinned
    // allocation in a process that is about to serve on the group is the cheaper mistake, and the
    // process reclaims it on exit.
    // Said once, on the arm that took it: a run that reports a switch must report taking it, and
    // this line is the only positive evidence that the collective group was replaced.
    eprintln!(
        "direct-RDMA exchange: opened rank {rank} of {world} over {:?}, {} x {} -byte rows in a {total} -byte pinned slab at {:#x}, control port {}",
        exchange.rail_names(),
        layout.world(),
        layout.row_cap(),
        exchange.address(),
        config.control_port,
    );
    let _ = REPLACES.set(opening.replaces);
    eprintln!("direct-RDMA exchange: replacing {}", opening.replaces);
    Ok(Mutex::new(ExchangeState {
        exchange,
        slab,
        layout,
        kernel,
        collect,
        _modules: [module, collect_module],
        stream: 0,
        switch: None,
    }))
}

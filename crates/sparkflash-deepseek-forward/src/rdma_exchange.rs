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

//! The direct-RDMA exchange, opened once per process and shared by every layer.
//!
//! It cannot be opened per layer. DeviceLayer is rebuilt once per step, and opening the exchange
//! means a page-locked registration, a QP mesh, a pump thread and a control-plane rendezvous with
//! the other ranks - tens of milliseconds of work that would swamp what a collective costs. So the
//! run opens it on its first collective and every later step borrows it.
//!
//! A collective is then three host calls and none waits: [ExchangeState::submit] takes the
//! round's sequence number, and the caller launches [ExchangeState::kernel] with it and the fold
//! behind it. The kernel stages the row, rings the doorbell and waits on the device for every
//! peer; the exchange's pump thread posts the writes when the doorbell rings; the fold starts when
//! the kernel's grid completes. A round that cannot complete is reported by [check], which the
//! forward pass calls once the device has drained.
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
//! same interface.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use sparkflash_comm::rdma::{ExchangeStats, RdmaConfig, SlabLayout, WriteExchange};
use sparkflash_kernels::cuda_driver::{Cuda, HostMapped, Kernel, Module};

/// The switch that selects the exchange over the NCCL group.
pub const EXCHANGE_ENV: &str = "SPARKFLASH_EXCHANGE";

/// The largest row a step can send: `dim` (5120) times the most rows a step carries
/// ([crate::tp4_forward::rows::R_MAX]) as bf16.
///
/// `rows::R_MAX` is derived rather than written down, so raising it grows the slab instead of
/// pushing the widest verify rows onto the NCCL fallback without a word. The 5120 is not: it is the
/// DeepSeek-V4.1 hidden width, and the forward is generic over `dim`. A profile whose `dim` is
/// wider would find `row_bytes` above this cap, and each caller answers for itself: a slot of a run
/// that opened the exchange declines the round and reports the fall-through once
/// ([note_fall_through]), while the per-layer all-reduce never reaches the exchange at all - its
/// arm is guarded by the same 5120 - and either keeps the group or refuses, as the switch says.
/// Making the cap a function of the run's `dim` is the right fix if a second arm ever takes this
/// path.
const ROW_CAP: usize = 5120 * crate::tp4_forward::rows::R_MAX * 2;

/// Arrival flags the exchange kernel can watch: one warp, a lane to a flag.
const KERNEL_FLAG_LANES: usize = 32;

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
}

/// Set once when the default stood down, so the reason is printed by one rank-visible line.
static WARNED: OnceLock<()> = OnceLock::new();

/// The process-wide exchange, opened on first use.
static SHARED: OnceLock<Result<Mutex<ExchangeState>, String>> = OnceLock::new();

/// Whether the environment asks for the exchange.
///
/// # Errors
/// Returns a message naming the value when it is not one this switch accepts; a misspelled switch
/// must not leave a run measuring the arm it was trying to replace.
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
/// Returns a message naming the value when it is not one this switch accepts. It is read once and
/// cached, and [validate] calls it during bring-up so a misspelled switch fails at start-up rather
/// than whenever the first collective happens to reach this path.
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
/// take. This turns it into an error where the choice is made, and all three callers are in
/// `reduce_rows`: the rows>1 branch, the one-shot arm's fallback after the exchange declines the
/// reduce, and the arm that has no one-shot exchange to take at all. The slot path does not refuse:
/// a slot wider than a round is the group's by construction, and `note_fall_through` reports it
/// instead of failing the step.
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

/// The widest row a round carries, in bytes. A collective wider than this is the group's.
#[must_use]
pub fn row_cap() -> usize {
    ROW_CAP
}

/// The paths that have already said they fell through to the group.
static FELL_THROUGH: OnceLock<Mutex<std::collections::BTreeSet<String>>> = OnceLock::new();

/// Reports once, per path, that a collective the exchange did not carry stays on the group.
///
/// The exchange names the collectives it replaces: the per-layer one-shot all-reduce, when that
/// arm is loaded, and the step's one-shot slot exchanges. A slot wider than a round (`row_cap`) is
/// the group's by construction, and it is the one *slot exchange* a run that opened the exchange
/// still leaves behind: bulk prefill, the ring gathers and the overlapped collectives are other
/// arms, and the open banner names them. Falling through is what such a run does; doing it *silently* is what a
/// run that reported an arm must not do, so the caller notes it once per path. It cannot
/// desynchronise the ranks: the width is a property of the step's shape, which is the same on every
/// rank. A run that never opened the exchange says nothing, because the group is the arm it
/// reported - the switch left it in place, the world is a single rank where there is nothing to
/// exchange, or an unset default found no peer table and stood down with its own line.
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
    eprintln!(
        "direct-RDMA exchange: {path} stays on the NCCL group ({why}); the exchange carries the per-layer one-shot all-reduce when that arm is loaded, and the step's one-shot slot exchanges that fit a round - this collective is neither on this run"
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

/// The shared exchange, opened on the first call.
///
/// Returns `Ok(None)` when the environment does not ask for it or the world is a single rank
/// (there is nothing to exchange).
///
/// # Errors
/// Returns the reason the open failed: a refused switch, a missing address table, no usable rail,
/// or a failed registration. A run that asked for the exchange does not silently fall back.
pub fn shared(
    cuda: &Cuda,
    world: usize,
    rank: usize,
    cubin_dir: Option<&Path>,
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
    let configured = std::env::var("SPARK_COMM_PEERS")
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false);
    let opened = SHARED.get_or_init(|| open(cuda, world, rank, cubin_dir));
    match opened {
        Ok(state) => Ok(Some(state)),
        Err(error) => {
            if selection == Selection::Required || configured {
                return Err(error.clone());
            }
            // Nothing was configured, so nothing can have committed. Say so once and keep going on
            // the group: a default must not refuse to start a run that never mentioned peers.
            WARNED.get_or_init(|| {
                eprintln!(
                    "direct-RDMA exchange: not used on this run ({error}); the per-layer all-reduce stays on the NCCL group. Set {EXCHANGE_ENV}=nccl to say so, or fix the above to use the exchange"
                );
            });
            Ok(None)
        }
    }
}

/// Opens the exchange and loads its kernel.
fn open(
    cuda: &Cuda,
    world: usize,
    rank: usize,
    cubin_dir: Option<&Path>,
) -> Result<Mutex<ExchangeState>, String> {
    let directory = cubin_dir.ok_or_else(|| {
        format!(
            "{EXCHANGE_ENV} asked for the direct-RDMA exchange but this run was opened without a cubin directory, so its kernel cannot be loaded"
        )
    })?;
    let config = RdmaConfig::from_env(world, rank).map_err(|error| {
        format!(
            "direct-RDMA exchange: {error} (SPARK_COMM_PEERS must name one address per rank, and SPARK_COMM_CONTROL_PORT must not collide with the NCCL bootstrap)"
        )
    })?;
    // A row is split over the rails the fabric configuration opens (SPARK_COMM_RAIL_COUNT, two
    // unless told otherwise): at a step's row sizes a round costs throughput, not latency.
    let layout = SlabLayout::new(world, ROW_CAP, config.rail_count).map_err(|error| {
        format!(
            "direct-RDMA exchange: a {world}-rank slab of {ROW_CAP} -byte rows over {} rail(s): {error}",
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
    let (cubin, _sha) = crate::tp4_forward::cached_cubin(&directory.join("opt_misc_sm121.cubin"))
        .map_err(|error| format!("direct-RDMA exchange kernel: {error}"))?;
    let (module, kernel) = cuda
        .load_named(&cubin, "opt_exchange_stage_bf16")
        .map_err(|error| format!("direct-RDMA exchange kernel: {error}"))?;
    let (collect_module, collect) = cuda
        .load_named(&cubin, "opt_exchange_collect")
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
    // into it. Freeing it there is a use-after-free across the fabric; leaking one 5 MiB pinned
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
    // What this replaces is narrower than "the collective": the per-layer one-shot all-reduce
    // (the decode and verify rows) and the step's one-shot slot exchanges (the argmax, embedding,
    // Engram and draft gathers) go through the exchange. Bulk prefill chunks, the overlapped bulk
    // collective, the ring gathers and any run without a device reduce handle keep the NCCL group,
    // and a reader of these logs must not conclude that every collective moved.
    eprintln!(
        "direct-RDMA exchange: replacing the per-layer one-shot all-reduce and the step's one-shot slot exchanges; bulk prefill, the ring gathers and the overlapped collectives stay on NCCL"
    );
    Ok(Mutex::new(ExchangeState {
        exchange,
        slab,
        layout,
        kernel,
        collect,
        _modules: [module, collect_module],
    }))
}

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

//! The direct-RDMA exchange: one write per peer and rail into its pinned slot, an arrival flag
//! per rank and rail, and no collective library on the path.
//!
//! The collective this replaces is one NCCL group of point-to-point sends and receives. Measured on
//! this fabric that group costs 43 us; a registered write of a 10 KiB row costs 5.7 us and the wait
//! for all three peers adds 1.6 us. A c1 step's usual row is 60 KiB, though, and there the last
//! rank to ring waits 17 us on one rail and 11 us on two, so the exchange is ahead of the group by
//! microseconds a collective and not by the 35 the small row promised.
//! The numbers, the gate that made them possible and the shape below are in
//! docs/vendor-survey-spark-ds41f.md section 6.0.6 and docs/rdma-exchange.md.
//!
//! The fabric side is deliberately plain: the payload and the flag are two writes on one
//! reliable-connected QP, so a peer that can see the flag has already been sent the bytes in front
//! of it. Nothing here needs a receive queue, an immediate value or a completion on the peer.
//!
//! The slab is host memory, page-locked and mapped, because that is the memory this fabric can
//! register: ibv_reg_mr refuses device memory here with EBADF, and accepts a
//! cudaHostAlloc(MAPPED | PORTABLE) allocation. On GB10 the host and device aliases of that
//! allocation are the same physical memory, so the GPU fills the send row directly, the NIC reads
//! it, and the sum kernel reads the rows the peers' NICs wrote, with no copy in either direction.
//!
//! # Nobody on the host waits for the device
//!
//! The thread that launches kernels never blocks here. [WriteExchange::submit] takes a sequence
//! number and returns; the caller launches one kernel that stages the row, rings the doorbell,
//! waits on the device for every peer's arrival flag and then folds the rows. A pump thread this
//! module owns watches the doorbell and posts the writes. So the launch thread keeps queueing the
//! rest of the step while the collective is in flight, exactly as it does with the collective
//! library, and the device never drains to idle waiting for a host that was waiting for it. The
//! first version of this module had the launch thread spin on the doorbell, post, and spin on the
//! flags before it launched the fold, so the device drained once per collective and the handshake
//! cost more than the wire it was wrapped around.
//!
//! # Two row sets, at fixed places
//!
//! The slab holds **two** row sets, because one is not safe: a peer that runs a round ahead would
//! overwrite rows a slow rank is still folding. Round n uses set n & 1. The argument that two are
//! enough does not depend on the pump keeping up:
//!
//! - A peer writes set n & 1 again in round n + 2. It posts that round after its own fold of
//!   n + 1, which needed this rank's flag for n + 1, which this rank's pump posts only after the
//!   doorbell of n + 1, which the device rings only after its fold of n has finished.
//! - This rank rewrites its send row of set n & 1 in round n + 2, after its fold of n + 1, which
//!   needed every peer's flag for n + 1, which each peer posts after its fold of n, which needed
//!   this rank's flag for n, which travels behind the row on the same QP. So every byte of the row
//!   has been delivered before it is rewritten, and no completion has to be reaped first.
//!
//! Each set starts at a fixed offset and holds one row per rank at r * row_bytes:
//!
//! ```text
//! | set 0: row 0 | row 1 | ... | | set 1: row 0 | row 1 | ... | | flags per rail | doorbell | abort | sources |
//! ```
//!
//! A row's position inside its set uses the calling collective's row size and not the slab's cap,
//! which is what lets the fold keep its gather + peer * count * 2 addressing. The set's base must
//! not: a base that moved with the row size would let a 10 KiB round's second set land inside a
//! 40 KiB round's first, and consecutive rounds do change size at a step boundary.
//!
//! # Two rails
//!
//! With the handshake out of the way a round costs what the wire costs, and at the sizes a step
//! sends that is throughput and not latency: a rank sends its row to every peer, a c1 step's
//! 60 KiB row is 180 KiB out through one device, and one device carries about 100 Gb/s. The second
//! rail is independent (both at once cost what either costs alone), so each row is split in two on
//! a 64-byte boundary and each rail carries one part with its own flag behind it. A peer's row is
//! there when both of its flags are, which is why the flags are per rank and per rail; the kernel
//! and the pump wait for the least of them.
//!
//! # The flag's source
//!
//! The flag write reads its eight bytes from the registered slab when the NIC transmits it, not
//! when it is posted, so the word must not change underneath an unsent write. Each round publishes
//! from its own source word (a ring of [FLAG_SOURCES]); a rank cannot get more than one round
//! ahead of a peer that has not seen its flag, so a word is long sent before the ring comes back
//! to it.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{JoinHandle, Thread};
use std::time::{Duration, Instant};

use super::abi::{
    IbvQpCap, IbvQpTypeUnion, IbvRdmaWr, IbvSendWr, IbvSendWrTail, IbvSge, IbvWc, IbvWrUnion,
    access, send_flags, wc, wr_opcode,
};
use super::dataplane::{init_attr, min_path_mtu, rtr_attr, rts_attr};
use super::qpinfo::{ControlConfig, RailEndpoint, RankEndpoints, open_star};
use super::verbs::{Context, Cq, Mr, Pd, Qp, Verbs};
use super::{OpenedRail, RdmaConfig, open_rail, registered_at};
use crate::error::CommError;

/// Largest value a row's byte count may take on the wire (a u32 in the exchange kernel).
pub const MAX_ROW_BYTES: usize = u32::MAX as usize;

/// Bytes one control word occupies; both ends index them as u64.
const WORD_BYTES: usize = 8;
/// Row sets a slab carries: a round uses one, so a peer one round ahead writes the other.
const ROW_SETS: usize = 2;
/// Rails an exchange can be opened over; the control plane's endpoint record carries this many.
pub const MAX_RAILS: usize = 2;
/// Source words the arrival flag is published from, one per round in a ring.
pub const FLAG_SOURCES: usize = 64;

/// Rounds whose published row length is kept, per peer.
///
/// A ring rather than one word per peer: the check has to name the round it is about, and a peer
/// is allowed to be a round ahead. It is the same depth as [FLAG_SOURCES], which is far more than
/// the two-round window the row sets admit.
pub const LENGTH_RING: usize = FLAG_SOURCES;

/// Where a round's published word puts its sequence number.
///
/// The word is `(seq << LENGTH_SEQ_SHIFT) | row_bytes`. The size alone is a weak identity: two
/// rounds of the same size are indistinguishable, so a rank that inserts or drops one shifts every
/// later sequence on itself and its peers fold its next collective while the size check agrees.
/// Publishing the sequence turns that into a refusal that names the publisher.
const LENGTH_SEQ_SHIFT: u32 = 20;

/// The mask that keeps the row length out of the packed word. `row_cap` is 655 360, well inside.
const LENGTH_BYTES_MASK: u64 = (1 << LENGTH_SEQ_SHIFT) - 1;

/// The word a round publishes for itself.
#[must_use]
pub fn length_word(seq: u64, row_bytes: usize) -> u64 {
    (seq << LENGTH_SEQ_SHIFT) | (row_bytes as u64 & LENGTH_BYTES_MASK)
}
/// What every region of the slab is aligned to: the kernel copies rows sixteen bytes at a time and
/// both ends read the control words as u64.
const REGION_ALIGN: usize = 64;
/// Writes one round posts to each peer on each rail: the row's length, its part of the row, and the
/// arrival flag.
const WRITES_PER_PEER: usize = 3;

/// Outstanding sends a peer QP is sized for. A round posts three writes per peer on each rail, and the
/// pump reaps between rounds, so this only has to cover the few rounds it may fall behind.
const SEND_WR_DEPTH: u32 = 64;
/// Rounds the launch thread may run ahead of the pump. A decode step queues about two per layer
/// before it waits for the device, so this is several steps of headroom; past it submit waits.
const ROUND_RING: usize = 1024;
/// Doorbell reads between two looks at anything else, so a ring is seen within a few tens of ns.
const DOORBELL_POLLS: usize = 32;
/// How long the pump keeps spinning with nothing in flight before it parks. A generation's steps
/// follow each other within a few ms, so the pump stays hot through one and sleeps between them.
const IDLE_SPIN: Duration = Duration::from_millis(10);

/// The device's "peer published another length" tag in the abort word, matching
/// `OPT_EXCHANGE_ABORT_LENGTH` in kernels/opt_misc_sm121.cu. Bits 16-23 name the peer.
const ABORT_TAG_LENGTH: u64 = 0x4c45;

/// Where the abort word puts the peer that disagreed.
const ABORT_PEER_SHIFT: u32 = 16;

/// How many times the pump re-arms the doorbell deadline before it gives up.
///
/// The doorbell is rung by the device when it reaches the exchange kernel, and everything the
/// launch thread queued between [WriteExchange::submit] and that kernel sits inside the interval:
/// it measures the device's queue depth, not the fabric. A single expiry is therefore not a fault
/// and is retried; three in a row with the doorbell still short is a device that will not get
/// there, which is reported as a stuck launch rather than as a wire problem.
const DOORBELL_TRIP_LIMIT: u32 = 3;

/// The default doorbell deadline when SPARK_COMM_DOORBELL_TIMEOUT_MS is unset: four times the
/// control-plane timeout, floored at a minute and capped at five.
///
/// The floor and the cap both matter. The control-plane timeout is sized for the rendezvous, which
/// happens once, and a deployment sets it generously (120 s on this cluster); multiplying that by a
/// large factor and then by the trip limit holds a serving rank for the better part of an hour on a
/// stalled launch. The cap keeps the worst case reportable in minutes.
const DOORBELL_TIMEOUT_FACTOR: u32 = 4;

/// The shortest doorbell deadline, whatever the multiple works out to.
const DOORBELL_TIMEOUT_FLOOR: Duration = Duration::from_secs(60);

/// The longest default doorbell deadline.
const DOORBELL_TIMEOUT_CEILING: Duration = Duration::from_secs(300);

/// Where each part of the slab lives.
///
/// Kept as a plain value rather than loose offsets so the arithmetic that decides where a peer's
/// write lands is in one place, unit-tested, and identical on both ends of the link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlabLayout {
    /// Bytes one row may occupy.
    row_cap: usize,
    /// Ranks in the exchange.
    world: usize,
    /// Rails a row is split over, each with its own arrival flags.
    rails: usize,
    /// Bytes from one row set's base to the next: row_cap * world, aligned.
    set_stride: usize,
}

impl SlabLayout {
    /// The layout for a world of world ranks whose rows are at most row_cap bytes and travel over
    /// rails rails.
    ///
    /// # Errors
    /// Returns [CommError::InvalidLayout] for a zero world or row cap, a rail count outside
    /// 1..=[MAX_RAILS], or a slab whose size overflows.
    pub fn new(world: usize, row_cap: usize, rails: usize) -> Result<Self, CommError> {
        if world == 0 {
            return Err(CommError::InvalidLayout(
                "an exchange needs at least one rank".to_owned(),
            ));
        }
        if rails == 0 || rails > MAX_RAILS {
            return Err(CommError::InvalidLayout(format!(
                "an exchange over {rails} rails is outside 1..={MAX_RAILS}"
            )));
        }
        if row_cap == 0 || row_cap > MAX_ROW_BYTES {
            return Err(CommError::InvalidLayout(format!(
                "a row cap of {row_cap} bytes is outside 1..={MAX_ROW_BYTES}"
            )));
        }
        let overflow = || {
            CommError::InvalidLayout(format!(
                "a {world}-rank slab of {row_cap} -byte rows overflows the address space"
            ))
        };
        let set_stride = row_cap
            .checked_mul(world)
            .and_then(|rows| rows.checked_next_multiple_of(REGION_ALIGN))
            .ok_or_else(overflow)?;
        let layout = Self {
            row_cap,
            world,
            rails,
            set_stride,
        };
        layout.total_bytes()?;
        Ok(layout)
    }

    /// Bytes the whole slab needs.
    ///
    /// # Errors
    /// Returns [CommError::InvalidLayout] when the total does not fit a usize.
    pub fn total_bytes(&self) -> Result<usize, CommError> {
        // The flags of every rail, the doorbell, the abort word, the flag sources, and the
        // per-round row lengths.
        let lengths = self
            .world
            .checked_mul(LENGTH_RING)
            .ok_or_else(|| CommError::InvalidLayout("the length ring overflows".to_owned()))?;
        let words = self
            .world
            .checked_mul(self.rails)
            .and_then(|flags| flags.checked_add(2 + FLAG_SOURCES + lengths))
            .ok_or_else(|| CommError::InvalidLayout("the control words overflow".to_owned()))?;
        let control = std::mem::size_of::<u64>()
            .checked_mul(words)
            .ok_or_else(|| CommError::InvalidLayout("the control words overflow".to_owned()))?;
        self.set_stride
            .checked_mul(ROW_SETS)
            .and_then(|rows| rows.checked_add(control))
            .ok_or_else(|| {
                CommError::InvalidLayout(format!(
                    "a {}-rank slab of {} -byte rows overflows the address space",
                    self.world, self.row_cap
                ))
            })
    }

    /// Offset of row set parity. It does not depend on the round's row size, so two rounds of
    /// different sizes can never overlap each other's sets.
    ///
    /// # Errors
    /// Returns [CommError::InvalidLayout] for a parity outside 0..2.
    pub fn set_offset(&self, parity: usize) -> Result<usize, CommError> {
        if parity >= ROW_SETS {
            return Err(CommError::InvalidLayout(format!(
                "row set {parity} does not exist; a round uses seq & 1 of {ROW_SETS} sets"
            )));
        }
        Ok(parity * self.set_stride)
    }

    /// Offset of rank's row for a collective whose rows are row_bytes long, in row set parity.
    ///
    /// # Errors
    /// Returns [CommError::RankOutOfRange] outside the world and [CommError::InvalidLayout] for a
    /// row longer than the slab's cap or a parity outside 0..2.
    pub fn row_offset(
        &self,
        rank: usize,
        row_bytes: usize,
        parity: usize,
    ) -> Result<usize, CommError> {
        if rank >= self.world {
            return Err(CommError::RankOutOfRange {
                rank,
                world_size: self.world,
            });
        }
        if row_bytes == 0 || row_bytes > self.row_cap {
            return Err(CommError::InvalidLayout(format!(
                "a {row_bytes} -byte row does not fit the slab's {} -byte cap",
                self.row_cap
            )));
        }
        Ok(self.set_offset(parity)? + rank * row_bytes)
    }

    /// Offset of the control words, past both row sets: per rail one arrival flag per rank, then
    /// the doorbell, then the abort word. The exchange kernel indexes them from this one address.
    #[must_use]
    pub fn flags_offset(&self) -> usize {
        self.set_stride * ROW_SETS
    }

    /// Offset of the flag that says rank peer's part of the row has arrived over rail.
    ///
    /// # Errors
    /// Returns [CommError::RankOutOfRange] outside the world and [CommError::InvalidLayout] for a
    /// rail this layout does not have.
    pub fn flag_offset(&self, rail: usize, peer: usize) -> Result<usize, CommError> {
        if peer >= self.world {
            return Err(CommError::RankOutOfRange {
                rank: peer,
                world_size: self.world,
            });
        }
        if rail >= self.rails {
            return Err(CommError::InvalidLayout(format!(
                "rail {rail} does not exist; the slab is laid out for {}",
                self.rails
            )));
        }
        Ok(self.flags_offset() + (rail * self.world + peer) * WORD_BYTES)
    }

    /// The part of a row_bytes row that rail carries, as its offset inside the row and its length.
    /// The parts are equal up to a 64-byte boundary and the last rail takes the remainder, so they
    /// cover the row exactly once; a row too short to split leaves the earlier rails nothing to
    /// carry but their flag.
    #[must_use]
    pub fn segment(&self, rail: usize, row_bytes: usize) -> (usize, usize) {
        let share = (row_bytes / self.rails) & !(REGION_ALIGN - 1);
        let start = rail.min(self.rails - 1) * share;
        if rail + 1 >= self.rails {
            (start, row_bytes - start)
        } else {
            (start, share)
        }
    }

    /// Offset of the doorbell word the exchange kernel rings once its row is staged.
    #[must_use]
    pub fn doorbell_offset(&self) -> usize {
        self.flags_offset() + self.world * self.rails * WORD_BYTES
    }

    /// Offset of the abort word. The pump sets it when a round can no longer complete, and a
    /// kernel waiting for a peer leaves its wait when it reads non-zero.
    #[must_use]
    pub fn abort_offset(&self) -> usize {
        self.doorbell_offset() + WORD_BYTES
    }

    /// Offset of the word round seq publishes its arrival flag from.
    #[must_use]
    pub fn source_offset(&self, seq: u64) -> usize {
        self.abort_offset() + WORD_BYTES + (seq % FLAG_SOURCES as u64) as usize * WORD_BYTES
    }

    /// The slot holding the row length `peer` published for round seq, at either end of the wire.
    ///
    /// The parameter is the **publisher** on both ends, and that is the whole contract: a sender
    /// writes the word into that peer's slot on itself before it posts it there, and a receiver
    /// reads it out of the *sender's* slot of its own ring. Using the destination's index at the
    /// sender instead puts every sender in one slot and leaves the slots a receiver reads at zero.
    ///
    /// The arrival flag says *that* a peer's row arrived; it does not say how long that row was.
    /// The write offset and the fold's addressing are both derived from this rank's own row size,
    /// so two ranks that disagree about a round's row length write and read different bytes of the
    /// same set - in bounds, with no fault, and the fold silently sums the wrong rows. The length
    /// travels on the wire, one word per peer per round, keyed by sequence number rather than by
    /// peer so that a peer a round ahead does not overwrite the length this round is checking.
    ///
    /// # Errors
    /// Returns [CommError::RankOutOfRange] for a peer outside the world.
    pub fn length_slot(&self, publisher: usize, seq: u64) -> Result<usize, CommError> {
        if publisher >= self.world {
            return Err(CommError::RankOutOfRange {
                rank: publisher,
                world_size: self.world,
            });
        }
        Ok(self.lengths_offset()
            + (publisher * LENGTH_RING + (seq % LENGTH_RING as u64) as usize) * WORD_BYTES)
    }

    /// Offset of the first published row length.
    #[must_use]
    pub fn lengths_offset(&self) -> usize {
        self.abort_offset() + WORD_BYTES * (1 + FLAG_SOURCES)
    }

    /// Rounds of published row length kept per peer.
    #[must_use]
    pub fn length_ring(&self) -> usize {
        LENGTH_RING
    }

    /// Largest row this slab can carry.
    #[must_use]
    pub fn row_cap(&self) -> usize {
        self.row_cap
    }

    /// Ranks in the exchange.
    #[must_use]
    pub fn world(&self) -> usize {
        self.world
    }

    /// Rails a row is split over.
    #[must_use]
    pub fn rails(&self) -> usize {
        self.rails
    }
}

/// The doorbell deadline: `SPARK_COMM_DOORBELL_TIMEOUT_MS`, else a multiple of the control-plane
/// timeout with a floor under it.
fn doorbell_timeout(control: Duration) -> Duration {
    let from_env = std::env::var("SPARK_COMM_DOORBELL_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|ms| *ms > 0)
        .map(Duration::from_millis);
    from_env.unwrap_or_else(|| {
        (control * DOORBELL_TIMEOUT_FACTOR)
            .max(DOORBELL_TIMEOUT_FLOOR)
            .min(DOORBELL_TIMEOUT_CEILING)
    })
}

/// The round rank peer has delivered in full: the least of its flags over the rails, or zero for a
/// peer outside the world.
///
/// # Safety
/// host must be the base of a live slab that layout describes.
unsafe fn delivered(host: *const u8, layout: SlabLayout, peer: usize) -> u64 {
    (0..layout.rails())
        .map(|rail| match layout.flag_offset(rail, peer) {
            // SAFETY: the offset is inside the slab (the layout placed it) and the NIC writes the
            // word as a u64, the width this reads.
            Ok(offset) => unsafe { std::ptr::read_volatile(host.add(offset).cast::<u64>()) },
            Err(_) => 0,
        })
        .min()
        .unwrap_or(0)
}

/// What one peer published about itself, reduced to what a write needs.
#[derive(Debug, Clone, Copy)]
struct RemotePeer {
    /// Address of the peer's registered slab.
    addr: u64,
    /// Key authorising writes into it.
    rkey: u32,
}

/// What the launch thread and the pump share. Every field is an atomic or behind the one mutex, so
/// neither side ever waits for the other.
struct Shared {
    /// The last round handed to a kernel.
    submitted: AtomicU64,
    /// The last round whose writes the pump has posted.
    posted: AtomicU64,
    /// Row size of round n at n % ROUND_RING, written before `submitted` reaches n.
    row_bytes: Vec<AtomicU32>,
    /// Whether the pump is parked and needs waking by the next submit.
    parked: AtomicBool,
    /// Set by the handle's drop.
    stop: AtomicBool,
    /// Set with `failure`; the cheap half of [WriteExchange::check].
    failed: AtomicBool,
    /// The first failure. Every later submit and check reports it instead of running on state this
    /// backend can no longer vouch for: a round that gave up part way leaves writes outstanding and
    /// peers a round apart, and folding a later round into that is a wrong answer, not a dead run.
    failure: Mutex<Option<String>>,
    /// Rounds posted.
    rounds: AtomicU64,
    /// Row bytes posted, once per round (not once per peer: one row goes to every peer).
    bytes: AtomicU64,
    /// ns the pump spent posting.
    post_ns: AtomicU64,
    /// Rounds whose arrival the pump timed.
    arrivals: AtomicU64,
    /// ns from a round's posts to every peer's flag, over `arrivals`.
    arrival_ns: AtomicU64,
    /// The longest of those.
    arrival_max_ns: AtomicU64,
    /// ns from a round's posts to each peer's flag, over `arrivals`. This rank's own entry stays 0.
    peer_arrival_ns: Vec<AtomicU64>,
    /// Times the pump parked.
    parks: AtomicU64,
}

impl Shared {
    fn failure(&self) -> Option<String> {
        if !self.failed.load(Ordering::Acquire) {
            return None;
        }
        Some(match self.failure.lock() {
            Ok(reason) => reason.clone().unwrap_or_default(),
            Err(poisoned) => poisoned.into_inner().clone().unwrap_or_default(),
        })
    }

    fn fail(&self, reason: &str) {
        let mut slot = match self.failure.lock() {
            Ok(slot) => slot,
            Err(poisoned) => poisoned.into_inner(),
        };
        if slot.is_none() {
            *slot = Some(reason.to_owned());
        }
        self.failed.store(true, Ordering::Release);
    }
}

/// What the pump has done since the exchange opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExchangeStats {
    /// The rank whose pump this is.
    pub rank: usize,
    /// Rounds posted.
    pub rounds: u64,
    /// Row bytes posted, counted once per round: `bytes / rounds` is the mean row size, and the
    /// traffic is that times the peers it went to.
    pub bytes: u64,
    /// ns the pump spent posting those rounds' writes.
    pub post_ns: u64,
    /// Rounds whose arrival the pump timed (it misses one only when the next doorbell beats it).
    pub arrivals: u64,
    /// ns from a round's posts to every peer's flag, summed over `arrivals`.
    pub arrival_ns: u64,
    /// The longest single wait in `arrival_ns`.
    pub arrival_max_ns: u64,
    /// ns from a round's posts to each peer's flag, summed over `arrivals`, one entry a rank with
    /// this rank's own left at 0. A rank that is late to ring shows up as the largest entry on
    /// every other rank and as the smallest total on its own, which is how a slow group names the
    /// rank that is holding it.
    pub peer_arrival_ns: Vec<u64>,
    /// Times the pump went to sleep with nothing in flight.
    pub parks: u64,
}

/// The direct-RDMA exchange for one rank.
///
/// Open it once. Per collective the caller takes a sequence number from [Self::submit] and launches
/// the exchange kernel with it; nothing else is asked of the launch thread. The caller owns the
/// slab: it allocates the page-locked, mapped memory itself (the forward pass already has the CUDA
/// driver) and passes the host alias here.
pub struct WriteExchange {
    shared: Arc<Shared>,
    /// The pump, joined on drop so the registration is gone before the caller frees the slab.
    pump: Option<JoinHandle<()>>,
    /// The pump's thread, for the wake a submit owes a parked pump.
    waker: Thread,
    layout: SlabLayout,
    rank: usize,
    host: *mut u8,
    /// The registered slab's address as peers see it.
    addr: u64,
    timeout: Duration,
    /// The device each rail runs on, for the bring-up report.
    rail_names: Vec<String>,
}

// SAFETY: the handle holds atomics, a join handle and the slab's address. The slab is only read
// and written through volatile u64 accesses of words the layout places, which is what the NIC and
// the device do to the same words.
unsafe impl Send for WriteExchange {}

/// One rail's fabric objects.
struct Rail {
    // Drop order is load-bearing: the QPs go first, then the completion queue they post to, then
    // the registration they write through, then the protection domain, then the device.
    /// qps[d] connects to rank d; this rank's own slot is never connected or used.
    qps: Vec<Qp>,
    cq: Cq,
    mr: Mr,
    _pd: Pd,
    _context: Context,
    /// peers[d] is rank d's slab as this rail reaches it.
    peers: Vec<RemotePeer>,
    /// Writes posted on this rail and not yet reaped.
    pending: usize,
}

/// The half that owns the fabric objects and runs on the pump thread.
struct Pump {
    // Drop order is load-bearing: every rail's objects go before the library. Each of them holds
    // bare function pointers into libibverbs, so the dlclose must outlive all of them; that is why
    // _verbs is declared after the rails even though nothing reads it.
    rails: Vec<Rail>,
    /// Declared last of the fabric objects: the library must outlive every object it created.
    _verbs: Verbs,
    layout: SlabLayout,
    rank: usize,
    host: *mut u8,
    timeout: Duration,
    /// How long the doorbell may lag a submitted round before the pump calls it stuck. Separate
    /// from `timeout` because the two measure different things: `timeout` is the rendezvous and
    /// the arrival wait, and this is the device's queue depth.
    doorbell_timeout: Duration,
    shared: Arc<Shared>,
    /// The last round posted and when, so a failed write can say how long it had been out.
    last_post: Option<(u64, Instant)>,
}

// SAFETY: the fabric objects are created by the opening thread and then moved, once, to the pump
// thread, which is the only thread that ever touches them again; libibverbs objects are not tied to
// the thread that created them.
unsafe impl Send for Pump {}

/// An empty completion record for the poll buffer.
const fn empty_wc() -> IbvWc {
    IbvWc {
        wr_id: 0,
        status: 0,
        opcode: 0,
        vendor_err: 0,
        byte_len: 0,
        imm_data: 0,
        qp_num: 0,
        src_qp: 0,
        flags: 0,
        pkey_index: 0,
        slid: 0,
        sl: 0,
        dlid_path_bits: 0,
        reserved: [0; 2],
    }
}

impl WriteExchange {
    /// Opens the exchange, brings the QP mesh up and starts the pump.
    ///
    /// host is the page-locked slab whose size [SlabLayout::total_bytes] reports, and config
    /// supplies the ranks, the control-plane addresses and the deadlines. The slab must stay
    /// allocated until this value is dropped.
    ///
    /// # Errors
    /// Returns [CommError::FabricUnavailable] when no rail is usable, [CommError::Unsupported] when
    /// libibverbs cannot be loaded, [CommError::Verbs] when the library refuses a call,
    /// [CommError::Timeout] when the control plane misses its deadline, and [CommError::Protocol]
    /// when a peer publishes an inconsistent record.
    pub fn open(
        config: &RdmaConfig,
        host: *mut c_void,
        layout: SlabLayout,
    ) -> Result<Self, CommError> {
        config.validate()?;
        if layout.world() != config.world_size {
            return Err(CommError::InvalidLayout(format!(
                "the slab describes {} ranks but the configuration describes {}",
                layout.world(),
                config.world_size
            )));
        }
        if host.is_null() {
            return Err(CommError::InvalidLayout(
                "the exchange needs a registered host slab; a null address would register nothing"
                    .to_owned(),
            ));
        }
        // Three writes per peer per rail: the round's row length, that rail's part of the row, and
        // the arrival flag. Counted here rather than written down again in the post, because a
        // check that is one short is worse than no check: the subtraction below would wrap and the
        // queue would be overrun rather than drained.
        let replies = layout.world().saturating_sub(1);
        if WRITES_PER_PEER * replies > SEND_WR_DEPTH as usize {
            return Err(CommError::InvalidLayout(format!(
                "a round posts {WRITES_PER_PEER} writes per peer on each rail and a {} -rank world's would not fit a send queue of {SEND_WR_DEPTH}",
                layout.world()
            )));
        }
        if layout.rails() != config.rail_count {
            return Err(CommError::InvalidLayout(format!(
                "the slab is laid out for {} rail(s) but the configuration opens {}",
                layout.rails(),
                config.rail_count
            )));
        }
        let total = layout.total_bytes()?;
        // Zero the slab before it is published. cuMemHostAlloc does not promise zeroed memory, and
        // every wait in this module is "the word is at least my sequence number": one stale word in
        // the flags or the doorbell makes the first exchange believe a peer has already arrived and
        // fold whatever the rows contained. The smoke test allocates zeroed memory and would never
        // have caught it.
        // SAFETY: the caller owns `total` bytes at `host` (checked against the layout above) and
        // has not published the address to any peer yet.
        unsafe { std::ptr::write_bytes(host.cast::<u8>(), 0, total) };
        let selection = super::select_rails_from_live_fabric()?;
        let verbs = Verbs::load()?;
        let cap = IbvQpCap {
            max_send_wr: SEND_WR_DEPTH,
            max_recv_wr: 8,
            max_send_sge: 1,
            max_recv_sge: 1,
            max_inline_data: 0,
        };
        let mut rails: Vec<Rail> = Vec::with_capacity(layout.rails());
        let mut rail_names = Vec::with_capacity(layout.rails());
        let mut mtus = Vec::with_capacity(layout.rails());
        let mut owned: [Option<RailEndpoint>; MAX_RAILS] = [None, None];
        for (rail_index, owned_slot) in owned.iter_mut().enumerate().take(layout.rails()) {
            // The same refusals the transport backend makes: a port that is not ACTIVE Ethernet or
            // a GID of the wrong family builds a QP that connects and never delivers.
            let OpenedRail {
                name,
                context,
                active_mtu,
                gid,
                ..
            } = open_rail(&verbs, &selection, config, rail_index)?;
            let pd = context.alloc_pd()?;
            // Every rail registers the same slab: the row a rail's NIC reads and the row the other
            // rail's NIC reads are parts of one row the device wrote once.
            let mr = pd.reg_mr(
                host,
                total,
                access::LOCAL_WRITE | access::REMOTE_WRITE | access::REMOTE_READ,
            )?;
            registered_at(&mr, host as u64, rail_index, "exchange slab")?;
            // Every write is signaled so a failed one is named by its status.
            let cq = context.create_cq(256)?;
            // One QP per rank, including this one's own slot: the endpoint record publishes a QP
            // number for every destination and a zero there is refused by the decoder, and the
            // existing backend sizes its rails the same way. The self QP is never connected or used.
            let mut qps: Vec<Qp> = Vec::with_capacity(config.world_size);
            for _peer in 0..config.world_size {
                qps.push(pd.create_qp(&cq, &cq, cap)?);
            }
            *owned_slot = Some(RailEndpoint {
                gid,
                gid_index: config.gid_index,
                active_mtu: u16::try_from(active_mtu).map_err(|_| {
                    CommError::verbs(
                        "ibv_query_port",
                        format!(
                            "active_mtu {active_mtu} does not fit the two-byte field of the control-plane record"
                        ),
                    )
                })?,
                mr_addr: mr.addr(),
                rkey: mr.rkey(),
                qpns: qps.iter().map(Qp::qp_num).collect(),
            });
            rail_names.push(name);
            mtus.push(active_mtu);
            rails.push(Rail {
                qps,
                cq,
                mr,
                _pd: pd,
                _context: context,
                peers: Vec::new(),
                pending: 0,
            });
        }
        let rail_count = u8::try_from(layout.rails()).unwrap_or(u8::MAX);
        let own = RankEndpoints {
            rank: config.rank,
            frame_cap: u32::try_from(layout.row_cap()).unwrap_or(u32::MAX),
            slot_bytes: u64::try_from(total).unwrap_or(u64::MAX),
            rail_count,
            // What this table is for. A peer that publishes the other arm is refused below
            // rather than waited on.
            arm: super::qpinfo::ARM_EXCHANGE,
            rails: [
                owned[0]
                    .take()
                    .ok_or_else(|| CommError::InvalidLayout("no rail was opened".to_owned()))?,
                // A single-rail exchange publishes the zeroed placeholder the decoder expects.
                owned[1]
                    .take()
                    .unwrap_or_else(|| RankEndpoints::unused_rail(config.world_size)),
            ],
        };
        let control = ControlConfig {
            rank: config.rank,
            world: config.world_size,
            peers: config.peers.clone(),
            port: config.control_port,
            timeout: config.timeout,
        };
        let (table, star) = open_star(&control, own)?;
        if table.frame_cap() != u32::try_from(layout.row_cap()).unwrap_or(u32::MAX)
            || table.slot_bytes() != u64::try_from(total).unwrap_or(u64::MAX)
            || table.rail_count() != rail_count
        {
            return Err(CommError::Protocol(format!(
                "the cluster disagrees on the exchange slab: this rank has {} -byte rows in {total} bytes over {} rail(s), the table says {}, {} and {}",
                layout.row_cap(),
                layout.rails(),
                table.frame_cap(),
                table.slot_bytes(),
                table.rail_count()
            )));
        }
        // The geometry check above compares the two ranks' slabs. This one compares their
        // *collectives*: a rank that publishes the transport arm is in the NCCL group, and no
        // amount of waiting will make it ring an arrival flag. Refusing here is the difference
        // between a named error at bring-up and a 120-second stall that reads like a fabric fault.
        for rank in 0..config.world_size {
            let published = table.rank(rank)?.arm;
            if published != super::qpinfo::ARM_EXCHANGE {
                return Err(CommError::Protocol(format!(
                    "rank {rank} published arm {published}, not the direct-RDMA exchange ({}) (rank {} of {})",
                    super::qpinfo::ARM_EXCHANGE,
                    config.rank,
                    config.world_size
                )));
            }
        }
        for (rail_index, rail) in rails.iter_mut().enumerate() {
            let mut peers = Vec::with_capacity(config.world_size);
            for (dest, qp) in rail.qps.iter().enumerate() {
                if dest == config.rank {
                    peers.push(RemotePeer { addr: 0, rkey: 0 });
                    continue;
                }
                let endpoint = table
                    .rank(dest)?
                    .rail(rail_index)
                    .ok_or_else(|| {
                        CommError::Protocol(format!(
                            "rank {dest} published no exchange endpoint for rail {rail_index}"
                        ))
                    })?
                    .clone();
                let dest_qpn = *endpoint.qpns.get(config.rank).ok_or_else(|| {
                    CommError::Protocol(format!(
                        "rank {dest} published {} QP numbers on rail {rail_index}, too few for rank {}",
                        endpoint.qpns.len(),
                        config.rank
                    ))
                })?;
                if dest_qpn == 0 {
                    return Err(CommError::Protocol(format!(
                        "rank {dest} published a zero QP number for rank {} on rail {rail_index}",
                        config.rank
                    )));
                }
                let path_mtu = min_path_mtu(mtus[rail_index], i32::from(endpoint.active_mtu))?;
                let (mut init, init_mask) = init_attr(config.port);
                qp.modify(&mut init, init_mask)?;
                let (mut rtr, rtr_mask) = rtr_attr(
                    config.port,
                    dest_qpn,
                    endpoint.gid,
                    config.gid_index,
                    path_mtu,
                );
                qp.modify(&mut rtr, rtr_mask)?;
                let (mut rts, rts_mask) = rts_attr();
                qp.modify(&mut rts, rts_mask)?;
                peers.push(RemotePeer {
                    addr: endpoint.mr_addr,
                    rkey: endpoint.rkey,
                });
            }
            rail.peers = peers;
        }
        let shared = Arc::new(Shared {
            submitted: AtomicU64::new(0),
            posted: AtomicU64::new(0),
            row_bytes: (0..ROUND_RING).map(|_| AtomicU32::new(0)).collect(),
            parked: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            failure: Mutex::new(None),
            rounds: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            post_ns: AtomicU64::new(0),
            arrivals: AtomicU64::new(0),
            arrival_ns: AtomicU64::new(0),
            arrival_max_ns: AtomicU64::new(0),
            peer_arrival_ns: (0..layout.world()).map(|_| AtomicU64::new(0)).collect(),
            parks: AtomicU64::new(0),
        });
        let addr = host as u64;
        let doorbell_timeout = doorbell_timeout(config.timeout);
        let pump = Pump {
            rails,
            _verbs: verbs,
            layout,
            rank: config.rank,
            host: host.cast::<u8>(),
            timeout: config.timeout,
            doorbell_timeout,
            shared: Arc::clone(&shared),
            last_post: None,
        };
        let pump = std::thread::Builder::new()
            .name(format!("rdma-exchange-{}", config.rank))
            .spawn(move || pump.run())
            .map_err(|error| {
                CommError::Unsupported(format!("the exchange's pump thread did not start: {error}"))
            })?;
        // Every rank connects its queue pairs at its own pace, and the first round is posted the
        // moment this returns, so without a meeting here a rank's first writes can reach a peer
        // that is still connecting. Measured, that first round took up to 17 ms where every later
        // one takes microseconds. Two launches also lost their first round to a remote invalid
        // request before this was here; that this was the cause is likely and not proven
        // (docs/rdma-exchange.md). No rank returns from here until every rank's queue pairs are
        // connected.
        //
        // This is deliberately the *last* fallible step of the open. Everything a rank needs to
        // take part - the slab and its registration, every queue pair in RTS, the pump thread -
        // is already live, and the caller loads the exchange's kernels before calling here. So a
        // rank that gets past this point cannot fail to take part, which is what makes the
        // group-wide commitment mean anything: the alternative is a rank that stands down to the
        // NCCL group after its peers have committed, which no timeout can tell apart from a
        // broken fabric.
        if let Err(error) = star.ready() {
            // The pump is already running; a failed meeting must not leave it behind.
            shared.stop.store(true, Ordering::SeqCst);
            pump.thread().unpark();
            let _ = pump.join();
            return Err(error);
        }
        Ok(Self {
            shared,
            waker: pump.thread().clone(),
            pump: Some(pump),
            layout,
            rank: config.rank,
            host: host.cast::<u8>(),
            addr,
            timeout: config.timeout,
            rail_names,
        })
    }

    /// The devices the exchange runs over, one per rail.
    #[must_use]
    pub fn rail_names(&self) -> &[String] {
        &self.rail_names
    }

    /// The slab layout this exchange was opened with.
    #[must_use]
    pub fn layout(&self) -> SlabLayout {
        self.layout
    }

    /// The rank this exchange serves.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The sequence number of the last submitted round; zero before the first.
    #[must_use]
    pub fn seq(&self) -> u64 {
        self.shared.submitted.load(Ordering::Acquire)
    }

    /// The registered slab's address, which peers write to.
    #[must_use]
    pub fn address(&self) -> u64 {
        self.addr
    }

    /// Starts one exchange of row_bytes bytes and returns its sequence number.
    ///
    /// The caller then launches the exchange kernel with that number (or, staging on the host,
    /// writes its row and calls [Self::ring_doorbell]). This never waits for the device or the
    /// fabric: the pump posts the writes when the doorbell rings, and the kernel waits for the
    /// peers. Rounds must be launched in the order they were submitted, which taking `&mut self`
    /// and one launch stream gives.
    ///
    /// # Errors
    /// Returns [CommError::InvalidLayout] for a row that does not fit, and [CommError::Protocol]
    /// carrying the first failure once any round has failed.
    pub fn submit(&mut self, row_bytes: usize) -> Result<u64, CommError> {
        self.check()?;
        // Refused here, by the thread that can report it, rather than later by the pump.
        self.layout.row_offset(self.rank, row_bytes, 0)?;
        let row_len = u32::try_from(row_bytes).map_err(|_| {
            CommError::InvalidLayout(format!(
                "a {row_bytes} -byte exchange row exceeds what one work request carries"
            ))
        })?;
        let seq = self.shared.submitted.load(Ordering::Relaxed) + 1;
        if seq - self.shared.posted.load(Ordering::Acquire) > ROUND_RING as u64 {
            // The launch thread is a whole ring ahead of the device. Nothing in the step does
            // that, so this is a wait for a stuck pump with a deadline, not a hot path.
            let deadline = Instant::now() + self.timeout;
            while seq - self.shared.posted.load(Ordering::Acquire) > ROUND_RING as u64 {
                self.check()?;
                if Instant::now() > deadline {
                    return Err(CommError::timeout(
                        self.timeout,
                        format!(
                            "rank {} has {ROUND_RING} exchange rounds queued and the pump has posted none of them",
                            self.rank
                        ),
                    ));
                }
                std::thread::yield_now();
            }
        }
        self.shared.row_bytes[(seq % ROUND_RING as u64) as usize].store(row_len, Ordering::Relaxed);
        // SeqCst on this store and the load below, and on the pump's mirror image of them, is what
        // makes the wake impossible to lose: either the pump's re-check sees this round, or this
        // load sees the pump parked.
        self.shared.submitted.store(seq, Ordering::SeqCst);
        if self.shared.parked.load(Ordering::SeqCst) {
            self.waker.unpark();
        }
        Ok(seq)
    }

    /// Marks the exchange failed from outside the pump.
    ///
    /// The launch thread owes this when it has taken a sequence number and cannot launch the
    /// kernel that would ring its doorbell: the round is already reserved, so the pump would post
    /// a skipped round from whatever bytes happen to be in the row. Setting the abort word wakes
    /// any kernel waiting on it, and every later submit and check reports the reason.
    pub fn fail(&self, reason: &str) {
        self.shared.fail(reason);
        let offset = self.layout.abort_offset();
        // SAFETY: the offset is inside the registered slab and the kernel reads a u64 there.
        unsafe { std::ptr::write_volatile(self.host.add(offset).cast::<u64>(), 1) };
    }

    /// The abort word: zero, or a tag naming why the device or the pump gave a round up.
    fn abort_word(&self) -> u64 {
        // SAFETY: the offset is inside the registered slab; the kernel, the pump and this method
        // all read and write a u64 there.
        unsafe { std::ptr::read_volatile(self.host.add(self.layout.abort_offset()).cast::<u64>()) }
    }

    /// Reports the first failure, if any round has failed.
    ///
    /// A kernel that was waiting for a peer when the pump gave up leaves its wait without folding,
    /// so whatever the step computed after that is not a result. The caller must call this once
    /// the device has drained and before it uses anything the step produced.
    ///
    /// # Errors
    /// Returns [CommError::Protocol] carrying the first failure.
    pub fn check(&self) -> Result<(), CommError> {
        if let Some(reason) = self.shared.failure() {
            return Err(CommError::Protocol(format!(
                "rank {}: the direct-RDMA exchange failed and will not be reused ({reason})",
                self.rank
            )));
        }
        // The device can end a round on its own - a peer published a length this round did not send -
        // and it says so in the abort word. Reading that word here, and not only in the pump, is
        // what makes the caller's "the device has drained, so the collective is finished" true. The
        // pump looks at the word between rounds, and while the launch thread is several rounds
        // ahead it can go a long time without doing so; a fold that ran on a refused round would be
        // reported by nobody until then.
        let abort = self.abort_word();
        if abort & 0xffff == ABORT_TAG_LENGTH {
            return Err(CommError::Protocol(format!(
                "rank {}: a round was refused by the device: rank {} published a row length that is not the one the round sent, so the fold behind it summed the wrong bytes (both ranks must reach every collective in the same order with the same row sizes)",
                self.rank,
                (abort >> ABORT_PEER_SHIFT) & 0xff
            )));
        }
        if abort != 0 {
            return Err(CommError::Protocol(format!(
                "rank {}: a round was refused by the device or given up by the pump (abort word {abort:#x}) and will not be reused",
                self.rank
            )));
        }
        Ok(())
    }

    /// The address inside the slab where rank's row starts, in the set round parity uses.
    ///
    /// # Errors
    /// As [SlabLayout::row_offset].
    pub fn row_address(
        &self,
        rank: usize,
        row_bytes: usize,
        parity: usize,
    ) -> Result<u64, CommError> {
        Ok(self.addr + self.layout.row_offset(rank, row_bytes, parity)? as u64)
    }

    /// The row set the round with this sequence number uses.
    #[must_use]
    pub fn parity(seq: u64) -> usize {
        (seq & 1) as usize
    }

    /// The round peer has delivered in full (the least of its flags over the rails), or zero
    /// before the first exchange.
    #[must_use]
    pub fn arrival(&self, peer: usize) -> u64 {
        // SAFETY: host is the registered slab the caller allocated and keeps alive, and layout is
        // the one it was opened with.
        unsafe { delivered(self.host, self.layout, peer) }
    }

    /// Publishes seq in the doorbell word, for a caller that stages the row on the host instead of
    /// on the device (the smoke test). The exchange kernel does this itself.
    pub fn ring_doorbell(&mut self, seq: u64) {
        // The length goes out before the doorbell for the same reason the pump posts it before the
        // flag: a peer that can see the round has already been sent the length it is checked
        // against. It is the length `submit` was given and not the caller's own idea of it: the
        // pump sends that many bytes, so publishing anything else would make the receiver's check
        // agree to a size the wire is not carrying - which is the one thing the length word exists
        // to prevent.
        let row_bytes =
            self.shared.row_bytes[(seq % ROUND_RING as u64) as usize].load(Ordering::Relaxed);
        let length = self.layout.length_slot(self.rank, seq).ok();
        if let Some(offset) = length {
            // SAFETY: inside the registered slab; the pump forwards this word to every peer.
            unsafe {
                std::ptr::write_volatile(
                    self.host.add(offset).cast::<u64>(),
                    length_word(seq, row_bytes as usize),
                )
            };
        }
        let offset = self.layout.doorbell_offset();
        // SAFETY: offset is inside the registered slab, and a u64 store is what the kernel does.
        unsafe { std::ptr::write_volatile(self.host.add(offset).cast::<u64>(), seq) };
    }

    /// The word peer published for round seq: its sequence in the high bits and its row length in
    /// the low `LENGTH_SEQ_SHIFT`, or 0 when it has published none.
    #[must_use]
    pub fn published_length(&self, peer: usize, seq: u64) -> u64 {
        let Ok(offset) = self.layout.length_slot(peer, seq) else {
            return 0;
        };
        // SAFETY: inside the registered slab; a peer's NIC wrote it, a u64 is what it writes.
        unsafe { std::ptr::read_volatile(self.host.add(offset).cast::<u64>()) }
    }

    /// Waits on the host until every peer has published at least seq, for a caller with no kernel
    /// to do the waiting (the smoke test). The comparison is >= and not ==: a rank that gets
    /// lapped must not wait for a round that has already been overwritten.
    ///
    /// # Errors
    /// Returns the first failure if the pump reported one, and [CommError::Timeout] when a flag
    /// misses the deadline.
    pub fn wait_arrival(&self, seq: u64, row_bytes: usize) -> Result<(), CommError> {
        let deadline = Instant::now() + self.timeout;
        loop {
            for _ in 0..DOORBELL_POLLS {
                if (0..self.layout.world())
                    .all(|peer| peer == self.rank || self.arrival(peer) >= seq)
                {
                    // The flags say the rows arrived; the length says they are the rows this round
                    // is about to fold. Same check the exchange kernel makes, for a caller with no
                    // kernel to make it.
                    for peer in 0..self.layout.world() {
                        if peer == self.rank {
                            continue;
                        }
                        let word = self.published_length(peer, seq);
                        let published_seq = word >> LENGTH_SEQ_SHIFT;
                        let published = word & LENGTH_BYTES_MASK;
                        if published_seq != seq {
                            return Err(CommError::Protocol(format!(
                                "rank {}: this rank's round {seq} found rank {peer} publishing round {published_seq} in that slot: the two ranks are not running the same sequence of collectives (a rank that inserts or drops one shifts every later round)",
                                self.rank
                            )));
                        }
                        if published != row_bytes as u64 {
                            return Err(CommError::Protocol(format!(
                                "rank {}: round {seq} sent a {row_bytes} -byte row but rank {peer} published {published} bytes: the two ranks agree on neither the write offset nor the fold's addressing",
                                self.rank
                            )));
                        }
                    }
                    return Ok(());
                }
                std::hint::spin_loop();
            }
            self.check()?;
            if Instant::now() > deadline {
                let flags: Vec<(usize, u64)> = (0..self.layout.world())
                    .filter(|peer| *peer != self.rank)
                    .map(|peer| (peer, self.arrival(peer)))
                    .collect();
                return Err(CommError::timeout(
                    self.timeout,
                    format!(
                        "rank {} waited for every peer to publish {seq}; arrival flags are {flags:?}",
                        self.rank
                    ),
                ));
            }
        }
    }

    /// What the pump has done so far.
    #[must_use]
    pub fn stats(&self) -> ExchangeStats {
        let shared = &self.shared;
        ExchangeStats {
            rank: self.rank(),
            rounds: shared.rounds.load(Ordering::Relaxed),
            bytes: shared.bytes.load(Ordering::Relaxed),
            post_ns: shared.post_ns.load(Ordering::Relaxed),
            arrivals: shared.arrivals.load(Ordering::Relaxed),
            arrival_ns: shared.arrival_ns.load(Ordering::Relaxed),
            arrival_max_ns: shared.arrival_max_ns.load(Ordering::Relaxed),
            peer_arrival_ns: shared
                .peer_arrival_ns
                .iter()
                .map(|ns| ns.load(Ordering::Relaxed))
                .collect(),
            parks: shared.parks.load(Ordering::Relaxed),
        }
    }
}

impl Drop for WriteExchange {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        self.waker.unpark();
        if let Some(pump) = self.pump.take() {
            // The join is what orders the deregistration before the caller frees the slab.
            let _ = pump.join();
        }
    }
}

impl Pump {
    /// The pump thread's body.
    fn run(mut self) {
        if let Err(error) = self.serve() {
            self.shared.fail(&error.to_string());
            // A kernel waiting for a peer that will now never publish must not hold the stream
            // forever: it leaves its wait when it reads this word, and the caller's check after
            // the device drains is what turns that into an error instead of a wrong row.
            let offset = self.layout.abort_offset();
            // SAFETY: offset is inside the registered slab; the kernel reads it as a u64.
            unsafe { std::ptr::write_volatile(self.host.add(offset).cast::<u64>(), 1) };
        }
        // The registration stays until the handle is dropped: peers may still be writing into it,
        // and the device still reads it.
        while !self.shared.stop.load(Ordering::Acquire) {
            std::thread::park_timeout(Duration::from_millis(100));
        }
        // Every path out of `serve` - the clean stop and every failure - comes through here, and
        // this is where the sends this rank posted have to be reaped before the QPs and the MR go.
        // A failure is the *likeliest* time for sends to be outstanding, so draining only on the
        // clean path drains exactly when it is least needed.
        match self.reap(Some(0)) {
            Ok(()) => {}
            Err(error) => {
                // The registration is about to be destroyed with writes still outstanding. Nothing
                // left can drain them, so say so rather than let the caller assume the slab is
                // quiescent.
                self.shared.fail(&format!(
                    "rank {}: the exchange's sends did not drain before teardown, so its slab is not safe to free until this process exits ({error})",
                    self.rank
                ));
            }
        }
    }

    /// Posts each round when its doorbell rings, reaps between rounds, and watches the last
    /// round's arrival so that a peer that never publishes ends the run with a reason.
    fn serve(&mut self) -> Result<(), CommError> {
        let mut next = 1u64;
        let mut idle_since = Instant::now();
        // The round whose arrival has not been seen, and when its writes went out. Only the last
        // posted round can be in that state: the doorbell of n + 1 rings after the fold of n.
        let mut outstanding: Option<(u64, Instant)> = None;
        // What each peer's flag took in that round, 0 until it has been seen.
        let mut peer_ns = vec![0u64; self.layout.world()];
        // When the launch thread was first seen to be ahead of the doorbell.
        let mut wanted_since: Option<Instant> = None;
        // Consecutive doorbell expiries. One is a queue, three is a stuck device.
        let mut doorbell_trips = 0u32;
        loop {
            let mut rang = false;
            for _ in 0..DOORBELL_POLLS {
                if self.doorbell() >= next {
                    rang = true;
                    break;
                }
                std::hint::spin_loop();
            }
            if rang {
                if next > self.shared.submitted.load(Ordering::Acquire) {
                    return Err(CommError::Protocol(format!(
                        "rank {}: the doorbell reads {} but round {next} was never submitted",
                        self.rank,
                        self.doorbell()
                    )));
                }
                let started = Instant::now();
                let row_bytes = self.post_round(next)?;
                let posted = Instant::now();
                self.last_post = Some((next, posted));
                let shared = &self.shared;
                shared.posted.store(next, Ordering::Release);
                shared.rounds.fetch_add(1, Ordering::Relaxed);
                shared.bytes.fetch_add(row_bytes as u64, Ordering::Relaxed);
                shared.post_ns.fetch_add(
                    posted.duration_since(started).as_nanos() as u64,
                    Ordering::Relaxed,
                );
                outstanding = Some((next, posted));
                peer_ns.fill(0);
                wanted_since = None;
                doorbell_trips = 0;
                idle_since = posted;
                next += 1;
                continue;
            }
            if self.shared.stop.load(Ordering::Acquire) {
                return Ok(());
            }
            self.reap(None)?;
            // The device sets the abort word when it refuses a round: a peer's row length is not
            // the one this round sent, or the host gave the round up. The first case is named here,
            // because it is a disagreement between ranks and not a fabric fault.
            let abort = self.abort_word();
            if abort & 0xffff == ABORT_TAG_LENGTH {
                return Err(CommError::Protocol(format!(
                    "rank {}: round {} did not complete: rank {} published a row length that is not the one this round sent, so the fold would have summed the wrong bytes (both ranks must reach every collective in the same order with the same row sizes)",
                    self.rank,
                    next.saturating_sub(1),
                    (abort >> ABORT_PEER_SHIFT) & 0xff
                )));
            }
            let now = Instant::now();
            if let Some((seq, posted)) = outstanding {
                let waited = now.duration_since(posted);
                let ns = (waited.as_nanos() as u64).max(1);
                // The comparison is >= and not ==: a peer already a round ahead has published
                // this one.
                let mut missing = 0usize;
                for (peer, seen) in peer_ns.iter_mut().enumerate() {
                    if peer == self.rank || *seen != 0 {
                        continue;
                    }
                    if self.arrival(peer) >= seq {
                        *seen = ns;
                    } else {
                        missing += 1;
                    }
                }
                if missing == 0 {
                    let shared = &self.shared;
                    shared.arrivals.fetch_add(1, Ordering::Relaxed);
                    shared.arrival_ns.fetch_add(ns, Ordering::Relaxed);
                    shared.arrival_max_ns.fetch_max(ns, Ordering::Relaxed);
                    for (total, seen) in shared.peer_arrival_ns.iter().zip(&peer_ns) {
                        total.fetch_add(*seen, Ordering::Relaxed);
                    }
                    outstanding = None;
                } else if waited > self.timeout {
                    return Err(self.arrival_timeout(seq));
                }
            }
            if self.shared.submitted.load(Ordering::Acquire) >= next {
                // Launched, and the device has not reached it yet. The interval below measures
                // every kernel the launch thread queued ahead of the exchange kernel, so it is the
                // device's queue depth, not the fabric: it is held to its own deadline and a
                // single expiry is retried rather than taken as a wire fault.
                let since = *wanted_since.get_or_insert(now);
                if now.duration_since(since) > self.doorbell_timeout {
                    doorbell_trips += 1;
                    if doorbell_trips >= DOORBELL_TRIP_LIMIT {
                        return Err(CommError::timeout(
                            self.doorbell_timeout,
                            format!(
                                "rank {} waited {:?} for the exchange kernel's doorbell to reach {next} (it is {}), {DOORBELL_TRIP_LIMIT} times over: the device is not reaching the exchange kernel, which is a stalled launch and not an arrival the peers owe (SPARK_COMM_DOORBELL_TIMEOUT_MS sets this deadline)",
                                self.rank,
                                self.doorbell_timeout.saturating_mul(DOORBELL_TRIP_LIMIT),
                                self.doorbell()
                            ),
                        ));
                    }
                    wanted_since = None;
                }
            } else if outstanding.is_none()
                && self.rails.iter().all(|rail| rail.pending == 0)
                && now.duration_since(idle_since) > IDLE_SPIN
            {
                self.shared.parked.store(true, Ordering::SeqCst);
                if self.shared.submitted.load(Ordering::SeqCst) < next
                    && !self.shared.stop.load(Ordering::SeqCst)
                {
                    self.shared.parks.fetch_add(1, Ordering::Relaxed);
                    std::thread::park();
                }
                self.shared.parked.store(false, Ordering::SeqCst);
                idle_since = Instant::now();
            }
        }
    }

    /// The abort word: zero, or a tag naming why the device gave the round up.
    fn abort_word(&self) -> u64 {
        // SAFETY: the offset is inside the registered slab and both the kernel and the pump write
        // a u64 there.
        unsafe { std::ptr::read_volatile(self.host.add(self.layout.abort_offset()).cast::<u64>()) }
    }

    /// The doorbell word.
    fn doorbell(&self) -> u64 {
        // SAFETY: the offset is inside the registered slab and the kernel stores a u64 there.
        unsafe {
            std::ptr::read_volatile(self.host.add(self.layout.doorbell_offset()).cast::<u64>())
        }
    }

    /// Peers that have not yet published seq. The comparison is >= and not ==: a peer already a
    /// round ahead has published this one.
    fn missing(&self, seq: u64) -> usize {
        (0..self.layout.world())
            .filter(|peer| *peer != self.rank && self.arrival(*peer) < seq)
            .count()
    }

    /// The round peer has delivered in full on every rail.
    fn arrival(&self, peer: usize) -> u64 {
        // SAFETY: host is the registered slab, which outlives the pump, and layout describes it.
        unsafe { delivered(self.host, self.layout, peer) }
    }

    /// Peer's flag on each rail, for a report.
    fn flags(&self, peer: usize) -> Vec<u64> {
        (0..self.layout.rails())
            .map(|rail| match self.layout.flag_offset(rail, peer) {
                // SAFETY: as [delivered].
                Ok(offset) => unsafe {
                    std::ptr::read_volatile(self.host.add(offset).cast::<u64>())
                },
                Err(_) => 0,
            })
            .collect()
    }

    /// The error for a round whose peers never all published it.
    fn arrival_timeout(&mut self, seq: u64) -> CommError {
        let flags: Vec<(usize, Vec<u64>)> = (0..self.layout.world())
            .filter(|peer| *peer != self.rank)
            .map(|peer| (peer, self.flags(peer)))
            .collect();
        // A write that failed on the wire leaves its status in the queue, and the status is the
        // only thing that says *why* nothing arrived: a retry exhaustion is a path problem, a
        // remote access error is a key or offset problem.
        let mut statuses: Vec<(usize, u32, i32)> = Vec::new();
        for (rail_index, rail) in self.rails.iter_mut().enumerate() {
            let mut completions = [empty_wc(); 16];
            let polled = rail.cq.poll(&mut completions).unwrap_or(0);
            // These completions are gone from the queue; the count has to go with them. Leaving it
            // high makes every later `reap(Some(0))` wait out its deadline for completions that
            // can never arrive, which is a two-minute stall on the path that just failed.
            rail.pending = rail.pending.saturating_sub(polled);
            statuses.extend(
                completions
                    .iter()
                    .take(polled)
                    .map(|completion| (rail_index, completion.qp_num, completion.status)),
            );
        }
        CommError::timeout(
            self.timeout,
            format!(
                "rank {} waited for {} peer(s) to publish {seq}; arrival flags per rail are {flags:?}, completion statuses (rail, QP, status) are {statuses:?}",
                self.rank,
                self.missing(seq)
            ),
        )
    }

    /// Posts round seq: on every rail and for every peer that rail's part of the row and then its
    /// flag, chained so one call hands the NIC both. Returns the row's size.
    fn post_round(&mut self, seq: u64) -> Result<usize, CommError> {
        let row_bytes = self.shared.row_bytes[(seq % ROUND_RING as u64) as usize]
            .load(Ordering::Relaxed) as usize;
        let layout = self.layout;
        let rank = self.rank;
        let parity = WriteExchange::parity(seq);
        // The row sits at the same offset in every rank's slab, so the local source offset is also
        // the remote destination offset.
        let row_offset = layout.row_offset(rank, row_bytes, parity)?;
        let source_offset = layout.source_offset(seq);
        let length_offset = layout.length_slot(rank, seq)?;
        // The length word and the flag are one write each, the row part another.
        let writes = WRITES_PER_PEER * (layout.world() - 1);
        if self
            .rails
            .iter()
            .any(|rail| rail.pending + writes > SEND_WR_DEPTH as usize)
        {
            self.reap(Some(SEND_WR_DEPTH as usize - writes))?;
        }
        // The flag is published from inside the registered slab, not from a stack buffer: an SGE
        // whose address the MR does not cover fails locally with LOC_PROT_ERR and takes the QP to
        // error with it, which reads downstream as every peer's arrival flag staying at zero.
        // SAFETY: both offsets are inside the slab (SlabLayout placed them).
        let (row, source, length_at) = unsafe {
            let source = self.host.add(source_offset);
            std::ptr::write_volatile(source.cast::<u64>(), seq);
            let length_at = self.host.add(length_offset);
            std::ptr::write_volatile(length_at.cast::<u64>(), length_word(seq, row_bytes));
            (self.host.add(row_offset), source, length_at)
        };
        for (rail_index, rail) in self.rails.iter_mut().enumerate() {
            let (start, length) = layout.segment(rail_index, row_bytes);
            let flag_offset = layout.flag_offset(rail_index, rank)?;
            let lkey = rail.mr.lkey();
            for peer in 0..layout.world() {
                if peer == rank {
                    continue;
                }
                let remote = rail.peers[peer];
                let mut flag_sge = IbvSge {
                    addr: source as u64,
                    length: WORD_BYTES as u32,
                    lkey,
                };
                let mut flag = write_request(
                    write_tag(seq, peer, true),
                    &mut flag_sge,
                    remote,
                    flag_offset,
                );
                // The round's row length reaches the peer ahead of anything that depends on it:
                // the peer's wait compares it against its own before the fold reads a byte.
                let mut length_sge = IbvSge {
                    addr: length_at as u64,
                    length: WORD_BYTES as u32,
                    lkey,
                };
                // The length lands in this rank's slot of the peer's ring, exactly as the flag
                // lands in this rank's flag slot: the receiver looks the length up by *sender*,
                // because the number it has to compare is the one the publisher chose. Writing it
                // to the destination's index instead puts every sender in slot `peer`, so the
                // receiver, which reads its own senders' slots, finds only zeroes.
                let mut length_wr = write_request(
                    write_tag(seq, peer, false),
                    &mut length_sge,
                    remote,
                    length_offset,
                );
                if length == 0 {
                    // A row too short to split: this rail carries the length and its flag, and
                    // the peer still waits for the flag.
                    length_wr.next = &mut flag;
                    rail.qps[peer].post_send(&mut length_wr)?;
                    rail.pending += 2;
                    continue;
                }
                let mut part_sge = IbvSge {
                    addr: row as u64 + start as u64,
                    // A row fits a u32 (submit refused one that does not), so its part does.
                    length: length as u32,
                    lkey,
                };
                let mut part = write_request(
                    write_tag(seq, peer, false),
                    &mut part_sge,
                    remote,
                    row_offset + start,
                );
                // The length, then the part, then the flag, all on the same QP: a peer that can
                // see the flag has already been sent the length and every byte in front of it.
                length_wr.next = &mut part;
                part.next = &mut flag;
                rail.qps[peer].post_send(&mut length_wr)?;
                rail.pending += 3;
            }
        }
        Ok(row_bytes)
    }

    /// Reaps completions. With `until` unset it polls every rail once and returns, which is what
    /// the pump does between rounds, so the send queues are normally empty when a round is posted
    /// and no round waits for the one before it. With `until` set it stays until no rail has more
    /// than that many writes outstanding.
    fn reap(&mut self, until: Option<usize>) -> Result<(), CommError> {
        if self.rails.iter().all(|rail| rail.pending == 0) {
            return Ok(());
        }
        let deadline = Instant::now() + self.timeout;
        loop {
            let mut polled_any = false;
            for (rail_index, rail) in self.rails.iter_mut().enumerate() {
                if rail.pending == 0 {
                    continue;
                }
                let mut completions = [empty_wc(); 16];
                let polled = rail.cq.poll(&mut completions)?;
                // Every completion this poll took is off the queue, including the ones that
                // failed: the outstanding count has to go with them. Leaving it high on the error
                // path makes the teardown drain wait out its deadline for writes that will never
                // arrive, and report a second, less useful failure after the first one.
                rail.pending = rail.pending.saturating_sub(polled);
                for completion in completions.iter().take(polled) {
                    if completion.status != wc::SUCCESS {
                        let (seq, peer, flag) = write_tag_parts(completion.wr_id);
                        let out = match self.last_post {
                            Some((posted, at)) if posted == seq => {
                                format!(
                                    "{:.1} us after it was posted",
                                    at.elapsed().as_secs_f64() * 1e6
                                )
                            }
                            Some((posted, _)) => format!("with round {posted} already posted"),
                            None => "with nothing posted".to_owned(),
                        };
                        return Err(CommError::verbs(
                            "ibv_poll_cq",
                            format!(
                                "rank {}: round {seq}'s {} to rank {peer} on rail {rail_index} failed {out}: {} (vendor {:#x}, QP {})",
                                self.rank,
                                // The length word and the row part share one tag, so a failed
                                // write here is one of the two and the message must not pick.
                                if flag {
                                    "arrival flag"
                                } else {
                                    "row part or published length"
                                },
                                wc::status_name(completion.status),
                                completion.vendor_err,
                                completion.qp_num
                            ),
                        ));
                    }
                }
                polled_any |= polled != 0;
            }
            match until {
                None => return Ok(()),
                Some(allowed) if self.rails.iter().all(|rail| rail.pending <= allowed) => {
                    return Ok(());
                }
                Some(_) => {}
            }
            if !polled_any && Instant::now() > deadline {
                let pending: Vec<usize> = self.rails.iter().map(|rail| rail.pending).collect();
                return Err(CommError::timeout(
                    self.timeout,
                    format!(
                        "rank {} waited for outstanding exchange writes to complete ({pending:?} per rail)",
                        self.rank
                    ),
                ));
            }
        }
    }
}

/// Peers a write's tag can name; a world this size is far past anything the exchange is opened for.
const TAG_PEERS: u64 = 1 << 15;

/// The wr_id of a write: its round, the rank it goes to, and whether it is the arrival flag or the
/// row part in front of it. A completion carries nothing else that says which write failed.
fn write_tag(seq: u64, peer: usize, flag: bool) -> u64 {
    (seq << 16) | ((peer as u64 % TAG_PEERS) << 1) | u64::from(flag)
}

/// What [write_tag] packed: the round, the peer, and whether the write was the arrival flag.
fn write_tag_parts(tag: u64) -> (u64, usize, bool) {
    (tag >> 16, ((tag >> 1) % TAG_PEERS) as usize, tag & 1 == 1)
}

/// One signaled RDMA write of sge to offset in remote's slab, tagged for the completion.
fn write_request(tag: u64, sge: &mut IbvSge, remote: RemotePeer, offset: usize) -> IbvSendWr {
    IbvSendWr {
        wr_id: tag,
        next: std::ptr::null_mut(),
        sg_list: sge,
        num_sge: 1,
        opcode: wr_opcode::RDMA_WRITE,
        send_flags: send_flags::SIGNALED,
        imm_data: 0,
        wr: IbvWrUnion {
            rdma: IbvRdmaWr {
                remote_addr: remote.addr + offset as u64,
                rkey: remote.rkey,
            },
        },
        qp_type: IbvQpTypeUnion { raw: [0u8; 4] },
        padding: [0u8; 4],
        tail: IbvSendWrTail { raw: [0u8; 48] },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layout_keeps_the_rows_compact_and_the_flags_out_of_their_way() {
        let layout = SlabLayout::new(4, 1024, 1).expect("layout");
        // Two row sets of four rows, then the flags, the doorbell, the abort word and the sources.
        assert_eq!(layout.flags_offset(), 8192);
        assert_eq!(layout.flag_offset(0, 2).expect("flag"), 8192 + 16);
        assert!(layout.flag_offset(1, 2).is_err());
        assert_eq!(layout.doorbell_offset(), 8192 + 32);
        assert_eq!(layout.abort_offset(), 8192 + 40);
        assert_eq!(layout.source_offset(0), 8192 + 48);
        assert_eq!(layout.source_offset(65), 8192 + 48 + 8);
        // The control words: the source ring, then a length ring per peer.
        assert_eq!(
            layout.total_bytes().expect("total"),
            8192 + 48 + FLAG_SOURCES * 8 + 4 * LENGTH_RING * 8
        );
        assert_eq!(layout.lengths_offset(), 8192 + 48 + FLAG_SOURCES * 8);
        // In one set a 100-byte row lands at peer * 100, which is the addressing the fold uses;
        // the second set starts a whole set of full-size rows further on, whatever the row size.
        assert_eq!(layout.row_offset(3, 100, 0).expect("row"), 300);
        assert_eq!(layout.row_offset(3, 100, 1).expect("row"), 4096 + 300);
        assert_eq!(layout.set_offset(1).expect("set"), 4096);
        assert!(layout.row_offset(0, 1025, 0).is_err());
        assert!(layout.row_offset(4, 16, 0).is_err());
        assert!(layout.row_offset(0, 16, 2).is_err());
    }

    #[test]
    fn rounds_of_different_sizes_never_share_a_byte_between_the_two_sets() {
        // Consecutive rounds use different sets and may carry different row sizes (a step boundary
        // changes the row count). A set base that scaled with the row size put a small round's
        // second set inside a large round's first; with fixed bases no pair of sizes can overlap.
        let layout = SlabLayout::new(4, 655_360, 2).expect("layout");
        for small in [2usize, 10_240, 40_960, 655_360] {
            for large in [10_240usize, 163_840, 655_360] {
                let end_of_set_0 = layout.row_offset(3, large, 0).expect("row") + large;
                let start_of_set_1 = layout.row_offset(0, small, 1).expect("row");
                assert!(end_of_set_0 <= start_of_set_1, "{small} against {large}");
                let end_of_set_1 = layout.row_offset(3, small, 1).expect("row") + small;
                assert!(
                    end_of_set_1 <= layout.flags_offset(),
                    "{small} runs into the flags"
                );
            }
        }
    }

    #[test]
    fn every_region_is_aligned_for_the_words_and_the_sixteen_byte_copy() {
        for (world, row_cap, rails) in [
            (3usize, 1001usize, 1usize),
            (4, 10_240, 2),
            (8, 655_360, 2),
            (2, 7, 1),
        ] {
            let layout = SlabLayout::new(world, row_cap, rails).expect("layout");
            assert_eq!(layout.set_offset(1).expect("set") % REGION_ALIGN, 0);
            assert_eq!(layout.flags_offset() % REGION_ALIGN, 0);
            // The sources ring ends where the length ring begins, and the length ring ends the slab.
            assert_eq!(
                layout.source_offset(FLAG_SOURCES as u64 - 1) + 8,
                layout.lengths_offset()
            );
            assert_eq!(
                layout.lengths_offset() + world * LENGTH_RING * 8,
                layout.total_bytes().expect("total")
            );
        }
    }

    #[test]
    fn the_smallest_and_largest_layouts_are_refused_or_sized() {
        assert!(SlabLayout::new(0, 16, 1).is_err());
        assert!(SlabLayout::new(4, 0, 1).is_err());
        assert!(SlabLayout::new(4, 16, 0).is_err());
        assert!(SlabLayout::new(4, 16, MAX_RAILS + 1).is_err());
        let layout = SlabLayout::new(8, 655_360, 1).expect("layout");
        assert_eq!(
            layout.total_bytes().expect("total"),
            2 * 8 * 655_360 + (8 + 2 + FLAG_SOURCES + 8 * LENGTH_RING) * 8
        );
    }

    #[test]
    fn two_rails_each_get_a_flag_per_rank_and_the_doorbell_moves_past_them() {
        let layout = SlabLayout::new(4, 1024, 2).expect("layout");
        assert_eq!(layout.flag_offset(0, 3).expect("flag"), 8192 + 24);
        assert_eq!(layout.flag_offset(1, 0).expect("flag"), 8192 + 32);
        assert_eq!(layout.flag_offset(1, 3).expect("flag"), 8192 + 56);
        assert!(layout.flag_offset(2, 0).is_err());
        assert_eq!(layout.doorbell_offset(), 8192 + 64);
        assert_eq!(layout.abort_offset(), 8192 + 72);
        assert_eq!(layout.source_offset(0), 8192 + 80);
        assert_eq!(
            layout.total_bytes().expect("total"),
            8192 + 80 + FLAG_SOURCES * 8 + 4 * LENGTH_RING * 8
        );
    }

    #[test]
    fn a_write_tag_gives_back_its_round_its_peer_and_its_kind() {
        for (seq, peer, flag) in [(1u64, 0usize, false), (1, 3, true), (128_844, 2, false)] {
            assert_eq!(
                write_tag_parts(write_tag(seq, peer, flag)),
                (seq, peer, flag)
            );
        }
        // The part and the flag of one round to one peer are told apart.
        assert_ne!(write_tag(7, 1, false), write_tag(7, 1, true));
    }

    #[test]
    fn the_rails_parts_cover_a_row_exactly_once_on_a_sixty_four_byte_boundary() {
        for rails in 1..=MAX_RAILS {
            let layout = SlabLayout::new(4, 655_360, rails).expect("layout");
            for row_bytes in [
                2usize, 62, 64, 126, 128, 130, 10_240, 61_440, 61_442, 655_360,
            ] {
                let mut next = 0usize;
                for rail in 0..rails {
                    let (start, length) = layout.segment(rail, row_bytes);
                    assert_eq!(start, next, "{rails} rails, {row_bytes} bytes, rail {rail}");
                    assert_eq!(start % REGION_ALIGN, 0);
                    next = start + length;
                }
                assert_eq!(next, row_bytes, "{rails} rails, {row_bytes} bytes");
            }
        }
        // An even split of the row a c1 step sends, and nothing for the first rail of a row too
        // short to split.
        let layout = SlabLayout::new(4, 655_360, 2).expect("layout");
        assert_eq!(layout.segment(0, 61_440), (0, 30_720));
        assert_eq!(layout.segment(1, 61_440), (30_720, 30_720));
        assert_eq!(layout.segment(0, 100), (0, 0));
        assert_eq!(layout.segment(1, 100), (0, 100));
    }

    /// A word of a test slab, read the way the pump reads what the wire wrote.
    fn read_word(host: *mut u8, offset: usize) -> u64 {
        // SAFETY: every offset these tests use comes from the layout of the slab they allocated.
        unsafe { std::ptr::read_volatile(host.add(offset).cast::<u64>()) }
    }

    fn write_word(host: *mut u8, offset: usize, value: u64) {
        // SAFETY: as read_word.
        unsafe { std::ptr::write_volatile(host.add(offset).cast::<u64>(), value) };
    }

    #[test]
    fn an_arrival_is_the_least_rail_flag_and_a_lap_is_not_a_miss() {
        // The device and the pump both ask "has this peer published at least seq", never "exactly
        // seq", because a peer that is a round ahead has published every earlier round too. This
        // pins the predicate: the least flag over the rails is the arrival, and a peer two rounds
        // ahead satisfies an earlier round.
        let layout = SlabLayout::new(4, 1024, 2).expect("layout");
        let total = layout.total_bytes().expect("total");
        let mut slab = vec![0u64; total / WORD_BYTES];
        let host = slab.as_mut_ptr().cast::<u8>();
        for rail in 0..2usize {
            write_word(host, layout.flag_offset(rail, 1).expect("flag"), 3);
        }
        // SAFETY: host is the slab these offsets describe and it outlives the read.
        assert_eq!(unsafe { delivered(host, layout, 1) }, 3);
        // One rail behind is a peer that has not delivered the round in full.
        write_word(host, layout.flag_offset(1, 1).expect("flag"), 2);
        assert_eq!(unsafe { delivered(host, layout, 1) }, 2);
        // The wait is >=, so a peer's flag of 3 answers rounds 1 to 3.
        write_word(host, layout.flag_offset(1, 1).expect("flag"), 3);
        for seq in 1..=3u64 {
            assert!(
                unsafe { delivered(host, layout, 1) } >= seq,
                "a peer three ahead must answer round {seq}"
            );
        }
        // A rank outside the world has published nothing.
        assert_eq!(unsafe { delivered(host, layout, 4) }, 0);
    }

    #[test]
    fn a_published_length_is_keyed_by_round_so_the_next_round_cannot_overwrite_it() {
        // The arrival flag says a row arrived; the length says it is the row this round will fold.
        // Keying the length by peer alone would let a peer a round ahead rewrite the length the
        // current round is still checking, which is why it lives in a ring keyed by round.
        let layout = SlabLayout::new(4, 4096, 2).expect("layout");
        let total = layout.total_bytes().expect("total");
        let mut slab = vec![0u64; total / WORD_BYTES];
        let host = slab.as_mut_ptr().cast::<u8>();
        let rounds = [(1u64, 64u64), (2, 4096), (3, 10_240)];
        for (seq, bytes) in rounds {
            write_word(host, layout.length_slot(1, seq).expect("length"), bytes);
        }
        for (seq, bytes) in rounds {
            assert_eq!(
                read_word(host, layout.length_slot(1, seq).expect("length")),
                bytes,
                "round {seq}"
            );
        }
        // Two peers do not share a slot, and every slot is inside the slab.
        assert_ne!(
            layout.length_slot(1, 7).expect("length"),
            layout.length_slot(2, 7).expect("length")
        );
        for peer in 0..4usize {
            for seq in 0..(LENGTH_RING as u64 * 2) {
                let offset = layout.length_slot(peer, seq).expect("length");
                assert!(offset >= layout.lengths_offset());
                assert!(offset + WORD_BYTES <= total, "peer {peer} round {seq}");
            }
        }
        // The ring wraps, which is what makes it bounded: round 1 and round 1 + LENGTH_RING share
        // a slot, and the two-round window the row sets admit is far inside that.
        assert_eq!(
            layout.length_slot(0, 1).expect("length"),
            layout
                .length_slot(0, 1 + LENGTH_RING as u64)
                .expect("length")
        );
    }

    #[test]
    fn the_abort_tag_names_every_peer_and_not_only_rank_zero() {
        // The kernel packs the tag in the low two bytes and the peer in bits 16-23. Packing the
        // peer at bit 8, as it first did, changed the tag for every peer but 0, so the pump's
        // decode missed them and a real length disagreement was silent.
        let kernel_word = |peer: u64| ABORT_TAG_LENGTH | (peer << ABORT_PEER_SHIFT);
        for peer in 0..8u64 {
            let word = kernel_word(peer);
            assert_eq!(word & 0xffff, ABORT_TAG_LENGTH, "peer {peer}");
            assert_eq!((word >> ABORT_PEER_SHIFT) & 0xff, peer, "peer {peer}");
            assert_ne!(
                word, 1,
                "the word must not read as the host's own abort flag"
            );
        }
    }

    #[test]
    fn a_published_length_slot_names_its_publisher_on_both_ends() {
        // The sender writes into tile \`publisher\` and the receiver reads that same tile; a sender
        // that wrote into the destination's tile instead would leave every tile a receiver reads
        // at zero, which is a round that can never complete.
        let layout = SlabLayout::new(4, 4096, 2).expect("layout");
        for publisher in 0..4usize {
            for seq in 1..=3u64 {
                assert_eq!(
                    layout.length_slot(publisher, seq).expect("length"),
                    layout.lengths_offset()
                        + (publisher * LENGTH_RING + (seq % LENGTH_RING as u64) as usize)
                            * WORD_BYTES,
                    "publisher {publisher} round {seq}"
                );
            }
        }
    }

    #[test]
    fn the_kernel_and_this_module_agree_on_the_published_word_and_the_abort_tag() {
        // The kernel packs the abort tag and the length word with #defines; this module decodes
        // them with constants. A test that builds the word from the Rust constants and decodes it
        // with the same constants proves only that Rust agrees with itself, which is how the peer
        // at bit 8 survived a review. Read the kernel's own text and compare instead.
        let source = include_str!("../../../../kernels/opt_misc_sm121.cu");
        let value = |name: &str| -> u64 {
            let line = source
                .lines()
                .find(|line| line.trim_start().starts_with(&format!("#define {name} ")))
                .unwrap_or_else(|| panic!("{name} is not defined in the kernel"));
            let raw = line
                .trim()
                .rsplit(' ')
                .next()
                .expect("a define with a value")
                .trim_end_matches("ull")
                .trim_end_matches(';');
            // The kernel writes the tag in hex and the shifts in decimal; read each as written.
            match raw.strip_prefix("0x") {
                Some(digits) => u64::from_str_radix(digits, 16)
                    .unwrap_or_else(|_| panic!("{name} is not a hex literal: {raw}")),
                None => raw
                    .parse::<u64>()
                    .unwrap_or_else(|_| panic!("{name} is not a decimal literal: {raw}")),
            }
        };
        assert_eq!(value("OPT_EXCHANGE_ABORT_LENGTH"), ABORT_TAG_LENGTH);
        assert_eq!(
            value("OPT_EXCHANGE_ABORT_PEER_SHIFT"),
            u64::from(ABORT_PEER_SHIFT)
        );
        assert_eq!(
            value("OPT_EXCHANGE_LENGTH_SEQ_SHIFT"),
            u64::from(LENGTH_SEQ_SHIFT)
        );
        // The mask is written as an expression; check what it evaluates to against the module's.
        assert!(source.contains(
            "#define OPT_EXCHANGE_LENGTH_BYTES_MASK ((1ull << OPT_EXCHANGE_LENGTH_SEQ_SHIFT) - 1ull)"
        ));
        assert_eq!(LENGTH_BYTES_MASK, (1 << LENGTH_SEQ_SHIFT) - 1);
    }

    #[test]
    fn a_published_word_names_its_round_not_only_its_size() {
        // A rank that inserts or drops a round shifts every later sequence on itself. With the size
        // alone the peers' check would agree whenever the sizes happened to match; with the
        // sequence it refuses.
        let layout = SlabLayout::new(4, 4096, 2).expect("layout");
        let total = layout.total_bytes().expect("total");
        let mut slab = vec![0u64; total / WORD_BYTES];
        let host = slab.as_mut_ptr().cast::<u8>();
        write_word(
            host,
            layout.length_slot(1, 5).expect("slot"),
            length_word(4, 61_440),
        );
        let word = read_word(host, layout.length_slot(1, 5).expect("slot"));
        assert_eq!(
            word >> LENGTH_SEQ_SHIFT,
            4,
            "the publisher's round, not the reader's"
        );
        assert_eq!(word & LENGTH_BYTES_MASK, 61_440);
    }
}

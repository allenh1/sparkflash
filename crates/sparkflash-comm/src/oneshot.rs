//! The device-buffer one-shot collective seam contract.
//!
//! `docs/build/rdma-backend.md` section 15 names the gap this module starts to close: the CUDA
//! one-shot kernel (`kernels/rocenante_oneshot_sm121.cu` with the slot ABI in its `.cuh`) and the
//! RDMA data plane in [`crate::rdma`] describe two protocols that never meet. The kernel stages a
//! rank's row in coherent UMA and waits on host-written arrival words; the data plane moves frames
//! between host-memory staging slots and publishes nothing. This module is the seam *contract*: the
//! slot geometry, the dispatch plan, the receive-side proxy state machine and the registration
//! descriptor. It is pure data and pure transitions, so all of it is exercised on a host with no
//! RDMA device and no GPU, which is what makes the window a run instead of a debug session.
//!
//! What it does **not** do: launch the kernel, allocate a device alias, register a memory region or
//! post a work request. Those are the four pieces section 15.3 lists for the wiring increment, and
//! they are only reachable on a node with a device. The rule for the seam is the transport's rule:
//! no silent fallback. A call that cannot execute the device path returns an error naming the
//! prerequisite; it never quietly runs the host-slice collective instead.
//!
//! The numbers are pinned twice on purpose. `sparkflash-kernels`'s `roce_abi` module pins the same
//! ABI against the kernel source and the build probe; this crate cannot depend on it (the kernels
//! crate depends on the collective layer, not the other way round), so the constants below are
//! restated with the source line they mirror and a unit test asserts the arithmetic that matters
//! (arrival words first, then four 2 MiB rows, 8-byte aligned). A drift in either place fails a
//! test in that place; neither can move alone.

use std::fmt;

/// Rank count of the one-shot kernel's slot: the CUDA header fixes it at four.
pub const ONESHOT_RANKS: usize = 4;

/// One rank's row capacity, in bf16 elements: `1024 * 1024` in the kernel header.
pub const ONESHOT_ROW_ELEMENTS: usize = 1024 * 1024;

/// One rank's row capacity in bytes.
pub const ONESHOT_ROW_BYTES: usize = ONESHOT_ROW_ELEMENTS * 2;

/// Bytes reserved for the four arrival words.
pub const ONESHOT_ARRIVAL_BYTES: usize = ONESHOT_RANKS * 8;

/// The whole slot: the arrival words, then the four payload rows.
pub const ONESHOT_SLOT_BYTES: usize = ONESHOT_ARRIVAL_BYTES + ONESHOT_RANKS * ONESHOT_ROW_BYTES;

/// Alignment the slot and its payload rows are allocated and registered with.
pub const ONESHOT_SLOT_ALIGN: usize = 8;

/// Rails the one-shot path posts each peer's row over.
pub const ONESHOT_RAILS: usize = 2;

/// The dispatch phase the proxy expects in a frame's immediate.
pub const ONESHOT_PHASE: u8 = 1;

/// The frame kind the proxy expects in a frame's immediate.
pub const ONESHOT_KIND: u8 = 1;

/// Byte offset of rank `rank`'s arrival word.
#[must_use]
pub fn arrival_offset(rank: usize) -> usize {
    rank * 8
}

/// Byte offset of rank `rank`'s payload row inside the slot.
#[must_use]
pub fn row_offset(rank: usize) -> usize {
    ONESHOT_ARRIVAL_BYTES + rank * ONESHOT_ROW_BYTES
}

/// Why a seam call was refused.
///
/// Every variant is a refusal before any device work: the seam validates its inputs first, exactly
/// as the transport validates its layout. There is no "carry on best effort" arm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeamError {
    /// The rank is outside the slot's four rows.
    RankOutOfRange {
        /// The offending rank.
        rank: usize,
    },
    /// The world size is not the kernel's four ranks.
    NotFourRanks {
        /// The world size the caller asked for.
        world: usize,
    },
    /// Zero elements in a collective.
    EmptyPayload,
    /// The payload does not fit one rank's row.
    PayloadTooLarge {
        /// Requested elements.
        elements: usize,
        /// The row limit.
        limit: usize,
    },
    /// The rail count is not a supported mode.
    RailCount {
        /// The requested rail count.
        rails: usize,
    },
    /// A pointer did not meet the slot's alignment.
    Misaligned {
        /// What the pointer points at.
        what: &'static str,
        /// The offending address.
        address: u64,
    },
    /// A pointer was null.
    NullAddress {
        /// What the pointer points at.
        what: &'static str,
    },
    /// A registered slot did not have the ABI's size.
    SlotBytes {
        /// The registered size.
        bytes: usize,
        /// The ABI's size.
        expected: usize,
    },
    /// The sequence number did not advance.
    Sequence {
        /// The previous sequence.
        previous: u64,
        /// The requested sequence.
        requested: u64,
    },
    /// A dispatch record shows the host-staging path ran instead of the device kernel.
    HostFallback {
        /// The dispatch's sequence number.
        seq: u64,
    },
    /// A payload size that is not a whole number of bf16 elements.
    OddPayload {
        /// The offending byte count.
        bytes: usize,
    },
    /// A frame arrived before the local doorbell, or for the wrong exchange.
    Protocol(String),
}

impl fmt::Display for SeamError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankOutOfRange { rank } => write!(
                formatter,
                "rank {rank} is outside the one-shot slot's {ONESHOT_RANKS} rows"
            ),
            Self::NotFourRanks { world } => write!(
                formatter,
                "the one-shot seam is a {ONESHOT_RANKS}-rank collective, world_size {world} was requested"
            ),
            Self::EmptyPayload => write!(
                formatter,
                "a one-shot collective needs at least one element"
            ),
            Self::PayloadTooLarge { elements, limit } => write!(
                formatter,
                "{elements} bf16 elements exceed the one-shot row capacity {limit}"
            ),
            Self::RailCount { rails } => write!(
                formatter,
                "rail count {rails} is not a one-shot mode (1 or {ONESHOT_RAILS})"
            ),
            Self::Misaligned { what, address } => write!(
                formatter,
                "{what} address {address:#x} is not {ONESHOT_SLOT_ALIGN}-byte aligned"
            ),
            Self::NullAddress { what } => write!(formatter, "{what} address is null"),
            Self::SlotBytes { bytes, expected } => write!(
                formatter,
                "the registered slot is {bytes} bytes, the ABI requires {expected}"
            ),
            Self::Sequence {
                previous,
                requested,
            } => write!(
                formatter,
                "sequence must strictly increase: previous {previous}, requested {requested}"
            ),
            Self::HostFallback { seq } => write!(
                formatter,
                "the dispatch at sequence {seq} took the host-staging path; the device seam refuses a silent fallback"
            ),
            Self::OddPayload { bytes } => write!(
                formatter,
                "{bytes} payload bytes are not whole bf16 elements"
            ),
            Self::Protocol(reason) => {
                write!(formatter, "one-shot seam protocol violation: {reason}")
            }
        }
    }
}

impl std::error::Error for SeamError {}

/// A device buffer the one-shot collective writes its result into.
///
/// The address is the **device** virtual address (a cudaDevicePtr), not the host address of a
/// pinned mapping; the pinned-host form has a separate device alias, which is
/// [`SlotKind::PinnedHostAlias`]. Validation happens here so a caller cannot hand the seam a
/// pointer that a registration would reject later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceBuffer {
    address: u64,
    bytes: usize,
}

impl DeviceBuffer {
    /// Wraps a device pointer and length.
    ///
    /// # Errors
    /// Refuses a null pointer, a misaligned address, a zero length and a length that is not a whole
    /// number of bf16 elements.
    pub fn new(address: u64, bytes: usize) -> Result<Self, SeamError> {
        if address == 0 {
            return Err(SeamError::NullAddress {
                what: "device buffer",
            });
        }
        if !address.is_multiple_of(ONESHOT_SLOT_ALIGN as u64) {
            return Err(SeamError::Misaligned {
                what: "device buffer",
                address,
            });
        }
        if bytes == 0 || !bytes.is_multiple_of(2) {
            return Err(SeamError::Protocol(format!(
                "a device buffer needs a positive whole number of bf16 elements, got {bytes} bytes"
            )));
        }
        Ok(Self { address, bytes })
    }

    /// The device address.
    #[must_use]
    pub fn address(&self) -> u64 {
        self.address
    }

    /// The buffer length in bytes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The buffer length in bf16 elements.
    #[must_use]
    pub fn elements(&self) -> usize {
        self.bytes / 2
    }
}

/// One one-shot exchange: a sequence number, a payload size and a rail count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exchange {
    seq: u64,
    elements: usize,
    rails: usize,
}

impl Exchange {
    /// Validates one exchange against the slot ABI.
    ///
    /// # Errors
    /// Refuses zero elements, a payload over one row, and a rail count other than 1 or
    /// [`ONESHOT_RAILS`].
    pub fn new(seq: u64, elements: usize, rails: usize) -> Result<Self, SeamError> {
        if elements == 0 {
            return Err(SeamError::EmptyPayload);
        }
        if elements > ONESHOT_ROW_ELEMENTS {
            return Err(SeamError::PayloadTooLarge {
                elements,
                limit: ONESHOT_ROW_ELEMENTS,
            });
        }
        if rails != 1 && rails != ONESHOT_RAILS {
            return Err(SeamError::RailCount { rails });
        }
        Ok(Self {
            seq,
            elements,
            rails,
        })
    }

    /// The sequence number.
    #[must_use]
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// The payload size in elements.
    #[must_use]
    pub fn elements(&self) -> usize {
        self.elements
    }

    /// The payload size in bytes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.elements * 2
    }

    /// The rail count.
    #[must_use]
    pub fn rails(&self) -> usize {
        self.rails
    }

    /// Bytes of the payload sent over one rail.
    ///
    /// The split is in whole bf16 elements: an odd element count puts the extra element on rail 0.
    /// Rail bytes for a single-rail exchange are the whole payload.
    #[must_use]
    pub fn rail_bytes(&self, rail: usize) -> usize {
        if self.rails == 1 {
            return if rail == 0 { self.bytes() } else { 0 };
        }
        let rail0 = (self.elements - self.elements / 2) * 2;
        match rail {
            0 => rail0,
            1 => self.bytes() - rail0,
            _ => 0,
        }
    }

    /// Total payload bytes posted for the exchange: this rank's row to every peer on every rail.
    #[must_use]
    pub fn posted_bytes(&self) -> usize {
        (ONESHOT_RANKS - 1) * self.bytes()
    }
}

/// One frame the receive-side proxy observed on a rail, decoded from the work-request immediate.
///
/// The immediate corroborates the frame; it does not source the arrival word. The proxy is told the
/// full `(seq, elements)` by the dispatch, and publishes `arrival_seq[sender] = seq` only after
/// `exchange.rails` frames for that `(sender, seq)` have been observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RailFrame {
    /// The rank that sent the frame, from the immediate's sender byte.
    pub sender: u8,
    /// The rail the frame arrived on.
    pub rail: usize,
    /// The sender's phase byte.
    pub phase: u8,
    /// The sender's kind byte.
    pub kind: u8,
    /// The low byte of the sequence number.
    pub sequence_low: u8,
}

/// The receive-side proxy state machine for one exchange.
///
/// Section 15.2(b) of `docs/build/rdma-backend.md` describes the shape: poll the local doorbell,
/// post the rows, drain the send CQ, and publish each peer's arrival word once both rails' frames
/// for the exchange have been observed. Only the transitions are here; the QP posting and polling
/// sit above this type.
#[derive(Debug, Clone)]
pub struct ArrivalProxy {
    rank: usize,
    exchange: Exchange,
    /// Whether the local doorbell for this exchange has been observed.
    doorbell: bool,
    /// Per peer, per rail: whether that rail's frame has been observed.
    observed: Vec<Vec<bool>>,
    /// Per peer, the sequence published into its arrival word (None until published).
    published: Vec<Option<u64>>,
}

impl ArrivalProxy {
    /// Creates a proxy for one exchange on one rank.
    ///
    /// # Errors
    /// The same refusals [`Exchange::new`] returns, plus a rank outside the slot.
    pub fn new(rank: usize, exchange: Exchange) -> Result<Self, SeamError> {
        if rank >= ONESHOT_RANKS {
            return Err(SeamError::RankOutOfRange { rank });
        }
        Ok(Self {
            rank,
            exchange,
            doorbell: false,
            observed: vec![vec![false; exchange.rails()]; ONESHOT_RANKS],
            published: vec![None; ONESHOT_RANKS],
        })
    }

    /// The rank this proxy serves.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The exchange this proxy serves.
    #[must_use]
    pub fn exchange(&self) -> Exchange {
        self.exchange
    }

    /// Records that the local doorbell for this exchange reached `seq`.
    ///
    /// # Errors
    /// Refuses a sequence that is not this exchange's, and a second doorbell for the same exchange.
    pub fn on_doorbell(&mut self, seq: u64) -> Result<(), SeamError> {
        if seq != self.exchange.seq() {
            return Err(SeamError::Protocol(format!(
                "doorbell sequence {seq} does not match the dispatch's {}",
                self.exchange.seq()
            )));
        }
        if self.doorbell {
            return Err(SeamError::Sequence {
                previous: self.exchange.seq(),
                requested: seq,
            });
        }
        self.doorbell = true;
        Ok(())
    }

    /// Observes one rail frame and, when it completes a peer's set, publishes that arrival word.
    ///
    /// Returns the peer whose arrival word was written by this call, `None` when the frame is
    /// recorded without completing a peer, and refuses anything the dispatch did not name.
    ///
    /// # Errors
    /// Refuses a frame before the doorbell, a frame from outside the world or from this rank, a
    /// frame whose `(sequence_low, phase, kind)` disagrees with the exchange, a rail outside the
    /// exchange, a duplicate that contradicts the recorded frame, and a peer already published.
    pub fn on_frame(&mut self, frame: &RailFrame) -> Result<Option<usize>, SeamError> {
        if !self.doorbell {
            return Err(SeamError::Protocol(
                "a rail frame arrived before the local doorbell".to_owned(),
            ));
        }
        let peer = frame.sender as usize;
        if peer >= ONESHOT_RANKS {
            return Err(SeamError::Protocol(format!(
                "frame claims sender rank {} outside the slot",
                frame.sender
            )));
        }
        if peer == self.rank {
            return Err(SeamError::Protocol(
                "a rank observed a frame from itself".to_owned(),
            ));
        }
        if frame.phase != ONESHOT_PHASE || frame.kind != ONESHOT_KIND {
            return Err(SeamError::Protocol(format!(
                "frame carries phase {} kind {} (expected {ONESHOT_PHASE}/{ONESHOT_KIND})",
                frame.phase, frame.kind
            )));
        }
        if frame.sequence_low != (self.exchange.seq() & 0xff) as u8 {
            return Err(SeamError::Protocol(format!(
                "frame carries sequence byte {} but the dispatch is at {}",
                frame.sequence_low,
                self.exchange.seq()
            )));
        }
        if frame.rail >= self.exchange.rails() {
            return Err(SeamError::Protocol(format!(
                "frame arrived on rail {} but the exchange uses {}",
                frame.rail,
                self.exchange.rails()
            )));
        }
        if self.published[peer].is_some() {
            return Err(SeamError::Sequence {
                previous: self.published[peer].unwrap_or_default(),
                requested: self.exchange.seq(),
            });
        }
        self.observed[peer][frame.rail] = true;
        if self.observed[peer].iter().all(|seen| *seen) {
            self.published[peer] = Some(self.exchange.seq());
            Ok(Some(peer))
        } else {
            Ok(None)
        }
    }

    /// The arrival word to write for a peer, if the proxy has published it.
    #[must_use]
    pub fn arrival(&self, peer: usize) -> Option<u64> {
        self.published.get(peer).copied().flatten()
    }

    /// Whether every peer's arrival word has been published.
    #[must_use]
    pub fn all_published(&self) -> bool {
        self.published
            .iter()
            .enumerate()
            .all(|(peer, published)| peer == self.rank || published.is_some())
    }

    /// The peers whose arrival word has been published, in ascending rank order.
    #[must_use]
    pub fn published_peers(&self) -> Vec<usize> {
        self.published
            .iter()
            .enumerate()
            .filter(|(peer, published)| *peer != self.rank && published.is_some())
            .map(|(peer, _)| peer)
            .collect()
    }
}

/// How the one-shot slot is registered with ibverbs.
///
/// Section 15.5 of `docs/build/rdma-backend.md` leaves both forms unverified on this hardware, so
/// the one-shot path keeps them separate instead of assuming one: the pinned-host form has a host
/// address plus the device alias `cuMemHostGetDevicePointer` returns, the device form registers the
/// device allocation directly (dmabuf) and has no host address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotKind {
    /// Pinned host memory with a device alias (the kernel writes through the alias).
    PinnedHostAlias {
        /// Host address of the pinned mapping.
        host_address: u64,
        /// Device alias the kernel dereferences.
        device_alias: u64,
    },
    /// A device allocation registered directly (dmabuf or an equivalent path).
    DeviceMemory {
        /// Device address of the slot.
        device_address: u64,
    },
}

/// A registered one-shot slot: its kind and size, validated before any MR or kernel touches it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotRegistration {
    kind: SlotKind,
    slot_bytes: usize,
}

impl SlotRegistration {
    /// Validates a slot registration against the ABI.
    ///
    /// # Errors
    /// Refuses a size other than [`ONESHOT_SLOT_BYTES`], a null or misaligned address, and an
    /// alias that is not distinct from the host address (the kernel must not write the mapping's
    /// host address through the device path; that is what the alias is for).
    pub fn new(kind: SlotKind, slot_bytes: usize) -> Result<Self, SeamError> {
        if slot_bytes != ONESHOT_SLOT_BYTES {
            return Err(SeamError::SlotBytes {
                bytes: slot_bytes,
                expected: ONESHOT_SLOT_BYTES,
            });
        }
        match kind {
            SlotKind::PinnedHostAlias {
                host_address,
                device_alias,
            } => {
                check_address(host_address, "pinned host slot")?;
                check_address(device_alias, "device alias")?;
                if host_address == device_alias {
                    return Err(SeamError::Protocol(
                        "the device alias equals the host address; a pinned-host slot needs a distinct alias"
                            .to_owned(),
                    ));
                }
            }
            SlotKind::DeviceMemory { device_address } => {
                check_address(device_address, "device slot")?;
            }
        }
        Ok(Self { kind, slot_bytes })
    }

    /// The registration kind.
    #[must_use]
    pub fn kind(&self) -> SlotKind {
        self.kind
    }

    /// The slot size in bytes.
    #[must_use]
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// The address the kernel uses for this slot.
    #[must_use]
    pub fn kernel_address(&self) -> u64 {
        match self.kind {
            SlotKind::PinnedHostAlias { device_alias, .. } => device_alias,
            SlotKind::DeviceMemory { device_address } => device_address,
        }
    }
}

fn check_address(address: u64, what: &'static str) -> Result<(), SeamError> {
    if address == 0 {
        return Err(SeamError::NullAddress { what });
    }
    if !address.is_multiple_of(ONESHOT_SLOT_ALIGN as u64) {
        return Err(SeamError::Misaligned { what, address });
    }
    Ok(())
}

/// The kernel launch parameters one dispatch hands the one-shot launcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dispatch {
    release: u64,
    peer_epoch: u64,
    rank: usize,
    slot: SlotRegistration,
    exchange: Exchange,
}

impl Dispatch {
    /// Validates a dispatch.
    ///
    /// # Errors
    /// Refuses a rank outside the slot and a release/epoch that is zero (the kernel treats zero as
    /// "not staged"/"not ready", so a dispatch may not use it).
    pub fn new(
        rank: usize,
        slot: SlotRegistration,
        exchange: Exchange,
        release: u64,
        peer_epoch: u64,
    ) -> Result<Self, SeamError> {
        if rank >= ONESHOT_RANKS {
            return Err(SeamError::RankOutOfRange { rank });
        }
        if release == 0 || peer_epoch == 0 {
            return Err(SeamError::Protocol(
                "the release and peer-epoch words must be non-zero".to_owned(),
            ));
        }
        Ok(Self {
            release,
            peer_epoch,
            rank,
            slot,
            exchange,
        })
    }

    /// The rank that dispatches.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The slot.
    #[must_use]
    pub fn slot(&self) -> SlotRegistration {
        self.slot
    }

    /// The exchange.
    #[must_use]
    pub fn exchange(&self) -> Exchange {
        self.exchange
    }

    /// The release word the kernel publishes when this rank's row is staged.
    #[must_use]
    pub fn release(&self) -> u64 {
        self.release
    }

    /// The peer epoch the kernel waits for.
    #[must_use]
    pub fn peer_epoch(&self) -> u64 {
        self.peer_epoch
    }
}

/// A monotonic sequence counter for one rank's dispatches.
///
/// The seam refuses a sequence that does not strictly increase, so the counter is the one place the
/// rule is enforced for the dispatch side. It starts at the caller's first value; zero is allowed
/// because the kernel's arrival words use the sequence itself, and only the release/epoch words
/// reserve zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SequenceCounter {
    previous: Option<u64>,
}

impl SequenceCounter {
    /// Creates a counter with no previous sequence.
    #[must_use]
    pub fn new() -> Self {
        Self { previous: None }
    }

    /// The last accepted sequence.
    #[must_use]
    pub fn previous(&self) -> Option<u64> {
        self.previous
    }

    /// Accepts `seq` and returns it.
    ///
    /// # Errors
    /// Refuses a sequence that does not strictly increase on the previous one.
    pub fn accept(&mut self, seq: u64) -> Result<u64, SeamError> {
        if let Some(previous) = self.previous
            && seq <= previous
        {
            return Err(SeamError::Sequence {
                previous,
                requested: seq,
            });
        }
        self.previous = Some(seq);
        Ok(seq)
    }
}

/// One chunk of a payload larger than a single one-shot row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunk {
    index: usize,
    offset_bytes: usize,
    elements: usize,
}

impl Chunk {
    /// The chunk's index, from zero.
    #[must_use]
    pub fn index(&self) -> usize {
        self.index
    }

    /// The byte offset of the chunk's first element in the payload.
    #[must_use]
    pub fn offset_bytes(&self) -> usize {
        self.offset_bytes
    }

    /// The chunk's element count.
    #[must_use]
    pub fn elements(&self) -> usize {
        self.elements
    }

    /// The chunk's byte count.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.elements * 2
    }
}

/// The chunked form of a prefill-sized collective: one row-sized one-shot exchange per chunk,
/// strictly in order, reusing the same slot rows.
///
/// **The contract (S5.4).** A payload larger than ONESHOT_ROW_BYTES is split into chunks of at most
/// one row. Each chunk is one Exchange with its own sequence number, and the chunks run in strict
/// order: a rank posts chunk i's rows to every peer on every rail, waits until every peer's arrival
/// word for chunk i is published, then posts chunk i+1 into the same rows. The reuse is what bounds
/// the ring; the order is what makes the reuse safe, so the receiver refuses a frame whose chunk is
/// not the one in flight (see ChunkedProxy). ChunkPlan::dispatches cuts the per-chunk descriptors a
/// launcher consumes: one dispatch per chunk, same slot and release/epoch words, strictly increasing
/// sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkPlan {
    total_bytes: usize,
    rails: usize,
    chunks: Vec<Chunk>,
}

impl ChunkPlan {
    /// Splits a payload into row-sized chunks.
    ///
    /// # Errors
    /// [SeamError::EmptyPayload] for zero bytes, [SeamError::OddPayload] for a byte count that is not
    /// whole bf16 elements, [SeamError::RailCount] for a rail count other than 1 or ONESHOT_RAILS.
    pub fn new(total_bytes: usize, rails: usize) -> Result<Self, SeamError> {
        if total_bytes == 0 {
            return Err(SeamError::EmptyPayload);
        }
        if rails != 1 && rails != ONESHOT_RAILS {
            return Err(SeamError::RailCount { rails });
        }
        if !total_bytes.is_multiple_of(2) {
            return Err(SeamError::OddPayload { bytes: total_bytes });
        }
        let mut chunks = Vec::new();
        let mut offset = 0usize;
        while offset < total_bytes {
            let bytes = (total_bytes - offset).min(ONESHOT_ROW_BYTES);
            chunks.push(Chunk {
                index: chunks.len(),
                offset_bytes: offset,
                elements: bytes / 2,
            });
            offset += bytes;
        }
        Ok(Self {
            total_bytes,
            rails,
            chunks,
        })
    }

    /// The chunks, in order.
    #[must_use]
    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }

    /// The number of chunks.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// The payload bytes.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// The payload elements.
    #[must_use]
    pub fn total_elements(&self) -> usize {
        self.total_bytes / 2
    }

    /// The rail count.
    #[must_use]
    pub fn rails(&self) -> usize {
        self.rails
    }

    /// The exchange for one chunk at a given sequence number.
    ///
    /// # Errors
    /// [SeamError::Protocol] for a chunk index outside the plan, plus whatever Exchange::new refuses.
    pub fn exchange(&self, index: usize, seq: u64) -> Result<Exchange, SeamError> {
        let chunk = self.chunks.get(index).ok_or_else(|| {
            SeamError::Protocol(format!(
                "chunk {index} is outside a plan of {} chunks",
                self.chunks.len()
            ))
        })?;
        Exchange::new(seq, chunk.elements, self.rails)
    }

    /// The sequence number chunk index runs at, given the first chunk's sequence.
    ///
    /// # Errors
    /// [SeamError::Protocol] for an index outside the plan or a sequence overflow.
    pub fn chunk_seq(&self, base_seq: u64, index: usize) -> Result<u64, SeamError> {
        if index >= self.chunks.len() {
            return Err(SeamError::Protocol(format!(
                "chunk {index} is outside a plan of {} chunks",
                self.chunks.len()
            )));
        }
        base_seq
            .checked_add(index as u64)
            .ok_or_else(|| SeamError::Protocol("the chunked sequence overflowed".to_owned()))
    }

    /// The bytes this rank posts over all chunks on all rails.
    ///
    /// # Errors
    /// [SeamError::Protocol] when the total overflows.
    pub fn posted_bytes(&self) -> Result<usize, SeamError> {
        (ONESHOT_RANKS - 1)
            .checked_mul(self.total_bytes)
            .ok_or_else(|| SeamError::Protocol("the posted byte total overflowed".to_owned()))
    }

    /// One dispatch per chunk: the list a launcher consumes.
    ///
    /// # Errors
    /// [SeamError] from Dispatch::new or a sequence overflow.
    pub fn dispatches(
        &self,
        rank: usize,
        slot: SlotRegistration,
        release: u64,
        peer_epoch: u64,
        base_seq: u64,
    ) -> Result<Vec<Dispatch>, SeamError> {
        let mut dispatches = Vec::with_capacity(self.chunks.len());
        for index in 0..self.chunks.len() {
            let exchange = self.exchange(index, self.chunk_seq(base_seq, index)?)?;
            dispatches.push(Dispatch::new(rank, slot, exchange, release, peer_epoch)?);
        }
        Ok(dispatches)
    }
}

/// The receive-side state machine for a chunked exchange.
///
/// One ArrivalProxy per chunk: the chunk in flight validates the frames, and the caller advances to
/// the next chunk after every peer's arrival word for the current chunk is published. A frame from a
/// different chunk is refused with a chunk-order message, which is what keeps row reuse safe: the
/// rows of chunk i+1 must not be posted before every peer has read chunk i.
#[derive(Debug, Clone)]
pub struct ChunkedProxy {
    rank: usize,
    plan: ChunkPlan,
    base_seq: u64,
    current: usize,
    completed: usize,
    proxy: ArrivalProxy,
}

impl ChunkedProxy {
    /// Creates the state machine at chunk zero.
    ///
    /// # Errors
    /// [SeamError::RankOutOfRange] plus whatever the first chunk's exchange refuses.
    pub fn new(rank: usize, plan: ChunkPlan, base_seq: u64) -> Result<Self, SeamError> {
        let proxy = ArrivalProxy::new(rank, plan.exchange(0, base_seq)?)?;
        Ok(Self {
            rank,
            plan,
            base_seq,
            current: 0,
            completed: 0,
            proxy,
        })
    }

    /// The plan this proxy walks.
    #[must_use]
    pub fn plan(&self) -> &ChunkPlan {
        &self.plan
    }

    /// The chunk in flight.
    #[must_use]
    pub fn current_index(&self) -> usize {
        self.current
    }

    /// The exchange in flight.
    #[must_use]
    pub fn current_exchange(&self) -> Exchange {
        self.proxy.exchange()
    }

    /// Observes the local doorbell for the chunk in flight.
    ///
    /// # Errors
    /// Whatever ArrivalProxy::on_doorbell refuses (a second doorbell for the same chunk is a
    /// sequence regression).
    pub fn on_doorbell(&mut self) -> Result<(), SeamError> {
        self.proxy.on_doorbell(self.proxy.exchange().seq())
    }

    /// Observes one rail frame for the chunk in flight.
    ///
    /// # Errors
    /// A frame whose sequence byte belongs to another chunk is refused as out of chunk order before
    /// the per-chunk proxy sees it; otherwise whatever ArrivalProxy::on_frame refuses.
    pub fn on_frame(&mut self, frame: &RailFrame) -> Result<Option<usize>, SeamError> {
        if self.is_finished() {
            return Err(SeamError::Protocol(
                "a rail frame arrived after the last chunk completed".to_owned(),
            ));
        }
        let expected = (self.proxy.exchange().seq() & 0xff) as u8;
        if frame.sequence_low != expected {
            return Err(SeamError::Protocol(format!(
                "a frame for sequence byte {} arrived while chunk {} (sequence byte {expected}) is in flight: chunks run in strict order",
                frame.sequence_low, self.current
            )));
        }
        self.proxy.on_frame(frame)
    }

    /// The arrival word published for a peer in the current chunk, if any.
    #[must_use]
    pub fn arrival(&self, peer: usize) -> Option<u64> {
        self.proxy.arrival(peer)
    }

    /// Whether every peer's arrival word for the current chunk is published.
    #[must_use]
    pub fn current_published(&self) -> bool {
        self.proxy.all_published()
    }

    /// The number of chunks completed.
    #[must_use]
    pub fn completed_chunks(&self) -> usize {
        self.completed
    }

    /// Whether every chunk has completed.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.completed == self.plan.chunk_count()
    }

    /// Advances to the next chunk once the current one is fully published.
    ///
    /// Returns true when it advanced, false when the current chunk is not complete yet or the last
    /// chunk just completed (then is_finished turns true).
    ///
    /// # Errors
    /// [SeamError] if the next chunk's exchange cannot be built (a plan built by this type cannot).
    pub fn advance(&mut self) -> Result<bool, SeamError> {
        if self.is_finished() || !self.proxy.all_published() {
            return Ok(false);
        }
        self.completed = self.current + 1;
        if self.completed == self.plan.chunk_count() {
            return Ok(false);
        }
        self.current = self.completed;
        let seq = self.plan.chunk_seq(self.base_seq, self.current)?;
        self.proxy = ArrivalProxy::new(self.rank, self.plan.exchange(self.current, seq)?)?;
        Ok(true)
    }
}

/// How a dispatch actually moved the bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchPath {
    /// The kernel reduced the rows in place from device memory.
    DeviceKernel,
    /// A host-slice collective ran instead: the fallback the seam must never take silently.
    HostStaging,
}

/// The pinned cubin baseline entry the launcher runs, by name.
pub const ONESHOT_KERNEL_NAME: &str = "sparkflash_rocenante_oneshot_sm121";

/// The device addresses one dispatch used.
///
/// `local_row` must be exactly `slot + row_offset(rank)`: the kernel dereferences one slot object,
/// and the local row is a fixed row inside it. `peer_rows` are the three peers' row addresses, in
/// ascending rank order with this rank skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchAddresses {
    /// The slot's kernel-visible device address on this rank.
    pub slot: u64,
    /// This rank's row inside that slot.
    pub local_row: u64,
    /// The peers' row addresses, ascending rank order, this rank skipped.
    pub peer_rows: [u64; ONESHOT_RANKS - 1],
}

/// One dispatch as the launcher observed it.
///
/// The launcher emits one record for **every** collective call, unconditionally - not behind a
/// verbosity flag and not as a log line - because a host fallback that only shows up in logging
/// nobody enabled is the same as no assertion (the S5.3 gate item (d)). A real dispatch carries
/// device addresses only and zero host-copy bytes; [`DispatchRecord::device_path_ok`] is the
/// check the evidence writer applies, and [`DispatchLog`] is the list a window run ships.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchRecord {
    rank: usize,
    seq: u64,
    exchange: Exchange,
    slot_address: u64,
    local_row_address: u64,
    peer_row_addresses: [u64; ONESHOT_RANKS - 1],
    launch: LaunchPath,
    host_copy_bytes: usize,
    elapsed_ns: u64,
}

impl DispatchRecord {
    /// Validates and records one dispatch.
    ///
    /// The addresses are the **device** addresses the kernel dereferences: the slot's base for this
    /// rank, this rank's row inside it (which must be exactly `slot + row_offset(rank)`), and the
    /// three peers' row addresses. A host-staging dispatch is recorded, not refused here: the
    /// record's job is to say what happened, and [`device_path_ok`](Self::device_path_ok) is the
    /// gate that refuses it.
    ///
    /// # Errors
    /// Refuses a rank outside the slot, a null or misaligned address, and a local row that is not
    /// the slot's `row_offset(rank)` (a record that claims that is not the seam this contract
    /// describes).
    pub fn new(
        rank: usize,
        seq: u64,
        exchange: Exchange,
        addresses: DispatchAddresses,
        launch: LaunchPath,
        host_copy_bytes: usize,
        elapsed_ns: u64,
    ) -> Result<Self, SeamError> {
        if rank >= ONESHOT_RANKS {
            return Err(SeamError::RankOutOfRange { rank });
        }
        check_address(addresses.slot, "dispatch slot")?;
        check_address(addresses.local_row, "local row")?;
        let expected_row = addresses.slot + row_offset(rank) as u64;
        if addresses.local_row != expected_row {
            return Err(SeamError::Protocol(format!(
                "local row {:#x} is not the slot's row_offset({rank}) at {expected_row:#x}",
                addresses.local_row
            )));
        }
        for address in &addresses.peer_rows {
            check_address(*address, "peer row")?;
        }
        Ok(Self {
            rank,
            seq,
            exchange,
            slot_address: addresses.slot,
            local_row_address: addresses.local_row,
            peer_row_addresses: addresses.peer_rows,
            launch,
            host_copy_bytes,
            elapsed_ns,
        })
    }

    /// The dispatching rank.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The exchange this dispatch ran.
    #[must_use]
    pub fn exchange(&self) -> Exchange {
        self.exchange
    }

    /// The dispatch's sequence number.
    #[must_use]
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// The slot's kernel-visible device address.
    #[must_use]
    pub fn slot_address(&self) -> u64 {
        self.slot_address
    }

    /// This rank's row inside the slot.
    #[must_use]
    pub fn local_row_address(&self) -> u64 {
        self.local_row_address
    }

    /// The three peers' row addresses.
    #[must_use]
    pub fn peer_row_addresses(&self) -> [u64; ONESHOT_RANKS - 1] {
        self.peer_row_addresses
    }

    /// Which path ran.
    #[must_use]
    pub fn launch(&self) -> LaunchPath {
        self.launch
    }

    /// Bytes the dispatch copied through the host (must be 0 for the gate).
    #[must_use]
    pub fn host_copy_bytes(&self) -> usize {
        self.host_copy_bytes
    }

    /// The dispatch's measured wall time, in nanoseconds (the launcher's own clock).
    #[must_use]
    pub fn elapsed_ns(&self) -> u64 {
        self.elapsed_ns
    }

    /// Whether this dispatch ran the device path with no host copy.
    ///
    /// # Errors
    /// Returns [`SeamError::HostFallback`] naming the sequence when the record shows the
    /// host-staging path or a non-zero host copy, so the evidence writer cannot accept it.
    pub fn device_path_ok(&self) -> Result<(), SeamError> {
        if self.launch != LaunchPath::DeviceKernel || self.host_copy_bytes != 0 {
            return Err(SeamError::HostFallback { seq: self.seq });
        }
        Ok(())
    }
}

/// Every dispatch of one run, in sequence order.
///
/// The evidence writer ships this list as-is; a window run that took the host path anywhere has a
/// record saying so, and [`DispatchLog::device_path_ok`] refuses it.
#[derive(Debug, Clone, Default)]
pub struct DispatchLog {
    records: Vec<DispatchRecord>,
}

impl DispatchLog {
    /// Creates an empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one dispatch.
    ///
    /// # Errors
    /// Refuses a sequence that does not strictly increase on the previous record.
    pub fn record(&mut self, record: DispatchRecord) -> Result<(), SeamError> {
        if let Some(previous) = self.records.last()
            && record.seq <= previous.seq
        {
            return Err(SeamError::Sequence {
                previous: previous.seq,
                requested: record.seq,
            });
        }
        self.records.push(record);
        Ok(())
    }

    /// The records, in sequence order.
    #[must_use]
    pub fn records(&self) -> &[DispatchRecord] {
        &self.records
    }

    /// The number of recorded dispatches.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the log is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Total host-copy bytes over the run (must be 0 for the gate).
    #[must_use]
    pub fn host_copy_bytes(&self) -> usize {
        self.records
            .iter()
            .map(|record| record.host_copy_bytes())
            .sum()
    }

    /// Whether every recorded dispatch took the device path.
    #[must_use]
    pub fn device_path_only(&self) -> bool {
        self.records
            .iter()
            .all(|record| record.device_path_ok().is_ok())
    }

    /// Whether every recorded dispatch took the device path, naming the first offender.
    ///
    /// # Errors
    /// Returns the first [`SeamError::HostFallback`] in sequence order.
    pub fn device_path_ok(&self) -> Result<(), SeamError> {
        self.records
            .iter()
            .try_for_each(DispatchRecord::device_path_ok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_slot_matches_the_kernel_abi_arithmetic() {
        // The CUDA header's struct: uint64_t arrival_seq[4]; then four bf16 payload rows of
        // 1024 * 1024 elements. Nothing else lives in the slot.
        assert_eq!(ONESHOT_ARRIVAL_BYTES, 32);
        assert_eq!(ONESHOT_ROW_BYTES, 2_097_152);
        assert_eq!(ONESHOT_SLOT_BYTES, 32 + 4 * 2_097_152);
        assert_eq!(ONESHOT_SLOT_BYTES, 8_388_640);
        assert_eq!(
            row_offset(ONESHOT_RANKS - 1) + ONESHOT_ROW_BYTES,
            ONESHOT_SLOT_BYTES
        );
        assert_eq!(ONESHOT_SLOT_BYTES % ONESHOT_SLOT_ALIGN, 0);
    }

    #[test]
    fn arrival_words_come_first_and_rows_are_rank_ordered() {
        assert_eq!(arrival_offset(0), 0);
        assert_eq!(arrival_offset(3), 24);
        assert_eq!(row_offset(0), 32);
        assert_eq!(row_offset(1), 32 + 2_097_152);
        assert_eq!(row_offset(3), 32 + 3 * 2_097_152);
    }

    #[test]
    fn an_exchange_refuses_empty_oversized_and_bad_rail_counts() {
        assert!(matches!(
            Exchange::new(1, 0, 2),
            Err(SeamError::EmptyPayload)
        ));
        assert!(matches!(
            Exchange::new(1, ONESHOT_ROW_ELEMENTS + 1, 2),
            Err(SeamError::PayloadTooLarge { .. })
        ));
        assert!(matches!(
            Exchange::new(1, 8, 3),
            Err(SeamError::RailCount { rails: 3 })
        ));
        assert!(Exchange::new(1, ONESHOT_ROW_ELEMENTS, 2).is_ok());
        assert!(Exchange::new(1, 8, 1).is_ok());
    }

    #[test]
    fn the_rail_split_is_whole_elements_and_conserves_bytes() {
        let even = Exchange::new(1, 5120, 2).expect("even");
        assert_eq!(even.rail_bytes(0), 5120);
        assert_eq!(even.rail_bytes(1), 5120);
        assert_eq!(even.rail_bytes(0) + even.rail_bytes(1), even.bytes());
        assert_eq!(even.posted_bytes(), 3 * even.bytes());

        let odd = Exchange::new(1, 5, 2).expect("odd");
        assert_eq!(odd.rail_bytes(0), 6);
        assert_eq!(odd.rail_bytes(1), 4);
        assert_eq!(odd.rail_bytes(0) + odd.rail_bytes(1), odd.bytes());

        let single = Exchange::new(1, 5120, 1).expect("single rail");
        assert_eq!(single.rail_bytes(0), single.bytes());
        assert_eq!(single.rail_bytes(1), 0);
    }

    #[test]
    fn a_device_buffer_must_be_aligned_non_null_and_whole_bf16() {
        assert!(matches!(
            DeviceBuffer::new(0, 5120 * 2),
            Err(SeamError::NullAddress { .. })
        ));
        assert!(matches!(
            DeviceBuffer::new(0x1001, 16),
            Err(SeamError::Misaligned { .. })
        ));
        assert!(matches!(
            DeviceBuffer::new(0x1000, 0),
            Err(SeamError::Protocol(_))
        ));
        assert!(matches!(
            DeviceBuffer::new(0x1000, 15),
            Err(SeamError::Protocol(_))
        ));
        let buffer = DeviceBuffer::new(0x1000, 10_240).expect("device buffer");
        assert_eq!(buffer.elements(), 5120);
    }

    #[test]
    fn the_proxy_waits_for_both_rails_and_publishes_exactly_once() {
        let exchange = Exchange::new(7, 5120, 2).expect("exchange");
        let mut proxy = ArrivalProxy::new(0, exchange).expect("proxy");
        assert!(!proxy.all_published());

        let frame = |rail: usize, sender: u8| RailFrame {
            sender,
            rail,
            phase: ONESHOT_PHASE,
            kind: ONESHOT_KIND,
            sequence_low: 7,
        };

        // A frame before the doorbell is a protocol violation, not a silent drop.
        assert!(proxy.on_frame(&frame(0, 1)).is_err());
        proxy.on_doorbell(7).expect("doorbell");
        assert!(
            proxy.on_doorbell(7).is_err(),
            "a second doorbell is refused"
        );

        assert_eq!(proxy.on_frame(&frame(0, 1)).expect("rail 0"), None);
        assert_eq!(proxy.on_frame(&frame(1, 1)).expect("rail 1"), Some(1));
        assert_eq!(proxy.arrival(1), Some(7));
        assert!(!proxy.all_published());

        // Out-of-order and duplicated frames are tolerated for the first peer, but a peer is
        // published once and only once.
        assert!(proxy.on_frame(&frame(1, 1)).is_err());
        assert_eq!(
            proxy.published_peers(),
            vec![1],
            "exactly one peer published"
        );

        for (peer, rails) in [(2usize, [1usize, 0usize]), (3, [0, 1])] {
            let mut seen = 0;
            for rail in rails {
                if proxy
                    .on_frame(&frame(rail, peer as u8))
                    .expect("frame")
                    .is_some()
                {
                    seen += 1;
                }
            }
            assert_eq!(seen, 1, "peer {peer} published exactly once");
        }
        assert!(proxy.all_published());
        assert_eq!(proxy.published_peers(), vec![1, 2, 3]);
    }

    #[test]
    fn the_proxy_refuses_foreign_frames_and_wrong_sequence_bytes() {
        let exchange = Exchange::new(9, 5120, 2).expect("exchange");
        let mut proxy = ArrivalProxy::new(2, exchange).expect("proxy");
        proxy.on_doorbell(9).expect("doorbell");
        let base = RailFrame {
            sender: 0,
            rail: 0,
            phase: ONESHOT_PHASE,
            kind: ONESHOT_KIND,
            sequence_low: 9,
        };
        assert!(
            proxy.on_frame(&RailFrame { sender: 2, ..base }).is_err(),
            "a rank cannot accept a frame from itself"
        );
        assert!(proxy.on_frame(&RailFrame { sender: 4, ..base }).is_err());
        assert!(proxy.on_frame(&RailFrame { phase: 2, ..base }).is_err());
        assert!(proxy.on_frame(&RailFrame { kind: 2, ..base }).is_err());
        assert!(
            proxy
                .on_frame(&RailFrame {
                    sequence_low: 8,
                    ..base
                })
                .is_err()
        );
        assert!(proxy.on_frame(&RailFrame { rail: 2, ..base }).is_err());
        proxy.on_frame(&base).expect("the well-formed frame");
    }

    #[test]
    fn a_single_rail_exchange_publishes_after_one_frame() {
        let exchange = Exchange::new(3, 5120, 1).expect("exchange");
        let mut proxy = ArrivalProxy::new(1, exchange).expect("proxy");
        proxy.on_doorbell(3).expect("doorbell");
        let frame = RailFrame {
            sender: 0,
            rail: 0,
            phase: ONESHOT_PHASE,
            kind: ONESHOT_KIND,
            sequence_low: 3,
        };
        assert_eq!(proxy.on_frame(&frame).expect("frame"), Some(0));
        assert_eq!(proxy.arrival(0), Some(3));
    }

    #[test]
    fn the_slot_registration_enforces_the_abi_size_and_alignment() {
        assert!(matches!(
            SlotRegistration::new(
                SlotKind::DeviceMemory {
                    device_address: 0x1000
                },
                ONESHOT_SLOT_BYTES - 8
            ),
            Err(SeamError::SlotBytes { .. })
        ));
        assert!(matches!(
            SlotRegistration::new(
                SlotKind::PinnedHostAlias {
                    host_address: 0x1000,
                    device_alias: 0
                },
                ONESHOT_SLOT_BYTES
            ),
            Err(SeamError::NullAddress {
                what: "device alias"
            })
        ));
        assert!(matches!(
            SlotRegistration::new(
                SlotKind::PinnedHostAlias {
                    host_address: 0x1000,
                    device_alias: 0x1000
                },
                ONESHOT_SLOT_BYTES
            ),
            Err(SeamError::Protocol(_))
        ));
        let device = SlotRegistration::new(
            SlotKind::DeviceMemory {
                device_address: 0x2000,
            },
            ONESHOT_SLOT_BYTES,
        )
        .expect("device slot");
        assert_eq!(device.kernel_address(), 0x2000);
        let pinned = SlotRegistration::new(
            SlotKind::PinnedHostAlias {
                host_address: 0x4000,
                device_alias: 0x8000,
            },
            ONESHOT_SLOT_BYTES,
        )
        .expect("pinned slot");
        assert_eq!(pinned.kernel_address(), 0x8000);
    }

    #[test]
    fn a_dispatch_requires_a_real_rank_and_non_zero_words() {
        let exchange = Exchange::new(1, 5120, 2).expect("exchange");
        let slot = SlotRegistration::new(
            SlotKind::DeviceMemory {
                device_address: 0x2000,
            },
            ONESHOT_SLOT_BYTES,
        )
        .expect("slot");
        assert!(matches!(
            Dispatch::new(4, slot, exchange, 1, 1),
            Err(SeamError::RankOutOfRange { rank: 4 })
        ));
        assert!(matches!(
            Dispatch::new(0, slot, exchange, 0, 1),
            Err(SeamError::Protocol(_))
        ));
        assert!(matches!(
            Dispatch::new(0, slot, exchange, 1, 0),
            Err(SeamError::Protocol(_))
        ));
        let dispatch = Dispatch::new(2, slot, exchange, 5, 5).expect("dispatch");
        assert_eq!(dispatch.rank(), 2);
        assert_eq!(dispatch.release(), 5);
    }

    #[test]
    fn the_chunk_plan_cuts_the_4096_token_wo_b_seam_into_twenty_rows() {
        // hidden 5120 bf16 x 4096 tokens = 41,943,040 B = exactly 20 rows.
        let plan = ChunkPlan::new(5120 * 4096 * 2, ONESHOT_RAILS).expect("plan");
        assert_eq!(plan.chunk_count(), 20);
        assert_eq!(plan.total_elements(), 20_971_520);
        for (index, chunk) in plan.chunks().iter().enumerate() {
            assert_eq!(chunk.index(), index);
            assert_eq!(chunk.elements(), ONESHOT_ROW_ELEMENTS);
            assert_eq!(chunk.offset_bytes(), index * ONESHOT_ROW_BYTES);
        }
        assert_eq!(plan.posted_bytes().expect("posted"), 3 * 41_943_040);
    }

    #[test]
    fn the_chunk_plan_splits_a_ragged_tail_and_refuses_bad_sizes() {
        let plan = ChunkPlan::new(ONESHOT_ROW_BYTES * 2 + 10, ONESHOT_RAILS).expect("plan");
        assert_eq!(plan.chunk_count(), 3);
        assert_eq!(plan.chunks()[2].elements(), 5);
        let small = ChunkPlan::new(5120 * 2, 1).expect("one row");
        assert_eq!(small.chunk_count(), 1);
        assert_eq!(small.chunks()[0].elements(), 5120);
        assert!(matches!(
            ChunkPlan::new(0, ONESHOT_RAILS),
            Err(SeamError::EmptyPayload)
        ));
        assert!(matches!(
            ChunkPlan::new(3, ONESHOT_RAILS),
            Err(SeamError::OddPayload { bytes: 3 })
        ));
        assert!(matches!(
            ChunkPlan::new(1024, 3),
            Err(SeamError::RailCount { rails: 3 })
        ));
    }

    #[test]
    fn chunk_dispatches_are_ordered_and_share_the_slot() {
        let plan = ChunkPlan::new(3 * ONESHOT_ROW_BYTES, ONESHOT_RAILS).expect("plan");
        let slot = SlotRegistration::new(
            SlotKind::DeviceMemory {
                device_address: 0x1000,
            },
            ONESHOT_SLOT_BYTES,
        )
        .expect("slot");
        let dispatches = plan.dispatches(2, slot, 7, 9, 100).expect("dispatches");
        assert_eq!(dispatches.len(), 3);
        for (index, dispatch) in dispatches.iter().enumerate() {
            assert_eq!(dispatch.exchange().seq(), 100 + index as u64);
            assert_eq!(dispatch.slot(), slot);
            assert_eq!(dispatch.release(), 7);
            assert_eq!(dispatch.peer_epoch(), 9);
            assert_eq!(dispatch.rank(), 2);
        }
        let error = plan.exchange(3, 100).expect_err("chunk outside the plan");
        assert!(error.to_string().contains("outside a plan"), "{error}");
    }

    #[test]
    fn the_chunked_proxy_publishes_chunk_zero_then_advances_and_refuses_a_future_chunk() {
        let plan = ChunkPlan::new(ONESHOT_ROW_BYTES + 5120 * 2, ONESHOT_RAILS).expect("plan");
        let mut proxy = ChunkedProxy::new(1, plan, 40).expect("proxy");
        proxy.on_doorbell().expect("doorbell");
        // A frame from chunk 1 (sequence byte 41) before chunk 0 completes is refused.
        let future = RailFrame {
            sender: 0,
            rail: 0,
            phase: ONESHOT_PHASE,
            kind: ONESHOT_KIND,
            sequence_low: 41,
        };
        let error = proxy.on_frame(&future).expect_err("future chunk");
        assert!(error.to_string().contains("strict order"), "{error}");
        // Complete chunk 0: three peers on two rails each.
        for peer in [0u8, 2, 3] {
            for rail in 0..ONESHOT_RAILS {
                let frame = RailFrame {
                    sender: peer,
                    rail,
                    phase: ONESHOT_PHASE,
                    kind: ONESHOT_KIND,
                    sequence_low: 40,
                };
                proxy.on_frame(&frame).expect("frame");
            }
        }
        assert!(proxy.current_published());
        assert!(proxy.advance().expect("advance"));
        assert_eq!(proxy.current_index(), 1);
        assert_eq!(proxy.completed_chunks(), 1);
        proxy.on_doorbell().expect("doorbell 2");
        assert_eq!(proxy.current_exchange().seq(), 41);
        for peer in [0u8, 2, 3] {
            for rail in 0..ONESHOT_RAILS {
                let frame = RailFrame {
                    sender: peer,
                    rail,
                    phase: ONESHOT_PHASE,
                    kind: ONESHOT_KIND,
                    sequence_low: 41,
                };
                proxy.on_frame(&frame).expect("frame 2");
            }
        }
        assert!(!proxy.advance().expect("advance last"));
        assert!(proxy.is_finished());
        assert_eq!(proxy.completed_chunks(), 2);
        let after = RailFrame {
            sender: 0,
            rail: 0,
            phase: ONESHOT_PHASE,
            kind: ONESHOT_KIND,
            sequence_low: 41,
        };
        assert!(proxy.on_frame(&after).is_err());
    }

    fn device_record(seq: u64) -> DispatchRecord {
        let exchange = Exchange::new(seq, 5120, ONESHOT_RAILS).expect("exchange");
        let slot = 0x1000u64;
        DispatchRecord::new(
            1,
            seq,
            exchange,
            DispatchAddresses {
                slot,
                local_row: slot + row_offset(1) as u64,
                peer_rows: [0x2000, 0x3000, 0x4000],
            },
            LaunchPath::DeviceKernel,
            0,
            12_000,
        )
        .expect("device record")
    }

    #[test]
    fn a_device_dispatch_record_is_the_seam_shape_and_validates() {
        let record = device_record(0);
        assert_eq!(record.rank(), 1);
        assert_eq!(record.seq(), 0);
        assert_eq!(record.slot_address(), 0x1000);
        assert_eq!(record.local_row_address(), 0x1000 + row_offset(1) as u64);
        assert_eq!(record.peer_row_addresses(), [0x2000, 0x3000, 0x4000]);
        assert_eq!(record.launch(), LaunchPath::DeviceKernel);
        assert_eq!(record.host_copy_bytes(), 0);
        assert_eq!(record.elapsed_ns(), 12_000);
        record.device_path_ok().expect("a device path");
    }

    #[test]
    fn a_host_staging_record_is_recorded_and_then_refused() {
        let exchange = Exchange::new(0, 5120, ONESHOT_RAILS).expect("exchange");
        let slot = 0x1000u64;
        let record = DispatchRecord::new(
            0,
            0,
            exchange,
            DispatchAddresses {
                slot,
                local_row: slot + row_offset(0) as u64,
                peer_rows: [0x2000, 0x3000, 0x4000],
            },
            LaunchPath::HostStaging,
            10_240,
            50_000,
        )
        .expect("the record says what happened");
        assert!(matches!(
            record.device_path_ok(),
            Err(SeamError::HostFallback { seq: 0 })
        ));

        let mut log = DispatchLog::new();
        log.record(record)
            .expect("the fallback is recorded, not hidden");
        assert!(!log.device_path_only());
        assert_eq!(log.host_copy_bytes(), 10_240);
        assert!(matches!(
            log.device_path_ok(),
            Err(SeamError::HostFallback { .. })
        ));
    }

    #[test]
    fn a_device_record_with_any_host_copy_is_refused_even_if_it_claims_the_kernel() {
        let exchange = Exchange::new(0, 5120, ONESHOT_RAILS).expect("exchange");
        let slot = 0x1000u64;
        let record = DispatchRecord::new(
            2,
            0,
            exchange,
            DispatchAddresses {
                slot,
                local_row: slot + row_offset(2) as u64,
                peer_rows: [0x2000, 0x3000, 0x4000],
            },
            LaunchPath::DeviceKernel,
            2,
            1,
        )
        .expect("record");
        assert!(matches!(
            record.device_path_ok(),
            Err(SeamError::HostFallback { seq: 0 })
        ));
    }

    #[test]
    fn a_dispatch_record_must_describe_this_ranks_row_of_its_own_slot() {
        let exchange = Exchange::new(0, 5120, ONESHOT_RAILS).expect("exchange");
        let slot = 0x1000u64;
        assert!(matches!(
            DispatchRecord::new(
                1,
                0,
                exchange,
                DispatchAddresses {
                    slot,
                    local_row: slot + row_offset(2) as u64,
                    peer_rows: [0x2000, 0x3000, 0x4000],
                },
                LaunchPath::DeviceKernel,
                0,
                1,
            ),
            Err(SeamError::Protocol(_))
        ));
        assert!(matches!(
            DispatchRecord::new(
                0,
                0,
                exchange,
                DispatchAddresses {
                    slot: 0,
                    local_row: 0,
                    peer_rows: [0x2000, 0x3000, 0x4000],
                },
                LaunchPath::DeviceKernel,
                0,
                1,
            ),
            Err(SeamError::NullAddress {
                what: "dispatch slot"
            })
        ));
        assert!(matches!(
            DispatchRecord::new(
                0,
                0,
                exchange,
                DispatchAddresses {
                    slot,
                    local_row: slot + row_offset(0) as u64,
                    peer_rows: [0x2000, 0, 0x4000],
                },
                LaunchPath::DeviceKernel,
                0,
                1,
            ),
            Err(SeamError::NullAddress { what: "peer row" })
        ));
    }

    #[test]
    fn the_dispatch_log_keeps_sequence_order_and_sums_host_copies() {
        let mut log = DispatchLog::new();
        log.record(device_record(0)).expect("first");
        log.record(device_record(1)).expect("second");
        assert_eq!(log.len(), 2);
        assert!(log.device_path_only());
        assert_eq!(log.host_copy_bytes(), 0);
        assert!(matches!(
            log.record(device_record(1)),
            Err(SeamError::Sequence {
                previous: 1,
                requested: 1
            })
        ));
        assert_eq!(log.records().len(), 2);
    }

    #[test]
    fn the_sequence_counter_refuses_regressions_including_equal_values() {
        let mut counter = SequenceCounter::new();
        assert_eq!(counter.accept(0).expect("first"), 0);
        assert_eq!(counter.accept(1).expect("second"), 1);
        assert!(matches!(
            counter.accept(1),
            Err(SeamError::Sequence {
                previous: 1,
                requested: 1
            })
        ));
        assert!(matches!(
            counter.accept(0),
            Err(SeamError::Sequence {
                previous: 1,
                requested: 0
            })
        ));
        assert_eq!(counter.previous(), Some(1));
    }
}

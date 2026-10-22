//! Real-socket TCP transport used to prove the RoCEnante protocol without RDMA hardware.
//!
//! Each rank binds 127.0.0.1:(base_port + rank) and forms one connection per peer (lower ranks
//! connect upward, so every pair has exactly one socket). Payloads travel in framed messages:
//! a 24-byte little-endian header followed by the raw bf16 bytes. A reader thread per expected
//! frame drains the socket while the calling thread writes, which is what keeps a symmetric
//! 4-rank all-to-all from deadlocking on full socket buffers. When the whole cluster is on this
//! host (loopback peers), small exchanges instead read their peer frames sequentially after the
//! sends, which removes three thread handoffs per call; `SPARK_COMM_SEQUENTIAL_READ=1` / `=0`
//! forces that choice either way for an experiment.
//!
//! What this backend proves that the loopback cannot: framing and byte order on a real
//! byte stream, partial reads (the tests feed frames one byte at a time), peer death
//! (EOF/EPIPE become CommError::PeerClosed) and deadlines (a silent peer becomes
//! CommError::Timeout).

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::error::CommError;
use crate::exchange::{self, Exchange, ExchangeKind, ExchangeTag, Received};
use crate::layout::CollectiveLayout;
use crate::transport::Transport;

/// Frame magic ("SPCF" little-endian) that opens every header.
pub const FRAME_MAGIC: u32 = 0x5350_4346;
/// Framing version carried in every header.
pub const FRAME_VERSION: u16 = 1;
/// Fixed header size in bytes.
pub const FRAME_HEADER_BYTES: usize = 24;
/// Default cap on one frame payload, comfortably above the 2 MiB one-shot slots.
pub const DEFAULT_MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

/// What a frame carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// Connection setup: exchanges the sender's rank.
    Hello = 1,
    /// All-reduce payload: a raw contribution or a reduced chunk.
    ReducePayload = 2,
    /// All-gather payload: one rank's segment.
    GatherPayload = 3,
}

impl FrameKind {
    /// Wire byte for this kind.
    #[must_use]
    pub fn as_byte(self) -> u8 {
        self as u8
    }

    /// Parses the wire byte, rejecting anything unknown.
    ///
    /// # Errors
    /// Returns CommError::Protocol for an unknown kind.
    pub fn from_byte(byte: u8) -> Result<Self, CommError> {
        match byte {
            1 => Ok(Self::Hello),
            2 => Ok(Self::ReducePayload),
            3 => Ok(Self::GatherPayload),
            other => Err(CommError::Protocol(format!("unknown frame kind {other}"))),
        }
    }
}

/// Fixed 24-byte frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    /// What the frame carries.
    pub kind: FrameKind,
    /// Collective phase (0, or 1 for the hairpin all-gather).
    pub phase: u8,
    /// Caller sequence number, identical on every rank.
    pub seq: u64,
    /// Sending rank.
    pub rank: u32,
    /// Payload length in bytes.
    pub payload_bytes: u32,
}

impl FrameHeader {
    /// Encodes the header: magic, version, kind, phase, seq, rank, payload length.
    #[must_use]
    pub fn encode(&self) -> [u8; FRAME_HEADER_BYTES] {
        let mut out = [0u8; FRAME_HEADER_BYTES];
        out[0..4].copy_from_slice(&FRAME_MAGIC.to_le_bytes());
        out[4..6].copy_from_slice(&FRAME_VERSION.to_le_bytes());
        out[6] = self.kind.as_byte();
        out[7] = self.phase;
        out[8..16].copy_from_slice(&self.seq.to_le_bytes());
        out[16..20].copy_from_slice(&self.rank.to_le_bytes());
        out[20..24].copy_from_slice(&self.payload_bytes.to_le_bytes());
        out
    }

    /// Decodes a header, validating magic and version.
    ///
    /// # Errors
    /// Returns CommError::Protocol for a short buffer, wrong magic or wrong version.
    pub fn decode(bytes: &[u8]) -> Result<Self, CommError> {
        if bytes.len() < FRAME_HEADER_BYTES {
            return Err(CommError::Protocol(format!(
                "frame header needs {FRAME_HEADER_BYTES} bytes, got {}",
                bytes.len()
            )));
        }
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if magic != FRAME_MAGIC {
            return Err(CommError::Protocol(format!(
                "bad frame magic 0x{magic:08x}, expected 0x{FRAME_MAGIC:08x}"
            )));
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != FRAME_VERSION {
            return Err(CommError::Protocol(format!(
                "unsupported framing version {version}, expected {FRAME_VERSION}"
            )));
        }
        Ok(Self {
            kind: FrameKind::from_byte(bytes[6])?,
            phase: bytes[7],
            seq: u64::from_le_bytes([
                bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14],
                bytes[15],
            ]),
            rank: u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
            payload_bytes: u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]),
        })
    }
}

/// Writes a header and its payload.
///
/// # Errors
/// Returns CommError::Protocol when the declared length disagrees with the payload, and
/// CommError::Io for socket failures.
pub fn write_frame<W: Write>(
    writer: &mut W,
    header: &FrameHeader,
    payload: &[u8],
) -> Result<(), CommError> {
    if payload.len() != header.payload_bytes as usize {
        return Err(CommError::Protocol(format!(
            "header declares {} payload bytes but {} were supplied",
            header.payload_bytes,
            payload.len()
        )));
    }
    writer.write_all(&header.encode())?;
    writer.write_all(payload)?;
    writer.flush()?;
    Ok(())
}

/// Reads one header and its payload.
///
/// Read errors keep their CommError::Io shape; callers that know the peer map them with
/// CommError::from_io_peer so an EOF becomes PeerClosed and a read timeout becomes Timeout.
///
/// # Errors
/// Returns CommError::Protocol for a malformed header or a payload above max_payload_bytes and
/// CommError::Io for stream failures.
pub fn read_frame<R: Read>(
    reader: &mut R,
    max_payload_bytes: usize,
) -> Result<(FrameHeader, Vec<u8>), CommError> {
    let mut header_bytes = [0u8; FRAME_HEADER_BYTES];
    reader.read_exact(&mut header_bytes)?;
    let header = FrameHeader::decode(&header_bytes)?;
    let length = header.payload_bytes as usize;
    if length > max_payload_bytes {
        return Err(CommError::Protocol(format!(
            "frame of {length} payload bytes exceeds the {max_payload_bytes} byte limit"
        )));
    }
    let mut payload = vec![0u8; length];
    reader.read_exact(&mut payload)?;
    Ok((header, payload))
}

/// Encodes bf16 elements as little-endian byte pairs.
#[must_use]
pub fn encode_bf16(values: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// Decodes little-endian byte pairs into bf16 elements.
///
/// # Errors
/// Returns CommError::Protocol for an odd byte count.
pub fn decode_bf16(bytes: &[u8]) -> Result<Vec<u16>, CommError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(CommError::Protocol(format!(
            "bf16 payload has an odd byte count ({})",
            bytes.len()
        )));
    }
    let (pairs, remainder) = bytes.as_chunks::<2>();
    debug_assert!(remainder.is_empty());
    Ok(pairs.iter().map(|pair| u16::from_le_bytes(*pair)).collect())
}

/// Configuration of one rank of a TCP RoCEnante cluster.
///
/// Two addressing modes share one protocol. The loopback mode (implemented by
/// [TcpConfig::loopback4] and the integration tests) derives every address from
/// `127.0.0.1:(base_port + rank)`. The cluster mode (implemented by
/// [TcpConfig::with_addresses]) takes an explicit per-rank address table, so the same
/// [TcpTransport] runs on the four nodes: rank `r` binds `peer_addrs[r]` and dials
/// `peer_addrs[peer]`.
#[derive(Debug, Clone)]
pub struct TcpConfig {
    /// Rank this transport serves.
    pub rank: usize,
    /// Number of ranks in the cluster.
    pub world_size: usize,
    /// Port of rank 0; rank r binds base_port + r on 127.0.0.1.
    ///
    /// Ignored when [TcpConfig::peer_addrs] is set: the address table then supplies every
    /// address, including this rank's listen address.
    pub base_port: u16,
    /// Read/write deadline for every frame.
    pub timeout: Duration,
    /// Deadline for connection setup.
    pub connect_timeout: Duration,
    /// Largest single frame payload accepted.
    pub max_frame_bytes: usize,
    /// Explicit listen address for this rank.
    ///
    /// This is separate from [TcpConfig::peer_addrs] so a rank may bind a wildcard or a
    /// specific interface (`0.0.0.0:29601`, `192.0.2.80:29601`) while its peers dial a
    /// routable address for the same port. When `None`, this rank binds
    /// `127.0.0.1:(base_port + rank)`.
    pub bind_addr: Option<SocketAddr>,
    /// Explicit per-rank dial address table, indexed by rank.
    ///
    /// When `Some`, it must hold exactly [TcpConfig::world_size] addresses: a cluster that
    /// describes three of its four ranks is refused at connect instead of silently dialing
    /// loopback for the missing one. When `None`, peer `p` is dialed at
    /// `127.0.0.1:(base_port + p)`.
    pub peer_addrs: Option<Vec<SocketAddr>>,
}

impl TcpConfig {
    /// The four-rank 127.0.0.1 configuration used by the integration tests.
    #[must_use]
    pub fn loopback4(rank: usize, base_port: u16) -> Self {
        Self {
            rank,
            world_size: 4,
            base_port,
            timeout: Duration::from_secs(5),
            connect_timeout: Duration::from_secs(5),
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
            bind_addr: None,
            peer_addrs: None,
        }
    }

    /// The cluster configuration: one explicit listen address per rank.
    ///
    /// `addresses[r]` is what rank `r` binds and what every peer dials. The address table is
    /// the whole cluster description, so an address that is unreachable is a connection
    /// timeout naming the peer, not a silent fallback to loopback.
    ///
    /// # Errors
    /// [CommError::RankOutOfRange] when `rank` is not an index into `addresses`,
    /// [CommError::InvalidLayout] when the table is empty.
    pub fn with_addresses(rank: usize, addresses: Vec<SocketAddr>) -> Result<Self, CommError> {
        if addresses.is_empty() {
            return Err(CommError::InvalidLayout(
                "a TCP cluster address table needs at least one address".to_owned(),
            ));
        }
        if rank >= addresses.len() {
            return Err(CommError::RankOutOfRange {
                rank,
                world_size: addresses.len(),
            });
        }
        Ok(Self {
            rank,
            world_size: addresses.len(),
            base_port: 0,
            timeout: Duration::from_secs(5),
            connect_timeout: Duration::from_secs(5),
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
            bind_addr: Some(addresses[rank]),
            peer_addrs: Some(addresses),
        })
    }

    /// Replaces the frame deadline.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Replaces the connection-setup deadline.
    #[must_use]
    pub fn with_connect_timeout(mut self, connect_timeout: Duration) -> Self {
        self.connect_timeout = connect_timeout;
        self
    }

    /// This rank's listen address: the explicit bind address, else `127.0.0.1:(base + rank)`.
    fn local_address(&self) -> SocketAddr {
        self.bind_addr.unwrap_or_else(|| {
            SocketAddr::from((
                [127, 0, 0, 1],
                (u32::from(self.base_port) + self.rank as u32) as u16,
            ))
        })
    }

    /// Peer `peer`'s dial address, from the table when present and from the loopback scheme
    /// otherwise.
    ///
    /// # Errors
    /// [CommError::InvalidLayout] when the table is present but does not cover the cluster:
    /// a partial address table is a configuration error, never a loopback fallback.
    fn peer_address(&self, peer: usize) -> Result<SocketAddr, CommError> {
        match &self.peer_addrs {
            Some(table) => {
                if table.len() != self.world_size {
                    return Err(CommError::InvalidLayout(format!(
                        "the address table holds {} addresses but world_size is {}",
                        table.len(),
                        self.world_size
                    )));
                }
                table.get(peer).copied().ok_or(CommError::RankOutOfRange {
                    rank: peer,
                    world_size: self.world_size,
                })
            }
            None => Ok(SocketAddr::from((
                [127, 0, 0, 1],
                (u32::from(self.base_port) + peer as u32) as u16,
            ))),
        }
    }
}

/// The (layer, phase) slot a posted reduce occupies.
///
/// The caller names the slot, not the transport: layer 12 phase 0 is one slot, and a second post
/// into it before the first is waited is refused. The phase is a LOCAL seam identity (for the loop,
/// 0 = attention consumption and 1 = MoE seam); it is not written on the wire, because frame
/// demultiplexing is by `seq` alone. Consequently `(layer, 0)` and `(layer, 1)` may both be
/// outstanding at the same time as long as their sequence numbers differ, while two outstanding
/// posts into the same `(layer, phase)` remain a hard error.
///
/// The asynchronous path is one-shot only; a hairpin's second leg stays on the synchronous API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReduceSlot {
    /// Chain layer whose seam owns the reduction.
    pub layer: u32,
    /// Phase inside that layer (0 for the MoE-seam reduce).
    pub phase: u8,
}

impl ReduceSlot {
    /// Builds a slot identity.
    #[must_use]
    pub fn new(layer: u32, phase: u8) -> Self {
        Self { layer, phase }
    }
}

/// A reduce whose wire exchange has been posted and not yet consumed.
///
/// The handle owns the caller's buffer mutably until [`TcpTransport::wait_reduce`] consumes it, so
/// the contribution cannot be overwritten between post and wait. It is deliberately not `Copy`
/// and not `Clone`: passing it to `wait_reduce` is the only way to complete it, which is what makes
/// a double wait a compile error and a stale handle a runtime error.
#[must_use = "a posted reduce must be waited exactly once; dropping the handle strands the slot"]
pub struct ReduceHandle<'a> {
    slot: ReduceSlot,
    seq: u64,
    buffer: &'a mut [u16],
}

impl ReduceHandle<'_> {
    /// The slot this reduce was posted into.
    #[must_use]
    pub fn slot(&self) -> ReduceSlot {
        self.slot
    }

    /// The caller sequence number carried on the wire.
    #[must_use]
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Number of bf16 elements this reduce covers.
    #[must_use]
    pub fn elements(&self) -> usize {
        self.buffer.len()
    }
}

/// Frames already decoded and waiting for their reduce's registered slot.
struct PendingReduce {
    slot: ReduceSlot,
    elements: usize,
    received: HashMap<usize, Vec<u16>>,
}

/// Shared state between the transport and its per-peer async reader threads.
struct AsyncShared {
    inner: Mutex<AsyncInner>,
    condvar: Condvar,
    running: AtomicBool,
}

struct AsyncInner {
    pending: HashMap<u64, PendingReduce>,
    slots: HashMap<ReduceSlot, u64>,
    /// A frame can arrive before this rank posts the matching reduce (the peers post independently),
    /// so unknown sequences are held here and adopted by `register` rather than rejected.
    orphan: HashMap<u64, Vec<(usize, Vec<u16>)>>,
    failure: Option<CommError>,
}

impl AsyncShared {
    fn new() -> Self {
        Self {
            inner: Mutex::new(AsyncInner {
                pending: HashMap::new(),
                slots: HashMap::new(),
                orphan: HashMap::new(),
                failure: None,
            }),
            condvar: Condvar::new(),
            running: AtomicBool::new(true),
        }
    }

    fn fail(&self, error: CommError) {
        let mut inner = self.inner.lock();
        if inner.failure.is_none() {
            inner.failure = Some(error);
        }
        self.condvar.notify_all();
    }

    fn deliver(&self, peer: usize, seq: u64, payload: Vec<u16>) {
        let mut inner = self.inner.lock();
        if let Some(pending) = inner.pending.get_mut(&seq) {
            if pending.received.insert(peer, payload).is_some() && inner.failure.is_none() {
                inner.failure = Some(CommError::Protocol(format!(
                    "peer {peer} sent two payloads for reduce seq {seq}"
                )));
            }
        } else {
            inner.orphan.entry(seq).or_default().push((peer, payload));
        }
        self.condvar.notify_all();
    }

    fn register(&self, slot: ReduceSlot, seq: u64, elements: usize) -> Result<(), CommError> {
        let mut inner = self.inner.lock();
        if let Some(existing) = inner.slots.get(&slot) {
            return Err(CommError::Protocol(format!(
                "reduce slot {slot:?} already has seq {existing} outstanding; one reduce per (layer, phase) slot"
            )));
        }
        if inner.pending.contains_key(&seq) {
            return Err(CommError::Protocol(format!(
                "reduce seq {seq} is already outstanding; sequence numbers must be unique"
            )));
        }
        let mut pending = PendingReduce {
            slot,
            elements,
            received: HashMap::new(),
        };
        if let Some(orphans) = inner.orphan.remove(&seq) {
            for (peer, payload) in orphans {
                pending.received.insert(peer, payload);
            }
        }
        inner.slots.insert(slot, seq);
        inner.pending.insert(seq, pending);
        Ok(())
    }

    fn abort(&self, slot: ReduceSlot, seq: u64) {
        let mut inner = self.inner.lock();
        if inner.slots.get(&slot).copied() == Some(seq) {
            inner.slots.remove(&slot);
        }
        inner.pending.remove(&seq);
        self.condvar.notify_all();
    }

    fn take(
        &self,
        slot: ReduceSlot,
        seq: u64,
        world_size: usize,
        rank: usize,
        timeout: Duration,
    ) -> Result<Vec<(usize, Vec<u16>)>, CommError> {
        let deadline = Instant::now() + timeout;
        let mut inner = self.inner.lock();
        loop {
            if let Some(error) = inner.failure.take() {
                inner.pending.remove(&seq);
                if inner.slots.get(&slot).copied() == Some(seq) {
                    inner.slots.remove(&slot);
                }
                return Err(error);
            }
            let complete = match inner.pending.get(&seq) {
                Some(pending) if pending.slot == slot => pending.received.len() == world_size - 1,
                Some(_) => {
                    return Err(CommError::Protocol(format!(
                        "reduce seq {seq} belongs to a different slot"
                    )));
                }
                None => {
                    return Err(CommError::Protocol(format!(
                        "reduce slot {slot:?} seq {seq} has no posted exchange (dead wait or already consumed)"
                    )));
                }
            };
            if complete {
                let pending = inner
                    .pending
                    .remove(&seq)
                    .ok_or_else(|| CommError::Protocol("pending reduce vanished".to_owned()))?;
                inner.slots.remove(&slot);
                let mut frames: Vec<(usize, Vec<u16>)> = Vec::with_capacity(world_size - 1);
                for (peer, payload) in pending.received {
                    if peer == rank || peer >= world_size {
                        return Err(CommError::Protocol(format!(
                            "reduce seq {seq} received a payload from invalid peer {peer}"
                        )));
                    }
                    if payload.len() != pending.elements {
                        return Err(CommError::Protocol(format!(
                            "reduce seq {seq} peer {peer} carried {} elements, expected {}",
                            payload.len(),
                            pending.elements
                        )));
                    }
                    frames.push((peer, payload));
                }
                frames.sort_by_key(|(peer, _)| *peer);
                return Ok(frames);
            }
            let now = Instant::now();
            if now >= deadline {
                outer_cleanup(&mut inner, slot, seq);
                return Err(CommError::timeout(
                    timeout,
                    format!("waiting for reduce slot {slot:?} seq {seq}"),
                ));
            }
            self.condvar.wait_for(&mut inner, deadline - now);
        }
    }
}

/// Removes a slot's pending state after a timeout; kept out of the loop body so the guard is
/// released the same way on every path.
fn outer_cleanup(inner: &mut AsyncInner, slot: ReduceSlot, seq: u64) {
    inner.pending.remove(&seq);
    if inner.slots.get(&slot).copied() == Some(seq) {
        inner.slots.remove(&slot);
    }
}

/// The async reader thread for one peer: frames are demultiplexed by seq into the shared state.
fn async_reader(
    peer: usize,
    mut stream: TcpStream,
    timeout: Duration,
    max_frame_bytes: usize,
    shared: Arc<AsyncShared>,
) {
    // A short read timeout is used only so the loop can observe `running`; the parser keeps the
    // partial header/payload state across timeouts instead of restarting read_frame. A timeout in
    // the middle of a frame must never throw the already-read bytes away (that was the first bug
    // this loop had), and a blocking read would strand the thread on Drop.
    let _ = stream.set_read_timeout(Some(Duration::from_millis(50)));
    let mut header_bytes = [0u8; FRAME_HEADER_BYTES];
    let mut header_filled = 0usize;
    let mut payload: Vec<u8> = Vec::new();
    let mut payload_filled = 0usize;
    let mut have_header = false;
    let mut pending_seq = 0u64;
    let fail = |error: CommError| {
        if shared.running.load(Ordering::Acquire) {
            shared.fail(error);
        }
    };
    'reader: while shared.running.load(Ordering::Acquire) {
        let outcome = if have_header {
            stream.read(&mut payload[payload_filled..])
        } else {
            stream.read(&mut header_bytes[header_filled..])
        };
        let filled = match outcome {
            Ok(0) => {
                fail(map_peer_error(
                    peer,
                    timeout,
                    "async reader",
                    CommError::Io(io::Error::from(io::ErrorKind::UnexpectedEof)),
                ));
                break 'reader;
            }
            Ok(count) => count,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                continue 'reader;
            }
            Err(error) => {
                fail(map_peer_error(
                    peer,
                    timeout,
                    "async reader",
                    CommError::Io(error),
                ));
                break 'reader;
            }
        };
        if have_header {
            payload_filled += filled;
            if payload_filled != payload.len() {
                continue;
            }
            match decode_bf16(&payload) {
                Ok(values) => shared.deliver(peer, pending_seq, values),
                Err(error) => {
                    fail(error);
                    break 'reader;
                }
            }
            have_header = false;
            header_filled = 0;
            payload.clear();
            payload_filled = 0;
        } else {
            header_filled += filled;
            if header_filled != FRAME_HEADER_BYTES {
                continue;
            }
            let header = match FrameHeader::decode(&header_bytes) {
                Ok(header) => header,
                Err(error) => {
                    fail(error);
                    break 'reader;
                }
            };
            if header.kind != FrameKind::ReducePayload || header.phase != 0 {
                fail(CommError::Protocol(format!(
                    "async reader for peer {peer} saw an unexpected frame: {header:?}"
                )));
                break 'reader;
            }
            if header.rank as usize != peer {
                fail(CommError::Protocol(format!(
                    "async reader for peer {peer} saw rank {} in the header",
                    header.rank
                )));
                break 'reader;
            }
            let length = header.payload_bytes as usize;
            if length > max_frame_bytes {
                fail(CommError::Protocol(format!(
                    "async frame from peer {peer} of {length} bytes exceeds the {max_frame_bytes} limit"
                )));
                break 'reader;
            }
            pending_seq = header.seq;
            payload.resize(length, 0);
            payload_filled = 0;
            have_header = true;
        }
    }
}

/// One rank's TCP transport.
pub struct TcpTransport {
    rank: usize,
    world_size: usize,
    timeout: Duration,
    max_frame_bytes: usize,
    readers: Vec<Option<TcpStream>>,
    writers: Vec<Option<TcpStream>>,
    /// Set by the first `post_reduce_bf16`; starts one demultiplexing reader per peer.
    async_shared: Option<Arc<AsyncShared>>,
    async_readers: Vec<JoinHandle<()>>,
    /// Whether small exchanges read peer frames sequentially after the sends instead of spawning
    /// one reader thread per peer. See [SEQUENTIAL_READ_MAX_BYTES].
    sequential_read: bool,
}

impl TcpTransport {
    /// Binds this rank's listener and forms one connection to every peer.
    ///
    /// Lower ranks connect upward, so each pair has exactly one connection; both directions
    /// share the socket. The call blocks until the whole cluster has connected, which is what
    /// makes the returned handle safe to use without any further barrier.
    ///
    /// # Errors
    /// Returns CommError::RankOutOfRange, CommError::InvalidLayout for an impossible port range,
    /// CommError::Timeout when a peer fails to appear, and CommError::Protocol when a peer
    /// presents a bad hello frame.
    pub fn connect(config: TcpConfig) -> Result<Self, CommError> {
        let world_size = config.world_size;
        if world_size == 0 {
            return Err(CommError::InvalidLayout(
                "a TCP cluster needs at least one rank".to_owned(),
            ));
        }
        if config.rank >= world_size {
            return Err(CommError::RankOutOfRange {
                rank: config.rank,
                world_size,
            });
        }
        let last_port =
            u32::from(config.base_port) + u32::try_from(world_size - 1).unwrap_or(u32::MAX);
        if last_port > u32::from(u16::MAX) {
            return Err(CommError::InvalidLayout(format!(
                "base_port {} does not have room for {world_size} ranks",
                config.base_port
            )));
        }
        // A partial address table is a configuration error, never a loopback fallback for the
        // ranks it leaves out: checked once here so it fails before any socket is opened.
        if let Some(table) = &config.peer_addrs
            && table.len() != world_size
        {
            return Err(CommError::InvalidLayout(format!(
                "the address table holds {} addresses but world_size is {world_size}",
                table.len()
            )));
        }
        let local_address = config.local_address();
        let listener = TcpListener::bind(local_address)?;
        listener.set_nonblocking(true)?;
        let deadline = Instant::now() + config.connect_timeout;
        let mut readers: Vec<Option<TcpStream>> = (0..world_size).map(|_| None).collect();
        let mut writers: Vec<Option<TcpStream>> = (0..world_size).map(|_| None).collect();

        // Lower ranks connect upward, and the kernel decides which of them lands in the accept
        // queue first, so an accepted socket is identified by the hello frame it carries rather
        // than by the order it arrived in.
        for _ in 0..config.rank {
            let mut reader = accept_retry(&listener, deadline, config.connect_timeout)?;
            tune(&reader, config.timeout)?;
            let (hello, payload) =
                read_frame(&mut reader, config.max_frame_bytes).map_err(|error| {
                    CommError::Protocol(format!(
                        "an accepted connection failed during the hello handshake: {error}"
                    ))
                })?;
            if hello.kind != FrameKind::Hello || hello.phase != 0 || !payload.is_empty() {
                return Err(CommError::Protocol(format!(
                    "malformed hello on an accepted connection: {hello:?}"
                )));
            }
            let peer = hello.rank as usize;
            if peer >= config.rank {
                return Err(CommError::Protocol(format!(
                    "rank {} accepted an upward connection claiming to come from rank {peer}",
                    config.rank
                )));
            }
            if readers[peer].is_some() {
                return Err(CommError::Protocol(format!(
                    "rank {peer} connected to rank {} twice",
                    config.rank
                )));
            }
            let mut writer = reader.try_clone()?;
            write_hello(&mut writer, config.rank)?;
            readers[peer] = Some(reader);
            writers[peer] = Some(writer);
        }
        for peer in config.rank + 1..world_size {
            let address = config.peer_address(peer)?;
            let mut writer = connect_retry(address, deadline, config.connect_timeout)?;
            tune(&writer, config.timeout)?;
            write_hello(&mut writer, config.rank)?;
            let (hello, payload) =
                read_frame(&mut writer, config.max_frame_bytes).map_err(|error| {
                    map_peer_error(peer, config.timeout, "reading a hello frame", error)
                })?;
            if hello.kind != FrameKind::Hello
                || hello.phase != 0
                || hello.rank as usize != peer
                || !payload.is_empty()
            {
                return Err(CommError::Protocol(format!(
                    "unexpected hello from rank {peer}: {hello:?}"
                )));
            }
            let reader = writer.try_clone()?;
            readers[peer] = Some(reader);
            writers[peer] = Some(writer);
        }
        for (peer, slot) in readers.iter().enumerate().take(config.rank) {
            if slot.is_none() {
                return Err(CommError::Protocol(format!(
                    "rank {} never heard from lower rank {peer}",
                    config.rank
                )));
            }
        }
        // The sequential read is a measured win only where the peers share this host (loopback,
        // no switch round trip). On a real cluster it is opt-in via SPARK_COMM_SEQUENTIAL_READ,
        // so the default row keeps the reader threads that overlapping sends with waits relies on.
        let sequential_read =
            sequential_read_override().unwrap_or_else(|| cluster_is_loopback(&config));
        Ok(Self {
            rank: config.rank,
            world_size,
            timeout: config.timeout,
            max_frame_bytes: config.max_frame_bytes,
            readers,
            writers,
            async_shared: None,
            async_readers: Vec::new(),
            sequential_read,
        })
    }

    /// Rank this handle belongs to.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Number of ranks in the cluster.
    #[must_use]
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Closes every peer socket. Used by the peer-death tests to model a rank that crashes.
    pub fn shutdown(&self) {
        if let Some(shared) = &self.async_shared {
            shared.running.store(false, Ordering::Release);
        }
        for stream in self.readers.iter().chain(self.writers.iter()).flatten() {
            let _ = stream.shutdown(Shutdown::Both);
        }
    }

    /// Starts the one demultiplexing reader per peer the first time an async reduce is posted.
    fn ensure_async(&mut self) -> Result<Arc<AsyncShared>, CommError> {
        if let Some(shared) = &self.async_shared {
            return Ok(Arc::clone(shared));
        }
        let shared = Arc::new(AsyncShared::new());
        let mut handles = Vec::with_capacity(self.world_size.saturating_sub(1));
        for peer in 0..self.world_size {
            if peer == self.rank {
                continue;
            }
            let stream = self.read_handle(peer)?.try_clone()?;
            let state = Arc::clone(&shared);
            let timeout = self.timeout;
            let max_frame_bytes = self.max_frame_bytes;
            handles.push(std::thread::spawn(move || {
                async_reader(peer, stream, timeout, max_frame_bytes, state);
            }));
        }
        self.async_readers = handles;
        self.async_shared = Some(Arc::clone(&shared));
        Ok(shared)
    }

    /// Posts a one-shot all-reduce without waiting for the peer payloads.
    ///
    /// The wire exchange starts inside this call: the contribution is framed and written to every
    /// peer, and the per-peer demultiplexing readers (started on the first post) collect replies by
    /// sequence number. The canonical reduction is deliberately NOT applied here; it is applied by
    /// [`Self::wait_reduce`] at the consumption point, so a posted reduce can stay in flight while
    /// the caller does other work. The result is the same canonical rule as the synchronous
    /// one-shot path: bf16 payloads, ascending-rank f32 sum from 0.0, one bf16 RNE round.
    ///
    /// The asynchronous path is one-shot only (`FabricSchedule::OneShotAllToAll`). A hairpin
    /// layout's second leg is a different schedule and stays on the synchronous API; this is
    /// stated here rather than silently reducing with the wrong transfer list. The frame's wire
    /// phase is always 0; the [`ReduceSlot::phase`] is local-only and is what the per-slot
    /// outstanding map keys on.
    ///
    /// Exactly-once contract: the returned handle owns the buffer until `wait_reduce`, a second
    /// post into the same slot is a hard error, and a wait with no outstanding slot is a hard
    /// error. Sequence numbers must be unique among outstanding reduces.
    ///
    /// # Errors
    /// Returns CommError::InvalidLayout for an empty/oversized buffer, CommError::Protocol for a
    /// reused slot or sequence number, and a peer/socket error if a frame cannot be written.
    pub fn post_reduce_bf16<'a>(
        &mut self,
        slot: ReduceSlot,
        seq: u64,
        buffer: &'a mut [u16],
    ) -> Result<ReduceHandle<'a>, CommError> {
        if buffer.is_empty() {
            return Err(CommError::InvalidLayout(
                "a posted reduce needs at least one element".to_owned(),
            ));
        }
        self.check_frame(buffer.len())?;
        let payload_bytes = u32::try_from(buffer.len() * 2).map_err(|_| {
            CommError::InvalidLayout(format!(
                "{} elements cannot be described by a u32 frame length",
                buffer.len()
            ))
        })?;
        let payload = encode_bf16(buffer);
        let shared = self.ensure_async()?;
        shared.register(slot, seq, buffer.len())?;
        let header = FrameHeader {
            kind: FrameKind::ReducePayload,
            phase: 0,
            seq,
            rank: self.rank as u32,
            payload_bytes,
        };
        // One write per peer: the header is prepended to the encoded payload once and the same
        // frame buffer is written to every peer. Posting is on the critical path of the issue half
        // of an overlapped reduce, so the extra 24-byte copy is cheaper than three more syscalls.
        let mut frame = Vec::with_capacity(FRAME_HEADER_BYTES + payload.len());
        frame.extend_from_slice(&header.encode());
        frame.extend_from_slice(&payload);
        for peer in 0..self.world_size {
            if peer == self.rank {
                continue;
            }
            let mut writer = match self.write_handle(peer) {
                Ok(writer) => writer,
                Err(error) => {
                    shared.abort(slot, seq);
                    return Err(error);
                }
            };
            if let Err(error) = writer.write_all(&frame).and_then(|()| writer.flush()) {
                shared.abort(slot, seq);
                return Err(map_peer_error(
                    peer,
                    self.timeout,
                    "posting a reduce frame",
                    CommError::Io(error),
                ));
            }
        }
        Ok(ReduceHandle { slot, seq, buffer })
    }

    /// Waits for a posted reduce and applies the canonical rule in place at the consumption point.
    ///
    /// # Errors
    /// Returns CommError::Timeout when a peer payload misses the transport deadline, a peer error
    /// from the demultiplexer, or CommError::Protocol for a dead/double wait (a handle whose slot
    /// has no outstanding exchange) or a malformed peer payload.
    pub fn wait_reduce<'a>(
        &mut self,
        handle: ReduceHandle<'a>,
    ) -> Result<&'a mut [u16], CommError> {
        let shared = self.async_shared.as_ref().ok_or_else(|| {
            CommError::Protocol("wait_reduce called before any post_reduce_bf16".to_owned())
        })?;
        let ReduceHandle { slot, seq, buffer } = handle;
        let frames = shared.take(slot, seq, self.world_size, self.rank, self.timeout)?;
        let own = buffer.to_vec();
        let mut panes: Vec<&[u16]> = vec![&[]; self.world_size];
        panes[self.rank] = own.as_slice();
        for (peer, payload) in &frames {
            panes[*peer] = payload.as_slice();
        }
        crate::reduce::canonical_all_reduce_into(buffer, &panes)?;
        Ok(buffer)
    }

    fn read_handle(&self, peer: usize) -> Result<&TcpStream, CommError> {
        self.readers
            .get(peer)
            .and_then(|stream| stream.as_ref())
            .ok_or(CommError::RankOutOfRange {
                rank: peer,
                world_size: self.world_size,
            })
    }

    fn write_handle(&self, peer: usize) -> Result<&TcpStream, CommError> {
        self.writers
            .get(peer)
            .and_then(|stream| stream.as_ref())
            .ok_or(CommError::RankOutOfRange {
                rank: peer,
                world_size: self.world_size,
            })
    }

    fn check_frame(&self, elements: usize) -> Result<(), CommError> {
        let bytes = elements.checked_mul(2).ok_or_else(|| {
            CommError::InvalidLayout(format!("{elements} elements overflow a byte count"))
        })?;
        if bytes > self.max_frame_bytes {
            return Err(CommError::InvalidLayout(format!(
                "frame of {bytes} bytes exceeds the {} byte limit",
                self.max_frame_bytes
            )));
        }
        Ok(())
    }
}

impl Drop for TcpTransport {
    fn drop(&mut self) {
        // Do NOT shutdown the sockets here: a rank that has consumed its last reduce may drop
        // before its peers have read frames already queued on the wire, and shutdown(Both) can
        // discard them. Settling `running` lets the readers exit after their next 50 ms timeout;
        // the socket close that follows flushes the send queue.
        if let Some(shared) = &self.async_shared {
            shared.running.store(false, Ordering::Release);
        }
    }
}

impl Transport for TcpTransport {
    fn world_size(&self) -> usize {
        self.world_size
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn all_reduce_bf16(
        &self,
        buf: &mut [u16],
        seq: u64,
        layout: &CollectiveLayout,
    ) -> Result<(), CommError> {
        if self.async_shared.is_some() {
            return Err(CommError::Protocol(
                "synchronous all_reduce after post_reduce_bf16: the async readers own the sockets; \
                 use wait_reduce, or start a new transport"
                    .to_owned(),
            ));
        }
        self.check_frame(buf.len())?;
        let mut exchange = TcpExchange::new(self);
        exchange::all_reduce(&mut exchange, self.rank, buf, seq, layout)
    }

    fn all_gather_bf16(&self, local: &[u16], out: &mut [u16], seq: u64) -> Result<(), CommError> {
        if self.async_shared.is_some() {
            return Err(CommError::Protocol(
                "synchronous all_gather after post_reduce_bf16: the async readers own the sockets"
                    .to_owned(),
            ));
        }
        self.check_frame(local.len())?;
        let mut exchange = TcpExchange::new(self);
        exchange::all_gather(&mut exchange, self.rank, local, out, seq, self.world_size)
    }
}

/// Accepts one connection, polling so a peer that never appears becomes a Timeout.
fn accept_retry(
    listener: &TcpListener,
    deadline: Instant,
    timeout: Duration,
) -> Result<TcpStream, CommError> {
    loop {
        match listener.accept() {
            Ok((stream, _address)) => return Ok(stream),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(CommError::timeout(
                        timeout,
                        "accepting the connections from the lower ranks",
                    ));
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(error) => return Err(CommError::Io(error)),
        }
    }
}

/// Connects to a peer, retrying until the deadline so a slow listener is not a hard failure.
fn connect_retry(
    address: SocketAddr,
    deadline: Instant,
    timeout: Duration,
) -> Result<TcpStream, CommError> {
    loop {
        match TcpStream::connect(address) {
            Ok(stream) => return Ok(stream),
            Err(error) => {
                if Instant::now() >= deadline {
                    return Err(CommError::timeout(
                        timeout,
                        format!("connecting to {address}: {error}"),
                    ));
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        }
    }
}

/// Applies the frame deadline and disables Nagle so a payload is never held back by batching.
fn tune(stream: &TcpStream, timeout: Duration) -> Result<(), CommError> {
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    Ok(())
}

/// Sends this rank's identity so the accepting side can name the connection.
fn write_hello(stream: &mut TcpStream, rank: usize) -> Result<(), CommError> {
    let header = FrameHeader {
        kind: FrameKind::Hello,
        phase: 0,
        seq: 0,
        rank: rank as u32,
        payload_bytes: 0,
    };
    write_frame(stream, &header, &[])
}

/// Maps a peer-less read error onto the transport vocabulary once the peer is known.
fn map_peer_error(peer: usize, timeout: Duration, what: &str, error: CommError) -> CommError {
    match error {
        CommError::Io(io) => CommError::from_io_peer(peer, timeout, what, io),
        other => other,
    }
}

/// Reads and validates one payload frame for a known peer and exchange tag.
fn read_expected(
    reader: &mut TcpStream,
    peer: usize,
    tag: ExchangeTag,
    elements: usize,
    timeout: Duration,
    max_frame_bytes: usize,
) -> Result<(usize, Vec<u16>), CommError> {
    let expected_kind = match tag.kind {
        ExchangeKind::Reduce => FrameKind::ReducePayload,
        ExchangeKind::Gather => FrameKind::GatherPayload,
    };
    let (header, payload) = read_frame(reader, max_frame_bytes)
        .map_err(|error| map_peer_error(peer, timeout, "reading a payload frame", error))?;
    if header.kind != expected_kind
        || header.phase != tag.phase
        || header.seq != tag.seq
        || header.rank as usize != peer
    {
        return Err(CommError::Protocol(format!(
            "unexpected frame from rank {peer}: {header:?} (wanted {expected_kind:?} phase {} seq {})",
            tag.phase, tag.seq
        )));
    }
    let values = decode_bf16(&payload)?;
    if values.len() != elements {
        return Err(CommError::Protocol(format!(
            "frame from rank {peer} carried {} elements, expected {elements}",
            values.len()
        )));
    }
    Ok((peer, values))
}

/// One reader thread's join handle: the peer it read from and the decoded payload.
type ReaderHandle = JoinHandle<Result<(usize, Vec<u16>), CommError>>;

/// Payload bytes up to which one exchange may read its peer frames sequentially instead of
/// spawning one reader thread per peer.
///
/// The synchronous path sends its whole contribution to every peer before it reads anything, so a
/// sequential read after the sends is deadlock-free only while the outstanding writes fit in the
/// socket send buffers. A 10 KiB contribution on four ranks is 3 x 10 KiB, far below the default
/// buffer, while a 2 MiB contribution is not: above this threshold the exchange keeps the reader
/// threads that were always used. The threshold applies to the sum of the frames this rank
/// expects, which is the same magnitude as the writes it has just made.
///
/// The size alone does not select the path: [TcpTransport::sequential_read] must also be set,
/// which is the default only for an all-loopback cluster. On a real cluster the reader threads
/// stay in the row until a row measures that the sequential path is not slower there.
const SEQUENTIAL_READ_MAX_BYTES: usize = 256 * 1024;

/// True only when this rank and every peer it dials are addressed on this host.
///
/// That is the one topology in which the local bench measured the sequential small-frame read as
/// strictly better: the peers' frames are microseconds away, so there is no switch round trip for
/// a read-after-send to serialise behind.
fn cluster_is_loopback(config: &TcpConfig) -> bool {
    if !config.local_address().ip().is_loopback() {
        return false;
    }
    (0..config.world_size).all(|peer| {
        peer == config.rank
            || config
                .peer_address(peer)
                .is_ok_and(|address| address.ip().is_loopback())
    })
}

/// `SPARK_COMM_SEQUENTIAL_READ=1` forces the sequential small-frame read, `=0` forces the
/// reader threads; unset or any other value keeps the topology default.
fn sequential_read_override() -> Option<bool> {
    static OVERRIDE: OnceLock<Option<bool>> = OnceLock::new();
    *OVERRIDE.get_or_init(
        || match std::env::var("SPARK_COMM_SEQUENTIAL_READ").as_deref() {
            Ok("1" | "true") => Some(true),
            Ok("0" | "false") => Some(false),
            _ => None,
        },
    )
}

/// The TCP side of the schedule driver.
struct TcpExchange<'a> {
    transport: &'a TcpTransport,
    tag: Option<ExchangeTag>,
    readers: Vec<ReaderHandle>,
    /// When set, the frames this exchange expects and no reader threads were spawned.
    sequential: Option<Vec<(usize, usize)>>,
}

impl<'a> TcpExchange<'a> {
    fn new(transport: &'a TcpTransport) -> Self {
        Self {
            transport,
            tag: None,
            readers: Vec::new(),
            sequential: None,
        }
    }

    fn timeout(&self) -> Duration {
        self.transport.timeout
    }
}

impl Exchange for TcpExchange<'_> {
    fn prepare(&mut self, tag: ExchangeTag, expected: &[(usize, usize)]) -> Result<(), CommError> {
        self.tag = Some(tag);
        // Small exchanges read peer frames after the sends in collect(); only larger ones pay for
        // one reader thread per peer. Both paths run the same read_expected parser and both feed
        // the same validate_received, so the schedule sees identical frames either way.
        let expected_bytes: usize = expected
            .iter()
            .map(|(_, elements)| elements.saturating_mul(2))
            .sum();
        if expected_bytes <= SEQUENTIAL_READ_MAX_BYTES && self.transport.sequential_read {
            crate::profile::record_read_path(true);
            self.sequential = Some(expected.to_vec());
            return Ok(());
        }
        crate::profile::record_read_path(false);
        for (peer, elements) in expected {
            let reader = self.transport.read_handle(*peer)?.try_clone()?;
            let peer = *peer;
            let elements = *elements;
            let timeout = self.timeout();
            let max_frame_bytes = self.transport.max_frame_bytes;
            let handle = std::thread::spawn(move || {
                let wait_clock = crate::profile::peer_wait_clock();
                let mut reader = reader;
                let result =
                    read_expected(&mut reader, peer, tag, elements, timeout, max_frame_bytes);
                crate::profile::record_peer_wait(wait_clock);
                result
            });
            self.readers.push(handle);
        }
        Ok(())
    }

    fn send(
        &mut self,
        to: usize,
        elements: std::ops::Range<usize>,
        payload: &[u16],
    ) -> Result<(), CommError> {
        let tag = self.tag.ok_or_else(|| {
            CommError::Protocol("send() before prepare() on the TCP exchange".to_owned())
        })?;
        if elements.len() != payload.len() {
            return Err(CommError::InvalidLayout(format!(
                "transfer to rank {to} declares {} elements but carries {}",
                elements.len(),
                payload.len()
            )));
        }
        let payload_bytes = u32::try_from(payload.len() * 2).map_err(|_| {
            CommError::InvalidLayout(format!(
                "{} elements cannot be described by a u32 frame length",
                payload.len()
            ))
        })?;
        let kind = match tag.kind {
            ExchangeKind::Reduce => FrameKind::ReducePayload,
            ExchangeKind::Gather => FrameKind::GatherPayload,
        };
        let header = FrameHeader {
            kind,
            phase: tag.phase,
            seq: tag.seq,
            rank: self.transport.rank as u32,
            payload_bytes,
        };
        let bytes = encode_bf16(payload);
        let mut writer = self.transport.write_handle(to)?;
        // A small frame goes out as one write instead of a 24-byte header write followed by the
        // payload write: on the served shape that halves the syscalls per peer, and the extra
        // 24-byte copy is irrelevant. Large frames keep the direct two-write path so a 2 MiB
        // payload is never copied into a second buffer just to save a syscall.
        if bytes.len() <= SEQUENTIAL_READ_MAX_BYTES {
            let mut frame = Vec::with_capacity(FRAME_HEADER_BYTES + bytes.len());
            frame.extend_from_slice(&header.encode());
            frame.extend_from_slice(&bytes);
            writer
                .write_all(&frame)
                .and_then(|()| writer.flush())
                .map_err(|error| {
                    map_peer_error(
                        to,
                        self.timeout(),
                        "writing a payload frame",
                        CommError::Io(error),
                    )
                })
        } else {
            write_frame(&mut writer, &header, &bytes).map_err(|error| {
                map_peer_error(to, self.timeout(), "writing a payload frame", error)
            })
        }
    }

    fn collect(&mut self, expected: &[(usize, usize)]) -> Result<Vec<Received>, CommError> {
        if let Some(peers) = self.sequential.take() {
            let tag = self.tag.ok_or_else(|| {
                CommError::Protocol("collect() before prepare() on the TCP exchange".to_owned())
            })?;
            let mut frames = Vec::with_capacity(peers.len());
            for (peer, elements) in peers {
                let wait_clock = crate::profile::peer_wait_clock();
                let mut reader = self.transport.read_handle(peer)?.try_clone()?;
                let result = read_expected(
                    &mut reader,
                    peer,
                    tag,
                    elements,
                    self.timeout(),
                    self.transport.max_frame_bytes,
                );
                crate::profile::record_peer_wait(wait_clock);
                let (peer, payload) = result?;
                frames.push(Received {
                    from: peer,
                    payload,
                });
            }
            exchange::validate_received(expected, &frames)?;
            return Ok(frames);
        }
        let mut first_error: Option<CommError> = None;
        let mut frames = Vec::with_capacity(self.readers.len());
        for handle in self.readers.drain(..) {
            match handle.join() {
                Ok(Ok((peer, payload))) => frames.push(Received {
                    from: peer,
                    payload,
                }),
                Ok(Err(error)) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
                Err(_panic) => {
                    if first_error.is_none() {
                        first_error = Some(CommError::Protocol(
                            "a TCP reader thread panicked".to_owned(),
                        ));
                    }
                }
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        exchange::validate_received(expected, &frames)?;
        Ok(frames)
    }

    fn finish(&mut self) -> Result<(), CommError> {
        self.sequential = None;
        for handle in self.readers.drain(..) {
            let _ = handle.join();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sequential small-frame read defaults to the all-loopback topology only, so a cluster
    /// row keeps the reader threads until an experiment asks for the other path.
    #[test]
    fn sequential_read_defaults_to_loopback_only() {
        assert!(cluster_is_loopback(&TcpConfig::loopback4(0, 19000)));
        let addresses: Vec<SocketAddr> = (0..4)
            .map(|rank| format!("192.0.2.{}:29601", 80 + rank).parse().unwrap())
            .collect();
        let cluster = TcpConfig::with_addresses(1, addresses).unwrap();
        assert!(!cluster_is_loopback(&cluster));
        // One non-loopback peer is enough: the read must not assume the others are local.
        let mixed: Vec<SocketAddr> = vec![
            "127.0.0.1:29601".parse().unwrap(),
            "127.0.0.1:29602".parse().unwrap(),
            "192.0.2.59:29603".parse().unwrap(),
            "127.0.0.1:29604".parse().unwrap(),
        ];
        assert!(!cluster_is_loopback(
            &TcpConfig::with_addresses(0, mixed).unwrap()
        ));
    }

    /// A reader that hands out one byte per call, so read_exact has to reassemble the frame.
    struct DribbleReader<'a> {
        bytes: &'a [u8],
        position: usize,
    }

    impl Read for DribbleReader<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.position >= self.bytes.len() || buf.is_empty() {
                return Ok(0);
            }
            buf[0] = self.bytes[self.position];
            self.position += 1;
            Ok(1)
        }
    }

    fn framed(header: FrameHeader, payload: &[u8]) -> Vec<u8> {
        let mut bytes = header.encode().to_vec();
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn header_round_trips_exactly() {
        let header = FrameHeader {
            kind: FrameKind::ReducePayload,
            phase: 1,
            seq: 0x0123_4567_89ab_cdef,
            rank: 3,
            payload_bytes: 4096,
        };
        assert_eq!(FrameHeader::decode(&header.encode()).unwrap(), header);
        assert_eq!(header.encode().len(), FRAME_HEADER_BYTES);
    }

    #[test]
    fn header_rejects_bad_magic_version_and_kind() {
        let mut bytes = FrameHeader {
            kind: FrameKind::Hello,
            phase: 0,
            seq: 0,
            rank: 0,
            payload_bytes: 0,
        }
        .encode();
        bytes[0] ^= 0xff;
        assert!(matches!(
            FrameHeader::decode(&bytes),
            Err(CommError::Protocol(_))
        ));

        let mut bytes = FrameHeader {
            kind: FrameKind::Hello,
            phase: 0,
            seq: 0,
            rank: 0,
            payload_bytes: 0,
        }
        .encode();
        bytes[4] = 99;
        assert!(matches!(
            FrameHeader::decode(&bytes),
            Err(CommError::Protocol(_))
        ));

        let mut bytes = FrameHeader {
            kind: FrameKind::Hello,
            phase: 0,
            seq: 0,
            rank: 0,
            payload_bytes: 0,
        }
        .encode();
        bytes[6] = 77;
        assert!(matches!(
            FrameHeader::decode(&bytes),
            Err(CommError::Protocol(_))
        ));

        assert!(matches!(
            FrameHeader::decode(&[0u8; 4]),
            Err(CommError::Protocol(_))
        ));
    }

    #[test]
    fn read_frame_reassembles_a_dribbled_frame() {
        let header = FrameHeader {
            kind: FrameKind::GatherPayload,
            phase: 0,
            seq: 7,
            rank: 2,
            payload_bytes: 6,
        };
        let bytes = framed(header, &[1, 2, 3, 4, 5, 6]);
        let mut reader = DribbleReader {
            bytes: &bytes,
            position: 0,
        };
        let (decoded, payload) = read_frame(&mut reader, 1024)
            .unwrap_or_else(|error| panic!("dribbled frame failed: {error}"));
        assert_eq!(decoded, header);
        assert_eq!(payload, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn read_frame_rejects_oversized_and_truncated_frames() {
        let header = FrameHeader {
            kind: FrameKind::ReducePayload,
            phase: 0,
            seq: 1,
            rank: 0,
            payload_bytes: 1024,
        };
        let bytes = framed(header, &[0u8; 1024]);
        let mut reader = DribbleReader {
            bytes: &bytes,
            position: 0,
        };
        assert!(matches!(
            read_frame(&mut reader, 16),
            Err(CommError::Protocol(_))
        ));

        // A header that promises more bytes than the stream holds is an I/O EOF.
        let truncated = framed(header, &[0u8; 4]);
        let mut reader = DribbleReader {
            bytes: &truncated,
            position: 0,
        };
        assert!(matches!(
            read_frame(&mut reader, 4096),
            Err(CommError::Io(error)) if error.kind() == io::ErrorKind::UnexpectedEof
        ));
    }

    #[test]
    fn write_frame_refuses_a_mismatched_length() {
        let header = FrameHeader {
            kind: FrameKind::ReducePayload,
            phase: 0,
            seq: 1,
            rank: 0,
            payload_bytes: 4,
        };
        let mut sink: Vec<u8> = Vec::new();
        assert!(matches!(
            write_frame(&mut sink, &header, &[0u8; 2]),
            Err(CommError::Protocol(_))
        ));
        write_frame(&mut sink, &header, &[0u8; 4]).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(sink.len(), FRAME_HEADER_BYTES + 4);
    }

    #[test]
    fn bf16_byte_order_is_little_endian() {
        let values = vec![0x3f80u16, 0x8000, 0xffff];
        let bytes = encode_bf16(&values);
        assert_eq!(bytes, vec![0x80, 0x3f, 0x00, 0x80, 0xff, 0xff]);
        assert_eq!(decode_bf16(&bytes).unwrap(), values);
        assert!(matches!(
            decode_bf16(&[0u8; 3]),
            Err(CommError::Protocol(_))
        ));
    }

    #[test]
    fn config_helpers_are_consistent() {
        let config = TcpConfig::loopback4(3, 41000).with_timeout(Duration::from_millis(10));
        assert_eq!(config.rank, 3);
        assert_eq!(config.world_size, 4);
        assert_eq!(config.timeout, Duration::from_millis(10));
        assert_eq!(config.base_port, 41000);
    }
}

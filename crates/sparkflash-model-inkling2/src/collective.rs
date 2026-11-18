// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The three exchanges the Inkling schedule performs.
//!
//! Expert parallelism needs one all-reduce a layer, and a vocabulary-parallel embedding and head
//! need one row gather a token and one slice gather a step. All three are small - 18 KiB for the
//! residual width, 401 KiB for the vocabulary at TP4 - beside the gigabytes of expert weights a
//! token reads, so this layer is a correctness seam first and a bandwidth argument second.
//!
//! The wire format is the shared one: [sparkflash_comm::tcp]'s 24-byte header, its frame kinds
//! and its address table are what this module speaks, so an Inkling rank and a DeepSeek rank
//! describe their cluster the same way. What is its own is the element type and the reduction
//! rule: the payload is f32, and a sum accumulates in ascending rank order from zero, which is
//! the rule [sparkflash_comm::reduce] applies to its bf16 payloads. Every rank therefore computes
//! the same bits, and the device arm's all-reduce cannot depend on which rank asked first.
//!
//! The exchange is a trait for one reason: a world of one must not open a socket. [Solo] is the
//! identity, and it is what a single-rank serve uses.

use std::io::{BufReader, BufWriter};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Mutex;
use std::time::Duration;

use sparkflash_comm::device_reduce::{
    BootstrapConfig, RunReduce, WordGather, open_run_reduce_over,
};
use sparkflash_comm::nccl::NcclError;
use sparkflash_comm::tcp::{FrameHeader, FrameKind, TcpConfig, read_frame, write_frame};

/// The exchanges a rank performs with its peers.
///
/// Both operations are collective: every rank of the world must call the same one with the same
/// length, in the same order.
pub trait Exchange: Send + Sync {
    /// This rank's index.
    fn rank(&self) -> usize;

    /// The number of ranks.
    fn world(&self) -> usize;

    /// Sum the values across the world, in place.
    ///
    /// # Errors
    /// A peer that is gone, a frame that disagrees with the collective, or a transport timeout.
    fn all_reduce_f32(&self, values: &mut [f32]) -> Result<(), String>;

    /// Gather every rank's values into the output buffer, in rank order.
    ///
    /// The buffer must hold the world's worth of elements.
    ///
    /// # Errors
    /// A length mismatch, a peer that is gone, or a transport timeout.
    fn all_gather_f32(&self, values: &[f32], out: &mut [f32]) -> Result<(), String>;

    /// Whether [Self::exchange_on_device] is wired, so a collective can stay on the device.
    fn has_device_leg(&self) -> bool {
        false
    }

    /// Enqueues on `stream` the copy of this rank's `bytes` at the device address `send` into
    /// every peer's `slots + rank * bytes`, and every peer's into this rank's
    /// `slots + peer * bytes`; this rank's own slot is not written and nothing is synchronised.
    /// The words are moved, never summed: the caller sums its own payload and the slots in
    /// ascending rank order from zero on the same stream, which is [Self::all_reduce_f32]'s rule,
    /// so the device leg and the host one give the same bits. (The contract is
    /// sparkflash_weights::tp::TpCollective::exchange_on_device's.)
    ///
    /// # Errors
    /// An exchange with no device leg, or one the library refuses to enqueue.
    fn exchange_on_device(
        &self,
        _stream: u64,
        _send: u64,
        _slots: u64,
        _bytes: usize,
    ) -> Result<(), String> {
        Err("this exchange has no device leg".to_owned())
    }

    /// Enqueues on `stream` the two-shot sum's first half: chunk `peer` of the `world` equal
    /// chunks of the `bytes` at `send` goes to `peer`, and every peer's chunk `rank` lands at
    /// `slots + peer * bytes / world`; this rank's own slot is not written. The caller sums its
    /// own chunk and the slots in ascending rank order and shares the sum with
    /// [Self::exchange_on_device], so every element is folded as the one-shot exchange folds it
    /// and each rank sends half the bytes. (The contract is
    /// sparkflash_comm::device_reduce's scatter_bf16, which DeepSeek's two-shot reduce uses.)
    ///
    /// # Errors
    /// An exchange with no device leg, or one the library refuses to enqueue.
    fn scatter_on_device(
        &self,
        _stream: u64,
        _send: u64,
        _slots: u64,
        _bytes: usize,
    ) -> Result<(), String> {
        Err("this exchange has no device leg".to_owned())
    }

    /// Reports a device-leg round that could not complete, once the device has drained.
    ///
    /// The direct-RDMA exchange's stage kernel gives a round up rather than hang when a peer's row
    /// never arrives or disagrees, and what the step wrote behind it is not usable; the walk calls
    /// this after each drain and before it reads a result back. Every other leg reports its
    /// failures where it enqueues, so the default has nothing to say.
    ///
    /// # Errors
    /// The exchange's first failed round.
    fn check_device(&self) -> Result<(), String> {
        Ok(())
    }
}

/// The switch the TP arms share for their collective path: `nccl` (the default for a world of
/// two or more) keeps the step's collectives on the device through the run's NCCL communicator,
/// `tcp` stages every one through the host exchange. The DeepSeek, GLM and Qwen serve paths read
/// the same variable.
pub const COMM_REDUCE_ENV: &str = "SPARK_COMM_REDUCE";

/// Whether this world's collectives take the NCCL device leg, from [COMM_REDUCE_ENV].
///
/// # Errors
/// A value this arm has no path for: anything but `nccl` or `tcp`, or `nccl` in a world of one.
pub fn nccl_requested(world: usize) -> Result<bool, String> {
    match std::env::var(COMM_REDUCE_ENV).ok().as_deref() {
        None | Some("") => Ok(world > 1),
        Some("nccl") if world > 1 => Ok(true),
        Some("nccl") => Err(format!(
            "{COMM_REDUCE_ENV}=nccl needs a world of 2 or more; a world of 1 has nothing to exchange"
        )),
        Some("tcp") => Ok(false),
        Some(other) => Err(format!(
            "{COMM_REDUCE_ENV}={other:?} has no Inkling path: nccl or tcp"
        )),
    }
}

/// A world of one: both exchanges are the identity, and no socket is opened.
#[derive(Debug, Clone, Copy, Default)]
pub struct Solo;

impl Exchange for Solo {
    fn rank(&self) -> usize {
        0
    }

    fn world(&self) -> usize {
        1
    }

    fn all_reduce_f32(&self, _values: &mut [f32]) -> Result<(), String> {
        Ok(())
    }

    fn all_gather_f32(&self, values: &[f32], out: &mut [f32]) -> Result<(), String> {
        if out.len() != values.len() {
            return Err(format!(
                "a world of one gathers {} elements into a {} element buffer",
                values.len(),
                out.len()
            ));
        }
        out.copy_from_slice(values);
        Ok(())
    }
}

/// One peer's connection: the framed writer and the framed reader share the socket.
struct Peer {
    writer: Mutex<BufWriter<TcpStream>>,
    reader: Mutex<BufReader<TcpStream>>,
}

/// An all-to-all exchange over the shared frame protocol.
///
/// Every rank writes its contribution to each peer and then reads each peer's. The write half
/// runs on its own thread inside one collective, because a blocking write of a 401 KiB logit
/// slice can fill a peer's receive buffer while that peer is blocked writing its own: the reads
/// have to make progress for the writes to finish.
pub struct TcpExchange {
    rank: usize,
    world: usize,
    peers: Vec<Option<Peer>>,
    max_frame_bytes: usize,
    sequence: Mutex<u64>,
}

impl std::fmt::Debug for TcpExchange {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TcpExchange")
            .field("rank", &self.rank)
            .field("world", &self.world)
            .finish()
    }
}

/// Dials one address before the deadline, retrying a refused connection.
fn dial_within(address: &SocketAddr, deadline: std::time::Instant) -> Result<TcpStream, String> {
    loop {
        match TcpStream::connect_timeout(address, Duration::from_millis(250)) {
            Ok(stream) => return Ok(stream),
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::TimedOut
                ) =>
            {
                if std::time::Instant::now() >= deadline {
                    return Err(format!("dial {address}: {error}"));
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(error) => return Err(format!("dial {address}: {error}")),
        }
    }
}

/// Accepts one connection before the deadline, polling the non-blocking listener.
fn accept_within(
    listener: &TcpListener,
    deadline: std::time::Instant,
) -> Result<TcpStream, String> {
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                return stream
                    .set_nonblocking(false)
                    .map_err(|error| format!("accept: {error}"))
                    .map(|()| stream);
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if std::time::Instant::now() >= deadline {
                    return Err(
                        "the cluster did not connect inside the connect timeout: a peer never \
                         dialed this rank"
                            .to_owned(),
                    );
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(error) => return Err(format!("accept: {error}")),
        }
    }
}

/// One completed connection: both sides sent their hello and read the peer's, and the peer's
/// claimed rank is returned so the caller can file it under the right index.
///
/// The rank is read from the frame rather than assumed from the accept order: a listener accepts
/// whichever peer dialed first, not the lowest one, and an exchange filed under the wrong index
/// reads the wrong stream for the rest of the run.
fn handshake(
    stream: TcpStream,
    rank: usize,
    timeout: Duration,
    max_frame_bytes: usize,
) -> Result<(usize, Peer), String> {
    stream
        .set_read_timeout(Some(timeout))
        .and_then(|()| stream.set_write_timeout(Some(timeout)))
        .and_then(|()| stream.set_nodelay(true))
        .map_err(|error| format!("a connection: {error}"))?;
    let clone = stream
        .try_clone()
        .map_err(|error| format!("a connection: {error}"))?;
    let mut writer = BufWriter::new(clone);
    let mut reader = BufReader::new(stream);
    let header = FrameHeader {
        kind: FrameKind::Hello,
        phase: 0,
        seq: 0,
        rank: rank as u32,
        payload_bytes: 0,
    };
    write_frame(&mut writer, &header, &[]).map_err(|error| format!("hello: {error}"))?;
    let (header, _) =
        read_frame(&mut reader, max_frame_bytes).map_err(|error| format!("hello: {error}"))?;
    if header.kind != FrameKind::Hello {
        return Err(format!("a peer opened with {:?}, not a hello", header.kind));
    }
    let peer = header.rank as usize;
    if std::env::var_os("SPARKFLASH_INKLING_TRACE").is_some() {
        eprintln!(
            "[inkling] rank {rank} handshake read peer {peer} of kind {:?}",
            header.kind
        );
    }
    let _ = rank;
    Ok((
        peer,
        Peer {
            writer: Mutex::new(writer),
            reader: Mutex::new(reader),
        },
    ))
}

impl TcpExchange {
    /// Form the cluster: bind this rank's address and dial every peer.
    ///
    /// Lower ranks dial upward, so each pair owns exactly one connection and both directions
    /// share it. The call blocks until the whole cluster is connected, which is what makes the
    /// returned handle safe to use without a further barrier.
    ///
    /// # Errors
    /// A rank outside the table, an address that cannot be bound or reached, or a hello frame
    /// that names the wrong rank.
    pub fn connect(config: &TcpConfig) -> Result<Self, String> {
        let world = config.world_size;
        let rank = config.rank;
        if world == 0 {
            return Err("a cluster needs at least one rank".to_owned());
        }
        if rank >= world {
            return Err(format!("rank {rank} is outside a world of {world}"));
        }
        let bind = config.bind_addr.unwrap_or_else(|| {
            SocketAddr::from((
                [127, 0, 0, 1],
                (u32::from(config.base_port) + rank as u32) as u16,
            ))
        });
        let dial = |peer: usize| -> SocketAddr {
            config.peer_addrs.as_ref().map_or_else(
                || {
                    SocketAddr::from((
                        [127, 0, 0, 1],
                        (u32::from(config.base_port) + peer as u32) as u16,
                    ))
                },
                |table| table[peer],
            )
        };
        if std::env::var_os("SPARKFLASH_INKLING_TRACE").is_some() {
            eprintln!("[inkling] rank {rank} binds {bind}");
        }
        let listener = TcpListener::bind(bind).map_err(|error| format!("bind {bind}: {error}"))?;
        // A listener has no accept deadline, and a rank whose peer died must not wait forever:
        // the cluster's setup budget bounds every accept below.
        listener
            .set_nonblocking(true)
            .map_err(|error| format!("bind {bind}: {error}"))?;
        let deadline = std::time::Instant::now() + config.connect_timeout;
        let mut peers: Vec<Option<Peer>> = (0..world).map(|_| None).collect();
        // Rank r accepts from every lower peer and dials every higher one, so each pair has one
        // connection and a rank never blocks on a peer that is waiting for it.
        for _ in 0..rank {
            let stream = accept_within(&listener, deadline)?;
            let (peer, connection) =
                handshake(stream, rank, config.timeout, config.max_frame_bytes)?;
            if peers[peer].is_some() {
                return Err(format!("rank {peer} connected twice"));
            }
            peers[peer] = Some(connection);
        }
        for (peer, slot) in peers.iter_mut().enumerate().skip(rank + 1) {
            let address = dial(peer);
            if std::env::var_os("SPARKFLASH_INKLING_TRACE").is_some() {
                eprintln!("[inkling] rank {rank} dials {peer} at {address}");
            }
            // A peer may not have bound its listener yet, so a refused connection is retried
            // until the cluster's setup budget runs out rather than failing the whole cluster on
            // a scheduling race.
            let stream = dial_within(&address, deadline)?;
            let (answered, connection) =
                handshake(stream, rank, config.timeout, config.max_frame_bytes)?;
            if answered != peer {
                return Err(format!("dialed rank {peer} and reached rank {answered}"));
            }
            *slot = Some(connection);
        }
        Ok(Self {
            rank,
            world,
            peers,
            max_frame_bytes: config.max_frame_bytes,
            sequence: Mutex::new(0),
        })
    }

    fn next_sequence(&self) -> Result<u64, String> {
        let mut sequence = self
            .sequence
            .lock()
            .map_err(|_| "the collective sequence lock is poisoned".to_owned())?;
        *sequence += 1;
        Ok(*sequence)
    }

    /// Write one frame to every peer, then read every peer's frame for this collective.
    fn exchange(
        &self,
        kind: FrameKind,
        payload: &[u8],
        expected_bytes: usize,
    ) -> Result<Vec<Vec<u8>>, String> {
        let sequence = self.next_sequence()?;
        let header = FrameHeader {
            kind,
            phase: 0,
            seq: sequence,
            rank: self.rank as u32,
            payload_bytes: u32::try_from(payload.len())
                .map_err(|_| format!("{} payload bytes past u32", payload.len()))?,
        };
        // The write half runs while this rank reads: a peer's blocking write needs our reads.
        std::thread::scope(|scope| -> Result<Vec<Vec<u8>>, String> {
            let writers = scope.spawn(|| -> Result<(), String> {
                for peer in 0..self.world {
                    if peer == self.rank {
                        continue;
                    }
                    let connection = self
                        .peers
                        .get(peer)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("no connection to rank {peer}"))?;
                    let mut writer = connection
                        .writer
                        .lock()
                        .map_err(|_| format!("the writer to rank {peer} is poisoned"))?;
                    write_frame(&mut *writer, &header, payload)
                        .map_err(|error| format!("write to rank {peer}: {error}"))?;
                }
                Ok(())
            });
            let mut frames = Vec::with_capacity(self.world.saturating_sub(1));
            for peer in 0..self.world {
                if peer == self.rank {
                    continue;
                }
                let connection = self
                    .peers
                    .get(peer)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("no connection to rank {peer}"))?;
                let mut reader = connection
                    .reader
                    .lock()
                    .map_err(|_| format!("the reader from rank {peer} is poisoned"))?;
                let (header, payload) = read_frame(&mut *reader, self.max_frame_bytes)
                    .map_err(|error| format!("read from rank {peer}: {error}"))?;
                if header.kind != kind || header.seq != sequence || header.rank as usize != peer {
                    return Err(format!(
                        "rank {peer} answered a {:?} {sequence} with a {:?} {} from {}",
                        kind, header.kind, header.seq, header.rank
                    ));
                }
                if payload.len() != expected_bytes {
                    return Err(format!(
                        "rank {peer} sent {} bytes where the collective expects {expected_bytes}",
                        payload.len()
                    ));
                }
                frames.push(payload);
            }
            writers
                .join()
                .map_err(|_| "the collective writer thread panicked".to_owned())??;
            Ok(frames)
        })
    }
}

/// The f32 payload of one buffer, little-endian, the framing's byte order.
fn encode_f32(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

impl Exchange for TcpExchange {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world(&self) -> usize {
        self.world
    }

    fn all_reduce_f32(&self, values: &mut [f32]) -> Result<(), String> {
        if self.world == 1 {
            return Ok(());
        }
        if values.is_empty() {
            return Err("an all-reduce needs at least one element".to_owned());
        }
        let frames = self.exchange(
            FrameKind::ReducePayload,
            &encode_f32(values),
            values.len() * 4,
        )?;
        // Ascending rank from zero, the shared reduce module's rule, so every rank holds the
        // same bits whatever order the frames arrived in.
        let mut panes: Vec<Vec<f32>> = Vec::with_capacity(self.world);
        let mut peer_frames = frames.into_iter();
        for peer in 0..self.world {
            if peer == self.rank {
                panes.push(values.to_vec());
            } else {
                let payload = peer_frames
                    .next()
                    .ok_or_else(|| "a peer's reduce frame is missing".to_owned())?;
                panes.push(decode_f32(&payload));
            }
        }
        for index in 0..values.len() {
            let mut sum = 0.0f32;
            for pane in &panes {
                sum += pane[index];
            }
            values[index] = sum;
        }
        Ok(())
    }

    fn all_gather_f32(&self, values: &[f32], out: &mut [f32]) -> Result<(), String> {
        if out.len() != values.len().saturating_mul(self.world) {
            return Err(format!(
                "a gather of {} elements a rank into a {} element buffer of {} ranks",
                values.len(),
                out.len(),
                self.world
            ));
        }
        if self.world == 1 {
            out.copy_from_slice(values);
            return Ok(());
        }
        let frames = self.exchange(
            FrameKind::GatherPayload,
            &encode_f32(values),
            values.len() * 4,
        )?;
        let width = values.len();
        out[self.rank * width..(self.rank + 1) * width].copy_from_slice(values);
        let mut peer_frames = frames.into_iter();
        for peer in 0..self.world {
            if peer == self.rank {
                continue;
            }
            let payload = peer_frames
                .next()
                .ok_or_else(|| "a peer's gather frame is missing".to_owned())?;
            out[peer * width..(peer + 1) * width].copy_from_slice(&decode_f32(&payload));
        }
        Ok(())
    }
}

/// The NCCL bootstrap's gather over this exchange: the rendezvous rides the collective's own
/// connections, so opening the communicator needs no second socket set. The frames carry this
/// exchange's own sequence, so the bootstrap's sequence number is not needed to tell them apart.
impl WordGather for TcpExchange {
    fn gather_words(&self, _sequence: u64, local: &[u16], out: &mut [u16]) -> Result<(), String> {
        if out.len() != local.len().saturating_mul(self.world) {
            return Err(format!(
                "a gather of {} words a rank into a {} word buffer of {} ranks",
                local.len(),
                out.len(),
                self.world
            ));
        }
        let payload: Vec<u8> = local.iter().flat_map(|word| word.to_le_bytes()).collect();
        let width = local.len();
        out[self.rank * width..(self.rank + 1) * width].copy_from_slice(local);
        if self.world == 1 {
            return Ok(());
        }
        let frames = self.exchange(FrameKind::GatherPayload, &payload, payload.len())?;
        let mut peer_frames = frames.into_iter();
        for peer in 0..self.world {
            if peer == self.rank {
                continue;
            }
            let frame = peer_frames
                .next()
                .ok_or_else(|| "a peer's gather frame is missing".to_owned())?;
            for (slot, word) in out[peer * width..(peer + 1) * width]
                .iter_mut()
                .zip(frame.as_chunks::<2>().0)
            {
                *slot = u16::from_le_bytes(*word);
            }
        }
        Ok(())
    }
}

/// The run's NCCL communicator, owned behind a lock, and the reference to the primary context it
/// was opened in. The fields drop in order, so the communicator is destroyed while its context
/// still stands: a context whose last reference is released is reset, with the communicator's
/// device buffers in it.
struct Communicator {
    reduce: Mutex<RunReduce>,
    // Held for its drop order in every build; read only by the doorbell's launches.
    #[cfg_attr(not(feature = "rdma"), allow(dead_code))]
    context: sparkflash_kernels::cuda_driver::Cuda,
}

// SAFETY: RunReduce holds the communicator's raw handles, which NCCL lets any one thread use at a
// time; the lock keeps every call on one thread at a time, and every rank enqueues its collectives
// on the default stream of the primary context the arm's kernels use. After the open the context
// reference launches only the direct-RDMA exchange's kernels, under that exchange's own lock.
unsafe impl Send for Communicator {}
// SAFETY: as above: shared access goes through the lock.
unsafe impl Sync for Communicator {}

/// [TcpExchange] with the run's NCCL communicator as its device leg, and the direct-RDMA
/// exchange (the GB10 doorbell) in front of it for the rounds that fit one.
///
/// The host collectives stay the TCP exchange's. [Exchange::exchange_on_device] is one round of
/// the doorbell when the run opened it and the payload is at most [DOORBELL_ROW_CAP] bytes - a
/// decode step's sums and its logit gather - through the DeepSeek arm's own stage and collect
/// kernels ([sparkflash_kernels::write_exchange]); a wider payload, a prefill chunk's, is one
/// RunReduce::exchange_bf16 of its bytes, the same one-hop exchange the DeepSeek and GLM device
/// arms use. Both move bits and sum nothing, so the fold behind either gives the same bits. The
/// choice is the payload's width, which every rank shares, so every rank takes the same leg.
pub struct NcclExchange {
    host: TcpExchange,
    reduce: Communicator,
    /// The doorbell, when this run opened it.
    #[cfg(feature = "rdma")]
    doorbell: Option<&'static Mutex<sparkflash_kernels::write_exchange::ExchangeState>>,
}

/// The widest round the doorbell carries for this arm, every arm's ([write_exchange's
/// cap](sparkflash_kernels::write_exchange::DOORBELL_ROW_CAP)): a decode step's logit slice is
/// the widest of its collectives (50,015 f32 a rank at TP4, 200,060 bytes), and the residual sums
/// are 18 KiB. A prefill chunk of a few rows fits too; a wider one stays on NCCL, which the open
/// banner says.
#[cfg(feature = "rdma")]
pub const DOORBELL_ROW_CAP: usize = sparkflash_kernels::write_exchange::DOORBELL_ROW_CAP;

/// What the open banner says the doorbell replaces in this arm.
#[cfg(feature = "rdma")]
const DOORBELL_REPLACES: &str = "every Inkling collective of at most 262144 bytes (a decode \
     step's embedding, attention and expert sums and its logit gather, and a prefill chunk's when \
     it is that narrow); wider prefill chunks stay on NCCL";

/// Opens the doorbell for this rank, or `None` when the run selected the NCCL group or was never
/// told where its peers are (the shared switch, [sparkflash_kernels::write_exchange::EXCHANGE_ENV]).
///
/// Collective: every rank calls it at the same point, right after the NCCL open, and the
/// exchange's own rendezvous is inside.
#[cfg(feature = "rdma")]
fn open_doorbell(
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    world: usize,
    rank: usize,
) -> Result<Option<&'static Mutex<sparkflash_kernels::write_exchange::ExchangeState>>, String> {
    use sparkflash_kernels::write_exchange::{Opening, shared, validate};
    validate()?;
    let opening = Opening {
        row_cap: DOORBELL_ROW_CAP,
        replaces: DOORBELL_REPLACES,
    };
    shared(cuda, world, rank, opening, || {
        let image = sparkflash_kernels::images::inkling_cubin();
        if image.is_empty() {
            return Err(
                "this build carries no Inkling cubin, whose image holds the exchange's \
                        stage and collect kernels"
                    .to_owned(),
            );
        }
        Ok(image)
    })
}

/// A build without the verbs backend has no doorbell; a run that asked for it by name is refused
/// rather than measured on the group it was trying to replace.
#[cfg(not(feature = "rdma"))]
fn refuse_doorbell() -> Result<(), String> {
    match std::env::var("SPARKFLASH_EXCHANGE")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        None | Some("" | "0" | "nccl" | "off" | "NCCL" | "OFF") => Ok(()),
        Some(other) => Err(format!(
            "SPARKFLASH_EXCHANGE={other:?}: this build has no direct-RDMA exchange (it was built \
             without the rdma feature); unset it or say nccl"
        )),
    }
}

impl std::fmt::Debug for NcclExchange {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NcclExchange")
            .field("rank", &self.host.rank)
            .field("world", &self.host.world)
            .finish()
    }
}

impl NcclExchange {
    /// Opens the run's communicator over `host`'s connections and pairs the two.
    ///
    /// Collective: every rank of the world calls it at the same point with the same `run`, whose
    /// digest is the agreement word, so ranks about to run different things refuse before any
    /// enters NCCL. The warm-up moves one `payload_words` bf16 slab; the calling thread's CUDA
    /// context is the one the communicator binds, which must be the primary context the arm
    /// launches on.
    ///
    /// # Errors
    /// A library without point-to-point calls, or any refusal of the shared open sequence.
    pub fn open(host: TcpExchange, payload_words: usize, run: &str) -> Result<Self, String> {
        use sparkflash_kernels::cuda_driver::Cuda;
        let cuda = Cuda::load()?;
        let mut allocated = 0usize;
        let scratch = cuda.alloc(payload_words.max(1) * 2, &mut allocated)?;
        let config = BootstrapConfig {
            native_rounding: 0,
            env_toggles: 0,
            // FNV-1a over the run description, the agreement digest the shared opener checks.
            plan_digest: sparkflash_weights::rank_file::fnv1a(run.as_bytes()),
        };
        let sync = || {
            cuda.synchronize()
                .map_err(|error| NcclError::Bootstrap(format!("the setup sync: {error}")))
        };
        // SAFETY: `scratch` is a fresh device allocation of `payload_words` bf16 values on the
        // context this thread holds current, stream 0 is that context's default stream, and every
        // rank passes the same payload.
        let opened = unsafe {
            open_run_reduce_over(
                &host,
                host.world,
                host.rank,
                0,
                scratch.0,
                payload_words.max(1),
                config,
                &sync,
            )
        };
        cuda.free_buffers(&[&scratch]);
        let reduce = opened.map_err(|error| format!("nccl open: {error}"))?;
        if !reduce.has_exchange() {
            return Err("the NCCL library has no point-to-point calls to exchange with".to_owned());
        }
        #[cfg(feature = "rdma")]
        let doorbell = open_doorbell(&cuda, host.world, host.rank)?;
        #[cfg(not(feature = "rdma"))]
        refuse_doorbell()?;
        Ok(Self {
            host,
            reduce: Communicator {
                reduce: Mutex::new(reduce),
                context: cuda,
            },
            #[cfg(feature = "rdma")]
            doorbell,
        })
    }
}

impl Exchange for NcclExchange {
    fn rank(&self) -> usize {
        self.host.rank
    }

    fn world(&self) -> usize {
        self.host.world
    }

    fn all_reduce_f32(&self, values: &mut [f32]) -> Result<(), String> {
        self.host.all_reduce_f32(values)
    }

    fn all_gather_f32(&self, values: &[f32], out: &mut [f32]) -> Result<(), String> {
        self.host.all_gather_f32(values, out)
    }

    fn has_device_leg(&self) -> bool {
        true
    }

    fn exchange_on_device(
        &self,
        stream: u64,
        send: u64,
        slots: u64,
        bytes: usize,
    ) -> Result<(), String> {
        if !bytes.is_multiple_of(2) || bytes == 0 {
            return Err(format!(
                "{bytes} bytes are not whole bf16 words, the unit the exchange moves"
            ));
        }
        #[cfg(feature = "rdma")]
        if let Some(doorbell) = self.doorbell
            && stream == 0
            && bytes <= DOORBELL_ROW_CAP
        {
            let mut state = doorbell.lock().map_err(|_| {
                "the direct-RDMA exchange is poisoned by an earlier panic".to_owned()
            })?;
            if bytes <= state.row_cap() {
                let (rank, world) = (self.host.rank, self.host.world);
                // The stage kernel copies `send` into this rank's row of the round and rings the
                // doorbell; the collect kernel behind it copies the peers' rows into their slots.
                // Both are on the default stream, which is the one the walk folds on.
                return state.exchange_on(
                    &self.reduce.context,
                    0,
                    send,
                    slots,
                    bytes,
                    rank,
                    world,
                    &"inkling exchange",
                );
            }
        }
        let mut reduce = self
            .reduce
            .reduce
            .lock()
            .map_err(|_| "the communicator lock is poisoned".to_owned())?;
        // SAFETY: Exchange::exchange_on_device's contract is exchange_bf16's: `send` holds
        // `bytes` and `slots` a world of them on the communicator's device, `stream` is a stream
        // there, and every rank enqueues the same exchanges in the same order.
        unsafe { reduce.exchange_bf16(send, slots, bytes / 2, stream) }
            .map_err(|error| format!("nccl exchange: {error}"))
    }

    fn scatter_on_device(
        &self,
        stream: u64,
        send: u64,
        slots: u64,
        bytes: usize,
    ) -> Result<(), String> {
        let world = self.host.world;
        if bytes == 0 || !bytes.is_multiple_of(2 * world) {
            return Err(format!(
                "{bytes} bytes are not {world} chunks of whole bf16 words, the unit the exchange \
                 moves"
            ));
        }
        let mut reduce = self
            .reduce
            .reduce
            .lock()
            .map_err(|_| "the communicator lock is poisoned".to_owned())?;
        // SAFETY: Exchange::scatter_on_device's contract is scatter_bf16's: `send` holds `bytes`
        // and `slots` as many on the communicator's device, `stream` is a stream there, and every
        // rank enqueues the same exchanges in the same order.
        unsafe { reduce.scatter_bf16(send, slots, bytes / 2 / world, stream) }
            .map_err(|error| format!("nccl scatter: {error}"))
    }

    fn check_device(&self) -> Result<(), String> {
        #[cfg(feature = "rdma")]
        if self.doorbell.is_some() {
            return sparkflash_kernels::write_exchange::check()
                .map_err(|error| format!("direct-RDMA exchange: {error}"));
        }
        Ok(())
    }
}

/// The exchange one rank of a TP group runs: the TCP exchange over `config`, with the run's NCCL
/// communicator as its device leg unless [COMM_REDUCE_ENV] says `tcp`.
///
/// # Errors
/// A transport that does not form, a switch value this arm has no path for, or an NCCL open that
/// refuses (which aborts every rank: there is no silent fall back to the host path).
pub fn connect_rank(
    config: &TcpConfig,
    payload_words: usize,
    run: &str,
) -> Result<Box<dyn Exchange>, String> {
    let host = TcpExchange::connect(config)?;
    if nccl_requested(host.world)? {
        Ok(Box::new(NcclExchange::open(host, payload_words, run)?))
    } else {
        Ok(Box::new(host))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_world_of_one_is_the_identity() {
        let solo = Solo;
        let mut values = vec![1.5f32, -2.0, 0.25];
        solo.all_reduce_f32(&mut values).expect("the identity");
        assert_eq!(values, vec![1.5, -2.0, 0.25]);
        let mut out = vec![0f32; 3];
        solo.all_gather_f32(&values, &mut out).expect("the copy");
        assert_eq!(out, values);
        assert!(solo.all_gather_f32(&values, &mut [0f32; 2]).is_err());
    }

    /// Four ranks in one process on loopback: the exchange, the ascending-rank sum and the
    /// gather order, without a cluster.
    #[test]
    fn four_loopback_ranks_reduce_and_gather_the_same_way() {
        let base_port = 29_611u16;
        let mut handles = Vec::new();
        for rank in 0..4usize {
            handles.push(std::thread::spawn(
                move || -> Result<(Vec<f32>, Vec<f32>), String> {
                    let config = TcpConfig::loopback4(rank, base_port);
                    let exchange = TcpExchange::connect(&config)?;
                    let mut values = vec![(rank as f32) + 1.0, 2.0 * rank as f32, -1.0];
                    exchange.all_reduce_f32(&mut values)?;
                    let mut gathered = vec![0f32; 12];
                    exchange.all_gather_f32(&values, &mut gathered)?;
                    Ok((values, gathered))
                },
            ));
        }
        let mut reduced = Vec::new();
        let mut gathered = Vec::new();
        let mut failures = Vec::new();
        for (rank, handle) in handles.into_iter().enumerate() {
            match handle.join().expect("the rank thread") {
                Ok((values, all)) => {
                    reduced.push(values);
                    gathered.push(all);
                }
                Err(error) => failures.push(format!("rank {rank}: {error}")),
            }
        }
        assert!(failures.is_empty(), "{failures:?}");
        // The first element sums 1 + 2 + 3 + 4, the second 0 + 2 + 4 + 6, the third four -1s.
        for values in &reduced {
            assert_eq!(values, &vec![10.0, 12.0, -4.0]);
        }
        // Every rank gathered in rank order, so all four buffers are the same.
        for all in &gathered {
            assert_eq!(all, &gathered[0]);
            assert_eq!(&all[0..3], &[10.0, 12.0, -4.0]);
        }
    }
}

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

//! A serve-owned control channel: Hello/Ready/Request/Token/End/Error frames.
//!
//! One message stream per request, separate from a model's data collective, so rank 0 can admit,
//! broadcast and cancel without changing the transport the forward uses. It is a transport
//! concern, so it lives here rather than in the CLI: the DeepSeek host group and the Inkling
//! device group share the frames, the hub and the worker.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// How long a rank's control socket may block on one frame.
///
/// The timeout is a liveness guard for a dead peer, not a performance budget: a step between
/// tokens can be tens of seconds on a device arm. It is deliberately much larger than one token.
pub const CONTROL_READ_TIMEOUT: Duration = Duration::from_secs(600);

/// How long a worker pauses between control-connect attempts while rank 0's hub is still binding.
pub const CONTROL_CONNECT_RETRY: Duration = Duration::from_millis(100);

// ---------------------------------------------------------------------------------------
// TP=4 serve control channel
// ---------------------------------------------------------------------------------------
//
// The engine collective and the serve control channel are deliberately separate. The engine's
// TcpTransport carries only its data collective frames (Hello/Data/Shutdown) and its send/recv
// are not part of the Transport trait; the serve protocol is a small length-prefixed JSON frame
// stream on its own port. Rank 0 is the hub: it accepts every worker's Hello, sends Ready only
// after all ranks declared ready, then broadcasts Request/Token/End or an Error. Workers never
// detokenize and never sample: they receive the prompt ids once per request and the sampled token
// ids one per step, and each runs the same forward step on its own SequenceState.

/// The largest control frame accepted: a device tick's prompt id lists (rank 0 stops admitting at
/// [`TP4_TICK_PROMPT_IDS`] ids) and a small error message fit below this. A frame larger than this
/// is a protocol error, not an allocation request.
pub const TP4_CONTROL_MAX_FRAME: usize = 8 << 20;

/// The largest canvas a [`Tp4ControlFrame::DevicePixels`] frame carries: the largest span
/// (`vision_max_n_token` tokens) is about 1024 cells of 42 x 42 RGB pixels, 5.4 MB.
pub const TP4_CONTROL_MAX_PIXELS: usize = 16 << 20;

/// The prompt ids one device tick's admissions carry at most (about 7 bytes each in JSON): a
/// request past this joins at the next tick.
pub const TP4_TICK_PROMPT_IDS: usize = 768 << 10;

/// One message on the serve-owned control channel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tp4ControlFrame {
    /// A worker's first frame: which rank this connection is.
    Hello {
        /// The worker rank, 1..world_size.
        rank: u32,
    },
    /// Rank 0 -> worker: every rank has declared ready; requests may start.
    Ready,
    /// Rank 0 -> every worker: one request's prompt ids and generation budget.
    Request {
        /// The tokens rank 0 rendered (the chat template is already applied).
        prompt_ids: Vec<u32>,
        /// The request's clamped generation budget.
        max_tokens: u32,
        /// The prompt-logprobs top count the request asked for, when it asked for any.
        ///
        /// `None` is "no prompt logprobs"; `Some(0)` is "each prompt position's own token
        /// only". The whole-vocabulary -1 selection is refused before a job is admitted: the
        /// bounded candidate exchange cannot carry it.
        #[serde(default)]
        prompt_logprobs: Option<u32>,
        /// Whether the request decodes speculatively ([Tp4Sampling::speculative]): every rank
        /// takes it from here, so every rank drafts and verifies the same steps.
        #[serde(default)]
        speculative: bool,
        /// Leading prompt tokens every rank's window already holds, from rank 0's plan: the ranks
        /// prefill only what follows them. One number for the group, because a rank that reused a
        /// different prefix would run different collectives in the same step.
        #[serde(default)]
        cached_tokens: u32,
        /// The prompt's image spans, in prompt order: every rank writes the images' rows over
        /// these positions, so every rank must know them. The canvases do not travel; the rank
        /// that encodes holds them.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        images: Vec<Tp4DeviceImage>,
        /// A sampled request's sampler: every rank's drafter perturbs its picks with the noise
        /// rank 0's sample takes, so every rank drafts the same ids. `None` (and a frame from a
        /// rank 0 that predates the field) is greedy.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sampler: Option<Tp4FrameSampler>,
    },
    /// Rank 0 -> every worker: the sampled token for the next lockstep decode.
    Token {
        /// The token rank 0 sampled; workers must not sample their own.
        token_id: u32,
    },
    /// Rank 0 -> every worker, before each device scheduler tick: the requests that join and
    /// the sequences that end there (a stop string matched, or the client left).
    DeviceTick {
        /// The new requests, in arrival order.
        admit: Vec<Tp4DeviceAdmit>,
        /// The sequence ids that end before the tick runs.
        cancel: Vec<u64>,
        /// The tick holds a long prefill back for a decode step (rank 0's clock).
        #[serde(default)]
        hold_prefill: bool,
    },
    /// Rank 0 -> the vision tower's rank only, before the DeviceTick that admits the request: the
    /// canvas of image `image` of sequence `seq`, as `bytes` raw RGB bytes that follow the frame on
    /// the wire (a canvas is megabytes; JSON would triple it).
    DevicePixels {
        /// The sequence the image belongs to.
        seq: u64,
        /// The image's index in the admission's `images`.
        image: u32,
        /// The canvas bytes after the frame.
        bytes: u64,
    },
    /// Rank 0 -> every worker of a fused group: lane `lane` takes a request's prompt. A lane that
    /// still holds an earlier request forgets it first, on every rank alike, unless `keep` says
    /// rank 0's lane kept it to continue.
    LaneRequest {
        /// The lane, `0..lanes`, in the order rank 0's pool opened them.
        lane: u32,
        /// The tokens rank 0 rendered (the chat template is already applied).
        prompt_ids: Vec<u32>,
        /// The prompt's image spans, as [Tp4ControlFrame::Request] carries them.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        images: Vec<Tp4DeviceImage>,
        /// Whether the lane keeps what it holds: the prompt then continues it, or the lane
        /// rewinds or starts over itself, as rank 0's lane does with the same history.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        keep: bool,
        /// The request's sampler, as [Tp4ControlFrame::Request] carries it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sampler: Option<Tp4FrameSampler>,
    },
    /// Rank 0 -> every worker of a fused group: lane `lane` is idle and gives up what it holds,
    /// so another lane can grow; its next request starts over.
    LaneRelease {
        /// The lane, as [Tp4ControlFrame::LaneRequest] names it.
        lane: u32,
    },
    /// Rank 0 -> every worker of a fused group: one step of these lanes, each consuming the token
    /// rank 0 sampled for it, in this order - the order is the step's row order on every rank.
    LaneStep {
        /// `(lane, token)` a member.
        members: Vec<(u32, u32)>,
    },
    /// Rank 0 -> every worker of a fused group: lane `lane`'s caller never used the last `count`
    /// answers of its run, so the lane rolls them back within its round (the shared engine's
    /// give-back), before the frame that follows.
    LaneGiveBack {
        /// The lane, as [Tp4ControlFrame::LaneRequest] names it.
        lane: u32,
        /// The answers to roll back.
        count: u32,
    },
    /// The request ended normally; every worker leaves the request loop.
    End,
    /// The request or the group failed; every worker leaves the process.
    Error {
        /// The same code the HTTP error event uses, e.g. engine_gone.
        code: String,
        /// An operator-facing message.
        message: String,
    },
}

impl Tp4ControlFrame {
    /// A short name for an unexpected-frame message.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Hello { .. } => "Hello",
            Self::Ready => "Ready",
            Self::Request { .. } => "Request",
            Self::Token { .. } => "Token",
            Self::DeviceTick { .. } => "DeviceTick",
            Self::DevicePixels { .. } => "DevicePixels",
            Self::LaneRequest { .. } => "LaneRequest",
            Self::LaneRelease { .. } => "LaneRelease",
            Self::LaneStep { .. } => "LaneStep",
            Self::LaneGiveBack { .. } => "LaneGiveBack",
            Self::End => "End",
            Self::Error { .. } => "Error",
        }
    }
}

/// A request's sampler as [Tp4ControlFrame::Request] carries it, in [Tp4DeviceAdmit]'s fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tp4FrameSampler {
    /// The temperature's f32 bits (0 is greedy); bits keep the frame `Eq`.
    pub temperature_bits: u32,
    /// The nucleus threshold's f32 bits.
    pub top_p_bits: u32,
    /// The candidate count (0 keeps every token).
    pub top_k: u32,
    /// The sampler seed every rank draws its noise from.
    pub seed: u64,
    /// The repetition penalty's f32 bits: 1.0, and a frame without the field, is none. A group
    /// whose workers pick what rank 0 picks (their picks decide what a verify keeps) needs the
    /// penalties as much as the draw.
    #[serde(default = "Tp4FrameSampler::no_repetition_penalty")]
    pub repetition_penalty_bits: u32,
    /// The presence penalty's f32 bits (0.0 is none).
    #[serde(default)]
    pub presence_penalty_bits: u32,
    /// The frequency penalty's f32 bits (0.0 is none).
    #[serde(default)]
    pub frequency_penalty_bits: u32,
}

impl Tp4FrameSampler {
    /// The bits of a repetition penalty that changes nothing.
    #[must_use]
    pub fn no_repetition_penalty() -> u32 {
        1.0f32.to_bits()
    }
}

/// One request on the device path, with its sampler, as a device tick admits it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tp4DeviceAdmit {
    /// The sequence id rank 0 gave the request: later ticks cancel it by this id.
    pub seq: u64,
    /// The tokens rank 0 rendered (the chat template is already applied).
    pub prompt_ids: Vec<u32>,
    /// The request's clamped generation budget.
    pub max_tokens: u32,
    /// The temperature's f32 bits (0 is greedy); bits keep the frame `Eq`.
    pub temperature_bits: u32,
    /// The nucleus threshold's f32 bits.
    pub top_p_bits: u32,
    /// The candidate count (0 keeps every token).
    pub top_k: u32,
    /// The sampler seed every rank draws its noise from.
    pub seed: u64,
    /// The ids that end the request once emitted.
    pub stop_ids: Vec<u32>,
    /// The prompt's images in prompt order; the canvases go to the tower rank in
    /// [`Tp4ControlFrame::DevicePixels`] frames.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<Tp4DeviceImage>,
    /// The JSON Schema the document must satisfy, when the request asked for structured output.
    /// Every rank rebuilds the constraint from this, so it is the schema and not a pointer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraint: Option<String>,
}

/// One image span of an admission: where it sits in the prompt and the canvas it is made from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tp4DeviceImage {
    /// The span's first position.
    pub start: u64,
    /// Token rows.
    pub n_llm_h: u32,
    /// Token columns.
    pub n_llm_w: u32,
    /// Canvas height in pixels.
    pub best_height: u32,
    /// Canvas width in pixels.
    pub best_width: u32,
    /// The canvas's content key, hex.
    pub key: String,
}

impl Tp4DeviceImage {
    /// A content key as the frame's 64 lowercase hex digits.
    #[must_use]
    pub fn hex_key(key: &[u8; 32]) -> String {
        key.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// The content key the frame's hex digits spell.
    ///
    /// # Errors
    /// A key that is not 64 hex digits.
    pub fn key_bytes(&self) -> Result<[u8; 32], String> {
        let hex = self.key.as_bytes();
        if hex.len() != 64 {
            return Err(format!("a key of {} digits, not 64", hex.len()));
        }
        let mut key = [0u8; 32];
        for (byte, pair) in key.iter_mut().zip(hex.as_chunks::<2>().0) {
            let pair = std::str::from_utf8(pair).map_err(|error| error.to_string())?;
            *byte = u8::from_str_radix(pair, 16).map_err(|_| "the key is not hex".to_owned())?;
        }
        Ok(key)
    }
}

/// A typed control-channel refusal or transport failure.
#[derive(Debug)]
pub enum Tp4ControlError {
    /// A socket read/write failed.
    Io(std::io::Error),
    /// A frame was not valid JSON.
    Json(serde_json::Error),
    /// Rank 0 waited for a rank that never connected.
    MissingRank {
        /// The world size the group expected.
        world: usize,
        /// How many ranks did connect.
        joined: usize,
        /// The deadline rank 0 waited.
        waited: Duration,
    },
    /// A control socket timed out while a specific stage was expected.
    ///
    /// Reserved for a stage deadline the reader does not yet enforce; kept in the wire vocabulary
    /// so a peer's failure has a stable name.
    #[allow(dead_code)]
    Timeout {
        /// The protocol stage that timed out.
        stage: &'static str,
    },
    /// A worker could not reach rank 0's control hub inside the startup budget.
    ConnectTimeout {
        /// Rank 0's control address the worker kept trying.
        address: SocketAddr,
        /// How long the worker kept retrying before giving up.
        waited: Duration,
    },
    /// A frame was received before the group was Ready.
    NotReady {
        /// What the receiver was waiting for.
        expected: &'static str,
    },
    /// A frame arrived that the protocol did not allow at that point.
    UnexpectedFrame {
        /// What the receiver expected.
        expected: &'static str,
        /// What it got, boxed so the error stays small beside the frames it reports.
        got: Box<Tp4ControlFrame>,
    },
    /// A Hello named a rank outside the group or a rank that already joined.
    BadRank {
        /// The offending rank.
        rank: u32,
        /// The world size.
        world: usize,
    },
    /// A frame declared a payload larger than the protocol cap.
    FrameTooLarge {
        /// The declared payload length.
        bytes: usize,
    },
}

impl std::fmt::Display for Tp4ControlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "control I/O: {error}"),
            Self::Json(error) => write!(f, "control frame is not valid JSON: {error}"),
            Self::MissingRank {
                world,
                joined,
                waited,
            } => write!(
                f,
                "TP=4 rank 0 waited {waited:?} for {world} ranks and only {joined} joined; the missing rank is a startup refusal, not a single-rank fallback"
            ),
            Self::ConnectTimeout { address, waited } => write!(
                f,
                "control connect to {address} was still refused after {waited:?}; rank 0's hub was not listening within the TP=4 startup budget"
            ),
            Self::Timeout { stage } => write!(f, "control timeout while waiting for {stage}"),
            Self::NotReady { expected } => {
                write!(f, "control frame {expected} arrived before Ready")
            }
            Self::UnexpectedFrame { expected, got } => {
                write!(f, "control expected {expected}, got {}", got.kind())
            }
            Self::BadRank { rank, world } => {
                write!(f, "control Hello rank {rank} is outside a world of {world}")
            }
            Self::FrameTooLarge { bytes } => {
                write!(f, "control frame declares {bytes} bytes, above the cap")
            }
        }
    }
}

impl std::error::Error for Tp4ControlError {}

impl From<std::io::Error> for Tp4ControlError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for Tp4ControlError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

/// One length-prefixed JSON control frame, as it goes on the wire.
pub fn control_frame_bytes(frame: &Tp4ControlFrame) -> Result<Vec<u8>, Tp4ControlError> {
    let payload = serde_json::to_vec(frame)?;
    let length = u32::try_from(payload.len()).map_err(|_| Tp4ControlError::FrameTooLarge {
        bytes: payload.len(),
    })?;
    let mut bytes = Vec::with_capacity(4 + payload.len());
    bytes.extend_from_slice(&length.to_be_bytes());
    bytes.extend_from_slice(&payload);
    Ok(bytes)
}

/// Writes one length-prefixed JSON control frame.
pub fn write_control_frame(
    stream: &mut TcpStream,
    frame: &Tp4ControlFrame,
) -> Result<(), Tp4ControlError> {
    // One write per frame: with TCP_NODELAY a frame is one segment, not a header the peer
    // waits on while the payload sits behind Nagle.
    stream.write_all(&control_frame_bytes(frame)?)?;
    stream.flush()?;
    Ok(())
}

/// A [`Tp4ControlFrame::DevicePixels`] frame and its canvas bytes, as they go on the wire.
pub fn control_pixels_bytes(
    seq: u64,
    image: u32,
    pixels: &[u8],
) -> Result<Vec<u8>, Tp4ControlError> {
    if pixels.len() > TP4_CONTROL_MAX_PIXELS {
        return Err(Tp4ControlError::FrameTooLarge {
            bytes: pixels.len(),
        });
    }
    let frame = Tp4ControlFrame::DevicePixels {
        seq,
        image,
        bytes: pixels.len() as u64,
    };
    let mut bytes = control_frame_bytes(&frame)?;
    bytes.extend_from_slice(pixels);
    Ok(bytes)
}

/// Reads the `bytes` canvas bytes that follow a [`Tp4ControlFrame::DevicePixels`] frame.
pub fn read_control_pixels(stream: &mut TcpStream, bytes: u64) -> Result<Vec<u8>, Tp4ControlError> {
    let length = usize::try_from(bytes).unwrap_or(usize::MAX);
    if length > TP4_CONTROL_MAX_PIXELS {
        return Err(Tp4ControlError::FrameTooLarge { bytes: length });
    }
    let mut pixels = vec![0u8; length];
    stream.read_exact(&mut pixels)?;
    Ok(pixels)
}

/// Whether a control read timed out with no request open, which is an idle wait rather than a
/// dead peer.
///
/// The socket's read timeout is a liveness guard for a stalled step; between requests it fires on
/// a healthy server that has simply received nothing. Only the timeout kinds mean "try again":
/// a reset or a broken pipe is still a failure.
pub fn control_read_is_idle(error: &Tp4ControlError) -> bool {
    matches!(
        error,
        Tp4ControlError::Io(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            )
    )
}

/// Reads one length-prefixed JSON control frame.
pub fn read_control_frame(stream: &mut TcpStream) -> Result<Tp4ControlFrame, Tp4ControlError> {
    let mut header = [0u8; 4];
    stream.read_exact(&mut header)?;
    let length = usize::try_from(u32::from_be_bytes(header)).unwrap_or(usize::MAX);
    if length == 0 || length > TP4_CONTROL_MAX_FRAME {
        return Err(Tp4ControlError::FrameTooLarge { bytes: length });
    }
    let mut payload = vec![0u8; length];
    stream.read_exact(&mut payload)?;
    Ok(serde_json::from_slice(&payload)?)
}

/// Rank 0's end of the control channel: one stream per worker, readiness, and broadcast.
#[derive(Debug)]
pub struct Tp4ControlHub {
    /// Index by worker rank; rank 0 is always None (it is the hub itself).
    pub streams: Vec<Option<TcpStream>>,
    /// The number of ranks in the group.
    pub world: usize,
    /// Whether every worker has declared Hello and Ready was sent.
    pub ready: bool,
}

impl Tp4ControlHub {
    /// Binds the control listener and waits for every worker's Hello.
    ///
    /// The acceptance loop uses a non-blocking listener plus a deadline so a missing rank is a
    /// typed error with the count that joined, not an indefinite accept. Ready is sent only after
    /// every rank has connected: that is what makes a Request-before-Ready a protocol violation
    /// rather than a race.
    pub fn start(
        address: SocketAddr,
        world: usize,
        timeout: Duration,
    ) -> Result<Self, Tp4ControlError> {
        if world == 0 {
            return Err(Tp4ControlError::BadRank { rank: 0, world });
        }
        Self::start_with_listener(TcpListener::bind(address)?, world, timeout)
    }

    /// Accepts on an already-bound listener; [Self::start] is this with the bind in front.
    ///
    /// A caller that needs the address before rank 0 starts - a test on an ephemeral port, or a
    /// launcher that reports the port it will use - binds the socket itself and hands it here.
    pub fn start_with_listener(
        listener: TcpListener,
        world: usize,
        timeout: Duration,
    ) -> Result<Self, Tp4ControlError> {
        if world == 0 {
            return Err(Tp4ControlError::BadRank { rank: 0, world });
        }
        listener.set_nonblocking(true)?;
        let deadline = Instant::now() + timeout;
        let mut streams: Vec<Option<TcpStream>> = (0..world).map(|_| None).collect();
        let mut joined = 0usize;
        while joined + 1 < world {
            if Instant::now() >= deadline {
                return Err(Tp4ControlError::MissingRank {
                    world,
                    joined,
                    waited: timeout,
                });
            }
            match listener.accept() {
                Ok((mut stream, _peer)) => {
                    stream.set_nonblocking(false)?;
                    stream.set_nodelay(true)?;
                    stream.set_read_timeout(Some(CONTROL_READ_TIMEOUT))?;
                    stream.set_write_timeout(Some(CONTROL_READ_TIMEOUT))?;
                    let frame = read_control_frame(&mut stream)?;
                    let Tp4ControlFrame::Hello { rank } = frame else {
                        return Err(Tp4ControlError::UnexpectedFrame {
                            expected: "Hello",
                            got: Box::new(frame),
                        });
                    };
                    let rank = usize::try_from(rank).unwrap_or(usize::MAX);
                    if rank == 0 || rank >= world || streams[rank].is_some() {
                        return Err(Tp4ControlError::BadRank {
                            rank: u32::try_from(rank).unwrap_or(u32::MAX),
                            world,
                        });
                    }
                    streams[rank] = Some(stream);
                    joined += 1;
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => return Err(Tp4ControlError::Io(error)),
            }
        }
        let mut hub = Self {
            streams,
            world,
            ready: false,
        };
        for stream in hub.streams.iter_mut().flatten() {
            write_control_frame(stream, &Tp4ControlFrame::Ready)?;
        }
        hub.ready = true;
        Ok(hub)
    }

    /// Broadcasts one frame to every worker.
    pub fn broadcast(&mut self, frame: &Tp4ControlFrame) -> Result<(), Tp4ControlError> {
        debug_assert_eq!(self.streams.len(), self.world);
        if !self.ready {
            return Err(Tp4ControlError::NotReady {
                expected: "broadcast",
            });
        }
        for stream in self.streams.iter_mut().flatten() {
            write_control_frame(stream, frame)?;
        }
        Ok(())
    }
}

/// One worker's end of the control channel.
pub struct Tp4ControlWorker {
    pub stream: TcpStream,
    /// This worker's rank.
    pub rank: u32,
    /// The world size, for validation and messages.
    pub world: usize,
    /// Whether rank 0's Ready has been received.
    pub ready: bool,
}

impl Tp4ControlWorker {
    /// Connects to rank 0 and sends Hello; Ready is awaited separately by handshake.
    ///
    /// Connection-refused is retried with a short fixed pause for the whole startup budget, because
    /// rank 0 binds its control hub only after its own checkpoint load: a worker whose load finished
    /// first arrives before the listener exists and must wait for it rather than fail. Any other
    /// connect error, and the budget running out, still ends the start with a typed error; running
    /// out names the peer address and the budget in `Tp4ControlError::ConnectTimeout`.
    pub fn connect(
        address: SocketAddr,
        rank: u32,
        world: usize,
        timeout: Duration,
    ) -> Result<Self, Tp4ControlError> {
        if rank == 0 || usize::try_from(rank).unwrap_or(usize::MAX) >= world {
            return Err(Tp4ControlError::BadRank { rank, world });
        }
        let deadline = Instant::now() + timeout;
        let stream = loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(Tp4ControlError::ConnectTimeout {
                    address,
                    waited: timeout,
                });
            }
            match TcpStream::connect_timeout(&address, remaining) {
                Ok(stream) => break stream,
                Err(error) if error.kind() == std::io::ErrorKind::ConnectionRefused => {
                    std::thread::sleep(CONTROL_CONNECT_RETRY.min(remaining));
                }
                Err(error) => {
                    if Instant::now() >= deadline {
                        return Err(Tp4ControlError::ConnectTimeout {
                            address,
                            waited: timeout,
                        });
                    }
                    return Err(Tp4ControlError::Io(error));
                }
            }
        };
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(CONTROL_READ_TIMEOUT))?;
        stream.set_write_timeout(Some(CONTROL_READ_TIMEOUT))?;
        Ok(Self {
            stream,
            rank,
            world,
            ready: false,
        })
    }

    /// Sends Hello and waits for rank 0's Ready.
    pub fn handshake(&mut self) -> Result<(), Tp4ControlError> {
        debug_assert!(usize::try_from(self.rank).unwrap_or(usize::MAX) < self.world);
        write_control_frame(
            &mut self.stream,
            &Tp4ControlFrame::Hello { rank: self.rank },
        )?;
        let frame = read_control_frame(&mut self.stream)?;
        match frame {
            Tp4ControlFrame::Ready => {
                self.ready = true;
                Ok(())
            }
            other => Err(Tp4ControlError::UnexpectedFrame {
                expected: "Ready",
                got: Box::new(other),
            }),
        }
    }

    /// Reads the next request-scoped frame. A frame before Ready is a typed refusal.
    pub fn next_frame(&mut self) -> Result<Tp4ControlFrame, Tp4ControlError> {
        if !self.ready {
            return Err(Tp4ControlError::NotReady {
                expected: "next frame",
            });
        }
        read_control_frame(&mut self.stream)
    }

    /// Reads the canvas bytes after a [`Tp4ControlFrame::DevicePixels`] frame.
    pub fn next_pixels(&mut self, bytes: u64) -> Result<Vec<u8>, Tp4ControlError> {
        read_control_pixels(&mut self.stream, bytes)
    }
}

/// Ends the process now, without the C exit handlers. CUDA's handlers tear the context down,
/// which waits for the kernels in flight, and a collective whose peer has gone never finishes:
/// a rank whose group failed would stay up, holding its memory and its port. Rust runs no
/// destructors on exit either, so only the standard streams need flushing.
pub fn exit_now(code: i32) -> ! {
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // SAFETY: _exit takes no pointers and does not return.
    unsafe { libc::_exit(code) }
}

/// Exits rank 0 when a worker's control socket closes. A dead rank must fail the server, not
/// leave its requests waiting in a collective.
///
/// # Errors
/// A stream that cannot be cloned for the watch, or a thread that cannot start.
pub fn watch_workers(hub: &Tp4ControlHub) -> Result<(), String> {
    for (rank, stream) in hub.streams.iter().enumerate() {
        let Some(stream) = stream else { continue };
        let mut probe = stream
            .try_clone()
            .map_err(|error| format!("rank {rank} control watch: {error}"))?;
        std::thread::Builder::new()
            .name(format!("sparkflash-watch-r{rank}"))
            .spawn(move || {
                // Workers send nothing after Hello: any read that is not a timeout is the end.
                let mut byte = [0u8; 1];
                loop {
                    match probe.read(&mut byte) {
                        Err(error)
                            if matches!(
                                error.kind(),
                                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                            ) => {}
                        Ok(0) | Err(_) => {
                            eprintln!(
                                "sparkflash serve: rank {rank} closed its control channel; the group cannot serve, exiting"
                            );
                            exit_now(1);
                        }
                        Ok(_) => {}
                    }
                }
            })
            .map_err(|error| format!("rank {rank} control watch: {error}"))?;
    }
    Ok(())
}

/// Exits a worker when rank 0's control socket closes, even while the worker waits in a
/// collective. A close after `ended` is set (rank 0 said End) is the group's clean end, and the
/// watch stops; `last_words` runs just before an exit.
///
/// # Errors
/// A stream that cannot be cloned for the watch, or a thread that cannot start.
pub fn watch_leader(
    stream: &TcpStream,
    rank: usize,
    ended: &'static std::sync::atomic::AtomicBool,
    last_words: fn(),
) -> Result<(), String> {
    let probe = stream
        .try_clone()
        .map_err(|error| format!("rank {rank} control watch: {error}"))?;
    std::thread::Builder::new()
        .name("sparkflash-watch-r0".to_owned())
        .spawn(move || {
            let mut byte = [0u8; 1];
            loop {
                match probe.peek(&mut byte) {
                    // Frames are waiting for the rank's own thread: look again once it has read
                    // them.
                    Ok(n) if n > 0 => std::thread::sleep(Duration::from_millis(200)),
                    Err(error)
                        if matches!(
                            error.kind(),
                            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                        ) => {}
                    _ => {
                        if ended.load(std::sync::atomic::Ordering::SeqCst) {
                            return;
                        }
                        eprintln!(
                            "sparkflash serve: rank {rank}: rank 0 closed the control channel, exiting"
                        );
                        last_words();
                        exit_now(1);
                    }
                }
            }
        })
        .map_err(|error| format!("rank {rank} control watch: {error}"))?;
    Ok(())
}

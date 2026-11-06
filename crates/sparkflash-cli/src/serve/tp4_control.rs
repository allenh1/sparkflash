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

//! The serve-owned TP=4 control channel: Hello/Ready/Request/Token/End/Error frames.
//!
//! This is one message stream a request, separate from the engine's data collective, so rank 0
//! can admit, broadcast and cancel without changing the transport the forward uses. The module is
//! included only under `--features tp4-serve`.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::{TP4_CONTROL_CONNECT_RETRY, TP4_CONTROL_READ_TIMEOUT};

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
pub(super) const TP4_CONTROL_MAX_FRAME: usize = 8 << 20;

/// The largest canvas a [`Tp4ControlFrame::DevicePixels`] frame carries: the largest span
/// (`vision_max_n_token` tokens) is about 1024 cells of 42 x 42 RGB pixels, 5.4 MB.
pub(super) const TP4_CONTROL_MAX_PIXELS: usize = 16 << 20;

/// The prompt ids one device tick's admissions carry at most (about 7 bytes each in JSON): a
/// request past this joins at the next tick.
pub(crate) const TP4_TICK_PROMPT_IDS: usize = 768 << 10;

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
    pub(super) fn kind(&self) -> &'static str {
        match self {
            Self::Hello { .. } => "Hello",
            Self::Ready => "Ready",
            Self::Request { .. } => "Request",
            Self::Token { .. } => "Token",
            Self::DeviceTick { .. } => "DeviceTick",
            Self::DevicePixels { .. } => "DevicePixels",
            Self::End => "End",
            Self::Error { .. } => "Error",
        }
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
        /// What it got.
        got: Tp4ControlFrame,
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
pub(super) fn control_frame_bytes(frame: &Tp4ControlFrame) -> Result<Vec<u8>, Tp4ControlError> {
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
pub(super) fn write_control_frame(
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
pub(super) fn control_pixels_bytes(
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
pub(super) fn read_control_pixels(
    stream: &mut TcpStream,
    bytes: u64,
) -> Result<Vec<u8>, Tp4ControlError> {
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
pub(super) fn control_read_is_idle(error: &Tp4ControlError) -> bool {
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
pub(super) fn read_control_frame(
    stream: &mut TcpStream,
) -> Result<Tp4ControlFrame, Tp4ControlError> {
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
pub(super) struct Tp4ControlHub {
    /// Index by worker rank; rank 0 is always None (it is the hub itself).
    pub(super) streams: Vec<Option<TcpStream>>,
    /// The number of ranks in the group.
    pub(super) world: usize,
    /// Whether every worker has declared Hello and Ready was sent.
    pub(super) ready: bool,
}

impl Tp4ControlHub {
    /// Binds the control listener and waits for every worker's Hello.
    ///
    /// The acceptance loop uses a non-blocking listener plus a deadline so a missing rank is a
    /// typed error with the count that joined, not an indefinite accept. Ready is sent only after
    /// every rank has connected: that is what makes a Request-before-Ready a protocol violation
    /// rather than a race.
    pub(super) fn start(
        address: SocketAddr,
        world: usize,
        timeout: Duration,
    ) -> Result<Self, Tp4ControlError> {
        if world == 0 {
            return Err(Tp4ControlError::BadRank { rank: 0, world });
        }
        let listener = TcpListener::bind(address)?;
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
                    stream.set_read_timeout(Some(TP4_CONTROL_READ_TIMEOUT))?;
                    stream.set_write_timeout(Some(TP4_CONTROL_READ_TIMEOUT))?;
                    let frame = read_control_frame(&mut stream)?;
                    let Tp4ControlFrame::Hello { rank } = frame else {
                        return Err(Tp4ControlError::UnexpectedFrame {
                            expected: "Hello",
                            got: frame,
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
    pub(super) fn broadcast(&mut self, frame: &Tp4ControlFrame) -> Result<(), Tp4ControlError> {
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
pub(super) struct Tp4ControlWorker {
    pub(super) stream: TcpStream,
    /// This worker's rank.
    pub(super) rank: u32,
    /// The world size, for validation and messages.
    pub(super) world: usize,
    /// Whether rank 0's Ready has been received.
    pub(super) ready: bool,
}

impl Tp4ControlWorker {
    /// Connects to rank 0 and sends Hello; Ready is awaited separately by handshake.
    ///
    /// Connection-refused is retried with a short fixed pause for the whole startup budget, because
    /// rank 0 binds its control hub only after its own checkpoint load: a worker whose load finished
    /// first arrives before the listener exists and must wait for it rather than fail. Any other
    /// connect error, and the budget running out, still ends the start with a typed error; running
    /// out names the peer address and the budget in `Tp4ControlError::ConnectTimeout`.
    pub(super) fn connect(
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
                    std::thread::sleep(TP4_CONTROL_CONNECT_RETRY.min(remaining));
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
        stream.set_read_timeout(Some(TP4_CONTROL_READ_TIMEOUT))?;
        stream.set_write_timeout(Some(TP4_CONTROL_READ_TIMEOUT))?;
        Ok(Self {
            stream,
            rank,
            world,
            ready: false,
        })
    }

    /// Sends Hello and waits for rank 0's Ready.
    pub(super) fn handshake(&mut self) -> Result<(), Tp4ControlError> {
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
                got: other,
            }),
        }
    }

    /// Reads the next request-scoped frame. A frame before Ready is a typed refusal.
    pub(super) fn next_frame(&mut self) -> Result<Tp4ControlFrame, Tp4ControlError> {
        if !self.ready {
            return Err(Tp4ControlError::NotReady {
                expected: "next frame",
            });
        }
        read_control_frame(&mut self.stream)
    }

    /// Reads the canvas bytes after a [`Tp4ControlFrame::DevicePixels`] frame.
    pub(super) fn next_pixels(&mut self, bytes: u64) -> Result<Vec<u8>, Tp4ControlError> {
        read_control_pixels(&mut self.stream, bytes)
    }
}

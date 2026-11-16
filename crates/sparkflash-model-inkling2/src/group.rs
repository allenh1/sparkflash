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

//! One rank of a four-rank Inkling-2-Small device group.
//!
//! This is the serve face of [crate::Tp4Run]: the batch walk takes a prompt and a decode count and
//! exits, while a server holds the group open, prefills once, and streams one token at a time with
//! the KV untouched. The arithmetic is the same code either way - the device arm owns the
//! schedule, the kernels and the three collectives - and this module owns only the rank agreement
//! a server adds.
//!
//! # The shape of a request
//!
//! Rank 0 owns HTTP, the tokenizer and the control channel. Every rank, rank 0 included, holds one
//! [InklingDeviceBackend] over its own converted rank file and mirrors every step; the device arm's
//! collectives are what force that. A request therefore runs as:
//!
//! 1. rank 0 broadcasts `Request` with the tokenizer's prompt ids and the prompt's image spans
//!    (where each image sits and its grid - not its pixels), then prefills its own rank; rank 0
//!    alone runs the vision tower, and every rank receives the image rows through one group sum;
//! 2. rank 0 samples a token (the host driver's sampler), broadcasts `Token`, then decodes it;
//! 3. the workers receive the same frames in the same order and run the same two steps, so every
//!    rank's sequence state advances identically and the collectives line up;
//! 4. when the request ends, rank 0 keeps its session (the host driver reuses a shared prefix) and
//!    the workers keep theirs; a session *dropped* before its next prefill broadcasts `End`, after
//!    which every rank discards its state and the next `Request` starts clean.
//!
//! With more than one lane the group runs the fused engine instead ([GroupLanes]): `LaneRequest`
//! and `LaneStep` carry a lane's prompt and every fused step's members in rank 0's order, so the
//! workers step the same rows in the same order.
//!
//! Rank 0 refuses what every rank would refuse - a prompt over the context, an image span the
//! tower cannot have made ([SpanCheck]) - before it broadcasts, so a bad request never reaches the
//! workers.
//!
//! # What is not claimed
//!
//! There is no drafter, and a worker that fails ends its process rather than reporting through
//! the control channel - rank 0 then stalls on the next collective until the transport's frame
//! deadline. That is stated in docs/models/inkling2-small-tp4.md rather than discovered in
//! production.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sparkflash_comm::control::{
    Tp4ControlFrame, Tp4ControlHub, Tp4ControlWorker, Tp4DeviceImage, Tp4FrameSampler,
    control_read_is_idle,
};
use sparkflash_comm::tcp::TcpConfig;
use sparkflash_model::backend::{BackendError, ModelBackend, ModelSession, Sampling, StepOutput};
use sparkflash_model::vision::{ImageGrid, ImageSpan, VisionImage};
use sparkflash_model::{ModelDescriptor, TpPlan};
use sparkflash_runtime::group::{broadcast, image_frames};
use sparkflash_runtime::{FusedEngine, GroupLanes, GroupWire, mirror_lanes};
use sparkflash_weights::tp::TpLayout;

use crate::collective::connect_rank;
use crate::config::Inkling2Config;
use crate::convert::RankFileSource;
use crate::device::{InklingDeviceBackend, InklingLanes};
use crate::forward::trace_on;
use crate::vision::SpanCheck;

/// The switch for rank 0's vision tower: `on` (the default) loads it from the checkpoint when the
/// checkpoint has one, `off` serves text only, and the group then refuses a request with images
/// as a missing capability. Any other value is refused at open.
pub const VISION_ENV: &str = "SPARKFLASH_INKLING_VISION";

/// Whether rank 0 loads the tower ([VISION_ENV]).
///
/// # Errors
/// A value that is neither `on` nor `off`.
fn vision_wanted() -> Result<bool, String> {
    match std::env::var(VISION_ENV) {
        Err(std::env::VarError::NotPresent) => Ok(true),
        Ok(value) if value == "on" => Ok(true),
        Ok(value) if value == "off" => Ok(false),
        Ok(value) => Err(format!("{VISION_ENV}={value}: expected on or off")),
        Err(error) => Err(format!("{VISION_ENV}: {error}")),
    }
}

/// The spans a worker reads from a frame, with no canvas: rank 0 encodes, and a worker's rows
/// arrive through the group's sum. Inkling packs an image's rows, so the stride is the width.
///
/// # Errors
/// A key that is not 64 hex digits, or a start past this host's address space.
fn frame_images(frames: Vec<Tp4DeviceImage>) -> Result<Vec<VisionImage>, String> {
    frames
        .into_iter()
        .enumerate()
        .map(|(index, frame)| {
            let number = index + 1;
            let key = frame
                .key_bytes()
                .map_err(|error| format!("image {number}: {error}"))?;
            let start = usize::try_from(frame.start)
                .map_err(|_| format!("image {number} starts at {}", frame.start))?;
            Ok(VisionImage {
                span: ImageSpan {
                    start,
                    grid: ImageGrid {
                        n_llm_h: frame.n_llm_h as usize,
                        n_llm_w: frame.n_llm_w as usize,
                        best_height: frame.best_height as usize,
                        best_width: frame.best_width as usize,
                        line_stride: frame.n_llm_w as usize,
                    },
                    key,
                },
                pixels: Arc::from(&[][..]),
            })
        })
        .collect()
}

/// How long a collective frame may wait for a peer.
///
/// The device arm's batch run uses the same deadline: a 594B-parameter rank's step is slow, and
/// the deadline is a liveness guard for a dead peer, not a per-token budget.
const FRAME_TIMEOUT: Duration = Duration::from_secs(900);

/// How long a rank may keep dialing a peer that has not bound its listener yet.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(180);

/// How long a rank waits for the control channel's handshake or one of its frames.
pub const CONTROL_TIMEOUT: Duration = Duration::from_secs(600);

/// One rank of a four-rank device group: its converted file, its collective and the control
/// endpoint the ranks agree on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupOptions {
    /// This process's rank, 0..world.
    pub rank: usize,
    /// The number of ranks.
    pub world: usize,
    /// Every rank's collective address, in rank order; each carries the port that rank binds.
    pub peers: Vec<SocketAddr>,
    /// Rank 0's control endpoint, which the workers dial.
    pub control: SocketAddr,
    /// The directory holding rank-R-of-W.bin.
    pub converted: PathBuf,
    /// The requests the group steps together. One is the serial mirror, which reuses a
    /// conversation's prefix; more is that many fused lanes ([InklingLanes]) on every rank, each
    /// step one walk over every waiting lane's row. Every rank must be given the same number.
    pub lanes: usize,
}

impl GroupOptions {
    /// The converted rank file this process reads.
    ///
    /// # Errors
    /// The directory holds no file for this rank and world.
    pub fn rank_file(&self) -> Result<PathBuf, String> {
        crate::rank_file(&self.converted, self.rank, self.world)
    }
}

/// Rank 0's opened backend, with the banner its boot line reports.
pub struct MasterRank {
    /// The rank-0 backend, ready to open sequences.
    pub backend: Box<dyn ModelBackend>,
    /// The bytes the rank's weights occupy on the device.
    pub weight_bytes: usize,
    /// The boot line's description of the arm.
    pub banner: String,
}

/// Opens rank 0: forms the collective, binds the control hub, and loads the rank's weights.
///
/// The order matters. The collective forms first on every rank, because a worker that dialed the
/// control channel before its peers bound their collectors would wait for a `Ready` that rank 0
/// cannot send until the collective is whole. The hub binds before the 97 GB load, so a worker
/// whose load finishes first finds its listener; then rank 0 reads its own file, and then the
/// vision tower from `checkpoint` (the released directory: the rank files hold only the text
/// model) when the checkpoint has one and [VISION_ENV] does not turn it off, and the DSpark
/// drafter's share from the same directory when the rank drafts with it.
///
/// # Errors
/// A rank file that does not open, a transport that does not connect, a peer that never joins,
/// a tower that does not load, or a device failure.
pub fn open_master(
    descriptor: ModelDescriptor,
    config: &Inkling2Config,
    group: &GroupOptions,
    context: usize,
    checkpoint: &Path,
) -> Result<MasterRank, String> {
    let vision = config.vision.is_some() && vision_wanted()?;
    let converted = group.rank_file()?;
    let transport = transport(group)?;
    let exchange = connect_rank(
        &transport,
        2 * config.hidden_size,
        &format!("{} TP{} context {context}", descriptor.id, group.world),
    )
    .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
    let hub = Tp4ControlHub::start(group.control, group.world, CONTROL_TIMEOUT)
        .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
    let layout = TpLayout {
        rank: group.rank,
        world_size: group.world,
    };
    let tp = TpPlan::new(group.world, group.rank)
        .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
    let source = RankFileSource::open(&converted, config, layout)
        .map_err(|error| format!("{}: {error}", converted.display()))?;
    let opened = std::time::Instant::now();
    let mut device = InklingDeviceBackend::open_rank_file(
        descriptor.clone(),
        config.clone(),
        &source,
        context,
        tp,
        exchange,
    )
    .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
    let drafter = load_drafter(&mut device, checkpoint, group)?;
    let weight_bytes = device.weight_bytes();
    let tower = if vision {
        let source = sparkflash_weights::safetensors::open_checkpoint(checkpoint)
            .map_err(|error| format!("{}: {error}", checkpoint.display()))?;
        let bytes = device.load_vision(&source).map_err(|error| {
            format!(
                "TP{}/rank{}: the vision tower: {error} (set {VISION_ENV}=off to serve text only)",
                group.world, group.rank
            )
        })?;
        format!(", vision tower ({bytes} bytes)")
    } else if config.vision.is_some() {
        format!(", vision tower off ({VISION_ENV}=off)")
    } else {
        String::new()
    };
    let check = device.span_check();
    let hub = Arc::new(Mutex::new(hub));
    let engine = if group.lanes > 1 {
        let lanes = device
            .lanes(group.lanes)
            .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
        let wire = InklingWire {
            context,
            check: check.clone(),
        };
        let arm = GroupLanes::new(lanes, Arc::clone(&hub), wire);
        Some(
            FusedEngine::start(Arc::new(arm), group.lanes)
                .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?,
        )
    } else {
        None
    };
    let banner = format!(
        "TP{}/rank{} {} on {} ({} weight bytes, opened in {} ms), context {context}, {}{tower}{drafter}, Inkling device arm",
        group.world,
        group.rank,
        descriptor.id,
        converted.display(),
        weight_bytes,
        opened.elapsed().as_millis(),
        lanes_label(group.lanes)
    );
    Ok(MasterRank {
        backend: Box::new(GroupBackend {
            descriptor,
            device,
            hub,
            context,
            engine,
            check,
        }),
        weight_bytes,
        banner,
    })
}

/// Loads this rank's share of the DSpark drafter from `checkpoint` when the rank drafts with it
/// (SPARKFLASH_INKLING_DRAFT=dspark), and answers the boot line's note of it.
fn load_drafter(
    device: &mut InklingDeviceBackend,
    checkpoint: &Path,
    group: &GroupOptions,
) -> Result<String, String> {
    if !device.wants_drafter() {
        return Ok(String::new());
    }
    let bytes = device.load_drafter(checkpoint).map_err(|error| {
        format!(
            "TP{}/rank{}: the DSpark drafter: {error} (set {}=off to step without it)",
            group.world,
            group.rank,
            crate::device::DRAFT_ENV
        )
    })?;
    Ok(format!(", DSpark drafter ({bytes} bytes)"))
}

/// The boot line's word for how many requests a group steps at once.
fn lanes_label(lanes: usize) -> String {
    if lanes > 1 {
        format!("{lanes} fused lanes")
    } else {
        "one request at a time".to_owned()
    }
}

/// Opens one worker rank and mirrors rank 0 until the control channel ends.
///
/// The control connection is a mirror of rank 0's *admissions*; the lockstep arithmetic is the
/// model collective above it. Connecting after the collective is what makes the two waits on
/// rank 0 (the cluster, then the ranks) never cross.
///
/// # Errors
/// A rank file that does not open, a transport that does not connect, a control channel that
/// ends, or a device failure.
pub fn run_worker(
    descriptor: &ModelDescriptor,
    config: &Inkling2Config,
    group: &GroupOptions,
    context: usize,
    checkpoint: &Path,
) -> Result<(), String> {
    let converted = group.rank_file()?;
    let transport = transport(group)?;
    let exchange = connect_rank(
        &transport,
        2 * config.hidden_size,
        &format!("{} TP{} context {context}", descriptor.id, group.world),
    )
    .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
    let mut control = Tp4ControlWorker::connect(
        group.control,
        u32::try_from(group.rank).unwrap_or(u32::MAX),
        group.world,
        CONTROL_TIMEOUT,
    )
    .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
    control
        .handshake()
        .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
    let layout = TpLayout {
        rank: group.rank,
        world_size: group.world,
    };
    let tp = TpPlan::new(group.world, group.rank)
        .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
    let opened = std::time::Instant::now();
    let source = RankFileSource::open(&converted, config, layout)
        .map_err(|error| format!("{}: {error}", converted.display()))?;
    let mut backend = InklingDeviceBackend::open_rank_file(
        descriptor.clone(),
        config.clone(),
        &source,
        context,
        tp,
        exchange,
    )
    .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
    let drafter = load_drafter(&mut backend, checkpoint, group)?;
    println!(
        "sparkflash serve: TP{}/rank{} {} on {} ({} weight bytes, opened in {} ms), context {context}, {}{drafter}, Inkling device arm worker",
        group.world,
        group.rank,
        descriptor.id,
        converted.display(),
        backend.weight_bytes(),
        opened.elapsed().as_millis(),
        lanes_label(group.lanes)
    );
    if group.lanes > 1 {
        let lanes = backend
            .lanes(group.lanes)
            .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
        let wire = InklingWire {
            context,
            check: backend.span_check(),
        };
        return mirror_lanes(&mut control, &lanes, &wire, group.lanes);
    }
    mirror(&mut control, &backend)
}

/// The transport description every rank of the group forms.
fn transport(group: &GroupOptions) -> Result<TcpConfig, String> {
    TcpConfig::with_addresses(group.rank, group.peers.clone())
        .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))
        .map(|config| {
            config
                .with_timeout(FRAME_TIMEOUT)
                .with_connect_timeout(CONNECT_TIMEOUT)
        })
}

/// Mirrors rank 0's request frames onto this rank's own session.
///
/// The four ranks must run the same steps in the same order, so the loop is the whole protocol:
/// a `Request` prefills, a `Token` decodes the token rank 0 sampled, and an `End` discards the
/// sequence because rank 0 has dropped its own.
fn mirror(control: &mut Tp4ControlWorker, backend: &dyn ModelBackend) -> Result<(), String> {
    let mut session: Option<Box<dyn ModelSession>> = None;
    loop {
        let frame = match control.next_frame() {
            Ok(frame) => frame,
            Err(error) if control_read_is_idle(&error) => continue,
            Err(error) => return Err(error.to_string()),
        };
        match frame {
            Tp4ControlFrame::Request {
                prompt_ids,
                images,
                sampler,
                ..
            } => {
                let images = frame_images(images)?;
                let held = match session.as_mut() {
                    Some(held) => held,
                    None => session.insert(
                        backend
                            .open_session()
                            .map_err(|error| format!("open a sequence: {error}"))?,
                    ),
                };
                held.set_sampling(frame_sampling(sampler));
                if trace_on() {
                    eprintln!("[inkling] serve worker prefill {:?}", prompt_ids);
                }
                held.prefill_images(&prompt_ids, &images)
                    .map_err(|error| format!("prefill: {error}"))?;
            }
            Tp4ControlFrame::Token { token_id } => {
                let held = session.as_mut().ok_or_else(|| {
                    "a Token frame arrived before a Request: the ranks are out of step".to_owned()
                })?;
                if trace_on() {
                    eprintln!("[inkling] serve worker token {token_id}");
                }
                held.decode(token_id)
                    .map_err(|error| format!("decode {token_id}: {error}"))?;
            }
            Tp4ControlFrame::End => {
                // Rank 0 dropped its sequence; this rank must not keep a history the next
                // Request could continue from, or a follow-up prompt would desync the group.
                session = None;
            }
            Tp4ControlFrame::Error { code, message } => {
                return Err(format!("rank 0 ended the group ({code}): {message}"));
            }
            other => {
                return Err(format!(
                    "the worker expected Request, Token, End or Error, not {}",
                    other.kind()
                ));
            }
        }
    }
}

/// What Inkling's group refuses and carries ([GroupWire]): rank 0 refuses a prompt over the
/// admitted context or with image spans the tower cannot have made ([admit]); a sampled request's
/// temperature and seed cross for the drafter's noise ([frame_sampler]); a worker's image spans
/// are Inkling's packed rows ([frame_images]).
pub struct InklingWire {
    context: usize,
    check: SpanCheck,
}

impl GroupWire for InklingWire {
    fn name(&self) -> &'static str {
        "inkling"
    }

    fn trace(&self) -> bool {
        trace_on()
    }

    fn admit(&self, tokens: &[u32], images: &[VisionImage]) -> Result<(), BackendError> {
        admit(self.context, &self.check, tokens, images)
    }

    fn sampler(&self, sampling: &Sampling) -> Option<Tp4FrameSampler> {
        frame_sampler(sampling)
    }

    fn sampling(&self, sampler: Option<Tp4FrameSampler>) -> Sampling {
        frame_sampling(sampler)
    }

    fn images(&self, frames: Vec<Tp4DeviceImage>) -> Result<Vec<VisionImage>, String> {
        frame_images(frames)
    }
}

/// What rank 0 refuses before it broadcasts a prompt: one longer than the admitted `context`, or
/// image spans every rank would refuse ([SpanCheck]). A refusal after the broadcast would leave
/// the workers inside a request rank 0 had abandoned.
fn admit(
    context: usize,
    check: &SpanCheck,
    tokens: &[u32],
    images: &[VisionImage],
) -> Result<(), BackendError> {
    if tokens.len() > context {
        return Err(BackendError::new(
            "context",
            format!(
                "the prompt needs {} positions, {context} remain",
                tokens.len()
            ),
        ));
    }
    check
        .check(tokens, images)
        .map_err(|error| BackendError::new("request", error))
}

/// The rank-0 backend: one device rank with the control hub every session broadcasts through.
struct GroupBackend {
    descriptor: ModelDescriptor,
    device: InklingDeviceBackend,
    hub: Arc<Mutex<Tp4ControlHub>>,
    /// The admitted context, so an over-long prompt is refused before it is broadcast.
    context: usize,
    /// The fused lanes, when the group steps more than one request at once: every session is
    /// then a handle on this engine rather than a serial sequence of its own.
    engine: Option<Arc<FusedEngine<GroupLanes<InklingLanes, InklingWire>>>>,
    /// The image spans every rank would refuse.
    check: SpanCheck,
}

impl ModelBackend for GroupBackend {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> sparkflash_model::Capabilities {
        self.device.capabilities()
    }

    fn fabric_mode(&self) -> Option<&'static str> {
        Some("tcp tp4 device")
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        if let Some(engine) = &self.engine {
            return Ok(Box::new(engine.open()?));
        }
        Ok(Box::new(GroupSession {
            inner: self.device.open_session()?,
            hub: Arc::clone(&self.hub),
            context: self.context,
            check: self.check.clone(),
            open: false,
            fed: Vec::new(),
            sampler: None,
        }))
    }
}

/// One request sequence on rank 0, mirrored to the group before every step.
///
/// The broadcast precedes the local step, never follows it: the collectives inside that step
/// block until every rank joins, and a rank that only learned of the step afterwards would never
/// arrive.
struct GroupSession {
    inner: Box<dyn ModelSession>,
    hub: Arc<Mutex<Tp4ControlHub>>,
    context: usize,
    check: SpanCheck,
    /// Whether a Request is in flight and not yet Ended; a held session says so between requests.
    open: bool,
    /// The tokens this request fed, for the trace line a gate compares with the batch arm.
    fed: Vec<u32>,
    /// The request's sampler as the next Request frame carries it ([frame_sampler]).
    sampler: Option<Tp4FrameSampler>,
}

impl GroupSession {
    /// Broadcasts one frame to the group.
    fn send(&self, frame: &Tp4ControlFrame) -> Result<(), BackendError> {
        broadcast(&self.hub, frame)
    }
}

impl ModelSession for GroupSession {
    fn set_sampling(&mut self, sampling: Sampling) {
        self.sampler = frame_sampler(&sampling);
        self.inner.set_sampling(sampling);
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        self.prefill_images(tokens, &[])
    }

    fn prefill_images(
        &mut self,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<StepOutput, BackendError> {
        // Refuse before the broadcast: a prompt the group cannot hold must not start a sequence
        // the workers would then have to unwind from inside a collective.
        admit(self.context, &self.check, tokens, images)?;
        if trace_on() {
            eprintln!("[inkling] serve rank 0 prefill {:?}", tokens);
        }
        self.fed.clear();
        self.send(&Tp4ControlFrame::Request {
            prompt_ids: tokens.to_vec(),
            max_tokens: u32::MAX,
            prompt_logprobs: None,
            speculative: false,
            cached_tokens: 0,
            images: image_frames(images),
            sampler: self.sampler,
        })?;
        self.open = true;
        self.inner.prefill_images(tokens, images)
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        self.send(&Tp4ControlFrame::Token { token_id: token })?;
        self.fed.push(token);
        if trace_on() {
            eprintln!("[inkling] serve rank 0 token {token}");
        }
        self.inner.decode(token)
    }

    fn reusable_prefix(&self) -> Option<&[u32]> {
        self.inner.reusable_prefix()
    }

    /// Rank 0's point. Every worker holds the same history and checkpoint from the same frames,
    /// so each rewinds to the same point when the prompt reaches it.
    fn rewind_prefix(&self) -> Option<&[u32]> {
        self.inner.rewind_prefix()
    }

    /// Rank 0's images. The workers' come from the same frames, keys included.
    fn held_images(&self) -> &[ImageSpan] {
        self.inner.held_images()
    }

    fn cached_tokens(&self) -> usize {
        self.inner.cached_tokens()
    }
}

impl Drop for GroupSession {
    fn drop(&mut self) {
        // A dropped sequence is the one case the workers cannot infer: a *finished* request keeps
        // its state for the driver's prefix reuse, so silence is the signal to wait and End is the
        // signal to forget.
        if !self.open {
            return;
        }
        if let Err(error) = self.send(&Tp4ControlFrame::End) {
            eprintln!("sparkflash serve: rank 0 could not end the group: {error}");
        }
        if trace_on() {
            eprintln!("[inkling] serve rank 0 fed {:?}", self.fed);
        }
    }
}

/// The greedy sampler every mirror rank uses; it never chooses a token the group follows, but a
/// session still wants a deterministic configuration before its first step.
fn greedy() -> Sampling {
    crate::Tp4Run::greedy()
}

/// A sampled request's sampler for the Request frame, `None` for a greedy one: the workers need
/// its temperature and seed for the drafter's noise ([crate::host::gumbel_sampler]); the
/// penalties stay on rank 0, which alone chooses the tokens.
fn frame_sampler(sampling: &Sampling) -> Option<Tp4FrameSampler> {
    let gumbel = crate::host::gumbel_sampler(sampling)?;
    Some(Tp4FrameSampler {
        temperature_bits: gumbel.temperature.to_bits(),
        top_p_bits: gumbel.top_p.to_bits(),
        top_k: gumbel.top_k,
        seed: gumbel.seed,
        repetition_penalty_bits: Tp4FrameSampler::no_repetition_penalty(),
        presence_penalty_bits: 0.0f32.to_bits(),
        frequency_penalty_bits: 0.0f32.to_bits(),
    })
}

/// A mirror rank's sampling from the Request frame: rank 0's temperature, cut and seed, greedy
/// without one.
fn frame_sampling(sampler: Option<Tp4FrameSampler>) -> Sampling {
    let mut sampling = greedy();
    if let Some(sampler) = sampler {
        sampling.temperature = f32::from_bits(sampler.temperature_bits);
        sampling.top_p = f32::from_bits(sampler.top_p_bits);
        sampling.top_k = i32::try_from(sampler.top_k).unwrap_or(0);
        sampling.seed = sampler.seed;
    }
    sampling
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;
    use std::time::{Duration, Instant};

    /// What the fake mirror recorded, in the order it recorded it.
    #[derive(Default)]
    struct Record {
        prefills: Vec<Vec<u32>>,
        decodes: Vec<u32>,
        sessions: usize,
    }

    struct FakeBackend {
        record: Arc<Mutex<Record>>,
        descriptor: ModelDescriptor,
    }

    struct FakeSession {
        record: Arc<Mutex<Record>>,
        history: Vec<u32>,
    }

    impl ModelBackend for FakeBackend {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.descriptor
        }

        fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
            self.record.lock().expect("the record").sessions += 1;
            Ok(Box::new(FakeSession {
                record: Arc::clone(&self.record),
                history: Vec::new(),
            }))
        }
    }

    impl ModelSession for FakeSession {
        fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
            self.record
                .lock()
                .expect("the record")
                .prefills
                .push(tokens.to_vec());
            self.history.extend_from_slice(tokens);
            Ok(StepOutput {
                logits: vec![0.0; 8],
                next_token: Some(1),
                draft: Vec::new(),
            })
        }

        fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
            self.record.lock().expect("the record").decodes.push(token);
            self.history.push(token);
            Ok(StepOutput {
                logits: vec![0.0; 8],
                next_token: Some(2),
                draft: Vec::new(),
            })
        }

        fn reusable_prefix(&self) -> Option<&[u32]> {
            Some(&self.history)
        }
    }

    /// Rank 0's own local session in these tests: it answers, and it is deliberately not the
    /// mirror under test, so a recorded decode is one the worker ran.
    struct SilentSession;

    impl ModelSession for SilentSession {
        fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
            let _ = tokens;
            Ok(StepOutput {
                logits: vec![0.0; 8],
                next_token: Some(1),
                draft: Vec::new(),
            })
        }

        fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
            let _ = token;
            Ok(StepOutput {
                logits: vec![0.0; 8],
                next_token: Some(2),
                draft: Vec::new(),
            })
        }

        fn reusable_prefix(&self) -> Option<&[u32]> {
            None
        }
    }

    /// Waits for the mirror to reach a state; the control channel is one-way for rank 0, so a
    /// test observes the worker rather than reading an acknowledgement.
    fn wait_for(mut ready: impl FnMut() -> bool, what: &str) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if ready() {
                return;
            }
            thread::sleep(Duration::from_millis(2));
        }
        panic!("the mirror did not {what} within five seconds");
    }

    fn fake(record: &Arc<Mutex<Record>>) -> Arc<FakeBackend> {
        Arc::new(FakeBackend {
            record: Arc::clone(record),
            descriptor: crate::descriptor(),
        })
    }

    #[test]
    fn the_mirror_follows_a_request_and_discards_it_on_end() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback listener");
        let address = listener.local_addr().expect("the bound address");
        let record = Arc::new(Mutex::new(Record::default()));
        let backend = fake(&record);
        let worker = {
            let backend = Arc::clone(&backend);
            thread::spawn(move || {
                let mut control = Tp4ControlWorker::connect(address, 1, 2, Duration::from_secs(5))
                    .expect("a control connection");
                control.handshake().expect("the handshake");
                let _ = mirror(&mut control, backend.as_ref());
            })
        };
        let hub = Arc::new(Mutex::new(
            Tp4ControlHub::start_with_listener(listener, 2, Duration::from_secs(5))
                .expect("the hub"),
        ));
        {
            let mut session = GroupSession {
                inner: Box::new(SilentSession),
                hub: Arc::clone(&hub),
                context: 4096,
                check: SpanCheck::new(None, false),
                open: false,
                fed: Vec::new(),
                sampler: None,
            };
            session.prefill(&[1, 2, 3]).expect("the prefill");
            session.decode(4).expect("the first decode");
            session.decode(5).expect("the second decode");
            wait_for(
                || record.lock().expect("the record").decodes == vec![4, 5],
                "mirror both decodes",
            );
        }
        {
            let mut session = GroupSession {
                inner: Box::new(SilentSession),
                hub: Arc::clone(&hub),
                context: 4096,
                check: SpanCheck::new(None, false),
                open: false,
                fed: Vec::new(),
                sampler: None,
            };
            session.prefill(&[9]).expect("the second prefill");
            wait_for(
                || record.lock().expect("the record").prefills == vec![vec![1, 2, 3], vec![9]],
                "mirror the second prefill",
            );
            wait_for(
                || record.lock().expect("the record").sessions == 2,
                "discard the ended sequence and open a fresh one",
            );
        }
        drop(hub);
        worker.join().expect("the worker finishes");
        let record = record.lock().expect("the record");
        assert_eq!(
            record.decodes,
            vec![4, 5],
            "End must not be mirrored as a decode"
        );
    }

    #[test]
    fn a_token_before_a_request_is_refused() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback listener");
        let address = listener.local_addr().expect("the bound address");
        let record = Arc::new(Mutex::new(Record::default()));
        let backend = fake(&record);
        let worker = {
            let backend = Arc::clone(&backend);
            thread::spawn(move || {
                let mut control = Tp4ControlWorker::connect(address, 1, 2, Duration::from_secs(5))
                    .expect("a control connection");
                control.handshake().expect("the handshake");
                mirror(&mut control, backend.as_ref())
            })
        };
        let mut hub = Tp4ControlHub::start_with_listener(listener, 2, Duration::from_secs(5))
            .expect("the hub");
        hub.broadcast(&Tp4ControlFrame::Token { token_id: 7 })
            .expect("the frame is written");
        let outcome = worker.join().expect("the worker finishes");
        let error = outcome.expect_err("a token with no request is refused");
        assert!(error.contains("before a Request"), "{error}");
    }

    #[test]
    fn a_group_reads_one_rank_file_per_rank() {
        let directory = tempfile::tempdir().expect("a temporary converted directory");
        std::fs::write(directory.path().join("rank-0-of-2.bin"), b"").expect("a rank file");
        let group = GroupOptions {
            rank: 0,
            world: 2,
            peers: vec![
                "127.0.0.1:29600".parse().expect("an address"),
                "127.0.0.1:29601".parse().expect("an address"),
            ],
            control: "127.0.0.1:29700".parse().expect("an address"),
            converted: directory.path().to_path_buf(),
            lanes: 1,
        };
        assert_eq!(
            group
                .rank_file()
                .expect("a rank file")
                .file_name()
                .and_then(std::ffi::OsStr::to_str),
            Some("rank-0-of-2.bin")
        );
    }
}

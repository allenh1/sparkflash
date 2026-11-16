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
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sparkflash_comm::control::{
    Tp4ControlFrame, Tp4ControlHub, Tp4ControlWorker, Tp4DeviceImage, Tp4FrameSampler,
    control_read_is_idle,
};
use sparkflash_comm::tcp::TcpConfig;
use sparkflash_model::backend::{
    BackendError, ModelBackend, ModelSession, Sampling, Speculation, StepOutput,
};
use sparkflash_model::vision::{ImageGrid, ImageSpan, VisionImage};
use sparkflash_model::{FusedArm, FusedEngine, ModelDescriptor, TpPlan};
use sparkflash_weights::tp::TpLayout;

use crate::collective::connect_rank;
use crate::config::Inkling2Config;
use crate::convert::RankFileSource;
use crate::device::{InklingDeviceBackend, InklingLanes};
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

/// A request's image spans as the control frames carry them.
fn image_frames(images: &[VisionImage]) -> Vec<Tp4DeviceImage> {
    let narrow = |value: usize| u32::try_from(value).unwrap_or(u32::MAX);
    images
        .iter()
        .map(|image| {
            let span = &image.span;
            Tp4DeviceImage {
                start: span.start as u64,
                n_llm_h: narrow(span.grid.n_llm_h),
                n_llm_w: narrow(span.grid.n_llm_w),
                best_height: narrow(span.grid.best_height),
                best_width: narrow(span.grid.best_width),
                key: Tp4DeviceImage::hex_key(&span.key),
            }
        })
        .collect()
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
        let arm = GroupLanes::new(lanes, Arc::clone(&hub), context, check.clone());
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
        return mirror_lanes(&mut control, &lanes, group.lanes);
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

/// Mirrors rank 0's lane frames onto this rank's own pool of `lanes` fused lanes.
///
/// The pool is opened up front, as rank 0's engine opens its own, so lane `n` is the same slot on
/// every rank. A `LaneRequest` forgets what the lane held, unless rank 0 kept its own, and
/// prefills it with rank 0's sampler; a `LaneStep` steps its
/// members in the frame's order, which is the step's row order on every rank and so the order the
/// collectives inside it line up in.
fn mirror_lanes<A: FusedArm>(
    control: &mut Tp4ControlWorker,
    arm: &A,
    lanes: usize,
) -> Result<(), String> {
    let mut pool: Vec<Option<A::Slot>> = Vec::with_capacity(lanes);
    for _ in 0..lanes {
        pool.push(Some(
            arm.open_slot()
                .map_err(|error| format!("open a lane: {error}"))?,
        ));
    }
    let out_of_step = |lane: u32| {
        format!(
            "lane {lane} is outside this rank's {lanes} lanes or already stepping: the ranks are out \
             of step (every rank must be given the same --concurrency)"
        )
    };
    loop {
        let frame = match control.next_frame() {
            Ok(frame) => frame,
            Err(error) if control_read_is_idle(&error) => continue,
            Err(error) => return Err(error.to_string()),
        };
        match frame {
            Tp4ControlFrame::LaneRequest {
                lane,
                prompt_ids,
                images,
                keep,
                sampler,
            } => {
                let images =
                    frame_images(images).map_err(|error| format!("lane {lane}: {error}"))?;
                let slot = pool
                    .get_mut(lane as usize)
                    .and_then(Option::as_mut)
                    .ok_or_else(|| out_of_step(lane))?;
                if !keep && arm.holds_request(slot) {
                    arm.renew(slot)
                        .map_err(|error| format!("lane {lane}: renew: {error}"))?;
                }
                arm.set_sampling(slot, frame_sampling(sampler));
                if trace_on() {
                    eprintln!("[inkling] serve worker lane {lane} prefill {prompt_ids:?}");
                }
                arm.prefill_images(slot, &prompt_ids, &images)
                    .map_err(|error| format!("lane {lane}: prefill: {error}"))?;
            }
            Tp4ControlFrame::LaneRelease { lane } => {
                let slot = pool
                    .get_mut(lane as usize)
                    .and_then(Option::as_mut)
                    .ok_or_else(|| out_of_step(lane))?;
                if trace_on() {
                    eprintln!("[inkling] serve worker lane {lane} release");
                }
                arm.release(slot)
                    .map_err(|error| format!("lane {lane}: release: {error}"))?;
            }
            Tp4ControlFrame::LaneStep { members } => {
                let mut stepping = Vec::with_capacity(members.len());
                for &(lane, token) in &members {
                    let slot = pool
                        .get_mut(lane as usize)
                        .and_then(Option::take)
                        .ok_or_else(|| out_of_step(lane))?;
                    stepping.push((slot, token));
                }
                if trace_on() {
                    eprintln!("[inkling] serve worker step {members:?}");
                }
                let answer = arm.step(&mut stepping);
                for ((slot, _), &(lane, _)) in stepping.into_iter().zip(&members) {
                    pool[lane as usize] = Some(slot);
                }
                answer.map_err(|error| format!("step {members:?}: {error}"))?;
            }
            Tp4ControlFrame::Error { code, message } => {
                return Err(format!("rank 0 ended the group ({code}): {message}"));
            }
            other => {
                return Err(format!(
                    "the fused worker expected LaneRequest, LaneRelease, LaneStep or Error, not {}",
                    other.kind()
                ));
            }
        }
    }
}

/// Rank 0's fused lanes: a lane arm whose every prompt and step is broadcast before it runs.
///
/// The engine thread is the only caller, so the frames leave in the order the steps run, and a
/// worker that mirrors them in that order runs the same walks with the same rows. Each slot
/// carries the lane number the workers know it by.
pub struct GroupLanes<A: FusedArm> {
    arm: A,
    hub: Arc<Mutex<Tp4ControlHub>>,
    /// The admitted context, so an over-long prompt is refused before it is broadcast.
    context: usize,
    /// The image spans every rank would refuse, refused before they are broadcast.
    check: SpanCheck,
    /// The next lane number [FusedArm::open_slot] hands out.
    next: AtomicU32,
}

/// One lane of [GroupLanes]: the inner arm's slot and the number the workers know it by. The
/// slot is lent to the inner arm for a step and back between steps.
pub struct GroupLane<S> {
    lane: u32,
    slot: Option<S>,
    /// Whether the engine renewed the slot since its last prompt: the workers renew theirs only
    /// then, so a kept lane continues on every rank alike.
    renewed: bool,
    /// The request's sampler for the workers ([frame_sampler]).
    sampler: Option<Tp4FrameSampler>,
}

impl<A: FusedArm> GroupLanes<A> {
    /// Wraps `arm` to broadcast through `hub`, refusing first a prompt longer than `context` or
    /// image spans `check` refuses.
    pub fn new(arm: A, hub: Arc<Mutex<Tp4ControlHub>>, context: usize, check: SpanCheck) -> Self {
        Self {
            arm,
            hub,
            context,
            check,
            next: AtomicU32::new(0),
        }
    }

    fn send(&self, frame: &Tp4ControlFrame) -> Result<(), BackendError> {
        broadcast(&self.hub, frame)
    }

    fn slot(lane: &mut GroupLane<A::Slot>) -> Result<&mut A::Slot, BackendError> {
        lane.slot
            .as_mut()
            .ok_or_else(|| BackendError::new("group", "a lane's slot is already stepping"))
    }
}

impl<A: FusedArm> FusedArm for GroupLanes<A> {
    type Slot = GroupLane<A::Slot>;

    fn open_slot(&self) -> Result<Self::Slot, BackendError> {
        let lane = self.next.fetch_add(1, Ordering::Relaxed);
        Ok(GroupLane {
            lane,
            slot: Some(self.arm.open_slot()?),
            renewed: false,
            sampler: None,
        })
    }

    fn holds_request(&self, slot: &Self::Slot) -> bool {
        slot.slot
            .as_ref()
            .is_some_and(|inner| self.arm.holds_request(inner))
    }

    /// Local: a worker renews its own lane when the next `LaneRequest` says this one was.
    fn renew(&self, slot: &mut Self::Slot) -> Result<(), BackendError> {
        self.arm.renew(Self::slot(slot)?)?;
        slot.renewed = true;
        Ok(())
    }

    fn set_sampling(&self, slot: &mut Self::Slot, sampling: Sampling) {
        slot.sampler = frame_sampler(&sampling);
        if let Some(inner) = slot.slot.as_mut() {
            self.arm.set_sampling(inner, sampling);
        }
    }

    fn prefill(&self, slot: &mut Self::Slot, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        self.prefill_images(slot, tokens, &[])
    }

    fn prefill_images(
        &self,
        slot: &mut Self::Slot,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<StepOutput, BackendError> {
        // Refuse before the broadcast, as the serial session does.
        admit(self.context, &self.check, tokens, images)?;
        if !self.arm.room(Self::slot(slot)?, Some(tokens)) {
            return Err(no_room(tokens.len()));
        }
        if trace_on() {
            eprintln!(
                "[inkling] serve rank 0 lane {} prefill {tokens:?}",
                slot.lane
            );
        }
        self.send(&Tp4ControlFrame::LaneRequest {
            lane: slot.lane,
            prompt_ids: tokens.to_vec(),
            images: image_frames(images),
            keep: !std::mem::take(&mut slot.renewed),
            sampler: slot.sampler,
        })?;
        self.arm.prefill_images(Self::slot(slot)?, tokens, images)
    }

    fn step(&self, members: &mut [(Self::Slot, u32)]) -> Result<Vec<StepOutput>, BackendError> {
        let lanes: Vec<(u32, u32)> = members
            .iter()
            .map(|(lane, token)| (lane.lane, *token))
            .collect();
        let mut stepping = Vec::with_capacity(members.len());
        for (lane, token) in members.iter_mut() {
            let inner = lane
                .slot
                .take()
                .ok_or_else(|| BackendError::new("group", "a lane's slot is already stepping"));
            match inner {
                Ok(inner) => stepping.push((inner, *token)),
                Err(error) => {
                    // Give back what was taken: the engine keeps the slots across a failed step.
                    for ((lane, _), (inner, _)) in members.iter_mut().zip(stepping) {
                        lane.slot = Some(inner);
                    }
                    return Err(error);
                }
            }
        }
        // Every rank grows the same lanes the same way, so a step without room fails here, before
        // the broadcast, and not on the workers.
        if stepping
            .iter()
            .any(|(inner, _)| !self.arm.room(inner, None))
        {
            for ((lane, _), (inner, _)) in members.iter_mut().zip(stepping) {
                lane.slot = Some(inner);
            }
            return Err(no_room(1));
        }
        if trace_on() {
            eprintln!("[inkling] serve rank 0 step {lanes:?}");
        }
        let answer = self
            .send(&Tp4ControlFrame::LaneStep { members: lanes })
            .and_then(|()| self.arm.step(&mut stepping));
        for ((lane, _), (inner, _)) in members.iter_mut().zip(stepping) {
            lane.slot = Some(inner);
        }
        answer
    }

    fn speculation(&self, slot: &Self::Slot) -> Speculation {
        slot.slot
            .as_ref()
            .map(|inner| self.arm.speculation(inner))
            .unwrap_or_default()
    }

    fn history<'s>(&self, slot: &'s Self::Slot) -> Option<&'s [u32]> {
        self.arm.history(slot.slot.as_ref()?)
    }

    fn room(&self, slot: &Self::Slot, prompt: Option<&[u32]>) -> bool {
        slot.slot
            .as_ref()
            .is_some_and(|inner| self.arm.room(inner, prompt))
    }

    /// Broadcast: every rank gives up the same lane's memory, so their pools stay alike.
    fn release(&self, slot: &mut Self::Slot) -> Result<(), BackendError> {
        if trace_on() {
            eprintln!("[inkling] serve rank 0 lane {} release", slot.lane);
        }
        self.send(&Tp4ControlFrame::LaneRelease { lane: slot.lane })?;
        self.arm.release(Self::slot(slot)?)?;
        slot.renewed = true;
        Ok(())
    }

    fn rewind_prefix<'s>(&self, slot: &'s Self::Slot) -> Option<&'s [u32]> {
        self.arm.rewind_prefix(slot.slot.as_ref()?)
    }

    fn held_images<'s>(&self, slot: &'s Self::Slot) -> &'s [ImageSpan] {
        slot.slot
            .as_ref()
            .map_or(&[], |inner| self.arm.held_images(inner))
    }

    fn cached_tokens(&self, slot: &Self::Slot) -> usize {
        slot.slot
            .as_ref()
            .map_or(0, |inner| self.arm.cached_tokens(inner))
    }

    fn max_batch(&self) -> usize {
        self.arm.max_batch()
    }
}

/// The refusal of a prompt of `rows` positions (or a step) the lanes' memory cannot hold beside
/// the sequences being served: the engine has already given up every idle one.
fn no_room(rows: usize) -> BackendError {
    BackendError::new(
        "context",
        format!(
            "the sessions' lane memory has no room for {rows} more positions beside the \
             sequences being served (runtime.lane_memory_mb / SPARKFLASH_LANE_MEMORY_MB sets the \
             pool)"
        ),
    )
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

/// Broadcasts one frame to the group.
fn broadcast(hub: &Mutex<Tp4ControlHub>, frame: &Tp4ControlFrame) -> Result<(), BackendError> {
    let kind = frame.kind();
    let mut hub = hub
        .lock()
        .map_err(|_| BackendError::new("group", "the control hub lock is poisoned"))?;
    hub.broadcast(frame)
        .map_err(|error| BackendError::new("group", format!("{kind}: {error}")))
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
    engine: Option<Arc<FusedEngine<GroupLanes<InklingLanes>>>>,
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

/// Whether the arm's trace is on.
fn trace_on() -> bool {
    std::env::var_os("SPARKFLASH_INKLING_TRACE").is_some()
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

    /// A toy lane arm: a slot is the tokens its lane has consumed, and the record is every
    /// prompt and step in the order the arm ran them.
    #[derive(Default)]
    struct ToyLanes {
        record: Mutex<Vec<String>>,
    }

    impl FusedArm for ToyLanes {
        type Slot = Vec<u32>;

        fn open_slot(&self) -> Result<Self::Slot, BackendError> {
            Ok(Vec::new())
        }

        fn holds_request(&self, slot: &Self::Slot) -> bool {
            !slot.is_empty()
        }

        fn renew(&self, slot: &mut Self::Slot) -> Result<(), BackendError> {
            slot.clear();
            Ok(())
        }

        fn set_sampling(&self, _slot: &mut Self::Slot, _sampling: Sampling) {}

        fn prefill(
            &self,
            slot: &mut Self::Slot,
            tokens: &[u32],
        ) -> Result<StepOutput, BackendError> {
            slot.extend_from_slice(tokens);
            self.record
                .lock()
                .expect("the record")
                .push(format!("prefill {tokens:?}"));
            Ok(StepOutput {
                logits: vec![0.0; 2],
                next_token: Some(1),
                draft: Vec::new(),
            })
        }

        fn step(&self, members: &mut [(Self::Slot, u32)]) -> Result<Vec<StepOutput>, BackendError> {
            let mut rows = Vec::new();
            for (slot, token) in members.iter_mut() {
                slot.push(*token);
                rows.push(slot.clone());
            }
            self.record
                .lock()
                .expect("the record")
                .push(format!("step {rows:?}"));
            Ok(members.iter().map(|_| StepOutput::default()).collect())
        }

        fn speculation(&self, _slot: &Self::Slot) -> Speculation {
            Speculation::default()
        }
    }

    #[test]
    fn the_lane_mirror_runs_rank_zeros_prompts_and_steps_in_order() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback listener");
        let address = listener.local_addr().expect("the bound address");
        let mirrored = Arc::new(ToyLanes::default());
        let worker = {
            let mirrored = Arc::clone(&mirrored);
            thread::spawn(move || {
                let mut control = Tp4ControlWorker::connect(address, 1, 2, Duration::from_secs(5))
                    .expect("a control connection");
                control.handshake().expect("the handshake");
                mirror_lanes(&mut control, mirrored.as_ref(), 2)
            })
        };
        let hub = Arc::new(Mutex::new(
            Tp4ControlHub::start_with_listener(listener, 2, Duration::from_secs(5))
                .expect("the hub"),
        ));
        let lanes = GroupLanes::new(
            ToyLanes::default(),
            Arc::clone(&hub),
            64,
            SpanCheck::new(None, false),
        );
        let mut first = lanes.open_slot().expect("lane 0");
        let mut second = lanes.open_slot().expect("lane 1");
        lanes.prefill(&mut first, &[1, 2]).expect("lane 0's prompt");
        lanes.prefill(&mut second, &[7]).expect("lane 1's prompt");
        // Rank 1 steps the members in rank 0's order, second lane first.
        let mut members = vec![(second, 8u32), (first, 3u32)];
        lanes.step(&mut members).expect("the fused step");
        let (mut first, _) = members.pop().expect("lane 0");
        // A lane that held a request forgets it on its next prompt, on rank 1 as on rank 0.
        assert!(lanes.holds_request(&first));
        lanes.renew(&mut first).expect("the lane renews");
        lanes
            .prefill(&mut first, &[5])
            .expect("lane 0's next prompt");
        let expected = vec![
            "prefill [1, 2]".to_owned(),
            "prefill [7]".to_owned(),
            "step [[7, 8], [1, 2, 3]]".to_owned(),
            "prefill [5]".to_owned(),
        ];
        wait_for(
            || *mirrored.record.lock().expect("the record") == expected,
            "mirror both prompts, the step and the renewed prompt",
        );
        assert_eq!(*lanes.arm.record.lock().expect("rank 0's record"), expected);
        // An unknown lane is the ranks out of step, and ends the worker.
        hub.lock()
            .expect("the hub")
            .broadcast(&Tp4ControlFrame::LaneStep {
                members: vec![(5, 1)],
            })
            .expect("the frame is written");
        let error = worker
            .join()
            .expect("the worker finishes")
            .expect_err("a lane outside the pool is refused");
        assert!(error.contains("out of step"), "{error}");
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

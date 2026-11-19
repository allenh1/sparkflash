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
//! 1. rank 0 broadcasts a lane's prompt with its image spans (where each image sits and its grid, not its pixels), then prefills its own rank; rank 0 alone runs the vision tower, and every rank receives the image rows through one group sum;
//! 2. rank 0 samples a token (the host driver's sampler), broadcasts the step, then runs it;
//! 3. the workers receive the same lane frames in the same order and run the same runtime over their own lanes, so every rank's sequence state advances identically and the collectives line up;
//! 4. when the request ends, rank 0 keeps its lane (the runtime reuses a shared prefix) and the workers keep theirs; a lane *dropped* before its next prompt broadcasts its end, after which every rank discards the sequence and the next prompt starts clean.
//!
//! Every group, one lane or several, runs the same [sparkflash_runtime::Runtime] on every rank,
//! and rank 0's decisions are broadcast in its order, so the workers run the same rows in the
//! same order. A text-only group runs the tick engine
//! ([sparkflash_runtime::TickServe::start_group], [sparkflash_runtime::mirror_ticks_until_the_group_ends]):
//! the frames carry the rows and the step shape rank 0 decided. A group whose rank 0 holds the
//! vision tower keeps the fused engine ([FusedEngine] over [GroupLanes]): rank 0's image encode
//! is a collective, the tick request frame follows rank 0's `start_prompt_slot` and so would
//! wait behind it, and the fused group's frame is ordered before the encode for exactly that
//! reason. [VISION_ENV] must be the same on every rank, like the rank files.
//!
//! Rank 0 refuses what every rank would refuse - a prompt over the context, an image span the
//! tower cannot have made ([SpanCheck]) - before it broadcasts, so a bad request never reaches the
//! workers.
//!
//! # What is not claimed
//!
//! The runtime's round path drafts exactly as it does alone. A worker that fails ends its process
//! rather than reporting through the control channel - rank 0 then stalls on the next collective
//! until the transport's frame deadline. That is stated in docs/models/inkling2-small-tp4.md
//! rather than discovered in production.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use sparkflash_comm::control::{Tp4ControlHub, Tp4ControlWorker, Tp4DeviceImage, Tp4FrameSampler};
use sparkflash_comm::tcp::TcpConfig;
use sparkflash_model::backend::{BackendError, ModelBackend, ModelSession, Sampling};
use sparkflash_model::vision::{ImageGrid, ImageSpan, VisionImage};
use sparkflash_model::{ModelDescriptor, TpPlan};
use sparkflash_runtime::{
    FusedEngine, GroupLanes, GroupWire, Runtime, TickServe, mirror_ticks_until_the_group_ends,
    mirror_until_the_group_ends, watched_hub,
};
use sparkflash_weights::tp::TpLayout;

use crate::collective::connect_rank;
use crate::config::Inkling2Config;
use crate::convert::RankFileSource;
use crate::device::{InklingDeviceBackend, InklingWalk};
use crate::forward::trace_on;
use crate::vision::SpanCheck;

/// The switch for rank 0's vision tower: `on` (the default) loads it from the checkpoint when the
/// checkpoint has one, `off` serves text only, and the group then refuses a request with images
/// as a missing capability. Any other value is refused at open. The name is neutral because
/// DeepSeek's device path reads it too; the per-model `SPARKFLASH_INKLING_VISION` is refused by
/// [sparkflash_model::profile::REFUSED_ENV] with this name in the message.
pub const VISION_ENV: &str = "SPARKFLASH_VISION";

/// What [VISION_ENV] asks for. The spellings are the profile's boolean ones, because the same
/// name is the profile's `diagnostics.vision` alias.
///
/// # Errors
/// A value that is not a boolean.
fn vision_request(value: Option<&str>) -> Result<bool, String> {
    match value {
        None => Ok(true),
        Some("on" | "1" | "true") => Ok(true),
        Some("off" | "0" | "false") => Ok(false),
        Some(other) => Err(format!("{VISION_ENV}={other}: expected on or off")),
    }
}

/// Whether rank 0 loads the tower ([VISION_ENV]).
///
/// # Errors
/// A value that is not a boolean, or a variable that is not UTF-8.
fn vision_wanted() -> Result<bool, String> {
    match std::env::var(VISION_ENV) {
        Err(std::env::VarError::NotPresent) => vision_request(None),
        Ok(value) => vision_request(Some(&value)),
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
    /// The requests the group steps together: that many lanes of the shared runtime on every
    /// rank, each step one walk over every waiting lane's row. Every rank must be given the
    /// same number.
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
    let wire = InklingWire {
        context,
        check: device.span_check(),
        serves_images: vision,
    };
    // Lanes run ahead of rank 0's answers, so a rank that dies or fails mid-job ends the whole
    // group rather than leave the rest in a collective (sparkflash_runtime::group). The engine
    // is the tick group unless this rank holds the tower: its image encode is a collective and
    // the tick group's request frame follows rank 0's start, so the encode would run before the
    // frame every worker's start waits for. [Engine] names the split; every rank must agree on
    // it, which is why [VISION_ENV] has to be the same on all of them.
    let (engine, engine_label) = if vision {
        let hub = watched_hub(hub)
            .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
        let lanes = device
            .runtime()
            .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
        let arm = GroupLanes::new(lanes, Arc::clone(&hub), wire);
        let engine = FusedEngine::start(Arc::new(arm), group.lanes)
            .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
        (Engine::Fused(engine), "the fused group, images on")
    } else {
        let walk = device
            .walk_arm()
            .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
        let engine = TickServe::start_group(walk, group.lanes, hub, wire)
            .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
        (Engine::Tick(engine), "the tick group, text only")
    };
    let banner = format!(
        "TP{}/rank{} {} on {} ({} weight bytes, opened in {} ms), context {context}, {}{tower}{drafter}, {engine_label}, Inkling device arm",
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
            engine,
        }),
        weight_bytes,
        banner,
    })
}

/// What a group's rank 0 serves on.
///
/// The tick group is the group's serve face: rank 0 decides the tick and the workers replay its
/// rows and step shape. A rank 0 that holds the vision tower keeps the fused group, whose request
/// frame is ordered before rank 0's image encode; the tick group's frame follows rank 0's start,
/// so the encode would run before the frame a worker's start waits for
/// ([sparkflash_runtime::TickGroup]). A text-only rank 0 (no tower in the checkpoint, or
/// [VISION_ENV]=off) takes the tick group and refuses an image request before any frame.
enum Engine {
    /// The tick engine: the scheduler, on every rank.
    Tick(Arc<TickServe<InklingWalk>>),
    /// The fused engine over [GroupLanes]: rank 0's image encode runs after the request frame.
    Fused(Arc<FusedEngine<GroupLanes<Runtime<InklingWalk>, InklingWire>>>),
}

impl Engine {
    fn open(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        match self {
            Self::Tick(engine) => Ok(Box::new(engine.open()?)),
            Self::Fused(engine) => Ok(Box::new(engine.open()?)),
        }
    }
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
    // The engine rank 0 chose, by the same rule and so with the same answer: a group that serves
    // images is the fused one (rank 0's encode is ordered after the request frame there), and a
    // text-only group is the tick one and replays its rows. [VISION_ENV] must be the same on
    // every rank; a rank that chose differently refuses rank 0's frames by name rather than
    // running a different row.
    let vision = config.vision.is_some() && vision_wanted()?;
    let engine_label = if vision {
        "the fused group, images on"
    } else {
        "the tick group, text only"
    };
    println!(
        "sparkflash serve: TP{}/rank{} {} on {} ({} weight bytes, opened in {} ms), context {context}, {}{drafter}, {engine_label}, Inkling device arm worker",
        group.world,
        group.rank,
        descriptor.id,
        converted.display(),
        backend.weight_bytes(),
        opened.elapsed().as_millis(),
        lanes_label(group.lanes)
    );
    let wire = InklingWire {
        context,
        check: backend.span_check(),
        serves_images: vision,
    };
    let lanes = backend
        .runtime()
        .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
    if vision {
        mirror_until_the_group_ends(&mut control, &lanes, &wire, group.lanes)
    } else {
        mirror_ticks_until_the_group_ends(&mut control, &lanes, &wire, group.lanes)
    }
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

/// What Inkling's group refuses and carries ([GroupWire]): rank 0 refuses a prompt over the
/// admitted context or with image spans the tower cannot have made ([admit]); a sampled request's
/// temperature and seed cross for the drafter's noise ([frame_sampler]); a worker's image spans
/// are Inkling's packed rows ([frame_images]).
pub struct InklingWire {
    context: usize,
    check: SpanCheck,
    /// Whether this group's rank 0 holds the tower. A text-only group is on the tick engine,
    /// whose request frame follows rank 0's start, so an image must be refused before the frame
    /// rather than fail rank 0's encode after it ([Engine]).
    serves_images: bool,
}

impl GroupWire for InklingWire {
    fn name(&self) -> &'static str {
        "inkling"
    }

    fn trace(&self) -> bool {
        trace_on()
    }

    fn admit(&self, tokens: &[u32], images: &[VisionImage]) -> Result<(), BackendError> {
        if !self.serves_images && !images.is_empty() {
            return Err(BackendError::new(
                "vision",
                format!(
                    "this group serves text only: its rank 0 holds no vision tower ({VISION_ENV}                      is off, or the checkpoint describes none)"
                ),
            ));
        }
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

/// The rank-0 backend: one device rank served by the shared runtime through an [Engine].
///
/// Every session is a handle on that engine, whether the group steps one request or several, so
/// the serial session and its mirror are gone: the runtime owns the prompt parts, the history,
/// resume, rewind and the round path here exactly as it does for a lane.
struct GroupBackend {
    descriptor: ModelDescriptor,
    device: InklingDeviceBackend,
    /// The runtime's lanes, every session a handle on this engine.
    engine: Engine,
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
        self.engine.open()
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

    /// A text-only group refuses an image before any frame: its rank 0 holds no tower, and the
    /// tick group's request frame follows rank 0's start, so a refusal after the frame would end
    /// the group rather than the request.
    #[test]
    fn a_text_only_group_refuses_an_image_before_any_frame() {
        use sparkflash_model::vision::{ImageGrid, ImageSpan, VisionImage};

        let wire = InklingWire {
            context: 64,
            check: SpanCheck::new(None, false),
            serves_images: false,
        };
        let image = VisionImage {
            span: ImageSpan {
                start: 1,
                grid: ImageGrid {
                    n_llm_h: 1,
                    n_llm_w: 1,
                    best_height: 1,
                    best_width: 1,
                    line_stride: 1,
                },
                key: [0u8; 32],
            },
            pixels: Arc::from(&[][..]),
        };
        let error = wire
            .admit(&[7, 9, 11], &[image])
            .expect_err("a text-only group takes no image");
        assert!(error.message().contains("text only"), "{error}");
        assert!(
            wire.admit(&[7, 9, 11], &[]).is_ok(),
            "a prompt with no image is admitted"
        );
    }

    /// The neutral switch keeps the split the Inkling name had: unset or `on` loads rank 0's
    /// tower, `off` serves text only, and anything else is refused with the switch named. The
    /// other boolean spellings are the profile's own (`SPARKFLASH_VISION` is its
    /// `diagnostics.vision` alias), so a `0` that turns the tower off for DeepSeek does here
    /// too.
    #[test]
    fn the_neutral_vision_switch_keeps_the_tower_and_text_only_split() {
        for on in [None, Some("on"), Some("1"), Some("true")] {
            assert_eq!(vision_request(on), Ok(true), "{on:?} loads the tower");
        }
        for off in [Some("off"), Some("0"), Some("false")] {
            assert_eq!(vision_request(off), Ok(false), "{off:?} serves text only");
        }
        let error = vision_request(Some("yes")).expect_err("an unknown value is refused");
        assert!(error.contains(VISION_ENV), "{error}");
        assert_eq!(VISION_ENV, "SPARKFLASH_VISION");
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

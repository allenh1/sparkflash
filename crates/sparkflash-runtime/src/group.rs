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

//! Lanes across a tensor-parallel group: rank 0 decides, every other rank replays.
//!
//! Every rank holds its share of the weights and runs the same walks, because the collectives
//! inside a walk block until every rank joins. Rank 0 alone runs the [crate::TickEngine]: it picks
//! which lanes step together, which rows a prompt chunk runs and the shape of a step, and
//! [TickGroup] broadcasts each decision (a lane's prompt, each chunk of it, a step's members in
//! their row order, a give-back, a release) *before* running it on its own rank. A worker runs
//! [mirror_tick_lanes] over its own lanes and replays the frames in the order they came, so lane
//! `n` is the same sequence on every rank and the collectives line up.
//!
//! This came from Inkling's group (`sparkflash-model-inkling2/src/group.rs`); what stays with an
//! arm is its [GroupWire]: what rank 0 refuses before it broadcasts, how a request's sampler
//! crosses the wire, and how a worker reads a prompt's image spans.
//!
//! A group that cannot stay in step stops as a whole rather than hang. Each frame is either
//! refused before it is broadcast, on rank 0 alone, or run on every rank. Once a frame has gone
//! out, a failure on rank 0's own half leaves the workers past it, inside collectives rank 0
//! will not join. Rank 0 then broadcasts [Tp4ControlFrame::Error] and ends its process
//! ([end_group]). A worker ends its own when its mirror fails or when rank 0's control socket
//! closes, even while it waits in a collective ([mirror_ticks_until_the_group_ends]). Rank 0 ends
//! when
//! a worker's socket closes ([watched_hub]).

use std::sync::{Arc, Mutex};

use sparkflash_comm::control::{
    Tp4ControlFrame, Tp4ControlHub, Tp4ControlWorker, Tp4DeviceImage, Tp4FrameSampler,
    Tp4FrameStepMode, control_read_is_idle, exit_now, watch_leader, watch_workers,
};
use sparkflash_model::backend::{BackendError, Sampling};
use sparkflash_model::vision::VisionImage;

use crate::runtime::{Runtime, RuntimeSlot, StepMode};
use crate::walk::Walk;

/// What an arm decides about its group's frames.
pub trait GroupWire: Send + Sync + 'static {
    /// The arm's name in trace lines.
    fn name(&self) -> &'static str;

    /// Whether to trace every frame on rank 0 and the workers.
    fn trace(&self) -> bool {
        false
    }

    /// Rank 0's refusal of a prompt every rank would refuse (over the context, an image span the
    /// arm cannot have made). It runs before the broadcast: a refusal after it would leave the
    /// workers inside a request rank 0 had abandoned.
    ///
    /// # Errors
    /// The refusal.
    fn admit(&self, tokens: &[u32], images: &[VisionImage]) -> Result<(), BackendError>;

    /// A request's sampler as the frame carries it, `None` for one the workers need not know.
    fn sampler(&self, sampling: &Sampling) -> Option<Tp4FrameSampler>;

    /// A worker's sampling from a frame's sampler.
    fn sampling(&self, sampler: Option<Tp4FrameSampler>) -> Sampling;

    /// A worker's image spans from a frame (no pixels: rank 0 encodes).
    ///
    /// # Errors
    /// A span the frame cannot describe.
    fn images(&self, frames: Vec<Tp4DeviceImage>) -> Result<Vec<VisionImage>, String>;
}

/// A request's image spans as the control frames carry them: where each sits, its grid and its
/// key, never its pixels.
#[must_use]
pub fn image_frames(images: &[VisionImage]) -> Vec<Tp4DeviceImage> {
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

/// A request's whole sampler as the frames carry it, `None` for the greedy default every worker
/// assumes without one.
///
/// This is for a group whose workers pick what rank 0 picks: their picks decide what a verify
/// keeps, so a worker that drew differently, or weighed a repetition differently, would keep a
/// different chain and fall out of step. Every field crosses; only the prompt's length, which
/// each rank's sampler counts from the prompt it prefills, does not.
#[must_use]
pub fn sampler_frame(sampling: &Sampling) -> Option<Tp4FrameSampler> {
    let greedy = Sampling {
        prompt_len: sampling.prompt_len,
        ..Sampling::default()
    };
    if *sampling == greedy {
        return None;
    }
    Some(Tp4FrameSampler {
        temperature_bits: sampling.temperature.to_bits(),
        top_p_bits: sampling.top_p.to_bits(),
        top_k: u32::from_ne_bytes(sampling.top_k.to_ne_bytes()),
        seed: sampling.seed,
        repetition_penalty_bits: sampling.repetition_penalty.to_bits(),
        presence_penalty_bits: sampling.presence_penalty.to_bits(),
        frequency_penalty_bits: sampling.frequency_penalty.to_bits(),
    })
}

/// A worker's sampling from [sampler_frame]'s frame: the greedy default without one.
#[must_use]
pub fn frame_sampling(sampler: Option<Tp4FrameSampler>) -> Sampling {
    let Some(sampler) = sampler else {
        return Sampling::default();
    };
    Sampling {
        temperature: f32::from_bits(sampler.temperature_bits),
        top_p: f32::from_bits(sampler.top_p_bits),
        top_k: i32::from_ne_bytes(sampler.top_k.to_ne_bytes()),
        repetition_penalty: f32::from_bits(sampler.repetition_penalty_bits),
        presence_penalty: f32::from_bits(sampler.presence_penalty_bits),
        frequency_penalty: f32::from_bits(sampler.frequency_penalty_bits),
        seed: sampler.seed,
        prompt_len: 0,
    }
}

/// The [GroupWire] of an arm whose group serves text: rank 0 refuses a prompt over the context or
/// with images (no rank loads a vision tower) before it broadcasts, and a request's whole sampler
/// crosses ([sampler_frame]) because every rank's picks decide what its verify keeps. GLM's,
/// Qwen's and the dense arm's `--tp4-rank` groups use it.
pub struct TextWire {
    /// The arm's name in trace lines.
    name: &'static str,
    /// The arm as the refusals name it.
    arm: &'static str,
    /// The positions a lane holds.
    context: usize,
}

impl TextWire {
    /// The wire of the arm `name` (trace lines) and `arm` (refusals) over lanes of `context`
    /// positions.
    #[must_use]
    pub fn new(name: &'static str, arm: &'static str, context: usize) -> Self {
        Self { name, arm, context }
    }
}

impl GroupWire for TextWire {
    fn name(&self) -> &'static str {
        self.name
    }

    fn trace(&self) -> bool {
        sparkflash_model::active_profile()
            .diagnostics
            .group_trace
            .unwrap_or(false)
    }

    fn admit(&self, tokens: &[u32], images: &[VisionImage]) -> Result<(), BackendError> {
        if !images.is_empty() {
            return Err(BackendError::new(
                "request",
                format!(
                    "the {} arm serves text: its vision tower is not loaded",
                    self.arm
                ),
            ));
        }
        if tokens.len() > self.context {
            return Err(BackendError::new(
                "context",
                format!(
                    "the prompt needs {} positions, {} remain",
                    tokens.len(),
                    self.context
                ),
            ));
        }
        Ok(())
    }

    fn sampler(&self, sampling: &Sampling) -> Option<Tp4FrameSampler> {
        sampler_frame(sampling)
    }

    fn sampling(&self, sampler: Option<Tp4FrameSampler>) -> Sampling {
        frame_sampling(sampler)
    }

    fn images(&self, frames: Vec<Tp4DeviceImage>) -> Result<Vec<VisionImage>, String> {
        if frames.is_empty() {
            Ok(Vec::new())
        } else {
            Err(format!(
                "the {} arm serves text: a frame named image spans",
                self.arm
            ))
        }
    }
}

/// The refusal of a prompt of `rows` positions (or a step) the lanes' memory cannot hold beside
/// the sequences being served: the engine has already given up every idle one.
#[must_use]
pub fn no_room(rows: usize) -> BackendError {
    BackendError::new(
        "context",
        format!(
            "the sessions' lane memory has no room for {rows} more positions beside the \
             sequences being served (runtime.lane_memory_mb / SPARKFLASH_LANE_MEMORY_MB sets the \
             pool)"
        ),
    )
}

/// What rank 0 does when its own half of a broadcast frame failed: the group is out of step.
pub type Split = fn(&str) -> !;

/// Ends rank 0's process when its group is out of step ([Split]). Rank 0's exit closes every
/// control socket, which ends each worker's process in turn, collective or not.
pub fn end_group(why: &str) -> ! {
    eprintln!(
        "sparkflash serve: rank 0: the group is out of step and cannot serve, exiting: {why}"
    );
    exit_now(1)
}

/// Rank 0's hub, watched: a worker whose control socket closes ends rank 0 too
/// ([watch_workers]), so a dead rank fails the server instead of leaving its requests in a
/// collective.
///
/// # Errors
/// A watch that cannot start.
pub fn watched_hub(hub: Tp4ControlHub) -> Result<Arc<Mutex<Tp4ControlHub>>, BackendError> {
    watch_workers(&hub).map_err(|error| BackendError::new("group", error))?;
    Ok(Arc::new(Mutex::new(hub)))
}

/// Set never: the lanes' rank 0 says no End, so any close of its socket ends a worker.
static NO_END: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Broadcasts one frame to the group.
///
/// # Errors
/// A poisoned hub or a frame that could not be written.
pub fn broadcast(hub: &Mutex<Tp4ControlHub>, frame: &Tp4ControlFrame) -> Result<(), BackendError> {
    let kind = frame.kind();
    let mut hub = hub
        .lock()
        .map_err(|_| BackendError::new("group", "the control hub lock is poisoned"))?;
    hub.broadcast(frame)
        .map_err(|error| BackendError::new("group", format!("{kind}: {error}")))
}

/// Rank 0's side of a tick group: the frames a [crate::TickEngine]'s decisions produce, each
/// broadcast before the runtime call it names.
///
/// The engine keeps every decision (which lane, which rows, which step shape, when to give back
/// or release) and calls this once per decision. [Tp4ControlFrame::LaneRows] carries the rows
/// [crate::Runtime::run_part_rows] must run, and [Tp4ControlFrame::LaneStepMode] carries the
/// shape [crate::Runtime::step_mode_slots] must run, with the members in rank 0's row order.
/// [Tp4ControlFrame::LaneRequest] carries the start rank 0's `start_prompt_slot` chose, so its
/// rows are a position from a start both ranks agree on. A prompt whose start runs a collective
/// (an image encode) cannot wait for that start: its request goes first ([Self::begin]) and
/// [Tp4ControlFrame::LaneStart] names the start after it ([Self::started]).
///
/// A worker replays it with [mirror_tick_lanes]. A failure of rank 0's own half after a frame
/// has gone out leaves the workers past rank 0, so [Self::out_of_step] tells them why and ends
/// the process ([Split]); a refusal before any frame is an ordinary error.
pub struct TickGroup {
    hub: Arc<Mutex<Tp4ControlHub>>,
    wire: Box<dyn GroupWire>,
    /// What a failure after a broadcast does ([end_group] unless [Self::with_split]).
    split: Split,
}

impl TickGroup {
    /// Rank 0's frames through `hub`, refused and described by `wire`.
    #[must_use]
    pub fn new(hub: Arc<Mutex<Tp4ControlHub>>, wire: Box<dyn GroupWire>) -> Self {
        Self {
            hub,
            wire,
            split: end_group,
        }
    }

    /// Splits the group with `split` instead of ending the process, for a test.
    #[must_use]
    pub fn with_split(mut self, split: Split) -> Self {
        self.split = split;
        self
    }

    /// The arm's name in trace lines.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.wire.name()
    }

    /// Rank 0's refusal of a prompt every rank would refuse, before any frame for it.
    ///
    /// # Errors
    /// The refusal.
    pub fn admit(&self, tokens: &[u32], images: &[VisionImage]) -> Result<(), BackendError> {
        self.wire.admit(tokens, images)
    }

    /// Broadcast: lane `lane` begins this prompt at rank 0's start, with rank 0's sampler.
    ///
    /// The frame follows rank 0's own start because it carries the start; no row has run yet,
    /// so the worker's check is against every row the frames after it name.
    ///
    /// # Errors
    /// A count the frame cannot carry, or a frame that could not be sent (the group is split).
    pub fn place(
        &self,
        lane: usize,
        tokens: &[u32],
        images: &[VisionImage],
        start: usize,
        sampling: &Sampling,
    ) -> Result<(), BackendError> {
        let start = u32::try_from(start).map_err(|_| {
            BackendError::new("group", format!("a start of {start} does not fit a frame"))
        })?;
        self.request(lane, tokens, images, Some(start), sampling)
    }

    /// Broadcast: lane `lane` begins this prompt before rank 0's own start, with rank 0's
    /// sampler. A prompt whose start runs a collective (an image encode every rank joins) has to
    /// have its request frame in front of that start, because the workers join the collective
    /// from the frame; [Self::started] names the start that follows.
    ///
    /// # Errors
    /// A count the frame cannot carry, or a frame that could not be sent (the group is split).
    pub fn begin(
        &self,
        lane: usize,
        tokens: &[u32],
        images: &[VisionImage],
        sampling: &Sampling,
    ) -> Result<(), BackendError> {
        self.request(lane, tokens, images, None, sampling)
    }

    /// Broadcast: the start rank 0's own `start_prompt_slot` chose for the prompt
    /// [Self::begin] opened. The worker refuses a start that differs, because every row frame
    /// after it is a position from this start.
    ///
    /// # Errors
    /// A start the frame cannot carry, or a frame that could not be sent (the group is split).
    pub fn started(&self, lane: usize, start: usize) -> Result<(), BackendError> {
        let lane = u32::try_from(lane)
            .map_err(|_| BackendError::new("group", format!("lane {lane} does not fit a frame")))?;
        let start = u32::try_from(start).map_err(|_| {
            BackendError::new("group", format!("a start of {start} does not fit a frame"))
        })?;
        if self.wire.trace() {
            eprintln!(
                "[{}] serve rank 0 lane {lane} starts at {start}",
                self.wire.name()
            );
        }
        self.in_step(self.send(&Tp4ControlFrame::LaneStart { lane, start }))
    }

    /// The request frame both [Self::place] and [Self::begin] send: the lane, the prompt, its
    /// image spans and the sampler, with the start when rank 0 already knows it.
    fn request(
        &self,
        lane: usize,
        tokens: &[u32],
        images: &[VisionImage],
        start: Option<u32>,
        sampling: &Sampling,
    ) -> Result<(), BackendError> {
        let lane = u32::try_from(lane)
            .map_err(|_| BackendError::new("group", format!("lane {lane} does not fit a frame")))?;
        if self.wire.trace() {
            match start {
                Some(start) => eprintln!(
                    "[{}] serve rank 0 lane {lane} prefill {} rows at {start}",
                    self.wire.name(),
                    tokens.len()
                ),
                None => eprintln!(
                    "[{}] serve rank 0 lane {lane} begins {} rows",
                    self.wire.name(),
                    tokens.len()
                ),
            }
        }
        self.in_step(self.send(&Tp4ControlFrame::LaneRequest {
            lane,
            prompt_ids: tokens.to_vec(),
            images: image_frames(images),
            // The tick worker starts from the frame's start, not from this: the lane's history
            // on the worker is what rank 0's was, so its own resume reaches the same start, and a
            // frame that disagrees is refused rather than obeyed.
            keep: true,
            sampler: self.wire.sampler(sampling),
            start,
        }))
    }

    /// Broadcast: lane `lane` runs exactly `rows` rows of the prompt its request frame began.
    ///
    /// # Errors
    /// A row count the frame cannot carry, or a frame that could not be sent (the group is
    /// split).
    pub fn rows(&self, lane: usize, rows: usize) -> Result<(), BackendError> {
        let lane = u32::try_from(lane)
            .map_err(|_| BackendError::new("group", format!("lane {lane} does not fit a frame")))?;
        let rows = u32::try_from(rows)
            .map_err(|_| BackendError::new("group", format!("{rows} rows do not fit a frame")))?;
        if self.wire.trace() {
            eprintln!(
                "[{}] serve rank 0 lane {lane} rows {rows}",
                self.wire.name()
            );
        }
        self.in_step(self.send(&Tp4ControlFrame::LaneRows { lane, rows }))
    }

    /// Broadcast: one step of `lanes` in this order, each consuming the token in `tokens`, in
    /// `mode`. The order is the step's row order on every rank.
    ///
    /// # Errors
    /// A member count the frame cannot carry, mismatched member lists, or a frame that could not
    /// be sent (the group is split).
    pub fn step_mode(
        &self,
        lanes: &[usize],
        tokens: &[u32],
        mode: StepMode,
    ) -> Result<(), BackendError> {
        if lanes.len() != tokens.len() {
            return Err(BackendError::new(
                "group",
                format!(
                    "{} lanes step {} tokens: every member needs one",
                    lanes.len(),
                    tokens.len()
                ),
            ));
        }
        let mut members = Vec::with_capacity(lanes.len());
        for (&lane, &token) in lanes.iter().zip(tokens) {
            let lane = u32::try_from(lane).map_err(|_| {
                BackendError::new("group", format!("lane {lane} does not fit a frame"))
            })?;
            members.push((lane, token));
        }
        let mode = match mode {
            StepMode::Plain => Tp4FrameStepMode::Plain,
            StepMode::Round { joint } => Tp4FrameStepMode::Round { joint },
        };
        if self.wire.trace() {
            eprintln!(
                "[{}] serve rank 0 step {members:?} {mode:?}",
                self.wire.name()
            );
        }
        self.in_step(self.send(&Tp4ControlFrame::LaneStepMode { members, mode }))
    }

    /// Broadcast: lane `lane` rolls back `count` answers, after rank 0's own lane did.
    ///
    /// # Errors
    /// A count the frame cannot carry, or a frame that could not be sent (the group is split).
    pub fn give_back(&self, lane: usize, count: usize) -> Result<(), BackendError> {
        if count == 0 {
            return Ok(());
        }
        let lane = u32::try_from(lane)
            .map_err(|_| BackendError::new("group", format!("lane {lane} does not fit a frame")))?;
        let count = u32::try_from(count)
            .map_err(|_| BackendError::new("group", format!("{count} answers to give back")))?;
        if self.wire.trace() {
            eprintln!(
                "[{}] serve rank 0 lane {lane} gives back {count}",
                self.wire.name()
            );
        }
        self.in_step(self.send(&Tp4ControlFrame::LaneGiveBack { lane, count }))
    }

    /// Broadcast: lane `lane` gives up what it holds, before rank 0's own lane does.
    ///
    /// # Errors
    /// A frame that could not be sent (the group is split).
    pub fn release(&self, lane: usize) -> Result<(), BackendError> {
        let lane = u32::try_from(lane)
            .map_err(|_| BackendError::new("group", format!("lane {lane} does not fit a frame")))?;
        if self.wire.trace() {
            eprintln!("[{}] serve rank 0 lane {lane} release", self.wire.name());
        }
        self.in_step(self.send(&Tp4ControlFrame::LaneRelease { lane }))
    }

    /// Rank 0's own half of a frame failed after the frame went out: the workers are past rank 0
    /// and cannot be caught up. Tells them why and ends the group.
    pub fn out_of_step(&self, why: &str) -> ! {
        let _ = self.send(&Tp4ControlFrame::Error {
            code: "group_out_of_step".to_owned(),
            message: why.to_owned(),
        });
        (self.split)(why)
    }

    fn send(&self, frame: &Tp4ControlFrame) -> Result<(), BackendError> {
        broadcast(&self.hub, frame)
    }

    /// `result` of an operation that may have reached the workers: a failure leaves them past
    /// rank 0, so the group ends ([Split]) once it has been told why.
    fn in_step<T>(&self, result: Result<T, BackendError>) -> Result<T, BackendError> {
        match result {
            Ok(value) => Ok(value),
            Err(error) => self.out_of_step(&error.to_string()),
        }
    }
}

/// The refusal of a start that differs from the one this rank reached: every row frame after it
/// is a position from rank 0's start, so the ranks cannot serve on.
fn start_differs(lane: u32, start: usize, at: usize) -> String {
    format!(
        "lane {lane}: rank 0 starts the prompt at {start}, this rank starts it at {at}: the ranks \
         are out of step"
    )
}

/// Mirrors rank 0's tick frames onto this rank's own pool of runtime slots, until the control
/// channel ends.
///
/// The pool is opened up front, as rank 0's engine opens its own, so lane `n` is the same slot
/// on every rank. A request frame starts the lane's prompt on this rank and refuses a start that
/// differs from rank 0's; when it carries no start (a prompt with images, whose start rank 0
/// had to broadcast the frame before) the lane's own start waits for the [Tp4ControlFrame::LaneStart]
/// that follows it; a rows frame runs exactly the rows it names through
/// [crate::Runtime::run_part_rows]; a step frame takes its member lanes in the frame's order and
/// runs [crate::Runtime::step_mode_slots] with the frame's shape, then drains the answers the
/// arm already holds exactly as rank 0's engine does (those rows run no device work, so they
/// travel as no frame, but their history bookkeeping must happen on every rank).
///
/// # Errors
/// A frame for a lane outside the pool (the ranks are out of step), a start that differs, an arm
/// failure, or the channel's end.
pub fn mirror_tick_lanes<W: Walk>(
    control: &mut Tp4ControlWorker,
    runtime: &Runtime<W>,
    wire: &dyn GroupWire,
    lanes: usize,
) -> Result<(), String> {
    mirror_tick_lanes_seen(control, runtime, wire, lanes, &mut |_| {})
}

/// [mirror_tick_lanes], handing every frame that arrived to `seen` before it runs. The CPU gate
/// uses it to pin the message order a worker was given; a serving process passes a no-op.
///
/// # Errors
/// As [mirror_tick_lanes].
pub fn mirror_tick_lanes_seen<W: Walk>(
    control: &mut Tp4ControlWorker,
    runtime: &Runtime<W>,
    wire: &dyn GroupWire,
    lanes: usize,
    seen: &mut dyn FnMut(&Tp4ControlFrame),
) -> Result<(), String> {
    let mut pool: Vec<Option<RuntimeSlot<W>>> = Vec::with_capacity(lanes);
    // The starts a request frame that carried none reached, waiting for rank 0's start frame.
    let mut begun: Vec<Option<usize>> = vec![None; lanes];
    for _ in 0..lanes {
        pool.push(Some(
            runtime
                .open_slot()
                .map_err(|error| format!("open a lane: {error}"))?,
        ));
    }
    let name = wire.name();
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
        seen(&frame);
        match frame {
            Tp4ControlFrame::LaneRequest {
                lane,
                prompt_ids,
                images,
                keep: _,
                sampler,
                start,
            } => {
                let images = wire
                    .images(images)
                    .map_err(|error| format!("lane {lane}: {error}"))?;
                let slot = pool
                    .get_mut(lane as usize)
                    .and_then(Option::as_mut)
                    .ok_or_else(|| out_of_step(lane))?;
                // The lane's sampler travels whole, as it does for every group: every rank's
                // picks decide what a verify keeps. The request's start does not travel through
                // it - the arm counts from the prompt it prefills.
                runtime
                    .walk()
                    .set_sampling(slot.seq_mut(), wire.sampling(sampler));
                if wire.trace() {
                    eprintln!(
                        "[{name}] serve worker lane {lane} prefill {} rows",
                        prompt_ids.len()
                    );
                }
                let at = runtime
                    .start_prompt_slot(slot, &prompt_ids, &images)
                    .map_err(|error| format!("lane {lane}: start: {error}"))?;
                begun[lane as usize] = None;
                if let Some(start) = start {
                    let start = start as usize;
                    if at != start {
                        return Err(start_differs(lane, start, at));
                    }
                } else {
                    // Rank 0's start runs a collective on this prompt (an image encode), so its
                    // frame had to come first: the start is checked against the next frame.
                    begun[lane as usize] = Some(at);
                }
            }
            Tp4ControlFrame::LaneStart { lane, start } => {
                if wire.trace() {
                    eprintln!("[{name}] serve worker lane {lane} starts at {start}");
                }
                let reached = begun
                    .get_mut(lane as usize)
                    .and_then(Option::take)
                    .ok_or_else(|| {
                        format!(
                            "lane {lane} starts at {start} with no request frame before it: the \
                             ranks are out of step"
                        )
                    })?;
                if reached != start as usize {
                    return Err(start_differs(lane, start as usize, reached));
                }
            }
            Tp4ControlFrame::LaneRows { lane, rows } => {
                let slot = pool
                    .get_mut(lane as usize)
                    .and_then(Option::as_mut)
                    .ok_or_else(|| out_of_step(lane))?;
                if wire.trace() {
                    eprintln!("[{name}] serve worker lane {lane} rows {rows}");
                }
                runtime
                    .run_part_rows(slot, rows as usize)
                    .map_err(|error| format!("lane {lane}: rows {rows}: {error}"))?;
            }
            Tp4ControlFrame::LaneStepMode { members, mode } => {
                let mode = match mode {
                    Tp4FrameStepMode::Plain => StepMode::Plain,
                    Tp4FrameStepMode::Round { joint } => StepMode::Round { joint },
                };
                let mut stepping = Vec::with_capacity(members.len());
                for &(lane, token) in &members {
                    let slot = pool
                        .get_mut(lane as usize)
                        .and_then(Option::take)
                        .ok_or_else(|| out_of_step(lane))?;
                    stepping.push((slot, token));
                }
                if wire.trace() {
                    eprintln!("[{name}] serve worker step {members:?} {mode:?}");
                }
                let answer = runtime.step_mode_slots(&mut stepping, mode);
                for ((slot, _), &(lane, _)) in stepping.into_iter().zip(&members) {
                    pool[lane as usize] = Some(slot);
                }
                let (steps, _) = answer.map_err(|error| format!("step {members:?}: {error}"))?;
                // The answers the arm already holds cost no row, so rank 0 runs them without a
                // frame; this rank runs the same runtime entry for the same tokens, whose only
                // effect is the history the next prompt's start is read from.
                for (&(lane, _), mut step) in members.iter().zip(steps) {
                    while let Some(token) = step.next_token {
                        let ready = pool[lane as usize]
                            .as_ref()
                            .is_some_and(|slot| runtime.ready_slot(slot, token));
                        if !ready {
                            break;
                        }
                        let slot = pool[lane as usize]
                            .take()
                            .ok_or_else(|| out_of_step(lane))?;
                        let mut member = [(slot, token)];
                        let held = runtime.step_slots(&mut member);
                        let [(slot, _)] = member;
                        pool[lane as usize] = Some(slot);
                        match held {
                            Ok(mut steps) => match steps.pop() {
                                Some(next) => step = next,
                                None => break,
                            },
                            Err(error) => {
                                return Err(format!("lane {lane}: held answer: {error}"));
                            }
                        }
                    }
                }
            }
            Tp4ControlFrame::LaneGiveBack { lane, count } => {
                let slot = pool
                    .get_mut(lane as usize)
                    .and_then(Option::as_mut)
                    .ok_or_else(|| out_of_step(lane))?;
                if wire.trace() {
                    eprintln!("[{name}] serve worker lane {lane} gives back {count}");
                }
                runtime
                    .give_back_slot(slot, count as usize)
                    .map_err(|error| format!("lane {lane}: give back {count}: {error}"))?;
            }
            Tp4ControlFrame::LaneRelease { lane } => {
                let slot = pool
                    .get_mut(lane as usize)
                    .and_then(Option::as_mut)
                    .ok_or_else(|| out_of_step(lane))?;
                if wire.trace() {
                    eprintln!("[{name}] serve worker lane {lane} release");
                }
                runtime
                    .release_slot(slot)
                    .map_err(|error| format!("lane {lane}: release: {error}"))?;
            }
            Tp4ControlFrame::Error { code, message } => {
                return Err(format!("rank 0 ended the group ({code}): {message}"));
            }
            other => {
                return Err(format!(
                    "the tick worker expected LaneRequest, LaneStart, LaneRows, LaneStepMode, \
                     LaneGiveBack, LaneRelease or Error, not {}",
                    other.kind()
                ));
            }
        }
    }
}

/// Runs a worker's [mirror_tick_lanes] and then ends its process: a tick mirror only ever stops
/// when the group can no longer serve. A watch on rank 0's control socket ends the process too
/// while it waits in a collective rank 0 will never join.
pub fn mirror_ticks_until_the_group_ends<W: Walk>(
    control: &mut Tp4ControlWorker,
    runtime: &Runtime<W>,
    wire: &dyn GroupWire,
    lanes: usize,
) -> ! {
    let rank = control.rank;
    if let Err(error) = watch_leader(&control.stream, rank as usize, &NO_END, || {}) {
        eprintln!("sparkflash serve: rank {rank}: {error}");
        exit_now(1);
    }
    let error = match mirror_tick_lanes(control, runtime, wire, lanes) {
        Ok(()) => "the control channel ended".to_owned(),
        Err(error) => error,
    };
    eprintln!("sparkflash serve: rank {rank}: the group cannot serve, exiting: {error}");
    exit_now(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_whole_sampler_crosses_the_frame_and_greedy_sends_none() {
        assert_eq!(
            sampler_frame(&Sampling {
                prompt_len: 9,
                ..Sampling::default()
            }),
            None
        );
        assert_eq!(frame_sampling(None), Sampling::default());
        let sampled = Sampling {
            temperature: 0.7,
            top_p: 0.9,
            top_k: -1,
            repetition_penalty: 1.1,
            presence_penalty: 0.5,
            frequency_penalty: 0.25,
            seed: 11,
            prompt_len: 0,
        };
        assert_eq!(frame_sampling(sampler_frame(&sampled)), sampled);
        // A greedy request with a penalty is not the default: its argmax moves.
        let penalized = Sampling {
            repetition_penalty: 1.2,
            ..Sampling::default()
        };
        assert_eq!(frame_sampling(sampler_frame(&penalized)), penalized);
    }
}

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

//! Fused lanes across a tensor-parallel group: rank 0 decides, every other rank replays.
//!
//! Every rank holds its share of the weights and runs the same walks, because the collectives
//! inside a walk block until every rank joins. Rank 0 alone runs the [crate::FusedEngine]: it picks
//! which lanes step together, and [GroupLanes] broadcasts each decision (a lane's prompt and each
//! later part of it, a step's members in their row order, a give-back, a release) *before* running
//! it on its own rank. A
//! worker runs [mirror_lanes] over its own lane arm and replays the frames in the order they came,
//! so lane `n` is the same sequence on every rank and the collectives line up.
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
//! closes, even while it waits in a collective ([mirror_until_the_group_ends]). Rank 0 ends when
//! a worker's socket closes ([watched_hub]).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use sparkflash_comm::control::{
    Tp4ControlFrame, Tp4ControlHub, Tp4ControlWorker, Tp4DeviceImage, Tp4FrameSampler,
    control_read_is_idle, exit_now, watch_leader, watch_workers,
};
use sparkflash_model::backend::{BackendError, ModelSession, Sampling, Speculation, StepOutput};
use sparkflash_model::vision::{ImageSpan, VisionImage};

use crate::fused::{FusedArm, FusedEngine, Part};
use crate::ledger::{Ledger, Mode, Strategy};

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
/// crosses ([sampler_frame]) because every rank's picks decide what its verify keeps. GLM's and
/// Qwen's `--tp4-rank` groups use it.
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

/// Runs a worker's [mirror_lanes] and then ends its process: a mirror only ever stops when the
/// group can no longer serve. A watch on rank 0's control socket ends the process too while it
/// waits in a collective rank 0 will never join.
pub fn mirror_until_the_group_ends<A: FusedArm, W: GroupWire>(
    control: &mut Tp4ControlWorker,
    arm: &A,
    wire: &W,
    lanes: usize,
) -> ! {
    let rank = control.rank;
    if let Err(error) = watch_leader(&control.stream, rank as usize, &NO_END, || {}) {
        eprintln!("sparkflash serve: rank {rank}: {error}");
        exit_now(1);
    }
    let error = match mirror_lanes(control, arm, wire, lanes) {
        Ok(()) => "the control channel ended".to_owned(),
        Err(error) => error,
    };
    eprintln!("sparkflash serve: rank {rank}: the group cannot serve, exiting: {error}");
    exit_now(1)
}

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

/// Mirrors rank 0's lane frames onto this rank's own pool of `lanes` fused lanes, until the
/// control channel ends.
///
/// The pool is opened up front, as rank 0's engine opens its own, so lane `n` is the same slot on
/// every rank. A `LaneRequest` forgets what the lane held, unless rank 0 kept its own, and
/// prefills it with rank 0's sampler; a `LaneStep` steps its members in the frame's order, which
/// is the step's row order on every rank and so the order the collectives inside it line up in.
///
/// # Errors
/// A frame for a lane outside the pool (the ranks are out of step), an arm failure, or the
/// channel's end.
pub fn mirror_lanes<A: FusedArm, W: GroupWire>(
    control: &mut Tp4ControlWorker,
    arm: &A,
    wire: &W,
    lanes: usize,
) -> Result<(), String> {
    let mut pool: Vec<Option<A::Slot>> = Vec::with_capacity(lanes);
    for _ in 0..lanes {
        pool.push(Some(
            arm.open_slot()
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
        match frame {
            Tp4ControlFrame::LaneRequest {
                lane,
                prompt_ids,
                images,
                keep,
                sampler,
            } => {
                let images = wire
                    .images(images)
                    .map_err(|error| format!("lane {lane}: {error}"))?;
                let slot = pool
                    .get_mut(lane as usize)
                    .and_then(Option::as_mut)
                    .ok_or_else(|| out_of_step(lane))?;
                if !keep && arm.holds_request(slot) {
                    arm.renew(slot)
                        .map_err(|error| format!("lane {lane}: renew: {error}"))?;
                }
                arm.set_sampling(slot, wire.sampling(sampler));
                if wire.trace() {
                    eprintln!("[{name}] serve worker lane {lane} prefill {prompt_ids:?}");
                }
                arm.begin_prompt(slot, &prompt_ids, &images)
                    .map_err(|error| format!("lane {lane}: prefill: {error}"))?;
            }
            Tp4ControlFrame::LanePart { lane } => {
                let slot = pool
                    .get_mut(lane as usize)
                    .and_then(Option::as_mut)
                    .ok_or_else(|| out_of_step(lane))?;
                if wire.trace() {
                    eprintln!("[{name}] serve worker lane {lane} next part");
                }
                arm.next_part(slot)
                    .map_err(|error| format!("lane {lane}: prefill part: {error}"))?;
            }
            Tp4ControlFrame::LaneRelease { lane } => {
                let slot = pool
                    .get_mut(lane as usize)
                    .and_then(Option::as_mut)
                    .ok_or_else(|| out_of_step(lane))?;
                if wire.trace() {
                    eprintln!("[{name}] serve worker lane {lane} release");
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
                if wire.trace() {
                    eprintln!("[{name}] serve worker step {members:?}");
                }
                let answer = arm.step(&mut stepping);
                for ((slot, _), &(lane, _)) in stepping.into_iter().zip(&members) {
                    pool[lane as usize] = Some(slot);
                }
                answer.map_err(|error| format!("step {members:?}: {error}"))?;
            }
            Tp4ControlFrame::LaneGiveBack { lane, count } => {
                let slot = pool
                    .get_mut(lane as usize)
                    .and_then(Option::as_mut)
                    .ok_or_else(|| out_of_step(lane))?;
                if wire.trace() {
                    eprintln!("[{name}] serve worker lane {lane} gives back {count}");
                }
                arm.give_back(slot, count as usize)
                    .map_err(|error| format!("lane {lane}: give back {count}: {error}"))?;
            }
            Tp4ControlFrame::Error { code, message } => {
                return Err(format!("rank 0 ended the group ({code}): {message}"));
            }
            other => {
                return Err(format!(
                    "the fused worker expected LaneRequest, LanePart, LaneRelease, LaneStep, \
                     LaneGiveBack or Error, not {}",
                    other.kind()
                ));
            }
        }
    }
}

/// The engine a fused backend's sessions are handles on: its own lanes, or rank 0 of a group's,
/// every lane's prompt and step broadcast before it runs ([GroupLanes]). Every arm that fuses keeps
/// one of these, so starting either form and opening a session on it is written once.
pub enum Fusion<A: FusedArm> {
    /// This process alone.
    Local(Arc<FusedEngine<A>>),
    /// Rank 0 of a group.
    Group(Arc<FusedEngine<GroupLanes<A, TextWire>>>),
}

impl<A: FusedArm> Fusion<A> {
    /// `capacity` lanes of `arm` on this process.
    ///
    /// # Errors
    /// As [FusedEngine::start].
    pub fn local(arm: A, capacity: usize) -> Result<Self, BackendError> {
        Ok(Self::Local(FusedEngine::start(Arc::new(arm), capacity)?))
    }

    /// Rank 0 of a group: `arm`'s `capacity` lanes, every frame broadcast through `hub` first
    /// (watched, [watched_hub]) and refused first as `wire` refuses it; each worker replays them
    /// ([mirror_until_the_group_ends]) over lanes of its own.
    ///
    /// # Errors
    /// A watch that cannot start, then as [FusedEngine::start].
    pub fn group(
        arm: A,
        hub: Tp4ControlHub,
        wire: TextWire,
        capacity: usize,
    ) -> Result<Self, BackendError> {
        let arm = GroupLanes::new(arm, watched_hub(hub)?, wire);
        Ok(Self::Group(FusedEngine::start(Arc::new(arm), capacity)?))
    }

    /// A session on the engine: one pooled lane for one request.
    ///
    /// # Errors
    /// As [FusedEngine::open].
    pub fn open(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        Ok(match self {
            Self::Local(engine) => Box::new(engine.open()?),
            Self::Group(engine) => Box::new(engine.open()?),
        })
    }
}

/// Rank 0's fused lanes: a lane arm whose every prompt, step, give-back and release is broadcast
/// before it runs.
///
/// The engine thread is the only caller, so the frames leave in the order the steps run, and a
/// worker that mirrors them in that order runs the same walks with the same rows. Each slot
/// carries the lane number the workers know it by.
pub struct GroupLanes<A: FusedArm, W: GroupWire> {
    arm: A,
    hub: Arc<Mutex<Tp4ControlHub>>,
    wire: W,
    /// The next lane number [FusedArm::open_slot] hands out.
    next: AtomicU32,
    /// What a failure after a broadcast does ([end_group] unless [Self::with_split]).
    split: Split,
}

/// One lane of [GroupLanes]: the inner arm's slot and the number the workers know it by. The
/// slot is lent to the inner arm for a step and back between steps.
pub struct GroupLane<S> {
    lane: u32,
    slot: Option<S>,
    /// Whether the engine renewed the slot since its last prompt: the workers renew theirs only
    /// then, so a kept lane continues on every rank alike.
    renewed: bool,
    /// The request's sampler for the workers ([GroupWire::sampler]).
    sampler: Option<Tp4FrameSampler>,
}

impl<A: FusedArm, W: GroupWire> GroupLanes<A, W> {
    /// Wraps `arm` to broadcast through `hub`, refusing first what `wire` refuses.
    pub fn new(arm: A, hub: Arc<Mutex<Tp4ControlHub>>, wire: W) -> Self {
        Self {
            arm,
            hub,
            wire,
            next: AtomicU32::new(0),
            split: end_group,
        }
    }

    /// Splits the group with `split` instead of ending the process, for a test.
    #[must_use]
    pub fn with_split(mut self, split: Split) -> Self {
        self.split = split;
        self
    }

    /// The wrapped arm.
    pub fn arm(&self) -> &A {
        &self.arm
    }

    fn send(&self, frame: &Tp4ControlFrame) -> Result<(), BackendError> {
        broadcast(&self.hub, frame)
    }

    /// `result` of a frame that may have reached the workers: a failure leaves them past rank 0,
    /// so the group ends ([Split]) once it has been told why.
    fn in_step<T>(&self, result: Result<T, BackendError>) -> Result<T, BackendError> {
        match result {
            Ok(value) => Ok(value),
            Err(error) => {
                let message = error.to_string();
                let _ = self.send(&Tp4ControlFrame::Error {
                    code: "group_out_of_step".to_owned(),
                    message: message.clone(),
                });
                (self.split)(&message)
            }
        }
    }

    fn slot(lane: &mut GroupLane<A::Slot>) -> Result<&mut A::Slot, BackendError> {
        lane.slot
            .as_mut()
            .ok_or_else(|| BackendError::new("group", "a lane's slot is already stepping"))
    }
}

impl<A: FusedArm, W: GroupWire> crate::fused::sealed::Sealed for GroupLanes<A, W> {}

impl<A: FusedArm, W: GroupWire> FusedArm for GroupLanes<A, W> {
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
        slot.sampler = self.wire.sampler(&sampling);
        if let Some(inner) = slot.slot.as_mut() {
            self.arm.set_sampling(inner, sampling);
        }
    }

    fn prefill(&self, slot: &mut Self::Slot, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        self.prefill_images(slot, tokens, &[])
    }

    /// Every part through the frames, as the engine would run them back to back: a worker's
    /// `LaneRequest` runs only the first part.
    fn prefill_images(
        &self,
        slot: &mut Self::Slot,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<StepOutput, BackendError> {
        let mut part = self.begin_prompt(slot, tokens, images)?;
        loop {
            match part {
                Part::Done(step) => return Ok(step),
                Part::More => part = self.next_part(slot)?,
            }
        }
    }

    /// Broadcast: every rank begins the same lane's prompt and runs its first part.
    fn begin_prompt(
        &self,
        slot: &mut Self::Slot,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<Part, BackendError> {
        // Refuse before the broadcast, as the serial session does.
        self.wire.admit(tokens, images)?;
        if !self.arm.room(Self::slot(slot)?, Some(tokens)) {
            return Err(no_room(tokens.len()));
        }
        if self.wire.trace() {
            eprintln!(
                "[{}] serve rank 0 lane {} prefill {tokens:?}",
                self.wire.name(),
                slot.lane
            );
        }
        let lane = slot.lane;
        let keep = !std::mem::take(&mut slot.renewed);
        let sampler = slot.sampler;
        let inner = Self::slot(slot)?;
        self.in_step(
            self.send(&Tp4ControlFrame::LaneRequest {
                lane,
                prompt_ids: tokens.to_vec(),
                images: image_frames(images),
                keep,
                sampler,
            })
            .and_then(|()| self.arm.begin_prompt(inner, tokens, images)),
        )
    }

    /// Broadcast: every rank runs the same lane's next part, between the same steps.
    fn next_part(&self, slot: &mut Self::Slot) -> Result<Part, BackendError> {
        if self.wire.trace() {
            eprintln!(
                "[{}] serve rank 0 lane {} next part",
                self.wire.name(),
                slot.lane
            );
        }
        let lane = slot.lane;
        let inner = Self::slot(slot)?;
        self.in_step(
            self.send(&Tp4ControlFrame::LanePart { lane })
                .and_then(|()| self.arm.next_part(inner)),
        )
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
        // the broadcast, and not on the workers. A member whose answer is already held grows
        // nothing.
        if stepping
            .iter()
            .any(|(inner, token)| !self.arm.ready(inner, *token) && !self.arm.room(inner, None))
        {
            for ((lane, _), (inner, _)) in members.iter_mut().zip(stepping) {
                lane.slot = Some(inner);
            }
            return Err(no_room(1));
        }
        if self.wire.trace() {
            eprintln!("[{}] serve rank 0 step {lanes:?}", self.wire.name());
        }
        let answer = self
            .send(&Tp4ControlFrame::LaneStep { members: lanes })
            .and_then(|()| self.arm.step(&mut stepping));
        for ((lane, _), (inner, _)) in members.iter_mut().zip(stepping) {
            lane.slot = Some(inner);
        }
        self.in_step(answer)
    }

    /// Local: rank 0's lane holds what every rank's does.
    fn ready(&self, slot: &Self::Slot, token: u32) -> bool {
        slot.slot
            .as_ref()
            .is_some_and(|inner| self.arm.ready(inner, token))
    }

    /// Broadcast: every rank rolls the same lane back by the same count.
    ///
    /// Unlike a prompt or a step, rank 0 runs it before it is sent. A give-back is the lane's own
    /// bookkeeping, with no collective for the workers to meet, so the order is free. An arm that
    /// refuses it (GLM's holds no answers to give back) refuses on every rank alike, and the
    /// engine renews the slot instead. Sending first would end every worker on that refusal,
    /// while rank 0 renewed and stepped on into a collective no worker would join.
    fn give_back(&self, slot: &mut Self::Slot, count: usize) -> Result<(), BackendError> {
        if count == 0 {
            return Ok(());
        }
        let frame_count = u32::try_from(count)
            .map_err(|_| BackendError::new("group", format!("{count} answers to give back")))?;
        self.arm.give_back(Self::slot(slot)?, count)?;
        if self.wire.trace() {
            eprintln!(
                "[{}] serve rank 0 lane {} gives back {count}",
                self.wire.name(),
                slot.lane
            );
        }
        // Rank 0's lane has given back: a worker that never hears of it is past rank 0.
        self.in_step(self.send(&Tp4ControlFrame::LaneGiveBack {
            lane: slot.lane,
            count: frame_count,
        }))
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

    /// Rank 0's answer: a slot it renews is renewed on every worker by the next `LaneRequest`.
    fn continues(&self, slot: &Self::Slot, prompt: &[u32]) -> bool {
        slot.slot
            .as_ref()
            .is_some_and(|inner| self.arm.continues(inner, prompt))
    }

    fn room(&self, slot: &Self::Slot, prompt: Option<&[u32]>) -> bool {
        slot.slot
            .as_ref()
            .is_some_and(|inner| self.arm.room(inner, prompt))
    }

    /// Broadcast: every rank gives up the same lane's memory, so their pools stay alike.
    fn release(&self, slot: &mut Self::Slot) -> Result<(), BackendError> {
        if self.wire.trace() {
            eprintln!(
                "[{}] serve rank 0 lane {} release",
                self.wire.name(),
                slot.lane
            );
        }
        let lane = slot.lane;
        let inner = Self::slot(slot)?;
        self.in_step(
            self.send(&Tp4ControlFrame::LaneRelease { lane })
                .and_then(|()| self.arm.release(inner)),
        )?;
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

    fn ledger(&self) -> Ledger {
        self.arm.ledger().with(
            Strategy::Group,
            Mode::native(format!(
                "{}: rank 0 decides and broadcasts, the workers replay",
                self.wire.name()
            )),
        )
    }

    fn max_batch(&self) -> usize {
        self.arm.max_batch()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;
    use std::time::{Duration, Instant};

    /// A toy arm's complete ledger: every strategy stated, none claimed.
    fn stated(arm: &'static str) -> Ledger {
        let mut ledger = Ledger::new(arm);
        for strategy in Strategy::ALL {
            ledger.set(strategy, Mode::off("the toy arm does not state it"));
        }
        ledger
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

    /// A wire that refuses prompts over `context` tokens and carries no sampler.
    struct Wire {
        context: usize,
    }

    impl GroupWire for Wire {
        fn name(&self) -> &'static str {
            "toy"
        }

        fn admit(&self, tokens: &[u32], _images: &[VisionImage]) -> Result<(), BackendError> {
            if tokens.len() > self.context {
                return Err(BackendError::new("context", "too long"));
            }
            Ok(())
        }

        fn sampler(&self, _sampling: &Sampling) -> Option<Tp4FrameSampler> {
            None
        }

        fn sampling(&self, _sampler: Option<Tp4FrameSampler>) -> Sampling {
            Sampling::default()
        }

        fn images(&self, frames: Vec<Tp4DeviceImage>) -> Result<Vec<VisionImage>, String> {
            if frames.is_empty() {
                Ok(Vec::new())
            } else {
                Err("no images here".to_owned())
            }
        }
    }

    /// A toy lane arm: a slot is the tokens its lane has consumed, and the record is every
    /// prompt, step and give-back in the order the arm ran them. `refuses` makes it refuse every
    /// give-back, as GLM's arm does; `fails` fails every step, as a driver error on one rank does.
    #[derive(Default)]
    struct ToyLanes {
        record: Mutex<Vec<String>>,
        refuses: bool,
        fails: bool,
    }

    impl crate::fused::sealed::Sealed for ToyLanes {}

    impl FusedArm for ToyLanes {
        fn ledger(&self) -> Ledger {
            stated("toy lanes")
        }

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
            if self.fails {
                return Err(BackendError::new("toy", "the device failed"));
            }
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

        fn give_back(&self, slot: &mut Self::Slot, count: usize) -> Result<(), BackendError> {
            if self.refuses {
                return Err(BackendError::new(
                    "toy",
                    "this arm holds no answers to give back",
                ));
            }
            slot.truncate(slot.len().saturating_sub(count));
            self.record
                .lock()
                .expect("the record")
                .push(format!("give back {count}"));
            Ok(())
        }

        fn speculation(&self, _slot: &Self::Slot) -> Speculation {
            Speculation::default()
        }
    }

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

    #[test]
    fn the_lane_mirror_runs_rank_zeros_prompts_steps_and_give_backs_in_order() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback listener");
        let address = listener.local_addr().expect("the bound address");
        let mirrored = Arc::new(ToyLanes::default());
        let worker = {
            let mirrored = Arc::clone(&mirrored);
            thread::spawn(move || {
                let mut control = Tp4ControlWorker::connect(address, 1, 2, Duration::from_secs(5))
                    .expect("a control connection");
                control.handshake().expect("the handshake");
                mirror_lanes(&mut control, mirrored.as_ref(), &Wire { context: 64 }, 2)
            })
        };
        let hub = Arc::new(Mutex::new(
            Tp4ControlHub::start_with_listener(listener, 2, Duration::from_secs(5))
                .expect("the hub"),
        ));
        let lanes = GroupLanes::new(ToyLanes::default(), Arc::clone(&hub), Wire { context: 64 });
        let mut first = lanes.open_slot().expect("lane 0");
        let mut second = lanes.open_slot().expect("lane 1");
        lanes.prefill(&mut first, &[1, 2]).expect("lane 0's prompt");
        lanes.prefill(&mut second, &[7]).expect("lane 1's prompt");
        // A prompt the wire refuses is never broadcast.
        lanes
            .prefill(&mut second, &[0; 65])
            .expect_err("an over-long prompt is refused");
        // Rank 1 steps the members in rank 0's order, second lane first.
        let mut members = vec![(second, 8u32), (first, 3u32)];
        lanes.step(&mut members).expect("the fused step");
        let (mut first, _) = members.pop().expect("lane 0");
        lanes.give_back(&mut first, 1).expect("the give-back");
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
            "give back 1".to_owned(),
            "prefill [5]".to_owned(),
        ];
        wait_for(
            || *mirrored.record.lock().expect("the record") == expected,
            "mirror both prompts, the step, the give-back and the renewed prompt",
        );
        assert_eq!(
            *lanes.arm().record.lock().expect("rank 0's record"),
            expected
        );
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
    fn a_give_back_the_arm_refuses_never_reaches_the_workers() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback listener");
        let address = listener.local_addr().expect("the bound address");
        let refusing = || ToyLanes {
            refuses: true,
            ..ToyLanes::default()
        };
        let mirrored = Arc::new(refusing());
        let worker = {
            let mirrored = Arc::clone(&mirrored);
            thread::spawn(move || {
                let mut control = Tp4ControlWorker::connect(address, 1, 2, Duration::from_secs(5))
                    .expect("a control connection");
                control.handshake().expect("the handshake");
                mirror_lanes(&mut control, mirrored.as_ref(), &Wire { context: 64 }, 1)
            })
        };
        let hub = Arc::new(Mutex::new(
            Tp4ControlHub::start_with_listener(listener, 2, Duration::from_secs(5))
                .expect("the hub"),
        ));
        let lanes = GroupLanes::new(refusing(), Arc::clone(&hub), Wire { context: 64 });
        let mut lane = lanes.open_slot().expect("lane 0");
        lanes.prefill(&mut lane, &[1, 2]).expect("the prompt");
        let mut members = vec![(lane, 3u32)];
        lanes.step(&mut members).expect("the step");
        let (mut lane, _) = members.pop().expect("lane 0");
        // The engine renews a slot whose give-back is refused, and steps on.
        lanes
            .give_back(&mut lane, 2)
            .expect_err("the arm refuses the give-back");
        lanes.renew(&mut lane).expect("the lane renews");
        lanes.prefill(&mut lane, &[5]).expect("the renewed prompt");
        let mut members = vec![(lane, 6u32)];
        lanes
            .step(&mut members)
            .expect("the step after the renewal");
        let expected = vec![
            "prefill [1, 2]".to_owned(),
            "step [[1, 2, 3]]".to_owned(),
            "prefill [5]".to_owned(),
            "step [[5, 6]]".to_owned(),
        ];
        wait_for(
            || *mirrored.record.lock().expect("the record") == expected,
            "mirror the renewed lane's prompt and step, with no give-back",
        );
        assert!(
            !worker.is_finished(),
            "the refused give-back ended the worker"
        );
        hub.lock()
            .expect("the hub")
            .broadcast(&Tp4ControlFrame::LaneRelease { lane: 9 })
            .expect("the frame is written");
        let error = worker
            .join()
            .expect("the worker finishes")
            .expect_err("a lane outside the pool is refused");
        assert!(error.contains("out of step"), "{error}");
    }

    fn splits(why: &str) -> ! {
        panic!("the group split: {why}")
    }

    #[test]
    fn a_step_that_fails_on_rank_zero_alone_tells_the_workers_and_ends_the_group() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback listener");
        let address = listener.local_addr().expect("the bound address");
        let mirrored = Arc::new(ToyLanes::default());
        let worker = {
            let mirrored = Arc::clone(&mirrored);
            thread::spawn(move || {
                let mut control = Tp4ControlWorker::connect(address, 1, 2, Duration::from_secs(5))
                    .expect("a control connection");
                control.handshake().expect("the handshake");
                mirror_lanes(&mut control, mirrored.as_ref(), &Wire { context: 64 }, 1)
            })
        };
        let hub = Arc::new(Mutex::new(
            Tp4ControlHub::start_with_listener(listener, 2, Duration::from_secs(5))
                .expect("the hub"),
        ));
        let failing = ToyLanes {
            fails: true,
            ..ToyLanes::default()
        };
        let lanes =
            GroupLanes::new(failing, Arc::clone(&hub), Wire { context: 64 }).with_split(splits);
        let mut lane = lanes.open_slot().expect("lane 0");
        lanes.prefill(&mut lane, &[1, 2]).expect("the prompt");
        let mut members = vec![(lane, 3u32)];
        // The worker has stepped past rank 0, so rank 0 does not hand the error back to serve on.
        let split = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = lanes.step(&mut members);
        }))
        .expect_err("the failed step splits the group");
        let why = split.downcast_ref::<String>().expect("the split's message");
        assert!(why.contains("the device failed"), "{why}");
        let error = worker
            .join()
            .expect("the worker finishes")
            .expect_err("rank 0's error ends the worker");
        assert!(error.contains("group_out_of_step"), "{error}");
        assert!(error.contains("the device failed"), "{error}");
        assert_eq!(
            *mirrored.record.lock().expect("the record"),
            vec!["prefill [1, 2]".to_owned(), "step [[1, 2, 3]]".to_owned()]
        );
    }
}

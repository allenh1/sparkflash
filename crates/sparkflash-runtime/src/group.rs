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
//! which lanes step together, and [GroupLanes] broadcasts each decision (a lane's prompt, a step's
//! members in their row order, a give-back, a release) *before* running it on its own rank. A
//! worker runs [mirror_lanes] over its own lane arm and replays the frames in the order they came,
//! so lane `n` is the same sequence on every rank and the collectives line up.
//!
//! This came from Inkling's group (`sparkflash-model-inkling2/src/group.rs`); what stays with an
//! arm is its [GroupWire]: what rank 0 refuses before it broadcasts, how a request's sampler
//! crosses the wire, and how a worker reads a prompt's image spans.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use sparkflash_comm::control::{
    Tp4ControlFrame, Tp4ControlHub, Tp4ControlWorker, Tp4DeviceImage, Tp4FrameSampler,
    control_read_is_idle,
};
use sparkflash_model::backend::{BackendError, Sampling, Speculation, StepOutput};
use sparkflash_model::vision::{ImageSpan, VisionImage};

use crate::fused::FusedArm;

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
                arm.prefill_images(slot, &prompt_ids, &images)
                    .map_err(|error| format!("lane {lane}: prefill: {error}"))?;
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
                    "the fused worker expected LaneRequest, LaneRelease, LaneStep, LaneGiveBack or \
                     Error, not {}",
                    other.kind()
                ));
            }
        }
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
        }
    }

    /// The wrapped arm.
    pub fn arm(&self) -> &A {
        &self.arm
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

    fn prefill_images(
        &self,
        slot: &mut Self::Slot,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<StepOutput, BackendError> {
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
        answer
    }

    /// Local: rank 0's lane holds what every rank's does.
    fn ready(&self, slot: &Self::Slot, token: u32) -> bool {
        slot.slot
            .as_ref()
            .is_some_and(|inner| self.arm.ready(inner, token))
    }

    /// Broadcast: every rank rolls the same lane back by the same count.
    fn give_back(&self, slot: &mut Self::Slot, count: usize) -> Result<(), BackendError> {
        if count == 0 {
            return Ok(());
        }
        if self.wire.trace() {
            eprintln!(
                "[{}] serve rank 0 lane {} gives back {count}",
                self.wire.name(),
                slot.lane
            );
        }
        let frame_count = u32::try_from(count)
            .map_err(|_| BackendError::new("group", format!("{count} answers to give back")))?;
        self.send(&Tp4ControlFrame::LaneGiveBack {
            lane: slot.lane,
            count: frame_count,
        })?;
        self.arm.give_back(Self::slot(slot)?, count)
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
        if self.wire.trace() {
            eprintln!(
                "[{}] serve rank 0 lane {} release",
                self.wire.name(),
                slot.lane
            );
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;
    use std::time::{Duration, Instant};

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
    /// prompt, step and give-back in the order the arm ran them.
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

        fn give_back(&self, slot: &mut Self::Slot, count: usize) -> Result<(), BackendError> {
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
}

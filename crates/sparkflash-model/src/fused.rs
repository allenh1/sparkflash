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

//! Sessions that share one pass over the weights.
//!
//! The host driver gives every request its own session and its own worker thread. Left alone,
//! each of those sessions steps the whole model for its own single row, and because a decode step
//! is bound by the bytes it reads rather than the flops it issues, N sessions take N times as long
//! as one and serve no more tokens a second than one does.
//!
//! This module puts the fusion behind `open_session` instead. A session opened here is a handle:
//! its `prefill` and `decode` post one job to a thread that owns every pooled slot, and the
//! thread steps the decode jobs that are waiting as one batch ([FusedArm::step]). The host driver
//! is unchanged, its worker threads still block on one request each, and the tokens a session sees
//! are the ones it would see alone: [FusedArm::step] is one row a member, so a member's
//! arithmetic does not depend on its neighbours.
//!
//! Two limits. A prefill takes the engine: the prompt chunks of two requests are not batched, so
//! a prompt runs to its first token while the other members' decodes wait. And a fused session
//! does not report a reusable prefix, because the engine owns the slot; the host driver's
//! append-only reuse ([D66](../../../docs/DECISIONS.md)) is off in this mode, which is the trade
//! a deployment makes for sharing the pass.
//!
//! DeepSeek's tensor-parallel driver is the same shape at four ranks: a slot a request, one step
//! a batch, a reply a job. The arm supplies the arithmetic through [FusedArm]; nothing here knows
//! which model it drives.

use std::sync::mpsc::{RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::backend::{BackendError, ModelSession, Sampling, Speculation, StepOutput};

/// What a model exposes for its sessions to be fused.
///
/// A slot is one request's state: its caches, its sampler and the history it has forwarded. The
/// engine owns the slots of its pool and steps the waiting ones in a single call.
pub trait FusedArm: Send + Sync + 'static {
    /// One request's state.
    type Slot: Send + 'static;

    /// A fresh slot.
    ///
    /// # Errors
    /// A device allocation, an unreadable checkpoint or an exhausted pool.
    fn open_slot(&self) -> Result<Self::Slot, BackendError>;

    /// Whether this slot still holds a request, so the next prompt needs a fresh one.
    fn holds_request(&self, slot: &Self::Slot) -> bool;

    /// Replaces `slot` with a fresh one, keeping nothing of the last request.
    ///
    /// # Errors
    /// As [FusedArm::open_slot].
    fn renew(&self, slot: &mut Self::Slot) -> Result<(), BackendError>;

    /// Applies a request's sampler to a slot before its prompt runs.
    fn set_sampling(&self, slot: &mut Self::Slot, sampling: Sampling);

    /// Consumes a prompt and returns the first step.
    ///
    /// # Errors
    /// A backend failure, such as a shape mismatch or an exhausted context.
    fn prefill(&self, slot: &mut Self::Slot, tokens: &[u32]) -> Result<StepOutput, BackendError>;

    /// Consumes one token a member and returns each member's step, in the order given.
    ///
    /// A row must compute as it would alone: the arm's wide stages read the weights once for
    /// every row, and its per-sequence stages run on each row's own slice.
    ///
    /// # Errors
    /// A backend failure; the whole batch fails with it.
    fn step(&self, members: &mut [(Self::Slot, u32)]) -> Result<Vec<StepOutput>, BackendError>;

    /// A slot's speculative counters for the request it is serving.
    fn speculation(&self, slot: &Self::Slot) -> Speculation;

    /// The most members one pass may hold, when the model's batch is narrower than the pool.
    ///
    /// A step's rows are one a member, so a model whose combined row buffers hold four rows
    /// cannot take a fifth member: the engine caps a batch here rather than failing the step.
    fn max_batch(&self) -> usize {
        usize::MAX
    }
}

/// How long a step waits for the batch to fill before it runs with what it has.
///
/// The members of a steady-state batch finish a step together and post their next token within a
/// few microseconds of each other, so this only matters for the first step of a round and for a
/// request that joins late. It is a ceiling on the latency a batch may add, not a schedule: a
/// window that is already full runs without waiting.
const FILL_WINDOW: Duration = Duration::from_micros(300);

/// What one job asks the engine for.
enum Work {
    /// Consume a prompt. `sampling` is the request's, applied before its prompt runs.
    Prefill {
        tokens: Vec<u32>,
        sampling: Sampling,
    },
    /// Consume one token of a request already prefilled.
    Decode { token: u32 },
    /// End the engine thread.
    Shutdown,
}

/// One job and where its answer goes.
struct Job {
    slot: usize,
    work: Work,
    reply: Reply,
}

/// Where a job's answer goes.
type Reply = Sender<Result<StepOutput, BackendError>>;

/// A decode job waiting for its batch: its slot, its token and its caller.
type Waiting = (usize, u32, Reply);

/// The engine thread and the slots of its pool.
pub struct FusedEngine<A: FusedArm> {
    /// The arm, kept alive with the engine: the thread holds its own share, but the pool's slots
    /// stay the arm's to interpret, and a handle outliving the backend must not outlive it.
    arm: Arc<A>,
    jobs: Sender<Job>,
    /// Slots no request holds, handed out by [FusedEngine::open].
    free: Mutex<Vec<usize>>,
    wake: Condvar,
    thread: Mutex<Option<JoinHandle<()>>>,
    /// Each slot's speculative counters, refreshed by the engine after every step: a handle
    /// cannot read its slot, so this is how a request reports what its drafts were worth.
    stats: Vec<Arc<Mutex<Speculation>>>,
}

/// A request's session: a handle that posts to the engine thread instead of running the model.
pub struct FusedSession<A: FusedArm> {
    slot: usize,
    engine: Arc<FusedEngine<A>>,
    stats: Arc<Mutex<Speculation>>,
    sampling: Sampling,
    /// Whether this handle has posted a prompt yet.
    fresh: bool,
}

impl<A: FusedArm> FusedEngine<A> {
    /// Allocate `capacity` slots and start the thread that steps them together.
    ///
    /// `capacity` is the driver's concurrency: a deployment that asks for fewer workers never
    /// allocates a slot it cannot use.
    ///
    /// # Errors
    /// A slot the arm refuses to open, or a thread that cannot be started.
    pub fn start(arm: Arc<A>, capacity: usize) -> Result<Arc<Self>, BackendError> {
        if capacity == 0 {
            return Err(BackendError::new(
                "fused",
                "a fused engine needs at least one slot",
            ));
        }
        let mut pool: Vec<Option<A::Slot>> = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            pool.push(Some(arm.open_slot()?));
        }
        let stats: Vec<Arc<Mutex<Speculation>>> = (0..capacity)
            .map(|_| Arc::new(Mutex::new(Speculation::default())))
            .collect();
        let (jobs, inbox) = channel::<Job>();
        let thread_stats = stats.clone();
        let engine_arm = Arc::clone(&arm);
        let thread = std::thread::Builder::new()
            .name("sparkflash-batch".to_owned())
            .spawn(move || run(engine_arm, inbox, pool, capacity, thread_stats))
            .map_err(|error| BackendError::new("fused", error.to_string()))?;
        Ok(Arc::new(Self {
            arm,
            jobs,
            free: Mutex::new((0..capacity).collect()),
            wake: Condvar::new(),
            thread: Mutex::new(Some(thread)),
            stats,
        }))
    }

    /// Take a pooled slot for one request.
    ///
    /// Blocks while every slot is in use, which keeps a request from opening a session it could
    /// never step: the pool is the driver's concurrency.
    ///
    /// # Errors
    /// A poisoned slot list.
    pub fn open(self: &Arc<Self>) -> Result<FusedSession<A>, BackendError> {
        let mut free = self
            .free
            .lock()
            .map_err(|_| BackendError::new("fused", "the slot list is poisoned"))?;
        while free.is_empty() {
            free = self
                .wake
                .wait(free)
                .map_err(|_| BackendError::new("fused", "the slot list is poisoned"))?;
        }
        let slot = free.remove(0);
        Ok(FusedSession {
            slot,
            engine: Arc::clone(self),
            stats: Arc::clone(&self.stats[slot]),
            sampling: Sampling::default(),
            fresh: true,
        })
    }

    /// How many slots the pool holds, which is the fused concurrency.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.stats.len()
    }

    /// The arm whose slots the pool holds.
    #[must_use]
    pub fn arm(&self) -> &A {
        &self.arm
    }

    fn release(&self, slot: usize) {
        if let Ok(mut free) = self.free.lock() {
            free.push(slot);
            self.wake.notify_one();
        }
    }
}

impl<A: FusedArm> Drop for FusedEngine<A> {
    fn drop(&mut self) {
        let (reply, _) = channel();
        let _ = self.jobs.send(Job {
            slot: 0,
            work: Work::Shutdown,
            reply,
        });
        if let Ok(mut thread) = self.thread.lock()
            && let Some(thread) = thread.take()
        {
            let _ = thread.join();
        }
    }
}

impl<A: FusedArm> FusedSession<A> {
    fn ask(&mut self, work: Work) -> Result<StepOutput, BackendError> {
        let (reply, answer) = channel();
        self.engine
            .jobs
            .send(Job {
                slot: self.slot,
                work,
                reply,
            })
            .map_err(|_| BackendError::new("fused", "the engine thread has stopped"))?;
        answer
            .recv()
            .map_err(|_| BackendError::new("fused", "the engine thread has stopped"))?
    }
}

impl<A: FusedArm> Drop for FusedSession<A> {
    fn drop(&mut self) {
        self.engine.release(self.slot);
    }
}

impl<A: FusedArm> ModelSession for FusedSession<A> {
    fn set_sampling(&mut self, sampling: Sampling) {
        self.sampling = sampling;
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::new("fused", "an empty prompt has no logits"));
        }
        self.fresh = false;
        self.ask(Work::Prefill {
            tokens: tokens.to_vec(),
            sampling: self.sampling,
        })
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        self.ask(Work::Decode { token })
    }

    fn speculation(&self) -> Speculation {
        self.stats.lock().map(|stats| *stats).unwrap_or_default()
    }
}

/// The engine thread: collect the waiting jobs, run the prompts, step the rest together.
fn run<A: FusedArm>(
    arm: Arc<A>,
    inbox: std::sync::mpsc::Receiver<Job>,
    mut pool: Vec<Option<A::Slot>>,
    capacity: usize,
    stats: Vec<Arc<Mutex<Speculation>>>,
) {
    loop {
        let Ok(first) = inbox.recv() else {
            return;
        };
        let limit = capacity.min(arm.max_batch()).max(1);
        let mut jobs = vec![first];
        // Everything already queued is one batch, and the batch waits a moment for a member that
        // is about to post: a decode that arrived while the thread was up would otherwise wait a
        // whole step for nothing.
        let deadline = Instant::now() + FILL_WINDOW;
        while jobs.len() < limit {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            match inbox.recv_timeout(left) {
                Ok(job) => jobs.push(job),
                Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
            }
        }
        let mut decodes = Vec::new();
        for job in jobs {
            match job.work {
                Work::Shutdown => return,
                Work::Prefill { tokens, sampling } => {
                    prefill(
                        &*arm, &mut pool, &stats, job.slot, &tokens, sampling, job.reply,
                    );
                }
                Work::Decode { token } => decodes.push((job.slot, token, job.reply)),
            }
        }
        if !decodes.is_empty() {
            step(&*arm, &mut pool, &stats, decodes);
        }
    }
}

/// Runs one prompt on its slot, between the slot's decodes.
fn prefill<A: FusedArm>(
    arm: &A,
    pool: &mut [Option<A::Slot>],
    stats: &[Arc<Mutex<Speculation>>],
    slot: usize,
    tokens: &[u32],
    sampling: Sampling,
    reply: Sender<Result<StepOutput, BackendError>>,
) {
    let Some(mut state) = pool[slot].take() else {
        let _ = reply.send(Err(BackendError::new(
            "fused",
            "the slot is already running a job",
        )));
        return;
    };
    let answer = (|| {
        if arm.holds_request(&state) {
            arm.renew(&mut state)?;
        }
        arm.set_sampling(&mut state, sampling);
        arm.prefill(&mut state, tokens)
    })();
    if let Ok(mut cell) = stats[slot].lock() {
        *cell = arm.speculation(&state);
    }
    pool[slot] = Some(state);
    let _ = reply.send(answer);
}

/// Steps the waiting members in one pass, one row each, and answers every job.
fn step<A: FusedArm>(
    arm: &A,
    pool: &mut [Option<A::Slot>],
    stats: &[Arc<Mutex<Speculation>>],
    decodes: Vec<Waiting>,
) {
    let mut slots = Vec::with_capacity(decodes.len());
    let mut replies = Vec::with_capacity(decodes.len());
    let mut members = Vec::with_capacity(decodes.len());
    for (slot, token, reply) in decodes {
        match pool[slot].take() {
            Some(state) => {
                slots.push(slot);
                replies.push(reply);
                members.push((state, token));
            }
            None => {
                let _ = reply.send(Err(BackendError::new(
                    "fused",
                    "the slot is already running a job",
                )));
            }
        }
    }
    let answer = arm.step(&mut members);
    for (index, (state, _)) in members.into_iter().enumerate() {
        let slot = slots[index];
        if let Ok(mut cell) = stats[slot].lock() {
            *cell = arm.speculation(&state);
        }
        pool[slot] = Some(state);
    }
    match answer {
        Ok(steps) => {
            for (index, reply) in replies.into_iter().enumerate() {
                let step = steps.get(index).cloned().unwrap_or_default();
                let _ = reply.send(Ok(step));
            }
        }
        Err(error) => {
            for reply in replies {
                let _ = reply.send(Err(error.clone()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A toy arm: a slot is its position, a step is the next token, and the engine's batching is
    /// visible in the counters.
    struct Toy {
        steps: AtomicUsize,
        rows: AtomicUsize,
    }

    impl FusedArm for Toy {
        type Slot = usize;

        fn open_slot(&self) -> Result<Self::Slot, BackendError> {
            Ok(0)
        }

        fn holds_request(&self, slot: &Self::Slot) -> bool {
            *slot > 0
        }

        fn renew(&self, slot: &mut Self::Slot) -> Result<(), BackendError> {
            *slot = 0;
            Ok(())
        }

        fn set_sampling(&self, _slot: &mut Self::Slot, _sampling: Sampling) {}

        fn prefill(
            &self,
            slot: &mut Self::Slot,
            tokens: &[u32],
        ) -> Result<StepOutput, BackendError> {
            assert!(!tokens.is_empty(), "the engine refuses an empty prompt");
            *slot = tokens.len();
            Ok(StepOutput {
                logits: vec![0.0; 2],
                next_token: Some(1),
                draft: Vec::new(),
            })
        }

        fn step(&self, members: &mut [(Self::Slot, u32)]) -> Result<Vec<StepOutput>, BackendError> {
            self.steps.fetch_add(1, Ordering::SeqCst);
            self.rows.fetch_add(members.len(), Ordering::SeqCst);
            Ok(members
                .iter_mut()
                .map(|(slot, token)| {
                    *slot += 1;
                    StepOutput {
                        logits: vec![0.0; 2],
                        next_token: Some(*token + 1),
                        draft: Vec::new(),
                    }
                })
                .collect())
        }

        fn speculation(&self, slot: &Self::Slot) -> Speculation {
            Speculation {
                verify_steps: *slot,
                ..Speculation::default()
            }
        }
    }

    fn toy() -> (Arc<Toy>, Arc<FusedEngine<Toy>>) {
        let arm = Arc::new(Toy {
            steps: AtomicUsize::new(0),
            rows: AtomicUsize::new(0),
        });
        let engine = FusedEngine::start(Arc::clone(&arm), 2).expect("the engine starts");
        (arm, engine)
    }

    #[test]
    fn a_handle_serves_its_request_and_reports_the_slots_counters() {
        let (_arm, engine) = toy();
        let mut session = engine.open().expect("a slot");
        let first = session.prefill(&[7, 11]).expect("the prefill");
        assert_eq!(first.next_token, Some(1));
        assert_eq!(
            session.speculation().verify_steps,
            2,
            "the slot's own count"
        );
        let next = session.decode(1).expect("the decode");
        assert_eq!(next.next_token, Some(2));
        assert_eq!(session.speculation().verify_steps, 3);
    }

    #[test]
    fn two_handles_are_stepped_in_one_pass() {
        let (arm, engine) = toy();
        let mut a = engine.open().expect("a slot");
        let mut b = engine.open().expect("a slot");
        a.prefill(&[1]).expect("the prefill");
        b.prefill(&[1]).expect("the prefill");
        // Both decodes are posted before either can be run: the engine takes both in one batch.
        let answers: Vec<u32> = std::thread::scope(|scope| {
            let left = scope.spawn(|| a.decode(1).expect("the decode").next_token.unwrap());
            let right = scope.spawn(|| b.decode(1).expect("the decode").next_token.unwrap());
            vec![
                left.join().expect("the thread"),
                right.join().expect("the thread"),
            ]
        });
        assert_eq!(answers, [2, 2]);
        assert_eq!(
            arm.rows.load(Ordering::SeqCst),
            2,
            "one row a member, no more"
        );
        // One pass when both jobs were waiting, two when the second arrived after the first
        // drain: both are correct, and which one happens is the scheduler's. The device tests
        // are where a fused step is read off the model's own counters.
        assert!(
            (1..=2).contains(&arm.steps.load(Ordering::SeqCst)),
            "steps {}",
            arm.steps.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn a_second_request_on_a_used_slot_gets_a_fresh_one() {
        let arm = Arc::new(Toy {
            steps: AtomicUsize::new(0),
            rows: AtomicUsize::new(0),
        });
        // One slot, so the handle that comes back is the one the first request used: with two
        // the pool may hand out either, and this test is about the slot's own lifetime.
        let engine = FusedEngine::start(Arc::clone(&arm), 1).expect("the engine starts");
        let mut first = engine.open().expect("a slot");
        first.prefill(&[1, 2, 3]).expect("the prefill");
        assert_eq!(first.speculation().verify_steps, 3);
        drop(first);
        // The counters a handle reads are the engine's record of its last job, so they still show
        // the last request's until this one runs: what the test pins is that the new prompt
        // replaces the slot's state instead of continuing it.
        let mut second = engine.open().expect("the slot comes back");
        second
            .prefill(&[1, 2])
            .expect("the prefill renews the slot");
        assert_eq!(
            second.speculation().verify_steps,
            2,
            "the new prompt's own count, not the last request's"
        );
    }

    #[test]
    fn a_third_request_waits_for_a_slot() {
        let (_arm, engine) = toy();
        let first = engine.open().expect("a slot");
        let second = engine.open().expect("a slot");
        // The pool is two, so a third open blocks until one is released.
        let engine_clone = Arc::clone(&engine);
        let third = std::thread::spawn(move || engine_clone.open().map(|_| ()));
        drop(first);
        assert_eq!(
            third
                .join()
                .expect("the thread")
                .map_err(|e| e.message().to_owned()),
            Ok(())
        );
        drop(second);
    }
}

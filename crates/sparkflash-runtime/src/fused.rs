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
//! A prefill takes the engine: the prompt chunks of two requests are not batched, so a prompt runs
//! to its first token while the other members' decodes wait.
//!
//! A fused session reports a reusable prefix when its arm keeps one ([FusedArm::history]): the
//! engine then hands the slot's next prompt to the arm as it stands, the arm continues what the
//! prompt shares with the slot or starts over, as its own session would, and the host driver's
//! append-only reuse ([D66](../../../docs/DECISIONS.md)) works as it does for a serial session.
//! The engine publishes each slot's held lengths after every job ([Kept]); the handle keeps the
//! ids it fed, so it can name the prefix without the engine copying a history a step. An arm that
//! keeps nothing (the default) has its slot renewed before every prompt.
//!
//! DeepSeek's tensor-parallel driver is the same shape at four ranks: a slot a request, one step
//! a batch, a reply a job. The arm supplies the arithmetic through [FusedArm]; nothing here knows
//! which model it drives.

use std::sync::mpsc::{RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use sparkflash_model::backend::{BackendError, ModelSession, Sampling, Speculation, StepOutput};
use sparkflash_model::vision::{ImageSpan, VisionImage};

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

    /// Consumes a prompt whose image spans take `images`' features, as
    /// [ModelSession::prefill_images]. The default refuses any image.
    ///
    /// # Errors
    /// As [FusedArm::prefill], and an arm without a vision path refuses images.
    fn prefill_images(
        &self,
        slot: &mut Self::Slot,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<StepOutput, BackendError> {
        if images.is_empty() {
            return self.prefill(slot, tokens);
        }
        Err(BackendError::new(
            "vision",
            format!(
                "this arm has no vision path for the request's {} images",
                images.len()
            ),
        ))
    }

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

    /// The tokens a slot's sequence holds and can continue from, as
    /// [ModelSession::reusable_prefix] names them for a session.
    ///
    /// `None` (the default) says the slot keeps nothing a later prompt may reuse, and the engine
    /// renews it before its next prompt. An arm that answers continues a slot itself: its prefill
    /// reuses what the prompt shares with the history, or starts over. The history must be the
    /// slot's last prompt followed by the tokens stepped since, cut to the length reported: the
    /// handle names it from the ids it fed.
    fn history<'s>(&self, slot: &'s Self::Slot) -> Option<&'s [u32]> {
        let _ = slot;
        None
    }

    /// A shorter prefix of [FusedArm::history] the slot can go back to, as
    /// [ModelSession::rewind_prefix] names it.
    fn rewind_prefix<'s>(&self, slot: &'s Self::Slot) -> Option<&'s [u32]> {
        let _ = slot;
        None
    }

    /// The image spans [FusedArm::history] holds, as [ModelSession::held_images] names them.
    fn held_images<'s>(&self, slot: &'s Self::Slot) -> &'s [ImageSpan] {
        let _ = slot;
        &[]
    }

    /// How many of the last prompt's tokens the slot already held, as
    /// [ModelSession::cached_tokens] counts them.
    fn cached_tokens(&self, slot: &Self::Slot) -> usize {
        let _ = slot;
        0
    }

    /// Whether the slot can take `prompt` (or, for `None`, its next step) with the memory the arm
    /// has spare now. The default always can, for an arm whose slots own their memory whole.
    ///
    /// The engine asks before a job runs, and when the answer is no it gives back the slot that
    /// has idled longest ([FusedArm::release]) and asks again, so several conversations share
    /// memory one long one would fill. An arm whose prefill or step fails on a slot without room
    /// must answer here as that failure would.
    fn room(&self, slot: &Self::Slot, prompt: Option<&[u32]>) -> bool {
        let _ = (slot, prompt);
        true
    }

    /// Gives up what an idle slot holds, so another can grow: its sequence is forgotten and the
    /// next prompt starts over. The default renews it.
    ///
    /// # Errors
    /// A backend failure.
    fn release(&self, slot: &mut Self::Slot) -> Result<(), BackendError> {
        self.renew(slot)
    }

    /// Whether the slot already holds its answer for a step fed `token` - a verify's row the
    /// round kept - so stepping it costs no device work. The default holds none.
    ///
    /// After a step the engine keeps feeding each member the token it just chose while this
    /// says yes, and hands the whole run to the member's handle in one reply: a drafting arm's
    /// member drains its round in one engine step, so every member starts its next round at the
    /// same step and one walk serves them all. An arm that answers yes must support
    /// [FusedArm::give_back] for the answers fed that way.
    fn ready(&self, slot: &Self::Slot, token: u32) -> bool {
        let _ = (slot, token);
        false
    }

    /// The caller never used the last `count` steps the slot answered (its run ended early: a stop
    /// string, the token limit, a forced token): roll them back, so the slot's history ends where
    /// the caller's does. They are always answers [FusedArm::ready] said were held, so they lie in
    /// the round that produced them. The default refuses any count but zero, as an arm that holds
    /// no answers never runs more than one step a reply.
    ///
    /// # Errors
    /// An arm that cannot give back what it was asked to.
    fn give_back(&self, slot: &mut Self::Slot, count: usize) -> Result<(), BackendError> {
        let _ = slot;
        if count == 0 {
            Ok(())
        } else {
            Err(BackendError::new(
                "fused",
                format!("this arm holds no answers to give {count} back"),
            ))
        }
    }

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
    /// Consume a prompt. `sampling` is the request's, applied before its prompt runs; `images`
    /// fill the prompt's image spans.
    Prefill {
        tokens: Vec<u32>,
        images: Vec<VisionImage>,
        sampling: Sampling,
    },
    /// Consume one token of a request already prefilled.
    Decode { token: u32 },
    /// The slot's request is over: what it holds may be given up for another ([FusedArm::release]).
    Park,
    /// The handle never used the last `count` answers of its run ([FusedArm::give_back]); posted
    /// before the job that follows them.
    GiveBack { count: usize },
    /// End the engine thread.
    Shutdown,
}

/// One job and where its answer goes.
struct Job {
    slot: usize,
    work: Work,
    reply: Reply,
}

/// Where a job's answer goes: a prompt's first step, or a decode's run (its step, then every step
/// the slot already held, [FusedArm::ready]).
type Reply = Sender<Result<Vec<StepOutput>, BackendError>>;

/// A decode job waiting for its batch: its slot, its token and its caller.
type Waiting = (usize, u32, Reply);

/// What a slot holds after its last job, as the engine publishes it: the lengths of
/// [FusedArm::history] and [FusedArm::rewind_prefix], the images the history holds and the last
/// prompt's hit. The handle names the prefixes from the ids it fed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Kept {
    history: Option<usize>,
    rewind: Option<usize>,
    images: Vec<ImageSpan>,
    cached: usize,
}

/// A slot's record after its last job: its speculative counters and what it holds.
#[derive(Debug, Clone, Default)]
struct Record {
    speculation: Speculation,
    kept: Kept,
}

/// Each slot's record, refreshed by the engine after every job: a handle cannot read its slot.
type Records = Vec<Arc<Mutex<Record>>>;

/// Refreshes `cell` from `state` after a job; a prompt (`prompt`) may change the images too.
fn record<A: FusedArm>(arm: &A, state: &A::Slot, cell: &Mutex<Record>, prompt: bool) {
    if let Ok(mut cell) = cell.lock() {
        cell.speculation = arm.speculation(state);
        cell.kept.history = arm.history(state).map(<[u32]>::len);
        cell.kept.rewind = arm.rewind_prefix(state).map(<[u32]>::len);
        cell.kept.cached = arm.cached_tokens(state);
        if prompt {
            cell.kept.images = arm.held_images(state).to_vec();
        }
    }
}

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
    /// Each slot's record, refreshed by the engine after every job: a handle cannot read its
    /// slot, so this is how a request reports what its drafts were worth and what it holds.
    stats: Records,
}

/// A request's session: a handle that posts to the engine thread instead of running the model.
pub struct FusedSession<A: FusedArm> {
    slot: usize,
    engine: Arc<FusedEngine<A>>,
    stats: Arc<Mutex<Record>>,
    sampling: Sampling,
    /// Whether this handle has posted a prompt yet.
    fresh: bool,
    /// The ids this handle fed its slot: its last prompt and every token decoded since. The slot's
    /// history is a prefix of them ([FusedArm::history]).
    fed: Vec<u32>,
    /// What the slot held after this handle's last job.
    kept: Kept,
    /// The rest of the last decode's run, answered here while the caller feeds the token each
    /// answer chose; `next` is the token the last answer chose.
    run: std::collections::VecDeque<StepOutput>,
    next: Option<u32>,
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
        let stats: Records = (0..capacity)
            .map(|_| Arc::new(Mutex::new(Record::default())))
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
            fed: Vec::new(),
            kept: Kept::default(),
            run: std::collections::VecDeque::new(),
            next: None,
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
    /// Gives the unused rest of the last run back before the next job ([Work::GiveBack]), and
    /// says whether there was any.
    fn drop_run(&mut self) -> bool {
        let count = self.run.len();
        self.run.clear();
        self.next = None;
        if count > 0 {
            self.post(Work::GiveBack { count });
        }
        count > 0
    }

    fn ask(&mut self, work: Work) -> Result<Vec<StepOutput>, BackendError> {
        let (reply, answer) = channel();
        self.engine
            .jobs
            .send(Job {
                slot: self.slot,
                work,
                reply,
            })
            .map_err(|_| BackendError::new("fused", "the engine thread has stopped"))?;
        let answer = answer
            .recv()
            .map_err(|_| BackendError::new("fused", "the engine thread has stopped"))?;
        if let Ok(cell) = self.stats.lock() {
            self.kept.clone_from(&cell.kept);
        }
        answer
    }

    /// The first `length` ids this handle fed, when it has fed that many.
    fn prefix(&self, length: Option<usize>) -> Option<&[u32]> {
        if self.fresh {
            return None;
        }
        length
            .filter(|length| *length <= self.fed.len())
            .map(|length| &self.fed[..length])
    }

    /// The slot's record as it stands: the engine may have given a parked slot's sequence up
    /// since this handle's last job.
    fn live(&self) -> Kept {
        self.stats
            .lock()
            .map(|cell| cell.kept.clone())
            .unwrap_or_default()
    }

    /// Posts a job whose answer nobody waits for.
    fn post(&self, work: Work) {
        let (reply, _) = channel();
        let _ = self.engine.jobs.send(Job {
            slot: self.slot,
            work,
            reply,
        });
    }
}

impl<A: FusedArm> Drop for FusedSession<A> {
    fn drop(&mut self) {
        // Parked before the slot goes back to the list: the jobs run in order, so the next
        // handle's first job finds the slot parked and takes it back.
        self.drop_run();
        self.post(Work::Park);
        self.engine.release(self.slot);
    }
}

impl<A: FusedArm> ModelSession for FusedSession<A> {
    fn set_sampling(&mut self, sampling: Sampling) {
        self.sampling = sampling;
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        self.prefill_images(tokens, &[])
    }

    fn prefill_images(
        &mut self,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<StepOutput, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::new("fused", "an empty prompt has no logits"));
        }
        self.fresh = false;
        self.drop_run();
        let answer = self
            .ask(Work::Prefill {
                tokens: tokens.to_vec(),
                images: images.to_vec(),
                sampling: self.sampling,
            })
            .and_then(first);
        if let Ok(step) = &answer {
            self.fed.clear();
            self.fed.extend_from_slice(tokens);
            self.next = step.next_token;
        }
        answer
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        // The run's next answer is the step this token would get, while the caller feeds the
        // token the last answer chose.
        if self.next == Some(token)
            && let Some(step) = self.run.pop_front()
        {
            self.fed.push(token);
            self.next = step.next_token;
            return Ok(step);
        }
        self.drop_run();
        let answer = self.ask(Work::Decode { token });
        match answer {
            Ok(steps) => {
                self.fed.push(token);
                let mut steps = std::collections::VecDeque::from(steps);
                let step = steps.pop_front().unwrap_or_default();
                self.next = step.next_token;
                self.run = steps;
                Ok(step)
            }
            Err(error) => Err(error),
        }
    }

    fn speculation(&self) -> Speculation {
        self.stats
            .lock()
            .map(|record| record.speculation)
            .unwrap_or_default()
    }

    fn reusable_prefix(&self) -> Option<&[u32]> {
        self.prefix(self.live().history)
    }

    fn rewind_prefix(&self) -> Option<&[u32]> {
        self.prefix(self.live().rewind)
    }

    fn held_images(&self) -> &[ImageSpan] {
        &self.kept.images
    }

    fn cached_tokens(&self) -> usize {
        self.kept.cached
    }

    fn park(&mut self) {
        // A run cut short is given back before the park, and waited for: the pool claims the
        // next request against the slot's record, which must hold the caller's history, not the
        // run's.
        if self.drop_run() {
            let _ = self.ask(Work::Park);
        } else {
            self.post(Work::Park);
        }
    }
}

/// A prompt's reply is its one step.
fn first(steps: Vec<StepOutput>) -> Result<StepOutput, BackendError> {
    steps
        .into_iter()
        .next()
        .ok_or_else(|| BackendError::new("fused", "the engine answered no step"))
}

/// The engine thread: collect the waiting jobs, run the prompts, step the rest together.
fn run<A: FusedArm>(
    arm: Arc<A>,
    inbox: std::sync::mpsc::Receiver<Job>,
    mut pool: Vec<Option<A::Slot>>,
    capacity: usize,
    stats: Records,
) {
    let mut idle = Idle::new(capacity);
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
                Work::Prefill {
                    tokens,
                    images,
                    sampling,
                } => {
                    let prompt = Prompt {
                        tokens: &tokens,
                        images: &images,
                        sampling,
                    };
                    idle.busy(job.slot);
                    let mut room = Room {
                        arm: &*arm,
                        pool: &mut pool,
                        stats: &stats,
                        idle: &mut idle,
                    };
                    room.make(job.slot, Some(&tokens));
                    prefill(&*arm, &mut pool, &stats, job.slot, prompt, job.reply);
                }
                Work::Decode { token } => {
                    idle.busy(job.slot);
                    decodes.push((job.slot, token, job.reply));
                }
                Work::Park => {
                    idle.park(job.slot);
                    let _ = job.reply.send(Ok(Vec::new()));
                }
                Work::GiveBack { count } => {
                    give_back(&*arm, &mut pool, &stats, job.slot, count);
                }
            }
        }
        if !decodes.is_empty() {
            let mut room = Room {
                arm: &*arm,
                pool: &mut pool,
                stats: &stats,
                idle: &mut idle,
            };
            let decodes = room.steps(decodes);
            if !decodes.is_empty() {
                step(&*arm, &mut pool, &stats, decodes);
            }
        }
    }
}

/// When each parked slot was parked, in the engine's job order.
struct Idle {
    parked: Vec<Option<u64>>,
    clock: u64,
}

impl Idle {
    fn new(capacity: usize) -> Self {
        Self {
            parked: vec![None; capacity],
            clock: 0,
        }
    }

    fn park(&mut self, slot: usize) {
        self.clock += 1;
        if let Some(parked) = self.parked.get_mut(slot) {
            *parked = Some(self.clock);
        }
    }

    fn busy(&mut self, slot: usize) {
        if let Some(parked) = self.parked.get_mut(slot) {
            *parked = None;
        }
    }
}

/// What the engine reclaims memory with: the parked slots, least recently parked first.
struct Room<'r, A: FusedArm> {
    arm: &'r A,
    pool: &'r mut [Option<A::Slot>],
    stats: &'r [Arc<Mutex<Record>>],
    idle: &'r mut Idle,
}

impl<A: FusedArm> Room<'_, A> {
    /// Gives up parked slots, the longest parked first, until `slot` has room for `prompt` (or
    /// its next step). False when every parked slot is given up and there is still no room: the
    /// job then fails as the arm refuses it.
    fn make(&mut self, slot: usize, prompt: Option<&[u32]>) -> bool {
        loop {
            let Some(state) = self.pool[slot].as_ref() else {
                return false;
            };
            if self.arm.room(state, prompt) {
                return true;
            }
            let victim = self
                .idle
                .parked
                .iter()
                .enumerate()
                .filter(|&(other, parked)| {
                    other != slot
                        && parked.is_some()
                        && self.pool[other]
                            .as_ref()
                            .is_some_and(|state| self.arm.holds_request(state))
                })
                .min_by_key(|&(_, parked)| *parked)
                .map(|(other, _)| other);
            let Some(victim) = victim else {
                return false;
            };
            let Some(mut state) = self.pool[victim].take() else {
                return false;
            };
            let released = self.arm.release(&mut state);
            record(self.arm, &state, &self.stats[victim], true);
            self.pool[victim] = Some(state);
            if released.is_err() {
                return false;
            }
        }
    }

    /// The decodes that have room for their step; the rest are answered with the refusal their
    /// step would have been.
    fn steps(&mut self, decodes: Vec<Waiting>) -> Vec<Waiting> {
        let mut kept = Vec::with_capacity(decodes.len());
        for (slot, token, reply) in decodes {
            if self.make(slot, None) {
                kept.push((slot, token, reply));
            } else {
                let _ = reply.send(Err(BackendError::new(
                    "context",
                    "no memory is left for this sequence's next step, and no idle sequence holds \
                     any to give back",
                )));
            }
        }
        kept
    }
}

/// One prompt job's request: its tokens, its images and its sampler.
struct Prompt<'a> {
    tokens: &'a [u32],
    images: &'a [VisionImage],
    sampling: Sampling,
}

/// Runs one prompt on its slot, between the slot's decodes.
fn prefill<A: FusedArm>(
    arm: &A,
    pool: &mut [Option<A::Slot>],
    stats: &[Arc<Mutex<Record>>],
    slot: usize,
    prompt: Prompt<'_>,
    reply: Reply,
) {
    let Some(mut state) = pool[slot].take() else {
        let _ = reply.send(Err(BackendError::new(
            "fused",
            "the slot is already running a job",
        )));
        return;
    };
    let answer = (|| {
        // A slot whose arm keeps its history goes to the prompt as it stands: the arm continues
        // what the prompt shares with it, or starts over.
        if arm.holds_request(&state) && arm.history(&state).is_none() {
            arm.renew(&mut state)?;
        }
        arm.set_sampling(&mut state, prompt.sampling);
        arm.prefill_images(&mut state, prompt.tokens, prompt.images)
    })();
    record(arm, &state, &stats[slot], true);
    pool[slot] = Some(state);
    let _ = reply.send(answer.map(|step| vec![step]));
}

/// Rolls back the answers a handle's run left unused ([FusedArm::give_back]). A slot that cannot
/// is renewed: its next prompt starts over rather than continuing a history its caller never saw.
fn give_back<A: FusedArm>(
    arm: &A,
    pool: &mut [Option<A::Slot>],
    stats: &[Arc<Mutex<Record>>],
    slot: usize,
    count: usize,
) {
    let Some(mut state) = pool[slot].take() else {
        return;
    };
    if arm.give_back(&mut state, count).is_err() {
        let _ = arm.renew(&mut state);
    }
    record(arm, &state, &stats[slot], true);
    pool[slot] = Some(state);
}

/// Steps the waiting members in one pass, one row each, and answers every job.
fn step<A: FusedArm>(
    arm: &A,
    pool: &mut [Option<A::Slot>],
    stats: &[Arc<Mutex<Record>>],
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
    let answer = arm.step(&mut members).map(|steps| {
        // Each member's run: its step, then every step its slot already holds, fed the token the
        // step before chose. A step that fails here ends that member's run where it stands.
        members
            .iter_mut()
            .zip(steps)
            .map(|(member, step)| {
                let mut run = vec![step];
                while let Some(token) = run.last().and_then(|step| step.next_token)
                    && arm.ready(&member.0, token)
                {
                    member.1 = token;
                    match arm
                        .step(std::slice::from_mut(member))
                        .ok()
                        .and_then(|mut steps| steps.pop())
                    {
                        Some(step) => run.push(step),
                        None => break,
                    }
                }
                run
            })
            .collect::<Vec<_>>()
    });
    for (index, (state, _)) in members.into_iter().enumerate() {
        let slot = slots[index];
        record(arm, &state, &stats[slot], false);
        pool[slot] = Some(state);
    }
    match answer {
        Ok(runs) => {
            for (index, reply) in replies.into_iter().enumerate() {
                let run = runs.get(index).cloned().unwrap_or_default();
                let _ = reply.send(Ok(run));
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

    /// An arm whose slot is its history: a prompt that extends it forwards only the rest.
    struct Keeper;

    /// A [Keeper] slot: the ids it holds and how many of the last prompt's it reused.
    #[derive(Default)]
    struct Held {
        ids: Vec<u32>,
        cached: usize,
    }

    impl FusedArm for Keeper {
        type Slot = Held;

        fn open_slot(&self) -> Result<Self::Slot, BackendError> {
            Ok(Held::default())
        }

        fn holds_request(&self, slot: &Self::Slot) -> bool {
            !slot.ids.is_empty()
        }

        fn renew(&self, slot: &mut Self::Slot) -> Result<(), BackendError> {
            *slot = Held::default();
            Ok(())
        }

        fn set_sampling(&self, _slot: &mut Self::Slot, _sampling: Sampling) {}

        fn prefill(
            &self,
            slot: &mut Self::Slot,
            tokens: &[u32],
        ) -> Result<StepOutput, BackendError> {
            slot.cached = if tokens.starts_with(&slot.ids) {
                slot.ids.len()
            } else {
                0
            };
            slot.ids = tokens.to_vec();
            Ok(StepOutput::default())
        }

        fn step(&self, members: &mut [(Self::Slot, u32)]) -> Result<Vec<StepOutput>, BackendError> {
            Ok(members
                .iter_mut()
                .map(|(slot, token)| {
                    slot.ids.push(*token);
                    StepOutput::default()
                })
                .collect())
        }

        fn speculation(&self, _slot: &Self::Slot) -> Speculation {
            Speculation::default()
        }

        fn history<'s>(&self, slot: &'s Self::Slot) -> Option<&'s [u32]> {
            Some(&slot.ids)
        }

        fn rewind_prefix<'s>(&self, slot: &'s Self::Slot) -> Option<&'s [u32]> {
            slot.ids.get(..1)
        }

        fn cached_tokens(&self, slot: &Self::Slot) -> usize {
            slot.cached
        }
    }

    #[test]
    fn a_slot_that_keeps_its_history_reports_it_and_continues_it() {
        let engine = FusedEngine::start(Arc::new(Keeper), 1).expect("the engine starts");
        let mut first = engine.open().expect("a slot");
        assert_eq!(
            first.reusable_prefix(),
            None,
            "a fresh handle holds nothing"
        );
        first.prefill(&[1, 2, 3]).expect("the prefill");
        first.decode(4).expect("the decode");
        assert_eq!(first.reusable_prefix(), Some(&[1, 2, 3, 4][..]));
        assert_eq!(first.rewind_prefix(), Some(&[1][..]));
        // The host keeps the handle and hands it the next turn: the slot is not renewed, so the
        // arm sees the prompt extend what it holds.
        first.prefill(&[1, 2, 3, 4, 5]).expect("the next turn");
        assert_eq!(first.cached_tokens(), 4);
        assert_eq!(first.reusable_prefix(), Some(&[1, 2, 3, 4, 5][..]));
        // A prompt that parts from the history starts over inside the arm.
        first.prefill(&[9]).expect("another conversation");
        assert_eq!(first.cached_tokens(), 0);
        assert_eq!(first.reusable_prefix(), Some(&[9][..]));
    }

    /// A [Keeper] whose slots share a budget of `limit` ids, counted by what each slot holds.
    struct Budget {
        limit: usize,
        held: Mutex<Vec<usize>>,
        released: AtomicUsize,
    }

    /// A [Budget] slot: its number, and the ids it holds.
    struct Counted {
        index: usize,
        ids: Vec<u32>,
    }

    impl Budget {
        fn used_besides(&self, index: usize) -> usize {
            let held = self.held.lock().unwrap();
            held.iter()
                .enumerate()
                .filter(|&(other, _)| other != index)
                .map(|(_, count)| count)
                .sum()
        }

        fn hold(&self, slot: &Counted) {
            self.held.lock().unwrap()[slot.index] = slot.ids.len();
        }
    }

    impl FusedArm for Budget {
        type Slot = Counted;

        fn open_slot(&self) -> Result<Self::Slot, BackendError> {
            let mut held = self.held.lock().unwrap();
            held.push(0);
            Ok(Counted {
                index: held.len() - 1,
                ids: Vec::new(),
            })
        }

        fn holds_request(&self, slot: &Self::Slot) -> bool {
            !slot.ids.is_empty()
        }

        fn renew(&self, slot: &mut Self::Slot) -> Result<(), BackendError> {
            slot.ids.clear();
            self.hold(slot);
            Ok(())
        }

        fn release(&self, slot: &mut Self::Slot) -> Result<(), BackendError> {
            self.released.fetch_add(1, Ordering::Relaxed);
            self.renew(slot)
        }

        fn set_sampling(&self, _slot: &mut Self::Slot, _sampling: Sampling) {}

        fn prefill(
            &self,
            slot: &mut Self::Slot,
            tokens: &[u32],
        ) -> Result<StepOutput, BackendError> {
            if !self.room(slot, Some(tokens)) {
                return Err(BackendError::new("context", "no room"));
            }
            slot.ids = tokens.to_vec();
            self.hold(slot);
            Ok(StepOutput::default())
        }

        fn step(&self, members: &mut [(Self::Slot, u32)]) -> Result<Vec<StepOutput>, BackendError> {
            Ok(members
                .iter_mut()
                .map(|(slot, token)| {
                    slot.ids.push(*token);
                    self.hold(slot);
                    StepOutput::default()
                })
                .collect())
        }

        fn speculation(&self, _slot: &Self::Slot) -> Speculation {
            Speculation::default()
        }

        fn history<'s>(&self, slot: &'s Self::Slot) -> Option<&'s [u32]> {
            Some(&slot.ids)
        }

        fn room(&self, slot: &Self::Slot, prompt: Option<&[u32]>) -> bool {
            let wants = prompt.map_or(slot.ids.len() + 1, <[u32]>::len);
            self.used_besides(slot.index) + wants <= self.limit
        }
    }

    #[test]
    fn a_slot_short_of_memory_takes_it_from_the_longest_parked_one() {
        let arm = Arc::new(Budget {
            limit: 8,
            held: Mutex::new(Vec::new()),
            released: AtomicUsize::new(0),
        });
        let engine = FusedEngine::start(Arc::clone(&arm), 3).expect("the engine starts");
        let mut first = engine.open().expect("a slot");
        let mut second = engine.open().expect("a slot");
        let mut third = engine.open().expect("a slot");
        first.prefill(&[1, 2, 3]).expect("the first conversation");
        second.prefill(&[4, 5, 6]).expect("the second conversation");
        first.park();
        second.park();
        // Five ids do not fit beside six held: the first conversation, parked longest, goes.
        third.prefill(&[7, 8, 9, 10, 11]).expect("room is made");
        assert_eq!(arm.released.load(Ordering::Relaxed), 1);
        assert_eq!(
            first.reusable_prefix(),
            Some(&[][..]),
            "the parked sequence is given up"
        );
        assert_eq!(second.reusable_prefix(), Some(&[4, 5, 6][..]));
        // The third conversation's step needs a sixth id: only the parked second conversation
        // can give it, and does.
        third.decode(12).expect("the step");
        assert_eq!(arm.released.load(Ordering::Relaxed), 2);
        assert_eq!(second.reusable_prefix(), Some(&[][..]));
        assert_eq!(third.reusable_prefix(), Some(&[7, 8, 9, 10, 11, 12][..]));
        // A busy slot is never given up: with the first conversation back and nothing parked
        // holding anything, a prompt that does not fit is refused.
        first.prefill(&[1, 2]).expect("two more ids fit");
        assert!(
            second.prefill(&[4]).is_err(),
            "no parked sequence holds anything"
        );
        assert_eq!(arm.released.load(Ordering::Relaxed), 2);
        assert_eq!(first.reusable_prefix(), Some(&[1, 2][..]));
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

    /// A drafting toy: a step that is not held runs a "round" that holds the next `HELD` answers,
    /// each the fed token plus one, so a run drains them in one engine step.
    struct Rounds {
        walks: AtomicUsize,
    }

    #[derive(Default)]
    struct Drafting {
        history: Vec<u32>,
        held: usize,
    }

    const HELD: usize = 3;

    impl FusedArm for Rounds {
        type Slot = Drafting;

        fn open_slot(&self) -> Result<Self::Slot, BackendError> {
            Ok(Drafting::default())
        }

        fn holds_request(&self, slot: &Self::Slot) -> bool {
            !slot.history.is_empty()
        }

        fn renew(&self, slot: &mut Self::Slot) -> Result<(), BackendError> {
            *slot = Drafting::default();
            Ok(())
        }

        fn set_sampling(&self, _slot: &mut Self::Slot, _sampling: Sampling) {}

        fn prefill(
            &self,
            slot: &mut Self::Slot,
            tokens: &[u32],
        ) -> Result<StepOutput, BackendError> {
            *slot = Drafting {
                history: tokens.to_vec(),
                held: 0,
            };
            Ok(StepOutput {
                next_token: tokens.last().map(|token| token + 1),
                ..StepOutput::default()
            })
        }

        fn step(&self, members: &mut [(Self::Slot, u32)]) -> Result<Vec<StepOutput>, BackendError> {
            Ok(members
                .iter_mut()
                .map(|(slot, token)| {
                    if slot.held > 0 {
                        slot.held -= 1;
                    } else {
                        self.walks.fetch_add(1, Ordering::SeqCst);
                        slot.held = HELD;
                    }
                    slot.history.push(*token);
                    StepOutput {
                        // The round ends the turn at 50.
                        next_token: Some(*token + 1).filter(|next| *next != 50),
                        ..StepOutput::default()
                    }
                })
                .collect())
        }

        fn ready(&self, slot: &Self::Slot, _token: u32) -> bool {
            slot.held > 0
        }

        fn give_back(&self, slot: &mut Self::Slot, count: usize) -> Result<(), BackendError> {
            let keep =
                slot.history.len().checked_sub(count).ok_or_else(|| {
                    BackendError::new("toy", "gave back more than the slot was fed")
                })?;
            slot.history.truncate(keep);
            slot.held += count;
            Ok(())
        }

        fn speculation(&self, _slot: &Self::Slot) -> Speculation {
            Speculation::default()
        }

        fn history<'s>(&self, slot: &'s Self::Slot) -> Option<&'s [u32]> {
            Some(&slot.history)
        }
    }

    fn rounds() -> (Arc<Rounds>, Arc<FusedEngine<Rounds>>) {
        let arm = Arc::new(Rounds {
            walks: AtomicUsize::new(0),
        });
        let engine = FusedEngine::start(Arc::clone(&arm), 2).expect("the engine starts");
        (arm, engine)
    }

    #[test]
    fn a_run_answers_its_held_rows_without_the_engine() {
        let (arm, engine) = rounds();
        let mut session = engine.open().expect("a slot");
        assert_eq!(session.prefill(&[1]).unwrap().next_token, Some(2));
        // One walk answers 2 and holds 3, 4 and 5: the handle answers those itself.
        for token in 2..=5 {
            assert_eq!(session.decode(token).unwrap().next_token, Some(token + 1));
        }
        assert_eq!(arm.walks.load(Ordering::SeqCst), 1);
        assert_eq!(session.reusable_prefix(), Some(&[1, 2, 3, 4, 5][..]));
        // The next token is past the round: a new walk.
        assert_eq!(session.decode(6).unwrap().next_token, Some(7));
        assert_eq!(arm.walks.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_run_cut_short_gives_its_unused_answers_back() {
        let (arm, engine) = rounds();
        let mut session = engine.open().expect("a slot");
        session.prefill(&[1]).unwrap();
        session.decode(2).unwrap();
        session.decode(3).unwrap();
        // The caller forces 9 where the run chose 4: the two unused answers go back first, and 9
        // is then a held step of the same round.
        assert_eq!(session.decode(9).unwrap().next_token, Some(10));
        assert_eq!(arm.walks.load(Ordering::SeqCst), 1);
        // A park with a run outstanding gives it back and waits, so the record is the caller's.
        session.park();
        assert_eq!(session.reusable_prefix(), Some(&[1, 2, 3, 9][..]));
    }

    #[test]
    fn a_run_stops_at_the_end_of_the_turn() {
        let (_arm, engine) = rounds();
        let mut session = engine.open().expect("a slot");
        session.prefill(&[47]).unwrap();
        // 48 then 49, whose step ends the turn: the run holds no answer past it.
        assert_eq!(session.decode(48).unwrap().next_token, Some(49));
        assert_eq!(session.decode(49).unwrap().next_token, None);
        session.park();
        assert_eq!(session.reusable_prefix(), Some(&[47, 48, 49][..]));
    }

    #[test]
    fn an_arm_that_holds_nothing_refuses_a_give_back() {
        let (arm, _engine) = toy();
        let mut slot = 0;
        assert!(arm.give_back(&mut slot, 0).is_ok());
        assert!(arm.give_back(&mut slot, 1).is_err());
    }
}

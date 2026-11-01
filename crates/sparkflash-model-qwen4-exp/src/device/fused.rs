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
//! each of those sessions steps the whole model for its own single row, and since a decode step is
//! bound by the bytes it reads rather than the flops it issues, N sessions take N times as long as
//! one and serve no more tokens a second than one does.
//!
//! This module puts the fusion behind `open_session` instead. A session opened here is a handle:
//! its `prefill` and `decode` post one job to a thread that owns every pooled sequence, and the
//! thread steps the jobs that are waiting as one batch (see `super::batch`). The host driver is
//! unchanged, its worker threads still block on one request each, and the tokens a session sees
//! are the same ones it would see alone.
//!
//! A member keeps its drafter. A step gives each member its own token plus the drafts it proposed,
//! so a member verifying two drafts contributes three rows and a two-member batch is a six-row
//! pass: the drafts are verified in parallel, across members as well as within one.
//!
//! One thing a fused session does not do: it runs a prefill alone. The prompt chunks of two
//! requests are not batched together yet, so a prefill takes the engine while it runs and the other
//! members' decodes wait for it.

use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use sparkflash_model::{BackendError, ModelSession, Sampling, Speculation, StepOutput};

use super::batch::{BatchBuffers, MAX_BATCH};
use super::{DeviceSequence, SequenceShared};

/// Whether a fused member may verify its speculative drafts, under
/// `SPARKFLASH_QWEN4_FUSE_SPEC`.
///
/// Exact either way (the gate runs it). Off by default because it does not measure better: one
/// member verifying reaches about 65 tok/s on its own, and two reach 43-63 aggregate. A verifying
/// step is 49 ms of weights read once plus about 5.7 ms a row (an eight-row step is 89.6 ms), so
/// two members at eight rows should cost 135 ms for the ten tokens they verify - about 1.3x a pair
/// of lone verifies - and that margin is inside this node's spread. The drafter is not the reason:
/// it is 6% of the step (d_layers 2.7 ms, d_lm_head 2.4, d_select 0.1 of 89.6).
fn fuse_spec() -> bool {
    std::env::var_os("SPARKFLASH_QWEN4_FUSE_SPEC").is_some()
}

/// Whether every step names its members, under `SPARKFLASH_QWEN4_BATCH_TRACE`.
///
/// A deployment diagnostic: it is how a run proves its requests were actually fused rather than
/// answered one at a time, which is otherwise indistinguishable from the outside.
fn trace() -> bool {
    std::env::var_os("SPARKFLASH_QWEN4_BATCH_TRACE").is_some()
}

/// How long a step waits for the batch to fill before it runs with what it has.
///
/// The members of a steady-state batch finish a step together and post their next token within a
/// few microseconds of each other, so this only matters for the first step of a round and for a
/// request that joins late. It is a ceiling on the latency a batch may add, not a schedule: a
/// window that is already full runs without waiting.
const FILL_WINDOW: Duration = Duration::from_micros(300);

/// Whether to name each step's members on stderr.
static TRACE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(trace);

/// One request a handle makes of the engine thread.
struct Job {
    slot: usize,
    reply: Sender<Reply>,
    work: Work,
}

enum Work {
    Prefill(Vec<u32>),
    Decode(u32),
    Sampling(Sampling),
    /// The engine's owner is dropping it; the thread frees its pool and returns.
    Shutdown,
}

enum Reply {
    Step(Result<StepOutput, BackendError>),
    Applied,
    Gone,
}

/// A prefill, with the sequence it runs on: a prompt is one sequence's chunks, and while the
/// worker holds it the engine cannot step that member - which is the point, because the other
/// members' decodes keep running.
struct PrefillJob {
    slot: usize,
    session: DeviceSequence,
    tokens: Vec<u32>,
    reply: Sender<Reply>,
}

/// A prefill's sequence coming back to the pool.
struct Finished {
    slot: usize,
    session: DeviceSequence,
}

/// The engine thread and the free slots of its session pool.
pub(super) struct FusedEngine {
    jobs: Sender<Job>,
    free: Mutex<Vec<usize>>,
    wake: Condvar,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// The prefill workers, which hold a sequence only while a prompt runs on it. The senders go
    /// with them: dropping the engine closes the channels and ends their loops.
    prefill_threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
    /// Each slot's speculative counters, which the engine thread refreshes after every step. A
    /// handle cannot read its sequence - the engine owns it - so this is how a request reports
    /// what its drafts were worth.
    stats: Vec<Arc<Mutex<Speculation>>>,
}

impl FusedEngine {
    /// Allocate `capacity` sessions and start the thread that steps them together.
    ///
    /// # Errors
    /// A session the device refuses to allocate, or a thread that cannot be started.
    pub(super) fn start(
        shared: Arc<SequenceShared>,
        context_limit: usize,
        capacity: usize,
    ) -> Result<Arc<Self>, BackendError> {
        if capacity == 0 {
            return Err(BackendError::new(
                "qwen4_batch",
                "a fused engine needs at least one session",
            ));
        }
        let cuda = &shared.cuda;
        cuda.make_current()
            .map_err(|error| BackendError::new("qwen4_batch", error))?;
        let mut pool: Vec<Option<DeviceSequence>> = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            let mut session = DeviceSequence::allocate(Arc::clone(&shared), context_limit)?;
            // A member may verify its drafts inside the batch - its rows join the same pass - but
            // the MTP drafter accepts too few of them to pay for the rows it adds, so that is opt
            // in until the block drafter, whose drafts are worth verifying, is the one loaded.
            if !fuse_spec() {
                session.spec_enabled = false;
            }
            pool.push(Some(session));
        }
        // Every member may verify as many drafts as a lone session does, so the combined rows have
        // to hold one member's worst case for each of them.
        let max_rows = capacity * shared.verify_rows;
        // Whether the members verify their drafts is the difference between a batch that reads the
        // weights once and one that also spends rows on rejected drafts; say which, once.
        eprintln!(
            "qwen4: fused batch of {capacity} sessions, {max_rows} rows, drafts {}",
            if fuse_spec() {
                "verified"
            } else {
                "not verified"
            }
        );
        let (jobs, inbox) = channel::<Job>();
        // Prefills get a thread of their own, and their sequence travels with the job: a prompt
        // takes a sequence out of the pool until its chunks are done, while the other members keep
        // decoding. Nothing else moves sequences between threads.
        // One worker a member, so two prompts prefill beside each other instead of queueing behind
        // one another: a prompt of a few thousand tokens is seconds, and the second request's
        // first token should not wait for it. The GPU is shared either way, which is why the pool
        // is the session count and not more.
        let (prefilled, funded) = channel::<Finished>();
        let mut senders = Vec::with_capacity(capacity);
        let mut prefill_threads = Vec::with_capacity(capacity);
        for worker in 0..capacity {
            let (prefills, awaiting) = channel::<PrefillJob>();
            let prefilled = prefilled.clone();
            prefill_threads.push(
                std::thread::Builder::new()
                    .name(format!("sparkflash-qwen4-prefill-{worker}"))
                    .spawn(move || {
                        while let Ok(mut job) = awaiting.recv() {
                            let answer = job.session.prefill(&job.tokens);
                            // The sequence goes back before the answer does: the caller's next job
                            // is a decode on it, and the reply is what lets the caller post that
                            // job at all.
                            if prefilled
                                .send(Finished {
                                    slot: job.slot,
                                    session: job.session,
                                })
                                .is_err()
                            {
                                return;
                            }
                            let _ = job.reply.send(Reply::Step(answer));
                        }
                    })
                    .map_err(|error| BackendError::new("qwen4_batch", error.to_string()))?,
            );
            senders.push(prefills);
        }
        let free = (0..capacity).collect();
        let stats: Vec<Arc<Mutex<Speculation>>> = (0..capacity)
            .map(|_| Arc::new(Mutex::new(Speculation::default())))
            .collect();
        let thread_stats = stats.clone();
        let engine_shared = Arc::clone(&shared);
        let prefill_senders = Mutex::new(senders);
        let thread = std::thread::Builder::new()
            .name("sparkflash-qwen4-batch".to_owned())
            .spawn(move || {
                run(
                    pool,
                    inbox,
                    engine_shared,
                    context_limit,
                    max_rows,
                    thread_stats,
                    prefill_senders,
                    funded,
                );
            })
            .map_err(|error| BackendError::new("qwen4_batch", error.to_string()))?;
        let prefill_threads = Mutex::new(prefill_threads);
        Ok(Arc::new(Self {
            jobs,
            free: Mutex::new(free),
            wake: Condvar::new(),
            thread: Mutex::new(Some(thread)),
            prefill_threads,
            stats,
        }))
    }

    /// Take a pooled session for one request.
    ///
    /// Blocks while every slot is in use, which is what keeps a request from opening a session it
    /// could never step: the engine's capacity is the driver's concurrency.
    pub(super) fn open(self: &Arc<Self>) -> Result<FusedSession, BackendError> {
        let mut free = self.free.lock().map_err(|_| {
            BackendError::new("qwen4_batch", "the fused engine's slots are poisoned")
        })?;
        while free.is_empty() {
            free = self.wake.wait(free).map_err(|_| {
                BackendError::new("qwen4_batch", "the fused engine's slots are poisoned")
            })?;
        }
        let slot = free.remove(0);
        Ok(FusedSession {
            slot,
            engine: Arc::clone(self),
            stats: Arc::clone(&self.stats[slot]),
            sampling: Sampling::default(),
        })
    }

    /// Give a session's slot back once its request is over.
    fn release(&self, slot: usize) {
        if let Ok(mut free) = self.free.lock() {
            free.push(slot);
            self.wake.notify_one();
        }
    }
}

impl Drop for FusedEngine {
    fn drop(&mut self) {
        // One shutdown job ends the thread; the reply channel it names is scratch, since the
        // thread's exit is what the join waits for.
        let (reply, _) = channel();
        let _ = self.jobs.send(Job {
            slot: 0,
            reply,
            work: Work::Shutdown,
        });
        if let Ok(mut thread) = self.thread.lock()
            && let Some(thread) = thread.take()
        {
            let _ = thread.join();
        }
        // The prefill workers' channels are dropped with the engine, which ends their loops; a
        // prompt still in flight takes its sequence with it, which at shutdown is the same as
        // freeing it.
        if let Ok(mut threads) = self.prefill_threads.lock() {
            for thread in threads.drain(..) {
                let _ = thread.join();
            }
        }
    }
}

/// A request's session, which posts to the engine thread instead of running the model itself.
pub(super) struct FusedSession {
    slot: usize,
    /// The engine's slots and thread, which outlive the handle that owns them.
    engine: Arc<FusedEngine>,
    /// This slot's speculative counters, refreshed by the engine after each step.
    stats: Arc<Mutex<Speculation>>,
    sampling: Sampling,
}

impl FusedSession {
    /// Post `work` and wait for the engine's answer.
    fn ask(&mut self, work: Work) -> Reply {
        let (reply, answers) = channel();
        if self
            .engine
            .jobs
            .send(Job {
                slot: self.slot,
                reply,
                work,
            })
            .is_err()
        {
            return Reply::Gone;
        }
        match answers.recv() {
            Ok(reply) => reply,
            Err(_) => Reply::Gone,
        }
    }
}

impl ModelSession for FusedSession {
    fn speculation(&self) -> Speculation {
        self.stats.lock().map(|stats| *stats).unwrap_or_default()
    }

    fn set_sampling(&mut self, sampling: Sampling) {
        self.sampling = sampling;
        // Ordering matters: the engine applies this before the next step of this slot, because
        // both travel on the same channel.
        let _ = self.ask(Work::Sampling(sampling));
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        match self.ask(Work::Prefill(tokens.to_vec())) {
            Reply::Step(result) => result,
            Reply::Applied => Err(BackendError::new(
                "qwen4_batch",
                "the fused engine answered a prefill with no step",
            )),
            Reply::Gone => Err(BackendError::new(
                "qwen4_batch",
                "the fused engine stopped before the prefill ran",
            )),
        }
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        match self.ask(Work::Decode(token)) {
            Reply::Step(result) => result,
            Reply::Applied => Err(BackendError::new(
                "qwen4_batch",
                "the fused engine answered a decode with no step",
            )),
            Reply::Gone => Err(BackendError::new(
                "qwen4_batch",
                "the fused engine stopped before the decode ran",
            )),
        }
    }
}

impl Drop for FusedSession {
    fn drop(&mut self) {
        // The slot goes back when the request is over. The engine owns the sequence, so nothing
        // else has to happen here.
        self.engine.release(self.slot);
    }
}

/// Run the engine: step the jobs that are waiting, and answer each one.
#[allow(clippy::too_many_arguments)]
fn run(
    mut pool: Vec<Option<DeviceSequence>>,
    inbox: Receiver<Job>,
    shared: Arc<SequenceShared>,
    context_limit: usize,
    max_rows: usize,
    stats: Vec<Arc<Mutex<Speculation>>>,
    prefills: Mutex<Vec<Sender<PrefillJob>>>,
    finished: Receiver<Finished>,
) {
    let capacity = pool.len();
    let cuda = &shared.cuda;
    if cuda.make_current().is_err() {
        return;
    }
    let mut buffers: Option<BatchBuffers> = None;
    let mut next_prefill = 0usize;
    let mut pending: Vec<Job> = Vec::new();
    loop {
        // A prompt that finished gives its sequence back before anything else looks at the pool.
        while let Ok(done) = finished.try_recv() {
            pool[done.slot] = Some(done.session);
        }
        // Wait for work when there is none, then give a batch a moment to assemble.
        if pending.is_empty() {
            match inbox.recv() {
                Ok(job) => pending.push(job),
                Err(_) => break,
            }
            let deadline = Instant::now() + FILL_WINDOW;
            while pending.len() < capacity {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    break;
                }
                match inbox.recv_timeout(left) {
                    Ok(job) => pending.push(job),
                    Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
                }
            }
        }
        // A prompt that finished while this thread was waiting for work has to be collected now:
        // the job that woke it is that prompt's own next decode.
        while let Ok(done) = finished.try_recv() {
            pool[done.slot] = Some(done.session);
        }
        // Sampling changes are free and must land before the slot's next step, so they are applied
        // first and answered at once.
        let mut still: Vec<Job> = Vec::with_capacity(pending.len());
        for job in pending.drain(..) {
            match job.work {
                Work::Sampling(sampling) => {
                    if let Some(session) = pool[job.slot].as_mut() {
                        session.set_sampling(sampling);
                        let _ = job.reply.send(Reply::Applied);
                    } else {
                        let _ = job.reply.send(Reply::Gone);
                    }
                }
                Work::Shutdown => return,
                work => still.push(Job { work, ..job }),
            }
        }
        pending = still;
        if pending.is_empty() {
            continue;
        }
        // A prefill runs alone: its prompt is one sequence's chunk, not a row of a batch.
        if let Some(at) = pending
            .iter()
            .position(|job| matches!(job.work, Work::Prefill(_)))
        {
            let job = pending.remove(at);
            let Work::Prefill(tokens) = job.work else {
                unreachable!("the job was chosen for its prefill");
            };
            // The sequence leaves the pool for as long as the prompt runs; the other members keep
            // decoding, which is what this thread is here for.
            let Some(session) = pool[job.slot].take() else {
                let _ = job.reply.send(Reply::Step(Err(BackendError::new(
                    "qwen4_batch",
                    "the session's sequence is busy with a prompt",
                ))));
                continue;
            };
            let job = PrefillJob {
                slot: job.slot,
                session,
                tokens,
                reply: job.reply,
            };
            // Round robin: a worker that is already prefilling takes the next prompt and runs it
            // when its own finishes, which keeps two prompts of similar length in step.
            let sent = match prefills.lock() {
                Ok(senders) if !senders.is_empty() => {
                    let index = next_prefill % senders.len();
                    next_prefill = next_prefill.wrapping_add(1);
                    senders[index].send(job).map_err(|error| error.0)
                }
                _ => Err(job),
            };
            let job = match sent {
                Ok(()) => continue,
                // No worker is left to run it: give the sequence back and refuse the request.
                Err(job) => job,
            };
            pool[job.slot] = Some(job.session);
            let _ = job.reply.send(Reply::Step(Err(BackendError::new(
                "qwen4_batch",
                "the prefill workers are gone",
            ))));
            continue;
        }
        // A verify computed the answers for the positions after the token it accepted, and the
        // caller feeds those tokens next. A member holding such an answer owes no forward at all:
        // a batch that stepped it anyway would pay rows for a token already decided, which is most
        // of the drafter's gain. A caller that fed a different token has to roll back, and only the
        // session's own decode does that, so it runs alone.
        let mut rest: Vec<Job> = Vec::with_capacity(pending.len());
        for job in pending.drain(..) {
            let slot = job.slot;
            let Some(session) = pool[slot].as_mut() else {
                let _ = job.reply.send(Reply::Step(Err(BackendError::new(
                    "qwen4_batch",
                    "the session's sequence is busy with a prompt",
                ))));
                continue;
            };
            match (
                job.work,
                session.spec_queue.first().map(|front| front.token),
            ) {
                (Work::Decode(token), Some(front)) if front == token => {
                    session.sampler.record(token);
                    let answer = session.spec_queue.remove(0);
                    let _ = job.reply.send(Reply::Step(Ok(StepOutput {
                        next_token: answer.next_token,
                        logits: answer.logits,
                        draft: Vec::new(),
                    })));
                }
                (Work::Decode(token), Some(_)) => {
                    let answer = session.decode(token);
                    let _ = job.reply.send(Reply::Step(answer));
                }
                (work, _) => rest.push(Job { work, ..job }),
            }
        }
        pending = rest;
        if pending.is_empty() {
            continue;
        }
        // Decodes: one member a token, in slot order, up to the batch's width and its row budget.
        // A member's rows are its token plus the drafts it proposed last step, so a batch of two
        // members verifying two drafts each is a six-row pass over the same weights.
        let mut chosen: Vec<Job> = Vec::with_capacity(MAX_BATCH.min(pending.len()));
        let mut rest: Vec<Job> = Vec::with_capacity(pending.len());
        let mut rows_taken = 0usize;
        for job in pending.drain(..) {
            let Some(session) = pool[job.slot].as_ref() else {
                let _ = job.reply.send(Reply::Step(Err(BackendError::new(
                    "qwen4_batch",
                    "the session's sequence is busy with a prompt",
                ))));
                continue;
            };
            let width = 1 + session.spec_drafts.len();
            let fits = chosen.len() < MAX_BATCH
                && !chosen.iter().any(|other| other.slot == job.slot)
                && rows_taken + width <= max_rows;
            if fits {
                rows_taken += width;
                chosen.push(job);
            } else {
                rest.push(job);
            }
        }
        pending = rest;
        if chosen.is_empty() {
            continue;
        }
        chosen.sort_by_key(|job| job.slot);
        let mut tokens = Vec::with_capacity(chosen.len());
        let mut drafts: Vec<Vec<u32>> = Vec::with_capacity(chosen.len());
        let mut confidence: Vec<Vec<f32>> = Vec::with_capacity(chosen.len());
        for job in &chosen {
            let Work::Decode(token) = job.work else {
                unreachable!("a prefill was taken first");
            };
            // The sampler's history takes the token before the row it leads to, exactly as a
            // plain decode does, so a penalty sees the same past.
            let session = pool[job.slot]
                .as_mut()
                .expect("a chosen slot holds its sequence");
            session.sampler.record(token);
            tokens.push(token);
            drafts.push(std::mem::take(&mut session.spec_drafts));
            confidence.push(std::mem::take(&mut session.spec_confidence));
        }
        let overruns = chosen.iter().zip(&drafts).any(|(job, drafts)| {
            pool[job.slot]
                .as_ref()
                .is_none_or(|session| session.position + drafts.len() + 1 > context_limit)
        });
        if overruns {
            for job in chosen {
                let _ = job.reply.send(Reply::Step(Err(BackendError::new(
                    "qwen4_batch",
                    "the session's context window is full",
                ))));
            }
            continue;
        }
        let score_stride = pool
            .iter()
            .flatten()
            .next()
            .map_or(0, |session| session.score_stride);
        if buffers.is_none() {
            match BatchBuffers::new(Arc::clone(&shared), score_stride, max_rows) {
                Ok(allocation) => buffers = Some(allocation),
                Err(error) => {
                    for job in chosen {
                        let _ = job.reply.send(Reply::Step(Err(error.clone())));
                    }
                    continue;
                }
            }
        }
        if *TRACE {
            let slots: Vec<usize> = chosen.iter().map(|job| job.slot).collect();
            let rows: Vec<usize> = drafts.iter().map(|drafts| drafts.len() + 1).collect();
            eprintln!(
                "qwen4: fused step of {} member(s) at slots {slots:?}, rows {rows:?}",
                chosen.len()
            );
        }
        let allocation = buffers.as_mut().expect("the buffers were just made");
        let steps = {
            let mut members: Vec<&mut DeviceSequence> = Vec::with_capacity(chosen.len());
            let mut slots = pool.iter_mut();
            let mut next = 0usize;
            for job in &chosen {
                let entry = slots
                    .nth(job.slot - next)
                    .expect("a chosen slot is in the pool");
                members.push(entry.as_mut().expect("a chosen slot holds its sequence"));
                next = job.slot + 1;
            }
            allocation.step(&mut members, &drafts, &confidence, &tokens)
        };
        for job in &chosen {
            if let Ok(mut cell) = stats[job.slot].lock()
                && let Some(session) = pool[job.slot].as_ref()
            {
                *cell = session.speculation;
            }
        }
        match steps {
            Ok(steps) => {
                for (step, job) in steps.into_iter().zip(chosen) {
                    let _ = job.reply.send(Reply::Step(Ok(step)));
                }
            }
            Err(error) => {
                for job in chosen {
                    let _ = job.reply.send(Reply::Step(Err(error.clone())));
                }
            }
        }
    }
}

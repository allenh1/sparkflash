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

//! The generic host driver.
//!
//! The driver owns one backend and one sequence per worker. A worker keeps its sequence between
//! requests when the backend says the sequence can continue from the prefix the next prompt shares
//! with it ([ModelSession::reusable_prefix]), so a chat turn prefills only the text it added
//! instead of the whole conversation. A prompt that does not extend the held tokens gets a fresh
//! session; the two are equivalent, and only the work differs.
//!
//! Two cases continue a held sequence although the ids differ. A client that sends the model's
//! reply back as text has it re-encoded, and the tokenizer may spell a word with other ids than the
//! model chose (`ĠRibb`+`on` against `Ġ`+`Ribbon`): when the prompt spells the held tokens' exact
//! bytes, the sequence continues from its own ids and forwards the rest
//! ([HostOptions::reuse_by_text]). And a request whose client left after its prefill finished keeps
//! the sequence and the step it computed, so a retry of the same prompt answers from that step
//! instead of prefilling it again. Requests arrive through a bounded
//! channel and their events leave through a bounded reply channel, so a client that stops reading
//! applies backpressure to the engine rather than growing an unbounded buffer.
//!
//! `HostOptions::concurrency` chooses how many requests execute at once. Each worker owns one
//! session at a time, so concurrency N means N independent sequences progressing in parallel; it
//! is not a fused batch and does not change the arithmetic of any one sequence. The default is 1,
//! which preserves the serial one-job-at-a-time behavior. A value above 1 is for backends whose
//! sessions are independent; a backend whose sessions share mutable engine state must serialize
//! that state itself (the Qwen device arm gives each session its own stream, so their kernels can
//! overlap). A backend that only supports one live session must stay at 1. With more than one
//! worker a request is served by whichever worker takes it, so a prefix is reused only when the
//! same worker sees the next request.
//!
//! This adapter is intentionally smaller than the V4.1 reference driver. It supports text
//! requests, stop strings, sample logprobs and the standard usage/timing envelope. Prompt
//! logprobs need per-position logits, which the backend contract does not expose yet, so a request
//! that asks for them receives a typed engine error rather than an invented payload.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, bounded};
use futures::StreamExt;
use futures::stream::{self, BoxStream};
use sparkflash_api::Metrics;
use sparkflash_api::engine::{
    Capabilities, ClusterStatsSnapshot, EngineEvent, EngineRequest, FinishReason, InferenceEngine,
    LogprobAlternative, LogprobSelection, LogprobsRequest, MissingCapability, TokenLogprob,
    default_engram_layers, empty_kv_snapshot, max_seq_len_for,
};
use sparkflash_api::telemetry_types::{
    CompletionTokensDetails, DSparkStepStats, DeepSeekSparkRequestStats, OpenAiUsageWithCache,
    PromptTokensDetails, RequestTimingStats, RoceTransportStats, finite_or_zero,
};
use sparkflash_model::telemetry::PrefixMatchPlan;
use sparkflash_model::vision::{ImageSpan, VisionImage};
use sparkflash_model::{
    ModelBackend, ModelDescriptor, ModelSession, PromptCodec, Sampling, Speculation, StepOutput,
};
use tokio::sync::mpsc;

/// The default engine queue depth.
pub const DEFAULT_QUEUE: usize = 64;

/// The default number of replies that may be buffered ahead of a client.
pub const DEFAULT_REPLY_BUFFER: usize = 32;

/// The default generation cap when a caller does not set one.
pub const DEFAULT_MAX_TOKENS: usize = 4096;

/// The error code for a request the backend failed.
pub const ENGINE_ERROR_CODE: &str = "engine_error";

/// The error code for a stream whose worker stopped before answering.
pub const ENGINE_GONE_CODE: &str = "engine_gone";

/// The error code for a request refused before any backend work.
pub const REFUSED_CODE: &str = "refused";

/// The environment switch for [HostOptions::reuse_by_text]: `0` (or `off`) turns it off.
pub const REUSE_BY_TEXT_ENV: &str = "SPARKFLASH_REUSE_BY_TEXT";

/// Reads [REUSE_BY_TEXT_ENV]: unset is on, `1`/`on`/`true`/`yes` is on, `0`/`off`/`false`/`no` is
/// off. Any other value is refused rather than read as either.
fn reuse_by_text_from_env() -> Result<bool, String> {
    match std::env::var(REUSE_BY_TEXT_ENV) {
        Err(std::env::VarError::NotPresent) => Ok(true),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(format!("{REUSE_BY_TEXT_ENV} is not valid UTF-8"))
        }
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "on" | "true" | "yes" => Ok(true),
            "0" | "off" | "false" | "no" => Ok(false),
            _ => Err(format!(
                "{REUSE_BY_TEXT_ENV}={value:?} is neither on (1) nor off (0)"
            )),
        },
    }
}

/// How the generic host driver is sized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostOptions {
    /// The ceiling on generated tokens for one request.
    pub max_tokens: usize,
    /// How many requests may wait for the backend.
    pub queue_capacity: usize,
    /// How many reply events may be buffered for one client.
    pub reply_buffer: usize,
    /// How many requests execute at once. Each worker owns one session at a time.
    ///
    /// Defaults to 1, the serial path. A value above 1 is a concurrency choice, not a fused
    /// batch: it opens that many independent sessions and lets the operating system interleave
    /// their steps. A backend whose sessions share mutable state must keep this at 1.
    pub concurrency: usize,
    /// Whether a held sequence continues a prompt that spells its tokens' exact bytes with other
    /// ids, from its own ids. The model then reads the ids it chose for that span, not the ones a
    /// cold prefill of the prompt would read. On by default; [REUSE_BY_TEXT_ENV] turns it off.
    pub reuse_by_text: bool,
}

impl Default for HostOptions {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_MAX_TOKENS,
            queue_capacity: DEFAULT_QUEUE,
            reply_buffer: DEFAULT_REPLY_BUFFER,
            concurrency: 1,
            reuse_by_text: true,
        }
    }
}

impl HostOptions {
    /// These options with the switches the environment sets ([REUSE_BY_TEXT_ENV]).
    ///
    /// # Errors
    /// A switch whose value is neither on nor off.
    pub fn with_env(self) -> Result<Self, String> {
        Ok(Self {
            reuse_by_text: reuse_by_text_from_env()?,
            ..self
        })
    }
}

/// The API engine implemented over a ModelBackend.
///
/// The backend is shared by every worker thread behind an `Arc`; this value holds that Arc's
/// sibling state (descriptor, queue and metrics), never a session. The backend's own contract
/// decides whether its sessions may execute concurrently.
pub struct HostDriver {
    model: String,
    descriptor: Arc<ModelDescriptor>,
    capabilities: Capabilities,
    jobs: Sender<Job>,
    /// Set once every worker has exited; a later request is refused instead of queued forever.
    closed: Arc<AtomicBool>,
    max_tokens: usize,
    reply_buffer: usize,
    metrics: Arc<Metrics>,
    cluster: ClusterStatsSnapshot,
}

impl HostDriver {
    /// Starts the worker thread and returns the HTTP-facing driver.
    ///
    /// # Errors
    /// A message when the worker thread cannot be started.
    pub fn start(
        backend: Box<dyn ModelBackend>,
        codec: Arc<dyn PromptCodec>,
        options: HostOptions,
    ) -> Result<Self, String> {
        if options.max_tokens == 0 {
            return Err("the host engine needs a non-zero generation cap".to_owned());
        }
        if options.concurrency == 0 {
            return Err("the host engine needs at least one worker".to_owned());
        }
        let descriptor = Arc::new(backend.descriptor().clone());
        let fabric = backend
            .fabric_mode()
            .unwrap_or("none (host backend)")
            .to_owned();
        // The model says what it can render; the driver adds what its own loop implements.
        // run_request feeds a forced token in place of the backend's choice, through the same
        // session.decode every request takes, so teacher forcing is this driver's to declare.
        let capabilities = Capabilities {
            forced_decoding: true,
            ..backend.capabilities()
        };
        let (jobs, queued) = bounded(options.queue_capacity.max(1));
        let (ready, reported) = std::sync::mpsc::channel();
        let metrics = Arc::new(Metrics::new());
        let backend: Arc<dyn ModelBackend> = Arc::from(backend);
        let mut handles = Vec::with_capacity(options.concurrency);
        for worker in 0..options.concurrency {
            let backend = Arc::clone(&backend);
            let codec = Arc::clone(&codec);
            let queued = queued.clone();
            let ready = ready.clone();
            let thread_metrics = Arc::clone(&metrics);
            let by_text = options.reuse_by_text;
            let handle = std::thread::Builder::new()
                .name(format!("sparkflash-host-engine-{worker}"))
                .spawn(move || {
                    host_thread(backend, codec, queued, ready, thread_metrics, by_text);
                })
                .map_err(|error| format!("cannot start host engine worker {worker}: {error}"))?;
            handles.push(handle);
        }
        drop(ready);
        let mut failures = Vec::new();
        for _ in 0..options.concurrency {
            match reported.recv() {
                Ok(Ok(())) => {}
                Ok(Err(message)) => failures.push(message),
                Err(_) => failures
                    .push("a host engine worker stopped before it reported ready".to_owned()),
            }
        }
        if let Some(message) = failures.into_iter().next() {
            drop(jobs);
            for handle in handles {
                let _ = handle.join();
            }
            return Err(message);
        }
        let closed = Arc::new(AtomicBool::new(false));
        let monitor_closed = Arc::clone(&closed);
        let monitor_queue = queued.clone();
        std::thread::Builder::new()
            .name("sparkflash-host-monitor".to_owned())
            .spawn(move || {
                for handle in handles {
                    let _ = handle.join();
                }
                monitor_closed.store(true, Ordering::SeqCst);
                // Every worker is gone. Refuse the jobs that never ran rather than leaving their
                // clients waiting for a reply the engine can no longer produce.
                while let Ok(job) = monitor_queue.try_recv() {
                    let _ = job.replies.blocking_send(Reply::Failed(
                        "the host engine worker stopped before this request finished".to_owned(),
                    ));
                }
            })
            .map_err(|error| format!("cannot start the host engine monitor: {error}"))?;
        drop(queued);
        let cluster = cluster_snapshot(&descriptor, &fabric);
        Ok(Self {
            model: descriptor.id.clone(),
            descriptor,
            capabilities,
            jobs,
            closed,
            max_tokens: options.max_tokens,
            reply_buffer: options.reply_buffer.max(1),
            metrics,
            cluster,
        })
    }

    /// The exporter this driver feeds.
    #[must_use]
    pub fn metrics(&self) -> Arc<Metrics> {
        Arc::clone(&self.metrics)
    }

    /// The descriptor the driver serves.
    #[must_use]
    pub fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

impl InferenceEngine for HostDriver {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn generate(&self, request: EngineRequest) -> BoxStream<'static, EngineEvent> {
        if self.closed.load(Ordering::SeqCst) {
            return refused(
                "the host engine stopped; no worker is left to run this request",
                ENGINE_GONE_CODE,
            );
        }
        if !request.images.is_empty() {
            let missing = MissingCapability::vision();
            return refused(&missing.to_string(), missing.code());
        }
        // The API prepares images into `vision` only for an engine that advertises vision, and the
        // session writes their rows over the placeholders ([ModelSession::prefill_images]). A
        // backend that does not advertise it would read the placeholders as ordinary text, so
        // refuse rather than answer from a prompt whose images were silently dropped.
        if !request.vision.is_empty() && !self.capabilities.vision {
            let missing = MissingCapability::vision();
            return refused(&missing.to_string(), missing.code());
        }
        if request.prompt_tokens.is_empty() {
            return refused(
                "the request carries no prompt tokens, and the host backend has nothing to prefill",
                REFUSED_CODE,
            );
        }
        let prompt_tokens = request.prompt_tokens.len();
        let max_tokens = request.max_tokens.min(self.max_tokens);
        // The serve cap would otherwise cut a forced sequence short and answer a prefix of it.
        if !request.forced_tokens.is_empty() && request.forced_tokens.len() != max_tokens {
            return refused(
                &format!(
                    "forced_tokens has {} tokens but this request generates {max_tokens} (the server's --max-tokens is {}); a forced sequence is generated whole or refused",
                    request.forced_tokens.len(),
                    self.max_tokens
                ),
                REFUSED_CODE,
            );
        }
        let enqueued_at = Instant::now();
        let (replies, answers) = mpsc::channel(self.reply_buffer);
        let job = Job {
            prompt_tokens: request.prompt_tokens,
            images: request.vision,
            forced_tokens: request.forced_tokens,
            max_tokens,
            sampling: Sampling {
                temperature: request.temperature,
                top_p: request.top_p,
                top_k: request.top_k,
                repetition_penalty: request.repetition_penalty,
                presence_penalty: request.presence_penalty,
                frequency_penalty: request.frequency_penalty,
                seed: request.seed.unwrap_or(0),
                prompt_len: prompt_tokens,
            },
            logprobs: request.logprobs,
            stop: request.stop,
            enqueued_at,
            replies,
        };
        self.metrics.request_enqueued();
        if self.jobs.try_send(job).is_err() {
            self.metrics.request_enqueue_refused();
            return refused(
                "the host engine queue is full; retry when the requests in flight have finished",
                REFUSED_CODE,
            );
        }
        // A worker may have died between the check above and the send. The monitor fails the job
        // if it saw it; the client still gets a definite error instead of waiting forever.
        if self.closed.load(Ordering::SeqCst) {
            self.metrics.request_enqueue_refused();
            return refused(
                "the host engine stopped while this request was being queued",
                ENGINE_GONE_CODE,
            );
        }
        // The worker sends `Started` once it has decided which session serves the request, so the
        // event can carry the prefix the engine already holds. It is still the first event.
        // The finish event's transport block names the fabric this driver's backend declared, so
        // a four-rank device server does not report "no cluster" on the response it served.
        let fabric = self.cluster.fabric_mode.clone();
        let answers = stream::unfold(
            AnswerState {
                answers,
                prompt_tokens,
                done: false,
                fabric,
            },
            move |mut state| async move {
                if state.done {
                    return None;
                }
                match state.answers.recv().await {
                    Some(Reply::Started { cached_tokens }) => Some((
                        EngineEvent::Started {
                            prompt_tokens: state.prompt_tokens,
                            cached_tokens,
                        },
                        state,
                    )),
                    Some(Reply::Delta(text)) => Some((EngineEvent::ContentDelta(text), state)),
                    Some(Reply::TokenLogprobs(logprob)) => {
                        Some((EngineEvent::TokenLogprobs(logprob), state))
                    }
                    Some(Reply::Done(summary)) => {
                        state.done = true;
                        let fabric = state.fabric.clone();
                        Some((
                            finished_event(state.prompt_tokens, *summary, &fabric),
                            state,
                        ))
                    }
                    Some(Reply::Failed(message)) => {
                        state.done = true;
                        Some((
                            EngineEvent::Error {
                                message,
                                code: ENGINE_ERROR_CODE.to_owned(),
                            },
                            state,
                        ))
                    }
                    None => {
                        state.done = true;
                        Some((
                            EngineEvent::Error {
                                message: "the host engine stopped before this request finished"
                                    .to_owned(),
                                code: ENGINE_GONE_CODE.to_owned(),
                            },
                            state,
                        ))
                    }
                }
            },
        );
        answers.boxed()
    }

    fn cluster_stats(&self) -> ClusterStatsSnapshot {
        self.cluster.clone()
    }

    fn capabilities(&self) -> Capabilities {
        self.capabilities
    }

    fn vocab_size(&self) -> Option<usize> {
        Some(self.descriptor.vocab_size)
    }
}

/// The state of one request's answer stream.
struct AnswerState {
    answers: mpsc::Receiver<Reply>,
    prompt_tokens: usize,
    done: bool,
    /// The fabric this driver's backend declared, carried to the finish event's transport block.
    fabric: String,
}

/// One queued generation request.
struct Job {
    prompt_tokens: Vec<u32>,
    /// The prepared images whose spans lie in `prompt_tokens`, in prompt order.
    images: Vec<VisionImage>,
    /// Teacher forcing: the tokens fed in place of the backend's choice; empty for a generation.
    forced_tokens: Vec<u32>,
    max_tokens: usize,
    sampling: Sampling,
    logprobs: LogprobsRequest,
    stop: Vec<String>,
    enqueued_at: Instant,
    replies: mpsc::Sender<Reply>,
}

/// One element of the worker's answer.
enum Reply {
    /// The worker admitted the request and knows how much of its prompt the engine holds.
    Started {
        cached_tokens: usize,
    },
    Delta(String),
    TokenLogprobs(Box<TokenLogprob>),
    Done(Box<HostSummary>),
    Failed(String),
}

/// What one request produced.
struct HostSummary {
    completion_tokens: usize,
    finish_reason: FinishReason,
    timing: RequestTiming,
    speculation: Speculation,
    /// Prompt tokens the session already held, from [ModelSession::cached_tokens].
    cached_tokens: usize,
}

/// The wall-clock numbers one request measured.
#[derive(Debug, Clone, Copy, Default)]
struct RequestTiming {
    ttft_ms: f64,
    prefill_tok_per_s: f64,
    decode_tok_per_s: f64,
}

impl RequestTiming {
    /// `prompt_tokens` is the tokens the prefill forwarded, not the whole prompt: the cached
    /// prefix neither took time nor belongs in the rate.
    fn measure(
        prefilled_tokens: usize,
        completion_tokens: usize,
        prefill: Duration,
        ttft: Option<Duration>,
        total: Duration,
    ) -> Self {
        let first = ttft.map_or(0.0, |instant| instant.as_secs_f64());
        Self {
            ttft_ms: first * 1000.0,
            prefill_tok_per_s: rate(prefilled_tokens, prefill.as_secs_f64()),
            decode_tok_per_s: rate(
                completion_tokens.saturating_sub(1),
                total.as_secs_f64() - first,
            ),
        }
    }
}

fn rate(count: usize, seconds: f64) -> f64 {
    finite_or_zero(count as f64 / seconds)
}

fn host_thread(
    backend: Arc<dyn ModelBackend>,
    codec: Arc<dyn PromptCodec>,
    jobs: Receiver<Job>,
    ready: std::sync::mpsc::Sender<Result<(), String>>,
    metrics: Arc<Metrics>,
    by_text: bool,
) {
    if ready.send(Ok(())).is_err() {
        return;
    }
    // Do not keep the readiness sender alive for the worker's lifetime: `start` waits for one
    // report per worker, and a clone held by a worker that later hangs would keep the channel
    // open after the reports it needs have already arrived.
    drop(ready);
    // The sequence this worker last served, kept for the next request when the backend says it
    // can continue from the prefix that request shares with it.
    let mut held: Option<Held> = None;
    while let Ok(job) = jobs.recv() {
        metrics.request_admitted();
        let replies = job.replies.clone();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            run_request(
                backend.as_ref(),
                codec.as_ref(),
                job,
                &metrics,
                &mut held,
                by_text,
            )
        }));
        let failure = match outcome {
            Ok(Ok(())) => None,
            Ok(Err(message)) => Some(message),
            Err(payload) => Some(format!(
                "the host backend panicked on this request: {}",
                panic_message(payload.as_ref())
            )),
        };
        if let Some(message) = failure {
            let _ = replies.blocking_send(Reply::Failed(message));
        }
        metrics.request_finished();
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_owned()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "a panic whose payload is not a string".to_owned()
    }
}

/// A sequence a worker keeps between requests.
struct Held {
    session: Box<dyn ModelSession>,
    /// The step after the last token the session holds, and the sampler that drew it.
    ///
    /// A request whose prompt is exactly the held history has nothing left to forward, so only
    /// this step can answer it: the retry of a request whose client left once its prefill was done.
    pending: Option<(Sampling, StepOutput)>,
}

/// How a request continues the held sequence.
struct Continuation {
    /// The ids the session is handed when they are not the request's own: its prompt with the
    /// held ids where it spells their bytes with others.
    respelled: Option<Vec<u32>>,
    /// How many of the ids the session is handed it already holds.
    held: usize,
    /// How many of the request's own prompt ids those stand for.
    covered: usize,
    /// The session has nothing left to forward and answers from its pending step.
    resume: bool,
    /// The session goes back to its rewind point ([ModelSession::rewind_prefix]) and forwards
    /// the prompt past it.
    rewound: bool,
}

impl Continuation {
    /// A fresh sequence: nothing held.
    const COLD: Self = Self {
        respelled: None,
        held: 0,
        covered: 0,
        resume: false,
        rewound: false,
    };
}

/// How `job` may continue `kept`, when it may.
///
/// The prompt must extend the held tokens, or, with `by_text`, spell their exact bytes and then
/// extend them. Extend means strictly unless the held step answers the request: a session with
/// nothing left to forward has no other logits for the last token. The images inside the held
/// tokens must be the ones the session holds there ([same_images_before]): a held history is ids,
/// and an image's placeholders are the same ids whatever the image. A re-spelled prompt carries
/// none and continues no history that holds one, because the re-spelling would move their spans.
///
/// A prompt that does neither may still go past the session's rewind point
/// ([ModelSession::rewind_prefix]) with the point's images before it: the session goes back there.
fn continuation(
    kept: &Held,
    job: &Job,
    codec: &dyn PromptCodec,
    by_text: bool,
) -> Option<Continuation> {
    let history = kept
        .session
        .reusable_prefix()
        .filter(|history| !history.is_empty())?;
    let held_images = kept.session.held_images();
    let prompt = job.prompt_tokens.as_slice();
    let respelling = || {
        (by_text && job.images.is_empty() && held_images.is_empty())
            .then(|| respelled_prompt(codec, history, prompt))
            .flatten()
    };
    let (respelled, covered) = if prompt.starts_with(history) {
        (None, history.len())
    } else if let Some((ids, covered)) = respelling() {
        (Some(ids), covered)
    } else {
        return rewound(kept, job);
    };
    if !same_images_before(held_images, &job.images, covered) {
        return None;
    }
    let handed = respelled.as_ref().map_or(prompt.len(), Vec::len);
    let resume = handed == history.len();
    if resume {
        let (sampling, _) = kept.pending.as_ref()?;
        if *sampling != sampling_for(job.sampling, handed) {
            return None;
        }
    }
    Some(Continuation {
        respelled,
        held: history.len(),
        covered,
        resume,
        rewound: false,
    })
}

/// How `job` continues `kept` from the session's rewind point, when its prompt keeps the point's
/// tokens, goes past them, and carries the point's images before their end and no other.
fn rewound(kept: &Held, job: &Job) -> Option<Continuation> {
    let point = kept
        .session
        .rewind_prefix()
        .filter(|point| !point.is_empty())?;
    let prompt = job.prompt_tokens.as_slice();
    if prompt.len() <= point.len()
        || !prompt.starts_with(point)
        || !same_images_before(kept.session.held_images(), &job.images, point.len())
    {
        return None;
    }
    Some(Continuation {
        respelled: None,
        held: point.len(),
        covered: point.len(),
        resume: false,
        rewound: true,
    })
}

/// Whether `images` carry exactly the `held` images among those that start before `end`: every
/// held span there comes back, content key and all, and no other image starts there.
fn same_images_before(held: &[ImageSpan], images: &[VisionImage], end: usize) -> bool {
    held.iter()
        .filter(|span| span.start < end)
        .all(|span| images.iter().any(|image| image.span == *span))
        && images
            .iter()
            .filter(|image| image.span.start < end)
            .all(|image| held.contains(&image.span))
}

/// `sampling` for a session handed `handed` prompt ids.
fn sampling_for(mut sampling: Sampling, handed: usize) -> Sampling {
    sampling.prompt_len = handed;
    sampling
}

/// `prompt` with the held ids in place of the ids that spell the same bytes, and how many of
/// `prompt`'s ids the held ones stand for.
///
/// The two agree id for id up to where they part. From there `prompt` must spell the rest of
/// `history` byte for byte in vocabulary spellings, special tokens included, ending on one of its
/// own id boundaries. The codec's decode of the result must then equal its decode of `prompt`, so
/// the conversation reads the same to it as well.
fn respelled_prompt(
    codec: &dyn PromptCodec,
    history: &[u32],
    prompt: &[u32],
) -> Option<(Vec<u32>, usize)> {
    let shared = history
        .iter()
        .zip(prompt)
        .take_while(|(held, given)| held == given)
        .count();
    if shared == history.len() || shared == prompt.len() {
        return None;
    }
    let mut held = Vec::new();
    for &id in &history[shared..] {
        held.extend_from_slice(codec.token_str(id)?.as_bytes());
    }
    let (mut spelled, mut covered) = (0usize, shared);
    while spelled < held.len() {
        let piece = codec.token_str(*prompt.get(covered)?)?;
        let piece = piece.as_bytes();
        if held.get(spelled..spelled + piece.len()) != Some(piece) {
            return None;
        }
        spelled += piece.len();
        covered += 1;
    }
    let mut ids = Vec::with_capacity(history.len() + prompt.len() - covered);
    ids.extend_from_slice(history);
    ids.extend_from_slice(&prompt[covered..]);
    (codec.decode(&ids)? == codec.decode(prompt)?).then_some((ids, covered))
}

/// The prompt ids a request's hit stands for, from the ids the session reports it reused.
fn client_cached(session_cached: usize, plan: &Continuation) -> usize {
    if session_cached >= plan.held {
        session_cached - plan.held + plan.covered
    } else {
        session_cached.min(plan.covered)
    }
}

/// Why a request stopped before it finished.
enum Ending {
    /// The client stopped reading. The sequence is intact and may serve a retry.
    ClientGone,
    /// The request could not be answered, and the sequence is intact.
    Refused(String),
    /// The backend failed, and the sequence's state is unknown.
    Failed(String),
}

impl Ending {
    fn message(self) -> String {
        match self {
            Self::ClientGone => "the client stopped reading this response".to_owned(),
            Self::Refused(message) | Self::Failed(message) => message,
        }
    }
}

fn run_request(
    backend: &dyn ModelBackend,
    codec: &dyn PromptCodec,
    job: Job,
    metrics: &Metrics,
    held: &mut Option<Held>,
    by_text: bool,
) -> Result<(), String> {
    // A request whose client is already gone, such as a retry queued behind the request it
    // replaced, must not cost the held sequence the next request may continue.
    if job.replies.is_closed() {
        return Err(Ending::ClientGone.message());
    }
    if job.logprobs.wants_prompt() {
        return Err(
            "the generic host backend does not expose per-position prompt logits yet".to_owned(),
        );
    }
    let started = Instant::now();
    let kept = held.take();
    let plan = kept
        .as_ref()
        .and_then(|kept| continuation(kept, &job, codec, by_text));
    // A held sequence this request cannot continue is dropped before the next one opens, never
    // after: a session owns its whole cache, and at a long context two of them do not fit (an
    // Inkling rank at the 1M window holds 8 GiB a session beside a chunk scratch, and opening the
    // second ran the lead out of memory).
    let (mut session, pending, plan) = match (kept, plan) {
        (Some(kept), Some(plan)) => (kept.session, kept.pending, plan),
        (kept, _) => {
            drop(kept);
            let session = backend
                .open_session()
                .map_err(|error| format!("the backend could not open a sequence: {error}"))?;
            (session, None, Continuation::COLD)
        }
    };
    if let Some(ids) = &plan.respelled {
        eprintln!(
            "sparkflash serve: prefix reuse by text: the prompt spells the held reply with other ids; {} held ids stand for {} of its {}, {} to forward",
            plan.held,
            plan.covered,
            job.prompt_tokens.len(),
            ids.len() - plan.held,
        );
    }
    if plan.rewound {
        eprintln!(
            "sparkflash serve: prefix reuse by rewind: the prompt keeps the held prompt of {} ids and parts from the reply after it; {} to forward",
            plan.held,
            job.prompt_tokens.len() - plan.held,
        );
    }
    if plan.resume {
        eprintln!(
            "sparkflash serve: prefix reuse: the prompt is the held sequence of {} ids; answering from its held step",
            plan.held,
        );
    }
    let (ending, last) = serve_request(
        session.as_mut(),
        codec,
        &job,
        metrics,
        &plan,
        pending.as_ref().map(|(_, step)| step),
        started,
    );
    // Keep the sequence unless the backend failed in it. One the request moved keeps the step
    // after its last token; one it never touched keeps the step it had.
    if !matches!(ending, Err(Ending::Failed(_)))
        && session
            .reusable_prefix()
            .is_some_and(|prefix| !prefix.is_empty())
    {
        let handed = plan
            .respelled
            .as_ref()
            .map_or(job.prompt_tokens.len(), Vec::len);
        let pending = match last {
            Some(step) => Some((sampling_for(job.sampling, handed), step)),
            None => pending,
        };
        *held = Some(Held { session, pending });
    }
    ending.map_err(Ending::message)
}

/// Runs `job` on `session` as `plan` says, and returns how it ended with the step after the last
/// token the session holds, when the request moved the session.
fn serve_request(
    session: &mut dyn ModelSession,
    codec: &dyn PromptCodec,
    job: &Job,
    metrics: &Metrics,
    plan: &Continuation,
    pending: Option<&StepOutput>,
    started: Instant,
) -> (Result<(), Ending>, Option<StepOutput>) {
    let send = |reply: Reply| {
        job.replies
            .blocking_send(reply)
            .map_err(|_| Ending::ClientGone)
    };
    // The worker owns the decision, so the client learns the hit before the prefill runs and the
    // event stays first in the stream.
    if let Err(ending) = send(Reply::Started {
        cached_tokens: plan.covered,
    }) {
        return (Err(ending), None);
    }
    let prompt = plan.respelled.as_deref().unwrap_or(&job.prompt_tokens);
    let prefill_started = Instant::now();
    let (mut step, cached) = if plan.resume {
        // The held step was drawn under this request's sampler (see `continuation`); setting it
        // again could restart the random source that step already drew from.
        let Some(step) = pending else {
            return (
                Err(Ending::Failed(
                    "the held sequence was to resume from a step it does not have".to_owned(),
                )),
                None,
            );
        };
        (step.clone(), job.prompt_tokens.len())
    } else {
        session.set_sampling(sampling_for(job.sampling, prompt.len()));
        match session.prefill_images(prompt, &job.images) {
            Ok(step) => (step, client_cached(session.cached_tokens(), plan)),
            Err(error) => {
                return (
                    Err(Ending::Failed(format!(
                        "prefill failed after {} prompt token(s): {error}",
                        job.prompt_tokens.len()
                    ))),
                    None,
                );
            }
        }
    };
    let prefill_elapsed = prefill_started.elapsed();
    // A client that left during the prefill is caught before the first decode, so the sequence
    // holds exactly the prompt and the retry answers from this step.
    if job.replies.is_closed() {
        return (Err(Ending::ClientGone), Some(step));
    }
    let mut generated = Vec::new();
    let mut decoded = String::new();
    let mut stop_buffer = StopBuffer::new(&job.stop);
    let mut next = step.next_token;
    let mut completion_tokens = 0usize;
    let mut ttft: Option<Duration> = None;
    let want_sample = job.logprobs.wants_sample();
    while completion_tokens < job.max_tokens {
        // A forced request feeds its own token whatever the backend chose, including past a
        // choice that would have ended generation; the logprobs below score it under this row.
        let Some(token) = job.forced_tokens.get(completion_tokens).copied().or(next) else {
            break;
        };
        if ttft.is_none() {
            metrics.observe_time_to_first_token(job.enqueued_at.elapsed().as_secs_f64());
            ttft = Some(started.elapsed());
        }
        generated.push(token);
        let piece = decode_new(codec, &mut decoded, &generated);
        let emit = stop_buffer.push(piece.as_deref());
        if want_sample {
            let entry = match token_logprob(&step.logits, token, &job.logprobs, codec) {
                Ok(entry) => entry,
                Err(message) => return (Err(Ending::Refused(message)), Some(step)),
            };
            if !emit.is_empty()
                && let Err(ending) = send(Reply::Delta(emit))
            {
                return (Err(ending), Some(step));
            }
            if let Err(ending) = send(Reply::TokenLogprobs(Box::new(entry))) {
                return (Err(ending), Some(step));
            }
        } else if !emit.is_empty()
            && let Err(ending) = send(Reply::Delta(emit))
        {
            return (Err(ending), Some(step));
        }
        completion_tokens += 1;
        metrics.record_generation_tokens(1);
        if stop_buffer.matched() {
            break;
        }
        step = match session.decode(token) {
            Ok(step) => step,
            Err(error) => {
                return (
                    Err(Ending::Failed(format!(
                        "decode failed after {completion_tokens} token(s): {error}"
                    ))),
                    None,
                );
            }
        };
        next = step.next_token;
    }
    let tail = stop_buffer.finish();
    if !tail.is_empty()
        && let Err(ending) = send(Reply::Delta(tail))
    {
        return (Err(ending), Some(step));
    }
    let elapsed = started.elapsed();
    // The prefill rate is the work the engine did, not the prompt it was given: the tokens the
    // session already held cost nothing and are reported as the request's cache hit.
    let timing = RequestTiming::measure(
        job.prompt_tokens.len().saturating_sub(cached),
        completion_tokens,
        prefill_elapsed,
        ttft,
        elapsed,
    );
    let finish_reason = if stop_buffer.matched() {
        FinishReason::Stop
    } else if completion_tokens >= job.max_tokens {
        FinishReason::Length
    } else {
        FinishReason::Stop
    };
    let speculation = session.speculation();
    let ending = send(Reply::Done(Box::new(HostSummary {
        completion_tokens,
        finish_reason,
        timing,
        speculation,
        cached_tokens: cached,
    })));
    (ending, Some(step))
}

/// Decodes the whole generated prefix and returns the part that is new.
///
/// A prompt codec's decode is a pure function of the ids. When a new token completes a multibyte
/// character the decoder may revise a trailing replacement glyph; the common-prefix rule emits the
/// corrected suffix at the cost of a possible duplicate, which is why the contract exposes a
/// one-shot decode rather than a stateful stream today.
fn decode_new(codec: &dyn PromptCodec, previous: &mut String, ids: &[u32]) -> Option<String> {
    let full = codec.decode(ids)?;
    let mut common = common_prefix_bytes(previous, &full);
    while common > 0 && !full.is_char_boundary(common) {
        common -= 1;
    }
    let delta = full[common..].to_owned();
    *previous = full;
    Some(delta)
}

/// The bytes two decodes share, the common prefix a token's text delta is measured against.
pub fn common_prefix_bytes(left: &str, right: &str) -> usize {
    left.bytes()
        .zip(right.bytes())
        .take_while(|(a, b)| a == b)
        .count()
}

/// The holdback buffer for stop strings; the rule is vLLM's: hold up to the longest stop minus
/// one character so a match split across tokens is still detected.
pub struct StopBuffer {
    sequences: Vec<String>,
    holdback: usize,
    pending: String,
    matched: bool,
}

impl StopBuffer {
    /// Builds the buffer for a stop-string set, holding back the longest stop minus one character.
    pub fn new(sequences: &[String]) -> Self {
        let holdback = sequences
            .iter()
            .map(|sequence| sequence.chars().count())
            .max()
            .unwrap_or(0)
            .saturating_sub(1);
        Self {
            sequences: sequences.to_vec(),
            holdback,
            pending: String::new(),
            matched: false,
        }
    }

    /// Appends the text one token decoded to and returns the part that is safe to emit.
    pub fn push(&mut self, piece: Option<&str>) -> String {
        if self.matched {
            return String::new();
        }
        if let Some(piece) = piece {
            self.pending.push_str(piece);
        }
        if self.sequences.is_empty() {
            return std::mem::take(&mut self.pending);
        }
        if let Some(index) = self
            .sequences
            .iter()
            .filter_map(|sequence| self.pending.find(sequence.as_str()))
            .min()
        {
            self.matched = true;
            let text = self.pending[..index].to_owned();
            self.pending.clear();
            return text;
        }
        let chars = self.pending.chars().count();
        if chars <= self.holdback {
            return String::new();
        }
        let emit_chars = chars - self.holdback;
        let split = self
            .pending
            .char_indices()
            .nth(emit_chars)
            .map_or(self.pending.len(), |(index, _)| index);
        let text = self.pending[..split].to_owned();
        self.pending.drain(..split);
        text
    }

    /// Returns the held-back text at end of stream, so a trailing match is not lost.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    /// Whether a stop string has been seen; the buffer emits nothing after that.
    pub fn matched(&self) -> bool {
        self.matched
    }
}

fn token_logprob(
    logits: &[f32],
    token: u32,
    request: &LogprobsRequest,
    codec: &dyn PromptCodec,
) -> Result<TokenLogprob, String> {
    if logits.is_empty() {
        return Err("the backend returned no logits for a token-logprob request".to_owned());
    }
    let row = log_softmax(logits);
    let own = row
        .get(token as usize)
        .copied()
        .unwrap_or(f32::NEG_INFINITY);
    let token_string = codec.token_str(token).unwrap_or_default();
    let own_rank = competition_rank(&row, own);
    let mut selected: Vec<u32> = Vec::new();
    if !request.token_ids.is_empty() {
        selected.extend_from_slice(&request.token_ids);
    } else {
        match request.sample {
            Some(LogprobSelection::All) => {
                selected.extend((0..row.len()).map(|index| index as u32));
            }
            Some(LogprobSelection::Top(count)) => {
                selected.extend(top_k(&row, count as usize).into_iter().map(|i| i as u32));
            }
            None => {}
        }
    }
    selected.retain(|id| *id != token);
    let mut top = vec![LogprobAlternative {
        token_id: token,
        token: token_string.clone(),
        logprob: own,
        rank: Some(own_rank),
    }];
    let unranked = matches!(request.sample, Some(LogprobSelection::All));
    for id in selected {
        let value = row.get(id as usize).copied().unwrap_or(f32::NEG_INFINITY);
        top.push(LogprobAlternative {
            token_id: id,
            token: codec.token_str(id).unwrap_or_default(),
            logprob: value,
            rank: (!unranked).then(|| competition_rank(&row, value)),
        });
    }
    Ok(TokenLogprob {
        token_id: token,
        token: token_string,
        logprob: own,
        rank: (!unranked).then_some(own_rank),
        top,
    })
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![f32::NEG_INFINITY; logits.len()];
    }
    let sum: f64 = logits
        .iter()
        .map(|&value| ((value - max) as f64).exp())
        .sum();
    if !(sum.is_finite() && sum > 0.0) {
        return vec![f32::NEG_INFINITY; logits.len()];
    }
    let log_sum = sum.ln();
    logits
        .iter()
        .map(|&value| ((value - max) as f64 - log_sum) as f32)
        .collect()
}

fn competition_rank(logprobs: &[f32], token_logprob: f32) -> u32 {
    let greater = logprobs
        .iter()
        .filter(|&&value| value > token_logprob)
        .count();
    u32::try_from(greater).unwrap_or(u32::MAX).saturating_add(1)
}

fn top_k(logprobs: &[f32], count: usize) -> Vec<usize> {
    let count = count.min(logprobs.len());
    let mut order: Vec<usize> = (0..logprobs.len()).collect();
    order.sort_unstable_by(|&left, &right| {
        logprobs[right]
            .total_cmp(&logprobs[left])
            .then_with(|| left.cmp(&right))
    });
    order.truncate(count);
    order
}

fn finished_event(prompt_tokens: usize, summary: HostSummary, fabric: &str) -> EngineEvent {
    let completion_tokens = summary.completion_tokens;
    let speculation = summary.speculation;
    let cached = summary.cached_tokens.min(prompt_tokens);
    let stats = DeepSeekSparkRequestStats {
        kv_cache: PrefixMatchPlan {
            total_prompt_tokens: prompt_tokens,
            global_kv_cached_tokens: cached,
            uma_hit_tokens: cached,
            nvme_restored_tokens: 0,
            swa_tail_hit: false,
            encoder_swa_replay_tokens: 0,
            uncached_suffix_tokens: prompt_tokens - cached,
            decoder_swa_replay_tokens: 0,
        },
        kv_cache_hit_rate: if prompt_tokens == 0 {
            0.0
        } else {
            cached as f64 / prompt_tokens as f64
        },
        engram_layers: default_engram_layers(),
        // The confidence gate is the V4.1 driver's; the generic host offers every draft it gets.
        dspark_speculative: DSparkStepStats::from_counts(
            speculation.verify_steps,
            speculation.drafts_proposed,
            0,
            speculation.drafts_accepted,
        ),
        transport: RoceTransportStats {
            fabric_mode: fabric.to_owned(),
            ..RoceTransportStats::default()
        },
        timing: RequestTimingStats {
            ttft_ms: summary.timing.ttft_ms,
            prefill_tok_per_s: summary.timing.prefill_tok_per_s,
            decode_tok_per_s: summary.timing.decode_tok_per_s,
            min_rank_uma_free_gib: 0.0,
        },
    };
    EngineEvent::Finished {
        finish_reason: summary.finish_reason,
        usage: OpenAiUsageWithCache {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: PromptTokensDetails {
                cached_tokens: cached,
                uma_cached_tokens: cached,
                nvme_cached_tokens: 0,
                image_tokens: 0,
            },
            completion_tokens_details: CompletionTokensDetails {
                reasoning_tokens: 0,
                accepted_prediction_tokens: 0,
                rejected_prediction_tokens: 0,
            },
            prompt_cache_hit_tokens: cached,
            prompt_cache_miss_tokens: prompt_tokens - cached,
        },
        stats: Box::new(stats),
    }
}

fn cluster_snapshot(descriptor: &ModelDescriptor, fabric: &str) -> ClusterStatsSnapshot {
    ClusterStatsSnapshot {
        model: descriptor.id.clone(),
        fabric_mode: fabric.to_owned(),
        nodes: Vec::new(),
        kv: empty_kv_snapshot(),
        engram: Vec::new(),
        transport: RoceTransportStats {
            fabric_mode: fabric.to_owned(),
            ..RoceTransportStats::default()
        },
        dspark: DSparkStepStats::default(),
        timing: RequestTimingStats::default(),
        max_seq_len: max_seq_len_for(descriptor.context_tokens),
    }
}

fn refused(message: &str, code: &str) -> BoxStream<'static, EngineEvent> {
    stream::iter([EngineEvent::Error {
        message: message.to_owned(),
        code: code.to_owned(),
    }])
    .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_buffer_holds_a_split_sequence_until_it_matches() {
        let mut buffer = StopBuffer::new(&["END".to_owned()]);
        assert_eq!(buffer.push(Some("hello EN")), "hello ");
        assert!(!buffer.matched());
        assert_eq!(buffer.push(Some("D")), "");
        assert!(buffer.matched());
        assert_eq!(buffer.push(Some("ignored")), "");
        assert_eq!(buffer.finish(), "");
    }

    #[test]
    fn stop_buffer_releases_the_tail_when_no_stop_matches() {
        let mut buffer = StopBuffer::new(&["END".to_owned()]);
        assert_eq!(buffer.push(Some("hello EN")), "hello ");
        assert_eq!(buffer.finish(), "EN");
    }

    #[test]
    fn the_common_prefix_is_the_part_already_decoded() {
        assert_eq!(common_prefix_bytes("hello", "hello world"), 5);
        assert_eq!(common_prefix_bytes("hello", "help"), 3);
        assert_eq!(common_prefix_bytes("", "x"), 0);
    }

    #[test]
    fn log_softmax_of_a_one_hot_row_puts_all_mass_on_that_token() {
        let row = log_softmax(&[f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY]);
        assert_eq!(row[0], f32::NEG_INFINITY);
        assert!((row[1] - 0.0).abs() < 1e-6);
        assert_eq!(row[2], f32::NEG_INFINITY);
    }
}

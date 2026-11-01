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
//! The driver owns one backend and opens one sequence per request. Requests arrive through a
//! bounded channel and their events leave through a bounded reply channel, so a client that stops
//! reading applies backpressure to the engine rather than growing an unbounded buffer.
//!
//! `HostOptions::concurrency` chooses how many requests execute at once. Each worker owns one
//! session at a time, so concurrency N means N independent sequences progressing in parallel; it
//! is not a fused batch and does not change the arithmetic of any one sequence. The default is 1,
//! which preserves the serial one-job-at-a-time behavior. A value above 1 is for backends whose
//! sessions are independent; a backend whose sessions share mutable engine state must serialize
//! that state itself (the Qwen device arm gives each session its own stream, so their kernels can
//! overlap). A backend that only supports one live session must stay at 1.
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
use sparkflash_model::{ModelBackend, ModelDescriptor, PromptCodec, Sampling, Speculation};
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
}

impl Default for HostOptions {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_MAX_TOKENS,
            queue_capacity: DEFAULT_QUEUE,
            reply_buffer: DEFAULT_REPLY_BUFFER,
            concurrency: 1,
        }
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
        let capabilities = backend.capabilities();
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
            let handle = std::thread::Builder::new()
                .name(format!("sparkflash-host-engine-{worker}"))
                .spawn(move || {
                    host_thread(backend, codec, queued, ready, thread_metrics);
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
        let cluster = cluster_snapshot(&descriptor);
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
        // The API prepares images into `vision` only for an engine that advertises vision. The
        // generic host driver has no way to write those rows over the placeholder embeddings yet,
        // and prefill would otherwise read the placeholders as ordinary text. Refuse rather than
        // answer from a prompt whose images were silently dropped.
        if !request.vision.is_empty() {
            return refused(
                "the generic host driver cannot prefill prepared images yet; use the V4.1 device arm or a text-only request",
                REFUSED_CODE,
            );
        }
        if request.prompt_tokens.is_empty() {
            return refused(
                "the request carries no prompt tokens, and the host backend has nothing to prefill",
                REFUSED_CODE,
            );
        }
        let prompt_tokens = request.prompt_tokens.len();
        let max_tokens = request.max_tokens.min(self.max_tokens);
        let enqueued_at = Instant::now();
        let (replies, answers) = mpsc::channel(self.reply_buffer);
        let job = Job {
            prompt_tokens: request.prompt_tokens,
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
        let head = stream::once(async move {
            EngineEvent::Started {
                prompt_tokens,
                cached_tokens: 0,
            }
        });
        let answers = stream::unfold(
            AnswerState {
                answers,
                done: false,
            },
            move |mut state| async move {
                if state.done {
                    return None;
                }
                match state.answers.recv().await {
                    Some(Reply::Delta(text)) => Some((EngineEvent::ContentDelta(text), state)),
                    Some(Reply::TokenLogprobs(logprob)) => {
                        Some((EngineEvent::TokenLogprobs(logprob), state))
                    }
                    Some(Reply::Done(summary)) => {
                        state.done = true;
                        Some((finished_event(prompt_tokens, *summary), state))
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
        head.chain(answers).boxed()
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
    done: bool,
}

/// One queued generation request.
struct Job {
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampling: Sampling,
    logprobs: LogprobsRequest,
    stop: Vec<String>,
    enqueued_at: Instant,
    replies: mpsc::Sender<Reply>,
}

/// One element of the worker's answer.
enum Reply {
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
}

/// The wall-clock numbers one request measured.
#[derive(Debug, Clone, Copy, Default)]
struct RequestTiming {
    ttft_ms: f64,
    prefill_tok_per_s: f64,
    decode_tok_per_s: f64,
}

impl RequestTiming {
    fn measure(
        prompt_tokens: usize,
        completion_tokens: usize,
        prefill: Duration,
        ttft: Option<Duration>,
        total: Duration,
    ) -> Self {
        let first = ttft.map_or(0.0, |instant| instant.as_secs_f64());
        Self {
            ttft_ms: first * 1000.0,
            prefill_tok_per_s: rate(prompt_tokens, prefill.as_secs_f64()),
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
) {
    if ready.send(Ok(())).is_err() {
        return;
    }
    // Do not keep the readiness sender alive for the worker's lifetime: `start` waits for one
    // report per worker, and a clone held by a worker that later hangs would keep the channel
    // open after the reports it needs have already arrived.
    drop(ready);
    while let Ok(job) = jobs.recv() {
        metrics.request_admitted();
        let replies = job.replies.clone();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            run_request(backend.as_ref(), codec.as_ref(), job, &metrics)
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

fn run_request(
    backend: &dyn ModelBackend,
    codec: &dyn PromptCodec,
    job: Job,
    metrics: &Metrics,
) -> Result<(), String> {
    let started = Instant::now();
    let mut session = backend
        .open_session()
        .map_err(|error| format!("the backend could not open a sequence: {error}"))?;
    session.set_sampling(job.sampling);
    if job.logprobs.wants_prompt() {
        return Err(
            "the generic host backend does not expose per-position prompt logits yet".to_owned(),
        );
    }
    let prefill_started = Instant::now();
    let mut step = session.prefill(&job.prompt_tokens).map_err(|error| {
        format!(
            "prefill failed after {} prompt token(s): {error}",
            job.prompt_tokens.len()
        )
    })?;
    let prefill_elapsed = prefill_started.elapsed();
    let mut generated = Vec::new();
    let mut decoded = String::new();
    let mut stop_buffer = StopBuffer::new(&job.stop);
    let mut next = step.next_token;
    let mut completion_tokens = 0usize;
    let mut ttft: Option<Duration> = None;
    let want_sample = job.logprobs.wants_sample();
    while completion_tokens < job.max_tokens {
        let Some(token) = next else {
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
            let entry = token_logprob(&step.logits, token, &job.logprobs, codec)?;
            if !emit.is_empty() {
                job.replies
                    .blocking_send(Reply::Delta(emit))
                    .map_err(|_| "the client stopped reading this response".to_owned())?;
            }
            job.replies
                .blocking_send(Reply::TokenLogprobs(Box::new(entry)))
                .map_err(|_| "the client stopped reading this response".to_owned())?;
        } else if !emit.is_empty() {
            job.replies
                .blocking_send(Reply::Delta(emit))
                .map_err(|_| "the client stopped reading this response".to_owned())?;
        }
        completion_tokens += 1;
        metrics.record_generation_tokens(1);
        if stop_buffer.matched() {
            break;
        }
        step = session.decode(token).map_err(|error| {
            format!("decode failed after {completion_tokens} token(s): {error}")
        })?;
        next = step.next_token;
    }
    let tail = stop_buffer.finish();
    if !tail.is_empty() {
        job.replies
            .blocking_send(Reply::Delta(tail))
            .map_err(|_| "the client stopped reading this response".to_owned())?;
    }
    let elapsed = started.elapsed();
    let timing = RequestTiming::measure(
        job.prompt_tokens.len(),
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
    job.replies
        .blocking_send(Reply::Done(Box::new(HostSummary {
            completion_tokens,
            finish_reason,
            timing,
            speculation: session.speculation(),
        })))
        .map_err(|_| "the client stopped reading this response".to_owned())?;
    Ok(())
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

fn common_prefix_bytes(left: &str, right: &str) -> usize {
    left.bytes()
        .zip(right.bytes())
        .take_while(|(a, b)| a == b)
        .count()
}

/// The holdback buffer for stop strings; the rule is vLLM's: hold up to the longest stop minus
/// one character so a match split across tokens is still detected.
struct StopBuffer {
    sequences: Vec<String>,
    holdback: usize,
    pending: String,
    matched: bool,
}

impl StopBuffer {
    fn new(sequences: &[String]) -> Self {
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

    fn push(&mut self, piece: Option<&str>) -> String {
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

    fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    fn matched(&self) -> bool {
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

fn finished_event(prompt_tokens: usize, summary: HostSummary) -> EngineEvent {
    let completion_tokens = summary.completion_tokens;
    let speculation = summary.speculation;
    let stats = DeepSeekSparkRequestStats {
        kv_cache: PrefixMatchPlan {
            total_prompt_tokens: prompt_tokens,
            global_kv_cached_tokens: 0,
            uma_hit_tokens: 0,
            nvme_restored_tokens: 0,
            swa_tail_hit: false,
            encoder_swa_replay_tokens: 0,
            // The host path has no prefix cache: every prompt token is part of the uncached
            // suffix. Reporting zero here made the extended stats look like a 100% hit while the
            // OpenAI usage block correctly said zero cached tokens.
            uncached_suffix_tokens: prompt_tokens,
            decoder_swa_replay_tokens: 0,
        },
        kv_cache_hit_rate: 0.0,
        engram_layers: default_engram_layers(),
        // The confidence gate is the V4.1 driver's; the generic host offers every draft it gets.
        dspark_speculative: DSparkStepStats::from_counts(
            speculation.verify_steps,
            speculation.drafts_proposed,
            0,
            speculation.drafts_accepted,
        ),
        transport: RoceTransportStats {
            fabric_mode: "none (host backend)".to_owned(),
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
                cached_tokens: 0,
                uma_cached_tokens: 0,
                nvme_cached_tokens: 0,
                image_tokens: 0,
            },
            completion_tokens_details: CompletionTokensDetails {
                reasoning_tokens: 0,
                accepted_prediction_tokens: 0,
                rejected_prediction_tokens: 0,
            },
            prompt_cache_hit_tokens: 0,
            prompt_cache_miss_tokens: prompt_tokens,
        },
        stats: Box::new(stats),
    }
}

fn cluster_snapshot(descriptor: &ModelDescriptor) -> ClusterStatsSnapshot {
    ClusterStatsSnapshot {
        model: descriptor.id.clone(),
        fabric_mode: "none (host backend)".to_owned(),
        nodes: Vec::new(),
        kv: empty_kv_snapshot(),
        engram: Vec::new(),
        transport: RoceTransportStats {
            fabric_mode: "none (host backend)".to_owned(),
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

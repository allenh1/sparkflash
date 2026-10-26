//! `sparkflash serve --tp4-rank R --tp4-device CUBINS`: the four-rank device forward with DSpark
//! speculative decoding behind the OpenAI API.
//!
//! Every rank is one process. Each connects the serve control channel first (rank 0's hub waits for
//! every worker's Hello), installs its [`DeviceLink`], and then runs the `tp4-chain` device epoch
//! on its main thread with a position capacity of `--max-model-len` plus one verify block. The
//! epoch's DSpark branch takes the link and serves its requests (`spec::serve`) instead of the
//! harness's one-shot generation, so the weights, the arena and the NCCL communicator are opened
//! once per process and every request starts from row 0.
//!
//! Rank 0 also owns HTTP, the tokenizer and the chat template. Its [`HubLink`] is fed by
//! [`DeviceDriver`] and turns each step's tokens into stream events on the device thread itself:
//! the reasoning channel up to `</think>` when the template opened one, the content channel after
//! it, with the stop strings cut there. The device loop serves several requests at once in
//! scheduler ticks; before each, rank 0 broadcasts the requests that join (each with its sampler,
//! seed included) and the sequences that end there (a stop string matched, or the client left).
//! Every rank runs the same scheduler on those decisions and draws the same counter-based noise,
//! so every rank emits the same tokens and ends every request at the same tick.
//!
//! A rank that dies takes the server down: rank 0 watches every worker's control socket and exits
//! when one closes, and a worker exits when rank 0's closes, instead of waiting in a collective.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use futures::stream::{self, BoxStream};
use sparkflash_api::encoding::{HfTokenizerEncoder, TokenizerSeam};
use sparkflash_api::engine::InferenceEngine as ApiInferenceEngine;
use sparkflash_api::engine::{
    Capabilities, ClusterStatsSnapshot, EngineEvent, EngineRequest, FinishReason, MAX_SEQ_LEN_ENV,
    MODEL_ID, MissingCapability,
};
use sparkflash_api::telemetry_types::DSparkStepStats;
use sparkflash_api::{AppState, Metrics};
use sparkflash_engine::config::ModelConfig;

use super::{
    ENGINE_ERROR_CODE, ENGINE_GONE_CODE, EXIT_REFUSED, REFUSED_CODE, RequestTiming, RunSummary,
    ServeOptions, StopBuffer, TP4_CONTROL_TIMEOUT, TP4_TICK_PROMPT_IDS, Tp4ControlFrame,
    Tp4ControlHub, Tp4ControlWorker, Tp4DeviceAdmit, Tp4DeviceImage, Tp4ServeOptions,
    bind_and_serve, default_tokenizer, epoch_millis, finished_event_with_fabric, load_tokenizer,
    refused, tp4_cluster_snapshot,
};
use crate::device_link::{DeviceFinish, DeviceJob, DeviceLink, DeviceOutcome, DeviceTick};
use sparkflash_api::VisionImage;
use sparkflash_vision::{Geometry, Grid, IMAGE_TOKEN_ID, ImageSpan};

/// Positions kept past `--max-model-len` for the last verify block (k drafts, the draft block and
/// the bonus row), so a request that fills the context still has its rows.
pub(super) const POSITION_HEADROOM: usize = 16;

/// The most positions the device path opens: the model's 1M-token context
/// (`max_position_embeddings`) plus the verify headroom. The radix select scores up to twice that
/// (opt_select_sm121.cu's FAST_MAX_POS).
pub(super) const DEVICE_MAX_POSITIONS: usize = (1 << 20) + POSITION_HEADROOM;

/// The default `--max-model-len`: the whole device capacity less the verify headroom.
pub(super) const DEFAULT_MAX_MODEL_LEN: usize = DEVICE_MAX_POSITIONS - POSITION_HEADROOM;

/// Requests that may wait for the device loop to admit them before a request is refused. The loop
/// admits as many as it has room for (its lanes and its own queue) at every tick.
const DEVICE_JOB_QUEUE: usize = 64;

/// Below this temperature a request is greedy, as in vLLM (`_SAMPLING_EPS`).
const SAMPLING_EPS: f32 = 1e-5;

/// The fabric the device path reports: NCCL over the RoCE rail, plus the TCP control channel.
const DEVICE_FABRIC: &str = "nccl roce tp4 device";

/// The template's reasoning delimiters and the end of sequence, resolved by name from the tokenizer.
const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";
const END_OF_SEQUENCE: &str = "<｜end▁of▁sentence｜>";

/// Set once a worker has read rank 0's End, so its watchdog does not treat the close that
/// follows as a lost leader.
static LEADER_ENDED: AtomicBool = AtomicBool::new(false);

/// Ends the process now, without the C exit handlers: CUDA's tears the context down, which waits
/// for the kernels in flight, and a collective whose peer has gone never finishes, so a rank whose
/// group failed would stay up, holding its memory and its port. Rust runs no destructors on exit
/// either, so only the standard streams need flushing.
fn exit_now(code: i32) -> ! {
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // SAFETY: _exit takes no pointers and does not return.
    unsafe { libc::_exit(code) }
}

/// The device path's flags, parsed only with `--tp4-device`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceServeOptions {
    /// The kernel cubin directory (the harness's `--cubin-dir` and `--kernel-cubin-dir`).
    pub cubins: PathBuf,
    /// The longest prompt plus completion served; the device capacity is this plus
    /// [`POSITION_HEADROOM`].
    pub max_model_len: usize,
    /// Where each rank's epoch writes its bind evidence.
    pub evidence_dir: PathBuf,
}

impl DeviceServeOptions {
    /// Builds the options from the flags, or `None` without `--tp4-device`.
    ///
    /// # Errors
    /// A `--max-model-len` of zero, or one whose capacity exceeds [`DEVICE_MAX_POSITIONS`].
    pub(super) fn from_cli(
        cubins: Option<PathBuf>,
        max_model_len: Option<usize>,
        evidence_dir: Option<PathBuf>,
    ) -> Result<Option<Self>, String> {
        let Some(cubins) = cubins else {
            return Ok(None);
        };
        let max_model_len = max_model_len.unwrap_or(DEFAULT_MAX_MODEL_LEN);
        if max_model_len == 0 || max_model_len > DEFAULT_MAX_MODEL_LEN {
            return Err(format!(
                "--max-model-len {max_model_len}: the device path serves 1..={DEFAULT_MAX_MODEL_LEN} tokens ({DEVICE_MAX_POSITIONS} positions, less {POSITION_HEADROOM} for the last verify block)"
            ));
        }
        Ok(Some(Self {
            cubins,
            max_model_len,
            evidence_dir: evidence_dir.unwrap_or_else(|| PathBuf::from(".")),
        }))
    }

    /// The device capacity the epoch opens.
    fn positions(&self) -> usize {
        self.max_model_len + POSITION_HEADROOM
    }
}

/// One admitted request, from the driver to rank 0's link.
struct Pending {
    job: DeviceJob,
    /// Whether the template opened a reasoning block: the output starts in the reasoning channel.
    reasoning: bool,
    skip_special_tokens: bool,
    stop: Vec<String>,
    enqueued_at: Instant,
    events: tokio::sync::mpsc::UnboundedSender<EngineEvent>,
}

/// The HTTP side of the device path: validates and queues requests for rank 0's device loop.
pub struct DeviceDriver {
    jobs: mpsc::SyncSender<Pending>,
    max_tokens: usize,
    max_model_len: usize,
    vocab_size: usize,
    think_start: Option<u32>,
    end_of_sequence: u32,
    cluster: ClusterStatsSnapshot,
    metrics: Arc<Metrics>,
    seeds: RandomState,
    draws: AtomicU64,
    /// Whether the epoch runs the vision tower (the cubin is staged; DECISIONS.md D59).
    vision: bool,
}

impl DeviceDriver {
    /// A seed for a request that named none: every request is its own draw.
    fn random_seed(&self) -> u64 {
        let mut hasher = self.seeds.build_hasher();
        hasher.write_u64(self.draws.fetch_add(1, Ordering::Relaxed));
        hasher.finish()
    }
}

/// The first request field the device path cannot honour, if any.
fn unsupported(request: &EngineRequest) -> Option<&'static str> {
    if !request.logprobs.is_empty() {
        return Some(
            "logprobs are not served on the TP=4 device path yet: the verify head exchanges each rank's maximum (or its top-128 for a top-p/top-k request), not a log-softmax",
        );
    }
    if request.repetition_penalty != 1.0
        || request.presence_penalty != 0.0
        || request.frequency_penalty != 0.0
    {
        return Some(
            "repetition, presence and frequency penalties are not served on the TP=4 device path yet",
        );
    }
    None
}

/// Why the prompt's image spans cannot be what the router built (each span in bounds, in order and
/// apart, every span position and no other holding the placeholder, the grid the canvas makes, and
/// a canvas of that size), if they are not.
fn bad_spans(prompt: &[u32], images: &[VisionImage]) -> Option<String> {
    let mut next = 0usize;
    for (index, image) in images.iter().enumerate() {
        let span = &image.span;
        let grid = span.grid;
        let patch = Geometry::V41.patch;
        if grid.best_height == 0
            || grid.best_width == 0
            || grid.best_height % patch != 0
            || grid.best_width % patch != 0
            || Geometry::V41.grid(grid.best_height, grid.best_width) != grid
        {
            return Some(format!("image {index}: {grid:?} is not a canvas's grid"));
        }
        if image.pixels.len() != grid.best_height * grid.best_width * 3 {
            return Some(format!(
                "image {index}: {} canvas bytes for a {} x {} canvas",
                image.pixels.len(),
                grid.best_width,
                grid.best_height
            ));
        }
        if span.start < next || span.end() > prompt.len() {
            return Some(format!(
                "image {index}: positions {}..{} overlap the span before it or pass the {}-token prompt",
                span.start,
                span.end(),
                prompt.len()
            ));
        }
        if let Some(offset) = prompt[next..span.start]
            .iter()
            .position(|&id| id == IMAGE_TOKEN_ID)
        {
            return Some(format!(
                "position {}: the image placeholder outside an image span",
                next + offset
            ));
        }
        if prompt[span.start..span.end()]
            .iter()
            .any(|&id| id != IMAGE_TOKEN_ID)
        {
            return Some(format!(
                "image {index}: a position of its span does not hold the placeholder"
            ));
        }
        next = span.end();
    }
    prompt[next..]
        .iter()
        .position(|&id| id == IMAGE_TOKEN_ID)
        .map(|offset| {
            format!(
                "position {}: the image placeholder outside an image span",
                next + offset
            )
        })
}

impl ApiInferenceEngine for DeviceDriver {
    fn model_id(&self) -> &str {
        MODEL_ID
    }

    /// Each generate call is its own request with its own seed (a fresh draw when the request
    /// named none), so the choices of one response are independent samples.
    fn independent_choices(&self) -> bool {
        true
    }

    fn generate(&self, request: EngineRequest) -> BoxStream<'static, EngineEvent> {
        if !request.images.is_empty() || (!request.vision.is_empty() && !self.vision) {
            let missing = MissingCapability::vision();
            return refused(&missing.to_string(), missing.code());
        }
        if let Some(reason) = unsupported(&request) {
            return refused(reason, REFUSED_CODE);
        }
        if let Some(reason) = bad_spans(&request.prompt_tokens, &request.vision) {
            return refused(&reason, REFUSED_CODE);
        }
        let prompt_tokens = request.prompt_tokens.len();
        if prompt_tokens == 0 {
            return refused(
                "the request carries no prompt tokens, and the device forward has nothing to prefill",
                REFUSED_CODE,
            );
        }
        if prompt_tokens >= self.max_model_len {
            return refused(
                &format!(
                    "a prompt of {prompt_tokens} tokens leaves no room in the {}-token device context",
                    self.max_model_len
                ),
                REFUSED_CODE,
            );
        }
        let greedy = request.temperature < SAMPLING_EPS;
        let seed = request.seed.unwrap_or_else(|| self.random_seed());
        let reasoning = request.thinking
            && self.think_start.is_some()
            && request.prompt_tokens.last().copied() == self.think_start;
        let job = DeviceJob {
            max_tokens: request.max_tokens.min(self.max_tokens).max(1),
            temperature: if greedy { 0.0 } else { request.temperature },
            top_p: if greedy {
                1.0
            } else {
                request.top_p.clamp(0.0, 1.0)
            },
            top_k: if greedy {
                0
            } else {
                u32::try_from(request.top_k).unwrap_or(0)
            },
            seed,
            stop_ids: vec![self.end_of_sequence],
            prompt: request.prompt_tokens,
            images: request.vision,
        };
        let (events, answers) = tokio::sync::mpsc::unbounded_channel();
        let pending = Pending {
            job,
            reasoning,
            skip_special_tokens: request.skip_special_tokens,
            stop: request.stop,
            enqueued_at: Instant::now(),
            events,
        };
        self.metrics.request_enqueued();
        if self.jobs.try_send(pending).is_err() {
            self.metrics.request_enqueue_refused();
            return refused(
                &format!(
                    "the device queue is full: {DEVICE_JOB_QUEUE} requests are waiting for the device"
                ),
                REFUSED_CODE,
            );
        }
        let head = stream::once(async move {
            EngineEvent::Started {
                prompt_tokens,
                cached_tokens: 0,
            }
        });
        let answers = stream::unfold((answers, false), |(mut answers, done)| async move {
            if done {
                return None;
            }
            match answers.recv().await {
                Some(event) => {
                    let terminal = matches!(
                        event,
                        EngineEvent::Finished { .. } | EngineEvent::Error { .. }
                    );
                    Some((event, (answers, terminal)))
                }
                None => Some((
                    EngineEvent::Error {
                        message: "the device loop stopped before this request finished".to_owned(),
                        code: ENGINE_GONE_CODE.to_owned(),
                    },
                    (answers, true),
                )),
            }
        });
        head.chain(answers).boxed()
    }

    fn cluster_stats(&self) -> ClusterStatsSnapshot {
        self.cluster.clone()
    }

    /// Tool calls are parsed from the content by the router; the reasoning channel and the
    /// effort are the chat template's, which renders both into the prompt this path runs. Images
    /// are served when the epoch runs the vision tower.
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            vision: self.vision,
            ..Capabilities::ALL
        }
    }

    fn vocab_size(&self) -> Option<usize> {
        Some(self.vocab_size)
    }
}

/// The streaming detokeniser's state (what `tokenizers::DecodeStream` keeps).
#[derive(Default)]
struct DecodeState {
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

/// A request on the device, as rank 0's link streams it.
struct Live {
    pending: Pending,
    started: Instant,
    ttft: Option<Duration>,
    /// Still in the reasoning channel (until `</think>`).
    reasoning: bool,
    reasoning_tokens: usize,
    completion_tokens: usize,
    decode: DecodeState,
    stop: StopBuffer,
    /// The client stopped reading: the rest of the request is not sent.
    closed: bool,
    /// A failure on rank 0 alone (the detokeniser): the request ends with it, the group carries on.
    failed: Option<String>,
}

impl Live {
    fn new(pending: Pending) -> Self {
        let stop = StopBuffer::new(&pending.stop);
        Self {
            reasoning: pending.reasoning,
            pending,
            started: Instant::now(),
            ttft: None,
            reasoning_tokens: 0,
            completion_tokens: 0,
            decode: DecodeState::default(),
            stop,
            closed: false,
            failed: None,
        }
    }

    fn send(&mut self, event: EngineEvent) {
        if !self.closed && self.pending.events.send(event).is_err() {
            self.closed = true;
        }
    }

    /// Streams one step's tokens; false when the request must end here (a stop string matched,
    /// the client left, or the text could not be decoded).
    fn emit(
        &mut self,
        tokens: &[u32],
        tokenizer: &HfTokenizerEncoder,
        think_end: Option<u32>,
        metrics: &Metrics,
    ) -> bool {
        for &token in tokens {
            // A tick sent ahead runs one more step after the stream ends: its tokens go nowhere.
            if self.stop.matched() || self.failed.is_some() {
                break;
            }
            self.completion_tokens += 1;
            metrics.record_generation_tokens(1);
            if self.ttft.is_none() {
                metrics
                    .observe_time_to_first_token(self.pending.enqueued_at.elapsed().as_secs_f64());
                self.ttft = Some(self.started.elapsed());
            }
            if self.reasoning && Some(token) == think_end {
                // The content channel starts after the delimiter, with its own decoder.
                self.reasoning = false;
                self.reasoning_tokens += 1;
                self.decode = DecodeState::default();
                continue;
            }
            let piece = match tokenizers::step_decode_stream(
                tokenizer.tokenizer(),
                vec![token],
                self.pending.skip_special_tokens,
                &mut self.decode.ids,
                &mut self.decode.prefix,
                &mut self.decode.prefix_index,
            ) {
                Ok(piece) => piece.filter(|piece| !piece.is_empty()),
                Err(error) => {
                    self.failed = Some(format!(
                        "the tokenizer could not decode token {token}: {error}"
                    ));
                    return false;
                }
            };
            if self.reasoning {
                self.reasoning_tokens += 1;
                if let Some(piece) = piece {
                    self.send(EngineEvent::ReasoningDelta(piece));
                }
            } else {
                // The stop matcher owns the content: it returns what is safe to send now.
                let text = self.stop.push(piece.as_deref());
                if !text.is_empty() {
                    self.send(EngineEvent::ContentDelta(text));
                }
            }
        }
        !(self.stop.matched() || self.closed || self.failed.is_some())
    }

    /// The request is over: the held-back text, then the terminal event.
    fn finish(mut self, outcome: Result<&DeviceOutcome, &str>, metrics: &Metrics) {
        let prompt_tokens = self.pending.job.prompt.len();
        let outcome = match (outcome, self.failed.take()) {
            (Err(message), _) => Err(message.to_owned()),
            (Ok(_), Some(message)) => Err(message),
            (Ok(outcome), None) => Ok(*outcome),
        };
        match outcome {
            Ok(outcome) => {
                let tail = self.stop.finish();
                if !tail.is_empty() {
                    self.send(EngineEvent::ContentDelta(tail));
                }
                let finish_reason = match outcome.finish {
                    DeviceFinish::Length if !self.stop.matched() => FinishReason::Length,
                    _ => FinishReason::Stop,
                };
                let timing = RequestTiming::measure(
                    prompt_tokens,
                    self.completion_tokens,
                    Duration::from_secs_f64(outcome.prefill_ms / 1e3),
                    self.ttft,
                    self.started.elapsed(),
                );
                let dspark = DSparkStepStats::from_counts(
                    outcome.steps,
                    outcome.drafts_verified,
                    0,
                    outcome.drafts_accepted,
                );
                println!(
                    "sparkflash serve: device_request epoch_ms={} prompt_tokens={prompt_tokens} cached_tokens={} completion_tokens={} reasoning_tokens={} finish={finish_reason:?} prefill_ms={:.1} ttft_ms={:.1} decode_tok_s={:.2} steps={} accept_length={:.3}",
                    epoch_millis(),
                    outcome.cached_tokens,
                    self.completion_tokens,
                    self.reasoning_tokens,
                    outcome.prefill_ms,
                    timing.ttft_ms,
                    timing.decode_tok_per_s,
                    outcome.steps,
                    dspark.mean_tokens_per_verify_step,
                );
                let mut event = finished_event_with_fabric(
                    prompt_tokens,
                    RunSummary {
                        completion_tokens: self.completion_tokens,
                        finish_reason,
                        timing,
                        dspark,
                    },
                    DEVICE_FABRIC,
                );
                if let EngineEvent::Finished { usage, .. } = &mut event {
                    usage.completion_tokens_details.reasoning_tokens = self.reasoning_tokens;
                    usage.prompt_tokens_details.cached_tokens = outcome.cached_tokens;
                    usage.prompt_tokens_details.image_tokens =
                        sparkflash_api::vision::image_tokens(&self.pending.job.images);
                    usage.prompt_cache_hit_tokens = outcome.cached_tokens;
                    usage.prompt_cache_miss_tokens =
                        prompt_tokens.saturating_sub(outcome.cached_tokens);
                }
                self.send(event);
            }
            Err(message) => {
                eprintln!("sparkflash serve: device request failed: {message}");
                self.send(EngineEvent::Error {
                    message,
                    code: ENGINE_ERROR_CODE.to_owned(),
                });
            }
        }
        metrics.request_finished();
    }
}

/// The admission every worker runs.
fn admit_frame(seq: u64, job: &DeviceJob) -> Tp4DeviceAdmit {
    Tp4DeviceAdmit {
        seq,
        prompt_ids: job.prompt.clone(),
        max_tokens: u32::try_from(job.max_tokens).unwrap_or(u32::MAX),
        temperature_bits: job.temperature.to_bits(),
        top_p_bits: job.top_p.to_bits(),
        top_k: job.top_k,
        seed: job.seed,
        stop_ids: job.stop_ids.clone(),
        images: job
            .images
            .iter()
            .map(|image| {
                let span = &image.span;
                let narrow = |value: usize| u32::try_from(value).unwrap_or(u32::MAX);
                Tp4DeviceImage {
                    start: span.start as u64,
                    n_llm_h: narrow(span.grid.n_llm_h),
                    n_llm_w: narrow(span.grid.n_llm_w),
                    best_height: narrow(span.grid.best_height),
                    best_width: narrow(span.grid.best_width),
                    key: span.key.iter().map(|byte| format!("{byte:02x}")).collect(),
                }
            })
            .collect(),
    }
}

/// The request a worker reads from an admission, with its canvases from `pixels` (the tower
/// rank's, by (sequence, image)); an image without one gets an empty canvas.
///
/// # Errors
/// An image's key is not 64 hex digits.
fn admitted_job(
    admit: Tp4DeviceAdmit,
    pixels: &mut std::collections::HashMap<(u64, u32), Arc<[u8]>>,
) -> Result<(u64, DeviceJob), String> {
    let seq = admit.seq;
    let mut images = Vec::with_capacity(admit.images.len());
    for (index, image) in admit.images.into_iter().enumerate() {
        let mut key = [0u8; 32];
        let hex = image.key.as_bytes();
        if hex.len() != 64 {
            return Err(format!(
                "sequence {seq} image {index}: a key of {} digits",
                hex.len()
            ));
        }
        for (byte, pair) in key.iter_mut().zip(hex.as_chunks::<2>().0) {
            let pair = std::str::from_utf8(pair).map_err(|error| error.to_string())?;
            *byte = u8::from_str_radix(pair, 16)
                .map_err(|_| format!("sequence {seq} image {index}: the key is not hex"))?;
        }
        let index = u32::try_from(index).unwrap_or(u32::MAX);
        images.push(VisionImage {
            span: ImageSpan {
                start: image.start as usize,
                grid: Grid {
                    n_llm_h: image.n_llm_h as usize,
                    n_llm_w: image.n_llm_w as usize,
                    best_height: image.best_height as usize,
                    best_width: image.best_width as usize,
                },
                key,
            },
            pixels: pixels
                .remove(&(seq, index))
                .unwrap_or_else(|| Arc::from(&[][..])),
        });
    }
    Ok((
        seq,
        DeviceJob {
            prompt: admit.prompt_ids,
            max_tokens: admit.max_tokens as usize,
            temperature: f32::from_bits(admit.temperature_bits),
            top_p: f32::from_bits(admit.top_p_bits),
            top_k: admit.top_k,
            seed: admit.seed,
            stop_ids: admit.stop_ids,
            images,
        },
    ))
}

/// Rank 0's link: the driver's queue in, the control broadcast and the stream events out.
struct HubLink {
    hub: Tp4ControlHub,
    jobs: mpsc::Receiver<Pending>,
    /// Fired at the first tick: the epoch is open and HTTP may bind.
    ready: Option<mpsc::Sender<()>>,
    tokenizer: HfTokenizerEncoder,
    think_end: Option<u32>,
    metrics: Arc<Metrics>,
    /// The requests on the device, by sequence id.
    live: std::collections::BTreeMap<u64, Live>,
    /// Sequences whose stream ended early: the next tick cancels them.
    ending: Vec<u64>,
    next_seq: u64,
    /// The next tick, already broadcast by [`DeviceLink::ahead`].
    sent: Option<DeviceTick>,
}

impl HubLink {
    /// Takes `pending` onto the device as the next sequence, unless its client already left.
    fn admit(&mut self, pending: Pending, admit: &mut Vec<(u64, DeviceJob)>) {
        self.metrics.request_admitted();
        if pending.events.is_closed() {
            // The client left while the request was queued.
            self.metrics.request_finished();
            return;
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        admit.push((seq, pending.job.clone()));
        self.live.insert(seq, Live::new(pending));
    }

    /// Decides a tick and broadcasts it: `idle` blocks for a request (None: the driver is gone
    /// and every worker was told to leave).
    fn decide(
        &mut self,
        idle: bool,
        room: usize,
        hold_prefill: bool,
    ) -> Result<Option<DeviceTick>, String> {
        if let Some(ready) = self.ready.take() {
            let _ = ready.send(());
        }
        let mut tick = DeviceTick {
            admit: Vec::new(),
            cancel: std::mem::take(&mut self.ending),
            hold_prefill,
        };
        for (&seq, live) in &self.live {
            if live.pending.events.is_closed() && !tick.cancel.contains(&seq) {
                tick.cancel.push(seq);
            }
        }
        let mut ids = 0usize;
        while tick.admit.len() < room && ids < TP4_TICK_PROMPT_IDS {
            let block = idle && tick.admit.is_empty() && tick.cancel.is_empty();
            let pending = if block {
                match self.jobs.recv() {
                    Ok(pending) => pending,
                    Err(_) => {
                        // The driver is gone: every worker leaves its loop.
                        let _ = self.hub.broadcast(&Tp4ControlFrame::End);
                        return Ok(None);
                    }
                }
            } else {
                match self.jobs.try_recv() {
                    Ok(pending) => pending,
                    Err(_) => break,
                }
            };
            ids += pending.job.prompt.len();
            self.admit(pending, &mut tick.admit);
        }
        // The canvases go to the tower rank ahead of the tick that admits their requests.
        let tower = crate::device_link::tower_rank();
        if tower != 0 {
            for (seq, job) in &tick.admit {
                for (index, image) in job.images.iter().enumerate() {
                    let index = u32::try_from(index).unwrap_or(u32::MAX);
                    self.hub
                        .send_pixels(tower, *seq, index, &image.pixels)
                        .map_err(|error| format!("control DevicePixels: {error}"))?;
                }
            }
        }
        self.hub
            .broadcast(&Tp4ControlFrame::DeviceTick {
                admit: tick
                    .admit
                    .iter()
                    .map(|(seq, job)| admit_frame(*seq, job))
                    .collect(),
                cancel: tick.cancel.clone(),
                hold_prefill,
            })
            .map_err(|error| format!("control DeviceTick: {error}"))?;
        Ok(Some(tick))
    }
}

impl DeviceLink for HubLink {
    fn tick(
        &mut self,
        idle: bool,
        room: usize,
        hold_prefill: bool,
    ) -> Result<Option<DeviceTick>, String> {
        match self.sent.take() {
            Some(tick) => Ok(Some(tick)),
            None => self.decide(idle, room, hold_prefill),
        }
    }

    fn ahead(&mut self, room: usize, hold_prefill: bool) -> Result<(), String> {
        if self.sent.is_none() {
            self.sent = self.decide(false, room, hold_prefill)?;
        }
        Ok(())
    }

    fn emit(&mut self, seq: u64, tokens: &[u32], last: bool) {
        let Some(live) = self.live.get_mut(&seq) else {
            return;
        };
        if !live.emit(tokens, &self.tokenizer, self.think_end, &self.metrics)
            && !last
            && !self.ending.contains(&seq)
        {
            self.ending.push(seq);
        }
    }

    fn finish(&mut self, seq: u64, outcome: Result<&DeviceOutcome, &str>) -> Result<(), String> {
        if let Some(live) = self.live.remove(&seq) {
            live.finish(outcome, &self.metrics);
        }
        self.ending.retain(|&ending| ending != seq);
        Ok(())
    }
}

/// A worker's link: the same decisions, read from rank 0.
struct WorkerLink {
    control: Tp4ControlWorker,
    /// The canvases that came ahead of their admissions (the tower rank's), by (sequence, image).
    pixels: std::collections::HashMap<(u64, u32), Arc<[u8]>>,
}

impl DeviceLink for WorkerLink {
    fn tick(
        &mut self,
        _idle: bool,
        _room: usize,
        _hold_prefill: bool,
    ) -> Result<Option<DeviceTick>, String> {
        let rank = self.control.rank;
        let mut frame = self.control.next_frame();
        while let Ok(Tp4ControlFrame::DevicePixels { seq, image, bytes }) = frame {
            let pixels = self
                .control
                .next_pixels(bytes)
                .map_err(|error| format!("rank {rank} control pixels: {error}"))?;
            self.pixels.insert((seq, image), pixels.into());
            frame = self.control.next_frame();
        }
        match frame {
            Ok(Tp4ControlFrame::DeviceTick {
                admit,
                cancel,
                hold_prefill,
            }) => {
                let admit = admit
                    .into_iter()
                    .map(|admit| admitted_job(admit, &mut self.pixels))
                    .collect::<Result<Vec<_>, _>>()?;
                // A canvas is for the tick that admits its request, never a later one.
                if !self.pixels.is_empty() {
                    return Err(format!(
                        "rank {rank}: {} canvases arrived for requests the tick does not admit",
                        self.pixels.len()
                    ));
                }
                Ok(Some(DeviceTick {
                    admit,
                    cancel,
                    hold_prefill,
                }))
            }
            Ok(Tp4ControlFrame::End) => {
                LEADER_ENDED.store(true, Ordering::SeqCst);
                Ok(None)
            }
            Ok(Tp4ControlFrame::Error { code, message }) => {
                Err(format!("rank 0 reported {code}: {message}"))
            }
            Ok(other) => Err(format!(
                "rank {} expected DeviceTick or End, got {}",
                self.control.rank,
                other.kind()
            )),
            Err(error) => Err(format!("rank {} control: {error}", self.control.rank)),
        }
    }

    fn emit(&mut self, _seq: u64, _tokens: &[u32], _last: bool) {}

    fn finish(&mut self, _seq: u64, _outcome: Result<&DeviceOutcome, &str>) -> Result<(), String> {
        Ok(())
    }
}

/// Exits rank 0 when a worker's control socket closes: a dead rank must fail the server, not
/// leave its requests waiting in a collective.
fn watch_workers(hub: &Tp4ControlHub) -> Result<(), String> {
    for (rank, stream) in hub.streams.iter().enumerate() {
        let Some(stream) = stream else { continue };
        let mut probe = stream
            .try_clone()
            .map_err(|error| format!("rank {rank} control watch: {error}"))?;
        std::thread::Builder::new()
            .name(format!("sparkflash-watch-r{rank}"))
            .spawn(move || {
                // Workers send nothing after Hello: any read that is not a timeout is the end.
                let mut byte = [0u8; 1];
                loop {
                    match std::io::Read::read(&mut probe, &mut byte) {
                        Err(error)
                            if matches!(
                                error.kind(),
                                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                            ) => {}
                        Ok(0) | Err(_) => {
                            eprintln!(
                                "sparkflash serve: rank {rank} closed its control channel; the group cannot serve, exiting"
                            );
                            exit_now(1);
                        }
                        Ok(_) => {}
                    }
                }
            })
            .map_err(|error| format!("rank {rank} control watch: {error}"))?;
    }
    Ok(())
}

/// Exits a worker when rank 0's control socket closes, even while it waits in a collective.
fn watch_leader(stream: &TcpStream, rank: usize) -> Result<(), String> {
    let probe = stream
        .try_clone()
        .map_err(|error| format!("rank {rank} control watch: {error}"))?;
    std::thread::Builder::new()
        .name("sparkflash-watch-r0".to_owned())
        .spawn(move || {
            let mut byte = [0u8; 1];
            loop {
                match probe.peek(&mut byte) {
                    // Frames are waiting for the device thread: look again once it has read them.
                    Ok(n) if n > 0 => std::thread::sleep(Duration::from_millis(200)),
                    Err(error)
                        if matches!(
                            error.kind(),
                            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                        ) => {}
                    _ => {
                        if LEADER_ENDED.load(Ordering::SeqCst) {
                            return;
                        }
                        eprintln!(
                            "sparkflash serve: rank {rank}: rank 0 closed the control channel, exiting"
                        );
                        exit_now(1);
                    }
                }
            }
        })
        .map_err(|error| format!("rank {rank} control watch: {error}"))?;
    Ok(())
}

/// The `device-chain` arguments of this rank's serving epoch: the harness's DSpark recipe with a
/// capacity of `positions` and a one-token placeholder prompt (each request brings its own).
fn epoch_args(
    options: &ServeOptions,
    tp4: &Tp4ServeOptions,
    device: &DeviceServeOptions,
) -> Result<Vec<String>, String> {
    let checkpoint = options.checkpoint.as_ref().ok_or(
        "--tp4-device requires --checkpoint DIR: the four ranks serve one released sharded checkpoint",
    )?;
    let positions = device.positions();
    let cubins = device.cubins.display().to_string();
    let peers = tp4
        .peers
        .iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    // The harness's prompt is `ids.len() - positions + 1` tokens long: `positions` ids make a
    // one-token prompt (id 0, the BOS), which the serving loop never runs.
    let ids = vec!["0"; positions].join(",");
    // The Engram n-grams hash the compressed ids of the tokenizer the requests are encoded with.
    let tokenizer = match options.tokenizer.clone() {
        Some(path) => path,
        None => default_tokenizer()?,
    };
    let args: Vec<String> = [
        "--run-stage",
        "tp4-chain",
        "--checkpoint",
        &checkpoint.display().to_string(),
        "--engram-tokenizer",
        &tokenizer.display().to_string(),
        "--tp-rank",
        &tp4.rank.to_string(),
        "--tp-world-size",
        &tp4.world_size.to_string(),
        "--evidence-dir",
        &device.evidence_dir.display().to_string(),
        "--upload",
        "--comm-peers",
        &peers,
        "--comm-base-port",
        &tp4.base_port.to_string(),
        "--comm-timeout",
        "600",
        "--comm-connect-timeout",
        "300",
        "--decode-steps",
        &positions.to_string(),
        "--cubin-dir",
        &cubins,
        "--loop",
        "--forward",
        "--forward-preflight",
        "--kernel-cubin-dir",
        &cubins,
        "--token-ids",
        &ids,
        "--forward-lean",
    ]
    .iter()
    .map(|arg| (*arg).to_owned())
    .collect();
    Ok(args)
}

/// Runs one rank of the device serving group until its epoch ends.
pub(super) fn run(
    options: &ServeOptions,
    tp4: &Tp4ServeOptions,
    device: &DeviceServeOptions,
) -> ExitCode {
    // SAFETY: no thread has been started yet, so nothing reads the environment concurrently. The
    // epoch takes its DSpark branch from SPARKFLASH_DSPARK, and the router bounds max_tokens by
    // the context SPARK_MAX_SEQ_LEN names.
    unsafe {
        std::env::set_var("SPARKFLASH_DSPARK", "1");
        std::env::set_var(MAX_SEQ_LEN_ENV, device.max_model_len.to_string());
    }
    let args = match epoch_args(options, tp4, device) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("sparkflash serve: {message}");
            return ExitCode::from(EXIT_REFUSED);
        }
    };
    let control = SocketAddr::new(tp4.peers[0], tp4.control_port);
    if tp4.rank != 0 {
        let started = (|| -> Result<(), String> {
            let rank = u32::try_from(tp4.rank).map_err(|_| "rank does not fit in u32")?;
            let mut worker =
                Tp4ControlWorker::connect(control, rank, tp4.world_size, TP4_CONTROL_TIMEOUT)
                    .map_err(|error| format!("control connect: {error}"))?;
            worker
                .handshake()
                .map_err(|error| format!("control handshake: {error}"))?;
            // Idle time between requests is not a failure; a closed socket is.
            worker
                .stream
                .set_read_timeout(None)
                .map_err(|error| format!("control: {error}"))?;
            watch_leader(&worker.stream, tp4.rank)?;
            crate::device_link::install(Box::new(WorkerLink {
                control: worker,
                pixels: std::collections::HashMap::new(),
            }));
            Ok(())
        })();
        if let Err(message) = started {
            eprintln!("sparkflash serve: rank {}: {message}", tp4.rank);
            return ExitCode::from(EXIT_REFUSED);
        }
        return crate::device_chain::run(args.into_iter());
    }
    match start_leader(options, tp4, device, control) {
        Ok(()) => {}
        Err(message) => {
            eprintln!("sparkflash serve: {message}");
            return ExitCode::from(EXIT_REFUSED);
        }
    }
    let code = crate::device_chain::run(args.into_iter());
    eprintln!("sparkflash serve: the device loop ended; the server cannot answer, exiting");
    code
}

/// Rank 0 before its epoch: the tokenizer, the workers' handshake, the link, and the HTTP thread
/// that binds once the epoch reaches its serving loop.
fn start_leader(
    options: &ServeOptions,
    tp4: &Tp4ServeOptions,
    device: &DeviceServeOptions,
    control: SocketAddr,
) -> Result<(), String> {
    let tokenizer_path = match options.tokenizer.clone() {
        Some(path) => path,
        None => default_tokenizer()?,
    };
    let tokenizer = load_tokenizer(&tokenizer_path)?;
    let seam = Arc::new(TokenizerSeam::new(load_tokenizer(&tokenizer_path)?));
    let id = |name: &str| tokenizer.tokenizer().token_to_id(name);
    let think_start = id(THINK_START);
    let think_end = id(THINK_END);
    let end_of_sequence = id(END_OF_SEQUENCE).ok_or_else(|| {
        format!(
            "--tokenizer {}: no {END_OF_SEQUENCE} token to stop on",
            tokenizer_path.display()
        )
    })?;
    println!(
        "sparkflash serve: TP=4 device rank 0/{}: waiting for the workers on control {control}",
        tp4.world_size
    );
    let hub = Tp4ControlHub::start(control, tp4.world_size, TP4_CONTROL_TIMEOUT)
        .map_err(|error| format!("rank 0 control channel: {error}"))?;
    watch_workers(&hub)?;
    let (jobs, queued) = mpsc::sync_channel(DEVICE_JOB_QUEUE);
    let (ready, readied) = mpsc::channel();
    let metrics = Arc::new(Metrics::new());
    crate::device_link::install(Box::new(HubLink {
        hub,
        jobs: queued,
        ready: Some(ready),
        tokenizer,
        think_end,
        metrics: Arc::clone(&metrics),
        live: std::collections::BTreeMap::new(),
        ending: Vec::new(),
        next_seq: 0,
        sent: None,
    }));
    let mut cluster = tp4_cluster_snapshot();
    cluster.fabric_mode = DEVICE_FABRIC.to_owned();
    cluster.transport.fabric_mode = DEVICE_FABRIC.to_owned();
    let driver = DeviceDriver {
        jobs,
        max_tokens: options.max_tokens,
        max_model_len: device.max_model_len,
        vocab_size: ModelConfig::v41_flash_cached().vocab_size,
        think_start,
        end_of_sequence,
        cluster,
        metrics: Arc::clone(&metrics),
        seeds: RandomState::new(),
        draws: AtomicU64::new(0),
        vision: crate::device_link::vision_enabled(&device.cubins),
    };
    let state = AppState::new(Arc::new(driver))
        .with_metrics(metrics)
        .with_max_logprobs(options.max_logprobs)
        .with_tokenizer(seam)
        .with_api_key(options.api_key.clone())
        .with_cors(options.cors.clone());
    let bind = options.bind;
    let boot = Instant::now();
    std::thread::Builder::new()
        .name("sparkflash-http".to_owned())
        .spawn(move || {
            // The epoch fires this at its first request wait; it never fires if the epoch failed.
            if readied.recv().is_err() {
                return;
            }
            println!(
                "sparkflash serve: device_timeline phase=ready epoch_ms={} boot_s={:.1}",
                epoch_millis(),
                boot.elapsed().as_secs_f64()
            );
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    eprintln!("sparkflash serve: cannot start the async runtime: {error}");
                    exit_now(1);
                }
            };
            let router = sparkflash_api::server::router(state.clone());
            let code = runtime.block_on(bind_and_serve(router, state, bind));
            // The HTTP side is the server: when it ends (a drain, or a bind failure), so does
            // the process, and the workers follow when the control channel closes.
            exit_now(i32::from(code != ExitCode::SUCCESS));
        })
        .map_err(|error| format!("cannot start the HTTP thread: {error}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_capacity_is_bounded_by_the_select_table() {
        let options = DeviceServeOptions::from_cli(Some("cubins".into()), None, None)
            .unwrap()
            .unwrap();
        assert_eq!(options.positions(), DEVICE_MAX_POSITIONS);
        assert!(
            DeviceServeOptions::from_cli(
                Some("cubins".into()),
                Some(DEFAULT_MAX_MODEL_LEN + 1),
                None
            )
            .is_err()
        );
        assert!(DeviceServeOptions::from_cli(Some("cubins".into()), Some(0), None).is_err());
        assert_eq!(DeviceServeOptions::from_cli(None, Some(5), None), Ok(None));
    }

    #[test]
    fn an_admission_round_trips_its_sampler() {
        let job = DeviceJob {
            prompt: vec![0, 5, 128_821],
            max_tokens: 64,
            temperature: 0.7,
            top_p: 0.95,
            top_k: 20,
            seed: u64::MAX - 3,
            stop_ids: vec![1],
            images: Vec::new(),
        };
        let frame = Tp4ControlFrame::DeviceTick {
            admit: vec![admit_frame(9, &job)],
            cancel: vec![4],
            hold_prefill: true,
        };
        let wire = serde_json::to_vec(&frame).unwrap();
        let Tp4ControlFrame::DeviceTick {
            admit,
            cancel,
            hold_prefill,
        } = serde_json::from_slice(&wire).unwrap()
        else {
            panic!("not a DeviceTick");
        };
        assert_eq!(cancel, vec![4]);
        assert!(hold_prefill);
        let [admit] = <[Tp4DeviceAdmit; 1]>::try_from(admit).unwrap();
        let mut pixels = std::collections::HashMap::new();
        assert_eq!(admitted_job(admit, &mut pixels), Ok((9, job)));
    }

    /// A 42 x 84 canvas: one token row of two tokens, a span of 1 + 3 + 1 positions.
    fn image(start: usize) -> VisionImage {
        let grid = Geometry::V41.grid(42, 84);
        VisionImage {
            span: ImageSpan {
                start,
                grid,
                key: [0xa5; 32],
            },
            pixels: vec![7u8; 42 * 84 * 3].into(),
        }
    }

    #[test]
    fn an_image_admission_round_trips_its_spans_and_takes_its_canvas() {
        let mut prompt = vec![0, 5];
        prompt.extend([IMAGE_TOKEN_ID; 5]);
        prompt.push(9);
        let job = DeviceJob {
            prompt,
            max_tokens: 8,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 1,
            stop_ids: vec![1],
            images: vec![image(2)],
        };
        assert_eq!(bad_spans(&job.prompt, &job.images), None);
        let wire = serde_json::to_vec(&admit_frame(3, &job)).unwrap();
        let admit: Tp4DeviceAdmit = serde_json::from_slice(&wire).unwrap();
        // The tower rank: its canvas came ahead of the admission.
        let mut pixels = std::collections::HashMap::new();
        pixels.insert((3, 0), Arc::clone(&job.images[0].pixels));
        assert_eq!(
            admitted_job(admit.clone(), &mut pixels),
            Ok((3, job.clone()))
        );
        assert!(pixels.is_empty());
        // Any other rank: the span without the canvas.
        let (_, other) = admitted_job(admit, &mut pixels).unwrap();
        assert_eq!(other.images[0].span, job.images[0].span);
        assert!(other.images[0].pixels.is_empty());
    }

    #[test]
    fn spans_that_are_not_the_routers_are_refused() {
        let mut prompt = vec![0, 5];
        prompt.extend([IMAGE_TOKEN_ID; 5]);
        prompt.push(9);
        assert!(bad_spans(&prompt, &[image(4)]).is_some(), "past the prompt");
        assert!(bad_spans(&prompt, &[image(1)]).is_some(), "a text position");
        assert!(bad_spans(&prompt, &[]).is_some(), "a stray placeholder");
        let mut short = image(2);
        short.pixels = vec![0u8; 3].into();
        assert!(bad_spans(&prompt, &[short]).is_some(), "the canvas size");
        let mut two = prompt.clone();
        two.extend([IMAGE_TOKEN_ID; 5]);
        assert_eq!(bad_spans(&two, &[image(2), image(8)]), None);
        assert!(
            bad_spans(&two, &[image(8), image(2)]).is_some(),
            "out of order"
        );
    }
}

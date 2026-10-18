//! `sparkflash serve`: the HTTP entry point over the reference engine.
//!
//! `sparkflash_api::server::router` has always been the whole HTTP surface of this project: it
//! owns the OpenAI-compatible routes, the SSE framing and the
//! [`sparkflash_api::engine::InferenceEngine`] seam. What was missing was something on the other
//! side of that seam. This module is that thing and nothing more: it builds a
//! [`sparkflash_engine::forward::InferenceEngine`] over either the synthetic plan or a real
//! checkpoint, presents it to the router through a local driver, and binds the router to a socket.
//!
//! `--engine device` selects the M1 `sparkflash-device` skeleton instead: a `DeviceEngine` over
//! the same reference engine, with a fake session, `vision: false`, non-greedy sampling refused
//! and no device execution (design `docs/build/device-forward-chain.md` 1.3, 7 M1). It is not
//! the default: `--engine reference` is, and that path is unchanged. The first real device
//! session is M2.
//!
//! # What it does not claim
//!
//! This is the *reference* engine: one sequence at a time, `f32`, on the CPU, with every weight
//! decoded through [`sparkflash_engine::linalg::decode_by_name`] under an element limit. It is not
//! a performance path and this module states no throughput figure for it. The numbers it reports
//! are the ones it measures for one request over that request's own wall clock, and every field it
//! does not measure is left at the zero it actually is, with the reason written beside it. There is
//! no cluster, no rank and no rail behind this process, so nothing here reports one.
//!
//! # The engine is `Send + Sync` (since !60), and it still has one owner thread
//!
//! The api crate's seam requires `Send + Sync`. Until !60 (`f72738b`, "engine: parallel reference
//! pass") the reference engine was neither: it held its weights as `Box<dyn WeightSource>`, and
//! `WeightSource` then declared no `Send` bound, so the box erased the concrete source's
//! thread-safety. That is no longer true - `sparkflash_engine::weights::WeightSource` is declared
//! `Send + Sync` so the MoE block can read independent experts concurrently, and every other field
//! of `sparkflash_engine::forward::InferenceEngine` (`ModelConfig`, `TensorPlan`, `usize`,
//! `Sampling`) is thread-safe - so the engine is `Send + Sync` as a type. The assertion below
//! pins that, and the earlier revision of these docs said the opposite.
//!
//! The driver still holds the engine on one dedicated thread, but thread-safety is no longer the
//! reason. The reason is the engine's shape: the reference pass runs one sequence at a time, and a
//! request applies its own sampler by consuming the engine (`with_sampling`), so the engine wants a
//! single mutable owner rather than a shared lock. Building one engine per request, the other way
//! to get `Send + Sync`, would re-open and re-verify a checkpoint for every request. The channel
//! handoff keeps the driver weight-free and the engine on the thread that built it; no `unsafe` is
//! involved.
//!
//! The consequence is that requests are serialised. That is the reference engine's own shape: one
//! sequence, one chunk at a time, no batching. A request is one job on that thread.
//!
//! # Failures are events, not silence
//!
//! A request that fails after `Started` ends with [`sparkflash_api::engine::EngineEvent::Error`]
//! rather than a stream that simply stops, so one bad request cannot take the server down and the
//! HTTP layer always has a terminal event to report. Every forward module is implemented, so a
//! request is driven through the whole CPU reference pass; whatever the pass reports - a completed
//! answer or an error from deep inside a layer - arrives as that request's terminal event.
//!
//! An engine *panic* is caught per request as well, but only where unwinding is enabled. The
//! workspace release profile sets `panic = "abort"`, and Cargo refuses to override `panic` for a
//! single package ("`panic` may not be specified in a `package` profile"), so the guard cannot be
//! made to cover this binary alone: a release `sparkflash` aborts the process on an engine panic by
//! design. Every dev and test build unwinds, and that is the build the per-request panic test runs
//! in. Nothing here has been run against a real sharded checkpoint, so nothing here claims that it
//! has.

use std::net::SocketAddr;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use futures::StreamExt;
use futures::stream::{self, BoxStream};
use sparkflash_api::AppState;
use sparkflash_api::encoding::{HfTokenizerEncoder, load_hf_tokenizer, reference_tokenizer_path};
use sparkflash_api::engine::InferenceEngine as ApiInferenceEngine;
use sparkflash_api::engine::{
    Capabilities, ClusterStatsSnapshot, EngineEvent, EngineRequest, FinishReason, MODEL_ID,
    MODEL_MAX_OUTPUT_TOKENS, MissingCapability, default_engram_layers, empty_kv_snapshot,
    max_seq_len,
};
use sparkflash_api::telemetry_types::{
    CompletionTokensDetails, DSparkStepStats, DeepSeekSparkRequestStats, OpenAiUsageWithCache,
    PromptTokensDetails, RequestTimingStats, RoceTransportStats, finite_or_zero,
};
use sparkflash_device::{DeviceEngine, FakeSession};
use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::{InferenceEngine, Sampling, SequenceState};
use sparkflash_engine::linalg::{CHECKPOINT_TENSOR_LIMIT, REFERENCE_TENSOR_LIMIT};
use sparkflash_engine::loader::ReferenceWeights;
use sparkflash_engine::synthetic::SyntheticWeights;
use sparkflash_engine::tensor::TensorPlan;
#[cfg(all(test, panic = "unwind"))]
use sparkflash_engine::tensor::TensorSpec;
#[cfg(all(test, panic = "unwind"))]
use sparkflash_engine::weights::WeightError;
use sparkflash_engine::weights::WeightSource;
use sparkflash_kv::PrefixMatchPlan;

/// The index file a sharded safetensors checkpoint is described by.
const CHECKPOINT_INDEX_NAME: &str = "model.safetensors.index.json";

/// The model configuration a checkpoint directory has to carry.
const CHECKPOINT_CONFIG_NAME: &str = "config.json";

/// The elements in the released `embed.weight` and `head.weight`: 129,280 x 5,120.
///
/// The two element caps' relation to the released checkpoint is a compile-time fact, not a note in
/// a comment: [CHECKPOINT_TENSOR_LIMIT] has to admit this tensor and [REFERENCE_TENSOR_LIMIT] has
/// to refuse it, or `default_element_limit` is choosing between two wrong answers. The literal is
/// the released figure; `the_released_embedding_size_matches_the_released_configuration` derives
/// it from [ModelConfig::v41_flash] so the two cannot drift.
const RELEASED_EMBEDDING_ELEMENTS: usize = 661_913_600;

const _: () = {
    assert!(CHECKPOINT_TENSOR_LIMIT > RELEASED_EMBEDDING_ELEMENTS);
    assert!(REFERENCE_TENSOR_LIMIT < RELEASED_EMBEDDING_ELEMENTS);
};

/// How many replies one request may have in flight before the engine thread blocks.
///
/// This is the backpressure: a client that stops reading its response stops the engine thread from
/// running further ahead than this many events.
const REPLY_BUFFER: usize = 32;

/// How many requests may wait for the engine thread before a request is refused.
const JOB_QUEUE: usize = 64;

/// The seed the synthetic weight source is generated from.
///
/// It is fixed rather than random so that two runs of `--synthetic` produce the same tokens: a
/// reproducible reference path is the whole point of the synthetic one, and the flag set carries no
/// seed knob because there is nothing to vary.
const SYNTHETIC_WEIGHT_SEED: u64 = 0;

/// The name the engine thread runs under, so a panic names it.
const ENGINE_THREAD_NAME: &str = "sparkflash-engine";

/// Pins the thread-safety fact the module docs state.
///
/// `WeightSource` gained `Send + Sync` in !60 (`f72738b`), so the boxed source no longer erases
/// the concrete source's thread-safety and the reference engine is `Send + Sync` as a type. This is
/// a compile-time assertion, not a test: if a future field or trait bound breaks the claim, the
/// crate stops compiling and the docs above have to change with it.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<sparkflash_engine::forward::InferenceEngine>();
};

/// Which serving engine implementation the process runs.
///
/// The reference engine is the existing CPU path and the default. `device` is the M1
/// `sparkflash-device` skeleton: a fake session that delegates `generate` back to the reference
/// engine, refuses images through the typed missing-capability path, refuses non-greedy sampling,
/// and claims no device execution (design `docs/build/device-forward-chain.md` 1.3, 7 M1).
/// The first real device session is M2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum EngineChoice {
    /// The existing CPU reference engine: unchanged serving behaviour.
    Reference,
    /// The M1 device skeleton: fake session, no device execution, no device claim.
    Device,
}

impl EngineChoice {
    /// The stable spelling used by the flag and the startup line.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Device => "device",
        }
    }
}

/// What this process can honestly say about the fabric it is running on: nothing.
const NO_FABRIC: &str = "none (reference serve: one process, no RoCE)";

/// The error code for a request the engine refused or failed.
const ENGINE_ERROR_CODE: &str = "engine_error";

/// The error code for a stream whose engine thread stopped before it answered.
const ENGINE_GONE_CODE: &str = "engine_gone";

/// The error code for a request refused before any engine work, such as an image request.
const REFUSED_CODE: &str = "refused";

/// The exit code for a refusal: bad arguments, an unreadable tokenizer, a directory that is not a
/// checkpoint. A runtime failure such as a bind error exits 1 instead.
const EXIT_REFUSED: u8 = 2;

/// Everything `sparkflash serve` needs once its arguments are parsed.
#[derive(Debug, Clone)]
pub struct ServeOptions {
    /// The socket to bind. The default is loopback: a reference server binds where an operator can
    /// reach it and nowhere else, and a wider bind is an explicit choice.
    pub bind: SocketAddr,
    /// Serve the synthetic tiny plan, generated from a seed, with no checkpoint on disk.
    pub synthetic: bool,
    /// Serve a real sharded checkpoint: a directory holding `config.json` and the safetensors
    /// index, or the index file itself.
    pub checkpoint: Option<PathBuf>,
    /// The `tokenizer.json` to detokenise with.
    pub tokenizer: PathBuf,
    /// The ceiling on tokens generated for one request. A request asking for more is clamped to it.
    pub max_tokens: usize,
    /// The element cap the reference pass applies to one tensor. Above it, a tensor is refused
    /// rather than materialised.
    ///
    /// Resolved at parse time: an explicit `--element-limit`, else [CHECKPOINT_TENSOR_LIMIT] for
    /// `--checkpoint` and [REFERENCE_TENSOR_LIMIT] for `--synthetic`.
    pub element_limit: usize,
    /// Which engine implementation to serve. The default is the existing reference engine; the
    /// device selection is the M1 skeleton behind a capability flag (design 7 M1).
    pub engine: EngineChoice,
}

impl ServeOptions {
    /// Parses the arguments that follow the `serve` subcommand.
    ///
    /// # Errors
    ///
    /// A message naming the flag that is wrong or missing: `--synthetic` together with
    /// `--checkpoint`, neither of them, or an argument clap cannot parse.
    pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let cli =
            ServeArgs::try_parse_from(std::iter::once("sparkflash serve".to_owned()).chain(args))
                .map_err(|error| error.to_string())?;
        let tokenizer = cli.tokenizer.unwrap_or_else(reference_tokenizer_path);
        let checkpoint = cli.checkpoint;
        // The cap follows the source when the operator does not name one: a real checkpoint's
        // largest tensors are the embedding and the head at 661,913,600 elements each, and
        // [REFERENCE_TENSOR_LIMIT] would refuse them mid-request. This is the one place the choice
        // is made, and `default_element_limit` is where a test pins it.
        let element_limit = cli
            .element_limit
            .unwrap_or_else(|| default_element_limit(checkpoint.as_deref()));
        match (cli.synthetic, checkpoint) {
            (true, Some(_)) => Err(
                "--synthetic and --checkpoint cannot both be given: --synthetic serves the tiny synthetic plan and --checkpoint serves a real sharded checkpoint, so choose one"
                    .to_owned(),
            ),
            (false, None) => Err(no_weights_message()),
            (synthetic, checkpoint) => Ok(Self {
                bind: cli.bind,
                synthetic,
                checkpoint,
                tokenizer,
                max_tokens: cli.max_tokens,
                element_limit,
                engine: cli.engine,
            }),
        }
    }

    /// The weights recipe this run serves, resolved before any thread starts.
    ///
    /// # Errors
    ///
    /// A message naming `--checkpoint` and the file that is missing or unusable.
    fn recipe(&self) -> Result<Recipe, String> {
        match (self.synthetic, self.checkpoint.as_deref()) {
            (true, _) => Ok(Recipe {
                config: ModelConfig::tiny(),
                weights: Weights::Synthetic {
                    seed: SYNTHETIC_WEIGHT_SEED,
                },
                tokenizer: self.tokenizer.clone(),
            }),
            (false, Some(path)) => checkpoint_recipe(path, &self.tokenizer),
            (false, None) => Err(no_weights_message()),
        }
    }
}

/// The refusal a run with no weights at all gets.
///
/// A server with no weights would answer with text that is not model output, so this is a refusal
/// rather than a default: the operator says which weights to serve.
fn no_weights_message() -> String {
    "no weights to serve: pass --synthetic for the tiny synthetic plan, or --checkpoint DIR for a real checkpoint".to_owned()
}

/// The element cap a run gets when the operator does not name one.
///
/// A real checkpoint gets [CHECKPOINT_TENSOR_LIMIT]: its largest materialised tensors are
/// `embed.weight` and `head.weight` at 661,913,600 elements, so [REFERENCE_TENSOR_LIMIT] would
/// refuse one of them in the middle of the first request. The tiny synthetic plan keeps the
/// reference guard, which is what makes that guard worth having. `linalg.rs` says no
/// real-checkpoint entry point may drift back to the reference guard; this is the serving entry
/// point that had, and this function is where the choice is made.
fn default_element_limit(checkpoint: Option<&Path>) -> usize {
    match checkpoint {
        Some(_) => CHECKPOINT_TENSOR_LIMIT,
        None => REFERENCE_TENSOR_LIMIT,
    }
}

/// The `serve` command line as clap sees it.
#[derive(Debug, Parser)]
#[command(
    name = "sparkflash serve",
    version,
    about = "Serve the OpenAI-compatible HTTP API over the reference engine",
    long_about = "Serves the HTTP API from the reference forward pass. Exactly one weight source is required: --synthetic generates the tiny reference plan from a seed and needs no checkpoint, while --checkpoint reads a real sharded checkpoint. The default run is read-only apart from what serving implies: no hardware state is changed and nothing is written to the checkpoint. --engine device selects the M1 sparkflash-device skeleton instead: a fake session that delegates to the reference engine, refuses images and non-greedy sampling, and claims no device execution; the default is --engine reference and is unchanged."
)]
struct ServeArgs {
    /// The address to bind.
    #[arg(long, default_value = "127.0.0.1:8000")]
    bind: SocketAddr,
    /// Serve the tiny synthetic plan, generated from a fixed seed, with no checkpoint on disk.
    #[arg(long)]
    synthetic: bool,
    /// Serve a real sharded checkpoint: its directory, or its safetensors index file.
    #[arg(long, value_name = "DIR")]
    checkpoint: Option<PathBuf>,
    /// The tokenizer.json to detokenise with; defaults to the vendored reference tokenizer.
    ///
    /// The default is the reference tokenizer that ships with this repository, which is what makes
    /// --synthetic self-contained. A real deployment passes the model's own tokenizer file, because
    /// detokenising with the wrong vocabulary produces text that is not the model's output.
    #[arg(long, value_name = "FILE")]
    tokenizer: Option<PathBuf>,
    /// The ceiling on tokens generated for one request.
    #[arg(long, value_name = "N", default_value_t = MODEL_MAX_OUTPUT_TOKENS)]
    max_tokens: usize,
    /// The element cap the reference pass applies to one tensor; the default follows the source.
    ///
    /// `--synthetic` keeps the reference guard, [REFERENCE_TENSOR_LIMIT], which the tiny plan fits
    /// inside; `--checkpoint` gets [CHECKPOINT_TENSOR_LIMIT], because the released `embed.weight`
    /// and `head.weight` hold 661,913,600 elements each and the reference guard would refuse one
    /// of them at the first request. An explicit flag overrides both.
    #[arg(
        long,
        value_name = "N",
        help = "Element cap for one tensor; defaults from the weight source (the 32 Mi reference guard for --synthetic, the checkpoint cap for --checkpoint)"
    )]
    element_limit: Option<usize>,
    /// Which engine to serve: the default `reference` CPU path, or the M1 `device` skeleton.
    ///
    /// `device` wraps the reference engine in `sparkflash-device`'s M1 `DeviceEngine`: a fake
    /// session, no kernel launch, no device execution, `vision: false` and non-greedy sampling
    /// refused. The first real device session is M2; this flag is the capability flag that keeps
    /// the existing serve path unchanged by default.
    #[arg(long, value_enum, default_value = "reference")]
    engine: EngineChoice,
}

/// What the engine thread builds: a configuration, where its weights come from, and the tokenizer.
#[derive(Debug, Clone)]
struct Recipe {
    config: ModelConfig,
    weights: Weights,
    tokenizer: PathBuf,
}

/// Where the weights come from.
#[derive(Debug, Clone)]
enum Weights {
    /// A seeded generator over the plan, with no checkpoint on disk.
    ///
    /// The configuration this runs is the tiny reference one: the released configuration's tensors
    /// are hundreds of times above the reference element limit, and the synthetic source refusing
    /// them is the safety property that keeps a released run from trying to allocate the model.
    Synthetic {
        /// The seed every tensor is generated from.
        seed: u64,
    },
    /// A real sharded checkpoint, already resolved to its directory and index.
    Checkpoint {
        /// The directory the shards live in.
        dir: PathBuf,
        /// The index file describing them.
        index: PathBuf,
    },
    /// Test-only: a source that panics once on one named tensor, so the per-request panic guard can
    /// be exercised. Compiled only into test builds that unwind; a shipped binary has the two
    /// variants above and nothing else.
    #[cfg(all(test, panic = "unwind"))]
    Panicking {
        /// The seed every other tensor is generated from.
        seed: u64,
        /// The tensor whose first read panics.
        name: &'static str,
    },
}

/// Resolves `--checkpoint` into a recipe, reading the configuration the checkpoint carries.
///
/// # Errors
///
/// A message naming `--checkpoint` and the file that is missing or unusable.
fn checkpoint_recipe(path: &Path, tokenizer: &Path) -> Result<Recipe, String> {
    let (dir, index) = if path.is_dir() {
        (path.to_path_buf(), path.join(CHECKPOINT_INDEX_NAME))
    } else {
        (
            path.parent().unwrap_or(Path::new(".")).to_path_buf(),
            path.to_path_buf(),
        )
    };
    if !index.is_file() {
        return Err(format!(
            "--checkpoint {}: no {CHECKPOINT_INDEX_NAME} to read, so this is not a sharded checkpoint directory",
            path.display()
        ));
    }
    let config_path = dir.join(CHECKPOINT_CONFIG_NAME);
    let text = std::fs::read_to_string(&config_path).map_err(|error| {
        format!(
            "--checkpoint {}: cannot read {}: {error}",
            dir.display(),
            config_path.display()
        )
    })?;
    let config = ModelConfig::from_json(&text).map_err(|error| {
        format!(
            "--checkpoint {}: {} is not a usable model configuration: {error}",
            dir.display(),
            config_path.display()
        )
    })?;
    Ok(Recipe {
        config,
        weights: Weights::Checkpoint { dir, index },
        tokenizer: tokenizer.to_path_buf(),
    })
}

/// Loads the detokeniser, naming the flag when it cannot.
///
/// # Errors
///
/// A message naming `--tokenizer` and the path.
fn load_tokenizer(path: &Path) -> Result<HfTokenizerEncoder, String> {
    load_hf_tokenizer(path).map_err(|error| format!("--tokenizer {}: {error}", path.display()))
}

/// The api crate's engine, backed by the engine thread.
///
/// It holds no engine state: a job in, replies out. That is what makes it `Send + Sync`; the
/// reference engine it drives is `Send + Sync` too (see the module docs) but still lives on one
/// dedicated thread by design, not because a boxed trait object forced it to.
pub struct ReferenceDriver {
    model: String,
    jobs: tokio::sync::mpsc::Sender<Job>,
    max_tokens: usize,
    cluster: ClusterStatsSnapshot,
}

impl ReferenceDriver {
    /// Starts the engine thread and waits for it to report that it built its engine and tokenizer.
    ///
    /// The handshake is what makes a refusal early and loud: the caller learns that the checkpoint,
    /// the configuration or the tokenizer is unusable before a socket exists, rather than on the
    /// first request. It is also why `start` is synchronous: it must not return until the engine
    /// thread has answered.
    ///
    /// # Errors
    ///
    /// A message naming what could not be built or loaded.
    fn start(recipe: &Recipe, element_limit: usize, max_tokens: usize) -> Result<Self, String> {
        let (jobs, queued) = tokio::sync::mpsc::channel(JOB_QUEUE);
        let (ready, reported) = std::sync::mpsc::channel();
        let owned = recipe.clone();
        std::thread::Builder::new()
            .name(ENGINE_THREAD_NAME.to_owned())
            .spawn(move || engine_thread(&owned, element_limit, queued, ready))
            .map_err(|error| format!("cannot start the engine thread: {error}"))?;
        match reported.recv() {
            Ok(Ok(())) => Ok(Self {
                model: MODEL_ID.to_owned(),
                jobs,
                max_tokens,
                cluster: cluster_snapshot(),
            }),
            Ok(Err(message)) => Err(message),
            Err(_) => Err("the engine thread stopped before it reported ready".to_owned()),
        }
    }
}

impl ApiInferenceEngine for ReferenceDriver {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn generate(&self, request: EngineRequest) -> BoxStream<'static, EngineEvent> {
        // The reference pass is text-only, and the image path is 263 vision/aligner tensors plus the
        // three delimiters that the tensor plan does not load (DECISIONS.md D18). Answering from the
        // text tokens alone would present an answer as if the image had been seen, so the request is
        // refused with the same typed missing-capability fact the seam reports (D27).
        if !request.images.is_empty() {
            let missing = MissingCapability::vision();
            return refused(&missing.to_string(), missing.code());
        }
        if request.prompt_tokens.is_empty() {
            return refused(
                "the request carries no prompt tokens, and the reference pass has nothing to prefill",
                REFUSED_CODE,
            );
        }
        let prompt_tokens = request.prompt_tokens.len();
        let max_tokens = request.max_tokens.min(self.max_tokens);
        let (replies, answers) = tokio::sync::mpsc::channel(REPLY_BUFFER);
        let job = Job {
            prompt_tokens: request.prompt_tokens,
            max_tokens,
            temperature: request.temperature,
            seed: request.seed.unwrap_or(0),
            replies,
        };
        if self.jobs.try_send(job).is_err() {
            return refused(
                "the engine queue is full: the reference engine runs one request at a time and the queue is bounded, so retry when the requests in flight have finished",
                REFUSED_CODE,
            );
        }
        // `cached_tokens` is zero because this process has no prefix cache at all: zero is the
        // measurement, not a miss that rounded down.
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
                                message: "the engine thread stopped before this request finished"
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

    /// The reference driver is text-only: it takes no image part, emits no tool call and has no
    /// reasoning channel, so it answers with the conservative set. That is the trait's default, and
    /// it relies on the trait's default `validate_request`, which refuses an image part for every
    /// engine that does not override it.
    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }
}

/// The state of one request's answer stream.
struct AnswerState {
    /// What the engine thread has sent so far.
    answers: tokio::sync::mpsc::Receiver<Reply>,
    /// Whether a terminal event has been emitted; the next poll ends the stream.
    done: bool,
}

/// A stream that is one refusal: no engine work happens behind it.
fn refused(message: &str, code: &str) -> BoxStream<'static, EngineEvent> {
    stream::iter([EngineEvent::Error {
        message: message.to_owned(),
        code: code.to_owned(),
    }])
    .boxed()
}

/// The terminal event for a request that produced a token count and timings.
fn finished_event(prompt_tokens: usize, summary: RunSummary) -> EngineEvent {
    let completion_tokens = summary.completion_tokens;
    EngineEvent::Finished {
        finish_reason: summary.finish_reason,
        usage: OpenAiUsageWithCache {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: PromptTokensDetails {
                // This process has no KV pool and no persistent tier, so nothing was served from a
                // cache; and an image request was refused before it got here, so no visual token
                // was counted. These zeros are the state of the world, not a missing measurement.
                cached_tokens: 0,
                uma_cached_tokens: 0,
                nvme_cached_tokens: 0,
                image_tokens: 0,
            },
            completion_tokens_details: CompletionTokensDetails {
                // The reference engine drafts tokens in its MTP stages and this driver never
                // verifies one, so nothing was proposed, accepted or rejected. A zero here means
                // "not attempted", which is exactly what happened.
                reasoning_tokens: 0,
                accepted_prediction_tokens: 0,
                rejected_prediction_tokens: 0,
            },
        },
        stats: Box::new(request_stats(prompt_tokens, summary.timing)),
    }
}

/// The per-request telemetry, with every unmeasured field left at zero and the reason beside it.
///
/// The filled fields are the token counts taken from the usage block and the three wall-clock
/// numbers in `timing`, all of which this process measured for this one request. Nothing here is an
/// estimate, a projection or another machine's number.
fn request_stats(prompt_tokens: usize, timing: RequestTiming) -> DeepSeekSparkRequestStats {
    DeepSeekSparkRequestStats {
        // The prompt length is known; every cache tier is zero because no cache exists in this
        // process, so no prefix was matched and nothing had to be replayed.
        kv_cache: PrefixMatchPlan {
            total_prompt_tokens: prompt_tokens,
            global_kv_cached_tokens: 0,
            uma_hit_tokens: 0,
            nvme_restored_tokens: 0,
            swa_tail_hit: false,
            encoder_swa_replay_tokens: 0,
            uncached_suffix_tokens: 0,
            decoder_swa_replay_tokens: 0,
        },
        kv_cache_hit_rate: 0.0,
        // The forward pass owns its Engram counters and the engine API exposes none of them, so
        // this process cannot report a non-zero one.
        engram_layers: default_engram_layers(),
        // One token at a time, with no draft verification: there is no speculative accounting.
        dspark_speculative: DSparkStepStats::default(),
        // This process speaks no RoCE and runs no collective.
        transport: RoceTransportStats::default(),
        timing: RequestTimingStats {
            ttft_ms: timing.ttft_ms,
            prefill_tok_per_s: timing.prefill_tok_per_s,
            decode_tok_per_s: timing.decode_tok_per_s,
            // No rank exists in this process and no UMA counter is read. The field means per-rank
            // coherent-memory headroom, which this process cannot see, so it stays zero rather than
            // carrying a host figure that means something else.
            min_rank_uma_free_gib: 0.0,
        },
    }
}

/// The cluster telemetry this process can actually see: none of it.
///
/// A reference serve is one process with one engine thread. It has no rank, no peer and no way to
/// read another node's memory, clocks or rails, so the node list is empty rather than holding this
/// host pretending to be a cluster.
fn cluster_snapshot() -> ClusterStatsSnapshot {
    ClusterStatsSnapshot {
        model: MODEL_ID.to_owned(),
        fabric_mode: NO_FABRIC.to_owned(),
        nodes: Vec::new(),
        kv: empty_kv_snapshot(),
        engram: default_engram_layers().to_vec(),
        transport: RoceTransportStats {
            fabric_mode: NO_FABRIC.to_owned(),
            ..RoceTransportStats::default()
        },
        dspark: DSparkStepStats::default(),
        timing: RequestTimingStats::default(),
        max_seq_len: max_seq_len(),
    }
}

/// One request, as the engine thread receives it.
struct Job {
    /// The tokens to prefill.
    prompt_tokens: Vec<u32>,
    /// The ceiling on generated tokens, already clamped by the driver.
    max_tokens: usize,
    /// The sampling temperature the request asked for.
    temperature: f32,
    /// The sampler seed, defaulting to zero when the request named none.
    seed: u64,
    /// Where the events go.
    replies: tokio::sync::mpsc::Sender<Reply>,
}

/// One element of the engine thread's answer.
enum Reply {
    /// A non-empty detokenised piece. The thread never sends an empty string.
    Delta(String),
    /// The request ended; the counts and the timings in here were measured.
    Done(Box<RunSummary>),
    /// The request failed. The string is written for an operator to read.
    Failed(String),
}

/// What one request produced.
struct RunSummary {
    /// How many tokens the request generated.
    completion_tokens: usize,
    /// Why generation stopped.
    finish_reason: FinishReason,
    /// The wall-clock numbers measured for this request.
    timing: RequestTiming,
}

/// The wall-clock numbers one request measured.
#[derive(Debug, Clone, Copy, Default)]
struct RequestTiming {
    /// Time to first token, in milliseconds.
    ttft_ms: f64,
    /// Prompt tokens over the prefill's wall clock.
    prefill_tok_per_s: f64,
    /// Tokens after the first over the wall clock that followed it.
    decode_tok_per_s: f64,
}

impl RequestTiming {
    /// Measures the timing block for one request.
    ///
    /// `ttft` is the time from the start of the request to the moment the engine produced its first
    /// token, so it carries the prefill and the first sampler step; it is `None`, and the reported
    /// time is zero, when the request generated no token at all, because there was no first token
    /// to time. The decode rate is then measured over what is left of the request's wall clock
    /// after that first token.
    ///
    /// A rate this cannot resolve - an interval the clock reports as zero - comes back as zero
    /// rather than as an infinity, so nothing on the wire is a number that is not a measurement.
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

/// A count over a wall clock, or zero when the interval is not a measurement.
fn rate(count: usize, seconds: f64) -> f64 {
    finite_or_zero(count as f64 / seconds)
}

/// The engine thread: build once, then run jobs one at a time until the queue closes.
///
/// The engine and the tokenizer are built here rather than on the caller's thread because the
/// engine must never leave the thread that built it, and the handshake tells the caller whether
/// that worked before anything is bound to a socket.
fn engine_thread(
    recipe: &Recipe,
    element_limit: usize,
    mut jobs: tokio::sync::mpsc::Receiver<Job>,
    ready: std::sync::mpsc::Sender<Result<(), String>>,
) {
    let built = build_engine(recipe, element_limit).and_then(|engine| {
        load_tokenizer(&recipe.tokenizer).map(|tokenizer| (Box::new(engine), tokenizer))
    });
    let (mut engine, tokenizer) = match built {
        Ok(pair) => pair,
        Err(message) => {
            let _ = ready.send(Err(message));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        // The caller gave up before the engine was ready; there is nobody to serve.
        return;
    }
    while let Some(job) = jobs.blocking_recv() {
        // The sampler is per request and `with_sampling` consumes the engine, so the value inside
        // the box is replaced around the same plan and the same weight source - no weight is
        // re-read and no checkpoint is re-opened by this.
        *engine = (*engine).with_sampling(Sampling {
            temperature: job.temperature,
            seed: job.seed,
        });
        // A panic ends this one request rather than the process - in a build that unwinds. The
        // workspace release profile's `panic = "abort"` makes this unreachable there, and Cargo does
        // not let a package override `panic`, so a release binary aborts by design; the module docs
        // say so, and the panic test below is compiled only where unwinding is on.
        let replies = job.replies.clone();
        let outcome = catch_unwind(AssertUnwindSafe(|| run_request(&engine, &tokenizer, job)));
        let failure = match outcome {
            Ok(Ok(())) => None,
            Ok(Err(message)) => Some(message),
            Err(payload) => Some(format!(
                "the reference engine panicked on this request: {}",
                panic_message(payload.as_ref())
            )),
        };
        if let Some(message) = failure {
            let _ = replies.blocking_send(Reply::Failed(message));
        }
    }
}

/// The text of a caught panic payload, for an operator.
///
/// Only reachable in a build that unwinds; the module docs say why a release binary never gets here.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_owned()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "a panic whose payload is not a string".to_owned()
    }
}

/// Builds the engine on the engine thread.
///
/// # Errors
///
/// A message naming the checkpoint or the configuration when the engine cannot be built.
fn build_engine(recipe: &Recipe, element_limit: usize) -> Result<InferenceEngine, String> {
    let plan = TensorPlan::language_model(&recipe.config);
    let source: Box<dyn WeightSource> = match &recipe.weights {
        Weights::Synthetic { seed } => Box::new(SyntheticWeights::from_plan(&plan, *seed)),
        Weights::Checkpoint { dir, index } => {
            // The reference pass decodes the *stored* forms: `wo_a` is F8E4M3 plus its own scale
            // grid, exactly as the plan names it. [ReferenceWeights] is the source that serves
            // those bytes and verifies the checkpoint first; the loader's `get` instead converts
            // `wo_a` to BF16 with the scale applied, which a F8E4M3 spec then reads as twice the
            // bytes it expects (linalg::decode_by_name -> quant::check_len). Serving through the
            // converted form failed on layer 0, and this is the seam that used to do it.
            let reference = ReferenceWeights::open(dir, index, &plan).map_err(|error| {
                format!(
                    "--checkpoint {}: the shards do not match the planned model: {error}",
                    dir.display()
                )
            })?;
            Box::new(reference)
        }
        #[cfg(all(test, panic = "unwind"))]
        Weights::Panicking { seed, name } => Box::new(PanickingSource {
            name,
            armed: std::sync::atomic::AtomicBool::new(true),
            inner: SyntheticWeights::from_plan(&plan, *seed),
        }),
    };
    InferenceEngine::new(recipe.config.clone(), source, element_limit)
        .map_err(|error| format!("cannot build the reference engine: {error}"))
}

/// Runs one request and sends its events, ending with exactly one terminal reply.
///
/// `Ok(())` means the terminal reply was sent. `Err(message)` means the request failed and the
/// caller still has to report it, which is what keeps the failure path in one place.
fn run_request(
    engine: &InferenceEngine,
    tokenizer: &HfTokenizerEncoder,
    job: Job,
) -> Result<(), String> {
    let request_started = Instant::now();
    // A sequence per request: the engine owns no per-sequence state, which is exactly the seam this
    // driver relies on.
    let mut state = SequenceState::new(engine.config());
    let prefill = engine
        .prefill(&mut state, &job.prompt_tokens)
        .map_err(|error| format!("prefill failed: {error}"))?;
    let prefill_elapsed = request_started.elapsed();
    // The stream decoder keeps the state an incomplete UTF-8 sequence needs, so a token that only
    // completes half a character is held back instead of being emitted as a replacement character
    // that the next token would then contradict.
    let mut decoder = tokenizer.tokenizer().decode_stream(false);
    let mut next = prefill.next_token;
    let mut completion_tokens = 0usize;
    let mut ttft: Option<Duration> = None;
    while completion_tokens < job.max_tokens {
        let Some(token) = next else { break };
        if ttft.is_none() {
            ttft = Some(request_started.elapsed());
        }
        if let Some(piece) = decoder
            .step(token)
            .map_err(|error| format!("the tokenizer could not decode token {token}: {error}"))?
            .filter(|piece| !piece.is_empty())
        {
            job.replies
                .blocking_send(Reply::Delta(piece))
                .map_err(|_| "the client stopped reading this response".to_owned())?;
        }
        completion_tokens += 1;
        next = engine
            .decode(&mut state, token)
            .map_err(|error| format!("decode failed after {completion_tokens} token(s): {error}"))?
            .next_token;
    }
    let elapsed = request_started.elapsed();
    let timing = RequestTiming::measure(
        job.prompt_tokens.len(),
        completion_tokens,
        prefill_elapsed,
        ttft,
        elapsed,
    );
    // The request cap is the only stop this driver has: the reference engine exposes no
    // end-of-sequence policy, so `Stop` is reported only when a step produced no token at all, and
    // a cap of zero is reported as `Length` because the budget is what stopped it.
    let finish_reason = if completion_tokens >= job.max_tokens {
        FinishReason::Length
    } else {
        FinishReason::Stop
    };
    job.replies
        .blocking_send(Reply::Done(Box::new(RunSummary {
            completion_tokens,
            finish_reason,
            timing,
        })))
        .map_err(|_| "the client stopped reading this response".to_owned())?;
    Ok(())
}

/// Entry point main.rs calls: parses arguments, starts the engine, binds the router.
pub fn run(args: impl IntoIterator<Item = String>) -> ExitCode {
    let options = match ServeOptions::parse(args) {
        Ok(options) => options,
        Err(message) => {
            eprintln!("sparkflash serve: {message}");
            return ExitCode::from(EXIT_REFUSED);
        }
    };
    let recipe = match options.recipe() {
        Ok(recipe) => recipe,
        Err(message) => {
            eprintln!("sparkflash serve: {message}");
            return ExitCode::from(EXIT_REFUSED);
        }
    };
    let driver = match ReferenceDriver::start(&recipe, options.element_limit, options.max_tokens) {
        Ok(driver) => driver,
        Err(message) => {
            eprintln!("sparkflash serve: {message}");
            return ExitCode::from(EXIT_REFUSED);
        }
    };
    let engine: Arc<dyn ApiInferenceEngine> = match options.engine {
        EngineChoice::Reference => Arc::new(driver),
        EngineChoice::Device => {
            let device = match DeviceEngine::open(FakeSession::new(), Arc::new(driver)) {
                Ok(device) => device,
                Err(error) => {
                    eprintln!("sparkflash serve: cannot open the device session: {error}");
                    return ExitCode::from(EXIT_REFUSED);
                }
            };
            let report = device.report();
            // The M1 skeleton says what it is before serving a request: a fake session with no
            // device execution, so a reader cannot mistake it for M2/M3 device work (design 7 M1).
            println!(
                "sparkflash serve: engine {} (M1 fake-session skeleton; fake_session={}, device_execution={}, vision={}); no device claim, no kernel launch; the first real session is M2",
                options.engine.as_str(),
                report.fake_session,
                report.device_execution,
                report.vision
            );
            Arc::new(device)
        }
    };
    let state = AppState::new(engine);
    let router = sparkflash_api::server::router(state);
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("sparkflash serve: cannot start the async runtime: {error}");
            return ExitCode::FAILURE;
        }
    };
    runtime.block_on(bind_and_serve(router, options.bind))
}

/// Binds the requested address and serves the router until the process is stopped.
///
/// The address printed is the one the listener actually bound. Those differ whenever the request
/// left the port to the kernel (`--bind 127.0.0.1:0`), and an operator needs the real one.
async fn bind_and_serve(router: axum::Router, requested: SocketAddr) -> ExitCode {
    let listener = match tokio::net::TcpListener::bind(requested).await {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("sparkflash serve: cannot bind {requested}: {error}");
            return ExitCode::FAILURE;
        }
    };
    let bound = match listener.local_addr() {
        Ok(bound) => bound,
        Err(error) => {
            eprintln!("sparkflash serve: cannot read the bound address: {error}");
            return ExitCode::FAILURE;
        }
    };
    println!("sparkflash serve: listening on http://{bound}");
    if let Err(error) = axum::serve(listener, router).await {
        eprintln!("sparkflash serve: the server stopped: {error}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

/// A weight source that panics once, on the first read of one named tensor.
///
/// No forward module panics on demand any more, so this is the only way to make the engine panic
/// and prove that the guard in [`engine_thread`] turns a panic into that one request's error event.
/// It fires once rather than on every read, so the request that follows the panic is answered
/// normally and the test can show the thread is still fit to serve. It is `#[cfg(test)]` and
/// `panic = "unwind"` only: where the process aborts there is no guard to test, and nothing a
/// shipped binary builds can construct this.
#[cfg(all(test, panic = "unwind"))]
struct PanickingSource {
    /// The tensor whose first read panics.
    name: &'static str,
    /// Whether the panic is still armed: it fires once, then the read is served normally.
    armed: std::sync::atomic::AtomicBool,
    /// The source every other tensor is served from.
    inner: SyntheticWeights,
}

#[cfg(all(test, panic = "unwind"))]
impl WeightSource for PanickingSource {
    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.inner.spec(name)
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        if name == self.name && self.armed.swap(false, std::sync::atomic::Ordering::SeqCst) {
            panic!("the test weight source panics reading {name}");
        }
        self.inner.read(name)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use sparkflash_engine::loader::Treatment;

    use super::*;

    /// A driver over the synthetic tiny plan: the path every behavioural test takes.
    fn synthetic_driver(max_tokens: usize) -> ReferenceDriver {
        let recipe = Recipe {
            config: ModelConfig::tiny(),
            weights: Weights::Synthetic {
                seed: SYNTHETIC_WEIGHT_SEED,
            },
            tokenizer: reference_tokenizer_path(),
        };
        ReferenceDriver::start(&recipe, REFERENCE_TENSOR_LIMIT, max_tokens)
            .expect("the synthetic driver starts")
    }

    /// A driver whose weight source panics on one tensor: the guard's only way in.
    #[cfg(panic = "unwind")]
    fn panicking_driver(max_tokens: usize) -> ReferenceDriver {
        let recipe = Recipe {
            config: ModelConfig::tiny(),
            weights: Weights::Panicking {
                seed: SYNTHETIC_WEIGHT_SEED,
                name: "embed.weight",
            },
            tokenizer: reference_tokenizer_path(),
        };
        ReferenceDriver::start(&recipe, REFERENCE_TENSOR_LIMIT, max_tokens)
            .expect("the panicking driver starts")
    }

    /// A request whose only interesting fields are the ones a test sets.
    fn request(prompt: &[u32], max_tokens: usize, temperature: f32, seed: u64) -> EngineRequest {
        EngineRequest {
            request_id: "test-request".to_owned(),
            model: MODEL_ID.to_owned(),
            prompt_tokens: prompt.to_vec(),
            images: Vec::new(),
            max_tokens,
            temperature,
            top_p: 1.0,
            thinking: false,
            reasoning_effort: None,
            stop: Vec::new(),
            seed: Some(seed),
            trace_scopes: 0,
            trace_steps: 0,
        }
    }

    /// Drains one request's stream.
    async fn events(driver: &ReferenceDriver, request: EngineRequest) -> Vec<EngineEvent> {
        driver.generate(request).collect().await
    }

    /// One request's events with the wall-clock measurements zeroed, so that two runs can be
    /// compared.
    ///
    /// The timing block is the one part of a terminal event that is a measurement rather than model
    /// output: two runs of the same request produce the same bytes, not the same microseconds.
    fn without_measurements(events: &[EngineEvent]) -> Vec<EngineEvent> {
        events
            .iter()
            .map(|event| match event {
                EngineEvent::Finished {
                    finish_reason,
                    usage,
                    stats,
                } => {
                    let mut measured = stats.clone();
                    measured.timing = RequestTimingStats::default();
                    EngineEvent::Finished {
                        finish_reason: *finish_reason,
                        usage: *usage,
                        stats: measured,
                    }
                }
                other => other.clone(),
            })
            .collect()
    }
    // ---- the stream ---------------------------------------------------------------------

    #[tokio::test]
    async fn the_stream_is_started_deltas_and_exactly_one_finish() {
        let driver = synthetic_driver(4);
        let stream = events(&driver, request(&[1, 2, 3], 4, 0.0, 0)).await;
        assert!(
            matches!(
                stream.first(),
                Some(EngineEvent::Started {
                    prompt_tokens: 3,
                    cached_tokens: 0
                })
            ),
            "the first event must be Started for the prompt that was sent: {stream:?}"
        );
        assert!(
            matches!(stream.last(), Some(EngineEvent::Finished { .. })),
            "the last event must be Finished: {stream:?}"
        );
        assert_eq!(
            stream
                .iter()
                .filter(|event| matches!(event, EngineEvent::Finished { .. }))
                .count(),
            1,
            "exactly one Finished: {stream:?}"
        );
        for event in &stream[1..stream.len() - 1] {
            match event {
                EngineEvent::ContentDelta(text) => {
                    assert!(!text.is_empty(), "an empty delta is not an event");
                }
                other => panic!("only deltas may sit between Started and Finished, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn the_usage_counts_the_tokens_that_were_generated() {
        let driver = synthetic_driver(16);
        let stream = events(&driver, request(&[1, 2, 3], 4, 0.0, 0)).await;
        let Some(EngineEvent::Finished {
            finish_reason,
            usage,
            stats,
        }) = stream.last()
        else {
            panic!("the stream did not finish: {stream:?}");
        };
        assert_eq!(usage.prompt_tokens, 3);
        assert_eq!(
            usage.completion_tokens, 4,
            "the request asked for four tokens and the cap is the only stop: {stream:?}"
        );
        assert_eq!(usage.total_tokens, 7);
        assert_eq!(finish_reason, &FinishReason::Length);
        assert!(
            stats.timing.ttft_ms >= 0.0 && stats.timing.ttft_ms.is_finite(),
            "ttft is a wall-clock measurement: {}",
            stats.timing.ttft_ms
        );
        assert!(
            stats.timing.prefill_tok_per_s.is_finite() && stats.timing.decode_tok_per_s.is_finite(),
            "a rate that is not a number is not a measurement: {:?}",
            stats.timing
        );
    }

    #[tokio::test]
    async fn the_unmeasured_telemetry_fields_are_the_zeros_they_are() {
        let driver = synthetic_driver(2);
        let stream = events(&driver, request(&[1, 2, 3], 2, 0.0, 0)).await;
        let Some(EngineEvent::Finished { usage, stats, .. }) = stream.last() else {
            panic!("the stream did not finish: {stream:?}");
        };
        // No cache exists in this process.
        assert_eq!(usage.prompt_tokens_details.cached_tokens, 0);
        assert_eq!(usage.prompt_tokens_details.uma_cached_tokens, 0);
        assert_eq!(stats.kv_cache.global_kv_cached_tokens, 0);
        assert_eq!(stats.kv_cache_hit_rate, 0.0);
        // No image was accepted, no draft was verified, no collective ran, no rank was visible.
        assert_eq!(usage.prompt_tokens_details.image_tokens, 0);
        assert_eq!(stats.dspark_speculative, DSparkStepStats::default());
        assert_eq!(stats.transport, RoceTransportStats::default());
        assert_eq!(stats.timing.min_rank_uma_free_gib, 0.0);
    }

    #[tokio::test]
    async fn a_zero_cap_finishes_without_a_delta() {
        let driver = synthetic_driver(4);
        let stream = events(&driver, request(&[1, 2, 3], 0, 0.0, 0)).await;
        assert!(
            !stream
                .iter()
                .any(|event| matches!(event, EngineEvent::ContentDelta(_))),
            "a request for no tokens must not carry content: {stream:?}"
        );
        let Some(EngineEvent::Finished { usage, .. }) = stream.last() else {
            panic!("a zero cap still finishes: {stream:?}");
        };
        assert_eq!(usage.completion_tokens, 0);
    }

    #[tokio::test]
    async fn the_same_request_twice_gives_the_same_bytes() {
        let driver = synthetic_driver(4);
        let first = events(&driver, request(&[1, 2, 3], 4, 0.0, 0)).await;
        let second = events(&driver, request(&[1, 2, 3], 4, 0.0, 0)).await;
        assert_eq!(
            without_measurements(&first),
            without_measurements(&second),
            "the synthetic path is deterministic"
        );
    }

    #[tokio::test]
    async fn temperature_zero_reproduces_the_engines_own_greedy_pick() {
        let prompt = [1_u32, 2, 3];
        let driver = synthetic_driver(1);
        let stream = events(&driver, request(&prompt, 1, 0.0, 4242)).await;
        let emitted: Vec<&String> = stream
            .iter()
            .filter_map(|event| match event {
                EngineEvent::ContentDelta(text) => Some(text),
                _ => None,
            })
            .collect();
        assert_eq!(emitted.len(), 1, "one token, one delta: {stream:?}");

        // The same configuration and the engine's own sampler at temperature 0, computed here and
        // not through the driver: the driver passes the request's temperature through rather than
        // sampling for itself, so the two have to agree token for token.
        let config = ModelConfig::tiny();
        let plan = TensorPlan::language_model(&config);
        let engine = InferenceEngine::new(
            config.clone(),
            Box::new(SyntheticWeights::from_plan(&plan, SYNTHETIC_WEIGHT_SEED)),
            REFERENCE_TENSOR_LIMIT,
        )
        .expect("the tiny engine builds");
        let mut state = SequenceState::new(&config);
        let step = engine.prefill(&mut state, &prompt).expect("prefill");
        let greedy = sparkflash_engine::forward::sample_gumbel(&step.logits, 0.0, 0)
            .expect("a greedy token");
        let tokenizer =
            load_tokenizer(&reference_tokenizer_path()).expect("the reference tokenizer");
        let mut decoder = tokenizer.tokenizer().decode_stream(false);
        let expected = decoder
            .step(greedy)
            .expect("the greedy token decodes")
            .expect("one whole piece");
        assert_eq!(
            emitted.first().map(|text| text.as_str()),
            Some(expected.as_str())
        );
    }

    #[tokio::test]
    async fn one_bad_request_does_not_take_the_engine_thread_with_it() {
        // A prompt token far outside the vocabulary cannot produce an answer: the engine errors when
        // it looks the row up. That failure has to arrive as this request's own terminal event, and
        // the thread has to still be there for the next request - a serve that dies with its first
        // bad request is worse than one that says what went wrong.
        let driver = synthetic_driver(4);
        let bad = events(&driver, request(&[100_000], 4, 0.0, 0)).await;
        assert!(
            matches!(bad.first(), Some(EngineEvent::Started { .. })),
            "the failure comes from inside the pass, after Started: {bad:?}"
        );
        assert!(
            matches!(bad.last(), Some(EngineEvent::Error { .. })),
            "an unanswerable request must end in an error: {bad:?}"
        );
        let good = events(&driver, request(&[1, 2, 3], 4, 0.0, 0)).await;
        assert!(
            matches!(good.last(), Some(EngineEvent::Finished { .. })),
            "the engine thread still serves after a bad request: {good:?}"
        );
    }

    /// A panic inside the engine ends that request and nothing else.
    ///
    /// The guard only exists where unwinding does: the workspace release profile sets
    /// `panic = "abort"` and Cargo refuses `panic` in a package profile, so this behaviour belongs to
    /// dev and test builds and a release binary aborts by design, as the module docs say.
    #[cfg(panic = "unwind")]
    #[tokio::test]
    async fn a_panic_inside_the_engine_ends_one_request_and_not_the_thread() {
        let driver = panicking_driver(4);
        let panicked = events(&driver, request(&[1, 2, 3], 4, 0.0, 0)).await;
        assert!(
            matches!(panicked.first(), Some(EngineEvent::Started { .. })),
            "the panic happens inside the pass, after Started: {panicked:?}"
        );
        match panicked.last() {
            Some(EngineEvent::Error { message, .. }) => assert!(
                message.contains("the test weight source panics reading embed.weight"),
                "the operator sees the panic text: {message}"
            ),
            other => panic!("a panic must arrive as this request's error event: {other:?}"),
        }
        let good = events(&driver, request(&[1, 2, 3], 4, 0.0, 0)).await;
        assert!(
            matches!(good.last(), Some(EngineEvent::Finished { .. })),
            "the engine thread still serves after a panic: {good:?}"
        );
    }

    // ---- refusals that never reach the engine --------------------------------------------

    #[tokio::test]
    async fn an_image_request_is_refused_with_the_typed_missing_capability() {
        let driver = synthetic_driver(4);
        let mut with_image = request(&[1, 2, 3], 4, 0.0, 0);
        with_image.images = vec![sparkflash_api::ImagePart::Placeholder];
        let stream = events(&driver, with_image).await;
        assert_eq!(stream.len(), 1, "a refusal is one event: {stream:?}");
        match stream.first() {
            Some(EngineEvent::Error { message, code }) => {
                assert_eq!(code, MissingCapability::vision().code());
                assert!(
                    message.contains("vision"),
                    "the refusal names the channel: {message}"
                );
            }
            other => panic!("a refusal is one Error event: {other:?}"),
        }
    }

    #[test]
    fn the_driver_declares_no_channel_it_does_not_serve() {
        // The image path and the tool/reasoning channels: the reference pass implements none of them.
        assert_eq!(synthetic_driver(1).capabilities(), Capabilities::default());
        // And the capability check refuses an image part rather than letting `generate` answer from
        // the text tokens alone.
        let mut with_image = request(&[1, 2, 3], 4, 0.0, 0);
        with_image.images = vec![sparkflash_api::ImagePart::Placeholder];
        assert_eq!(
            synthetic_driver(1).validate_request(&with_image),
            Err(MissingCapability::vision())
        );
    }

    #[tokio::test]
    async fn a_request_with_no_prompt_is_refused() {
        let driver = synthetic_driver(4);
        let stream = events(&driver, request(&[], 4, 0.0, 0)).await;
        assert!(matches!(stream.first(), Some(EngineEvent::Error { .. })));
    }

    // ---- cluster telemetry ---------------------------------------------------------------

    #[tokio::test]
    async fn cluster_stats_report_only_what_this_process_can_see() {
        let driver = synthetic_driver(1);
        let snapshot = driver.cluster_stats();
        assert_eq!(snapshot.model, MODEL_ID);
        assert!(
            snapshot.nodes.is_empty(),
            "one process has no ranks to report: {:?}",
            snapshot.nodes
        );
        assert_eq!(snapshot.kv, empty_kv_snapshot());
        assert_eq!(snapshot.max_seq_len, max_seq_len());
        assert_eq!(snapshot.dspark, DSparkStepStats::default());
        assert_eq!(
            snapshot.transport,
            RoceTransportStats {
                fabric_mode: NO_FABRIC.to_owned(),
                ..RoceTransportStats::default()
            }
        );
    }

    #[test]
    fn the_driver_reports_the_canonical_model_id() {
        let driver = synthetic_driver(1);
        assert_eq!(driver.model_id(), MODEL_ID);
    }

    // ---- option parsing ------------------------------------------------------------------

    #[test]
    fn synthetic_and_checkpoint_together_are_refused() {
        let error = ServeOptions::parse(vec![
            "--synthetic".to_owned(),
            "--checkpoint".to_owned(),
            "/tmp/checkpoint".to_owned(),
        ])
        .expect_err("both weight sources cannot be served at once");
        assert!(error.contains("--synthetic"), "{error}");
        assert!(error.contains("--checkpoint"), "{error}");
    }

    #[test]
    fn no_weight_source_at_all_is_refused() {
        let error = ServeOptions::parse(Vec::new()).expect_err("a server needs weights");
        assert!(error.contains("--synthetic"), "{error}");
        assert!(error.contains("--checkpoint"), "{error}");
    }

    #[test]
    fn a_bad_tokenizer_path_is_refused_by_name() {
        let path = PathBuf::from("/definitely/not/a/tokenizer.json");
        let error = load_tokenizer(&path)
            .err()
            .expect("there is no tokenizer there");
        assert!(error.contains("--tokenizer"), "{error}");
        assert!(error.contains("tokenizer.json"), "{error}");
    }

    #[test]
    fn a_checkpoint_directory_without_an_index_is_refused_by_name() {
        let error = checkpoint_recipe(Path::new("/tmp/not-a-checkpoint"), Path::new("t.json"))
            .expect_err("there is no checkpoint there");
        assert!(error.contains("--checkpoint"), "{error}");
        assert!(error.contains(CHECKPOINT_INDEX_NAME), "{error}");
    }

    #[test]
    fn the_engine_flag_defaults_to_reference_and_accepts_device() {
        let reference = ServeOptions::parse(vec!["--synthetic".to_owned()]).expect("parses");
        assert_eq!(reference.engine, EngineChoice::Reference);

        let device = ServeOptions::parse(vec![
            "--synthetic".to_owned(),
            "--engine".to_owned(),
            "device".to_owned(),
        ])
        .expect("parses");
        assert_eq!(device.engine, EngineChoice::Device);
        assert_eq!(device.engine.as_str(), "device");

        let error = ServeOptions::parse(vec![
            "--synthetic".to_owned(),
            "--engine".to_owned(),
            "gpu".to_owned(),
        ])
        .expect_err("an unknown engine is refused");
        assert!(error.contains("gpu"), "{error}");
    }

    #[test]
    fn the_defaults_are_loopback_and_the_reference_caps() {
        let options = ServeOptions::parse(vec!["--synthetic".to_owned()]).expect("parses");
        assert_eq!(options.bind, SocketAddr::from(([127, 0, 0, 1], 8000)));
        assert_eq!(options.max_tokens, MODEL_MAX_OUTPUT_TOKENS);
        assert_eq!(options.element_limit, REFERENCE_TENSOR_LIMIT);
        assert_eq!(options.tokenizer, reference_tokenizer_path());
        assert!(options.synthetic);
        assert!(options.checkpoint.is_none());
    }

    #[test]
    fn a_checkpoint_defaults_to_the_checkpoint_cap_and_an_explicit_flag_wins() {
        // The bug this pins: a --checkpoint run defaulted to the 32 Mi reference guard, so the
        // released embed.weight/head.weight (661,913,600 elements each) would be refused in the
        // middle of the first request. The cap follows the source, and an operator's explicit
        // value still wins for both sources.
        let checkpoint = ServeOptions::parse(vec![
            "--checkpoint".to_owned(),
            "/tmp/not-read-at-parse-time".to_owned(),
        ])
        .expect("parses");
        assert_eq!(checkpoint.element_limit, CHECKPOINT_TENSOR_LIMIT);
        // The caps' relation to the released embedding is a compile-time assertion beside
        // RELEASED_EMBEDDING_ELEMENTS, which `default_element_limit` returns at the one place the
        // choice is made; this test pins which cap the parse selects, for both sources.
        let synthetic = ServeOptions::parse(vec!["--synthetic".to_owned()]).expect("parses");
        assert_eq!(synthetic.element_limit, REFERENCE_TENSOR_LIMIT);

        let explicit = ServeOptions::parse(vec![
            "--checkpoint".to_owned(),
            "/tmp/not-read-at-parse-time".to_owned(),
            "--element-limit".to_owned(),
            "1234".to_owned(),
        ])
        .expect("parses");
        assert_eq!(explicit.element_limit, 1234);

        let explicit_synthetic = ServeOptions::parse(vec![
            "--synthetic".to_owned(),
            "--element-limit".to_owned(),
            "1234".to_owned(),
        ])
        .expect("parses");
        assert_eq!(explicit_synthetic.element_limit, 1234);

        // The constant the compile-time assertions compare against is the released
        // vocab x dim, derived here from the released configuration rather than trusted.
        let released = ModelConfig::v41_flash().expect("the vendored released config parses");
        assert_eq!(
            released.vocab_size * released.dim,
            RELEASED_EMBEDDING_ELEMENTS,
            "the released embed/head element count must match the vendored configuration"
        );
    }

    // ---- the checkpoint path, on a checkpoint this test writes ------------------------

    /// The one shard the fixture writes.
    const FIXTURE_SHARD: &str = "model-00001-of-00001.safetensors";

    /// The fixture configuration: `tiny`, but with a `wo_a` the released 32x32 scale grid
    /// divides. `tiny` alone has a [16, 16] `wo_a`, which the loader refuses by name because its
    /// grid would be (16, 16); heads is n_heads * head_dim = 8 * 16 = 128, so the in dimension is
    /// heads / o_groups = 64 and the out dimension is o_groups * o_lora_rank = 64.
    fn checkpoint_config() -> ModelConfig {
        ModelConfig {
            o_lora_rank: 32,
            n_heads: 8,
            ..ModelConfig::tiny()
        }
    }

    /// One tensor as the fixture stores it.
    struct FixtureEntry {
        name: String,
        dtype: String,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    /// How many bytes a stored word of that shape occupies.
    fn stored_bytes(word: &str, shape: &[usize]) -> usize {
        let elements: usize = shape.iter().product();
        let per = match word {
            "F32" => 4,
            "BF16" | "F16" => 2,
            _ => 1,
        };
        elements * per
    }

    /// The stored dtype word and shape the plan's rules require of one planned tensor, exactly as
    /// `crates/sparkflash-engine/tests/loader.rs` does it: the packed expert is I8 with the last
    /// dimension halved, everything else is the plan's own dtype and shape. `wo_a` is the plan's
    /// F8E4M3 plus its separate E8M0 grid, because that is the stored form the reference decodes.
    fn stored_form(spec: &TensorSpec) -> (String, Vec<usize>) {
        if Treatment::of(spec, &spec.name) == Treatment::PackedExpert {
            let mut shape = spec.shape.clone();
            let last = shape.len() - 1;
            assert_eq!(shape[last] % 2, 0, "{} is packed two per byte", spec.name);
            shape[last] /= 2;
            ("I8".to_string(), shape)
        } else {
            (spec.dtype.name().to_string(), spec.shape.clone())
        }
    }

    /// A safetensors container: an 8-byte little-endian header length, the JSON header, then data.
    fn fixture_container(entries: &[FixtureEntry]) -> Vec<u8> {
        let mut offset = 0u64;
        let mut tensors = serde_json::Map::new();
        let mut data = Vec::new();
        for item in entries {
            let begin = offset;
            offset += item.data.len() as u64;
            tensors.insert(
                item.name.clone(),
                serde_json::json!({
                    "dtype": item.dtype,
                    "shape": item.shape,
                    "data_offsets": [begin, offset],
                }),
            );
            data.extend_from_slice(&item.data);
        }
        let header = serde_json::Value::Object(tensors).to_string();
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&data);
        out
    }

    /// A plan-complete checkpoint directory: one shard, the index that names it, and the
    /// configuration the checkpoint carries. Every tensor is stored the way the plan's rules say;
    /// the layer-0 `wo_a` weight and its scale carry real codes (E4M3 0x38 is 1.0, E8M0 0x80 is
    /// 2.0 under the 32x32 grid), so a read of it has a value to check rather than only a length.
    struct CheckpointFixture {
        dir: tempfile::TempDir,
        index: PathBuf,
    }

    impl CheckpointFixture {
        fn dir(&self) -> &Path {
            self.dir.path()
        }

        fn index(&self) -> &Path {
            &self.index
        }
    }

    fn written_checkpoint(config: &ModelConfig) -> CheckpointFixture {
        let plan = TensorPlan::language_model(config);
        let mut entries: Vec<FixtureEntry> = plan
            .specs()
            .iter()
            .map(|spec| {
                let (dtype, shape) = stored_form(spec);
                let data = vec![0u8; stored_bytes(&dtype, &shape)];
                FixtureEntry {
                    name: spec.name.clone(),
                    dtype,
                    shape,
                    data,
                }
            })
            .collect();
        for entry in &mut entries {
            if entry.name == "layers.0.attn.wo_a.weight" {
                entry.data = vec![0x38; entry.data.len()];
            } else if entry.name == "layers.0.attn.wo_a.scale" {
                entry.data = vec![0x80; entry.data.len()];
            }
        }

        let dir = tempfile::TempDir::new().expect("a temporary directory");
        std::fs::write(dir.path().join(FIXTURE_SHARD), fixture_container(&entries))
            .expect("the fixture shard");
        let weight_map: BTreeMap<&str, &str> = entries
            .iter()
            .map(|item| (item.name.as_str(), FIXTURE_SHARD))
            .collect();
        let total: u64 = entries.iter().map(|item| item.data.len() as u64).sum();
        let index = serde_json::json!({
            "metadata": { "total_size": total },
            "weight_map": weight_map,
        })
        .to_string();
        let index_path = dir.path().join(CHECKPOINT_INDEX_NAME);
        std::fs::write(&index_path, index).expect("the fixture index");
        std::fs::write(
            dir.path().join(CHECKPOINT_CONFIG_NAME),
            serde_json::to_string(config).expect("the configuration serialises"),
        )
        .expect("the fixture configuration");
        CheckpointFixture {
            dir,
            index: index_path,
        }
    }

    #[test]
    fn a_stored_fp8_wo_a_reads_through_reference_weights_with_its_own_grid() {
        // The decode that the serving path used to get wrong: the loader's converted form is BF16
        // (2 bytes/element) under the plan's F8E4M3 spec (1 byte/element), so decoding it fails on
        // length. The reference source serves the stored F8E4M3 codes and the separate grid.
        let config = checkpoint_config();
        let plan = TensorPlan::language_model(&config);
        let checkpoint = written_checkpoint(&config);
        let source =
            ReferenceWeights::open(checkpoint.dir(), checkpoint.index(), &plan).expect("opens");
        let weight = sparkflash_engine::linalg::decode_by_name(
            &source,
            "layers.0.attn.wo_a.weight",
            CHECKPOINT_TENSOR_LIMIT,
        )
        .expect("the stored wo_a decodes under the plan's F8E4M3 spec");
        assert_eq!(weight.len(), 64 * 64, "the plan's wo_a is [64, 64]");
        assert!(
            weight.iter().all(|&value| value == 1.0),
            "E4M3 0x38 decodes to 1.0"
        );
        let scales = sparkflash_engine::linalg::decode_by_name(
            &source,
            "layers.0.attn.wo_a.scale",
            CHECKPOINT_TENSOR_LIMIT,
        )
        .expect("the grid decodes");
        let mut scaled = weight;
        sparkflash_engine::linalg::apply_scales(
            &mut scaled,
            64,
            64,
            &scales,
            sparkflash_engine::linalg::ScaleGrid::Block32,
        )
        .expect("the stored grid divides the weight");
        assert!(
            scaled.iter().all(|&value| value == 2.0),
            "E4M3 1.0 under an E8M0 2.0 grid is 2.0"
        );
    }

    #[test]
    fn a_plan_complete_checkpoint_builds_the_serve_engine_and_prefills() {
        // The whole `serve --checkpoint` path on a shard this test writes: the recipe reads
        // config.json, build_engine verifies the checkpoint and hands the reference pass a
        // ReferenceWeights, and prefill reads every tensor of layer 0 - wo_a included. With the
        // loader's converted source this failed on layer 0's wo_a with "F8_E4M3 needs 4096 bytes
        // for 4096 elements, but 8192 were given"; this test is the regression for that.
        let config = checkpoint_config();
        let checkpoint = written_checkpoint(&config);
        let recipe =
            checkpoint_recipe(checkpoint.dir(), &reference_tokenizer_path()).expect("the recipe");
        assert!(
            matches!(&recipe.weights, Weights::Checkpoint { .. }),
            "the recipe reads the checkpoint, not the synthetic plan"
        );
        let engine = build_engine(&recipe, CHECKPOINT_TENSOR_LIMIT)
            .expect("a plan-complete checkpoint builds the reference engine");
        assert_eq!(engine.config(), &config);
        let mut state = SequenceState::new(engine.config());
        let step = engine
            .prefill(&mut state, &[1, 2, 3])
            .expect("the reference pass reads the stored wo_a as F8E4M3");
        assert_eq!(step.logits.len(), config.vocab_size);
    }

    // ---- the timing block ----------------------------------------------------------------

    #[test]
    fn a_timing_block_that_cannot_be_resolved_is_zero_rather_than_infinite() {
        let timing =
            RequestTiming::measure(3, 1, Duration::ZERO, Some(Duration::ZERO), Duration::ZERO);
        assert_eq!(timing.ttft_ms, 0.0);
        assert_eq!(timing.prefill_tok_per_s, 0.0);
        assert_eq!(timing.decode_tok_per_s, 0.0);
        assert!(timing.prefill_tok_per_s.is_finite());
    }

    #[test]
    fn a_request_that_generated_nothing_reports_no_time_to_first_token() {
        let half = Duration::from_millis(500);
        let timing = RequestTiming::measure(3, 0, half, None, half);
        assert_eq!(timing.ttft_ms, 0.0, "there was no first token to time");
        assert!(
            (timing.prefill_tok_per_s - 6.0).abs() < 1.0,
            "three tokens over half a second: {}",
            timing.prefill_tok_per_s
        );
        assert_eq!(timing.decode_tok_per_s, 0.0, "no decode step ran");
    }
}

//! `sparkflash serve`: the HTTP entry point over the reference engine.
//!
//! `sparkflash_api::server::router` has always been the whole HTTP surface of this project: it
//! owns the OpenAI-compatible routes, the SSE framing and the
//! [`sparkflash_api::engine::InferenceEngine`] seam. What was missing was something on the other
//! side of that seam. This module is that thing and nothing more: it builds a
//! [`sparkflash_engine::forward::InferenceEngine`] over either the synthetic plan or a real
//! checkpoint, presents it to the router through a local driver, and binds the router to a socket.
//!
//! `--engine device` selects the `sparkflash-device` engine instead: a `DeviceEngine` over the
//! same reference engine. Without a cubin it is the M1 fake session (no driver, no launch,
//! `vision: false`). With `--device-cubin PATH` and a binary built with the `cuda` feature it
//! is the M2 real session: one pinned `moe_router_sm121` launch, evidence only, no chain and no
//! token claim (design `docs/build/device-forward-chain.md` 1.3, 7 M1, 7 M2). It is not the
//! default: `--engine reference` is, and that path is unchanged. The first real chain is M3.
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

use std::collections::HashMap;
use std::net::SocketAddr;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use futures::StreamExt;
use futures::stream::{self, BoxStream};
use sparkflash_api::AppState;
use sparkflash_api::encoding::{
    HfTokenizerEncoder, TokenizerSeam, load_hf_tokenizer, resolve_reference_tokenizer,
};
use sparkflash_api::engine::InferenceEngine as ApiInferenceEngine;
use sparkflash_api::engine::{
    Capabilities, ClusterStatsSnapshot, EngineEvent, EngineRequest, FinishReason, LogprobSelection,
    LogprobsRequest, MODEL_ID, MODEL_MAX_OUTPUT_TOKENS, MissingCapability, PromptLogprobs,
    TokenLogprob, TokenString, default_engram_layers, empty_kv_snapshot, max_seq_len,
};
use sparkflash_api::telemetry_types::{
    CompletionTokensDetails, DSparkStepStats, DeepSeekSparkRequestStats, OpenAiUsageWithCache,
    PromptTokensDetails, RequestTimingStats, RoceTransportStats, finite_or_zero,
};
#[cfg(feature = "cuda")]
use sparkflash_device::CudaSession;
use sparkflash_device::{DeviceEngine, DeviceSession, FakeSession};
use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::{InferenceEngine, Sampling, SequenceState, StepOutput};
use sparkflash_engine::linalg::{CHECKPOINT_TENSOR_LIMIT, REFERENCE_TENSOR_LIMIT};
use sparkflash_engine::loader::ReferenceWeights;
use sparkflash_engine::synthetic::SyntheticWeights;
use sparkflash_engine::tensor::{TensorPlan, TensorSpec};
use sparkflash_engine::weights::{WeightError, WeightSource};
use sparkflash_engram::layout::SCALE_BLOCK;
use sparkflash_engram::packer::{MANIFEST_NAME, PackManifest};
use sparkflash_engram::paged::{EngramTensor, PagedTable};
use sparkflash_engram::store::{BackendPreference, DirectTable, RowTable, specs_from_manifest};
use sparkflash_kv::PrefixMatchPlan;
use sparkflash_scheduler::config::SchedulerConfig;
use sparkflash_scheduler::prefill::ChunkedPrefillPlan;

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
/// The reference engine is the existing CPU path and the default. `device` is the
/// `sparkflash-device` engine in one of two forms: without `--device-cubin` it is the M1 fake
/// session (delegates `generate` back to the reference engine, refuses images through the typed
/// missing-capability path, refuses non-greedy sampling, claims no device execution); with
/// `--device-cubin` and a binary built with the `cuda` feature it is the M2 real session (one
/// pinned `moe_router_sm121` launch as an execution proof, evidence only, still no chain and no
/// token claim). Design `docs/build/device-forward-chain.md` 1.3, 7 M1, 7 M2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum EngineChoice {
    /// The existing CPU reference engine: unchanged serving behaviour.
    Reference,
    /// The device engine: the M1 fake session without a cubin, the M2 real session with one.
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

/// What the cli should construct for the engine, decided from the flags with no side effects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceSelection {
    /// The existing CPU reference engine: the default, and unchanged by M1/M2.
    Reference,
    /// The M1 fake-session skeleton: no driver, no launch, no device claim.
    FakeSession,
    /// The M2 real session: open the driver and run one pinned router launch from this cubin.
    CudaSession(PathBuf),
}

/// Decides which engine to construct, purely, so the selection rules are testable on a CPU host.
///
/// `--device-cubin` is only meaningful with `--engine device`, and the real session only exists
/// when the binary was built with the `cuda` feature. Without a cubin, `--engine device` is the
/// fake session: that keeps the M1 skeleton and the CI default constructible without a device.
///
/// # Errors
/// A message naming the flag combination or the missing feature; the caller prints it and refuses
/// to start.
pub fn select_engine(
    engine: EngineChoice,
    device_cubin: Option<&Path>,
    cuda_built: bool,
) -> Result<DeviceSelection, String> {
    match (engine, device_cubin) {
        (EngineChoice::Reference, None) => Ok(DeviceSelection::Reference),
        (EngineChoice::Reference, Some(_)) => Err(
            "--device-cubin is only used with --engine device; the reference engine has no device session"
                .to_owned(),
        ),
        (EngineChoice::Device, None) => Ok(DeviceSelection::FakeSession),
        (EngineChoice::Device, Some(cubin)) => {
            if !cuda_built {
                return Err(
                    "this binary was built without the cuda feature: --device-cubin needs a build with --features cuda"
                        .to_owned(),
                );
            }
            Ok(DeviceSelection::CudaSession(cubin.to_path_buf()))
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
    /// The `tokenizer.json` to detokenise with, when the operator passed `--tokenizer`.
    ///
    /// `None` means the default is resolved at startup from the executable's directory, then the
    /// current directory (`reference/tokenizer.json` under either), so a binary copied to a node
    /// does not depend on the build host's checkout path. `--tokenizer` always overrides.
    pub tokenizer: Option<PathBuf>,
    /// The ceiling on tokens generated for one request. A request asking for more is clamped to it.
    pub max_tokens: usize,
    /// The cap on requested logprobs, vLLM's `--max-logprobs`: 20 by default, `-1` for no cap.
    ///
    /// `usize::MAX` is how "no cap" is carried: the router compares each request against it.
    pub max_logprobs: usize,
    /// The element cap the reference pass applies to one tensor. Above it, a tensor is refused
    /// rather than materialised.
    ///
    /// Resolved at parse time: an explicit `--element-limit`, else [CHECKPOINT_TENSOR_LIMIT] for
    /// `--checkpoint` and [REFERENCE_TENSOR_LIMIT] for `--synthetic`.
    pub element_limit: usize,
    /// Which engine implementation to serve. The default is the existing reference engine; the
    /// device selection is the M1 skeleton or, with a cubin and the cuda feature, the M2 session.
    pub engine: EngineChoice,
    /// The pinned router cubin for the M2 real session. Only with `--engine device` and a binary
    /// built with the `cuda` feature; absent means the M1 fake session.
    pub device_cubin: Option<PathBuf>,
    /// The packed Engram store directory, when `--engram-dir` was given.
    ///
    /// Only meaningful with `--checkpoint`; the parse refuses it with `--synthetic`. When set,
    /// [build_engine] wraps the checkpoint source so the two Engram tables are paged from the
    /// packed rows instead of being read as tensors from the shards.
    pub engram_dir: Option<PathBuf>,
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
        let tokenizer = cli.tokenizer;
        let checkpoint = cli.checkpoint;
        // The cap follows the source when the operator does not name one: a real checkpoint's
        // largest tensors are the embedding and the head at 661,913,600 elements each, and
        // [REFERENCE_TENSOR_LIMIT] would refuse them mid-request. This is the one place the choice
        // is made, and `default_element_limit` is where a test pins it.
        let element_limit = cli
            .element_limit
            .unwrap_or_else(|| default_element_limit(checkpoint.as_deref()));
        // vLLM's `--max-logprobs` semantics: a positive cap, or -1 for no cap. 0 is not a cap and
        // is refused rather than treated as "no logprobs", which would silently answer every
        // request with an empty `top_logprobs`.
        let max_logprobs = match cli.max_logprobs {
            value if value < -1 || value == 0 => {
                return Err(
                    "--max-logprobs must be a positive value or -1 (-1 means no cap)".to_owned(),
                );
            }
            -1 => usize::MAX,
            value => usize::try_from(value).unwrap_or(usize::MAX),
        };
        // The packed Engram store is rows of a checkpoint's tensors; the synthetic plan carries
        // its own tiny table in memory, so there is no checkpoint to page and the combination is
        // refused rather than silently ignored.
        if cli.synthetic && cli.engram_dir.is_some() {
            return Err(
                "--engram-dir needs --checkpoint: the packed store holds rows of a checkpoint's Engram tensors, and --synthetic already carries its tiny table in memory"
                    .to_owned(),
            );
        }
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
                max_logprobs,
                element_limit,
                engine: cli.engine,
                device_cubin: cli.device_cubin,
                engram_dir: cli.engram_dir,
            }),
        }
    }

    /// The weights recipe this run serves, resolved before any thread starts.
    ///
    /// The tokenizer is passed in because the default is resolved from the *process* directories
    /// (see [`resolve_reference_tokenizer`]) rather than at parse time; `--tokenizer` is still
    /// the override, and the caller has already chosen between it and the resolved default.
    ///
    /// # Errors
    ///
    /// A message naming `--checkpoint` and the file that is missing or unusable.
    fn recipe(&self, tokenizer: &Path) -> Result<Recipe, String> {
        let mut recipe = match (self.synthetic, self.checkpoint.as_deref()) {
            (true, _) => Recipe {
                config: ModelConfig::tiny(),
                weights: Weights::Synthetic {
                    seed: SYNTHETIC_WEIGHT_SEED,
                },
                tokenizer: tokenizer.to_path_buf(),
                engram_dir: None,
            },
            (false, Some(path)) => checkpoint_recipe(path, tokenizer)?,
            (false, None) => return Err(no_weights_message()),
        };
        // The parse already refused --engram-dir with --synthetic; this is where the checkpoint
        // recipe learns where its Engram rows come from.
        recipe.engram_dir = self.engram_dir.clone();
        Ok(recipe)
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
pub(crate) fn default_element_limit(checkpoint: Option<&Path>) -> usize {
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
    long_about = "Serves the HTTP API from the reference forward pass. Exactly one weight source is required: --synthetic generates the tiny reference plan from a seed and needs no checkpoint, while --checkpoint reads a real sharded checkpoint. The default run is read-only apart from what serving implies: no hardware state is changed and nothing is written to the checkpoint. --engine device selects the sparkflash-device engine instead: the M1 fake session that delegates to the reference engine, or with --device-cubin and a cuda build the M2 one-launch session; both refuse images and non-greedy sampling, and neither claims a chain or tokens. The default is --engine reference and is unchanged."
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
    /// The tokenizer.json to detokenise with; without it the default is resolved at startup.
    ///
    /// The default is `reference/tokenizer.json` next to the executable, then under the current
    /// directory. That makes a copied binary work when the stage carries the `reference/`
    /// directory, and never depends on the checkout that built it. A real deployment passes the
    /// model's own tokenizer file, because detokenising with the wrong vocabulary produces text
    /// that is not the model's output.
    #[arg(long, value_name = "FILE")]
    tokenizer: Option<PathBuf>,
    /// The ceiling on tokens generated for one request.
    #[arg(long, value_name = "N", default_value_t = MODEL_MAX_OUTPUT_TOKENS)]
    max_tokens: usize,
    /// The cap on requested logprobs, vLLM's `--max-logprobs`.
    ///
    /// A request asking for more, or for the whole vocabulary with `logprobs=-1`, is refused with
    /// vLLM's error text. The default is 20, vLLM's own default; the cluster's oracle window runs
    /// with 1000 because it compares full distributions.
    #[arg(
        long,
        value_name = "N",
        allow_hyphen_values = true,
        default_value_t = 20
    )]
    max_logprobs: i64,
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
    /// Which engine to serve: the default `reference` CPU path, or the `device` engine.
    ///
    /// `device` wraps the reference engine in `sparkflash-device`'s `DeviceEngine`. Without
    /// `--device-cubin` it is the M1 fake session: no kernel launch, no device execution,
    /// `vision: false` and non-greedy sampling refused. With `--device-cubin` and a binary built
    /// with the `cuda` feature it is the M2 real session: one pinned router launch, evidence only,
    /// still no chain and no token claim. The first real chain is M3.
    #[arg(long, value_enum, default_value = "reference")]
    engine: EngineChoice,
    /// The pinned router cubin for the M2 real session (only with `--engine device`).
    ///
    /// Requires a binary built with the `cuda` feature. Without the flag, `--engine device` is
    /// the M1 fake session, which is what the CI host and the default build use.
    #[arg(long, value_name = "PATH")]
    device_cubin: Option<PathBuf>,
    /// Page the two Engram tables from the packed dense store instead of the checkpoint shards.
    ///
    /// The directory is what `sparkflash pack-engram` wrote: `manifest.json` plus the packed
    /// rank files. The serving path already asks for Engram rows a chunk at a time through the
    /// weight source's range interface; this flag makes those requests read the 264-byte packed
    /// rows with the engram crate's windowed reader instead of materialising tensors from the
    /// checkpoint shards. Nothing about it is held in memory: a request reads only the rows its
    /// hash ids name.
    ///
    /// Requires `--checkpoint`: the packed tables are rows of that checkpoint's Engram tensors,
    /// and `--synthetic` already carries its tiny table in memory. The reference server has no
    /// rank, so every rank file named by the manifest is opened and a row is read from whichever
    /// file owns it; a rank-local or device-side placement is a window/serving-wiring question
    /// this host path does not answer.
    #[arg(long, value_name = "DIR")]
    engram_dir: Option<PathBuf>,
}

/// What the engine thread builds: a configuration, where its weights come from, and the tokenizer.
#[derive(Debug, Clone)]
struct Recipe {
    config: ModelConfig,
    weights: Weights,
    tokenizer: PathBuf,
    /// Where the two Engram tables are paged from: the packed dense store, or `None` when the
    /// run reads them the old way (which `--synthetic` can, since its tiny table fits in memory,
    /// and which a real checkpoint does not, since the shards refuse them).
    engram_dir: Option<PathBuf>,
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
        // The caller attaches --engram-dir after resolving the source; a checkpoint recipe
        // itself does not know where a packed store lives.
        engram_dir: None,
    })
}

// ---------------------------------------------------------------------------------------
// Engram on the packed store
// ---------------------------------------------------------------------------------------
//
// The reference pass already pages the two Engram tables: `engram_forward` asks the weight source
// for the byte ranges of the rows a chunk looks up, and falls back to a whole-tensor read only
// when the source cannot answer ranges. What was missing in `sparkflash serve` was a source that
// answers them from the packed dense store. This section is that source.
//
// It deliberately does no decoding and no hashing: the engine owns which rows a chunk wants, this
// only translates "weight bytes [id*width, id*width+width)" into "packed row id" and back. The
// rows are read through `sparkflash_engram::store::DirectTable`, which is the existing O_DIRECT
// windowed reader over the packer's 264-byte rows, with its mmap fallback for a filesystem that
// refuses direct I/O.
//
// Everything here is single-process and host-side. Multi-rank placement, a device-side reader and
// the measured cost of this path are not claimed here: the read count and granularity are host
// properties the tests state, and the latency/bandwidth are what the window has to measure.

/// One Engram layer's tensor geometry, resolved from the served configuration.
#[derive(Debug, Clone, Copy)]
struct EngramRoute {
    /// The backbone layer whose table this is.
    layer_id: usize,
    /// Bytes of one weight row (the released 256).
    width: usize,
    /// Bytes of one scale row (the released 8).
    scale_width: usize,
}

/// The read counters of one serving process's Engram paging.
///
/// These count what the *engine asked for*: two range requests per layer per chunk (weight and
/// scale), the rows those requests named, the batched row reads that served them and the bytes
/// returned. They are not a performance measurement: the windowed reader fetches aligned windows,
/// so its real I/O is a multiple of `bytes_requested` and that amplification is a window result.
#[derive(Debug, Default)]
struct EngramPagingStats {
    weight_requests: AtomicU64,
    scale_requests: AtomicU64,
    rows_requested: AtomicU64,
    read_batches: AtomicU64,
    bytes_requested: AtomicU64,
}

/// A point-in-time copy of [`EngramPagingStats`], for a test.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct EngramPagingSnapshot {
    weight_requests: u64,
    scale_requests: u64,
    rows_requested: u64,
    read_batches: u64,
    bytes_requested: u64,
}

impl EngramPagingStats {
    #[cfg(test)]
    fn snapshot(&self) -> EngramPagingSnapshot {
        EngramPagingSnapshot {
            weight_requests: self.weight_requests.load(Ordering::Relaxed),
            scale_requests: self.scale_requests.load(Ordering::Relaxed),
            rows_requested: self.rows_requested.load(Ordering::Relaxed),
            read_batches: self.read_batches.load(Ordering::Relaxed),
            bytes_requested: self.bytes_requested.load(Ordering::Relaxed),
        }
    }
}

/// A [`WeightSource`] that pages the two Engram tables from the packed dense store.
///
/// Every other tensor is delegated unchanged to `inner`; only `layers.N.engram.embed.weight` and
/// `layers.N.engram.embed.scale` are answered from the packed rows. A whole-tensor read of those
/// two names is *not* intercepted: it goes to `inner` and is refused there, which is the honest
/// answer for a 98 GB table and keeps a caller from silently allocating one.
struct DiskEngramWeights {
    /// The rest of the model: the checkpoint source (or, in a host test, a synthetic one).
    inner: Box<dyn WeightSource>,
    /// The windowed reader over the packed rank files.
    table: DirectTable,
    /// The layers this run's configuration declares, with their tensor widths.
    routes: Vec<EngramRoute>,
    /// What has been asked for and served so far.
    stats: Arc<EngramPagingStats>,
}

impl DiskEngramWeights {
    /// Opens the packed store in `dir` and wraps `inner`, which keeps serving every other tensor.
    ///
    /// The manifest names the rank files and the world size; every rank file is opened because a
    /// single reference process has no rank of its own and a chunk's hash ids may fall in any
    /// rank's row range. A rank-local serving process is a later wiring question.
    ///
    /// # Errors
    /// When the manifest is missing or unreadable, when it disagrees with the compiled geometry
    /// or the served configuration, or when a rank file cannot be opened as the packer wrote it.
    fn open(
        inner: Box<dyn WeightSource>,
        dir: &Path,
        config: &ModelConfig,
    ) -> Result<Self, String> {
        let manifest = load_engram_manifest(dir)?;
        let world_size = manifest.world_size;
        if world_size == 0 {
            return Err(format!(
                "--engram-dir {}: the manifest says world size 0",
                dir.display()
            ));
        }
        let mut specs = Vec::new();
        for rank in 0..world_size {
            let rank_specs = specs_from_manifest(&manifest, dir, rank, world_size)
                .map_err(|error| format!("--engram-dir {}: {error}", dir.display()))?;
            specs.extend(rank_specs);
        }
        let mut log = |reason: &str| {
            eprintln!("sparkflash serve: engram: {reason}");
        };
        let table = DirectTable::open_with(specs, BackendPreference::Auto, &mut log)
            .map_err(|error| format!("--engram-dir {}: {error}", dir.display()))?;
        for index in 0..table.table_count() {
            let (Some(spec), Some(backend)) = (table.spec(index), table.backend_of(index)) else {
                continue;
            };
            eprintln!(
                "sparkflash serve: engram layer {} rank {}/{}: {} on the {backend:?} backend, {} row window",
                spec.layer_id,
                spec.rank,
                spec.world_size,
                spec.path.display(),
                spec.window_bytes
            );
        }
        if let Some(reason) = table.fallback_reason() {
            eprintln!("sparkflash serve: engram: O_DIRECT unavailable: {reason}");
        }
        Self::from_table(inner, table, config)
    }

    /// Wraps an already-opened reader. Split out so a host test can use a temporary fixture
    /// instead of a 188 GiB packed store, without a second code path.
    fn from_table(
        inner: Box<dyn WeightSource>,
        table: DirectTable,
        config: &ModelConfig,
    ) -> Result<Self, String> {
        if config.engram_layer_ids.is_empty() {
            return Err(
                "--engram-dir: the served configuration declares no Engram layers".to_owned(),
            );
        }
        let width = config.engram_head_dim;
        let scale_width = width.div_ceil(SCALE_BLOCK);
        let mut routes = Vec::with_capacity(config.engram_layer_ids.len());
        for &layer_id in &config.engram_layer_ids {
            let Some(rows) = config.engram_rows(layer_id) else {
                continue;
            };
            let spec = (0..table.table_count())
                .filter_map(|index| table.spec(index))
                .find(|spec| spec.layer_id == layer_id)
                .ok_or_else(|| {
                    format!(
                        "--engram-dir: the packed store has no table for Engram layer {layer_id}"
                    )
                })?;
            if spec.logical_rows != rows as u64 {
                return Err(format!(
                    "--engram-dir: Engram layer {layer_id} has {} packed rows but the configuration says {rows}",
                    spec.logical_rows
                ));
            }
            // Reuse the serving translation's own validation for the width, so a configuration
            // whose head width does not fit the packed row is refused at start-up rather than
            // decoded wrongly at the first request.
            PagedTable::new(&table, layer_id, width, scale_width)
                .map_err(|error| format!("--engram-dir: {error}"))?;
            routes.push(EngramRoute {
                layer_id,
                width,
                scale_width,
            });
        }
        // A packed layer the configuration does not use would be dead storage and a sign the two
        // were produced from different checkpoints, so it is refused rather than ignored.
        for index in 0..table.table_count() {
            let Some(spec) = table.spec(index) else {
                continue;
            };
            if !config.engram_layer_ids.contains(&spec.layer_id) {
                return Err(format!(
                    "--engram-dir: the packed store holds layer {} but the configuration's Engram layers are {:?}",
                    spec.layer_id, config.engram_layer_ids
                ));
            }
        }
        Ok(Self {
            inner,
            table,
            routes,
            stats: Arc::new(EngramPagingStats::default()),
        })
    }

    /// The route for one layer, if the configuration declared it.
    fn route(&self, layer_id: usize) -> Option<EngramRoute> {
        self.routes
            .iter()
            .copied()
            .find(|route| route.layer_id == layer_id)
    }
}

/// The Engram tensor a stored name addresses, with its layer id.
///
/// Only the two tensors the paged path exists for are recognised; every other name is `None` and
/// is delegated to the wrapped source.
fn engram_tensor_name(name: &str) -> Option<(usize, EngramTensor)> {
    let rest = name.strip_prefix("layers.")?;
    let (layer, tail) = rest.split_once('.')?;
    let layer_id: usize = layer.parse().ok()?;
    match tail {
        "engram.embed.weight" => Some((layer_id, EngramTensor::Weight)),
        "engram.embed.scale" => Some((layer_id, EngramTensor::Scale)),
        _ => None,
    }
}

/// Reads the packer's manifest from a directory before the engine thread starts using it.
fn load_engram_manifest(dir: &Path) -> Result<PackManifest, String> {
    let path = dir.join(MANIFEST_NAME);
    let bytes = std::fs::read(&path).map_err(|error| {
        format!(
            "--engram-dir {}: cannot read {}: {error}",
            dir.display(),
            path.display()
        )
    })?;
    serde_json::from_slice(&bytes).map_err(|error| {
        format!(
            "--engram-dir {}: {} is not a usable packed-store manifest: {error}",
            dir.display(),
            path.display()
        )
    })
}

impl WeightSource for DiskEngramWeights {
    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.inner.spec(name)
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        // A whole-tensor read of an Engram table is exactly what this source exists to avoid, and
        // the checkpoint source refuses it; delegating keeps that refusal in one place.
        self.inner.read(name)
    }

    fn read_ranges(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
    ) -> Option<Result<Vec<u8>, WeightError>> {
        let Some((layer_id, tensor)) = engram_tensor_name(name) else {
            return self.inner.read_ranges(name, ranges);
        };
        let Some(route) = self.route(layer_id) else {
            return Some(Err(WeightError::Missing {
                name: name.to_owned(),
            }));
        };
        let paged = match PagedTable::new(&self.table, layer_id, route.width, route.scale_width) {
            Ok(paged) => paged,
            Err(error) => {
                return Some(Err(WeightError::Malformed {
                    name: name.to_owned(),
                    detail: error.to_string(),
                }));
            }
        };
        match paged.read(tensor, ranges) {
            Ok(read) => {
                let requests = match tensor {
                    EngramTensor::Weight => &self.stats.weight_requests,
                    EngramTensor::Scale => &self.stats.scale_requests,
                };
                requests.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .rows_requested
                    .fetch_add(read.rows as u64, Ordering::Relaxed);
                self.stats
                    .read_batches
                    .fetch_add(read.read_batches as u64, Ordering::Relaxed);
                self.stats
                    .bytes_requested
                    .fetch_add(read.data.len() as u64, Ordering::Relaxed);
                Some(Ok(read.data))
            }
            Err(error) => Some(Err(WeightError::Malformed {
                name: name.to_owned(),
                detail: error.to_string(),
            })),
        }
    }
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
    /// The configuration's vocabulary size, reported to the router so `logprobs=-1` and
    /// `logprob_token_ids` are validated against the model rather than guessed.
    vocab_size: usize,
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
                vocab_size: recipe.config.vocab_size,
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
            logprobs: request.logprobs,
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
                    Some(Reply::PromptLogprobs(prompt)) => {
                        Some((EngineEvent::PromptLogprobs(prompt), state))
                    }
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

    /// The configuration's vocabulary size: every logprob row this driver scores is that long.
    fn vocab_size(&self) -> Option<usize> {
        Some(self.vocab_size)
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
    /// The logprobs the request asked for; the default asks for none and costs nothing.
    logprobs: LogprobsRequest,
    /// Where the events go.
    replies: tokio::sync::mpsc::Sender<Reply>,
}

/// One element of the engine thread's answer.
enum Reply {
    /// A non-empty detokenised piece. The thread never sends an empty string.
    Delta(String),
    /// Prompt logprobs, sent once after `Started` and before any token.
    PromptLogprobs(Box<PromptLogprobs>),
    /// Logprobs for one generated token, sent after that token's delta.
    TokenLogprobs(Box<TokenLogprob>),
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
/// When the recipe carries an `--engram-dir`, the checkpoint source is wrapped so the two Engram
/// tables are paged from the packed dense store instead of being read as checkpoint tensors. The
/// wrapper serves every other tensor unchanged, so this is the only difference in the engine the
/// reference pass sees.
///
/// # Errors
///
/// A message naming the checkpoint, the packed Engram store or the configuration when the engine
/// cannot be built.
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
    let source: Box<dyn WeightSource> = match &recipe.engram_dir {
        Some(dir) => Box::new(DiskEngramWeights::open(source, dir, &recipe.config)?),
        None => source,
    };
    InferenceEngine::new(recipe.config.clone(), source, element_limit)
        .map_err(|error| format!("cannot build the reference engine: {error}"))
}

// ---------------------------------------------------------------------------------------
// Logprob arithmetic
// ---------------------------------------------------------------------------------------
//
// vLLM's default `logprobs_mode` is "raw_logprobs": a reported logprob is the log-softmax of the
// *raw* logits, before temperature and before any penalty. That is what this section computes from
// the `StepOutput.logits` the reference pass already returns (the raw logits of the position it
// just fed). The logits are f32; the softmax accumulates in f64 so a 129,280-entry row does not
// lose the sum to f32 rounding, and every entry is `logit - max - ln(sum)`, which makes the row a
// distribution even when the logits are large.
//
// Everything here is gated on the request: a request that asked for no logprobs runs the loop
// below with `want_sample` and `want_prompt` false, so no row is normalised and no string is
// decoded. A request for the whole vocabulary is O(vocabulary) per position by definition; a
// prompt-logprobs request costs one forward pass per prompt token (see
// [`prompt_logprobs_pass`]).

/// The log-softmax of one raw-logit row, in output-token-id order.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    // An all-masked row has no distribution; reporting the floor for every token is the only
    // honest answer, and the API layer clamps `-inf` to -9999 on the wire.
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

/// The 1-based competition rank of `token_logprob` in the row: one plus the number of tokens with a
/// strictly greater logprob, so tied tokens share a rank. That is vLLM's rank.
fn competition_rank(logprobs: &[f32], token_logprob: f32) -> u32 {
    let greater = logprobs
        .iter()
        .filter(|&&value| value > token_logprob)
        .count();
    u32::try_from(greater).unwrap_or(u32::MAX).saturating_add(1)
}

/// Descending logprob, ties broken by ascending token id.
///
/// The tie-break is what makes a tie deterministic: two tokens with the same logprob are reported
/// in token-id order, so the same request produces the same `top_logprobs` every time.
fn compare_logprobs(logprobs: &[f32], left: usize, right: usize) -> std::cmp::Ordering {
    logprobs[right]
        .partial_cmp(&logprobs[left])
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| left.cmp(&right))
}

/// The indices of the `k` largest logprobs, in descending order.
fn top_k_indices(logprobs: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(logprobs.len());
    if k == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..logprobs.len()).collect();
    if k < order.len() {
        order.select_nth_unstable_by(k, |left, right| compare_logprobs(logprobs, *left, *right));
        order.truncate(k);
    }
    order.sort_unstable_by(|left, right| compare_logprobs(logprobs, *left, *right));
    order
}

/// Collapses duplicate token ids the way vLLM's logprob dict does: the first insertion position is
/// kept and the later value overwrites it. Only the rank can differ for one id, so this is what
/// gives the sampled token the top-k positional rank when it also appears in the top-k.
fn collapse_entries(
    entries: Vec<sparkflash_api::LogprobAlternative>,
) -> Vec<sparkflash_api::LogprobAlternative> {
    let mut collapsed: Vec<sparkflash_api::LogprobAlternative> = Vec::with_capacity(entries.len());
    let mut positions: HashMap<u32, usize> = HashMap::with_capacity(entries.len());
    for entry in entries {
        match positions.get(&entry.token_id) {
            Some(&index) => collapsed[index] = entry,
            None => {
                positions.insert(entry.token_id, collapsed.len());
                collapsed.push(entry);
            }
        }
    }
    collapsed
}

/// Decodes token ids the way vLLM decodes a logprob token: one id at a time, with the preceding
/// sequence ids as context, because a byte-fallback token that completes a multi-byte UTF-8
/// character decodes to a replacement glyph on its own.
///
/// Single-token decodes that are already complete text are cached: they do not depend on the
/// context, and a top-k's alternatives recur at every position.
struct LogprobDecoder<'a> {
    /// Decodes a token-id slice to text. A closure rather than the tokenizer itself so the
    /// arithmetic and the correction are host-testable without a tokenizer file.
    decode_ids: &'a dyn Fn(&[u32]) -> Result<String, String>,
    /// The last (up to four) sequence ids before the position being decoded; vLLM's context depth.
    context: Vec<u32>,
    /// Complete single-token decodes, by token id.
    clean: HashMap<u32, String>,
}

impl<'a> LogprobDecoder<'a> {
    fn new(decode_ids: &'a dyn Fn(&[u32]) -> Result<String, String>) -> Self {
        Self {
            decode_ids,
            context: Vec::new(),
            clean: HashMap::new(),
        }
    }

    /// Decodes one id against the current context, without advancing the context.
    fn decode(&mut self, token_id: u32) -> Result<String, String> {
        if let Some(text) = self.clean.get(&token_id) {
            return Ok(text.clone());
        }
        let raw = (self.decode_ids)(&[token_id])?;
        if raw.ends_with('\u{FFFD}') {
            return self.corrected(token_id);
        }
        self.clean.insert(token_id, raw.clone());
        Ok(raw)
    }

    /// Makes `token_id` the newest context id: the next position's tokens are decoded after it.
    fn push(&mut self, token_id: u32) {
        self.context.push(token_id);
        if self.context.len() > 4 {
            self.context.remove(0);
        }
    }

    /// vLLM's byte-fallback correction: retry the decode with up to four preceding ids until the
    /// text no longer ends in a replacement glyph, then report the part of that decode the clean
    /// prefix did not already carry.
    fn corrected(&self, token_id: u32) -> Result<String, String> {
        let available = self.context.len().min(4);
        for count in 1..=available {
            let start = self.context.len() - count;
            let mut ids: Vec<u32> = self.context[start..].to_vec();
            ids.push(token_id);
            let Ok(full) = (self.decode_ids)(&ids) else {
                continue;
            };
            if full.ends_with('\u{FFFD}') {
                continue;
            }
            // Byte-fallback context tokens decode to a replacement glyph; the clean prefix is the
            // start of the trailing run that is not one of those.
            let mut clean_end = count;
            for index in (0..count).rev() {
                match (self.decode_ids)(&[self.context[start + index]]) {
                    Ok(text) if text.ends_with('\u{FFFD}') => clean_end = index,
                    _ => break,
                }
            }
            let clean_prefix = if clean_end == 0 {
                String::new()
            } else {
                (self.decode_ids)(&self.context[start..start + clean_end]).unwrap_or_default()
            };
            return Ok(match full.strip_prefix(clean_prefix.as_str()) {
                Some(rest) => rest.to_owned(),
                None => {
                    let common = common_prefix_bytes(&clean_prefix, &full);
                    full[common..].to_owned()
                }
            });
        }
        Ok(String::new())
    }
}

/// The length in bytes of the longest common prefix of two strings.
fn common_prefix_bytes(left: &str, right: &str) -> usize {
    left.chars()
        .zip(right.chars())
        .take_while(|(left, right)| left == right)
        .map(|(character, _)| character.len_utf8())
        .sum()
}

/// Builds one position's [`TokenLogprob`] from its log-softmax row.
fn token_logprob(
    logprobs: &[f32],
    token_id: u32,
    token: String,
    selection: Option<LogprobSelection>,
    token_ids: &[u32],
    decoder: &mut LogprobDecoder<'_>,
) -> Result<TokenLogprob, String> {
    let token_logprob = logprobs
        .get(token_id as usize)
        .copied()
        .unwrap_or(f32::NEG_INFINITY);
    let unranked = token_ids.is_empty() && matches!(selection, Some(LogprobSelection::All));
    let rank = if unranked {
        None
    } else {
        Some(competition_rank(logprobs, token_logprob))
    };
    let mut entries = vec![sparkflash_api::LogprobAlternative {
        token_id,
        token: token.clone(),
        logprob: token_logprob,
        rank,
    }];
    if !token_ids.is_empty() {
        // vLLM's `logprob_token_ids`: the sampled token first, then the requested ids in request
        // order, each with its positional rank.
        for (index, &requested) in token_ids.iter().enumerate() {
            let Some(&value) = logprobs.get(requested as usize) else {
                continue;
            };
            entries.push(sparkflash_api::LogprobAlternative {
                token_id: requested,
                token: decoder.decode(requested)?,
                logprob: value,
                rank: Some(u32::try_from(index + 1).unwrap_or(u32::MAX)),
            });
        }
    } else {
        match selection {
            // No alternatives: the position's own token is all that was asked for.
            None => {}
            Some(LogprobSelection::All) => {
                // The whole vocabulary, id-ordered and unranked: -1's "unsorted and unranked".
                entries.clear();
                for (index, &value) in logprobs.iter().enumerate() {
                    entries.push(sparkflash_api::LogprobAlternative {
                        token_id: u32::try_from(index).unwrap_or(u32::MAX),
                        token: decoder.decode(u32::try_from(index).unwrap_or(u32::MAX))?,
                        logprob: value,
                        rank: None,
                    });
                }
            }
            Some(LogprobSelection::Top(count)) => {
                for (position, index) in top_k_indices(logprobs, count as usize)
                    .into_iter()
                    .enumerate()
                {
                    entries.push(sparkflash_api::LogprobAlternative {
                        token_id: u32::try_from(index).unwrap_or(u32::MAX),
                        token: decoder.decode(u32::try_from(index).unwrap_or(u32::MAX))?,
                        logprob: logprobs[index],
                        rank: Some(u32::try_from(position + 1).unwrap_or(u32::MAX)),
                    });
                }
                entries = collapse_entries(entries);
            }
        }
    }
    Ok(TokenLogprob {
        token_id,
        token,
        logprob: token_logprob,
        rank,
        top: entries,
    })
}

/// Feeds the prompt one token at a time to score every position, returning the step output a batch
/// prefill would have returned.
///
/// This is the gated cost of prompt logprobs: the reference engine reports logits only for the
/// position it just fed, so scoring prompt position `i` needs the forward pass after
/// `tokens[..i]`. The prompt is therefore fed token by token - one forward pass per prompt token -
/// only when the request asked for prompt logprobs (or for `echo` with logprobs, which asks for
/// them implicitly). A device chain that keeps the KV cache and reports every position's logits in
/// one pass makes this one pass; until then this is the honest O(n) implementation, and the branch
/// document records the measured cost.
fn prompt_logprobs_pass(
    engine: &InferenceEngine,
    prompt: &[u32],
    state: &mut SequenceState,
    logprobs: &LogprobsRequest,
    decoder: &mut LogprobDecoder<'_>,
) -> Result<(StepOutput, PromptLogprobs), String> {
    let mut feed = engine
        .prefill(state, &prompt[..1])
        .map_err(|error| format!("prefill failed: {error}"))?;
    let mut tokens = Vec::with_capacity(prompt.len());
    let mut positions: Vec<Option<TokenLogprob>> = Vec::with_capacity(prompt.len());
    tokens.push(TokenString {
        token_id: prompt[0],
        token: decoder.decode(prompt[0])?,
    });
    decoder.push(prompt[0]);
    // Position 0 has no conditioning token, so it reports `None` exactly as vLLM does.
    positions.push(None);
    for (index, &token_id) in prompt.iter().enumerate().skip(1) {
        let row = log_softmax(&feed.logits);
        let token = decoder.decode(token_id)?;
        tokens.push(TokenString {
            token_id,
            token: token.clone(),
        });
        positions.push(Some(token_logprob(
            &row,
            token_id,
            token,
            logprobs.prompt,
            &[],
            decoder,
        )?));
        decoder.push(token_id);
        feed = engine
            .decode(state, token_id)
            .map_err(|error| format!("decode failed at prompt position {index}: {error}"))?;
    }
    Ok((feed, PromptLogprobs { tokens, positions }))
}

/// Feeds one prompt to the engine in the plan's chunks, returning the last chunk's step output.
///
/// The invariant this path relies on is asserted in
/// `crates/sparkflash-engine/tests/chunked_prefill.rs`: prefilling contiguous chunks on one
/// [`SequenceState`] is bitwise identical to prefilling the whole prompt in one call. The host
/// driver runs the chunks back to back - no decode between them, no second sequence - because the
/// reference engine's shape is one sequence at a time; the scheduler's inter-chunk decode window is
/// a device policy this pass does not claim.
fn prefill_chunked(
    engine: &InferenceEngine,
    state: &mut SequenceState,
    prompt: &[u32],
    plan: &ChunkedPrefillPlan,
) -> Result<StepOutput, String> {
    debug_assert_eq!(plan.prompt_tokens(), prompt.len());
    let mut last = None;
    for range in plan.ranges() {
        let chunk = prompt.get(range.clone()).ok_or_else(|| {
            format!(
                "chunk {range:?} lies outside the {} token prompt",
                prompt.len()
            )
        })?;
        last = Some(
            engine
                .prefill(state, chunk)
                .map_err(|error| format!("prefill failed: {error}"))?,
        );
    }
    last.ok_or_else(|| "the prompt is empty and has no prefill chunk".to_owned())
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
    let want_sample = job.logprobs.wants_sample();
    let want_prompt = job.logprobs.wants_prompt() && !job.prompt_tokens.is_empty();
    let decode_ids = |ids: &[u32]| -> Result<String, String> {
        tokenizer.tokenizer().decode(ids, false).map_err(|error| {
            format!(
                "the tokenizer could not decode {} token id(s): {error}",
                ids.len()
            )
        })
    };
    let mut logprob_decoder = LogprobDecoder::new(&decode_ids);
    let (prefill, prompt_logprobs) = if want_prompt {
        let (step, payload) = prompt_logprobs_pass(
            engine,
            &job.prompt_tokens,
            &mut state,
            &job.logprobs,
            &mut logprob_decoder,
        )?;
        (step, Some(payload))
    } else {
        // The chunked prefill policy: short prompts (any prompt under the smallest configured
        // chunk, 4096 tokens) are one chunk and the call is the pre-task-40 whole-prompt prefill; a
        // longer prompt is fed in policy chunks, which the host bits prove is the same function.
        let plan =
            ChunkedPrefillPlan::plan(job.prompt_tokens.len(), 0, &SchedulerConfig::default())
                .map_err(|error| format!("the chunked prefill plan could not be built: {error}"))?;
        (
            prefill_chunked(engine, &mut state, &job.prompt_tokens, &plan)?,
            None,
        )
    };
    let prefill_elapsed = request_started.elapsed();
    // Prompt logprobs are one payload for the prompt, delivered once at the end of prefill, which
    // is where vLLM delivers them.
    if let Some(payload) = prompt_logprobs {
        job.replies
            .blocking_send(Reply::PromptLogprobs(Box::new(payload)))
            .map_err(|_| "the client stopped reading this response".to_owned())?;
    }
    // The stream decoder keeps the state an incomplete UTF-8 sequence needs, so a token that only
    // completes half a character is held back instead of being emitted as a replacement character
    // that the next token would then contradict.
    let mut decoder = tokenizer.tokenizer().decode_stream(false);
    let mut logits = prefill.logits;
    let mut next = prefill.next_token;
    let mut completion_tokens = 0usize;
    let mut ttft: Option<Duration> = None;
    while completion_tokens < job.max_tokens {
        let Some(token) = next else { break };
        if ttft.is_none() {
            ttft = Some(request_started.elapsed());
        }
        let piece = decoder
            .step(token)
            .map_err(|error| format!("the tokenizer could not decode token {token}: {error}"))?
            .filter(|piece| !piece.is_empty());
        if want_sample {
            // The logprob belongs to the token the logits just produced, so it is built before that
            // token is fed back in.
            let row = log_softmax(&logits);
            let token_string = logprob_decoder.decode(token)?;
            let entry = token_logprob(
                &row,
                token,
                token_string,
                job.logprobs.sample,
                &job.logprobs.token_ids,
                &mut logprob_decoder,
            )?;
            if let Some(piece) = piece {
                job.replies
                    .blocking_send(Reply::Delta(piece))
                    .map_err(|_| "the client stopped reading this response".to_owned())?;
            }
            job.replies
                .blocking_send(Reply::TokenLogprobs(Box::new(entry)))
                .map_err(|_| "the client stopped reading this response".to_owned())?;
            logprob_decoder.push(token);
        } else if let Some(piece) = piece {
            job.replies
                .blocking_send(Reply::Delta(piece))
                .map_err(|_| "the client stopped reading this response".to_owned())?;
        }
        completion_tokens += 1;
        let step = engine.decode(&mut state, token).map_err(|error| {
            format!("decode failed after {completion_tokens} token(s): {error}")
        })?;
        logits = step.logits;
        next = step.next_token;
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

/// Wraps a device session in the engine and prints the report's startup line.
///
/// The report is derived from the session (fake/real, evidence or none), so this line cannot
/// overstate what ran; the caller only chooses which session to build.
fn device_engine<S: DeviceSession + 'static>(
    session: S,
    driver: ReferenceDriver,
) -> Result<Arc<dyn ApiInferenceEngine>, String> {
    let device = DeviceEngine::open(session, Arc::new(driver))
        .map_err(|error| format!("cannot open the device session: {error}"))?;
    println!("sparkflash serve: {}", device.report().startup_line());
    Ok(Arc::new(device))
}

/// The default tokenizer, resolved from where this process actually is.
///
/// The executable's directory comes first so a deployed binary can stage `reference/` next to
/// itself, then the current directory for a run started from the stage root. This deliberately does
/// not consult the build host's checkout path (see `resolve_reference_tokenizer`).
fn default_tokenizer() -> Result<PathBuf, String> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf));
    let cwd = std::env::current_dir().ok();
    resolve_reference_tokenizer(exe_dir.as_deref(), cwd.as_deref())
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
    let selection = match select_engine(
        options.engine,
        options.device_cubin.as_deref(),
        cfg!(feature = "cuda"),
    ) {
        Ok(selection) => selection,
        Err(message) => {
            eprintln!("sparkflash serve: {message}");
            return ExitCode::from(EXIT_REFUSED);
        }
    };
    let tokenizer = match options.tokenizer.clone() {
        Some(tokenizer) => tokenizer,
        None => match default_tokenizer() {
            Ok(tokenizer) => tokenizer,
            Err(message) => {
                eprintln!("sparkflash serve: {message}");
                return ExitCode::from(EXIT_REFUSED);
            }
        },
    };
    let recipe = match options.recipe(&tokenizer) {
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
    let engine: Arc<dyn ApiInferenceEngine> = match selection {
        DeviceSelection::Reference => Arc::new(driver),
        DeviceSelection::FakeSession => match device_engine(FakeSession::new(), driver) {
            Ok(engine) => engine,
            Err(message) => {
                eprintln!("sparkflash serve: {message}");
                return ExitCode::from(EXIT_REFUSED);
            }
        },
        DeviceSelection::CudaSession(cubin) => {
            #[cfg(feature = "cuda")]
            {
                match device_engine(CudaSession::new(cubin), driver) {
                    Ok(engine) => engine,
                    Err(message) => {
                        eprintln!("sparkflash serve: {message}");
                        return ExitCode::from(EXIT_REFUSED);
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = cubin;
                unreachable!("select_engine refuses a cubin without the cuda feature")
            }
        }
    };
    // The same tokenizer.json the engine detokenises with also drives the HTTP prompt seam:
    // /v1/completions text becomes the checkpoint's token ids and /v1/chat/completions renders
    // the released server's V4.1 template. Failing to load it refuses startup instead of serving
    // placeholder ids.
    let seam = match load_tokenizer(&tokenizer) {
        Ok(loaded) => Arc::new(TokenizerSeam::new(loaded)),
        Err(message) => {
            eprintln!("sparkflash serve: {message}");
            return ExitCode::from(EXIT_REFUSED);
        }
    };
    let state = AppState::new(engine)
        .with_max_logprobs(options.max_logprobs)
        .with_tokenizer(seam);
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
    use sparkflash_api::encoding::reference_tokenizer_path;

    /// A driver over the synthetic tiny plan: the path every behavioural test takes.
    fn synthetic_driver(max_tokens: usize) -> ReferenceDriver {
        let recipe = Recipe {
            config: ModelConfig::tiny(),
            weights: Weights::Synthetic {
                seed: SYNTHETIC_WEIGHT_SEED,
            },
            tokenizer: reference_tokenizer_path(),
            engram_dir: None,
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
            engram_dir: None,
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
            logprobs: LogprobsRequest::default(),
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

    /// The host driver's chunked prefill is the same function as one whole-prompt prefill: same
    /// state, same logits bit for bit, same next token. task-40's invariant, on the engine the
    /// serve path actually runs.
    #[tokio::test]
    async fn chunked_prefill_matches_unchunked_on_the_driver_engine() {
        let config = ModelConfig::tiny();
        let plan = TensorPlan::language_model(&config);
        let engine = InferenceEngine::new(
            config.clone(),
            Box::new(SyntheticWeights::from_plan(&plan, SYNTHETIC_WEIGHT_SEED)),
            REFERENCE_TENSOR_LIMIT,
        )
        .expect("the tiny engine builds");
        // Long enough to cross the tiny window and to split unevenly at most chunk sizes.
        let prompt: Vec<u32> = (1..=11).collect();
        let whole = engine
            .prefill(&mut SequenceState::new(&config), &prompt)
            .expect("the whole-prompt prefill runs");
        for chunk in [1usize, 2, 3, 4, 5, 8, 11] {
            let plan =
                ChunkedPrefillPlan::with_chunk_tokens(prompt.len(), 0, chunk, 0).expect("plan");
            let mut state = SequenceState::new(&config);
            let got =
                prefill_chunked(&engine, &mut state, &prompt, &plan).expect("chunked prefill");
            assert_eq!(
                got.logits
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                whole
                    .logits
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "logits differ at chunk size {chunk}"
            );
            assert_eq!(
                got.head_input
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                whole
                    .head_input
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "head input differs at chunk size {chunk}"
            );
            assert_eq!(
                got.next_token, whole.next_token,
                "next token at chunk size {chunk}"
            );
            assert_eq!(
                state.tokens(),
                prompt,
                "the state must hold the whole prompt"
            );
        }
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
    fn engram_dir_is_refused_with_the_synthetic_plan_and_carried_with_a_checkpoint() {
        let error = ServeOptions::parse(vec![
            "--synthetic".to_owned(),
            "--engram-dir".to_owned(),
            "/tmp/engram-tables".to_owned(),
        ])
        .expect_err("--synthetic has no checkpoint to page");
        assert!(error.contains("--engram-dir"), "{error}");
        assert!(error.contains("--checkpoint"), "{error}");

        let options = ServeOptions::parse(vec![
            "--checkpoint".to_owned(),
            "/tmp/checkpoint".to_owned(),
            "--engram-dir".to_owned(),
            "/tmp/engram-tables".to_owned(),
        ])
        .expect("a checkpoint run may page its Engram tables");
        assert_eq!(
            options.engram_dir,
            Some(PathBuf::from("/tmp/engram-tables")),
            "the flag reaches the recipe that build_engine reads"
        );
        assert!(
            options.checkpoint.is_some(),
            "the checkpoint is still the weight source; --engram-dir only changes where its Engram rows come from"
        );
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
        assert!(
            device.device_cubin.is_none(),
            "no cubin means the fake session"
        );

        let with_cubin = ServeOptions::parse(vec![
            "--synthetic".to_owned(),
            "--engine".to_owned(),
            "device".to_owned(),
            "--device-cubin".to_owned(),
            "out/moe_router_sm121.cubin".to_owned(),
        ])
        .expect("parses");
        assert_eq!(
            with_cubin.device_cubin,
            Some(PathBuf::from("out/moe_router_sm121.cubin"))
        );

        let error = ServeOptions::parse(vec![
            "--synthetic".to_owned(),
            "--engine".to_owned(),
            "gpu".to_owned(),
        ])
        .expect_err("an unknown engine is refused");
        assert!(error.contains("gpu"), "{error}");
    }

    #[test]
    fn the_engine_selection_is_pure_and_refuses_mismatches() {
        let cubin = Path::new("out/moe_router_sm121.cubin");
        assert_eq!(
            select_engine(EngineChoice::Reference, None, false).expect("reference"),
            DeviceSelection::Reference
        );
        assert_eq!(
            select_engine(EngineChoice::Device, None, false).expect("fake device"),
            DeviceSelection::FakeSession
        );
        assert_eq!(
            select_engine(EngineChoice::Device, Some(cubin), true).expect("cuda device"),
            DeviceSelection::CudaSession(PathBuf::from(cubin))
        );

        // A cubin means a device session, so it is not silently ignored on the reference path.
        let reference_error = select_engine(EngineChoice::Reference, Some(cubin), true)
            .expect_err("the reference engine has no device session");
        assert!(
            reference_error.contains("--device-cubin") && reference_error.contains("--engine"),
            "{reference_error}"
        );

        // A cubin without the feature cannot be honoured, and the message names the feature.
        let feature_error = select_engine(EngineChoice::Device, Some(cubin), false)
            .expect_err("the binary has no cuda feature");
        assert!(feature_error.contains("cuda"), "{feature_error}");
    }

    /// The exit code every pre-serve refusal returns; the server has not bound a socket.
    fn refused(code: ExitCode) -> bool {
        code == ExitCode::from(EXIT_REFUSED)
    }

    #[test]
    fn a_bad_engine_value_is_a_usage_refusal() {
        let code = run(vec![
            "--synthetic".to_owned(),
            "--engine".to_owned(),
            "gpu".to_owned(),
        ]);
        assert!(refused(code), "{code:?}");
    }

    #[test]
    fn a_cubin_with_the_reference_engine_is_refused() {
        let code = run(vec![
            "--synthetic".to_owned(),
            "--device-cubin".to_owned(),
            "out/moe_router_sm121.cubin".to_owned(),
        ]);
        assert!(refused(code), "{code:?}");
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn a_cubin_without_the_cuda_feature_is_refused() {
        let code = run(vec![
            "--synthetic".to_owned(),
            "--engine".to_owned(),
            "device".to_owned(),
            "--device-cubin".to_owned(),
            "out/moe_router_sm121.cubin".to_owned(),
        ]);
        assert!(refused(code), "{code:?}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn a_missing_cubin_with_the_cuda_feature_is_refused() {
        // The reference driver starts (synthetic, explicit tokenizer), then the device session
        // fails on the missing cubin before any driver is opened; the exit code is a refusal.
        let code = run(vec![
            "--synthetic".to_owned(),
            "--engine".to_owned(),
            "device".to_owned(),
            "--device-cubin".to_owned(),
            "/definitely/not/a/router.cubin".to_owned(),
            "--tokenizer".to_owned(),
            reference_tokenizer_path().to_string_lossy().into_owned(),
        ]);
        assert!(refused(code), "{code:?}");
    }

    #[test]
    fn an_explicit_tokenizer_overrides_the_default_resolution() {
        let options = ServeOptions::parse(vec![
            "--synthetic".to_owned(),
            "--tokenizer".to_owned(),
            "/models/deepseek/tokenizer.json".to_owned(),
        ])
        .expect("parses");
        assert_eq!(
            options.tokenizer,
            Some(PathBuf::from("/models/deepseek/tokenizer.json"))
        );
    }

    #[test]
    fn the_defaults_are_loopback_and_the_reference_caps() {
        let options = ServeOptions::parse(vec!["--synthetic".to_owned()]).expect("parses");
        assert_eq!(options.bind, SocketAddr::from(([127, 0, 0, 1], 8000)));
        assert_eq!(options.max_tokens, MODEL_MAX_OUTPUT_TOKENS);
        assert_eq!(options.element_limit, REFERENCE_TENSOR_LIMIT);
        assert!(
            options.tokenizer.is_none(),
            "no --tokenizer means resolve next to the executable, then the cwd"
        );
        assert!(options.device_cubin.is_none());
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

    #[test]
    fn the_checkpoint_cap_admits_the_released_wq_b_and_the_reference_cap_refuses_it() {
        // wq_b is [32768, 1280] in the released layer-0 attention projection: 41,943,040
        // elements. The checkpoint arm must admit it by default; the finite/synthetic guard must
        // still refuse it, which is the whole reason two caps exist.
        const RELEASED_WQ_B_ELEMENTS: usize = 32_768 * 1_280;
        assert_eq!(RELEASED_WQ_B_ELEMENTS, 41_943_040);
        assert!(RELEASED_WQ_B_ELEMENTS <= default_element_limit(Some(Path::new("/checkpoint"))));
        assert!(RELEASED_WQ_B_ELEMENTS > default_element_limit(None));
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

    // ---- logprob arithmetic (host only, no tokenizer file) -------------------------------

    /// A decoder that spells one id per token: the arithmetic tests need no tokenizer file.
    fn spelling_decoder(ids: &[u32]) -> Result<String, String> {
        Ok(ids
            .iter()
            .map(|id| format!("t{id}"))
            .collect::<Vec<_>>()
            .join("|"))
    }

    #[test]
    fn log_softmax_is_a_distribution_over_the_raw_logits() {
        let row = log_softmax(&[0.0, 0.0, 0.0]);
        let expected = -f64::from(3.0f32).ln() as f32;
        for value in &row {
            assert!((value - expected).abs() < 1e-6, "{value} != {expected}");
        }
        // A full-vocabulary row must still sum to one: the accumulation is f64, so the sum is not
        // lost to the row's width.
        let mut wide = vec![0.0f32; 129_280];
        wide[7] = 40.0;
        let total: f64 = log_softmax(&wide)
            .iter()
            .map(|value| f64::from(*value).exp())
            .sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "a 129,280-entry row sums to {total}"
        );
        // An all-masked row has no distribution: every entry is the floor, never a NaN.
        let masked = log_softmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]);
        assert!(masked.iter().all(|value| *value == f32::NEG_INFINITY));
    }

    #[test]
    fn ranks_are_vllms_competition_ranks() {
        let row = [0.0f32, -0.1, -0.1, -2.0];
        assert_eq!(competition_rank(&row, 0.0), 1);
        assert_eq!(
            competition_rank(&row, -0.1),
            2,
            "only the strictly greater token outranks it; the tie shares the rank"
        );
        assert_eq!(competition_rank(&row, -2.0), 4);
    }

    #[test]
    fn top_k_ties_break_by_token_id_so_the_answer_is_deterministic() {
        let row = [-1.0f32, -2.0, -2.0, -3.0];
        assert_eq!(top_k_indices(&row, 2), vec![0, 1]);
        for _ in 0..8 {
            assert_eq!(
                top_k_indices(&row, 2),
                vec![0, 1],
                "the tie-break is stable"
            );
        }
        assert_eq!(top_k_indices(&row, 0), Vec::<usize>::new());
        assert_eq!(
            top_k_indices(&row, 99),
            vec![0, 1, 2, 3],
            "asking for more than the vocabulary reports the whole vocabulary in order"
        );
    }

    #[test]
    fn a_natural_top_k_reports_the_sampled_token_first_then_the_alternatives() {
        let decode = spelling_decoder;
        let mut decoder = LogprobDecoder::new(&decode);
        let row = log_softmax(&[-1.0, -2.0, -3.0, -0.5]);
        let entry = token_logprob(
            &row,
            3,
            "t3".to_owned(),
            Some(LogprobSelection::Top(2)),
            &[],
            &mut decoder,
        )
        .expect("the entry builds");
        assert_eq!(entry.token_id, 3);
        assert_eq!(entry.rank, Some(1), "the sampled token is the argmax here");
        assert_eq!(entry.top[0].token_id, 3);
        assert_eq!(
            entry.top.len(),
            2,
            "the sampled token is also in the top-2, so it collapses to one entry and the map is \
             sampled + one alternative"
        );
        assert_eq!(entry.top[0].rank, Some(1));
        assert_eq!(entry.top[1].token_id, 0);
        assert_eq!(entry.top[1].rank, Some(2));

        let only_sampled = token_logprob(
            &row,
            1,
            "t1".to_owned(),
            Some(LogprobSelection::Top(0)),
            &[],
            &mut decoder,
        )
        .expect("the entry builds");
        assert_eq!(
            only_sampled.top.len(),
            1,
            "top_logprobs=0 is the token alone"
        );
        assert_eq!(only_sampled.top[0].token_id, 1);
        assert_eq!(only_sampled.rank, Some(3), "one token outranks it");
    }

    #[test]
    fn the_requested_token_ids_replace_the_natural_top_k() {
        let decode = spelling_decoder;
        let mut decoder = LogprobDecoder::new(&decode);
        let row = log_softmax(&[-1.0, -2.0, -3.0, -0.5]);
        let entry = token_logprob(
            &row,
            1,
            "t1".to_owned(),
            Some(LogprobSelection::Top(2)),
            &[2, 3],
            &mut decoder,
        )
        .expect("the entry builds");
        let ids: Vec<u32> = entry
            .top
            .iter()
            .map(|alternative| alternative.token_id)
            .collect();
        assert_eq!(
            ids,
            vec![1, 2, 3],
            "sampled first, then the requested ids in order"
        );
        assert_eq!(entry.top[1].rank, Some(1));
        assert_eq!(entry.top[2].rank, Some(2));
    }

    #[test]
    fn whole_vocabulary_logprobs_are_id_ordered_and_unranked() {
        let decode = spelling_decoder;
        let mut decoder = LogprobDecoder::new(&decode);
        let row = log_softmax(&[-1.0, -2.0, -3.0]);
        let entry = token_logprob(
            &row,
            2,
            "t2".to_owned(),
            Some(LogprobSelection::All),
            &[],
            &mut decoder,
        )
        .expect("the entry builds");
        assert_eq!(entry.top.len(), 3);
        assert_eq!(entry.rank, None, "vLLM's -1 is unranked");
        assert!(
            entry
                .top
                .iter()
                .all(|alternative| alternative.rank.is_none())
        );
        let ids: Vec<u32> = entry
            .top
            .iter()
            .map(|alternative| alternative.token_id)
            .collect();
        assert_eq!(ids, vec![0, 1, 2], "id order, not logprob order");
    }

    #[test]
    fn a_byte_fallback_token_gets_the_character_it_completes() {
        let decode = |ids: &[u32]| -> Result<String, String> {
            match ids {
                [7] => Ok("\u{FFFD}".to_owned()),
                [6] => Ok("\u{FFFD}".to_owned()),
                [6, 7] => Ok("\u{e9}".to_owned()),
                [8] => Ok("a".to_owned()),
                other => spelling_decoder(other),
            }
        };
        let mut decoder = LogprobDecoder::new(&decode);
        assert_eq!(
            decoder.decode(7).expect("decodes"),
            "",
            "a genuinely incomplete sequence has no text yet"
        );
        decoder.push(6);
        assert_eq!(
            decoder.decode(7).expect("decodes"),
            "\u{e9}",
            "with the preceding half in context the character is the correction, not a replacement"
        );
        assert_eq!(decoder.decode(8).expect("decodes"), "a");
        assert_eq!(
            decoder.decode(8).expect("decodes"),
            "a",
            "a clean decode is cached"
        );
    }

    // ---- logprobs through the reference driver -------------------------------------------

    fn logprobs_request(
        prompt: &[u32],
        max_tokens: usize,
        sample: LogprobSelection,
        prompt_logprobs: Option<LogprobSelection>,
    ) -> EngineRequest {
        let mut request = request(prompt, max_tokens, 0.0, 0);
        request.logprobs = LogprobsRequest {
            sample: Some(sample),
            prompt: prompt_logprobs,
            token_ids: Vec::new(),
        };
        request
    }

    fn token_logprobs(stream: &[EngineEvent]) -> Vec<TokenLogprob> {
        stream
            .iter()
            .filter_map(|event| match event {
                EngineEvent::TokenLogprobs(logprob) => Some((**logprob).clone()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn a_request_that_asked_for_no_logprobs_gets_no_logprob_events() {
        let driver = synthetic_driver(4);
        let stream = events(&driver, request(&[1, 2, 3], 4, 0.0, 0)).await;
        assert!(
            !stream.iter().any(|event| matches!(
                event,
                EngineEvent::TokenLogprobs(_) | EngineEvent::PromptLogprobs(_)
            )),
            "the logprob path must cost nothing when it was not asked for: {stream:?}"
        );
    }

    #[tokio::test]
    async fn prompt_logprobs_are_none_at_position_zero_and_delivered_after_started() {
        let driver = synthetic_driver(2);
        let stream = events(
            &driver,
            logprobs_request(
                &[1, 2, 3],
                2,
                LogprobSelection::Top(2),
                Some(LogprobSelection::Top(1)),
            ),
        )
        .await;
        let started = stream
            .iter()
            .position(|event| matches!(event, EngineEvent::Started { .. }))
            .expect("Started");
        let prompt = stream
            .iter()
            .position(|event| matches!(event, EngineEvent::PromptLogprobs(_)))
            .expect("prompt logprobs");
        assert_eq!(
            prompt,
            started + 1,
            "prompt logprobs arrive at the end of prefill"
        );
        let EngineEvent::PromptLogprobs(payload) = &stream[prompt] else {
            unreachable!()
        };
        assert_eq!(payload.positions.len(), 3);
        assert_eq!(payload.tokens.len(), 3);
        assert!(
            payload.positions[0].is_none(),
            "prompt position 0 has no conditioning token"
        );
        for position in &payload.positions[1..] {
            let entry = position.as_ref().expect("a scored position");
            assert_eq!(
                entry.top[0].token_id, entry.token_id,
                "the prompt token itself is the first dict entry"
            );
            assert!(entry.rank.unwrap_or(0) >= 1);
        }
    }

    #[tokio::test]
    async fn temperature_zero_logprobs_are_reproducible_and_cover_every_token() {
        let driver = synthetic_driver(3);
        let request = logprobs_request(&[1, 2, 3], 3, LogprobSelection::Top(2), None);
        let first = events(&driver, request.clone()).await;
        let second = events(&driver, request).await;
        let first_logprobs = token_logprobs(&first);
        assert_eq!(
            first_logprobs,
            token_logprobs(&second),
            "the same request must produce the same logprobs"
        );
        let completion_tokens = first
            .iter()
            .find_map(|event| match event {
                EngineEvent::Finished { usage, .. } => Some(usage.completion_tokens),
                _ => None,
            })
            .expect("Finished");
        assert_eq!(
            first_logprobs.len(),
            completion_tokens,
            "one logprob entry per generated token"
        );
        for entry in &first_logprobs {
            assert_eq!(entry.top[0].token_id, entry.token_id);
            assert_eq!(
                entry.rank,
                Some(1),
                "at temperature 0 the generated token is the raw-logit argmax"
            );
            assert!(!entry.top.is_empty());
        }
    }

    #[tokio::test]
    async fn whole_vocabulary_logprobs_cover_the_configured_vocabulary() {
        let driver = synthetic_driver(1);
        let stream = events(
            &driver,
            logprobs_request(&[1, 2, 3], 1, LogprobSelection::All, None),
        )
        .await;
        let entries = token_logprobs(&stream);
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].top.len(),
            ModelConfig::tiny().vocab_size,
            "logprobs=-1 reports the whole vocabulary"
        );
        let ids: Vec<u32> = entries[0]
            .top
            .iter()
            .map(|alternative| alternative.token_id)
            .collect();
        let mut ordered = ids.clone();
        ordered.sort_unstable();
        assert_eq!(ids, ordered, "the whole vocabulary is reported in id order");
        assert!(
            entries[0]
                .top
                .iter()
                .all(|alternative| alternative.rank.is_none()),
            "logprobs=-1 is unranked"
        );
        // The synthetic tiny plan's logits are not a distribution (they are NaN at this prompt), so
        // the honest row is the floor for every token. This pins that the whole-vocabulary path
        // reports the row it was given instead of inventing finite values; the distribution
        // property itself is pinned on a real logit row by `log_softmax_is_a_distribution_...`.
        assert!(
            entries[0]
                .top
                .iter()
                .all(|alternative| alternative.logprob == f32::NEG_INFINITY),
            "an all-NaN logit row reports the floor for every token"
        );
    }
}

#[cfg(test)]
mod engram_packed_store {
    //! The packed-store serving path, on a temporary fixture: what it reads, how much, and that
    //! it is the same arithmetic as the in-memory table.
    //!
    //! The fixture is the tiny configuration's real Engram table, obtained from
    //! [SyntheticWeights] and re-packed into the packer's 264-byte dense rows in a temporary
    //! file. The released 98 GB store is not needed to state the host properties: the packed
    //! layout, the reader and the translation are the same code at any row count, and a test
    //! that wrote the released geometry could not run here. The bytes the disk path returns are
    //! the synthetic table's own bytes, so a logits comparison against the in-memory path is a
    //! comparison of the two table *read* paths, not of two different models.

    use std::sync::atomic::AtomicU64;

    use sparkflash_api::encoding::reference_tokenizer_path;
    use sparkflash_engine::forward::SequenceState;
    use sparkflash_engram::layout::{ENGRAM_HEAD_DIM, ENGRAM_ROW_BYTES};
    use sparkflash_engram::packer::align_up;
    use sparkflash_engram::store::{TableSpec, TailFill};

    use super::*;

    /// An in-memory source that counts (and refuses) any whole-table Engram read, so the test can
    /// show the paged path never fell back to materialising a table.
    struct NoWholeTableEngramReads {
        inner: SyntheticWeights,
        engram_reads: Arc<AtomicU64>,
    }

    impl WeightSource for NoWholeTableEngramReads {
        fn names(&self) -> Vec<String> {
            self.inner.names()
        }

        fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
            self.inner.spec(name)
        }

        fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
            if engram_tensor_name(name).is_some() {
                self.engram_reads.fetch_add(1, Ordering::Relaxed);
                return Err(WeightError::Refused {
                    name: name.to_owned(),
                    detail: "the paged Engram path must not read the whole table".to_owned(),
                });
            }
            self.inner.read(name)
        }

        fn read_ranges(
            &self,
            name: &str,
            ranges: &[(u64, usize)],
        ) -> Option<Result<Vec<u8>, WeightError>> {
            if engram_tensor_name(name).is_some() {
                self.engram_reads.fetch_add(1, Ordering::Relaxed);
                return Some(Err(WeightError::Refused {
                    name: name.to_owned(),
                    detail: "the paged Engram path must not read the table as a tensor".to_owned(),
                }));
            }
            self.inner.read_ranges(name, ranges)
        }
    }

    /// A packed dense store holding the tiny configuration's layer-1 Engram table.
    struct PackedFixture {
        _dir: tempfile::TempDir,
        spec: TableSpec,
    }

    fn packed_fixture(config: &ModelConfig) -> PackedFixture {
        let plan = TensorPlan::language_model(config);
        let synthetic = SyntheticWeights::from_plan(&plan, SYNTHETIC_WEIGHT_SEED);
        let rows = config
            .engram_rows(1)
            .expect("the tiny configuration has layer 1");
        let width = config.engram_head_dim;
        let scale_width = width.div_ceil(SCALE_BLOCK);
        assert!(
            width <= ENGRAM_HEAD_DIM
                && scale_width <= sparkflash_engram::layout::ENGRAM_SCALE_BYTES
        );
        let weight = synthetic
            .read("layers.1.engram.embed.weight")
            .expect("the synthetic table's weights");
        let scale = synthetic
            .read("layers.1.engram.embed.scale")
            .expect("the synthetic table's scales");
        assert_eq!(weight.len(), rows * width);
        assert_eq!(scale.len(), rows * scale_width);
        let mut bytes = Vec::with_capacity(rows * ENGRAM_ROW_BYTES);
        for row in 0..rows {
            let mut dense = [0u8; ENGRAM_ROW_BYTES];
            dense[..width].copy_from_slice(&weight[row * width..(row + 1) * width]);
            dense[ENGRAM_HEAD_DIM..ENGRAM_HEAD_DIM + scale_width]
                .copy_from_slice(&scale[row * scale_width..(row + 1) * scale_width]);
            bytes.extend_from_slice(&dense);
        }
        bytes.resize(align_up(bytes.len() as u64) as usize, 0);
        let dir = tempfile::TempDir::new().expect("a temporary directory");
        let path = dir.path().join("engram-l1-r0of1.bin");
        std::fs::write(&path, &bytes).expect("write the packed fixture table");
        let spec = TableSpec {
            layer_id: 1,
            layer_index: 0,
            rank: 0,
            world_size: 1,
            path,
            row_start: 0,
            physical_rows: rows as u64,
            logical_rows: rows as u64,
            window_bytes: 512,
            tail_fill: TailFill::Reference,
        };
        PackedFixture { _dir: dir, spec }
    }

    #[test]
    fn the_disk_paged_path_returns_the_same_logits_as_the_in_memory_table() {
        let config = ModelConfig::tiny();
        let plan = TensorPlan::language_model(&config);
        let fixture = packed_fixture(&config);
        let tokens = [3u32, 4, 5, 6];

        let in_memory = InferenceEngine::new(
            config.clone(),
            Box::new(SyntheticWeights::from_plan(&plan, SYNTHETIC_WEIGHT_SEED)),
            REFERENCE_TENSOR_LIMIT,
        )
        .expect("the in-memory engine builds");
        let reference = in_memory
            .prefill(&mut SequenceState::new(&config), &tokens)
            .expect("the in-memory prefill runs");

        let table = DirectTable::open(vec![fixture.spec.clone()]).expect("open the packed fixture");
        let engram_reads = Arc::new(AtomicU64::new(0));
        let source = DiskEngramWeights::from_table(
            Box::new(NoWholeTableEngramReads {
                inner: SyntheticWeights::from_plan(&plan, SYNTHETIC_WEIGHT_SEED),
                engram_reads: engram_reads.clone(),
            }),
            table,
            &config,
        )
        .expect("the packed source wraps the engine's source");
        let stats = source.stats.clone();
        let disk = InferenceEngine::new(config.clone(), Box::new(source), REFERENCE_TENSOR_LIMIT)
            .expect("the disk-paged engine builds");
        let got = disk
            .prefill(&mut SequenceState::new(&config), &tokens)
            .expect("the disk-paged prefill runs");

        assert_eq!(
            got.logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            reference
                .logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "the two read paths must be bitwise identical, not merely close"
        );
        assert_eq!(
            got.head_input
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            reference
                .head_input
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(got.next_token, reference.next_token);
        assert_eq!(
            engram_reads.load(Ordering::Relaxed),
            0,
            "the engine asked the checkpoint source for no whole Engram table"
        );

        // What was asked for: one range request per stored tensor per layer per chunk, and one
        // batched row read per request.
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.weight_requests, 1, "one weight range request");
        assert_eq!(snapshot.scale_requests, 1, "one scale range request");
        assert_eq!(
            snapshot.read_batches, 2,
            "each range request became one batched row read"
        );
        assert_eq!(
            snapshot.rows_requested % 2,
            0,
            "the weight and scale requests name the same rows"
        );
        let distinct = snapshot.rows_requested / 2;
        let columns = (config.engram_max_ngram_size - 1) * config.engram_n_heads;
        assert!(
            (1..=tokens.len() * columns).contains(&(distinct as usize)),
            "{distinct} distinct rows for {} tokens x {columns} hash columns",
            tokens.len()
        );
        let width = config.engram_head_dim;
        let scale_width = width.div_ceil(SCALE_BLOCK);
        assert_eq!(
            snapshot.bytes_requested,
            distinct * (width + scale_width) as u64,
            "the bytes requested are exactly the rows' bytes, not a table"
        );
    }

    #[test]
    fn a_whole_table_engram_read_still_goes_to_the_inner_source_and_is_refused() {
        let config = ModelConfig::tiny();
        let fixture = packed_fixture(&config);
        let table = DirectTable::open(vec![fixture.spec.clone()]).expect("open the packed fixture");
        let engram_reads = Arc::new(AtomicU64::new(0));
        let plan = TensorPlan::language_model(&config);
        let source = DiskEngramWeights::from_table(
            Box::new(NoWholeTableEngramReads {
                inner: SyntheticWeights::from_plan(&plan, SYNTHETIC_WEIGHT_SEED),
                engram_reads: engram_reads.clone(),
            }),
            table,
            &config,
        )
        .expect("the packed source wraps the engine's source");
        let error = source
            .read("layers.1.engram.embed.weight")
            .expect_err("a whole-table read is refused");
        assert!(matches!(error, WeightError::Refused { .. }), "{error}");
        assert_eq!(
            engram_reads.load(Ordering::Relaxed),
            1,
            "the whole-table read reached the inner source's refusal"
        );
    }

    #[test]
    fn only_the_two_paged_engram_tensors_are_recognised() {
        assert_eq!(
            engram_tensor_name("layers.1.engram.embed.weight"),
            Some((1, EngramTensor::Weight))
        );
        assert_eq!(
            engram_tensor_name("layers.14.engram.embed.scale"),
            Some((14, EngramTensor::Scale))
        );
        for name in [
            "layers.1.engram.wkv.weight",
            "layers.1.attn.wq.weight",
            "layers.1.engram.embed.weight.extra",
            "embed.weight",
            "layers.x.engram.embed.weight",
        ] {
            assert_eq!(
                engram_tensor_name(name),
                None,
                "{name} is not a paged tensor"
            );
        }
    }

    #[test]
    fn an_engram_dir_reaches_the_weight_source_and_a_missing_manifest_is_named() {
        let config = ModelConfig::tiny();
        let missing = tempfile::TempDir::new().expect("a temporary directory");
        let recipe = Recipe {
            config,
            weights: Weights::Synthetic {
                seed: SYNTHETIC_WEIGHT_SEED,
            },
            tokenizer: reference_tokenizer_path(),
            engram_dir: Some(missing.path().to_path_buf()),
        };
        let error = match build_engine(&recipe, REFERENCE_TENSOR_LIMIT) {
            Ok(_) => panic!("a directory without a manifest cannot serve Engram rows"),
            Err(error) => error,
        };
        assert!(error.contains("--engram-dir"), "{error}");
        assert!(error.contains(MANIFEST_NAME), "{error}");
    }
}

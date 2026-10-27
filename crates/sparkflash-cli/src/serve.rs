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
//! token claim (design `docs/archive/build/device-forward-chain.md` 1.3, 7 M1, 7 M2). It is not the
//! default: `--engine reference` is, and that path is unchanged. The first real chain is M3.
//!
//! # What it does not claim
//!
//! This is the *reference* engine: `f32`, on the CPU, with every weight decoded through
//! [`sparkflash_engine::linalg::decode_by_name`] under an element limit. Its default engine thread
//! runs one sequence at a time; `--continuous-batching` opts into the S4 host scheduler described
//! below, which interleaves up to `--max-batch` admitted requests over that same single engine. It
//! is not a performance path and this module states no throughput figure for it. The numbers it
//! reports are the ones it measures for one request over that request's own wall clock, and every
//! field it does not measure is left at the zero it actually is, with the reason written beside it.
//! There is no cluster, no rank and no rail behind this process, so nothing here reports one.
//!
//! # Phase-1 host TP=4 (feature `tp4-serve`)
//!
//! With `--features tp4-serve`, `--tp4-rank`/`--tp4-world-size`/`--tp4-peers` select one process of
//! a four-rank host group. Rank 0 owns the HTTP listener, the tokenizer, the chat template and the
//! streaming response; ranks 1-3 run the same resident forward and never detokenize. Every rank
//! receives the same prompt ids once per request and the same rank-0-sampled token per step, so the
//! lockstep arithmetic is one sequence, not four. The control channel is a small serve-owned TCP
//! protocol (Hello/Ready/Request/Token/End/Error) on its own port; it is not the engine collective.
//!
//! Attention is replicated in TP mode: every rank keeps its own `SequenceState` and per-layer KV
//! copy for the sequence, so four ranks hold four KV copies (4x KV memory) while computing
//! identical arithmetic. There is no shared KV pool and no paged-serving claim here.
//!
//! Phase 1 is single-sequence and unbatched: one request runs at a time, with one more permitted to
//! queue behind it and any further concurrent request refused by the bounded job channel. The four
//! ranks must see the same token sequence, so batching a different request into this group would
//! desync them; that is a stated limitation, not an implicit queue policy. Prompt logprobs are
//! served through the task-18 all-position rank contract
//! ([tp4-prompt-logprobs-rank-contract.md](../../../docs/archive/build/tp4-prompt-logprobs-rank-contract.md)):
//! every rank screens its local LM-head shard and exchanges bounded candidate blocks over the
//! forward's collectives, and the deterministic host merge reports a position only when its f32
//! boundary proves the requested order. A top-k request is bounded (the HTTP cap is 20, the
//! contract's bound is 63 alternatives); the whole-vocabulary -1 selection and any position whose
//! order cannot be proven are refused with a typed message rather than approximated.
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
//! The default consequence is that requests are serialised: one sequence, one chunk at a time, one
//! job at a time. The opt-in `--continuous-batching` mode keeps the same Job/Reply channel contract
//! but admits up to `--max-batch` jobs FIFO and gives each admitted request one scheduler step per
//! round (one prefill, then one emitted token plus one decode). That is host serving behaviour on
//! the reference CPU engine; the device arm's batched attention, paged KV, TTFT and throughput are
//! window measurements and are not claimed here.
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

use std::collections::{HashMap, VecDeque};
#[cfg(feature = "tp4-serve")]
use std::io::{Read, Write};
use std::net::SocketAddr;
#[cfg(feature = "tp4-serve")]
use std::net::{IpAddr, TcpListener, TcpStream};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use futures::StreamExt;
use futures::stream::{self, BoxStream};
#[cfg(feature = "tp4-serve")]
use serde::{Deserialize, Serialize};
use sparkflash_api::device_chain::{ClaimMode, DeviceChainEnvelope};
use sparkflash_api::device_token::{
    DeviceChainTokenSource, DeviceTokenSource, LiveWindowTokenSource, TokenPieceDecoder,
};
use sparkflash_api::engine::InferenceEngine as ApiInferenceEngine;
use sparkflash_api::engine::{
    Capabilities, ClusterStatsSnapshot, EngineEvent, EngineRequest, FinishReason, LogprobSelection,
    LogprobsRequest, MODEL_ID, MODEL_MAX_OUTPUT_TOKENS, MissingCapability, PromptLogprobs,
    TokenLogprob, TokenString, default_engram_layers, empty_kv_snapshot, max_seq_len,
};
use sparkflash_api::state::CorsConfig;
use sparkflash_api::telemetry_types::{
    CompletionTokensDetails, DSparkStepStats, DeepSeekSparkRequestStats, OpenAiUsageWithCache,
    PromptTokensDetails, RequestTimingStats, RoceTransportStats, finite_or_zero,
};
use sparkflash_api::{AppState, Metrics};
#[cfg(feature = "cuda")]
use sparkflash_device::CudaSession;
use sparkflash_device::{DeviceEngine, DeviceSession, FakeSession};
use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::{
    InferenceEngine, SamplerOptions, Sampling, SequenceState, StepOutput,
};
use sparkflash_engine::linalg::{CHECKPOINT_TENSOR_LIMIT, REFERENCE_TENSOR_LIMIT, argmax};
use sparkflash_engine::loader::ReferenceWeights;
#[cfg(feature = "tp4-serve")]
use sparkflash_engine::prompt_ranks::MergedPosition;
use sparkflash_engine::synthetic::SyntheticWeights;
use sparkflash_engine::tensor::{TensorPlan, TensorSpec};
use sparkflash_engine::weights::{WeightError, WeightSource};
use sparkflash_engram::layout::SCALE_BLOCK;
use sparkflash_engram::packer::{MANIFEST_NAME, PackManifest};
use sparkflash_engram::paged::{EngramTensor, PagedTable};
use sparkflash_engram::store::{BackendPreference, DirectTable, RowTable, specs_from_manifest};
use sparkflash_kv::PrefixMatchPlan;
use sparkflash_model_deepseek_v41::encoding::{
    HfTokenizerEncoder, TokenizerSeam, load_hf_tokenizer, resolve_reference_tokenizer,
};
use sparkflash_scheduler::config::SchedulerConfig;
use sparkflash_scheduler::prefill::ChunkedPrefillPlan;

#[cfg(feature = "tp4-serve")]
mod device;
#[cfg(feature = "tp4-serve")]
pub use device::DeviceServeOptions;

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

/// The admission cap --continuous-batching uses when --max-batch is not given.
///
/// The queue is JOB_QUEUE jobs, and an admitted request holds one SequenceState; 32 keeps a host
/// scheduler round under a millisecond of bookkeeping while still covering the 32-client ladder
/// the serving census measures.
const DEFAULT_MAX_BATCH: usize = 32;

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
/// token claim). Design `docs/archive/build/device-forward-chain.md` 1.3, 7 M1, 7 M2.
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

impl DeviceSelection {
    /// The operator-facing name of this selection, for a boot line.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Reference => "--engine reference (CPU host twin)",
            Self::FakeSession => "--engine device (M1 fake session)",
            Self::CudaSession(_) => "--engine device (M2 cubin session)",
        }
    }
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

/// The control-channel error code for a deterministic TP=4 prompt-logprob contract refusal.
///
/// It is deliberately distinct from [ENGINE_ERROR_CODE]: every rank made the same decision after
/// the same collectives, so the group can keep serving and the workers resynchronise on the next
/// Request frame instead of exiting.
#[cfg(feature = "tp4-serve")]
const TP4_LOGPROBS_REFUSED_CODE: &str = "tp4_prompt_logprobs_refused";

/// The exit code for a refusal: bad arguments, an unreadable tokenizer, a directory that is not a
/// checkpoint. A runtime failure such as a bind error exits 1 instead.
const EXIT_REFUSED: u8 = 2;
/// The default number of ranks in the Phase-1 host TP=4 path.
#[cfg(feature = "tp4-serve")]
const TP4_DEFAULT_WORLD_SIZE: usize = 4;

/// The default first collective port: rank R binds this plus R.
#[cfg(feature = "tp4-serve")]
const TP4_DEFAULT_BASE_PORT: u16 = 29601;

/// The default control port is this far above the first collective port.
#[cfg(feature = "tp4-serve")]
const TP4_CONTROL_PORT_OFFSET: u16 = 100;

/// The one honest fabric label for the host TP=4 path. It is a TCP control channel plus the
/// engine's canonical host collective, not RoCE and not NCCL.
#[cfg(feature = "tp4-serve")]
const TP4_FABRIC: &str = "tcp tp4 host";

/// The Phase-1 job queue: one request may be queued behind the one in flight. A second concurrent
/// request is refused by the existing bounded queue rather than admitted, because the four ranks
/// must run the same token sequence in lockstep and batching one sequence would desync them.
#[cfg(feature = "tp4-serve")]
const TP4_JOB_QUEUE: usize = 1;

/// How long rank 0 waits for a missing rank, and how long every rank waits for the control peer.
///
/// This is a startup deadline, not a per-token deadline: open() reads the resident checkpoint and
/// can take minutes, so it deliberately sits outside the control channel. A missing rank is a
/// typed startup refusal, not a retry loop.
#[cfg(feature = "tp4-serve")]
const TP4_CONTROL_TIMEOUT: Duration = Duration::from_secs(600);

/// How long a rank's control socket may block on one frame.
///
/// The timeout is a liveness guard for a dead peer, not a performance budget: a step between
/// tokens can be tens of seconds on the reference forward. It is deliberately much larger than one
/// token.
#[cfg(feature = "tp4-serve")]
const TP4_CONTROL_READ_TIMEOUT: Duration = Duration::from_secs(600);

/// How long a worker pauses between control-connect attempts while rank 0's hub is still binding.
///
/// Rank 0 binds its control hub only after its own checkpoint load, but a worker connects as soon
/// as its load finishes, so a worker can legitimately arrive before the listener exists. Retrying
/// on connection-refused turns that startup race into a short wait instead of a failed start; the
/// retries stay inside the same startup budget rank 0's hub waits with.
#[cfg(feature = "tp4-serve")]
const TP4_CONTROL_CONNECT_RETRY: Duration = Duration::from_millis(100);

/// The Phase-1 TP=4 flags, parsed only when `--tp4-rank` is present.
///
/// The four ranks are four processes with the same binary. Rank 0 owns the HTTP listener, the
/// tokenizer and the chat-template seam; ranks 1-3 run the same resident forward and never
/// detokenize. Every rank opens the checkpoint once before the control handshake, which is why
/// the handshake deadline is minutes rather than seconds.
#[cfg(feature = "tp4-serve")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tp4ServeOptions {
    /// This process's rank, 0..world_size.
    pub rank: usize,
    /// The number of ranks in the group.
    pub world_size: usize,
    /// Every rank's address, in rank order; the control hub is `peers[0]`.
    pub peers: Vec<IpAddr>,
    /// Rank R binds the engine collective on `base_port + R`.
    pub base_port: u16,
    /// Rank 0 binds the serve-owned control channel here; workers connect to it.
    pub control_port: u16,
    /// The device forward with DSpark (`--tp4-device`) instead of the host twin.
    pub device: Option<DeviceServeOptions>,
}

#[cfg(feature = "tp4-serve")]
impl Tp4ServeOptions {
    /// Parses the TP=4 flags, or returns `None` when no TP=4 flag was given.
    ///
    /// # Errors
    ///
    /// A message naming the flag combination that is wrong: TP=4 without `--checkpoint`, with
    /// `--synthetic`, with `--engine device` or `--device-cubin`, a rank outside the world, a
    /// peer list of the wrong length, or a control port that collides with a collective port.
    // One argument per TP=4 CLI flag: keep the parse site shaped like the flag surface.
    #[allow(clippy::too_many_arguments)]
    fn from_cli(
        rank: Option<usize>,
        world_size: Option<usize>,
        peers: Option<&str>,
        base_port: Option<u16>,
        control_port: Option<u16>,
        synthetic: bool,
        checkpoint: Option<&Path>,
        engine: EngineChoice,
        device_cubin: Option<&Path>,
    ) -> Result<Option<Self>, String> {
        if rank.is_none()
            && world_size.is_none()
            && peers.is_none()
            && base_port.is_none()
            && control_port.is_none()
        {
            return Ok(None);
        }
        let rank = rank.ok_or_else(|| {
            "--tp4-rank is required when any --tp4-* flag is given; the TP=4 path is one process per rank"
                .to_owned()
        })?;
        if synthetic {
            return Err(
                "--synthetic cannot be combined with --tp4-rank: the TP=4 path serves the resident sharded checkpoint, and the synthetic tiny plan is a single-process path"
                    .to_owned(),
            );
        }
        if checkpoint.is_none() {
            return Err(
                "--tp4-rank requires --checkpoint DIR: the four ranks serve one released sharded checkpoint"
                    .to_owned(),
            );
        }
        if engine != EngineChoice::Reference || device_cubin.is_some() {
            return Err(
                "--tp4-rank is the host reference forward in four processes; it cannot be combined with --engine device or --device-cubin"
                    .to_owned(),
            );
        }
        let world_size = world_size.unwrap_or(TP4_DEFAULT_WORLD_SIZE);
        if world_size == 0 {
            return Err("--tp4-world-size needs at least one rank".to_owned());
        }
        if world_size > usize::from(u16::MAX) {
            return Err("--tp4-world-size is too large for a u16 port range".to_owned());
        }
        if rank >= world_size {
            return Err(format!(
                "--tp4-rank {rank} is outside a world of {world_size}"
            ));
        }
        let raw_peers = peers.ok_or_else(|| {
            format!(
                "--tp4-peers is required for --tp4-rank: give all {world_size} rank addresses in rank order, comma separated"
            )
        })?;
        let peers = parse_tp4_peers(raw_peers)?;
        if peers.len() != world_size {
            return Err(format!(
                "--tp4-peers lists {} address(es) but --tp4-world-size is {world_size}",
                peers.len()
            ));
        }
        let base_port = base_port.unwrap_or(TP4_DEFAULT_BASE_PORT);
        let last_collective = base_port
            .checked_add(u16::try_from(world_size - 1).map_err(|_| {
                "--tp4-world-size is too large for a u16 collective port range".to_owned()
            })?)
            .ok_or_else(|| {
                format!(
                    "--tp4-base-port {base_port} leaves no port for rank {}",
                    world_size - 1
                )
            })?;
        let control_port = match control_port {
            Some(port) => port,
            None => base_port
                .checked_add(TP4_CONTROL_PORT_OFFSET)
                .ok_or_else(|| {
                    format!(
                        "--tp4-base-port {base_port} leaves no default control port; pass --tp4-control-port"
                    )
                })?,
        };
        if (base_port..=last_collective).contains(&control_port) {
            return Err(format!(
                "--tp4-control-port {control_port} collides with the engine collective ports {base_port}..={last_collective}"
            ));
        }
        Ok(Some(Self {
            rank,
            world_size,
            peers,
            base_port,
            control_port,
            device: None,
        }))
    }
}

/// Parses a comma-separated peer list.
#[cfg(feature = "tp4-serve")]
fn parse_tp4_peers(raw: &str) -> Result<Vec<IpAddr>, String> {
    let mut peers = Vec::new();
    for item in raw.split(',') {
        let item = item.trim();
        if item.is_empty() {
            return Err("--tp4-peers contains an empty address".to_owned());
        }
        let address = item
            .parse::<IpAddr>()
            .map_err(|_| format!("--tp4-peers: {item} is not an IP address"))?;
        peers.push(address);
    }
    Ok(peers)
}

/// Which serve implementation this process runs.
///
/// The default is the single-process reference path. `Tp4` means one process in a four-rank group:
/// rank 0 owns HTTP, ranks 1-3 are workers, and every rank runs the same resident forward.
#[cfg(feature = "tp4-serve")]
#[derive(Debug, Clone)]
pub enum ServeEngine {
    /// The existing single-process reference server, unchanged.
    Reference,
    /// The Phase-1 host TP=4 path.
    Tp4(Tp4ServeOptions),
}

/// Chooses the serve implementation from the parsed options.
#[cfg(feature = "tp4-serve")]
fn serve_engine(options: &ServeOptions) -> ServeEngine {
    match &options.tp4 {
        Some(tp4) => ServeEngine::Tp4(tp4.clone()),
        None => ServeEngine::Reference,
    }
}

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
    /// Whether the reference engine thread runs the S4 host continuous-batching scheduler.
    ///
    /// Off is the serial path, byte for byte: one job at a time, the shape every existing serving
    /// test and claim was measured on. On is the batching scheduler in this module.
    pub continuous_batching: bool,
    /// How many requests the scheduler admits at once; only meaningful with
    /// [Self::continuous_batching], and 1 is a legal (degenerate) batch of one.
    pub max_batch: usize,
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
    /// The Phase-1 host TP=4 group, when `--tp4-rank` was given.
    ///
    /// `None` is the existing single-process server. `Some` means this process is one rank of a
    /// four-process group: rank 0 owns HTTP, every rank opens the resident checkpoint, and the
    /// ranks run the same token ids in lockstep over the engine collective plus a serve-owned
    /// control channel. See [ServeEngine].
    #[cfg(feature = "tp4-serve")]
    pub tp4: Option<Tp4ServeOptions>,
    /// The bearer key every route except `/health` and `/metrics` requires, when set.
    ///
    /// `None` is the default and keeps the server open, exactly as before. The key is never
    /// logged; a missing or wrong one gets the OpenAI 401 envelope.
    pub api_key: Option<String>,
    /// The browser cross-origin policy, or `None` to serve the pre-CORS surface.
    ///
    /// The default is [CorsConfig::any], the same permissive default the reference vLLM server
    /// mounts, so a browser client works without an extra flag.
    pub cors: Option<CorsConfig>,
    /// The recorded device token window to serve, when the operator passed --device-token-dir.
    pub device_token_dir: Option<PathBuf>,
    /// Whether a window without the writer's token_claimed assertion may be served (S7 rehearsal).
    pub device_token_derived_claim: bool,
    /// Whether `--device-token-dir` is a *live* directory to follow (W5) rather than one fixed
    /// window to load once.
    pub device_token_live: bool,
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
        // An empty key would authenticate only an empty bearer token; refuse it at parse time
        // instead of turning the server into something neither open nor protected.
        if cli
            .api_key
            .as_deref()
            .is_some_and(|key| key.trim().is_empty())
        {
            return Err(
                "--api-key must not be empty: an empty bearer token would be the only accepted one"
                    .to_owned(),
            );
        }
        // CORS: the default is the reference server's permissive policy. --no-cors serves the
        // pre-CORS bytes (a proxy may own the policy) and --cors-allow-origin restricts the origin
        // list instead of allowing any origin.
        let cors = if cli.no_cors {
            if !cli.cors_allow_origin.is_empty() {
                return Err(
                    "--no-cors and --cors-allow-origin cannot both be given: choose whether the CORS layer is served or who may cross origins"
                        .to_owned(),
                );
            }
            None
        } else if cli.cors_allow_origin.is_empty() {
            Some(CorsConfig::any())
        } else {
            Some(CorsConfig::allow_list(cli.cors_allow_origin.clone()))
        };
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
        // The batching scheduler is a reference-engine path: the device engine delegates its
        // generation elsewhere and carries its own limitations, and the TP=4 group deliberately
        // runs one job at a time so the four ranks see the same sequence. Both are refused rather
        // than silently running serial.
        if cli.continuous_batching && cli.engine == EngineChoice::Device {
            return Err(
                "--continuous-batching serves the reference engine path only: the device engine's generation path is unchanged and the scheduler state is not wired through it"
                    .to_owned(),
            );
        }
        #[cfg(feature = "tp4-serve")]
        if cli.continuous_batching && cli.tp4_rank.is_some() {
            return Err(
                "--continuous-batching cannot be combined with --tp4-rank: the TP=4 group runs one request at a time so every rank sees the same token sequence, and batching a different request into the group would desync them"
                    .to_owned(),
            );
        }
        let max_batch = match cli.max_batch {
            Some(0) => {
                return Err(
                    "--max-batch must be at least 1: a batch of zero would admit nothing"
                        .to_owned(),
                );
            }
            Some(value) if value > JOB_QUEUE => {
                return Err(format!(
                    "--max-batch {value} exceeds the {JOB_QUEUE}-request job queue, so the scheduler could never fill it"
                ));
            }
            Some(_) if !cli.continuous_batching => {
                return Err(
                    "--max-batch needs --continuous-batching: without the scheduler the engine thread runs one request at a time and there is no batch to cap"
                        .to_owned(),
                );
            }
            Some(value) => value,
            None => DEFAULT_MAX_BATCH,
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
        #[cfg(feature = "tp4-serve")]
        let tp4 = {
            let mut tp4 = Tp4ServeOptions::from_cli(
                cli.tp4_rank,
                cli.tp4_world_size,
                cli.tp4_peers.as_deref(),
                cli.tp4_base_port,
                cli.tp4_control_port,
                cli.synthetic,
                checkpoint.as_deref(),
                cli.engine,
                cli.device_cubin.as_deref(),
            )?;
            if let Some(tp4) = tp4.as_mut() {
                tp4.device = DeviceServeOptions::from_cli(
                    cli.tp4_device,
                    cli.max_model_len,
                    cli.tp4_evidence_dir,
                )?;
            }
            tp4
        };
        if cli.device_token_derived_claim && cli.device_token_dir.is_none() {
            return Err(
                "--device-token-derived-claim needs --device-token-dir: the flag only changes how a recorded window's claim is established"
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
                continuous_batching: cli.continuous_batching,
                max_batch,
                element_limit,
                engine: cli.engine,
                device_cubin: cli.device_cubin,
                engram_dir: cli.engram_dir,
                #[cfg(feature = "tp4-serve")]
                tp4,
                api_key: cli.api_key,
                cors,
                device_token_dir: cli.device_token_dir,
                device_token_derived_claim: cli.device_token_derived_claim,
                device_token_live: cli.device_token_live,
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
                batch: None,
            },
            (false, Some(path)) => checkpoint_recipe(path, tokenizer)?,
            (false, None) => return Err(no_weights_message()),
        };
        // The parse already refused --engram-dir with --synthetic; this is where the checkpoint
        // recipe learns where its Engram rows come from.
        recipe.engram_dir = self.engram_dir.clone();
        recipe.batch = self.continuous_batching.then_some(self.max_batch);
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
    /// Require a bearer API key on every route except /health, /readyz and /metrics.
    ///
    /// Without it (and without SPARKFLASH_SERVE_API_KEY) the server is open, exactly as it was
    /// before auth existed. With it, a missing or wrong Authorization: Bearer header is answered
    /// with the OpenAI 401 envelope; /health, /readyz and /metrics stay reachable for liveness
    /// probes, readiness probes and local scrapers.
    #[arg(long, value_name = "KEY", env = "SPARKFLASH_SERVE_API_KEY")]
    api_key: Option<String>,
    /// Allow browser requests only from this origin; repeat for several. Default: any origin.
    ///
    /// The default matches the reference vLLM server, which mounts a CORS middleware answering
    /// `access-control-allow-origin: *`. An explicit list answers only those origins with an echoed
    /// allow-origin header and `Vary: Origin`.
    #[arg(long, value_name = "ORIGIN", action = clap::ArgAction::Append)]
    cors_allow_origin: Vec<String>,
    /// Serve without the CORS layer: no `access-control-*` headers and no preflight answer.
    ///
    /// This is the pre-CORS wire surface, for a deployment behind a proxy that adds its own
    /// policy. It cannot be combined with --cors-allow-origin.
    #[arg(long)]
    no_cors: bool,
    /// Serve a claimed four-rank device token envelope from this window directory.
    ///
    /// The directory is a recorded token window (rank{rank}/tp4-forward-embed-r{rank}.json and
    /// tp4-token-r{rank}.json, or the same files flat). A plain greedy request whose prompt is the
    /// recorded prompt ids is answered from the envelope and labelled with the UNMEASURED device
    /// provenance headers; every other request falls back to the engine exactly as before. The
    /// window's claim state is printed at startup, so a refused window is visible instead of
    /// silently serving.
    #[arg(long, value_name = "DIR")]
    device_token_dir: Option<PathBuf>,
    /// Follow --device-token-dir as a live window directory instead of loading one fixed window.
    ///
    /// With this flag the directory is a *parent* to watch: the source picks the newest completed
    /// `tp4-token-r*.json` set under it, re-reads when the run id or the rank-0 token file changes,
    /// and reloads only on that transition. A request is served only when its prompt ids match the
    /// active window exactly, so a window that is still being written, or one recorded for a
    /// different prompt, falls back to the host twin rather than answering with the wrong tokens.
    /// Without the flag the directory must already hold one complete claimed window, as before.
    #[arg(long, requires = "device_token_dir")]
    device_token_live: bool,
    /// Accept a window whose writer did not assert token_claimed, deriving the claim from the
    /// complete four-rank loop evidence (consensus, per-position claimable, nine reference
    /// matches, fed forward_ids). This is the S7 rehearsal form; it is not the S8 acceptance form.
    #[arg(long)]
    device_token_derived_claim: bool,
    /// Admit several requests to one reference engine and interleave their steps.
    ///
    /// This is the S4 host continuous-batching scheduler: a FIFO admission queue under
    /// --max-batch, one prefill per admitted request, then one emitted token and one decode step
    /// per active request per scheduler step. It is the reference CPU engine, so it is a host
    /// serving-behaviour seam and a measured diagnostic path, not a device throughput claim.
    /// Off by default: without it the engine thread runs one request at a time, exactly as before.
    #[arg(long)]
    continuous_batching: bool,
    /// The most requests the continuous-batching scheduler admits at once (requires
    /// --continuous-batching).
    #[arg(long, value_name = "N")]
    max_batch: Option<usize>,
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
    /// The TP=4 rank of this process. Any `--tp4-*` flag switches to the Phase-1 host TP=4 path.
    ///
    /// Rank 0 owns HTTP, the tokenizer and the chat template; ranks 1-3 run the resident forward
    /// and never detokenize. Every rank opens the checkpoint before the control handshake.
    #[cfg(feature = "tp4-serve")]
    #[arg(long, value_name = "N")]
    tp4_rank: Option<usize>,
    /// The number of ranks in the TP=4 group; default 4.
    #[cfg(feature = "tp4-serve")]
    #[arg(long, value_name = "N")]
    tp4_world_size: Option<usize>,
    /// Every rank's IP address in rank order, comma separated (required with `--tp4-rank`).
    #[cfg(feature = "tp4-serve")]
    #[arg(long, value_name = "IP,IP,...")]
    tp4_peers: Option<String>,
    /// Rank R binds the engine collective on this port plus R; default 29601.
    #[cfg(feature = "tp4-serve")]
    #[arg(long, value_name = "PORT")]
    tp4_base_port: Option<u16>,
    /// The serve-owned control channel rank 0 binds; default base_port + 100.
    #[cfg(feature = "tp4-serve")]
    #[arg(long, value_name = "PORT")]
    tp4_control_port: Option<u16>,
    /// Serve the device forward with DSpark speculative decoding from these kernel cubins.
    ///
    /// Every rank runs the `tp4-chain` device epoch (NCCL over the RoCE rail, the resident arena,
    /// the on-device drafter) and serves one request at a time through it; requests queue behind
    /// the one on the device. Without the flag the TP=4 group is the host twin.
    #[cfg(feature = "tp4-serve")]
    #[arg(long, value_name = "DIR", requires = "tp4_rank")]
    tp4_device: Option<PathBuf>,
    /// The longest prompt plus completion the device path serves; default and ceiling 1,048,576.
    #[cfg(feature = "tp4-serve")]
    #[arg(long, value_name = "N", requires = "tp4_device")]
    max_model_len: Option<usize>,
    /// Where each rank's device epoch writes its bind evidence; default the current directory.
    #[cfg(feature = "tp4-serve")]
    #[arg(long, value_name = "DIR", requires = "tp4_device")]
    tp4_evidence_dir: Option<PathBuf>,
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
    /// The continuous-batching admission cap, or None for the serial one-job-at-a-time loop.
    batch: Option<usize>,
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
    /// Test-only: every weight byte is zero, so the forward's logits are finite (all-zero) and the
    /// acceptance rule has a genuinely measurable served path without a checkpoint.
    #[cfg(test)]
    Zero,
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
        batch: None,
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

/// Splits a decoded stream into the pieces a device window's token ids contribute, in order.
///
/// The seam decodes the ids it is given; a piece is the text decode(prefix + [id]) adds over
/// decode(prefix). A tokenizer whose decode is not prefix-monotonic (a byte-level BPE that
/// rewrites the previous bytes) refuses the piece instead of serving text that is not the decode
/// of the claimed ids.
struct SeamPieceDecoder {
    seam: Arc<TokenizerSeam>,
}

impl TokenPieceDecoder for SeamPieceDecoder {
    fn piece(&self, ids_so_far: &[u32], token_id: u32) -> Result<String, String> {
        let mut ids = Vec::with_capacity(ids_so_far.len() + 1);
        ids.extend_from_slice(ids_so_far);
        ids.push(token_id);
        let full = self
            .seam
            .decode(&ids)
            .ok_or_else(|| format!("the tokenizer could not decode ids {ids:?}"))?;
        let prefix = if ids_so_far.is_empty() {
            String::new()
        } else {
            self.seam
                .decode(ids_so_far)
                .ok_or_else(|| format!("the tokenizer could not decode ids {ids_so_far:?}"))?
        };
        match full.strip_prefix(prefix.as_str()) {
            Some(piece) => Ok(piece.to_owned()),
            None => Err(format!(
                "the tokenizer's decode is not prefix-monotonic at {} ids: {:?} does not start with {:?}",
                ids_so_far.len(),
                full,
                prefix
            )),
        }
    }
}

/// Builds the device token source when --device-token-dir was given, printing its claim state.
///
/// The window is loaded once at startup: a schema error refuses startup (a typo or a wrong window
/// must not silently serve the host twin), while a refused claim only makes every eligible request
/// fall back to the host twin, which is the fail-closed contract. The claim state is printed so the
/// operator sees which form the S7 probe will exercise.
fn device_token_source(
    options: &ServeOptions,
    seam: &Arc<TokenizerSeam>,
) -> Result<Option<Arc<dyn DeviceTokenSource>>, String> {
    let Some(dir) = options.device_token_dir.as_deref() else {
        return Ok(None);
    };
    if !dir.is_dir() {
        return Err(format!(
            "--device-token-dir {}: not a directory",
            dir.display()
        ));
    }
    let mode = if options.device_token_derived_claim {
        ClaimMode::Derived
    } else {
        ClaimMode::Asserted
    };
    let decoder: Arc<dyn TokenPieceDecoder> = Arc::new(SeamPieceDecoder {
        seam: Arc::clone(seam),
    });
    if options.device_token_live {
        eprintln!(
            "sparkflash serve: device token source {}: live window follower ({} claim), serving a request only when its prompt ids match the active window",
            dir.display(),
            mode.label()
        );
        return Ok(Some(Arc::new(
            LiveWindowTokenSource::new(dir, decoder).with_claim_mode(mode),
        )));
    }
    let envelope = DeviceChainEnvelope::load(dir)
        .map_err(|error| format!("--device-token-dir {}: {error}", dir.display()))?;
    match envelope.claim_with(mode) {
        Ok(claimed) => eprintln!(
            "sparkflash serve: device token source {}: {} claim ok ({} tokens, prompt_len {}, dump {})",
            dir.display(),
            mode.label(),
            claimed.tokens.len(),
            claimed.metadata.prompt_len,
            if claimed.logits_dump.is_some() {
                "present"
            } else {
                "absent"
            }
        ),
        Err(refusal) => eprintln!(
            "sparkflash serve: device token source {}: {} claim refused, serving the host twin for every request: {refusal}",
            dir.display(),
            mode.label()
        ),
    }
    Ok(Some(Arc::new(
        DeviceChainTokenSource::new(dir, decoder).with_claim_mode(mode),
    )))
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
    /// Whether the engine thread runs the continuous-batching scheduler, for the queue-refusal text.
    batched: bool,
    /// The exporter the engine thread feeds as requests are admitted, generate and finish.
    ///
    /// The router serves this same registry (`AppState::with_metrics`), so the scheduler gauges,
    /// the generator counter and the TTFT histogram describe this driver's real requests instead
    /// of a second, empty registry.
    metrics: Arc<Metrics>,
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
        // The exporter is created here, before the thread, so the engine thread and the router can
        // share one registry: the driver's own request lifecycle is what feeds the scheduler
        // gauges, the generator counter and the TTFT histogram.
        let metrics = Arc::new(Metrics::new());
        let thread_metrics = Arc::clone(&metrics);
        std::thread::Builder::new()
            .name(ENGINE_THREAD_NAME.to_owned())
            .spawn(move || engine_thread(&owned, element_limit, queued, ready, thread_metrics))
            .map_err(|error| format!("cannot start the engine thread: {error}"))?;
        match reported.recv() {
            Ok(Ok(())) => Ok(Self {
                model: MODEL_ID.to_owned(),
                jobs,
                max_tokens,
                vocab_size: recipe.config.vocab_size,
                cluster: cluster_snapshot(),
                batched: recipe.batch.is_some(),
                metrics,
            }),
            Ok(Err(message)) => Err(message),
            Err(_) => Err("the engine thread stopped before it reported ready".to_owned()),
        }
    }
}

impl ReferenceDriver {
    /// The exporter this driver's request lifecycle feeds.
    ///
    /// `sparkflash serve` hands the same registry to the router, so /metrics reports the
    /// scheduler gauges, generator counter and TTFT histogram of the requests this driver ran.
    #[must_use]
    pub fn metrics(&self) -> Arc<Metrics> {
        Arc::clone(&self.metrics)
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
        let enqueued_at = Instant::now();
        let (replies, answers) = tokio::sync::mpsc::channel(REPLY_BUFFER);
        let job = Job {
            prompt_tokens: request.prompt_tokens,
            max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            repetition_penalty: request.repetition_penalty,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            skip_special_tokens: request.skip_special_tokens,
            seed: request.seed.unwrap_or(0),
            logprobs: request.logprobs,
            stop: request.stop,
            enqueued_at,
            replies,
        };
        // The request is queued from this moment; the engine thread moves it from waiting to
        // running when it admits it. A refused enqueue never entered the queue, so the record is
        // taken back instead of leaving a phantom waiting request behind.
        self.metrics.request_enqueued();
        if self.jobs.try_send(job).is_err() {
            self.metrics.request_enqueue_refused();
            let message = if self.batched {
                "the engine queue is full: the continuous-batching scheduler admits up to --max-batch requests and the job queue is bounded, so retry when the requests in flight have finished"
            } else {
                "the engine queue is full: the reference engine runs one request at a time and the queue is bounded, so retry when the requests in flight have finished"
            };
            return refused(message, REFUSED_CODE);
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
    finished_event_with_fabric(prompt_tokens, summary, "")
}

/// The terminal event with an explicit fabric label for the stats block.
fn finished_event_with_fabric(
    prompt_tokens: usize,
    summary: RunSummary,
    fabric_mode: &str,
) -> EngineEvent {
    let completion_tokens = summary.completion_tokens;
    let dspark = summary.dspark;
    let mut stats = request_stats_with_fabric(prompt_tokens, summary.timing, fabric_mode);
    // The reference driver counts the engine's real MTP drafts against the target tokens it
    // actually emitted; the TP=4 driver has no draft source yet and carries the default zero.
    stats.dspark_speculative = dspark;
    EngineEvent::Finished {
        finish_reason: summary.finish_reason,
        usage: OpenAiUsageWithCache {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: PromptTokensDetails {
                // This process has no KV pool and no persistent tier, so nothing was served from a
                // cache, and the host paths refuse images. These zeros are the state of the world,
                // not a missing measurement; the device driver sets its prefix and image counts.
                cached_tokens: 0,
                uma_cached_tokens: 0,
                nvme_cached_tokens: 0,
                image_tokens: 0,
            },
            completion_tokens_details: CompletionTokensDetails {
                // The MTP drafter proposed real draft tokens and the acceptance rule above replayed
                // the target against them. A gated candidate is counted as rejected here because it
                // was proposed and not accepted; the confidence-gated subtotal is in the DSpark
                // telemetry block instead.
                reasoning_tokens: 0,
                accepted_prediction_tokens: dspark.draft_tokens_accepted,
                rejected_prediction_tokens: dspark
                    .draft_tokens_proposed
                    .saturating_sub(dspark.draft_tokens_accepted),
            },
            prompt_cache_hit_tokens: 0,
            prompt_cache_miss_tokens: prompt_tokens,
        },
        stats: Box::new(stats),
    }
}

/// The per-request telemetry, with every unmeasured field left at zero and the reason beside it.
///
/// The filled fields are the token counts taken from the usage block and the three wall-clock
/// numbers in `timing`, all of which this process measured for this one request. Nothing here is an
/// estimate, a projection or another machine's number.
/// The per-request telemetry with an explicit fabric label; the reference path passes "" so its
/// existing bytes do not change.
fn request_stats_with_fabric(
    prompt_tokens: usize,
    timing: RequestTiming,
    fabric_mode: &str,
) -> DeepSeekSparkRequestStats {
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
        // The reference path overwrites this with the request's measured MTP acceptance counters;
        // the TP=4 path has no draft source and keeps the honest default zero.
        dspark_speculative: DSparkStepStats::default(),
        // The reference path passes "" (its existing bytes); the TP=4 path passes its honest
        // fabric label. No rail byte counter or NCCL call is invented on either path.
        transport: RoceTransportStats {
            fabric_mode: fabric_mode.to_owned(),
            ..RoceTransportStats::default()
        },
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

// ---------------------------------------------------------------------------------------
// TP=4 serve control channel
// ---------------------------------------------------------------------------------------
//
// The engine collective and the serve control channel are deliberately separate. The engine's
// TcpTransport carries only its data collective frames (Hello/Data/Shutdown) and its send/recv
// are not part of the Transport trait; the serve protocol is a small length-prefixed JSON frame
// stream on its own port. Rank 0 is the hub: it accepts every worker's Hello, sends Ready only
// after all ranks declared ready, then broadcasts Request/Token/End or an Error. Workers never
// detokenize and never sample: they receive the prompt ids once per request and the sampled token
// ids one per step, and each runs the same forward step on its own SequenceState.

/// The largest control frame accepted: a device tick's prompt id lists (rank 0 stops admitting at
/// [`TP4_TICK_PROMPT_IDS`] ids) and a small error message fit below this. A frame larger than this
/// is a protocol error, not an allocation request.
#[cfg(feature = "tp4-serve")]
const TP4_CONTROL_MAX_FRAME: usize = 8 << 20;

/// The largest canvas a [`Tp4ControlFrame::DevicePixels`] frame carries: the largest span
/// (`vision_max_n_token` tokens) is about 1024 cells of 42 x 42 RGB pixels, 5.4 MB.
#[cfg(feature = "tp4-serve")]
const TP4_CONTROL_MAX_PIXELS: usize = 16 << 20;

/// The prompt ids one device tick's admissions carry at most (about 7 bytes each in JSON): a
/// request past this joins at the next tick.
#[cfg(feature = "tp4-serve")]
pub(crate) const TP4_TICK_PROMPT_IDS: usize = 768 << 10;

/// One message on the serve-owned control channel.
#[cfg(feature = "tp4-serve")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tp4ControlFrame {
    /// A worker's first frame: which rank this connection is.
    Hello {
        /// The worker rank, 1..world_size.
        rank: u32,
    },
    /// Rank 0 -> worker: every rank has declared ready; requests may start.
    Ready,
    /// Rank 0 -> every worker: one request's prompt ids and generation budget.
    Request {
        /// The tokens rank 0 rendered (the chat template is already applied).
        prompt_ids: Vec<u32>,
        /// The request's clamped generation budget.
        max_tokens: u32,
        /// The prompt-logprobs top count the request asked for, when it asked for any.
        ///
        /// `None` is "no prompt logprobs"; `Some(0)` is "each prompt position's own token
        /// only". The whole-vocabulary -1 selection is refused before a job is admitted: the
        /// bounded candidate exchange cannot carry it.
        #[serde(default)]
        prompt_logprobs: Option<u32>,
    },
    /// Rank 0 -> every worker: the sampled token for the next lockstep decode.
    Token {
        /// The token rank 0 sampled; workers must not sample their own.
        token_id: u32,
    },
    /// Rank 0 -> every worker, before each device scheduler tick: the requests that join and
    /// the sequences that end there (a stop string matched, or the client left).
    DeviceTick {
        /// The new requests, in arrival order.
        admit: Vec<Tp4DeviceAdmit>,
        /// The sequence ids that end before the tick runs.
        cancel: Vec<u64>,
        /// The tick holds a long prefill back for a decode step (rank 0's clock).
        #[serde(default)]
        hold_prefill: bool,
    },
    /// Rank 0 -> the vision tower's rank only, before the DeviceTick that admits the request: the
    /// canvas of image `image` of sequence `seq`, as `bytes` raw RGB bytes that follow the frame on
    /// the wire (a canvas is megabytes; JSON would triple it).
    DevicePixels {
        /// The sequence the image belongs to.
        seq: u64,
        /// The image's index in the admission's `images`.
        image: u32,
        /// The canvas bytes after the frame.
        bytes: u64,
    },
    /// The request ended normally; every worker leaves the request loop.
    End,
    /// The request or the group failed; every worker leaves the process.
    Error {
        /// The same code the HTTP error event uses, e.g. engine_gone.
        code: String,
        /// An operator-facing message.
        message: String,
    },
}

#[cfg(feature = "tp4-serve")]
impl Tp4ControlFrame {
    /// A short name for an unexpected-frame message.
    fn kind(&self) -> &'static str {
        match self {
            Self::Hello { .. } => "Hello",
            Self::Ready => "Ready",
            Self::Request { .. } => "Request",
            Self::Token { .. } => "Token",
            Self::DeviceTick { .. } => "DeviceTick",
            Self::DevicePixels { .. } => "DevicePixels",
            Self::End => "End",
            Self::Error { .. } => "Error",
        }
    }
}

/// One request on the device path, with its sampler, as a device tick admits it.
#[cfg(feature = "tp4-serve")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tp4DeviceAdmit {
    /// The sequence id rank 0 gave the request: later ticks cancel it by this id.
    pub seq: u64,
    /// The tokens rank 0 rendered (the chat template is already applied).
    pub prompt_ids: Vec<u32>,
    /// The request's clamped generation budget.
    pub max_tokens: u32,
    /// The temperature's f32 bits (0 is greedy); bits keep the frame `Eq`.
    pub temperature_bits: u32,
    /// The nucleus threshold's f32 bits.
    pub top_p_bits: u32,
    /// The candidate count (0 keeps every token).
    pub top_k: u32,
    /// The sampler seed every rank draws its noise from.
    pub seed: u64,
    /// The ids that end the request once emitted.
    pub stop_ids: Vec<u32>,
    /// The prompt's images in prompt order; the canvases go to the tower rank in
    /// [`Tp4ControlFrame::DevicePixels`] frames.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<Tp4DeviceImage>,
}

/// One image span of an admission: where it sits in the prompt and the canvas it is made from.
#[cfg(feature = "tp4-serve")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tp4DeviceImage {
    /// The span's first position.
    pub start: u64,
    /// Token rows.
    pub n_llm_h: u32,
    /// Token columns.
    pub n_llm_w: u32,
    /// Canvas height in pixels.
    pub best_height: u32,
    /// Canvas width in pixels.
    pub best_width: u32,
    /// The canvas's content key, hex.
    pub key: String,
}

/// A typed control-channel refusal or transport failure.
#[cfg(feature = "tp4-serve")]
#[derive(Debug)]
pub enum Tp4ControlError {
    /// A socket read/write failed.
    Io(std::io::Error),
    /// A frame was not valid JSON.
    Json(serde_json::Error),
    /// Rank 0 waited for a rank that never connected.
    MissingRank {
        /// The world size the group expected.
        world: usize,
        /// How many ranks did connect.
        joined: usize,
        /// The deadline rank 0 waited.
        waited: Duration,
    },
    /// A control socket timed out while a specific stage was expected.
    Timeout {
        /// The protocol stage that timed out.
        stage: &'static str,
    },
    /// A worker could not reach rank 0's control hub inside the startup budget.
    ConnectTimeout {
        /// Rank 0's control address the worker kept trying.
        address: SocketAddr,
        /// How long the worker kept retrying before giving up.
        waited: Duration,
    },
    /// A frame was received before the group was Ready.
    NotReady {
        /// What the receiver was waiting for.
        expected: &'static str,
    },
    /// A frame arrived that the protocol did not allow at that point.
    UnexpectedFrame {
        /// What the receiver expected.
        expected: &'static str,
        /// What it got.
        got: Tp4ControlFrame,
    },
    /// A Hello named a rank outside the group or a rank that already joined.
    BadRank {
        /// The offending rank.
        rank: u32,
        /// The world size.
        world: usize,
    },
    /// A frame declared a payload larger than the protocol cap.
    FrameTooLarge {
        /// The declared payload length.
        bytes: usize,
    },
}

#[cfg(feature = "tp4-serve")]
impl std::fmt::Display for Tp4ControlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "control I/O: {error}"),
            Self::Json(error) => write!(f, "control frame is not valid JSON: {error}"),
            Self::MissingRank {
                world,
                joined,
                waited,
            } => write!(
                f,
                "TP=4 rank 0 waited {waited:?} for {world} ranks and only {joined} joined; the missing rank is a startup refusal, not a single-rank fallback"
            ),
            Self::ConnectTimeout { address, waited } => write!(
                f,
                "control connect to {address} was still refused after {waited:?}; rank 0's hub was not listening within the TP=4 startup budget"
            ),
            Self::Timeout { stage } => write!(f, "control timeout while waiting for {stage}"),
            Self::NotReady { expected } => {
                write!(f, "control frame {expected} arrived before Ready")
            }
            Self::UnexpectedFrame { expected, got } => {
                write!(f, "control expected {expected}, got {}", got.kind())
            }
            Self::BadRank { rank, world } => {
                write!(f, "control Hello rank {rank} is outside a world of {world}")
            }
            Self::FrameTooLarge { bytes } => {
                write!(f, "control frame declares {bytes} bytes, above the cap")
            }
        }
    }
}

#[cfg(feature = "tp4-serve")]
impl std::error::Error for Tp4ControlError {}

#[cfg(feature = "tp4-serve")]
impl From<std::io::Error> for Tp4ControlError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[cfg(feature = "tp4-serve")]
impl From<serde_json::Error> for Tp4ControlError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

/// One length-prefixed JSON control frame, as it goes on the wire.
#[cfg(feature = "tp4-serve")]
fn control_frame_bytes(frame: &Tp4ControlFrame) -> Result<Vec<u8>, Tp4ControlError> {
    let payload = serde_json::to_vec(frame)?;
    let length = u32::try_from(payload.len()).map_err(|_| Tp4ControlError::FrameTooLarge {
        bytes: payload.len(),
    })?;
    let mut bytes = Vec::with_capacity(4 + payload.len());
    bytes.extend_from_slice(&length.to_be_bytes());
    bytes.extend_from_slice(&payload);
    Ok(bytes)
}

/// Writes one length-prefixed JSON control frame.
#[cfg(feature = "tp4-serve")]
fn write_control_frame(
    stream: &mut TcpStream,
    frame: &Tp4ControlFrame,
) -> Result<(), Tp4ControlError> {
    // One write per frame: with TCP_NODELAY a frame is one segment, not a header the peer
    // waits on while the payload sits behind Nagle.
    stream.write_all(&control_frame_bytes(frame)?)?;
    stream.flush()?;
    Ok(())
}

/// A [`Tp4ControlFrame::DevicePixels`] frame and its canvas bytes, as they go on the wire.
#[cfg(feature = "tp4-serve")]
fn control_pixels_bytes(seq: u64, image: u32, pixels: &[u8]) -> Result<Vec<u8>, Tp4ControlError> {
    if pixels.len() > TP4_CONTROL_MAX_PIXELS {
        return Err(Tp4ControlError::FrameTooLarge {
            bytes: pixels.len(),
        });
    }
    let frame = Tp4ControlFrame::DevicePixels {
        seq,
        image,
        bytes: pixels.len() as u64,
    };
    let mut bytes = control_frame_bytes(&frame)?;
    bytes.extend_from_slice(pixels);
    Ok(bytes)
}

/// Reads the `bytes` canvas bytes that follow a [`Tp4ControlFrame::DevicePixels`] frame.
#[cfg(feature = "tp4-serve")]
fn read_control_pixels(stream: &mut TcpStream, bytes: u64) -> Result<Vec<u8>, Tp4ControlError> {
    let length = usize::try_from(bytes).unwrap_or(usize::MAX);
    if length > TP4_CONTROL_MAX_PIXELS {
        return Err(Tp4ControlError::FrameTooLarge { bytes: length });
    }
    let mut pixels = vec![0u8; length];
    stream.read_exact(&mut pixels)?;
    Ok(pixels)
}

/// Reads one length-prefixed JSON control frame.
#[cfg(feature = "tp4-serve")]
fn read_control_frame(stream: &mut TcpStream) -> Result<Tp4ControlFrame, Tp4ControlError> {
    let mut header = [0u8; 4];
    stream.read_exact(&mut header)?;
    let length = usize::try_from(u32::from_be_bytes(header)).unwrap_or(usize::MAX);
    if length == 0 || length > TP4_CONTROL_MAX_FRAME {
        return Err(Tp4ControlError::FrameTooLarge { bytes: length });
    }
    let mut payload = vec![0u8; length];
    stream.read_exact(&mut payload)?;
    Ok(serde_json::from_slice(&payload)?)
}

/// Rank 0's end of the control channel: one stream per worker, readiness, and broadcast.
#[cfg(feature = "tp4-serve")]
#[derive(Debug)]
struct Tp4ControlHub {
    /// Index by worker rank; rank 0 is always None (it is the hub itself).
    streams: Vec<Option<TcpStream>>,
    /// The number of ranks in the group.
    world: usize,
    /// Whether every worker has declared Hello and Ready was sent.
    ready: bool,
}

#[cfg(feature = "tp4-serve")]
impl Tp4ControlHub {
    /// Binds the control listener and waits for every worker's Hello.
    ///
    /// The acceptance loop uses a non-blocking listener plus a deadline so a missing rank is a
    /// typed error with the count that joined, not an indefinite accept. Ready is sent only after
    /// every rank has connected: that is what makes a Request-before-Ready a protocol violation
    /// rather than a race.
    fn start(
        address: SocketAddr,
        world: usize,
        timeout: Duration,
    ) -> Result<Self, Tp4ControlError> {
        if world == 0 {
            return Err(Tp4ControlError::BadRank { rank: 0, world });
        }
        let listener = TcpListener::bind(address)?;
        listener.set_nonblocking(true)?;
        let deadline = Instant::now() + timeout;
        let mut streams: Vec<Option<TcpStream>> = (0..world).map(|_| None).collect();
        let mut joined = 0usize;
        while joined + 1 < world {
            if Instant::now() >= deadline {
                return Err(Tp4ControlError::MissingRank {
                    world,
                    joined,
                    waited: timeout,
                });
            }
            match listener.accept() {
                Ok((mut stream, _peer)) => {
                    stream.set_nonblocking(false)?;
                    stream.set_nodelay(true)?;
                    stream.set_read_timeout(Some(TP4_CONTROL_READ_TIMEOUT))?;
                    stream.set_write_timeout(Some(TP4_CONTROL_READ_TIMEOUT))?;
                    let frame = read_control_frame(&mut stream)?;
                    let Tp4ControlFrame::Hello { rank } = frame else {
                        return Err(Tp4ControlError::UnexpectedFrame {
                            expected: "Hello",
                            got: frame,
                        });
                    };
                    let rank = usize::try_from(rank).unwrap_or(usize::MAX);
                    if rank == 0 || rank >= world || streams[rank].is_some() {
                        return Err(Tp4ControlError::BadRank {
                            rank: u32::try_from(rank).unwrap_or(u32::MAX),
                            world,
                        });
                    }
                    streams[rank] = Some(stream);
                    joined += 1;
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => return Err(Tp4ControlError::Io(error)),
            }
        }
        let mut hub = Self {
            streams,
            world,
            ready: false,
        };
        for stream in hub.streams.iter_mut().flatten() {
            write_control_frame(stream, &Tp4ControlFrame::Ready)?;
        }
        hub.ready = true;
        Ok(hub)
    }

    /// Broadcasts one frame to every worker.
    fn broadcast(&mut self, frame: &Tp4ControlFrame) -> Result<(), Tp4ControlError> {
        debug_assert_eq!(self.streams.len(), self.world);
        if !self.ready {
            return Err(Tp4ControlError::NotReady {
                expected: "broadcast",
            });
        }
        for stream in self.streams.iter_mut().flatten() {
            write_control_frame(stream, frame)?;
        }
        Ok(())
    }
}

/// One worker's end of the control channel.
#[cfg(feature = "tp4-serve")]
struct Tp4ControlWorker {
    stream: TcpStream,
    /// This worker's rank.
    rank: u32,
    /// The world size, for validation and messages.
    world: usize,
    /// Whether rank 0's Ready has been received.
    ready: bool,
}

#[cfg(feature = "tp4-serve")]
impl Tp4ControlWorker {
    /// Connects to rank 0 and sends Hello; Ready is awaited separately by handshake.
    ///
    /// Connection-refused is retried with a short fixed pause for the whole startup budget, because
    /// rank 0 binds its control hub only after its own checkpoint load: a worker whose load finished
    /// first arrives before the listener exists and must wait for it rather than fail. Any other
    /// connect error, and the budget running out, still ends the start with a typed error; running
    /// out names the peer address and the budget in `Tp4ControlError::ConnectTimeout`.
    fn connect(
        address: SocketAddr,
        rank: u32,
        world: usize,
        timeout: Duration,
    ) -> Result<Self, Tp4ControlError> {
        if rank == 0 || usize::try_from(rank).unwrap_or(usize::MAX) >= world {
            return Err(Tp4ControlError::BadRank { rank, world });
        }
        let deadline = Instant::now() + timeout;
        let stream = loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(Tp4ControlError::ConnectTimeout {
                    address,
                    waited: timeout,
                });
            }
            match TcpStream::connect_timeout(&address, remaining) {
                Ok(stream) => break stream,
                Err(error) if error.kind() == std::io::ErrorKind::ConnectionRefused => {
                    std::thread::sleep(TP4_CONTROL_CONNECT_RETRY.min(remaining));
                }
                Err(error) => {
                    if Instant::now() >= deadline {
                        return Err(Tp4ControlError::ConnectTimeout {
                            address,
                            waited: timeout,
                        });
                    }
                    return Err(Tp4ControlError::Io(error));
                }
            }
        };
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(TP4_CONTROL_READ_TIMEOUT))?;
        stream.set_write_timeout(Some(TP4_CONTROL_READ_TIMEOUT))?;
        Ok(Self {
            stream,
            rank,
            world,
            ready: false,
        })
    }

    /// Sends Hello and waits for rank 0's Ready.
    fn handshake(&mut self) -> Result<(), Tp4ControlError> {
        debug_assert!(usize::try_from(self.rank).unwrap_or(usize::MAX) < self.world);
        write_control_frame(
            &mut self.stream,
            &Tp4ControlFrame::Hello { rank: self.rank },
        )?;
        let frame = read_control_frame(&mut self.stream)?;
        match frame {
            Tp4ControlFrame::Ready => {
                self.ready = true;
                Ok(())
            }
            other => Err(Tp4ControlError::UnexpectedFrame {
                expected: "Ready",
                got: other,
            }),
        }
    }

    /// Reads the next request-scoped frame. A frame before Ready is a typed refusal.
    fn next_frame(&mut self) -> Result<Tp4ControlFrame, Tp4ControlError> {
        if !self.ready {
            return Err(Tp4ControlError::NotReady {
                expected: "next frame",
            });
        }
        read_control_frame(&mut self.stream)
    }

    /// Reads the canvas bytes after a [`Tp4ControlFrame::DevicePixels`] frame.
    fn next_pixels(&mut self, bytes: u64) -> Result<Vec<u8>, Tp4ControlError> {
        read_control_pixels(&mut self.stream, bytes)
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
    /// The nucleus threshold the request asked for (1.0 when it named none).
    top_p: f32,
    /// The candidate count the request asked for (0 when it named none: disabled).
    top_k: i32,
    /// The repetition penalty the request asked for (1.0 when it named none: disabled).
    repetition_penalty: f32,
    /// The presence penalty the request asked for (0.0 when it named none: disabled).
    presence_penalty: f32,
    /// The frequency penalty the request asked for (0.0 when it named none: disabled).
    frequency_penalty: f32,
    /// Whether special tokens are hidden from the emitted text (vLLM's OpenAI default true).
    skip_special_tokens: bool,
    /// The sampler seed, defaulting to zero when the request named none.
    seed: u64,
    /// The logprobs the request asked for; the default asks for none and costs nothing.
    logprobs: LogprobsRequest,
    /// The stop strings the request asked for, in request order. Empty means no stop string.
    stop: Vec<String>,
    /// When the driver accepted the request into the engine queue.
    ///
    /// Captured immediately before the channel send, so the TTFT histogram built from it includes
    /// the queue wait a client actually sees. The existing per-request timing block keeps its own
    /// definition (from the start of run_request), so no wire field changes meaning.
    enqueued_at: Instant,
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
    /// The real MTP-draft acceptance counters for this request.
    dspark: DSparkStepStats,
}

/// Counts the real DSpark acceptance rule over the MTP drafts this driver produced.
///
/// The draft tokens are the engine's `StepOutput.draft`: the anchor (already emitted) followed by
/// the MTP stages' sampled candidates. For a token index `t`, the block anchored at `t` predicts
/// `t + 1`, `t + 2`, ...; the acceptance prefix is the longest run of candidates that equals the
/// target's argmax at that position. The first mismatch is the target's bonus token, and the next
/// block starts there. A block whose candidates all match ends at the token after the last one,
/// which is the bonus token too.
///
/// This is the whole five-position rule from the investigation's section 3.1. It deliberately does
/// not consult the scheduler's confidence cap: that cap is `sparkflash-scheduler::dspark` policy for
/// choosing how many rows to forward, not part of the target's accept rule. The serving path has the
/// target's real continuation, so it compares every MTP candidate and reports
/// `draft_tokens_confidence_gated` as zero (nothing was masked).
///
/// The engine still decodes one token per step, so this is an accounting overlay on the target's
/// real continuation, not a second forward pass. It is the same comparison the speculative loop
/// would make: every acceptance counted here is a position where the MTP draft matched the target's
/// own argmax.
#[derive(Debug, Default, Clone, PartialEq)]
struct SpeculativeAccounting {
    target_verify_steps: usize,
    draft_tokens_proposed: usize,
    draft_tokens_accepted: usize,
    /// Every draft row offered to the prefix walk, whether or not the target row was finite. This
    /// is the evidence denominator: it must not shrink when a non-finite row refuses a block.
    rows_seen: usize,
    /// Set when any target row was not finite. Kept separately from the row list because a
    /// non-finite anchor row has no draft row to name.
    nonfinite_seen: bool,
    /// 0-based draft-row ids whose target row was not finite. The ids are in the same draft-row
    /// space as `rows_seen`/`draft_tokens_accepted` (anchor excluded); an anchor-only non-finite row
    /// is represented by `accept_status=nonfinite` with an empty list.
    nonfinite_rows: Vec<usize>,
    active: Option<ActiveDraft>,
}

/// One draft block being replayed against the token stream.
#[derive(Debug, Clone, PartialEq)]
struct ActiveDraft {
    /// Every MTP draft position in the block, in order.
    candidates: Vec<u32>,
    /// How many of those candidates have matched so far.
    matched: usize,
}

impl SpeculativeAccounting {
    /// Adds one target argmax position and the MTP block anchored at that position.
    ///
    /// `candidates` is `StepOutput.draft[1..]`; it is empty when the step had no draft (the prefill
    /// that starts a sequence, for example). Every candidate is checked: the confidence cap is not
    /// an acceptance input on this path, so `draft_tokens_confidence_gated` stays zero.
    fn observe_token(&mut self, target: u32, candidates: &[u32]) {
        let mut start_block_here = self.active.is_none();
        if let Some(active) = self.active.as_mut() {
            if active.candidates.is_empty() {
                // The block had no draft rows; the current token is the bonus token and the block
                // for it can start now.
                self.draft_tokens_accepted += active.matched;
                self.active = None;
                start_block_here = true;
            } else if active.candidates[active.matched] == target {
                active.matched += 1;
                if active.matched == active.candidates.len() {
                    // The last candidate was accepted. The current position *is* that candidate;
                    // the next block starts at the following (bonus) token.
                    self.draft_tokens_accepted += active.matched;
                    self.active = None;
                    start_block_here = false;
                }
            } else {
                // The current position is the bonus token: the first draft that did not match.
                // The next block is anchored here.
                self.draft_tokens_accepted += active.matched;
                self.active = None;
                start_block_here = true;
            }
        }
        if start_block_here && !candidates.is_empty() {
            self.target_verify_steps += 1;
            self.draft_tokens_proposed += candidates.len();
            self.rows_seen += candidates.len();
            self.active = Some(ActiveDraft {
                candidates: candidates.to_vec(),
                matched: 0,
            });
        }
    }

    /// Refuses a position whose target logits are not finite.
    ///
    /// A NaN or infinity cannot equal a draft token and cannot anchor a target step. The finite
    /// prefix before this position still stands, but this row and every later position in its block
    /// are not accepted, no block is started from the poisoned position, and the index is recorded
    /// for the request's acceptance evidence.
    fn observe_nonfinite(&mut self, candidates: &[u32]) {
        self.nonfinite_seen = true;
        // The row that failed is the active block's next draft row. If no block is active the
        // poisoned row is the anchor itself, which has no draft-row id; the status still flags it.
        if let Some(active) = self.active.as_ref() {
            self.nonfinite_rows.push(active.matched);
        }
        // The rows offered at this position are part of the denominator before the non-finite check.
        self.rows_seen += candidates.len();
        // Commit the finite prefix that had already matched; the poisoned row itself is never
        // counted as accepted. The block anchored here is not proposed: a non-finite target row
        // cannot verify it.
        if let Some(active) = self.active.take() {
            self.draft_tokens_accepted += active.matched;
        }
    }

    /// The one evidence line the served request emits after the guard has run.
    ///
    /// The canonical public convention is drafts-only: `accept_rows_seen` counts the MTP draft rows
    /// offered (the already-emitted anchor is excluded), `accept_committed_tokens` counts the draft
    /// rows in the finite prefix, and `accept_nonfinite_rows` carries 0-based draft-row ids in that
    /// same space. An anchor-only non-finite row has no draft-row id, so it is `nonfinite` with an
    /// empty list. The device acceptance path keeps the anchor as row 0 internally; its evidence
    /// boundary converts once (subtract the anchor from `rows_seen`/`committed_tokens` and rebase
    /// the non-finite row ids). The `accept_rows_convention=drafts_only` marker makes that
    /// conversion explicit: a row without it, or with a different value, must not be compared to
    /// this one. The values are derived in this method only, so they cannot disagree.
    fn evidence_line(&self) -> String {
        let status = if self.nonfinite_seen {
            "nonfinite"
        } else {
            "accepted"
        };
        format!(
            "accept_status={status} accept_nonfinite_rows={:?} accept_rows_seen={} accept_committed_tokens={} accept_rows_convention=drafts_only",
            self.nonfinite_rows, self.rows_seen, self.draft_tokens_accepted
        )
    }

    /// Commits a block still open when generation stops.
    fn finish(&mut self) {
        if let Some(active) = self.active.take() {
            self.draft_tokens_accepted += active.matched;
        }
    }

    /// The telemetry counters, with the rates derived by the API type.
    fn stats(&self) -> DSparkStepStats {
        DSparkStepStats::from_counts(
            self.target_verify_steps,
            self.draft_tokens_proposed,
            0,
            self.draft_tokens_accepted,
        )
    }
}

/// The target row's argmax, or `None` when any logit is NaN or infinite.
///
/// A non-finite row cannot accept a draft and cannot anchor a target step. Returning `None` here
/// keeps every acceptance comparison away from NaN semantics (where `NaN != NaN` and a malformed
/// row could otherwise look like a mismatch or, worse, a match).
fn finite_target_argmax(logits: &[f32]) -> Option<u32> {
    if !logits.iter().all(|value| value.is_finite()) {
        return None;
    }
    argmax(logits).map(|index| index as u32)
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

/// The engine thread: build once, then run jobs until the queue closes.
///
/// The default loop runs one job at a time. A recipe with a batch cap runs the S4 host scheduler
/// instead: FIFO admission up to that cap, round-robin steps, same replies.
///
/// The engine and the tokenizer are built here rather than on the caller's thread because the
/// engine must never leave the thread that built it, and the handshake tells the caller whether
/// that worked before anything is bound to a socket.
fn engine_thread(
    recipe: &Recipe,
    element_limit: usize,
    mut jobs: tokio::sync::mpsc::Receiver<Job>,
    ready: std::sync::mpsc::Sender<Result<(), String>>,
    metrics: Arc<Metrics>,
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
    if let Some(max_batch) = recipe.batch {
        // The opt-in S4 host scheduler: same jobs channel, same replies, admission and round-robin
        // interleaving instead of one request at a time.
        run_continuous_engine_loop(&mut engine, &tokenizer, &mut jobs, max_batch, &metrics);
        return;
    }
    while let Some(job) = jobs.blocking_recv() {
        // The request leaves the waiting gauge and enters the running one before its first engine
        // call; the terminal branch below returns it to zero whichever way the request ends.
        metrics.request_admitted();
        // The sampler is per request and `with_sampling` consumes the engine, so the value inside
        // the box is replaced around the same plan and the same weight source - no weight is
        // re-read and no checkpoint is re-opened by this.
        *engine = (*engine).with_sampling(Sampling {
            temperature: job.temperature,
            seed: job.seed,
        });
        // The OpenAI sampling parameters are honoured on the reference path; the TP=4 host path
        // keeps its temperature/seed-only sampler and documents that as a gap.
        engine.set_sampler_options(SamplerOptions {
            top_p: job.top_p,
            top_k: job.top_k,
            repetition_penalty: job.repetition_penalty,
            presence_penalty: job.presence_penalty,
            frequency_penalty: job.frequency_penalty,
            prompt_len: job.prompt_tokens.len(),
        });
        // A panic ends this one request rather than the process - in a build that unwinds. The
        // workspace release profile's `panic = "abort"` makes this unreachable there, and Cargo does
        // not let a package override `panic`, so a release binary aborts by design; the module docs
        // say so, and the panic test below is compiled only where unwinding is on.
        let replies = job.replies.clone();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            run_request(&engine, &tokenizer, job, &metrics)
        }));
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
        metrics.request_finished();
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

// ---------------------------------------------------------------------------------------
// Host continuous batching
// ---------------------------------------------------------------------------------------
//
// The default reference loop runs one Job at a time: engine_thread receives a job, runs it to
// completion, then receives the next. Everything a request needs to make progress between two
// engine calls (the SequenceState, the detokeniser, the stop matcher, DSpark accounting, timing)
// is local to run_request, so the queue wait a client sees is the whole request that is ahead of
// it.
//
// This section is the S4 scheduler behind --continuous-batching: the same Job/Reply channel
// contract, but the loop admits up to max_batch jobs FIFO, gives every admitted request one
// scheduler step per round, and interleaves their engine calls. One step is one prefill (whole
// prompt, the same prefill_chunked call the serial path makes), and after that one emitted token
// plus one decode call. The engine is single-threaded and owned by this thread, so per-request
// sampling is applied with the non-consuming set_sampling/set_sampler_options before each engine
// call rather than by swapping the engine.
//
// This is host serving behaviour on the reference CPU engine, not a device claim. The device arm's
// batched attention, paged KV and TTFT are window measurements; nothing here speaks to them. The
// per-request state is the state the serial loop already keeps, just lifted into a slot: logprobs,
// prompt logprobs, stop strings, DSpark accounting and timings are the same computations, and a
// one-request batch reaches the same token sequence as the serial path.

/// One scheduler step's outcome for one admitted request.
enum BatchAdvance {
    /// The request is still running.
    Active,
    /// The request's terminal reply has been sent.
    Done,
}

/// Where one batched request is in its life.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchPhase {
    /// The prompt has not been prefilled.
    Prefill,
    /// The prompt is prefilled; tokens are emitted and decoded one scheduler step at a time.
    Decode,
}

/// One admitted request's scheduling slot.
struct BatchedRequest {
    /// The queued request: prompt, sampler parameters, replies and stop strings.
    job: Job,
    /// The engine sequence this slot has consumed.
    state: SequenceState,
    /// When the job entered the scheduler, for the timing block.
    started: Instant,
    /// Prefill or generation.
    phase: BatchPhase,
    /// The raw logits of the position last fed, for the token about to be emitted.
    logits: Vec<f32>,
    /// The token the last step sampled, or None when the row produced none.
    next: Option<u32>,
    /// The drafts after the anchor, for the DSpark accept rule.
    drafts_for_next: Vec<u32>,
    /// Generated tokens so far, including the one that completed a stop string.
    completion_tokens: usize,
    /// Time to first token, measured at the first emission.
    ttft: Option<Duration>,
    /// Wall clock the prefill took.
    prefill_elapsed: Duration,
    /// The request's speculative-decoding accounting.
    dspark: SpeculativeAccounting,
    /// The stop-string matcher that owns emitted text.
    stop_buffer: StopBuffer,
    /// Streaming detokeniser state: the same three values tokenizers::DecodeStream keeps.
    decode_ids: Vec<u32>,
    /// Previously produced text, trimmed from the next decode.
    decode_prefix: String,
    /// The index in decode_ids the decode prefix ends at.
    decode_prefix_index: usize,
    /// The last (up to four) generated ids, for vLLM's byte-fallback logprob correction.
    logprob_context: Vec<u32>,
}

impl BatchedRequest {
    /// Lifts one queued job into a scheduling slot.
    fn new(job: Job, config: &ModelConfig) -> Self {
        let stop_buffer = StopBuffer::new(&job.stop);
        Self {
            job,
            state: SequenceState::new(config),
            started: Instant::now(),
            phase: BatchPhase::Prefill,
            logits: Vec::new(),
            next: None,
            drafts_for_next: Vec::new(),
            completion_tokens: 0,
            ttft: None,
            prefill_elapsed: Duration::ZERO,
            dspark: SpeculativeAccounting::default(),
            stop_buffer,
            decode_ids: Vec::new(),
            decode_prefix: String::new(),
            decode_prefix_index: 0,
            logprob_context: Vec::new(),
        }
    }

    /// Applies this request's sampler to the shared engine before one of its calls.
    ///
    /// The serial loop swaps the engine with with_sampling once per request. Here the engine is
    /// shared by the active requests, so the same choice is applied non-consumingly before every
    /// call: the host engine is single-threaded and this thread owns it, so nothing else runs
    /// between the call to this method and the engine call it guards.
    fn apply_sampling(&self, engine: &mut InferenceEngine) {
        engine.set_sampling(Sampling {
            temperature: self.job.temperature,
            seed: self.job.seed,
        });
        engine.set_sampler_options(SamplerOptions {
            top_p: self.job.top_p,
            top_k: self.job.top_k,
            repetition_penalty: self.job.repetition_penalty,
            presence_penalty: self.job.presence_penalty,
            frequency_penalty: self.job.frequency_penalty,
            prompt_len: self.job.prompt_tokens.len(),
        });
    }

    /// Sends one text delta to the client that owns this request.
    fn send_delta(&self, text: String) -> Result<(), String> {
        self.job
            .replies
            .blocking_send(Reply::Delta(text))
            .map_err(|_| "the client stopped reading this response".to_owned())
    }

    /// Sends the terminal reply for a request that is over.
    ///
    /// The tail-flush and finish-reason rules are the ones run_request uses: a matched stop owns
    /// the text and reports Stop, an exhausted budget reports Length, and the token that
    /// completed a stop string counts as generated.
    fn finish(&mut self, finish_reason: FinishReason) -> Result<BatchAdvance, String> {
        let tail = self.stop_buffer.finish();
        if !tail.is_empty() {
            self.send_delta(tail)?;
        }
        self.dspark.finish();
        eprintln!("sparkflash serve: {}", self.dspark.evidence_line());
        let timing = RequestTiming::measure(
            self.job.prompt_tokens.len(),
            self.completion_tokens,
            self.prefill_elapsed,
            self.ttft,
            self.started.elapsed(),
        );
        self.job
            .replies
            .blocking_send(Reply::Done(Box::new(RunSummary {
                completion_tokens: self.completion_tokens,
                finish_reason,
                timing,
                dspark: self.dspark.stats(),
            })))
            .map_err(|_| "the client stopped reading this response".to_owned())?;
        Ok(BatchAdvance::Done)
    }

    /// Runs one scheduler step for this request.
    fn advance(
        &mut self,
        engine: &mut InferenceEngine,
        tokenizer: &HfTokenizerEncoder,
        metrics: &Metrics,
    ) -> Result<BatchAdvance, String> {
        self.apply_sampling(engine);
        match self.phase {
            BatchPhase::Prefill => {
                let want_prompt =
                    self.job.logprobs.wants_prompt() && !self.job.prompt_tokens.is_empty();
                let prefill = if want_prompt {
                    let decode_ids = |ids: &[u32]| -> Result<String, String> {
                        tokenizer.tokenizer().decode(ids, false).map_err(|error| {
                            format!(
                                "the tokenizer could not decode {} token id(s): {error}",
                                ids.len()
                            )
                        })
                    };
                    let mut decoder = LogprobDecoder::new(&decode_ids);
                    let (step, payload) = prompt_logprobs_pass(
                        engine,
                        &self.job.prompt_tokens,
                        &mut self.state,
                        &self.job.logprobs,
                        &mut decoder,
                    )?;
                    self.job
                        .replies
                        .blocking_send(Reply::PromptLogprobs(Box::new(payload)))
                        .map_err(|_| "the client stopped reading this response".to_owned())?;
                    step
                } else {
                    let plan = ChunkedPrefillPlan::plan(
                        self.job.prompt_tokens.len(),
                        0,
                        &SchedulerConfig::default(),
                    )
                    .map_err(|error| {
                        format!("the chunked prefill plan could not be built: {error}")
                    })?;
                    prefill_chunked(engine, &mut self.state, &self.job.prompt_tokens, &plan)?
                };
                self.prefill_elapsed = self.started.elapsed();
                self.logits = prefill.logits;
                self.next = prefill.next_token;
                self.drafts_for_next = prefill
                    .draft
                    .get(1..)
                    .map_or_else(Vec::new, <[u32]>::to_vec);
                self.phase = BatchPhase::Decode;
                Ok(BatchAdvance::Active)
            }
            BatchPhase::Decode => {
                if self.completion_tokens >= self.job.max_tokens {
                    return self.finish(FinishReason::Length);
                }
                let Some(token) = self.next else {
                    return self.finish(FinishReason::Stop);
                };
                if let Some(target_argmax) = finite_target_argmax(&self.logits) {
                    self.dspark
                        .observe_token(target_argmax, &self.drafts_for_next);
                } else {
                    self.dspark.observe_nonfinite(&self.drafts_for_next);
                }
                if self.ttft.is_none() {
                    // Same histogram rule as the serial path: driver entry to first token, so the
                    // scheduler queue wait and the prefill are both included.
                    metrics
                        .observe_time_to_first_token(self.job.enqueued_at.elapsed().as_secs_f64());
                    self.ttft = Some(self.started.elapsed());
                }
                let piece = tokenizers::step_decode_stream(
                    tokenizer.tokenizer(),
                    vec![token],
                    self.job.skip_special_tokens,
                    &mut self.decode_ids,
                    &mut self.decode_prefix,
                    &mut self.decode_prefix_index,
                )
                .map_err(|error| format!("the tokenizer could not decode token {token}: {error}"))?
                .filter(|piece| !piece.is_empty());
                // The stop matcher owns the text: it returns only the characters safe to emit now,
                // and a match truncates them at the stop string.
                let emit = self.stop_buffer.push(piece.as_deref());
                if self.job.logprobs.wants_sample() {
                    // The logprob belongs to the token the logits just produced, so it is built
                    // before that token is fed back in. The decoder is rebuilt from the request's
                    // four-id context because the serial path holds it across the whole request;
                    // the correction only reads the last four ids, so the reconstruction is exact.
                    let row = log_softmax(&self.logits);
                    let decode_ids = |ids: &[u32]| -> Result<String, String> {
                        tokenizer.tokenizer().decode(ids, false).map_err(|error| {
                            format!(
                                "the tokenizer could not decode {} token id(s): {error}",
                                ids.len()
                            )
                        })
                    };
                    let mut decoder = LogprobDecoder::new(&decode_ids);
                    for &context in &self.logprob_context {
                        decoder.push(context);
                    }
                    let token_string = decoder.decode(token)?;
                    let entry = token_logprob(
                        &row,
                        token,
                        token_string,
                        self.job.logprobs.sample,
                        &self.job.logprobs.token_ids,
                        &mut decoder,
                    )?;
                    if !emit.is_empty() {
                        self.send_delta(emit)?;
                    }
                    self.job
                        .replies
                        .blocking_send(Reply::TokenLogprobs(Box::new(entry)))
                        .map_err(|_| "the client stopped reading this response".to_owned())?;
                    self.logprob_context.push(token);
                    if self.logprob_context.len() > 4 {
                        self.logprob_context.remove(0);
                    }
                } else if !emit.is_empty() {
                    self.send_delta(emit)?;
                }
                self.completion_tokens += 1;
                metrics.record_generation_tokens(1);
                if self.stop_buffer.matched() {
                    return self.finish(FinishReason::Stop);
                }
                if self.completion_tokens >= self.job.max_tokens {
                    // The serial loop decodes one more token after the budget is exhausted and then
                    // drops it; this path stops before that unused call, which changes no output and
                    // saves one decode per request.
                    return self.finish(FinishReason::Length);
                }
                let step = engine.decode(&mut self.state, token).map_err(|error| {
                    format!(
                        "decode failed after {} token(s): {error}",
                        self.completion_tokens
                    )
                })?;
                self.logits = step.logits;
                self.next = step.next_token;
                // draft[0] is the anchor, the token this step just sampled; only the candidates
                // after it are drafts the target has to accept.
                self.drafts_for_next = step.draft.get(1..).map_or_else(Vec::new, <[u32]>::to_vec);
                Ok(BatchAdvance::Active)
            }
        }
    }
}

/// The engine loop for --continuous-batching: admit, step round-robin, finish until the queue
/// closes and every admitted request is done.
///
/// Admission is FIFO and bounded by max_batch: the job channel already bounds how many requests a
/// client can queue, and this bounds how many of those are in flight at once. A request keeps its
/// engine sequence until it emits its terminal reply, then its slot is reused.
fn run_continuous_engine_loop(
    engine: &mut InferenceEngine,
    tokenizer: &HfTokenizerEncoder,
    jobs: &mut tokio::sync::mpsc::Receiver<Job>,
    max_batch: usize,
    metrics: &Metrics,
) {
    debug_assert!(max_batch >= 1);
    let mut pending: VecDeque<Job> = VecDeque::new();
    let mut active: Vec<BatchedRequest> = Vec::new();
    let mut closed = false;
    loop {
        if active.is_empty() && pending.is_empty() {
            match jobs.blocking_recv() {
                Some(job) => pending.push_back(job),
                None => break,
            }
        }
        if !closed {
            // Drain what is already waiting without blocking: an empty queue means the round-robin
            // step below starts now, and a new request is admitted at the top of the next round.
            while active.len() + pending.len() < max_batch {
                match jobs.try_recv() {
                    Ok(job) => pending.push_back(job),
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                        closed = true;
                        break;
                    }
                }
            }
        }
        while active.len() < max_batch {
            let Some(job) = pending.pop_front() else {
                break;
            };
            let slot = BatchedRequest::new(job, engine.config());
            // The request leaves the waiting gauge and enters the running one at admission, the
            // same instant it gains a scheduler slot.
            metrics.request_admitted();
            active.push(slot);
        }
        let mut index = 0;
        while index < active.len() {
            let replies = active[index].job.replies.clone();
            // One request's panic is that request's failure, exactly as the serial loop treats it;
            // the build only unwinds where the workspace profile allows it.
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                active[index].advance(engine, tokenizer, metrics)
            }));
            match outcome {
                Ok(Ok(BatchAdvance::Active)) => index += 1,
                Ok(Ok(BatchAdvance::Done)) => {
                    active.swap_remove(index);
                    metrics.request_finished();
                }
                Ok(Err(message)) => {
                    let _ = replies.blocking_send(Reply::Failed(message));
                    active.swap_remove(index);
                    metrics.request_finished();
                }
                Err(payload) => {
                    let _ = replies.blocking_send(Reply::Failed(format!(
                        "the reference engine panicked on this request: {}",
                        panic_message(payload.as_ref())
                    )));
                    active.swap_remove(index);
                    metrics.request_finished();
                }
            }
        }
        if closed && active.is_empty() && pending.is_empty() {
            break;
        }
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
        #[cfg(test)]
        Weights::Zero => Box::new(ZeroWeights(SyntheticWeights::from_plan(
            &plan,
            SYNTHETIC_WEIGHT_SEED,
        ))),
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
// prompt-logprobs request costs one pass over the prompt in bounded spans, not one re-forward per
// position (see [`prompt_logprobs_pass`]).

/// The log-softmax of one raw-logit row, in output-token-id order.
pub(crate) fn log_softmax(logits: &[f32]) -> Vec<f32> {
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
pub(crate) fn competition_rank(logprobs: &[f32], token_logprob: f32) -> u32 {
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
pub(crate) fn compare_logprobs(logprobs: &[f32], left: usize, right: usize) -> std::cmp::Ordering {
    logprobs[right]
        .partial_cmp(&logprobs[left])
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| left.cmp(&right))
}

/// The indices of the `k` largest logprobs, in descending order.
pub(crate) fn top_k_indices(logprobs: &[f32], k: usize) -> Vec<usize> {
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

/// The peak logits rows one prompt-logprobs span may hold: 1 << 22 f32 values (16 MiB).
///
/// This bounds activation memory, not arithmetic: the token-at-a-time and one-pass paths agree
/// bit for bit, and `crates/sparkflash-engine/tests/chunked_prefill.rs` pins that chunk
/// boundaries do not change a value. It is a span policy only, so a caller on the tiny plan sees
/// one call and a released-vocabulary caller sees bounded calls, both from the same function.
const PROMPT_LOGPROBS_ROW_BUDGET: usize = 1 << 22;

/// The contiguous spans one prompt is fed to the engine in when prompt logprobs are requested.
///
/// The scheduler's chunk policy bounds a prefill call; this splits each chunk again so a call
/// returns at most [`PROMPT_LOGPROBS_ROW_BUDGET`] logits values. The whole prompt is still
/// consumed once, in order, on one [`SequenceState`]; only the row memory per call is bounded.
fn prompt_logprobs_spans(
    prompt_len: usize,
    vocab_size: usize,
) -> Result<Vec<std::ops::Range<usize>>, String> {
    let plan = ChunkedPrefillPlan::plan(prompt_len, 0, &SchedulerConfig::default())
        .map_err(|error| format!("the chunked prefill plan could not be built: {error}"))?;
    let rows_per_span = (PROMPT_LOGPROBS_ROW_BUDGET / vocab_size.max(1)).max(1);
    let mut spans = Vec::new();
    for range in plan.ranges() {
        let mut start = range.start;
        while start < range.end {
            let end = (start + rows_per_span).min(range.end);
            spans.push(start..end);
            start = end;
        }
    }
    Ok(spans)
}

/// Scores every prompt position with one pass over the prompt, returning the step output a
/// whole-prompt prefill would have returned.
///
/// The old honest implementation (kept below for the equality test) fed the prompt one token at a
/// time and read position `i`'s distribution from the forward after `tokens[..i]`: O(n) passes
/// with a growing prefix, because the engine reported logits only for the position it just fed.
/// The engine now returns the LM-head rows of selected positions from the same forward, so this
/// function feeds the prompt in contiguous spans and asks each span for the rows that score the
/// tokens after it. Every token is consumed once and every scored distribution is computed once.
///
/// The payload shape and the decoder sequencing are unchanged: position 0 is `None` (no
/// conditioning token), each later entry scores the prompt token itself, and the alternatives are
/// decoded with the same context the token-at-a-time path had. A span only changes where a value
/// is computed, never what it is; the equality test pins that. The final step output is the chunk
/// prefill's, which the bit-for-bit chunked-prefill invariant equates to a whole-prompt prefill.
///
/// If the engine cannot serve one of the requested positions from a pass it returns an error, and
/// this function propagates it: the request fails with the typed engine error rather than being
/// answered with a different number. (The Phase-1 TP=4 path has no all-position rank contract and
/// refuses prompt logprobs outright, before this function.)
fn prompt_logprobs_pass(
    engine: &InferenceEngine,
    prompt: &[u32],
    state: &mut SequenceState,
    logprobs: &LogprobsRequest,
    decoder: &mut LogprobDecoder<'_>,
) -> Result<(StepOutput, PromptLogprobs), String> {
    let prompt_len = prompt.len();
    let mut tokens = Vec::with_capacity(prompt_len);
    let mut positions: Vec<Option<TokenLogprob>> = vec![None; prompt_len];
    // Position 0 has no conditioning token, so it reports `None` exactly as vLLM does.
    tokens.push(TokenString {
        token_id: prompt[0],
        token: decoder.decode(prompt[0])?,
    });
    decoder.push(prompt[0]);
    let spans = prompt_logprobs_spans(prompt_len, engine.config().vocab_size)?;
    let mut feed = None;
    for span in spans {
        let at_end = span.end == prompt_len;
        // Scoring token i uses the distribution after token i-1: the final position's row is the
        // generation distribution, not a prompt position, so the last span does not ask for it.
        let rows_needed = span.len() - usize::from(at_end);
        let selected: Vec<usize> = (0..rows_needed).collect();
        let produced = engine
            .prefill_with_position_logits(state, &prompt[span.clone()], &selected)
            .map_err(|error| format!("prefill failed: {error}"))?;
        for (index, row) in produced.rows.iter().enumerate() {
            let token_index = span.start + index + 1;
            let token = decoder.decode(prompt[token_index])?;
            tokens.push(TokenString {
                token_id: prompt[token_index],
                token: token.clone(),
            });
            let row = log_softmax(row);
            positions[token_index] = Some(token_logprob(
                &row,
                prompt[token_index],
                token,
                logprobs.prompt,
                &[],
                decoder,
            )?);
            decoder.push(prompt[token_index]);
        }
        feed = Some(produced.step);
    }
    Ok((
        feed.expect("a non-empty prompt has at least one span"),
        PromptLogprobs { tokens, positions },
    ))
}

/// The token-at-a-time prompt logprob reference the one-pass path is tested against.
///
/// Only the test build compiles it: it is the honest O(n) implementation this increment replaced,
/// kept as the executable contract for the equality test (same payload, same strings, same bits).
#[cfg(test)]
fn prompt_logprobs_pass_token_at_a_time(
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

/// vLLM's stop-string rule for the reference driver's text stream.
///
/// OpenAI's `stop` is a list of strings that end generation when they are *generated*: the
/// returned text does not contain them, and `finish_reason` becomes `"stop"`. A stop string can
/// span token boundaries, so vLLM holds back `max(len(stop)) - 1` characters before emitting
/// (its `SamplingParams.output_text_buffer_length`) and checks the stop strings against the held
/// text. This is that rule, byte for byte: a partial stop string is never emitted, a completed one
/// is removed from the output, and characters still held back when generation ends for another
/// reason (length/EOS) are flushed rather than dropped.
///
/// Matching is on the decoded text and therefore on whole characters, which is what vLLM matches;
/// a stop string is never split inside a UTF-8 sequence because `str::find` can only match at
/// character boundaries. An empty stop string is refused by the protocol layer before a request
/// reaches this buffer.
struct StopBuffer {
    /// The stop strings in request order. Empty means "no stop string": then every piece is
    /// emitted immediately and nothing is held back.
    sequences: Vec<String>,
    /// How many characters of the tail can start a stop string, so they cannot be emitted yet.
    holdback: usize,
    /// Decoded characters generated but not emitted yet.
    pending: String,
    /// Whether a stop string has been matched; the driver stops after this becomes true.
    matched: bool,
}

impl StopBuffer {
    /// Builds the buffer; `holdback` is one less than the longest stop string, as in vLLM.
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

    /// Adds one token's decoded text (which may be empty or absent) and returns the characters that
    /// are safe to emit now. A stop match clears the pending text and truncates the returned text
    /// at the match, which is vLLM's "the output will not contain the stop strings".
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

    /// Returns the text still held back when generation ends without a stop match. A matched stop
    /// has already cleared it, so this is empty on the stop path.
    fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    /// Whether a stop string was matched, i.e. the request must report `"stop"`.
    fn matched(&self) -> bool {
        self.matched
    }
}

/// Runs one request and sends its events, ending with exactly one terminal reply.
///
/// `Ok(())` means the terminal reply was sent. `Err(message)` means the request failed and the
/// caller still has to report it, which is what keeps the failure path in one place.
fn run_request(
    engine: &InferenceEngine,
    tokenizer: &HfTokenizerEncoder,
    job: Job,
    metrics: &Metrics,
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
    // that the next token would then contradict. The client chooses whether special tokens are
    // hidden: vLLM's OpenAI default is true, and an explicit false passes them through.
    let mut decoder = tokenizer.tokenizer().decode_stream(job.skip_special_tokens);
    // The stop-string rule is vLLM's: hold back up to `max(len(stop)) - 1` characters so a stop
    // split across token boundaries is never emitted, and truncate at a match.
    let mut stop_buffer = StopBuffer::new(&job.stop);
    let mut logits = prefill.logits;
    let mut next = prefill.next_token;
    let mut completion_tokens = 0usize;
    let mut ttft: Option<Duration> = None;
    let mut dspark = SpeculativeAccounting::default();
    // The MTP block anchored at the token the next iteration will emit. A fresh prefill drafts
    // nothing; a chunked prefill that ended past position 0 does have a real block for the first
    // generated token, so it starts from that step's draft instead of throwing it away.
    let mut drafts_for_next = prefill
        .draft
        .get(1..)
        .map_or_else(Vec::new, <[u32]>::to_vec);
    while completion_tokens < job.max_tokens {
        let Some(token) = next else { break };
        // The accept rule compares a draft with the target's argmax, not with the sampled token.
        // They are the same at temperature 0; at any temperature the rule's input is the argmax.
        // A non-finite row has no valid argmax: it closes the finite prefix and is never allowed to
        // accept a draft or anchor the next block.
        if let Some(target_argmax) = finite_target_argmax(&logits) {
            dspark.observe_token(target_argmax, &drafts_for_next);
        } else {
            dspark.observe_nonfinite(&drafts_for_next);
        }
        if ttft.is_none() {
            // The histogram measures the request as a client saw it: from the driver accepting it
            // into the engine queue to this first token, so the queue wait is included. The
            // per-request timing block that goes on the wire keeps its own, post-dequeue,
            // definition above, so this new metric changes no existing field's meaning.
            metrics.observe_time_to_first_token(job.enqueued_at.elapsed().as_secs_f64());
            ttft = Some(request_started.elapsed());
        }
        let piece = decoder
            .step(token)
            .map_err(|error| format!("the tokenizer could not decode token {token}: {error}"))?
            .filter(|piece| !piece.is_empty());
        // The stop matcher owns the text: it returns only the characters that are safe to emit
        // now, and a match truncates them at the stop string.
        let emit = stop_buffer.push(piece.as_deref());
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
            if !emit.is_empty() {
                job.replies
                    .blocking_send(Reply::Delta(emit))
                    .map_err(|_| "the client stopped reading this response".to_owned())?;
            }
            job.replies
                .blocking_send(Reply::TokenLogprobs(Box::new(entry)))
                .map_err(|_| "the client stopped reading this response".to_owned())?;
            logprob_decoder.push(token);
        } else if !emit.is_empty() {
            job.replies
                .blocking_send(Reply::Delta(emit))
                .map_err(|_| "the client stopped reading this response".to_owned())?;
        }
        completion_tokens += 1;
        metrics.record_generation_tokens(1);
        if stop_buffer.matched() {
            // The token that completed the stop string was generated (vLLM counts it), and the
            // stop string and anything after it never reach the client.
            break;
        }
        let step = engine.decode(&mut state, token).map_err(|error| {
            format!("decode failed after {completion_tokens} token(s): {error}")
        })?;
        logits = step.logits;
        next = step.next_token;
        // `draft[0]` is the anchor, which is the `next` token this step just sampled; only the
        // candidates after it are drafts the target has to accept. No token from the scheduler's
        // old philox path exists here: the MTP forward is the only producer.
        drafts_for_next = step.draft.get(1..).map_or_else(Vec::new, <[u32]>::to_vec);
    }
    // vLLM emits the whole generated text when generation ends for a reason other than a stop: the
    // characters still held back so a split stop string could be detected go out here. A matched
    // stop already cleared them, so this is empty on that path.
    let tail = stop_buffer.finish();
    if !tail.is_empty() {
        job.replies
            .blocking_send(Reply::Delta(tail))
            .map_err(|_| "the client stopped reading this response".to_owned())?;
    }
    dspark.finish();
    // The batch-19 evidence reader keys on the engine acceptance contract's names. The line is
    // emitted on both paths: `accepted` keeps the shape uniform, `nonfinite` names the poisoned
    // rows and never claims their drafts were accepted.
    eprintln!("sparkflash serve: {}", dspark.evidence_line());
    let elapsed = request_started.elapsed();
    let timing = RequestTiming::measure(
        job.prompt_tokens.len(),
        completion_tokens,
        prefill_elapsed,
        ttft,
        elapsed,
    );
    // OpenAI's finish reasons: a matched stop string (or EOS, which ends the loop with no next
    // token) is `Stop`; exhausting the token budget is `Length`. A cap of zero is `Length`
    // because the budget is what stopped it.
    let finish_reason = if stop_buffer.matched() {
        FinishReason::Stop
    } else if completion_tokens >= job.max_tokens {
        FinishReason::Length
    } else {
        FinishReason::Stop
    };
    job.replies
        .blocking_send(Reply::Done(Box::new(RunSummary {
            completion_tokens,
            finish_reason,
            timing,
            dspark: dspark.stats(),
        })))
        .map_err(|_| "the client stopped reading this response".to_owned())?;
    Ok(())
}

// ---------------------------------------------------------------------------------------
// TP=4 rank adapter, driver, and worker loop
// ---------------------------------------------------------------------------------------

/// One forward step's output, reduced to what serve needs.
///
/// The real Tp4Rank adapter maps the engine's StepOutput into this; serve reads only the logits
/// and the rank-0 sampled token. Workers ignore next_token (it is computed identically on every
/// rank, but it must not drive the token stream).
#[cfg(feature = "tp4-serve")]
#[derive(Debug, Clone, Default)]
pub struct Tp4Step {
    /// The raw logits for the position just fed.
    pub logits: Vec<f32>,
    /// The greedy next token, when the rank produced one.
    pub next_token: Option<u32>,
}

/// The request sampler a rank applies before its first step.
///
/// The reference driver consumes its engine with with_sampling once per request. A resident TP=4
/// rank holds one engine for the process lifetime, so the same choice arrives through
/// begin_request instead. Sampling changes only which token rank 0 chooses; it does not change the
/// forward arithmetic, and workers keep the greedy default because they ignore next_token.
#[cfg(feature = "tp4-serve")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tp4Sampling {
    /// The request temperature.
    pub temperature: f32,
    /// The request seed.
    pub seed: u64,
}

#[cfg(feature = "tp4-serve")]
impl Default for Tp4Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            seed: 0,
        }
    }
}

/// The evidence counters the rank's extraction exposes.
#[cfg(feature = "tp4-serve")]
#[derive(Debug, Clone, Default)]
pub struct Tp4RankMetrics {
    /// Startup load time.
    pub load_ms: u64,
    /// Resident raw-cache bytes.
    pub resident_cache_bytes: u64,
    /// Resident raw-cache tensors.
    pub resident_cache_tensors: usize,
    /// Collective calls made by this rank.
    pub reduce_calls: u64,
    /// Collective elements reduced by this rank.
    pub reduce_elements: u64,
    /// Forward steps run by this rank.
    pub steps: u64,
    /// Wall milliseconds of the most recent step.
    pub last_step_ms: u64,
}

/// The serve-side seam over tp-dev's Tp4Rank extraction.
///
/// This is deliberately a trait rather than a direct dependency: the serve branch compiles against
/// a stub until the extraction lands, and the real adapter is one impl on top of the extracted
/// Tp4Rank. The trait contains no forward arithmetic; Tp4Rank owns the lockstep loop and the
/// resident cache.
#[cfg(feature = "tp4-serve")]
pub trait Tp4RankApi: Send {
    /// Resets the per-request SequenceState and applies the request sampler.
    ///
    /// This runs exactly once per request on every rank, before that rank feeds the Request
    /// frame's prompt ids: replace the rank's SequenceState with a fresh
    /// SequenceState::new(global_config) and assert it is empty, then apply the sampler. It must
    /// not read weights or touch the collective; the engine and collective are opened once in
    /// open_tp4_rank and stay owned by Tp4Rank. A rank that holds InferenceEngine by value applies
    /// the sampler with the non-consuming InferenceEngine::set_sampling(Sampling { temperature,
    /// seed }) (task/engine-resident 74e9ff2); it must not rebuild the engine or re-read weights.
    /// A rank that cannot apply a non-greedy sampler must return an error rather than silently run
    /// greedy.
    fn begin_request(&mut self, sampling: Tp4Sampling) -> Result<(), String>;

    /// Runs the prompt through this rank's forward and returns the last position's step.
    fn prefill(&mut self, prompt: &[u32]) -> Result<Tp4Step, String>;

    /// Runs the prompt through this rank's forward and returns the last position's step together
    /// with every scored prompt position's merged logprob contract.
    ///
    /// Selection is the OpenAI-style top count: `None` scores only the owning token and its
    /// competition rank, `Some(n)` also returns the n best alternatives. A contract refusal (a
    /// certified rank entry, an unprovable f32 order, a malformed shard layout) comes back as a
    /// string beginning with the refusal prefix; every rank makes the same decision after the
    /// same collectives, so the group can keep serving.
    fn prefill_prompt_logprobs(
        &mut self,
        prompt: &[u32],
        selection: Option<usize>,
    ) -> Result<(Tp4Step, Vec<MergedPosition>), String>;

    /// Runs one sampled token through this rank's forward.
    fn decode(&mut self, token: u32) -> Result<Tp4Step, String>;

    /// The evidence counters this rank collected.
    fn metrics(&self) -> Tp4RankMetrics;
}

/// What rank 0 passes to the rank factory: the checkpoint and the group geometry, not a tokenizer.
///
/// Token ids are the contract between serve and the rank; the tokenizer stays on rank 0 and never
/// reaches a worker.
#[cfg(feature = "tp4-serve")]
#[derive(Debug, Clone)]
pub struct Tp4RankConfig {
    /// This process's rank.
    pub rank: usize,
    /// The number of ranks in the group.
    pub world_size: usize,
    /// Every rank's address, in rank order.
    pub peers: Vec<IpAddr>,
    /// Rank R binds the engine collective on base_port + R.
    pub base_port: u16,
    /// Rank 0's serve control port.
    pub control_port: u16,
    /// The released sharded checkpoint directory or index.
    pub checkpoint: PathBuf,
    /// The element cap for one tensor.
    pub element_limit: usize,
    /// The startup and control deadline.
    pub timeout: Duration,
}

#[cfg(feature = "tp4-serve")]
impl Tp4RankConfig {
    /// The control channel address.
    fn control_address(&self) -> SocketAddr {
        SocketAddr::new(self.peers[0], self.control_port)
    }
}

/// Builds the serve rank config from the parsed options.
#[cfg(feature = "tp4-serve")]
fn tp4_rank_config(
    options: &ServeOptions,
    tp4: &Tp4ServeOptions,
    timeout: Duration,
) -> Result<Tp4RankConfig, String> {
    let checkpoint = options.checkpoint.clone().ok_or_else(|| {
        "--tp4-rank requires --checkpoint DIR: the four ranks serve one released sharded checkpoint"
            .to_owned()
    })?;
    Ok(Tp4RankConfig {
        rank: tp4.rank,
        world_size: tp4.world_size,
        peers: tp4.peers.clone(),
        base_port: tp4.base_port,
        control_port: tp4.control_port,
        checkpoint,
        element_limit: options.element_limit,
        timeout,
    })
}

/// The serve adapter over tp-dev's extracted Tp4Rank.
///
/// Tp4Rank owns the resident checkpoint, the per-rank engine, the collective and the forward. This
/// struct only maps its per-rank step API onto the serve seam; it performs no arithmetic and keeps
/// no second copy of the sequence state.
#[cfg(feature = "tp4-serve")]
struct HostTp4Rank {
    rank: crate::tp4_host::Tp4Rank,
}

#[cfg(feature = "tp4-serve")]
impl Tp4RankApi for HostTp4Rank {
    fn begin_request(&mut self, sampling: Tp4Sampling) -> Result<(), String> {
        self.rank
            .begin_request(crate::tp4_host::Tp4Sampling {
                temperature: sampling.temperature,
                seed: sampling.seed,
            })
            .map_err(|error| error.to_string())
    }

    fn prefill(&mut self, prompt: &[u32]) -> Result<Tp4Step, String> {
        let output = self
            .rank
            .prefill(prompt)
            .map_err(|error| error.to_string())?;
        Ok(Tp4Step {
            logits: output.logits,
            next_token: output.token,
        })
    }

    fn prefill_prompt_logprobs(
        &mut self,
        prompt: &[u32],
        selection: Option<usize>,
    ) -> Result<(Tp4Step, Vec<MergedPosition>), String> {
        let (output, merged) = self
            .rank
            .prefill_prompt_logprobs(prompt, selection)
            .map_err(|error| error.to_string())?;
        Ok((
            Tp4Step {
                logits: output.logits,
                next_token: output.token,
            },
            merged,
        ))
    }

    fn decode(&mut self, token: u32) -> Result<Tp4Step, String> {
        let output = self.rank.decode(token).map_err(|error| error.to_string())?;
        Ok(Tp4Step {
            logits: output.logits,
            next_token: output.token,
        })
    }

    fn metrics(&self) -> Tp4RankMetrics {
        let metrics = self.rank.metrics();
        Tp4RankMetrics {
            load_ms: metrics.load_ms,
            resident_cache_bytes: metrics.resident_cache_bytes,
            resident_cache_tensors: metrics.resident_cache_tensors,
            reduce_calls: metrics.reduce_calls,
            reduce_elements: metrics.reduce_elements,
            steps: metrics.steps,
            last_step_ms: metrics.last_step_ms,
        }
    }
}

/// Opens this process's extracted Tp4Rank and wraps it in the serve seam.
///
/// The control port stays in serve: Tp4Rank opens only the engine collective on base_port + rank.
/// resident_cache is always true here: task-57's resident path is the one this endpoint measures,
/// and no serve flag may silently turn it into a re-read-per-token path.
#[cfg(feature = "tp4-serve")]
fn open_tp4_rank(config: &Tp4RankConfig) -> Result<Box<dyn Tp4RankApi>, String> {
    let rank = crate::tp4_host::Tp4Rank::open(&crate::tp4_host::Tp4RankConfig {
        rank: config.rank,
        world_size: config.world_size,
        peers: config.peers.clone(),
        base_port: config.base_port,
        checkpoint: config.checkpoint.clone(),
        element_limit: config.element_limit,
        resident_cache: true,
        timeout: config.timeout,
    })
    .map_err(|error| format!("TP=4 rank {} open failed: {error}", config.rank))?;
    Ok(Box::new(HostTp4Rank { rank }))
}

/// The TP=4 HTTP driver: the same ApiInferenceEngine/EngineEvent contract as ReferenceDriver,
/// backed by one rank-0 engine thread that owns the rank and the control hub.
#[cfg(feature = "tp4-serve")]
pub struct Tp4Driver {
    model: String,
    jobs: tokio::sync::mpsc::Sender<Job>,
    max_tokens: usize,
    vocab_size: usize,
    cluster: ClusterStatsSnapshot,
    /// The exporter the rank-0 engine thread feeds; the router serves this same registry.
    metrics: Arc<Metrics>,
}

#[cfg(feature = "tp4-serve")]
impl Tp4Driver {
    /// Starts the rank-0 engine thread and waits for the resident rank plus the control hub.
    ///
    /// The handshake is after open() and after every worker's Hello, so a missing rank or an
    /// unreadable checkpoint refuses startup before the HTTP socket exists.
    fn start(
        recipe: &Recipe,
        rank_config: Tp4RankConfig,
        max_tokens: usize,
    ) -> Result<Self, String> {
        let (jobs, queued) = tokio::sync::mpsc::channel(TP4_JOB_QUEUE);
        let (ready, reported) = std::sync::mpsc::channel();
        let owned = recipe.clone();
        // Same sharing rule as the reference driver: one exporter, fed by the thread, served by
        // the router.
        let metrics = Arc::new(Metrics::new());
        let thread_metrics = Arc::clone(&metrics);
        std::thread::Builder::new()
            .name(ENGINE_THREAD_NAME.to_owned())
            .spawn(move || tp4_engine_thread(&owned, rank_config, queued, ready, thread_metrics))
            .map_err(|error| format!("cannot start the TP=4 engine thread: {error}"))?;
        match reported.recv() {
            Ok(Ok(())) => Ok(Self {
                model: MODEL_ID.to_owned(),
                jobs,
                max_tokens,
                vocab_size: recipe.config.vocab_size,
                cluster: tp4_cluster_snapshot(),
                metrics,
            }),
            Ok(Err(message)) => Err(message),
            Err(_) => Err("the TP=4 engine thread stopped before it reported ready".to_owned()),
        }
    }

    /// A driver with a pre-filled job channel, for the one-request guard test.
    #[cfg(test)]
    fn for_test(jobs: tokio::sync::mpsc::Sender<Job>, max_tokens: usize) -> Self {
        Self {
            model: MODEL_ID.to_owned(),
            jobs,
            max_tokens,
            vocab_size: 8,
            cluster: tp4_cluster_snapshot(),
            metrics: Arc::new(Metrics::new()),
        }
    }

    /// The exporter this driver's request lifecycle feeds.
    #[must_use]
    pub fn metrics(&self) -> Arc<Metrics> {
        Arc::clone(&self.metrics)
    }
}

#[cfg(feature = "tp4-serve")]
impl ApiInferenceEngine for Tp4Driver {
    fn model_id(&self) -> &str {
        &self.model
    }

    /// The four ranks run one token sequence in lockstep and the host-side sampler carries no
    /// per-sequence RNG, so choices of one request cannot be claimed independent; the router
    /// refuses n > 1 with a typed error instead of returning repeated samples.
    fn independent_choices(&self) -> bool {
        false
    }

    fn generate(&self, request: EngineRequest) -> BoxStream<'static, EngineEvent> {
        if !request.images.is_empty() {
            let missing = MissingCapability::vision();
            return refused(&missing.to_string(), missing.code());
        }
        if request.prompt_tokens.is_empty() {
            return refused(
                "the request carries no prompt tokens, and the TP=4 forward has nothing to prefill",
                REFUSED_CODE,
            );
        }
        if let Some(selection) = request.logprobs.prompt {
            let refusal = match selection {
                LogprobSelection::All => Some(
                    "the whole-vocabulary -1 selection needs every vocabulary entry at every scored position, and the rank contract exchanges bounded candidate blocks instead"
                        .to_owned(),
                ),
                LogprobSelection::Top(count)
                    if count as usize >= sparkflash_engine::prompt_ranks::CANDIDATES_PER_RANK_MAX =>
                {
                    Some(format!(
                        "a prompt_logprobs count of {count} exceeds the {} alternatives the rank contract can carry",
                        sparkflash_engine::prompt_ranks::CANDIDATES_PER_RANK_MAX - 1
                    ))
                }
                _ => None,
            };
            if let Some(detail) = refusal {
                return refused(
                    &format!(
                        "{} {detail}",
                        crate::tp4_host::PROMPT_LOGPROBS_REFUSED_PREFIX
                    ),
                    TP4_LOGPROBS_REFUSED_CODE,
                );
            }
        }
        let prompt_tokens = request.prompt_tokens.len();
        let max_tokens = request.max_tokens.min(self.max_tokens);
        let enqueued_at = Instant::now();
        let (replies, answers) = tokio::sync::mpsc::channel(REPLY_BUFFER);
        let job = Job {
            prompt_tokens: request.prompt_tokens,
            max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            repetition_penalty: request.repetition_penalty,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            skip_special_tokens: request.skip_special_tokens,
            seed: request.seed.unwrap_or(0),
            logprobs: request.logprobs,
            stop: request.stop,
            enqueued_at,
            replies,
        };
        self.metrics.request_enqueued();
        if self.jobs.try_send(job).is_err() {
            self.metrics.request_enqueue_refused();
            return refused(
                "the TP=4 engine already has the one request it runs at a time and one queued behind it; a second concurrent request is refused because the four ranks must run the same token sequence in lockstep",
                REFUSED_CODE,
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
                    Some(Reply::PromptLogprobs(prompt)) => {
                        Some((EngineEvent::PromptLogprobs(prompt), state))
                    }
                    Some(Reply::TokenLogprobs(logprob)) => {
                        Some((EngineEvent::TokenLogprobs(logprob), state))
                    }
                    Some(Reply::Done(summary)) => {
                        state.done = true;
                        Some((
                            finished_event_with_fabric(prompt_tokens, *summary, TP4_FABRIC),
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
                                message:
                                    "the TP=4 engine thread stopped before this request finished"
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
        Capabilities::default()
    }

    fn vocab_size(&self) -> Option<usize> {
        Some(self.vocab_size)
    }
}

/// Machine-readable wall clock for the serve timeline lines.
fn epoch_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_millis())
}

/// Writes one rank's counters to stderr, where the ssh log captures them as evidence.
#[cfg(feature = "tp4-serve")]
fn log_tp4_metrics(rank: usize, phase: &str, metrics: &Tp4RankMetrics) {
    eprintln!(
        "sparkflash serve: tp4_metrics rank={rank} phase={phase} load_ms={} last_step_ms={} steps={} reduce_calls={} reduce_elements={} resident_cache_bytes={} resident_cache_tensors={}",
        metrics.load_ms,
        metrics.last_step_ms,
        metrics.steps,
        metrics.reduce_calls,
        metrics.reduce_elements,
        metrics.resident_cache_bytes,
        metrics.resident_cache_tensors
    );
}

/// Rank 0's engine thread: open the resident rank, accept the workers, then run one job at a time.
#[cfg(feature = "tp4-serve")]
fn tp4_engine_thread(
    recipe: &Recipe,
    rank_config: Tp4RankConfig,
    mut jobs: tokio::sync::mpsc::Receiver<Job>,
    ready: std::sync::mpsc::Sender<Result<(), String>>,
    metrics: Arc<Metrics>,
) {
    let built = open_tp4_rank(&rank_config)
        .map_err(|error| format!("rank {}: {error}", rank_config.rank))
        .and_then(|rank| {
            log_tp4_metrics(rank_config.rank, "open", &rank.metrics());
            let hub = Tp4ControlHub::start(
                rank_config.control_address(),
                rank_config.world_size,
                rank_config.timeout,
            )
            .map_err(|error| format!("rank 0 control channel: {error}"))?;
            let tokenizer = load_tokenizer(&recipe.tokenizer)?;
            Ok((rank, hub, tokenizer))
        });
    let (mut rank, mut hub, tokenizer) = match built {
        Ok(built) => built,
        Err(message) => {
            let _ = ready.send(Err(message));
            return;
        }
    };
    println!(
        "sparkflash serve: tp4_timeline rank=0 phase=ready epoch_ms={} load_ms={}",
        epoch_millis(),
        rank.metrics().load_ms
    );
    if ready.send(Ok(())).is_err() {
        return;
    }
    while let Some(job) = jobs.blocking_recv() {
        metrics.request_admitted();
        let replies = job.replies.clone();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            run_tp4_request(rank.as_mut(), &mut hub, &tokenizer, job, &metrics)
        }));
        let failure = match outcome {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(error),
            Err(payload) => Some(Tp4RequestError::Failed(format!(
                "the TP=4 rank panicked on this request: {}",
                panic_message(payload.as_ref())
            ))),
        };
        log_tp4_metrics(rank_config.rank, "request", &rank.metrics());
        if let Some(error) = failure {
            let (code, message) = match error {
                Tp4RequestError::Refused(message) => {
                    (TP4_LOGPROBS_REFUSED_CODE.to_owned(), message)
                }
                Tp4RequestError::Failed(message) => (ENGINE_ERROR_CODE.to_owned(), message),
            };
            let _ = hub.broadcast(&Tp4ControlFrame::Error {
                code: code.clone(),
                message: message.clone(),
            });
            let _ = replies.blocking_send(Reply::Failed(message));
        }
        metrics.request_finished();
    }
}

/// How one TP=4 request failed.
///
/// The distinction matters to the group, not just to the HTTP status: a [Refused] failure was the
/// same deterministic decision on every rank after the same collectives, so rank 0 broadcasts a
/// typed refusal and every worker resynchronises on the next Request frame. A [Failed] failure
/// ends the worker ranks, which is what a desynced or unreachable group must do.
#[cfg(feature = "tp4-serve")]
#[derive(Debug)]
enum Tp4RequestError {
    /// A prompt-logprob contract refusal: request-scoped and recoverable for the group.
    Refused(String),
    /// Anything else: the forward, the tokenizer or the control channel failed.
    Failed(String),
}

#[cfg(feature = "tp4-serve")]
impl From<String> for Tp4RequestError {
    fn from(message: String) -> Self {
        Self::Failed(message)
    }
}

/// Classifies a rank error by its typed refusal prefix.
#[cfg(feature = "tp4-serve")]
fn classify_tp4_request_error(message: String) -> Tp4RequestError {
    if message.starts_with(crate::tp4_host::PROMPT_LOGPROBS_REFUSED_PREFIX) {
        Tp4RequestError::Refused(message)
    } else {
        Tp4RequestError::Failed(message)
    }
}

/// Runs one request on rank 0 while every worker runs the same ids.
#[cfg(feature = "tp4-serve")]
fn run_tp4_request(
    rank: &mut dyn Tp4RankApi,
    hub: &mut Tp4ControlHub,
    tokenizer: &HfTokenizerEncoder,
    job: Job,
    metrics: &Metrics,
) -> Result<(), Tp4RequestError> {
    // The whole-vocabulary -1 selection needs every vocabulary entry at every scored position;
    // the rank contract exchanges bounded candidate blocks, so the request is refused before any
    // rank does work rather than answered with a prefix of what was asked for.
    let prompt_logprobs_frame = match job.logprobs.prompt {
        None => None,
        Some(LogprobSelection::Top(count)) => Some(count),
        Some(LogprobSelection::All) => {
            return Err(Tp4RequestError::Refused(format!(
                "{} the whole-vocabulary -1 selection needs every vocabulary entry at every scored position, and the rank contract exchanges bounded candidate blocks instead",
                crate::tp4_host::PROMPT_LOGPROBS_REFUSED_PREFIX
            )));
        }
    };
    rank.begin_request(Tp4Sampling {
        temperature: job.temperature,
        seed: job.seed,
    })?;
    let max_tokens = u32::try_from(job.max_tokens).unwrap_or(u32::MAX);
    hub.broadcast(&Tp4ControlFrame::Request {
        prompt_ids: job.prompt_tokens.clone(),
        max_tokens,
        prompt_logprobs: prompt_logprobs_frame,
    })
    .map_err(|error| format!("control Request: {error}"))?;
    let request_started = Instant::now();
    let prefill_steps_before = rank.metrics().steps;
    let decode_ids = |ids: &[u32]| -> Result<String, String> {
        tokenizer.tokenizer().decode(ids, false).map_err(|error| {
            format!(
                "the tokenizer could not decode {} token id(s): {error}",
                ids.len()
            )
        })
    };
    let mut logprob_decoder = LogprobDecoder::new(&decode_ids);
    // `Some(0)` is the own-token-only contract: the rank takes the same path as `None`, and the
    // payload builder reports the competition rank of each prompt token with no alternatives.
    let prompt_selection = prompt_logprobs_frame.map(|count| {
        if count == 0 {
            None
        } else {
            Some(count as usize)
        }
    });
    let (prefill, prompt_logprobs) = match prompt_selection {
        Some(selection) => {
            let (step, merged) = rank
                .prefill_prompt_logprobs(&job.prompt_tokens, selection)
                .map_err(classify_tp4_request_error)?;
            let payload = tp4_prompt_logprobs_payload(
                &job.prompt_tokens,
                &merged,
                &job.logprobs,
                &mut logprob_decoder,
            )?;
            (step, Some(payload))
        }
        None => (rank.prefill(&job.prompt_tokens)?, None),
    };
    let prefill_elapsed = request_started.elapsed();
    // One line per request so a log reader can check the TTFT identity directly: the host
    // prefill loop runs one decode-shaped forward per prompt token, so TTFT is linear in the
    // prompt length at the step cost. See docs/archive/build/tp4-host-twin.md "TTFT and readiness".
    println!(
        "sparkflash serve: tp4_prefill rank=0 epoch_ms={} prompt_tokens={} prefill_steps={} prefill_ms={}",
        epoch_millis(),
        job.prompt_tokens.len(),
        rank.metrics().steps.saturating_sub(prefill_steps_before),
        prefill_elapsed.as_millis()
    );
    // Prompt logprobs are one payload for the prompt, delivered once at the end of prefill,
    // which is where the serial host path and vLLM deliver them.
    if let Some(payload) = prompt_logprobs {
        job.replies
            .blocking_send(Reply::PromptLogprobs(Box::new(payload)))
            .map_err(|_| "the client stopped reading this response".to_owned())?;
    }
    let want_sample = job.logprobs.wants_sample();
    let mut decoder = tokenizer.tokenizer().decode_stream(job.skip_special_tokens);
    let mut logits = prefill.logits;
    let mut next = prefill.next_token;
    let mut completion_tokens = 0usize;
    let mut ttft: Option<Duration> = None;
    while completion_tokens < job.max_tokens {
        let Some(token) = next else { break };
        if ttft.is_none() {
            // Same histogram rule as the reference paths: driver entry to first token, so the
            // rank-0 job queue wait and the host prefill are both included.
            metrics.observe_time_to_first_token(job.enqueued_at.elapsed().as_secs_f64());
            ttft = Some(request_started.elapsed());
        }
        let piece = decoder
            .step(token)
            .map_err(|error| format!("the tokenizer could not decode token {token}: {error}"))?
            .filter(|piece| !piece.is_empty());
        if want_sample {
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
        metrics.record_generation_tokens(1);
        // The last generated token is never fed back: max_tokens=8 is prefill plus seven
        // decode calls. Sending it would make every rank run one extra, unnecessary step.
        if completion_tokens >= job.max_tokens {
            break;
        }
        hub.broadcast(&Tp4ControlFrame::Token { token_id: token })
            .map_err(|error| format!("control Token: {error}"))?;
        let step = rank.decode(token)?;
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
    let finish_reason = if completion_tokens >= job.max_tokens {
        FinishReason::Length
    } else {
        FinishReason::Stop
    };
    hub.broadcast(&Tp4ControlFrame::End)
        .map_err(|error| format!("control End: {error}"))?;
    job.replies
        .blocking_send(Reply::Done(Box::new(RunSummary {
            completion_tokens,
            finish_reason,
            timing,
            // The TP=4 driver has no draft source in this slice, so it reports the honest zero:
            // no MTP draft was proposed, accepted or gated on the four-rank arm.
            dspark: DSparkStepStats::default(),
        })))
        .map_err(|_| "the client stopped reading this response".to_owned())?;
    Ok(())
}

/// Builds one TP=4 prompt position's wire entry from the merged contract values.
///
/// The entry list is the serial token_logprob list for a Top selection: the owning token first
/// with its competition rank, then the alternatives with positional ranks, collapsed by token id
/// exactly as vLLM's logprob dict collapses duplicates.
#[cfg(feature = "tp4-serve")]
fn token_logprob_from_merged(
    position: &MergedPosition,
    decoder: &mut LogprobDecoder<'_>,
) -> Result<TokenLogprob, String> {
    let own_token = position.own_token_id;
    let token = decoder.decode(own_token)?;
    let mut entries = vec![sparkflash_api::LogprobAlternative {
        token_id: own_token,
        token: token.clone(),
        logprob: position.own_logprob,
        rank: Some(position.own_rank),
    }];
    for (slot, alternative) in position.alternatives.iter().enumerate() {
        entries.push(sparkflash_api::LogprobAlternative {
            token_id: alternative.token_id,
            token: decoder.decode(alternative.token_id)?,
            logprob: alternative.logprob,
            rank: Some(u32::try_from(slot + 1).unwrap_or(u32::MAX)),
        });
    }
    Ok(TokenLogprob {
        token_id: own_token,
        token,
        logprob: position.own_logprob,
        rank: Some(position.own_rank),
        top: collapse_entries(entries),
    })
}

/// The whole prompt-logprobs payload from the TP=4 rank contract's merged positions.
///
/// Position 0 is None (no conditioning token), each later position reports its own token first
/// and the alternatives after it, and the token strings are decoded with the same preceding
/// context the serial path uses, so the payload is the serial one whenever the merge could prove
/// the order.
#[cfg(feature = "tp4-serve")]
fn tp4_prompt_logprobs_payload(
    prompt: &[u32],
    merged: &[MergedPosition],
    logprobs: &LogprobsRequest,
    decoder: &mut LogprobDecoder<'_>,
) -> Result<PromptLogprobs, String> {
    let expected = prompt.len().saturating_sub(1);
    if merged.len() != expected {
        return Err(format!(
            "the TP=4 rank contract returned {} scored positions for a {} token prompt",
            merged.len(),
            prompt.len()
        ));
    }
    let requested = match logprobs.prompt {
        Some(LogprobSelection::Top(count)) => count as usize,
        _ => 0,
    };
    let mut tokens = Vec::with_capacity(prompt.len());
    let mut positions: Vec<Option<TokenLogprob>> = vec![None; prompt.len()];
    tokens.push(TokenString {
        token_id: prompt[0],
        token: decoder.decode(prompt[0])?,
    });
    decoder.push(prompt[0]);
    for (index, position) in merged.iter().enumerate() {
        let token_index = index + 1;
        let own_token = prompt[token_index];
        if position.position != index || position.own_token_id != own_token {
            return Err(format!(
                "the TP=4 rank contract named position {} token {} where the prompt has position {index} token {own_token}",
                position.position, position.own_token_id
            ));
        }
        if position.alternatives.len() != requested {
            return Err(format!(
                "the TP=4 rank contract returned {} alternatives at position {index} where the request asked for {requested}",
                position.alternatives.len()
            ));
        }
        tokens.push(TokenString {
            token_id: own_token,
            token: decoder.decode(own_token)?,
        });
        positions[token_index] = Some(token_logprob_from_merged(position, decoder)?);
        decoder.push(own_token);
    }
    Ok(PromptLogprobs { tokens, positions })
}

/// Runs one worker rank until its control channel ends.
#[cfg(feature = "tp4-serve")]
fn tp4_worker(options: &ServeOptions, tp4: &Tp4ServeOptions) -> Result<(), String> {
    let rank_config = tp4_rank_config(options, tp4, TP4_CONTROL_TIMEOUT)?;
    let mut rank = open_tp4_rank(&rank_config)?;
    log_tp4_metrics(rank_config.rank, "open", &rank.metrics());
    let mut control = Tp4ControlWorker::connect(
        rank_config.control_address(),
        u32::try_from(rank_config.rank).map_err(|_| "rank does not fit in u32".to_owned())?,
        rank_config.world_size,
        rank_config.timeout,
    )
    .map_err(|error| format!("rank {} control connect: {error}", rank_config.rank))?;
    control
        .handshake()
        .map_err(|error| format!("rank {} control handshake: {error}", rank_config.rank))?;
    loop {
        match control.next_frame() {
            Ok(Tp4ControlFrame::Request {
                prompt_ids,
                max_tokens,
                prompt_logprobs,
            }) => {
                // Workers do not sample; reset their state with the greedy default. The engine's
                // forward arithmetic is unchanged by the sampler, and their next_token is ignored.
                rank.begin_request(Tp4Sampling::default())?;
                let selection = prompt_logprobs.map(|count| {
                    if count == 0 {
                        None
                    } else {
                        Some(count as usize)
                    }
                });
                let prefill = match selection {
                    Some(selection) => rank
                        .prefill_prompt_logprobs(&prompt_ids, selection)
                        .map(|_| ()),
                    None => rank.prefill(&prompt_ids).map(|_| ()),
                };
                if let Err(error) = prefill {
                    if !error.starts_with(crate::tp4_host::PROMPT_LOGPROBS_REFUSED_PREFIX) {
                        return Err(error);
                    }
                    // The refusal is the same deterministic decision on every rank after the same
                    // collectives. Rank 0 sends it as its typed Error frame; drain that frame and
                    // resynchronise on the next Request instead of taking down the group.
                    match control.next_frame() {
                        Ok(Tp4ControlFrame::Error { code, message })
                            if code == TP4_LOGPROBS_REFUSED_CODE =>
                        {
                            println!(
                                "sparkflash serve: tp4_prompt_logprobs_refused rank={} message={message}",
                                rank_config.rank
                            );
                        }
                        Ok(other) => {
                            return Err(format!(
                                "rank {} expected the prompt-logprob refusal frame, got {}",
                                rank_config.rank,
                                other.kind()
                            ));
                        }
                        Err(error) => {
                            return Err(format!(
                                "rank {} control after a prompt-logprob refusal: {error}",
                                rank_config.rank
                            ));
                        }
                    }
                    log_tp4_metrics(rank_config.rank, "refused", &rank.metrics());
                    continue;
                }
                let mut decoded = 0usize;
                loop {
                    match control.next_frame() {
                        Ok(Tp4ControlFrame::Token { token_id }) => {
                            if decoded >= max_tokens as usize {
                                return Err(format!(
                                    "rank 0 sent more than the {max_tokens} tokens it declared"
                                ));
                            }
                            rank.decode(token_id)?;
                            decoded += 1;
                        }
                        Ok(Tp4ControlFrame::End) => break,
                        Ok(Tp4ControlFrame::Error { code, message }) => {
                            return Err(format!("rank 0 reported {code}: {message}"));
                        }
                        Ok(other) => {
                            return Err(format!(
                                "rank {} expected Token or End, got {}",
                                rank_config.rank,
                                other.kind()
                            ));
                        }
                        Err(error) => {
                            return Err(format!(
                                "rank {} control after prefill: {error}",
                                rank_config.rank
                            ));
                        }
                    }
                }
                log_tp4_metrics(rank_config.rank, "request", &rank.metrics());
            }
            Ok(Tp4ControlFrame::End) => break,
            Ok(Tp4ControlFrame::Error { code, message }) => {
                if code == TP4_LOGPROBS_REFUSED_CODE {
                    println!(
                        "sparkflash serve: tp4_prompt_logprobs_refused rank={} message={message}",
                        rank_config.rank
                    );
                    log_tp4_metrics(rank_config.rank, "refused", &rank.metrics());
                    continue;
                }
                return Err(format!("rank 0 reported {code}: {message}"));
            }
            Ok(other) => {
                return Err(format!(
                    "rank {} expected Request, End or Error, got {}",
                    rank_config.rank,
                    other.kind()
                ));
            }
            Err(error) => {
                return Err(format!("rank {} control: {error}", rank_config.rank));
            }
        }
    }
    Ok(())
}

/// The cluster telemetry a TP=4 process can state: its fabric mode, not a node list or a rail.
#[cfg(feature = "tp4-serve")]
fn tp4_cluster_snapshot() -> ClusterStatsSnapshot {
    ClusterStatsSnapshot {
        model: MODEL_ID.to_owned(),
        fabric_mode: TP4_FABRIC.to_owned(),
        nodes: Vec::new(),
        kv: empty_kv_snapshot(),
        engram: default_engram_layers().to_vec(),
        transport: RoceTransportStats {
            fabric_mode: TP4_FABRIC.to_owned(),
            ..RoceTransportStats::default()
        },
        dspark: DSparkStepStats::default(),
        timing: RequestTimingStats::default(),
        max_seq_len: max_seq_len(),
    }
}

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

/// Runs one process of the Phase-1 TP=4 group.
///
/// Rank 0 owns HTTP, the tokenizer, the chat template and streaming; every other rank runs the
/// resident forward and a control worker and never loads a tokenizer. A missing rank is a typed
/// startup refusal from Tp4Driver::start before the HTTP socket exists.
#[cfg(feature = "tp4-serve")]
fn run_tp4(options: &ServeOptions, tp4: Tp4ServeOptions) -> ExitCode {
    crate::bootlog::init();
    if let Some(device) = tp4.device.as_ref() {
        return device::run(options, &tp4, device);
    }
    if tp4.rank != 0 {
        return match tp4_worker(options, &tp4) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("sparkflash serve: rank {}: {message}", tp4.rank);
                ExitCode::from(EXIT_REFUSED)
            }
        };
    }
    let role = crate::bootlog::role_label(tp4.rank);
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
    crate::bootlog::info(
        &role,
        "serve.rs:4820",
        format!("Tokenizer: {}", tokenizer.display()),
    );
    let recipe = match options.recipe(&tokenizer) {
        Ok(recipe) => recipe,
        Err(message) => {
            eprintln!("sparkflash serve: {message}");
            return ExitCode::from(EXIT_REFUSED);
        }
    };
    let rank_config = match tp4_rank_config(options, &tp4, TP4_CONTROL_TIMEOUT) {
        Ok(config) => config,
        Err(message) => {
            eprintln!("sparkflash serve: {message}");
            return ExitCode::from(EXIT_REFUSED);
        }
    };
    crate::bootlog::info(
        &role,
        "serve.rs:4833",
        format!(
            "Tensor parallel size {} (rank {} of {}), peers {}, collective base port {}, nccl backend tcp",
            tp4.world_size,
            tp4.rank,
            tp4.world_size,
            tp4.peers
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
            tp4.base_port
        ),
    );
    crate::bootlog::info(
        &role,
        "serve.rs:4840",
        format!(
            "Prefill mode: {} (host rank loop; the device forward is a separate arm)",
            crate::tp4_host::shipped_prefill_mode().label()
        ),
    );
    crate::bootlog::info(
        &role,
        "serve.rs:4846",
        format!(
            "Detokenization: rank 0 only; workers stream token ids over control {}",
            tp4.control_port
        ),
    );
    let load_started = std::time::Instant::now();
    let driver = match Tp4Driver::start(&recipe, rank_config, options.max_tokens) {
        Ok(driver) => driver,
        Err(message) => {
            eprintln!("sparkflash serve: {message}");
            return ExitCode::from(EXIT_REFUSED);
        }
    };
    crate::bootlog::info(
        &role,
        "serve.rs:4856",
        format!(
            "Loading weights took {:.2}s",
            load_started.elapsed().as_secs_f64()
        ),
    );
    let seam = match load_tokenizer(&tokenizer) {
        Ok(loaded) => Arc::new(TokenizerSeam::new(loaded)),
        Err(message) => {
            eprintln!("sparkflash serve: {message}");
            return ExitCode::from(EXIT_REFUSED);
        }
    };
    // Only rank 0 owns HTTP, so only rank 0 can claim a device envelope.
    let device_token_source = if tp4.rank == 0 {
        match device_token_source(options, &seam) {
            Ok(source) => source,
            Err(message) => {
                eprintln!("sparkflash serve: {message}");
                return ExitCode::from(EXIT_REFUSED);
            }
        }
    } else {
        None
    };
    println!(
        "sparkflash serve: TP=4 rank 0/{} on {}; control {}; workers never detokenize",
        tp4.world_size, tp4.peers[0], tp4.control_port
    );
    // Same sharing rule as the reference path: the rank-0 engine thread feeds this exporter and
    // the router serves it.
    let metrics = driver.metrics();
    let state = AppState::new(Arc::new(driver))
        .with_metrics(metrics)
        .with_model_descriptor(Arc::new(sparkflash_model::default_model().clone()))
        .with_max_logprobs(options.max_logprobs)
        .with_tokenizer(seam)
        .with_api_key(options.api_key.clone())
        .with_cors(options.cors.clone())
        .with_device_token_source(device_token_source);
    let drain_state = state.clone();
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
    runtime.block_on(bind_and_serve(router, drain_state, options.bind))
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
    #[cfg(feature = "tp4-serve")]
    if let ServeEngine::Tp4(tp4) = serve_engine(&options) {
        return run_tp4(&options, tp4);
    }
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
    // A debug-build test seam: serve a scripted completion instead of loading a checkpoint, so a
    // wire test can drive the real binary through the real router on a host with no model. It is
    // compiled only into debug builds and keyed by SPARKFLASH_SERVE_SCRIPT (see
    // scripted_engine_spec); it is not part of the serving contract.
    #[cfg(debug_assertions)]
    if let Some(scripted) = scripted_engine_spec() {
        let scripted = match scripted {
            Ok(scripted) => scripted,
            Err(message) => {
                eprintln!("sparkflash serve: {message}");
                return ExitCode::from(EXIT_REFUSED);
            }
        };
        return run_scripted(options, tokenizer, scripted);
    }
    let recipe = match options.recipe(&tokenizer) {
        Ok(recipe) => recipe,
        Err(message) => {
            eprintln!("sparkflash serve: {message}");
            return ExitCode::from(EXIT_REFUSED);
        }
    };

    crate::bootlog::init();
    let role = "APIServer";
    crate::bootlog::info(
        role,
        "serve.rs:5018",
        format!("Tokenizer: {}", tokenizer.display()),
    );
    crate::bootlog::info(
        role,
        "serve.rs:5021",
        format!(
            "Resolved architecture: DeepseekV41ForCausalLM (engine {})",
            selection.label()
        ),
    );
    crate::bootlog::info(
        role,
        "serve.rs:5024",
        format!(
            "Using max model len {}",
            sparkflash_api::engine::max_seq_len()
        ),
    );
    crate::bootlog::info(
        role,
        "serve.rs:5028",
        format!(
            "Prefill mode: {} (host rank loop; the device forward is a separate arm)",
            crate::tp4_host::shipped_prefill_mode().label()
        ),
    );
    crate::bootlog::info(role, "serve.rs:5033", "Loading model from scratch...");
    let load_started = std::time::Instant::now();
    let driver = match ReferenceDriver::start(&recipe, options.element_limit, options.max_tokens) {
        Ok(driver) => driver,
        Err(message) => {
            eprintln!("sparkflash serve: {message}");
            return ExitCode::from(EXIT_REFUSED);
        }
    };
    // The host path audits the checkpoint's headers and opens the shard index; the weight bytes
    // themselves are read per layer on first use, so this is the open cost, not a full load.
    crate::bootlog::info(
        role,
        "serve.rs:5040",
        format!(
            "Checkpoint opened in {:.2}s; weight bytes are read lazily per layer",
            load_started.elapsed().as_secs_f64()
        ),
    );
    // The exporter the engine thread feeds; the router must serve this same registry or every
    // request-lifecycle family would report a second, empty one.
    let metrics = driver.metrics();
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
    serve_with_engine(engine, metrics, &tokenizer, &options)
}

/// Loads the tokenizer seam, builds the app state and serves the router until stopped.
///
/// The checkpoint path and the debug-build scripted path both end here, so the wire contract
/// (prompt seam, max-logprobs cap, keep-alive, router) cannot drift between them.
fn serve_with_engine(
    engine: Arc<dyn ApiInferenceEngine>,
    metrics: Arc<Metrics>,
    tokenizer: &Path,
    options: &ServeOptions,
) -> ExitCode {
    // The same tokenizer.json the engine detokenises with also drives the HTTP prompt seam:
    // /v1/completions text becomes the checkpoint's token ids and /v1/chat/completions renders
    // the released server's V4.1 template. Failing to load it refuses startup instead of serving
    // placeholder ids.
    let seam = match load_tokenizer(tokenizer) {
        Ok(loaded) => Arc::new(TokenizerSeam::new(loaded)),
        Err(message) => {
            eprintln!("sparkflash serve: {message}");
            return ExitCode::from(EXIT_REFUSED);
        }
    };
    let device_token_source = match device_token_source(options, &seam) {
        Ok(source) => source,
        Err(message) => {
            eprintln!("sparkflash serve: {message}");
            return ExitCode::from(EXIT_REFUSED);
        }
    };
    let state = AppState::new(engine)
        .with_metrics(metrics)
        .with_model_descriptor(Arc::new(sparkflash_model::default_model().clone()))
        .with_max_logprobs(options.max_logprobs)
        .with_tokenizer(seam)
        .with_api_key(options.api_key.clone())
        .with_cors(options.cors.clone())
        .with_device_token_source(device_token_source);
    // A debug-build test seam: a wire test can shorten the keep-alive interval so an idle engine
    // gap produces observable comment frames without waiting the production 15 s. Debug-only, so
    // a release binary has no such switch.
    #[cfg(debug_assertions)]
    let state = match debug_keep_alive_ms() {
        Some(milliseconds) => state.with_sse_keep_alive(Some(Duration::from_millis(milliseconds))),
        None => state,
    };
    let drain_state = state.clone();
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
    runtime.block_on(bind_and_serve(router, drain_state, options.bind))
}

/// A scripted completion for the debug-build wire seed, read from SPARKFLASH_SERVE_SCRIPT.
#[cfg(debug_assertions)]
struct ScriptedEngineSpec {
    pieces: Vec<String>,
    finish_reason: FinishReason,
    delay_ms: Option<u64>,
}

/// Reads the debug-build wire seed, when it is set.
#[cfg(debug_assertions)]
fn scripted_engine_spec() -> Option<Result<ScriptedEngineSpec, String>> {
    let path = std::env::var_os("SPARKFLASH_SERVE_SCRIPT")?;
    Some(parse_scripted_engine(Path::new(&path)))
}

/// Parses the debug-build wire seed.
///
/// Shape: {"pieces": ["first content piece", ...], "finish_reason": "stop", "delay_ms": 0}.
/// The pieces are handed to the mock engine as its content deltas, so the wire test controls
/// the exact token boundaries the streaming parser sees.
#[cfg(debug_assertions)]
fn parse_scripted_engine(path: &Path) -> Result<ScriptedEngineSpec, String> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        format!(
            "cannot read SPARKFLASH_SERVE_SCRIPT {}: {error}",
            path.display()
        )
    })?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|error| {
        format!(
            "SPARKFLASH_SERVE_SCRIPT {} is not JSON: {error}",
            path.display()
        )
    })?;
    let pieces = value
        .get("pieces")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "SPARKFLASH_SERVE_SCRIPT needs a \"pieces\" array".to_owned())?
        .iter()
        .map(|piece| {
            piece
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| "SPARKFLASH_SERVE_SCRIPT pieces must be strings".to_owned())
        })
        .collect::<Result<Vec<_>, String>>()?;
    let finish_reason = match value
        .get("finish_reason")
        .and_then(serde_json::Value::as_str)
    {
        None | Some("stop") => FinishReason::Stop,
        Some("tool_calls") => FinishReason::ToolCalls,
        Some(other) => {
            return Err(format!(
                "SPARKFLASH_SERVE_SCRIPT has an unknown finish_reason {other:?}"
            ));
        }
    };
    let delay_ms = value.get("delay_ms").and_then(serde_json::Value::as_u64);
    Ok(ScriptedEngineSpec {
        pieces,
        finish_reason,
        delay_ms,
    })
}

/// The debug-build keep-alive override, in milliseconds.
#[cfg(debug_assertions)]
fn debug_keep_alive_ms() -> Option<u64> {
    std::env::var("SPARKFLASH_SERVE_KEEP_ALIVE_MS")
        .ok()
        .and_then(|value| value.parse().ok())
}

/// Serves a scripted mock engine instead of the checkpoint (debug-build wire seed).
#[cfg(debug_assertions)]
fn run_scripted(
    options: ServeOptions,
    tokenizer: PathBuf,
    scripted: ScriptedEngineSpec,
) -> ExitCode {
    use sparkflash_api::engine::MockEngine;
    let pieces: Vec<&str> = scripted.pieces.iter().map(String::as_str).collect();
    let mut engine = MockEngine::new(MODEL_ID)
        .with_cluster_stats(ClusterStatsSnapshot::default())
        .with_capabilities(Capabilities::ALL)
        .content(&pieces)
        .with_finish_reason(scripted.finish_reason);
    if let Some(delay_ms) = scripted.delay_ms {
        engine = engine.with_delay(Duration::from_millis(delay_ms));
    }
    eprintln!(
        "sparkflash serve: debug scripted engine (no checkpoint): {} content piece(s), keep-alive {:?} ms",
        scripted.pieces.len(),
        debug_keep_alive_ms(),
    );
    serve_with_engine(
        Arc::new(engine),
        Arc::new(Metrics::new()),
        &tokenizer,
        &options,
    )
}

/// Binds the requested address and serves the router until the process is stopped.
///
/// The address printed is the one the listener actually bound. Those differ whenever the request
/// left the port to the kernel (`--bind 127.0.0.1:0`), and an operator needs the real one.
async fn bind_and_serve(router: axum::Router, state: AppState, requested: SocketAddr) -> ExitCode {
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
    println!(
        "sparkflash serve: serve_timeline phase=bind epoch_ms={} addr={bound}",
        epoch_millis()
    );
    println!("sparkflash serve: listening on http://{bound}");
    crate::bootlog::info(
        "APIServer",
        "serve.rs:5250",
        format!("Starting server on http://{bound}"),
    );
    crate::bootlog::info(
        "APIServer",
        "serve.rs:5254",
        "Application startup complete.",
    );
    // Graceful drain: the first SIGTERM/SIGINT starts the drain, /readyz turns 503 immediately so a
    // load balancer or readiness probe stops sending new work, and the listener stops accepting
    // after the configured pre-stop delay. Requests already accepted are allowed to finish until
    // the drain deadline; the process then exits 0 (or 1 if the deadline forced it).
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
    let server = axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            let _ = stop_rx.await;
        })
        .into_future();
    let mut server = Box::pin(server);
    let drain_state = state.clone();
    let drain_metrics = Arc::clone(&state.metrics);
    let watchdog = async move {
        shutdown_signal().await;
        drain_state.begin_drain();
        drain_metrics.set_serving(false);
        let timeout = drain_timeout();
        println!(
            "sparkflash serve: serve_timeline phase=drain_start epoch_ms={} timeout_ms={}",
            epoch_millis(),
            timeout.as_millis()
        );
        let delay = drain_delay();
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        println!(
            "sparkflash serve: serve_timeline phase=drain_accepting_stopped epoch_ms={}",
            epoch_millis()
        );
        // The receiver drops with this future when the server wins the race below; that is
        // harmless because the server is already over in that branch.
        let _ = stop_tx.send(());
    };
    let mut watchdog = Box::pin(watchdog);
    tokio::select! {
        result = &mut server => return serve_result(result),
        () = &mut watchdog => {}
    }
    // The drain signal is in and the listener has been asked to stop; bounds the time the
    // connections already accepted may take to finish.
    match tokio::time::timeout(drain_timeout(), &mut server).await {
        Ok(result) => {
            println!(
                "sparkflash serve: serve_timeline phase=drain_complete epoch_ms={}",
                epoch_millis()
            );
            serve_result(result)
        }
        Err(_) => {
            eprintln!(
                "sparkflash serve: serve_timeline phase=drain_timeout epoch_ms={} forced_exit=1",
                epoch_millis()
            );
            ExitCode::FAILURE
        }
    }
}

/// Maps the server future's result to the process exit code.
fn serve_result(result: Result<(), std::io::Error>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sparkflash serve: the server stopped: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Resolves on the first SIGTERM or SIGINT, the two drain signals an orchestrator sends.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut terminate = signal(SignalKind::terminate()).ok();
        let mut interrupt = signal(SignalKind::interrupt()).ok();
        tokio::select! {
            _ = async {
                match terminate.as_mut() {
                    Some(signal) => {
                        signal.recv().await;
                    }
                    None => std::future::pending().await,
                }
            } => {}
            _ = async {
                match interrupt.as_mut() {
                    Some(signal) => {
                        signal.recv().await;
                    }
                    None => std::future::pending().await,
                }
            } => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// The wall-clock budget a drain may take before the process is forced out.
///
/// \`SPARKFLASH_SERVE_DRAIN_TIMEOUT_MS\` overrides it (positive milliseconds only); the default is
/// 30 s, long enough for a typical in-flight decode to finish and short enough that an
/// orchestrator's termination grace period is not exceeded.
fn drain_timeout() -> Duration {
    drain_millis_env("SPARKFLASH_SERVE_DRAIN_TIMEOUT_MS", Duration::from_secs(30))
}

/// The grace period between the drain signal and the listener closing.
///
/// \`SPARKFLASH_SERVE_DRAIN_DELAY_MS\` overrides it; the default is zero because a probe-driven
/// deployment does not need one. A Kubernetes deployment that relies on endpoint propagation may
/// set it to a few seconds, which is the same purpose as a preStop hook.
fn drain_delay() -> Duration {
    drain_millis_env("SPARKFLASH_SERVE_DRAIN_DELAY_MS", Duration::ZERO)
}

/// Reads a positive millisecond duration from \`name\`, or returns \`default\` for unset/invalid.
fn drain_millis_env(name: &str, default: Duration) -> Duration {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|milliseconds| *milliseconds > 0)
        .map(Duration::from_millis)
        .unwrap_or(default)
}

/// A test-only weight source whose every tensor is zero.
///
/// The synthetic byte generator produces random bit patterns, and in the tiny plan those decode to
/// all-NaN logits; that is useful for the non-finite guard but useless for measuring acceptance.
/// Zero weights keep every forward value finite (all-zero logits, all-zero drafts), so a served
/// request exercises the real acceptance comparison and produces a non-zero accepted count without
/// a checkpoint. It is `#[cfg(test)]` only: nothing a shipped binary builds can construct it.
#[cfg(test)]
struct ZeroWeights(SyntheticWeights);

#[cfg(test)]
impl WeightSource for ZeroWeights {
    fn names(&self) -> Vec<String> {
        self.0.names()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.0.spec(name)
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        let spec = self.0.spec(name)?;
        Ok(vec![0; spec.storage_bytes() as usize])
    }
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

// The host harness has no production caller yet; task-22 will consume it from the
// continuous-batch tests. Keeping the items crate-internal is what the missing-docs gate
// needs; the dead-code allowance is limited to non-test builds, where nothing calls it yet.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod paged_serve {
    //! Host serving-sequence harness for paged format-2 KV.
    //!
    //! The reference is the same format-2 bytes held contiguously, not the engine's `f32` dense
    //! state: paging is storage, not arithmetic. Device attention, full `lm_head` logits and load
    //! behaviour are named window measurements, not claims made by this module.
    use std::collections::HashMap;

    use sparkflash_device::paged_kv::{AbsoluteRows, IndexerKeys, PagedKvView, WindowRows};
    use sparkflash_kernels::m4_quant::bf16_bits;
    use sparkflash_kernels::mla_ref::{
        HSI_BLOCK_TOKENS, HSI_BLOCKS, INDEX_HEAD_DIM, attention_with_sink_union,
    };
    use sparkflash_kv::config::compress_ratio;
    use sparkflash_kv::fp4::{IndexerKEntry, dequantize_indexer_k, quantize_indexer_k};
    use sparkflash_kv::layout::{
        GLOBAL_ENTRY_BYTES, INDEXER_K_DATA_OFFSET, INDEXER_K_DIM, INDEXER_K_ENTRY_BYTES,
        MAIN_KV_DATA_BYTES, MAIN_KV_DATA_OFFSET, MAIN_KV_SCALE_BYTES, MAIN_KV_SCALE_OFFSET,
        SWA_SLOT_DATA_BYTES, SWA_SLOT_SCALE_BYTES, SWA_SLOTS_PER_TAIL, SWA_TAIL_BYTES,
        entries_per_page, source_slot, swa_slot_offset,
    };
    use sparkflash_kv::paged::{
        AppendReport, FreeReport, PagedGeometry, PagedKvCache, PagingError, SeqId,
        SequenceState as KvSequenceState,
    };

    const QUERY_VALUES: usize = 512;
    const REFERENCE_SINK: f32 = 0.25;
    const REFERENCE_VOCAB: usize = 64;

    /// One position's observable output, compared bit for bit across the two paths.
    #[derive(Debug, Clone, PartialEq)]
    pub(crate) struct ServedStep {
        pub(crate) attention: Vec<f32>,
        pub(crate) logits: Vec<f32>,
        pub(crate) next_token: u32,
    }

    fn reference_head(attention: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; REFERENCE_VOCAB];
        for (token, logit) in logits.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (index, value) in attention.iter().enumerate() {
                let weight = (((index * 31 + token * 17) % 101) as f32 - 50.0) / 64.0;
                sum += *value * weight;
            }
            *logit = sum;
        }
        logits
    }

    fn next_token(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        for (index, value) in logits.iter().enumerate().skip(1) {
            if *value > logits[best] {
                best = index;
            }
        }
        best as u32
    }

    fn reference_query() -> Vec<f32> {
        (0..QUERY_VALUES)
            .map(|index| ((index as f32) * 0.001).sin() * 0.1)
            .collect()
    }

    fn synthetic_entry(layer: usize, group: usize) -> Vec<u8> {
        let mut entry = vec![0u8; GLOBAL_ENTRY_BYTES];
        let main_end = MAIN_KV_DATA_OFFSET + MAIN_KV_DATA_BYTES;
        for (offset, slot) in entry[MAIN_KV_DATA_OFFSET..main_end].iter_mut().enumerate() {
            *slot = ((group * 7 + (MAIN_KV_DATA_OFFSET + offset) * 13) % 251) as u8;
        }
        let scale_end = MAIN_KV_SCALE_OFFSET + MAIN_KV_SCALE_BYTES;
        for slot in &mut entry[MAIN_KV_SCALE_OFFSET..scale_end] {
            *slot = 0x38;
        }
        let values: [f32; INDEXER_K_DIM] =
            std::array::from_fn(|channel| (((group * 13 + channel) % 9) as f32) - 4.0);
        let quantised = quantize_indexer_k(&values).expect("synthetic indexer k");
        let mut encoded = [0u8; INDEXER_K_ENTRY_BYTES];
        quantised.to_bytes(&mut encoded).expect("encode indexer k");
        entry[INDEXER_K_DATA_OFFSET..INDEXER_K_DATA_OFFSET + INDEXER_K_ENTRY_BYTES]
            .copy_from_slice(&encoded);
        let _ = layer;
        entry
    }

    fn write_window_slot(tail: &mut [u8], layer: usize, logical: usize) {
        let slot = logical % SWA_SLOTS_PER_TAIL;
        let offset = swa_slot_offset(layer, slot);
        for (byte, slot) in tail[offset..offset + SWA_SLOT_DATA_BYTES]
            .iter_mut()
            .enumerate()
        {
            *slot = ((logical * 7 + byte) % 200 + 1) as u8;
        }
        for slot in &mut tail
            [offset + SWA_SLOT_DATA_BYTES..offset + SWA_SLOT_DATA_BYTES + SWA_SLOT_SCALE_BYTES]
        {
            *slot = 0x38;
        }
    }

    fn read_window_slot(tail: &[u8], layer: usize, logical: usize) -> (Vec<u8>, Vec<u8>) {
        let slot = logical % SWA_SLOTS_PER_TAIL;
        let offset = swa_slot_offset(layer, slot);
        (
            tail[offset..offset + SWA_SLOT_DATA_BYTES].to_vec(),
            tail[offset + SWA_SLOT_DATA_BYTES..offset + SWA_SLOT_DATA_BYTES + SWA_SLOT_SCALE_BYTES]
                .to_vec(),
        )
    }

    /// A serving state whose KV lives in the paged format-2 pool.
    #[derive(Debug)]
    pub(crate) struct PagedSequenceState {
        cache: PagedKvCache,
        pool: Vec<u8>,
        layer: usize,
        ratio: usize,
        tails: HashMap<SeqId, Vec<u8>>,
        known_tokens: HashMap<SeqId, Vec<u32>>,
    }

    impl PagedSequenceState {
        pub(crate) fn new(pages: u32, tails: u32, layer: usize) -> Result<Self, PagingError> {
            let slot = source_slot(layer).ok_or_else(|| {
                PagingError::Invariant(format!("layer {layer} owns no compressed global KV"))
            })?;
            let ratio = compress_ratio(layer);
            let geometry = PagedGeometry::new(pages, tails)?;
            let entries = pages as usize * entries_per_page(slot);
            let pool_len = entries.checked_mul(GLOBAL_ENTRY_BYTES).ok_or_else(|| {
                PagingError::Invariant("paged pool size overflows usize".to_owned())
            })?;
            Ok(Self {
                cache: PagedKvCache::new(geometry),
                pool: vec![0u8; pool_len],
                layer,
                ratio,
                tails: HashMap::new(),
                known_tokens: HashMap::new(),
            })
        }

        pub(crate) fn add_sequence(&mut self, id: SeqId) -> Result<(), PagingError> {
            self.cache.add_sequence(id)?;
            self.tails.insert(id, vec![0u8; SWA_TAIL_BYTES]);
            self.known_tokens.insert(id, Vec::new());
            self.check_invariants()
        }

        pub(crate) fn append(
            &mut self,
            id: SeqId,
            tokens: &[u32],
        ) -> Result<AppendReport, PagingError> {
            let result = self.cache.append(id, tokens);
            // A refused append may still have applied a prefix of `tokens`; rewrite whatever the
            // cache now holds so the source pool and the block table cannot disagree on error.
            self.tails
                .entry(id)
                .or_insert_with(|| vec![0u8; SWA_TAIL_BYTES]);
            self.rewrite(id)?;
            self.check_invariants()?;
            result
        }

        pub(crate) fn preempt(&mut self, id: SeqId) -> Result<FreeReport, PagingError> {
            let report = self.cache.preempt(id)?;
            self.tails.remove(&id);
            self.check_invariants()?;
            Ok(report)
        }

        pub(crate) fn recompute(&mut self, id: SeqId) -> Result<AppendReport, PagingError> {
            if self.cache.sequence(id).map(|sequence| sequence.state())
                != Some(KvSequenceState::Preempted)
            {
                return Err(PagingError::Invariant(
                    "recompute requires a preempted sequence".to_owned(),
                ));
            }
            let tokens = self
                .known_tokens
                .get(&id)
                .cloned()
                .ok_or(PagingError::UnknownSequence(id))?;
            if tokens.is_empty() {
                return Err(PagingError::Invariant(
                    "cannot recompute a request with no known tokens".to_owned(),
                ));
            }
            self.append(id, &tokens)
        }

        pub(crate) fn free(&mut self, id: SeqId) -> Result<FreeReport, PagingError> {
            let report = self.cache.free_sequence(id)?;
            self.tails.remove(&id);
            self.known_tokens.remove(&id);
            self.check_invariants()?;
            Ok(report)
        }

        pub(crate) fn check_invariants(&self) -> Result<(), PagingError> {
            self.cache.check_invariants()
        }

        pub(crate) fn len(&self, id: SeqId) -> Result<usize, PagingError> {
            self.cache
                .sequence(id)
                .map(|sequence| sequence.tokens().len())
                .ok_or(PagingError::UnknownSequence(id))
        }

        fn rewrite(&mut self, id: SeqId) -> Result<(), PagingError> {
            let tokens = self
                .cache
                .sequence(id)
                .ok_or(PagingError::UnknownSequence(id))?
                .tokens()
                .to_vec();
            self.known_tokens.insert(id, tokens.clone());
            let layer = self.layer;
            let tail = self
                .tails
                .get_mut(&id)
                .ok_or(PagingError::UnknownSequence(id))?;
            tail.fill(0);
            for logical in 0..tokens.len() {
                write_window_slot(tail, layer, logical);
            }
            for logical in 0..tokens.len() {
                self.write_source(id, logical)?;
            }
            Ok(())
        }

        fn write_source(&mut self, id: SeqId, logical: usize) -> Result<(), PagingError> {
            let entry = PagedKvView::new(&self.cache).source_entry(id, self.layer, logical)?;
            let group = logical / self.ratio;
            let bytes = synthetic_entry(self.layer, group);
            let start = entry.byte_offset;
            let end = start + GLOBAL_ENTRY_BYTES;
            let pool_len = self.pool.len();
            let destination = self.pool.get_mut(start..end).ok_or_else(|| {
                PagingError::Invariant(format!(
                    "pool is {pool_len} bytes, the mapping reaches byte {end}"
                ))
            })?;
            destination.copy_from_slice(&bytes);
            Ok(())
        }

        pub(crate) fn stage(
            &self,
            id: SeqId,
            position: usize,
        ) -> Result<(WindowRows, AbsoluteRows, IndexerKeys), PagingError> {
            let held = self.len(id)?;
            if position >= held {
                return Err(PagingError::OutOfRange {
                    sequence: id,
                    logical: position,
                    held,
                });
            }
            let tail = self
                .tails
                .get(&id)
                .ok_or(PagingError::UnknownSequence(id))?;
            let view = PagedKvView::new(&self.cache);
            let window = view.window_rows(id, self.layer, position + 1, tail)?;
            let rows = view.absolute_source_rows(id, self.layer, position, &self.pool)?;
            let indexer = view.absolute_indexer_keys(id, self.layer, position, &self.pool)?;
            Ok((window, rows, indexer))
        }

        pub(crate) fn step(&self, id: SeqId, position: usize) -> Result<ServedStep, PagingError> {
            let (window, rows, _) = self.stage(id, position)?;
            let selected: Vec<u32> = (0..=position as u32).collect();
            let attention = attention_with_sink_union(
                &reference_query(),
                &window.codes,
                &window.scales,
                window.rows,
                &rows.codes,
                &rows.scales,
                &selected,
                position as u32,
                REFERENCE_SINK,
            );
            let logits = reference_head(&attention);
            Ok(ServedStep {
                next_token: next_token(&logits),
                attention,
                logits,
            })
        }

        pub(crate) fn indexer_keys(
            &self,
            id: SeqId,
            position: usize,
        ) -> Result<IndexerKeys, PagingError> {
            self.stage(id, position).map(|(_, _, keys)| keys)
        }
    }

    /// The dense reference with the same format-2 bytes held contiguously.
    #[derive(Debug)]
    pub(crate) struct DenseSequenceState {
        layer: usize,
        ratio: usize,
        tokens: Vec<u32>,
        pool: Vec<u8>,
        tail: Vec<u8>,
    }

    impl DenseSequenceState {
        pub(crate) fn new(layer: usize, max_tokens: usize) -> Result<Self, PagingError> {
            if source_slot(layer).is_none() {
                return Err(PagingError::Invariant(format!(
                    "layer {layer} owns no compressed global KV"
                )));
            }
            let ratio = compress_ratio(layer);
            let groups = max_tokens / ratio + 2;
            Ok(Self {
                layer,
                ratio,
                tokens: Vec::new(),
                pool: vec![0u8; groups * GLOBAL_ENTRY_BYTES],
                tail: vec![0u8; SWA_TAIL_BYTES],
            })
        }

        pub(crate) fn append(&mut self, tokens: &[u32]) {
            for token in tokens {
                let logical = self.tokens.len();
                let group = logical / self.ratio;
                let bytes = synthetic_entry(self.layer, group);
                self.pool[group * GLOBAL_ENTRY_BYTES..(group + 1) * GLOBAL_ENTRY_BYTES]
                    .copy_from_slice(&bytes);
                write_window_slot(&mut self.tail, self.layer, logical);
                self.tokens.push(*token);
            }
        }

        fn entry(&self, logical: usize) -> &[u8] {
            let group = logical / self.ratio;
            &self.pool[group * GLOBAL_ENTRY_BYTES..(group + 1) * GLOBAL_ENTRY_BYTES]
        }

        pub(crate) fn stage(
            &self,
            position: usize,
        ) -> Result<(WindowRows, AbsoluteRows, IndexerKeys), PagingError> {
            if position >= self.tokens.len() {
                return Err(PagingError::OutOfRange {
                    sequence: 0,
                    logical: position,
                    held: self.tokens.len(),
                });
            }
            let attended = position + 1;
            let rows = attended.min(SWA_SLOTS_PER_TAIL);
            let first = attended - rows;
            let mut window = WindowRows {
                codes: Vec::with_capacity(rows * SWA_SLOT_DATA_BYTES),
                scales: Vec::with_capacity(rows * SWA_SLOT_SCALE_BYTES),
                rows,
            };
            for logical in first..attended {
                let (codes, scales) = read_window_slot(&self.tail, self.layer, logical);
                window.codes.extend_from_slice(&codes);
                window.scales.extend_from_slice(&scales);
            }
            let mut absolute = AbsoluteRows {
                codes: Vec::with_capacity(attended * MAIN_KV_DATA_BYTES),
                scales: Vec::with_capacity(attended * MAIN_KV_SCALE_BYTES),
                rows: attended,
            };
            for logical in 0..attended {
                let entry = self.entry(logical);
                absolute.codes.extend_from_slice(
                    &entry[MAIN_KV_DATA_OFFSET..MAIN_KV_DATA_OFFSET + MAIN_KV_DATA_BYTES],
                );
                absolute.scales.extend_from_slice(
                    &entry[MAIN_KV_SCALE_OFFSET..MAIN_KV_SCALE_OFFSET + MAIN_KV_SCALE_BYTES],
                );
            }
            let slots = HSI_BLOCKS * HSI_BLOCK_TOKENS;
            let mut indexer = IndexerKeys {
                keys: vec![0u16; slots * INDEX_HEAD_DIM],
                reachable_blocks: attended.div_ceil(HSI_BLOCK_TOKENS),
                position: position as u32,
            };
            for logical in 0..attended {
                let entry = self.entry(logical);
                let bytes =
                    &entry[INDEXER_K_DATA_OFFSET..INDEXER_K_DATA_OFFSET + INDEXER_K_ENTRY_BYTES];
                let parsed = IndexerKEntry::from_bytes(bytes).map_err(|error| {
                    PagingError::Invariant(format!("dense indexer k for token {logical}: {error}"))
                })?;
                let mut values = [0.0f32; INDEXER_K_DIM];
                dequantize_indexer_k(&parsed, &mut values).map_err(|error| {
                    PagingError::Invariant(format!("dense indexer k values: {error}"))
                })?;
                for (slot, value) in indexer.keys
                    [logical * INDEX_HEAD_DIM..(logical + 1) * INDEX_HEAD_DIM]
                    .iter_mut()
                    .zip(values.iter())
                {
                    *slot = bf16_bits(*value);
                }
            }
            Ok((window, absolute, indexer))
        }

        pub(crate) fn step(&self, position: usize) -> Result<ServedStep, PagingError> {
            let (window, rows, _) = self.stage(position)?;
            let selected: Vec<u32> = (0..=position as u32).collect();
            let attention = attention_with_sink_union(
                &reference_query(),
                &window.codes,
                &window.scales,
                window.rows,
                &rows.codes,
                &rows.scales,
                &selected,
                position as u32,
                REFERENCE_SINK,
            );
            let logits = reference_head(&attention);
            Ok(ServedStep {
                next_token: next_token(&logits),
                attention,
                logits,
            })
        }

        pub(crate) fn indexer_keys(&self, position: usize) -> Result<IndexerKeys, PagingError> {
            self.stage(position).map(|(_, _, keys)| keys)
        }
    }

    #[cfg(test)]
    fn assert_same_step(left: &ServedStep, right: &ServedStep, context: &str) {
        assert_eq!(
            left.attention
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            right
                .attention
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "attention bits differ: {context}"
        );
        assert_eq!(
            left.logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            right.logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "reference-head logits bits differ: {context}"
        );
        assert_eq!(
            left.next_token, right.next_token,
            "argmax next token differs: {context}"
        );
    }

    #[cfg(test)]
    mod tests {
        use super::{DenseSequenceState, PagedSequenceState, assert_same_step};

        const PROMPT_LEN: usize = 130;

        fn prompt() -> Vec<u32> {
            (0..PROMPT_LEN as u32).map(|index| index + 1).collect()
        }

        fn paged(layer: usize) -> PagedSequenceState {
            let mut state = PagedSequenceState::new(8, 2, layer).expect("paged geometry");
            state.add_sequence(1).expect("add sequence");
            state
        }

        fn dense(layer: usize, max_tokens: usize) -> DenseSequenceState {
            DenseSequenceState::new(layer, max_tokens).expect("dense state")
        }

        #[test]
        fn paging_is_storage_not_arithmetic_for_prefill_and_decode() {
            let mut paged = paged(20);
            paged.append(1, &prompt()).expect("prefill");
            let mut dense = dense(20, PROMPT_LEN + 4);
            dense.append(&prompt());
            for position in [0usize, 1, 17, 64, 127, 129] {
                let left = paged.step(1, position).expect("paged step");
                let right = dense.step(position).expect("dense step");
                assert_same_step(&left, &right, &format!("prefill position {position}"));
            }
            let mut next = 999u32;
            for decode in 0..3 {
                paged.append(1, &[next]).expect("paged decode");
                dense.append(&[next]);
                let position = PROMPT_LEN + decode;
                let left = paged.step(1, position).expect("paged decode step");
                let right = dense.step(position).expect("dense decode step");
                assert_same_step(&left, &right, &format!("decode step {position}"));
                next = left.next_token;
            }
        }

        #[test]
        fn ratio_two_absolute_rows_are_storage_not_arithmetic() {
            let mut paged = paged(2);
            paged.append(1, &prompt()).expect("prefill");
            let mut dense = dense(2, PROMPT_LEN + 1);
            dense.append(&prompt());
            for position in [0usize, 1, 2, 63, 64, 129] {
                let left = paged.step(1, position).expect("paged step");
                let right = dense.step(position).expect("dense step");
                assert_same_step(&left, &right, &format!("ratio-2 position {position}"));
            }
        }

        #[test]
        fn absolute_rows_and_window_are_storage_not_arithmetic() {
            let mut paged = paged(20);
            paged.append(1, &prompt()).expect("prefill");
            let mut dense = dense(20, PROMPT_LEN + 1);
            dense.append(&prompt());
            for position in [0usize, 1, 63, 64, 127, 129] {
                let (paged_window, paged_rows, _) = paged.stage(1, position).expect("paged stage");
                let (dense_window, dense_rows, _) = dense.stage(position).expect("dense stage");
                assert_eq!(paged_rows, dense_rows, "absolute rows at {position}");
                assert_eq!(
                    paged_window.rows, dense_window.rows,
                    "window rows at {position}"
                );
                assert_eq!(
                    paged_window.codes, dense_window.codes,
                    "window codes at {position}"
                );
                assert_eq!(
                    paged_window.scales, dense_window.scales,
                    "window scales at {position}"
                );
            }
        }

        #[test]
        fn prefix_sharing_changes_storage_not_output() {
            let mut paged = paged(20);
            paged.add_sequence(2).expect("second sequence");
            paged.append(1, &prompt()).expect("first prefill");
            let report = paged.append(2, &prompt()).expect("shared prefill");
            assert!(
                report.shared > 0,
                "a repeated prompt must share complete blocks"
            );
            let mut dense = dense(20, PROMPT_LEN + 1);
            dense.append(&prompt());
            let reference = dense.step(PROMPT_LEN - 1).expect("dense step");
            for id in [1u64, 2] {
                let left = paged.step(id, PROMPT_LEN - 1).expect("shared step");
                assert_same_step(&left, &reference, &format!("prefix-shared sequence {id}"));
            }
        }

        #[test]
        fn preemption_and_recompute_match_the_dense_reference() {
            use sparkflash_kv::paged::PagingError;

            let mut paged = PagedSequenceState::new(3, 2, 20).expect("paged geometry");
            paged.add_sequence(1).expect("add 1");
            paged.add_sequence(2).expect("add 2");
            let short1: Vec<u32> = (0..65u32).collect();
            let short2: Vec<u32> = (1000..1065u32).collect();
            paged.append(1, &short1).expect("seq 1");
            let before = paged.step(1, 64).expect("seq 1 step");
            let error = paged
                .append(2, &short2)
                .expect_err("the third page is full");
            assert!(
                matches!(error, PagingError::Capacity { .. }),
                "expected Capacity, got {error}"
            );
            assert_eq!(paged.len(2).expect("seq 2 length"), 64);
            paged.preempt(1).expect("preempt seq 1 frees two pages");
            paged
                .append(2, &short2[64..])
                .expect("seq 2 finishes from the freed pages");
            let mut dense2 = dense(20, short2.len());
            dense2.append(&short2);
            let reference2 = dense2.step(64).expect("dense seq 2 step");
            let seq2 = paged.step(2, 64).expect("seq 2 step");
            assert_same_step(&seq2, &reference2, "seq 2 vs dense");
            paged.free(2).expect("free seq 2");
            let recomputed = paged.recompute(1).expect("recompute seq 1");
            assert!(
                recomputed.tokens >= short1.len(),
                "recompute must restore the tokens"
            );
            let after = paged.step(1, 64).expect("seq 1 after recompute");
            assert_same_step(&before, &after, "preemption/recompute");
            let mut dense1 = dense(20, short1.len());
            dense1.append(&short1);
            let reference1 = dense1.step(64).expect("dense seq 1 step");
            assert_same_step(&after, &reference1, "recomputed vs dense");
        }

        #[test]
        fn absolute_indexer_keys_match_the_dense_reference() {
            let mut paged = paged(20);
            paged.append(1, &prompt()).expect("prefill");
            let mut dense = dense(20, PROMPT_LEN + 1);
            dense.append(&prompt());
            for position in [0usize, 7, 63, 64, 129] {
                assert_eq!(
                    paged.indexer_keys(1, position).expect("paged keys"),
                    dense.indexer_keys(position).expect("dense keys"),
                    "indexer keys differ at position {position}"
                );
            }
        }

        #[test]
        fn randomized_multi_request_trace_keeps_invariants_after_every_step() {
            let mut paged = PagedSequenceState::new(16, 4, 20).expect("paged geometry");
            let mut state = 0x1234_5678_9abc_def0u64;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let mut ids: Vec<u64> = Vec::new();
            let mut next_id = 1u64;
            for step in 0..200 {
                let choice = next() % 6;
                match choice {
                    0 if ids.len() < 4 => {
                        paged.add_sequence(next_id).expect("add");
                        ids.push(next_id);
                        next_id += 1;
                    }
                    1 | 2 if !ids.is_empty() => {
                        let id = ids[(next() as usize) % ids.len()];
                        let count = (next() % 8) as usize + 1;
                        let base = (next() % 1000) as u32;
                        let tokens: Vec<u32> =
                            (0..count as u32).map(|index| base + index).collect();
                        let _ = paged.append(id, &tokens);
                    }
                    3 if !ids.is_empty() => {
                        let id = ids[(next() as usize) % ids.len()];
                        let _ = paged.preempt(id);
                    }
                    4 if !ids.is_empty() => {
                        let id = ids[(next() as usize) % ids.len()];
                        let _ = paged.recompute(id);
                    }
                    5 if !ids.is_empty() => {
                        let index = (next() as usize) % ids.len();
                        let id = ids.remove(index);
                        let _ = paged.free(id);
                    }
                    _ => {}
                }
                paged
                    .check_invariants()
                    .unwrap_or_else(|error| panic!("step {step}: {error}"));
            }
        }
    }
}

// A host-only continuous-batch executor. It is crate-internal because the serve driver still
// runs one job at a time; task-22 will consume it from the tests and the device path can adopt
// the same admission shape later. The dead-code allowance is for the non-test build only.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod continuous {
    //! Host continuous batching over one reference engine.
    //!
    //! One reference engine can drive several SequenceStates. This module is the host
    //! scheduler/executor that uses that fact: FIFO admission, one prompt chunk per prefilling
    //! request per step, round-robin decode for admitted requests, and a deterministic record of
    //! which request advanced at which step. The chunk plan is the scheduler crate's
    //! ChunkedPrefillPlan; the fairness policy is the FIFO admission plus round-robin execution
    //! stated here.
    //!
    //! It is a host executor, not an HTTP-path switch. The serve driver still runs one job at a
    //! time; this module proves the scheduling semantics and gives the device path the admission
    //! shape it will use. Device batch attention, TTFT and throughput are named window
    //! measurements, not claims made here.

    use sparkflash_engine::forward::{InferenceEngine, SequenceState};
    use sparkflash_scheduler::prefill::ChunkedPrefillPlan;

    /// Admission and execution limits for one continuous batch.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) struct BatchPolicy {
        /// Maximum requests admitted at once.
        pub(crate) max_batch: usize,
        /// Prompt tokens fed per prefill step. The host may choose a size below the device
        /// policy's 4096-token chunk so a test can cross a boundary with a short prompt.
        pub(crate) chunk_tokens: usize,
    }

    impl BatchPolicy {
        /// A policy with a positive batch width and chunk size.
        ///
        /// # Errors
        /// A zero width or chunk size.
        pub(crate) fn new(max_batch: usize, chunk_tokens: usize) -> Result<Self, String> {
            if max_batch == 0 {
                return Err("a continuous batch needs max_batch >= 1".to_owned());
            }
            if chunk_tokens == 0 {
                return Err("a continuous batch needs chunk_tokens >= 1".to_owned());
            }
            Ok(Self {
                max_batch,
                chunk_tokens,
            })
        }
    }

    /// One request offered to the batch.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct RequestSpec {
        /// Stable request id; the same id keys the output.
        pub(crate) id: u64,
        /// Prompt token ids; must not be empty.
        pub(crate) prompt: Vec<u32>,
        /// Generated-token cap. Zero prefills and finishes without a token.
        pub(crate) max_tokens: usize,
        /// Scheduler step at which the request may first be admitted.
        pub(crate) arrival_step: u64,
    }

    impl RequestSpec {
        /// A request that arrived before the batch starts.
        #[must_use]
        pub(crate) fn new(id: u64, prompt: Vec<u32>, max_tokens: usize) -> Self {
            Self {
                id,
                prompt,
                max_tokens,
                arrival_step: 0,
            }
        }

        /// The same request arriving at arrival_step.
        #[must_use]
        pub(crate) fn arriving_at(mut self, arrival_step: u64) -> Self {
            self.arrival_step = arrival_step;
            self
        }
    }

    /// Which engine operation advanced one request in a step.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum BatchOperation {
        /// One contiguous prompt chunk was fed through InferenceEngine::prefill.
        Prefill,
        /// One generated token was fed back through InferenceEngine::decode.
        Decode,
    }

    /// One request's operation in one step.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct StepOperation {
        /// Request id.
        pub(crate) request: u64,
        /// What ran.
        pub(crate) operation: BatchOperation,
        /// Tokens added to the request's sequence by this operation: a prompt chunk for
        /// Prefill, the emitted token for Decode.
        pub(crate) appended: Vec<u32>,
    }

    /// What ran in one scheduler step.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct BatchStep {
        /// Zero-based scheduler step.
        pub(crate) step: u64,
        /// Operations in execution order. A request appears at most once per step.
        pub(crate) operations: Vec<StepOperation>,
    }

    /// One request's generated output.
    #[derive(Debug, Clone, PartialEq)]
    pub(crate) struct RequestOutput {
        /// Request id.
        pub(crate) id: u64,
        /// Generated tokens, in order.
        pub(crate) tokens: Vec<u32>,
        /// Raw logits that produced each token: the prefill step for the first token, then one
        /// decode step per later token. tokens.len() == logits.len().
        pub(crate) logits: Vec<Vec<f32>>,
        /// Step the request was admitted at.
        pub(crate) admitted_step: u64,
        /// Step its first token was emitted at; None for a zero-token request.
        pub(crate) first_token_step: Option<u64>,
        /// Step it finished at.
        pub(crate) finished_step: u64,
        /// Engine operations this request consumed (prefill chunks plus decodes).
        pub(crate) steps_advanced: usize,
    }

    /// The whole batch's outputs and the exact schedule that produced them.
    #[derive(Debug, Clone, PartialEq)]
    pub(crate) struct BatchReport {
        /// One entry per request, ordered by id.
        pub(crate) outputs: Vec<RequestOutput>,
        /// Every step, in order, with the operations that ran in it.
        pub(crate) steps: Vec<BatchStep>,
        /// Batch width used.
        pub(crate) max_batch: usize,
        /// Prefill chunk size used.
        pub(crate) chunk_tokens: usize,
    }

    /// One active request: its own dense engine state, its chunk plan and its generated tokens.
    struct ActiveRequest {
        spec: RequestSpec,
        state: SequenceState,
        plan: ChunkedPrefillPlan,
        cursor: usize,
        generated: Vec<u32>,
        logits: Vec<Vec<f32>>,
        last_token: Option<u32>,
        admitted_step: u64,
        first_token_step: Option<u64>,
        finished_step: Option<u64>,
        steps_advanced: usize,
        finished: bool,
    }

    impl ActiveRequest {
        fn new(
            spec: RequestSpec,
            state: SequenceState,
            plan: ChunkedPrefillPlan,
            admitted_step: u64,
        ) -> Self {
            Self {
                spec,
                state,
                plan,
                cursor: 0,
                generated: Vec::new(),
                logits: Vec::new(),
                last_token: None,
                admitted_step,
                first_token_step: None,
                finished_step: None,
                steps_advanced: 0,
                finished: false,
            }
        }

        /// One engine operation. The prefill path emits the first token when it feeds the last
        /// prompt chunk; the decode path feeds the previous token and emits the next one.
        fn advance(
            &mut self,
            engine: &InferenceEngine,
            step: u64,
        ) -> Result<StepOperation, String> {
            self.steps_advanced += 1;
            if self.cursor < self.plan.ranges().len() {
                let range = self.plan.ranges()[self.cursor].clone();
                let chunk = self
                    .spec
                    .prompt
                    .get(range.clone())
                    .ok_or_else(|| "a prefill chunk lies outside the prompt".to_owned())?
                    .to_vec();
                let output = engine
                    .prefill(&mut self.state, &chunk)
                    .map_err(|error| format!("prefill for request {}: {error}", self.spec.id))?;
                self.cursor += 1;
                let mut operation = StepOperation {
                    request: self.spec.id,
                    operation: BatchOperation::Prefill,
                    appended: chunk,
                };
                if self.cursor == self.plan.ranges().len() {
                    if self.spec.max_tokens == 0 {
                        self.finish(step);
                    } else if let Some(token) = output.next_token {
                        self.emit(step, output, token, &mut operation);
                    } else {
                        self.finish(step);
                    }
                }
                return Ok(operation);
            }
            let token = self
                .last_token
                .ok_or_else(|| format!("request {} decodes before a first token", self.spec.id))?;
            let output = engine
                .decode(&mut self.state, token)
                .map_err(|error| format!("decode for request {}: {error}", self.spec.id))?;
            let mut operation = StepOperation {
                request: self.spec.id,
                operation: BatchOperation::Decode,
                appended: Vec::new(),
            };
            if self.generated.len() >= self.spec.max_tokens {
                self.finish(step);
            } else if let Some(next) = output.next_token {
                self.emit(step, output, next, &mut operation);
            } else {
                self.finish(step);
            }
            Ok(operation)
        }

        fn emit(
            &mut self,
            step: u64,
            output: sparkflash_engine::forward::StepOutput,
            token: u32,
            operation: &mut StepOperation,
        ) {
            if self.first_token_step.is_none() {
                self.first_token_step = Some(step);
            }
            self.generated.push(token);
            self.logits.push(output.logits);
            self.last_token = Some(token);
            operation.appended.push(token);
            if self.generated.len() >= self.spec.max_tokens {
                self.finish(step);
            }
        }

        fn finish(&mut self, step: u64) {
            self.finished = true;
            self.finished_step = Some(step);
        }

        fn into_output(self) -> RequestOutput {
            RequestOutput {
                id: self.spec.id,
                tokens: self.generated,
                logits: self.logits,
                admitted_step: self.admitted_step,
                first_token_step: self.first_token_step,
                finished_step: self.finished_step.unwrap_or(self.admitted_step),
                steps_advanced: self.steps_advanced,
            }
        }
    }

    /// The continuous-batch executor: pending requests, active requests and the step clock.
    pub(crate) struct ContinuousBatch {
        policy: BatchPolicy,
        pending: Vec<RequestSpec>,
        active: Vec<ActiveRequest>,
        outputs: Vec<RequestOutput>,
        step: u64,
        cursor: usize,
    }

    impl ContinuousBatch {
        /// An empty batch under policy.
        ///
        /// # Errors
        /// An invalid policy.
        pub(crate) fn new(policy: BatchPolicy) -> Result<Self, String> {
            Ok(Self {
                policy,
                pending: Vec::new(),
                active: Vec::new(),
                outputs: Vec::new(),
                step: 0,
                cursor: 0,
            })
        }

        /// True when no pending request remains and no active request is unfinished.
        #[must_use]
        pub(crate) fn is_done(&self) -> bool {
            self.pending.is_empty() && self.active.is_empty()
        }

        /// Offers one request. Pending requests are kept in arrival order, and equal arrivals in
        /// offer order, so a later offer cannot overtake an earlier one.
        ///
        /// # Errors
        /// An empty prompt, a duplicate id, or an arrival in the past.
        pub(crate) fn submit(&mut self, spec: RequestSpec) -> Result<(), String> {
            if spec.prompt.is_empty() {
                return Err(format!("request {} has an empty prompt", spec.id));
            }
            let duplicate = self.pending.iter().any(|pending| pending.id == spec.id)
                || self.active.iter().any(|active| active.spec.id == spec.id)
                || self.outputs.iter().any(|output| output.id == spec.id);
            if duplicate {
                return Err(format!("request id {} is already known", spec.id));
            }
            if spec.arrival_step < self.step {
                return Err(format!(
                    "request {} arrived at step {}, before the current step {}",
                    spec.id, spec.arrival_step, self.step
                ));
            }
            self.pending.push(spec);
            self.pending.sort_by_key(|pending| pending.arrival_step);
            Ok(())
        }

        /// Admits every eligible request that fits under the batch width.
        fn admit(&mut self, engine: &InferenceEngine) -> Result<(), String> {
            while self.active.len() < self.policy.max_batch {
                let Some(front) = self.pending.first() else {
                    break;
                };
                if front.arrival_step > self.step {
                    break;
                }
                let spec = self.pending.remove(0);
                let plan = ChunkedPrefillPlan::with_chunk_tokens(
                    spec.prompt.len(),
                    self.step,
                    self.policy.chunk_tokens,
                    0,
                )
                .map_err(|error| format!("prefill plan for request {}: {error}", spec.id))?;
                let state = SequenceState::new(engine.config());
                self.active
                    .push(ActiveRequest::new(spec, state, plan, self.step));
            }
            Ok(())
        }

        /// Runs one scheduler step.
        ///
        /// Admission is FIFO. Every active request advances exactly once, in round-robin order,
        /// so a decoding request is never starved by another request's prefill chunks. The
        /// returned record is the schedule the caller can replay against a paged mirror.
        ///
        /// # Errors
        /// An engine failure or a malformed request.
        pub(crate) fn step_once(&mut self, engine: &InferenceEngine) -> Result<BatchStep, String> {
            if self.active.is_empty()
                && let Some(next) = self.pending.first()
            {
                // Do not spin over steps in which no request can be admitted.
                self.step = self.step.max(next.arrival_step);
            }
            self.admit(engine)?;
            if self.active.is_empty() {
                let step = self.step;
                self.step += 1;
                return Ok(BatchStep {
                    step,
                    operations: Vec::new(),
                });
            }
            let active_len = self.active.len();
            let start = self.cursor % active_len;
            let mut operations = Vec::with_capacity(active_len);
            for offset in 0..active_len {
                let index = (start + offset) % active_len;
                let operation = self.active[index].advance(engine, self.step)?;
                operations.push(operation);
            }
            let next_cursor = (start + 1) % active_len;
            let mut still_active = Vec::with_capacity(active_len);
            for request in std::mem::take(&mut self.active) {
                if request.finished {
                    self.outputs.push(request.into_output());
                } else {
                    still_active.push(request);
                }
            }
            self.active = still_active;
            self.cursor = if self.active.is_empty() {
                0
            } else {
                next_cursor % self.active.len()
            };
            let step = self.step;
            self.step += 1;
            Ok(BatchStep { step, operations })
        }

        /// Runs until every submitted request finishes.
        ///
        /// # Errors
        /// An engine failure, or a schedule that does not terminate.
        pub(crate) fn run(&mut self, engine: &InferenceEngine) -> Result<BatchReport, String> {
            let mut steps = Vec::new();
            while !self.is_done() {
                let step = self.step_once(engine)?;
                steps.push(step);
                if steps.len() > 1_000_000 {
                    return Err("the continuous batch did not terminate".to_owned());
                }
            }
            let mut outputs = std::mem::take(&mut self.outputs);
            outputs.sort_by_key(|output| output.id);
            Ok(BatchReport {
                outputs,
                steps,
                max_batch: self.policy.max_batch,
                chunk_tokens: self.policy.chunk_tokens,
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use std::collections::HashMap;

        use super::super::paged_serve::{DenseSequenceState, PagedSequenceState, ServedStep};
        use super::super::{
            ModelConfig, REFERENCE_TENSOR_LIMIT, SYNTHETIC_WEIGHT_SEED, SyntheticWeights,
            TensorPlan,
        };
        use super::*;

        fn engine() -> InferenceEngine {
            let config = ModelConfig::tiny();
            let plan = TensorPlan::language_model(&config);
            InferenceEngine::new(
                config,
                Box::new(SyntheticWeights::from_plan(&plan, SYNTHETIC_WEIGHT_SEED)),
                REFERENCE_TENSOR_LIMIT,
            )
            .expect("the tiny engine builds")
        }

        fn run(
            engine: &InferenceEngine,
            policy: BatchPolicy,
            specs: Vec<RequestSpec>,
        ) -> BatchReport {
            let mut batch = ContinuousBatch::new(policy).expect("policy");
            for spec in specs {
                batch.submit(spec).expect("submit");
            }
            batch.run(engine).expect("batch run")
        }

        fn output(report: &BatchReport, id: u64) -> &RequestOutput {
            report
                .outputs
                .iter()
                .find(|output| output.id == id)
                .expect("request output")
        }

        fn assert_output_equal(left: &RequestOutput, right: &RequestOutput, context: &str) {
            assert_eq!(left.tokens, right.tokens, "tokens differ: {context}");
            assert_eq!(
                left.logits.len(),
                right.logits.len(),
                "logit count differs: {context}"
            );
            for (index, (left_row, right_row)) in
                left.logits.iter().zip(right.logits.iter()).enumerate()
            {
                assert_eq!(
                    left_row
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    right_row
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "logits bits differ at token {index}: {context}"
                );
            }
        }

        fn assert_same_served(left: &ServedStep, right: &ServedStep, context: &str) {
            assert_eq!(
                left.attention
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                right
                    .attention
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "attention bits differ: {context}"
            );
            assert_eq!(
                left.logits
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                right
                    .logits
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "reference-head logits bits differ: {context}"
            );
            assert_eq!(
                left.next_token, right.next_token,
                "next token differs: {context}"
            );
        }

        fn concurrent_specs() -> Vec<RequestSpec> {
            vec![
                RequestSpec::new(1, vec![1, 2, 3, 4, 5], 4),
                RequestSpec::new(2, (6..=12).collect(), 4),
                RequestSpec::new(3, vec![13, 14, 15], 4),
            ]
        }

        #[test]
        fn interleaved_requests_match_their_solo_outputs_bit_for_bit() {
            let engine = engine();
            let specs = concurrent_specs();
            let batch = run(
                &engine,
                BatchPolicy::new(3, 2).expect("policy"),
                specs.clone(),
            );
            for spec in specs {
                let solo = run(
                    &engine,
                    BatchPolicy::new(1, 2).expect("policy"),
                    vec![spec.clone()],
                );
                assert_output_equal(
                    &solo.outputs[0],
                    output(&batch, spec.id),
                    &format!("request {} alone versus interleaved", spec.id),
                );
            }
            let mixed_step = batch.steps.iter().any(|step| {
                let prefill = step
                    .operations
                    .iter()
                    .any(|operation| operation.operation == BatchOperation::Prefill);
                let decode = step
                    .operations
                    .iter()
                    .any(|operation| operation.operation == BatchOperation::Decode);
                prefill && decode
            });
            assert!(
                mixed_step,
                "the batch must actually interleave a decode with a prefill"
            );
            assert!(
                batch.steps.iter().any(|step| step.operations.len() > 1),
                "the batch must run more than one request in a step"
            );
        }

        #[test]
        fn chunk_boundaries_and_decode_windows_do_not_change_the_result() {
            let engine = engine();
            let specs = concurrent_specs();
            let baseline = run(
                &engine,
                BatchPolicy::new(3, 2).expect("policy"),
                specs.clone(),
            );
            for chunk in [1usize, 3, 4, 8] {
                let got = run(
                    &engine,
                    BatchPolicy::new(3, chunk).expect("policy"),
                    specs.clone(),
                );
                for spec in &specs {
                    assert_output_equal(
                        output(&baseline, spec.id),
                        output(&got, spec.id),
                        &format!("chunk {chunk}, request {}", spec.id),
                    );
                }
            }
            for width in [1usize, 2, 3] {
                let got = run(
                    &engine,
                    BatchPolicy::new(width, 2).expect("policy"),
                    specs.clone(),
                );
                for spec in &specs {
                    assert_output_equal(
                        output(&baseline, spec.id),
                        output(&got, spec.id),
                        &format!("batch width {width}, request {}", spec.id),
                    );
                }
            }
            let chunked = run(
                &engine,
                BatchPolicy::new(3, 1).expect("policy"),
                specs.clone(),
            );
            assert_ne!(
                baseline.steps.len(),
                chunked.steps.len(),
                "the chunk size must actually change the schedule"
            );
        }

        #[test]
        fn admission_is_fifo_and_a_later_arrival_cannot_overtake() {
            let engine = engine();
            let specs = vec![
                RequestSpec::new(1, vec![1, 2, 3, 4], 1),
                RequestSpec::new(2, vec![5, 6, 7, 8], 1),
                RequestSpec::new(3, vec![9, 10, 11, 12], 1).arriving_at(1),
                RequestSpec::new(4, vec![13, 14, 15, 16], 1).arriving_at(1),
            ];
            let report = run(&engine, BatchPolicy::new(2, 2).expect("policy"), specs);
            let first = output(&report, 1);
            let second = output(&report, 2);
            let third = output(&report, 3);
            let fourth = output(&report, 4);
            assert_eq!(first.admitted_step, 0);
            assert_eq!(second.admitted_step, 0);
            assert_eq!(third.admitted_step, 2, "the third request waits for a slot");
            assert_eq!(fourth.admitted_step, 2);
            assert!(
                third.admitted_step <= fourth.admitted_step,
                "no later arrival overtakes"
            );
            assert_eq!(first.finished_step, 1);
            assert_eq!(second.finished_step, 1);
        }

        #[test]
        fn round_robin_advances_every_active_request_once_per_step() {
            let engine = engine();
            let specs = vec![
                RequestSpec::new(1, (1..=21).collect(), 2),
                RequestSpec::new(2, (21..=41).collect(), 2),
                RequestSpec::new(3, vec![41, 42, 43, 44], 3),
            ];
            let report = run(&engine, BatchPolicy::new(3, 1).expect("policy"), specs);
            for step in &report.steps {
                let mut requests: Vec<u64> = step
                    .operations
                    .iter()
                    .map(|operation| operation.request)
                    .collect();
                let count = requests.len();
                requests.sort_unstable();
                requests.dedup();
                assert_eq!(
                    requests.len(),
                    count,
                    "a request advanced twice in one step"
                );
            }
            let short = output(&report, 3);
            let long = output(&report, 1);
            assert!(
                short.finished_step < long.finished_step,
                "the short request must not starve behind a long prefill"
            );
            for output in &report.outputs {
                let operations = report
                    .steps
                    .iter()
                    .flat_map(|step| step.operations.iter())
                    .filter(|operation| operation.request == output.id)
                    .count();
                assert_eq!(output.steps_advanced, operations);
            }
        }

        #[test]
        fn paged_mirrors_stay_invariant_and_match_dense_after_every_step() {
            let engine = engine();
            let specs = vec![
                RequestSpec::new(1, (1..=6).collect(), 2),
                RequestSpec::new(2, (6..=11).collect(), 2),
            ];
            let mut batch =
                ContinuousBatch::new(BatchPolicy::new(2, 2).expect("policy")).expect("batch");
            for spec in &specs {
                batch.submit(spec.clone()).expect("submit");
            }
            let mut paged = PagedSequenceState::new(8, 2, 20).expect("paged geometry");
            let mut dense: HashMap<u64, DenseSequenceState> = HashMap::new();
            for spec in &specs {
                paged.add_sequence(spec.id).expect("paged sequence");
                dense.insert(
                    spec.id,
                    DenseSequenceState::new(20, spec.prompt.len() + spec.max_tokens + 1)
                        .expect("dense state"),
                );
            }
            let mut positions: HashMap<u64, usize> = HashMap::new();
            while !batch.is_done() {
                let step = batch.step_once(&engine).expect("batch step");
                for operation in &step.operations {
                    paged
                        .append(operation.request, &operation.appended)
                        .expect("paged append");
                    dense
                        .get_mut(&operation.request)
                        .expect("dense state")
                        .append(&operation.appended);
                    *positions.entry(operation.request).or_insert(0) += operation.appended.len();
                }
                paged
                    .check_invariants()
                    .expect("paged invariant after the step");
                for (id, mirror) in &dense {
                    let position = positions.get(id).copied().unwrap_or(0);
                    if position > 0 {
                        let paged_step = paged.step(*id, position - 1).expect("paged step");
                        let dense_step = mirror.step(position - 1).expect("dense step");
                        assert_same_served(
                            &paged_step,
                            &dense_step,
                            &format!("request {id} at position {}", position - 1),
                        );
                    }
                }
            }
        }

        #[test]
        fn a_zero_token_request_finishes_without_losing_a_slot() {
            let engine = engine();
            let specs = vec![
                RequestSpec::new(1, vec![1, 2, 3], 0),
                RequestSpec::new(2, vec![4, 5, 6], 2),
            ];
            let report = run(&engine, BatchPolicy::new(2, 2).expect("policy"), specs);
            let zero = output(&report, 1);
            assert!(zero.tokens.is_empty());
            assert!(zero.logits.is_empty());
            assert_eq!(zero.first_token_step, None);
            assert_eq!(
                zero.steps_advanced, 2,
                "the zero cap still prefills in two chunks"
            );
            assert_eq!(output(&report, 2).tokens.len(), 2);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use sparkflash_engine::loader::Treatment;

    use super::*;
    use sparkflash_model_deepseek_v41::encoding::reference_tokenizer_path;

    /// A driver over the synthetic tiny plan: the path every behavioural test takes.
    ///
    /// Its random weight bytes decode to all-NaN logits in the tiny plan, which is useful for the
    /// non-finite guard but not for measuring acceptance. **Any synthetic test that asserts a
    /// non-zero DSpark accepted count must use [`zero_driver`] instead**; a non-zero assertion here
    /// would only be measuring a NaN argmax.
    fn synthetic_driver(max_tokens: usize) -> ReferenceDriver {
        let recipe = Recipe {
            config: ModelConfig::tiny(),
            weights: Weights::Synthetic {
                seed: SYNTHETIC_WEIGHT_SEED,
            },
            tokenizer: reference_tokenizer_path(),
            engram_dir: None,
            batch: None,
        };
        ReferenceDriver::start(&recipe, REFERENCE_TENSOR_LIMIT, max_tokens)
            .expect("the synthetic driver starts")
    }

    /// A driver over the tiny plan with every weight zero: finite logits, so acceptance is real.
    /// This is the only synthetic driver a test may use when it asserts a non-zero accepted count;
    /// see [`synthetic_driver`] for why the random-weight source cannot support that assertion.
    fn zero_driver(max_tokens: usize) -> ReferenceDriver {
        let recipe = Recipe {
            config: ModelConfig::tiny(),
            weights: Weights::Zero,
            tokenizer: reference_tokenizer_path(),
            engram_dir: None,
            batch: None,
        };
        ReferenceDriver::start(&recipe, REFERENCE_TENSOR_LIMIT, max_tokens)
            .expect("the zero-weight driver starts")
    }

    /// A driver over the synthetic tiny plan with the S4 host scheduler enabled.
    fn batched_driver(max_tokens: usize, max_batch: usize) -> ReferenceDriver {
        let recipe = Recipe {
            config: ModelConfig::tiny(),
            weights: Weights::Synthetic {
                seed: SYNTHETIC_WEIGHT_SEED,
            },
            tokenizer: reference_tokenizer_path(),
            engram_dir: None,
            batch: Some(max_batch),
        };
        ReferenceDriver::start(&recipe, REFERENCE_TENSOR_LIMIT, max_tokens)
            .expect("the batched synthetic driver starts")
    }

    /// Drains two request streams concurrently and reports the order their terminal events
    /// arrived in.
    async fn finish_order(
        first: BoxStream<'static, EngineEvent>,
        second: BoxStream<'static, EngineEvent>,
    ) -> (Vec<&'static str>, Vec<EngineEvent>, Vec<EngineEvent>) {
        let (sender, mut receiver) =
            tokio::sync::mpsc::channel::<(&'static str, EngineEvent)>(4 * REPLY_BUFFER);
        for (label, stream) in [("first", first), ("second", second)] {
            let sender = sender.clone();
            tokio::spawn(async move {
                let mut stream = stream;
                while let Some(event) = stream.next().await {
                    if sender.send((label, event)).await.is_err() {
                        break;
                    }
                }
            });
        }
        drop(sender);
        let mut order = Vec::new();
        let mut first_events = Vec::new();
        let mut second_events = Vec::new();
        let mut terminal = 0usize;
        while terminal < 2 {
            let Some((label, event)) = receiver.recv().await else {
                break;
            };
            let is_terminal = matches!(
                event,
                EngineEvent::Finished { .. } | EngineEvent::Error { .. }
            );
            if label == "first" {
                first_events.push(event);
            } else {
                second_events.push(event);
            }
            if is_terminal {
                terminal += 1;
                order.push(label);
            }
        }
        (order, first_events, second_events)
    }

    /// What two runs of the same logprob request must agree on without comparing the NaN logprob
    /// values the synthetic tiny plan produces.
    fn logprob_shape(
        stream: &[EngineEvent],
    ) -> (String, Vec<u32>, Vec<u32>, Option<FinishReason>, usize) {
        let mut text = String::new();
        let mut prompt_ids = Vec::new();
        let mut sample_ids = Vec::new();
        let mut finish = None;
        let mut completion = 0usize;
        for event in stream {
            match event {
                EngineEvent::ContentDelta(delta) => text.push_str(delta),
                EngineEvent::PromptLogprobs(payload) => {
                    for position in &payload.positions {
                        let ids: Vec<u32> = position.as_ref().map_or_else(Vec::new, |entry| {
                            let mut ids = vec![entry.token_id];
                            ids.extend(entry.top.iter().map(|alternative| alternative.token_id));
                            ids
                        });
                        prompt_ids.push(ids.len() as u32);
                        prompt_ids.extend(ids);
                    }
                }
                EngineEvent::TokenLogprobs(logprob) => {
                    let mut ids = vec![logprob.token_id];
                    ids.extend(logprob.top.iter().map(|alternative| alternative.token_id));
                    sample_ids.push(ids.len() as u32);
                    sample_ids.extend(ids);
                }
                EngineEvent::Finished {
                    finish_reason,
                    usage,
                    ..
                } => {
                    finish = Some(*finish_reason);
                    completion = usage.completion_tokens;
                }
                _ => {}
            }
        }
        (text, prompt_ids, sample_ids, finish, completion)
    }

    #[tokio::test]
    async fn continuous_batching_admits_both_requests_and_keeps_their_output() {
        let long_request = request(&[1, 2, 3], 24, 0.0, 11);
        let short_request = request(&[4, 5, 6], 2, 0.0, 13);
        let serial = synthetic_driver(64);
        let serial_long = without_measurements(&events(&serial, long_request.clone()).await);
        let serial_short = without_measurements(&events(&serial, short_request.clone()).await);

        let batched = batched_driver(64, 8);
        let (order, long_events, short_events) = finish_order(
            batched.generate(long_request),
            batched.generate(short_request),
        )
        .await;
        assert_eq!(
            order,
            ["second", "first"],
            "the short request must not wait for the long one to finish: {order:?}"
        );
        assert_eq!(
            without_measurements(&long_events),
            serial_long,
            "interleaving must not change the long request's tokens, logprobs or finish reason"
        );
        assert_eq!(
            without_measurements(&short_events),
            serial_short,
            "interleaving must not change the short request's tokens, logprobs or finish reason"
        );
    }

    #[tokio::test]
    async fn continuous_batching_carries_prompt_logprobs_and_stop_strings() {
        let mut scored = request(&[1, 2, 3, 4], 8, 0.0, 3);
        scored.logprobs = LogprobsRequest {
            sample: Some(LogprobSelection::Top(2)),
            prompt: Some(LogprobSelection::Top(2)),
            token_ids: Vec::new(),
        };
        // The tiny plan's generated token decodes to the BOS special token, whose text contains
        // a b; with skip_special_tokens false that is the first stop-string match.
        scored.stop = vec!["b".to_owned()];
        let filler = request(&[5, 6], 3, 0.0, 5);

        let serial = synthetic_driver(64);
        let serial_events = without_measurements(&events(&serial, scored.clone()).await);
        let batched = batched_driver(64, 4);
        let (_, batched_events, filler_events) =
            finish_order(batched.generate(scored), batched.generate(filler)).await;
        assert_eq!(
            logprob_shape(&without_measurements(&batched_events)),
            logprob_shape(&serial_events),
            "the batched request's prompt logprobs, sample logprobs, stop match and counts must match the serial path"
        );
        assert!(
            !filler_events.is_empty(),
            "the filler request shares the scheduler round and its own stream is intact"
        );
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
            batch: None,
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
            vision: Vec::new(),
            max_tokens,
            temperature,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            // These driver tests compare the emitted text with a raw decode_stream(false) pass,
            // so they ask for the explicit false passthrough; the default hiding has its own
            // test below.
            skip_special_tokens: false,
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
        // No image was accepted, no collective ran, no rank was visible.
        assert_eq!(usage.prompt_tokens_details.image_tokens, 0);
        assert_eq!(stats.transport, RoceTransportStats::default());
        assert_eq!(stats.timing.min_rank_uma_free_gib, 0.0);
        // The synthetic tiny engine's random weights decode to all-NaN logits. The non-finite guard
        // must refuse every acceptance position, so these counters are the honest zeros, and the
        // rejected count cannot invent a draft either.
        assert_eq!(stats.dspark_speculative, DSparkStepStats::default());
        assert_eq!(
            usage.completion_tokens_details.accepted_prediction_tokens,
            0
        );
        assert_eq!(
            usage.completion_tokens_details.rejected_prediction_tokens,
            0
        );
    }

    /// The acceptance rule is the longest matching draft prefix: three matches, then the target's
    /// own token is the bonus. The accounting must count exactly those three, not the whole block.
    #[test]
    fn dspark_accounting_replays_the_acceptance_rule_on_mtp_drafts() {
        let mut accounting = SpeculativeAccounting::default();
        accounting.observe_token(10, &[]);
        accounting.observe_token(11, &[12, 13, 14]);
        accounting.observe_token(12, &[]);
        accounting.observe_token(13, &[]);
        accounting.observe_token(14, &[]);
        accounting.observe_token(99, &[]);
        accounting.finish();
        assert_eq!(accounting.target_verify_steps, 1);
        assert_eq!(accounting.draft_tokens_proposed, 3);
        assert_eq!(accounting.draft_tokens_accepted, 3);
        let stats = accounting.stats();
        assert_eq!(stats.draft_tokens_accepted, 3);
        assert_eq!(stats.draft_tokens_confidence_gated, 0);
        assert_eq!(stats.acceptance_rate, 1.0);
        assert_eq!(stats.mean_tokens_per_verify_step, 4.0);
    }

    /// A mismatch ends the block: the mismatching token is the bonus token, and the next block is
    /// anchored there. The rejected candidates in the old block stay rejected.
    #[test]
    fn dspark_accounting_starts_the_next_block_at_the_bonus_token() {
        let mut accounting = SpeculativeAccounting::default();
        accounting.observe_token(10, &[]);
        accounting.observe_token(11, &[12, 99, 14]);
        accounting.observe_token(12, &[]);
        accounting.observe_token(77, &[13]);
        accounting.observe_token(13, &[]);
        accounting.observe_token(14, &[]);
        accounting.finish();
        assert_eq!(accounting.target_verify_steps, 2);
        assert_eq!(accounting.draft_tokens_proposed, 4);
        assert_eq!(accounting.draft_tokens_accepted, 2);
    }

    /// Every candidate is checked; the scheduler's confidence cap is not part of the accept rule, so
    /// a five-of-five match is five accepted tokens.
    #[test]
    fn dspark_accounting_accepts_every_matching_position() {
        let mut accounting = SpeculativeAccounting::default();
        accounting.observe_token(10, &[]);
        accounting.observe_token(11, &[12, 13, 14, 15, 16]);
        for token in 12..=16 {
            accounting.observe_token(token, &[]);
        }
        accounting.observe_token(99, &[]);
        accounting.finish();
        assert_eq!(accounting.target_verify_steps, 1);
        assert_eq!(accounting.draft_tokens_proposed, 5);
        assert_eq!(accounting.draft_tokens_accepted, 5);
        assert_eq!(accounting.stats().draft_tokens_confidence_gated, 0);
    }

    /// NaN and infinity are never valid target rows: neither is an argmax the accept rule may use.
    #[test]
    fn finite_target_argmax_refuses_nan_and_infinity() {
        assert_eq!(finite_target_argmax(&[0.1, 0.2]), Some(1));
        assert_eq!(finite_target_argmax(&[f32::NAN, 0.2]), None);
        assert_eq!(finite_target_argmax(&[0.1, f32::INFINITY]), None);
        assert_eq!(finite_target_argmax(&[0.1, f32::NEG_INFINITY]), None);
        assert_eq!(finite_target_argmax(&[]), None);
    }

    /// A non-finite target row truncates the prefix: the finite matches before it stand, the row
    /// itself is never accepted, no block starts from it, and the evidence carries its index.
    #[test]
    fn dspark_accounting_truncates_at_a_nonfinite_target_row() {
        let mut accounting = SpeculativeAccounting::default();
        accounting.observe_token(10, &[]);
        accounting.observe_token(11, &[12, 13]);
        accounting.observe_token(12, &[]);
        accounting.observe_nonfinite(&[]);
        accounting.observe_token(13, &[]);
        accounting.observe_nonfinite(&[15]);
        accounting.finish();
        assert_eq!(accounting.target_verify_steps, 1);
        assert_eq!(accounting.draft_tokens_proposed, 2);
        assert_eq!(accounting.draft_tokens_accepted, 1);
        assert_eq!(accounting.rows_seen, 3);
        assert_eq!(
            accounting.nonfinite_rows,
            [1],
            "the active block failed on its second draft row"
        );
        assert_eq!(
            accounting.evidence_line(),
            "accept_status=nonfinite accept_nonfinite_rows=[1] accept_rows_seen=3 accept_committed_tokens=1 accept_rows_convention=drafts_only"
        );
        assert_eq!(accounting.stats().acceptance_rate, 0.5);
    }

    /// A block that starts at token `t` must not be compared against `t` itself: the token is the
    /// block's anchor, and the block's first draft predicts `t + 1`. This pins the `active.is_none()`
    /// start-then-compare state machine, so an apparent off-by-one cannot be introduced silently by
    /// comparing during the start step. (The random-weight and zero-weight synthetic models both
    /// hide that class of bug, which is why the invariant is tested directly.)
    #[test]
    fn a_new_block_is_started_at_its_anchor_without_comparing_the_anchor() {
        let mut accounting = SpeculativeAccounting::default();
        // Iteration 0 has no previous block: an empty candidate list must contribute no rows seen
        // and must not leave an active block behind.
        accounting.observe_token(10, &[]);
        assert_eq!(accounting.rows_seen, 0);
        assert!(accounting.active.is_none());

        // Token 11 is now the anchor; the block's draft rows are 12 and 13 and are only compared
        // from token 12 onward. `rows_seen` is the offered-row denominator, so the block's rows are
        // in it as soon as the block starts, but no row has been accepted at the anchor.
        accounting.observe_token(11, &[12, 13]);
        assert_eq!(accounting.draft_tokens_accepted, 0);
        assert_eq!(accounting.rows_seen, 2);
        let active = accounting.active.as_ref().expect("the block started");
        assert_eq!(active.matched, 0, "the anchor was compared to a draft row");

        accounting.observe_token(12, &[]);
        assert_eq!(accounting.draft_tokens_accepted, 0);
        assert_eq!(
            accounting.active.as_ref().expect("still active").matched,
            1,
            "the first draft row is compared against token 12, not the anchor"
        );

        accounting.observe_token(13, &[]);
        assert_eq!(accounting.draft_tokens_accepted, 2);
        assert!(accounting.active.is_none());
    }

    /// The finite path uses the same four acceptance keys plus the convention marker and reports
    /// the full prefix.
    #[test]
    fn dspark_accounting_accepts_a_full_prefix_with_uniform_evidence() {
        let mut accounting = SpeculativeAccounting::default();
        accounting.observe_token(10, &[]);
        accounting.observe_token(11, &[12, 13]);
        accounting.observe_token(12, &[]);
        accounting.observe_token(13, &[]);
        accounting.finish();
        assert_eq!(
            accounting.evidence_line(),
            "accept_status=accepted accept_nonfinite_rows=[] accept_rows_seen=2 accept_committed_tokens=2 accept_rows_convention=drafts_only"
        );
    }

    /// A non-finite row before any block starts cannot promote an active draft into acceptance.
    #[test]
    fn dspark_accounting_does_not_start_a_block_from_a_nonfinite_row() {
        let mut accounting = SpeculativeAccounting::default();
        accounting.observe_nonfinite(&[]);
        accounting.observe_token(11, &[12, 13]);
        accounting.observe_token(12, &[]);
        accounting.observe_token(13, &[]);
        accounting.finish();
        assert_eq!(accounting.target_verify_steps, 1);
        assert_eq!(accounting.draft_tokens_accepted, 2);
        assert!(
            accounting.nonfinite_rows.is_empty(),
            "an anchor-only non-finite row has no draft-row id"
        );
        assert!(accounting.nonfinite_seen);
        assert!(
            accounting
                .evidence_line()
                .starts_with("accept_status=nonfinite accept_nonfinite_rows=[]")
        );
    }

    /// A served request with finite logits reports the acceptance rule's count, not a hardcoded
    /// zero. The zero-weight model has all-zero logits and all-zero MTP drafts, so every candidate
    /// matches and the count is measured on the real serving path; the assertions compare the usage
    /// block with the telemetry counters, not with a constant.
    #[tokio::test]
    async fn a_served_request_reports_the_real_mtp_acceptance_count() {
        let driver = zero_driver(128);
        let stream = events(&driver, request(&[1, 2, 3], 128, 0.0, 0)).await;
        let Some(EngineEvent::Finished { usage, stats, .. }) = stream.last() else {
            panic!("the stream did not finish: {stream:?}");
        };
        let dspark = stats.dspark_speculative;
        assert!(
            dspark.draft_tokens_accepted > 0,
            "a 128-token finite request must accept at least one MTP draft: {dspark:?}"
        );
        assert_eq!(
            usage.completion_tokens_details.accepted_prediction_tokens,
            dspark.draft_tokens_accepted
        );
        assert_eq!(
            usage.completion_tokens_details.rejected_prediction_tokens,
            dspark.draft_tokens_proposed - dspark.draft_tokens_accepted
        );
        assert!(
            dspark.draft_tokens_accepted + dspark.draft_tokens_confidence_gated
                <= dspark.draft_tokens_proposed
        );
    }

    /// A served request whose target rows are all non-finite accepts no draft and reports zero in
    /// both the usage block and the DSpark telemetry. The accounting unit tests above pin the finite
    /// rule; the real-checkpoint evidence run is what supplies a non-zero accepted count.
    #[tokio::test]
    async fn a_served_request_with_nonfinite_logits_accepts_nothing() {
        let driver = synthetic_driver(128);
        let stream = events(&driver, request(&[1, 2, 3], 128, 0.0, 0)).await;
        let Some(EngineEvent::Finished { usage, stats, .. }) = stream.last() else {
            panic!("the stream did not finish: {stream:?}");
        };
        assert_eq!(stats.dspark_speculative, DSparkStepStats::default());
        assert_eq!(
            usage.completion_tokens_details.accepted_prediction_tokens,
            0
        );
        assert_eq!(
            usage.completion_tokens_details.rejected_prediction_tokens,
            0
        );
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

    /// vLLM's OpenAI default hides special tokens; an explicit false is the only way to see them.
    #[tokio::test]
    async fn skip_special_tokens_hides_a_generated_special_token_unless_it_is_false() {
        let driver = synthetic_driver(1);
        // The synthetic plan's only generated token is the BOS special token, so the default hides
        // it and the explicit false exposes it; nothing else about the request changes.
        let mut hidden_request = request(&[1, 2, 3], 1, 0.0, 0);
        hidden_request.skip_special_tokens = true;
        let hidden = events(&driver, hidden_request).await;
        let shown = events(&driver, request(&[1, 2, 3], 1, 0.0, 0)).await;
        let text_of = |stream: &[EngineEvent]| -> String {
            stream
                .iter()
                .filter_map(|event| match event {
                    EngineEvent::ContentDelta(text) => Some(text.clone()),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(text_of(&hidden), "", "the default hides the special token");
        assert!(
            text_of(&shown).contains('\u{FF5C}'),
            "an explicit false passes the special token through: {:?}",
            text_of(&shown)
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
    fn the_api_key_flag_parses_and_an_empty_key_is_refused() {
        let open = ServeOptions::parse(vec!["--synthetic".to_owned()]).expect("parses");
        assert!(
            open.api_key.is_none(),
            "no flag means no auth, exactly the pre-task-20 behaviour"
        );
        let keyed = ServeOptions::parse(vec![
            "--synthetic".to_owned(),
            "--api-key".to_owned(),
            "s3cret".to_owned(),
        ])
        .expect("parses");
        assert_eq!(keyed.api_key.as_deref(), Some("s3cret"));
        for empty in ["", "   "] {
            let error = ServeOptions::parse(vec![
                "--synthetic".to_owned(),
                "--api-key".to_owned(),
                empty.to_owned(),
            ])
            .expect_err("an empty bearer key is refused");
            assert!(
                error.contains("--api-key must not be empty"),
                "a blank key must be named, not turned into auth that accepts only a blank token: {error}"
            );
        }
    }

    #[test]
    fn continuous_batching_flags_parse_and_their_misuses_are_refused() {
        let off = ServeOptions::parse(vec!["--synthetic".to_owned()]).expect("parses");
        assert!(
            !off.continuous_batching,
            "the serial one-at-a-time loop is the default"
        );
        assert_eq!(off.max_batch, DEFAULT_MAX_BATCH);

        let on = ServeOptions::parse(vec![
            "--synthetic".to_owned(),
            "--continuous-batching".to_owned(),
        ])
        .expect("parses");
        assert!(on.continuous_batching);
        assert_eq!(on.max_batch, DEFAULT_MAX_BATCH);

        let wide = ServeOptions::parse(vec![
            "--synthetic".to_owned(),
            "--continuous-batching".to_owned(),
            "--max-batch".to_owned(),
            "4".to_owned(),
        ])
        .expect("parses");
        assert_eq!(wide.max_batch, 4);
        let recipe = wide
            .recipe(&reference_tokenizer_path())
            .expect("the recipe resolves");
        assert_eq!(
            recipe.batch,
            Some(4),
            "the parsed cap is what the engine thread reads"
        );

        let without_scheduler = ServeOptions::parse(vec![
            "--synthetic".to_owned(),
            "--max-batch".to_owned(),
            "4".to_owned(),
        ])
        .expect_err("--max-batch alone does nothing");
        assert!(
            without_scheduler.contains("--continuous-batching"),
            "{without_scheduler}"
        );

        let zero = ServeOptions::parse(vec![
            "--synthetic".to_owned(),
            "--continuous-batching".to_owned(),
            "--max-batch".to_owned(),
            "0".to_owned(),
        ])
        .expect_err("a batch of zero admits nothing");
        assert!(zero.contains("at least 1"), "{zero}");

        let too_wide = ServeOptions::parse(vec![
            "--synthetic".to_owned(),
            "--continuous-batching".to_owned(),
            "--max-batch".to_owned(),
            "65".to_owned(),
        ])
        .expect_err("the queue is the bound");
        assert!(too_wide.contains(&JOB_QUEUE.to_string()), "{too_wide}");

        let device = ServeOptions::parse(vec![
            "--synthetic".to_owned(),
            "--continuous-batching".to_owned(),
            "--engine".to_owned(),
            "device".to_owned(),
        ])
        .expect_err("the device engine path is unchanged");
        assert!(device.contains("reference engine"), "{device}");
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

    #[test]
    fn one_pass_prompt_logprobs_match_the_token_at_a_time_reference() {
        let config = ModelConfig::tiny();
        let plan = TensorPlan::language_model(&config);
        let engine = InferenceEngine::new(
            config.clone(),
            Box::new(SyntheticWeights::from_plan(&plan, SYNTHETIC_WEIGHT_SEED)),
            REFERENCE_TENSOR_LIMIT,
        )
        .expect("the synthetic tiny engine builds");
        let decode = spelling_decoder;
        let prompts = [vec![1u32], vec![1, 2, 3], vec![1, 2, 3, 4, 5, 6]];
        let selections = [
            LogprobSelection::Top(2),
            LogprobSelection::Top(0),
            LogprobSelection::All,
        ];
        for prompt in prompts {
            for selection in selections {
                let request = LogprobsRequest {
                    sample: None,
                    prompt: Some(selection),
                    token_ids: Vec::new(),
                };
                let mut one_pass_decoder = LogprobDecoder::new(&decode);
                let mut one_pass_state = SequenceState::new(engine.config());
                let (one_pass_step, one_pass_payload) = prompt_logprobs_pass(
                    &engine,
                    &prompt,
                    &mut one_pass_state,
                    &request,
                    &mut one_pass_decoder,
                )
                .expect("the one-pass prompt logprobs run");
                let mut reference_decoder = LogprobDecoder::new(&decode);
                let mut reference_state = SequenceState::new(engine.config());
                let (reference_step, reference_payload) = prompt_logprobs_pass_token_at_a_time(
                    &engine,
                    &prompt,
                    &mut reference_state,
                    &request,
                    &mut reference_decoder,
                )
                .expect("the token-at-a-time reference runs");
                assert_eq!(one_pass_payload.tokens, reference_payload.tokens);
                assert_eq!(
                    one_pass_payload.positions, reference_payload.positions,
                    "prompt {prompt:?} at {selection:?}: the payload differs from the reference"
                );
                assert_eq!(
                    one_pass_step
                        .logits
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    reference_step
                        .logits
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    "the final logits differ from the token-at-a-time reference"
                );
                assert_eq!(
                    one_pass_step
                        .head_input
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    reference_step
                        .head_input
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    "the final head input differs from the token-at-a-time reference"
                );
                assert_eq!(one_pass_step.next_token, reference_step.next_token);
                assert_eq!(one_pass_state.tokens(), reference_state.tokens());
                assert_eq!(one_pass_payload.positions.len(), prompt.len());
                assert!(one_pass_payload.positions[0].is_none());
            }
        }
    }

    #[test]
    fn prompt_logprobs_spans_cover_the_prompt_within_the_row_budget() {
        for prompt_len in [1usize, 2, 7, 4096, 4097] {
            let vocab = PROMPT_LOGPROBS_ROW_BUDGET / 4;
            let spans = prompt_logprobs_spans(prompt_len, vocab).expect("spans build");
            assert_eq!(spans[0].start, 0);
            assert_eq!(spans[spans.len() - 1].end, prompt_len);
            for pair in spans.windows(2) {
                assert_eq!(pair[0].end, pair[1].start, "spans are contiguous");
            }
            for span in &spans {
                assert!(!span.is_empty());
                assert!(
                    span.len() <= 4,
                    "a span returns at most the row budget: {span:?}"
                );
            }
        }
        let tiny = prompt_logprobs_spans(4096, 128).expect("spans build");
        assert_eq!(
            tiny.len(),
            1,
            "the tiny vocabulary fits a whole scheduler chunk in one span"
        );
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

    // ---- stop strings (vLLM semantics) --------------------------------------------------

    #[test]
    fn stop_buffer_holds_back_partial_stop_strings() {
        let mut buffer = StopBuffer::new(&["zz".to_owned()]);
        // One character is held back because appending it could complete "zz".
        assert_eq!(buffer.push(Some("Hello")), "Hell");
        assert_eq!(buffer.push(Some(" world")), "o worl");
        assert!(!buffer.matched());
        assert_eq!(buffer.finish(), "d");
    }

    #[test]
    fn stop_buffer_truncates_at_a_match_across_token_boundaries() {
        let mut buffer = StopBuffer::new(&["stop".to_owned()]);
        assert_eq!(buffer.push(Some("a st")), "a");
        // "stop" is completed by the next token; the match and everything before it in the held
        // tail beyond the returned prefix leave the output.
        assert_eq!(buffer.push(Some("opb")), " ");
        assert!(buffer.matched());
        assert_eq!(buffer.finish(), "");
        assert_eq!(buffer.push(Some("more")), "");
    }

    #[test]
    fn stop_buffer_uses_the_earliest_match_across_sequences() {
        let mut buffer = StopBuffer::new(&["aaa".to_owned(), "bc".to_owned()]);
        assert_eq!(buffer.push(Some("xxbc aaa")), "xx");
        assert!(buffer.matched());
    }

    #[test]
    fn stop_buffer_without_sequences_emits_every_character_immediately() {
        let mut buffer = StopBuffer::new(&[]);
        assert_eq!(buffer.push(Some("abc")), "abc");
        assert_eq!(buffer.push(None), "");
        assert_eq!(buffer.finish(), "");
        assert!(!buffer.matched());
    }

    /// The generated text of one stream: every content delta concatenated.
    fn generated_text(events: &[EngineEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                EngineEvent::ContentDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn a_stop_string_ends_generation_and_never_reaches_the_client() {
        let driver = synthetic_driver(16);
        let full = events(&driver, request(&[1, 2, 3], 8, 0.0, 0)).await;
        let full_text = generated_text(&full);
        assert!(!full_text.is_empty(), "the synthetic plan generates text");
        let stop: String = full_text.chars().take(3).collect();
        let mut stopped_request = request(&[1, 2, 3], 8, 0.0, 0);
        stopped_request.stop = vec![stop.clone()];
        let stopped = events(&driver, stopped_request).await;
        let stopped_text = generated_text(&stopped);
        assert!(
            !stopped_text.contains(&stop),
            "the stop string must not be in the output: {stopped_text:?}"
        );
        assert_eq!(
            stopped_text, "",
            "the first token completes the stop string, so nothing before it is emitted"
        );
        let Some(EngineEvent::Finished {
            finish_reason,
            usage,
            ..
        }) = stopped.last()
        else {
            panic!("the stream did not finish: {stopped:?}");
        };
        assert_eq!(finish_reason, &FinishReason::Stop);
        assert_eq!(
            usage.completion_tokens, 1,
            "the token that completed the stop string was generated and is counted"
        );
    }

    #[tokio::test]
    async fn a_stop_string_that_never_matches_does_not_change_the_output() {
        let driver = synthetic_driver(16);
        let full = events(&driver, request(&[1, 2, 3], 6, 0.0, 0)).await;
        let full_text = generated_text(&full);
        let mut held_request = request(&[1, 2, 3], 6, 0.0, 0);
        // Longer than the output and not present in it: the held-back tail is flushed when the
        // token budget stops generation, so the client still receives the whole text.
        held_request.stop = vec![format!("{full_text}ZZZZ")];
        let held = events(&driver, held_request).await;
        assert_eq!(generated_text(&held), full_text);
        let Some(EngineEvent::Finished { finish_reason, .. }) = held.last() else {
            panic!("the stream did not finish: {held:?}");
        };
        assert_eq!(finish_reason, &FinishReason::Length);
    }

    #[tokio::test]
    async fn stop_strings_do_not_break_the_logprob_events() {
        let driver = synthetic_driver(16);
        let full = events(&driver, request(&[1, 2, 3], 8, 0.0, 0)).await;
        let full_text = generated_text(&full);
        let stop: String = full_text.chars().take(3).collect();
        let mut stopped_request = request(&[1, 2, 3], 8, 0.0, 0);
        stopped_request.logprobs.sample = Some(LogprobSelection::Top(0));
        stopped_request.stop = vec![stop.clone()];
        let stopped = events(&driver, stopped_request).await;
        let scored = stopped
            .iter()
            .filter(|event| matches!(event, EngineEvent::TokenLogprobs(_)))
            .count();
        assert_eq!(scored, 1, "one generated token still carries its logprob");
        assert_eq!(generated_text(&stopped), "");
        let Some(EngineEvent::Finished { finish_reason, .. }) = stopped.last() else {
            panic!("the stream did not finish: {stopped:?}");
        };
        assert_eq!(finish_reason, &FinishReason::Stop);
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

    use sparkflash_engine::forward::SequenceState;
    use sparkflash_engram::layout::{ENGRAM_HEAD_DIM, ENGRAM_ROW_BYTES};
    use sparkflash_engram::packer::align_up;
    use sparkflash_engram::store::{TableSpec, TailFill};
    use sparkflash_model_deepseek_v41::encoding::reference_tokenizer_path;

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
            batch: None,
        };
        let error = match build_engine(&recipe, REFERENCE_TENSOR_LIMIT) {
            Ok(_) => panic!("a directory without a manifest cannot serve Engram rows"),
            Err(error) => error,
        };
        assert!(error.contains("--engram-dir"), "{error}");
        assert!(error.contains(MANIFEST_NAME), "{error}");
    }
}

#[cfg(all(test, feature = "tp4-serve"))]
mod tp4_serve_tests {
    use std::collections::BTreeSet;
    use std::net::{TcpListener, TcpStream};
    use std::thread;

    use futures::StreamExt;

    use super::*;

    fn engine_request(prompt: &[u32], max_tokens: usize) -> EngineRequest {
        EngineRequest {
            request_id: "tp4-test".to_owned(),
            model: MODEL_ID.to_owned(),
            prompt_tokens: prompt.to_vec(),
            logprobs: LogprobsRequest::default(),
            images: Vec::new(),
            vision: Vec::new(),
            max_tokens,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            skip_special_tokens: true,
            thinking: false,
            reasoning_effort: None,
            stop: Vec::new(),
            seed: Some(0),
            trace_scopes: 0,
            trace_steps: 0,
        }
    }

    fn tp4_args(extra: &[&str]) -> Vec<String> {
        let mut args = vec![
            "--tp4-rank",
            "0",
            "--tp4-world-size",
            "2",
            "--tp4-peers",
            "127.0.0.1,127.0.0.2",
            "--checkpoint",
            "/tmp/tp4-checkpoint",
            "--tokenizer",
            "/tmp/tp4-tokenizer.json",
        ];
        args.extend_from_slice(extra);
        args.into_iter().map(str::to_owned).collect()
    }

    fn keys(value: &serde_json::Value, prefix: &str, out: &mut BTreeSet<String>) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, child) in map {
                    let path = format!("{prefix}{key}");
                    out.insert(path.clone());
                    keys(child, &format!("{path}."), out);
                }
            }
            serde_json::Value::Array(items) => {
                if let Some(first) = items.first() {
                    keys(first, &format!("{prefix}[]."), out);
                }
            }
            _ => {}
        }
    }

    fn shape(value: &serde_json::Value) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        keys(value, "", &mut out);
        out
    }

    #[test]
    fn tp4_flags_parse_with_defaults() {
        let options = ServeOptions::parse(tp4_args(&[])).expect("the TP=4 flags parse");
        let tp4 = options.tp4.expect("the parse selected TP=4");
        assert_eq!(tp4.rank, 0);
        assert_eq!(tp4.world_size, 2);
        assert_eq!(tp4.base_port, TP4_DEFAULT_BASE_PORT);
        assert_eq!(
            tp4.control_port,
            TP4_DEFAULT_BASE_PORT + TP4_CONTROL_PORT_OFFSET
        );
        assert_eq!(
            tp4.peers,
            vec![
                "127.0.0.1".parse::<IpAddr>().unwrap(),
                "127.0.0.2".parse::<IpAddr>().unwrap()
            ]
        );
    }

    #[test]
    fn tp4_flags_require_checkpoint_and_refuse_synthetic_and_device() {
        let no_checkpoint = ServeOptions::parse(
            tp4_args(&[])
                .into_iter()
                .filter(|arg| arg != "--checkpoint" && arg != "/tmp/tp4-checkpoint")
                .collect::<Vec<_>>(),
        )
        .expect_err("TP=4 without a checkpoint must be refused");
        assert!(no_checkpoint.contains("--checkpoint"), "{no_checkpoint}");

        let synthetic = ServeOptions::parse(tp4_args(&["--synthetic"]))
            .expect_err("TP=4 with the synthetic plan must be refused");
        assert!(synthetic.contains("--synthetic"), "{synthetic}");

        let device = ServeOptions::parse(tp4_args(&["--engine", "device"]))
            .expect_err("TP=4 with the device engine must be refused");
        assert!(device.contains("--engine device"), "{device}");
    }

    #[test]
    fn tp4_flags_name_a_missing_rank_or_wrong_peer_count() {
        let mut no_rank = tp4_args(&[]);
        let position = no_rank.iter().position(|arg| arg == "--tp4-rank").unwrap();
        no_rank.drain(position..position + 2);
        let error = ServeOptions::parse(no_rank).expect_err("the other TP=4 flags need a rank");
        assert!(error.contains("--tp4-rank"), "{error}");

        let mut wrong_world = tp4_args(&[]);
        let world_position = wrong_world
            .iter()
            .position(|arg| arg == "--tp4-world-size")
            .unwrap();
        wrong_world[world_position + 1] = "4".to_owned();
        let error = ServeOptions::parse(wrong_world)
            .expect_err("a two-address peer list cannot describe four ranks");
        assert!(error.contains("--tp4-peers"), "{error}");
    }

    #[test]
    fn control_frames_round_trip_on_a_loopback_socket() {
        let frames = vec![
            Tp4ControlFrame::Hello { rank: 1 },
            Tp4ControlFrame::Ready,
            Tp4ControlFrame::Request {
                prompt_ids: vec![1, 2, 3],
                max_tokens: 4,
                prompt_logprobs: None,
            },
            Tp4ControlFrame::Request {
                prompt_ids: vec![9, 8],
                max_tokens: 2,
                prompt_logprobs: Some(3),
            },
            Tp4ControlFrame::Token { token_id: 7 },
            Tp4ControlFrame::End,
            Tp4ControlFrame::Error {
                code: "engine_gone".to_owned(),
                message: "rank stopped".to_owned(),
            },
        ];
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback listener");
        let address = listener.local_addr().expect("the listener address");
        let echoed = frames.clone();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("the test client connects");
            for expected in echoed {
                let got = read_control_frame(&mut stream).expect("read one frame");
                assert_eq!(got, expected);
                write_control_frame(&mut stream, &got).expect("echo one frame");
            }
        });
        let mut client = TcpStream::connect(address).expect("connect to the test server");
        for frame in &frames {
            write_control_frame(&mut client, frame).expect("write one frame");
            let got = read_control_frame(&mut client).expect("read the echo");
            assert_eq!(&got, frame);
        }
        server.join().expect("the test server joins");
    }

    /// A rank-0 API stub for the control-protocol test: no forward, canned contract values.
    struct FakeTp4Rank {
        prompt: Vec<u32>,
        seen_selection: Option<Option<usize>>,
    }

    impl Tp4RankApi for FakeTp4Rank {
        fn begin_request(&mut self, _sampling: Tp4Sampling) -> Result<(), String> {
            Ok(())
        }

        fn prefill(&mut self, _prompt: &[u32]) -> Result<Tp4Step, String> {
            Ok(Tp4Step {
                logits: vec![0.0; 8],
                next_token: Some(500),
            })
        }

        fn prefill_prompt_logprobs(
            &mut self,
            prompt: &[u32],
            selection: Option<usize>,
        ) -> Result<(Tp4Step, Vec<MergedPosition>), String> {
            self.seen_selection = Some(selection);
            assert_eq!(prompt, self.prompt.as_slice());
            let alternatives = selection.unwrap_or(0);
            let merged = (0..prompt.len() - 1)
                .map(|position| MergedPosition {
                    position,
                    own_token_id: prompt[position + 1],
                    own_logprob: -1.25 - position as f32,
                    own_rank: 1,
                    alternatives: (0..alternatives)
                        .map(|slot| sparkflash_engine::prompt_ranks::MergedAlternative {
                            token_id: 600 + slot as u32,
                            logprob: -2.5 - slot as f32,
                        })
                        .collect(),
                })
                .collect();
            Ok((
                Tp4Step {
                    logits: vec![0.0; 8],
                    next_token: Some(500),
                },
                merged,
            ))
        }

        fn decode(&mut self, _token: u32) -> Result<Tp4Step, String> {
            Ok(Tp4Step {
                logits: vec![0.0; 8],
                next_token: Some(501),
            })
        }

        fn metrics(&self) -> Tp4RankMetrics {
            Tp4RankMetrics {
                load_ms: 0,
                resident_cache_bytes: 0,
                resident_cache_tensors: 0,
                reduce_calls: 0,
                reduce_elements: 0,
                steps: 0,
                last_step_ms: 0,
            }
        }
    }

    #[test]
    fn the_prompt_logprob_request_frame_carries_the_selection_and_the_events_lead_with_the_payload()
    {
        let prompt = vec![100u32, 200, 300, 400];
        let tokenizer =
            load_hf_tokenizer(&sparkflash_model_deepseek_v41::encoding::reference_tokenizer_path())
                .expect("tokenizer");
        // Bind a free loopback port, release it, and let the hub take it: the hub does not expose
        // its bound address, and the race window is a test-only detail.
        let probe = TcpListener::bind("127.0.0.1:0").expect("probe listener");
        let address = probe.local_addr().expect("probe address");
        drop(probe);
        let control_address = address;
        // The worker connects before the hub waits for it: connect retries until the hub binds,
        // and the hub cannot return until every worker has said Hello.
        let worker = thread::spawn(move || {
            let mut control =
                Tp4ControlWorker::connect(control_address, 1, 2, Duration::from_secs(10))
                    .expect("worker connect");
            control.handshake().expect("handshake");
            let mut requests = Vec::new();
            while let Ok(frame) = control.next_frame() {
                match frame {
                    Tp4ControlFrame::Request {
                        prompt_ids,
                        max_tokens,
                        prompt_logprobs,
                    } => {
                        requests.push((prompt_ids, max_tokens, prompt_logprobs));
                        loop {
                            match control.next_frame().expect("decode frame") {
                                Tp4ControlFrame::Token { token_id } => {
                                    assert_eq!(token_id, 500);
                                }
                                Tp4ControlFrame::End => break,
                                other => panic!("expected Token or End, got {other:?}"),
                            }
                        }
                    }
                    Tp4ControlFrame::End => break,
                    other => panic!("expected Request or End, got {other:?}"),
                }
            }
            requests
        });
        let mut hub = Tp4ControlHub::start(address, 2, Duration::from_secs(10)).expect("hub");
        let (replies, mut answers) = tokio::sync::mpsc::channel(REPLY_BUFFER);
        let job = Job {
            prompt_tokens: prompt.clone(),
            max_tokens: 2,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            skip_special_tokens: true,
            seed: 0,
            logprobs: LogprobsRequest {
                sample: None,
                prompt: Some(LogprobSelection::Top(3)),
                token_ids: Vec::new(),
            },
            stop: Vec::new(),
            enqueued_at: Instant::now(),
            replies,
        };
        let metrics = Metrics::new();
        let mut rank = FakeTp4Rank {
            prompt: prompt.clone(),
            seen_selection: None,
        };
        run_tp4_request(&mut rank, &mut hub, &tokenizer, job, &metrics).expect("request runs");
        assert_eq!(rank.seen_selection, Some(Some(3)));
        // Closing the hub is what ends the worker loop in the test; the real worker exits with
        // the process.
        drop(hub);
        let requests = worker.join().expect("worker joins");
        assert_eq!(
            requests,
            vec![(prompt.clone(), 2, Some(3u32))],
            "the Request frame must carry the prompt-logprob selection"
        );
        // The first reply is the whole-prompt payload, delivered after prefill and before the
        // first generated token, exactly where the serial host path delivers it.
        let first = answers.try_recv().expect("the prompt payload");
        let Reply::PromptLogprobs(payload) = first else {
            panic!("expected PromptLogprobs first, got a different reply");
        };
        assert_eq!(payload.tokens.len(), prompt.len());
        assert!(payload.positions[0].is_none());
        for (index, position) in payload.positions.iter().enumerate().skip(1) {
            let position = position.as_ref().expect("a scored position");
            assert_eq!(position.token_id, prompt[index]);
            assert_eq!(position.top.len(), 4, "own token plus three alternatives");
            assert_eq!(position.top[0].token_id, prompt[index]);
            assert_eq!(position.top[0].logprob, position.logprob);
            assert_eq!(position.top[0].rank, Some(1));
            for (slot, alternative) in position.top.iter().skip(1).enumerate() {
                assert_eq!(alternative.rank, Some(u32::try_from(slot + 1).unwrap()));
            }
        }
        // Then the generated-token deltas and the terminal reply.
        let mut deltas = 0usize;
        let mut done = false;
        while let Ok(reply) = answers.try_recv() {
            match reply {
                Reply::Delta(_) => deltas += 1,
                Reply::Done(_) => done = true,
                _ => panic!("unexpected reply after the payload"),
            }
        }
        assert!(deltas >= 1);
        assert!(done, "the request must end with the Done summary");
    }

    #[test]
    fn the_tp4_prompt_logprobs_payload_matches_the_serial_assembly() {
        // A fabricated finite log-softmax row and a prompt that walks it: the rank contract's
        // merged values are derived from the same row the serial formula uses, so the payload the
        // TP=4 path builds must equal the serial token_logprob assembly exactly, strings and order
        // included.
        let prompt: Vec<u32> = vec![2, 5, 8, 13, 21];
        let row: Vec<f32> = (0..64).map(|index| index as f32 * 0.7 - 6.0).collect();
        let values = log_softmax(&row);
        let decode_ids = |ids: &[u32]| -> Result<String, String> {
            Ok(ids
                .iter()
                .map(|token| format!("<{token}>"))
                .collect::<String>())
        };
        for requested in [None, Some(0u32), Some(3u32)] {
            let selection = requested.map(LogprobSelection::Top);
            let request = LogprobsRequest {
                sample: None,
                prompt: selection,
                token_ids: Vec::new(),
            };
            let alternatives: usize = requested.unwrap_or(0) as usize;
            let merged: Vec<MergedPosition> = (0..prompt.len() - 1)
                .map(|index| {
                    let own_token_id = prompt[index + 1];
                    let own_logprob = values[own_token_id as usize];
                    MergedPosition {
                        position: index,
                        own_token_id,
                        own_logprob,
                        own_rank: competition_rank(&values, own_logprob),
                        alternatives: top_k_indices(&values, alternatives)
                            .into_iter()
                            .map(|token| sparkflash_engine::prompt_ranks::MergedAlternative {
                                token_id: token as u32,
                                logprob: values[token],
                            })
                            .collect(),
                    }
                })
                .collect();
            let mut decoder = LogprobDecoder::new(&decode_ids);
            let payload = tp4_prompt_logprobs_payload(&prompt, &merged, &request, &mut decoder)
                .expect("the TP=4 payload builds");
            // The serial assembly of the same request, with the same decoding sequence.
            let mut serial_decoder = LogprobDecoder::new(&decode_ids);
            let mut tokens = Vec::new();
            let mut positions: Vec<Option<TokenLogprob>> = vec![None; prompt.len()];
            tokens.push(TokenString {
                token_id: prompt[0],
                token: serial_decoder.decode(prompt[0]).expect("decode"),
            });
            serial_decoder.push(prompt[0]);
            for index in 0..prompt.len() - 1 {
                let token_index = index + 1;
                let token = serial_decoder.decode(prompt[token_index]).expect("decode");
                tokens.push(TokenString {
                    token_id: prompt[token_index],
                    token: token.clone(),
                });
                positions[token_index] = Some(
                    token_logprob(
                        &values,
                        prompt[token_index],
                        token,
                        selection,
                        &[],
                        &mut serial_decoder,
                    )
                    .expect("serial token logprob"),
                );
                serial_decoder.push(prompt[token_index]);
            }
            assert_eq!(
                payload,
                PromptLogprobs { tokens, positions },
                "requested {requested:?} changed the payload"
            );
        }
    }

    #[test]
    fn missing_rank_is_a_typed_startup_timeout() {
        let error = Tp4ControlHub::start(
            SocketAddr::from(([127, 0, 0, 1], 0)),
            2,
            Duration::from_millis(80),
        )
        .expect_err("a missing rank cannot complete the group");
        match error {
            Tp4ControlError::MissingRank {
                world,
                joined,
                waited,
            } => {
                assert_eq!(world, 2);
                assert_eq!(joined, 0);
                assert_eq!(waited, Duration::from_millis(80));
            }
            other => panic!("expected MissingRank, got {other}"),
        }
    }

    #[test]
    fn worker_control_connect_retries_until_the_hub_binds() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback listener");
        let address = listener.local_addr().expect("the listener address");
        // Release the port so the worker's early connect attempts are refused, exactly the
        // rank-0-loads-slower startup race; the hub binds after a short delay.
        drop(listener);
        let server = thread::spawn(move || {
            thread::sleep(Duration::from_millis(250));
            let listener =
                TcpListener::bind(address).expect("the hub binds after the worker starts trying");
            let (mut stream, _) = listener.accept().expect("the retried worker connects");
            let frame = read_control_frame(&mut stream).expect("read the worker Hello");
            assert!(
                matches!(frame, Tp4ControlFrame::Hello { rank: 1 }),
                "{frame:?}"
            );
            write_control_frame(&mut stream, &Tp4ControlFrame::Ready).expect("send Ready");
        });
        let mut worker = Tp4ControlWorker::connect(address, 1, 2, Duration::from_secs(5))
            .expect("the connect retries until the hub is listening");
        worker
            .handshake()
            .expect("the retried connection handshakes");
        assert!(worker.ready);
        server.join().expect("the test hub joins");
    }

    #[test]
    fn worker_control_connect_names_the_peer_when_the_hub_never_binds() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback listener");
        let address = listener.local_addr().expect("the listener address");
        drop(listener);
        let waited = Duration::from_millis(150);
        let error = match Tp4ControlWorker::connect(address, 1, 2, waited) {
            Ok(_) => panic!("a hub that never binds must not report a connected worker"),
            Err(error) => error,
        };
        let display = error.to_string();
        match error {
            Tp4ControlError::ConnectTimeout {
                address: tried,
                waited: reported,
            } => {
                assert_eq!(tried, address);
                assert_eq!(reported, waited);
            }
            other => panic!("expected ConnectTimeout, got {other}"),
        }
        assert!(
            display.contains(&address.to_string()),
            "the refusal must name the peer: {display}"
        );
        assert!(
            display.contains("150ms"),
            "the refusal must name the elapsed budget: {display}"
        );
    }

    #[test]
    fn request_before_ready_is_refused_by_the_worker() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback listener");
        let address = listener.local_addr().expect("the listener address");
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("the test hub connects");
            let mut worker = Tp4ControlWorker {
                stream,
                rank: 1,
                world: 2,
                ready: false,
            };
            assert!(matches!(
                worker.next_frame(),
                Err(Tp4ControlError::NotReady { .. })
            ));
        });
        let _client = TcpStream::connect(address).expect("connect to the test worker");
        server.join().expect("the test worker joins");
    }

    #[tokio::test]
    async fn one_request_guard_refuses_a_second_concurrent_request() {
        let (jobs, mut queued) = tokio::sync::mpsc::channel(TP4_JOB_QUEUE);
        let driver = Tp4Driver::for_test(jobs, 4);
        assert!(
            !driver.independent_choices(),
            "the router must refuse n > 1 on the lockstep TP=4 group"
        );
        let _first = driver.generate(engine_request(&[1, 2, 3], 2));
        let second = driver.generate(engine_request(&[1, 2, 3], 2));
        let events: Vec<EngineEvent> = second.collect().await;
        assert!(
            matches!(
                events.first(),
                Some(EngineEvent::Error { code, .. }) if code == REFUSED_CODE
            ),
            "the second concurrent request must be the existing typed refusal: {events:?}"
        );
        let _ = queued.try_recv();
    }

    #[test]
    fn tp4_stats_keep_the_reference_shape_and_name_the_fabric() {
        let timing = RequestTiming::default();
        let reference = serde_json::to_value(request_stats_with_fabric(3, timing, ""))
            .expect("the reference stats serialize");
        let tp4 = serde_json::to_value(request_stats_with_fabric(3, timing, TP4_FABRIC))
            .expect("the TP=4 stats serialize");
        assert_eq!(
            shape(&reference),
            shape(&tp4),
            "the stats keys must not drift"
        );
        assert_eq!(tp4["transport"]["fabric_mode"], TP4_FABRIC);
        assert_eq!(reference["transport"]["fabric_mode"], "");
        assert_eq!(tp4["kv_cache"]["total_prompt_tokens"], 3);
        assert_eq!(tp4["kv_cache"]["global_kv_cached_tokens"], 0);
    }
}

#[cfg(test)]
mod device_token_serve_tests {
    use super::*;
    use sparkflash_model_deepseek_v41::encoding::reference_tokenizer_path;

    #[test]
    fn device_token_flags_parse_and_require_each_other() {
        let error = ServeOptions::parse(vec![
            "--synthetic".to_owned(),
            "--device-token-derived-claim".to_owned(),
        ])
        .expect_err("the derived claim needs a window");
        assert!(
            error.contains("--device-token-dir"),
            "the message names the missing flag: {error}"
        );
        let options = ServeOptions::parse(vec![
            "--synthetic".to_owned(),
            "--device-token-dir".to_owned(),
            "/tmp/sparkflash-window".to_owned(),
            "--device-token-derived-claim".to_owned(),
        ])
        .expect("the flags parse");
        assert_eq!(
            options.device_token_dir,
            Some(std::path::PathBuf::from("/tmp/sparkflash-window"))
        );
        assert!(options.device_token_derived_claim);
        let plain = ServeOptions::parse(vec!["--synthetic".to_owned()]).expect("parses");
        assert_eq!(plain.device_token_dir, None);
        assert!(!plain.device_token_derived_claim);
        assert!(!plain.device_token_live);
    }

    #[test]
    fn the_live_follower_needs_a_window_directory() {
        let error = ServeOptions::parse(vec![
            "--synthetic".to_owned(),
            "--device-token-live".to_owned(),
        ])
        .expect_err("--device-token-live alone has nothing to follow");
        assert!(
            error.contains("--device-token-dir"),
            "the message names the missing flag: {error}"
        );
        let options = ServeOptions::parse(vec![
            "--synthetic".to_owned(),
            "--device-token-dir".to_owned(),
            "/tmp/sparkflash-windows".to_owned(),
            "--device-token-live".to_owned(),
        ])
        .expect("the live follower parses with a directory");
        assert!(options.device_token_live);
        assert_eq!(
            options.device_token_dir,
            Some(std::path::PathBuf::from("/tmp/sparkflash-windows"))
        );
    }

    #[test]
    fn the_piece_decoder_reconstructs_the_tokenizer_decode() {
        // The device source streams pieces, but the served text must be exactly the tokenizer's
        // decode of the claimed ids: that is the piece contract, and it is host-only.
        let seam = Arc::new(
            TokenizerSeam::from_path(&reference_tokenizer_path()).expect("reference tokenizer"),
        );
        let decoder = SeamPieceDecoder {
            seam: Arc::clone(&seam),
        };
        let ids = sparkflash_api::device_chain::fixtures::DECODE_IDS;
        let mut decoded = String::new();
        let mut ids_so_far: Vec<u32> = Vec::new();
        for id in ids {
            decoded.push_str(&decoder.piece(&ids_so_far, id).expect("every piece decodes"));
            ids_so_far.push(id);
        }
        assert_eq!(Some(decoded), seam.decode(&ids));
    }
}

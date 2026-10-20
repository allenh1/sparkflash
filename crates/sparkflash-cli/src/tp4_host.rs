//! The task-53 host TWIN: four ranks, one checkpoint, one token sequence.
//!
//! The single-node reference engine answers one completion request by re-reading the weights for
//! every token (measured at 2m54s for a 7-token prompt plus 4 completion tokens on node0, about
//! 16s/token, cold). This stage is the opposite end of that curve: the per-rank slices are read from
//! the checkpoint where it lies, the node's page cache holds them after the first sweep, and the only
//! bytes a warm token pays for are the slices it decodes.
//!
//! Phase 1 (the tonight output path) shards exactly one class: the routed experts keep the section-3
//! TP rule (w1/w3 rows / world, w2 input columns / world), each rank runs the local routed block and
//! one canonical bf16 all-reduce per layer puts the four partials back together. Every other class -
//! attention, embed/head, shared expert, router, mHC, norms, indexer, compressor, Engram - is
//! replicated, which is numerically a no-op because the reference arithmetic never shards those
//! tensors. That is not vLLM's TP=4 layout; it is an intermediate that computes the same arithmetic
//! and produces the same tokens, and it is named that way in the evidence it writes.
//!
//! Two runs are what this module exists for:
//!
//! - the host twin (a 40-layer tiny fixture, one rank per thread, loopback TCP), which must produce
//!   the same token sequence and near-identical logits as the single-rank reference engine, and
//! - the node run (the released checkpoint on four GB10 nodes), which must say, with evidence, how
//!   long a token took and how many bytes it read from disk versus from page cache.
//!
//! The stage opens no driver and touches no GPU: `device_execution: false` is written
//! unconditionally. It is also not a serving path: Tp4Rank exposes the per-rank step API (open, begin_request,
//! prefill, decode, metrics) and the serve layer (task-55) owns the HTTP, tokenizer and control
//! protocol.
//!
//! `--bench-steps N` is the shared timing harness for the token-per-second iteration: it runs
//! begin_request, one prefill of the prompt, then N decode() calls with no HTTP server and no token
//! streaming, and records per-step wall milliseconds, the mean/median step and decode-only
//! tokens/second in the evidence JSON. It changes no arithmetic; with --profile the existing
//! per-step and per-layer timers are the ones it records.

use std::collections::BTreeMap;
use std::fs;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::allocator;

use sparkflash_comm::{
    CollectiveLayout, FabricSchedule, TcpConfig, TcpTransport, Transport, bf16_to_f32,
    f32_to_bf16_rn,
};
use sparkflash_device::shard::{ShardError, ShardPlan, ShardRule};
use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::dtype::DType;
use sparkflash_engine::forward::engram;
use sparkflash_engine::forward::layer_zero::{LayerZeroEntry, LayerZeroEntrySource};
use sparkflash_engine::forward::mhc::{self, Mixes};
use sparkflash_engine::forward::moe::{self, MoeDeviceBackend, MoeScratch};
use sparkflash_engine::forward::{
    DeferredHandoff, HybridEntry, HybridPost, InferenceEngine, SequenceState,
};
use sparkflash_engine::linalg::{self, EngineError};
use sparkflash_engine::quant;
use sparkflash_engine::safetensors::{SafeTensorsSource, TensorHeader};
use sparkflash_engine::tensor::{TensorPlan, TensorSpec};
use sparkflash_engine::weights::{WeightError, WeightSource};

/// The evidence schema version this stage writes.
pub const SCHEMA_VERSION: u32 = 1;

/// The element limit a single tensor may materialise at.
///
/// The released head is 662M elements, so the engine reference limit (32M) would refuse it; the
/// Engram tables are paged by range and never materialise, and everything else is far below this.
const ELEMENT_LIMIT: usize = 1 << 31;

/// The 40-layer tiny twin's expert intermediate width.
///
/// It is 128 rather than tiny's 32 so that the expert scale grid (`ceil(inter/32)` columns, one per
/// 32 input values) still has at least one column per rank at world 4: 4 columns / 4 = 1. The
/// released checkpoint has 72 columns and divides the same way.
const TWIN_MOE_INTER_DIM: usize = 128;

/// The usage line for the tp4-host stage.
pub const USAGE: &str = "sparkflash device-chain --run-stage tp4-host --tp-rank N --tp-world-size N --comm-peers IP,IP,... --comm-base-port 29601 (--fixture DIR | --checkpoint DIR) [--prompt TEXT | --prompt-tokens 3,7,11] [--max-tokens 4] [--tokenizer PATH] [--evidence-dir DIR] [--resident-cache] [--profile] [--bench-steps N] [--record-seams] [--reference-check]; tp4-host --write-fixture DIR writes the 40-layer tiny twin, and one rank binds comm-base-port + tp-rank";

/// One rank's collected evidence.
#[derive(Debug, Clone)]
struct RankOutcome {
    rank: usize,
    world: usize,
    prompt: Vec<u32>,
    generated: Vec<u32>,
    step_tokens: Vec<Option<u32>>,
    completion_text: Option<String>,
    step_ms: Vec<u64>,
    step_rchar: Vec<u64>,
    step_read_bytes: Vec<u64>,
    step_collective_ms: Vec<u64>,
    reduce_ms_total: u64,
    total_ms: u64,
    load_ms: u64,
    io_rchar: u64,
    io_read_bytes: u64,
    reduce_calls: u64,
    reduce_elements: u64,
    shard_read_calls: u64,
    shard_read_bytes: u64,
    shard_tensors: usize,
    resident_bytes_estimate: u64,
    cache_tensors: usize,
    cache_bytes: u64,
    cache_plan_bytes: u64,
    cache_replicated_bytes: u64,
    cache_fill_ms: u64,
    cache_hits: u64,
    rss_load_kib: u64,
    rss_peak_kib: u64,
    plan_hash: u64,
    collective_label: String,
    seams: Vec<SeamRecord>,
    reference: Option<ReferenceCheck>,
    source_mode: &'static str,
    profile: Option<Profiler>,
    bench: Option<BenchReport>,
}

/// The decode loop a bench run measured, recorded in the evidence so the natural metric
/// (tokens/second) is a number in a file, not a line on a terminal.
#[derive(Debug, Clone)]
struct BenchReport {
    /// Decode steps requested with --bench-steps.
    steps_requested: usize,
    /// Decode steps actually measured (a failure cannot leave this short; the loop errors).
    steps_run: usize,
    /// Prompt tokens the prefill consumed.
    prefill_tokens: usize,
    /// Wall milliseconds of the prefill.
    prefill_ms: u64,
    /// The token prefill selected at the last prompt position.
    prefill_token: Option<u32>,
    /// Per-decode-step wall milliseconds.
    step_ms: Vec<u64>,
    /// Per-decode-step bytes returned by read() (Engram rows only when resident).
    step_rchar: Vec<u64>,
    /// Per-decode-step disk-sourced bytes.
    step_read_bytes: Vec<u64>,
    /// Per-decode-step canonical reduce milliseconds.
    step_collective_ms: Vec<u64>,
    /// Decode total wall milliseconds (sum of step_ms).
    decode_ms: u64,
    /// Mean decode step, milliseconds.
    mean_step_ms: f64,
    /// Median decode step, milliseconds.
    median_step_ms: f64,
    /// Decode-only tokens per second: steps_run * 1000 / decode_ms.
    tokens_per_second: f64,
    /// Whether the --profile timers were recording during the bench.
    profile_on: bool,
}

/// One per-layer routed seam measurement against the unsharded f32 arithmetic on the same input.
#[derive(Debug, Clone)]
struct SeamRecord {
    layer: usize,
    position: usize,
    relative_l2: f64,
    max_abs: f32,
    compared: usize,
}

/// The comparison of one rank's token sequence and logits with the single-rank reference engine.
#[derive(Debug, Clone)]
struct ReferenceCheck {
    tokens_equal: bool,
    reference_tokens: Vec<Option<u32>>,
    logits_rel_l2: Vec<f64>,
    max_logits_rel_l2: f64,
    seam_max_rel_l2: f64,
    seam_comparisons: usize,
}

/// The parsed command line of one tp4-host invocation.
#[derive(Debug, Clone)]
struct HostArgs {
    rank: usize,
    world: usize,
    peers: Vec<IpAddr>,
    base_port: u16,
    fixture: Option<PathBuf>,
    checkpoint: Option<PathBuf>,
    write_fixture: Option<PathBuf>,
    prompt_text: Option<String>,
    prompt_tokens: Option<Vec<u32>>,
    max_tokens: usize,
    tokenizer: Option<PathBuf>,
    evidence_dir: Option<PathBuf>,
    record_seams: bool,
    reference_check: bool,
    resident_cache: bool,
    profile: bool,
    bench_steps: usize,
}

impl Default for HostArgs {
    fn default() -> Self {
        Self {
            rank: 0,
            world: 1,
            peers: vec![IpAddr::from([127, 0, 0, 1])],
            base_port: 29601,
            fixture: None,
            checkpoint: None,
            write_fixture: None,
            prompt_text: None,
            prompt_tokens: None,
            max_tokens: 4,
            tokenizer: None,
            evidence_dir: None,
            record_seams: false,
            reference_check: false,
            resident_cache: false,
            profile: false,
            bench_steps: 0,
        }
    }
}

impl HostArgs {
    fn parse(args: &[&str]) -> Result<Self, String> {
        let mut parsed = Self::default();
        let mut index = 0;
        while index < args.len() {
            let flag = args[index];
            let mut take = |what: &str| -> Result<String, String> {
                index += 1;
                args.get(index)
                    .map(|value| (*value).to_owned())
                    .ok_or_else(|| format!("{what} needs a value"))
            };
            match flag {
                "--tp-rank" => {
                    parsed.rank = take("--tp-rank")?
                        .parse()
                        .map_err(|error| format!("--tp-rank: {error}"))?;
                }
                "--tp-world-size" => {
                    parsed.world = take("--tp-world-size")?
                        .parse()
                        .map_err(|error| format!("--tp-world-size: {error}"))?;
                }
                "--comm-peers" => {
                    let text = take("--comm-peers")?;
                    parsed.peers = text
                        .split(',')
                        .map(|peer| {
                            peer.trim()
                                .parse::<IpAddr>()
                                .map_err(|error| format!("--comm-peers {peer}: {error}"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                }
                "--comm-base-port" => {
                    parsed.base_port = take("--comm-base-port")?
                        .parse()
                        .map_err(|error| format!("--comm-base-port: {error}"))?;
                }
                "--fixture" => parsed.fixture = Some(PathBuf::from(take("--fixture")?)),
                "--checkpoint" => parsed.checkpoint = Some(PathBuf::from(take("--checkpoint")?)),
                "--write-fixture" => {
                    parsed.write_fixture = Some(PathBuf::from(take("--write-fixture")?));
                }
                "--prompt" => parsed.prompt_text = Some(take("--prompt")?),
                "--prompt-tokens" => {
                    let text = take("--prompt-tokens")?;
                    parsed.prompt_tokens = Some(
                        text.split(',')
                            .map(|token| {
                                token
                                    .trim()
                                    .parse::<u32>()
                                    .map_err(|error| format!("--prompt-tokens {token}: {error}"))
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                }
                "--max-tokens" => {
                    parsed.max_tokens = take("--max-tokens")?
                        .parse()
                        .map_err(|error| format!("--max-tokens: {error}"))?;
                }
                "--tokenizer" => parsed.tokenizer = Some(PathBuf::from(take("--tokenizer")?)),
                "--evidence-dir" => {
                    parsed.evidence_dir = Some(PathBuf::from(take("--evidence-dir")?));
                }
                "--resident-cache" => parsed.resident_cache = true,
                "--profile" => parsed.profile = true,
                "--bench-steps" => {
                    parsed.bench_steps = take("--bench-steps")?
                        .parse()
                        .map_err(|error| format!("--bench-steps: {error}"))?;
                }
                "--record-seams" => parsed.record_seams = true,
                "--reference-check" => parsed.reference_check = true,
                other => return Err(format!("unknown argument {other}")),
            }
            index += 1;
        }
        if parsed.write_fixture.is_none() {
            if parsed.fixture.is_some() == parsed.checkpoint.is_some() {
                return Err("exactly one of --fixture and --checkpoint is required".to_owned());
            }
            if parsed.peers.len() != parsed.world {
                return Err(format!(
                    "--comm-peers names {} addresses for --tp-world-size {}",
                    parsed.peers.len(),
                    parsed.world
                ));
            }
            if parsed.rank >= parsed.world {
                return Err(format!(
                    "--tp-rank {} is outside --tp-world-size {}",
                    parsed.rank, parsed.world
                ));
            }
            if parsed.prompt_text.is_none() && parsed.prompt_tokens.is_none() {
                return Err("one of --prompt and --prompt-tokens is required".to_owned());
            }
        }
        Ok(parsed)
    }
}

/// Entry point for `--run-stage tp4-host`.
pub fn run(args: &[&str]) -> ExitCode {
    match run_checked(args) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("tp4-host: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run_checked(args: &[&str]) -> Result<ExitCode, String> {
    let parsed = HostArgs::parse(args)?;
    if let Some(dir) = &parsed.write_fixture {
        let summary = write_twin_fixture(dir)?;
        println!(
            "tp4-host: wrote {} tensors, {} bytes to {}",
            summary.tensors,
            summary.bytes,
            dir.display()
        );
        println!("tp4-host: fixture hash {:#018x}", summary.hash);
        return Ok(ExitCode::SUCCESS);
    }
    let prompt = match (&parsed.prompt_tokens, &parsed.prompt_text) {
        (Some(tokens), _) => tokens.clone(),
        (None, Some(text)) => {
            let path = parsed
                .tokenizer
                .as_deref()
                .ok_or("--prompt needs --tokenizer PATH")?;
            let encoder = sparkflash_api::encoding::load_hf_tokenizer(path)
                .map_err(|error| format!("tokenizer: {error}"))?;
            encoder
                .tokenizer()
                .encode(text.as_str(), false)
                .map_err(|error| format!("tokenize prompt: {error}"))?
                .get_ids()
                .to_vec()
        }
        (None, None) => return Err("no prompt".to_owned()),
    };
    let source_dir = parsed
        .fixture
        .clone()
        .or_else(|| parsed.checkpoint.clone())
        .ok_or("no source")?;
    let source_mode = if parsed.fixture.is_some() {
        "fixture"
    } else {
        "checkpoint"
    };
    let input = RankInput {
        rank: parsed.rank,
        world: parsed.world,
        source_dir: source_dir.clone(),
        source_mode,
        prompt,
        max_tokens: parsed.max_tokens,
        record_seams: parsed.record_seams,
        reference_check: parsed.reference_check,
        resident_cache: parsed.resident_cache || parsed.fixture.is_none(),
        profile: parsed.profile || std::env::var("SPARKFLASH_TP4_PROFILE").as_deref() == Ok("1"),
        bench_steps: parsed.bench_steps,
        tokenizer: parsed.tokenizer.clone(),
    };
    let transport = if parsed.world == 1 {
        None
    } else {
        let addresses: Vec<SocketAddr> = parsed
            .peers
            .iter()
            .enumerate()
            .map(|(peer, ip)| SocketAddr::new(*ip, parsed.base_port + peer as u16))
            .collect();
        let mut config = TcpConfig::with_addresses(parsed.rank, addresses)
            .map_err(|error| format!("tcp config: {error}"))?;
        // A node run's ranks reach the collective at different times: rank 0 starts with a warm
        // page cache and ranks 1-3 cold, and the first token's cold reads can exceed the 5 s
        // default. The fixture runs keep the default; the released-checkpoint run gets a timeout
        // that no cold first token can hit.
        if parsed.fixture.is_none() {
            config = config
                .with_timeout(Duration::from_secs(180))
                .with_connect_timeout(Duration::from_secs(60));
        }
        let transport =
            TcpTransport::connect(config).map_err(|error| format!("tcp connect: {error}"))?;
        Some(Box::new(transport) as Box<dyn Transport>)
    };
    let started = Instant::now();
    let outcome = run_rank(&input, transport)?;
    let wall_ms = started.elapsed().as_millis() as u64;
    let consumed = outcome.prompt.len() + outcome.generated.len();
    let ms_per_token = if consumed == 0 {
        0.0
    } else {
        wall_ms as f64 / consumed as f64
    };
    if outcome.bench.is_some() {
        println!(
            "tp4-host rank {}/{}: {} prompt tokens, {} generated, {} ms wall including load and prefill; the bench line below is the decode-only metric",
            outcome.rank,
            outcome.world,
            outcome.prompt.len(),
            outcome.generated.len(),
            wall_ms
        );
    } else {
        println!(
            "tp4-host rank {}/{}: {} prompt tokens, {} generated, {} ms wall ({ms_per_token:.2} ms/token, {:.2} tokens/s), {} all-reduces",
            outcome.rank,
            outcome.world,
            outcome.prompt.len(),
            outcome.generated.len(),
            wall_ms,
            tokens_per_second(wall_ms, consumed),
            outcome.reduce_calls
        );
    }
    println!("tp4-host tokens: {:?}", outcome.generated);
    if let Some(bench) = &outcome.bench {
        println!(
            "tp4-host bench: {} decode steps ({} requested) after {}-token prefill in {} ms; mean {:.1} ms/step, median {:.1} ms, {:.3} tokens/s decode-only{}",
            bench.steps_run,
            bench.steps_requested,
            bench.prefill_tokens,
            bench.prefill_ms,
            bench.mean_step_ms,
            bench.median_step_ms,
            bench.tokens_per_second,
            if bench.profile_on {
                " (profile on)"
            } else {
                ""
            }
        );
    }
    if let Some(directory) = &parsed.evidence_dir {
        write_rank_evidence(directory, &outcome, wall_ms)?;
        if outcome.rank == 0 {
            write_summary(directory, &outcome, wall_ms)?;
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn tokens_per_second(wall_ms: u64, tokens: usize) -> f64 {
    if wall_ms == 0 {
        0.0
    } else {
        tokens as f64 * 1000.0 / wall_ms as f64
    }
}

/// What a rank is asked to run.
struct RankInput {
    rank: usize,
    world: usize,
    source_dir: PathBuf,
    source_mode: &'static str,
    prompt: Vec<u32>,
    max_tokens: usize,
    record_seams: bool,
    reference_check: bool,
    resident_cache: bool,
    profile: bool,
    bench_steps: usize,
    tokenizer: Option<PathBuf>,
}

/// One rank's configuration for the TP=4 host path.
///
/// The checkpoint directory is the snapshot root that holds model.safetensors.index.json and
/// the shards; peers are in rank order and rank R binds base_port + R.
#[derive(Debug, Clone)]
pub struct Tp4RankConfig {
    /// Zero-based rank.
    pub rank: usize,
    /// Number of ranks in the collective; 1 is a strict no-op.
    pub world_size: usize,
    /// Peer IP addresses in rank order.
    pub peers: Vec<IpAddr>,
    /// First collective port.
    pub base_port: u16,
    /// Checkpoint directory (the snapshot root).
    pub checkpoint: PathBuf,
    /// Element guard for weight reads; the CLI uses 1 << 31.
    pub element_limit: usize,
    /// Hold every non-Engram tensor in process memory. Engram tables are always paged.
    pub resident_cache: bool,
    /// Per-collective read timeout.
    pub timeout: Duration,
}

impl Default for Tp4RankConfig {
    fn default() -> Self {
        Self {
            rank: 0,
            world_size: 1,
            peers: Vec::new(),
            base_port: 29601,
            checkpoint: PathBuf::new(),
            element_limit: ELEMENT_LIMIT,
            resident_cache: true,
            timeout: Duration::from_secs(180),
        }
    }
}

/// Autoregressive sampling settings for one request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tp4Sampling {
    /// 0.0 selects the engine's greedy argmax. Any other value is refused today.
    pub temperature: f32,
    /// Seed for a future sampler. Recorded, unused while greedy.
    pub seed: u64,
}

impl Default for Tp4Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            seed: 0,
        }
    }
}

/// One forward step's output.
#[derive(Debug, Clone)]
pub struct StepOutput {
    /// The token the engine selected at this position.
    pub token: Option<u32>,
    /// Full f32 logits at this position.
    pub logits: Vec<f32>,
}

/// The failure mode of the TP=4 host path: a message, nothing else.
#[derive(Debug, Clone)]
pub struct Tp4Error {
    message: String,
}

impl Tp4Error {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for Tp4Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for Tp4Error {}

impl From<String> for Tp4Error {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

/// The counters a server reports for one rank of the TP=4 host path.
#[derive(Debug, Clone, Default)]
pub struct Tp4Metrics {
    /// Checkpoint open, plan build and resident fill, before the first token.
    pub load_ms: u64,
    /// Time spent filling the resident cache.
    pub cache_fill_ms: u64,
    /// Raw bytes held by the resident cache.
    pub resident_cache_bytes: u64,
    /// Tensor count held by the resident cache.
    pub resident_cache_tensors: usize,
    /// Resident set size after the fill.
    pub rss_load_kib: u64,
    /// Peak resident set size observed.
    pub rss_peak_kib: u64,
    /// Canonical all-reduce calls.
    pub reduce_calls: u64,
    /// Elements summed by all-reduce calls.
    pub reduce_elements: u64,
    /// Milliseconds spent inside all-reduce calls.
    pub reduce_ms_total: u64,
    /// Plan reads served by the cache (should equal every plan read after the fill).
    pub shard_read_calls: u64,
    /// Bytes read for plan slices.
    pub shard_read_bytes: u64,
    /// Forward steps run in the current request.
    pub steps: u64,
    /// Wall milliseconds of the most recent step.
    pub last_step_ms: u64,
    /// The shard plan hash.
    pub plan_hash: u64,
}

/// One layer's timers for kernel-dev's decode-vs-GEMM question; only collected with --profile.
#[derive(Debug, Clone, Default)]
struct ProfileLayer {
    layer: usize,
    provider_us: u64,
    deferred_us: u64,
    post_us: u64,
    ffn_local_us: u64,
    ffn_reduce_us: u64,
    shared_us: u64,
    engine_host_ms: u64,
}

/// One forward step's profile entry.
#[derive(Debug, Clone, Default)]
struct ProfileStep {
    position: usize,
    wall_us: u64,
    user_us: u64,
    sys_us: u64,
    rchar: u64,
    read_bytes: u64,
    collective_us: u64,
    layers: Vec<ProfileLayer>,
}

/// The per-rank profile document written next to the evidence JSON.
#[derive(Debug, Clone, Default)]
struct Profiler {
    steps: Vec<ProfileStep>,
}

fn profile_layer(merged: &mut Vec<ProfileLayer>, layer: usize) -> &mut ProfileLayer {
    let index = match merged.iter().position(|entry| entry.layer == layer) {
        Some(index) => index,
        None => {
            merged.push(ProfileLayer {
                layer,
                ..ProfileLayer::default()
            });
            merged.len() - 1
        }
    };
    &mut merged[index]
}

/// The allocator tune the process actually requested, recorded so a run is self-describing.
///
/// The CLI main calls [allocator::tune] at startup; the writer calls it again because the call is
/// idempotent and because an evidence file can otherwise not tell a tuned run from an untuned one.
fn allocator_json(tuning: allocator::Tuning) -> Value {
    json!({
        "enabled": tuning.enabled,
        "mmap_threshold": tuning.mmap_threshold,
        "trim_threshold": tuning.trim_threshold,
        "kill_switch": allocator::KILL_SWITCH,
        "requested_mmap_threshold_bytes": allocator::REQUESTED_MMAP_THRESHOLD_BYTES,
        "requested_trim_threshold_bytes": allocator::REQUESTED_TRIM_THRESHOLD_BYTES,
        "note": "pair-only win: M_MMAP_THRESHOLD moves the per-layer 134/168 MB buffers onto the heap and M_TRIM_THRESHOLD retains the freed heap; either alone is noise, and the 32 MiB row falsifies a clamp of the 1 GiB ask",
    })
}

fn profile_json(profile: &Profiler) -> Value {
    let steps: Vec<Value> = profile
        .steps
        .iter()
        .map(|step| {
            let layers: Vec<Value> = step
                .layers
                .iter()
                .map(|layer| {
                    let attributed = layer.provider_us
                        + layer.post_us
                        + layer.ffn_local_us
                        + layer.ffn_reduce_us
                        + layer.shared_us;
                    json!({
                        "layer": layer.layer,
                        "provider_us": layer.provider_us,
                        "deferred_us": layer.deferred_us,
                        "post_us": layer.post_us,
                        "ffn_local_us": layer.ffn_local_us,
                        "ffn_reduce_us": layer.ffn_reduce_us,
                        "shared_expert_us": layer.shared_us,
                        "engine_host_ms": layer.engine_host_ms,
                        "unattributed_us": step.wall_us.saturating_sub(attributed),
                    })
                })
                .collect();
            json!({
                "position": step.position,
                "wall_us": step.wall_us,
                "user_us": step.user_us,
                "sys_us": step.sys_us,
                "rchar": step.rchar,
                "read_bytes": step.read_bytes,
                "collective_us": step.collective_us,
                "layers": layers,
            })
        })
        .collect();
    json!({
        "profile_on": true,
        "reference_threads": std::env::var("SPARKFLASH_REFERENCE_THREADS").ok(),
        "note": "timers only: provider/post are the host entry and post closures, deferred_us is the part of provider time taken when a MidLayer handoff was present, ffn_local_us is the routed local block, ffn_reduce_us the canonical all-reduce, shared_expert_us the replicated shared expert, engine_host_ms is the engine's own per-layer host timer (attention block + entry totals, same value the CLI progress callback receives); unattributed_us = step wall - measured layer components, so the engine-internal remainder is visible rather than hidden",
        "steps": steps,
    })
}

fn process_cpu_us() -> (u64, u64) {
    let Ok(stat) = fs::read_to_string("/proc/self/stat") else {
        return (0, 0);
    };
    let Some(close) = stat.rfind(')') else {
        return (0, 0);
    };
    let fields: Vec<&str> = stat[close + 1..].split_whitespace().collect();
    let ticks = |index: usize| -> u64 {
        fields
            .get(index)
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0)
    };
    // After the comm field the state is fields[0]; utime/stime are stat fields 14/15.
    (ticks(11) * 10_000, ticks(12) * 10_000)
}

/// Per-step costs recorded by the rank's evidence loop.
#[derive(Debug, Clone, Copy)]
struct StepCost {
    ms: u64,
    rchar: u64,
    read_bytes: u64,
    collective_ms: u64,
}

/// The bisection switch for prefill: when set, the runner keeps the pre-batch-17 one-step-per-token
/// loop instead of the chunked hybrid step. Read once per prefill call; the decode path never
/// consults it.
/// Chunked prefill is opt-in: the batch-17 serve census measured a slower endpoint TTFT with it
/// than with the token loop, so the serve path keeps the loop as its default until the overhead is
/// understood (batch 18). Set `SPARKFLASH_TP4_PREFILL_CHUNK=1` to run one layer-major chunk pass.
fn prefill_chunk_enabled() -> bool {
    std::env::var_os("SPARKFLASH_TP4_PREFILL_CHUNK").is_some()
}

/// One rank of the four-node TP=4 host path, as a library.
///
/// Owns the checkpoint source, the shard plan, the one resident cache, the engine, the rank's
/// collective and the sequence state. Tp4Rank::open does all the I/O and collective setup; a
/// request then runs begin_request, prefill and decode. No tokenizer, HTTP or control protocol
/// lives here: token ids are the contract.
pub struct Tp4Rank {
    rank: usize,
    world: usize,
    config: ModelConfig,
    source: Arc<SafeTensorsSource>,
    plan: ShardPlan,
    cache: Option<Arc<ResidentCache>>,
    weights: Arc<RankWeights>,
    shared: SharedSource,
    engine: InferenceEngine,
    collective: Arc<TpCollective>,
    backend: TpMoeBackend,
    state: SequenceState,
    sampling: Tp4Sampling,
    metrics: Tp4Metrics,
    resident_bytes: u64,
    source_mode: &'static str,
    profiler: Option<Profiler>,
}

impl Tp4Rank {
    /// Opens the checkpoint, builds the rank's plan and resident cache, and connects the
    /// collective. Engram tables stay on disk and are paged by read_ranges.
    pub fn open(config: &Tp4RankConfig) -> Result<Self, Tp4Error> {
        let transport = if config.world_size == 1 {
            None
        } else {
            let addresses: Vec<SocketAddr> = config
                .peers
                .iter()
                .enumerate()
                .map(|(peer, ip)| SocketAddr::new(*ip, config.base_port + peer as u16))
                .collect();
            let tcp = TcpConfig::with_addresses(config.rank, addresses)
                .map_err(|error| Tp4Error::new(format!("tcp config: {error}")))?
                .with_timeout(config.timeout)
                .with_connect_timeout(config.timeout);
            let transport = TcpTransport::connect(tcp)
                .map_err(|error| Tp4Error::new(format!("tcp connect: {error}")))?;
            Some(Box::new(transport) as Box<dyn Transport>)
        };
        let input = RankInput {
            rank: config.rank,
            world: config.world_size,
            source_dir: config.checkpoint.clone(),
            source_mode: "checkpoint",
            prompt: Vec::new(),
            max_tokens: 0,
            record_seams: false,
            reference_check: false,
            resident_cache: config.resident_cache,
            profile: false,
            bench_steps: 0,
            tokenizer: None,
        };
        Self::open_input(&input, transport, config.element_limit).map_err(Tp4Error::new)
    }

    /// Builds a rank from the CLI's parsed input; the CLI --run-stage tp4-host path goes through
    /// this together with the per-step evidence loop.
    fn open_input(
        input: &RankInput,
        transport: Option<Box<dyn Transport>>,
        element_limit: usize,
    ) -> Result<Self, String> {
        Self::open_input_inner(input, transport, element_limit).map_err(|error| error.to_string())
    }

    fn open_input_inner(
        input: &RankInput,
        transport: Option<Box<dyn Transport>>,
        element_limit: usize,
    ) -> Result<Self, Tp4Error> {
        let load_started = Instant::now();
        let config = if input.source_mode == "fixture" {
            twin_config()
        } else {
            released_config().map_err(Tp4Error::new)?
        };
        let index_path = input.source_dir.join("model.safetensors.index.json");
        let source = Arc::new(
            SafeTensorsSource::open(&input.source_dir, &index_path)
                .map_err(|error| Tp4Error::new(format!("open checkpoint: {error}")))?,
        );
        let plan = expert_shard_plan(&source, input.rank, input.world).map_err(Tp4Error::new)?;
        // One cache instance, two views: the engine's replicated reads and the rank's plan
        // slices all go through the same allocation, so the resident set is held once.
        let cache = if input.resident_cache {
            Some(Arc::new(
                ResidentCache::load(&source, &plan).map_err(Tp4Error::new)?,
            ))
        } else {
            None
        };
        let resident = match &cache {
            Some(cache) => cache.bytes(),
            None => resident_estimate(&source, &plan).map_err(Tp4Error::new)?,
        };
        let weights = Arc::new(RankWeights::new(
            Arc::clone(&source),
            plan.clone(),
            cache.clone(),
        ));
        let shared = SharedSource {
            inner: Arc::clone(&source),
            cache: cache.clone(),
        };
        let engine = InferenceEngine::new(
            config.clone(),
            Box::new(SharedSource {
                inner: Arc::clone(&source),
                cache: cache.clone(),
            }),
            element_limit,
        )
        .map_err(|error| Tp4Error::new(format!("engine: {error}")))?;
        let collective = Arc::new(TpCollective::new(input.world, transport));
        let backend = TpMoeBackend {
            local_config: local_config(&config, input.world).map_err(Tp4Error::new)?,
            global_config: config.clone(),
            local: Arc::clone(&weights),
            collective: Arc::clone(&collective),
            position: 0,
            record_seams: input.record_seams,
            seams: Vec::new(),
            profile_on: input.profile,
            ffn_profile: Vec::new(),
            shared_profile: Vec::new(),
        };
        let state = SequenceState::new(&config);
        let (rss_load_kib, rss_peak_kib) = rss_kib();
        let metrics = Tp4Metrics {
            load_ms: load_started.elapsed().as_millis() as u64,
            cache_fill_ms: cache.as_ref().map_or(0, |cache| cache.fill_ms),
            resident_cache_bytes: resident,
            resident_cache_tensors: cache.as_ref().map_or(0, |cache| cache.tensors.len()),
            rss_load_kib,
            rss_peak_kib,
            reduce_calls: 0,
            reduce_elements: 0,
            reduce_ms_total: 0,
            shard_read_calls: 0,
            shard_read_bytes: 0,
            steps: 0,
            last_step_ms: 0,
            plan_hash: plan_hash(&plan),
        };
        Ok(Self {
            rank: input.rank,
            world: input.world,
            state,
            config,
            source,
            plan,
            cache,
            weights,
            shared,
            engine,
            collective,
            backend,
            sampling: Tp4Sampling::default(),
            metrics,
            resident_bytes: resident,
            source_mode: input.source_mode,
            profiler: if input.profile {
                Some(Profiler::default())
            } else {
                None
            },
        })
    }

    /// Applies a request's sampler and clears the sequence state, keeping the engine and the
    /// resident cache.
    pub fn begin_request(&mut self, sampling: Tp4Sampling) -> Result<(), Tp4Error> {
        self.set_sampling(sampling)?;
        self.state = SequenceState::new(&self.config);
        self.backend.position = 0;
        self.backend.seams.clear();
        self.metrics.steps = 0;
        self.metrics.last_step_ms = 0;
        Ok(())
    }

    /// Applies the request sampler without resetting the sequence. Greedy only today.
    pub fn set_sampling(&mut self, sampling: Tp4Sampling) -> Result<(), Tp4Error> {
        if sampling.temperature != 0.0 {
            return Err(Tp4Error::new(
                "the TP=4 host path implements greedy decoding only (temperature must be 0.0); use the reference sampler for sampling",
            ));
        }
        self.sampling = sampling;
        Ok(())
    }

    /// Prefills the prompt and returns the last position's output. The default is the
    /// pre-batch-17 one-step-per-token loop (one hybrid forward per prompt token).
    ///
    /// Set `SPARKFLASH_TP4_PREFILL_CHUNK=1` to run the batch-17 chunked hybrid step instead: one
    /// forward per layer over the whole prompt. It is bit-identical but opt-in, because the batch-17
    /// serve census measured a slower endpoint TTFT with the chunk path than with the loop (43.6 s
    /// vs 39.1 s) while the bench harness measures the chunk faster (24.6 s vs 33.1 s at 37 prompt
    /// tokens); the serve-path overhead is batch-18 work. Single-token prompts always take the
    /// one-token path.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, Tp4Error> {
        if tokens.is_empty() {
            return Err(Tp4Error::new("prefill needs at least one token"));
        }
        if tokens.len() == 1 || !prefill_chunk_enabled() {
            self.prefill_token_loop(tokens)
        } else {
            self.prefill_chunk(tokens)
        }
    }

    /// The pre-batch-17 prefill: one hybrid step per prompt token. Kept for bisection and for the
    /// one-token case; it is exactly what every earlier RUN measured.
    fn prefill_token_loop(&mut self, tokens: &[u32]) -> Result<StepOutput, Tp4Error> {
        let mut last = StepOutput {
            token: None,
            logits: Vec::new(),
        };
        for &token in tokens {
            last = self.step(token)?;
        }
        Ok(last)
    }

    /// The chunked prefill: one `run_hybrid_sequence_chunk` call over the whole prompt.
    ///
    /// The engine still asks the host for one provider entry per (layer, position) and one post
    /// handoff per layer for the whole chunk; the runner feeds per-position Engram hashes and the
    /// chunk-sized post buffers, then records ONE profile entry for the chunk (`prefill_tokens`
    /// of them) so the decode-only slice stays the last `steps_run` entries.
    fn prefill_chunk(&mut self, tokens: &[u32]) -> Result<StepOutput, Tp4Error> {
        let Self {
            config,
            source,
            shared,
            engine,
            collective,
            backend,
            state,
            profiler,
            ..
        } = self;
        let start_pos = state.tokens().len();
        let seqlen = tokens.len();
        // One embedding row per token, concatenated in token order: the chunk entry layout.
        let mut embeddings = Vec::with_capacity(seqlen * config.dim);
        for &token in tokens {
            let row = read_embedding_row(source, config, token).map_err(Tp4Error::new)?;
            embeddings.extend_from_slice(&row);
        }
        // The n-gram rows for every chunk position, hashed once against history + chunk; each
        // token takes its own per_token slice of the flat [positions, layers, columns] buffer.
        let mut history = state.tokens().to_vec();
        history.extend_from_slice(tokens);
        let mut hashes = Vec::new();
        engram::ngram_hashes(config, &history, start_pos, &mut hashes)
            .map_err(|error| Tp4Error::new(format!("engram hashes: {error}")))?;
        let per_token = hashes.len() / seqlen.max(1);
        backend.position = start_pos;
        let providers = HostProviders {
            config,
            source: shared,
        };
        let profiling = profiler.is_some();
        let mut layer_provider: Vec<(usize, u64, u64)> = Vec::new();
        let mut layer_post: Vec<(usize, u64)> = Vec::new();
        let mut provider = |layer: usize,
                            position: usize,
                            hidden: &[f32],
                            pre_mix: &[f32],
                            deferred: Option<&DeferredHandoff>| {
            let started = Instant::now();
            let span = position * per_token..(position + 1) * per_token;
            let result = providers.deferred_entry(layer, hidden, pre_mix, deferred, &hashes[span]);
            if profiling {
                let micros = started.elapsed().as_micros() as u64;
                let deferred_us = if deferred.is_some() { micros } else { 0 };
                layer_provider.push((layer, micros, deferred_us));
            }
            result
        };
        let mut post = |layer: usize, residual: &[f32], attention_out: &[f32], mixes: &Mixes| {
            let started = Instant::now();
            let result = providers.post(layer, residual, attention_out, mixes);
            if profiling {
                layer_post.push((layer, started.elapsed().as_micros() as u64));
            }
            result
        };
        let mut layer_progress: Vec<(usize, u64)> = Vec::new();
        let mut progress = |layer: usize, host_ms: u64| {
            if profiling {
                layer_progress.push((layer, host_ms));
            }
        };
        let (rchar_before, read_before) = io_counters();
        let collective_before = collective.elapsed_ms();
        let (user_before, sys_before) = process_cpu_us();
        let started = Instant::now();
        let output = engine
            .run_hybrid_sequence_chunk(
                state,
                tokens,
                &embeddings,
                start_pos,
                &mut provider,
                &mut post,
                // A chunk has no attention-projection seam: the host attention runs inside the
                // block over the chunk's positions.
                None,
                Some(backend),
                &mut progress,
            )
            .map_err(|error| Tp4Error::new(format!("prefill chunk at {start_pos}: {error}")))?;
        let (rchar_after, read_after) = io_counters();
        if let Some(profiler) = profiler {
            let (user_after, sys_after) = process_cpu_us();
            let mut merged: Vec<ProfileLayer> = Vec::new();
            for (layer, provider_us, deferred_us) in layer_provider.drain(..) {
                let entry = profile_layer(&mut merged, layer);
                entry.provider_us += provider_us;
                entry.deferred_us += deferred_us;
            }
            for (layer, post_us) in layer_post.drain(..) {
                profile_layer(&mut merged, layer).post_us += post_us;
            }
            for (layer, local_us, reduce_us) in backend.ffn_profile.drain(..) {
                let entry = profile_layer(&mut merged, layer);
                entry.ffn_local_us = local_us;
                entry.ffn_reduce_us = reduce_us;
            }
            for (layer, shared_us) in backend.shared_profile.drain(..) {
                profile_layer(&mut merged, layer).shared_us = shared_us;
            }
            for (layer, host_ms) in layer_progress.drain(..) {
                profile_layer(&mut merged, layer).engine_host_ms = host_ms;
            }
            merged.sort_by_key(|entry| entry.layer);
            profiler.steps.push(ProfileStep {
                position: start_pos,
                wall_us: started.elapsed().as_micros() as u64,
                user_us: user_after.saturating_sub(user_before),
                sys_us: sys_after.saturating_sub(sys_before),
                rchar: rchar_after.saturating_sub(rchar_before),
                read_bytes: read_after.saturating_sub(read_before),
                collective_us: collective.elapsed_ms().saturating_sub(collective_before) * 1000,
                layers: merged,
            });
        }
        let (_, rss_peak_kib) = rss_kib();
        let metrics = &mut self.metrics;
        metrics.steps += seqlen as u64;
        metrics.last_step_ms = started.elapsed().as_millis() as u64;
        metrics.rss_peak_kib = metrics.rss_peak_kib.max(rss_peak_kib);
        metrics.reduce_calls = self.collective.calls.load(Ordering::Relaxed);
        metrics.reduce_elements = self.collective.elements.load(Ordering::Relaxed);
        metrics.reduce_ms_total = self.collective.elapsed_ms();
        metrics.shard_read_calls = self.weights.shard_read_calls();
        metrics.shard_read_bytes = self.weights.shard_read_bytes();
        Ok(StepOutput {
            token: output.token,
            logits: output.logits,
        })
    }

    /// Consumes one token and returns the token that follows it.
    pub fn decode(&mut self, token: u32) -> Result<StepOutput, Tp4Error> {
        self.step(token)
    }

    /// The rank's running counters.
    pub fn metrics(&self) -> &Tp4Metrics {
        &self.metrics
    }

    /// Zero-based rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Number of ranks in the collective.
    pub fn world_size(&self) -> usize {
        self.world
    }
    /// The active request sampler.
    pub fn sampling(&self) -> Tp4Sampling {
        self.sampling
    }

    fn step(&mut self, token: u32) -> Result<StepOutput, Tp4Error> {
        let position = self.state.tokens().len();
        let (output, _cost) = self.step_with_cost(token, position)?;
        Ok(output)
    }

    /// Runs one forward at the given position and records the step's cost. The evidence loop
    /// uses this directly.
    fn step_with_cost(
        &mut self,
        token: u32,
        position: usize,
    ) -> Result<(StepOutput, StepCost), Tp4Error> {
        let Self {
            config,
            source,
            shared,
            engine,
            collective,
            backend,
            state,
            profiler,
            ..
        } = self;
        let mut history = state.tokens().to_vec();
        history.push(token);
        let mut hashes = Vec::new();
        engram::ngram_hashes(config, &history, position, &mut hashes)
            .map_err(|error| Tp4Error::new(format!("engram hashes: {error}")))?;
        let embedding = read_embedding_row(source, config, token).map_err(Tp4Error::new)?;
        backend.position = position;
        let providers = HostProviders {
            config,
            source: shared,
        };
        let profiling = profiler.is_some();
        let mut layer_provider: Vec<(usize, u64, u64)> = Vec::new();
        let mut layer_post: Vec<(usize, u64)> = Vec::new();
        let mut provider =
            |layer: usize, hidden: &[f32], pre_mix: &[f32], deferred: Option<&DeferredHandoff>| {
                let started = Instant::now();
                let result = providers.deferred_entry(layer, hidden, pre_mix, deferred, &hashes);
                if profiling {
                    let micros = started.elapsed().as_micros() as u64;
                    let deferred_us = if deferred.is_some() { micros } else { 0 };
                    layer_provider.push((layer, micros, deferred_us));
                }
                result
            };
        let mut post = |layer: usize, residual: &[f32], attention_out: &[f32], mixes: &Mixes| {
            let started = Instant::now();
            let result = providers.post(layer, residual, attention_out, mixes);
            if profiling {
                layer_post.push((layer, started.elapsed().as_micros() as u64));
            }
            result
        };
        let mut layer_progress: Vec<(usize, u64)> = Vec::new();
        let mut progress = |layer: usize, host_ms: u64| {
            if profiling {
                layer_progress.push((layer, host_ms));
            }
        };
        let (rchar_before, read_before) = io_counters();
        let collective_before = collective.elapsed_ms();
        let (user_before, sys_before) = process_cpu_us();
        let started = Instant::now();
        let output = engine
            .run_hybrid_sequence_step(
                state,
                token,
                &embedding,
                position,
                &mut provider,
                &mut post,
                None,
                Some(backend),
                &mut progress,
            )
            .map_err(|error| Tp4Error::new(format!("step {position}: {error}")))?;
        let (rchar_after, read_after) = io_counters();
        let cost = StepCost {
            ms: started.elapsed().as_millis() as u64,
            rchar: rchar_after.saturating_sub(rchar_before),
            read_bytes: read_after.saturating_sub(read_before),
            collective_ms: collective.elapsed_ms().saturating_sub(collective_before),
        };
        if let Some(profiler) = profiler {
            let (user_after, sys_after) = process_cpu_us();
            let mut merged: Vec<ProfileLayer> = Vec::new();
            for (layer, provider_us, deferred_us) in layer_provider.drain(..) {
                let entry = profile_layer(&mut merged, layer);
                entry.provider_us = provider_us;
                entry.deferred_us = deferred_us;
            }
            for (layer, post_us) in layer_post.drain(..) {
                profile_layer(&mut merged, layer).post_us = post_us;
            }
            for (layer, local_us, reduce_us) in backend.ffn_profile.drain(..) {
                let entry = profile_layer(&mut merged, layer);
                entry.ffn_local_us = local_us;
                entry.ffn_reduce_us = reduce_us;
            }
            for (layer, shared_us) in backend.shared_profile.drain(..) {
                profile_layer(&mut merged, layer).shared_us = shared_us;
            }
            for (layer, host_ms) in layer_progress.drain(..) {
                profile_layer(&mut merged, layer).engine_host_ms = host_ms;
            }
            merged.sort_by_key(|entry| entry.layer);
            profiler.steps.push(ProfileStep {
                position,
                wall_us: started.elapsed().as_micros() as u64,
                user_us: user_after.saturating_sub(user_before),
                sys_us: sys_after.saturating_sub(sys_before),
                rchar: rchar_after.saturating_sub(rchar_before),
                read_bytes: read_after.saturating_sub(read_before),
                collective_us: collective.elapsed_ms().saturating_sub(collective_before) * 1000,
                layers: merged,
            });
        }
        let (_, rss_peak_kib) = rss_kib();
        let metrics = &mut self.metrics;
        metrics.steps += 1;
        metrics.last_step_ms = cost.ms;
        metrics.rss_peak_kib = metrics.rss_peak_kib.max(rss_peak_kib);
        metrics.reduce_calls = self.collective.calls.load(Ordering::Relaxed);
        metrics.reduce_elements = self.collective.elements.load(Ordering::Relaxed);
        metrics.reduce_ms_total = self.collective.elapsed_ms();
        metrics.shard_read_calls = self.weights.shard_read_calls();
        metrics.shard_read_bytes = self.weights.shard_read_bytes();
        Ok((
            StepOutput {
                token: output.token,
                logits: output.logits,
            },
            cost,
        ))
    }
}

/// The 40-layer tiny twin's configuration.
fn twin_config() -> ModelConfig {
    let mut config = ModelConfig::tiny();
    config.n_layers = 40;
    config.compress_ratios = vec![0; config.n_layers + config.n_mtp_layers];
    config.engram_layer_ids = vec![1, 14];
    config.engram_num_embeddings = vec![1024; 2];
    config.moe_inter_dim = TWIN_MOE_INTER_DIM;
    config
}

fn released_config() -> Result<ModelConfig, String> {
    let config = ModelConfig::v41_flash_cached().clone();
    // The cached construction is the released model; validating here keeps a bad build from
    // reaching a node run half-way through a layer.
    config
        .validate()
        .map_err(|error| format!("config: {error}"))?;
    Ok(config)
}

/// The finite f32 value the twin fixture writes, the same recipe as the engine test source.
fn fixture_value(index: usize, salt: u64, seed: u64) -> f32 {
    let mut hashed = seed ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    hashed ^= index as u64;
    for _ in 0..3 {
        hashed = hashed
            .wrapping_mul(0xbf58_476d_1ce4_e5b9)
            .rotate_left(17)
            .wrapping_add(0x94d0_49bb_1331_11eb);
    }
    0.01 + 0.0001 * ((hashed >> 56) as f32)
}

fn name_salt(name: &str) -> u64 {
    let mut salt = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.as_bytes() {
        salt ^= u64::from(*byte);
        salt = salt.wrapping_mul(0x0000_0100_0000_01b3);
    }
    salt
}

/// One written fixture: how many tensors and bytes, and its FNV-1a content hash.
#[derive(Debug, Clone)]
struct FixtureSummary {
    tensors: usize,
    bytes: u64,
    hash: u64,
}

/// Writes the deterministic 40-layer twin safetensors fixture and its index.
fn write_twin_fixture(dir: &Path) -> Result<FixtureSummary, String> {
    fs::create_dir_all(dir).map_err(|error| format!("create {}: {error}", dir.display()))?;
    let config = twin_config();
    let plan = TensorPlan::language_model(&config);
    let mut entries: Vec<String> = Vec::with_capacity(plan.specs().len());
    let mut data: Vec<u8> = Vec::new();
    let mut tensors = 0usize;
    for spec in plan.specs() {
        let (dtype_word, shape, bytes) = fixture_tensor_bytes(spec);
        entries.push(format!(
            "{}:{{\"dtype\":\"{}\",\"shape\":{},\"data_offsets\":[{},{}]}}",
            serde_json::to_string(&spec.name).map_err(|error| error.to_string())?,
            dtype_word,
            serde_json::to_string(&shape).map_err(|error| error.to_string())?,
            data.len(),
            data.len() + bytes.len()
        ));
        data.extend_from_slice(&bytes);
        tensors += 1;
    }
    let header = format!(
        "{{{},\"__metadata__\":{{\"format\":\"pt\"}}}}",
        entries.join(",")
    );
    let header_len = header.len() as u64;
    let mut file = Vec::with_capacity(8 + header.len() + data.len());
    file.extend_from_slice(&header_len.to_le_bytes());
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);
    let shard_path = dir.join("model.safetensors");
    fs::write(&shard_path, &file)
        .map_err(|error| format!("write {}: {error}", shard_path.display()))?;
    let index = json!({
        "metadata": {"total_size": data.len()},
        "weight_map": plan.specs().iter().map(|spec| (spec.name.clone(), "model.safetensors".to_owned())).collect::<BTreeMap<_, _>>(),
    });
    let index_path = dir.join("model.safetensors.index.json");
    fs::write(
        &index_path,
        serde_json::to_vec_pretty(&index).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write {}: {error}", index_path.display()))?;
    Ok(FixtureSummary {
        tensors,
        bytes: data.len() as u64,
        hash: sparkflash_engine::synthetic::fnv1a(&file),
    })
}

/// The dtype word, stored shape and bytes for one twin tensor.
///
/// Everything is finite f32 except the engram tensors whose plan storage is FP8, because the
/// engine's paged Engram reader is a byte-range reader that must be described by the header (the
/// table rows are 1 byte per value and 1 scale per 32 values); writing those as f32 would make the
/// paged read decode the wrong bytes. The other Engram tensors stay f32.
fn fixture_tensor_bytes(spec: &TensorSpec) -> (&'static str, Vec<usize>, Vec<u8>) {
    let engram = spec.name.contains(".engram.");
    if engram && spec.dtype == DType::F8E4M3 {
        let salt = name_salt(&spec.name);
        let elements = spec.elements() as usize;
        let bytes = (0..elements)
            .map(|index| quant::e4m3_encode(fixture_value(index, salt, 0x5eed)))
            .collect();
        ("F8_E4M3", spec.shape.clone(), bytes)
    } else if engram && spec.dtype == DType::F8E8M0 {
        let elements = spec.elements() as usize;
        ("F8_E8M0", spec.shape.clone(), vec![127u8; elements])
    } else {
        let salt = name_salt(&spec.name);
        let elements = spec.elements() as usize;
        let mut bytes = Vec::with_capacity(elements * 4);
        for index in 0..elements {
            bytes.extend_from_slice(&fixture_value(index, salt, 0x5eed).to_le_bytes());
        }
        ("F32", spec.shape.clone(), bytes)
    }
}

/// Every non-Engram tensor the forward can touch, held in this process's memory exactly once.
///
/// The released checkpoint scatters 12.7 GiB of replicated weights plus the rank's 68.9 GiB expert
/// slice across 48 shard files. Reading them through `read()` on every step delivered 9.65 GB per
/// token and cost about 16 s/token, which is the same page-cache churn the single-node baseline
/// pays. This cache is the vLLM shape: the raw stored bytes (not decoded f32, which would be 2-4x
/// larger) are read once at startup and every later read is a memcpy. Only the Engram tables stay
/// on NVMe, because their 188.8 GiB cannot fit and the forward pages rows, not tensors.
struct ResidentCache {
    tensors: BTreeMap<String, Arc<Vec<u8>>>,
    plan_bytes: u64,
    replicated_bytes: u64,
    fill_ms: u64,
}

impl ResidentCache {
    /// Reads every plan slice and every non-table checkpoint tensor once, in that order.
    fn load(source: &SafeTensorsSource, plan: &ShardPlan) -> Result<Self, String> {
        let started = Instant::now();
        let mut tensors = BTreeMap::new();
        let mut plan_bytes = 0u64;
        let mut replicated_bytes = 0u64;
        for name in plan.rules().keys() {
            let shard = plan
                .read(source, name)
                .map_err(|error| format!("resident cache {name}: {error}"))?;
            plan_bytes = plan_bytes.saturating_add(shard.bytes.len() as u64);
            tensors.insert(name.to_owned(), Arc::new(shard.bytes));
        }
        for name in source.index().names() {
            if plan.rule(name).is_some() || is_engram_table(name) {
                continue;
            }
            let bytes = source
                .read(name)
                .map_err(|error| format!("resident cache {name}: {error}"))?;
            replicated_bytes = replicated_bytes.saturating_add(bytes.len() as u64);
            tensors.insert(name.to_owned(), Arc::new(bytes));
        }
        Ok(Self {
            tensors,
            plan_bytes,
            replicated_bytes,
            fill_ms: started.elapsed().as_millis() as u64,
        })
    }

    fn bytes(&self) -> u64 {
        self.plan_bytes.saturating_add(self.replicated_bytes)
    }

    fn get(&self, name: &str) -> Option<Vec<u8>> {
        self.tensors.get(name).map(|bytes| bytes.as_ref().clone())
    }
}

/// The resident-set size and high-water mark this process reports, straight from /proc.
fn rss_kib() -> (u64, u64) {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return (0, 0);
    };
    let mut rss = 0u64;
    let mut peak = 0u64;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            rss = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("VmHWM:") {
            peak = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
        }
    }
    (rss, peak)
}

/// A global source over the checkpoint, shared by the engine and the host providers.
///
/// Reads are answered from the resident cache when it holds the name; the range reader is left on
/// the checkpoint because its only callers are the Engram tables and the embedding row.
struct SharedSource {
    inner: Arc<SafeTensorsSource>,
    cache: Option<Arc<ResidentCache>>,
}

impl WeightSource for SharedSource {
    fn names(&self) -> Vec<String> {
        self.inner.names()
    }
    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.inner.spec(name)
    }
    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        if let Some(cache) = &self.cache
            && let Some(bytes) = cache.get(name)
        {
            return Ok(bytes);
        }
        self.inner.read(name)
    }
    fn read_view(&self, name: &str) -> Result<Option<&[u8]>, WeightError> {
        Ok(self
            .cache
            .as_ref()
            .and_then(|cache| cache.tensors.get(name))
            .map(|bytes| bytes.as_slice()))
    }
    fn read_ranges(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
    ) -> Option<Result<Vec<u8>, WeightError>> {
        Some(self.inner.read_ranges(name, ranges))
    }
}

/// The rank's weight view: plan-named expert tensors are sliced, everything else is the full tensor.
struct RankWeights {
    source: Arc<SafeTensorsSource>,
    plan: ShardPlan,
    world: usize,
    cache: Option<Arc<ResidentCache>>,
    read_calls: AtomicU64,
    read_bytes: AtomicU64,
    cache_hits: AtomicU64,
}

impl RankWeights {
    fn new(
        source: Arc<SafeTensorsSource>,
        plan: ShardPlan,
        cache: Option<Arc<ResidentCache>>,
    ) -> Self {
        Self {
            world: plan.tp_world(),
            source,
            plan,
            cache,
            read_calls: AtomicU64::new(0),
            read_bytes: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
        }
    }

    fn cache_hits(&self) -> u64 {
        self.cache_hits.load(Ordering::Relaxed)
    }

    fn shard_read_calls(&self) -> u64 {
        self.read_calls.load(Ordering::Relaxed)
    }

    fn shard_read_bytes(&self) -> u64 {
        self.read_bytes.load(Ordering::Relaxed)
    }
}

fn shard_error_to_weight(name: &str, error: ShardError) -> WeightError {
    match error {
        ShardError::Weights(inner) => inner,
        ShardError::Plan(detail) => WeightError::Malformed {
            name: name.to_owned(),
            detail,
        },
    }
}

impl WeightSource for RankWeights {
    fn names(&self) -> Vec<String> {
        self.source.names()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        match self.plan.rule(name) {
            None => self.source.spec(name),
            Some(rule) => {
                let header = self.source.header(name)?;
                let descriptor = header.tensor(name).map_err(|error| WeightError::Missing {
                    name: format!("{name} ({error})"),
                })?;
                shard_spec(name, descriptor, rule, self.world).map_err(|detail| {
                    WeightError::Malformed {
                        name: name.to_owned(),
                        detail,
                    }
                })
            }
        }
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        if let Some(cache) = &self.cache
            && let Some(bytes) = cache.get(name)
        {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(bytes);
        }
        match self.plan.rule(name) {
            None => self.source.read(name),
            Some(_) => {
                let shard = self
                    .plan
                    .read(&self.source, name)
                    .map_err(|error| shard_error_to_weight(name, error))?;
                self.read_calls.fetch_add(1, Ordering::Relaxed);
                self.read_bytes
                    .fetch_add(shard.bytes.len() as u64, Ordering::Relaxed);
                Ok(shard.bytes)
            }
        }
    }

    fn read_ranges(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
    ) -> Option<Result<Vec<u8>, WeightError>> {
        match self.plan.rule(name) {
            // The plan's own range reader is used through read(); a caller that pages a sharded
            // tensor would have to translate its offsets, and no code path does that.
            Some(_) => None,
            None => WeightSource::read_ranges(&*self.source, name, ranges),
        }
    }
}

/// The local logical spec of one plan slice, from the stored header.
fn shard_spec(
    name: &str,
    descriptor: &TensorHeader,
    rule: ShardRule,
    world: usize,
) -> Result<TensorSpec, String> {
    let mut stored = descriptor.shape.clone();
    match rule {
        ShardRule::Replicated => {}
        ShardRule::Column => stored[0] /= world,
        ShardRule::Row => stored[1] /= world,
    }
    let (dtype, shape) = logical_from_stored(&descriptor.dtype, stored)?;
    Ok(TensorSpec {
        name: name.to_owned(),
        dtype,
        shape,
    })
}

/// Maps a stored dtype word and shape to the logical dtype and shape the arithmetic sees.
fn logical_from_stored(word: &str, stored: Vec<usize>) -> Result<(DType, Vec<usize>), String> {
    let mut shape = stored;
    let dtype = match word {
        "F32" => DType::F32,
        "BF16" => DType::BF16,
        "F16" => DType::F16,
        "F8_E4M3" => DType::F8E4M3,
        "F8_E8M0" => DType::F8E8M0,
        "I8" => {
            let last = shape
                .last_mut()
                .ok_or_else(|| "I8 tensor with no last dimension".to_owned())?;
            *last = last.saturating_mul(2);
            DType::F4E2M1
        }
        other => return Err(format!("unknown stored dtype {other}")),
    };
    Ok((dtype, shape))
}

/// The Phase-1 plan: routed experts sharded by the section-3 rule, everything else replicated.
fn expert_shard_plan(
    source: &SafeTensorsSource,
    rank: usize,
    world: usize,
) -> Result<ShardPlan, String> {
    let mut plan = ShardPlan::new(rank, world).map_err(|error| format!("shard plan: {error}"))?;
    for name in source.index().names() {
        let Some(rule) = expert_rule(name) else {
            continue;
        };
        plan.insert(name, rule);
    }
    plan.verify(source)
        .map_err(|error| format!("shard plan verify: {error}"))?;
    Ok(plan)
}

/// The shard rule for one checkpoint name, if it is a routed expert the Phase-1 plan shards.
fn expert_rule(name: &str) -> Option<ShardRule> {
    if !name.starts_with("layers.") || !name.contains(".ffn.experts.") {
        return None;
    }
    if name.ends_with(".w1.weight")
        || name.ends_with(".w3.weight")
        || name.ends_with(".w1.scale")
        || name.ends_with(".w3.scale")
    {
        Some(ShardRule::Column)
    } else if name.ends_with(".w2.weight") || name.ends_with(".w2.scale") {
        Some(ShardRule::Row)
    } else {
        None
    }
}

/// One rank's canonical all-reduce seam plus the evidence counters around it.
struct TpCollective {
    transport: Option<Box<dyn Transport>>,
    seq: AtomicU64,
    world: usize,
    label: String,
    calls: AtomicU64,
    elements: AtomicU64,
    ms: AtomicU64,
}

impl TpCollective {
    fn new(world: usize, transport: Option<Box<dyn Transport>>) -> Self {
        Self {
            transport,
            seq: AtomicU64::new(0),
            world,
            label: format!("tcp-hostonly/world{world}"),
            calls: AtomicU64::new(0),
            elements: AtomicU64::new(0),
            ms: AtomicU64::new(0),
        }
    }

    fn elapsed_ms(&self) -> u64 {
        self.ms.load(Ordering::Relaxed)
    }

    /// All-reduces one f32 vector with the canonical rule: bf16 payloads, f32 ascending-rank sum,
    /// one bf16 RNE round. At world 1 this is a strict no-op, not a round trip.
    fn all_reduce_f32(&self, values: &mut [f32]) -> Result<(), String> {
        if self.world == 1 {
            return Ok(());
        }
        let started = Instant::now();
        let mut buffer: Vec<u16> = values.iter().map(|value| f32_to_bf16_rn(*value)).collect();
        let layout =
            CollectiveLayout::new(self.world, buffer.len(), FabricSchedule::OneShotAllToAll)
                .map_err(|error| format!("collective layout: {error}"))?;
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        let transport = self
            .transport
            .as_deref()
            .ok_or("world > 1 needs a transport")?;
        transport
            .all_reduce_bf16(&mut buffer, seq, &layout)
            .map_err(|error| format!("all-reduce {seq}: {error}"))?;
        for (value, bits) in values.iter_mut().zip(buffer) {
            *value = bf16_to_f32(bits);
        }
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.elements
            .fetch_add(values.len() as u64, Ordering::Relaxed);
        self.ms
            .fetch_add(started.elapsed().as_millis() as u64, Ordering::Relaxed);
        Ok(())
    }
}

/// The host half of the deferred chain: the same arithmetic as the engine's host-control tests.
struct HostProviders<'a> {
    config: &'a ModelConfig,
    source: &'a SharedSource,
}

impl HostProviders<'_> {
    fn entry(
        &self,
        layer: usize,
        hidden: &[f32],
        pre_mix: &[f32],
    ) -> Result<HybridEntry, EngineError> {
        let config = self.config;
        let source = self.source;
        let prefix = format!("layers.{layer}");
        let dim = config.dim;
        let limit = ELEMENT_LIMIT;
        let fn_w = linalg::decode_by_name(source, &format!("{prefix}.hc_attn_fn"), limit)?;
        let scale = linalg::decode_by_name(source, &format!("{prefix}.hc_attn_scale"), limit)?;
        let base = linalg::decode_by_name(source, &format!("{prefix}.hc_attn_base"), limit)?;
        let mut mixes = Mixes::for_config(config);
        mhc::hc_mixes(config, hidden, &fn_w, &scale, &base, &mut mixes)?;
        let mut attn_in = vec![0.0f32; dim];
        mhc::hc_pre(config, pre_mix, hidden, &mut attn_in)?;
        let norm = linalg::decode_by_name(source, &format!("{prefix}.attn_norm.weight"), limit)?;
        let mut attention_hidden = attn_in;
        linalg::rms_norm(&mut attention_hidden, &norm, dim, config.norm_eps as f32)?;
        Ok(HybridEntry {
            entry: LayerZeroEntry {
                pre_attention_hidden: attention_hidden,
                pre: mixes.pre,
                post: mixes.post,
                comb: mixes.comb,
                source: LayerZeroEntrySource::Host,
            },
            x: hidden.to_vec(),
        })
    }

    fn deferred_entry(
        &self,
        layer: usize,
        hidden: &[f32],
        pre_mix: &[f32],
        deferred: Option<&DeferredHandoff>,
        hashes: &[u32],
    ) -> Result<HybridEntry, EngineError> {
        let mut x = if let Some(deferred) = deferred {
            let mut x = vec![0.0f32; hidden.len()];
            mhc::hc_post(
                self.config,
                &deferred.y,
                &deferred.x,
                &deferred.mixes.post,
                &deferred.mixes.comb,
                &mut x,
            )?;
            x
        } else {
            hidden.to_vec()
        };
        if self.config.engram_rows(layer).is_some() {
            engram::engram_forward(
                self.config,
                layer,
                &mut x,
                hashes,
                self.source,
                ELEMENT_LIMIT,
            )?;
        }
        self.entry(layer, &x, pre_mix)
    }

    fn post(
        &self,
        layer: usize,
        residual: &[f32],
        attention_out: &[f32],
        attn_mixes: &Mixes,
    ) -> Result<HybridPost, EngineError> {
        let config = self.config;
        let source = self.source;
        let prefix = format!("layers.{layer}");
        let dim = config.dim;
        let width = config.hc_width();
        let limit = ELEMENT_LIMIT;
        // A chunk hands over one row group per token: positions = values / hc_width. Every mHC
        // helper below is row-count driven, so the single-position call is the positions == 1 case
        // of the same code and produces the same bytes it always did.
        if width == 0 || !residual.len().is_multiple_of(width) {
            return Err(EngineError::Shape {
                what: format!("layer {layer} host post"),
                detail: format!(
                    "{} residual values are not whole {width}-wide token rows",
                    residual.len()
                ),
            });
        }
        let positions = residual.len() / width;
        if attention_out.len() != positions * dim {
            return Err(EngineError::Shape {
                what: format!("layer {layer} host post"),
                detail: format!(
                    "attention output is {} values, expected {}",
                    attention_out.len(),
                    positions * dim
                ),
            });
        }
        let mut x = vec![0.0f32; residual.len()];
        mhc::hc_post(
            config,
            attention_out,
            residual,
            &attn_mixes.post,
            &attn_mixes.comb,
            &mut x,
        )?;
        let fn_w = linalg::decode_by_name(source, &format!("{prefix}.hc_ffn_fn"), limit)?;
        let scale = linalg::decode_by_name(source, &format!("{prefix}.hc_ffn_scale"), limit)?;
        let base = linalg::decode_by_name(source, &format!("{prefix}.hc_ffn_base"), limit)?;
        let mut ffn_mixes = Mixes::for_tokens(config, positions);
        mhc::hc_mixes(config, &x, &fn_w, &scale, &base, &mut ffn_mixes)?;
        let mut ffn_in = vec![0.0f32; positions * dim];
        mhc::hc_pre(config, &attn_mixes.pre, &x, &mut ffn_in)?;
        let ffn_norm = linalg::decode_by_name(source, &format!("{prefix}.ffn_norm.weight"), limit)?;
        linalg::rms_norm(&mut ffn_in, &ffn_norm, dim, config.norm_eps as f32)?;
        Ok(HybridPost {
            x,
            ffn_in,
            ffn_mixes,
            source: LayerZeroEntrySource::Host,
        })
    }
}

/// The routed-expert half of the MoE block: local shards, then one canonical all-reduce.
struct TpMoeBackend {
    local_config: ModelConfig,
    global_config: ModelConfig,
    local: Arc<RankWeights>,
    collective: Arc<TpCollective>,
    position: usize,
    record_seams: bool,
    seams: Vec<SeamRecord>,
    profile_on: bool,
    ffn_profile: Vec<(usize, u64, u64)>,
    shared_profile: Vec<(usize, u64)>,
}

impl MoeDeviceBackend for TpMoeBackend {
    fn ffn(
        &mut self,
        _config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        source: &dyn WeightSource,
        limit: usize,
        out: &mut [f32],
    ) -> Result<bool, EngineError> {
        let profile_started = Instant::now();
        let mut scratch = MoeScratch::for_config(&self.local_config);
        moe::routed_forward(
            &self.local_config,
            prefix,
            x,
            &*self.local,
            limit,
            &mut scratch,
            out,
        )?;
        let local_us = profile_started.elapsed().as_micros() as u64;
        self.collective
            .all_reduce_f32(out)
            .map_err(collective_error)?;
        if self.profile_on {
            let reduce_us = profile_started.elapsed().as_micros() as u64 - local_us;
            self.ffn_profile.push((layer, local_us, reduce_us));
        }
        if self.record_seams {
            let mut reference_scratch = MoeScratch::for_config(&self.global_config);
            let mut reference = vec![0.0f32; out.len()];
            moe::routed_forward(
                &self.global_config,
                prefix,
                x,
                source,
                limit,
                &mut reference_scratch,
                &mut reference,
            )?;
            self.seams
                .push(seam_record(layer, self.position, &reference, out));
        }
        Ok(true)
    }

    fn shared_expert(
        &mut self,
        _config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        source: &dyn WeightSource,
        limit: usize,
        out: &mut [f32],
    ) -> Result<bool, EngineError> {
        let profile_started = Instant::now();
        let mut scratch = MoeScratch::for_config(&self.global_config);
        moe::add_shared_expert(
            &self.global_config,
            prefix,
            x,
            source,
            limit,
            &mut scratch,
            out,
        )?;
        if self.profile_on {
            self.shared_profile
                .push((layer, profile_started.elapsed().as_micros() as u64));
        }
        Ok(true)
    }

    /// The chunk seam: one routed pass over the whole chunk, then one all-reduce over every token's
    /// partial row. [moe::routed_forward] is already chunk-capable (its choice groups and its
    /// contrib buffer are sized per chunk), and the batched F4 kernel now takes every token's
    /// choices, so this is the same arithmetic the per-token loop ran, in one call per layer.
    fn ffn_chunk(
        &mut self,
        _config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        source: &dyn WeightSource,
        limit: usize,
        tokens: usize,
        out: &mut [f32],
    ) -> Result<bool, EngineError> {
        let dim = self.local_config.dim;
        if tokens == 0 || x.len() != tokens * dim || out.len() != tokens * dim {
            return Err(EngineError::Shape {
                what: format!("layer {layer} chunked routed FFN"),
                detail: format!(
                    "{tokens} tokens need {} values, got x {} and out {}",
                    tokens * dim,
                    x.len(),
                    out.len()
                ),
            });
        }
        let profile_started = Instant::now();
        let mut scratch = MoeScratch::for_config(&self.local_config);
        moe::routed_forward(
            &self.local_config,
            prefix,
            x,
            &*self.local,
            limit,
            &mut scratch,
            out,
        )?;
        let local_us = profile_started.elapsed().as_micros() as u64;
        self.collective
            .all_reduce_f32(out)
            .map_err(collective_error)?;
        if self.profile_on {
            let reduce_us = profile_started.elapsed().as_micros() as u64 - local_us;
            self.ffn_profile.push((layer, local_us, reduce_us));
        }
        if self.record_seams {
            let mut reference_scratch = MoeScratch::for_config(&self.global_config);
            let mut reference = vec![0.0f32; out.len()];
            moe::routed_forward(
                &self.global_config,
                prefix,
                x,
                source,
                limit,
                &mut reference_scratch,
                &mut reference,
            )?;
            self.seams
                .push(seam_record(layer, self.position, &reference, out));
        }
        Ok(true)
    }

    /// The chunk seam of the dense shared expert: replicated, one pass over the chunk, added to the
    /// already-reduced routed rows.
    fn shared_expert_chunk(
        &mut self,
        _config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        source: &dyn WeightSource,
        limit: usize,
        tokens: usize,
        out: &mut [f32],
    ) -> Result<bool, EngineError> {
        let dim = self.global_config.dim;
        if tokens == 0 || x.len() != tokens * dim || out.len() != tokens * dim {
            return Err(EngineError::Shape {
                what: format!("layer {layer} chunked shared expert"),
                detail: format!(
                    "{tokens} tokens need {} values, got x {} and out {}",
                    tokens * dim,
                    x.len(),
                    out.len()
                ),
            });
        }
        let profile_started = Instant::now();
        let mut scratch = MoeScratch::for_config(&self.global_config);
        moe::add_shared_expert(
            &self.global_config,
            prefix,
            x,
            source,
            limit,
            &mut scratch,
            out,
        )?;
        if self.profile_on {
            self.shared_profile
                .push((layer, profile_started.elapsed().as_micros() as u64));
        }
        Ok(true)
    }
}

fn collective_error(detail: String) -> EngineError {
    EngineError::Shape {
        what: "tp4-host routed all-reduce".to_owned(),
        detail,
    }
}

/// Relative L2 of two f32 vectors: ||a-b||2 / max(||a||2, ||b||2).
fn relative_l2(reference: &[f32], actual: &[f32]) -> (f64, f32) {
    let mut delta = 0.0f64;
    let mut left = 0.0f64;
    let mut right = 0.0f64;
    let mut max_abs = 0.0f32;
    for (a, b) in reference.iter().zip(actual.iter()) {
        let abs = (a - b).abs();
        if abs > max_abs {
            max_abs = abs;
        }
        let a = f64::from(*a);
        let b = f64::from(*b);
        delta += (a - b) * (a - b);
        left += a * a;
        right += b * b;
    }
    let denominator = left.max(right).sqrt();
    let rel = if denominator == 0.0 {
        0.0
    } else {
        delta.sqrt() / denominator
    };
    (rel, max_abs)
}

fn seam_record(layer: usize, position: usize, reference: &[f32], actual: &[f32]) -> SeamRecord {
    let (rel, max_abs) = relative_l2(reference, actual);
    SeamRecord {
        layer,
        position,
        relative_l2: rel,
        max_abs,
        compared: reference.len().min(actual.len()),
    }
}

/// Reads one embedding row through the checkpoint range reader instead of materialising the table.
fn read_embedding_row(
    source: &SafeTensorsSource,
    config: &ModelConfig,
    token: u32,
) -> Result<Vec<f32>, String> {
    if token as usize >= config.vocab_size {
        return Err(format!("token {token} is outside the vocabulary"));
    }
    let header = source
        .header("embed.weight")
        .map_err(|error| error.to_string())?;
    let descriptor = header
        .tensor("embed.weight")
        .map_err(|error| error.to_string())?;
    if descriptor.shape.len() != 2 {
        return Err("embed.weight is not two-dimensional".to_owned());
    }
    let row_bytes = descriptor.byte_len() / descriptor.shape[0] as u64;
    let offset = u64::from(token) * row_bytes;
    let bytes = source
        .read_ranges("embed.weight", &[(offset, row_bytes as usize)])
        .map_err(|error| format!("embedding row {token}: {error}"))?;
    match descriptor.dtype.as_str() {
        "F32" => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| Ok(f32::from_le_bytes(*chunk)))
            .collect::<Result<Vec<f32>, String>>(),
        "BF16" => quant::decode_bf16(&bytes).map_err(|error| error.to_string()),
        "F8_E4M3" => {
            let mut values = quant::decode_f8e4m3(&bytes);
            // A dense FP8 row carries its E8M0 scales per 32 values; the released embedding is
            // BF16, so this arm only exists so a future checkpoint variant fails loud, not wrong.
            if values.len() != config.dim {
                values.truncate(config.dim);
            }
            Ok(values)
        }
        other => Err(format!("embedding row dtype {other} is not supported")),
    }
}

fn io_counters() -> (u64, u64) {
    let text = fs::read_to_string("/proc/self/io").unwrap_or_default();
    let mut rchar = 0u64;
    let mut read_bytes = 0u64;
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("rchar: ") {
            rchar = value.trim().parse().unwrap_or(0);
        } else if let Some(value) = line.strip_prefix("read_bytes: ") {
            read_bytes = value.trim().parse().unwrap_or(0);
        }
    }
    (rchar, read_bytes)
}

/// The per-rank resident estimate for Phase 1: routed slices plus every replicated tensor in full.
///
/// The two Engram tables are excluded: they are 94 GiB each in the released checkpoint and stay
/// on NVMe, read by row range per token, exactly as the design says (0 bytes resident).
fn resident_estimate(source: &SafeTensorsSource, plan: &ShardPlan) -> Result<u64, String> {
    let mut bytes = 0u64;
    for name in source.index().names() {
        if plan.rule(name).is_some() || is_engram_table(name) {
            continue;
        }
        let header = source.header(name).map_err(|error| error.to_string())?;
        let descriptor = header.tensor(name).map_err(|error| error.to_string())?;
        bytes = bytes.saturating_add(descriptor.byte_len());
    }
    for (name, rule) in plan.rules() {
        let header = source.header(name).map_err(|error| error.to_string())?;
        let descriptor = header.tensor(name).map_err(|error| error.to_string())?;
        let mut stored = descriptor.shape.clone();
        match rule {
            ShardRule::Replicated => {}
            ShardRule::Column => stored[0] /= plan.tp_world(),
            ShardRule::Row => stored[1] /= plan.tp_world(),
        }
        let logical = logical_from_stored(&descriptor.dtype, stored)?;
        let elements: usize = logical.1.iter().product();
        bytes = bytes.saturating_add(logical.0.storage_bytes(elements) as u64);
    }
    Ok(bytes)
}

fn is_engram_table(name: &str) -> bool {
    name.ends_with(".engram.embed.weight") || name.ends_with(".engram.embed.scale")
}

fn plan_hash(plan: &ShardPlan) -> u64 {
    let mut text = String::new();
    for (name, rule) in plan.rules() {
        text.push_str(name);
        text.push(':');
        text.push_str(match rule {
            ShardRule::Replicated => "rep",
            ShardRule::Column => "col",
            ShardRule::Row => "row",
        });
        text.push(';');
    }
    sparkflash_engine::synthetic::fnv1a(text.as_bytes())
}

/// Runs one rank of the host twin or the node run and returns its evidence.
/// Runs the bench loop: one prefill of the prompt, then N decode calls with no HTTP or token
/// streaming. The per-step cost comes from the same step path as the evidence loop, so the
/// --profile timers are exactly the ones a normal run records.
#[allow(clippy::too_many_arguments)]
fn run_bench_loop(
    rank: &mut Tp4Rank,
    input: &RankInput,
    step_tokens: &mut Vec<Option<u32>>,
    step_logits: &mut Vec<Vec<f32>>,
    step_ms: &mut Vec<u64>,
    step_rchar: &mut Vec<u64>,
    step_read_bytes: &mut Vec<u64>,
    step_collective_ms: &mut Vec<u64>,
    generated: &mut Vec<u32>,
) -> Result<BenchReport, String> {
    rank.begin_request(Tp4Sampling::default())
        .map_err(|error| format!("bench begin_request: {error}"))?;
    let prefill_started = Instant::now();
    let last = rank
        .prefill(&input.prompt)
        .map_err(|error| format!("bench prefill: {error}"))?;
    let prefill_ms = prefill_started.elapsed().as_millis() as u64;
    let prefill_token = last.token;
    if let Some(token) = prefill_token {
        generated.push(token);
    }
    let mut next = prefill_token;
    for step in 0..input.bench_steps {
        let token = next.ok_or_else(|| format!("bench step {step} has no token to consume"))?;
        let position = rank.state.tokens().len();
        let (output, cost) = rank
            .step_with_cost(token, position)
            .map_err(|error| error.to_string())?;
        step_ms.push(cost.ms);
        step_rchar.push(cost.rchar);
        step_read_bytes.push(cost.read_bytes);
        step_collective_ms.push(cost.collective_ms);
        step_tokens.push(output.token);
        step_logits.push(output.logits);
        next = output.token;
        if let Some(token) = output.token {
            generated.push(token);
        }
    }
    let decode_ms = step_ms.iter().sum::<u64>();
    let mut sorted = step_ms.clone();
    sorted.sort_unstable();
    let steps_run = sorted.len();
    let median_step_ms = match steps_run {
        0 => 0.0,
        len if len.is_multiple_of(2) => (sorted[len / 2 - 1] + sorted[len / 2]) as f64 / 2.0,
        len => sorted[len / 2] as f64,
    };
    let mean_step_ms = if steps_run == 0 {
        0.0
    } else {
        decode_ms as f64 / steps_run as f64
    };
    let tokens_per_second = if decode_ms == 0 {
        0.0
    } else {
        steps_run as f64 * 1000.0 / decode_ms as f64
    };
    Ok(BenchReport {
        steps_requested: input.bench_steps,
        steps_run,
        prefill_tokens: input.prompt.len(),
        prefill_ms,
        prefill_token,
        step_ms: step_ms.clone(),
        step_rchar: step_rchar.clone(),
        step_read_bytes: step_read_bytes.clone(),
        step_collective_ms: step_collective_ms.clone(),
        decode_ms,
        mean_step_ms,
        median_step_ms,
        tokens_per_second,
        profile_on: rank.profiler.is_some(),
    })
}

fn run_rank(
    input: &RankInput,
    transport: Option<Box<dyn Transport>>,
) -> Result<RankOutcome, String> {
    let mut rank = Tp4Rank::open_input(input, transport, ELEMENT_LIMIT)?;
    let load_ms = rank.metrics.load_ms;
    let (rchar_before, read_before) = io_counters();
    let run_started = Instant::now();
    let total_steps = input.prompt.len() + input.max_tokens.saturating_sub(1);
    let mut step_tokens: Vec<Option<u32>> = Vec::with_capacity(total_steps);
    let mut step_logits: Vec<Vec<f32>> = Vec::with_capacity(total_steps);
    let mut step_ms: Vec<u64> = Vec::with_capacity(total_steps);
    let mut step_rchar: Vec<u64> = Vec::with_capacity(total_steps);
    let mut step_read_bytes: Vec<u64> = Vec::with_capacity(total_steps);
    let mut step_collective_ms: Vec<u64> = Vec::with_capacity(total_steps);
    let mut generated = Vec::with_capacity(input.max_tokens.max(input.bench_steps + 1));
    let mut bench: Option<BenchReport> = None;
    if input.bench_steps > 0 {
        bench = Some(run_bench_loop(
            &mut rank,
            input,
            &mut step_tokens,
            &mut step_logits,
            &mut step_ms,
            &mut step_rchar,
            &mut step_read_bytes,
            &mut step_collective_ms,
            &mut generated,
        )?);
    } else {
        let mut next_token: Option<u32> = None;
        for step in 0..total_steps {
            let token = if step < input.prompt.len() {
                input.prompt[step]
            } else {
                next_token.ok_or_else(|| format!("step {step} has no token to consume"))?
            };
            let (output, cost) = rank
                .step_with_cost(token, step)
                .map_err(|error| error.to_string())?;
            step_ms.push(cost.ms);
            step_rchar.push(cost.rchar);
            step_read_bytes.push(cost.read_bytes);
            step_collective_ms.push(cost.collective_ms);
            step_tokens.push(output.token);
            let output_token = output.token;
            step_logits.push(output.logits);
            next_token = output_token;
            if step + 1 >= input.prompt.len()
                && generated.len() < input.max_tokens
                && let Some(token) = output_token
            {
                generated.push(token);
            }
        }
    }
    let run_ms = run_started.elapsed().as_millis() as u64;
    let (rchar_after, read_after) = io_counters();
    let reference = if input.reference_check && input.rank == 0 {
        Some(compare_with_reference(
            &rank.engine,
            &rank.config,
            &input.prompt,
            &generated,
            &step_tokens,
            &step_logits,
            &rank.backend.seams,
        )?)
    } else {
        None
    };
    if let Some(check) = reference.as_ref()
        && !check.tokens_equal
    {
        return Err(format!(
            "rank {} token sequence differs from the reference: {:?} vs {:?}",
            input.rank,
            step_tokens
                .iter()
                .map(|token| token.map_or(-1, i64::from))
                .collect::<Vec<_>>(),
            check
                .reference_tokens
                .iter()
                .map(|token| token.map_or(-1, i64::from))
                .collect::<Vec<_>>()
        ));
    }
    let completion_text = match &input.tokenizer {
        Some(path) => {
            let encoder = sparkflash_api::encoding::load_hf_tokenizer(path)
                .map_err(|error| format!("tokenizer: {error}"))?;
            encoder.tokenizer().decode(&generated, false).ok()
        }
        None => None,
    };
    let (_, rss_peak_kib) = rss_kib();
    let cache = rank.cache.as_ref();
    let (cache_tensors, cache_plan_bytes, cache_replicated_bytes, cache_fill_ms) = match cache {
        Some(cache) => (
            cache.tensors.len(),
            cache.plan_bytes,
            cache.replicated_bytes,
            cache.fill_ms,
        ),
        None => (0, 0, 0, 0),
    };
    let outcome = RankOutcome {
        rank: rank.rank,
        world: rank.world,
        prompt: input.prompt.clone(),
        generated,
        step_tokens,
        completion_text,
        step_ms,
        step_rchar,
        step_read_bytes,
        step_collective_ms,
        reduce_ms_total: rank.collective.elapsed_ms(),
        total_ms: run_ms,
        load_ms,
        io_rchar: rchar_after.saturating_sub(rchar_before),
        io_read_bytes: read_after.saturating_sub(read_before),
        reduce_calls: rank.collective.calls.load(Ordering::Relaxed),
        reduce_elements: rank.collective.elements.load(Ordering::Relaxed),
        shard_read_calls: rank.weights.shard_read_calls(),
        shard_read_bytes: rank.weights.shard_read_bytes(),
        shard_tensors: rank.plan.len(),
        resident_bytes_estimate: rank.resident_bytes,
        cache_tensors,
        cache_bytes: rank.metrics.resident_cache_bytes,
        cache_plan_bytes,
        cache_replicated_bytes,
        cache_fill_ms,
        cache_hits: rank.weights.cache_hits(),
        rss_load_kib: rank.metrics.rss_load_kib,
        rss_peak_kib,
        plan_hash: plan_hash(&rank.plan),
        collective_label: rank.collective.label.clone(),
        seams: rank.backend.seams.clone(),
        reference,
        source_mode: rank.source_mode,
        profile: rank.profiler.take(),
        bench,
    };
    Ok(outcome)
}

/// The local configuration a rank's routed block runs: same model, quarter (or half) expert width.
fn local_config(config: &ModelConfig, world: usize) -> Result<ModelConfig, String> {
    if !config.moe_inter_dim.is_multiple_of(world) {
        return Err(format!(
            "moe_inter_dim {} does not divide by world {world}",
            config.moe_inter_dim
        ));
    }
    let mut local = config.clone();
    local.moe_inter_dim /= world;
    Ok(local)
}

/// Compares the rank's per-step logits and tokens with a fresh single-rank reference run.
fn compare_with_reference(
    engine: &InferenceEngine,
    config: &ModelConfig,
    prompt: &[u32],
    generated: &[u32],
    step_tokens: &[Option<u32>],
    step_logits: &[Vec<f32>],
    seams: &[SeamRecord],
) -> Result<ReferenceCheck, String> {
    let mut reference_tokens: Vec<Option<u32>> = Vec::with_capacity(step_tokens.len());
    let mut logits_rel_l2: Vec<f64> = Vec::with_capacity(step_logits.len());
    let mut all_tokens = prompt.to_vec();
    all_tokens.extend_from_slice(generated);
    for step in 0..step_tokens.len() {
        let prefix_end = (step + 1).min(all_tokens.len());
        let mut state = SequenceState::new(config);
        let output = engine
            .prefill(&mut state, &all_tokens[..prefix_end])
            .map_err(|error| format!("reference prefill {step}: {error}"))?;
        reference_tokens.push(output.next_token);
        if let Some(logits) = step_logits.get(step) {
            let (rel, _) = relative_l2(&output.logits, logits);
            logits_rel_l2.push(rel);
        }
    }
    let max_logits_rel_l2 = logits_rel_l2.iter().copied().fold(0.0f64, f64::max);
    let seam_max_rel_l2 = seams
        .iter()
        .map(|record| record.relative_l2)
        .fold(0.0f64, f64::max);
    let tokens_equal = step_tokens == reference_tokens.as_slice();
    Ok(ReferenceCheck {
        tokens_equal,
        reference_tokens,
        logits_rel_l2,
        max_logits_rel_l2,
        seam_max_rel_l2,
        seam_comparisons: seams.len(),
    })
}

fn write_rank_evidence(
    directory: &Path,
    outcome: &RankOutcome,
    wall_ms: u64,
) -> Result<(), String> {
    fs::create_dir_all(directory)
        .map_err(|error| format!("create {}: {error}", directory.display()))?;
    let path = directory.join(format!("tp4-host-r{}.json", outcome.rank));
    let text = serde_json::to_vec_pretty(&rank_evidence(outcome, wall_ms))
        .map_err(|error| error.to_string())?;
    fs::write(&path, text).map_err(|error| format!("write {}: {error}", path.display()))?;
    if !outcome.seams.is_empty() {
        let seam_path = directory.join(format!("tp4-host-seams-r{}.json", outcome.rank));
        let seams = json!({
            "schema": SCHEMA_VERSION,
            "rank": outcome.rank,
            "world": outcome.world,
            "records": outcome.seams.iter().map(|record| json!({
                "layer": record.layer,
                "position": record.position,
                "relative_l2": record.relative_l2,
                "max_abs": record.max_abs,
                "compared": record.compared,
            })).collect::<Vec<_>>(),
        });
        fs::write(
            &seam_path,
            serde_json::to_vec_pretty(&seams).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("write {}: {error}", seam_path.display()))?;
    }
    Ok(())
}

/// The bench section of the evidence: the natural metric in a file.
fn bench_json(report: &BenchReport, profile_entries: Option<usize>) -> Value {
    json!({
        "steps_requested": report.steps_requested,
        "steps_run": report.steps_run,
        "prefill_tokens": report.prefill_tokens,
        "prefill_ms": report.prefill_ms,
        "prefill_token": report.prefill_token,
        "decode_ms": report.decode_ms,
        "mean_step_ms": report.mean_step_ms,
        "median_step_ms": report.median_step_ms,
        "tokens_per_second": report.tokens_per_second,
        "step_ms": report.step_ms,
        "step_rchar": report.step_rchar,
        "step_read_bytes": report.step_read_bytes,
        "step_collective_ms": report.step_collective_ms,
        "profile_on": report.profile_on,
        "profile_entries": profile_entries,
        "profile_decode_first_index": profile_entries.map(|entries| entries.saturating_sub(report.steps_run)),
        "measure": "decode-only: prefill(ids) once, then --bench-steps decode() calls timed with the same step path and --profile timers; no HTTP server, no token streaming; tokens_per_second = steps_run * 1000 / decode_ms; with --profile, profile.steps holds the prefill entries first (one per prompt token on the token-loop path, one for the whole chunk on the chunked path) and then the steps_run decode entries, so decode per-class averages use the last steps_run entries",
    })
}

fn rank_evidence(outcome: &RankOutcome, wall_ms: u64) -> Value {
    let steps = outcome.step_ms.len().max(1) as u64;
    let per_layer_max: Vec<f64> = {
        let mut by_layer: BTreeMap<usize, f64> = BTreeMap::new();
        for record in &outcome.seams {
            let entry = by_layer.entry(record.layer).or_insert(0.0);
            if record.relative_l2 > *entry {
                *entry = record.relative_l2;
            }
        }
        by_layer.into_values().collect()
    };
    let tuning = allocator::tune();
    let mut value = json!({
        "stage": "tp4-host",
        "schema": SCHEMA_VERSION,
        "rank": outcome.rank,
        "world": outcome.world,
        "source_mode": outcome.source_mode,
        "prompt_tokens": outcome.prompt,
        "generated_tokens": outcome.generated,
        "completion_text": outcome.completion_text,
        "allocator_tuning": allocator_json(tuning),
        "bench": outcome
            .bench
            .as_ref()
            .map(|bench| bench_json(bench, outcome.profile.as_ref().map(|profile| profile.steps.len()))),
        "step_tokens": outcome.step_tokens,
        "tokens_per_second": tokens_per_second(outcome.total_ms, outcome.prompt.len() + outcome.generated.len()),
        "timing": {
            "wall_ms": wall_ms,
            "load_ms": outcome.load_ms,
            "run_ms": outcome.total_ms,
            "total_step_ms": outcome.step_ms.iter().sum::<u64>(),
            "mean_step_ms": outcome.step_ms.iter().sum::<u64>() / steps,
            "max_step_ms": outcome.step_ms.iter().copied().max().unwrap_or(0),
            "step_ms": outcome.step_ms,
        },
        "breakdown": {
            "step_ms_total": outcome.step_ms.iter().sum::<u64>(),
            "step_collective_ms_total": outcome.reduce_ms_total,
            "step_compute_ms_total": outcome.step_ms.iter().sum::<u64>().saturating_sub(outcome.reduce_ms_total),
            "mean_step_ms": outcome.step_ms.iter().sum::<u64>() / steps,
            "mean_collective_ms": outcome.reduce_ms_total / steps,
            "mean_compute_ms": outcome.step_ms.iter().sum::<u64>().saturating_sub(outcome.reduce_ms_total) / steps,
            "step_rchar": outcome.step_rchar,
            "step_read_bytes_disk": outcome.step_read_bytes,
            "step_collective_ms": outcome.step_collective_ms,
            "note": "collective is measured around the canonical reduce (bf16 pack + TCP all-reduce + widen); compute is the remainder, i.e. attention/head/mHC host matmuls plus weight decode. Read volume is the per-step rchar/read_bytes arrays, so a step that still costs seconds with rchar ~0 is compute-bound, not I/O-bound",
        },
        "bytes": {
            "rchar": outcome.io_rchar,
            "read_bytes_disk": outcome.io_read_bytes,
            "rchar_per_token": outcome.io_rchar / steps,
            "read_bytes_disk_per_token": outcome.io_read_bytes / steps,
            "note": "rchar counts bytes delivered to read() system calls (page-cache hits included); read_bytes counts bytes that came from the block device, so a warm page cache shows rchar > 0 and read_bytes near zero",
        },
        "resident_cache": {
            "mode": if outcome.cache_tensors > 0 { "resident" } else { "none" },
            "tensors": outcome.cache_tensors,
            "bytes": outcome.cache_bytes,
            "plan_slice_bytes": outcome.cache_plan_bytes,
            "replicated_bytes": outcome.cache_replicated_bytes,
            "fill_ms": outcome.cache_fill_ms,
            "cached_hits": outcome.cache_hits,
            "engram_tables_excluded": true,
            "note": "raw stored bytes of every plan slice and every non-Engram checkpoint tensor, read exactly once at startup; Engram tables stay on NVMe and are paged by row range",
        },
        "rss": {
            "load_kib": outcome.rss_load_kib,
            "peak_kib": outcome.rss_peak_kib,
            "load_gib": outcome.rss_load_kib as f64 / (1024.0 * 1024.0),
            "peak_gib": outcome.rss_peak_kib as f64 / (1024.0 * 1024.0),
        },
        "shard": {
            "plan_tensors": outcome.shard_tensors,
            "plan_hash": format!("{:#018x}", outcome.plan_hash),
            "plan_rule": "routed experts column/row; attention, embed, head, shared, router, mHC, norms, engram replicated (Phase 1)",
            "shard_read_calls": outcome.shard_read_calls,
            "shard_read_bytes": outcome.shard_read_bytes,
            "resident_bytes_estimate": outcome.resident_bytes_estimate,
            "engram_tables_excluded": true,
            "resident_gib_estimate": outcome.resident_bytes_estimate as f64 / (1u64 << 30) as f64,
            "resident_source": if outcome.cache_tensors > 0 { "in-process cache of every non-Engram tensor" } else { "header-only estimate; the page cache holds the bytes" },
        },
        "collective": {
            "label": outcome.collective_label,
            "reduces": outcome.reduce_calls,
            "elements": outcome.reduce_elements,
            "canonical_rule": "bf16 payloads, f32 ascending-rank sum, one bf16 RNE round; world 1 is a strict no-op",
        },
        "reference": outcome.reference.as_ref().map(|check| json!({
            "tokens_equal": check.tokens_equal,
            "reference_tokens": check.reference_tokens,
            "logits_rel_l2_per_step": check.logits_rel_l2,
            "max_logits_rel_l2": check.max_logits_rel_l2,
            "seam_max_rel_l2": check.seam_max_rel_l2,
            "seam_comparisons": check.seam_comparisons,
            "seam_max_rel_l2_per_layer": per_layer_max,
            "note": "tokens and logits against a fresh single-rank InferenceEngine::prefill over the same checkpoint; seams against the unsharded f32 routed block on the same input",
        })),
        "flags": {
            "device_execution": false,
            "token_produced": !outcome.generated.is_empty(),
            "gpu": false,
            "vllm": false,
            "node_side": outcome.source_mode == "checkpoint",
            "window": "none",
            "gpu_window": "not_used",
            "mode": "host_twin",
            "allocator_tuning": tuning.enabled,
        },
        "notes": [
            "Phase 1 shards only the routed experts; the replicated remainder is numerically a no-op but is not vLLM's TP=4 layout.",
            "One all-reduce per layer (40) plus no gather; the routed output is the only sharded seam.",
            "Tokens are greedy argmax over the replicated head, so every rank must agree without a token gather.",
            "allocator_tuning records the startup allocator tune from sparkflash_cli::allocator::tune; the pair of raised glibc thresholds is what turns the per-layer 134/168 MB buffers into retained heap, and either threshold alone is noise.",
        ],
    });
    if let Some(profile) = &outcome.profile {
        value["profile"] = profile_json(profile);
    }
    value
}

fn write_summary(directory: &Path, outcome: &RankOutcome, wall_ms: u64) -> Result<(), String> {
    let path = directory.join("tp4-host-summary.json");
    let tuning = allocator::tune();
    let values = json!({
        "stage": "tp4-host",
        "schema": SCHEMA_VERSION,
        "rank": outcome.rank,
        "world": outcome.world,
        "source_mode": outcome.source_mode,
        "generated_tokens": outcome.generated,
        "completion_text": outcome.completion_text,
        "wall_ms": wall_ms,
        "bench": outcome
            .bench
            .as_ref()
            .map(|bench| bench_json(bench, outcome.profile.as_ref().map(|profile| profile.steps.len()))),
        "tokens_per_second": tokens_per_second(outcome.total_ms, outcome.prompt.len() + outcome.generated.len()),
        "collective": outcome.collective_label,
        "plan_hash": format!("{:#018x}", outcome.plan_hash),
        "allocator_tuning": allocator_json(tuning),
        "profile_steps": outcome.profile.as_ref().map_or(0, |profile| profile.steps.len()),
        "resident_gib_estimate": outcome.resident_bytes_estimate as f64 / (1u64 << 30) as f64,
        "resident_cache_tensors": outcome.cache_tensors,
        "resident_cache_bytes": outcome.cache_bytes,
        "rss_load_gib": outcome.rss_load_kib as f64 / (1024.0 * 1024.0),
        "rss_peak_gib": outcome.rss_peak_kib as f64 / (1024.0 * 1024.0),
        "flags": {"device_execution": false, "gpu": false, "vllm": false, "node_side": outcome.source_mode == "checkpoint"},
        "note": "rank 0 summary; per-rank evidence lives in tp4-host-r<rank>.json and must be collected from every node by the launcher",
    });
    fs::write(
        &path,
        serde_json::to_vec_pretty(&values).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write {}: {error}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn free_base_port(world: u16) -> u16 {
        for attempt in 0..200u16 {
            let candidate = 21000 + attempt * 8;
            if candidate + world >= 65000 {
                break;
            }
            let mut listeners = Vec::new();
            let mut available = true;
            for offset in 0..world {
                match std::net::TcpListener::bind(("127.0.0.1", candidate + offset)) {
                    Ok(listener) => listeners.push(listener),
                    Err(_) => {
                        available = false;
                        break;
                    }
                }
            }
            drop(listeners);
            if available {
                return candidate;
            }
        }
        panic!("no free adjacent port range");
    }

    fn run_two_to_four_ranks(
        world: usize,
        prompt: Vec<u32>,
        max_tokens: usize,
    ) -> Vec<RankOutcome> {
        let directory = tempfile::tempdir().expect("tempdir");
        write_twin_fixture(directory.path()).expect("fixture");
        let base_port = free_base_port(world as u16);
        let outcomes: Vec<RankOutcome> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..world)
                .map(|rank| {
                    let directory = directory.path().to_path_buf();
                    let prompt = prompt.clone();
                    scope.spawn(move || {
                        let addresses: Vec<SocketAddr> = (0..world)
                            .map(|peer| {
                                SocketAddr::new(
                                    IpAddr::from([127, 0, 0, 1]),
                                    base_port + peer as u16,
                                )
                            })
                            .collect();
                        let config = TcpConfig::with_addresses(rank, addresses).expect("config");
                        let transport = TcpTransport::connect(config).expect("transport connect");
                        let input = RankInput {
                            rank,
                            world,
                            source_dir: directory,
                            source_mode: "fixture",
                            prompt: prompt.clone(),
                            max_tokens,
                            record_seams: rank == 0,
                            reference_check: rank == 0,
                            resident_cache: false,
                            profile: false,
                            bench_steps: 0,
                            tokenizer: None,
                        };
                        run_rank(&input, Some(Box::new(transport))).expect("rank")
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("rank thread"))
                .collect()
        });
        outcomes
    }

    #[test]
    fn the_twin_fixture_opens_and_the_reference_engine_prefills_it() {
        let directory = tempfile::tempdir().expect("tempdir");
        let summary = write_twin_fixture(directory.path()).expect("fixture");
        assert!(summary.tensors > 1000);
        assert!(summary.bytes > 0);
        let config = twin_config();
        let index = directory.path().join("model.safetensors.index.json");
        let source = SafeTensorsSource::open(directory.path(), &index).expect("source");
        let plan = TensorPlan::language_model(&config);
        assert!(plan.specs().len() > 1000);
        let engine = InferenceEngine::new(
            config.clone(),
            Box::new(SharedSource {
                inner: Arc::new(source),
                cache: None,
            }),
            ELEMENT_LIMIT,
        )
        .expect("engine");
        let mut state = SequenceState::new(&config);
        let output = engine.prefill(&mut state, &[3, 7, 11]).expect("prefill");
        assert!(output.next_token.is_some());
        assert!(output.logits.iter().all(|value| value.is_finite()));
        let second = write_twin_fixture(directory.path()).expect("rewrite");
        assert_eq!(
            summary.hash, second.hash,
            "the fixture writer is deterministic"
        );
    }

    #[test]
    fn the_twin_fixture_expert_plan_verifies_at_world_two_and_four() {
        let directory = tempfile::tempdir().expect("tempdir");
        write_twin_fixture(directory.path()).expect("fixture");
        let index = directory.path().join("model.safetensors.index.json");
        let source = SafeTensorsSource::open(directory.path(), &index).expect("source");
        let config = twin_config();
        for world in [2usize, 4] {
            let plan = expert_shard_plan(&source, 0, world).expect("plan");
            assert_eq!(plan.tp_world(), world);
            let weights = RankWeights::new(
                Arc::new(SafeTensorsSource::open(directory.path(), &index).expect("source")),
                plan,
                None,
            );
            let w1: TensorSpec = weights
                .spec("layers.0.ffn.experts.0.w1.weight")
                .expect("w1 spec");
            assert_eq!(w1.shape, vec![TWIN_MOE_INTER_DIM / world, config.dim]);
            let w2 = weights
                .spec("layers.0.ffn.experts.0.w2.weight")
                .expect("w2 spec");
            assert_eq!(w2.shape, vec![config.dim, TWIN_MOE_INTER_DIM / world]);
            let w2_bytes = weights
                .read("layers.0.ffn.experts.0.w2.weight")
                .expect("w2 bytes");
            assert_eq!(
                w2_bytes.len() as u128,
                w2.storage_bytes(),
                "local bytes must be the local logical storage"
            );
            assert!(
                weights
                    .spec("layers.0.attn_norm.weight")
                    .expect("norm")
                    .shape
                    .len()
                    == 1,
                "non-expert tensors stay replicated"
            );
        }
    }

    #[test]
    fn a_missing_shard_divisibility_is_refused_before_any_read() {
        let directory = tempfile::tempdir().expect("tempdir");
        write_twin_fixture(directory.path()).expect("fixture");
        let index = directory.path().join("model.safetensors.index.json");
        let source = SafeTensorsSource::open(directory.path(), &index).expect("source");
        let mut plan = ShardPlan::new(0, 3).expect("plan");
        plan.insert("layers.0.ffn.experts.0.w1.weight", ShardRule::Column);
        let error = plan
            .verify(&source)
            .expect_err("world 3 must not divide the 128-row expert");
        let text = error.to_string();
        assert!(text.contains("divide"), "unexpected error: {text}");
    }

    #[test]
    fn world_one_reduce_is_a_strict_bit_no_op() {
        let collective = TpCollective::new(1, None);
        let mut values = vec![1.0f32 / 3.0, -7.0e-8, 12345.678];
        let before: Vec<u32> = values.iter().map(|value| value.to_bits()).collect();
        collective.all_reduce_f32(&mut values).expect("reduce");
        let after: Vec<u32> = values.iter().map(|value| value.to_bits()).collect();
        assert_eq!(before, after);
        assert_eq!(collective.calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn the_embedded_cell_reads_one_row_without_materialising_the_table() {
        let directory = tempfile::tempdir().expect("tempdir");
        write_twin_fixture(directory.path()).expect("fixture");
        let index = directory.path().join("model.safetensors.index.json");
        let source = SafeTensorsSource::open(directory.path(), &index).expect("source");
        let config = twin_config();
        let row = read_embedding_row(&source, &config, 3).expect("row");
        let table = linalg::decode_by_name(&source, "embed.weight", ELEMENT_LIMIT).expect("table");
        assert_eq!(row, table[3 * config.dim..4 * config.dim]);
    }

    #[test]
    fn two_ranks_produce_the_reference_token_sequence() {
        let outcomes = run_two_to_four_ranks(2, vec![3, 7, 11], 2);
        assert_eq!(outcomes.len(), 2);
        assert_eq!(outcomes[0].generated, outcomes[1].generated);
        assert_eq!(outcomes[0].step_tokens, outcomes[1].step_tokens);
        let reference = outcomes[0].reference.as_ref().expect("rank 0 reference");
        assert!(
            reference.tokens_equal,
            "tokens: {:?}",
            reference.reference_tokens
        );
        assert!(
            reference.max_logits_rel_l2 <= 2.0e-2,
            "max logits rel L2 {} over 2e-2",
            reference.max_logits_rel_l2
        );
        for outcome in &outcomes {
            assert_eq!(outcome.reduce_calls, 40 * outcome.step_ms.len() as u64);
            assert_eq!(
                outcome.reduce_elements,
                40 * outcome.step_ms.len() as u64 * 64
            );
        }
    }

    #[test]
    fn four_ranks_produce_the_reference_token_sequence() {
        let outcomes = run_two_to_four_ranks(4, vec![3, 7], 2);
        assert_eq!(outcomes.len(), 4);
        for outcome in &outcomes[1..] {
            assert_eq!(outcome.step_tokens, outcomes[0].step_tokens);
        }
        let reference = outcomes[0].reference.as_ref().expect("rank 0 reference");
        assert!(
            reference.tokens_equal,
            "tokens: {:?}",
            reference.reference_tokens
        );
        assert!(
            reference.max_logits_rel_l2 <= 2.0e-2,
            "max logits rel L2 {} over 2e-2",
            reference.max_logits_rel_l2
        );
        assert!(reference.seam_comparisons >= 40);
    }

    #[test]
    fn the_resident_cache_holds_every_non_engram_tensor_once() {
        let directory = tempfile::tempdir().expect("tempdir");
        write_twin_fixture(directory.path()).expect("fixture");
        let index = directory.path().join("model.safetensors.index.json");
        let source = SafeTensorsSource::open(directory.path(), &index).expect("source");
        let plan = expert_shard_plan(&source, 0, 4).expect("plan");
        let plan_names: Vec<String> = plan.rules().keys().cloned().collect();
        let replicated_names: Vec<String> = source
            .index()
            .names()
            .filter(|name| plan.rule(name).is_none() && !is_engram_table(name))
            .map(str::to_owned)
            .collect();
        let cache = ResidentCache::load(&source, &plan).expect("cache");
        assert_eq!(
            cache.tensors.len(),
            plan_names.len() + replicated_names.len()
        );
        assert_eq!(cache.bytes(), cache.plan_bytes + cache.replicated_bytes);
        for name in plan_names.iter().take(3) {
            let direct = plan.read(&source, name).expect("direct").bytes;
            assert_eq!(cache.get(name).expect("cached"), direct);
        }
        assert!(
            cache.tensors.keys().all(|name| !is_engram_table(name)),
            "the Engram tables must stay on disk"
        );
        // A cache-backed rank view serves plan reads from memory: the shard reader is not touched
        // after the fill, and the hit counter records the read.
        let weights = RankWeights::new(Arc::new(source), plan, Some(Arc::new(cache)));
        let name = "layers.0.ffn.experts.0.w1.weight";
        let first = weights.read(name).expect("cached read");
        let second = weights.read(name).expect("cached read again");
        assert_eq!(first, second);
        assert_eq!(weights.shard_read_calls(), 0);
        assert_eq!(weights.shard_read_bytes(), 0);
        assert_eq!(weights.cache_hits(), 2);
    }

    #[test]
    fn the_rank_library_open_prefill_decode_surface_runs_a_greedy_request() {
        let directory = tempfile::tempdir().expect("tempdir");
        write_twin_fixture(directory.path()).expect("fixture");
        let input = RankInput {
            rank: 0,
            world: 1,
            source_dir: directory.path().to_path_buf(),
            source_mode: "fixture",
            prompt: Vec::new(),
            max_tokens: 0,
            record_seams: false,
            reference_check: false,
            resident_cache: false,
            profile: false,
            bench_steps: 0,
            tokenizer: None,
        };
        let mut rank = Tp4Rank::open_input(&input, None, ELEMENT_LIMIT).expect("rank");
        assert_eq!(rank.rank(), 0);
        assert_eq!(rank.world_size(), 1);
        assert_eq!(rank.metrics().steps, 0);
        assert_ne!(rank.metrics().plan_hash, 0);
        rank.begin_request(Tp4Sampling::default()).expect("greedy");
        let first = rank.prefill(&[3, 7, 11]).expect("prefill");
        let token = first.token.expect("prompt ends in a token");
        assert_eq!(
            rank.metrics().steps,
            3,
            "steps counts consumed positions: the default token-loop prefill consumes three"
        );
        assert_eq!(rank.state.tokens(), &[3, 7, 11]);
        assert!(rank.decode(token).expect("decode").token.is_some());
        assert_eq!(rank.metrics().steps, 4);
        assert!(rank.metrics().last_step_ms > 0);
        assert_eq!(rank.sampling(), Tp4Sampling::default());
        let refused = rank.begin_request(Tp4Sampling {
            temperature: 0.7,
            seed: 1,
        });
        assert!(refused.is_err(), "the host path must refuse sampling");
        rank.begin_request(Tp4Sampling::default())
            .expect("greedy again");
        assert_eq!(rank.metrics().steps, 0, "begin_request clears the counters");
        rank.prefill(&[5]).expect("single-token prefill");
    }

    /// Batch 17's bar: one chunked prefill must be bit-identical to the same prompt run as single
    /// token steps, on the fixture engine: the same logits bits and the same greedy token.
    #[test]
    fn the_chunked_prefill_is_bit_identical_to_single_token_steps() {
        let directory = tempfile::tempdir().expect("tempdir");
        write_twin_fixture(directory.path()).expect("fixture");
        let input = RankInput {
            rank: 0,
            world: 1,
            source_dir: directory.path().to_path_buf(),
            source_mode: "fixture",
            prompt: Vec::new(),
            max_tokens: 0,
            record_seams: false,
            reference_check: false,
            resident_cache: false,
            profile: false,
            bench_steps: 0,
            tokenizer: None,
        };
        let mut chunked = Tp4Rank::open_input(&input, None, ELEMENT_LIMIT).expect("chunked rank");
        let mut looped = Tp4Rank::open_input(&input, None, ELEMENT_LIMIT).expect("token-loop rank");
        let prompt = [3u32, 7, 11, 5, 13, 2, 9, 4];
        chunked
            .begin_request(Tp4Sampling::default())
            .expect("greedy");
        let chunked_out = chunked.prefill_chunk(&prompt).expect("chunked prefill");
        looped
            .begin_request(Tp4Sampling::default())
            .expect("greedy");
        let looped_out = looped
            .prefill_token_loop(&prompt)
            .expect("token-loop prefill");
        assert_eq!(chunked.metrics().steps, prompt.len() as u64);
        assert_eq!(looped.metrics().steps, prompt.len() as u64);
        assert_eq!(chunked.state.tokens(), looped.state.tokens());
        assert_eq!(
            chunked_out.token, looped_out.token,
            "the greedy token must agree"
        );
        assert_eq!(chunked_out.logits.len(), looped_out.logits.len());
        let differing = chunked_out
            .logits
            .iter()
            .zip(looped_out.logits.iter())
            .filter(|(chunked, looped)| chunked.to_bits() != looped.to_bits())
            .count();
        assert_eq!(
            differing,
            0,
            "chunked prefill logits differ in {differing} of {} values",
            chunked_out.logits.len()
        );
    }

    #[test]
    fn the_bench_flag_parses_and_is_off_by_default() {
        let off = HostArgs::parse(&["--fixture", "/tmp/twin", "--prompt-tokens", "3,7,11"])
            .expect("parse");
        assert_eq!(off.bench_steps, 0);
        let on = HostArgs::parse(&[
            "--fixture",
            "/tmp/twin",
            "--prompt-tokens",
            "3,7,11",
            "--bench-steps",
            "10",
        ])
        .expect("parse");
        assert_eq!(on.bench_steps, 10);
    }

    #[test]
    fn the_world_one_bench_mode_runs_prefill_then_decodes_and_records_tokens_per_second() {
        let directory = tempfile::tempdir().expect("tempdir");
        write_twin_fixture(directory.path()).expect("fixture");
        let input = RankInput {
            rank: 0,
            world: 1,
            source_dir: directory.path().to_path_buf(),
            source_mode: "fixture",
            prompt: vec![3, 7, 11],
            max_tokens: 0,
            record_seams: false,
            reference_check: false,
            resident_cache: false,
            profile: false,
            bench_steps: 4,
            tokenizer: None,
        };
        let outcome = run_rank(&input, None).expect("bench rank");
        assert_eq!(outcome.world, 1);
        assert_eq!(outcome.reduce_calls, 0, "world 1 is a strict no-op");
        assert_eq!(
            outcome.generated.len(),
            5,
            "prefill token plus four decodes"
        );
        let bench = outcome.bench.as_ref().expect("bench report");
        assert_eq!(bench.steps_requested, 4);
        assert_eq!(bench.steps_run, 4);
        assert_eq!(bench.step_ms.len(), 4);
        assert_eq!(outcome.step_ms.len(), 4);
        assert_eq!(bench.prefill_tokens, 3);
        assert!(bench.prefill_ms > 0);
        assert!(bench.mean_step_ms > 0.0);
        assert!(bench.median_step_ms > 0.0);
        let expected = 4.0 * 1000.0 / bench.decode_ms as f64;
        assert!((bench.tokens_per_second - expected).abs() < 1e-9);
        assert!(!bench.profile_on);
        let evidence = tempfile::tempdir().expect("evidence");
        write_rank_evidence(evidence.path(), &outcome, outcome.total_ms).expect("evidence");
        let text = fs::read_to_string(evidence.path().join("tp4-host-r0.json")).expect("read");
        let value: Value = serde_json::from_str(&text).expect("json");
        assert_eq!(value["bench"]["steps_run"], 4);
        assert!(
            value["bench"]["tokens_per_second"]
                .as_f64()
                .expect("tokens/s")
                > 0.0
        );
        assert!(value["bench"]["mean_step_ms"].as_f64().expect("mean") > 0.0);
    }
}

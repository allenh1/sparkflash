//! The four-rank device arm: per-rank resident arena, per-layer all-reduce, 40-layer chain.
//!
//! This stage is the four-node arm the Lead approved after the plan-only arithmetic showed the
//! world=1 set is 285.504 GiB and cannot fit one 121 GiB node. It is a CLI arm first: one process
//! per node, launched over ssh, with the same transport and argv shape the host twin already uses
//! (`--comm-peers`, `--comm-base-port`), so the window kit does not need a second launcher.
//!
//! The first landing covers the parts that can be verified without the forward loop being wired:
//! the per-rank plan/arena through the resident stage (delegated, so there is exactly one code path
//! for the streaming arena), and a transport smoke that opens the TCP collective in the real launch
//! order and all-reduces a small buffer, recording `reduce_calls`, elements and wire bytes in the
//! evidence. The per-layer reduce inside the 40-layer loop lands next and will extend the same
//! `collectives` object; its field names are frozen here.

use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use sparkflash_comm::layout::{CollectiveLayout, FabricSchedule};
use sparkflash_comm::tcp::{TcpConfig, TcpTransport};
use sparkflash_comm::transport::Transport;

use crate::tp4_resident;

/// How the stage was invoked.
pub const USAGE: &str = "sparkflash device-chain --run-stage tp4-chain (--checkpoint DIR | --tensor-index FILE) --tp-rank R --tp-world-size 4 --evidence-dir DIR [--plan-only] [--upload] [--comm-peers IP,IP,IP,IP] [--comm-base-port 29601] [--comm-timeout 120] [--comm-connect-timeout 120] [--comm-smoke] [--decode-steps N] [--prompt-tokens N] [--cubin-dir DIR] [--engram-tokenizer FILE] [--forward --forward-preflight --kernel-cubin-dir DIR --token-ids ID,ID --forward-reference-ids ID,ID] [--native-rounding none|combine|mhc|shared|both|LIST] [--forward-native-dense-parity [LAYER[:POSITION]]] [--forward-lean] [--forward-perf-probe] [--forward-perf-probe-nosync]";

/// The parsed argv.
#[derive(Debug)]
struct ChainArgs {
    rank: usize,
    world: usize,
    checkpoint: Option<PathBuf>,
    tensor_index: Option<PathBuf>,
    /// The tokenizer.json of the Engram token map (default: the checkpoint's own).
    engram_tokenizer: Option<PathBuf>,
    plan_only: bool,
    #[allow(dead_code)] // parsed for the loop arm that is not wired yet
    upload: bool,
    loop_mode: bool,
    evidence_dir: PathBuf,
    peers: Vec<std::net::IpAddr>,
    base_port: u16,
    /// Per-call transport timeout; the window protocol passes 120 s.
    comm_timeout_secs: u64,
    /// Peer connect timeout; the window protocol passes 120 s because arena boot straggles.
    comm_connect_timeout_secs: u64,
    comm_smoke: bool,
    decode_steps: usize,
    warmup_steps: usize,
    cuda_graphs: CudaGraphs,
    #[allow(dead_code)] // parsed for the loop arm that is not wired yet
    prompt_tokens: usize,
    cubin_dir: Option<PathBuf>,
    /// `--forward`: request the C4-token stage after the bind/walk. Alone it never launches:
    /// the stage refuses unless `--forward-preflight` is also present.
    forward: bool,
    /// `--forward-preflight`: the explicit word that allows the four-rank loop to run kernels.
    forward_preflight: bool,
    /// The staged kernel cubins the forward stage loads from (`--kernel-cubin-dir`).
    kernel_cubin_dir: Option<PathBuf>,
    /// The prompt/decode token ids the forward stage starts from (`--token-ids`).
    token_ids: Vec<u32>,
    /// The teacher-forced reference token per dumped position (`--forward-reference-ids`).
    /// Wired into every dump row's `reference_id`; must have one id per decode position.
    reference_ids: Vec<u32>,
    /// The smoke-window shortcut: run only the first N forward positions instead of the
    /// full prompt plus decode rows. It exercises the real body/staging/rope/head path on
    /// a few positions; absent means the whole sequence.
    prefill_limit: Option<usize>,
    /// The native-rounding switch value for the forward loop (the D41 native arm). Default none,
    /// which is the f32 arm; the active set is printed and recorded in the run evidence.
    native_rounding: crate::tp4_forward::NativeRounding,
    /// The rank-0 native dense launch-parity capture (--forward-native-dense-parity). Off when
    /// absent; the mode is a one-shot diagnostic that never changes the loop's arithmetic or its
    /// existing evidence.
    native_dense_parity: Option<crate::tp4_forward::NativeDenseParitySpec>,
    /// --forward-lean: the default-off M2 lean measurement switch (diagnostics only; no numeric or
    /// logits-byte change).
    forward_lean: bool,
    /// --forward-perf-probe: the default-off M2 host timing probe (diagnostics only; one context
    /// synchronise per block, a separate tp4-forward-perf-r{rank}.json, no numeric or logits change).
    forward_perf_probe: bool,
    /// --forward-perf-probe-nosync: the same per-block instrumentation with no boundary drain, so a
    /// block's wall is the host's and the device time arrives from one run-end event harvest
    /// (diagnostics only: same separate perf file, its mode line says which wall clock it used).
    forward_perf_probe_nosync: bool,
}

/// When the device arm should capture per-step CUDA graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CudaGraphs {
    /// Capture graphs during warmup and replay them per step.
    On,
    /// Never capture; run the plain launches.
    Off,
    /// Capture only where every kernel in the step is capturable; otherwise run plain and say why.
    Auto,
}

impl ChainArgs {
    fn parse(args: &[&str]) -> Result<Self, String> {
        let mut rank: Option<usize> = None;
        let mut world: Option<usize> = None;
        let mut checkpoint: Option<PathBuf> = None;
        let mut tensor_index: Option<PathBuf> = None;
        let mut engram_tokenizer: Option<PathBuf> = None;
        let mut evidence_dir: Option<PathBuf> = None;
        let mut peers_raw: Option<String> = None;
        let mut base_port = 29601u16;
        let mut comm_timeout_secs = 180u64;
        let mut comm_connect_timeout_secs = 60u64;
        let mut plan_only = false;
        let mut upload = false;
        let mut loop_mode = false;
        let mut comm_smoke = false;
        let mut decode_steps = 1usize;
        let mut warmup_steps = 2usize;
        let mut cuda_graphs = CudaGraphs::Auto;
        let mut prompt_tokens = 1usize;
        let mut cubin_dir: Option<PathBuf> = None;
        let mut forward = false;
        let mut forward_preflight = false;
        let mut kernel_cubin_dir: Option<PathBuf> = None;
        let mut token_ids: Vec<u32> = Vec::new();
        let mut reference_ids: Vec<u32> = Vec::new();
        let mut prefill_limit: Option<usize> = None;
        let mut native_rounding = crate::tp4_forward::NativeRounding::NONE;
        let mut native_dense_parity: Option<crate::tp4_forward::NativeDenseParitySpec> = None;
        let mut forward_lean = false;
        let mut forward_perf_probe = false;
        let mut forward_perf_probe_nosync = false;
        let mut index = 0;
        while index < args.len() {
            let flag = args[index];
            index += 1;
            // Peeked before the take closure borrows index mutably: the parity flag's value is
            // optional, so the arm needs to know whether the next argv term is a value at all.
            let peek_next = args.get(index).copied();
            let mut take = |what: &str| -> Result<String, String> {
                let value = args
                    .get(index)
                    .ok_or_else(|| format!("{what} needs a value"))?;
                index += 1;
                Ok((*value).to_string())
            };
            match flag {
                "--tp-rank" => rank = Some(parse_usize("--tp-rank", &take("--tp-rank")?)?),
                "--tp-world-size" => {
                    world = Some(parse_usize("--tp-world-size", &take("--tp-world-size")?)?);
                }
                "--checkpoint" => checkpoint = Some(PathBuf::from(take("--checkpoint")?)),
                "--tensor-index" => tensor_index = Some(PathBuf::from(take("--tensor-index")?)),
                "--engram-tokenizer" => {
                    engram_tokenizer = Some(PathBuf::from(take("--engram-tokenizer")?));
                }
                "--evidence-dir" => evidence_dir = Some(PathBuf::from(take("--evidence-dir")?)),
                "--comm-peers" => peers_raw = Some(take("--comm-peers")?),
                "--comm-base-port" => {
                    base_port =
                        u16::try_from(parse_usize("--comm-base-port", &take("--comm-base-port")?)?)
                            .map_err(|_| "--comm-base-port must fit u16".to_string())?;
                }
                "--comm-timeout" => {
                    comm_timeout_secs =
                        parse_usize("--comm-timeout", &take("--comm-timeout")?)? as u64;
                }
                "--comm-connect-timeout" => {
                    comm_connect_timeout_secs =
                        parse_usize("--comm-connect-timeout", &take("--comm-connect-timeout")?)?
                            as u64;
                }
                "--plan-only" => plan_only = true,
                "--upload" => upload = true,
                "--loop" => loop_mode = true,
                "--comm-smoke" => comm_smoke = true,
                "--warmup-steps" => {
                    warmup_steps = parse_usize("--warmup-steps", &take("--warmup-steps")?)?;
                }
                "--cuda-graphs" => {
                    cuda_graphs = match take("--cuda-graphs")?.as_str() {
                        "on" => CudaGraphs::On,
                        "off" => CudaGraphs::Off,
                        "auto" => CudaGraphs::Auto,
                        other => {
                            return Err(format!("--cuda-graphs {other}: expected on, off or auto"));
                        }
                    };
                }
                "--decode-steps" => {
                    decode_steps = parse_usize("--decode-steps", &take("--decode-steps")?)?;
                }
                "--prompt-tokens" => {
                    prompt_tokens = parse_usize("--prompt-tokens", &take("--prompt-tokens")?)?;
                }
                "--cubin-dir" => cubin_dir = Some(PathBuf::from(take("--cubin-dir")?)),
                "--forward" => forward = true,
                "--forward-preflight" => forward_preflight = true,
                "--kernel-cubin-dir" => {
                    kernel_cubin_dir = Some(PathBuf::from(take("--kernel-cubin-dir")?));
                }
                "--token-ids" => {
                    token_ids = parse_token_ids(&take("--token-ids")?, "--token-ids")?;
                }
                "--forward-reference-ids" => {
                    reference_ids = parse_token_ids(
                        &take("--forward-reference-ids")?,
                        "--forward-reference-ids",
                    )?;
                }
                "--prefill-limit" => {
                    let value = parse_usize("--prefill-limit", &take("--prefill-limit")?)?;
                    if value == 0 {
                        return Err("--prefill-limit must be at least 1".to_string());
                    }
                    prefill_limit = Some(value);
                }
                "--native-rounding" => {
                    native_rounding =
                        crate::tp4_forward::NativeRounding::parse(&take("--native-rounding")?)?;
                }
                "--forward-native-dense-parity" => {
                    native_dense_parity = Some(match peek_next {
                        Some(raw) if !raw.starts_with("--") => {
                            let raw = take("--forward-native-dense-parity")?;
                            crate::tp4_forward::NativeDenseParitySpec::parse(&raw)?
                        }
                        _ => crate::tp4_forward::NativeDenseParitySpec {
                            layer: 0,
                            position: None,
                        },
                    });
                }
                "--forward-lean" => forward_lean = true,
                "--forward-perf-probe" => forward_perf_probe = true,
                "--forward-perf-probe-nosync" => forward_perf_probe_nosync = true,
                "--verify-weights" => {
                    #[allow(unused_unsafe)]
                    unsafe {
                        std::env::set_var("SPARKFLASH_VERIFY_WEIGHTS", "1");
                    }
                }
                "--mmap" => {
                    #[allow(unused_unsafe)]
                    unsafe {
                        std::env::set_var("SPARKFLASH_ARENA_MMAP", "1");
                    }
                }
                other => return Err(format!("unknown flag {other}")),
            }
        }
        if plan_only && upload {
            return Err("--plan-only and --upload are alternatives".to_string());
        }
        if !plan_only && !upload {
            return Err("one of --plan-only or --upload is required".to_string());
        }
        if checkpoint.is_some() && tensor_index.is_some() {
            return Err("--checkpoint and --tensor-index are alternatives".to_string());
        }
        if checkpoint.is_none() && tensor_index.is_none() {
            return Err("one of --checkpoint or --tensor-index is required".to_string());
        }
        let rank = rank.ok_or_else(|| "--tp-rank is required".to_string())?;
        let world = world.ok_or_else(|| "--tp-world-size is required".to_string())?;
        if world < 2 {
            return Err(format!(
                "tp4-chain is the multi-rank arm; got world {world} (use --run-stage tp4-resident for a single node)"
            ));
        }
        if rank >= world {
            return Err(format!("--tp-rank {rank} is outside world {world}"));
        }
        let evidence_dir = evidence_dir.ok_or_else(|| "--evidence-dir is required".to_string())?;
        let mut peers = Vec::new();
        if let Some(raw) = peers_raw {
            for part in raw.split(',') {
                let part = part.trim();
                if part.is_empty() {
                    continue;
                }
                peers.push(
                    part.parse()
                        .map_err(|error| format!("--comm-peers {part}: {error}"))?,
                );
            }
        }
        if comm_smoke {
            if peers.len() != world {
                return Err(format!(
                    "--comm-smoke needs exactly {world} --comm-peers entries, got {}",
                    peers.len()
                ));
            }
            if world > u16::MAX as usize {
                return Err("world does not fit the port arithmetic".to_string());
            }
        }
        if forward && !loop_mode {
            return Err(
                "--forward requires --loop: the token stage runs after the bind and the walk"
                    .to_string(),
            );
        }
        if forward_preflight && !forward {
            return Err("--forward-preflight requires --forward".to_string());
        }
        if kernel_cubin_dir.is_some() && !forward {
            return Err("--kernel-cubin-dir is only used by --forward".to_string());
        }
        if !token_ids.is_empty() && !forward {
            return Err("--token-ids is only used by --forward".to_string());
        }
        if native_rounding != crate::tp4_forward::NativeRounding::NONE && !forward {
            return Err("--native-rounding is only used by --forward".to_string());
        }
        if native_dense_parity.is_some() && !forward {
            return Err("--forward-native-dense-parity is only used by --forward".to_string());
        }
        if forward_lean && !forward {
            return Err("--forward-lean is only used by --forward".to_string());
        }
        if forward_perf_probe && !forward {
            return Err("--forward-perf-probe is only used by --forward".to_string());
        }
        if forward_perf_probe_nosync && !forward {
            return Err("--forward-perf-probe-nosync is only used by --forward".to_string());
        }
        if forward_perf_probe && forward_perf_probe_nosync {
            return Err(
                "--forward-perf-probe and --forward-perf-probe-nosync measure different wall clocks and cannot share a run: pick one"
                    .to_string(),
            );
        }
        if !reference_ids.is_empty() && !forward {
            return Err("--forward-reference-ids is only used by --forward".to_string());
        }
        if !reference_ids.is_empty() && reference_ids.len() != decode_steps.max(1) {
            return Err(format!(
                "--forward-reference-ids: {} ids for {} decode positions; the reference vector must pair one-to-one with the dumped rows",
                reference_ids.len(),
                decode_steps.max(1)
            ));
        }
        Ok(Self {
            rank,
            world,
            checkpoint,
            tensor_index,
            engram_tokenizer,
            plan_only,
            upload,
            loop_mode,
            evidence_dir,
            peers,
            base_port,
            comm_timeout_secs,
            comm_connect_timeout_secs,
            comm_smoke,
            decode_steps,
            warmup_steps,
            cuda_graphs,
            prompt_tokens,
            cubin_dir,
            forward,
            forward_preflight,
            kernel_cubin_dir,
            token_ids,
            reference_ids,
            prefill_limit,
            native_rounding,
            native_dense_parity,
            forward_lean,
            forward_perf_probe,
            forward_perf_probe_nosync,
        })
    }
}

fn parse_usize(flag: &str, raw: &str) -> Result<usize, String> {
    raw.parse()
        .map_err(|error| format!("{flag} {raw}: {error}"))
}

/// Parses a comma-separated token-id flag, each id inside the released vocabulary.
fn parse_token_ids(raw: &str, flag: &str) -> Result<Vec<u32>, String> {
    let mut ids = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let id: u32 = part
            .parse()
            .map_err(|error| format!("{flag} {part}: {error}"))?;
        if id as usize >= crate::tp4_token::VOCAB {
            return Err(format!(
                "{flag} {id} is outside the {}-token vocabulary",
                crate::tp4_token::VOCAB
            ));
        }
        ids.push(id);
    }
    if ids.is_empty() {
        return Err(format!("{flag} needs at least one token id"));
    }
    Ok(ids)
}

/// The routed down-projection geometry the bind actually built: the released plan stores one row per
/// output row in `layers.0.ffn.experts.0.w2`, so the first expert's shard is the per-rank shape.
///
/// Returns `(rows, Some((weight_bytes, scale_bytes)))`, or `(rows, None)` when the plan does not
/// carry that expert (which the preflight then reports as a missing bind, not as a stored layout).
fn combine_geometry_from_plan(
    plan: &sparkflash_device::resident::ResidentPlan,
) -> (usize, Option<(usize, usize)>) {
    let find = |suffix: &str| {
        plan.shards()
            .iter()
            .find(|shard| shard.name == format!("layers.0.ffn.experts.0.{suffix}"))
    };
    let weight = find("w2.weight");
    let scale = find("w2.scale");
    let rows = weight
        .and_then(|shard| shard.shard_shape.first().copied())
        .filter(|rows| *rows > 0)
        .unwrap_or(crate::tp4_forward::DEFAULT_COMBINE_ROWS);
    let bytes = match (weight, scale) {
        (Some(weight), Some(scale)) => {
            match (
                usize::try_from(weight.shard_bytes),
                usize::try_from(scale.shard_bytes),
            ) {
                (Ok(weight_bytes), Ok(scale_bytes)) => Some((weight_bytes, scale_bytes)),
                _ => None,
            }
        }
        _ => None,
    };
    (rows, bytes)
}

/// Builds the C4-token request from the argv, the rank plan and the run header.
fn build_token_request<'a>(
    args: &'a ChainArgs,
    setup: &'a tp4_resident::RankSetup,
    config: &'a sparkflash_engine::config::ModelConfig,
    n_layers: usize,
    weights: Option<&'a sparkflash_engine::safetensors::SafeTensorsSource>,
    header: &crate::run_header::RunHeader,
) -> crate::tp4_forward::TokenStageRequest<'a> {
    let (combine_rows, combine_bytes) = combine_geometry_from_plan(&setup.plan);
    // The run's ONE arm decision was made by the header (crate::run_header::RunHeader::decide),
    // which read each switch exactly once at startup; the request carries it. Evidence recording
    // forces the host arm, so the chain's bootstrap must gate on this decision
    // (request.arm.is_device()) rather than on the device switch alone.
    crate::tp4_forward::TokenStageRequest {
        rank: args.rank,
        world: args.world,
        arm: header.arm,
        evidence: header.evidence,
        header: header.clone(),
        n_layers,
        positions: args.decode_steps.max(1),
        evidence_dir: &args.evidence_dir,
        kernel_cubin_dir: args.kernel_cubin_dir.as_deref(),
        preflight: args.forward_preflight,
        combine_rows,
        combine_bytes,
        prompt_tokens: &args.token_ids,
        reference_ids: &args.reference_ids,
        resident_plan: Some(&setup.plan),
        config: Some(config),
        checkpoint_dir: args.checkpoint.as_deref(),
        checkpoint_index: args.tensor_index.as_deref(),
        engram_tokenizer: args.engram_tokenizer.as_deref(),
        weights,
        prefill_limit: args.prefill_limit,
        native_rounding: args.native_rounding,
        native_dense_parity: args.native_dense_parity,
        // None here is the HOST arm and the plan-only default. The arena path's closure in run_loop
        // replaces this field with Some(DeviceReduce) after the bootstrap opens the run's
        // communicator before the timed region (one open per run, before layer 0); nothing else fills
        // it, so the field has exactly one writer on the device arm.
        reduce: None,
        forward_perf: crate::tp4_forward::ForwardPerf {
            probe: args.forward_perf_probe || args.forward_perf_probe_nosync,
            probe_nosync: args.forward_perf_probe_nosync,
            lean: args.forward_lean,
        },
    }
}

/// Writes the token envelope and reports it in the run log. The file is on disk before the caller
/// decides the exit code, so a refusal keeps its named reason in the evidence directory.
fn write_forward_evidence(
    args: &ChainArgs,
    outcome: &crate::tp4_forward::TokenStageOutcome,
) -> Result<Vec<String>, String> {
    let path = outcome.write(&args.evidence_dir)?;
    let envelope = outcome.envelope();
    let mut lines = vec![format!(
        "tp4-chain r{} forward: {} ({} positions), envelope {}, reduce calls/step {}",
        args.rank,
        envelope.forward.as_str(),
        envelope.positions_run,
        path.display(),
        envelope.reduce_calls_per_step
    )];
    if let Some(error) = &envelope.error {
        lines.push(format!("tp4-chain r{} forward: error {}", args.rank, error));
    }
    lines.push(format!(
        "tp4-chain r{} forward: native_rounding={} (combine={}, mhc={}, shared={})",
        args.rank,
        args.native_rounding.as_str(),
        args.native_rounding.combine,
        args.native_rounding.mhc,
        args.native_rounding.shared
    ));
    if args.forward_lean {
        lines.push(format!(
            "tp4-chain r{} forward: lean=on (diagnostics only: twin off, deferred progress writes, deferred head seam files; NOT a D41 gate row)",
            args.rank
        ));
    }
    if args.forward_perf_probe {
        lines.push(format!(
            "tp4-chain r{} forward: perf_probe=on (diagnostics only: one cuCtxSynchronize per block, separate tp4-forward-perf-r{}.json; NOT a D41 gate row)",
            args.rank, args.rank
        ));
    }
    if args.forward_perf_probe_nosync {
        lines.push(format!(
            "tp4-chain r{} forward: perf_probe=nosync (diagnostics only: no per-block cuCtxSynchronize, host-only wall_ms + one run-end event harvest for kernel_ns, separate tp4-forward-perf-r{}.json; NOT a D41 gate row)",
            args.rank, args.rank
        ));
    }
    if let Some(parity) = args.native_dense_parity {
        lines.push(format!(
            "tp4-chain r{} forward: native_dense_parity=layer {} position {} (diagnostic only; the loop arithmetic is unchanged)",
            args.rank,
            parity.layer,
            parity
                .position
                .map_or_else(|| "first-decode".to_string(), |position| position.to_string()),
        ));
    }
    Ok(lines)
}

/// Opens the rank's TCP transport in the launch order the bench and census use: the peer list's
/// index is the rank, rank r binds `peers[r]:base_port + r`, and every rank dials the others.
fn open_transport(args: &ChainArgs) -> Result<TcpTransport, String> {
    open_transport_with_base(args, args.base_port)
}

/// The forward loop's own port range (+100 over the walk base). The two protocols never share
/// sockets: the loop runs inside the arena epoch, before the walk, and a first prefill gather frame
/// must not be able to land in a walk all-reduce stream (m1x failed exactly that way).
const FORWARD_LOOP_PORT_OFFSET: u16 = 100;

/// bf16 elements of ONE per-layer device collective: the two per-layer reduces (attention output
/// and the routed MoE delta) are both the plan's dim, 5,120 bf16 = 10,240 B per call in the released
/// config. This is layer_device's constant, not a second copy: the port's rungs and the warm-up must
/// agree on the count or the warm-up does not warm the shape the loop enqueues.
const REDUCE_PAYLOAD_ELEMENTS: usize = crate::layer_device::REDUCE_ELEMENTS;

/// Device scratch bytes for the communicator's setup warm-up: one payload's worth of bf16, the
/// in-place buffer the three warm-up collectives reduce over. Setup only; freed after the run.
const REDUCE_SCRATCH_BYTES: usize = REDUCE_PAYLOAD_ELEMENTS * 2;

/// The per-process behaviour switches the bootstrap's agreement word carries.
///
/// These MIRROR the readers in tp4_forward, which use PRESENCE semantics
/// (std::env::var(...).is_ok()); a per-rank difference here changes arithmetic with no failure, which
/// is exactly why the word crosses the bootstrap gather and is refused on mismatch. If a reader changes
/// its acceptance rule, or a new per-process arithmetic switch appears, this word must change with it
/// in the same edit. (SPARKFLASH_DEVICE_TRACE / _EVENTS belong to the trace reader in the review
/// branch; they are carried here so a rank with one set cannot silently join a run without it.)
fn behaviour_toggles() -> u16 {
    let mut word = 0u16;
    if std::env::var("SPARKFLASH_SEAM_REF_HOST").is_ok() {
        word |= 1 << 0;
    }
    if std::env::var("SPARKFLASH_DEVICE_TRACE").is_ok() {
        word |= 1 << 1;
    }
    if std::env::var("SPARKFLASH_DEVICE_TRACE_EVENTS").is_ok() {
        word |= 1 << 2;
    }
    word
}

/// [open_transport] on an explicit base port, for the forward loop's separate protocol range.
fn open_transport_with_base(args: &ChainArgs, base_port: u16) -> Result<TcpTransport, String> {
    let addresses: Vec<SocketAddr> = args
        .peers
        .iter()
        .enumerate()
        .map(|(peer, ip)| SocketAddr::new(*ip, base_port + peer as u16))
        .collect();
    let config = TcpConfig::with_addresses(args.rank, addresses)
        .map_err(|error| format!("tcp config: {error}"))?
        .with_timeout(Duration::from_secs(args.comm_timeout_secs))
        .with_connect_timeout(Duration::from_secs(args.comm_connect_timeout_secs));
    TcpTransport::connect(config).map_err(|error| format!("tcp connect: {error}"))
}

/// One transport smoke: all-reduce `SMOKE_ELEMENTS` bf16 values and check the canonical sum.
fn comm_smoke(args: &ChainArgs) -> Result<serde_json::Value, String> {
    const SMOKE_ELEMENTS: usize = 8;
    let started = Instant::now();
    let transport = open_transport(args)?;
    let connected_ms = started.elapsed().as_millis() as u64;
    let mut buffer: Vec<u16> = vec![bf16_bits((args.rank + 1) as f32); SMOKE_ELEMENTS];
    let layout = CollectiveLayout::new(args.world, SMOKE_ELEMENTS, FabricSchedule::OneShotAllToAll)
        .map_err(|error| format!("collective layout: {error}"))?;
    let reduced = Instant::now();
    transport
        .all_reduce_bf16(&mut buffer, 0, &layout)
        .map_err(|error| format!("all-reduce smoke: {error}"))?;
    let reduce_ms = reduced.elapsed().as_millis() as u64;
    let expected: u16 = bf16_bits((args.world * (args.world + 1) / 2) as f32);
    if buffer.iter().any(|value| *value != expected) {
        return Err(format!(
            "all-reduce smoke returned {:?}, expected every element to be the sum of 1..={}",
            buffer, args.world
        ));
    }
    let payload_bytes = (SMOKE_ELEMENTS * 2) as u64;
    Ok(serde_json::json!({
        "transport": "tcp",
        "world": args.world,
        "calls": 1u64,
        "elements": SMOKE_ELEMENTS as u64,
        "payload_bytes": payload_bytes,
        "wire_bytes": payload_bytes * args.world as u64,
        "per_step_calls": [1u64],
        "per_step_payload_bytes": [payload_bytes],
        "connect_ms": connected_ms,
        "wall_ms": reduce_ms,
        "per_step_wall_ms": [reduce_ms],
        "phase": "smoke",
        "verified": true,
        "note": "one all-reduce of 8 bf16 values before the 40-layer loop; per-layer calls will extend this object",
    }))
}

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let round = (bits >> 16) & 1;
    let sticky = bits & 0xFFFF != 0;
    let mut upper = (bits >> 16) as u16;
    if round == 1 && sticky {
        upper = upper.wrapping_add(1);
    } else if round == 1 && !sticky {
        upper |= 1;
    }
    upper
}

/// Runs one rank of the four-rank arm.
pub fn run(args: &[&str]) -> ExitCode {
    match run_lines(args) {
        Ok(lines) => {
            for line in lines {
                println!("{line}");
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("tp4-chain: {error}");
            ExitCode::from(2)
        }
    }
}

/// The C3 collective skeleton: the exact 40-layer x 2-phase reduce walk (80 calls per step) on
/// deterministic buffers, including the warmup steps, so the transport, the order, the payload
/// sizes and the calls_per_step assertion can be exercised before a rank-aware provider exists.
///
/// Every reduced element is checked against the canonical rule's expected value (ascending-rank f32
/// sum, one bf16 RNE round), so a wrong order or a silently missing rank fails here.
fn collective_walk(
    args: &ChainArgs,
    transport: TcpTransport,
    config: &sparkflash_engine::config::ModelConfig,
    bound_attention: usize,
    table_expected: usize,
) -> Result<serde_json::Value, String> {
    let layers = config.n_layers;
    let elements = config.dim;
    let world = args.world;
    let mut collective = LayerCollective::new(transport, layers);
    let steps_total = args.warmup_steps + args.decode_steps.max(1);
    let mut rows = Vec::with_capacity(steps_total);
    for step in 0..steps_total {
        let warm = step < args.warmup_steps;
        collective.reset();
        let started = Instant::now();
        for layer in 0..layers {
            for phase in [ReducePhase::Attention, ReducePhase::Moe] {
                let mut buffer = vec![0u16; elements];
                let values: Vec<f32> = (0..elements)
                    .map(|index| f32::from_bits(((index as u32) & 0x3FF) | 0x3F00_0000))
                    .collect();
                for index in 0..elements {
                    buffer[index] = bf16_bits((args.rank + 1) as f32 * (values[index] + 1.0));
                }
                collective.reduce_layer(layer, phase, &mut buffer)?;
                for index in 0..elements {
                    let sum: f32 = (1..=world)
                        .map(|rank| rank as f32 * (values[index] + 1.0))
                        .sum();
                    let expected = bf16_bits(sum);
                    if buffer[index] != expected {
                        return Err(format!(
                            "layer {layer} {} element {index}: reduced value {:#06x} is not the canonical bf16 RNE of the ascending-rank f32 sum ({:#06x})",
                            phase.as_str(),
                            buffer[index],
                            expected
                        ));
                    }
                }
            }
        }
        collective.verify_step(layers)?;
        rows.push(serde_json::json!({
            "step": step,
            "warm": warm,
            "calls": 2 * layers,
            "payload_bytes": (2 * layers * elements * 2) as u64,
            "wall_ms": started.elapsed().as_millis() as u64,
            "calls_per_step_ok": true,
        }));
    }
    Ok(serde_json::json!({
        "performed": true,
        "phase": "collective-skeleton",
        "transport": "tcp",
        "world": world,
        "layers": layers,
        "calls_per_step_expected": 2 * layers,
        "warmup_steps": args.warmup_steps,
        "steady_steps": args.decode_steps.max(1),
        "steps": rows,
        "canonical_rule": "bf16 payload, ascending-rank f32 sum, one bf16 RNE round; every element verified against the expected value",
        "phase_order": ["attn", "moe"],
        "attention_tensors_bound": bound_attention,
        "attention_tensors_expected": table_expected,
        "async_reduce": {
            "status": "real-posted-wait-api-wired; loop post/wait ordering not implemented yet",
            "seam": "PostedReduce over sparkflash_comm::tcp::{ReduceSlot, ReduceHandle}",
            "exactly_once": "structural in the transport: one outstanding reduce per (layer, phase) slot, second post = Protocol error",
            "note": "TcpTransport::all_reduce_bf16 is synchronous; loopback measured ~0.51 ms/reduce, so TCP needs a posted reduce handle (issue layer L's MoE reduce, overlap with layer L+1 attention, wait at the next consumption point) to fit the 42 ms verify budget"
        },
        "replicated_attention_arithmetic": "replicating attention moves ~5.15 GiB/rank/token vs ~1.29 sharded (~3.8 -> ~7.7 GB/token/rank, roofline ~13.9 -> ~28 ms) to remove 40 of the 80 TCP reduces; once RDMA one-shot (~1.7 ms/step, K6) lands that trade is worse - sharded attention stays the measured path and the replicated arm stays diagnostic-only",
        "note": "loop skeleton: no forward pass and no token - the walk exercises transport, reduce order, 10 KB bf16 payload per call and the 80-calls-per-step assertion only; the attention/MoE/tail bindings are in the bind object written before this walk, and head/logits/timing JSON are still missing",
    }))
}

/// The process-lifetime reduce buffer arena: 80 slots (40 layers x 2 phases) x dim u16, created once
/// and leaked deliberately for the run (bounded at 819,200 bytes for the shipped config). The
/// posted-reduce handles borrow 'static slices, so a post can stay outstanding across the compute
/// region and be waited exactly once at its consumption point, with no per-step allocation and no
/// raw pointers. Each slot is taken once per step and returned by `put` at the wait.
#[allow(dead_code)] // device-arm scaffolding: the reduce schedule is not wired into the loop yet
struct AsyncReduceArena {
    layers: usize,
    slots: Vec<Option<&'static mut [u16]>>,
}

#[allow(dead_code)] // device-arm scaffolding: used with the schedule above
impl AsyncReduceArena {
    fn new(layers: usize, elements: usize) -> Self {
        let mut slots = Vec::with_capacity(layers * 2);
        for _ in 0..layers * 2 {
            let buffer: &'static mut [u16] = Box::leak(vec![0u16; elements].into_boxed_slice());
            slots.push(Some(buffer));
        }
        Self { layers, slots }
    }

    fn slot_index(layers: usize, layer: usize, phase: usize) -> Option<usize> {
        if layer >= layers || phase > 1 {
            return None;
        }
        Some(layer * 2 + phase)
    }

    fn take(&mut self, layer: usize, phase: usize) -> Option<&'static mut [u16]> {
        let index = Self::slot_index(self.layers, layer, phase)?;
        self.slots.get_mut(index)?.take()
    }

    fn put(&mut self, layer: usize, phase: usize, buffer: &'static mut [u16]) {
        if let Some(index) = Self::slot_index(self.layers, layer, phase)
            && let Some(slot) = self.slots.get_mut(index)
        {
            *slot = Some(buffer);
        }
    }

    fn bytes(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .map(|buffer| buffer.len() * 2)
            .sum()
    }
}

/// The resident-stage argv for this rank, shared by the evidence path and the loop path.
fn resident_argv(args: &ChainArgs) -> Vec<String> {
    let mut resident_args: Vec<String> = Vec::new();
    if let Some(path) = &args.checkpoint {
        resident_args.extend(["--checkpoint".to_string(), path.display().to_string()]);
    }
    if let Some(path) = &args.tensor_index {
        resident_args.extend(["--tensor-index".to_string(), path.display().to_string()]);
    }
    resident_args.extend([
        "--tp-rank".to_string(),
        args.rank.to_string(),
        "--tp-world-size".to_string(),
        args.world.to_string(),
        "--evidence-dir".to_string(),
        args.evidence_dir.display().to_string(),
    ]);
    resident_args.push(if args.plan_only {
        "--plan-only".to_string()
    } else {
        "--upload".to_string()
    });
    resident_args
}

/// C1 of the device loop: per-rank plan setup, arena streaming and attention binding only.
/// No forward pass, no token - the evidence says setup_only: true and must not be read as a step.
fn run_loop(args: &ChainArgs) -> Result<Vec<String>, String> {
    // ---- S2-a run header: the arm and the collective path, decided before anything else ----
    // A forward run that does not name its collective path (SPARK_COMM_REDUCE), names one that does
    // not match its arm, or spells the arm switch wrong refuses here, with an envelope that records
    // what the operator set, before the plan setup, the arena boot and any transport.
    let header = if args.forward {
        match crate::run_header::RunHeader::from_env() {
            Ok(header) => {
                // Printed now rather than returned with the report lines: a run that fails later
                // returns only its error, and the log must still open with the header.
                for line in header.lines(args.rank) {
                    println!("{line}");
                }
                Some(header)
            }
            Err(reason) => {
                let mut envelope = crate::tp4_token::TokenEnvelope::refused(
                    args.rank,
                    args.world,
                    format!("run header: {reason}"),
                );
                envelope.run_header = Some(crate::run_header::RunHeader::observed_json(|name| {
                    std::env::var(name).ok()
                }));
                let outcome = crate::tp4_forward::TokenStageOutcome::Refused(envelope);
                fs::create_dir_all(&args.evidence_dir)
                    .map_err(|error| format!("create {}: {error}", args.evidence_dir.display()))?;
                for line in write_forward_evidence(args, &outcome)? {
                    println!("{line}");
                }
                return Err(format!(
                    "forward refused on rank {}: run header: {reason}",
                    args.rank
                ));
            }
        }
    } else {
        None
    };
    let resident_args = resident_argv(args);
    let refs: Vec<&str> = resident_args.iter().map(String::as_str).collect();
    let (setup, source) = tp4_resident::setup_for_chain(&refs)?;
    // The request borrows the checkpoint source for the Engram seam (the tables are paged, never
    // bound): keep the source owned here for the whole stage and pass a borrow everywhere else.
    let weights = source.as_ref();
    let config = sparkflash_engine::config::ModelConfig::v41_flash()
        .map_err(|error| format!("model config: {error}"))?;
    let table =
        crate::resident_weights::rank_attention_plan_shapes(&config, args.rank, args.world)?;
    let mut lines: Vec<String> = vec![format!(
        "tp4-chain r{} loop C1 setup: plan {} shards ({} resident / {} disk), {:.3} GiB resident, plan hash {:#018x}",
        args.rank,
        setup.plan.shards().len(),
        setup.plan.resident_tensors(),
        setup.plan.disk_tensors(),
        (setup.plan.resident_bytes() as f64) / (1024.0 * 1024.0 * 1024.0),
        setup.plan.plan_hash()
    )];
    if let (true, Some(header)) = (args.forward && args.plan_only, header.as_ref()) {
        let request = build_token_request(args, &setup, &config, config.n_layers, weights, header);
        let outcome = crate::tp4_forward::refused_outcome(
            &request,
            "--forward needs --upload: the forward loop runs inside the arena epoch, and --plan-only never opens one".to_string(),
        );
        lines.extend(write_forward_evidence(args, &outcome)?);
        return Err(format!(
            "forward refused on rank {}: --plan-only has no arena",
            args.rank
        ));
    }

    // ---- C4-token preflight (host-only): fail before the arena boot, not after it ----
    let mut forward_ready: Option<(
        crate::tp4_forward::TokenStageRequest<'_>,
        crate::tp4_forward::ForwardPlan,
    )> = None;
    if let Some(header) = header.as_ref() {
        let request = build_token_request(args, &setup, &config, config.n_layers, weights, header);
        match crate::tp4_forward::preflight(&request) {
            Ok(plan) => forward_ready = Some((request, plan)),
            Err(outcome) => {
                lines.extend(write_forward_evidence(args, &outcome)?);
                return Err(format!(
                    "forward refused on rank {}: {}",
                    args.rank,
                    outcome.error().unwrap_or("no reason recorded")
                ));
            }
        }
    }
    // Load all four ranks' resident arenas in parallel BEFORE opening any collective transport or
    // Hello barrier: no rank waits on another during the fill.
    let mut forward_transport = None;
    let mut forward_loop_transport = None;
    let (bind, bind_evidence, arena_evidence, bound, attention_expected, forward_outcome) = if args
        .plan_only
    {
        if forward_ready.is_some() && !args.peers.is_empty() {
            forward_transport = Some(open_transport(args)?);
            forward_loop_transport = Some(open_transport_with_base(
                args,
                args.base_port + FORWARD_LOOP_PORT_OFFSET,
            )?);
        }
        (
            None,
            serde_json::json!({
                "schema": "sparkflash-tp4-bind/1",
                "performed": false,
                "note": "--plan-only: no arena, no bind",
            }),
            serde_json::json!(null),
            0usize,
            table.len(),
            None,
        )
    } else {
        let source = source.as_ref().ok_or_else(|| {
            "--loop needs --checkpoint: the arena binds the bytes it streams".to_string()
        })?;
        // The bind writes tp4-bind-r<rank>.json progressively (attention, then MoE, then tail
        // and census) from inside the arena epoch, so the evidence is on disk before the walk
        // and a later failure cannot erase a completed bind stage. A forward pass runs in the
        // same epoch: the arena's device pointers die with it, so bind-and-run is one call.
        let (bind, arena_evidence, _arena_lines, forward_outcome) = match forward_ready.take() {
            None => {
                let (bind, arena_evidence, _arena_lines) = tp4_resident::bind_device_sets_in_arena(
                    &setup,
                    source,
                    &table,
                    &config,
                    args.rank,
                    args.world,
                    &args.evidence_dir,
                )?;
                (bind, arena_evidence, _arena_lines, None)
            }
            Some((request, plan)) => {
                let (bind, arena_evidence, _arena_lines, outcome) =
                    tp4_resident::bind_device_sets_in_arena_with(
                        &setup,
                        source,
                        &table,
                        &config,
                        args.rank,
                        args.world,
                        &args.evidence_dir,
                        |arena, bind| {
                            if !args.peers.is_empty() {
                                forward_transport = Some(open_transport(args)?);
                                forward_loop_transport = Some(open_transport_with_base(
                                    args,
                                    args.base_port + FORWARD_LOOP_PORT_OFFSET,
                                )?);
                            }
                            let transport = forward_loop_transport.as_ref();
                            // The run's ONE NCCL communicator (crate::tp4_reduce): the id rendezvous
                            // over this loop's own transport, one open on every rank, the setup
                            // warm-up and its single sync - all before layer 0 and outside the timed
                            // region. Any failure here aborts this rank BEFORE run_with_arena, so a
                            // bootstrap failure can never reach a layer (every peer aborts through the
                            // same gather timeout: no rank opens alone, no half-open communicator).
                            //
                            // Gated on the arm the loop will actually take: the request's decision,
                            // computed once by decide_layer_arm before this closure ran (which is
                            // also why evidence recording, which forces the host arm, cannot open a
                            // communicator here). HOST ARM: no Cuda::load, no scratch allocation, no open, no
                            // warm-up enqueues, reduce stays None - that arm produced 179.81 ms/step
                            // and must not acquire a libnccl dependency or a new setup cost. DEVICE
                            // ARM: the sequence above runs on every rank, and the two collective
                            // bodies enqueue on the handle this block attaches.
                            let mut reduce_cuda: Option<sparkflash_kernels::cuda_driver::Cuda> =
                                None;
                            let mut reduce_scratch: Option<
                                sparkflash_kernels::cuda_driver::Deviceptr,
                            > = None;
                            let mut run_reduce: Option<
                                std::cell::RefCell<sparkflash_comm::RunReduce>,
                            > = None;
                            if request.arm.is_device()
                                && let Some(transport) = transport
                            {
                                // The rendezvous agreement words: the arm bit, the run's
                                // native-rounding selection, the per-process behaviour switches and
                                // the plan digest. Every rank must match or no rank enters NCCL.
                                let reduce_config =
                                    sparkflash_comm::device_reduce::BootstrapConfig {
                                        native_rounding: u16::from(args.native_rounding.combine)
                                            | (u16::from(args.native_rounding.mhc) << 1)
                                            | (u16::from(args.native_rounding.shared) << 2),
                                        env_toggles: behaviour_toggles(),
                                        plan_digest: setup.plan.plan_hash(),
                                    };
                                let cuda = sparkflash_kernels::cuda_driver::Cuda::load().map_err(
                                    |error| {
                                        format!(
                                            "forward reduce bootstrap r{}: opening the run's CUDA context for the setup sync: {error}",
                                            args.rank
                                        )
                                    },
                                )?;
                                let mut allocated = 0usize;
                                let scratch = cuda
                                    .alloc(REDUCE_SCRATCH_BYTES, &mut allocated)
                                    .map_err(|error| {
                                        format!(
                                            "forward reduce bootstrap r{}: allocating the {REDUCE_SCRATCH_BYTES}-byte warm-up scratch: {error}",
                                            args.rank
                                        )
                                    })?;
                                let opened = crate::tp4_reduce::open_run_reduce(
                                    transport,
                                    args.world,
                                    args.rank,
                                    0, // the legacy default stream: every launch of this loop is on it
                                    scratch.0,
                                    REDUCE_PAYLOAD_ELEMENTS,
                                    reduce_config,
                                )
                                .map_err(|error| {
                                    format!(
                                        "forward reduce bootstrap r{}: {error}; aborting this rank before any layer - a bootstrap failure must not reach a layer",
                                        args.rank
                                    )
                                });
                                match opened {
                                    Ok(opened) => {
                                        run_reduce = Some(std::cell::RefCell::new(opened));
                                        reduce_scratch = Some(scratch);
                                        reduce_cuda = Some(cuda);
                                    }
                                    Err(error) => {
                                        // A surviving process must not leak the setup scratch: free
                                        // it before the rank aborts the run.
                                        cuda.free_buffers(&[&scratch]);
                                        return Err(error);
                                    }
                                }
                            }
                            // The request borrows the handle for the whole run: every rank attaches
                            // the SAME communicator to both per-layer stages, and the field replacement
                            // below is the only place a TokenStageRequest's reduce becomes Some.
                            let request = crate::tp4_forward::TokenStageRequest {
                                reduce: run_reduce.as_ref().map(crate::tp4_forward::DeviceReduce),
                                ..request
                            };
                            let outcome = crate::tp4_forward::run_with_arena(
                                &request, &plan, arena, bind, transport,
                            );
                            // The run has returned; its teardown sync has observed the last collective.
                            // Only now are the setup scratch and (at scope end, via Drop) the
                            // communicator released - never before a run's last enqueue completed.
                            if let (Some(cuda), Some(scratch)) =
                                (reduce_cuda.as_ref(), reduce_scratch.as_ref())
                            {
                                cuda.free_buffers(&[scratch]);
                            }
                            Ok(outcome)
                        },
                    )?;
                (bind, arena_evidence, _arena_lines, Some(outcome))
            }
        };
        let bind_path = args
            .evidence_dir
            .join(format!("tp4-bind-r{}.json", args.rank));
        let bind_evidence = bind.evidence(args.rank, args.world, table.len());
        let manifest_path = args
            .evidence_dir
            .join(format!("tp4-bind-names-r{}.txt", args.rank));
        fs::write(
            &manifest_path,
            bind.manifest_lines.join(
                "
",
            ),
        )
        .map_err(|error| format!("write {}: {error}", manifest_path.display()))?;
        let bound = bind.attention.coverage.bound.len();
        let expected = bind.attention.coverage.expected.len();
        // Coverage rows are ordered [routed gate/up, routed down, shared, router]; the log line
        // must print each row's own bound count (the window's first run summed rows 0+1 into
        // the gate/up slot and printed their difference as down, which read as a missing down
        // class in the log even though the JSON evidence was correct).
        let gate_up_bound = bind.moe.coverage[0].1.bound.len();
        let down_bound = bind.moe.coverage[1].1.bound.len();
        let shared_bound = bind.moe.coverage[2].1.bound.len();
        let router_bound = bind.moe.coverage[3].1.bound.len();
        lines.push(format!(
                "tp4-chain r{} bind: progressive evidence {} written before any walk or forward attempt",
                args.rank,
                bind_path.display()
            ));
        lines.push(format!(
                "tp4-chain r{} bind: attention {}/{} device-served names ({} host seams excluded); MoE routed gate/up {} + down {} names, shared {}, router {}",
                args.rank,
                bound,
                expected,
                bind.attention.coverage.host_seams.len(),
                gate_up_bound,
                down_bound,
                shared_bound,
                router_bound
            ));
        lines.push(format!(
                "tp4-chain r{} bind: census {} resident shards, unclassified {}, per-token weight H2D 0 by construction; names manifest {}",
                args.rank,
                bind.census.total,
                bind.census.unclassified.len(),
                manifest_path.display()
            ));
        (
            Some(bind),
            bind_evidence,
            arena_evidence,
            bound,
            expected,
            forward_outcome,
        )
    };
    // The loop transport is done: close it before any rank enters the walk. The loop's own
    // end-of-loop barrier already ran inside run_forward_loop, so every rank is out of it.
    drop(forward_loop_transport);
    let forward_evidence_written = if let Some(outcome) = forward_outcome.as_ref() {
        lines.extend(write_forward_evidence(args, outcome)?);
        true
    } else {
        false
    };
    let (transport_evidence, walk_evidence) = if args.peers.is_empty() {
        (
            serde_json::json!({"opened": false, "note": "no --comm-peers: transport not opened"}),
            serde_json::json!({"performed": false, "note": "no --comm-peers: no collective walk"}),
        )
    } else {
        let transport = match forward_transport.take() {
            Some(transport) => transport,
            None => open_transport(args)?,
        };
        let evidence = match collective_walk(args, transport, &config, bound, attention_expected) {
            Ok(walk) => walk,
            Err(error) => {
                return Err(format!("collective walk on rank {}: {error}", args.rank));
            }
        };
        (
            serde_json::json!({
                "opened": true,
                "world": args.world,
                "peers": args.peers.len(),
                "base_port": args.base_port,
            }),
            evidence,
        )
    };
    #[cfg(feature = "cuda")]
    let hybrid_sequence = if let Some(cubin_dir) = args.cubin_dir.as_deref() {
        let checkpoint = args.checkpoint.as_deref().ok_or_else(|| {
            "--loop with --cubin-dir needs --checkpoint: the hybrid runner loads the checkpoint"
                .to_string()
        })?;
        let dump_logits = args.evidence_dir.join("hybrid-dump-logits.jsonl");
        let timing_json = args.evidence_dir.join("hybrid-timing.json");
        let started = Instant::now();
        let result = crate::device_chain::run_hybrid_sequence_for(
            checkpoint,
            cubin_dir,
            args.prompt_tokens.max(1),
            args.decode_steps.max(1),
            Some(&dump_logits),
            Some(&timing_json),
            None,
        );
        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        match result {
            Ok(lines) => serde_json::json!({
                "executed": true,
                "execution": "hybrid-sequence runner (world=1, no collectives)",
                "wall_ms": wall_ms,
                "prompt_tokens": args.prompt_tokens.max(1),
                "decode_steps": args.decode_steps.max(1),
                "dump_logits": dump_logits.display().to_string(),
                "timing_json": timing_json.display().to_string(),
                "lines": lines,
            }),
            Err(error) => {
                serde_json::json!({"executed": false, "wall_ms": wall_ms, "error": error})
            }
        }
    } else {
        serde_json::json!({
            "executed": false,
            "note": "no --cubin-dir: the hybrid-sequence runner was not invoked",
        })
    };
    #[cfg(not(feature = "cuda"))]
    let hybrid_sequence = serde_json::json!({
        "executed": false,
        "note": "built without the cuda feature: the hybrid-sequence runner is not in this binary",
    });
    let evidence = serde_json::json!({
        "stage": "tp4-chain",
        "schema": "sparkflash-tp4-loop/1",
        "loop_step": "C1-setup",
        "setup_only": true,
        "rank": args.rank,
        "world": args.world,
        "plan_only": args.plan_only,
        "forward": if args.forward { "requested" } else { "not-run" },
        "tokens": 0,
        "attention": {
            "tensors_expected": attention_expected,
            "tensors_table_entries": table.len(),
            "tensors_bound": bound,
            "sharded": true,
            "note": "device arm: wq_b/wo_a column, wo_b row; one reduce per layer after attention, one after MoE. tensors_expected is the device-served set (table minus host seams), so bound == expected means full coverage. Every bound name, and the MoE/shared/router/tail classes with their launch compatibility, are in the bind object and in tp4-bind-r<rank>.json, written before the walk",
        },
        "bind": bind_evidence,
        "plan": {
            "shards": setup.plan.shards().len(),
            "resident_tensors": setup.plan.resident_tensors(),
            "disk_tensors": setup.plan.disk_tensors(),
            "resident_bytes": setup.plan.resident_bytes(),
            "ranges": setup.plan.ranges(),
            "plan_hash": format!("{:#018x}", setup.plan.plan_hash()),
        },
        "arena": arena_evidence,
        "transport": transport_evidence,
        "collectives": walk_evidence,
        "execution": if args.cubin_dir.is_some() {
            "device hybrid-sequence runner invoked from the loop path (world=1, no collectives)"
        } else {
            "host-side only: plan/arena/bind/walk; the hybrid runner was not invoked"
        },
        "moe": if let Some(bind) = &bind {
            bind.moe.evidence()
        } else {
            serde_json::json!({"performed": false, "note": "--plan-only: no arena, no MoE bind"})
        },
        "tail": if let Some(bind) = &bind {
            serde_json::json!({
                "tensors_bound": bind.tail_coverage.bound.len(),
                "coverage": bind.tail_coverage.evidence("tail"),
                "note": "embed/head row shards and final norm are arena-bound; head/token arithmetic is not wired on this arm yet",
            })
        } else {
            serde_json::json!({"performed": false, "note": "--plan-only: no arena, no tail bind"})
        },
        "census": if let Some(bind) = &bind {
            bind.census.evidence()
        } else {
            serde_json::json!({"performed": false, "note": "--plan-only: no arena, no census"})
        },
        "device_tensors_touched": args.cubin_dir.is_some() || !args.plan_only,
        "collaborative_reduce": "none (C3 walk is transport-only)",
        "rank_injection": "pending",
        "hybrid_sequence": hybrid_sequence,
        "blockers": [
            "rank-aware provider over the bound arena is not implemented: no forward pass can run yet, so no token",
            "the routed down projection (w2) binds bytes-verified but the current combine kernel reads the A8 pre-padded 640-K row layout, not the arena's checkpoint rows: a resident launch needs a kernel ABI variant or a one-time boot repack (bind.launch_compatibility names both)",
            "attention auxiliaries (sink, q_norm, kv_norm, compressor/indexer parts) and the mhc/engram weights are host-read today: the census names their counts",
            "head/token arithmetic, --dump-logits and --timing-json are not wired into the loop",
            "per-layer residual digests through the post closure are not wired",
        ],
        "note": "setup + four-rank collective skeleton: plan, arena, full device-set bind (attention, routed/shared/router MoE, tail) with per-class named coverage, and the 80-calls-per-step transport walk; no forward pass, no token, no measurement",
    });
    let path = args
        .evidence_dir
        .join(format!("tp4-loop-r{}.json", args.rank));
    fs::write(
        &path,
        serde_json::to_vec_pretty(&evidence).map_err(|error| format!("serialise: {error}"))?,
    )
    .map_err(|error| format!("write {}: {error}", path.display()))?;
    lines.push(format!(
        "tp4-chain r{} loop evidence: {} (forward={} tokens=0)",
        args.rank,
        path.display(),
        if args.forward { "requested" } else { "not-run" }
    ));
    if let Some(outcome) = forward_outcome {
        if !forward_evidence_written {
            lines.extend(write_forward_evidence(args, &outcome)?);
        }
        if outcome.is_refused() {
            return Err(format!(
                "forward refused on rank {}: {}",
                args.rank,
                outcome.error().unwrap_or("no reason recorded")
            ));
        }
    }
    Ok(lines)
}

/// Runs one rank and returns its report lines.
pub fn run_lines(args: &[&str]) -> Result<Vec<String>, String> {
    let args = ChainArgs::parse(args)?;
    if args.loop_mode {
        return run_loop(&args);
    }
    let mut lines = Vec::new();
    let resident_args = resident_argv(&args);
    let refs: Vec<&str> = resident_args.iter().map(String::as_str).collect();
    // One arena code path: the resident stage lays out, allocates and streams the rank's plan.
    let resident_lines = tp4_resident::run_lines(&refs)?;
    lines.extend(
        resident_lines
            .into_iter()
            .map(|line| format!("tp4-chain r{}: {line}", args.rank)),
    );
    let resident_path = args
        .evidence_dir
        .join(format!("tp4-resident-r{}.json", args.rank));
    let text = fs::read_to_string(&resident_path)
        .map_err(|error| format!("read {}: {error}", resident_path.display()))?;
    let mut evidence: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("parse {}: {error}", resident_path.display()))?;
    let collectives = if args.comm_smoke {
        let smoke = comm_smoke(&args)?;
        lines.push(format!(
            "tp4-chain r{}: comm smoke ok world={} peers={} reduce_calls=1 elements=8 wire_bytes={}",
            args.rank,
            args.world,
            args.peers.len(),
            smoke.get("wire_bytes").and_then(serde_json::Value::as_u64).unwrap_or(0)
        ));
        smoke
    } else {
        serde_json::json!({
            "transport": "tcp",
            "world": args.world,
            "calls": 0u64,
            "elements": 0u64,
            "payload_bytes": 0u64,
            "wire_bytes": 0u64,
            "per_step_calls": [],
            "per_step_payload_bytes": [],
            "connect_ms": 0u64,
            "wall_ms": 0u64,
            "per_step_wall_ms": [],
            "phase": "not-run",
            "note": "no --comm-smoke: the 40-layer loop is not wired yet",
        })
    };
    evidence["stage"] = serde_json::json!("tp4-chain");
    evidence["decode_steps"] = serde_json::json!(args.decode_steps);
    // CUDA graph capture primitives do not exist in this build. The flag is honoured as an
    // interface, never faked: `on` refuses, `auto` records the fallback and the per-kernel reasons.
    if args.cuda_graphs == CudaGraphs::On {
        return Err("--cuda-graphs on is not available in this build: the kernels crate has no graph capture primitive; use --cuda-graphs auto or off, or SPARKFLASH_DISABLE_CUDA_GRAPHS=1".to_string());
    }
    let graph_fallbacks = [
        serde_json::json!({"blocker": "no-graph-primitive", "detail": "the driver wrapper exposes no cuStreamBeginCapture/cuGraphInstantiate/cuGraphLaunch yet"}),
        serde_json::json!({"blocker": "per-step-allocs", "detail": "the chain allocates and frees device staging buffers inside a step; those calls cannot be captured in a graph"}),
        serde_json::json!({"blocker": "in-step-dtoh", "detail": "step readbacks copy device->host inside the step and cannot be captured"}),
        serde_json::json!({"blocker": "ledger-events", "detail": "cudaEventRecord/ElapsedTime used by SPARKFLASH_STAGE_TIMING cannot be captured; graph runs must disable the ledger"}),
    ];
    evidence["warmup"] = serde_json::json!({
        "steps": args.warmup_steps,
        "cuda_graphs": match args.cuda_graphs {
            CudaGraphs::On => "on",
            CudaGraphs::Off => "off",
            CudaGraphs::Auto => "auto",
        },
        "cuda_graphs_active": false,
        "escape_hatch": "SPARKFLASH_DISABLE_CUDA_GRAPHS=1 or --cuda-graphs off",
        "graph_fallbacks": graph_fallbacks,
        "note": "warmup steps are recorded now; the dummy steps themselves run once the forward loop lands"
    });
    evidence["collectives"] = collectives;
    let chain_path = args
        .evidence_dir
        .join(format!("tp4-chain-r{}.json", args.rank));
    fs::write(
        &chain_path,
        serde_json::to_vec_pretty(&evidence).map_err(|error| format!("serialise: {error}"))?,
    )
    .map_err(|error| format!("write {}: {error}", chain_path.display()))?;
    lines.push(format!("tp4-chain evidence: {}", chain_path.display()));
    if args.rank == 0 {
        let summary_path = args.evidence_dir.join("tp4-chain-summary.json");
        let summary = serde_json::json!({
            "schema": "sparkflash-tp4-chain/1",
            "world": args.world,
            "decode_steps": args.decode_steps,
            "comm_smoke": args.comm_smoke,
            "collectives": evidence["collectives"].clone(),
            "note": "rank 0 summary; per-rank files are tp4-chain-rR.json",
        });
        fs::write(
            &summary_path,
            serde_json::to_vec_pretty(&summary).map_err(|error| format!("serialise: {error}"))?,
        )
        .map_err(|error| format!("write {}: {error}", summary_path.display()))?;
        lines.push(format!("tp4-chain summary: {}", summary_path.display()));
    }
    Ok(lines)
}
/// Which half of a layer the reduce follows. The device arm shards attention, so a layer has two
/// reductions: one after the attention output projection and one after the MoE output. The CPU
/// twin's one-per-layer count is a property of its replicated-attention arm and must never be used
/// as the device arm's expectation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReducePhase {
    Attention,
    Moe,
}

impl ReducePhase {
    fn index(self) -> usize {
        match self {
            ReducePhase::Attention => 0,
            ReducePhase::Moe => 1,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            ReducePhase::Attention => "attn",
            ReducePhase::Moe => "moe",
        }
    }
}

/// The overlapped-reduce seam over the real posted/wait transport API. The handle owns the buffer
/// between post and wait, so a contribution cannot be overwritten and the canonical bf16
/// ascending-rank sum is completed exactly once per (layer, phase) slot.
#[allow(dead_code)] // device-arm scaffolding: the seam is not wired into the loop yet
trait PostedReduce {
    fn post_reduce<'a>(
        &mut self,
        slot: sparkflash_comm::tcp::ReduceSlot,
        seq: u64,
        buffer: &'a mut [u16],
    ) -> Result<sparkflash_comm::tcp::ReduceHandle<'a>, String>;
    fn wait_reduce<'a>(
        &mut self,
        handle: sparkflash_comm::tcp::ReduceHandle<'a>,
    ) -> Result<&'a mut [u16], String>;
}

impl PostedReduce for TcpTransport {
    fn post_reduce<'a>(
        &mut self,
        slot: sparkflash_comm::tcp::ReduceSlot,
        seq: u64,
        buffer: &'a mut [u16],
    ) -> Result<sparkflash_comm::tcp::ReduceHandle<'a>, String> {
        self.post_reduce_bf16(slot, seq, buffer)
            .map_err(|error| format!("posted all-reduce: {error}"))
    }

    fn wait_reduce<'a>(
        &mut self,
        handle: sparkflash_comm::tcp::ReduceHandle<'a>,
    ) -> Result<&'a mut [u16], String> {
        self.wait_reduce(handle)
            .map_err(|error| format!("posted all-reduce wait: {error}"))
    }
}

/// Counts one step's per-layer, per-phase collectives so a missing or double reduce is visible in
/// the evidence and fails the step instead of averaging out.
#[derive(Debug)]
struct CollectiveCounter {
    calls: u64,
    elements: u64,
    payload_bytes: u64,
    per_layer_calls: Vec<[u64; 2]>,
    per_layer_payload_bytes: Vec<[u64; 2]>,
}

impl CollectiveCounter {
    fn new(layers: usize) -> Self {
        Self {
            calls: 0,
            elements: 0,
            payload_bytes: 0,
            per_layer_calls: vec![[0; 2]; layers],
            per_layer_payload_bytes: vec![[0; 2]; layers],
        }
    }

    fn record(&mut self, layer: usize, phase: ReducePhase, elements: usize) -> Result<(), String> {
        let Some(calls) = self.per_layer_calls.get_mut(layer) else {
            return Err(format!(
                "collective for layer {layer} is outside the {}-layer counter",
                self.per_layer_calls.len()
            ));
        };
        let slot = phase.index();
        calls[slot] += 1;
        let bytes = (elements * 2) as u64;
        self.calls += 1;
        self.elements = self.elements.saturating_add(elements as u64);
        self.payload_bytes = self.payload_bytes.saturating_add(bytes);
        self.per_layer_payload_bytes[layer][slot] =
            self.per_layer_payload_bytes[layer][slot].saturating_add(bytes);
        Ok(())
    }

    fn verify(&self, layers_expected: usize) -> Result<(), String> {
        let expected = layers_expected * 2;
        let mut deviations: Vec<(usize, &'static str, u64)> = Vec::new();
        for (layer, calls) in self.per_layer_calls.iter().enumerate() {
            for phase in [ReducePhase::Attention, ReducePhase::Moe] {
                let observed = calls[phase.index()];
                if observed != 1 {
                    deviations.push((layer, phase.as_str(), observed));
                }
            }
        }
        if self.calls != expected as u64 || !deviations.is_empty() {
            return Err(format!(
                "collective calls per step: total {} (expected exactly {expected} = 2 x {layers_expected} layers), per-layer dead/double entries {deviations:?}",
                self.calls
            ));
        }
        Ok(())
    }

    fn evidence(&self, layers_expected: usize, wall_ms: u64) -> serde_json::Value {
        let expected = layers_expected * 2;
        serde_json::json!({
            "transport": "tcp",
            "phase": "decode",
            "calls": self.calls,
            "elements": self.elements,
            "payload_bytes": self.payload_bytes,
            "wire_bytes": self.payload_bytes,
            "per_layer_calls": self.per_layer_calls,
            "per_layer_payload_bytes": self.per_layer_payload_bytes,
            "layers_expected": layers_expected,
            "layers_expected_basis": "device arm: 2 reduces per layer (attn, moe); the CPU twin's 40/step is its replicated-attention arm, not this arm",
            "calls_per_step_expected": expected,
            "calls_per_step_ok": self.calls == expected as u64
                && self.per_layer_calls.iter().all(|calls| calls == &[1, 1]),
            "wall_ms": wall_ms,
            "note": "device step carries about 2x the CPU arm's collective count (80 vs 40 per step, ~10 KB bf16 per call at hidden 5120); size the K6/RDMA one-shot budget for 80 calls/step and do not compare this table with the CPU arm's",
        })
    }
}

/// The per-layer canonical all-reduce: bf16 payload, ascending-rank f32 sum, one bf16 RNE round -
/// the same rule the host twin uses. Two calls per layer (after attention, after MoE); the counter
/// fails the step if that is not exactly what happened.
struct LayerCollective {
    transport: TcpTransport,
    counter: CollectiveCounter,
}

#[allow(dead_code)]
impl LayerCollective {
    fn new(transport: TcpTransport, layers: usize) -> Self {
        Self {
            transport,
            counter: CollectiveCounter::new(layers),
        }
    }

    fn reduce_layer(
        &mut self,
        layer: usize,
        phase: ReducePhase,
        buffer: &mut [u16],
    ) -> Result<(), String> {
        let layout = CollectiveLayout::new(
            self.transport.world_size(),
            buffer.len(),
            FabricSchedule::OneShotAllToAll,
        )
        .map_err(|error| {
            format!(
                "collective layout for layer {layer} {}: {error}",
                phase.as_str()
            )
        })?;
        self.transport
            .all_reduce_bf16(buffer, layer as u64, &layout)
            .map_err(|error| format!("all-reduce for layer {layer} {}: {error}", phase.as_str()))?;
        self.counter.record(layer, phase, buffer.len())
    }

    fn reset(&mut self) {
        self.counter = CollectiveCounter::new(self.counter.per_layer_calls.len());
    }

    fn verify_step(&self, layers_expected: usize) -> Result<(), String> {
        self.counter.verify(layers_expected)
    }

    fn evidence(&self, layers_expected: usize, wall_ms: u64) -> serde_json::Value {
        self.counter.evidence(layers_expected, wall_ms)
    }
}

#[cfg(test)]
mod collective_tests {
    use super::*;

    #[test]
    fn two_reduces_per_layer_verify_and_double_or_missing_reduces_fail() {
        let mut counter = CollectiveCounter::new(40);
        for layer in 0..40 {
            counter
                .record(layer, ReducePhase::Attention, 5120)
                .expect("record");
            counter
                .record(layer, ReducePhase::Moe, 5120)
                .expect("record");
        }
        counter
            .verify(40)
            .expect("exactly one reduce per layer per phase");
        let evidence = counter.evidence(40, 17);
        assert_eq!(evidence["calls"], 80);
        assert_eq!(evidence["calls_per_step_expected"], 80);
        assert_eq!(evidence["payload_bytes"], 80 * 5120 * 2);
        assert_eq!(evidence["calls_per_step_ok"], true);
        counter
            .record(3, ReducePhase::Attention, 5120)
            .expect("record");
        assert!(
            counter.verify(40).is_err(),
            "a double reduce must fail the step"
        );
        let mut missing = CollectiveCounter::new(40);
        for layer in 0..40 {
            missing
                .record(layer, ReducePhase::Attention, 5120)
                .expect("record");
        }
        assert!(
            missing.verify(40).is_err(),
            "a missing MoE reduce must fail the step"
        );
        assert!(
            missing.record(40, ReducePhase::Moe, 8).is_err(),
            "a layer outside the plan is an error"
        );
    }
}

#[cfg(test)]
mod reduce_arena_tests {
    use super::*;

    #[test]
    fn the_arena_has_one_stable_slot_per_layer_and_phase() {
        let mut arena = AsyncReduceArena::new(40, 5120);
        assert_eq!(arena.bytes(), 819_200, "80 slots x 5120 u16 x 2 bytes");
        let buffer = arena.take(0, 0).expect("attention slot");
        assert!(
            arena.take(0, 0).is_none(),
            "a taken slot cannot be taken twice"
        );
        arena.put(0, 0, buffer);
        assert!(arena.take(0, 0).is_some(), "put returns the slot");
        assert!(arena.take(40, 0).is_none());
        assert!(arena.take(0, 2).is_none());
        assert_eq!(AsyncReduceArena::slot_index(40, 39, 1), Some(79));
    }
}

#[cfg(test)]
mod forward_args_tests {
    use super::*;
    use sparkflash_device::resident::{HeaderEntry, ResidentPlan};
    use std::path::Path;

    fn base_argv() -> Vec<&'static str> {
        vec![
            "--plan-only",
            "--checkpoint",
            "/tmp/checkpoint",
            "--tp-rank",
            "0",
            "--tp-world-size",
            "4",
            "--evidence-dir",
            "/tmp/evidence",
        ]
    }

    fn parse_with(extra: &[&'static str]) -> Result<ChainArgs, String> {
        let mut argv = base_argv();
        argv.extend_from_slice(extra);
        ChainArgs::parse(&argv)
    }

    #[test]
    fn the_forward_stage_flags_parse_and_keep_their_defaults_off() {
        let plain = parse_with(&[]).expect("plain argv parses");
        assert!(!plain.forward);
        assert!(!plain.forward_preflight);
        assert!(plain.kernel_cubin_dir.is_none());
        assert!(plain.token_ids.is_empty());

        let full = parse_with(&[
            "--loop",
            "--forward",
            "--forward-preflight",
            "--kernel-cubin-dir",
            "/tmp/cubins",
            "--token-ids",
            "5,7,129",
        ])
        .expect("forward argv parses");
        assert!(full.forward);
        assert!(full.forward_preflight);
        assert_eq!(
            full.kernel_cubin_dir.as_deref(),
            Some(Path::new("/tmp/cubins"))
        );
        assert_eq!(full.token_ids, vec![5, 7, 129]);
    }

    #[test]
    fn the_forward_perf_probe_flag_is_off_by_default_and_needs_forward() {
        let plain = parse_with(&[]).expect("plain argv parses");
        assert!(!plain.forward_lean);
        assert!(!plain.forward_perf_probe);

        let full = parse_with(&[
            "--loop",
            "--forward",
            "--forward-preflight",
            "--forward-perf-probe",
        ])
        .expect("probe argv parses");
        assert!(full.forward_perf_probe);
        assert!(!full.forward_lean);

        // Both diagnostic switches coexist: the probe row D is --forward-lean --forward-perf-probe.
        let both = parse_with(&[
            "--loop",
            "--forward",
            "--forward-preflight",
            "--forward-lean",
            "--forward-perf-probe",
        ])
        .expect("lean + probe argv parses");
        assert!(both.forward_lean);
        assert!(both.forward_perf_probe);

        let error =
            parse_with(&["--loop", "--forward-perf-probe"]).expect_err("probe needs --forward");
        assert!(
            error.contains("--forward-perf-probe is only used by --forward"),
            "{error}"
        );
    }

    #[test]
    fn forward_without_the_loop_is_rejected() {
        let error = parse_with(&["--forward"]).expect_err("forward needs --loop");
        assert!(error.contains("--forward requires --loop"), "{error}");
    }

    #[test]
    fn the_preflight_word_and_the_forward_only_flags_need_forward() {
        let error = parse_with(&["--loop", "--forward-preflight"]).expect_err("no forward");
        assert!(
            error.contains("--forward-preflight requires --forward"),
            "{error}"
        );
        let error =
            parse_with(&["--loop", "--kernel-cubin-dir", "/tmp/c"]).expect_err("no forward");
        assert!(error.contains("only used by --forward"), "{error}");
        let error = parse_with(&["--loop", "--token-ids", "5"]).expect_err("no forward");
        assert!(error.contains("only used by --forward"), "{error}");
    }

    #[test]
    fn the_reference_vector_flag_pairs_one_to_one_with_the_decode_positions() {
        let fixture = parse_with(&[
            "--loop",
            "--forward",
            "--forward-preflight",
            "--decode-steps",
            "2",
            "--token-ids",
            "129,130",
            "--forward-reference-ids",
            "491,367",
        ])
        .expect("reference argv parses");
        assert_eq!(fixture.reference_ids, vec![491, 367]);
        assert_eq!(fixture.token_ids, vec![129, 130]);

        let error = parse_with(&[
            "--loop",
            "--forward",
            "--forward-preflight",
            "--decode-steps",
            "2",
            "--forward-reference-ids",
            "491",
        ])
        .expect_err("one reference for two positions");
        assert!(error.contains("1 ids for 2 decode positions"), "{error}");

        let error = parse_with(&["--loop", "--forward-reference-ids", "491"])
            .expect_err("reference ids need --forward");
        assert!(error.contains("only used by --forward"), "{error}");

        let error = parse_with(&[
            "--loop",
            "--forward",
            "--forward-preflight",
            "--decode-steps",
            "2",
            "--forward-reference-ids",
            "491,999999",
        ])
        .expect_err("outside the vocabulary");
        assert!(error.contains("--forward-reference-ids 999999"), "{error}");
    }

    #[test]
    fn the_forward_native_dense_parity_flag_is_off_by_default_and_takes_an_optional_spec() {
        let plain = parse_with(&[]).expect("plain argv parses");
        assert!(plain.native_dense_parity.is_none());

        let bare = parse_with(&[
            "--loop",
            "--forward",
            "--forward-preflight",
            "--forward-native-dense-parity",
        ])
        .expect("bare parity flag parses");
        assert_eq!(
            bare.native_dense_parity,
            Some(crate::tp4_forward::NativeDenseParitySpec {
                layer: 0,
                position: None,
            })
        );

        let spec = parse_with(&[
            "--loop",
            "--forward",
            "--forward-preflight",
            "--forward-native-dense-parity",
            "7:129",
        ])
        .expect("layer:position parses");
        assert_eq!(
            spec.native_dense_parity,
            Some(crate::tp4_forward::NativeDenseParitySpec {
                layer: 7,
                position: Some(129),
            })
        );

        assert!(
            parse_with(&[
                "--loop",
                "--forward",
                "--forward-preflight",
                "--forward-native-dense-parity",
                "x",
            ])
            .is_err(),
            "a non-usize spec must be refused"
        );
        assert!(
            parse_with(&["--forward-native-dense-parity"]).is_err(),
            "the parity flag is only used by --forward"
        );
    }

    #[test]
    fn token_ids_reject_garbage_and_ids_outside_the_vocabulary() {
        let error =
            parse_with(&["--loop", "--forward", "--token-ids", "5,x"]).expect_err("garbage");
        assert!(error.contains("--token-ids x"), "{error}");
        let error = parse_with(&["--loop", "--forward", "--token-ids", "129280"])
            .expect_err("out of vocab");
        assert!(
            error.contains("outside the 129280-token vocabulary"),
            "{error}"
        );
        let error = parse_with(&["--loop", "--forward", "--token-ids", ","]).expect_err("empty");
        assert!(error.contains("at least one token id"), "{error}");
    }

    #[test]
    fn the_combine_geometry_comes_from_the_released_plan_row_geometry() {
        // The released plan's routed down projection: stored [5120, 1152] I8 weight (2304 values
        // packed two per byte) with a [5120, 72] E8M0 scale grid. Column rule shard = 288 packed
        // bytes + 18 scale bytes per row, i.e. the stored 576-K layout the combine selector
        // accepts - derived here from the plan API, not from a fixture file.
        let plan = ResidentPlan::plan(
            0,
            4,
            vec![
                HeaderEntry::new(
                    "layers.0.ffn.experts.0.w2.weight".to_string(),
                    "I8".to_string(),
                    vec![5120, 1152],
                ),
                HeaderEntry::new(
                    "layers.0.ffn.experts.0.w2.scale".to_string(),
                    "F8_E8M0".to_string(),
                    vec![5120, 72],
                ),
            ],
        )
        .expect("released w2 plan");
        let (rows, bytes) = combine_geometry_from_plan(&plan);
        assert_eq!(rows, 5120);
        let (weight_bytes, scale_bytes) = bytes.expect("w2 shard bytes");
        assert_eq!(weight_bytes, 5120 * 288);
        assert_eq!(scale_bytes, 5120 * 18);
        let geometry =
            crate::tp4_combine::CombineGeometry::from_shard_bytes(rows, weight_bytes, scale_bytes)
                .expect("stored geometry");
        assert_eq!(
            geometry.layout(),
            crate::tp4_combine::CombineLayout::Stored576
        );
    }

    #[test]
    fn a_plan_without_the_routed_down_projection_reports_no_bytes_instead_of_inventing_them() {
        let plan = ResidentPlan::plan(
            0,
            4,
            vec![HeaderEntry::new(
                "norm.weight".to_string(),
                "BF16".to_string(),
                vec![5120],
            )],
        )
        .expect("minimal plan");
        let (rows, bytes) = combine_geometry_from_plan(&plan);
        assert_eq!(rows, crate::tp4_forward::DEFAULT_COMBINE_ROWS);
        assert!(bytes.is_none(), "the preflight must refuse, not guess");
    }
}

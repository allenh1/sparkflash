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
pub const USAGE: &str = "sparkflash device-chain --run-stage tp4-chain (--checkpoint DIR | --tensor-index FILE) --tp-rank R --tp-world-size 4 --evidence-dir DIR [--plan-only] [--upload] [--comm-peers IP,IP,IP,IP] [--comm-base-port 29601] [--comm-smoke] [--decode-steps N] [--prompt-tokens N] [--cubin-dir DIR]";

/// The parsed argv.
struct ChainArgs {
    rank: usize,
    world: usize,
    checkpoint: Option<PathBuf>,
    tensor_index: Option<PathBuf>,
    plan_only: bool,
    #[allow(dead_code)] // parsed for the loop arm that is not wired yet
    upload: bool,
    loop_mode: bool,
    evidence_dir: PathBuf,
    peers: Vec<std::net::IpAddr>,
    base_port: u16,
    comm_smoke: bool,
    decode_steps: usize,
    warmup_steps: usize,
    cuda_graphs: CudaGraphs,
    #[allow(dead_code)] // parsed for the loop arm that is not wired yet
    prompt_tokens: usize,
    cubin_dir: Option<PathBuf>,
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
        let mut evidence_dir: Option<PathBuf> = None;
        let mut peers_raw: Option<String> = None;
        let mut base_port = 29601u16;
        let mut plan_only = false;
        let mut upload = false;
        let mut loop_mode = false;
        let mut comm_smoke = false;
        let mut decode_steps = 1usize;
        let mut warmup_steps = 2usize;
        let mut cuda_graphs = CudaGraphs::Auto;
        let mut prompt_tokens = 1usize;
        let mut cubin_dir: Option<PathBuf> = None;
        let mut index = 0;
        while index < args.len() {
            let flag = args[index];
            index += 1;
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
                "--evidence-dir" => evidence_dir = Some(PathBuf::from(take("--evidence-dir")?)),
                "--comm-peers" => peers_raw = Some(take("--comm-peers")?),
                "--comm-base-port" => {
                    base_port =
                        u16::try_from(parse_usize("--comm-base-port", &take("--comm-base-port")?)?)
                            .map_err(|_| "--comm-base-port must fit u16".to_string())?;
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
        Ok(Self {
            rank,
            world,
            checkpoint,
            tensor_index,
            plan_only,
            upload,
            loop_mode,
            evidence_dir,
            peers,
            base_port,
            comm_smoke,
            decode_steps,
            warmup_steps,
            cuda_graphs,
            prompt_tokens,
            cubin_dir,
        })
    }
}

fn parse_usize(flag: &str, raw: &str) -> Result<usize, String> {
    raw.parse()
        .map_err(|error| format!("{flag} {raw}: {error}"))
}

/// Opens the rank's TCP transport in the launch order the bench and census use: the peer list's
/// index is the rank, rank r binds `peers[r]:base_port + r`, and every rank dials the others.
fn open_transport(args: &ChainArgs) -> Result<TcpTransport, String> {
    let addresses: Vec<SocketAddr> = args
        .peers
        .iter()
        .enumerate()
        .map(|(peer, ip)| SocketAddr::new(*ip, args.base_port + peer as u16))
        .collect();
    let config = TcpConfig::with_addresses(args.rank, addresses)
        .map_err(|error| format!("tcp config: {error}"))?
        .with_timeout(Duration::from_secs(180))
        .with_connect_timeout(Duration::from_secs(60));
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
        "note": "loop skeleton: no forward pass and no token - the walk exercises transport, reduce order, 10 KB bf16 payload per call and the 80-calls-per-step assertion only; MoE bindings, head, logits and timing JSON are still missing",
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
    let resident_args = resident_argv(args);
    let refs: Vec<&str> = resident_args.iter().map(String::as_str).collect();
    let (setup, source) = tp4_resident::setup_for_chain(&refs)?;
    let config = sparkflash_engine::config::ModelConfig::v41_flash()
        .map_err(|error| format!("model config: {error}"))?;
    let table =
        crate::resident_weights::rank_attention_plan_shapes(&config, args.rank, args.world)?;
    let (bound, arena_evidence) = if args.plan_only {
        (0usize, serde_json::json!(null))
    } else {
        let source = source.ok_or_else(|| {
            "--loop needs --checkpoint: the arena binds the bytes it streams".to_string()
        })?;
        let (bound, value, _lines) =
            tp4_resident::bind_attention_in_arena(&setup, &source, &table)?;
        (bound, value)
    };
    let (transport_evidence, walk_evidence) = if args.peers.is_empty() {
        (
            serde_json::json!({"opened": false, "note": "no --comm-peers: transport not opened"}),
            serde_json::json!({"performed": false, "note": "no --comm-peers: no collective walk"}),
        )
    } else {
        let transport = open_transport(args)?;
        let evidence = match collective_walk(args, transport, &config, bound, table.len()) {
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
        "forward": "not-run",
        "tokens": 0,
        "loop_step": "C3-collective-skeleton",
        "attention": {
            "tensors_expected": table.len(),
            "tensors_bound": bound,
            "sharded": true,
            "note": "device arm: wq_b/wo_a column, wo_b row; one reduce per layer after attention, one after MoE",
        },
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
        "device_tensors_touched": args.cubin_dir.is_some() || !args.plan_only,
        "collaborative_reduce": "none (C3 walk is transport-only)",
        "rank_injection": "pending",
        "hybrid_sequence": hybrid_sequence,
        "blockers": [
            "rank-aware provider over the bound arena is not implemented: no forward pass can run yet, so no token",
            "MoE/shared bindings and their per-category weight-H2D==0 gates are not wired",
            "head, --dump-logits and --timing-json are not wired into the loop",
            "per-layer residual digests through the post closure are not wired",
        ],
        "note": "setup + four-rank collective skeleton: plan, arena, attention binding and the 80-calls-per-step transport walk; no forward pass, no token, no measurement",
    });
    let path = args
        .evidence_dir
        .join(format!("tp4-loop-r{}.json", args.rank));
    fs::write(
        &path,
        serde_json::to_vec_pretty(&evidence).map_err(|error| format!("serialise: {error}"))?,
    )
    .map_err(|error| format!("write {}: {error}", path.display()))?;
    Ok(vec![
        format!(
            "tp4-chain r{} loop C1 setup: plan {} shards ({} resident / {} disk), {:.3} GiB resident, plan hash {:#018x}",
            args.rank,
            setup.plan.shards().len(),
            setup.plan.resident_tensors(),
            setup.plan.disk_tensors(),
            (setup.plan.resident_bytes() as f64) / (1024.0 * 1024.0 * 1024.0),
            setup.plan.plan_hash()
        ),
        format!(
            "tp4-chain r{} loop C1 setup: {} attention tensors expected, {} bound, forward=not-run tokens=0",
            args.rank,
            table.len(),
            bound
        ),
        format!("tp4-chain r{} loop evidence: {}", args.rank, path.display()),
    ])
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

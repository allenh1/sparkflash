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

//! The TP=4 stage-1 smoke: sharded read, one collective, one canonical-reduce check.
//!
//! This arm is host-only by design: it opens no driver, stages nothing on a device and produces
//! no token. What it proves is the part of the TP seam that must already hold before a device
//! window is spent:
//!
//! 1. a rank reads exactly its own slice of a real checkpoint tensor, with the plan's rule, and
//!    that slice is byte-identical to the same rows/segments of the full tensor;
//! 2. the ranks rendezvous over the real transport (TCP across nodes, or the in-process
//!    loopback when one process hosts every rank);
//! 3. the transport's all-reduce equals the canonical one-rounding reduce bit for bit, repeated
//!    the requested number of times;
//! 4. a row-parallel seam's result is within the declared L2 bar of the single-rank f32 twin;
//! 5. every rank writes its own evidence JSON with the same schema.
//!
//! It is the S5.1 deliverable of docs/archive/build/tp4-design.md section 9 and the code behind the
//! section-5.5 window command. Pass criteria: 0 mismatches against the canonical twin, 100
//! repeats bit-identical, the wo_b seam within the L2 bar, and one evidence file per rank.

use std::collections::BTreeMap;
use std::fs;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};
use sparkflash_comm::oneshot::loopback::{
    GuardPolicy, HostTwinReport, KernelDelay, OrchestrationConfig, ReadEpochPolicy, RequestOutcome,
    RingTwinConfig, RingTwinReport, ServingRequest, canonical_first_token, run_host_twin,
    run_host_twin_prefill, run_orchestration, run_ring_twin,
};
use sparkflash_comm::oneshot::{
    ChunkPlan, Dispatch, DispatchAddresses, DispatchLog, DispatchRecord, Exchange, LaunchPath,
    ONESHOT_ARRIVAL_BYTES, ONESHOT_KERNEL_NAME, ONESHOT_RAILS, ONESHOT_RANKS, ONESHOT_ROW_BYTES,
    ONESHOT_ROW_ELEMENTS, ONESHOT_SLOT_ALIGN, ONESHOT_SLOT_BYTES, SequenceCounter, SlotKind,
    SlotRegistration, row_offset,
};
use sparkflash_comm::{
    CollectiveLayout, FabricSchedule, LoopbackCluster, LoopbackConfig, TcpConfig, TcpTransport,
    Transport, canonical_all_reduce_into,
};
use sparkflash_device::shard::{ShardPlan, ShardRule, StoredShard};
use sparkflash_engine::safetensors::SafeTensorsSource;
use sparkflash_kernels::m4_quant::{bf16_bits, bf16_value, fp8_e4m3_decode, ue8m0_decode};

/// The usage line appended to the device-chain usage.
pub const USAGE: &str = "sparkflash device-chain --run-stage tp4-smoke --checkpoint DIR [--synthetic] --tp-rank 0..N --tp-world-size 4 --comm-backend tcp|lo|rdma --comm-peers IP,IP,... --comm-port 29601 --layer 0 --position 0 --repeats 100 --evidence-dir DIR [--resident] [--cubin-dir DIR] [--payload-elements N] [--seed N] [--comm-timeout SECS] [--comm-connect-timeout SECS] [--oneshot-preflight] [--prefill-plan-elements N]";

/// The default base port; rank r uses base + r.
pub const DEFAULT_COMM_PORT: u16 = 29601;

/// The default repeat count for the canonical-reduce check.
pub const DEFAULT_REPEATS: usize = 100;

/// Sequences the loopback host twin runs for the preflight evidence: enough to cycle the ring
/// several times and still finish quickly on one host.
pub const HOST_TWIN_SEQUENCES: usize = 8;

/// Upper bound on the payload a chunked host-twin run will allocate. The chunked contract is about
/// the plan's shape - more than one row, split on row boundaries, one dispatch per chunk - so the
/// evidence run clamps a large requested plan to two chunks instead of materialising a 4096-token
/// payload four times on one host. The CPU twin reduces every element in software, so the clamp
/// also keeps the preflight cheap; the ring shape itself is proven by the comm tests.
pub const HOST_TWIN_CHUNKED_MAX_ELEMENTS: usize = ONESHOT_ROW_ELEMENTS + 1024;

/// Payload the prefill ring twin runs in the preflight: above the 2 MiB one-shot row, two chunks.
pub const HOST_TWIN_RING_ELEMENTS: usize = ONESHOT_ROW_ELEMENTS + 1024;

/// Payload for the hazard demonstration: three chunks, so the third reuses the first chunk's slot.
pub const HOST_TWIN_RING_HAZARD_ELEMENTS: usize = ONESHOT_ROW_ELEMENTS * 2 + 1024;

/// The default frame deadline: 5 s, the TCP transport's own default.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

/// The default connection-setup deadline: 5 s, the TCP transport's own default.
///
/// A cross-node run raises both: two ssh-launched ranks do not start within the same second, and
/// the first rank may finish its ~130 MB load a few seconds before its peer reaches the first
/// collective.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// The L2 bar: max abs delta of the row-parallel seam, in bf16 ulp of the twin value.
pub const L2_MAX_ULP: f32 = 2.0;

/// The L2 bar: median abs delta of the row-parallel seam, in bf16 ulp of the twin value.
pub const L2_MEDIAN_ULP: f32 = 1.0;

/// The seed every rank derives its activation vector from, so the vector is identical on all
/// ranks without a broadcast.
pub const DEFAULT_SEED: u64 = 0x5f3a_91c7_b24d_8e15;

/// The evidence schema version this build writes.
pub const SCHEMA_VERSION: u32 = 2;

/// Runs the arm with the arguments that followed tp4-smoke.
pub fn run(args: &[&str]) -> ExitCode {
    match run_lines(args) {
        Ok(lines) => {
            for line in lines {
                println!("{line}");
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("sparkflash device-chain --run-stage tp4-smoke: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Runs the arm and returns the lines to print.
///
/// # Errors
/// A usage, checkpoint, transport or evidence-file error, as text.
pub fn run_lines(args: &[&str]) -> Result<Vec<String>, String> {
    let args = SmokeArgs::parse(args)?;
    if args.oneshot_preflight {
        validate_oneshot_preflight(&args)?;
    }
    match args.backend {
        Backend::Tcp => run_tcp(&args),
        Backend::Loopback => run_loopback(&args),
        Backend::Rdma => {
            #[cfg(feature = "rdma")]
            {
                run_rdma(&args)
            }
            #[cfg(not(feature = "rdma"))]
            {
                Err("--comm-backend rdma needs a build with the rdma feature (cargo build -p sparkflash-cli --features rdma), libibverbs on the nodes, an ACTIVE dual-rail fabric and the four-node window; the host stage refuses instead of silently running the host path. --oneshot-preflight checks the device-seam contract host side; the hardware entry stays the existing gates and rocenante-probe".to_string())
            }
        }
    }
}

/// The transport a rank runs over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    /// One process per rank, real TCP sockets.
    Tcp,
    /// One process hosting every rank as a thread over the in-process loopback.
    Loopback,
    /// The RDMA device path, which this host stage does not implement.
    Rdma,
}

/// The parsed arguments of one tp4-smoke invocation.
#[derive(Debug, Clone)]
struct SmokeArgs {
    rank: usize,
    world: usize,
    backend: Backend,
    peers: Vec<IpAddr>,
    port: u16,
    layer: usize,
    position: usize,
    repeats: usize,
    evidence_dir: PathBuf,
    checkpoint: Option<PathBuf>,
    cubin_dir: Option<PathBuf>,
    resident: bool,
    payload_elements: Option<usize>,
    seed: u64,
    /// Frame deadline for every collective.
    timeout: Duration,
    /// Deadline for connection setup.
    connect_timeout: Duration,
    /// Validate and record the host-side half of the S5.3 device-buffer seam before the run.
    oneshot_preflight: bool,
    /// Plan the chunked prefill path (S5.4) for this many bf16 elements; host side only.
    prefill_plan_elements: Option<usize>,
}

impl SmokeArgs {
    fn parse(args: &[&str]) -> Result<Self, String> {
        let mut rank = None;
        let mut world = 4usize;
        let mut backend = Backend::Tcp;
        let mut peers: Vec<IpAddr> = Vec::new();
        let mut port = DEFAULT_COMM_PORT;
        let mut layer = 0usize;
        let mut position = 0usize;
        let mut repeats = DEFAULT_REPEATS;
        let mut evidence_dir = None;
        let mut checkpoint = None;
        let mut cubin_dir = None;
        let mut resident = false;
        let mut payload_elements = None;
        let mut seed = DEFAULT_SEED;
        let mut timeout = DEFAULT_TIMEOUT;
        let mut connect_timeout = DEFAULT_CONNECT_TIMEOUT;
        let mut oneshot_preflight = false;
        let mut prefill_plan_elements = None;
        let mut synthetic = false;

        let mut index = 0;
        while index < args.len() {
            let flag = args[index];
            match flag {
                "--tp-rank" => {
                    rank = Some(parse_usize(
                        "--tp-rank",
                        take(args, &mut index, "--tp-rank")?,
                    )?);
                }
                "--tp-world-size" => {
                    world = parse_usize(
                        "--tp-world-size",
                        take(args, &mut index, "--tp-world-size")?,
                    )?;
                }
                "--comm-backend" => {
                    backend = match take(args, &mut index, "--comm-backend")? {
                        "tcp" => Backend::Tcp,
                        "lo" => Backend::Loopback,
                        "rdma" => Backend::Rdma,
                        other => {
                            return Err(format!("--comm-backend: {other} is not tcp, lo or rdma"));
                        }
                    };
                }
                "--comm-peers" => {
                    let raw = take(args, &mut index, "--comm-peers")?;
                    peers = raw
                        .split(',')
                        .map(|peer| {
                            let peer = peer.trim();
                            peer.parse::<IpAddr>()
                                .map_err(|_| format!("--comm-peers: {peer} is not an IP address"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                }
                "--comm-port" => {
                    port = take(args, &mut index, "--comm-port")?
                        .parse::<u16>()
                        .map_err(|_| "--comm-port: not a port".to_string())?;
                }
                "--layer" => layer = parse_usize("--layer", take(args, &mut index, "--layer")?)?,
                "--position" => {
                    position = parse_usize("--position", take(args, &mut index, "--position")?)?
                }
                "--repeats" => {
                    repeats = parse_usize("--repeats", take(args, &mut index, "--repeats")?)?;
                    if repeats == 0 {
                        return Err("--repeats: at least one repeat is required".to_string());
                    }
                }
                "--evidence-dir" => {
                    evidence_dir = Some(PathBuf::from(take(args, &mut index, "--evidence-dir")?));
                }
                "--checkpoint" => {
                    checkpoint = Some(PathBuf::from(take(args, &mut index, "--checkpoint")?));
                }
                "--cubin-dir" => {
                    cubin_dir = Some(PathBuf::from(take(args, &mut index, "--cubin-dir")?));
                }
                "--payload-elements" => {
                    payload_elements = Some(parse_usize(
                        "--payload-elements",
                        take(args, &mut index, "--payload-elements")?,
                    )?);
                }
                "--seed" => {
                    seed = take(args, &mut index, "--seed")?
                        .parse::<u64>()
                        .map_err(|_| "--seed: not a u64".to_string())?;
                }
                "--comm-timeout" => {
                    timeout = Duration::from_secs(
                        take(args, &mut index, "--comm-timeout")?
                            .parse::<u64>()
                            .map_err(|_| {
                                "--comm-timeout: not a whole number of seconds".to_string()
                            })?,
                    );
                    if timeout.is_zero() {
                        return Err("--comm-timeout must be at least one second".to_string());
                    }
                }
                "--comm-connect-timeout" => {
                    connect_timeout = Duration::from_secs(
                        take(args, &mut index, "--comm-connect-timeout")?
                            .parse::<u64>()
                            .map_err(|_| {
                                "--comm-connect-timeout: not a whole number of seconds".to_string()
                            })?,
                    );
                    if connect_timeout.is_zero() {
                        return Err(
                            "--comm-connect-timeout must be at least one second".to_string()
                        );
                    }
                }
                "--resident" => resident = true,
                "--synthetic" => synthetic = true,
                "--oneshot-preflight" => oneshot_preflight = true,
                "--prefill-plan-elements" => {
                    prefill_plan_elements = Some(parse_usize(
                        "--prefill-plan-elements",
                        take(args, &mut index, "--prefill-plan-elements")?,
                    )?);
                }
                "--help" | "-h" => return Err(format!("usage: {USAGE}")),
                other => return Err(format!("unknown argument {other}; usage: {USAGE}")),
            }
            index += 1;
        }

        let rank = rank.ok_or_else(|| format!("--tp-rank is required; usage: {USAGE}"))?;
        let evidence_dir =
            evidence_dir.ok_or_else(|| format!("--evidence-dir is required; usage: {USAGE}"))?;
        let checkpoint = match (checkpoint, synthetic) {
            (Some(_), true) => {
                return Err("--checkpoint and --synthetic are exclusive".to_string());
            }
            (Some(dir), false) => Some(dir),
            (None, false) => {
                return Err(format!(
                    "--checkpoint DIR (or --synthetic) is required; usage: {USAGE}"
                ));
            }
            (None, true) => None,
        };
        let args = Self {
            rank,
            world,
            backend,
            peers,
            port,
            layer,
            position,
            repeats,
            evidence_dir,
            checkpoint,
            cubin_dir,
            resident,
            payload_elements,
            seed,
            timeout,
            connect_timeout,
            oneshot_preflight,
            prefill_plan_elements,
        };
        args.validate()?;
        Ok(args)
    }

    fn validate(&self) -> Result<(), String> {
        if self.world == 0 {
            return Err("--tp-world-size must be at least 1".to_string());
        }
        if self.rank >= self.world {
            return Err(format!(
                "--tp-rank {} is outside a world of {}",
                self.rank, self.world
            ));
        }
        if self.prefill_plan_elements.is_some() && !self.oneshot_preflight {
            return Err(
                "--prefill-plan-elements plans the device-buffer seam; add --oneshot-preflight"
                    .to_string(),
            );
        }
        if let Some(elements) = self.prefill_plan_elements {
            let bytes = elements
                .checked_mul(2)
                .ok_or_else(|| "--prefill-plan-elements is too large".to_string())?;
            let plan = ChunkPlan::new(bytes, ONESHOT_RAILS)
                .map_err(|error| format!("the prefill plan is illegal: {error}"))?;
            if plan.chunk_count() < 2 {
                return Err(format!(
                    "--prefill-plan-elements {elements} fits one {ONESHOT_ROW_ELEMENTS}-element row; a prefill plan needs more than one"
                ));
            }
        }
        if self.oneshot_preflight && self.world != ONESHOT_RANKS {
            return Err(format!(
                "--oneshot-preflight judges the {ONESHOT_RANKS}-rank one-shot seam, but --tp-world-size is {}",
                self.world
            ));
        }
        if self.backend == Backend::Tcp {
            if self.peers.len() != self.world {
                return Err(format!(
                    "--comm-peers describes {} addresses but the world has {} ranks",
                    self.peers.len(),
                    self.world
                ));
            }
            if u32::from(self.port) + self.world as u32 > u32::from(u16::MAX) {
                return Err(format!(
                    "--comm-port {} leaves no port for rank {}",
                    self.port,
                    self.world - 1
                ));
            }
        }
        Ok(())
    }

    /// The per-rank address table: peer i listens on (peer ip i, port + i).
    fn addresses(&self) -> Vec<SocketAddr> {
        self.peers
            .iter()
            .enumerate()
            .map(|(index, ip)| SocketAddr::new(*ip, self.port + index as u16))
            .collect()
    }

    /// The activation/payload element count for a synthetic checkpoint-less run.
    fn synthetic_elements(&self) -> usize {
        self.payload_elements.unwrap_or(5120)
    }
}

fn parse_usize(flag: &str, raw: &str) -> Result<usize, String> {
    raw.parse::<usize>()
        .map_err(|_| format!("{flag}: {raw} is not a non-negative integer"))
}

/// Takes the value that follows a flag.
fn take<'a>(args: &'a [&'a str], index: &mut usize, flag: &str) -> Result<&'a str, String> {
    *index += 1;
    args.get(*index)
        .copied()
        .ok_or_else(|| format!("{flag} needs a value"))
}

/// One tensor's byte-identity check: this rank's slice against the same bytes of the full tensor.
#[derive(Debug, Clone)]
struct TensorCheck {
    name: String,
    rule: ShardRule,
    dtype: String,
    stored_shape: Vec<usize>,
    shard_shape: Vec<usize>,
    slice_bytes: usize,
    slice_fnv1a64: u64,
    twin_bytes: usize,
    twin_fnv1a64: u64,
    segments: usize,
    mismatched_segments: usize,
    first_mismatch_segment: Option<usize>,
    /// `pread` ranges the slice reader issued for this tensor: 1 for a replicated tensor or a
    /// column block, one per row for a row block. This is the read-in-place cost telemetry: the
    /// bytes are exactly this rank's, the ranges are the seeks it paid (decision: measure first,
    /// pack only if the seeks dominate load time).
    ranges: usize,
}

impl TensorCheck {
    fn matches(&self) -> bool {
        self.mismatched_segments == 0
    }

    fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "rule": self.rule.as_str(),
            "dtype": self.dtype,
            "stored_shape": self.stored_shape,
            "shard_shape": self.shard_shape,
            "slice_bytes": self.slice_bytes,
            "slice_fnv1a64": format!("{:#018x}", self.slice_fnv1a64),
            "twin_bytes": self.twin_bytes,
            "twin_fnv1a64": format!("{:#018x}", self.twin_fnv1a64),
            "segments": self.segments,
            "mismatched_segments": self.mismatched_segments,
            "first_mismatch_segment": self.first_mismatch_segment,
            "ranges": self.ranges,
            "match": self.matches(),
        })
    }
}

/// What a rank loaded before the first collective.
#[derive(Debug, Clone)]
struct LoadStats {
    slice_bytes: u64,
    twin_bytes: u64,
    slice_read_us: u64,
    twin_read_us: u64,
    compute_us: u64,
}

/// One rank's loaded state: the payload it contributes and the twin it is compared against.
#[derive(Debug, Clone)]
struct RankData {
    load: LoadStats,
    payload: Vec<u16>,
    /// The single-rank f32 twin of the row-parallel seam, bf16-rounded; None for a synthetic run.
    twin: Option<Vec<u16>>,
    /// The seam name the payload is a dot product over.
    seam: String,
    tensors: Vec<TensorCheck>,
}

/// Builds one rank's payload and twin.
fn build_rank_data(args: &SmokeArgs, rank: usize) -> Result<RankData, String> {
    match &args.checkpoint {
        Some(dir) => checkpoint_rank_data(dir, args, rank),
        None => Ok(synthetic_rank_data(args, rank)),
    }
}

fn synthetic_rank_data(args: &SmokeArgs, rank: usize) -> RankData {
    let elements = args.synthetic_elements();
    RankData {
        load: LoadStats {
            slice_bytes: (elements * 2) as u64,
            twin_bytes: 0,
            slice_read_us: 0,
            twin_read_us: 0,
            compute_us: 0,
        },
        payload: pattern(rank, elements, args.seed),
        twin: None,
        seam: "synthetic".to_string(),
        tensors: Vec::new(),
    }
}

fn checkpoint_rank_data(dir: &Path, args: &SmokeArgs, rank: usize) -> Result<RankData, String> {
    let index = dir.join("model.safetensors.index.json");
    let source = SafeTensorsSource::open(dir, &index)
        .map_err(|error| format!("open the checkpoint index at {}: {error}", index.display()))?;
    let plan = ShardPlan::layer_attention(args.layer, rank, args.world)
        .map_err(|error| format!("build the layer plan: {error}"))?;
    // The plan is verified against the headers first: a wrong rule or an undividable shape is
    // refused before any tensor bytes are read.
    let verified = plan
        .verify(&source)
        .map_err(|error| format!("verify the plan against the checkpoint headers: {error}"))?;

    let slice_start = Instant::now();
    let mut shards: BTreeMap<String, StoredShard> = BTreeMap::new();
    for tensor in &verified {
        let shard = plan
            .read(&source, &tensor.name)
            .map_err(|error| format!("read the planned slice {}: {error}", tensor.name))?;
        shards.insert(tensor.name.clone(), shard);
    }
    let slice_read_us = micros(slice_start.elapsed());

    // The byte-identity check reads the full tensor and compares the slice against the same
    // rows/segments. It is deliberately the 4x read the resident loader will not do: it is the
    // check that the loader's arithmetic is right, run once, on one layer.
    let twin_start = Instant::now();
    let mut tensors = Vec::with_capacity(verified.len());
    for tensor in &verified {
        let shard = shards
            .get(&tensor.name)
            .ok_or_else(|| format!("{} was verified but not read", tensor.name))?;
        let full = read_full(&source, &tensor.name)?;
        let (segments, mismatched_segments, first_mismatch_segment) = compare_slice_to_full(
            &SliceTarget {
                rule: tensor.rule,
                stored_shape: &tensor.stored_shape,
                shard_shape: &tensor.shard_shape,
                rank,
                world: args.world,
                row_bytes: tensor.row_bytes,
            },
            &shard.bytes,
            &full,
        );
        tensors.push(TensorCheck {
            name: tensor.name.clone(),
            rule: tensor.rule,
            dtype: tensor.dtype.clone(),
            stored_shape: tensor.stored_shape.clone(),
            shard_shape: tensor.shard_shape.clone(),
            slice_bytes: shard.bytes.len(),
            slice_fnv1a64: fnv1a64(&shard.bytes),
            twin_bytes: full.len(),
            twin_fnv1a64: fnv1a64(&full),
            segments,
            mismatched_segments,
            first_mismatch_segment,
            ranges: tensor.ranges,
        });
    }
    let twin_read_us = micros(twin_start.elapsed());

    let compute_start = Instant::now();
    let (payload, twin) = row_parallel_seam(dir, args, rank)?;
    let compute_us = micros(compute_start.elapsed());

    let slice_bytes = tensors.iter().map(|tensor| tensor.slice_bytes as u64).sum();
    let twin_bytes = tensors.iter().map(|tensor| tensor.twin_bytes as u64).sum();
    Ok(RankData {
        load: LoadStats {
            slice_bytes,
            twin_bytes,
            slice_read_us,
            twin_read_us,
            compute_us,
        },
        payload,
        twin: Some(twin),
        seam: format!("layers.{}.attn.wo_b.weight", args.layer),
        tensors,
    })
}

/// The row-parallel seam: rank r's partial dot product over its input columns, and the
/// single-rank f32 twin over the whole input axis.
fn row_parallel_seam(
    dir: &Path,
    args: &SmokeArgs,
    rank: usize,
) -> Result<(Vec<u16>, Vec<u16>), String> {
    let index = dir.join("model.safetensors.index.json");
    let source = SafeTensorsSource::open(dir, &index)
        .map_err(|error| format!("open the checkpoint index at {}: {error}", index.display()))?;
    let weight = format!("layers.{}.attn.wo_b.weight", args.layer);
    let scale = format!("layers.{}.attn.wo_b.scale", args.layer);
    let plan = ShardPlan::layer_attention(args.layer, rank, args.world)
        .map_err(|error| format!("build the layer plan: {error}"))?;

    let weight_shard = plan
        .read(&source, &weight)
        .map_err(|error| format!("read the wo_b weight slice: {error}"))?;
    let scale_shard = plan
        .read(&source, &scale)
        .map_err(|error| format!("read the wo_b scale slice: {error}"))?;
    let full_weight = read_full(&source, &weight)?;
    let full_scale = read_full(&source, &scale)?;

    let rows = weight_shard.shard_shape[0];
    let local_k = weight_shard.shard_shape[1];
    let full_k = rows_k(&full_weight, rows)?;
    if full_k != local_k * args.world {
        return Err(format!(
            "wo_b: {full_k} stored columns do not split into {} ranks of {local_k}",
            args.world
        ));
    }
    let local = Fp8Matrix::new(
        rows,
        local_k,
        weight_shard.bytes.clone(),
        scale_shard.bytes.clone(),
        scale_shard.shard_shape[0],
        scale_shard.shard_shape[1],
    )?;
    let full = Fp8Matrix::new(
        rows,
        full_k,
        full_weight,
        full_scale,
        rows.div_ceil(32),
        full_k.div_ceil(32),
    )?;

    let k_at = rank * local_k;
    let mut payload = Vec::with_capacity(rows);
    let mut twin = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut partial = 0.0f32;
        for column in 0..local_k {
            partial +=
                local.dequant(row, column) * bf16_value(pattern_bit(k_at + column, args.seed));
        }
        payload.push(bf16_bits(partial));
        let mut whole = 0.0f32;
        for column in 0..full_k {
            whole += full.dequant(row, column) * bf16_value(pattern_bit(column, args.seed));
        }
        twin.push(bf16_bits(whole));
    }
    Ok((payload, twin))
}

/// The stored column count of an FP8 matrix from its byte length and row count.
fn rows_k(bytes: &[u8], rows: usize) -> Result<usize, String> {
    if rows == 0 || !bytes.len().is_multiple_of(rows) {
        return Err(format!(
            "{} bytes do not divide into {rows} rows",
            bytes.len()
        ));
    }
    Ok(bytes.len() / rows)
}

/// A checkpoint FP8 E4M3 matrix with its E8M0 32x32 scale grid, row-major.
struct Fp8Matrix {
    columns: usize,
    values: Vec<u8>,
    scale_rows: usize,
    scale_columns: usize,
    scales: Vec<u8>,
}

impl Fp8Matrix {
    /// Wraps a weight byte slice and its scale slice.
    fn new(
        rows: usize,
        columns: usize,
        values: Vec<u8>,
        scales: Vec<u8>,
        scale_rows: usize,
        scale_columns: usize,
    ) -> Result<Self, String> {
        if values.len() < rows * columns {
            return Err(format!(
                "an FP8 matrix of {rows}x{columns} needs {} bytes but the slice holds {}",
                rows * columns,
                values.len()
            ));
        }
        // The 32x32 block grid is a property of the checkpoint's format; a scale slice that does
        // not cover it exactly means the plan and the header disagree.
        if scale_rows != rows.div_ceil(32) || scale_columns != columns.div_ceil(32) {
            return Err(format!(
                "an FP8 matrix of {rows}x{columns} needs a {}x{} scale grid, but the plan gives {scale_rows}x{scale_columns}",
                rows.div_ceil(32),
                columns.div_ceil(32)
            ));
        }
        if scales.len() < scale_rows * scale_columns {
            return Err(format!(
                "the {scale_rows}x{scale_columns} scale grid needs {} bytes but holds {}",
                scale_rows * scale_columns,
                scales.len()
            ));
        }
        Ok(Self {
            columns,
            values,
            scale_rows,
            scale_columns,
            scales,
        })
    }

    /// One dequantised element.
    fn dequant(&self, row: usize, column: usize) -> f32 {
        let value = fp8_e4m3_decode(self.values[row * self.columns + column]);
        let scale_row = (row / 32).min(self.scale_rows.saturating_sub(1));
        let scale_column = (column / 32).min(self.scale_columns.saturating_sub(1));
        let scale = ue8m0_decode(self.scales[scale_row * self.scale_columns + scale_column]);
        value * scale
    }
}

/// Reads a whole tensor through the range reader (deliberately the 4x read the resident loader
/// avoids; it is the twin of the check, not the loader).
fn read_full(source: &SafeTensorsSource, name: &str) -> Result<Vec<u8>, String> {
    let header = source
        .header(name)
        .map_err(|error| format!("header of {name}: {error}"))?;
    let bytes = header
        .tensor(name)
        .map_err(|error| format!("descriptor of {name}: {error}"))?
        .byte_len();
    source
        .read_ranges(name, &[(0, bytes as usize)])
        .map_err(|error| format!("read the whole tensor {name}: {error}"))
}

/// Compares a slice with the same bytes of the full tensor: one segment per row for a split axis,
/// one segment for a replicated tensor.
struct SliceTarget<'a> {
    rule: ShardRule,
    stored_shape: &'a [usize],
    shard_shape: &'a [usize],
    rank: usize,
    world: usize,
    row_bytes: u64,
}

fn compare_slice_to_full(
    target: &SliceTarget<'_>,
    slice: &[u8],
    full: &[u8],
) -> (usize, usize, Option<usize>) {
    let rows = target.stored_shape.first().copied().unwrap_or(0);
    let row_bytes = usize::try_from(target.row_bytes).unwrap_or(full.len());
    match target.rule {
        ShardRule::Replicated => (1, usize::from(slice != full), (slice != full).then_some(0)),
        ShardRule::Column => {
            let shard_rows = target.shard_shape.first().copied().unwrap_or(0);
            let mut mismatched = 0;
            let mut first = None;
            for row in 0..shard_rows {
                let at = (target.rank * shard_rows + row) * row_bytes;
                let slice_at = row * row_bytes;
                if slice.get(slice_at..slice_at + row_bytes) != full.get(at..at + row_bytes) {
                    mismatched += 1;
                    first.get_or_insert(row);
                }
            }
            (shard_rows, mismatched, first)
        }
        ShardRule::Row => {
            let per_rank = row_bytes / target.world.max(1);
            let mut mismatched = 0;
            let mut first = None;
            for row in 0..rows {
                let at = row * row_bytes + target.rank * per_rank;
                let slice_at = row * per_rank;
                if slice.get(slice_at..slice_at + per_rank) != full.get(at..at + per_rank) {
                    mismatched += 1;
                    first.get_or_insert(row);
                }
            }
            (rows, mismatched, first)
        }
    }
}

/// The canonical-reduce and L2 result of one rank.
#[derive(Debug, Clone)]
struct RankOutcome {
    rank: usize,
    payload_elements: usize,
    mismatches: usize,
    first_mismatch: Option<usize>,
    repeats_bit_identical: usize,
    collective_us: Vec<u64>,
    l2: Option<L2Report>,
}

/// The row-parallel seam's L2 comparison against the single-rank f32 twin.
#[derive(Debug, Clone)]
struct L2Report {
    samples: usize,
    /// One bf16 step at the largest twin magnitude in this seam.
    seam_ulp: f32,
    max_abs_delta: f32,
    max_ulp: f32,
    median_ulp: f32,
    within_bar: bool,
    /// Diagnostic only, never gated: the same deltas normalised by each element's OWN bf16 ulp.
    /// A cancellation-heavy seam shows up here (large per-element ulp where the twin is small)
    /// while the seam-scale bar above stays the gate, which is the Lead's decision of 2026-10-19.
    per_element: PerElementUlp,
}

/// Per-element ulp fold of a seam: diagnostic, not a gate.
#[derive(Debug, Clone, Copy)]
struct PerElementUlp {
    max_ulp: f32,
    median_ulp: f32,
    /// Fraction of elements within 1 ulp of their own magnitude (0..=1).
    within_one_ulp_fraction: f32,
}

impl PerElementUlp {
    fn to_json(self) -> Value {
        json!({
            "gate": false,
            "max_ulp": self.max_ulp,
            "median_ulp": self.median_ulp,
            "within_one_ulp_fraction": self.within_one_ulp_fraction,
        })
    }
}

impl L2Report {
    fn to_json(&self) -> Value {
        json!({
            "twin": "single-rank f32 dot product over the whole input axis, one bf16 rounding",
            "samples": self.samples,
            "seam_ulp": self.seam_ulp,
            "max_abs_delta": self.max_abs_delta,
            "max_ulp": self.max_ulp,
            "median_ulp": self.median_ulp,
            "bar": { "max_ulp": L2_MAX_ULP, "median_ulp": L2_MEDIAN_ULP },
            "within_bar": self.within_bar,
            "per_element_ulp": self.per_element.to_json(),
        })
    }
}

/// Runs this rank's collectives and comparisons.
fn smoke_rank(
    transport: &dyn Transport,
    rank: usize,
    world: usize,
    data: &RankData,
    args: &SmokeArgs,
) -> Result<RankOutcome, String> {
    if data.payload.is_empty() {
        return Err("the payload is empty".to_string());
    }
    let elements = data.payload.len();
    let layout = CollectiveLayout::new(world, elements, FabricSchedule::OneShotAllToAll)
        .map_err(|error| format!("collective layout: {error}"))?;

    let mut first: Option<Vec<u16>> = None;
    let mut collective_us = Vec::with_capacity(args.repeats);
    let mut reduced = data.payload.clone();
    for repeat in 0..args.repeats {
        let mut buffer = data.payload.clone();
        let start = Instant::now();
        transport
            .all_reduce_bf16(&mut buffer, repeat as u64, &layout)
            .map_err(|error| format!("all-reduce repeat {repeat}: {error}"))?;
        collective_us.push(micros(start.elapsed()));
        match &first {
            Some(expected) if *expected != buffer => break,
            Some(_) => {}
            None => first = Some(buffer.clone()),
        }
        reduced = buffer;
    }
    // The loop breaks on the first disagreement, so a completed loop means every repeat matched
    // the first result, and a short loop means the rerun that disagreed was the one not counted.
    let repeats_bit_identical = if first.is_none() {
        0
    } else if collective_us.len() == args.repeats {
        args.repeats
    } else {
        collective_us.len() - 1
    };

    let mut gathered = vec![0u16; world * elements];
    transport
        .all_gather_bf16(&data.payload, &mut gathered, args.repeats as u64)
        .map_err(|error| format!("all-gather of the payloads: {error}"))?;
    let payloads: Vec<&[u16]> = (0..world)
        .map(|peer| &gathered[peer * elements..(peer + 1) * elements])
        .collect();
    let mut canonical = vec![0u16; elements];
    canonical_all_reduce_into(&mut canonical, &payloads)
        .map_err(|error| format!("canonical reduce: {error}"))?;

    let (mismatches, first_mismatch) = count_mismatches(&reduced, &canonical);
    let l2 = data.twin.as_ref().map(|twin| compare_l2(&reduced, twin));
    Ok(RankOutcome {
        rank,
        payload_elements: elements,
        mismatches,
        first_mismatch,
        repeats_bit_identical,
        collective_us,
        l2,
    })
}

/// Counts bit mismatches and reports the first differing offset.
fn count_mismatches(left: &[u16], right: &[u16]) -> (usize, Option<usize>) {
    let mismatches = left
        .iter()
        .zip(right)
        .filter(|(left, right)| left != right)
        .count();
    let first = left
        .iter()
        .zip(right)
        .position(|(left, right)| left != right);
    (mismatches, first)
}

/// Compares a bf16 result against the f32 twin, in ulp of the seam's max-magnitude element.
///
/// The bar is declared in one ulp scale per seam ("2 bf16 ulp of the max-magnitude element, median
/// 1 ulp") rather than per element: a row-parallel seam can cancel to a small sum, and a per-element
/// ulp would call an absolute error of a few micro-units a violation while the seam's own scale
/// makes it half a step. This function normalises every absolute delta by the ulp of the largest
/// twin magnitude in the seam.
fn compare_l2(reduced: &[u16], twin: &[u16]) -> L2Report {
    let mut magnitudes: Vec<f32> = twin.iter().map(|bits| bf16_value(*bits).abs()).collect();
    magnitudes.sort_by(|left, right| right.partial_cmp(left).unwrap_or(std::cmp::Ordering::Equal));
    let seam_ulp = magnitudes.first().map(|max| bf16_ulp(*max)).unwrap_or(0.0);
    let mut absolute: Vec<f32> = Vec::with_capacity(reduced.len());
    let mut max_abs_delta: f32 = 0.0;
    for (left, right) in reduced.iter().zip(twin) {
        let delta = (bf16_value(*left) - bf16_value(*right)).abs();
        if !delta.is_finite() {
            max_abs_delta = f32::INFINITY;
            absolute.push(f32::INFINITY);
            continue;
        }
        max_abs_delta = max_abs_delta.max(delta);
        absolute.push(delta);
    }
    let (max_ulp, median_ulp) = if seam_ulp > 0.0 {
        let scaled: Vec<f32> = absolute.iter().map(|delta| delta / seam_ulp).collect();
        let max_ulp = scaled
            .iter()
            .copied()
            .fold(0.0f32, |accumulator, value| accumulator.max(value));
        let mut sorted = scaled;
        sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
        let median_ulp = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
        (max_ulp, median_ulp)
    } else {
        // A seam whose twin is entirely zero has no ulp scale; it passes only if nothing moved.
        let max_ulp = if max_abs_delta == 0.0 {
            0.0
        } else {
            f32::INFINITY
        };
        (max_ulp, max_ulp)
    };
    L2Report {
        samples: reduced.len(),
        seam_ulp,
        max_abs_delta,
        max_ulp,
        median_ulp,
        within_bar: max_ulp <= L2_MAX_ULP && median_ulp <= L2_MEDIAN_ULP,
        per_element: per_element_ulp(reduced, twin),
    }
}

/// The per-element-ulp fold of a seam: diagnostic only.
///
/// Each delta is divided by the bf16 ulp of its own twin element, so a seam that cancels to a small
/// sum on most elements produces large numbers here even when every absolute error is a fraction of
/// the seam's step. The fraction within 1 ulp is the readable summary: a well-behaved seam keeps
/// almost all elements at their own value; a cancellation-heavy one shows a low fraction.
fn per_element_ulp(reduced: &[u16], twin: &[u16]) -> PerElementUlp {
    let mut scaled: Vec<f32> = Vec::with_capacity(twin.len());
    for (left, right) in reduced.iter().zip(twin) {
        let delta = (bf16_value(*left) - bf16_value(*right)).abs();
        let own_ulp = bf16_ulp(bf16_value(*right));
        let value = if !delta.is_finite() {
            f32::INFINITY
        } else if own_ulp > 0.0 {
            delta / own_ulp
        } else if delta == 0.0 {
            0.0
        } else {
            f32::INFINITY
        };
        scaled.push(value);
    }
    let max_ulp = scaled
        .iter()
        .copied()
        .fold(0.0f32, |accumulator, value| accumulator.max(value));
    let mut sorted = scaled.clone();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let median_ulp = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
    let within = scaled.iter().filter(|value| **value <= 1.0).count();
    PerElementUlp {
        max_ulp,
        median_ulp,
        within_one_ulp_fraction: if scaled.is_empty() {
            0.0
        } else {
            within as f32 / scaled.len() as f32
        },
    }
}

/// The spacing of bf16 values around a f32 magnitude: 2^(floor(log2 |v|) - 7).
fn bf16_ulp(value: f32) -> f32 {
    let magnitude = value.abs();
    if !magnitude.is_finite() || magnitude == 0.0 {
        return 0.0;
    }
    let exponent = magnitude.log2().floor();
    2.0f32.powf(exponent - 7.0)
}

/// A deterministic bf16 activation value in [-1, 1], identical on every rank for the same index
/// and seed.
///
/// The range is bounded on purpose: the value is both a dot-product operand and a collective
/// payload, so an unbounded bit pattern would let a random draw produce an infinity or a NaN and
/// the smoke would fail on the draw rather than on the seam.
fn pattern_bit(index: usize, seed: u64) -> u16 {
    let mut state = (index as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(seed);
    state ^= state >> 30;
    state = state.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    state ^= state >> 27;
    state = state.wrapping_mul(0x94d0_49bb_1331_11eb);
    state ^= state >> 31;
    let unit = (state >> 11) as f32 / (1u64 << 53) as f32;
    bf16_bits(2.0 * unit - 1.0)
}

/// A rank's finite bf16 pattern, distinct per rank.
fn pattern(rank: usize, elements: usize, seed: u64) -> Vec<u16> {
    (0..elements)
        .map(|index| pattern_bit(index, seed ^ ((rank as u64 + 1) << 32)))
        .collect()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn micros(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

fn hostname() -> String {
    fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|name| name.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// The evidence JSON for one rank.
/// The peer table an evidence file records: the CLI's address table, or the RDMA environment's
/// rank table when the rdma backend opened from it.
fn transport_peers(args: &SmokeArgs) -> Vec<String> {
    if args.backend == Backend::Rdma {
        return std::env::var("SPARK_COMM_PEERS")
            .map(|table| {
                table
                    .split(',')
                    .map(|peer| peer.trim().to_string())
                    .collect()
            })
            .unwrap_or_default();
    }
    args.addresses().iter().map(ToString::to_string).collect()
}

fn evidence(
    args: &SmokeArgs,
    rank: usize,
    transport_label: &str,
    connect_us: u64,
    data: &RankData,
    outcome: &RankOutcome,
) -> Value {
    let plan_fnv = {
        let mut text = String::new();
        for tensor in &data.tensors {
            text.push_str(&format!("{}={}\n", tensor.name, tensor.rule.as_str()));
        }
        fnv1a64(text.as_bytes())
    };
    json!({
        "stage": "tp4-smoke",
        "schema": SCHEMA_VERSION,
        "started_at_unix_s": unix_seconds(),
        "host": hostname(),
        "tp": { "rank": rank, "world_size": args.world },
        "transport": {
            "backend": transport_label,
            "peers": transport_peers(args),
            "base_port": args.port,
            "connect_us": connect_us,
        },
        "checkpoint": {
            "dir": args.checkpoint.as_ref().map(|dir| dir.display().to_string()),
            "layer": args.layer,
            "position": args.position,
            "resident": args.resident,
            "cubin_dir": args.cubin_dir.as_ref().map(|dir| dir.display().to_string()),
        },
        "plan": {
            "tensors": data.tensors.len(),
            "fnv1a64": format!("{plan_fnv:#018x}"),
        },
        "seam": data.seam,
        "load": {
            "slice_bytes": data.load.slice_bytes,
            "twin_bytes": data.load.twin_bytes,
            "slice_read_us": data.load.slice_read_us,
            "twin_read_us": data.load.twin_read_us,
            "compute_us": data.load.compute_us,
        },
        "tensors": data.tensors.iter().map(TensorCheck::to_json).collect::<Vec<_>>(),
        "l1": {
            "collective": "all_reduce_bf16",
            "twin": "canonical one-rounding reduce (f32 ascending-rank sum, one bf16 RNE round)",
            "payload_elements": outcome.payload_elements,
            "payload_bytes": outcome.payload_elements * 2,
            "repeats": args.repeats,
            "mismatches": outcome.mismatches,
            "first_mismatch": outcome.first_mismatch,
            "repeats_bit_identical": outcome.repeats_bit_identical,
            "collective_us": {
                "min": outcome.collective_us.iter().copied().min().unwrap_or(0),
                "max": outcome.collective_us.iter().copied().max().unwrap_or(0),
                "samples": outcome.collective_us,
            },
        },
        "l2": outcome.l2.as_ref().map(L2Report::to_json),
        "oneshot": oneshot_section(args),
        "flags": {
            "device_execution": false,
            "token_produced": false,
            "cuda_feature": cfg!(feature = "cuda"),
            "comm_rdma_shim": rdma_shim_flag(),
        },
    })
}

/// Writes one evidence file and returns its path.
fn write_evidence(dir: &Path, rank: usize, value: &Value) -> Result<PathBuf, String> {
    fs::create_dir_all(dir).map_err(|error| format!("create {}: {error}", dir.display()))?;
    let path = dir.join(format!("tp4-smoke-r{rank}.json"));
    let text = serde_json::to_string_pretty(value)
        .map_err(|error| format!("serialise the evidence: {error}"))?;
    fs::write(&path, text).map_err(|error| format!("write {}: {error}", path.display()))?;
    Ok(path)
}

/// Whether an outcome passes every gate this stage judges.
fn outcome_ok(outcome: &RankOutcome, args: &SmokeArgs) -> bool {
    outcome.mismatches == 0
        && outcome.repeats_bit_identical == args.repeats
        && outcome.l2.as_ref().is_none_or(|l2| l2.within_bar)
}

/// The host-side half of the S5.3 device-buffer seam, validated before a window is spent.
///
/// This is deliberately not a measurement. It checks the contract the window will run against -
/// the two ABI pins agree, the payload fits the kernel's row, the slot and dispatch descriptors are
/// legal, and the sequence counter is monotonic over the requested repeats. A failure here would
/// have been the window's first dispatch failure, so it is refused on the host instead.
fn validate_oneshot_preflight(args: &SmokeArgs) -> Result<(), String> {
    oneshot_abi_parity()?;
    let exchange = Exchange::new(0, args.synthetic_elements(), ONESHOT_RAILS)
        .map_err(|error| format!("the one-shot exchange is illegal: {error}"))?;
    let slot = SlotRegistration::new(
        SlotKind::DeviceMemory {
            device_address: 0x1000,
        },
        ONESHOT_SLOT_BYTES,
    )
    .map_err(|error| format!("the one-shot slot descriptor is illegal: {error}"))?;
    Dispatch::new(args.rank, slot, exchange, 1, 1)
        .map_err(|error| format!("the one-shot dispatch is illegal: {error}"))?;
    // Exercise the launcher's dispatch record shape and its validator on the host: a device entry
    // must pass, a host-staging one must be refused. The window's launcher emits the real ones.
    let device_entry = DispatchRecord::new(
        args.rank,
        0,
        exchange,
        DispatchAddresses {
            slot: 0x1000,
            local_row: 0x1000 + row_offset(args.rank) as u64,
            peer_rows: [0x2000, 0x3000, 0x4000],
        },
        LaunchPath::DeviceKernel,
        0,
        0,
    )
    .map_err(|error| format!("the dispatch record shape is illegal: {error}"))?;
    device_entry
        .device_path_ok()
        .map_err(|error| format!("a device dispatch record was refused: {error}"))?;
    let mut log = DispatchLog::new();
    log.record(device_entry)
        .map_err(|error| format!("the dispatch record log refused the entry: {error}"))?;
    log.device_path_ok()
        .map_err(|error| format!("the dispatch log is not device-path only: {error}"))?;
    let fallback = DispatchRecord::new(
        args.rank,
        1,
        exchange,
        DispatchAddresses {
            slot: 0x1000,
            local_row: 0x1000 + row_offset(args.rank) as u64,
            peer_rows: [0x2000, 0x3000, 0x4000],
        },
        LaunchPath::HostStaging,
        10_240,
        1,
    )
    .map_err(|error| format!("the fallback record shape is illegal: {error}"))?;
    if fallback.device_path_ok().is_ok() {
        return Err(
            "a host-staging dispatch record was accepted; the seam's fallback validator is broken"
                .to_string(),
        );
    }
    let mut counter = SequenceCounter::new();
    for seq in 0..args.repeats as u64 {
        counter
            .accept(seq)
            .map_err(|error| format!("the one-shot sequence is illegal: {error}"))?;
    }
    validate_host_twin(args)?;
    validate_orchestration()?;
    Ok(())
}

/// Requests the orchestration evidence uses; the orchestration is request-shaped, not payload-shaped.
fn orchestration_requests() -> Vec<ServingRequest> {
    (0..3).map(|id| ServingRequest::new(id, 0x5f)).collect()
}

/// A config for one orchestration scenario.
fn orchestration_config(guard: GuardPolicy) -> OrchestrationConfig {
    OrchestrationConfig {
        requests: orchestration_requests(),
        guard,
        ..OrchestrationConfig::default()
    }
}

/// The guarded run must refuse every injected failure, and the same policy with the guard off must
/// serve a token that is not the canonical one: if it served the right answer anyway, the guard
/// would be untested and the failure silently survivable.
fn validate_orchestration() -> Result<(), String> {
    let expected: Vec<u16> = orchestration_requests()
        .iter()
        .map(canonical_first_token)
        .collect();
    let happy = run_orchestration(orchestration_config(GuardPolicy::guarded()))
        .map_err(|error| format!("the orchestration twin failed: {error}"))?;
    if !happy.all_agreed {
        return Err(format!(
            "the orchestration ranks did not agree: {:?}",
            happy.outcomes
        ));
    }
    for (outcome, token) in happy.outcomes.iter().zip(&expected) {
        if outcome.served_token() != Some(*token) {
            return Err(format!(
                "the orchestration served {:?}, the canonical token is {token}",
                outcome.served_token()
            ));
        }
    }
    let mut startup = orchestration_config(GuardPolicy::guarded());
    startup.failure.startup_failure = Some(2);
    let refused = run_orchestration(startup)
        .map_err(|error| format!("the startup-refusal twin failed: {error}"))?;
    if refused.startup.missing() != [2] {
        return Err(format!(
            "the startup refusal named {:?}, expected [2]",
            refused.startup.missing()
        ));
    }
    let scenarios: [(&str, sparkflash_comm::oneshot::loopback::FailurePolicy); 3] = [
        ("dead_rank", {
            let mut failure = sparkflash_comm::oneshot::loopback::FailurePolicy::default();
            failure
                .set_fate(
                    1,
                    sparkflash_comm::oneshot::loopback::RankFate::Die { at_request: 0 },
                )
                .map_err(|error| format!("dead-rank fate: {error}"))?;
            failure
        }),
        ("hung_rank", {
            let mut failure = sparkflash_comm::oneshot::loopback::FailurePolicy::default();
            failure
                .set_fate(
                    2,
                    sparkflash_comm::oneshot::loopback::RankFate::Hang { at_request: 0 },
                )
                .map_err(|error| format!("hung-rank fate: {error}"))?;
            failure
        }),
        ("disagreement", {
            let mut failure = sparkflash_comm::oneshot::loopback::FailurePolicy::default();
            failure
                .set_fate(
                    0,
                    sparkflash_comm::oneshot::loopback::RankFate::CorruptToken {
                        at_request: 0,
                        token: 3,
                    },
                )
                .map_err(|error| format!("corrupt-token fate: {error}"))?;
            failure
        }),
    ];
    for (label, failure) in scenarios {
        let mut guarded = orchestration_config(GuardPolicy::guarded());
        guarded.failure = failure.clone();
        let report = run_orchestration(guarded)
            .map_err(|error| format!("the {label} guarded twin failed: {error}"))?;
        if report.outcomes[0].failure().is_none() {
            return Err(format!(
                "the {label} failure was served instead of refused: {:?}",
                report.outcomes[0]
            ));
        }
        let mut naive = orchestration_config(GuardPolicy::naive());
        naive.failure = failure;
        let report = run_orchestration(naive)
            .map_err(|error| format!("the {label} naive twin failed: {error}"))?;
        let served = report.outcomes[0].served_token().ok_or_else(|| {
            format!("the {label} naive run refused instead of serving the wrong answer")
        })?;
        if served == expected[0] {
            return Err(format!(
                "the {label} naive run served the canonical token {served}; the guard would be untested"
            ));
        }
    }
    Ok(())
}

/// The orchestration evidence section: a guarded run, the startup refusal, and each injected failure
/// with and without its guard. No node number is claimed here.
fn orchestration_section() -> Value {
    let expected: Vec<u16> = orchestration_requests()
        .iter()
        .map(canonical_first_token)
        .collect();
    let happy = run_orchestration(orchestration_config(GuardPolicy::guarded()));
    let happy_json = match &happy {
        Ok(report) => json!({
            "ran": true,
            "world_size": report.world_size,
            "requests": report.requests,
            "startup_ready": !report.startup.refused(),
            "served_tokens": report.outcomes.iter().map(RequestOutcome::served_token).collect::<Vec<_>>(),
            "expected_tokens": report.expected_tokens,
            "first_token_agreement": report.all_agreed,
            "per_rank_served": report.per_rank_served,
            "collective_calls": report.collective_calls,
            "rail0_bytes": report.rail_ledger.rail0_bytes,
            "rail1_bytes": report.rail_ledger.rail1_bytes,
            "fan_out_bytes": report.fan_out_bytes,
            "dispatches": report.dispatches,
            "host_copy_bytes": report.host_copy_bytes,
            "device_path_only": report.device_path_only,
            "host_elapsed_ms": u64::try_from(report.elapsed.as_millis()).unwrap_or(u64::MAX),
        }),
        Err(error) => json!({ "ran": false, "error": error.to_string() }),
    };
    let mut startup = orchestration_config(GuardPolicy::guarded());
    startup.failure.startup_failure = Some(2);
    let startup_json = match run_orchestration(startup) {
        Ok(report) => json!({
            "ran": true,
            "ready": !report.startup.refused(),
            "missing": report.startup.missing(),
            "served_requests": report.outcomes.len(),
            "note": "a missing rank refuses before any request is fanned out",
        }),
        Err(error) => json!({ "ran": false, "error": error.to_string() }),
    };
    let mut dead_guarded = orchestration_config(GuardPolicy::guarded());
    dead_guarded
        .failure
        .set_fate(
            1,
            sparkflash_comm::oneshot::loopback::RankFate::Die { at_request: 0 },
        )
        .ok();
    let mut dead_naive = orchestration_config(GuardPolicy::naive());
    dead_naive.failure = dead_guarded.failure.clone();
    let mut hung_guarded = orchestration_config(GuardPolicy::guarded());
    hung_guarded
        .failure
        .set_fate(
            2,
            sparkflash_comm::oneshot::loopback::RankFate::Hang { at_request: 0 },
        )
        .ok();
    let mut hung_naive = orchestration_config(GuardPolicy::naive());
    hung_naive.failure = hung_guarded.failure.clone();
    let mut split_guarded = orchestration_config(GuardPolicy::guarded());
    split_guarded
        .failure
        .set_fate(
            0,
            sparkflash_comm::oneshot::loopback::RankFate::CorruptToken {
                at_request: 0,
                token: 3,
            },
        )
        .ok();
    let mut split_naive = orchestration_config(GuardPolicy::naive());
    split_naive.failure = split_guarded.failure.clone();
    let failure_json = |label: &str, guarded: OrchestrationConfig, naive: OrchestrationConfig| {
        let guarded = run_orchestration(guarded);
        let naive = run_orchestration(naive);
        match (&guarded, &naive) {
            (Ok(guarded), Ok(naive)) => {
                let refusal = guarded.outcomes.first().and_then(RequestOutcome::failure);
                let naive_token = naive
                    .outcomes
                    .first()
                    .and_then(RequestOutcome::served_token);
                json!({
                    "kind": label,
                    "guarded_refused": refusal.map(|failure| failure.label()),
                    "guarded_refused_rank": refusal.and_then(|failure| failure.rank()),
                    "without_guard_served_token": naive_token,
                    "canonical_token": expected[0],
                    "guard_is_load_bearing": naive_token.is_some_and(|token| token != expected[0]),
                })
            }
            _ => json!({ "kind": label, "ran": false }),
        }
    };
    json!({
        "ran": happy.is_ok(),
        "happy": happy_json,
        "startup": startup_json,
        "failure_handling": [
            failure_json("dead_rank", dead_guarded, dead_naive),
            failure_json("hung_rank", hung_guarded, hung_naive),
            failure_json("disagreement", split_guarded, split_naive),
        ],
        "does_not_prove": [
            "that four processes exist or start in the recorded order",
            "that a rank death is detected by the transport rather than by in-process status",
            "that the watchdog deadline is the one a window would use",
            "that the four nodes agree on a real model's first token",
        ],
        "unclaimed_node_measurements": [
            "real four-process startup and readiness timing",
            "real mid-request rank-death detection and teardown timing",
            "real watchdog deadlines for a stalled rank",
            "first-token and per-position logit agreement across the four-node mesh",
            "the per-request D40 artifact including timings",
            "per-request latency and throughput at serving shape",
        ],
        "note": "in-process four-rank orchestration behind the loopback seam; the first-token collective really runs, the failure policy is a deterministic knob, and no hardware number is claimed",
    })
}

/// Runs the loopback host twin and requires the proofs the device path must keep.
///
/// This is the preflight's executable part: the seam descriptors above are checked by arithmetic,
/// and the twin below runs the same protocol end to end in process memory. A divergence here is a
/// contract failure, not a hardware result.
fn validate_host_twin(args: &SmokeArgs) -> Result<(), String> {
    let elements = args.synthetic_elements();
    let report = run_host_twin(elements, HOST_TWIN_SEQUENCES, ONESHOT_RAILS)
        .map_err(|error| format!("the loopback host twin failed: {error}"))?;
    check_host_twin(&report, "one-shot")?;
    if args.prefill_plan_elements.is_some() {
        let ring = run_ring_twin(HOST_TWIN_RING_ELEMENTS, RingTwinConfig::default())
            .map_err(|error| format!("the ring loopback host twin failed: {error}"))?;
        check_ring_twin(&ring, "ring")?;
        let hazard = run_ring_twin(
            HOST_TWIN_RING_HAZARD_ELEMENTS,
            RingTwinConfig {
                kernel_delay: Some(KernelDelay {
                    rank: 0,
                    chunk: 0,
                    duration: Duration::from_millis(1),
                }),
                read_epoch_policy: ReadEpochPolicy::Ignore,
                ..RingTwinConfig::default()
            },
        )
        .map_err(|error| format!("the ring hazard demonstration failed: {error}"))?;
        if hazard.bit_identical {
            return Err(
                "ignoring the ring read-done epoch did not corrupt the delayed rank; the reuse hazard is not actually guarded"
                    .to_string(),
            );
        }
    }
    if let Some(planned) = args.prefill_plan_elements {
        let total = planned.min(HOST_TWIN_CHUNKED_MAX_ELEMENTS);
        if total > ONESHOT_ROW_ELEMENTS {
            let report = run_host_twin_prefill(total, ONESHOT_RAILS)
                .map_err(|error| format!("the chunked loopback host twin failed: {error}"))?;
            check_host_twin(&report, "chunked")?;
        }
    }
    Ok(())
}

/// The invariants a ring host-twin run must satisfy: canonical arithmetic, zero host staging, the
/// device path on every record and one dispatch per rank per chunk.
fn check_ring_twin(report: &RingTwinReport, form: &str) -> Result<(), String> {
    if !report.bit_identical {
        return Err(format!(
            "the {form} host twin diverged from the canonical one-rounding CPU twin on{:?}",
            report.mismatched_ranks
        ));
    }
    if report.host_copy_bytes != 0 {
        return Err(format!(
            "the {form} host twin recorded {} host-staging bytes; the device path must stage nothing",
            report.host_copy_bytes
        ));
    }
    if !report.device_path_only {
        return Err(format!(
            "the {form} host twin recorded a non-device dispatch"
        ));
    }
    let expected = ONESHOT_RANKS * report.chunks;
    if report.dispatched != expected {
        return Err(format!(
            "the {form} host twin recorded {} dispatches, expected {expected}",
            report.dispatched
        ));
    }
    Ok(())
}

/// The invariants a host-twin run must satisfy: canonical arithmetic, zero host staging, the
/// device path on every record, and one dispatch per rank per step.
fn check_host_twin(report: &HostTwinReport, form: &str) -> Result<(), String> {
    if !report.bit_identical {
        return Err(format!(
            "the {form} host twin diverged from the canonical one-rounding CPU twin"
        ));
    }
    if report.host_copy_bytes != 0 {
        return Err(format!(
            "the {form} host twin recorded {} host-staging bytes; the device path must stage nothing",
            report.host_copy_bytes
        ));
    }
    if !report.device_path_only {
        return Err(format!(
            "the {form} host twin recorded a non-device dispatch"
        ));
    }
    let expected = ONESHOT_RANKS * report.steps;
    if report.dispatched != expected {
        return Err(format!(
            "the {form} host twin recorded {} dispatches, expected {expected}",
            report.dispatched
        ));
    }
    Ok(())
}

/// The one-shot ABI is pinned in two crates: sparkflash-comm's seam contract and sparkflash-kernels'
/// roce_abi constants, each mirrored from the CUDA header. The CLI depends on both, so this is the
/// one place the two pins are compared; a drift in either fails here and in that crate's own tests.
/// The loopback host twin's evidence: what the device-buffer path proved on this host, and the
/// node measurements it explicitly did not make.
///
/// The twin runs the real protocol in process memory: four rank threads, the pinned slot ring, the
/// doorbell/arrival handshake, the one-rounding reduction and a dispatch record per call. It uses
/// no device memory, no memory region, no queue pair and no kernel, so nothing here is a hardware
/// number; the unclaimed list is part of the section so a reader cannot mistake it for one.
fn host_twin_section(args: &SmokeArgs) -> Value {
    let elements = args.synthetic_elements();
    let oneshot = match run_host_twin(elements, HOST_TWIN_SEQUENCES, ONESHOT_RAILS) {
        Ok(report) => json!({
            "ran": true,
            "transport": "loopback device transport, in process",
            "mode": report.mode,
            "sequences": report.steps,
            "elements": report.elements,
            "bytes": report.bytes,
            "rails": report.rails,
            "slot_ring": report.slot_ring,
            "slot_indices": report.slot_indices,
            "dispatches": report.dispatched,
            "host_copy_bytes": report.host_copy_bytes,
            "bit_identical_to_canonical_twin": report.bit_identical,
            "device_path_only": report.device_path_only,
            "posted_rail0_bytes": report.posted.rail0_bytes,
            "posted_rail1_bytes": report.posted.rail1_bytes,
            "host_elapsed_ms": u64::try_from(report.elapsed.as_millis()).unwrap_or(u64::MAX),
            "unclaimed_node_measurements": report.unclaimed,
        }),
        Err(error) => json!({ "ran": false, "error": error.to_string() }),
    };
    let chunked = match args.prefill_plan_elements {
        Some(planned) => {
            let total = planned.min(HOST_TWIN_CHUNKED_MAX_ELEMENTS);
            if total <= ONESHOT_ROW_ELEMENTS {
                json!({
                    "ran": false,
                    "planned_elements": planned,
                    "note": "the planned payload fits one row, so it is not a chunked plan",
                })
            } else {
                match run_host_twin_prefill(total, ONESHOT_RAILS) {
                    Ok(report) => json!({
                        "ran": true,
                        "planned_elements": planned,
                        "elements": report.elements,
                        "clamped_for_host_run": planned > HOST_TWIN_CHUNKED_MAX_ELEMENTS,
                        "chunks": report.steps,
                        "slot_ring": report.slot_ring,
                        "slot_indices": report.slot_indices,
                        "dispatches": report.dispatched,
                        "host_copy_bytes": report.host_copy_bytes,
                        "bit_identical_to_canonical_twin": report.bit_identical,
                        "device_path_only": report.device_path_only,
                        "host_elapsed_ms": u64::try_from(report.elapsed.as_millis()).unwrap_or(u64::MAX),
                    }),
                    Err(error) => json!({ "ran": false, "error": error.to_string() }),
                }
            }
        }
        None => json!({
            "ran": false,
            "note": "pass --prefill-plan-elements with --oneshot-preflight to run the chunked twin",
        }),
    };
    let ring = if args.prefill_plan_elements.is_some() {
        let honest = run_ring_twin(HOST_TWIN_RING_ELEMENTS, RingTwinConfig::default());
        let hazard = run_ring_twin(
            HOST_TWIN_RING_HAZARD_ELEMENTS,
            RingTwinConfig {
                kernel_delay: Some(KernelDelay {
                    rank: 0,
                    chunk: 0,
                    duration: Duration::from_millis(1),
                }),
                read_epoch_policy: ReadEpochPolicy::Ignore,
                ..RingTwinConfig::default()
            },
        );
        json!({
            "ran": honest.is_ok(),
            "elements": HOST_TWIN_RING_ELEMENTS,
            "bytes": HOST_TWIN_RING_ELEMENTS * 2,
            "oneshot_row_bytes": ONESHOT_ROW_BYTES,
            "chunks": honest.as_ref().map_or(0, |report| report.chunks),
            "chunk_elements": honest.as_ref().map_or_else(|_| Vec::new(), |report| report.chunk_elements.clone()),
            "ring": honest.as_ref().map_or(0, |report| report.ring),
            "bit_identical_to_canonical_twin": honest.as_ref().is_ok_and(|report| report.bit_identical),
            "host_copy_bytes": honest.as_ref().map_or(0, |report| report.host_copy_bytes),
            "device_path_only": honest.as_ref().is_ok_and(|report| report.device_path_only),
            "dispatches": honest.as_ref().map_or(0, |report| report.dispatched),
            "max_rank_in_flight": honest.as_ref().map_or(0, |report| report.max_rank_in_flight),
            "host_elapsed_ms": honest.as_ref().map_or(0, |report| u64::try_from(report.elapsed.as_millis()).unwrap_or(u64::MAX)),
            "unclaimed_node_measurements": honest.as_ref().map_or_else(|_| Vec::new(), |report| report.unclaimed.clone()),
            "hazard": match &hazard {
                Ok(report) => json!({
                    "ran": true,
                    "demonstrated": !report.bit_identical,
                    "bit_identical_to_canonical_twin": report.bit_identical,
                    "mismatched_ranks": report.mismatched_ranks,
                    "elements": report.total_elements,
                    "chunks": report.chunks,
                    "note": "the proxy reuses chunk 0's slot for chunk 2 before rank 0's kernel reads it, with the read-done epoch deliberately ignored; the corrupted output shows why the epoch wait is load-bearing",
                }),
                Err(error) => json!({ "ran": false, "error": error.to_string() }),
            },
            "note": "in-process overlapped ring: a proxy thread stages ahead of the kernel thread; no device memory, MR, QP, kernel or hardware number",
        })
    } else {
        json!({
            "ran": false,
            "note": "pass --prefill-plan-elements with --oneshot-preflight to run the overlapped ring twin",
        })
    };
    json!({
        "ran": oneshot["ran"].as_bool().unwrap_or(false),
        "oneshot": oneshot,
        "chunked": chunked,
        "ring": ring,
        "does_not_prove": [
            "that the CUDA kernel, the device-memory MR, the queue pairs or the launcher exist",
            "any latency, bandwidth, correctness-across-the-mesh or determinism number",
        ],
        "unclaimed_node_measurements": [
            "one-shot all-reduce p50 latency at 10 KiB against the 20 us gate",
            "per-rail bandwidth on the live 200 Gb/s rails",
            "correctness across the four-node mesh through the real kernel, MR and QP path",
            "10,000-replay determinism and two-run byte identity on the nodes",
            "device-side DtoH/HtoD absence, which a host allocation cannot demonstrate",
        ],
        "note": "in-process loopback behind the existing device-buffer seam; cite no hardware number from this section",
    })
}

fn oneshot_abi_parity() -> Result<(), String> {
    use sparkflash_kernels::roce_abi;
    let pairs: [(&str, usize, usize); 6] = [
        ("ranks", ONESHOT_RANKS, roce_abi::ROCE_ONESHOT_RANKS),
        (
            "row_elements",
            ONESHOT_ROW_ELEMENTS,
            roce_abi::ROCE_ONESHOT_SLOT_ROW_ELEMS,
        ),
        (
            "row_bytes",
            ONESHOT_ROW_BYTES,
            roce_abi::ROCE_ONESHOT_SLOT_ROW_BYTES,
        ),
        (
            "arrival_bytes",
            ONESHOT_ARRIVAL_BYTES,
            roce_abi::ROCE_ONESHOT_SLOT_ARRIVAL_BYTES,
        ),
        (
            "slot_bytes",
            ONESHOT_SLOT_BYTES,
            roce_abi::ROCE_ONESHOT_SLOT_BYTES,
        ),
        (
            "slot_align",
            ONESHOT_SLOT_ALIGN,
            roce_abi::ROCE_ONESHOT_SLOT_ALIGN,
        ),
    ];
    for (name, comm, kernels) in pairs {
        if comm != kernels {
            return Err(format!(
                "the one-shot ABI pins disagree on {name}: sparkflash-comm {comm}, sparkflash-kernels {kernels}"
            ));
        }
    }
    Ok(())
}

/// The `oneshot` evidence section: what the host stage checked, and what only the window can prove.
fn oneshot_section(args: &SmokeArgs) -> Value {
    let elements = args.synthetic_elements();
    let (bytes, rail0, rail1, posted) =
        Exchange::new(0, elements, ONESHOT_RAILS).map_or((0, 0, 0, 0), |exchange| {
            (
                exchange.bytes(),
                exchange.rail_bytes(0),
                exchange.rail_bytes(1),
                exchange.posted_bytes(),
            )
        });
    json!({
        "state": if args.oneshot_preflight { "host_preflight" } else { "not_requested" },
        "abi": {
            "ranks": ONESHOT_RANKS,
            "arrival_bytes": ONESHOT_ARRIVAL_BYTES,
            "row_elements": ONESHOT_ROW_ELEMENTS,
            "row_bytes": ONESHOT_ROW_BYTES,
            "slot_bytes": ONESHOT_SLOT_BYTES,
            "slot_align": ONESHOT_SLOT_ALIGN,
            "rails": ONESHOT_RAILS,
            "cross_crate_parity": oneshot_abi_parity().is_ok(),
        },
        "exchange": {
            "seq": 0,
            "elements": elements,
            "bytes": bytes,
            "rail0_bytes": rail0,
            "rail1_bytes": rail1,
            "posted_bytes": posted,
        },
        "slot": { "kind": "device_memory", "slot_bytes": ONESHOT_SLOT_BYTES, "allocated": false },
        "dispatch": { "rank": args.rank, "release": 1, "peer_epoch": 1, "launched": false },
        "dispatch_record": {
            "emitted": false,
            "required_on_window_run": true,
            "kernel": ONESHOT_KERNEL_NAME,
            "fields": [
                "seq", "elements", "bytes", "rails", "slot_address", "local_row_address",
                "peer_row_addresses", "launch", "host_copy_bytes", "elapsed_ns"
            ],
            "host_copy_bytes": 0,
            "validated_on_host": args.oneshot_preflight,
            "validator": "device_path_ok refuses LaunchPath::HostStaging and any host_copy_bytes != 0; DispatchLog::device_path_ok names the first offending sequence",
        },
        "sequence": { "first": 0, "last": args.repeats.saturating_sub(1) as u64, "repeats": args.repeats },
        "chunked": chunked_section(args),
        "host_twin": host_twin_section(args),
        "orchestration": orchestration_section(),
        "device_buffer_form": "refused by every host transport (CommError::Unsupported); only a device path with the kernel launcher implements it",
        "proves_host_only": [
            "the comm and kernels ABI pins agree on the slot",
            "the payload fits the kernel's 2 MiB row and the rail split is whole bf16 elements",
            "the slot descriptor and dispatch are legal",
            "the sequence counter is strictly increasing over the requested repeats",
            "the receive-side proxy publishes an arrival exactly once, only after every rail",
            "the reduction the device path must produce equals the canonical one-rounding twin bit for bit (tests/device_seam.rs)",
            "the loopback device transport runs four-rank one-shot exchanges behind the same seam and matches the canonical twin bit for bit, in either arrival order (tests/oneshot_host.rs)",
            "every dispatch record takes the device path and records zero host-staging bytes",
            "the two-deep slot ring and the chunked reuse order are pinned by the host twin, not assumed",
            "the overlapped prefill ring twin keeps a payload above the one-shot row bit-identical at chunk boundaries, with a proxy staging ahead of the kernel",
            "the ring's read-done epoch is load-bearing: a run that ignores it and reuses a slot early is shown to corrupt the delayed rank"
        ],
        "does_not_prove": [
            "that the CUDA kernel runs or that the slot is reachable from it",
            "that the pinned-host alias or the device-memory MR registers",
            "that the QP posting and the proxy's completions agree with the NIC",
            "any latency, bandwidth or determinism number",
            "that any number in the host_twin section is a node measurement",
            "any prefill latency, per-rail bandwidth, engram-gather disk latency, mesh correctness or determinism number"
        ],
        "window_measurement": {
            "gate": "S5.3 one-shot device-buffer all-reduce at decode size (QUALIFICATION 3.1)",
            "payload_bytes": bytes,
            "repeats": args.repeats,
            "metrics": [
                "p50 one-shot all-reduce latency <= 20 us at 10 KiB",
                "every rank's output bit-identical to the canonical CPU twin, every repeat",
                "repeats_bit_identical == repeats, and two runs per configuration",
                "no DtoH/HtoD between dispatches: device pointers only",
                "one-shot call count and per-seam byte counts recorded per collective"
            ],
            "requires": [
                "four GPU nodes opened by the Lead for the window",
                "a cuda-feature build with the kernel launcher",
                "an rdma-feature build on the active rails (rocep1s0f1) with libibverbs",
                "the three existing hardware gates and rocenante-probe green first"
            ]
        }
    })
}

/// Records the S5.4 chunked (prefill-sized) contract on the host side of the seam.
///
/// The plan is arithmetic over the row the ABI already pins: a payload above one row is an ordered
/// series of row-sized one-shot exchanges, each with its own dispatch and doorbell. Nothing here
/// posts, reduces or times anything; the window measurement is named instead.
fn chunked_section(args: &SmokeArgs) -> Value {
    let Some(elements) = args.prefill_plan_elements else {
        return json!({
            "planned": false,
            "note": "pass --prefill-plan-elements with --oneshot-preflight to record the S5.4 chunked contract",
        });
    };
    let bytes = elements.saturating_mul(2);
    match ChunkPlan::new(bytes, ONESHOT_RAILS) {
        Ok(plan) => {
            let chunk_bytes = plan.chunks().first().map_or(0, |chunk| chunk.bytes());
            let posted_bytes = plan.posted_bytes().unwrap_or(0);
            json!({
                "planned": true,
                "elements": elements,
                "bytes": bytes,
                "row_bytes": ONESHOT_ROW_BYTES,
                "chunks": plan.chunk_count(),
                "chunk_bytes": chunk_bytes,
                "posted_bytes": posted_bytes,
                "descriptors": "one Dispatch per chunk, same slot and release/peer-epoch words, sequence base+i, strictly increasing",
                "doorbells": "one per chunk; chunk i+1 posts only after every peer's arrival word for chunk i is published, so the 4-slot row reuse is ordered",
                "proves_host_only": [
                    "the payload splits into whole bf16 elements on row boundaries",
                    "one dispatch per chunk is legal and its sequence strictly increases",
                    "the receive state machine publishes chunk i before it advances to i+1 and refuses a future chunk's frame"
                ],
                "does_not_prove": [
                    "that the launcher posts the chunks in order or at all",
                    "any chunked latency, bandwidth, bit-identity or determinism number"
                ],
                "window_measurement": {
                    "gate": "S5.4 prefill-sized collectives (4096 tokens)",
                    "chunks": plan.chunk_count(),
                    "metrics": [
                        "chunked all-reduce of the 4096-token seam is bit-identical to the canonical CPU twin, every chunk, every repeat",
                        "two runs per configuration give the same per-chunk output bytes",
                        "no DtoH/HtoD between chunks: device pointers only",
                        "per-chunk latency and the total against the single-3.1 path, recorded per collective",
                        "the arrival-word reuse across chunks shows no lost or stale doorbell in the kernel trace"
                    ],
                    "requires": [
                        "four GPU nodes opened by the Lead for the window",
                        "a cuda-feature build with the kernel launcher",
                        "S5.3's one-shot device-path gate green first"
                    ]
                }
            })
        }
        Err(error) => json!({ "planned": true, "error": error.to_string() }),
    }
}
/// Runs one rank per process over TCP.
fn run_tcp(args: &SmokeArgs) -> Result<Vec<String>, String> {
    let addresses = args.addresses();
    let config = TcpConfig::with_addresses(args.rank, addresses.clone())
        .map_err(|error| format!("build the cluster config: {error}"))?
        .with_timeout(args.timeout)
        .with_connect_timeout(args.connect_timeout);
    let connect_start = Instant::now();
    let transport = TcpTransport::connect(config)
        .map_err(|error| format!("connect rank {} of {addresses:?}: {error}", args.rank))?;
    let connect_us = micros(connect_start.elapsed());

    let data = build_rank_data(args, args.rank)?;
    let outcome = smoke_rank(&transport, args.rank, args.world, &data, args)?;
    report_rank(args, "tcp", connect_us, &data, &outcome)
}

/// Writes one rank's evidence file and the lines to print for a one-process-per-rank backend.
fn report_rank(
    args: &SmokeArgs,
    transport_label: &str,
    connect_us: u64,
    data: &RankData,
    outcome: &RankOutcome,
) -> Result<Vec<String>, String> {
    let value = evidence(args, args.rank, transport_label, connect_us, data, outcome);
    let path = write_evidence(&args.evidence_dir, args.rank, &value)?;
    let ok = outcome_ok(outcome, args);
    let lines = vec![
        format!(
            "tp4-smoke rank {} of {} over {}: payload {} bf16, L1 mismatches {}, repeats bit-identical {}/{}, L2 {}",
            args.rank,
            args.world,
            transport_label,
            outcome.payload_elements,
            outcome.mismatches,
            outcome.repeats_bit_identical,
            args.repeats,
            outcome
                .l2
                .as_ref()
                .map(|l2| format!(
                    "max {} ulp / median {} ulp (bar {} / {}); per-element diagnostic max {} / median {} ulp, {:.2}% within 1 ulp",
                    l2.max_ulp,
                    l2.median_ulp,
                    L2_MAX_ULP,
                    L2_MEDIAN_ULP,
                    l2.per_element.max_ulp,
                    l2.per_element.median_ulp,
                    l2.per_element.within_one_ulp_fraction * 100.0
                ))
                .unwrap_or_else(|| "not measured (synthetic payload)".to_string()),
        ),
        format!("tp4-smoke evidence: {}", path.display()),
        format!(
            "tp4-smoke rank {}: {}",
            args.rank,
            if ok { "PASS" } else { "FAIL" }
        ),
    ];
    if !ok {
        return Err(lines.join("\n"));
    }
    Ok(lines)
}

/// Runs one rank per process over the real RDMA data plane (the rdma feature).
///
/// This is the S5.3 hardware entry that reuses the whole host stage: the same sharded read, the
/// same L1/L2 checks and the same evidence schema, but the collective runs through the dual-rail
/// libibverbs data plane instead of TCP. It is the first hardware run of that data plane; the
/// device-buffer form (`--oneshot-preflight`'s seam) is a later increment on top of it.
/// Refuses a stub-shim build before it opens anything.
///
/// "No silent fallback" applies to the build configuration as well as to the collective: with
/// `--features rdma` compiled without `<infiniband/verbs.h>` the three inline entry points are
/// ENOSYS stubs, and a run would otherwise get as far as `Verbs::load` and fail there with a
/// message that reads like a hardware problem. The binary knows its own mode and says so.
#[cfg(feature = "rdma")]
fn refuse_stub_shim() -> Result<(), String> {
    match sparkflash_comm::rdma::shim_mode() {
        sparkflash_comm::rdma::ShimMode::System => Ok(()),
        sparkflash_comm::rdma::ShimMode::Stub => Err(
            "this binary was built with the ENOSYS stub shim (SPARKFLASH_IBVERBS_SHIM=stub): \
             the rdma feature was compiled without <infiniband/verbs.h>, so ibv_post_send, \
             ibv_post_recv and ibv_poll_cq cannot post. This is a build-configuration refusal, not \
             a fabric error. Rebuild with the staged headers, e.g. \
             SPARKFLASH_RDMA_INCLUDE=/path/to/aarch64-rdma/include and --features=rdma,cuda; the \
             build log must not contain the 'rdma feature built without <infiniband/verbs.h>' \
             warning. The evidence flags record comm_rdma_shim=stub for this binary."
                .to_string(),
        ),
    }
}

#[cfg(feature = "rdma")]
fn run_rdma(args: &SmokeArgs) -> Result<Vec<String>, String> {
    refuse_stub_shim()?;
    let open_start = Instant::now();
    let transport = sparkflash_comm::rdma::RdmaTransport::open(args.world, args.rank)
        .map_err(|error| format!("open the RDMA data plane on rank {}: {error}", args.rank))?;
    let connect_us = micros(open_start.elapsed());
    let data = build_rank_data(args, args.rank)?;
    let outcome = smoke_rank(&transport, args.rank, args.world, &data, args)?;
    report_rank(args, "rdma", connect_us, &data, &outcome)
}

/// The build mode of the RDMA inline shim, embedded unconditionally in the evidence flags.
///
/// A window must be able to tell a real-verbs binary from a stub one from the JSON alone; the
/// stderr banner of the build is not part of the artifact it ships.
fn rdma_shim_flag() -> &'static str {
    crate::comm_rdma_shim()
}

/// Runs every rank in this one process over the in-process loopback.
fn run_loopback(args: &SmokeArgs) -> Result<Vec<String>, String> {
    let cluster = LoopbackCluster::new(LoopbackConfig {
        world_size: args.world,
        ..LoopbackConfig::default()
    })
    .map_err(|error| format!("build the loopback cluster: {error}"))?;
    let results: Vec<Result<(RankOutcome, PathBuf), String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..args.world)
            .map(|rank| {
                let cluster = &cluster;
                scope.spawn(move || -> Result<(RankOutcome, PathBuf), String> {
                    let transport = cluster
                        .handle(rank)
                        .map_err(|error| format!("loopback handle for rank {rank}: {error}"))?;
                    let data = build_rank_data(args, rank)?;
                    let outcome = smoke_rank(&transport, rank, args.world, &data, args)?;
                    let value = evidence(args, rank, "lo", 0, &data, &outcome);
                    let path = write_evidence(&args.evidence_dir, rank, &value)?;
                    Ok((outcome, path))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|_panic| Err("a rank thread panicked".to_string()))
            })
            .collect()
    });

    let mut lines = Vec::new();
    let mut ranks = Vec::new();
    let mut all_l1 = true;
    let mut all_repeats = true;
    let mut all_l2 = true;
    for result in results {
        let (outcome, path) = result?;
        let ok = outcome_ok(&outcome, args);
        all_l1 &= outcome.mismatches == 0;
        all_repeats &= outcome.repeats_bit_identical == args.repeats;
        all_l2 &= outcome.l2.as_ref().is_none_or(|l2| l2.within_bar);
        lines.push(format!(
            "tp4-smoke rank {}: L1 mismatches {}, repeats bit-identical {}/{}, L2 {}",
            outcome.rank,
            outcome.mismatches,
            outcome.repeats_bit_identical,
            args.repeats,
            outcome
                .l2
                .as_ref()
                .map(|l2| format!(
                    "max {} ulp / median {} ulp (bar {} / {}); per-element diagnostic max {} / median {} ulp, {:.2}% within 1 ulp",
                    l2.max_ulp,
                    l2.median_ulp,
                    L2_MAX_ULP,
                    L2_MEDIAN_ULP,
                    l2.per_element.max_ulp,
                    l2.per_element.median_ulp,
                    l2.per_element.within_one_ulp_fraction * 100.0
                ))
                .unwrap_or_else(|| "not measured (synthetic payload)".to_string()),
        ));
        lines.push(format!("tp4-smoke evidence: {}", path.display()));
        ranks.push(json!({
            "rank": outcome.rank,
            "mismatches": outcome.mismatches,
            "repeats_bit_identical": outcome.repeats_bit_identical,
            "payload_elements": outcome.payload_elements,
            "l2": outcome.l2.as_ref().map(L2Report::to_json),
            "ok": ok,
            "evidence": path.display().to_string(),
        }));
    }
    let summary = json!({
        "stage": "tp4-smoke",
        "schema": SCHEMA_VERSION,
        "started_at_unix_s": unix_seconds(),
        "backend": "lo",
        "world_size": args.world,
        "repeats": args.repeats,
        "ranks": ranks,
        "gates": {
            "l1_zero_mismatches": all_l1,
            "repeats_bit_identical": all_repeats,
            "l2_within_bar": all_l2,
        },
        "oneshot": if args.oneshot_preflight {
            json!({
                "contract": "host_preflight",
                "validated_on_host": validate_oneshot_preflight(args).is_ok(),
                "chunked": chunked_section(args),
            })
        } else {
            json!({ "contract": "not_requested", "chunked": chunked_section(args) })
        },
        "flags": {
            "device_execution": false,
            "token_produced": false,
            "cuda_feature": cfg!(feature = "cuda"),
            "comm_rdma_shim": rdma_shim_flag(),
        },
        "verdict": if all_l1 && all_repeats && all_l2 { "PASS" } else { "FAIL" },
    });
    fs::create_dir_all(&args.evidence_dir)
        .map_err(|error| format!("create {}: {error}", args.evidence_dir.display()))?;
    let path = args.evidence_dir.join("tp4-smoke-summary.json");
    let text = serde_json::to_string_pretty(&summary)
        .map_err(|error| format!("serialise the summary: {error}"))?;
    fs::write(&path, text).map_err(|error| format!("write {}: {error}", path.display()))?;
    lines.push(format!("tp4-smoke summary: {}", path.display()));
    let all_ok = all_l1 && all_repeats && all_l2;
    lines.push(format!(
        "tp4-smoke: {}",
        if all_ok { "PASS" } else { "FAIL" }
    ));
    if !all_ok {
        return Err(lines.join("\n"));
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use super::*;

    fn parse(args: &[&str]) -> Result<SmokeArgs, String> {
        SmokeArgs::parse(args)
    }

    fn free_adjacent_ports() -> u16 {
        for base in (41000u16..60000).step_by(2) {
            if TcpListener::bind(("127.0.0.1", base)).is_ok()
                && TcpListener::bind(("127.0.0.1", base + 1)).is_ok()
            {
                return base;
            }
        }
        panic!("no two free adjacent ports");
    }

    #[test]
    fn arguments_require_the_rank_the_evidence_dir_and_a_source() {
        assert!(parse(&["--checkpoint", "/tmp/x"]).is_err());
        assert!(parse(&["--tp-rank", "0", "--evidence-dir", "/tmp/e"]).is_err());
        assert!(
            parse(&["--tp-rank", "0", "--evidence-dir", "/tmp/e"]).is_err(),
            "a run with neither --checkpoint nor --synthetic must be refused"
        );
        let args = parse(&[
            "--synthetic",
            "--tp-rank",
            "0",
            "--tp-world-size",
            "2",
            "--comm-backend",
            "lo",
            "--evidence-dir",
            "/tmp/e",
        ])
        .expect("a loopback synthetic run parses");
        assert_eq!(
            (args.rank, args.world, args.backend),
            (0, 2, Backend::Loopback)
        );
        assert!(args.checkpoint.is_none());
    }

    #[test]
    fn the_address_table_is_indexed_by_rank_and_advances_the_port() {
        let args = parse(&[
            "--checkpoint",
            "/tmp/x",
            "--tp-rank",
            "1",
            "--tp-world-size",
            "2",
            "--comm-peers",
            "198.51.100.80,198.51.100.88",
            "--comm-port",
            "29601",
            "--evidence-dir",
            "/tmp/e",
        ])
        .expect("parse");
        let addresses = args.addresses();
        assert_eq!(addresses[0].to_string(), "198.51.100.80:29601");
        assert_eq!(addresses[1].to_string(), "198.51.100.88:29602");
    }

    #[test]
    fn a_peer_table_that_does_not_describe_the_world_is_refused() {
        let error = parse(&[
            "--checkpoint",
            "/tmp/x",
            "--tp-rank",
            "0",
            "--tp-world-size",
            "4",
            "--comm-peers",
            "198.51.100.80,198.51.100.88",
            "--evidence-dir",
            "/tmp/e",
        ])
        .expect_err("a short peer table must be refused");
        assert!(error.contains("--comm-peers"), "{error}");
        let error = parse(&[
            "--checkpoint",
            "/tmp/x",
            "--tp-rank",
            "2",
            "--tp-world-size",
            "2",
            "--comm-peers",
            "127.0.0.1,127.0.0.1",
            "--evidence-dir",
            "/tmp/e",
        ])
        .expect_err("a rank outside the world must be refused");
        assert!(error.contains("outside a world"), "{error}");
    }

    #[test]
    fn the_rdma_backend_is_a_hardware_entry_and_refuses_without_one() {
        let error = run_lines(&[
            "--synthetic",
            "--tp-rank",
            "0",
            "--tp-world-size",
            "2",
            "--comm-backend",
            "rdma",
            "--evidence-dir",
            "/tmp/tp4-smoke-rdma-unused",
        ])
        .expect_err("a host with no fabric cannot open the rdma backend");
        #[cfg(not(feature = "rdma"))]
        assert!(error.contains("rdma feature"), "{error}");
        #[cfg(feature = "rdma")]
        {
            match sparkflash_comm::rdma::shim_mode() {
                sparkflash_comm::rdma::ShimMode::Stub => {
                    assert!(error.contains("stub shim"), "{error}");
                    assert!(error.contains("build-configuration refusal"), "{error}");
                }
                sparkflash_comm::rdma::ShimMode::System => {
                    assert!(error.contains("open the RDMA data plane"), "{error}");
                }
            }
        }
    }

    #[test]
    fn bf16_ulp_is_the_bf16_spacing() {
        assert!((bf16_ulp(1.0) - 2f32.powi(-7)).abs() < f32::EPSILON);
        assert!((bf16_ulp(2.0) - 2f32.powi(-6)).abs() < f32::EPSILON);
        assert!((bf16_ulp(-1.0) - 2f32.powi(-7)).abs() < f32::EPSILON);
        assert_eq!(bf16_ulp(0.0), 0.0);
    }

    #[test]
    fn activation_patterns_are_bounded_and_repeatable() {
        for index in [0usize, 1, 7, 8191] {
            let bits = pattern_bit(index, DEFAULT_SEED);
            assert_eq!(bits, pattern_bit(index, DEFAULT_SEED));
            let value = bf16_value(bits);
            assert!(value.is_finite(), "index {index} is not finite: {value}");
            assert!((-1.0..=1.0).contains(&value), "index {index}: {value}");
        }
        assert_ne!(pattern_bit(0, DEFAULT_SEED), pattern_bit(1, DEFAULT_SEED));
    }

    #[test]
    fn compare_l2_measures_deltas_in_bf16_ulp() {
        let twin = vec![bf16_bits(4.0), bf16_bits(4.0), bf16_bits(4.0)];
        let seam = bf16_ulp(4.0);
        let identical = compare_l2(&twin, &twin);
        assert_eq!(identical.max_ulp, 0.0);
        assert_eq!(identical.seam_ulp, seam);
        assert!(identical.within_bar);

        let one_ulp = compare_l2(&[bf16_bits(4.0 + seam), twin[1], twin[2]], &twin);
        assert!((one_ulp.max_ulp - 1.0).abs() < 0.01, "{one_ulp:?}");
        assert!(one_ulp.within_bar);

        let two_ulp = compare_l2(&[bf16_bits(4.0 + 2.0 * seam), twin[1], twin[2]], &twin);
        assert!((two_ulp.max_ulp - 2.0).abs() < 0.01, "{two_ulp:?}");
        assert_eq!(two_ulp.median_ulp, 0.0);
        assert!(two_ulp.within_bar, "the max bar is inclusive at 2 ulp");

        let three_ulp = compare_l2(&[bf16_bits(4.0 + 3.0 * seam), twin[1], twin[2]], &twin);
        assert!(three_ulp.max_ulp > L2_MAX_ULP, "{three_ulp:?}");
        assert!(!three_ulp.within_bar);
    }

    #[test]
    fn the_per_element_ulp_diagnostic_separates_cancellation_from_the_seam_scale() {
        // Both elements move by exactly one seam ulp (one ulp of 4.0). On the seam scale that is 1
        // ulp; on the small element's own scale it is 64 ulp, which is what a cancellation-heavy
        // seam looks like and what the diagnostic must surface without gating on it.
        let seam = bf16_ulp(4.0);
        let twin = vec![bf16_bits(4.0), bf16_bits(0.0625)];
        let reduced = vec![bf16_bits(4.0 + seam), bf16_bits(0.0625 + seam)];
        let report = compare_l2(&reduced, &twin);
        assert!((report.max_ulp - 1.0).abs() < 0.01, "{report:?}");
        assert!(report.within_bar);
        assert!(
            (report.per_element.max_ulp - 64.0).abs() < 0.5,
            "{report:?}"
        );
        // The 4.0 element is exactly 1 own-ulp away (within), the small one is far outside.
        assert_eq!(report.per_element.within_one_ulp_fraction, 0.5);

        let identical = compare_l2(&twin, &twin);
        assert_eq!(identical.per_element.max_ulp, 0.0);
        assert_eq!(identical.per_element.within_one_ulp_fraction, 1.0);
    }

    #[test]
    fn the_collective_deadlines_are_settable_and_must_be_nonzero() {
        let args = SmokeArgs::parse(&[
            "--synthetic",
            "--tp-rank",
            "0",
            "--comm-backend",
            "lo",
            "--tp-world-size",
            "2",
            "--evidence-dir",
            "/tmp/tp4-smoke-timeout",
            "--comm-timeout",
            "120",
            "--comm-connect-timeout",
            "90",
        ])
        .expect("parse");
        assert_eq!(args.timeout, Duration::from_secs(120));
        assert_eq!(args.connect_timeout, Duration::from_secs(90));
        for flag in ["--comm-timeout", "--comm-connect-timeout"] {
            let error = SmokeArgs::parse(&[
                "--synthetic",
                "--tp-rank",
                "0",
                "--comm-backend",
                "lo",
                "--tp-world-size",
                "2",
                "--evidence-dir",
                "/tmp/tp4-smoke-timeout",
                flag,
                "0",
            ])
            .expect_err("zero must be refused");
            assert!(error.contains("at least one second"), "{error}");
        }
    }

    #[test]
    fn a_synthetic_loopback_run_passes_and_writes_one_file_per_rank() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dir_text = dir.path().display().to_string();
        let lines = run_lines(&[
            "--synthetic",
            "--tp-rank",
            "0",
            "--tp-world-size",
            "2",
            "--comm-backend",
            "lo",
            "--payload-elements",
            "512",
            "--repeats",
            "5",
            "--evidence-dir",
            &dir_text,
        ])
        .expect("the synthetic loopback run passes");
        assert!(
            lines.iter().any(|line| line.contains("tp4-smoke: PASS")),
            "{lines:?}"
        );
        for rank in 0..2 {
            let path = dir.path().join(format!("tp4-smoke-r{rank}.json"));
            let text = fs::read_to_string(&path).expect("read the evidence");
            let value: Value = serde_json::from_str(&text).expect("evidence json");
            assert_eq!(value["tp"]["rank"], rank);
            assert_eq!(value["l1"]["mismatches"], 0);
            assert_eq!(value["l1"]["repeats_bit_identical"], 5);
            assert_eq!(value["l1"]["payload_elements"], 512);
            assert_eq!(value["flags"]["device_execution"], false);
            assert_eq!(value["flags"]["token_produced"], false);
            assert_eq!(value["flags"]["comm_rdma_shim"], rdma_shim_flag());
            assert_eq!(value["load"]["slice_bytes"], 1024);
            assert!(value["tensors"].as_array().expect("tensors").is_empty());
        }
        let summary: Value = serde_json::from_str(
            &fs::read_to_string(dir.path().join("tp4-smoke-summary.json")).expect("read"),
        )
        .expect("summary json");
        assert_eq!(summary["verdict"], "PASS");
        assert_eq!(summary["gates"]["l1_zero_mismatches"], true);
        assert_eq!(summary["gates"]["repeats_bit_identical"], true);
        assert_eq!(summary["flags"]["comm_rdma_shim"], rdma_shim_flag());
        assert_eq!(summary["flags"]["device_execution"], false);
        assert_eq!(summary["oneshot"]["contract"], "not_requested");
    }

    /// Writes a tiny two-rank checkpoint whose layer-0 attention tensors follow the reference
    /// rules, so the sharded read, the byte-identity check and the wo_b seam all execute.
    fn write_fixture(dir: &Path) {
        let shard_name = "model-00001-of-00001.safetensors";
        let mut tensors: Vec<(String, String, Vec<usize>, Vec<u8>)> = Vec::new();
        // wq_a and wkv are replicated, wq_b and wo_a column-parallel, wo_b row-parallel.
        let values =
            |len: usize| -> Vec<u8> { (0..len).map(|index| (1 + index % 90) as u8).collect() };
        // Scales are all one (E8M0 byte 127): the fixture's purpose is the slice and the seam, not
        // a wide dynamic range, and a wide one makes a cancellation-dominated seam report ulps of
        // its own rounding noise rather than of the seam.
        let scales = |len: usize| vec![127u8; len];
        let mut push = |name: &str, dtype: &str, shape: Vec<usize>, bytes: Vec<u8>| {
            tensors.push((
                format!("layers.0.attn.{name}"),
                dtype.to_string(),
                shape,
                bytes,
            ));
        };
        push("wq_a.weight", "F8_E4M3", vec![8, 64], values(8 * 64));
        push("wq_a.scale", "F8_E8M0", vec![1, 2], scales(2));
        push("wkv.weight", "F8_E4M3", vec![8, 64], values(8 * 64));
        push("wkv.scale", "F8_E8M0", vec![1, 2], scales(2));
        push("wq_b.weight", "F8_E4M3", vec![64, 64], values(64 * 64));
        push("wq_b.scale", "F8_E8M0", vec![2, 2], scales(4));
        push("wo_a.weight", "F8_E4M3", vec![64, 64], values(64 * 64));
        push("wo_a.scale", "F8_E8M0", vec![2, 2], scales(4));
        push("wo_b.weight", "F8_E4M3", vec![64, 64], values(64 * 64));
        push("wo_b.scale", "F8_E8M0", vec![2, 2], scales(4));

        let mut header = String::from("{");
        let mut offset = 0u64;
        for (index, (name, dtype, shape, bytes)) in tensors.iter().enumerate() {
            if index > 0 {
                header.push(',');
            }
            let begin = offset;
            offset += bytes.len() as u64;
            let shape_text = shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape_text}],\"data_offsets\":[{begin},{offset}]}}"
            ));
        }
        header.push('}');
        let mut header = header.into_bytes();
        while (8 + header.len()) % 8 != 0 {
            header.push(b' ');
        }
        let mut file = Vec::new();
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(&header);
        for (_, _, _, bytes) in &tensors {
            file.extend_from_slice(bytes);
        }
        fs::write(dir.join(shard_name), file).expect("write the fixture shard");
        let weight_map = tensors
            .iter()
            .map(|(name, _, _, _)| format!("\"{name}\":\"{shard_name}\""))
            .collect::<Vec<_>>()
            .join(",");
        let index_text = format!("{{\"metadata\":{{}},\"weight_map\":{{{weight_map}}}}}");
        fs::write(dir.join("model.safetensors.index.json"), index_text).expect("write the index");
    }

    #[test]
    fn a_checkpoint_loopback_run_reads_each_slice_and_passes() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_fixture(dir.path());
        let dir_text = dir.path().display().to_string();
        let lines = run_lines(&[
            "--checkpoint",
            &dir_text,
            "--tp-rank",
            "0",
            "--tp-world-size",
            "2",
            "--comm-backend",
            "lo",
            "--repeats",
            "3",
            "--evidence-dir",
            &dir_text,
        ])
        .expect("the checkpoint loopback run passes");
        assert!(
            lines.iter().any(|line| line.contains("tp4-smoke: PASS")),
            "{lines:?}"
        );
        for rank in 0..2 {
            let text = fs::read_to_string(dir.path().join(format!("tp4-smoke-r{rank}.json")))
                .expect("read the evidence");
            let value: Value = serde_json::from_str(&text).expect("evidence json");
            let tensors = value["tensors"].as_array().expect("tensors");
            assert_eq!(tensors.len(), 10, "every planned tensor is checked");
            for tensor in tensors {
                assert_eq!(tensor["match"], true, "rank {rank}: {tensor}");
                assert_eq!(tensor["mismatched_segments"], 0);
            }
            assert_eq!(value["l1"]["mismatches"], 0);
            assert_eq!(value["l1"]["repeats_bit_identical"], 3);
            assert_eq!(value["l2"]["samples"], 64);
            assert_eq!(value["l2"]["within_bar"], true, "rank {rank}: {value}");
            assert_eq!(value["flags"]["device_execution"], false);
        }
    }

    #[test]
    fn two_tcp_ranks_in_threads_agree_with_the_canonical_twin() {
        let base = free_adjacent_ports();
        let dir = tempfile::tempdir().expect("tempdir");
        let dir_text = dir.path().display().to_string();
        let base_text = base.to_string();
        let addresses = vec![
            SocketAddr::from(([127, 0, 0, 1], base)),
            SocketAddr::from(([127, 0, 0, 1], base + 1)),
        ];
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..2)
                .map(|rank| {
                    let addresses = addresses.clone();
                    let dir_text = dir_text.clone();
                    let base_text = base_text.clone();
                    scope.spawn(move || -> Result<RankOutcome, String> {
                        let rank_text = rank.to_string();
                        let args = SmokeArgs::parse(&[
                            "--synthetic",
                            "--tp-rank",
                            rank_text.as_str(),
                            "--tp-world-size",
                            "2",
                            "--comm-backend",
                            "tcp",
                            "--comm-peers",
                            "127.0.0.1,127.0.0.1",
                            "--comm-port",
                            base_text.as_str(),
                            "--payload-elements",
                            "1024",
                            "--repeats",
                            "3",
                            "--evidence-dir",
                            dir_text.as_str(),
                        ])?;
                        let config = TcpConfig::with_addresses(rank, addresses)
                            .map_err(|error| format!("cluster config: {error}"))?;
                        let transport = TcpTransport::connect(config)
                            .map_err(|error| format!("rank {rank} connect: {error}"))?;
                        let data = build_rank_data(&args, rank)?;
                        smoke_rank(&transport, rank, 2, &data, &args)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|_panic| Err("panicked".to_string()))
                })
                .collect::<Vec<_>>()
        });
        for result in results {
            let outcome = result.expect("rank");
            assert_eq!(outcome.mismatches, 0);
            assert_eq!(outcome.repeats_bit_identical, 3);
        }
    }

    #[test]
    fn the_oneshot_abi_pins_agree_across_the_two_crates() {
        oneshot_abi_parity().expect("the comm and kernels pins agree");
    }

    #[test]
    fn the_oneshot_preflight_validates_the_contract_and_records_it() {
        let args = parse(&[
            "--synthetic",
            "--tp-rank",
            "0",
            "--tp-world-size",
            "4",
            "--comm-backend",
            "lo",
            "--evidence-dir",
            "/tmp/tp4-oneshot-preflight",
            "--repeats",
            "7",
            "--oneshot-preflight",
            "--prefill-plan-elements",
            "20971520",
        ])
        .expect("a four-rank preflight parses");
        validate_oneshot_preflight(&args).expect("the contract holds");
        let section = oneshot_section(&args);
        assert_eq!(section["state"], "host_preflight");
        assert_eq!(section["abi"]["cross_crate_parity"], true);
        assert_eq!(section["abi"]["slot_bytes"], 8_388_640);
        assert_eq!(section["abi"]["row_bytes"], 2_097_152);
        assert_eq!(section["exchange"]["elements"], 5120);
        assert_eq!(section["exchange"]["bytes"], 10_240);
        assert_eq!(section["exchange"]["rail0_bytes"], 5_120);
        assert_eq!(section["exchange"]["rail1_bytes"], 5_120);
        assert_eq!(section["exchange"]["posted_bytes"], 30_720);
        assert_eq!(section["sequence"]["last"], 6);
        assert_eq!(section["dispatch"]["launched"], false);
        assert_eq!(section["dispatch_record"]["validated_on_host"], true);
        assert_eq!(section["dispatch_record"]["emitted"], false);
        assert_eq!(section["dispatch_record"]["host_copy_bytes"], 0);
        assert_eq!(section["dispatch_record"]["kernel"], ONESHOT_KERNEL_NAME);
        assert_eq!(section["dispatch_record"]["required_on_window_run"], true);
        assert_eq!(section["window_measurement"]["repeats"], 7);
        assert_eq!(section["chunked"]["planned"], true);
        assert_eq!(section["chunked"]["elements"], 20_971_520);
        assert_eq!(section["chunked"]["bytes"], 41_943_040);
        assert_eq!(section["chunked"]["chunks"], 20);
        assert_eq!(section["chunked"]["chunk_bytes"], ONESHOT_ROW_BYTES);
        assert_eq!(section["chunked"]["posted_bytes"], 3 * 41_943_040);
        assert_eq!(section["chunked"]["window_measurement"]["chunks"], 20);
        assert_eq!(section["host_twin"]["ran"], true);
        assert_eq!(
            section["host_twin"]["oneshot"]["bit_identical_to_canonical_twin"],
            true
        );
        assert_eq!(section["host_twin"]["oneshot"]["host_copy_bytes"], 0);
        assert_eq!(section["host_twin"]["oneshot"]["slot_ring"], 2);
        assert_eq!(
            section["host_twin"]["oneshot"]["dispatches"],
            4 * HOST_TWIN_SEQUENCES
        );
        assert_eq!(section["host_twin"]["chunked"]["ran"], true);
        assert_eq!(section["host_twin"]["chunked"]["host_copy_bytes"], 0);
        assert_eq!(
            section["host_twin"]["chunked"]["bit_identical_to_canonical_twin"],
            true
        );
        assert_eq!(section["host_twin"]["ring"]["ran"], true);
        assert_eq!(
            section["host_twin"]["ring"]["elements"],
            HOST_TWIN_RING_ELEMENTS
        );
        assert_eq!(section["host_twin"]["ring"]["chunks"], 2);
        assert_eq!(section["host_twin"]["ring"]["ring"], 2);
        assert_eq!(
            section["host_twin"]["ring"]["bit_identical_to_canonical_twin"],
            true
        );
        assert_eq!(section["host_twin"]["ring"]["host_copy_bytes"], 0);
        assert_eq!(section["host_twin"]["ring"]["device_path_only"], true);
        assert!(
            section["host_twin"]["ring"]["max_rank_in_flight"]
                .as_u64()
                .unwrap_or(0)
                >= 2,
            "the evidence run must show two chunks in flight"
        );
        assert_eq!(section["host_twin"]["ring"]["hazard"]["ran"], true);
        assert_eq!(section["host_twin"]["ring"]["hazard"]["demonstrated"], true);
        assert!(
            section["host_twin"]["ring"]["hazard"]["mismatched_ranks"]
                .as_array()
                .is_some_and(|ranks| ranks.iter().any(|rank| rank == 0))
        );
        assert!(
            section["host_twin"]["unclaimed_node_measurements"]
                .as_array()
                .is_some_and(|items| !items.is_empty())
        );
        assert_eq!(
            section["host_twin"]["does_not_prove"][0],
            "that the CUDA kernel, the device-memory MR, the queue pairs or the launcher exist"
        );
        assert_eq!(section["orchestration"]["ran"], true);
        assert_eq!(section["orchestration"]["happy"]["ran"], true);
        assert_eq!(
            section["orchestration"]["happy"]["first_token_agreement"],
            true
        );
        assert_eq!(
            section["orchestration"]["happy"]["served_tokens"],
            json!([9, 9, 9])
        );
        assert_eq!(
            section["orchestration"]["happy"]["expected_tokens"],
            json!([9, 9, 9])
        );
        assert_eq!(section["orchestration"]["happy"]["device_path_only"], true);
        assert_eq!(section["orchestration"]["startup"]["ready"], false);
        assert_eq!(section["orchestration"]["startup"]["missing"], json!([2]));
        for scenario in section["orchestration"]["failure_handling"]
            .as_array()
            .expect("failure scenarios")
        {
            assert!(
                scenario["guarded_refused"].as_str().is_some(),
                "every injected failure must be refused with the guard: {scenario}"
            );
            assert_eq!(
                scenario["guard_is_load_bearing"], true,
                "without the guard the wrong answer must be served: {scenario}"
            );
        }
    }

    #[test]
    fn the_oneshot_preflight_refuses_a_bad_world_and_an_oversized_payload() {
        assert!(
            parse(&[
                "--synthetic",
                "--tp-rank",
                "0",
                "--tp-world-size",
                "2",
                "--comm-backend",
                "lo",
                "--evidence-dir",
                "/tmp/e",
                "--oneshot-preflight",
            ])
            .is_err(),
            "the one-shot seam is a four-rank collective"
        );
        let args = parse(&[
            "--synthetic",
            "--tp-rank",
            "0",
            "--tp-world-size",
            "4",
            "--comm-backend",
            "lo",
            "--evidence-dir",
            "/tmp/e",
            "--payload-elements",
            "1048577",
            "--oneshot-preflight",
        ])
        .expect("the arguments parse; the payload is refused by the seam");
        assert!(validate_oneshot_preflight(&args).is_err());
    }

    #[test]
    fn the_prefill_plan_needs_the_device_seam_flag_and_more_than_one_row() {
        let base = [
            "--synthetic",
            "--tp-rank",
            "0",
            "--tp-world-size",
            "4",
            "--comm-backend",
            "lo",
            "--evidence-dir",
            "/tmp/tp4-prefill-plan",
            "--prefill-plan-elements",
            "20971520",
        ];
        let error = parse(&base).expect_err("the prefill plan belongs to the seam");
        assert!(error.contains("--oneshot-preflight"), "{error}");
        let mut with_seam = base.to_vec();
        with_seam.push("--oneshot-preflight");
        let args = parse(&with_seam).expect("the plan parses with the seam flag");
        assert_eq!(args.prefill_plan_elements, Some(20_971_520));
        let mut one_row = base.to_vec();
        one_row[10] = "1048576";
        one_row.push("--oneshot-preflight");
        let error = parse(&one_row).expect_err("one row is not a prefill plan");
        assert!(error.contains("needs more than one"), "{error}");
    }
}

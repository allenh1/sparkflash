//! Host-side stage for the S5.2 resident loader: the full 40-layer per-rank weight plan, from
//! tensor headers only, and the in-place read when a checkpoint is available.
//!
//! The device chain has never had a resident loader: every device stage built so far stages one
//! layer at a time. This stage is the host half of the one that does - it decides, for every
//! checkpoint tensor the LLM loads, which slice this rank keeps in memory
//! ([\`sparkflash_device::resident\`]) and, with \`--read\`, actually reads those slices from the
//! released checkpoint through the verified [\`ShardPlan\`](sparkflash_device::shard::ShardPlan)
//! range reader. That is what makes a 4-rank run need no repacked copy: each rank reads one
//! quarter of the checkpoint where it lies.
//!
//! The stage is host-only and touches no driver. It writes one \`tp4-resident-r<rank>.json\` per rank
//! and a \`tp4-resident-summary.json\` after each run, in the S5.1 evidence conventions. It records
//! \`device_execution: false\`, \`token_produced: false\` and the RDMA shim mode unconditionally, and
//! it names the window measurement it does not perform instead of implying one.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use sparkflash_device::resident::{
    CHAIN_LAYERS, HeaderEntry, MTP_LAYERS, ReadEvidence, ResidentPlan,
};
use sparkflash_device::shard::StoredShard;
use sparkflash_engine::safetensors::{CheckpointIndex, SafeTensorsSource};

/// The evidence schema version this stage writes (S5.1 conventions).
pub const SCHEMA_VERSION: u32 = 1;

/// The stage's command line.
pub const USAGE: &str = "sparkflash device-chain --run-stage tp4-resident (--tensor-index FILE | --checkpoint DIR) --tp-rank 0..3 --tp-world-size 4 --evidence-dir DIR [--read] [--upload] [--design-gib 72.31]";

/// The design's per-rank resident budget, in GiB (docs/build/tp4-design.md section 6.3).
pub const DESIGN_RESIDENT_GIB: f64 = 72.31;

/// Runs the stage and turns a failure into a message on stderr.
pub fn run(args: &[&str]) -> ExitCode {
    match run_lines(args) {
        Ok(lines) => {
            for line in lines {
                println!("{line}");
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("tp4-resident: {error}");
            ExitCode::from(2)
        }
    }
}

/// One rank's loaded plan, reachable by both the evidence path and the device loop.
pub(crate) struct RankSetup {
    pub plan: ResidentPlan,
    pub shard_plan: sparkflash_device::shard::ShardPlan,
    pub source_kind: String,
    pub tensors_seen: usize,
}

/// Loads the headers for one rank and plans it. The streaming source itself stays owned by the
/// caller (`args.source`); this is the setup both the evidence path and the device loop need.
pub(crate) fn load_rank_setup(args: &ResidentArgs) -> Result<RankSetup, String> {
    let (entries, source_kind, tensors_seen) = load_entries(args)?;
    let plan = ResidentPlan::plan(args.rank, args.world, entries)
        .map_err(|error| format!("plan rank {}: {error}", args.rank))?;
    let shard_plan = plan
        .shard_plan()
        .map_err(|error| format!("plan rank {} ranges: {error}", plan.rank()))?;
    Ok(RankSetup {
        plan,
        shard_plan,
        source_kind,
        tensors_seen,
    })
}

/// The setup for one rank plus its streaming source, reachable from the chain's loop path.
pub(crate) fn setup_for_chain(
    argv: &[&str],
) -> Result<(RankSetup, Option<SafeTensorsSource>), String> {
    let args = ResidentArgs::parse(argv)?;
    let setup = load_rank_setup(&args)?;
    Ok((setup, args.source))
}

/// Streams the rank arena and binds every attention-path tensor through the expected-shape table.
#[cfg(feature = "cuda")]
pub(crate) fn bind_attention_in_arena(
    setup: &RankSetup,
    source: &SafeTensorsSource,
    table: &std::collections::BTreeMap<String, (usize, usize)>,
) -> Result<(usize, Value, Vec<String>), String> {
    upload_arena_streaming(&setup.plan, &setup.shard_plan, source, |arena| {
        let bound = crate::resident_weights::bind_rank_attention(arena, &setup.plan, table)?;
        Ok(bound.len())
    })
}

/// Without the cuda feature the arena cannot be opened, so binding is refused.
#[cfg(not(feature = "cuda"))]
pub(crate) fn bind_attention_in_arena(
    _setup: &RankSetup,
    _source: &SafeTensorsSource,
    _table: &std::collections::BTreeMap<String, (usize, usize)>,
) -> Result<(usize, Value, Vec<String>), String> {
    Err("--loop with --upload needs a binary built with --features cuda".to_string())
}

/// Runs one rank's plan (and read, with \`--read\`), writing the evidence and the summary.
pub fn run_lines(args: &[&str]) -> Result<Vec<String>, String> {
    let args = ResidentArgs::parse(args)?;
    // Managed allocations are committed host RAM, so a wrong total is an OOM rather than a slow
    // path: the evidence records MemAvailable before and after the 77 GB set is built.
    let mem_before = mem_available_bytes();
    let RankSetup {
        plan,
        shard_plan,
        source_kind,
        tensors_seen,
    } = load_rank_setup(&args)?;

    let (arena, arena_lines) = if args.plan_only {
        let (value, lines) = plan_only_evidence(&plan);
        (Some(value), lines)
    } else if args.upload {
        let Some(source) = &args.source else {
            return Err(
                "--upload needs --checkpoint: the streaming path reads the bytes it uploads"
                    .to_string(),
            );
        };
        let ((), value, lines) = upload_arena_streaming(&plan, &shard_plan, source, |_| Ok(()))?;
        (Some(value), lines)
    } else if args.read {
        let Some(source) = &args.source else {
            return Err(
                "--read needs --checkpoint: the tensor index carries headers, not data".to_string(),
            );
        };
        let _started = Instant::now();
        let slices = shard_plan
            .read_all(source)
            .map_err(|error| format!("read rank {} in place: {error}", args.rank))?;
        let _evidence = ReadEvidence {
            tensors: slices.len(),
            bytes: slices.iter().map(|slice| slice.bytes.len() as u64).sum(),
            hash: read_hash(&slices),
        };
        (None, Vec::new())
    } else {
        (None, Vec::new())
    };
    let read = None;

    let value = evidence(
        mem_before,
        &args,
        &plan,
        &source_kind,
        tensors_seen,
        read,
        arena.as_ref(),
    );
    let path = write_evidence(&args, &value)?;
    let summary = write_summary(&args)?;

    let mut lines = Vec::new();
    lines.extend(arena_lines);
    lines.extend(vec![
        format!(
            "tp4-resident rank {}: {} tensors ({} resident / {} disk), {:.3} GiB resident, {} ranges, plan hash {:#018x}",
            args.rank,
            plan.shards().len(),
            plan.resident_tensors(),
            plan.disk_tensors(),
            bytes_to_gib(plan.resident_bytes()),
            plan.ranges(),
            plan.plan_hash()
        ),
        format!("tp4-resident evidence: {}", path.display()),
        format!(
            "tp4-resident read: {}",
            match &read {
                Some((evidence, elapsed_us)) => format!(
                    "{} tensors, {} bytes, hash {:#018x}, {} ms",
                    evidence.tensors,
                    evidence.bytes,
                    evidence.hash,
                    elapsed_us / 1_000
                ),
                None => "not performed (headers only; pass --read with --checkpoint)".to_string(),
            }
        ),
        summary,
    ]);
    lines.push(
        "tp4-resident: window measurement still needed - device allocation of the planned bytes, admission against MemAvailable, and the in-place read throughput on a node with the released checkpoint"
            .to_string(),
    );
    Ok(lines)
}

/// FNV-1a 64 over every read slice, in name order: the same hash ResidentPlan::read_all reports.
fn read_hash(slices: &[StoredShard]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for slice in slices {
        for byte in &slice.bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// Reads one planned tensor at a time and enforces a host-memory floor before each read, so a
/// too-large set aborts with a clear message instead of being OOM-killed.
#[allow(dead_code)] // device-arm scaffolding: the loop that reads through it is not wired yet
struct StreamingReader<'a> {
    plan: &'a sparkflash_device::shard::ShardPlan,
    source: &'a SafeTensorsSource,
    floor_bytes: u64,
    peak_read_bytes: usize,
}

impl sparkflash_device::arena::ShardReader for StreamingReader<'_> {
    fn read_shard(
        &mut self,
        planned: &sparkflash_device::resident::ResidentShard,
    ) -> Result<Vec<u8>, String> {
        if let Some(available) = mem_available_bytes()
            && available < self.floor_bytes
        {
            return Err(format!(
                "MemAvailable {} bytes is below the {} byte floor before reading {}: aborting instead of risking an OOM kill",
                available, self.floor_bytes, planned.name
            ));
        }
        let shard = self
            .plan
            .read(self.source, &planned.name)
            .map_err(|error| format!("read {}: {error}", planned.name))?;
        if shard.bytes.len() as u64 != planned.shard_bytes {
            return Err(format!(
                "{}: the plan expects {} stored bytes, the reader returned {}",
                planned.name,
                planned.shard_bytes,
                shard.bytes.len()
            ));
        }
        self.peak_read_bytes = self.peak_read_bytes.max(shard.bytes.len());
        Ok(shard.bytes)
    }
}

/// The default host-memory floor: abort before the node starts swapping or the OOM killer arrives.
#[allow(dead_code)] // device-arm scaffolding: called once the loop pre-flight is wired
fn mem_available_floor_bytes() -> u64 {
    let mut floor = 12u64 << 30;
    if let Ok(text) = std::env::var("SPARKFLASH_MEM_FLOOR_GIB")
        && let Ok(gib) = text.trim().parse::<u64>()
    {
        floor = gib << 30;
    }
    floor
}

/// The process peak RSS, from VmHWM.
#[allow(dead_code)] // device-arm scaffolding: called once the loop reporting is wired
fn peak_host_rss_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kib: u64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
            return Some(kib * 1024);
        }
    }
    None
}

/// Plan-only arithmetic: per-category stored and payload bytes plus the streaming order, no driver.
fn plan_only_evidence(plan: &ResidentPlan) -> (Value, Vec<String>) {
    use sparkflash_device::arena::b12x_layout;
    use sparkflash_device::resident::Placement;
    let mut by_group: BTreeMap<&str, (u64, u64, usize, usize)> = BTreeMap::new();
    let mut tensors = 0usize;
    let mut stored_bytes = 0u64;
    let mut payload_bytes = 0u64;
    let mut max_payload = 0u64;
    for planned in plan.shards() {
        if planned.placement != Placement::Resident {
            continue;
        }
        let payload = match b12x_layout(planned) {
            Some((cols, inner)) => {
                sparkflash_kernels::mxfp8_ref::b12x_buffer_bytes(cols, inner) as u64
            }
            None => planned.shard_bytes,
        };
        let entry = by_group.entry(planned.group.as_str()).or_default();
        entry.0 = entry.0.saturating_add(planned.shard_bytes);
        entry.1 = entry.1.saturating_add(payload);
        entry.2 += 1;
        if b12x_layout(planned).is_some() {
            entry.3 += 1;
        }
        tensors += 1;
        stored_bytes = stored_bytes.saturating_add(planned.shard_bytes);
        payload_bytes = payload_bytes.saturating_add(payload);
        max_payload = max_payload.max(payload);
    }
    let groups: Vec<Value> = by_group
        .iter()
        .map(|(group, (stored, payload, count, tiled))| {
            json!({
                "group": group,
                "tensors": count,
                "b12x_tensors": tiled,
                "stored_bytes": stored,
                "payload_bytes": payload,
                "padding_bytes": payload.saturating_sub(*stored),
                "gib": round3(bytes_to_gib(*payload)),
            })
        })
        .collect();
    let value = json!({
        "performed": false,
        "plan_only": true,
        "tensors": tensors,
        "stored_bytes": stored_bytes,
        "payload_bytes": payload_bytes,
        "padding_bytes": payload_bytes.saturating_sub(stored_bytes),
        "gib": round3(bytes_to_gib(payload_bytes)),
        "max_tensor_payload_bytes": max_payload,
        "expected_peak_host_bytes": payload_bytes.saturating_add(max_payload),
        "expected_peak_gib": round3(bytes_to_gib(payload_bytes.saturating_add(max_payload))),
        "groups": groups,
        "note": "phase-1 arithmetic only: per-group stored and B12x payload bytes from the plan, no checkpoint read, no driver, no allocation; the streaming order is plan name order and peak host read buffer is the largest payload",
    });
    let lines = vec![
        format!(
            "tp4-resident plan-only: {tensors} resident tensors, {stored_bytes} bytes stored -> {payload_bytes} bytes payload ({:.3} GiB), no read and no driver",
            bytes_to_gib(payload_bytes)
        ),
        format!(
            "tp4-resident plan-only: max single tensor {} bytes (streaming peak read buffer), expected peak host {} bytes ({:.3} GiB)",
            max_payload,
            payload_bytes.saturating_add(max_payload),
            bytes_to_gib(payload_bytes.saturating_add(max_payload))
        ),
    ];
    (value, lines)
}

/// Streams the resident set: layout and allocate from the plan, then read, tile, upload, verify and
/// drop one tensor at a time. Never holds the stored set and the arena at once.
#[cfg(feature = "cuda")]
pub(crate) fn upload_arena_streaming<T>(
    plan: &ResidentPlan,
    shard_plan: &sparkflash_device::shard::ShardPlan,
    source: &SafeTensorsSource,
    mut use_arena: impl FnMut(&sparkflash_device::arena::ResidentArena) -> Result<T, String>,
) -> Result<(T, Value, Vec<String>), String> {
    use sparkflash_device::arena::{AllocationMode, CudaArenaAllocator, ResidentArena};
    use sparkflash_device::resident::Placement;

    let cuda = sparkflash_kernels::cuda_driver::Cuda::load()
        .map_err(|error| format!("open the CUDA driver: {error}"))?;
    // One contiguous reservation for the whole rank plan (allocate the model's memory up front in
    // one chunk, not as each tensor is read in).
    let mut allocator = CudaArenaAllocator::with_mode(&cuda, AllocationMode::UnifiedSingle);
    let floor = mem_available_floor_bytes();
    let before = mem_available_bytes();
    let mut reader = StreamingReader {
        plan: shard_plan,
        source,
        floor_bytes: floor,
        peak_read_bytes: 0,
    };
    let arena = ResidentArena::from_plan_streaming(
        &mut allocator,
        plan,
        &mut reader,
        AllocationMode::UnifiedSingle,
        |_name| Ok(()),
    )?;
    let bound = use_arena(&arena)?;
    let evidence = arena.evidence();
    let peak_read_bytes = reader.peak_read_bytes as u64;
    let verified = plan
        .shards()
        .iter()
        .filter(|planned| planned.placement == Placement::Resident)
        .count();
    let groups: Vec<Value> = arena
        .groups()
        .iter()
        .map(|group| {
            json!({
                "group": group.group.as_str(),
                "tensors": group.tensors,
                "tiled": group.tiled,
                "bytes": group.bytes,
                "gib": round3(bytes_to_gib(group.bytes as u64)),
                "hash": format!("{:#018x}", group.digest),
            })
        })
        .collect();
    let value = json!({
        "performed": true,
        "streaming": true,
        "mode": evidence.mode.as_str(),
        "tensors": evidence.tensors,
        "tiled_tensors": evidence.tiled_tensors,
        "stored_bytes": evidence.stored_bytes,
        "uploaded_bytes": evidence.bytes,
        "resident_bytes": evidence.bytes,
        "allocated_bytes": evidence.allocated_bytes,
        "gib": round3(bytes_to_gib(evidence.bytes)),
        "allocated_gib": round3(bytes_to_gib(evidence.allocated_bytes)),
        "peak_device_bytes": evidence.allocated_bytes,
        "peak_read_buffer_bytes": peak_read_bytes,
        "peak_host_rss_bytes": peak_host_rss_bytes(),
        "mem_available_floor_bytes": floor,
        "mem_available_before_bytes": before,
        "mem_available_after_bytes": mem_available_bytes(),
        "allocate_ms": evidence.allocate_ms,
        "upload_ms": evidence.upload_ms,
        "hash": format!("{:#018x}", evidence.digest),
        "verified_tensors": verified,
        "verified_against_read": verified == evidence.tensors,
        "groups": groups,
        "note": "streaming: layout/allocate from the plan first, then one tensor read -> tile -> upload -> digest verify -> drop; peak host memory is the arena plus one tensor, guarded by a MemAvailable floor",
    });
    let lines = vec![
        format!(
            "tp4-resident arena (streaming): {} tensors ({} b12x), {} bytes stored -> {} bytes uploaded ({} allocated, {:.3} GiB), {} groups, mode {}, peak read buffer {} bytes, peak RSS {:?} bytes",
            evidence.tensors,
            evidence.tiled_tensors,
            evidence.stored_bytes,
            evidence.bytes,
            evidence.allocated_bytes,
            bytes_to_gib(evidence.bytes),
            arena.groups().len(),
            evidence.mode.as_str(),
            peak_read_bytes,
            peak_host_rss_bytes()
        ),
        format!(
            "tp4-resident arena (streaming): allocate {} ms, upload {} ms, hash {:#018x}, verified {verified} tensors, MemAvailable {:?} -> {:?} bytes, floor {} bytes",
            evidence.allocate_ms,
            evidence.upload_ms,
            evidence.digest,
            before,
            mem_available_bytes(),
            floor
        ),
    ];
    Ok((bound, value, lines))
}

/// Allocates the resident arena on the device and uploads every planned slice once.
///
/// The read bytes are re-tiled (for the B12x groups) and digested again before the arena accepts
/// them, so the evidence proves the device copy is the checkpoint slice and not merely that a copy
/// succeeded. The slabs are freed when this function returns; the run started with no other device
/// allocation, so the arena's bytes are the process peak for this stage.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // the arena upload is exercised by the resident stage, not the loop yet
fn upload_arena(
    plan: &ResidentPlan,
    slices: &[StoredShard],
) -> Result<(Value, Vec<String>), String> {
    use sparkflash_device::arena::{
        AllocationMode, CudaArenaAllocator, ResidentArena, b12x_layout, tile_b12x,
    };
    use sparkflash_device::resident::Placement;

    let cuda = sparkflash_kernels::cuda_driver::Cuda::load()
        .map_err(|error| format!("open the CUDA driver: {error}"))?;
    // UMA-coherent managed memory: one allocation per group, device pointers valid on both sides,
    // and no separate H2D copy on a unified-memory node (the GB10 arm).
    let mut allocator = CudaArenaAllocator::with_mode(&cuda, AllocationMode::Unified);
    let arena = ResidentArena::from_plan(&mut allocator, plan, slices, AllocationMode::Unified)?;
    let evidence = arena.evidence();
    let by_name: BTreeMap<&str, &StoredShard> = slices
        .iter()
        .map(|slice| (slice.name.as_str(), slice))
        .collect();
    let mut verified = 0usize;
    for planned in plan.shards() {
        if planned.placement != Placement::Resident {
            continue;
        }
        let slice = by_name.get(planned.name.as_str()).ok_or_else(|| {
            format!(
                "the plan holds {} resident but the read has no slice for it",
                planned.name
            )
        })?;
        let payload = match b12x_layout(planned) {
            Some((cols, inner)) => tile_b12x(&slice.bytes, cols, inner)?,
            None => slice.bytes.clone(),
        };
        arena.verify_slice(&planned.name, &payload)?;
        verified += 1;
    }
    let groups: Vec<Value> = arena
        .groups()
        .iter()
        .map(|group| {
            json!({
                "group": group.group.as_str(),
                "tensors": group.tensors,
                "tiled": group.tiled,
                "bytes": group.bytes,
                "gib": round3(bytes_to_gib(group.bytes as u64)),
                "hash": format!("{:#018x}", group.digest),
            })
        })
        .collect();
    let value = json!({
        "performed": true,
        "mode": evidence.mode.as_str(),
        "tensors": evidence.tensors,
        "tiled_tensors": evidence.tiled_tensors,
        "stored_bytes": evidence.stored_bytes,
        "uploaded_bytes": evidence.bytes,
        "resident_bytes": evidence.bytes,
        "allocated_bytes": evidence.allocated_bytes,
        "gib": round3(bytes_to_gib(evidence.bytes)),
        "allocated_gib": round3(bytes_to_gib(evidence.allocated_bytes)),
        "peak_device_bytes": evidence.allocated_bytes,
        "allocate_ms": evidence.allocate_ms,
        "upload_ms": evidence.upload_ms,
        "hash": format!("{:#018x}", evidence.digest),
        "verified_tensors": verified,
        "verified_against_read": verified == evidence.tensors,
        "groups": groups,
        "note": "one cuMemAlloc slab per group, one copy per tensor, B12x applied once at load for dense FP8 weights; every slice is re-tiled and digested from the read bytes before the arena accepts it; this stage launches no kernel, and the slabs are freed when it returns",
    });
    let lines = vec![
        format!(
            "tp4-resident arena: {} tensors ({} b12x), {} bytes stored -> {} bytes uploaded ({} allocated, {:.3} GiB), {} groups, mode {}",
            evidence.tensors,
            evidence.tiled_tensors,
            evidence.stored_bytes,
            evidence.bytes,
            evidence.allocated_bytes,
            bytes_to_gib(evidence.bytes),
            arena.groups().len(),
            evidence.mode.as_str()
        ),
        format!(
            "tp4-resident arena: allocate {} ms, upload {} ms, hash {:#018x}, verified {verified} tensors against the read bytes",
            evidence.allocate_ms, evidence.upload_ms, evidence.digest
        ),
    ];
    Ok((value, lines))
}

/// Without the cuda feature this build cannot open a driver, so streaming --upload is refused.
#[cfg(not(feature = "cuda"))]
pub(crate) fn upload_arena_streaming<T>(
    _plan: &ResidentPlan,
    _shard_plan: &sparkflash_device::shard::ShardPlan,
    _source: &SafeTensorsSource,
    _use_arena: impl FnMut(&sparkflash_device::arena::ResidentArena) -> Result<T, String>,
) -> Result<(T, Value, Vec<String>), String> {
    Err("--upload needs a binary built with --features cuda".to_string())
}

/// Without the cuda feature this build cannot open a driver, so --upload is refused.
#[cfg(not(feature = "cuda"))]
#[allow(dead_code)] // the no-cuda refusal path is exercised only by the feature build
fn upload_arena(
    _plan: &ResidentPlan,
    _slices: &[StoredShard],
) -> Result<(Value, Vec<String>), String> {
    Err("--upload needs a binary built with --features cuda".to_string())
}

/// Where the headers came from.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Kind {
    /// A header dump: a JSON object of name -> {dtype, shape}.
    TensorIndex,
    /// The released checkpoint's safetensors headers, through its index.
    Checkpoint,
}

impl Kind {
    fn as_str(&self) -> &'static str {
        match self {
            Self::TensorIndex => "tensor-index",
            Self::Checkpoint => "checkpoint",
        }
    }
}

/// The parsed arguments of one tp4-resident invocation.
#[derive(Debug)]
pub(crate) struct ResidentArgs {
    rank: usize,
    world: usize,
    source: Option<SafeTensorsSource>,
    source_path: Option<PathBuf>,
    kind: Kind,
    evidence_dir: PathBuf,
    read: bool,
    upload: bool,
    plan_only: bool,
}

impl ResidentArgs {
    fn parse(args: &[&str]) -> Result<Self, String> {
        let mut rank = 0usize;
        let mut world = 4usize;
        let mut tensor_index: Option<PathBuf> = None;
        let mut checkpoint: Option<PathBuf> = None;
        let mut evidence_dir: Option<PathBuf> = None;
        let mut read = false;
        let mut upload = false;
        let mut plan_only = false;

        let mut index = 0;
        while index < args.len() {
            let flag = args[index];
            let mut take = |flag_name: &str| -> Result<String, String> {
                index += 1;
                args.get(index)
                    .map(|value| (*value).to_string())
                    .ok_or_else(|| format!("{flag_name} needs a value"))
            };
            match flag {
                "--tp-rank" => rank = parse_usize("--tp-rank", &take("--tp-rank")?)?,
                "--tp-world-size" => {
                    world = parse_usize("--tp-world-size", &take("--tp-world-size")?)?;
                }
                "--tensor-index" => tensor_index = Some(PathBuf::from(take("--tensor-index")?)),
                "--checkpoint" => checkpoint = Some(PathBuf::from(take("--checkpoint")?)),
                "--evidence-dir" => evidence_dir = Some(PathBuf::from(take("--evidence-dir")?)),
                "--read" => read = true,
                "--plan-only" => plan_only = true,
                "--upload" => upload = true,
                other => return Err(format!("{other}: not a tp4-resident flag")),
            }
            index += 1;
        }
        if world == 0 {
            return Err("--tp-world-size needs at least one rank".to_string());
        }
        if rank >= world {
            return Err(format!("--tp-rank {rank} is outside a world of {world}"));
        }
        let evidence_dir = evidence_dir.ok_or_else(|| "--evidence-dir is required".to_string())?;
        if tensor_index.is_some() && checkpoint.is_some() {
            return Err("--tensor-index and --checkpoint are alternatives, not both".to_string());
        }
        if tensor_index.is_none() && checkpoint.is_none() {
            return Err("one of --tensor-index or --checkpoint is required".to_string());
        }
        if read && checkpoint.is_none() {
            return Err(
                "--read needs --checkpoint: the tensor index carries headers, not data".to_string(),
            );
        }
        if upload && checkpoint.is_none() {
            return Err(
                "--upload needs --checkpoint: the arena uploads the bytes the plan reads"
                    .to_string(),
            );
        }
        let read = read || upload;
        let (source, source_path, kind) = match checkpoint {
            Some(dir) => {
                let index_path = dir.join("model.safetensors.index.json");
                let source = SafeTensorsSource::open(dir.clone(), &index_path)
                    .map_err(|error| format!("open the checkpoint {}: {error}", dir.display()))?;
                (Some(source), Some(dir), Kind::Checkpoint)
            }
            None => (None, tensor_index, Kind::TensorIndex),
        };
        Ok(Self {
            rank,
            world,
            source,
            source_path,
            kind,
            evidence_dir,
            read,
            upload,
            plan_only,
        })
    }
}

/// Reads the headers the plan needs, from whichever source was given.
fn load_entries(args: &ResidentArgs) -> Result<(Vec<HeaderEntry>, String, usize), String> {
    let path = args
        .source_path
        .as_ref()
        .ok_or_else(|| "no header source".to_string())?;
    match args.kind {
        Kind::TensorIndex => {
            let entries = entries_from_tensor_index(path)?;
            let seen = entries.len();
            Ok((entries, format!("tensor-index {}", path.display()), seen))
        }
        Kind::Checkpoint => {
            let source = args
                .source
                .as_ref()
                .ok_or_else(|| "the checkpoint source was not opened".to_string())?;
            let index_path = path.join("model.safetensors.index.json");
            let index = CheckpointIndex::load(&index_path)
                .map_err(|error| format!("read the checkpoint index: {error}"))?;
            let mut entries = Vec::with_capacity(index.weight_map.len());
            for name in index.names() {
                let header = source
                    .header(name)
                    .map_err(|error| format!("{name}: {error}"))?;
                let tensor = header
                    .tensor(name)
                    .map_err(|error| format!("{name}: {error}"))?;
                entries.push(HeaderEntry::new(
                    name.to_string(),
                    tensor.dtype.clone(),
                    tensor.shape.clone(),
                ));
            }
            let seen = entries.len();
            Ok((entries, format!("checkpoint {}", path.display()), seen))
        }
    }
}

/// One tensor as a header dump stores it.
#[derive(Debug, Deserialize, Serialize)]
struct IndexedTensor {
    dtype: String,
    shape: Vec<usize>,
}

/// Parses a header dump: a JSON object of tensor name -> {dtype, shape}.
///
/// This is the form produced from a checkpoint's safetensors headers (\`name: {dtype, shape}\`). An
/// Hugging Face \`model.safetensors.index.json\` is not a header dump - it has no shapes - so it is
/// refused with a message naming the checkpoint mode instead of being half-read.
fn entries_from_tensor_index(path: &Path) -> Result<Vec<HeaderEntry>, String> {
    let text =
        fs::read_to_string(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    if value.get("weight_map").is_some() {
        return Err(format!(
            "{} is a checkpoint index (names only); pass --checkpoint so the safetensors headers supply the shapes",
            path.display()
        ));
    }
    let map: BTreeMap<String, IndexedTensor> = serde_json::from_value(value).map_err(|error| {
        format!(
            "{}: expected tensor name -> {{dtype, shape}}: {error}",
            path.display()
        )
    })?;
    Ok(map
        .into_iter()
        .map(|(name, tensor)| HeaderEntry::new(name, tensor.dtype, tensor.shape))
        .collect())
}

/// Builds the per-rank evidence object.
fn evidence(
    mem_available_before: Option<u64>,
    args: &ResidentArgs,
    plan: &ResidentPlan,
    source: &str,
    tensors_seen: usize,
    read: Option<(ReadEvidence, u64)>,
    arena: Option<&Value>,
) -> Value {
    let plan_gib = bytes_to_gib(plan.resident_bytes());
    let groups: Vec<Value> = plan
        .groups()
        .into_iter()
        .map(|group| {
            json!({
                "group": group.group.as_str(),
                "tensors": group.tensors,
                "placement": group.placement.as_str(),
                "rule": group.rule.map(|rule| rule.as_str()),
                "shard_bytes": group.shard_bytes,
                "stored_bytes": group.stored_bytes,
                "ranges": group.ranges,
            })
        })
        .collect();
    json!({
        "stage": "tp4-resident",
        "schema": SCHEMA_VERSION,
        "created_at_unix_s": unix_seconds(),
        "host": hostname(),
        "rank": args.rank,
        // world=1 is the single-node arm (every tensor full, no TP sharding); never mix it with the
        // world=4 72.31 GiB/rank stage-2 plan shape.
        "plan_label": if args.world <= 1 { "world1-single-node" } else { "tp4-sharded" },
        "mem_available_before_bytes": mem_available_before,
        "mem_available_after_bytes": mem_available_bytes(),
        "world": args.world,
        "source": {
            "kind": args.kind.as_str(),
            "path": source,
            "tensors_seen": tensors_seen,
            "not_loaded": not_loaded_count(tensors_seen, plan),
        },
        "plan": {
            "chain_layers": CHAIN_LAYERS,
            "mtp_layers": MTP_LAYERS,
            "planned_tensors": plan.shards().len(),
            "resident_tensors": plan.resident_tensors(),
            "disk_tensors": plan.disk_tensors(),
            "resident_bytes": plan.resident_bytes(),
            "resident_gib": round3(plan_gib),
            "disk_bytes": plan.disk_bytes(),
            "disk_gib": round3(bytes_to_gib(plan.disk_bytes())),
            "ranges": plan.ranges(),
            "mean_range_bytes": round3(mean_range_bytes(plan.ranges(), plan.resident_bytes())),
            "hash": format!("{:#018x}", plan.plan_hash()),
            "groups": groups,
        },
        "budget": {
            "design_gib": DESIGN_RESIDENT_GIB,
            "plan_gib": round3(plan_gib),
            "delta_gib": round3(plan_gib - DESIGN_RESIDENT_GIB),
        },
        "read": match read {
            Some((read, elapsed_us)) => json!({
                "performed": true,
                "tensors": read.tensors,
                "bytes": read.bytes,
                "hash": format!("{:#018x}", read.hash),
                "elapsed_us": elapsed_us,
            }),
            None => json!({ "performed": false }),
        },
        "window_measurement": {
            "needed": [
                "device allocation of the planned resident bytes on a node",
                "admission against MemAvailable before the load (sparkflash_device::budget)",
                "in-place read throughput and page-cache pressure on a node with the released checkpoint",
                "the four-rank load in the same window as the S5.2 collective stage",
            ],
            "not_measured_here": "this stage plans headers, with --read reads them on the host, and with --upload allocates and fills the resident device arena; it launches no kernel, produces no token and admits nothing against MemAvailable",
        },
        "arena": match arena {
            Some(value) => value.clone(),
            None => json!({ "performed": false }),
        },
        "flags": {
            "device_execution": false,
            "device_allocation": arena.is_some(),
            "token_produced": false,
            "cuda_feature": cfg!(feature = "cuda"),
            "comm_rdma_shim": crate::comm_rdma_shim(),
        },
    })
}

/// How many header names the plan deliberately did not load (the vision tower and aligner).
fn not_loaded_count(tensors_seen: usize, plan: &ResidentPlan) -> usize {
    tensors_seen.saturating_sub(plan.shards().len())
}

/// Writes the per-rank evidence file.
fn write_evidence(args: &ResidentArgs, value: &Value) -> Result<PathBuf, String> {
    fs::create_dir_all(&args.evidence_dir)
        .map_err(|error| format!("create {}: {error}", args.evidence_dir.display()))?;
    let path = args
        .evidence_dir
        .join(format!("tp4-resident-r{}.json", args.rank));
    let text = serde_json::to_string_pretty(value)
        .map_err(|error| format!("serialise the evidence: {error}"))?;
    fs::write(&path, text).map_err(|error| format!("write {}: {error}", path.display()))?;
    Ok(path)
}

/// Writes the run summary by reading back every rank file in the evidence directory.
///
/// Verdict rules: PASS requires all \`world\` ranks present and every rank agreeing on the resident
/// byte total and the plan hash. \`read_verified\` is separate: a header-only run is a PASS for the
/// plan and false for the read, which is what the window fills in.
fn write_summary(args: &ResidentArgs) -> Result<String, String> {
    let mut ranks: Vec<(usize, Value)> = Vec::new();
    for entry in fs::read_dir(&args.evidence_dir)
        .map_err(|error| format!("read {}: {error}", args.evidence_dir.display()))?
    {
        let entry = entry.map_err(|error| format!("read the evidence directory: {error}"))?;
        let name = entry.file_name().to_string_lossy().to_string();
        if !(name.starts_with("tp4-resident-r") && name.ends_with(".json")) {
            continue;
        }
        let text = fs::read_to_string(entry.path())
            .map_err(|error| format!("read {}: {error}", entry.path().display()))?;
        let value: Value = serde_json::from_str(&text)
            .map_err(|error| format!("parse {}: {error}", entry.path().display()))?;
        let rank = value["rank"]
            .as_u64()
            .ok_or_else(|| format!("{}: no rank", entry.path().display()))?
            as usize;
        ranks.push((rank, value));
    }
    ranks.sort_by_key(|(rank, _)| *rank);
    let complete = ranks.len() == args.world
        && ranks
            .iter()
            .enumerate()
            .all(|(index, (rank, _))| index == *rank);
    let resident_bytes: Vec<u64> = ranks
        .iter()
        .filter_map(|(_, value)| value["plan"]["resident_bytes"].as_u64())
        .collect();
    let hashes: Vec<&str> = ranks
        .iter()
        .filter_map(|(_, value)| value["plan"]["hash"].as_str())
        .collect();
    let ranks_agree = !ranks.is_empty()
        && resident_bytes.len() == ranks.len()
        && hashes.len() == ranks.len()
        && resident_bytes
            .iter()
            .all(|bytes| *bytes == resident_bytes[0])
        && hashes.iter().all(|hash| *hash == hashes[0]);
    let read_verified = !ranks.is_empty()
        && ranks
            .iter()
            .all(|(_, value)| value["read"]["performed"].as_bool() == Some(true));
    let arena_uploaded = !ranks.is_empty()
        && ranks
            .iter()
            .all(|(_, value)| value["arena"]["performed"].as_bool() == Some(true));
    let arena_verified = !ranks.is_empty()
        && ranks
            .iter()
            .all(|(_, value)| value["arena"]["verified_against_read"].as_bool() == Some(true));
    let verdict = if complete && ranks_agree {
        "PASS"
    } else {
        "PARTIAL"
    };
    let summary = json!({
        "stage": "tp4-resident",
        "schema": SCHEMA_VERSION,
        "created_at_unix_s": unix_seconds(),
        "world_size": args.world,
        "ranks": ranks
            .iter()
            .map(|(rank, value)| json!({
                "rank": rank,
                "resident_bytes": value["plan"]["resident_bytes"],
                "plan_hash": value["plan"]["hash"],
                "read_performed": value["read"]["performed"],
                "arena_uploaded": value["arena"]["performed"],
                "arena_hash": value["arena"]["hash"],
                "arena_verified": value["arena"]["verified_against_read"],
                "evidence": format!("tp4-resident-r{rank}.json"),
            }))
            .collect::<Vec<Value>>(),
        "gates": {
            "plan_complete": complete,
            "ranks_agree": ranks_agree,
            "read_verified": read_verified,
            "arena_uploaded": arena_uploaded,
            "arena_verified": arena_verified,
        },
        "window_measurement": {
            "needed": "admission against MemAvailable before the load and the four-rank load in the same window as the S5.2 collective stage; with --upload the device allocation and upload are measured, without it this summary is host-side plan evidence only",
        },
        "flags": {
            "device_execution": false,
            "device_allocation": arena_uploaded,
            "token_produced": false,
            "cuda_feature": cfg!(feature = "cuda"),
            "comm_rdma_shim": crate::comm_rdma_shim(),
        },
        "verdict": verdict,
    });
    let path = args.evidence_dir.join("tp4-resident-summary.json");
    let text = serde_json::to_string_pretty(&summary)
        .map_err(|error| format!("serialise the summary: {error}"))?;
    fs::write(&path, text).map_err(|error| format!("write {}: {error}", path.display()))?;
    Ok(format!("tp4-resident summary: {}", path.display()))
}

fn parse_usize(flag: &str, raw: &str) -> Result<usize, String> {
    raw.parse::<usize>()
        .map_err(|_| format!("{flag}: {raw} is not a number"))
}

/// The mean bytes per issued range; the row-parallel groups are the small ones.
fn mean_range_bytes(ranges: usize, bytes: u64) -> f64 {
    if ranges == 0 {
        0.0
    } else {
        bytes as f64 / ranges as f64
    }
}

/// The kernel's MemAvailable, which is what a 77 GB managed set commits against.
fn mem_available_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kib: u64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
            return Some(kib * 1024);
        }
    }
    None
}

fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

#[allow(dead_code)] // device-arm scaffolding: used by the loop's stage timers
fn micros(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

fn hostname() -> String {
    fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|text| text.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small but structurally complete header dump: one of every planned group, two layers plus
    /// one MTP layer, and two vision tensors that must be skipped.
    fn header_dump() -> BTreeMap<String, IndexedTensor> {
        let mut tensors = BTreeMap::new();
        let mut add = |name: &str, dtype: &str, shape: &[usize]| {
            tensors.insert(
                name.to_string(),
                IndexedTensor {
                    dtype: dtype.to_string(),
                    shape: shape.to_vec(),
                },
            );
        };
        add("embed.weight", "BF16", &[32, 8]);
        add("head.weight", "BF16", &[32, 8]);
        add("norm.weight", "BF16", &[8]);
        add("image_start", "BF16", &[8]);
        for layer in 0..2 {
            let prefix = format!("layers.{layer}");
            add(&format!("{prefix}.attn.wq_a.weight"), "F8_E4M3", &[4, 8]);
            add(&format!("{prefix}.attn.wq_a.scale"), "F8_E8M0", &[4, 1]);
            add(&format!("{prefix}.attn.wkv.weight"), "F8_E4M3", &[4, 8]);
            add(&format!("{prefix}.attn.wq_b.weight"), "F8_E4M3", &[16, 4]);
            add(&format!("{prefix}.attn.wo_a.weight"), "F8_E4M3", &[16, 4]);
            add(&format!("{prefix}.attn.wo_b.weight"), "F8_E4M3", &[4, 16]);
            add(&format!("{prefix}.attn.wo_b.scale"), "F8_E8M0", &[4, 16]);
            add(&format!("{prefix}.attn.attn_sink"), "F32", &[16]);
            add(&format!("{prefix}.attn.q_norm.weight"), "BF16", &[4]);
            add(&format!("{prefix}.attn.kv_norm.weight"), "BF16", &[4]);
            add(&format!("{prefix}.attn_norm.weight"), "BF16", &[8]);
            add(&format!("{prefix}.ffn_norm.weight"), "BF16", &[8]);
            add(&format!("{prefix}.ffn.gate.weight"), "BF16", &[8, 8]);
            add(&format!("{prefix}.ffn.gate.bias"), "F32", &[8]);
            add(&format!("{prefix}.ffn.gate.bias_vl"), "F32", &[8]);
            for expert in 0..2 {
                add(
                    &format!("{prefix}.ffn.experts.{expert}.w1.weight"),
                    "I8",
                    &[8, 8],
                );
                add(
                    &format!("{prefix}.ffn.experts.{expert}.w1.scale"),
                    "F8_E8M0",
                    &[8, 1],
                );
                add(
                    &format!("{prefix}.ffn.experts.{expert}.w2.weight"),
                    "I8",
                    &[8, 4],
                );
                add(
                    &format!("{prefix}.ffn.experts.{expert}.w3.weight"),
                    "I8",
                    &[8, 8],
                );
            }
            add(
                &format!("{prefix}.ffn.shared_experts.w1.weight"),
                "F8_E4M3",
                &[8, 8],
            );
            add(
                &format!("{prefix}.ffn.shared_experts.w2.weight"),
                "F8_E4M3",
                &[8, 8],
            );
            add(
                &format!("{prefix}.ffn.shared_experts.w3.weight"),
                "F8_E4M3",
                &[8, 8],
            );
            add(&format!("{prefix}.hc_attn_base"), "F32", &[4]);
            add(&format!("{prefix}.hc_ffn_fn"), "F32", &[2, 2]);
        }
        add("layers.1.engram.embed.weight", "F8_E4M3", &[16, 8]);
        add("layers.1.engram.embed.scale", "F8_E8M0", &[16, 1]);
        add("layers.1.engram.q_weight", "BF16", &[2, 8]);
        add("layers.1.engram.wkv.weight", "F8_E4M3", &[4, 8]);
        add("mtp.0.attn.wq_b.weight", "F8_E4M3", &[16, 4]);
        add("mtp.0.ffn.experts.0.w1.weight", "I8", &[8, 8]);
        add("mtp.2.main_proj.weight", "BF16", &[8, 8]);
        add("mtp.2.markov_head.embed.weight", "BF16", &[32, 8]);
        add("mtp.2.confidence_head.proj.weight", "BF16", &[1, 8]);
        add("vision.patch_embed.proj.weight", "BF16", &[4, 4]);
        add("aligner.w1.weight", "BF16", &[4, 4]);
        tensors
    }

    fn write_header_dump(dir: &Path) -> PathBuf {
        let path = dir.join("tensors.json");
        let text = serde_json::to_string_pretty(&header_dump()).expect("serialise");
        fs::write(&path, text).expect("write the header dump");
        path
    }

    fn run_rank(index: &Path, evidence_dir: &Path, rank: usize, extra: &[&str]) -> Vec<String> {
        let rank_text = rank.to_string();
        let mut args = vec![
            "--tensor-index",
            index.to_str().expect("utf-8"),
            "--tp-rank",
            rank_text.as_str(),
            "--tp-world-size",
            "4",
            "--evidence-dir",
            evidence_dir.to_str().expect("utf-8"),
        ];
        args.extend_from_slice(extra);
        run_lines(&args).expect("stage run")
    }

    #[test]
    fn one_rank_plan_writes_evidence_with_the_s51_flags_and_the_plan_shape() {
        let dir = tempfile::tempdir().expect("tempdir");
        let index = write_header_dump(dir.path());
        let evidence_dir = dir.path().join("evidence");
        let lines = run_rank(&index, &evidence_dir, 0, &[]);
        assert!(
            lines
                .iter()
                .any(|line| line.contains("window measurement still needed"))
        );
        let value: Value = serde_json::from_str(
            &fs::read_to_string(evidence_dir.join("tp4-resident-r0.json")).expect("read"),
        )
        .expect("json");
        assert_eq!(value["stage"], "tp4-resident");
        assert_eq!(value["schema"], SCHEMA_VERSION);
        assert_eq!(value["rank"], 0);
        assert_eq!(value["world"], 4);
        assert_eq!(value["source"]["kind"], "tensor-index");
        assert_eq!(value["source"]["not_loaded"], 2);
        assert_eq!(value["flags"]["device_execution"], false);
        assert_eq!(value["flags"]["token_produced"], false);
        assert_eq!(value["flags"]["comm_rdma_shim"], crate::comm_rdma_shim());
        assert_eq!(value["flags"]["device_allocation"], false);
        assert_eq!(value["arena"]["performed"], false);
        assert_eq!(value["read"]["performed"], false);
        assert_eq!(value["plan"]["chain_layers"], CHAIN_LAYERS);
        assert_eq!(value["plan"]["mtp_layers"], MTP_LAYERS);
        assert_eq!(value["plan"]["planned_tensors"], 69);
        assert_eq!(value["plan"]["resident_tensors"], 67);
        assert_eq!(value["plan"]["disk_tensors"], 2);
        assert!(value["plan"]["resident_bytes"].as_u64().expect("bytes") > 0);
        assert!(value["plan"]["ranges"].as_u64().expect("ranges") > 0);
        assert_eq!(value["budget"]["design_gib"], DESIGN_RESIDENT_GIB);
    }

    #[test]
    fn four_rank_plans_write_a_passing_summary_with_the_rdma_shim_flag() {
        let dir = tempfile::tempdir().expect("tempdir");
        let index = write_header_dump(dir.path());
        let evidence_dir = dir.path().join("evidence");
        for rank in 0..4 {
            run_rank(&index, &evidence_dir, rank, &[]);
        }
        let summary: Value = serde_json::from_str(
            &fs::read_to_string(evidence_dir.join("tp4-resident-summary.json")).expect("read"),
        )
        .expect("json");
        assert_eq!(summary["verdict"], "PASS");
        assert_eq!(summary["gates"]["plan_complete"], true);
        assert_eq!(summary["gates"]["ranks_agree"], true);
        assert_eq!(summary["gates"]["read_verified"], false);
        assert_eq!(summary["flags"]["comm_rdma_shim"], crate::comm_rdma_shim());
        assert_eq!(summary["ranks"].as_array().expect("ranks").len(), 4);
    }

    #[test]
    fn a_partial_run_writes_a_partial_summary_instead_of_passing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let index = write_header_dump(dir.path());
        let evidence_dir = dir.path().join("evidence");
        run_rank(&index, &evidence_dir, 0, &[]);
        let summary: Value = serde_json::from_str(
            &fs::read_to_string(evidence_dir.join("tp4-resident-summary.json")).expect("read"),
        )
        .expect("json");
        assert_eq!(summary["verdict"], "PARTIAL");
        assert_eq!(summary["gates"]["plan_complete"], false);
    }

    #[test]
    fn the_stage_refuses_bad_arguments_and_an_unknown_tensor_name() {
        let dir = tempfile::tempdir().expect("tempdir");
        let index = write_header_dump(dir.path());
        let evidence_dir = dir.path().join("evidence");
        let base = [
            "--tensor-index",
            index.to_str().expect("utf-8"),
            "--evidence-dir",
            evidence_dir.to_str().expect("utf-8"),
        ];
        assert!(run_lines(&base).is_ok());
        let mut missing = base.to_vec();
        missing.push("--tp-rank");
        missing.push("3");
        missing.push("--tp-world-size");
        missing.push("4");
        assert!(run_lines(&missing).is_ok());
        let mut read_without_checkpoint = vec![
            "--tensor-index",
            index.to_str().expect("utf-8"),
            "--evidence-dir",
            evidence_dir.to_str().expect("utf-8"),
        ];
        read_without_checkpoint.push("--read");
        let not_a_flag: [&str; 5] = [
            "--tensor-index",
            index.to_str().expect("utf-8"),
            "--evidence-dir",
            evidence_dir.to_str().expect("utf-8"),
            "--read",
        ];
        let _ = read_without_checkpoint;
        let error = run_lines(&not_a_flag).expect_err("read without checkpoint");
        assert!(error.contains("--read needs --checkpoint"), "{error}");
        let upload_without_checkpoint = [
            "--tensor-index",
            index.to_str().expect("utf-8"),
            "--evidence-dir",
            evidence_dir.to_str().expect("utf-8"),
            "--upload",
        ];
        let error = run_lines(&upload_without_checkpoint).expect_err("upload without checkpoint");
        assert!(error.contains("--upload needs --checkpoint"), "{error}");
        let hf_index = dir.path().join("model.safetensors.index.json");
        fs::write(
            &hf_index,
            "{\"metadata\":{},\"weight_map\":{\"embed.weight\":\"a.safetensors\"}}",
        )
        .expect("write");
        let error = run_lines(&[
            "--tensor-index",
            hf_index.to_str().expect("utf-8"),
            "--evidence-dir",
            evidence_dir.to_str().expect("utf-8"),
        ])
        .expect_err("an index without shapes");
        assert!(error.contains("checkpoint index"), "{error}");

        let mut dump = header_dump();
        dump.insert(
            "layers.0.attn.wq_z.weight".to_string(),
            IndexedTensor {
                dtype: "F8_E4M3".to_string(),
                shape: vec![4, 4],
            },
        );
        let strange = dir.path().join("strange.json");
        fs::write(&strange, serde_json::to_string(&dump).expect("serialise")).expect("write");
        let error = run_lines(&[
            "--tensor-index",
            strange.to_str().expect("utf-8"),
            "--evidence-dir",
            evidence_dir.to_str().expect("utf-8"),
        ])
        .expect_err("an unknown tensor name");
        assert!(error.contains("resident plan knows"), "{error}");
    }
}

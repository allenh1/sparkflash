//! The attention-operand gate: a HOST-ONLY check that the released rank plan offers every
//! attention projection as the operand layout the projection kernels read.
//!
//! Why this exists. The four-rank arm binds its weights from the checkpoint's own bytes into one
//! resident arena and launches B12x-weight GEMMs against those slices. A projection whose planned
//! dtype is not F8_E4M3, or whose shape does not tile, binds as stored bytes and has no B12x
//! kernel: that would be a real device-side block (the arm has no per-layer host repack staging).
//! This gate derives the layout from the plan record alone - dtype, group, shard shape - and fails
//! by tensor name if any of the five projections on any attention layer is not B12x. It opens no
//! driver, reads no data and launches no kernel; it is the host half of the operand path, and the
//! device half is the bind itself (bind_rank_attention requires the same predicate).
//!
//! What it proves, stated as checks. For the released checkpoint at world 4 this gate asserts:
//!   * 43 attention layers (40 chain + 3 MTP) carry 215 projection weights and 215 matching scale
//!     grids, and every weight is B12x with the rank's own geometry (wq_a and wkv Replicated whole,
//!     wq_b and wo_a Column-split on cols, wo_b Row-split on inner);
//!   * every scale tensor is the stored F8_E8M0 grid whose byte count is exactly the B12x scale
//!     bytes of its weight (no scale transform, no padding);
//!   * the operand byte totals are the ones the 2026-10-20 bind window recorded for the arena:
//!     attention-replicated group 395,066,112 B (394,912,000 B of tiled wq_a+wkv codes and scales
//!     plus the 154,112 B of BF16 q/kv norms that share the group), column 812,390,400 B, row
//!     451,328,000 B; 86/86/43 tiled tensors by group and 258/172/86 group members;
//!   * optionally, the plan fingerprint: --expect-plan-hash 0x9a59035fc40776c8 is the released
//!     plan the bind window ran (all four ranks hash the same, because the hash covers names and
//!     sharding rules, not pointers).
//!
//! What it does NOT prove: that the arena bind succeeded, that a kernel launched, or that any
//! output is right. A pass here means the plan offers the right operand layout; the device bind
//! and any launch are a window result.
//!
//! Evidence and exit. The gate writes tp4-attn-operands-r<R>.json into --evidence-dir with one row
//! per projection kind and the checks above, prints one line per check, and exits 0 on PASS or 2
//! on FAIL (naming the offending tensors). No GPU is touched; the same binary runs on the dev host
//! against a header dump (--tensor-index) and on a node against the checkpoint (--checkpoint).

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use serde_json::{Value, json};

use sparkflash_device::resident::{
    CHAIN_LAYERS, HeaderEntry, MTP_LAYERS, OperandLayout, ResidentPlan,
};
use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::safetensors::{CheckpointIndex, SafeTensorsSource};

/// The stage name in --run-stage.
pub const STAGE: &str = "attn-operands";
/// The evidence schema version this stage writes.
pub const SCHEMA_VERSION: u32 = 1;
/// The stage's command line.
pub const USAGE: &str = "sparkflash device-chain --run-stage attn-operands (--tensor-index FILE | --checkpoint DIR) --tp-rank 0..3 --tp-world-size 4 --evidence-dir DIR [--expect-plan-hash 0x...]";

/// The checkpoint's native dense projection kernel family (wq_a, wkv, wq_b, wo_b): E4M3
/// activation codes with ue8m0 block-32 scales against B12x E4M3 weights, f32 accumulate, bf16
/// output (D39 rule 1; docs/build/native-kernel-arithmetic-audit.md section 2.1).
pub const KERNEL_DENSE: &str = "b12x_mxfp8_dense_sm121";
/// The BF16-activation grouped wo_a kernel: the released rule's native projection for wo_a only
/// (D39 task-29 note).
pub const KERNEL_WO_A: &str = "b12x_bf16a_mxfp8w_wo_a_sm121";
/// The released plan fingerprint at world 4, from the 2026-10-20 bind window's chain log.
pub const RELEASED_PLAN_HASH: u64 = 0x9a59_035f_c407_76c8;

/// The attention layers the released checkpoint stores: 40 chain plus 3 MTP.
pub const ATTENTION_LAYERS: usize = CHAIN_LAYERS + MTP_LAYERS;
/// The five attention projections per layer.
pub const PROJECTIONS_PER_LAYER: usize = 5;
/// The projection weight tensors a released plan must carry.
pub const PROJECTIONS: usize = ATTENTION_LAYERS * PROJECTIONS_PER_LAYER;

/// World-4 tiled counts by attention group, from the bind window's arena evidence.
pub const REPLICATED_TILED: usize = ATTENTION_LAYERS * 2;
/// World-4 tiled counts by attention group: wq_b and wo_a.
pub const COLUMN_TILED: usize = ATTENTION_LAYERS * 2;
/// World-4 tiled counts by attention group: wo_b.
pub const ROW_TILED: usize = ATTENTION_LAYERS;
/// World-4 group member counts (weights plus their scale and norm tensors).
pub const REPLICATED_TENSORS: usize = ATTENTION_LAYERS * 6;
/// World-4 group member counts: wq_b and wo_a, each weight plus its scale.
pub const COLUMN_TENSORS: usize = ATTENTION_LAYERS * 4;
/// World-4 group member counts: wo_b weight plus its scale.
pub const ROW_TENSORS: usize = ATTENTION_LAYERS * 2;

/// World-4 tiled operand bytes (codes plus scale grids) for the replicated group.
pub const WORLD4_REPLICATED_OPERAND_BYTES: u64 = 394_912_000;
/// World-4 tiled operand bytes for the column group.
pub const WORLD4_COLUMN_OPERAND_BYTES: u64 = 812_390_400;
/// World-4 tiled operand bytes for the row group.
pub const WORLD4_ROW_OPERAND_BYTES: u64 = 451_328_000;
/// World-4 resident bytes of the whole replicated group (operands plus BF16 norms).
pub const WORLD4_REPLICATED_GROUP_BYTES: u64 = 395_066_112;
/// World-4 resident bytes of the whole column group.
pub const WORLD4_COLUMN_GROUP_BYTES: u64 = 812_390_400;
/// World-4 resident bytes of the whole row group.
pub const WORLD4_ROW_GROUP_BYTES: u64 = 451_328_000;

/// The gate's parsed arguments.
#[derive(Debug)]
struct GateArgs {
    rank: usize,
    world: usize,
    tensor_index: Option<PathBuf>,
    checkpoint: Option<PathBuf>,
    evidence_dir: PathBuf,
    expect_plan_hash: Option<u64>,
}

impl GateArgs {
    fn parse(args: &[&str]) -> Result<Self, String> {
        let mut rank = 0usize;
        let mut world = 4usize;
        let mut tensor_index: Option<PathBuf> = None;
        let mut checkpoint: Option<PathBuf> = None;
        let mut evidence_dir: Option<PathBuf> = None;
        let mut expect_plan_hash: Option<u64> = None;
        let mut index = 0usize;
        while index < args.len() {
            let flag = args[index];
            let take = |at: usize, flag_name: &str| -> Result<String, String> {
                args.get(at + 1)
                    .map(|value| (*value).to_string())
                    .ok_or_else(|| format!("{flag_name} needs a value"))
            };
            match flag {
                "--tp-rank" => rank = parse_usize("--tp-rank", &take(index, "--tp-rank")?)?,
                "--tp-world-size" => {
                    world = parse_usize("--tp-world-size", &take(index, "--tp-world-size")?)?;
                }
                "--tensor-index" => {
                    tensor_index = Some(PathBuf::from(take(index, "--tensor-index")?));
                }
                "--checkpoint" => checkpoint = Some(PathBuf::from(take(index, "--checkpoint")?)),
                "--evidence-dir" => {
                    evidence_dir = Some(PathBuf::from(take(index, "--evidence-dir")?));
                }
                "--expect-plan-hash" => {
                    let text = take(index, "--expect-plan-hash")?;
                    let trimmed = text.trim_start_matches("0x");
                    expect_plan_hash = Some(
                        u64::from_str_radix(trimmed, 16)
                            .map_err(|error| format!("--expect-plan-hash {text}: {error}"))?,
                    );
                }
                other => return Err(format!("{other}: not an attn-operands flag")),
            }
            index += 2;
        }
        if world == 0 {
            return Err("--tp-world-size needs at least one rank".to_string());
        }
        if rank >= world {
            return Err(format!("--tp-rank {rank} is outside a world of {world}"));
        }
        if tensor_index.is_some() && checkpoint.is_some() {
            return Err("--tensor-index and --checkpoint are alternatives, not both".to_string());
        }
        if tensor_index.is_none() && checkpoint.is_none() {
            return Err("one of --tensor-index or --checkpoint is required".to_string());
        }
        let evidence_dir = evidence_dir.ok_or_else(|| "--evidence-dir is required".to_string())?;
        Ok(Self {
            rank,
            world,
            tensor_index,
            checkpoint,
            evidence_dir,
            expect_plan_hash,
        })
    }
}

fn parse_usize(flag: &str, text: &str) -> Result<usize, String> {
    text.parse::<usize>()
        .map_err(|error| format!("{flag} {text}: {error}"))
}

/// One projection kind's aggregate row, the shape the evidence and the printed lines both use.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ProjectionRow {
    leaf: &'static str,
    kernel: &'static str,
    cols: usize,
    inner: usize,
    weight_bytes: u64,
    scale_bytes: u64,
}

/// The gate's evaluated result: evidence, lines, and whether every check passed.
struct GateOutcome {
    evidence: Value,
    lines: Vec<String>,
    passed: bool,
}

/// The console entry point: parse, plan, evaluate, write evidence, print, and exit.
pub fn run(args: &[&str]) -> ExitCode {
    match run_lines(args) {
        Ok((outcome, evidence_path)) => {
            for line in &outcome.lines {
                println!("{line}");
            }
            println!("tp4-attn-operands evidence: {}", evidence_path.display());
            if outcome.passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(2)
            }
        }
        Err(error) => {
            eprintln!("tp4-attn-operands: {error}");
            ExitCode::from(2)
        }
    }
}

/// Runs the gate and returns its outcome and the evidence path.
fn run_lines(args: &[&str]) -> Result<(GateOutcome, PathBuf), String> {
    let args = GateArgs::parse(args)?;
    let (entries, source) = load_entries(&args)?;
    let plan = ResidentPlan::plan(args.rank, args.world, entries)
        .map_err(|error| format!("plan rank {}: {error}", args.rank))?;
    let outcome = evaluate(&plan, args.rank, args.world, &source, args.expect_plan_hash)?;
    let path = args
        .evidence_dir
        .join(format!("tp4-attn-operands-r{}.json", args.rank));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let bytes = serde_json::to_vec_pretty(&outcome.evidence)
        .map_err(|error| format!("serialise evidence: {error}"))?;
    fs::write(&path, bytes).map_err(|error| format!("write {}: {error}", path.display()))?;
    Ok((outcome, path))
}

/// Loads the header entries from a checkpoint directory or a header dump.
fn load_entries(args: &GateArgs) -> Result<(Vec<HeaderEntry>, String), String> {
    if let Some(path) = &args.tensor_index {
        let entries = crate::tp4_resident::entries_from_tensor_index(path)?;
        let seen = entries.len();
        return Ok((
            entries,
            format!("tensor-index {} ({seen} tensors)", path.display()),
        ));
    }
    let dir = args
        .checkpoint
        .as_ref()
        .ok_or_else(|| "one of --tensor-index or --checkpoint is required".to_string())?;
    let index_path = dir.join("model.safetensors.index.json");
    let index = CheckpointIndex::load(&index_path)
        .map_err(|error| format!("read the checkpoint index: {error}"))?;
    let source = SafeTensorsSource::open(dir.clone(), &index_path)
        .map_err(|error| format!("open the checkpoint {}: {error}", dir.display()))?;
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
    Ok((
        entries,
        format!("checkpoint {} ({seen} tensors)", dir.display()),
    ))
}

/// The projection leaf of a planned attention weight name.
fn projection_leaf(name: &str) -> Option<&'static str> {
    let leaf = name.rsplit_once(".attn.")?.1.split('.').next()?;
    match leaf {
        "wq_a" => Some("wq_a"),
        "wq_b" => Some("wq_b"),
        "wkv" => Some("wkv"),
        "wo_a" => Some("wo_a"),
        "wo_b" => Some("wo_b"),
        _ => None,
    }
}

/// The projection leaf of an attention weight shard, chain or MTP, or None for any other tensor.
fn attention_weight_leaf(name: &str) -> Option<&'static str> {
    let leaf = name.rsplit_once(".attn.")?.1.strip_suffix(".weight")?;
    match leaf {
        "wq_a" => Some("wq_a"),
        "wq_b" => Some("wq_b"),
        "wkv" => Some("wkv"),
        "wo_a" => Some("wo_a"),
        "wo_b" => Some("wo_b"),
        _ => None,
    }
}

/// The kernel family a projection's B12x operand is launched on: the native dense family for
/// wq_a/wkv/wq_b/wo_b, and the bf16-activation wo_a variant for wo_a (its native rule).
fn kernel_family(leaf: &str) -> &'static str {
    match leaf {
        "wo_a" => KERNEL_WO_A,
        _ => KERNEL_DENSE,
    }
}

/// Evaluates the plan and returns the evidence, lines and verdict.
fn evaluate(
    plan: &ResidentPlan,
    rank: usize,
    world: usize,
    source: &str,
    expect_plan_hash: Option<u64>,
) -> Result<GateOutcome, String> {
    let config = ModelConfig::v41_flash().map_err(|error| format!("model config: {error}"))?;
    let geometry = crate::resident_weights::rank_attention_geometry(&config, rank, world)?;
    let planned: BTreeMap<&str, &sparkflash_device::resident::ResidentShard> = plan
        .shards()
        .iter()
        .map(|shard| (shard.name.as_str(), shard))
        .collect();

    // The chain geometry gives this rank's expected tiling per projection leaf; every layer of a
    // kind shares the same shape, so the map also covers the MTP layers the plan carries besides
    // the chain's 40.
    let mut expected_leaf: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    for (name, shape) in &geometry {
        let leaf = projection_leaf(name).ok_or_else(|| format!("{name}: not a projection name"))?;
        if let Some(previous) = expected_leaf.insert(leaf, *shape)
            && previous != *shape
        {
            return Err(format!(
                "{leaf}: the chain geometry disagrees between layers: {previous:?} vs {shape:?}"
            ));
        }
    }
    // Every chain projection must be planned exactly as the rank geometry says, and tiled.
    let mut chain_b12x = 0usize;
    for (name, (cols, inner)) in &geometry {
        match plan.operand_layout(name) {
            Some(OperandLayout::B12x {
                cols: planned_cols,
                inner: planned_inner,
                ..
            }) if (planned_cols, planned_inner) == (*cols, *inner) => chain_b12x += 1,
            Some(OperandLayout::B12x { .. }) => {
                return Err(format!(
                    "{name}: the plan tiles a different geometry than the rank's {cols}x{inner}"
                ));
            }
            // A stored chain projection is not an error here: the plan scan below names it in the
            // blocked list and the gate fails on it, so the refusal is evidence rather than a
            // short-circuited message.
            Some(OperandLayout::Stored { .. }) => {}
            None => return Err(format!("{name}: planned geometry but not in the rank plan")),
        }
    }

    // Every attention projection weight in the plan - chain and MTP alike - plus its scale grid.
    let mut rows: BTreeMap<&'static str, ProjectionRow> = BTreeMap::new();
    let mut blocked: Vec<Value> = Vec::new();
    let mut group_tiled: BTreeMap<&str, usize> = BTreeMap::new();
    let mut group_operand_bytes: BTreeMap<&str, u64> = BTreeMap::new();
    let mut attention_weights = 0usize;
    for shard in plan.shards() {
        let Some(leaf) = attention_weight_leaf(&shard.name) else {
            continue;
        };
        attention_weights += 1;
        let Some(expected) = expected_leaf.get(leaf) else {
            return Err(format!(
                "{}: the {leaf} projection is not in the chain attention geometry",
                shard.name
            ));
        };
        let group = shard.group.as_str();
        match shard.operand_layout() {
            OperandLayout::B12x {
                cols,
                inner,
                weight_bytes,
                scale_bytes,
            } => {
                if (cols, inner) != *expected {
                    return Err(format!(
                        "{}: plan tiles {cols}x{inner} but the rank geometry asks {}x{}",
                        shard.name, expected.0, expected.1
                    ));
                }
                let scale_name = format!("{}.scale", shard.name.trim_end_matches(".weight"));
                match planned
                    .get(scale_name.as_str())
                    .map(|scale| scale.operand_layout())
                {
                    Some(OperandLayout::Stored { dtype, bytes })
                        if dtype == "F8_E8M0" && bytes == scale_bytes => {}
                    Some(OperandLayout::Stored { dtype, bytes }) => {
                        return Err(format!(
                            "{scale_name}: stored {dtype} {bytes} B but the B12x scale grid is F8_E8M0 {scale_bytes} B"
                        ));
                    }
                    Some(OperandLayout::B12x { .. }) => {
                        return Err(format!(
                            "{scale_name}: a scale grid must be stored, not tiled"
                        ));
                    }
                    None => {
                        return Err(format!(
                            "{scale_name}: planned B12x weight without a planned scale grid"
                        ));
                    }
                }
                *group_tiled.entry(group).or_insert(0) += 1;
                *group_operand_bytes.entry(group).or_insert(0) += weight_bytes + scale_bytes;
                rows.entry(leaf).or_insert(ProjectionRow {
                    leaf,
                    kernel: kernel_family(leaf),
                    cols,
                    inner,
                    weight_bytes,
                    scale_bytes,
                });
            }
            OperandLayout::Stored { dtype, bytes } => {
                blocked.push(json!({
                    "name": shard.name,
                    "leaf": leaf,
                    "reason": "a stored operand has no B12x kernel on the four-rank arm; the forward refuses it by name instead of launching",
                    "dtype": dtype,
                    "bytes": bytes,
                }));
            }
        }
    }

    // Group totals from the plan itself, so the operand bytes can be checked against the whole
    // group the bind window recorded (operand bytes plus the BF16 norms sharing the group).
    let mut group_in_plan: BTreeMap<&str, (usize, u64)> = BTreeMap::new();
    for group in plan.groups() {
        group_in_plan.insert(group.group.as_str(), (group.tensors, group.shard_bytes));
    }

    let total = attention_weights;
    let b12x: usize = group_tiled.values().sum();
    let mut checks: Vec<Value> = Vec::new();
    {
        let mut check = |name: &str, expected: Value, actual: Value, pass: bool| {
            checks.push(json!({
                "check": name,
                "expected": expected,
                "actual": actual,
                "pass": pass,
            }));
        };
        check(
            "chain projections B12x",
            json!(geometry.len()),
            json!(chain_b12x),
            chain_b12x == geometry.len(),
        );
        check(
            "attention projections",
            json!(PROJECTIONS),
            json!(total),
            total == PROJECTIONS,
        );
        check(
            "b12x projections",
            json!(total),
            json!(b12x),
            b12x == total && !rows.is_empty(),
        );
        check(
            "stored projections",
            json!(0),
            json!(blocked.len()),
            blocked.is_empty(),
        );
        // The five layer-0 operands, the ones the first forward layer binds before any loop work.
        let layer0 = ["wq_a", "wkv", "wq_b", "wo_a", "wo_b"];
        let layer0_b12x = layer0
            .iter()
            .filter(|leaf| {
                plan.operand_layout(&format!("layers.0.attn.{leaf}.weight"))
                    .is_some_and(|layout| layout.is_b12x())
            })
            .count();
        check(
            "layer-0 five attention operands B12x",
            json!(layer0.len()),
            json!(layer0_b12x),
            layer0_b12x == layer0.len(),
        );
        // Tiled and member parity holds at any world: 43 released layers tile two Replicated
        // (wq_a, wkv), two Column (wq_b, wo_a) and one Row (wo_b) projection each.
        let parity = [
            ("attention-replicated", REPLICATED_TILED, REPLICATED_TENSORS),
            ("attention-column", COLUMN_TILED, COLUMN_TENSORS),
            ("attention-row", ROW_TILED, ROW_TENSORS),
        ];
        for (group, tiled, tensors) in parity {
            let actual_tiled = group_tiled.get(group).copied().unwrap_or(0);
            check(
                &format!("{group} tiled tensors"),
                json!(tiled),
                json!(actual_tiled),
                actual_tiled == tiled,
            );
            let actual_members = group_in_plan.get(group).copied().unwrap_or((0, 0));
            check(
                &format!("{group} group members"),
                json!([tensors, actual_members.1]),
                json!([actual_members.0, actual_members.1]),
                actual_members.0 == tensors,
            );
        }
        if world == 4 {
            let expected_members = [
                ("attention-replicated", WORLD4_REPLICATED_GROUP_BYTES),
                ("attention-column", WORLD4_COLUMN_GROUP_BYTES),
                ("attention-row", WORLD4_ROW_GROUP_BYTES),
            ];
            for (group, bytes) in expected_members {
                let actual = group_in_plan.get(group).copied().unwrap_or((0, 0)).1;
                check(
                    &format!("{group} group bytes"),
                    json!(bytes),
                    json!(actual),
                    actual == bytes,
                );
            }
            let expected_operand = [
                ("attention-replicated", WORLD4_REPLICATED_OPERAND_BYTES),
                ("attention-column", WORLD4_COLUMN_OPERAND_BYTES),
                ("attention-row", WORLD4_ROW_OPERAND_BYTES),
            ];
            for (group, bytes) in expected_operand {
                let actual = group_operand_bytes.get(group).copied().unwrap_or(0);
                check(
                    &format!("{group} operand bytes"),
                    json!(bytes),
                    json!(actual),
                    actual == bytes,
                );
            }
        }
        if let Some(expected) = expect_plan_hash {
            let actual = plan.plan_hash();
            check(
                "plan hash",
                json!(format!("{expected:#018x}")),
                json!(format!("{actual:#018x}")),
                actual == expected,
            );
        }
    }
    let checks_pass = checks.iter().all(|check| check["pass"] == json!(true));
    let passed = checks_pass && blocked.is_empty() && b12x == total && total == PROJECTIONS;

    let rows: Vec<&ProjectionRow> = rows.values().collect();
    let operand_total: u64 = group_operand_bytes.values().sum();
    let layers = ATTENTION_LAYERS;
    let evidence = json!({
        "stage": "tp4-attn-operands",
        "schema": format!("sparkflash-tp4-attn-operands/{SCHEMA_VERSION}"),
        "performed": true,
        "host_only": true,
        "rank": rank,
        "world": world,
        "source": source,
        "plan": {
            "shards": plan.shards().len(),
            "resident_tensors": plan.resident_tensors(),
            "disk_tensors": plan.disk_tensors(),
            "resident_bytes": plan.resident_bytes(),
            "ranges": plan.ranges(),
            "plan_hash": format!("{:#018x}", plan.plan_hash()),
        },
        "attention": {
            "layers": layers,
            "chain_projection_rows": geometry.len(),
            "projections": total,
            "b12x": b12x,
            "stored": blocked.len(),
            "blocked": &blocked,
            "activation_dtype": "BF16",
            "note": "the chain-native path feeds these projections BF16 activations on the D37 bf16a kernels; the weights are the checkpoint's B12x-tiled F8_E4M3 codes and F8_E8M0 grids, read from the arena with no repack",
        },
        "operands": rows.iter().map(|row| json!({
            "projection": row.leaf,
            "kernel": row.kernel,
            "layers": layers,
            "dtype": "F8_E4M3",
            "scale_dtype": "F8_E8M0",
            "cols": row.cols,
            "inner": row.inner,
            "weight_bytes": row.weight_bytes,
            "scale_bytes": row.scale_bytes,
        })).collect::<Vec<Value>>(),
        "byte_totals": {
            "by_group": group_operand_bytes.iter().map(|(group, bytes)| (group.to_string(), json!(bytes))).collect::<BTreeMap<String, Value>>(),
            "attention_operands": operand_total,
            "repack_bytes": 0,
        },
        "repack": {
            "needed": false,
            "kernel": Value::Null,
            "reason": "the released plan stores wq_a/wkv as F8_E4M3 with E8M0 scales in the B12x-tiled attention groups; there is no BF16-stored projection to convert, so the arena needs no boot repack and per-token weight H2D stays 0",
        },
        "bind_expectation": {
            "scale_pointer_distinct_from_weight": true,
            "per_token_weight_h2d_bytes": 0,
        },
        "checks": checks,
        "verdict": if passed { "PASS" } else { "FAIL" },
        "note": "host-only gate: the rank plan is built from headers, the operand layouts are derived from dtype/group/shard shape, no driver is opened, no kernel is launched and no output is claimed",
    });

    let mut lines = Vec::new();
    lines.push(format!(
        "tp4-attn-operands rank {rank} world {world}: plan {} shards ({} resident / {} disk), {layers} attention layers, plan hash {:#018x}",
        plan.shards().len(),
        plan.resident_tensors(),
        plan.disk_tensors(),
        plan.plan_hash()
    ));
    lines.push(format!(
        "tp4-attn-operands: attention projections {b12x}/{total} B12x-launchable ({} replicated + {} column + {} row), {} stored, {} blocked",
        group_tiled.get("attention-replicated").copied().unwrap_or(0),
        group_tiled.get("attention-column").copied().unwrap_or(0),
        group_tiled.get("attention-row").copied().unwrap_or(0),
        blocked.len(),
        blocked.len(),
    ));
    lines.push(format!(
        "tp4-attn-operands: chain geometry {chain_b12x}/{} B12x; layer-0 five operands B12x",
        geometry.len()
    ));
    for row in &rows {
        lines.push(format!(
            "tp4-attn-operands: {} {layers} x [{}x{}] F8_E4M3 -> B12x {} B codes + {} B F8_E8M0 scales ({}, activation BF16)",
            row.leaf, row.cols, row.inner, row.weight_bytes, row.scale_bytes, row.kernel
        ));
    }
    lines.push(format!(
        "tp4-attn-operands: operand bytes replicated {}, column {}, row {}, total {} B; repack bytes 0",
        group_operand_bytes.get("attention-replicated").copied().unwrap_or(0),
        group_operand_bytes.get("attention-column").copied().unwrap_or(0),
        group_operand_bytes.get("attention-row").copied().unwrap_or(0),
        operand_total,
    ));
    if passed {
        lines.push(format!(
            "tp4-attn-operands gate: PASS ({b12x}/{total} attention projections on {layers} layers are B12x MXFP8; no BF16-stored operand; no boot repack; per-token weight H2D stays 0)"
        ));
    } else {
        let names: Vec<String> = blocked
            .iter()
            .filter_map(|row| row["name"].as_str().map(str::to_string))
            .collect();
        lines.push(format!(
            "tp4-attn-operands gate: FAIL ({} stored/unlaunchable projection(s): {})",
            names.len(),
            if names.is_empty() {
                "a plan check failed".to_string()
            } else {
                names.join(", ")
            }
        ));
    }
    Ok(GateOutcome {
        evidence,
        lines,
        passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, dtype: &str, shape: &[usize]) -> HeaderEntry {
        HeaderEntry::new(name.to_string(), dtype.to_string(), shape.to_vec())
    }

    fn released_attention_entries() -> Vec<HeaderEntry> {
        let mut entries = Vec::new();
        for layer in 0..CHAIN_LAYERS {
            for (name, weight, scale) in [
                ("wq_a", [1280usize, 5120], [40usize, 160]),
                ("wq_b", [32768, 1280], [1024, 40]),
                ("wkv", [512, 5120], [16, 160]),
                ("wo_a", [8192, 4096], [256, 128]),
                ("wo_b", [5120, 8192], [160, 256]),
            ] {
                entries.push(entry(
                    &format!("layers.{layer}.attn.{name}.weight"),
                    "F8_E4M3",
                    &weight,
                ));
                entries.push(entry(
                    &format!("layers.{layer}.attn.{name}.scale"),
                    "F8_E8M0",
                    &scale,
                ));
            }
            entries.push(entry(
                &format!("layers.{layer}.attn.q_norm.weight"),
                "BF16",
                &[1280],
            ));
            entries.push(entry(
                &format!("layers.{layer}.attn.kv_norm.weight"),
                "BF16",
                &[512],
            ));
        }
        for layer in 0..MTP_LAYERS {
            let prefix = format!("mtp.{layer}.attn");
            for (name, weight, scale) in [
                ("wq_a", [1280usize, 5120], [40usize, 160]),
                ("wq_b", [32768, 1280], [1024, 40]),
                ("wkv", [512, 5120], [16, 160]),
                ("wo_a", [8192, 4096], [256, 128]),
                ("wo_b", [5120, 8192], [160, 256]),
            ] {
                entries.push(entry(
                    &format!("{prefix}.{name}.weight"),
                    "F8_E4M3",
                    &weight,
                ));
                entries.push(entry(&format!("{prefix}.{name}.scale"), "F8_E8M0", &scale));
            }
            entries.push(entry(&format!("{prefix}.q_norm.weight"), "BF16", &[1280]));
            entries.push(entry(&format!("{prefix}.kv_norm.weight"), "BF16", &[512]));
        }
        entries
    }

    #[test]
    fn the_released_attention_plan_is_b12x_on_every_projection_and_rank() {
        for rank in 0..4 {
            let plan = ResidentPlan::plan(rank, 4, released_attention_entries()).expect("plan");
            let outcome = evaluate(&plan, rank, 4, "test", None).expect("evaluate");
            assert!(
                outcome.passed,
                "gate must pass on the released shapes: {:?}",
                outcome.lines
            );
            assert_eq!(outcome.evidence["attention"]["b12x"], json!(PROJECTIONS));
            assert_eq!(outcome.evidence["attention"]["stored"], json!(0));
            assert_eq!(outcome.evidence["repack"]["needed"], json!(false));
            assert_eq!(outcome.evidence["byte_totals"]["repack_bytes"], json!(0));
            assert_eq!(
                outcome.evidence["byte_totals"]["attention_operands"],
                json!(
                    WORLD4_REPLICATED_OPERAND_BYTES
                        + WORLD4_COLUMN_OPERAND_BYTES
                        + WORLD4_ROW_OPERAND_BYTES
                )
            );
            let groups = &outcome.evidence["byte_totals"]["by_group"];
            assert_eq!(
                groups["attention-replicated"],
                json!(WORLD4_REPLICATED_OPERAND_BYTES)
            );
            assert_eq!(
                groups["attention-column"],
                json!(WORLD4_COLUMN_OPERAND_BYTES)
            );
            assert_eq!(groups["attention-row"], json!(WORLD4_ROW_OPERAND_BYTES));
            let operands = outcome.evidence["operands"].as_array().expect("operands");
            assert_eq!(operands.len(), PROJECTIONS_PER_LAYER);
            assert!(outcome.lines.iter().any(|line| line.contains("gate: PASS")));
        }
    }

    #[test]
    fn a_bf16_stored_projection_fails_the_gate_by_name() {
        // The retracted premise, made concrete: if the plan carried wq_a as BF16, the gate must
        // fail and name it, because no B12x kernel can read that operand and this arm has no
        // per-layer host repack staging.
        let mut entries = released_attention_entries();
        for tensor in &mut entries {
            if tensor.name == "layers.0.attn.wq_a.weight" {
                tensor.dtype = "BF16".to_string();
            }
        }
        let plan = ResidentPlan::plan(0, 4, entries).expect("plan");
        let outcome = evaluate(&plan, 0, 4, "test", None).expect("evaluate");
        assert!(!outcome.passed, "a BF16 wq_a must fail the gate");
        let blocked = outcome.evidence["attention"]["blocked"]
            .as_array()
            .expect("blocked");
        assert_eq!(blocked.len(), 1);
        assert_eq!(blocked[0]["name"], json!("layers.0.attn.wq_a.weight"));
        assert!(
            outcome
                .lines
                .iter()
                .any(|line| line.contains("gate: FAIL") && line.contains("wq_a")),
            "{:?}",
            outcome.lines
        );
    }

    #[test]
    fn the_plan_hash_flag_is_asserted_not_ignored() {
        let plan = ResidentPlan::plan(0, 4, released_attention_entries()).expect("plan");
        // The flag is compared, so the plan's own fingerprint passes and anything else fails.
        let hash = plan.plan_hash();
        let outcome = evaluate(&plan, 0, 4, "test", Some(hash)).expect("evaluate");
        assert!(outcome.passed);
        let wrong = evaluate(&plan, 0, 4, "test", Some(hash ^ 1)).expect("evaluate");
        assert!(!wrong.passed, "a wrong expected hash must not pass");
        assert!(wrong.lines.iter().any(|line| line.contains("gate: FAIL")));
    }

    #[test]
    fn arguments_are_the_two_header_sources_and_an_evidence_dir() {
        let parsed = GateArgs::parse(&[
            "--tensor-index",
            "/tmp/h.json",
            "--tp-rank",
            "2",
            "--tp-world-size",
            "4",
            "--evidence-dir",
            "/tmp/ev",
            "--expect-plan-hash",
            "0x9a59035fc40776c8",
        ])
        .expect("parse");
        assert_eq!(parsed.rank, 2);
        assert_eq!(parsed.world, 4);
        assert_eq!(parsed.expect_plan_hash, Some(RELEASED_PLAN_HASH));
        assert!(GateArgs::parse(&["--tp-rank", "0", "--evidence-dir", "/tmp/ev"]).is_err());
        assert!(
            GateArgs::parse(&[
                "--tensor-index",
                "/tmp/h.json",
                "--checkpoint",
                "/tmp/ckpt",
                "--evidence-dir",
                "/tmp/ev"
            ])
            .is_err()
        );
        assert!(
            GateArgs::parse(&[
                "--tensor-index",
                "/tmp/h.json",
                "--tp-rank",
                "4",
                "--tp-world-size",
                "4",
                "--evidence-dir",
                "/tmp/ev"
            ])
            .is_err()
        );
    }

    #[test]
    fn a_non_world4_plan_still_names_its_operands_without_the_world4_totals() {
        // The gate's contract is the released world-4 plan; another world must still enumerate the
        // projections and their layouts rather than silently skip the byte invariants.
        let plan = ResidentPlan::plan(0, 2, released_attention_entries()).expect("plan");
        let outcome = evaluate(&plan, 0, 2, "test", None).expect("evaluate");
        assert!(outcome.passed, "{:?}", outcome.lines);
        assert_eq!(outcome.evidence["attention"]["b12x"], json!(PROJECTIONS));
    }
}

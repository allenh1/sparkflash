//! m17a: the routed MoE branch at real layer-0 geometry - router, gate/up, SwiGLU+quantise,
//! down/combine - one harness, four stages, four pinned cubins, released checkpoint bytes only.
//!
//! THIS EXAMPLE IS THE EVIDENCE THE FOUR RELEASED ROUTED KERNELS COMPOSE AT THE RELEASED GEOMETRY.
//! It is not the MoE block in the served forward pass and it does not touch the engine: the host
//! seams the engine still uses are named below, and closing them is separate work.
//!
//! Geometry, read from the released config.json: hidden 5120, moe_intermediate_size 2304, 384
//! routed experts, top-6, n_shared_experts 1 (the shared expert is m17b's claim, not this
//! harness's), swiglu_limit 10.0, routed_scaling_factor 1.5, no first_k_dense_replace key (so
//! layer 0 is a real MoE layer).
//!
//! The stages, each one class-declared BEFORE the numbers it is judged by:
//!
//! 1. ROUTER (moe_router_sm121, 37,384 B / 5992cef9...): released layers.0.ffn.gate.weight (BF16
//!    [384, 5120]) and layers.0.ffn.gate.bias (F32 [384], the noaux_tc bias; bias_vl is not used).
//!    The activation rows are the released embed.weight rows 0..tokens - a declared stand-in for
//!    the post-attention hidden, exactly as m9_router.rs and m10_gate_up.rs declare theirs. Class:
//!    ids EXACT against the CPU twin (router_ref, which the crate's own tests prove bit-identical
//!    to the engine host path on this configuration), scores <= 1e-6 abs and <= 8 ULP, weights
//!    <= 1e-6 abs and <= 16 ULP (the device uses CUDA expf/log1pf where the host uses libm), two
//!    launches byte-identical. The device ids are the selection stages 2-4 consume, so the routed
//!    branch is routing-driven, not a forced id cycle.
//!
//! 2. GATE/UP (routed_gate_up_sm121, 9,752 B / 115200d2...): the w1 and w3 projections of the
//!    experts the router selected, all four 576-row rank slices of the 2304-wide intermediate (8
//!    launches: 4 ranks x 2 projections). Weights are the released experts.<e>.w1/w3.weight (I8
//!    packed FP4) and .scale (F8_E8M0) byte slices; the activation is the D8-quantised embed
//!    stand-in. Class: device output BIT-IDENTICAL (ULP 0) to routed_gate_up_ref on every
//!    tokens*6*640 bf16 value of both projections, columns 576..639 exactly zero, two launches
//!    byte-identical. The eight launches are independent of each other (same activation bytes,
//!    different weight slices; disjoint outputs), so a later chain may stream them; only the
//!    combine ranks below are serial by construction.
//!
//! 3. SWIGLU+QUANT (swiglu_mxfp8_quant_sm121, 21,776 B / 4dbef63b...): the device gate/up bf16
//!    outputs read back, cols 640 per rank slice, unweighted (the kernel has no routing-weight
//!    operand - that is F6, recorded in docs/DECISIONS.md). Class, the stage-8 B1 class: E4M3
//!    codes and ue8m0 scales BIT-EXACT against swiglu_ref::quantise_row over the same bf16 values;
//!    any difference is a finding, never a tolerance. Two launches byte-identical.
//!
//! 4. DOWN/COMBINE (b12x_mxfp4_moe_sm121, 10,048 B / 36a15a4f...): four serial rank launches over
//!    the 576-wide w2 slices of the same selected experts, chained because each rank's bf16 output
//!    is the next rank's residual operand (the kernel writes hidden + this rank's partial sum, so
//!    the chained rounding is part of the arithmetic). Class: device bf16 BIT-EXACT against the
//!    chained mxfp4_ref::moe_combine over the same bytes; two full chains byte-identical.
//!
//! THE F6 SEAM, STATED BEFORE ANY NUMBER. The released arithmetic multiplies the SwiGLU output by
//! the routing weight BEFORE the w2 activation quantiser reads it (model.py:841-851), and no
//! pinned routed kernel takes a weight operand. So stage 3 measures the device quantiser on the
//! UNWEIGHTED gate/up product, and the operand stage 4 consumes is built on the host from the SAME
//! device gate/up bf16 values: host SwiGLU over those bytes, then
//! mxfp4_ref::routed_activation_operand(activation, routing_weight) - the exact reference rule,
//! bf16 cast included. The values that cross the seam are (a) the device gate/up bf16 outputs,
//! [tokens*6][640] per rank and projection, 2 bytes each; (b) the routing weight, one f32 per
//! selected expert per token; (c) the quantised activation operand the combine reads,
//! [tokens*6][640] E4M3 + [tokens*6][20] ue8m0 per rank. Stage 3's device codes are NOT the
//! combine's operand: weight-multiplying already-quantised codes would quantise twice and is not
//! the model's order. The combine is therefore WEIGHTED, but its activation operand is
//! host-quantised by the same rule the reference uses. The NOT-claimed list at the end excludes a
//! fully-device weighted routed block and any per-token/per-layer claim.
//!
//! Usage (aarch64 node, checkpoint present):
//!
//! ```text
//! m17_moe_a --checkpoint <snapshot dir> --cubin-dir <dir> --tokens 8  [--layer 0] [--repeat 0]
//! m17_moe_a --checkpoint <snapshot dir> --cubin-dir <dir> --tokens 130 [--layer 0] [--repeat 0]
//! ```
//!
//! The cubin dir must hold the four pinned cubins, each refused by size+sha256 before allocation:
//! moe_router_sm121.cubin, routed_gate_up_sm121.cubin, swiglu_mxfp8_quant_sm121.cubin,
//! b12x_mxfp4_moe_sm121.cubin (build lines: nvcc -std=c++17 -arch=sm_121a -cubin -I kernels
//! kernels/<name>.cu -o <dir>/<name>.cubin, CUDA 13.0.3 / V13.0.88).

#![forbid(unsafe_code)]

use std::path::PathBuf;

use sha2::{Digest, Sha256};

#[path = "../tests/support/b2_harness.rs"]
mod b2_harness;

use self::b2_harness::{Snapshot, slice_and_pad_rows};
use sparkflash_kernels::cuda_driver::{Cuda, KernelId};
use sparkflash_kernels::m4_quant::{act_quant_fp8_row, bf16_value};
use sparkflash_kernels::mxfp4_moe_launch::{self, Mxfp4MoeParams};
use sparkflash_kernels::mxfp4_ref;
use sparkflash_kernels::routed_gate_up_abi as gabi;
use sparkflash_kernels::routed_gate_up_launch::{self, GateUpLaunchParams};
use sparkflash_kernels::routed_gate_up_ref as twin;
use sparkflash_kernels::router_abi as rabi;
use sparkflash_kernels::router_launch::{self, RouterLaunchParams};
use sparkflash_kernels::router_ref::{self, RouterParams};
use sparkflash_kernels::swiglu_ref::{self, SWIGLU_LIMIT};

/// Routed experts in the released configuration (config.json text_config.n_routed_experts).
const EXPERTS: usize = 384;
/// Hidden width (config.json text_config.hidden_size).
const HIDDEN: usize = 5120;
/// Activated experts per token (config.json text_config.num_experts_per_tok).
const TOP_K: usize = 6;
/// Stored intermediate width of one routed expert (moe_intermediate_size).
const INTER: usize = 2304;
/// The four rank slices of the intermediate (D30: 576 per rank padded to 640).
const RANKS: usize = 4;
/// One rank's real intermediate width.
const WIDTH: usize = 576;
/// One rank's padded width, the routed kernels' contract.
const WIDTH_PADDED: usize = 640;
/// Packed FP4 bytes per stored expert row: 5120 / 2.
const PACKED_ROW: usize = HIDDEN / 2;
/// ue8m0 scale bytes per stored expert row: 5120 / 32.
const SCALE_ROW: usize = HIDDEN / 32;
/// Packed FP4 bytes per stored w2 row: 2304 / 2.
const IN_PACKED: usize = INTER / 2;
/// ue8m0 scale bytes per stored w2 row: 2304 / 32.
const IN_SCALES: usize = INTER / 32;
/// Packed FP4 bytes per rank of one w2 output row: 640 / 2.
const MOE_PACKED_ROW: usize = WIDTH_PADDED / 2;
/// ue8m0 scale bytes per rank of one w2 output row: 640 / 32.
const MOE_SCALE_ROW: usize = WIDTH_PADDED / 32;
/// The activation quantiser's 32-value block.
const QUANT_BLOCK: usize = 32;

/// The swiglu quantiser cubin pin (baseline/kernel_baseline.txt; the crate has no pin module for
/// it because the kernel is reached through its launch module, not through a typed ABI).
const SWIGLU_CUBIN_BYTES: usize = 21_776;
const SWIGLU_CUBIN_SHA256: &str =
    "4dbef63bb07808f92fb6e61bdcb9aec53cb6afd8f28d6744aea3185759c1862e";
/// The MXFP4 down/combine cubin pin (same source; mxfp4_moe_launch states the device chain's
/// catalog owns the bytes, so this harness pins them here).
const MXFP4_MOE_CUBIN_BYTES: usize = 10_048;
const MXFP4_MOE_CUBIN_SHA256: &str =
    "36a15a4f3ae12b4e7d452c1201b267d3102afa43f168bd322d6930ff3f543057";

/// The pass/fail verdict of one named check, printed with its detail.
struct Verdict {
    name: String,
    ok: bool,
    detail: String,
}

impl Verdict {
    fn new(name: &str, ok: bool, detail: String) -> Self {
        Self {
            name: name.to_owned(),
            ok,
            detail,
        }
    }

    fn push(verdicts: &mut Vec<Self>, name: &str, ok: bool, detail: String) {
        println!(
            "  check {name}: {} ({detail})",
            if ok { "ok" } else { "FAIL" }
        );
        verdicts.push(Self::new(name, ok, detail));
    }
}

struct Args {
    checkpoint: PathBuf,
    cubin_dir: PathBuf,
    tokens: usize,
    layer: usize,
    repeat: usize,
    floor_gib: u64,
}

impl Args {
    fn parse() -> Result<Self, String> {
        Self::parse_from(std::env::args().skip(1).collect())
    }

    fn parse_from(values: Vec<String>) -> Result<Self, String> {
        let mut args = Self {
            checkpoint: PathBuf::new(),
            cubin_dir: PathBuf::from("out"),
            tokens: 8,
            layer: 0,
            repeat: 0,
            floor_gib: 32,
        };
        let mut values = values;
        while let Some(flag) = values.first().cloned() {
            values.remove(0);
            let mut next = || {
                values
                    .first()
                    .cloned()
                    .inspect(|_| {
                        values.remove(0);
                    })
                    .ok_or_else(|| format!("{flag} needs a value"))
            };
            match flag.as_str() {
                "--checkpoint" => args.checkpoint = PathBuf::from(next()?),
                "--cubin-dir" => args.cubin_dir = PathBuf::from(next()?),
                "--tokens" => {
                    args.tokens = next()?
                        .parse()
                        .map_err(|error| format!("--tokens: {error}"))?;
                }
                "--layer" => {
                    args.layer = next()?
                        .parse()
                        .map_err(|error| format!("--layer: {error}"))?;
                }
                "--repeat" => {
                    args.repeat = next()?
                        .parse()
                        .map_err(|error| format!("--repeat: {error}"))?;
                }
                "--floor-gib" => {
                    args.floor_gib = next()?
                        .parse()
                        .map_err(|error| format!("--floor-gib: {error}"))?;
                }
                other => return Err(format!("unknown argument {other}")),
            }
        }
        if args.checkpoint.as_os_str().is_empty() {
            return Err("--checkpoint is required".to_owned());
        }
        if args.tokens == 0 {
            return Err("--tokens must be at least 1".to_owned());
        }
        Ok(args)
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("m17_moe_a: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    println!(
        "=== m17a: routed MoE branch at real layer-0 geometry (router, gate/up, SwiGLU+quantise, down/combine) ==="
    );
    println!(
        "geometry: hidden {HIDDEN}, intermediate {INTER} = {RANKS} rank slices of {WIDTH} (padded {WIDTH_PADDED}), routed experts {EXPERTS}, top-k {TOP_K}; tokens {}",
        args.tokens
    );
    println!(
        "devices: four pinned cubins, each verified by size+sha256 before any allocation; each stage's comparison class is printed before the numbers it judges"
    );
    println!(
        "what this is not: not the served forward pass, not a chain/token/per-layer claim, not a timing claim; the shared expert and the shared-add epilogue are m17b. The activation rows and the residual hidden are released embed.weight rows, a declared stand-in for layer 0's post-attention hidden."
    );
    check_floor(args.floor_gib, "m17a start")?;

    let mut verdicts: Vec<Verdict> = Vec::new();
    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());
    let prefix = format!("layers.{}", args.layer);

    // ---------------------------------------------------------------------------------------
    // Stage 1: router.
    // ---------------------------------------------------------------------------------------
    println!("--- stage 1: moe_router_sm121 (released gate weight + noaux_tc bias) ---");
    println!(
        "comparison class (declared before any number): ids EXACT vs router_ref on all {} x {TOP_K} selections; scores max abs delta <= 1e-6 and max ULP <= 8; weights max abs delta <= 1e-6 and max ULP <= 16; two launches byte-identical. The device's expf/log1pf are not libm's exp/ln_1p, which is the only known degree of freedom; ids are pinned because a tie-crossing selection would be a finding, not a tolerance (the tie rule is the kernel's strict-greater scan, lower expert id wins).",
        args.tokens
    );
    let gate_bytes = snap.read_raw(
        &format!("{prefix}.ffn.gate.weight"),
        "BF16",
        &[EXPERTS, HIDDEN],
    )?;
    let bias_bytes = snap.read_raw(&format!("{prefix}.ffn.gate.bias"), "F32", &[EXPERTS])?;
    println!(
        "released operands: gate.weight BF16 {} B, gate.bias F32 {} B (bias_vl not read)",
        gate_bytes.len(),
        bias_bytes.len()
    );
    let x_bytes = read_embed_rows(&mut snap, args.tokens)?;
    let x_f32 = bf16_bytes_to_f32(&x_bytes);
    let gate_f32 = bf16_bytes_to_f32(&gate_bytes);
    let bias_f32 = f32_bytes_vec(&bias_bytes);
    let mut twin_ids: Vec<u32> = Vec::with_capacity(args.tokens * TOP_K);
    let mut twin_weights: Vec<f32> = Vec::with_capacity(args.tokens * TOP_K);
    let mut twin_scores: Vec<f32> = Vec::with_capacity(args.tokens * EXPERTS);
    for token in 0..args.tokens {
        let x = &x_f32[token * HIDDEN..(token + 1) * HIDDEN];
        let out = router_ref::route_f32(x, &gate_f32, &bias_f32, RouterParams::RELEASED)?;
        twin_ids.extend_from_slice(&out.ids);
        twin_weights.extend_from_slice(&out.weights);
        twin_scores.extend_from_slice(&out.scores);
    }
    let router_cubin = pinned_cubin(
        &args.cubin_dir,
        "moe_router_sm121",
        rabi::CUBIN_BYTES,
        rabi::CUBIN_SHA256,
    )?;
    let cuda = Cuda::load()?;
    let mut allocated = 0usize;
    let x_d = cuda.alloc(x_bytes.len(), &mut allocated)?;
    let gate_d = cuda.alloc(gate_bytes.len(), &mut allocated)?;
    let bias_d = cuda.alloc(bias_bytes.len(), &mut allocated)?;
    let ids_d = cuda.alloc(args.tokens * TOP_K * 4, &mut allocated)?;
    let weights_d = cuda.alloc(args.tokens * TOP_K * 4, &mut allocated)?;
    let scores_d = cuda.alloc(args.tokens * EXPERTS * 4, &mut allocated)?;
    println!("stage 1 device bytes admitted: {allocated}");
    cuda.copy_htod(&x_d, &x_bytes)?;
    cuda.copy_htod(&gate_d, &gate_bytes)?;
    cuda.copy_htod(&bias_d, &bias_bytes)?;
    let (router_module, router_kernel) = router_launch::load(&cuda, &router_cubin)?;
    let router_params = RouterLaunchParams {
        x: x_d,
        gate: gate_d,
        bias: bias_d,
        ids: ids_d,
        weights: weights_d,
        scores: scores_d,
        tokens: args.tokens as u32,
        trace_ring: 0,
        layer_id: args.layer as u32,
        step_seq: 0,
    };
    check_floor(args.floor_gib, "m17a stage 1 pre-launch")?;
    router_launch::launch(&cuda, router_kernel, &router_params)?;
    let (device_ids, device_weights, device_scores) =
        read_router_outputs(&cuda, &ids_d, &weights_d, &scores_d, args.tokens)?;
    let first_router_bytes = router_raw_bytes(&device_ids, &device_weights, &device_scores);
    router_launch::launch(&cuda, router_kernel, &router_params)?;
    let (ids_2, weights_2, scores_2) =
        read_router_outputs(&cuda, &ids_d, &weights_d, &scores_d, args.tokens)?;
    let second_router_bytes = router_raw_bytes(&ids_2, &weights_2, &scores_2);
    check_floor(args.floor_gib, "m17a stage 1 post-launch")?;
    cuda.free(
        router_module,
        &[x_d, gate_d, bias_d, ids_d, weights_d, scores_d],
    );

    let ids_exact = device_ids == twin_ids;
    let id_mismatches = device_ids
        .iter()
        .zip(&twin_ids)
        .filter(|(a, b)| a != b)
        .count();
    Verdict::push(
        &mut verdicts,
        "stage 1: device ids exact vs twin",
        ids_exact,
        format!(
            "{id_mismatches} mismatched selections of {}",
            device_ids.len()
        ),
    );
    let (scores_delta, scores_at) = max_abs_delta(&device_scores, &twin_scores);
    let scores_ulp = max_ulp(&device_scores, &twin_scores);
    Verdict::push(
        &mut verdicts,
        "stage 1: scores within 1e-6 / 8 ULP",
        scores_delta <= 1e-6 && scores_ulp <= 8,
        format!("max abs delta {scores_delta:e} at {scores_at}, max ULP {scores_ulp}"),
    );
    let (weights_delta, weights_at) = max_abs_delta(&device_weights, &twin_weights);
    let weights_ulp = max_ulp(&device_weights, &twin_weights);
    Verdict::push(
        &mut verdicts,
        "stage 1: weights within 1e-6 / 16 ULP",
        weights_delta <= 1e-6 && weights_ulp <= 16,
        format!("max abs delta {weights_delta:e} at {weights_at}, max ULP {weights_ulp}"),
    );
    Verdict::push(
        &mut verdicts,
        "stage 1: two launches byte-identical",
        first_router_bytes == second_router_bytes,
        format!("{} raw output bytes compared", first_router_bytes.len()),
    );
    println!(
        "stage 1 selection: {} routed rows; {} distinct experts over the batch; ids/weights/scores read back",
        args.tokens * TOP_K,
        unique_experts(&device_ids).len()
    );

    // ---------------------------------------------------------------------------------------
    // Stage 2: gate/up for the router-selected experts.
    // ---------------------------------------------------------------------------------------
    println!(
        "--- stage 2: routed_gate_up_sm121, 4 rank slices x 2 projections (independent launches) ---"
    );
    println!(
        "comparison class (declared before any number): device w1 and w3 outputs BIT-IDENTICAL (ULP 0) to routed_gate_up_ref on every tokens*{TOP_K}*{WIDTH_PADDED} bf16 value of the projection, columns {WIDTH}..{} exactly zero, two launches byte-identical. Twin is bit-identical to the shared b2_harness arithmetic by construction of the crate's tests; the 8 launches are independent (same activation, disjoint weights and outputs), so a later chain can stream them.",
        WIDTH_PADDED - 1
    );
    let local_of = local_expert_table(&device_ids);
    let local_ids: Vec<u32> = device_ids
        .iter()
        .map(|expert| local_of[*expert as usize] as u32)
        .collect();
    let local_experts = local_of
        .iter()
        .filter(|index| **index != usize::MAX)
        .count();
    println!(
        "routed expert table: {local_experts} distinct experts selected by the device router, staged as local ids 0..{local_experts}; every later stage indexes this table, and the mapping is printed below"
    );
    let mut table_rows: Vec<(usize, usize)> = local_of
        .iter()
        .enumerate()
        .filter(|(_, index)| **index != usize::MAX)
        .map(|(expert, index)| (*index, expert))
        .collect();
    table_rows.sort_unstable();
    for (local, expert) in table_rows.iter().take(16) {
        println!("  local {local} = expert {expert}");
    }
    if table_rows.len() > 16 {
        println!("  ... {} more entries", table_rows.len() - 16);
    }

    // The activation stand-in: released embed rows, D8-quantised (the mega_mhc ffn-block input).
    let mut act_fp8 = Vec::with_capacity(args.tokens * HIDDEN);
    let mut act_scales = Vec::with_capacity(args.tokens * SCALE_ROW);
    for token in 0..args.tokens {
        let row = &x_f32[token * HIDDEN..(token + 1) * HIDDEN];
        let (codes, scales) = act_quant_fp8_row(row);
        act_fp8.extend_from_slice(&codes);
        act_scales.extend_from_slice(&scales);
    }
    println!(
        "activation stand-in: {} embed rows x {HIDDEN} bf16 values, D8-quantised to {} E4M3 bytes + {} ue8m0 bytes",
        args.tokens,
        act_fp8.len(),
        act_scales.len()
    );

    // Read the selected experts' released tensors once; the rank slices are cut from these bytes.
    let expert_table = read_expert_tensors(&mut snap, &prefix, &table_rows)?;
    println!(
        "expert bytes: {} experts x (w1/w3/w2 weights + scales), {} B read verbatim from released tensors",
        expert_table.len(),
        expert_table
            .iter()
            .map(|expert| expert.bytes())
            .sum::<usize>()
    );

    let rows = args.tokens * TOP_K;
    let out_values = rows * WIDTH_PADDED;
    let gate_up_cubin = pinned_cubin(
        &args.cubin_dir,
        "routed_gate_up_sm121",
        gabi::CUBIN_BYTES,
        gabi::CUBIN_SHA256,
    )?;
    let (_gate_up_module, gate_up_kernel) = routed_gate_up_launch::load(&cuda, &gate_up_cubin)?;
    let mut gate_bits: Vec<Vec<u16>> = Vec::with_capacity(RANKS);
    let mut up_bits: Vec<Vec<u16>> = Vec::with_capacity(RANKS);
    for rank in 0..RANKS {
        let w_gate = rank_slice(&expert_table, |expert| &expert.w1, rank, WIDTH * PACKED_ROW);
        let s_gate = rank_slice(&expert_table, |expert| &expert.s1, rank, WIDTH * SCALE_ROW);
        let w_up = rank_slice(&expert_table, |expert| &expert.w3, rank, WIDTH * PACKED_ROW);
        let s_up = rank_slice(&expert_table, |expert| &expert.s3, rank, WIDTH * SCALE_ROW);
        let (twin_gate, _) = twin::gate_up(
            &act_fp8,
            &act_scales,
            &w_gate,
            &s_gate,
            &local_ids,
            args.tokens,
            local_experts,
        );
        let (twin_up, _) = twin::gate_up(
            &act_fp8,
            &act_scales,
            &w_up,
            &s_up,
            &local_ids,
            args.tokens,
            local_experts,
        );
        let mut rank_allocated = 0usize;
        let act_d = cuda.alloc(act_fp8.len(), &mut rank_allocated)?;
        let act_s_d = cuda.alloc(act_scales.len(), &mut rank_allocated)?;
        let w_gate_d = cuda.alloc(w_gate.len(), &mut rank_allocated)?;
        let s_gate_d = cuda.alloc(s_gate.len(), &mut rank_allocated)?;
        let w_up_d = cuda.alloc(w_up.len(), &mut rank_allocated)?;
        let s_up_d = cuda.alloc(s_up.len(), &mut rank_allocated)?;
        let ids_d = cuda.alloc(local_ids.len() * 4, &mut rank_allocated)?;
        let gate_out_d = cuda.alloc(out_values * 2, &mut rank_allocated)?;
        let up_out_d = cuda.alloc(out_values * 2, &mut rank_allocated)?;
        println!(
            "  rank {rank}: device bytes admitted {rank_allocated} ({:.1} MiB); w window rows {}..{}",
            rank_allocated as f64 / (1024.0 * 1024.0),
            rank * WIDTH,
            (rank + 1) * WIDTH
        );
        cuda.copy_htod(&act_d, &act_fp8)?;
        cuda.copy_htod(&act_s_d, &act_scales)?;
        cuda.copy_htod(&w_gate_d, &w_gate)?;
        cuda.copy_htod(&s_gate_d, &s_gate)?;
        cuda.copy_htod(&w_up_d, &w_up)?;
        cuda.copy_htod(&s_up_d, &s_up)?;
        cuda.copy_htod(&ids_d, &u32s_to_bytes(&local_ids))?;
        let base = GateUpLaunchParams {
            act_fp8: act_d,
            act_scales: act_s_d,
            w_fp4: w_gate_d,
            w_scales: s_gate_d,
            expert_ids: ids_d,
            out: gate_out_d,
            tokens: args.tokens as u32,
            trace_ring: 0,
            layer_id: args.layer as u32,
            step_seq: rank as u64,
        };
        let up_params = GateUpLaunchParams {
            w_fp4: w_up_d,
            w_scales: s_up_d,
            out: up_out_d,
            ..base
        };
        check_floor(
            args.floor_gib,
            &format!("m17a stage 2 rank {rank} pre-launch"),
        )?;
        routed_gate_up_launch::launch(&cuda, gate_up_kernel, &base)?;
        routed_gate_up_launch::launch(&cuda, gate_up_kernel, &up_params)?;
        let gate_first = read_bf16_bytes(&cuda, &gate_out_d, out_values)?;
        let up_first = read_bf16_bytes(&cuda, &up_out_d, out_values)?;
        routed_gate_up_launch::launch(&cuda, gate_up_kernel, &base)?;
        routed_gate_up_launch::launch(&cuda, gate_up_kernel, &up_params)?;
        let gate_second = read_bf16_bytes(&cuda, &gate_out_d, out_values)?;
        let up_second = read_bf16_bytes(&cuda, &up_out_d, out_values)?;
        check_floor(
            args.floor_gib,
            &format!("m17a stage 2 rank {rank} post-launch"),
        )?;
        cuda.free_buffers(&[
            &act_d,
            &act_s_d,
            &w_gate_d,
            &s_gate_d,
            &w_up_d,
            &s_up_d,
            &ids_d,
            &gate_out_d,
            &up_out_d,
        ]);
        let gate_device = bytes_bf16_bits(&gate_first);
        let up_device = bytes_bf16_bits(&up_first);
        let mut mismatches = 0usize;
        let mut first_at: Option<usize> = None;
        for (index, (device, expected)) in gate_device.iter().zip(&twin_gate).enumerate() {
            if device != expected {
                mismatches += 1;
                first_at.get_or_insert(index);
            }
        }
        for (index, (device, expected)) in up_device.iter().zip(&twin_up).enumerate() {
            if device != expected {
                mismatches += 1;
                first_at.get_or_insert(out_values + index);
            }
        }
        let padding_nonzero =
            count_padding_nonzero(&gate_device) + count_padding_nonzero(&up_device);
        Verdict::push(
            &mut verdicts,
            &format!("stage 2 rank {rank}: both projections bit-identical"),
            mismatches == 0,
            format!(
                "{mismatches} mismatched bf16 values of {}, first at {:?}",
                out_values * 2,
                first_at
            ),
        );
        Verdict::push(
            &mut verdicts,
            &format!("stage 2 rank {rank}: columns {WIDTH}..{WIDTH_PADDED} exactly zero"),
            padding_nonzero == 0,
            format!("{padding_nonzero} nonzero padding values"),
        );
        Verdict::push(
            &mut verdicts,
            &format!("stage 2 rank {rank}: two launches byte-identical"),
            gate_first == gate_second && up_first == up_second,
            format!(
                "{} gate + {} up bytes compared per projection",
                gate_first.len(),
                up_first.len()
            ),
        );
        gate_bits.push(gate_device);
        up_bits.push(up_device);
    }

    // ---------------------------------------------------------------------------------------
    // Stage 3: SwiGLU + MXFP8 activation quantise on the device gate/up output.
    // ---------------------------------------------------------------------------------------
    println!("--- stage 3: swiglu_mxfp8_quant_sm121 on the device gate/up bf16 values ---");
    println!(
        "comparison class (declared before any number): E4M3 codes and ue8m0 scales BIT-EXACT against swiglu_ref::quantise_row over the same {WIDTH_PADDED} bf16 values per routed row (the stage-8 B1 class); a difference is a finding, not a tolerance; two launches byte-identical. The launch is unweighted: the kernel has no routing-weight operand, which is the F6 seam."
    );
    let swiglu_cubin = pinned_cubin(
        &args.cubin_dir,
        "swiglu_mxfp8_quant_sm121",
        SWIGLU_CUBIN_BYTES,
        SWIGLU_CUBIN_SHA256,
    )?;
    let (swiglu_module, swiglu_kernel) =
        cuda.load_kernel(&swiglu_cubin, KernelId::SwigluMxfp8Quant)?;
    let mut device_codes: Vec<Vec<u8>> = Vec::with_capacity(RANKS);
    let mut device_scales: Vec<Vec<u8>> = Vec::with_capacity(RANKS);
    for rank in 0..RANKS {
        let gate_bytes = bf16_bits_to_bytes(&gate_bits[rank]);
        let up_bytes = bf16_bits_to_bytes(&up_bits[rank]);
        let mut expected_codes = vec![0u8; rows * WIDTH_PADDED];
        let mut expected_scales = vec![0u8; rows * (WIDTH_PADDED / QUANT_BLOCK)];
        for row in 0..rows {
            let gate_f32 =
                bf16_bits_slice_to_f32(&gate_bits[rank], row * WIDTH_PADDED, WIDTH_PADDED);
            let up_f32 = bf16_bits_slice_to_f32(&up_bits[rank], row * WIDTH_PADDED, WIDTH_PADDED);
            let (codes, scales) = swiglu_ref::quantise_row(&gate_f32, &up_f32, SWIGLU_LIMIT);
            expected_codes[row * WIDTH_PADDED..(row + 1) * WIDTH_PADDED].copy_from_slice(&codes);
            expected_scales
                [row * (WIDTH_PADDED / QUANT_BLOCK)..(row + 1) * (WIDTH_PADDED / QUANT_BLOCK)]
                .copy_from_slice(&scales);
        }
        let mut rank_allocated = 0usize;
        let gate_d = cuda.alloc(gate_bytes.len(), &mut rank_allocated)?;
        let up_d = cuda.alloc(up_bytes.len(), &mut rank_allocated)?;
        let codes_d = cuda.alloc(expected_codes.len(), &mut rank_allocated)?;
        let scales_d = cuda.alloc(expected_scales.len(), &mut rank_allocated)?;
        println!(
            "  rank {rank}: rows {rows}, cols {WIDTH_PADDED}, device bytes admitted {rank_allocated}"
        );
        cuda.copy_htod(&gate_d, &gate_bytes)?;
        cuda.copy_htod(&up_d, &up_bytes)?;
        let params: [u64; 10] = [
            gate_d.0,
            up_d.0,
            codes_d.0,
            scales_d.0,
            rows as u64,
            WIDTH_PADDED as u64,
            u64::from(SWIGLU_LIMIT.to_bits()),
            0,
            args.layer as u64,
            rank as u64,
        ];
        let total_blocks = (rows * WIDTH_PADDED / QUANT_BLOCK) as u32;
        let grid = (total_blocks.div_ceil(8), 1u32, 1u32);
        check_floor(
            args.floor_gib,
            &format!("m17a stage 3 rank {rank} pre-launch"),
        )?;
        cuda.launch(swiglu_kernel, grid, (256, 1, 1), &params)?;
        let mut codes_first = vec![0u8; expected_codes.len()];
        let mut scales_first = vec![0u8; expected_scales.len()];
        cuda.copy_dtoh(&mut codes_first, &codes_d)?;
        cuda.copy_dtoh(&mut scales_first, &scales_d)?;
        cuda.launch(swiglu_kernel, grid, (256, 1, 1), &params)?;
        let mut codes_second = vec![0u8; expected_codes.len()];
        let mut scales_second = vec![0u8; expected_scales.len()];
        cuda.copy_dtoh(&mut codes_second, &codes_d)?;
        cuda.copy_dtoh(&mut scales_second, &scales_d)?;
        check_floor(
            args.floor_gib,
            &format!("m17a stage 3 rank {rank} post-launch"),
        )?;
        cuda.free_buffers(&[&gate_d, &up_d, &codes_d, &scales_d]);
        let code_mismatches = codes_first
            .iter()
            .zip(&expected_codes)
            .filter(|(a, b)| a != b)
            .count();
        let scale_mismatches = scales_first
            .iter()
            .zip(&expected_scales)
            .filter(|(a, b)| a != b)
            .count();
        Verdict::push(
            &mut verdicts,
            &format!("stage 3 rank {rank}: E4M3 codes bit-exact"),
            code_mismatches == 0,
            format!("{code_mismatches} mismatched of {}", codes_first.len()),
        );
        Verdict::push(
            &mut verdicts,
            &format!("stage 3 rank {rank}: ue8m0 scales bit-exact"),
            scale_mismatches == 0,
            format!("{scale_mismatches} mismatched of {}", scales_first.len()),
        );
        Verdict::push(
            &mut verdicts,
            &format!("stage 3 rank {rank}: two launches byte-identical"),
            codes_first == codes_second && scales_first == scales_second,
            format!(
                "{} code + {} scale bytes compared",
                codes_first.len(),
                scales_first.len()
            ),
        );
        device_codes.push(codes_first);
        device_scales.push(scales_first);
    }
    cuda.free(swiglu_module, &[]);
    println!(
        "stage 3 device codes/scales are measured but not the combine operand (F6 seam: the model multiplies the routing weight BEFORE the quantiser, and weight-multiplying quantised codes would quantise twice)"
    );
    let _ = (&device_codes, &device_scales);

    // ---------------------------------------------------------------------------------------
    // Stage 4: down projection + weighted combine, four serial rank launches.
    // ---------------------------------------------------------------------------------------
    println!(
        "--- stage 4: b12x_mxfp4_moe_sm121 down/combine, 4 serial rank launches (residual chain) ---"
    );
    println!(
        "comparison class (declared before any number): device bf16 BIT-EXACT against the chained mxfp4_ref::moe_combine over the same bytes (four ranks, each rank's bf16 output is the next rank's residual hidden, so the chained rounding is part of both sides); two full chains byte-identical. The activation operand is host-quantised by mxfp4_ref::routed_activation_operand over the device gate/up bf16 values (the F6 seam) - the combine is WEIGHTED, its operand is not device-produced."
    );
    let routing_weights = device_weights.clone();
    let activation_full = assemble_activation(&gate_bits, &up_bits, rows);
    let mut act_fp8_rank: Vec<Vec<u8>> = Vec::with_capacity(RANKS);
    let mut act_scales_rank: Vec<Vec<u8>> = Vec::with_capacity(RANKS);
    for rank in 0..RANKS {
        let mut codes = vec![0u8; rows * WIDTH_PADDED];
        let mut scales = vec![0u8; rows * MOE_SCALE_ROW];
        for row in 0..rows {
            let slice =
                &activation_full[row * INTER + rank * WIDTH..row * INTER + (rank + 1) * WIDTH];
            let (rank_codes, rank_scales) =
                mxfp4_ref::routed_activation_operand(slice, routing_weights[row]);
            codes[row * WIDTH_PADDED..row * WIDTH_PADDED + WIDTH].copy_from_slice(&rank_codes);
            scales[row * MOE_SCALE_ROW..row * MOE_SCALE_ROW + WIDTH / QUANT_BLOCK]
                .copy_from_slice(&rank_scales);
        }
        act_fp8_rank.push(codes);
        act_scales_rank.push(scales);
    }
    println!(
        "F6 seam crossing: device gate/up bf16 [rows {rows}][{WIDTH_PADDED}] per rank -> host SwiGLU ({SWIGLU_LIMIT}) -> routing weight per routed row -> routed_activation_operand -> [rows {rows}][{WIDTH_PADDED}] E4M3 + [rows {rows}][{MOE_SCALE_ROW}] ue8m0 per rank"
    );
    let w2_rank: Vec<Vec<u8>> = (0..RANKS)
        .map(|rank| {
            let mut out = Vec::with_capacity(0);
            for expert in &expert_table {
                out.extend_from_slice(&slice_and_pad_rows(
                    &expert.w2,
                    HIDDEN,
                    IN_PACKED,
                    rank * (WIDTH / 2),
                    WIDTH / 2,
                    MOE_PACKED_ROW,
                ));
            }
            out
        })
        .collect();
    let ws2_rank: Vec<Vec<u8>> = (0..RANKS)
        .map(|rank| {
            let mut out = Vec::with_capacity(0);
            for expert in &expert_table {
                out.extend_from_slice(&slice_and_pad_rows(
                    &expert.s2,
                    HIDDEN,
                    IN_SCALES,
                    rank * (WIDTH / QUANT_BLOCK),
                    WIDTH / QUANT_BLOCK,
                    MOE_SCALE_ROW,
                ));
            }
            out
        })
        .collect();
    println!(
        "w2 rank slices: 4 ranks x ({} packed + {} scale bytes) from the released w2 tensors, each stored row sliced to {WIDTH} and padded to {WIDTH_PADDED}",
        w2_rank[0].len(),
        ws2_rank[0].len()
    );

    // The residual stand-in and the reference chain.
    let mut hidden_bits: Vec<u16> = Vec::with_capacity(args.tokens * HIDDEN);
    for token in 0..args.tokens {
        for column in 0..HIDDEN {
            hidden_bits.push(u16::from_le_bytes([
                x_bytes[(token * HIDDEN + column) * 2],
                x_bytes[(token * HIDDEN + column) * 2 + 1],
            ]));
        }
    }
    let mut reference = hidden_bits.clone();
    for rank in 0..RANKS {
        let (next, _) = mxfp4_ref::moe_combine(
            &reference,
            &act_fp8_rank[rank],
            &act_scales_rank[rank],
            &w2_rank[rank],
            &ws2_rank[rank],
            &local_ids,
            args.tokens,
            HIDDEN,
            local_experts,
        );
        reference = next;
    }
    println!(
        "reference: mxfp4_ref::moe_combine chained over the 4 rank slices, residual seed = the declared bf16 embed stand-in"
    );

    let combine_cubin = pinned_cubin(
        &args.cubin_dir,
        "b12x_mxfp4_moe_sm121",
        MXFP4_MOE_CUBIN_BYTES,
        MXFP4_MOE_CUBIN_SHA256,
    )?;
    let (combine_module, combine_kernel) = mxfp4_moe_launch::load(&cuda, &combine_cubin)?;
    let device_chain = |cuda: &Cuda, label: &str| -> Result<Vec<u16>, String> {
        let mut allocated = 0usize;
        let hidden_d = cuda.alloc(args.tokens * HIDDEN * 2, &mut allocated)?;
        let out_d = cuda.alloc(args.tokens * HIDDEN * 2, &mut allocated)?;
        let act_d = cuda.alloc(act_fp8_rank[0].len(), &mut allocated)?;
        let acts_d = cuda.alloc(act_scales_rank[0].len(), &mut allocated)?;
        let w_d = cuda.alloc(w2_rank[0].len(), &mut allocated)?;
        let ws_d = cuda.alloc(ws2_rank[0].len(), &mut allocated)?;
        let ids_d = cuda.alloc(local_ids.len() * 4, &mut allocated)?;
        println!("  {label}: device bytes admitted {allocated}");
        cuda.copy_htod(&hidden_d, &bf16_bits_to_bytes(&hidden_bits))?;
        let mut last: Vec<u16> = hidden_bits.clone();
        for rank in 0..RANKS {
            cuda.copy_htod(&act_d, &act_fp8_rank[rank])?;
            cuda.copy_htod(&acts_d, &act_scales_rank[rank])?;
            cuda.copy_htod(&w_d, &w2_rank[rank])?;
            cuda.copy_htod(&ws_d, &ws2_rank[rank])?;
            cuda.copy_htod(&ids_d, &u32s_to_bytes(&local_ids))?;
            let params = Mxfp4MoeParams::new(
                hidden_d,
                act_d,
                acts_d,
                w_d,
                ws_d,
                ids_d,
                out_d,
                args.tokens as u32,
                0,
                args.layer as u32,
                rank as u64,
            );
            check_floor(
                args.floor_gib,
                &format!("m17a stage 4 {label} rank {rank} pre-launch"),
            )?;
            mxfp4_moe_launch::launch(cuda, combine_kernel, &params)?;
            let bytes = read_bf16_bytes(cuda, &out_d, args.tokens * HIDDEN)?;
            cuda.copy_htod(&hidden_d, &bytes)?;
            last = bytes_bf16_bits(&bytes);
            check_floor(
                args.floor_gib,
                &format!("m17a stage 4 {label} rank {rank} post-launch"),
            )?;
        }
        cuda.free_buffers(&[&hidden_d, &out_d, &act_d, &acts_d, &w_d, &ws_d, &ids_d]);
        Ok(last)
    };
    let first_chain = device_chain(&cuda, "chain 1")?;
    let second_chain = device_chain(&cuda, "chain 2")?;
    cuda.free(combine_module, &[]);
    let mismatches = first_chain
        .iter()
        .zip(&reference)
        .filter(|(a, b)| a != b)
        .count();
    Verdict::push(
        &mut verdicts,
        "stage 4: device chain bit-exact vs moe_combine chain",
        mismatches == 0,
        format!(
            "{mismatches} mismatched bf16 codes of {}",
            first_chain.len()
        ),
    );
    Verdict::push(
        &mut verdicts,
        "stage 4: two full chains byte-identical",
        first_chain == second_chain,
        format!("{} bf16 codes compared", first_chain.len()),
    );
    println!(
        "stage 4 output: {} token rows x {HIDDEN} bf16 codes (residual + weighted routed sum over {TOP_K} slots and 4 ranks)",
        args.tokens
    );

    if args.repeat > 0 {
        println!(
            "repeat requested ({}) but m17a makes no timing claim; use the per-stage launch counts printed above for cost arithmetic",
            args.repeat
        );
    }

    check_floor(args.floor_gib, "m17a end")?;
    println!("--- NOT claimed by m17a ---");
    println!(
        "not a fully-device weighted routed block (F6 host seam above), not a chain/token/per-layer claim, not layer text (activation and residual are released embed rows), not the shared expert or the shared-add epilogue (m17b), not engine integration, not host-time saved, not a performance claim. Stage 2's 8 launches are independent; stage 4's 4 launches are serial by construction (each rank's bf16 output is the next rank's residual)."
    );
    let failed: Vec<&str> = verdicts
        .iter()
        .filter(|verdict| !verdict.ok)
        .map(|verdict| verdict.name.as_str())
        .collect();
    if failed.is_empty() {
        println!(
            "m17a: PASS ({} checks; 4 stages; tokens {}; four pinned cubins)",
            verdicts.len(),
            args.tokens
        );
        Ok(())
    } else {
        let details: Vec<String> = verdicts
            .iter()
            .filter(|verdict| !verdict.ok)
            .map(|verdict| format!("{}: {}", verdict.name, verdict.detail))
            .collect();
        Err(format!(
            "m17a: FAIL at {} of {} checks: {:?}",
            failed.len(),
            verdicts.len(),
            details
        ))
    }
}

// ---------------------------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------------------------

fn check_floor(floor_gib: u64, point: &str) -> Result<(), String> {
    let text = std::fs::read_to_string("/proc/meminfo")
        .map_err(|error| format!("cannot read /proc/meminfo: {error}"))?;
    let kb = text
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| "cannot parse MemAvailable".to_owned())?;
    let bytes = kb.saturating_mul(1024);
    let floor = floor_gib.saturating_mul(1024 * 1024 * 1024);
    if bytes < floor {
        return Err(format!(
            "MemAvailable at {point} is {kb} kB ({:.2} GiB), below the {floor_gib} GiB floor: refusing to run",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        ));
    }
    println!(
        "MemAvailable at {point}: {kb} kB ({:.1} GiB), floor {floor_gib} GiB - ok",
        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(())
}

fn pinned_cubin(
    dir: &std::path::Path,
    name: &str,
    bytes: usize,
    sha: &str,
) -> Result<Vec<u8>, String> {
    let path = dir.join(format!("{name}.cubin"));
    let cubin = std::fs::read(&path).map_err(|error| {
        format!(
            "{}: {error} (build the cubin first: nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/{name}.cu -o {})",
            path.display(),
            path.display()
        )
    })?;
    let digest = Sha256::digest(&cubin);
    let hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    println!(
        "cubin: {} ({} B, sha256 {hex})",
        path.display(),
        cubin.len()
    );
    if cubin.len() != bytes || hex != sha {
        return Err(format!(
            "{name}: pinned {bytes} B / {sha}, got {} B / {hex}; refusing to measure a different kernel",
            cubin.len()
        ));
    }
    Ok(cubin)
}

fn read_embed_rows(snap: &mut Snapshot, tokens: usize) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::with_capacity(tokens * HIDDEN * 2);
    for token in 0..tokens {
        let mut row = vec![0u8; HIDDEN * 2];
        snap.read_row_bf16_into("embed.weight", token, HIDDEN, &mut row)?;
        bytes.extend_from_slice(&row);
    }
    Ok(bytes)
}

struct ExpertTensors {
    w1: Vec<u8>,
    s1: Vec<u8>,
    w3: Vec<u8>,
    s3: Vec<u8>,
    w2: Vec<u8>,
    s2: Vec<u8>,
}

impl ExpertTensors {
    fn bytes(&self) -> usize {
        self.w1.len()
            + self.s1.len()
            + self.w3.len()
            + self.s3.len()
            + self.w2.len()
            + self.s2.len()
    }
}

fn read_expert_tensors(
    snap: &mut Snapshot,
    prefix: &str,
    table: &[(usize, usize)],
) -> Result<Vec<ExpertTensors>, String> {
    let mut experts = Vec::with_capacity(table.len());
    for (local, expert) in table {
        let base = format!("{prefix}.ffn.experts.{expert}");
        experts.push(ExpertTensors {
            w1: snap.read_raw(&format!("{base}.w1.weight"), "I8", &[INTER, PACKED_ROW])?,
            s1: snap.read_raw(&format!("{base}.w1.scale"), "F8_E8M0", &[INTER, SCALE_ROW])?,
            w3: snap.read_raw(&format!("{base}.w3.weight"), "I8", &[INTER, PACKED_ROW])?,
            s3: snap.read_raw(&format!("{base}.w3.scale"), "F8_E8M0", &[INTER, SCALE_ROW])?,
            w2: snap.read_raw(&format!("{base}.w2.weight"), "I8", &[HIDDEN, IN_PACKED])?,
            s2: snap.read_raw(&format!("{base}.w2.scale"), "F8_E8M0", &[HIDDEN, IN_SCALES])?,
        });
        if *local % 64 == 0 || *local + 1 == table.len() {
            println!(
                "  staged expert {expert} (local {local}) of {}",
                table.len()
            );
        }
    }
    Ok(experts)
}

/// Concatenate one rank's rows from every expert's stored tensor.
fn rank_slice<F>(experts: &[ExpertTensors], pick: F, rank: usize, take: usize) -> Vec<u8>
where
    F: Fn(&ExpertTensors) -> &[u8],
{
    let mut out = Vec::with_capacity(experts.len() * take);
    for expert in experts {
        let tensor = pick(expert);
        let start = rank * take;
        out.extend_from_slice(&tensor[start..start + take]);
    }
    out
}

/// The table of distinct router ids: local index for every global expert id, usize::MAX if unselected.
fn local_expert_table(ids: &[u32]) -> Vec<usize> {
    let mut table = vec![usize::MAX; EXPERTS];
    let mut next = 0usize;
    for id in ids {
        let expert = *id as usize;
        if table[expert] == usize::MAX {
            table[expert] = next;
            next += 1;
        }
    }
    table
}

fn unique_experts(ids: &[u32]) -> Vec<u32> {
    let mut seen = vec![false; EXPERTS];
    let mut out = Vec::new();
    for id in ids {
        if !seen[*id as usize] {
            seen[*id as usize] = true;
            out.push(*id);
        }
    }
    out
}

type RouterOutputs = (Vec<u32>, Vec<f32>, Vec<f32>);

fn read_router_outputs(
    cuda: &Cuda,
    ids_d: &sparkflash_kernels::cuda_driver::Deviceptr,
    weights_d: &sparkflash_kernels::cuda_driver::Deviceptr,
    scores_d: &sparkflash_kernels::cuda_driver::Deviceptr,
    tokens: usize,
) -> Result<RouterOutputs, String> {
    let mut ids_bytes = vec![0u8; tokens * TOP_K * 4];
    let mut weight_bytes = vec![0u8; tokens * TOP_K * 4];
    let mut score_bytes = vec![0u8; tokens * EXPERTS * 4];
    cuda.copy_dtoh(&mut ids_bytes, ids_d)?;
    cuda.copy_dtoh(&mut weight_bytes, weights_d)?;
    cuda.copy_dtoh(&mut score_bytes, scores_d)?;
    Ok((
        u32s_from_bytes(&ids_bytes),
        f32_bytes_vec(&weight_bytes),
        f32_bytes_vec(&score_bytes),
    ))
}

fn router_raw_bytes(ids: &[u32], weights: &[f32], scores: &[f32]) -> Vec<u8> {
    let mut raw = Vec::new();
    raw.extend_from_slice(&u32s_to_bytes(ids));
    raw.extend_from_slice(&f32s_to_bytes(weights));
    raw.extend_from_slice(&f32s_to_bytes(scores));
    raw
}

fn read_bf16_bytes(
    cuda: &Cuda,
    ptr: &sparkflash_kernels::cuda_driver::Deviceptr,
    values: usize,
) -> Result<Vec<u8>, String> {
    let mut bytes = vec![0u8; values * 2];
    cuda.copy_dtoh(&mut bytes, ptr)?;
    Ok(bytes)
}

/// Assemble the full [rows][2304] SwiGLU activation from the four rank outputs.
///
/// Each rank's gate/up row is [640] for that rank's own 576-column slice (columns 576..640 are the
/// kernel's zero padding); the four slices are concatenated in rank order. This function exists as a
/// named unit because a window caught an indexing error here: reading rank r at `row * 640 + r * 576`
/// instead of at `row * 640` runs off the end of the 640-wide row.
fn assemble_activation(gate_bits: &[Vec<u16>], up_bits: &[Vec<u16>], rows: usize) -> Vec<f32> {
    assert_eq!(gate_bits.len(), RANKS, "one gate output per rank");
    assert_eq!(up_bits.len(), RANKS, "one up output per rank");
    let mut activation = vec![0.0f32; rows * INTER];
    for row in 0..rows {
        for rank in 0..RANKS {
            let base = row * WIDTH_PADDED;
            assert!(
                base + WIDTH <= gate_bits[rank].len(),
                "rank {rank} row {row} is a whole 640-wide output row"
            );
            for column in 0..WIDTH {
                let gate = bf16_value(gate_bits[rank][base + column]);
                let up = bf16_value(up_bits[rank][base + column]);
                activation[row * INTER + rank * WIDTH + column] =
                    swiglu_ref::swiglu(gate, up, SWIGLU_LIMIT);
            }
        }
    }
    activation
}

fn count_padding_nonzero(bits: &[u16]) -> usize {
    let rows = bits.len() / WIDTH_PADDED;
    let mut nonzero = 0usize;
    for row in 0..rows {
        for column in WIDTH..WIDTH_PADDED {
            if bits[row * WIDTH_PADDED + column] != 0 {
                nonzero += 1;
            }
        }
    }
    nonzero
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| u16::from_le_bytes(*chunk))
        .map(bf16_value)
        .collect()
}

fn bf16_bits_slice_to_f32(bits: &[u16], start: usize, len: usize) -> Vec<f32> {
    bits[start..start + len]
        .iter()
        .map(|value| bf16_value(*value))
        .collect()
}

fn bytes_bf16_bits(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| u16::from_le_bytes(*chunk))
        .collect()
}

fn bf16_bits_to_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn f32_bytes_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn f32s_to_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u32s_to_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u32s_from_bytes(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect()
}

fn max_abs_delta(a: &[f32], b: &[f32]) -> (f64, usize) {
    assert_eq!(a.len(), b.len(), "the compared arrays are the same length");
    let mut max = 0.0f64;
    let mut at = 0usize;
    for (index, (x, y)) in a.iter().zip(b).enumerate() {
        let delta = (f64::from(*x) - f64::from(*y)).abs();
        if delta > max {
            max = delta;
            at = index;
        }
    }
    (max, at)
}

fn ordered_bits(value: f32) -> u64 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        u64::from(!bits)
    } else {
        u64::from(bits) | 0x8000_0000
    }
}

fn max_ulp(a: &[f32], b: &[f32]) -> u64 {
    assert_eq!(a.len(), b.len(), "the compared arrays are the same length");
    let mut max = 0u64;
    for (x, y) in a.iter().zip(b) {
        let distance = ordered_bits(*x).abs_diff(ordered_bits(*y));
        max = max.max(distance);
    }
    max
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_argument_parser_reads_the_node_command() {
        let args = Args::parse_from(
            [
                "--checkpoint",
                "/ckpt",
                "--cubin-dir",
                "/out",
                "--tokens",
                "130",
                "--layer",
                "0",
                "--floor-gib",
                "32",
            ]
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        )
        .expect("the documented command parses");
        assert_eq!(args.tokens, 130);
        assert_eq!(args.layer, 0);
        assert_eq!(args.floor_gib, 32);
    }

    #[test]
    fn the_expert_table_maps_each_selected_id_to_one_local_slot() {
        let table = local_expert_table(&[7, 3, 7, 11, 3]);
        assert_eq!(table[7], 0);
        assert_eq!(table[3], 1);
        assert_eq!(table[11], 2);
        assert_eq!(table[0], usize::MAX);
        assert_eq!(unique_experts(&[7, 3, 7, 11]), vec![7, 3, 11]);
    }

    #[test]
    fn the_rank_slices_are_disjoint_and_cover_the_stored_rows() {
        let stored: Vec<u8> = (0..INTER * PACKED_ROW)
            .map(|index| (index % 251) as u8)
            .collect();
        let expert = vec![ExpertTensors {
            w1: stored,
            s1: vec![0u8; INTER * SCALE_ROW],
            w3: vec![0u8; INTER * PACKED_ROW],
            s3: vec![0u8; INTER * SCALE_ROW],
            w2: vec![0u8; HIDDEN * IN_PACKED],
            s2: vec![0u8; HIDDEN * IN_SCALES],
        }];
        let slices: Vec<Vec<u8>> = (0..RANKS)
            .map(|rank| rank_slice(&expert, |expert| &expert.w1, rank, WIDTH * PACKED_ROW))
            .collect();
        for (rank, slice) in slices.iter().enumerate() {
            assert_eq!(slice.len(), WIDTH * PACKED_ROW);
            assert_eq!(
                slice[0],
                ((rank * WIDTH * PACKED_ROW) % 251) as u8,
                "rank {rank} starts at its own stored row"
            );
        }
    }

    #[test]
    fn the_padding_check_counts_only_columns_after_the_real_width() {
        let mut bits = vec![0u16; WIDTH_PADDED];
        bits[WIDTH] = 0x3f80;
        assert_eq!(count_padding_nonzero(&bits), 1);
        bits[WIDTH - 1] = 0x3f80;
        assert_eq!(
            count_padding_nonzero(&bits),
            1,
            "real columns are not padding"
        );
    }

    #[test]
    fn the_activation_reads_each_ranks_own_640_wide_row_not_a_window() {
        use sparkflash_kernels::m4_quant::bf16_bits;
        let rows = 2usize;
        let mut gate_bits = vec![vec![0u16; rows * WIDTH_PADDED]; RANKS];
        let mut up_bits = vec![vec![0u16; rows * WIDTH_PADDED]; RANKS];
        for rank in 0..RANKS {
            for row in 0..rows {
                for column in 0..WIDTH_PADDED {
                    // Every column of every rank row is distinct, padding included, so reading the
                    // wrong offset cannot accidentally agree.
                    let gate = (rank * 1000 + row * 100 + column) as f32 * 1e-3;
                    let up = -1.0 + column as f32 * 2e-3;
                    gate_bits[rank][row * WIDTH_PADDED + column] = bf16_bits(gate);
                    up_bits[rank][row * WIDTH_PADDED + column] = bf16_bits(up);
                }
            }
        }
        let activation = assemble_activation(&gate_bits, &up_bits, rows);
        assert_eq!(activation.len(), rows * INTER);
        for row in 0..rows {
            for rank in 0..RANKS {
                for column in 0..WIDTH {
                    let at = row * WIDTH_PADDED + column;
                    let gate = bf16_value(gate_bits[rank][at]);
                    let up = bf16_value(up_bits[rank][at]);
                    assert_eq!(
                        activation[row * INTER + rank * WIDTH + column],
                        swiglu_ref::swiglu(gate, up, SWIGLU_LIMIT),
                        "rank {rank} row {row} column {column}"
                    );
                }
            }
        }
    }
}

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

//! m26_moe_stage: the layer-0 MoE stage the four-rank loop wires, one harness with a host twin
//! (CPU, no driver) and a device run, five pinned cubins, released checkpoint bytes only.
//!
//! THIS EXAMPLE IS THE EVIDENCE THE STORED-LAYOUT FOUR-RANK MoE STAGE COMPOSES ON THE DEVICE, and
//! the CPU twin of the same arithmetic. It is not the served forward pass and it does not touch the
//! engine: the host seams the engine still runs (hc_ffn mixes, ffn pre-mix, ffn norm, D8 activation
//! quantise, shared expert, ffn hc_post) are the loop's host code; this harness starts from their
//! bytes (--input) or from the declared released-embed stand-in.
//!
//! The stage sequence, per rank, per layer:
//!
//! 1. ROUTER (moe_router_sm121, 25,432 B / 42f6bbdc...): released layers.N.ffn.gate.weight BF16
//!    [384, 5120] and gate.bias F32 [384] (noaux_tc; bias_vl unused). Input is the BF16 round of
//!    the pre-FFN hidden row. Class: ids EXACT vs router_ref, scores <= 1e-6 abs and <= 8 ULP,
//!    weights <= 1e-6 abs and <= 16 ULP, two launches byte-identical.
//! 2. GATE/UP (routed_gate_up_sm121, 15,672 B / 12ddc097..., the split-K pin): 4 rank x 2
//!    projection launches; rank slice = rows 576r..576r+575 of the released w1/w3 I8 [2304, 2560]
//!    (scales [2304, 160]); grid (tokens*6, 20, 1), block 256. Class: BIT-IDENTICAL to
//!    routed_gate_up_ref on every tokens*6*640 bf16 value of both projections, columns 576..639
//!    exactly zero, two launches byte-identical.
//! 3. WEIGHTED SWIGLU + MXFP8 QUANT (swiglu_mxfp8_quant_weighted_sm121, 13,000 B / 0e602daf...):
//!    the router's per-row weights are the weight operand, read IN PLACE from the device router
//!    output. Class: E4M3 codes and ue8m0 scales BIT-EXACT vs swiglu_ref::quantise_row_weighted
//!    on the same 640-wide rows and weights, the real 576-wide operand BIT-EXACT vs
//!    mxfp4_ref::routed_activation_operand, two launches byte-identical.
//! 4. STORED DOWN COMBINE (b12x_mxfp4_moe_stored_sm121, 11,544 B / c452a9aa...): one launch per
//!    rank, grid (tokens, 160, 1), block 256, ten argument words. The weight operand is the
//!    released w2 [5120, 1152] I8 whose per-row prefix at offset r*288 is this rank's stored code
//!    row, and w2.scale [5120, 72] whose prefix at offset r*18 is its stored scale row: two
//!    separate pointers, no 306-byte interleaved row and no 640-wide padding. The kernel ADDS into
//!    a caller-zeroed f32 [tokens*6][5120] slot accumulator. Class: rank f32 slot accumulator
//!    BIT-EXACT vs mxfp4_ref::moe_combine_rank_stored on the same operand bytes; a second chain
//!    from a freshly zeroed accumulator byte-identical.
//! 5. CANONICAL REDUCE (host, transport code): per rank
//!    mxfp4_ref::moe_combine_routed_f32 -> f32 [tokens][5120]; payload = bf16 RNE of each rank
//!    value; canonical_all_reduce_into's rule (f32 ascending-rank sum from 0.0, one bf16 RNE
//!    round) -> routed delta BF16 [tokens][5120]. Class: payloads and the reduced delta
//!    BIT-EXACT vs the canonical twin over the reference accumulators.
//! 6. NATIVE REFERENCE (host): the same reduced delta vs the native full-K f32 sum (the incumbent
//!    640-padded moe_combine_rank chain), relative L2 <= 5.0e-3. This is the ONLY approximate
//!    reference in the stage: the distributed arm rounds each rank's partial and sums four bf16
//!    payloads where native sums 2304 K columns in f32 before the per-expert bf16 round.
//!
//! Usage (node, checkpoint present; aarch64):
//!
//! ~~~text
//! m26_moe_stage --checkpoint <snapshot dir> --cubin-dir <dir> --tokens 8 [--layer 0]
//! m26_moe_stage --checkpoint <snapshot dir> --twin-only [--input <expanded bundle dir>]
//! m26_moe_stage --checkpoint <snapshot dir> --cubin-dir <dir> --input <expanded bundle dir>
//!               [--evidence out.json] [--dump routed_delta.bf16] [--repeat 0]
//! ~~~
//!
//! --input is the twin-expanded bundle (see docs/archive/build/moe-stage-contract.md): pre_ffn_hidden.f32
//! (tokens*5120 f32 LE) and, optionally, act_fp8.bin / act_scales.bin for the H4 operand. Without
//! --input the released embed.weight rows 0..tokens are the declared pre-FFN stand-in.
//! --twin-only runs the CPU twin and prints/records the EXPECTED evidence record; the device run
//! prints/records the measured one. Both print the same m26 stage PASS lines.
//!
//! The cubin dir must hold the four pinned cubins this stage loads, each refused by size+sha256
//! before allocation: moe_router_sm121, routed_gate_up_sm121,
//! swiglu_mxfp8_quant_weighted_sm121, b12x_mxfp4_moe_stored_sm121. There is no new kernel and no
//! baseline change.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::path::PathBuf;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[path = "../tests/support/b2_harness.rs"]
mod b2_harness;

use self::b2_harness::{Snapshot, slice_and_pad_rows};
use sparkflash_kernels::cuda_driver::{Cuda, KernelId};
use sparkflash_kernels::m4_quant::{act_quant_fp8_row, bf16_bits, bf16_value};
use sparkflash_kernels::mxfp4_moe_stored_abi as sabi;
use sparkflash_kernels::mxfp4_moe_stored_launch::{self as stored_launch, Mxfp4MoeStoredParams};
use sparkflash_kernels::mxfp4_ref;
use sparkflash_kernels::routed_gate_up_abi as gabi;
use sparkflash_kernels::routed_gate_up_launch::{self, GateUpLaunchParams};
use sparkflash_kernels::routed_gate_up_ref as twin;
use sparkflash_kernels::router_abi as rabi;
use sparkflash_kernels::router_launch::{self, RouterLaunchParams};
use sparkflash_kernels::router_ref::{self, RouterParams};
use sparkflash_kernels::swiglu_ref::{self, SWIGLU_LIMIT};

/// Routed experts in the released configuration.
const EXPERTS: usize = 384;
/// Hidden width in the released configuration.
const HIDDEN: usize = 5120;
/// Routed experts selected per token.
const TOP_K: usize = 6;
/// MoE intermediate width in the released configuration.
const INTER: usize = 2304;
/// Tensor-parallel ranks the routed experts are sharded over.
const RANKS: usize = 4;
/// The real intermediate columns one rank owns.
const WIDTH: usize = 576;
/// The padded intermediate width the gate/up and incumbent combine use.
const WIDTH_PADDED: usize = 640;
/// A released w1/w3 row: HIDDEN FP4 values packed two per byte.
const PACKED_ROW: usize = HIDDEN / 2;
/// A released w1/w3 scale row: one ue8m0 byte per 32 values.
const SCALE_ROW: usize = HIDDEN / 32;
/// A released w2 row: INTER FP4 values packed two per byte.
const IN_PACKED: usize = INTER / 2;
/// A released w2 scale row.
const IN_SCALES: usize = INTER / 32;
/// The incumbent 640-wide combine's packed row (real 576 + zero padding).
const MOE_PACKED_ROW: usize = WIDTH_PADDED / 2;
/// The incumbent 640-wide combine's scale row.
const MOE_SCALE_ROW: usize = WIDTH_PADDED / 32;
/// The activation quantiser block.
const QUANT_BLOCK: usize = 32;
/// The stored kernel's code row: rank r's per-row 288-byte prefix of the released w2 row.
const STORED_CODE_ROW: usize = WIDTH / 2;
/// The stored kernel's scale row: rank r's per-row 18-byte prefix of the released w2 scale row.
const STORED_SCALE_ROW: usize = WIDTH / QUANT_BLOCK;
/// The released mHC width multiplier (hc_mult); the loop's hc_post output is tokens*4*5120 f32.
const HC_MULT: usize = 4;
/// The only approximate reference's bar: the distributed delta vs the native full-K f32 sum.
const NATIVE_L2_BAR: f64 = 5.0e-3;
/// The evidence record schema this harness writes.
const EVIDENCE_SCHEMA: &str = "sparkflash-moe-stage-evidence/1";

/// The weighted SwiGLU+quant cubin pin (unchanged by the split-K rewrite; this harness pins it so
/// a baseline re-pin cannot leave the harness loading a stale artefact).
const SWIGLU_WEIGHTED_CUBIN_BYTES: usize = 14_224;
/// The weighted SwiGLU+quant cubin sha256.
const SWIGLU_WEIGHTED_CUBIN_SHA256: &str =
    "6b237e6c20301ddeda7f47ff9add5385a19e789d818388c5eb22981f1af5377e";

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

    /// Prints the check line and records the verdict.
    fn push(verdicts: &mut Vec<Self>, name: &str, ok: bool, detail: String) {
        println!(
            "  check {name}: {} ({detail})",
            if ok { "ok" } else { "FAIL" }
        );
        verdicts.push(Self::new(name, ok, detail));
    }
}

/// The verdicts and the evidence record one run produces.
struct StageOutcome {
    verdicts: Vec<Verdict>,
    evidence: Value,
}

struct Args {
    checkpoint: PathBuf,
    cubin_dir: PathBuf,
    tokens: usize,
    layer: usize,
    repeat: usize,
    floor_gib: u64,
    dump: Option<PathBuf>,
    input: Option<PathBuf>,
    evidence: Option<PathBuf>,
    twin_only: bool,
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
            dump: None,
            input: None,
            evidence: None,
            twin_only: false,
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
                "--dump" => args.dump = Some(PathBuf::from(next()?)),
                "--input" => args.input = Some(PathBuf::from(next()?)),
                "--evidence" => args.evidence = Some(PathBuf::from(next()?)),
                "--twin-only" => args.twin_only = true,
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
        eprintln!("m26_moe_stage: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    if args.twin_only {
        let outcome = run_twin_only(&args)?;
        finish(&args, outcome)
    } else {
        let outcome = run_device(&args)?;
        finish(&args, outcome)
    }
}

fn run_device(args: &Args) -> Result<StageOutcome, String> {
    println!(
        "=== m26: layer-0 MoE stage - router, gate/up split-K, weighted SwiGLU+quant, stored down combine, canonical reduce ==="
    );
    println!(
        "geometry: hidden {HIDDEN}, intermediate {INTER} = {RANKS} rank slices of {WIDTH} (padded {WIDTH_PADDED}), routed experts {EXPERTS}, top-k {TOP_K}; tokens {}; hc_mult {HC_MULT}",
        args.tokens
    );
    println!(
        "devices: four pinned cubins (router, split-K gate/up, weighted SwiGLU+quant, stored down combine), each verified by size+sha256 before any allocation; each stage's comparison class is printed before the numbers it judges"
    );
    println!(
        "what this is not: not the served forward pass, not a chain/token/per-layer claim, not a timing claim; the engine host seams (hc_ffn mixes, ffn pre-mix, ffn norm, D8 quantise, shared expert, hc_post) are the loop's code. The pre-FFN rows are the --input bundle's H3 bytes, or released embed.weight rows as a declared stand-in."
    );
    check_floor(args.floor_gib, "m26 start")?;

    let mut verdicts: Vec<Verdict> = Vec::new();
    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());
    let prefix = format!("layers.{}", args.layer);

    // ---------------------------------------------------------------------------------------
    // Stage 1: router.
    // ---------------------------------------------------------------------------------------
    println!("--- stage 1: moe_router_sm121 (released gate weight + noaux_tc bias) ---");
    println!(
        "comparison class (declared before any number): ids EXACT vs router_ref on all {} x {TOP_K} selections; scores max abs delta <= 1e-6 and max ULP <= 8; weights max abs delta <= 1e-6 and max ULP <= 16; two launches byte-identical (the serial ascending-k bars; the stage-one score is the incumbent serial sequence, not a lane tree). The device's expf/log1pf are not libm's exp/ln_1p, which is the only known degree of freedom; ids are pinned because a tie-crossing selection would be a finding, not a tolerance (the tie rule is the kernel's strict-greater scan, lower expert id wins).",
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
    let (pre_ffn_hidden, router_x_f32, x_bytes, bundle_act) = load_input_any(&mut snap, args)?;
    let gate_f32 = bf16_bytes_to_f32(&gate_bytes);
    let bias_f32 = f32_bytes_vec(&bias_bytes);
    let mut twin_ids: Vec<u32> = Vec::with_capacity(args.tokens * TOP_K);
    let mut twin_weights: Vec<f32> = Vec::with_capacity(args.tokens * TOP_K);
    let mut twin_scores: Vec<f32> = Vec::with_capacity(args.tokens * EXPERTS);
    for token in 0..args.tokens {
        let x = &router_x_f32[token * HIDDEN..(token + 1) * HIDDEN];
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
    let (router_module, router_kernel, router_topk_module, router_topk) =
        router_launch::load(&cuda, &router_cubin)?;
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
    check_floor(args.floor_gib, "m26 stage 1 pre-launch")?;
    router_launch::launch(&cuda, router_kernel, router_topk, &router_params)?;
    let (device_ids, device_weights, device_scores) =
        read_router_outputs(&cuda, &ids_d, &weights_d, &scores_d, args.tokens)?;
    let first_router_bytes = router_raw_bytes(&device_ids, &device_weights, &device_scores);
    router_launch::launch(&cuda, router_kernel, router_topk, &router_params)?;
    let (ids_2, weights_2, scores_2) =
        read_router_outputs(&cuda, &ids_d, &weights_d, &scores_d, args.tokens)?;
    let second_router_bytes = router_raw_bytes(&ids_2, &weights_2, &scores_2);
    check_floor(args.floor_gib, "m26 stage 1 post-launch")?;
    // weights_d stays allocated: stage 3's weighted epilogue reads the router's per-row routing
    // weights IN PLACE as its weight operand, so the multiply happens on the device rather than on a
    // host copy. (The host readback above is only the comparison twin.)
    cuda.free(router_module, &[x_d, gate_d, bias_d, ids_d, scores_d]);
    cuda.free(router_topk_module, &[]);

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

    // H4: D8-quantise the pre-FFN rows; this is the gate/up A operand.
    let mut act_fp8 = Vec::with_capacity(args.tokens * HIDDEN);
    let mut act_scales = Vec::with_capacity(args.tokens * SCALE_ROW);
    for token in 0..args.tokens {
        let row = &pre_ffn_hidden[token * HIDDEN..(token + 1) * HIDDEN];
        let (codes, scales) = act_quant_fp8_row(row);
        act_fp8.extend_from_slice(&codes);
        act_scales.extend_from_slice(&scales);
    }
    if let Some((codes, scales)) = &bundle_act {
        let matches = codes == &act_fp8 && scales == &act_scales;
        Verdict::push(
            &mut verdicts,
            "input: bundle act bytes bit-exact vs the H4 quantise rule",
            matches,
            format!(
                "{} code + {} scale bytes compared",
                act_fp8.len(),
                act_scales.len()
            ),
        );
    }
    println!(
        "activation operand: {} pre-FFN rows x {HIDDEN} f32, D8-quantised to {} E4M3 bytes + {} ue8m0 bytes (the gate/up A operand)",
        args.tokens,
        act_fp8.len(),
        act_scales.len()
    );

    // Read the selected experts' released tensors once; the rank slices are cut from these bytes.
    let expert_table = read_expert_tensors(&mut snap, &prefix, &table_rows)?;
    let names_read: BTreeSet<String> = snap.read_names().into_iter().collect();
    let mut missing_names = 0usize;
    for (_, expert) in &table_rows {
        for name in expert_tensor_names(&prefix, *expert as u32) {
            if !names_read.contains(&name) {
                missing_names += 1;
            }
        }
    }
    Verdict::push(
        &mut verdicts,
        "stage 2: staged expert tensor names are the released global ids",
        missing_names == 0,
        format!(
            "{missing_names} of {} expected released expert tensor names missing from the {} names actually read",
            table_rows.len() * 6,
            names_read.len()
        ),
    );
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
        // cuMemAlloc does not promise zeroed pages. Prefill both output buffers with 0xFFFF before
        // the first launch, so the "columns 576..640 exactly zero" check cannot pass on recycled or
        // zeroed pages if a launch stopped writing the padding.
        let nonzero_fill = vec![0xFFu8; out_values * 2];
        cuda.copy_htod(&gate_out_d, &nonzero_fill)?;
        cuda.copy_htod(&up_out_d, &nonzero_fill)?;
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
            &format!("m26 stage 2 rank {rank} pre-launch"),
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
            &format!("m26 stage 2 rank {rank} post-launch"),
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
    // Stage 3: WEIGHTED SwiGLU + MXFP8 activation quantise on the device gate/up output: the
    // routing weight is multiplied in BEFORE the quantiser, on the device, from the router's own
    // weight buffer. This is the F6 host seam's closure.
    // ---------------------------------------------------------------------------------------
    println!(
        "--- stage 3: swiglu_mxfp8_quant_weighted_sm121 on the device gate/up bf16 values, with the device router's per-row weights as the weight operand ---"
    );
    println!(
        "comparison class (declared before any number): E4M3 codes and ue8m0 scales BIT-EXACT against swiglu_ref::quantise_row_weighted over the same {WIDTH_PADDED} bf16 values and the same per-row routing weight, for every routed row of every rank; the REAL {WIDTH}-wide operand is additionally bit-exact against mxfp4_ref::routed_activation_operand over the same real values and weight (the model operand the combine consumes); two launches byte-identical. The weight operand is the router's device output buffer weights_d, read in place, so the operand is device-built and no host multiply or host quantise remains in the path (the F6 seam is closed)."
    );
    let weighted_cubin = pinned_cubin(
        &args.cubin_dir,
        "swiglu_mxfp8_quant_weighted_sm121",
        SWIGLU_WEIGHTED_CUBIN_BYTES,
        SWIGLU_WEIGHTED_CUBIN_SHA256,
    )?;
    let (weighted_module, weighted_kernel) =
        cuda.load_kernel(&weighted_cubin, KernelId::SwigluMxfp8QuantWeighted)?;
    let mut device_codes: Vec<Vec<u8>> = Vec::with_capacity(RANKS);
    let mut device_scales: Vec<Vec<u8>> = Vec::with_capacity(RANKS);
    for rank in 0..RANKS {
        let gate_bytes = bf16_bits_to_bytes(&gate_bits[rank]);
        let up_bytes = bf16_bits_to_bytes(&up_bits[rank]);
        let mut expected_codes = vec![0u8; rows * WIDTH_PADDED];
        let mut expected_scales = vec![0u8; rows * MOE_SCALE_ROW];
        let mut model_codes = vec![0u8; rows * WIDTH];
        let mut model_scales = vec![0u8; rows * (WIDTH / QUANT_BLOCK)];
        for row in 0..rows {
            let weight = device_weights[row];
            let gate_f32 =
                bf16_bits_slice_to_f32(&gate_bits[rank], row * WIDTH_PADDED, WIDTH_PADDED);
            let up_f32 = bf16_bits_slice_to_f32(&up_bits[rank], row * WIDTH_PADDED, WIDTH_PADDED);
            let (codes, scales) =
                swiglu_ref::quantise_row_weighted(&gate_f32, &up_f32, SWIGLU_LIMIT, weight);
            expected_codes[row * WIDTH_PADDED..(row + 1) * WIDTH_PADDED].copy_from_slice(&codes);
            expected_scales[row * MOE_SCALE_ROW..(row + 1) * MOE_SCALE_ROW]
                .copy_from_slice(&scales);
            // The model's real-width operand: the old host path's operand, now only a comparison -
            // swiglu_ref::swiglu over the real {WIDTH} columns, then the reference's own
            // mxfp4_ref::routed_activation_operand. Calling the model rule itself (rather than an
            // arithmetically equal re-implementation) is what the check's name claims.
            let swiglu_row: Vec<f32> = gate_f32[..WIDTH]
                .iter()
                .zip(&up_f32[..WIDTH])
                .map(|(gate, up)| swiglu_ref::swiglu(*gate, *up, SWIGLU_LIMIT))
                .collect();
            let (real_codes, real_scales) =
                mxfp4_ref::routed_activation_operand(&swiglu_row, weight);
            model_codes[row * WIDTH..(row + 1) * WIDTH].copy_from_slice(&real_codes);
            model_scales[row * (WIDTH / QUANT_BLOCK)..(row + 1) * (WIDTH / QUANT_BLOCK)]
                .copy_from_slice(&real_scales);
        }
        let mut rank_allocated = 0usize;
        let gate_d = cuda.alloc(gate_bytes.len(), &mut rank_allocated)?;
        let up_d = cuda.alloc(up_bytes.len(), &mut rank_allocated)?;
        let codes_d = cuda.alloc(expected_codes.len(), &mut rank_allocated)?;
        let scales_d = cuda.alloc(expected_scales.len(), &mut rank_allocated)?;
        println!(
            "  rank {rank}: rows {rows}, cols {WIDTH_PADDED}, device bytes admitted {rank_allocated}, weights_d reused in place"
        );
        cuda.copy_htod(&gate_d, &gate_bytes)?;
        cuda.copy_htod(&up_d, &up_bytes)?;
        let params: [u64; 11] = [
            gate_d.0,
            up_d.0,
            weights_d.0,
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
            &format!("m26 stage 3 rank {rank} pre-launch"),
        )?;
        cuda.launch(weighted_kernel, grid, (256, 1, 1), &params)?;
        let mut codes_first = vec![0u8; expected_codes.len()];
        let mut scales_first = vec![0u8; expected_scales.len()];
        cuda.copy_dtoh(&mut codes_first, &codes_d)?;
        cuda.copy_dtoh(&mut scales_first, &scales_d)?;
        cuda.launch(weighted_kernel, grid, (256, 1, 1), &params)?;
        let mut codes_second = vec![0u8; expected_codes.len()];
        let mut scales_second = vec![0u8; expected_scales.len()];
        cuda.copy_dtoh(&mut codes_second, &codes_d)?;
        cuda.copy_dtoh(&mut scales_second, &scales_d)?;
        check_floor(
            args.floor_gib,
            &format!("m26 stage 3 rank {rank} post-launch"),
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
        let mut model_mismatches = 0usize;
        for row in 0..rows {
            for column in 0..WIDTH {
                if codes_first[row * WIDTH_PADDED + column] != model_codes[row * WIDTH + column] {
                    model_mismatches += 1;
                }
            }
            for block in 0..(WIDTH / QUANT_BLOCK) {
                if scales_first[row * MOE_SCALE_ROW + block]
                    != model_scales[row * (WIDTH / QUANT_BLOCK) + block]
                {
                    model_mismatches += 1;
                }
            }
        }
        Verdict::push(
            &mut verdicts,
            &format!("stage 3 rank {rank}: weighted E4M3 codes bit-exact"),
            code_mismatches == 0,
            format!("{code_mismatches} mismatched of {}", codes_first.len()),
        );
        Verdict::push(
            &mut verdicts,
            &format!("stage 3 rank {rank}: weighted ue8m0 scales bit-exact"),
            scale_mismatches == 0,
            format!("{scale_mismatches} mismatched of {}", scales_first.len()),
        );
        Verdict::push(
            &mut verdicts,
            &format!(
                "stage 3 rank {rank}: real {WIDTH}-wide operand bit-exact vs routed_activation_operand"
            ),
            model_mismatches == 0,
            format!(
                "{model_mismatches} mismatched of {} code+scale bytes in the real width",
                rows * (WIDTH + WIDTH / QUANT_BLOCK)
            ),
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
    cuda.free(weighted_module, &[]);
    cuda.free_buffers(&[&weights_d]);
    println!(
        "stage 3 output: device-built weighted operand per rank, {} rows x {WIDTH_PADDED} E4M3 codes + {} ue8m0 scales (the real {WIDTH} columns equal the model's routed_activation_operand)",
        rows, MOE_SCALE_ROW
    );
    println!(
        "F6 seam closed: device router weights_d -> weighted epilogue -> device codes/scales -> combine; the host no longer multiplies the routing weight or quantises the operand"
    );

    // ---------------------------------------------------------------------------------------
    // Stage 4: STORED-layout down projection + combine, one launch per rank into a caller-zeroed
    // f32 [tokens*top_k][hidden] slot accumulator.
    // ---------------------------------------------------------------------------------------
    println!(
        "--- stage 4: b12x_mxfp4_moe_stored_sm121 down/combine, one launch per rank into a caller-zeroed f32 [tokens*top_k][hidden] slot accumulator ---"
    );
    println!(
        "comparison class (declared before any number): rank f32 slot accumulator BIT-EXACT against mxfp4_ref::moe_combine_rank_stored over the SAME stage-3 device operand bytes, the same stored w2 rank slices (288 code bytes + 18 scale bytes per row, no padding) and the same ids; a second chain from a freshly zeroed accumulator byte-identical. The activation operand is stage 3's device codes/scales read back byte for byte, not a host-built tensor."
    );
    let stored_cubin = pinned_cubin(
        &args.cubin_dir,
        sabi::KERNEL,
        sabi::CUBIN_BYTES,
        sabi::CUBIN_SHA256,
    )?;
    let (stored_module, stored_kernel) = stored_launch::load(&cuda, &stored_cubin)?;
    let stored_codes: Vec<Vec<u8>> = (0..RANKS)
        .map(|rank| {
            rank_stored_rows(
                &expert_table,
                |expert| &expert.w2,
                HIDDEN,
                IN_PACKED,
                rank * STORED_CODE_ROW,
                STORED_CODE_ROW,
            )
        })
        .collect();
    let stored_scales: Vec<Vec<u8>> = (0..RANKS)
        .map(|rank| {
            rank_stored_rows(
                &expert_table,
                |expert| &expert.s2,
                HIDDEN,
                IN_SCALES,
                rank * STORED_SCALE_ROW,
                STORED_SCALE_ROW,
            )
        })
        .collect();
    println!(
        "stored w2 rank slices: 4 ranks x ({} code + {} scale bytes per expert) from the released w2 tensors, per-row prefixes at offset r*{STORED_CODE_ROW} / r*{STORED_SCALE_ROW}",
        stored_codes[0].len(),
        stored_scales[0].len()
    );
    let mut device_acc: Vec<Vec<f32>> = Vec::with_capacity(RANKS);
    let mut twin_acc: Vec<Vec<f32>> = Vec::with_capacity(RANKS);
    for rank in 0..RANKS {
        let mut expected = vec![0.0f32; rows * HIDDEN];
        let _ = mxfp4_ref::moe_combine_rank_stored(
            &mut expected,
            &device_codes[rank],
            &device_scales[rank],
            &stored_codes[rank],
            &stored_scales[rank],
            &local_ids,
            args.tokens,
            HIDDEN,
            local_experts,
        );
        let mut rank_allocated = 0usize;
        let act_d = cuda.alloc(device_codes[rank].len(), &mut rank_allocated)?;
        let acts_d = cuda.alloc(device_scales[rank].len(), &mut rank_allocated)?;
        let w_d = cuda.alloc(stored_codes[rank].len(), &mut rank_allocated)?;
        let ws_d = cuda.alloc(stored_scales[rank].len(), &mut rank_allocated)?;
        let ids_d = cuda.alloc(local_ids.len() * 4, &mut rank_allocated)?;
        let acc_d = cuda.alloc(rows * HIDDEN * 4, &mut rank_allocated)?;
        println!(
            "  rank {rank}: grid ({}, 160, 1), block (256, 1, 1), 10 argument words, device bytes admitted {rank_allocated}",
            args.tokens
        );
        cuda.copy_htod(&act_d, &device_codes[rank])?;
        cuda.copy_htod(&acts_d, &device_scales[rank])?;
        cuda.copy_htod(&w_d, &stored_codes[rank])?;
        cuda.copy_htod(&ws_d, &stored_scales[rank])?;
        cuda.copy_htod(&ids_d, &u32s_to_bytes(&local_ids))?;
        let zero = f32s_to_bytes(&vec![0.0f32; rows * HIDDEN]);
        cuda.copy_htod(&acc_d, &zero)?;
        let params = Mxfp4MoeStoredParams::new(
            act_d,
            acts_d,
            w_d,
            ws_d,
            ids_d,
            acc_d,
            args.tokens as u32,
            0,
            args.layer as u32,
            rank as u64,
        );
        check_floor(
            args.floor_gib,
            &format!("m26 stage 4 rank {rank} pre-launch"),
        )?;
        stored_launch::launch(&cuda, stored_kernel, &params)?;
        let mut acc_first = vec![0u8; rows * HIDDEN * 4];
        cuda.copy_dtoh(&mut acc_first, &acc_d)?;
        // The second chain starts from a freshly zeroed accumulator: the kernel ADDS, so a stale
        // accumulator would double-count and this check would see it.
        cuda.copy_htod(&acc_d, &zero)?;
        stored_launch::launch(&cuda, stored_kernel, &params)?;
        let mut acc_second = vec![0u8; rows * HIDDEN * 4];
        cuda.copy_dtoh(&mut acc_second, &acc_d)?;
        check_floor(
            args.floor_gib,
            &format!("m26 stage 4 rank {rank} post-launch"),
        )?;
        cuda.free_buffers(&[&act_d, &acts_d, &w_d, &ws_d, &ids_d, &acc_d]);
        let first = f32_bytes_vec(&acc_first);
        let mismatches = first
            .iter()
            .zip(&expected)
            .filter(|(device, twin)| device.to_bits() != twin.to_bits())
            .count();
        Verdict::push(
            &mut verdicts,
            &format!("stage 4 rank {rank}: slot accumulator bit-exact vs moe_combine_rank_stored"),
            mismatches == 0,
            format!("{mismatches} mismatched f32 partials of {}", first.len()),
        );
        Verdict::push(
            &mut verdicts,
            &format!("stage 4 rank {rank}: second chain from a zeroed accumulator byte-identical"),
            acc_first == acc_second,
            format!("{} f32 bytes compared", acc_first.len()),
        );
        device_acc.push(first);
        twin_acc.push(expected);
    }
    cuda.free(stored_module, &[]);
    println!(
        "stage 4 output: 4 rank slot accumulators, {rows} routed rows x {HIDDEN} f32 each; no bf16 rounding in the kernel (native rounds each expert output once, in the epilogue)"
    );

    // The native full-K reference: the incumbent 640-wide padded combine chain (real 576 + zero
    // padding), summed in f32, then the same host epilogue rule. This is the approximate reference.
    let w2_padded: Vec<Vec<u8>> = (0..RANKS)
        .map(|rank| {
            let mut out = Vec::new();
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
    let ws2_padded: Vec<Vec<u8>> = (0..RANKS)
        .map(|rank| {
            let mut out = Vec::new();
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
    let mut native_acc = vec![0.0f32; rows * HIDDEN];
    for rank in 0..RANKS {
        let _ = mxfp4_ref::moe_combine_rank(
            &mut native_acc,
            &device_codes[rank],
            &device_scales[rank],
            &w2_padded[rank],
            &ws2_padded[rank],
            &local_ids,
            args.tokens,
            HIDDEN,
            local_experts,
        );
    }
    let native_f32 =
        mxfp4_ref::moe_combine_routed_f32(&native_acc, &local_ids, args.tokens, HIDDEN);
    println!(
        "native reference: the incumbent 640-padded moe_combine_rank chain over the same device operand bytes, summed in f32 in rank order (the only approximate reference)"
    );

    // ---------------------------------------------------------------------------------------
    // Stage 5: host rank finalise + canonical bf16 all-reduce payload rule.
    // ---------------------------------------------------------------------------------------
    println!(
        "--- stage 5: rank finalise (moe_combine_routed_f32) + canonical reduce (bf16 payload, ascending-rank f32 sum from 0.0, one bf16 RNE round) ---"
    );
    let mut device_rank_f32: Vec<Vec<f32>> = Vec::with_capacity(RANKS);
    let mut twin_rank_f32: Vec<Vec<f32>> = Vec::with_capacity(RANKS);
    for rank in 0..RANKS {
        device_rank_f32.push(mxfp4_ref::moe_combine_routed_f32(
            &device_acc[rank],
            &local_ids,
            args.tokens,
            HIDDEN,
        ));
        twin_rank_f32.push(mxfp4_ref::moe_combine_routed_f32(
            &twin_acc[rank],
            &local_ids,
            args.tokens,
            HIDDEN,
        ));
    }
    let device_payloads: Vec<Vec<u16>> = device_rank_f32
        .iter()
        .map(|values| values.iter().map(|value| bf16_bits(*value)).collect())
        .collect();
    let twin_payloads: Vec<Vec<u16>> = twin_rank_f32
        .iter()
        .map(|values| values.iter().map(|value| bf16_bits(*value)).collect())
        .collect();
    let reduced = canonical_reduce(&device_payloads)?;
    let twin_reduced = canonical_reduce(&twin_payloads)?;
    let payload_mismatches = device_payloads
        .iter()
        .zip(&twin_payloads)
        .map(|(device, twin)| device.iter().zip(twin).filter(|(a, b)| a != b).count())
        .sum::<usize>();
    let reduced_mismatches = reduced
        .iter()
        .zip(&twin_reduced)
        .filter(|(a, b)| a != b)
        .count();
    Verdict::push(
        &mut verdicts,
        "stage 5: four rank bf16 payloads bit-exact vs the canonical twin",
        payload_mismatches == 0,
        format!(
            "{payload_mismatches} mismatched bf16 values of {}",
            RANKS * args.tokens * HIDDEN
        ),
    );
    Verdict::push(
        &mut verdicts,
        "stage 5: reduced routed delta bit-exact vs the canonical reduce of the twin payloads",
        reduced_mismatches == 0,
        format!(
            "{reduced_mismatches} mismatched bf16 values of {}",
            args.tokens * HIDDEN
        ),
    );
    let delta_f32: Vec<f32> = reduced.iter().map(|bits| bf16_value(*bits)).collect();
    let native_l2 = relative_l2(&delta_f32, &native_f32);
    println!(
        "stage 5 output: {RANKS} rank payloads of {} bf16 values -> one routed delta of {} bf16 values; canonical twin mismatches {reduced_mismatches}",
        args.tokens * HIDDEN,
        args.tokens * HIDDEN
    );
    Verdict::push(
        &mut verdicts,
        "stage 6: routed delta vs native full-K reference relative L2 <= 5.0e-3",
        native_l2 <= NATIVE_L2_BAR,
        format!("relative L2 {native_l2:e} (bar {NATIVE_L2_BAR:e})"),
    );
    if let Some(path) = &args.dump {
        let bytes = bf16_bits_to_bytes(&reduced);
        std::fs::write(path, &bytes)
            .map_err(|error| format!("dump {}: {error}", path.display()))?;
        println!(
            "stage 5 dump: {} bytes ({} routed-delta bf16 codes) written to {}",
            bytes.len(),
            reduced.len(),
            path.display()
        );
    }
    if args.repeat > 0 {
        println!(
            "repeat requested ({}) but m26 makes no timing claim; every device stage already runs each launch twice for determinism, and the launch count is structure, not a measured cost",
            args.repeat
        );
    }

    // ---------------------------------------------------------------------------------------
    // Evidence record: the measured side of docs/archive/build/moe-stage-contract.md section 4.
    // ---------------------------------------------------------------------------------------
    let cubin_pins = json!([
        {"name": "moe_router_sm121", "cubin_bytes": rabi::CUBIN_BYTES, "cubin_sha256": rabi::CUBIN_SHA256},
        {"name": "routed_gate_up_sm121", "cubin_bytes": gabi::CUBIN_BYTES, "cubin_sha256": gabi::CUBIN_SHA256},
        {"name": "swiglu_mxfp8_quant_weighted_sm121", "cubin_bytes": SWIGLU_WEIGHTED_CUBIN_BYTES, "cubin_sha256": SWIGLU_WEIGHTED_CUBIN_SHA256},
        {"name": sabi::KERNEL, "cubin_bytes": sabi::CUBIN_BYTES, "cubin_sha256": sabi::CUBIN_SHA256},
    ]);
    let gate_up: Vec<Value> = (0..RANKS)
        .map(|rank| {
            let ok = stage_ok(&verdicts, &format!("stage 2 rank {rank}"));
            json!({
                "rank": rank,
                "gate_sha256": sha256_hex(&bf16_bits_to_bytes(&gate_bits[rank])),
                "up_sha256": sha256_hex(&bf16_bits_to_bytes(&up_bits[rank])),
                "mismatches_vs_twin": if ok { 0 } else { -1 },
                "padding_nonzero": if ok { 0 } else { -1 },
                "second_launch_byte_identical": ok,
            })
        })
        .collect();
    let swiglu: Vec<Value> = (0..RANKS)
        .map(|rank| {
            let ok = stage_ok(&verdicts, &format!("stage 3 rank {rank}"));
            json!({
                "rank": rank,
                "codes_sha256": sha256_hex(&device_codes[rank]),
                "scales_sha256": sha256_hex(&device_scales[rank]),
                "code_mismatches_vs_twin": if ok { 0 } else { -1 },
                "scale_mismatches_vs_twin": if ok { 0 } else { -1 },
                "real_width_operand_mismatches": if ok { 0 } else { -1 },
                "second_launch_byte_identical": ok,
            })
        })
        .collect();
    let combine: Vec<Value> = (0..RANKS)
        .map(|rank| {
            let ok = stage_ok(&verdicts, &format!("stage 4 rank {rank}"));
            json!({
                "rank": rank,
                "arg_words": 10,
                "grid": [args.tokens, 160, 1],
                "block": [256, 1, 1],
                "slot_acc_f32_values": rows * HIDDEN,
                "slot_acc_sha256": sha256_hex(&f32s_to_bytes(&device_acc[rank])),
                "mismatches_vs_twin": if ok { 0 } else { -1 },
                "second_launch_byte_identical": ok,
            })
        })
        .collect();
    let payloads_by_rank: Vec<String> = device_payloads
        .iter()
        .map(|payload| sha256_hex(&bf16_bits_to_bytes(payload)))
        .collect();
    let rank_routed_by_rank: Vec<String> = device_rank_f32
        .iter()
        .map(|values| sha256_hex(&f32s_to_bytes(values)))
        .collect();
    let evidence = json!({
        "schema": EVIDENCE_SCHEMA,
        "mode": "device",
        "stage": "layer0-moe",
        "layer": args.layer,
        "tokens": args.tokens,
        "ranks": RANKS,
        "experts": EXPERTS,
        "top_k": TOP_K,
        "hidden": HIDDEN,
        "inter_per_rank": WIDTH,
        "inter_padded": WIDTH_PADDED,
        "hc_mult": HC_MULT,
        "selected_rows": rows,
        "launches_per_rank": 5,
        "launches_stage": 5 * RANKS,
        "all_reduce_calls_per_rank": 1,
        "weight_h2d_bytes_per_token": 0,
        "cubin_pins": cubin_pins,
        "input": {
            "bundle": args.input.as_ref().map(|path| path.display().to_string()),
            "pre_ffn_hidden_sha256": sha256_hex(&f32s_to_bytes(&pre_ffn_hidden)),
            "pre_ffn_hidden_bytes": pre_ffn_hidden.len() * 4,
            "act_fp8_sha256": sha256_hex(&act_fp8),
            "act_scales_sha256": sha256_hex(&act_scales),
            "act_fp8_bytes": act_fp8.len(),
            "act_scales_bytes": act_scales.len(),
        },
        "router": {
            "ids": device_ids,
            "ids_sha256": sha256_hex(&u32s_to_bytes(&device_ids)),
            "weights_sha256": sha256_hex(&f32s_to_bytes(&device_weights)),
            "scores_sha256": sha256_hex(&f32s_to_bytes(&device_scores)),
            "scores_max_abs_delta": scores_delta,
            "scores_max_ulp": scores_ulp,
            "weights_max_abs_delta": weights_delta,
            "weights_max_ulp": weights_ulp,
            "second_launch_byte_identical": first_router_bytes == second_router_bytes,
        },
        "gate_up": gate_up,
        "swiglu": swiglu,
        "combine": combine,
        "reduce": {
            "rank_routed_f32_sha256_by_rank": rank_routed_by_rank,
            "payload_sha256_by_rank": payloads_by_rank,
            "routed_delta_sha256": sha256_hex(&bf16_bits_to_bytes(&reduced)),
            "routed_delta_bytes": reduced.len() * 2,
            "rule": "bf16 payload, ascending-rank f32 sum from 0.0, one bf16 RNE round",
            "mismatches_vs_canonical_twin": reduced_mismatches,
            "native_relative_l2": native_l2,
            "native_relative_l2_bar": NATIVE_L2_BAR,
        },
        "epilogue": {
            "ran": false,
            "note": "hc_ffn mixes, shared expert and ffn hc_post are the four-rank loop's host seams (engine code); this harness stops at the routed delta",
        },
    });
    println!(
        "evidence record built: mode device, routed delta sha256 {}",
        sha256_hex(&bf16_bits_to_bytes(&reduced))
    );
    Ok(StageOutcome { verdicts, evidence })
}

/// Whether every recorded verdict whose name starts with the prefix passed.
fn stage_ok(verdicts: &[Verdict], prefix: &str) -> bool {
    let matching: Vec<&Verdict> = verdicts
        .iter()
        .filter(|verdict| verdict.name.starts_with(prefix))
        .collect();
    !matching.is_empty() && matching.iter().all(|verdict| verdict.ok)
}

/// The number of recorded verdicts whose name starts with the prefix.
fn stage_checks(verdicts: &[Verdict], prefix: &str) -> usize {
    verdicts
        .iter()
        .filter(|verdict| verdict.name.starts_with(prefix))
        .count()
}

/// Lowercase hex sha256 of a byte payload.
fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest
        .iter()
        .fold(String::with_capacity(64), |mut text, byte| {
            use std::fmt::Write as _;
            let _ = write!(text, "{byte:02x}");
            text
        })
}

/// The canonical all-reduce rule, elementwise: bf16 payloads widened to f32, summed in the given
/// (ascending rank) order from 0.0, one bf16 round-to-nearest-even of the total.
fn canonical_reduce(payloads: &[Vec<u16>]) -> Result<Vec<u16>, String> {
    if payloads.is_empty() {
        return Err("the canonical reduce needs at least one rank payload".to_owned());
    }
    let len = payloads[0].len();
    if let Some(bad) = payloads.iter().find(|payload| payload.len() != len) {
        return Err(format!(
            "canonical reduce payload lengths differ: {len} vs {}",
            bad.len()
        ));
    }
    let mut out = Vec::with_capacity(len);
    for index in 0..len {
        let mut sum = 0.0f32;
        for payload in payloads {
            sum += bf16_value(payload[index]);
        }
        out.push(bf16_bits(sum));
    }
    Ok(out)
}

/// Relative L2 of the candidate against the reference: sqrt(sum((a-b)^2) / sum(b^2)).
fn relative_l2(candidate: &[f32], reference: &[f32]) -> f64 {
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for (value, native) in candidate.iter().zip(reference) {
        let difference = f64::from(*value) - f64::from(*native);
        numerator += difference * difference;
        denominator += f64::from(*native) * f64::from(*native);
    }
    if denominator == 0.0 {
        return if numerator == 0.0 { 0.0 } else { f64::INFINITY };
    }
    (numerator / denominator).sqrt()
}

/// One rank's stored w2 rows: for each staged expert, the per-row take-byte prefix at offset
/// in every row (no padding, no interleaving - the stored layout's two separate tensors).
fn rank_stored_rows<F>(
    experts: &[ExpertTensors],
    pick: F,
    rows: usize,
    stride: usize,
    offset: usize,
    take: usize,
) -> Vec<u8>
where
    F: Fn(&ExpertTensors) -> &[u8],
{
    let mut out = Vec::with_capacity(experts.len() * rows * take);
    for expert in experts {
        let bytes = pick(expert);
        for row in 0..rows {
            let start = row * stride + offset;
            out.extend_from_slice(&bytes[start..start + take]);
        }
    }
    out
}

/// The rows an input bundle or the released-embed stand-in supplies.
type InputRows = (Vec<f32>, Vec<f32>, Vec<u8>, Option<(Vec<u8>, Vec<u8>)>);

/// The pre-FFN hidden rows this harness runs on: the expanded bundle's pre_ffn_hidden.f32 (the
/// loop's H3 host-seam bytes) or the declared released-embed stand-in. Returns
/// (pre_ffn_hidden f32, BF16-rounded router rows f32, router input bytes BF16, optional act bytes).
fn load_input_any(snap: &mut Snapshot, args: &Args) -> Result<InputRows, String> {
    match &args.input {
        Some(dir) => {
            let path = dir.join("pre_ffn_hidden.f32");
            let bytes = std::fs::read(&path)
                .map_err(|error| format!("read {}: {error}", path.display()))?;
            let expected = args.tokens * HIDDEN * 4;
            if bytes.len() != expected {
                return Err(format!(
                    "bundle {} is {} bytes, expected {expected} ({} tokens x {HIDDEN} f32)",
                    path.display(),
                    bytes.len(),
                    args.tokens
                ));
            }
            let pre_ffn_hidden = f32_bytes_vec(&bytes);
            let router_x_f32: Vec<f32> = pre_ffn_hidden
                .iter()
                .map(|value| bf16_value(bf16_bits(*value)))
                .collect();
            let x_bytes: Vec<u8> = pre_ffn_hidden
                .iter()
                .flat_map(|value| bf16_bits(*value).to_le_bytes())
                .collect();
            let codes_path = dir.join("act_fp8.bin");
            let scales_path = dir.join("act_scales.bin");
            let bundle_act = if codes_path.exists() && scales_path.exists() {
                let codes = std::fs::read(&codes_path)
                    .map_err(|error| format!("read {}: {error}", codes_path.display()))?;
                let scales = std::fs::read(&scales_path)
                    .map_err(|error| format!("read {}: {error}", scales_path.display()))?;
                Some((codes, scales))
            } else {
                None
            };
            Ok((pre_ffn_hidden, router_x_f32, x_bytes, bundle_act))
        }
        None => {
            let bytes = read_embed_rows(snap, args.tokens)?;
            let pre_ffn_hidden = bf16_bytes_to_f32(&bytes);
            Ok((pre_ffn_hidden.clone(), pre_ffn_hidden, bytes, None))
        }
    }
}

/// Prints the stage PASS lines, writes the evidence record and the final verdict line.
fn finish(args: &Args, outcome: StageOutcome) -> Result<(), String> {
    let mut evidence = outcome.evidence;
    let failed: Vec<String> = outcome
        .verdicts
        .iter()
        .filter(|verdict| !verdict.ok)
        .map(|verdict| verdict.name.clone())
        .collect();
    let mut pass_lines: Vec<String> = Vec::new();
    let stage_line = |label: String, prefix: &str, pass_lines: &mut Vec<String>| {
        let checks = stage_checks(&outcome.verdicts, prefix);
        let ok = stage_ok(&outcome.verdicts, prefix);
        let line = format!(
            "m26 {label}: {} ({checks} checks)",
            if ok { "PASS" } else { "FAIL" }
        );
        println!("{line}");
        pass_lines.push(line);
    };
    stage_line("stage 1 router".to_owned(), "stage 1", &mut pass_lines);
    for rank in 0..RANKS {
        stage_line(
            format!("stage 2 gate/up rank {rank}"),
            &format!("stage 2 rank {rank}:"),
            &mut pass_lines,
        );
        stage_line(
            format!("stage 3 swiglu rank {rank}"),
            &format!("stage 3 rank {rank}:"),
            &mut pass_lines,
        );
        stage_line(
            format!("stage 4 stored combine rank {rank}"),
            &format!("stage 4 rank {rank}:"),
            &mut pass_lines,
        );
    }
    stage_line(
        "stage 5 canonical reduce".to_owned(),
        "stage 5:",
        &mut pass_lines,
    );
    stage_line(
        "stage 6 native reference".to_owned(),
        "stage 6:",
        &mut pass_lines,
    );
    println!("--- NOT claimed by m26 ---");
    println!(
        "not a chain/token/per-layer claim, not layer text (the pre-FFN rows are the loop's host-seam bytes or released embed rows), not the shared expert or the shared-add epilogue (engine host seam, the loop's H5), not engine integration, not host-time saved, not a performance claim. The routed delta this harness records ends at the canonical bf16 all-reduce; hc_post is the loop's H6 host seam."
    );
    let verdict = if failed.is_empty() { "PASS" } else { "FAIL" };
    evidence["verdict"] = json!(verdict);
    evidence["failed_checks"] = json!(failed);
    evidence["pass_lines"] = json!(pass_lines);
    if let Some(path) = &args.evidence {
        let text = serde_json::to_string_pretty(&evidence)
            .map_err(|error| format!("serialise evidence: {error}"))?;
        std::fs::write(path, format!("{text}\n"))
            .map_err(|error| format!("write {}: {error}", path.display()))?;
        println!("m26_moe_stage: evidence {}", path.display());
    }
    if failed.is_empty() {
        println!(
            "m26_moe_stage: PASS ({} checks; layer {}; tokens {}; 4 rank shards; 5 launches per rank; four pinned cubins)",
            outcome.verdicts.len(),
            args.layer,
            args.tokens
        );
        Ok(())
    } else {
        let details: Vec<String> = outcome
            .verdicts
            .iter()
            .filter(|verdict| !verdict.ok)
            .map(|verdict| format!("{}: {}", verdict.name, verdict.detail))
            .collect();
        Err(format!(
            "m26_moe_stage: FAIL at {} of {} checks: {:?}",
            failed.len(),
            outcome.verdicts.len(),
            details
        ))
    }
}

/// The CPU twin run: no driver, no allocation; prints and records the EXPECTED evidence record.
fn run_twin_only(args: &Args) -> Result<StageOutcome, String> {
    println!(
        "=== m26: layer-0 MoE stage host twin (CPU, no driver) - router, gate/up, weighted SwiGLU+quant, stored down combine, canonical reduce ==="
    );
    println!(
        "geometry: hidden {HIDDEN}, intermediate {INTER} = {RANKS} rank slices of {WIDTH} (padded {WIDTH_PADDED}), routed experts {EXPERTS}, top-k {TOP_K}; tokens {}; hc_mult {HC_MULT}",
        args.tokens
    );
    println!(
        "what this is: the CPU twin of documented stages 1-6 over the same input bundle or released-embed stand-in; it records the EXPECTED digests and the kernel-twin comparisons, and the device run (without --twin-only) records the measured ones"
    );
    check_floor(args.floor_gib, "m26 twin start")?;
    let mut verdicts: Vec<Verdict> = Vec::new();
    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());
    let prefix = format!("layers.{}", args.layer);
    let gate_bytes = snap.read_raw(
        &format!("{prefix}.ffn.gate.weight"),
        "BF16",
        &[EXPERTS, HIDDEN],
    )?;
    let bias_bytes = snap.read_raw(&format!("{prefix}.ffn.gate.bias"), "F32", &[EXPERTS])?;
    let gate_f32 = bf16_bytes_to_f32(&gate_bytes);
    let bias_f32 = f32_bytes_vec(&bias_bytes);
    let (pre_ffn_hidden, router_x_f32, _x_bytes, bundle_act) = load_input_any(&mut snap, args)?;

    // H4: the D8 activation operand the gate/up kernels consume.
    let mut act_fp8 = Vec::with_capacity(args.tokens * HIDDEN);
    let mut act_scales = Vec::with_capacity(args.tokens * SCALE_ROW);
    for token in 0..args.tokens {
        let row = &pre_ffn_hidden[token * HIDDEN..(token + 1) * HIDDEN];
        let (codes, scales) = act_quant_fp8_row(row);
        act_fp8.extend_from_slice(&codes);
        act_scales.extend_from_slice(&scales);
    }
    if let Some((codes, scales)) = &bundle_act {
        let matches = codes == &act_fp8 && scales == &act_scales;
        Verdict::push(
            &mut verdicts,
            "input: bundle act bytes bit-exact vs the H4 quantise rule",
            matches,
            format!(
                "{} code + {} scale bytes compared",
                act_fp8.len(),
                act_scales.len()
            ),
        );
    }

    // Stage 1 twin: router_ref over the BF16-rounded rows.
    let mut twin_ids: Vec<u32> = Vec::with_capacity(args.tokens * TOP_K);
    let mut twin_weights: Vec<f32> = Vec::with_capacity(args.tokens * TOP_K);
    let mut twin_scores: Vec<f32> = Vec::with_capacity(args.tokens * EXPERTS);
    for token in 0..args.tokens {
        let x = &router_x_f32[token * HIDDEN..(token + 1) * HIDDEN];
        let out = router_ref::route_f32(x, &gate_f32, &bias_f32, RouterParams::RELEASED)?;
        twin_ids.extend_from_slice(&out.ids);
        twin_weights.extend_from_slice(&out.weights);
        twin_scores.extend_from_slice(&out.scores);
    }
    Verdict::push(
        &mut verdicts,
        "stage 1 twin: router selections and scores computed",
        true,
        format!(
            "{} ids, {} weights, {} scores",
            twin_ids.len(),
            twin_weights.len(),
            twin_scores.len()
        ),
    );
    println!(
        "twin router: {} routed rows, {} distinct experts",
        twin_ids.len(),
        unique_experts(&twin_ids).len()
    );

    // Stage the selected experts and the two operands once.
    let local_of = local_expert_table(&twin_ids);
    let mut local_ids: Vec<u32> = Vec::with_capacity(twin_ids.len());
    let mut local_experts = 0usize;
    for id in &twin_ids {
        let index = local_of[*id as usize];
        if index == usize::MAX {
            return Err(format!("twin id {id} is not in the routed table"));
        }
        if index + 1 > local_experts {
            local_experts = index + 1;
        }
        local_ids.push(index as u32);
    }
    let mut table_rows: Vec<(usize, usize)> = local_of
        .iter()
        .enumerate()
        .filter(|(_, index)| **index != usize::MAX)
        .map(|(expert, index)| (*index, expert))
        .collect();
    table_rows.sort_unstable();
    let expert_table = read_expert_tensors(&mut snap, &prefix, &table_rows)?;
    let rows = args.tokens * TOP_K;
    let out_values = rows * WIDTH_PADDED;

    // Twin stages 2-3 per rank.
    let mut gate_bits: Vec<Vec<u16>> = Vec::with_capacity(RANKS);
    let mut up_bits: Vec<Vec<u16>> = Vec::with_capacity(RANKS);
    let mut codes_ranks: Vec<Vec<u8>> = Vec::with_capacity(RANKS);
    let mut scales_ranks: Vec<Vec<u8>> = Vec::with_capacity(RANKS);
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
        Verdict::push(
            &mut verdicts,
            &format!("stage 2 rank {rank}: twin gate/up produced"),
            twin_gate.len() == out_values && twin_up.len() == out_values,
            format!(
                "{} gate + {} up bf16 values",
                twin_gate.len(),
                twin_up.len()
            ),
        );
        let mut codes = vec![0u8; rows * WIDTH_PADDED];
        let mut scales = vec![0u8; rows * MOE_SCALE_ROW];
        let mut model_mismatches = 0usize;
        for row in 0..rows {
            let weight = twin_weights[row];
            let gate_f32_row = bf16_bits_slice_to_f32(&twin_gate, row * WIDTH_PADDED, WIDTH_PADDED);
            let up_f32_row = bf16_bits_slice_to_f32(&twin_up, row * WIDTH_PADDED, WIDTH_PADDED);
            let (row_codes, row_scales) =
                swiglu_ref::quantise_row_weighted(&gate_f32_row, &up_f32_row, SWIGLU_LIMIT, weight);
            codes[row * WIDTH_PADDED..(row + 1) * WIDTH_PADDED].copy_from_slice(&row_codes);
            scales[row * MOE_SCALE_ROW..(row + 1) * MOE_SCALE_ROW].copy_from_slice(&row_scales);
            let real: Vec<f32> = gate_f32_row[..WIDTH]
                .iter()
                .zip(&up_f32_row[..WIDTH])
                .map(|(gate, up)| swiglu_ref::swiglu(*gate, *up, SWIGLU_LIMIT))
                .collect();
            let (real_codes, real_scales) = mxfp4_ref::routed_activation_operand(&real, weight);
            for column in 0..WIDTH {
                if codes[row * WIDTH_PADDED + column] != real_codes[column] {
                    model_mismatches += 1;
                }
            }
            for block in 0..(WIDTH / QUANT_BLOCK) {
                if scales[row * MOE_SCALE_ROW + block] != real_scales[block] {
                    model_mismatches += 1;
                }
            }
        }
        Verdict::push(
            &mut verdicts,
            &format!("stage 3 rank {rank}: twin weighted operand built"),
            codes.len() == rows * WIDTH_PADDED && scales.len() == rows * MOE_SCALE_ROW,
            format!("{} code + {} scale bytes", codes.len(), scales.len()),
        );
        Verdict::push(
            &mut verdicts,
            &format!(
                "stage 3 rank {rank}: real {WIDTH}-wide operand bit-exact vs routed_activation_operand"
            ),
            model_mismatches == 0,
            format!(
                "{model_mismatches} mismatched of {} code+scale bytes",
                rows * (WIDTH + WIDTH / QUANT_BLOCK)
            ),
        );
        gate_bits.push(twin_gate);
        up_bits.push(twin_up);
        codes_ranks.push(codes);
        scales_ranks.push(scales);
    }

    // Stage 4 twin: stored layout, two separate tensors, no padding.
    let stored_codes: Vec<Vec<u8>> = (0..RANKS)
        .map(|rank| {
            rank_stored_rows(
                &expert_table,
                |expert| &expert.w2,
                HIDDEN,
                IN_PACKED,
                rank * STORED_CODE_ROW,
                STORED_CODE_ROW,
            )
        })
        .collect();
    let stored_scales: Vec<Vec<u8>> = (0..RANKS)
        .map(|rank| {
            rank_stored_rows(
                &expert_table,
                |expert| &expert.s2,
                HIDDEN,
                IN_SCALES,
                rank * STORED_SCALE_ROW,
                STORED_SCALE_ROW,
            )
        })
        .collect();
    let mut twin_accs: Vec<Vec<f32>> = Vec::with_capacity(RANKS);
    for rank in 0..RANKS {
        let mut acc = vec![0.0f32; rows * HIDDEN];
        let _ = mxfp4_ref::moe_combine_rank_stored(
            &mut acc,
            &codes_ranks[rank],
            &scales_ranks[rank],
            &stored_codes[rank],
            &stored_scales[rank],
            &local_ids,
            args.tokens,
            HIDDEN,
            local_experts,
        );
        Verdict::push(
            &mut verdicts,
            &format!("stage 4 rank {rank}: twin slot accumulator built"),
            acc.len() == rows * HIDDEN,
            format!("{} f32 partials", acc.len()),
        );
        twin_accs.push(acc);
    }
    println!(
        "twin stored w2 rank slices: 4 ranks x ({} code + {} scale bytes per expert)",
        stored_codes[0].len(),
        stored_scales[0].len()
    );

    // The native full-K reference (the only approximate one): the incumbent 640-padded chain.
    let mut native_acc = vec![0.0f32; rows * HIDDEN];
    for rank in 0..RANKS {
        let mut w2 = Vec::new();
        let mut ws2 = Vec::new();
        for expert in &expert_table {
            w2.extend_from_slice(&slice_and_pad_rows(
                &expert.w2,
                HIDDEN,
                IN_PACKED,
                rank * (WIDTH / 2),
                WIDTH / 2,
                MOE_PACKED_ROW,
            ));
            ws2.extend_from_slice(&slice_and_pad_rows(
                &expert.s2,
                HIDDEN,
                IN_SCALES,
                rank * (WIDTH / QUANT_BLOCK),
                WIDTH / QUANT_BLOCK,
                MOE_SCALE_ROW,
            ));
        }
        let _ = mxfp4_ref::moe_combine_rank(
            &mut native_acc,
            &codes_ranks[rank],
            &scales_ranks[rank],
            &w2,
            &ws2,
            &local_ids,
            args.tokens,
            HIDDEN,
            local_experts,
        );
    }
    let native_f32 =
        mxfp4_ref::moe_combine_routed_f32(&native_acc, &local_ids, args.tokens, HIDDEN);

    // Stage 5 twin: canonical reduce over the four twin payloads.
    let mut rank_f32s: Vec<Vec<f32>> = Vec::with_capacity(RANKS);
    let mut payloads: Vec<Vec<u16>> = Vec::with_capacity(RANKS);
    for acc in &twin_accs {
        let values = mxfp4_ref::moe_combine_routed_f32(acc, &local_ids, args.tokens, HIDDEN);
        payloads.push(values.iter().map(|value| bf16_bits(*value)).collect());
        rank_f32s.push(values);
    }
    let reduced = canonical_reduce(&payloads)?;
    let delta_f32: Vec<f32> = reduced.iter().map(|bits| bf16_value(*bits)).collect();
    let native_l2 = relative_l2(&delta_f32, &native_f32);
    Verdict::push(
        &mut verdicts,
        "stage 5: canonical reduce over four twin payloads",
        reduced.len() == args.tokens * HIDDEN,
        format!(
            "{} rank payloads of {} bf16 values -> {} reduced values",
            RANKS,
            args.tokens * HIDDEN,
            reduced.len()
        ),
    );
    Verdict::push(
        &mut verdicts,
        "stage 6: twin delta vs native full-K reference relative L2 <= 5.0e-3",
        native_l2 <= NATIVE_L2_BAR,
        format!("relative L2 {native_l2:e} (bar {NATIVE_L2_BAR:e})"),
    );

    let gate_up: Vec<Value> = (0..RANKS)
        .map(|rank| {
            json!({
                "rank": rank,
                "gate_sha256": sha256_hex(&bf16_bits_to_bytes(&gate_bits[rank])),
                "up_sha256": sha256_hex(&bf16_bits_to_bytes(&up_bits[rank])),
                "mismatches_vs_twin": 0,
                "padding_nonzero": 0,
                "second_launch_byte_identical": true,
            })
        })
        .collect();
    let swiglu: Vec<Value> = (0..RANKS)
        .map(|rank| {
            json!({
                "rank": rank,
                "codes_sha256": sha256_hex(&codes_ranks[rank]),
                "scales_sha256": sha256_hex(&scales_ranks[rank]),
                "code_mismatches_vs_twin": 0,
                "scale_mismatches_vs_twin": 0,
                "real_width_operand_mismatches": 0,
                "second_launch_byte_identical": true,
            })
        })
        .collect();
    let combine: Vec<Value> = (0..RANKS)
        .map(|rank| {
            json!({
                "rank": rank,
                "arg_words": 10,
                "grid": [args.tokens, 160, 1],
                "block": [256, 1, 1],
                "slot_acc_f32_values": rows * HIDDEN,
                "slot_acc_sha256": sha256_hex(&f32s_to_bytes(&twin_accs[rank])),
                "mismatches_vs_twin": 0,
                "second_launch_byte_identical": true,
            })
        })
        .collect();
    let payloads_by_rank: Vec<String> = payloads
        .iter()
        .map(|payload| sha256_hex(&bf16_bits_to_bytes(payload)))
        .collect();
    let rank_routed_by_rank: Vec<String> = rank_f32s
        .iter()
        .map(|values| sha256_hex(&f32s_to_bytes(values)))
        .collect();
    let cubin_pins = json!([
        {"name": "moe_router_sm121", "cubin_bytes": rabi::CUBIN_BYTES, "cubin_sha256": rabi::CUBIN_SHA256},
        {"name": "routed_gate_up_sm121", "cubin_bytes": gabi::CUBIN_BYTES, "cubin_sha256": gabi::CUBIN_SHA256},
        {"name": "swiglu_mxfp8_quant_weighted_sm121", "cubin_bytes": SWIGLU_WEIGHTED_CUBIN_BYTES, "cubin_sha256": SWIGLU_WEIGHTED_CUBIN_SHA256},
        {"name": sabi::KERNEL, "cubin_bytes": sabi::CUBIN_BYTES, "cubin_sha256": sabi::CUBIN_SHA256},
    ]);
    let evidence = json!({
        "schema": EVIDENCE_SCHEMA,
        "mode": "twin",
        "stage": "layer0-moe",
        "layer": args.layer,
        "tokens": args.tokens,
        "ranks": RANKS,
        "experts": EXPERTS,
        "top_k": TOP_K,
        "hidden": HIDDEN,
        "inter_per_rank": WIDTH,
        "inter_padded": WIDTH_PADDED,
        "hc_mult": HC_MULT,
        "selected_rows": rows,
        "launches_per_rank": 5,
        "launches_stage": 5 * RANKS,
        "all_reduce_calls_per_rank": 1,
        "weight_h2d_bytes_per_token": 0,
        "cubin_pins": cubin_pins,
        "input": {
            "bundle": args.input.as_ref().map(|path| path.display().to_string()),
            "pre_ffn_hidden_sha256": sha256_hex(&f32s_to_bytes(&pre_ffn_hidden)),
            "pre_ffn_hidden_bytes": pre_ffn_hidden.len() * 4,
            "act_fp8_sha256": sha256_hex(&act_fp8),
            "act_scales_sha256": sha256_hex(&act_scales),
            "act_fp8_bytes": act_fp8.len(),
            "act_scales_bytes": act_scales.len(),
        },
        "router": {
            "ids": twin_ids,
            "ids_sha256": sha256_hex(&u32s_to_bytes(&twin_ids)),
            "weights_sha256": sha256_hex(&f32s_to_bytes(&twin_weights)),
            "scores_sha256": sha256_hex(&f32s_to_bytes(&twin_scores)),
            "router_weight_source": "twin",
            "note": "the device run's stage-3 twin uses the device router weights in place; this record is the CPU arithmetic reference",
        },
        "gate_up": gate_up,
        "swiglu": swiglu,
        "combine": combine,
        "reduce": {
            "rank_routed_f32_sha256_by_rank": rank_routed_by_rank,
            "payload_sha256_by_rank": payloads_by_rank,
            "routed_delta_sha256": sha256_hex(&bf16_bits_to_bytes(&reduced)),
            "routed_delta_bytes": reduced.len() * 2,
            "rule": "bf16 payload, ascending-rank f32 sum from 0.0, one bf16 RNE round",
            "mismatches_vs_canonical_twin": 0,
            "native_relative_l2": native_l2,
            "native_relative_l2_bar": NATIVE_L2_BAR,
        },
        "epilogue": {
            "ran": false,
            "note": "hc_ffn mixes, shared expert and ffn hc_post are the four-rank loop's host seams (engine code); this twin stops at the routed delta",
        },
    });
    Ok(StageOutcome { verdicts, evidence })
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

/// The six released tensor names of one routed expert, keyed by its GLOBAL released expert id.
///
/// The staging table maps a local id to the released id; only this function builds tensor names, and
/// a window finding showed why that matters: swapping the global id for the local id in a copy left
/// every test green because the twin consumes whatever bytes were staged. The run asserts the names
/// actually read against [expert_tensor_names], and the unit test pins the format.
fn expert_tensor_names(prefix: &str, expert: u32) -> [String; 6] {
    let base = format!("{prefix}.ffn.experts.{expert}");
    [
        format!("{base}.w1.weight"),
        format!("{base}.w1.scale"),
        format!("{base}.w3.weight"),
        format!("{base}.w3.scale"),
        format!("{base}.w2.weight"),
        format!("{base}.w2.scale"),
    ]
}

fn read_expert_tensors(
    snap: &mut Snapshot,
    prefix: &str,
    table: &[(usize, usize)],
) -> Result<Vec<ExpertTensors>, String> {
    let mut experts = Vec::with_capacity(table.len());
    for (local, expert) in table {
        let names = expert_tensor_names(prefix, *expert as u32);
        experts.push(ExpertTensors {
            w1: snap.read_raw(&names[0], "I8", &[INTER, PACKED_ROW])?,
            s1: snap.read_raw(&names[1], "F8_E8M0", &[INTER, SCALE_ROW])?,
            w3: snap.read_raw(&names[2], "I8", &[INTER, PACKED_ROW])?,
            s3: snap.read_raw(&names[3], "F8_E8M0", &[INTER, SCALE_ROW])?,
            w2: snap.read_raw(&names[4], "I8", &[HIDDEN, IN_PACKED])?,
            s2: snap.read_raw(&names[5], "F8_E8M0", &[HIDDEN, IN_SCALES])?,
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
#[cfg(test)]
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

    #[test]
    fn the_real_width_comparison_calls_the_reference_rule() {
        // Review finding: stage 3's class named mxfp4_ref::routed_activation_operand while the model
        // operand was built with an arithmetically equal helper, so the printed record claimed a
        // comparison that never ran. The real-width operand is now built by calling the reference
        // function itself; this pins that against the example source, the way the other examples pin
        // their kernels' text. The source is split before this test so the assertion's own literal
        // cannot satisfy it.
        let source = include_str!("m26_moe_stage.rs");
        let (production, _) = source
            .split_once("fn the_real_width_comparison_calls_the_reference_rule")
            .expect("the source contains this test");
        assert!(
            production.contains("mxfp4_ref::routed_activation_operand(&swiglu_row, weight)"),
            "the real-width model operand must call mxfp4_ref::routed_activation_operand directly"
        );
        assert!(
            !production
                .contains("let (real_codes, real_scales) = swiglu_ref::quantise_row_weighted("),
            "the model operand must not be an equal re-implementation of the reference rule"
        );
    }

    #[test]
    fn the_weighted_epilogue_pin_is_the_baseline_line() {
        // The harness names the weighted epilogue's bytes itself; the embedded baseline is the single
        // source of record. Comparing them here means a baseline re-pin cannot leave this harness
        // loading the previous cubin (and the window refusing it by size/sha at run time) unnoticed.
        let pin = sparkflash_kernels::baseline::pin("swiglu_mxfp8_quant_weighted_sm121")
            .expect("the embedded baseline must parse")
            .expect("the weighted epilogue must be pinned");
        assert_eq!(pin.cubin_bytes, SWIGLU_WEIGHTED_CUBIN_BYTES);
        assert_eq!(pin.cubin_sha256, SWIGLU_WEIGHTED_CUBIN_SHA256);
        assert_eq!(pin.ptx_membar_lines, 0);
    }

    #[test]
    fn the_staged_expert_names_use_the_global_released_id() {
        let names = expert_tensor_names("layers.0", 250);
        for name in &names {
            assert!(
                name.starts_with("layers.0.ffn.experts.250."),
                "{name}: the staged name must be the global released id"
            );
        }
        let suffixes: Vec<&str> = names
            .iter()
            .map(|name| name.rsplit_once(".ffn.experts.250.").unwrap().1)
            .collect();
        assert_eq!(
            suffixes,
            [
                "w1.weight",
                "w1.scale",
                "w3.weight",
                "w3.scale",
                "w2.weight",
                "w2.scale"
            ]
        );
        // A local id must never appear in a staged name.
        assert!(!names.iter().any(|name| name.contains(".ffn.experts.3.")));
    }
}

#[cfg(test)]
mod m26_tests {
    use super::*;

    #[test]
    fn the_canonical_reduce_sums_ascending_then_rounds_once() {
        let a = vec![bf16_bits(1.0), bf16_bits(0.1), bf16_bits(-2.0)];
        let b = vec![bf16_bits(2.0), bf16_bits(0.2), bf16_bits(0.5)];
        let c = vec![bf16_bits(3.0), bf16_bits(0.3), bf16_bits(1.5)];
        let reduced = canonical_reduce(&[a.clone(), b.clone(), c.clone()]).expect("reduce");
        for index in 0..3 {
            let mut sum = 0.0f32;
            for payload in [&a, &b, &c] {
                sum += bf16_value(payload[index]);
            }
            assert_eq!(reduced[index], bf16_bits(sum), "element {index}");
        }
        assert_eq!(reduced.len(), 3);
        assert!(canonical_reduce(&[]).is_err(), "no payloads is a refusal");
        assert!(
            canonical_reduce(&[vec![0u16; 3], vec![0u16; 4]]).is_err(),
            "mixed payload lengths are a refusal"
        );
    }

    #[test]
    fn the_stored_rank_rows_are_the_prefix_not_a_window() {
        let rows = 3usize;
        let experts = 1usize;
        let bytes: Vec<u8> = (0..experts * rows * IN_PACKED)
            .map(|index| (index % 251) as u8)
            .collect();
        let expert = vec![ExpertTensors {
            w1: vec![],
            s1: vec![],
            w3: vec![],
            s3: vec![],
            w2: bytes,
            s2: vec![0u8; experts * rows * IN_SCALES],
        }];
        let codes = rank_stored_rows(
            &expert,
            |expert| &expert.w2,
            rows,
            IN_PACKED,
            2 * STORED_CODE_ROW,
            STORED_CODE_ROW,
        );
        assert_eq!(codes.len(), experts * rows * STORED_CODE_ROW);
        for (expert_index, _) in (0..experts).enumerate() {
            for row in 0..rows {
                for at in 0..STORED_CODE_ROW {
                    let source = expert_index * rows * IN_PACKED
                        + row * IN_PACKED
                        + 2 * STORED_CODE_ROW
                        + at;
                    assert_eq!(
                        codes[(expert_index * rows + row) * STORED_CODE_ROW + at],
                        (source % 251) as u8,
                        "expert {expert_index} row {row} at {at}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_stored_combine_pin_is_the_baseline_line() {
        let pin = sparkflash_kernels::baseline::pin(sabi::KERNEL)
            .expect("the embedded baseline must parse")
            .expect("the stored combine must be pinned");
        assert_eq!(pin.cubin_bytes, sabi::CUBIN_BYTES);
        assert_eq!(pin.cubin_sha256, sabi::CUBIN_SHA256);
        assert_eq!(pin.ptx_membar_lines, 0);
    }

    #[test]
    fn the_native_l2_bar_and_relative_l2_are_the_documented_rule() {
        assert_eq!(NATIVE_L2_BAR, 5.0e-3);
        assert_eq!(relative_l2(&[0.0, 0.0], &[1.0, 2.0]), 1.0);
        assert_eq!(relative_l2(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    }
}

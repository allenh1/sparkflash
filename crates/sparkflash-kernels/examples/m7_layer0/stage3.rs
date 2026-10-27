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

use super::*;
use crate::stage2::*;

// ---------------------------------------------------------------------------------------------
// Stage 3: the MoE block on a bounded expert subset. The router is a HOST SEAM and is labelled as
// one; the device side is the W4A8 down/combine projection, four serial launches accumulating one
// f32 partial buffer over the 2304-wide intermediate; gate/up, SwiGLU and the activation quantiser
// are host arithmetic here
// because no device kernel in the six-cubin set reduces K=5120 with packed FP4 weights.
// ---------------------------------------------------------------------------------------------

/// text_config.n_routed_experts.
pub(crate) const MOE_EXPERTS: usize = 384;
/// text_config.num_experts_per_tok, mirrored by mxfp4_ref::MOE_TOP_K.
pub(crate) const MOE_TOP_K: usize = 6;
/// text_config.moe_intermediate_size.
pub(crate) const MOE_INTERMEDIATE: usize = 2304;
/// The per-rank expert width and the padded width the kernel actually walks.
pub(crate) const MOE_WIDTH: usize = mxfp4_ref::MOE_WIDTH;
pub(crate) const MOE_WIDTH_PADDED: usize = mxfp4_ref::MOE_WIDTH_PADDED;
/// 2304 / 576: the number of serial combine launches one routed sum needs.
pub(crate) const MOE_RANKS: usize = MOE_INTERMEDIATE / MOE_WIDTH;
/// The pre-padded weight strides: 640 values are 320 packed bytes and 20 ue8m0 bytes.
pub(crate) const MOE_PACKED_ROW: usize = mxfp4_ref::MOE_WEIGHT_BYTES_PER_ROW;
pub(crate) const MOE_SCALE_ROW: usize = mxfp4_ref::MOE_SCALE_BYTES_PER_ROW;
/// One expert's stored w2 row and scale row, before the rank slice: 2304 values.
pub(crate) const MOE_IN_PACKED: usize = MOE_INTERMEDIATE / 2;
pub(crate) const MOE_IN_SCALES: usize = MOE_INTERMEDIATE / 32;
/// One expert's stored w1/w3 row: 5120 packed values and 160 scales.
pub(crate) const MOE_EXPERT_IN_PACKED: usize = HIDDEN / 2;
pub(crate) const MOE_EXPERT_IN_SCALES: usize = HIDDEN / 32;
/// The router's constants: gate_temp is absent from the released config and defaults to 1.0
/// (model.py:67-68), route_scale is text_config.routed_scaling_factor, and the normalising epsilon
/// is model.py:824's.
const GATE_TEMP: f32 = 1.0;
const ROUTE_SCALE: f32 = 1.5;
const ROUTE_NORM_EPS: f32 = 1e-20;
/// The kernel's launch geometry: 32 output columns x 8 parallel K groups, 256 threads, 160 tiles of
/// 5120 (the retired old-ABI row still points at the current symbol, so it must use its tiling).
pub(crate) const MOE_TILE_D: u32 = 32;
pub(crate) const MOE_THREADS: u32 = 256;
/// The stage-3 pass bar, printed before the numbers it judges.
const STAGE3_EXACT_BAR: f64 = 99.9;

const _: () =
    assert!(MOE_EXPERTS == 384 && MOE_TOP_K == 6 && MOE_INTERMEDIATE == 2304 && MOE_RANKS == 4);

/// The forced-operand verifier's geometry, from this checkpoint's real constants.
pub(crate) const FORCED_OPERAND_GEOMETRY: ForcedOperandGeometry = ForcedOperandGeometry {
    hidden: HIDDEN,
    intermediate: MOE_INTERMEDIATE,
    width: MOE_WIDTH,
    width_padded: MOE_WIDTH_PADDED,
    top_k: MOE_TOP_K,
    ranks: MOE_RANKS,
};

/// Repack a checkpoint FP8 weight from row-major [cols, inner] into the B12x micro-tile order that
/// mxfp8_ref::dense_gemm walks. The scale grid needs no repack: b12x_scale_index is
/// (n / 32) * k_blocks + (k / 32), which is the checkpoint's own [ceil(cols / 32), inner / 32]
/// row-major order, so the stored .scale tensor is already in the layout the GEMM reads.
pub(crate) fn pack_b12x(weights: &[u8], cols: usize, inner: usize) -> Vec<u8> {
    assert_eq!(
        weights.len(),
        cols * inner,
        "the checkpoint stores one E4M3 byte per FP8 value"
    );
    let mut tiled = vec![0u8; cols * inner];
    for n in 0..cols {
        for k in 0..inner {
            tiled[mxfp8_ref::b12x_weight_index(n, k, inner)] = weights[n * inner + k];
        }
    }
    tiled
}

/// One token's routing decision with this harness's released-router constants.
pub(crate) fn route_token(
    x: &[f32],
    gate_weight: &[f32],
    bias: &[f32],
    candidates: &[u32],
) -> (Vec<u32>, Vec<f32>) {
    route_token_with(
        x,
        gate_weight,
        bias,
        candidates,
        RouterParams {
            hidden: HIDDEN,
            top_k: MOE_TOP_K,
            gate_temp: GATE_TEMP,
            norm_eps: ROUTE_NORM_EPS,
            scale: ROUTE_SCALE,
        },
    )
}

/// The pre-quantisation routed activation: the SwiGLU value times the routing weight, cast to bf16,
/// because model.py:851 casts before the quantiser reads it. mxfp4_ref::routed_activation_operand
/// applies exactly this before quantising; this form exists so the stage can also feed the
/// UNSQUANTISED value to a second reference and measure what the E4M3 quantiser costs.
fn routed_activation_f32(activation: &[f32], routing: f32) -> Vec<f32> {
    activation
        .iter()
        .map(|value| {
            sparkflash_kernels::m4_quant::bf16_value(sparkflash_kernels::m4_quant::bf16_bits(
                value * routing,
            ))
        })
        .collect()
}

/// The reference the device is compared against: the same pre-padded rank slices accumulated in f32
/// into one [tokens][top_k][hidden] per-(token, slot, hidden) accumulator, exactly as the four serial
/// rank launches accumulate, and then the native routed epilogue over that accumulator: one bf16
/// round per expert, the bf16 values added in ascending expert-id order in f32 (model.py:899-903).
/// There is no per-rank bf16 rounding: that was the old chain and is not the reference's arithmetic
/// (audit M2/F7). The shared expert is not part of this stage, so the epilogue's shared input is zero
/// and its single output bf16 round is applied to the routed sum.
pub(crate) fn reference_combine_chain(
    act_fp8: &[Vec<u8>],
    act_scales: &[Vec<u8>],
    w_rank: &[Vec<u8>],
    ws_rank: &[Vec<u8>],
    expert_ids: &[u32],
    tokens: usize,
    experts: usize,
) -> Vec<u16> {
    let mut slot_acc = vec![0.0f32; tokens * MOE_TOP_K * HIDDEN];
    for rank in 0..MOE_RANKS {
        let _cost = mxfp4_ref::moe_combine_rank(
            &mut slot_acc,
            &act_fp8[rank],
            &act_scales[rank],
            &w_rank[rank],
            &ws_rank[rank],
            expert_ids,
            tokens,
            HIDDEN,
            experts,
        );
    }
    mxfp4_ref::moe_combine_finalize(
        &slot_acc,
        &vec![0u16; tokens * HIDDEN],
        expert_ids,
        tokens,
        HIDDEN,
    )
}

/// The same chain with the activation kept in f32 instead of E4M3: not a reference the stage
/// passes, only the measurement of what the activation quantiser costs, because a comparison over
/// identical quantised bytes cannot see the quantiser at all.
fn unquantised_combine_chain(
    routed_activations: &[Vec<f32>],
    w2_natural: &[Vec<u8>],
    w2_scales: &[Vec<u8>],
    expert_ids: &[u32],
    tokens: usize,
) -> Vec<u16> {
    let blocks = MOE_INTERMEDIATE / 32;
    let mut slot_acc = vec![0.0f32; tokens * MOE_TOP_K * HIDDEN];
    for _rank in 0..MOE_RANKS {
        for token in 0..tokens {
            for column in 0..HIDDEN {
                for slot in 0..MOE_TOP_K {
                    let expert = expert_ids[token * MOE_TOP_K + slot] as usize;
                    let activation = &routed_activations[token * MOE_TOP_K + slot];
                    let weight = &w2_natural[expert];
                    let scales = &w2_scales[expert];
                    let weight_row = &weight[column * MOE_IN_PACKED..(column + 1) * MOE_IN_PACKED];
                    let scale_row = &scales[column * blocks..(column + 1) * blocks];
                    let mut acc = 0.0f32;
                    for (block, scale) in scale_row.iter().enumerate().take(blocks) {
                        let mut partial = 0.0f32;
                        for i in 0..32 {
                            let index = block * 32 + i;
                            let w = fp8_e4m3_decode(fp8_e4m3_encode(mxfp4_ref::fp4_load(
                                weight_row, index,
                            )));
                            partial += activation[index] * w;
                        }
                        acc += partial * ue8m0_decode(*scale);
                    }
                    slot_acc[(token * MOE_TOP_K + slot) * HIDDEN + column] += acc;
                }
            }
        }
    }
    mxfp4_ref::moe_combine_finalize(
        &slot_acc,
        &vec![0u16; tokens * HIDDEN],
        expert_ids,
        tokens,
        HIDDEN,
    )
}

/// Compares two bf16 writes code for code: exact count, absolute delta and the code distance.
pub(crate) fn compare_bf16_bits(label: &str, device: &[u16], reference: &[u16]) -> (Stats, i32) {
    assert_eq!(
        device.len(),
        reference.len(),
        "{label}: the buffers are the same length"
    );
    let mut stats = Stats {
        len: device.len(),
        exact: 0,
        max_delta: 0.0,
        at: 0,
    };
    let mut max_ulp = 0i32;
    for (index, (device_bits, reference_bits)) in device.iter().zip(reference).enumerate() {
        if device_bits == reference_bits {
            stats.exact += 1;
        }
        let delta = (half::bf16::from_bits(*device_bits).to_f32()
            - half::bf16::from_bits(*reference_bits).to_f32())
        .abs();
        if delta > stats.max_delta {
            stats.max_delta = delta;
            stats.at = index;
        }
        let ulp = (bf16_order(*device_bits) - bf16_order(*reference_bits)).abs();
        if ulp > max_ulp {
            max_ulp = ulp;
        }
    }
    println!(
        "  {label}: {} bf16 values, exact {}/{} ({:.4}%), max ULP distance {max_ulp}, max abs delta {:e} at index {}",
        stats.len,
        stats.exact,
        stats.len,
        stats.exact_pct(),
        stats.max_delta,
        stats.at,
    );
    (stats, max_ulp)
}

pub(crate) fn stage3(args: &Args) -> Result<(), String> {
    let kernel = "b12x_mxfp4_moe_sm121";
    let prefix = format!("layers.{}", args.layer);
    let tokens = args.tokens;
    let experts = args.experts;
    println!(
        "=== stage 3: the MoE block, bounded to {experts} experts, router on the host seam ==="
    );
    println!(
        "router: host seam, not device; selection bias gate.bias only - gate.bias_vl is the image-span bias, not read on the text path"
    );
    println!(
        "comparison class: TOLERANCE-BASED, not exact, and the tolerance is named here before any number it judges: at least {STAGE3_EXACT_BAR}% of the {} combine outputs bit-identical to the reference's bf16 write, and no output further than 2 bf16 codes. Reason: both sides consume the same packed bytes and accumulate in the same order (slot ascending, 32-value block ascending, scales applied to each block's partial product), so the only degree of freedom in the f32 rank accumulation is the device's fused multiply-add against the reference's separate multiply and add; the native epilogue (per-expert bf16 round, ascending expert-id add) is applied identically to the device accumulator, and no rank partial is rounded.",
        tokens * HIDDEN
    );
    println!(
        "tolerance by construction: the activation quantiser sits between the f32 model and BOTH sides, so agreement here says the kernel computes the reference's arithmetic, not that either reproduces the f32 expert. The stage measures that gap separately, by running the same chain with an UNSQUANTISED f32 activation and reporting what the E4M3 quantiser alone contributes."
    );
    println!(
        "bounded subset: {experts} of the {MOE_EXPERTS} routed experts are resident - this prompt's most-routed experts first, then the lowest ids to fill - and the router selects over the RESIDENT set, which is what an expert-parallel shard with hot-expert residency sees. The coverage line below reports how much of the full-{MOE_EXPERTS} top-{MOE_TOP_K} the rung captures. This is a bounded test, not the full routing: all {MOE_EXPERTS} experts as f32 would be about 54 GB, which the memory rule puts out of scope tonight."
    );
    println!(
        "device boundary: only the down/combine projection runs on the device, as {MOE_RANKS} serial launches of {kernel} accumulating one f32 [tokens][top_k][hidden] partial buffer over the {MOE_INTERMEDIATE}-wide intermediate. The gate/up projections, the SwiGLU and the activation quantiser are HOST arithmetic in this stage: no device kernel in the six-cubin set reduces K={HIDDEN} against packed FP4 weights, and this stage says so instead of implying the whole block is device-proven. The shared expert is host arithmetic through mxfp8_ref::dense_gemm and is not device-compared either."
    );
    println!(
        "input: ffn_norm applied to the harness's attention-side xhat. The true ffn-side residual stream is the mHC FFN output, which stage 1's boundary does not hand over, so the input row is a stand-in and every claim below is about the MoE arithmetic over that row, not about end-to-end layer text."
    );
    println!(
        "route_scale {ROUTE_SCALE} is inside the routing weight and therefore inside the activation operand's quantiser, which is the model's order (model.py:824-826 applies it to the weights, model.py:849-851 multiplies the activation before the bf16 cast). It is not applied again after the combine."
    );
    check_floor(args, "stage 3 start")?;

    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());
    let gate_weight =
        snap.read_f32_or_bf16(&format!("{prefix}.ffn.gate.weight"), &[MOE_EXPERTS, HIDDEN])?;
    let gate_bias = snap.read_f32(&format!("{prefix}.ffn.gate.bias"), &[MOE_EXPERTS])?;
    let gate_bias_vl = snap.read_f32(&format!("{prefix}.ffn.gate.bias_vl"), &[MOE_EXPERTS])?;
    let ffn_norm = snap.read_f32_or_bf16(&format!("{prefix}.ffn_norm.weight"), &[HIDDEN])?;
    let bias_range = gate_bias
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |acc, value| {
            (acc.0.min(*value), acc.1.max(*value))
        });
    let bias_vl_range = gate_bias_vl
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |acc, value| {
            (acc.0.min(*value), acc.1.max(*value))
        });
    println!(
        "gate.bias: {MOE_EXPERTS} f32 in [{:.4}, {:.4}] - the selection bias, consumed; gate.bias_vl in [{:.4}, {:.4}] - read for the record and NOT consumed on the text path",
        bias_range.0, bias_range.1, bias_vl_range.0, bias_vl_range.1
    );

    let shared_w1_row_major = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w1.weight"),
        "F8_E4M3",
        &[MOE_INTERMEDIATE, HIDDEN],
    )?;
    let shared_s1 = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w1.scale"),
        "F8_E8M0",
        &[MOE_INTERMEDIATE / 32, HIDDEN / 32],
    )?;
    let shared_w3_row_major = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w3.weight"),
        "F8_E4M3",
        &[MOE_INTERMEDIATE, HIDDEN],
    )?;
    let shared_s3 = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w3.scale"),
        "F8_E8M0",
        &[MOE_INTERMEDIATE / 32, HIDDEN / 32],
    )?;
    let shared_w2_row_major = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w2.weight"),
        "F8_E4M3",
        &[HIDDEN, MOE_INTERMEDIATE],
    )?;
    let shared_s2 = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w2.scale"),
        "F8_E8M0",
        &[HIDDEN / 32, MOE_INTERMEDIATE / 32],
    )?;
    // dense_gemm walks the B12x micro-tile order, and the checkpoint stores row-major FP8 tiles, so
    // the three shared weights are repacked once here instead of being compared in the wrong order.
    let shared_w1 = pack_b12x(&shared_w1_row_major, MOE_INTERMEDIATE, HIDDEN);
    let shared_w3 = pack_b12x(&shared_w3_row_major, MOE_INTERMEDIATE, HIDDEN);
    let shared_w2 = pack_b12x(&shared_w2_row_major, HIDDEN, MOE_INTERMEDIATE);
    drop(shared_w1_row_major);
    drop(shared_w3_row_major);
    drop(shared_w2_row_major);
    println!(
        "shared expert weights: repacked from the checkpoint's row-major FP8 tiles into the B12x micro-tile order dense_gemm walks; the ue8m0 scale grid needs no repack because b12x_scale_index is already the checkpoint's out-block-major then k-block order"
    );

    // The stage-1 chain, exactly as stages 1 and 2 run it: the released tensors and the boundary
    // seed, whose returned codes and scales are the hidden state this stage routes.
    let mhc = read_released_mhc_tensors(&mut snap, &prefix, "attn")?;
    let attn_norm = snap.read_f32_or_bf16(&format!("{prefix}.attn_norm.weight"), &[HIDDEN])?;
    let predictor = mhc.predictor;
    let hc_base = mhc.hc_base;
    let hc_scale = mhc.hc_scale;
    let a_prev = [1.0f32, 0.0, 0.0, 0.0];

    let mut block_input: Vec<Vec<f32>> = Vec::with_capacity(tokens);
    let mut block_quant: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(tokens);
    let mut block_activation: Vec<Vec<f32>> = Vec::with_capacity(tokens);
    for token in 0..tokens {
        let row = snap.read_row_bf16("embed.weight", token, HIDDEN)?;
        let mut x = Vec::with_capacity(HC_MULT * HIDDEN);
        for _ in 0..HC_MULT {
            x.extend_from_slice(&row);
        }
        let mut streams = Streams::new(x);
        let mut ledger = TrafficLedger::default();
        let (_next, _coefficients, codes, scales) = mega_mhc_layer_with_scale(
            &mut streams,
            &attn_norm,
            &a_prev,
            &predictor,
            &hc_base,
            &hc_scale,
            &mut ledger,
        );
        let mut hidden: Vec<f32> = codes
            .as_slice()
            .iter()
            .enumerate()
            .map(|(index, code)| {
                fp8_e4m3_decode(*code) * ue8m0_decode(scales.as_slice()[index / QUANT_BLOCK])
            })
            .collect();
        rms_norm_row(&mut hidden, &ffn_norm, NORM_EPS);
        let (quant, quant_scales) = act_quant_fp8_row(&hidden);
        let activation: Vec<f32> = quant
            .iter()
            .enumerate()
            .map(|(index, code)| {
                fp8_e4m3_decode(*code) * ue8m0_decode(quant_scales[index / QUANT_BLOCK])
            })
            .collect();
        block_input.push(hidden);
        block_quant.push((quant, quant_scales));
        block_activation.push(activation);
        let _ = ledger;
    }
    println!(
        "input: {tokens} rows of ffn_norm(xhat) at hidden width {HIDDEN}, quantised to E4M3 for the experts"
    );

    // The resident subset is this prompt's most-routed experts first, then the lowest ids to fill:
    // that is an expert-parallel shard with hot-expert residency, and it makes the coverage number
    // mean something. The full-384 selection is computed from gate.weight alone, before any expert
    // weight is read, so choosing the subset costs no expert bytes.
    let all_experts: Vec<u32> = (0..MOE_EXPERTS as u32).collect();
    let mut full_selections: Vec<Vec<u32>> = Vec::with_capacity(tokens);
    let mut tally = vec![0usize; MOE_EXPERTS];
    for hidden in block_input.iter().take(tokens) {
        let (full_ids, _) = route_token(hidden, &gate_weight, &gate_bias, &all_experts);
        for expert in &full_ids {
            tally[*expert as usize] += 1;
        }
        full_selections.push(full_ids);
    }
    let mut ranked: Vec<u32> = all_experts.clone();
    ranked.sort_by_key(|expert| (std::cmp::Reverse(tally[*expert as usize]), *expert));
    let mut loaded: Vec<u32> = ranked.iter().copied().take(experts).collect();
    loaded.sort_unstable();
    let mut table_of = vec![usize::MAX; MOE_EXPERTS];
    for (table, expert) in loaded.iter().enumerate() {
        table_of[*expert as usize] = table;
    }
    let distinct = tally.iter().filter(|count| **count > 0).count();
    let covered: usize = full_selections
        .iter()
        .flatten()
        .filter(|expert| table_of[**expert as usize] != usize::MAX)
        .count();
    println!(
        "resident subset: {experts} of {MOE_EXPERTS} experts, ids {loaded:?}, chosen as this prompt's most-routed experts (ties to the lower id) then the lowest ids to fill, so every rung holds exactly {experts} resident weight sets"
    );
    println!(
        "router coverage: {covered} of {} full-{MOE_EXPERTS} top-{MOE_TOP_K} selections are resident at this rung; {distinct} distinct experts are routed over {tokens} tokens",
        tokens * MOE_TOP_K
    );
    let mut expert_ids: Vec<u32> = Vec::with_capacity(tokens * MOE_TOP_K);
    let mut weights_all: Vec<f32> = Vec::with_capacity(tokens * MOE_TOP_K);
    for (token, hidden) in block_input.iter().enumerate().take(tokens) {
        let (ids, weights) = route_token(hidden, &gate_weight, &gate_bias, &loaded);
        println!(
            "  token {token}: resident-set selection {ids:?} weights {weights:?}; full selection {:?}",
            full_selections[token]
        );
        for expert in &ids {
            let table = table_of[*expert as usize];
            assert!(table != usize::MAX, "the selected expert is resident");
            expert_ids.push(table as u32);
        }
        weights_all.extend_from_slice(&weights);
    }

    let per_expert = MOE_INTERMEDIATE * MOE_EXPERT_IN_PACKED
        + MOE_INTERMEDIATE * MOE_EXPERT_IN_SCALES
        + HIDDEN * MOE_IN_PACKED
        + HIDDEN * MOE_IN_SCALES;
    let host_planned = experts * per_expert
        + MOE_RANKS * (experts * HIDDEN * MOE_PACKED_ROW + experts * HIDDEN * MOE_SCALE_ROW)
        + tokens * MOE_TOP_K * 4 * MOE_WIDTH_PADDED;
    admit_memory(args, host_planned as u64, "stage 3 expert weights")?;
    let mut w1: Vec<Vec<u8>> = Vec::with_capacity(experts);
    let mut s1: Vec<Vec<u8>> = Vec::with_capacity(experts);
    let mut w3: Vec<Vec<u8>> = Vec::with_capacity(experts);
    let mut s3: Vec<Vec<u8>> = Vec::with_capacity(experts);
    let mut w2: Vec<Vec<u8>> = Vec::with_capacity(experts);
    let mut s2: Vec<Vec<u8>> = Vec::with_capacity(experts);
    for expert in &loaded {
        let base = format!("{prefix}.ffn.experts.{expert}");
        w1.push(snap.read_raw(
            &format!("{base}.w1.weight"),
            "I8",
            &[MOE_INTERMEDIATE, MOE_EXPERT_IN_PACKED],
        )?);
        s1.push(snap.read_raw(
            &format!("{base}.w1.scale"),
            "F8_E8M0",
            &[MOE_INTERMEDIATE, MOE_EXPERT_IN_SCALES],
        )?);
        w3.push(snap.read_raw(
            &format!("{base}.w3.weight"),
            "I8",
            &[MOE_INTERMEDIATE, MOE_EXPERT_IN_PACKED],
        )?);
        s3.push(snap.read_raw(
            &format!("{base}.w3.scale"),
            "F8_E8M0",
            &[MOE_INTERMEDIATE, MOE_EXPERT_IN_SCALES],
        )?);
        w2.push(snap.read_raw(&format!("{base}.w2.weight"), "I8", &[HIDDEN, MOE_IN_PACKED])?);
        s2.push(snap.read_raw(
            &format!("{base}.w2.scale"),
            "F8_E8M0",
            &[HIDDEN, MOE_IN_SCALES],
        )?);
    }
    println!(
        "expert weights: {experts} experts x {per_expert} stored bytes (w1/w3 packed FP4 + ue8m0 scales, w2 packed FP4 + ue8m0 scales) = {} B, read exactly as the checkpoint stores them - no expert is decoded to f32",
        experts * per_expert
    );

    let mut routed_activations: Vec<Vec<f32>> = Vec::with_capacity(tokens * MOE_TOP_K);
    let mut act_fp8: Vec<Vec<u8>> = (0..MOE_RANKS).map(|_| Vec::new()).collect();
    let mut act_scales: Vec<Vec<u8>> = (0..MOE_RANKS).map(|_| Vec::new()).collect();
    for token in 0..tokens {
        for slot in 0..MOE_TOP_K {
            let expert = expert_ids[token * MOE_TOP_K + slot] as usize;
            let gate = gate_up_rows(
                &block_activation[token],
                &w1[expert],
                &s1[expert],
                MOE_INTERMEDIATE,
                HIDDEN,
            );
            let up = gate_up_rows(
                &block_activation[token],
                &w3[expert],
                &s3[expert],
                MOE_INTERMEDIATE,
                HIDDEN,
            );
            let activation: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(gate, up)| swiglu_ref::swiglu(*gate, *up, swiglu_ref::SWIGLU_LIMIT))
                .collect();
            let routing = weights_all[token * MOE_TOP_K + slot];
            routed_activations.push(routed_activation_f32(&activation, routing));
            for rank in 0..MOE_RANKS {
                let (codes, scales) = mxfp4_ref::routed_activation_operand(
                    &activation[rank * MOE_WIDTH..(rank + 1) * MOE_WIDTH],
                    routing,
                );
                let mut row = vec![0u8; MOE_WIDTH_PADDED];
                row[..MOE_WIDTH].copy_from_slice(&codes);
                act_fp8[rank].extend_from_slice(&row);
                let mut scale_row = vec![0u8; MOE_SCALE_ROW];
                scale_row[..MOE_WIDTH / QUANT_BLOCK].copy_from_slice(&scales);
                act_scales[rank].extend_from_slice(&scale_row);
            }
        }
    }

    let mut w_rank: Vec<Vec<u8>> = Vec::with_capacity(MOE_RANKS);
    let mut ws_rank: Vec<Vec<u8>> = Vec::with_capacity(MOE_RANKS);
    for rank in 0..MOE_RANKS {
        let mut weight = Vec::with_capacity(experts * HIDDEN * MOE_PACKED_ROW);
        let mut scale = Vec::with_capacity(experts * HIDDEN * MOE_SCALE_ROW);
        for expert in 0..experts {
            weight.extend_from_slice(&slice_and_pad_rows(
                &w2[expert],
                HIDDEN,
                MOE_IN_PACKED,
                rank * (MOE_WIDTH / 2),
                MOE_WIDTH / 2,
                MOE_PACKED_ROW,
            ));
            scale.extend_from_slice(&slice_and_pad_rows(
                &s2[expert],
                HIDDEN,
                MOE_IN_SCALES,
                rank * (MOE_WIDTH / 32),
                MOE_WIDTH / 32,
                MOE_SCALE_ROW,
            ));
        }
        w_rank.push(weight);
        ws_rank.push(scale);
    }
    println!(
        "rank slices: {MOE_RANKS} ranks of {} packed + {} scale bytes each, built by slicing every expert's stored row and padding the 576-wide rank to the kernel's 640",
        w_rank[0].len(),
        ws_rank[0].len()
    );

    let reference = reference_combine_chain(
        &act_fp8,
        &act_scales,
        &w_rank,
        &ws_rank,
        &expert_ids,
        tokens,
        experts,
    );
    println!(
        "reference: mxfp4_ref::moe_combine_rank x{MOE_RANKS} into one f32 accumulator, then moe_combine_finalize (per-expert bf16, ascending expert-id add; this stage has no shared expert)"
    );

    check_floor(args, "stage 3 device setup")?;
    let cuda = Cuda::load()?;
    let cubin_path = crate::cubin_path(&args.cubin_dir, kernel)?;
    let cubin = std::fs::read(&cubin_path).map_err(|e| {
        format!(
            "{}: {e} (build the cubin first; the module docs give the nvcc line)",
            cubin_path.display()
        )
    })?;
    let (module, func) = cuda.load_kernel(&cubin, kernel_id(kernel)?)?;
    let device_planned = tokens * MOE_TOP_K * HIDDEN * 4
        + act_fp8[0].len()
        + act_scales[0].len()
        + w_rank[0].len()
        + ws_rank[0].len()
        + tokens * MOE_TOP_K * 4;
    admit_memory(args, device_planned as u64, "stage 3 device buffers")?;
    let mut allocated = 0usize;
    let act_d = cuda.alloc(act_fp8[0].len(), &mut allocated)?;
    let acts_d = cuda.alloc(act_scales[0].len(), &mut allocated)?;
    let w_d = cuda.alloc(w_rank[0].len(), &mut allocated)?;
    let ws_d = cuda.alloc(ws_rank[0].len(), &mut allocated)?;
    let ids_d = cuda.alloc(tokens * MOE_TOP_K * 4, &mut allocated)?;
    let acc_d = cuda.alloc(tokens * MOE_TOP_K * HIDDEN * 4, &mut allocated)?;
    println!(
        "device bytes allocated: {allocated} ({} MiB)",
        allocated / (1024 * 1024)
    );
    // The four rank launches ADD into the f32 accumulator, so it starts zeroed.
    cuda.copy_htod(&acc_d, &vec![0u8; tokens * MOE_TOP_K * HIDDEN * 4])?;
    let grid = (tokens as u32, (HIDDEN / MOE_TILE_D as usize) as u32, 1u32);
    let block = (MOE_THREADS, 1u32, 1u32);
    let mut last_params: [u64; 10] = [0; 10];
    for rank in 0..MOE_RANKS {
        cuda.copy_htod(&act_d, &act_fp8[rank])?;
        cuda.copy_htod(&acts_d, &act_scales[rank])?;
        cuda.copy_htod(&w_d, &w_rank[rank])?;
        cuda.copy_htod(&ws_d, &ws_rank[rank])?;
        cuda.copy_htod(&ids_d, &u32s_to_bytes(&expert_ids))?;
        let params: [u64; 10] = [
            act_d.0,
            acts_d.0,
            w_d.0,
            ws_d.0,
            ids_d.0,
            acc_d.0,
            tokens as u64,
            0, // trace_ring: null, built without -DSPARKFLASH_ENABLE_GPU_TRACE
            args.layer as u64,
            rank as u64,
        ];
        last_params = params;
        check_floor(args, &format!("stage 3 rank {rank} pre-launch"))?;
        cuda.launch(func, grid, block, &params)?;
        check_floor(args, &format!("stage 3 rank {rank} post-launch"))?;
    }
    // One readback after all four ranks: the f32 accumulator is the expert outputs summed over the
    // tensor-parallel ranks, and the native epilogue rounds each expert once.
    let mut acc_bytes = vec![0u8; tokens * MOE_TOP_K * HIDDEN * 4];
    cuda.copy_dtoh(&mut acc_bytes, &acc_d)?;
    let previous: Vec<u16> = mxfp4_ref::moe_combine_finalize(
        &acc_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect::<Vec<f32>>(),
        &vec![0u16; tokens * HIDDEN],
        &expert_ids,
        tokens,
        HIDDEN,
    );

    let (stats, max_ulp) = compare_bf16_bits("combine output", &previous, &reference);
    let unquantised = unquantised_combine_chain(&routed_activations, &w2, &s2, &expert_ids, tokens);
    let (gap, gap_ulp) = compare_bf16_bits(
        "activation quantisation effect (quantised reference vs the same chain with an f32 activation)",
        &reference,
        &unquantised,
    );
    let mut routed_energy = 0.0f64;
    let mut shared_energy = 0.0f64;
    for token in 0..tokens {
        let (quant, quant_scales) = &block_quant[token];
        let gate_bits = mxfp8_ref::dense_gemm(
            quant,
            quant_scales,
            &shared_w1,
            &shared_s1,
            1,
            MOE_INTERMEDIATE,
            HIDDEN,
        );
        let up_bits = mxfp8_ref::dense_gemm(
            quant,
            quant_scales,
            &shared_w3,
            &shared_s3,
            1,
            MOE_INTERMEDIATE,
            HIDDEN,
        );
        let gate: Vec<f32> = gate_bits
            .iter()
            .map(|bits| half::bf16::from_bits(*bits).to_f32())
            .collect();
        let up: Vec<f32> = up_bits
            .iter()
            .map(|bits| half::bf16::from_bits(*bits).to_f32())
            .collect();
        // The shared expert's activation is quantised by the same reference activation rule the
        // device kernel now carries: swiglu_ref::quantise_row delegates to act_quant_fp8_row, and
        // this call site spells the two steps out because it needs the f32 activation as well.
        let activated: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| swiglu_ref::swiglu(*gate, *up, swiglu_ref::SWIGLU_LIMIT))
            .collect();
        let (codes, scales) = act_quant_fp8_row(&activated);
        let out_bits = mxfp8_ref::dense_gemm(
            &codes,
            &scales,
            &shared_w2,
            &shared_s2,
            1,
            HIDDEN,
            MOE_INTERMEDIATE,
        );
        for column in 0..HIDDEN {
            let routed = half::bf16::from_bits(previous[token * HIDDEN + column]).to_f32();
            let shared = half::bf16::from_bits(out_bits[column]).to_f32();
            routed_energy += f64::from(routed) * f64::from(routed);
            shared_energy += f64::from(shared) * f64::from(shared);
        }
    }
    let routed_rms = (routed_energy / (tokens * HIDDEN) as f64).sqrt();
    let shared_rms = (shared_energy / (tokens * HIDDEN) as f64).sqrt();
    println!(
        "shared expert (host arithmetic through dense_gemm, NOT device-compared): rms routed {routed_rms:.4}, rms shared {shared_rms:.4}, shared share {:.1}% of the summed energy",
        100.0 * shared_energy / (routed_energy + shared_energy)
    );

    if args.repeat > 1 {
        check_floor(args, "stage 3 timing pre-launch")?;
        let (mut per_launch, batched_ms) =
            cuda.launch_repeated(func, grid, block, &last_params, args.repeat)?;
        check_floor(args, "stage 3 timing post-launch")?;
        per_launch.sort_by(|a, b| a.partial_cmp(b).expect("event times are finite"));
        let count = per_launch.len();
        let p50 = per_launch[count / 2];
        println!(
            "timing: {count} launches of the last rank at grid ({tokens}, {}, 1); per-launch ms min {:.4} p50 {p50:.4} max {:.4}; batched ms/launch {batched_ms:.4}; the routed combine is {MOE_RANKS} launches, so {:.3} ms for {tokens} tokens = {:.3} ms/token",
            HIDDEN / MOE_TILE_D as usize,
            per_launch[0],
            per_launch[count - 1],
            MOE_RANKS as f64 * p50,
            MOE_RANKS as f64 * p50 / tokens as f64
        );
    }

    cuda.free(module, &[act_d, acts_d, w_d, ws_d, ids_d, acc_d]);
    check_floor(args, "stage 3 end")?;
    let exact_ok = stats.exact_pct() >= STAGE3_EXACT_BAR;
    let ulp_ok = max_ulp <= 2;
    println!(
        "stage 3 verdict: {} (combine exact {:.4}% against a {STAGE3_EXACT_BAR}% bar, max ULP distance {max_ulp} against a 2-code bar; quantisation gap {:.2}% of outputs differing, max abs delta {:e}, {gap_ulp} codes)",
        if exact_ok && ulp_ok {
            "within tolerance"
        } else {
            "OVER TOLERANCE"
        },
        stats.exact_pct(),
        100.0 - gap.exact_pct(),
        gap.max_delta
    );
    if exact_ok && ulp_ok {
        println!(
            "stage 3: PASS ({experts} experts, {tokens} tokens, {MOE_RANKS} rank-accumulating launches)"
        );
        Ok(())
    } else {
        Err(format!(
            "stage 3: FAIL at the combine (exact {:.4}%, max ULP {max_ulp})",
            stats.exact_pct()
        ))
    }
}

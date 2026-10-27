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

//! CPU twin, split-K model and the D51 tolerance row for the fused routed gate/up kernel
//! (pending_kernels/routed_gate_up_swiglu_splitk_sm121.cu).
//!
//! Two arms, one arithmetic class:
//!
//! * the **reference arm** ([twin_activation]) is the incumbent three-kernel chain computed by the
//!   pinned bit-exact references: [crate::routed_gate_up_ref::gate_up] for w1 and w3 (the serial
//!   ascending 160-block fold, one bf16 rounding per output element), then
//!   [crate::swiglu_ref::weighted_activation] and [crate::m4_quant::act_quant_fp8_row] for the
//!   weighted SwiGLU and the activation quantiser. That is what the pinned
//!   `routed_gate_up_sm121` (twice) and `swiglu_mxfp8_quant_weighted_sm121` compute on the
//!   device, and it is the arm D51 keeps as the regression anchor;
//! * the **performance arm** ([splitk_activation]) is this crate's model of the new kernel: the
//!   same block terms in the same quantisation, with the f32 K-sum reassociated exactly as the
//!   kernel's warp-sharded schedule reassociates it ([splitk_fold]), and the same epilogue.
//!
//! The measured difference between the arms IS the D51 per-kernel tolerance row
//! ([deviation] -> [DeviationRow]). It is recorded, asserted against [DECLARED_TOLERANCE], and -
//! on a host with a device - reproduced by the kernel itself; [summary] formats the row so a log
//! carries the numbers rather than a claim.
//!
//! Why this is not a second implementation of the engine's f32 expert body: the operands are the
//! quantised ones (D14/D39 rule 1), the block term is the twin's
//! `(32-value f32 dot) * act_scale * w_scale` with one rounding per multiply/add pair, and the
//! epilogue calls the SAME Rust functions in both arms. The only difference the row can see is the
//! order in which the 160 block terms are added, which is the only thing D51 amendment A permits a
//! performance kernel to change. One epilogue difference is inherited, not introduced: the pinned
//! weighted-SwiGLU kernel computes silu with CUDA's `__expf` while the reference uses `exp`, so
//! the device chain's own epilogue already differs from the twin there; both arms here use the
//! reference's expression, which is what isolates the reduction order in this row.
//!
//! What this module cannot measure: device time, achieved device bandwidth and occupancy. [Traffic]
//! is arithmetic from the shapes the launch addresses (as in kernel-occupancy-inventory.md), and
//! [crate::roofline::achieved_gbs] turns a wall clock into GB/s for the CPU arm only - labelled as
//! such, never as a device throughput claim.
//! [crate::routed_gate_up_swiglu_splitk_abi::GPU_VERIFICATION_PLAN] is the device row list.

use crate::m4_quant::{
    QUANT_BLOCK as MX_BLOCK, act_quant_fp8_row, bf16_bits, bf16_value, fp4_e2m1_decode,
    fp8_e4m3_decode, fp8_e4m3_encode, ue8m0_decode,
};
use crate::routed_gate_up_ref::{self, GateUpCost};
use crate::swiglu_ref;

/// The hidden size both projections reduce (config.json hidden_size).
pub const HIDDEN: usize = 5120;
/// The routed width per rank before padding.
pub const WIDTH: usize = 576;
/// The padded routed width of the activation operand (D30).
pub const WIDTH_PADDED: usize = 640;
/// Routed slots per token.
pub const TOP_K: usize = 6;
/// MX block width: one scale per 32 values along K.
pub const QUANT_BLOCK: usize = MX_BLOCK;
/// MX blocks per hidden row.
pub const K_BLOCKS: usize = HIDDEN / QUANT_BLOCK;
/// Warps per CTA: also the number of K shards the split-K schedule creates.
pub const WARPS: usize = 8;
/// K blocks one warp-shard folds: 160 / 8.
pub const BLOCKS_PER_SHARD: usize = K_BLOCKS / WARPS;
/// Output columns one CTA owns: exactly one MX block.
pub const TILE: usize = 32;
/// Blocks along the padded output width: 640 / 32.
pub const TILES: usize = WIDTH_PADDED / TILE;
/// Columns one warp owns in the shuffle fold: 32 / 8.
pub const COLS_PER_GROUP: usize = TILE / WARPS;
/// Accumulators per (shard, column): gate and up per slot.
pub const ROWS: usize = TOP_K * 2;
/// Packed FP4 bytes per weight row: 5120 / 2.
pub const WEIGHT_ROW_BYTES: usize = HIDDEN / 2;
/// ue8m0 scale bytes per activation row and per weight row: 5120 / 32.
pub const SCALE_ROW_BYTES: usize = K_BLOCKS;
/// ue8m0 scale bytes per quantised activation row: 640 / 32.
pub const OUT_SCALE_ROW_BYTES: usize = WIDTH_PADDED / QUANT_BLOCK;

/// The routing-weighted SwiGLU's clamp limit: `swiglu_limit`, 10.0 in the released config.
pub const SWIGLU_LIMIT: f32 = 10.0;

/// One E2M1 value of a packed weight row: low nibble even, high nibble odd.
///
/// # Panics
/// Panics if the row is too short.
#[must_use]
pub fn fp4_load(row: &[u8], index: usize) -> f32 {
    let byte = index / 2;
    assert!(byte < row.len(), "the packed FP4 row is too short");
    let packed = row[byte];
    let nibble = if index.is_multiple_of(2) {
        packed & 0x0F
    } else {
        packed >> 4
    };
    fp4_e2m1_decode(nibble)
}

/// One 32-value MX block's scaled term for one output element, in the twin's order: the f32 dot
/// with one rounding per multiply/add pair, then both power-of-two scales on the PARTIAL.
#[must_use]
pub fn block_term(
    act_row: &[u8],
    act_scale_row: &[u8],
    w_row: &[u8],
    w_scale_row: &[u8],
    block: usize,
) -> f32 {
    let base = block * QUANT_BLOCK;
    let mut partial = 0.0f32;
    for i in 0..QUANT_BLOCK {
        let index = base + i;
        let act_value = fp8_e4m3_decode(act_row[index]);
        let w_value = fp8_e4m3_decode(fp8_e4m3_encode(fp4_load(w_row, index)));
        // Separate roundings, as the kernel's __fadd_rn(__fmul_rn(..)) pair: Rust does not
        // contract f32 multiply/add into an FMA, so this is the same arithmetic.
        partial += act_value * w_value;
    }
    (partial * ue8m0_decode(act_scale_row[block])) * ue8m0_decode(w_scale_row[block])
}

/// Every block term of one output element, in ascending block order.
#[must_use]
pub fn block_terms(
    act_row: &[u8],
    act_scale_row: &[u8],
    w_row: &[u8],
    w_scale_row: &[u8],
) -> [f32; K_BLOCKS] {
    let mut terms = [0.0f32; K_BLOCKS];
    for (block, term) in terms.iter_mut().enumerate() {
        *term = block_term(act_row, act_scale_row, w_row, w_scale_row, block);
    }
    terms
}

/// The reference arm's fold: one ascending chain from `+0.0` over all 160 block terms. This is
/// what [crate::routed_gate_up_ref::gate_up] does per element, and a test pins that composition.
#[must_use]
pub fn serial_fold(terms: &[f32; K_BLOCKS]) -> f32 {
    let mut acc = 0.0f32;
    for term in terms {
        acc += *term;
    }
    acc
}

/// The performance arm's fold, exactly as the kernel's schedule computes it.
///
/// * warp shard `s` folds blocks `{b : b % 8 == s}` serially in ascending order
///   (`block = s + 8 * j`, `j` ascending) - [splitk_shard_fold];
/// * the eight shard totals are combined by the three-level `__shfl_xor_sync` butterfly
///   (offsets 1, 2, 4), whose value is
///   `((t0 + t1) + (t2 + t3)) + ((t4 + t5) + (t6 + t7))`, every add one f32 rounding.
#[must_use]
pub fn splitk_fold(terms: &[f32; K_BLOCKS]) -> f32 {
    let mut value = [0.0f32; WARPS];
    for (shard, total) in value.iter_mut().enumerate() {
        *total = splitk_shard_fold(terms, shard);
    }
    // The butterfly: after xor 1 every lane holds t(k) + t(k^1); after xor 2 that pair plus its
    // k^2 partner; after xor 4 the four-lane group sums. The result is the same in all eight
    // lanes and is the expression in the kernel's comment.
    let mut offset = 1;
    while offset < WARPS {
        let mut next = value;
        for lane in 0..WARPS {
            next[lane] = value[lane] + value[lane ^ offset];
        }
        value = next;
        offset <<= 1;
    }
    for lane in 1..WARPS {
        assert!(
            value[lane].to_bits() == value[0].to_bits(),
            "the shuffle butterfly must converge in every lane"
        );
    }
    value[0]
}

/// One warp shard's serial fold: blocks `s + 8*j` for ascending `j`, from `+0.0`.
#[must_use]
pub fn splitk_shard_fold(terms: &[f32; K_BLOCKS], shard: usize) -> f32 {
    assert!(shard < WARPS, "the CTA has {WARPS} warp shards");
    let mut acc = 0.0f32;
    for j in 0..BLOCKS_PER_SHARD {
        acc += terms[shard + WARPS * j];
    }
    acc
}

/// What one launch of either arm reads and writes, counted from the shapes the arguments address.
///
/// The counts are arithmetic, not a measurement: the packed weights are read once per routed row
/// per projection (the grouped kernel reads each expert's row set once because one CTA owns all
/// six slots for its 32 columns), the activation row once per token in the fused arm and once per
/// gate/up launch in the incumbent arm, and [Traffic::intermediate_bytes] is the two bf16 gate/up
/// tensors the incumbent writes and the weighted epilogue reads back - the tensors the fusion
/// removes.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Traffic {
    /// Packed E2M1 weight bytes.
    pub weight_bytes: u64,
    /// ue8m0 weight-scale bytes.
    pub weight_scale_bytes: u64,
    /// E4M3 activation bytes.
    pub activation_bytes: u64,
    /// ue8m0 activation-scale bytes.
    pub activation_scale_bytes: u64,
    /// E4M3 output (activation-operand) bytes.
    pub output_code_bytes: u64,
    /// ue8m0 output-scale bytes.
    pub output_scale_bytes: u64,
    /// The incumbent chain's intermediate bf16 gate/up traffic; zero for the fused arm.
    pub intermediate_bytes: u64,
}

impl Traffic {
    /// Every byte the launch addresses, including the intermediates.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.weight_bytes
            + self.weight_scale_bytes
            + self.activation_bytes
            + self.activation_scale_bytes
            + self.output_code_bytes
            + self.output_scale_bytes
            + self.intermediate_bytes
    }
}

/// The performance arm's output: the quantised activation operand plus the f32 accumulators the
/// tolerance row compares, and the traffic count.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitKOutput {
    /// `[tokens * top_k][640]` E4M3 activation codes.
    pub codes: Vec<u8>,
    /// `[tokens * top_k][20]` ue8m0 block scales.
    pub scales: Vec<u8>,
    /// `[tokens * top_k][640]` the weighted activation before quantisation.
    pub weighted: Vec<f32>,
    /// `[tokens * top_k][576]` gate f32 accumulators for the live columns.
    pub gate_f32: Vec<f32>,
    /// `[tokens * top_k][576]` up f32 accumulators for the live columns.
    pub up_f32: Vec<f32>,
    /// The launch's counted traffic.
    pub traffic: Traffic,
    /// Multiply-accumulates issued.
    pub macs: u64,
}

/// The reference arm's output: the bit-exact chain's bytes, its f32 accumulators and the values
/// its epilogue reads.
#[derive(Debug, Clone, PartialEq)]
pub struct TwinOutput {
    /// `[tokens * top_k][640]` E4M3 activation codes.
    pub codes: Vec<u8>,
    /// `[tokens * top_k][20]` ue8m0 block scales.
    pub scales: Vec<u8>,
    /// `[tokens * top_k][640]` the weighted activation the quantiser reads.
    pub weighted: Vec<f32>,
    /// `[tokens * top_k][640]` the bf16 gate tensor the chain stores between launches.
    pub gate_bf16: Vec<u16>,
    /// `[tokens * top_k][640]` the bf16 up tensor the chain stores between launches.
    pub up_bf16: Vec<u16>,
    /// `[tokens * top_k][576]` gate f32 accumulators, the serial fold before the bf16 round.
    pub gate_f32: Vec<f32>,
    /// `[tokens * top_k][576]` up f32 accumulators, the serial fold before the bf16 round.
    pub up_f32: Vec<f32>,
    /// The two gate/up launches' own counts, as the pinned twin returns them.
    pub gate_up_cost: [GateUpCost; 2],
    /// The chain's counted traffic.
    pub traffic: Traffic,
}

/// The routed-row count a token batch produces.
#[must_use]
pub fn routed_rows(tokens: usize) -> usize {
    tokens * TOP_K
}

/// The traffic either arm addresses for `tokens` tokens, from the shapes alone.
#[must_use]
pub fn traffic(tokens: usize, fused: bool) -> Traffic {
    let rows = routed_rows(tokens) as u64;
    Traffic {
        // Both projections' staged rows: two packed rows per routed row, one scale row each.
        weight_bytes: 2 * rows * WIDTH as u64 * WEIGHT_ROW_BYTES as u64,
        weight_scale_bytes: 2 * rows * WIDTH as u64 * SCALE_ROW_BYTES as u64,
        activation_bytes: tokens as u64 * HIDDEN as u64 * if fused { 1 } else { 2 },
        activation_scale_bytes: tokens as u64 * SCALE_ROW_BYTES as u64 * if fused { 1 } else { 2 },
        output_code_bytes: rows * WIDTH_PADDED as u64,
        output_scale_bytes: rows * OUT_SCALE_ROW_BYTES as u64,
        // Two bf16 tensors, each written once and read once: (write + read) per projection.
        intermediate_bytes: if fused {
            0
        } else {
            2 * rows * WIDTH_PADDED as u64 * 2 * 2
        },
    }
}

/// The reference arm: the incumbent three-kernel chain, through the pinned bit-exact references.
///
/// `gate_up` is called twice (w1 and w3), each output element is widened from its bf16 bits, the
/// weighted SwiGLU is [crate::swiglu_ref::weighted_activation] (multiply, round the product to
/// bf16, then quantise), and the quantiser is the released [act_quant_fp8_row]. The routing weight
/// is inside the quantiser's input, which is the model's order (model.py:849-851).
///
/// The f32 accumulators are re-walked here because the pinned twin returns only the rounded bf16
/// tensor: the tolerance row's f32 level needs the pre-rounding value, and the block terms it folds
/// are pinned to [crate::routed_gate_up_ref::gate_up]'s arithmetic by this module's tests.
///
/// # Panics
/// Panics if a buffer is too short for the declared shape or an expert index is out of range
/// (both are [crate::routed_gate_up_ref::gate_up]'s panics).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn twin_activation(
    act_fp8: &[u8],
    act_scales: &[u8],
    w1_fp4: &[u8],
    w1_scales: &[u8],
    w3_fp4: &[u8],
    w3_scales: &[u8],
    expert_ids: &[u32],
    routing_weights: &[f32],
    tokens: usize,
    experts: usize,
) -> TwinOutput {
    let rows = routed_rows(tokens);
    assert_eq!(routing_weights.len(), rows, "one routing weight per row");
    let (gate_bf16, gate_cost) = routed_gate_up_ref::gate_up(
        act_fp8, act_scales, w1_fp4, w1_scales, expert_ids, tokens, experts,
    );
    let (up_bf16, up_cost) = routed_gate_up_ref::gate_up(
        act_fp8, act_scales, w3_fp4, w3_scales, expert_ids, tokens, experts,
    );
    let mut codes = Vec::with_capacity(rows * WIDTH_PADDED);
    let mut scales = Vec::with_capacity(rows * OUT_SCALE_ROW_BYTES);
    let mut weighted = Vec::with_capacity(rows * WIDTH_PADDED);
    for row in 0..rows {
        let gate: Vec<f32> = gate_bf16[row * WIDTH_PADDED..][..WIDTH_PADDED]
            .iter()
            .map(|bits| bf16_value(*bits))
            .collect();
        let up: Vec<f32> = up_bf16[row * WIDTH_PADDED..][..WIDTH_PADDED]
            .iter()
            .map(|bits| bf16_value(*bits))
            .collect();
        let row_weighted =
            swiglu_ref::weighted_activation(&gate, &up, SWIGLU_LIMIT, routing_weights[row]);
        let (row_codes, row_scales) = act_quant_fp8_row(&row_weighted);
        weighted.extend_from_slice(&row_weighted);
        codes.extend_from_slice(&row_codes);
        scales.extend_from_slice(&row_scales);
    }
    let mut gate_f32 = vec![0.0f32; rows * WIDTH];
    let mut up_f32 = vec![0.0f32; rows * WIDTH];
    for token in 0..tokens {
        let act_row = &act_fp8[token * HIDDEN..][..HIDDEN];
        let act_scale_row = &act_scales[token * SCALE_ROW_BYTES..][..SCALE_ROW_BYTES];
        for slot in 0..TOP_K {
            let row = token * TOP_K + slot;
            let expert = expert_ids[row] as usize;
            for column in 0..WIDTH {
                let weight_row = expert * WIDTH + column;
                let w1_row = &w1_fp4[weight_row * WEIGHT_ROW_BYTES..][..WEIGHT_ROW_BYTES];
                let s1_row = &w1_scales[weight_row * SCALE_ROW_BYTES..][..SCALE_ROW_BYTES];
                let w3_row = &w3_fp4[weight_row * WEIGHT_ROW_BYTES..][..WEIGHT_ROW_BYTES];
                let s3_row = &w3_scales[weight_row * SCALE_ROW_BYTES..][..SCALE_ROW_BYTES];
                gate_f32[row * WIDTH + column] =
                    serial_fold(&block_terms(act_row, act_scale_row, w1_row, s1_row));
                up_f32[row * WIDTH + column] =
                    serial_fold(&block_terms(act_row, act_scale_row, w3_row, s3_row));
            }
        }
    }
    TwinOutput {
        codes,
        scales,
        weighted,
        gate_bf16,
        up_bf16,
        gate_f32,
        up_f32,
        gate_up_cost: [gate_cost, up_cost],
        traffic: traffic(tokens, false),
    }
}

/// The performance arm: this crate's model of the fused split-K kernel.
///
/// The schedule is the kernel's, not an approximation of it: one CTA per (token, 32-column tile);
/// for every live column the 160 block terms are computed once, folded per warp shard in ascending
/// block order, then combined by the shuffle butterfly ([splitk_fold]); the epilogue rounds each
/// projection to bf16 exactly as the chain's intermediate tensor does, runs the same weighted
/// SwiGLU, and quantises each 32-column tile with [act_quant_fp8_row] - one tile is exactly one MX
/// block, which is why the grid is 20 tiles wide and the padding columns quantise as zeros.
///
/// # Panics
/// Panics if a buffer is too short for the declared shape or an expert index is out of range.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn splitk_activation(
    act_fp8: &[u8],
    act_scales: &[u8],
    w1_fp4: &[u8],
    w1_scales: &[u8],
    w3_fp4: &[u8],
    w3_scales: &[u8],
    expert_ids: &[u32],
    routing_weights: &[f32],
    tokens: usize,
    experts: usize,
    limit: f32,
) -> SplitKOutput {
    let rows = routed_rows(tokens);
    assert_eq!(routing_weights.len(), rows, "one routing weight per row");
    assert_eq!(
        act_fp8.len(),
        tokens * HIDDEN,
        "the activation operand shape"
    );
    assert_eq!(
        act_scales.len(),
        tokens * SCALE_ROW_BYTES,
        "the activation scale grid"
    );
    assert_eq!(
        w1_fp4.len(),
        experts * WIDTH * WEIGHT_ROW_BYTES,
        "the w1 weight buffer"
    );
    assert_eq!(
        w3_fp4.len(),
        experts * WIDTH * WEIGHT_ROW_BYTES,
        "the w3 weight buffer"
    );
    assert_eq!(w1_scales.len(), experts * WIDTH * SCALE_ROW_BYTES);
    assert_eq!(w3_scales.len(), experts * WIDTH * SCALE_ROW_BYTES);

    let mut codes = vec![0u8; rows * WIDTH_PADDED];
    let mut scales = vec![0u8; rows * OUT_SCALE_ROW_BYTES];
    let mut weighted = vec![0.0f32; rows * WIDTH_PADDED];
    let mut gate_f32 = vec![0.0f32; rows * WIDTH];
    let mut up_f32 = vec![0.0f32; rows * WIDTH];
    let mut macs = 0u64;
    for token in 0..tokens {
        let act_row = &act_fp8[token * HIDDEN..][..HIDDEN];
        let act_scale_row = &act_scales[token * SCALE_ROW_BYTES..][..SCALE_ROW_BYTES];
        for tile in 0..TILES {
            let base_column = tile * TILE;
            // One CTA's epilogue: 32 weighted values per slot, quantised as one MX block.
            for slot in 0..TOP_K {
                let row = token * TOP_K + slot;
                let expert = expert_ids[row] as usize;
                assert!(expert < experts, "expert index {expert} is out of range");
                let mut block = [0.0f32; TILE];
                for (column_in_tile, value) in block.iter_mut().enumerate() {
                    let column = base_column + column_in_tile;
                    let (gate, up) = if column < WIDTH {
                        let weight_row = expert * WIDTH + column;
                        let w1_row = &w1_fp4[weight_row * WEIGHT_ROW_BYTES..][..WEIGHT_ROW_BYTES];
                        let s1_row = &w1_scales[weight_row * SCALE_ROW_BYTES..][..SCALE_ROW_BYTES];
                        let w3_row = &w3_fp4[weight_row * WEIGHT_ROW_BYTES..][..WEIGHT_ROW_BYTES];
                        let s3_row = &w3_scales[weight_row * SCALE_ROW_BYTES..][..SCALE_ROW_BYTES];
                        let gate_terms = block_terms(act_row, act_scale_row, w1_row, s1_row);
                        let up_terms = block_terms(act_row, act_scale_row, w3_row, s3_row);
                        macs += 2 * K_BLOCKS as u64 * QUANT_BLOCK as u64;
                        let gate = splitk_fold(&gate_terms);
                        let up = splitk_fold(&up_terms);
                        gate_f32[row * WIDTH + column] = gate;
                        up_f32[row * WIDTH + column] = up;
                        // The chain stores each projection as bf16 before the SwiGLU reads it.
                        (bf16_value(bf16_bits(gate)), bf16_value(bf16_bits(up)))
                    } else {
                        (0.0f32, 0.0f32)
                    };
                    *value = weighted_value(gate, up, limit, routing_weights[row]);
                }
                let (tile_codes, tile_scales) = act_quant_fp8_row(&block);
                codes[row * WIDTH_PADDED + base_column..][..TILE].copy_from_slice(&tile_codes);
                scales[row * OUT_SCALE_ROW_BYTES + tile] = tile_scales[0];
                weighted[row * WIDTH_PADDED + base_column..][..TILE].copy_from_slice(&block);
            }
        }
    }
    SplitKOutput {
        codes,
        scales,
        weighted,
        gate_f32,
        up_f32,
        traffic: traffic(tokens, true),
        macs,
    }
}

/// One element of the weighted SwiGLU epilogue: the reference's asymmetric clamp, the f32 silu
/// product, then ONE bf16 rounding of the routing-weighted value.
///
/// This is [crate::swiglu_ref::weighted_activation]'s per-element map, named here so the fused arm
/// calls the same expression the epilogue kernel implements. A test pins the two against each
/// other on a random row, so neither can drift.
#[must_use]
pub fn weighted_value(gate: f32, up: f32, limit: f32, weight: f32) -> f32 {
    let clamped_gate = gate.min(limit);
    let clamped_up = up.clamp(-limit, limit);
    let activated = swiglu_ref::silu(clamped_gate) * clamped_up;
    bf16_value(bf16_bits(activated * weight))
}

/// The declared D51 tolerance of the performance arm against the reference arm.
///
/// Declared BEFORE the harness measured the row (D51 amendment A, rule 3: the tolerance is not
/// inferred from a pass), from the arithmetic rather than from a run: a 160-term f32 sum
/// reassociated over eight shards changes each element by a few f32 ulps at most; a bf16 round can
/// therefore move by one bf16 ulp on a small fraction of elements; and the activation quantiser -
/// three mantissa bits and a power-of-two block scale - can change a code when a value sits on a
/// rounding boundary, and can change a whole block's scale when the block amax sits on a
/// power-of-two boundary. The bounds below are the envelope of those three effects with headroom,
/// not the measured values; the measured values are printed beside them and persisted by the
/// harness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// Max relative deviation of the f32 accumulators.
    pub f32_max_rel: f64,
    /// Mean relative deviation of the f32 accumulators.
    pub f32_mean_rel: f64,
    /// Fraction of bf16 gate/up elements that differ at all.
    pub bf16_diff_fraction: f64,
    /// Largest bf16 ulp distance allowed on any element.
    pub bf16_max_ulp: u32,
    /// Max relative deviation of the weighted activation before quantisation.
    pub weighted_max_rel: f64,
    /// Mean relative deviation of the weighted activation before quantisation.
    pub weighted_mean_rel: f64,
    /// Fraction of weighted-activation elements that differ at all.
    pub weighted_diff_fraction: f64,
    /// Max relative deviation of the DEQUANTISED activation operand (what the combine kernel reads).
    pub dequant_max_rel: f64,
    /// Mean relative deviation of the dequantised activation operand.
    pub dequant_mean_rel: f64,
    /// Fraction of E4M3 codes that differ.
    pub code_mismatch_fraction: f64,
    /// Fraction of 32-value blocks whose ue8m0 scale code differs.
    pub scale_mismatch_fraction: f64,
}

/// The declared tolerance. See [Tolerance] for how these numbers were chosen.
pub const DECLARED_TOLERANCE: Tolerance = Tolerance {
    f32_max_rel: 1.0e-5,
    f32_mean_rel: 1.0e-6,
    bf16_diff_fraction: 0.10,
    bf16_max_ulp: 1,
    weighted_max_rel: 1.0e-2,
    weighted_mean_rel: 1.0e-4,
    weighted_diff_fraction: 0.10,
    dequant_max_rel: 0.125,
    dequant_mean_rel: 1.0e-3,
    code_mismatch_fraction: 0.02,
    scale_mismatch_fraction: 0.01,
};

impl Tolerance {
    /// Checks a measured row against this tolerance.
    ///
    /// # Errors
    /// Returns a description of every bound the row exceeds; a pass returns `Ok(())` and says
    /// nothing, because the row itself is the record.
    pub fn check(&self, row: &DeviationRow) -> Result<(), String> {
        let mut failures = Vec::new();
        let mut bound = |name: &str, got: f64, limit: f64| {
            if got > limit {
                failures.push(format!("{name}: measured {got:.3e} > declared {limit:.3e}"));
            }
        };
        bound("f32_max_rel", row.f32_max_rel, self.f32_max_rel);
        bound("f32_mean_rel", row.f32_mean_rel, self.f32_mean_rel);
        bound(
            "bf16_diff_fraction",
            row.bf16_diff_fraction,
            self.bf16_diff_fraction,
        );
        bound(
            "bf16_max_ulp",
            f64::from(row.bf16_max_ulp),
            f64::from(self.bf16_max_ulp),
        );
        bound(
            "weighted_max_rel",
            row.weighted_max_rel,
            self.weighted_max_rel,
        );
        bound(
            "weighted_mean_rel",
            row.weighted_mean_rel,
            self.weighted_mean_rel,
        );
        bound(
            "weighted_diff_fraction",
            row.weighted_diff_fraction,
            self.weighted_diff_fraction,
        );
        bound("dequant_max_rel", row.dequant_max_rel, self.dequant_max_rel);
        bound(
            "dequant_mean_rel",
            row.dequant_mean_rel,
            self.dequant_mean_rel,
        );
        bound(
            "code_mismatch_fraction",
            row.code_mismatch_fraction,
            self.code_mismatch_fraction,
        );
        bound(
            "scale_mismatch_fraction",
            row.scale_mismatch_fraction,
            self.scale_mismatch_fraction,
        );
        bound("padding_mismatches", row.padding_mismatches as f64, 0.0);
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("; "))
        }
    }
}

/// The measured per-element deviation of the performance arm against the reference arm: the D51
/// per-kernel tolerance row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviationRow {
    /// Routed rows compared.
    pub rows: usize,
    /// Live (non-padding) elements compared, both projections.
    pub live_elements: usize,
    /// Max absolute deviation of the f32 accumulators.
    pub f32_max_abs: f64,
    /// Mean absolute deviation of the f32 accumulators.
    pub f32_mean_abs: f64,
    /// Max relative deviation of the f32 accumulators.
    pub f32_max_rel: f64,
    /// Mean relative deviation of the f32 accumulators.
    pub f32_mean_rel: f64,
    /// bf16 gate/up elements that differ from the twin's.
    pub bf16_differing: usize,
    /// Fraction of bf16 gate/up elements that differ.
    pub bf16_diff_fraction: f64,
    /// Largest bf16 ulp distance.
    pub bf16_max_ulp: u32,
    /// Max absolute deviation of the weighted activation.
    pub weighted_max_abs: f64,
    /// Mean absolute deviation of the weighted activation.
    pub weighted_mean_abs: f64,
    /// Max relative deviation of the weighted activation.
    pub weighted_max_rel: f64,
    /// Mean relative deviation of the weighted activation.
    pub weighted_mean_rel: f64,
    /// Weighted-activation elements that differ.
    pub weighted_differing: usize,
    /// Fraction of weighted-activation elements that differ.
    pub weighted_diff_fraction: f64,
    /// E4M3 codes that differ.
    pub code_mismatches: usize,
    /// Fraction of E4M3 codes that differ.
    pub code_mismatch_fraction: f64,
    /// Blocks whose ue8m0 scale differs.
    pub scale_mismatches: usize,
    /// Fraction of blocks whose scale differs.
    pub scale_mismatch_fraction: f64,
    /// Max absolute deviation of the dequantised activation operand.
    pub dequant_max_abs: f64,
    /// Mean absolute deviation of the dequantised activation operand.
    pub dequant_mean_abs: f64,
    /// Max relative deviation of the dequantised activation operand.
    pub dequant_max_rel: f64,
    /// Mean relative deviation of the dequantised activation operand.
    pub dequant_mean_rel: f64,
    /// Padding-column mismatches (both arms must quantise the zero tail identically).
    pub padding_mismatches: usize,
}

fn relative(deviation: f64, magnitude: f64) -> f64 {
    deviation / magnitude.max(1.0e-30)
}

/// Compares the two arms element by element and returns the tolerance row.
///
/// # Panics
/// Panics if the two outputs are not the same shape.
#[must_use]
pub fn deviation(twin: &TwinOutput, split: &SplitKOutput) -> DeviationRow {
    let rows = split.codes.len() / WIDTH_PADDED;
    assert_eq!(twin.codes.len(), split.codes.len(), "row counts agree");
    assert_eq!(twin.scales.len(), split.scales.len(), "scale counts agree");
    assert_eq!(twin.weighted.len(), split.weighted.len());
    assert_eq!(twin.gate_f32.len(), rows * WIDTH);
    assert_eq!(twin.up_f32.len(), rows * WIDTH);
    assert_eq!(split.gate_f32.len(), rows * WIDTH);

    let mut f32_max_abs = 0.0f64;
    let mut f32_sum_abs = 0.0f64;
    let mut f32_max_rel = 0.0f64;
    let mut f32_sum_rel = 0.0f64;
    for row in 0..rows {
        for column in 0..WIDTH {
            let index = row * WIDTH + column;
            for (split_value, twin_value) in [
                (split.gate_f32[index], twin.gate_f32[index]),
                (split.up_f32[index], twin.up_f32[index]),
            ] {
                let deviation = (f64::from(split_value) - f64::from(twin_value)).abs();
                f32_max_abs = f32_max_abs.max(deviation);
                f32_sum_abs += deviation;
                let rel = relative(deviation, f64::from(twin_value).abs());
                f32_max_rel = f32_max_rel.max(rel);
                f32_sum_rel += rel;
            }
        }
    }
    let live = rows * WIDTH * 2;

    let mut bf16_differing = 0usize;
    let mut bf16_max_ulp = 0u32;
    let mut weighted_differing = 0usize;
    let mut weighted_max_abs = 0.0f64;
    let mut weighted_sum_abs = 0.0f64;
    let mut weighted_max_rel = 0.0f64;
    let mut weighted_sum_rel = 0.0f64;
    let mut code_mismatches = 0usize;
    let mut scale_mismatches = 0usize;
    let mut dequant_max_abs = 0.0f64;
    let mut dequant_sum_abs = 0.0f64;
    let mut dequant_max_rel = 0.0f64;
    let mut dequant_sum_rel = 0.0f64;
    let mut padding_mismatches = 0usize;
    let mut weighted_counted = 0usize;
    for row in 0..rows {
        for block in 0..OUT_SCALE_ROW_BYTES {
            if twin.scales[row * OUT_SCALE_ROW_BYTES + block]
                != split.scales[row * OUT_SCALE_ROW_BYTES + block]
            {
                scale_mismatches += 1;
            }
        }
        for column in 0..WIDTH_PADDED {
            let index = row * WIDTH_PADDED + column;
            // The bf16 tensors: the twin's is the pinned reference's output, the performance
            // arm's is its f32 accumulator rounded once (padding columns are zero in both).
            let split_gate = if column < WIDTH {
                split.gate_f32[row * WIDTH + column]
            } else {
                0.0
            };
            let split_up = if column < WIDTH {
                split.up_f32[row * WIDTH + column]
            } else {
                0.0
            };
            for (twin_bits, split_bits) in [
                (twin.gate_bf16[index], bf16_bits(split_gate)),
                (twin.up_bf16[index], bf16_bits(split_up)),
            ] {
                if twin_bits != split_bits {
                    bf16_differing += 1;
                }
                bf16_max_ulp = bf16_max_ulp.max(bf16_ulp_distance(twin_bits, split_bits));
            }
            // The weighted activation, before quantisation, over all 640 columns.
            let twin_weighted = twin.weighted[index];
            let split_weighted = split.weighted[index];
            if split_weighted.to_bits() != twin_weighted.to_bits() {
                weighted_differing += 1;
            }
            let difference = (f64::from(split_weighted) - f64::from(twin_weighted)).abs();
            weighted_max_abs = weighted_max_abs.max(difference);
            weighted_sum_abs += difference;
            let rel = relative(difference, f64::from(twin_weighted).abs());
            weighted_max_rel = weighted_max_rel.max(rel);
            weighted_sum_rel += rel;
            weighted_counted += 1;
            if column >= WIDTH {
                continue;
            }
            let twin_scale = twin.scales[row * OUT_SCALE_ROW_BYTES + column / QUANT_BLOCK];
            let split_scale = split.scales[row * OUT_SCALE_ROW_BYTES + column / QUANT_BLOCK];
            if split.codes[index] != twin.codes[index] {
                code_mismatches += 1;
            }
            let twin_value = f64::from(dequantise_code(twin.codes[index], twin_scale));
            let split_value = f64::from(dequantise_code(split.codes[index], split_scale));
            let value_deviation = (split_value - twin_value).abs();
            dequant_max_abs = dequant_max_abs.max(value_deviation);
            dequant_sum_abs += value_deviation;
            let value_rel = relative(value_deviation, twin_value.abs());
            dequant_max_rel = dequant_max_rel.max(value_rel);
            dequant_sum_rel += value_rel;
        }
        // The zero tail: both arms must write zero codes and the quantiser's amax-floor scale.
        for column in WIDTH..WIDTH_PADDED {
            let index = row * WIDTH_PADDED + column;
            let block = column / QUANT_BLOCK;
            if split.codes[index] != 0
                || twin.codes[index] != 0
                || split.scales[row * OUT_SCALE_ROW_BYTES + block]
                    != twin.scales[row * OUT_SCALE_ROW_BYTES + block]
            {
                padding_mismatches += 1;
            }
        }
    }
    let bf16_elements = rows * WIDTH_PADDED * 2;
    DeviationRow {
        rows,
        live_elements: live,
        f32_max_abs,
        f32_mean_abs: f32_sum_abs / live as f64,
        f32_max_rel,
        f32_mean_rel: f32_sum_rel / live as f64,
        bf16_differing,
        bf16_diff_fraction: bf16_differing as f64 / bf16_elements as f64,
        bf16_max_ulp,
        weighted_max_abs,
        weighted_mean_abs: weighted_sum_abs / weighted_counted as f64,
        weighted_max_rel,
        weighted_mean_rel: weighted_sum_rel / weighted_counted as f64,
        weighted_differing,
        weighted_diff_fraction: weighted_differing as f64 / weighted_counted as f64,
        code_mismatches,
        code_mismatch_fraction: code_mismatches as f64 / live as f64,
        scale_mismatches,
        scale_mismatch_fraction: scale_mismatches as f64 / (rows * OUT_SCALE_ROW_BYTES) as f64,
        dequant_max_abs,
        dequant_mean_abs: dequant_sum_abs / live as f64,
        dequant_max_rel,
        dequant_mean_rel: dequant_sum_rel / live as f64,
        padding_mismatches,
    }
}

/// One E4M3 code at one ue8m0 block scale, as the combine kernel dequantises it.
#[must_use]
pub fn dequantise_code(code: u8, scale: u8) -> f32 {
    fp8_e4m3_decode(code) * ue8m0_decode(scale)
}

/// The monotonic distance between two bf16 patterns, in ulps (sign-magnitude corrected).
#[must_use]
pub fn bf16_ulp_distance(first: u16, second: u16) -> u32 {
    fn monotonic(bits: u16) -> i32 {
        if bits & 0x8000 != 0 {
            -i32::from(bits & 0x7FFF)
        } else {
            i32::from(bits)
        }
    }
    monotonic(first).abs_diff(monotonic(second))
}

/// The D51 ledger line: the measured row plus the declared bounds, one line per arm.
#[must_use]
pub fn summary(row: &DeviationRow) -> String {
    let declared = DECLARED_TOLERANCE;
    format!(
        "d51_tolerance: kernel=routed_gate_up_swiglu_splitk_sm121 rows={} live_elements={} \
         f32_max_abs={:.3e} f32_mean_abs={:.3e} f32_max_rel={:.3e} f32_mean_rel={:.3e} \
         bf16_differing={} bf16_diff_fraction={:.6} bf16_max_ulp={} \
         weighted_max_abs={:.3e} weighted_mean_abs={:.3e} weighted_max_rel={:.3e} \
         weighted_mean_rel={:.3e} weighted_differing={} weighted_diff_fraction={:.6} \
         code_mismatch_fraction={:.6} scale_mismatches={} scale_mismatch_fraction={:.6} \
         dequant_max_abs={:.3e} dequant_mean_abs={:.3e} dequant_max_rel={:.3e} \
         dequant_mean_rel={:.3e} padding_mismatches={} \
         declared_f32_max_rel={:.1e} declared_f32_mean_rel={:.1e} \
         declared_bf16_diff_fraction={:.2} declared_bf16_max_ulp={} \
         declared_weighted_max_rel={:.1e} declared_weighted_mean_rel={:.1e} \
         declared_weighted_diff_fraction={:.2} declared_dequant_max_rel={:.3} \
         declared_dequant_mean_rel={:.1e} declared_code_mismatch_fraction={:.2} \
         declared_scale_mismatch_fraction={:.2}",
        row.rows,
        row.live_elements,
        row.f32_max_abs,
        row.f32_mean_abs,
        row.f32_max_rel,
        row.f32_mean_rel,
        row.bf16_differing,
        row.bf16_diff_fraction,
        row.bf16_max_ulp,
        row.weighted_max_abs,
        row.weighted_mean_abs,
        row.weighted_max_rel,
        row.weighted_mean_rel,
        row.weighted_differing,
        row.weighted_diff_fraction,
        row.code_mismatch_fraction,
        row.scale_mismatches,
        row.scale_mismatch_fraction,
        row.dequant_max_abs,
        row.dequant_mean_abs,
        row.dequant_max_rel,
        row.dequant_mean_rel,
        row.padding_mismatches,
        declared.f32_max_rel,
        declared.f32_mean_rel,
        declared.bf16_diff_fraction,
        declared.bf16_max_ulp,
        declared.weighted_max_rel,
        declared.weighted_mean_rel,
        declared.weighted_diff_fraction,
        declared.dequant_max_rel,
        declared.dequant_mean_rel,
        declared.code_mismatch_fraction,
        declared.scale_mismatch_fraction,
    )
}

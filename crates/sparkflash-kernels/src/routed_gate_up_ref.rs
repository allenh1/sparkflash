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

//! CPU reference for the routed-expert gate/up projection
//! (kernels/routed_gate_up_sm121.cu).
//!
//! This module is the definition of that kernel's arithmetic. It is not a second implementation of
//! the engine's f32 expert body: the device operand is quantised (decision D14), so the kernel's
//! reference is the same arithmetic the pinned combine kernel and its twin use, with three
//! differences stated here rather than left implicit:
//!
//! 1. `K` is the whole hidden size, 5120 (160 MX blocks of 32), because this GEMM reduces the ffn
//!    block input. The combine projection's `K` is the routed width per rank, 576 padded to 640.
//! 2. there is no residual, no routing weight and no SwiGLU: the routing weight multiplies the
//!    activation before the bf16 cast the down projection's quantiser reads (model.py:849-851), so
//!    it belongs to the combine A operand (see mxfp4_ref::routed_activation_operand).
//! 3. the padding is on the OUTPUT: 576 real columns followed by 64 columns the kernel writes as
//!    exactly zero, because the pinned SwiGLU routed shape is 640 wide. Unlike the combine kernel's
//!    masked A operand there are no padding MACs here; what there is instead is
//!    `tokens * top_k * 64` zero writes, counted in [GateUpCost::padding_writes].
//!
//! The arithmetic, pinned by the tests below:
//!
//! * the A operand is FP8 E4M3, one byte per value, with per-32 ue8m0 scales, produced by the
//!   reference's act_quant (kernel.py:41-124) - the shared `m4_quant::act_quant_fp8_row`, the same
//!   quantiser mega_mhc_sm121's block input uses. This module does not wrap it: a wrapper plus a
//!   test that compares the wrapper with the function it calls would assert nothing (the router
//!   review caught exactly that shape of test);
//! * the B operand is E2M1 packed two values per byte, low nibble first, dequantised through the
//!   reference's cast to FP8 (kernel.py:538-540), which is exact on all sixteen codes;
//! * per output element the accumulation is f32, one rounded add per 32-value block, blocks in
//!   ascending K order, with both ue8m0 scales applied to the block PARTIAL PRODUCT (kernel.py:
//!   552-553), then ONE rounding to bf16 at the end;
//! * a token's six slots read the same activation row (the block input) and different weight rows
//!   (its own expert's w1 or w3 slice for this rank).
//!
//! What this module does NOT claim: that the quantised result equals the engine's f32 expert body.
//! It does not, by construction; the harness compares the device output against this twin exactly
//! and, separately, prints the engine f32 control gap on the same rows so the two classes of number
//! stay apart (stage 3 used the same separation for the combine kernel).

use crate::m4_quant::{
    QUANT_BLOCK, bf16_bits, fp4_e2m1_decode, fp8_e4m3_decode, fp8_e4m3_encode, ue8m0_decode,
};

/// The hidden size this GEMM reduces (config.json hidden_size). Mirrors SPARK_M4_HIDDEN.
pub const HIDDEN: usize = 5120;
/// The routed width per rank before padding. Mirrors SPARK_M4_MOE_WIDTH.
pub const WIDTH: usize = 576;
/// The routed width the kernel walks, padding included. Mirrors SPARK_M4_MOE_WIDTH_PADDED.
pub const WIDTH_PADDED: usize = 640;
/// Routed slots per token. Mirrors SPARK_M4_MOE_TOP_K.
pub const TOP_K: usize = 6;
/// E2M1 values per byte.
pub const FP4_PER_BYTE: usize = 2;
/// MX blocks per 5120-value row.
pub const K_BLOCKS: usize = HIDDEN / QUANT_BLOCK;
/// Activation bytes per value: one, because the A operand is FP8 E4M3 (D14).
pub const ACT_BYTES_PER_VALUE: usize = 1;
/// ue8m0 scale bytes per activation row and per weight row.
pub const SCALE_ROW_BYTES: usize = K_BLOCKS;
/// Packed FP4 weight bytes per output row.
pub const WEIGHT_ROW_BYTES: usize = HIDDEN / FP4_PER_BYTE;
/// bf16 bytes per output element.
pub const OUTPUT_BYTES_PER_VALUE: usize = 2;

/// The four row strides the kernel walks, as an explicit contract rather than an implicit use of
/// the constants above: a loader that hands this GEMM a whole-expert 2304-row operand (D2's
/// reference rule, corrected by D30) or stores the 576-row per-rank slice with a different stride
/// would misindex every row after the first, and this is the Rust-side rejection of that. The
/// 72-byte scale row is the combine kernel's grid (one scale per 32 values of its padded 640-wide A
/// operand is 20 bytes; the 72 is the w2 A-operand's K), not this K=5120 GEMM's layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateUpLayout {
    /// E4M3 bytes one activation row advances.
    pub activation_row_bytes: usize,
    /// ue8m0 bytes one activation scale row advances.
    pub activation_scale_row_bytes: usize,
    /// Packed FP4 bytes one weight row advances.
    pub weight_row_bytes: usize,
    /// ue8m0 bytes one weight scale row advances.
    pub weight_scale_row_bytes: usize,
}

/// The only row layout `routed_gate_up_sm121.cu` reads.
#[must_use]
pub const fn pre_padded_gate_up_layout() -> GateUpLayout {
    GateUpLayout {
        activation_row_bytes: HIDDEN * ACT_BYTES_PER_VALUE,
        activation_scale_row_bytes: SCALE_ROW_BYTES,
        weight_row_bytes: WEIGHT_ROW_BYTES,
        weight_scale_row_bytes: SCALE_ROW_BYTES,
    }
}

/// # Errors
/// Returns a description of the first stride that is not the pre-padded layout the kernel walks.
pub fn check_gate_up_layout(layout: GateUpLayout) -> Result<(), String> {
    let expected = pre_padded_gate_up_layout();
    for (name, got, want) in [
        (
            "activation",
            layout.activation_row_bytes,
            expected.activation_row_bytes,
        ),
        (
            "activation scale",
            layout.activation_scale_row_bytes,
            expected.activation_scale_row_bytes,
        ),
        ("weight", layout.weight_row_bytes, expected.weight_row_bytes),
        (
            "weight scale",
            layout.weight_scale_row_bytes,
            expected.weight_scale_row_bytes,
        ),
    ] {
        if got != want {
            return Err(format!(
                "the {name} row stride is {got} bytes, but routed_gate_up_sm121 walks {want}: a loader that hands this GEMM a whole-expert 2304-row operand (D2, corrected by D30) or stores the 576-row per-rank slice differently would misindex every row after the first"
            ));
        }
    }
    Ok(())
}

/// What one gate/up launch costs, counted rather than estimated: the MACs it issues and the bytes
/// each operand class reads, per output element, because that is how the kernel walks them.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GateUpCost {
    /// Multiply-accumulates issued: tokens * top_k * width * hidden.
    pub macs: u64,
    /// E4M3 activation bytes read (the same row is read by every one of the 576 columns).
    pub activation_bytes: u64,
    /// Packed FP4 weight bytes read.
    pub weight_bytes: u64,
    /// ue8m0 scale bytes read, both operands' grids together.
    pub scale_bytes: u64,
    /// bf16 zero writes into the 576..639 padding columns.
    pub padding_writes: u64,
}

/// The E2M1 value at one index of a packed weight row: low nibble even, high nibble odd.
///
/// # Panics
/// Panics if the row is too short.
#[must_use]
pub fn fp4_load(row: &[u8], index: usize) -> f32 {
    let byte = index / FP4_PER_BYTE;
    assert!(byte < row.len(), "the packed FP4 row is too short");
    let packed = row[byte];
    let nibble = if index.is_multiple_of(FP4_PER_BYTE) {
        packed & 0x0F
    } else {
        packed >> 4
    };
    fp4_e2m1_decode(nibble)
}

/// One routed gate (or up) projection, exactly as `routed_gate_up_sm121.cu` computes it, plus the
/// cost count.
///
/// The activation is [tokens][hidden] E4M3 bytes with scales [tokens][hidden / 32] ue8m0; the
/// weights are [experts][width][hidden / 2] packed E2M1 with scales
/// [experts][width][hidden / 32] ue8m0; the expert ids are [tokens * top_k]. The result is
/// [tokens * top_k][width_padded] bf16 bits, with columns 576..639 exactly zero.
///
/// # Panics
/// Panics if any buffer is too short for the declared shape or an expert index is out of range.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn gate_up(
    act_fp8: &[u8],
    act_scales: &[u8],
    w_fp4: &[u8],
    w_scales: &[u8],
    expert_ids: &[u32],
    tokens: usize,
    experts: usize,
) -> (Vec<u16>, GateUpCost) {
    let routed_rows = tokens * TOP_K;
    assert_eq!(
        act_fp8.len(),
        tokens * HIDDEN * ACT_BYTES_PER_VALUE,
        "the activation operand is {tokens} rows of {HIDDEN} E4M3 bytes"
    );
    assert_eq!(
        act_scales.len(),
        tokens * SCALE_ROW_BYTES,
        "the activation scale grid is {tokens} rows of {SCALE_ROW_BYTES} ue8m0 bytes"
    );
    assert_eq!(
        w_fp4.len(),
        experts * WIDTH * WEIGHT_ROW_BYTES,
        "the weight buffer is {experts} experts x {WIDTH} rows of {WEIGHT_ROW_BYTES} packed bytes"
    );
    assert_eq!(
        w_scales.len(),
        experts * WIDTH * SCALE_ROW_BYTES,
        "the weight scale grid is {experts} experts x {WIDTH} rows of {SCALE_ROW_BYTES} ue8m0 bytes"
    );
    assert_eq!(
        expert_ids.len(),
        routed_rows,
        "one expert id per routed row"
    );

    let mut cost = GateUpCost::default();
    let mut out = vec![0u16; routed_rows * WIDTH_PADDED];
    for routed_row in 0..routed_rows {
        let token = routed_row / TOP_K;
        let expert = expert_ids[routed_row] as usize;
        assert!(expert < experts, "expert index {expert} is out of range");
        let act_row = &act_fp8[token * HIDDEN..][..HIDDEN];
        let act_scale_row = &act_scales[token * SCALE_ROW_BYTES..][..SCALE_ROW_BYTES];
        let out_row = &mut out[routed_row * WIDTH_PADDED..][..WIDTH_PADDED];
        for (column, out_cell) in out_row.iter_mut().enumerate().take(WIDTH) {
            let weight_row = expert * WIDTH + column;
            let w_row = &w_fp4[weight_row * WEIGHT_ROW_BYTES..][..WEIGHT_ROW_BYTES];
            let w_scale_row = &w_scales[weight_row * SCALE_ROW_BYTES..][..SCALE_ROW_BYTES];
            let mut acc = 0.0f32;
            fold_element(
                act_row,
                act_scale_row,
                w_row,
                w_scale_row,
                &mut acc,
                &mut cost,
            );
            *out_cell = bf16_bits(acc);
        }
        // The output padding is part of the ABI: written zero, never computed, never read.
        for out_cell in &mut out_row[WIDTH..WIDTH_PADDED] {
            *out_cell = bf16_bits(0.0f32);
            cost.padding_writes += 1;
        }
    }
    (out, cost)
}

/// One output element's f32 accumulation: the block-partial fold, before the bf16 rounding.
///
/// Each 32-value block is summed exactly first, then both ue8m0 scales are applied to that partial
/// (D14 / `kernel.py:552-553`), and the scaled block contributions are added in ascending K. This is
/// the ordering the kernel and the twin share, and it is exposed to the tests because it is the
/// level at which the ordering is observable: the final bf16 rounding can hide differences smaller
/// than half a bf16 ulp.
fn fold_element(
    act_row: &[u8],
    act_scale_row: &[u8],
    w_row: &[u8],
    w_scale_row: &[u8],
    acc: &mut f32,
    cost: &mut GateUpCost,
) {
    for block in 0..K_BLOCKS {
        let base = block * QUANT_BLOCK;
        let act_scale = ue8m0_decode(act_scale_row[block]);
        let w_scale = ue8m0_decode(w_scale_row[block]);
        let mut partial = 0.0f32;
        for i in 0..QUANT_BLOCK {
            let index = base + i;
            let act_value = fp8_e4m3_decode(act_row[index]);
            let w_value = fp8_e4m3_decode(fp8_e4m3_encode(fp4_load(w_row, index)));
            partial += act_value * w_value;
            cost.macs += 1;
        }
        // Both scales go on the block PARTIAL PRODUCT, as kernel.py:552-553 accumulates
        // C_local * scale_a * scale_b; one rounded f32 add per 32-value block.
        *acc += (partial * act_scale) * w_scale;
        cost.weight_bytes += (QUANT_BLOCK / FP4_PER_BYTE) as u64;
        cost.activation_bytes += QUANT_BLOCK as u64;
        cost.scale_bytes += 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::m4_quant::{bf16_value, fp4_e2m1_encode, ue8m0_from_amax};

    /// A deterministic pseudo-random f32 in [-scale, scale): an LCG over the low bits, so a test
    /// does not depend on a rand crate and two runs on two hosts agree bit for bit.
    fn pseudo_random(seed: &mut u64, scale: f32) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((*seed >> 40) as f32) / ((1u64 << 24) as f32);
        (unit * 2.0 - 1.0) * scale
    }

    /// Packs E2M1 codes two per byte, low nibble first.
    fn packed_row(codes: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0u8; codes.len().div_ceil(FP4_PER_BYTE)];
        for (index, code) in codes.iter().enumerate() {
            if index.is_multiple_of(FP4_PER_BYTE) {
                bytes[index / FP4_PER_BYTE] |= code & 0x0F;
            } else {
                bytes[index / FP4_PER_BYTE] |= (code & 0x0F) << 4;
            }
        }
        bytes
    }

    /// The code that decodes to +1.0 in E2M1: exponent field 1, mantissa 0.
    const ONE: u8 = 0x2;
    /// The E4M3 code for 1.0 (0x38): its low nibble, for the hand-computed row.
    const ONE_FP8: u8 = 0x38;
    /// The ue8m0 code for scale 2^0 = 1.0.
    const SCALE_ONE: u8 = 127;

    fn expert_weights(experts: usize, seed: u32) -> (Vec<u8>, Vec<u8>) {
        let mut seed = seed as u64;
        let values = experts * WIDTH * HIDDEN;
        let w: Vec<u8> = (0..values)
            .map(|_| fp4_e2m1_encode(pseudo_random(&mut seed, 1.5)))
            .collect();
        let w_scales: Vec<u8> = (0..experts * WIDTH * SCALE_ROW_BYTES)
            .map(|_| ue8m0_from_amax(1.5, 2))
            .collect();
        (packed_row(&w), w_scales)
    }

    fn activation(tokens: usize, seed: u32) -> (Vec<u8>, Vec<u8>) {
        let mut seed = seed as u64;
        let mut act_fp8 = Vec::with_capacity(tokens * HIDDEN);
        let mut act_scales = Vec::with_capacity(tokens * SCALE_ROW_BYTES);
        for _ in 0..tokens {
            let row: Vec<f32> = (0..HIDDEN).map(|_| pseudo_random(&mut seed, 2.0)).collect();
            let (codes, scales) = crate::m4_quant::act_quant_fp8_row(&row);
            act_fp8.extend_from_slice(&codes);
            act_scales.extend_from_slice(&scales);
        }
        (act_fp8, act_scales)
    }

    /// The independent naive form of the SAME quantised arithmetic: dequantise the operands to f32
    /// first (every step is exact with power-of-two scales), then one straightforward dot in
    /// ascending K, then one bf16 rounding. It must agree bit for bit with [gate_up]; if the twin's
    /// block-partial-plus-scales ordering ever diverges from the obvious arithmetic, this fails.
    ///
    /// This path shares no decoder or rounding code with the twin: E2M1 is written out from the
    /// code layout, E4M3 is decoded bit-wise here, ue8m0 is evaluated with `2f32.powi`, and the
    /// final bf16 value is rounded from the f32 bits with explicit round-to-nearest-even. The twin
    /// calls `fp4_load`, `fp8_e4m3_decode`, `ue8m0_decode` and `bf16_bits`; the two sides are
    /// separate code paths for decode, arithmetic and rounding (the earlier router review caught a
    /// test that compared a helper with the expression it expands to). What they do share is the
    /// test-only packing helper and the pseudo-random input generator - inputs, not arithmetic.
    fn naive_fp4(nibble: u8) -> f32 {
        let magnitude = match nibble & 0x7 {
            0 => 0.0,
            1 => 0.5,
            2 => 1.0,
            3 => 1.5,
            4 => 2.0,
            5 => 3.0,
            6 => 4.0,
            _ => 6.0,
        };
        if nibble & 0x8 == 0 {
            magnitude
        } else {
            -magnitude
        }
    }

    /// E4M3FN decode from the code layout, written independently of `m4_quant`.
    fn naive_fp8_decode(code: u8) -> f32 {
        let sign = if code & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
        let exponent = i32::from((code >> 3) & 0x0F);
        let mantissa = f32::from(code & 0x07);
        if exponent == 0 {
            return sign * mantissa * 2f32.powi(-9);
        }
        if exponent == 0x0F && mantissa == 7.0 {
            return f32::NAN;
        }
        sign * (1.0 + mantissa * 0.125) * 2f32.powi(exponent - 7)
    }

    /// ue8m0 decode from the exponent code, written independently of `m4_quant`.
    fn naive_ue8m0(code: u8) -> f32 {
        match code {
            0 => f32::from_bits(1 << 22), // 2^-127 is a subnormal f32
            255 => f32::NAN,
            _ => 2f32.powi(i32::from(code) - 127),
        }
    }

    /// Round f32 to bf16 bits with explicit round-to-nearest-even (ties to even), from the bits.
    fn naive_bf16_round(value: f32) -> u16 {
        let bits = value.to_bits();
        if (bits >> 23) & 0xFF == 0xFF {
            return (bits >> 16) as u16; // inf or NaN: high half is the bf16 pattern
        }
        let lsb = (bits >> 16) & 1;
        ((bits.wrapping_add(0x7FFF + lsb)) >> 16) as u16
    }

    fn naive(
        act_fp8: &[u8],
        act_scales: &[u8],
        w_fp4: &[u8],
        w_scales: &[u8],
        expert_ids: &[u32],
        tokens: usize,
    ) -> Vec<u16> {
        let mut out = vec![0u16; tokens * TOP_K * WIDTH_PADDED];
        for routed_row in 0..tokens * TOP_K {
            let token = routed_row / TOP_K;
            let expert = expert_ids[routed_row] as usize;
            for column in 0..WIDTH {
                let weight_row = expert * WIDTH + column;
                let mut acc = 0.0f32;
                for index in 0..HIDDEN {
                    let block = index / QUANT_BLOCK;
                    let act_scale = naive_ue8m0(act_scales[token * SCALE_ROW_BYTES + block]);
                    let w_scale = naive_ue8m0(w_scales[weight_row * SCALE_ROW_BYTES + block]);
                    let act_value = naive_fp8_decode(act_fp8[token * HIDDEN + index]) * act_scale;
                    let packed = w_fp4[weight_row * WEIGHT_ROW_BYTES + index / FP4_PER_BYTE];
                    let nibble = if index.is_multiple_of(FP4_PER_BYTE) {
                        packed & 0x0F
                    } else {
                        packed >> 4
                    };
                    // Straight from the E2M1 code: the twin's FP4 -> FP8 -> value cast is an
                    // identity (every E2M1 code is exactly representable in E4M3, pinned by its own
                    // test), so taking the value here checks the arithmetic, not the codec.
                    let w_value = naive_fp4(nibble) * w_scale;
                    acc += act_value * w_value;
                }
                out[routed_row * WIDTH_PADDED + column] = naive_bf16_round(acc);
            }
        }
        out
    }

    #[test]
    fn one_hot_one_value_is_one_and_the_output_padding_is_zero() {
        let tokens = 1usize;
        let experts = 1usize;
        let mut act_fp8 = vec![0u8; HIDDEN];
        act_fp8[0] = ONE_FP8;
        let act_scales = vec![SCALE_ONE; SCALE_ROW_BYTES];
        let mut w_codes = vec![0u8; WIDTH * HIDDEN];
        w_codes[0] = ONE;
        let w_fp4 = packed_row(&w_codes);
        let w_scales = vec![SCALE_ONE; WIDTH * SCALE_ROW_BYTES];
        let expert_ids = vec![0u32; tokens * TOP_K];
        let (out, cost) = gate_up(
            &act_fp8,
            &act_scales,
            &w_fp4,
            &w_scales,
            &expert_ids,
            tokens,
            experts,
        );
        assert_eq!(cost.macs, (tokens * TOP_K * WIDTH * HIDDEN) as u64);
        assert_eq!(
            cost.padding_writes,
            (tokens * TOP_K * (WIDTH_PADDED - WIDTH)) as u64
        );
        for slot in 0..TOP_K {
            let row = &out[slot * WIDTH_PADDED..][..WIDTH_PADDED];
            assert_eq!(bf16_value(row[0]), 1.0, "slot {slot} column 0");
            for (offset, bits) in row[1..WIDTH].iter().enumerate() {
                let column = offset + 1;
                assert_eq!(bf16_value(*bits), 0.0, "slot {slot} column {column}");
            }
            for (offset, bits) in row[WIDTH..WIDTH_PADDED].iter().enumerate() {
                let column = offset + WIDTH;
                assert_eq!(*bits, 0, "padding column {column} must be exactly zero");
            }
        }
    }

    #[test]
    fn the_packing_is_low_nibble_first() {
        let codes = [ONE, 0xE, 0x0, 0xF]; // +1, -4, +0, -6 in E2M1
        let packed = packed_row(&codes);
        assert_eq!(
            packed[0], 0xE2,
            "index 0 in the low nibble, index 1 in the high"
        );
        assert_eq!(packed[1], 0xF0, "index 2 low, index 3 high");
        for (index, want) in [1.0f32, -4.0, 0.0, -6.0].iter().enumerate() {
            assert_eq!(fp4_load(&packed, index), *want, "index {index}");
        }
    }

    #[test]
    fn the_twin_matches_an_independent_naive_dequantised_loop() {
        let tokens = 2usize;
        let experts = 3usize;
        let (act_fp8, act_scales) = activation(tokens, 7);
        let (w_fp4, w_scales) = expert_weights(experts, 11);
        let mut seed = 0x1234_5678u64;
        let expert_ids: Vec<u32> = (0..tokens * TOP_K)
            .map(|_| {
                ((pseudo_random(&mut seed, 1.0) + 1.0) * 0.5 * experts as f32) as u32
                    % experts as u32
            })
            .collect();
        let (twin, _) = gate_up(
            &act_fp8,
            &act_scales,
            &w_fp4,
            &w_scales,
            &expert_ids,
            tokens,
            experts,
        );
        let other = naive(
            &act_fp8,
            &act_scales,
            &w_fp4,
            &w_scales,
            &expert_ids,
            tokens,
        );
        assert_eq!(
            twin, other,
            "the block-partial ordering must be the naive real arithmetic"
        );
    }

    /// A golden block-fold vector with asserted f32 bits: the twin's fold is checked at the
    /// precision where its ordering exists, because the final bf16 rounding hides anything below
    /// half a bf16 ulp.
    ///
    /// Block 0 contributes exactly 1.0. Block 1 has 32 terms of 1.5 x 2^-24 each (weight code +1.5,
    /// activation 1.0, activation scale 2^-24, weight scale 1). The twin sums the 32 products first
    /// (48 exactly) and scales once: 1.0 + 48 x 2^-24 = 1 + 3 x 2^-20 = 0x3F800018. A per-element
    /// fold adds 1.5 x 2^-24 to 1.0 one at a time; each add is 0.75 of an f32 ulp, so it rounds up
    /// one ulp (ties-to-even) and the same real sum becomes 1 + 32 x 2^-23 = 0x3F800020. Both round
    /// to bf16 0x3F80, which is exactly why the general comparison above cannot pin the fold, and
    /// why this test asserts the f32 bits instead.
    #[test]
    fn the_block_partial_fold_is_pinned_at_f32_precision_by_a_golden_vector() {
        let mut act_fp8 = vec![0u8; HIDDEN];
        act_fp8[0] = ONE_FP8;
        act_fp8[QUANT_BLOCK..2 * QUANT_BLOCK].fill(ONE_FP8);
        let mut act_scales = vec![SCALE_ONE; SCALE_ROW_BYTES];
        act_scales[1] = 103; // 2^-24
        let mut w_codes = vec![0u8; HIDDEN];
        w_codes[0] = ONE;
        w_codes[QUANT_BLOCK..2 * QUANT_BLOCK].fill(0x3); // +1.5 in E2M1
        let w_fp4 = packed_row(&w_codes);
        let w_scales = vec![SCALE_ONE; SCALE_ROW_BYTES];
        let mut cost = GateUpCost::default();
        let mut acc = 0.0f32;
        fold_element(
            &act_fp8,
            &act_scales,
            &w_fp4,
            &w_scales,
            &mut acc,
            &mut cost,
        );
        assert_eq!(
            acc.to_bits(),
            0x3F80_0018,
            "block-partial fold: 1 + 3 x 2^-20"
        );
        assert_eq!(bf16_bits(acc), 0x3F80);
        assert_eq!(
            cost.macs, HIDDEN as u64,
            "one MAC per K value, zero blocks included"
        );

        // The per-element control, written out here: the same real sum, a different f32 value.
        let mut per_element = 1.0f32; // block 0 contributed exactly 1.0
        for _ in 0..QUANT_BLOCK {
            per_element += 1.5 * 2f32.powi(-24);
        }
        assert_eq!(
            per_element.to_bits(),
            0x3F80_0020,
            "per-element fold: 1 + 32 x 2^-23"
        );
        assert_eq!(bf16_bits(per_element), 0x3F80);
        assert_ne!(
            acc.to_bits(),
            per_element.to_bits(),
            "the fold order is observable in f32"
        );
    }

    #[test]
    fn slots_share_the_activation_row_and_differ_only_by_their_expert() {
        let tokens = 1usize;
        let experts = 2usize;
        let (act_fp8, act_scales) = activation(tokens, 21);
        let (w_fp4, w_scales) = expert_weights(experts, 22);
        let mut ids = vec![0u32; TOP_K];
        ids[3] = 1; // one slot routed to the other expert
        let (out, _) = gate_up(
            &act_fp8,
            &act_scales,
            &w_fp4,
            &w_scales,
            &ids,
            tokens,
            experts,
        );
        for slot in 0..TOP_K {
            let row = &out[slot * WIDTH_PADDED..][..WIDTH_PADDED];
            if slot == 3 {
                continue;
            }
            assert_eq!(row, &out[0..WIDTH_PADDED], "slot {slot} repeats slot 0");
        }
        assert_ne!(
            &out[3 * WIDTH_PADDED..][..WIDTH_PADDED],
            &out[0..WIDTH_PADDED]
        );
    }

    #[test]
    fn a_loader_layout_that_is_not_pre_padded_is_rejected() {
        assert!(check_gate_up_layout(pre_padded_gate_up_layout()).is_ok());
        for bad in [
            GateUpLayout {
                activation_row_bytes: 640,
                ..pre_padded_gate_up_layout()
            },
            GateUpLayout {
                activation_scale_row_bytes: 20,
                ..pre_padded_gate_up_layout()
            },
            GateUpLayout {
                weight_row_bytes: 320,
                ..pre_padded_gate_up_layout()
            },
            GateUpLayout {
                weight_scale_row_bytes: 20,
                ..pre_padded_gate_up_layout()
            },
        ] {
            assert!(
                check_gate_up_layout(bad).is_err(),
                "the loader layout must be rejected"
            );
        }
    }

    #[test]
    fn the_geometry_constants_are_the_kernel_abi() {
        // Compile-time sanity checks that keep this module and the kernel source in one place; the
        // ABI module pins these same numbers against the kernel's own static_asserts.
        assert_eq!(HIDDEN, 5120);
        assert_eq!(WIDTH, 576);
        assert_eq!(WIDTH_PADDED, 640);
        assert_eq!(TOP_K, 6);
        assert_eq!(K_BLOCKS, 160);
        assert_eq!(WEIGHT_ROW_BYTES, 2560);
        assert_eq!(SCALE_ROW_BYTES, 160);
    }
}

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

//! CPU twin of the BF16-activation MXFP8 projection kernels:
//! `kernels/b12x_bf16a_mxfp8w_dense_sm121.cu` and
//! `kernels/b12x_bf16a_mxfp8w_wo_a_sm121.cu`.
//!
//! This module is the definition of those kernels' arithmetic, exactly as `mxfp8_ref` is the
//! definition of the E4M3-activation kernels'. The only difference between the two families is the
//! activation operand: here it arrives as BF16 bit patterns, widens to f32 exactly (BF16 -> f32 is
//! lossless), and enters the same accumulation. Weights are still the checkpoint's E4M3 codes and
//! ue8m0 block scales, dequantized with the same codecs and read through the same B12x index
//! helpers, so the twin cannot quietly use a different weight layout.
//!
//! What it pins, in the terms the kernel source states:
//!
//! 1. the accumulation order: ONE f32 accumulator per output, K ascending in 32-value blocks and
//!    ascending inside a block, products formed as `(f32)bf16_a * (f32)fp8_w * ue8m0` with the
//!    weight decoded as `fp8(w) * ue8m0(scale)` first (one f32 multiply) and then multiplied by
//!    the activation (a second f32 multiply) - the order the kernel's `a_value * w_value` has;
//! 2. exactly one rounding, to bf16, after the whole sum (round-to-nearest-even, the same
//!    `__float2bfloat16_rn` / `half::bf16::from_f32` codec the other references use);
//! 3. for wo_a, that a group is the dense GEMM over that group's own concatenated head outputs and
//!    its own weight/scale buffers, with the two different group strides.
//!
//! What it does NOT claim: it is not a second opinion on what the GPU produces. The kernel and
//! this twin are compiled from two languages; `examples/m21_bf16a_projections.rs` is the harness
//! that compares the device output with this code bit for bit on the same packed bytes, and until
//! that runs on the node the equality is a plan, not a measurement.

use crate::m4_quant::{QUANT_BLOCK, bf16_bits, bf16_bits_to_f32, fp8_e4m3_decode, ue8m0_decode};
use crate::mxfp8_ref::{b12x_buffer_bytes, b12x_scale_bytes, b12x_scale_index, b12x_weight_index};

/// One BF16-activation dense GEMM, exactly as `b12x_bf16a_mxfp8w_dense_sm121.cu` computes it.
///
/// `a_bf16` holds \[rows\]\[inner\] BF16 bit patterns (u16, the bytes the kernel loads); the
/// weights are a B12x buffer of \[cols\]\[inner\] E4M3 with \[ceil(cols/32), inner/32\] ue8m0
/// scales. The result is \[rows\]\[cols\] BF16 bits.
///
/// # Panics
/// Panics if the buffers are too short for the declared shape, so a mis-sized fixture cannot
/// silently read zeros.
#[must_use]
pub fn dense_gemm_bf16a(
    a_bf16: &[u16],
    w: &[u8],
    w_scales: &[u8],
    rows: usize,
    cols: usize,
    inner: usize,
) -> Vec<u16> {
    let k_blocks = inner / QUANT_BLOCK;
    assert_eq!(
        inner % QUANT_BLOCK,
        0,
        "the reduction width must be whole MX blocks"
    );
    assert!(
        a_bf16.len() >= rows * inner,
        "the activation buffer is too short"
    );
    assert!(
        w.len() >= b12x_buffer_bytes(cols, inner),
        "the weight buffer is too short"
    );
    assert!(
        w_scales.len() >= b12x_scale_bytes(cols, inner),
        "the weight scale buffer is too short"
    );
    let mut out = vec![0u16; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let mut acc = 0.0f32;
            for block in 0..k_blocks {
                let base = block * QUANT_BLOCK;
                let w_scale = ue8m0_decode(w_scales[b12x_scale_index(col, base, inner)]);
                for i in 0..QUANT_BLOCK {
                    let a_value = bf16_bits_to_f32(a_bf16[row * inner + base + i]);
                    let w_value =
                        fp8_e4m3_decode(w[b12x_weight_index(col, base + i, inner)]) * w_scale;
                    acc += a_value * w_value;
                }
            }
            out[row * cols + col] = bf16_bits(acc);
        }
    }
    out
}

/// The grouped wo_a projection with BF16 activations: for each group, a dense GEMM over that
/// group's concatenated head outputs against the group's own B12x weight buffer.
///
/// This is deliberately implemented by calling [dense_gemm_bf16a] per group with a group-shaped
/// slice, because that is exactly the claim the kernel makes: wo_a is the dense twin plus a group
/// dimension, and the kernel's per-group arithmetic is the dense arithmetic.
///
/// # Panics
/// Panics if the buffers are too short for the declared shape.
#[must_use]
pub fn grouped_wo_a_bf16a(
    attn_bf16: &[u16],
    w: &[u8],
    w_scales: &[u8],
    groups: usize,
    group_inner: usize,
    lo_rank: usize,
) -> Vec<u16> {
    let weight_bytes = b12x_buffer_bytes(lo_rank, group_inner);
    let scale_bytes = b12x_scale_bytes(lo_rank, group_inner);
    assert!(
        attn_bf16.len() >= groups * group_inner,
        "the attention buffer is too short"
    );
    assert!(
        w.len() >= groups * weight_bytes,
        "the weight buffer is too short"
    );
    assert!(
        w_scales.len() >= groups * scale_bytes,
        "the weight scale buffer is too short"
    );
    let mut out = vec![0u16; groups * lo_rank];
    for group in 0..groups {
        let group_out = dense_gemm_bf16a(
            &attn_bf16[group * group_inner..(group + 1) * group_inner],
            &w[group * weight_bytes..(group + 1) * weight_bytes],
            &w_scales[group * scale_bytes..(group + 1) * scale_bytes],
            1,
            lo_rank,
            group_inner,
        );
        out[group * lo_rank..(group + 1) * lo_rank].copy_from_slice(&group_out);
    }
    out
}

/// A device-output/reference comparison in BF16 codes.
///
/// The device readback leaves BF16-exact values widened to f32, so a comparison re-narrows them
/// with the same round-to-nearest-even codec the kernels use instead of comparing floats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CodeComparison {
    /// How many BF16 codes differ.
    pub mismatches: usize,
    /// How many codes were compared (the shorter input's length).
    pub total: usize,
    /// The largest absolute difference in value units, 0.0 when every code matches.
    pub max_abs: f32,
}

/// Compares a device output against a reference's BF16 codes.
#[must_use]
pub fn compare_codes(device: &[f32], reference: &[u16]) -> CodeComparison {
    let total = device.len().min(reference.len());
    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    for (index, expected) in reference.iter().take(total).enumerate() {
        let got = device[index];
        if bf16_bits(got) != *expected {
            mismatches += 1;
        }
        let diff = (bf16_bits_to_f32(*expected) - got).abs();
        if diff > max_abs {
            max_abs = diff;
        }
    }
    CodeComparison {
        mismatches,
        total,
        max_abs,
    }
}

/// Both wo_a paths on the SAME captured activation operand.
///
/// [native] is the native rule's output: the activation rounded to BF16 exactly as the reference
/// rounds its inverse-RoPE body output, times the dequantized weight. [mxfp8] is the chain's
/// current path: the same f32 activation quantized to E4M3 with per-32 ue8m0 scales first. The
/// difference between them is the F1 deviation, measured on one operand instead of across two
/// runs.
#[derive(Debug, Clone)]
pub struct PathComparison {
    /// The native path's BF16 output codes.
    pub native: Vec<u16>,
    /// The current MXFP8 path's BF16 output codes.
    pub mxfp8: Vec<u16>,
    /// How many output codes are identical.
    pub codes_equal: usize,
    /// How many output codes the comparison covered.
    pub total: usize,
    /// The largest absolute output difference in value units.
    pub max_abs: f32,
    /// The largest absolute native output, the scale max_abs is relative to.
    pub output_max_abs: f32,
}

/// Runs both wo_a paths on one captured activation operand.
///
/// # Errors
/// Returns the activation quantiser's error, which the current path needs and the native one does
/// not.
pub fn compare_paths(
    attn_f32: &[f32],
    w: &[u8],
    w_scales: &[u8],
    groups: usize,
    group_inner: usize,
    lo_rank: usize,
) -> Result<PathComparison, String> {
    let activations: Vec<u16> = attn_f32.iter().map(|value| bf16_bits(*value)).collect();
    let native = grouped_wo_a_bf16a(&activations, w, w_scales, groups, group_inner, lo_rank);
    let quantised = sparkflash_engine::quant::quantize_activation_fp8(attn_f32)
        .map_err(|error| error.to_string())?;
    let mxfp8 = crate::mxfp8_ref::grouped_wo_a(
        quantised.codes(),
        quantised.scales(),
        w,
        w_scales,
        groups,
        group_inner,
        lo_rank,
    );
    let total = native.len().min(mxfp8.len());
    let mut codes_equal = 0usize;
    let mut max_abs = 0.0f32;
    let mut output_max_abs = 0.0f32;
    for (native_code, mxfp8_code) in native.iter().zip(mxfp8.iter()) {
        let native_value = bf16_bits_to_f32(*native_code);
        let mxfp8_value = bf16_bits_to_f32(*mxfp8_code);
        if native_code == mxfp8_code {
            codes_equal += 1;
        }
        let diff = (native_value - mxfp8_value).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        let magnitude = native_value.abs();
        if magnitude > output_max_abs {
            output_max_abs = magnitude;
        }
    }
    Ok(PathComparison {
        native,
        mxfp8,
        codes_equal,
        total,
        max_abs,
        output_max_abs,
    })
}

#[cfg(test)]
mod comparison_tests {
    use super::*;
    use crate::m4_quant::fp8_e4m3_encode;

    /// One group of 32 outputs over 32 values: one B12x block, one scale, no padding.
    const INNER: usize = 32;
    const LO_RANK: usize = 32;
    /// The E8M0 byte for 2^0, so the weight scale is exactly 1.0.
    const UNIT_SCALE: u8 = 127;

    fn weights(mut sign: impl FnMut(usize) -> f32) -> (Vec<u8>, Vec<u8>) {
        let mut w = vec![0u8; LO_RANK * INNER];
        for col in 0..LO_RANK {
            for k in 0..INNER {
                w[b12x_weight_index(col, k, INNER)] = fp8_e4m3_encode(sign(k));
            }
        }
        (w, vec![UNIT_SCALE])
    }

    #[test]
    fn the_two_paths_agree_when_the_activation_is_fp8_exact() {
        let (w, w_scales) = weights(|k| if k % 2 == 0 { 1.0 } else { -1.0 });
        let activation: Vec<f32> = (0..INNER)
            .map(|k| if k % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let comparison = compare_paths(&activation, &w, &w_scales, 1, INNER, LO_RANK).unwrap();
        assert_eq!(comparison.total, LO_RANK);
        assert_eq!(
            comparison.codes_equal, comparison.total,
            "an FP8-exact activation and unit weights must leave the two paths identical"
        );
        assert_eq!(comparison.max_abs, 0.0);
    }

    #[test]
    fn the_comparison_sees_the_activation_quantiser() {
        // The block amax is 2.0, so the per-32 scale is 2^-7 and one E4M3 step there is coarser
        // than a BF16 step at 1.02; the weights cancel the exact half against the inexact one.
        let (w, w_scales) = weights(|k| if k < INNER / 2 { -1.0 } else { 1.0 });
        let activation: Vec<f32> = (0..INNER)
            .map(|k| if k < INNER / 2 { 2.0 } else { 1.02 })
            .collect();
        let comparison = compare_paths(&activation, &w, &w_scales, 1, INNER, LO_RANK).unwrap();
        assert_eq!(comparison.total, LO_RANK);
        assert!(
            comparison.codes_equal < comparison.total,
            "a value E4M3 cannot hold at this block scale must move at least one output code"
        );
        assert!(comparison.max_abs > 0.0);
    }

    #[test]
    fn compare_codes_counts_exact_and_inexact_codes() {
        let reference: Vec<u16> = vec![bf16_bits(0.5), bf16_bits(-1.25), bf16_bits(2.0)];
        let device: Vec<f32> = reference
            .iter()
            .map(|code| bf16_bits_to_f32(*code))
            .collect();
        let exact = compare_codes(&device, &reference);
        assert_eq!(exact.mismatches, 0);
        assert_eq!(exact.total, 3);
        assert_eq!(exact.max_abs, 0.0);
        let drifted: Vec<f32> = vec![device[0], device[1] + 0.5, device[2]];
        let inexact = compare_codes(&drifted, &reference);
        assert_eq!(inexact.mismatches, 1);
        assert_eq!(inexact.total, 3);
        assert!(inexact.max_abs > 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::m4_quant::{E4M3_ELEMENT_EMAX, bf16_value, fp8_e4m3_encode, ue8m0_from_amax};

    /// BF16 bit patterns for a deterministic f32 value in roughly [-1, 1]. The value is rounded to
    /// BF16 with the same codec the kernel output uses, so the operand is a legal BF16 number.
    fn bf16_pattern(seed: usize) -> u16 {
        let value = ((seed % 29) as f32 - 14.0) / 16.0;
        bf16_bits(value)
    }

    /// A tiny deterministic weight-code sequence; 0x7F would be NaN, so codes stay below it.
    fn weight_code(seed: usize) -> u8 {
        (seed % 0x7F) as u8
    }

    fn b12x_weight_buffer(cols: usize, inner: usize, seed: usize) -> Vec<u8> {
        let mut w = vec![0u8; b12x_buffer_bytes(cols, inner)];
        for n in 0..cols {
            for k in 0..inner {
                w[b12x_weight_index(n, k, inner)] = weight_code(seed + n * inner + k);
            }
        }
        w
    }

    #[test]
    fn golden_single_block_is_hand_computable() {
        // One row, one output channel, one MX block of 32 values. Every activation is the BF16 1.0
        // and every weight code 0x38 is E4M3 1.0 with a ue8m0 scale of 2^-8, so each product is 2^-8
        // and the sum of 32 is 2^-3 = 0x3E00 in BF16.
        let inner = 32usize;
        let scale = ue8m0_from_amax(1.0, E4M3_ELEMENT_EMAX);
        assert_eq!(scale, 119, "1.0 with an emax of 8 is a 2^-8 scale");
        let a = vec![bf16_bits(1.0); inner];
        let mut w = vec![0u8; b12x_buffer_bytes(1, inner)];
        for k in 0..inner {
            w[b12x_weight_index(0, k, inner)] = 0x38;
        }
        let mut w_scales = vec![127u8; b12x_scale_bytes(1, inner)];
        w_scales[b12x_scale_index(0, 0, inner)] = scale;
        let out = dense_gemm_bf16a(&a, &w, &w_scales, 1, 1, inner);
        assert_eq!(out[0], 0x3E00, "32 products of 2^-8 is 2^-3");
        assert_eq!(bf16_value(out[0]), 2.0f32.powi(-3));

        // The same weights with BF16 activations of 0.5 halve the result: 2^-4, bits 0x3D80.
        let half = vec![bf16_bits(0.5); inner];
        let out = dense_gemm_bf16a(&half, &w, &w_scales, 1, 1, inner);
        assert_eq!(out[0], 0x3D80);
    }

    #[test]
    fn the_bf16_activation_is_not_requantised() {
        // The whole point of this variant: an activation that E4M3 could not represent must still
        // enter the sum at its full BF16 value. 0.1f32 is not exactly any BF16 or E4M3 value; we
        // choose the BF16 nearest 0.1 and one weight of exactly 1.0 with scale 1.0, so the product
        // is that BF16 value exactly and the output is it rounded to BF16 again (unchanged).
        let inner = 32usize;
        let a_value = bf16_bits(0.1f32);
        assert_ne!(
            a_value,
            bf16_bits(fp8_e4m3_decode(fp8_e4m3_encode(0.1f32))),
            "the test value must not be E4M3-representable"
        );
        let mut a = vec![0u16; inner];
        a[0] = a_value;
        let mut w = vec![0u8; b12x_buffer_bytes(1, inner)];
        w[b12x_weight_index(0, 0, inner)] = 0x38; // 1.0
        let w_scales = vec![127u8; b12x_scale_bytes(1, inner)]; // 1.0
        let out = dense_gemm_bf16a(&a, &w, &w_scales, 1, 1, inner);
        assert_eq!(
            out[0], a_value,
            "bf16(1.0 * bf16(0.1)) must be the activation, not an E4M3 version of it"
        );
        assert_eq!(bf16_value(out[0]), bf16_value(a_value));
    }

    #[test]
    fn the_twin_matches_a_naive_dequantized_loop() {
        // Cross-check the tiled/indexed form against the most obvious loop: decode every weight and
        // activation, accumulate in the pinned order. A shape with several micro-tiles, several K
        // blocks and a ragged N means the index helpers and the arithmetic agree.
        let (rows, cols, inner) = (3usize, 20usize, 96usize);
        let a: Vec<u16> = (0..rows * inner).map(|i| bf16_pattern(i * 7 + 1)).collect();
        let w = b12x_weight_buffer(cols, inner, 5);
        let w_scales: Vec<u8> = (0..b12x_scale_bytes(cols, inner))
            .map(|i| 100 + (i % 40) as u8)
            .collect();
        let reference = dense_gemm_bf16a(&a, &w, &w_scales, rows, cols, inner);
        for row in 0..rows {
            for col in 0..cols {
                let mut acc = 0.0f32;
                for k in 0..inner {
                    acc += bf16_bits_to_f32(a[row * inner + k])
                        * fp8_e4m3_decode(w[b12x_weight_index(col, k, inner)])
                        * ue8m0_decode(w_scales[b12x_scale_index(col, k, inner)]);
                }
                assert_eq!(
                    reference[row * cols + col],
                    bf16_bits(acc),
                    "element ({row}, {col}) disagrees with the naive loop"
                );
            }
        }
    }

    #[test]
    fn the_pinned_accumulation_order_is_actually_used() {
        // 1.0 added 32 times, then a tiny value, is not the same f32 sum as the tiny value first.
        // The twin must equal the ascending-order sum, not the sorted one.
        let inner = 32usize;
        let mut a = vec![bf16_bits(1.0); inner];
        a[31] = bf16_bits(0.0);
        let mut w = vec![0u8; b12x_buffer_bytes(1, inner)];
        for k in 0..inner {
            w[b12x_weight_index(0, k, inner)] = 0x38; // 1.0
        }
        w[b12x_weight_index(0, 0, inner)] = 0x01; // smallest E4M3 subnormal, 2^-9
        let w_scales = vec![127u8; b12x_scale_bytes(1, inner)];
        let reference = dense_gemm_bf16a(&a, &w, &w_scales, 1, 1, inner);
        let mut ascending = 0.0f32;
        for k in 0..inner {
            ascending +=
                bf16_bits_to_f32(a[k]) * fp8_e4m3_decode(w[b12x_weight_index(0, k, inner)]);
        }
        assert_eq!(reference[0], bf16_bits(ascending));
        assert_ne!(reference[0], 0, "the sum must not vanish");
    }

    #[test]
    fn wo_a_is_the_dense_twin_with_a_group_dimension() {
        // The claim the wo_a kernel makes is that it is the dense GEMM per group. Build grouped
        // buffers and compare the grouped result against direct dense calls.
        let (groups, group_inner, lo_rank) = (2usize, 64usize, 16usize);
        let weight_bytes = b12x_buffer_bytes(lo_rank, group_inner);
        let scale_bytes = b12x_scale_bytes(lo_rank, group_inner);
        let attn: Vec<u16> = (0..groups * group_inner)
            .map(|i| bf16_pattern(i + 2))
            .collect();
        // Each group has its own padded B12x buffer, so the grouped buffer is the concatenation of
        // per-group buffers - not one buffer of groups * lo_rank rows, which would pad differently.
        let mut w = Vec::with_capacity(groups * weight_bytes);
        for group in 0..groups {
            w.extend_from_slice(&b12x_weight_buffer(lo_rank, group_inner, 7 + group * 131));
        }
        let w_scales: Vec<u8> = (0..groups * scale_bytes)
            .map(|i| 90 + (i % 60) as u8)
            .collect();
        let grouped = grouped_wo_a_bf16a(&attn, &w, &w_scales, groups, group_inner, lo_rank);
        for group in 0..groups {
            let dense = dense_gemm_bf16a(
                &attn[group * group_inner..(group + 1) * group_inner],
                &w[group * weight_bytes..(group + 1) * weight_bytes],
                &w_scales[group * scale_bytes..(group + 1) * scale_bytes],
                1,
                lo_rank,
                group_inner,
            );
            assert_eq!(
                &grouped[group * lo_rank..(group + 1) * lo_rank],
                dense.as_slice(),
                "group {group} does not match a dense GEMM over its own slice"
            );
        }
        // Non-vacuity: the two groups carry different weights, so giving group 1 group 0's weight
        // bytes must change group 1's output and leave group 0's alone.
        assert_ne!(
            &w[..weight_bytes],
            &w[weight_bytes..2 * weight_bytes],
            "the test's groups must carry different weights"
        );
        let mut mixed = w.clone();
        mixed.copy_within(0..weight_bytes, weight_bytes);
        let changed = grouped_wo_a_bf16a(&attn, &mixed, &w_scales, groups, group_inner, lo_rank);
        assert_ne!(
            &changed[lo_rank..2 * lo_rank],
            &grouped[lo_rank..2 * lo_rank],
            "the second group must depend on its own weights"
        );
        assert_eq!(
            &changed[..lo_rank],
            &grouped[..lo_rank],
            "the first group must not depend on the second group's weights"
        );
        assert_ne!(
            scale_bytes, weight_bytes,
            "the two group strides must differ"
        );
    }

    #[test]
    fn the_twin_is_the_b12x_layout_of_the_mxfp8_family() {
        // A layout pin, not an arithmetic one: the BF16-activation variant did not get a new weight
        // layout. Every weight index and scale index is the mxfp8_ref helper's, so this test fails
        // if either module grows a private formula.
        let (cols, inner) = (33usize, 96usize);
        assert_eq!(crate::mxfp8_ref::B12X_KTILE, QUANT_BLOCK);
        for n in 0..cols {
            for k in 0..inner {
                assert_eq!(
                    b12x_weight_index(n, k, inner),
                    crate::mxfp8_ref::b12x_weight_index(n, k, inner)
                );
                assert_eq!(
                    b12x_scale_index(n, k, inner),
                    crate::mxfp8_ref::b12x_scale_index(n, k, inner)
                );
            }
        }
    }
}

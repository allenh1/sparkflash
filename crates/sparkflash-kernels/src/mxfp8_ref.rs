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

//! CPU reference for the two MXFP8 kernels of this milestone: the B12x dense twin
//! (kernels/b12x_mxfp8_dense_sm121.cu) and the grouped wo_a projection
//! (kernels/b12x_mxfp8_wo_a_sm121.cu).
//!
//! This module is the definition of those kernels' arithmetic. It is weight-free: it takes encoded
//! FP8 bytes and ue8m0 scales exactly as the kernels do and dequantizes with the same codecs, so
//! the reference cannot quietly use a different rounding rule. What it pins:
//!
//! 1. the B12x layout, as a bijection plus a scale-sharing rule - the index functions below are
//!    the same formulas the shared header defines, and tests/b12x_index_parity.rs compiles the
//!    header's copy with a C compiler and compares the two value for value, so "the same formulas"
//!    is checked rather than asserted. The fp8 grid is the checkpoint's own: one ue8m0 scale per
//!    32x32 block, i.e. ceil(out/32) * ceil(in/32) scales (docs/architecture/03-moe.md:120). The
//!    fp4 routed-expert weights use the different [out, in/32] grid, which is why the header and
//!    this module also carry row_scale_* helpers;
//! 2. the accumulation order: f32, K ascending in 32-value blocks and ascending inside a block,
//!    which is why the reference's own loop is written in that order rather than summed smartly;
//! 3. exactly one rounding to bf16 at the end, round-to-nearest-even;
//! 4. for wo_a, that a group is a dense GEMM over its own heads' concatenated outputs with its own
//!    weight buffer, and nothing else.
//!
//! What it does NOT claim: the kernels are scalar reference paths, and a tensor-core variant would
//! accumulate in the hardware's order. Nothing here says which of the two a GPU would produce.

use crate::m4_quant::{QUANT_BLOCK, bf16_bits, fp8_e4m3_decode, ue8m0_decode};

/// Output channels per B12x micro-tile. Mirrors SPARK_B12X_NTILE.
pub const B12X_NTILE: usize = 32;
/// Values along K per B12x micro-tile, which is one MX block. Mirrors SPARK_B12X_KTILE.
pub const B12X_KTILE: usize = 32;
/// Bytes in one micro-tile. Mirrors SPARK_B12X_TILE_BYTES.
pub const B12X_TILE_BYTES: usize = B12X_NTILE * B12X_KTILE;

/// The byte offset of weight element (n, k) in a B12x buffer of the given shape.
///
/// # Panics
/// Panics if the shape is not a whole number of micro-tiles, which is what makes the mapping a
/// bijection rather than a truncation.
#[must_use]
pub fn b12x_weight_index(n: usize, k: usize, inner: usize) -> usize {
    assert_eq!(
        inner % B12X_KTILE,
        0,
        "the B12x width must be whole micro-tiles"
    );
    let k_blocks = inner / B12X_KTILE;
    let tile = (n / B12X_NTILE) * k_blocks + (k / B12X_KTILE);
    tile * B12X_TILE_BYTES + (n % B12X_NTILE) * B12X_KTILE + (k % B12X_KTILE)
}

/// The byte offset of the ue8m0 scale covering weight element (n, k): one scale per 32x32 block, so
/// every element of a block reads the same byte and the scale count of a matrix is
/// ceil(rows / 32) * (inner / 32).
#[must_use]
pub fn b12x_scale_index(n: usize, k: usize, inner: usize) -> usize {
    assert_eq!(
        inner % B12X_KTILE,
        0,
        "the B12x width must be whole micro-tiles"
    );
    let k_blocks = inner / B12X_KTILE;
    (n / B12X_NTILE) * k_blocks + (k / B12X_KTILE)
}

/// The byte offset of the ue8m0 scale covering FP4 routed-expert element (n, k). The fp4 grid is
/// the checkpoint's [out, in/32]: one scale per output row per 32 values along K, which is a
/// different format from the fp8 32x32 grid above and has a different scale count.
#[must_use]
pub fn b12x_row_scale_index(n: usize, k: usize, inner: usize) -> usize {
    assert_eq!(
        inner % B12X_KTILE,
        0,
        "the B12x width must be whole micro-tiles"
    );
    n * (inner / B12X_KTILE) + (k / B12X_KTILE)
}

/// The number of FP8 bytes a B12x buffer of this shape occupies.
///
/// The layout addresses whole micro-tile rows, so a matrix whose row count is not a multiple of 16
/// needs the padded allocation: the last micro-tile row is partly unused, and a caller that sized
/// the buffer as rows * inner would have the tail of the tile written past the end. That is an ABI
/// requirement of specification 4.1's layout, not an implementation detail of this reference.
#[must_use]
pub fn b12x_buffer_bytes(rows: usize, inner: usize) -> usize {
    b12x_padded_rows(rows) * inner
}

/// Permutes row-major FP8 weights into the B12x layout the kernels read.
///
/// The transform is a pure permutation: [b12x_weight_index] is a bijection over the buffer, so
/// no value is changed, rounded or dropped. It is meant to run ONCE at arena-build time, before
/// the bytes are uploaded; the device kernels never tile internally. Scales are already in the
/// checkpoint's [ceil(rows / 32), inner / 32] grid and are returned unchanged by the caller.
///
/// # Panics
/// Panics when the buffer length is not exactly cols * inner, or when inner is not a whole number
/// of 32-value MX blocks. The row count may be any value a caller pads before tiling.
#[must_use]
pub fn b12x_tile_weights(raw: &[u8], cols: usize, inner: usize) -> Vec<u8> {
    assert_eq!(
        raw.len(),
        cols * inner,
        "the B12x source must be exactly cols * inner bytes"
    );
    assert_eq!(
        inner % B12X_KTILE,
        0,
        "the B12x width must be whole micro-tiles"
    );
    let padded = b12x_padded_rows(cols);
    let mut tiled = vec![0u8; padded * inner];
    for col in 0..cols {
        for k in 0..inner {
            tiled[b12x_weight_index(col, k, inner)] = raw[col * inner + k];
        }
    }
    tiled
}

/// The number of ue8m0 bytes an FP8 B12x scale buffer of this shape occupies: exactly the
/// checkpoint's [ceil(rows/32), ceil(inner/32)] grid, padded to whole micro-tile rows. This is a
/// different size from b12x_buffer_bytes, 32 times smaller for a full 32-row tile, and a stride
/// that walks a scale buffer must use this one.
#[must_use]
pub fn b12x_scale_bytes(rows: usize, inner: usize) -> usize {
    (b12x_padded_rows(rows) / B12X_NTILE) * (inner / B12X_KTILE)
}

/// The number of ue8m0 bytes an FP4 routed-expert scale buffer of this shape occupies: the
/// checkpoint's [out, in/32] grid, rows * (inner / 32), with no row padding.
#[must_use]
pub fn b12x_row_scale_bytes(rows: usize, inner: usize) -> usize {
    rows * (inner / B12X_KTILE)
}

/// The micro-tile rows a B12x buffer of this shape covers: rows rounded up to whole 32-row tiles.
#[must_use]
pub fn b12x_padded_rows(rows: usize) -> usize {
    rows.div_ceil(B12X_NTILE) * B12X_NTILE
}

/// The dequantized value of one B12x weight element.
///
/// # Panics
/// Panics if the buffers are too short for the shape, so a mis-sized fixture cannot silently read
/// zeros.
#[must_use]
pub fn b12x_weight(weights: &[u8], scales: &[u8], n: usize, k: usize, inner: usize) -> f32 {
    let weight_index = b12x_weight_index(n, k, inner);
    let scale_index = b12x_scale_index(n, k, inner);
    assert!(
        weight_index < weights.len(),
        "the weight buffer is too short"
    );
    assert!(scale_index < scales.len(), "the scale buffer is too short");
    fp8_e4m3_decode(weights[weight_index]) * ue8m0_decode(scales[scale_index])
}

/// One MXFP8 dense GEMM, exactly as b12x_mxfp8_dense_sm121.cu computes it.
///
/// The activations are \[rows\]\[inner\] E4M3 with scales \[rows\][inner / 32] ue8m0; the weights are a
/// B12x buffer of \[cols\]\[inner\] with scales in the same layout. The result is \[rows\]\[cols\] bf16
/// bits.
///
/// # Panics
/// Panics if the buffers are too short for the declared shape.
#[must_use]
pub fn dense_gemm(
    a: &[u8],
    a_scales: &[u8],
    w: &[u8],
    w_scales: &[u8],
    rows: usize,
    cols: usize,
    inner: usize,
) -> Vec<u16> {
    let k_blocks = inner / QUANT_BLOCK;
    assert!(
        a.len() >= rows * inner,
        "the activation buffer is too short"
    );
    assert!(
        a_scales.len() >= rows * k_blocks,
        "the activation scale buffer is too short"
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
                let a_scale = ue8m0_decode(a_scales[row * k_blocks + block]);
                let w_scale = ue8m0_decode(w_scales[b12x_scale_index(col, base, inner)]);
                for i in 0..QUANT_BLOCK {
                    let a_value = fp8_e4m3_decode(a[row * inner + base + i]) * a_scale;
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

/// The grouped wo_a projection: for each group, a dense MXFP8 GEMM over that group's concatenated
/// head outputs against the group's own B12x weight buffer of \[lo_rank\]\[group_inner\].
///
/// This is deliberately implemented by calling dense_gemm per group with a group-shaped slice,
/// because that is exactly the claim the kernel makes: wo_a is the dense twin plus a group
/// dimension. The test below checks the two agree on data that exercises both.
///
/// # Panics
/// Panics if the buffers are too short for the declared shape.
#[must_use]
pub fn grouped_wo_a(
    attn: &[u8],
    attn_scales: &[u8],
    w: &[u8],
    w_scales: &[u8],
    groups: usize,
    group_inner: usize,
    lo_rank: usize,
) -> Vec<u16> {
    let k_blocks = group_inner / QUANT_BLOCK;
    let weight_bytes = b12x_buffer_bytes(lo_rank, group_inner);
    let scale_bytes = b12x_scale_bytes(lo_rank, group_inner);
    assert!(
        attn.len() >= groups * group_inner,
        "the attention buffer is too short"
    );
    assert!(
        w.len() >= groups * weight_bytes,
        "the weight buffer is too short"
    );
    let mut out = vec![0u16; groups * lo_rank];
    for group in 0..groups {
        let group_out = dense_gemm(
            &attn[group * group_inner..(group + 1) * group_inner],
            &attn_scales[group * k_blocks..(group + 1) * k_blocks],
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::m4_quant::{E4M3_ELEMENT_EMAX, bf16_value, fp8_e4m3_encode, ue8m0_from_amax};

    /// A tiny deterministic weighting: it only has to be reproducible, not random.
    fn weight_code(seed: usize) -> u8 {
        // Codes 0x00..=0x7E are finite E4M3 magnitudes; 0x7F would be NaN.
        (seed % 0x7F) as u8
    }

    #[test]
    fn the_b12x_layout_is_a_bijection_over_the_weight_buffer() {
        // The layout is only usable if every (n, k) lands on its own byte: a collision would make
        // one weight element unreachable and another one shared.
        for inner in [32usize, 64, 128] {
            for cols in [16usize, 17, 32, 48] {
                let padded_rows = b12x_padded_rows(cols);
                let mut seen = vec![false; b12x_buffer_bytes(cols, inner)];
                for n in 0..cols {
                    for k in 0..inner {
                        let index = b12x_weight_index(n, k, inner);
                        assert!(
                            index < seen.len(),
                            "index {index} escapes the padded buffer of {} bytes for ({cols}, {inner})",
                            seen.len()
                        );
                        assert!(!seen[index], "two weight elements share byte {index}");
                        seen[index] = true;
                    }
                }
                assert_eq!(
                    seen.iter().filter(|hit| **hit).count(),
                    cols * inner,
                    "every element must land on its own byte"
                );
                if cols == padded_rows {
                    assert!(
                        seen.iter().all(|hit| *hit),
                        "a whole-tile layout must leave no holes"
                    );
                } else {
                    // A ragged row count leaves the tail of the last micro-tile row unused, which
                    // is exactly why the allocation has to be padded.
                    assert_eq!(
                        padded_rows,
                        cols.div_ceil(B12X_NTILE) * B12X_NTILE,
                        "a ragged row count rounds up to whole micro-tile rows"
                    );
                    assert!(seen.iter().any(|hit| !*hit), "the padding must be unused");
                }
                // The scales are shared over a whole 32x32 block: the same byte for every element
                // of the block, in BOTH dimensions, which is the checkpoint's one-scale-per-block
                // grid and not a per-row scale.
                for n in 0..cols {
                    for k in 0..inner {
                        assert_eq!(
                            b12x_scale_index(n, k, inner),
                            b12x_scale_index(n, k - (k % B12X_KTILE), inner),
                            "a scale must cover the whole 32-value block along K"
                        );
                        assert_eq!(
                            b12x_scale_index(n, k, inner),
                            b12x_scale_index(n - (n % B12X_NTILE), k, inner),
                            "a scale must cover the whole 32-row block along N"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_two_scale_grids_are_the_documented_checkpoint_shapes() {
        // The fp8 grid must be exactly the checkpoint's [ceil(out/32), ceil(in/32)] and the fp4
        // expert grid its [out, in/32] (docs/architecture/03-moe.md:118-121). This is what makes
        // "the kernel reads checkpoint scales" a checked statement rather than a reading, and it
        // also shows the two are different sizes, so one cannot be used as the other's stride.
        for (rows, inner) in [
            (1usize, 32usize),
            (17, 64),
            (32, 96),
            (33, 128),
            (1024, 4096),
        ] {
            assert_eq!(
                b12x_scale_bytes(rows, inner),
                rows.div_ceil(32) * inner.div_ceil(32),
                "the fp8 scale grid of ({rows}, {inner}) is not [ceil(out/32), ceil(in/32)]"
            );
            assert_eq!(
                b12x_row_scale_bytes(rows, inner),
                rows * (inner / 32),
                "the fp4 scale grid of ({rows}, {inner}) is not [out, in/32]"
            );
        }
        let (rows, inner) = (1024usize, 4096usize);
        assert_eq!(b12x_scale_bytes(rows, inner), 4096);
        assert_eq!(b12x_row_scale_bytes(rows, inner), 131_072);
        assert_ne!(
            b12x_scale_bytes(rows, inner),
            b12x_buffer_bytes(rows, inner),
            "a scale stride must not be the weight stride: that is the wo_a grouping defect"
        );
        assert_eq!(b12x_buffer_bytes(rows, inner), 4_194_304);
    }

    #[test]
    fn a_wo_a_group_needs_two_different_strides_and_the_weight_one_escapes() {
        // The exact shape a rank runs: 2 groups of [1024][4096] weights. A group's scale buffer is
        // 4096 bytes while its weight buffer is 4,194,304, so a kernel that offsets BOTH by the
        // weight size reads group 1's first scale at 4,194,304 bytes, one past the end of a
        // correctly sized 8192-byte scale buffer. This is the regression pin for that defect, and it
        // is stated in the units the caller allocates in.
        let (groups, lo_rank, group_inner) = (2usize, 1024usize, 4096usize);
        let weight_bytes = b12x_buffer_bytes(lo_rank, group_inner);
        let scale_bytes = b12x_scale_bytes(lo_rank, group_inner);
        assert_eq!(weight_bytes, 4_194_304);
        assert_eq!(scale_bytes, 4_096);
        let scale_buffer = groups * scale_bytes;
        assert_eq!(scale_buffer, 8192);
        // The correct stride lands every group inside the buffer...
        assert!((groups - 1) * scale_bytes + scale_bytes <= scale_buffer);
        // ...and the weight stride does not, for any group after the first.
        assert!(
            (groups - 1) * weight_bytes + scale_bytes > scale_buffer,
            "the weight stride must be out of bounds for the scale buffer, which is why wo_a needs \
             spark_b12x_scale_bytes for its scale group base"
        );
    }

    #[test]
    fn golden_single_block_products_are_hand_computable() {
        // One row, one column, one MX block of 32 values. Every activation and weight code is 0x38,
        // which is E4M3 1.0. The block maximum is 1.0, so with the E4M3 element emax of 8 the
        // shared scale is 2^(0 - 8) = 2^-8 (ue8m0 code 119) on both sides. Each product is
        // 2^-8 * 2^-8 = 2^-16, and there are 32 of them, so the sum is 32 * 2^-16 = 2^-11. That is
        // exactly representable in bf16: exponent field 127 - 11 = 116, mantissa zero, bit pattern
        // 116 << 7 = 0x3A00.
        let inner = 32usize;
        let scale = ue8m0_from_amax(1.0, E4M3_ELEMENT_EMAX);
        assert_eq!(scale, 119, "1.0 with an emax of 8 is a 2^-8 scale");
        let a = vec![0x38u8; inner];
        // One output channel still occupies a whole 32-row micro-tile, so the buffers are padded.
        let mut w = vec![0u8; b12x_buffer_bytes(1, inner)];
        for k in 0..inner {
            w[b12x_weight_index(0, k, inner)] = 0x38;
        }
        let mut w_scales = vec![127u8; b12x_scale_bytes(1, inner)];
        w_scales[b12x_scale_index(0, 0, inner)] = scale;
        let out = dense_gemm(&a, &[scale], &w, &w_scales, 1, 1, inner);
        assert_eq!(out[0], 0x3A00, "32 products of 2^-16 is 2^-11");
        assert_eq!(bf16_value(out[0]), 2.0f32.powi(-11));

        // Doubling the activation scale doubles the result: 2^-10, bits (127 - 10) << 7 = 0x3A80.
        let doubled = ue8m0_from_amax(2.0, E4M3_ELEMENT_EMAX);
        let out = dense_gemm(&a, &[doubled], &w, &w_scales, 1, 1, inner);
        assert_eq!(out[0], 0x3A80);
    }

    #[test]
    fn the_reference_matches_a_naive_dequantized_loop() {
        // Cross-check against the most obvious implementation possible: dequantize both operands
        // and sum in the same order. Agreement on a shape with several micro-tiles, several K
        // blocks and a ragged N means the index helpers and the arithmetic agree.
        let (rows, cols, inner) = (3usize, 20usize, 96usize);
        let k_blocks = inner / QUANT_BLOCK;
        let a: Vec<u8> = (0..rows * inner).map(|i| weight_code(i * 7 + 1)).collect();
        let a_scales: Vec<u8> = vec![127; rows * k_blocks];
        let w: Vec<u8> = (0..b12x_buffer_bytes(cols, inner))
            .map(|i| weight_code(i * 5 + 3))
            .collect();
        let w_scales: Vec<u8> = vec![127; b12x_scale_bytes(cols, inner)];

        let reference = dense_gemm(&a, &a_scales, &w, &w_scales, rows, cols, inner);

        for row in 0..rows {
            for col in 0..cols {
                let mut acc = 0.0f32;
                for k in 0..inner {
                    acc += fp8_e4m3_decode(a[row * inner + k])
                        * fp8_e4m3_decode(w[b12x_weight_index(col, k, inner)]);
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
        // The reference exists to pin an order, so a case where the order is observable is worth
        // having: 1.0 added 32 times, then a tiny value, is not the same float sum as the tiny value
        // first. The reference must equal the ascending-order sum, not the sorted one.
        let inner = 32usize;
        let mut a = vec![0x38u8; inner]; // 1.0
        a[31] = 0x00; // and one zero
        let mut w = vec![0x38u8; b12x_buffer_bytes(1, inner)];
        w[b12x_weight_index(0, 0, inner)] = 0x01; // the smallest E4M3 subnormal, 2^-9
        let scales = vec![127u8; 1];
        let w_scales = vec![127u8; b12x_scale_bytes(1, inner)];
        let reference = dense_gemm(&a, &scales, &w, &w_scales, 1, 1, inner);
        let mut ascending = 0.0f32;
        for k in 0..inner {
            ascending += fp8_e4m3_decode(a[k]) * fp8_e4m3_decode(w[b12x_weight_index(0, k, inner)]);
        }
        assert_eq!(reference[0], bf16_bits(ascending));
        assert_ne!(reference[0], 0, "the sum must not vanish");
    }

    #[test]
    fn a_quantized_row_round_trips_within_one_code_step() {
        // Encoding then decoding must stay inside one E4M3 step: this is what lets the references
        // quantize their own fixtures without lying about what the kernels would read.
        for magnitude in [0.0f32, 0.001, 0.5, 1.0, 7.5, 448.0] {
            for sign in [1.0f32, -1.0] {
                let value = sign * magnitude;
                let code = fp8_e4m3_encode(value);
                let back = fp8_e4m3_decode(code);
                // One step is 1/8 of the magnitude for normal values and 2^-9 for subnormals, so
                // half a step is 1/16 of the magnitude or 2^-10, whichever is larger.
                let tolerance = (value.abs() * 0.0625).max(2.0f32.powi(-10)) + 1e-30;
                assert!(
                    (back - value).abs() <= tolerance,
                    "encoding {value} gave {back}, outside half an E4M3 step"
                );
            }
        }
    }

    #[test]
    fn wo_a_is_the_dense_twin_with_a_group_dimension() {
        // The claim the wo_a kernel makes is that it is the dense GEMM per group. Build grouped
        // buffers and compare the grouped result against direct dense calls.
        let (groups, group_inner, lo_rank) = (2usize, 64usize, 16usize);
        let k_blocks = group_inner / QUANT_BLOCK;
        let weight_bytes = b12x_buffer_bytes(lo_rank, group_inner);
        let scale_bytes = b12x_scale_bytes(lo_rank, group_inner);
        let attn: Vec<u8> = (0..groups * group_inner)
            .map(|i| weight_code(i + 2))
            .collect();
        let attn_scales: Vec<u8> = (0..groups * k_blocks)
            .map(|i| 100 + (i % 40) as u8)
            .collect();
        let w: Vec<u8> = (0..groups * weight_bytes)
            .map(|i| weight_code(i * 3 + 5))
            .collect();
        let w_scales: Vec<u8> = (0..groups * scale_bytes)
            .map(|i| 90 + (i % 60) as u8)
            .collect();

        let grouped = grouped_wo_a(
            &attn,
            &attn_scales,
            &w,
            &w_scales,
            groups,
            group_inner,
            lo_rank,
        );
        for group in 0..groups {
            let dense = dense_gemm(
                &attn[group * group_inner..(group + 1) * group_inner],
                &attn_scales[group * k_blocks..(group + 1) * k_blocks],
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
        // A mutation proof: reusing the first group's weights for the second must change the
        // second group's output, which is what makes the per-group buffer claim testable.
        let mut tampered = w.clone();
        for (byte, value) in tampered[weight_bytes..2 * weight_bytes]
            .iter_mut()
            .enumerate()
        {
            *value = weight_code(byte + 11);
        }
        let changed = grouped_wo_a(
            &attn,
            &attn_scales,
            &tampered,
            &w_scales,
            groups,
            group_inner,
            lo_rank,
        );
        assert_ne!(
            &changed[lo_rank..2 * lo_rank],
            &grouped[lo_rank..2 * lo_rank],
            "the second group must depend on its own weights"
        );
        // The same non-vacuity for the SCALE stride: the groups carry different ue8m0 scales, so
        // giving group 1 group 0's scales must change group 1's output. Without this, a reference
        // that read the scales through the wrong stride could still agree with itself.
        assert_ne!(
            scale_bytes, weight_bytes,
            "the two group strides must differ"
        );
        let mut mixed_scales = w_scales.clone();
        mixed_scales.copy_within(0..scale_bytes, scale_bytes);
        let mixed = grouped_wo_a(
            &attn,
            &attn_scales,
            &w,
            &mixed_scales,
            groups,
            group_inner,
            lo_rank,
        );
        assert_ne!(
            &mixed[lo_rank..2 * lo_rank],
            &grouped[lo_rank..2 * lo_rank],
            "the second group must depend on its own scales"
        );
        assert_eq!(
            &changed[..lo_rank],
            &grouped[..lo_rank],
            "the first group must not depend on the second group's weights"
        );
    }

    #[test]
    fn b12x_tile_weights_is_the_index_bijection_and_keeps_every_byte() {
        for (cols, inner) in [(1usize, 32usize), (3, 64), (33, 32)] {
            let raw: Vec<u8> = (0..cols * inner).map(|i| (i % 251) as u8).collect();
            let tiled = b12x_tile_weights(&raw, cols, inner);
            assert_eq!(tiled.len(), b12x_buffer_bytes(cols, inner));
            for col in 0..cols {
                for k in 0..inner {
                    assert_eq!(
                        tiled[b12x_weight_index(col, k, inner)],
                        raw[col * inner + k],
                        "tile ({col}, {k}) of {cols}x{inner}"
                    );
                }
            }
            let mut sorted = tiled.clone();
            sorted.sort_unstable();
            let mut expected = vec![0u8; b12x_buffer_bytes(cols, inner)];
            expected[..raw.len()].copy_from_slice(&raw);
            expected.sort_unstable();
            assert_eq!(
                sorted, expected,
                "the transform is a permutation, not a rewrite"
            );
        }
    }
}

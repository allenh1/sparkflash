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

//! The host-side oracle compare for the WS-D columns-only M=1 dense kernel
//! (`kernels/b12x_mxfp8_dense_m1_sm121.cu`).
//!
//! What this file proves, and what it does not:
//!
//! * it proves the tiling **model** every served shape would run is the pinned arithmetic: the
//!   model below is the kernel's own structure - one output column per thread, `ceil(cols/256)`
//!   blocks, K ascending in 32-value blocks and ascending inside each block, one ue8m0 decode per
//!   block for the activation scale and one for the weight scale, and exactly one round to bf16 at
//!   the end - and it is compared **bit for bit** (bf16 words, not tolerances) against
//!   [mxfp8_ref::dense_gemm], which is the CPU twin of the pinned `b12x_mxfp8_dense_sm121` kernel.
//!   The comparison runs on every shape the decode step puts through the dense family: the
//!   attention projections wq_a and wkv, wo_b, and wq_b. A shape whose tile boundary is not exact
//!   (cols not a multiple of 256) is included so the column guard is exercised too.
//! * it proves the grid covers every output column exactly once: the model records a write count
//!   per column and the test asserts one, so a tiling that skipped or double-wrote a column fails
//!   here instead of on a device.
//! * it does NOT prove the compiled device artifact. The device kernel is compiled by nvcc from the
//!   same source, and the contract that ties the two is the arithmetic order above; the identity
//!   gate (tokens + logits sha) and the kernel-vs-twin harness on the CUDA host are what measure the
//!   artifact. A NaN or a rounding difference introduced by nvcc would show up there, not here.
//!
//! The oracle-shaped inputs are built with the crate's own codecs ([m4_quant]), never with hand
//! assembly, so the bytes are the kind the model would see in a window: E4M3 codes that the encoder
//! produced and ue8m0 scales that the activation quantiser would emit.

use sparkflash_kernels::m4_quant::{bf16_bits, fp8_e4m3_encode, ue8m0_code_of_power_of_two};
use sparkflash_kernels::mxfp8_ref::{
    b12x_buffer_bytes, b12x_scale_bytes, b12x_scale_index, b12x_weight_index, dense_gemm,
};

/// The kernel's column tile: one output column per thread, 256 threads.
const COLUMNS: usize = 256;
/// One MX scale per 32 values along K, as the kernel's SPARK_M4_QUANT_BLOCK.
const QUANT_BLOCK: usize = 32;

/// A deterministic activation row: E4M3 codes and one ue8m0 scale per 32 values.
fn activation_row(inner: usize, salt: u32) -> (Vec<u8>, Vec<u8>) {
    let mut codes = Vec::with_capacity(inner);
    for k in 0..inner {
        let value = ((k as u32).wrapping_mul(37).wrapping_add(salt) % 251) as f32 * 0.017 - 2.0;
        codes.push(fp8_e4m3_encode(value));
    }
    let mut scales = Vec::with_capacity(inner / QUANT_BLOCK);
    for block in 0..inner / QUANT_BLOCK {
        let shift = (block as u32 % 5) as i32 - 2;
        scales.push(ue8m0_code_of_power_of_two(2.0f32.powi(shift)));
    }
    (codes, scales)
}

/// A deterministic B12x weight buffer and its scale grid, laid out by the reference's own indexes.
fn weight_buffer(cols: usize, inner: usize, salt: u32) -> (Vec<u8>, Vec<u8>) {
    let mut codes = vec![0u8; b12x_buffer_bytes(cols, inner)];
    let mut scales = vec![0u8; b12x_scale_bytes(cols, inner)];
    for col in 0..cols {
        for k in 0..inner {
            let value = ((col * 17 + k * 7 + salt as usize) % 97) as f32 * 0.043 - 1.5;
            codes[b12x_weight_index(col, k, inner)] = fp8_e4m3_encode(value);
        }
        for block in 0..inner / QUANT_BLOCK {
            let exponent = ((col + block) % 5) as i32 - 2;
            let scale = 2.0f32.powi(exponent);
            scales[b12x_scale_index(col, block * QUANT_BLOCK, inner)] =
                ue8m0_code_of_power_of_two(scale);
        }
    }
    (codes, scales)
}

/// The M=1 kernel's arithmetic and tiling, written the way the kernel writes it.
///
/// Returns the bf16 words and the per-column write count, so the caller can prove the coverage as
/// well as the values. The loops index their operands on purpose: they are the kernel's block and
/// element loops, not iteration over the buffers, and a rewritten iterator would hide exactly the
/// order this model exists to pin.
#[allow(clippy::needless_range_loop)]
fn m1_kernel_model(
    a: &[u8],
    a_scales: &[u8],
    w: &[u8],
    w_scales: &[u8],
    cols: usize,
    inner: usize,
) -> (Vec<u16>, Vec<u32>) {
    let k_blocks = inner / QUANT_BLOCK;
    let blocks = cols.div_ceil(COLUMNS);
    let mut out = vec![0u16; cols];
    let mut writes = vec![0u32; cols];
    // The kernel serves exactly one output row; the shape contract is the launch wrapper's.
    for block in 0..blocks {
        for lane in 0..COLUMNS {
            let col = block * COLUMNS + lane;
            if col >= cols {
                continue;
            }
            let mut acc = 0.0f32;
            for k_block in 0..k_blocks {
                let base = k_block * QUANT_BLOCK;
                let a_scale = sparkflash_kernels::m4_quant::ue8m0_decode(a_scales[k_block]);
                let w_scale = sparkflash_kernels::m4_quant::ue8m0_decode(
                    w_scales[b12x_scale_index(col, base, inner)],
                );
                for i in 0..QUANT_BLOCK {
                    let a_value =
                        sparkflash_kernels::m4_quant::fp8_e4m3_decode(a[base + i]) * a_scale;
                    let w_value = sparkflash_kernels::m4_quant::fp8_e4m3_decode(
                        w[b12x_weight_index(col, base + i, inner)],
                    ) * w_scale;
                    acc += a_value * w_value;
                }
            }
            out[col] = bf16_bits(acc);
            writes[col] += 1;
        }
    }
    (out, writes)
}

/// The shapes the decode step puts through the dense family, with the block count the kernel's grid
/// would launch. Every one of them is an exact multiple of the 256-column tile.
const SERVED_SHAPES: [(&str, usize, usize); 4] = [
    ("attention projections wq_a", 1280, 5120),
    ("attention projections wkv", 512, 5120),
    ("wo_b", 5120, 256),
    ("wq_b", 8192, 1280),
];

#[test]
fn the_m1_tiling_is_the_pinned_arithmetic_on_every_served_shape() {
    for (name, cols, inner) in SERVED_SHAPES {
        let (a, a_scales) = activation_row(inner, 3);
        let (w, w_scales) = weight_buffer(cols, inner, 11);
        let oracle = dense_gemm(&a, &a_scales, &w, &w_scales, 1, cols, inner);
        let (model, writes) = m1_kernel_model(&a, &a_scales, &w, &w_scales, cols, inner);
        assert_eq!(
            writes,
            vec![1u32; cols],
            "{name}: the tiling must write every output column exactly once"
        );
        assert_eq!(
            model, oracle,
            "{name} ({cols}x{inner}): the M=1 tiling model must be the pinned arithmetic bit for bit"
        );
        assert_eq!(
            cols % COLUMNS,
            0,
            "{name}: the served shapes are exact multiples of the column tile"
        );
    }
}

#[test]
fn the_column_tail_is_guarded_once() {
    // A shape that is not a multiple of the tile: the last block has active and inactive lanes, so
    // the guard is exercised. The oracle is the same pinned dense GEMM, on a shape the kernel would
    // still serve correctly (the launch wrapper allows any cols >= 1).
    let (cols, inner) = (1300usize, 128usize);
    let (a, a_scales) = activation_row(inner, 5);
    let (w, w_scales) = weight_buffer(cols, inner, 13);
    let oracle = dense_gemm(&a, &a_scales, &w, &w_scales, 1, cols, inner);
    let (model, writes) = m1_kernel_model(&a, &a_scales, &w, &w_scales, cols, inner);
    assert_eq!(
        writes,
        vec![1u32; cols],
        "the tail block must write the real columns once and leave the padding columns alone"
    );
    assert_eq!(
        model, oracle,
        "the guarded tail must be the pinned arithmetic"
    );
}

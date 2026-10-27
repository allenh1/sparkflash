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

//! The F3 kernel half (task-45): the native fp8 W8A8 indexer query projection at the released
//! shape, proven against the pinned kernel's arithmetic without a GPU.
//!
//! The chosen seam is projection-on-device-only. The engine hands the kernel pre-quantised E4M3
//! activations for the bf16 q_norm latent (act_quant: one absolute maximum per 32-value block, the
//! 1e-4 amax floor, a ue8m0 round-up-to-power-of-two scale) and the checkpoint's own fp8 wq_b codes
//! plus its 32x32 ue8m0 grid, and gets bf16 rows back; rope, the fp4 q round trip, the scores and
//! the top-k stay host and are already native (docs/archive/build/indexer-native-dtype.md).
//!
//! The Lead's decision (2026-10-19) is that the already-pinned, already-measured
//! b12x_mxfp8_dense_sm121 IS that arithmetic, so this milestone adds no kernel, no KernelId, no
//! baseline entry and no ABI: it pins the contract at the indexer geometry with two independent
//! implementations of the same declared order.
//!
//! Arithmetic contract (kernels/b12x_mxfp8_dense_sm121.cu): f32 accumulation, K ascending in
//! 32-value blocks and ascending inside a block; per element a_value = decode(a) * a_scale and
//! w_value = decode(w) * w_scale, acc += a_value * w_value; exactly one bf16 round at the end.
//! Applying a block's scale product to a block dot instead is a different f32 rounding, and
//! the_scale_on_the_block_dot_is_a_different_arithmetic shows the difference is observable.
//!
//! Exact packed bytes means the operands that cross the seam: a [cols][inner] checkpoint-order
//! E4M3 weight plus its [cols/32][inner/32] ue8m0 grid (the read_packed B12x tiling is the one
//! re-applied here, via b12x_weight_index), and the E4M3 codes and ue8m0 scales that
//! act_quant_fp8_row produces from bf16-representable latent rows.

use sparkflash_kernels::cuda_driver::KernelId;
use sparkflash_kernels::m4_quant::{
    act_quant_fp8_row, bf16_bits, bf16_value, fp8_e4m3_decode, ue8m0_decode,
};
use sparkflash_kernels::mxfp8_dense_launch::{
    ARG_WORDS, COLS_INDEX, INNER_INDEX, KERNEL_ID, PARAMS, ROWS_INDEX, declaration,
    declaration_matches, grid,
};
use sparkflash_kernels::mxfp8_ref::{
    b12x_buffer_bytes, b12x_padded_rows, b12x_scale_bytes, b12x_scale_index, b12x_weight_index,
    dense_gemm,
};

/// The released indexer geometry: 32 index heads of 128 values over a 1280-wide q latent.
const INDEX_N_HEADS: usize = 32;
const INDEX_HEAD_DIM: usize = 128;
const Q_LORA_RANK: usize = 1280;
const COLS: usize = INDEX_N_HEADS * INDEX_HEAD_DIM;
const K_BLOCK: usize = 32;
const K_BLOCKS: usize = Q_LORA_RANK / K_BLOCK;

/// The shared staging buffer's size in projection outputs: MAX_OUT_BYTES = 32,768 bf16 values in
/// crates/sparkflash-cli/src/attention_device.rs. Indexer wq_b emits rows * 4096 values, so the
/// seam chunks rows at eight per launch rather than silently growing the shared constant; the
/// rows_are_independent_so_a_chunked_launch_is_the_same_bytes test proves chunking is
/// bit-transparent, which is the decision the docs record.
const MAX_OUT_VALUES: usize = 32_768;

/// A deterministic xorshift64* so the fixture is reproducible and needs no dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn latent(&mut self, count: usize) -> Vec<f32> {
        (0..count)
            .map(|_| {
                let raw = (self.next() % 4_000) as f32 / 1_000.0 - 2.0;
                bf16_value(bf16_bits(raw))
            })
            .collect()
    }

    fn weight(&mut self, count: usize) -> Vec<u8> {
        // Finite E4M3 codes only: 0x7F and 0xFF are NaN, and a fixture full of NaNs would make
        // every comparison look equal for the wrong reason.
        (0..count)
            .map(|_| match (self.next() & 0xFF) as u8 {
                0x7F => 0x7E,
                0xFF => 0xFE,
                code => code,
            })
            .collect()
    }

    fn scales(&mut self, count: usize) -> Vec<u8> {
        // ue8m0 codes around the bias: powers of two in a sane range, never the NaN code.
        (0..count)
            .map(|_| 120u8 + (self.next() % 17) as u8)
            .collect()
    }
}

/// The B12x tiling read_packed applies to a checkpoint-order weight: [cols][inner] row-major E4M3
/// into the kernel's 32x32 micro-tile order. The allocation is the layout's padded size, so a cols
/// count that is not a whole N tile keeps its unused tail as zeros.
fn pack_b12x(raw: &[u8], cols: usize, inner: usize) -> Vec<u8> {
    assert_eq!(raw.len(), cols * inner, "the raw weight is [cols][inner]");
    let mut tiled = vec![0u8; b12x_buffer_bytes(cols, inner)];
    for col in 0..cols {
        for k in 0..inner {
            tiled[b12x_weight_index(col, k, inner)] = raw[col * inner + k];
        }
    }
    tiled
}

/// The op-by-op reference: the kernel's declared order written out literally, starting from the
/// seam's own inputs (bf16-representable latent rows -> act_quant -> wq_b matmul -> bf16).
fn reference_projection(
    latent: &[f32],
    rows: usize,
    raw_weight: &[u8],
    weight_scales: &[u8],
    cols: usize,
    inner: usize,
) -> Vec<u16> {
    assert_eq!(latent.len(), rows * inner);
    let k_blocks = inner / K_BLOCK;
    let mut out = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        let (codes, scales) = act_quant_fp8_row(&latent[row * inner..(row + 1) * inner]);
        assert_eq!(codes.len(), inner);
        assert_eq!(scales.len(), k_blocks);
        for col in 0..cols {
            let mut acc = 0.0f32;
            for (block, &scale_code) in scales.iter().enumerate() {
                let base = block * K_BLOCK;
                let a_scale = ue8m0_decode(scale_code);
                let w_scale = ue8m0_decode(weight_scales[b12x_scale_index(col, base, inner)]);
                for i in 0..K_BLOCK {
                    let a_value = fp8_e4m3_decode(codes[base + i]) * a_scale;
                    let w_value = fp8_e4m3_decode(raw_weight[col * inner + base + i]) * w_scale;
                    acc += a_value * w_value;
                }
            }
            out.push(bf16_bits(acc));
        }
    }
    out
}

/// The same operands, with each block's scale product applied to the block dot instead of to every
/// element. This is the arithmetic the kernel does NOT do; it exists to show that the accumulation
/// grouping is load-bearing once the sum cancels.
fn block_dot_projection(
    latent: &[f32],
    rows: usize,
    raw_weight: &[u8],
    weight_scales: &[u8],
    cols: usize,
    inner: usize,
) -> Vec<u16> {
    let mut out = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        let (codes, scales) = act_quant_fp8_row(&latent[row * inner..(row + 1) * inner]);
        for col in 0..cols {
            let mut acc = 0.0f32;
            for (block, &scale_code) in scales.iter().enumerate() {
                let base = block * K_BLOCK;
                let mut dot = 0.0f32;
                for i in 0..K_BLOCK {
                    dot += fp8_e4m3_decode(codes[base + i])
                        * fp8_e4m3_decode(raw_weight[col * inner + base + i]);
                }
                let a_scale = ue8m0_decode(scale_code);
                let w_scale = ue8m0_decode(weight_scales[b12x_scale_index(col, base, inner)]);
                acc += dot * a_scale * w_scale;
            }
            out.push(bf16_bits(acc));
        }
    }
    out
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn the_indexer_shape_is_whole_32_value_blocks_and_the_launch_geometry_is_the_pinned_one() {
    assert_eq!(COLS, 4096);
    assert_eq!(Q_LORA_RANK, 1280);
    assert_eq!(Q_LORA_RANK % K_BLOCK, 0);
    assert_eq!(K_BLOCKS, 40);
    // The released shape is whole B12x tiles in both dimensions, so the tiled weight and its scale
    // grid are exactly the checkpoint's own byte counts: no padding is read and no byte moves.
    assert_eq!(b12x_buffer_bytes(COLS, Q_LORA_RANK), COLS * Q_LORA_RANK);
    assert_eq!(b12x_padded_rows(COLS), COLS);
    assert_eq!(b12x_scale_bytes(COLS, Q_LORA_RANK), (COLS / 32) * K_BLOCKS);
    // One ue8m0 activation scale per 32-value block of a latent row.
    assert_eq!(K_BLOCKS, Q_LORA_RANK / 32);
    // The seam chunks at eight released-geometry rows; the shared constant is not touched.
    assert_eq!(MAX_OUT_VALUES / COLS, 8, "eight rows fit the shared buffer");
    assert!(
        9 * COLS > MAX_OUT_VALUES,
        "nine released-geometry rows overflow it"
    );
    for rows in [1u32, 8, 137] {
        let (x, y, z) = grid(rows, COLS as u32).expect("a non-zero grid");
        assert_eq!(x, rows.div_ceil(64));
        assert_eq!(y, (COLS as u32).div_ceil(64));
        assert_eq!(z, 1);
    }
    // The pinned typed launch is the contract at these dims: the declaration still parses, and the
    // three shape words are where Mxfp8DenseParams puts them.
    let decl = declaration();
    assert!(declaration_matches(&decl), "the pinned declaration changed");
    assert_eq!(ARG_WORDS, 11);
    assert_eq!(KERNEL_ID, KernelId::Mxfp8Dense);
    for (index, name) in [
        (ROWS_INDEX, "rows"),
        (COLS_INDEX, "cols"),
        (INNER_INDEX, "inner"),
    ] {
        let (pinned_name, base, pointer, const_pointee) = PARAMS[index];
        assert_eq!(pinned_name, name);
        assert_eq!(base, "uint32_t");
        assert!(!pointer && !const_pointee);
    }
}

#[test]
fn the_kernel_arithmetic_matches_the_op_by_op_reference_on_exact_packed_bytes() {
    let mut rng = Rng(0x5EED_0045_0000_0001);
    // One released-geometry row keeps the scalar reference honest without a multi-second test; the
    // chunking test below uses a tiny geometry to exercise the row loop and the seam's chunking.
    let rows = 1usize;
    let latent = rng.latent(rows * Q_LORA_RANK);
    let raw_weight = rng.weight(COLS * Q_LORA_RANK);
    let weight_scales = rng.scales((COLS / 32) * K_BLOCKS);
    let tiled = pack_b12x(&raw_weight, COLS, Q_LORA_RANK);

    let (codes, scales) = act_quant_fp8_row(&latent[..Q_LORA_RANK]);
    let expected = dense_gemm(
        &codes,
        &scales,
        &tiled,
        &weight_scales,
        rows,
        COLS,
        Q_LORA_RANK,
    );
    let twinned = reference_projection(
        &latent,
        rows,
        &raw_weight,
        &weight_scales,
        COLS,
        Q_LORA_RANK,
    );
    assert_eq!(expected.len(), rows * COLS);
    assert_eq!(
        twinned, expected,
        "the op-by-op reference and mxfp8_ref disagree on the indexer shape"
    );
}

#[test]
fn rows_are_independent_so_a_chunked_launch_is_the_same_bytes() {
    // The seam chunks at eight rows; chunking is bit-transparent because every output row depends
    // only on its own activation row. The tiny configuration's geometry (2 x 8 = 16 columns,
    // q_lora_rank 32) keeps this fast while exercising the same code path, including a cols count
    // that is not a whole B12x N tile.
    let (cols, inner) = (16usize, 32usize);
    let rows = 5usize;
    let mut rng = Rng(0x5EED_0045_0000_0002);
    let latent = rng.latent(rows * inner);
    let raw_weight = rng.weight(cols * inner);
    let weight_scales = rng.scales(cols.div_ceil(32) * (inner / 32));
    let tiled = pack_b12x(&raw_weight, cols, inner);
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    for row in 0..rows {
        let (chunk_codes, chunk_scales) =
            act_quant_fp8_row(&latent[row * inner..(row + 1) * inner]);
        codes.extend_from_slice(&chunk_codes);
        scales.extend_from_slice(&chunk_scales);
    }
    let whole = dense_gemm(&codes, &scales, &tiled, &weight_scales, rows, cols, inner);
    let mut chunked = Vec::new();
    for start in 0..rows {
        let chunk = dense_gemm(
            &codes[start * inner..(start + 1) * inner],
            &scales[start * (inner / 32)..(start + 1) * (inner / 32)],
            &tiled,
            &weight_scales,
            1,
            cols,
            inner,
        );
        chunked.extend_from_slice(&chunk);
    }
    assert_eq!(chunked, whole, "chunked rows changed the bf16 output");
    // The op-by-op reference agrees with both on the same bytes.
    let twinned = reference_projection(&latent, rows, &raw_weight, &weight_scales, cols, inner);
    assert_eq!(twinned, whole);
}

#[test]
fn an_accumulation_grouping_change_flips_the_bf16_result_when_the_sum_cancels() {
    // Property demo of the mechanism only (the same category as the engine's bf16-margin demo): on
    // ordinary data the per-element order and the block-dot order differ in f32 by about 1e-5
    // relative, and the single bf16 round hides it. When the real sum cancels, the relative gap
    // grows past one bf16 ulp and the bf16 outputs differ - which is why the twin mirrors the
    // kernel's declared order instead of a "mathematically equivalent" one.
    //
    // Construction at the released reduction width: the first 20 blocks are random finite weight
    // codes, the last 20 are their sign-negations with the same scale, and the latent's second half
    // repeats the first so the act_quant scale codes pair up. Every pair's real contribution then
    // cancels exactly, and only the two accumulation groupings' rounding residuals remain.
    let inner = Q_LORA_RANK;
    let half = inner / 2;
    let half_blocks = K_BLOCKS / 2;
    let mut rng = Rng(0x5EED_0045_0000_0005);
    let base = rng.latent(half);
    let latent: Vec<f32> = base.iter().chain(base.iter()).copied().collect();
    let first_half = rng.weight(half);
    let mut raw_weight = first_half.clone();
    raw_weight.extend(first_half.iter().map(|code| code ^ 0x80));
    let mut weight_scales = rng.scales(half_blocks);
    let first_scales = weight_scales.clone();
    weight_scales.extend_from_slice(&first_scales);
    let (_, act_scales) = act_quant_fp8_row(&latent);
    assert_eq!(
        &act_scales[..half_blocks],
        &act_scales[half_blocks..],
        "the paired blocks must share their activation scale codes"
    );
    let per_element = reference_projection(&latent, 1, &raw_weight, &weight_scales, 1, inner);
    let block_dot = block_dot_projection(&latent, 1, &raw_weight, &weight_scales, 1, inner);
    assert_ne!(
        per_element, block_dot,
        "the two accumulations rounded identically on the cancellation fixture"
    );
}

#[test]
fn the_tiny_config_geometry_is_accepted_with_zero_padded_weight_tiles() {
    // The tiny configuration has index_n_heads * index_head_dim = 16 and q_lora_rank = 32, so the
    // weight is half of one B12x N tile: the tiled buffer's upper 16 columns are padding. The
    // kernel never reads col >= cols; the padding stays zero and the pack loses no real byte.
    let (cols, inner) = (16usize, 32usize);
    let mut rng = Rng(0x5EED_0045_0000_0004);
    let rows = 2usize;
    let latent = rng.latent(rows * inner);
    let raw_weight = rng.weight(cols * inner);
    let weight_scales = rng.scales(cols.div_ceil(32) * (inner / 32));
    let tiled = pack_b12x(&raw_weight, cols, inner);
    assert_eq!(tiled.len(), b12x_buffer_bytes(cols, inner));
    assert!(
        tiled.len() > cols * inner,
        "the tiny shape has tile padding"
    );
    // Every raw byte lands in the tiled buffer at a distinct index.
    let mut seen = vec![false; tiled.len()];
    for col in 0..cols {
        for k in 0..inner {
            let index = b12x_weight_index(col, k, inner);
            assert!(!seen[index], "the B12x tiling repeated a byte");
            assert_eq!(tiled[index], raw_weight[col * inner + k]);
            seen[index] = true;
        }
    }
    // The unused tail of the padded tile is zero, so a reference that reads it sees zeros.
    assert!(
        tiled
            .iter()
            .zip(&seen)
            .all(|(byte, real)| *real || *byte == 0)
    );
    let (codes, scales) = act_quant_fp8_row(&latent[..inner]);
    let expected = dense_gemm(&codes, &scales, &tiled, &weight_scales, 1, cols, inner);
    let twinned = reference_projection(
        &latent[..inner],
        1,
        &raw_weight,
        &weight_scales,
        cols,
        inner,
    );
    assert_eq!(twinned, expected);
}

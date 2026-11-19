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

//! The two shared token bodies on a device: the FP8 E4M3 few-row body
//! (`q4p_dense_fp8_rows`, Qwen's bf16 block scales; `glm_decode_fp8b128_rows`, GLM's f32
//! ones) and the packed BF16 body (`q4p_dense_packed_rows`), against the bulk tile each has to
//! reproduce.
//!
//! One arithmetic image per precision, and it is the tile's. For every token count in 1, 2, 4
//! and 8, this test launches the prompt tile - `q4p_dense_fp8` / `glm_prefill_fp8b128_mma`
//! for FP8, `q4p_dense_mma` for BF16 - and the token body over the same operands, and holds
//! the token body's row to the tile's row **bit for bit**: the same E4M3 -> bf16 words, the
//! same head-then-remainder MMA chain over K, and the same 128 x 128 block scales folded in
//! ascending block order.
//!
//! Two shapes a run: one the tile covers, and one it does not. The tile's column grid is
//! 32-wide (a 40-column matrix loses its last eight columns), so the 40-column case runs the
//! tile on a 64-column copy and compares the first 40 columns.
//!
//! The packed body reads the tiled layout of kernels/bf16_packed_sm121.cuh, escapes and offsets
//! included. The test builds that copy with the image's own `q4e_pack_bf16`, checks it unpacks
//! to the BF16 rows word for word, and then gives it to the packed body.
//!
//! Ignored by default (it needs a CUDA device); run on the CUDA host:
//!
//!   cargo test -p sparkflash-kernels --test token_bodies_device -- --ignored --nocapture

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_kernels::images::{
    glm_decode_cubin, glm_prefill_cubin, qwen4_exp_cubin, qwen4_prefill_cubin,
};
use sparkflash_kernels::m4_quant::{bf16_bits, fp8_e4m3_encode};
use sparkflash_kernels::test_io::{self, download, upload};

/// The token counts every claim is measured at.
const TOKENS: [usize; 4] = [1, 2, 4, 8];

fn open() -> Option<Cuda> {
    test_io::cuda_or_skip(
        (qwen4_prefill_cubin().is_empty()
            || qwen4_exp_cubin().is_empty()
            || glm_prefill_cubin().is_empty()
            || glm_decode_cubin().is_empty())
        .then_some("a shared model image"),
    )
}

/// A deterministic value stream: the test needs variety, not randomness, and a rerun compares the
/// same bits.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// An E4M3 weight matrix, a row a column, deterministic in `salt`.
fn e4m3_weights(cols: usize, inner: usize, salt: u64) -> Vec<u8> {
    (0..cols * inner)
        .map(|at| {
            let value = ((at as u64 * 37 + salt) % 251) as f32 * 0.013 - 1.6;
            fp8_e4m3_encode(value)
        })
        .collect()
}

/// The 128 x 128 block scales of an FP8 matrix, as bf16 words (Qwen's release) or f32 (GLM's).
fn scales(cols: usize, inner: usize, salt: u64, as_f32: bool) -> Vec<u8> {
    let count = cols.div_ceil(128) * (inner / 128);
    if as_f32 {
        (0..count)
            .flat_map(|at| {
                let value = 0.5 + ((at as u64 * 13 + salt) % 17) as f32 * 0.0625;
                value.to_le_bytes()
            })
            .collect()
    } else {
        (0..count)
            .flat_map(|at| {
                let shift = ((at as u64 + salt) % 7) as i32 - 3;
                bf16_bits(2.0f32.powi(shift)).to_le_bytes()
            })
            .collect()
    }
}

fn activations(tokens: usize, inner: usize, salt: u64) -> Vec<f32> {
    let mut stream = Lcg(salt);
    (0..tokens * inner).map(|_| stream.next()).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The row the token body computes must be the tile's row, bit for bit.
fn assert_rows_are_the_tiles(
    label: &str,
    tile: &[f32],
    rows: &[f32],
    tokens: usize,
    tile_cols: usize,
    cols: usize,
) {
    assert_eq!(
        tile.len(),
        tokens * tile_cols,
        "{label}: the tile's output size"
    );
    assert_eq!(
        rows.len(),
        tokens * cols,
        "{label}: the token body's output size"
    );
    for token in 0..tokens {
        for col in 0..cols {
            let at = tile[token * tile_cols + col];
            let got = rows[token * cols + col];
            assert_eq!(
                at.to_bits(),
                got.to_bits(),
                "{label}: row {token}, column {col}: the token body {got} and the tile {at} \
                 must agree bit for bit"
            );
        }
    }
}

/// One FP8 case: the same weights and activations through the prompt tile and the token body,
/// at every token count, with the token body held to the tile's row.
#[allow(clippy::too_many_arguments)]
fn check_fp8(
    cuda: &Cuda,
    tile_kernel: Kernel,
    rows_kernel: Kernel,
    label: &str,
    cols: usize,
    tile_cols: usize,
    inner: usize,
    scale_bytes: &[u8],
) {
    let tokens = TOKENS[TOKENS.len() - 1];
    let w = upload::<u8>(cuda, &e4m3_weights(tile_cols, inner, 11));
    let ws = upload::<u8>(cuda, scale_bytes);
    let a = upload::<u8>(cuda, &f32_bytes(&activations(tokens, inner, 7)));
    let tile_out = upload::<u8>(cuda, &vec![0u8; tokens * tile_cols * 4]);
    let rows_out = upload::<u8>(cuda, &vec![0u8; tokens * cols * 4]);
    for count in TOKENS {
        cuda.memset_d8_async(tile_out, 0xff, tokens * tile_cols * 4)
            .expect("poison the tile's output");
        cuda.memset_d8_async(rows_out, 0xff, tokens * cols * 4)
            .expect("poison the token body's output");
        cuda.launch(
            tile_kernel,
            (
                tile_cols.div_ceil(128) as u32,
                count.div_ceil(128) as u32,
                1,
            ),
            (256, 1, 1),
            &[
                w.0,
                ws.0,
                a.0,
                tile_out.0,
                tile_cols as u64,
                inner as u64,
                count as u64,
            ],
        )
        .expect("the prompt tile");
        cuda.launch(
            rows_kernel,
            (cols.div_ceil(32) as u32, 1, 1),
            (128, 1, 1),
            &[
                w.0,
                ws.0,
                a.0,
                rows_out.0,
                cols as u64,
                inner as u64,
                count as u64,
            ],
        )
        .expect("the token body");
        let tile: Vec<f32> = download::<u32>(cuda, &tile_out, count * tile_cols)
            .into_iter()
            .map(f32::from_bits)
            .collect();
        let rows: Vec<f32> = download::<u32>(cuda, &rows_out, count * cols)
            .into_iter()
            .map(f32::from_bits)
            .collect();
        assert_rows_are_the_tiles(label, &tile, &rows, count, tile_cols, cols);
    }
}

/// The FP8 E4M3 token bodies are the tile's rows: Qwen's bf16 block scales and GLM's f32 ones,
/// at the row counts a decode step, a verify and a short chunk take, on a whole tile and on a
/// 96-column partial one.
#[test]
#[ignore = "needs a CUDA device and the shared model images"]
fn the_fp8_token_bodies_are_the_tiles_rows() {
    let Some(cuda) = open() else { return };
    let (_qwen_tile_module, qwen_tile) = cuda
        .load_named(qwen4_prefill_cubin(), "q4p_dense_fp8")
        .expect("q4p_dense_fp8");
    let (_qwen_rows_module, qwen_rows) = cuda
        .load_named(qwen4_prefill_cubin(), "q4p_dense_fp8_rows")
        .expect("q4p_dense_fp8_rows");
    let (_glm_tile_module, glm_tile) = cuda
        .load_named(glm_prefill_cubin(), "glm_prefill_fp8b128_mma")
        .expect("glm_prefill_fp8b128_mma");
    let (_glm_rows_module, glm_rows) = cuda
        .load_named(glm_decode_cubin(), "glm_decode_fp8b128_rows")
        .expect("glm_decode_fp8b128_rows");

    for (cols, inner) in [(256usize, 512usize), (96, 384)] {
        check_fp8(
            &cuda,
            qwen_tile,
            qwen_rows,
            &format!("qwen fp8 {cols} x {inner}"),
            cols,
            cols,
            inner,
            &scales(cols, inner, 3, false),
        );
        check_fp8(
            &cuda,
            glm_tile,
            glm_rows,
            &format!("glm fp8 {cols} x {inner}"),
            cols,
            cols,
            inner,
            &scales(cols, inner, 5, true),
        );
    }

    // A shape the tile does not take: 40 columns. The tile's 32-column grid loses columns
    // 32..39, so it runs a 64-column copy of the same 64-row matrix and the token body runs the
    // first 40 columns.
    let inner = 384usize;
    check_fp8(
        &cuda,
        qwen_tile,
        qwen_rows,
        "qwen fp8 40 of 64 x 384",
        40,
        64,
        inner,
        &scales(64, inner, 17, false),
    );
    check_fp8(
        &cuda,
        glm_tile,
        glm_rows,
        "glm fp8 40 of 64 x 384",
        40,
        64,
        inner,
        &scales(64, inner, 19, true),
    );
}

struct PackedCopy {
    rows: Deviceptr,
    offsets: Deviceptr,
    entries: Deviceptr,
    row_bytes: usize,
}

/// The steps a packed row's walk takes, as crates/sparkflash-ops/src/packed.rs computes them: the
/// bytes one packed row holds.
fn packed_row_bytes(columns: usize) -> usize {
    let chunks = columns / 8;
    let (mut chunk, mut steps) = (0usize, 0usize);
    while chunk + 96 < chunks {
        chunk += 128;
        steps += 1;
    }
    if chunk + 32 < chunks {
        chunk += 64;
        steps += 1;
    }
    while chunk < chunks {
        chunk += 32;
        steps += 1;
    }
    (columns + columns / 2 + 32 * steps).next_multiple_of(16)
}

/// One packed matrix, built by the image's own packer in the tiled layout and checked against the
/// BF16 rows it stands for.
fn pack(
    cuda: &Cuda,
    pack_kernel: Kernel,
    unpack_kernel: Kernel,
    weights: &[u16],
    rows: usize,
    columns: usize,
) -> PackedCopy {
    let source = upload::<u8>(
        cuda,
        &weights
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let row_bytes = packed_row_bytes(columns);
    let counts = upload::<u32>(cuda, &vec![0u32; rows]);
    let zeros = upload::<u32>(cuda, &vec![0u32; rows + 1]);
    let grid = (rows.div_ceil(4) as u32, 1, 1);
    cuda.launch(
        pack_kernel,
        grid,
        (128, 1, 1),
        &[
            source.0,
            0,
            zeros.0,
            0,
            counts.0,
            rows as u64,
            columns as u64,
            row_bytes as u64,
            1,
        ],
    )
    .expect("the pack count");
    let counted: Vec<u32> = download::<u32>(cuda, &counts, rows);
    let mut offsets = vec![0u32; rows + 1];
    for row in 0..rows {
        offsets[row + 1] = offsets[row] + counted[row];
    }
    let total = offsets[rows] as usize;
    let offsets = upload::<u32>(cuda, &offsets);
    let entries = upload::<u32>(cuda, &vec![0u32; total.max(1)]);
    let packed = upload::<u8>(cuda, &vec![0u8; rows * row_bytes]);
    cuda.launch(
        pack_kernel,
        grid,
        (128, 1, 1),
        &[
            source.0,
            packed.0,
            offsets.0,
            entries.0,
            counts.0,
            rows as u64,
            columns as u64,
            row_bytes as u64,
            1,
        ],
    )
    .expect("the pack");
    // The copy must be the checkpoint's words: the packer's own unpack, compared word for word.
    let differ = upload::<u32>(cuda, &[0u32]);
    cuda.launch(
        unpack_kernel,
        grid,
        (128, 1, 1),
        &[
            packed.0,
            offsets.0,
            entries.0,
            0,
            source.0,
            differ.0,
            rows as u64,
            columns as u64,
            row_bytes as u64,
            1,
        ],
    )
    .expect("the unpack check");
    assert_eq!(
        download::<u32>(cuda, &differ, 1)[0],
        0,
        "the packed copy must unpack to the BF16 rows word for word"
    );
    PackedCopy {
        rows: packed,
        offsets,
        entries,
        row_bytes,
    }
}

/// The packed BF16 token body is the BF16 tile's rows, escapes and offsets included: at every
/// token count, over a whole matrix and over a 40-column shape the tile's grid does not take.
#[test]
#[ignore = "needs a CUDA device and the shared model images"]
fn the_packed_token_body_is_the_bf16_tiles_rows() {
    let Some(cuda) = open() else { return };
    let (_tile_module, bf16_tile) = cuda
        .load_named(qwen4_prefill_cubin(), "q4p_dense_mma")
        .expect("q4p_dense_mma");
    let (_rows_module, packed_rows) = cuda
        .load_named(qwen4_prefill_cubin(), "q4p_dense_packed_rows")
        .expect("q4p_dense_packed_rows");
    let (_pack_module, pack_kernel) = cuda
        .load_named(qwen4_exp_cubin(), "q4e_pack_bf16")
        .expect("q4e_pack_bf16");
    let (_unpack_module, unpack_kernel) = cuda
        .load_named(qwen4_exp_cubin(), "q4e_unpack_bf16")
        .expect("q4e_unpack_bf16");

    let inner = 512usize;
    let mut stream = Lcg(0x5eed);
    // A matrix with escapes on the path: zeros and values many binades under their step's
    // largest, which the form keeps as escapes.
    let weights: Vec<u16> = (0..512 * inner)
        .map(|at| match at % 997 {
            0 => 0,
            1 => bf16_bits(stream.next() * 1e-9),
            _ => bf16_bits(stream.next() * 0.05),
        })
        .collect();
    let a = upload::<u8>(&cuda, &f32_bytes(&activations(TOKENS[3], inner, 29)));

    for (cols, tile_cols) in [(512usize, 512usize), (40, 64)] {
        let copy = pack(
            &cuda,
            pack_kernel,
            unpack_kernel,
            &weights[..cols * inner],
            cols,
            inner,
        );
        // The tile reads the BF16 rows of the 64-column matrix; the packed body the 40 real
        // rows' copy. The tile's first 40 columns are the same arithmetic.
        let w = upload::<u8>(
            &cuda,
            &weights[..tile_cols * inner]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let tile_out = upload::<u8>(&cuda, &vec![0u8; TOKENS[3] * tile_cols * 4]);
        let rows_out = upload::<u8>(&cuda, &vec![0u8; TOKENS[3] * cols * 4]);
        let label = format!("packed bf16 {cols} of {tile_cols} x {inner}");
        for count in TOKENS {
            cuda.memset_d8_async(tile_out, 0xff, TOKENS[3] * tile_cols * 4)
                .expect("poison the tile's output");
            cuda.memset_d8_async(rows_out, 0xff, TOKENS[3] * cols * 4)
                .expect("poison the token body's output");
            cuda.launch(
                bf16_tile,
                (tile_cols.div_ceil(128) as u32, 1, 1),
                (256, 1, 1),
                &[
                    w.0,
                    a.0,
                    tile_out.0,
                    tile_cols as u64,
                    inner as u64,
                    count as u64,
                    1u64,
                ],
            )
            .expect("the BF16 tile");
            cuda.launch(
                packed_rows,
                (cols.div_ceil(32) as u32, 1, 1),
                (128, 1, 1),
                &[
                    a.0,
                    copy.rows.0,
                    copy.offsets.0,
                    copy.entries.0,
                    copy.row_bytes as u64,
                    rows_out.0,
                    cols as u64,
                    inner as u64,
                    count as u64,
                ],
            )
            .expect("the packed token body");
            let tile: Vec<f32> = download::<u32>(&cuda, &tile_out, count * tile_cols)
                .into_iter()
                .map(f32::from_bits)
                .collect();
            let rows: Vec<f32> = download::<u32>(&cuda, &rows_out, count * cols)
                .into_iter()
                .map(f32::from_bits)
                .collect();
            assert_rows_are_the_tiles(&label, &tile, &rows, count, tile_cols, cols);
        }
    }
}

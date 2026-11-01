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

//! Rows of a batched decode kernel against the same rows computed alone, bit for bit.
//!
//! A speculative verify runs the drafts as extra rows of one forward, and greedy output stays the
//! model's own only if every row computes exactly as the one-token step it stands in for. The
//! one-row and batched BF16 row dots used to group a row's chunk sums differently, and the
//! hyper-connection's one-token mix contracted its four stream terms into different FMAs; either
//! moved verify logits by ~1e-6 and flipped near-ties. The widths here take every branch of the
//! row dot's grouping (four chunks at a time, a pair, single chunks) at the released model's
//! projection widths.
//!
//! The tiled decode GEMV and its hyper-connection passes are held to the same bar, at shapes that
//! take one block a tile and shapes that split the inner dimension across blocks, and checked
//! against the one-warp-a-row kernels they stand in for to within f32 rounding. So is the
//! drafter's E4M3 GEMV, against the codes and row scales its quantiser writes.
//!
//! It skips - loudly - without an embedded cubin or a CUDA driver.

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_model_qwen4_exp::device::{
    TILED_GEMV_BLOCKS, TILED_GEMV_PARTIALS, cubin_available, embedded_cubin,
    tiled_gemv_e4m3_splits, tiled_gemv_splits,
};
use sparkflash_weights::quant::{e4m3_encode, f8e4m3_to_f32};

/// The decode GEMV's register tile, which is the widest verify the kernels take.
const BATCH: usize = 8;
/// One warp per output row, eight warps a block - the launch shape the session uses.
const WARPS: usize = 8;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn floats(&mut self, count: usize) -> Vec<u8> {
        (0..count).flat_map(|_| self.next().to_le_bytes()).collect()
    }

    fn bf16s(&mut self, count: usize) -> Vec<u8> {
        (0..count)
            .flat_map(|_| ((self.next().to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }
}

fn upload(cuda: &Cuda, bytes: &[u8], total: &mut usize) -> Deviceptr {
    let pointer = cuda.alloc(bytes.len(), total).expect("device buffer");
    cuda.copy_htod(&pointer, bytes).expect("upload");
    pointer
}

fn download(cuda: &Cuda, pointer: &Deviceptr, floats: usize) -> Vec<u32> {
    let mut bytes = vec![0_u8; floats * 4];
    cuda.copy_dtoh(&mut bytes, pointer).expect("read-back");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect()
}

/// Asserts that row `token` of `together` (`BATCH` rows of `width` outputs) is `alone(token)`.
fn assert_rows(
    what: &str,
    together: &[u32],
    width: usize,
    mut alone: impl FnMut(usize) -> Vec<u32>,
) {
    for token in 0..BATCH {
        let single = alone(token);
        let row = &together[token * width..(token + 1) * width];
        let differing = row.iter().zip(&single).filter(|(a, b)| a != b).count();
        assert_eq!(
            differing, 0,
            "{differing} of {width} outputs of token {token} differ between the batch and the \
             one-token {what}"
        );
    }
}

/// Asserts that `tiled` agrees with `reference` to the f32 rounding of a sum of `terms` products:
/// a random walk of `terms` roundings of the output's size, against the output's RMS. A wrong
/// weight or activation would be off by about the RMS itself.
fn assert_close(what: &str, tiled: &[u32], reference: &[u32], terms: usize) {
    let values = |bits: &[u32]| {
        bits.iter()
            .map(|bits| f32::from_bits(*bits))
            .collect::<Vec<_>>()
    };
    let (tiled, reference) = (values(tiled), values(reference));
    let rms = (reference.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>()
        / reference.len() as f64)
        .sqrt() as f32;
    let worst = tiled
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let bound = 16.0 * (terms as f32).sqrt() * f32::EPSILON * rms;
    assert!(
        worst <= bound,
        "{what}: the tiled kernel is {worst:e} from the one-warp-a-row kernel (bound {bound:e}, \
         rms {rms:e})"
    );
}

/// The split sums and arrival counters a tiled launch needs, the counters zeroed.
fn tiled_scratch(cuda: &Cuda, total: &mut usize) -> (Deviceptr, Deviceptr) {
    let partials = cuda
        .alloc(TILED_GEMV_PARTIALS * 4, total)
        .expect("partials");
    let counters = upload(cuda, &vec![0_u8; TILED_GEMV_BLOCKS * 4], total);
    (partials, counters)
}

fn cuda_or_skip() -> Option<Cuda> {
    if !cubin_available() {
        eprintln!("skipped: this build embeds no Qwen4-Exp cubin");
        return None;
    }
    match Cuda::load() {
        Ok(cuda) => Some(cuda),
        Err(error) => {
            eprintln!("skipped: no CUDA driver ({error})");
            None
        }
    }
}

#[test]
fn a_batched_gemv_row_is_bit_identical_to_the_row_alone() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let (_, gemv) = cuda
        .load_named(embedded_cubin(), "q4e_gemv_bf16")
        .expect("the cubin carries q4e_gemv_bf16");
    let mut rng = Lcg(0x5eed);
    // 2,560 is the hidden size, 10,240 the hyper-connection width and 4,416 leaves a pair and a
    // single chunk after the groups of four on some lanes.
    for columns in [2560_usize, 10_240, 4_416] {
        let rows = 96;
        let mut total = 0;
        let w = upload(&cuda, &rng.bf16s(rows * columns), &mut total);
        let x = upload(&cuda, &rng.floats(BATCH * columns), &mut total);
        let batched = cuda
            .alloc(BATCH * rows * 4, &mut total)
            .expect("batched out");
        let single = cuda.alloc(rows * 4, &mut total).expect("single out");
        let grid = (rows.div_ceil(WARPS) as u32, 1, 1);
        let block = ((WARPS * 32) as u32, 1, 1);
        cuda.launch(
            gemv,
            grid,
            block,
            &[
                w.0,
                x.0,
                batched.0,
                rows as u64,
                columns as u64,
                BATCH as u64,
            ],
        )
        .expect("batched launch");
        let together = download(&cuda, &batched, BATCH * rows);
        assert_rows(
            &format!("GEMV at {columns} columns"),
            &together,
            rows,
            |token| {
                cuda.launch(
                    gemv,
                    grid,
                    block,
                    &[
                        w.0,
                        x.offset(token * columns * 4).0,
                        single.0,
                        rows as u64,
                        columns as u64,
                        1,
                    ],
                )
                .expect("single launch");
                download(&cuda, &single, rows)
            },
        );
    }
}

#[test]
fn a_batched_hyper_connection_row_is_bit_identical_to_the_row_alone() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let (_, down) = cuda
        .load_named(embedded_cubin(), "q4e_hyper_down")
        .expect("the cubin carries q4e_hyper_down");
    let (_, up) = cuda
        .load_named(embedded_cubin(), "q4e_hyper_up")
        .expect("the cubin carries q4e_hyper_up");
    // The released model's four streams of 2,560 and its 320-wide low-rank mix.
    let (hc, hidden, lowrank) = (4_usize, 2560_usize, 320_usize);
    let width = hc * hidden;
    let mut rng = Lcg(0x4c0de);
    let mut total = 0;
    let normed = upload(&cuda, &rng.floats(BATCH * width), &mut total);
    let low_in = upload(&cuda, &rng.floats(BATCH * lowrank), &mut total);
    let down_w = upload(&cuda, &rng.bf16s(lowrank * width), &mut total);
    let inject_w = upload(&cuda, &rng.bf16s(hc * width), &mut total);
    let up_w = upload(&cuda, &rng.bf16s(width * lowrank), &mut total);
    let low = cuda.alloc(BATCH * lowrank * 4, &mut total).expect("low");
    let injection = cuda.alloc(BATCH * hc * 4, &mut total).expect("injection");
    let mixed = cuda.alloc(BATCH * hidden * 4, &mut total).expect("mixed");
    let block = ((WARPS * 32) as u32, 1, 1);
    let down_grid = ((lowrank + hc).div_ceil(WARPS) as u32, 1, 1);
    let up_grid = (hidden.div_ceil(WARPS) as u32, 1, 1);
    let run_down = |token: Option<usize>| {
        let (offset, rows) = token.map_or((0, BATCH), |token| (token, 1));
        cuda.launch(
            down,
            down_grid,
            block,
            &[
                normed.offset(offset * width * 4).0,
                down_w.0,
                inject_w.0,
                low.0,
                injection.0,
                hc as u64,
                width as u64,
                lowrank as u64,
                rows as u64,
            ],
        )
        .expect("hyper_down launch");
        (
            download(&cuda, &low, rows * lowrank),
            download(&cuda, &injection, rows * hc),
        )
    };
    let run_up = |token: Option<usize>| {
        let (offset, rows) = token.map_or((0, BATCH), |token| (token, 1));
        cuda.launch(
            up,
            up_grid,
            block,
            &[
                normed.offset(offset * width * 4).0,
                up_w.0,
                low_in.offset(offset * lowrank * 4).0,
                mixed.0,
                hc as u64,
                hidden as u64,
                lowrank as u64,
                rows as u64,
            ],
        )
        .expect("hyper_up launch");
        download(&cuda, &mixed, rows * hidden)
    };
    let (low_together, injection_together) = run_down(None);
    assert_rows(
        "hyper-connection low rank",
        &low_together,
        lowrank,
        |token| run_down(Some(token)).0,
    );
    assert_rows(
        "hyper-connection injection",
        &injection_together,
        hc,
        |token| run_down(Some(token)).1,
    );
    let mixed_together = run_up(None);
    assert_rows("hyper-connection mix", &mixed_together, hidden, |token| {
        run_up(Some(token))
    });
}

#[test]
fn a_tiled_gemv_row_is_bit_identical_to_the_row_alone() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let (_, tiled) = cuda
        .load_named(embedded_cubin(), "q4e_gemv_tiled_bf16")
        .expect("the cubin carries q4e_gemv_tiled_bf16");
    let (_, gemv) = cuda
        .load_named(embedded_cubin(), "q4e_gemv_bf16")
        .expect("the cubin carries q4e_gemv_bf16");
    let mut rng = Lcg(0x711ed);
    // The output projections (split three ways), a whole-tile matrix with no split, a ragged
    // last tile with a ragged last round, and a narrow split matrix.
    for (rows, columns) in [
        (2560_usize, 6144_usize),
        (6400, 4096),
        (300, 4424),
        (40, 10_240),
    ] {
        let tiles = rows.div_ceil(32);
        let splits = tiled_gemv_splits(tiles, columns);
        let mut total = 0;
        let (partials, counters) = tiled_scratch(&cuda, &mut total);
        let w = upload(&cuda, &rng.bf16s(rows * columns), &mut total);
        let x = upload(&cuda, &rng.floats(BATCH * columns), &mut total);
        let out = cuda.alloc(BATCH * rows * 4, &mut total).expect("out");
        let run = |kernel, grid, x: Deviceptr, batch: usize, extra: bool| {
            let mut args = vec![w.0, x.0, out.0, rows as u64, columns as u64, batch as u64];
            if extra {
                args.extend([partials.0, counters.0]);
            }
            cuda.launch(kernel, grid, ((WARPS * 32) as u32, 1, 1), &args)
                .expect("launch");
            download(&cuda, &out, batch * rows)
        };
        let grid = (tiles as u32, splits as u32, 1);
        let together = run(tiled, grid, x, BATCH, true);
        let what = format!("tiled GEMV {rows}x{columns} in {splits} splits");
        assert_rows(&what, &together, rows, |token| {
            run(tiled, grid, x.offset(token * columns * 4), 1, true)
        });
        let reference = run(gemv, (rows.div_ceil(WARPS) as u32, 1, 1), x, BATCH, false);
        assert_close(&what, &together, &reference, columns);
    }
}

#[test]
fn an_e4m3_gemv_row_is_its_quantised_matrix_and_the_row_alone() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let (_, quantize) = cuda
        .load_named(embedded_cubin(), "q4e_quantize_e4m3_rows")
        .expect("the cubin carries q4e_quantize_e4m3_rows");
    let (_, tiled) = cuda
        .load_named(embedded_cubin(), "q4e_gemv_tiled_e4m3")
        .expect("the cubin carries q4e_gemv_tiled_e4m3");
    let mut rng = Lcg(0xe4a3);
    // The drafter's query projection (split two ways) and down projection (three ways), a ragged
    // last tile with a last round of one chunk, and a narrow split matrix.
    for (rows, columns) in [
        (4096_usize, 2560_usize),
        (2560, 7680),
        (300, 2576),
        (40, 10_240),
    ] {
        let tiles = rows.div_ceil(32);
        let splits = tiled_gemv_e4m3_splits(tiles, columns);
        let mut total = 0;
        let (partials, counters) = tiled_scratch(&cuda, &mut total);
        let bf16 = rng.bf16s(rows * columns);
        let w = upload(&cuda, &bf16, &mut total);
        let codes = cuda.alloc(rows * columns, &mut total).expect("codes");
        let scales = cuda.alloc(rows * 4, &mut total).expect("scales");
        cuda.launch(
            quantize,
            (rows.div_ceil(WARPS) as u32, 1, 1),
            ((WARPS * 32) as u32, 1, 1),
            &[w.0, codes.0, scales.0, rows as u64, columns as u64],
        )
        .expect("quantize");
        let mut code_bytes = vec![0_u8; rows * columns];
        cuda.copy_dtoh(&mut code_bytes, &codes).expect("codes back");
        let scale_bits = download(&cuda, &scales, rows);
        // The quantiser: the row's largest magnitude over 448, and each weight's nearest code.
        let weights: Vec<f32> = bf16
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16))
            .collect();
        for row in 0..rows {
            let values = &weights[row * columns..(row + 1) * columns];
            let largest = values.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            let scale = if largest > 0.0 { largest / 448.0 } else { 1.0 };
            assert_eq!(scale_bits[row], scale.to_bits(), "row {row}'s scale");
            let wrong = values
                .iter()
                .zip(&code_bytes[row * columns..(row + 1) * columns])
                .filter(|(value, code)| e4m3_encode(*value / scale) != **code)
                .count();
            assert_eq!(
                wrong, 0,
                "{wrong} of row {row}'s {columns} codes are not the nearest"
            );
        }
        let x_bytes = rng.floats(BATCH * columns);
        let x = upload(&cuda, &x_bytes, &mut total);
        let out = cuda.alloc(BATCH * rows * 4, &mut total).expect("out");
        let grid = (tiles as u32, splits as u32, 1);
        let run = |x: Deviceptr, batch: usize| {
            cuda.launch(
                tiled,
                grid,
                ((WARPS * 32) as u32, 1, 1),
                &[
                    codes.0,
                    scales.0,
                    x.0,
                    out.0,
                    rows as u64,
                    columns as u64,
                    batch as u64,
                    partials.0,
                    counters.0,
                ],
            )
            .expect("launch");
            download(&cuda, &out, batch * rows)
        };
        let together = run(x, BATCH);
        let what = format!("E4M3 GEMV {rows}x{columns} in {splits} splits");
        assert_rows(&what, &together, rows, |token| {
            run(x.offset(token * columns * 4), 1)
        });
        // Against the decoded codes, summed in f64 and rounded once.
        let xs: Vec<f32> = x_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        let reference: Vec<u32> = (0..BATCH)
            .flat_map(|token| {
                let x_row = &xs[token * columns..(token + 1) * columns];
                let (code_bytes, scale_bits) = (&code_bytes, &scale_bits);
                (0..rows).map(move |row| {
                    let dot: f64 = code_bytes[row * columns..(row + 1) * columns]
                        .iter()
                        .zip(x_row)
                        .map(|(code, x)| f64::from(f8e4m3_to_f32(*code)) * f64::from(*x))
                        .sum();
                    ((dot * f64::from(f32::from_bits(scale_bits[row]))) as f32).to_bits()
                })
            })
            .collect();
        assert_close(&what, &together, &reference, columns);
    }
}

#[test]
fn a_tiled_hyper_connection_row_is_bit_identical_to_the_row_alone() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let load = |name| {
        cuda.load_named(embedded_cubin(), name)
            .unwrap_or_else(|error| panic!("{name}: {error}"))
            .1
    };
    let (down, up) = (load("q4e_hyper_down"), load("q4e_hyper_up"));
    let (down_tiled, up_tiled) = (load("q4e_hyper_down_tiled"), load("q4e_hyper_up_tiled"));
    let (hc, hidden, lowrank) = (4_usize, 2560_usize, 320_usize);
    let width = hc * hidden;
    let mut rng = Lcg(0x4c0de5);
    let mut total = 0;
    let (partials, counters) = tiled_scratch(&cuda, &mut total);
    let normed = upload(&cuda, &rng.floats(BATCH * width), &mut total);
    let low_in = upload(&cuda, &rng.floats(BATCH * lowrank), &mut total);
    let down_w = upload(&cuda, &rng.bf16s(lowrank * width), &mut total);
    let inject_w = upload(&cuda, &rng.bf16s(hc * width), &mut total);
    let up_w = upload(&cuda, &rng.bf16s(width * lowrank), &mut total);
    let low = cuda.alloc(BATCH * lowrank * 4, &mut total).expect("low");
    let injection = cuda.alloc(BATCH * hc * 4, &mut total).expect("injection");
    let mixed = cuda.alloc(BATCH * hidden * 4, &mut total).expect("mixed");
    let block = ((WARPS * 32) as u32, 1, 1);
    let tiles = (lowrank + hc).div_ceil(32);
    let splits = tiled_gemv_splits(tiles, width);
    let run_down = |token: Option<usize>, tiled: bool| {
        let (offset, rows) = token.map_or((0, BATCH), |token| (token, 1));
        let mut args = vec![
            normed.offset(offset * width * 4).0,
            down_w.0,
            inject_w.0,
            low.0,
            injection.0,
            hc as u64,
            width as u64,
            lowrank as u64,
            rows as u64,
        ];
        let grid = if tiled {
            args.extend([partials.0, counters.0]);
            (tiles as u32, splits as u32, 1)
        } else {
            ((lowrank + hc).div_ceil(WARPS) as u32, 1, 1)
        };
        cuda.launch(if tiled { down_tiled } else { down }, grid, block, &args)
            .expect("hyper_down launch");
        (
            download(&cuda, &low, rows * lowrank),
            download(&cuda, &injection, rows * hc),
        )
    };
    let run_up = |token: Option<usize>, tiled: bool| {
        let (offset, rows) = token.map_or((0, BATCH), |token| (token, 1));
        let grid = if tiled {
            (hidden.div_ceil(8) as u32, 1, 1)
        } else {
            (hidden.div_ceil(WARPS) as u32, 1, 1)
        };
        cuda.launch(
            if tiled { up_tiled } else { up },
            grid,
            block,
            &[
                normed.offset(offset * width * 4).0,
                up_w.0,
                low_in.offset(offset * lowrank * 4).0,
                mixed.0,
                hc as u64,
                hidden as u64,
                lowrank as u64,
                rows as u64,
            ],
        )
        .expect("hyper_up launch");
        download(&cuda, &mixed, rows * hidden)
    };
    let (low_together, injection_together) = run_down(None, true);
    assert_rows("tiled low rank", &low_together, lowrank, |token| {
        run_down(Some(token), true).0
    });
    assert_rows("tiled injection", &injection_together, hc, |token| {
        run_down(Some(token), true).1
    });
    let (low_reference, injection_reference) = run_down(None, false);
    assert_close("tiled low rank", &low_together, &low_reference, width);
    assert_close(
        "tiled injection",
        &injection_together,
        &injection_reference,
        width,
    );
    let mixed_together = run_up(None, true);
    assert_rows("tiled mix", &mixed_together, hidden, |token| {
        run_up(Some(token), true)
    });
    assert_close("tiled mix", &mixed_together, &run_up(None, false), lowrank);
}

#[test]
fn a_replayed_recurrence_is_the_state_its_rows_left_behind() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    // The released head shape (128 x 128), fewer heads: two key heads of three value heads each.
    let (heads, key_heads, dim) = (6_usize, 2_usize, 128_usize);
    let key_width = key_heads * dim;
    let value_width = heads * dim;
    let conv_width = 2 * key_width + value_width;
    let cells = heads * dim * dim;
    let mut rng = Lcg(0x6d17);
    let mut total = 0;
    let mixed = upload(&cuda, &rng.floats(BATCH * conv_width), &mut total);
    let z = upload(&cuda, &rng.floats(BATCH * value_width), &mut total);
    let b = upload(&cuda, &rng.floats(BATCH * heads), &mut total);
    let a = upload(&cuda, &rng.floats(BATCH * heads), &mut total);
    let a_log = upload(&cuda, &rng.bf16s(heads), &mut total);
    let dt_bias = upload(&cuda, &rng.bf16s(heads), &mut total);
    let norm = upload(&cuda, &rng.bf16s(dim), &mut total);
    let start = rng.floats(cells);
    let state = cuda.alloc(cells * 4, &mut total).expect("state");
    let other = cuda.alloc(cells * 4, &mut total).expect("other state");
    let out = cuda
        .alloc(BATCH * value_width * 4, &mut total)
        .expect("out");
    let staged = cuda
        .load_named_shared(embedded_cubin(), "q4e_gdn_recur", (dim * dim * 4) as u32)
        .expect("q4e_gdn_recur with its staged state")
        .1;
    let global = cuda
        .load_named(embedded_cubin(), "q4e_gdn_recur")
        .expect("q4e_gdn_recur")
        .1;
    for (label, kernel) in [("staged", staged), ("global", global)] {
        // Rows from `first`, `rows` of them, from `from` into `to`, outputs to `out` unless `out`
        // is 0.
        let run = |first: usize, rows: usize, from: &Deviceptr, to: &Deviceptr, out: u64| {
            cuda.launch(
                kernel,
                (heads as u32, 1, 1),
                (128, 1, 1),
                &[
                    mixed.offset(first * conv_width * 4).0,
                    z.offset(first * value_width * 4).0,
                    b.offset(first * heads * 4).0,
                    a.offset(first * heads * 4).0,
                    a_log.0,
                    dt_bias.0,
                    norm.0,
                    from.0,
                    to.0,
                    out,
                    key_width as u64,
                    heads as u64,
                    dim as u64,
                    dim as u64,
                    (heads / key_heads) as u64,
                    1e-6_f32.to_bits() as u64,
                    1,
                    rows as u64,
                    u64::from(kernel.shared_bytes()),
                    // No channel decay: Gated DeltaNet decays a head by one factor.
                    0,
                ],
            )
            .expect("gdn_recur launch");
        };
        // The one-token steps a verify stands in for, the state after each row kept.
        cuda.copy_htod(&state, &start).expect("start state");
        let mut alone_states = Vec::with_capacity(BATCH);
        let mut alone_outputs = Vec::with_capacity(BATCH);
        for token in 0..BATCH {
            run(token, 1, &state, &state, out.0);
            alone_outputs.push(download(&cuda, &out, value_width));
            alone_states.push(download(&cuda, &state, cells));
        }
        // The verify: every row's output as the step's, the state as it was found, and the last
        // row's state in the spare.
        cuda.copy_htod(&state, &start).expect("start state");
        run(0, BATCH, &state, &other, out.0);
        let together = download(&cuda, &out, BATCH * value_width);
        assert_rows(
            &format!("{label} recurrence"),
            &together,
            value_width,
            |token| alone_outputs[token].clone(),
        );
        let start_bits: Vec<u32> = start
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect();
        assert!(
            download(&cuda, &state, cells) == start_bits,
            "the {label} verify wrote the state it read"
        );
        assert!(
            download(&cuda, &other, cells) == alone_states[BATCH - 1],
            "the {label} verify's spare is not the last row's state"
        );
        // The replay of every kept prefix, from the state the verify read, into the other buffer.
        for kept in 1..=BATCH {
            run(0, kept, &state, &other, 0);
            assert!(
                download(&cuda, &other, cells) == alone_states[kept - 1],
                "the {label} replay of {kept} rows is not the state those rows left"
            );
        }
        assert!(
            download(&cuda, &state, cells) == start_bits,
            "the {label} replay wrote the state it read"
        );
    }
}

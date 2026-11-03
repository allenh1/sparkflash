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

//! What one shared-memory GEMM launch costs at the shapes the model actually asks for.
//!
//! The kernel's row tile shrinks with the block, so a matrix with few rows can be cut into more
//! blocks by launching fewer threads - the trade the profile says matters for `hyper_down`, where
//! 64 rows a block leaves five blocks on a GPU with forty-eight SMs.
//!
//! ```sh
//! cargo test -p sparkflash-model-qwen4-exp --test gemm_bench -- --ignored --nocapture
//! ```

use std::time::Instant;

use sparkflash_kernels::cuda_driver::{Cuda, Kernel};
use sparkflash_model_qwen4_exp::device::{cubin_available, embedded_cubin};

const ROUNDS: usize = 20;

fn open() -> Option<Cuda> {
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

fn measure(
    cuda: &Cuda,
    kernel: Kernel,
    shape: (usize, usize),
    threads: usize,
    batch: usize,
) -> f64 {
    let (rows, columns) = shape;
    let mut total = 0usize;
    let mut weight_bytes = vec![0u8; rows * columns * 2];
    for (index, byte) in weight_bytes.iter_mut().enumerate() {
        *byte = (index.wrapping_mul(37) >> 3) as u8;
    }
    let w = cuda.alloc(weight_bytes.len(), &mut total).expect("weights");
    cuda.copy_htod(&w, &weight_bytes).expect("upload weights");
    let x_bytes: Vec<u8> = (0..batch * columns)
        .flat_map(|index| (((index % 17) as f32) * 0.031).to_le_bytes())
        .collect();
    let x = cuda.alloc(x_bytes.len(), &mut total).expect("x");
    cuda.copy_htod(&x, &x_bytes).expect("upload x");
    let y = cuda.alloc(batch * rows * 4, &mut total).expect("y");
    let rows_per_block = threads / 2;
    let grid = (rows.div_ceil(rows_per_block) as u32, 1, 1);
    let launch = || {
        cuda.launch(
            kernel,
            grid,
            (threads as u32, 1, 1),
            &[
                w.0,
                x.0,
                y.0,
                rows as u64,
                columns as u64,
                batch as u64,
                0_u64,
                u64::from(1.0_f32.to_bits()),
                1_u64,
                0_u64,
            ],
        )
        .expect("gemm launch");
        cuda.synchronize().expect("drain");
    };
    launch();
    let mut best = f64::INFINITY;
    for _ in 0..ROUNDS {
        let start = Instant::now();
        launch();
        best = best.min(start.elapsed().as_secs_f64() * 1e3);
    }
    best
}

/// The same product computed whole and in `segments` slices, back to back.
fn split_agreement(
    cuda: &Cuda,
    kernel: Kernel,
    rows: usize,
    columns: usize,
    segments: usize,
) -> (Vec<f32>, Vec<f32>) {
    let batch = 128usize;
    let count = batch * rows;
    let mut total = 0usize;
    let weights: Vec<u8> = (0..rows * columns * 2)
        .map(|index| (index.wrapping_mul(37) >> 3) as u8)
        .collect();
    let w = cuda.alloc(weights.len(), &mut total).expect("weights");
    cuda.copy_htod(&w, &weights).expect("upload weights");
    let x_values: Vec<f32> = (0..batch * columns)
        .map(|index| ((index % 17) as f32) * 0.031 - 0.2)
        .collect();
    let x_bytes: Vec<u8> = x_values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let x = cuda.alloc(x_bytes.len(), &mut total).expect("x");
    cuda.copy_htod(&x, &x_bytes).expect("upload x");
    let whole_out = cuda.alloc(count * 4, &mut total).expect("whole out");
    let split_out = cuda.alloc(count * 4, &mut total).expect("split out");
    let buffer = cuda
        .alloc(segments * count * 4, &mut total)
        .expect("partials");
    let (_, reduce) = cuda
        .load_named(embedded_cubin(), "q4e_gemm_reduce")
        .expect("reduce");
    let run = |segments: usize, partials: u64, out: sparkflash_kernels::cuda_driver::Deviceptr| {
        cuda.launch(
            kernel,
            (rows.div_ceil(64) as u32, segments as u32, 1),
            (128, 1, 1),
            &[
                w.0,
                x.0,
                out.0,
                rows as u64,
                columns as u64,
                batch as u64,
                0_u64,
                u64::from(1.0_f32.to_bits()),
                segments as u64,
                partials,
            ],
        )
        .expect("gemm");
        if segments > 1 {
            cuda.launch(
                reduce,
                (count.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                &[
                    partials,
                    out.0,
                    count as u64,
                    segments as u64,
                    0_u64,
                    u64::from(1.0_f32.to_bits()),
                ],
            )
            .expect("reduce");
        }
        cuda.synchronize().expect("drain");
        let mut host = vec![0u8; count * 4];
        cuda.copy_dtoh(&mut host, &out).expect("out back");
        host.as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<f32>>()
    };
    let whole = run(1, 0, whole_out);
    let split = run(segments, buffer.0, split_out);
    (whole, split)
}

#[test]
#[ignore = "allocates and launches real kernels; run with --ignored --nocapture"]
fn the_gemm_cost_at_the_models_shapes() {
    let Some(cuda) = open() else {
        return;
    };
    let (_, kernel) = cuda
        .load_named(embedded_cubin(), "q4e_gemm_bf16")
        .expect("q4e_gemm_bf16");
    let shapes = [
        (320_usize, 10240_usize),
        (320, 1280),
        (10240, 320),
        (2560, 2560),
        (640, 2560),
        (2560, 640),
    ];
    for shape in shapes {
        let mut line = format!("{:>6} x {:<6}", shape.0, shape.1);
        let mut best_threads = 0;
        let mut best = f64::INFINITY;
        for threads in [128_usize, 64, 32] {
            let elapsed = measure(&cuda, kernel, shape, threads, 128);
            line += &format!("  {threads:>4}t {elapsed:7.3} ms");
            if elapsed < best {
                best = elapsed;
                best_threads = threads;
            }
        }
        eprintln!("{line}   best {best_threads}t {best:.3} ms");
    }
    // The split path must agree with the single-block path: same products, summed in two
    // different associations.
    for (rows, columns) in [(320_usize, 10240_usize), (640, 2560), (2560, 640)] {
        for segments in [2usize, 4, 8] {
            let (whole, split) = split_agreement(&cuda, kernel, rows, columns, segments);
            let worst = whole
                .iter()
                .zip(&split)
                .map(|(a, b)| (a - b).abs() / a.abs().max(1e-6))
                .fold(0.0_f32, f32::max);
            eprintln!(
                "{rows} x {columns} in {segments} segments: worst relative difference {worst:.2e}"
            );
            assert!(worst < 1e-5, "the split sum disagrees with the whole one");
        }
    }
    // How the narrow shape scales with the token count says whether it is the activation
    // staging or the arithmetic that costs.
    for batch in [128_usize, 32, 8] {
        let elapsed = measure(&cuda, kernel, (320, 10240), 128, batch);
        eprintln!("320 x 10240 at batch {batch:>3}: {elapsed:.3} ms");
    }
}

/// A prefill chunk's whole 128-token tiles in one launch (blockIdx.z a tile) give every token the
/// bits a launch of its tile alone gives it, split or not: the narrow projections' bits do not
/// move when the tiles share a launch.
#[test]
fn tiles_in_one_launch_match_launches_of_their_own() {
    let Some(cuda) = open() else {
        return;
    };
    let (_, kernel) = cuda
        .load_named(embedded_cubin(), "q4e_gemm_bf16")
        .expect("q4e_gemm_bf16");
    let (_, reduce) = cuda
        .load_named(embedded_cubin(), "q4e_gemm_reduce")
        .expect("q4e_gemm_reduce");
    // A TP=4 rank's Gated DeltaNet `a` projection: eight rows of the 2,560-wide hidden row.
    let (rows, columns, batch, tiles) = (8_usize, 2560_usize, 128_usize, 3_usize);
    let count = batch * rows;
    let mut total = 0usize;
    let weights: Vec<u8> = (0..rows * columns * 2)
        .map(|index| (index.wrapping_mul(37) >> 3) as u8 & 0x3f)
        .collect();
    let w = cuda.alloc(weights.len(), &mut total).expect("weights");
    cuda.copy_htod(&w, &weights).expect("upload weights");
    let x_bytes: Vec<u8> = (0..tiles * batch * columns)
        .flat_map(|index| ((((index * 7919) % 211) as f32) * 0.013 - 1.3).to_le_bytes())
        .collect();
    let x = cuda.alloc(x_bytes.len(), &mut total).expect("x");
    cuda.copy_htod(&x, &x_bytes).expect("upload x");
    let download = |out: &sparkflash_kernels::cuda_driver::Deviceptr| {
        cuda.synchronize().expect("drain");
        let mut host = vec![0u8; tiles * count * 4];
        cuda.copy_dtoh(&mut host, out).expect("out back");
        host.as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect::<Vec<u32>>()
    };
    for segments in [1_usize, 48] {
        let partials = cuda
            .alloc(tiles * segments * count * 4, &mut total)
            .expect("partials");
        let partials = if segments > 1 { partials.0 } else { 0 };
        let mut outputs = Vec::new();
        // One launch of all the tiles, then a launch a tile.
        for (launches, per_launch) in [(1_usize, tiles), (tiles, 1)] {
            let out = cuda.alloc(tiles * count * 4, &mut total).expect("out");
            for launch in 0..launches {
                let first = launch * per_launch;
                let at = out.offset(first * count * 4);
                cuda.launch(
                    kernel,
                    (rows.div_ceil(64) as u32, segments as u32, per_launch as u32),
                    (128, 1, 1),
                    &[
                        w.0,
                        x.offset(first * batch * columns * 4).0,
                        at.0,
                        rows as u64,
                        columns as u64,
                        batch as u64,
                        // The silu epilogue, so the reduce's tail is covered too.
                        1,
                        u64::from(1.5_f32.to_bits()),
                        segments as u64,
                        partials,
                    ],
                )
                .expect("gemm");
                if segments > 1 {
                    cuda.launch(
                        reduce,
                        (count.div_ceil(256) as u32, per_launch as u32, 1),
                        (256, 1, 1),
                        &[
                            partials,
                            at.0,
                            count as u64,
                            segments as u64,
                            1,
                            u64::from(1.5_f32.to_bits()),
                        ],
                    )
                    .expect("reduce");
                }
            }
            outputs.push(download(&out));
        }
        assert_eq!(
            outputs[0], outputs[1],
            "{segments} segments: the tiles in one launch differ from a launch each"
        );
        assert!(
            outputs[0].iter().any(|bits| *bits != 0),
            "nothing was written"
        );
    }
}

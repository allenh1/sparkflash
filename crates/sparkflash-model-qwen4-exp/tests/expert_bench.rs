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

//! What the routed-expert kernels cost, against how many experts a chunk actually touches.
//!
//! The run-leader schedule reads an expert's rows from DRAM once per chunk and lets the run's
//! later slots hit them in L2, so its benefit is exactly the gap between the slot count and the
//! distinct-expert count. This launches the real kernels against a synthetic table at a chosen
//! number of distinct experts, which is the only way to tell "the schedule did not help" from
//! "the schedule is not being given duplicates".
//!
//! It runs only when asked for (it allocates a couple of GiB):
//!
//! ```sh
//! cargo test -p sparkflash-model-qwen4-exp --test expert_bench -- --ignored --nocapture
//! ```

use std::time::Instant;

use sparkflash_kernels::cuda_driver::Cuda;
use sparkflash_model_qwen4_exp::device::{cubin_available, embedded_moe_cubin};

const HIDDEN: usize = 2560;
const INTER: usize = 640;
const TOPK: usize = 10;
const BATCH: usize = 64;
/// Rows a warp owns, mirroring `Q4E_ROWS_PER_WARP` in the kernel.
const ROWS: usize = 2;
/// Rows a down-pass warp owns, mirroring `Q4E_DOWN_ROWS_PER_WARP`.
const DOWN_ROWS: usize = 4;
const ROUNDS: usize = 40;

/// Bytes one expert's six projections occupy: gate/up are HIDDEN wide, down is INTER wide.
const fn expert_bytes() -> usize {
    let hidden_packed = HIDDEN / 2;
    let hidden_scales = HIDDEN / 16;
    let inter_packed = INTER / 2;
    let inter_scales = INTER / 16;
    2 * (INTER * (hidden_packed + hidden_scales)) + HIDDEN * (inter_packed + inter_scales)
}

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

fn u64_bytes(values: &[u64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Launch the three routed-expert kernels once and return the elapsed milliseconds.
#[allow(clippy::too_many_arguments)]
fn run_once(
    cuda: &Cuda,
    table: &sparkflash_kernels::cuda_driver::Deviceptr,
    scales: &sparkflash_kernels::cuda_driver::Deviceptr,
    indices: &sparkflash_kernels::cuda_driver::Deviceptr,
    perm: &sparkflash_kernels::cuda_driver::Deviceptr,
    runs: &sparkflash_kernels::cuda_driver::Deviceptr,
    x: &sparkflash_kernels::cuda_driver::Deviceptr,
    activated: &sparkflash_kernels::cuda_driver::Deviceptr,
    partials: &sparkflash_kernels::cuda_driver::Deviceptr,
    weights: &sparkflash_kernels::cuda_driver::Deviceptr,
    slots: usize,
    warps: usize,
) -> f64 {
    let (_, permute) = cuda
        .load_named(sparkflash_kernels::images::moe_plan_cubin(), "moe_plan")
        .expect("moe_plan");
    let (_, gate_up) = cuda
        .load_named(embedded_moe_cubin(), "q4e_expert_gate_up")
        .expect("q4e_expert_gate_up");
    let (_, down) = cuda
        .load_named(embedded_moe_cubin(), "q4e_expert_down")
        .expect("q4e_expert_down");
    let start = Instant::now();
    cuda.launch(
        permute,
        (1, 1, 1),
        (256, 1, 1),
        &[
            indices.0,
            perm.0,
            0,
            0,
            runs.0,
            slots as u64,
            512,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    .expect("plan");
    cuda.launch(
        gate_up,
        (
            INTER.div_ceil(warps * ROWS) as u32,
            TOPK as u32,
            BATCH as u32,
        ),
        ((warps * 32) as u32, 1, 1),
        &[
            table.0,
            scales.0,
            indices.0,
            perm.0,
            runs.0,
            x.0,
            activated.0,
            TOPK as u64,
            HIDDEN as u64,
            INTER as u64,
        ],
    )
    .expect("gate up");
    cuda.launch(
        down,
        (
            HIDDEN.div_ceil(warps * DOWN_ROWS) as u32,
            TOPK as u32,
            BATCH as u32,
        ),
        ((warps * 32) as u32, 1, 1),
        &[
            table.0,
            scales.0,
            indices.0,
            perm.0,
            runs.0,
            weights.0,
            activated.0,
            partials.0,
            TOPK as u64,
            HIDDEN as u64,
            INTER as u64,
            512_u64,
        ],
    )
    .expect("down");
    cuda.synchronize().expect("drain");
    start.elapsed().as_secs_f64() * 1e3
}

/// Time the routed-expert kernels with `distinct` experts covering `TOPK * BATCH` slots.
fn measure(cuda: &Cuda, distinct: usize, warps: usize) -> f64 {
    let slots = TOPK * BATCH;
    let mut total = 0_usize;
    let backing_bytes = distinct * expert_bytes();
    let backing = cuda.alloc(backing_bytes, &mut total).expect("weights");
    cuda.copy_htod(&backing, &vec![0_u8; backing_bytes])
        .expect("zero weights");
    let mut table = Vec::with_capacity(distinct * 6);
    for expert in 0..distinct {
        let base = backing.0 + (expert * expert_bytes()) as u64;
        let hidden_packed = HIDDEN / 2;
        let hidden_scales = HIDDEN / 16;
        let inter_packed = INTER / 2;
        let gate = base;
        let gate_scales = gate + (INTER * hidden_packed) as u64;
        let up = gate_scales + (INTER * hidden_scales) as u64;
        let up_scales = up + (INTER * hidden_packed) as u64;
        let down = up_scales + (INTER * hidden_scales) as u64;
        let down_scales = down + (HIDDEN * inter_packed) as u64;
        table.extend_from_slice(&[gate, gate_scales, up, up_scales, down, down_scales]);
    }
    let table_buffer = cuda.alloc(table.len() * 8, &mut total).expect("table");
    cuda.copy_htod(&table_buffer, &u64_bytes(&table))
        .expect("upload table");
    let scale_values: Vec<f32> = (0..distinct * 3).map(|_| 1.0).collect();
    let scale_bytes: Vec<u8> = scale_values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let scale_buffer = cuda.alloc(scale_bytes.len(), &mut total).expect("scales");
    cuda.copy_htod(&scale_buffer, &scale_bytes)
        .expect("upload scales");
    let selected: Vec<i32> = (0..slots).map(|slot| (slot % distinct) as i32).collect();
    let index_buffer = cuda.alloc(slots * 4, &mut total).expect("indices");
    cuda.copy_htod(&index_buffer, &i32_bytes(&selected))
        .expect("upload indices");
    let perm = cuda.alloc(slots * 4, &mut total).expect("perm");
    let runs = cuda.alloc(slots * 2 * 4, &mut total).expect("runs");
    let x_bytes = BATCH * HIDDEN * 4;
    let x = cuda.alloc(x_bytes, &mut total).expect("x");
    cuda.copy_htod(&x, &vec![0_u8; x_bytes]).expect("zero x");
    let activated = cuda
        .alloc(slots * INTER * 4, &mut total)
        .expect("activated");
    let partials = cuda
        .alloc(slots * HIDDEN * 4, &mut total)
        .expect("partials");
    let weights = cuda
        .alloc(BATCH * 512 * 4, &mut total)
        .expect("router weights");

    // One warm launch, then the best of a few so a scheduling hiccup does not set the number.
    let _ = run_once(
        cuda,
        &table_buffer,
        &scale_buffer,
        &index_buffer,
        &perm,
        &runs,
        &x,
        &activated,
        &partials,
        &weights,
        slots,
        warps,
    );
    let mut best = f64::INFINITY;
    for _ in 0..ROUNDS {
        let elapsed = run_once(
            cuda,
            &table_buffer,
            &scale_buffer,
            &index_buffer,
            &perm,
            &runs,
            &x,
            &activated,
            &partials,
            &weights,
            slots,
            warps,
        );
        best = best.min(elapsed);
    }
    best
}

#[test]
#[ignore = "allocates several GiB; run with --ignored --nocapture"]
fn the_expert_kernels_follow_the_distinct_expert_count() {
    let Some(cuda) = open() else {
        return;
    };
    for distinct in [TOPK * BATCH, 256_usize, 64] {
        let mut line = format!(
            "{slots} slots over {distinct:>3} experts:",
            slots = TOPK * BATCH
        );
        for warps in [4_usize, 8] {
            let elapsed = measure(&cuda, distinct, warps);
            line += &format!("  {warps:>2}w {elapsed:6.2} ms");
        }
        eprintln!("{line}");
    }
}

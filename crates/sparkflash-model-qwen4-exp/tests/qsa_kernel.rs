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

//! The indexer's block scores (q4e_qsa_scores) over pooled keys (q4e_qsa_pool), against the
//! host.
//!
//! A warp scores a block, and its sums promise the order the 128-thread block tree took when a
//! CTA scored one block for one row: channel t + 64 into t, t + 32 into t, then halving. The host
//! runs that tree on the pooled keys the device wrote, so a score that differs in any bit means
//! the order slipped. They skip - loudly - without an embedded cubin or a CUDA driver.

use half::bf16;
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_model_qwen4_exp::device::{cubin_available, embedded_cubin};

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

/// A small deterministic generator, so a failure reproduces.
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

fn upload(cuda: &Cuda, bytes: &[u8]) -> Deviceptr {
    let mut total = 0;
    let pointer = cuda.alloc(bytes.len().max(4), &mut total).expect("alloc");
    cuda.copy_htod(&pointer, bytes).expect("upload");
    pointer
}

fn upload_f32(cuda: &Cuda, values: &[f32]) -> Deviceptr {
    upload(
        cuda,
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn download_f32(cuda: &Cuda, pointer: &Deviceptr, elements: usize) -> Vec<f32> {
    cuda.synchronize().expect("synchronize");
    let mut bytes = vec![0u8; elements * 4];
    cuda.copy_dtoh(&mut bytes, pointer).expect("download");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

/// q4e_block_sum at 128 threads.
fn block_tree(mut lanes: [f32; 128]) -> f32 {
    let mut stride = 64;
    while stride > 0 {
        let (low, high) = lanes.split_at_mut(stride);
        for (thread, other) in low.iter_mut().zip(&high[..stride]) {
            *thread += *other;
        }
        stride /= 2;
    }
    lanes[0]
}

#[test]
fn the_block_scores_sum_in_the_block_trees_order() {
    let Some(cuda) = open() else {
        return;
    };
    const DIM: usize = 128;
    const HEADS: usize = 4;
    const RATIO: usize = 4;
    // Rows that straddle block ends and a score tile's 64 blocks.
    let position = 1_013;
    let rows = 300;
    let stride = (position + rows) / RATIO + 1;
    let pooled_blocks = (position + rows) / RATIO;
    let mut rng = Lcg(0x1d_e8e5);
    let keys: Vec<f32> = (0..(position + rows) * DIM).map(|_| rng.next()).collect();
    let weight: Vec<bf16> = (0..DIM)
        .map(|_| bf16::from_f32(0.25 * rng.next()))
        .collect();
    let queries: Vec<f32> = (0..rows * HEADS * DIM).map(|_| rng.next()).collect();
    let keys_on = upload_f32(&cuda, &keys);
    let weight_on = upload(
        &cuda,
        &weight
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let queries_on = upload_f32(&cuda, &queries);
    let pooled_on = upload_f32(&cuda, &vec![f32::NAN; pooled_blocks * DIM]);
    let scores_on = upload_f32(&cuda, &vec![f32::NAN; rows * stride]);
    let (_, pool) = cuda
        .load_named(embedded_cubin(), "q4e_qsa_pool")
        .expect("the cubin carries q4e_qsa_pool");
    let (_, score) = cuda
        .load_named(embedded_cubin(), "q4e_qsa_scores")
        .expect("the cubin carries q4e_qsa_scores");
    cuda.launch(
        pool,
        (pooled_blocks as u32, 1, 1),
        (128, 1, 1),
        &[
            keys_on.0,
            weight_on.0,
            pooled_on.0,
            DIM as u64,
            RATIO as u64,
            64,
            10_000_000f32.to_bits() as u64,
            1e-6f32.to_bits() as u64,
        ],
    )
    .expect("pool");
    cuda.launch(
        score,
        (pooled_blocks.div_ceil(64) as u32, rows as u32, 1),
        (256, 1, 1),
        &[
            queries_on.0,
            pooled_on.0,
            scores_on.0,
            position as u64,
            HEADS as u64,
            DIM as u64,
            RATIO as u64,
            stride as u64,
        ],
    )
    .expect("score");
    let pooled = download_f32(&cuda, &pooled_on, pooled_blocks * DIM);
    let scores = download_f32(&cuda, &scores_on, rows * stride);
    assert!(
        pooled.iter().all(|value| value.is_finite()),
        "q4e_qsa_pool left a block unwritten"
    );
    for row in 0..rows {
        let blocks = (position + row + 1) / RATIO;
        for block in 0..blocks {
            let key = &pooled[block * DIM..(block + 1) * DIM];
            let mut want = 0.0f32;
            for head in 0..HEADS {
                let query = &queries[(row * HEADS + head) * DIM..(row * HEADS + head + 1) * DIM];
                let mut lanes = [0.0f32; 128];
                for (lane, (q, k)) in lanes.iter_mut().zip(query.iter().zip(key)) {
                    *lane = q * k;
                }
                want += block_tree(lanes).max(0.0);
            }
            want /= (DIM as f32).sqrt();
            let got = scores[row * stride + block];
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "row {row} block {block}: {got} against the tree's {want}"
            );
        }
    }
}

fn download_i32(cuda: &Cuda, pointer: &Deviceptr, elements: usize) -> Vec<i32> {
    cuda.synchronize().expect("synchronize");
    let mut bytes = vec![0u8; elements * 4];
    cuda.copy_dtoh(&mut bytes, pointer).expect("download");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| i32::from_le_bytes(*word))
        .collect()
}

/// The selection the pick kernel made before the radix select: a min-heap of the first budget
/// blocks, fed the rest in order, then the kept blocks ascending and the incomplete tail.
fn heap_pick(scores: &[f32], position: usize, ratio: usize, budget: usize) -> Vec<i32> {
    let blocks = (position + 1) / ratio;
    let kept = budget.min(blocks).min(1024);
    if kept == 0 {
        return Vec::new();
    }
    let mut score: Vec<f32> = scores[..kept].to_vec();
    let mut block: Vec<usize> = (0..kept).collect();
    let worse = |score: &[f32], block: &[usize], a: usize, b: usize| {
        score[a] < score[b] || (score[a] == score[b] && block[a] > block[b])
    };
    let sift = |score: &mut [f32], block: &mut [usize], mut root: usize| loop {
        let left = root * 2 + 1;
        if left >= kept {
            break;
        }
        let mut worst = left;
        if left + 1 < kept && worse(score, block, left + 1, left) {
            worst = left + 1;
        }
        if !worse(score, block, worst, root) {
            break;
        }
        score.swap(root, worst);
        block.swap(root, worst);
        root = worst;
    };
    if blocks > kept {
        for start in (0..kept / 2).rev() {
            sift(&mut score, &mut block, start);
        }
        for (next, &candidate) in scores.iter().enumerate().take(blocks).skip(kept) {
            if score[0] < candidate || (score[0] == candidate && block[0] > next) {
                score[0] = candidate;
                block[0] = next;
                sift(&mut score, &mut block, 0);
            }
        }
    }
    block.sort_unstable();
    let mut selected: Vec<i32> = block
        .iter()
        .flat_map(|&kept| (kept * ratio..(kept + 1) * ratio).map(|token| token as i32))
        .collect();
    selected.extend((blocks * ratio..=position).map(|token| token as i32));
    selected
}

#[test]
fn the_radix_pick_keeps_what_the_heap_kept() {
    let Some(cuda) = open() else {
        return;
    };
    const RATIO: usize = 4;
    const BUDGET: usize = 512;
    const SELECT_STRIDE: usize = 2304;
    let (_, pick) = cuda
        .load_named(embedded_cubin(), "q4e_qsa_pick")
        .expect("the cubin carries q4e_qsa_pick");
    let mut rng = Lcg(0x7_0b1c);
    // Under a block, across the budget, and far past it; the scores on a coarse grid, so ties
    // at the threshold are the rule rather than the exception.
    for (position, rows, levels) in [(0, 40, 4.0), (2_000, 200, 8.0), (9_000, 64, 3.0)] {
        let stride = (position + rows) / RATIO + 1;
        let scores: Vec<f32> = (0..rows * stride)
            .map(|_| (rng.next().abs() * levels).floor() / levels)
            .collect();
        let scores_on = upload_f32(&cuda, &scores);
        let selected_on = upload(&cuda, &vec![0xFFu8; rows * SELECT_STRIDE * 4]);
        let count_on = upload(&cuda, &vec![0xFFu8; rows * 4]);
        cuda.launch(
            pick,
            (rows as u32, 1, 1),
            (256, 1, 1),
            &[
                scores_on.0,
                selected_on.0,
                count_on.0,
                position as u64,
                RATIO as u64,
                BUDGET as u64,
                stride as u64,
                SELECT_STRIDE as u64,
            ],
        )
        .expect("pick");
        let selected = download_i32(&cuda, &selected_on, rows * SELECT_STRIDE);
        let count = download_i32(&cuda, &count_on, rows);
        for row in 0..rows {
            let want = heap_pick(
                &scores[row * stride..(row + 1) * stride],
                position + row,
                RATIO,
                BUDGET,
            );
            assert_eq!(
                count[row] as usize,
                want.len(),
                "position {}: count",
                position + row
            );
            let got = &selected[row * SELECT_STRIDE..row * SELECT_STRIDE + want.len()];
            assert_eq!(got, &want[..], "position {}: selection", position + row);
        }
    }
}

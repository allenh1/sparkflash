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

//! The grouped sparse attention (q4e_attn_group) against the per-head kernel it replaces.
//!
//! The grouped kernel reads a token's selected keys and values once for four query heads of a
//! KV head, and promises each head's output bit for bit as q4e_attn computes it alone. These
//! tests hold it to that at the released shape and a narrower one, over selections from empty
//! to the most it stages. They skip - loudly - without an embedded cubin or a CUDA driver.

use half::bf16;
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_model_qwen4_exp::device::{
    ATTN_GROUP_HEADS, ATTN_GROUP_SELECTED, ATTN_GROUP_SHARED_BYTES, MAX_SELECTED_TOKENS,
    cubin_available, embedded_cubin,
};

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
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn next(&mut self) -> f32 {
        ((self.next_u64() >> 7) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
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

fn upload_bf16(cuda: &Cuda, values: &[bf16]) -> Deviceptr {
    upload(
        cuda,
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn upload_i32(cuda: &Cuda, values: &[i32]) -> Deviceptr {
    upload(
        cuda,
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn download_bits(cuda: &Cuda, pointer: &Deviceptr, elements: usize) -> Vec<u32> {
    cuda.synchronize().expect("synchronize");
    let mut bytes = vec![0u8; elements * 4];
    cuda.copy_dtoh(&mut bytes, pointer).expect("download");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

struct Shape {
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    context: usize,
}

/// Every token's output from the per-head kernel and the grouped one, bit for bit. A selection
/// past what the grouped kernel stages is left out: the host sends that configuration to
/// q4e_attn.
fn grouped_matches_per_head(shape: &Shape, counts: &[usize]) {
    let Some(cuda) = open() else {
        return;
    };
    let Shape {
        heads,
        kv_heads,
        head_dim,
        context,
    } = *shape;
    let groups = heads / kv_heads;
    let tokens = counts.len();
    let width = heads * head_dim;
    let mut rng = Lcg(0x5eed_a77e);
    // A query four times the keys' scale gives a peaked softmax, so the weights span the range.
    let q: Vec<f32> = (0..tokens * width).map(|_| 4.0 * rng.next()).collect();
    let gate: Vec<f32> = (0..tokens * width).map(|_| 3.0 * rng.next()).collect();
    let keys: Vec<bf16> = (0..context * kv_heads * head_dim)
        .map(|_| bf16::from_f32(rng.next()))
        .collect();
    let values: Vec<bf16> = (0..context * kv_heads * head_dim)
        .map(|_| bf16::from_f32(rng.next()))
        .collect();
    let mut selected = vec![0i32; tokens * MAX_SELECTED_TOKENS];
    for (token, &count) in counts.iter().enumerate() {
        for slot in 0..count {
            selected[token * MAX_SELECTED_TOKENS + slot] = (rng.next_u64() % context as u64) as i32;
        }
    }
    let count: Vec<i32> = counts.iter().map(|&count| count as i32).collect();
    let q = upload_f32(&cuda, &q);
    let gate = upload_f32(&cuda, &gate);
    let keys = upload_bf16(&cuda, &keys);
    let values = upload_bf16(&cuda, &values);
    let selected = upload_i32(&cuda, &selected);
    let count = upload_i32(&cuda, &count);
    let run = |kernel: Kernel, group: usize| {
        // Filled with NaN, so a head the kernel never writes cannot match by accident.
        let out = upload_f32(&cuda, &vec![f32::NAN; tokens * width]);
        cuda.launch(
            kernel,
            ((heads / group) as u32, tokens as u32, 1),
            (256, 1, 1),
            &[
                q.0,
                gate.0,
                keys.0,
                values.0,
                selected.0,
                count.0,
                out.0,
                heads as u64,
                kv_heads as u64,
                head_dim as u64,
                groups as u64,
                MAX_SELECTED_TOKENS as u64,
            ],
        )
        .expect("launch");
        download_bits(&cuda, &out, tokens * width)
    };
    let (_, per_head) = cuda
        .load_named(embedded_cubin(), "q4e_attn")
        .expect("the cubin carries q4e_attn");
    let reference = run(per_head, 1);
    assert!(
        reference.iter().all(|&bits| !f32::from_bits(bits).is_nan()),
        "q4e_attn left an output unwritten"
    );
    let (_, kernel) = cuda
        .load_named_shared(embedded_cubin(), "q4e_attn_group", ATTN_GROUP_SHARED_BYTES)
        .expect("the cubin carries q4e_attn_group");
    let grouped = run(kernel, ATTN_GROUP_HEADS);
    for (token, &selected) in counts.iter().enumerate() {
        if selected > ATTN_GROUP_SELECTED {
            continue;
        }
        for head in 0..heads {
            let at = (token * heads + head) * head_dim;
            let span = at..at + head_dim;
            if let Some(dim) = span
                .clone()
                .position(|index| grouped[index] != reference[index])
            {
                panic!(
                    "token {token} ({selected} selected) head {head} channel {dim}: {} against \
                     q4e_attn's {}",
                    f32::from_bits(grouped[at + dim]),
                    f32::from_bits(reference[at + dim]),
                );
            }
        }
    }
}

#[test]
fn the_grouped_attention_matches_the_per_head_kernel_at_the_released_shape() {
    grouped_matches_per_head(
        &Shape {
            heads: 24,
            kv_heads: 2,
            head_dim: 256,
            context: 9000,
        },
        &[
            0,
            1,
            33,
            700,
            2047,
            2051,
            ATTN_GROUP_SELECTED,
            MAX_SELECTED_TOKENS,
        ],
    );
}

#[test]
fn the_grouped_attention_matches_the_per_head_kernel_at_a_narrow_head() {
    grouped_matches_per_head(
        &Shape {
            heads: 8,
            kv_heads: 2,
            head_dim: 96,
            context: 500,
        },
        &[5, 0, 256, 257, 1000],
    );
}

/// Microseconds a launch of each attention kernel takes over `rows` rows of `count` selected
/// tokens at the released shape: a decode step, a speculative verify and a prefill piece.
#[test]
#[ignore = "a benchmark: cargo test --release --test attn_kernel -- --ignored --nocapture"]
fn bench_the_attention_kernels() {
    let Some(cuda) = open() else {
        return;
    };
    let (heads, kv_heads, head_dim, context) = (24usize, 2usize, 256usize, 8192usize);
    let groups = heads / kv_heads;
    let mut rng = Lcg(0xbe_7c4);
    let keys = upload_bf16(
        &cuda,
        &(0..context * kv_heads * head_dim)
            .map(|_| bf16::from_f32(rng.next()))
            .collect::<Vec<_>>(),
    );
    let values = upload_bf16(
        &cuda,
        &(0..context * kv_heads * head_dim)
            .map(|_| bf16::from_f32(rng.next()))
            .collect::<Vec<_>>(),
    );
    for (rows, count) in [
        (1usize, 2048usize),
        (4, 2048),
        (8, 2048),
        (16, 2048),
        (128, 2048),
        (128, 1024),
    ] {
        let width = heads * head_dim;
        let q = upload_f32(
            &cuda,
            &(0..rows * width).map(|_| rng.next()).collect::<Vec<_>>(),
        );
        let gate = upload_f32(
            &cuda,
            &(0..rows * width).map(|_| rng.next()).collect::<Vec<_>>(),
        );
        // Blocks of four ascending, as the pick emits them.
        let mut selected = vec![0i32; rows * MAX_SELECTED_TOKENS];
        for row in 0..rows {
            let mut blocks: Vec<usize> = (0..context / 4).collect();
            for at in 0..count / 4 {
                let other = at + (rng.next().abs() * (blocks.len() - at - 1) as f32) as usize;
                blocks.swap(at, other);
            }
            let mut kept = blocks[..count / 4].to_vec();
            kept.sort_unstable();
            for (slot, block) in kept.iter().enumerate() {
                for token in 0..4 {
                    selected[row * MAX_SELECTED_TOKENS + slot * 4 + token] =
                        (block * 4 + token) as i32;
                }
            }
        }
        let selected = upload_i32(&cuda, &selected);
        let counts = upload_i32(&cuda, &vec![count as i32; rows]);
        let out = upload_f32(&cuda, &vec![0.0; rows * width]);
        let mut line = format!("rows {rows:>3} selected {count}:");
        for (group, name, shared) in [
            (1, "q4e_attn", 0),
            (ATTN_GROUP_HEADS, "q4e_attn_group", ATTN_GROUP_SHARED_BYTES),
        ] {
            let (_, kernel) = cuda
                .load_named_shared(embedded_cubin(), name, shared)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let launch = || {
                cuda.launch(
                    kernel,
                    ((heads / group) as u32, rows as u32, 1),
                    (256, 1, 1),
                    &[
                        q.0,
                        gate.0,
                        keys.0,
                        values.0,
                        selected.0,
                        counts.0,
                        out.0,
                        heads as u64,
                        kv_heads as u64,
                        head_dim as u64,
                        groups as u64,
                        MAX_SELECTED_TOKENS as u64,
                    ],
                )
                .expect("launch")
            };
            for _ in 0..3 {
                launch();
            }
            cuda.synchronize().expect("synchronize");
            let start = std::time::Instant::now();
            let reps = 20;
            for _ in 0..reps {
                launch();
            }
            cuda.synchronize().expect("synchronize");
            let micros = start.elapsed().as_secs_f64() * 1e6 / reps as f64;
            line += &format!("  {name} {micros:7.1}");
        }
        println!("{line}");
    }
}

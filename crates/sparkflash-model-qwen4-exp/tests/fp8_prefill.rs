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

//! The Qwen FP8 grouped prefill's three kernels on the device, against a scalar oracle:
//! q4p_moe_gate_up_fp8, q4p_moe_swiglu_fp8 and q4p_moe_down_fp8.
//!
//! The plan's sorted array is a non-identity permutation in both cases: gate/up must read each
//! item's *token* row and write its *slot* row, SwiGLU maps those rows one to one, and down must
//! take each slot's token-row routing weight from Qwen's [batch][experts] probability row
//! (`route[(slot / topk) * weight_stride + slot % topk]`), not `route[slot]`.
//!
//! Case one is the blocker's minimum: two tokens, topk 1, experts swapped, different weights.
//! Case two has more slots than topk (four slots over two tokens, topk 2) and a weight stride
//! wider than topk, so the slot-indexed read lands in another token's row.
//!
//! It skips - loudly - without the prefill image or a CUDA driver.

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_model_qwen4_exp::device::{cubin_available, embedded_prefill_cubin};
use sparkflash_weights::quant;

const HIDDEN: usize = 128;
const INTERMEDIATE: usize = 128;

fn open() -> Option<Cuda> {
    if !cubin_available() || embedded_prefill_cubin().is_empty() {
        eprintln!("skipped: this build embeds no Qwen prefill image");
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

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn values(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.next()).collect()
    }
    fn code(&mut self) -> u8 {
        let code = ((self.next() + 1.0) * 128.0) as u8;
        if code & 0x7F == 0x7F { code - 1 } else { code }
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
            .flat_map(|value| value.to_le_bytes())
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

/// The device silu, rounded as the kernel's `gate / (1 + e^-gate)`.
fn silu(gate: f32) -> f32 {
    gate / (1.0 + (-gate).exp())
}

/// One grouped-prefill case: every slot's expert and weight, and the plan's stable order.
struct Case<'a> {
    topk: usize,
    experts: usize,
    tokens: usize,
    /// The expert slot `s` routes to.
    expert_of: &'a [usize],
    /// The routing weight slot `s` carries, stored in its token row's column `s % topk`.
    weight_of: &'a [f32],
}

/// Run one case through gate/up, SwiGLU and down, and return the worst relative error against
/// the scalar oracle.
fn run_case(cuda: &Cuda, case: &Case<'_>) -> f32 {
    let &Case {
        topk,
        experts,
        tokens,
        expert_of,
        weight_of,
    } = case;
    let slots = tokens * topk;
    assert_eq!(expert_of.len(), slots);
    assert_eq!(weight_of.len(), slots);
    let load = |name| {
        cuda.load_named(embedded_prefill_cubin(), name)
            .unwrap_or_else(|error| panic!("{name}: {error}"))
            .1
    };
    let gate_up = load("q4p_moe_gate_up_fp8");
    let swiglu = load("q4p_moe_swiglu_fp8");
    let down = load("q4p_moe_down_fp8");
    let mut rng = Lcg(0x4650_3847 ^ (topk as u64) << 8 ^ experts as u64);

    // Per expert: gate [intermediate][hidden] and up, then down [hidden][intermediate], each one
    // E4M3 128 x 128 tile with one BF16 scale.
    let mut weight = vec![Vec::<u8>::new(); experts * 6];
    let mut scale = vec![0.0f32; experts * 3];
    for expert in 0..experts {
        for matrix in 0..3 {
            let (rows, columns) = if matrix < 2 {
                (INTERMEDIATE, HIDDEN)
            } else {
                (HIDDEN, INTERMEDIATE)
            };
            weight[expert * 6 + 2 * matrix] = (0..rows * columns).map(|_| rng.code()).collect();
            // Exactly representable in BF16, so the host's scalar decode and the kernel's typed
            // scale read the same value.
            let value = [0.0009765625f32, 0.001953125, 0.00390625][matrix];
            weight[expert * 6 + 2 * matrix + 1] =
                half::bf16::from_f32(value).to_le_bytes().to_vec();
            scale[expert * 3 + matrix] = value;
        }
    }
    let pointers: Vec<Deviceptr> = weight.iter().map(|bytes| upload(cuda, bytes)).collect();
    let table: Vec<u8> = pointers
        .iter()
        .flat_map(|pointer| pointer.0.to_le_bytes())
        .collect();
    let table = upload(cuda, &table);

    // The plan's stable order: by expert, ties by slot.
    let mut order: Vec<usize> = (0..slots).collect();
    order.sort_by_key(|&slot| (expert_of[slot], slot));
    let sorted = upload(
        cuda,
        &order
            .iter()
            .flat_map(|slot| (*slot as i32).to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let matrices = rank_items(&order, expert_of);
    let mut item_bytes = Vec::with_capacity(matrices.len() * 16);
    for (n, (expert, first, count)) in matrices.iter().enumerate() {
        for value in [*expert, *first, *count, n as i32] {
            item_bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let items = upload(cuda, &item_bytes);

    let mixed = rng.values(tokens * HIDDEN);
    let input = upload_f32(cuda, &mixed);
    let gate_up_buffer = upload_f32(cuda, &vec![0.0; slots * 2 * INTERMEDIATE]);
    let activated = upload_f32(cuda, &vec![0.0; slots * INTERMEDIATE]);
    let partials = upload_f32(cuda, &vec![0.0; slots * HIDDEN]);
    // Qwen's routing probabilities: [batch][experts], the top-k weights in the first TOPK
    // columns of each token's row.
    let mut probabilities = vec![0.0f32; tokens * experts];
    for slot in 0..slots {
        probabilities[(slot / topk) * experts + slot % topk] = weight_of[slot];
    }
    let route = upload_f32(cuda, &probabilities);

    let launch = |kernel, grid: (u32, u32, u32), args: &[u64]| {
        cuda.launch(kernel, grid, (256, 1, 1), args)
            .unwrap_or_else(|error| panic!("launch: {error}"));
    };
    for (matrix, at) in [(0u64, 0usize), (2, INTERMEDIATE)] {
        launch(
            gate_up,
            ((INTERMEDIATE / 128) as u32, 1, matrices.len() as u32),
            &[
                table.0,
                items.0,
                input.0,
                gate_up_buffer.offset(at * 4).0,
                INTERMEDIATE as u64,
                HIDDEN as u64,
                sorted.0,
                topk as u64,
                matrix,
            ],
        );
    }
    launch(
        swiglu,
        ((INTERMEDIATE.div_ceil(256)) as u32, slots as u32, 1),
        &[
            gate_up_buffer.0,
            activated.0,
            slots as u64,
            INTERMEDIATE as u64,
        ],
    );
    launch(
        down,
        ((HIDDEN / 128) as u32, 1, matrices.len() as u32),
        &[
            table.0,
            items.0,
            activated.0,
            route.0,
            partials.0,
            HIDDEN as u64,
            INTERMEDIATE as u64,
            sorted.0,
            topk as u64,
            experts as u64,
        ],
    );
    let got = download_f32(cuda, &partials, slots * HIDDEN);

    // The scalar twin: slot -> (token, expert), its gate/up, SwiGLU, and down times the token's
    // routing weight for that slot.
    let mut want = vec![0.0f32; slots * HIDDEN];
    for slot in 0..slots {
        let token = slot / topk;
        let expert = expert_of[slot];
        let x = &mixed[token * HIDDEN..(token + 1) * HIDDEN];
        let gate_codes = &weight[expert * 6];
        let up_codes = &weight[expert * 6 + 2];
        let down_codes = &weight[expert * 6 + 4];
        let (gate_scale, up_scale, down_scale) = (
            scale[expert * 3],
            scale[expert * 3 + 1],
            scale[expert * 3 + 2],
        );
        let mut act = vec![0.0f32; INTERMEDIATE];
        for column in 0..INTERMEDIATE {
            let mut gate = 0.0f32;
            let mut up = 0.0f32;
            for k in 0..HIDDEN {
                gate += quant::f8e4m3_to_f32(gate_codes[column * HIDDEN + k]) * x[k];
                up += quant::f8e4m3_to_f32(up_codes[column * HIDDEN + k]) * x[k];
            }
            act[column] = silu(gate * gate_scale) * (up * up_scale);
        }
        for column in 0..HIDDEN {
            let mut sum = 0.0f32;
            for j in 0..INTERMEDIATE {
                sum += quant::f8e4m3_to_f32(down_codes[column * INTERMEDIATE + j]) * act[j];
            }
            want[slot * HIDDEN + column] = sum * down_scale * weight_of[slot];
        }
    }
    let scale = want.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    let worst = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "fp8 grouped prefill (topk {topk}, {experts} experts, {tokens} tokens): worst {worst:.3e} (scale {scale:.3})"
    );
    assert!(
        got.iter().all(|value| value.is_finite()),
        "non-finite routed partial"
    );
    worst / scale.max(1e-3)
}

/// The items the plan cuts from the stable order: one run a whole expert here, since every case
/// has at most two slots an expert.
fn rank_items(order: &[usize], expert_of: &[usize]) -> Vec<(i32, i32, i32)> {
    let mut items = Vec::new();
    let mut rank = 0;
    while rank < order.len() {
        let expert = expert_of[order[rank]];
        let mut end = rank + 1;
        while end < order.len() && expert_of[order[end]] == expert {
            end += 1;
        }
        items.push((expert as i32, rank as i32, (end - rank) as i32));
        rank = end;
    }
    items
}

#[test]
fn the_fp8_grouped_prefill_follows_the_plan_permutation_and_weight_stride() {
    let Some(cuda) = open() else { return };
    // Two tokens, topk 1, experts swapped: token 0's slot 0 routes to expert 1, token 1's slot 1
    // to expert 0, so the plan's sorted array is [1, 0]. The two tokens carry different weights.
    let case = Case {
        topk: 1,
        experts: 2,
        tokens: 2,
        expert_of: &[1, 0],
        weight_of: &[0.25, 0.75],
    };
    let relative = run_case(&cuda, &case);
    assert!(
        relative < 5e-3,
        "two tokens, experts swapped: relative {relative:.3e}"
    );
}

/// The same plan machinery on a whole chunk's worth of rows: 40 tokens x topk 2 over four
/// experts, so every item holds ten rows and the tile's 16-row masks run. It also proves the
/// slot row is written for every token of a multi-row item, not only the first.
#[test]
fn the_fp8_grouped_prefill_covers_multi_row_items() {
    let Some(cuda) = open() else { return };
    let (tokens, topk, experts) = (40usize, 2usize, 4usize);
    let slots = tokens * topk;
    let expert_of: Vec<usize> = (0..slots)
        .map(|slot| 2 * (slot / 4) % experts + slot % 2)
        .collect();
    // Every token routes to two distinct experts, and every slot carries its own weight.
    let weight_of: Vec<f32> = (0..slots).map(|slot| 0.05 + 0.002 * slot as f32).collect();
    let case = Case {
        topk,
        experts,
        tokens,
        expert_of: &expert_of,
        weight_of: &weight_of,
    };
    let relative = run_case(&cuda, &case);
    assert!(
        relative < 5e-3,
        "forty tokens of ten-row items: relative {relative:.3e}"
    );
}

#[test]
fn the_fp8_grouped_prefill_weights_a_slot_from_its_token_row() {
    let Some(cuda) = open() else { return };
    // More slots than topk: two tokens x topk 2 over four experts, a weight stride of four.
    // Token 0 routes to experts 1 and 0 (slots 0 and 1), token 1 to experts 3 and 2 (slots 2 and
    // 3); the stable plan order is [1, 0, 3, 2].
    let case = Case {
        topk: 2,
        experts: 4,
        tokens: 2,
        expert_of: &[1, 0, 3, 2],
        weight_of: &[0.1, 0.2, 0.3, 0.4],
    };
    let relative = run_case(&cuda, &case);
    assert!(
        relative < 5e-3,
        "four slots over two tokens: relative {relative:.3e}"
    );
}

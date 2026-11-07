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

//! The routed-expert block order on its own: the shared plan's (`moe_plan`) permutation and
//! the `(begin, end)` bounds of the run each rank belongs to.
//!
//! The expert kernels launch one block per run and let the rest exit, so a wrong bound is either
//! a slot nobody computes (a missing activation) or a slot computed twice from a stale staging
//! row. Both are silent numeric errors two hundred lines away from here, which is why the bounds
//! get their own seam.
//!
//! The tests skip - loudly - without an embedded cubin or a CUDA driver, like the parity suite.

use sparkflash_kernels::cuda_driver::Cuda;
use sparkflash_kernels::images::moe_plan_cubin;
use sparkflash_model_qwen4_exp::device::{cubin_available, embedded_moe_cubin};

fn open() -> Option<Cuda> {
    if !cubin_available() || moe_plan_cubin().is_empty() {
        eprintln!("skipped: this build embeds no Qwen4-Exp or plan cubin");
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

fn as_i32(bytes: &[u8]) -> Vec<i32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Run the permutation kernel and return `(perm, run_bounds)` as the device wrote them.
fn permute(cuda: &Cuda, indices: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let mut total = 0_usize;
    let slots = indices.len();
    let index_buffer = cuda.alloc(slots * 4, &mut total).expect("indices");
    cuda.copy_htod(&index_buffer, &i32_bytes(indices))
        .expect("upload indices");
    let perm = cuda.alloc(slots * 4, &mut total).expect("perm");
    let bounds = cuda.alloc(slots * 2 * 4, &mut total).expect("run bounds");
    let (_, kernel) = cuda
        .load_named(moe_plan_cubin(), "moe_plan")
        .expect("the plan cubin carries moe_plan");
    let arguments = [
        index_buffer.0,
        perm.0,
        0,
        0,
        bounds.0,
        slots as u64,
        // The test's ids are expert ids, so every value below this bound is valid and the
        // policy for one outside it (MOE_PLAN_INVALID_FIRST) is what keeps a -1 slot in
        // its own group ahead of expert 0.
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
        // zero_stride: this harness keeps no expert-row buffer, so a slot the plan leaves out has
        // no row of its own to zero. Named here rather than omitted: the launch reads one cell per
        // parameter, and an argument list that stops early is a read past the end of this array.
        0,
    ];
    cuda.launch(kernel, (1, 1, 1), (256, 1, 1), &arguments)
        .expect("plan launch");
    cuda.synchronize().expect("plan synchronize");
    cuda.synchronize().expect("permute synchronize");
    let mut perm_host = vec![0_u8; slots * 4];
    cuda.copy_dtoh(&mut perm_host, &perm).expect("perm back");
    let mut bounds_host = vec![0_u8; slots * 2 * 4];
    cuda.copy_dtoh(&mut bounds_host, &bounds)
        .expect("run bounds back");
    (as_i32(&perm_host), as_i32(&bounds_host))
}

/// The permutation the kernel should produce: rank by expert id, ties in slot order.
fn expected_perm(indices: &[i32]) -> Vec<i32> {
    let mut order: Vec<i32> = (0..indices.len() as i32).collect();
    order.sort_by_key(|index| (indices[*index as usize], *index));
    order
}

/// The run bounds the kernel should produce, from the host's permutation.
fn expected_bounds(indices: &[i32], perm: &[i32]) -> Vec<i32> {
    let mut bounds = vec![0_i32; perm.len() * 2];
    let mut rank = 0_usize;
    while rank < perm.len() {
        let expert = indices[perm[rank] as usize];
        let begin = rank;
        while rank + 1 < perm.len() && indices[perm[rank + 1] as usize] == expert {
            rank += 1;
        }
        for r in begin..=rank {
            bounds[2 * r] = begin as i32;
            bounds[2 * r + 1] = rank as i32;
        }
        rank += 1;
    }
    bounds
}

/// The token routed-expert GEMV at the FP8 release's expert width, above the old static caps.
///
/// The staging used to be a fixed `Q4E_MAX_ROW`-sized array, and the kernel returned early for
/// a hidden width past 2,560 or an intermediate past 1,024; the FP8 release's 5,120 x 1,536
/// expert therefore could not take it. The staging is now dynamic shared memory the host sizes,
/// so this launches the real kernel at that geometry with a single dropped expert (whose rows
/// are written as zero) and asserts the launch reserved the host-computed bytes and produced a
/// finite, zero activation.
#[test]
fn the_expert_gemv_serves_the_release_width_above_the_old_caps() {
    let Some(cuda) = open() else {
        return;
    };
    const HIDDEN: usize = 5120;
    const INTER: usize = 1536;
    let mut total = 0_usize;
    let indices = [-1_i32];
    let (perm, runs) = {
        // The plan's outputs are this test's inputs; reuse the helper's launch but keep its
        // allocations alive by asking for the values, then re-uploading them here.
        let (perm_values, run_values) = permute(&cuda, &indices);
        (
            cuda.alloc(perm_values.len() * 4, &mut total).expect("perm"),
            cuda.alloc(run_values.len() * 4, &mut total).expect("runs"),
        )
    };
    let (perm_host, run_host) = permute(&cuda, &indices);
    cuda.copy_htod(&perm, &i32_bytes(&perm_host))
        .expect("perm upload");
    cuda.copy_htod(&runs, &i32_bytes(&run_host))
        .expect("runs upload");
    let indices_d = cuda.alloc(4, &mut total).expect("indices");
    cuda.copy_htod(&indices_d, &i32_bytes(&indices))
        .expect("indices upload");
    let table = cuda.alloc(6 * 8, &mut total).expect("table");
    let x = cuda.alloc(HIDDEN * 4, &mut total).expect("x");
    let activated = cuda.alloc(INTER * 4, &mut total).expect("activated");
    cuda.copy_htod(&activated, &vec![1_u8; INTER * 4])
        .expect("activated fill");
    let shared = (2 * HIDDEN.div_ceil(32) * 33 * std::mem::size_of::<f32>()) as u32;
    let (_, gate_up) = cuda
        .load_named_shared(embedded_moe_cubin(), "q4e_expert_gate_up", shared)
        .expect("q4e_expert_gate_up");
    assert_eq!(gate_up.shared_bytes(), shared);
    let scales = cuda.alloc(3 * 4, &mut total).expect("scales");
    cuda.launch(
        gate_up,
        (INTER.div_ceil(16) as u32, 1, 1),
        (256, 1, 1),
        &[
            table.0,
            scales.0,
            indices_d.0,
            perm.0,
            runs.0,
            x.0,
            activated.0,
            1,
            HIDDEN as u64,
            INTER as u64,
        ],
    )
    .expect("gate up launch");
    cuda.synchronize().expect("synchronize");
    let mut host = vec![0_u8; INTER * 4];
    cuda.copy_dtoh(&mut host, &activated)
        .expect("activated back");
    assert!(
        host.as_chunks::<4>()
            .0
            .iter()
            .all(|value| value == &[0, 0, 0, 0]),
        "a dropped expert's activation row is not zero"
    );
}

#[test]
fn the_run_bounds_cover_every_slot_exactly_once() {
    let Some(cuda) = open() else {
        return;
    };
    // Ten slots per token over four tokens, with experts repeating the way a trained router
    // makes them repeat: a handful of hot experts carry most of the chunk.
    //
    // Expert 0 is in the fixture on purpose. The reserved bucket of MOE_PLAN_INVALID_FIRST sits
    // ahead of expert 0, so a plan that put an invalid id in expert 0's bucket instead would merge
    // the -1 slots below into expert 0's run; with no 0 slot in the fixture that merge lands in a
    // bucket no valid slot occupies and the permutation still looks right.
    let slots_per_token = 10_usize;
    let batch = 4_usize;
    let hot = [0_i32, 3, 17, 3, 5, 5, 3, 61, 3, 5];
    let indices: Vec<i32> = (0..batch)
        .flat_map(|token| hot.iter().map(move |expert| expert + token as i32 % 2))
        .collect();
    assert_eq!(indices.len(), slots_per_token * batch);
    // A dropped slot (probability zero) is a run of its own only among the other drops.
    let mut indices = indices;
    indices[7] = -1;
    indices[26] = -1;

    let (perm, bounds) = permute(&cuda, &indices);
    let expected = expected_perm(&indices);
    assert_eq!(perm, expected, "the permutation is not the expert order");
    assert_eq!(bounds, expected_bounds(&indices, &expected));

    // Every rank is claimed by exactly one leader, and every leader's run stays in range.
    let mut claimed = vec![0_usize; indices.len()];
    for r in 0..indices.len() {
        let (begin, end) = (bounds[2 * r] as usize, bounds[2 * r + 1] as usize);
        assert!(begin <= r && r <= end, "rank {r} lies outside its run");
        if r == begin {
            for member in begin..=end {
                claimed[member] += 1;
                assert_eq!(
                    indices[perm[member] as usize], indices[perm[r] as usize],
                    "rank {member} shares a run with a different expert"
                );
            }
        }
    }
    assert!(
        claimed.iter().all(|count| *count == 1),
        "the runs do not partition the slots: {claimed:?}"
    );
}

/// One router launch over `tokens` rows: returns the selected expert ids.
fn route(cuda: &Cuda, logits: &[f32], tokens: usize, experts: usize, topk: usize) -> Vec<i32> {
    let mut total = 0_usize;
    let logit_buffer = cuda.alloc(logits.len() * 4, &mut total).expect("logits");
    let bytes: Vec<u8> = logits
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    cuda.copy_htod(&logit_buffer, &bytes)
        .expect("upload logits");
    let probabilities = cuda
        .alloc(logits.len() * 4, &mut total)
        .expect("probabilities");
    let indices = cuda.alloc(tokens * topk * 4, &mut total).expect("indices");
    let (_, kernel) = cuda
        .load_named(embedded_moe_cubin(), "q4e_router")
        .expect("the cubin carries q4e_router");
    cuda.launch(
        kernel,
        (tokens as u32, 1, 1),
        (256, 1, 1),
        &[
            logit_buffer.0,
            probabilities.0,
            indices.0,
            experts as u64,
            topk as u64,
        ],
    )
    .expect("router launch");
    cuda.synchronize().expect("router synchronize");
    let mut host = vec![0_u8; tokens * topk * 4];
    cuda.copy_dtoh(&mut host, &indices).expect("indices back");
    as_i32(&host)
}

/// The chunk the released router actually produces: a skewed popularity over 512 experts, a
/// different per-token preference on top, and top-10 out of 512. A run longer than one means the
/// expert kernels have weight rows to share.
fn chunk_indices(cuda: &Cuda) -> Vec<i32> {
    const EXPERTS: usize = 512;
    const TOPK: usize = 10;
    const TOKENS: usize = 128;
    let mut logits = vec![0.0_f32; TOKENS * EXPERTS];
    for token in 0..TOKENS {
        for expert in 0..EXPERTS {
            // Popularity is static; the per-token term reorders the head of the list so tokens
            // do not all pick the same ten experts, which would be a dedup factor nothing like
            // the model's.
            let popular = ((expert * 2654435761) % 977) as f32 * 0.002;
            let preference = (((token * 131 + expert * 17) % 251) as f32) * 0.03;
            logits[token * EXPERTS + expert] = popular + preference;
        }
    }
    route(cuda, &logits, TOKENS, EXPERTS, TOPK)
}

#[test]
fn a_chunk_of_real_routing_shares_experts_between_slots() {
    let Some(cuda) = open() else {
        return;
    };
    let indices = chunk_indices(&cuda);
    let (perm, bounds) = permute(&cuda, &indices);
    assert_eq!(perm, expected_perm(&indices));
    assert_eq!(bounds, expected_bounds(&indices, &expected_perm(&indices)));

    let distinct: std::collections::BTreeSet<i32> = indices.iter().copied().collect();
    let leaders = (0..indices.len())
        .filter(|rank| bounds[2 * *rank] as usize == *rank)
        .count();
    assert_eq!(leaders, distinct.len(), "one leader per distinct expert");
    assert!(
        leaders * 2 < indices.len(),
        "the chunk's {} slots landed on {} experts, so almost no weight row is shared",
        indices.len(),
        leaders
    );
    eprintln!(
        "chunk routing: {} slots over {} experts, {:.2} slots per run",
        indices.len(),
        leaders,
        indices.len() as f64 / leaders as f64
    );
}

#[test]
fn a_single_expert_for_the_whole_chunk_is_one_run() {
    let Some(cuda) = open() else {
        return;
    };
    let indices = vec![9_i32; 12];
    let (perm, bounds) = permute(&cuda, &indices);
    assert_eq!(perm, (0..12).collect::<Vec<i32>>());
    for r in 0..12 {
        assert_eq!((bounds[2 * r], bounds[2 * r + 1]), (0, 11));
    }
}

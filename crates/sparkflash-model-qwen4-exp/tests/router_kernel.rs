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

//! The router kernel on its own: softmax, top-k, and the indices it writes.
//!
//! These launch one kernel with chosen arguments against the embedded cubin, so the selection
//! logic is exercised at the released configuration's shape (512 experts, top-10) in a second
//! rather than behind a 77.8 GiB upload. A wrong index here is a wrong index in the MoE path, and
//! it used to surface as an index-out-of-bounds panic two hundred lines away.
//!
//! The tests skip - loudly - without an embedded cubin or a CUDA driver, like the parity suite.

use sparkflash_kernels::cuda_driver::Cuda;
use sparkflash_model_qwen4_exp::device::{cubin_available, embedded_cubin};

const MAX_TOPK: usize = 64;

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

/// A softmax the way the kernel computes it, for the expected selection.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials: Vec<f32> = logits.iter().map(|value| (value - max).exp()).collect();
    let sum: f32 = exponentials.iter().sum();
    exponentials.iter().map(|value| value / sum).collect()
}

/// The expected top-k: highest probability first, no repeats.
fn top_k(probabilities: &[f32], k: usize) -> Vec<usize> {
    let mut ranked: Vec<usize> = (0..probabilities.len()).collect();
    ranked.sort_by(|left, right| {
        probabilities[*right]
            .partial_cmp(&probabilities[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(left.cmp(right))
    });
    ranked.truncate(k);
    ranked
}

/// One router launch: returns the selected indices and their normalised weights.
fn route(cuda: &Cuda, logits: &[f32], topk: usize) -> (Vec<i32>, Vec<f32>, Vec<f32>) {
    let mut total = 0_usize;
    let device = cuda.alloc(logits.len() * 4, &mut total).expect("logits");
    let bytes: Vec<u8> = logits
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    cuda.copy_htod(&device, &bytes).expect("upload logits");
    let index_bytes = 0x7F_u8;
    let indices = cuda.alloc(topk * 4, &mut total).expect("indices");
    cuda.copy_htod(&indices, &vec![index_bytes; topk * 4])
        .expect("poison indices");
    let (_, kernel) = cuda
        .load_named(embedded_cubin(), "q4e_router")
        .expect("the cubin carries q4e_router");
    cuda.launch(
        kernel,
        (1, 1, 1),
        (32, 1, 1),
        &[
            device.0,
            device.0,
            indices.0,
            logits.len() as u64,
            topk as u64,
        ],
    )
    .expect("router launch");
    cuda.synchronize().expect("router synchronize");
    let mut index_host = vec![0_u8; topk * 4];
    cuda.copy_dtoh(&mut index_host, &indices)
        .expect("indices back");
    let mut weight_host = vec![0_u8; topk * 4];
    cuda.copy_dtoh(&mut weight_host, &device)
        .expect("weights back");
    let selected: Vec<i32> = index_host
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let weights: Vec<f32> = weight_host
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let mut all = vec![0_u8; logits.len() * 4];
    cuda.copy_dtoh(&mut all, &device)
        .expect("probabilities back");
    let probabilities: Vec<f32> = all
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    (selected, weights, probabilities)
}

#[test]
fn the_router_selects_ten_distinct_experts_at_the_released_shape() {
    let Some(cuda) = open() else {
        return;
    };
    // The released shape: 512 experts, top-10. The spread is what a trained router produces, and
    // the asymmetry keeps the argmax unambiguous.
    let experts = 512_usize;
    let logits: Vec<f32> = (0..experts)
        .map(|expert| -1.5 - (expert as f32) * 0.0086 + ((expert * 37) % 11) as f32 * 0.05)
        .collect();
    let topk = 10_usize;
    assert!(topk <= MAX_TOPK);
    let (selected, weights, probabilities) = route(&cuda, &logits, topk);

    let expected = top_k(&softmax(&logits), topk);
    let actual: Vec<usize> = selected.iter().map(|index| *index as usize).collect();
    assert_eq!(
        actual, expected,
        "the kernel selected {actual:?}, the softmax ranks {expected:?}"
    );
    let unique: std::collections::BTreeSet<i32> = selected.iter().copied().collect();
    assert_eq!(
        unique.len(),
        topk,
        "the router repeated an expert: {selected:?}"
    );

    // The kernel reuses the softmax array as its output: entries past the selection keep their
    // probability, and entries before it hold the normalised weight. So the two halves are checked
    // separately rather than the buffer as a whole.
    let tail: f32 = probabilities[topk..].iter().sum();
    assert!(
        tail > 0.0 && tail < 1.0,
        "the softmax tail is not a probability mass: {tail}"
    );
    let weight_total: f32 = weights.iter().sum();
    assert!(
        (weight_total - 1.0).abs() < 1e-4,
        "the selected weights do not sum to one: {weight_total}"
    );
}

#[test]
fn a_short_selection_does_not_repeat_an_expert() {
    let Some(cuda) = open() else {
        return;
    };
    let logits: Vec<f32> = (0..512)
        .map(|expert| (expert as f32 * 0.011).sin())
        .collect();
    let (selected, weights, _) = route(&cuda, &logits, 2);
    let unique: std::collections::BTreeSet<i32> = selected.iter().copied().collect();
    assert_eq!(
        unique.len(),
        2,
        "the router repeated an expert: {selected:?}"
    );
    assert_eq!(
        selected,
        top_k(&softmax(&logits), 2)
            .iter()
            .map(|i| *i as i32)
            .collect::<Vec<_>>()
    );
    assert!(
        weights
            .iter()
            .all(|weight| weight.is_finite() && *weight > 0.0)
    );
}

/// The exact host sequence the model uses around the router, to catch an ordering bug.
#[test]
fn the_selection_survives_the_accumulator_reset_that_follows_it() {
    let Some(cuda) = open() else {
        return;
    };
    let experts = 512_usize;
    let topk = 10_usize;
    let logits: Vec<f32> = (0..experts)
        .map(|expert| -1.5 - (expert as f32) * 0.0086 + ((expert * 37) % 11) as f32 * 0.05)
        .collect();
    let mut total = 0_usize;
    let device = cuda.alloc(logits.len() * 4, &mut total).expect("logits");
    let bytes: Vec<u8> = logits
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    cuda.copy_htod(&device, &bytes).expect("upload logits");
    let indices = cuda.alloc(topk * 4, &mut total).expect("indices");
    let accumulator = cuda.alloc(10240, &mut total).expect("accumulator");
    let (_, kernel) = cuda
        .load_named(embedded_cubin(), "q4e_router")
        .expect("router");
    cuda.launch(
        kernel,
        (1, 1, 1),
        (32, 1, 1),
        &[device.0, device.0, indices.0, experts as u64, topk as u64],
    )
    .expect("router launch");
    // The probe the model runs here: the whole probabilities row.
    let mut row = vec![0_u8; experts * 4];
    cuda.copy_dtoh(&mut row, &device)
        .expect("probabilities row");
    let probabilities: Vec<f32> = row
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    eprintln!("row: {:?}", &probabilities[..6]);
    // The accumulator reset the model issues immediately afterwards.
    let zeros = vec![0_u8; 10240];
    cuda.copy_htod(&accumulator, &zeros)
        .expect("accumulator reset");
    // The two small read-backs the model then makes.
    let mut weights_host = vec![0_u8; topk * 4];
    cuda.copy_dtoh(&mut weights_host, &device).expect("weights");
    let mut indices_host = vec![0_u8; topk * 4];
    cuda.copy_dtoh(&mut indices_host, &indices)
        .expect("indices");
    let weights: Vec<f32> = weights_host
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let selected: Vec<i32> = indices_host
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    eprintln!("weights: {weights:?}");
    eprintln!("indices: {selected:?}");
    assert!(
        selected
            .iter()
            .all(|index| *index >= 0 && (*index as usize) < experts)
    );
    assert!(
        weights
            .iter()
            .all(|weight| weight.is_finite() && *weight >= 0.0)
    );
}

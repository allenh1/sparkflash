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

//! The absorbed MLA's logits above the old sixteen-head cap: q4e_glm_mla_scores at thirty-two
//! heads (two blockIdx.z tiles) against the host dot product. The decode engine launches with
//! heads.div_ceil(16) as the head-tile axis, so this is the shape a 32-head rank now takes.

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_model_qwen4_exp::device::cubin_available;

const HEADS: usize = 32;
const LATENT: usize = 64;
const NOPE: usize = 64;
const SELECT: usize = 64;
const MAX_SELECTED: usize = 2304;
const SCORE_SLOTS: usize = 64;

fn open() -> Option<Cuda> {
    if !cubin_available() || sparkflash_kernels::images::glm_decode_cubin().is_empty() {
        eprintln!("skipped: this build embeds no GLM decode image");
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
}

fn upload(cuda: &Cuda, values: &[f32], total: &mut usize) -> Deviceptr {
    let pointer = cuda.alloc((values.len() * 4).max(4), total).expect("alloc");
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    cuda.copy_htod(&pointer, &bytes).expect("upload");
    pointer
}

#[test]
fn the_absorbed_mla_scores_thirty_two_heads() {
    let Some(cuda) = open() else {
        return;
    };
    let mut total = 0usize;
    let mut rng = Lcg(0x32_0000_0040);
    let query = rng.values(HEADS * LATENT);
    let cache = rng.values(LATENT);
    let query_d = upload(&cuda, &query, &mut total);
    let cache_d = upload(&cuda, &cache, &mut total);
    let selected_d = {
        let pointer = cuda.alloc(4, &mut total).expect("selected");
        cuda.copy_htod(&pointer, &0_i32.to_le_bytes())
            .expect("selected upload");
        pointer
    };
    let count_d = {
        let pointer = cuda.alloc(4, &mut total).expect("count");
        cuda.copy_htod(&pointer, &1_i32.to_le_bytes())
            .expect("count upload");
        pointer
    };
    let logits_d = upload(&cuda, &vec![0.0; HEADS * MAX_SELECTED], &mut total);
    let (_, kernel) = cuda
        .load_named(
            sparkflash_kernels::images::glm_decode_cubin(),
            "q4e_glm_mla_scores",
        )
        .expect("q4e_glm_mla_scores");
    cuda.launch(
        kernel,
        (
            (MAX_SELECTED / SCORE_SLOTS) as u32,
            1,
            HEADS.div_ceil(16) as u32,
        ),
        (256, 1, 1),
        &[
            query_d.0,
            cache_d.0,
            0,
            0,
            selected_d.0,
            count_d.0,
            logits_d.0,
            0,
            HEADS as u64,
            NOPE as u64,
            LATENT as u64,
            SELECT as u64,
            0,
        ],
    )
    .expect("mla scores");
    cuda.synchronize().expect("synchronize");
    let mut bytes = vec![0u8; HEADS * MAX_SELECTED * 4];
    cuda.copy_dtoh(&mut bytes, &logits_d).expect("download");
    let logits: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    let scale = (NOPE as f32).sqrt();
    for head in 0..HEADS {
        let want: f32 = (0..LATENT)
            .map(|c| query[head * LATENT + c] * cache[c])
            .sum::<f32>()
            / scale;
        let got = logits[head * MAX_SELECTED];
        let relative = (want - got).abs() / want.abs().max(1e-3);
        assert!(
            relative < 1e-4,
            "head {head}: want {want}, got {got} (relative {relative})"
        );
    }
}

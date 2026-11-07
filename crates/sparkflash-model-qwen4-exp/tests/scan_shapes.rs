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

//! The templated GDN scan's non-released instantiation on a device: the (64, 64) entry against
//! the host delta rule. The released configuration is (128, 128); this pair exercises the
//! template axis and the split cap (V_DIM 64 allows at most eight splits) without a checkpoint.

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_model_qwen4_exp::device::{
    cubin_available, embedded_scan_cubin, gdn_scan_shape_for,
};

/// The launch bound of each compiled scan entry: q4p_gdn_scan_64_64 is 256 threads,
/// q4p_gdn_scan / q4p_gdn_scan_128_256 / the KDA forms 512.
fn scan_thread_bound(value_dim: usize) -> u32 {
    if value_dim == 64 { 256 } else { 512 }
}

/// Every GDN/KDA host shape the dispatch can produce fits its entry's launch bound: a block past
/// it is a launch failure, and a block that is not a multiple of a warp is not a block the body
/// accepts. The split count times the block must still cover every value column.
#[test]
fn every_scan_shape_fits_its_launch_bound_and_warp() {
    for value_dim in [64usize, 128, 256] {
        for heads in [1usize, 2, 4, 8, 16, 32, 47, 48, 64, 96, 128] {
            let (grid, threads) = gdn_scan_shape_for(heads, value_dim);
            let (_, splits, launches) = (grid.0, grid.1, grid.2);
            assert_eq!(
                grid.0 as usize, heads,
                "one block x a head, {heads} x {value_dim}"
            );
            assert_eq!(launches, 1);
            assert!(
                threads <= scan_thread_bound(value_dim),
                "{heads} heads x {value_dim}: {threads} threads over the bound {}",
                scan_thread_bound(value_dim)
            );
            assert!(
                threads > 0 && threads % 32 == 0,
                "{heads} x {value_dim}: {threads} threads"
            );
            assert_eq!(
                splits * threads,
                4 * value_dim as u32,
                "{heads} x {value_dim}: {splits} splits x {threads} threads do not cover the head"
            );
            assert!(splits.is_power_of_two());
        }
    }
}

const HEADS: usize = 2;
const KEY_HEADS: usize = 2;
const KEY: usize = 64;
const VALUE: usize = 64;
const TOKENS: usize = 3;
const HALF: usize = KEY + KEY / 8;
const ROW: usize = 2 * HALF;

fn open() -> Option<Cuda> {
    if !cubin_available() || embedded_scan_cubin().is_empty() {
        eprintln!("skipped: this build embeds no Qwen scan image");
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

fn upload(cuda: &Cuda, bytes: &[u8], total: &mut usize) -> Deviceptr {
    let pointer = cuda.alloc(bytes.len().max(4), total).expect("alloc");
    cuda.copy_htod(&pointer, bytes).expect("upload");
    pointer
}

fn upload_f32(cuda: &Cuda, values: &[f32], total: &mut usize) -> Deviceptr {
    upload(
        cuda,
        &values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
        total,
    )
}

fn upload_bf16(cuda: &Cuda, values: &[f32], total: &mut usize) -> Deviceptr {
    upload(
        cuda,
        &values
            .iter()
            .flat_map(|value| half::bf16::from_f32(*value).to_le_bytes())
            .collect::<Vec<_>>(),
        total,
    )
}

fn download(cuda: &Cuda, pointer: &Deviceptr, count: usize) -> Vec<f32> {
    cuda.synchronize().expect("synchronize");
    let mut bytes = vec![0u8; count * 4];
    cuda.copy_dtoh(&mut bytes, pointer).expect("download");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

fn assert_close(what: &str, host: &[f32], device: &[f32]) {
    let scale = host
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max)
        .max(1e-3);
    let worst = host
        .iter()
        .zip(device)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let relative = worst / scale;
    eprintln!("{what}: worst {worst:.3e} (scale {scale:.3}), relative {relative:.3e}");
    assert!(relative < 1e-3, "{what}: relative {relative:.3e}");
}

#[test]
fn the_64_64_gdn_scan_matches_the_host_delta_rule() {
    let Some(cuda) = open() else {
        return;
    };
    let mut total = 0usize;
    let key_width = KEY_HEADS * KEY;
    let conv_width = key_width * 2 + HEADS * VALUE;
    let mut rng = Lcg(0x5eed_0064_0064);
    let mixed = rng.values(TOKENS * conv_width);
    let b = rng.values(TOKENS * HEADS);
    let a = rng.values(TOKENS * HEADS);
    let a_log = rng
        .values(HEADS)
        .iter()
        .map(|x| 0.1 + 0.2 * x.abs())
        .collect::<Vec<_>>();
    let dt_bias = rng
        .values(HEADS)
        .iter()
        .map(|x| 0.1 * x)
        .collect::<Vec<_>>();
    let qk = upload_f32(&cuda, &[0.0; TOKENS * KEY_HEADS * ROW], &mut total);
    let decay = upload_f32(&cuda, &[0.0; TOKENS * HEADS * 2], &mut total);
    let mixed_d = upload_f32(&cuda, &mixed, &mut total);
    let a_d = upload_f32(&cuda, &a, &mut total);
    let b_d = upload_f32(&cuda, &b, &mut total);
    let a_log_d = upload_bf16(&cuda, &a_log, &mut total);
    let dt_d = upload_bf16(&cuda, &dt_bias, &mut total);
    let state = upload_f32(&cuda, &[0.0; HEADS * KEY * VALUE], &mut total);
    let raw = upload_f32(&cuda, &[0.0; TOKENS * HEADS * VALUE], &mut total);
    let (_, prep) = cuda
        .load_named(embedded_scan_cubin(), "q4p_gdn_prep")
        .expect("q4p_gdn_prep");
    let (_, scan) = cuda
        .load_named(embedded_scan_cubin(), "q4p_gdn_scan_64_64")
        .expect("q4p_gdn_scan_64_64");
    cuda.launch(
        prep,
        (KEY_HEADS as u32, TOKENS as u32, 1),
        (KEY as u32, 1, 1),
        &[
            mixed_d.0,
            a_d.0,
            b_d.0,
            a_log_d.0,
            dt_d.0,
            qk.0,
            decay.0,
            key_width as u64,
            KEY_HEADS as u64,
            HEADS as u64,
            VALUE as u64,
            1,
        ],
    )
    .expect("prep");
    let splits = 2usize;
    cuda.launch(
        scan,
        (HEADS as u32, splits as u32, 1),
        ((4 * VALUE / splits) as u32, 1, 1),
        &[
            qk.0,
            decay.0,
            mixed_d.0,
            state.0,
            state.0,
            raw.0,
            key_width as u64,
            KEY_HEADS as u64,
            HEADS as u64,
            VALUE as u64,
            1,
            TOKENS as u64,
            0,
            0,
            0,
        ],
    )
    .expect("scan");
    let got = download(&cuda, &raw, TOKENS * HEADS * VALUE);
    let got_state = download(&cuda, &state, HEADS * KEY * VALUE);

    // The host twin: the same normalisation and the delta-rule recurrence, in f32.
    let mut want = vec![0.0f32; TOKENS * HEADS * VALUE];
    let mut want_state = vec![0.0f32; HEADS * KEY * VALUE];
    for token in 0..TOKENS {
        for head in 0..HEADS {
            let kh = head;
            let row = &mixed[token * conv_width..(token + 1) * conv_width];
            let q = &row[kh * KEY..kh * KEY + KEY];
            let k = &row[key_width + kh * KEY..key_width + kh * KEY + KEY];
            let v = &row[key_width * 2 + head * VALUE..key_width * 2 + head * VALUE + VALUE];
            let q_norm = (q.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt().recip()
                * (KEY as f32).sqrt().recip();
            let k_norm = (k.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt().recip();
            let decay = (-a_log[head].exp()
                * (1.0 + (a[token * HEADS + head] + dt_bias[head]).exp()).ln())
            .exp();
            let beta = 1.0 / (1.0 + (-b[token * HEADS + head]).exp());
            let s = &mut want_state[head * KEY * VALUE..(head + 1) * KEY * VALUE];
            let mut memory = vec![0.0f32; VALUE];
            for ki in 0..KEY {
                for vi in 0..VALUE {
                    s[ki * VALUE + vi] *= decay;
                    memory[vi] += s[ki * VALUE + vi] * (k[ki] * k_norm);
                }
            }
            for vi in 0..VALUE {
                let delta = (v[vi] - memory[vi]) * beta;
                let mut output = 0.0f32;
                for ki in 0..KEY {
                    s[ki * VALUE + vi] += (k[ki] * k_norm) * delta;
                    output += s[ki * VALUE + vi] * (q[ki] * q_norm);
                }
                want[token * HEADS * VALUE + head * VALUE + vi] = output;
            }
        }
    }
    assert_close("scan outputs", &want, &got);
    assert_close("scan state", &want_state, &got_state);
}

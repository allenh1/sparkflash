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

//! The GLM bulk prefill and routed-expert (MoE) images: the block-scaled FP8 dense GEMMs (over
//! the f32 activation and over its dynamic E4M3 codes), the routed-expert plan/MMA/SwiGLU and the
//! skinny BF16 projection against a host reference, the wide BF16 GEMM against the 128-column
//! one, and the bulk indexer scores and the exact projection against the decode step's kernels,
//! on an sm_121a node. The MoE symbols live in their own image
//! (kernels/glm_moe_sm121.cu); glm_quantise_rows stays in the prefill image.

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_model_qwen4_exp::device::{
    cubin_available, embedded_glm_moe_cubin, embedded_glm_prefill_cubin,
};
use sparkflash_weights::quant;

fn open() -> Option<Cuda> {
    if !cubin_available()
        || embedded_glm_prefill_cubin().is_empty()
        || embedded_glm_moe_cubin().is_empty()
    {
        eprintln!("skipped: this build embeds no GLM prefill or routed-expert cubin");
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
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn download(cuda: &Cuda, pointer: &Deviceptr, bytes: usize) -> Vec<u8> {
    cuda.synchronize().expect("synchronize");
    let mut out = vec![0u8; bytes];
    cuda.copy_dtoh(&mut out, pointer).expect("download");
    out
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
fn assert_close(what: &str, host: &[f32], device: &[f32], tolerance: f32) {
    assert_eq!(host.len(), device.len(), "{what}: lengths");
    let scale = host
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-3);
    let worst = host
        .iter()
        .zip(device)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "{what}: worst {worst:.3e} scale {scale:.3} relative {:.3e}",
        worst / scale
    );
    assert!(device.iter().all(|v| v.is_finite()), "{what}: non-finite");
    assert!(
        worst / scale < tolerance,
        "{what}: relative {} exceeds {tolerance}",
        worst / scale
    );
}

/// The release's dynamic activation of row-major `x` (rows of whole 128-column blocks): each
/// block's amax / 448 and the E4M3 codes of x over it, as q4e_tg_fp8_activation quantises.
fn quantised(x: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut codes = Vec::with_capacity(x.len());
    let mut scales = Vec::with_capacity(x.len() / 128);
    for block in x.as_chunks::<128>().0 {
        let scale = block.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 448.0;
        scales.push(scale);
        codes.extend(block.iter().map(|v| {
            if scale == 0.0 {
                0
            } else {
                quant::e4m3_encode(v / scale)
            }
        }));
    }
    (codes, scales)
}

/// [quantised]'s codes times their block's scale.
fn dequantised(x: &[f32]) -> Vec<f32> {
    let (codes, scales) = quantised(x);
    codes
        .iter()
        .enumerate()
        .map(|(i, code)| quant::f8e4m3_to_f32(*code) * scales[i / 128])
        .collect()
}

/// The E4M3 x E4M3 GEMMs' dynamic shared memory (GLM_Q8_SHARED).
const Q8_SHARED: u32 = 68_608;
/// The wide BF16 GEMM's dynamic shared memory (GLM_WIDE_SHARED).
const WIDE_SHARED: u32 = 98_304;

fn launch(cuda: &Cuda, name: &'static str, grid: (u32, u32, u32), threads: u32, args: &[u64]) {
    // The routed-expert family has its own image. glm_quantise_rows stays in the prefill image
    // because the dense q8 path uses it too.
    let image = if name == "moe_plan" {
        sparkflash_kernels::images::moe_plan_cubin()
    } else if name.starts_with("glm_moe_") {
        embedded_glm_moe_cubin()
    } else if name.starts_with("glm_prefill_heads_")
        || name.starts_with("glm_prefill_mla_")
        || name == "glm_prefill_index_scores"
    {
        sparkflash_kernels::images::glm_attention_cubin()
    } else if name.starts_with("q4e_glm_") || name == "q4e_gemv_tiled_bf16_parts" {
        sparkflash_kernels::images::glm_decode_cubin()
    } else {
        embedded_glm_prefill_cubin()
    };
    let loaded = if name.ends_with("_q8_mma") {
        cuda.load_named_shared(image, name, Q8_SHARED)
    } else if name.starts_with("glm_prefill_bf16_wide") {
        cuda.load_named_shared(image, name, WIDE_SHARED)
    } else {
        cuda.load_named(image, name)
    };
    let (_module, kernel) = loaded.unwrap_or_else(|error| panic!("{name}: {error}"));
    cuda.launch(kernel, grid, (threads, 1, 1), args)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
}

#[test]
fn the_bulk_fp8_dense_gemm_matches_the_host_block_scaled_linear() {
    let Some(cuda) = open() else { return };
    let (cols, inner, tokens) = (128usize, 256usize, 32usize);
    let mut rng = Lcg(0x676c_6d31);
    let codes: Vec<u8> = (0..cols * inner).map(|_| rng.code()).collect();
    let scales: Vec<f32> = (0..(cols / 128) * (inner / 128))
        .map(|_| (0.5 + rng.next().abs()) * 1e-3)
        .collect();
    let x = rng.values(tokens * inner);
    let weights: Vec<f32> = codes
        .iter()
        .map(|code| quant::f8e4m3_to_f32(*code))
        .collect();
    let mut host = vec![0.0f32; tokens * cols];
    for t in 0..tokens {
        for c in 0..cols {
            let mut sum = 0.0f32;
            for k in 0..inner {
                let scale = scales[(c / 128) * (inner / 128) + k / 128];
                sum += weights[c * inner + k] * scale * x[t * inner + k];
            }
            host[t * cols + c] = sum;
        }
    }
    let w = upload(&cuda, &codes);
    let ws = upload_f32(&cuda, &scales);
    let a = upload_f32(&cuda, &x);
    let out = upload_f32(&cuda, &vec![0.0; tokens * cols]);
    launch(
        &cuda,
        "glm_prefill_fp8b128_mma",
        (1, 1, 1),
        256,
        &[
            w.0,
            ws.0,
            a.0,
            out.0,
            cols as u64,
            inner as u64,
            tokens as u64,
        ],
    );
    let got = download_f32(&cuda, &out, tokens * cols);
    assert_close("bulk fp8 dense", &host, &got, 5e-3);
    // The bf16-activation form drops the split's low half: the same block-scaled sum to the
    // precision a bf16 activation carries, which is finer than the FP8 the model quantises its
    // activations with.
    let out = upload_f32(&cuda, &vec![0.0; tokens * cols]);
    launch(
        &cuda,
        "glm_prefill_fp8b128_mma_bf16a",
        (1, 1, 1),
        256,
        &[
            w.0,
            ws.0,
            a.0,
            out.0,
            cols as u64,
            inner as u64,
            tokens as u64,
        ],
    );
    let got = download_f32(&cuda, &out, tokens * cols);
    assert_close("bulk fp8 dense, bf16 activation", &host, &got, 5e-2);
}

/// The per-head projection groups its weight and scale strides: head h's tile scales start
/// ws_grp_stride *elements* on, not bytes (D88's regression). Two heads make an f32 stride read
/// as four times too far, so heads >= 2 read the wrong tile scale; the test holds head 1 to the
/// host block-scaled linear over its own codes.
///
/// The layout is the host's: kv_b's [head][nope + value][latent] rows, the weight pointer at the
/// first value row, and one f32 scale a 128 x 128 block over the head's key and value blocks.
#[test]
fn the_head_tiled_fp8_projection_strides_its_per_head_scales_as_elements() {
    let Some(cuda) = open() else { return };
    let (heads, nope, value, latent, tokens) = (2usize, 128usize, 128usize, 512usize, 40usize);
    let per_head = nope + value;
    let blocks = latent / 128;
    let mut rng = Lcg(0x676c_6d68);
    // kv_b's codes as the checkpoint stores them, every head whole.
    let codes: Vec<u8> = (0..heads * per_head * latent).map(|_| rng.code()).collect();
    // One f32 tile scale a 128 x 128 block of each head: the key row block (nope rows) then the
    // value row block, each with one scale a K block. The host passes the value block of head 0
    // and an eight-element head stride; a byte stride reads head 1 two elements on instead of
    // eight, which is the wrong tile scale.
    let scales: Vec<f32> = (0..heads * 2 * blocks)
        .map(|index| ((1 + index % 3) as f32 * 0.5 + rng.next().abs()) * 1e-3)
        .collect();
    // The absorbed query as [tokens][heads][latent]; the output as [tokens][heads][value].
    let x = rng.values(tokens * heads * latent);
    let w = upload(&cuda, &codes);
    let ws = upload_f32(&cuda, &scales);
    let a = upload_f32(&cuda, &x);
    let out = upload_f32(&cuda, &vec![0.0; tokens * heads * value]);
    launch(
        &cuda,
        "glm_prefill_heads_fp8b128_mma",
        (
            (value / 128) as u32,
            tokens.div_ceil(128) as u32,
            heads as u32,
        ),
        256,
        &[
            // The host points at the first value row of head 0 and strides whole heads.
            w.offset(nope * latent).0,
            ws.offset(4 * (nope / 128) * blocks).0,
            a.0,
            out.0,
            value as u64,
            latent as u64,
            tokens as u64,
            (heads * latent) as u64,
            (heads * value) as u64,
            latent as u64,
            value as u64,
            (per_head * latent) as u64,
            ((per_head / 128) * blocks) as u64,
        ],
    );
    let got = download_f32(&cuda, &out, tokens * heads * value);
    let mut host = vec![0.0f32; tokens * heads * value];
    for t in 0..tokens {
        for head in 0..heads {
            let first = head * 2 * blocks + blocks;
            for column in 0..value {
                let row = (head * per_head + nope + column) * latent;
                let mut sum = 0.0f32;
                for k in 0..latent {
                    sum += quant::f8e4m3_to_f32(codes[row + k])
                        * scales[first + (column / 128) * blocks + k / 128]
                        * x[(t * heads + head) * latent + k];
                }
                host[(t * heads + head) * value + column] = sum;
            }
        }
    }
    assert_close("head-tiled fp8 projection", &host, &got, 5e-3);
}

#[test]
fn the_bulk_q8_gemm_matches_the_host_dynamic_activation_linear() {
    let Some(cuda) = open() else { return };
    // Three K blocks (an odd count of 128-column folds), two column tiles and a partial row tile.
    let (cols, inner, tokens) = (256usize, 384usize, 200usize);
    let mut rng = Lcg(0x676c_6d33);
    let codes: Vec<u8> = (0..cols * inner).map(|_| rng.code()).collect();
    let scales: Vec<f32> = (0..(cols / 128) * (inner / 128))
        .map(|_| (0.5 + rng.next().abs()) * 1e-3)
        .collect();
    let mut x = rng.values(tokens * inner);
    // A zero block quantises to zero codes under a zero scale.
    x[..128].fill(0.0);
    let (x_codes, x_scales) = quantised(&x);
    let input = upload_f32(&cuda, &x);
    let q = upload(&cuda, &vec![0u8; tokens * inner]);
    let qs = upload_f32(&cuda, &vec![1.0; tokens * inner / 128]);
    launch(
        &cuda,
        "glm_quantise_rows",
        ((tokens * inner / 128 * 32).div_ceil(256) as u32, 1, 1),
        256,
        &[input.0, q.0, qs.0, tokens as u64, inner as u64],
    );
    assert_eq!(
        download(&cuda, &q, tokens * inner),
        x_codes,
        "the device's codes are the host's"
    );
    let got_scales = download_f32(&cuda, &qs, tokens * inner / 128);
    assert!(
        got_scales
            .iter()
            .zip(&x_scales)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the device's scales are the host's"
    );
    let x = dequantised(&x);
    let mut host = vec![0.0f32; tokens * cols];
    for t in 0..tokens {
        for c in 0..cols {
            let mut sum = 0.0f64;
            for k in 0..inner {
                let scale = scales[(c / 128) * (inner / 128) + k / 128];
                sum += f64::from(quant::f8e4m3_to_f32(codes[c * inner + k]) * scale)
                    * f64::from(x[t * inner + k]);
            }
            host[t * cols + c] = sum as f32;
        }
    }
    let w = upload(&cuda, &codes);
    let ws = upload_f32(&cuda, &scales);
    let out = upload_f32(&cuda, &vec![f32::NAN; tokens * cols]);
    launch(
        &cuda,
        "glm_prefill_q8_mma",
        ((cols / 128) as u32, tokens.div_ceil(128) as u32, 1),
        256,
        &[
            w.0,
            ws.0,
            q.0,
            qs.0,
            out.0,
            cols as u64,
            inner as u64,
            tokens as u64,
        ],
    );
    let got = download_f32(&cuda, &out, tokens * cols);
    assert_close("bulk q8 dense", &host, &got, 1e-5);
}

#[test]
fn the_skinny_bf16_projection_matches_the_host() {
    let Some(cuda) = open() else { return };
    // An odd row count (a warp's second row past the end) and columns past the last whole
    // 256-column round.
    let (rows, columns) = (37usize, 1032usize);
    let mut rng = Lcg(0x676c_6d34);
    let x = rng.values(rows * columns);
    let input = upload_f32(&cuda, &x);
    for outs in [24usize, 16, 5, 32] {
        let weights: Vec<u16> = (0..outs * columns)
            .map(|_| (rng.next().to_bits() >> 16) as u16)
            .collect();
        let widened = |i: usize| f32::from_bits(u32::from(weights[i]) << 16);
        let mut host = vec![0.0f32; rows * outs];
        for r in 0..rows {
            for j in 0..outs {
                let sum: f64 = (0..columns)
                    .map(|k| f64::from(widened(j * columns + k)) * f64::from(x[r * columns + k]))
                    .sum();
                host[r * outs + j] = sum as f32;
            }
        }
        let w = upload(
            &cuda,
            &weights
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let out = upload_f32(&cuda, &vec![f32::NAN; rows * outs]);
        let name = match outs.div_ceil(8) {
            1 => "glm_prefill_skinny_bf16_1",
            2 => "glm_prefill_skinny_bf16_2",
            3 => "glm_prefill_skinny_bf16_3",
            _ => "glm_prefill_skinny_bf16_4",
        };
        launch(
            &cuda,
            name,
            (rows.div_ceil(16) as u32, 1, 1),
            256,
            &[
                w.0,
                input.0,
                out.0,
                outs as u64,
                columns as u64,
                rows as u64,
            ],
        );
        let got = download_f32(&cuda, &out, rows * outs);
        assert_close(&format!("skinny bf16, {outs} outputs"), &host, &got, 1e-5);
        // And the kernel's own chain to the bit: lane l's fmaf over its eight columns of each
        // 256-column round in column order from 0.0, then the xor butterfly, output j from lane j.
        for r in 0..rows {
            for j in 0..outs {
                let mut lanes = [0.0f32; 32];
                for (l, sum) in lanes.iter_mut().enumerate() {
                    for k in (l * 8..columns).step_by(256) {
                        for c in k..k + 8 {
                            *sum = widened(j * columns + c).mul_add(x[r * columns + c], *sum);
                        }
                    }
                }
                for offset in [16, 8, 4, 2, 1] {
                    let before = lanes;
                    for (l, sum) in lanes.iter_mut().enumerate() {
                        *sum = before[l] + before[l ^ offset];
                    }
                }
                assert_eq!(
                    got[r * outs + j].to_bits(),
                    lanes[j].to_bits(),
                    "skinny bf16, {outs} outputs: row {r} output {j} is not the lane chain"
                );
            }
        }
    }
}

#[test]
fn the_bulk_routed_experts_match_the_host_swiglu_and_down() {
    // 150 rows of two picks: each expert's 150 slots are a whole 128-row item and a partial
    // one, so the plan splits an expert and the MMA skips a partial item's empty row tiles.
    let picks: Vec<[u32; 2]> = (0..150)
        .map(|t| if t % 3 == 0 { [1, 0] } else { [0, 1] })
        .collect();
    routed_experts(2, 128, 128, &picks, false);
    routed_experts(2, 128, 128, &picks, true);
}

#[test]
fn the_bulk_routed_experts_hold_a_skewed_chunk() {
    // Expert 0 takes every row (two items, one partial past 64 rows) and experts 1 and 2 split
    // the second pick, over two FP8 blocks of K and several column tiles.
    let picks: Vec<[u32; 2]> = (0..200).map(|t| [0, 1 + (t % 2) as u32]).collect();
    routed_experts(3, 256, 256, &picks, false);
    routed_experts(3, 256, 256, &picks, true);
}

/// The plan, gate/up GEMM, SwiGLU and down GEMM of `picks` (a row's two experts) against the
/// host's slot-by-slot SwiGLU and down projection; with `q8`, over the release's dynamic
/// activations (the input rows and the activated slots quantised as the decode GEMV does).
fn routed_experts(experts: usize, hidden: usize, width: usize, picks: &[[u32; 2]], q8: bool) {
    let Some(cuda) = open() else { return };
    let (rows, topk) = (picks.len(), 2usize);
    let limit = 10.0f32;
    let mut rng = Lcg(0x676c_6d32);
    let x = rng.values(rows * hidden);
    let route: Vec<f32> = (0..rows * topk).map(|_| 0.5 + rng.next().abs()).collect();
    let mut gate_up_codes = Vec::new();
    let mut gate_up_scales = Vec::new();
    let mut down_codes = Vec::new();
    let mut down_scales = Vec::new();
    for _ in 0..experts {
        gate_up_codes.extend((0..2 * width * hidden).map(|_| rng.code()));
        gate_up_scales.extend(
            (0..(2 * width / 128) * (hidden / 128)).map(|_| (0.5 + rng.next().abs()) * 1e-3),
        );
        down_codes.extend((0..hidden * width).map(|_| rng.code()));
        down_scales
            .extend((0..(hidden / 128) * (width / 128)).map(|_| (0.5 + rng.next().abs()) * 1e-3));
    }
    // Host reference, slot by slot.
    let f8 = |code: u8| quant::f8e4m3_to_f32(code);
    let gu = |e: usize, r: usize, k: usize| {
        f8(gate_up_codes[e * 2 * width * hidden + r * hidden + k])
            * gate_up_scales
                [e * (2 * width / 128) * (hidden / 128) + (r / 128) * (hidden / 128) + k / 128]
    };
    let dn = |e: usize, r: usize, k: usize| {
        f8(down_codes[e * hidden * width + r * width + k])
            * down_scales[e * (hidden / 128) * (width / 128) + (r / 128) * (width / 128) + k / 128]
    };
    let mut expected = vec![0.0f32; rows * topk * hidden];
    let x_host = if q8 { dequantised(&x) } else { x.clone() };
    for token in 0..rows {
        let x = &x_host;
        let mut act = vec![0.0f32; width];
        for (slot, &pick) in picks[token].iter().enumerate() {
            let e = pick as usize;
            for (column, value) in act.iter_mut().enumerate() {
                let mut gate = 0.0f32;
                let mut up = 0.0f32;
                for k in 0..hidden {
                    gate += gu(e, column, k) * x[token * hidden + k];
                    up += gu(e, width + column, k) * x[token * hidden + k];
                }
                let gate = gate.min(limit);
                let up = up.clamp(-limit, limit);
                *value = gate / (1.0 + (-gate).exp()) * up;
            }
            if q8 {
                act = dequantised(&act);
            }
            let at = token * topk + slot;
            for (c, out) in expected[at * hidden..(at + 1) * hidden]
                .iter_mut()
                .enumerate()
            {
                let mut sum = 0.0f32;
                for (k, value) in act.iter().enumerate() {
                    sum += dn(e, c, k) * value;
                }
                *out = sum * route[at];
            }
        }
    }
    // The device buffers.
    let mut gu_tables = Vec::new();
    let mut dn_tables = Vec::new();
    let mut keep = Vec::new();
    for e in 0..experts {
        let gw = upload(
            &cuda,
            &gate_up_codes[e * 2 * width * hidden..(e + 1) * 2 * width * hidden],
        );
        let gs = upload_f32(
            &cuda,
            &gate_up_scales[e * (2 * width / 128) * (hidden / 128)
                ..(e + 1) * (2 * width / 128) * (hidden / 128)],
        );
        let dw = upload(
            &cuda,
            &down_codes[e * hidden * width..(e + 1) * hidden * width],
        );
        let ds = upload_f32(
            &cuda,
            &down_scales
                [e * (hidden / 128) * (width / 128)..(e + 1) * (hidden / 128) * (width / 128)],
        );
        gu_tables.extend([gw.0.to_le_bytes(), gs.0.to_le_bytes()].concat());
        dn_tables.extend([dw.0.to_le_bytes(), ds.0.to_le_bytes()].concat());
        keep.extend([gw, gs, dw, ds]);
    }
    let gu_table = upload(&cuda, &gu_tables);
    let dn_table = upload(&cuda, &dn_tables);
    let indices: Vec<i32> = picks.iter().flatten().map(|v| *v as i32).collect();
    let indices = upload(
        &cuda,
        &indices
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    // glm_prefill_sm121.cu's GLM_MOE_ROWS: one expert's slots a 128-row item.
    let item_rows = 128usize;
    let max_items = (rows * topk).div_ceil(item_rows) + experts.min(rows * topk) + 1;
    let moe_rows = max_items * item_rows;
    let slot_of = upload_f32(&cuda, &vec![0.0; moe_rows]); // overwritten with ints by the plan
    let items = upload(&cuda, &vec![0u8; max_items * 16]);
    let item_count = upload(&cuda, &[0u8; 4]);
    let input = upload_f32(&cuda, &x);
    let gate_up = upload_f32(&cuda, &vec![0.0; moe_rows * 2 * width]);
    let activated = upload_f32(&cuda, &vec![0.0; moe_rows * width]);
    let route_dev = upload_f32(&cuda, &route);
    let out = upload_f32(&cuda, &vec![0.0; rows * topk * hidden]);
    // The shared plan (kernels/moe_plan_sm121.cu) writes `sorted` and the item list; the MMA
    // bodies read an item's slots from `sorted` at the item's first rank.
    launch(
        &cuda,
        "moe_plan",
        (1, 1, 1),
        1024,
        &[
            indices.0,
            slot_of.0,
            items.0,
            item_count.0,
            0,
            (rows * topk) as u64,
            experts as u64,
            topk as u64,
            item_rows as u64,
            max_items as u64,
            0,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    );
    let plan_count = download_i32(&cuda, &item_count, 1)[0].max(0) as usize;
    let plan_items = download_i32(&cuda, &items, max_items * 4);
    let plan_slots = download_i32(&cuda, &slot_of, moe_rows);
    let want_items: usize = (0..experts)
        .map(|e| {
            picks
                .iter()
                .flatten()
                .filter(|v| **v as usize == e)
                .count()
                .div_ceil(item_rows)
        })
        .sum();
    assert_eq!(plan_count, want_items, "plan item count");
    for item in 0..plan_count {
        let expert = plan_items[item * 4] as usize;
        let count = plan_items[item * 4 + 2] as usize;
        assert!(
            (1..=item_rows).contains(&count),
            "plan item {item} holds {count} slots"
        );
        let first = plan_items[item * 4 + 1] as usize;
        for k in 0..count {
            let slot = plan_slots[first + k] as usize;
            assert_eq!(
                picks[slot / topk][slot % topk] as usize,
                expert,
                "slot {slot}"
            );
        }
    }
    for item in plan_count..max_items {
        assert_eq!(
            plan_items[item * 4 + 2],
            0,
            "unused item {item} runs nothing"
        );
    }
    if q8 {
        let codes = upload(&cuda, &vec![0u8; rows * hidden]);
        let scales = upload_f32(&cuda, &vec![0.0; rows * hidden / 128]);
        let act_codes = upload(&cuda, &vec![0u8; moe_rows * width]);
        let act_scales = upload_f32(&cuda, &vec![0.0; moe_rows * width / 128]);
        launch(
            &cuda,
            "glm_quantise_rows",
            ((rows * hidden / 128 * 32).div_ceil(256) as u32, 1, 1),
            256,
            &[input.0, codes.0, scales.0, rows as u64, hidden as u64],
        );
        launch(
            &cuda,
            "glm_moe_q8_mma",
            ((2 * width).div_ceil(128) as u32, 1, max_items as u32),
            256,
            &[
                gu_table.0,
                items.0,
                codes.0,
                scales.0,
                gate_up.0,
                (2 * width) as u64,
                hidden as u64,
                slot_of.0,
                topk as u64,
                0,
            ],
        );
        launch(
            &cuda,
            "glm_moe_swiglu_q8",
            ((moe_rows * width / 128 * 32).div_ceil(256) as u32, 1, 1),
            256,
            &[
                gate_up.0,
                width as u64,
                moe_rows as u64,
                u64::from(limit.to_bits()),
                act_codes.0,
                act_scales.0,
            ],
        );
        launch(
            &cuda,
            "glm_moe_q8_mma",
            (hidden.div_ceil(128) as u32, 1, max_items as u32),
            256,
            &[
                dn_table.0,
                items.0,
                act_codes.0,
                act_scales.0,
                out.0,
                hidden as u64,
                width as u64,
                slot_of.0,
                topk as u64,
                route_dev.0,
            ],
        );
        let got = download_f32(&cuda, &out, rows * topk * hidden);
        // A device gate/up sum a rounding away from the host's can land its activation on the
        // neighbouring code, so the down projection's tolerance is the FP8 step's, not f32's.
        assert_close("bulk routed experts, q8", &expected, &got, 5e-3);
        return;
    }
    launch(
        &cuda,
        "glm_moe_fp8b128_mma",
        ((2 * width).div_ceil(128) as u32, 1, max_items as u32),
        256,
        &[
            gu_table.0,
            items.0,
            input.0,
            gate_up.0,
            (2 * width) as u64,
            hidden as u64,
            slot_of.0,
            topk as u64,
            0,
        ],
    );
    launch(
        &cuda,
        "glm_moe_swiglu",
        ((moe_rows * width).div_ceil(256) as u32, 1, 1),
        256,
        &[
            gate_up.0,
            width as u64,
            moe_rows as u64,
            u64::from(limit.to_bits()),
            activated.0,
        ],
    );
    launch(
        &cuda,
        "glm_moe_fp8b128_mma",
        (hidden.div_ceil(128) as u32, 1, max_items as u32),
        256,
        &[
            dn_table.0,
            items.0,
            activated.0,
            out.0,
            hidden as u64,
            width as u64,
            slot_of.0,
            topk as u64,
            route_dev.0,
        ],
    );
    let got = download_f32(&cuda, &out, rows * topk * hidden);
    assert_close("bulk routed experts", &expected, &got, 5e-3);
}

#[test]
fn the_bulk_swiglu_reads_gate_and_up_halves() {
    let Some(cuda) = open() else { return };
    let width = 4usize;
    let gate = [1.0f32, -2.0, 3.0, -4.0];
    let up = [5.0f32, 6.0, -7.0, -8.0];
    let mut source = Vec::new();
    source.extend_from_slice(&gate);
    source.extend_from_slice(&up);
    let input = upload_f32(&cuda, &source);
    let out = upload_f32(&cuda, &vec![0.0; width]);
    launch(
        &cuda,
        "glm_moe_swiglu",
        (1, 1, 1),
        256,
        &[
            input.0,
            width as u64,
            1,
            u64::from(10.0f32.to_bits()),
            out.0,
        ],
    );
    let got = download_f32(&cuda, &out, width);
    let want: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    assert_close("bulk swiglu", &want, &got, 1e-5);
}

/// The decode step's index score of one row against one pool, as q4e_glm_index_scores computes
/// it: lane l's fmaf chain over channels l, l + 32, l + 64, l + 96 from 0.0, q4e_warp_sum's
/// shuffle-down fold (lane 0's value), and the head-ordered sum of w_h max(dot * score_scale, 0).
fn decode_index_score(
    query: &[f32],
    weights: &[f32],
    key: &[f32],
    score_scale: f32,
    head_scale: f32,
) -> f32 {
    let mut score = 0.0f32;
    for head in 0..32 {
        let q = &query[head * 128..head * 128 + 128];
        let mut lanes = [0.0f32; 32];
        for (l, part) in lanes.iter_mut().enumerate() {
            for step in 0..4 {
                *part = q[l + 32 * step].mul_add(key[l + 32 * step], *part);
            }
        }
        for offset in [16, 8, 4, 2, 1] {
            let before = lanes;
            for (l, part) in lanes.iter_mut().enumerate().take(32 - offset) {
                *part = before[l] + before[l + offset];
            }
        }
        let weight = (weights[head] * head_scale) * 1.0;
        score += weight * (lanes[0] * score_scale).max(0.0);
    }
    score
}

#[test]
fn the_bulk_index_scores_are_the_decode_kernels_bit_for_bit() {
    let Some(cuda) = open() else { return };
    if sparkflash_kernels::images::glm_attention_cubin().is_empty()
        || sparkflash_kernels::images::glm_decode_cubin().is_empty()
    {
        eprintln!("skipped: this build embeds no GLM attention or decode cubin");
        return;
    }
    let (heads, dim, pool) = (32usize, 128usize, 4usize);
    let score_scale = 1.0 / (dim as f32).sqrt();
    let head_scale = 1.0 / (heads as f32).sqrt();
    let mut rng = Lcg(0x676c_6978);
    // A first chunk (the triangle, a partial row tile), a run starting off a pool boundary over
    // a partial pool tile, and a short run behind a long context.
    for (position, rows) in [(0usize, 77usize), (4093, 300), (1000, 33)] {
        let pools = (position + rows) / pool;
        // Past the last pool, keys the kernels must not read into a score.
        let mut keys = rng.values((pools + 70) * dim);
        let mut query = rng.values(rows * heads * dim);
        let weights = rng.values(rows * heads);
        // Signed zeros, whose sums the chain order decides.
        for (i, value) in query.iter_mut().enumerate().step_by(97) {
            *value = if i % 2 == 0 { 0.0 } else { -0.0 };
        }
        for value in keys.iter_mut().step_by(89) {
            *value = -0.0;
        }
        let score_stride = pools + 70;
        let sentinel = f32::from_bits(0x7FC0_1234);
        let (q, w, k) = (
            upload_f32(&cuda, &query),
            upload_f32(&cuda, &weights),
            upload_f32(&cuda, &keys),
        );
        let decode = upload_f32(&cuda, &vec![sentinel; rows * score_stride]);
        let bulk = upload_f32(&cuda, &vec![sentinel; rows * score_stride]);
        launch(
            &cuda,
            "q4e_glm_index_scores",
            (pools.div_ceil(64).max(1) as u32, rows as u32, 1),
            256,
            &[
                q.0,
                w.0,
                k.0,
                decode.0,
                position as u64,
                heads as u64,
                dim as u64,
                pool as u64,
                u64::from(score_scale.to_bits()),
                u64::from(head_scale.to_bits()),
                score_stride as u64,
            ],
        );
        launch(
            &cuda,
            "glm_prefill_index_scores",
            (
                pools.div_ceil(64).max(1) as u32,
                rows.div_ceil(32) as u32,
                1,
            ),
            128,
            &[
                q.0,
                w.0,
                k.0,
                bulk.0,
                position as u64,
                pool as u64,
                u64::from(score_scale.to_bits()),
                u64::from(head_scale.to_bits()),
                score_stride as u64,
                rows as u64,
            ],
        );
        let expected = download_f32(&cuda, &decode, rows * score_stride);
        let got = download_f32(&cuda, &bulk, rows * score_stride);
        for row in 0..rows {
            let own = (position + row + 1) / pool;
            for p in 0..score_stride {
                let at = row * score_stride + p;
                assert_eq!(
                    got[at].to_bits(),
                    expected[at].to_bits(),
                    "position {position}: row {row} pool {p} (of {own}) is {} against {}",
                    got[at],
                    expected[at]
                );
                if p >= own {
                    assert_eq!(
                        got[at].to_bits(),
                        sentinel.to_bits(),
                        "row {row} wrote pool {p}"
                    );
                }
            }
            // And the decode kernel's own order, on a sample of the row's pools.
            for p in (row % 7..own).step_by(53) {
                let host = decode_index_score(
                    &query[row * heads * dim..(row + 1) * heads * dim],
                    &weights[row * heads..(row + 1) * heads],
                    &keys[p * dim..(p + 1) * dim],
                    score_scale,
                    head_scale,
                );
                assert_eq!(
                    got[row * score_stride + p].to_bits(),
                    host.to_bits(),
                    "position {position}: row {row} pool {p} is not the decode order"
                );
            }
        }
    }
}

#[test]
fn the_wide_bf16_gemm_is_the_128_column_tiles_bit_for_bit() {
    let Some(cuda) = open() else { return };
    // Whole and partial 256-column tiles (a 128-column remainder, a 96-column part), partial row
    // tiles and a 17-row chunk, each with the split activation and its head alone.
    let mut rng = Lcg(0x7769_6465);
    for (rows, outs, columns) in [
        (300usize, 384usize, 512usize),
        (130, 96, 256),
        (17, 2048, 128),
        (513, 640, 1024),
    ] {
        // Rows spanning magnitudes, so the remainder is often far below the head, and signed
        // zeros among them.
        let mut x: Vec<f32> = (0..rows * columns)
            .map(|i| rng.next() * f32::powi(2.0, (i / columns % 41) as i32 - 20))
            .collect();
        for (i, value) in x.iter_mut().enumerate().step_by(97) {
            *value = if i % 2 == 0 { 0.0 } else { -0.0 };
        }
        let weights: Vec<u16> = (0..outs * columns)
            .map(|_| ((rng.next() * 0.05).to_bits() >> 16) as u16)
            .collect();
        let w = upload(
            &cuda,
            &weights
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let input = upload_f32(&cuda, &x);
        let sentinel = f32::from_bits(0x7FC0_4321);
        for (narrow, wide) in [
            ("glm_prefill_bf16_mma", "glm_prefill_bf16_wide"),
            ("glm_prefill_bf16_mma_bf16a", "glm_prefill_bf16_wide_bf16a"),
        ] {
            // A row past the last, which neither kernel may write.
            let expected = upload_f32(&cuda, &vec![sentinel; (rows + 1) * outs]);
            let got = upload_f32(&cuda, &vec![sentinel; (rows + 1) * outs]);
            let args = |out: &Deviceptr| {
                [
                    w.0,
                    input.0,
                    out.0,
                    outs as u64,
                    columns as u64,
                    rows as u64,
                ]
            };
            launch(
                &cuda,
                narrow,
                (outs.div_ceil(128) as u32, rows.div_ceil(128) as u32, 1),
                256,
                &args(&expected),
            );
            launch(
                &cuda,
                wide,
                (outs.div_ceil(256) as u32, rows.div_ceil(128) as u32, 1),
                256,
                &args(&got),
            );
            let expected = download_f32(&cuda, &expected, (rows + 1) * outs);
            let got = download_f32(&cuda, &got, (rows + 1) * outs);
            assert!(
                expected[..rows * outs]
                    .iter()
                    .any(|v| *v != 0.0 && v.is_finite()),
                "{narrow} {rows} x {outs} x {columns}: no output"
            );
            for (at, (got, expected)) in got.iter().zip(&expected).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "{wide} {rows} x {outs} x {columns}: row {} output {} is {got} against \
                     {expected}",
                    at / outs,
                    at % outs
                );
            }
        }
    }
}

#[test]
fn the_exact_projection_is_the_tiled_gemvs_bit_for_bit() {
    let Some(cuda) = open() else { return };
    if sparkflash_kernels::images::glm_decode_cubin().is_empty() {
        eprintln!("skipped: this build embeds no GLM decode cubin");
        return;
    }
    use sparkflash_ops::gemv::{TILED_GEMV_ROUND, tiled_gemv_splits};
    // The router's shape, a partial output tile over four rounds, and one round (no split sum).
    // Each is a shape the tiled GEMV splits one round a block, the only one the kernel takes;
    // 600 rows run the tiled GEMV as a 512-row group and a partial one, as gemv_chunked does.
    let rows = 600usize;
    let mut rng = Lcg(0x676c_6578);
    for (outs, columns) in [(288usize, 4096usize), (40, 1024), (32, 256)] {
        let tiles = outs.div_ceil(32);
        let rounds = columns / TILED_GEMV_ROUND;
        assert_eq!(
            tiled_gemv_splits(tiles, columns),
            rounds,
            "{outs} x {columns}"
        );
        let mut x = rng.values(rows * columns);
        for (i, value) in x.iter_mut().enumerate().step_by(101) {
            *value = if i % 2 == 0 { 0.0 } else { -0.0 };
        }
        let mut weights: Vec<u16> = (0..outs * columns)
            .map(|_| (rng.next().to_bits() >> 16) as u16)
            .collect();
        for value in weights.iter_mut().step_by(83) {
            *value = 0x8000;
        }
        let widened = |i: usize| f32::from_bits(u32::from(weights[i]) << 16);
        let w = upload(
            &cuda,
            &weights
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let input = upload_f32(&cuda, &x);
        let sentinel = f32::from_bits(0x7FC0_1234);
        // A row past the last, which neither kernel may write.
        let tiled = upload_f32(&cuda, &vec![sentinel; (rows + 1) * outs]);
        let exact = upload_f32(&cuda, &vec![sentinel; (rows + 1) * outs]);
        let groups = 64usize;
        let mut total = 0;
        let partials = cuda
            .alloc(groups * rounds * 8 * tiles * 32 * 4, &mut total)
            .expect("alloc");
        let counters = upload(&cuda, &vec![0u8; groups * tiles * 4]);
        for done in (0..rows).step_by(512) {
            let count = 512.min(rows - done);
            launch(
                &cuda,
                "q4e_gemv_tiled_bf16_parts",
                (tiles as u32, rounds as u32, count.div_ceil(8) as u32),
                256,
                &[
                    w.0,
                    input.0 + (done * columns * 4) as u64,
                    tiled.0 + (done * outs * 4) as u64,
                    outs as u64,
                    columns as u64,
                    count as u64,
                    partials.0,
                    counters.0,
                    outs as u64,
                    outs as u64,
                    outs as u64,
                ],
            );
        }
        launch(
            &cuda,
            "glm_prefill_exact_bf16",
            (tiles as u32, rows.div_ceil(32) as u32, 1),
            128,
            &[
                w.0,
                input.0,
                exact.0,
                outs as u64,
                columns as u64,
                rows as u64,
            ],
        );
        let expected = download_f32(&cuda, &tiled, (rows + 1) * outs);
        let got = download_f32(&cuda, &exact, (rows + 1) * outs);
        for (at, (got, expected)) in got.iter().zip(&expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "{outs} x {columns}: row {} output {} is {got} against {expected}",
                at / outs,
                at % outs
            );
        }
        assert!(
            got[rows * outs..]
                .iter()
                .all(|v| v.to_bits() == sentinel.to_bits()),
            "{outs} x {columns}: a write past the last row"
        );
        // And the tiled GEMV's own chain on a sample: lane l's fmaf over its eight columns of a
        // round from 0.0, the xor butterfly, and the rounds' partials added in order from 0.0.
        for at in (0..rows * outs).step_by(173) {
            let (r, j) = (at / outs, at % outs);
            let mut sum = 0.0f32;
            for round in 0..rounds {
                let mut lanes = [0.0f32; 32];
                for (l, lane) in lanes.iter_mut().enumerate() {
                    for c in round * 256 + l * 8..round * 256 + l * 8 + 8 {
                        *lane = widened(j * columns + c).mul_add(x[r * columns + c], *lane);
                    }
                }
                for offset in [16, 8, 4, 2, 1] {
                    let before = lanes;
                    for (l, lane) in lanes.iter_mut().enumerate() {
                        *lane = before[l] + before[l ^ offset];
                    }
                }
                sum = if rounds > 1 { sum + lanes[0] } else { lanes[0] };
            }
            assert_eq!(
                got[at].to_bits(),
                sum.to_bits(),
                "{outs} x {columns}: row {r} output {j} is not the tiled chain"
            );
        }
    }
}

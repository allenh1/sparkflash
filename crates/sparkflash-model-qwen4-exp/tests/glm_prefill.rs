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

//! The GLM bulk prefill image: the block-scaled FP8 dense GEMMs (over the f32 activation and over
//! its dynamic E4M3 codes), the routed-expert plan/MMA/SwiGLU and the skinny BF16 projection
//! against a host reference, on an sm_121a node.

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_model_qwen4_exp::device::{cubin_available, embedded_glm_prefill_cubin};
use sparkflash_weights::quant;

fn open() -> Option<Cuda> {
    if !cubin_available() || embedded_glm_prefill_cubin().is_empty() {
        eprintln!("skipped: this build embeds no GLM prefill cubin");
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

fn launch(cuda: &Cuda, name: &'static str, grid: (u32, u32, u32), threads: u32, args: &[u64]) {
    let loaded = if name.ends_with("_q8_mma") {
        cuda.load_named_shared(embedded_glm_prefill_cubin(), name, Q8_SHARED)
    } else {
        cuda.load_named(embedded_glm_prefill_cubin(), name)
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
    launch(
        &cuda,
        "glm_moe_plan",
        (1, 1, 1),
        1024,
        &[
            indices.0,
            (rows * topk) as u64,
            experts as u64,
            slot_of.0,
            items.0,
            item_count.0,
            max_items as u64,
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
        for k in 0..count {
            let slot = plan_slots[item * item_rows + k] as usize;
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

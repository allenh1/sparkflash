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

//! Inkling-2-Small's sm_121a kernels against the host oracle, at a TP4 rank's geometry.
//!
//! The oracle is [sparkflash_model_inkling2::reference], the same functions the f32 host forward
//! calls, so a kernel cannot pass by agreeing with a second hand-written reference. Every case is
//! at one rank's widths: 12 query heads and 2 key/value heads of 128, 2,304 intermediate
//! columns, a 1,024-token window and eight relative dimensions.
//!
//! The kernels sum in their own association and use fast reciprocal square roots and
//! exponentials, so the bound is relative to the host's largest magnitude, the bound the other
//! packs hold their kernels to. It skips - loudly - without an embedded cubin or a CUDA driver.

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_kernels::images::inkling_cubin;
use sparkflash_model_inkling2::reference::{
    self, AttentionParams, logit_softcap, short_conv, short_conv_decode, short_conv_dynamic,
    short_conv_dynamic_decode, softcapped_swiglu,
};

/// One TP4 rank's attention geometry.
const HEADS: usize = 12;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const D_REL: usize = 8;
const REL_EXTENT: usize = 16;
const WINDOW: usize = 37;
const VALUE_NORM: i32 = 1;
const EPS: f32 = 1e-6;
const QK_CLIP: f32 = 22.627_417;
const REL_CLIP: f32 = 8.0;
/// The largest difference allowed, relative to the host's largest magnitude.
const RELATIVE_TOLERANCE: f32 = 2e-5;

/// An integer argument.
fn u(value: usize) -> u64 {
    value as u64
}

/// A float argument: the driver takes one 8-byte cell per parameter.
fn f(value: f32) -> u64 {
    u64::from(value.to_bits())
}

fn open() -> Option<Cuda> {
    if inkling_cubin().is_empty() {
        eprintln!("skipped: this build embeds no Inkling cubin");
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

/// A deterministic generator, so a failure is reproducible.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

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

fn upload(cuda: &Cuda, bytes: &[u8]) -> Deviceptr {
    let mut total = 0;
    let pointer = cuda.alloc(bytes.len().max(4), &mut total).expect("alloc");
    cuda.copy_htod(&pointer, bytes).expect("upload");
    pointer
}

fn upload_f32(cuda: &Cuda, values: &[f32]) -> Deviceptr {
    upload(
        cuda,
        &values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    )
}

/// bf16 bits, as the kernels read the convolution taps and the key/value cache.
fn upload_bf16(cuda: &Cuda, values: &[f32]) -> Deviceptr {
    let bits: Vec<u8> = values
        .iter()
        .flat_map(|value| {
            let rounded = sparkflash_weights::quant::f32_to_bf16_bits(*value);
            rounded.to_le_bytes()
        })
        .collect();
    upload(cuda, &bits)
}

fn upload_zeros(cuda: &Cuda, elements: usize) -> Deviceptr {
    upload_f32(cuda, &vec![0f32; elements])
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

fn download_bf16(cuda: &Cuda, pointer: &Deviceptr, elements: usize) -> Vec<f32> {
    cuda.synchronize().expect("synchronize");
    let mut bytes = vec![0u8; elements * 2];
    cuda.copy_dtoh(&mut bytes, pointer).expect("download");
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|word| sparkflash_weights::quant::bf16_to_f32(u16::from_le_bytes(*word)))
        .collect()
}

/// A bound for a kernel that writes bf16: one part in 128 covers a one-ulp difference at the
/// stored precision, which is what a fast tanh or sigmoid can move.
fn assert_close_bf16(what: &str, host: &[f32], device: &[f32]) {
    assert_eq!(host.len(), device.len(), "{what}: lengths differ");
    let largest = host.iter().fold(0f32, |value, item| value.max(item.abs()));
    let mut worst = 0f32;
    for (expected, actual) in host.iter().zip(device) {
        worst = worst.max((expected - actual).abs());
    }
    assert!(
        worst <= largest / 128.0,
        "{what}: the device is off by {worst} where the host's largest value is {largest}"
    );
}

/// The largest difference between two runs, relative to the host's largest magnitude.
fn assert_close(what: &str, host: &[f32], device: &[f32]) {
    assert_eq!(host.len(), device.len(), "{what}: lengths differ");
    let largest = host.iter().fold(0f32, |value, item| value.max(item.abs()));
    let mut worst = 0f32;
    for (expected, actual) in host.iter().zip(device) {
        worst = worst.max((expected - actual).abs());
    }
    assert!(
        worst <= RELATIVE_TOLERANCE * largest.max(1e-12),
        "{what}: the device is off by {worst} where the host's largest value is {largest}"
    );
}

fn kernel<'a>(
    cuda: &'a Cuda,
    name: &'static str,
) -> (sparkflash_kernels::cuda_driver::Module, sparkflash_kernels::cuda_driver::Kernel) {
    cuda.load_named(inkling_cubin(), name)
        .unwrap_or_else(|error| panic!("load {name}: {error}"))
}

#[test]
fn the_static_short_convolution_matches_the_host() {
    let Some(cuda) = open() else {
        return;
    };
    let channels = KV_HEADS * HEAD_DIM;
    let kernel_width = 4usize;
    let rows = 5usize;
    let mut rng = Lcg::new(0x1c0d);
    let x = rng.values(rows * channels);
    let taps = rng.values(channels * kernel_width);
    let host = short_conv(&x, &taps, rows, channels, kernel_width);

    let (_, kernel) = kernel(&cuda, "inkling_short_conv");
    let x_device = upload_f32(&cuda, &x);
    let taps_device = upload_bf16(&cuda, &taps);
    let out = upload_zeros(&cuda, rows * channels);
    let blocks = channels.div_ceil(256) as u32;
    cuda.launch(
        kernel,
        (blocks, rows as u32, 1),
        (256, 1, 1),
        &[x_device.0, taps_device.0, out.0, u(rows), u(channels), u(kernel_width)],
    )
    .expect("short_conv launch");
    let device = download_f32(&cuda, &out, rows * channels);
    // The taps are bf16 on the device, so the host reads the same rounded values.
    let rounded: Vec<f32> = taps
        .iter()
        .map(|value| {
            sparkflash_weights::quant::bf16_to_f32(
                sparkflash_weights::quant::f32_to_bf16_bits(*value),
            )
        })
        .collect();
    let host = short_conv(&x, &rounded, rows, channels, kernel_width);
    assert_close("short_conv", &host, &device);
}

#[test]
fn the_static_short_convolution_decode_matches_the_prefill() {
    let Some(cuda) = open() else {
        return;
    };
    let channels = KV_HEADS * HEAD_DIM;
    let kernel_width = 4usize;
    let history = kernel_width - 1;
    let mut rng = Lcg::new(0xdec0de);
    let taps: Vec<f32> = rng.values(channels * kernel_width);
    let rounded: Vec<f32> = taps
        .iter()
        .map(|value| {
            sparkflash_weights::quant::bf16_to_f32(
                sparkflash_weights::quant::f32_to_bf16_bits(*value),
            )
        })
        .collect();
    let tokens: Vec<Vec<f32>> = (0..6).map(|_| rng.values(channels)).collect();

    // The host decodes the same stream, one token at a time, through the shared oracle.
    let mut state = vec![0f32; channels * history];
    let mut host = Vec::new();
    for token in &tokens {
        host.extend(short_conv_decode(token, &rounded, &mut state, channels, kernel_width));
    }

    let (_, kernel) = kernel(&cuda, "inkling_short_conv_decode");
    let taps_device = upload_bf16(&cuda, &taps);
    let state_device = upload_zeros(&cuda, channels * history);
    let out = upload_zeros(&cuda, channels);
    let blocks = channels.div_ceil(256) as u32;
    let mut device = Vec::new();
    for token in &tokens {
        let x = upload_f32(&cuda, token);
        cuda.launch(
            kernel,
            (blocks, 1, 1),
            (256, 1, 1),
            &[
                x.0,
                taps_device.0,
                state_device.0,
                out.0,
                u(channels),
                u(kernel_width),
            ],
        )
        .expect("short_conv_decode launch");
        device.extend(download_f32(&cuda, &out, channels));
    }
    assert_close("short_conv_decode", &host, &device);
}

#[test]
fn the_dynamic_short_convolution_matches_the_host() {
    let Some(cuda) = open() else {
        return;
    };
    let channels = KV_HEADS * HEAD_DIM;
    let kernel_width = 4usize;
    let rows = 3usize;
    let mut rng = Lcg::new(0xd1);
    let x = rng.values(rows * channels);
    let static_taps = rng.values(kernel_width * channels);
    let dynamic = rng.values(rows * KV_HEADS * kernel_width);
    let rounded: Vec<f32> = static_taps
        .iter()
        .map(|value| {
            sparkflash_weights::quant::bf16_to_f32(
                sparkflash_weights::quant::f32_to_bf16_bits(*value),
            )
        })
        .collect();
    let host = short_conv_dynamic(&x, &rounded, &dynamic, rows, channels, kernel_width, HEAD_DIM);

    let (_, kernel) = kernel(&cuda, "inkling_short_conv_dynamic");
    let x_device = upload_f32(&cuda, &x);
    let static_device = upload_bf16(&cuda, &static_taps);
    let dynamic_device = upload_f32(&cuda, &dynamic);
    let out = upload_zeros(&cuda, rows * channels);
    let blocks = channels.div_ceil(256) as u32;
    cuda.launch(
        kernel,
        (blocks, rows as u32, 1),
        (256, 1, 1),
        &[
            x_device.0,
            static_device.0,
            dynamic_device.0,
            out.0,
            u(rows),
            u(channels),
            u(kernel_width),
            u(HEAD_DIM),
        ],
    )
    .expect("short_conv_dynamic launch");
    let device = download_f32(&cuda, &out, rows * channels);
    assert_close("short_conv_dynamic", &host, &device);
}

#[test]
fn the_dynamic_short_convolution_decode_matches_the_host() {
    let Some(cuda) = open() else {
        return;
    };
    let channels = KV_HEADS * HEAD_DIM;
    let kernel_width = 4usize;
    let history = kernel_width - 1;
    let mut rng = Lcg::new(0xd2);
    let static_taps = rng.values(kernel_width * channels);
    let rounded: Vec<f32> = static_taps
        .iter()
        .map(|value| {
            sparkflash_weights::quant::bf16_to_f32(
                sparkflash_weights::quant::f32_to_bf16_bits(*value),
            )
        })
        .collect();
    let tokens: Vec<Vec<f32>> = (0..5).map(|_| rng.values(channels)).collect();
    let dynamic: Vec<Vec<f32>> = (0..5).map(|_| rng.values(KV_HEADS * kernel_width)).collect();

    let mut state = vec![0f32; channels * history];
    let mut host = Vec::new();
    for (token, taps) in tokens.iter().zip(&dynamic) {
        host.extend(short_conv_dynamic_decode(
            token,
            &rounded,
            taps,
            &mut state,
            channels,
            kernel_width,
            HEAD_DIM,
        ));
    }

    let (_, kernel) = kernel(&cuda, "inkling_short_conv_dynamic_decode");
    let static_device = upload_bf16(&cuda, &static_taps);
    let state_device = upload_zeros(&cuda, channels * history);
    let out = upload_zeros(&cuda, channels);
    let blocks = channels.div_ceil(256) as u32;
    let mut device = Vec::new();
    for (token, taps) in tokens.iter().zip(&dynamic) {
        let x = upload_f32(&cuda, token);
        let dynamic_device = upload_f32(&cuda, taps);
        cuda.launch(
            kernel,
            (blocks, 1, 1),
            (256, 1, 1),
            &[
                x.0,
                static_device.0,
                dynamic_device.0,
                state_device.0,
                out.0,
                u(channels),
                u(kernel_width),
                u(HEAD_DIM),
            ],
        )
        .expect("short_conv_dynamic_decode launch");
        device.extend(download_f32(&cuda, &out, channels));
    }
    assert_close("short_conv_dynamic_decode", &host, &device);
}

#[test]
fn the_softcapped_swiglu_matches_the_host() {
    let Some(cuda) = open() else {
        return;
    };
    let width = 2_304usize;
    let rows = 2usize;
    let gate_cap = 4.0f32;
    let up_cap = 25.0f32;
    let mut rng = Lcg::new(0x5161);
    let gate_up = rng.values(rows * 2 * width);
    let mut host = Vec::new();
    for row in 0..rows {
        host.extend(softcapped_swiglu(
            &gate_up[row * 2 * width..(row + 1) * 2 * width],
            width,
            gate_cap,
            up_cap,
        ));
    }
    let (_, kernel) = kernel(&cuda, "inkling_softcapped_swiglu");
    let source = upload_f32(&cuda, &gate_up);
    let out = upload_zeros(&cuda, rows * width);
    let blocks = width.div_ceil(256) as u32;
    cuda.launch(
        kernel,
        (blocks, rows as u32, 1),
        (256, 1, 1),
        &[source.0, out.0, u(width), f(gate_cap), f(up_cap)],
    )
    .expect("softcapped_swiglu launch");
    let device = download_f32(&cuda, &out, rows * width);
    assert_close("softcapped_swiglu", &host, &device);
}

#[test]
fn the_logit_softcap_matches_the_host() {
    let Some(cuda) = open() else {
        return;
    };
    let padded = 50_304usize;
    let unpadded = 50_256usize;
    let cap = 30.0f32;
    let mut rng = Lcg::new(0x10917);
    let logits = rng.values(padded);
    let host = logit_softcap(&logits, 1, padded, unpadded, cap);
    let (_, kernel) = kernel(&cuda, "inkling_logit_softcap");
    let source = upload_f32(&cuda, &logits);
    let out = upload_bf16(&cuda, &vec![0f32; unpadded]);
    cuda.launch(
        kernel,
        (1, 1, 1),
        (256, 1, 1),
        &[source.0, out.0, 1, u(padded), u(unpadded), f(cap)],
    )
    .expect("logit_softcap launch");
    let device = download_bf16(&cuda, &out, unpadded);
    // The device writes bf16, so compare against the host rounded to bf16's grid.
    let rounded: Vec<f32> = host
        .iter()
        .map(|value| {
            sparkflash_weights::quant::bf16_to_f32(
                sparkflash_weights::quant::f32_to_bf16_bits(*value),
            )
        })
        .collect();
    assert_close_bf16("logit_softcap", &rounded, &device);
}

#[test]
fn the_relative_attention_decode_matches_the_host() {
    let Some(cuda) = open() else {
        return;
    };
    let mut rng = Lcg::new(0xa77e);
    let q_width = HEADS * HEAD_DIM;
    let kv_width = KV_HEADS * HEAD_DIM;
    let q = rng.values(q_width);
    let gate = rng.values(q_width);
    let rel = rng.values(HEADS * D_REL);
    let q_scale = rng.values(q_width);
    let rel_scale = rng.values(HEADS * D_REL);
    let sink = rng.values(HEADS);
    let rel_bank_raw = rng.values(D_REL * REL_EXTENT);
    let positions = WINDOW;
    let keys: Vec<Vec<f32>> = (0..positions).map(|_| rng.values(kv_width)).collect();
    let values: Vec<Vec<f32>> = (0..positions).map(|_| rng.values(kv_width)).collect();
    let round: fn(&[f32]) -> Vec<f32> = |items| {
        items
            .iter()
            .map(|value| {
                sparkflash_weights::quant::bf16_to_f32(
                    sparkflash_weights::quant::f32_to_bf16_bits(*value),
                )
            })
            .collect()
    };
    let rel_bank = round(&rel_bank_raw);
    let host = reference::attention(
        &q,
        &gate,
        &round(&keys.concat()).chunks_exact(kv_width).map(<[f32]>::to_vec).collect::<Vec<_>>(),
        &round(&values.concat()).chunks_exact(kv_width).map(<[f32]>::to_vec).collect::<Vec<_>>(),
        &rel,
        &rel_bank,
        &sink,
        &q_scale,
        None,
        &rel_scale,
        &AttentionParams {
            heads: HEADS,
            kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            d_rel: D_REL,
            rel_extent: REL_EXTENT,
            qk_clip: QK_CLIP,
            rel_clip: REL_CLIP,
            tau: 1.0,
            value_norm: VALUE_NORM != 0,
            eps: f64::from(EPS),
        },
    )
    .expect("the host attention");

    let (_, kernel) = kernel(&cuda, "inkling_attn_decode");
    let q_device = upload_f32(&cuda, &q);
    let gate_device = upload_f32(&cuda, &gate);
    let keys_device = upload_bf16(&cuda, &keys.concat());
    let values_device = upload_bf16(&cuda, &values.concat());
    let rel_device = upload_f32(&cuda, &rel);
    let bank_device = upload_bf16(&cuda, &rel_bank_raw);
    let sink_device = upload_f32(&cuda, &sink);
    let q_scale_device = upload_f32(&cuda, &q_scale);
    let k_scale_device = upload_f32(&cuda, &vec![1.0f32; kv_width]);
    let rel_scale_device = upload_f32(&cuda, &rel_scale);
    let out = upload_bf16(&cuda, &vec![0f32; q_width]);
    cuda.launch(
        kernel,
        (HEADS as u32, 1, 1),
        (32, 1, 1),
        &[
            q_device.0,
            gate_device.0,
            keys_device.0,
            values_device.0,
            rel_device.0,
            bank_device.0,
            sink_device.0,
            q_scale_device.0,
            k_scale_device.0,
            rel_scale_device.0,
            out.0,
            u(HEADS),
            u(KV_HEADS),
            u(HEAD_DIM),
            u(HEADS / KV_HEADS),
            u(positions - 1),
            u(WINDOW),
            u(REL_EXTENT),
            u(D_REL),
            f(QK_CLIP),
            f(REL_CLIP),
            f(1.0),
            u(VALUE_NORM as usize),
            f(EPS),
        ],
    )
    .expect("attn_decode launch");
    let device = download_bf16(&cuda, &out, q_width);
    // The device writes bf16, so the host is rounded to the same grid before comparing.
    let rounded: Vec<f32> = host
        .iter()
        .map(|value| {
            sparkflash_weights::quant::bf16_to_f32(
                sparkflash_weights::quant::f32_to_bf16_bits(*value),
            )
        })
        .collect();
    assert_close_bf16("attn_decode", &rounded, &device);
}
#[test]
fn the_nvfp4_expert_matvec_matches_the_host() {
    let Some(cuda) = open() else {
        return;
    };
    // One rank's gate/up row count: 2 * 2,304 rows of a 128-wide hidden state would be too
    // large a dump to compare, so this runs a 96-row slice of the same geometry.
    let rows = 96usize;
    let columns = 128usize;
    let scale_2 = 0.017_5f32;
    let mut rng = Lcg::new(0xf4);
    let mut codes = vec![0u8; rows * columns.div_ceil(2)];
    for byte in &mut codes {
        *byte = (rng.next().abs() * 255.0) as u8;
    }
    let scale_columns = columns.div_ceil(16);
    let mut scales = vec![0u8; rows * scale_columns];
    for code in &mut scales {
        *code = 0x30 + (rng.next().abs() * 0x20 as f32) as u8;
    }
    let x = rng.values(columns);
    let host = reference::nvfp4_matvec(&codes, &scales, scale_2, &x, rows, columns)
        .expect("the host product");

    let (_, kernel) = kernel(&cuda, "inkling_nvfp4_expert_matvec");
    let codes_device = upload(&cuda, &codes);
    let scales_device = upload(&cuda, &scales);
    let x_device = upload_f32(&cuda, &x);
    let out = upload_zeros(&cuda, rows);
    cuda.launch(
        kernel,
        (rows as u32, 1, 1),
        (256, 1, 1),
        &[
            codes_device.0,
            scales_device.0,
            x_device.0,
            out.0,
            u(rows),
            u(columns),
            f(scale_2),
        ],
    )
    .expect("nvfp4_expert_matvec launch");
    let device = download_f32(&cuda, &out, rows);
    assert_close("nvfp4_expert_matvec", &host, &device);
}

#[test]
fn the_mxfp8_expert_matvec_matches_the_host() {
    let Some(cuda) = open() else {
        return;
    };
    let rows = 96usize;
    let columns = 128usize;
    let block = 32usize;
    let mut rng = Lcg::new(0xf8);
    let mut codes = vec![0u8; rows * columns];
    for byte in &mut codes {
        *byte = (rng.next().abs() * 255.0) as u8;
    }
    let scale_columns = columns.div_ceil(block);
    let mut scales = vec![0u8; rows * scale_columns];
    for code in &mut scales {
        *code = 0x78 + (rng.next().abs() * 8.0) as u8;
    }
    let x = rng.values(columns);
    let host = reference::mxfp8_matvec(&codes, &scales, &x, rows, columns, block)
        .expect("the host product");

    let (_, kernel) = kernel(&cuda, "inkling_mxfp8_expert_matvec");
    let codes_device = upload(&cuda, &codes);
    let scales_device = upload(&cuda, &scales);
    let x_device = upload_f32(&cuda, &x);
    let out = upload_zeros(&cuda, rows);
    cuda.launch(
        kernel,
        (rows as u32, 1, 1),
        (256, 1, 1),
        &[
            codes_device.0,
            scales_device.0,
            x_device.0,
            out.0,
            u(rows),
            u(columns),
            u(block),
        ],
    )
    .expect("mxfp8_expert_matvec launch");
    let device = download_f32(&cuda, &out, rows);
    assert_close("mxfp8_expert_matvec", &host, &device);
}

#[test]
fn the_sigmoid_router_matches_the_host() {
    let Some(cuda) = open() else {
        return;
    };
    let experts = 384usize;
    let top_k = 8usize;
    let route_scale = 8.0f32;
    let mut rng = Lcg::new(0x4074);
    let logits = rng.values(experts);
    let bias = rng.values(experts);
    let (host_indices, host_weights) =
        reference::router_topk(&logits, &bias, experts, top_k, route_scale);

    let (_, kernel) = kernel(&cuda, "inkling_router_topk");
    let logits_device = upload_f32(&cuda, &logits);
    let bias_device = upload_f32(&cuda, &bias);
    let indices_device = upload(&cuda, &vec![0u8; top_k * 4]);
    let weights_device = upload_zeros(&cuda, top_k);
    cuda.launch(
        kernel,
        (1, 1, 1),
        (32, 1, 1),
        &[
            logits_device.0,
            bias_device.0,
            indices_device.0,
            weights_device.0,
            u(experts),
            u(top_k),
            f(route_scale),
        ],
    )
    .expect("router_topk launch");
    cuda.synchronize().expect("synchronize");
    let mut index_bytes = vec![0u8; top_k * 4];
    cuda.copy_dtoh(&mut index_bytes, &indices_device)
        .expect("download indices");
    let device_indices: Vec<usize> = index_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| i32::from_le_bytes(*word) as usize)
        .collect();
    let device_weights = download_f32(&cuda, &weights_device, top_k);
    assert_eq!(device_indices, host_indices, "the router chose other experts");
    assert_close("router_topk", &host_weights, &device_weights);
}


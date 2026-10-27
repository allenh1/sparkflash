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

//! CPU twin of the Engram head's three device kernels: the native dtype rule of audit F4.
//!
//! The released Engram head is not the chain's f32 seam (inference/model.py:311-365):
//!
//! * the fp8 table row is dequantised in f32 and **cast to bf16** before the projection
//!   (model.py:318-319);
//! * wkv is an fp8 **W8A8** Linear over that bf16 row (model.py:345, linear() at model.py:186-204):
//!   act_quant(block 32, ue8m0) then an fp8 GEMM whose block scale is applied to the *block dot*;
//! * the key/value/gate math is f32 over the bf16 wkv output, and the write is
//!   `(h + gate * value).to(x.dtype)`, one bf16 cast (model.py:355-365).
//!
//! Each function here reproduces the corresponding kernel (`kernels/engram_embed_sm121.cu`,
//! `kernels/engram_wkv_sm121.cu`, `kernels/engram_gate_write_sm121.cu`) op for op, including the
//! kernel's reduction order, so a device run and this reference are comparable bit for bit. The one
//! declared exception is the gate's f32 sigmoid, where the host and the device each evaluate the
//! F8-class library exponential: the twin test compares it with a stated tolerance and records the
//! measured maximum, it does not claim bit equality.
//!
//! Values are carried as f32 throughout, because the engine's buffers are f32. Every point where the
//! released code casts to bf16 is a [bf16_round] here, so the bf16 typing is explicit in the code
//! rather than implicit in the buffer type.

use crate::engram_head_ref::{HASH_SENTINEL, HeadShape};
use crate::m4_quant::{
    QUANT_BLOCK, act_quant_fp8_row, bf16_bits, bf16_value, fp8_e4m3_decode, ue8m0_decode,
};

/// The largest hc the gate kernel's shared gate array holds (it refuses larger).
pub const MAX_HC: usize = 8;
/// The reference's gate clamp before the signed square root (model.py:355, 1e-6).
pub const GATE_CLAMP: f32 = 1e-6;

/// The nearest bf16 to an f32, as an f32 (ties to even): one of the released code's casts.
#[must_use]
pub fn bf16_round(value: f32) -> f32 {
    bf16_value(bf16_bits(value))
}

/// The bf16 bits of an f32, i.e. what the kernel stores in a `__nv_bfloat16`.
#[must_use]
pub fn bf16_bits_of(value: f32) -> u16 {
    bf16_bits(value)
}

/// One table row dequantised and cast to bf16, as `engram_embed_sm121` writes it.
///
/// `codes` is the row's `head_dim` E4M3 bytes and `scales` its `head_dim / 32` E8M0 bytes; the
/// multiply is exact (the scale is a power of two) and the only rounding is the bf16 cast, which is
/// the released code's `values.to(torch.bfloat16)`.
///
/// # Panics
/// Panics when the scale grid does not cover the row width.
#[must_use]
pub fn embed_row(codes: &[u8], scales: &[u8], head_dim: usize) -> Vec<f32> {
    assert!(codes.len() >= head_dim, "the row is shorter than head_dim");
    assert!(
        scales.len() >= head_dim.div_ceil(QUANT_BLOCK),
        "the scale grid does not cover the row"
    );
    let mut row = Vec::with_capacity(head_dim);
    for column in 0..head_dim {
        let scale = ue8m0_decode(scales[column / QUANT_BLOCK]);
        row.push(bf16_round(fp8_e4m3_decode(codes[column]) * scale));
    }
    row
}

/// The activation quantiser of `engram_embed_sm121`: the reference act_quant rule (kernel.py:41-112
/// with scale_fmt=`"ue8m0"`) over the bf16 row, i.e. [act_quant_fp8_row] on bf16-valued inputs.
///
/// # Panics
/// Panics when the row is not a whole number of 32-value blocks.
#[must_use]
pub fn act_quantise(fetched: &[f32]) -> (Vec<u8>, Vec<u8>) {
    act_quant_fp8_row(fetched)
}

/// One 32-value K block's dot product in `engram_wkv_sm121`'s order: the five-step
/// `__shfl_down_sync` tree (offsets 16, 8, 4, 2, 1), which is what the kernel's reduction is.
///
/// # Panics
/// Panics unless both slices hold exactly [QUANT_BLOCK] values.
#[must_use]
pub fn wkv_block_dot(act_codes: &[u8], weight_codes: &[u8]) -> f32 {
    assert_eq!(act_codes.len(), QUANT_BLOCK, "a block dot takes 32 codes");
    assert_eq!(
        weight_codes.len(),
        QUANT_BLOCK,
        "a block dot takes 32 codes"
    );
    let mut partial = [0.0f32; QUANT_BLOCK];
    for lane in 0..QUANT_BLOCK {
        partial[lane] = fp8_e4m3_decode(act_codes[lane]) * fp8_e4m3_decode(weight_codes[lane]);
    }
    for offset in [16usize, 8, 4, 2, 1] {
        for lane in 0..offset {
            partial[lane] += partial[lane + offset];
        }
    }
    partial[0]
}

/// The W8A8 wkv projection for one token, as `engram_wkv_sm121` computes it.
///
/// `weight` is the checkpoint layout `[out_width][inner]` E4M3 and `weight_scales` its
/// `[out_width / 32][inner / 32]` E8M0 grid; the activation is one `inner`-long code vector with
/// `inner / 32` scales. The returned values are the bf16 outputs of the released Linear, carried as
/// f32 (each is bf16-exact), which is the dtype the gate consumes (`kv` at model.py:354).
///
/// # Panics
/// Panics when the buffers do not hold the declared shape.
#[must_use]
pub fn wkv(
    act_codes: &[u8],
    act_scales: &[u8],
    weight: &[u8],
    weight_scales: &[u8],
    out_width: usize,
    inner: usize,
) -> Vec<f32> {
    assert_eq!(act_codes.len(), inner, "the activation width");
    assert_eq!(act_scales.len(), inner / QUANT_BLOCK, "the activation grid");
    assert!(weight.len() >= out_width * inner, "the weight buffer");
    assert!(
        weight_scales.len() >= out_width.div_ceil(QUANT_BLOCK) * (inner / QUANT_BLOCK),
        "the weight scale grid"
    );
    let k_blocks = inner / QUANT_BLOCK;
    let mut out = Vec::with_capacity(out_width);
    for row in 0..out_width {
        let weight_row = &weight[row * inner..(row + 1) * inner];
        let scale_row =
            &weight_scales[(row / QUANT_BLOCK) * k_blocks..((row / QUANT_BLOCK) + 1) * k_blocks];
        let mut accumulator = 0.0f32;
        for kb in 0..k_blocks {
            let base = kb * QUANT_BLOCK;
            let block_dot = wkv_block_dot(
                &act_codes[base..base + QUANT_BLOCK],
                &weight_row[base..base + QUANT_BLOCK],
            );
            accumulator += block_dot * ue8m0_decode(act_scales[kb]) * ue8m0_decode(scale_row[kb]);
        }
        out.push(bf16_round(accumulator));
    }
    out
}

/// The Engram gate for one hc copy, as `engram_gate_write_sm121` computes it.
///
/// `stream` and `key` are bf16-valued (the kernel reads bf16 tensors); `product` is the f32
/// `q_weight * k_weight` the released code forms per column.
#[must_use]
pub fn gate(stream: &[f32], key: &[f32], product: &[f32], dim: usize, eps: f32) -> f32 {
    let mut stream_square = 0.0f32;
    let mut key_square = 0.0f32;
    let mut dot = 0.0f32;
    for column in 0..dim {
        stream_square += stream[column] * stream[column];
        key_square += key[column] * key[column];
        dot += stream[column] * product[column] * key[column];
    }
    let dim_f = dim as f32;
    let rstd =
        (1.0 / (stream_square / dim_f + eps).sqrt()) * (1.0 / (key_square / dim_f + eps).sqrt());
    let dot = dot * rstd * (1.0 / dim_f.sqrt());
    let clamped = dot.abs().max(GATE_CLAMP).sqrt().copysign(dot);
    1.0 / (1.0 + (-clamped).exp())
}

/// The gate and the residual write for one token, as `engram_gate_write_sm121` computes them.
///
/// Returns the hc gates and the hc*dim write result (bf16-exact f32, the released write's cast).
///
/// # Panics
/// Panics when the buffers disagree with the shape or when hc is beyond [MAX_HC].
#[must_use]
pub fn gate_and_write(
    kv: &[f32],
    stream: &[f32],
    product: &[f32],
    dim: usize,
    hc: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert!(
        hc <= MAX_HC,
        "the gate kernel holds at most {MAX_HC} hc copies"
    );
    assert_eq!(
        stream.len(),
        hc * dim,
        "the stream must be one whole hc row"
    );
    assert_eq!(
        product.len(),
        hc * dim,
        "the product must be one whole hc row"
    );
    assert!(kv.len() >= hc * dim + dim, "kv must hold key and value");
    let value = &kv[hc * dim..hc * dim + dim];
    let mut gates = Vec::with_capacity(hc);
    for copy in 0..hc {
        gates.push(gate(
            &stream[copy * dim..(copy + 1) * dim],
            &kv[copy * dim..(copy + 1) * dim],
            &product[copy * dim..(copy + 1) * dim],
            dim,
            eps,
        ));
    }
    let mut out = Vec::with_capacity(hc * dim);
    for copy in 0..hc {
        for column in 0..dim {
            let h = stream[copy * dim + column];
            out.push(bf16_round(h + gates[copy] * value[column]));
        }
    }
    (gates, out)
}

/// One token of the Engram head in the native dtype rule: fetch, bf16 row, W8A8 wkv, gate, write.
///
/// `rows` returns the (codes, scales) of one table row; `x_hc` is updated in place with bf16-exact
/// values, exactly as the released write does. Returns the hc gates, or `None` for a sentinel row,
/// which the engine skips entirely.
///
/// # Panics
/// Panics when the buffers disagree with `shape`.
#[must_use]
pub fn head_native(
    x_hc: &mut [f32],
    ids: &[u32],
    rows: &mut dyn FnMut(u32) -> (Vec<u8>, Vec<u8>),
    wkv_codes: &[u8],
    wkv_scales: &[u8],
    product: &[f32],
    shape: HeadShape,
) -> Option<Vec<f32>> {
    let HeadShape {
        dim,
        hc,
        eps,
        head_dim,
    } = shape;
    assert_eq!(x_hc.len(), hc * dim, "x_hc must be one whole hc row");
    if ids.contains(&HASH_SENTINEL) {
        return None;
    }
    let mut fetched = Vec::with_capacity(ids.len() * head_dim);
    for &id in ids {
        let (codes, scales) = rows(id);
        fetched.extend_from_slice(&embed_row(&codes, &scales, head_dim));
    }
    let (act_codes, act_scales) = act_quantise(&fetched);
    let kv = wkv(
        act_codes.as_slice(),
        act_scales.as_slice(),
        wkv_codes,
        wkv_scales,
        dim * (hc + 1),
        ids.len() * head_dim,
    );
    let stream: Vec<f32> = x_hc.iter().map(|&value| bf16_round(value)).collect();
    let (gates, out) = gate_and_write(&kv, &stream, product, dim, hc, eps);
    x_hc.copy_from_slice(&out);
    Some(gates)
}

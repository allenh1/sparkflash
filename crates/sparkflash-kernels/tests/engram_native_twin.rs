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

//! The Engram head's CPU twin: the native dtype rule (audit F4) against the f32 seam, on released
//! checkpoint rows and on synthetic operands where the arithmetic order is the whole claim.
//!
//! These tests are host-only: they pin what the three kernels implement (engram_embed_sm121,
//! engram_wkv_sm121, engram_gate_write_sm121) against the released code's rule and against the
//! existing f32 reference (engram_head_ref), so a device run at window time compares a kernel whose
//! contract is already fixed. Two things are deliberately NOT claimed here: the device arithmetic
//! itself (that is the window's comparison) and the gate's sigmoid bit (the F8 class, compared with a
//! stated tolerance and its measured maximum printed).

use sparkflash_kernels::engram_head_ref::{
    HeadShape, REAL_ROWS, decode_row, gate as gate_f32, head_ref,
};
use sparkflash_kernels::engram_native_ref::{
    act_quantise, bf16_round, embed_row, gate, gate_and_write, head_native, wkv, wkv_block_dot,
};
use sparkflash_kernels::m4_quant::{act_quant_fp8_row, bf16_bits, fp8_e4m3_decode, ue8m0_decode};

fn hex_bytes(text: &str) -> Vec<u8> {
    assert_eq!(text.len() % 2, 0, "an even hex length");
    (0..text.len() / 2)
        .map(|index| u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).expect("hex"))
        .collect()
}

fn real_row(index: usize) -> (Vec<u8>, Vec<u8>) {
    let row = &REAL_ROWS[index];
    (hex_bytes(row.weight), hex_bytes(row.scale))
}

/// The f32 reference's dequantisation of one real row, rounded to bf16.
fn real_row_f32(index: usize) -> Vec<f32> {
    let (weight, scale) = real_row(index);
    let weight: [u8; 256] = weight.try_into().expect("256 weight bytes");
    let scale: [u8; 8] = scale.try_into().expect("8 scale bytes");
    decode_row(&weight, &scale).to_vec()
}

/// A deterministic E4M3 code stream (no randomness: the tests must reproduce anywhere).
fn code_stream(count: usize, seed: u32) -> Vec<u8> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 16) & 0x7E) as u8
        })
        .collect()
}

#[test]
fn the_row_dequant_is_the_f32_reference_rounded_to_bf16() {
    // The bf16 cast model.py:318-319 adds to the chain's f32 decode is the ONLY difference: on the
    // eight released rows the native rule must be bit-for-bit bf16(f32 decode).
    for (index, row) in REAL_ROWS.iter().enumerate() {
        let (weight, scale) = real_row(index);
        let native = embed_row(&weight, &scale, 256);
        let f32_row = real_row_f32(index);
        assert_eq!(native.len(), 256);
        for column in 0..256 {
            assert_eq!(
                bf16_bits(native[column]),
                bf16_bits(f32_row[column]),
                "row {} column {column}: the dequant is the f32 one",
                row.id
            );
            assert_eq!(
                bf16_round(native[column]),
                native[column],
                "the dequantised row is bf16-exact"
            );
        }
        // The cast is a NO-OP on this format, and that is pinned rather than assumed: an E4M3 value
        // has three mantissa bits and the block scale is a power of two, so every dequantised value
        // is bf16-exact and the f32 seam's missing cast cannot move a bit. The F4 deviation is
        // therefore the activation quantisation and the bf16 wkv output, not row 35.
        for column in 0..256 {
            assert_eq!(
                native[column].to_bits(),
                f32_row[column].to_bits(),
                "row {} column {column}: the bf16 cast is exact for e4m3 x 2^k",
                row.id
            );
        }
        println!(
            "released row {} (layer {}, id {}): {} of 256 values moved under the bf16 cast",
            index,
            REAL_ROWS[index].layer,
            REAL_ROWS[index].id,
            (0..256)
                .filter(
                    |&column| bf16_bits(f32_row[column]) != bf16_bits(bf16_round(f32_row[column]))
                )
                .count()
        );
    }
}

#[test]
fn the_activation_quantiser_is_the_reference_act_quant_rule() {
    // Pinned values, each derived from kernel.py:76-91 and the released constants:
    // * a row of +448 has amax/448 = 1, so the scale is 2^0 (code 127) and every code is E4M3 448;
    // * a row of +1 has amax/448 = 1/448 = 2^-8.8, so the ceil scale is 2^-8 (code 119);
    // * an all-zero row takes the 1e-4 amax floor: 1e-4/448 = 2.23e-7, ceil = 2^-22 (code 105), and
    //   the codes are zero;
    // * the E4M3 tie at 2^-10 rounds to the even code, which is zero.
    let mut row = vec![448.0f32; 64];
    let (codes, scales) = act_quantise(&row);
    assert_eq!(scales, vec![127u8, 127u8]);
    assert_eq!(codes[0], 0x7E, "448 encodes to the format maximum");
    row = vec![1.0f32; 32];
    let (codes_one, scales) = act_quantise(&row);
    assert_eq!(scales, vec![119u8], "1 / 448 ceils to 2^-8");
    assert_eq!(ue8m0_decode(scales[0]), 1.0 / 256.0, "code 119 is 2^-8");
    // 1 / 2^-8 = 256, and E4M3 256 is exponent field 8 + bias 7 = 15 with a zero mantissa: 0x78.
    assert_eq!(codes_one[0], 0x78, "1 / 2^-8 = 256 is E4M3 256");
    let (zero_codes, zero_scales) = act_quantise(&[0.0f32; 32]);
    assert_eq!(zero_scales, vec![105u8], "the 1e-4 floor gives 2^-22");
    assert_eq!(
        zero_codes,
        vec![0u8; 32],
        "a zero block quantises to zero codes"
    );
    // The E4M3 tie: a block whose amax 0.5 gives the scale 2^-9, and a value of 2^-19 then has the
    // quotient 2^-10, exactly between the codes 0 and 1 (the subnormal step is 2^-9), so it rounds
    // to the even one: zero.
    let mut tie_row = vec![0.0f32; 32];
    tie_row[0] = 0.5;
    tie_row[1] = 2.0f32.powf(-19.0);
    let (tie_codes, tie_scales) = act_quantise(&tie_row);
    assert_eq!(tie_scales, vec![118u8], "0.5 / 448 ceils to 2^-9");
    assert_eq!(tie_codes[0], 0x78, "0.5 / 2^-9 = 256 is E4M3 256");
    assert_eq!(tie_codes[1], 0x00, "the 2^-10 tie rounds to the even code");

    // And the shared rule: the native quantiser is the crate's act_quant_fp8_row, not a second copy.
    let stream: Vec<f32> = code_stream(6144, 7)
        .iter()
        .map(|&code| bf16_round(fp8_e4m3_decode(code) * ue8m0_decode(120)))
        .collect();
    let (native_codes, native_scales) = act_quantise(&stream);
    let (rule_codes, rule_scales) = act_quant_fp8_row(&stream);
    assert_eq!(native_codes, rule_codes);
    assert_eq!(native_scales, rule_scales);
}

#[test]
fn the_wkv_block_dot_is_the_kernels_shuffle_tree() {
    let act = code_stream(32, 11);
    let weight = code_stream(32, 13);
    let mut products = [0.0f32; 32];
    for lane in 0..32 {
        products[lane] = fp8_e4m3_decode(act[lane]) * fp8_e4m3_decode(weight[lane]);
    }
    // The independent re-derivation: the same five offsets, written as explicit pairs.
    let mut expected = products;
    for offset in [16usize, 8, 4, 2, 1] {
        for lane in 0..offset {
            expected[lane] += expected[lane + offset];
        }
    }
    assert_eq!(
        wkv_block_dot(&act, &weight).to_bits(),
        expected[0].to_bits()
    );

    // The order matters: the tree and a left-to-right sum disagree on some operand, and this test
    // searches for such a pair instead of asserting the fact on faith.
    let mut differing: Option<(Vec<u8>, Vec<u8>)> = None;
    for seed in 0..2048u32 {
        let a = code_stream(32, seed * 2 + 1);
        let w = code_stream(32, seed * 2 + 2);
        let tree = wkv_block_dot(&a, &w);
        let mut sequential = 0.0f32;
        for lane in 0..32 {
            sequential += fp8_e4m3_decode(a[lane]) * fp8_e4m3_decode(w[lane]);
        }
        if tree.to_bits() != sequential.to_bits() {
            differing = Some((a, w));
            break;
        }
    }
    let (a, w) = differing.expect("the tree and a sequential sum must disagree somewhere");
    println!(
        "shuffle tree {:#010x} vs ascending sum {:#010x}",
        wkv_block_dot(&a, &w).to_bits(),
        (0..32)
            .fold(0.0f32, |acc, lane| acc
                + fp8_e4m3_decode(a[lane]) * fp8_e4m3_decode(w[lane]))
            .to_bits()
    );
}

#[test]
fn the_wkv_projection_scales_the_block_dot_and_rounds_to_bf16() {
    // One 32-column block, so the projection is exactly bf16(tree dot * act scale * weight scale).
    let inner = 32;
    let out_width = 64;
    let act_codes = code_stream(inner, 17);
    let act_scales = vec![121u8];
    let weight = code_stream(out_width * inner, 19);
    let weight_scales = vec![118u8; out_width / 32];
    let projected = wkv(
        &act_codes,
        &act_scales,
        &weight,
        &weight_scales,
        out_width,
        inner,
    );
    assert_eq!(projected.len(), out_width);
    let act_scale = ue8m0_decode(121);
    for row in 0..out_width {
        let block = &weight[row * inner..(row + 1) * inner];
        let dot = wkv_block_dot(&act_codes, block);
        let weight_scale = ue8m0_decode(weight_scales[row / 32]);
        let expected = bf16_round(dot * act_scale * weight_scale);
        assert_eq!(
            projected[row].to_bits(),
            expected.to_bits(),
            "row {row}: the block scale lands on the block dot, then one bf16 cast"
        );
    }
}

#[test]
fn the_gate_is_the_reference_normalised_dot_product() {
    let dim = 5120;
    let stream: Vec<f32> = (0..dim)
        .map(|column| bf16_round(((column as f32) * 0.001).sin()))
        .collect();
    let key: Vec<f32> = (0..dim)
        .map(|column| bf16_round(((column as f32) * 0.002).cos()))
        .collect();
    let product: Vec<f32> = (0..dim).map(|column| 1.0 + column as f32 * 1e-6).collect();
    let eps = 1e-6f32;
    let native = gate(&stream, &key, &product, dim, eps);
    // The independent side: the same formula in f64, which is the reference rule's arithmetic
    // without the f32 rounding. The f32 result may differ by the F8 class's few ulps.
    let (mut stream_square, mut key_square, mut dot) = (0.0f64, 0.0f64, 0.0f64);
    for column in 0..dim {
        let h = f64::from(stream[column]);
        let k = f64::from(key[column]);
        let p = f64::from(product[column]);
        stream_square += h * h;
        key_square += k * k;
        dot += h * p * k;
    }
    let rstd = (1.0 / (stream_square / dim as f64 + f64::from(eps)).sqrt())
        * (1.0 / (key_square / dim as f64 + f64::from(eps)).sqrt());
    let scaled = dot * rstd * (1.0 / (dim as f64).sqrt());
    let f64_gate = 1.0 / (1.0 + (-scaled.abs().max(1e-6).sqrt().copysign(scaled)).exp());
    let delta = (f64::from(native) - f64_gate).abs();
    println!("gate {native:.9} vs f64 {f64_gate:.9}: delta {delta:.3e}");
    // The gate is a normalised dot over 5120 f32 terms, so its f32 accumulation alone is a
    // ~1e-5-class difference from the f64 form before the F8 library term is added. The claim is
    // that class, not bit equality.
    assert!(
        delta < 1e-4,
        "the f32 gate is within the declared accumulation class"
    );

    // And the reference's own f32 gate agrees with this one to the same class.
    let reference = gate_f32(&stream, &key, &product, dim, eps);
    assert!((reference - native).abs() < 1e-6);
}

#[test]
fn the_write_is_the_gated_sum_cast_to_bf16() {
    let dim = 256;
    let hc = 4;
    let kv: Vec<f32> = (0..hc * dim + dim)
        .map(|index| bf16_round(((index as f32) * 0.000_07).cos() * 0.5))
        .collect();
    let stream: Vec<f32> = (0..hc * dim)
        .map(|index| bf16_round(((index as f32) * 0.000_03).sin() * 1.5))
        .collect();
    let product: Vec<f32> = (0..hc * dim)
        .map(|index| 1.0 + index as f32 * 1e-5)
        .collect();
    let (gates, out) = gate_and_write(&kv, &stream, &product, dim, hc, 1e-6);
    assert_eq!(gates.len(), hc);
    let value = &kv[hc * dim..];
    for copy in 0..hc {
        for column in 0..dim {
            let expected = bf16_round(stream[copy * dim + column] + gates[copy] * value[column]);
            assert_eq!(
                out[copy * dim + column].to_bits(),
                expected.to_bits(),
                "copy {copy} column {column}"
            );
        }
    }
}

#[test]
fn the_native_head_on_released_rows_agrees_with_the_f32_head_to_the_declared_class() {
    // A tiny head on the released rows: 24 lookups x 256 columns fetched, dim 64, hc 2. The wkv
    // weights are deterministic synthetic codes, so this test isolates the RULE (bf16 dequant, fp8
    // W8A8, bf16 write) from the checkpoint's 157 MB of real projection weights, which the window
    // comparison will use.
    let head_dim = 256;
    let shape = HeadShape {
        dim: 64,
        hc: 2,
        eps: 1e-6,
        head_dim,
    };
    let ids: Vec<u32> = (0..24u32)
        .map(|index| REAL_ROWS[index as usize % REAL_ROWS.len()].id as u32)
        .collect();
    let fetched_width = ids.len() * head_dim;
    let out_width = shape.dim * (shape.hc + 1);
    let wkv_codes = code_stream(out_width * fetched_width, 23);
    let wkv_scales = vec![121u8; (out_width / 32) * (fetched_width / 32)];
    let product: Vec<f32> = (0..shape.hc * shape.dim)
        .map(|index| 1.0 + index as f32 * 1e-4)
        .collect();
    let x_start: Vec<f32> = (0..shape.hc * shape.dim)
        .map(|index| bf16_round(((index as f32) * 0.000_04).sin()))
        .collect();

    // Native rule.
    let mut x_native = x_start.clone();
    let native_gates = head_native(
        &mut x_native,
        &ids,
        &mut |id| {
            let index = REAL_ROWS
                .iter()
                .position(|row| row.id == u64::from(id))
                .expect("a released row");
            real_row(index)
        },
        &wkv_codes,
        &wkv_scales,
        &product,
        shape,
    )
    .expect("no sentinel id");

    // The f32 seam: the same rows dequantised without the bf16 cast, the same weights dequantised
    // to f32, and the f32 head (engram_head_ref::head_ref).
    let f32_rows: Vec<Vec<f32>> = (0..REAL_ROWS.len()).map(real_row_f32).collect();
    let wkv_f32: Vec<f32> = (0..out_width * fetched_width)
        .map(|index| {
            let row = index / fetched_width;
            let column = index % fetched_width;
            fp8_e4m3_decode(wkv_codes[index])
                * ue8m0_decode(wkv_scales[(row / 32) * (fetched_width / 32) + column / 32])
        })
        .collect();
    let mut x_f32 = x_start.clone();
    let f32_gates = head_ref(
        &mut x_f32,
        &ids,
        &mut |id| {
            f32_rows[REAL_ROWS
                .iter()
                .position(|row| row.id == u64::from(id))
                .expect("a released row")]
            .clone()
        },
        &wkv_f32,
        &product,
        shape,
    )
    .expect("no sentinel id");

    // The declared differences only: the bf16 dequant cast, the fp8 activation quantisation, the
    // bf16 wkv output and the bf16 write. Record the measured deltas; assert they are small enough
    // that the f32 seam was the deviation the audit declared, not an unrelated arithmetic error.
    let mut gate_delta = 0.0f64;
    for (native, reference) in native_gates.iter().zip(&f32_gates) {
        gate_delta = gate_delta.max((f64::from(*native) - f64::from(*reference)).abs());
    }
    let mut out_delta = 0.0f64;
    let mut out_scale = 0.0f64;
    for (native, reference) in x_native.iter().zip(&x_f32) {
        out_delta = out_delta.max((f64::from(*native) - f64::from(*reference)).abs());
        out_scale = out_scale.max(f64::from(reference.abs()));
    }
    println!(
        "native vs f32 seam: max |gate delta| {gate_delta:.3e}, max |write delta| {out_delta:.3e}          (write magnitude {out_scale:.3e})"
    );
    assert!(
        gate_delta < 1e-3,
        "the gate moved by exactly the declared rule"
    );
    assert!(
        out_delta / out_scale.max(1e-30) < 1e-2,
        "the write moved by the declared rule's relative size, not by an unrelated arithmetic error"
    );

    // Every native value is bf16-exact, which is the rule the f32 seam lacks.
    for value in &x_native {
        assert_eq!(bf16_round(*value), *value, "the write is one bf16 cast");
    }
}

#[test]
fn a_sentinel_lookup_writes_nothing_under_the_native_rule() {
    let mut x_hc = vec![0.25f32; 2 * 64];
    let before = x_hc.clone();
    let ids = [1u32, u32::MAX];
    let shape = HeadShape {
        dim: 64,
        hc: 2,
        eps: 1e-6,
        head_dim: 256,
    };
    let result = head_native(
        &mut x_hc,
        &ids,
        &mut |_| (vec![0u8; 256], vec![127u8; 8]),
        &[0u8; 2 * (64 + 1) * 2 * 256],
        &[121u8; (2 * (64 + 1) / 32) * (2 * 256 / 32)],
        &[1.0f32; 2 * 64],
        shape,
    );
    assert!(
        result.is_none(),
        "a sentinel row takes no part in the n-gram"
    );
    assert_eq!(x_hc, before, "and the stream is untouched");
}

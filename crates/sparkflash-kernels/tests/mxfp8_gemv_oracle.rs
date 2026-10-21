//! The host-side oracle compare for the WS-D decode GEMV
//! (`kernels/b12x_mxfp8_gemv_m1_sm121.cu`).
//!
//! The kernel's identity claim rests on three things, and this file pins the two that are checkable
//! on a CPU host without a GPU:
//!
//! 1. the activation decode is the pinned `a_value`: for each 32-value MX block one ue8m0 decode,
//!    an exact E4M3 widening to f32, and one f32 multiply by that scale. [every_e4m3_code_survives_
//!    the_f16_round_trip] proves the widening the kernel uses (`cvt.rn.f16x2.e4m3x2` then
//!    `cvt.f32.f16`) is exact for every one of the 256 E4M3 codes, i.e. it produces the identical
//!    f32 the scalar `fp8_e4m3_decode` produces;
//! 2. the weights are the same per-value decode times the per-(tile, K-block) ue8m0 broadcast, with
//!    the same f32 rounding;
//! 3. the per-output chain is ascending K in 32-value blocks, ascending inside a block, one fused
//!    multiply-add per element, one bf16 round at the end.
//!
//! [the_gemv_tiling_is_the_pinned_arithmetic_on_every_served_shape] runs a model of that tiling -
//! one warp per 32-output B12x micro-tile, one lane per output column, the kernel's prefetch depth
//! and tail included as control flow only - and compares its bf16 words **bit for bit** against
//! [mxfp8_ref::dense_gemm], the CPU twin of the pinned `b12x_mxfp8_dense_sm121` symbol, on every
//! shape the decode step serves. [the_coverage_is_one_write_per_output_column] proves the tiling
//! writes each output column exactly once.
//!
//! Honest caveat, same as the M=1 oracle: the CPU twin forms each product as a multiply then an add
//! while the device contracts the same multiply-add into one `fma.rn.f32`, so this compares order,
//! layout, ownership and rounding count - not the FMA's rounding mode. Device-vs-device bit identity
//! is what the window's identity gate measures.

use sparkflash_kernels::m4_quant::{
    bf16_bits, fp8_e4m3_decode, fp8_e4m3_encode, ue8m0_code_of_power_of_two, ue8m0_decode,
};
use sparkflash_kernels::mxfp8_gemv_launch::{MAX_INNER, Mode, OUT_COLUMNS, PIPELINE};
use sparkflash_kernels::mxfp8_ref::{
    b12x_buffer_bytes, b12x_scale_bytes, b12x_scale_index, b12x_weight_index, dense_gemm,
};

/// One MX scale per 32 values along K.
const QUANT_BLOCK: usize = 32;

fn activation_row(inner: usize, salt: u32) -> (Vec<u8>, Vec<u8>) {
    let mut codes = Vec::with_capacity(inner);
    for k in 0..inner {
        let value = ((k as u32).wrapping_mul(37).wrapping_add(salt) % 251) as f32 * 0.017 - 2.0;
        codes.push(fp8_e4m3_encode(value));
    }
    let mut scales = Vec::with_capacity(inner / QUANT_BLOCK);
    for block in 0..inner / QUANT_BLOCK {
        let shift = (block as u32 % 5) as i32 - 2;
        scales.push(ue8m0_code_of_power_of_two(2.0f32.powi(shift)));
    }
    (codes, scales)
}

fn weight_buffer(cols: usize, inner: usize, salt: u32) -> (Vec<u8>, Vec<u8>) {
    let mut codes = vec![0u8; b12x_buffer_bytes(cols, inner)];
    let mut scales = vec![0u8; b12x_scale_bytes(cols, inner)];
    for col in 0..cols {
        for k in 0..inner {
            let value = ((col * 17 + k * 7 + salt as usize) % 97) as f32 * 0.043 - 1.5;
            codes[b12x_weight_index(col, k, inner)] = fp8_e4m3_encode(value);
        }
        for block in 0..inner / QUANT_BLOCK {
            let exponent = ((col + block) % 5) as i32 - 2;
            let scale = 2.0f32.powi(exponent);
            scales[b12x_scale_index(col, block * QUANT_BLOCK, inner)] =
                ue8m0_code_of_power_of_two(scale);
        }
    }
    (codes, scales)
}

/// The kernel's tiling and arithmetic, written the way the kernel writes it: one lane per output
/// column of a 32-column B12x micro-tile, K ascending in 32-value blocks and ascending inside the
/// block, one bf16 round. The `pipeline` argument is control flow only - it decides when the
/// prefetch loads of a lane's weight bytes are issued, never which order an output accumulates in -
/// so the model runs at both ends of the kernel's prefetch depth to make that claim checkable.
///
/// Returns the bf16 words and the per-column write count so the caller can prove the coverage.
#[allow(clippy::too_many_arguments)]
fn gemv_model(
    a: &[u8],
    a_scales: &[u8],
    w: &[u8],
    w_scales: &[u8],
    cols: usize,
    inner: usize,
    pipeline: usize,
    mode: Mode,
) -> (Vec<u16>, Vec<u32>) {
    let (accs, writes) = gemv_accumulators(a, a_scales, w, w_scales, cols, inner, pipeline, mode);
    (accs.into_iter().map(bf16_bits).collect(), writes)
}

/// The same tiling without the final bf16 round: the accumulators, so a test can see the f32
/// arithmetic the two modes differ in without the 8-bit mantissa hiding it.
#[allow(clippy::too_many_arguments)]
fn gemv_accumulators(
    a: &[u8],
    a_scales: &[u8],
    w: &[u8],
    w_scales: &[u8],
    cols: usize,
    inner: usize,
    pipeline: usize,
    mode: Mode,
) -> (Vec<f32>, Vec<u32>) {
    let k_blocks = inner / QUANT_BLOCK;
    let out_columns = OUT_COLUMNS as usize;
    let tiles = cols.div_ceil(out_columns);
    let mut out = vec![0.0f32; cols];
    let mut writes = vec![0u32; cols];
    // The activation row as the kernel's shared buffer holds it: decoded once, exact f32 values.
    let mut a_values = vec![0.0f32; inner];
    for k in 0..inner {
        a_values[k] = fp8_e4m3_decode(a[k]) * ue8m0_decode(a_scales[k / QUANT_BLOCK]);
    }
    for tile in 0..tiles {
        for lane in 0..out_columns {
            let col = tile * out_columns + lane;
            if col >= cols {
                continue;
            }
            let mut acc = 0.0f32;
            // The kernel prefetches `pipeline` blocks and drains the tail unpipelined; the model
            // walks the same block sequence, because the pipeline only changes when bytes arrive.
            let mut block = 0usize;
            while block + pipeline <= k_blocks {
                for p in 0..pipeline {
                    let current = block + p;
                    acc = model_block(
                        a_values.as_slice(),
                        w,
                        w_scales,
                        col,
                        current,
                        inner,
                        acc,
                        mode,
                    );
                }
                block += pipeline;
            }
            while block < k_blocks {
                acc = model_block(
                    a_values.as_slice(),
                    w,
                    w_scales,
                    col,
                    block,
                    inner,
                    acc,
                    mode,
                );
                block += 1;
            }
            out[col] = acc;
            writes[col] += 1;
        }
    }
    (out, writes)
}

/// One K block for one output column: the kernel's `spark_m4_gemv_block` order, 32 elements
/// ascending, each product formed by a rounded f32 multiply followed by a fused multiply-add.
#[allow(clippy::too_many_arguments)]
fn model_block(
    a_values: &[f32],
    w: &[u8],
    w_scales: &[u8],
    col: usize,
    block: usize,
    inner: usize,
    mut acc: f32,
    mode: Mode,
) -> f32 {
    let base = block * QUANT_BLOCK;
    let w_scale = ue8m0_decode(w_scales[b12x_scale_index(col, base, inner)]);
    for i in 0..QUANT_BLOCK {
        let w_value = fp8_e4m3_decode(w[b12x_weight_index(col, base + i, inner)]) * w_scale;
        if mode == Mode::Fused {
            acc = a_values[base + i].mul_add(w_value, acc);
        } else {
            // The engine twin of the non-fused modes: a rounded multiply, then a rounded add.
            acc += a_values[base + i] * w_value;
        }
    }
    acc
}

/// The shapes the decode step puts through the dense family.
const SERVED_SHAPES: [(&str, usize, usize); 6] = [
    ("attention projections wq_a", 1280, 5120),
    ("attention projections wkv", 512, 5120),
    ("wo_b", 5120, 256),
    ("wq_b", 8192, 1280),
    ("shared expert gate and up", 2304, 5120),
    ("shared expert down", 5120, 2304),
];

#[test]
fn the_gemv_tiling_is_the_pinned_arithmetic_on_every_served_shape() {
    for (name, cols, inner) in SERVED_SHAPES {
        assert!(
            inner <= MAX_INNER as usize,
            "{name}: the shape must fit the kernel's shared activation row"
        );
        let (a, a_scales) = activation_row(inner, 3);
        let (w, w_scales) = weight_buffer(cols, inner, 11);
        let oracle = dense_gemm(&a, &a_scales, &w, &w_scales, 1, cols, inner);
        // Prefetch depth is control flow, not order: the drained end and the deep end must agree.
        for pipeline in [1usize, PIPELINE as usize, 6] {
            let (model, writes) = gemv_model(
                &a,
                &a_scales,
                &w,
                &w_scales,
                cols,
                inner,
                pipeline,
                Mode::Fused,
            );
            assert_eq!(
                writes,
                vec![1u32; cols],
                "{name}, pipeline {pipeline}: every output column is written exactly once"
            );
            assert_eq!(
                model, oracle,
                "{name} ({cols}x{inner}), pipeline {pipeline}: the GEMV tiling must be the pinned arithmetic bit for bit"
            );
        }
    }
}

#[test]
fn the_coverage_is_one_write_per_output_column() {
    // A shape that is not a multiple of the 32-column tile: the last tile has inactive lanes, so the
    // column guard is exercised, and the reduction is not a multiple of the prefetch depth.
    let (cols, inner) = (1300usize, 128usize);
    let (a, a_scales) = activation_row(inner, 5);
    let (w, w_scales) = weight_buffer(cols, inner, 13);
    let oracle = dense_gemm(&a, &a_scales, &w, &w_scales, 1, cols, inner);
    let (model, writes) = gemv_model(&a, &a_scales, &w, &w_scales, cols, inner, 6, Mode::Fused);
    assert_eq!(
        writes,
        vec![1u32; cols],
        "the tail tile must write the real columns once and leave the padding columns alone"
    );
    assert_eq!(
        model, oracle,
        "the guarded tail must be the pinned arithmetic"
    );
}

/// A scalar ascending-K reference of the engine's accumulate: a rounded multiply, then a rounded
/// add, exactly as `coded_matmul` writes it. This is the arithmetic the non-fused modes are pinned
/// to, and it is `acc += a * w`, not a fused multiply-add.
fn scalar_non_fused(
    a_values: &[f32],
    w: &[u8],
    w_scales: &[u8],
    cols: usize,
    inner: usize,
) -> Vec<u16> {
    let k_blocks = inner / QUANT_BLOCK;
    (0..cols)
        .map(|col| {
            let mut acc = 0.0f32;
            for block in 0..k_blocks {
                let base = block * QUANT_BLOCK;
                let w_scale = ue8m0_decode(w_scales[b12x_scale_index(col, base, inner)]);
                for i in 0..QUANT_BLOCK {
                    let w_value =
                        fp8_e4m3_decode(w[b12x_weight_index(col, base + i, inner)]) * w_scale;
                    acc += a_values[base + i] * w_value;
                }
            }
            bf16_bits(acc)
        })
        .collect()
}

/// The coded activation row as the kernel's shared buffer holds it.
fn decoded_activation(a: &[u8], a_scales: &[u8], inner: usize) -> Vec<f32> {
    (0..inner)
        .map(|k| fp8_e4m3_decode(a[k]) * ue8m0_decode(a_scales[k / QUANT_BLOCK]))
        .collect()
}

#[test]
fn the_non_fused_mode_is_the_rounded_multiply_then_rounded_add() {
    for (name, cols, inner) in SERVED_SHAPES {
        let (a, a_scales) = activation_row(inner, 3);
        let (w, w_scales) = weight_buffer(cols, inner, 11);
        let a_values = decoded_activation(&a, &a_scales, inner);
        let oracle = scalar_non_fused(&a_values, &w, &w_scales, cols, inner);
        let (model, writes) = gemv_model(
            &a,
            &a_scales,
            &w,
            &w_scales,
            cols,
            inner,
            PIPELINE as usize,
            Mode::NonFused,
        );
        assert_eq!(
            writes,
            vec![1u32; cols],
            "{name}: one write per output column"
        );
        assert_eq!(
            model, oracle,
            "{name} ({cols}x{inner}): the non-fused mode must be the rounded multiply then rounded add, bit for bit"
        );
    }
}

/// Every product of two decoded E4M3 values is EXACT in f32: a 4-bit mantissa times a 4-bit
/// mantissa needs at most 8 significant bits, far inside f32's 24. That fact is what makes the
/// coded-activation modes interchangeable: when the multiply cannot round, a fused multiply-add
/// and a rounded multiply followed by a rounded add round exactly the same single value (the sum),
/// in the same order, so the words are identical. The mode exists for the f32-activation seam,
/// where the activation carries 24 mantissa bits and the product DOES round.
#[test]
fn every_e4m3_product_is_exact_in_f32() {
    for a in 0u16..256 {
        for b in 0u16..256 {
            let av = fp8_e4m3_decode(a as u8);
            let bv = fp8_e4m3_decode(b as u8);
            if av.is_nan() || bv.is_nan() {
                // The two E4M3 NaN codes never appear as trained weights; the property is about
                // the finite value set.
                continue;
            }
            let product = av * bv;
            assert_eq!(
                f64::from(product),
                f64::from(av) * f64::from(bv),
                "codes {a:#04x} x {b:#04x}: the f32 product must be exact"
            );
        }
    }
}

#[test]
fn the_coded_accumulate_modes_agree_on_every_served_shape() {
    // The shared expert's device twin can run either coded mode and stay byte-identical, because
    // the multiply never rounds on coded operands (every_e4m3_product_is_exact_in_f32). Comparing
    // the f32 accumulators, not just the bf16 words, keeps this a statement about the arithmetic
    // rather than about the output rounding.
    for (name, cols, inner) in SERVED_SHAPES {
        let (a, a_scales) = activation_row(inner, 3);
        let (w, w_scales) = weight_buffer(cols, inner, 11);
        let (fused, _) =
            gemv_accumulators(&a, &a_scales, &w, &w_scales, cols, inner, 6, Mode::Fused);
        let (non_fused, _) =
            gemv_accumulators(&a, &a_scales, &w, &w_scales, cols, inner, 6, Mode::NonFused);
        assert_eq!(
            fused, non_fused,
            "{name}: an E4M3-coded dot product must not depend on the accumulate contraction"
        );
    }
}

#[test]
fn the_f32_activation_mode_stages_the_row_unchanged() {
    // The compressor seam never quantises its activation: the row reaches the kernel as f32 and the
    // kernel copies it into shared memory unchanged. The model of that mode must equal the scalar
    // reference built from the same f32 row, and it must differ from the coded-activation mode,
    // because quantising the row would change the values.
    for (cols, inner) in [(512usize, 5120usize), (1300usize, 512usize)] {
        let (w, w_scales) = weight_buffer(cols, inner, 17);
        let a_row: Vec<f32> = (0..inner)
            .map(|k| ((k as u32).wrapping_mul(29) % 211) as f32 * 0.013 - 1.1)
            .collect();
        let oracle = scalar_non_fused(&a_row, &w, &w_scales, cols, inner);
        let modelled = f32_activation_model(&a_row, &w, &w_scales, cols, inner);
        assert_eq!(
            modelled, oracle,
            "{cols}x{inner}: the f32 activation mode must be the rounded multiply then rounded add on the raw row"
        );
        // Non-vacuous: a coded activation is a different value set, so the two staging paths differ.
        let (a, a_scales) = activation_row(inner, 3);
        let (coded, _) = gemv_model(&a, &a_scales, &w, &w_scales, cols, inner, 6, Mode::NonFused);
        assert_ne!(
            coded, modelled,
            "{cols}x{inner}: coded and f32 activation must differ"
        );
    }
}

/// The kernel's tiling with the f32-activation staging: the same warp tiles, the same ascending-K
/// chain, but the shared buffer holds the caller's f32 row instead of a decode of E4M3 codes.
fn f32_activation_model(
    a_row: &[f32],
    w: &[u8],
    w_scales: &[u8],
    cols: usize,
    inner: usize,
) -> Vec<u16> {
    let k_blocks = inner / QUANT_BLOCK;
    let out_columns = OUT_COLUMNS as usize;
    let tiles = cols.div_ceil(out_columns);
    let mut out = vec![0u16; cols];
    for tile in 0..tiles {
        for lane in 0..out_columns {
            let col = tile * out_columns + lane;
            if col >= cols {
                continue;
            }
            let mut acc = 0.0f32;
            for block in 0..k_blocks {
                let base = block * QUANT_BLOCK;
                let w_scale = ue8m0_decode(w_scales[b12x_scale_index(col, base, inner)]);
                for i in 0..QUANT_BLOCK {
                    let w_value =
                        fp8_e4m3_decode(w[b12x_weight_index(col, base + i, inner)]) * w_scale;
                    acc += a_row[base + i] * w_value;
                }
            }
            out[col] = bf16_bits(acc);
        }
    }
    out
}
/// The exact f32 -> f16 -> f32 round trip for a value that f16 can represent.
fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        // Inf or NaN: f16 carries both.
        return sign | 0x7c00 | if mantissa != 0 { 0x0200 } else { 0 };
    }
    let unbiased = exponent - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased < -14 {
        // Subnormal: shift the implicit 1 in and round to the 10-bit field.
        let shift = (-14 - unbiased) as u32;
        if shift > 24 {
            return sign;
        }
        let full = mantissa | 0x80_0000;
        let scaled = full >> (13 + shift);
        let remainder = full & ((1u32 << (13 + shift)) - 1);
        let round = u32::from(remainder > (1u32 << (12 + shift)));
        return sign | (scaled as u16 + round as u16);
    }
    let field = (unbiased + 15) as u32;
    let half_mantissa = mantissa >> 13;
    let round = u32::from((mantissa & 0x1fff) > 0x1000);
    sign | ((field as u16) << 10) | ((half_mantissa + round) as u16)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x3ff);
    if exponent == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13));
    }
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal f16: normalise it into an f32.
        let mut m = mantissa;
        let mut e = -14i32;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        let field = (e + 127) as u32;
        return f32::from_bits(sign | (field << 23) | ((m & 0x3ff) << 13));
    }
    f32::from_bits(sign | (((exponent - 15 + 127) as u32) << 23) | (mantissa << 13))
}

/// Claim (a): the kernel widens every E4M3 code through f16, and that widening is exact, so the
/// value it multiplies by the ue8m0 scale is bit for bit the scalar decoder's value.
#[test]
fn every_e4m3_code_survives_the_f16_round_trip() {
    for code in 0u16..=0xff {
        let decoded = fp8_e4m3_decode(code as u8);
        if decoded.is_nan() {
            // The encoder never emits the two NaN codes; they carry no value to compare.
            continue;
        }
        let round_tripped = f16_to_f32(f16_bits(decoded));
        assert_eq!(
            round_tripped.to_bits(),
            decoded.to_bits(),
            "E4M3 code {code:#04x} ({decoded}) must survive the f16 widening exactly"
        );
    }
}

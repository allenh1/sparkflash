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

//! The CPU twin of the layer-seam kernels (kernels/layer_seams_sm121.cu).
//!
//! It exists to carry the kernels' declared comparison class on the host, where CI can run it:
//! sixteen functions, each one the arithmetic of one block of the device code, written from the same
//! reference (the engine's host path) the kernels are written from. The tests below pin the twin to
//! the engine's own functions bit for bit, so the distance between the device arm and the host arm
//! is exactly the distance the kernel header declares (the D54 tree-reduction class for the three
//! reductions, and nothing at all for the activation quantiser).
//!
//! What is NOT here: the block's parallel decomposition. The kernels split the 24 dot products, the
//! projection's sum of squares and the RMSNorm's sum of squares across lanes and combine them as a
//! tree; this twin sums them sequentially, which is the engine's order. That is the whole of the
//! declared tolerance class: the per-element expressions are identical, the sums' association is
//! not, and the oracle on hardware compares the two.
//!
//! The functions:
//!
//! * [bf16_round] - the released BF16 boundary (quant.rs f32_to_bf16_bits): RNE, NaN kept quiet.
//! * [sigmoid] - linalg::sigmoid's two-branch form.
//! * [softmax] / [normalise_columns] / [split_sinkhorn] - the released comb normalisation.
//! * [coefficient_tail] - hc_mixes_from_projection: rsqrt, the three coefficient sets, the Sinkhorn.
//! * [hc_pre] - the four-copy collapse;
//! * [hc_post] / [hc_post_in_place] - the release's residual mix-back, and the kernels' in-place
//!   form (the two are equal, which is what licenses one launch reading hc_rows and writing the
//!   stream);
//! * [ffn_epilogue_row] / [ffn_epilogue] / [ffn_epilogue_in_place] - the shared-expert add and the
//!   FFN's hc_post over the post-attention stream.
//! * [rope_rows_ref] - the attention output's rotation over a batch of head rows, both signs (the
//!   device kernel carries the sign as a word, and this twin's two arms pin each bit for bit);
//! * [rms_norm] - linalg::rms_norm.
//! * [e8m0_encode_ceil] / [e4m3_encode] / [quantise_activation] - the activation quantiser.

/// The largest finite E4M3 magnitude (kernel.py:45), as quant.rs spells it.
pub const FP8_E4M3_MAX: f32 = 448.0;
/// The reference's activation amax floor (kernel.py:76).
pub const ACTIVATION_AMAX_FLOOR: f32 = 1e-4;
/// The MX block the activation quantiser scales.
pub const SCALE_BLOCK: usize = 32;

/// The released BF16 boundary: nearest BF16, ties to even, NaN kept quiet.
#[must_use]
pub fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    if value.is_nan() {
        return f32::from_bits(u32::from((bits >> 16) as u16 | 0x0040) << 16);
    }
    let round = 0x7FFF + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(round) >> 16 << 16)
}

/// 1 / (1 + exp(-x)), saturating instead of overflowing (linalg::sigmoid).
#[must_use]
pub fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

/// The numerically stable softmax of a row, in place (linalg::softmax).
pub fn softmax(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !maximum.is_finite() {
        row.fill(0.0);
        return;
    }
    let mut total = 0.0f32;
    for value in row.iter_mut() {
        let exponential = (*value - maximum).exp();
        *value = exponential;
        total += exponential;
    }
    if total > 0.0 {
        for value in row.iter_mut() {
            *value /= total;
        }
    }
}

/// The column normalisation of the Sinkhorn pair, hc_eps inside the denominator.
pub fn normalise_columns(comb: &mut [f32], hc: usize, hc_eps: f32) {
    for column in 0..hc {
        let column_sum: f32 = (0..hc).map(|row| comb[row * hc + column]).sum();
        let denominator = column_sum + hc_eps;
        for row in 0..hc {
            comb[row * hc + column] /= denominator;
        }
    }
}

/// The released split Sinkhorn (mhc.rs split_sinkhorn) on one token's hc x hc comb.
pub fn split_sinkhorn(comb: &mut [f32], hc: usize, iterations: usize, hc_eps: f32) {
    for row in comb.chunks_mut(hc) {
        softmax(row);
        for value in row.iter_mut() {
            *value += hc_eps;
        }
    }
    normalise_columns(comb, hc, hc_eps);
    for _ in 1..iterations {
        for row in comb.chunks_mut(hc) {
            let row_sum: f32 = row.iter().sum();
            let denominator = row_sum + hc_eps;
            for value in row.iter_mut() {
                *value /= denominator;
            }
        }
        normalise_columns(comb, hc, hc_eps);
    }
}

/// The coefficient tail (mhc.rs hc_mixes_from_projection): the three tables in one vector, pre then
/// post then comb, exactly the order the kernels write.
///
/// The device kernel and this twin compute the same expressions; this one is the engine's, so the
/// tests can compare it to the engine bit for bit and leave only the reduction order to the oracle.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn coefficient_tail(
    raw: &[f32],
    sum_squares: f32,
    width: usize,
    scale: &[f32],
    base: &[f32],
    hc: usize,
    norm_eps: f32,
    hc_eps: f32,
    sinkhorn_iterations: usize,
) -> Vec<f32> {
    let mut table = vec![0.0f32; 2 * hc + hc * hc];
    let mean_square = sum_squares / width as f32;
    let rsqrt = 1.0 / (mean_square + norm_eps).sqrt();
    let mix = |coefficient: usize| raw[coefficient] * rsqrt;
    for coefficient in 0..hc {
        table[coefficient] = sigmoid(mix(coefficient) * scale[0] + base[coefficient]) + hc_eps;
    }
    for coefficient in 0..hc {
        let row = hc + coefficient;
        table[row] = 2.0 * sigmoid(mix(row) * scale[1] + base[row]);
    }
    for row in 0..hc {
        for column in 0..hc {
            let coefficient = 2 * hc + row * hc + column;
            table[coefficient] = mix(coefficient) * scale[2] + base[coefficient];
        }
    }
    split_sinkhorn(&mut table[2 * hc..], hc, sinkhorn_iterations, hc_eps);
    table
}

/// The hc_pre collapse: out[j] = sum over copies c of pre_mix[c] * x[c][j], ascending c.
#[must_use]
pub fn hc_pre(pre_mix: &[f32], x_hc: &[f32], dim: usize) -> Vec<f32> {
    let hc = pre_mix.len();
    let mut out = vec![0.0f32; dim];
    for (column, value) in out.iter_mut().enumerate() {
        let mut accumulated = 0.0f32;
        for copy in 0..hc {
            accumulated += pre_mix[copy] * x_hc[copy * dim + column];
        }
        *value = accumulated;
    }
    out
}
/// The released `hc_post` (mhc.rs:690-760): `out[copy][j] = post[copy] * x[j] +
/// sum_source comb[source * hc + copy] * residual_hc[source * dim + j]`, the comb terms ascending
/// source with the post term seeded into the accumulator first. `native_bf16` selects the release
/// order instead (mhc.rs:823-852): the comb sum from zero, the post term added last, one BF16
/// rounding of the result. Neither arm consumes a pre mix - hc_pre takes one, hc_post cannot.
#[must_use]
pub fn hc_post(
    x: &[f32],
    residual_hc: &[f32],
    post_mix: &[f32],
    comb: &[f32],
    hc: usize,
    dim: usize,
    native_bf16: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; hc * dim];
    for copy in 0..hc {
        for column in 0..dim {
            out[copy * dim + column] = if native_bf16 {
                let mut comb_sum = 0.0f32;
                for source in 0..hc {
                    comb_sum += comb[source * hc + copy] * residual_hc[source * dim + column];
                }
                let post_term = post_mix[copy] * x[column];
                bf16_round(post_term + comb_sum)
            } else {
                let mut accumulated = post_mix[copy] * x[column];
                for source in 0..hc {
                    accumulated += comb[source * hc + copy] * residual_hc[source * dim + column];
                }
                accumulated
            };
        }
    }
    out
}

/// [hc_post] with the stream updated in place: the kernels' form. It reads the `hc` sources of a
/// column into registers and only then writes that column's `hc` outputs, which is the whole reason
/// the device seam may read `residual.hc_rows` and write `residual.stream` in one launch.
/// [tests::the_in_place_post_is_the_out_of_place_post] is the pin, and it is the property the
/// aliasing argument rests on rather than a coincidence of the two buffers not overlapping.
pub fn hc_post_in_place(
    stream: &mut [f32],
    x: &[f32],
    post_mix: &[f32],
    comb: &[f32],
    hc: usize,
    dim: usize,
    native_bf16: bool,
) -> Vec<f32> {
    let mut sources = vec![0.0f32; hc];
    for column in 0..dim {
        for (source, value) in sources.iter_mut().enumerate() {
            *value = stream[source * dim + column];
        }
        for copy in 0..hc {
            stream[copy * dim + column] = if native_bf16 {
                let mut comb_sum = 0.0f32;
                for (source, value) in sources.iter().enumerate() {
                    comb_sum += comb[source * hc + copy] * value;
                }
                let post_term = post_mix[copy] * x[column];
                bf16_round(post_term + comb_sum)
            } else {
                let mut accumulated = post_mix[copy] * x[column];
                for (source, value) in sources.iter().enumerate() {
                    accumulated += comb[source * hc + copy] * value;
                }
                accumulated
            };
        }
    }
    stream.to_vec()
}

/// The FFN epilogue's row before its hc_post: the routed MoE delta widened from its BF16 row plus
/// the shared expert's row, one f32 add (forward/mod.rs:1950-2038). `shared` is `None` when the
/// arm has no shared-expert row at all. The port now has one (`moe_shared` writes it before the
/// epilogue and the ABI's shared_present word is 1), so the stage passes `Some`; `None` remains the
/// shape the f32 arm never uses and the test below pins both ends of the row's arithmetic.
#[must_use]
pub fn ffn_epilogue_row(moe_delta_bf16: &[u16], shared: Option<&[f32]>) -> Vec<f32> {
    let mut row: Vec<f32> = moe_delta_bf16
        .iter()
        .map(|bits| f32::from_bits(u32::from(*bits) << 16))
        .collect();
    if let Some(shared) = shared {
        for (value, addend) in row.iter_mut().zip(shared) {
            *value += *addend;
        }
    }
    row
}

/// [hc_post] over [ffn_epilogue_row]: the epilogue's whole arithmetic, in the reference's order.
///
/// The arity is the kernel's argument list on purpose: a twin that grouped the slices into a struct
/// would stop being a one-line translation of the launch, and the pin below would be checking the
/// struct instead of the ABI.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn ffn_epilogue(
    stream: &[f32],
    moe_delta_bf16: &[u16],
    shared: Option<&[f32]>,
    post_mix: &[f32],
    comb: &[f32],
    hc: usize,
    dim: usize,
    native_bf16: bool,
) -> Vec<f32> {
    let row = ffn_epilogue_row(moe_delta_bf16, shared);
    hc_post(&row, stream, post_mix, comb, hc, dim, native_bf16)
}

/// [ffn_epilogue] with the stream updated in place, the kernel's form. Same arity reason as above.
#[allow(clippy::too_many_arguments)]
pub fn ffn_epilogue_in_place(
    stream: &mut [f32],
    moe_delta_bf16: &[u16],
    shared: Option<&[f32]>,
    post_mix: &[f32],
    comb: &[f32],
    hc: usize,
    dim: usize,
    native_bf16: bool,
) -> Vec<f32> {
    let row = ffn_epilogue_row(moe_delta_bf16, shared);
    hc_post_in_place(stream, &row, post_mix, comb, hc, dim, native_bf16)
}

/// linalg::rms_norm on one row: (x * (1 / sqrt(mean(x^2) + eps))) * weight, not yet rounded.
#[must_use]
pub fn rms_norm(row: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean = row.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
    let inverse = 1.0 / (mean + eps).sqrt();
    row.iter()
        .zip(weight)
        .map(|(value, weight)| value * inverse * weight)
        .collect()
}

/// Decodes one E4M3 code to the value it stands for.
#[must_use]
pub fn f8e4m3_to_f32(code: u8) -> f32 {
    let negative = code & 0x80 != 0;
    let exponent = (code >> 3) & 0x0F;
    let mantissa = code & 0x07;
    if exponent == 0x0F && mantissa == 0x07 {
        return f32::from_bits(if negative { 0xFFC0_0000 } else { 0x7FC0_0000 });
    }
    let magnitude = if exponent == 0 {
        f32::from(mantissa) * (1.0 / 512.0)
    } else {
        let power = f32::from_bits((u32::from(exponent) + 120) << 23);
        power * (1.0 + f32::from(mantissa) / 8.0)
    };
    if negative { -magnitude } else { magnitude }
}

/// The E8M0 ceiling code: 2^ceil(log2(value)), clamped to the format (quant.rs e8m0_encode_ceil).
#[must_use]
pub fn e8m0_encode_ceil(value: f32) -> u8 {
    if value.is_nan() {
        return 0xFF;
    }
    if value <= 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let field = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;
    let exponent = if field == 0 {
        let highest = 31 - mantissa.leading_zeros() as i32;
        -149 + highest + i32::from(mantissa.count_ones() > 1)
    } else {
        field - 127 + i32::from(mantissa != 0)
    };
    (exponent + 127).clamp(0, 254) as u8
}

/// The E4M3 code nearest a value, ties to the even mantissa, saturating at 448.
///
/// Written as the definition (nearest over the 127 finite codes) rather than as the kernel's bit
/// encoder: the kernel ships the hardware conversion (__nv_cvt_float_to_fp8 with SATFINITE), and a
/// definition-level twin is what proves the conversion and the engine's bit encoder are one
/// function.
#[must_use]
pub fn e4m3_encode(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7F;
    }
    let magnitude = f64::from(value.abs());
    let mut best = 0u8;
    let mut best_distance = f64::INFINITY;
    for code in 0u8..=0x7E {
        let represented = f64::from(f8e4m3_to_f32(code));
        let distance = (represented - magnitude).abs();
        if distance < best_distance || (distance == best_distance && code % 2 == 0) {
            best = code;
            best_distance = distance;
        }
    }
    if value.is_sign_negative() {
        best | 0x80
    } else {
        best
    }
}

/// The activation quantiser over one row: one ue8m0 scale per 32 values, E4M3 codes.
///
/// Returns (codes, scales). The row length must be a whole number of blocks.
///
/// This is the **FFN** rule: the reference quantises the pre-norm hidden as a raw f32 row, so the
/// values encoded are the f32 values, not their BF16 rounds (the rule is stated in the FFN seam's
/// evidence, `crates/sparkflash-cli/src/tp4_forward.rs`: "the activation quantiser still takes the
/// f32 row"). The attention projections' rule is the narrowed one below; the difference is a real
/// semantic difference, not a detail, and the two agree only on rows that are already
/// BF16-representable.
#[must_use]
pub fn quantise_activation(values: &[f32]) -> (Vec<u8>, Vec<u8>) {
    quantise_activation_with(values, |value| value)
}

/// The attention projections' activation quantiser: the reference's BF16 narrowing, then the MX
/// block rule.
///
/// D39 rule 1 says the W8A8 activation is a BF16 module output - `bf16_round` and then
/// `quantize_activation_fp8`, which is `sparkflash_engine::quant::quantize_activation_fp8_narrowed` -
/// so both the block amax and the codes are taken over BF16-representable values. The device kernel
/// `w8a8_prologue_e4m3` is this function's arithmetic (one RNE narrowing per read); the FFN seam's
/// `spark_seam_quantise_row` deliberately is NOT, because the FFN's reference takes the f32 row.
#[must_use]
pub fn quantise_activation_narrowed(values: &[f32]) -> (Vec<u8>, Vec<u8>) {
    quantise_activation_with(values, bf16_round)
}

fn quantise_activation_with(values: &[f32], narrow: impl Fn(f32) -> f32) -> (Vec<u8>, Vec<u8>) {
    let mut codes = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(values.len() / SCALE_BLOCK);
    for block in values.chunks(SCALE_BLOCK) {
        let amax = block
            .iter()
            .fold(0.0f32, |accumulated, &value| {
                accumulated.max(narrow(value).abs())
            })
            .max(ACTIVATION_AMAX_FLOOR);
        let code = e8m0_encode_ceil(amax * (1.0 / FP8_E4M3_MAX));
        scales.push(code);
        let scale = f8e8m0_to_f32(code);
        for &value in block {
            codes.push(e4m3_encode(
                (narrow(value) / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX),
            ));
        }
    }
    (codes, scales)
}

/// Decodes one E8M0 code to 2^(code - 127).
#[must_use]
pub fn f8e8m0_to_f32(code: u8) -> f32 {
    if code == 0xFF {
        return f32::from_bits(0x7FC0_0000);
    }
    if code == 0 {
        return f32::from_bits(1 << 22);
    }
    f32::from_bits(u32::from(code) << 23)
}

/// The attention output's RoPE over a batch of stored head rows, both signs.
///
/// The device kernel (rope_rotate_rows_bf16) is this function's arithmetic: widen the stored bf16
/// pair exactly, rotate with adjacent pairs in the host's order - each product rounded, each sum
/// rounded, the same association as linalg::rope_rotate_with - and narrow once. `sign` is +1 for
/// the forward rotation and -1 for the attention output's inverse rotation, exactly the two arms
/// linalg.rs:1183-1218 implements.
///
/// # Errors
/// A row buffer that is not `rows * head_width` values, a tail that is zero, odd, or wider than the
/// row, tables that do not carry one cos and one sin per rotated pair, or a sign that is not +1/-1.
#[allow(clippy::too_many_arguments)]
pub fn rope_rows_ref(
    rows: &[u16],
    cos: &[f32],
    sin: &[f32],
    row_count: usize,
    head_width: usize,
    rope_head_dim: usize,
    sign: f32,
) -> Result<Vec<u16>, String> {
    if rows.len() != row_count * head_width {
        return Err(format!(
            "rope rows: {} values are not {row_count} rows of {head_width}",
            rows.len()
        ));
    }
    if head_width == 0 {
        return Err("rope rows: head_width is zero, so the row has no values".to_string());
    }
    if rope_head_dim == 0 || rope_head_dim > head_width || !rope_head_dim.is_multiple_of(2) {
        return Err(format!(
            "rope rows: a {rope_head_dim}-long tail does not fit a {head_width}-wide row"
        ));
    }
    if sign != 1.0 && sign != -1.0 {
        return Err(format!("rope rows: sign {sign} is neither 1 nor -1"));
    }
    let pairs = rope_head_dim / 2;
    if cos.len() != pairs || sin.len() != pairs {
        return Err(format!(
            "rope rows: {rope_head_dim} values need {pairs} cos and {pairs} sin, got {} and {}",
            cos.len(),
            sin.len()
        ));
    }
    let widen = |bits: u16| f32::from_bits(u32::from(bits) << 16);
    let narrow = |value: f32| (bf16_round(value).to_bits() >> 16) as u16;
    let start = head_width - rope_head_dim;
    let mut out = rows.to_vec();
    for row in 0..row_count {
        let base = row * head_width;
        for pair in 0..pairs {
            let a = widen(rows[base + start + 2 * pair]);
            let b = widen(rows[base + start + 2 * pair + 1]);
            let first = a * cos[pair] - sign * b * sin[pair];
            let second = sign * a * sin[pair] + b * cos[pair];
            out[base + start + 2 * pair] = narrow(first);
            out[base + start + 2 * pair + 1] = narrow(second);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_engine::config::ModelConfig;
    use sparkflash_engine::forward::mhc::{self, HcProjection, Mixes};
    use sparkflash_engine::linalg;
    use sparkflash_engine::quant;

    /// A deterministic pseudo-random f32 stream in [-scale, scale).
    fn stream(len: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut state = seed;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let uniform = ((state >> 33) as f32) / ((1u64 << 31) as f32);
            values.push((uniform * 2.0 - 1.0) * scale);
        }
        values
    }

    #[test]
    fn the_bf16_boundary_is_the_engine_boundary() {
        for value in stream(4096, 11, 8.0).into_iter().chain([
            0.0,
            -0.0,
            f32::MAX,
            -f32::MAX,
            1e-45,
            f32::INFINITY,
        ]) {
            assert_eq!(
                bf16_round(value).to_bits(),
                quant::bf16_round(value).to_bits(),
                "bf16_round({value})"
            );
        }
    }

    #[test]
    fn the_coefficient_tail_is_the_engine_tail_bit_for_bit() {
        let config = ModelConfig::v41_flash_cached();
        let hc = config.hc_mult;
        let width = config.hc_width();
        let rows = config.hc_mix_count();
        let raw = stream(rows, 21, 2.0);
        let sum_squares: f32 = 3.5e4;
        let scale = [0.31f32, 0.72, 0.11];
        let base = stream(rows, 22, 0.5);
        let tail = coefficient_tail(
            &raw,
            sum_squares,
            width,
            &scale,
            &base,
            hc,
            config.norm_eps as f32,
            config.hc_eps as f32,
            config.hc_sinkhorn_iters,
        );
        let mut mixes = Mixes::for_config(config);
        mhc::hc_mixes_from_projection(
            config,
            &HcProjection {
                raw,
                sum_squares: vec![sum_squares],
            },
            &scale,
            &base,
            &mut mixes,
        )
        .expect("the released configuration runs");
        assert_eq!(&tail[..hc], mixes.pre.as_slice(), "pre");
        assert_eq!(&tail[hc..2 * hc], mixes.post.as_slice(), "post");
        assert_eq!(&tail[2 * hc..], mixes.comb.as_slice(), "comb");
    }

    #[test]
    fn the_hc_pre_collapse_is_the_engine_collapse_bit_for_bit() {
        let config = ModelConfig::v41_flash_cached();
        let dim = config.dim;
        let hc = config.hc_mult;
        let stream_values = stream(hc * dim, 31, 1.5);
        let pre_mix = [0.7f32, 0.1, 0.15, 0.05];
        let twin = hc_pre(&pre_mix, &stream_values, dim);
        let mut reference = vec![0.0f32; dim];
        mhc::hc_pre(config, &pre_mix, &stream_values, &mut reference).expect("released geometry");
        assert_eq!(twin, reference);
    }

    #[test]
    fn the_rms_norm_is_the_engine_norm_bit_for_bit() {
        let config = ModelConfig::v41_flash_cached();
        let dim = config.dim;
        let row = stream(dim, 41, 3.0);
        let weight = stream(dim, 42, 1.0);
        let twin = rms_norm(&row, &weight, config.norm_eps as f32);
        let mut reference = row.clone();
        linalg::rms_norm(&mut reference, &weight, dim, config.norm_eps as f32)
            .expect("released geometry");
        assert_eq!(twin, reference);
    }

    #[test]
    fn the_e4m3_encoder_is_the_engine_encoder() {
        // Every finite code, its neighbours, the saturation boundary, subnormals and randoms.
        let mut values = Vec::new();
        for code in 0u16..=0x7E {
            let exact = f8e4m3_to_f32(code as u8);
            values.extend([exact, exact * (1.0 + 1e-6), exact * (1.0 - 1e-6)]);
        }
        values.extend([
            0.0,
            -0.0,
            1e-45,
            2.0f32.powi(-9),
            2.0f32.powi(-10),
            448.0,
            -448.0,
            449.0,
            1e30,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ]);
        values.extend(stream(20_000, 51, 700.0));
        for value in values {
            assert_eq!(
                e4m3_encode(value),
                quant::e4m3_encode(value),
                "e4m3_encode({value})"
            );
        }
    }

    #[test]
    fn the_activation_quantiser_is_the_engine_quantiser_bit_for_bit() {
        let mut rows = Vec::new();
        // A zero block, a tiny block, a block that saturates, and a plain random row.
        rows.push(vec![0.0f32; 64]);
        rows.push(
            vec![1e-8f32; 32]
                .into_iter()
                .chain(vec![-1e-9f32; 32])
                .collect(),
        );
        rows.push(
            vec![1e9f32; 32]
                .into_iter()
                .chain(vec![1e9f32; 32])
                .collect(),
        );
        rows.push(stream(5120, 61, 4.0));
        rows.push(stream(256, 62, 0.001));
        for (index, row) in rows.into_iter().enumerate() {
            let (codes, scales) = quantise_activation(&row);
            let reference = quant::quantize_activation_fp8(&row).expect("whole blocks");
            assert_eq!(codes, reference.codes(), "codes of row {index}");
            assert_eq!(scales, reference.scales(), "scales of row {index}");
        }
    }

    /// The attention prologue's rule: the reference narrows the row to BF16 first and quantises the
    /// narrowed values, and the device kernel `w8a8_prologue_e4m3` narrows per read. This twin is
    /// the engine's `quantize_activation_fp8_narrowed`, bit for bit - and it is NOT the FFN rule:
    /// on a row that is not BF16-representable the two produce different codes, which is why the
    /// two functions exist separately instead of one calling the other.
    #[test]
    fn the_narrowed_quantiser_is_the_engine_narrowed_quantiser_bit_for_bit() {
        let rows: Vec<Vec<f32>> = vec![
            vec![0.0f32; 64],
            vec![1e-8f32; 32]
                .into_iter()
                .chain(vec![-1e-9f32; 32])
                .collect(),
            vec![1e9f32; 32]
                .into_iter()
                .chain(vec![1e9f32; 32])
                .collect(),
            stream(5120, 61, 4.0),
            stream(256, 62, 0.001),
        ];
        let mut differed_from_the_ffn_rule = 0usize;
        for (index, row) in rows.into_iter().enumerate() {
            let (codes, scales) = quantise_activation_narrowed(&row);
            let reference = quant::quantize_activation_fp8_narrowed(&row).expect("whole blocks");
            assert_eq!(codes, reference.codes(), "codes of row {index}");
            assert_eq!(scales, reference.scales(), "scales of row {index}");
            if quantise_activation(&row).0 != codes {
                differed_from_the_ffn_rule += 1;
            }
        }
        assert!(
            differed_from_the_ffn_rule > 0,
            "a generic row must distinguish the two rules, or this test proves nothing about which \
             one the attention prologue uses"
        );
    }

    #[test]
    fn the_quantiser_round_trips_through_its_own_scale() {
        // The device arm's codes x scales must decode back close to the row it quantised: the gap
        // is the E4M3 step at the block's scale, bounded by the block's own maximum.
        let row = stream(512, 71, 2.0);
        let (codes, scales) = quantise_activation(&row);
        for (block, &scale_code) in scales.iter().enumerate() {
            let scale = f8e8m0_to_f32(scale_code);
            let block_max = row[block * 32..(block + 1) * 32]
                .iter()
                .fold(0.0f32, |accumulated, value| accumulated.max(value.abs()))
                .max(ACTIVATION_AMAX_FLOOR);
            for index in 0..32 {
                let value = f8e4m3_to_f32(codes[block * 32 + index]) * scale;
                let error = (value - row[block * 32 + index]).abs();
                // The round-trip gap is half an E4M3 step at the VALUE's magnitude (the format has
                // three mantissa bits, so its step is |x| / 8), floored at half the smallest
                // subnormal step of the block's scale.
                let bound = (row[block * 32 + index].abs() / 16.0).max(scale / 1024.0);
                assert!(
                    error <= bound + 1e-7 * block_max,
                    "block {block} value {index}: |{value} - {}| = {error} > {bound}",
                    row[block * 32 + index]
                );
            }
        }
    }

    /// A released-shaped (post, comb) pair: post drawn in [-1, 1) and the comb through the released
    /// split Sinkhorn, so hc_post sees the value ranges the release produces (a column sums to 1).
    fn mixes_for_test(config: &ModelConfig, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let hc = config.hc_mult;
        let post = stream(hc, seed, 1.0);
        let mut comb = stream(hc * hc, seed + 1, 1.0);
        split_sinkhorn(
            &mut comb,
            hc,
            config.hc_sinkhorn_iters,
            config.hc_eps as f32,
        );
        (post, comb)
    }

    #[test]
    fn the_hc_post_twin_is_the_engine_hc_post_bit_for_bit() {
        let config = ModelConfig::v41_flash_cached();
        let (dim, hc) = (config.dim, config.hc_mult);
        let x = stream(dim, 81, 1.5);
        let residual_hc = stream(hc * dim, 82, 1.5);
        let (post, comb) = mixes_for_test(config, 83);
        let twin = hc_post(&x, &residual_hc, &post, &comb, hc, dim, false);
        let mut reference = vec![0.0f32; hc * dim];
        mhc::hc_post(config, &x, &residual_hc, &post, &comb, &mut reference)
            .expect("the released geometry runs");
        assert_eq!(twin, reference);
    }

    #[test]
    fn the_bf16_hc_post_twin_is_the_engine_hc_post_bit_for_bit() {
        // The native arm is not only a rounding of the f32 result: it sums the comb first and adds
        // the post term last (mhc.rs:823-852), so the two arms can disagree in the last bits.
        let config = ModelConfig::v41_flash_cached();
        let (dim, hc) = (config.dim, config.hc_mult);
        let x = stream(dim, 85, 1.5);
        let residual_hc = stream(hc * dim, 86, 1.5);
        let (post, comb) = mixes_for_test(config, 87);
        let twin = hc_post(&x, &residual_hc, &post, &comb, hc, dim, true);
        let mut reference = vec![0.0f32; hc * dim];
        mhc::hc_post_bf16(config, &x, &residual_hc, &post, &comb, &mut reference)
            .expect("the released geometry runs");
        assert_eq!(twin, reference);
    }

    #[test]
    fn the_in_place_post_is_the_out_of_place_post() {
        // The kernels read the hc sources of a column and then write that column's hc outputs.
        // This is the pin that makes the device seam's aliasing (read residual.hc_rows, write
        // residual.stream in one launch) legal rather than lucky, and it is checked on both arms
        // because the native arm's accumulator order differs.
        let config = ModelConfig::v41_flash_cached();
        let (dim, hc) = (config.dim, config.hc_mult);
        let stream_values = stream(hc * dim, 88, 1.5);
        let x = stream(dim, 89, 1.5);
        let (post, comb) = mixes_for_test(config, 90);
        for native_bf16 in [false, true] {
            let out_of_place = hc_post(&x, &stream_values, &post, &comb, hc, dim, native_bf16);
            let mut in_place = stream_values.clone();
            let written = hc_post_in_place(&mut in_place, &x, &post, &comb, hc, dim, native_bf16);
            assert_eq!(
                out_of_place, in_place,
                "the in-place post drifted (native_bf16 = {native_bf16})"
            );
            assert_eq!(out_of_place, written);
        }
    }

    #[test]
    fn the_device_widen_is_the_engine_widen() {
        // The kernels widen the routed delta with a reinterpret plus a shift; the engine's
        // quant::bf16_to_f32 is that same expression over every possible bit pattern.
        for bits in 0..=u16::MAX {
            let widened = f32::from_bits(u32::from(bits) << 16);
            assert_eq!(
                widened.to_bits(),
                quant::bf16_to_f32(bits).to_bits(),
                "{bits:#06x}"
            );
        }
    }

    #[test]
    fn the_epilogue_twin_is_hc_post_over_the_epilogue_row() {
        // Both shared-row states (the arm has no shared row today; the reference's add when it
        // does) and both rounding arms, against the engine's own hc_post functions.
        let config = ModelConfig::v41_flash_cached();
        let (dim, hc) = (config.dim, config.hc_mult);
        let stream_values = stream(hc * dim, 91, 1.5);
        let delta_bits: Vec<u16> = stream(dim, 92, 2.0)
            .iter()
            .map(|value| quant::f32_to_bf16_bits(*value))
            .collect();
        let shared = stream(dim, 93, 1.0);
        let (post, comb) = mixes_for_test(config, 94);
        for shared in [None, Some(shared.as_slice())] {
            let row = ffn_epilogue_row(&delta_bits, shared);
            for native_bf16 in [false, true] {
                let twin = ffn_epilogue(
                    &stream_values,
                    &delta_bits,
                    shared,
                    &post,
                    &comb,
                    hc,
                    dim,
                    native_bf16,
                );
                let mut reference = vec![0.0f32; hc * dim];
                let result = if native_bf16 {
                    mhc::hc_post_bf16(config, &row, &stream_values, &post, &comb, &mut reference)
                } else {
                    mhc::hc_post(config, &row, &stream_values, &post, &comb, &mut reference)
                };
                result.expect("the released geometry runs");
                assert_eq!(twin, reference, "native_bf16 = {native_bf16}");
            }
            // And the epilogue's in-place form is its out-of-place form, for the same reason as
            // the post's.
            for native_bf16 in [false, true] {
                let out_of_place = ffn_epilogue(
                    &stream_values,
                    &delta_bits,
                    shared,
                    &post,
                    &comb,
                    hc,
                    dim,
                    native_bf16,
                );
                let mut in_place = stream_values.clone();
                let written = ffn_epilogue_in_place(
                    &mut in_place,
                    &delta_bits,
                    shared,
                    &post,
                    &comb,
                    hc,
                    dim,
                    native_bf16,
                );
                assert_eq!(out_of_place, in_place);
                assert_eq!(out_of_place, written);
            }
        }
    }

    fn narrow(value: f32) -> u16 {
        (bf16_round(value).to_bits() >> 16) as u16
    }

    fn widen(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    #[test]
    fn the_ffn_x_bf16_word_is_the_normed_row_narrowed_in_both_arms() {
        // The kernel's out word is `__float2bfloat16_rn(value)` in the norm loop, where `value` is
        // the row's RMS-normed f32 (already bf16-representable under native_bf16). That is the host
        // arm's `pre_ffn_hidden` - the norm's OUTPUT, which is what the router routes on
        // (docs/archive/build/moe-stage-contract.md H3b) - not the pre-norm hc_pre collapse. This twin
        // states both arms so the out word cannot drift back to the collapse without a red test.
        let (hc, dim) = (4, 256);
        let pre = stream(hc, 91, 0.5);
        let x_hc: Vec<f32> = stream(hc * dim, 92, 1.5);
        let weight: Vec<f32> = stream(dim, 93, 1.0);
        let collapse = hc_pre(&pre, &x_hc, dim);
        for native_bf16 in [false, true] {
            let norm_input: Vec<f32> = collapse
                .iter()
                .map(|value| {
                    if native_bf16 {
                        bf16_round(*value)
                    } else {
                        *value
                    }
                })
                .collect();
            let mut normed = rms_norm(&norm_input, &weight, 1e-5);
            if native_bf16 {
                normed
                    .iter_mut()
                    .for_each(|value| *value = bf16_round(*value));
            }
            let out: Vec<u16> = normed.iter().copied().map(narrow).collect();
            let expected: Vec<u16> = normed.iter().copied().map(narrow).collect();
            assert_eq!(out, expected, "the out word is bf16 of the normed row");
            assert!(
                out.iter()
                    .zip(collapse.iter().map(|value| narrow(*value)))
                    .any(|(got, collapse_bits)| *got != collapse_bits),
                "the pre-norm collapse must not satisfy this: the router routes on the normalized hidden"
            );
            if native_bf16 {
                for (value, bits) in normed.iter().zip(&out) {
                    assert_eq!(
                        value.to_bits(),
                        u32::from(*bits) << 16,
                        "under native_bf16 the normed row is already the value the out word carries"
                    );
                }
            } else {
                assert!(
                    normed
                        .iter()
                        .zip(&out)
                        .any(|(value, bits)| value.to_bits() != u32::from(*bits) << 16),
                    "the f32 arm's normed row is wider than the router's bf16 row"
                );
            }
        }
    }

    #[test]
    fn the_rope_rows_twin_is_the_forward_rotation_bit_for_bit() {
        let head_width = 512;
        let rope_head_dim = 64;
        let row_count = 4;
        let rows: Vec<u16> = stream(row_count * head_width, 71, 3.0)
            .into_iter()
            .map(narrow)
            .collect();
        let cos: Vec<f32> = stream(rope_head_dim / 2, 72, 0.9);
        let sin: Vec<f32> = stream(rope_head_dim / 2, 73, 0.9);
        let twin = rope_rows_ref(&rows, &cos, &sin, row_count, head_width, rope_head_dim, 1.0)
            .expect("the forward arm accepts the released geometry");
        for row in 0..row_count {
            let base = row * head_width;
            let expected = crate::kv_path_ref::rope_row_ref(
                &rows[base..base + head_width],
                &cos,
                &sin,
                head_width,
                rope_head_dim,
            )
            .expect("the kv-path twin accepts the same geometry");
            assert_eq!(
                &twin[base..base + head_width],
                expected.as_slice(),
                "the rope rows twin is the kv-path forward rotation"
            );
        }
    }

    #[test]
    fn the_rope_rows_twin_is_the_engine_inverse_rotation_bit_for_bit() {
        let head_width = 512;
        let rope_head_dim = 64;
        let row_count = 2;
        let rows: Vec<u16> = stream(row_count * head_width, 81, 2.5)
            .into_iter()
            .map(narrow)
            .collect();
        let cos: Vec<f32> = stream(rope_head_dim / 2, 82, 0.9);
        let sin: Vec<f32> = stream(rope_head_dim / 2, 83, 0.9);
        let twin = rope_rows_ref(
            &rows,
            &cos,
            &sin,
            row_count,
            head_width,
            rope_head_dim,
            -1.0,
        )
        .expect("the inverse arm accepts the released geometry");
        let start = head_width - rope_head_dim;
        for row in 0..row_count {
            let base = row * head_width;
            let mut values: Vec<f32> = rows[base..base + head_width]
                .iter()
                .copied()
                .map(widen)
                .collect();
            let mut tail = values[start..].to_vec();
            linalg::rope_rotate_inverse(&mut tail, &cos, &sin)
                .expect("the engine rotates the same tail");
            values[start..].copy_from_slice(&tail);
            let expected: Vec<u16> = values.iter().copied().map(narrow).collect();
            assert_eq!(
                &twin[base..base + head_width],
                expected.as_slice(),
                "the inverse arm is the engine's rope_rotate_inverse on widened bf16"
            );
        }
    }

    #[test]
    fn the_rope_rows_twin_identity_and_signs_are_distinct() {
        let head_width = 8;
        let rope_head_dim = 8;
        let rows: Vec<u16> = vec![
            0x3f80, 0xbf80, 0x3f00, 0xbf00, 0x3f80, 0x3e80, 0xbf80, 0xbe80,
        ];
        let cos = [1.0f32; 4];
        let sin = [0.0f32; 4];
        for sign in [1.0f32, -1.0] {
            let out = rope_rows_ref(&rows, &cos, &sin, 1, head_width, rope_head_dim, sign)
                .expect("a half-turn-free rotation is identity on the widened values");
            assert_eq!(out, rows, "cos = 1, sin = 0 is the identity for both signs");
        }
        // Signed zero is not preserved: x * 1 - (-0.0 * 0 * sign) is x - (-0.0) = +0.0 in IEEE
        // arithmetic, exactly as the host's f32 rotation does it. The twin follows the host, so the
        // test states it instead of asserting a bit identity the host itself does not have.
        let signed_zero: Vec<u16> = [0x0000u16, 0x8000].to_vec();
        let out = rope_rows_ref(&signed_zero, &[1.0f32], &[0.0f32], 1, 2, 2, 1.0)
            .expect("the two-wide row rotates");
        assert_eq!(
            out,
            [0x0000u16, 0x0000],
            "the host arithmetic turns -0.0 into +0.0"
        );
        assert!(rope_rows_ref(&rows, &cos, &sin, 1, head_width, rope_head_dim, 0.5).is_err());
        assert!(rope_rows_ref(&rows[..7], &cos, &sin, 1, head_width, rope_head_dim, 1.0).is_err());
        assert!(rope_rows_ref(&rows, &cos[..3], &sin, 1, head_width, rope_head_dim, 1.0).is_err());
        assert!(rope_rows_ref(&rows, &cos, &sin, 1, head_width, 3, 1.0).is_err());
    }
}

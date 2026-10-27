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

//! The host twin of kernels/rmsnorm_sm121.cu: the engine's own final-norm arithmetic.
//!
//! The reference is not a second implementation of the formula. It decodes the released BF16 bytes
//! to the exact f32 values the engine holds (bf16 -> f32 is exact), calls
//! [sparkflash_engine::linalg::rms_norm] - the same function the served forward pass calls for the
//! final norm at crates/sparkflash-engine/src/forward/mod.rs:1092 - and then applies the kernel's
//! output rounding, \`bf16_bits\` (the CUDA intrinsic \`__float2bfloat16_rn\`'s round-to-nearest-even).
//! So the comparison in the m18 harness is device versus the engine code, and a change to the
//! engine's norm arithmetic shows up as a device/reference mismatch rather than being copied into a
//! twin that changed with it.
//!
//! The engine performs, per row, \`mean = sum(x^2) / cols\`, \`inv = 1 / sqrt(mean + eps)\`, then
//! \`x * inv * w\` in that multiplication order (crates/sparkflash-engine/src/linalg.rs:318). The
//! kernel pins the same order with the correctly-rounded intrinsics; this module deliberately does
//! not restate it.

/// The BF16 bytes of \`rows\` x \`cols\` rows after the engine's final RMSNorm, as BF16 bytes.
///
/// \`x_bf16\` is \`rows * cols\` BF16 values, \`weight_bf16\` is \`cols\` BF16 values and \`eps\` is the
/// configured \`rms_norm_eps\` cast to f32.
///
/// # Errors
/// A zero row/column count, a byte length that is not whole rows, or a weight of the wrong length.
pub fn final_norm_bf16(
    x_bf16: &[u8],
    weight_bf16: &[u8],
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<Vec<u8>, String> {
    if rows == 0 || cols == 0 {
        return Err(format!(
            "the final norm needs at least one row and one column, got {rows} x {cols}"
        ));
    }
    let expected_x = rows
        .checked_mul(cols)
        .and_then(|values| values.checked_mul(2))
        .ok_or_else(|| format!("{rows} x {cols} BF16 rows overflow usize"))?;
    if x_bf16.len() != expected_x {
        return Err(format!(
            "{rows} x {cols} BF16 rows are {expected_x} bytes, got {}",
            x_bf16.len()
        ));
    }
    let expected_weight = cols
        .checked_mul(2)
        .ok_or_else(|| format!("{cols} BF16 weights overflow usize"))?;
    if weight_bf16.len() != expected_weight {
        return Err(format!(
            "{cols} BF16 weights are {expected_weight} bytes, got {}",
            weight_bf16.len()
        ));
    }

    let mut values = decode_bf16(x_bf16);
    let weight = decode_bf16(weight_bf16);
    sparkflash_engine::linalg::rms_norm(&mut values, &weight, cols, eps).map_err(|error| {
        format!("the engine's final RMSNorm refused the reference row: {error}")
    })?;

    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        out.extend_from_slice(&crate::m4_quant::bf16_bits(value).to_le_bytes());
    }
    Ok(out)
}

/// Decodes little-endian BF16 bytes to the exact f32 values they mean.
///
/// # Panics
/// Panics when \`bytes\` is not a whole number of BF16 values; callers validate lengths first.
#[must_use]
pub fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| crate::m4_quant::bf16_bits_to_f32(u16::from_le_bytes(*pair)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| crate::m4_quant::bf16_bits(*value).to_le_bytes())
            .collect()
    }

    fn bits(values: &[u8]) -> Vec<u16> {
        values
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect()
    }

    #[test]
    fn a_unit_row_with_a_flat_weight_is_the_engine_scaling() {
        // The engine's order on a row [1, 0, ...]: sum = 1.0 exactly, mean = 1.0 / cols,
        // inv = 1.0 / sqrt(mean + eps), and out[0] = bf16((1.0 * inv) * 1.0).
        let cols = 5_120;
        let mut x = vec![0.0f32; cols];
        x[0] = 1.0;
        let weight = vec![1.0f32; cols];
        let out = final_norm_bf16(&bf16_bytes(&x), &bf16_bytes(&weight), 1, cols, 1e-20)
            .expect("the reference runs");
        let mean = 1.0f32 / cols as f32;
        let inv = 1.0f32 / (mean + 1e-20f32).sqrt();
        let mut expected = vec![0u16; cols];
        expected[0] = crate::m4_quant::bf16_bits(inv);
        assert_eq!(
            bits(&out),
            expected,
            "a unit row on a flat weight is bf16(inv) at column 0 and zero elsewhere"
        );
    }

    #[test]
    fn a_constant_row_on_an_odd_width_is_scaled_column_by_column() {
        // 513 columns is not a multiple of eight, so the kernel's scalar path and the reference's
        // whole-row call must still agree: every output column is bf16(0.5 * inv), i.e. the same
        // code in every column, and the mean is 0.25 + eps.
        let cols = 513usize;
        let x = vec![0.5f32; cols];
        let weight = vec![1.0f32; cols];
        let out = final_norm_bf16(&bf16_bytes(&x), &bf16_bytes(&weight), 1, cols, 1e-20)
            .expect("the reference runs");
        let mean = (0.5f32 * 0.5f32 * cols as f32) / cols as f32;
        let inv = 1.0f32 / (mean + 1e-20f32).sqrt();
        let expected = crate::m4_quant::bf16_bits(0.5f32 * inv);
        let codes = bits(&out);
        assert_eq!(codes.len(), cols);
        assert!(
            codes.iter().all(|code| *code == expected),
            "every column of a constant odd-width row must be the same bf16 code"
        );
    }

    #[test]
    fn the_reference_calls_the_engines_own_rms_norm() {
        // Review ratchet: this twin must not become a re-implementation of the formula. The source
        // is split before this test so the assertion's own literal cannot satisfy it.
        let source = include_str!("rmsnorm_ref.rs");
        let (production, _) = source
            .split_once("fn the_reference_calls_the_engines_own_rms_norm")
            .expect("the source contains this test");
        assert!(
            production
                .contains("sparkflash_engine::linalg::rms_norm(&mut values, &weight, cols, eps)"),
            "the reference must call the engine's rms_norm, not a hand copy of it"
        );
        assert!(
            !production.contains("mean + eps).sqrt()"),
            "the reference must not restate the engine's arithmetic"
        );
    }

    #[test]
    fn the_kernel_pins_the_engine_rounding_order_and_the_vector_load() {
        // The kernel is the other half of the class: the engine's exact operation order, no FMA
        // contraction, no approximate rsqrt, and the 128-bit read the plan claims.
        let kernel = include_str!("../../../kernels/rmsnorm_sm121.cu");
        for required in [
            "__fsqrt_rn(__fadd_rn(mean, eps))",
            "__fdiv_rn(1.0f,",
            "__fadd_rn(sum, __fmul_rn(value, value))",
            "__fmul_rn(__fmul_rn(value, inv), weight)",
            "ld.global.nc.v4.u32",
            "SPARKFLASH_CUDA_SYNTAX_STUB",
        ] {
            assert!(
                kernel.contains(required),
                "the kernel must contain {required}"
            );
        }
        for forbidden in ["rsqrtf", "__fmaf_rn", "__frsqrt_rn"] {
            assert!(
                !kernel.contains(forbidden),
                "the kernel must not use {forbidden}: it is not the engine's rounding"
            );
        }
    }

    #[test]
    fn the_reference_refuses_bad_shapes() {
        assert!(final_norm_bf16(&[], &[0u8; 2], 1, 0, 1e-20).is_err());
        assert!(final_norm_bf16(&[0u8; 3], &[0u8; 2], 1, 1, 1e-20).is_err());
        assert!(final_norm_bf16(&[0u8; 2], &[0u8; 4], 1, 1, 1e-20).is_err());
    }
}

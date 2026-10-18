//! The host reference for the fused SwiGLU and MX quantise of the seventh M7 kernel,
//! `kernels/swiglu_mxfp8_quant_sm121.cu`.
//!
//! It mirrors two things it does not own, and both are cited rather than paraphrased:
//!
//! * The activation is `crates/sparkflash-engine/src/forward/moe.rs:681-686`: the gate branch is
//!   clamped on the upper side only, the up branch on both sides, and the result is
//!   `silu(gate) * up`. The asymmetry is not an accident to be tidied up: `silu` is not odd,
//!   so the two plausible readings disagree for a gate below the limit, and [swiglu] is the one the
//!   reference uses.
//! * The quantise step is the reference's ACTIVATION quantiser, not the checkpoint weight-grid
//!   rule: one ue8m0 scale per 32 values from the ceil rule (amax floored at 1e-4, then the
//!   smallest power of two at or above amax/448), followed by the clamp to +-448 and a
//!   round-to-nearest-even E4M3 encode. That is [crate::m4_quant::act_quant_fp8_row], which mirrors
//!   kernel.py:36-37/76-124 and is the rule the engine's quant::quantize_activation_fp8 uses; the
//!   weight-grid rule (floor(log2(amax)) - emax) is for stored weights and is deliberately a
//!   different rule.
//!
//! The device side calls `kernels/m4_activation_quant_sm121.cuh`'s
//! `spark_act_quant_scale_code` and `spark_fp8_e4m3_encode`, so what the harness compares is this
//! module against those header functions. This module must never be adjusted to agree with a kernel; the kernel has to agree
//! with it, and with the engine reference behind it.

/// The release configuration's `swiglu_limit`, from the layer-0 config.
pub const SWIGLU_LIMIT: f32 = 10.0;

/// The MX block width: one ue8m0 scale per this many values.
pub const QUANT_BLOCK: usize = 32;

/// The largest finite E4M3 magnitude, the value a block scale must cover.
pub const E4M3_MAX: f32 = 448.0;

/// `silu(x) = x / (1 + exp(-x))`, the f32 form the engine reference evaluates.
#[must_use]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The reference activation of one expert element: the upper-clamped gate times the both-clamped up
/// branch, both at `limit`.
///
/// This is `forward/moe.rs`'s `swiglu` for one element, and the asymmetry in the two clamps is
/// the reason it is written out here instead of being folded into one `clamp` call.
#[must_use]
pub fn swiglu(gate: f32, up: f32, limit: f32) -> f32 {
    let clamped_gate = gate.min(limit);
    let clamped_up = up.clamp(-limit, limit);
    silu(clamped_gate) * clamped_up
}

/// The symmetric variant - both branches clamped on both sides - which is the plausible reading and
/// the wrong one. It exists so the tests can prove [swiglu] is not it.
#[cfg(test)]
fn swiglu_symmetric(gate: f32, up: f32, limit: f32) -> f32 {
    silu(gate.clamp(-limit, limit)) * up.clamp(-limit, limit)
}

/// The value a ue8m0 code denotes: `2^(bits - 127)`, with code 0 the 2^-127 float subnormal
/// and code 255 the reserved NaN.
///
/// This is [crate::m4_quant]'s decoder, re-exported rather than reimplemented. The copy that used
/// to live here added `0x0C00_0000` to the exponent, decoding code 127 as 2^24 instead of 1.0, so
/// [quantise_row]'s inverse scale rounded every E4M3 code to zero. One decoder is the fix.
pub use crate::m4_quant::ue8m0_decode;

/// Encodes an f32 as the nearest E4M3 code, ties to the even mantissa, saturating at 448.
///
/// The finite codes `0x00..=0x7E` increase strictly in magnitude, so the nearest code is one of
/// the two around the insertion point.
#[must_use]
pub fn e4m3_encode(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7F;
    }
    let magnitude = f64::from(value.abs());
    let mut low = 0u8;
    let mut high = 0x7Eu8;
    while low < high {
        let middle = low.midpoint(high);
        if f64::from(e4m3_decode(middle)) < magnitude {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    let mut best = low;
    if low > 0 {
        let above = f64::from(e4m3_decode(low));
        let below = f64::from(e4m3_decode(low - 1));
        if magnitude - below < above - magnitude
            || (magnitude - below == above - magnitude && (low - 1).is_multiple_of(2))
        {
            best = low - 1;
        }
    }
    if value.is_sign_negative() {
        best | 0x80
    } else {
        best
    }
}

/// Decodes an E4M3 code to its f32 value; `0x7F` is a NaN, as the format defines it.
#[must_use]
pub fn e4m3_decode(code: u8) -> f32 {
    let magnitude = code & 0x7F;
    let value = if magnitude == 0x7F {
        f32::NAN
    } else {
        let exponent = (magnitude >> 3) & 0x0F;
        let mantissa = (magnitude & 0x07) as f32 / 8.0;
        if exponent == 0 {
            mantissa * 2.0f32.powi(-6)
        } else {
            (1.0 + mantissa) * 2.0f32.powi(i32::from(exponent) - 7)
        }
    };
    if code & 0x80 != 0 { -value } else { value }
}

/// The fused reference for one row: activations in f32, then the `[cols]` E4M3 bytes and the
/// `[cols / QUANT_BLOCK]` ue8m0 scales the MoE kernel consumes.
#[must_use]
pub fn quantise_row(gate: &[f32], up: &[f32], limit: f32) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(gate.len(), up.len(), "the two branches are one row each");
    assert_eq!(
        gate.len() % QUANT_BLOCK,
        0,
        "a row is a whole number of MX blocks"
    );
    let activated: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(gate, up)| swiglu(*gate, *up, limit))
        .collect();
    // One function, one rule: the released activation quantiser. A second implementation here is
    // how the previous floor-rule scale (and, before that, a decoder that made every code zero)
    // drifted away from the kernel; delegating leaves the kernel nothing to disagree with.
    crate::m4_quant::act_quant_fp8_row(&activated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::m4_quant::{
        ACT_QUANT_AMAX_FLOOR, E4M3_ELEMENT_EMAX, act_quant_scale_code, ue8m0_from_amax,
    };

    #[test]
    fn the_gate_branch_is_clamped_on_one_side_only() {
        // A gate far below the limit is NOT clamped upward: silu(-20) is what enters the product.
        let gate = -20.0f32;
        let up = 5.0f32;
        assert_eq!(swiglu(gate, up, SWIGLU_LIMIT), silu(gate) * up);
        assert_ne!(
            swiglu(gate, up, SWIGLU_LIMIT),
            swiglu_symmetric(gate, up, SWIGLU_LIMIT)
        );
        // And a gate above the limit IS clamped.
        assert_eq!(
            swiglu(20.0, up, SWIGLU_LIMIT),
            silu(SWIGLU_LIMIT) * up,
            "the gate is clamped on the upper side"
        );
    }

    #[test]
    fn the_up_branch_is_clamped_on_both_sides() {
        let gate = 1.0f32;
        assert_eq!(
            swiglu(gate, 20.0, SWIGLU_LIMIT),
            silu(gate) * SWIGLU_LIMIT,
            "an up branch above the limit is clamped down"
        );
        assert_eq!(
            swiglu(gate, -20.0, SWIGLU_LIMIT),
            silu(gate) * -SWIGLU_LIMIT,
            "an up branch below the limit is clamped up"
        );
        assert_ne!(
            swiglu(gate, 20.0, SWIGLU_LIMIT),
            silu(gate.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)) * 20.0
        );
    }

    #[test]
    fn the_kernel_text_clamps_both_branches_the_same_way() {
        // The device kernel must carry the same asymmetry; a source check is what a host test can
        // do about that, and it fails the moment someone tidies the two clamps into one.
        let source = include_str!("../../../kernels/swiglu_mxfp8_quant_sm121.cu");
        assert!(
            source.contains("fminf(g, limit)"),
            "the gate branch is clamped on the upper side only"
        );
        assert!(
            source.contains("fminf(fmaxf(u, -limit), limit)"),
            "the up branch is clamped on both sides"
        );
        assert!(
            !source.contains("fminf(fmaxf(g, -limit), limit)"),
            "the gate branch must not be clamped on both sides"
        );
    }

    #[test]
    fn the_ue8m0_decoder_handles_the_code_127_boundary_and_the_subnormal_zero() {
        assert_eq!(ue8m0_decode(127), 1.0, "code 127 is 2^0, not 2^24");
        assert_eq!(ue8m0_decode(0), f32::from_bits(1 << 22));
        assert!(ue8m0_decode(0) > 0.0, "code 0 is 2^-127, not zero");
        assert!(ue8m0_decode(255).is_nan(), "255 is the reserved NaN");
        for bits in [1u8, 2, 63, 126, 127, 128, 129, 200, 254] {
            assert_eq!(
                ue8m0_decode(bits),
                2.0f32.powi(i32::from(bits) - 127),
                "code {bits}"
            );
        }
    }

    #[test]
    fn a_non_zero_row_does_not_quantise_to_all_zeros() {
        let gate = [1.0f32; QUANT_BLOCK];
        let up = [1.0f32; QUANT_BLOCK];
        let (codes, scales) = quantise_row(&gate, &up, SWIGLU_LIMIT);
        // silu(1) * 1 = 0.731..., so the reference's ceil rule picks the smallest power of two at or
        // above 0.731/448, which is 2^-9 (ue8m0 code 118) - not the weight-grid floor rule's code and
        // not the 2^0 scale a "start doubling from one" helper would return.
        assert_eq!(scales, vec![118], "a unit row takes the 2^-9 scale");
        assert!(
            codes.iter().any(|code| *code != 0),
            "every code was zero: the quantiser's inverse scale is wrong"
        );
        let expected = silu(1.0);
        let got = e4m3_decode(codes[0]) * ue8m0_decode(scales[0]);
        assert!(
            (got - expected).abs() <= 0.0625 / 2.0 + f32::EPSILON,
            "{got} against {expected}"
        );
    }

    #[test]
    fn the_activation_scale_is_the_reference_ceil_rule() {
        // The reference's rule is the SMALLEST power of two at or above amax/448 after the 1e-4 amax
        // floor, so every block fits +-448 without saturation and no block loses a binade. This is
        // the property the kernel's spark_act_quant_scale_code has to reproduce; it is not the
        // weight-grid rule, which floors log2(amax) instead.
        for amax in [1e-6f32, 1e-3, 0.5, 1.0, 3.0, 7.0, 448.0, 449.0, 1e6] {
            let wanted = amax.max(ACT_QUANT_AMAX_FLOOR) / E4M3_MAX;
            let scale = ue8m0_decode(act_quant_scale_code(amax));
            assert!(
                wanted <= scale,
                "amax {amax}: scale {scale} is below the required {wanted}"
            );
            assert!(
                (scale / 2.0) < wanted || scale == 2.0f32.powi(-22),
                "amax {amax}: scale {scale} is not the smallest power of two that covers it"
            );
            assert!(
                scale > 0.0 && (scale.to_bits() & 0x7F_FFFF) == 0,
                "amax {amax}: {scale} is not a power of two"
            );
        }
        // The zero block is the floor's job: amax 1e-4 -> scale 2^-22 -> the code for that scale.
        assert_eq!(
            act_quant_scale_code(ACT_QUANT_AMAX_FLOOR),
            105,
            "an all-zero block takes the 2^-22 floor scale, not the zero code"
        );
    }

    /// Mutation guard for the D8 review's M4a: reinstating the checkpoint weight-grid floor rule in
    /// [quantise_row] left the suite green, because the existing case quantised at an amax where the
    /// two rules agree. With gate = 1 and up = 2.7 the activation's amax is silu(1) * 2.7 = 1.974,
    /// inside the binade (1.75, 2) where they disagree: the activation ceil rule takes 2^-7 (code
    /// 120) and the weight-grid floor rule would take 2^-8 (code 119).
    #[test]
    fn the_quantiser_uses_the_ceil_rule_where_the_weight_grid_rule_would_differ() {
        let activated = swiglu(1.0, 2.7, SWIGLU_LIMIT).abs();
        assert!(
            activated > 1.75 && activated < 2.0,
            "amax {activated} is not in the distinguishing binade"
        );
        let gate = [1.0f32; QUANT_BLOCK];
        let up = [2.7f32; QUANT_BLOCK];
        let (_, scales) = quantise_row(&gate, &up, SWIGLU_LIMIT);
        assert_eq!(
            scales,
            vec![120],
            "the ceil rule's code at amax {activated}"
        );
        assert_eq!(
            ue8m0_from_amax(activated, E4M3_ELEMENT_EMAX),
            119,
            "the weight-grid rule's code at the same amax"
        );
    }

    /// The E4M3 step at the binade a code represents, not the format's finest step.
    fn e4m3_step(code: u8) -> f32 {
        let exponent = (code & 0x7F) >> 3;
        if exponent == 0 {
            2.0f32.powi(-9)
        } else {
            2.0f32.powi(i32::from(exponent) - 10)
        }
    }

    #[test]
    fn the_quantiser_round_trips_within_half_a_step() {
        let mut gate = [0.0f32; 64];
        let mut up = [0.0f32; 64];
        for index in 0..64 {
            gate[index] = (index as f32 - 32.0) * 0.37;
            up[index] = (index as f32 * 0.11).sin() * 3.0;
        }
        let (codes, scales) = quantise_row(&gate, &up, SWIGLU_LIMIT);
        assert_eq!(scales.len(), 2);
        assert!(
            codes.iter().any(|code| *code != 0),
            "a non-zero row must not quantise to all-zero E4M3 codes"
        );
        for (block, scale) in scales.iter().enumerate() {
            let denominator = ue8m0_decode(*scale);
            assert_eq!(
                denominator,
                2.0f32.powi(i32::from(*scale) - 127),
                "the tolerance's step must use the code's true value"
            );
            for offset in 0..QUANT_BLOCK {
                let index = block * QUANT_BLOCK + offset;
                let expected = swiglu(gate[index], up[index], SWIGLU_LIMIT);
                let got = e4m3_decode(codes[index]) * denominator;
                let step = e4m3_step(codes[index]);
                assert!(
                    (got - expected).abs() <= step * 0.5 + f32::EPSILON * expected.abs(),
                    "element {index}: {got} against {expected} exceeds half the {step} step"
                );
            }
        }
    }
}

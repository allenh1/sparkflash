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

//! D8's activation quantiser is one rule, and this test is where the two implementations of it
//! are compared: the engine's reference (sparkflash_engine::quant, the rule the token stream is
//! actually quantised by) and the host oracle the device kernels are diffed against
//! (sparkflash_kernels::m4_quant). The D8 review's B1 found the kernel oracle still routing an
//! infinite block maximum through a fast_pow2-based path that panics, while the engine and the
//! kernel header were total; this file pins the two together on exactly the inputs the old path
//! could not take.
//!
//! The cli crate is the one place that depends on both crates, so the equivalence lives here
//! rather than being copied into either crate as a second implementation.

use sparkflash_engine::quant::{ACTIVATION_AMAX_FLOOR, FP8_E4M3_MAX, e8m0_encode_ceil};
use sparkflash_kernels::m4_quant::{act_quant_scale_code, ue8m0_code_ceil};

/// The engine's block-scale rule for one activation block: the 1e-4 floor, the /448 product, then
/// the ceiling as an E8M0 code (quant::quantize_activation_fp8 calls block_scale_code, which is
/// exactly this composition).
fn engine_rule(amax: f32) -> u8 {
    e8m0_encode_ceil(amax.max(ACTIVATION_AMAX_FLOOR) * (1.0 / FP8_E4M3_MAX))
}

#[test]
fn the_kernel_oracle_and_the_engine_agree_on_the_awkward_inputs() {
    let mut cases = vec![
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        0.0,
        -0.0,
        -1.0,
        ACTIVATION_AMAX_FLOOR,
        f32::MAX,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(0x007F_FFFF),
        448.0,
        449.0,
        224.0,
        1.9,
        1.0,
        1.5,
        0.5,
    ];
    // Every power of two the rule can express: 448 * 2^e for e in -40..=40, so the ceiling lands on
    // a code with a zero mantissa (the fast_log2_ceil path's only easy case).
    for exponent in -40..=40 {
        cases.push(448.0 * 2.0f32.powi(exponent));
    }
    // Deterministic bit patterns across every exponent field, so the comparison is not limited to
    // values a hand-written table likes: a plain LCG walks mantissas, signs and subnormals.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for _ in 0..20_000 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (state >> 32) as u32;
        cases.push(f32::from_bits(bits));
    }
    for amax in cases {
        assert_eq!(
            act_quant_scale_code(amax),
            engine_rule(amax),
            "amax {amax:e} ({:#010x}): the kernel oracle and the engine disagree",
            amax.to_bits()
        );
    }
}

#[test]
fn the_ceiling_helper_is_the_engines_total_ceiling() {
    // Direct comparison of the two ceiling functions, including the reserved codes: the engine's
    // e8m0_encode_ceil returns 0xFF for NaN and clamps infinity to 254, and so does the oracle's.
    let mut cases = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(2),
        f32::from_bits(3),
        f32::from_bits(1 << 22),
        f32::from_bits((1 << 23) - 1),
        0.5,
        1.0,
        1.5,
        2.0,
        f32::MAX,
    ];
    for exponent in -120..=120 {
        cases.push(2.0f32.powi(exponent));
    }
    for value in cases {
        assert_eq!(
            ue8m0_code_ceil(value),
            e8m0_encode_ceil(value),
            "value {value:e} ({:#010x})",
            value.to_bits()
        );
    }
}

#[test]
fn the_old_fast_pow2_path_really_did_panic_on_infinity() {
    // The regression being guarded: the reference's fast_pow2 chain built an infinite scale for an
    // infinite amax, and the storing step rejects it. Keeping this assertion means a future change
    // cannot quietly reintroduce that composition into an oracle path.
    let scale = sparkflash_kernels::m4_quant::act_quant_scale(f32::INFINITY);
    assert!(scale.is_infinite());
    let outcome = std::panic::catch_unwind(|| {
        sparkflash_kernels::m4_quant::ue8m0_code_of_power_of_two(scale)
    });
    assert!(
        outcome.is_err(),
        "the old path is expected to panic; it is not used"
    );
}

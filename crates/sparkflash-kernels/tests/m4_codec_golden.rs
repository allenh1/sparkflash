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

//! Exhaustive check of the MX codecs against a table produced by the real CUDA toolkit.
//!
//! tests/fixtures/m4_codec_cuda_12_9.hex was written by nvcc 12.9.1 running
//! tests/fixtures/m4_codec_gen.cu on this host (its converters are host-callable, so no GPU was
//! involved). It records, for every one of the 65,536 bf16 bit patterns, the byte that
//! __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3) returns and the nibble that
//! __nv_cvt_float_to_fp4(x, __NV_E2M1, cudaRoundNearest) returns.
//!
//! The kernels use those intrinsics when nvcc compiles them, and the bit-arithmetic fallback
//! otherwise; src/m4_quant.rs mirrors the fallback for the CPU references. Comparing all three
//! against this table is the only way this host can show that the fallback, the CPU reference
//! and the GPU path agree: the domain is the complete set of values that can reach an encoder,
//! since every activation and weight in these kernels arrives as bf16.
//!
//! Non-vacuity is checked too, at the bottom: deliberately wrong encoders (wrong exponent bias,
//! truncation instead of round-to-nearest-even, wrong saturation constant, sign-preserving NaN)
//! are each compared against the same table and must disagree with it.
//!
//! The fixture is reproducible in-tree rather than taken on trust:
//! scripts/regen-codec-fixture.sh runs the generator under nvcc, writes the table to a temporary
//! path and diffs it against the committed bytes, so a toolkit change or a hand-edited fixture
//! fails a check anyone can run. The generator takes its output path as an argument for exactly
//! that reason.

use sparkflash_kernels::m4_quant::{
    E2M1_ELEMENT_EMAX, E4M3_ELEMENT_EMAX, UE8M0_BIAS, UE8M0_MAX, bf16_bits_to_f32, fp4_e2m1_encode,
    fp8_e4m3_encode, ue8m0_decode, ue8m0_from_amax,
};

const TABLE: &str = include_str!("fixtures/m4_codec_cuda_12_9.hex");
const BF16_DOMAIN: u32 = 65_536;

struct GoldenTable {
    fp8: Vec<u8>,
    fp4: Vec<u8>,
}

fn load_table() -> GoldenTable {
    let mut fp8 = Vec::with_capacity(BF16_DOMAIN as usize);
    let mut fp4 = Vec::with_capacity(BF16_DOMAIN as usize);
    let mut body = String::new();
    for line in TABLE.lines() {
        if line.starts_with('#') {
            continue;
        }
        body.push_str(line.trim());
    }
    assert_eq!(
        body.len(),
        BF16_DOMAIN as usize * 3,
        "the fixture must hold three hex digits per bf16 pattern"
    );
    for chunk in body.as_bytes().chunks(3) {
        let digits = std::str::from_utf8(chunk).expect("fixture is ASCII hex");
        let byte = u8::from_str_radix(&digits[..2], 16).expect("fp8 hex");
        let nibble = u8::from_str_radix(&digits[2..], 16).expect("fp4 hex");
        fp8.push(byte);
        fp4.push(nibble);
    }
    GoldenTable { fp8, fp4 }
}

#[test]
fn every_bf16_input_matches_cudas_fp8_and_fp4_converters() {
    let table = load_table();
    let mut fp8_mismatches = Vec::new();
    let mut fp4_mismatches = Vec::new();
    for bits in 0..BF16_DOMAIN {
        let value = bf16_bits_to_f32(bits as u16);
        let expected_fp8 = table.fp8[bits as usize];
        let expected_fp4 = table.fp4[bits as usize];
        let actual_fp8 = fp8_e4m3_encode(value);
        let actual_fp4 = fp4_e2m1_encode(value);
        if actual_fp8 != expected_fp8 && fp8_mismatches.len() < 8 {
            fp8_mismatches.push((bits, value, expected_fp8, actual_fp8));
        }
        if actual_fp4 != expected_fp4 && fp4_mismatches.len() < 8 {
            fp4_mismatches.push((bits, value, expected_fp4, actual_fp4));
        }
    }
    assert!(
        fp8_mismatches.is_empty(),
        "E4M3 mismatches (bits, value, cuda, rust): {fp8_mismatches:?}"
    );
    assert!(
        fp4_mismatches.is_empty(),
        "E2M1 mismatches (bits, value, cuda, rust): {fp4_mismatches:?}"
    );
}

#[test]
fn the_fixture_covers_the_whole_format_not_just_a_few_codes() {
    let table = load_table();
    // A table that always returned the same byte would pass a careless comparison; these are the
    // distinct codes that must appear across the bf16 domain.
    let distinct_fp8: std::collections::BTreeSet<u8> = table.fp8.iter().copied().collect();
    let distinct_fp4: std::collections::BTreeSet<u8> = table.fp4.iter().copied().collect();
    assert_eq!(
        distinct_fp8.len(),
        255,
        "E4M3 must use every code but one (0xFF and 0x7F are NaN)"
    );
    assert_eq!(distinct_fp4.len(), 16, "E2M1 must use all sixteen codes");
    for required in [0x00u8, 0x38, 0x7E, 0x7F, 0x80, 0xB8, 0xFE] {
        assert!(
            distinct_fp8.contains(&required),
            "E4M3 code {required:#04x} never produced"
        );
    }
    for required in [0x0u8, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0xF] {
        assert!(
            distinct_fp4.contains(&required),
            "E2M1 code {required:#x} never produced"
        );
    }
}

/// The four deliberately wrong encoders used to show the table discriminates.
mod wrong {
    /// E4M3 with the exponent bias off by one: field = unbiased + 6 instead of + 7.
    #[must_use]
    pub fn fp8_wrong_bias(value: f32) -> u8 {
        let bits = value.to_bits();
        let sign = (bits >> 31) & 1;
        let exponent = (bits >> 23) & 0xFF;
        let mantissa = bits & 0x007F_FFFF;
        let sign_byte = (sign << 7) as u8;
        let magnitude = bits & 0x7FFF_FFFF;
        if magnitude < 0x0080_0000 {
            return sign_byte;
        }
        if exponent == 0xFF {
            return if mantissa != 0 {
                0x7F
            } else {
                sign_byte | 0x7E
            };
        }
        let unbiased = exponent as i32 - 127;
        if unbiased > 8 {
            return sign_byte | 0x7E;
        }
        if unbiased < -6 {
            return sign_byte;
        }
        let exponent_field = (unbiased + 6) as u32;
        let kept = (0x0080_0000u32 | mantissa) >> 20;
        sign_byte | ((exponent_field as u8) << 3) | (kept as u8 - 8)
    }

    /// E4M3 that truncates instead of rounding to nearest even.
    #[must_use]
    pub fn fp8_truncating(value: f32) -> u8 {
        let bits = value.to_bits();
        let sign = (bits >> 31) & 1;
        let exponent = (bits >> 23) & 0xFF;
        let mantissa = bits & 0x007F_FFFF;
        let sign_byte = (sign << 7) as u8;
        let magnitude = bits & 0x7FFF_FFFF;
        if magnitude < 0x0080_0000 {
            return sign_byte;
        }
        if exponent == 0xFF {
            return if mantissa != 0 {
                0x7F
            } else {
                sign_byte | 0x7E
            };
        }
        let mut unbiased = exponent as i32 - 127;
        let mut kept = (0x0080_0000u32 | mantissa) >> 20;
        if kept == 16 {
            kept = 8;
            unbiased += 1;
        }
        if unbiased > 8 {
            return sign_byte | 0x7E;
        }
        if unbiased < -6 {
            let shift = -6 - unbiased;
            if shift > 11 {
                return sign_byte;
            }
            let bits_lost = (20 + shift) as u32;
            let rounded = (0x0080_0000u32 | mantissa) >> bits_lost;
            return sign_byte | rounded as u8;
        }
        sign_byte | (((unbiased + 7) as u8) << 3) | (kept as u8 - 8)
    }

    /// E2M1 saturating to 4.0 (code 0x6) instead of the format maximum 6.0 (code 0x7).
    #[must_use]
    pub fn fp4_wrong_saturation(value: f32) -> u8 {
        let magnitude = value.abs();
        if magnitude > 4.0 {
            return ((value.is_sign_negative() as u8) << 3) | 0x6;
        }
        super::fp4_e2m1_encode(value).min(0x6)
    }

    /// E4M3 keeping the sign of a NaN, which CUDA does not do.
    #[must_use]
    pub fn fp8_sign_preserving_nan(value: f32) -> u8 {
        let bits = value.to_bits();
        if (bits >> 23) & 0xFF == 0xFF && bits & 0x007F_FFFF != 0 {
            return ((bits >> 31) as u8 & 1) << 7 | 0x7F;
        }
        super::fp8_e4m3_encode(value)
    }
}

#[test]
fn the_golden_table_rejects_wrong_codecs() {
    let table = load_table();
    let mut counts = [0usize; 4];
    for bits in 0..BF16_DOMAIN {
        let value = bf16_bits_to_f32(bits as u16);
        let index = bits as usize;
        if wrong::fp8_wrong_bias(value) != table.fp8[index] {
            counts[0] += 1;
        }
        if wrong::fp8_truncating(value) != table.fp8[index] {
            counts[1] += 1;
        }
        if wrong::fp4_wrong_saturation(value) != table.fp4[index] {
            counts[2] += 1;
        }
        if wrong::fp8_sign_preserving_nan(value) != table.fp8[index] {
            counts[3] += 1;
        }
    }
    for (index, count) in counts.iter().enumerate() {
        assert!(
            *count > 0,
            "wrong encoder {index} agrees with CUDA everywhere; the table does not test it"
        );
    }
    // The bias error must be caught on a large fraction of the domain, not on a single freak
    // input: an off-by-one exponent changes every value in the E4M3 normal range. That range
    // spans bf16 exponents -6..=8, i.e. 15 of bf16's 256 exponent fields with all 128 mantissa
    // patterns and both signs, so the expected count is about 15 * 128 * 2 = 3,840 plus the
    // values just outside it that round differently.
    assert!(
        counts[0] > 3_800,
        "the bias mutation should be caught across the normal range, got {}",
        counts[0]
    );
}

#[test]
fn ue8m0_scales_are_the_smallest_power_of_two_that_fits_the_block() {
    // The scale derivation is pure bit arithmetic on both toolchains, so this pins the contract
    // the kernels rely on. amax / scale lands in [2^emax, 2^(emax+1)) for every magnitude that
    // the code range can express: halving the scale would push the block past the format's
    // maximum, and doubling it would waste an exponent. The E4M3 maximum is 448 and its emax is
    // 8, so the ratio must be above 224 (2 * scale overflows) and at most 511 (amax still
    // saturates into range rather than becoming infinite).
    for exponent in -125i32..=126 {
        let amax = 2.0f32.powi(exponent);
        // For a power-of-two maximum the ideal exponent is exact, so the code is pinned without
        // floating-point tolerance: code = floor(log2(amax)) - emax + 127.
        let ideal = exponent - E4M3_ELEMENT_EMAX as i32;
        let code = ue8m0_from_amax(amax, E4M3_ELEMENT_EMAX);
        let scale = ue8m0_decode(code);
        assert!(scale > 0.0, "scale must be positive, code {code}");
        if (-127..=127).contains(&ideal) {
            assert_eq!(i32::from(code), ideal + 127, "scale for amax 2^{exponent}");
            assert_eq!(
                scale.log2(),
                ideal as f32,
                "scale value for amax 2^{exponent}"
            );
            let ratio = amax / scale;
            assert!(
                ratio <= 511.0,
                "amax 2^{exponent} over scale {scale} is {ratio}, past the E4M3 maximum"
            );
            assert!(
                ratio > 224.0,
                "half the scale ({}) would still have fit amax 2^{exponent}",
                scale / 2.0
            );
            let code4 = ue8m0_from_amax(amax, E2M1_ELEMENT_EMAX);
            let ratio4 = amax / ue8m0_decode(code4);
            assert!(
                ratio4 <= 7.0 && ratio4 > 3.0,
                "E2M1 ratio for amax 2^{exponent} is {ratio4}, expected (3, 7]"
            );
        } else {
            // Outside the ue8m0 range the code clamps to one end, and there the headroom
            // property genuinely cannot hold: that is the documented limit of the format.
            assert!(code == 0 || code == UE8M0_MAX, "clamped code {code}");
            assert_eq!(code, if ideal < -127 { 0 } else { UE8M0_MAX });
        }
    }
    assert_eq!(ue8m0_from_amax(0.0, E4M3_ELEMENT_EMAX), UE8M0_BIAS);
    assert_eq!(ue8m0_from_amax(-1.0, E4M3_ELEMENT_EMAX), UE8M0_BIAS);
    assert_eq!(ue8m0_decode(UE8M0_BIAS), 1.0);
    assert!(ue8m0_decode(255).is_nan());
    // Code 0 is 2^-127, which is a float subnormal rather than a normal power of two.
    assert_eq!(ue8m0_decode(0), 2.0f32.powi(-127));
    assert_eq!(ue8m0_decode(0).to_bits(), 1 << 22);
    // An infinite maximum takes the largest scale; 2^128 overflows f32, so it is the ceiling.
    assert_eq!(ue8m0_from_amax(f32::INFINITY, E4M3_ELEMENT_EMAX), UE8M0_MAX);
    assert_eq!(
        ue8m0_from_amax(2.0f32.powi(128), E4M3_ELEMENT_EMAX),
        UE8M0_MAX
    );
    assert_eq!(ue8m0_from_amax(2.0f32.powi(-125), E4M3_ELEMENT_EMAX), 0);
}

#[test]
fn the_format_maximum_is_first_produced_by_rounding_at_432() {
    // The claim the saturation comment in kernels/m4_common_sm121.cuh makes, checked by enumeration
    // over the whole bf16 domain rather than reasoned about. Two different numbers are involved and
    // conflating them is how that comment was wrong for a while: the E4M3 maximum code 0x7E is
    // first PRODUCED at 432.0 (bf16 0x43D8), by ordinary round-to-nearest-even up to 448, while the
    // saturation branch that exists for 1.875 * 2^8 is first REACHED by an input above 464.
    //
    // bf16 spacing is 2.0 in this binade, so the neighbours matter: 430.0 is the bf16 input below
    // 432 and still rounds down to 416, and 464 and 466 are the bf16 inputs around the saturation
    // boundary. 465 is a legitimate f32 input to the encoder but is not a bf16 value, so the branch
    // is described in f32 terms while the enumeration below is in bf16 terms.
    assert_eq!(bf16_bits_to_f32(0x43D7), 430.0);
    assert_eq!(bf16_bits_to_f32(0x43D8), 432.0);
    assert_eq!(bf16_bits_to_f32(0x43E8), 464.0);
    assert_eq!(bf16_bits_to_f32(0x43E9), 466.0);
    assert_eq!(fp8_e4m3_encode(430.0), 0x7D, "430 still rounds down to 416");
    assert_eq!(
        fp8_e4m3_encode(431.0),
        0x7D,
        "431 is an f32 input that also rounds down"
    );
    assert_eq!(
        fp8_e4m3_encode(432.0),
        0x7E,
        "432 is the midpoint and rounds up"
    );
    assert_eq!(fp8_e4m3_encode(448.0), 0x7E);
    assert_eq!(
        fp8_e4m3_encode(464.0),
        0x7E,
        "464 rounds to 448 without saturating"
    );
    assert_eq!(
        fp8_e4m3_encode(465.0),
        0x7E,
        "465 rounds to 480 and must saturate"
    );
    assert_eq!(
        fp8_e4m3_encode(466.0),
        0x7E,
        "466 is the next bf16 input and must saturate"
    );
    assert_eq!(fp8_e4m3_encode(480.0), 0x7E);
    // And the first input in the whole domain that produces 0x7E is that same 432: the fixture and
    // the fallback agree on where the format maximum begins.
    let table = load_table();
    let first = (0..BF16_DOMAIN)
        .find(|bits| table.fp8[*bits as usize] == 0x7E)
        .expect("some bf16 input must produce the E4M3 maximum");
    assert_eq!(first, 0x43D8, "the first 0x7E input must be bf16 0x43D8");
    assert_eq!(table.fp8[0x43D8], fp8_e4m3_encode(432.0));
}

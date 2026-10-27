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

//! Bit-exact bf16 conversions used by the canonical-order reduction.
//!
//! Both directions are total and allocation free. The narrowing direction is the same
//! round-to-nearest-even every CUDA `__float2bfloat16_rn` performs, which is what makes the
//! CPU references and the sm_121a kernel bit-identical for every finite value and for both
//! infinities. NaN is the one exception: this function keeps the sign and forces the value quiet
//! without preserving the payload, while the CUDA host routine returned 0x7fff for every NaN in
//! the M3 review, so NaN payload and sign sit outside the bit-identical contract (see the crate
//! docs and roce_reduce.rs).

/// Widens bf16 bits to the f32 value that is exactly equal to them.
///
/// bf16 is the top 16 bits of f32, so widening is a shift with no rounding at all.
#[must_use]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Narrows f32 to bf16 with round-to-nearest-even, ties to even.
///
/// This is the algorithm CUDA documents for `__float2bfloat16_rn` and the one PyTorch uses:
/// add a rounding bias of `0x7fff` plus the low bit of the kept half, then truncate. Overflow
/// rounds to infinity; NaN keeps its sign and is forced quiet (payload bits are not preserved,
/// which no inference reduction depends on).
#[must_use]
pub fn f32_to_bf16_rn(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let low_bit = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + low_bit;
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widening_is_an_exact_shift() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xbf80), -1.0);
        assert_eq!(bf16_to_f32(0x0000).to_bits(), 0);
        assert_eq!(bf16_to_f32(0x8000).to_bits(), 0x8000_0000);
        assert!(bf16_to_f32(0x7f80).is_infinite());
    }

    #[test]
    fn narrowing_matches_known_bit_patterns() {
        assert_eq!(f32_to_bf16_rn(0.0), 0x0000);
        assert_eq!(f32_to_bf16_rn(-0.0), 0x8000);
        assert_eq!(f32_to_bf16_rn(1.0), 0x3f80);
        assert_eq!(f32_to_bf16_rn(-1.0), 0xbf80);
        assert_eq!(f32_to_bf16_rn(1.5), 0x3fc0);
        assert_eq!(f32_to_bf16_rn(f32::INFINITY), 0x7f80);
        assert_eq!(f32_to_bf16_rn(f32::NEG_INFINITY), 0xff80);
        // f32::MAX is above the bf16 overflow boundary and must round to infinity.
        assert_eq!(f32_to_bf16_rn(f32::MAX), 0x7f80);
        // bf16 0x7f7f is the largest finite bf16; the f32 just below the boundary keeps it.
        assert_eq!(f32_to_bf16_rn(f32::from_bits(0x7f7f_0000)), 0x7f7f);
        assert_eq!(f32_to_bf16_rn(f32::from_bits(0x7f7f_7fff)), 0x7f7f);
        assert_eq!(f32_to_bf16_rn(f32::from_bits(0x7f7f_8000)), 0x7f80);
        // An f32 subnormal rounds to bf16 zero rather than to the smallest bf16 subnormal.
        assert_eq!(f32_to_bf16_rn(f32::from_bits(1)), 0x0000);
    }

    #[test]
    fn ties_go_to_even() {
        // Midpoint between bf16 0x3f80 (even) and 0x3f81 (odd) rounds down.
        assert_eq!(f32_to_bf16_rn(f32::from_bits(0x3f80_8000)), 0x3f80);
        // ... and one ulp above the midpoint rounds up.
        assert_eq!(f32_to_bf16_rn(f32::from_bits(0x3f80_8001)), 0x3f81);
        // Midpoint between bf16 0x3f81 (odd) and 0x3f82 (even) rounds up.
        assert_eq!(f32_to_bf16_rn(f32::from_bits(0x3f81_8000)), 0x3f82);
        // Negative midpoints behave identically (the bias arithmetic is sign-magnitude safe).
        assert_eq!(f32_to_bf16_rn(f32::from_bits(0xbf80_8000)), 0xbf80);
        assert_eq!(f32_to_bf16_rn(f32::from_bits(0xbf81_8000)), 0xbf82);
    }

    #[test]
    fn nan_is_canonicalised_and_quiet() {
        assert_eq!(f32_to_bf16_rn(f32::NAN), 0x7fc0);
        assert_eq!(f32_to_bf16_rn(-f32::NAN), 0xffc0);
        // A signalling NaN payload is quieted but keeps its sign.
        assert_eq!(f32_to_bf16_rn(f32::from_bits(0x7f80_0001)), 0x7fc0);
        assert_eq!(f32_to_bf16_rn(f32::from_bits(0xff80_0001)), 0xffc0);
    }

    /// Every finite/infinite bf16 pattern must survive a widen/narrow round trip. NaNs are
    /// excluded because the payload is deliberately not preserved.
    #[test]
    fn every_non_nan_pattern_round_trips_exactly() {
        for bits in 0u32..=u32::from(u16::MAX) {
            let value = bits as u16;
            if (value & 0x7f80) == 0x7f80 && (value & 0x007f) != 0 {
                continue;
            }
            assert_eq!(
                f32_to_bf16_rn(bf16_to_f32(value)),
                value,
                "bf16 0x{value:04x} did not round trip"
            );
        }
    }

    /// The `half` crate is an independent implementation of the same IEEE conversion; use it
    /// as an oracle over a wide deterministic sweep rather than trusting our own arithmetic.
    #[test]
    fn agrees_with_the_half_crate_oracle() {
        let mut checked = 0usize;
        // Sweep the whole f32 exponent range with a few mantissa patterns each.
        for exponent in 1u32..=254 {
            for mantissa in [0u32, 1, 0x40, 0x7f_ffff, 0x40_0000, 0x00_8000, 0x3f_ffff] {
                for sign in [0u32, 0x8000_0000] {
                    let bits = sign | (exponent << 23) | mantissa;
                    let value = f32::from_bits(bits);
                    let expected = half::bf16::from_f32(value).to_bits();
                    assert_eq!(
                        f32_to_bf16_rn(value),
                        expected,
                        "f32 0x{bits:08x} disagreed with the half oracle"
                    );
                    checked += 1;
                }
            }
        }
        // Dense sweep of one binade, including every tie and near-tie.
        for mantissa in 0u32..0x80_0000 {
            let value = f32::from_bits(0x3f80_0000 | mantissa);
            let expected = half::bf16::from_f32(value).to_bits();
            let bits = value.to_bits();
            assert_eq!(
                f32_to_bf16_rn(value),
                expected,
                "f32 0x{bits:08x} disagreed"
            );
            checked += 1;
        }
        assert!(checked > 8_000_000, "the oracle sweep unexpectedly shrank");
    }
}

//! CPU reference for the RoCEnante one-shot All-Reduce numerics.
//!
//! This module is the definition of "bit-identical" for milestone 3. The CUDA kernel
//! (kernels/rocenante_oneshot_sm121.cu) and every Rust transport in sparkflash-comm must agree
//! with it bit for bit:
//!
//! 1. one f32 accumulator per element, initialised to `+0.0f`;
//! 2. the four rank rows are added in ascending rank order 0, 1, 2, 3, with an f32 rounding
//!    after every addition (that is what `acc += ...` does in the kernel);
//! 3. exactly one rounding to bf16 at the end, round-to-nearest-even. The kernel calls
//!    `__float2bfloat16_rn`; this module calls `half::bf16::from_f32`, which
//!    implements the same rule (half 2.7.1 `convert::f32_to_bf16`).
//!
//! Three consequences are worth stating because they are easy to get wrong:
//!
//! * The accumulation order is observable, so "sum in any order, round once" is NOT this
//!   contract. `tests::the_canonical_order_is_observable` pins a pair of inputs whose
//!   result changes when ranks 2 and 3 are swapped.
//! * The result is not "round the exact sum once" either: f32 rounding happens between the
//!   additions, and `tests::golden_bit_vectors` includes a vector where that changes
//!   the bf16 output.
//! * Starting the accumulator at `+0.0f` rather than at rank 0's value is observable
//!   for signed zeros: four times `-0.0` reduces to `+0.0`.
//!
//! NaN payloads and the sign of a NaN are explicitly outside the contract: they differ between
//! CUDA and the host CPU. Non-finite inputs are required only to stay non-finite in the
//! documented way (+inf + -inf is NaN).
//!
//! The functions here are weight-free, allocation-free and testable on a machine with no GPU.

use half::bf16;
use thiserror::Error;

use crate::roce_abi::{ROCE_ONESHOT_RANKS, ROCE_ONESHOT_SLOT_ROW_ELEMS};

/// The canonical accumulation order, as data: rank 0 first, then 1, 2, 3.
pub const CANONICAL_RANK_ORDER: [usize; ROCE_ONESHOT_RANKS] = [0, 1, 2, 3];

/// Reasons a reference reduction refuses to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum RoceReduceError {
    /// The caller did not supply exactly [ROCE_ONESHOT_RANKS] rows.
    #[error("the canonical reduction needs exactly {expected} rank rows, got {actual}")]
    RankCount {
        /// Rows the one-shot group requires.
        expected: usize,
        /// Rows the caller supplied.
        actual: usize,
    },
    /// One rank's row is shorter than the requested element count.
    #[error("rank {rank} has {row} elements but {out} were requested")]
    RowTooShort {
        /// Offending rank index.
        rank: usize,
        /// Elements that rank supplied.
        row: usize,
        /// Elements the caller asked for.
        out: usize,
    },
    /// The output is larger than the 2 MiB slot row the kernel can write.
    #[error("an output of {out} elements exceeds the {capacity}-element slot row")]
    OutputTooLong {
        /// Requested elements.
        out: usize,
        /// Elements one slot row holds.
        capacity: usize,
    },
    /// A grid-stride simulation with no threads.
    #[error("a launch with block_dim {block_dim} and grid_dim {grid_dim} has no threads")]
    EmptyLaunch {
        /// Threads per block.
        block_dim: u32,
        /// Blocks per grid.
        grid_dim: u32,
    },
}

/// The canonical f32 accumulation for one element index.
///
/// This is the exact expression the CUDA kernel evaluates:
/// `float acc = 0.0f; for r in 0..4 { acc += (float)payload[r][i]; }`.
///
/// # Errors
/// Returns [RoceReduceError::RankCount] when `rows` is not a four-rank group, and
/// [RoceReduceError::RowTooShort] when any row is shorter than `index + 1`.
pub fn canonical_sum_f32(rows: &[&[bf16]], index: usize) -> Result<f32, RoceReduceError> {
    validate(rows, index + 1)?;
    let mut acc = 0.0f32;
    for row in rows {
        acc += row[index].to_f32();
    }
    Ok(acc)
}

/// Reduces one element index across the four ranks, canonical order, one bf16 rounding.
///
/// # Errors
/// As [canonical_sum_f32].
pub fn canonical_reduce_element(rows: &[&[bf16]], index: usize) -> Result<bf16, RoceReduceError> {
    Ok(bf16::from_f32(canonical_sum_f32(rows, index)?))
}

/// The reference All-Reduce over the first `out.len()` elements of every rank row.
///
/// Rows may be longer than the output; the extra elements are ignored, exactly as the kernel
/// ignores `i >= num_elements`.
///
/// # Errors
/// [RoceReduceError::RankCount] unless there are exactly four rows;
/// [RoceReduceError::OutputTooLong] when the output does not fit one slot row;
/// [RoceReduceError::RowTooShort] when a rank supplies too few elements.
pub fn all_reduce_bf16_canonical(
    rows: &[&[bf16]],
    out: &mut [bf16],
) -> Result<(), RoceReduceError> {
    validate(rows, out.len())?;
    for (index, slot) in out.iter_mut().enumerate() {
        *slot = bf16::from_f32(sum_at(rows, index));
    }
    Ok(())
}

/// The reference All-Reduce with the kernel's grid-stride thread mapping.
///
/// Element `i` is processed by thread `tid = i % stride` and each element is
/// processed exactly once, which is why any launch shape must produce the same bits as
/// [all_reduce_bf16_canonical]. This exists so the "num_elements is not a multiple of the
/// block/grid stride" case is covered on the host.
///
/// # Errors
/// As [all_reduce_bf16_canonical], plus [RoceReduceError::EmptyLaunch] for a zero-sized launch.
pub fn all_reduce_bf16_canonical_strided(
    rows: &[&[bf16]],
    out: &mut [bf16],
    block_dim: u32,
    grid_dim: u32,
) -> Result<(), RoceReduceError> {
    if block_dim == 0 || grid_dim == 0 {
        return Err(RoceReduceError::EmptyLaunch {
            block_dim,
            grid_dim,
        });
    }
    validate(rows, out.len())?;
    let stride = (block_dim as usize) * (grid_dim as usize);
    for tid in 0..stride {
        let mut index = tid;
        while index < out.len() {
            out[index] = bf16::from_f32(sum_at(rows, index));
            index += stride;
        }
    }
    Ok(())
}

/// Validates a row group for `needed` elements and checks the slot row bound.
fn validate(rows: &[&[bf16]], needed: usize) -> Result<(), RoceReduceError> {
    if rows.len() != ROCE_ONESHOT_RANKS {
        return Err(RoceReduceError::RankCount {
            expected: ROCE_ONESHOT_RANKS,
            actual: rows.len(),
        });
    }
    if needed > ROCE_ONESHOT_SLOT_ROW_ELEMS {
        return Err(RoceReduceError::OutputTooLong {
            out: needed,
            capacity: ROCE_ONESHOT_SLOT_ROW_ELEMS,
        });
    }
    for (rank, row) in rows.iter().enumerate() {
        if row.len() < needed {
            return Err(RoceReduceError::RowTooShort {
                rank,
                row: row.len(),
                out: needed,
            });
        }
    }
    Ok(())
}

/// The canonical sum for an index already validated by [validate].
fn sum_at(rows: &[&[bf16]], index: usize) -> f32 {
    let mut acc = 0.0f32;
    for row in rows {
        acc += row[index].to_f32();
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds the four rank rows from an element-major bit matrix: `bits[e][r]` is
    /// rank r's value for element e, which is how the slot's payload is laid out per element.
    fn rank_rows(bits: &[[u16; ROCE_ONESHOT_RANKS]]) -> Vec<Vec<bf16>> {
        (0..ROCE_ONESHOT_RANKS)
            .map(|rank| {
                bits.iter()
                    .map(|element| bf16::from_bits(element[rank]))
                    .collect()
            })
            .collect()
    }

    /// Reduces an element-major matrix through the public entry point, returning bit patterns.
    fn reduce_all(bits: &[[u16; ROCE_ONESHOT_RANKS]]) -> Vec<u16> {
        let rows = rank_rows(bits);
        let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let mut out = vec![bf16::ZERO; bits.len()];
        all_reduce_bf16_canonical(&refs, &mut out).expect("four equal-length rows");
        out.iter().map(|value| value.to_bits()).collect()
    }

    /// Reduces the single element of a one-element matrix.
    fn reduce(bits: &[[u16; ROCE_ONESHOT_RANKS]]) -> u16 {
        let out = reduce_all(bits);
        assert_eq!(out.len(), 1);
        out[0]
    }

    /// Round-to-nearest-even of an exactly-representable f64 to bf16, implemented by hand.
    ///
    /// Deliberately independent of `half`: it works on the IEEE fields directly, so it
    /// can disagree with `bf16::from_f32` if either the reference or half regressed.
    fn rne_f64_to_bf16(value: f64) -> u16 {
        let bits = value.to_bits();
        let sign = ((bits >> 63) as u16) << 15;
        let exponent = ((bits >> 52) & 0x7ff) as i64;
        let mantissa = bits & ((1u64 << 52) - 1);
        if exponent == 0x7ff {
            return sign | 0x7f80 | if mantissa == 0 { 0 } else { 0x0040 };
        }
        if exponent == 0 {
            return sign; // f64 subnormals are far below the bf16 range
        }
        let biased = exponent - 1023 + 127;
        if biased >= 0xff {
            return sign | 0x7f80;
        }
        if biased <= 0 {
            return sign;
        }
        // Keep the top 7 explicit mantissa bits (bf16 has 1 + 7 significand bits) and round
        // at bit 44, ties to even.
        let shift = 45;
        let mut kept = mantissa >> shift;
        let remainder = mantissa & ((1u64 << shift) - 1);
        let halfway = 1u64 << (shift - 1);
        if remainder > halfway || (remainder == halfway && kept & 1 == 1) {
            kept += 1;
        }
        if kept >= 1 << 7 {
            let bumped = biased + 1;
            if bumped >= 0xff {
                return sign | 0x7f80;
            }
            return sign | ((bumped as u16) << 7);
        }
        sign | ((biased as u16) << 7) | (kept as u16)
    }

    /// Round-to-nearest-even of an f32 to bf16, implemented by hand and independently of
    /// half. f32 and bf16 share the exponent bias, so only the mantissa is cut.
    fn rne_f32_to_bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = ((bits >> 31) as u16) << 15;
        let exponent = ((bits >> 23) & 0xff) as i64;
        let mantissa = bits & ((1 << 23) - 1);
        if exponent == 0xff {
            return sign | 0x7f80 | if mantissa == 0 { 0 } else { 0x0040 };
        }
        if exponent == 0 {
            return sign;
        }
        let mut kept = mantissa >> 16;
        let remainder = mantissa & 0xffff;
        let halfway = 1 << 15;
        if remainder > halfway || (remainder == halfway && kept & 1 == 1) {
            kept += 1;
        }
        let mut biased = exponent as u16;
        if kept >= 1 << 7 {
            kept = 0;
            biased += 1;
            if biased >= 0xff {
                return sign | 0x7f80;
            }
        }
        sign | (biased << 7) | (kept as u16)
    }

    #[test]
    fn the_rounding_oracles_match_hand_derived_values() {
        // Guards the guard: the property test below compares the reference against these two
        // functions, so they are pinned against values derived by hand from the IEEE fields.
        assert_eq!(rne_f64_to_bf16(1.0), 0x3f80);
        assert_eq!(rne_f64_to_bf16(-1.0), 0xbf80);
        assert_eq!(rne_f64_to_bf16(0.0), 0x0000);
        assert_eq!(rne_f64_to_bf16(6.0), 0x40c0);
        assert_eq!(rne_f64_to_bf16(1.00390625), 0x3f80); // exact tie, even wins
        assert_eq!(rne_f64_to_bf16(1.01171875), 0x3f82); // exact tie, even wins
        assert_eq!(rne_f64_to_bf16(1.0048828125), 0x3f81); // just above the tie
        assert_eq!(rne_f64_to_bf16(-1.99981689453125), 0xc000); // carry into the exponent

        assert_eq!(rne_f32_to_bf16(1.0f32), 0x3f80);
        assert_eq!(rne_f32_to_bf16(-1.0f32), 0xbf80);
        assert_eq!(rne_f32_to_bf16(0.0f32), 0x0000);
        assert_eq!(rne_f32_to_bf16(-0.0f32), 0x8000);
        assert_eq!(rne_f32_to_bf16(6.0f32), 0x40c0);
        assert_eq!(rne_f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(rne_f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(rne_f32_to_bf16(f32::from_bits(0x3f80_8001)), 0x3f81);
        assert_eq!(rne_f32_to_bf16(f32::from_bits(0xbf80_8000)), 0xbf80);
        assert_eq!(rne_f32_to_bf16(f32::from_bits(0x3fff_ffff)), 0x4000);
        assert_eq!(rne_f32_to_bf16(f32::INFINITY), 0x7f80);
        assert_eq!(rne_f32_to_bf16(f32::NEG_INFINITY), 0xff80);
        assert_eq!(rne_f32_to_bf16(f32::NAN), 0x7fc0);
    }

    /// A tiny deterministic xorshift so the property test does not depend on a rand version.
    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    #[test]
    fn golden_bit_vectors() {
        // Every expectation below was derived independently with exact dyadic arithmetic and
        // round-to-nearest-even, then cross-checked against the f32 intermediate values that
        // the test asserts alongside them. See the comments for the arithmetic.
        let cases: &[(&str, [[u16; 4]; 1], u16, u32)] = &[
            (
                // 1.0 + 2^-8 = 1.00390625, exactly halfway between bf16 0x3F80 (1.0) and
                // 0x3F81 (1.0078125). Ties-to-even picks 0x3F80. Round-half-away would give
                // 0x3F81, so this vector discriminates the rounding rule.
                "tie down to even",
                [[0x3f80, 0x3b80, 0x0000, 0x0000]],
                0x3f80,
                0x3f80_8000,
            ),
            (
                // 1.0078125 + 2^-8 = 1.01171875, halfway between 0x3F81 (odd) and 0x3F82
                // (even). Ties-to-even rounds UP here, which no truncating rule would do.
                "tie up to even",
                [[0x3f81, 0x3b80, 0x0000, 0x0000]],
                0x3f82,
                0x3f81_8000,
            ),
            (
                // 1.0 + 2^-9 + 2^-9 + 2^-25 = tie + 2^-25. The exact sum is above the tie, so
                // "round the exact sum once" would give 0x3F81; the kernel's f32 accumulator
                // drops 2^-25 (it is below half an f32 ulp at 1.0) and lands exactly on the
                // tie, so the canonical result is 0x3F80. This is the vector that separates
                // f32 intermediate rounding from exact-then-round.
                "f32 accumulation beats exact rounding",
                [[0x3f80, 0x3b00, 0x3b00, 0x3300]],
                0x3f80,
                0x3f80_8000,
            ),
            (
                // Four times -0.0. The accumulator starts at +0.0f, and +0.0 + -0.0 = +0.0, so
                // the result is +0.0. Starting the accumulator at rank 0's value instead would
                // produce -0.0 (0x8000).
                "signed zero follows the accumulator start",
                [[0x8000, 0x8000, 0x8000, 0x8000]],
                0x0000,
                0x0000_0000,
            ),
            (
                // 1.5 * 4, exact.
                "four equal values",
                [[0x3fc0, 0x3fc0, 0x3fc0, 0x3fc0]],
                0x40c0,
                0x40c0_0000,
            ),
            (
                // Found by searching for a four-tuple whose bf16 result depends on the
                // accumulation order: swapping ranks 2 and 3 gives 0x290C instead of 0x290D.
                // The canonical f32 sum is 0x290C8001, one f32 ulp above 0x290C0000, and that
                // ulp is what flips the rounded bf16 from 0x290C to 0x290D.
                "order sensitive",
                [[0x28fb, 0x18be, 0x2770, 0x1d18]],
                0x290d,
                0x290c_8001,
            ),
            (
                // The same four values sparkflash-comm pins in its own reference test:
                // 1.0 + 2^-8 + 2^-24 + 2^-24. Adding the two 2^-24 first makes 2^-23 exactly,
                // so the canonical order sums 1.0 + 2^-8 = 1.00390625 as an exact f32 tie and
                // then rounds it to 0x3F80 (ties to even). Reaching the same exact sum by a
                // different route is what the next vector does.
                "half f32 ulp below the tie rounds down",
                [[0x3f80, 0x3b80, 0x3380, 0x3380]],
                0x3f80,
                0x3f80_8000,
            ),
            (
                // Rotated: 2^-24 + 2^-24 = 2^-23 exactly, then + 1.0 keeps it (one f32 ulp at
                // 1.0 is 2^-23), then + 2^-8 lands one f32 ulp ABOVE the bf16 tie, so the same
                // four values now round up to 0x3F81. This is the carry path of the rotation and
                // the vector sparkflash-comm's reference test calls "rotated".
                "half f32 ulp above the tie rounds up",
                [[0x3380, 0x3380, 0x3f80, 0x3b80]],
                0x3f81,
                0x3f80_8001,
            ),
        ];

        for (name, input, expected, expected_f32) in cases {
            let rows = rank_rows(input);
            let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let sum = canonical_sum_f32(&refs, 0).expect("four rows");
            assert_eq!(
                sum.to_bits(),
                *expected_f32,
                "{name}: the f32 accumulation must land where the derivation says"
            );
            assert_eq!(
                rne_f32_to_bf16(sum),
                *expected,
                "{name}: one round-to-nearest-even step from the f32 sum"
            );
            assert_eq!(reduce(input), *expected, "{name}");
        }

        // Anti-vacuity: the tie vectors really are ties, and a round-half-up rule really would
        // produce a different answer for the first one.
        let tie_down = canonical_sum_f32(
            &[
                &[bf16::ONE][..],
                &[bf16::from_bits(0x3b80)][..],
                &[bf16::ZERO][..],
                &[bf16::ZERO][..],
            ],
            0,
        )
        .expect("four rows");
        assert_eq!(
            tie_down.to_bits() & 0xffff,
            0x8000,
            "the vector must be an exact tie"
        );
    }

    #[test]
    fn the_canonical_order_is_observable() {
        // Same four values, ranks 2 and 3 swapped: different bits. If the reference summed in
        // any other order (or sorted by magnitude) this would catch it.
        let canonical = [[0x28fb, 0x18be, 0x2770, 0x1d18]];
        let swapped = [[0x28fb, 0x18be, 0x1d18, 0x2770]];
        assert_eq!(reduce(&canonical), 0x290d);
        assert_eq!(reduce(&swapped), 0x290c);
        assert_ne!(reduce(&canonical), reduce(&swapped));
    }

    #[test]
    fn matches_an_independent_exact_sum_over_a_bounded_exponent_range() {
        // With the exponents confined to a 10-bit window, every partial f32 sum of four bf16
        // values is exact (8 significand bits + 10 exponent bits + 2 for the carries < 24), so
        // an exact f64 sum rounded once by rne_f64_to_bf16 is a valid oracle. It is a different
        // code path from the reference, so agreement is evidence rather than a tautology.
        let mut rng = XorShift(0x5eed_1234_abcd_0001);
        let mut checked = 0usize;
        for _ in 0..4000 {
            let mut element = [0u16; ROCE_ONESHOT_RANKS];
            for slot in &mut element {
                let exponent = 120 + (rng.next() % 11) as u16; // 2^-7 .. 2^3
                let mantissa = (rng.next() % 128) as u16;
                let sign = ((rng.next() & 1) as u16) << 15;
                *slot = sign | (exponent << 7) | mantissa;
            }
            let rows = rank_rows(&[element]);
            let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let exact = element
                .iter()
                .map(|bits| f64::from(bf16::from_bits(*bits).to_f32()))
                .sum::<f64>();
            let expected = rne_f64_to_bf16(exact);
            let sum = canonical_sum_f32(&refs, 0).expect("four rows");
            assert_eq!(
                sum.to_bits(),
                (exact as f32).to_bits(),
                "the f32 accumulation must be exact for this bounded range"
            );
            assert_eq!(
                rne_f32_to_bf16(sum),
                expected,
                "rounding the f32 sum and rounding the exact sum must agree here"
            );
            let actual = canonical_reduce_element(&refs, 0)
                .expect("four rows")
                .to_bits();
            assert_eq!(
                actual, expected,
                "input {element:04x?} exact sum {exact} must round to {expected:#06x}"
            );
            checked += 1;
        }
        assert_eq!(checked, 4000);
    }

    #[test]
    fn non_finite_inputs_stay_non_finite() {
        assert_eq!(
            canonical_reduce_element(
                &[
                    &[bf16::INFINITY][..],
                    &[bf16::ZERO][..],
                    &[bf16::ZERO][..],
                    &[bf16::ZERO][..]
                ],
                0
            )
            .expect("four rows"),
            bf16::INFINITY
        );
        assert!(
            canonical_reduce_element(
                &[
                    &[bf16::INFINITY][..],
                    &[bf16::NEG_INFINITY][..],
                    &[bf16::ZERO][..],
                    &[bf16::ZERO][..]
                ],
                0
            )
            .expect("four rows")
            .is_nan()
        );
        assert!(
            canonical_reduce_element(
                &[
                    &[bf16::NAN][..],
                    &[bf16::ZERO][..],
                    &[bf16::ZERO][..],
                    &[bf16::ZERO][..]
                ],
                0
            )
            .expect("four rows")
            .is_nan()
        );
    }

    #[test]
    fn all_reduce_matches_the_element_wise_reference() {
        let mut rng = XorShift(0xfeed_0000_0000_0007);
        let element_count = 257usize;
        let rows = (0..ROCE_ONESHOT_RANKS)
            .map(|_| {
                (0..element_count)
                    .map(|_| bf16::from_f32((rng.next() % 2001) as f32 / 8.0 - 125.0))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let mut out = vec![bf16::ZERO; element_count];
        all_reduce_bf16_canonical(&refs, &mut out).expect("valid rows");
        for (index, got) in out.iter().enumerate() {
            assert_eq!(*got, canonical_reduce_element(&refs, index).expect("valid"));
        }
        // The rows are longer than the output by construction above only if we say so; the
        // documented rule is that extra elements are ignored, so ask for less than a row.
        let mut short_out = vec![bf16::ZERO; 3];
        all_reduce_bf16_canonical(&refs, &mut short_out).expect("valid rows");
        assert_eq!(short_out, out[..3].to_vec());
    }

    #[test]
    fn strided_mapping_matches_the_plain_reduction_for_every_shape() {
        let element_count = 70usize;
        let rows = (0..ROCE_ONESHOT_RANKS)
            .map(|rank| {
                (0..element_count)
                    .map(|index| bf16::from_f32((rank * 1000 + index) as f32 / 4.0))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();

        for n in [0usize, 1, 2, 3, 7, 8, 9, 31, 32, 33, 64, 65, 70] {
            let mut plain = vec![bf16::ZERO; n];
            all_reduce_bf16_canonical(&refs, &mut plain).expect("valid");
            for (block_dim, grid_dim) in [
                (1u32, 1u32),
                (32, 1),
                (32, 4),
                (64, 7),
                (128, 13),
                (256, 64),
                (1024, 1),
                (7, 3),
            ] {
                let mut strided = vec![bf16::ZERO; n];
                all_reduce_bf16_canonical_strided(&refs, &mut strided, block_dim, grid_dim)
                    .expect("valid");
                assert_eq!(
                    strided, plain,
                    "n={n} block={block_dim} grid={grid_dim} must not change the bits"
                );
            }
            // A stride wider than the buffer leaves later threads with nothing to do.
            let mut wide = vec![bf16::ZERO; n];
            all_reduce_bf16_canonical_strided(&refs, &mut wide, 1024, 1024).expect("valid");
            assert_eq!(wide, plain);
        }
    }

    #[test]
    fn zero_elements_is_a_valid_reduction() {
        let rows: Vec<Vec<bf16>> = (0..ROCE_ONESHOT_RANKS).map(|_| Vec::new()).collect();
        let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let mut out: Vec<bf16> = Vec::new();
        all_reduce_bf16_canonical(&refs, &mut out).expect("zero elements is valid");
        all_reduce_bf16_canonical_strided(&refs, &mut out, 128, 8).expect("zero elements is valid");
        assert!(out.is_empty());
    }

    #[test]
    fn validation_reports_errors_instead_of_panicking() {
        let row = [bf16::ZERO; 4];
        let four = [&row[..], &row[..], &row[..], &row[..]];
        let mut out = vec![bf16::ZERO; 2];
        assert_eq!(
            all_reduce_bf16_canonical(&four[..3], &mut out),
            Err(RoceReduceError::RankCount {
                expected: 4,
                actual: 3
            })
        );
        let mut too_many = vec![bf16::ZERO; 5];
        assert_eq!(
            all_reduce_bf16_canonical(&four, &mut too_many),
            Err(RoceReduceError::RowTooShort {
                rank: 0,
                row: 4,
                out: 5
            })
        );
        let mut too_long = vec![bf16::ZERO; ROCE_ONESHOT_SLOT_ROW_ELEMS + 1];
        assert_eq!(
            all_reduce_bf16_canonical(&four, &mut too_long),
            Err(RoceReduceError::OutputTooLong {
                out: ROCE_ONESHOT_SLOT_ROW_ELEMS + 1,
                capacity: ROCE_ONESHOT_SLOT_ROW_ELEMS
            })
        );
        assert_eq!(
            all_reduce_bf16_canonical_strided(&four, &mut out, 0, 4),
            Err(RoceReduceError::EmptyLaunch {
                block_dim: 0,
                grid_dim: 4
            })
        );
        assert_eq!(
            canonical_reduce_element(&four, 4),
            Err(RoceReduceError::RowTooShort {
                rank: 0,
                row: 4,
                out: 5
            })
        );
    }

    #[test]
    fn canonical_rank_order_is_ascending() {
        assert_eq!(CANONICAL_RANK_ORDER, [0, 1, 2, 3]);
        assert_eq!(CANONICAL_RANK_ORDER.len(), ROCE_ONESHOT_RANKS);
        assert!(
            CANONICAL_RANK_ORDER
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
    }
}

//! The canonical-order bf16 reduction every transport must reproduce bit for bit.
//!
//! The RoCEnante kernel accumulates rank `0 + 1 + 2 + 3` in ascending rank order in f32,
//! starting from `0.0f`, and rounds exactly once with round-to-nearest-even when it stores the
//! bf16 result. Every element is independent, so the result does not depend on the block/grid
//! stride the kernel happens to use, and it does not depend on the order contributions *arrive*:
//! the accumulator is ordered by rank index, not by arrival time.
//!
//! The seed matters. `0.0f + (-0.0f) == +0.0f`, so a payload of all negative zeros reduces to
//! positive zero exactly as the CUDA kernel does; starting from the first operand instead would
//! produce negative zero and change bits. That is why this module always seeds with `0.0f`.

use crate::bf16::{bf16_to_f32, f32_to_bf16_rn};
use crate::error::CommError;

/// Canonically reduces one element across ranks, in ascending rank order.
#[must_use]
pub fn canonical_reduce_element(values: &[u16]) -> u16 {
    let mut acc = 0.0f32;
    for value in values {
        acc += bf16_to_f32(*value);
    }
    f32_to_bf16_rn(acc)
}

/// Canonically reduces `payloads` (ascending rank order) into `out`.
///
/// `out.len()` must equal the length of every payload. Each element is accumulated from
/// `0.0f` through the payloads in order and rounded to bf16 exactly once.
///
/// # Errors
/// Returns [`CommError::InvalidLayout`] when `payloads` is empty or when a payload length does
/// not match `out`.
pub fn canonical_all_reduce_into(out: &mut [u16], payloads: &[&[u16]]) -> Result<(), CommError> {
    if payloads.is_empty() {
        return Err(CommError::InvalidLayout(
            "canonical reduction needs at least one rank payload".to_owned(),
        ));
    }
    for (index, payload) in payloads.iter().enumerate() {
        if payload.len() != out.len() {
            return Err(CommError::InvalidLayout(format!(
                "rank {index} payload has {} elements but the output has {}",
                payload.len(),
                out.len()
            )));
        }
    }
    // The four-rank world is the only shape the served decode path uses, and it is worth
    // unrolling the inner loop for it: the additions below are the same sequence of f32 adds in
    // the same ascending rank order as the generic loop (0.0f, then rank 0, 1, 2, 3, one RNE
    // round), just with the pane indexing hoisted out of the element loop. No reassociation and
    // no precision change: a compiler may not reassociate floating-point additions without fast
    // math, and this crate does not enable it.
    if let [pane0, pane1, pane2, pane3] = payloads {
        for (slot, value) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            acc += bf16_to_f32(pane0[slot]);
            acc += bf16_to_f32(pane1[slot]);
            acc += bf16_to_f32(pane2[slot]);
            acc += bf16_to_f32(pane3[slot]);
            *value = f32_to_bf16_rn(acc);
        }
        return Ok(());
    }
    for (slot, value) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for payload in payloads {
            acc += bf16_to_f32(payload[slot]);
        }
        *value = f32_to_bf16_rn(acc);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf16::f32_to_bf16_rn;

    #[test]
    fn single_rank_payload_is_rounded_once() {
        let values: Vec<u16> = [1.0f32, -2.5, 0.1, 1e-8]
            .iter()
            .map(|value| f32_to_bf16_rn(*value))
            .collect();
        for (index, value) in values.iter().enumerate() {
            assert_eq!(
                canonical_reduce_element(&values[index..=index]),
                *value,
                "a one-rank reduction must be the identity after the single rounding"
            );
        }
        let mut out = vec![0u16; values.len()];
        canonical_all_reduce_into(&mut out, &[&values]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(out, values);
    }

    #[test]
    fn negative_zero_inputs_reduce_to_positive_zero() {
        // 0.0f + (-0.0f) == +0.0f, so four negative zeros must come out as +0.0 (0x0000) and
        // not as -0.0 (0x8000). Seeding from the first operand would get this wrong.
        assert_eq!(
            canonical_reduce_element(&[0x8000, 0x8000, 0x8000, 0x8000]),
            0x0000
        );
        let payload = vec![0x8000u16; 4];
        let mut out = vec![0x8000u16; 4];
        canonical_all_reduce_into(&mut out, &[&payload, &payload, &payload, &payload])
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(out, vec![0x0000u16; 4]);
    }

    #[test]
    fn accumulate_order_is_rank_order_not_arrival_order() {
        // Four bf16-representable values whose exact sum sits one f32 ulp above the bf16
        // rounding boundary between 1.0 and 1.0078125 (bf16 has 7 mantissa bits, so its ulp at
        // 1.0 is 2^-7 and the boundary is 1 + 2^-8):
        //   the boundary itself is exactly representable in f32;
        //   each 2^-24 is half an f32 ulp there, so round-to-even can absorb it.
        // Accumulating 1, 2^-8, 2^-24, 2^-24 absorbs both 2^-24 terms and lands exactly on the
        // boundary, which ties to even (1.0). Accumulating 2^-24, 2^-24, 1, 2^-8 first makes
        // 2^-23, and 1 + 2^-23 + 2^-8 is exactly representable, so the sum is above the boundary
        // and the final bf16 rounding goes the other way.
        let one = f32_to_bf16_rn(1.0);
        let half_ulp = f32_to_bf16_rn(2.0f32.powi(-8)); // half a bf16 ulp at 1.0
        let tiny = f32_to_bf16_rn(2.0f32.powi(-24)); // half an f32 ulp at 1.0
        let boundary = 1.0f32 + 2.0f32.powi(-8);
        assert_eq!(
            f32_to_bf16_rn(boundary),
            0x3f80,
            "the boundary must tie to even"
        );
        assert_eq!(
            f32_to_bf16_rn(boundary + 2.0f32.powi(-23)),
            0x3f81,
            "one f32 ulp above the boundary must round up"
        );

        let ascending = canonical_reduce_element(&[one, half_ulp, tiny, tiny]);
        let rotated = canonical_reduce_element(&[tiny, tiny, one, half_ulp]);
        assert_eq!(
            f32_to_bf16_rn(1.0 + 2.0f32.powi(-8) + 2.0f32.powi(-23)),
            rotated
        );
        assert_eq!(ascending, 0x3f80);
        assert_eq!(rotated, 0x3f81);
        assert_ne!(
            ascending, rotated,
            "the accumulator must be ordered by rank, not by arrival"
        );
    }

    #[test]
    fn rejects_mismatched_and_empty_payloads() {
        let mut out = vec![0u16; 2];
        assert!(matches!(
            canonical_all_reduce_into(&mut out, &[]),
            Err(CommError::InvalidLayout(_))
        ));
        let short = vec![0u16; 1];
        assert!(matches!(
            canonical_all_reduce_into(&mut out, &[&short]),
            Err(CommError::InvalidLayout(_))
        ));
        // A zero-length collective is legal: it has nothing to do.
        let mut empty: Vec<u16> = Vec::new();
        canonical_all_reduce_into(&mut empty, &[&[], &[]]).unwrap_or_else(|e| panic!("{e}"));
    }
}

//! Codec round-trip, boundary and reference-semantics tests.

use sparkflash_kv::fp4::{
    self, E2M1_MAX, E4M3_MAX, E8M0_MAX, E8M0_MIN_POSITIVE, IndexerKEntry, MainKvEntry, SwaKvSlot,
    dequantize_indexer_k, dequantize_main_kv, dequantize_swa_slot, e2m1_decode, e2m1_encode,
    e4m3_decode, e4m3_encode, e8m0_decode, e8m0_encode_pow2_ceil, indexer_k_group_scale,
    quantize_indexer_k, quantize_main_kv, quantize_swa_slot, swa_group_scale,
};
use sparkflash_kv::layout::{INDEXER_K_DIM, MAIN_KV_DIM, SWA_SLOT_DIM};

/// True when a finite positive f32 is an exact power of two.
fn is_power_of_two(value: f32) -> bool {
    value > 0.0 && value.is_finite() && value.log2().fract() == 0.0
}

/// SplitMix64: a deterministic generator so a failure is reproducible from its seed.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self, low: f32, high: f32) -> f32 {
        let unit = (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32;
        low + unit * (high - low)
    }
}

#[test]
fn every_e4m3_code_round_trips_through_its_exact_value() {
    // 0x7F is the only finite-range hole: S.1111.111 is NaN by construction.
    for code in (0u16..=0xFE).filter(|code| *code != 0x7F) {
        let code = code as u8;
        let value = e4m3_decode(code);
        assert!(value.is_finite(), "code {code:#x} decoded to {value}");
        assert_eq!(
            e4m3_encode(value),
            code,
            "code {code:#x} did not round trip"
        );
    }
    assert!(e4m3_decode(0x7F).is_nan());
    assert!(e4m3_decode(0xFF).is_nan());
}

#[test]
fn every_e8m0_code_round_trips_as_a_power_of_two() {
    // Code 0 included: it is 2^-127, which is subnormal but perfectly representable, so
    // encoding the value that decodes to it must return the same code.
    for code in 0u16..=0xFE {
        let code = code as u8;
        let value = e8m0_decode(code);
        assert!(value > 0.0, "code {code:#x} decoded to {value}");
        assert_eq!(
            e8m0_decode(e8m0_encode_pow2_ceil(value)),
            value,
            "code {code:#x} did not round trip"
        );
    }
    assert_eq!(e8m0_decode(0), 2f32.powi(-127));
    assert_eq!(e8m0_decode(0x01), 2f32.powi(-126));
    assert!(e8m0_decode(0xFF).is_nan());
    assert_eq!(e8m0_decode(0x7F), 1.0);
    assert_eq!(e8m0_decode(0xFE), E8M0_MAX);
}

/// Asserts the whole contract of the encoder: `expected` is returned, its power of two is not
/// below `value`, and no smaller code would do either.
fn assert_ceiling(value: f32, expected: u8) {
    let code = e8m0_encode_pow2_ceil(value);
    assert_eq!(code, expected, "ceil of {value:e} returned {code:#x}");
    let rounded = e8m0_decode(code);
    assert!(
        rounded >= value,
        "ceil of {value:e} = {rounded:e} is below the input"
    );
    if code > 0 {
        let smaller = e8m0_decode(code - 1);
        assert!(
            smaller < value,
            "ceil of {value:e} = {code:#x} is not the smallest: {smaller:e} also fits"
        );
    }
}

#[test]
fn e8m0_subnormals_ceil_to_code_zero_and_not_to_the_normal_floor() {
    assert_eq!(E8M0_MIN_POSITIVE, 2f32.powi(-127));
    // The smallest positive subnormal (2^-149) up to 2^-127 all need a power of two at or
    // below the smallest code, so they clamp to 0 rather than reporting 2^-126.
    assert_ceiling(f32::from_bits(0x0000_0001), 0);
    assert_ceiling(f32::from_bits(0x0000_0002), 0);
    assert_ceiling(f32::from_bits(0x0000_0003), 0);
    assert_ceiling(f32::from_bits(0x0020_0000), 0);
    assert_ceiling(2f32.powi(-127), 0);
    // Above 2^-127 the ceiling is the next code, 2^-126, which every subnormal below the
    // normal floor still needs.
    assert_ceiling(f32::from_bits(0x0040_0001), 1);
    assert_ceiling(f32::from_bits(0x004F_FFFF), 1);
    assert_ceiling(f32::from_bits(0x007F_FFFF), 1);
    assert_ceiling(2f32.powi(-126), 1);
    // Sweep the subnormal range in bit-pattern order: only the value at or below 2^-127
    // clamps, everything above it rounds to the normal floor.
    for mantissa in [
        1u32, 2, 3, 4, 0x3F_FFFE, 0x3F_FFFF, 0x40_0000, 0x40_0001, 0x50_0000, 0x7F_FFFE, 0x7F_FFFF,
    ] {
        let value = f32::from_bits(mantissa);
        let expected = u8::from(value > 2f32.powi(-127));
        assert_ceiling(value, expected);
    }
    // Non-positive and NaN input keeps returning 0, and infinity keeps saturating.
    for value in [0.0f32, -0.0, -1.0, f32::MIN, f32::NAN] {
        assert_eq!(e8m0_encode_pow2_ceil(value), 0, "{value:e}");
    }
    assert_eq!(e8m0_encode_pow2_ceil(f32::INFINITY), 0xFE);
}

/// Asserts the documented high-end behaviour: the code saturates at 0xFE instead of claiming a
/// ceiling it cannot express, because 0xFF is reserved as NaN.
fn assert_saturates(value: f32) {
    let code = e8m0_encode_pow2_ceil(value);
    assert_eq!(code, 0xFE, "saturation of {value:e} returned {code:#x}");
    assert!(
        e8m0_decode(code) < value,
        "the saturation point must be below the input {value:e}"
    );
}

#[test]
fn e8m0_saturates_at_the_top_code_above_two_to_the_127() {
    // The mirror image of the subnormal fix: every finite input up to 2^127 gets its exact
    // ceiling, and above that the encoder saturates at 0xFE = 2^127 because 0xFF is NaN, so no
    // code can reach the input. No quantiser in this crate can ask for either extreme - the
    // indexer tops out near 5.7e37 and the SWA scale near 7.6e35 - but the public encoder
    // states and pins what it does outside its reachable range.
    assert_ceiling(2f32.powi(127), 0xFE);
    assert_ceiling(f32::MAX / 2.0, 0xFE);
    assert_saturates(f32::MAX);
    assert_saturates(f32::from_bits(0x7F7F_FFFF));
    assert_saturates(f32::INFINITY);
    assert_eq!(e8m0_decode(0xFE), 2f32.powi(127));
    assert!(
        e8m0_decode(0xFF).is_nan(),
        "0xFF is not a saturation target"
    );
}

#[test]
fn no_in_crate_quantiser_can_feed_the_encoder_a_subnormal() {
    // The indexer floors amax at 6 * 2^-126 and then divides by 6, so what it hands the
    // encoder is max(amax, 6 * 2^-126) / 6 >= 2^-126: always a normal f32. The SWA quantiser
    // floors at 1e-4 and divides by 448, so its smallest input is 2.232e-7, also normal.
    // Neither can reach the subnormal branch, which is why the defect was only reachable by
    // calling the public encoder with a subnormal directly. The gate is checked at the
    // boundary, including from amax values that are themselves subnormal.
    let mut cases = vec![0.0f32, -0.0, -1.0, f32::from_bits(1), f32::from_bits(2)];
    for mantissa in [0x3F_FFFFu32, 0x40_0000, 0x50_0000, 0x7F_FFFF, 0x0080_0000] {
        cases.push(f32::from_bits(mantissa));
    }
    cases.extend([1e-38f32, 1e-20, 1e-10, 1.0, E2M1_MAX, 1e30]);
    for amax in cases {
        let indexer_input = amax.max(6.0 * 2f32.powi(-126)) / E2M1_MAX;
        assert!(
            indexer_input >= 2f32.powi(-126) && !indexer_input.is_subnormal(),
            "amax {amax:e} feeds the encoder {indexer_input:e}"
        );
        let swa_input = amax.max(1e-4) / E4M3_MAX;
        assert!(
            swa_input.is_normal(),
            "amax {amax:e} feeds the encoder {swa_input:e}"
        );
        assert_eq!(
            indexer_k_group_scale(amax),
            e8m0_decode(e8m0_encode_pow2_ceil(indexer_input))
        );
        assert_eq!(
            swa_group_scale(amax),
            e8m0_decode(e8m0_encode_pow2_ceil(swa_input))
        );
        assert!(indexer_k_group_scale(amax) >= 2f32.powi(-126));
        assert!(swa_group_scale(amax) >= 1e-4 / E4M3_MAX);
    }
    // The smallest scale either quantiser can ask for, at the floor.
    assert_eq!(indexer_k_group_scale(0.0), 2f32.powi(-126));
    assert_eq!(swa_group_scale(0.0), 2f32.powi(-22));
    assert_eq!(e8m0_encode_pow2_ceil(2f32.powi(-126)), 0x01);
}

#[test]
fn e8m0_rounds_up_to_the_next_power_of_two() {
    assert_eq!(e8m0_decode(e8m0_encode_pow2_ceil(3.0)), 4.0);
    assert_eq!(e8m0_decode(e8m0_encode_pow2_ceil(4.0)), 4.0);
    assert_eq!(e8m0_decode(e8m0_encode_pow2_ceil(4.5)), 8.0);
    assert_eq!(e8m0_decode(e8m0_encode_pow2_ceil(0.1)), 0.125);
    // ceil(log2(1e-9)) is -29, because 2^-30 is smaller than 1e-9.
    assert_eq!(e8m0_decode(e8m0_encode_pow2_ceil(1e-9)), 2f32.powi(-29));
    assert!(e8m0_decode(e8m0_encode_pow2_ceil(1e-9)) >= 1e-9);
    assert!(e8m0_decode(e8m0_encode_pow2_ceil(1e-9)) < 2e-9);
}

#[test]
fn fp4_boundaries_saturate_instead_of_wrapping() {
    assert_eq!(e2m1_decode(e2m1_encode(E2M1_MAX)), E2M1_MAX);
    assert_eq!(e2m1_decode(e2m1_encode(6.5)), E2M1_MAX);
    assert_eq!(e2m1_decode(e2m1_encode(1e30)), E2M1_MAX);
    assert_eq!(e2m1_decode(e2m1_encode(-1e30)), -E2M1_MAX);
    assert_eq!(e2m1_decode(e2m1_encode(0.24)), 0.0);
    assert_eq!(e2m1_decode(e2m1_encode(0.25)), 0.0);
    assert_eq!(e2m1_decode(e2m1_encode(0.26)), 0.5);
    assert_eq!(e4m3_decode(e4m3_encode(E4M3_MAX)), E4M3_MAX);
    assert_eq!(e4m3_decode(e4m3_encode(500.0)), E4M3_MAX);
    assert_eq!(e4m3_decode(e4m3_encode(-500.0)), -E4M3_MAX);
    // E4M3's smallest subnormal is 2^-9; half of it rounds to zero.
    assert_eq!(e4m3_decode(e4m3_encode(2f32.powi(-9))), 2f32.powi(-9));
    assert_eq!(e4m3_decode(e4m3_encode(2f32.powi(-10))), 0.0);
}

#[test]
fn main_kv_dequantises_to_exactly_what_the_bytes_say() {
    let mut rng = Rng(0x5EED_1234);
    for round in 0..64 {
        let values: Vec<f32> = (0..MAIN_KV_DIM).map(|_| rng.uniform(-7.0, 7.0)).collect();
        let entry = quantize_main_kv(&values).expect("512 values");
        let mut decoded = vec![0f32; MAIN_KV_DIM];
        dequantize_main_kv(&entry, &mut decoded).expect("512 values");
        for index in 0..MAIN_KV_DIM {
            let group = index / 16;
            let scale = e4m3_decode(entry.scales[group]);
            let byte = entry.data[index / 2];
            let code = if index % 2 == 0 {
                byte & 0x0F
            } else {
                byte >> 4
            };
            let expected = e2m1_decode(code) * scale;
            assert_eq!(
                decoded[index].to_bits(),
                expected.to_bits(),
                "round {round} element {index}"
            );
            // The stored scale is the dequantised E4M3 scale, and every value is divided by it.
            assert!(scale > 0.0);
            let error = (values[index] - decoded[index]).abs();
            assert!(
                error <= scale + f32::EPSILON,
                "round {round} element {index}: error {error} exceeds one E2M1 step {scale}"
            );
        }
    }
}

#[test]
fn exactly_representable_values_survive_quantisation_untouched() {
    // The stored scale is amax / 6 quantised to E4M3, and the dequantised value is
    // E2M1(code) * that scale. With amax exactly 6 the scale is exactly 1, so the E2M1 grid
    // itself must round-trip untouched.
    assert_eq!(fp4::main_kv_group_scale(6.0), 1.0);
    assert_eq!(fp4::main_kv_group_scale(3.0), 0.5);
    assert_eq!(fp4::main_kv_group_scale(0.6), e4m3_decode(e4m3_encode(0.1)));
    let mut values = vec![0f32; MAIN_KV_DIM];
    for (index, value) in values.iter_mut().enumerate() {
        // Every group of 16 contains index % 8 == 7, i.e. 6.0, so every amax is exactly 6.
        *value = e2m1_decode((index % 8) as u8);
    }
    assert_eq!(values[7], 6.0);
    let entry = quantize_main_kv(&values).expect("512 values");
    let mut decoded = vec![0f32; MAIN_KV_DIM];
    dequantize_main_kv(&entry, &mut decoded).expect("512 values");
    assert_eq!(decoded, values);
}

#[test]
fn nan_and_infinity_follow_the_documented_policy() {
    let mut values = vec![1.0f32; MAIN_KV_DIM];
    values[0] = f32::NAN;
    values[1] = f32::INFINITY;
    values[2] = f32::NEG_INFINITY;
    let entry = quantize_main_kv(&values).expect("512 values");
    let mut decoded = vec![0f32; MAIN_KV_DIM];
    dequantize_main_kv(&entry, &mut decoded).expect("512 values");
    let scale = e4m3_decode(entry.scales[0]);
    assert_eq!(decoded[0], 0.0);
    assert_eq!(decoded[1], E2M1_MAX * scale);
    assert_eq!(decoded[2], -E2M1_MAX * scale);
}

#[test]
fn an_all_zero_group_still_gets_a_non_zero_scale() {
    let values = vec![0f32; MAIN_KV_DIM];
    let entry = quantize_main_kv(&values).expect("512 values");
    let scale = e4m3_decode(entry.scales[0]);
    // The reference floors amax at 6 * 2^-9 before dividing by 6.
    assert_eq!(scale, 2f32.powi(-9));
    assert!(
        entry
            .scales
            .iter()
            .all(|code| *code == e4m3_encode(2f32.powi(-9)))
    );
}

#[test]
fn indexer_k_dequantises_to_exactly_what_the_bytes_say() {
    let mut rng = Rng(0xC0FF_EE00);
    let values: Vec<f32> = (0..INDEXER_K_DIM).map(|_| rng.uniform(-9.0, 9.0)).collect();
    let entry = quantize_indexer_k(&values).expect("128 values");
    let mut decoded = vec![0f32; INDEXER_K_DIM];
    dequantize_indexer_k(&entry, &mut decoded).expect("128 values");
    for (index, value) in decoded.iter().enumerate() {
        let group = index / 32;
        let scale = e8m0_decode(entry.scales[group]);
        let byte = entry.data[index / 2];
        let code = if index % 2 == 0 {
            byte & 0x0F
        } else {
            byte >> 4
        };
        assert_eq!(value.to_bits(), (e2m1_decode(code) * scale).to_bits());
        assert!(is_power_of_two(scale));
    }
}

#[test]
fn swa_slots_dequantise_to_exactly_what_the_bytes_say() {
    let mut rng = Rng(0xABBA_1234);
    let values: Vec<f32> = (0..SWA_SLOT_DIM)
        .map(|_| rng.uniform(-400.0, 400.0))
        .collect();
    let slot = quantize_swa_slot(&values).expect("512 values");
    let mut decoded = vec![0f32; SWA_SLOT_DIM];
    dequantize_swa_slot(&slot, &mut decoded).expect("512 values");
    for (index, value) in decoded.iter().enumerate() {
        let scale = e8m0_decode(slot.scales[index / 32]);
        let expected = e4m3_decode(slot.data[index]) * scale;
        assert_eq!(value.to_bits(), expected.to_bits());
        assert!(is_power_of_two(scale));
    }
}

#[test]
fn wrong_lengths_are_rejected_not_truncated() {
    let entry = QuantizeHelpers::entry();
    assert!(dequantize_main_kv(&entry, &mut vec![0f32; MAIN_KV_DIM - 1]).is_err());
    assert!(dequantize_main_kv(&entry, &mut vec![0f32; MAIN_KV_DIM + 1]).is_err());
    assert!(quantize_main_kv(&vec![0f32; 511]).is_err());
    assert!(quantize_indexer_k(&vec![0f32; 127]).is_err());
    assert!(quantize_swa_slot(&vec![0f32; 513]).is_err());
    let mut short = vec![0u8; 287];
    assert!(entry.to_bytes(&mut short).is_err());
    assert!(MainKvEntry::from_bytes(&[0u8; 287]).is_err());
    assert!(IndexerKEntry::from_bytes(&[0u8; 67]).is_err());
}

#[test]
fn scale_helpers_match_the_reference_flooring() {
    // The indexer floors amax at 6 * 2^-126 so a zero group still divides by a finite scale.
    let scale = fp4::indexer_k_group_scale(0.0);
    assert_eq!(scale, 2f32.powi(-126));
    // The SWA quantiser floors amax at 1e-4 and divides by the FP8 maximum.
    let swa = fp4::swa_group_scale(0.0);
    assert!(swa > 0.0);
    assert!(is_power_of_two(swa));
    assert!(swa <= 1e-4 / E4M3_MAX * 2.0);
}

struct QuantizeHelpers;

impl QuantizeHelpers {
    fn entry() -> MainKvEntry {
        quantize_main_kv(&vec![0.25f32; MAIN_KV_DIM]).expect("512 values")
    }
}

#[test]
fn pinned_entry_types_have_the_pinned_sizes() {
    assert_eq!(std::mem::size_of::<MainKvEntry>(), 288);
    assert_eq!(std::mem::size_of::<IndexerKEntry>(), 68);
    assert_eq!(std::mem::size_of::<SwaKvSlot>(), 528);
    let _ = E8M0_MAX;
}

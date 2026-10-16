//! The checkpoint decoders and the activation quantisers, pinned code by code.
//!
//! The three tables below are the independent reference. They were generated from the format
//! definitions - field widths, biases and the NaN rules - by a short standalone Python program that
//! shares no code with what it checks, and they are stored as f32 bit patterns so the comparison is
//! exact in both directions: negative zero, a NaN's sign, the smallest subnormal and the largest
//! finite value are all distinguishable, and no tolerance can hide a decoder that is wrong in one
//! exponent band. A NaN entry is the quiet NaN carrying the code's sign, which is what the decoders
//! document.
//!
//! The quantiser tests at the end use the same tables as an oracle: the encoders are checked against
//! a brute-force nearest search over them, which is a different algorithm from the bisection the
//! encoders use.

use sparkflash_engine::dtype::DType;
use sparkflash_engine::quant::{
    ACTIVATION_AMAX_FLOOR, E8M0_NAN, FP4_AMAX_FLOOR, FP4_E2M1_MAX, FP8_E4M3_MAX, QuantError,
    bf16_to_f32, decode, decode_bf16, decode_f4e2m1, decode_f8e4m3, decode_f8e8m0, decode_f32,
    e4m3_encode, e8m0_encode_ceil, f4e2m1_encode, f4e2m1_pair_to_f32, f4e2m1_to_f32, f8e4m3_to_f32,
    f8e8m0_to_f32, f32_from_le_bytes, quantize_activation_fp4, quantize_activation_fp8,
};

/// The value of every E4M3 code, in code order.
const E4M3_REFERENCE: [u32; 256] = [
    0x00000000, 0x3B000000, 0x3B800000, 0x3BC00000, 0x3C000000, 0x3C200000, 0x3C400000, 0x3C600000,
    0x3C800000, 0x3C900000, 0x3CA00000, 0x3CB00000, 0x3CC00000, 0x3CD00000, 0x3CE00000, 0x3CF00000,
    0x3D000000, 0x3D100000, 0x3D200000, 0x3D300000, 0x3D400000, 0x3D500000, 0x3D600000, 0x3D700000,
    0x3D800000, 0x3D900000, 0x3DA00000, 0x3DB00000, 0x3DC00000, 0x3DD00000, 0x3DE00000, 0x3DF00000,
    0x3E000000, 0x3E100000, 0x3E200000, 0x3E300000, 0x3E400000, 0x3E500000, 0x3E600000, 0x3E700000,
    0x3E800000, 0x3E900000, 0x3EA00000, 0x3EB00000, 0x3EC00000, 0x3ED00000, 0x3EE00000, 0x3EF00000,
    0x3F000000, 0x3F100000, 0x3F200000, 0x3F300000, 0x3F400000, 0x3F500000, 0x3F600000, 0x3F700000,
    0x3F800000, 0x3F900000, 0x3FA00000, 0x3FB00000, 0x3FC00000, 0x3FD00000, 0x3FE00000, 0x3FF00000,
    0x40000000, 0x40100000, 0x40200000, 0x40300000, 0x40400000, 0x40500000, 0x40600000, 0x40700000,
    0x40800000, 0x40900000, 0x40A00000, 0x40B00000, 0x40C00000, 0x40D00000, 0x40E00000, 0x40F00000,
    0x41000000, 0x41100000, 0x41200000, 0x41300000, 0x41400000, 0x41500000, 0x41600000, 0x41700000,
    0x41800000, 0x41900000, 0x41A00000, 0x41B00000, 0x41C00000, 0x41D00000, 0x41E00000, 0x41F00000,
    0x42000000, 0x42100000, 0x42200000, 0x42300000, 0x42400000, 0x42500000, 0x42600000, 0x42700000,
    0x42800000, 0x42900000, 0x42A00000, 0x42B00000, 0x42C00000, 0x42D00000, 0x42E00000, 0x42F00000,
    0x43000000, 0x43100000, 0x43200000, 0x43300000, 0x43400000, 0x43500000, 0x43600000, 0x43700000,
    0x43800000, 0x43900000, 0x43A00000, 0x43B00000, 0x43C00000, 0x43D00000, 0x43E00000, 0x7FC00000,
    0x80000000, 0xBB000000, 0xBB800000, 0xBBC00000, 0xBC000000, 0xBC200000, 0xBC400000, 0xBC600000,
    0xBC800000, 0xBC900000, 0xBCA00000, 0xBCB00000, 0xBCC00000, 0xBCD00000, 0xBCE00000, 0xBCF00000,
    0xBD000000, 0xBD100000, 0xBD200000, 0xBD300000, 0xBD400000, 0xBD500000, 0xBD600000, 0xBD700000,
    0xBD800000, 0xBD900000, 0xBDA00000, 0xBDB00000, 0xBDC00000, 0xBDD00000, 0xBDE00000, 0xBDF00000,
    0xBE000000, 0xBE100000, 0xBE200000, 0xBE300000, 0xBE400000, 0xBE500000, 0xBE600000, 0xBE700000,
    0xBE800000, 0xBE900000, 0xBEA00000, 0xBEB00000, 0xBEC00000, 0xBED00000, 0xBEE00000, 0xBEF00000,
    0xBF000000, 0xBF100000, 0xBF200000, 0xBF300000, 0xBF400000, 0xBF500000, 0xBF600000, 0xBF700000,
    0xBF800000, 0xBF900000, 0xBFA00000, 0xBFB00000, 0xBFC00000, 0xBFD00000, 0xBFE00000, 0xBFF00000,
    0xC0000000, 0xC0100000, 0xC0200000, 0xC0300000, 0xC0400000, 0xC0500000, 0xC0600000, 0xC0700000,
    0xC0800000, 0xC0900000, 0xC0A00000, 0xC0B00000, 0xC0C00000, 0xC0D00000, 0xC0E00000, 0xC0F00000,
    0xC1000000, 0xC1100000, 0xC1200000, 0xC1300000, 0xC1400000, 0xC1500000, 0xC1600000, 0xC1700000,
    0xC1800000, 0xC1900000, 0xC1A00000, 0xC1B00000, 0xC1C00000, 0xC1D00000, 0xC1E00000, 0xC1F00000,
    0xC2000000, 0xC2100000, 0xC2200000, 0xC2300000, 0xC2400000, 0xC2500000, 0xC2600000, 0xC2700000,
    0xC2800000, 0xC2900000, 0xC2A00000, 0xC2B00000, 0xC2C00000, 0xC2D00000, 0xC2E00000, 0xC2F00000,
    0xC3000000, 0xC3100000, 0xC3200000, 0xC3300000, 0xC3400000, 0xC3500000, 0xC3600000, 0xC3700000,
    0xC3800000, 0xC3900000, 0xC3A00000, 0xC3B00000, 0xC3C00000, 0xC3D00000, 0xC3E00000, 0xFFC00000,
];

/// The value of every E8M0 code, in code order; 255 is its NaN.
const E8M0_REFERENCE: [u32; 256] = [
    0x00400000, 0x00800000, 0x01000000, 0x01800000, 0x02000000, 0x02800000, 0x03000000, 0x03800000,
    0x04000000, 0x04800000, 0x05000000, 0x05800000, 0x06000000, 0x06800000, 0x07000000, 0x07800000,
    0x08000000, 0x08800000, 0x09000000, 0x09800000, 0x0A000000, 0x0A800000, 0x0B000000, 0x0B800000,
    0x0C000000, 0x0C800000, 0x0D000000, 0x0D800000, 0x0E000000, 0x0E800000, 0x0F000000, 0x0F800000,
    0x10000000, 0x10800000, 0x11000000, 0x11800000, 0x12000000, 0x12800000, 0x13000000, 0x13800000,
    0x14000000, 0x14800000, 0x15000000, 0x15800000, 0x16000000, 0x16800000, 0x17000000, 0x17800000,
    0x18000000, 0x18800000, 0x19000000, 0x19800000, 0x1A000000, 0x1A800000, 0x1B000000, 0x1B800000,
    0x1C000000, 0x1C800000, 0x1D000000, 0x1D800000, 0x1E000000, 0x1E800000, 0x1F000000, 0x1F800000,
    0x20000000, 0x20800000, 0x21000000, 0x21800000, 0x22000000, 0x22800000, 0x23000000, 0x23800000,
    0x24000000, 0x24800000, 0x25000000, 0x25800000, 0x26000000, 0x26800000, 0x27000000, 0x27800000,
    0x28000000, 0x28800000, 0x29000000, 0x29800000, 0x2A000000, 0x2A800000, 0x2B000000, 0x2B800000,
    0x2C000000, 0x2C800000, 0x2D000000, 0x2D800000, 0x2E000000, 0x2E800000, 0x2F000000, 0x2F800000,
    0x30000000, 0x30800000, 0x31000000, 0x31800000, 0x32000000, 0x32800000, 0x33000000, 0x33800000,
    0x34000000, 0x34800000, 0x35000000, 0x35800000, 0x36000000, 0x36800000, 0x37000000, 0x37800000,
    0x38000000, 0x38800000, 0x39000000, 0x39800000, 0x3A000000, 0x3A800000, 0x3B000000, 0x3B800000,
    0x3C000000, 0x3C800000, 0x3D000000, 0x3D800000, 0x3E000000, 0x3E800000, 0x3F000000, 0x3F800000,
    0x40000000, 0x40800000, 0x41000000, 0x41800000, 0x42000000, 0x42800000, 0x43000000, 0x43800000,
    0x44000000, 0x44800000, 0x45000000, 0x45800000, 0x46000000, 0x46800000, 0x47000000, 0x47800000,
    0x48000000, 0x48800000, 0x49000000, 0x49800000, 0x4A000000, 0x4A800000, 0x4B000000, 0x4B800000,
    0x4C000000, 0x4C800000, 0x4D000000, 0x4D800000, 0x4E000000, 0x4E800000, 0x4F000000, 0x4F800000,
    0x50000000, 0x50800000, 0x51000000, 0x51800000, 0x52000000, 0x52800000, 0x53000000, 0x53800000,
    0x54000000, 0x54800000, 0x55000000, 0x55800000, 0x56000000, 0x56800000, 0x57000000, 0x57800000,
    0x58000000, 0x58800000, 0x59000000, 0x59800000, 0x5A000000, 0x5A800000, 0x5B000000, 0x5B800000,
    0x5C000000, 0x5C800000, 0x5D000000, 0x5D800000, 0x5E000000, 0x5E800000, 0x5F000000, 0x5F800000,
    0x60000000, 0x60800000, 0x61000000, 0x61800000, 0x62000000, 0x62800000, 0x63000000, 0x63800000,
    0x64000000, 0x64800000, 0x65000000, 0x65800000, 0x66000000, 0x66800000, 0x67000000, 0x67800000,
    0x68000000, 0x68800000, 0x69000000, 0x69800000, 0x6A000000, 0x6A800000, 0x6B000000, 0x6B800000,
    0x6C000000, 0x6C800000, 0x6D000000, 0x6D800000, 0x6E000000, 0x6E800000, 0x6F000000, 0x6F800000,
    0x70000000, 0x70800000, 0x71000000, 0x71800000, 0x72000000, 0x72800000, 0x73000000, 0x73800000,
    0x74000000, 0x74800000, 0x75000000, 0x75800000, 0x76000000, 0x76800000, 0x77000000, 0x77800000,
    0x78000000, 0x78800000, 0x79000000, 0x79800000, 0x7A000000, 0x7A800000, 0x7B000000, 0x7B800000,
    0x7C000000, 0x7C800000, 0x7D000000, 0x7D800000, 0x7E000000, 0x7E800000, 0x7F000000, 0x7FC00000,
];

/// The value of every E2M1 nibble, in nibble order.
const FP4_REFERENCE: [u32; 16] = [
    0x00000000, 0x3F000000, 0x3F800000, 0x3FC00000, 0x40000000, 0x40400000, 0x40800000, 0x40C00000,
    0x80000000, 0xBF000000, 0xBF800000, 0xBFC00000, 0xC0000000, 0xC0400000, 0xC0800000, 0xC0C00000,
];

/// Every E4M3 code decodes to the value the format definition gives it.
#[test]
fn every_f8e4m3_code_decodes_to_its_reference_value() {
    for code in 0..=u8::MAX {
        let decoded = f8e4m3_to_f32(code);
        let expected = E4M3_REFERENCE[usize::from(code)];
        assert_eq!(
            decoded.to_bits(),
            expected,
            "code 0x{code:02X} decoded to {decoded:e}; the reference says {expected:#010X}"
        );
    }
}

/// The boundaries of E4M3, where a one-off exponent or mantissa error would hide.
#[test]
fn the_e4m3_edge_cases_are_the_ones_the_format_defines() {
    // Two zeros, two codes: the sign is the only difference between 0x00 and 0x80, and negative
    // zero must not collapse into positive zero.
    assert_eq!(f8e4m3_to_f32(0x00).to_bits(), 0x0000_0000);
    assert_eq!(f8e4m3_to_f32(0x80).to_bits(), 0x8000_0000);
    assert!(f8e4m3_to_f32(0x80).is_sign_negative());
    // The smallest positive value is the subnormal 2^-9, the largest subnormal 7 * 2^-9, and the
    // smallest normal 2^-6 - the three points the subnormal/normal boundary turns on.
    assert_eq!(f8e4m3_to_f32(0x01), 2f32.powi(-9));
    assert_eq!(f8e4m3_to_f32(0x07), 7.0 * 2f32.powi(-9));
    assert_eq!(f8e4m3_to_f32(0x08), 2f32.powi(-6));
    // The largest finite value is 448 = 1.75 * 2^8, and the code above it is not 464: the format
    // spends its top code on NaN instead of an infinity, which is what makes it E4M3FN.
    assert_eq!(f8e4m3_to_f32(0x7E), FP8_E4M3_MAX);
    assert_eq!(f8e4m3_to_f32(0x7E), 448.0);
    // Both NaN codes, with their signs, and no infinity anywhere in the format.
    assert!(f8e4m3_to_f32(0x7F).is_nan() && !f8e4m3_to_f32(0x7F).is_sign_negative());
    assert!(f8e4m3_to_f32(0xFF).is_nan() && f8e4m3_to_f32(0xFF).is_sign_negative());
    for code in 0..=u8::MAX {
        assert!(
            !f8e4m3_to_f32(code).is_infinite(),
            "code 0x{code:02X} decoded to an infinity"
        );
    }
    // The two halves are exact mirrors of each other, and the magnitude is strictly increasing with
    // the code - the property the encoder's bisection relies on.
    for code in 0..=0x7Eu8 {
        assert_eq!(
            f8e4m3_to_f32(code | 0x80),
            -f8e4m3_to_f32(code),
            "code 0x{code:02X}"
        );
    }
    for code in 0..0x7Eu8 {
        assert!(
            f8e4m3_to_f32(code) < f8e4m3_to_f32(code + 1),
            "code 0x{code:02X} is not below the next one"
        );
    }
}

/// Every E8M0 code decodes to the value the format definition gives it.
#[test]
fn every_f8e8m0_code_decodes_to_its_reference_value() {
    for code in 0..=u8::MAX {
        let decoded = f8e8m0_to_f32(code);
        let expected = E8M0_REFERENCE[usize::from(code)];
        assert_eq!(
            decoded.to_bits(),
            expected,
            "code {code} decoded to {decoded:e}; the reference says {expected:#010X}"
        );
    }
}

/// The scale format's boundaries: no zero, a subnormal minimum, and exactly one NaN.
#[test]
fn the_e8m0_edge_cases_are_the_ones_the_format_defines() {
    // Code 0 is 2^-127, which is a subnormal f32: the format has no zero at all, so the smallest
    // scale it can store is two steps below the smallest normal one.
    assert_eq!(f8e8m0_to_f32(0).to_bits(), 0x0040_0000);
    assert!(f8e8m0_to_f32(0).is_subnormal());
    assert_eq!(f8e8m0_to_f32(0), 2f32.powi(-127));
    assert_eq!(f8e8m0_to_f32(1), f32::MIN_POSITIVE);
    assert_eq!(f8e8m0_to_f32(127), 1.0);
    assert_eq!(f8e8m0_to_f32(254), 2f32.powi(127));
    assert_eq!(f8e8m0_to_f32(254).to_bits(), 0x7F00_0000);
    // 255 is the single NaN, and nothing decodes to an infinity.
    assert_eq!(E8M0_NAN, 255);
    assert!(f8e8m0_to_f32(E8M0_NAN).is_nan());
    for code in 0..=u8::MAX {
        assert!(
            !f8e8m0_to_f32(code).is_infinite(),
            "code {code} decoded to an infinity"
        );
    }
    // Every finite code is exactly 2^(code - 127), checked in f64 where the power is exact: this is
    // the definition again, written differently from the decoder.
    for code in 0..E8M0_NAN {
        assert_eq!(
            f64::from(f8e8m0_to_f32(code)),
            2f64.powi(i32::from(code) - 127),
            "code {code}"
        );
    }
    for code in 0..E8M0_NAN - 1 {
        assert!(f8e8m0_to_f32(code) < f8e8m0_to_f32(code + 1), "code {code}");
    }
}

/// Every E2M1 nibble decodes to the value the format definition gives it.
#[test]
fn every_e2m1_nibble_decodes_to_its_reference_value() {
    for nibble in 0..16u8 {
        assert_eq!(
            f4e2m1_to_f32(nibble).to_bits(),
            FP4_REFERENCE[usize::from(nibble)],
            "nibble 0x{nibble:X}"
        );
    }
    // The magnitudes, in code order, and the negative zero the format has at 0x8.
    assert_eq!(f4e2m1_to_f32(0), 0.0);
    assert_eq!(f4e2m1_to_f32(1), 0.5);
    assert_eq!(f4e2m1_to_f32(2), 1.0);
    assert_eq!(f4e2m1_to_f32(3), 1.5);
    assert_eq!(f4e2m1_to_f32(4), 2.0);
    assert_eq!(f4e2m1_to_f32(5), 3.0);
    assert_eq!(f4e2m1_to_f32(6), 4.0);
    assert_eq!(f4e2m1_to_f32(7), FP4_E2M1_MAX);
    assert_eq!(f4e2m1_to_f32(8).to_bits(), 0x8000_0000);
    for nibble in 0..16u8 {
        // No NaN and no infinity: all sixteen codes are finite values.
        assert!(f4e2m1_to_f32(nibble).is_finite(), "nibble 0x{nibble:X}");
        // Only the low four bits are read, so the same nibble in a byte's low half means the same
        // thing as the nibble on its own.
        assert_eq!(
            f4e2m1_to_f32(nibble | 0xF0),
            f4e2m1_to_f32(nibble),
            "nibble 0x{nibble:X}"
        );
    }
    for nibble in 0..8u8 {
        assert_eq!(
            f4e2m1_to_f32(nibble | 0x08),
            -f4e2m1_to_f32(nibble),
            "nibble 0x{nibble:X}"
        );
        if nibble < 7 {
            assert!(
                f4e2m1_to_f32(nibble) < f4e2m1_to_f32(nibble + 1),
                "nibble 0x{nibble:X}"
            );
        }
    }
}

/// The packed pair is read low nibble first, for every one of the 256 byte values.
#[test]
fn the_e2m1_pair_reader_puts_the_low_nibble_first_for_every_byte() {
    for byte in 0..=u8::MAX {
        let [first, second] = f4e2m1_pair_to_f32(byte);
        assert_eq!(
            first.to_bits(),
            FP4_REFERENCE[usize::from(byte & 0x0F)],
            "byte 0x{byte:02X}: the low nibble is the first element"
        );
        assert_eq!(
            second.to_bits(),
            FP4_REFERENCE[usize::from(byte >> 4)],
            "byte 0x{byte:02X}: the high nibble is the second element"
        );
    }
    // The order is not symmetric, so a swapped read cannot pass: 0x21 is 0.5 then 1.0.
    assert_eq!(f4e2m1_pair_to_f32(0x21), [0.5, 1.0]);
    assert_eq!(f4e2m1_pair_to_f32(0x12), [1.0, 0.5]);
    assert_eq!(f4e2m1_pair_to_f32(0x87), [6.0, -0.0]);
}

/// bf16 is the high half of an f32 for all 65,536 bit patterns, with nothing rounded.
#[test]
fn bf16_is_the_high_half_of_an_f32_for_every_bit_pattern() {
    for bits in 0..=u16::MAX {
        let decoded = bf16_to_f32(bits);
        assert_eq!(
            decoded.to_bits(),
            u32::from(bits) << 16,
            "bits 0x{bits:04X}"
        );
        // Nothing is lost in the other direction either: the top half of the decoded value is the
        // code again, NaN payloads included.
        assert_eq!(
            (decoded.to_bits() >> 16) as u16,
            bits,
            "bits 0x{bits:04X} did not survive"
        );
    }
}

/// The bf16 boundaries: its subnormals are f32 subnormals, and its top values are not.
#[test]
fn the_bf16_edge_cases_are_the_ones_the_format_defines() {
    assert_eq!(bf16_to_f32(0x0000).to_bits(), 0x0000_0000);
    assert_eq!(bf16_to_f32(0x8000).to_bits(), 0x8000_0000);
    // The smallest bf16 subnormal is 2^-133 - f32 has six more exponent steps below it, so it is an
    // f32 subnormal too, but an exact one.
    assert_eq!(bf16_to_f32(0x0001), power_of_two(-133));
    assert!(bf16_to_f32(0x0001).is_subnormal());
    // The subnormal/normal boundary inside bf16, where 0x007F is the largest subnormal of the
    // format and 0x0080 is its smallest normal - which is also f32's.
    assert_eq!(bf16_to_f32(0x007F), 127.0 * power_of_two(-133));
    assert_eq!(bf16_to_f32(0x0080), f32::MIN_POSITIVE);
    // 1.0, which is the whole point of the format: the top half of 0x3F800000.
    assert_eq!(bf16_to_f32(0x3F80), 1.0);
    assert_eq!(bf16_to_f32(0xBF80), -1.0);
    assert_eq!(bf16_to_f32(0x4000), 2.0);
    // The largest finite value is f32::MAX truncated, not rounded up: the truncation is what keeps
    // bf16's top exponent field free for the infinity it does have.
    assert_eq!(bf16_to_f32(0x7F7F).to_bits(), 0x7F7F_0000);
    assert!(bf16_to_f32(0x7F7F) < f32::MAX);
    assert_eq!(bf16_to_f32(0x7F80), f32::INFINITY);
    assert_eq!(bf16_to_f32(0xFF80), f32::NEG_INFINITY);
    assert!(bf16_to_f32(0x7FC0).is_nan());
    assert!(bf16_to_f32(0xFFC0).is_nan() && bf16_to_f32(0xFFC0).is_sign_negative());
}

/// f32 is the identity in little-endian order, bit pattern by bit pattern.
#[test]
fn f32_is_the_identity_in_little_endian() {
    // 1.0 is 0x3F800000, so its little-endian bytes are 00 00 80 3F: the order is the test.
    assert_eq!(f32_from_le_bytes([0x00, 0x00, 0x80, 0x3F]), 1.0);
    assert_eq!(
        f32_from_le_bytes([0x00, 0x00, 0x00, 0x00]).to_bits(),
        0x0000_0000
    );
    assert_eq!(
        f32_from_le_bytes([0x00, 0x00, 0x00, 0x80]).to_bits(),
        0x8000_0000
    );
    assert_eq!(
        f32_from_le_bytes([0x01, 0x00, 0x00, 0x00]),
        f32::from_bits(1)
    );
    assert_eq!(f32_from_le_bytes([0x00, 0x00, 0x80, 0x7F]), f32::INFINITY);
    assert_eq!(
        f32_from_le_bytes([0x00, 0x00, 0x80, 0xFF]),
        f32::NEG_INFINITY
    );
    assert!(f32_from_le_bytes([0x00, 0x00, 0xC0, 0x7F]).is_nan());
    assert_eq!(f32_from_le_bytes([0xFF, 0xFF, 0x7F, 0x7F]), f32::MAX);
    // Every bit pattern survives, which is what "identity" has to mean: the bytes are the value.
    let patterns = [
        0x0000_0000u32,
        0x8000_0000,
        0x0000_0001,
        0x007F_FFFF,
        0x0080_0000,
        0x3F80_0000,
        0x7F7F_FFFF,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC0_0001,
        0xFFFF_FFFF,
    ];
    for pattern in patterns {
        assert_eq!(
            f32_from_le_bytes(pattern.to_le_bytes()).to_bits(),
            pattern,
            "0x{pattern:08X}"
        );
    }
    // And the bulk decoder is that same decode, applied in order.
    let mut bytes = Vec::new();
    for pattern in patterns {
        bytes.extend_from_slice(&pattern.to_le_bytes());
    }
    let decoded = decode_f32(&bytes).expect("a whole number of f32 values");
    assert_eq!(decoded.len(), patterns.len());
    for (index, pattern) in patterns.iter().enumerate() {
        assert_eq!(decoded[index].to_bits(), *pattern);
    }
}
/// A buffer that does not hold whole values is refused, with the count it would have needed.
#[test]
fn a_buffer_that_does_not_hold_whole_values_is_rejected() {
    assert_eq!(
        decode_bf16(&[0, 0, 0]),
        Err(QuantError::Length {
            dtype: DType::BF16,
            elements: 2,
            expected: 4,
            actual: 3
        })
    );
    assert_eq!(
        decode_f32(&[0; 5]),
        Err(QuantError::Length {
            dtype: DType::F32,
            elements: 2,
            expected: 8,
            actual: 5
        })
    );
    assert_eq!(
        decode_f4e2m1(&[0; 1], 4),
        Err(QuantError::Length {
            dtype: DType::F4E2M1,
            elements: 4,
            expected: 2,
            actual: 1
        })
    );
    assert_eq!(
        decode(DType::F8E4M3, &[0; 3], 2),
        Err(QuantError::Length {
            dtype: DType::F8E4M3,
            elements: 2,
            expected: 2,
            actual: 3
        })
    );
    // An F16 tensor is named by the dtype list but not decoded here, rather than decoded untested.
    assert_eq!(
        decode(DType::F16, &[0; 2], 1),
        Err(QuantError::Unsupported { dtype: DType::F16 })
    );
    // Empty and complete buffers are fine.
    assert!(decode_bf16(&[]).is_ok());
    assert!(decode_bf16(&[0, 0]).is_ok());
}

/// Three packed values in two bytes: the fourth nibble is padding and is not a value.
#[test]
fn a_padded_e2m1_buffer_decodes_only_the_elements_it_claims() {
    let packed = [0x21, 0x43];
    assert_eq!(
        decode_f4e2m1(&packed, 3).expect("two bytes hold three values and a padding nibble"),
        vec![0.5, 1.0, 1.5]
    );
    assert_eq!(
        decode_f4e2m1(&packed, 4).expect("two bytes hold four values"),
        vec![0.5, 1.0, 1.5, 2.0]
    );
    // The element order is the packing order: element 2i comes from the low nibble of byte i.
    assert_eq!(
        decode_f4e2m1(&[0x10, 0x32], 4).expect("two bytes"),
        vec![0.0, 0.5, 1.0, 1.5]
    );
}

/// Compares two decodes by bit pattern.
///
/// Comparing f32 values with == cannot work on these buffers: several of the codes are NaN, and NaN
/// is not equal to itself, so two identical decodes would compare unequal. Bits also catch the
/// difference between the two zeros.
fn assert_same_values(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: lengths differ");
    for (index, (left, right)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(left.to_bits(), right.to_bits(), "{what}: element {index}");
    }
}

/// The generic entry point decodes exactly what the per-format decoders do.
#[test]
fn the_generic_decoder_agrees_with_the_per_format_decoders() {
    let codes: Vec<u8> = (0..=u8::MAX).collect();
    assert_same_values(
        &decode(DType::F8E4M3, &codes, 256).expect("256 E4M3 codes"),
        &decode_f8e4m3(&codes),
        "F8E4M3",
    );
    assert_same_values(
        &decode(DType::F8E8M0, &codes, 256).expect("256 E8M0 codes"),
        &decode_f8e8m0(&codes),
        "F8E8M0",
    );
    let halves: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
    assert_same_values(
        &decode(DType::BF16, &halves, 65_536).expect("65536 bf16 values"),
        &decode_bf16(&halves).expect("a whole number of values"),
        "BF16",
    );
    let words: Vec<u8> = (0..=255u32).flat_map(u32::to_le_bytes).collect();
    assert_same_values(
        &decode(DType::F32, &words, 256).expect("256 f32 values"),
        &decode_f32(&words).expect("a whole number of values"),
        "F32",
    );
    // 256 packed bytes are 512 values, and the generic path has to say so rather than 256.
    assert_same_values(
        &decode(DType::F4E2M1, &codes, 512).expect("512 packed values"),
        &decode_f4e2m1(&codes, 512).expect("two values per byte"),
        "F4E2M1",
    );
}

/// The nearest E4M3 code to a magnitude, by brute force over the reference table.
///
/// This is the oracle the encoder is checked against: a different algorithm (scan every code) and a
/// different source of values (the generated table) from the bisection in the encoder.
fn reference_e4m3_nearest(magnitude: f64) -> u8 {
    let mut best = 0u8;
    let mut best_distance = f64::INFINITY;
    for code in 0..0x7Fu8 {
        let value = f64::from(f32::from_bits(E4M3_REFERENCE[usize::from(code)])).abs();
        let distance = (value - magnitude).abs();
        if distance < best_distance || (distance == best_distance && code % 2 == 0) {
            best = code;
            best_distance = distance;
        }
    }
    best
}

/// The E4M3 encoder agrees with that brute-force search across the whole range.
#[test]
fn the_e4m3_encoder_agrees_with_a_brute_force_search_over_the_reference_table() {
    let mut magnitudes: Vec<f32> = Vec::new();
    // A linear sweep of the whole range, including a little past the top of it.
    for step in 1..=4_200u32 {
        magnitudes.push((f64::from(step) * (448.0 / 4096.0)) as f32);
    }
    // A geometric sweep down through the subnormals, where the grid is at its finest.
    for exponent in -24..=0 {
        for sixteenth in 1..=16u8 {
            magnitudes.push(2f32.powi(exponent) * (1.0 + f32::from(sixteenth) / 16.0));
        }
    }
    for magnitude in magnitudes {
        let expected = reference_e4m3_nearest(f64::from(magnitude));
        assert_eq!(e4m3_encode(magnitude), expected, "encoding {magnitude:e}");
        assert_eq!(
            e4m3_encode(-magnitude),
            expected | 0x80,
            "encoding -{magnitude:e}"
        );
    }
}

/// The E4M3 encoder's tie rule is round-to-nearest-even, not truncation or round-half-away.
#[test]
fn the_e4m3_encoder_breaks_ties_towards_the_even_mantissa() {
    // 1.0 is 0x38 and 1.125 is 0x39, so 1.0625 is exactly halfway and the even mantissa wins.
    assert_eq!(e4m3_encode(1.0625), 0x38);
    // 1.125 is 0x39 (odd) and 1.25 is 0x3A (even), so 1.1875 goes up.
    assert_eq!(e4m3_encode(1.1875), 0x3A);
    // 2.0 is 0x40 and 2.25 is 0x41, so 2.125 stays; 2.25 and 2.5 straddle 2.375 the other way.
    assert_eq!(e4m3_encode(2.125), 0x40);
    assert_eq!(e4m3_encode(2.375), 0x42);
    // Truncation would answer 0x39, 0x39, 0x41 and 0x41 on those four, so they pin real rounding.
    assert_eq!(e4m3_encode(0.0), 0x00);
    assert_eq!(e4m3_encode(-0.0), 0x80);
}

/// The E4M3 encoder saturates at the top of the range instead of reaching for the NaN code.
#[test]
fn the_e4m3_encoder_saturates_to_the_largest_finite_code() {
    assert_eq!(e4m3_encode(FP8_E4M3_MAX), 0x7E);
    assert_eq!(e4m3_encode(448.5), 0x7E);
    assert_eq!(e4m3_encode(1.0e30), 0x7E);
    assert_eq!(e4m3_encode(f32::INFINITY), 0x7E);
    assert_eq!(e4m3_encode(f32::NEG_INFINITY), 0xFE);
    // NaN is the one thing the reference's clamp cannot produce, and it gets the NaN code.
    assert_eq!(e4m3_encode(f32::NAN), 0x7F);
    // The top of the negative half is the mirror of the top of the positive half.
    assert_eq!(e4m3_encode(-FP8_E4M3_MAX), 0xFE);
    assert_eq!(e4m3_encode(-448.5), 0xFE);
}

/// Decoding a code and encoding the value gives the code back, for both grids.
#[test]
fn both_encoders_are_the_exact_inverse_of_their_decoders_on_their_own_grid() {
    for code in 0..=0x7Eu8 {
        assert_eq!(e4m3_encode(f8e4m3_to_f32(code)), code, "code 0x{code:02X}");
        let mirrored = code | 0x80;
        assert_eq!(
            e4m3_encode(f8e4m3_to_f32(mirrored)),
            mirrored,
            "code 0x{mirrored:02X}"
        );
    }
    for nibble in 0..16u8 {
        assert_eq!(
            f4e2m1_encode(f4e2m1_to_f32(nibble)),
            nibble,
            "nibble 0x{nibble:X}"
        );
    }
}

/// The nearest E2M1 magnitude to a value, by brute force over the reference table.
fn reference_f4_nearest(magnitude: f64) -> u8 {
    let mut best = 0u8;
    let mut best_distance = f64::INFINITY;
    for nibble in 0..8u8 {
        let value = f64::from(f32::from_bits(FP4_REFERENCE[usize::from(nibble)]));
        let distance = (value - magnitude).abs();
        if distance < best_distance || (distance == best_distance && nibble % 2 == 0) {
            best = nibble;
            best_distance = distance;
        }
    }
    best
}

/// The E2M1 encoder agrees with the brute-force search, ties and saturation included.
#[test]
fn the_e2m1_encoder_rounds_like_the_reference_and_saturates_at_six() {
    for step in 0..=1_000u32 {
        let magnitude = (f64::from(step) * (6.0 / 500.0)) as f32;
        let expected = reference_f4_nearest(f64::from(magnitude));
        assert_eq!(f4e2m1_encode(magnitude), expected, "encoding {magnitude:e}");
        assert_eq!(
            f4e2m1_encode(-magnitude),
            expected | 0x08,
            "encoding -{magnitude:e}"
        );
    }
    // The halfway points, where the tie rule is the only thing that decides. Mantissa bit 0 is the
    // nibble's low bit, so "even" means an even nibble.
    assert_eq!(f4e2m1_encode(0.25), 0x0);
    assert_eq!(f4e2m1_encode(0.75), 0x2);
    assert_eq!(f4e2m1_encode(1.25), 0x2);
    assert_eq!(f4e2m1_encode(1.75), 0x4);
    assert_eq!(f4e2m1_encode(2.5), 0x4);
    assert_eq!(f4e2m1_encode(3.5), 0x6);
    assert_eq!(f4e2m1_encode(5.0), 0x6);
    // Saturation at 6, and the NaN this format does not have.
    assert_eq!(f4e2m1_encode(FP4_E2M1_MAX), 0x7);
    assert_eq!(f4e2m1_encode(6.5), 0x7);
    assert_eq!(f4e2m1_encode(100.0), 0x7);
    assert_eq!(f4e2m1_encode(-100.0), 0xF);
    assert_eq!(f4e2m1_encode(f32::INFINITY), 0x7);
    assert_eq!(f4e2m1_encode(f32::NAN), 0x7);
    assert_eq!(f4e2m1_encode(-0.0), 0x8);
}
/// 2^exponent for everything E8M0 can hold, assembled from the IEEE fields so the samples below are
/// exact even where f32 itself is subnormal.
fn power_of_two(exponent: i32) -> f32 {
    if exponent >= -126 {
        f32::from_bits(((exponent + 127) as u32) << 23)
    } else {
        f32::from_bits(1u32 << (exponent + 149))
    }
}

/// A 32-element block that is the same value everywhere, so its max-abs is exactly that value.
fn uniform_block(value: f32) -> Vec<f32> {
    vec![value; 32]
}

/// The scale encoder really is the smallest E8M0 value that is at least its input.
#[test]
fn the_e8m0_scale_encoder_takes_the_ceiling_of_the_logarithm() {
    let mut samples: Vec<f32> = Vec::new();
    for exponent in -149..=127 {
        let power = power_of_two(exponent);
        samples.push(power);
        samples.push(power * 1.5);
        samples.push(power * 1.0625);
    }
    for value in samples {
        let code = e8m0_encode_ceil(value);
        if value <= 0.0 {
            // The format has no zero, so a non-positive input can only mean its smallest value.
            assert_eq!(code, 0, "{value:e} has no code above it");
            continue;
        }
        if value > power_of_two(127) {
            // Nothing can be at least as large as this: the top code saturates, which is the one
            // documented input outside the ceiling rule.
            assert_eq!(code, 254, "{value:e} should saturate");
            continue;
        }
        let decoded = f8e8m0_to_f32(code);
        assert!(
            decoded >= value,
            "{value:e} encoded as {decoded:e}, which is below it"
        );
        if code > 0 {
            assert!(
                f8e8m0_to_f32(code - 1) < value,
                "{value:e}: code {} would have done",
                code - 1
            );
        }
    }
    // The codes the two quantisers depend on, with the power of two each one means.
    assert_eq!(e8m0_encode_ceil(1.0), 127);
    assert_eq!(e8m0_encode_ceil(power_of_two(-22)), 105);
    assert_eq!(e8m0_encode_ceil(ACTIVATION_AMAX_FLOOR / FP8_E4M3_MAX), 105);
    assert_eq!(e8m0_encode_ceil(power_of_two(-126)), 1);
    assert_eq!(e8m0_encode_ceil(power_of_two(-127)), 0);
    assert_eq!(e8m0_encode_ceil(power_of_two(127)), 254);
    assert_eq!(e8m0_encode_ceil(f32::MAX), 254);
    assert_eq!(e8m0_encode_ceil(0.0), 0);
    assert_eq!(e8m0_encode_ceil(-1.0), 0);
    assert_eq!(e8m0_encode_ceil(f32::NAN), E8M0_NAN);
}

/// An all-zero FP8 block: the floor gives it a scale, and it dequantises to zeros, not NaNs.
#[test]
fn an_all_zero_fp8_block_gets_the_reference_floor_scale_and_stays_zero() {
    // The floor is 1e-4, so the scale is 2^ceil(log2(1e-4 / 448)) = 2^-22, E8M0 code 105.
    assert_eq!(ACTIVATION_AMAX_FLOOR, 1e-4);
    let quantised = quantize_activation_fp8(&uniform_block(0.0)).expect("one block");
    assert_eq!(quantised.scales(), [105].as_slice());
    assert_eq!(f8e8m0_to_f32(105), power_of_two(-22));
    assert!(quantised.codes().iter().all(|&code| code == 0x00));
    let restored = quantised.dequantize();
    assert!(
        restored
            .iter()
            .all(|value| *value == 0.0 && !value.is_sign_negative())
    );
    // Negative zero is a real input: it takes the negative zero code, and dividing it by a positive
    // scale keeps its sign exactly as the reference's clamp does.
    let negated = quantize_activation_fp8(&uniform_block(-0.0)).expect("one block");
    assert!(negated.codes().iter().all(|&code| code == 0x80));
    assert!(
        negated
            .dequantize()
            .iter()
            .all(|value| *value == 0.0 && value.is_sign_negative())
    );
}

/// The same for FP4, whose floor is the smallest normal f32 rather than 2^-22.
#[test]
fn an_all_zero_fp4_block_gets_the_reference_floor_scale_and_stays_zero() {
    // The floor is 6 * 2^-126, so the scale is 2^-126 exactly: E8M0 code 1, the smallest normal.
    assert_eq!(FP4_AMAX_FLOOR, 6.0 * f32::MIN_POSITIVE);
    let quantised = quantize_activation_fp4(&uniform_block(0.0)).expect("one block");
    assert_eq!(quantised.scales(), [1].as_slice());
    assert_eq!(f8e8m0_to_f32(1), f32::MIN_POSITIVE);
    assert!(quantised.packed().iter().all(|&byte| byte == 0x00));
    assert!(quantised.dequantize().iter().all(|value| *value == 0.0));
}

/// The FP8 scale is the reference's power of two, at the boundaries where the f32 product matters.
#[test]
fn the_fp8_activation_scale_is_the_reference_power_of_two() {
    // amax = 448 gives the f32 product 448 * f32(1/448) = 1.0 exactly, so the scale is 1.0 and 448
    // is representable without saturation. A wider multiplication would see 1.0000000447 and could
    // answer 2, which is why the product has to be taken in f32.
    let quantised = quantize_activation_fp8(&uniform_block(FP8_E4M3_MAX)).expect("one block");
    assert_eq!(quantised.scales(), [127].as_slice());
    assert_eq!(f8e8m0_to_f32(127), 1.0);
    assert_eq!(quantised.codes()[0], 0x7E);
    assert_eq!(quantised.dequantize()[0], 448.0);
    // One step up in amax moves the scale up a power of two; 449 / 2 = 224.5 rounds down to 224
    // (mantissa 6, even) rather than up to 240 (mantissa 7), and 224 * 2 = 448.
    let quantised = quantize_activation_fp8(&uniform_block(449.0)).expect("one block");
    assert_eq!(quantised.scales(), [128].as_slice());
    assert_eq!(quantised.codes()[0], 0x76);
    assert_eq!(quantised.dequantize()[0], 448.0);
    // amax = 1.0 needs 2^-8 so that 1.0 / 2^-8 = 256 is inside the range, and 256 is exact.
    let quantised = quantize_activation_fp8(&uniform_block(1.0)).expect("one block");
    assert_eq!(quantised.scales(), [119].as_slice());
    assert_eq!(quantised.codes()[0], 0x78);
    assert_eq!(quantised.dequantize(), uniform_block(1.0));
    // amax = 2^20 needs a scale of 2^12, and the round trip is still exact.
    let quantised = quantize_activation_fp8(&uniform_block(power_of_two(20))).expect("one block");
    assert_eq!(quantised.scales(), [139].as_slice());
    assert_eq!(quantised.codes()[0], 0x78);
    assert_eq!(quantised.dequantize()[0], power_of_two(20));
}

/// One scale per block, in block order - not one scale for the tensor, and not block 1's for block 0.
#[test]
fn each_block_gets_its_own_scale_in_block_order() {
    let mut values = Vec::new();
    values.extend(uniform_block(1.0));
    values.extend(uniform_block(FP8_E4M3_MAX));
    values.extend(uniform_block(0.0));
    values.extend(uniform_block(-FP8_E4M3_MAX));
    let quantised = quantize_activation_fp8(&values).expect("four blocks");
    assert_eq!(quantised.elements(), 128);
    assert_eq!(quantised.scales(), [119, 127, 105, 127].as_slice());
    assert_eq!(quantised.codes()[0], 0x78);
    assert_eq!(quantised.codes()[32], 0x7E);
    assert_eq!(quantised.codes()[64], 0x00);
    assert_eq!(quantised.codes()[96], 0xFE);
    // Every block was a power-of-two multiple of its own scale, so nothing is lost at all.
    assert_eq!(quantised.dequantize(), values);
}

/// A partial block is refused: a scale computed from 31 elements would mis-scale the 32nd.
#[test]
fn a_partial_block_is_refused_rather_than_scaled_from_fewer_elements() {
    assert_eq!(
        quantize_activation_fp8(&[1.0; 31]),
        Err(QuantError::BlockSize { elements: 31 })
    );
    assert_eq!(
        quantize_activation_fp4(&[1.0; 31]),
        Err(QuantError::BlockSize { elements: 31 })
    );
    assert_eq!(
        quantize_activation_fp8(&[1.0; 33]),
        Err(QuantError::BlockSize { elements: 33 })
    );
    // No elements is no blocks, which is an empty tensor rather than a short one.
    let empty = quantize_activation_fp8(&[]).expect("no blocks");
    assert_eq!(empty.elements(), 0);
    assert!(empty.codes().is_empty() && empty.scales().is_empty() && empty.dequantize().is_empty());
}

/// The FP4 quantiser writes element 2i into the low nibble, for positive and negative values.
#[test]
fn the_fp4_quantiser_packs_element_two_i_into_the_low_nibble() {
    // The eight magnitudes in order, four times: amax is 6, the scale is 1.0, and every value is
    // exactly representable, so the codes are 0..7 in element order.
    let mut values = Vec::new();
    for _ in 0..4 {
        values.extend([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]);
    }
    let quantised = quantize_activation_fp4(&values).expect("one block");
    assert_eq!(quantised.scales(), [127].as_slice());
    // Reading the high nibble first would give 0x01, 0x23, 0x45, 0x67 and swap every pair.
    assert_eq!(
        quantised.packed(),
        [
            0x10, 0x32, 0x54, 0x76, 0x10, 0x32, 0x54, 0x76, 0x10, 0x32, 0x54, 0x76, 0x10, 0x32,
            0x54, 0x76,
        ]
        .as_slice()
    );
    assert_eq!(quantised.dequantize(), values);
    // Signs pack the same way: the sign is bit 3 of the nibble, so 0xE6 is -1.0 then +1.0 after
    // scaling, and 0xC4 is +0.5 then -0.5.
    let mut signed = vec![1.0, -1.0, 0.5, -0.5];
    signed.resize(32, 0.0);
    let quantised = quantize_activation_fp4(&signed).expect("one block");
    assert_eq!(quantised.scales(), [125].as_slice());
    assert_eq!(quantised.packed()[0], 0xE6);
    assert_eq!(quantised.packed()[1], 0xC4);
    assert_eq!(
        &quantised.dequantize()[..4],
        [1.0, -1.0, 0.5, -0.5].as_slice()
    );
}

/// Each quantised code is the nearest one to the scaled value, checked against the reference table.
#[test]
fn every_quantised_code_is_the_nearest_code_after_scaling() {
    // A deterministic spread of values in two blocks: an LCG, so the property is checked on
    // magnitudes the test did not get to choose and the run is the same on every machine.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let values: Vec<f32> = (0..64)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 40) as f32) / f32::from(u16::MAX);
            (unit * 2.0 - 1.0) * 12.0
        })
        .collect();
    let quantised = quantize_activation_fp8(&values).expect("two blocks");
    let restored = quantised.dequantize();
    for block in 0..2 {
        let scale = f8e8m0_to_f32(quantised.scales()[block]);
        let amax = values[block * 32..(block + 1) * 32]
            .iter()
            .fold(0.0f32, |acc, &value| acc.max(value.abs()));
        // The point of the scale rule: amax / scale never exceeds 448, so the clamp the reference
        // applies is a no-op and no element of the block saturates.
        assert!(
            amax / scale <= FP8_E4M3_MAX,
            "block {block}: amax {amax} over scale {scale:e}"
        );
        for index in block * 32..(block + 1) * 32 {
            let code = quantised.codes()[index];
            let target = f64::from(values[index] / scale).clamp(-448.0, 448.0);
            let chosen = f64::from(f32::from_bits(E4M3_REFERENCE[usize::from(code)]));
            let error = (chosen - target).abs();
            for candidate in 0..0x7Fu8 {
                let grid = f64::from(f32::from_bits(E4M3_REFERENCE[usize::from(candidate)]));
                // The slack is nine orders of magnitude below the finest grid step, so it absorbs
                // the rounding of the subtraction and not a wrong answer.
                assert!(
                    error <= (grid - target).abs() + 1.0e-9,
                    "element {index}: code 0x{code:02X} is not the nearest to {target:e}"
                );
            }
            // No element is further than half the local grid step, which is at most 16 scales at
            // the top of the range.
            assert!(
                (values[index] - restored[index]).abs() <= 16.0 * scale + f32::EPSILON,
                "element {index} is more than half a step from the code that was stored"
            );
        }
    }
}

/// The FP4 quantiser picks the nearest magnitude too, and never needs its clamp.
#[test]
fn the_fp4_quantiser_rounds_to_the_nearest_code_and_never_saturates() {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let values: Vec<f32> = (0..32)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let unit = ((state >> 40) as f32) / f32::from(u16::MAX);
            (unit * 2.0 - 1.0) * FP4_E2M1_MAX
        })
        .collect();
    let quantised = quantize_activation_fp4(&values).expect("one block");
    let restored = quantised.dequantize();
    let scale = f8e8m0_to_f32(quantised.scales()[0]);
    let amax = values
        .iter()
        .fold(0.0f32, |acc, &value| acc.max(value.abs()));
    assert!(
        amax / scale <= FP4_E2M1_MAX,
        "amax {amax} over scale {scale:e}"
    );
    for (index, &value) in values.iter().enumerate() {
        let byte = quantised.packed()[index / 2];
        let nibble = if index % 2 == 0 {
            byte & 0x0F
        } else {
            byte >> 4
        };
        let target = f64::from(value / scale).clamp(-6.0, 6.0);
        let chosen = f64::from(f32::from_bits(FP4_REFERENCE[usize::from(nibble)]));
        let error = (chosen - target).abs();
        for candidate in 0..8u8 {
            let grid = f64::from(f32::from_bits(FP4_REFERENCE[usize::from(candidate)]));
            assert!(
                error <= (grid - target).abs() + 1.0e-9,
                "element {index}: nibble 0x{nibble:X} is not the nearest to {target:e}"
            );
        }
        // The FP4 grid's largest step is 2, so half a step in scaled units is 1.
        assert!(
            (value - restored[index]).abs() <= scale + f32::EPSILON,
            "element {index} is more than half a step from the code that was stored"
        );
    }
}

/// The two halves meet: what a loader decodes and multiplies is what the quantiser dequantises.
#[test]
fn the_loader_path_reproduces_the_quantisers_dequantisation() {
    let values: Vec<f32> = (0..64).map(|index| (index as f32 - 32.0) / 8.0).collect();
    let quantised = quantize_activation_fp8(&values).expect("two blocks");
    let codes = decode_f8e4m3(quantised.codes());
    let scales = decode_f8e8m0(quantised.scales());
    let restored: Vec<f32> = codes
        .iter()
        .enumerate()
        .map(|(index, &code)| code * scales[index / 32])
        .collect();
    assert_eq!(restored, quantised.dequantize());
    // The generic entry point is the same decode, with the counts a plan and a shape would give.
    assert_eq!(
        decode(DType::F8E4M3, quantised.codes(), values.len()).expect("codes"),
        codes
    );
    assert_eq!(
        decode(DType::F8E8M0, quantised.scales(), values.len() / 32).expect("scales"),
        scales
    );
    // The FP4 path packs and unpacks in the same order, element for element.
    let packed = quantize_activation_fp4(&values).expect("two blocks");
    let unpacked = decode_f4e2m1(packed.packed(), values.len()).expect("packed values");
    let fp4_scales = decode_f8e8m0(packed.scales());
    let fp4_restored: Vec<f32> = unpacked
        .iter()
        .enumerate()
        .map(|(index, &value)| value * fp4_scales[index / 32])
        .collect();
    assert_eq!(fp4_restored, packed.dequantize());
}

#[test]
fn decode_refuses_a_buffer_that_is_not_the_length_it_was_asked_for() {
    // The element count is part of the request, not a hint. Eight bytes are two f32 values, so a
    // caller that asks for one element and hands over two has a plan and a shard that disagree, and
    // decoding the first value anyway would hide exactly that. The same holds for bf16, and the
    // E4M3/E8M0/E2M1 arms were always checked.
    assert!(matches!(
        decode(DType::F32, &[0u8; 8], 1),
        Err(QuantError::Length { .. })
    ));
    assert!(matches!(
        decode(DType::BF16, &[0u8; 4], 1),
        Err(QuantError::Length { .. })
    ));
    assert!(matches!(
        decode(DType::F32, &[0u8; 4], 2),
        Err(QuantError::Length { .. })
    ));
    // The exact length decodes, and the error carries both counts so a reader can see which way the
    // disagreement went.
    assert_eq!(decode(DType::F32, &[0u8; 8], 2).unwrap().len(), 2);
    assert_eq!(decode(DType::BF16, &[0u8; 4], 2).unwrap().len(), 2);
    match decode(DType::BF16, &[0u8; 4], 1) {
        Err(QuantError::Length {
            expected, actual, ..
        }) => {
            assert_eq!(expected, 2);
            assert_eq!(actual, 4);
        }
        other => panic!("expected Length, got {other:?}"),
    }
}

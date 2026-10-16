//! The checkpoint's storage formats decoded to f32, and the activation quantiser that fills them.
//!
//! [crate::dtype] names the formats and sizes them; this module turns their bytes into numbers.
//! Every decode here is exact: each of these formats stores values that f32 represents without
//! rounding, so a decoded weight is the weight rather than a close copy of it. That is why the tests
//! pin the formats code by code instead of sampling them - a decoder that is wrong in one exponent
//! band, or one that reads an FP4 byte's nibbles in the wrong order, corrupts a real weight silently
//! and the forward pass still produces plausible numbers.
//!
//! The formats, as the released checkpoint stores them (docs/architecture/03-moe.md):
//!
//! - E4M3FN: 1 sign, 4 exponent, 3 mantissa, bias 7. 0x7F and 0xFF are NaN, 0x80 is negative zero,
//!   there are NO infinities, 448 is the largest finite value (kernel.py:44-46) and 2^-9 the
//!   smallest positive one.
//! - E8M0: an unsigned 8-bit exponent with bias 127, 255 the only NaN and no zero at all, so code 0
//!   is 2^-127 - a value f32 holds as a subnormal. This is the scale format, which is why decoding
//!   it must be exact: one value multiplies a whole block of weights.
//! - E2M1: 1 sign, 2 exponent, 1 mantissa, bias 1. The eight magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4
//!   and 6, nibble 0x8 is negative zero, and two values share a byte with the LOW nibble first.
//! - bf16: the top 16 bits of an f32 - the same 1/8/7 fields at the same offsets, not a rounding of
//!   the f32 value.
//! - f32: little-endian IEEE-754 binary32, the identity case.
//!
//! The quantisers at the end are the reference's: kernel.py:41-95 quantises an FP8 activation block
//! and kernel.py:128-181 an FP4 one. Both use the same shape of rule - per-32-element block
//! max-abs, a floor that keeps an all-zero block usable, and an E8M0 scale that is the smallest
//! power of two at least as large as amax divided by the format's maximum, so that a scaled block
//! never needs the saturation the reference applies anyway.
//!
//! Nothing here reads a checkpoint or touches a device: bytes in, floats out.

use crate::dtype::{DType, SCALE_BLOCK};

/// The largest finite E4M3 value (kernel.py:45).
pub const FP8_E4M3_MAX: f32 = 448.0;

/// The largest finite E2M1 magnitude (kernel.py:163).
pub const FP4_E2M1_MAX: f32 = 6.0;

/// The code an E8M0 byte has when it means NaN: the format has exactly one, 255.
pub const E8M0_NAN: u8 = 0xFF;

/// The largest E4M3 code that is a number; 0x7F is the format's positive NaN.
const E4M3_MAX_CODE: u8 = 0x7E;

/// The E4M3 code that means NaN.
const E4M3_NAN: u8 = 0x7F;

/// The largest E2M1 nibble: all sixteen are numbers, so 0x07 is the top of the magnitude grid.
const E2M1_MAX_CODE: u8 = 0x07;

/// The quiet NaN a NaN code decodes to, with the code's sign bit or-ed in.
///
/// The payload is canonical rather than derived from the code because no reference conversion
/// preserves an E4M3 or E8M0 NaN payload; the sign is kept because it costs nothing and IEEE
/// conversion keeps it too, so a negative NaN code stays negative.
const QUIET_NAN: u32 = 0x7FC0_0000;

/// The reciprocal of [FP8_E4M3_MAX], kept as an f32.
///
/// The reference multiplies by this instead of dividing (kernel.py:46, 78), and the product is what
/// picks the scale, so the product is taken in f32 too: at amax = 448 the f32 product rounds to
/// exactly 1.0 and the scale is 1.0, while the exact product is 1.0000000447 and a wider
/// multiplication would be free to answer 2^1 instead.
const FP8_MAX_INV: f32 = 1.0 / FP8_E4M3_MAX;

/// The reciprocal of [FP4_E2M1_MAX], kept as an f32 for the same reason.
const FP4_MAX_INV: f32 = 1.0 / FP4_E2M1_MAX;

/// The floor the reference puts under an FP8 block's max-abs before choosing a scale
/// (kernel.py:76).
///
/// It exists so that an all-zero block still has a usable scale. Without it the scale would be
/// zero, every element of the block would quantise to NaN, and the block would dequantise to NaNs
/// rather than to the zeros it started as. With it, an all-zero block stores 32 zero codes and the
/// scale 2^-22, which dequantises to exactly zero.
pub const ACTIVATION_AMAX_FLOOR: f32 = 1e-4;

/// The floor under an FP4 block's max-abs before choosing a scale (kernel.py:165): six times
/// 2^-126, the point where the scale amax / 6 reaches the smallest normal f32.
pub const FP4_AMAX_FLOOR: f32 = 6.0 * f32::MIN_POSITIVE;

/// The exact power of two 2^exponent, built from the IEEE fields rather than from a library call.
///
/// The domain is the union of what the two formats need: -127 (the E8M0 minimum, which f32 holds as
/// a subnormal and is the reason this is not a shift of the exponent field alone) up to +127 (the
/// E8M0 maximum). Every call site stays inside it by construction.
const fn pow2(exponent: i32) -> f32 {
    debug_assert!(
        exponent >= -127 && exponent <= 127,
        "pow2 is only defined where the formats need it"
    );
    if exponent >= -126 {
        f32::from_bits(((exponent + 127) as u32) << 23)
    } else {
        // Subnormal: 2^-127 is 2^-149 times 2^22, a single mantissa bit set.
        f32::from_bits(1u32 << (exponent + 149))
    }
}

/// Decodes one E4M3 code to the value it stands for.
///
/// Exact by construction: an E4M3 value is a dyadic rational with at most four significant bits
/// (the implicit one plus three), so both branches below are a small integer times a power of two
/// and cannot round. Nothing decodes to an infinity, because the format has none, and the two NaN
/// codes decode to a quiet NaN carrying their sign. The sign is applied after the magnitude rather
/// than written into a sign field so that 0x80 decodes to negative zero.
#[must_use]
pub fn f8e4m3_to_f32(code: u8) -> f32 {
    let negative = code & 0x80 != 0;
    let exponent = (code >> 3) & 0x0F;
    let mantissa = code & 0x07;
    if exponent == 0x0F && mantissa == 0x07 {
        return f32::from_bits(if negative {
            QUIET_NAN | 0x8000_0000
        } else {
            QUIET_NAN
        });
    }
    let magnitude = if exponent == 0 {
        // Subnormal: (mantissa / 8) * 2^-6 = mantissa * 2^-9, exact for a three-bit mantissa.
        f32::from(mantissa) * pow2(-9)
    } else {
        // Normal: (1 + mantissa / 8) * 2^(exponent - 7), exact for a four-bit significand.
        (1.0 + f32::from(mantissa) / 8.0) * pow2(i32::from(exponent) - 7)
    };
    if negative { -magnitude } else { magnitude }
}

/// Decodes one E8M0 code to the value it stands for: 2^(code - 127), or NaN for 255.
///
/// Exact for all 255 finite codes, including both ends: code 0 is 2^-127, which f32 holds as a
/// subnormal, and code 254 is 2^127. The format has no zero and no infinities.
#[must_use]
pub fn f8e8m0_to_f32(code: u8) -> f32 {
    if code == E8M0_NAN {
        return f32::from_bits(QUIET_NAN);
    }
    // The unsigned code is its own exponent: 0 means 2^-127, 254 means 2^127, and both are inside
    // f32's range.
    pow2(i32::from(code) - 127)
}

/// Decodes a bf16 bit pattern to the f32 it is the top half of.
///
/// Nothing is rounded and nothing is lost: bf16 is not merely similar to the high half of an f32,
/// it is those bits, with the same sign, exponent and mantissa fields at the same offsets. Every
/// bf16 value is inside f32's range with eight bits of precision to spare, subnormals included -
/// the smallest bf16 subnormal is 2^-133, well above f32's 2^-149.
#[must_use]
pub const fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Decodes four little-endian bytes as an f32.
///
/// The checkpoint's F32 tensors are raw binary32 in little-endian order, the layout safetensors
/// uses for every dtype, so this is byte assembly rather than conversion. Every bit pattern is a
/// value, NaNs and infinities included.
#[must_use]
pub const fn f32_from_le_bytes(bytes: [u8; 4]) -> f32 {
    f32::from_bits(u32::from_le_bytes(bytes))
}

/// Decodes one E2M1 nibble to the value it stands for.
///
/// Only the low four bits are read, which is what lets a packed byte's low half be passed straight
/// in; [f4e2m1_pair_to_f32] is the entry point for a whole byte. The eight magnitudes are 0, 0.5,
/// 1, 1.5, 2, 3, 4 and 6, every one of them exact in f32, and nibble 0x8 is negative zero just as
/// code 0x80 is for E4M3. The format has no NaN and no infinity.
#[must_use]
pub fn f4e2m1_to_f32(nibble: u8) -> f32 {
    let nibble = nibble & 0x0F;
    let negative = nibble & 0x08 != 0;
    let exponent = (nibble >> 1) & 0x03;
    let mantissa = nibble & 0x01;
    let magnitude = if exponent == 0 {
        // Subnormal: mantissa * 0.5, so 0.0 or 0.5.
        f32::from(mantissa) * 0.5
    } else {
        // Normal: (1 + mantissa / 2) * 2^(exponent - 1), the eight magnitudes above.
        (1.0 + f32::from(mantissa) * 0.5) * pow2(i32::from(exponent) - 1)
    };
    if negative { -magnitude } else { magnitude }
}

/// Decodes both values of a packed E2M1 byte, LOW nibble first.
///
/// That order is the storage order of torch.float4_e2m1fn_x2, which is what convert.py:181 views
/// the quantised tensor as, and what kernel.py:572 documents as two FP4 values per byte packed
/// along K. Reading the high nibble first would swap every adjacent pair of a weight row and no
/// later stage could detect it, so tests/quant.rs pins all 256 byte values against the nibble
/// table.
#[must_use]
pub fn f4e2m1_pair_to_f32(byte: u8) -> [f32; 2] {
    [f4e2m1_to_f32(byte & 0x0F), f4e2m1_to_f32(byte >> 4)]
}

/// What can go wrong while decoding or quantising.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum QuantError {
    /// The bytes on hand do not match the element count.
    #[error(
        "{} needs {expected} bytes for {elements} elements, but {actual} were given",
        dtype.name()
    )]
    Length {
        /// The format that was asked for.
        dtype: DType,
        /// How many elements were asked for.
        elements: usize,
        /// How many bytes those elements occupy, per [DType::storage_bytes].
        expected: usize,
        /// How many bytes were handed over.
        actual: usize,
    },
    /// The format is named by [DType] but this module does not decode it.
    #[error("{} is not a checkpoint value format, so there is no decoder here", dtype.name())]
    Unsupported {
        /// The format that was asked for.
        dtype: DType,
    },
    /// The element count is not a whole number of scale blocks.
    #[error(
        "{elements} elements is not a whole number of {block}-element blocks",
        block = SCALE_BLOCK
    )]
    BlockSize {
        /// How many elements were handed over.
        elements: usize,
    },
}

/// Checks an element count against a byte count using [DType::storage_bytes].
fn check_len(dtype: DType, elements: usize, actual: usize) -> Result<(), QuantError> {
    let expected = dtype.storage_bytes(elements);
    if actual == expected {
        Ok(())
    } else {
        Err(QuantError::Length {
            dtype,
            elements,
            expected,
            actual,
        })
    }
}

/// Checks that a buffer holds whole units of a multi-byte format.
///
/// The element count in the error is rounded up so that the message reads as the tensor the bytes
/// would have to belong to: five bytes of bf16 is three values with one byte missing, not two.
fn check_whole_units(dtype: DType, actual: usize, unit: usize) -> Result<(), QuantError> {
    if actual.is_multiple_of(unit) {
        return Ok(());
    }
    let elements = actual.div_ceil(unit);
    Err(QuantError::Length {
        dtype,
        elements,
        expected: elements * unit,
        actual,
    })
}

/// Decodes a whole E4M3 buffer: one code per byte, and every byte is a valid code.
#[must_use]
pub fn decode_f8e4m3(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|&code| f8e4m3_to_f32(code)).collect()
}

/// Decodes a whole E8M0 scale buffer.
///
/// The result is the scales, not the scaled weights: the caller multiplies, in f32, because that is
/// where the reference does it (convert.py:171, kernel.py:85).
#[must_use]
pub fn decode_f8e8m0(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|&code| f8e8m0_to_f32(code)).collect()
}

/// Decodes a little-endian bf16 buffer.
///
/// # Errors
///
/// [QuantError::Length] when the buffer is not a whole number of two-byte values: half a bf16 is
/// not a value, and padding it with a zero byte would invent one.
pub fn decode_bf16(bytes: &[u8]) -> Result<Vec<f32>, QuantError> {
    check_whole_units(DType::BF16, bytes.len(), 2)?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| bf16_to_f32(u16::from_le_bytes(*pair)))
        .collect())
}

/// Decodes a little-endian f32 buffer.
///
/// # Errors
///
/// [QuantError::Length] when the buffer is not a whole number of four-byte values.
pub fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, QuantError> {
    check_whole_units(DType::F32, bytes.len(), 4)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32_from_le_bytes(*word))
        .collect())
}

/// Decodes a packed E2M1 buffer in element order: a byte's low nibble before its high one.
///
/// A trailing nibble is padding, exactly as [DType::storage_bytes] rounds an odd element count up
/// to a whole byte, so the element count is what says how much of the last byte is real.
///
/// # Errors
///
/// [QuantError::Length] when the buffer is not the size [DType::storage_bytes] computes for
/// elements. Accepting more bytes than that would silently drop values off the end of a tensor.
pub fn decode_f4e2m1(bytes: &[u8], elements: usize) -> Result<Vec<f32>, QuantError> {
    check_len(DType::F4E2M1, elements, bytes.len())?;
    Ok((0..elements)
        .map(|index| {
            let byte = bytes[index / 2];
            if index % 2 == 0 {
                f4e2m1_to_f32(byte)
            } else {
                f4e2m1_to_f32(byte >> 4)
            }
        })
        .collect())
}

/// Decodes elements values of any checkpoint format out of bytes.
///
/// This is the entry point a loader wants: the dtype comes from the tensor plan, the element count
/// from the logical shape, and the one thing that can be wrong is the byte count - which is the
/// check that catches a plan and a shard disagreeing before an uninitialised weight reaches the
/// arithmetic.
///
/// # Errors
///
/// [QuantError::Length] when the buffer is not [DType::storage_bytes] long for the element count;
/// [QuantError::Unsupported] for [DType::F16], which dtype.rs names for completeness but the
/// released checkpoint never uses, so no decoder is asserted for it here rather than an untested
/// one being shipped.
pub fn decode(dtype: DType, bytes: &[u8], elements: usize) -> Result<Vec<f32>, QuantError> {
    match dtype {
        // Every fixed-width arm checks the length against the element count, not merely against a
        // whole number of units: `decode_f32` and `decode_bf16` will happily decode eight bytes as
        // two f32s, which is a correct decoding of the wrong tensor. A caller that asks for one
        // element and hands over eight bytes has a plan and a shard that disagree, and that is the
        // one failure this function exists to catch.
        DType::F32 => {
            check_len(dtype, elements, bytes.len())?;
            decode_f32(bytes)
        }
        DType::BF16 => {
            check_len(dtype, elements, bytes.len())?;
            decode_bf16(bytes)
        }
        DType::F8E4M3 => {
            check_len(dtype, elements, bytes.len())?;
            Ok(decode_f8e4m3(bytes))
        }
        DType::F8E8M0 => {
            check_len(dtype, elements, bytes.len())?;
            Ok(decode_f8e8m0(bytes))
        }
        DType::F4E2M1 => decode_f4e2m1(bytes, elements),
        DType::F16 => Err(QuantError::Unsupported { dtype }),
    }
}

/// The E8M0 code of the smallest E8M0 value at least as large as value.
///
/// This is the reference's scale rule in one function. Its fast_round_scale (kernel.py:36-37) takes
/// the ceiling of the base-2 logarithm by reading the exponent of the f32 product and adding one
/// when the mantissa is nonzero, and its fast_pow2 then builds that power of two; the cast to E8M0
/// at kernel.py:91 is exact because what is being cast is already a power of two. So the code
/// returned here is the byte the reference stores for a value it computed the same way.
///
/// Two inputs cannot come out of that rule and are still defined, because this is public: a
/// non-positive value returns code 0, the smallest E8M0 value (the format has no zero), and a value
/// of 2^127 or more returns 254, which is 2^127 - the largest code the format has. Everything above
/// 2^127 saturates there, 1.5 * 2^127 and 2^128 alike, because no larger power of two is
/// expressible. NaN returns
/// [E8M0_NAN]. The reference's fast_log2_ceil is wrong for subnormal products - it reports 2^-126
/// for any of them (kernel.py:22-27) - but neither quantiser below can produce one: their floors
/// keep every scale at or above 2^-126. This function reads the band correctly instead of
/// reproducing that, and the divergence is unreachable from the checkpoint.
#[must_use]
pub fn e8m0_encode_ceil(value: f32) -> u8 {
    if value.is_nan() {
        return E8M0_NAN;
    }
    if value <= 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let field = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;
    let exponent = if field == 0 {
        // Subnormal: the value is mantissa * 2^-149, so the ceiling of its logarithm is the index of
        // its highest set bit, one higher again when a lower bit is also set.
        let highest = 31 - mantissa.leading_zeros() as i32;
        -149 + highest + i32::from(mantissa.count_ones() > 1)
    } else {
        field - 127 + i32::from(mantissa != 0)
    };
    (exponent + 127).clamp(0, 254) as u8
}

/// Encodes an f32 as the nearest E4M3 code, ties going to the even mantissa.
///
/// Saturation matches the reference's clamp to +/-448 before the cast (kernel.py:89): a value at or
/// beyond the top of the range becomes 0x7E, the largest finite code, and never the NaN at 0x7F. A
/// NaN input is the one thing the reference's clamp cannot produce and is encoded as 0x7F.
///
/// The search is a bisection because the finite codes 0x00..=0x7E are strictly increasing in
/// magnitude, so the nearest code is one of the two around the insertion point; the tie between two
/// equally distant codes is broken on the low mantissa bit, which is what round-to-nearest-even
/// means. Distances are compared in f64: an E4M3 value has four significant bits and the input
/// twenty-four, so the difference of two neighbours - the only difference that can decide the
/// answer - is exact.
#[must_use]
pub fn e4m3_encode(value: f32) -> u8 {
    if value.is_nan() {
        return E4M3_NAN;
    }
    let magnitude = f64::from(value.abs());
    // Bisect for the smallest code whose value is at least the magnitude. When the magnitude is
    // above 448 no code qualifies and the search settles on 0x7E on its own: saturation falls out
    // of the same loop.
    let mut low = 0u8;
    let mut high = E4M3_MAX_CODE;
    while low < high {
        let middle = low.midpoint(high);
        if f64::from(f8e4m3_to_f32(middle)) < magnitude {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    let mut best = low;
    if low > 0 {
        let above = f64::from(f8e4m3_to_f32(low));
        let below = f64::from(f8e4m3_to_f32(low - 1));
        let (distance_below, distance_above) = (magnitude - below, above - magnitude);
        if distance_below < distance_above
            || (distance_below == distance_above && (low - 1).is_multiple_of(2))
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

/// Encodes an f32 as the nearest E2M1 nibble, ties going to the even mantissa.
///
/// The same shape as [e4m3_encode] over a grid of eight magnitudes, saturated at 6. The format has
/// no NaN, so a NaN input saturates to 0x7 the way an infinity does; the reference clamps to +/-6
/// before its cast (kernel.py:163-166), which gives the same answer for every value that can
/// actually reach the cast.
#[must_use]
pub fn f4e2m1_encode(value: f32) -> u8 {
    if value.is_nan() {
        // This format has no NaN code, so a NaN saturates to the largest magnitude instead of
        // propagating: the reference clamps to +/-6 before its cast, and this is that clamp's
        // answer for the one input it cannot produce.
        return E2M1_MAX_CODE;
    }
    let magnitude = f64::from(value.abs());
    let mut low = 0u8;
    let mut high = E2M1_MAX_CODE;
    while low < high {
        let middle = low.midpoint(high);
        if f64::from(f4e2m1_to_f32(middle)) < magnitude {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    let mut best = low;
    if low > 0 {
        let above = f64::from(f4e2m1_to_f32(low));
        let below = f64::from(f4e2m1_to_f32(low - 1));
        let (distance_below, distance_above) = (magnitude - below, above - magnitude);
        if distance_below < distance_above
            || (distance_below == distance_above && (low - 1).is_multiple_of(2))
        {
            best = low - 1;
        }
    }
    if value.is_sign_negative() {
        best | 0x08
    } else {
        best
    }
}

/// One FP8 E4M3 tensor with the E8M0 scale of each 32-element block.
///
/// The fields are private because the two lengths have to agree: the codes are a whole number of
/// blocks long and there is exactly one scale per block, which is what lets [Self::dequantize]
/// index without a fallible case. The only way to build one is [quantize_activation_fp8].
#[derive(Debug, Clone, PartialEq)]
pub struct QuantisedFp8 {
    /// One E4M3 code per element, in element order.
    codes: Vec<u8>,
    /// One E8M0 code per [SCALE_BLOCK] elements, in block order.
    scales: Vec<u8>,
}

impl QuantisedFp8 {
    /// The E4M3 codes, one per element.
    #[must_use]
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    /// The E8M0 scale codes, one per 32-element block.
    #[must_use]
    pub fn scales(&self) -> &[u8] {
        &self.scales
    }

    /// How many elements this tensor holds.
    #[must_use]
    pub fn elements(&self) -> usize {
        self.codes.len()
    }

    /// Dequantises exactly as the reference does: every code times its block's scale, in f32
    /// (convert.py:171, kernel.py:85).
    ///
    /// The multiplication is by a power of two, so it is exact for every code in range; it rounds
    /// only where the reference also rounds, at the edges of f32's range.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        self.codes
            .iter()
            .enumerate()
            .map(|(index, &code)| {
                f8e4m3_to_f32(code) * f8e8m0_to_f32(self.scales[index / SCALE_BLOCK])
            })
            .collect()
    }
}

/// One FP4 E2M1 tensor, packed two values per byte with the low nibble first, and the E8M0 scale of
/// each 32-element block.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantisedFp4 {
    /// The packed nibbles: element 2i in the low half of byte i, element 2i+1 in the high half.
    packed: Vec<u8>,
    /// One E8M0 code per [SCALE_BLOCK] elements, in block order.
    scales: Vec<u8>,
    /// How many elements the packed bytes hold; the last byte's high nibble is padding when this is
    /// odd.
    elements: usize,
}

impl QuantisedFp4 {
    /// The packed bytes, two elements each except possibly the last.
    #[must_use]
    pub fn packed(&self) -> &[u8] {
        &self.packed
    }

    /// The E8M0 scale codes, one per 32-element block.
    #[must_use]
    pub fn scales(&self) -> &[u8] {
        &self.scales
    }

    /// How many elements this tensor holds.
    #[must_use]
    pub fn elements(&self) -> usize {
        self.elements
    }

    /// Dequantises in element order, reading each byte's low nibble first, and multiplies by the
    /// block's scale in f32.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        (0..self.elements)
            .map(|index| {
                let byte = self.packed[index / 2];
                let nibble = if index % 2 == 0 { byte } else { byte >> 4 };
                f4e2m1_to_f32(nibble) * f8e8m0_to_f32(self.scales[index / SCALE_BLOCK])
            })
            .collect()
    }
}

/// The E8M0 scale code the reference picks for one block: 2^ceil(log2(amax * max_inv)).
///
/// The max-abs is raised to the floor first (kernel.py:76, 165) and the product is taken in f32
/// because that is the value whose exponent and mantissa the reference inspects.
fn block_scale_code(block: &[f32], max_inv: f32, floor: f32) -> u8 {
    let amax = block
        .iter()
        .fold(0.0f32, |acc, &value| acc.max(value.abs()))
        .max(floor);
    e8m0_encode_ceil(amax * max_inv)
}

/// Quantises an f32 slice as FP8 E4M3 with one E8M0 scale per 32 elements.
///
/// This is the reference's activation quantiser: kernel.py:41-95 with block_size 32,
/// round_scale True and scale_dtype torch.float8_e8m0fnu, which is the combination the FP8 GEMMs
/// use for their A operand. The rule, in the reference's order:
///
/// 1. amax is the largest absolute value in the block, raised to [ACTIVATION_AMAX_FLOOR] (1e-4);
/// 2. the scale is 2^ceil(log2(amax / 448)) - the smallest power of two that keeps the scaled block
///    inside +/-448 - stored as the E8M0 byte that is exactly that power of two;
/// 3. each element is clamped to +/-448 after division and rounded to the nearest E4M3 code, ties
///    to the even mantissa.
///
/// The all-zero block case is step 1's floor: amax is 1e-4 rather than 0, the scale is 2^-22 rather
/// than zero, and the block quantises to 32 zero codes that dequantise back to exactly zero. The
/// floor is not a hack to avoid a division by zero - it is what the reference does, and it is why a
/// zero block survives a round trip instead of turning into NaN.
///
/// # Errors
///
/// [QuantError::BlockSize] when the element count is not a whole number of blocks. The reference
/// asserts N % block_size == 0 (kernel.py:108) rather than quantising a short block, because a
/// short block would get a scale computed from fewer elements than the block it claims to scale.
pub fn quantize_activation_fp8(values: &[f32]) -> Result<QuantisedFp8, QuantError> {
    if !values.len().is_multiple_of(SCALE_BLOCK) {
        return Err(QuantError::BlockSize {
            elements: values.len(),
        });
    }
    let mut codes = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(values.len() / SCALE_BLOCK);
    for block in values.as_chunks::<SCALE_BLOCK>().0 {
        let code = block_scale_code(block, FP8_MAX_INV, ACTIVATION_AMAX_FLOOR);
        scales.push(code);
        // Dividing by the decoded scale rather than by the power of two the code came from is the
        // same number: an E8M0 code is exact, and the reference's own cast of that power of two to
        // E8M0 is exact as well.
        let scale = f8e8m0_to_f32(code);
        for &value in block {
            codes.push(e4m3_encode(
                (value / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX),
            ));
        }
    }
    Ok(QuantisedFp8 { codes, scales })
}

/// Quantises an f32 slice as FP4 E2M1, packed two per byte, with one E8M0 scale per 32 elements.
///
/// This is the reference's FP4 activation quantiser (kernel.py:128-181), the one the routed-expert
/// GEMMs use for their A operand. It is [quantize_activation_fp8] with two differences, both the
/// reference's: the floor is [FP4_AMAX_FLOOR] (6 * 2^-126, so an all-zero block gets the scale
/// 2^-126, the smallest normal) and the clamp is +/-6, the largest E2M1 magnitude. Element 2i is
/// written into the low nibble of byte i and element 2i+1 into the high nibble, which is the
/// packing [f4e2m1_pair_to_f32] and [decode_f4e2m1] undo.
///
/// # Errors
///
/// [QuantError::BlockSize] when the element count is not a whole number of blocks.
pub fn quantize_activation_fp4(values: &[f32]) -> Result<QuantisedFp4, QuantError> {
    if !values.len().is_multiple_of(SCALE_BLOCK) {
        return Err(QuantError::BlockSize {
            elements: values.len(),
        });
    }
    let mut packed = vec![0u8; values.len().div_ceil(2)];
    let mut scales = Vec::with_capacity(values.len() / SCALE_BLOCK);
    for (block_index, block) in values.as_chunks::<SCALE_BLOCK>().0.iter().enumerate() {
        let code = block_scale_code(block, FP4_MAX_INV, FP4_AMAX_FLOOR);
        scales.push(code);
        let scale = f8e8m0_to_f32(code);
        for (offset, &value) in block.iter().enumerate() {
            let index = block_index * SCALE_BLOCK + offset;
            let nibble = f4e2m1_encode((value / scale).clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX));
            packed[index / 2] |= if index.is_multiple_of(2) {
                nibble
            } else {
                nibble << 4
            };
        }
    }
    Ok(QuantisedFp4 {
        packed,
        scales,
        elements: values.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_power_of_two_helper_covers_its_whole_domain() {
        // The three ends: the E8M0 minimum (subnormal in f32, and the reason this is not a shift of
        // the exponent field alone), the smallest normal, and the E8M0 maximum.
        assert_eq!(pow2(-127).to_bits(), 0x0040_0000);
        assert_eq!(pow2(-126), f32::MIN_POSITIVE);
        assert_eq!(pow2(-1), 0.5);
        assert_eq!(pow2(0), 1.0);
        assert_eq!(pow2(127).to_bits(), 0x7F00_0000);
        for exponent in -127..=127 {
            let value = pow2(exponent);
            assert!(
                value.is_finite() && value > 0.0,
                "2^{exponent} came out as {value}"
            );
        }
    }
}

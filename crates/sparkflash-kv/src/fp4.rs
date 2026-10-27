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

//! Reference FP4 E2M1 / FP8 E4M3 quantisation with E4M3 and E8M0 block scales.
//!
//! This is the definition of the 356-byte global KV entry: a 512-wide MLA latent packed as
//! FP4 E2M1 with one E4M3 scale per 16 channels, followed by a 128-wide indexer key packed as
//! FP4 E2M1 with one E8M0 scale per 32 channels. It mirrors the reference quantisers in
//! /home/user/dsv41-research/inference/kernel.py:
//!
//! * main KV uses fp4_quant_kernel with scale_dtype float8_e4m3fn and block 16; the scale is
//!   max(amax, 6 * 2^-9) / 6 cast to E4M3 and then used in its dequantised form, so an all-zero
//!   group still gets a non-zero scale;
//! * indexer K uses fp4_quant_kernel with the default E8M0 scale and block 32, where the scale
//!   is 2^ceil(log2(max(amax, 6 * 2^-126) / 6));
//! * SWA KV uses act_quant_kernel with FP8 E4M3 data, scale_fmt = "ue8m0" and block 32.
//!
//! **NaN and infinity policy.** FP4 E2M1 has no NaN or infinity encoding. This codec maps
//! NaN to +0.0 (sign dropped, payload zero) and clamps both infinities to +/-6, the largest
//! finite magnitude. That is deterministic, testable, and matches what a saturating cast on
//! the device does; the reference kernel's behaviour for those inputs is undefined, so the
//! policy is ours to pin.

use crate::error::KvError;
use crate::layout::{
    INDEXER_K_DATA_BYTES, INDEXER_K_DIM, INDEXER_K_SCALE_BYTES, INDEXER_K_SCALE_GROUP,
    MAIN_KV_DATA_BYTES, MAIN_KV_DIM, MAIN_KV_SCALE_BYTES, MAIN_KV_SCALE_GROUP, SWA_FP8_SCALE_GROUP,
    SWA_SLOT_DATA_BYTES, SWA_SLOT_DIM, SWA_SLOT_SCALE_BYTES,
};

/// Largest finite magnitude representable in FP4 E2M1.
pub const E2M1_MAX: f32 = 6.0;

/// Smallest positive E2M1 magnitude; E2M1 has no subnormals below this.
pub const E2M1_MIN_POSITIVE: f32 = 0.5;

/// Largest finite magnitude representable in FP8 E4M3 (OCP e4m3fn).
pub const E4M3_MAX: f32 = 448.0;

/// Smallest positive E4M3 subnormal, 2^-9.
pub const E4M3_MIN_SUBNORMAL: f32 = 1.0 / 512.0;

/// Largest finite value representable by an E8M0 scale, 2^127.
pub const E8M0_MAX: f32 = f32::from_bits(254 << 23);

/// Smallest positive E8M0 value, 2^-127.
pub const E8M0_MIN_POSITIVE: f32 = f32::from_bits(1 << 22);

/// The eight non-negative E2M1 magnitudes, indexed by the low three code bits.
const E2M1_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Number of E2M1 codes.
pub const E2M1_CODE_COUNT: usize = 16;

/// Decodes one FP4 E2M1 code.
#[must_use]
pub fn e2m1_decode(code: u8) -> f32 {
    let magnitude = E2M1_MAGNITUDES[(code & 0x7) as usize];
    if code & 0x8 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Encodes one f32 to the nearest FP4 E2M1 code, ties to even mantissa.
///
/// Ties are broken towards the candidate whose code is even, which is exactly the E2M1
/// mantissa-least-significant-bit-even rule. NaN becomes 0 and infinities saturate.
#[must_use]
pub fn e2m1_encode(value: f32) -> u8 {
    if value.is_nan() {
        return 0;
    }
    let sign = if value.is_sign_negative() { 0x8u8 } else { 0 };
    let x = value.abs();
    if x >= E2M1_MAX {
        return sign | 0x7;
    }
    let mut best = 0usize;
    let mut best_error = f32::INFINITY;
    for (index, magnitude) in E2M1_MAGNITUDES.iter().enumerate() {
        let error = (x - magnitude).abs();
        if error < best_error || (error == best_error && index % 2 == 0) {
            best = index;
            best_error = error;
        }
    }
    sign | best as u8
}

/// Decodes one FP8 E4M3 code to its exact value.
///
/// Codes 0x7F and 0xFF are the two NaN encodings of OCP e4m3fn.
#[must_use]
pub fn e4m3_decode(code: u8) -> f32 {
    let exponent = (code >> 3) & 0x0F;
    let mantissa = (code & 0x07) as f32;
    if exponent == 0x0F && code & 0x07 == 0x07 {
        return f32::NAN;
    }
    let magnitude = if exponent == 0 {
        mantissa * (1.0 / 8.0) * exp2(-6)
    } else {
        (1.0 + mantissa / 8.0) * exp2(i32::from(exponent) - 7)
    };
    if code & 0x80 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Encodes one f32 to the nearest FP8 E4M3 code, ties to even mantissa.
///
/// NaN becomes the canonical 0x7F and infinities saturate to +/-448.
#[must_use]
pub fn e4m3_encode(value: f32) -> u8 {
    const MAX_CODE: u8 = 0x7E;
    if value.is_nan() {
        return 0x7F;
    }
    let sign = if value.is_sign_negative() { 0x80u8 } else { 0 };
    let x = value.abs();
    if x.is_infinite() || x >= E4M3_MAX {
        return sign | MAX_CODE;
    }
    let mut low = 0u8;
    let mut high = MAX_CODE;
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if e4m3_decode(mid).abs() <= x {
            low = mid
        } else {
            high = mid - 1
        }
    }
    if low < MAX_CODE {
        let up = low + 1;
        let down_error = x - e4m3_decode(low).abs();
        let up_error = e4m3_decode(up).abs() - x;
        if up_error < down_error || (up_error == down_error && up.is_multiple_of(2)) {
            return sign | up;
        }
    }
    sign | low
}

/// Decodes one E8M0 (ue8m0) scale byte: 2^(code - 127), with 0xFF reserved as NaN.
#[must_use]
pub fn e8m0_decode(code: u8) -> f32 {
    match code {
        0x00 => E8M0_MIN_POSITIVE,
        0xFF => f32::NAN,
        _ => f32::from_bits(u32::from(code) << 23),
    }
}

/// Encodes the smallest power of two greater than or equal to the magnitude, as E8M0.
///
/// This is the reference fast_round_scale / fast_log2_ceil pair: the exponent is taken from
/// the IEEE-754 bit pattern and bumped by one whenever the mantissa is non-zero. Non-positive
/// and NaN inputs return 0, and positive infinity saturates to 2^127.
///
/// Subnormal inputs are handled explicitly rather than through the exponent field, which is
/// zero for all of them. The mathematical ceiling of a value at or below 2^-127 is below the
/// smallest code the format can express, so those clamp to code 0 (= 2^-127); a subnormal in
/// (2^-127, 2^-126) still rounds up to 2^-126, which is the correct answer for it. Reporting
/// 2^-126 for *every* positive subnormal would break the documented contract.
///
/// The high end is the mirror image and is a documented saturation rather than a ceiling:
/// above 2^127 no code can reach the input, because 0xFF is reserved as NaN, so every finite
/// input from 2^127 upwards (and positive infinity) returns 0xFE = 2^127, which is below the
/// input. Every finite input up to and including 2^127 still gets its exact ceiling. No
/// quantiser in this crate can reach either end - the indexer scale tops out near 5.7e37 and
/// the SWA scale near 7.6e35, both below 2^127, and both floors keep the low end normal - but
/// the public encoder states what it does outside that range anyway. The project reference
/// computes the same code here (kernel.py:162-166); its separate fast_log2_ceil would not.
#[must_use]
pub fn e8m0_encode_pow2_ceil(value: f32) -> u8 {
    if value.is_nan() || value <= 0.0 {
        return 0;
    }
    if value.is_infinite() {
        return 0xFE;
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;
    let ceil_log2 = if exponent == 0 {
        // value = mantissa * 2^-149 with the mantissa a positive integer below 2^23, so the
        // ceiling is the highest set bit, bumped by one unless the mantissa is a power of two.
        let highest = 31 - mantissa.leading_zeros() as i32;
        -149 + highest + i32::from(!mantissa.is_power_of_two())
    } else {
        exponent - 127 + i32::from(mantissa != 0)
    };
    (ceil_log2 + 127).clamp(0, 254) as u8
}

/// 2^exponent for the small integer exponents these codecs use.
fn exp2(exponent: i32) -> f32 {
    f32::from_bits(((exponent + 127) as u32) << 23)
}

/// The E4M3 scale the main KV quantiser uses for one 16-channel group.
///
/// Mirrors the reference: floor amax at 6 * 2^-9 so an all-zero group keeps a non-zero scale,
/// divide by 6, round to E4M3, and use the dequantised value for the division.
#[must_use]
pub fn main_kv_group_scale(amax: f32) -> f32 {
    let floored = amax.max(6.0 * exp2(-9));
    e4m3_decode(e4m3_encode(floored / E2M1_MAX))
}

/// The E8M0 scale the indexer K quantiser uses for one 32-channel group.
#[must_use]
pub fn indexer_k_group_scale(amax: f32) -> f32 {
    let floored = amax.max(6.0 * exp2(-126));
    e8m0_decode(e8m0_encode_pow2_ceil(floored / E2M1_MAX))
}

/// The E8M0 scale the SWA quantiser uses for one 32-channel group.
///
/// The reference act_quant floors amax at 1e-4 and divides by the FP8 maximum 448.
#[must_use]
pub fn swa_group_scale(amax: f32) -> f32 {
    let floored = amax.max(1e-4);
    e8m0_decode(e8m0_encode_pow2_ceil(floored / E4M3_MAX))
}

/// Packs two 4-bit codes per byte, low nibble first.
fn pack_nibbles(codes: &[u8], out: &mut [u8]) {
    for (index, code) in codes.iter().enumerate() {
        if index % 2 == 0 {
            out[index / 2] = code & 0x0F;
        } else {
            out[index / 2] |= (code & 0x0F) << 4;
        }
    }
}

/// Unpacks two 4-bit codes per byte, low nibble first.
fn unpack_nibbles(bytes: &[u8], out: &mut [u8]) {
    for (index, byte) in bytes.iter().enumerate() {
        out[index * 2] = byte & 0x0F;
        out[index * 2 + 1] = (byte >> 4) & 0x0F;
    }
}

fn check_len(context: &'static str, expected: usize, actual: usize) -> Result<(), KvError> {
    if expected == actual {
        Ok(())
    } else {
        Err(KvError::InvalidLength {
            context,
            expected,
            actual,
        })
    }
}

/// One 512-wide MLA latent quantised to FP4 E2M1 with per-16 E4M3 scales.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct MainKvEntry {
    /// 256 bytes of packed FP4 codes.
    pub data: [u8; MAIN_KV_DATA_BYTES],
    /// 32 E4M3 group scales.
    pub scales: [u8; MAIN_KV_SCALE_BYTES],
}

impl Default for MainKvEntry {
    fn default() -> Self {
        Self {
            data: [0u8; MAIN_KV_DATA_BYTES],
            scales: [0u8; MAIN_KV_SCALE_BYTES],
        }
    }
}

impl std::fmt::Debug for MainKvEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MainKvEntry")
            .field("data", &self.data.len())
            .field("scales", &self.scales.len())
            .finish()
    }
}

impl MainKvEntry {
    /// Serialises the entry into a 288-byte buffer.
    ///
    /// # Errors
    /// Returns KvError::InvalidLength when the buffer is not exactly 288 bytes.
    pub fn to_bytes(&self, out: &mut [u8]) -> Result<(), KvError> {
        check_len(
            "main kv entry",
            crate::layout::MAIN_KV_ENTRY_BYTES,
            out.len(),
        )?;
        out[..MAIN_KV_DATA_BYTES].copy_from_slice(&self.data);
        out[MAIN_KV_DATA_BYTES..].copy_from_slice(&self.scales);
        Ok(())
    }

    /// Parses an entry from a 288-byte buffer.
    ///
    /// # Errors
    /// Returns KvError::InvalidLength when the buffer is not exactly 288 bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, KvError> {
        check_len(
            "main kv entry",
            crate::layout::MAIN_KV_ENTRY_BYTES,
            bytes.len(),
        )?;
        let mut entry = Self::default();
        entry.data.copy_from_slice(&bytes[..MAIN_KV_DATA_BYTES]);
        entry.scales.copy_from_slice(&bytes[MAIN_KV_DATA_BYTES..]);
        Ok(entry)
    }
}

/// One 128-wide indexer key quantised to FP4 E2M1 with per-32 E8M0 scales.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct IndexerKEntry {
    /// 64 bytes of packed FP4 codes.
    pub data: [u8; INDEXER_K_DATA_BYTES],
    /// 4 E8M0 group scales.
    pub scales: [u8; INDEXER_K_SCALE_BYTES],
}

impl Default for IndexerKEntry {
    fn default() -> Self {
        Self {
            data: [0u8; INDEXER_K_DATA_BYTES],
            scales: [0u8; INDEXER_K_SCALE_BYTES],
        }
    }
}

impl std::fmt::Debug for IndexerKEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexerKEntry")
            .field("data", &self.data.len())
            .field("scales", &self.scales.len())
            .finish()
    }
}

impl IndexerKEntry {
    /// Serialises the entry into a 68-byte buffer.
    ///
    /// # Errors
    /// Returns KvError::InvalidLength when the buffer is not exactly 68 bytes.
    pub fn to_bytes(&self, out: &mut [u8]) -> Result<(), KvError> {
        check_len(
            "indexer k entry",
            crate::layout::INDEXER_K_ENTRY_BYTES,
            out.len(),
        )?;
        out[..INDEXER_K_DATA_BYTES].copy_from_slice(&self.data);
        out[INDEXER_K_DATA_BYTES..].copy_from_slice(&self.scales);
        Ok(())
    }

    /// Parses an entry from a 68-byte buffer.
    ///
    /// # Errors
    /// Returns KvError::InvalidLength when the buffer is not exactly 68 bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, KvError> {
        check_len(
            "indexer k entry",
            crate::layout::INDEXER_K_ENTRY_BYTES,
            bytes.len(),
        )?;
        let mut entry = Self::default();
        entry.data.copy_from_slice(&bytes[..INDEXER_K_DATA_BYTES]);
        entry.scales.copy_from_slice(&bytes[INDEXER_K_DATA_BYTES..]);
        Ok(entry)
    }
}

/// One SWA slot: a 512-wide FP8 E4M3 vector with per-32 E8M0 scales.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SwaKvSlot {
    /// 512 bytes of E4M3 codes.
    pub data: [u8; SWA_SLOT_DATA_BYTES],
    /// 16 E8M0 group scales.
    pub scales: [u8; SWA_SLOT_SCALE_BYTES],
}

impl Default for SwaKvSlot {
    fn default() -> Self {
        Self {
            data: [0u8; SWA_SLOT_DATA_BYTES],
            scales: [0u8; SWA_SLOT_SCALE_BYTES],
        }
    }
}

impl std::fmt::Debug for SwaKvSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SwaKvSlot")
            .field("data", &self.data.len())
            .field("scales", &self.scales.len())
            .finish()
    }
}

/// Quantises a 512-wide vector to one main KV entry.
///
/// # Errors
/// Returns KvError::InvalidLength when the slice is not exactly 512 values.
pub fn quantize_main_kv(values: &[f32]) -> Result<MainKvEntry, KvError> {
    check_len("main kv values", MAIN_KV_DIM, values.len())?;
    let mut entry = MainKvEntry::default();
    let mut codes = [0u8; MAIN_KV_DIM];
    for group in 0..MAIN_KV_DIM / MAIN_KV_SCALE_GROUP {
        let block = &values[group * MAIN_KV_SCALE_GROUP..(group + 1) * MAIN_KV_SCALE_GROUP];
        let amax = block.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let scale = main_kv_group_scale(amax);
        entry.scales[group] = e4m3_encode(amax.max(6.0 * exp2(-9)) / E2M1_MAX);
        for (offset, value) in block.iter().enumerate() {
            codes[group * MAIN_KV_SCALE_GROUP + offset] =
                e2m1_encode((value / scale).clamp(-E2M1_MAX, E2M1_MAX));
        }
    }
    pack_nibbles(&codes, &mut entry.data);
    Ok(entry)
}

/// Dequantises one main KV entry back to 512 f32 values.
///
/// # Errors
/// Returns KvError::InvalidLength when the output slice is not exactly 512 values.
pub fn dequantize_main_kv(entry: &MainKvEntry, out: &mut [f32]) -> Result<(), KvError> {
    check_len("main kv values", MAIN_KV_DIM, out.len())?;
    let mut codes = [0u8; MAIN_KV_DIM];
    unpack_nibbles(&entry.data, &mut codes);
    for group in 0..MAIN_KV_DIM / MAIN_KV_SCALE_GROUP {
        let scale = e4m3_decode(entry.scales[group]);
        for offset in 0..MAIN_KV_SCALE_GROUP {
            let index = group * MAIN_KV_SCALE_GROUP + offset;
            out[index] = e2m1_decode(codes[index]) * scale;
        }
    }
    Ok(())
}

/// Quantises a 128-wide vector to one indexer K entry.
///
/// # Errors
/// Returns KvError::InvalidLength when the slice is not exactly 128 values.
pub fn quantize_indexer_k(values: &[f32]) -> Result<IndexerKEntry, KvError> {
    check_len("indexer k values", INDEXER_K_DIM, values.len())?;
    let mut entry = IndexerKEntry::default();
    let mut codes = [0u8; INDEXER_K_DIM];
    for group in 0..INDEXER_K_DIM / INDEXER_K_SCALE_GROUP {
        let block = &values[group * INDEXER_K_SCALE_GROUP..(group + 1) * INDEXER_K_SCALE_GROUP];
        let amax = block.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let scale = indexer_k_group_scale(amax);
        let floored = amax.max(6.0 * exp2(-126));
        entry.scales[group] = e8m0_encode_pow2_ceil(floored / E2M1_MAX);
        for (offset, value) in block.iter().enumerate() {
            codes[group * INDEXER_K_SCALE_GROUP + offset] =
                e2m1_encode((value / scale).clamp(-E2M1_MAX, E2M1_MAX));
        }
    }
    pack_nibbles(&codes, &mut entry.data);
    Ok(entry)
}

/// Dequantises one indexer K entry back to 128 f32 values.
///
/// # Errors
/// Returns KvError::InvalidLength when the output slice is not exactly 128 values.
pub fn dequantize_indexer_k(entry: &IndexerKEntry, out: &mut [f32]) -> Result<(), KvError> {
    check_len("indexer k values", INDEXER_K_DIM, out.len())?;
    let mut codes = [0u8; INDEXER_K_DIM];
    unpack_nibbles(&entry.data, &mut codes);
    for group in 0..INDEXER_K_DIM / INDEXER_K_SCALE_GROUP {
        let scale = e8m0_decode(entry.scales[group]);
        for offset in 0..INDEXER_K_SCALE_GROUP {
            let index = group * INDEXER_K_SCALE_GROUP + offset;
            out[index] = e2m1_decode(codes[index]) * scale;
        }
    }
    Ok(())
}

/// Quantises a 512-wide vector to one SWA slot.
///
/// # Errors
/// Returns KvError::InvalidLength when the slice is not exactly 512 values.
pub fn quantize_swa_slot(values: &[f32]) -> Result<SwaKvSlot, KvError> {
    check_len("swa slot values", SWA_SLOT_DIM, values.len())?;
    let mut slot = SwaKvSlot::default();
    for group in 0..SWA_SLOT_DIM / SWA_FP8_SCALE_GROUP {
        let block = &values[group * SWA_FP8_SCALE_GROUP..(group + 1) * SWA_FP8_SCALE_GROUP];
        let amax = block.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let scale = swa_group_scale(amax);
        slot.scales[group] = e8m0_encode_pow2_ceil(amax.max(1e-4) / E4M3_MAX);
        for (offset, value) in block.iter().enumerate() {
            slot.data[group * SWA_FP8_SCALE_GROUP + offset] =
                e4m3_encode((value / scale).clamp(-E4M3_MAX, E4M3_MAX));
        }
    }
    Ok(slot)
}

/// Dequantises one SWA slot back to 512 f32 values.
///
/// # Errors
/// Returns KvError::InvalidLength when the output slice is not exactly 512 values.
pub fn dequantize_swa_slot(slot: &SwaKvSlot, out: &mut [f32]) -> Result<(), KvError> {
    check_len("swa slot values", SWA_SLOT_DIM, out.len())?;
    for group in 0..SWA_SLOT_DIM / SWA_FP8_SCALE_GROUP {
        let scale = e8m0_decode(slot.scales[group]);
        for offset in 0..SWA_FP8_SCALE_GROUP {
            let index = group * SWA_FP8_SCALE_GROUP + offset;
            out[index] = e4m3_decode(slot.data[index]) * scale;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_grid_is_exact() {
        let expected = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (index, value) in expected.iter().enumerate() {
            assert_eq!(e2m1_decode(index as u8), *value);
            assert_eq!(e2m1_encode(*value), index as u8);
            assert_eq!(e2m1_decode(0x8 | index as u8), -*value);
            assert_eq!(e2m1_encode(-*value), 0x8 | index as u8);
        }
    }

    #[test]
    fn e2m1_ties_go_to_even_mantissa() {
        assert_eq!(e2m1_decode(e2m1_encode(0.75)), 1.0);
        assert_eq!(e2m1_decode(e2m1_encode(2.5)), 2.0);
        assert_eq!(e2m1_decode(e2m1_encode(1.25)), 1.0);
    }

    #[test]
    fn e2m1_policy_for_nan_and_infinity_is_pinned() {
        assert_eq!(e2m1_encode(f32::NAN), 0);
        assert_eq!(e2m1_encode(f32::INFINITY), 0x7);
        assert_eq!(e2m1_encode(f32::NEG_INFINITY), 0xF);
        assert_eq!(e2m1_encode(-f32::INFINITY), 0xF);
    }
}

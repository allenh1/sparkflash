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

//! CPU reference for the greedy head's screen (kernels/opt_rows_sm121.cu, opt_head_screen_*).
//!
//! A greedy step wants one thing from the head: the arg-max of a rank's 32,320 logits per row.
//! opt_rows_lm_head reads the whole bf16 head slice to get it. The screen reads an int8 copy of
//! the slice instead, bounds every logit from both sides with an exact integer dot product, and
//! leaves opt_rows_lm_head's own arithmetic to the few entries whose upper bound reaches the best
//! lower bound. The output is that kernel's bits; the screen only decides which entries need not
//! be scored.
//!
//! This module is the screen's arithmetic on the host, operation for operation where the device
//! result must be the same bits (the codes, the digits, the two scales and the proxy), and in f64
//! where the device only has to stay on the safe side (the norms the bound multiplies). It serves
//! two readers:
//!
//! - the tests below, which run the bound against an emulation of opt_rows_lm_head's f32
//!   arithmetic (160 FMAs a lane, then the five-level lane sum) on rows chosen to stress it, and
//!   show by mutation that each term of the bound is load-bearing;
//! - the tail stage's self test (`diagnostics.head_screen_check`), which compares what the device
//!   kernels built and computed against these functions for every entry of the real head.
//!
//! The bound. Write E for the logit opt_rows_lm_head computes for (row h, entry w), u = 2^-24:
//!
//!   |E - w.h| <= gamma_165 sum |w_j h_j| <= 1e-5 |w| |h|
//!
//! (every product bf16 x bf16 is exact, so only the 165 additions round). The codes are q = rint(w
//! / s) with s = max |w| / 127, and the row is an integer H = rint(h / t), t = max |h| / 16256,
//! carried as two int8 digits H = 128 p1 + p0. The screen's value is P = fl(fl(s t) fl(q.H)) with
//! q.H an exact integer, so
//!
//!   |P - E| <= a |h| + b |h - t H| + 3.1 u |P|,   a = |w - s q| + 1e-5 |w|,   b = |s q|.
//!
//! The device stores a and b 1% high and the row norms 0.05% high, and adds 1e-6 |P| + 1e-30. The
//! first line needs every partial sum to stay finite, so a row or an entry whose squares sum to
//! [`SS_MAX`] or more is never ruled out.

use crate::m4_quant::bf16_bits_to_f32;

/// The hidden width.
pub const HIDDEN: usize = 5120;
/// Entries a tile of codes (OHS_TILE).
pub const TILE: usize = 128;
/// K blocks of 32 a row (OHS_KB).
pub const K_BLOCKS: usize = HIDDEN / 32;
/// The largest |H| (OHS_DIGIT): 127 * 128.
pub const DIGIT_MAX: f32 = 16256.0;
/// A sum of squares the bound's no-overflow premise allows (OHS_SS_MAX).
pub const SS_MAX: f32 = 1e30;

/// Where the code of entry `entry` at hidden index `k` lives in the tiled table: K block b of
/// entry v at ((v / 128 * 160 + b) * 128 + v % 128) * 32.
#[must_use]
pub const fn code_offset(entry: usize, k: usize) -> usize {
    ((entry / TILE * K_BLOCKS + k / 32) * TILE + entry % TILE) * 32 + k % 32
}

/// The bytes of a table over `vocab` entries (whole tiles).
#[must_use]
pub const fn table_bytes(vocab: usize) -> usize {
    vocab.div_ceil(TILE) * TILE * HIDDEN
}

/// The MMA row of digit `digit` of hidden row `row` (ohs_act_row): rows 0..8 of a 16-row tile are
/// the low digits of eight hidden rows, rows 8..16 their high digits.
#[must_use]
pub const fn digit_row(row: usize, digit: usize) -> usize {
    (row >> 3) * 16 + digit * 8 + (row & 7)
}

/// One head entry as the screen holds it.
#[derive(Debug, Clone)]
pub struct EntryCodes {
    /// s: the entry's scale, the device's bits.
    pub scale: f32,
    /// q: the int8 codes, the device's bits.
    pub codes: Vec<i8>,
    /// |w - s q|, exact to f64.
    pub residual_norm: f64,
    /// |s q|, exact to f64.
    pub coded_norm: f64,
    /// |w|, exact to f64.
    pub norm: f64,
}

impl EntryCodes {
    /// Whether the bound covers the entry: its squares sum to less than [`SS_MAX`]. (The device
    /// takes the sum in f32, so its cut is not this one to the last bit; nothing real is near it.)
    #[must_use]
    pub fn covered(&self) -> bool {
        self.norm * self.norm < f64::from(SS_MAX)
    }

    /// The least a covered entry's `a` may be: |w - s q| + 1e-5 |w|.
    #[must_use]
    pub fn a_floor(&self) -> f64 {
        self.residual_norm + 1e-5 * self.norm
    }
}

/// fminf(fmaxf(x, -limit), limit): a NaN comes out as -limit, as it does on the device.
fn device_clamp(x: f32, limit: f32) -> f32 {
    let low = if x > -limit { x } else { -limit };
    if low < limit { low } else { limit }
}

/// opt_head_screen_build's quantiser for one entry (bf16 bits, [`HIDDEN`] of them).
#[must_use]
pub fn entry_codes(w: &[u16]) -> EntryCodes {
    let amax = w
        .iter()
        .fold(0.0f32, |m, &bits| m.max(bf16_bits_to_f32(bits).abs()));
    let scale = amax / 127.0;
    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    let mut codes = Vec::with_capacity(w.len());
    let (mut ss_r, mut ss_q, mut ss_w) = (0.0f64, 0.0f64, 0.0f64);
    for &bits in w {
        let v = bf16_bits_to_f32(bits);
        let q = device_clamp((v * inv).round_ties_even(), 127.0);
        let coded = f64::from(scale) * f64::from(q);
        let r = f64::from(v) - coded;
        ss_r += r * r;
        ss_q += coded * coded;
        ss_w += f64::from(v) * f64::from(v);
        codes.push(q as i8);
    }
    EntryCodes {
        scale,
        codes,
        residual_norm: ss_r.sqrt(),
        coded_norm: ss_q.sqrt(),
        norm: ss_w.sqrt(),
    }
}

/// One hidden row as the screen holds it.
#[derive(Debug, Clone)]
pub struct RowDigits {
    /// Whether the bound covers the row; a row that is all zero, not finite or too large is not
    /// screened (every entry is scored).
    pub usable: bool,
    /// t: the row's scale, the device's bits (0 for a row that is not usable).
    pub scale: f32,
    /// H = 128 p1 + p0, |H| <= 16256.
    pub digits: Vec<i16>,
    /// |h|, exact to f64.
    pub norm: f64,
    /// |h - t H|, exact to f64.
    pub dropped_norm: f64,
}

impl RowDigits {
    /// The int8 pair (p0, p1) of digit index `k`: p1 = (H + 64) >> 7, p0 = H - 128 p1 in [-64, 63].
    #[must_use]
    pub fn pair(&self, k: usize) -> (i8, i8) {
        let h = i32::from(self.digits[k]);
        let p1 = (h + 64) >> 7;
        ((h - p1 * 128) as i8, p1 as i8)
    }
}

/// opt_head_screen_rows' quantiser for one hidden row (bf16 bits, [`HIDDEN`] of them).
#[must_use]
pub fn row_digits(h: &[u16]) -> RowDigits {
    let amax = h
        .iter()
        .fold(0.0f32, |m, &bits| m.max(bf16_bits_to_f32(bits).abs()));
    let ss_h: f64 = h
        .iter()
        .map(|&bits| {
            let v = f64::from(bf16_bits_to_f32(bits));
            v * v
        })
        .sum();
    // The device takes the sum in f32, so its cut is not this one to the last bit; the self test
    // keeps its rows well away from the cut.
    let usable = amax > 0.0 && ss_h < f64::from(SS_MAX);
    let scale = if usable { amax / DIGIT_MAX } else { 0.0 };
    let inv = if usable { 1.0 / scale } else { 0.0 };
    let mut digits = Vec::with_capacity(h.len());
    let mut ss_d = 0.0f64;
    for &bits in h {
        let v = if usable { bf16_bits_to_f32(bits) } else { 0.0 };
        let hf = device_clamp((v * inv).round_ties_even(), DIGIT_MAX);
        let r = f64::from(v) - f64::from(scale) * f64::from(hf);
        ss_d += r * r;
        digits.push(hf as i16);
    }
    RowDigits {
        usable,
        scale,
        digits,
        norm: ss_h.sqrt(),
        dropped_norm: ss_d.sqrt(),
    }
}

/// q.H, exact.
#[must_use]
pub fn integer_dot(codes: &[i8], digits: &[i16]) -> i64 {
    // 512 terms of at most 127 * 16256 stay inside an i32.
    codes
        .chunks(512)
        .zip(digits.chunks(512))
        .map(|(q, h)| {
            i64::from(
                q.iter()
                    .zip(h)
                    .map(|(&q, &h)| i32::from(q) * i32::from(h))
                    .sum::<i32>(),
            )
        })
        .sum()
}

/// The screen's value P = fl(fl(s t) fl(q.H)), the device's bits.
#[must_use]
pub fn proxy(entry_scale: f32, row_scale: f32, dot: i64) -> f32 {
    (entry_scale * row_scale) * (dot as f32)
}

/// The half width the device puts around P, from the coefficients it stored (`a`, `b`) and the
/// row norms it took (`row_norm`, `dropped_norm`). The device may fuse these operations, so its
/// value is this one to a few units in the last place, not to the bit.
#[must_use]
pub fn half_width(a: f32, b: f32, row_norm: f32, dropped_norm: f32, p: f32) -> f32 {
    a * row_norm + b * dropped_norm + 1e-6 * p.abs() + 1e-30
}

/// The coefficients the device stores for an entry, from the exact norms: {a, b}, 1% high.
#[must_use]
pub fn entry_coefficients(entry: &EntryCodes) -> (f32, f32) {
    (
        if entry.covered() {
            (1.01 * entry.a_floor() + 1e-17) as f32
        } else {
            f32::INFINITY
        },
        (1.01 * entry.coded_norm + 1e-17) as f32,
    )
}

/// The norms the device takes for a row, from the exact ones: {|h|, |h - t H|}, 0.05% high; a row
/// that is not usable has an infinite |h|.
#[must_use]
pub fn row_norms(row: &RowDigits) -> (f32, f32) {
    let high = |norm: f64| ((norm * norm * 1.001).sqrt() * 1.0001 + 1e-17) as f32;
    (
        if row.usable {
            high(row.norm)
        } else {
            f32::INFINITY
        },
        high(row.dropped_norm),
    )
}

/// opt_rows_lm_head's logit for (entry `w`, row `h`), its f32 arithmetic: lane l accumulates the
/// eight values of every 32nd 16-byte word from zero with one FMA each, and the lanes are summed
/// by five xor shuffles.
#[must_use]
pub fn head_logit(w: &[u16], h: &[u16]) -> f32 {
    let mut lanes = [0.0f32; 32];
    for (lane, acc) in lanes.iter_mut().enumerate() {
        for word in 0..HIDDEN / 256 {
            let at = (lane + 32 * word) * 8;
            for k in at..at + 8 {
                *acc = bf16_bits_to_f32(w[k]).mul_add(bf16_bits_to_f32(h[k]), *acc);
            }
        }
    }
    for offset in [16usize, 8, 4, 2, 1] {
        let before = lanes;
        for (lane, acc) in lanes.iter_mut().enumerate() {
            *acc = before[lane] + before[lane ^ offset];
        }
    }
    lanes[0]
}

/// The screened arg-max of `rows` hidden rows over a head of whole entries: per row the
/// (value, lowest index) of the best logit among the entries the bounds keep, and how many they
/// kept. `logit` is the exact scorer (opt_rows_lm_head's arithmetic).
#[must_use]
pub fn screened_argmax(head: &[u16], hidden: &[u16]) -> Vec<(f32, usize, usize)> {
    let entries: Vec<EntryCodes> = head.chunks(HIDDEN).map(entry_codes).collect();
    let coefficients: Vec<(f32, f32)> = entries.iter().map(entry_coefficients).collect();
    hidden
        .chunks(HIDDEN)
        .map(|h| {
            let row = row_digits(h);
            let (row_norm, dropped_norm) = row_norms(&row);
            let bounds: Vec<(f32, f32)> = entries
                .iter()
                .zip(&coefficients)
                .map(|(entry, &(a, b))| {
                    let p = proxy(
                        entry.scale,
                        row.scale,
                        integer_dot(&entry.codes, &row.digits),
                    );
                    let half = half_width(a, b, row_norm, dropped_norm, p);
                    (p - half, p + half)
                })
                .collect();
            // A NaN never wins the threshold and never rules an entry out.
            let threshold =
                bounds.iter().fold(
                    f32::NEG_INFINITY,
                    |m, &(lower, _)| if lower > m { lower } else { m },
                );
            let mut best: Option<(f32, usize)> = None;
            let mut kept = 0usize;
            for (index, &(_, upper)) in bounds.iter().enumerate() {
                if upper < threshold {
                    continue;
                }
                kept += 1;
                let value = head_logit(&head[index * HIDDEN..(index + 1) * HIDDEN], h);
                if best.is_none_or(|(top, _)| value > top) {
                    best = Some((value, index));
                }
            }
            let (value, index) = best.unwrap_or((f32::NEG_INFINITY, 0));
            (value, index, kept)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::m4_quant::bf16_bits;

    /// A reproducible stream of values in [-1, 1) with a bell-shaped spread.
    struct Stream(u64);

    impl Stream {
        fn unit(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }

        fn bell(&mut self) -> f32 {
            (self.unit() + self.unit() + self.unit() + self.unit()) * 0.5
        }

        fn row(&mut self, scale: f32) -> Vec<u16> {
            (0..HIDDEN)
                .map(|_| bf16_bits(self.bell() * scale))
                .collect()
        }
    }

    /// The rows the bound is run against: plain rows at several scales, a row with a few large
    /// values (which makes the code grid coarse for the rest), a sparse row, and tiny values.
    fn stress_rows(stream: &mut Stream) -> Vec<Vec<u16>> {
        let mut rows = Vec::new();
        for scale in [1.0e-3f32, 0.02, 1.0, 37.0, 4.0e4] {
            rows.push(stream.row(scale));
        }
        let mut spiked = stream.row(0.05);
        for k in [3usize, 911, 4095] {
            spiked[k] = bf16_bits(9.0);
        }
        rows.push(spiked);
        let mut sparse = vec![0u16; HIDDEN];
        for k in (0..HIDDEN).step_by(97) {
            sparse[k] = bf16_bits(stream.bell());
        }
        rows.push(sparse);
        rows.push(stream.row(1.0e-12));
        rows
    }

    fn bounds_of(w: &[u16], h: &[u16]) -> (f32, f32, f32) {
        let entry = entry_codes(w);
        let row = row_digits(h);
        let (a, b) = entry_coefficients(&entry);
        let (row_norm, dropped_norm) = row_norms(&row);
        let p = proxy(
            entry.scale,
            row.scale,
            integer_dot(&entry.codes, &row.digits),
        );
        (p, half_width(a, b, row_norm, dropped_norm, p), a * row_norm)
    }

    #[test]
    fn the_codes_and_digits_fit_their_bytes() {
        let mut stream = Stream(0x9e37_79b9_7f4a_7c15);
        for row in stress_rows(&mut stream) {
            let entry = entry_codes(&row);
            assert!(entry.codes.iter().all(|&q| q >= -127));
            let digits = row_digits(&row);
            assert!(digits.usable);
            for k in 0..HIDDEN {
                let (p0, p1) = digits.pair(k);
                assert!((-64..=63).contains(&p0), "p0 {p0}");
                assert!((-127..=127).contains(&i16::from(p1)), "p1 {p1}");
                assert_eq!(i16::from(p1) * 128 + i16::from(p0), digits.digits[k]);
            }
        }
    }

    #[test]
    fn the_table_layout_is_one_contiguous_stage_a_tile_and_k_block() {
        assert_eq!(code_offset(0, 0), 0);
        assert_eq!(code_offset(1, 0), 32);
        assert_eq!(code_offset(0, 32), TILE * 32);
        assert_eq!(code_offset(TILE, 0), K_BLOCKS * TILE * 32);
        assert_eq!(table_bytes(32_320), 253 * TILE * HIDDEN);
        assert_eq!(digit_row(0, 1), 8);
        assert_eq!(digit_row(8, 0), 16);
        assert_eq!(digit_row(63, 1), 127);
    }

    #[test]
    fn the_bound_holds_against_the_heads_own_arithmetic() {
        let mut stream = Stream(0x1234_5678_9abc_def1);
        let heads = stress_rows(&mut stream);
        let hiddens = stress_rows(&mut stream);
        let mut widest = 0.0f32;
        for w in &heads {
            for h in &hiddens {
                let exact = head_logit(w, h);
                let (p, half, _) = bounds_of(w, h);
                assert!(
                    p - half <= exact && exact <= p + half,
                    "logit {exact} is outside {p} +- {half}"
                );
                let scale = (entry_codes(w).norm * row_digits(h).norm) as f32;
                widest = widest.max(half / scale);
            }
            // The entry against itself: the largest logit a row of that norm can have.
            let exact = head_logit(w, w);
            let (p, half, _) = bounds_of(w, w);
            assert!(p - half <= exact && exact <= p + half);
        }
        // The bound is worth having: a few percent of |w| |h| even on the coarse rows.
        assert!(widest < 0.2, "the widest half width is {widest} of |w| |h|");
    }

    #[test]
    fn each_term_of_the_bound_is_load_bearing() {
        let mut stream = Stream(0xfeed_face_cafe_beef);
        // Without the entry's quantisation term the interval misses real logits.
        let heads: Vec<Vec<u16>> = (0..24).map(|_| stream.row(0.03)).collect();
        let hiddens: Vec<Vec<u16>> = (0..24).map(|_| stream.row(2.0)).collect();
        let mut missed = 0usize;
        for w in &heads {
            for h in &hiddens {
                let exact = head_logit(w, h);
                let (p, half, a_term) = bounds_of(w, h);
                assert!(p - half <= exact && exact <= p + half);
                let narrow = half - a_term;
                if exact < p - narrow || exact > p + narrow {
                    missed += 1;
                }
            }
        }
        assert!(missed > 0, "the a |h| term never mattered on these rows");
        // Without the row's own term a row whose small values fall under its grid is missed. The
        // entry is constant at 127 / 512 (its codes are exact, so a is the 1e-5 |w| alone) and the row's small
        // values all have its sign, which makes |s q| |h - t H| tight: the interval has to hold
        // with almost nothing to spare, and cannot without that term.
        let w = vec![bf16_bits(127.0 / 512.0); HIDDEN];
        let mut h = vec![bf16_bits(0.0015); HIDDEN];
        for k in [5usize, 2000, 5000] {
            h[k] = bf16_bits(60.0);
        }
        let entry = entry_codes(&w);
        assert!(entry.residual_norm == 0.0 && entry.codes.iter().all(|&q| q == 127));
        let row = row_digits(&h);
        assert_eq!(row.digits.iter().filter(|&&d| d != 0).count(), 3);
        let exact = head_logit(&w, &h);
        let (p, half, _) = bounds_of(&w, &h);
        assert!(p - half <= exact && exact <= p + half);
        assert!(
            (exact - p).abs() > 0.9 * half,
            "the case is tight: {exact} {p} {half}"
        );
        let (a, _) = entry_coefficients(&entry);
        let (row_norm, _) = row_norms(&row);
        let without = half_width(a, 0.0, row_norm, 0.0, p);
        assert!(
            (exact - p).abs() > without,
            "the b |h - t H| term did not matter"
        );
    }

    #[test]
    fn a_row_the_bound_does_not_cover_is_never_ruled_out() {
        let mut stream = Stream(0x0bad_cafe_dead_beef);
        let head: Vec<u16> = (0..8).flat_map(|_| stream.row(0.05)).collect();
        let zero = vec![0u16; HIDDEN];
        let mut not_finite = stream.row(1.0);
        not_finite[17] = 0x7fc0; // a NaN
        let huge = vec![bf16_bits(3.0e16); HIDDEN];
        for h in [zero, not_finite, huge] {
            let row = row_digits(&h);
            assert!(!row.usable);
            assert_eq!(row_norms(&row).0, f32::INFINITY);
            let kept = screened_argmax(&head, &h)[0].2;
            assert_eq!(kept, 8, "every entry is scored");
        }
        // An entry too large for the bound is never ruled out either.
        let mut wide = head.clone();
        for value in &mut wide[..HIDDEN] {
            *value = bf16_bits(2.0e13);
        }
        assert_eq!(
            entry_coefficients(&entry_codes(&wide[..HIDDEN])).0,
            f32::INFINITY
        );
    }

    #[test]
    fn the_screened_argmax_is_the_full_heads() {
        let mut stream = Stream(0x5eed_1e55_0fa1_1e55);
        let vocab = 96usize;
        let mut head: Vec<u16> = (0..vocab).flat_map(|_| stream.row(0.03)).collect();
        // Two identical entries, so the best logit of a row aligned with them is tied.
        let twin: Vec<u16> = head[40 * HIDDEN..41 * HIDDEN].to_vec();
        head[77 * HIDDEN..78 * HIDDEN].copy_from_slice(&twin);
        let mut hidden: Vec<u16> = (0..6).flat_map(|_| stream.row(1.5)).collect();
        // A row aligned with entry 12, and one aligned with the twins.
        let aligned: Vec<u16> = head[12 * HIDDEN..13 * HIDDEN]
            .iter()
            .map(|&bits| bf16_bits(bf16_bits_to_f32(bits) * 40.0))
            .collect();
        hidden.extend_from_slice(&aligned);
        let tied: Vec<u16> = twin
            .iter()
            .map(|&bits| bf16_bits(bf16_bits_to_f32(bits) * 40.0))
            .collect();
        hidden.extend_from_slice(&tied);
        let screened = screened_argmax(&head, &hidden);
        let mut skipped = 0usize;
        for (row, h) in hidden.chunks(HIDDEN).enumerate() {
            let mut best = (f32::NEG_INFINITY, 0usize);
            for index in 0..vocab {
                let value = head_logit(&head[index * HIDDEN..(index + 1) * HIDDEN], h);
                if index == 0 || value > best.0 {
                    best = (value, index);
                }
            }
            let (value, index, kept) = screened[row];
            assert_eq!(value.to_bits(), best.0.to_bits(), "row {row}");
            assert_eq!(index, best.1, "row {row}");
            skipped += vocab - kept;
        }
        assert_eq!(screened[6].1, 12);
        assert_eq!(screened[7].1, 40, "a tie goes to the lower entry");
        assert!(
            screened[6].2 < 4 && screened[7].2 < 4,
            "an aligned row keeps few entries"
        );
        assert!(
            skipped > 0,
            "a screen that rules nothing out proves nothing"
        );
    }
}

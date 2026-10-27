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

//! CPU reference for the certified LM head (kernels/cert_lm_head_sm121.cu).
//!
//! The LM head is the most expensive single step of decode: vocab times hidden multiply-accumulates.
//! The kernel screens each vocabulary entry with a cheap upper bound and skips the exact dot product
//! when the bound proves the entry cannot be the maximum:
//!
//!   bound_j = sum over the 40 hidden blocks b of ( sum_{i in b} |h_i| ) * ( max_{i in b} |w_ji| )
//!
//!   |sum_i h_i w_ji| <= sum_i |h_i| |w_ji| <= sum_b (sum_{i in b} |h_i|) * (max_{i in b} |w_ji|)
//!
//! and the scan keeps a running maximum, skipping an entry only when bound_j < running_max. Because
//! the running maximum never decreases outside a skipped entry, every skipped entry satisfies
//! logits_j <= bound_j < running_max, so pruning can never hide the true maximum. This module defines
//! that rule and the tests check the three claims that make it a certificate:
//!
//! 1. an entry with certified == 0 is bit-for-bit the exact dot product;
//! 2. an entry with certified == 1 carries a bound that is still an upper bound on its exact value;
//! 3. the maximum survives the pruning, and at least one entry really is skipped (a certificate
//!    that prunes nothing proves nothing). The argmax survives it too, but only because the
//!    fixture's maximum happens to be unique - that is checked rather than assumed, and ties have
//!    their own tests: the kernel deals entries round-robin and combines lanes in ascending order,
//!    so a value-only reduction returns the earlier LANE while the rule this module implements
//!    returns the lower ENTRY INDEX. a_tie_is_broken_by_the_lower_entry_index... runs the kernel's
//!    own order in Rust to show the two agree, and the value-only mutant to show they are not
//!    trivially the same.
//!
//! Plus a mutation proof: the bound is only valid because it takes the ABSOLUTE value of the weight.
//! The tests build a mutant that takes the signed maximum and show it certifies an entry whose exact
//! value beats the running maximum - that is, it produces a wrong answer, which is what makes the
//! absolute value load-bearing rather than decorative.
//!
//! What is NOT here: any claim that the screen is faster than the exact path on any particular
//! hardware. On scalar code the bound costs a max-reduce per 128 values while the exact path costs a
//! multiply-accumulate, and which wins belongs to a GPU measurement.

use crate::m4_quant::bf16_bits_to_f32;

/// The hidden width. Mirrors SPARK_M4_HIDDEN.
pub const HIDDEN: usize = 5120;
/// Entries of the hidden state per screen block. Mirrors SPARK_M4_LM_SCREEN_TILE.
pub const SCREEN_TILE: usize = 128;
/// The number of screen blocks. Mirrors SPARK_M4_LM_BLOCKS.
pub const LM_BLOCKS: usize = HIDDEN / SCREEN_TILE;

/// How much of the vocabulary was computed exactly and how much was pruned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Screening {
    /// Entries whose exact dot product was computed.
    pub exact: u64,
    /// Entries that were certified and skipped.
    pub certified: u64,
}

/// The per-block absolute sums of the hidden state: one reduction per block, shared by every entry's
/// bound, which is what makes the screen cheap.
///
/// # Panics
/// Panics if the hidden state is not exactly HIDDEN wide.
#[must_use]
pub fn hidden_block_sums(hidden: &[f32]) -> Vec<f32> {
    assert_eq!(hidden.len(), HIDDEN, "the hidden state is d wide");
    (0..LM_BLOCKS)
        .map(|block| {
            let base = block * SCREEN_TILE;
            hidden[base..base + SCREEN_TILE]
                .iter()
                .map(|value| value.abs())
                .sum()
        })
        .collect()
}

/// The upper bound for one entry, from the block sums and the entry's weight row.
///
/// # Panics
/// Panics if the block sums or the weight row are the wrong shape.
#[must_use]
pub fn entry_bound(hidden_block_sums: &[f32], weight_row: &[f32]) -> f32 {
    assert_eq!(
        hidden_block_sums.len(),
        LM_BLOCKS,
        "one sum per screen block"
    );
    assert_eq!(weight_row.len(), HIDDEN, "the weight row is d wide");
    let mut bound = 0.0f32;
    for (block, block_sum) in hidden_block_sums.iter().enumerate() {
        let base = block * SCREEN_TILE;
        let row_max = weight_row[base..base + SCREEN_TILE]
            .iter()
            .fold(0.0f32, |acc, value| acc.max(value.abs()));
        bound += block_sum * row_max;
    }
    bound
}

/// The per-block absolute maxima of a BF16 weight matrix, in the kernel's [vocab][40] layout.
///
/// This is the load-time precompute for the certified head: for each vocabulary entry and each
/// 128-value screen block, the largest absolute weight in that block, stored as the BF16 bit pattern
/// that already held it. The checkpoint's weights are BF16, so taking the maximum of a block is a
/// selection, not a rounding: replaying these values gives exactly the same row maxima as scanning
/// the row, and therefore exactly the same bounds, certified flags and pruned logits.
///
/// The input is the little-endian BF16 byte image (two bytes per weight, row-major [vocab][HIDDEN]);
/// the output is vocab * LM_BLOCKS little-endian BF16 values. Working on bytes keeps a 1.32 GB head
/// from needing a second 1.32 GB u16 staging copy.
///
/// # Panics
/// Panics if the image is not a whole number of BF16 rows.
#[must_use]
pub fn block_maxima_bf16(weights_le: &[u8]) -> Vec<u8> {
    assert_eq!(weights_le.len() % 2, 0, "BF16 weights are whole bytes");
    let values = weights_le.len() / 2;
    assert_eq!(
        values % HIDDEN,
        0,
        "the weight image is a whole number of HIDDEN-wide rows"
    );
    let vocab = values / HIDDEN;
    let mut out = vec![0u8; vocab * LM_BLOCKS * 2];
    for entry in 0..vocab {
        let row = &weights_le[entry * HIDDEN * 2..(entry + 1) * HIDDEN * 2];
        for block in 0..LM_BLOCKS {
            let mut best_bits = 0u16;
            let mut best = 0.0f32;
            for i in 0..SCREEN_TILE {
                let at = (block * SCREEN_TILE + i) * 2;
                let bits = u16::from_le_bytes([row[at], row[at + 1]]);
                let magnitude = bf16_bits_to_f32(bits).abs();
                if magnitude > best {
                    best = magnitude;
                    // Store the magnitude's BF16 bits: the sign is not part of a row maximum, and a
                    // negative bit pattern would turn the bound into a subtraction.
                    best_bits = bits & 0x7fff;
                }
            }
            let at = (entry * LM_BLOCKS + block) * 2;
            out[at..at + 2].copy_from_slice(&best_bits.to_le_bytes());
        }
    }
    out
}

/// The exact dot product of the hidden state with one entry's weight row.
///
/// # Panics
/// Panics if the operands are not HIDDEN wide.
#[must_use]
pub fn exact_logit(hidden: &[f32], weight_row: &[f32]) -> f32 {
    assert_eq!(hidden.len(), HIDDEN);
    assert_eq!(weight_row.len(), HIDDEN);
    let mut acc = 0.0f32;
    for index in 0..HIDDEN {
        acc += hidden[index] * weight_row[index];
    }
    acc
}

/// Screen the whole vocabulary: logits holds the exact value where certified == 0 and the bound where
/// certified == 1, with the entries visited in ascending order exactly as the kernel visits them.
///
/// # Panics
/// Panics if the slices are shorter than the vocabulary requires.
#[must_use]
pub fn screen_logits(
    hidden: &[f32],
    weights: &[f32],
    vocab: usize,
    certified: &mut [u8],
    logits: &mut [f32],
) -> Screening {
    assert_eq!(hidden.len(), HIDDEN);
    assert!(
        weights.len() >= vocab * HIDDEN,
        "the weight matrix is too short"
    );
    assert!(certified.len() >= vocab && logits.len() >= vocab);
    let sums = hidden_block_sums(hidden);
    let mut result = Screening::default();
    let mut running_max = f32::MIN;
    for entry in 0..vocab {
        let row = &weights[entry * HIDDEN..(entry + 1) * HIDDEN];
        let bound = entry_bound(&sums, row);
        if bound < running_max {
            // Certified: the exact dot cannot beat the running maximum. The stored value is the
            // bound, which is why the array is an upper envelope rather than a set of logits.
            logits[entry] = bound;
            certified[entry] = 1;
            result.certified += 1;
        } else {
            let acc = exact_logit(hidden, row);
            logits[entry] = acc;
            certified[entry] = 0;
            result.exact += 1;
            if acc > running_max {
                running_max = acc;
            }
        }
    }
    result
}

/// The best entry of a screen result: the largest value and the index of its first occurrence, which
/// is what the kernel's separate block reduction produces. Entries that were certified carry bounds,
/// and a bound can never be larger than the running maximum that pruned it, so a certified entry can
/// never win this reduction.
///
/// # Panics
/// Panics if the slice is empty.
#[must_use]
pub fn best_entry(logits: &[f32]) -> (f32, usize) {
    assert!(
        !logits.is_empty(),
        "there is no best entry of an empty vocabulary"
    );
    let mut best = f32::MIN;
    let mut best_index = 0usize;
    for (index, value) in logits.iter().enumerate() {
        if *value > best {
            best = *value;
            best_index = index;
        }
    }
    (best, best_index)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic pseudo-random weight matrix: the tests only need it to be reproducible and to
    /// have a spread of magnitudes, not to be statistical.
    fn weights(vocab: usize) -> Vec<f32> {
        (0..vocab * HIDDEN)
            .map(|index| {
                let mixed = (index as u64).wrapping_mul(6364136223846793005) >> 33;
                ((mixed % 2001) as f32 - 1000.0) / 250.0
            })
            .collect()
    }

    /// A hidden state with a single non-zero entry: block 0 then carries the whole bound, which makes
    /// the screening behaviour easy to read off a test failure.
    fn sparse_hidden() -> Vec<f32> {
        let mut hidden = vec![0.0f32; HIDDEN];
        hidden[7] = 1.0;
        hidden
    }

    #[test]
    fn certified_entries_hold_a_bound_and_uncertified_entries_are_bit_exact() {
        let vocab = 64usize;
        let hidden = sparse_hidden();
        let w = weights(vocab);
        let mut certified = vec![0u8; vocab];
        let mut logits = vec![0.0f32; vocab];
        let screening = screen_logits(&hidden, &w, vocab, &mut certified, &mut logits);
        assert_eq!(screening.exact + screening.certified, vocab as u64);

        let sums = hidden_block_sums(&hidden);
        let mut seen_certified = 0u64;
        for entry in 0..vocab {
            let row = &w[entry * HIDDEN..(entry + 1) * HIDDEN];
            let exact = exact_logit(&hidden, row);
            if certified[entry] == 1 {
                seen_certified += 1;
                assert_eq!(
                    logits[entry],
                    entry_bound(&sums, row),
                    "a certified entry must carry its bound"
                );
                assert!(
                    exact <= logits[entry] + 1e-6,
                    "entry {entry}: the bound {} is below the exact value {exact}",
                    logits[entry]
                );
            } else {
                assert_eq!(
                    logits[entry].to_bits(),
                    exact.to_bits(),
                    "an uncertified entry must be bit-for-bit the exact dot product"
                );
            }
        }
        assert_eq!(seen_certified, screening.certified);
    }

    #[test]
    fn block_maxima_reproduce_the_scan_the_kernel_used_to_do() {
        let vocab = 5usize;
        let mut weights = vec![0u8; vocab * HIDDEN * 2];
        for (index, pair) in weights.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            // A spread of magnitudes with enough variation that no block's maximum is accidental.
            let raw = ((index as u64).wrapping_mul(6364136223846793005) >> 35) as u16;
            let mut bits = 0x3f80u16.wrapping_add(raw & 0x0fff);
            if index % 3 == 0 {
                bits |= 0x8000; // negative, so a signed maximum would be wrong
            }
            *pair = bits.to_le_bytes();
        }
        let maxima = block_maxima_bf16(&weights);
        assert_eq!(maxima.len(), vocab * LM_BLOCKS * 2);
        for entry in 0..vocab {
            for block in 0..LM_BLOCKS {
                let mut expected = 0.0f32;
                for i in 0..SCREEN_TILE {
                    let at = (entry * HIDDEN + block * SCREEN_TILE + i) * 2;
                    let bits = u16::from_le_bytes([weights[at], weights[at + 1]]);
                    expected = expected.max(bf16_bits_to_f32(bits).abs());
                }
                let at = (entry * LM_BLOCKS + block) * 2;
                let bits = u16::from_le_bytes([maxima[at], maxima[at + 1]]);
                assert_eq!(
                    bf16_bits_to_f32(bits),
                    expected,
                    "entry {entry} block {block} must keep its exact BF16 maximum"
                );
            }
        }
    }

    #[test]
    fn the_certificate_skips_entries_and_the_running_maximum_only_grows() {
        let vocab = 256usize;
        let hidden = sparse_hidden();
        let w = weights(vocab);
        let mut certified = vec![0u8; vocab];
        let mut logits = vec![0.0f32; vocab];
        let screening = screen_logits(&hidden, &w, vocab, &mut certified, &mut logits);
        assert!(
            screening.certified > 0,
            "the fixture must prune something, or these tests prove nothing"
        );
        assert!(
            screening.exact > 0,
            "at least the first entry is always exact, since the running maximum starts at -inf"
        );
        assert_eq!(certified[0], 0, "the first entry cannot be certified");
    }

    #[test]
    fn the_maximum_survives_the_pruning_and_the_fixture_has_a_unique_maximum() {
        let vocab = 256usize;
        let hidden = sparse_hidden();
        let w = weights(vocab);
        let mut certified = vec![0u8; vocab];
        let mut logits = vec![0.0f32; vocab];
        let _ = screen_logits(&hidden, &w, vocab, &mut certified, &mut logits);

        // The truth, computed with no screening at all.
        let exact: Vec<f32> = (0..vocab)
            .map(|entry| exact_logit(&hidden, &w[entry * HIDDEN..(entry + 1) * HIDDEN]))
            .collect();
        let (true_max, true_argmax) = best_entry(&exact);
        let (screen_max, screen_argmax) = best_entry(&logits);
        assert_eq!(
            screen_max.to_bits(),
            true_max.to_bits(),
            "the pruning changed the maximum"
        );
        assert_eq!(screen_argmax, true_argmax, "the pruning changed the argmax");
        // Every pruned entry must be provably below the winner, which is the certificate itself.
        for entry in 0..vocab {
            if certified[entry] == 1 {
                assert!(
                    logits[entry] < screen_max,
                    "a certified entry reached the maximum: the certificate is unsound"
                );
            }
        }
    }

    #[test]
    fn a_bound_that_ignores_the_absolute_weight_is_unsound() {
        // Construct a case where the bound must use |w|: the hidden entry is negative and the weight
        // is negative, so the contribution is large and POSITIVE, but a bound built from the signed
        // maximum of the weight sees a large negative number and concludes the entry is irrelevant.
        let vocab = 2usize;
        let mut hidden = vec![0.0f32; HIDDEN];
        hidden[0] = -1.0;
        let mut w = vec![0.0f32; vocab * HIDDEN];
        // Entry 0 sets the running maximum to (-1) * (-1) = 1.
        w[0] = -1.0;
        // Entry 1 would contribute +10, but every weight in the block is negative.
        for index in 0..SCREEN_TILE {
            w[HIDDEN + index] = -10.0;
        }

        let sums = hidden_block_sums(&hidden);
        let signed_bound = |row: &[f32]| {
            let mut bound = 0.0f32;
            for (block, block_sum) in sums.iter().enumerate() {
                let base = block * SCREEN_TILE;
                let row_max = row[base..base + SCREEN_TILE]
                    .iter()
                    .fold(f32::MIN, |acc, value| acc.max(*value));
                bound += block_sum * row_max;
            }
            bound
        };

        // The correct bound must certify nothing here, because entry 1's exact value is the maximum.
        let correct = entry_bound(&sums, &w[HIDDEN..2 * HIDDEN]);
        let exact_one = exact_logit(&hidden, &w[HIDDEN..2 * HIDDEN]);
        assert!(
            correct >= exact_one,
            "the real bound {correct} must cover the exact value {exact_one}"
        );

        // The mutant's bound is negative, so once entry 0 has set the running maximum to 1.0 the
        // mutant skips entry 1 - whose exact value is 10 - and returns the wrong answer.
        let mutant_bound = signed_bound(&w[HIDDEN..2 * HIDDEN]);
        assert!(
            mutant_bound < 0.0,
            "the mutant bound must be the signed trap"
        );
        let running_max = exact_logit(&hidden, &w[..HIDDEN]);
        assert!(
            mutant_bound < running_max,
            "the mutant certifies entry 1 (bound {mutant_bound} < running {running_max})"
        );
        assert!(
            exact_one > running_max,
            "and the certified entry really was the answer: {exact_one} > {running_max}"
        );
    }

    /// The certified head's own reduction, transcribed from kernels/cert_lm_head_sm121.cu: entries
    /// are dealt round-robin to a fixed number of lanes, each lane keeps the largest value it saw
    /// with ties going to the lower entry, and the lanes are then combined in ascending order with
    /// the same tie rule. The point of the transcription is that the reference's "first occurrence"
    /// rule and the kernel's "lower entry index" rule have to be the same rule, and the only way to
    /// show that without a GPU is to run the kernel's order here.
    fn kernel_style_argmax(logits: &[f32], threads: usize) -> usize {
        assert!(threads > 0);
        let mut thread_max = vec![-f32::MAX; threads];
        let mut thread_argmax = vec![0usize; threads];
        for (lane, (max, argmax)) in thread_max
            .iter_mut()
            .zip(thread_argmax.iter_mut())
            .enumerate()
        {
            let mut running_max = -f32::MAX;
            let mut running_argmax = 0usize;
            let mut entry = lane;
            while entry < logits.len() {
                let acc = logits[entry];
                if acc > running_max || (acc == running_max && entry < running_argmax) {
                    running_max = acc;
                    running_argmax = entry;
                }
                entry += threads;
            }
            *max = running_max;
            *argmax = running_argmax;
        }
        let mut best = thread_max[0];
        let mut best_index = thread_argmax[0];
        for i in 1..threads {
            if thread_max[i] > best || (thread_max[i] == best && thread_argmax[i] < best_index) {
                best = thread_max[i];
                best_index = thread_argmax[i];
            }
        }
        best_index
    }

    #[test]
    fn a_tie_is_broken_by_the_lower_entry_index_in_both_reductions() {
        let vocab = 1024usize;
        let mut logits = vec![-5.0f32; vocab];
        // Entries 100 and 300 tie for the maximum. They are dealt to different lanes - 100 to lane
        // 100 and 300 to lane 44 - and lane 44 is combined first, which is exactly the case the
        // kernel's comment describes.
        logits[100] = 3.5;
        logits[300] = 3.5;
        // ...and entries 5 and 261 tie inside one lane, which exercises the per-thread rule.
        logits[5] = 2.0;
        logits[261] = 2.0;
        let lowest = logits
            .iter()
            .enumerate()
            .filter(|(_, value)| **value == 3.5)
            .map(|(index, _)| index)
            .min()
            .expect("the fixture has a maximum");
        assert_eq!(lowest, 100);
        assert_eq!(best_entry(&logits), (3.5, 100));
        assert_eq!(
            kernel_style_argmax(&logits, 128),
            100,
            "the lane that holds the later entry must not win the tie"
        );
        assert_eq!(kernel_style_argmax(&logits, 64), 100);
        // The tie-break is load-bearing: a value-only reduction returns the earlier LANE, which
        // holds entry 300, so the two rules genuinely differ on this fixture.
        let value_only = |logits: &[f32], threads: usize| -> usize {
            let mut thread_max = vec![-f32::MAX; threads];
            let mut thread_argmax = vec![0usize; threads];
            for (lane, (max, argmax)) in thread_max
                .iter_mut()
                .zip(thread_argmax.iter_mut())
                .enumerate()
            {
                let mut entry = lane;
                while entry < logits.len() {
                    if logits[entry] > *max {
                        *max = logits[entry];
                        *argmax = entry;
                    }
                    entry += threads;
                }
            }
            let mut best = thread_max[0];
            let mut best_index = thread_argmax[0];
            for i in 1..threads {
                if thread_max[i] > best {
                    best = thread_max[i];
                    best_index = thread_argmax[i];
                }
            }
            best_index
        };
        assert_eq!(
            value_only(&logits, 128),
            300,
            "the mutant must disagree, or the tie rule proves nothing"
        );

        // A tie inside a single lane, where the global maximum is the tied value.
        let mut same_lane = vec![0.0f32; vocab];
        same_lane[5] = 4.0;
        same_lane[261] = 4.0;
        assert_eq!(best_entry(&same_lane).1, 5);
        assert_eq!(kernel_style_argmax(&same_lane, 128), 5);
    }

    #[test]
    fn the_parallel_reduction_agrees_with_the_sequential_scan_on_every_tie_pattern() {
        // Values are drawn from five distinct magnitudes, so ties are common rather than a special
        // case, and the vocabulary sizes include the boundaries of the lane stride.
        for (vocab, threads) in [
            (1usize, 128usize),
            (127, 128),
            (128, 128),
            (129, 128),
            (256, 128),
            (1000, 96),
            (77, 32),
            (1024, 128),
        ] {
            let mut values = vec![0.0f32; vocab];
            let mut state = 12_345u64;
            for value in &mut values {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                *value = ((state >> 60) % 5) as f32;
            }
            let (expected_max, expected_index) = values.iter().enumerate().fold(
                (f32::MIN, 0usize),
                |(best, best_index), (index, value)| {
                    if *value > best {
                        (*value, index)
                    } else {
                        (best, best_index)
                    }
                },
            );
            assert_eq!(best_entry(&values), (expected_max, expected_index));
            assert_eq!(
                kernel_style_argmax(&values, threads),
                expected_index,
                "vocabulary {vocab} with {threads} lanes"
            );
            // Non-vacuity: every vocabulary here has at least one entry holding the maximum.
            assert!(
                values
                    .iter()
                    .filter(|value| **value == expected_max)
                    .count()
                    >= 1
            );
        }
    }

    #[test]
    fn the_pruning_fixture_has_a_unique_maximum() {
        // The test above asserts that the argmax survives the pruning. That assertion is only a
        // statement about the maximum when the maximum is unique, so the uniqueness is checked here
        // rather than assumed, and the tie rules have their own tests.
        let vocab = 256usize;
        let hidden = sparse_hidden();
        let w = weights(vocab);
        let exact: Vec<f32> = (0..vocab)
            .map(|entry| exact_logit(&hidden, &w[entry * HIDDEN..(entry + 1) * HIDDEN]))
            .collect();
        let (maximum, argmax) = best_entry(&exact);
        let winners = exact.iter().filter(|value| **value == maximum).count();
        assert_eq!(
            winners, 1,
            "the pruning fixture must have a unique maximum for the argmax claim to mean anything"
        );
        assert_eq!(
            argmax,
            exact
                .iter()
                .position(|value| *value == maximum)
                .expect("the maximum occurs at least once")
        );
    }
}

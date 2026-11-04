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

//! The n-gram PLE hash: the Engram trick, applied to Qwen4-Exp's per-layer embedding.
//!
//! DeepSeek-V4.1's Engram and Qwen3.8-Flash-Next's PLE are the same idea with different
//! constants: a token n-gram is hashed into one bucket per head, each head owns a prime-sized
//! bucket range, and only the 16 rows a token actually indexes are ever read. V4.1 reads 24
//! columns out of a 189 GiB table across two layers; Qwen reads 16 columns out of 51.2 GiB at
//! layer 2. Both are paged, not resident.
//!
//! This module reuses the Engram arithmetic exactly:
//!
//! * splitmix64 with the same golden-ratio increment and the same two mixing constants
//!   (sparkflash_engram::primes documents the numpy draw it replaces).
//! * The same odd-multiplier rule: 2 * (splitmix64(seed + GAMMA * (i + 1)) % half_bound) + 1
//!   with half_bound = (i64::MAX / vocab) / 2, which is what keeps token * multiplier inside an
//!   i64 and rem_euclid well-defined.
//! * The same per-order prime draw: every head's modulus is the next prime after the previous
//!   head's, starting just above ngram_vocab_size_base, and the head's bucket range starts at the
//!   running sum of the preceding moduli.
//! * The same rolling multiplicative-XOR combination of the lookback window.
//!
//! Where they differ is the segment rule. Engram marks image spans DEAD and gives every blocked
//! shift the pad id. Qwen's reference (Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos)
//! restarts the window at every EOS: a shift is valid only inside the current EOS-delimited
//! segment, and an invalid shift contributes the EOS id. Padding positions are rewritten to EOS
//! before hashing, so padding is also a segment boundary. Both rules exist so that a token never
//! hashes across a sequence boundary; this module implements Qwen's, and the tests pin the rows
//! against a Python transcription of the reference.

use std::collections::VecDeque;

use crate::config::Qwen4ExpConfig;

use sparkflash_weights::rng::SPLITMIX_GAMMA;
pub use sparkflash_weights::rng::{is_prime, splitmix64};

/// The factor in the per-PLE-layer seed: 10007 * ple_layer_index.
pub const MULTIPLIER_SEED_FACTOR: u64 = 10_007;

/// The count-th prime strictly after start.
///
/// # Panics
/// Panics when count is zero; the reference always asks for at least one draw.
#[must_use]
pub fn nth_prime_after(start: u64, count: usize) -> u64 {
    assert!(count > 0, "nth_prime_after needs at least one draw");
    let mut prime = start;
    for _ in 0..count {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
    }
    prime
}

/// The odd hash multipliers of one PLE layer, one per lookback shift.
///
/// This is _build_layer_multipliers from the reference: the seed is
/// seed + 10007 * ple_layer_index, the draw is splitmix64, and the bound is the largest odd
/// multiplier that cannot overflow i64 when multiplied by the largest unigram id.
///
/// # Panics
/// Panics when unigram_vocab_size or ngram_size is zero, which the configuration validation
/// rules out.
#[must_use]
pub fn layer_multipliers(
    unigram_vocab_size: usize,
    ngram_size: usize,
    ple_layer_index: usize,
    seed: u64,
) -> Vec<u64> {
    assert!(unigram_vocab_size > 0, "the vocabulary cannot be empty");
    assert!(ngram_size > 0, "ngram_size must be positive");
    let max_long = (1u64 << 63) - 1;
    let multiplier_max = max_long / unigram_vocab_size as u64;
    let half_bound = (multiplier_max / 2).max(1);
    let base_seed = seed.wrapping_add(MULTIPLIER_SEED_FACTOR * ple_layer_index as u64);
    (0..ngram_size)
        .map(|index| {
            let value = base_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(index as u64 + 1));
            2 * (splitmix64(value) % half_bound) + 1
        })
        .collect()
}

/// One hashed head's bucket range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramHead {
    /// The bucket modulus: a prime above ngram_vocab_size_base.
    pub vocab_size: u64,
    /// The row offset of this head's bucket range in the concatenated table.
    pub offset: u64,
}

/// The per-head bucket layout of one PLE layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleLayout {
    /// Heads in reference order: all bigram heads, then all trigram heads.
    pub heads: Vec<NgramHead>,
    /// The sum of every head's modulus, before padding.
    pub rows: u64,
    /// rows rounded up so the table divides into split_ngram_parts equal shards.
    pub padded_rows: u64,
}

impl PleLayout {
    /// Derive the layout from the configuration.
    ///
    /// split_ngram_parts comes from the checkpoint's sharding (128). Rounding to the shard
    /// count, rather than to the configured divisor alone, is what makes every shard the same
    /// [2500012, 160]; the released table satisfies both rules at once.
    #[must_use]
    pub fn new(config: &Qwen4ExpConfig) -> Self {
        let mut heads = Vec::with_capacity(config.ngram_heads());
        let mut offset = 0u64;
        for head in 0..config.ngram_heads() {
            let vocab_size = nth_prime_after(config.ngram_vocab_size_base - 1, head + 1);
            heads.push(NgramHead { vocab_size, offset });
            offset += vocab_size;
        }
        let divisor = config
            .make_ngram_vocab_size_divisible_by
            .max(config.split_ngram_parts as u64);
        let padded_rows = offset.div_ceil(divisor) * divisor;
        Self {
            heads,
            rows: offset,
            padded_rows,
        }
    }

    /// The global table row that head's bucket lives at.
    #[must_use]
    pub fn row(&self, head: usize, bucket: u64) -> u64 {
        self.heads[head].offset + bucket
    }
}

/// The rolling n-gram hasher for one PLE layer.
///
/// The hasher keeps the last ngram_size - 1 ids and the position of the most recent EOS, so a
/// prefill and a decode of the same sequence produce identical rows (the same property
/// sparkflash_engram::hash::NgramHasher has, for the same reason).
#[derive(Debug, Clone)]
pub struct PleHasher {
    ngram_size: usize,
    eos: u32,
    multipliers: Vec<u64>,
    layout: PleLayout,
    /// (absolute position, id) for the last ngram_size - 1 positions, oldest first.
    window: VecDeque<(u64, u32)>,
    /// The absolute position the next pushed token will have.
    next_position: u64,
    /// The absolute position of the most recent EOS at or before the current position.
    last_eos: Option<u64>,
}

impl PleHasher {
    /// Build a hasher for ple_layer_index (0 for the first PLE layer, which is the released
    /// layout) and the configuration's hash constants.
    ///
    /// # Errors
    /// When the configuration has no EOS id, which PLE requires.
    pub fn new(config: &Qwen4ExpConfig, ple_layer_index: usize) -> Result<Self, String> {
        let eos = *config
            .eos_token_ids
            .first()
            .ok_or_else(|| "the PLE requires an eos token".to_owned())?;
        Ok(Self {
            ngram_size: config.ngram_size,
            eos,
            multipliers: layer_multipliers(
                config.vocab_size,
                config.ngram_size,
                ple_layer_index,
                config.seed,
            ),
            layout: PleLayout::new(config),
            window: VecDeque::with_capacity(config.ngram_size + 1),
            next_position: 0,
            last_eos: None,
        })
    }

    /// The bucket layout this hasher indexes.
    #[must_use]
    pub fn layout(&self) -> &PleLayout {
        &self.layout
    }

    /// The next absolute position the hasher will label.
    #[must_use]
    pub fn position(&self) -> u64 {
        self.next_position
    }

    /// Rewind to the start of a new sequence.
    pub fn reset(&mut self) {
        self.window.clear();
        self.next_position = 0;
        self.last_eos = None;
    }

    /// Hash ids, returning one row vector per position.
    ///
    /// mask, when given, is one bool per id; a false position is a padding position and is
    /// rewritten to EOS before hashing, exactly as the reference's
    /// ple_input_ids = where(conv_mask, ids, eos) does.
    ///
    /// # Errors
    /// When the mask length does not match.
    pub fn push(&mut self, ids: &[u32], mask: Option<&[bool]>) -> Result<Vec<Vec<u64>>, String> {
        if let Some(mask) = mask
            && mask.len() != ids.len()
        {
            return Err(format!(
                "the PLE token mask has {} entries for {} ids",
                mask.len(),
                ids.len()
            ));
        }
        let mut out = Vec::with_capacity(ids.len());
        for (offset, &raw) in ids.iter().enumerate() {
            let id = match mask {
                Some(mask) if !mask[offset] => self.eos,
                _ => raw,
            };
            let position = self.next_position;
            self.window.push_back((position, id));
            // The lookback spans the current token plus ngram_size - 1 older ones, so the window
            // holds ngram_size entries, not ngram_size - 1.
            while self.window.len() > self.ngram_size {
                self.window.pop_front();
            }
            // The reference's segment rule looks at the last EOS *strictly before* this position,
            // so an EOS token still hashes with the segment it terminates and only starts a new
            // one for the tokens after it.
            out.push(self.hash_at(position));
            if id == self.eos {
                self.last_eos = Some(position);
            }
            self.next_position += 1;
        }
        Ok(out)
    }

    fn hash_at(&self, position: u64) -> Vec<u64> {
        let heads = self.layout.heads.len();
        let per_order = heads / (self.ngram_size - 1);
        let mut columns = vec![0u64; heads];
        let segment_start = self.last_eos.map_or(0, |eos| eos + 1);
        let position_in_segment = position.saturating_sub(segment_start);
        // The lookback window, newest first. A shift is valid only inside the current segment;
        // anything else contributes EOS, which is the reference's rule. ngram_size is 3 in the
        // released configuration; the fixed window keeps the hash allocation-free.
        let mut window = [self.eos; 8];
        let depth = self.ngram_size.clamp(1, window.len());
        for (shift, slot) in window.iter_mut().enumerate().take(depth) {
            let valid = position_in_segment >= shift as u64 && position >= shift as u64;
            let source = position.saturating_sub(shift as u64);
            let id = self
                .window
                .iter()
                .rev()
                .find(|(at, _)| *at == source)
                .map(|(_, id)| *id);
            *slot = if valid {
                id.unwrap_or(self.eos)
            } else {
                self.eos
            };
        }
        for order in 2..=depth {
            let base = (order - 2) * per_order;
            let mut mixed = (window[0] as u64).wrapping_mul(self.multipliers[0]);
            for (shift, value) in window.iter().enumerate().take(order).skip(1) {
                mixed ^= (*value as u64).wrapping_mul(self.multipliers[shift]);
            }
            for (head, slot) in columns.iter_mut().enumerate().skip(base).take(per_order) {
                let spec = self.layout.heads[head];
                let bucket = (mixed % spec.vocab_size + spec.vocab_size) % spec.vocab_size;
                *slot = spec.offset + bucket;
            }
        }
        columns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_released_layout_is_16_ascending_primes_above_twenty_million() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
        let layout = PleLayout::new(&config);
        assert_eq!(layout.heads.len(), 16);
        assert_eq!(layout.heads[0].vocab_size, 20_000_003);
        assert_eq!(layout.heads[0].offset, 0);
        assert_eq!(layout.heads[15].offset, 300_001_275);
        assert_eq!(layout.rows, 320_001_446);
        assert_eq!(layout.padded_rows, 320_001_536);
        assert_eq!(
            layout.padded_rows / config.split_ngram_parts as u64,
            2_500_012
        );
        let mut ascending = true;
        for pair in layout.heads.windows(2) {
            ascending &= pair[1].vocab_size > pair[0].vocab_size;
            assert_eq!(pair[1].offset, pair[0].offset + pair[0].vocab_size);
        }
        assert!(ascending, "the reference draws ascending primes");
        for head in &layout.heads {
            assert!(
                is_prime(head.vocab_size),
                "{} is not prime",
                head.vocab_size
            );
        }
    }

    #[test]
    fn the_released_multipliers_match_the_reference_pins() {
        // Recomputed from _build_layer_multipliers with seed 1234 and vocabulary 248320.
        assert_eq!(
            layer_multipliers(248_320, 3, 0, 1234),
            vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
        );
        assert_eq!(
            layer_multipliers(248_320, 3, 1, 1234),
            vec![3_352_040_966_061, 19_752_442_147_475, 10_742_103_100_073]
        );
        for multiplier in layer_multipliers(248_320, 3, 0, 1234) {
            assert_eq!(multiplier % 2, 1, "every multiplier is odd");
        }
    }

    #[test]
    fn the_reference_rows_are_reproduced() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
        let mut hasher = PleHasher::new(&config, 0).expect("the PLE layer has an eos");
        let rows = hasher
            .push(&[10, 20, 30, 248_044, 50], None)
            .expect("the ids hash");
        let expected: [[u64; 16]; 5] = [
            [
                6826666, 27775725, 51991156, 74082527, 82622748, 119600976, 135816374, 152166807,
                174244281, 190221032, 211723794, 232787707, 243645790, 275729718, 280030017,
                303574322,
            ],
            [
                4810669, 34962340, 40038404, 63145186, 97237011, 115267695, 122313706, 158375242,
                171840321, 191688874, 219996824, 228806855, 252506948, 276566639, 284666446,
                305764640,
            ],
            [
                9878115, 26555603, 54895210, 62571545, 80580723, 119917398, 128922427, 147596134,
                168936175, 195223391, 219226064, 233524685, 246670267, 279816194, 297531600,
                306108296,
            ],
            [
                13714628, 30087929, 52684691, 77259450, 97195726, 118115344, 130376726, 155038357,
                177884092, 193470248, 210338423, 236000444, 254907184, 274872366, 295096076,
                317978052,
            ],
            [
                6968327, 28671820, 53265338, 63886977, 96882706, 102012836, 130456444, 143111416,
                171370341, 182647785, 202002617, 234079976, 252591074, 272328110, 299179713,
                315479992,
            ],
        ];
        assert_eq!(rows, expected.to_vec());
    }

    #[test]
    fn a_prefill_and_a_decode_produce_the_same_rows() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
        let mut once = PleHasher::new(&config, 0).expect("the PLE layer has an eos");
        let whole = once.push(&[5, 6, 7, 8, 9], None).expect("ids hash");
        let mut split = PleHasher::new(&config, 0).expect("the PLE layer has an eos");
        let mut pieces = split.push(&[5, 6], None).expect("ids hash");
        pieces.extend(split.push(&[7], None).expect("ids hash"));
        pieces.extend(split.push(&[8, 9], None).expect("ids hash"));
        assert_eq!(whole, pieces);
    }

    #[test]
    fn a_padding_position_behaves_as_an_eos() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
        let mut padded = PleHasher::new(&config, 0).expect("the PLE layer has an eos");
        let rows = padded
            .push(&[5, 6, 7], Some(&[true, false, true]))
            .expect("ids hash");
        let mut explicit = PleHasher::new(&config, 0).expect("the PLE layer has an eos");
        let expected = explicit.push(&[5, 248_044, 7], None).expect("ids hash");
        assert_eq!(rows, expected);
    }

    #[test]
    fn the_tiny_layout_divides_into_its_shards() {
        let config = Qwen4ExpConfig::tiny();
        let layout = PleLayout::new(&config);
        assert_eq!(layout.heads.len(), 4);
        assert_eq!(layout.padded_rows % config.split_ngram_parts as u64, 0);
        assert!(layout.rows <= layout.padded_rows);
        assert_eq!(
            layout.padded_rows / config.split_ngram_parts as u64,
            config.ple_table_rows() / config.split_ngram_parts as u64
        );
    }
}

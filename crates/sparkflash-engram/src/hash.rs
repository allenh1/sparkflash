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

//! 24-column multiplicative-XOR n-gram hash. OWNER: engram-hash.
//!
//! Exact port of `NgramHashState.forward` from `reference/inference/engram.py`. For every
//! position the layer looks back over the compressed token ids and builds the 2-, 3- and
//! 4-grams that end there. Each n-gram order has 8 heads, each head owns its own prime-sized
//! bucket range in the layer's 189 GiB table, and the 24 columns are ordered
//! `24 * order_index + head` with order index 0 = 2-gram, 1 = 3-gram, 2 = 4-gram.
//!
//! # Algorithm
//!
//! 1. The compressed id of every position is written into a persistent, position-indexed
//!    cache that carries across the prefill/decode split. A position outside the sequence
//!    (an image span) is stored as `DEAD = -1`.
//! 2. For shift `0..=3` the source is `cache[max(position - shift, 0)]` (clamped gather, so
//!    the sequence start is not a special case). Blockage accumulates:
//!    `blocked |= (position < shift) | (source == DEAD)`, and once blocked every longer shift
//!    stays blocked - an n-gram never spans an image.
//! 3. A blocked shift contributes the pad id instead of the cached id.
//! 4. `products[shift] = token[shift] * MULTIPLIERS[layer][shift]` in `i64`; the bound in
//!    `crate::primes::multiplier_bound` guarantees no overflow. The rolling value starts at
//!    `products[0]` and for `i in 1..4` does `rolling ^= products[i]`, which after step `i`
//!    is the hash of the `(i + 1)`-gram.
//! 5. Column `8 * (i - 1) + head` of layer `l` is
//!    `OFFSETS[l][...] + rolling.rem_euclid(PRIMES[l][i - 1][head])`. `rem_euclid` is the
//!    floor-modulo the reference's `%` performs on int64 tensors.
//!
//! The cache is shared by both Engram layers: the id sequence is the same, only the primes
//! and multipliers differ, so one [`NgramHasher`] emits both layers' columns per position.

use crate::compress::{EngramError, TokenizerCompressor};
use crate::layout::{COMPRESSED_VOCAB_SIZE, ENGRAM_NUM_HEADS, ENGRAM_TOTAL_LOOKUPS_PER_TOK};
use crate::primes::{MULTIPLIERS, OFFSETS, PRIMES};

/// Number of Engram layers (backbone layers 1 and 14).
pub const ENGRAM_LAYERS: usize = 2;

/// Number of n-gram orders (2-, 3- and 4-gram).
pub const NGRAM_ORDERS: usize = 3;

/// Lookbacks per position: the token itself plus the three older ones.
pub const MAX_NGRAM_SIZE: usize = 4;

/// Hash columns per position per layer: `NGRAM_ORDERS * ENGRAM_NUM_HEADS`.
pub const HASH_COLUMNS: usize = ENGRAM_TOTAL_LOOKUPS_PER_TOK;

/// Cache marker for a position that must not take part in any n-gram (an image span).
pub const DEAD: i64 = -1;

/// Hashes one already-gathered lookback window into the 24 columns of one layer.
///
/// `window[shift]` is the token at `position - shift` (clamped at the sequence start) and is
/// already substituted with the pad id wherever that shift is blocked. Column
/// `8 * order_index + head` is returned.
///
/// # Panics
/// Panics when `layer_index` is not 0 or 1; callers iterate `0..ENGRAM_LAYERS`.
#[must_use]
pub fn hash_window(layer_index: usize, window: &[i64; MAX_NGRAM_SIZE]) -> [u32; HASH_COLUMNS] {
    let mut columns = [0u32; HASH_COLUMNS];
    let multipliers = &MULTIPLIERS[layer_index];
    let mut rolling = window[0].wrapping_mul(multipliers[0]);
    for i in 1..MAX_NGRAM_SIZE {
        rolling ^= window[i].wrapping_mul(multipliers[i]);
        let order = i - 1;
        let row_start = order * ENGRAM_NUM_HEADS;
        for (head, &prime) in PRIMES[layer_index][order].iter().enumerate() {
            let column = row_start + head;
            let bucket = rolling.rem_euclid(prime as i64) as u64;
            columns[column] = (bucket + OFFSETS[layer_index][column]) as u32;
        }
    }
    columns
}

/// The columns of positions `start..history.len()` of a raw-id history, appended to `out` as
/// `[position][layer][column]`: the stateless form of [`NgramHasher::push`] for a caller that
/// keeps the whole history (the serving loop, whose rows can be rewound and redrafted).
///
/// Each raw id goes through `compressor` first, exactly as `push` does; the reference hashes
/// compressed ids, never raw ones. A raw id equal to `dead` marks a position outside every
/// n-gram (an image span), like a `false` in `push`'s token mask: it blocks the longer n-grams
/// of the positions after it, and its own row is filled with `dead` so the caller can skip it.
///
/// # Panics
/// Panics when `start` is past the end of `history`.
pub fn hash_history(
    compressor: &TokenizerCompressor,
    history: &[u32],
    start: usize,
    dead: u32,
    out: &mut Vec<u32>,
) {
    assert!(start <= history.len(), "start is past the history");
    let pad = i64::from(compressor.pad_id());
    out.reserve((history.len() - start) * ENGRAM_LAYERS * HASH_COLUMNS);
    for position in start..history.len() {
        if history[position] == dead {
            out.resize(out.len() + ENGRAM_LAYERS * HASH_COLUMNS, dead);
            continue;
        }
        let mut blocked = false;
        let mut window = [pad; MAX_NGRAM_SIZE];
        for (shift, slot) in window.iter_mut().enumerate() {
            // Blocked is sticky: before the sequence start or through a dead id, this shift and
            // every longer one keep the pad.
            match position.checked_sub(shift).map(|at| history[at]) {
                Some(id) if !blocked && id != dead => *slot = i64::from(compressor.compress(id)),
                _ => blocked = true,
            }
        }
        for layer in 0..ENGRAM_LAYERS {
            out.extend_from_slice(&hash_window(layer, &window));
        }
    }
}

/// Streaming 24-column n-gram hasher over one persistent compressed-id cache.
///
/// The cache is indexed by absolute position and survives across calls, which is exactly how
/// the reference handles the prefill/decode split: pushing `8 + 1` ids produces byte-identical
/// columns to pushing the 9 ids at once. Call [`reset`](Self::reset) before starting a new
/// sequence.
#[derive(Debug, Clone)]
pub struct NgramHasher {
    compressor: TokenizerCompressor,
    cache: Vec<i64>,
    start_pos: usize,
}

impl NgramHasher {
    /// Creates a hasher with room for `max_seq_len` positions.
    ///
    /// # Panics
    /// Panics when `max_seq_len` is zero.
    #[must_use]
    pub fn new(compressor: TokenizerCompressor, max_seq_len: usize) -> Self {
        assert!(max_seq_len > 0, "max_seq_len must be non-zero");
        Self {
            compressor,
            cache: vec![0; max_seq_len],
            start_pos: 0,
        }
    }

    /// Compresses and pushes raw tokenizer ids, returning the columns of the pushed
    /// positions: `result[position_offset][layer][column]`, with column
    /// `8 * order_index + head`.
    ///
    /// `token_mask`, when given, is one bool per id; `false` marks a position that must not
    /// take part in any n-gram (an image span). It is stored as `DEAD` in the cache, which
    /// blocks that position and every later position's longer n-grams, exactly like the
    /// reference.
    ///
    /// Out-of-vocabulary raw ids never fail here: `TokenizerCompressor::compress` maps them
    /// to the pad id, so every id this method forwards is already below
    /// `COMPRESSED_VOCAB_SIZE` and the range check in
    /// [`push_compressed`](Self::push_compressed) cannot fire.
    ///
    /// # Errors
    /// Fails when the mask length does not match the id count, or when the push would run past
    /// `max_seq_len`.
    pub fn push(
        &mut self,
        raw_token_ids: &[u32],
        token_mask: Option<&[bool]>,
    ) -> Result<Vec<[[u32; HASH_COLUMNS]; ENGRAM_LAYERS]>, EngramError> {
        let compressed: Vec<u32> = raw_token_ids
            .iter()
            .map(|&id| self.compressor.compress(id))
            .collect();
        debug_assert!(
            compressed.iter().all(|&id| id < COMPRESSED_VOCAB_SIZE),
            "compress() must never return an id outside the compressed vocabulary"
        );
        self.push_compressed(&compressed, token_mask)
    }

    /// Pushes already-compressed ids. Otherwise identical to [`push`](Self::push); useful
    /// when the caller keeps its own compressed cache.
    ///
    /// Every id must be below `COMPRESSED_VOCAB_SIZE`. This is checked before anything is
    /// written, because the rolling hash multiplies ids by ~7.7e13 and a raw id that leaked in
    /// here would silently wrap `i64` instead of failing.
    ///
    /// # Errors
    /// Fails with [`EngramError::CompressedIdOutOfRange`] when an id is not below
    /// `COMPRESSED_VOCAB_SIZE`, with [`EngramError::TokenMaskLength`] when the mask length
    /// does not match, or with [`EngramError::CacheCapacity`] when the push would run past
    /// `max_seq_len`. A rejected batch leaves the hasher untouched: no id is cached and
    /// [`start_pos`](Self::start_pos) does not move.
    pub fn push_compressed(
        &mut self,
        compressed_ids: &[u32],
        token_mask: Option<&[bool]>,
    ) -> Result<Vec<[[u32; HASH_COLUMNS]; ENGRAM_LAYERS]>, EngramError> {
        let len = compressed_ids.len();
        if let Some(mask) = token_mask
            && mask.len() != len
        {
            return Err(EngramError::TokenMaskLength {
                given: mask.len(),
                expected: len,
            });
        }
        let start = self.start_pos;
        let end = start + len;
        if end > self.cache.len() {
            return Err(EngramError::CacheCapacity {
                start_pos: start,
                len,
                max_seq_len: self.cache.len(),
            });
        }
        if let Some((position, id)) = compressed_ids
            .iter()
            .copied()
            .enumerate()
            .find(|&(_, id)| id >= COMPRESSED_VOCAB_SIZE)
        {
            return Err(EngramError::CompressedIdOutOfRange {
                id,
                position,
                compressed_vocab_size: COMPRESSED_VOCAB_SIZE,
            });
        }

        for (offset, &id) in compressed_ids.iter().enumerate() {
            let dead = token_mask.is_some_and(|mask| !mask[offset]);
            self.cache[start + offset] = if dead { DEAD } else { id as i64 };
        }

        let pad = self.compressor.pad_id() as i64;
        let mut out = Vec::with_capacity(len);
        for offset in 0..len {
            let position = start + offset;
            let mut blocked = false;
            let mut window = [pad; MAX_NGRAM_SIZE];
            for (shift, slot) in window.iter_mut().enumerate() {
                let source = self.cache[position.saturating_sub(shift)];
                blocked |= position < shift || source == DEAD;
                *slot = if blocked { pad } else { source };
            }
            let mut columns = [[0u32; HASH_COLUMNS]; ENGRAM_LAYERS];
            for (layer, layer_columns) in columns.iter_mut().enumerate() {
                *layer_columns = hash_window(layer, &window);
            }
            out.push(columns);
        }

        self.start_pos = end;
        Ok(out)
    }

    /// Clears the cache and rewinds to position 0, for the next sequence.
    pub fn reset(&mut self) {
        self.cache.fill(0);
        self.start_pos = 0;
    }

    /// Absolute position the next push starts at.
    #[must_use]
    pub fn start_pos(&self) -> usize {
        self.start_pos
    }

    /// Capacity of the position cache.
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.cache.len()
    }

    /// The compressor this hasher reads ids through.
    #[must_use]
    pub fn compressor(&self) -> &TokenizerCompressor {
        &self.compressor
    }
}

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

//! Fixed Engram geometry, taken from the published checkpoint config and from the shapes the
//! checkpoint actually stores.
//!
//! The Engram tables are the dominant storage consumer of the whole model: two layers with
//! 384,006,168 and 384,016,682 rows of 256 fp8 weights plus 8 ue8m0 block scales (one scale per
//! 32 values) = 264 raw bytes per row. That is 188.83 GiB of conditional memory, 47.2 GiB per
//! rank at TP=4, which is why it lives on local NVMe instead of in the 121.7 GiB unified memory
//! ceiling. Rows are packed densely at 264 bytes; see docs/DECISIONS.md D4 for why the
//! specification's 512-byte padded stride is rejected.

/// Backbone layers that own an Engram table.
pub const ENGRAM_LAYERS: [usize; 2] = [1, 14];

/// N-gram orders handled by each layer's hash heads: 2-gram, 3-gram, 4-gram.
pub const ENGRAM_ORDERS: [usize; 3] = [2, 3, 4];

/// Hash heads per order.
pub const ENGRAM_NUM_HEADS: usize = 8;

/// Total lookups per token: 3 orders x 8 heads. Every rank performs all 24 lookups; the table
/// is sharded by ROW, not by head (the reference slices `engram.embed.*` with
/// `shard_size = (rows + mp - 1) / mp` and all-reduces the gathered embeddings).
pub const ENGRAM_TOTAL_LOOKUPS_PER_TOK: usize = 3 * ENGRAM_NUM_HEADS;

/// Embedding width per head.
pub const ENGRAM_HEAD_DIM: usize = 256;

/// ue8m0 block scales per row (256 / 32).
pub const ENGRAM_SCALE_BYTES: usize = 8;

/// Bytes per row exactly as the checkpoint stores it: 256 fp8 values + 8 ue8m0 scales.
pub const ENGRAM_ROW_BYTES: usize = ENGRAM_HEAD_DIM + ENGRAM_SCALE_BYTES;

/// Padding token id substituted for out-of-vocabulary ids.
pub const ENGRAM_PAD_TOKEN_ID: u32 = 2;

/// Image placeholder token id; image spans are masked out of Engram lookup.
pub const IMAGE_TOKEN_ID: u32 = 129_264;

/// Size of the compressed (surjective) tokenizer vocabulary; every hash multiplier is derived
/// from it, so the compressor asserts this value at boot.
pub const COMPRESSED_VOCAB_SIZE: u32 = 99_092;

/// Raw row count per layer, from engram_num_embeddings in config.json. Verified: the 24 bucket
/// primes per layer sum to exactly these numbers.
pub const ENGRAM_NUM_EMBEDDINGS: [u64; 2] = [384_006_168, 384_016_682];

/// Values per ue8m0 block scale, matching fp8_block_size in the reference.
pub const SCALE_BLOCK: usize = 32;

/// Row count of one layer's table.
#[must_use]
pub fn num_embeddings(layer_index: usize) -> u64 {
    ENGRAM_NUM_EMBEDDINGS[layer_index]
}

/// Total logical bytes of both Engram tables (dense rows, no padding).
#[must_use]
pub fn total_bytes() -> u64 {
    ENGRAM_NUM_EMBEDDINGS
        .iter()
        .map(|rows| rows * ENGRAM_ROW_BYTES as u64)
        .sum()
}

/// Number of rows rank `rank` owns for `layer_index`, matching the reference's
/// `shard_size = (rows + mp - 1) / mp` with the final shard zero-padded.
///
/// # Panics
/// Panics when `world_size` is zero.
#[must_use]
pub fn rows_per_rank(layer_index: usize, world_size: usize) -> u64 {
    assert!(world_size > 0, "world size must be non-zero");
    num_embeddings(layer_index).div_ceil(world_size as u64)
}

/// Row range rank `rank` owns, as the reference slices it. The range may extend past the real
/// table for the last rank; readers must treat those rows as zero.
///
/// # Panics
/// Panics when `world_size` is zero or `rank >= world_size`.
#[must_use]
pub fn row_range_for_rank(
    layer_index: usize,
    rank: usize,
    world_size: usize,
) -> std::ops::Range<u64> {
    assert!(world_size > 0, "world size must be non-zero");
    assert!(
        rank < world_size,
        "rank {rank} out of range for world size {world_size}"
    );
    let shard = rows_per_rank(layer_index, world_size);
    let start = rank as u64 * shard;
    start..start + shard
}

/// Byte offset of `row` inside a packed dense table.
///
/// # Panics
/// Panics when `row` is past the end of the table.
#[must_use]
pub fn row_offset(layer_index: usize, row: u64) -> u64 {
    assert!(
        row < num_embeddings(layer_index),
        "row {row} is past the end of engram layer index {layer_index}"
    );
    row * ENGRAM_ROW_BYTES as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_geometry_matches_the_checkpoint() {
        assert_eq!(ENGRAM_ROW_BYTES, 264);
        assert_eq!(ENGRAM_HEAD_DIM / SCALE_BLOCK, ENGRAM_SCALE_BYTES);
        assert_eq!(SCALE_BLOCK, 32);
    }

    #[test]
    fn engram_footprint_is_about_189_gib() {
        let raw_gib = total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
        assert!(
            (188.5..189.0).contains(&raw_gib),
            "dense Engram footprint {raw_gib} GiB"
        );
        // The rejected 512-byte stride would have cost 94% more.
        let padded: u64 = ENGRAM_NUM_EMBEDDINGS.iter().map(|r| r * 512).sum();
        assert!(padded as f64 / total_bytes() as f64 > 1.9);
    }

    #[test]
    fn rows_shard_across_four_ranks_like_the_reference() {
        // Layer 0 is 384006168 = 4 * 96001542, so it divides evenly and no rank is padded:
        // rank 3 ends on the last real row.
        assert_eq!(rows_per_rank(0, 4), 96_001_542);
        assert_eq!(row_range_for_rank(0, 0, 4), 0..96_001_542);
        assert_eq!(row_range_for_rank(0, 3, 4), 288_004_626..384_006_168);
        // Layer 1 is the layer that pads: 384016682 / 4 = 96004170.5, so every rank owns
        // 96004171 rows and rank 3's slice runs 2 rows past the table - the zero-padded final
        // shard convert.py produces.
        assert_eq!(rows_per_rank(1, 4), 96_004_171);
        assert_eq!(row_range_for_rank(1, 3, 4), 288_012_513..384_016_684);
        assert_eq!(
            rows_per_rank(1, 4) * 4 - num_embeddings(1),
            2,
            "layer 1's last rank is padded by 2 rows"
        );
    }

    #[test]
    fn row_offsets_are_dense() {
        assert_eq!(row_offset(0, 0), 0);
        assert_eq!(row_offset(0, 1), 264);
        assert_eq!(row_offset(1, 100), 26_400);
    }
}

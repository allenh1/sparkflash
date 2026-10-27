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

//! Byte-exact layout of the paged global KV cache, the SWA tail ring and the node memory
//! budget.
//!
//! Everything here is *derived* from the published checkpoint geometry in crate::config,
//! never transcribed from a prose table. The two numbers the rest of the engine cares about
//! are re-derived by tests from the config:
//!
//! * one **global KV entry** is 288 B of FP4 E2M1 main latent (512 dims, E4M3 scale per 16
//!   channels) plus 68 B of MXFP4 indexer K (128 dims, E8M0 scale per 32 channels) = 356 B;
//! * the four Full-mode layers store 3 x 0.5 + 1 x 1.0 = 2.5 entries per token, so a token
//!   costs 2.5 x 356 = **890 B**, and a 64-token page costs 160 x 356 = **56,960 B**.
//!
//! The specification's prose itemisation (spec line 76) adds qk_rope_head_dim = 64 on top of
//! head_dim = 512 and therefore sums to 1050 B/token. The 64 rotary dimensions are *inside*
//! head_dim in the reference (model.py:758 applies RoPE to the last rope_head_dim columns of
//! the 512-wide latent), so the 890 B figure only closes when the rotary dims are not counted
//! twice. See docs/architecture/08-kv-footprint.md section 2.

use crate::config::{
    ENCODER_LAST_LAYER, KV_PAGE_TOKENS, KV_SOURCE_LAYERS, NUM_HIDDEN_LAYERS, compress_ratio,
};

/// Identifier of a page in the shared global-KV pool.
///
/// A page id is stable for as long as the pool holds the page; the radix tree, the tiered
/// page store and the telemetry counters all key on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageId(pub u64);

/// Bytes per 32-bit token id.
pub const TOKEN_ID_BYTES: usize = std::mem::size_of::<u32>();

/// Number of FP4 values that share one byte.
pub const FP4_VALUES_PER_BYTE: usize = 2;

/// Width of the MLA latent that FlashMLA attends over, rotary dims included.
pub const MAIN_KV_DIM: usize = 512;

/// Elements per E4M3 scale in the compressed main KV.
pub const MAIN_KV_SCALE_GROUP: usize = 16;

/// Packed FP4 payload of one main KV entry.
pub const MAIN_KV_DATA_BYTES: usize = MAIN_KV_DIM / FP4_VALUES_PER_BYTE;

/// E4M3 scale bytes of one main KV entry.
pub const MAIN_KV_SCALE_BYTES: usize = MAIN_KV_DIM / MAIN_KV_SCALE_GROUP;

/// Total bytes of one main KV entry: 256 data + 32 scales.
pub const MAIN_KV_ENTRY_BYTES: usize = MAIN_KV_DATA_BYTES + MAIN_KV_SCALE_BYTES;

/// Width of the indexer key.
pub const INDEXER_K_DIM: usize = 128;

/// Elements per E8M0 scale in the indexer key.
pub const INDEXER_K_SCALE_GROUP: usize = 32;

/// Packed FP4 payload of one indexer key.
pub const INDEXER_K_DATA_BYTES: usize = INDEXER_K_DIM / FP4_VALUES_PER_BYTE;

/// E8M0 scale bytes of one indexer key.
pub const INDEXER_K_SCALE_BYTES: usize = INDEXER_K_DIM / INDEXER_K_SCALE_GROUP;

/// Total bytes of one indexer key: 64 data + 4 scales.
pub const INDEXER_K_ENTRY_BYTES: usize = INDEXER_K_DATA_BYTES + INDEXER_K_SCALE_BYTES;

/// Offset of the main KV payload inside a global entry.
pub const MAIN_KV_DATA_OFFSET: usize = 0;

/// Offset of the main KV E4M3 scales inside a global entry.
pub const MAIN_KV_SCALE_OFFSET: usize = MAIN_KV_DATA_OFFSET + MAIN_KV_DATA_BYTES;

/// Offset of the indexer K payload inside a global entry.
pub const INDEXER_K_DATA_OFFSET: usize = MAIN_KV_SCALE_OFFSET + MAIN_KV_SCALE_BYTES;

/// Offset of the indexer K E8M0 scale inside a global entry.
pub const INDEXER_K_SCALE_OFFSET: usize = INDEXER_K_DATA_OFFSET + INDEXER_K_DATA_BYTES;

/// Total bytes of one compressed global KV entry across both structures.
pub const GLOBAL_ENTRY_BYTES: usize = MAIN_KV_ENTRY_BYTES + INDEXER_K_ENTRY_BYTES;

/// Compressed entries per token, as an exact integer ratio: 5 entries per 2 tokens.
pub const GLOBAL_ENTRIES_PER_TOKEN_NUMERATOR: u64 = 5;

/// Denominator of GLOBAL_ENTRIES_PER_TOKEN_NUMERATOR.
pub const GLOBAL_ENTRIES_PER_TOKEN_DENOMINATOR: u64 = 2;

/// Compressed entries stored in one 64-token global page, across all four Full layers.
pub const GLOBAL_PAGE_ENTRIES: usize = 160;

/// Bytes of one 64-token global page.
pub const GLOBAL_PAGE_BYTES: usize = GLOBAL_PAGE_ENTRIES * GLOBAL_ENTRY_BYTES;

/// Bytes of global KV one token costs, re-derived from the page geometry.
#[must_use]
pub const fn derived_global_kv_bytes_per_token() -> usize {
    GLOBAL_PAGE_BYTES / KV_PAGE_TOKENS
}

/// Number of FP8 values in one SWA key/value vector.
pub const SWA_SLOT_DIM: usize = 512;

/// Elements per E8M0 scale in an SWA slot.
pub const SWA_FP8_SCALE_GROUP: usize = 32;

/// FP8 E4M3 payload of one SWA slot.
pub const SWA_SLOT_DATA_BYTES: usize = SWA_SLOT_DIM;

/// E8M0 scale bytes of one SWA slot.
pub const SWA_SLOT_SCALE_BYTES: usize = SWA_SLOT_DIM / SWA_FP8_SCALE_GROUP;

/// Total bytes of one layer's SWA slot: 512 data + 16 scales.
pub const SWA_SLOT_BYTES: usize = SWA_SLOT_DATA_BYTES + SWA_SLOT_SCALE_BYTES;

/// Slots in one SWA ring: the sliding window width.
pub const SWA_SLOTS_PER_TAIL: usize = crate::config::SWA_WINDOW_SIZE;

/// Backbone layers that keep a sliding-window cache.
pub const SWA_LAYERS: usize = NUM_HIDDEN_LAYERS;

/// Encoder layers 0..=20, i.e. the 21 layers of the D7 encoder half.
pub const SWA_ENCODER_LAYERS: usize = ENCODER_LAST_LAYER + 1;

/// Decoder layers 21..=39, i.e. the 19 layers of the D7 decoder half.
pub const SWA_DECODER_LAYERS: usize = SWA_LAYERS - SWA_ENCODER_LAYERS;

/// Bytes of one sequence's complete 128-token SWA tail across all 40 layers.
pub const SWA_TAIL_BYTES: usize = SWA_LAYERS * SWA_SLOTS_PER_TAIL * SWA_SLOT_BYTES;

/// Bytes of one sequence's SWA tail for the encoder half (layers 0..=20) of the backbone.
pub const SWA_TAIL_ENCODER_BYTES: usize = SWA_ENCODER_LAYERS * SWA_SLOTS_PER_TAIL * SWA_SLOT_BYTES;

/// Bytes of one sequence's SWA tail for the decoder half (layers 21..=39) of the backbone.
pub const SWA_TAIL_DECODER_BYTES: usize = SWA_DECODER_LAYERS * SWA_SLOTS_PER_TAIL * SWA_SLOT_BYTES;

/// One gibibyte, as u64.
pub const GIB: u64 = 1 << 30;

/// The task-level per-node unified-memory ceiling: 121.7 GiB.
pub const NODE_MEMORY_BUDGET_BYTES: u64 = 1217 * GIB / 10;

/// Guard band the pool must leave unallocated: 8 GiB.
///
/// The specification's own budget table (spec line 180) books a 9.12 GiB "MemAvailable floor
/// guard" and the preflight gate wants MemAvailable >= 114 GiB before weights load; the
/// milestone task states the floor as 8 GiB. Both are exposed so a caller can pick, and the
/// discrepancy is reported rather than silently reconciled.
pub const MEMORY_FLOOR_BYTES: u64 = 8 * GIB;

/// The specification's MemAvailable floor guard, 9.12 GiB.
pub const SPEC_SAFETY_HEADROOM_BYTES: u64 = 912 * GIB / 100;

/// Global KV pool size from the specification's budget table: 13.4 GiB.
pub const DEFAULT_GLOBAL_KV_POOL_BYTES: u64 = 134 * GIB / 10;

/// Active SWA tails the specification budgets for.
pub const DEFAULT_SWA_TAIL_SLOTS: usize = 128;

/// Rounds value up to the next multiple of alignment.
///
/// # Panics
/// Panics when alignment is zero.
#[must_use]
pub const fn align_up(value: u64, alignment: u64) -> u64 {
    assert!(alignment != 0, "alignment must be non-zero");
    value.div_ceil(alignment) * alignment
}

/// Rounds value down to the previous multiple of alignment.
///
/// # Panics
/// Panics when alignment is zero.
#[must_use]
pub const fn align_down(value: u64, alignment: u64) -> u64 {
    assert!(alignment != 0, "alignment must be non-zero");
    value / alignment * alignment
}

/// Position of layer in KV_SOURCE_LAYERS, or None when it owns no global KV.
#[must_use]
pub fn source_slot(layer: usize) -> Option<usize> {
    KV_SOURCE_LAYERS.iter().position(|l| *l == layer)
}

/// Entries a Full-mode source layer stores per 64-token page: 64 / compress_ratio.
///
/// # Panics
/// Panics when slot is not a valid index into KV_SOURCE_LAYERS.
#[must_use]
pub fn entries_per_page(slot: usize) -> usize {
    let layer = KV_SOURCE_LAYERS[slot];
    KV_PAGE_TOKENS / compress_ratio(layer)
}

/// Byte offset of source slot inside a page.
///
/// # Panics
/// Panics when slot is not a valid index into KV_SOURCE_LAYERS.
#[must_use]
pub fn page_slot_offset(slot: usize) -> usize {
    assert!(slot < KV_SOURCE_LAYERS.len(), "source slot out of range");
    (0..slot)
        .map(|s| entries_per_page(s) * GLOBAL_ENTRY_BYTES)
        .sum()
}

/// Byte offset of the entry-th compressed entry of a source slot inside a page.
///
/// # Panics
/// Panics when slot or entry is out of range for the page layout.
#[must_use]
pub fn page_entry_offset(slot: usize, entry: usize) -> usize {
    let count = entries_per_page(slot);
    assert!(entry < count, "entry out of range for source slot");
    page_slot_offset(slot) + entry * GLOBAL_ENTRY_BYTES
}

/// Byte offset of one layer's slot inside an SWA tail.
///
/// # Panics
/// Panics when layer or slot is outside the 40 x 128 window.
#[must_use]
pub fn swa_slot_offset(layer: usize, slot: usize) -> usize {
    assert!(layer < SWA_LAYERS, "swa layer out of range");
    assert!(slot < SWA_SLOTS_PER_TAIL, "swa slot out of range");
    (layer * SWA_SLOTS_PER_TAIL + slot) * SWA_SLOT_BYTES
}

/// Byte range one layer occupies inside an SWA tail.
///
/// # Panics
/// Panics when layer is outside the backbone.
#[must_use]
pub fn swa_layer_range(layer: usize) -> std::ops::Range<usize> {
    assert!(layer < SWA_LAYERS, "swa layer out of range");
    layer * SWA_SLOTS_PER_TAIL * SWA_SLOT_BYTES..(layer + 1) * SWA_SLOTS_PER_TAIL * SWA_SLOT_BYTES
}

/// Bytes reserved for global KV by a token count of context, rounded up to whole pages.
#[must_use]
pub const fn global_kv_bytes_for_tokens(tokens: u64) -> u64 {
    tokens.div_ceil(KV_PAGE_TOKENS as u64) * GLOBAL_PAGE_BYTES as u64
}

/// How many whole 64-token pages a token count covers.
#[must_use]
pub const fn pages_for_tokens(tokens: u64) -> u64 {
    tokens / KV_PAGE_TOKENS as u64
}

/// A per-node memory budget split into a ceiling and a floor guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudget {
    /// Total unified memory the node may use, in bytes.
    pub node_bytes: u64,
    /// Bytes that must stay unallocated as an OOM guard.
    pub floor_bytes: u64,
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self {
            node_bytes: NODE_MEMORY_BUDGET_BYTES,
            floor_bytes: MEMORY_FLOOR_BYTES,
        }
    }
}

impl MemoryBudget {
    /// Builds a budget, rejecting a floor larger than the ceiling.
    ///
    /// # Errors
    /// Returns crate::KvError::BudgetExceeded when the floor exceeds the ceiling.
    pub fn new(node_bytes: u64, floor_bytes: u64) -> Result<Self, crate::KvError> {
        if floor_bytes > node_bytes {
            return Err(crate::KvError::BudgetExceeded {
                required: floor_bytes,
                budget: node_bytes,
            });
        }
        Ok(Self {
            node_bytes,
            floor_bytes,
        })
    }

    /// Bytes left for pools after the floor guard.
    #[must_use]
    pub const fn usable_bytes(&self) -> u64 {
        self.node_bytes - self.floor_bytes
    }

    /// Whole pages of a given size that fit in the usable budget.
    ///
    /// # Panics
    /// Panics when page_bytes is zero.
    #[must_use]
    pub const fn pages_for(&self, page_bytes: u64) -> u64 {
        assert!(page_bytes != 0, "page size must be non-zero");
        self.usable_bytes() / page_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_offsets_are_dense_and_cover_the_page() {
        let mut expected = 0usize;
        for slot in 0..KV_SOURCE_LAYERS.len() {
            assert_eq!(page_slot_offset(slot), expected);
            expected += entries_per_page(slot) * GLOBAL_ENTRY_BYTES;
        }
        assert_eq!(expected, GLOBAL_PAGE_BYTES);
        assert_eq!(
            page_entry_offset(3, 63),
            GLOBAL_PAGE_BYTES - GLOBAL_ENTRY_BYTES
        );
    }

    #[test]
    fn budget_floor_is_checked() {
        assert!(MemoryBudget::new(GIB, 2 * GIB).is_err());
        let budget = MemoryBudget::new(100 * GIB, 8 * GIB).expect("valid budget");
        assert_eq!(budget.usable_bytes(), 92 * GIB);
        assert_eq!(
            budget.pages_for(GLOBAL_PAGE_BYTES as u64),
            92 * GIB / 56_960
        );
    }
}

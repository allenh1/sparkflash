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

//! Re-derives the byte geometry from the checkpoint config instead of re-asserting it.

use sparkflash_kv::config::{KV_PAGE_TOKENS, KV_SOURCE_LAYERS, NUM_HIDDEN_LAYERS, compress_ratio};
use sparkflash_kv::fp4::{IndexerKEntry, MainKvEntry, SwaKvSlot};
use sparkflash_kv::layout::{
    self, DEFAULT_SWA_TAIL_SLOTS, GIB, GLOBAL_ENTRY_BYTES, GLOBAL_PAGE_BYTES, GLOBAL_PAGE_ENTRIES,
    INDEXER_K_DATA_BYTES, INDEXER_K_DATA_OFFSET, INDEXER_K_DIM, INDEXER_K_ENTRY_BYTES,
    INDEXER_K_SCALE_BYTES, INDEXER_K_SCALE_GROUP, INDEXER_K_SCALE_OFFSET, MAIN_KV_DATA_BYTES,
    MAIN_KV_DATA_OFFSET, MAIN_KV_DIM, MAIN_KV_ENTRY_BYTES, MAIN_KV_SCALE_BYTES,
    MAIN_KV_SCALE_GROUP, MAIN_KV_SCALE_OFFSET, MEMORY_FLOOR_BYTES, MemoryBudget,
    NODE_MEMORY_BUDGET_BYTES, SWA_LAYERS, SWA_SLOT_BYTES, SWA_SLOTS_PER_TAIL, SWA_TAIL_BYTES,
    align_up, entries_per_page, global_kv_bytes_for_tokens, pages_for_tokens, swa_layer_range,
    swa_slot_offset,
};

#[test]
fn one_entry_is_356_bytes_from_the_dimensions() {
    assert_eq!(MAIN_KV_DIM, 512);
    assert_eq!(INDEXER_K_DIM, 128);
    assert_eq!(MAIN_KV_SCALE_GROUP, 16);
    assert_eq!(INDEXER_K_SCALE_GROUP, 32);
    // 512 FP4 values at two per byte, plus one E4M3 scale per 16 channels.
    assert_eq!(MAIN_KV_DIM / 2 + MAIN_KV_DIM / 16, MAIN_KV_ENTRY_BYTES);
    assert_eq!(MAIN_KV_ENTRY_BYTES, 288);
    // 128 FP4 values at two per byte, plus one E8M0 scale per 32 channels.
    assert_eq!(
        INDEXER_K_DIM / 2 + INDEXER_K_DIM / 32,
        INDEXER_K_ENTRY_BYTES
    );
    assert_eq!(INDEXER_K_ENTRY_BYTES, 68);
    assert_eq!(
        MAIN_KV_ENTRY_BYTES + INDEXER_K_ENTRY_BYTES,
        GLOBAL_ENTRY_BYTES
    );
    assert_eq!(GLOBAL_ENTRY_BYTES, 356);
}

#[test]
fn pinned_offsets_are_dense_and_match_the_buffer_sizes() {
    assert_eq!(MAIN_KV_DATA_OFFSET, 0);
    // MainKvEntry is the whole main entry: the quantised latent plus its scales.
    assert_eq!(std::mem::size_of::<MainKvEntry>(), MAIN_KV_ENTRY_BYTES);
    assert_eq!(
        MAIN_KV_DATA_BYTES + MAIN_KV_SCALE_BYTES,
        MAIN_KV_ENTRY_BYTES
    );
    assert_eq!(
        MAIN_KV_SCALE_OFFSET,
        MAIN_KV_DATA_OFFSET + MAIN_KV_DATA_BYTES
    );
    assert_eq!(MAIN_KV_SCALE_BYTES, MAIN_KV_DIM / MAIN_KV_SCALE_GROUP);
    assert_eq!(
        INDEXER_K_DATA_OFFSET,
        MAIN_KV_SCALE_OFFSET + MAIN_KV_SCALE_BYTES
    );
    assert_eq!(INDEXER_K_DATA_BYTES, INDEXER_K_DIM / 2);
    assert_eq!(std::mem::size_of::<IndexerKEntry>(), INDEXER_K_ENTRY_BYTES);
    assert_eq!(
        INDEXER_K_SCALE_OFFSET,
        INDEXER_K_DATA_OFFSET + INDEXER_K_DATA_BYTES
    );
    assert_eq!(
        INDEXER_K_SCALE_OFFSET + INDEXER_K_SCALE_BYTES,
        GLOBAL_ENTRY_BYTES
    );
    assert_eq!(INDEXER_K_SCALE_BYTES, INDEXER_K_DIM / INDEXER_K_SCALE_GROUP);
}

#[test]
fn an_entry_round_trips_through_the_pinned_layout() {
    let main = MainKvEntry {
        data: [0xA5; MAIN_KV_DATA_BYTES],
        scales: [0x11; MAIN_KV_SCALE_BYTES],
    };
    let indexer = IndexerKEntry {
        data: [0x5A; INDEXER_K_DATA_BYTES],
        scales: [0x22; INDEXER_K_SCALE_BYTES],
    };
    let mut buffer = [0u8; GLOBAL_ENTRY_BYTES];
    main.to_bytes(&mut buffer[MAIN_KV_DATA_OFFSET..INDEXER_K_DATA_OFFSET])
        .expect("main fits");
    indexer
        .to_bytes(&mut buffer[INDEXER_K_DATA_OFFSET..])
        .expect("indexer fits");
    assert!(buffer[..MAIN_KV_DATA_BYTES].iter().all(|b| *b == 0xA5));
    assert!(
        buffer[MAIN_KV_SCALE_OFFSET..INDEXER_K_DATA_OFFSET]
            .iter()
            .all(|b| *b == 0x11)
    );
    assert!(
        buffer[INDEXER_K_DATA_OFFSET..INDEXER_K_SCALE_OFFSET]
            .iter()
            .all(|b| *b == 0x5A)
    );
    assert!(buffer[INDEXER_K_SCALE_OFFSET..].iter().all(|b| *b == 0x22));
    let back_main = MainKvEntry::from_bytes(&buffer[MAIN_KV_DATA_OFFSET..INDEXER_K_DATA_OFFSET])
        .expect("main parses");
    let back_indexer =
        IndexerKEntry::from_bytes(&buffer[INDEXER_K_DATA_OFFSET..]).expect("indexer parses");
    assert_eq!(back_main, main);
    assert_eq!(back_indexer, indexer);
}

#[test]
fn two_and_a_half_entries_per_token_makes_890_bytes() {
    // Entries per 64-token page, summed over the four Full-mode layers.
    let entries: usize = KV_SOURCE_LAYERS
        .iter()
        .map(|layer| KV_PAGE_TOKENS / compress_ratio(*layer))
        .sum();
    assert_eq!(entries, GLOBAL_PAGE_ENTRIES);
    assert_eq!(entries, 32 + 32 + 32 + 64);
    // 160 entries per 64 tokens is exactly 5/2 entries per token.
    assert_eq!(entries as u64 * 2, KV_PAGE_TOKENS as u64 * 5);
    assert_eq!(entries * GLOBAL_ENTRY_BYTES, GLOBAL_PAGE_BYTES);
    assert_eq!(GLOBAL_PAGE_BYTES, 56_960);
    assert_eq!(
        GLOBAL_PAGE_BYTES / KV_PAGE_TOKENS,
        layout::derived_global_kv_bytes_per_token()
    );
    assert_eq!(layout::derived_global_kv_bytes_per_token(), 890);
    assert_eq!(
        layout::derived_global_kv_bytes_per_token(),
        sparkflash_kv::config::GLOBAL_KV_BYTES_PER_TOKEN
    );
}

#[test]
fn the_specs_itemisation_double_counts_the_rotary_dimensions() {
    // Spec line 76 itemises 256 + 32 + 64 (qk_rope) + 64 + 4 per entry. head_dim = 512 already
    // contains the 64 rotary dims, so that sum is 420, and 2.5 * 420 = 1050, not 890.
    let spec_entry = 256 + 32 + 64 + 64 + 4;
    assert_eq!(spec_entry, 420);
    let spec_per_token = spec_entry * 5 / 2;
    assert_eq!(spec_per_token, 1050);
    assert_ne!(spec_per_token, layout::derived_global_kv_bytes_per_token());
    assert_eq!(
        spec_per_token - layout::derived_global_kv_bytes_per_token(),
        64 * 5 / 2
    );
}

#[test]
fn page_offsets_cover_every_layer_exactly_once() {
    let mut expected = 0usize;
    for (slot, layer) in KV_SOURCE_LAYERS.iter().enumerate() {
        assert_eq!(layout::page_slot_offset(slot), expected);
        let count = entries_per_page(slot);
        assert_eq!(count, KV_PAGE_TOKENS / compress_ratio(*layer));
        for entry in 0..count {
            assert_eq!(
                layout::page_entry_offset(slot, entry),
                expected + entry * GLOBAL_ENTRY_BYTES
            );
        }
        expected += count * GLOBAL_ENTRY_BYTES;
    }
    assert_eq!(expected, GLOBAL_PAGE_BYTES);
}

#[test]
fn swa_geometry_is_a_128_slot_40_layer_ring() {
    assert_eq!(SWA_SLOTS_PER_TAIL, 128);
    assert_eq!(SWA_LAYERS, NUM_HIDDEN_LAYERS);
    assert_eq!(SWA_SLOT_BYTES, std::mem::size_of::<SwaKvSlot>());
    assert_eq!(SWA_SLOT_BYTES, 512 + 16);
    assert_eq!(
        SWA_TAIL_BYTES,
        SWA_LAYERS * SWA_SLOTS_PER_TAIL * SWA_SLOT_BYTES
    );
    assert_eq!(SWA_TAIL_BYTES, 2_703_360);
    let mut expected = 0usize;
    for layer in 0..SWA_LAYERS {
        let range = swa_layer_range(layer);
        assert_eq!(range.start, expected);
        assert_eq!(range.len(), SWA_SLOTS_PER_TAIL * SWA_SLOT_BYTES);
        expected = range.end;
        assert_eq!(swa_slot_offset(layer, 0), range.start);
        assert_eq!(
            swa_slot_offset(layer, SWA_SLOTS_PER_TAIL - 1) + SWA_SLOT_BYTES,
            range.end
        );
    }
    assert_eq!(expected, SWA_TAIL_BYTES);
}

#[test]
fn the_default_budget_fits_the_default_configuration() {
    let budget = MemoryBudget::default();
    assert_eq!(budget.node_bytes, NODE_MEMORY_BUDGET_BYTES);
    assert_eq!(budget.floor_bytes, MEMORY_FLOOR_BYTES);
    assert_eq!(
        budget.usable_bytes(),
        NODE_MEMORY_BUDGET_BYTES - MEMORY_FLOOR_BYTES
    );
    // 121.7 GiB node ceiling, 8 GiB floor: the pools must fit in the remainder.
    let swa_bytes = DEFAULT_SWA_TAIL_SLOTS as u64 * SWA_TAIL_BYTES as u64;
    assert!(swa_bytes < budget.usable_bytes());
    assert_eq!(
        budget.pages_for(GLOBAL_PAGE_BYTES as u64),
        budget.usable_bytes() / 56_960
    );
}

#[test]
fn token_arithmetic_rounds_up_to_whole_pages() {
    // pages_for_tokens counts *whole* pages (it is the floor); global_kv_bytes_for_tokens
    // reserves *enough* pages (it is the ceiling). The asymmetry is deliberate and pinned.
    assert_eq!(pages_for_tokens(0), 0);
    assert_eq!(pages_for_tokens(63), 0);
    assert_eq!(pages_for_tokens(64), 1);
    assert_eq!(pages_for_tokens(65), 1);
    assert_eq!(pages_for_tokens(128), 2);
    assert_eq!(global_kv_bytes_for_tokens(0), 0);
    assert_eq!(global_kv_bytes_for_tokens(1), GLOBAL_PAGE_BYTES as u64);
    assert_eq!(global_kv_bytes_for_tokens(64), GLOBAL_PAGE_BYTES as u64);
    assert_eq!(global_kv_bytes_for_tokens(65), 2 * GLOBAL_PAGE_BYTES as u64);
    assert_eq!(
        global_kv_bytes_for_tokens(128),
        2 * GLOBAL_PAGE_BYTES as u64
    );
    // One million tokens at the derived 890 B/token is 890 MB, i.e. 0.829 GiB.
    let million = (1_000_000u64 * layout::derived_global_kv_bytes_per_token() as u64) as f64;
    assert_eq!(million, 890_000_000.0);
    let gib = million / GIB as f64;
    assert!((gib - 0.8289).abs() < 0.001, "1M tokens = {gib} GiB");
    // The specification's own sentence names 890 B/token; its itemisation implies 1050.
    // Both are recorded here so a future reader sees the discrepancy, not a tidied constant.
    assert_eq!(million as u64, 1_000_000 * 890);
}

#[test]
fn alignment_helpers_round_the_right_way() {
    assert_eq!(align_up(0, 512), 0);
    assert_eq!(align_up(1, 512), 512);
    assert_eq!(align_up(512, 512), 512);
    assert_eq!(align_up(56_960, 4096), 57_344);
    assert_eq!(layout::align_down(56_960, 4096), 53_248);
}

//! The layout-contract guard: the device seam's assumed strides are the kv layout's definition.
//!
//! The task-49 design note warned that a mismatch between the device-side addressing constants
//! (356-byte interleaved entries, 288/256/352 offsets, 64-token pages) and sparkflash-kv::layout is
//! silent corruption, not a failed check. These tests turn that warning into a guard: they exercise
//! the already-wired PagedKvView/paged_kv paths and the paged-read reference against the layout
//! constants, and they hold the CUDA common header's own defines to the same source of truth. The
//! compile-time half lives in sparkflash_device::layout_contract.

use sparkflash_device::layout_contract::*;
use sparkflash_device::paged_kv::PagedKvView;
use sparkflash_device::paged_read::{paged_entry, read_hsi_keys, read_main_rows};
use sparkflash_kernels::cuda_harness::WindowRingLayout;
use sparkflash_kernels::m4_abi::FLASH_MLA_CSA2;
use sparkflash_kernels::m4_quant::bf16_bits;
use sparkflash_kernels::mla_ref::{
    HSI_BLOCK_TOKENS, HSI_BLOCKS, INDEX_HEAD_DIM, MLA_HEADS, MLA_LATENT,
};
use sparkflash_kv::config::{KV_PAGE_TOKENS, KV_SOURCE_LAYERS, compress_ratio};
use sparkflash_kv::fp4::{IndexerKEntry, dequantize_indexer_k, quantize_indexer_k};
use sparkflash_kv::layout::{
    GLOBAL_ENTRY_BYTES, INDEXER_K_DATA_BYTES, INDEXER_K_DATA_OFFSET, INDEXER_K_DIM,
    INDEXER_K_ENTRY_BYTES, INDEXER_K_SCALE_BYTES, INDEXER_K_SCALE_GROUP, INDEXER_K_SCALE_OFFSET,
    MAIN_KV_DATA_BYTES, MAIN_KV_DATA_OFFSET, MAIN_KV_DIM, MAIN_KV_ENTRY_BYTES, MAIN_KV_SCALE_BYTES,
    MAIN_KV_SCALE_GROUP, MAIN_KV_SCALE_OFFSET, SWA_FP8_SCALE_GROUP, SWA_LAYERS, SWA_SLOT_BYTES,
    SWA_SLOT_DATA_BYTES, SWA_SLOT_DIM, SWA_SLOT_SCALE_BYTES, SWA_SLOTS_PER_TAIL, SWA_TAIL_BYTES,
    entries_per_page, source_slot, swa_slot_offset,
};
use sparkflash_kv::paged::{PagedGeometry, PagedKvCache};

const SEQ: u64 = 1;

fn tokens(count: usize) -> Vec<u32> {
    (0..count as u32).collect()
}

fn served(pages: u32, seq_tokens: usize) -> PagedKvCache {
    let mut cache = PagedKvCache::new(PagedGeometry::new(pages, 1).expect("geometry"));
    cache.add_sequence(SEQ).expect("add");
    cache.append(SEQ, &tokens(seq_tokens)).expect("append");
    cache.check_invariants().expect("invariants");
    cache
}

fn table_array(cache: &PagedKvCache, sequence: u64) -> Vec<u32> {
    cache
        .table(sequence)
        .expect("table")
        .pages()
        .iter()
        .map(|page| u32::try_from(page.0).expect("page id fits u32"))
        .collect()
}

fn indexed_pool(layer: usize, pages: u32) -> Vec<u8> {
    let slot = source_slot(layer).expect("a source layer");
    let entries = pages as usize * entries_per_page(slot);
    let mut pool = vec![0u8; entries * GLOBAL_ENTRY_BYTES];
    for entry in 0..entries {
        pool[entry * GLOBAL_ENTRY_BYTES..(entry + 1) * GLOBAL_ENTRY_BYTES]
            .fill((entry % 251) as u8);
    }
    pool
}

fn write_indexer(pool: &mut [u8], block_table: &[u32], token: u32) {
    let per_page = KV_PAGE_TOKENS / compress_ratio(20);
    let entry = block_table[token as usize / KV_PAGE_TOKENS] as usize * per_page
        + token as usize % KV_PAGE_TOKENS;
    let values: [f32; INDEXER_K_DIM] = std::array::from_fn(|channel| {
        ((f64::from(token) * 0.01 + f64::from(channel as u32) * 0.001).sin() * 1.5) as f32
    });
    let indexer = quantize_indexer_k(&values).expect("indexer k");
    let base = entry * GLOBAL_ENTRY_BYTES;
    pool[base + INDEXER_K_DATA_OFFSET..base + INDEXER_K_DATA_OFFSET + INDEXER_K_DATA_BYTES]
        .copy_from_slice(&indexer.data);
    pool[base + INDEXER_K_SCALE_OFFSET..base + INDEXER_K_SCALE_OFFSET + INDEXER_K_SCALE_BYTES]
        .copy_from_slice(&indexer.scales);
}

/// The compile-time half already refuses a layout change; this restates every assumed value at
/// runtime and checks the derived page/ring geometry the addressing relies on.
#[test]
fn the_device_assumptions_are_the_kv_layout_single_source() {
    assert_eq!(ASSUMED_GLOBAL_ENTRY_BYTES, GLOBAL_ENTRY_BYTES);
    assert_eq!(ASSUMED_MAIN_KV_ENTRY_BYTES, MAIN_KV_ENTRY_BYTES);
    assert_eq!(ASSUMED_MAIN_KV_DATA_OFFSET, MAIN_KV_DATA_OFFSET);
    assert_eq!(ASSUMED_MAIN_KV_DATA_BYTES, MAIN_KV_DATA_BYTES);
    assert_eq!(ASSUMED_MAIN_KV_SCALE_OFFSET, MAIN_KV_SCALE_OFFSET);
    assert_eq!(ASSUMED_MAIN_KV_SCALE_BYTES, MAIN_KV_SCALE_BYTES);
    assert_eq!(ASSUMED_INDEXER_K_DATA_OFFSET, INDEXER_K_DATA_OFFSET);
    assert_eq!(ASSUMED_INDEXER_K_DATA_BYTES, INDEXER_K_DATA_BYTES);
    assert_eq!(ASSUMED_INDEXER_K_SCALE_OFFSET, INDEXER_K_SCALE_OFFSET);
    assert_eq!(ASSUMED_INDEXER_K_SCALE_BYTES, INDEXER_K_SCALE_BYTES);
    assert_eq!(ASSUMED_INDEXER_K_ENTRY_BYTES, INDEXER_K_ENTRY_BYTES);
    assert_eq!(ASSUMED_KV_PAGE_TOKENS, KV_PAGE_TOKENS);
    assert_eq!(ASSUMED_MLA_LATENT, MAIN_KV_DIM);
    assert_eq!(ASSUMED_MLA_HEADS, MLA_HEADS);
    assert_eq!(ASSUMED_MAIN_KV_SCALE_GROUP, MAIN_KV_SCALE_GROUP);
    assert_eq!(ASSUMED_INDEXER_K_DIM, INDEXER_K_DIM);
    assert_eq!(ASSUMED_INDEXER_K_SCALE_GROUP, INDEXER_K_SCALE_GROUP);
    assert_eq!(ASSUMED_SWA_SLOTS, SWA_SLOTS_PER_TAIL);
    assert_eq!(ASSUMED_SWA_LAYERS, SWA_LAYERS);
    assert_eq!(ASSUMED_SWA_SLOT_DATA_BYTES, SWA_SLOT_DATA_BYTES);
    assert_eq!(ASSUMED_SWA_SLOT_SCALE_BYTES, SWA_SLOT_SCALE_BYTES);
    assert_eq!(ASSUMED_SWA_SLOT_BYTES, SWA_SLOT_BYTES);
    assert_eq!(ASSUMED_SWA_SLOT_DIM, SWA_SLOT_DIM);
    assert_eq!(ASSUMED_SWA_SCALE_GROUP, SWA_FP8_SCALE_GROUP);
    assert_eq!(ASSUMED_INDEX_TOPK, 512);
    assert_eq!(ASSUMED_HSI_BLOCKS, HSI_BLOCKS);
    assert_eq!(ASSUMED_HSI_BLOCK_TOKENS, HSI_BLOCK_TOKENS);
    assert_eq!(ASSUMED_INDEXER_K_DIM, INDEX_HEAD_DIM);
    assert_eq!(ASSUMED_MLA_LATENT, MLA_LATENT);

    // The frame the paged-read design addresses: 64 tokens per page, 1,152-byte pages of 32 main
    // rows, and the main-KV page fraction the byte arithmetic in the design note quotes.
    assert_eq!(MAIN_KV_DATA_BYTES, 256);
    assert_eq!(MAIN_KV_SCALE_BYTES, 32);
    assert_eq!(MAIN_KV_ENTRY_BYTES, 288);
    assert_eq!(INDEXER_K_ENTRY_BYTES, 68);
    assert_eq!(GLOBAL_ENTRY_BYTES, 356);
    assert_eq!(KV_PAGE_TOKENS, 64);
    assert_eq!(MAIN_KV_ENTRY_BYTES * 4, 1_152);
    assert_eq!(GLOBAL_ENTRY_BYTES * 32, 11_392);

    // The page geometry is the kv layout's, and both source ratios still divide the page.
    assert_eq!(KV_SOURCE_LAYERS, [2, 8, 14, 20]);
    assert_eq!(compress_ratio(2), 2);
    assert_eq!(compress_ratio(8), 2);
    assert_eq!(compress_ratio(14), 2);
    assert_eq!(compress_ratio(20), 1);
    assert_eq!(source_slot(2), Some(0));
    assert_eq!(source_slot(8), Some(1));
    assert_eq!(source_slot(14), Some(2));
    assert_eq!(source_slot(20), Some(3));
    assert_eq!(entries_per_page(0), ASSUMED_KV_PAGE_TOKENS / 2);
    assert_eq!(entries_per_page(1), ASSUMED_KV_PAGE_TOKENS / 2);
    assert_eq!(entries_per_page(2), ASSUMED_KV_PAGE_TOKENS / 2);
    assert_eq!(entries_per_page(3), ASSUMED_KV_PAGE_TOKENS);
    assert_eq!(ASSUMED_KV_PAGE_TOKENS % compress_ratio(2), 0);
    assert_eq!(ASSUMED_KV_PAGE_TOKENS % compress_ratio(20), 0);

    // The window ring is the kv slot layout, not a second definition of it.
    assert_eq!(WindowRingLayout::ENGINE.slots, SWA_SLOTS_PER_TAIL);
    assert_eq!(WindowRingLayout::ENGINE.code_row_bytes, SWA_SLOT_DATA_BYTES);
    assert_eq!(
        WindowRingLayout::ENGINE.scale_row_bytes,
        SWA_SLOT_SCALE_BYTES
    );
    assert_eq!(WindowRingLayout::ENGINE.slot_bytes(), SWA_SLOT_BYTES);
    assert_eq!(WindowRingLayout::ENGINE.slot_bytes(), 528);
    assert_eq!(WindowRingLayout::ENGINE.ring_bytes(), 67_584);
    assert_eq!(
        SWA_TAIL_BYTES,
        SWA_LAYERS * WindowRingLayout::ENGINE.ring_bytes()
    );
    assert_eq!(SWA_TAIL_BYTES, 2_703_360);
    assert_eq!(
        swa_slot_offset(0, 1) - swa_slot_offset(0, 0),
        SWA_SLOT_BYTES
    );
    assert_eq!(
        swa_slot_offset(1, 0) - swa_slot_offset(0, 0),
        SWA_SLOT_BYTES * SWA_SLOTS_PER_TAIL
    );
}

/// The already-wired PagedKvView path computes byte offsets from the layout constants, and the
/// paged-read reference maps every position to the same pool entry. This is the guard that the
/// staging path and the proposed read path share one layout.
#[test]
fn paged_view_byte_offsets_are_the_layout_strides() {
    let cache = served(8, 137);
    let view = PagedKvView::new(&cache);
    let positions = [0usize, 1, 31, 32, 62, 63, 64, 65, 127, 128, 136];

    for layer in [2usize, 20] {
        let slot = source_slot(layer).expect("source layer");
        let ratio = compress_ratio(layer);
        let table = table_array(&cache, SEQ);
        let pool = indexed_pool(layer, 8);
        let pool_entries = pool.len() / GLOBAL_ENTRY_BYTES;

        for &position in &positions {
            let entry = view.source_entry(SEQ, layer, position).expect("entry");
            assert_eq!(entry.page, u64::from(table[position / KV_PAGE_TOKENS]));
            assert_eq!(entry.entry_in_page, position % KV_PAGE_TOKENS / ratio);
            assert_eq!(
                entry.pool_index,
                entry.page as usize * entries_per_page(slot) + entry.entry_in_page
            );
            assert_eq!(entry.byte_offset, entry.pool_index * GLOBAL_ENTRY_BYTES);

            // Every field the device reads lies inside the one interleaved entry.
            let base = entry.byte_offset;
            assert!(base + GLOBAL_ENTRY_BYTES <= pool.len());
            assert_eq!(
                MAIN_KV_DATA_OFFSET + MAIN_KV_DATA_BYTES,
                MAIN_KV_SCALE_OFFSET
            );
            assert_eq!(
                MAIN_KV_SCALE_OFFSET + MAIN_KV_SCALE_BYTES,
                MAIN_KV_ENTRY_BYTES
            );
            assert_eq!(MAIN_KV_ENTRY_BYTES, INDEXER_K_DATA_OFFSET);
            assert_eq!(
                INDEXER_K_DATA_OFFSET + INDEXER_K_DATA_BYTES,
                INDEXER_K_SCALE_OFFSET
            );
            assert_eq!(
                INDEXER_K_SCALE_OFFSET + INDEXER_K_SCALE_BYTES,
                GLOBAL_ENTRY_BYTES
            );

            // The paged-read reference resolves the same id to the same entry.
            assert_eq!(
                paged_entry(
                    pool_entries,
                    &table,
                    ratio,
                    u32::try_from(position).unwrap()
                )
                .expect("paged entry"),
                Some(entry.pool_index)
            );

            // The staged main rows are the bytes at the layout offsets, not at guessed ones.
            let rows = read_main_rows(&pool, &table, ratio, &[u32::try_from(position).unwrap()])
                .expect("main rows");
            assert_eq!(
                rows.codes,
                pool[base + MAIN_KV_DATA_OFFSET..base + MAIN_KV_DATA_OFFSET + MAIN_KV_DATA_BYTES]
            );
            assert_eq!(
                rows.scales,
                pool[base + MAIN_KV_SCALE_OFFSET
                    ..base + MAIN_KV_SCALE_OFFSET + MAIN_KV_SCALE_BYTES]
            );
        }

        // The dense staging path uses the same layout strides too.
        let last = *positions.last().unwrap();
        let staged = view
            .absolute_source_rows(SEQ, layer, last, &pool)
            .expect("absolute rows");
        assert_eq!(staged.codes.len(), (last + 1) * MAIN_KV_DATA_BYTES);
        assert_eq!(staged.scales.len(), (last + 1) * MAIN_KV_SCALE_BYTES);
    }
}

/// The window gather (the already-wired SWA operand path) is the kv slot layout: 128 rows of 528
/// bytes, addressed by swa_slot_offset, with ring slot = logical position % 128.
#[test]
fn window_gather_stride_is_the_layout_slot() {
    let cache = served(8, 300);
    let view = PagedKvView::new(&cache);
    let layer = 20usize;
    let position = 299usize;

    let mut tail = vec![0u8; SWA_TAIL_BYTES];
    for slot in 0..SWA_SLOTS_PER_TAIL {
        let offset = swa_slot_offset(layer, slot);
        for byte in 0..SWA_SLOT_DATA_BYTES {
            tail[offset + byte] = ((slot * 7 + byte) % 251) as u8;
        }
        for byte in 0..SWA_SLOT_SCALE_BYTES {
            tail[offset + SWA_SLOT_DATA_BYTES + byte] = ((slot * 3 + byte) % 61) as u8;
        }
    }

    let window = view
        .window_rows(SEQ, layer, position, &tail)
        .expect("window rows");
    assert_eq!(window.rows, SWA_SLOTS_PER_TAIL);
    assert_eq!(window.codes.len(), window.rows * SWA_SLOT_DATA_BYTES);
    assert_eq!(window.scales.len(), window.rows * SWA_SLOT_SCALE_BYTES);
    assert_eq!(window.rows * SWA_SLOT_BYTES, 67_584);

    for row in 0..window.rows {
        let logical = position - window.rows + row;
        let slot = logical % SWA_SLOTS_PER_TAIL;
        let offset = swa_slot_offset(layer, slot);
        assert_eq!(
            window.codes[row * SWA_SLOT_DATA_BYTES..(row + 1) * SWA_SLOT_DATA_BYTES],
            tail[offset..offset + SWA_SLOT_DATA_BYTES]
        );
        assert_eq!(
            window.scales[row * SWA_SLOT_SCALE_BYTES..(row + 1) * SWA_SLOT_SCALE_BYTES],
            tail[offset + SWA_SLOT_DATA_BYTES..offset + SWA_SLOT_BYTES]
        );
    }
}

/// The HSI indexer pool (the already-wired absolute-indexer-keys path) reads each 68-byte indexer
/// entry out of the interleaved global entry at the layout offsets.
#[test]
fn indexer_pool_offsets_are_the_layout_entry() {
    let cache = served(8, 137);
    let view = PagedKvView::new(&cache);
    let layer = 20usize;
    let position = 133usize;
    let last = u32::try_from(position).unwrap();
    let table = table_array(&cache, SEQ);
    let mut pool = indexed_pool(layer, 8);
    for token in 0..=last {
        write_indexer(&mut pool, &table, token);
    }

    let staged = view
        .absolute_indexer_keys(SEQ, layer, position, &pool)
        .expect("indexer keys");
    assert_eq!(
        staged.keys.len(),
        HSI_BLOCKS * HSI_BLOCK_TOKENS * INDEX_HEAD_DIM
    );
    assert_eq!(staged.keys.len() * 2, ASSUMED_HSI_INDEX_KEY_BYTES);
    assert_eq!(
        staged.reachable_blocks,
        (position + 1).div_ceil(HSI_BLOCK_TOKENS)
    );
    assert_eq!(INDEX_HEAD_DIM, INDEXER_K_DIM);
    assert_eq!(
        INDEXER_K_DATA_BYTES + INDEXER_K_SCALE_BYTES,
        INDEXER_K_ENTRY_BYTES
    );

    for token in 0..=last {
        let entry = view
            .source_entry(SEQ, layer, token as usize)
            .expect("entry");
        assert_eq!(entry.entry_in_page, token as usize % KV_PAGE_TOKENS);
        let start = entry.byte_offset + INDEXER_K_DATA_OFFSET;
        let parsed = IndexerKEntry::from_bytes(&pool[start..start + INDEXER_K_ENTRY_BYTES])
            .expect("68-byte indexer entry");
        let mut values = [0f32; INDEXER_K_DIM];
        dequantize_indexer_k(&parsed, &mut values).expect("dequantize");
        let row = token as usize * INDEX_HEAD_DIM;
        for (channel, value) in values.iter().enumerate() {
            assert_eq!(staged.keys[row + channel], bf16_bits(*value));
        }
    }

    // Slots past the query position are zero-filled, not read from the pool.
    for slot in (position + 1)..HSI_BLOCK_TOKENS {
        let row = slot * INDEX_HEAD_DIM;
        assert!(
            staged.keys[row..row + INDEX_HEAD_DIM]
                .iter()
                .all(|value| *value == 0)
        );
    }

    // The paged-read reference produces the same indexer operand from the same layout.
    assert_eq!(
        read_hsi_keys(&pool, &table, last).expect("hsi keys"),
        staged.keys
    );
}

/// The CUDA common header and the flash MLA kernel carry their own copies of 512/16/64/128/2048/8.
/// This test reads those defines and holds them to the kv layout, so a kernel-side constant that
/// drifts from the layout fails the device test suite even though the frozen ABI is untouched.
#[test]
fn cuda_kernel_defines_are_the_kv_layout() {
    let common = |name: &str| {
        kernel_common_define(name).unwrap_or_else(|| panic!("missing common define {name}"))
    };
    let flash = |name: &str| {
        kernel_flash_mla_define(name).unwrap_or_else(|| panic!("missing flash MLA define {name}"))
    };

    assert_eq!(common("SPARK_M4_KV_LORA_RANK") as usize, MAIN_KV_DIM);
    assert_eq!(
        common("SPARK_M4_KV_SCALE_CHANNELS") as usize,
        MAIN_KV_SCALE_GROUP
    );
    assert_eq!(common("SPARK_M4_KV_PAGE_TOKENS") as usize, KV_PAGE_TOKENS);
    assert_eq!(common("SPARK_M4_SWA_WINDOW") as usize, SWA_SLOTS_PER_TAIL);
    assert_eq!(common("SPARK_M4_INDEX_HEAD_DIM") as usize, INDEXER_K_DIM);
    assert_eq!(common("SPARK_M4_HSI_BLOCKS") as usize, HSI_BLOCKS);
    assert_eq!(
        common("SPARK_M4_HSI_BLOCK_TOKENS") as usize,
        HSI_BLOCK_TOKENS
    );
    assert_eq!(common("SPARK_M4_INDEX_TOPK") as usize, ASSUMED_INDEX_TOPK);

    let latent = common("SPARK_M4_KV_LORA_RANK") as usize;
    let scale_channels = common("SPARK_M4_KV_SCALE_CHANNELS") as usize;
    assert_eq!(latent, SWA_SLOT_DIM);
    assert_eq!(latent / 2, MAIN_KV_DATA_BYTES);
    assert_eq!(latent / scale_channels, MAIN_KV_SCALE_BYTES);
    assert_eq!(scale_channels, MAIN_KV_SCALE_GROUP);

    let window_quant = flash("SPARK_M4_MLA_WINDOW_QUANT_BLOCK") as usize;
    assert_eq!(window_quant, SWA_FP8_SCALE_GROUP);
    assert_eq!(latent / window_quant, SWA_SLOT_SCALE_BYTES);
    assert_eq!(latent, SWA_SLOT_DATA_BYTES);

    // The kernel's arithmetic is written in terms of the same defines, not fresh literals.
    assert!(FLASH_MLA_CSA2.contains("#define SPARK_M4_MLA_LATENT SPARK_M4_KV_LORA_RANK"));
    assert!(FLASH_MLA_CSA2.contains("SPARK_M4_MLA_LATENT / 2u"));
    assert!(FLASH_MLA_CSA2.contains("SPARK_M4_MLA_LATENT / SPARK_M4_KV_SCALE_CHANNELS"));
    assert!(FLASH_MLA_CSA2.contains("(SPARK_M4_MLA_LATENT / SPARK_M4_MLA_WINDOW_QUANT_BLOCK)"));
}

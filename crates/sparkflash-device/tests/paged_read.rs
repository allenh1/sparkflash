//! The paged-read attention kernel's CPU twin (task-49, design only).
//!
//! The design note docs/archive/build/paged-read-attention-kernel-design.md proposes a kernel that reads
//! the paged pool directly instead of the Path B dense staging. Before such an entry is frozen, the
//! read path it would use has to be proved to be the bytes the staging path already produces. These
//! tests are that proof, and they are the only evidence: no kernel, no cubin, no device launch, no
//! throughput claim. The arithmetic is the released mla_ref reference on both sides.

use sparkflash_device::paged_kv::PagedKvView;
use sparkflash_device::paged_read::{BLOCK_TABLE_HOLE, read_hsi_keys, read_main_rows};
use sparkflash_kernels::m4_quant::bf16_bits;
use sparkflash_kernels::mla_ref::{
    HSI_BLOCK_TOKENS, HSI_BLOCKS, INDEX_HEAD_DIM, attention_with_sink_union, indexer_block_scores,
};
use sparkflash_kv::config::{KV_PAGE_TOKENS, compress_ratio};
use sparkflash_kv::fp4::quantize_indexer_k;
use sparkflash_kv::layout::{
    GLOBAL_ENTRY_BYTES, INDEXER_K_DATA_BYTES, INDEXER_K_DATA_OFFSET, INDEXER_K_DIM,
    INDEXER_K_SCALE_BYTES, INDEXER_K_SCALE_OFFSET, MAIN_KV_DATA_BYTES, MAIN_KV_SCALE_BYTES,
    SWA_SLOT_DATA_BYTES, SWA_SLOT_SCALE_BYTES, SWA_SLOTS_PER_TAIL, SWA_TAIL_BYTES,
    entries_per_page, source_slot, swa_slot_offset,
};
use sparkflash_kv::paged::{PagedGeometry, PagedKvCache};

const SEQ: u64 = 1;

fn tokens(count: usize) -> Vec<u32> {
    (0..count as u32).collect()
}

/// A cache with pages pages and seq_tokens tokens in one sequence.
fn served(pages: u32, seq_tokens: usize) -> PagedKvCache {
    let mut cache = PagedKvCache::new(PagedGeometry::new(pages, 1).expect("geometry"));
    cache.add_sequence(SEQ).expect("add");
    cache.append(SEQ, &tokens(seq_tokens)).expect("append");
    cache.check_invariants().expect("invariants");
    cache
}

/// The device form of a sequence's block table: one u32 physical page id per logical page.
fn table_array(cache: &PagedKvCache, sequence: u64) -> Vec<u32> {
    cache
        .table(sequence)
        .expect("table")
        .pages()
        .iter()
        .map(|page| u32::try_from(page.0).expect("page id fits u32"))
        .collect()
}

/// A pool of pages pages for one source layer, every entry filled with its own index byte.
fn indexed_pool(layer: usize, pages: u32) -> Vec<u8> {
    let slot = source_slot(layer).expect("a source layer");
    let entries = pages as usize * entries_per_page(slot);
    let mut pool = vec![0u8; entries * GLOBAL_ENTRY_BYTES];
    for entry in 0..entries {
        let byte = (entry % 251) as u8;
        pool[entry * GLOBAL_ENTRY_BYTES..(entry + 1) * GLOBAL_ENTRY_BYTES].fill(byte);
    }
    pool
}

/// Writes a real indexer K entry for token into the layer's source pool.
fn write_indexer(pool: &mut [u8], block_table: &[u32], token: u32) {
    let per_page = KV_PAGE_TOKENS / compress_ratio(20);
    let entry = block_table[token as usize / KV_PAGE_TOKENS] as usize * per_page
        + token as usize % KV_PAGE_TOKENS;
    let values: [f32; INDEXER_K_DIM] =
        std::array::from_fn(|channel| ((token as f32) * 0.01 + channel as f32 * 0.001).sin() * 1.5);
    let indexer = quantize_indexer_k(&values).expect("indexer k");
    let base = entry * GLOBAL_ENTRY_BYTES;
    pool[base + INDEXER_K_DATA_OFFSET..base + INDEXER_K_DATA_OFFSET + INDEXER_K_DATA_BYTES]
        .copy_from_slice(&indexer.data);
    pool[base + INDEXER_K_SCALE_OFFSET..base + INDEXER_K_SCALE_OFFSET + INDEXER_K_SCALE_BYTES]
        .copy_from_slice(&indexer.scales);
}

/// Deterministic query values, small and finite.
fn query() -> Vec<f32> {
    (0..512u32)
        .map(|index| ((index as f32) * 0.001).sin() * 0.1)
        .collect()
}

/// The layer's staged window rows, built from a synthetic tail ring.
fn window(view: &PagedKvView, layer: usize, position: usize) -> (Vec<u8>, Vec<u8>, usize) {
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
    let rows = view
        .window_rows(SEQ, layer, position, &tail)
        .expect("window rows");
    (rows.codes, rows.scales, rows.rows)
}

/// The staged bytes of each absolute position, as Path B produces them.
fn staged_absolute(
    view: &PagedKvView,
    layer: usize,
    position: usize,
    pool: &[u8],
) -> (Vec<u8>, Vec<u8>) {
    let staged = view
        .absolute_source_rows(SEQ, layer, position, pool)
        .expect("absolute rows");
    (staged.codes, staged.scales)
}

/// Page boundaries: the paged read of one absolute id is the staged row of that position, for every
/// position in a sequence that spans four pages, at both compression ratios.
#[test]
fn main_rows_across_page_boundaries_are_the_staged_rows_bit_for_bit() {
    let cache = served(8, 300);
    let view = PagedKvView::new(&cache);
    let block_table = table_array(&cache, SEQ);
    for layer in [2usize, 20] {
        let ratio = compress_ratio(layer);
        let pool = indexed_pool(layer, 8);
        let (codes, scales) = staged_absolute(&view, layer, 299, &pool);
        for position in 0..=299usize {
            let rows =
                read_main_rows(&pool, &block_table, ratio, &[position as u32]).expect("paged read");
            assert_eq!(rows.ids, vec![position as u32]);
            let codes_row =
                &codes[position * MAIN_KV_DATA_BYTES..(position + 1) * MAIN_KV_DATA_BYTES];
            let scales_row =
                &scales[position * MAIN_KV_SCALE_BYTES..(position + 1) * MAIN_KV_SCALE_BYTES];
            assert_eq!(rows.codes, codes_row, "layer {layer} position {position}");
            assert_eq!(rows.scales, scales_row, "layer {layer} position {position}");
        }
    }
}

/// The selected candidates across page boundaries plus the CPU attention twin: the staged rows and
/// the paged rows are the same bytes, so the released reference produces the same output.
#[test]
fn selected_candidates_across_a_page_boundary_attend_identically() {
    let cache = served(8, 300);
    let view = PagedKvView::new(&cache);
    let block_table = table_array(&cache, SEQ);
    let layer = 20usize;
    let ratio = compress_ratio(layer);
    let pool = indexed_pool(layer, 8);
    let selected = [0u32, 1, 62, 63, 64, 65, 127, 128, 255, 299];
    let staged = view
        .candidate_rows(SEQ, layer, &selected, &pool)
        .expect("staged candidates");
    let paged = read_main_rows(&pool, &block_table, ratio, &selected).expect("paged read");
    assert_eq!(paged.ids, selected.to_vec());
    assert_eq!(
        paged.codes, staged.codes,
        "the paged read reordered candidates"
    );
    assert_eq!(
        paged.scales, staged.scales,
        "the paged read lost a scale row"
    );

    let (window_codes, window_scales, rows) = window(&view, layer, 299);
    let q = query();
    let compact: Vec<u32> = (0..selected.len() as u32).collect();
    let staged_out = attention_with_sink_union(
        &q,
        &window_codes,
        &window_scales,
        rows,
        &staged.codes,
        &staged.scales,
        &compact,
        299,
        0.25,
    );
    let paged_out = attention_with_sink_union(
        &q,
        &window_codes,
        &window_scales,
        rows,
        &paged.codes,
        &paged.scales,
        &compact,
        299,
        0.25,
    );
    assert_eq!(
        staged_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        paged_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "the paged read changed the attention output"
    );
}

/// A partially filled tail page: only the held tokens are readable, the paged read stops at the
/// same causal position, and the HSI pool built from the same mapping matches the staged pool.
#[test]
fn a_partially_filled_tail_page_reads_like_the_staging_path() {
    let held = 137usize;
    let position = 133usize;
    let cache = served(8, held);
    let view = PagedKvView::new(&cache);
    let block_table = table_array(&cache, SEQ);
    let layer = 20usize;
    let ratio = compress_ratio(layer);
    let mut pool = indexed_pool(layer, 8);
    // The tail page (page 2) holds tokens 128..=136: nine entries, not a whole page.
    assert_eq!(held % KV_PAGE_TOKENS, 9);
    for token in 0..=position {
        write_indexer(&mut pool, &block_table, token as u32);
    }
    let ids: Vec<u32> = (128..=136).collect();
    let staged = view
        .candidate_rows(SEQ, layer, &ids, &pool)
        .expect("staged tail rows");
    let paged = read_main_rows(&pool, &block_table, ratio, &ids).expect("paged tail rows");
    assert_eq!(paged.ids, ids);
    assert_eq!(
        paged.codes, staged.codes,
        "the tail page row was read from the wrong entry"
    );
    assert_eq!(paged.scales, staged.scales);

    // The indexer K pool in absolute block order: the partially filled block 16 (tokens 128..135)
    // has two slots past the query, which are zero in the staged pool and in the device-read pool.
    let staged_keys = view
        .absolute_indexer_keys(SEQ, layer, position, &pool)
        .expect("staged indexer keys");
    let paged_keys =
        read_hsi_keys(&pool, &block_table, position as u32).expect("paged indexer keys");
    assert_eq!(paged_keys, staged_keys.keys, "the HSI pool differs");
    let row = |token: usize| &paged_keys[token * INDEX_HEAD_DIM..(token + 1) * INDEX_HEAD_DIM];
    let zero = vec![0u16; INDEX_HEAD_DIM];
    for token in 0..=position {
        assert_ne!(row(token), &zero[..], "valid token {token} was zeroed");
    }
    for token in (position + 1)..HSI_BLOCK_TOKENS * 2 {
        assert_eq!(
            row(token),
            &zero[..],
            "token {token} is past the query and must stay zero"
        );
    }
    let index_query: Vec<u16> = (0..INDEX_HEAD_DIM)
        .map(|channel| bf16_bits((channel as f32 * 0.002).cos()))
        .collect();
    let staged_scores = indexer_block_scores(&staged_keys.keys, &index_query, position as u32);
    let paged_scores = indexer_block_scores(&paged_keys, &index_query, position as u32);
    assert_eq!(
        staged_scores
            .iter()
            .map(|s| (s.0.to_bits(), s.1))
            .collect::<Vec<_>>(),
        paged_scores
            .iter()
            .map(|s| (s.0.to_bits(), s.1))
            .collect::<Vec<_>>(),
        "the indexer scores changed"
    );
    assert_eq!(staged_keys.reachable_blocks, 17);
}

/// An evicted block's page is reused: the reader follows the current table, not the id order, so a
/// stale table would read a new owner's bytes. The staged rows for the live owner and the paged
/// rows agree bit for bit.
#[test]
fn an_evicted_and_reused_block_is_read_through_the_current_table() {
    let mut cache = PagedKvCache::new(PagedGeometry::new(2, 1).expect("geometry"));
    cache.add_sequence(SEQ).expect("add A");
    cache.append(SEQ, &tokens(64)).expect("append A");
    let old_table = table_array(&cache, SEQ);
    let old_page = old_table[0] as usize;
    cache.free_sequence(SEQ).expect("free A");
    cache.add_sequence(SEQ).expect("add B");
    cache.append(SEQ, &tokens(64)).expect("append B");
    cache.check_invariants().expect("invariants");
    let new_table = table_array(&cache, SEQ);
    let new_page = new_table[0] as usize;
    // A 2-page pool: the current owner's page holds B's bytes; the previously evicted page (the
    // free list may or may not have handed the same page back) holds the old owner's bytes.
    let per_page = KV_PAGE_TOKENS / compress_ratio(20);
    let page_bytes = per_page * GLOBAL_ENTRY_BYTES;
    let mut pool = vec![0u8; 2 * page_bytes];
    for entry in 0..per_page {
        let base = old_page * page_bytes + entry * GLOBAL_ENTRY_BYTES;
        pool[base..base + GLOBAL_ENTRY_BYTES].fill(0x11);
        let base = new_page * page_bytes + entry * GLOBAL_ENTRY_BYTES;
        pool[base..base + GLOBAL_ENTRY_BYTES].fill(0x22);
    }
    let view = PagedKvView::new(&cache);
    let ids: Vec<u32> = (0..64).collect();
    let staged = view
        .candidate_rows(SEQ, 20, &ids, &pool)
        .expect("staged B rows");
    let paged = read_main_rows(&pool, &new_table, 1, &ids).expect("paged B rows");
    assert_eq!(paged.ids, ids);
    assert_eq!(
        paged.codes, staged.codes,
        "the reused page did not follow the table"
    );
    assert!(
        paged.codes.iter().all(|byte| *byte == 0x22),
        "the current table must read the current owner's bytes"
    );

    // The block table is authoritative, so a stale table reads the evicted page: a two-page pool
    // where page 1 belongs to the new owner and page 0 holds the evicted bytes. A kernel that
    // cached the table instead of reading the current one would publish the stale values.
    let mut split_pool = vec![0u8; 2 * page_bytes];
    split_pool[..page_bytes].fill(0x11);
    split_pool[page_bytes..].fill(0x22);
    let current = read_main_rows(&split_pool, &[1u32], 1, &ids).expect("current owner");
    let stale = read_main_rows(&split_pool, &[0u32], 1, &ids).expect("stale owner");
    assert!(current.codes.iter().all(|byte| *byte == 0x22));
    assert!(stale.codes.iter().all(|byte| *byte == 0x11));
    assert_ne!(
        current.codes, stale.codes,
        "the reader must not be able to ignore the block table"
    );
}

/// A block table with holes: a free slot is a mask, not physical page 0. The staged path cannot
/// produce a row for a hole, so the twin compares the held rows and asserts the hole is skipped
/// instead of read; a hole below the query is unattended, and an id outside the table fails closed.
#[test]
fn a_block_table_with_holes_is_masked_not_read() {
    let table = [0u32, BLOCK_TABLE_HOLE, BLOCK_TABLE_HOLE];
    let pool = indexed_pool(20, 2);
    let ids = [0u32, 1, 63, 64, 128];
    let rows = read_main_rows(&pool, &table, 1, &ids).expect("paged read");
    assert_eq!(rows.ids, vec![0, 1, 63], "hole ids must be skipped");

    // The same held rows are what the dense staging path produces for a sequence holding one page.
    let cache = served(1, 64);
    let view = PagedKvView::new(&cache);
    let staged = view
        .candidate_rows(SEQ, 20, &rows.ids, &pool)
        .expect("staged held rows");
    assert_eq!(rows.codes, staged.codes);
    assert_eq!(rows.scales, staged.scales);

    // A hole in the middle masks only itself: id 0 is in the hole and is not read, while id 64 is
    // in the mapped page 1 and is read normally. The hole is a mask, not physical page 0.
    let holed = [BLOCK_TABLE_HOLE, 1u32];
    let mid = read_main_rows(&pool, &holed, 1, &[0u32, 64]).expect("holed read");
    assert_eq!(mid.ids, vec![64]);
    // Past the table, the kernel has no mapping: the read is refused, not wrapped.
    assert!(read_main_rows(&pool, &table, 1, &[192]).is_err());

    // The HSI pool obeys the same masking: positions in a hole are zero, never a read of page 0.
    let hole_table = [BLOCK_TABLE_HOLE, 0u32];
    let keys = read_hsi_keys(&pool, &hole_table, 63).expect("holed indexer keys");
    assert!(
        keys.iter().all(|key| *key == 0),
        "a hole slot was published"
    );
    assert_eq!(keys.len(), HSI_BLOCKS * HSI_BLOCK_TOKENS * INDEX_HEAD_DIM);
}

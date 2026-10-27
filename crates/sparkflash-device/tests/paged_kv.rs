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

//! The paged device-attention seam (task-21, S3): addressing through a block table, window
//! rotation gathers, candidate staging, and a CPU twin that runs the released attention reference
//! over the staged operands and over the equivalent contiguous operands and compares bit for bit.

use sparkflash_device::paged_kv::PagedKvView;
use sparkflash_kernels::m4_quant::bf16_bits;
use sparkflash_kernels::mla_ref::{
    HSI_BLOCK_TOKENS, HSI_BLOCKS, INDEX_HEAD_DIM, attention_with_sink_union, indexer_block_scores,
};
use sparkflash_kv::config::{KV_PAGE_TOKENS, compress_ratio};
use sparkflash_kv::fp4::{dequantize_indexer_k, quantize_indexer_k};
use sparkflash_kv::layout::{
    GLOBAL_ENTRY_BYTES, INDEXER_K_DATA_OFFSET, INDEXER_K_DIM, INDEXER_K_ENTRY_BYTES,
    MAIN_KV_DATA_BYTES, MAIN_KV_ENTRY_BYTES, MAIN_KV_SCALE_BYTES, MAIN_KV_SCALE_OFFSET, SWA_LAYERS,
    SWA_SLOT_DATA_BYTES, SWA_SLOT_SCALE_BYTES, SWA_SLOTS_PER_TAIL, SWA_TAIL_BYTES,
    entries_per_page, source_slot, swa_slot_offset,
};
use sparkflash_kv::paged::{PagedGeometry, PagedKvCache, PagingError};

const SEQ: u64 = 1;

fn tokens(count: usize) -> Vec<u32> {
    (0..count as u32).collect()
}

/// A cache with `pages` pages and `seq_tokens` tokens in one sequence.
fn served(pages: u32, seq_tokens: usize) -> PagedKvCache {
    let mut cache = PagedKvCache::new(PagedGeometry::new(pages, 1).expect("geometry"));
    cache.add_sequence(SEQ).expect("add");
    cache.append(SEQ, &tokens(seq_tokens)).expect("append");
    cache.check_invariants().expect("invariants");
    cache
}

/// A pool of `pages` pages for one source layer, every entry filled with its own pool index.
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

#[test]
fn every_logical_token_maps_into_its_blocks_page() {
    let cache = served(8, 200);
    let view = PagedKvView::new(&cache);
    let table = cache.table(SEQ).expect("table").clone();
    // Layer 2 has compression ratio 2; layer 20 has ratio 1: both mappings are checked, because the
    // ratio is the part of the format-2 layout a block table must not flatten.
    for layer in [2usize, 20] {
        let slot = source_slot(layer).expect("source layer");
        let ratio = compress_ratio(layer);
        for logical in 0..200usize {
            let entry = view.source_entry(SEQ, layer, logical).expect("entry");
            let page = table.page_for(logical / KV_PAGE_TOKENS).expect("page");
            assert_eq!(entry.page, page.0, "layer {layer} logical {logical}");
            assert_eq!(
                entry.entry_in_page,
                (logical % KV_PAGE_TOKENS) / ratio,
                "layer {layer} logical {logical}"
            );
            let expected = page.0 as usize * entries_per_page(slot) + entry.entry_in_page;
            assert_eq!(entry.pool_index, expected);
            assert_eq!(entry.byte_offset, expected * GLOBAL_ENTRY_BYTES);
        }
    }
}

#[test]
fn a_token_run_never_crosses_a_page_boundary_in_one_entry() {
    // 130 tokens => 3 blocks; the two ratio-2 tokens of each compressed entry are in the same page.
    let cache = served(8, 130);
    let view = PagedKvView::new(&cache);
    for logical in [0usize, 1, 62, 63, 64, 65, 126, 127, 128, 129] {
        let entry = view.source_entry(SEQ, 2, logical).expect("entry");
        assert_eq!(entry.entry_in_page, (logical % 64) / 2);
        assert!(entry.entry_in_page < entries_per_page(0));
    }
}

#[test]
fn candidate_rows_are_the_main_part_of_the_mapped_entry_in_order() {
    let cache = served(8, 200);
    let view = PagedKvView::new(&cache);
    let pool = indexed_pool(20, 8);
    let selected = [0u32, 3, 64, 129, 199];
    let operands = view
        .candidate_rows(SEQ, 20, &selected, &pool)
        .expect("rows");
    assert_eq!(operands.len(), selected.len());
    assert_eq!(operands.codes.len(), selected.len() * MAIN_KV_DATA_BYTES);
    assert_eq!(operands.scales.len(), selected.len() * MAIN_KV_SCALE_BYTES);
    for (index, logical) in selected.iter().enumerate() {
        let entry = view
            .source_entry(SEQ, 20, *logical as usize)
            .expect("entry");
        let expected = (entry.pool_index % 251) as u8;
        let codes = &operands.codes[index * MAIN_KV_DATA_BYTES..(index + 1) * MAIN_KV_DATA_BYTES];
        assert!(
            codes.iter().all(|byte| *byte == expected),
            "candidate {logical}"
        );
        let scales =
            &operands.scales[index * MAIN_KV_SCALE_BYTES..(index + 1) * MAIN_KV_SCALE_BYTES];
        assert!(
            scales.iter().all(|byte| *byte == expected),
            "candidate {logical}"
        );
    }
}

#[test]
fn the_window_gather_is_oldest_first_through_the_ring() {
    let cache = served(8, 300);
    let view = PagedKvView::new(&cache);
    let mut tail = vec![0u8; SWA_TAIL_BYTES];
    for slot in 0..SWA_SLOTS_PER_TAIL {
        let offset = swa_slot_offset(3, slot);
        tail[offset..offset + SWA_SLOT_DATA_BYTES].fill(slot as u8);
        tail[offset + SWA_SLOT_DATA_BYTES..offset + SWA_SLOT_DATA_BYTES + SWA_SLOT_SCALE_BYTES]
            .fill((slot as u8).wrapping_add(100));
    }
    let window = view.window_rows(SEQ, 3, 300, &tail).expect("window");
    assert_eq!(window.rows, SWA_SLOTS_PER_TAIL);
    for row in 0..window.rows {
        let logical = 300 - window.rows + row;
        let slot = logical % SWA_SLOTS_PER_TAIL;
        let codes = &window.codes[row * SWA_SLOT_DATA_BYTES..(row + 1) * SWA_SLOT_DATA_BYTES];
        assert!(codes.iter().all(|byte| *byte == slot as u8), "row {row}");
        let scales = &window.scales[row * SWA_SLOT_SCALE_BYTES..(row + 1) * SWA_SLOT_SCALE_BYTES];
        assert!(
            scales
                .iter()
                .all(|byte| *byte == (slot as u8).wrapping_add(100))
        );
    }
}

#[test]
fn an_incomplete_window_has_exactly_the_tokens_seen() {
    let cache = served(4, 5);
    let view = PagedKvView::new(&cache);
    let tail = vec![0u8; SWA_TAIL_BYTES];
    let window = view.window_rows(SEQ, 0, 5, &tail).expect("window");
    assert_eq!(window.rows, 5);
    assert_eq!(window.codes.len(), 5 * SWA_SLOT_DATA_BYTES);
    assert_eq!(window.scales.len(), 5 * SWA_SLOT_SCALE_BYTES);
    let empty = view.window_rows(SEQ, 0, 0, &tail).expect("window at 0");
    assert_eq!(empty.rows, 0);
    assert!(empty.codes.is_empty());
}

#[test]
fn the_mapping_refuses_layers_and_positions_it_cannot_resolve() {
    let cache = served(4, 70);
    let view = PagedKvView::new(&cache);
    let pool = indexed_pool(20, 4);
    // Layer 0 is SWA-only: it owns no compressed global KV.
    assert!(matches!(
        view.source_entry(SEQ, 0, 0),
        Err(PagingError::Invariant(_))
    ));
    // A token the sequence does not hold.
    assert!(matches!(
        view.source_entry(SEQ, 20, 70),
        Err(PagingError::OutOfRange { .. })
    ));
    assert!(matches!(
        view.candidate_rows(SEQ, 20, &[70], &pool),
        Err(PagingError::OutOfRange { .. })
    ));
    // A sequence that is not in the pool.
    assert!(matches!(
        view.window_rows(99, 0, 0, &[0u8; SWA_TAIL_BYTES]),
        Err(PagingError::UnknownSequence(99))
    ));
    // A layer outside the SWA backbone.
    assert!(matches!(
        view.window_rows(SEQ, SWA_LAYERS, 1, &[0u8; SWA_TAIL_BYTES]),
        Err(PagingError::Invariant(_))
    ));
    // A short tail buffer.
    assert!(matches!(
        view.window_rows(SEQ, 0, 1, &[0u8; 8]),
        Err(PagingError::Invariant(_))
    ));
}

/// Eight bytes per query coefficient, deterministic and small enough to stay out of the E4M3
/// specials when used as a scale code.
fn query() -> Vec<f32> {
    (0..512u32)
        .map(|index| ((index as f32) * 0.001).sin() * 0.1)
        .collect()
}

/// The CPU twin: the staged operands (gathered from the paged pool, compact ids) and the equivalent
/// contiguous operands (logical ids over a flat row array) must give the same f32 attention output
/// bit for bit. The arithmetic is the released `mla_ref` reference on both sides; only the
/// addressing differs, which is exactly what the paging seam changes.
#[test]
fn the_cpu_twin_matches_the_contiguous_operands_bit_for_bit() {
    let cache = served(8, 300);
    let view = PagedKvView::new(&cache);
    let layer = 20usize;
    let ratio = compress_ratio(layer);
    assert_eq!(ratio, 1, "the twin uses the ratio-1 source layer");
    let pool_pages = 8u32;
    let pool = indexed_pool(layer, pool_pages);
    let selected = [0u32, 1, 5, 64, 127, 255, 299];

    // Stage the candidates through the block table as a proved-causal compact list.
    let staged = view
        .candidate_rows(SEQ, layer, &selected, &pool)
        .expect("staged");
    let compact = view
        .compact_candidates(SEQ, &selected, 299)
        .expect("every candidate is causal");
    assert_eq!(
        compact.ids(),
        &(0..selected.len() as u32).collect::<Vec<_>>()
    );

    // The contiguous operand arrays: row p is the main part of pool entry p for ratio 1.
    let contiguous_rows = 300usize;
    let mut contiguous_codes = Vec::with_capacity(contiguous_rows * MAIN_KV_DATA_BYTES);
    let mut contiguous_scales = Vec::with_capacity(contiguous_rows * MAIN_KV_SCALE_BYTES);
    for logical in 0..contiguous_rows {
        let start = logical * GLOBAL_ENTRY_BYTES;
        contiguous_codes.extend_from_slice(&pool[start..start + MAIN_KV_DATA_BYTES]);
        contiguous_scales
            .extend_from_slice(&pool[start + MAIN_KV_SCALE_OFFSET..start + MAIN_KV_ENTRY_BYTES]);
    }
    let expected_codes: Vec<u8> = selected
        .iter()
        .flat_map(|logical| {
            let start = *logical as usize * GLOBAL_ENTRY_BYTES;
            pool[start..start + MAIN_KV_DATA_BYTES].to_vec()
        })
        .collect();
    assert_eq!(
        staged.codes, expected_codes,
        "the staging copy lost or reordered rows"
    );

    // The window is the same bytes on both sides: gathered oldest first from the ring.
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
        .window_rows(SEQ, layer, 300, &tail)
        .expect("window rows");
    assert_eq!(window.rows, SWA_SLOTS_PER_TAIL);
    // The reference needs the window as one flat array per code type; the gather already produces
    // that order, so the contiguous side can use the same window.
    let window_codes = window.codes.clone();
    let window_scales = window.scales.clone();

    let q = query();
    let sink = 0.25f32;
    let staged_out = attention_with_sink_union(
        &q,
        &window_codes,
        &window_scales,
        window.rows,
        &staged.codes,
        &staged.scales,
        compact.ids(),
        compact.position(),
        sink,
    );
    let contiguous_out = attention_with_sink_union(
        &q,
        &window_codes,
        &window_scales,
        window.rows,
        &contiguous_codes,
        &contiguous_scales,
        &selected,
        299,
        sink,
    );
    assert_eq!(
        staged_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        contiguous_out
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "the paged staging changed the attention result"
    );
}

/// A pool of `rows` ratio-1 entries where every row is zero except `future`, whose codes are the
/// largest E2M1 magnitude (0x77) and whose E4M3 scale is 1.0 (0x38): if a mask lets it through it
/// dominates the softmax, so a masked and an unmasked result cannot be confused.
fn masked_pool(rows: usize, future: usize) -> Vec<u8> {
    let mut pool = vec![0u8; rows * GLOBAL_ENTRY_BYTES];
    let start = future * GLOBAL_ENTRY_BYTES;
    pool[start..start + MAIN_KV_DATA_BYTES].fill(0x77);
    pool[start + MAIN_KV_SCALE_OFFSET..start + MAIN_KV_ENTRY_BYTES].fill(0x38);
    pool
}

/// The contiguously laid out operands for a pool of `rows` ratio-1 entries.
fn split_pool(pool: &[u8], rows: usize) -> (Vec<u8>, Vec<u8>) {
    let mut codes = Vec::with_capacity(rows * MAIN_KV_DATA_BYTES);
    let mut scales = Vec::with_capacity(rows * MAIN_KV_SCALE_BYTES);
    for row in 0..rows {
        let start = row * GLOBAL_ENTRY_BYTES;
        codes.extend_from_slice(&pool[start..start + MAIN_KV_DATA_BYTES]);
        scales.extend_from_slice(&pool[start + MAIN_KV_SCALE_OFFSET..start + MAIN_KV_ENTRY_BYTES]);
    }
    (codes, scales)
}

/// Requirement: the mask must be non-vacuous. A candidate past the query position is rejected by the
/// full-length reference mask (attending it changes nothing), it is refused by the seam, and a
/// compaction that renumbers it into range would attend it - which is exactly why the seam refuses
/// instead of letting the kernel's now-vacuous mask decide.
#[test]
fn a_future_candidate_is_masked_by_the_reference_and_refused_by_the_seam() {
    let cache = served(4, 64);
    let view = PagedKvView::new(&cache);
    let layer = 20usize;
    let rows = 64usize;
    let position = 40u32;
    let future = 41usize;
    let causal = [0u32, 1, 5];
    let pool = masked_pool(rows, future);
    let (codes, scales) = split_pool(&pool, rows);
    let q = query();
    let sink = 0.0f32;
    let window_rows = position as usize;
    let window_codes = vec![0u8; window_rows * SWA_SLOT_DATA_BYTES];
    let window_scales = vec![0u8; window_rows * SWA_SLOT_SCALE_BYTES];

    // The full-length reference with and without the future candidate: the mask must reject it.
    let mut with_future = causal.to_vec();
    with_future.push(future as u32);
    let masked = attention_with_sink_union(
        &q,
        &window_codes,
        &window_scales,
        window_rows,
        &codes,
        &scales,
        &with_future,
        position,
        sink,
    );
    let without_future = attention_with_sink_union(
        &q,
        &window_codes,
        &window_scales,
        window_rows,
        &codes,
        &scales,
        &causal,
        position,
        sink,
    );
    assert_eq!(
        masked
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        without_future
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "the full-length mask did not reject the future candidate"
    );

    // The seam refuses to compact the same candidate list.
    assert_eq!(
        view.compact_candidates(SEQ, &with_future, position),
        Err(PagingError::NonCausalCandidate {
            sequence: SEQ,
            candidate: future as u32,
            position,
        })
    );
    // ...and the causal list is accepted, with the compact ids the staged call passes.
    let compact = view
        .compact_candidates(SEQ, &causal, position)
        .expect("causal");
    assert_eq!(compact.ids(), &[0, 1, 2]);
    assert_eq!(compact.position(), 2);

    // The hazard: if the same list were compacted anyway, the future row becomes attendable.
    let staged = view
        .candidate_rows(SEQ, layer, &with_future, &pool)
        .expect("staged");
    let compacted = attention_with_sink_union(
        &q,
        &window_codes,
        &window_scales,
        window_rows,
        &staged.codes,
        &staged.scales,
        &(0..with_future.len() as u32).collect::<Vec<_>>(),
        (with_future.len() - 1) as u32,
        sink,
    );
    assert_ne!(
        compacted
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        masked
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "compacting a non-causal candidate must change the attention output"
    );
}

/// The indexer keys the test quantises for one token: small integers, which FP4 E2M1 represents
/// exactly, so the staged bf16 values and the reference agree bit for bit instead of by tolerance.
fn indexer_values(token: usize) -> [f32; INDEXER_K_DIM] {
    std::array::from_fn(|channel| (((token * 13 + channel) % 9) as f32) - 4.0)
}

/// Writes one quantised indexer K entry into the pool at the mapped offset.
fn write_indexer_entry(pool: &mut [u8], offset: usize, values: &[f32; INDEXER_K_DIM]) {
    let entry = quantize_indexer_k(values).expect("indexer k");
    let mut encoded = [0u8; INDEXER_K_ENTRY_BYTES];
    entry.to_bytes(&mut encoded).expect("encode");
    pool[offset + INDEXER_K_DATA_OFFSET..offset + INDEXER_K_DATA_OFFSET + INDEXER_K_ENTRY_BYTES]
        .copy_from_slice(&encoded);
}

/// The reference HSI pool in absolute block order, built from the synthetic values directly (not
/// through the paged pool), so the comparison exercises the seam rather than sharing its code path.
fn reference_indexer_keys(position: usize) -> Vec<u16> {
    let slots = HSI_BLOCKS * HSI_BLOCK_TOKENS;
    let mut keys = vec![0u16; slots * INDEX_HEAD_DIM];
    for token in 0..=position {
        let values = indexer_values(token);
        let quantised = quantize_indexer_k(&values).expect("indexer k");
        let mut decoded = [0.0f32; INDEXER_K_DIM];
        dequantize_indexer_k(&quantised, &mut decoded).expect("dequantise");
        for channel in 0..INDEXER_K_DIM {
            keys[token * INDEX_HEAD_DIM + channel] = bf16_bits(decoded[channel]);
        }
    }
    keys
}

#[test]
fn absolute_staging_is_the_kernel_view_and_the_absolute_mask_rejects_a_future_id() {
    let cache = served(8, 64);
    let view = PagedKvView::new(&cache);
    let layer = 20usize;
    let position = 40usize;
    let future = 41usize;
    let pool = masked_pool(64, future);
    let staged = view
        .absolute_source_rows(SEQ, layer, position, &pool)
        .expect("staged");
    assert_eq!(staged.rows, 41);
    assert_eq!(staged.codes.len(), 41 * MAIN_KV_DATA_BYTES);
    assert_eq!(staged.scales.len(), 41 * MAIN_KV_SCALE_BYTES);
    // Row p is position p's row: the huge future row is not in the operands at all here, because
    // only 0..=position is staged.
    for row in 0..=position {
        let start = row * MAIN_KV_DATA_BYTES;
        assert_eq!(
            &staged.codes[start..start + MAIN_KV_DATA_BYTES],
            &pool[row * GLOBAL_ENTRY_BYTES..row * GLOBAL_ENTRY_BYTES + MAIN_KV_DATA_BYTES]
        );
    }

    let q = query();
    let sink = 0.0f32;
    let (window_codes, window_scales) = (Vec::new(), Vec::new());
    let causal = [0u32, 1, 5];
    let with_future = [0u32, 1, 5, future as u32];
    let masked = attention_with_sink_union(
        &q,
        &window_codes,
        &window_scales,
        0,
        &staged.codes,
        &staged.scales,
        &with_future,
        position as u32,
        sink,
    );
    let without_future = attention_with_sink_union(
        &q,
        &window_codes,
        &window_scales,
        0,
        &staged.codes,
        &staged.scales,
        &causal,
        position as u32,
        sink,
    );
    assert_eq!(
        masked
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        without_future
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "the absolute mask did not reject a candidate past the query position"
    );

    // Move the query one token: now 41 is attendable and its dominant row changes the output, so
    // the mask above is doing work rather than being vacuous.
    let later = view
        .absolute_source_rows(SEQ, layer, position + 1, &pool)
        .expect("staged at 41");
    let attends = attention_with_sink_union(
        &q,
        &window_codes,
        &window_scales,
        0,
        &later.codes,
        &later.scales,
        &with_future,
        (position + 1) as u32,
        sink,
    );
    assert_ne!(
        attends
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        masked
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "the dominant future row was attended at the next position, so the mask was live"
    );

    // Refusals: a position the sequence does not hold.
    assert_eq!(
        view.absolute_source_rows(SEQ, layer, 64, &pool),
        Err(PagingError::OutOfRange {
            sequence: SEQ,
            logical: 64,
            held: 64,
        })
    );
}

#[test]
fn expanded_absolute_rows_reproduce_the_contiguous_kernel_view() {
    // Layer 2 has compression ratio 2: the kernel still sees one row per token, so both rows of a
    // group carry the same compressed entry. The independent reference below rebuilds the expected
    // row from the layout formula, not through source_entry.
    let cache = served(8, 64);
    let view = PagedKvView::new(&cache);
    let layer = 2usize;
    let position = 40usize;
    let pool = indexed_pool(layer, 8);
    let staged = view
        .absolute_source_rows(SEQ, layer, position, &pool)
        .expect("staged");
    let ratio = compress_ratio(layer);
    let entries = entries_per_page(source_slot(layer).expect("source layer"));
    let mut contiguous_codes = Vec::new();
    let mut contiguous_scales = Vec::new();
    for row in 0..=position {
        let entry_in_page = (row % KV_PAGE_TOKENS) / ratio;
        let page = row / KV_PAGE_TOKENS;
        let pool_index = page * entries + entry_in_page;
        let value = (pool_index % 251) as u8;
        assert_eq!(
            &staged.codes[row * MAIN_KV_DATA_BYTES..(row + 1) * MAIN_KV_DATA_BYTES],
            &[value; MAIN_KV_DATA_BYTES][..],
            "row {row} is not the mapped entry"
        );
        assert_eq!(
            &staged.scales[row * MAIN_KV_SCALE_BYTES..(row + 1) * MAIN_KV_SCALE_BYTES],
            &[value; MAIN_KV_SCALE_BYTES][..],
            "row {row} scales are not the mapped entry"
        );
        contiguous_codes.extend_from_slice(&[value; MAIN_KV_DATA_BYTES]);
        contiguous_scales.extend_from_slice(&[value; MAIN_KV_SCALE_BYTES]);
    }
    for pair in 0..(position / 2) {
        let even = pair * 2;
        assert_eq!(
            &staged.codes[even * MAIN_KV_DATA_BYTES..(even + 1) * MAIN_KV_DATA_BYTES],
            &staged.codes[(even + 1) * MAIN_KV_DATA_BYTES..(even + 2) * MAIN_KV_DATA_BYTES],
            "a ratio-2 group must publish the same entry to both token rows"
        );
    }
    assert_eq!(staged.codes, contiguous_codes);
    assert_eq!(staged.scales, contiguous_scales);

    // Absolute ids over the staged view and over the contiguous view give the same selection.
    let selected = [0u32, 4, 7, 9, 20];
    let q = query();
    let sink = 0.25f32;
    let (window_codes, window_scales) = (Vec::new(), Vec::new());
    let staged_out = attention_with_sink_union(
        &q,
        &window_codes,
        &window_scales,
        0,
        &staged.codes,
        &staged.scales,
        &selected,
        position as u32,
        sink,
    );
    let contiguous_out = attention_with_sink_union(
        &q,
        &window_codes,
        &window_scales,
        0,
        &contiguous_codes,
        &contiguous_scales,
        &selected,
        position as u32,
        sink,
    );
    assert_eq!(
        staged_out
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        contiguous_out
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "absolute ids over the staged view changed the attention result"
    );
}

#[test]
fn absolute_indexer_keys_match_the_reference_block_scores_bit_for_bit() {
    let cache = served(8, 200);
    let view = PagedKvView::new(&cache);
    let layer = 20usize;
    let position = 40usize;
    let mut pool = vec![
        0u8;
        8 * entries_per_page(source_slot(layer).expect("source layer"))
            * GLOBAL_ENTRY_BYTES
    ];
    for token in 0..64usize {
        let entry = view.source_entry(SEQ, layer, token).expect("entry");
        write_indexer_entry(&mut pool, entry.byte_offset, &indexer_values(token));
    }
    // A future token that would dominate the block if the kernel leaked it.
    let future = 41usize;
    let mut future_values = [0.0f32; INDEXER_K_DIM];
    future_values.fill(384.0);
    let future_entry = view.source_entry(SEQ, layer, future).expect("entry");
    write_indexer_entry(&mut pool, future_entry.byte_offset, &future_values);

    let staged = view
        .absolute_indexer_keys(SEQ, layer, position, &pool)
        .expect("keys");
    assert_eq!(staged.position, position as u32);
    assert_eq!(staged.slots(), HSI_BLOCKS * HSI_BLOCK_TOKENS);
    assert_eq!(staged.reachable_blocks, 6);
    let expected = reference_indexer_keys(position);
    assert_eq!(
        staged.keys, expected,
        "the staged HSI pool is not absolute-block order"
    );
    // Slots past the query are zero.
    for token in (position + 1)..(HSI_BLOCK_TOKENS * 8) {
        let row = &staged.keys[token * INDEX_HEAD_DIM..(token + 1) * INDEX_HEAD_DIM];
        assert!(
            row.iter().all(|bits| *bits == 0),
            "future slot {token} is not zero"
        );
    }

    let query: Vec<u16> = (0..INDEX_HEAD_DIM)
        .map(|channel| bf16_bits((channel as f32 * 0.01).sin() * 0.5))
        .collect();
    let staged_scores = indexer_block_scores(&staged.keys, &query, position as u32);
    let reference_scores = indexer_block_scores(&expected, &query, position as u32);
    assert_eq!(staged_scores.len(), HSI_BLOCKS);
    for (block, (staged_block, reference_block)) in staged_scores
        .iter()
        .zip(reference_scores.iter())
        .enumerate()
    {
        assert_eq!(
            staged_block.0.to_bits(),
            reference_block.0.to_bits(),
            "block {block} score"
        );
        assert_eq!(
            staged_block.1, reference_block.1,
            "block {block} best token"
        );
        if block * HSI_BLOCK_TOKENS > position {
            assert_eq!(
                *staged_block,
                (-f32::MAX, 0),
                "block {block} past the query"
            );
        }
    }
    // The query's own block is the only one that sees a partial scan, and its best token is the
    // query itself: the dominant future token one past it never wins.
    let own = &staged_scores[position / HSI_BLOCK_TOKENS];
    assert_eq!(own.1, position as u32);
    // Stage at position 41: the same dominant key is now attendable and wins its block, so the
    // masked result above is a live mask, not an artefact of the data.
    let later = view
        .absolute_indexer_keys(SEQ, layer, position + 1, &pool)
        .expect("keys at 41");
    let later_scores = indexer_block_scores(&later.keys, &query, (position + 1) as u32);
    assert_eq!(later_scores[position / HSI_BLOCK_TOKENS].1, future as u32);

    // Refusals: a position the sequence does not hold.
    assert_eq!(
        view.absolute_indexer_keys(SEQ, layer, 200, &pool),
        Err(PagingError::OutOfRange {
            sequence: SEQ,
            logical: 200,
            held: 200,
        })
    );
}

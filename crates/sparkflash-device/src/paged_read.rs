//! The paged-read attention kernel's host reference (task-49, design only: no kernel, no pin, no
//! launch, no KernelId).
//!
//! Today's Path B stages dense absolute-position rows for the unchanged 19-word entry, which is
//! correct but O(context) in staging. The named follow-on is a kernel that reads the paged pool
//! directly: one block-table indirection, the absolute position derived from the page index and the
//! offset, and the causal / HSI masks applied per page. This module is the CPU definition of that
//! read path, written so its arithmetic is the kernel's addressing and nothing else:
//!
//! * [paged_entry] maps an absolute compressed position id to the entry index inside the layer's
//!   source array, through the sequence's block table. A hole (free table slot) maps to None, which
//!   the kernel treats as unattended, exactly as the released union mask treats a future id.
//! * [read_main_rows] returns the main-KV bytes of the attended ids in the kernel's row order. It is
//!   byte-for-byte what [crate::paged_kv::PagedKvView::candidate_rows] stages, without the staging
//!   copy: that equality is what crates/sparkflash-device/tests/paged_read.rs asserts across page
//!   boundaries, a partial tail page, an evicted and reused block, and a block table with holes.
//! * [read_hsi_keys] returns the HSI index-key pool in absolute block order by reading the indexer
//!   K entries through the same table and dequantising FP4 E2M1 + E8M0 on the fly. It is value for
//!   value what [crate::paged_kv::PagedKvView::absolute_indexer_keys] stages (zero past the query
//!   and in holes), so the indexer's block scores are unchanged.
//!
//! What this module is not: a device launch, a measured throughput claim, or a frozen ABI. The
//! design note docs/build/paged-read-attention-kernel-design.md records the argument set such a
//! kernel would need and the cost of pinning it.

use sparkflash_kernels::m4_quant::bf16_bits;
use sparkflash_kernels::mla_ref::{HSI_BLOCK_TOKENS, HSI_BLOCKS, INDEX_HEAD_DIM};
use sparkflash_kv::config::KV_PAGE_TOKENS;
use sparkflash_kv::fp4::{IndexerKEntry, dequantize_indexer_k};
use sparkflash_kv::layout::{
    GLOBAL_ENTRY_BYTES, INDEXER_K_DATA_BYTES, INDEXER_K_DATA_OFFSET, INDEXER_K_SCALE_BYTES,
    INDEXER_K_SCALE_OFFSET, MAIN_KV_DATA_BYTES, MAIN_KV_DATA_OFFSET, MAIN_KV_SCALE_BYTES,
    MAIN_KV_SCALE_OFFSET,
};

/// A block-table slot that is not mapped. The kernel must treat it as unattended rather than read
/// physical page 0xFFFFFFFF.
pub const BLOCK_TABLE_HOLE: u32 = u32::MAX;

/// Entries one 64-token page holds for a source layer of the given compression ratio.
///
/// # Errors
/// A ratio that does not divide the page: the store only realises 1 and 2.
pub fn entries_per_page(ratio: usize) -> Result<usize, String> {
    if ratio == 0 || !KV_PAGE_TOKENS.is_multiple_of(ratio) {
        return Err(format!(
            "compression ratio {ratio} does not divide the {KV_PAGE_TOKENS}-token page"
        ));
    }
    Ok(KV_PAGE_TOKENS / ratio)
}

/// The entry index inside the layer's source array for one absolute compressed position id.
///
/// This is the kernel's address arithmetic, in order: the logical page is the token id divided by
/// the page size, the block table maps it to a physical page, and the entry inside the page is the
/// token offset divided by the compression ratio. The result is the pool entry index, so the byte
/// address is entry * GLOBAL_ENTRY_BYTES + field offset.
///
/// # Errors
/// A block-table slot beyond the table, or an entry index that leaves the pool, is refused here
/// rather than read as a wrapped address.
pub fn paged_entry(
    pool_len: usize,
    block_table: &[u32],
    ratio: usize,
    id: u32,
) -> Result<Option<usize>, String> {
    let per_page = entries_per_page(ratio)?;
    let logical_page = id as usize / KV_PAGE_TOKENS;
    let page = *block_table.get(logical_page).ok_or_else(|| {
        format!(
            "block table has {} pages, the id {id} needs logical page {logical_page}",
            block_table.len()
        )
    })?;
    if page == BLOCK_TABLE_HOLE {
        return Ok(None);
    }
    let entry_in_page = (id as usize % KV_PAGE_TOKENS) / ratio;
    let entry = page as usize * per_page + entry_in_page;
    if entry >= pool_len {
        return Err(format!(
            "the mapping reaches entry {entry}, the pool holds {pool_len}"
        ));
    }
    Ok(Some(entry))
}

/// The main-KV rows the paged-read kernel would read for these ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedMainRows {
    /// The attended ids, in the kernel's row order (holes are skipped, as the mask skips them).
    pub ids: Vec<u32>,
    /// Packed FP4 codes, one MAIN_KV_DATA_BYTES row per attended id.
    pub codes: Vec<u8>,
    /// E4M3 scales, one MAIN_KV_SCALE_BYTES row per attended id.
    pub scales: Vec<u8>,
}

impl PagedMainRows {
    /// Attended rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether no id was attendable.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Reads the main-KV bytes of [ids] through [block_table] from the layer's source array.
///
/// The returned bytes are the same slices the dense staging path copies out of the same pool; the
/// only difference is that no staging buffer exists. Holes are skipped and reported by their
/// absence from [PagedMainRows::ids].
///
/// # Errors
/// A malformed table, a mapping outside the pool, or a pool shorter than the table reaches.
pub fn read_main_rows(
    pool: &[u8],
    block_table: &[u32],
    ratio: usize,
    ids: &[u32],
) -> Result<PagedMainRows, String> {
    let entries = pool.len() / GLOBAL_ENTRY_BYTES;
    let mut rows = PagedMainRows {
        ids: Vec::with_capacity(ids.len()),
        codes: Vec::with_capacity(ids.len() * MAIN_KV_DATA_BYTES),
        scales: Vec::with_capacity(ids.len() * MAIN_KV_SCALE_BYTES),
    };
    for id in ids {
        let Some(entry) = paged_entry(entries, block_table, ratio, *id)? else {
            continue;
        };
        let base = entry * GLOBAL_ENTRY_BYTES;
        let codes_start = base + MAIN_KV_DATA_OFFSET;
        let scales_start = base + MAIN_KV_SCALE_OFFSET;
        rows.codes
            .extend_from_slice(&pool[codes_start..codes_start + MAIN_KV_DATA_BYTES]);
        rows.scales
            .extend_from_slice(&pool[scales_start..scales_start + MAIN_KV_SCALE_BYTES]);
        rows.ids.push(*id);
    }
    Ok(rows)
}

/// The HSI index-key pool the paged-read indexer would build on the device.
///
/// The kernel scores HSI block b from eight absolute token positions, b * 8 + slot. For each valid
/// position it maps the token through the block table, reads the layer's indexer K entry, and
/// dequantises FP4 E2M1 with its four E8M0 scales to bf16, which is the same value
/// [crate::paged_kv::PagedKvView::absolute_indexer_keys] stages from the host. Positions past the
/// query and block-table holes are zero, which the kernel's own mask makes unreachable.
///
/// # Errors
/// A malformed table, a mapping outside the pool, or an indexer K entry that does not fit.
pub fn read_hsi_keys(pool: &[u8], block_table: &[u32], position: u32) -> Result<Vec<u16>, String> {
    let entries = pool.len() / GLOBAL_ENTRY_BYTES;
    let mut keys = vec![0u16; HSI_BLOCKS * HSI_BLOCK_TOKENS * INDEX_HEAD_DIM];
    // The indexer K source is layer 20, whose compression ratio is 1, so one token is one entry.
    let ratio = 1usize;
    for block in 0..HSI_BLOCKS {
        for slot in 0..HSI_BLOCK_TOKENS {
            let token = block * HSI_BLOCK_TOKENS + slot;
            if token as u64 > u64::from(position) {
                break;
            }
            let entry = match paged_entry(entries, block_table, ratio, token as u32) {
                Ok(Some(entry)) => entry,
                Ok(None) => continue,
                Err(error) => return Err(error),
            };
            let base = entry * GLOBAL_ENTRY_BYTES;
            let mut indexer = IndexerKEntry::default();
            indexer.data.copy_from_slice(
                &pool[base + INDEXER_K_DATA_OFFSET
                    ..base + INDEXER_K_DATA_OFFSET + INDEXER_K_DATA_BYTES],
            );
            indexer.scales.copy_from_slice(
                &pool[base + INDEXER_K_SCALE_OFFSET
                    ..base + INDEXER_K_SCALE_OFFSET + INDEXER_K_SCALE_BYTES],
            );
            let mut values = [0.0f32; INDEX_HEAD_DIM];
            dequantize_indexer_k(&indexer, &mut values)
                .map_err(|error| format!("indexer K entry {entry}: {error}"))?;
            let row = (block * HSI_BLOCK_TOKENS + slot) * INDEX_HEAD_DIM;
            for (channel, value) in values.iter().enumerate() {
                keys[row + channel] = bf16_bits(*value);
            }
        }
    }
    Ok(keys)
}

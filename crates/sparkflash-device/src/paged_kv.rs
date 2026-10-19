//! Serving attention operands over the paged format-2 KV pool (task-21, S3).
//!
//! [sparkflash_kv::paged] decides *which physical page* a served sequence's logical tokens live in.
//! This module is the device-attention half of that seam: it turns a sequence's block table into the
//! operands the released `flash_mla_csa2_sm121` entry consumes - the oldest-first window rows and
//! the selected compressed candidates - by reading the format-2 source entries through the mapping.
//! It is host code and pure: it never launches, allocates device memory or opens a driver.
//!
//! # What it produces
//!
//! * [window_rows]: the last min(position, 128) tokens' SWA slots, gathered oldest first from the
//!   sequence's tail ring into the kernel's contiguous row order. This is the rotation copy
//!   `docs/build/phase2-device-kv-store.md` lists as open item 1; the store keeps the
//!   reference's slot-indexed ring, the kernel wants row order.
//! * [candidate_rows]: the main-KV row of each selected compressed entry, gathered from the paged
//!   global-KV pool in candidate order. The kernel's `kv_fp4`/`kv_scales` operands are flat
//!   arrays indexed by position, so a paged sequence has to stage its rows; the staging is one
//!   copy of `MAIN_KV_ENTRY_BYTES` per candidate, and that cost is the price of keeping the released
//!   kernel ABI unchanged (a block-table read path inside the kernel would be a new pinned entry,
//!   which is not this milestone).
//! * [source_entry]: the one mapping both of the above use, exposed so the tests can assert the
//!   addressing instead of only the gathered bytes.
//! * [absolute_source_rows] and [absolute_indexer_keys]: the task-43 absolute-position operands.
//!   The first stages positions 0..=position as dense kv_fp4/kv_scales rows with compression
//!   expanded; the second stages the layer's indexer K entries as a bf16 HSI key pool in absolute
//!   block order. These are the operands the paged serving path passes;
//!   [PagedKvView::compact_candidates] remains the checked helper for callers that legitimately
//!   pass compact ids.
//!
//! # The mask invariant (one line a reader can check)
//!
//! Compacted ids are only safe while every candidate is causal; whenever masking is semantically
//! required the device kernel must be given full-length ids (or equivalent positions).
//!
//! The seam enforces the first half: [PagedKvView::compact_candidates] refuses a candidate past
//! the query position ([PagingError::NonCausalCandidate]) and its fields are private, so a staged
//! call cannot be handed an unchecked list. The mask test proves the distinction the hard way: the
//! full-length reference mask rejects a future candidate bit-identically to never naming it, while
//! compacting that same candidate renumbers it into range and changes the attention output. The
//! kernel's indexer stage scores blocks by absolute position. Task-43 selected the absolute-path
//! option: the paged path stages the sequence's main-KV rows in absolute token order and an HSI key
//! pool in absolute block order, so the pinned kernel sees absolute ids and absolute positions and
//! both stages are correct for reasons a reviewer can check without new kernel logic. That staging
//! is O(context) per source layer per step - a throughput cost recorded in
//! docs/build/serving-paged-kv.md, which also names the positions-array kernel as the follow-on
//! that would make it O(candidates) at the price of a frozen-ABI change (FlashMlaParams 19 -> 20).
//! The device run of this path - a served sequence attending through flash_mla over paged rows - is
//! a window measurement and is recorded as unclaimed; the CPU twin below is what the tests assert.

use sparkflash_kernels::m4_quant::bf16_bits;
use sparkflash_kernels::mla_ref::{HSI_BLOCK_TOKENS, HSI_BLOCKS, INDEX_HEAD_DIM};
use sparkflash_kv::config::KV_PAGE_TOKENS;
use sparkflash_kv::config::{KV_SOURCE_LAYERS, compress_ratio};
use sparkflash_kv::fp4::{IndexerKEntry, dequantize_indexer_k};
use sparkflash_kv::layout::{
    GLOBAL_ENTRY_BYTES, INDEXER_K_DATA_OFFSET, INDEXER_K_DIM, INDEXER_K_ENTRY_BYTES,
    MAIN_KV_DATA_BYTES, MAIN_KV_DATA_OFFSET, MAIN_KV_SCALE_BYTES, MAIN_KV_SCALE_OFFSET, SWA_LAYERS,
    SWA_SLOT_DATA_BYTES, SWA_SLOT_SCALE_BYTES, SWA_SLOTS_PER_TAIL, entries_per_page, source_slot,
    swa_slot_offset,
};
use sparkflash_kv::paged::{PagedKvCache, PagingError, SeqId};

/// Where one logical token's compressed KV entry sits in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceEntry {
    /// The physical page the token's block lives in.
    pub page: u64,
    /// The entry's index inside that page (a page holds 64 / ratio entries).
    pub entry_in_page: usize,
    /// The entry's index in the layer's whole source array (page * entries_per_page + entry).
    pub pool_index: usize,
    /// The entry's byte offset in the layer's source array.
    pub byte_offset: usize,
}

/// The kernel's window operands, oldest row first.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WindowRows {
    /// Packed FP4 latent codes, `rows * SWA_SLOT_DATA_BYTES`.
    pub codes: Vec<u8>,
    /// One E8M0 scale per 32 channels, `rows * SWA_SLOT_SCALE_BYTES`.
    pub scales: Vec<u8>,
    /// Number of rows gathered (min(position, 128); zero at position 0).
    pub rows: usize,
}

/// The kernel's candidate operands: packed FP4 codes and E4M3 scales, one row per candidate.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CandidateOperands {
    /// `kv_fp4`: `MAIN_KV_DATA_BYTES` packed codes per candidate row.
    pub codes: Vec<u8>,
    /// `kv_scales`: `MAIN_KV_SCALE_BYTES` E4M3 scales per candidate row.
    pub scales: Vec<u8>,
}

impl CandidateOperands {
    /// Candidates staged.
    #[must_use]
    pub fn len(&self) -> usize {
        self.codes.len() / MAIN_KV_DATA_BYTES
    }

    /// Whether nothing was staged.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }
}

/// A compact candidate list proved causal by [PagedKvView::compact_candidates].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactCandidates {
    ids: Vec<u32>,
    position: u32,
}

impl CompactCandidates {
    /// The compact ids the kernel takes as its candidate list: `0..len`.
    #[must_use]
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// The query position to pass with the compact ids (`len - 1`; 0 for an empty list).
    #[must_use]
    pub const fn position(&self) -> u32 {
        self.position
    }

    /// Candidate count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether no candidate was selected (empty is legal: pure window attention).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Dense absolute-position main-KV operands for one source layer.
///
/// The kernel indexes kv_fp4 and kv_scales by absolute token position, so the paged path stages
/// every position up to the query into these arrays. A ratio-2 store keeps one entry per two
/// tokens; both rows of a group read that entry, which is the expanded view the kernel expects.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AbsoluteRows {
    /// Packed FP4 codes, MAIN_KV_DATA_BYTES per absolute row.
    pub codes: Vec<u8>,
    /// E4M3 scales, MAIN_KV_SCALE_BYTES per absolute row.
    pub scales: Vec<u8>,
    /// Rows staged: the query position plus one.
    pub rows: usize,
}

impl AbsoluteRows {
    /// Rows staged.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows
    }

    /// Whether nothing was staged (a query position always stages its own row).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }
}

/// The kernel's indexer operands: the HSI key pool in absolute block order.
///
/// Block b, slot s, channel c lives at (b * HSI_BLOCK_TOKENS + s) * INDEX_HEAD_DIM + c as a bf16
/// bit pattern. Every slot past the query position is zero; the kernel's position mask makes those
/// unreachable, so a served sequence never publishes a future key.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IndexerKeys {
    /// bf16 bit patterns in absolute block order.
    pub keys: Vec<u16>,
    /// Blocks carrying at least one reachable token (the rest are zero and masked).
    pub reachable_blocks: usize,
    /// The query position the pool was staged for.
    pub position: u32,
}

impl IndexerKeys {
    /// Slot count in the pool: HSI_BLOCKS times HSI_BLOCK_TOKENS.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.keys.len() / INDEX_HEAD_DIM
    }
}
/// The mapping from a sequence's logical tokens to the format-2 pools.
#[derive(Debug)]
pub struct PagedKvView<'a> {
    cache: &'a PagedKvCache,
}

impl<'a> PagedKvView<'a> {
    /// A view over one paged cache.
    #[must_use]
    pub const fn new(cache: &'a PagedKvCache) -> Self {
        Self { cache }
    }

    /// The mapping of logical token `logical` of `sequence` to its compressed entry in `layer`'s
    /// source array.
    ///
    /// # Errors
    /// Unknown sequence, a logical position beyond the sequence, or a layer that owns no compressed
    /// KV (SWA-only layers and layers outside the 40-layer backbone).
    pub fn source_entry(
        &self,
        sequence: SeqId,
        layer: usize,
        logical: usize,
    ) -> Result<SourceEntry, PagingError> {
        let slot = source_slot(layer).ok_or_else(|| {
            PagingError::Invariant(format!("layer {layer} owns no compressed global KV"))
        })?;
        let ratio = compress_ratio(layer);
        let page = self.cache.page_of_token(sequence, logical)?;
        let entry_in_page = (logical % KV_PAGE_TOKENS) / ratio;
        let pool_index = usize::try_from(page.0)
            .ok()
            .and_then(|page| page.checked_mul(entries_per_page(slot)))
            .and_then(|base| base.checked_add(entry_in_page))
            .ok_or_else(|| {
                PagingError::Invariant("source entry index overflows usize".to_owned())
            })?;
        Ok(SourceEntry {
            page: page.0,
            entry_in_page,
            pool_index,
            byte_offset: pool_index * GLOBAL_ENTRY_BYTES,
        })
    }

    /// Gathers the main-KV rows of `selected` (logical token positions) from `layer`'s paged source
    /// pool, in candidate order, split into the two flat operands the kernel takes.
    ///
    /// [CandidateOperands::codes] is the kernel's `kv_fp4` operand (`MAIN_KV_DATA_BYTES` per row,
    /// packed FP4 codes) and [CandidateOperands::scales] its `kv_scales` operand
    /// (`MAIN_KV_SCALE_BYTES` E4M3 scales per row). The store keeps them interleaved inside one
    /// 288-byte entry; the kernel wants two arrays, so this is where the split happens. Row order is
    /// candidate order, which is the compact id order of [CompactCandidates].
    ///
    /// # Errors
    /// Unknown sequence, a candidate beyond the sequence, a layer with no compressed KV, or a pool
    /// shorter than the deepest entry the mapping reaches.
    pub fn candidate_rows(
        &self,
        sequence: SeqId,
        layer: usize,
        selected: &[u32],
        pool: &[u8],
    ) -> Result<CandidateOperands, PagingError> {
        let mut operands = CandidateOperands {
            codes: Vec::with_capacity(selected.len() * MAIN_KV_DATA_BYTES),
            scales: Vec::with_capacity(selected.len() * MAIN_KV_SCALE_BYTES),
        };
        for &logical in selected {
            let entry = self.source_entry(sequence, layer, logical as usize)?;
            let codes_start = entry.byte_offset + MAIN_KV_DATA_OFFSET;
            let scales_start = entry.byte_offset + MAIN_KV_SCALE_OFFSET;
            let codes = pool
                .get(codes_start..codes_start + MAIN_KV_DATA_BYTES)
                .ok_or_else(|| {
                    PagingError::Invariant(format!(
                        "source pool is {} bytes, the mapping reaches byte {}",
                        pool.len(),
                        codes_start + MAIN_KV_DATA_BYTES
                    ))
                })?;
            let scales = pool
                .get(scales_start..scales_start + MAIN_KV_SCALE_BYTES)
                .ok_or_else(|| {
                    PagingError::Invariant(format!(
                        "source pool is {} bytes, the mapping reaches byte {}",
                        pool.len(),
                        scales_start + MAIN_KV_SCALE_BYTES
                    ))
                })?;
            operands.codes.extend_from_slice(codes);
            operands.scales.extend_from_slice(scales);
        }
        Ok(operands)
    }

    /// Proves every selected candidate is causal at `position` and returns the compact list the
    /// staged kernel call may pass.
    ///
    /// Compaction renumbers candidates `0..len` and sets the kernel position to `len - 1`, which
    /// makes the kernel's `candidate <= position` mask true for every compact id. That is only safe
    /// while every candidate is causal, so this constructor refuses a candidate past the query
    /// position ([PagingError::NonCausalCandidate]) instead of trusting the caller; the compact
    /// fields are private, so a staged call cannot be handed an unchecked list.
    ///
    /// # Errors
    /// Unknown sequence, a query position the sequence does not hold, or a non-causal candidate.
    pub fn compact_candidates(
        &self,
        sequence: SeqId,
        selected: &[u32],
        position: u32,
    ) -> Result<CompactCandidates, PagingError> {
        let held = self
            .cache
            .sequence(sequence)
            .ok_or(PagingError::UnknownSequence(sequence))?
            .tokens()
            .len();
        if position as usize >= held {
            return Err(PagingError::OutOfRange {
                sequence,
                logical: position as usize,
                held,
            });
        }
        for &candidate in selected {
            if candidate > position {
                return Err(PagingError::NonCausalCandidate {
                    sequence,
                    candidate,
                    position,
                });
            }
        }
        Ok(CompactCandidates {
            ids: (0..selected.len() as u32).collect(),
            position: selected.len().saturating_sub(1) as u32,
        })
    }

    /// Gathers the window rows for `position` (the number of tokens already attended) from
    /// `layer`'s SWA tail ring, oldest row first.
    ///
    /// `tail` is one sequence's tail buffer: [sparkflash_kv::layout::SWA_TAIL_BYTES] bytes holding
    /// all [SWA_LAYERS] layers' 128-slot rings. The ring keeps `slot = position % 128`; the kernel
    /// wants the attended window in row order, so this is the rotation copy.
    ///
    /// # Errors
    /// Unknown sequence, a position beyond the sequence's tokens, a layer outside the backbone, or a
    /// tail buffer shorter than [sparkflash_kv::layout::SWA_TAIL_BYTES].
    pub fn window_rows(
        &self,
        sequence: SeqId,
        layer: usize,
        position: usize,
        tail: &[u8],
    ) -> Result<WindowRows, PagingError> {
        let held = self
            .cache
            .sequence(sequence)
            .ok_or(PagingError::UnknownSequence(sequence))?
            .tokens()
            .len();
        if position > held {
            return Err(PagingError::OutOfRange {
                sequence,
                logical: position,
                held,
            });
        }
        if layer >= SWA_LAYERS {
            return Err(PagingError::Invariant(format!(
                "layer {layer} is outside the SWA backbone"
            )));
        }
        let rows = position.min(SWA_SLOTS_PER_TAIL);
        let first = position - rows;
        let mut window = WindowRows {
            codes: Vec::with_capacity(rows * SWA_SLOT_DATA_BYTES),
            scales: Vec::with_capacity(rows * SWA_SLOT_SCALE_BYTES),
            rows,
        };
        for logical in first..position {
            let slot = logical % SWA_SLOTS_PER_TAIL;
            let offset = swa_slot_offset(layer, slot);
            let data_end = offset + SWA_SLOT_DATA_BYTES;
            let scale_end = data_end + SWA_SLOT_SCALE_BYTES;
            let codes = tail.get(offset..data_end).ok_or_else(|| {
                PagingError::Invariant(format!(
                    "tail is {} bytes, layer {layer} slot {slot} ends at byte {data_end}",
                    tail.len()
                ))
            })?;
            let scales = tail.get(data_end..scale_end).ok_or_else(|| {
                PagingError::Invariant(format!(
                    "tail is {} bytes, layer {layer} slot {slot} scales end at byte {scale_end}",
                    tail.len()
                ))
            })?;
            window.codes.extend_from_slice(codes);
            window.scales.extend_from_slice(scales);
        }
        Ok(window)
    }

    /// Stages the main-KV operands of positions 0..=position in absolute token order.
    ///
    /// The kernel's kv_fp4/kv_scales are flat arrays indexed by absolute token position, so a
    /// paged sequence passes absolute ids and this staged view: the kernel's candidate <= position
    /// mask then compares two absolute values and is not vacuous. Compression is expanded here: a
    /// ratio-2 entry serves both of its tokens' rows.
    ///
    /// Cost: one MAIN_KV_ENTRY_BYTES copy per position per source layer per step, and the window
    /// rows are still gathered separately. That is the price of leaving the pinned kernel ABI
    /// untouched; the positions-array kernel that would make staging O(candidates) is the named
    /// follow-on in docs/build/serving-paged-kv.md.
    ///
    /// # Errors
    /// Unknown sequence, a position beyond the sequence, a layer with no compressed KV, or a pool
    /// shorter than the deepest entry the mapping reaches.
    pub fn absolute_source_rows(
        &self,
        sequence: SeqId,
        layer: usize,
        position: usize,
        pool: &[u8],
    ) -> Result<AbsoluteRows, PagingError> {
        let held = self
            .cache
            .sequence(sequence)
            .ok_or(PagingError::UnknownSequence(sequence))?
            .tokens()
            .len();
        if position >= held {
            return Err(PagingError::OutOfRange {
                sequence,
                logical: position,
                held,
            });
        }
        let rows = position + 1;
        let mut staged = AbsoluteRows {
            codes: Vec::with_capacity(rows * MAIN_KV_DATA_BYTES),
            scales: Vec::with_capacity(rows * MAIN_KV_SCALE_BYTES),
            rows,
        };
        for logical in 0..rows {
            let entry = self.source_entry(sequence, layer, logical)?;
            let codes_start = entry.byte_offset + MAIN_KV_DATA_OFFSET;
            let scales_start = entry.byte_offset + MAIN_KV_SCALE_OFFSET;
            let codes = pool
                .get(codes_start..codes_start + MAIN_KV_DATA_BYTES)
                .ok_or_else(|| {
                    PagingError::Invariant(format!(
                        "source pool is {} bytes, the mapping reaches byte {}",
                        pool.len(),
                        codes_start + MAIN_KV_DATA_BYTES
                    ))
                })?;
            let scales = pool
                .get(scales_start..scales_start + MAIN_KV_SCALE_BYTES)
                .ok_or_else(|| {
                    PagingError::Invariant(format!(
                        "source pool is {} bytes, the mapping reaches byte {}",
                        pool.len(),
                        scales_start + MAIN_KV_SCALE_BYTES
                    ))
                })?;
            staged.codes.extend_from_slice(codes);
            staged.scales.extend_from_slice(scales);
        }
        Ok(staged)
    }

    /// Stages the layer's indexer K entries into the HSI operand in absolute block order.
    ///
    /// The kernel scores HSI block b from index_key[b][slot][channel] against the absolute
    /// position; it never learns the block table. Serving therefore gathers the sequence's
    /// layer-20 indexer K entries through the mapping, dequantises FP4 E2M1 with E8M0 scales on
    /// the host and narrows each value to bf16, then hands the kernel the pool in absolute block
    /// order. Slots past the query position are zero and unreachable under the kernel's own mask.
    ///
    /// Cost: the operand is HSI_BLOCKS * HSI_BLOCK_TOKENS * INDEX_HEAD_DIM bf16 values (4 MiB)
    /// per step; only ceil((position + 1) / HSI_BLOCK_TOKENS) blocks are written, the rest is the
    /// zero fill. A device-side gather is phase2 open item 2, not this milestone.
    ///
    /// # Errors
    /// Unknown sequence, a position beyond the sequence or outside the HSI pool, a layer with no
    /// compressed KV, a pool shorter than the deepest entry the mapping reaches, or an indexer K
    /// entry that does not parse.
    pub fn absolute_indexer_keys(
        &self,
        sequence: SeqId,
        layer: usize,
        position: usize,
        pool: &[u8],
    ) -> Result<IndexerKeys, PagingError> {
        let held = self
            .cache
            .sequence(sequence)
            .ok_or(PagingError::UnknownSequence(sequence))?
            .tokens()
            .len();
        if position >= held {
            return Err(PagingError::OutOfRange {
                sequence,
                logical: position,
                held,
            });
        }
        let slots = HSI_BLOCKS * HSI_BLOCK_TOKENS;
        if position >= slots {
            return Err(PagingError::Invariant(format!(
                "query position {position} is outside the {slots}-token HSI pool"
            )));
        }
        let mut keys = vec![0u16; slots * INDEX_HEAD_DIM];
        for logical in 0..=position {
            let entry = self.source_entry(sequence, layer, logical)?;
            let start = entry.byte_offset + INDEXER_K_DATA_OFFSET;
            let end = start + INDEXER_K_ENTRY_BYTES;
            let bytes = pool.get(start..end).ok_or_else(|| {
                PagingError::Invariant(format!(
                    "source pool is {} bytes, the mapping reaches byte {end}",
                    pool.len()
                ))
            })?;
            let indexer_entry = IndexerKEntry::from_bytes(bytes).map_err(|error| {
                PagingError::Invariant(format!("indexer k entry for token {logical}: {error}"))
            })?;
            let mut values = [0.0f32; INDEXER_K_DIM];
            dequantize_indexer_k(&indexer_entry, &mut values).map_err(|error| {
                PagingError::Invariant(format!("indexer k values for token {logical}: {error}"))
            })?;
            let row = &mut keys[logical * INDEX_HEAD_DIM..(logical + 1) * INDEX_HEAD_DIM];
            for (slot, value) in row.iter_mut().zip(values.iter()) {
                *slot = bf16_bits(*value);
            }
        }
        Ok(IndexerKeys {
            keys,
            reachable_blocks: (position + 1).div_ceil(HSI_BLOCK_TOKENS),
            position: position as u32,
        })
    }
    /// The source layers this view can resolve, in slot order.
    #[must_use]
    pub fn source_layers() -> &'static [usize] {
        &KV_SOURCE_LAYERS
    }
}

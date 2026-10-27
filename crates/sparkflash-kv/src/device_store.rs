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

//! The device KV store: the flash_mla_csa2_sm121 structure-of-arrays ABI, owned explicitly.
//!
//! The kernel at 6d02313 takes nineteen parameters and reads the KV state as flat arrays
//! (device-path-design section 3): kv_fp4 at p * 256, kv_scales at p * 32, the engine FP8
//! window as window_codes at row * 512 plus window_scales at row * 16, the HSI candidate
//! pool as [2048][8][128] bf16, and the current Top-K list as at most 512 u32. This module
//! owns exactly those buffers, plus nothing else: no activations, no weights, no paging, no
//! eviction. The host cache in this crate stays the metadata layer (radix pages, page tables,
//! plan_request); its 56,960-byte array-of-structs page is unpacked into these per-layer arrays
//! on write.
//!
//! # Ownership
//!
//! One [DeviceKvStore] owns every buffer and frees them together. There are no process globals
//! and no hidden allocation: construction takes a [StoreMemory] backend and a [StoreGeometry],
//! allocates the eight buffer families once, and never reallocates. The layout, the page
//! unpacking, the entry codecs and the audit manifest are pure Rust; the CUDA backend lives in
//! the feature-gated device_cuda module so host-only users never link or build against CUDA.
//!
//! # Audit
//!
//! Every written entry can be represented by an [AuditRow]: a SHA-256 over the raw stored
//! bytes and a SHA-256 over the dequantise round-trip bytes the reference semantics define
//! (main KV and indexer K decoded to f32; a window slot as code * scale rounded to bf16, the
//! engine's stored dtype). [DeviceKvStore::audit] reads the store back, recomputes both
//! digests and fails on the first mismatch, naming (kind, layer, position). A one-byte
//! corruption is therefore a testable failure, not a silent pass.

use sha2::{Digest, Sha256};

use crate::config::{self, CANDIDATE_POSITIONS, INDEX_TOPK, KV_PAGE_TOKENS, KV_SOURCE_LAYERS};
use crate::fp4::{
    IndexerKEntry, MainKvEntry, SwaKvSlot, dequantize_indexer_k, dequantize_main_kv,
    dequantize_swa_slot,
};
use crate::layout::{
    GLOBAL_PAGE_BYTES, GLOBAL_PAGE_ENTRIES, INDEXER_K_DATA_BYTES, INDEXER_K_DIM,
    INDEXER_K_ENTRY_BYTES, INDEXER_K_SCALE_BYTES, MAIN_KV_DATA_BYTES, MAIN_KV_DIM,
    MAIN_KV_ENTRY_BYTES, MAIN_KV_SCALE_BYTES, SWA_LAYERS, SWA_SLOT_BYTES, SWA_SLOT_DATA_BYTES,
    SWA_SLOT_SCALE_BYTES, SWA_SLOTS_PER_TAIL, entries_per_page, page_entry_offset,
};

/// Alignment every store buffer requests. cuMemAlloc returns 256-byte aligned memory.
pub const STORE_BUFFER_ALIGNMENT: u64 = 256;

/// Number of 128-slot window rings: one per backbone layer.
pub const WINDOW_RINGS: usize = SWA_LAYERS;

/// Slots in one window ring (the 128-token window).
pub const WINDOW_SLOTS: usize = SWA_SLOTS_PER_TAIL;

/// Bytes of the HSI candidate pool: [2048 blocks][8 tokens][128 dims] bf16.
pub const HSI_POOL_BYTES: usize = CANDIDATE_POSITIONS * INDEXER_K_DIM * 2;

/// Bytes of the current Top-K list: 512 u32 position ids.
pub const SELECTED_BYTES: usize = INDEX_TOPK * 4;

/// Number of buffer families the store allocates.
pub const BUFFER_COUNT: usize = 8;

/// One buffer family the store must allocate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferSpec {
    /// Stable name, used in logs and measurements.
    pub name: &'static str,
    /// Size in bytes at the store's geometry.
    pub bytes: u64,
}

/// A per-layer compressed-KV source: one of the four Full-mode layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceShape {
    /// Backbone layer id.
    pub layer: usize,
    /// compress_ratios[layer]: one entry per ratio tokens.
    pub ratio: usize,
    /// Entries per 64-token page: 32 at ratio 2, 64 at ratio 1.
    pub page_entries: u64,
    /// Total entries the store holds for this layer.
    pub entries: u64,
    /// Index of this layer's first entry inside the shared entry arrays.
    pub entry_base: u64,
}

/// Everything the store's byte layout depends on.
///
/// The geometry is page-rounded: a token capacity of n becomes ceil(n / 64) whole pages, and
/// each source layer stores pages * (64 / ratio) entries. That is the same arithmetic the host
/// cache uses for whole pages, so a page payload always lands in an in-range entry range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreGeometry {
    max_seq_len: usize,
    pages: u64,
    sources: [SourceShape; 4],
}

impl StoreGeometry {
    /// Builds the geometry for a token capacity, rounded up to whole 64-token pages.
    ///
    /// # Errors
    /// Returns [StoreError::EmptyGeometry] for zero tokens and [StoreError::TooLarge] when the
    /// page count would not fit a device offset.
    pub fn for_max_seq_len(tokens: usize) -> Result<Self, StoreError> {
        if tokens == 0 {
            return Err(StoreError::EmptyGeometry);
        }
        let pages = u64::try_from(tokens)
            .map_err(|_| StoreError::TooLarge)?
            .div_ceil(KV_PAGE_TOKENS as u64);
        let max_seq_len = usize::try_from(pages)
            .ok()
            .and_then(|pages| pages.checked_mul(KV_PAGE_TOKENS))
            .ok_or(StoreError::TooLarge)?;
        let mut sources = [SourceShape {
            layer: 0,
            ratio: 0,
            page_entries: 0,
            entries: 0,
            entry_base: 0,
        }; 4];
        let mut entry_base = 0u64;
        for (slot, layer) in KV_SOURCE_LAYERS.iter().enumerate() {
            let ratio = config::compress_ratio(*layer);
            let page_entries = entries_per_page(slot) as u64;
            let entries = pages
                .checked_mul(page_entries)
                .ok_or(StoreError::TooLarge)?;
            sources[slot] = SourceShape {
                layer: *layer,
                ratio,
                page_entries,
                entries,
                entry_base,
            };
            entry_base = entry_base
                .checked_add(entries)
                .ok_or(StoreError::TooLarge)?;
        }
        Ok(Self {
            max_seq_len,
            pages,
            sources,
        })
    }

    /// Token capacity actually reserved (a whole number of 64-token pages).
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Whole 64-token pages the store holds.
    #[must_use]
    pub fn pages(&self) -> u64 {
        self.pages
    }

    /// The four Full-mode compressed-KV sources.
    #[must_use]
    pub fn sources(&self) -> &[SourceShape; 4] {
        &self.sources
    }

    /// Shape of the source layer, when it owns compressed KV.
    #[must_use]
    pub fn source(&self, layer: usize) -> Option<&SourceShape> {
        self.sources.iter().find(|source| source.layer == layer)
    }

    /// The Full layer whose arrays a given layer consumes, or None when it consumes none.
    ///
    /// Reuse and Reindex layers read the nearest preceding Full layer's compressed KV and
    /// indexer K (model.py:500, 517-525; owns_k is only the kv_source list). SWA-only layers,
    /// and layers outside the 40-layer backbone (the MTP stages 40-42, which the store does not
    /// hold), have no source and return None. The [crate::config::compress_ratio] lookup is only
    /// defined inside that backbone, so the bound is checked first.
    #[must_use]
    pub fn source_layer_for(&self, layer: usize) -> Option<usize> {
        if layer >= SWA_LAYERS || config::compress_ratio(layer) == 0 {
            None
        } else {
            Some(config::kv_source_for(layer))
        }
    }

    /// Total entries across the four source layers.
    #[must_use]
    pub fn total_entries(&self) -> u64 {
        self.sources.iter().map(|source| source.entries).sum()
    }

    /// Bytes of the packed FP4 main-KV payload array.
    #[must_use]
    pub fn main_kv_data_bytes(&self) -> u64 {
        self.total_entries() * MAIN_KV_DATA_BYTES as u64
    }

    /// Bytes of the main-KV E4M3 scale array.
    #[must_use]
    pub fn main_kv_scale_bytes(&self) -> u64 {
        self.total_entries() * MAIN_KV_SCALE_BYTES as u64
    }

    /// Bytes of the packed indexer-K payload array.
    #[must_use]
    pub fn indexer_k_data_bytes(&self) -> u64 {
        self.total_entries() * INDEXER_K_DATA_BYTES as u64
    }

    /// Bytes of the indexer-K E8M0 scale array.
    #[must_use]
    pub fn indexer_k_scale_bytes(&self) -> u64 {
        self.total_entries() * INDEXER_K_SCALE_BYTES as u64
    }

    /// Main KV + indexer K: 356 bytes per entry, 890.0 bytes per token at 2^20 tokens.
    #[must_use]
    pub fn global_kv_bytes(&self) -> u64 {
        self.total_entries() * (MAIN_KV_ENTRY_BYTES + INDEXER_K_ENTRY_BYTES) as u64
    }

    /// Window rings: 40 layers x 128 slots x (512 code + 16 scale) bytes, constant in context.
    #[must_use]
    pub fn window_bytes(&self) -> u64 {
        (WINDOW_RINGS * WINDOW_SLOTS * SWA_SLOT_BYTES) as u64
    }

    /// The two published handles: HSI candidate pool and Top-K list.
    #[must_use]
    pub fn handle_bytes(&self) -> u64 {
        (HSI_POOL_BYTES + SELECTED_BYTES) as u64
    }

    /// Everything the store allocates.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.global_kv_bytes() + self.window_bytes() + self.handle_bytes()
    }

    /// Main KV + indexer K per reserved token: 890.0 at 2^20 tokens.
    ///
    /// The window rings and handles are per sequence and are deliberately not in this figure.
    #[must_use]
    pub fn kv_bytes_per_token(&self) -> f64 {
        self.global_kv_bytes() as f64 / self.max_seq_len as f64
    }

    /// The eight buffer families, in allocation order.
    #[must_use]
    pub fn buffer_plan(&self) -> [BufferSpec; BUFFER_COUNT] {
        [
            BufferSpec {
                name: "kv_fp4",
                bytes: self.main_kv_data_bytes(),
            },
            BufferSpec {
                name: "kv_scales",
                bytes: self.main_kv_scale_bytes(),
            },
            BufferSpec {
                name: "index_k_fp4",
                bytes: self.indexer_k_data_bytes(),
            },
            BufferSpec {
                name: "index_k_scales",
                bytes: self.indexer_k_scale_bytes(),
            },
            BufferSpec {
                name: "window_codes",
                bytes: (WINDOW_RINGS * WINDOW_SLOTS * SWA_SLOT_DATA_BYTES) as u64,
            },
            BufferSpec {
                name: "window_scales",
                bytes: (WINDOW_RINGS * WINDOW_SLOTS * SWA_SLOT_SCALE_BYTES) as u64,
            },
            BufferSpec {
                name: "hsi_pool",
                bytes: HSI_POOL_BYTES as u64,
            },
            BufferSpec {
                name: "selected",
                bytes: SELECTED_BYTES as u64,
            },
        ]
    }

    /// The expected audit rows for one 56,960-byte host page payload.
    ///
    /// The host page is an array of 356-byte entries grouped by source layer
    /// (page_entry_offset); entry e of layer l is compressed position
    /// page * page_entries + e. The rows are computed from the payload itself, so they
    /// describe what the store should return, independent of the store.
    ///
    /// # Errors
    /// Returns [StoreError::InvalidLength] unless the payload is exactly one page, and
    /// [StoreError::OutOfBounds] for a page index at or past the geometry.
    pub fn page_manifest(&self, page: u64, payload: &[u8]) -> Result<Vec<AuditRow>, StoreError> {
        if payload.len() != GLOBAL_PAGE_BYTES {
            return Err(StoreError::InvalidLength {
                context: "host page payload",
                expected: GLOBAL_PAGE_BYTES,
                actual: payload.len(),
            });
        }
        if page >= self.pages {
            return Err(StoreError::OutOfBounds {
                what: "page",
                value: page,
                bound: self.pages,
            });
        }
        let mut rows = Vec::with_capacity(GLOBAL_PAGE_ENTRIES * 2);
        for (slot, source) in self.sources.iter().enumerate() {
            for entry in 0..source.page_entries {
                let offset = page_entry_offset(slot, entry as usize);
                let position = page * source.page_entries + entry;
                let main = StoreEntry::MainKv {
                    layer: source.layer,
                    position,
                };
                let indexer = StoreEntry::IndexerK {
                    layer: source.layer,
                    position,
                };
                let main_end = offset + MAIN_KV_ENTRY_BYTES;
                let index_end = main_end + INDEXER_K_ENTRY_BYTES;
                rows.push(AuditRow::from_stored(main, &payload[offset..main_end])?);
                rows.push(AuditRow::from_stored(
                    indexer,
                    &payload[main_end..index_end],
                )?);
            }
        }
        Ok(rows)
    }
}

/// One named entry in the store, used by the audit manifest and by corruption injection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StoreEntry {
    /// One 288-byte main-KV entry of a Full layer.
    MainKv {
        /// Owning Full layer.
        layer: usize,
        /// Compressed position within that layer's bound.
        position: u64,
    },
    /// One 68-byte indexer-K entry of a Full layer.
    IndexerK {
        /// Owning Full layer.
        layer: usize,
        /// Compressed position within that layer's bound.
        position: u64,
    },
    /// One 528-byte window slot of a backbone layer.
    Window {
        /// Backbone layer, 0..40.
        layer: usize,
        /// Ring slot, 0..128.
        slot: usize,
    },
}

impl StoreEntry {
    /// Bytes this entry occupies in its stored form.
    #[must_use]
    pub fn stored_bytes(self) -> usize {
        match self {
            StoreEntry::MainKv { .. } => MAIN_KV_ENTRY_BYTES,
            StoreEntry::IndexerK { .. } => INDEXER_K_ENTRY_BYTES,
            StoreEntry::Window { .. } => SWA_SLOT_BYTES,
        }
    }
}

/// A digest pair for one written entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuditRow {
    /// Which entry this row covers.
    pub entry: StoreEntry,
    /// SHA-256 of the raw stored entry bytes.
    pub stored_sha256: [u8; 32],
    /// SHA-256 of the dequantise round-trip bytes (see the module docs for the exact form).
    pub canonical_sha256: [u8; 32],
}

impl AuditRow {
    /// Builds the expected digests from known-correct stored bytes.
    ///
    /// # Errors
    /// Returns [StoreError::InvalidLength] when the bytes are not the entry's stored size,
    /// and [StoreError::Codec] when the stored bytes are not decodable.
    pub fn from_stored(entry: StoreEntry, stored: &[u8]) -> Result<Self, StoreError> {
        let expected = entry.stored_bytes();
        if stored.len() != expected {
            return Err(StoreError::InvalidLength {
                context: "stored entry",
                expected,
                actual: stored.len(),
            });
        }
        Ok(Self {
            entry,
            stored_sha256: sha256(stored),
            canonical_sha256: canonical_sha256(entry, stored)?,
        })
    }
}

/// One array as the kernel sees it: a device address, a length and a per-entry stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrayHandle {
    /// Device (or backend) address of the first byte.
    pub address: u64,
    /// Bytes from the first byte to the end of the last entry.
    pub bytes: u64,
    /// Bytes between consecutive entries.
    pub stride_bytes: u64,
}

/// The four arrays of one compressed-KV source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceHandles {
    /// The Full layer that owns these arrays (the resolution of the requested layer).
    pub layer: usize,
    /// Kernel's kv_fp4.
    pub kv_fp4: ArrayHandle,
    /// Kernel's kv_scales.
    pub kv_scales: ArrayHandle,
    /// MXFP4 indexer-K payload.
    pub index_k_fp4: ArrayHandle,
    /// Indexer-K E8M0 scales.
    pub index_k_scales: ArrayHandle,
}

/// The two window arrays of one backbone layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowHandles {
    /// Backbone layer.
    pub layer: usize,
    /// Kernel's window_codes: 128 rows of 512 E4M3 bytes.
    pub codes: ArrayHandle,
    /// Kernel's window_scales: 128 rows of 16 E8M0 bytes.
    pub scales: ArrayHandle,
}

/// The published Top-K list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectedHandle {
    /// Device address of the u32 ids.
    pub address: u64,
    /// Number of ids published, at most 512.
    pub count: u32,
}

/// Everything that can go wrong in the device store.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// The geometry was asked for zero tokens.
    #[error("store geometry must cover at least one token")]
    EmptyGeometry,
    /// The geometry does not fit a 64-bit device offset.
    #[error("store geometry does not fit the device address space")]
    TooLarge,
    /// A write or read fell outside a page-derived bound.
    #[error("{what} out of bounds: {value} >= {bound}")]
    OutOfBounds {
        /// What was addressed (page, window layer, entry byte).
        what: &'static str,
        /// The offending value.
        value: u64,
        /// The exclusive bound it violated.
        bound: u64,
    },
    /// A write or read fell outside a per-layer bound.
    #[error("layer {layer} {what} out of bounds: {value} >= {bound}")]
    LayerOutOfBounds {
        /// The owning layer.
        layer: usize,
        /// What was addressed (main KV position, indexer-K position, window slot).
        what: &'static str,
        /// The offending value.
        value: u64,
        /// The exclusive bound it violated.
        bound: u64,
    },
    /// A payload had the wrong length for the pinned layout.
    #[error("{context}: expected {expected} bytes, got {actual}")]
    InvalidLength {
        /// What was being read or written.
        context: &'static str,
        /// Length the layout requires.
        expected: usize,
        /// Length actually supplied.
        actual: usize,
    },
    /// A layer was asked for an array it does not own.
    #[error("layer {layer} does not own {what}")]
    WrongLayerMode {
        /// The offending layer.
        layer: usize,
        /// The array family that was requested.
        what: &'static str,
    },
    /// A handle was read before its buffer was published.
    #[error("the {what} has not been published")]
    NotPublished {
        /// The handle that was requested.
        what: &'static str,
    },
    /// The store backend failed.
    #[error("store backend: {0}")]
    Backend(String),
    /// A codec could not parse or dequantise the stored bytes.
    #[error("codec: {0}")]
    Codec(#[from] crate::KvError),
    /// The audit read a different entry than the manifest expected.
    #[error(
        "audit mismatch at {entry:?}: expected stored {:02x?}, found {:02x?} (canonical {:02x?} vs {:02x?})",
        expected_stored,
        found_stored,
        expected_canonical,
        found_canonical
    )]
    AuditMismatch {
        /// The entry that did not match.
        entry: StoreEntry,
        /// Digest the manifest expects from the stored bytes.
        expected_stored: Box<[u8; 32]>,
        /// Digest the store actually hashes for the stored bytes.
        found_stored: Box<[u8; 32]>,
        /// Digest the manifest expects from the canonical bytes.
        expected_canonical: Box<[u8; 32]>,
        /// Digest the store actually hashes for the canonical bytes.
        found_canonical: Box<[u8; 32]>,
    },
}

/// The allocation and copy backend the store is built on.
///
/// The store's layout and audit logic never mention CUDA; this trait is the only seam. The
/// host-only tests implement it over Vec<u8>; the node run implements it over the CUDA driver
/// behind the cuda feature.
pub trait StoreMemory {
    /// Backend buffer handle.
    type Buffer;

    /// Allocates bytes, aligned to [STORE_BUFFER_ALIGNMENT].
    ///
    /// # Errors
    /// Returns [StoreError::Backend] when the backend cannot allocate.
    fn allocate(&mut self, bytes: usize) -> Result<Self::Buffer, StoreError>;

    /// Releases a buffer previously returned by [StoreMemory::allocate].
    fn release(&mut self, buffer: Self::Buffer);

    /// Copies bytes from the host into the buffer at offset.
    ///
    /// # Errors
    /// Returns [StoreError::Backend] on a backend copy failure and
    /// [StoreError::OutOfBounds] when the range exceeds the buffer.
    fn write(&mut self, buffer: &Self::Buffer, offset: u64, bytes: &[u8])
    -> Result<(), StoreError>;

    /// Copies out.len() bytes out of the buffer at offset.
    ///
    /// # Errors
    /// Returns [StoreError::Backend] on a backend copy failure and
    /// [StoreError::OutOfBounds] when the range exceeds the buffer.
    fn read(&self, buffer: &Self::Buffer, offset: u64, out: &mut [u8]) -> Result<(), StoreError>;

    /// The device address the kernel would be given for this buffer.
    fn address(&self, buffer: &Self::Buffer) -> u64;

    /// Total bytes currently allocated through this backend, as the backend accounts them.
    fn allocated_bytes(&self) -> u64;
}

/// The eight buffer families, held together so they free as one.
struct Buffers<B> {
    kv_fp4: B,
    kv_scales: B,
    index_k_fp4: B,
    index_k_scales: B,
    window_codes: B,
    window_scales: B,
    hsi_pool: B,
    selected: B,
}

/// Which buffer family a read or write targets.
#[derive(Debug, Clone, Copy)]
enum BufferKind {
    KvFp4,
    KvScales,
    IndexKFp4,
    IndexKScales,
    WindowCodes,
    WindowScales,
    HsiPool,
    Selected,
}

/// The device KV store.
///
/// Construct once from a [StoreGeometry]; every buffer is allocated then and never moves.
/// Writes go through [DeviceKvStore::write_page] (the host page path) or the per-entry and
/// per-slot writers, all of which validate their bounds and refuse invalid writes.
pub struct DeviceKvStore<M: StoreMemory> {
    memory: Option<M>,
    geometry: StoreGeometry,
    buffers: Option<Buffers<M::Buffer>>,
    current_source: Option<usize>,
    selected_count: Option<u32>,
    hsi_published: bool,
}

impl<M: StoreMemory> DeviceKvStore<M> {
    /// Allocates the whole store for the geometry.
    ///
    /// # Errors
    /// Returns [StoreError::Backend] if any allocation fails; allocations already made are
    /// released before returning.
    pub fn new(memory: M, geometry: StoreGeometry) -> Result<Self, StoreError> {
        let mut memory = memory;
        let plan = geometry.buffer_plan();
        let mut raw = Vec::with_capacity(BUFFER_COUNT);
        for spec in plan {
            match memory.allocate(usize::try_from(spec.bytes).map_err(|_| StoreError::TooLarge)?) {
                Ok(buffer) => raw.push(buffer),
                Err(error) => {
                    for buffer in raw {
                        memory.release(buffer);
                    }
                    return Err(error);
                }
            }
        }
        let mut buffers = raw.into_iter();
        let mut next = || {
            buffers
                .next()
                .expect("the buffer plan always yields exactly BUFFER_COUNT buffers")
        };
        let buffers = Buffers {
            kv_fp4: next(),
            kv_scales: next(),
            index_k_fp4: next(),
            index_k_scales: next(),
            window_codes: next(),
            window_scales: next(),
            hsi_pool: next(),
            selected: next(),
        };
        Ok(Self {
            memory: Some(memory),
            geometry,
            buffers: Some(buffers),
            current_source: None,
            selected_count: None,
            hsi_published: false,
        })
    }

    /// The geometry this store was built for.
    #[must_use]
    pub fn geometry(&self) -> &StoreGeometry {
        &self.geometry
    }

    /// Bytes the backend reports allocated for this store.
    #[must_use]
    pub fn allocated_bytes(&self) -> u64 {
        self.memory
            .as_ref()
            .expect("store memory is present until into_memory or drop")
            .allocated_bytes()
    }

    /// The backend, for measurement or diagnostics.
    #[must_use]
    pub fn memory(&self) -> &M {
        self.memory
            .as_ref()
            .expect("store memory is present until into_memory or drop")
    }

    /// Releases every buffer and returns the backend, for post-free measurement.
    #[must_use]
    pub fn into_memory(mut self) -> M {
        self.release_buffers();
        self.memory
            .take()
            .expect("store memory is present until into_memory or drop")
    }

    fn release_buffers(&mut self) {
        if let Some(buffers) = self.buffers.take() {
            let memory = self
                .memory
                .as_mut()
                .expect("store memory is present while buffers are present");
            memory.release(buffers.kv_fp4);
            memory.release(buffers.kv_scales);
            memory.release(buffers.index_k_fp4);
            memory.release(buffers.index_k_scales);
            memory.release(buffers.window_codes);
            memory.release(buffers.window_scales);
            memory.release(buffers.hsi_pool);
            memory.release(buffers.selected);
        }
    }

    fn buffers(&self) -> &Buffers<M::Buffer> {
        self.buffers
            .as_ref()
            .expect("store buffers are present until drop")
    }

    fn target(&self, kind: BufferKind) -> &M::Buffer {
        let buffers = self.buffers();
        match kind {
            BufferKind::KvFp4 => &buffers.kv_fp4,
            BufferKind::KvScales => &buffers.kv_scales,
            BufferKind::IndexKFp4 => &buffers.index_k_fp4,
            BufferKind::IndexKScales => &buffers.index_k_scales,
            BufferKind::WindowCodes => &buffers.window_codes,
            BufferKind::WindowScales => &buffers.window_scales,
            BufferKind::HsiPool => &buffers.hsi_pool,
            BufferKind::Selected => &buffers.selected,
        }
    }

    fn write_part(
        &mut self,
        kind: BufferKind,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), StoreError> {
        let Self {
            memory, buffers, ..
        } = self;
        let buffers = buffers
            .as_ref()
            .expect("store buffers are present until drop");
        let target = match kind {
            BufferKind::KvFp4 => &buffers.kv_fp4,
            BufferKind::KvScales => &buffers.kv_scales,
            BufferKind::IndexKFp4 => &buffers.index_k_fp4,
            BufferKind::IndexKScales => &buffers.index_k_scales,
            BufferKind::WindowCodes => &buffers.window_codes,
            BufferKind::WindowScales => &buffers.window_scales,
            BufferKind::HsiPool => &buffers.hsi_pool,
            BufferKind::Selected => &buffers.selected,
        };
        memory
            .as_mut()
            .expect("store memory is present until into_memory or drop")
            .write(target, offset, bytes)
    }

    fn read_part(&self, kind: BufferKind, offset: u64, out: &mut [u8]) -> Result<(), StoreError> {
        self.memory
            .as_ref()
            .expect("store memory is present until into_memory or drop")
            .read(self.target(kind), offset, out)
    }

    fn main_offsets(&self, layer: usize, position: u64) -> Result<(u64, u64), StoreError> {
        let source = self
            .geometry
            .source(layer)
            .ok_or(StoreError::WrongLayerMode {
                layer,
                what: "main KV",
            })?;
        if position >= source.entries {
            return Err(StoreError::LayerOutOfBounds {
                layer,
                what: "main KV position",
                value: position,
                bound: source.entries,
            });
        }
        let entry = source.entry_base + position;
        Ok((
            entry * MAIN_KV_DATA_BYTES as u64,
            entry * MAIN_KV_SCALE_BYTES as u64,
        ))
    }

    fn indexer_offsets(&self, layer: usize, position: u64) -> Result<(u64, u64), StoreError> {
        let source = self
            .geometry
            .source(layer)
            .ok_or(StoreError::WrongLayerMode {
                layer,
                what: "indexer K",
            })?;
        if position >= source.entries {
            return Err(StoreError::LayerOutOfBounds {
                layer,
                what: "indexer-K position",
                value: position,
                bound: source.entries,
            });
        }
        let entry = source.entry_base + position;
        Ok((
            entry * INDEXER_K_DATA_BYTES as u64,
            entry * INDEXER_K_SCALE_BYTES as u64,
        ))
    }

    fn window_offsets(&self, layer: usize, slot: usize) -> Result<(u64, u64), StoreError> {
        if layer >= WINDOW_RINGS {
            return Err(StoreError::OutOfBounds {
                what: "window layer",
                value: layer as u64,
                bound: WINDOW_RINGS as u64,
            });
        }
        if slot >= WINDOW_SLOTS {
            return Err(StoreError::LayerOutOfBounds {
                layer,
                what: "window slot",
                value: slot as u64,
                bound: WINDOW_SLOTS as u64,
            });
        }
        let row = layer as u64 * WINDOW_SLOTS as u64 + slot as u64;
        Ok((
            row * SWA_SLOT_DATA_BYTES as u64,
            row * SWA_SLOT_SCALE_BYTES as u64,
        ))
    }

    /// Writes one 288-byte main-KV entry.
    ///
    /// # Errors
    /// Refuses a non-Full layer ([StoreError::WrongLayerMode]), a position at or past that
    /// layer's ratio-derived bound ([StoreError::LayerOutOfBounds]), and a wrong entry length
    /// ([StoreError::InvalidLength]).
    pub fn write_main_entry(
        &mut self,
        layer: usize,
        position: u64,
        stored: &[u8],
    ) -> Result<(), StoreError> {
        if stored.len() != MAIN_KV_ENTRY_BYTES {
            return Err(StoreError::InvalidLength {
                context: "main KV entry",
                expected: MAIN_KV_ENTRY_BYTES,
                actual: stored.len(),
            });
        }
        let (data, scales) = self.main_offsets(layer, position)?;
        self.write_part(BufferKind::KvFp4, data, &stored[..MAIN_KV_DATA_BYTES])?;
        self.write_part(BufferKind::KvScales, scales, &stored[MAIN_KV_DATA_BYTES..])
    }

    /// Writes one 68-byte indexer-K entry.
    ///
    /// # Errors
    /// As [DeviceKvStore::write_main_entry], for the indexer-K array.
    pub fn write_indexer_entry(
        &mut self,
        layer: usize,
        position: u64,
        stored: &[u8],
    ) -> Result<(), StoreError> {
        if stored.len() != INDEXER_K_ENTRY_BYTES {
            return Err(StoreError::InvalidLength {
                context: "indexer-K entry",
                expected: INDEXER_K_ENTRY_BYTES,
                actual: stored.len(),
            });
        }
        let (data, scales) = self.indexer_offsets(layer, position)?;
        self.write_part(BufferKind::IndexKFp4, data, &stored[..INDEXER_K_DATA_BYTES])?;
        self.write_part(
            BufferKind::IndexKScales,
            scales,
            &stored[INDEXER_K_DATA_BYTES..],
        )
    }

    /// Writes one 528-byte window slot.
    ///
    /// # Errors
    /// Refuses a layer or slot out of range ([StoreError::OutOfBounds] /
    /// [StoreError::LayerOutOfBounds]) and a wrong slot length
    /// ([StoreError::InvalidLength]).
    pub fn write_window_slot(
        &mut self,
        layer: usize,
        slot: usize,
        stored: &[u8],
    ) -> Result<(), StoreError> {
        if stored.len() != SWA_SLOT_BYTES {
            return Err(StoreError::InvalidLength {
                context: "window slot",
                expected: SWA_SLOT_BYTES,
                actual: stored.len(),
            });
        }
        let (codes, scales) = self.window_offsets(layer, slot)?;
        self.write_part(
            BufferKind::WindowCodes,
            codes,
            &stored[..SWA_SLOT_DATA_BYTES],
        )?;
        self.write_part(
            BufferKind::WindowScales,
            scales,
            &stored[SWA_SLOT_DATA_BYTES..],
        )
    }

    /// Unpacks and writes one 56,960-byte host page payload.
    ///
    /// The page covers tokens [64 * page, 64 * page + 64); entry e of each source layer is
    /// compressed position page * page_entries + e. Nothing is inferred and nothing is
    /// allocated: this is the only bulk write path, and it refuses an out-of-range page
    /// rather than growing the store.
    ///
    /// # Errors
    /// [StoreError::OutOfBounds] for a page at or past the geometry,
    /// [StoreError::InvalidLength] for a payload that is not exactly one page, and the
    /// per-entry errors of [DeviceKvStore::write_main_entry].
    pub fn write_page(&mut self, page: u64, payload: &[u8]) -> Result<(), StoreError> {
        if payload.len() != GLOBAL_PAGE_BYTES {
            return Err(StoreError::InvalidLength {
                context: "host page payload",
                expected: GLOBAL_PAGE_BYTES,
                actual: payload.len(),
            });
        }
        if page >= self.geometry.pages() {
            return Err(StoreError::OutOfBounds {
                what: "page",
                value: page,
                bound: self.geometry.pages(),
            });
        }
        let sources = *self.geometry.sources();
        for (slot, source) in sources.iter().enumerate() {
            for entry in 0..source.page_entries {
                let offset = page_entry_offset(slot, entry as usize);
                let position = page * source.page_entries + entry;
                let main_end = offset + MAIN_KV_ENTRY_BYTES;
                let index_end = main_end + INDEXER_K_ENTRY_BYTES;
                self.write_main_entry(source.layer, position, &payload[offset..main_end])?;
                self.write_indexer_entry(source.layer, position, &payload[main_end..index_end])?;
            }
        }
        Ok(())
    }

    /// Writes the HSI candidate pool and marks it published.
    ///
    /// # Errors
    /// [StoreError::InvalidLength] unless the buffer is exactly 4,194,304 bytes.
    pub fn publish_hsi_pool(&mut self, bytes: &[u8]) -> Result<(), StoreError> {
        if bytes.len() != HSI_POOL_BYTES {
            return Err(StoreError::InvalidLength {
                context: "HSI pool",
                expected: HSI_POOL_BYTES,
                actual: bytes.len(),
            });
        }
        self.write_part(BufferKind::HsiPool, 0, bytes)?;
        self.hsi_published = true;
        Ok(())
    }

    /// Writes the current Top-K list and marks it published.
    ///
    /// # Errors
    /// [StoreError::InvalidLength] when more than 512 ids are supplied.
    pub fn publish_selected(&mut self, ids: &[u32]) -> Result<(), StoreError> {
        if ids.len() > INDEX_TOPK {
            return Err(StoreError::InvalidLength {
                context: "selected top-k list",
                expected: INDEX_TOPK,
                actual: ids.len(),
            });
        }
        let mut bytes = Vec::with_capacity(ids.len() * 4);
        for id in ids {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        self.write_part(BufferKind::Selected, 0, &bytes)?;
        self.selected_count = Some(ids.len() as u32);
        Ok(())
    }

    /// Records the plan's current source layer (a Full layer).
    ///
    /// # Errors
    /// [StoreError::WrongLayerMode] for a layer that owns no compressed KV.
    pub fn set_current_source(&mut self, layer: usize) -> Result<(), StoreError> {
        if self.geometry.source(layer).is_none() {
            return Err(StoreError::WrongLayerMode {
                layer,
                what: "compressed KV",
            });
        }
        self.current_source = Some(layer);
        Ok(())
    }

    /// The plan's current source layer, when set.
    #[must_use]
    pub fn current_source(&self) -> Option<usize> {
        self.current_source
    }

    /// Resolves a layer to the source arrays it consumes.
    ///
    /// Full layers resolve to themselves, Reuse and Reindex layers to the nearest preceding
    /// Full layer (kv_source_for); SWA-only layers are refused.
    ///
    /// # Errors
    /// [StoreError::OutOfBounds] for a layer outside the 40-layer backbone (the MTP stages
    /// 40-42, which are outside the store), and [StoreError::WrongLayerMode] for an in-store
    /// SWA-only layer.
    pub fn source_handles(&self, layer: usize) -> Result<SourceHandles, StoreError> {
        if layer >= SWA_LAYERS {
            return Err(StoreError::OutOfBounds {
                what: "source layer",
                value: layer as u64,
                bound: SWA_LAYERS as u64,
            });
        }
        let owner = self
            .geometry
            .source_layer_for(layer)
            .ok_or(StoreError::WrongLayerMode {
                layer,
                what: "compressed KV",
            })?;
        let source = self
            .geometry
            .source(owner)
            .ok_or(StoreError::WrongLayerMode {
                layer: owner,
                what: "compressed KV",
            })?;
        let buffers = self.buffers();
        let kv_fp4_base = self.memory().address(&buffers.kv_fp4);
        let kv_scales_base = self.memory().address(&buffers.kv_scales);
        let index_fp4_base = self.memory().address(&buffers.index_k_fp4);
        let index_scales_base = self.memory().address(&buffers.index_k_scales);
        Ok(SourceHandles {
            layer: owner,
            kv_fp4: ArrayHandle {
                address: kv_fp4_base + source.entry_base * MAIN_KV_DATA_BYTES as u64,
                bytes: source.entries * MAIN_KV_DATA_BYTES as u64,
                stride_bytes: MAIN_KV_DATA_BYTES as u64,
            },
            kv_scales: ArrayHandle {
                address: kv_scales_base + source.entry_base * MAIN_KV_SCALE_BYTES as u64,
                bytes: source.entries * MAIN_KV_SCALE_BYTES as u64,
                stride_bytes: MAIN_KV_SCALE_BYTES as u64,
            },
            index_k_fp4: ArrayHandle {
                address: index_fp4_base + source.entry_base * INDEXER_K_DATA_BYTES as u64,
                bytes: source.entries * INDEXER_K_DATA_BYTES as u64,
                stride_bytes: INDEXER_K_DATA_BYTES as u64,
            },
            index_k_scales: ArrayHandle {
                address: index_scales_base + source.entry_base * INDEXER_K_SCALE_BYTES as u64,
                bytes: source.entries * INDEXER_K_SCALE_BYTES as u64,
                stride_bytes: INDEXER_K_SCALE_BYTES as u64,
            },
        })
    }

    /// The kernel's two window arrays for a layer, in ring-slot order.
    ///
    /// The kernel's window_rows input is contiguous oldest-first; the store keeps the
    /// reference's slot-indexed ring (slot = position % 128). Presenting that as the kernel's
    /// row order is the sequence milestone's rotation copy (design note section 9).
    ///
    /// # Errors
    /// [StoreError::OutOfBounds] for a layer outside 0..40.
    pub fn window_handles(&self, layer: usize) -> Result<WindowHandles, StoreError> {
        if layer >= WINDOW_RINGS {
            return Err(StoreError::OutOfBounds {
                what: "window layer",
                value: layer as u64,
                bound: WINDOW_RINGS as u64,
            });
        }
        let buffers = self.buffers();
        let row = layer as u64 * WINDOW_SLOTS as u64;
        Ok(WindowHandles {
            layer,
            codes: ArrayHandle {
                address: self.memory().address(&buffers.window_codes)
                    + row * SWA_SLOT_DATA_BYTES as u64,
                bytes: (WINDOW_SLOTS * SWA_SLOT_DATA_BYTES) as u64,
                stride_bytes: SWA_SLOT_DATA_BYTES as u64,
            },
            scales: ArrayHandle {
                address: self.memory().address(&buffers.window_scales)
                    + row * SWA_SLOT_SCALE_BYTES as u64,
                bytes: (WINDOW_SLOTS * SWA_SLOT_SCALE_BYTES) as u64,
                stride_bytes: SWA_SLOT_SCALE_BYTES as u64,
            },
        })
    }

    /// The published HSI candidate pool.
    ///
    /// # Errors
    /// [StoreError::NotPublished] before a publish call.
    pub fn hsi_pool_handle(&self) -> Result<ArrayHandle, StoreError> {
        if !self.hsi_published {
            return Err(StoreError::NotPublished { what: "HSI pool" });
        }
        Ok(ArrayHandle {
            address: self.memory().address(&self.buffers().hsi_pool),
            bytes: HSI_POOL_BYTES as u64,
            stride_bytes: (INDEXER_K_DIM * 2) as u64,
        })
    }

    /// The published Top-K list.
    ///
    /// # Errors
    /// [StoreError::NotPublished] before a publish call.
    pub fn selected_handle(&self) -> Result<SelectedHandle, StoreError> {
        let count = self.selected_count.ok_or(StoreError::NotPublished {
            what: "selected top-k list",
        })?;
        Ok(SelectedHandle {
            address: self.memory().address(&self.buffers().selected),
            count,
        })
    }

    /// Reads one stored entry back into a fresh vector.
    ///
    /// # Errors
    /// The bounds error of the entry and backend read failures.
    pub fn read_stored(&self, entry: StoreEntry) -> Result<Vec<u8>, StoreError> {
        let mut out = vec![0u8; entry.stored_bytes()];
        match entry {
            StoreEntry::MainKv { layer, position } => {
                let (data, scales) = self.main_offsets(layer, position)?;
                self.read_part(BufferKind::KvFp4, data, &mut out[..MAIN_KV_DATA_BYTES])?;
                self.read_part(BufferKind::KvScales, scales, &mut out[MAIN_KV_DATA_BYTES..])?;
            }
            StoreEntry::IndexerK { layer, position } => {
                let (data, scales) = self.indexer_offsets(layer, position)?;
                self.read_part(
                    BufferKind::IndexKFp4,
                    data,
                    &mut out[..INDEXER_K_DATA_BYTES],
                )?;
                self.read_part(
                    BufferKind::IndexKScales,
                    scales,
                    &mut out[INDEXER_K_DATA_BYTES..],
                )?;
            }
            StoreEntry::Window { layer, slot } => {
                let (codes, scales) = self.window_offsets(layer, slot)?;
                self.read_part(
                    BufferKind::WindowCodes,
                    codes,
                    &mut out[..SWA_SLOT_DATA_BYTES],
                )?;
                self.read_part(
                    BufferKind::WindowScales,
                    scales,
                    &mut out[SWA_SLOT_DATA_BYTES..],
                )?;
            }
        }
        Ok(out)
    }

    /// Reads the store back and audits it against a manifest of expected digests.
    ///
    /// # Errors
    /// [StoreError::AuditMismatch] naming the first entry whose stored or canonical digest
    /// differs from the manifest.
    pub fn audit(&self, expected: &[AuditRow]) -> Result<(), StoreError> {
        for row in expected {
            let stored = self.read_stored(row.entry)?;
            let found = AuditRow::from_stored(row.entry, &stored)?;
            if found.stored_sha256 != row.stored_sha256
                || found.canonical_sha256 != row.canonical_sha256
            {
                return Err(StoreError::AuditMismatch {
                    entry: row.entry,
                    expected_stored: Box::new(row.stored_sha256),
                    found_stored: Box::new(found.stored_sha256),
                    expected_canonical: Box::new(row.canonical_sha256),
                    found_canonical: Box::new(found.canonical_sha256),
                });
            }
        }
        Ok(())
    }

    /// Flips one bit in a stored entry: the corruption injector the audit test needs.
    ///
    /// # Errors
    /// The entry's bounds error and [StoreError::OutOfBounds] when byte_index is past the
    /// entry.
    pub fn inject_corruption(
        &mut self,
        entry: StoreEntry,
        byte_index: usize,
    ) -> Result<(), StoreError> {
        let len = entry.stored_bytes();
        if byte_index >= len {
            return Err(StoreError::OutOfBounds {
                what: "entry byte",
                value: byte_index as u64,
                bound: len as u64,
            });
        }
        let stored = self.read_stored(entry)?;
        let flipped = stored[byte_index] ^ 0x01;
        let (kind, offset) = match entry {
            StoreEntry::MainKv { layer, position } => {
                let (data, scales) = self.main_offsets(layer, position)?;
                if byte_index < MAIN_KV_DATA_BYTES {
                    (BufferKind::KvFp4, data + byte_index as u64)
                } else {
                    (
                        BufferKind::KvScales,
                        scales + (byte_index - MAIN_KV_DATA_BYTES) as u64,
                    )
                }
            }
            StoreEntry::IndexerK { layer, position } => {
                let (data, scales) = self.indexer_offsets(layer, position)?;
                if byte_index < INDEXER_K_DATA_BYTES {
                    (BufferKind::IndexKFp4, data + byte_index as u64)
                } else {
                    (
                        BufferKind::IndexKScales,
                        scales + (byte_index - INDEXER_K_DATA_BYTES) as u64,
                    )
                }
            }
            StoreEntry::Window { layer, slot } => {
                let (codes, scales) = self.window_offsets(layer, slot)?;
                if byte_index < SWA_SLOT_DATA_BYTES {
                    (BufferKind::WindowCodes, codes + byte_index as u64)
                } else {
                    (
                        BufferKind::WindowScales,
                        scales + (byte_index - SWA_SLOT_DATA_BYTES) as u64,
                    )
                }
            }
        };
        self.write_part(kind, offset, &[flipped])
    }
}

impl<M: StoreMemory> Drop for DeviceKvStore<M> {
    fn drop(&mut self) {
        self.release_buffers();
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// The dequantise round-trip bytes a valid entry represents.
fn canonical_sha256(entry: StoreEntry, stored: &[u8]) -> Result<[u8; 32], StoreError> {
    match entry {
        StoreEntry::MainKv { .. } => {
            let parsed = MainKvEntry::from_bytes(stored)?;
            let mut values = [0.0f32; MAIN_KV_DIM];
            dequantize_main_kv(&parsed, &mut values)?;
            let mut bytes = Vec::with_capacity(MAIN_KV_DIM * 4);
            for value in values {
                bytes.extend_from_slice(&canonical_f32(value));
            }
            Ok(sha256(&bytes))
        }
        StoreEntry::IndexerK { .. } => {
            let parsed = IndexerKEntry::from_bytes(stored)?;
            let mut values = [0.0f32; INDEXER_K_DIM];
            dequantize_indexer_k(&parsed, &mut values)?;
            let mut bytes = Vec::with_capacity(INDEXER_K_DIM * 4);
            for value in values {
                bytes.extend_from_slice(&canonical_f32(value));
            }
            Ok(sha256(&bytes))
        }
        StoreEntry::Window { .. } => {
            let mut parsed = SwaKvSlot::default();
            parsed.data.copy_from_slice(&stored[..SWA_SLOT_DATA_BYTES]);
            parsed
                .scales
                .copy_from_slice(&stored[SWA_SLOT_DATA_BYTES..]);
            let mut values = [0.0f32; SWA_SLOT_DATA_BYTES];
            dequantize_swa_slot(&parsed, &mut values)?;
            let mut bytes = Vec::with_capacity(SWA_SLOT_DATA_BYTES * 2);
            for value in values {
                bytes.extend_from_slice(&bf16_round(value).to_le_bytes());
            }
            Ok(sha256(&bytes))
        }
    }
}

/// Canonical f32 bytes: NaN is normalised so the host and the node hash the same digest.
fn canonical_f32(value: f32) -> [u8; 4] {
    if value.is_nan() {
        0x7fc0_0000u32.to_le_bytes()
    } else {
        value.to_bits().to_le_bytes()
    }
}

/// Round an f32 to bf16, round-to-nearest-even, NaN normalised.
fn bf16_round(value: f32) -> u16 {
    if value.is_nan() {
        return 0x7fc0;
    }
    let bits = value.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(round)) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::GLOBAL_ENTRY_BYTES;

    const LIVE_TOKENS: usize = 1_048_576;

    #[test]
    fn live_geometry_matches_the_design_note() {
        let geometry = StoreGeometry::for_max_seq_len(LIVE_TOKENS).expect("live geometry");
        assert_eq!(geometry.max_seq_len(), LIVE_TOKENS);
        assert_eq!(geometry.pages(), 16_384);
        let shapes: Vec<(usize, usize, u64, u64)> = geometry
            .sources()
            .iter()
            .map(|source| {
                (
                    source.layer,
                    source.ratio,
                    source.page_entries,
                    source.entries,
                )
            })
            .collect();
        assert_eq!(
            shapes,
            vec![
                (2, 2, 32, 524_288),
                (8, 2, 32, 524_288),
                (14, 2, 32, 524_288),
                (20, 1, 64, 1_048_576),
            ]
        );
        assert_eq!(geometry.total_entries(), 2_621_440);
        assert_eq!(geometry.main_kv_data_bytes(), 671_088_640);
        assert_eq!(geometry.main_kv_scale_bytes(), 83_886_080);
        assert_eq!(geometry.indexer_k_data_bytes(), 167_772_160);
        assert_eq!(geometry.indexer_k_scale_bytes(), 10_485_760);
        assert_eq!(geometry.global_kv_bytes(), 933_232_640);
        assert_eq!(geometry.window_bytes(), 2_703_360);
        assert_eq!(geometry.handle_bytes(), 4_196_352);
        assert_eq!(geometry.total_bytes(), 940_132_352);
        assert_eq!(geometry.kv_bytes_per_token(), 890.0);
    }

    #[test]
    fn geometry_rounds_up_to_whole_pages() {
        let one = StoreGeometry::for_max_seq_len(1).expect("one token");
        assert_eq!(one.pages(), 1);
        assert_eq!(one.max_seq_len(), 64);
        let cross = StoreGeometry::for_max_seq_len(65).expect("65 tokens");
        assert_eq!(cross.pages(), 2);
        assert_eq!(cross.max_seq_len(), 128);
        assert!(matches!(
            StoreGeometry::for_max_seq_len(0),
            Err(StoreError::EmptyGeometry)
        ));
    }

    #[test]
    fn buffer_plan_sums_to_the_total() {
        let geometry = StoreGeometry::for_max_seq_len(LIVE_TOKENS).expect("live geometry");
        let plan = geometry.buffer_plan();
        let names: Vec<&str> = plan.iter().map(|spec| spec.name).collect();
        assert_eq!(
            names,
            vec![
                "kv_fp4",
                "kv_scales",
                "index_k_fp4",
                "index_k_scales",
                "window_codes",
                "window_scales",
                "hsi_pool",
                "selected",
            ]
        );
        let sum: u64 = plan.iter().map(|spec| spec.bytes).sum();
        assert_eq!(sum, geometry.total_bytes());
        assert_eq!(plan[6].bytes, HSI_POOL_BYTES as u64);
        assert_eq!(plan[7].bytes, SELECTED_BYTES as u64);
    }

    #[test]
    fn page_manifest_has_one_row_per_stored_entry() {
        let geometry = StoreGeometry::for_max_seq_len(LIVE_TOKENS).expect("live geometry");
        let payload = vec![0x5au8; GLOBAL_PAGE_BYTES];
        let rows = geometry.page_manifest(0, &payload).expect("manifest");
        assert_eq!(rows.len(), GLOBAL_PAGE_ENTRIES * 2);
        assert_eq!(
            rows[0].entry,
            StoreEntry::MainKv {
                layer: 2,
                position: 0
            }
        );
        assert_eq!(
            rows[1].entry,
            StoreEntry::IndexerK {
                layer: 2,
                position: 0
            }
        );
        assert_eq!(
            rows.last().expect("rows").entry,
            StoreEntry::IndexerK {
                layer: 20,
                position: 63
            }
        );
        assert_eq!(
            GLOBAL_ENTRY_BYTES,
            MAIN_KV_ENTRY_BYTES + INDEXER_K_ENTRY_BYTES
        );
    }

    #[test]
    fn audit_manifest_is_page_bound() {
        let geometry = StoreGeometry::for_max_seq_len(LIVE_TOKENS).expect("live geometry");
        let payload = vec![0u8; GLOBAL_PAGE_BYTES];
        assert!(matches!(
            geometry.page_manifest(geometry.pages(), &payload),
            Err(StoreError::OutOfBounds {
                what: "page",
                value: 16_384,
                bound: 16_384,
            })
        ));
        assert!(matches!(
            geometry.page_manifest(0, &payload[..GLOBAL_PAGE_BYTES - 1]),
            Err(StoreError::InvalidLength { .. })
        ));
        assert!(
            geometry
                .page_manifest(geometry.pages() - 1, &payload)
                .is_ok()
        );
    }

    #[test]
    fn bf16_rounding_is_round_to_nearest_even() {
        assert_eq!(bf16_round(1.0), 0x3f80);
        assert_eq!(bf16_round(-2.0), 0xc000);
        assert_eq!(bf16_round(f32::NAN), 0x7fc0);
        let tie = 1.0f32 + 2f32.powi(-8);
        assert_eq!(bf16_round(tie), 0x3f80);
        let up = 1.0f32 + 5.0 * 2f32.powi(-9);
        assert_eq!(bf16_round(up), 0x3f81);
    }

    #[test]
    fn entry_stored_sizes_are_the_pinned_layout() {
        assert_eq!(
            StoreEntry::MainKv {
                layer: 2,
                position: 0
            }
            .stored_bytes(),
            288
        );
        assert_eq!(
            StoreEntry::IndexerK {
                layer: 2,
                position: 0
            }
            .stored_bytes(),
            68
        );
        assert_eq!(StoreEntry::Window { layer: 0, slot: 0 }.stored_bytes(), 528);
    }
}

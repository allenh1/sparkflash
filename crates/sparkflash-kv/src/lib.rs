//! Three-tier radix prefix cache, CSA2 layer topology and SWA Bounded Replay.
//!
//! DeepSeek-V4.1-Flash separates two KV structures:
//!
//! * **Global KV**, stored only by the four Full-mode layers (2, 8, 14, 20) as an FP4 E2M1
//!   MLA latent plus an MXFP4 indexer key: 890 logical bytes per token across the model.
//! * **SWA KV**, a 128-token sliding window present in all 40 layers.
//!
//! The Causal Encoder-Decoder split means the decoder's global KV is projected from the
//! final encoder hidden states, so a prompt never needs to traverse the decoder layers in
//! full: only the last SWA_WINDOW_SIZE tokens do. Everything in this crate that decides
//! *how many* tokens run where is pure arithmetic and therefore testable without weights.
//!
//! # The storage half
//!
//! On top of that planning layer this crate implements the real cache:
//!
//! * crate::radix indexes whole 64-token pages in a radix tree with per-page reference
//!   counting, so a page a live sequence is reading can never be evicted;
//! * crate::tier holds the three tiers behind a trait, with an in-memory backend for tests
//!   and coherent UMA, and an O_DIRECT file backend whose alignment is probed from the
//!   device rather than hardcoded, plus the promotion/demotion policy that joins them;
//! * crate::fp4 is the weight-free reference codec that defines the 356-byte entry layout
//!   behind the specification's 890 bytes per token;
//! * crate::swa is the 128-slot prefix-tail ring and the bounded-replay accounting;
//! * crate::cache wires them together. Its replay methods hold only the SWA lock, and the
//!   store refuses a global-KV write from a replaying thread on both the publish path and
//!   every backend it owns, so a bounded replay cannot touch cached global KV even by
//!   accident.

pub mod cache;
pub mod config;
#[cfg(feature = "cuda")]
pub mod device_cuda;
pub mod device_store;
pub mod error;
pub mod fp4;
pub mod layout;
pub mod plan;
pub mod radix;
pub mod swa;
pub mod telemetry;
pub mod tier;

pub use cache::{
    CacheOutcome, KvCache, KvCacheConfig, KvCacheSnapshot, Sequence, replay_layer_count,
    usable_cached_tokens,
};
pub use config::{
    CANDIDATE_BLOCK_SIZE, CANDIDATE_POSITIONS, CANDIDATE_SOURCE_LAYER, CANDIDATE_TOPK_BLOCKS,
    COMPRESS_RATIOS, Csa2Mode, GLOBAL_KV_BYTES_PER_TOKEN, INDEX_SOURCE_LAYERS, INDEX_TOPK,
    KV_PAGE_TOKENS, KV_SOURCE_LAYERS, NUM_HIDDEN_LAYERS, SWA_WINDOW_SIZE, compress_ratio,
    csa2_mode, index_source_for, kv_source_for,
};
pub use device_store::{
    ArrayHandle, AuditRow, BufferSpec, DeviceKvStore, SelectedHandle, SourceHandles, SourceShape,
    StoreEntry, StoreError, StoreGeometry, StoreMemory, WindowHandles,
};
pub use error::KvError;
pub use fp4::{
    E2M1_MAX, E4M3_MAX, E8M0_MAX, IndexerKEntry, MainKvEntry, SwaKvSlot, dequantize_indexer_k,
    dequantize_main_kv, dequantize_swa_slot, e2m1_decode, e2m1_encode, e4m3_decode, e4m3_encode,
    e8m0_decode, e8m0_encode_pow2_ceil, quantize_indexer_k, quantize_main_kv, quantize_swa_slot,
};
pub use layout::{
    DEFAULT_GLOBAL_KV_POOL_BYTES, DEFAULT_SWA_TAIL_SLOTS, GIB, GLOBAL_ENTRY_BYTES,
    GLOBAL_PAGE_BYTES, GLOBAL_PAGE_ENTRIES, INDEXER_K_DIM, INDEXER_K_ENTRY_BYTES, MAIN_KV_DIM,
    MAIN_KV_ENTRY_BYTES, MEMORY_FLOOR_BYTES, MemoryBudget, NODE_MEMORY_BUDGET_BYTES, PageId,
    SPEC_SAFETY_HEADROOM_BYTES, SWA_LAYERS, SWA_SLOT_BYTES, SWA_SLOTS_PER_TAIL, SWA_TAIL_BYTES,
    align_down, align_up, derived_global_kv_bytes_per_token, entries_per_page,
    global_kv_bytes_for_tokens, page_entry_offset, page_slot_offset, pages_for_tokens, source_slot,
    swa_layer_range, swa_slot_offset,
};
pub use plan::{KvCacheStatsSnapshot, KvCacheTelemetry, PrefixMatchPlan};
pub use radix::{InsertOutcome, NodeId, PrefixMatch, RadixTree, block_hash};
pub use swa::{
    ALL_SWA_LAYERS, ReplayKind, ReplayOutcome, SwaProducer, SwaTail, SwaTailRing, TailId, tail_hash,
};
pub use telemetry::{KvStoreDelta, KvStoreSnapshot, KvStoreTelemetry};
pub use tier::{
    FileBackedTier, InMemoryTier, PageLocation, Tier, TierBackend, TieredPageStore,
    probe_direct_alignment,
};

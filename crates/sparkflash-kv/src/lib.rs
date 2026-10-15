//! Three-tier radix prefix cache, CSA2 layer topology and SWA Bounded Replay planning.
//!
//! DeepSeek-V4.1-Flash separates two KV structures:
//!
//! * **Global KV**, stored only by the four Full-mode layers (2, 8, 14, 20) as an FP4 E2M1
//!   MLA latent plus an MXFP4 indexer key: 890 logical bytes per token across the model.
//! * **SWA KV**, a 128-token sliding window present in all 40 layers.
//!
//! The Causal Encoder-Decoder split means the decoder's global KV is projected from the
//! final encoder hidden states, so a prompt never needs to traverse the decoder layers in
//! full: only the last `SWA_WINDOW_SIZE` tokens do. Everything in this crate that decides
//! *how many* tokens run where is pure arithmetic and therefore testable without weights.

pub mod config;
pub mod plan;

pub use config::{
    CANDIDATE_BLOCK_SIZE, CANDIDATE_POSITIONS, CANDIDATE_SOURCE_LAYER, CANDIDATE_TOPK_BLOCKS,
    COMPRESS_RATIOS, Csa2Mode, GLOBAL_KV_BYTES_PER_TOKEN, INDEX_SOURCE_LAYERS, INDEX_TOPK,
    KV_PAGE_TOKENS, KV_SOURCE_LAYERS, NUM_HIDDEN_LAYERS, SWA_WINDOW_SIZE, compress_ratio,
    csa2_mode, index_source_for, kv_source_for,
};
pub use plan::{KvCacheStatsSnapshot, KvCacheTelemetry, PrefixMatchPlan};

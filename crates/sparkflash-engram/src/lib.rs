//! Deterministic Engram conditional-memory engine.
//!
//! DeepSeek-V4.1-Flash augments two backbone layers (1 and 14) with a 189 GiB n-gram
//! conditional-memory table. Per token the layer performs 24 lookups (3 n-gram orders x 8
//! hash heads), gathers 8 x 256-dimensional embeddings per lookup, and mixes the resulting
//! 3 x 2048-dimensional vectors into the residual stream through a learned gate.
//!
//! Milestone 1 provides the weight-free geometry and telemetry. Milestone 2 adds the
//! tokenizer compression map, the deterministic multiplicative-XOR hash heads, the
//! DRM/UMA/NVMe tiered cache and the io_uring O_DIRECT store.

pub mod layout;
pub mod telemetry;

pub use layout::{
    COMPRESSED_VOCAB_SIZE, ENGRAM_HEAD_DIM, ENGRAM_LAYERS, ENGRAM_NUM_EMBEDDINGS, ENGRAM_NUM_HEADS,
    ENGRAM_ORDERS, ENGRAM_PAD_TOKEN_ID, ENGRAM_ROW_BYTES, ENGRAM_SCALE_BYTES,
    ENGRAM_TOTAL_LOOKUPS_PER_TOK, IMAGE_TOKEN_ID, SCALE_BLOCK, num_embeddings, row_offset,
    row_range_for_rank, rows_per_rank, total_bytes,
};
pub use telemetry::{EngramLayerTelemetry, EngramTelemetrySnapshot, GATE_ALPHA_SCALE};

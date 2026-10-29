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

//! Deterministic Engram conditional-memory engine.
//!
//! **Attribution:** Bounded `O_DIRECT` / `io_uring` NVMe row gathering and unified-memory
//! caching for out-of-core Engram/PLE tables on DGX Spark (`GB10`) draw on prior art from
//! [`b12x`](https://github.com/local-inference-lab/b12x),
//! [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference) (`helios/src/engram/reader.rs`),
//! [`MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks),
//! and [`knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4`](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4).
//! See the root `NOTICE` file for full provenance details.
//!
//! DeepSeek-V4.1-Flash augments two backbone layers (1 and 14) with a 189 GiB n-gram
//! conditional-memory table. Per token the layer performs 24 lookups (3 n-gram orders x 8
//! hash heads), gathers 8 x 256-dimensional embeddings per lookup, and mixes the resulting
//! 3 x 2048-dimensional vectors into the residual stream through a learned gate.
//!
//! Milestone 1 provides the weight-free geometry and telemetry. Milestone 2 adds:
//!
//! * primes - the 48 bucket primes, their 24 per-layer offsets and the 8 hash multipliers,
//!   all re-derived from scratch by tests/prime_layout.rs (engram-hash).
//! * compress - the surjective tokenizer map from 129,280 raw ids onto 99,092 compressed ids,
//!   which is also the number every multiplier is derived from (engram-hash).
//! * hash - the 24-column rolling multiplicative-XOR hash, including the DEAD/image-span
//!   rule and the prefill/decode cache (engram-hash).
//! * packer, store, cache, prefetch - the 512-byte aligned per-rank table files, the O_DIRECT
//!   reader, the 16-way set-associative UMA/DRAM cache and the io_uring prefetch worker
//!   (engram-store).
//!
//! Every module is testable without model weights: the geometry, the prime draw and the hash
//! are checked against Python-generated golden vectors, and the store is exercised against
//! synthetic tables and temp files.

pub mod cache;
pub mod compress;
pub mod hash;
pub mod layout;
pub mod packer;
pub mod paged;
pub mod prefetch;
pub mod primes;
pub mod store;
pub mod telemetry;

pub use layout::{
    COMPRESSED_VOCAB_SIZE, ENGRAM_HEAD_DIM, ENGRAM_LAYERS, ENGRAM_NUM_EMBEDDINGS, ENGRAM_NUM_HEADS,
    ENGRAM_ORDERS, ENGRAM_PAD_TOKEN_ID, ENGRAM_ROW_BYTES, ENGRAM_SCALE_BYTES,
    ENGRAM_TOTAL_LOOKUPS_PER_TOK, IMAGE_TOKEN_ID, SCALE_BLOCK, num_embeddings, row_offset,
    row_range_for_rank, rows_per_rank, total_bytes,
};
pub use telemetry::{EngramLayerTelemetry, EngramTelemetrySnapshot, GATE_ALPHA_SCALE};

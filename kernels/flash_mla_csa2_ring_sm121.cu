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

// kernels/flash_mla_csa2_ring_sm121.cu
//
// The D49 ring-capacity CSA2 entry: the same position-causal, format-2 (CSA union) attention as
// flash_mla_csa2_seq_sm121, but the KV ring capacity is an explicit argument separate from the
// 128-token arithmetic window. The DSpark/MTP ring holds 134 slots (window 128 + nextn + 1), so
// slot = position % ring_capacity while the attended window stays the last 128 positions and the
// union score array is bounded by 512 candidates + ring_capacity = 646. The frozen 19-word sequence
// entry is not mutated; this symbol has its own ABI row, source file, cubin pin and KernelId.
//
// The deferred sequence's position-causal CSA2 entry. It is the same format-2 (CSA union) attention
// as kernels/flash_mla_csa2_sm121.cu - window rows supplied oldest first, selected compressed ids as
// absolute token positions masked by candidate <= position, one f32 softmax with the sink in the
// denominator after the maximum - plus a position-causal INDEXER stage:
//
//   * stage 1 (indexer scoring) only scores high-speed-index tokens at or behind the query. A block
//     whose first token is past the query publishes block_scores = -FLT_MAX and block_best = 0.
//     The base entry keeps its measured position-0 behaviour (unmasked); this entry exists so a
//     per-position sequence caller can use block_best for candidate selection without leaking
//     future tokens.
//   * stage 2 (the union attention) is byte-for-byte the base entry's arithmetic. At position 0,
//     with the same operands, this entry is expected to reproduce the base entry's out[] exactly;
//     the m20 harness measures that, and the base entry/pin/artifact are untouched so the
//     b31/b33/b39/b40 evidence keeps referring to the same cubin.
//
// Scope is deliberately the same as the base kernel: the candidates arrive already selected; the
// top-512 merge over the 2,048 block candidates is still a separate pipeline step and is NOT
// implemented here. Rotary position embedding is assumed already applied by the caller. The FP4
// latent is dequantized in the scalar path. No tensor-core path, no fuse with the projections, no
// host-side validation beyond the entry guards: format must be exactly 2, ring_capacity must be
// 1..=134 with window_rows <= ring_capacity, compression_ratio must be non-zero, and the window
// plus selected counts must fit the shared score array. The ring slot itself is the caller's
// position % ring_capacity; this entry receives the already-gathered window rows oldest first.
//
// Verification status: source added for D49, compiled and pinned, and parsed by the clang
// stub-header probe. No device run and no numeric claim: m20 exercises the frozen sequence entry,
// not this symbol. The D49 gate needs a device run of the 134-slot ring before any serving claim.

#include "m4_common_sm121.cuh"

#define SPARK_M4_MLA_THREADS 256u
#define SPARK_M4_MLA_LATENT SPARK_M4_KV_LORA_RANK
#define SPARK_M4_MLA_FP4_PER_BYTE 2u
// One score per selected position lives in shared memory, which is what bounds the candidate
// count: 512 positions of f32 plus the running statistics. In union format the window rows are
// attended in the same array, so its capacity is one candidate list plus one full ring. The ring
// capacity is written as its own symbol (no C cast) so the Rust reader evaluates the same sum.
#define SPARK_M4_MLA_MAX_SELECTED 512u
#define SPARK_M4_MLA_RING_CAPACITY 134u
#define SPARK_M4_MLA_RING_MAX_ATTENDED (SPARK_M4_MLA_MAX_SELECTED + SPARK_M4_MLA_RING_CAPACITY)

static_assert(SPARK_M4_MLA_LATENT % SPARK_M4_KV_SCALE_CHANNELS == 0u,
              "the latent divides into 16-channel scale groups");
static_assert(SPARK_M4_MLA_LATENT % SPARK_M4_MLA_THREADS == 0u, "the latent tiles across the block");
static_assert(SPARK_M4_MLA_LATENT / SPARK_M4_MLA_THREADS == 2u, "two latent values per thread");
static_assert(SPARK_M4_MLA_MAX_SELECTED == SPARK_M4_INDEX_TOPK, "the candidate list is index_topk");
static_assert(SPARK_M4_MLA_RING_MAX_ATTENDED == 646u,
              "the union capacity is 512 candidates plus the 134-slot ring");
static_assert(SPARK_M4_MLA_RING_MAX_ATTENDED >= SPARK_M4_MLA_MAX_SELECTED,
              "the union array must hold every historical candidate list");
static_assert(SPARK_M4_MLA_RING_CAPACITY == 134u,
              "the D49 ring holds window 128 + nextn + 1 = 134 slots");
static_assert(SPARK_M4_MLA_RING_CAPACITY >= SPARK_M4_SWA_WINDOW,
              "the ring must be able to hold the whole arithmetic window");
static_assert(SPARK_M4_KV_SCALE_CHANNELS == 16u, "an E4M3 scale per 16 channels");
static_assert(SPARK_M4_SWA_WINDOW == 128u, "the sliding window is 128 tokens");
static_assert(SPARK_M4_MLA_THREADS == SPARK_M4_REDUCE_SLOTS, "the block reduction needs 256 threads");
// The query heads and the indexer heads are two different index spaces: 64 query heads, 32 indexer
// heads (8 per rank), so one indexer head serves two query heads and the indexer query must be
// addressed as head / SPARK_M4_MLA_HEADS_PER_INDEX_HEAD. Using the query head directly would read
// past the end of a [32][128] index query for half the blocks and give the wrong head's score for
// the rest. SPARK_M4_MLA_SCORE_SCALE is the MLA score scale, defined once in the shared header.
static_assert(SPARK_M4_HEADS == 64u, "64 query heads");
static_assert(SPARK_M4_INDEX_HEADS == 32u, "32 indexer heads");
static_assert(SPARK_M4_MLA_HEADS_PER_INDEX_HEAD == 2u, "two query heads per indexer head");
// No C cast in a static_assert: the Rust reader evaluates these expressions itself, and a
// parenthesised type name would be an identifier it cannot resolve.
static_assert(SPARK_M4_HSI_BLOCKS * SPARK_M4_HSI_BLOCK_TOKENS == SPARK_M4_HSI_POSITIONS,
              "the HSI pool is 2048 blocks of 8 tokens");

// One warp scores one HSI block. The indexer's dot is 128 values wide, four per lane, so a
// five-step __shfl_down_sync tree reduces it with no shared memory and no block barrier. The old
// shape reduced the same 128 lanes inside a 256-thread shared-memory tree: nine __syncthreads()
// per slot-dot and half the threads contributing nothing, which made the indexer a barrier chain
// rather than arithmetic. Eight warps now score eight independent HSI blocks at once, and the
// blockIdx.y slice strides in units of eight. The shuffle is declared by the clang syntax stub as
// well, so the host-side ABI probe still parses.
#define SPARK_M4_MLA_WARP_LANES 32u
#define SPARK_M4_MLA_WARPS (SPARK_M4_MLA_THREADS / SPARK_M4_MLA_WARP_LANES)
static_assert(SPARK_M4_MLA_THREADS % SPARK_M4_MLA_WARP_LANES == 0u,
              "the block divides into whole warps");
static_assert(SPARK_M4_INDEX_HEAD_DIM == 4u * SPARK_M4_MLA_WARP_LANES,
              "four indexer dimensions per lane");


// The engine's released window-KV format (model.py:700-707, kernel.py:98-124): one E4M3 code per
// latent channel with one E8M0 power-of-two scale per 32 channels, stored after the engine's
// in-place quantise/dequantise round trip, which returns the input dtype and therefore stores bf16.
// The format is an explicit runtime argument, never inferred from another argument's value:
// 0 keeps the historical FP4 latent path byte-identical, 1 reads the engine window arrays, and
// 2 (CSA union) attends the engine window rows plus the causal FP4 compressed candidates in one
// softmax. Callers pass only 0, 1 or 2; any other value is poisoned at kernel entry. The format
// VALUE cannot be a static_assert because it is a runtime parameter; only the layout constants
// below are compile-time facts.
#define SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4 0u
#define SPARK_M4_MLA_WINDOW_KV_FORMAT_ENGINE_FP8 1u
#define SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION 2u
#define SPARK_M4_MLA_WINDOW_QUANT_BLOCK 32u
#define SPARK_M4_MLA_WINDOW_SCALES (SPARK_M4_MLA_LATENT / SPARK_M4_MLA_WINDOW_QUANT_BLOCK)
#include "flash_mla_csa2_common_sm121.cuh"

// The shared header's helpers, under the names this entry used.
#define spark_mla_ring_warp_sum spark_mla_warp_sum
#define spark_mla_ring_in_window spark_mla_in_window
#define spark_mla_ring_load_latent spark_mla_load_latent
#define spark_mla_ring_scale spark_mla_scale
#define spark_mla_ring_load_window_stored spark_mla_load_window_stored
#define spark_mla_ring_attends spark_mla_attends
#define spark_mla_ring_union_attends spark_mla_union_attends
static_assert(SPARK_M4_MLA_LATENT % SPARK_M4_MLA_WINDOW_QUANT_BLOCK == 0u,
              "the engine window row is a whole number of 32-channel scale blocks");
static_assert(SPARK_M4_MLA_WINDOW_SCALES == 16u,
              "one E8M0 scale per 32 channels is sixteen bytes for a 512-wide row");

// A position is attendable when it is inside the sliding window behind the query. The predicate
// is a function of the query position and the candidate id, so the CPU reference can test it
// directly instead of duplicating the comparison.


// The E4M3 scale that covers latent channel index: one per 16 channels, stored as bytes.

// One latent channel of one engine-format window row: the E4M3 code times its 32-channel E8M0
// power-of-two scale, in f32, rounded to bf16. The round is load-bearing: the engine's window cache
// holds bf16(dequant(code, scale)) because model.py:707 quantises in place and returns the input
// dtype, and this kernel and src/mla_ref.rs both apply the same round so they consume the same
// stored values.

// Whether one selected id is attendable in the selected format. FP4 candidates are token positions
// filtered by the sliding-window predicate; engine-format candidates are row indices into the
// supplied window rows (oldest first), so any id at or past window_rows is out of window and gets
// the same excluded score, which makes its contribution exactly zero in every pass.

// Union format: the window rows are attended unconditionally (the caller supplies only rows at or
// behind the query, oldest first), and a selected compressed id is attended when it is not in the
// future. The reference's reachability mask is the same comparison for a ratio-1 compressed store;
// a ratio-2 store reaches this kernel as a per-position expanded view (row p = store row p/ratio),
// so an absolute candidate id is still a row index here. A candidate inside the sliding window is
// deliberately NOT filtered out: the union is the released attention's whole candidate set.

extern "C" __global__ void flash_mla_csa2_ring_sm121(
    const __nv_bfloat16 *__restrict__ q,
    const uint8_t *__restrict__ kv_fp4,
    const uint8_t *__restrict__ kv_scales,
    const uint8_t *__restrict__ window_codes,
    const uint8_t *__restrict__ window_scales,
    const uint32_t window_kv_format,
    const uint32_t window_rows,
    const uint32_t ring_capacity,
    const uint32_t compression_ratio,
    const uint32_t *__restrict__ selected,
    const __nv_bfloat16 *__restrict__ index_key,
    const __nv_bfloat16 *__restrict__ index_query,
    const float *__restrict__ sink,
    float *__restrict__ block_scores,
    uint32_t *__restrict__ block_best,
    __nv_bfloat16 *__restrict__ out,
    const uint32_t position,
    const uint32_t selected_count,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    spark_mla_attention_body<SparkMlaRing>(
        q, kv_fp4, kv_scales, window_codes, window_scales, window_kv_format, window_rows,
        ring_capacity, compression_ratio, selected, index_key, index_query, sink, block_scores,
        block_best, out, position, selected_count, trace_ring, layer_id, step_seq);
}

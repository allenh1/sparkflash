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

// kernels/flash_mla_csa2_seq_sm121.cu
//
// The deferred sequence's position-causal CSA2 entry. It is the same format-2 (CSA union) attention
// as kernels/flash_mla_csa2_sm121.cu - window rows supplied oldest first, selected compressed ids as
// absolute token positions masked by candidate <= position, one f32 softmax with the sink in the
// denominator after the maximum - plus a position-causal INDEXER stage:
//
//   * stage 1 (indexer scoring) only scores high-speed-index tokens at or behind the query. A block
//     whose first token is past the query publishes block_scores = -FLT_MAX and block_best = 0.
//     The published block arrays are index head 0's: the frozen ABI carries one [hsi_blocks] pair,
//     not one per index head, so only index head 0 writes it (its two query heads split the blocks).
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
// host-side validation beyond the entry guards: format must be exactly 2, and the window plus
// selected counts must fit the shared score array.
//
// Verification status: source added for the c4a-2 start_pos > 0 extension, compiled and pinned, and
// exercised by examples/m20_csa2_sequence.rs (device vs mla_ref::attention_with_sink_union per
// position, position-0 byte identity against the base entry, causal indexer twin, determinism).
// The window record lives in docs/archive/build/phase3-csa2-sequence.md; until that window reports, this file
// carries no device measurement.

#include "m4_common_sm121.cuh"

#define SPARK_M4_MLA_THREADS 256u
#define SPARK_M4_MLA_LATENT SPARK_M4_KV_LORA_RANK
#define SPARK_M4_MLA_FP4_PER_BYTE 2u
// One score per selected position lives in shared memory, which is what bounds the candidate
// count: 512 positions of f32 plus the running statistics. In union format the window rows are
// attended in the same array, so its capacity is one candidate list plus one full window. The
// capacity is written as that expression (no C cast) so the Rust reader evaluates the same sum.
#define SPARK_M4_MLA_MAX_SELECTED 512u
#define SPARK_M4_MLA_MAX_ATTENDED (SPARK_M4_MLA_MAX_SELECTED + SPARK_M4_SWA_WINDOW)

static_assert(SPARK_M4_MLA_LATENT % SPARK_M4_KV_SCALE_CHANNELS == 0u,
              "the latent divides into 16-channel scale groups");
static_assert(SPARK_M4_MLA_LATENT % SPARK_M4_MLA_THREADS == 0u, "the latent tiles across the block");
static_assert(SPARK_M4_MLA_LATENT / SPARK_M4_MLA_THREADS == 2u, "two latent values per thread");
static_assert(SPARK_M4_MLA_MAX_SELECTED == SPARK_M4_INDEX_TOPK, "the candidate list is index_topk");
static_assert(SPARK_M4_MLA_MAX_ATTENDED == 640u,
              "the union capacity is 512 candidates plus the 128-token window");
static_assert(SPARK_M4_MLA_MAX_ATTENDED >= SPARK_M4_MLA_MAX_SELECTED,
              "the union array must hold every historical candidate list");
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

__device__ __forceinline__ float spark_mla_seq_warp_sum(float value) {
    value += __shfl_down_sync(0xffffffffu, value, 16u);
    value += __shfl_down_sync(0xffffffffu, value, 8u);
    value += __shfl_down_sync(0xffffffffu, value, 4u);
    value += __shfl_down_sync(0xffffffffu, value, 2u);
    value += __shfl_down_sync(0xffffffffu, value, 1u);
    return value;
}

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
static_assert(SPARK_M4_MLA_LATENT % SPARK_M4_MLA_WINDOW_QUANT_BLOCK == 0u,
              "the engine window row is a whole number of 32-channel scale blocks");
static_assert(SPARK_M4_MLA_WINDOW_SCALES == 16u,
              "one E8M0 scale per 32 channels is sixteen bytes for a 512-wide row");

// A position is attendable when it is inside the sliding window behind the query. The predicate
// is a function of the query position and the candidate id, so the CPU reference can test it
// directly instead of duplicating the comparison.
__device__ __forceinline__ uint32_t spark_mla_seq_in_window(uint32_t query_position, uint32_t candidate) {
    return query_position >= candidate && (query_position - candidate) < (uint32_t)SPARK_M4_SWA_WINDOW;
}

__device__ __forceinline__ float spark_mla_seq_load_latent(const uint8_t *row, uint32_t index) {
    const uint8_t packed = row[index / SPARK_M4_MLA_FP4_PER_BYTE];
    const uint8_t nibble = (index % SPARK_M4_MLA_FP4_PER_BYTE) == 0u ? (uint8_t)(packed & 0x0Fu)
                                                                    : (uint8_t)(packed >> 4);
    return spark_fp4_e2m1_decode(nibble);
}

// The E4M3 scale that covers latent channel index: one per 16 channels, stored as bytes.
__device__ __forceinline__ float spark_mla_seq_scale(const uint8_t *scales, uint32_t index) {
    return spark_fp8_e4m3_decode(scales[index / SPARK_M4_KV_SCALE_CHANNELS]);
}

// One latent channel of one engine-format window row: the E4M3 code times its 32-channel E8M0
// power-of-two scale, in f32, rounded to bf16. The round is load-bearing: the engine's window cache
// holds bf16(dequant(code, scale)) because model.py:707 quantises in place and returns the input
// dtype, and this kernel and src/mla_ref.rs both apply the same round so they consume the same
// stored values.
__device__ __forceinline__ float spark_mla_seq_load_window_stored(const uint8_t *window_codes,
                                                              const uint8_t *window_scales,
                                                              uint32_t row,
                                                              uint32_t index) {
    const float code = spark_fp8_e4m3_decode(window_codes[(uint64_t)row * SPARK_M4_MLA_LATENT + index]);
    const float scale = spark_ue8m0_decode(
        window_scales[(uint64_t)row * SPARK_M4_MLA_WINDOW_SCALES + index / SPARK_M4_MLA_WINDOW_QUANT_BLOCK]);
    return __bfloat162float(__float2bfloat16_rn(code * scale));
}

// Whether one selected id is attendable in the selected format. FP4 candidates are token positions
// filtered by the sliding-window predicate; engine-format candidates are row indices into the
// supplied window rows (oldest first), so any id at or past window_rows is out of window and gets
// the same excluded score, which makes its contribution exactly zero in every pass.
__device__ __forceinline__ uint32_t spark_mla_seq_attends(uint32_t window_kv_format,
                                                      uint32_t position,
                                                      uint32_t candidate,
                                                      uint32_t window_rows) {
    return window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4
               ? spark_mla_seq_in_window(position, candidate)
               : (candidate < window_rows ? 1u : 0u);
}

// Union format: the window rows are attended unconditionally (the caller supplies only rows at or
// behind the query, oldest first), and a selected compressed id is attended when it is not in the
// future. The reference's reachability mask is the same comparison for a ratio-1 compressed store;
// a ratio-2 store reaches this kernel as a per-position expanded view (row p = store row p/ratio),
// so an absolute candidate id is still a row index here. A candidate inside the sliding window is
// deliberately NOT filtered out: the union is the released attention's whole candidate set.
__device__ __forceinline__ uint32_t spark_mla_seq_union_attends(uint32_t position, uint32_t candidate) {
    return candidate <= position ? 1u : 0u;
}

extern "C" __global__ void flash_mla_csa2_seq_sm121(
    const __nv_bfloat16 *__restrict__ q,             // [heads][latent] query, rope already applied
    const uint8_t *__restrict__ kv_fp4,              // [positions][latent/2] packed E2M1 latent
    const uint8_t *__restrict__ kv_scales,           // [positions][latent/16] E4M3 scales
    const uint8_t *__restrict__ window_codes,        // engine format: [window_rows][latent] E4M3 codes
    const uint8_t *__restrict__ window_scales,       // engine format: [window_rows][latent/32] E8M0 scales
    const uint32_t window_kv_format,                 // 0 = FP4 window, 1 = engine FP8 window, 2 = CSA union
    const uint32_t window_rows,                      // engine format: window rows supplied, oldest first
    const uint32_t *__restrict__ selected,           // [selected_count] candidate position ids (2: absolute token positions)
    const __nv_bfloat16 *__restrict__ index_key,     // [hsi_blocks][8][index_head_dim]
    const __nv_bfloat16 *__restrict__ index_query,   // [index_heads][index_head_dim]
    const float *__restrict__ sink,                  // [heads] additive per-head attention sink logit
    float *__restrict__ block_scores,                // [hsi_blocks] out: best score in the block
    uint32_t *__restrict__ block_best,               // [hsi_blocks] out: the token that scored it
    __nv_bfloat16 *__restrict__ out,                 // [heads][latent] attention output
    const uint32_t position,
    const uint32_t selected_count,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t head = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t heads = (uint32_t)SPARK_M4_HEADS;
    // The indexer head that serves this query head: the two are different index spaces (64 query
    // heads, 32 indexer heads), so this division is required, not cosmetic.

    __shared__ float s_scores[SPARK_M4_MLA_MAX_ATTENDED];
    __shared__ float s_q[SPARK_M4_MLA_LATENT];
    __shared__ uint32_t s_candidates[SPARK_M4_MLA_MAX_SELECTED];
    __shared__ float s_max;
    __shared__ float s_sum;

    // This entry exists for the deferred sequence: it is the format-2 (CSA union) attention only,
    // at an arbitrary position. Any other format is refused the same way the base kernel refuses an
    // unknown one: NaN in out[] and no trace pair. The base entry keeps its 0/1/2 contract; this
    // guard is deliberately narrower so a sequence caller cannot select a window-only format here.
    if (window_kv_format != SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION) {
        for (uint32_t i = lane; i < SPARK_M4_MLA_LATENT; i += SPARK_M4_MLA_THREADS) {
            out[(uint64_t)head * SPARK_M4_MLA_LATENT + i] =
                __float2bfloat16_rn(__uint_as_float(0x7FC00000u));
        }
        return;
    }
    // Union format needs room for the implicit window rows as well as the candidate list. The
    // caller supplies both; this guard is what keeps a direct launch from indexing past s_scores,
    // so the kernel refuses it the same way it refuses an unknown format.
    if (window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION &&
        (window_rows > (uint32_t)SPARK_M4_SWA_WINDOW ||
         selected_count > (uint32_t)SPARK_M4_MLA_MAX_SELECTED ||
         window_rows + selected_count > (uint32_t)SPARK_M4_MLA_MAX_ATTENDED)) {
        for (uint32_t i = lane; i < SPARK_M4_MLA_LATENT; i += SPARK_M4_MLA_THREADS) {
            out[(uint64_t)head * SPARK_M4_MLA_LATENT + i] =
                __float2bfloat16_rn(__uint_as_float(0x7FC00000u));
        }
        return;
    }

    uint64_t t_enter = 0u;
    // The trace is recorded by one block (blockIdx.x == 0, thread 0) and the slice is the extra
    // dimension the indexer added. Guarding the entry here keeps the enter/exit pair balanced
    // with the early return below: only slice 0 enters and only slice 0 exits.
    if (blockIdx.y == 0u) {
        SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_CSA2_ATTN, t_enter);
    }

    // Stage 1: score the HSI blocks for index head 0. Each worker owns disjoint block slots, so
    // there is no cross-block communication and nothing to acquire here.
    if (lane == 0u) {
        s_max = -3.402823466e38f;
        s_sum = 1.0f;
    }
    __syncthreads();
    // The two query heads of one indexer head compute the same 128-wide dots over the same keys:
    // one indexer head serves both, and the index query is addressed by it, not by the query head.
    // block_scores/block_best are one [hsi_blocks] pair in the frozen ABI, so they can publish
    // only ONE index head's scores. Every launch block cooperates on index head 0's query and owns
    // a disjoint subset of the 2,048 blocks (block = worker, stride = worker count), so each block
    // has exactly one writer and all blocks are covered. Publishing all 32 index heads into this
    // single array was the race: whichever head wrote last won, and no head alone covered every
    // block.
    const __nv_bfloat16 *query_row = index_query;
    const uint32_t lane_in_warp = lane % SPARK_M4_MLA_WARP_LANES;
    const uint32_t warp = lane / SPARK_M4_MLA_WARP_LANES;
    static_assert((uint32_t)SPARK_M4_HSI_BLOCKS %
                      ((uint32_t)SPARK_M4_HEADS * SPARK_M4_MLA_WARPS) == 0u,
                  "the fixed grid covers the index pool in whole rounds");
    const uint32_t worker =
        ((blockIdx.y * (uint32_t)SPARK_M4_HEADS + blockIdx.x) * SPARK_M4_MLA_WARPS) + warp;
    const uint32_t block_stride = (uint32_t)SPARK_M4_HEADS * SPARK_M4_MLA_WARPS;
    const float q0 = __bfloat162float(query_row[lane_in_warp]);
    const float q1 = __bfloat162float(query_row[lane_in_warp + SPARK_M4_MLA_WARP_LANES]);
    const float q2 = __bfloat162float(query_row[lane_in_warp + 2u * SPARK_M4_MLA_WARP_LANES]);
    const float q3 = __bfloat162float(query_row[lane_in_warp + 3u * SPARK_M4_MLA_WARP_LANES]);
#pragma unroll
    for (uint32_t round = 0u; round < ((uint32_t)SPARK_M4_HSI_BLOCKS / block_stride);
         ++round) {
        const uint32_t block = worker + round * block_stride;
        if (block >= (uint32_t)SPARK_M4_HSI_BLOCKS) {
            continue;
        }
        // Position-causal indexer: only HSI tokens at or behind the query are scored. A block whose
        // first token is past the query publishes the empty result (-FLT_MAX, token 0), which is the
        // same "no attendable token" value a fully masked block would have. Without this mask the
        // indexer scores future tokens and a per-position candidate selection built from block_best
        // would leak them into the attention; the base entry deliberately keeps its unmasked
        // position-0 behaviour, so this mask lives only in the sequence entry.
        const uint32_t first_token = block * SPARK_M4_HSI_BLOCK_TOKENS;
        float best = -3.402823466e38f;
        uint32_t best_token = 0u;
        if (first_token <= position) {
            for (uint32_t slot = 0; slot < (uint32_t)SPARK_M4_HSI_BLOCK_TOKENS; ++slot) {
                const uint32_t token = first_token + slot;
                // Tokens inside a block are ascending, so the first one past the query ends the
                // scan. The test depends only on block/slot, so every lane takes the same branch
                // and the __shfl_down_sync below keeps all 32 lanes active.
                if (token > position) {
                    break;
                }
                const __nv_bfloat16 *key_row =
                    index_key + ((uint64_t)block * SPARK_M4_HSI_BLOCK_TOKENS + slot) *
                                    SPARK_M4_INDEX_HEAD_DIM;
                float score = q0 * __bfloat162float(key_row[lane_in_warp]) +
                              q1 * __bfloat162float(key_row[lane_in_warp + SPARK_M4_MLA_WARP_LANES]) +
                              q2 *
                                  __bfloat162float(key_row[lane_in_warp + 2u * SPARK_M4_MLA_WARP_LANES]) +
                              q3 *
                                  __bfloat162float(key_row[lane_in_warp + 3u * SPARK_M4_MLA_WARP_LANES]);
                // Reduce the four partial products of each lane inside the warp.
                score = spark_mla_seq_warp_sum(score);
                if (lane_in_warp == 0u && score > best) {
                    best = score;
                    best_token = token;
                }
            }
        }
        // Each worker owns disjoint blocks, so every lane-0 write is single-writer and the
        // published array is index head 0's scores for all 2,048 blocks.
        if (lane_in_warp == 0u) {
            block_scores[block] = best;
            block_best[block] = best_token;
        }
    }

    // blockIdx.y exists only to slice the indexer's HSI pool, but stage 2 used to run unguarded
    // in every slice: each blockIdx.y re-ran the whole attention for the same head and wrote the
    // same out[] values. At a grid of 64 that performed 64 times the attention work for one
    // result, and the launch paid about 1.47 ms for it at 128 candidates (measured on spark-0
    // as the --tokens 130 minus --tokens 1 difference at grid.y 64, before this guard). One
    // execution of that attention is 0.068 ms, so the redundant copies overlapped rather than
    // serialising; the work was still real. Only slice 0 runs the attention now.
    if (blockIdx.y != 0u) {
        return;
    }

    // Format 2: the CSA union. The score array holds the supplied window rows first (all attended,
    // oldest first, engine format) and then the selected compressed candidates (absolute token
    // positions, attended when candidate <= position, FP4 latent). One f32 three-pass softmax runs
    // over that union: maximum, normalising sum plus the sink after the maximum, weighted values.
    // This is the released attention's concatenation (window_indices then compressed_ids) and the
    // reason a selected id older than the window must contribute. Formats 0 and 1 do not enter this
    // branch and their arithmetic below is unchanged.
    if (window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION) {
        const uint32_t union_total = window_rows + selected_count;
        const __nv_bfloat16 *q_head = q + (uint64_t)head * SPARK_M4_MLA_LATENT;
        s_q[lane] = __bfloat162float(q_head[lane]);
        s_q[lane + SPARK_M4_MLA_THREADS] = __bfloat162float(q_head[lane + SPARK_M4_MLA_THREADS]);
        for (uint32_t idx = lane; idx < selected_count; idx += SPARK_M4_MLA_THREADS) {
            s_candidates[idx] = selected[idx];
        }
        __syncthreads();
        // Pass one: every union slot's score, accumulated in ascending channel order with scale
        // decodes hoisted per scale group and the running maximum kept in a register.
        float local_max = -3.402823466e38f;
        for (uint32_t slot = lane; slot < union_total; slot += SPARK_M4_MLA_THREADS) {
            float score = 0.0f;
            if (slot < window_rows) {
                const uint8_t *row_codes = window_codes + (uint64_t)slot * SPARK_M4_MLA_LATENT;
                const uint8_t *row_scales = window_scales + (uint64_t)slot * SPARK_M4_MLA_WINDOW_SCALES;
                for (uint32_t blk = 0u; blk < SPARK_M4_MLA_WINDOW_SCALES; ++blk) {
                    const float scale = spark_ue8m0_decode(row_scales[blk]);
                    const uint32_t base = blk * SPARK_M4_MLA_WINDOW_QUANT_BLOCK;
#pragma unroll 8
                    for (uint32_t ch = 0u; ch < SPARK_M4_MLA_WINDOW_QUANT_BLOCK; ++ch) {
                        const uint32_t i = base + ch;
                        const float key = __bfloat162float(
                            __float2bfloat16_rn(spark_fp8_e4m3_decode(row_codes[i]) * scale));
                        score += s_q[i] * key;
                    }
                }
                score *= SPARK_M4_MLA_SCORE_SCALE;
            } else {
                const uint32_t candidate = s_candidates[slot - window_rows];
                if (spark_mla_seq_union_attends(position, candidate) != 0u) {
                    const uint8_t *row_fp4 =
                        kv_fp4 + (uint64_t)candidate * (SPARK_M4_MLA_LATENT / SPARK_M4_MLA_FP4_PER_BYTE);
                    const uint8_t *row_scales =
                        kv_scales + (uint64_t)candidate *
                                        (SPARK_M4_MLA_LATENT / SPARK_M4_KV_SCALE_CHANNELS);
                    for (uint32_t grp = 0u; grp < (SPARK_M4_MLA_LATENT / SPARK_M4_KV_SCALE_CHANNELS); ++grp) {
                        const float scale = spark_fp8_e4m3_decode(row_scales[grp]);
                        const uint32_t byte_base = grp * (SPARK_M4_KV_SCALE_CHANNELS / SPARK_M4_MLA_FP4_PER_BYTE);
                        const uint32_t ch_base = grp * SPARK_M4_KV_SCALE_CHANNELS;
#pragma unroll
                        for (uint32_t b = 0u; b < (SPARK_M4_KV_SCALE_CHANNELS / SPARK_M4_MLA_FP4_PER_BYTE); ++b) {
                            const uint8_t packed = row_fp4[byte_base + b];
                            const uint32_t i0 = ch_base + 2u * b;
                            const float k0 = spark_fp4_e2m1_decode((uint8_t)(packed & 0x0Fu)) * scale;
                            score += s_q[i0] * k0;
                            const float k1 = spark_fp4_e2m1_decode((uint8_t)(packed >> 4)) * scale;
                            score += s_q[i0 + 1u] * k1;
                        }
                    }
                    score *= SPARK_M4_MLA_SCORE_SCALE;
                } else {
                    score = -3.402823466e38f;
                }
            }
            s_scores[slot] = score;
            local_max = score > local_max ? score : local_max;
        }
        local_max = spark_block_max(local_max, lane, SPARK_M4_MLA_THREADS);
        if (lane == 0u) {
            s_max = local_max;
        }
        __syncthreads();
        // The sink is a denominator term only, added after the maximum as in the released order.
        float local_sum = lane == 0u ? spark_exp(sink[head] - s_max) : 0.0f;
        for (uint32_t slot = lane; slot < union_total; slot += SPARK_M4_MLA_THREADS) {
            const float probability = spark_exp(s_scores[slot] - s_max);
            // kernel.py:339-341: the released body copies the f32 probabilities into a bf16 tile
            // before the P*V GEMM, so the value sum consumes the rounded probability. The
            // normalising sum keeps the f32 probability (kernel.py:342-353), so local_sum
            // below adds the unrounded value on purpose.
            s_scores[slot] = __bfloat162float(__float2bfloat16_rn(probability));
            local_sum += probability;
        }
        local_sum = spark_block_sum(local_sum, lane, SPARK_M4_MLA_THREADS);
        if (lane == 0u) {
            s_sum = local_sum > 0.0f ? local_sum : 1.0f;
        }
        __syncthreads();
        // Each thread owns two latent channels (lane, lane+256) and walks the union once in
        // ascending slot order: window rows first, then the selected list as supplied.
        const uint32_t i0 = lane;
        const uint32_t i1 = lane + SPARK_M4_MLA_THREADS;
        const uint32_t w_scale_off0 = i0 / SPARK_M4_MLA_WINDOW_QUANT_BLOCK;
        const uint32_t w_scale_off1 = i1 / SPARK_M4_MLA_WINDOW_QUANT_BLOCK;
        const uint32_t fp4_byte_off0 = i0 / SPARK_M4_MLA_FP4_PER_BYTE;
        const uint32_t fp4_byte_off1 = i1 / SPARK_M4_MLA_FP4_PER_BYTE;
        const uint32_t fp4_shift = (i0 % SPARK_M4_MLA_FP4_PER_BYTE) * 4u;
        const uint32_t kv_scale_off0 = i0 / SPARK_M4_KV_SCALE_CHANNELS;
        const uint32_t kv_scale_off1 = i1 / SPARK_M4_KV_SCALE_CHANNELS;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t slot = 0u; slot < window_rows; ++slot) {
            const float prob = s_scores[slot];
            const uint8_t *row_codes = window_codes + (uint64_t)slot * SPARK_M4_MLA_LATENT;
            const uint8_t *row_scales = window_scales + (uint64_t)slot * SPARK_M4_MLA_WINDOW_SCALES;
            const float v0 = __bfloat162float(__float2bfloat16_rn(
                spark_fp8_e4m3_decode(row_codes[i0]) * spark_ue8m0_decode(row_scales[w_scale_off0])));
            const float v1 = __bfloat162float(__float2bfloat16_rn(
                spark_fp8_e4m3_decode(row_codes[i1]) * spark_ue8m0_decode(row_scales[w_scale_off1])));
            acc0 += prob * v0;
            acc1 += prob * v1;
        }
        for (uint32_t idx = 0u; idx < selected_count; ++idx) {
            const uint32_t candidate = s_candidates[idx];
            if (spark_mla_seq_union_attends(position, candidate) != 0u) {
                const float prob = s_scores[window_rows + idx];
                const uint8_t *row_fp4 =
                    kv_fp4 + (uint64_t)candidate * (SPARK_M4_MLA_LATENT / SPARK_M4_MLA_FP4_PER_BYTE);
                const uint8_t *row_scales =
                    kv_scales + (uint64_t)candidate * (SPARK_M4_MLA_LATENT / SPARK_M4_KV_SCALE_CHANNELS);
                const uint8_t nib0 = (uint8_t)((row_fp4[fp4_byte_off0] >> fp4_shift) & 0x0Fu);
                const uint8_t nib1 = (uint8_t)((row_fp4[fp4_byte_off1] >> fp4_shift) & 0x0Fu);
                const float v0 = spark_fp4_e2m1_decode(nib0) * spark_fp8_e4m3_decode(row_scales[kv_scale_off0]);
                const float v1 = spark_fp4_e2m1_decode(nib1) * spark_fp8_e4m3_decode(row_scales[kv_scale_off1]);
                acc0 += prob * v0;
                acc1 += prob * v1;
            }
        }
        out[(uint64_t)head * SPARK_M4_MLA_LATENT + i0] = __float2bfloat16_rn(acc0 / s_sum);
        out[(uint64_t)head * SPARK_M4_MLA_LATENT + i1] = __float2bfloat16_rn(acc1 / s_sum);
        if (lane == 0u) {
            const uint32_t meta = (uint32_t)(SPARK_M4_MLA_LATENT * 2u);
            SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_CSA2_ATTN, layer_id, meta, t_enter);
            (void)heads;
            (void)step_seq;
        }
        (void)t_enter;
        return;
    }

    // Stage 2: attention over the selected candidates. Pass one computes every score and the
    // running maximum, pass two the normalising sum, pass three the weighted latent sum. The
    // candidate count is bounded by SPARK_M4_MLA_MAX_SELECTED; the caller guarantees that.
    for (uint32_t slot = lane; slot < selected_count; slot += SPARK_M4_MLA_THREADS) {
        const uint32_t candidate = selected[slot];
        float score = 0.0f;
        if (spark_mla_seq_attends(window_kv_format, position, candidate, window_rows) != 0u) {
            for (uint32_t i = 0; i < SPARK_M4_MLA_LATENT; ++i) {
                const float key = window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4
                                      ? spark_mla_seq_load_latent(
                                            kv_fp4 + (uint64_t)candidate * (SPARK_M4_MLA_LATENT / 2u), i) *
                                            spark_mla_seq_scale(
                                                kv_scales + (uint64_t)candidate *
                                                                (SPARK_M4_MLA_LATENT / SPARK_M4_KV_SCALE_CHANNELS),
                                                i)
                                      : spark_mla_seq_load_window_stored(window_codes, window_scales, candidate, i);
                score += __bfloat162float(q[(uint64_t)head * SPARK_M4_MLA_LATENT + i]) * key;
            }
            // The MLA score scale, 1/sqrt(512) over the latent width. Specification 2.3 fixes the
            // selection rule but not this constant, so it is named once in m4_common_sm121.cuh and
            // src/mla_ref.rs carries the same literal, compared bit for bit by a test there.
            score *= SPARK_M4_MLA_SCORE_SCALE;
        } else {
            // Outside the format's window: excluded by score, not merely down-weighted. In engine
            // format an id at or past window_rows gets this exact value too, so its contribution is
            // exactly zero in pass three; that zeroing is a requirement and is tested by injection.
            score = -3.402823466e38f;
        }
        s_scores[slot] = score;
    }
    __syncthreads();
    // The maximum is taken per thread and then reduced as a tree. It must not be accumulated into
    // the shared s_max by every thread: that read-modify-write is a data race, lost updates leave
    // s_max below the true maximum, and the softmax below then overflows to infinity and divides
    // to NaN. spark_block_max is a rendezvous with a fixed tree order, so every thread reaches it
    // and the result is the same on every run, which is what the CPU reference can pin.
    float local_max = -3.402823466e38f;
    for (uint32_t slot = lane; slot < selected_count; slot += SPARK_M4_MLA_THREADS) {
        local_max = s_scores[slot] > local_max ? s_scores[slot] : local_max;
    }
    local_max = spark_block_max(local_max, lane, SPARK_M4_MLA_THREADS);
    if (lane == 0u) {
        s_max = local_max;
    }
    __syncthreads();
    // The sink contributes probability mass but no value, so it is one term of the denominator
    // and nothing else. Adding it after the maximum is the released order (kernel.py:382-383);
    // thread 0 owns the term so the reduction order stays fixed run to run.
    float local_sum = lane == 0u ? spark_exp(sink[head] - s_max) : 0.0f;
    for (uint32_t slot = lane; slot < selected_count; slot += SPARK_M4_MLA_THREADS) {
        const float probability = spark_exp(s_scores[slot] - s_max);
        // kernel.py:339-341: the released body copies the f32 probabilities into a bf16 tile
        // before the P*V GEMM, so the value sum consumes the rounded probability. The
        // normalising sum keeps the f32 probability (kernel.py:342-353), so local_sum
        // below adds the unrounded value on purpose.
        s_scores[slot] = __bfloat162float(__float2bfloat16_rn(probability));
        local_sum += probability;
    }
    local_sum = spark_block_sum(local_sum, lane, SPARK_M4_MLA_THREADS);
    if (lane == 0u) {
        s_sum = local_sum > 0.0f ? local_sum : 1.0f;
    }
    // The publish of s_sum needs its own rendezvous, for the same reason s_max does: without it a
    // warp that loses the race reads the prologue's 1.0f and divides the latent sum by one instead
    // of by the normaliser. That is not a rounding error - it scales the head's output by s_sum and
    // it is nondeterministic, so the CPU reference cannot agree with it. Stage 2 of the layer-0
    // harness caught exactly that: identical input scored 96.1% exact on one run and 87.5% on the
    // next, and every over-tolerance value was a head that had divided by the wrong denominator.
    __syncthreads();
    // Each thread owns two latent channels and walks the candidate list in ascending slot order.
    // That order is part of the contract, so this deliberately does not use shared-memory atomics:
    // an atomic sum would make the rounding depend on warp scheduling and no CPU reference could
    // pin it. Out-of-window candidates contribute exactly zero, because exp(-FLT_MAX - max) is zero
    // and the window test below also skips them; the reference does the same.
    for (uint32_t i = lane; i < SPARK_M4_MLA_LATENT; i += SPARK_M4_MLA_THREADS) {
        float acc = 0.0f;
        for (uint32_t slot = 0; slot < selected_count; ++slot) {
            const uint32_t candidate = selected[slot];
            if (spark_mla_seq_attends(window_kv_format, position, candidate, window_rows) != 0u) {
                const float value = window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4
                                        ? spark_mla_seq_load_latent(
                                              kv_fp4 + (uint64_t)candidate * (SPARK_M4_MLA_LATENT / 2u), i) *
                                              spark_mla_seq_scale(
                                                  kv_scales + (uint64_t)candidate *
                                                                  (SPARK_M4_MLA_LATENT / SPARK_M4_KV_SCALE_CHANNELS),
                                                  i)
                                        : spark_mla_seq_load_window_stored(window_codes, window_scales, candidate, i);
                acc += s_scores[slot] * value;
            }
        }
        out[(uint64_t)head * SPARK_M4_MLA_LATENT + i] = __float2bfloat16_rn(acc / s_sum);
    }

    if (lane == 0u) {
        const uint32_t meta = (uint32_t)(SPARK_M4_MLA_LATENT * 2u);
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_CSA2_ATTN, layer_id, meta, t_enter);
        (void)heads;
        (void)step_seq;
    }
    (void)t_enter;
}
// ---------------------------------------------------------------------------------------------
// K8 batched sequence entry: ONE block per query head serves SIX consecutive positions. The
// indexer's HSI key pool is read once and each key vector scores all six rows (six 128-wide dots),
// which is the traffic win; stage 2 then runs the existing three-pass CSA-union attention six times
// sequentially, one row at a time, reusing the same shared score array. Per-row arithmetic and
// ordering are the scalar sequence entry's, so out[] and the block arrays are byte-identical per row.
//
// Batch-major layout, fixed six rows: q[6][heads][latent]; window_codes[6][window_rows][latent];
// window_scales[6][window_rows][16]; selected[6][selected_count]; index_query[6][heads][128];
// out[6][heads][latent]; block_scores/block_best[6][hsi_blocks]. The six positions are
// position..position+5. This first cut requires the SAME window_rows and selected_count on every
// row (the decode-verify case at position >= 127 has 128 window rows and one 512-id selected list
// per row). The selected row stride is selected_count: pack the rows contiguously. A caller that
// pads each row to a wider buffer must pass that padded width as selected_count and mask the
// future/pad ids (the kernel already excludes candidate > that row's position), and pass the full
// 128-row window for every row. OPEN ABI ITEM for K7: this is mandatory caller-side pad+mask, not
// optional; a per-row window_rows/selected_count pointer pair is a follow-up symbol, not this v1.
#define SPARK_M4_MLA_BATCH_ROWS 6u
static_assert(SPARK_M4_MLA_BATCH_ROWS * SPARK_M4_MLA_MAX_ATTENDED <= 3840u,
              "one row's score array is reused sequentially, so shared stays at one row");

extern "C" __global__ void flash_mla_csa2_seq_batched_sm121(
    const __nv_bfloat16 *__restrict__ q,             // [6][heads][latent]
    const uint8_t *__restrict__ kv_fp4,              // [positions][latent/2]
    const uint8_t *__restrict__ kv_scales,           // [positions][latent/16]
    const uint8_t *__restrict__ window_codes,        // [6][window_rows][latent]
    const uint8_t *__restrict__ window_scales,       // [6][window_rows][16]
    const uint32_t window_kv_format,                 // must be 2 (CSA union)
    const uint32_t window_rows,
    const uint32_t *__restrict__ selected,           // [6][selected_count]
    const __nv_bfloat16 *__restrict__ index_key,     // [hsi_blocks][8][index_head_dim]
    const __nv_bfloat16 *__restrict__ index_query,   // [6][index_heads][index_head_dim]
    const float *__restrict__ sink,                  // [heads]
    float *__restrict__ block_scores,                // [6][hsi_blocks]
    uint32_t *__restrict__ block_best,               // [6][hsi_blocks]
    __nv_bfloat16 *__restrict__ out,                 // [6][heads][latent]
    const uint32_t position,                         // base position; rows are position..position+5
    const uint32_t selected_count,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t head = blockIdx.x;
    const uint32_t lane = threadIdx.x;

    __shared__ float s_scores[SPARK_M4_MLA_MAX_ATTENDED];
    __shared__ float s_q[SPARK_M4_MLA_LATENT];
    __shared__ uint32_t s_candidates[SPARK_M4_MLA_MAX_SELECTED];
    __shared__ float s_max;
    __shared__ float s_sum;

    if (window_kv_format != SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION ||
        window_rows > (uint32_t)SPARK_M4_SWA_WINDOW ||
        selected_count > (uint32_t)SPARK_M4_MLA_MAX_SELECTED ||
        window_rows + selected_count > (uint32_t)SPARK_M4_MLA_MAX_ATTENDED) {
        for (uint32_t r = 0u; r < SPARK_M4_MLA_BATCH_ROWS; ++r) {
            for (uint32_t i = lane; i < SPARK_M4_MLA_LATENT; i += SPARK_M4_MLA_THREADS) {
                out[(uint64_t)r * SPARK_M4_HEADS * SPARK_M4_MLA_LATENT +
                    (uint64_t)head * SPARK_M4_MLA_LATENT + i] =
                    __float2bfloat16_rn(__uint_as_float(0x7FC00000u));
            }
        }
        return;
    }

    uint64_t t_enter = 0u;
    if (head == 0u) {
        SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_CSA2_ATTN, t_enter);
    }

    const uint32_t lane_in_warp = lane % SPARK_M4_MLA_WARP_LANES;
    const uint32_t warp = lane / SPARK_M4_MLA_WARP_LANES;
    static_assert((uint32_t)SPARK_M4_HSI_BLOCKS %
                      ((uint32_t)SPARK_M4_HEADS * SPARK_M4_MLA_WARPS) == 0u,
                  "the fixed grid covers the index pool in whole rounds");
    const uint32_t worker =
        ((blockIdx.y * (uint32_t)SPARK_M4_HEADS + blockIdx.x) * SPARK_M4_MLA_WARPS) + warp;
    const uint32_t block_stride = (uint32_t)SPARK_M4_HEADS * SPARK_M4_MLA_WARPS;

    // Six rows' four query fragments per lane, in registers; the key vector is loaded once below.
    // The published block arrays are index head 0's, so every launch block loads index head 0's
    // query row for the indexer; the attention below still uses each query head's own q row.
    float qi[SPARK_M4_MLA_BATCH_ROWS][4];
#pragma unroll
    for (uint32_t r = 0u; r < SPARK_M4_MLA_BATCH_ROWS; ++r) {
        const uint32_t base = lane_in_warp;
        qi[r][0] = __bfloat162float(
            index_query[(uint64_t)r * SPARK_M4_INDEX_HEADS * SPARK_M4_INDEX_HEAD_DIM + base]);
        qi[r][1] = __bfloat162float(
            index_query[(uint64_t)r * SPARK_M4_INDEX_HEADS * SPARK_M4_INDEX_HEAD_DIM +
                        base + SPARK_M4_MLA_WARP_LANES]);
        qi[r][2] = __bfloat162float(
            index_query[(uint64_t)r * SPARK_M4_INDEX_HEADS * SPARK_M4_INDEX_HEAD_DIM +
                        base + 2u * SPARK_M4_MLA_WARP_LANES]);
        qi[r][3] = __bfloat162float(
            index_query[(uint64_t)r * SPARK_M4_INDEX_HEADS * SPARK_M4_INDEX_HEAD_DIM +
                        base + 3u * SPARK_M4_MLA_WARP_LANES]);
    }

    float best[SPARK_M4_MLA_BATCH_ROWS];
    uint32_t best_token[SPARK_M4_MLA_BATCH_ROWS];

#pragma unroll
    for (uint32_t round = 0u; round < ((uint32_t)SPARK_M4_HSI_BLOCKS / block_stride);
         ++round) {
        const uint32_t block = worker + round * block_stride;
        if (block >= (uint32_t)SPARK_M4_HSI_BLOCKS) {
            continue;
        }
        // Each round's block is a fresh block; carrying the previous round's best into a future
        // block was the batch-vs-scalar divergence (blocks 512+ re-published block 0..511).
#pragma unroll
        for (uint32_t r = 0u; r < SPARK_M4_MLA_BATCH_ROWS; ++r) {
            best[r] = -3.402823466e38f;
            best_token[r] = 0u;
        }
        const uint32_t first_token = block * SPARK_M4_HSI_BLOCK_TOKENS;
        for (uint32_t slot = 0u; slot < (uint32_t)SPARK_M4_HSI_BLOCK_TOKENS; ++slot) {
            const uint32_t token = first_token + slot;
            if (token > position + (SPARK_M4_MLA_BATCH_ROWS - 1u)) {
                break;
            }
            const __nv_bfloat16 *key_row =
                index_key + ((uint64_t)block * SPARK_M4_HSI_BLOCK_TOKENS + slot) *
                                SPARK_M4_INDEX_HEAD_DIM;
            const float k0 = __bfloat162float(key_row[lane_in_warp]);
            const float k1 = __bfloat162float(key_row[lane_in_warp + SPARK_M4_MLA_WARP_LANES]);
            const float k2 = __bfloat162float(key_row[lane_in_warp + 2u * SPARK_M4_MLA_WARP_LANES]);
            const float k3 = __bfloat162float(key_row[lane_in_warp + 3u * SPARK_M4_MLA_WARP_LANES]);
#pragma unroll
            for (uint32_t r = 0u; r < SPARK_M4_MLA_BATCH_ROWS; ++r) {
                if (token <= position + r) {
                    float score = qi[r][0] * k0 + qi[r][1] * k1 + qi[r][2] * k2 + qi[r][3] * k3;
                    score = spark_mla_seq_warp_sum(score);
                    if (lane_in_warp == 0u && score > best[r]) {
                        best[r] = score;
                        best_token[r] = token;
                    }
                }
            }
        }
        // Disjoint worker blocks mean one writer per block, so the row arrays are deterministic.
        if (lane_in_warp == 0u) {
#pragma unroll
            for (uint32_t r = 0u; r < SPARK_M4_MLA_BATCH_ROWS; ++r) {
                block_scores[(uint64_t)r * SPARK_M4_HSI_BLOCKS + block] = best[r];
                block_best[(uint64_t)r * SPARK_M4_HSI_BLOCKS + block] = best_token[r];
            }
        }
    }
    __syncthreads();

    const uint32_t union_total = window_rows + selected_count;
    for (uint32_t r = 0u; r < SPARK_M4_MLA_BATCH_ROWS; ++r) {
        const uint32_t row_position = position + r;
        const __nv_bfloat16 *q_row =
            q + ((uint64_t)r * SPARK_M4_HEADS + head) * SPARK_M4_MLA_LATENT;
        const uint8_t *row_window_codes =
            window_codes + (uint64_t)r * window_rows * SPARK_M4_MLA_LATENT;
        const uint8_t *row_window_scales = window_scales +
            (uint64_t)r * window_rows * SPARK_M4_MLA_WINDOW_SCALES;
        const uint32_t *row_selected = selected + (uint64_t)r * selected_count;
        __nv_bfloat16 *row_out =
            out + ((uint64_t)r * SPARK_M4_HEADS + head) * SPARK_M4_MLA_LATENT;

        s_q[lane] = __bfloat162float(q_row[lane]);
        s_q[lane + SPARK_M4_MLA_THREADS] = __bfloat162float(q_row[lane + SPARK_M4_MLA_THREADS]);
        for (uint32_t idx = lane; idx < selected_count; idx += SPARK_M4_MLA_THREADS) {
            s_candidates[idx] = row_selected[idx];
        }
        __syncthreads();
        float local_max = -3.402823466e38f;
        for (uint32_t slot = lane; slot < union_total; slot += SPARK_M4_MLA_THREADS) {
            float score = 0.0f;
            if (slot < window_rows) {
                const uint8_t *row_codes = row_window_codes + (uint64_t)slot * SPARK_M4_MLA_LATENT;
                const uint8_t *row_scales = row_window_scales + (uint64_t)slot * SPARK_M4_MLA_WINDOW_SCALES;
                for (uint32_t blk = 0u; blk < SPARK_M4_MLA_WINDOW_SCALES; ++blk) {
                    const float scale = spark_ue8m0_decode(row_scales[blk]);
                    const uint32_t base = blk * SPARK_M4_MLA_WINDOW_QUANT_BLOCK;
#pragma unroll 8
                    for (uint32_t ch = 0u; ch < SPARK_M4_MLA_WINDOW_QUANT_BLOCK; ++ch) {
                        const uint32_t i = base + ch;
                        const float key = __bfloat162float(
                            __float2bfloat16_rn(spark_fp8_e4m3_decode(row_codes[i]) * scale));
                        score += s_q[i] * key;
                    }
                }
                score *= SPARK_M4_MLA_SCORE_SCALE;
            } else {
                const uint32_t candidate = s_candidates[slot - window_rows];
                if (spark_mla_seq_union_attends(row_position, candidate) != 0u) {
                    const uint8_t *row_fp4 =
                        kv_fp4 + (uint64_t)candidate * (SPARK_M4_MLA_LATENT / SPARK_M4_MLA_FP4_PER_BYTE);
                    const uint8_t *row_scales =
                        kv_scales + (uint64_t)candidate *
                                        (SPARK_M4_MLA_LATENT / SPARK_M4_KV_SCALE_CHANNELS);
                    for (uint32_t grp = 0u; grp < (SPARK_M4_MLA_LATENT / SPARK_M4_KV_SCALE_CHANNELS); ++grp) {
                        const float scale = spark_fp8_e4m3_decode(row_scales[grp]);
                        const uint32_t byte_base = grp * (SPARK_M4_KV_SCALE_CHANNELS / SPARK_M4_MLA_FP4_PER_BYTE);
                        const uint32_t ch_base = grp * SPARK_M4_KV_SCALE_CHANNELS;
#pragma unroll
                        for (uint32_t b = 0u; b < (SPARK_M4_KV_SCALE_CHANNELS / SPARK_M4_MLA_FP4_PER_BYTE); ++b) {
                            const uint8_t packed = row_fp4[byte_base + b];
                            const uint32_t i0 = ch_base + 2u * b;
                            const float k0 = spark_fp4_e2m1_decode((uint8_t)(packed & 0x0Fu)) * scale;
                            score += s_q[i0] * k0;
                            const float k1 = spark_fp4_e2m1_decode((uint8_t)(packed >> 4)) * scale;
                            score += s_q[i0 + 1u] * k1;
                        }
                    }
                    score *= SPARK_M4_MLA_SCORE_SCALE;
                } else {
                    score = -3.402823466e38f;
                }
            }
            s_scores[slot] = score;
            local_max = score > local_max ? score : local_max;
        }
        local_max = spark_block_max(local_max, lane, SPARK_M4_MLA_THREADS);
        if (lane == 0u) {
            s_max = local_max;
        }
        __syncthreads();
        float local_sum = lane == 0u ? spark_exp(sink[head] - s_max) : 0.0f;
        for (uint32_t slot = lane; slot < union_total; slot += SPARK_M4_MLA_THREADS) {
            const float probability = spark_exp(s_scores[slot] - s_max);
            s_scores[slot] = __bfloat162float(__float2bfloat16_rn(probability));
            local_sum += probability;
        }
        local_sum = spark_block_sum(local_sum, lane, SPARK_M4_MLA_THREADS);
        if (lane == 0u) {
            s_sum = local_sum > 0.0f ? local_sum : 1.0f;
        }
        __syncthreads();
        const uint32_t i0 = lane;
        const uint32_t i1 = lane + SPARK_M4_MLA_THREADS;
        const uint32_t w_scale_off0 = i0 / SPARK_M4_MLA_WINDOW_QUANT_BLOCK;
        const uint32_t w_scale_off1 = i1 / SPARK_M4_MLA_WINDOW_QUANT_BLOCK;
        const uint32_t fp4_byte_off0 = i0 / SPARK_M4_MLA_FP4_PER_BYTE;
        const uint32_t fp4_byte_off1 = i1 / SPARK_M4_MLA_FP4_PER_BYTE;
        const uint32_t fp4_shift = (i0 % SPARK_M4_MLA_FP4_PER_BYTE) * 4u;
        const uint32_t kv_scale_off0 = i0 / SPARK_M4_KV_SCALE_CHANNELS;
        const uint32_t kv_scale_off1 = i1 / SPARK_M4_KV_SCALE_CHANNELS;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t slot = 0u; slot < window_rows; ++slot) {
            const float prob = s_scores[slot];
            const uint8_t *row_codes = row_window_codes + (uint64_t)slot * SPARK_M4_MLA_LATENT;
            const uint8_t *row_scales = row_window_scales + (uint64_t)slot * SPARK_M4_MLA_WINDOW_SCALES;
            const float v0 = __bfloat162float(__float2bfloat16_rn(
                spark_fp8_e4m3_decode(row_codes[i0]) * spark_ue8m0_decode(row_scales[w_scale_off0])));
            const float v1 = __bfloat162float(__float2bfloat16_rn(
                spark_fp8_e4m3_decode(row_codes[i1]) * spark_ue8m0_decode(row_scales[w_scale_off1])));
            acc0 += prob * v0;
            acc1 += prob * v1;
        }
        for (uint32_t idx = 0u; idx < selected_count; ++idx) {
            const uint32_t candidate = s_candidates[idx];
            if (spark_mla_seq_union_attends(row_position, candidate) != 0u) {
                const float prob = s_scores[window_rows + idx];
                const uint8_t *row_fp4 =
                    kv_fp4 + (uint64_t)candidate * (SPARK_M4_MLA_LATENT / SPARK_M4_MLA_FP4_PER_BYTE);
                const uint8_t *row_scales =
                    kv_scales + (uint64_t)candidate * (SPARK_M4_MLA_LATENT / SPARK_M4_KV_SCALE_CHANNELS);
                const uint8_t nib0 = (uint8_t)((row_fp4[fp4_byte_off0] >> fp4_shift) & 0x0Fu);
                const uint8_t nib1 = (uint8_t)((row_fp4[fp4_byte_off1] >> fp4_shift) & 0x0Fu);
                const float v0 = spark_fp4_e2m1_decode(nib0) * spark_fp8_e4m3_decode(row_scales[kv_scale_off0]);
                const float v1 = spark_fp4_e2m1_decode(nib1) * spark_fp8_e4m3_decode(row_scales[kv_scale_off1]);
                acc0 += prob * v0;
                acc1 += prob * v1;
            }
        }
        row_out[i0] = __float2bfloat16_rn(acc0 / s_sum);
        row_out[i1] = __float2bfloat16_rn(acc1 / s_sum);
        __syncthreads();
    }

    if (lane == 0u) {
        const uint32_t meta = (uint32_t)(SPARK_M4_MLA_LATENT * 2u);
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_CSA2_ATTN, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}

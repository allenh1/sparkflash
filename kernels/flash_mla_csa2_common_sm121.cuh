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

// kernels/flash_mla_csa2_common_sm121.cuh
//
// The seven device predicates and codecs the three CSA2 attention entries (token, sequence and
// ring) had byte-identical copies of, under three prefixes. The entries differ in which positions
// a query attends over; every helper below reads the same layout and returns the same value.
//
// The caller defines SPARK_M4_MLA_LATENT, the window formats and SPARK_M4_MLA_WINDOW_SCALES
// before including this header, as each entry already did.

#pragma once

#include <cstdint>

// The three entries define these differently (the base file has no ring at all), so the shared
// body supplies the ones a given entry does not name.
#ifndef SPARK_M4_MLA_MAX_ATTENDED
#define SPARK_M4_MLA_MAX_ATTENDED (SPARK_M4_MLA_MAX_SELECTED + SPARK_M4_SWA_WINDOW)
#endif
#ifndef SPARK_M4_MLA_RING_CAPACITY
#define SPARK_M4_MLA_RING_CAPACITY 134u
#endif
#ifndef SPARK_M4_MLA_RING_MAX_ATTENDED
#define SPARK_M4_MLA_RING_MAX_ATTENDED \
    (SPARK_M4_MLA_MAX_SELECTED + SPARK_M4_MLA_RING_CAPACITY)
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// A warp's descending tree sum.
__device__ __forceinline__ float spark_mla_warp_sum(float value) {
    value += __shfl_down_sync(0xffffffffu, value, 16u);
    value += __shfl_down_sync(0xffffffffu, value, 8u);
    value += __shfl_down_sync(0xffffffffu, value, 4u);
    value += __shfl_down_sync(0xffffffffu, value, 2u);
    value += __shfl_down_sync(0xffffffffu, value, 1u);
    return value;
}

__device__ __forceinline__ uint32_t spark_mla_in_window(uint32_t query_position,
                                                         uint32_t candidate) {
    return query_position >= candidate &&
           (query_position - candidate) < (uint32_t)SPARK_M4_SWA_WINDOW;
}

__device__ __forceinline__ float spark_mla_load_latent(const uint8_t *row, uint32_t index) {
    const uint8_t packed = row[index / SPARK_M4_MLA_FP4_PER_BYTE];
    const uint8_t nibble = (index % SPARK_M4_MLA_FP4_PER_BYTE) == 0u ? (uint8_t)(packed & 0x0Fu)
                                                                     : (uint8_t)(packed >> 4);
    return spark_fp4_e2m1_decode(nibble);
}

__device__ __forceinline__ float spark_mla_scale(const uint8_t *scales, uint32_t index) {
    return spark_fp8_e4m3_decode(scales[index / SPARK_M4_KV_SCALE_CHANNELS]);
}

__device__ __forceinline__ float spark_mla_load_window_stored(const uint8_t *window_codes,
                                                               const uint8_t *window_scales,
                                                               uint32_t row, uint32_t index) {
    const float code =
        spark_fp8_e4m3_decode(window_codes[(uint64_t)row * SPARK_M4_MLA_LATENT + index]);
    const float scale = spark_ue8m0_decode(
        window_scales[(uint64_t)row * SPARK_M4_MLA_WINDOW_SCALES +
                      index / SPARK_M4_MLA_WINDOW_QUANT_BLOCK]);
    return __bfloat162float(__float2bfloat16_rn(code * scale));
}

__device__ __forceinline__ uint32_t spark_mla_attends(uint32_t window_kv_format,
                                                       uint32_t position, uint32_t candidate,
                                                       uint32_t window_rows) {
    return window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4
               ? spark_mla_in_window(position, candidate)
               : (candidate < window_rows ? 1u : 0u);
}

__device__ __forceinline__ uint32_t spark_mla_union_attends(uint32_t position,
                                                             uint32_t candidate) {
    return candidate <= position ? 1u : 0u;
}

// The three CSA2 entries' one body. MODE picks the only things they differ in:
//
//   * SparkMlaBase - accepts formats 0/1/2, scores the indexer unmasked (the measured position-0
//     behaviour), and keeps the f32 probability in the value sum;
//   * SparkMlaSeq  - format 2 only, index head 0 scores the pool from a worker-stride walk, the
//     indexer is position-causal, and the value sum consumes the bf16-rounded probability;
//   * SparkMlaRing - as Seq, but the 512+134 score bound and an explicit ring/compression guard,
//     and the pool splits by the query head's parity within its index pair.
//
// The attention arithmetic after stage 1 is one statement: window rows oldest first, then the
// selected compressed ids masked by candidate <= position, one f32 softmax with the sink in the
// denominator after the maximum, and two latent channels a thread in ascending slot order.
enum SparkMlaMode { SparkMlaBase, SparkMlaSeq, SparkMlaRing };

template <SparkMlaMode MODE>
__device__ __forceinline__ void spark_mla_attention_body(
    const __nv_bfloat16 *__restrict__ q, const uint8_t *__restrict__ kv_fp4,
    const uint8_t *__restrict__ kv_scales, const uint8_t *__restrict__ window_codes,
    const uint8_t *__restrict__ window_scales, const uint32_t window_kv_format,
    const uint32_t window_rows, const uint32_t ring_capacity, const uint32_t compression_ratio,
    const uint32_t *__restrict__ selected, const __nv_bfloat16 *__restrict__ index_key,
    const __nv_bfloat16 *__restrict__ index_query, const float *__restrict__ sink,
    float *__restrict__ block_scores, uint32_t *__restrict__ block_best,
    __nv_bfloat16 *__restrict__ out, const uint32_t position, const uint32_t selected_count,
    GpuTraceRing *__restrict__ trace_ring, const uint32_t layer_id, const uint64_t step_seq) {
    constexpr bool BASE = MODE == SparkMlaBase;
    constexpr bool RING = MODE == SparkMlaRing;
    constexpr bool PAIR = !(MODE == SparkMlaSeq);
    constexpr bool CAUSAL = !BASE;
    constexpr bool ROUND_PROB = !BASE;
    constexpr uint32_t kMaxAttended =
        RING ? SPARK_M4_MLA_RING_MAX_ATTENDED : SPARK_M4_MLA_MAX_ATTENDED;

    const uint32_t head = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t heads = (uint32_t)SPARK_M4_HEADS;
    const uint32_t index_head = head / (uint32_t)SPARK_M4_MLA_HEADS_PER_INDEX_HEAD;

    __shared__ float s_scores[kMaxAttended];
    __shared__ float s_q[SPARK_M4_MLA_LATENT];
    __shared__ uint32_t s_candidates[SPARK_M4_MLA_MAX_SELECTED];
    __shared__ float s_max;
    __shared__ float s_sum;

    // Entry guards. A refused launch poisons out[] with NaN and enters no trace pair.
    bool refuse = false;
    if constexpr (BASE) {
        refuse = window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION;
        if (!refuse && window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION) {
            refuse = window_rows > (uint32_t)SPARK_M4_SWA_WINDOW ||
                     selected_count > (uint32_t)SPARK_M4_MLA_MAX_SELECTED ||
                     window_rows + selected_count > (uint32_t)SPARK_M4_MLA_MAX_ATTENDED;
        }
    } else if constexpr (RING) {
        refuse = window_kv_format != SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION ||
                 ring_capacity == 0u ||
                 ring_capacity > (uint32_t)SPARK_M4_MLA_RING_CAPACITY ||
                 compression_ratio == 0u || window_rows > ring_capacity ||
                 selected_count > (uint32_t)SPARK_M4_MLA_MAX_SELECTED ||
                 window_rows + selected_count > kMaxAttended;
    } else {
        refuse = window_kv_format != SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION ||
                 window_rows > (uint32_t)SPARK_M4_SWA_WINDOW ||
                 selected_count > (uint32_t)SPARK_M4_MLA_MAX_SELECTED ||
                 window_rows + selected_count > kMaxAttended;
    }
    if (refuse) {
        for (uint32_t i = lane; i < SPARK_M4_MLA_LATENT; i += SPARK_M4_MLA_THREADS) {
            out[(uint64_t)head * SPARK_M4_MLA_LATENT + i] =
                __float2bfloat16_rn(__uint_as_float(0x7FC00000u));
        }
        return;
    }

    uint64_t t_enter = 0u;
    if (blockIdx.y == 0u) {
        SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_CSA2_ATTN, t_enter);
    }

    // Stage 1: score the HSI blocks. Each worker owns disjoint block slots, so there is no
    // cross-block communication and nothing to acquire here.
    if (lane == 0u) {
        s_max = -3.402823466e38f;
        s_sum = 1.0f;
    }
    __syncthreads();
    const uint32_t lane_in_warp = lane % SPARK_M4_MLA_WARP_LANES;
    const uint32_t warp = lane / SPARK_M4_MLA_WARP_LANES;
    const __nv_bfloat16 *query_row;
    uint32_t first_block;
    uint32_t block_step;
    uint32_t rounds = 0u;
    if constexpr (PAIR) {
        // The two query heads of one indexer head compute the same 128-wide dots over the same
        // keys. The pair splits the pool instead of duplicating it: the even head takes the even
        // HSI blocks and the odd head the odd ones, so every block is written by exactly one head
        // of the pair and a full-head launch covers all 2,048.
        query_row = index_query + (uint64_t)index_head * SPARK_M4_INDEX_HEAD_DIM;
        const uint32_t pair = head % (uint32_t)SPARK_M4_MLA_HEADS_PER_INDEX_HEAD;
        const uint32_t pair_stride = (uint32_t)SPARK_M4_MLA_HEADS_PER_INDEX_HEAD;
        const uint32_t blocks_per_slice = SPARK_M4_MLA_WARPS * pair_stride;
        first_block = blockIdx.y * blocks_per_slice + warp * pair_stride + pair;
        block_step = gridDim.y * blocks_per_slice;
    } else {
        // The frozen sequence ABI carries one [hsi_blocks] pair, so only index head 0 writes it;
        // every launch block cooperates on that query and owns a disjoint subset of the pool.
        query_row = index_query;
        const uint32_t worker =
            ((blockIdx.y * (uint32_t)SPARK_M4_HEADS + blockIdx.x) * SPARK_M4_MLA_WARPS) + warp;
        block_step = (uint32_t)SPARK_M4_HEADS * SPARK_M4_MLA_WARPS;
        rounds = (uint32_t)SPARK_M4_HSI_BLOCKS / block_step;
        first_block = worker;
    }
    const float q0 = __bfloat162float(query_row[lane_in_warp]);
    const float q1 = __bfloat162float(query_row[lane_in_warp + SPARK_M4_MLA_WARP_LANES]);
    const float q2 = __bfloat162float(query_row[lane_in_warp + 2u * SPARK_M4_MLA_WARP_LANES]);
    const float q3 = __bfloat162float(query_row[lane_in_warp + 3u * SPARK_M4_MLA_WARP_LANES]);
    const uint32_t visits = PAIR ? (uint32_t)SPARK_M4_HSI_BLOCKS : rounds;
    for (uint32_t visit = 0u;; ++visit) {
        const uint32_t block = PAIR ? first_block + visit * block_step : first_block + visit * block_step;
        if constexpr (PAIR) {
            if (block >= (uint32_t)SPARK_M4_HSI_BLOCKS) break;
        } else {
            if (visit >= visits) break;
            if (block >= (uint32_t)SPARK_M4_HSI_BLOCKS) continue;
        }
        float best = -3.402823466e38f;
        uint32_t best_token = 0u;
        const uint32_t first_token = block * SPARK_M4_HSI_BLOCK_TOKENS;
        if (CAUSAL && first_token > position) {
            // A block entirely past the query publishes the empty result, so a per-position
            // candidate selection built from block_best cannot leak a future token.
        } else {
            for (uint32_t slot = 0; slot < (uint32_t)SPARK_M4_HSI_BLOCK_TOKENS; ++slot) {
                const uint32_t token = first_token + slot;
                if (CAUSAL && token > position) {
                    break;
                }
                const __nv_bfloat16 *key_row =
                    index_key + ((uint64_t)block * SPARK_M4_HSI_BLOCK_TOKENS + slot) *
                                    SPARK_M4_INDEX_HEAD_DIM;
                float score =
                    q0 * __bfloat162float(key_row[lane_in_warp]) +
                    q1 * __bfloat162float(key_row[lane_in_warp + SPARK_M4_MLA_WARP_LANES]) +
                    q2 * __bfloat162float(key_row[lane_in_warp + 2u * SPARK_M4_MLA_WARP_LANES]) +
                    q3 * __bfloat162float(key_row[lane_in_warp + 3u * SPARK_M4_MLA_WARP_LANES]);
                score = spark_mla_warp_sum(score);
                if (lane_in_warp == 0u && score > best) {
                    best = score;
                    best_token = token;
                }
            }
        }
        if (lane_in_warp == 0u) {
            block_scores[block] = best;
            block_best[block] = best_token;
        }
    }

    // blockIdx.y exists only to slice the indexer's HSI pool. Only slice 0 runs the attention.
    if (blockIdx.y != 0u) {
        return;
    }

    // Format 2: the CSA union. The score array holds the supplied window rows first (all attended,
    // oldest first, engine format) and then the selected compressed candidates (absolute token
    // positions, attended when candidate <= position, FP4 latent). One f32 three-pass softmax runs
    // over that union: maximum, normalising sum plus the sink after the maximum, weighted values.
    if (MODE != SparkMlaBase || window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION) {
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
                const uint8_t *row_scales =
                    window_scales + (uint64_t)slot * SPARK_M4_MLA_WINDOW_SCALES;
                for (uint32_t blk = 0u; blk < SPARK_M4_MLA_WINDOW_SCALES; ++blk) {
                    const float scale = spark_ue8m0_decode(row_scales[blk]);
                    const uint32_t base = blk * SPARK_M4_MLA_WINDOW_QUANT_BLOCK;
#pragma unroll 8
                    for (uint32_t ch = 0u; ch < SPARK_M4_MLA_WINDOW_QUANT_BLOCK; ++ch) {
                        const uint32_t i = base + ch;
                        const float key = __bfloat162float(__float2bfloat16_rn(
                            spark_fp8_e4m3_decode(row_codes[i]) * scale));
                        score += s_q[i] * key;
                    }
                }
                score *= SPARK_M4_MLA_SCORE_SCALE;
            } else {
                const uint32_t candidate = s_candidates[slot - window_rows];
                if (spark_mla_union_attends(position, candidate) != 0u) {
                    const uint8_t *row_fp4 = kv_fp4 + (uint64_t)candidate *
                                                           (SPARK_M4_MLA_LATENT /
                                                            SPARK_M4_MLA_FP4_PER_BYTE);
                    const uint8_t *row_scales =
                        kv_scales + (uint64_t)candidate *
                                        (SPARK_M4_MLA_LATENT / SPARK_M4_KV_SCALE_CHANNELS);
                    for (uint32_t grp = 0u; grp < (SPARK_M4_MLA_LATENT / SPARK_M4_KV_SCALE_CHANNELS);
                         ++grp) {
                        const float scale = spark_fp8_e4m3_decode(row_scales[grp]);
                        const uint32_t byte_base =
                            grp * (SPARK_M4_KV_SCALE_CHANNELS / SPARK_M4_MLA_FP4_PER_BYTE);
                        const uint32_t ch_base = grp * SPARK_M4_KV_SCALE_CHANNELS;
#pragma unroll
                        for (uint32_t b = 0u;
                             b < (SPARK_M4_KV_SCALE_CHANNELS / SPARK_M4_MLA_FP4_PER_BYTE); ++b) {
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
            if constexpr (ROUND_PROB) {
                // kernel.py:339-341: the released body copies the f32 probabilities into a bf16
                // tile before the P*V GEMM, so the value sum consumes the rounded probability. The
                // normalising sum keeps the unrounded value on purpose.
                s_scores[slot] = __bfloat162float(__float2bfloat16_rn(probability));
            } else {
                s_scores[slot] = probability;
            }
            local_sum += probability;
        }
        local_sum = spark_block_sum(local_sum, lane, SPARK_M4_MLA_THREADS);
        if (lane == 0u) {
            s_sum = local_sum > 0.0f ? local_sum : 1.0f;
        }
        __syncthreads();
        // Each thread owns two latent channels and walks the union in ascending slot order:
        // window rows first, then the selected list as supplied.
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
            const uint8_t *row_scales =
                window_scales + (uint64_t)slot * SPARK_M4_MLA_WINDOW_SCALES;
            const float v0 = __bfloat162float(__float2bfloat16_rn(
                spark_fp8_e4m3_decode(row_codes[i0]) *
                spark_ue8m0_decode(row_scales[w_scale_off0])));
            const float v1 = __bfloat162float(__float2bfloat16_rn(
                spark_fp8_e4m3_decode(row_codes[i1]) *
                spark_ue8m0_decode(row_scales[w_scale_off1])));
            acc0 += prob * v0;
            acc1 += prob * v1;
        }
        for (uint32_t idx = 0u; idx < selected_count; ++idx) {
            const uint32_t candidate = s_candidates[idx];
            if (spark_mla_union_attends(position, candidate) != 0u) {
                const float prob = s_scores[window_rows + idx];
                const uint8_t *row_fp4 = kv_fp4 + (uint64_t)candidate *
                                                       (SPARK_M4_MLA_LATENT /
                                                        SPARK_M4_MLA_FP4_PER_BYTE);
                const uint8_t *row_scales =
                    kv_scales + (uint64_t)candidate *
                                    (SPARK_M4_MLA_LATENT / SPARK_M4_KV_SCALE_CHANNELS);
                const uint8_t nib0 = (uint8_t)((row_fp4[fp4_byte_off0] >> fp4_shift) & 0x0Fu);
                const uint8_t nib1 = (uint8_t)((row_fp4[fp4_byte_off1] >> fp4_shift) & 0x0Fu);
                const float v0 =
                    spark_fp4_e2m1_decode(nib0) * spark_fp8_e4m3_decode(row_scales[kv_scale_off0]);
                const float v1 =
                    spark_fp4_e2m1_decode(nib1) * spark_fp8_e4m3_decode(row_scales[kv_scale_off1]);
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

    if constexpr (BASE) {
        // Stage 2 for the historical window-only formats (0/1): attention over the selected
        // candidates, scores bounded by SPARK_M4_MLA_MAX_SELECTED.
        for (uint32_t slot = lane; slot < selected_count; slot += SPARK_M4_MLA_THREADS) {
            const uint32_t candidate = selected[slot];
            float score = 0.0f;
            if (spark_mla_attends(window_kv_format, position, candidate, window_rows) != 0u) {
                for (uint32_t i = 0; i < SPARK_M4_MLA_LATENT; ++i) {
                    const float key =
                        window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4
                            ? spark_mla_load_latent(
                                  kv_fp4 + (uint64_t)candidate * (SPARK_M4_MLA_LATENT / 2u), i) *
                                  spark_mla_scale(
                                      kv_scales + (uint64_t)candidate *
                                                      (SPARK_M4_MLA_LATENT /
                                                       SPARK_M4_KV_SCALE_CHANNELS),
                                      i)
                            : spark_mla_load_window_stored(window_codes, window_scales, candidate,
                                                            i);
                    score += __bfloat162float(q[(uint64_t)head * SPARK_M4_MLA_LATENT + i]) * key;
                }
                score *= SPARK_M4_MLA_SCORE_SCALE;
            } else {
                score = -3.402823466e38f;
            }
            s_scores[slot] = score;
        }
        __syncthreads();
        float local_max = -3.402823466e38f;
        for (uint32_t slot = lane; slot < selected_count; slot += SPARK_M4_MLA_THREADS) {
            local_max = s_scores[slot] > local_max ? s_scores[slot] : local_max;
        }
        local_max = spark_block_max(local_max, lane, SPARK_M4_MLA_THREADS);
        if (lane == 0u) {
            s_max = local_max;
        }
        __syncthreads();
        float local_sum = lane == 0u ? spark_exp(sink[head] - s_max) : 0.0f;
        for (uint32_t slot = lane; slot < selected_count; slot += SPARK_M4_MLA_THREADS) {
            const float probability = spark_exp(s_scores[slot] - s_max);
            s_scores[slot] = probability;
            local_sum += probability;
        }
        local_sum = spark_block_sum(local_sum, lane, SPARK_M4_MLA_THREADS);
        if (lane == 0u) {
            s_sum = local_sum > 0.0f ? local_sum : 1.0f;
        }
        __syncthreads();
        for (uint32_t i = lane; i < SPARK_M4_MLA_LATENT; i += SPARK_M4_MLA_THREADS) {
            float acc = 0.0f;
            for (uint32_t slot = 0; slot < selected_count; ++slot) {
                const uint32_t candidate = selected[slot];
                if (spark_mla_attends(window_kv_format, position, candidate, window_rows) != 0u) {
                    const float value =
                        window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4
                            ? spark_mla_load_latent(
                                  kv_fp4 + (uint64_t)candidate * (SPARK_M4_MLA_LATENT / 2u), i) *
                                  spark_mla_scale(
                                      kv_scales + (uint64_t)candidate *
                                                      (SPARK_M4_MLA_LATENT /
                                                       SPARK_M4_KV_SCALE_CHANNELS),
                                      i)
                            : spark_mla_load_window_stored(window_codes, window_scales, candidate,
                                                            i);
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
}

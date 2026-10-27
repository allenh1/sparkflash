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

// kernels/opt_attn_sm121.cu
//
// opt_flash_mla_csa2_seq: flash_mla_csa2_seq_sm121's CSA-union decode attention with the slot loops
// spread over warps. Same 19-word ABI and the same grid (one block of 256 per head).
//
//   scores  warp per slot, lane per 16 channels: one 16 B (window E4M3) or 8 B (store E2M1) load,
//           sixteen products, a warp tree; the slot's key values are the reference's (window:
//           bf16(e4m3 * ue8m0), store: e2m1 * e4m3 scale), only the summation order differs.
//   softmax as the reference: max, exp(score - max) rounded to bf16 for the weighted sum, the sum
//           of the unrounded exps plus the sink term.
//   values  warp per slot again, sixteen channel accumulators per lane, then the eight warp partials
//           are folded in warp order through shared memory.
// The HSI index-scoring prologue is the reference's code unchanged (it publishes block_scores and
// block_best for the indexer).

#include "opt_pdl_sm121.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "m4_common_sm121.cuh"

#define OA_LATENT 512u
#define OA_THREADS 256u
#define OA_WARPS 8u
#define OA_MAX_SELECTED 512u
#define OA_MAX_ATTENDED 640u
#define OA_WINDOW 128u
#define OA_WSCALES 16u
#define OA_KSCALES 32u
#define OA_CSA_UNION 2u

__device__ __forceinline__ void oa_e4m3x4(uint32_t w, float *f) {
    uint32_t h01, h23;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h01) : "h"((uint16_t)(w & 0xffffu)));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h23) : "h"((uint16_t)(w >> 16)));
    const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&h01));
    const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&h23));
    f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}

__device__ __forceinline__ float oa_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float oa_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, v, o);
        v = other > v ? other : v;
    }
    return v;
}

// The sixteen key/value channels [16 * lane, 16 * lane + 16) of union slot `slot`.
__device__ __forceinline__ void oa_row16(uint32_t slot, uint32_t window_rows, uint32_t lane,
                                         const uint8_t *__restrict__ window_codes,
                                         const uint8_t *__restrict__ window_scales,
                                         const uint8_t *__restrict__ kv_fp4,
                                         const uint8_t *__restrict__ kv_scales,
                                         const uint32_t *s_candidates, float k[16]) {
    if (slot < window_rows) {
        const uint4 codes = __ldg(reinterpret_cast<const uint4 *>(window_codes + (uint64_t)slot * OA_LATENT) + lane);
        const float scale = spark_ue8m0_decode(window_scales[(uint64_t)slot * OA_WSCALES + (lane >> 1)]);
        oa_e4m3x4(codes.x, k);
        oa_e4m3x4(codes.y, k + 4);
        oa_e4m3x4(codes.z, k + 8);
        oa_e4m3x4(codes.w, k + 12);
#pragma unroll
        for (int i = 0; i < 16; ++i) k[i] = __bfloat162float(__float2bfloat16_rn(k[i] * scale));
    } else {
        const uint32_t candidate = s_candidates[slot - window_rows];
        const uint2 packed = __ldg(reinterpret_cast<const uint2 *>(kv_fp4 + (uint64_t)candidate * (OA_LATENT / 2u)) + lane);
        const float scale = spark_fp8_e4m3_decode(kv_scales[(uint64_t)candidate * OA_KSCALES + lane]);
        const uint32_t w[2] = {packed.x, packed.y};
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const uint8_t byte = (uint8_t)(w[j] >> (8 * b));
                k[8 * j + 2 * b] = spark_fp4_e2m1_decode((uint8_t)(byte & 0x0Fu)) * scale;
                k[8 * j + 2 * b + 1] = spark_fp4_e2m1_decode((uint8_t)(byte >> 4)) * scale;
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(OA_THREADS) opt_flash_mla_csa2_seq(
    const __nv_bfloat16 *__restrict__ q,
    const uint8_t *__restrict__ kv_fp4,
    const uint8_t *__restrict__ kv_scales,
    const uint8_t *__restrict__ window_codes,
    const uint8_t *__restrict__ window_scales,
    const uint32_t window_kv_format,
    const uint32_t window_rows,
    const uint32_t *__restrict__ selected,
    const __nv_bfloat16 *__restrict__ index_key,
    const __nv_bfloat16 *__restrict__ index_query,
    const float *__restrict__ sink,
    float *__restrict__ block_scores,
    uint32_t *__restrict__ block_best,
    __nv_bfloat16 *__restrict__ out,
    const uint32_t position,
    const uint32_t selected_count,
    void *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    OPT_PDL_ENTRY();
    (void)trace_ring; (void)layer_id; (void)step_seq;
    const uint32_t head = blockIdx.x;
    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31u;
    const uint32_t warp = t >> 5;
    if (window_kv_format != OA_CSA_UNION || window_rows > OA_WINDOW || selected_count > OA_MAX_SELECTED ||
        window_rows + selected_count > OA_MAX_ATTENDED) {
        for (uint32_t i = t; i < OA_LATENT; i += OA_THREADS)
            out[(uint64_t)head * OA_LATENT + i] = __float2bfloat16_rn(__uint_as_float(0x7FC00000u));
        return;
    }
    __shared__ float s_scores[OA_MAX_ATTENDED];
    __shared__ __align__(16) float s_q[OA_LATENT];
    __shared__ uint32_t s_candidates[OA_MAX_SELECTED];
    __shared__ __align__(16) float s_red[OA_WARPS][OA_LATENT];
    __shared__ float s_wmax[OA_WARPS];
    __shared__ float s_wsum[OA_WARPS];

    // The HSI index-scoring prologue (flash_mla_csa2_seq_sm121.cu, unchanged).
    {
        const uint32_t worker = ((blockIdx.y * (uint32_t)SPARK_M4_HEADS + blockIdx.x) * OA_WARPS) + warp;
        const uint32_t block_stride = (uint32_t)SPARK_M4_HEADS * OA_WARPS;
        const float q0 = __bfloat162float(index_query[lane]);
        const float q1 = __bfloat162float(index_query[lane + 32u]);
        const float q2 = __bfloat162float(index_query[lane + 64u]);
        const float q3 = __bfloat162float(index_query[lane + 96u]);
#pragma unroll
        for (uint32_t round = 0u; round < ((uint32_t)SPARK_M4_HSI_BLOCKS / block_stride); ++round) {
            const uint32_t block = worker + round * block_stride;
            if (block >= (uint32_t)SPARK_M4_HSI_BLOCKS) continue;
            const uint32_t first_token = block * SPARK_M4_HSI_BLOCK_TOKENS;
            float best = -3.402823466e38f;
            uint32_t best_token = 0u;
            if (first_token <= position) {
                for (uint32_t slot = 0; slot < (uint32_t)SPARK_M4_HSI_BLOCK_TOKENS; ++slot) {
                    const uint32_t token = first_token + slot;
                    if (token > position) break;
                    const __nv_bfloat16 *key_row =
                        index_key + ((uint64_t)block * SPARK_M4_HSI_BLOCK_TOKENS + slot) * SPARK_M4_INDEX_HEAD_DIM;
                    float score = q0 * __bfloat162float(key_row[lane]) + q1 * __bfloat162float(key_row[lane + 32u]) +
                                  q2 * __bfloat162float(key_row[lane + 64u]) +
                                  q3 * __bfloat162float(key_row[lane + 96u]);
                    score += __shfl_down_sync(0xffffffffu, score, 16u);
                    score += __shfl_down_sync(0xffffffffu, score, 8u);
                    score += __shfl_down_sync(0xffffffffu, score, 4u);
                    score += __shfl_down_sync(0xffffffffu, score, 2u);
                    score += __shfl_down_sync(0xffffffffu, score, 1u);
                    if (lane == 0u && score > best) { best = score; best_token = token; }
                }
            }
            if (lane == 0u) { block_scores[block] = best; block_best[block] = best_token; }
        }
    }
    if (blockIdx.y != 0u) return;

    const uint32_t union_total = window_rows + selected_count;
    const __nv_bfloat16 *q_head = q + (uint64_t)head * OA_LATENT;
    s_q[t] = __bfloat162float(q_head[t]);
    s_q[t + OA_THREADS] = __bfloat162float(q_head[t + OA_THREADS]);
    for (uint32_t i = t; i < selected_count; i += OA_THREADS) s_candidates[i] = selected[i];
    __syncthreads();

    float qv[16];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 v = reinterpret_cast<const float4 *>(s_q)[4u * lane + j];
        qv[4 * j] = v.x; qv[4 * j + 1] = v.y; qv[4 * j + 2] = v.z; qv[4 * j + 3] = v.w;
    }

    float wmax = -3.402823466e38f;
    for (uint32_t slot = warp; slot < union_total; slot += OA_WARPS) {
        float score;
        if (slot >= window_rows && s_candidates[slot - window_rows] > position) {
            score = -3.402823466e38f;
        } else {
            float k[16];
            oa_row16(slot, window_rows, lane, window_codes, window_scales, kv_fp4, kv_scales, s_candidates, k);
            float partial = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; ++i) partial = fmaf(qv[i], k[i], partial);
            score = oa_warp_sum(partial) * SPARK_M4_MLA_SCORE_SCALE;
        }
        if (lane == 0u) s_scores[slot] = score;
        wmax = score > wmax ? score : wmax;
    }
    if (lane == 0u) s_wmax[warp] = wmax;
    __syncthreads();
    float smax = s_wmax[0];
#pragma unroll
    for (uint32_t w = 1; w < OA_WARPS; ++w) smax = s_wmax[w] > smax ? s_wmax[w] : smax;

    float lsum = 0.0f;
    for (uint32_t slot = t; slot < union_total; slot += OA_THREADS) {
        const float probability = spark_exp(s_scores[slot] - smax);
        s_scores[slot] = __bfloat162float(__float2bfloat16_rn(probability));
        lsum += probability;
    }
    lsum = oa_warp_sum(lsum);
    if (lane == 0u) s_wsum[warp] = lsum;
    __syncthreads();
    float ssum = spark_exp(sink[head] - smax);
#pragma unroll
    for (uint32_t w = 0; w < OA_WARPS; ++w) ssum += s_wsum[w];
    ssum = ssum > 0.0f ? ssum : 1.0f;

    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    for (uint32_t slot = warp; slot < union_total; slot += OA_WARPS) {
        if (slot >= window_rows && s_candidates[slot - window_rows] > position) continue;
        const float prob = s_scores[slot];
        float v[16];
        oa_row16(slot, window_rows, lane, window_codes, window_scales, kv_fp4, kv_scales, s_candidates, v);
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] = fmaf(prob, v[i], acc[i]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        reinterpret_cast<float4 *>(s_red[warp])[4u * lane + j] =
            make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]);
    }
    __syncthreads();
#pragma unroll
    for (uint32_t r = 0; r < 2; ++r) {
        const uint32_t i = t + r * OA_THREADS;
        float total = s_red[0][i];
#pragma unroll
        for (uint32_t w = 1; w < OA_WARPS; ++w) total += s_red[w][i];
        out[(uint64_t)head * OA_LATENT + i] = __float2bfloat16_rn(total / ssum);
    }
}

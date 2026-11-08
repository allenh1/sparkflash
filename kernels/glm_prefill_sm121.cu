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

// GLM-5.3-Flash's bulk dense-prefill kernels.
//
// The decode arm reads every weight once per eight rows. A prefill chunk of 128-1024 rows can
// read each weight once instead, which is what makes GLM's 288-expert MoE affordable: at 512 rows
// every expert is routed to and its 4.2 MiB rank slice is read once for the whole chunk.
//
// The dense projection is an m16n8k16 bf16 MMA over an f32 activation split into a bf16 head and
// remainder. A block-scaled FP8 weight (GLM's E4M3 codes with one f32 scale a 128 x 128 block) is
// widened to bf16 in shared memory with its block scale applied; the scale is constant over the
// 128 x 128 block, so one scale a stage is exact. This is the b12x SM121 tile pattern the tree's
// other packs use, with GLM's block-scaled weight layout.
//
// The chunk's routed-expert (MoE) kernels live in their own part image
// (kernels/glm_moe_sm121.cu), and its sparse MLA attention and per-head projections in
// kernels/glm_attention_sm121.cu. This translation unit keeps the dense GEMM, the skinny
// projection, the exact projections of the bulk prefill's router and indexer, and
// glm_quantise_rows, the dynamic activation quantiser both q8 GEMMs read.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "opt_mma_common_sm121.cuh"
#include "opt_mma_bf16_sm121.cuh"
#include "glm_common_sm121.cuh"
#include "q4e_gemv_sm121.cuh"


extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_prefill_bf16_mma(
    const __nv_bfloat16 *__restrict__ w, const float *__restrict__ a,
    float *__restrict__ out, uint32_t cols, uint32_t inner, uint32_t tokens) {
    glm_mma_body<false, false, true>(a, reinterpret_cast<const uint8_t *>(w), nullptr, nullptr,
                                     nullptr, out, cols, inner, tokens, inner, cols, 0, 0);
}

// The same body over the bf16 head of each activation alone: the low half of the split costs a
// second MMA chain a stage and no weight read, so a caller that can spend the precision reads
// every weight once for one chain (GLM's own weights are FP8 with a dynamic activation scale).
extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_prefill_bf16_mma_bf16a(
    const __nv_bfloat16 *__restrict__ w, const float *__restrict__ a,
    float *__restrict__ out, uint32_t cols, uint32_t inner, uint32_t tokens) {
    glm_mma_body<false, false, false>(a, reinterpret_cast<const uint8_t *>(w), nullptr, nullptr,
                                      nullptr, out, cols, inner, tokens, inner, cols, 0, 0);
}

// The two above over the wide tile (shared_prefill_wide_body), their bits, a grid of
// (cols / 256, rows / 128) and GLM_WIDE_SHARED bytes of dynamic shared memory: the bulk prefill's
// dense BF16 projections launch these.
extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_prefill_bf16_wide(
    const __nv_bfloat16 *__restrict__ w, const float *__restrict__ a,
    float *__restrict__ out, uint32_t cols, uint32_t inner, uint32_t tokens) {
    shared_prefill_wide_body<true>(w, a, out, cols, inner, tokens);
}

extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_prefill_bf16_wide_bf16a(
    const __nv_bfloat16 *__restrict__ w, const float *__restrict__ a,
    float *__restrict__ out, uint32_t cols, uint32_t inner, uint32_t tokens) {
    shared_prefill_wide_body<false>(w, a, out, cols, inner, tokens);
}

extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_prefill_fp8b128_mma(
    const uint8_t *__restrict__ w, const float *__restrict__ ws,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    uint32_t tokens) {
    glm_mma_body<true, false, true>(a, w, ws, nullptr, nullptr, out, cols, inner, tokens, inner,
                                    cols, 0, 0);
}

extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_prefill_fp8b128_mma_bf16a(
    const uint8_t *__restrict__ w, const float *__restrict__ ws,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    uint32_t tokens) {
    glm_mma_body<true, false, false>(a, w, ws, nullptr, nullptr, out, cols, inner, tokens, inner,
                                     cols, 0, 0);
}

// ---- FP8 x FP8: the checkpoint's own dynamic activation scheme ----
//
// GLM's FP8 release quantises each activation row per 128 columns (scale = amax / 448, codes
// e4m3(x / scale)), as the decode GEMV's q4e_tg_fp8_activation does. The prefill's FP8 projections
// take those codes straight into an E4M3 MMA (om_mma_e4m3 at unit ue8m0 scales), and every
// 128-column block's sum is folded into the f32 total times its activation and weight scales.

// x ([rows][cols] f32, cols a multiple of 128) as codes q ([rows][cols]) and scales s
// ([rows][cols / 128]). One warp a 128-column block.
extern "C" __global__ void glm_quantise_rows(const float *__restrict__ x, uint8_t *__restrict__ q,
                                             float *__restrict__ s, int rows, int cols) {
    const int unit = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31u);
    const int per_row = cols >> 7;
    if (unit >= rows * per_row) return;
    const uint64_t at = (uint64_t)unit * 128u + (uint64_t)lane * 4u;
    float scale;
    const uint32_t codes = glm_q8_codes(*reinterpret_cast<const float4 *>(x + at), scale);
    *reinterpret_cast<uint32_t *>(q + at) = codes;
    if (lane == 0) s[unit] = scale;
}


// The dense FP8 projection over quantised activations: grid (ceil(cols / 128), ceil(tokens / 128)).
extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_prefill_q8_mma(
    const uint8_t *__restrict__ w, const float *__restrict__ ws, const uint8_t *__restrict__ a,
    const float *__restrict__ as, float *__restrict__ out, uint32_t cols, uint32_t inner,
    uint32_t tokens) {
    glm_q8_body<false>(a, as, w, ws, nullptr, nullptr, out, cols, inner, tokens, nullptr, 1u,
                       nullptr);
}

// ---- Skinny projections ----

// out = W x for a BF16 W of `outs` <= 8 JT rows (a 24-row mixer, a 16-row gate: shapes a 128-column
// tile would mostly waste) over `rows` f32 rows of `columns` (a multiple of 8): one warp two rows,
// eight columns a lane a 256-column round, so every activation is read once. out is [rows][outs].
//
// The block stages each round's weights (outs rows of 256 columns, at most 16 KiB) in shared
// memory once, holding the next round's in registers while the warps multiply, and every warp
// reads them from there. Each warp loading them itself through L1 moved eight times the weight
// bytes from L2 as the activations from DRAM: 3.16 ms against 1.12 for a 4,096-row chunk of the
// 24-row mixer over 16,384 columns, the DRAM rate. A lane's sums are the same chain either way.
template <uint32_t JT>
__device__ __forceinline__ void glm_skinny_body(const __nv_bfloat16 *__restrict__ w,
                                                const float *__restrict__ x,
                                                float *__restrict__ out, uint32_t outs,
                                                uint32_t columns, uint32_t rows) {
    constexpr uint32_t J = JT * 8u;
    // [j][lane]: row j's eight columns of lane `lane` in the round, one 16-byte word.
    __shared__ __align__(16) uint4 staged[J * 32u];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 2u;
    // A warp past the rows still stages its share of every round.
    const bool live = row0 < rows;
    const bool second = row0 + 1u < rows;
    const float *x0 = x + (uint64_t)(live ? row0 : 0u) * columns;
    const float *x1 = second ? x0 + columns : x0;
    float acc[2][J];
#pragma unroll
    for (uint32_t j = 0; j < J; ++j) acc[0][j] = acc[1][j] = 0.0f;
    // Word i of a round is row i / 32's columns 8 (i % 32)..+8; thread t holds words t + 256 m.
    uint4 next[JT];
#pragma unroll
    for (uint32_t m = 0; m < JT; ++m) next[m] = make_uint4(0u, 0u, 0u, 0u);
    const auto fetch = [&](uint32_t k0) {
#pragma unroll
        for (uint32_t m = 0; m < JT; ++m) {
            const uint32_t word = threadIdx.x + 256u * m;
            const uint32_t j = word >> 5;
            const uint32_t k = k0 + (word & 31u) * 8u;
            if (j < outs && k < columns) {
                next[m] = __ldg(reinterpret_cast<const uint4 *>(w + (uint64_t)j * columns + k));
            }
        }
    };
    fetch(0u);
    for (uint32_t k0 = 0; k0 < columns; k0 += 256u) {
        __syncthreads();
#pragma unroll
        for (uint32_t m = 0; m < JT; ++m) staged[threadIdx.x + 256u * m] = next[m];
        __syncthreads();
        if (k0 + 256u < columns) fetch(k0 + 256u);
        const uint32_t k = k0 + lane * 8u;
        if (!live || k >= columns) continue;
        const float4 a0 = *reinterpret_cast<const float4 *>(x0 + k);
        const float4 a1 = *reinterpret_cast<const float4 *>(x0 + k + 4u);
        const float4 b0 = *reinterpret_cast<const float4 *>(x1 + k);
        const float4 b1 = *reinterpret_cast<const float4 *>(x1 + k + 4u);
        const float xa[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const float xb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (uint32_t j = 0; j < J; ++j) {
            if (j >= outs) break;
            const uint4 raw = staged[j * 32u + lane];
            const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
            for (uint32_t p = 0; p < 4u; ++p) {
                const float lo = __uint_as_float(words[p] << 16);
                const float hi = __uint_as_float(words[p] & 0xFFFF0000u);
                acc[0][j] = fmaf(lo, xa[2u * p], acc[0][j]);
                acc[0][j] = fmaf(hi, xa[2u * p + 1u], acc[0][j]);
                acc[1][j] = fmaf(lo, xb[2u * p], acc[1][j]);
                acc[1][j] = fmaf(hi, xb[2u * p + 1u], acc[1][j]);
            }
        }
    }
    if (!live) return;
#pragma unroll
    for (uint32_t j = 0; j < J; ++j) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[0][j] += __shfl_xor_sync(0xFFFFFFFFu, acc[0][j], offset);
            acc[1][j] += __shfl_xor_sync(0xFFFFFFFFu, acc[1][j], offset);
        }
    }
    // Lane j writes output j of both rows.
#pragma unroll
    for (uint32_t j = 0; j < J; ++j) {
        if (lane == j && j < outs) {
            out[(uint64_t)row0 * outs + j] = acc[0][j];
            if (second) out[(uint64_t)(row0 + 1u) * outs + j] = acc[1][j];
        }
    }
}

// grid (ceil(rows / 16)), 256 threads; the entry is the one whose 8 JT covers `outs`.
#define GLM_SKINNY_ENTRY(JT)                                                                       \
    extern "C" __global__ __launch_bounds__(256) void glm_prefill_skinny_bf16_##JT(               \
        const __nv_bfloat16 *__restrict__ w, const float *__restrict__ x,                          \
        float *__restrict__ out, uint32_t outs, uint32_t columns, uint32_t rows) {                 \
        glm_skinny_body<JT>(w, x, out, outs, columns, rows);                                       \
    }
GLM_SKINNY_ENTRY(1)
GLM_SKINNY_ENTRY(2)
GLM_SKINNY_ENTRY(3)
GLM_SKINNY_ENTRY(4)

// ---- Exact projections ----

// out = W x for a BF16 W ([outs][columns], columns a multiple of 256) over `rows` f32 rows, as the
// tiled GEMV (q4e_gemv_tiled_bf16_parts) computes it when its split count is the matrix's round
// count, one 256-column round a split: GLM's router and the indexer's parts, whose decode-step
// floats a verified row is defined by. There a token's sum is
//
//   split s's partial = the xor butterfly over lanes l of fma8(W[s 256 + 8 l ..), x[..), 0)
//   sum               = (0 + partial_0) + partial_1 + ...     (partial_0 alone for one round)
//
// and the tiled GEMV reaches it with five shuffles for every eight multiplies over 512-row
// groups: 3.46 ms for the router's 288 rows over a 4,096-row chunk. Here a thread holds whole
// sums instead. Thread (rg = lane & 7, kg = (lane >> 3) + 4 warp) owns outputs rg + 8 i of the
// block's 32 and rows kg + 16 k of its 32. The butterfly is the balanced pairwise sum of the lanes
// in bit-reversed order (leaf j is lane rev5(j)), so the thread evaluates the 32 lane chains
// leaf by leaf and folds them with a carry over five levels: leaf j adds the pending subtree at
// each set low bit of j, as the butterfly adds its halves. The block stages eight leaves of a round
// at a time (its weights and activations, 12 KiB) by cp.async into one of two buffers while the
// other is multiplied: 1.37 ms for the router, 0.64 for a 128-row indexer part.
//
// grid (ceil(outs / 32), ceil(rows / 32)), 128 threads; out is [rows][outs].
#define GPE_OUTS 32
#define GPE_ROWS 32
#define GPE_LEAVES 8
#define GPE_THREADS 128

__device__ __forceinline__ constexpr uint32_t gpe_lane(uint32_t j) {
    return ((j & 1u) << 4) | ((j & 2u) << 2) | (j & 4u) | ((j & 8u) >> 2) | ((j & 16u) >> 4);
}

extern "C" __global__ __launch_bounds__(GPE_THREADS) void glm_prefill_exact_bf16(
    const __nv_bfloat16 *__restrict__ w, const float *__restrict__ x, float *__restrict__ out,
    uint32_t outs, uint32_t columns, uint32_t rows) {
    constexpr uint32_t kOuts = GPE_OUTS / 8u;
    constexpr uint32_t kRows = GPE_ROWS / 16u;
    constexpr uint32_t kStages = 32u / GPE_LEAVES;
    // [buffer][output][leaf ^ (output & 7)]: an output's eight columns of a leaf's lane.
    __shared__ __align__(16) uint4 ws[2][GPE_OUTS * GPE_LEAVES];
    // [buffer][(row * 8 + (leaf ^ (row & 7))) * 2 + half]: a row's four columns of each half.
    __shared__ __align__(16) float4 xs[2][GPE_ROWS * GPE_LEAVES * 2];
    const uint32_t out0 = blockIdx.x * GPE_OUTS;
    const uint32_t row0 = blockIdx.y * GPE_ROWS;
    const uint32_t rounds = columns >> 8;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t rg = lane & 7u;
    const uint32_t kg = (lane >> 3) + 4u * (threadIdx.x >> 5);
    // Stage (round, g): leaves 8 g .. 8 g + 8 of the round.
    const auto stage = [&](uint32_t round, uint32_t g, uint32_t buffer) {
        for (uint32_t index = threadIdx.x; index < GPE_OUTS * GPE_LEAVES; index += GPE_THREADS) {
            const uint32_t o = index / GPE_LEAVES, leaf = index % GPE_LEAVES;
            const bool valid = out0 + o < outs;
            const uint32_t column = round * 256u + 8u * gpe_lane(g * GPE_LEAVES + leaf);
            om_cp16(om_smem(&ws[buffer][o * GPE_LEAVES + (leaf ^ (o & 7u))]),
                    w + (uint64_t)(valid ? out0 + o : 0u) * columns + column, valid);
        }
        for (uint32_t index = threadIdx.x; index < GPE_ROWS * GPE_LEAVES * 2u;
             index += GPE_THREADS) {
            const uint32_t half = index & 1u, e = index >> 1;
            const uint32_t r = e / GPE_LEAVES, leaf = e % GPE_LEAVES;
            const bool valid = row0 + r < rows;
            const uint32_t column = round * 256u + 8u * gpe_lane(g * GPE_LEAVES + leaf) + 4u * half;
            om_cp16(om_smem(&xs[buffer][(r * GPE_LEAVES + (leaf ^ (r & 7u))) * 2u + half]),
                    x + (uint64_t)(valid ? row0 + r : 0u) * columns + column, valid);
        }
        om_commit();
    };
    float total[kOuts][kRows];
#pragma unroll
    for (uint32_t i = 0; i < kOuts; ++i)
#pragma unroll
        for (uint32_t k = 0; k < kRows; ++k) total[i][k] = 0.0f;
    stage(0u, 0u, 0u);
    uint32_t buffer = 0u;
    for (uint32_t round = 0; round < rounds; ++round) {
        float pending[5][kOuts][kRows];
        float sum[kOuts][kRows];
#pragma unroll
        for (uint32_t g = 0; g < kStages; ++g) {
            if (g + 1u < kStages) {
                stage(round, g + 1u, buffer ^ 1u);
            } else if (round + 1u < rounds) {
                stage(round + 1u, 0u, buffer ^ 1u);
            } else {
                om_commit();
            }
            om_wait<1>();
            __syncthreads();
#pragma unroll
            for (uint32_t leaf = 0; leaf < GPE_LEAVES; ++leaf) {
                const uint32_t j = g * GPE_LEAVES + leaf;
                uint4 wv[kOuts];
                float4 xv[kRows][2];
#pragma unroll
                for (uint32_t i = 0; i < kOuts; ++i)
                    wv[i] = ws[buffer][(rg + 8u * i) * GPE_LEAVES + (leaf ^ rg)];
#pragma unroll
                for (uint32_t k = 0; k < kRows; ++k) {
                    const uint32_t r = kg + 16u * k;
                    const uint32_t at = (r * GPE_LEAVES + (leaf ^ (r & 7u))) * 2u;
                    xv[k][0] = xs[buffer][at];
                    xv[k][1] = xs[buffer][at + 1u];
                }
#pragma unroll
                for (uint32_t i = 0; i < kOuts; ++i) {
#pragma unroll
                    for (uint32_t k = 0; k < kRows; ++k) {
                        float v = q4e_tg_fma8(wv[i], xv[k][0], xv[k][1], 0.0f);
                        uint32_t level = 0;
#pragma unroll
                        for (; level < 5u; ++level) {
                            if (((j >> level) & 1u) == 0u) break;
                            v = pending[level][i][k] + v;
                        }
                        if (level < 5u) {
                            pending[level][i][k] = v;
                        } else {
                            sum[i][k] = v;
                        }
                    }
                }
            }
            __syncthreads();
            buffer ^= 1u;
        }
#pragma unroll
        for (uint32_t i = 0; i < kOuts; ++i)
#pragma unroll
            for (uint32_t k = 0; k < kRows; ++k)
                total[i][k] = rounds > 1u ? total[i][k] + sum[i][k] : sum[i][k];
    }
#pragma unroll
    for (uint32_t i = 0; i < kOuts; ++i) {
        const uint32_t o = out0 + rg + 8u * i;
#pragma unroll
        for (uint32_t k = 0; k < kRows; ++k) {
            const uint32_t r = row0 + kg + 16u * k;
            if (o < outs && r < rows) out[(uint64_t)r * outs + o] = total[i][k];
        }
    }
}

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
// projection and glm_quantise_rows, the dynamic activation quantiser both q8 GEMMs read.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "opt_mma_common_sm121.cuh"
#include "opt_mma_bf16_sm121.cuh"
#include "glm_common_sm121.cuh"


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
// eight columns a lane a 256-column round, so every activation is read once and the weight rows
// stream from L1. out is [rows][outs].
template <uint32_t JT>
__device__ __forceinline__ void glm_skinny_body(const __nv_bfloat16 *__restrict__ w,
                                                const float *__restrict__ x,
                                                float *__restrict__ out, uint32_t outs,
                                                uint32_t columns, uint32_t rows) {
    constexpr uint32_t J = JT * 8u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 2u;
    if (row0 >= rows) return;
    const bool second = row0 + 1u < rows;
    const float *x0 = x + (uint64_t)row0 * columns;
    const float *x1 = second ? x0 + columns : x0;
    float acc[2][J];
#pragma unroll
    for (uint32_t j = 0; j < J; ++j) acc[0][j] = acc[1][j] = 0.0f;
    for (uint32_t k = lane * 8u; k < columns; k += 256u) {
        const float4 a0 = *reinterpret_cast<const float4 *>(x0 + k);
        const float4 a1 = *reinterpret_cast<const float4 *>(x0 + k + 4u);
        const float4 b0 = *reinterpret_cast<const float4 *>(x1 + k);
        const float4 b1 = *reinterpret_cast<const float4 *>(x1 + k + 4u);
        const float xa[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const float xb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (uint32_t j = 0; j < J; ++j) {
            if (j >= outs) break;
            const uint4 raw = __ldg(reinterpret_cast<const uint4 *>(w + (uint64_t)j * columns + k));
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

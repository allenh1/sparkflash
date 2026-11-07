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

// GLM-5.3-Flash's bulk routed-expert (MoE) kernels.
//
// The decode arm reads every weight once per eight rows. A prefill chunk of 128-1024 rows can
// read each weight once instead, which is what makes GLM's 288-expert MoE affordable: at 512 rows
// every expert is routed to and its 4.2 MiB rank slice is read once for the whole chunk.
//
// These kernels were split out of glm_prefill_sm121.cu into their own part image, so the routed
// experts build and load independently of the dense GEMM. The routed experts and the dense
// projection share one tile body, so this translation unit carries its own copy of glm_mma_body
// and glm_q8_body and the geometry they read; it is self-contained.
//
// The routed experts run in items: one expert's up-to-128 sorted slots, one 128-row tile. The
// projection reads each item row straight from its token's row (the plan's sorted array names the
// slot), SwiGLU
// writes the padded [items][128][width] activations, and the down projection writes each kept
// row, times its routing weight, straight to its slot row. Rows past an item's count load as zero
// and are never written, and a warp whose rows are all past the count skips its MMAs.
//
// glm_quantise_rows, the dynamic activation quantiser the q8 GEMMs read, stays in the dense
// prefill image because the dense q8 GEMM uses it too; the MoE path loads it from there.
//
// The chunk's dense GEMM and skinny projections live in kernels/glm_prefill_sm121.cu, and its
// sparse MLA attention and per-head projections in kernels/glm_attention_sm121.cu.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "opt_mma_common_sm121.cuh"
#include "opt_mma_bf16_sm121.cuh"
#include "glm_common_sm121.cuh"
#include "shared_swiglu_sm121.cuh"


// The routed experts: table is the [weight, scales] pointer pairs, items the item list the
// shared plan (kernels/moe_plan_sm121.cu) writes, and sorted its rank-ordered slot list, which
// each item reads from its first rank. Grid (cols / 128, 1, items). Without route (gate/up), a
// row is its slot's token row of a ([tokens][inner]) and the output the padded [items][128][cols];
// with route (down), a is the padded [items][128][inner] and each row lands on its slot row of
// out ([slots][cols]) times route[slot].
extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_moe_fp8b128_mma(
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    const int *__restrict__ sorted, uint32_t topk, const float *__restrict__ route) {
    glm_mma_body<true, true, true>(a, nullptr, nullptr, table, items, out, cols, inner,
                                   GLM_MOE_ROWS, inner, cols, GLM_MOE_ROWS * (uint64_t)inner,
                                   route ? 0ull : GLM_MOE_ROWS * (uint64_t)cols, 0, 0, sorted,
                                   topk, route);
}

extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_moe_fp8b128_mma_bf16a(
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    const int *__restrict__ sorted, uint32_t topk, const float *__restrict__ route) {
    glm_mma_body<true, true, false>(a, nullptr, nullptr, table, items, out, cols, inner,
                                    GLM_MOE_ROWS, inner, cols, GLM_MOE_ROWS * (uint64_t)inner,
                                    route ? 0ull : GLM_MOE_ROWS * (uint64_t)cols, 0, 0, sorted,
                                    topk, route);
}


// SwiGLU the gate and up halves of every padded item row into its intermediate row.
extern "C" __global__ void glm_moe_swiglu(const float *__restrict__ gate_up, int width, int rows,
                                          float limit, float *__restrict__ out) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < rows * width;
         index += blockDim.x * gridDim.x) {
        const int row = index / width;
        const int column = index - row * width;
        out[index] = shared_swiglu_value<true, false>(
            gate_up[row * 2 * width + column], gate_up[row * 2 * width + width + column], limit);
    }
}

// ---- FP8 x FP8: the checkpoint's own dynamic activation scheme ----
//
// GLM's FP8 release quantises each activation row per 128 columns (scale = amax / 448, codes
// e4m3(x / scale)), as the decode GEMV's q4e_tg_fp8_activation does. The prefill's FP8 projections
// take those codes straight into an E4M3 MMA (om_mma_e4m3 at unit ue8m0 scales), and every
// 128-column block's sum is folded into the f32 total times its activation and weight scales.

// glm_moe_swiglu with its output quantised as the down projection's activation: codes q
// ([rows][width]) and scales s ([rows][width / 128]). One warp a 128-column block, width a multiple
// of 128.
extern "C" __global__ void glm_moe_swiglu_q8(const float *__restrict__ gate_up, int width,
                                             int rows, float limit, uint8_t *__restrict__ q,
                                             float *__restrict__ s) {
    const int unit = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31u);
    const int per_row = width >> 7;
    if (unit >= rows * per_row) return;
    const int row = unit / per_row;
    const int column = (unit - row * per_row) * 128 + lane * 4;
    const float *src = gate_up + (uint64_t)row * 2u * width + column;
    const float4 g4 = *reinterpret_cast<const float4 *>(src);
    const float4 u4 = *reinterpret_cast<const float4 *>(src + width);
    const float4 v = make_float4(
        shared_swiglu_value<true, false>(g4.x, u4.x, limit),
        shared_swiglu_value<true, false>(g4.y, u4.y, limit),
        shared_swiglu_value<true, false>(g4.z, u4.z, limit),
        shared_swiglu_value<true, false>(g4.w, u4.w, limit));
    float scale;
    const uint32_t codes = glm_q8_codes(v, scale);
    *reinterpret_cast<uint32_t *>(q + (uint64_t)row * width + column) = codes;
    if (lane == 0) s[unit] = scale;
}


// The routed experts over quantised activations: grid (cols / 128, 1, items), the arguments of
// glm_moe_fp8b128_mma with the activation's codes and scales.
extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_moe_q8_mma(
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    const uint8_t *__restrict__ a, const float *__restrict__ as, float *__restrict__ out,
    uint32_t cols, uint32_t inner, const int *__restrict__ sorted, uint32_t topk,
    const float *__restrict__ route) {
    glm_q8_body<true>(a, as, nullptr, nullptr, table, items, out, cols, inner, GLM_MOE_ROWS,
                      sorted, topk, route);
}


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

// kernels/routed_format_sm121.cu
//
// DeepSeek's routed experts in every stored format the released MXFP4 bodies do not read, on the
// one shared bulk-prefill tile (shared_prefill_mma_sm121.cuh's routed mode). The body, the tile,
// the staged weight words and each format's decode live in the header; this file only binds the
// routed mode to DeepSeek's expert table and slot layout:
//
//   * The table is [layer][expert][9] words: (weight, scales, wide) for w1, w3 and w2, in that
//     order. `wide` is the f32 bit pattern of an NVFP4 convention's tensor-wide factor (zero
//     words elsewhere), so the factor stays the expert's own.
//   * The plan is the shared moe_plan (kernels/moe_plan_sm121.cu) cut at GLM_MOE_ROWS slots an
//     item: `sorted[first + i]` is the slot (token * top_k + choice) of item row i.
//   * Gate and up read each slot's token row of the f32 FFN input and write that slot's row of a
//     [slots][intermediate] f32 plane, one launch a matrix. The base's clamped SwiGLU maps the
//     two planes into a third; down reads that plane by slot, multiplies the slot's routing
//     weight in and writes the slot's [hidden] partial, which the base's routed combine
//     (opt_moe_combine_sm121) sums in ascending expert id with the shared expert's addend.
//
// The activation stays f32 (the high and low BF16 halves, kLow), as the host reference's coded
// matmul multiplies an unquantised row. NVFP4 applies its tensor-wide factor to the output for
// both conventions (kWide 2): E2M1 times the E4M3 group scale is exact in the tile's BF16 word,
// and folding a non-power-of-two factor into it would round every weight. The Qwen and Inkling
// entries differ only in the format they claim; the factor each reads is the one its convention
// defines, computed at open.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "opt_mma_common_sm121.cuh"
#include "opt_pdl_sm121.cuh"
#include "shared_prefill_mma_sm121.cuh"

// The expert table's words a matrix and the three matrices' offsets in an expert's entry.
#define RF_TABLE_STRIDE 9
#define RF_GATE 0
#define RF_UP 3
#define RF_DOWN 6

template <int kFormat, typename ScaleT, int kWide>
__device__ __forceinline__ void rf_gate_up(const unsigned long long *__restrict__ table,
                                           const int4 *__restrict__ items,
                                           const float *__restrict__ a, float *__restrict__ out,
                                           uint32_t cols, uint32_t inner,
                                           const int *__restrict__ sorted, uint32_t topk,
                                           uint32_t matrix) {
    if (matrix == 0u) {
        shared_prefill_mma_body<kFormat, true, true, ScaleT, RF_TABLE_STRIDE, RF_GATE, false, true,
                                kWide>(a, nullptr, nullptr, table, items, out, cols, inner,
                                       GLM_MOE_ROWS, inner, cols, 0ull, 0ull, 0ull, 0ull, sorted,
                                       topk);
    } else {
        shared_prefill_mma_body<kFormat, true, true, ScaleT, RF_TABLE_STRIDE, RF_UP, false, true,
                                kWide>(a, nullptr, nullptr, table, items, out, cols, inner,
                                       GLM_MOE_ROWS, inner, cols, 0ull, 0ull, 0ull, 0ull, sorted,
                                       topk);
    }
}

template <int kFormat, typename ScaleT, int kWide>
__device__ __forceinline__ void rf_down(const unsigned long long *__restrict__ table,
                                        const int4 *__restrict__ items,
                                        const float *__restrict__ a,
                                        const float *__restrict__ route, float *__restrict__ out,
                                        uint32_t cols, uint32_t inner,
                                        const int *__restrict__ sorted, uint32_t topk) {
    shared_prefill_mma_body<kFormat, true, true, ScaleT, RF_TABLE_STRIDE, RF_DOWN, true, false,
                            kWide>(a, nullptr, nullptr, table, items, out, cols, inner,
                                   GLM_MOE_ROWS, inner, cols, 0ull, 0ull, 0ull, 0ull, sorted,
                                   topk, route, 0u);
}

// One gate/up and one down entry a format. gate/up: grid (ceil(intermediate / GLM_BN), 1,
// max_items), `matrix` 0 for w1 into `out`, 1 for w3. down: grid (ceil(hidden / GLM_BN), 1,
// max_items). Both take OM_THREADS threads.

// FP8 on 128 x 128 tiles, the inverse scale stored f32 (GLM's spelling).
extern "C" __global__ __launch_bounds__(OM_THREADS) void routed_format_gate_up_fp8_sm121(
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    const int *__restrict__ sorted, uint32_t topk, uint32_t matrix) {
    OPT_PDL_ENTRY();
    rf_gate_up<SPM_W_FP8, float, 0>(table, items, a, out, cols, inner, sorted, topk, matrix);
}

extern "C" __global__ __launch_bounds__(OM_THREADS) void routed_format_down_fp8_sm121(
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    const float *__restrict__ a, const float *__restrict__ route, float *__restrict__ out,
    uint32_t cols, uint32_t inner, const int *__restrict__ sorted, uint32_t topk) {
    OPT_PDL_ENTRY();
    rf_down<SPM_W_FP8, float, 0>(table, items, a, route, out, cols, inner, sorted, topk);
}

// FP8 on 128 x 128 tiles, the inverse scale stored BF16 (Qwen's spelling).
extern "C" __global__ __launch_bounds__(OM_THREADS) void routed_format_gate_up_fp8_bf16_sm121(
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    const int *__restrict__ sorted, uint32_t topk, uint32_t matrix) {
    OPT_PDL_ENTRY();
    rf_gate_up<SPM_W_FP8, __nv_bfloat16, 0>(table, items, a, out, cols, inner, sorted, topk, matrix);
}

extern "C" __global__ __launch_bounds__(OM_THREADS) void routed_format_down_fp8_bf16_sm121(
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    const float *__restrict__ a, const float *__restrict__ route, float *__restrict__ out,
    uint32_t cols, uint32_t inner, const int *__restrict__ sorted, uint32_t topk) {
    OPT_PDL_ENTRY();
    rf_down<SPM_W_FP8, __nv_bfloat16, 0>(table, items, a, route, out, cols, inner, sorted, topk);
}

// NVFP4, Qwen's convention (weight_scale_2), the factor on the output.
extern "C" __global__ __launch_bounds__(OM_THREADS) void routed_format_gate_up_nvfp4_qwen_sm121(
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    const int *__restrict__ sorted, uint32_t topk, uint32_t matrix) {
    OPT_PDL_ENTRY();
    rf_gate_up<SPM_W_FP4, uint8_t, 2>(table, items, a, out, cols, inner, sorted, topk, matrix);
}

extern "C" __global__ __launch_bounds__(OM_THREADS) void routed_format_down_nvfp4_qwen_sm121(
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    const float *__restrict__ a, const float *__restrict__ route, float *__restrict__ out,
    uint32_t cols, uint32_t inner, const int *__restrict__ sorted, uint32_t topk) {
    OPT_PDL_ENTRY();
    rf_down<SPM_W_FP4, uint8_t, 2>(table, items, a, route, out, cols, inner, sorted, topk);
}

// NVFP4, Inkling's convention (amax / outer_scale_denominator), the factor on the output.
extern "C" __global__ __launch_bounds__(OM_THREADS) void routed_format_gate_up_nvfp4_inkling_sm121(
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    const int *__restrict__ sorted, uint32_t topk, uint32_t matrix) {
    OPT_PDL_ENTRY();
    rf_gate_up<SPM_W_FP4, uint8_t, 2>(table, items, a, out, cols, inner, sorted, topk, matrix);
}

extern "C" __global__ __launch_bounds__(OM_THREADS) void routed_format_down_nvfp4_inkling_sm121(
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    const float *__restrict__ a, const float *__restrict__ route, float *__restrict__ out,
    uint32_t cols, uint32_t inner, const int *__restrict__ sorted, uint32_t topk) {
    OPT_PDL_ENTRY();
    rf_down<SPM_W_FP4, uint8_t, 2>(table, items, a, route, out, cols, inner, sorted, topk);
}

// Row-scaled MXFP8: one ue8m0 a row every 32 columns.
extern "C" __global__ __launch_bounds__(OM_THREADS) void routed_format_gate_up_mxfp8_row_sm121(
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    const int *__restrict__ sorted, uint32_t topk, uint32_t matrix) {
    OPT_PDL_ENTRY();
    rf_gate_up<SPM_W_MX8R, uint8_t, 0>(table, items, a, out, cols, inner, sorted, topk, matrix);
}

extern "C" __global__ __launch_bounds__(OM_THREADS) void routed_format_down_mxfp8_row_sm121(
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    const float *__restrict__ a, const float *__restrict__ route, float *__restrict__ out,
    uint32_t cols, uint32_t inner, const int *__restrict__ sorted, uint32_t topk) {
    OPT_PDL_ENTRY();
    rf_down<SPM_W_MX8R, uint8_t, 0>(table, items, a, route, out, cols, inner, sorted, topk);
}

// A split chunk's FFN rows reach the routed experts as the gathered BF16 rows (the f32 seam rows
// are each rank's own); widening them is exact (the low 16 mantissa bits are zeros), so the
// experts read the same values the BF16 rows hold. Grid-stride over `count` values.
extern "C" __global__ void routed_format_widen_sm121(const __nv_bfloat16 *__restrict__ src,
                                                     float *__restrict__ dst, uint32_t count) {
    OPT_PDL_ENTRY();
    const uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += stride) {
        dst[index] = __bfloat162float(src[index]);
    }
}

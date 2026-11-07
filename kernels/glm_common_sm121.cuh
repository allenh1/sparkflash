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

// kernels/glm_common_sm121.cuh
//
// The GLM-5.3-Flash images' names for the one shared bulk-prefill tile body. The body, the
// geometry macros (GLM_BM, GLM_BN, ...), glm_raster, the dynamic activation quantiser and the
// q8 stage body now live once in kernels/shared_prefill_mma_sm121.cuh, which Qwen's prefill
// includes too. This header keeps the GLM images' existing spellings and the meaning they had:
// the dense projection and the routed experts reach the same tile.

#pragma once

#include "shared_prefill_mma_sm121.cuh"

// The GLM images instantiate the shared body through the name they have always used: the
// template's first three parameters are the FP8 weight, routed-expert and low-plane switches,
// and the scale type defaults to GLM's f32 128 x 128 tile scale.
template <bool FP8, bool MOE, bool LOW, typename ScaleT = float>
__device__ __forceinline__ void glm_mma_body(
    const float *__restrict__ a, const uint8_t *__restrict__ w, const void *__restrict__ ws,
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    float *__restrict__ out, uint32_t cols, uint32_t inner, uint32_t tokens,
    uint64_t a_tok_stride, uint64_t out_tok_stride, uint64_t a_grp_stride,
    uint64_t out_grp_stride, uint64_t w_grp_stride = 0, uint64_t ws_grp_stride = 0,
    const int *__restrict__ sorted = nullptr, uint32_t topk = 1,
    const float *__restrict__ route = nullptr) {
    shared_prefill_mma_body<FP8 ? SPM_W_FP8 : SPM_W_BF16, MOE, LOW, ScaleT>(
        a, w, ws, table, items, out, cols, inner, tokens, a_tok_stride, out_tok_stride,
        a_grp_stride, out_grp_stride, w_grp_stride, ws_grp_stride, sorted, topk, route);
}

// The q8 body (dynamic E4M3 activation, one block scale a 128 x 128 weight tile) is the same
// 128 x 128 tile over a cp.async pipeline; the shared header states it once.
template <bool MOE, typename ScaleT = float>
__device__ __forceinline__ void glm_q8_body(
    const uint8_t *__restrict__ a, const float *__restrict__ as, const uint8_t *__restrict__ w,
    const void *__restrict__ ws, const unsigned long long *__restrict__ table,
    const int4 *__restrict__ items, float *__restrict__ out, uint32_t cols, uint32_t inner,
    uint32_t tokens, const int *__restrict__ sorted, uint32_t topk,
    const float *__restrict__ route) {
    shared_prefill_q8_body<MOE, ScaleT>(a, as, w, ws, table, items, out, cols, inner, tokens,
                                        sorted, topk, route);
}

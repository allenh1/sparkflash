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

// kernels/nvfp4_prefill_sm121.cu
//
// The wrappers that give NVFP4 its two operations on the one shared bulk-prefill tile
// (shared_prefill_mma_sm121.cuh's SPM_W_FP4 mode). The body, the 128 x 128 tile, the staged
// weight words and the group-scale decode live in the header; this file only names the two
// tensor-wide conventions and the two row forms:
//
//   * Qwen folds weight_scale_2 into the group scale at the weight load, which is where its
//     separate q4e/q4p bodies apply it.
//   * Inkling applies amax / outer_scale_denominator to the output after the accumulation.
//
// The prompt entries read a 128-row x 128-column tile; the rows entries are the few-row end of
// the same tile (shared_prefill_mma_rows_body), so a decode step and a short chunk have the
// prompt body's arithmetic. Neither entry owns a body of its own.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "opt_mma_common_sm121.cuh"
#include "shared_prefill_mma_sm121.cuh"

// The matrix shapes are runtime launch words; the instance's hidden/intermediate values only pin
// the build the device arm loads. They are used in the static assertion so a row that claims a
// shape cannot be built without it.
extern "C" __global__ __launch_bounds__(OM_THREADS) void nvfp4_prefill_qwen_gate_up_sm121(
    const uint8_t *__restrict__ w, const uint8_t *__restrict__ ws, float wide,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    uint32_t tokens) {
    shared_prefill_mma_body<SPM_W_FP4, false, true, float, 2, 0, false, false, 1>(
        a, w, ws, nullptr, nullptr, out, cols, inner, tokens, inner, cols, 0, 0, 0, 0, nullptr,
        1u, nullptr, 0u, wide);
}

extern "C" __global__ __launch_bounds__(OM_THREADS) void nvfp4_prefill_qwen_down_sm121(
    const uint8_t *__restrict__ w, const uint8_t *__restrict__ ws, float wide,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    uint32_t tokens) {
    shared_prefill_mma_body<SPM_W_FP4, false, true, float, 2, 0, false, false, 1>(
        a, w, ws, nullptr, nullptr, out, cols, inner, tokens, inner, cols, 0, 0, 0, 0, nullptr,
        1u, nullptr, 0u, wide);
}

extern "C" __global__ __launch_bounds__(OM_THREADS) void nvfp4_prefill_inkling_gate_up_sm121(
    const uint8_t *__restrict__ w, const uint8_t *__restrict__ ws, float wide,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    uint32_t tokens) {
    shared_prefill_mma_body<SPM_W_FP4, false, true, float, 2, 0, false, false, 2>(
        a, w, ws, nullptr, nullptr, out, cols, inner, tokens, inner, cols, 0, 0, 0, 0, nullptr,
        1u, nullptr, 0u, wide);
}

extern "C" __global__ __launch_bounds__(OM_THREADS) void nvfp4_prefill_inkling_down_sm121(
    const uint8_t *__restrict__ w, const uint8_t *__restrict__ ws, float wide,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    uint32_t tokens) {
    shared_prefill_mma_body<SPM_W_FP4, false, true, float, 2, 0, false, false, 2>(
        a, w, ws, nullptr, nullptr, out, cols, inner, tokens, inner, cols, 0, 0, 0, 0, nullptr,
        1u, nullptr, 0u, wide);
}

extern "C" __global__ __launch_bounds__(128) void nvfp4_rows_qwen_gate_up_sm121(
    const uint8_t *__restrict__ w, const uint8_t *__restrict__ ws, float wide,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    uint32_t tokens) {
    SpmRowsFp4 stage{w, ws, inner, wide, true};
    shared_prefill_mma_rows_body<false>(stage, a, out, cols, inner, tokens);
}

extern "C" __global__ __launch_bounds__(128) void nvfp4_rows_qwen_down_sm121(
    const uint8_t *__restrict__ w, const uint8_t *__restrict__ ws, float wide,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    uint32_t tokens) {
    SpmRowsFp4 stage{w, ws, inner, wide, true};
    shared_prefill_mma_rows_body<false>(stage, a, out, cols, inner, tokens);
}

extern "C" __global__ __launch_bounds__(128) void nvfp4_rows_inkling_gate_up_sm121(
    const uint8_t *__restrict__ w, const uint8_t *__restrict__ ws, float wide,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    uint32_t tokens) {
    SpmRowsFp4 stage{w, ws, inner, wide, false};
    shared_prefill_mma_rows_body<false>(stage, a, out, cols, inner, tokens, wide);
}

extern "C" __global__ __launch_bounds__(128) void nvfp4_rows_inkling_down_sm121(
    const uint8_t *__restrict__ w, const uint8_t *__restrict__ ws, float wide,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    uint32_t tokens) {
    SpmRowsFp4 stage{w, ws, inner, wide, false};
    shared_prefill_mma_rows_body<false>(stage, a, out, cols, inner, tokens, wide);
}

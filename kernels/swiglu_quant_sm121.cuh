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

// kernels/swiglu_quant_sm121.cuh
//
// The SwiGLU + MX-quantise epilogue the unweighted and weighted entries had two copies of. The
// two differ in exactly one expression: the weighted form multiplies the activation by the
// slot's routing weight and rounds the product to bf16 before the quantiser, the unweighted form
// rounds the activation itself. Everything else - the asymmetric clamp, the 32-value block, the
// scale rule, the trace scope and the payload count - is one statement here.
//
// The activation clamp is the engine's, not the textbook's: gate = min(gate, limit) on the upper
// side only and up = clamp(up, -limit, limit) on both. silu is not odd, so clamping the gate on
// both sides changes the operand for a negative gate past the limit; src/swiglu_ref.rs pins it.

#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "m4_common_sm121.cuh"
#include "m4_activation_quant_sm121.cuh"

// One MX activation block's scale and one value's code, for a warp that owns the whole 32-value
// block: the amax is a butterfly reduction across the warp, lane 0 writes the scale, and every
// lane encodes its own value against the reference's activation scale.
__device__ __forceinline__ uint8_t spark_activation_quantize_block(
    float value,
    uint32_t lane,
    uint8_t *__restrict__ scale_out) {
    float amax = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, offset));
    }
    // THE ACTIVATION RULE: the reference floors amax at 1e-4 and rounds the scale exponent up.
    const uint8_t scale = spark_act_quant_scale_code(amax);
    if (lane == 0u) {
        *scale_out = scale;
    }
    const float inv_scale = 1.0f / spark_ue8m0_decode(scale);
    return spark_fp8_e4m3_encode(value * inv_scale);
}

// The epilogue both entries run. WEIGHTED multiplies by one routing weight a row and rounds the
// product; otherwise there is no product and the activation itself is the value cast.
template <bool WEIGHTED>
__device__ __forceinline__ void swiglu_quant_body(
    const __nv_bfloat16 *__restrict__ gate, const __nv_bfloat16 *__restrict__ up,
    const float *__restrict__ weights, uint8_t *__restrict__ out_fp8,
    uint8_t *__restrict__ out_scales, const uint32_t rows, const uint32_t cols, const float limit,
    GpuTraceRing *__restrict__ trace_ring, const uint32_t layer_id, const uint64_t step_seq) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint64_t warp = (uint64_t)blockIdx.x * warps_per_block + (uint64_t)(threadIdx.x >> 5u);
    const uint32_t blocks_per_row = cols >> 5u;  // cols is a multiple of 32 by contract
    const uint64_t total_blocks = (uint64_t)rows * (uint64_t)blocks_per_row;

    // The scope is a property of the launch shape, not of this file: cols = 640 is the
    // routed-expert epilogue, any other width the shared-expert path whose outputs feed the dense
    // W8A8 GEMMs. Both macros take this one expression.
    const uint64_t trace_scope = cols == SPARK_M4_MOE_WIDTH_PADDED
                                     ? TRACE_SCOPE_MOE_MXFP4
                                     : TRACE_SCOPE_DENSE_MXFP8;
    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, trace_scope, t_enter);

    // Guard instead of an early return, so every thread of the block reaches the trace barrier.
    if (warp < total_blocks) {
        const uint32_t row = (uint32_t)(warp / blocks_per_row);
        const uint32_t block = (uint32_t)(warp % blocks_per_row);
        const uint64_t index =
            (uint64_t)row * (uint64_t)cols + (uint64_t)block * 32u + (uint64_t)lane;

        const float g = __bfloat162float(gate[index]);
        const float u = __bfloat162float(up[index]);
        const float clamped_gate = fminf(g, limit);               // upper side only
        const float clamped_up = fminf(fmaxf(u, -limit), limit);  // both sides
        const float activated = (clamped_gate / (1.0f + __expf(-clamped_gate))) * clamped_up;
        // Native rounds the activation (or its weighted product) to BF16 before the next act_quant.
        // The weighted routed epilogue rounds the product for the same reason the shared expert
        // rounds the activation: quantising the unrounded f32 keeps bits the model never sees.
        const float value = WEIGHTED ? activated * weights[row] : activated;
        const float rounded = __bfloat162float(__float2bfloat16_rn(value));

        out_fp8[index] = spark_activation_quantize_block(
            rounded,
            lane,
            &out_scales[(uint64_t)row * (uint64_t)blocks_per_row + (uint64_t)block]);
    }
    // The exit timestamp has to cover the whole block, so the barrier is taken after its work,
    // and only when the trace is compiled in.
#if defined(SPARKFLASH_ENABLE_GPU_TRACE)
    __syncthreads();
#endif
    // payload_meta: one E4M3 activation byte per value the reporting block quantised, which is
    // active warps times 32 (a short grid's last block owns fewer than warps_per_block blocks).
    const uint64_t first_warp = (uint64_t)blockIdx.x * warps_per_block;
    const uint64_t remaining = total_blocks > first_warp ? total_blocks - first_warp : 0u;
    const uint32_t active_warps = remaining < warps_per_block ? (uint32_t)remaining : warps_per_block;
    const uint32_t meta = active_warps * 32u;
    if (lane == 0u) {
        SPARK_GPU_TRACE_EXIT(trace_ring, trace_scope, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
    (void)meta;
    (void)trace_scope;
}

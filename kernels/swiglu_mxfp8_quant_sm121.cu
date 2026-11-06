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

// The unweighted SwiGLU + MX-quantise epilogue and the standalone activation quantiser the
// shared expert needs beside it. The body is kernels/swiglu_quant_sm121.cuh's; this file only
// names the two entry points and the shapes each serves.
//
// swiglu_mxfp8_quant_sm121 is launched for two shapes: the routed-expert epilogue (cols = 640,
// SPARK_M4_MOE_WIDTH_PADDED, under TRACE_SCOPE_MOE_MXFP4 because its operands are what
// b12x_mxfp4_moe_sm121 consumes) and the shared-expert path (cols = 2304, under
// TRACE_SCOPE_DENSE_MXFP8 because its operands feed the dense W8A8 GEMMs). The scope follows the
// shape, never the file. mxfp8_quantize_activations_sm121 is always TRACE_SCOPE_DENSE_MXFP8.
//
// The activation is the reference's asymmetric SwiGLU clamp and is rounded to BF16 before the
// quantiser, exactly as the weighted variant rounds its product. The quantise rule is the
// activation rule (amax over the 32-value block floored at 1e-4, scale rounded up), not the
// stored-weight grid rule.

#include "swiglu_quant_sm121.cuh"

extern "C" __global__ void swiglu_mxfp8_quant_sm121(
    const __nv_bfloat16 *__restrict__ gate,   // [rows][cols] bf16, w1 x
    const __nv_bfloat16 *__restrict__ up,     // [rows][cols] bf16, w3 x
    uint8_t *__restrict__ out_fp8,            // [rows][cols] E4M3, row stride cols
    uint8_t *__restrict__ out_scales,         // [rows][cols / 32] ue8m0, row stride cols / 32
    const uint32_t rows,
    const uint32_t cols,
    const float limit,                        // swiglu_limit, 10.0 in the released config
    GpuTraceRing *__restrict__ trace_ring,    // scope follows cols
    const uint32_t layer_id,
    const uint64_t step_seq) {
    swiglu_quant_body<false>(gate, up, nullptr, out_fp8, out_scales, rows, cols, limit,
                             trace_ring, layer_id, step_seq);
}

extern "C" __global__ void mxfp8_quantize_activations_sm121(
    const __nv_bfloat16 *__restrict__ values,  // [rows][cols] bf16 activation
    uint8_t *__restrict__ out_fp8,             // [rows][cols] E4M3, row stride cols
    uint8_t *__restrict__ out_scales,          // [rows][cols / 32] ue8m0, row stride cols / 32
    const uint32_t rows,
    const uint32_t cols,
    GpuTraceRing *__restrict__ trace_ring,    // TRACE_SCOPE_DENSE_MXFP8
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint64_t warp = (uint64_t)blockIdx.x * warps_per_block + (uint64_t)(threadIdx.x >> 5u);
    const uint32_t blocks_per_row = cols >> 5u;
    const uint64_t total_blocks = (uint64_t)rows * (uint64_t)blocks_per_row;

    uint64_t t_enter = 0u;
    // The shared expert's input quantiser feeds the W8A8 gate-up GEMMs, which are the dense MXFP8
    // path's arithmetic, so it records under DENSE_MXFP8 rather than the FP4 expert scope.
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    if (warp < total_blocks) {
        const uint32_t row = (uint32_t)(warp / blocks_per_row);
        const uint32_t block = (uint32_t)(warp % blocks_per_row);
        const uint64_t index =
            (uint64_t)row * (uint64_t)cols + (uint64_t)block * 32u + (uint64_t)lane;

        out_fp8[index] = spark_activation_quantize_block(
            __bfloat162float(values[index]),
            lane,
            &out_scales[(uint64_t)row * (uint64_t)blocks_per_row + (uint64_t)block]);
    }
#if defined(SPARKFLASH_ENABLE_GPU_TRACE)
    __syncthreads();
#endif
    const uint64_t first_warp = (uint64_t)blockIdx.x * warps_per_block;
    const uint64_t remaining = total_blocks > first_warp ? total_blocks - first_warp : 0u;
    const uint32_t active_warps = remaining < warps_per_block ? (uint32_t)remaining : warps_per_block;
    const uint32_t meta = active_warps * 32u;
    if (lane == 0u) {
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
    (void)meta;
}

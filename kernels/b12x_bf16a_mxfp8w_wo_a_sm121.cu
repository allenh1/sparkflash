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

// The BF16-activation variant of the grouped output projection wo_a: the grouped twin of
// b12x_bf16a_mxfp8w_dense_sm121.cu and the D37 precision-first path for K6. Weights are exactly
// the B12x FP8 codes and ue8m0 scales that b12x_mxfp8_wo_a_sm121.cu reads; the activation operand
// is __nv_bfloat16 instead of E4M3 codes plus per-32 ue8m0 scales.
//
//     lo_out[group][col] = __float2bfloat16_rn(
//         sum over k ascending in 32-value blocks, ascending inside a block,
//             of __bfloat162float(attn_bf16[group][k]) * (fp8(w[col][k]) * ue8m0(w_scale)))
//
// The group is a grid dimension, not a GEMM shape, and the two group strides differ in size. The
// body is kernels/b12x_dense_sm121.cuh's, shared with the MXFP8 wo_a kernel.

#include "b12x_dense_sm121.cuh"

// The emitted symbol's geometry, the same tile the MXFP8 wo_a kernel uses.
#define SPARK_M4_BF16A_WOA_TILE 256u
#define SPARK_M4_BF16A_WOA_THREADS 256u
#define SPARK_M4_BF16A_WOA_GROUP_INNER (SPARK_M4_HEAD_DIM * (SPARK_M4_HEADS / SPARK_M4_O_GROUPS))

extern "C" __global__ void b12x_bf16a_mxfp8w_wo_a_sm121(
    const __nv_bfloat16 *__restrict__ attn_bf16,  // [groups][group_inner] BF16 attention outputs
    const uint8_t *__restrict__ w_fp8,            // B12x tiled [groups][o_lora_rank][group_inner]
    const uint8_t *__restrict__ w_scales,         // B12x tiled ue8m0 scales
    __nv_bfloat16 *__restrict__ lo_out,           // [groups][o_lora_rank] bf16 result
    const uint32_t groups,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    b12x_woa_scalar_body<true, SPARK_M4_BF16A_WOA_TILE, SPARK_M4_BF16A_WOA_THREADS,
                          SPARK_M4_BF16A_WOA_GROUP_INNER, SPARK_M4_O_LORA_RANK>(
        nullptr, nullptr, attn_bf16, w_fp8, w_scales, lo_out, groups, trace_ring, layer_id,
        step_seq);
}

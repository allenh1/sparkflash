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

// The BF16-activation variant of the dense MXFP8 GEMM: the same B12x weights, layout, tile and
// trace ABI as b12x_mxfp8_dense_sm121.cu, with the ACTIVATION operand in __nv_bfloat16 instead of
// E4M3 + ue8m0 block scales. This is the D37 precision-first path: the activation quantiser is
// removed from the GEMM's input while the weights stay the checkpoint's own FP8 codes.
//
//     out[row][col] = __float2bfloat16_rn(
//         sum over k ascending in 32-value blocks, ascending inside a block,
//             of __bfloat162float(a_bf16[row][k]) * (fp8(w[col][k]) * ue8m0(w_scale)))
//
// The bf16 activation widens to f32 exactly, so there is no activation rounding inside this
// kernel; each product is one f32 multiply and the single rounding is the final bf16. The body is
// kernels/b12x_dense_sm121.cuh's, shared with the MXFP8 twin; src/bf16a_mxfp8w_ref.rs pins the
// order.

#include "b12x_dense_sm121.cuh"

// The emitted symbol's geometry, kept as defines for the launch module's source-text pin (the
// body itself reads the shared header's constants).
#define SPARK_M4_BF16A_DENSE_TILE_M 64u
#define SPARK_M4_BF16A_DENSE_TILE_N 64u
#define SPARK_M4_BF16A_DENSE_THREADS 256u
#define SPARK_M4_BF16A_DENSE_SUBTILES_PER_ROW 16u

extern "C" __global__ void b12x_bf16a_mxfp8w_dense_sm121(
    const __nv_bfloat16 *__restrict__ a_bf16,  // [rows][inner] BF16 activations
    const uint8_t *__restrict__ w_fp8,         // B12x tiled E4M3 weights
    const uint8_t *__restrict__ w_scales,      // B12x tiled ue8m0 weight scales
    __nv_bfloat16 *__restrict__ out,           // [rows][cols] bf16 result
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t inner,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    b12x_dense_scalar_body<true>(nullptr, nullptr, a_bf16, w_fp8, w_scales, out, rows, cols, inner,
                                 trace_ring, layer_id, step_seq);
}

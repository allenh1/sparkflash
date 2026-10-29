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

// kernels/b12x_mxfp8_dense_m1_sm121.cu
//
// Attribution: The "b12x" naming prefix and SM120/SM121 block-scaled MXFP8/MXFP4 micro-tile
// layout concepts originate from b12x (https://github.com/local-inference-lab/b12x, now in
// FlashInfer under flashinfer/experimental/b12x, Apache-2.0; Luke Alonso @lukealonso,
// @original-el8, and contributors) and NVIDIA Flare
// (https://github.com/NVIDIA/flare-local-inference, Apache-2.0). See the root NOTICE file.
//
// The columns-only M=1 dense MXFP8 GEMM: the decode-step projection kernel of WS-D.
//
// Why this symbol exists. The pinned b12x_mxfp8_dense_sm121 kernel owns a 64x64 output tile with
// 256 threads, each thread a 4x4 sub-tile, so its row axis is tiled for a prefill batch. At decode
// M=1 only lane group 0 (lanes 0-15) has a row inside the tile: 16 of 256 thread slots do the
// arithmetic, and the projection launches that reach it (wq_a, wkv, wo_b, wq_b) run at 1.92-16.9
// GMAC/s. This kernel keeps the pinned kernel's arithmetic EXACTLY - the per-output pass is the same
// loop, in the same order, with the same decodes and the same single bf16 round - and changes only
// which thread owns which output: one row, 256 output columns per block, one output column per
// thread. K still ascends in 32-value MX blocks and ascends inside each block, so every output's
// running sum and its rounding sequence are unchanged; the tiling decides only who writes it.
//
// This is an ADDITION. b12x_mxfp8_dense_sm121.cu is not edited and its pinned cubin bytes and digest
// are not touched; this file is its own cubin, its own baseline entry and its own symbol. The scalar
// symbol in that file stays the reference oracle and the pin.
//
// Shape contract: rows == 1. The launch wrapper refuses any other row count; the kernel re-checks it
// so a wrong launch cannot read past the single activation row.
//
// Geometry: grid = (ceil(cols / 256), 1, 1) blocks of 256 threads. Blocks along the column axis drop
// the tail columns that fall outside `cols`, exactly as the pinned kernel's per-column guard does.
//
// What is proven here: nvcc compiles it for sm_121a (the build script probes every kernel file, and
// scripts/check-kernel-baseline.sh recompiles it against the committed baseline), its signature and
// geometry are pinned by parsed-declaration tests in src/mxfp8_dense_m1_launch.rs, and the tiling
// model it implements is compared against the pinned arithmetic by src/mxfp8_ref.rs's dense_gemm in
// tests/m1_dense_oracle.rs for every shape it serves. What is NOT proven here: it has never executed
// on a GPU on this branch, and a tensor-core or split-K path would be a different arithmetic, which
// this file deliberately does not contain.

#include "m4_common_sm121.cuh"

// One output column per thread: 256 threads cover 256 columns of the single output row.
#define SPARK_M4_DENSE_M1_COLUMNS 256u
#define SPARK_M4_DENSE_M1_THREADS 256u

static_assert(SPARK_M4_DENSE_M1_COLUMNS == SPARK_M4_DENSE_M1_THREADS,
              "one output column per thread");
static_assert(SPARK_M4_DENSE_M1_THREADS % 32u == 0u, "a block is a whole number of warps");
static_assert(SPARK_M4_DENSE_M1_COLUMNS % SPARK_B12X_NTILE == 0u,
              "a block covers whole B12x micro-tiles");

extern "C" __global__ void b12x_mxfp8_dense_m1_sm121(
    const uint8_t *__restrict__ a_fp8,      // [rows][inner] E4M3 activations, row 0 only
    const uint8_t *__restrict__ a_scales,   // [rows][inner / 32] ue8m0, one per MX block
    const uint8_t *__restrict__ w_fp8,      // B12x tiled E4M3 weights
    const uint8_t *__restrict__ w_scales,   // B12x tiled ue8m0 weight scales
    __nv_bfloat16 *__restrict__ out,        // [rows][cols] bf16 result
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t inner,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t lane = threadIdx.x;
    const uint32_t col = blockIdx.x * SPARK_M4_DENSE_M1_COLUMNS + lane;
    const uint32_t k_blocks = inner / SPARK_M4_QUANT_BLOCK;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    // The accumulation order per output is the pinned kernel's contract: K ascending in 32-value
    // blocks and ascending within a block, one rounding to bf16 at the end. This kernel only changes
    // which thread owns the output, so the reference does not have to model the tiling.
    if (rows == 1u && col < cols) {
        const uint8_t *a_row = a_fp8;
        const uint8_t *a_scale_row = a_scales;
        float acc = 0.0f;
        for (uint32_t block = 0; block < k_blocks; ++block) {
            const uint32_t base = block * SPARK_M4_QUANT_BLOCK;
            const float a_scale = spark_ue8m0_decode(a_scale_row[block]);
            const float w_scale =
                spark_ue8m0_decode(w_scales[spark_b12x_scale_index(col, base, inner)]);
#pragma unroll 8
            for (uint32_t i = 0; i < SPARK_M4_QUANT_BLOCK; ++i) {
                const float a_value = spark_fp8_e4m3_decode(a_row[base + i]) * a_scale;
                const float w_value =
                    spark_fp8_e4m3_decode(
                        w_fp8[spark_b12x_weight_index(col, base + i, inner)]) *
                    w_scale;
                acc += a_value * w_value;
            }
        }
        out[col] = __float2bfloat16_rn(acc);
    }
    __syncthreads();

    if (lane == 0u) {
        // payload_meta: one bf16 per output element of this block's tile (M=1: one row x 256 cols).
        const uint32_t meta = SPARK_M4_DENSE_M1_COLUMNS * 2u;
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}

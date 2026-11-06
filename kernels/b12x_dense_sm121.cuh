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

// kernels/b12x_dense_sm121.cuh
//
// The scalar dense MXFP8 GEMM body the two b12x dense entries had a copy of: one 64x64 output
// tile a block, 256 threads owning 4x4 sub-tiles, K ascending in 32-value blocks. BF16A widens a
// bf16 activation exactly instead of decoding E4M3 codes times a ue8m0 scale; the accumulation
// order and the output rounding are then the same statement. The K1 warp path and the split-K
// forms in b12x_mxfp8_dense_sm121.cu stay there: they are different tilings, not this body.

#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "m4_common_sm121.cuh"
#include "b12x_common_sm121.cuh"

#define SPARK_B12X_DENSE_TILE_M 64u
#define SPARK_B12X_DENSE_TILE_N 64u
#define SPARK_B12X_DENSE_THREADS 256u
#define SPARK_B12X_DENSE_ROWS_PER_THREAD 4u
#define SPARK_B12X_DENSE_COLS_PER_THREAD 4u
#define SPARK_B12X_DENSE_SUBTILES_PER_ROW \
    (SPARK_B12X_DENSE_TILE_N / SPARK_B12X_DENSE_COLS_PER_THREAD)

static_assert(SPARK_B12X_DENSE_TILE_M * SPARK_B12X_DENSE_TILE_N ==
                  SPARK_B12X_DENSE_THREADS * SPARK_B12X_DENSE_ROWS_PER_THREAD *
                      SPARK_B12X_DENSE_COLS_PER_THREAD,
              "the 64x64 tile is exactly covered by 256 threads of 4x4");
static_assert(SPARK_B12X_DENSE_TILE_N % SPARK_M4_QUANT_BLOCK == 0u,
              "the N tile is whole MX blocks");

// out[row][col] = rn_bf16(sum over k ascending in 32-value blocks of a(row,k) * w(col,k) * scale).
// A is either E4M3 codes with one ue8m0 scale a block or BF16, whose widening is exact.
template <bool BF16A>
__device__ __forceinline__ void b12x_dense_scalar_body(
    const uint8_t *__restrict__ a_fp8, const uint8_t *__restrict__ a_scales,
    const __nv_bfloat16 *__restrict__ a_bf16, const uint8_t *__restrict__ w_fp8,
    const uint8_t *__restrict__ w_scales, __nv_bfloat16 *__restrict__ out, const uint32_t rows,
    const uint32_t cols, const uint32_t inner, GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id, const uint64_t step_seq) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row_base = blockIdx.x * SPARK_B12X_DENSE_TILE_M +
                              (lane / SPARK_B12X_DENSE_SUBTILES_PER_ROW) *
                                  SPARK_B12X_DENSE_ROWS_PER_THREAD;
    const uint32_t col_base =
        blockIdx.y * SPARK_B12X_DENSE_TILE_N +
        (lane % SPARK_B12X_DENSE_SUBTILES_PER_ROW) * SPARK_B12X_DENSE_COLS_PER_THREAD;
    const uint32_t k_blocks = inner / SPARK_M4_QUANT_BLOCK;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    // The accumulation order per output is the contract: K ascending in 32-value blocks and
    // ascending within a block. The 4x4 sub-tile only decides which outputs a thread owns; it
    // never reorders a sum, so the reference does not have to model the tiling.
    for (uint32_t di = 0; di < SPARK_B12X_DENSE_ROWS_PER_THREAD; ++di) {
        const uint32_t row = row_base + di;
        if (row >= rows) {
            continue;
        }
        const uint8_t *a_row = a_fp8 + (uint64_t)row * inner;
        const uint8_t *a_scale_row = a_scales + (uint64_t)row * k_blocks;
        const __nv_bfloat16 *a_bf16_row = a_bf16 + (uint64_t)row * inner;
        float acc[SPARK_B12X_DENSE_COLS_PER_THREAD];
#pragma unroll
        for (uint32_t dj = 0; dj < SPARK_B12X_DENSE_COLS_PER_THREAD; ++dj) {
            acc[dj] = 0.0f;
        }
        for (uint32_t block = 0; block < k_blocks; ++block) {
            const uint32_t base = block * SPARK_M4_QUANT_BLOCK;
            const float a_scale = BF16A ? 1.0f : spark_ue8m0_decode(a_scale_row[block]);
            float w_scale[SPARK_B12X_DENSE_COLS_PER_THREAD];
#pragma unroll
            for (uint32_t dj = 0; dj < SPARK_B12X_DENSE_COLS_PER_THREAD; ++dj) {
                const uint32_t col = col_base + dj;
                w_scale[dj] = col < cols
                                  ? spark_ue8m0_decode(
                                        w_scales[spark_b12x_scale_index(col, base, inner)])
                                  : 0.0f;
            }
#pragma unroll 8
            for (uint32_t i = 0; i < SPARK_M4_QUANT_BLOCK; ++i) {
                // A bf16 activation widens exactly and enters the sum unrounded; an E4M3 code is
                // one multiply by its block scale. Both are shared by the row's four columns.
                const float a_value = BF16A ? __bfloat162float(a_bf16_row[base + i])
                                             : spark_fp8_e4m3_decode(a_row[base + i]) * a_scale;
#pragma unroll
                for (uint32_t dj = 0; dj < SPARK_B12X_DENSE_COLS_PER_THREAD; ++dj) {
                    const uint32_t col = col_base + dj;
                    if (col < cols) {
                        const float w_value =
                            spark_fp8_e4m3_decode(
                                w_fp8[spark_b12x_weight_index(col, base + i, inner)]) *
                            w_scale[dj];
                        acc[dj] += a_value * w_value;
                    }
                }
            }
        }
#pragma unroll
        for (uint32_t dj = 0; dj < SPARK_B12X_DENSE_COLS_PER_THREAD; ++dj) {
            const uint32_t col = col_base + dj;
            if (col < cols) {
                out[(uint64_t)row * cols + col] = __float2bfloat16_rn(acc[dj]);
            }
        }
    }
    __syncthreads();

    if (lane == 0u) {
        // payload_meta: one bf16 per output element of this block's tile.
        const uint32_t meta = SPARK_B12X_DENSE_TILE_M * SPARK_B12X_DENSE_TILE_N * 2u;
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}

// The grouped output projection wo_a: one block a group, one thread an output channel, the group's
// whole reduction width staged in shared memory. BF16A widens a bf16 attention row exactly; the
// other form stages E4M3 codes and their ue8m0 block scales.
template <bool BF16A, int TILE, int THREADS, int GROUP_INNER, int LORA_RANK>
__device__ __forceinline__ void b12x_woa_scalar_body(
    const uint8_t *__restrict__ attn_fp8, const uint8_t *__restrict__ attn_scales,
    const __nv_bfloat16 *__restrict__ attn_bf16, const uint8_t *__restrict__ w_fp8,
    const uint8_t *__restrict__ w_scales, __nv_bfloat16 *__restrict__ lo_out,
    const uint32_t groups, GpuTraceRing *__restrict__ trace_ring, const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t group = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t column = blockIdx.y * TILE + lane;
    const uint32_t k_blocks = GROUP_INNER / SPARK_M4_QUANT_BLOCK;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    __shared__ __align__(16) uint8_t s_a_row[BF16A ? GROUP_INNER * 2u : GROUP_INNER];
    __shared__ uint8_t s_a_scales[BF16A ? 1u : (GROUP_INNER / SPARK_M4_QUANT_BLOCK)];
    if (group < groups) {
        if constexpr (BF16A) {
            const __nv_bfloat16 *a_row =
                attn_bf16 + (uint64_t)group * GROUP_INNER;
            __nv_bfloat16 *staged = reinterpret_cast<__nv_bfloat16 *>(s_a_row);
            for (uint32_t k = lane; k < (uint32_t)GROUP_INNER; k += THREADS) {
                staged[k] = a_row[k];
            }
        } else {
            const uint8_t *a_row = attn_fp8 + (uint64_t)group * GROUP_INNER;
            const uint8_t *a_scale_row = attn_scales + (uint64_t)group * k_blocks;
            for (uint32_t k = lane; k < (uint32_t)GROUP_INNER; k += THREADS) {
                s_a_row[k] = a_row[k];
            }
            if (lane < k_blocks) {
                s_a_scales[lane] = a_scale_row[lane];
            }
        }
    }
    __syncthreads();

    if (group < groups && column < (uint32_t)LORA_RANK) {
        // Each group owns a weight buffer AND a scale buffer, and the two are different sizes, so
        // there are two group strides. The offsets inside each buffer come from the same helpers
        // the dense kernel uses.
        const uint64_t group_weight_base =
            (uint64_t)group * spark_b12x_buffer_bytes(LORA_RANK, GROUP_INNER);
        const uint64_t group_scale_base =
            (uint64_t)group * spark_b12x_scale_bytes(LORA_RANK, GROUP_INNER);
        float acc = 0.0f;
        for (uint32_t block = 0; block < k_blocks; ++block) {
            const uint32_t base = block * SPARK_M4_QUANT_BLOCK;
            const float a_scale =
                BF16A ? 1.0f : spark_ue8m0_decode(s_a_scales[block]);
            const float w_scale = spark_ue8m0_decode(
                w_scales[group_scale_base +
                         spark_b12x_scale_index(column, base, GROUP_INNER)]);
#pragma unroll 8
            for (uint32_t i = 0; i < SPARK_M4_QUANT_BLOCK; ++i) {
                const float a_value =
                    BF16A ? __bfloat162float(
                                reinterpret_cast<const __nv_bfloat16 *>(s_a_row)[base + i])
                          : spark_fp8_e4m3_decode(s_a_row[base + i]) * a_scale;
                const float w_value =
                    spark_fp8_e4m3_decode(
                        w_fp8[group_weight_base +
                              spark_b12x_weight_index(column, base + i, GROUP_INNER)]) *
                    w_scale;
                acc += a_value * w_value;
            }
        }
        lo_out[(uint64_t)group * LORA_RANK + column] = __float2bfloat16_rn(acc);
    }
    __syncthreads();

    if (lane == 0u) {
        const uint32_t meta = TILE * 2u;
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}

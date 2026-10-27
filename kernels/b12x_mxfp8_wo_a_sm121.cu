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

// kernels/b12x_mxfp8_wo_a_sm121.cu
//
// The grouped output projection wo_a of specification 2.1/6.2, MXFP8 like the dense twin:
// 8 output groups (2 per rank) with o_lora_rank 1024, each group consuming the concatenated
// 512-dim outputs of its 8 attention heads (8 * 512 = 4096 reduction width), weights in the
// B12x layout with one ue8m0 scale per 32 values and an f32 accumulator.
//
// Why this is a separate kernel from b12x_mxfp8_dense_sm121.cu rather than a call into it: the
// group is a grid dimension, not a GEMM shape, and each group has its own weight matrix width. The
// arithmetic is deliberately identical, so mxfp8_ref.rs's accumulation contract covers both, and
// mxfp8_ref.rs's wo_a test asserts the grouped result equals a per-group dense GEMM over the same
// slice. The CUDA index helpers themselves are compared against their Rust twins value for value
// by tests/b12x_index_parity.rs, which compiles the helper block out of the shared header.
//
// Each group has TWO buffers with different sizes: weights of [o_lora_rank][group_inner] and scales
// of one ue8m0 byte per 32x32 block, ceil(o_lora_rank/32) * (group_inner/32). The two group
// strides are therefore different numbers and this kernel must not share one between them; see the
// group_weight_base and group_scale_base below.
//
// Verification status: never executed. nvcc 12.9.1 compiles it for sm_121a, clang parses it
// against the stub headers, src/m4_abi.rs pins the signature, the group geometry and the shared
// index helpers, and src/mxfp8_ref.rs defines and tests the numerics on the CPU. No tensor-core
// path is claimed.

#include "m4_common_sm121.cuh"

#define SPARK_M4_WOA_THREADS 256u
#define SPARK_M4_WOA_TILE 256u
// Heads per group: 64 heads over 8 groups.
#define SPARK_M4_WOA_HEADS_PER_GROUP (SPARK_M4_HEADS / SPARK_M4_O_GROUPS)
// The reduction width of one group: its heads' 512-dim outputs concatenated.
#define SPARK_M4_WOA_GROUP_INNER (SPARK_M4_WOA_HEADS_PER_GROUP * SPARK_M4_HEAD_DIM)

static_assert(SPARK_M4_WOA_HEADS_PER_GROUP == 8u, "64 heads over 8 groups is 8 heads per group");
static_assert(SPARK_M4_WOA_GROUP_INNER == 4096u, "the group reduction width is 4096");
static_assert(SPARK_M4_WOA_GROUP_INNER % SPARK_M4_QUANT_BLOCK == 0u, "the width tiles by 32");
static_assert(SPARK_M4_O_LORA_RANK % SPARK_M4_WOA_TILE == 0u, "1024 output channels tile by 256");
static_assert(SPARK_M4_WOA_THREADS == SPARK_M4_WOA_TILE, "one output element per thread");
static_assert(SPARK_M4_O_GROUPS == 8u && SPARK_M4_O_GROUPS_PER_RANK == 2u, "8 groups, 2 per rank");
static_assert(SPARK_M4_O_LORA_RANK % SPARK_B12X_NTILE == 0u,
              "the output rank is whole B12x micro-tile rows, so nothing is padded away");
static_assert(SPARK_M4_WOA_GROUP_INNER % SPARK_B12X_KTILE == 0u,
              "the group reduction width is whole B12x micro-tiles");
// The scale grid of one group is ceil(o_lora_rank / 32) * (group_inner / 32) bytes, which is 32
// times smaller than the weight block: a group stride shared between the two would overrun the
// scale buffer by 31 group strides, which is the defect this pin exists to prevent returning.
static_assert(SPARK_M4_O_LORA_RANK / SPARK_B12X_NTILE == 32u,
              "1024 output channels are 32 micro-tile rows of scales");
static_assert(SPARK_M4_WOA_GROUP_INNER / SPARK_B12X_KTILE == 128u,
              "4096 reduction values are 128 scale blocks along K");
static_assert(SPARK_B12X_TILE_BYTES == 1024u, "a 32x32 block is 1024 weight bytes, one scale");

extern "C" __global__ void b12x_mxfp8_wo_a_sm121(
    const uint8_t *__restrict__ attn_fp8,     // [groups][group_inner] E4M3 attention outputs
    const uint8_t *__restrict__ attn_scales,  // [groups][group_inner / 32] ue8m0
    const uint8_t *__restrict__ w_fp8,        // B12x tiled [groups][o_lora_rank][group_inner]
    const uint8_t *__restrict__ w_scales,     // B12x tiled ue8m0 scales
    __nv_bfloat16 *__restrict__ lo_out,       // [groups][o_lora_rank] bf16 result
    const uint32_t groups,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t group = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t column = blockIdx.y * SPARK_M4_WOA_TILE + lane;
    const uint32_t k_blocks = SPARK_M4_WOA_GROUP_INNER / SPARK_M4_QUANT_BLOCK;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    __shared__ uint8_t s_a_row[SPARK_M4_WOA_GROUP_INNER];
    __shared__ uint8_t s_a_scales[SPARK_M4_WOA_GROUP_INNER / SPARK_M4_QUANT_BLOCK];
    if (group < groups) {
        const uint8_t *a_row = attn_fp8 + (uint64_t)group * SPARK_M4_WOA_GROUP_INNER;
        const uint8_t *a_scale_row = attn_scales + (uint64_t)group * k_blocks;
        for (uint32_t k = lane; k < (uint32_t)SPARK_M4_WOA_GROUP_INNER;
             k += SPARK_M4_WOA_THREADS) {
            s_a_row[k] = a_row[k];
        }
        if (lane < k_blocks) {
            s_a_scales[lane] = a_scale_row[lane];
        }
    }
    __syncthreads();

    if (group < groups && column < (uint32_t)SPARK_M4_O_LORA_RANK) {
        // Each group owns a weight buffer AND a scale buffer, and the two are different sizes, so
        // there are two group strides. The offsets inside each buffer come from the same helpers the
        // dense kernel uses.
        const uint64_t group_weight_base =
            (uint64_t)group * spark_b12x_buffer_bytes(SPARK_M4_O_LORA_RANK, SPARK_M4_WOA_GROUP_INNER);
        const uint64_t group_scale_base =
            (uint64_t)group * spark_b12x_scale_bytes(SPARK_M4_O_LORA_RANK, SPARK_M4_WOA_GROUP_INNER);
        float acc = 0.0f;
        for (uint32_t block = 0; block < k_blocks; ++block) {
            const uint32_t base = block * SPARK_M4_QUANT_BLOCK;
            const float a_scale = spark_ue8m0_decode(s_a_scales[block]);
            const float w_scale =
                spark_ue8m0_decode(w_scales[group_scale_base +
                                            spark_b12x_scale_index(column, base, SPARK_M4_WOA_GROUP_INNER)]);
#pragma unroll 8
            for (uint32_t i = 0; i < SPARK_M4_QUANT_BLOCK; ++i) {
                const float a_value = spark_fp8_e4m3_decode(s_a_row[base + i]) * a_scale;
                const float w_value =
                    spark_fp8_e4m3_decode(
                        w_fp8[group_weight_base +
                              spark_b12x_weight_index(column, base + i, SPARK_M4_WOA_GROUP_INNER)]) *
                    w_scale;
                acc += a_value * w_value;
            }
        }
        lo_out[(uint64_t)group * SPARK_M4_O_LORA_RANK + column] = __float2bfloat16_rn(acc);
    }
    __syncthreads();

    if (lane == 0u) {
        const uint32_t meta = SPARK_M4_WOA_TILE * 2u;
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}

// ---------------------------------------------------------------------------------------------
// K1 warp-per-column wo_a path. One warp owns one output channel of one group; the 32 lanes walk
// the 4096-value group reduction in 512-value groups (16 consecutive E4M3 bytes per lane), the
// five-step shuffle folds the lane partials, and the eight warps of a block cover eight output
// channels. This is the same tree-vs-serial tolerance as the dense warp symbol; the scalar symbol
// above stays the pinned oracle.
#define SPARK_M4_WOA_WARP_THREADS 256u
#define SPARK_M4_WOA_WARP_LANES 32u
#define SPARK_M4_WOA_WARP_WARPS (SPARK_M4_WOA_WARP_THREADS / SPARK_M4_WOA_WARP_LANES)
#define SPARK_M4_WOA_WARP_GROUP 512u
#define SPARK_M4_WOA_WARP_BYTES 16u

static_assert(SPARK_M4_WOA_WARP_THREADS == SPARK_M4_WOA_WARP_WARPS * SPARK_M4_WOA_WARP_LANES,
              "the wo_a warp block is whole warps");
static_assert(SPARK_M4_WOA_WARP_GROUP == SPARK_M4_WOA_WARP_LANES * SPARK_M4_WOA_WARP_BYTES,
              "one wo_a group is one vector load per lane");
static_assert(SPARK_M4_WOA_WARP_BYTES == 16u, "a wo_a lane load is 16 E4M3 bytes");
static_assert(SPARK_M4_WOA_GROUP_INNER % SPARK_M4_WOA_WARP_GROUP == 0u,
              "the wo_a reduction is whole warp groups");

__device__ __forceinline__ uint4 spark_woa_load128(const uint8_t *p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint8_t spark_woa_byte(const uint4 v, uint32_t i) {
    const uint32_t word = i >> 2;
    const uint32_t shift = (i & 3u) * 8u;
    const uint32_t w = word == 0u ? v.x : word == 1u ? v.y : word == 2u ? v.z : v.w;
    return (uint8_t)((w >> shift) & 0xffu);
}

__device__ __forceinline__ float spark_woa_warp_sum(float value) {
    value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 16u));
    value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 8u));
    value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 4u));
    value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 2u));
    value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 1u));
    return value;
}

extern "C" __global__ void b12x_mxfp8_wo_a_warp_sm121(
    const uint8_t *__restrict__ attn_fp8,
    const uint8_t *__restrict__ attn_scales,
    const uint8_t *__restrict__ w_fp8,
    const uint8_t *__restrict__ w_scales,
    __nv_bfloat16 *__restrict__ lo_out,
    const uint32_t groups,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t lane = threadIdx.x % SPARK_M4_WOA_WARP_LANES;
    const uint32_t warp = threadIdx.x / SPARK_M4_WOA_WARP_LANES;
    const uint32_t group = blockIdx.y;
    const uint32_t col = blockIdx.x * SPARK_M4_WOA_WARP_WARPS + warp;
    const uint32_t k_blocks = SPARK_M4_WOA_GROUP_INNER / SPARK_M4_QUANT_BLOCK;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);
    if (group < groups && col < (uint32_t)SPARK_M4_O_LORA_RANK) {
        const uint8_t *a_row = attn_fp8 + (uint64_t)group * SPARK_M4_WOA_GROUP_INNER;
        const uint8_t *a_scale_row = attn_scales + (uint64_t)group * k_blocks;
        const uint64_t group_weight_base =
            (uint64_t)group * spark_b12x_buffer_bytes(SPARK_M4_O_LORA_RANK, SPARK_M4_WOA_GROUP_INNER);
        const uint64_t group_scale_base =
            (uint64_t)group * spark_b12x_scale_bytes(SPARK_M4_O_LORA_RANK, SPARK_M4_WOA_GROUP_INNER);
        float acc = 0.0f;
        for (uint32_t base = lane * SPARK_M4_WOA_WARP_BYTES;
             base < (uint32_t)SPARK_M4_WOA_GROUP_INNER;
             base += SPARK_M4_WOA_WARP_GROUP) {
            const uint32_t block = base / SPARK_M4_QUANT_BLOCK;
            const uint4 a_vec = spark_woa_load128(a_row + base);
            const uint4 w_vec = spark_woa_load128(
                w_fp8 + group_weight_base +
                spark_b12x_weight_index(col, base, SPARK_M4_WOA_GROUP_INNER));
            const float a_scale = spark_ue8m0_decode(a_scale_row[block]);
            const float w_scale = spark_ue8m0_decode(
                w_scales[group_scale_base +
                         spark_b12x_scale_index(col, base, SPARK_M4_WOA_GROUP_INNER)]);
            float local = 0.0f;
            for (uint32_t i = 0u; i < SPARK_M4_WOA_WARP_BYTES; ++i) {
                const float a_value =
                    spark_fp8_e4m3_decode(spark_woa_byte(a_vec, i)) * a_scale;
                const float w_value =
                    spark_fp8_e4m3_decode(spark_woa_byte(w_vec, i)) * w_scale;
                local = __fadd_rn(local, __fmul_rn(a_value, w_value));
            }
            const float total = spark_woa_warp_sum(local);
            if (lane == 0u) {
                acc = __fadd_rn(acc, total);
            }
        }
        if (lane == 0u) {
            lo_out[(uint64_t)group * SPARK_M4_O_LORA_RANK + col] = __float2bfloat16_rn(acc);
        }
    }
    if (threadIdx.x == 0u) {
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, 0u, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}
// ---------------------------------------------------------------------------------------------
// K1-b grouped wo_a A-reuse tile: one block owns one group and 128 output channels. The group's
// single 4096-value activation row is staged in shared memory once, the B12x weights stream, and
// one thread per output channel keeps the scalar K-ascending f32 order, so this path is
// BIT-IDENTICAL to mxfp8_ref::grouped_wo_a (the scalar oracle above). E4M3 pairs are converted with
// cvt.rn.f16x2.e4m3x2, which is exact for every E4M3 value.
#define SPARK_M4_WOA_TILE_THREADS 128u
#define SPARK_M4_WOA_TILE_COLUMNS 128u
#define SPARK_M4_WOA_TILE_STAGE 4096u

static_assert(SPARK_M4_WOA_TILE_COLUMNS == SPARK_M4_WOA_TILE_THREADS,
              "one thread per output channel of the tile");
static_assert(SPARK_M4_WOA_TILE_STAGE == SPARK_M4_WOA_GROUP_INNER,
              "the wo_a group reduction fits one stage");

__device__ __forceinline__ uint16_t spark_woa_u16_pair(const uint4 v, uint32_t i) {
    const uint32_t word = i < 4u ? v.x : i < 8u ? v.y : i < 12u ? v.z : v.w;
    return (uint16_t)((word >> ((i & 3u) * 8u)) & 0xffffu);
}

__device__ __forceinline__ void spark_woa_fp8x2_to_f32(uint16_t two, float &lo, float &hi) {
    asm("{\n\t"
        ".reg .b32 h2;\n\t"
        ".reg .b16 lo, hi;\n\t"
        "cvt.rn.f16x2.e4m3x2 h2, %2;\n\t"
        "mov.b32 {lo, hi}, h2;\n\t"
        "cvt.f32.f16 %0, lo;\n\t"
        "cvt.f32.f16 %1, hi;\n\t"
        "}"
        : "=f"(lo), "=f"(hi)
        : "h"(two));
}

extern "C" __global__ void b12x_mxfp8_wo_a_tile_sm121(
    const uint8_t *__restrict__ attn_fp8,
    const uint8_t *__restrict__ attn_scales,
    const uint8_t *__restrict__ w_fp8,
    const uint8_t *__restrict__ w_scales,
    __nv_bfloat16 *__restrict__ lo_out,
    const uint32_t groups,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t group = blockIdx.y;
    const uint32_t col = blockIdx.x * SPARK_M4_WOA_TILE_COLUMNS + threadIdx.x;
    const uint32_t k_blocks = SPARK_M4_WOA_GROUP_INNER / SPARK_M4_QUANT_BLOCK;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    __shared__ __align__(16) uint8_t a_row[SPARK_M4_WOA_TILE_STAGE];
    __shared__ uint8_t a_scales_row[SPARK_M4_WOA_TILE_STAGE / SPARK_M4_QUANT_BLOCK];
    if (group < groups) {
        const uint8_t *a_ptr = attn_fp8 + (uint64_t)group * SPARK_M4_WOA_GROUP_INNER;
        const uint8_t *as_ptr = attn_scales + (uint64_t)group * k_blocks;
        for (uint32_t off = threadIdx.x * 32u; off < (uint32_t)SPARK_M4_WOA_GROUP_INNER;
             off += SPARK_M4_WOA_TILE_THREADS * 32u) {
            const uint4 v = spark_woa_load128(a_ptr + off);
            *reinterpret_cast<uint4 *>(&a_row[off]) = v;
            const uint4 v2 = spark_woa_load128(a_ptr + off + 16u);
            *reinterpret_cast<uint4 *>(&a_row[off + 16u]) = v2;
        }
        for (uint32_t b = threadIdx.x; b < k_blocks; b += SPARK_M4_WOA_TILE_THREADS) {
            a_scales_row[b] = as_ptr[b];
        }
    }
    __syncthreads();

    float acc = 0.0f;
    if (group < groups && col < (uint32_t)SPARK_M4_O_LORA_RANK) {
        const uint64_t group_weight_base =
            (uint64_t)group * spark_b12x_buffer_bytes(SPARK_M4_O_LORA_RANK, SPARK_M4_WOA_GROUP_INNER);
        const uint64_t group_scale_base =
            (uint64_t)group * spark_b12x_scale_bytes(SPARK_M4_O_LORA_RANK, SPARK_M4_WOA_GROUP_INNER);
        for (uint32_t b = 0u; b < k_blocks; ++b) {
            const uint32_t base = b * SPARK_M4_QUANT_BLOCK;
            const float a_scale = spark_ue8m0_decode(a_scales_row[b]);
            const uint64_t w_at = group_weight_base +
                                  spark_b12x_weight_index(col, base, SPARK_M4_WOA_GROUP_INNER);
            const uint4 w0 = spark_woa_load128(w_fp8 + w_at);
            const uint4 w1 = spark_woa_load128(w_fp8 + w_at + 16u);
            const float w_scale = spark_ue8m0_decode(
                w_scales[group_scale_base +
                         spark_b12x_scale_index(col, base, SPARK_M4_WOA_GROUP_INNER)]);
#pragma unroll
            for (uint32_t i = 0u; i < 32u; i += 2u) {
                const uint16_t a_two =
                    *reinterpret_cast<const uint16_t *>(&a_row[base + i]);
                const uint16_t w_two = spark_woa_u16_pair(i < 16u ? w0 : w1, i & 15u);
                float a_lo = 0.0f;
                float a_hi = 0.0f;
                float w_lo = 0.0f;
                float w_hi = 0.0f;
                spark_woa_fp8x2_to_f32(a_two, a_lo, a_hi);
                spark_woa_fp8x2_to_f32(w_two, w_lo, w_hi);
                acc = __fadd_rn(acc, __fmul_rn(a_lo * a_scale, w_lo * w_scale));
                acc = __fadd_rn(acc, __fmul_rn(a_hi * a_scale, w_hi * w_scale));
            }
        }
        lo_out[(uint64_t)group * SPARK_M4_O_LORA_RANK + col] = __float2bfloat16_rn(acc);
    }
    if (threadIdx.x == 0u) {
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, 0u, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}
// ---------------------------------------------------------------------------------------------
// K8 batched grouped wo_a: one launch for up to six positions. One block owns (group, 128 output
// channels) and stages every position's 4096-value activation row in shared memory once; the B12x
// weights stream and the six row threads of a channel share the same 32-byte weight line in L1, so
// the weights are read once per block rather than once per row. Each output keeps the scalar
// K-ascending f32 order and stays BIT-IDENTICAL to mxfp8_ref::grouped_wo_a.
#define SPARK_M4_WOA_BATCH_ROWS 6u
#define SPARK_M4_WOA_BATCH_THREADS (SPARK_M4_WOA_TILE_COLUMNS * SPARK_M4_WOA_BATCH_ROWS)

static_assert(SPARK_M4_WOA_BATCH_THREADS == 768u,
              "six rows x 128 channels is one 768-thread block");
static_assert(SPARK_M4_WOA_BATCH_ROWS * SPARK_M4_WOA_TILE_STAGE <= 24576u + 1024u,
              "the six staged activation rows plus scales fit shared memory");

extern "C" __global__ void b12x_mxfp8_wo_a_batched_perrow_sm121(
    const uint8_t *__restrict__ attn_fp8,
    const uint8_t *__restrict__ attn_scales,
    const uint8_t *__restrict__ w_fp8,
    const uint8_t *__restrict__ w_scales,
    __nv_bfloat16 *__restrict__ lo_out,
    const uint32_t groups,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t rows = gridDim.z;
    const uint32_t group = blockIdx.y;
    const uint32_t row = blockIdx.z;
    const uint32_t lane = threadIdx.x % SPARK_M4_WOA_TILE_COLUMNS;
    const uint32_t lane_row = threadIdx.x / SPARK_M4_WOA_TILE_COLUMNS;
    const uint32_t col = blockIdx.x * SPARK_M4_WOA_TILE_COLUMNS + lane;
    const uint32_t k_blocks = SPARK_M4_WOA_GROUP_INNER / SPARK_M4_QUANT_BLOCK;
    const uint64_t attn_row_stride =
        (uint64_t)SPARK_M4_O_GROUPS * SPARK_M4_WOA_GROUP_INNER;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    __shared__ __align__(16) uint8_t a_rows[SPARK_M4_WOA_BATCH_ROWS][SPARK_M4_WOA_TILE_STAGE];
    __shared__ uint8_t a_scale_rows[SPARK_M4_WOA_BATCH_ROWS]
                                    [SPARK_M4_WOA_TILE_STAGE / SPARK_M4_QUANT_BLOCK];

    if (rows > SPARK_M4_WOA_BATCH_ROWS) {
        if (threadIdx.x == 0u) {
            lo_out[0] = __float2bfloat16_rn(__uint_as_float(0x7FC00000u));
        }
        return;
    }

    if (lane_row < rows && group < groups) {
        const uint8_t *a_ptr = attn_fp8 + (uint64_t)lane_row * attn_row_stride +
                               (uint64_t)group * SPARK_M4_WOA_GROUP_INNER;
        const uint8_t *as_ptr = attn_scales +
                                (uint64_t)lane_row * (attn_row_stride / SPARK_M4_QUANT_BLOCK) +
                                (uint64_t)group * k_blocks;
        const uint4 v0 = spark_woa_load128(a_ptr + lane * 32u);
        const uint4 v1 = spark_woa_load128(a_ptr + lane * 32u + 16u);
        *reinterpret_cast<uint4 *>(&a_rows[lane_row][lane * 32u]) = v0;
        *reinterpret_cast<uint4 *>(&a_rows[lane_row][lane * 32u + 16u]) = v1;
        a_scale_rows[lane_row][lane] = as_ptr[lane];
    }
    __syncthreads();

    if (lane_row < rows && group < groups && col < (uint32_t)SPARK_M4_O_LORA_RANK) {
        const uint32_t row_index = lane_row;
        const uint64_t group_weight_base =
            (uint64_t)group * spark_b12x_buffer_bytes(SPARK_M4_O_LORA_RANK, SPARK_M4_WOA_GROUP_INNER);
        const uint64_t group_scale_base =
            (uint64_t)group * spark_b12x_scale_bytes(SPARK_M4_O_LORA_RANK, SPARK_M4_WOA_GROUP_INNER);
        float acc = 0.0f;
        for (uint32_t b = 0u; b < k_blocks; ++b) {
            const uint32_t base = b * SPARK_M4_QUANT_BLOCK;
            const float a_scale = spark_ue8m0_decode(a_scale_rows[row_index][b]);
            const uint64_t w_at = group_weight_base +
                                  spark_b12x_weight_index(col, base, SPARK_M4_WOA_GROUP_INNER);
            const uint4 w0 = spark_woa_load128(w_fp8 + w_at);
            const uint4 w1 = spark_woa_load128(w_fp8 + w_at + 16u);
            const float w_scale = spark_ue8m0_decode(
                w_scales[group_scale_base +
                         spark_b12x_scale_index(col, base, SPARK_M4_WOA_GROUP_INNER)]);
#pragma unroll
            for (uint32_t i = 0u; i < 32u; i += 2u) {
                const uint16_t a_two =
                    *reinterpret_cast<const uint16_t *>(&a_rows[row_index][base + i]);
                const uint16_t w_two = spark_woa_u16_pair(i < 16u ? w0 : w1, i & 15u);
                float a_lo = 0.0f;
                float a_hi = 0.0f;
                float w_lo = 0.0f;
                float w_hi = 0.0f;
                spark_woa_fp8x2_to_f32(a_two, a_lo, a_hi);
                spark_woa_fp8x2_to_f32(w_two, w_lo, w_hi);
                acc = __fadd_rn(acc, __fmul_rn(a_lo * a_scale, w_lo * w_scale));
                acc = __fadd_rn(acc, __fmul_rn(a_hi * a_scale, w_hi * w_scale));
            }
        }
        lo_out[(uint64_t)row_index * SPARK_M4_O_GROUPS * SPARK_M4_O_LORA_RANK +
               (uint64_t)group * SPARK_M4_O_LORA_RANK + col] = __float2bfloat16_rn(acc);
    }
    if (threadIdx.x == 0u) {
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, 0u, t_enter);
        (void)step_seq;
        (void)row;
    }
    (void)t_enter;
}
// K8 batched wo_a, weight-reuse shape: one block owns (group, 128 channels) and one thread owns
// one channel across all six positions. The six activation rows are staged in shared memory once;
// the thread loads and decodes each 32-value weight block once and applies it to six independent
// f32 accumulators, so weight instruction issue drops sixfold instead of being repeated per row.
// Each accumulator still adds in scalar K-ascending order, so every row is BIT-IDENTICAL to
// mxfp8_ref::grouped_wo_a.
#define SPARK_M4_WOA_BATCH6_ROWS 6u
#define SPARK_M4_WOA_BATCH6_THREADS SPARK_M4_WOA_TILE_COLUMNS

static_assert(SPARK_M4_WOA_BATCH6_ROWS == 6u, "the six-row verify batch is the fixed shape");

extern "C" __global__ void b12x_mxfp8_wo_a_batched_sm121(
    const uint8_t *__restrict__ attn_fp8,
    const uint8_t *__restrict__ attn_scales,
    const uint8_t *__restrict__ w_fp8,
    const uint8_t *__restrict__ w_scales,
    __nv_bfloat16 *__restrict__ lo_out,
    const uint32_t groups,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t group = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t col = blockIdx.x * SPARK_M4_WOA_TILE_COLUMNS + lane;
    const uint32_t k_blocks = SPARK_M4_WOA_GROUP_INNER / SPARK_M4_QUANT_BLOCK;
    const uint64_t attn_row_stride =
        (uint64_t)SPARK_M4_O_GROUPS * SPARK_M4_WOA_GROUP_INNER;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    __shared__ __align__(16) uint8_t a_rows[SPARK_M4_WOA_BATCH6_ROWS][SPARK_M4_WOA_TILE_STAGE];
    __shared__ uint8_t a_scale_rows[SPARK_M4_WOA_BATCH6_ROWS]
                                    [SPARK_M4_WOA_TILE_STAGE / SPARK_M4_QUANT_BLOCK];

    if (group < groups) {
#pragma unroll
        for (uint32_t r = 0u; r < SPARK_M4_WOA_BATCH6_ROWS; ++r) {
            const uint8_t *a_ptr = attn_fp8 + (uint64_t)r * attn_row_stride +
                                   (uint64_t)group * SPARK_M4_WOA_GROUP_INNER;
            const uint8_t *as_ptr = attn_scales +
                                    (uint64_t)r * (attn_row_stride / SPARK_M4_QUANT_BLOCK) +
                                    (uint64_t)group * k_blocks;
            const uint4 v0 = spark_woa_load128(a_ptr + lane * 32u);
            const uint4 v1 = spark_woa_load128(a_ptr + lane * 32u + 16u);
            *reinterpret_cast<uint4 *>(&a_rows[r][lane * 32u]) = v0;
            *reinterpret_cast<uint4 *>(&a_rows[r][lane * 32u + 16u]) = v1;
            a_scale_rows[r][lane] = as_ptr[lane];
        }
    }
    __syncthreads();

    if (group < groups && col < (uint32_t)SPARK_M4_O_LORA_RANK) {
        const uint64_t group_weight_base =
            (uint64_t)group * spark_b12x_buffer_bytes(SPARK_M4_O_LORA_RANK, SPARK_M4_WOA_GROUP_INNER);
        const uint64_t group_scale_base =
            (uint64_t)group * spark_b12x_scale_bytes(SPARK_M4_O_LORA_RANK, SPARK_M4_WOA_GROUP_INNER);
        float acc[SPARK_M4_WOA_BATCH6_ROWS];
#pragma unroll
        for (uint32_t r = 0u; r < SPARK_M4_WOA_BATCH6_ROWS; ++r) {
            acc[r] = 0.0f;
        }
        for (uint32_t b = 0u; b < k_blocks; ++b) {
            const uint32_t base = b * SPARK_M4_QUANT_BLOCK;
            const uint64_t w_at = group_weight_base +
                                  spark_b12x_weight_index(col, base, SPARK_M4_WOA_GROUP_INNER);
            const uint4 w0 = spark_woa_load128(w_fp8 + w_at);
            const uint4 w1 = spark_woa_load128(w_fp8 + w_at + 16u);
            const float w_scale = spark_ue8m0_decode(
                w_scales[group_scale_base +
                         spark_b12x_scale_index(col, base, SPARK_M4_WOA_GROUP_INNER)]);
            float row_scale[SPARK_M4_WOA_BATCH6_ROWS];
#pragma unroll
            for (uint32_t r = 0u; r < SPARK_M4_WOA_BATCH6_ROWS; ++r) {
                row_scale[r] = spark_ue8m0_decode(a_scale_rows[r][b]);
            }
#pragma unroll
            for (uint32_t i = 0u; i < 32u; i += 2u) {
                const uint16_t w_two = spark_woa_u16_pair(i < 16u ? w0 : w1, i & 15u);
                float w_lo = 0.0f;
                float w_hi = 0.0f;
                spark_woa_fp8x2_to_f32(w_two, w_lo, w_hi);
                const float w_lo_scaled = w_lo * w_scale;
                const float w_hi_scaled = w_hi * w_scale;
#pragma unroll
                for (uint32_t r = 0u; r < SPARK_M4_WOA_BATCH6_ROWS; ++r) {
                    const uint16_t a_two =
                        *reinterpret_cast<const uint16_t *>(&a_rows[r][base + i]);
                    float a_lo = 0.0f;
                    float a_hi = 0.0f;
                    spark_woa_fp8x2_to_f32(a_two, a_lo, a_hi);
                    acc[r] = __fadd_rn(acc[r], __fmul_rn(a_lo * row_scale[r], w_lo_scaled));
                    acc[r] = __fadd_rn(acc[r], __fmul_rn(a_hi * row_scale[r], w_hi_scaled));
                }
            }
        }
#pragma unroll
        for (uint32_t r = 0u; r < SPARK_M4_WOA_BATCH6_ROWS; ++r) {
            lo_out[(uint64_t)r * SPARK_M4_O_GROUPS * SPARK_M4_O_LORA_RANK +
                   (uint64_t)group * SPARK_M4_O_LORA_RANK + col] = __float2bfloat16_rn(acc[r]);
        }
    }
    if (threadIdx.x == 0u) {
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, 0u, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}

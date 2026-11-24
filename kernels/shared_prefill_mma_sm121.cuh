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

// kernels/shared_prefill_mma_sm121.cuh
//
// The one bulk-prefill tensor-core tile the GLM dense projections, GLM routed experts and the
// Qwen3.8-Flash-Next routed experts launch. One block computes 128 token rows x 128 output
// columns, K in 32-column stages, and a routed-expert form reads its rows from the shared plan
// sorted array at the item's first rank.
//
// The weight format is a template parameter, so the same body serves
//   * BF16 weights (GLM's dense BF16 projections),
//   * E4M3 weights with one block scale a 128 x 128 tile, where the scale is f32 (GLM) or
//     __nv_bfloat16 (Qwen's FP8 release) - typed, never indexed by byte (D88).
//
// Qwen's NVFP4 experts keep their own q4p_moe_body: this body has no E2M1 weight form, because
// that body applies the per-tensor scale to a bf16 weight where the scalar decoder applies it in
// f32 after accumulation, and an unused branch is not parity.
//
// The activation is f32 split into a bf16 head and a bf16 remainder, or the head alone. The
// output is f32. A routed-expert item's rows are its GLM_MOE_ROWS entries of the shared plan's
// sorted array, from the item's first rank: without route, row r reads the token row of slot
// sorted[first + r]; with route, it reads the item row and writes slot row sorted[first + r]
// times route[slot]. kSlotRows reads each item row from its slot row (Qwen's prefill); without
// route, kOutSlotRows writes each row to its slot row as well, so gate/up can hand SwiGLU the
// [slot][width] activation the down pass reads.
//
// The tile geometry is a tuning decision the dense GEMM, the routed experts and the sparse MLA
// attention share, so it is stated once here. GLM's images and Qwen's prefill include this
// header and instantiate the body; none carries a second staging loop. Two more bodies live
// here with it: the q8 body (E4M3 activation codes into an E4M3 MMA) and the wide body (a dense
// BF16 weight over a 128 x 256 tile, the 128 x 128 body's bits).

#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "opt_mma_common_sm121.cuh"
#include "opt_mma_bf16_sm121.cuh"
#include "quant_decode_sm121.cuh"

// The shared tile geometry: 128 rows x 128 columns, 32-column K stages.
#define GLM_BM 128u
#define GLM_BN 128u
#define GLM_A_PER 4u
#define GLM_W_TILES 4u
// A routed-expert item: up to this many of one expert's slots, one row tile.
#define GLM_MOE_ROWS 128u
// The row tiles a dense GEMM's grid walks down each column tile before the next ([glm_raster]).
#define GLM_RASTER 8u

// The q8 body's cp.async stages, and its dynamic shared memory: a stage's activation tile, weight
// tiles and the tile rows' activation scales, then the tile rows' activation rows and the column
// tile's weight scales (an inner of at most GLM_Q8_BLOCKS 128-column blocks).
#define GLM_Q8_STAGES 4u
#define GLM_Q8_BLOCKS 128u
#define GLM_Q8_STAGE_BYTES (GLM_BM * 64u + GLM_BN * 64u + GLM_BM * 4u)
#define GLM_Q8_SHARED (GLM_Q8_STAGES * GLM_Q8_STAGE_BYTES + GLM_BM * 4u + GLM_Q8_BLOCKS * 4u)

// The wide body's tile (128 rows x GLM_WIDE_BN columns), its cp.async stages and its dynamic
// shared memory: a stage's f32 activation tile (128 rows x 32 K) and BF16 weight tile
// (GLM_WIDE_BN rows x 32 K).
#define GLM_WIDE_BN 256u
#define GLM_WIDE_STAGES 3u
#define GLM_WIDE_STAGE_BYTES (GLM_BM * 128u + GLM_WIDE_BN * 64u)
#define GLM_WIDE_SHARED (GLM_WIDE_STAGES * GLM_WIDE_STAGE_BYTES)

// The weight formats the tile body serves.
#define SPM_W_BF16 0
#define SPM_W_FP8 1
// E2M1 codes with one E4M3 scale a 16-value group, dequantised into the tile's BF16 weight
// words. kWide selects where the convention's tensor-wide factor is applied: 1 folds it into the
// group scale at the load (Qwen), 2 applies it to the output (Inkling), 0 carries none.
#define SPM_W_FP4 2
// E4M3 codes with one ue8m0 scale a row every 32 values along the contraction, row-major
// (StoredFormat::MXFP8_ROW_32). E4M3 times a power of two is exact in the tile's BF16 weight
// words, so the stage folds the scale at the load, as SPM_W_FP4 folds its group scale.
#define SPM_W_MX8R 3

// A dense GEMM block's (column tile, row tile) for the grid's (x column tiles, y row tiles): the
// linear block order walks GLM_RASTER row tiles down each column tile before moving on, so the
// blocks resident at once share a few activation row tiles and weight column tiles. In the
// grid's own order each row tile streams the whole weight, which from the DRAM, once the weight
// is past the L2 (a 6144 x 4096 FP8 one), costs a GEMM four to five times its arithmetic.
__device__ __forceinline__ uint2 glm_raster() {
    const uint32_t id = blockIdx.y * gridDim.x + blockIdx.x;
    const uint32_t per_group = GLM_RASTER * gridDim.x;
    const uint32_t first = id / per_group * GLM_RASTER;
    const uint32_t size = min(gridDim.y - first, GLM_RASTER);
    const uint32_t within = id % per_group;
    return make_uint2(within / size, first + within % size);
}


// One 128-column block of a row, four columns a lane of a warp, as the release's dynamic
// activation: its four codes (low byte first) and, in `scale`, amax / 448 (zero for an all-zero
// block, whose codes are zero).
__device__ __forceinline__ uint32_t glm_q8_codes(const float4 v, float &scale) {
    float largest = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        largest = fmaxf(largest, __shfl_xor_sync(0xFFFFFFFFu, largest, offset));
    scale = largest / 448.0f;
    if (scale == 0.0f) return 0u;
    return (uint32_t)__nv_cvt_float_to_fp8(v.x / scale, __NV_SATFINITE, __NV_E4M3) |
           (uint32_t)__nv_cvt_float_to_fp8(v.y / scale, __NV_SATFINITE, __NV_E4M3) << 8 |
           (uint32_t)__nv_cvt_float_to_fp8(v.z / scale, __NV_SATFINITE, __NV_E4M3) << 16 |
           (uint32_t)__nv_cvt_float_to_fp8(v.w / scale, __NV_SATFINITE, __NV_E4M3) << 24;
}


// A stage's activation and weight tiles are [128 rows][4 chunks of 16 B] (64 K a stage, two 32-K
// MMA blocks), swizzled as opt_mma_sm121.cu's om_a_off lays the activation out. A weight row's
// 64 bytes stay together: split into opt_mma's two 32-K tiles, the cp.async stage streams the
// weights ~40% slower (126 against 205 GB/s on a routed expert's shape).
__device__ __forceinline__ uint32_t glm_q8_off(uint32_t row, uint32_t chunk) {
    return row * 64u + ((chunk ^ ((row >> 1) & 3u)) << 4);
}


// A stage's f32 activation tile is [128 rows][8 chunks of 16 B] (32 K), the chunks XOR-swizzled
// by twice the row's low two bits: a fragment load reads two adjacent chunks of eight rows, and
// each half-warp's four rows then fall on eight distinct chunks, every bank once.
__device__ __forceinline__ uint32_t glm_wide_a_off(uint32_t row, uint32_t chunk) {
    return row * 128u + ((chunk ^ ((row & 3u) << 1)) << 4);
}


// The block scale of a tile as f32, whatever dtype the release stores it in: GLM's E4M3
// release keeps one f32 a 128 x 128 tile, Qwen's FP8 release one __nv_bfloat16 (D88).
template <typename ScaleT>
__device__ __forceinline__ float spm_scale(const ScaleT value) {
    if constexpr (sizeof(ScaleT) == 2) {
        return __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&value));
    } else {
        return static_cast<float>(value);
    }
}


// One block: 128 token rows x 128 output columns, K in 32-column blocks. w/ws name a single
// matrix (dense; group z's is w_grp_stride bytes and ws_grp_stride scales on, a head's slice of a
// per-head projection), or table names the experts' [weight, scales] pointer pairs and items the
// per-group (expert, ..., count, ...) list (MoE): group z is item z, its count the rows. An MoE
// item's rows are its GLM_MOE_ROWS entries of the shared plan's `sorted` array, from the item's
// first rank: without route, row r reads the token row of slot sorted[first + r] (a_tok_stride
// floats a token) and writes the padded row; with route, it reads
// the padded row and writes slot row sorted[first + r] (out_tok_stride floats a slot) times
// route[slot]. kSlotRows reads the slot row instead of the token row; kOutSlotRows writes the
// slot row instead of the padded item row. route_stride, when nonzero, reads a routing weight
// from route[(slot / topk) * route_stride + slot % topk] - Qwen's [batch][experts] probability
// rows - and 0 reads route[slot], GLM's per-slot combine weights.
// The activation is f32 split into bf16 head and remainder; the output is f32.
template <int kFormat, bool kRoutedMoe, bool kLow, typename ScaleT = float,
          int kTableStride = 2, int kTableOffset = 0, bool kSlotRows = false,
          bool kOutSlotRows = false, int kWide = 0>
__device__ __forceinline__ void shared_prefill_mma_body(
    const float *__restrict__ a, const uint8_t *__restrict__ w, const void *__restrict__ ws,
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    float *__restrict__ out, uint32_t cols, uint32_t inner, uint32_t tokens,
    uint64_t a_tok_stride, uint64_t out_tok_stride, uint64_t a_grp_stride,
    uint64_t out_grp_stride, uint64_t w_grp_stride = 0, uint64_t ws_grp_stride = 0,
    const int *__restrict__ sorted = nullptr, uint32_t topk = 1,
    const float *__restrict__ route = nullptr, uint32_t route_stride = 0,
    float wide = 1.0f) {
    constexpr bool kFp8 = kFormat == SPM_W_FP8;
    constexpr bool kFp4 = kFormat == SPM_W_FP4;
    constexpr bool kMx8r = kFormat == SPM_W_MX8R;
    constexpr bool kFoldWide = kFp4 && kWide == 1;
    constexpr bool kOutWide = kFp4 && kWide == 2;
    __shared__ __align__(16) __nv_bfloat16 s_ah[GLM_BM * OB_LD];
    __shared__ __align__(16) __nv_bfloat16 s_al[kLow ? GLM_BM * OB_LD : 1];
    __shared__ __align__(16) __nv_bfloat16 s_w[GLM_BN * OB_LD];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint2 place = kRoutedMoe ? make_uint2(blockIdx.x, blockIdx.y) : glm_raster();
    const uint32_t n0 = place.x * GLM_BN;
    const uint32_t m0 = place.y * GLM_BM;
    const uint32_t group = blockIdx.z;
    const uint32_t k_blocks = inner >> 5;
    const uint32_t col_tiles = min(GLM_BN / 32u, (cols - n0) >> 5);

    const uint8_t *w_grp = w + group * w_grp_stride;
    // ws_grp_stride is a count of ScaleT elements, not bytes: GLM's f32 tile scales and Qwen's
    // bf16 ones are both addressed typed (D88).
    const ScaleT *ws_grp =
        ws == nullptr ? nullptr
                      : reinterpret_cast<const ScaleT *>(ws) + group * ws_grp_stride;
    const int *item_slots = nullptr;
    if constexpr (kRoutedMoe) {
        const int4 item = items[group];
        if (item.z <= 0) return;
        tokens = min((uint32_t)item.z, GLM_MOE_ROWS);
        w_grp = reinterpret_cast<const uint8_t *>(table[(uint64_t)item.x * kTableStride +
                                                        kTableOffset]);
        // Every scaled format reads its expert's scale plane beside the weight. An NVFP4 expert
        // with a tensor-wide convention carries its f32 factor's bits in the third word
        // ([weight, scales, wide] a matrix), since the factor is the expert's, not the launch's.
        if constexpr (kFp8 || kFp4 || kMx8r) {
            ws_grp = reinterpret_cast<const ScaleT *>(
                table[(uint64_t)item.x * kTableStride + kTableOffset + 1]);
        }
        if constexpr (kFp4 && kWide != 0) {
            wide = __uint_as_float(
                (uint32_t)table[(uint64_t)item.x * kTableStride + kTableOffset + 2]);
        }
        item_slots = sorted + item.y;
    }
    const uint64_t a_grp = group * a_grp_stride;

    // The four activation rows this thread loads every stage, resolved once: the element offset
    // of each row's first column, or -1 for a row past the tile's tokens (loaded as zero).
    long long a_row[GLM_A_PER];
#pragma unroll
    for (uint32_t i = 0; i < GLM_A_PER; ++i) {
        const uint32_t token = m0 + ((tid + i * OM_THREADS) >> 3);
        a_row[i] = -1;
        if (token < tokens) {
            if (kRoutedMoe && route == nullptr) {
                const int slot = item_slots[token];
                if (slot >= 0) a_row[i] = (long long)(slot / (int)topk) * (long long)a_tok_stride;
            } else if (kRoutedMoe && kSlotRows) {
                const int slot = item_slots[token];
                if (slot >= 0) a_row[i] = (long long)slot * (long long)a_tok_stride;
            } else {
                a_row[i] = (long long)(a_grp + (uint64_t)token * a_tok_stride);
            }
        }
    }

    const uint32_t w_tile = tid >> 6;
    const uint32_t w_col = (tid & 63u) >> 1;
    const uint32_t w_half = tid & 1u;
    const bool w_live = w_tile < col_tiles;

    uint4 ra[GLM_A_PER];
    uint4 rw[4];
    float rs = 0.0f;
    // The scale the current stage's weight tile was stored with: a 128 x 128 block of GLM's
    // weights carries one f32 scale, Qwen's FP8 release one bf16 scale, applied to the block's
    // f32 partial sum after the MMAs.
    float stage_scale = 0.0f;

    auto load = [&](uint32_t kb) {
#pragma unroll
        for (uint32_t i = 0; i < GLM_A_PER; ++i) {
            const uint32_t idx = tid + i * OM_THREADS;
            const uint32_t row = idx >> 3;
            const uint32_t c = idx & 7u;
            (void)row;
            if (a_row[i] >= 0) {
                const uint64_t e = (uint64_t)a_row[i] + (uint64_t)kb * 32u + c * 4u;
                ra[i] = __ldg(reinterpret_cast<const uint4 *>(a + e));
            } else {
                ra[i] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        if (w_live) {
            const uint64_t col = n0 + w_tile * 32u + w_col;
            if constexpr (kFp8) {
                const uint8_t *src = w_grp + col * inner + kb * 32u + w_half * 16u;
                rw[0] = __ldg(reinterpret_cast<const uint4 *>(src));
                rs = spm_scale<ScaleT>(
                    __ldg(&ws_grp[(col >> 7) * (inner >> 7) + (kb >> 2)]));
            } else if constexpr (kMx8r) {
                // One 32-column stage of one output row: this half's 16 E4M3 codes and the row's
                // power-of-two scale for the stage's 32 columns.
                const uint8_t *src = w_grp + col * inner + kb * 32u + w_half * 16u;
                rw[0] = __ldg(reinterpret_cast<const uint4 *>(src));
                const uint8_t *scale_row =
                    reinterpret_cast<const uint8_t *>(ws_grp) + col * (inner >> 5);
                rs = opt_ue8m0_f32(__ldg(&scale_row[kb]));
            } else if constexpr (kFp4) {
                // One 32-column stage of one output row: 16 packed E2M1 codes (8 bytes) for this
                // half, and one E4M3 group scale for those 16 values.
                const uint8_t *src = w_grp + (col * inner + kb * 32u + w_half * 16u) / 2u;
                const uint2 packed = __ldg(reinterpret_cast<const uint2 *>(src));
                rw[0] = make_uint4(packed.x, packed.y, 0u, 0u);
                const uint8_t *scale_row =
                    reinterpret_cast<const uint8_t *>(ws_grp) + col * (inner / 16u);
                rs = spark_fp8_e4m3_decode(scale_row[kb * 2u + w_half]);
                if constexpr (kFoldWide) rs *= wide;
            } else {
                const uint8_t *src = w_grp + (col * inner + kb * 32u + w_half * 16u) * 2u;
                rw[0] = __ldg(reinterpret_cast<const uint4 *>(src));
                rw[1] = __ldg(reinterpret_cast<const uint4 *>(src) + 1);
            }
        } else {
#pragma unroll
            for (uint32_t v = 0; v < 4u; ++v) rw[v] = make_uint4(0u, 0u, 0u, 0u);
            rs = 0.0f;
        }
    };
    auto store = [&]() {
#pragma unroll
        for (uint32_t i = 0; i < GLM_A_PER; ++i) {
            const uint32_t idx = tid + i * OM_THREADS;
            const uint32_t row = idx >> 3;
            const uint32_t c = idx & 7u;
            const float v[4] = {__uint_as_float(ra[i].x), __uint_as_float(ra[i].y),
                                __uint_as_float(ra[i].z), __uint_as_float(ra[i].w)};
            __nv_bfloat16 h[4], l[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                h[e] = __float2bfloat16_rn(v[e]);
                l[e] = __float2bfloat16_rn(v[e] - __bfloat162float(h[e]));
            }
            *reinterpret_cast<uint2 *>(&s_ah[row * OB_LD + c * 4u]) =
                make_uint2(ob_pack(h[0], h[1]), ob_pack(h[2], h[3]));
            if constexpr (kLow) {
                *reinterpret_cast<uint2 *>(&s_al[row * OB_LD + c * 4u]) =
                    make_uint2(ob_pack(l[0], l[1]), ob_pack(l[2], l[3]));
            }
        }
        const uint32_t w_at = (w_tile * 32u + w_col) * OB_LD + w_half * 16u;
        __nv_bfloat16 *dst = &s_w[w_at];
        if constexpr (kMx8r) {
            const uint4 r = rw[0];
            *reinterpret_cast<uint4 *>(dst) =
                make_uint4(ob_e4m3x2(r.x, rs), ob_e4m3x2(r.x >> 16, rs),
                           ob_e4m3x2(r.y, rs), ob_e4m3x2(r.y >> 16, rs));
            *reinterpret_cast<uint4 *>(dst + 8) =
                make_uint4(ob_e4m3x2(r.z, rs), ob_e4m3x2(r.z >> 16, rs),
                           ob_e4m3x2(r.w, rs), ob_e4m3x2(r.w >> 16, rs));
        } else if constexpr (kFp8) {
            stage_scale = rs;
            const uint4 r = rw[0];
            *reinterpret_cast<uint4 *>(dst) =
                make_uint4(ob_e4m3x2(r.x, 1.0f), ob_e4m3x2(r.x >> 16, 1.0f),
                           ob_e4m3x2(r.y, 1.0f), ob_e4m3x2(r.y >> 16, 1.0f));
            *reinterpret_cast<uint4 *>(dst + 8) =
                make_uint4(ob_e4m3x2(r.z, 1.0f), ob_e4m3x2(r.z >> 16, 1.0f),
                           ob_e4m3x2(r.w, 1.0f), ob_e4m3x2(r.w >> 16, 1.0f));
        } else if constexpr (kFp4) {
            // E2M1 * E4M3 is exact in BF16, so the group scale folded into the staged word is
            // the release's own product. Qwen's weight global joins it here; Inkling's joins the
            // output below.
            const float scale = rs;
            const uint32_t words[2] = {rw[0].x, rw[0].y};
#pragma unroll
            for (uint32_t byte = 0; byte < 8u; ++byte) {
                const uint8_t code = (uint8_t)(words[byte >> 2] >> (8u * (byte & 3u)));
                dst[2u * byte] = __float2bfloat16_rn(
                    spark_fp4_e2m1_decode(code & 0x0Fu) * scale);
                dst[2u * byte + 1u] = __float2bfloat16_rn(
                    spark_fp4_e2m1_decode(code >> 4) * scale);
            }
        } else {
            *reinterpret_cast<uint4 *>(dst) = rw[0];
            *reinterpret_cast<uint4 *>(dst + 8) = rw[1];
        }
    };

    const uint32_t wm = warp & 1u;
    const uint32_t wn = warp >> 1;
    const bool active = wn < col_tiles && m0 + wm * 64u < tokens;
    // The warp's 16-row tiles that hold a token; the rest are zero rows and skip their MMAs.
    const uint32_t mt_live = active ? min(4u, (tokens - m0 - wm * 64u + 15u) >> 4) : 0u;

    // A block-scaled FP8 weight folds its scale into a per-128-column partial sum; the BF16
    // weight has none and accumulates the whole K run in one array.
    float accum[kFp8 ? 2 : 1][4][4][4];
#pragma unroll
    for (uint32_t a = 0; a < (kFp8 ? 2u : 1u); ++a)
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i)
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j)
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) accum[a][i][j][e] = 0.0f;

    load(0);
    for (uint32_t kb = 0; kb < k_blocks; ++kb) {
        store();
        __syncthreads();
        if (kb + 1u < k_blocks) load(kb + 1u);
        if (active) {
#pragma unroll
            for (uint32_t ks = 0; ks < 2u; ++ks) {
                const uint32_t k0 = ks * 16u;
                uint32_t b[4][2];
#pragma unroll
                for (uint32_t p = 0; p < 2u; ++p) {
                    const uint32_t n = wn * 32u + p * 16u + (lane & 7u) + 8u * (lane >> 4);
                    const uint32_t k = k0 + 8u * ((lane >> 3) & 1u);
                    om_ldsm4(om_smem(&s_w[n * OB_LD + k]), b[2 * p][0], b[2 * p][1],
                             b[2 * p + 1][0], b[2 * p + 1][1]);
                }
#pragma unroll
                for (uint32_t mt = 0; mt < 4u; ++mt) {
                    if (mt >= mt_live) break;
                    const uint32_t row = wm * 64u + mt * 16u + (lane & 15u);
                    const uint32_t k = k0 + 8u * (lane >> 4);
                    uint32_t af[4];
                    om_ldsm4(om_smem(&s_ah[row * OB_LD + k]), af[0], af[1], af[2], af[3]);
#pragma unroll
                    for (uint32_t nt = 0; nt < 4u; ++nt)
                        ob_mma(accum[kFp8 ? 1 : 0][mt][nt], af, b[nt][0], b[nt][1]);
                    if constexpr (kLow) {
                        om_ldsm4(om_smem(&s_al[row * OB_LD + k]), af[0], af[1], af[2], af[3]);
#pragma unroll
                        for (uint32_t nt = 0; nt < 4u; ++nt)
                            ob_mma(accum[kFp8 ? 1 : 0][mt][nt], af, b[nt][0], b[nt][1]);
                    }
                }
            }
        }
        if constexpr (kFp8) {
            // A 128-wide block is four 32-column stages. Fold its scaled sum into the total in
            // ascending block order, then start the next block from zero.
            if ((kb & 3u) == 3u || kb + 1u == k_blocks) {
#pragma unroll
                for (uint32_t mt = 0; mt < 4u; ++mt)
#pragma unroll
                    for (uint32_t nt = 0; nt < 4u; ++nt)
#pragma unroll
                        for (uint32_t e = 0; e < 4u; ++e) {
                            accum[0][mt][nt][e] += accum[1][mt][nt][e] * stage_scale;
                            accum[1][mt][nt][e] = 0.0f;
                        }
            }
        }
        __syncthreads();
    }

    if (!active) return;
    const uint32_t g = lane >> 2;
    const uint32_t q = lane & 3u;
#pragma unroll
    for (uint32_t mt = 0; mt < 4u; ++mt) {
#pragma unroll
        for (uint32_t half = 0; half < 2u; ++half) {
            const uint32_t token = m0 + wm * 64u + mt * 16u + g + 8u * half;
            if (token >= tokens) continue;
            uint64_t row = group * out_grp_stride + (uint64_t)token * out_tok_stride;
            float weight = 1.0f;
            bool scaled = false;
            if (kRoutedMoe && route != nullptr) {
                const int slot = item_slots[token];
                if (slot < 0) continue;
                row = (uint64_t)slot * out_tok_stride;
                // GLM holds one combine weight a slot; Qwen's probabilities are
                // [batch][experts] with this slot's top-k weight in row slot / topk.
                weight = route_stride == 0
                             ? route[slot]
                             : route[(uint64_t)(slot / (int)topk) * route_stride +
                                     (uint32_t)(slot % (int)topk)];
                scaled = true;
            } else if (kRoutedMoe && (kSlotRows || kOutSlotRows)) {
                const int slot = item_slots[token];
                if (slot < 0) continue;
                row = (uint64_t)slot * out_tok_stride;
            }
            const uint64_t base = row + n0 + wn * 32u + 2u * q;
#pragma unroll
            for (uint32_t nt = 0; nt < 4u; ++nt) {
                const uint64_t o = base + nt * 8u;
                float2 v = make_float2(accum[0][mt][nt][2u * half],
                                       accum[0][mt][nt][2u * half + 1u]);
                if (scaled) v = make_float2(__fmul_rn(v.x, weight), __fmul_rn(v.y, weight));
                if constexpr (kOutWide) {
                    v = make_float2(__fmul_rn(v.x, wide), __fmul_rn(v.y, wide));
                }
                *reinterpret_cast<float2 *>(out + o) = v;
            }
        }
    }
}


// ---------------------------------------------------------------------------------------------
// The few-row (token) form of the tile.
// ---------------------------------------------------------------------------------------------

// The most token rows a few-row launch carries, and the bytes one warp stages one 32-column K
// block of its eight weight rows into: eight rows of 32 bf16 words (64 bytes) plus 16, so the
// eight row addresses of an ldmatrix phase start four words apart and cover every bank once.
#define SPM_ROWS_MAX 8u
#define SPM_ROWS_STRIDE 80u

// `stage` writes the 32-column K block `kb` of the warp's eight weight rows into `tile` - row
// `threadIdx.x >> 2` and part `threadIdx.x & 3`, eight bf16 words a part - and returns the scale
// that block's 128-column K block was stored with (unread when kScaled is false).
//
// The chain is the prompt tile's: for each 16-column step of the block, the activation's bf16
// head MMA and then the bf16 of what the head leaves, K ascending, into one f32 accumulator. With
// kScaled the 128-column fold is the prompt body's too: the four stages of a block accumulate
// into a separate sum, which is multiplied by that block's scale and added to the total in
// ascending block order, the last and short block included, as shared_prefill_mma_body folds it.
template <bool kScaled, class Stage>
__device__ __forceinline__ void shared_prefill_mma_rows_chain(
    Stage &stage, const float *__restrict__ a, uint32_t inner, uint32_t tokens, bool live,
    uint32_t n0, uint32_t k_blocks, unsigned char *tile, float acc[4]) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t g = lane >> 2;
    const uint32_t q = lane & 3u;
    const bool low_row = g < tokens;
    const bool high_row = g + 8u < tokens;
    const float *a_low = a + (unsigned long long)g * inner + 2u * q;
    const float *a_high = a_low + 8ull * inner;
    // ldmatrix's row addresses: lane l points its matrix l / 8 - the 16-byte part (l / 8) of a
    // row - at staged row l % 8, the layout q4p_rows_tile fills.
    const uint32_t parts = om_smem(tile + (lane & 7u) * SPM_ROWS_STRIDE + 16u * (lane >> 3));
    float part[4];
#pragma unroll
    for (uint32_t r = 0; r < 4u; ++r) {
        acc[r] = 0.0f;
        part[r] = 0.0f;
    }
    for (uint32_t kb = 0; kb < k_blocks; ++kb) {
        __syncwarp();
        const float scale = stage(kb, n0, tile, live);
        (void)scale;
        __syncwarp();
        uint32_t b[4];
        om_ldsm4(parts, b[0], b[1], b[2], b[3]);
#pragma unroll
        for (uint32_t ks = 0; ks < 2u; ++ks) {
            const uint32_t k = kb * 32u + ks * 16u;
            const float2 zero = make_float2(0.0f, 0.0f);
            const float2 v[4] = {
                low_row ? __ldg(reinterpret_cast<const float2 *>(a_low + k)) : zero,
                high_row ? __ldg(reinterpret_cast<const float2 *>(a_high + k)) : zero,
                low_row ? __ldg(reinterpret_cast<const float2 *>(a_low + k + 8u)) : zero,
                high_row ? __ldg(reinterpret_cast<const float2 *>(a_high + k + 8u)) : zero,
            };
            uint32_t head[4];
            uint32_t rest[4];
#pragma unroll
            for (uint32_t r = 0; r < 4u; ++r) {
                const __nv_bfloat16 h0 = __float2bfloat16_rn(v[r].x);
                const __nv_bfloat16 h1 = __float2bfloat16_rn(v[r].y);
                head[r] = ob_pack(h0, h1);
                rest[r] = ob_pack(__float2bfloat16_rn(v[r].x - __bfloat162float(h0)),
                                  __float2bfloat16_rn(v[r].y - __bfloat162float(h1)));
            }
            float *dst = kScaled ? part : acc;
            ob_mma(dst, head, b[2u * ks], b[2u * ks + 1u]);
            ob_mma(dst, rest, b[2u * ks], b[2u * ks + 1u]);
        }
        if constexpr (kScaled) {
            if ((kb & 3u) == 3u || kb + 1u == k_blocks) {
#pragma unroll
                for (uint32_t r = 0; r < 4u; ++r) {
                    acc[r] += part[r] * scale;
                    part[r] = 0.0f;
                }
            }
        }
    }
}

// One CTA of the few-row grid: four warps, each an m16 x n8 tile of up to [SPM_ROWS_MAX] token
// rows by eight output columns, over the 32-column K blocks of `inner`. The wrapper supplies the
// staging (a BF16 row, an E4M3 row with its block scale, a packed row) and the output is one
// f32 per row and column, the tile's own bits for the same row.
template <bool kScaled, class Stage>
__device__ __forceinline__ void shared_prefill_mma_rows_body(
    Stage &stage, const float *__restrict__ a, float *__restrict__ out, uint32_t cols,
    uint32_t inner, uint32_t tokens, float out_scale = 1.0f) {
    __shared__ __align__(16) unsigned char tile[4][8u * SPM_ROWS_STRIDE];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    if (warp >= 4u) return;
    const uint32_t g = lane >> 2;
    const uint32_t q = lane & 3u;
    const uint32_t n0 = (blockIdx.x * 4u + warp) * 8u;
    const bool live = n0 < cols;
    float acc[4];
    shared_prefill_mma_rows_chain<kScaled>(stage, a, inner, min(tokens, SPM_ROWS_MAX), live, n0,
                                            inner >> 5, tile[warp], acc);
    if (!live) return;
#pragma unroll
    for (uint32_t half = 0; half < 2u; ++half) {
        const uint32_t token = g + 8u * half;
        if (token >= tokens) continue;
        *reinterpret_cast<float2 *>(out + (unsigned long long)token * cols + n0 + 2u * q) =
            make_float2(__fmul_rn(acc[2u * half], out_scale),
                        __fmul_rn(acc[2u * half + 1u], out_scale));
    }
}

// A BF16 weight row's 32-column K block, staged as it stands.
struct SpmRowsBf16 {
    const __nv_bfloat16 *w;
    uint32_t inner;
    __device__ __forceinline__ float operator()(uint32_t kb, uint32_t n0, unsigned char *tile,
                                                bool live) const {
        const uint32_t lane = threadIdx.x & 31u;
        const uint32_t r = lane >> 2;
        const uint32_t part = lane & 3u;
        const uint4 value =
            live ? __ldg(reinterpret_cast<const uint4 *>(
                       w + (unsigned long long)(n0 + r) * inner + kb * 32u + 8u * part))
                 : make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4 *>(tile + r * SPM_ROWS_STRIDE + 16u * part) = value;
        return 1.0f;
    }
};

// An E4M3 weight row's 32-column K block, widened to the bf16 words the prompt tile stages
// (ob_e4m3x2 at unit scale: the scale is the 128-column block's, applied after accumulation),
// and that block's scale as the release stores it - GLM's f32, Qwen's bf16.
template <typename ScaleT>
struct SpmRowsFp8 {
    const uint8_t *w;
    const ScaleT *ws;
    uint32_t inner;
    __device__ __forceinline__ float operator()(uint32_t kb, uint32_t n0, unsigned char *tile,
                                                bool live) const {
        const uint32_t lane = threadIdx.x & 31u;
        const uint32_t r = lane >> 2;
        const uint32_t part = lane & 3u;
        uint2 codes = make_uint2(0u, 0u);
        if (live) {
            codes = __ldg(reinterpret_cast<const uint2 *>(
                w + (unsigned long long)(n0 + r) * inner + kb * 32u + 8u * part));
        }
        const uint4 value =
            make_uint4(ob_e4m3x2(codes.x, 1.0f), ob_e4m3x2(codes.x >> 16, 1.0f),
                       ob_e4m3x2(codes.y, 1.0f), ob_e4m3x2(codes.y >> 16, 1.0f));
        *reinterpret_cast<uint4 *>(tile + r * SPM_ROWS_STRIDE + 16u * part) = value;
        // One block scale a 128 x 128 tile, [cols / 128][inner / 128]; a warp's eight columns
        // lie in one 128-column block.
        return live ? spm_scale<ScaleT>(__ldg(&ws[(n0 >> 7) * (inner >> 7) + (kb >> 2)])) : 0.0f;
    }
};


// An E2M1 weight row's 32-column K block, dequantised by its 16-value E4M3 group scale into the
// bf16 words the prompt tile stages. A lane's four bytes are eight codes and lie in one group:
// parts 0 and 1 are the stage's low 16 values, parts 2 and 3 its high 16.
struct SpmRowsFp4 {
    const uint8_t *w;
    const uint8_t *ws;
    uint32_t inner;
    float fold;
    bool fold_wide;
    __device__ __forceinline__ float operator()(uint32_t kb, uint32_t n0, unsigned char *tile,
                                                bool live) const {
        const uint32_t lane = threadIdx.x & 31u;
        const uint32_t r = lane >> 2;
        const uint32_t part = lane & 3u;
        uint32_t codes = 0u;
        if (live) {
            codes = __ldg(reinterpret_cast<const uint32_t *>(
                w + (unsigned long long)(n0 + r) * (inner / 2u) + kb * 16u + part * 4u));
        }
        const uint32_t row = n0 + r;
        const uint32_t group = kb * 2u + (part >> 1);
        float scale = live ? spark_fp8_e4m3_decode(ws[row * (inner / 16u) + group]) : 0.0f;
        if (fold_wide) scale *= fold;
        __nv_bfloat16 *dst = reinterpret_cast<__nv_bfloat16 *>(tile + r * SPM_ROWS_STRIDE +
                                                              16u * part);
#pragma unroll
        for (uint32_t byte = 0; byte < 4u; ++byte) {
            const uint8_t code = (uint8_t)(codes >> (8u * byte));
            dst[2u * byte] = __float2bfloat16_rn(spark_fp4_e2m1_decode(code & 0x0Fu) * scale);
            dst[2u * byte + 1u] =
                __float2bfloat16_rn(spark_fp4_e2m1_decode(code >> 4) * scale);
        }
        return 1.0f;
    }
};

// A bf16x2 of two f32 (lo in the low half), each rounded as __float2bfloat16_rn rounds it, and
// the two halves of one back as f32.
__device__ __forceinline__ uint32_t glm_bf16x2(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&v);
}
__device__ __forceinline__ float glm_bf16_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float glm_bf16_hi(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }


// The dense BF16-weight projection over a wider tile: one block 128 token rows x GLM_WIDE_BN
// output columns (eight warps of 64 x 64), the f32 activation and BF16 weight tiles through
// GLM_WIDE_STAGES cp.async stages. shared_prefill_mma_body's 128 x 128 tile reads each f32
// activation tile once a 128 columns, and on GLM's KDA projections that traffic, not the tensor
// cores, sets its time; read once a 256 columns, the 6144 x 4096 projection over 4,096 rows takes
// 4.4 ms instead of 5.5.
//
// The arithmetic is shared_prefill_mma_body's for a dense BF16 weight, element for element: a
// fragment splits each f32 into the bf16 head and the bf16 of what the head leaves, as that body
// splits it when staging the tile, and every output's chain is the same MMA sequence, each 16-K
// step's head and then (kLow) its remainder, ascending K, so the outputs are its bits. The grid
// is (cols / GLM_WIDE_BN, rows / 128); cols and inner are multiples of 32.
//
// shared_prefill_wide_tile is one block's tile at output column n0 and token row m0, so a caller
// whose grid covers several matrices (dory_mma_wide4) places each block itself; an output's
// chain does not depend on which grid computed it.
template <bool kLow>
__device__ __forceinline__ void shared_prefill_wide_tile(const __nv_bfloat16 *__restrict__ w,
                                                         const float *__restrict__ a,
                                                         float *__restrict__ out, uint32_t cols,
                                                         uint32_t inner, uint32_t tokens,
                                                         uint32_t n0, uint32_t m0) {
    extern __shared__ __align__(128) uint8_t glm_wide_shared[];
    auto s_a = [&](uint32_t buf) { return glm_wide_shared + buf * GLM_WIDE_STAGE_BYTES; };
    auto s_w = [&](uint32_t buf) { return s_a(buf) + GLM_BM * 128u; };

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t stages = inner >> 5;
    const uint32_t col_tiles = min(GLM_WIDE_BN / 32u, (cols - n0) >> 5);

    // A stage's tiles: eight 16-byte chunks of each activation row (zero past the tokens) and
    // four of each weight row (zero past the columns).
    auto issue = [&](uint32_t stage, uint32_t buf) {
        const uint32_t k0 = stage * 32u;
#pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) {
            const uint32_t idx = tid + i * OM_THREADS;
            const uint32_t row = idx >> 3;
            const uint32_t chunk = idx & 7u;
            const bool valid = m0 + row < tokens;
            const float *src = valid ? a + (uint64_t)(m0 + row) * inner + k0 + chunk * 4u : a;
            om_cp16(om_smem(s_a(buf) + glm_wide_a_off(row, chunk)), src, valid);
        }
#pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) {
            const uint32_t idx = tid + i * OM_THREADS;
            const uint32_t row = idx >> 2;
            const uint32_t chunk = idx & 3u;
            const bool valid = (row >> 5) < col_tiles;
            const __nv_bfloat16 *src =
                valid ? w + (uint64_t)(n0 + row) * inner + k0 + chunk * 8u : w;
            om_cp16(om_smem(s_w(buf) + glm_q8_off(row, chunk)), src, valid);
        }
    };

    const uint32_t wm = warp & 1u;
    const uint32_t wn = warp >> 1;
    // A warp's 64 columns are two 32-column tiles, live while the first is; its 16-row tiles
    // that hold a token, the rest zero rows that skip their MMAs.
    const bool active = 2u * wn < col_tiles && m0 + wm * 64u < tokens;
    const uint32_t nt_live = active ? min(8u, (col_tiles - 2u * wn) * 4u) : 0u;
    const uint32_t mt_live = active ? min(4u, (tokens - m0 - wm * 64u + 15u) >> 4) : 0u;
    const uint32_t g = lane >> 2;
    const uint32_t q = lane & 3u;

    float acc[4][8][4];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i)
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j)
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

#pragma unroll
    for (uint32_t stage = 0; stage + 1u < GLM_WIDE_STAGES; ++stage) {
        if (stage < stages) issue(stage, stage);
        om_commit();
    }
    for (uint32_t stage = 0; stage < stages; ++stage) {
        // This stage's group is in once at most the stages after it are still in flight; the
        // barrier also ends every warp's reads of the buffer the next issue overwrites.
        om_wait<GLM_WIDE_STAGES - 2u>();
        __syncthreads();
        const uint32_t next = stage + GLM_WIDE_STAGES - 1u;
        if (next < stages) issue(next, next % GLM_WIDE_STAGES);
        om_commit();
        if (!active) continue;
        const uint8_t *tile_a = s_a(stage % GLM_WIDE_STAGES);
        const uint8_t *tile_w = s_w(stage % GLM_WIDE_STAGES);
        // Two adjacent f32 of row `row` from K offset `k` of the stage.
        auto pair = [&](uint32_t row, uint32_t k) {
            return *reinterpret_cast<const float2 *>(tile_a + glm_wide_a_off(row, k >> 2) +
                                                     (k & 3u) * 4u);
        };
#pragma unroll
        for (uint32_t step = 0; step < 2u; ++step) {
            const uint32_t k0 = step * 16u;
            uint32_t b[8][2];
#pragma unroll
            for (uint32_t p = 0; p < 4u; ++p) {
                const uint32_t n = wn * 64u + p * 16u + (lane & 7u) + 8u * (lane >> 4);
                const uint32_t chunk = (k0 >> 3) + ((lane >> 3) & 1u);
                om_ldsm4(om_smem(tile_w + glm_q8_off(n, chunk)), b[2 * p][0], b[2 * p][1],
                         b[2 * p + 1][0], b[2 * p + 1][1]);
            }
#pragma unroll
            for (uint32_t mt = 0; mt < 4u; ++mt) {
                if (mt >= mt_live) break;
                const uint32_t r = wm * 64u + mt * 16u + g;
                const float2 x[4] = {pair(r, k0 + 2u * q), pair(r + 8u, k0 + 2u * q),
                                     pair(r, k0 + 8u + 2u * q), pair(r + 8u, k0 + 8u + 2u * q)};
                uint32_t head[4];
#pragma unroll
                for (uint32_t i = 0; i < 4u; ++i) head[i] = glm_bf16x2(x[i].x, x[i].y);
#pragma unroll
                for (uint32_t nt = 0; nt < 8u; ++nt)
                    if (nt < nt_live) ob_mma(acc[mt][nt], head, b[nt][0], b[nt][1]);
                if constexpr (kLow) {
                    uint32_t low[4];
#pragma unroll
                    for (uint32_t i = 0; i < 4u; ++i)
                        low[i] = glm_bf16x2(x[i].x - glm_bf16_lo(head[i]),
                                            x[i].y - glm_bf16_hi(head[i]));
#pragma unroll
                    for (uint32_t nt = 0; nt < 8u; ++nt)
                        if (nt < nt_live) ob_mma(acc[mt][nt], low, b[nt][0], b[nt][1]);
                }
            }
        }
    }

    if (!active) return;
#pragma unroll
    for (uint32_t mt = 0; mt < 4u; ++mt) {
#pragma unroll
        for (uint32_t half = 0; half < 2u; ++half) {
            const uint32_t token = m0 + wm * 64u + mt * 16u + g + 8u * half;
            if (token >= tokens) continue;
            const uint64_t base = (uint64_t)token * cols + n0 + wn * 64u + 2u * q;
#pragma unroll
            for (uint32_t nt = 0; nt < 8u; ++nt)
                if (nt < nt_live)
                    *reinterpret_cast<float2 *>(out + base + nt * 8u) =
                        make_float2(acc[mt][nt][2u * half], acc[mt][nt][2u * half + 1u]);
        }
    }
}

template <bool kLow>
__device__ __forceinline__ void shared_prefill_wide_body(const __nv_bfloat16 *__restrict__ w,
                                                         const float *__restrict__ a,
                                                         float *__restrict__ out, uint32_t cols,
                                                         uint32_t inner, uint32_t tokens) {
    const uint2 place = glm_raster();
    shared_prefill_wide_tile<kLow>(w, a, out, cols, inner, tokens, place.x * GLM_WIDE_BN,
                                   place.y * GLM_BM);
}


// The FP8 E4M3 body's 128 x 128 tile with a block scale, kept for the q8 activation path: one
// block 128 rows x 128 output columns over E4M3 activation codes, 64 K a stage through
// GLM_Q8_STAGES cp.async stages. The FP8 weight scale is f32 (GLM) or bf16 (a caller that has
// one); the activation scale is f32.
template <bool MOE, typename ScaleT = float, int kTableStride = 2, int kTableOffset = 0>
__device__ __forceinline__ void shared_prefill_q8_body(
    const uint8_t *__restrict__ a, const float *__restrict__ as, const uint8_t *__restrict__ w,
    const void *__restrict__ ws, const unsigned long long *__restrict__ table,
    const int4 *__restrict__ items, float *__restrict__ out, uint32_t cols, uint32_t inner,
    uint32_t tokens, const int *__restrict__ sorted, uint32_t topk,
    const float *__restrict__ route) {
    extern __shared__ __align__(128) uint8_t glm_q8_shared[];
    auto s_a = [&](uint32_t buf) { return glm_q8_shared + buf * GLM_Q8_STAGE_BYTES; };
    auto s_w = [&](uint32_t buf) { return s_a(buf) + GLM_BM * 64u; };
    auto s_as = [&](uint32_t buf) {
        return reinterpret_cast<float *>(s_w(buf) + GLM_BN * 64u);
    };
    int *s_row = reinterpret_cast<int *>(glm_q8_shared + GLM_Q8_STAGES * GLM_Q8_STAGE_BYTES);
    float *s_ws = reinterpret_cast<float *>(s_row + GLM_BM);
    const ScaleT *ws_typed = reinterpret_cast<const ScaleT *>(ws);

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint2 place = MOE ? make_uint2(blockIdx.x, blockIdx.y) : glm_raster();
    const uint32_t n0 = place.x * GLM_BN;
    const uint32_t m0 = place.y * GLM_BM;
    const uint32_t group = blockIdx.z;
    const uint32_t col_tiles = min(GLM_BN / 32u, (cols - n0) >> 5);
    const uint32_t stages = inner >> 6;
    const uint32_t blocks = inner >> 7;
    if (blocks > GLM_Q8_BLOCKS) return;
    const int *item_slots = nullptr;
    if constexpr (MOE) {
        const int4 item = items[group];
        if (item.z <= 0) return;
        tokens = min((uint32_t)item.z, GLM_MOE_ROWS);
        w = reinterpret_cast<const uint8_t *>(table[(uint64_t)item.x * kTableStride + kTableOffset]);
        ws_typed = reinterpret_cast<const ScaleT *>(
            table[(uint64_t)item.x * kTableStride + kTableOffset + 1]);
        item_slots = sorted + item.y;
    }
    // The column tile's weight scales ride the first stage's group: a routed expert's are cold,
    // and read at each fold they would put a DRAM round trip on every other stage.
    if (tid < blocks) {
        const ScaleT sc = __ldg(&ws_typed[(uint64_t)(n0 >> 7) * blocks + tid]);
        s_ws[tid] = spm_scale<ScaleT>(sc);
    }
    // Each tile row's activation row, or -1 past the tile's tokens (loaded as zero).
    if (tid < GLM_BM) {
        const uint32_t token = m0 + tid;
        int row = -1;
        if (token < tokens) {
            if (!MOE) {
                row = (int)token;
            } else if (route == nullptr) {
                const int slot = item_slots[token];
                row = slot >= 0 ? slot / (int)topk : -1;
            } else {
                row = (int)(group * GLM_MOE_ROWS + token);
            }
        }
        s_row[tid] = row;
    }
    __syncthreads();

    // A stage's tiles, and its 128-column block's activation scales with them.
    auto issue = [&](uint32_t stage, uint32_t buf) {
        const uint32_t k0 = stage * 64u;
#pragma unroll
        for (uint32_t i = 0; i < 2u; ++i) {
            const uint32_t idx = tid + i * OM_THREADS;
            const uint32_t row = idx >> 2;
            const uint32_t chunk = idx & 3u;
            const int r = s_row[row];
            const uint8_t *src = r >= 0 ? a + (uint64_t)r * inner + k0 + chunk * 16u : a;
            om_cp16(om_smem(s_a(buf) + glm_q8_off(row, chunk)), src, r >= 0);
        }
        // Four threads a weight row's 64 bytes, so a warp asks the DRAM for whole 64-byte runs
        // of eight rows: split across two warps, a row's halves stream at half the bandwidth.
#pragma unroll
        for (uint32_t i = 0; i < 2u; ++i) {
            const uint32_t idx = tid + i * OM_THREADS;
            const uint32_t row = idx >> 2;
            const uint32_t chunk = idx & 3u;
            const bool valid = (row >> 5) < col_tiles;
            const uint8_t *src = valid ? w + (uint64_t)(n0 + row) * inner + k0 + chunk * 16u : w;
            om_cp16(om_smem(s_w(buf) + glm_q8_off(row, chunk)), src, valid);
        }
        if (tid < GLM_BM) {
            const int r = s_row[tid];
            om_cp4(om_smem(s_as(buf) + tid), r >= 0 ? as + (uint64_t)r * blocks + (stage >> 1) : as,
                   r >= 0);
        }
    };

    const uint32_t wm = warp & 1u;
    const uint32_t wn = warp >> 1;
    const bool active = wn < col_tiles && m0 + wm * 64u < tokens;
    const uint32_t mt_live = active ? min(4u, (tokens - m0 - wm * 64u + 15u) >> 4) : 0u;
    const uint32_t g = lane >> 2;
    const uint32_t q = lane & 3u;

    float acc[4][4][4];
    float part[4][4][4];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i)
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j)
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) acc[i][j][e] = part[i][j][e] = 0.0f;

#pragma unroll
    for (uint32_t stage = 0; stage + 1u < GLM_Q8_STAGES; ++stage) {
        if (stage < stages) issue(stage, stage);
        om_commit();
    }
    for (uint32_t stage = 0; stage < stages; ++stage) {
        // This stage's group is in once at most the stages after it are still in flight; the
        // barrier also ends every warp's reads of the buffer the next issue overwrites.
        om_wait<GLM_Q8_STAGES - 2u>();
        __syncthreads();
        const uint32_t next = stage + GLM_Q8_STAGES - 1u;
        if (next < stages) issue(next, next % GLM_Q8_STAGES);
        om_commit();
        const uint32_t buf = stage % GLM_Q8_STAGES;
        if (active) {
#pragma unroll
            for (uint32_t j = 0; j < 2u; ++j) {
                uint32_t b[4][2];
                const uint32_t n = wn * 32u + 8u * (lane >> 4) + (lane & 7u);
                const uint32_t chunk = 2u * j + ((lane >> 3) & 1u);
                om_ldsm4(om_smem(s_w(buf) + glm_q8_off(n, chunk)), b[0][0], b[0][1], b[1][0],
                         b[1][1]);
                om_ldsm4(om_smem(s_w(buf) + glm_q8_off(n + 16u, chunk)), b[2][0], b[2][1],
                         b[3][0], b[3][1]);
#pragma unroll
                for (uint32_t mt = 0; mt < 4u; ++mt) {
                    if (mt >= mt_live) break;
                    const uint32_t r = wm * 64u + mt * 16u + (lane & 7u) + ((lane >> 3) & 1u) * 8u;
                    uint32_t af[4];
                    om_ldsm4(om_smem(s_a(buf) + glm_q8_off(r, 2u * j + (lane >> 4))), af[0],
                             af[1], af[2], af[3]);
#pragma unroll
                    for (uint32_t nt = 0; nt < 4u; ++nt)
                        om_mma_e4m3(part[mt][nt], af, b[nt][0], b[nt][1], 127u, 127u);
                }
            }
            // A 128-column block is two stages: fold its sum into the total times the row's
            // activation scale and the block's weight scale, in ascending block order.
            if (stage & 1u) {
                const float wsc = s_ws[stage >> 1];
                const float *scales = s_as(buf);
#pragma unroll
                for (uint32_t mt = 0; mt < 4u; ++mt)
#pragma unroll
                    for (uint32_t h = 0; h < 2u; ++h) {
                        const float scale = scales[wm * 64u + mt * 16u + g + 8u * h] * wsc;
#pragma unroll
                        for (uint32_t nt = 0; nt < 4u; ++nt)
#pragma unroll
                            for (uint32_t e = 0; e < 2u; ++e) {
                                acc[mt][nt][2u * h + e] += part[mt][nt][2u * h + e] * scale;
                                part[mt][nt][2u * h + e] = 0.0f;
                            }
                    }
            }
        }
    }

    if (!active) return;
#pragma unroll
    for (uint32_t mt = 0; mt < 4u; ++mt) {
#pragma unroll
        for (uint32_t half = 0; half < 2u; ++half) {
            const uint32_t token = m0 + wm * 64u + mt * 16u + g + 8u * half;
            if (token >= tokens) continue;
            uint64_t row = (uint64_t)token * cols;
            float weight = 1.0f;
            bool scaled = false;
            if constexpr (MOE) {
                if (route != nullptr) {
                    const int slot = item_slots[token];
                    if (slot < 0) continue;
                    row = (uint64_t)slot * cols;
                    weight = route[slot];
                    scaled = true;
                } else {
                    row = ((uint64_t)group * GLM_MOE_ROWS + token) * cols;
                }
            }
            const uint64_t base = row + n0 + wn * 32u + 2u * q;
#pragma unroll
            for (uint32_t nt = 0; nt < 4u; ++nt) {
                float2 v = make_float2(acc[mt][nt][2u * half], acc[mt][nt][2u * half + 1u]);
                if (scaled) v = make_float2(__fmul_rn(v.x, weight), __fmul_rn(v.y, weight));
                *reinterpret_cast<float2 *>(out + base + nt * 8u) = v;
            }
        }
    }
}

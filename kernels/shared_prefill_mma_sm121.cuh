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
// header and instantiate the body; none carries a second staging loop.

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

// The weight formats the tile body serves.
#define SPM_W_BF16 0
#define SPM_W_FP8 1

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


// 4 bytes global -> shared, zero-filled when !valid (the source is then not read).
__device__ __forceinline__ void glm_cp4(uint32_t dst, const void *src, bool valid) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(dst), "l"(src),
                 "r"(valid ? 4 : 0));
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
          bool kOutSlotRows = false>
__device__ __forceinline__ void shared_prefill_mma_body(
    const float *__restrict__ a, const uint8_t *__restrict__ w, const void *__restrict__ ws,
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    float *__restrict__ out, uint32_t cols, uint32_t inner, uint32_t tokens,
    uint64_t a_tok_stride, uint64_t out_tok_stride, uint64_t a_grp_stride,
    uint64_t out_grp_stride, uint64_t w_grp_stride = 0, uint64_t ws_grp_stride = 0,
    const int *__restrict__ sorted = nullptr, uint32_t topk = 1,
    const float *__restrict__ route = nullptr, uint32_t route_stride = 0) {
    constexpr bool kFp8 = kFormat == SPM_W_FP8;
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
        if constexpr (kFp8) {
            ws_grp = reinterpret_cast<const ScaleT *>(
                table[(uint64_t)item.x * kTableStride + kTableOffset + 1]);
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
        if constexpr (kFp8) {
            stage_scale = rs;
            const uint4 r = rw[0];
            *reinterpret_cast<uint4 *>(dst) =
                make_uint4(ob_e4m3x2(r.x, 1.0f), ob_e4m3x2(r.x >> 16, 1.0f),
                           ob_e4m3x2(r.y, 1.0f), ob_e4m3x2(r.y >> 16, 1.0f));
            *reinterpret_cast<uint4 *>(dst + 8) =
                make_uint4(ob_e4m3x2(r.z, 1.0f), ob_e4m3x2(r.z >> 16, 1.0f),
                           ob_e4m3x2(r.w, 1.0f), ob_e4m3x2(r.w >> 16, 1.0f));
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
                *reinterpret_cast<float2 *>(out + o) = v;
            }
        }
    }
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
            glm_cp4(om_smem(s_as(buf) + tid), r >= 0 ? as + (uint64_t)r * blocks + (stage >> 1) : as,
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

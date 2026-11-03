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

// GLM-5.3-Flash's bulk prefill kernels.
//
// The decode arm reads every weight once per eight rows. A prefill chunk of 128-1024 rows can
// read each weight once instead, which is what makes GLM's 288-expert MoE affordable: at 512 rows
// every expert is routed to and its 4.2 MiB rank slice is read once for the whole chunk.
//
// The dense projection is an m16n8k16 bf16 MMA over an f32 activation split into a bf16 head and
// remainder. A block-scaled FP8 weight (GLM's E4M3 codes with one f32 scale a 128 x 128 block) is
// widened to bf16 in shared memory with its block scale applied; the scale is constant over the
// 128 x 128 block, so one scale a stage is exact. This is the b12x SM121 tile pattern the tree's
// other packs use, with GLM's block-scaled weight layout.
//
// The routed experts run in items: one expert's up-to-128 sorted slots, one 128-row tile. The
// gate/up projection reads each item row straight from its token's row (slot_of names the slot),
// SwiGLU writes the padded [items][128][width] activations, and the down projection writes each
// kept row, times its routing weight, straight to its slot row. Rows past an item's count load as
// zero and are never written, and a warp whose rows are all past the count skips its MMAs.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "opt_mma_common_sm121.cuh"
#include "opt_mma_bf16_sm121.cuh"

// The shared tile geometry of opt_mma_bf16_body: 128 rows x 128 columns, 32-column K stages.
#define GLM_BM 128u
#define GLM_BN 128u
#define GLM_A_PER 4u
#define GLM_W_TILES 4u
// A routed-expert item: up to this many of one expert's slots, one row tile.
#define GLM_MOE_ROWS 128u
// The row tiles a dense GEMM's grid walks down each column tile before the next ([glm_raster]).
#define GLM_RASTER 8u

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

// One block: 128 token rows x 128 output columns, K in 32-column blocks. w/ws name a single
// matrix (dense; group z's is w_grp_stride bytes and ws_grp_stride scales on, a head's slice of a
// per-head projection), or table names the experts' [weight, scales] pointer pairs and items the
// per-group (expert, ..., count, ...) list (MoE): group z is item z, its count the rows. An MoE
// item's rows are its GLM_MOE_ROWS slot_of entries: without route, row r reads the token row of
// slot slot_of[r] (a_tok_stride floats a token) and writes the padded row; with route, it reads
// the padded row and writes slot row slot_of[r] (out_tok_stride floats a slot) times route[slot].
// The activation is f32 split into bf16 head and remainder; the output is f32.
template <bool FP8, bool MOE, bool LOW>
__device__ __forceinline__ void glm_mma_body(
    const float *__restrict__ a, const uint8_t *__restrict__ w, const float *__restrict__ ws,
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    float *__restrict__ out, uint32_t cols, uint32_t inner, uint32_t tokens,
    uint64_t a_tok_stride, uint64_t out_tok_stride, uint64_t a_grp_stride,
    uint64_t out_grp_stride, uint64_t w_grp_stride = 0, uint64_t ws_grp_stride = 0,
    const int *__restrict__ slot_of = nullptr, uint32_t topk = 1,
    const float *__restrict__ route = nullptr) {
    __shared__ __align__(16) __nv_bfloat16 s_ah[GLM_BM * OB_LD];
    __shared__ __align__(16) __nv_bfloat16 s_al[LOW ? GLM_BM * OB_LD : 1];
    __shared__ __align__(16) __nv_bfloat16 s_w[GLM_BN * OB_LD];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint2 place = MOE ? make_uint2(blockIdx.x, blockIdx.y) : glm_raster();
    const uint32_t n0 = place.x * GLM_BN;
    const uint32_t m0 = place.y * GLM_BM;
    const uint32_t group = blockIdx.z;
    const uint32_t k_blocks = inner >> 5;
    const uint32_t col_tiles = min(GLM_BN / 32u, (cols - n0) >> 5);

    const uint8_t *w_grp = w + group * w_grp_stride;
    const float *ws_grp = ws + group * ws_grp_stride;
    const int *item_slots = nullptr;
    if constexpr (MOE) {
        const int4 item = items[group];
        if (item.z <= 0) return;
        tokens = min((uint32_t)item.z, GLM_MOE_ROWS);
        w_grp = reinterpret_cast<const uint8_t *>(table[2ull * item.x]);
        ws_grp = reinterpret_cast<const float *>(table[2ull * item.x + 1ull]);
        item_slots = slot_of + group * GLM_MOE_ROWS;
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
            if (MOE && route == nullptr) {
                const int slot = item_slots[token];
                if (slot >= 0) a_row[i] = (long long)(slot / (int)topk) * (long long)a_tok_stride;
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
    uint4 rw[FP8 ? 1u : 2u];
    float rs = 0.0f;
    // The scale the current stage's weight tile was stored with: a 128 x 128 block of GLM's
    // weights carries one f32 scale, applied to the block's f32 partial sum after the MMAs.
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
            if constexpr (FP8) {
                const uint8_t *src = w_grp + col * inner + kb * 32u + w_half * 16u;
                rw[0] = __ldg(reinterpret_cast<const uint4 *>(src));
                rs = __ldg(&ws_grp[(col >> 7) * (inner >> 7) + (kb >> 2)]);
            } else {
                const uint8_t *src = w_grp + (col * inner + kb * 32u + w_half * 16u) * 2u;
                rw[0] = __ldg(reinterpret_cast<const uint4 *>(src));
                rw[1] = __ldg(reinterpret_cast<const uint4 *>(src) + 1);
            }
        } else {
#pragma unroll
            for (uint32_t v = 0; v < (FP8 ? 1u : 2u); ++v) rw[v] = make_uint4(0u, 0u, 0u, 0u);
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
            if constexpr (LOW) {
                *reinterpret_cast<uint2 *>(&s_al[row * OB_LD + c * 4u]) =
                    make_uint2(ob_pack(l[0], l[1]), ob_pack(l[2], l[3]));
            }
        }
        const uint32_t w_at = (w_tile * 32u + w_col) * OB_LD + w_half * 16u;
        __nv_bfloat16 *dst = &s_w[w_at];
        if constexpr (FP8) {
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
    float accum[FP8 ? 2 : 1][4][4][4];
#pragma unroll
    for (uint32_t a = 0; a < (FP8 ? 2u : 1u); ++a)
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
                        ob_mma(accum[FP8 ? 1 : 0][mt][nt], af, b[nt][0], b[nt][1]);
                    if constexpr (LOW) {
                        om_ldsm4(om_smem(&s_al[row * OB_LD + k]), af[0], af[1], af[2], af[3]);
#pragma unroll
                        for (uint32_t nt = 0; nt < 4u; ++nt)
                            ob_mma(accum[FP8 ? 1 : 0][mt][nt], af, b[nt][0], b[nt][1]);
                    }
                }
            }
        }
        if constexpr (FP8) {
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
            if (MOE && route != nullptr) {
                const int slot = item_slots[token];
                if (slot < 0) continue;
                row = (uint64_t)slot * out_tok_stride;
                weight = route[slot];
                scaled = true;
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

extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_prefill_bf16_mma(
    const __nv_bfloat16 *__restrict__ w, const float *__restrict__ a,
    float *__restrict__ out, uint32_t cols, uint32_t inner, uint32_t tokens) {
    glm_mma_body<false, false, true>(a, reinterpret_cast<const uint8_t *>(w), nullptr, nullptr,
                                     nullptr, out, cols, inner, tokens, inner, cols, 0, 0);
}

// The same body over the bf16 head of each activation alone: the low half of the split costs a
// second MMA chain a stage and no weight read, so a caller that can spend the precision reads
// every weight once for one chain (GLM's own weights are FP8 with a dynamic activation scale).
extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_prefill_bf16_mma_bf16a(
    const __nv_bfloat16 *__restrict__ w, const float *__restrict__ a,
    float *__restrict__ out, uint32_t cols, uint32_t inner, uint32_t tokens) {
    glm_mma_body<false, false, false>(a, reinterpret_cast<const uint8_t *>(w), nullptr, nullptr,
                                      nullptr, out, cols, inner, tokens, inner, cols, 0, 0);
}

extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_prefill_fp8b128_mma(
    const uint8_t *__restrict__ w, const float *__restrict__ ws,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    uint32_t tokens) {
    glm_mma_body<true, false, true>(a, w, ws, nullptr, nullptr, out, cols, inner, tokens, inner,
                                    cols, 0, 0);
}

extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_prefill_fp8b128_mma_bf16a(
    const uint8_t *__restrict__ w, const float *__restrict__ ws,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    uint32_t tokens) {
    glm_mma_body<true, false, false>(a, w, ws, nullptr, nullptr, out, cols, inner, tokens, inner,
                                     cols, 0, 0);
}

// A per-head projection of the chunk: group z (blockIdx.z) is head z, whose activation is
// a_grp_stride floats into each row's a_tok_stride, whose weight is w_grp_stride bytes (and
// ws_grp_stride scales) on, and whose output is out_grp_stride floats into each row's
// out_tok_stride. The MLA's absorbed query (kv_b's key rows, transposed) and its value projection
// (kv_b's value rows). Grid (cols / 128, ceil(tokens / 128), heads).
extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_prefill_heads_bf16_mma(
    const __nv_bfloat16 *__restrict__ w, const float *__restrict__ a, float *__restrict__ out,
    uint32_t cols, uint32_t inner, uint32_t tokens, uint64_t a_tok_stride,
    uint64_t out_tok_stride, uint64_t a_grp_stride, uint64_t out_grp_stride,
    uint64_t w_grp_stride) {
    glm_mma_body<false, false, true>(a, reinterpret_cast<const uint8_t *>(w), nullptr, nullptr,
                                     nullptr, out, cols, inner, tokens, a_tok_stride,
                                     out_tok_stride, a_grp_stride, out_grp_stride, w_grp_stride, 0);
}

extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_prefill_heads_fp8b128_mma(
    const uint8_t *__restrict__ w, const float *__restrict__ ws, const float *__restrict__ a,
    float *__restrict__ out, uint32_t cols, uint32_t inner, uint32_t tokens,
    uint64_t a_tok_stride, uint64_t out_tok_stride, uint64_t a_grp_stride,
    uint64_t out_grp_stride, uint64_t w_grp_stride, uint64_t ws_grp_stride) {
    glm_mma_body<true, false, true>(a, w, ws, nullptr, nullptr, out, cols, inner, tokens,
                                    a_tok_stride, out_tok_stride, a_grp_stride, out_grp_stride,
                                    w_grp_stride, ws_grp_stride);
}

// ---------------------------------------------------------------------------------------------
// The chunk's sparse MLA in latent space - q4e_glm_mla_attend's softmax and weighted latent sum -
// with the rank's 16 heads as the M of bf16 m16n8k16 MMAs: DeepSeek's opt_bulk_attn shape
// (kernels/opt_mma_sm121.cu) over GLM's cache. The decode kernel runs a block a head and a row on
// the CUDA cores, so every head re-reads a row's ~2,300 selected latents; here a row's 16 heads
// read each of them once, from L2, and the products run on the tensor cores.
//
// One block a row (position `base` + blockIdx.x of the chunk's rows), 256 threads. The row's
// selected positions go 32 at a time into shared memory, their E4M3 codes widened to bf16 (exact)
// and their four 128-column block scales beside them: a score is the sum over the four blocks of
// scale x (q . codes), and a probability is multiplied by a block's scale as P x V reads the
// block's columns, so no product rounds a scaled code. With LOW the query and the scaled
// probabilities are each a bf16 head and remainder (the dense kernels' split), two MMAs where one
// would do. An online softmax over the tiles; the score divides by `score_div` (sqrt(nope)) as the
// decode kernel's does. The output is the normalised weighted latent, f32 [rows][16][512], written
// over the row's own query (`q_abs`, read whole before the first tile), which
// glm_prefill_heads_*_mma then projects by kv_b's value rows. A row with no selection is zero, as
// the decode kernel leaves it; one past GMA_MAX_SELECTED is NaN rather than wrong.
//
// Not bit-identical to q4e_glm_mla_attend: the sums run in MMA order.
// ---------------------------------------------------------------------------------------------
#define GMA_HEADS 16u
#define GMA_LATENT 512u
#define GMA_TILE 32u
#define GMA_BLOCKS 4u
#define GMA_MAX_SELECTED 2304u
#define GMA_P_LD 40u

__device__ __forceinline__ uint32_t gma_k_off(uint32_t key, uint32_t chunk) {
    return om_swz_off(key, chunk, GMA_LATENT * 2u);
}

// An f32 pair as a bf16x2 head and the bf16x2 of what the head leaves.
__device__ __forceinline__ void gma_split(float x, float y, uint32_t &hi, uint32_t &lo) {
    const __nv_bfloat16 hx = __float2bfloat16_rn(x), hy = __float2bfloat16_rn(y);
    hi = ob_pack(hx, hy);
    lo = ob_pack(__float2bfloat16_rn(x - __bfloat162float(hx)),
                 __float2bfloat16_rn(y - __bfloat162float(hy)));
}

template <bool LOW>
__device__ __forceinline__ void glm_mla_attend_body(
    float *q_abs, const uint8_t *__restrict__ codes, const float *__restrict__ scales,
    const int *__restrict__ selected, const int *__restrict__ selected_count, uint32_t base,
    uint32_t select_stride, float score_div) {
    __shared__ __align__(128) uint8_t s_k[GMA_TILE * GMA_LATENT * 2u];
    __shared__ __align__(8) float s_s[GMA_BLOCKS][GMA_HEADS][GMA_TILE];
    __shared__ __align__(8) float s_p[GMA_HEADS * GMA_P_LD];
    __shared__ float s_scale[GMA_TILE][GMA_BLOCKS];
    __shared__ uint32_t s_valid[GMA_TILE];
    __shared__ float s_m[GMA_HEADS], s_l[GMA_HEADS], s_alpha[GMA_HEADS];

    const uint32_t row = base + blockIdx.x;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, q = lane & 3u;
    float *io = q_abs + (uint64_t)row * GMA_HEADS * GMA_LATENT;
    const int *sel = selected + (uint64_t)row * select_stride;
    const int count = selected_count[row];
    if (count <= 0 || count > (int)GMA_MAX_SELECTED) {
        const float fill = count <= 0 ? 0.0f : __uint_as_float(0x7FC00000u);
        __syncthreads();
        for (uint32_t i = tid; i < GMA_HEADS * GMA_LATENT; i += blockDim.x) io[i] = fill;
        return;
    }
    const uint32_t nsel = (uint32_t)count;
    const uint32_t tiles = (nsel + GMA_TILE - 1u) / GMA_TILE;

    // S = Q K^T: warp w scores keys 16 (w % 2) .. + 15 over the latent's block w / 2.
    const uint32_t kq = warp & 1u, kd = warp >> 1;
    uint32_t qh[8][4], ql[LOW ? 8 : 1][4];
#pragma unroll
    for (uint32_t j = 0; j < 8u; ++j) {
        const uint32_t k0 = kd * 128u + j * 16u + 2u * q;
        const float2 v0 = *reinterpret_cast<const float2 *>(io + g * GMA_LATENT + k0);
        const float2 v1 = *reinterpret_cast<const float2 *>(io + (g + 8u) * GMA_LATENT + k0);
        const float2 v2 = *reinterpret_cast<const float2 *>(io + g * GMA_LATENT + k0 + 8u);
        const float2 v3 = *reinterpret_cast<const float2 *>(io + (g + 8u) * GMA_LATENT + k0 + 8u);
        uint32_t lo[4];
        gma_split(v0.x, v0.y, qh[j][0], lo[0]);
        gma_split(v1.x, v1.y, qh[j][1], lo[1]);
        gma_split(v2.x, v2.y, qh[j][2], lo[2]);
        gma_split(v3.x, v3.y, qh[j][3], lo[3]);
        if constexpr (LOW) {
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) ql[j][e] = lo[e];
        }
    }
    // O: warp w holds latent columns 64 w .. + 63 (block w / 2) of the 16 heads.
    const uint32_t vd = warp >> 1;
    float acc[8][4];
#pragma unroll
    for (uint32_t nt = 0; nt < 8u; ++nt)
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) acc[nt][e] = 0.0f;
    if (tid < GMA_HEADS) {
        s_m[tid] = -3.402823466e38f;
        s_l[tid] = 0.0f;
    }

    // The lane's 16 codes of keys warp + 8 i, a tile ahead of the MMAs that read them.
    uint4 raw[4];
    float raw_scale[4];
    auto fetch = [&](uint32_t tile) {
#pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) {
            const uint32_t slot = tile * GMA_TILE + warp + 8u * i;
            raw[i] = make_uint4(0u, 0u, 0u, 0u);
            raw_scale[i] = 0.0f;
            if (slot < nsel) {
                const uint64_t at = (uint64_t)(uint32_t)__ldg(sel + slot);
                raw[i] = __ldg(reinterpret_cast<const uint4 *>(codes + at * GMA_LATENT) + lane);
                if (lane < GMA_BLOCKS) raw_scale[i] = __ldg(scales + at * GMA_BLOCKS + lane);
            }
        }
    };
    fetch(0);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
#pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) {
            const uint32_t key = warp + 8u * i;
            const uint4 r = raw[i];
            *reinterpret_cast<uint4 *>(&s_k[gma_k_off(key, 2u * lane)]) =
                make_uint4(ob_e4m3x2(r.x, 1.0f), ob_e4m3x2(r.x >> 16, 1.0f),
                           ob_e4m3x2(r.y, 1.0f), ob_e4m3x2(r.y >> 16, 1.0f));
            *reinterpret_cast<uint4 *>(&s_k[gma_k_off(key, 2u * lane + 1u)]) =
                make_uint4(ob_e4m3x2(r.z, 1.0f), ob_e4m3x2(r.z >> 16, 1.0f),
                           ob_e4m3x2(r.w, 1.0f), ob_e4m3x2(r.w >> 16, 1.0f));
            if (lane < GMA_BLOCKS) s_scale[key][lane] = raw_scale[i];
            if (lane == 0u) s_valid[key] = tile * GMA_TILE + key < nsel;
        }
        __syncthreads();
        if (tile + 1u < tiles) fetch(tile + 1u);
        {
            float c[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
            for (uint32_t j2 = 0; j2 < 4u; ++j2) {
                const uint32_t chunk = kd * 16u + j2 * 4u + (lane >> 3);
#pragma unroll
                for (uint32_t t = 0; t < 2u; ++t) {
                    const uint32_t key = kq * 16u + t * 8u + (lane & 7u);
                    uint32_t b0, b1, b2, b3;
                    om_ldsm4(om_smem(&s_k[gma_k_off(key, chunk)]), b0, b1, b2, b3);
                    ob_mma(c[t], qh[2u * j2], b0, b1);
                    ob_mma(c[t], qh[2u * j2 + 1u], b2, b3);
                    if constexpr (LOW) {
                        ob_mma(c[t], ql[2u * j2], b0, b1);
                        ob_mma(c[t], ql[2u * j2 + 1u], b2, b3);
                    }
                }
            }
#pragma unroll
            for (uint32_t t = 0; t < 2u; ++t) {
                const uint32_t col = kq * 16u + t * 8u + 2u * q;
                const float sc0 = s_scale[col][kd], sc1 = s_scale[col + 1u][kd];
                *reinterpret_cast<float2 *>(&s_s[kd][g][col]) =
                    make_float2(c[t][0] * sc0, c[t][1] * sc1);
                *reinterpret_cast<float2 *>(&s_s[kd][g + 8u][col]) =
                    make_float2(c[t][2] * sc0, c[t][3] * sc1);
            }
        }
        __syncthreads();
        // Online softmax: warp w takes heads 2 w (lanes 0-15) and 2 w + 1 (lanes 16-31), two keys
        // a lane, so one 16-lane butterfly reduces both heads.
        {
            const uint32_t h = 2u * warp + (lane >> 4), k2 = 2u * (lane & 15u);
            const bool v0 = s_valid[k2] != 0u, v1 = s_valid[k2 + 1u] != 0u;
            const float2 a0 = *reinterpret_cast<const float2 *>(&s_s[0][h][k2]);
            const float2 a1 = *reinterpret_cast<const float2 *>(&s_s[1][h][k2]);
            const float2 a2 = *reinterpret_cast<const float2 *>(&s_s[2][h][k2]);
            const float2 a3 = *reinterpret_cast<const float2 *>(&s_s[3][h][k2]);
            const float score0 =
                v0 ? ((a0.x + a1.x) + (a2.x + a3.x)) / score_div : -3.402823466e38f;
            const float score1 =
                v1 ? ((a0.y + a1.y) + (a2.y + a3.y)) / score_div : -3.402823466e38f;
            float tile_max = fmaxf(score0, score1);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1)
                tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, o));
            const float m_old = s_m[h];
            const float m_new = fmaxf(m_old, tile_max);
            const float p0 = v0 ? __expf(score0 - m_new) : 0.0f;
            const float p1 = v1 ? __expf(score1 - m_new) : 0.0f;
            *reinterpret_cast<float2 *>(&s_p[h * GMA_P_LD + k2]) = make_float2(p0, p1);
            float sum = p0 + p1;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            if ((lane & 15u) == 0u) {
                const float alpha = __expf(m_old - m_new);
                s_alpha[h] = alpha;
                s_l[h] = s_l[h] * alpha + sum;
                s_m[h] = m_new;
            }
        }
        __syncthreads();
        // O = O alpha + P V over the tile's 32 keys, each probability times its key's scale for
        // the warp's block.
        {
            const float a_lo = s_alpha[g], a_hi = s_alpha[g + 8u];
#pragma unroll
            for (uint32_t nt = 0; nt < 8u; ++nt) {
                acc[nt][0] *= a_lo;
                acc[nt][1] *= a_lo;
                acc[nt][2] *= a_hi;
                acc[nt][3] *= a_hi;
            }
#pragma unroll
            for (uint32_t ks = 0; ks < 2u; ++ks) {
                const uint32_t k0 = ks * 16u + 2u * q;
                const float s0 = s_scale[k0][vd], s1 = s_scale[k0 + 1u][vd];
                const float s8 = s_scale[k0 + 8u][vd], s9 = s_scale[k0 + 9u][vd];
                const float2 p00 = *reinterpret_cast<const float2 *>(&s_p[g * GMA_P_LD + k0]);
                const float2 p10 = *reinterpret_cast<const float2 *>(&s_p[(g + 8u) * GMA_P_LD + k0]);
                const float2 p08 = *reinterpret_cast<const float2 *>(&s_p[g * GMA_P_LD + k0 + 8u]);
                const float2 p18 =
                    *reinterpret_cast<const float2 *>(&s_p[(g + 8u) * GMA_P_LD + k0 + 8u]);
                uint32_t ph[4], pl[4];
                gma_split(p00.x * s0, p00.y * s1, ph[0], pl[0]);
                gma_split(p10.x * s0, p10.y * s1, ph[1], pl[1]);
                gma_split(p08.x * s8, p08.y * s9, ph[2], pl[2]);
                gma_split(p18.x * s8, p18.y * s9, ph[3], pl[3]);
                const uint32_t key = ks * 16u + (lane & 7u) + 8u * ((lane >> 3) & 1u);
#pragma unroll
                for (uint32_t pair = 0; pair < 4u; ++pair) {
                    const uint32_t chunk = warp * 8u + pair * 2u + (lane >> 4);
                    uint32_t b0, b1, b2, b3;
                    om_ldsm4_t(om_smem(&s_k[gma_k_off(key, chunk)]), b0, b1, b2, b3);
                    ob_mma(acc[2u * pair], ph, b0, b1);
                    ob_mma(acc[2u * pair + 1u], ph, b2, b3);
                    if constexpr (LOW) {
                        ob_mma(acc[2u * pair], pl, b0, b1);
                        ob_mma(acc[2u * pair + 1u], pl, b2, b3);
                    }
                }
            }
        }
        __syncthreads();
    }

    const float inv_lo = 1.0f / (s_l[g] > 0.0f ? s_l[g] : 1.0f);
    const float inv_hi = 1.0f / (s_l[g + 8u] > 0.0f ? s_l[g + 8u] : 1.0f);
#pragma unroll
    for (uint32_t nt = 0; nt < 8u; ++nt) {
        const uint32_t d = warp * 64u + nt * 8u + 2u * q;
        *reinterpret_cast<float2 *>(io + g * GMA_LATENT + d) =
            make_float2(acc[nt][0] * inv_lo, acc[nt][1] * inv_lo);
        *reinterpret_cast<float2 *>(io + (g + 8u) * GMA_LATENT + d) =
            make_float2(acc[nt][2] * inv_hi, acc[nt][3] * inv_hi);
    }
}

// Grid (rows of the run), 256 threads; see glm_mla_attend_body.
extern "C" __global__ __launch_bounds__(256) void glm_prefill_mla_attend(
    float *q_abs, const uint8_t *__restrict__ codes, const float *__restrict__ scales,
    const int *__restrict__ selected, const int *__restrict__ selected_count, uint32_t base,
    uint32_t select_stride, float score_div) {
    glm_mla_attend_body<true>(q_abs, codes, scales, selected, selected_count, base,
                              select_stride, score_div);
}

// The same over the bf16 head of the query and the probabilities alone
// (SPARKFLASH_GLM_PREFILL_ACTIVATION=bf16): one MMA chain where the split runs two.
extern "C" __global__ __launch_bounds__(256) void glm_prefill_mla_attend_bf16a(
    float *q_abs, const uint8_t *__restrict__ codes, const float *__restrict__ scales,
    const int *__restrict__ selected, const int *__restrict__ selected_count, uint32_t base,
    uint32_t select_stride, float score_div) {
    glm_mla_attend_body<false>(q_abs, codes, scales, selected, selected_count, base,
                               select_stride, score_div);
}

// The routed experts: table is the [weight, scales] pointer pairs, items the item list and
// slot_of each item's GLM_MOE_ROWS slots. Grid (cols / 128, 1, items). Without route (gate/up), a
// row is its slot's token row of a ([tokens][inner]) and the output the padded [items][128][cols];
// with route (down), a is the padded [items][128][inner] and each row lands on its slot row of
// out ([slots][cols]) times route[slot].
extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_moe_fp8b128_mma(
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    const int *__restrict__ slot_of, uint32_t topk, const float *__restrict__ route) {
    glm_mma_body<true, true, true>(a, nullptr, nullptr, table, items, out, cols, inner,
                                   GLM_MOE_ROWS, inner, cols, GLM_MOE_ROWS * (uint64_t)inner,
                                   route ? 0ull : GLM_MOE_ROWS * (uint64_t)cols, 0, 0, slot_of,
                                   topk, route);
}

extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_moe_fp8b128_mma_bf16a(
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    const float *__restrict__ a, float *__restrict__ out, uint32_t cols, uint32_t inner,
    const int *__restrict__ slot_of, uint32_t topk, const float *__restrict__ route) {
    glm_mma_body<true, true, false>(a, nullptr, nullptr, table, items, out, cols, inner,
                                    GLM_MOE_ROWS, inner, cols, GLM_MOE_ROWS * (uint64_t)inner,
                                    route ? 0ull : GLM_MOE_ROWS * (uint64_t)cols, 0, 0, slot_of,
                                    topk, route);
}

// Counts the routed slots by expert and lays out items of up to GLM_MOE_ROWS slots each; rows
// past an item's count stay -1. Items past the last are written with a zero count, so a launch
// over max_items runs nothing for them. One block of 1024 threads. items holds (expert, 0, count,
// item). Items are contiguous per expert.
extern "C" __global__ __launch_bounds__(1024) void glm_moe_plan(
    const int *__restrict__ indices, int slots, int experts, int *__restrict__ slot_of,
    int4 *__restrict__ items, int *__restrict__ item_count, int max_items) {
    __shared__ int count[512];
    __shared__ int cursor[512];
    __shared__ int base[512];
    __shared__ int used;
    const int tid = threadIdx.x;
    const int rows = (int)GLM_MOE_ROWS;
    for (int i = tid; i < max_items * rows; i += blockDim.x) slot_of[i] = -1;
    if (experts <= 0 || experts > 512) {
        for (int i = tid; i < max_items; i += blockDim.x) items[i] = make_int4(0, 0, 0, i);
        if (tid == 0) *item_count = 0;
        return;
    }
    for (int e = tid; e < experts; e += blockDim.x) count[e] = 0;
    __syncthreads();
    for (int i = tid; i < slots; i += blockDim.x) {
        const int e = indices[i];
        if (e >= 0 && e < experts) atomicAdd(&count[e], 1);
    }
    __syncthreads();
    if (tid == 0) {
        int at = 0;
        for (int e = 0; e < experts; ++e) {
            base[e] = at;
            cursor[e] = 0;
            const int n = count[e];
            for (int first = 0; first < n; first += rows) {
                const int item = at++;
                if (item < max_items) items[item] = make_int4(e, 0, min(rows, n - first), item);
            }
        }
        used = min(at, max_items);
        *item_count = used;
    }
    __syncthreads();
    for (int i = used + tid; i < max_items; i += blockDim.x) items[i] = make_int4(0, 0, 0, i);
    for (int i = tid; i < slots; i += blockDim.x) {
        const int e = indices[i];
        if (e >= 0 && e < experts) {
            const int rank = atomicAdd(&cursor[e], 1);
            const int item = base[e] + rank / rows;
            if (item < used) slot_of[item * rows + rank % rows] = i;
        }
    }
}

// SwiGLU the gate and up halves of every padded item row into its intermediate row.
extern "C" __global__ void glm_moe_swiglu(const float *__restrict__ gate_up, int width, int rows,
                                          float limit, float *__restrict__ out) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < rows * width;
         index += blockDim.x * gridDim.x) {
        const int row = index / width;
        const int column = index - row * width;
        const float gate = fminf(gate_up[row * 2 * width + column], limit);
        const float up = fminf(fmaxf(gate_up[row * 2 * width + width + column], -limit), limit);
        out[index] = __fmul_rn(gate / (1.0f + __expf(-gate)), up);
    }
}

// ---- FP8 x FP8: the checkpoint's own dynamic activation scheme ----
//
// GLM's FP8 release quantises each activation row per 128 columns (scale = amax / 448, codes
// e4m3(x / scale)), as the decode GEMV's q4e_tg_fp8_activation does. The prefill's FP8 projections
// take those codes straight into an E4M3 MMA (om_mma_e4m3 at unit ue8m0 scales), and every
// 128-column block's sum is folded into the f32 total times its activation and weight scales.

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

// x ([rows][cols] f32, cols a multiple of 128) as codes q ([rows][cols]) and scales s
// ([rows][cols / 128]). One warp a 128-column block.
extern "C" __global__ void glm_quantise_rows(const float *__restrict__ x, uint8_t *__restrict__ q,
                                             float *__restrict__ s, int rows, int cols) {
    const int unit = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31u);
    const int per_row = cols >> 7;
    if (unit >= rows * per_row) return;
    const uint64_t at = (uint64_t)unit * 128u + (uint64_t)lane * 4u;
    float scale;
    const uint32_t codes = glm_q8_codes(*reinterpret_cast<const float4 *>(x + at), scale);
    *reinterpret_cast<uint32_t *>(q + at) = codes;
    if (lane == 0) s[unit] = scale;
}

// glm_moe_swiglu with its output quantised as the down projection's activation: codes q
// ([rows][width]) and scales s ([rows][width / 128]). One warp a 128-column block, width a multiple
// of 128.
extern "C" __global__ void glm_moe_swiglu_q8(const float *__restrict__ gate_up, int width,
                                             int rows, float limit, uint8_t *__restrict__ q,
                                             float *__restrict__ s) {
    const int unit = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31u);
    const int per_row = width >> 7;
    if (unit >= rows * per_row) return;
    const int row = unit / per_row;
    const int column = (unit - row * per_row) * 128 + lane * 4;
    const float *src = gate_up + (uint64_t)row * 2u * width + column;
    const float4 g4 = *reinterpret_cast<const float4 *>(src);
    const float4 u4 = *reinterpret_cast<const float4 *>(src + width);
    auto act = [limit](float g, float u) {
        const float gate = fminf(g, limit);
        const float up = fminf(fmaxf(u, -limit), limit);
        return __fmul_rn(gate / (1.0f + __expf(-gate)), up);
    };
    const float4 v = make_float4(act(g4.x, u4.x), act(g4.y, u4.y), act(g4.z, u4.z), act(g4.w, u4.w));
    float scale;
    const uint32_t codes = glm_q8_codes(v, scale);
    *reinterpret_cast<uint32_t *>(q + (uint64_t)row * width + column) = codes;
    if (lane == 0) s[unit] = scale;
}

// A stage's activation and weight tiles are [128 rows][4 chunks of 16 B] (64 K a stage, two 32-K
// MMA blocks), swizzled as opt_mma_sm121.cu's om_a_off lays the activation out. A weight row's
// 64 bytes stay together: split into opt_mma's two 32-K tiles, the cp.async stage streams the
// weights ~40% slower (126 against 205 GB/s on a routed expert's shape).
__device__ __forceinline__ uint32_t glm_q8_off(uint32_t row, uint32_t chunk) {
    return row * 64u + ((chunk ^ ((row >> 1) & 3u)) << 4);
}

// The q8 body's cp.async stages, and its dynamic shared memory: a stage's activation tile, weight
// tiles and the tile rows' activation scales, then the tile rows' activation rows and the column
// tile's weight scales (an inner of at most GLM_Q8_BLOCKS 128-column blocks).
#define GLM_Q8_STAGES 4u
#define GLM_Q8_BLOCKS 128u
#define GLM_Q8_STAGE_BYTES (GLM_BM * 64u + GLM_BN * 64u + GLM_BM * 4u)
#define GLM_Q8_SHARED (GLM_Q8_STAGES * GLM_Q8_STAGE_BYTES + GLM_BM * 4u + GLM_Q8_BLOCKS * 4u)

// 4 bytes global -> shared, zero-filled when !valid (the source is then not read).
__device__ __forceinline__ void glm_cp4(uint32_t dst, const void *src, bool valid) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(dst), "l"(src),
                 "r"(valid ? 4 : 0));
}

// One block: 128 rows x 128 output columns of an E4M3 weight ([cols][inner], one f32 scale a
// 128 x 128 block) over E4M3 activation codes ([rows][inner], one f32 scale a row's 128 columns),
// 64 K a stage through GLM_Q8_STAGES cp.async stages (GLM_Q8_SHARED bytes of dynamic shared
// memory), inner a multiple of 128. One block fits an SM, so the stages in flight are what keeps
// the DRAM busy: a routed expert's weights stream once an item. Dense: row r is token r and lands
// at out row r, the tiles rastered ([glm_raster]). MoE: group z is item z (items/table/slot_of
// as glm_mma_body's); without route a row reads its slot's token row and writes the padded row,
// with route it reads the padded row and writes its slot row times route[slot].
template <bool MOE>
__device__ __forceinline__ void glm_q8_body(
    const uint8_t *__restrict__ a, const float *__restrict__ as, const uint8_t *__restrict__ w,
    const float *__restrict__ ws, const unsigned long long *__restrict__ table,
    const int4 *__restrict__ items, float *__restrict__ out, uint32_t cols, uint32_t inner,
    uint32_t tokens, const int *__restrict__ slot_of, uint32_t topk,
    const float *__restrict__ route) {
    extern __shared__ __align__(128) uint8_t glm_q8_shared[];
    auto s_a = [&](uint32_t buf) { return glm_q8_shared + buf * GLM_Q8_STAGE_BYTES; };
    auto s_w = [&](uint32_t buf) { return s_a(buf) + GLM_BM * 64u; };
    auto s_as = [&](uint32_t buf) {
        return reinterpret_cast<float *>(s_w(buf) + GLM_BN * 64u);
    };
    int *s_row = reinterpret_cast<int *>(glm_q8_shared + GLM_Q8_STAGES * GLM_Q8_STAGE_BYTES);
    float *s_ws = reinterpret_cast<float *>(s_row + GLM_BM);

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
        w = reinterpret_cast<const uint8_t *>(table[2ull * item.x]);
        ws = reinterpret_cast<const float *>(table[2ull * item.x + 1ull]);
        item_slots = slot_of + group * GLM_MOE_ROWS;
    }
    // The column tile's weight scales ride the first stage's group: a routed expert's are cold,
    // and read at each fold they would put a DRAM round trip on every other stage.
    if (tid < blocks) {
        glm_cp4(om_smem(s_ws + tid), ws + (uint64_t)(n0 >> 7) * blocks + tid, true);
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

// The dense FP8 projection over quantised activations: grid (ceil(cols / 128), ceil(tokens / 128)).
extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_prefill_q8_mma(
    const uint8_t *__restrict__ w, const float *__restrict__ ws, const uint8_t *__restrict__ a,
    const float *__restrict__ as, float *__restrict__ out, uint32_t cols, uint32_t inner,
    uint32_t tokens) {
    glm_q8_body<false>(a, as, w, ws, nullptr, nullptr, out, cols, inner, tokens, nullptr, 1u,
                       nullptr);
}

// The routed experts over quantised activations: grid (cols / 128, 1, items), the arguments of
// glm_moe_fp8b128_mma with the activation's codes and scales.
extern "C" __global__ __launch_bounds__(OM_THREADS) void glm_moe_q8_mma(
    const unsigned long long *__restrict__ table, const int4 *__restrict__ items,
    const uint8_t *__restrict__ a, const float *__restrict__ as, float *__restrict__ out,
    uint32_t cols, uint32_t inner, const int *__restrict__ slot_of, uint32_t topk,
    const float *__restrict__ route) {
    glm_q8_body<true>(a, as, nullptr, nullptr, table, items, out, cols, inner, GLM_MOE_ROWS,
                      slot_of, topk, route);
}

// ---- Skinny projections ----

// out = W x for a BF16 W of `outs` <= 8 JT rows (a 24-row mixer, a 16-row gate: shapes a 128-column
// tile would mostly waste) over `rows` f32 rows of `columns` (a multiple of 8): one warp two rows,
// eight columns a lane a 256-column round, so every activation is read once and the weight rows
// stream from L1. out is [rows][outs].
template <uint32_t JT>
__device__ __forceinline__ void glm_skinny_body(const __nv_bfloat16 *__restrict__ w,
                                                const float *__restrict__ x,
                                                float *__restrict__ out, uint32_t outs,
                                                uint32_t columns, uint32_t rows) {
    constexpr uint32_t J = JT * 8u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 2u;
    if (row0 >= rows) return;
    const bool second = row0 + 1u < rows;
    const float *x0 = x + (uint64_t)row0 * columns;
    const float *x1 = second ? x0 + columns : x0;
    float acc[2][J];
#pragma unroll
    for (uint32_t j = 0; j < J; ++j) acc[0][j] = acc[1][j] = 0.0f;
    for (uint32_t k = lane * 8u; k < columns; k += 256u) {
        const float4 a0 = *reinterpret_cast<const float4 *>(x0 + k);
        const float4 a1 = *reinterpret_cast<const float4 *>(x0 + k + 4u);
        const float4 b0 = *reinterpret_cast<const float4 *>(x1 + k);
        const float4 b1 = *reinterpret_cast<const float4 *>(x1 + k + 4u);
        const float xa[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const float xb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (uint32_t j = 0; j < J; ++j) {
            if (j >= outs) break;
            const uint4 raw = __ldg(reinterpret_cast<const uint4 *>(w + (uint64_t)j * columns + k));
            const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
            for (uint32_t p = 0; p < 4u; ++p) {
                const float lo = __uint_as_float(words[p] << 16);
                const float hi = __uint_as_float(words[p] & 0xFFFF0000u);
                acc[0][j] = fmaf(lo, xa[2u * p], acc[0][j]);
                acc[0][j] = fmaf(hi, xa[2u * p + 1u], acc[0][j]);
                acc[1][j] = fmaf(lo, xb[2u * p], acc[1][j]);
                acc[1][j] = fmaf(hi, xb[2u * p + 1u], acc[1][j]);
            }
        }
    }
#pragma unroll
    for (uint32_t j = 0; j < J; ++j) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[0][j] += __shfl_xor_sync(0xFFFFFFFFu, acc[0][j], offset);
            acc[1][j] += __shfl_xor_sync(0xFFFFFFFFu, acc[1][j], offset);
        }
    }
    // Lane j writes output j of both rows.
#pragma unroll
    for (uint32_t j = 0; j < J; ++j) {
        if (lane == j && j < outs) {
            out[(uint64_t)row0 * outs + j] = acc[0][j];
            if (second) out[(uint64_t)(row0 + 1u) * outs + j] = acc[1][j];
        }
    }
}

// grid (ceil(rows / 16)), 256 threads; the entry is the one whose 8 JT covers `outs`.
#define GLM_SKINNY_ENTRY(JT)                                                                       \
    extern "C" __global__ __launch_bounds__(256) void glm_prefill_skinny_bf16_##JT(               \
        const __nv_bfloat16 *__restrict__ w, const float *__restrict__ x,                          \
        float *__restrict__ out, uint32_t outs, uint32_t columns, uint32_t rows) {                 \
        glm_skinny_body<JT>(w, x, out, outs, columns, rows);                                       \
    }
GLM_SKINNY_ENTRY(1)
GLM_SKINNY_ENTRY(2)
GLM_SKINNY_ENTRY(3)
GLM_SKINNY_ENTRY(4)

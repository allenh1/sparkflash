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

// GLM-5.3-Flash's bulk sparse MLA attention kernels.
//
// The decode arm reads every weight once per eight rows. A prefill chunk of 128-1024 rows runs
// the absorbed query and the value projection as per-head GEMMs over the chunk, and the sparse
// attention over the rank's selected latents with the 16 heads as the M of bf16 MMAs.
//
// These kernels were split out of glm_prefill_sm121.cu into their own part image, so the MLA
// attention builds and loads independently of the bulk prefill GEMM and MoE kernels. The per-head
// projections share the dense GEMM's tile body with the prefill, so this translation unit carries
// its own copy of glm_mma_body and the geometry it reads; it is self-contained.
//
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "opt_mma_common_sm121.cuh"
#include "opt_mma_bf16_sm121.cuh"
#include "glm_common_sm121.cuh"


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
// Heads one blockIdx.z tile computes. It is a tile size, not a rank cap: the host launches
// heads.div_ceil(GMA_HEADS) tiles and the body masks the last tile's partial head count.
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
    uint32_t select_stride, float score_div, uint32_t heads) {
    __shared__ __align__(128) uint8_t s_k[GMA_TILE * GMA_LATENT * 2u];
    __shared__ __align__(8) float s_s[GMA_BLOCKS][GMA_HEADS][GMA_TILE];
    __shared__ __align__(8) float s_p[GMA_HEADS * GMA_P_LD];
    __shared__ float s_scale[GMA_TILE][GMA_BLOCKS];
    __shared__ uint32_t s_valid[GMA_TILE];
    __shared__ float s_m[GMA_HEADS], s_l[GMA_HEADS], s_alpha[GMA_HEADS];

    const uint32_t row = base + blockIdx.x;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, q = lane & 3u;
    // One blockIdx.z tile is GMA_HEADS heads; a rank may hold more (8, 16, 32, 64), and the
    // last tile may be partial.
    const uint32_t head_base = blockIdx.z * GMA_HEADS;
    const uint32_t local_heads =
        heads > head_base ? min(GMA_HEADS, heads - head_base) : 0u;
    float *io = q_abs + (uint64_t)row * heads * GMA_LATENT +
                (uint64_t)head_base * GMA_LATENT;
    const int *sel = selected + (uint64_t)row * select_stride;
    const int count = selected_count[row];
    if (count <= 0 || count > (int)GMA_MAX_SELECTED) {
        const float fill = count <= 0 ? 0.0f : __uint_as_float(0x7FC00000u);
        __syncthreads();
        for (uint32_t i = tid; i < local_heads * GMA_LATENT; i += blockDim.x) io[i] = fill;
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
        const float2 zero = make_float2(0.0f, 0.0f);
        const float2 v0 = g < local_heads
                              ? *reinterpret_cast<const float2 *>(io + g * GMA_LATENT + k0)
                              : zero;
        const float2 v1 = g + 8u < local_heads
                              ? *reinterpret_cast<const float2 *>(io + (g + 8u) * GMA_LATENT + k0)
                              : zero;
        const float2 v2 = g < local_heads
                              ? *reinterpret_cast<const float2 *>(io + g * GMA_LATENT + k0 + 8u)
                              : zero;
        const float2 v3 = g + 8u < local_heads
                              ? *reinterpret_cast<const float2 *>(io + (g + 8u) * GMA_LATENT + k0 + 8u)
                              : zero;
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
        if (g < local_heads) {
            *reinterpret_cast<float2 *>(io + g * GMA_LATENT + d) =
                make_float2(acc[nt][0] * inv_lo, acc[nt][1] * inv_lo);
        }
        if (g + 8u < local_heads) {
            *reinterpret_cast<float2 *>(io + (g + 8u) * GMA_LATENT + d) =
                make_float2(acc[nt][2] * inv_hi, acc[nt][3] * inv_hi);
        }
    }
}

// Grid (rows of the run), 256 threads; see glm_mla_attend_body.
extern "C" __global__ __launch_bounds__(256) void glm_prefill_mla_attend(
    float *q_abs, const uint8_t *__restrict__ codes, const float *__restrict__ scales,
    const int *__restrict__ selected, const int *__restrict__ selected_count, uint32_t base,
    uint32_t select_stride, float score_div, uint32_t heads) {
    glm_mla_attend_body<true>(q_abs, codes, scales, selected, selected_count, base,
                              select_stride, score_div, heads);
}

// The same over the bf16 head of the query and the probabilities alone
// (SPARKFLASH_GLM_PREFILL_ACTIVATION=bf16): one MMA chain where the split runs two.
extern "C" __global__ __launch_bounds__(256) void glm_prefill_mla_attend_bf16a(
    float *q_abs, const uint8_t *__restrict__ codes, const float *__restrict__ scales,
    const int *__restrict__ selected, const int *__restrict__ selected_count, uint32_t base,
    uint32_t select_stride, float score_div, uint32_t heads) {
    glm_mla_attend_body<false>(q_abs, codes, scales, selected, selected_count, base,
                               select_stride, score_div, heads);
}

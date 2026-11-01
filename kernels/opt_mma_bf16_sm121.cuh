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

// kernels/opt_mma_bf16_sm121.cuh
//
// The bf16 tensor-core GEMM body (opt_mma_bf16_body), lifted out of kernels/opt_mma_sm121.cu so
// every model pack's bulk prefill runs one implementation. It multiplies a chunk of activation rows
// (f32 split into a bf16 head and the bf16 of its remainder - about 16 of 24 significant bits - or
// bf16) by weights in one of six layouts (the B12x E4M3 and host tilings, plain row-major bf16 with
// or without an explicit row stride, row-major E4M3), with mma.sync m16n8k16 bf16 x bf16 into f32:
// a 128-token x 128-column block, the next K stage loaded into registers while the current one
// computes. The strided row-major mode is the one addition since the lift; the DeepSeek pack does not
// instantiate it, so its pinned cubin is unchanged.
//
// Constraints the caller owns: `cols` and `inner` are multiples of 32 (a partial 32-column tile or
// K block is not computed), and the f32 / bf16 activation rows are 16-byte aligned.
//
// Attribution:
//   - The SM120/SM121 (GB10) mma.sync / ldmatrix / cp.async tiling patterns draw on prior art from
//     b12x (https://github.com/local-inference-lab/b12x, flashinfer/experimental/b12x, Apache-2.0;
//     Luke Alonso @lukealonso, @original-el8, and contributors) and NVIDIA Flare
//     (https://github.com/NVIDIA/flare-local-inference, Apache-2.0). See the root NOTICE file.
//
// opt_mma_sm121.cu includes this header where the body used to stand, so its pinned cubin is
// unchanged; see kernels/opt_mma_common_sm121.cuh.

#ifndef SPARKFLASH_OPT_MMA_BF16_SM121_CUH
#define SPARKFLASH_OPT_MMA_BF16_SM121_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "opt_mma_common_sm121.cuh"

#define OB_A_BF16 0
#define OB_A_F32 2
#define OB_W_FP8 0    // B12x E4M3 tiles + one ue8m0 per tile
#define OB_W_BF16 1   // host weights: the same tiling, bf16 elements, no scales
#define OB_W_F32 2    // host weights, f32 elements (split like an f32 activation)
#define OB_W_ROW 3    // plain row-major bf16 [cols][inner] (the router gate)
#define OB_W_FP8ROW 4 // row-major E4M3 [cols][inner], one ue8m0 per row per 32 along K (the drafter head)
#define OB_W_ROWK 5   // row-major bf16 whose rows are k_stride_blocks * 32 long: one K slice of a wider
                      // matrix per blockIdx.z (a split-K launch; the Qwen prefill's narrow projections)
#define OB_LD 40u   // bf16 per shared row: 32 + 8 padding, 80 bytes, so ldmatrix phases do not conflict

__device__ __forceinline__ void ob_mma(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t ob_pack(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
}

// Two E4M3 codes (low 16 bits) times a power-of-two scale -> bf16x2, exact.
__device__ __forceinline__ uint32_t ob_e4m3x2(uint32_t two, float scale) {
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((uint16_t)(two & 0xffffu)));
    const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(&h));
    return ob_pack(__float2bfloat16_rn(f.x * scale), __float2bfloat16_rn(f.y * scale));
}

template <int AMODE, int WMODE, bool OUT_F32>
__device__ __forceinline__ void opt_mma_bf16_body(
    const void *__restrict__ a, const uint8_t *__restrict__ w, const uint8_t *__restrict__ ws,
    void *__restrict__ out, const uint32_t cols, const uint32_t inner, const uint32_t tokens,
    const uint64_t a_tok_stride, const uint64_t out_tok_stride, const uint64_t a_grp_stride,
    const uint64_t w_grp_stride, const uint64_t s_grp_stride, const uint64_t out_grp_stride,
    const uint32_t k_stride_blocks) {
    constexpr bool SPLIT = AMODE == OB_A_F32;
    constexpr bool WSPLIT = WMODE == OB_W_F32;
    __shared__ __align__(16) __nv_bfloat16 s_ah[OM_BM * OB_LD];
    __shared__ __align__(16) __nv_bfloat16 s_al[SPLIT ? OM_BM * OB_LD : 8];
    __shared__ __align__(16) __nv_bfloat16 s_w[OM_BN * OB_LD];
    __shared__ __align__(16) __nv_bfloat16 s_wl[WSPLIT ? OM_BN * OB_LD : 8];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t n0 = blockIdx.x * OM_BN;
    const uint32_t m0 = blockIdx.y * OM_BM;
    const uint32_t group = blockIdx.z;
    const uint32_t k_blocks = inner >> 5;
    const uint32_t k_row = k_stride_blocks != 0u ? k_stride_blocks : k_blocks;
    const uint32_t col_tiles = min(OM_BN / 32u, (cols - n0) >> 5);

    const uint64_t a_grp = group * a_grp_stride;
    // A weight element's bytes: E4M3 codes, or the host weights' bf16 / f32.
    constexpr bool WFP8 = WMODE == OB_W_FP8 || WMODE == OB_W_FP8ROW;
    constexpr uint32_t WB = WFP8 ? 1u : WMODE == OB_W_F32 ? 4u : 2u;
    const uint8_t *w_grp = WMODE == OB_W_ROW || WMODE == OB_W_ROWK || WMODE == OB_W_FP8ROW ? w + group * w_grp_stride
                                             : w + group * w_grp_stride + (uint64_t)(n0 >> 5) * k_row * 1024u * WB;
    const uint8_t *ws_grp = WMODE == OB_W_FP8 ? ws + group * s_grp_stride + (uint64_t)(n0 >> 5) * k_row : ws;

    // Register prefetch of one stage: the f32 form holds four float4 per thread (128 rows x 8),
    // the bf16 form two uint4 (128 rows x 4); the weights WB uint4 (4 tiles x 32 columns x 32 / 16
    // bytes per element byte): thread t owns column (t % 64) / 2 of tile t / 64, values 16 (t % 2)..
    constexpr uint32_t A_PER = SPLIT ? 4u : 2u;
    uint4 ra[A_PER];
    uint4 rw[WB];
    uint8_t rs;
    const uint32_t w_tile = tid >> 6;
    const uint32_t w_col = (tid & 63u) >> 1;
    const uint32_t w_half = tid & 1u;
    const bool w_live = w_tile < col_tiles;

    auto load = [&](uint32_t kb) {
#pragma unroll
        for (uint32_t i = 0; i < A_PER; ++i) {
            const uint32_t idx = tid + i * OM_THREADS;
            const uint32_t row = SPLIT ? idx >> 3 : idx >> 2;
            const uint32_t c = SPLIT ? idx & 7u : idx & 3u;
            const uint32_t token = m0 + row;
            if (token < tokens) {
                const uint64_t e = a_grp + (uint64_t)token * a_tok_stride + (uint64_t)kb * 32u + c * (SPLIT ? 4u : 8u);
                ra[i] = SPLIT ? __ldg(reinterpret_cast<const uint4 *>(reinterpret_cast<const float *>(a) + e))
                              : __ldg(reinterpret_cast<const uint4 *>(reinterpret_cast<const __nv_bfloat16 *>(a) + e));
            } else {
                ra[i] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        if (w_live) {
            const uint4 *src;
            if constexpr (WMODE == OB_W_ROW) {
                const uint64_t col = n0 + w_tile * 32u + w_col;
                src = reinterpret_cast<const uint4 *>(w_grp + (col * inner + kb * 32u + w_half * 16u) * 2u);
            } else if constexpr (WMODE == OB_W_ROWK) {
                const uint64_t col = n0 + w_tile * 32u + w_col;
                src = reinterpret_cast<const uint4 *>(w_grp + (col * k_row * 32u + kb * 32u + w_half * 16u) * 2u);
            } else if constexpr (WMODE == OB_W_FP8ROW) {
                const uint64_t col = n0 + w_tile * 32u + w_col;
                src = reinterpret_cast<const uint4 *>(w_grp + col * inner + kb * 32u + w_half * 16u);
            } else {
                const uint8_t *tile = w_grp + ((uint64_t)w_tile * k_row + kb) * 1024u * WB;
                src = reinterpret_cast<const uint4 *>(tile + (w_col * 32u + w_half * 16u) * WB);
            }
#pragma unroll
            for (uint32_t v = 0; v < WB; ++v) rw[v] = __ldg(src + v);
            rs = WMODE == OB_W_FP8      ? __ldg(ws_grp + (uint64_t)w_tile * k_row + kb)
                 : WMODE == OB_W_FP8ROW ? __ldg(ws + (uint64_t)(n0 + w_tile * 32u + w_col) * k_row + kb)
                                        : (uint8_t)127u;
        } else {
#pragma unroll
            for (uint32_t v = 0; v < WB; ++v) rw[v] = make_uint4(0u, 0u, 0u, 0u);
            rs = 127u;
        }
    };
    auto store = [&]() {
#pragma unroll
        for (uint32_t i = 0; i < A_PER; ++i) {
            const uint32_t idx = tid + i * OM_THREADS;
            if constexpr (SPLIT) {
                const uint32_t row = idx >> 3;
                const uint32_t c = idx & 7u;
                const float v[4] = {__uint_as_float(ra[i].x), __uint_as_float(ra[i].y), __uint_as_float(ra[i].z),
                                    __uint_as_float(ra[i].w)};
                __nv_bfloat16 h[4], l[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    h[e] = __float2bfloat16_rn(v[e]);
                    l[e] = __float2bfloat16_rn(v[e] - __bfloat162float(h[e]));
                }
                *reinterpret_cast<uint2 *>(&s_ah[row * OB_LD + c * 4u]) = make_uint2(ob_pack(h[0], h[1]), ob_pack(h[2], h[3]));
                *reinterpret_cast<uint2 *>(&s_al[row * OB_LD + c * 4u]) = make_uint2(ob_pack(l[0], l[1]), ob_pack(l[2], l[3]));
            } else {
                const uint32_t row = idx >> 2;
                const uint32_t c = idx & 3u;
                *reinterpret_cast<uint4 *>(&s_ah[row * OB_LD + c * 8u]) = ra[i];
            }
        }
        const uint32_t w_at = (w_tile * 32u + w_col) * OB_LD + w_half * 16u;
        __nv_bfloat16 *dst = &s_w[w_at];
        if constexpr (WFP8) {
            const float scale = opt_ue8m0_f32(rs);
            const uint4 r = rw[0];
            *reinterpret_cast<uint4 *>(dst) =
                make_uint4(ob_e4m3x2(r.x, scale), ob_e4m3x2(r.x >> 16, scale), ob_e4m3x2(r.y, scale), ob_e4m3x2(r.y >> 16, scale));
            *reinterpret_cast<uint4 *>(dst + 8) =
                make_uint4(ob_e4m3x2(r.z, scale), ob_e4m3x2(r.z >> 16, scale), ob_e4m3x2(r.w, scale), ob_e4m3x2(r.w >> 16, scale));
        } else if constexpr (WMODE == OB_W_BF16 || WMODE == OB_W_ROW || WMODE == OB_W_ROWK) {
            *reinterpret_cast<uint4 *>(dst) = rw[0];
            *reinterpret_cast<uint4 *>(dst + 8) = rw[1];
        } else {
            // f32 weights: the bf16 head and the bf16 of the remainder, like an f32 activation.
#pragma unroll
            for (uint32_t v = 0; v < 4u; ++v) {
                const float f[4] = {__uint_as_float(rw[v].x), __uint_as_float(rw[v].y), __uint_as_float(rw[v].z),
                                    __uint_as_float(rw[v].w)};
                __nv_bfloat16 h[4], l[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    h[e] = __float2bfloat16_rn(f[e]);
                    l[e] = __float2bfloat16_rn(f[e] - __bfloat162float(h[e]));
                }
                *reinterpret_cast<uint2 *>(dst + 4u * v) = make_uint2(ob_pack(h[0], h[1]), ob_pack(h[2], h[3]));
                *reinterpret_cast<uint2 *>(&s_wl[w_at + 4u * v]) = make_uint2(ob_pack(l[0], l[1]), ob_pack(l[2], l[3]));
            }
        }
    };

    const uint32_t wm = warp & 1u;   // rows 64 wm .. 64 wm + 63
    const uint32_t wn = warp >> 1;   // columns 32 wn .. 32 wn + 31
    const bool active = wn < col_tiles && m0 + wm * 64u < tokens;

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

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
                uint32_t bl[WSPLIT ? 4 : 1][2];
#pragma unroll
                for (uint32_t p = 0; p < 2u; ++p) {
                    const uint32_t n = wn * 32u + p * 16u + (lane & 7u) + 8u * (lane >> 4);
                    const uint32_t k = k0 + 8u * ((lane >> 3) & 1u);
                    om_ldsm4(om_smem(&s_w[n * OB_LD + k]), b[2 * p][0], b[2 * p][1], b[2 * p + 1][0], b[2 * p + 1][1]);
                    if constexpr (WSPLIT) {
                        om_ldsm4(om_smem(&s_wl[n * OB_LD + k]), bl[2 * p][0], bl[2 * p][1], bl[2 * p + 1][0], bl[2 * p + 1][1]);
                    }
                }
#pragma unroll
                for (uint32_t mt = 0; mt < 4u; ++mt) {
                    const uint32_t row = wm * 64u + mt * 16u + (lane & 15u);
                    const uint32_t k = k0 + 8u * (lane >> 4);
                    uint32_t af[4];
                    om_ldsm4(om_smem(&s_ah[row * OB_LD + k]), af[0], af[1], af[2], af[3]);
#pragma unroll
                    for (uint32_t nt = 0; nt < 4u; ++nt) ob_mma(acc[mt][nt], af, b[nt][0], b[nt][1]);
                    if constexpr (WSPLIT) {
#pragma unroll
                        for (uint32_t nt = 0; nt < 4u; ++nt) ob_mma(acc[mt][nt], af, bl[nt][0], bl[nt][1]);
                    }
                    if constexpr (SPLIT) {
                        om_ldsm4(om_smem(&s_al[row * OB_LD + k]), af[0], af[1], af[2], af[3]);
#pragma unroll
                        for (uint32_t nt = 0; nt < 4u; ++nt) ob_mma(acc[mt][nt], af, b[nt][0], b[nt][1]);
                    }
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
            const uint64_t base = group * out_grp_stride + (uint64_t)token * out_tok_stride + n0 + wn * 32u + 2u * q;
#pragma unroll
            for (uint32_t nt = 0; nt < 4u; ++nt) {
                const float v0 = acc[mt][nt][2u * half];
                const float v1 = acc[mt][nt][2u * half + 1u];
                const uint64_t o = base + nt * 8u;
                if constexpr (OUT_F32) {
                    *reinterpret_cast<float2 *>(reinterpret_cast<float *>(out) + o) = make_float2(v0, v1);
                } else {
                    *reinterpret_cast<__nv_bfloat162 *>(reinterpret_cast<__nv_bfloat16 *>(out) + o) =
                        __floats2bfloat162_rn(v0, v1);
                }
            }
        }
    }
}

#endif  // SPARKFLASH_OPT_MMA_BF16_SM121_CUH

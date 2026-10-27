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

// kernels/opt_gemv_sm121.cu
//
// Split-K MXFP8 GEMV for the decode step (M = 1..8 tokens) over the B12x 32x32 tiled weights.
//
// One block = 8 warps = one 32-column tile of one group. Warp w owns K-blocks w, w+8, w+16, ... of
// that tile; lane l owns column l and reads its 32 contiguous code bytes per K-block (the warp reads
// the 1024-byte tile contiguously, as streaming loads). Per K-block the 32 products are folded into
// one f32 block partial with FMAs, scaled once by the (power-of-two) ue8m0 weight scale - and, in the
// E4M3 activation mode, by the activation block scale - and added to the lane's accumulator. The
// eight warp partials of a column are summed in fixed warp order through shared memory, so the
// result is deterministic.
//
// Activations: when the block's token rows (and E4M3 scales) fit in OPT_G_STAGE bytes they are
// copied to shared memory once, behind the weight prologue, and read from there as broadcast loads;
// otherwise they are read from global memory as broadcast loads. Read from global memory, a row is
// an L2 round trip on every first touch per SM and is evicted by the streaming weights, which costs
// several microseconds per extra token; staged, a token costs about its arithmetic.
//
// Activation modes: OPT_A_BF16 (bf16 row), OPT_A_FP8 (E4M3 codes + one ue8m0 per 32), OPT_A_F32.
// Output: bf16 (round to nearest) or f32.
//
// With E4M3 activations and more than one token per block, both operands stay f16x2 (E4M3 is exact
// in f16) and the products use the mixed-precision FMA (f16 x f16 into f32). An f16 product is exact
// in f32, so that is one rounding, the same bits as the f32 FMA on the widened values, without the
// per-token widening.
//
// Shapes: cols % 32 == 0, inner % 32 == 0. Grid (cols / 32, groups, ceil(tokens / MAXT)), block
// 256: each block serves one tile and tokens z * MAXT .. z * MAXT + MAXT - 1 (fewer in the last). The
// blocks are numbered token split fastest (block (x, z) runs tile (z * gridDim.x + x) / gridDim.z), so
// a tile's splits run side by side and all but the first read its weights from L2, not DRAM. A
// token's chain does not depend on MAXT, the split, the numbering or the staging, so every
// instantiation and grid gives the same bits.

#include "opt_pdl_sm121.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define OPT_G_WARPS 8u
#define OPT_G_THREADS (OPT_G_WARPS * 32u)
#define OPT_G_DEPTH 4u
// Staging bytes per block: two blocks per SM (the register bound) fit in the SM's 100 KB. The
// one-token form runs three blocks per SM and needs at most one row.
#define OPT_G_STAGE 48128u
#define OPT_G_STAGE_T1 20480u

#define OPT_A_BF16 0
#define OPT_A_FP8 1
#define OPT_A_F32 2

__device__ __forceinline__ uint4 opt_g_ldw(const uint8_t *p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// 16 activation bytes from the staged rows (shared memory) or from global memory.
template <bool STAGED>
__device__ __forceinline__ uint4 opt_g_lda(const uint8_t *p) {
    if constexpr (STAGED) {
        uint4 v;
        asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "r"((uint32_t)__cvta_generic_to_shared(p)));
        return v;
    } else {
        return __ldg(reinterpret_cast<const uint4 *>(p));
    }
}

template <bool STAGED>
__device__ __forceinline__ uint8_t opt_g_lds8(const uint8_t *p) {
    if constexpr (STAGED) {
        return *p;
    } else {
        return __ldg(p);
    }
}

__device__ __forceinline__ float opt_g_ue8m0(uint8_t code) {
    // 2^(code - 127); code 0 is 2^-127 (subnormal), 255 is NaN in the format but never produced.
    return code == 0u ? __uint_as_float(0x00400000u) : __uint_as_float((uint32_t)code << 23);
}

// 4 bytes of E4M3 -> 4 floats.
__device__ __forceinline__ void opt_g_e4m3x4(uint32_t w, float &f0, float &f1, float &f2, float &f3) {
    uint32_t h01, h23;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h01) : "h"((uint16_t)(w & 0xffffu)));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h23) : "h"((uint16_t)(w >> 16)));
    const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&h01));
    const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&h23));
    f0 = a.x; f1 = a.y; f2 = b.x; f3 = b.y;
}

// Two E4M3 bytes (the low 16 bits) -> f16x2, the low byte in the low half.
__device__ __forceinline__ uint32_t opt_g_e4m3x2(uint32_t two) {
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((uint16_t)(two & 0xffffu)));
    return h;
}

// c + a.lo * b.lo, then + a.hi * b.hi, on f16x2 pairs: two single-rounding f32 FMAs.
__device__ __forceinline__ float opt_g_fma2_f16(uint32_t a, uint32_t b, float c) {
    asm("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(c) : "h"((uint16_t)(a & 0xffffu)), "h"((uint16_t)(b & 0xffffu)));
    asm("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(c) : "h"((uint16_t)(a >> 16)), "h"((uint16_t)(b >> 16)));
    return c;
}

// The 32 E4M3 activation values of K-block kb of one token row as 16 f16x2 pairs, plus the scale.
template <bool STAGED>
__device__ __forceinline__ void opt_g_act_fp8_pairs(const uint8_t *row, const uint8_t *srow, uint32_t kb,
                                                    uint32_t v[16], float &scale) {
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const uint4 u = opt_g_lda<STAGED>(row + kb * 32u + q * 16u);
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[8 * q + 2 * i] = opt_g_e4m3x2(w[i]);
            v[8 * q + 2 * i + 1] = opt_g_e4m3x2(w[i] >> 16);
        }
    }
    scale = opt_g_ue8m0(opt_g_lds8<STAGED>(srow + kb));
}

// The 32 activation values of K-block kb of one row staged as f16 (64 bytes), as 16 f16x2 pairs.
__device__ __forceinline__ void opt_g_act_f16_pairs(const uint8_t *row, uint32_t kb, uint32_t v[16]) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const uint4 u = opt_g_lda<true>(row + kb * 64u + q * 16u);
        v[4 * q] = u.x; v[4 * q + 1] = u.y; v[4 * q + 2] = u.z; v[4 * q + 3] = u.w;
    }
}

// The 32 activation values of K-block kb of one token row, as f32, plus the block scale (1 unless
// FP8). `row` is the row's first byte; `srow` its first E4M3 scale.
template <int AMODE, bool STAGED>
__device__ __forceinline__ void opt_g_act(const uint8_t *row, const uint8_t *srow, uint32_t kb, float v[32],
                                          float &scale) {
    if constexpr (AMODE == OPT_A_BF16) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint4 u = opt_g_lda<STAGED>(row + kb * 64u + q * 16u);
            const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v[q * 8 + 2 * i] = __uint_as_float(w[i] << 16);
                v[q * 8 + 2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
            }
        }
        scale = 1.0f;
    } else if constexpr (AMODE == OPT_A_FP8) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const uint4 u = opt_g_lda<STAGED>(row + kb * 32u + q * 16u);
            const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                opt_g_e4m3x4(w[i], v[q * 16 + 4 * i], v[q * 16 + 4 * i + 1], v[q * 16 + 4 * i + 2], v[q * 16 + 4 * i + 3]);
            }
        }
        scale = opt_g_ue8m0(opt_g_lds8<STAGED>(srow + kb));
    } else {
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const uint4 u = opt_g_lda<STAGED>(row + kb * 128u + q * 16u);
            v[q * 4] = __uint_as_float(u.x); v[q * 4 + 1] = __uint_as_float(u.y);
            v[q * 4 + 2] = __uint_as_float(u.z); v[q * 4 + 3] = __uint_as_float(u.w);
        }
        scale = 1.0f;
    }
}

// Where a block's token rows are read from: global memory, shared memory as staged (the input
// encoding), or shared memory widened (E4M3 rows as f16, their scales as f32).
#define OPT_G_GLOBAL 0
#define OPT_G_STAGED 1
#define OPT_G_WIDE 2

template <int S>
struct opt_g_tag {
    static constexpr int value = S;
};

template <int AMODE, bool OUT_F32, int MAXT>
__device__ __forceinline__ void opt_gemv_body(
    const void *__restrict__ a, const uint8_t *__restrict__ a_scales,
    const uint8_t *__restrict__ w, const uint8_t *__restrict__ ws, void *__restrict__ out,
    const uint32_t cols, const uint32_t inner, const uint32_t tokens,
    const uint64_t a_tok_stride, const uint64_t out_tok_stride,
    const uint64_t a_grp_stride, const uint64_t w_grp_stride, const uint64_t s_grp_stride,
    const uint64_t out_grp_stride, const uint32_t k_stride_blocks) {
    // E4M3 activations with several tokens, or widened in shared memory: f16x2 operands and the
    // mixed-precision FMA (one token converted per lane is faster on the f32 chain).
    constexpr uint32_t EB = AMODE == OPT_A_BF16 ? 2u : AMODE == OPT_A_FP8 ? 1u : 4u;
    constexpr uint32_t STAGE = MAXT == 1 ? OPT_G_STAGE_T1 : OPT_G_STAGE;
    static_assert(STAGE >= OPT_G_WARPS * MAXT * 32u * 4u, "the warp partials reuse the staging bytes");
    __shared__ __align__(16) uint8_t s_mem[STAGE];

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    // Token split fastest: consecutive blocks are the splits of one tile.
    const uint32_t lin = blockIdx.z * gridDim.x + blockIdx.x;
    const uint32_t tile = lin / gridDim.z;
    const uint32_t group = blockIdx.y;
    const uint32_t k_blocks = inner >> 5;
    const uint32_t z0 = (lin - tile * gridDim.z) * (uint32_t)MAXT;
    const uint32_t live = tokens - z0 < (uint32_t)MAXT ? tokens - z0 : (uint32_t)MAXT;

    // A K-slice of a wider matrix keeps the full matrix's tile-row stride.
    const uint32_t k_row = k_stride_blocks != 0u ? k_stride_blocks : k_blocks;
    const uint8_t *wt = w + group * w_grp_stride + (uint64_t)tile * k_row * 1024u + lane * 32u;
    const uint8_t *st = ws + group * s_grp_stride + (uint64_t)tile * k_row;
    // A token-split launch offsets the activation and output rows by this block's first token.
    const uint64_t a_base = group * a_grp_stride + (uint64_t)z0 * a_tok_stride;

    float acc[MAXT];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) acc[t] = 0.0f;

    // This warp's K-blocks: warp, warp + 8, ...; DEPTH of them in flight.
    const uint32_t mine = warp < k_blocks ? (k_blocks - warp + OPT_G_WARPS - 1u) / OPT_G_WARPS : 0u;
    uint4 lo[OPT_G_DEPTH], hi[OPT_G_DEPTH];
    uint8_t sc[OPT_G_DEPTH];
#pragma unroll
    for (uint32_t d = 0; d < OPT_G_DEPTH; ++d) {
        if (d < mine) {
            const uint32_t kb = warp + d * OPT_G_WARPS;
            lo[d] = opt_g_ldw(wt + (uint64_t)kb * 1024u);
            hi[d] = opt_g_ldw(wt + (uint64_t)kb * 1024u + 16u);
            sc[d] = __ldg(st + kb);
        }
    }

    // Stage the rows behind the weight prologue: row t at t * row_bytes, the E4M3 scales after them.
    // E4M3 rows are widened to f16 (exact) and their scales to f32 when that fits, so a lane reads
    // operands instead of converting every token's codes itself (32 lanes converting the same bytes);
    // otherwise they are copied as they are.
    const uint32_t row_bytes = inner * EB;
    const uint32_t wide_bytes = inner * 2u;
    const uint32_t wide_scale_off = live * wide_bytes;
    const bool wide = AMODE == OPT_A_FP8 && wide_scale_off + live * k_blocks * 4u <= STAGE;
    const uint32_t scale_off = live * row_bytes;
    const uint32_t need = scale_off + (AMODE == OPT_A_FP8 ? live * k_blocks : 0u);
    const bool staged = wide || need <= STAGE;
    if (wide) {
        const uint32_t row16 = inner / 16u;
#pragma unroll 4
        for (uint32_t idx = threadIdx.x; idx < live * row16; idx += OPT_G_THREADS) {
            const uint32_t t = idx / row16;
            const uint32_t c = idx - t * row16;
            const uint8_t *src = reinterpret_cast<const uint8_t *>(a) + a_base + (uint64_t)t * a_tok_stride;
            const uint4 u = __ldg(reinterpret_cast<const uint4 *>(src) + c);
            uint8_t *dst = s_mem + t * wide_bytes + c * 32u;
            *reinterpret_cast<uint4 *>(dst) = make_uint4(opt_g_e4m3x2(u.x), opt_g_e4m3x2(u.x >> 16),
                                                         opt_g_e4m3x2(u.y), opt_g_e4m3x2(u.y >> 16));
            *reinterpret_cast<uint4 *>(dst + 16u) = make_uint4(opt_g_e4m3x2(u.z), opt_g_e4m3x2(u.z >> 16),
                                                               opt_g_e4m3x2(u.w), opt_g_e4m3x2(u.w >> 16));
        }
        float *s_scale = reinterpret_cast<float *>(s_mem + wide_scale_off);
        for (uint32_t idx = threadIdx.x; idx < live * k_blocks; idx += OPT_G_THREADS) {
            const uint32_t t = idx / k_blocks;
            const uint32_t c = idx - t * k_blocks;
            s_scale[idx] = opt_g_ue8m0(__ldg(a_scales + (a_base + (uint64_t)t * a_tok_stride) / 32u + c));
        }
        __syncthreads();
    } else if (staged) {
        const uint32_t row16 = row_bytes / 16u;
#pragma unroll 4
        for (uint32_t idx = threadIdx.x; idx < live * row16; idx += OPT_G_THREADS) {
            const uint32_t t = idx / row16;
            const uint32_t c = idx - t * row16;
            const uint8_t *src = reinterpret_cast<const uint8_t *>(a) + (a_base + (uint64_t)t * a_tok_stride) * EB;
            *reinterpret_cast<uint4 *>(s_mem + idx * 16u) = __ldg(reinterpret_cast<const uint4 *>(src) + c);
        }
        if constexpr (AMODE == OPT_A_FP8) {
            for (uint32_t idx = threadIdx.x; idx < live * k_blocks; idx += OPT_G_THREADS) {
                const uint32_t t = idx / k_blocks;
                const uint32_t c = idx - t * k_blocks;
                s_mem[scale_off + idx] = __ldg(a_scales + (a_base + (uint64_t)t * a_tok_stride) / 32u + c);
            }
        }
        __syncthreads();
    }

    auto main_loop = [&](auto tag) {
        constexpr int SRC = decltype(tag)::value;
        constexpr bool STAGED = SRC != OPT_G_GLOBAL;
        constexpr bool PAIRS = AMODE == OPT_A_FP8 && (MAXT > 1 || SRC == OPT_G_WIDE);
        for (uint32_t i0 = 0; i0 < mine; i0 += OPT_G_DEPTH) {
#pragma unroll
            for (uint32_t d = 0; d < OPT_G_DEPTH; ++d) {
                const uint32_t i = i0 + d;
                if (i < mine) {
                    const uint32_t kb = warp + i * OPT_G_WARPS;
                    // The weights as 32 f32, or as 16 f16x2 pairs on the pairs path.
                    float wf[PAIRS ? 1 : 32];
                    uint32_t wp[PAIRS ? 16 : 1];
                    {
                        const uint32_t ww[8] = {lo[d].x, lo[d].y, lo[d].z, lo[d].w, hi[d].x, hi[d].y, hi[d].z, hi[d].w};
#pragma unroll
                        for (int q = 0; q < 8; ++q) {
                            if constexpr (PAIRS) {
                                wp[2 * q] = opt_g_e4m3x2(ww[q]);
                                wp[2 * q + 1] = opt_g_e4m3x2(ww[q] >> 16);
                            } else {
                                opt_g_e4m3x4(ww[q], wf[4 * q], wf[4 * q + 1], wf[4 * q + 2], wf[4 * q + 3]);
                            }
                        }
                    }
                    const float wscale = opt_g_ue8m0(sc[d]);
                    // Refill this slot DEPTH blocks ahead before the arithmetic.
                    const uint32_t nx = i + OPT_G_DEPTH;
                    if (nx < mine) {
                        const uint32_t kbn = warp + nx * OPT_G_WARPS;
                        lo[d] = opt_g_ldw(wt + (uint64_t)kbn * 1024u);
                        hi[d] = opt_g_ldw(wt + (uint64_t)kbn * 1024u + 16u);
                        sc[d] = __ldg(st + kbn);
                    }
#pragma unroll
                    for (int t = 0; t < MAXT; ++t) {
                        if (t < (int)live) {
                            const uint64_t base = a_base + (uint64_t)t * a_tok_stride;
                            float ascale;
                            float blk = 0.0f;
                            if constexpr (SRC == OPT_G_WIDE) {
                                uint32_t ap[16];
                                opt_g_act_f16_pairs(s_mem + t * wide_bytes, kb, ap);
                                ascale = reinterpret_cast<const float *>(s_mem + wide_scale_off)[t * k_blocks + kb];
#pragma unroll
                                for (int j = 0; j < 16; ++j) blk = opt_g_fma2_f16(ap[j], wp[j], blk);
                            } else {
                                const uint8_t *row = STAGED ? s_mem + t * row_bytes
                                                            : reinterpret_cast<const uint8_t *>(a) + base * EB;
                                const uint8_t *srow = STAGED ? s_mem + scale_off + t * k_blocks : a_scales + base / 32u;
                                if constexpr (PAIRS) {
                                    uint32_t ap[16];
                                    opt_g_act_fp8_pairs<STAGED>(row, srow, kb, ap, ascale);
#pragma unroll
                                    for (int j = 0; j < 16; ++j) blk = opt_g_fma2_f16(ap[j], wp[j], blk);
                                } else {
                                    float av[32];
                                    opt_g_act<AMODE, STAGED>(row, srow, kb, av, ascale);
#pragma unroll
                                    for (int e = 0; e < 32; ++e) blk = __fmaf_rn(av[e], wf[e], blk);
                                }
                            }
                            acc[t] = __fmaf_rn(blk, wscale * ascale, acc[t]);
                        }
                    }
                }
            }
        }
    };
    if constexpr (AMODE == OPT_A_FP8) {
        if (wide) {
            main_loop(opt_g_tag<OPT_G_WIDE>{});
        } else if (staged) {
            main_loop(opt_g_tag<OPT_G_STAGED>{});
        } else {
            main_loop(opt_g_tag<OPT_G_GLOBAL>{});
        }
    } else if (staged) {
        main_loop(opt_g_tag<OPT_G_STAGED>{});
    } else {
        main_loop(opt_g_tag<OPT_G_GLOBAL>{});
    }

    // The warp partials reuse the staging bytes once every warp is done with the rows.
    float (*red)[MAXT][32] = reinterpret_cast<float (*)[MAXT][32]>(s_mem);
    if (staged) __syncthreads();
#pragma unroll
    for (int t = 0; t < MAXT; ++t) red[warp][t][lane] = acc[t];
    __syncthreads();
    for (uint32_t idx = threadIdx.x; idx < live * 32u; idx += OPT_G_THREADS) {
        const uint32_t t = idx >> 5;
        const uint32_t c = idx & 31u;
        float s = 0.0f;
#pragma unroll
        for (uint32_t q = 0; q < OPT_G_WARPS; ++q) s += red[q][t][c];
        if (tile * 32u + c >= cols) continue;
        const uint64_t o = group * out_grp_stride + ((uint64_t)t + z0) * out_tok_stride + (uint64_t)tile * 32u + c;
        if constexpr (OUT_F32) {
            reinterpret_cast<float *>(out)[o] = s;
        } else {
            reinterpret_cast<__nv_bfloat16 *>(out)[o] = __float2bfloat16_rn(s);
        }
    }
}

#define OPT_GEMV_ENTRY(name, AMODE, OUTF32, MAXT)                                                   \
    extern "C" __global__ void __launch_bounds__(OPT_G_THREADS) name(                              \
        const void *__restrict__ a, const uint8_t *__restrict__ a_scales,                            \
        const uint8_t *__restrict__ w, const uint8_t *__restrict__ ws, void *__restrict__ out,     \
        const uint32_t cols, const uint32_t inner, const uint32_t tokens,                          \
        const uint64_t a_tok_stride, const uint64_t out_tok_stride, const uint64_t a_grp_stride,   \
        const uint64_t w_grp_stride, const uint64_t s_grp_stride, const uint64_t out_grp_stride,  \
        const uint32_t k_stride_blocks) {                                                          \
        OPT_PDL_ENTRY();                                                                           \
        opt_gemv_body<AMODE, OUTF32, MAXT>(a, a_scales, w, ws, out, cols, inner, tokens,           \
                                           a_tok_stride, out_tok_stride, a_grp_stride,             \
                                           w_grp_stride, s_grp_stride, out_grp_stride,             \
                                           k_stride_blocks);                                       \
    }

OPT_GEMV_ENTRY(opt_gemv_bf16a_bf16o_t1, OPT_A_BF16, false, 1)
OPT_GEMV_ENTRY(opt_gemv_fp8a_bf16o_t1, OPT_A_FP8, false, 1)
OPT_GEMV_ENTRY(opt_gemv_f32a_f32o_t1, OPT_A_F32, true, 1)
OPT_GEMV_ENTRY(opt_gemv_f32a_bf16o_t1, OPT_A_F32, false, 1)
OPT_GEMV_ENTRY(opt_gemv_fp8a_f32o_t1, OPT_A_FP8, true, 1)
OPT_GEMV_ENTRY(opt_gemv_bf16a_bf16o_t2, OPT_A_BF16, false, 2)
OPT_GEMV_ENTRY(opt_gemv_fp8a_bf16o_t2, OPT_A_FP8, false, 2)
OPT_GEMV_ENTRY(opt_gemv_f32a_f32o_t2, OPT_A_F32, true, 2)
OPT_GEMV_ENTRY(opt_gemv_f32a_bf16o_t2, OPT_A_F32, false, 2)
OPT_GEMV_ENTRY(opt_gemv_bf16a_bf16o_t4, OPT_A_BF16, false, 4)
OPT_GEMV_ENTRY(opt_gemv_fp8a_bf16o_t4, OPT_A_FP8, false, 4)
OPT_GEMV_ENTRY(opt_gemv_f32a_f32o_t4, OPT_A_F32, true, 4)
OPT_GEMV_ENTRY(opt_gemv_f32a_bf16o_t4, OPT_A_F32, false, 4)
OPT_GEMV_ENTRY(opt_gemv_bf16a_bf16o_t8, OPT_A_BF16, false, 8)
OPT_GEMV_ENTRY(opt_gemv_fp8a_bf16o_t8, OPT_A_FP8, false, 8)
OPT_GEMV_ENTRY(opt_gemv_fp8a_f32o_t8, OPT_A_FP8, true, 8)
OPT_GEMV_ENTRY(opt_gemv_f32a_f32o_t8, OPT_A_F32, true, 8)
OPT_GEMV_ENTRY(opt_gemv_f32a_bf16o_t8, OPT_A_F32, false, 8)

// ---------------------------------------------------------------------------------------------
// Rows GEMV (a few to a few dozen tokens): the same per-token chains, cheaper per token.
// ---------------------------------------------------------------------------------------------
//
// opt_gemv_rows_*: grid (ceil(cols / (32 * CPL)), groups, ceil(tokens / TB)), 256 threads. A block
// serves CPL 32-column tiles (lane l: column l of each) and up to TB tokens, numbered token split
// fastest as above. K goes eight blocks at a time, one a warp (warp w: K blocks w, w + 8, ... as in
// opt_gemv_body), and the chunk's activations for the block's tokens are staged in shared memory
// (two buffers: the next chunk is loaded while this one computes): E4M3 widened to f16 pairs with
// its scales as f32, bf16 pairs and f32 as they are. A warp converts its K block's weights once for
// all the block's tokens, and each staged activation it reads serves all CPL columns, where the
// t1..t8 forms convert a tile's weights once per eight tokens at most, read a token's activations
// once per column and stage whole rows before the first product.
//
// Every token's chain is opt_gemv_body's: the K block's 32 products in ascending order, each one
// rounding (E4M3 and bf16 pairs on the mixed-precision FMA, whose products are exact in f32 like
// those of the f32 FMA on the widened values; f32 rows on the f32 FMA), the chain over the warp's
// K blocks, the eight warp partials summed in warp order. So the bits are the same.
#define OPT_GR_CHUNK OPT_G_WARPS

// c + a.lo * b.lo, then + a.hi * b.hi, on bf16x2 pairs: two single-rounding f32 FMAs.
__device__ __forceinline__ float opt_g_fma2_bf16(uint32_t a, uint32_t b, float c) {
    asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(c) : "h"((uint16_t)(a & 0xffffu)), "h"((uint16_t)(b & 0xffffu)));
    asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(c) : "h"((uint16_t)(a >> 16)), "h"((uint16_t)(b >> 16)));
    return c;
}

// Two E4M3 bytes (the low 16 bits) -> bf16x2, the low byte in the low half (E4M3 is exact in bf16).
__device__ __forceinline__ uint32_t opt_g_e4m3x2_bf16(uint32_t two) {
    const uint32_t h = opt_g_e4m3x2(two);
    const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(&h));
    uint32_t b;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b) : "f"(f.y), "f"(f.x));
    return b;
}

template <int AMODE, bool OUT_F32, uint32_t TB, uint32_t CPL>
__device__ __forceinline__ void opt_gemv_rows_body(
    const uint8_t *__restrict__ a, const uint8_t *__restrict__ a_scales,
    const uint8_t *__restrict__ w, const uint8_t *__restrict__ ws, void *__restrict__ out,
    const uint32_t cols, const uint32_t inner, const uint32_t tokens,
    const uint64_t a_tok_stride, const uint64_t out_tok_stride,
    const uint64_t a_grp_stride, const uint64_t w_grp_stride, const uint64_t s_grp_stride,
    const uint64_t out_grp_stride, const uint32_t k_stride_blocks) {
    constexpr bool FP8 = AMODE == OPT_A_FP8;
    constexpr bool F32 = AMODE == OPT_A_F32;
    constexpr uint32_t EB = AMODE == OPT_A_BF16 ? 2u : FP8 ? 1u : 4u;
    // Staged words of a token's K block: 16 f16 or bf16 pairs, or 32 f32.
    constexpr uint32_t KW = F32 ? 32u : 16u;
    // 16-byte source loads per token per chunk, and per thread.
    constexpr uint32_t UPT = OPT_GR_CHUNK * 2u * EB;
    constexpr uint32_t NU = (TB * UPT + OPT_G_THREADS - 1u) / OPT_G_THREADS;
    // One staged chunk: [token][K block][KW words], then (E4M3) [token][K block] f32 scales.
    constexpr uint32_t ACT_WORDS = TB * OPT_GR_CHUNK * KW;
    constexpr uint32_t BUF_WORDS = ACT_WORDS + (FP8 ? TB * OPT_GR_CHUNK : 0u);
    constexpr uint32_t RED_WORDS = OPT_G_WARPS * TB * 32u * CPL;
    constexpr uint32_t SMEM_WORDS = RED_WORDS > 2u * BUF_WORDS ? RED_WORDS : 2u * BUF_WORDS;
    static_assert(!FP8 || TB * OPT_GR_CHUNK <= OPT_G_THREADS, "one scale a thread a chunk");
    __shared__ __align__(16) uint32_t s_mem[SMEM_WORDS];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    // Token split fastest: consecutive blocks are the splits of one tile group.
    const uint32_t lin = blockIdx.z * gridDim.x + blockIdx.x;
    const uint32_t pair = lin / gridDim.z;
    const uint32_t group = blockIdx.y;
    const uint32_t tiles = cols >> 5;
    const uint32_t k_blocks = inner >> 5;
    const uint32_t z0 = (lin - pair * gridDim.z) * TB;
    const uint32_t live = tokens - z0 < TB ? tokens - z0 : TB;
    const uint32_t k_row = k_stride_blocks != 0u ? k_stride_blocks : k_blocks;
    const uint64_t a_base = group * a_grp_stride + (uint64_t)z0 * a_tok_stride;

    const uint8_t *wt[CPL];
    const uint8_t *st[CPL];
    bool have[CPL];
#pragma unroll
    for (uint32_t c = 0; c < CPL; ++c) {
        const uint32_t tile = pair * CPL + c;
        have[c] = tile < tiles;
        const uint32_t t_ = have[c] ? tile : 0u;
        wt[c] = w + group * w_grp_stride + (uint64_t)t_ * k_row * 1024u + lane * 32u;
        st[c] = ws + group * s_grp_stride + (uint64_t)t_ * k_row;
    }

    // Staging: load u (u = tid + 256 i) is 16 bytes of token u / UPT's chunk, at byte 16 * (u % UPT)
    // of it (K block 16 * (u % UPT) / (32 * EB)); thread i < TB * 8 also loads the E4M3 scale of
    // token i / 8, K block i % 8.
    auto stage_load = [&](uint32_t chunk, uint4 (&u)[NU], uint8_t &sc) {
#pragma unroll
        for (uint32_t i = 0; i < NU; ++i) {
            const uint32_t idx = tid + i * OPT_G_THREADS;
            const uint32_t t = idx / UPT;
            const uint32_t r = idx - t * UPT;
            const uint32_t kb = chunk * OPT_GR_CHUNK + r * 16u / (32u * EB);
            u[i] = make_uint4(0u, 0u, 0u, 0u);
            if (idx < TB * UPT && t < live && kb < k_blocks) {
                const uint8_t *src = a + (a_base + (uint64_t)t * a_tok_stride) * EB +
                                     (uint64_t)chunk * OPT_GR_CHUNK * 32u * EB + r * 16u;
                u[i] = __ldg(reinterpret_cast<const uint4 *>(src));
            }
        }
        sc = 0u;
        if constexpr (FP8) {
            const uint32_t kb = chunk * OPT_GR_CHUNK + (tid & 7u);
            if (tid < TB * OPT_GR_CHUNK && (tid >> 3) < live && kb < k_blocks) {
                sc = __ldg(a_scales + (a_base + (uint64_t)(tid >> 3) * a_tok_stride) / 32u + kb);
            }
        }
    };
    auto stage_store = [&](uint32_t buf, const uint4 (&u)[NU], uint8_t sc) {
        uint32_t *base = s_mem + buf * BUF_WORDS;
#pragma unroll
        for (uint32_t i = 0; i < NU; ++i) {
            const uint32_t idx = tid + i * OPT_G_THREADS;
            if (idx < TB * UPT) {
                const uint32_t t = idx / UPT;
                const uint32_t r = idx - t * UPT;
                if constexpr (FP8) {
                    uint4 *dst = reinterpret_cast<uint4 *>(base + t * OPT_GR_CHUNK * KW + r * 8u);
                    dst[0] = make_uint4(opt_g_e4m3x2(u[i].x), opt_g_e4m3x2(u[i].x >> 16), opt_g_e4m3x2(u[i].y),
                                        opt_g_e4m3x2(u[i].y >> 16));
                    dst[1] = make_uint4(opt_g_e4m3x2(u[i].z), opt_g_e4m3x2(u[i].z >> 16), opt_g_e4m3x2(u[i].w),
                                        opt_g_e4m3x2(u[i].w >> 16));
                } else {
                    *reinterpret_cast<uint4 *>(base + t * OPT_GR_CHUNK * KW + r * 4u) = u[i];
                }
            }
        }
        if constexpr (FP8) {
            if (tid < TB * OPT_GR_CHUNK) {
                reinterpret_cast<float *>(base + ACT_WORDS)[tid] = opt_g_ue8m0(sc);
            }
        }
    };

    float acc[TB][CPL];
#pragma unroll
    for (uint32_t t = 0; t < TB; ++t) {
#pragma unroll
        for (uint32_t c = 0; c < CPL; ++c) acc[t][c] = 0.0f;
    }

    const uint32_t chunks = (k_blocks + OPT_GR_CHUNK - 1u) / OPT_GR_CHUNK;
    // A missing second tile (odd tile count) converts zeros and stores nothing.
    uint4 lo[CPL], hi[CPL];
    uint8_t sc[CPL];
#pragma unroll
    for (uint32_t c = 0; c < CPL; ++c) {
        lo[c] = make_uint4(0u, 0u, 0u, 0u);
        hi[c] = lo[c];
        sc[c] = 0u;
    }
    auto load_w = [&](uint32_t chunk) {
        const uint32_t kb = chunk * OPT_GR_CHUNK + warp;
#pragma unroll
        for (uint32_t c = 0; c < CPL; ++c) {
            if (have[c] && kb < k_blocks) {
                lo[c] = opt_g_ldw(wt[c] + (uint64_t)kb * 1024u);
                hi[c] = opt_g_ldw(wt[c] + (uint64_t)kb * 1024u + 16u);
                sc[c] = __ldg(st[c] + kb);
            }
        }
    };
    load_w(0u);
    {
        uint4 u[NU];
        uint8_t s8;
        stage_load(0u, u, s8);
        stage_store(0u, u, s8);
    }
    __syncthreads();

    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        const uint32_t kb = chunk * OPT_GR_CHUNK + warp;
        const bool more = chunk + 1u < chunks;
        uint4 nu[NU];
        uint8_t ns;
        if (more) stage_load(chunk + 1u, nu, ns);
        if (kb < k_blocks) {
            // The weights as f16 or bf16 pairs, or as 32 f32.
            uint32_t wp[CPL][F32 ? 1 : 16];
            float wf[CPL][F32 ? 32 : 1];
            float wscale[CPL];
#pragma unroll
            for (uint32_t c = 0; c < CPL; ++c) {
                const uint32_t ww[8] = {lo[c].x, lo[c].y, lo[c].z, lo[c].w, hi[c].x, hi[c].y, hi[c].z, hi[c].w};
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    if constexpr (F32) {
                        opt_g_e4m3x4(ww[q], wf[c][4 * q], wf[c][4 * q + 1], wf[c][4 * q + 2], wf[c][4 * q + 3]);
                    } else if constexpr (FP8) {
                        wp[c][2 * q] = opt_g_e4m3x2(ww[q]);
                        wp[c][2 * q + 1] = opt_g_e4m3x2(ww[q] >> 16);
                    } else {
                        wp[c][2 * q] = opt_g_e4m3x2_bf16(ww[q]);
                        wp[c][2 * q + 1] = opt_g_e4m3x2_bf16(ww[q] >> 16);
                    }
                }
                wscale[c] = opt_g_ue8m0(sc[c]);
            }
            if (more) load_w(chunk + 1u);
            const uint32_t *buf = s_mem + (chunk & 1u) * BUF_WORDS;
#pragma unroll
            for (uint32_t t = 0; t < TB; ++t) {
                if (t < live) {
                    const uint4 *src = reinterpret_cast<const uint4 *>(buf + (t * OPT_GR_CHUNK + warp) * KW);
                    float blk[CPL];
#pragma unroll
                    for (uint32_t c = 0; c < CPL; ++c) blk[c] = 0.0f;
                    if constexpr (F32) {
#pragma unroll
                        for (int q = 0; q < 8; ++q) {
                            const uint4 v = src[q];
                            const float av[4] = {__uint_as_float(v.x), __uint_as_float(v.y), __uint_as_float(v.z),
                                                 __uint_as_float(v.w)};
#pragma unroll
                            for (int e = 0; e < 4; ++e) {
#pragma unroll
                                for (uint32_t c = 0; c < CPL; ++c) blk[c] = __fmaf_rn(av[e], wf[c][4 * q + e], blk[c]);
                            }
                        }
                    } else {
                        uint32_t ap[16];
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const uint4 v = src[q];
                            ap[4 * q] = v.x; ap[4 * q + 1] = v.y; ap[4 * q + 2] = v.z; ap[4 * q + 3] = v.w;
                        }
#pragma unroll
                        for (uint32_t c = 0; c < CPL; ++c) {
#pragma unroll
                            for (int j = 0; j < 16; ++j) {
                                if constexpr (FP8) {
                                    blk[c] = opt_g_fma2_f16(ap[j], wp[c][j], blk[c]);
                                } else {
                                    blk[c] = opt_g_fma2_bf16(ap[j], wp[c][j], blk[c]);
                                }
                            }
                        }
                    }
                    float ascale = 1.0f;
                    if constexpr (FP8) {
                        ascale = reinterpret_cast<const float *>(buf + ACT_WORDS)[t * OPT_GR_CHUNK + warp];
                    }
#pragma unroll
                    for (uint32_t c = 0; c < CPL; ++c) acc[t][c] = __fmaf_rn(blk[c], wscale[c] * ascale, acc[t][c]);
                }
            }
        }
        if (more) stage_store((chunk + 1u) & 1u, nu, ns);
        __syncthreads();
    }

    // The warp partials reuse the staging words; the loop's last barrier freed them.
    float *red = reinterpret_cast<float *>(s_mem);
#pragma unroll
    for (uint32_t t = 0; t < TB; ++t) {
#pragma unroll
        for (uint32_t c = 0; c < CPL; ++c) {
            red[(warp * TB + t) * (32u * CPL) + c * 32u + lane] = acc[t][c];
        }
    }
    __syncthreads();
    for (uint32_t idx = tid; idx < live * 32u * CPL; idx += OPT_G_THREADS) {
        const uint32_t t = idx / (32u * CPL);
        const uint32_t cc = idx % (32u * CPL);
        const uint32_t tile = pair * CPL + cc / 32u;
        if (tile >= tiles) continue;
        float s = 0.0f;
#pragma unroll
        for (uint32_t q = 0; q < OPT_G_WARPS; ++q) s += red[(q * TB + t) * (32u * CPL) + cc];
        const uint64_t o = group * out_grp_stride + ((uint64_t)t + z0) * out_tok_stride + (uint64_t)tile * 32u + cc % 32u;
        if constexpr (OUT_F32) {
            reinterpret_cast<float *>(out)[o] = s;
        } else {
            reinterpret_cast<__nv_bfloat16 *>(out)[o] = __float2bfloat16_rn(s);
        }
    }
}

#define OPT_GEMV_ROWS_ENTRY(name, AMODE, OUTF32, TB, CPL, MINB)                                     \
    extern "C" __global__ void __launch_bounds__(OPT_G_THREADS, MINB) name(                        \
        const void *__restrict__ a, const uint8_t *__restrict__ a_scales,                            \
        const uint8_t *__restrict__ w, const uint8_t *__restrict__ ws, void *__restrict__ out,     \
        const uint32_t cols, const uint32_t inner, const uint32_t tokens,                          \
        const uint64_t a_tok_stride, const uint64_t out_tok_stride, const uint64_t a_grp_stride,   \
        const uint64_t w_grp_stride, const uint64_t s_grp_stride, const uint64_t out_grp_stride,  \
        const uint32_t k_stride_blocks) {                                                          \
        OPT_PDL_ENTRY();                                                                           \
        opt_gemv_rows_body<AMODE, OUTF32, TB, CPL>(reinterpret_cast<const uint8_t *>(a), a_scales,  \
                                                   w, ws, out, cols, inner, tokens, a_tok_stride,  \
                                                   out_tok_stride, a_grp_stride, w_grp_stride,     \
                                                   s_grp_stride, out_grp_stride, k_stride_blocks); \
    }

// Three shapes of block for E4M3 rows, by sweep over the decode projections at 2 to 32 tokens: eight
// tokens and two tiles (wide matrices), eight tokens and one tile, four tokens and one tile (narrow
// matrices, three blocks an SM).
OPT_GEMV_ROWS_ENTRY(opt_gemv_rows_fp8a_bf16o_r8c2, OPT_A_FP8, false, 8u, 2u, 2)
OPT_GEMV_ROWS_ENTRY(opt_gemv_rows_fp8a_bf16o_r8c1, OPT_A_FP8, false, 8u, 1u, 3)
OPT_GEMV_ROWS_ENTRY(opt_gemv_rows_fp8a_bf16o_r4c1, OPT_A_FP8, false, 4u, 1u, 3)
// bf16 rows (wo_a): eight tokens and two tiles or one. f32 rows (the shared expert): four tokens and
// two tiles (the wide down projection) or one (gate/up).
OPT_GEMV_ROWS_ENTRY(opt_gemv_rows_bf16a_bf16o_r8c2, OPT_A_BF16, false, 8u, 2u, 2)
OPT_GEMV_ROWS_ENTRY(opt_gemv_rows_bf16a_bf16o_r8c1, OPT_A_BF16, false, 8u, 1u, 3)
OPT_GEMV_ROWS_ENTRY(opt_gemv_rows_f32a_f32o_r4c2, OPT_A_F32, true, 4u, 2u, 2)
OPT_GEMV_ROWS_ENTRY(opt_gemv_rows_f32a_f32o_r4c1, OPT_A_F32, true, 4u, 1u, 3)

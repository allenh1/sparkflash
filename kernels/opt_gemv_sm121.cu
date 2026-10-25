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

template <bool B>
struct opt_g_tag {
    static constexpr bool value = B;
};

template <int AMODE, bool OUT_F32, int MAXT>
__device__ __forceinline__ void opt_gemv_body(
    const void *__restrict__ a, const uint8_t *__restrict__ a_scales,
    const uint8_t *__restrict__ w, const uint8_t *__restrict__ ws, void *__restrict__ out,
    const uint32_t cols, const uint32_t inner, const uint32_t tokens,
    const uint64_t a_tok_stride, const uint64_t out_tok_stride,
    const uint64_t a_grp_stride, const uint64_t w_grp_stride, const uint64_t s_grp_stride,
    const uint64_t out_grp_stride, const uint32_t k_stride_blocks) {
    // E4M3 activations with several tokens: f16x2 operands and the mixed-precision FMA (one token
    // is faster on the f32 chain).
    constexpr bool PAIRS = AMODE == OPT_A_FP8 && MAXT > 1;
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
    const uint32_t row_bytes = inner * EB;
    const uint32_t scale_off = live * row_bytes;
    const uint32_t need = scale_off + (AMODE == OPT_A_FP8 ? live * k_blocks : 0u);
    const bool staged = need <= STAGE;
    if (staged) {
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
        constexpr bool STAGED = decltype(tag)::value;
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
                            const uint8_t *row = STAGED ? s_mem + t * row_bytes
                                                        : reinterpret_cast<const uint8_t *>(a) + base * EB;
                            const uint8_t *srow = STAGED ? s_mem + scale_off + t * k_blocks : a_scales + base / 32u;
                            float ascale;
                            float blk = 0.0f;
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
                            acc[t] = __fmaf_rn(blk, wscale * ascale, acc[t]);
                        }
                    }
                }
            }
        }
    };
    if (staged) {
        main_loop(opt_g_tag<true>{});
    } else {
        main_loop(opt_g_tag<false>{});
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

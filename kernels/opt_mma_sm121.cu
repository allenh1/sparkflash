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

// kernels/opt_mma_sm121.cu
//
// Tensor-core GEMMs for the bulk prefill: a chunk of many tokens through one matrix per launch,
// with mma.sync instead of the decode GEMV's CUDA-core FMAs. The decode GEMV (opt_gemv_sm121.cu)
// re-reads the weights once per eight tokens; these read each weight tile once per 128 tokens and
// run the products on the tensor cores, so a chunk costs about one pass over the weights.
//
// The arithmetic is not the GEMV's: the block-scaled MMA sums a K-block's 32 products in its own
// order and the K-blocks are folded in ascending order into one f32 accumulator per output, so the
// result matches the GEMV to accumulation-order rounding, not bit for bit. The bulk prefill is
// gated on prompt length for that reason; the exact rows path stays the decode reference.
//
// opt_mma_fp8_{bf16o,f32o}: E4M3 activations (one ue8m0 per 32 along K) x B12x E4M3 weights (32x32
//   tiles, [n][k] inside a tile, one ue8m0 per tile), mma.sync m16n8k32 kind::mxf8f6f4 with the
//   scales applied by the MMA. The opt GEMV's argument list (OptArgs), any token count.
//   Grid (ceil(tokens / 128), ceil(cols / 128), groups), 256 threads. A block owns a 128-token x
//   128-column tile and walks K two MX blocks per stage through a double-buffered cp.async
//   pipeline (the scale bytes through registers a stage ahead); warp (wm, wn) computes 64 tokens x
//   one 32-column weight tile (4 x 4 MMA tiles).
//   Blocks with the same column tile are adjacent in launch order, so a weight tile is read from
//   DRAM about once while the activation rows stay in L2.
// opt_mma_fp8_rowscale_f32o: the same over row-major E4M3 weights with one ue8m0 per row per 32
//   along K (the MX layout of opt_draft_head_fp8's copy of the head), f32 out: the drafter's head
//   logits, which need not match the GEMV's bits.

#include "opt_pdl_sm121.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define OM_THREADS 256u
#define OM_BM 128u
#define OM_BN 128u
#define OM_KB 2u                 // MX blocks per stage
#define OM_A_ROW (OM_KB * 32u)   // activation bytes per row per stage (E4M3)
#define OM_W_TILES ((OM_BN / 32u) * OM_KB)

// 2^(code - 127) as f32; code 0 is 2^-127 (subnormal).
__device__ __forceinline__ float opt_ue8m0_f32(uint8_t code) {
    return code == 0u ? __uint_as_float(0x00400000u) : __uint_as_float((uint32_t)code << 23);
}

__device__ __forceinline__ uint32_t om_smem(const void *p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

// 16 bytes global -> shared, zero-filled when !valid (the source is then not read).
__device__ __forceinline__ void om_cp16(uint32_t dst, const void *src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void om_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void om_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ __forceinline__ void om_ldsm4(uint32_t addr, uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

// D += A x B with the ue8m0 scales: lane l supplies A row (l / 4) + 8 (l % 2) and B column l / 4.
__device__ __forceinline__ void om_mma_e4m3(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1,
                                            uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0),
          "h"((uint16_t)0), "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

// The activation stage is [128 rows][4 chunks of 16 B]; chunk c of row r lives at c ^ ((r / 2) % 4)
// so the eight rows an ldmatrix phase reads fall in eight bank groups.
__device__ __forceinline__ uint32_t om_a_off(uint32_t row, uint32_t chunk) {
    return row * OM_A_ROW + ((chunk ^ ((row >> 1) & 3u)) << 4);
}
// A weight tile is [32 rows (n)][2 chunks]; chunk h of row r lives at h ^ ((r / 4) % 2).
__device__ __forceinline__ uint32_t om_w_off(uint32_t tile, uint32_t row, uint32_t half) {
    return tile * 1024u + row * 32u + ((half ^ ((row >> 2) & 1u)) << 4);
}

// W_ROWS: row-major E4M3 weights ([cols][inner], the Engram wkv's) instead of B12x tiles; the
// scales are the same one per 32 x 32 block, or with W_RSCALE one per row per 32 along K
// ([cols][inner / 32]).
template <bool OUT_F32, bool W_ROWS = false, bool W_RSCALE = false>
__device__ __forceinline__ void opt_mma_fp8_body(
    const uint8_t *__restrict__ a, const uint8_t *__restrict__ a_scales, const uint8_t *__restrict__ w,
    const uint8_t *__restrict__ ws, void *__restrict__ out, const uint32_t cols, const uint32_t inner,
    const uint32_t tokens, const uint64_t a_tok_stride, const uint64_t out_tok_stride,
    const uint64_t a_grp_stride, const uint64_t w_grp_stride, const uint64_t s_grp_stride,
    const uint64_t out_grp_stride, const uint32_t k_stride_blocks) {
    __shared__ __align__(128) uint8_t s_a[2][OM_BM * OM_A_ROW];
    __shared__ __align__(128) uint8_t s_w[2][OM_W_TILES * 1024u];
    __shared__ uint8_t s_as[2][OM_BM * OM_KB];
    __shared__ uint8_t s_ws[2][W_RSCALE ? OM_BN * OM_KB : OM_W_TILES];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t m0 = blockIdx.x * OM_BM;
    const uint32_t n0 = blockIdx.y * OM_BN;
    const uint32_t group = blockIdx.z;
    const uint32_t k_blocks = inner >> 5;
    const uint32_t k_row = k_stride_blocks != 0u ? k_stride_blocks : k_blocks;
    const uint32_t col_tiles = min(OM_BN / 32u, (cols - n0) >> 5);
    const uint32_t stages = (k_blocks + OM_KB - 1u) / OM_KB;

    const uint8_t *a_grp = a + group * a_grp_stride;
    const uint8_t *as_grp = a_scales + group * a_grp_stride / 32u;
    const uint8_t *w_grp = w + group * w_grp_stride + (uint64_t)(n0 >> 5) * k_row * 1024u;
    const uint8_t *ws_grp = ws + group * s_grp_stride + (uint64_t)(n0 >> 5) * k_row;

    auto issue = [&](uint32_t stage, uint32_t buf) {
        const uint32_t kb0 = stage * OM_KB;
        // Activations: 128 rows x 4 chunks, two per thread.
#pragma unroll
        for (uint32_t i = 0; i < 2u; ++i) {
            const uint32_t idx = tid + i * OM_THREADS;
            const uint32_t row = idx >> 2;
            const uint32_t chunk = idx & 3u;
            const uint32_t token = m0 + row;
            const uint32_t kb = kb0 + (chunk >> 1);
            const bool valid = token < tokens && kb < k_blocks;
            const uint8_t *src = valid ? a_grp + (uint64_t)token * a_tok_stride + (uint64_t)kb0 * 32u + chunk * 16u : a;
            om_cp16(om_smem(&s_a[buf][om_a_off(row, chunk)]), src, valid);
        }
        // Weights: OM_W_TILES tiles x 64 chunks, two per thread.
#pragma unroll
        for (uint32_t i = 0; i < 2u; ++i) {
            const uint32_t idx = tid + i * OM_THREADS;
            const uint32_t tile = idx >> 6;
            const uint32_t within = idx & 63u;
            const uint32_t ct = tile / OM_KB;
            const uint32_t kt = tile % OM_KB;
            const bool valid = ct < col_tiles && kb0 + kt < k_blocks;
            const uint8_t *src = !valid ? w
                                 : W_ROWS ? w + (uint64_t)(n0 + ct * 32u + (within >> 1)) * inner + (kb0 + kt) * 32u + (within & 1u) * 16u
                                          : w_grp + ((uint64_t)ct * k_row + kb0 + kt) * 1024u + within * 16u;
            om_cp16(om_smem(&s_w[buf][om_w_off(tile, within >> 1, within & 1u)]), src, valid);
        }
    };
    // Scales by plain loads (a row's two bytes need not be 4-byte aligned), into registers a stage
    // ahead of their store, so the loads run under a stage of MMAs instead of before the barrier.
    auto load_scales = [&](uint32_t stage, uint8_t (&r)[OM_KB]) {
        const uint32_t kb0 = stage * OM_KB;
        if (tid < OM_BM) {
            const uint32_t token = m0 + tid;
#pragma unroll
            for (uint32_t j = 0; j < OM_KB; ++j) {
                const uint32_t kb = kb0 + j;
                r[j] = token < tokens && kb < k_blocks ? __ldg(as_grp + (uint64_t)token * a_tok_stride / 32u + kb) : (uint8_t)0;
            }
        } else if (W_RSCALE) {
            // Thread 128 + c loads column c's scales (OM_BN = 128 columns).
            const uint32_t col = tid - OM_BM;
#pragma unroll
            for (uint32_t j = 0; j < OM_KB; ++j) {
                const uint32_t kb = kb0 + j;
                r[j] = n0 + col < cols && kb < k_blocks ? __ldg(ws + (uint64_t)(n0 + col) * k_blocks + kb) : (uint8_t)0;
            }
        } else if (tid < OM_BM + OM_W_TILES) {
            const uint32_t tile = tid - OM_BM;
            const uint32_t ct = tile / OM_KB;
            const uint32_t kt = tile % OM_KB;
            const uint32_t kb = kb0 + kt;
            r[0] = ct < col_tiles && kb < k_blocks ? __ldg(ws_grp + (uint64_t)ct * k_row + kb) : (uint8_t)0;
        }
    };
    auto store_scales = [&](const uint8_t (&r)[OM_KB], uint32_t buf) {
        if (tid < OM_BM) {
#pragma unroll
            for (uint32_t j = 0; j < OM_KB; ++j) s_as[buf][tid * OM_KB + j] = r[j];
        } else if (W_RSCALE) {
#pragma unroll
            for (uint32_t j = 0; j < OM_KB; ++j) s_ws[buf][(tid - OM_BM) * OM_KB + j] = r[j];
        } else if (tid < OM_BM + OM_W_TILES) {
            s_ws[buf][tid - OM_BM] = r[0];
        }
    };

    const uint32_t wm = warp & 1u;   // rows 64 wm .. 64 wm + 63
    const uint32_t wn = warp >> 1;   // weight tile wn of the block's four
    const bool active = wn < col_tiles && m0 + wm * 64u < tokens;

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    uint8_t sc[OM_KB] = {};
    load_scales(0, sc);
    store_scales(sc, 0);
    issue(0, 0);
    om_commit();
    if (stages > 1u) load_scales(1, sc);
    for (uint32_t stage = 0; stage < stages; ++stage) {
        const uint32_t buf = stage & 1u;
        if (stage + 1u < stages) {
            store_scales(sc, buf ^ 1u);
            issue(stage + 1u, buf ^ 1u);
            om_commit();
            if (stage + 2u < stages) load_scales(stage + 2u, sc);
            om_wait<1>();
        } else {
            om_wait<0>();
        }
        __syncthreads();
        if (active) {
#pragma unroll
            for (uint32_t j = 0; j < OM_KB; ++j) {
                const uint32_t tile = wn * OM_KB + j;
                uint32_t b[4][2];
                {
                    const uint32_t sub = lane >> 4;        // n8 sub-tile 0/1 of the pair
                    const uint32_t row = (lane & 7u);
                    const uint32_t half = (lane >> 3) & 1u;
                    const uint32_t r01 = 8u * sub + row;
                    om_ldsm4(om_smem(&s_w[buf][om_w_off(tile, r01, half)]), b[0][0], b[0][1], b[1][0], b[1][1]);
                    const uint32_t r23 = 16u + 8u * sub + row;
                    om_ldsm4(om_smem(&s_w[buf][om_w_off(tile, r23, half)]), b[2][0], b[2][1], b[3][0], b[3][1]);
                }
                // A tile's scale, or column (lane / 4) of each n8 sub-tile's (the B scale lanes).
                uint32_t sfb[4];
#pragma unroll
                for (uint32_t nt = 0; nt < 4u; ++nt)
                    sfb[nt] = W_RSCALE ? s_ws[buf][(wn * 32u + nt * 8u + (lane >> 2)) * OM_KB + j] : s_ws[buf][tile];
#pragma unroll
                for (uint32_t mt = 0; mt < 4u; ++mt) {
                    const uint32_t rb = wm * 64u + mt * 16u;
                    uint32_t af[4];
                    const uint32_t row = rb + (lane & 7u) + ((lane >> 3) & 1u) * 8u;
                    const uint32_t chunk = 2u * j + (lane >> 4);
                    om_ldsm4(om_smem(&s_a[buf][om_a_off(row, chunk)]), af[0], af[1], af[2], af[3]);
                    const uint32_t sfa = s_as[buf][(rb + (lane >> 2) + 8u * (lane & 1u)) * OM_KB + j];
#pragma unroll
                    for (uint32_t nt = 0; nt < 4u; ++nt) om_mma_e4m3(acc[mt][nt], af, b[nt][0], b[nt][1], sfa, sfb[nt]);
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

#define OPT_MMA_FP8_ENTRY(name, OUTF32)                                                             \
    extern "C" __global__ void __launch_bounds__(OM_THREADS) name(                                 \
        const uint8_t *__restrict__ a, const uint8_t *__restrict__ a_scales,                       \
        const uint8_t *__restrict__ w, const uint8_t *__restrict__ ws, void *__restrict__ out,     \
        const uint32_t cols, const uint32_t inner, const uint32_t tokens,                          \
        const uint64_t a_tok_stride, const uint64_t out_tok_stride, const uint64_t a_grp_stride,   \
        const uint64_t w_grp_stride, const uint64_t s_grp_stride, const uint64_t out_grp_stride,  \
        const uint32_t k_stride_blocks) {                                                          \
        OPT_PDL_ENTRY();                                                                           \
        opt_mma_fp8_body<OUTF32>(a, a_scales, w, ws, out, cols, inner, tokens, a_tok_stride,       \
                                 out_tok_stride, a_grp_stride, w_grp_stride, s_grp_stride,         \
                                 out_grp_stride, k_stride_blocks);                                 \
    }

OPT_MMA_FP8_ENTRY(opt_mma_fp8_bf16o, false)
OPT_MMA_FP8_ENTRY(opt_mma_fp8_f32o, true)

// The Engram wkv over a chunk: row-major weights, bf16 out, the OptArgs list (groups 1).
extern "C" __global__ void __launch_bounds__(OM_THREADS) opt_mma_fp8_rows_bf16o(
    const uint8_t *__restrict__ a, const uint8_t *__restrict__ a_scales, const uint8_t *__restrict__ w,
    const uint8_t *__restrict__ ws, void *__restrict__ out, const uint32_t cols, const uint32_t inner,
    const uint32_t tokens, const uint64_t a_tok_stride, const uint64_t out_tok_stride) {
    OPT_PDL_ENTRY();
    opt_mma_fp8_body<false, true>(a, a_scales, w, ws, out, cols, inner, tokens, a_tok_stride, out_tok_stride,
                                  0u, 0u, 0u, 0u, 0u);
}

// The drafter's head logits: row-major weights with per-row scales, f32 out (groups 1).
extern "C" __global__ void __launch_bounds__(OM_THREADS) opt_mma_fp8_rowscale_f32o(
    const uint8_t *__restrict__ a, const uint8_t *__restrict__ a_scales, const uint8_t *__restrict__ w,
    const uint8_t *__restrict__ ws, float *__restrict__ out, const uint32_t cols, const uint32_t inner,
    const uint32_t tokens, const uint64_t a_tok_stride, const uint64_t out_tok_stride) {
    OPT_PDL_ENTRY();
    opt_mma_fp8_body<true, true, true>(a, a_scales, w, ws, out, cols, inner, tokens, a_tok_stride,
                                       out_tok_stride, 0u, 0u, 0u, 0u, 0u);
}

// ---- f32 and bf16 activations: the same B12x E4M3 weights on the bf16 tensor cores.
//
// opt_mma_{f32a_f32o, f32a_bf16o, bf16a_bf16o}: the opt GEMV's f32- and bf16-activation kinds
//   (OptArgs) for any token count, with mma.sync m16n8k16 bf16 x bf16 into f32. A weight tile's
//   E4M3 codes times its ue8m0 scale are exact in bf16 (four significant bits, the f32 exponent
//   range), so the weights are decoded once per stage into shared memory. An f32 activation is
//   split into a bf16 head and the bf16 of the remainder, two MMAs per tile, which carries about
//   16 of its 24 significant bits; a bf16 activation is one MMA.
//   Grid (ceil(cols / 128), ceil(tokens / 128), groups), 256 threads: blocks with the same token
//   tile are adjacent in launch order, so the (wider) activation rows are read from DRAM about once
//   and the FP8 weight tiles come from L2. A block owns 128 tokens x 128 columns and walks K one MX
//   block per stage: the next stage's operands are loaded into registers while the current one
//   computes, then converted into the single shared buffer. Warp (wm, wn) computes 64 tokens x 32
//   columns (4 x 4 MMA tiles per 16-wide K step).

#define OB_A_BF16 0
#define OB_A_F32 2
#define OB_W_FP8 0    // B12x E4M3 tiles + one ue8m0 per tile
#define OB_W_BF16 1   // host weights: the same tiling, bf16 elements, no scales
#define OB_W_F32 2    // host weights, f32 elements (split like an f32 activation)
#define OB_W_ROW 3    // plain row-major bf16 [cols][inner] (the router gate)
#define OB_W_FP8ROW 4 // row-major E4M3 [cols][inner], one ue8m0 per row per 32 along K (the drafter head)
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
    const uint8_t *w_grp = WMODE == OB_W_ROW || WMODE == OB_W_FP8ROW ? w + group * w_grp_stride
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
        } else if constexpr (WMODE == OB_W_BF16 || WMODE == OB_W_ROW) {
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

// The OptArgs list; a_scales is unused (the f32 and bf16 kinds carry none).
#define OPT_MMA_BF16_ENTRY(name, AMODE, OUTF32)                                                     \
    extern "C" __global__ void __launch_bounds__(OM_THREADS) name(                                 \
        const void *__restrict__ a, const uint8_t *__restrict__ a_scales,                          \
        const uint8_t *__restrict__ w, const uint8_t *__restrict__ ws, void *__restrict__ out,     \
        const uint32_t cols, const uint32_t inner, const uint32_t tokens,                          \
        const uint64_t a_tok_stride, const uint64_t out_tok_stride, const uint64_t a_grp_stride,   \
        const uint64_t w_grp_stride, const uint64_t s_grp_stride, const uint64_t out_grp_stride,  \
        const uint32_t k_stride_blocks) {                                                          \
        OPT_PDL_ENTRY();                                                                           \
        (void)a_scales;                                                                            \
        opt_mma_bf16_body<AMODE, OB_W_FP8, OUTF32>(a, w, ws, out, cols, inner, tokens, a_tok_stride,         \
                                         out_tok_stride, a_grp_stride, w_grp_stride, s_grp_stride, \
                                         out_grp_stride, k_stride_blocks);                         \
    }

OPT_MMA_BF16_ENTRY(opt_mma_f32a_f32o, OB_A_F32, true)
OPT_MMA_BF16_ENTRY(opt_mma_f32a_bf16o, OB_A_F32, false)
OPT_MMA_BF16_ENTRY(opt_mma_bf16a_bf16o, OB_A_BF16, false)
OPT_MMA_BF16_ENTRY(opt_mma_bf16a_f32o, OB_A_BF16, true)

// The drafter's head logits: bf16 rows x the row-major E4M3 head copy (opt_draft_head_fp8), f32
// out. Grid (ceil(cols / 128), ceil(tokens / 128)), 256 threads; cols a multiple of 32.
extern "C" __global__ void __launch_bounds__(OM_THREADS) opt_mma_bf16a_fp8row_f32o(
    const __nv_bfloat16 *__restrict__ a, const uint8_t *__restrict__ w, const uint8_t *__restrict__ ws,
    float *__restrict__ out, const uint32_t cols, const uint32_t inner, const uint32_t tokens,
    const uint64_t a_tok_stride, const uint64_t out_tok_stride) {
    OPT_PDL_ENTRY();
    opt_mma_bf16_body<OB_A_BF16, OB_W_FP8ROW, true>(a, w, ws, out, cols, inner, tokens, a_tok_stride, out_tok_stride,
                                                   0u, 0u, 0u, 0u, 0u);
}

// opt_mma_head_fp8t_f32o: opt_mma_bf16a_fp8row_f32o over the head copy in 128-column tiles
//   (opt_draft_head_fp8t: tile t, K block kb is 4 KB of codes, [128 columns][32], then its 128 scale
//   bytes in the scale copy), so a stage's weights are one contiguous read. A cp.async pipeline keeps
//   OH_STAGES - 1 stages of codes, scales and activation rows in flight; each stage's codes times
//   their scales are decoded to bf16 in shared memory and go through opt_mma_bf16_body's MMAs in its
//   order, so every logit is that kernel's bits. A block owns 64 rows x 128 columns: warp w the 16
//   columns 16 w .. 16 w + 15 of every row tile of 16 the block's rows reach.
//   Grid (ceil(tokens / 64), ceil(cols / 128)), 256 threads: a column tile's row blocks are adjacent
//   in launch order, so its codes come from DRAM about once.
#define OH_BM 64u
#define OH_STAGES 4u
#define OH_W_BYTES (OM_BN * 32u)        // a stage's codes
#define OH_A_BYTES (OH_BM * 32u * 2u)   // a stage's activation rows (bf16)

extern "C" __global__ void __launch_bounds__(OM_THREADS) opt_mma_head_fp8t_f32o(
    const __nv_bfloat16 *__restrict__ a, const uint8_t *__restrict__ w, const uint8_t *__restrict__ ws,
    float *__restrict__ out, const uint32_t cols, const uint32_t inner, const uint32_t tokens,
    const uint64_t a_tok_stride, const uint64_t out_tok_stride) {
    OPT_PDL_ENTRY();
    __shared__ __align__(16) uint8_t s_w8[OH_STAGES][OH_W_BYTES];
    __shared__ __align__(16) uint8_t s_a[OH_STAGES][OH_A_BYTES];
    __shared__ __align__(16) uint8_t s_s[OH_STAGES][OM_BN];
    __shared__ __align__(16) __nv_bfloat16 s_w[OM_BN * OB_LD];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t m0 = blockIdx.x * OH_BM;
    const uint32_t tile = blockIdx.y;
    const uint32_t n0 = tile * OM_BN;
    const uint32_t k_blocks = inner >> 5;
    const uint32_t row_tiles = min(OH_BM, tokens - m0 + 15u) >> 4;  // 16-row MMA tiles with a live row
    const uint8_t *w_tile = w + (uint64_t)tile * k_blocks * OH_W_BYTES;
    const uint8_t *s_tile = ws + (uint64_t)tile * k_blocks * OM_BN;

    // Thread t copies 16 bytes of codes (t of 256), 16 bytes of activation row t / 4 (chunk t % 4,
    // zero past the last row) and, for t < 8, 16 scale bytes.
    const uint32_t a_row = tid >> 2;
    const uint32_t a_chunk = tid & 3u;
    const bool a_live = m0 + a_row < tokens;
    const __nv_bfloat16 *a_src = a + (uint64_t)(a_live ? m0 + a_row : 0u) * a_tok_stride + a_chunk * 8u;
    auto load = [&](uint32_t kb) {
        const uint32_t slot = kb % OH_STAGES;
        om_cp16(om_smem(&s_w8[slot][tid * 16u]), w_tile + (uint64_t)kb * OH_W_BYTES + tid * 16u, true);
        om_cp16(om_smem(&s_a[slot][om_a_off(a_row, a_chunk)]), a_src + kb * 32u, a_live);
        if (tid < OM_BN / 16u) om_cp16(om_smem(&s_s[slot][tid * 16u]), s_tile + (uint64_t)kb * OM_BN + tid * 16u, true);
    };

    float acc[4][2][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

#pragma unroll
    for (uint32_t kb = 0; kb + 1u < OH_STAGES; ++kb) {
        if (kb < k_blocks) load(kb);
        om_commit();
    }
    // The decode of a stage: thread t the 16 codes of column t / 2, half t % 2 (opt_mma_bf16_body's
    // store of an FP8ROW stage).
    const uint32_t d_col = tid >> 1;
    const uint32_t d_half = tid & 1u;
    for (uint32_t kb = 0; kb < k_blocks; ++kb) {
        om_wait<OH_STAGES - 2u>();
        __syncthreads();
        if (kb + OH_STAGES - 1u < k_blocks) load(kb + OH_STAGES - 1u);
        om_commit();
        const uint32_t slot = kb % OH_STAGES;
        {
            const float scale = opt_ue8m0_f32(s_s[slot][d_col]);
            const uint4 r = *reinterpret_cast<const uint4 *>(&s_w8[slot][d_col * 32u + d_half * 16u]);
            __nv_bfloat16 *dst = &s_w[d_col * OB_LD + d_half * 16u];
            *reinterpret_cast<uint4 *>(dst) =
                make_uint4(ob_e4m3x2(r.x, scale), ob_e4m3x2(r.x >> 16, scale), ob_e4m3x2(r.y, scale), ob_e4m3x2(r.y >> 16, scale));
            *reinterpret_cast<uint4 *>(dst + 8) =
                make_uint4(ob_e4m3x2(r.z, scale), ob_e4m3x2(r.z >> 16, scale), ob_e4m3x2(r.w, scale), ob_e4m3x2(r.w >> 16, scale));
        }
        __syncthreads();
#pragma unroll
        for (uint32_t ks = 0; ks < 2u; ++ks) {
            const uint32_t k0 = ks * 16u;
            uint32_t b[2][2];
            {
                const uint32_t n = warp * 16u + (lane & 7u) + 8u * (lane >> 4);
                const uint32_t k = k0 + 8u * ((lane >> 3) & 1u);
                om_ldsm4(om_smem(&s_w[n * OB_LD + k]), b[0][0], b[0][1], b[1][0], b[1][1]);
            }
#pragma unroll
            for (uint32_t mt = 0; mt < 4u; ++mt) {
                if (mt < row_tiles) {
                    const uint32_t row = mt * 16u + (lane & 15u);
                    const uint32_t chunk = (k0 >> 3) + (lane >> 4);
                    uint32_t af[4];
                    om_ldsm4(om_smem(&s_a[slot][om_a_off(row, chunk)]), af[0], af[1], af[2], af[3]);
#pragma unroll
                    for (uint32_t nt = 0; nt < 2u; ++nt) ob_mma(acc[mt][nt], af, b[nt][0], b[nt][1]);
                }
            }
        }
    }
    om_wait<0>();

    const uint32_t g = lane >> 2;
    const uint32_t q = lane & 3u;
#pragma unroll
    for (uint32_t mt = 0; mt < 4u; ++mt) {
#pragma unroll
        for (uint32_t half = 0; half < 2u; ++half) {
            const uint32_t token = m0 + mt * 16u + g + 8u * half;
            if (mt >= row_tiles || token >= tokens) continue;
#pragma unroll
            for (uint32_t nt = 0; nt < 2u; ++nt) {
                const uint32_t col = n0 + warp * 16u + nt * 8u + 2u * q;
                if (col < cols) {
                    *reinterpret_cast<float2 *>(out + (uint64_t)token * out_tok_stride + col) =
                        make_float2(acc[mt][nt][2u * half], acc[mt][nt][2u * half + 1u]);
                }
            }
        }
    }
}

// opt_mma_hostw_{bf16w, bf16w_f32o, f32w_f32o}: the host-weight GEMVs (b12x_hostw_gemv_m1's three
//   modes: an f32 activation row against bf16 or f32 weights in the B12x tiling) over many rows.
//   Grid (ceil(cols / 128), ceil(tokens / 128)), 256 threads. f32 weights are split like the
//   activation: head x head + tail x head + head x tail. The bf16a forms take bf16 activation rows
//   (a sequence-parallel seam's gathered rows), the f32 forms' bits on bf16-exact values.
#define OPT_MMA_HOSTW_ENTRY(name, AMODE, A_T, WMODE, W_T, OUTF32)                                   \
    extern "C" __global__ void __launch_bounds__(OM_THREADS) name(                                 \
        const A_T *__restrict__ a, const W_T *__restrict__ w, void *__restrict__ out,             \
        const uint32_t cols, const uint32_t inner, const uint32_t tokens,                          \
        const uint64_t a_tok_stride, const uint64_t out_tok_stride) {                              \
        OPT_PDL_ENTRY();                                                                           \
        opt_mma_bf16_body<AMODE, WMODE, OUTF32>(a, reinterpret_cast<const uint8_t *>(w), nullptr, out, \
                                                cols, inner, tokens, a_tok_stride,                 \
                                                out_tok_stride, 0u, 0u, 0u, 0u, 0u);               \
    }

OPT_MMA_HOSTW_ENTRY(opt_mma_hostw_bf16w, OB_A_F32, float, OB_W_BF16, __nv_bfloat16, false)
OPT_MMA_HOSTW_ENTRY(opt_mma_hostw_bf16w_f32o, OB_A_F32, float, OB_W_BF16, __nv_bfloat16, true)
OPT_MMA_HOSTW_ENTRY(opt_mma_hostw_f32w_f32o, OB_A_F32, float, OB_W_F32, float, true)
OPT_MMA_HOSTW_ENTRY(opt_mma_hostw_bf16a_bf16w, OB_A_BF16, __nv_bfloat16, OB_W_BF16, __nv_bfloat16, false)
OPT_MMA_HOSTW_ENTRY(opt_mma_hostw_bf16a_bf16w_f32o, OB_A_BF16, __nv_bfloat16, OB_W_BF16, __nv_bfloat16, true)
OPT_MMA_HOSTW_ENTRY(opt_mma_hostw_bf16a_f32w_f32o, OB_A_BF16, __nv_bfloat16, OB_W_F32, float, true)

// The router over a chunk: opt_mma_router_dots writes the raw gate dots [tokens][experts] (bf16
//   x against the row-major bf16 gate, f32 accumulation; grid (experts / 128, ceil(tokens / 128))),
//   then opt_mma_router_topk applies opt_router_sm121's stage two per token: score =
//   sqrt(softplus(dot)), the biased picks (warp argmax, ties to the lower id), the unbiased scores
//   normalised in selection order (+ 1e-20) and scaled by 1.5. One warp per token, grid
//   ceil(tokens / 8), 256 threads.
extern "C" __global__ void __launch_bounds__(OM_THREADS) opt_mma_router_dots(
    const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ gate, float *__restrict__ out,
    const uint32_t cols, const uint32_t inner, const uint32_t tokens) {
    OPT_PDL_ENTRY();
    opt_mma_bf16_body<OB_A_BF16, OB_W_ROW, true>(x, reinterpret_cast<const uint8_t *>(gate), nullptr, out, cols,
                                                 inner, tokens, inner, cols, 0u, 0u, 0u, 0u, 0u);
}

#define OMR_EXPERTS 384u
#define OMR_TOPK 6u
extern "C" __global__ void __launch_bounds__(256) opt_mma_router_topk(
    const float *__restrict__ dots, const float *__restrict__ bias, uint32_t *__restrict__ ids,
    float *__restrict__ weights, const uint32_t tokens) {
    OPT_PDL_ENTRY();
    constexpr uint32_t PER_LANE = OMR_EXPERTS / 32u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t t = blockIdx.x * 8u + (threadIdx.x >> 5);
    if (t >= tokens) return;
    float sc[PER_LANE], sig[PER_LANE];
#pragma unroll
    for (uint32_t j = 0; j < PER_LANE; ++j) {
        const uint32_t e = lane + 32u * j;
        const float value = dots[(uint64_t)t * OMR_EXPERTS + e];
        const float sp = value > 20.0f ? value : log1pf(expf(value));
        sc[j] = sqrtf(sp);
        sig[j] = __fadd_rn(sc[j], bias[e]);
    }
    uint32_t chosen[OMR_TOPK];
    float picked[OMR_TOPK];
#pragma unroll
    for (uint32_t k = 0; k < OMR_TOPK; ++k) {
        float bv = sig[0];
        uint32_t bi = lane;
#pragma unroll
        for (uint32_t j = 1; j < PER_LANE; ++j) {
            if (sig[j] > bv) { bv = sig[j]; bi = lane + 32u * j; }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        float own = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < PER_LANE; ++j) {
            if (lane + 32u * j == bi) { own = sc[j]; sig[j] = -INFINITY; }
        }
        chosen[k] = bi;
        picked[k] = __shfl_sync(0xffffffffu, own, bi & 31u);
    }
    if (lane == 0u) {
        float sum = 0.0f;
#pragma unroll
        for (uint32_t s = 0; s < OMR_TOPK; ++s) sum = __fadd_rn(sum, picked[s]);
        const float den = __fadd_rn(sum, 1e-20f);
#pragma unroll
        for (uint32_t s = 0; s < OMR_TOPK; ++s) {
            ids[(uint64_t)t * OMR_TOPK + s] = chosen[s];
            weights[(uint64_t)t * OMR_TOPK + s] = __fmul_rn(picked[s] / den, 1.5f);
        }
    }
}

// ---- Routed experts: grouped W4A8 GEMMs over a chunk's routed rows.
//
// The rows (token * 6 + slot) are grouped by expert (opt_mma_moe_plan), and each block serves one
// expert's group of at most 64 rows against one column tile of that expert's slices, so an
// expert's weights are read about once per chunk instead of once per routed row. The MMA is
// kind::mxf8f6f4 E4M3 x E2M1: the FP4 weights are spread into the 8-bit containers the MMA reads
// (bits 5:2) in registers. Inside a 32-value MX block the K order is permuted the same way for
// both operands (lane q's two B registers hold packed bytes 4q..4q+3, i.e. values 8q..8q+7, and
// its A registers the matching activation bytes), which leaves every dot product unchanged and
// lets each lane read its operands as one 4-byte (B) and one 8-byte (A) shared load.
//
// opt_mma_moe_plan: one block of 1024 threads. Counts the rows per expert, lays the groups out in
//   expert order, splits each into items of <= 64 rows and scatters the row ids. items[i] =
//   (expert, first, count, 0); items_count[0] = the item count. With seg_tokens != 0 the tokens
//   are segments of seg_tokens (several chunks' rows, each at its segment's start) and only the
//   first seg_n[s] tokens of segment s are routed: the rest are gaps no item holds.
// opt_mma_moe_gate_up: grid (576 / 64, max items), 256 threads; block (x, y) = item y, intermediate
//   columns 64 x .. 64 x + 63, gate and up together. Epilogue: the opt_moe chain (bf16 gate and up,
//   clamp, SwiGLU, routing weight, bf16, MX quantiser) into codes [rows][576], scales [rows][18].
// opt_mma_moe_down: grid (5120 / 128, max items), 256 threads; block (x, y) = item y, output
//   columns 128 x .. 128 x + 127; writes each row's slot value rounded to bf16 into slot [rows][5120].
// opt_mma_moe_combine: grid (5120 / 1024, tokens), 256 threads of four columns; opt_moe's routed combine
//   (slots in ascending expert id, f32 adds, then the addend) into delta_bf16 and, when set, delta_f32.

#include "m4_common_sm121.cuh"
#include "m4_activation_quant_sm121.cuh"

#define OMM_HIDDEN 5120u
#define OMM_INTER 576u
#define OMM_TOPK 6u
#define OMM_ROWS 64u
#define OMM_GU_COLS 64u
#define OMM_D_COLS 128u
#define OMM_KB 4u                     // MX blocks per stage
#define OMM_A_ROW (OMM_KB * 32u)      // E4M3 bytes per row per stage
#define OMM_W_ROW (OMM_KB * 16u)      // packed FP4 bytes per column per stage

__device__ __forceinline__ void om_cp4(uint32_t dst, const void *src, bool valid) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(dst), "l"(src), "r"(valid ? 4 : 0));
}

__device__ __forceinline__ void om_mma_e4m3_e2m1(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1,
                                                 uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0),
          "h"((uint16_t)0), "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

// Four packed E2M1 values (16 bits, ascending nibbles) -> four 8-bit containers 0b00xxxx00.
__device__ __forceinline__ uint32_t om_spread_fp4(uint32_t x) {
    x &= 0xFFFFu;
    const uint32_t y = (x & 0x000Fu) | ((x & 0x00F0u) << 4) | ((x & 0x0F00u) << 8) | ((x & 0xF000u) << 12);
    return y << 2;
}

// A row of the stage holds OMM_KB 32-byte MX blocks; block j of row r lives at j ^ (r % 4).
__device__ __forceinline__ uint32_t omm_a_off(uint32_t row, uint32_t block) {
    return row * OMM_A_ROW + ((block ^ (row & 3u)) << 5);
}
// A column of the stage holds OMM_KB 16-byte packed blocks; block j of column c lives at
// j ^ ((c / 2) % 4).
__device__ __forceinline__ uint32_t omm_w_off(uint32_t col, uint32_t block) {
    return col * OMM_W_ROW + ((block ^ ((col >> 1) & 3u)) << 4);
}

#define OMM_MAX_SEGS 4u

extern "C" __global__ void __launch_bounds__(1024) opt_mma_moe_plan(
    const uint32_t *__restrict__ ids, const uint32_t rows, const uint32_t experts,
    uint32_t *__restrict__ sorted, uint32_t *__restrict__ items, uint32_t *__restrict__ items_count,
    const uint32_t seg_tokens, const uint32_t seg_n0, const uint32_t seg_n1, const uint32_t seg_n2,
    const uint32_t seg_n3) {
    OPT_PDL_ENTRY();
    __shared__ uint32_t s_count[512];
    __shared__ uint32_t s_cursor[512];
    const uint32_t tid = threadIdx.x;
    const uint32_t seg_n[OMM_MAX_SEGS] = {seg_n0, seg_n1, seg_n2, seg_n3};
    auto live = [&](uint32_t r) {
        if (seg_tokens == 0u) return true;
        const uint32_t token = r / OMM_TOPK;
        const uint32_t seg = token / seg_tokens;
        return seg < OMM_MAX_SEGS && token - seg * seg_tokens < seg_n[seg];
    };
    for (uint32_t e = tid; e < 512u; e += blockDim.x) s_count[e] = 0u;
    __syncthreads();
    for (uint32_t r = tid; r < rows; r += blockDim.x) {
        if (!live(r)) continue;
        const uint32_t id = ids[r] < experts ? ids[r] : 0u;
        atomicAdd(&s_count[id], 1u);
    }
    __syncthreads();
    if (tid == 0u) {
        uint32_t offset = 0u, n = 0u;
        for (uint32_t e = 0u; e < experts; ++e) {
            const uint32_t count = s_count[e];
            s_cursor[e] = offset;
            for (uint32_t s = 0u; s < count; s += OMM_ROWS) {
                items[4u * n] = e;
                items[4u * n + 1u] = offset + s;
                items[4u * n + 2u] = min(OMM_ROWS, count - s);
                items[4u * n + 3u] = 0u;
                ++n;
            }
            offset += count;
        }
        items_count[0] = n;
    }
    __syncthreads();
    for (uint32_t r = tid; r < rows; r += blockDim.x) {
        if (!live(r)) continue;
        const uint32_t id = ids[r] < experts ? ids[r] : 0u;
        sorted[atomicAdd(&s_cursor[id], 1u)] = r;
    }
}

extern "C" __global__ void __launch_bounds__(OM_THREADS) opt_mma_moe_gate_up(
    const uint64_t *__restrict__ gate_table, const uint64_t *__restrict__ up_table,
    const uint32_t *__restrict__ sorted, const uint32_t *__restrict__ items,
    const uint32_t *__restrict__ items_count, const float *__restrict__ weights,
    const uint8_t *__restrict__ act_fp8, const uint8_t *__restrict__ act_scales,
    uint8_t *__restrict__ out_codes, uint8_t *__restrict__ out_scales, const float limit) {
    OPT_PDL_ENTRY();
    const uint32_t item = blockIdx.y;
    if (item >= items_count[0]) return;
    __shared__ __align__(128) uint8_t s_a[2][OMM_ROWS * OMM_A_ROW];
    __shared__ __align__(128) uint8_t s_wg[2][OMM_GU_COLS * OMM_W_ROW];
    __shared__ __align__(128) uint8_t s_wu[2][OMM_GU_COLS * OMM_W_ROW];
    __shared__ __align__(16) uint8_t s_as[2][OMM_ROWS * OMM_KB];
    __shared__ __align__(16) uint8_t s_gs[2][OMM_GU_COLS * OMM_KB];
    __shared__ __align__(16) uint8_t s_us[2][OMM_GU_COLS * OMM_KB];
    __shared__ uint32_t s_row[OMM_ROWS];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t expert = items[4u * item];
    const uint32_t first = items[4u * item + 1u];
    const uint32_t count = items[4u * item + 2u];
    const uint32_t c0 = blockIdx.x * OMM_GU_COLS;
    if (tid < OMM_ROWS) s_row[tid] = tid < count ? sorted[first + tid] : 0u;
    __syncthreads();

    const uint8_t *gw = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(gate_table[2u * expert]));
    const uint8_t *gs = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(gate_table[2u * expert + 1u]));
    const uint8_t *uw = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(up_table[2u * expert]));
    const uint8_t *us = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(up_table[2u * expert + 1u]));
    constexpr uint32_t KB = OMM_HIDDEN / 32u;
    constexpr uint32_t STAGES = KB / OMM_KB;

    auto issue = [&](uint32_t stage, uint32_t buf) {
        const uint32_t kb0 = stage * OMM_KB;
        // Activations: 64 rows x 8 chunks.
#pragma unroll
        for (uint32_t i = 0; i < 2u; ++i) {
            const uint32_t idx = tid + i * OM_THREADS;
            const uint32_t row = idx >> 3;
            const uint32_t chunk = idx & 7u;
            const bool valid = row < count;
            const uint32_t token = s_row[row] / OMM_TOPK;
            const uint8_t *src = act_fp8 + (uint64_t)token * OMM_HIDDEN + kb0 * 32u + chunk * 16u;
            om_cp16(om_smem(&s_a[buf][omm_a_off(row, chunk >> 1) + (chunk & 1u) * 16u]), src, valid);
        }
        // Gate and up: 64 columns x 4 chunks each, one of each per thread.
        {
            const uint32_t col = tid >> 2;
            const uint32_t block = tid & 3u;
            const uint64_t at = (uint64_t)(c0 + col) * (OMM_HIDDEN / 2u) + (kb0 + block) * 16u;
            om_cp16(om_smem(&s_wg[buf][omm_w_off(col, block)]), gw + at, true);
            om_cp16(om_smem(&s_wu[buf][omm_w_off(col, block)]), uw + at, true);
        }
        // Scales: four bytes per row / column.
        if (tid < OMM_ROWS) {
            const uint32_t token = s_row[tid] / OMM_TOPK;
            om_cp4(om_smem(&s_as[buf][tid * OMM_KB]), act_scales + (uint64_t)token * KB + kb0, tid < count);
        } else if (tid < OMM_ROWS + OMM_GU_COLS) {
            const uint32_t col = tid - OMM_ROWS;
            om_cp4(om_smem(&s_gs[buf][col * OMM_KB]), gs + (uint64_t)(c0 + col) * KB + kb0, true);
        } else if (tid < OMM_ROWS + 2u * OMM_GU_COLS) {
            const uint32_t col = tid - OMM_ROWS - OMM_GU_COLS;
            om_cp4(om_smem(&s_us[buf][col * OMM_KB]), us + (uint64_t)(c0 + col) * KB + kb0, true);
        }
    };

    const uint32_t g = lane >> 2;
    const uint32_t q = lane & 3u;
    const uint32_t m_tiles = (count + 15u) >> 4;
    const uint32_t col = warp * 8u + g;   // this lane's B column (gate and up)
    float accg[4][4], accu[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int e = 0; e < 4; ++e) accg[i][e] = accu[i][e] = 0.0f;

    issue(0, 0);
    om_commit();
    for (uint32_t stage = 0; stage < STAGES; ++stage) {
        const uint32_t buf = stage & 1u;
        if (stage + 1u < STAGES) {
            issue(stage + 1u, buf ^ 1u);
            om_commit();
            om_wait<1>();
        } else {
            om_wait<0>();
        }
        __syncthreads();
#pragma unroll
        for (uint32_t j = 0; j < OMM_KB; ++j) {
            const uint32_t wg = *reinterpret_cast<const uint32_t *>(&s_wg[buf][omm_w_off(col, j) + 4u * q]);
            const uint32_t wu = *reinterpret_cast<const uint32_t *>(&s_wu[buf][omm_w_off(col, j) + 4u * q]);
            const uint32_t bg0 = om_spread_fp4(wg), bg1 = om_spread_fp4(wg >> 16);
            const uint32_t bu0 = om_spread_fp4(wu), bu1 = om_spread_fp4(wu >> 16);
            const uint32_t sfg = s_gs[buf][col * OMM_KB + j];
            const uint32_t sfu = s_us[buf][col * OMM_KB + j];
#pragma unroll
            for (uint32_t mt = 0; mt < 4u; ++mt) {
                if (mt < m_tiles) {
                    const uint32_t r0 = mt * 16u + g;
                    const uint2 lo = *reinterpret_cast<const uint2 *>(&s_a[buf][omm_a_off(r0, j) + 8u * q]);
                    const uint2 hi = *reinterpret_cast<const uint2 *>(&s_a[buf][omm_a_off(r0 + 8u, j) + 8u * q]);
                    const uint32_t af[4] = {lo.x, hi.x, lo.y, hi.y};
                    const uint32_t sfa = s_as[buf][(mt * 16u + g + 8u * (lane & 1u)) * OMM_KB + j];
                    om_mma_e4m3_e2m1(accg[mt], af, bg0, bg1, sfa, sfg);
                    om_mma_e4m3_e2m1(accu[mt], af, bu0, bu1, sfa, sfu);
                }
            }
        }
        __syncthreads();
    }

    // Epilogue: the activated, weighted, bf16-rounded values into [64 rows][64 cols] f32 (the
    // activation stages are free now), then the MX quantiser per 32-column block.
    float *s_val = reinterpret_cast<float *>(&s_a[0][0]);
#pragma unroll
    for (uint32_t mt = 0; mt < 4u; ++mt) {
        if (mt >= m_tiles) continue;
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) {
            const uint32_t row = mt * 16u + g + 8u * (e >> 1);
            const uint32_t c = warp * 8u + 2u * q + (e & 1u);
            if (row >= count) continue;
            const float gv = __bfloat162float(__float2bfloat16_rn(accg[mt][e]));
            const float uv = __bfloat162float(__float2bfloat16_rn(accu[mt][e]));
            const float clamped_gate = fminf(gv, limit);
            const float clamped_up = fminf(fmaxf(uv, -limit), limit);
            const float activated = (clamped_gate / (1.0f + __expf(-clamped_gate))) * clamped_up;
            s_val[row * OMM_GU_COLS + c] = __bfloat162float(__float2bfloat16_rn(activated * weights[s_row[row]]));
        }
    }
    __syncthreads();
    for (uint32_t pair = warp; pair < count * 2u; pair += 8u) {
        const uint32_t row = pair >> 1;
        const uint32_t blk = pair & 1u;
        const float value = s_val[row * OMM_GU_COLS + blk * 32u + lane];
        float amax = fabsf(value);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, offset));
        const uint8_t scale = spark_act_quant_scale_code(amax);
        const uint32_t routed = s_row[row];
        if (lane == 0u) out_scales[(uint64_t)routed * (OMM_INTER / 32u) + (c0 >> 5) + blk] = scale;
        const float inv_scale = 1.0f / spark_ue8m0_decode(scale);
        out_codes[(uint64_t)routed * OMM_INTER + c0 + blk * 32u + lane] = spark_fp8_e4m3_encode(value * inv_scale);
    }
}

extern "C" __global__ void __launch_bounds__(OM_THREADS) opt_mma_moe_down(
    const uint64_t *__restrict__ down_table, const uint32_t *__restrict__ sorted,
    const uint32_t *__restrict__ items, const uint32_t *__restrict__ items_count,
    const uint8_t *__restrict__ h_codes, const uint8_t *__restrict__ h_scales,
    __nv_bfloat16 *__restrict__ slot) {
    OPT_PDL_ENTRY();
    const uint32_t item = blockIdx.y;
    if (item >= items_count[0]) return;
    __shared__ __align__(128) uint8_t s_a[2][OMM_ROWS * OMM_A_ROW];
    __shared__ __align__(128) uint8_t s_w[2][OMM_D_COLS * OMM_W_ROW];
    __shared__ uint8_t s_as[2][OMM_ROWS * OMM_KB];
    __shared__ uint8_t s_ws[2][OMM_D_COLS * OMM_KB];
    __shared__ uint32_t s_row[OMM_ROWS];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t expert = items[4u * item];
    const uint32_t first = items[4u * item + 1u];
    const uint32_t count = items[4u * item + 2u];
    const uint32_t n0 = blockIdx.x * OMM_D_COLS;
    if (tid < OMM_ROWS) s_row[tid] = tid < count ? sorted[first + tid] : 0u;
    __syncthreads();
    const uint8_t *dw = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(down_table[2u * expert]));
    const uint8_t *ds = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(down_table[2u * expert + 1u]));
    constexpr uint32_t KB = OMM_INTER / 32u;   // 18
    constexpr uint32_t STAGES = (KB + OMM_KB - 1u) / OMM_KB;

    auto issue = [&](uint32_t stage, uint32_t buf) {
        const uint32_t kb0 = stage * OMM_KB;
#pragma unroll
        for (uint32_t i = 0; i < 2u; ++i) {
            const uint32_t idx = tid + i * OM_THREADS;
            const uint32_t row = idx >> 3;
            const uint32_t chunk = idx & 7u;
            const bool valid = row < count && kb0 + (chunk >> 1) < KB;
            const uint8_t *src = h_codes + (uint64_t)s_row[row] * OMM_INTER + kb0 * 32u + chunk * 16u;
            om_cp16(om_smem(&s_a[buf][omm_a_off(row, chunk >> 1) + (chunk & 1u) * 16u]), src, valid);
        }
#pragma unroll
        for (uint32_t i = 0; i < 2u; ++i) {
            const uint32_t idx = tid + i * OM_THREADS;
            const uint32_t c = idx >> 2;
            const uint32_t block = idx & 3u;
            const bool valid = kb0 + block < KB;
            const uint8_t *src = dw + (uint64_t)(n0 + c) * (OMM_INTER / 2u) + (kb0 + block) * 16u;
            om_cp16(om_smem(&s_w[buf][omm_w_off(c, block)]), src, valid);
        }
        // Scales by byte loads: the 18-byte rows are not 4-byte aligned.
        {
            const uint32_t row = tid >> 2;
            const uint32_t block = tid & 3u;
            const uint32_t kb = kb0 + block;
            s_as[buf][tid] = row < count && kb < KB ? __ldg(h_scales + (uint64_t)s_row[row] * KB + kb) : (uint8_t)0;
        }
#pragma unroll
        for (uint32_t i = 0; i < 2u; ++i) {
            const uint32_t idx = tid + i * OM_THREADS;
            const uint32_t c = idx >> 2;
            const uint32_t kb = kb0 + (idx & 3u);
            s_ws[buf][idx] = kb < KB ? __ldg(ds + (uint64_t)(n0 + c) * KB + kb) : (uint8_t)0;
        }
    };

    const uint32_t g = lane >> 2;
    const uint32_t q = lane & 3u;
    const uint32_t m_tiles = (count + 15u) >> 4;
    float acc[4][2][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][t][e] = 0.0f;

    issue(0, 0);
    om_commit();
    for (uint32_t stage = 0; stage < STAGES; ++stage) {
        const uint32_t buf = stage & 1u;
        if (stage + 1u < STAGES) {
            issue(stage + 1u, buf ^ 1u);
            om_commit();
            om_wait<1>();
        } else {
            om_wait<0>();
        }
        __syncthreads();
        const uint32_t live = min(OMM_KB, KB - stage * OMM_KB);
#pragma unroll
        for (uint32_t j = 0; j < OMM_KB; ++j) {
            if (j >= live) continue;
            uint32_t b[2][2], sfb[2];
#pragma unroll
            for (uint32_t t = 0; t < 2u; ++t) {
                const uint32_t c = warp * 16u + t * 8u + g;
                const uint32_t w = *reinterpret_cast<const uint32_t *>(&s_w[buf][omm_w_off(c, j) + 4u * q]);
                b[t][0] = om_spread_fp4(w);
                b[t][1] = om_spread_fp4(w >> 16);
                sfb[t] = s_ws[buf][c * OMM_KB + j];
            }
#pragma unroll
            for (uint32_t mt = 0; mt < 4u; ++mt) {
                if (mt < m_tiles) {
                    const uint32_t r0 = mt * 16u + g;
                    const uint2 lo = *reinterpret_cast<const uint2 *>(&s_a[buf][omm_a_off(r0, j) + 8u * q]);
                    const uint2 hi = *reinterpret_cast<const uint2 *>(&s_a[buf][omm_a_off(r0 + 8u, j) + 8u * q]);
                    const uint32_t af[4] = {lo.x, hi.x, lo.y, hi.y};
                    const uint32_t sfa = s_as[buf][(mt * 16u + g + 8u * (lane & 1u)) * OMM_KB + j];
#pragma unroll
                    for (uint32_t t = 0; t < 2u; ++t) om_mma_e4m3_e2m1(acc[mt][t], af, b[t][0], b[t][1], sfa, sfb[t]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t mt = 0; mt < 4u; ++mt) {
        if (mt >= m_tiles) continue;
#pragma unroll
        for (uint32_t half = 0; half < 2u; ++half) {
            const uint32_t row = mt * 16u + g + 8u * half;
            if (row >= count) continue;
            __nv_bfloat16 *dst = slot + (uint64_t)s_row[row] * OMM_HIDDEN + n0 + warp * 16u + 2u * q;
#pragma unroll
            for (uint32_t t = 0; t < 2u; ++t) {
                *reinterpret_cast<__nv_bfloat162 *>(dst + t * 8u) =
                    __floats2bfloat162_rn(acc[mt][t][2u * half], acc[mt][t][2u * half + 1u]);
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) opt_mma_moe_combine(
    const uint32_t *__restrict__ expert_ids, const __nv_bfloat16 *__restrict__ slot,
    float *__restrict__ delta_f32, __nv_bfloat16 *__restrict__ delta_bf16, const uint32_t tokens,
    const float *addend) {
    OPT_PDL_ENTRY();
    const uint32_t token = blockIdx.y;
    // Four adjacent columns a thread: a warp's six slot reads are 256 B (two full lines) and the
    // bf16 store is one 8 B vector, where one column a thread read 64 B and stored 2 B.
    const uint32_t column = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (token >= tokens || column >= OMM_HIDDEN) return;
    uint32_t ids[OMM_TOPK];
#pragma unroll
    for (uint32_t s = 0u; s < OMM_TOPK; ++s) ids[s] = expert_ids[token * OMM_TOPK + s];
    uint32_t used = 0u;
    float2 lo = make_float2(0.0f, 0.0f), hi = lo;
    const uint64_t base = (uint64_t)token * OMM_TOPK * OMM_HIDDEN;
#pragma unroll
    for (uint32_t position = 0u; position < OMM_TOPK; ++position) {
        uint32_t best = OMM_TOPK;
#pragma unroll
        for (uint32_t s = 0u; s < OMM_TOPK; ++s) {
            if (((used >> s) & 1u) != 0u) continue;
            if (best == OMM_TOPK || ids[s] < ids[best]) best = s;
        }
        used |= 1u << best;
        const __nv_bfloat162 *src = reinterpret_cast<const __nv_bfloat162 *>(slot + base + (uint64_t)best * OMM_HIDDEN + column);
        const float2 a = __bfloat1622float2(src[0]);
        const float2 b = __bfloat1622float2(src[1]);
        lo.x = __fadd_rn(lo.x, a.x);
        lo.y = __fadd_rn(lo.y, a.y);
        hi.x = __fadd_rn(hi.x, b.x);
        hi.y = __fadd_rn(hi.y, b.y);
    }
    const uint64_t out = (uint64_t)token * OMM_HIDDEN + column;
    if (addend != nullptr) {
        const float4 a = *reinterpret_cast<const float4 *>(addend + out);
        lo.x = __fadd_rn(lo.x, a.x);
        lo.y = __fadd_rn(lo.y, a.y);
        hi.x = __fadd_rn(hi.x, a.z);
        hi.y = __fadd_rn(hi.y, a.w);
    }
    if (delta_f32 != nullptr) *reinterpret_cast<float4 *>(delta_f32 + out) = make_float4(lo.x, lo.y, hi.x, hi.y);
    const __nv_bfloat162 p0 = __floats2bfloat162_rn(lo.x, lo.y);
    const __nv_bfloat162 p1 = __floats2bfloat162_rn(hi.x, hi.y);
    const uint2 packed = make_uint2(*reinterpret_cast<const uint32_t *>(&p0),
                                    *reinterpret_cast<const uint32_t *>(&p1));
    *reinterpret_cast<uint2 *>(delta_bf16 + out) = packed;
}

// ---- The indexer over a chunk: the scores on the FP4 tensor cores.
//
// opt_bulk_fp4_pack: re-encodes FP4 round-tripped f32 values (an E2M1 value times a power of two
//   per 32, as fp4_round_trip_bf16_to_f32 writes them) as packed E2M1 codes (two per byte, the
//   lower dimension in the low nibble: the kind::mxf4nvf4 operand form) plus one ue8m0 per 32.
//   With the scale 2^(E - 2), E the exponent of the block's largest magnitude, every value is a
//   code times the scale exactly (the E2M1 grid is closed under doubling up to 6), so the MMA
//   multiplies the reference's operands. One thread per 32-value block. frag = 0 writes the codes
//   row-major (16 bytes per block: the queries); frag = 1 writes 128-wide rows (the keys) in MMA
//   fragment order: eight keys per 512 bytes, the 16 bytes lane l of a warp loads at l * 16, i.e.
//   key 8 G + g's dimensions 32 c + 8 q .. + 7 at G * 512 + (4 g + q) * 16 + 4 c.
extern "C" __global__ void __launch_bounds__(256) opt_bulk_fp4_pack(
    const float *__restrict__ src, uint8_t *__restrict__ codes, uint8_t *__restrict__ scales,
    const uint32_t blocks, const uint32_t frag) {
    OPT_PDL_ENTRY();
    const uint32_t blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= blocks) return;
    const float4 *v = reinterpret_cast<const float4 *>(src + (uint64_t)blk * 32u);
    float x[32];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float4 f = v[i];
        x[4 * i] = f.x; x[4 * i + 1] = f.y; x[4 * i + 2] = f.z; x[4 * i + 3] = f.w;
    }
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[i]));
    int ex = 0;
    (void)frexpf(amax, &ex);   // amax = m 2^ex, m in [0.5, 1): the largest exponent E is ex - 1
    const int code = amax > 0.0f ? max(1, min(254, ex - 3 + 127)) : 127;
    const float inv = __uint_as_float((uint32_t)(254 - code) << 23);   // 2^-(code - 127)
    uint32_t packed[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint32_t word = 0u;
#pragma unroll
        for (int n = 0; n < 8; ++n) word |= (uint32_t)spark_fp4_e2m1_encode_fallback(x[8 * w + n] * inv) << (4 * n);
        packed[w] = word;
    }
    if (frag == 0u) {
        *reinterpret_cast<uint4 *>(codes + (uint64_t)blk * 16u) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    } else {
        const uint32_t key = blk >> 2, c = blk & 3u;
        uint32_t *base = reinterpret_cast<uint32_t *>(codes + (uint64_t)(key >> 3) * 512u + (key & 7u) * 64u + c * 4u);
#pragma unroll
        for (int w = 0; w < 4; ++w) base[w * 4] = packed[w];
    }
    scales[blk] = (uint8_t)code;
}

__device__ __forceinline__ void om_mma_nvf4(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1,
                                            uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0),
          "h"((uint16_t)0), "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}
// d = a x b + 0: an accumulator's first K-block, so no zero fill is kept live.
__device__ __forceinline__ void om_mma_nvf4_first(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1,
                                                  uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%14,%14,%14,%14},{%10},{%11,%12},{%13},{%11,%12};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0),
          "h"((uint16_t)0), "r"(sfb), "f"(0.0f));
}
// The pair (lo, hi) with negatives clamped to 0, rounded to bf16 (lo in the low half).
__device__ __forceinline__ uint32_t ois_relu_bf16x2(float lo, float hi) {
    uint32_t d;
    asm("cvt.rn.relu.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}
__device__ __forceinline__ uint32_t ois_bf16x2(float lo, float hi) {
    uint32_t d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}
// x * w in bf16 pairs, rounded once: the product of two bf16 values is exact in f32, so this is
// the reference's bf16(raw * weight).
__device__ __forceinline__ uint32_t ois_mul_bf16x2(uint32_t x, uint32_t w) {
    uint32_t d;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(x), "r"(w));
    return d;
}
// lo += the pair's low bf16, hi += its high bf16: one rounded f32 add each (FHADD.BF16).
__device__ __forceinline__ void ois_fold(float &lo, float &hi, uint32_t pair) {
    asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %2;\n\tadd.rn.f32.bf16 %0, l, %0;\n\tadd.rn.f32.bf16 %1, h, %1;\n\t}"
        : "+f"(lo), "+f"(hi)
        : "r"(pair));
}

// opt_mma_index_scores: the index scores of a chunk's rows against the compressed keys. Row y is
//   position p0 + y and scores keys 0 .. (p0 + y + 1) / ratio - 1 (its reach). Per (row, key), the
//   reference's chain: per head, raw = bf16(dot(q_head, key)); if raw > 0 the head adds
//   bf16(raw * weight); the heads are folded in ascending order with rounded adds and the sum is
//   rounded to bf16. The dots are the MMA's (exact products, f32 accumulation in its own order);
//   everything after them is the reference's arithmetic. The scores land as bf16 (they are bf16
//   values), `scores[y * score_stride + key]`.
//   Operands from opt_bulk_fp4_pack: q [tokens][32 heads][64 bytes] (frag 0), keys in fragment
//   order (frag 1), 4 ue8m0 per 128. weights: the rows' 32 head weights, bf16 values in f32.
//   mask_in (or null): a consumed candidate mask per row (mask_stride words); an eight-key group
//   with no candidate is not computed and a 64-key run with none is not written (opt_select_rows
//   reads neither).
//   Grid (ceil(tokens / 8), ceil(max reach / (128 tiles_per_block))), 256 threads: warp w owns row
//   8 blockIdx.x + w over the block's keys, 64 at a time: its q fragments stay in registers, the
//   keys stream from global (one 16-byte load per lane per 8 keys; the block's other rows find
//   them in L1), 32 MMAs (m16n8k64 kind::mxf4nvf4: 32 heads x 64 keys x 128), the products to
//   shared memory as bf16 pairs [32 key pairs][32 heads] (head h of pair p at h ^ (8 (p % 4) +
//   4 (p / 4 % 2)): conflict-free stores, and conflict-free 16-byte loads for the fold), then lane
//   l folds keys 2 l and 2 l + 1 while the next 64 keys load. No block barriers: the warps drift,
//   so one's MMAs run under another's fold.
#define OIS_ROWS 8u
#define OIS_PASS 64u
#define OIS_QROW 64u   // packed bytes per head (and per key)
extern "C" __global__ void __launch_bounds__(OM_THREADS, 2) opt_mma_index_scores(
    const uint8_t *__restrict__ q_codes, const uint8_t *__restrict__ q_scales,
    const uint8_t *__restrict__ k_frag, const uint8_t *__restrict__ k_scales,
    const float *__restrict__ weights, uint16_t *__restrict__ scores, const uint32_t tokens,
    const uint32_t p0, const uint32_t ratio, const uint32_t score_stride, const uint32_t tiles_per_block,
    const uint32_t *__restrict__ mask_in, const uint32_t mask_stride) {
    OPT_PDL_ENTRY();
    __shared__ __align__(16) uint32_t s_t[OIS_ROWS][32u * 32u];

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t y = blockIdx.x * OIS_ROWS + warp;
    if (y >= tokens) return;   // no barriers below
    const uint32_t n_row = (p0 + y + 1u) / ratio;
    const uint32_t key_begin = blockIdx.y * tiles_per_block * 128u;
    const uint32_t key_end = min(key_begin + tiles_per_block * 128u, n_row);
    if (key_begin >= key_end) return;
    const uint32_t g = lane >> 2;
    const uint32_t q = lane & 3u;

    // The row's A fragments: heads 16 mt + g (+ 8), dimensions 64 kb + 8 q .. (+ 32); the scale
    // pair of head 16 mt + g + 8 (lane % 2); the weights of the lane's heads as bf16 pairs.
    uint32_t a[2][2][4];
    uint32_t sfa[2][2];
    uint32_t w2[2][2];
    {
        const uint8_t *qrow = q_codes + (uint64_t)y * 32u * OIS_QROW;
#pragma unroll
        for (uint32_t mt = 0; mt < 2u; ++mt) {
            const uint32_t h0 = mt * 16u + g;
#pragma unroll
            for (uint32_t kb = 0; kb < 2u; ++kb) {
                a[mt][kb][0] = __ldg(reinterpret_cast<const uint32_t *>(qrow + h0 * OIS_QROW + kb * 32u + 4u * q));
                a[mt][kb][1] = __ldg(reinterpret_cast<const uint32_t *>(qrow + (h0 + 8u) * OIS_QROW + kb * 32u + 4u * q));
                a[mt][kb][2] = __ldg(reinterpret_cast<const uint32_t *>(qrow + h0 * OIS_QROW + kb * 32u + 16u + 4u * q));
                a[mt][kb][3] = __ldg(reinterpret_cast<const uint32_t *>(qrow + (h0 + 8u) * OIS_QROW + kb * 32u + 16u + 4u * q));
            }
            const uint32_t sw = __ldg(reinterpret_cast<const uint32_t *>(q_scales) + (uint64_t)y * 32u + mt * 16u + g + 8u * (lane & 1u));
            sfa[mt][0] = sw & 0xFFFFu;
            sfa[mt][1] = sw >> 16;
#pragma unroll
            for (uint32_t hh = 0; hh < 2u; ++hh) {
                const uint32_t wb = __float_as_uint(__ldg(weights + (uint64_t)y * 32u + mt * 16u + g + 8u * hh)) >> 16;
                w2[mt][hh] = wb | (wb << 16);
            }
        }
    }
    const uint32_t *mrow = mask_in != nullptr ? mask_in + (uint64_t)y * mask_stride : nullptr;
    // The eight-key groups of the 64 keys at kbase that are computed (bit nt).
    const auto pass_groups = [&](uint32_t kbase) -> uint32_t {
        if (mrow == nullptr) return 0xFFu;
        const uint32_t m0 = __ldg(mrow + kbase / 32u), m1 = __ldg(mrow + kbase / 32u + 1u);
        uint32_t bits = 0u;
#pragma unroll
        for (uint32_t b = 0; b < 4u; ++b) {
            bits |= ((m0 >> (8u * b)) & 0xFFu) != 0u ? 1u << b : 0u;
            bits |= ((m1 >> (8u * b)) & 0xFFu) != 0u ? 1u << (b + 4u) : 0u;
        }
        return bits;
    };
    uint4 bf[8];
    uint32_t sb[8];
    const auto fetch = [&](uint32_t kbase, uint32_t groups) {
#pragma unroll
        for (uint32_t nt = 0; nt < 8u; ++nt) {
            const uint32_t key = kbase + nt * 8u;
            if (((groups >> nt) & 1u) != 0u && key < n_row) {
                bf[nt] = __ldg(reinterpret_cast<const uint4 *>(k_frag + (uint64_t)(key / 8u) * 512u) + lane);
                sb[nt] = __ldg(reinterpret_cast<const uint32_t *>(k_scales) + min(key + g, n_row - 1u));
            } else {
                bf[nt] = make_uint4(0u, 0u, 0u, 0u);
                sb[nt] = 0x7F7F7F7Fu;
            }
        }
    };
    uint32_t *st = s_t[warp];
    uint32_t kbase = key_begin;
    fetch(kbase, pass_groups(kbase));
    while (kbase < key_end) {
        float acc[8][2][4];
#pragma unroll
        for (uint32_t nt = 0; nt < 8u; ++nt) {
#pragma unroll
            for (uint32_t mt = 0; mt < 2u; ++mt) {
                om_mma_nvf4_first(acc[nt][mt], a[mt][0], bf[nt].x, bf[nt].y, sfa[mt][0], sb[nt] & 0xFFFFu);
                om_mma_nvf4(acc[nt][mt], a[mt][1], bf[nt].z, bf[nt].w, sfa[mt][1], sb[nt] >> 16);
            }
        }
        // acc[nt][mt][e]: head 16 mt + g + 8 (e / 2), key 8 nt + 2 q + e % 2: pair 4 nt + q.
#pragma unroll
        for (uint32_t nt = 0; nt < 8u; ++nt)
#pragma unroll
            for (uint32_t mt = 0; mt < 2u; ++mt)
#pragma unroll
                for (uint32_t hh = 0; hh < 2u; ++hh) {
                    const uint32_t raw = ois_relu_bf16x2(acc[nt][mt][2u * hh], acc[nt][mt][2u * hh + 1u]);
                    const uint32_t swz = (q << 3) | ((nt & 1u) << 2);
                    st[(nt * 4u + q) * 32u + ((mt * 16u + 8u * hh + g) ^ swz)] = ois_mul_bf16x2(raw, w2[mt][hh]);
                }
        __syncwarp();
        const uint32_t here = kbase;
        // The next 64 keys with a candidate (a consumed mask can leave whole runs without one).
        uint32_t groups = 0u;
        kbase += OIS_PASS;
        while (kbase < key_end && (groups = pass_groups(kbase)) == 0u) kbase += OIS_PASS;
        if (kbase < key_end) fetch(kbase, groups);
        float lo = 0.0f, hi = 0.0f;
        const uint32_t swz = ((lane & 3u) << 3) | (((lane >> 2) & 1u) << 2);
#pragma unroll
        for (uint32_t i = 0; i < 8u; ++i) {
            const uint4 v = *reinterpret_cast<const uint4 *>(&st[lane * 32u + ((4u * i) ^ swz)]);
            ois_fold(lo, hi, v.x);
            ois_fold(lo, hi, v.y);
            ois_fold(lo, hi, v.z);
            ois_fold(lo, hi, v.w);
        }
        const uint32_t key = here + 2u * lane;
        const uint32_t out = ois_bf16x2(lo, hi);
        uint16_t *dst = scores + (uint64_t)y * score_stride + key;
        if (key + 1u < n_row) *reinterpret_cast<uint32_t *>(dst) = out;
        else if (key < n_row) *dst = (uint16_t)(out & 0xFFFFu);
        __syncwarp();
    }
}

// ---------------------------------------------------------------------------------------------
// opt_bulk_attn: the chunk's sparse attention (opt_rows_attn's keys and softmax) with the rank's 16
//   heads as the M of bf16 m16n8k16 MMAs. One block per row y (position p0 + y): the window keys
//   (E4M3 x ue8m0, the layer's linear ring) then the selected compressed keys (E2M1 x E4M3), 32 at
//   a time decoded to bf16 in shared memory (both decodes are exact in bf16), an online softmax
//   over the tiles with the probabilities rounded to bf16 for P x V, and the sink in the
//   denominator. Grid (rows), 256 threads, two blocks an SM: at 128 registers a block's loads,
//   decode and barriers run under the other block's MMAs. Not bit-identical to opt_rows_attn (the
//   sums run in MMA order).
// ---------------------------------------------------------------------------------------------
#define OBA_HEADS 16u
#define OBA_LATENT 512u
#define OBA_TILE 32u
#define OBA_WINDOW 128u
#define OBA_MAX_SELECTED 512u
#define OBA_MAX_ATTENDED 640u
#define OBA_P_LD 40u   // s_p's row stride in bf16 (80 B: the ldmatrix rows fall in distinct banks)
#define OBA_SCORE_SCALE 0.04419417382415922f

// Key row k's 16-byte chunk c (64 per row) lives at c ^ (k % 8).
__device__ __forceinline__ uint32_t oba_k_off(uint32_t key, uint32_t chunk) {
    return key * (OBA_LATENT * 2u) + ((chunk ^ (key & 7u)) << 4);
}

__device__ __forceinline__ void oba_ldsm4_t(uint32_t addr, uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ float oba_ue8m0(uint32_t code) {
    if (code == 255u) return __uint_as_float(0x7FC00000u);
    return code == 0u ? __uint_as_float(1u << 22) : __uint_as_float(code << 23);
}

__device__ __forceinline__ float oba_e4m3(uint32_t code) {
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((uint16_t)(code & 0xffu)));
    return __half2float(__ushort_as_half((uint16_t)(h & 0xffffu)));
}

// Eight E2M1 codes (one 32-bit word, low nibble first) times an E4M3 scale -> four bf16x2, exact:
// the codes widen to f16 in hardware and a product has at most 6 significant bits.
__device__ __forceinline__ void oba_e2m1x8(uint32_t word, float scale, uint32_t v[4]) {
    uint32_t h[4];
    asm("{\n.reg .b8 b0, b1, b2, b3;\nmov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\ncvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\ncvt.rn.f16x2.e2m1x2 %3, b3;\n}"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
        : "r"(word));
#pragma unroll
    for (uint32_t b = 0; b < 4u; ++b) {
        const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(&h[b]));
        v[b] = ob_pack(__float2bfloat16_rn(f.x * scale), __float2bfloat16_rn(f.y * scale));
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) opt_bulk_attn(
    const __nv_bfloat16 *__restrict__ q_all,   // [rows][16][512]
    const uint8_t *__restrict__ kv_fp4, const uint8_t *__restrict__ kv_scales,
    const uint8_t *__restrict__ ring_codes, const uint8_t *__restrict__ ring_scales,
    const uint32_t *__restrict__ selected_all, const float *__restrict__ sink,
    __nv_bfloat16 *__restrict__ out_all,       // [rows][16][512]
    const uint32_t p0, const uint32_t ratio, const uint32_t topk, const uint32_t selected_stride,
    const uint32_t ring_rows) {
    OPT_PDL_ENTRY();
    __shared__ __align__(128) uint8_t s_k[OBA_TILE * OBA_LATENT * 2u];
    __shared__ __align__(8) float s_s[4][OBA_HEADS][OBA_TILE];
    __shared__ __align__(16) __nv_bfloat16 s_p[OBA_HEADS * OBA_P_LD];
    __shared__ uint32_t s_cand[OBA_MAX_SELECTED];
    __shared__ uint32_t s_valid[OBA_TILE];
    __shared__ float s_m[OBA_HEADS], s_l[OBA_HEADS], s_alpha[OBA_HEADS];

    const uint32_t y = blockIdx.x;
    const uint32_t position = p0 + y;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, q = lane & 3u;
    const uint32_t window_rows = position + 1u < OBA_WINDOW ? position + 1u : OBA_WINDOW;
    const uint32_t window_start = position + 1u - window_rows;
    uint32_t selected_count = 0u;
    if (ratio > 0u) {
        const uint32_t n_pos = (position + 1u) / ratio;
        selected_count = n_pos < topk ? n_pos : topk;
    }
    __nv_bfloat16 *out = out_all + (uint64_t)y * OBA_HEADS * OBA_LATENT;
    if (selected_count > OBA_MAX_SELECTED || window_rows + selected_count > OBA_MAX_ATTENDED) {
        for (uint32_t i = tid; i < OBA_HEADS * OBA_LATENT; i += 256u) out[i] = __float2bfloat16_rn(__uint_as_float(0x7FC00000u));
        return;
    }
    const uint32_t window_slot = window_start % ring_rows;
    const uint8_t *window_codes = ring_codes + (uint64_t)window_slot * OBA_LATENT;
    const uint8_t *window_scales = ring_scales + (uint64_t)window_slot * (OBA_LATENT / 32u);
    const uint32_t *selected = selected_all + (uint64_t)y * selected_stride;
    for (uint32_t i = tid; i < selected_count; i += 256u) s_cand[i] = selected[i];
    if (tid < OBA_HEADS) {
        s_m[tid] = -3.402823466e38f;
        s_l[tid] = 0.0f;
    }
    const uint32_t total = window_rows + selected_count;
    const uint32_t tiles = (total + OBA_TILE - 1u) / OBA_TILE;

    // S = Q K^T: warp w scores keys 16 (w % 2) .. + 15 over dims 128 (w / 2) .. + 127.
    const uint32_t kq = warp & 1u, kd = warp >> 1;
    const __nv_bfloat16 *qrow = q_all + (uint64_t)y * OBA_HEADS * OBA_LATENT;
    uint32_t qa[8][4];
#pragma unroll
    for (uint32_t j = 0; j < 8u; ++j) {
        const uint32_t k0 = kd * 128u + j * 16u + 2u * q;
        qa[j][0] = __ldg(reinterpret_cast<const uint32_t *>(qrow + g * OBA_LATENT + k0));
        qa[j][1] = __ldg(reinterpret_cast<const uint32_t *>(qrow + (g + 8u) * OBA_LATENT + k0));
        qa[j][2] = __ldg(reinterpret_cast<const uint32_t *>(qrow + g * OBA_LATENT + k0 + 8u));
        qa[j][3] = __ldg(reinterpret_cast<const uint32_t *>(qrow + (g + 8u) * OBA_LATENT + k0 + 8u));
    }
    // O: warp w holds dims 64 w .. + 63 of the 16 heads.
    float acc[8][4];
#pragma unroll
    for (uint32_t nt = 0; nt < 8u; ++nt)
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) acc[nt][e] = 0.0f;

    __syncthreads();   // s_cand
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        // The lane's dims 16 lane .. + 15 of keys warp + 8 i: all four loads, then the decodes.
        uint4 raw[4];
        uint32_t raw_scale[4], raw_kind[4];
#pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) {
            const uint32_t slot = tile * OBA_TILE + warp + 8u * i;
            raw_kind[i] = 0u;
            if (slot < window_rows) {
                raw[i] = __ldg(reinterpret_cast<const uint4 *>(window_codes + (uint64_t)slot * OBA_LATENT) + lane);
                raw_scale[i] = window_scales[(uint64_t)slot * (OBA_LATENT / 32u) + (lane >> 1)];
                raw_kind[i] = 1u;
            } else if (slot < total) {
                const uint32_t candidate = s_cand[slot - window_rows];
                if (candidate <= position) {
                    const uint2 packed = __ldg(reinterpret_cast<const uint2 *>(kv_fp4 + (uint64_t)candidate * (OBA_LATENT / 2u)) + lane);
                    raw[i] = make_uint4(packed.x, packed.y, 0u, 0u);
                    raw_scale[i] = kv_scales[(uint64_t)candidate * (OBA_LATENT / 16u) + lane];
                    raw_kind[i] = 2u;
                }
            }
        }
#pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) {
            const uint32_t key = warp + 8u * i;
            uint32_t v[8];
            if (raw_kind[i] == 1u) {
                const float scale = oba_ue8m0(raw_scale[i]);
                const uint32_t w4[4] = {raw[i].x, raw[i].y, raw[i].z, raw[i].w};
#pragma unroll
                for (uint32_t j = 0; j < 4u; ++j) {
                    v[2u * j] = ob_e4m3x2(w4[j], scale);
                    v[2u * j + 1u] = ob_e4m3x2(w4[j] >> 16, scale);
                }
            } else if (raw_kind[i] == 2u) {
                const float scale = oba_e4m3(raw_scale[i]);
                oba_e2m1x8(raw[i].x, scale, v);
                oba_e2m1x8(raw[i].y, scale, v + 4);
            } else {
#pragma unroll
                for (uint32_t j = 0; j < 8u; ++j) v[j] = 0u;
            }
            *reinterpret_cast<uint4 *>(&s_k[oba_k_off(key, 2u * lane)]) = make_uint4(v[0], v[1], v[2], v[3]);
            *reinterpret_cast<uint4 *>(&s_k[oba_k_off(key, 2u * lane + 1u)]) = make_uint4(v[4], v[5], v[6], v[7]);
            if (lane == 0u) s_valid[key] = raw_kind[i] != 0u;
        }
        __syncthreads();
        // One accumulator per 8-key column tile: two independent chains of eight MMAs.
        {
            float c[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
            for (uint32_t j2 = 0; j2 < 4u; ++j2) {
                const uint32_t chunk = kd * 16u + j2 * 4u + (lane >> 3);
#pragma unroll
                for (uint32_t t = 0; t < 2u; ++t) {
                    const uint32_t key = kq * 16u + t * 8u + (lane & 7u);
                    uint32_t b0, b1, b2, b3;
                    om_ldsm4(om_smem(&s_k[oba_k_off(key, chunk)]), b0, b1, b2, b3);
                    ob_mma(c[t], qa[2u * j2], b0, b1);
                    ob_mma(c[t], qa[2u * j2 + 1u], b2, b3);
                }
            }
#pragma unroll
            for (uint32_t t = 0; t < 2u; ++t) {
                const uint32_t col = kq * 16u + t * 8u + 2u * q;
                *reinterpret_cast<float2 *>(&s_s[kd][g][col]) = make_float2(c[t][0], c[t][1]);
                *reinterpret_cast<float2 *>(&s_s[kd][g + 8u][col]) = make_float2(c[t][2], c[t][3]);
            }
        }
        __syncthreads();
        // Online softmax: warp w takes heads 2 w (lanes 0-15) and 2 w + 1 (lanes 16-31), two keys a
        // lane, so one 16-lane butterfly reduces both heads.
        {
            const uint32_t h = 2u * warp + (lane >> 4), k2 = 2u * (lane & 15u);
            const bool v0 = s_valid[k2] != 0u, v1 = s_valid[k2 + 1u] != 0u;
            const float2 a0 = *reinterpret_cast<const float2 *>(&s_s[0][h][k2]);
            const float2 a1 = *reinterpret_cast<const float2 *>(&s_s[1][h][k2]);
            const float2 a2 = *reinterpret_cast<const float2 *>(&s_s[2][h][k2]);
            const float2 a3 = *reinterpret_cast<const float2 *>(&s_s[3][h][k2]);
            const float score0 = v0 ? ((a0.x + a1.x) + (a2.x + a3.x)) * OBA_SCORE_SCALE : -3.402823466e38f;
            const float score1 = v1 ? ((a0.y + a1.y) + (a2.y + a3.y)) * OBA_SCORE_SCALE : -3.402823466e38f;
            float tile_max = fmaxf(score0, score1);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, o));
            const float m_old = s_m[h];
            const float m_new = fmaxf(m_old, tile_max);
            const float p0 = v0 ? expf(score0 - m_new) : 0.0f;
            const float p1 = v1 ? expf(score1 - m_new) : 0.0f;
            *reinterpret_cast<__nv_bfloat162 *>(&s_p[h * OBA_P_LD + k2]) = __floats2bfloat162_rn(p0, p1);
            float sum = p0 + p1;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            if ((lane & 15u) == 0u) {
                const float alpha = expf(m_old - m_new);
                s_alpha[h] = alpha;
                s_l[h] = s_l[h] * alpha + sum;
                s_m[h] = m_new;
            }
        }
        __syncthreads();
        // O = O alpha + P V over the tile's 32 keys.
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
                const uint32_t k0 = ks * 16u;
                uint32_t pa[4];
                om_ldsm4(om_smem(&s_p[((lane & 7u) + 8u * ((lane >> 3) & 1u)) * OBA_P_LD + k0 + 8u * (lane >> 4)]),
                         pa[0], pa[1], pa[2], pa[3]);
                const uint32_t key = k0 + (lane & 7u) + 8u * ((lane >> 3) & 1u);
#pragma unroll
                for (uint32_t pair = 0; pair < 4u; ++pair) {
                    const uint32_t chunk = warp * 8u + pair * 2u + (lane >> 4);
                    uint32_t b0, b1, b2, b3;
                    oba_ldsm4_t(om_smem(&s_k[oba_k_off(key, chunk)]), b0, b1, b2, b3);
                    ob_mma(acc[2u * pair], pa, b0, b1);
                    ob_mma(acc[2u * pair + 1u], pa, b2, b3);
                }
            }
        }
        __syncthreads();
    }

    const float d_lo = s_l[g] + expf(sink[g] - s_m[g]);
    const float d_hi = s_l[g + 8u] + expf(sink[g + 8u] - s_m[g + 8u]);
    const float inv_lo = 1.0f / (d_lo > 0.0f ? d_lo : 1.0f);
    const float inv_hi = 1.0f / (d_hi > 0.0f ? d_hi : 1.0f);
#pragma unroll
    for (uint32_t nt = 0; nt < 8u; ++nt) {
        const uint32_t d = warp * 64u + nt * 8u + 2u * q;
        *reinterpret_cast<__nv_bfloat162 *>(out + (uint64_t)g * OBA_LATENT + d) =
            __floats2bfloat162_rn(acc[nt][0] * inv_lo, acc[nt][1] * inv_lo);
        *reinterpret_cast<__nv_bfloat162 *>(out + (uint64_t)(g + 8u) * OBA_LATENT + d) =
            __floats2bfloat162_rn(acc[nt][2] * inv_hi, acc[nt][3] * inv_hi);
    }
}

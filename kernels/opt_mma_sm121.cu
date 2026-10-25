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
//   pipeline; warp (wm, wn) computes 64 tokens x one 32-column weight tile (4 x 4 MMA tiles).
//   Blocks with the same column tile are adjacent in launch order, so a weight tile is read from
//   DRAM about once while the activation rows stay in L2.

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
// scales are the same one per 32 x 32 block.
template <bool OUT_F32, bool W_ROWS = false>
__device__ __forceinline__ void opt_mma_fp8_body(
    const uint8_t *__restrict__ a, const uint8_t *__restrict__ a_scales, const uint8_t *__restrict__ w,
    const uint8_t *__restrict__ ws, void *__restrict__ out, const uint32_t cols, const uint32_t inner,
    const uint32_t tokens, const uint64_t a_tok_stride, const uint64_t out_tok_stride,
    const uint64_t a_grp_stride, const uint64_t w_grp_stride, const uint64_t s_grp_stride,
    const uint64_t out_grp_stride, const uint32_t k_stride_blocks) {
    __shared__ __align__(128) uint8_t s_a[2][OM_BM * OM_A_ROW];
    __shared__ __align__(128) uint8_t s_w[2][OM_W_TILES * 1024u];
    __shared__ uint8_t s_as[2][OM_BM * OM_KB];
    __shared__ uint8_t s_ws[2][OM_W_TILES];

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
        // Scales by plain loads (a row's two bytes need not be 4-byte aligned).
        if (tid < OM_BM) {
            const uint32_t token = m0 + tid;
#pragma unroll
            for (uint32_t j = 0; j < OM_KB; ++j) {
                const uint32_t kb = kb0 + j;
                s_as[buf][tid * OM_KB + j] =
                    token < tokens && kb < k_blocks ? __ldg(as_grp + (uint64_t)token * a_tok_stride / 32u + kb) : (uint8_t)0;
            }
        } else if (tid < OM_BM + OM_W_TILES) {
            const uint32_t tile = tid - OM_BM;
            const uint32_t ct = tile / OM_KB;
            const uint32_t kt = tile % OM_KB;
            const uint32_t kb = kb0 + kt;
            s_ws[buf][tile] = ct < col_tiles && kb < k_blocks ? __ldg(ws_grp + (uint64_t)ct * k_row + kb) : (uint8_t)0;
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

    issue(0, 0);
    om_commit();
    for (uint32_t stage = 0; stage < stages; ++stage) {
        const uint32_t buf = stage & 1u;
        if (stage + 1u < stages) {
            issue(stage + 1u, buf ^ 1u);
            om_commit();
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
                const uint32_t sfb = s_ws[buf][tile];
#pragma unroll
                for (uint32_t mt = 0; mt < 4u; ++mt) {
                    const uint32_t rb = wm * 64u + mt * 16u;
                    uint32_t af[4];
                    const uint32_t row = rb + (lane & 7u) + ((lane >> 3) & 1u) * 8u;
                    const uint32_t chunk = 2u * j + (lane >> 4);
                    om_ldsm4(om_smem(&s_a[buf][om_a_off(row, chunk)]), af[0], af[1], af[2], af[3]);
                    const uint32_t sfa = s_as[buf][(rb + (lane >> 2) + 8u * (lane & 1u)) * OM_KB + j];
#pragma unroll
                    for (uint32_t nt = 0; nt < 4u; ++nt) om_mma_e4m3(acc[mt][nt], af, b[nt][0], b[nt][1], sfa, sfb);
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
    constexpr uint32_t WB = WMODE == OB_W_FP8 ? 1u : WMODE == OB_W_F32 ? 4u : 2u;
    const uint8_t *w_grp = WMODE == OB_W_ROW ? w + group * w_grp_stride
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
            } else {
                const uint8_t *tile = w_grp + ((uint64_t)w_tile * k_row + kb) * 1024u * WB;
                src = reinterpret_cast<const uint4 *>(tile + (w_col * 32u + w_half * 16u) * WB);
            }
#pragma unroll
            for (uint32_t v = 0; v < WB; ++v) rw[v] = __ldg(src + v);
            rs = WMODE == OB_W_FP8 ? __ldg(ws_grp + (uint64_t)w_tile * k_row + kb) : (uint8_t)127u;
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
        if constexpr (WMODE == OB_W_FP8) {
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

// opt_mma_hostw_{bf16w, bf16w_f32o, f32w_f32o}: the host-weight GEMVs (b12x_hostw_gemv_m1's three
//   modes: an f32 activation row against bf16 or f32 weights in the B12x tiling) over many rows.
//   Grid (ceil(cols / 128), ceil(tokens / 128)), 256 threads. f32 weights are split like the
//   activation: head x head + tail x head + head x tail.
#define OPT_MMA_HOSTW_ENTRY(name, WMODE, W_T, OUTF32)                                               \
    extern "C" __global__ void __launch_bounds__(OM_THREADS) name(                                 \
        const float *__restrict__ a, const W_T *__restrict__ w, void *__restrict__ out,           \
        const uint32_t cols, const uint32_t inner, const uint32_t tokens,                          \
        const uint64_t a_tok_stride, const uint64_t out_tok_stride) {                              \
        OPT_PDL_ENTRY();                                                                           \
        opt_mma_bf16_body<OB_A_F32, WMODE, OUTF32>(a, reinterpret_cast<const uint8_t *>(w), nullptr, out, \
                                                   cols, inner, tokens, a_tok_stride,              \
                                                   out_tok_stride, 0u, 0u, 0u, 0u, 0u);            \
    }

OPT_MMA_HOSTW_ENTRY(opt_mma_hostw_bf16w, OB_W_BF16, __nv_bfloat16, false)
OPT_MMA_HOSTW_ENTRY(opt_mma_hostw_bf16w_f32o, OB_W_BF16, __nv_bfloat16, true)
OPT_MMA_HOSTW_ENTRY(opt_mma_hostw_f32w_f32o, OB_W_F32, float, true)

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
//   (expert, first, count, 0); items_count[0] = the item count.
// opt_mma_moe_gate_up: grid (576 / 64, max items), 256 threads; block (x, y) = item y, intermediate
//   columns 64 x .. 64 x + 63, gate and up together. Epilogue: the opt_moe chain (bf16 gate and up,
//   clamp, SwiGLU, routing weight, bf16, MX quantiser) into codes [rows][576], scales [rows][18].
// opt_mma_moe_down: grid (5120 / 128, max items), 256 threads; block (x, y) = item y, output
//   columns 128 x .. 128 x + 127; writes each row's slot value rounded to bf16 into slot [rows][5120].
// opt_mma_moe_combine: grid (5120 / 256, tokens), 256 threads; the routed combine of opt_moe (slots
//   in ascending expert id, f32 adds, then the addend) into delta_f32 and delta_bf16.

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

extern "C" __global__ void __launch_bounds__(1024) opt_mma_moe_plan(
    const uint32_t *__restrict__ ids, const uint32_t rows, const uint32_t experts,
    uint32_t *__restrict__ sorted, uint32_t *__restrict__ items, uint32_t *__restrict__ items_count) {
    OPT_PDL_ENTRY();
    __shared__ uint32_t s_count[512];
    __shared__ uint32_t s_cursor[512];
    const uint32_t tid = threadIdx.x;
    for (uint32_t e = tid; e < 512u; e += blockDim.x) s_count[e] = 0u;
    __syncthreads();
    for (uint32_t r = tid; r < rows; r += blockDim.x) {
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
    const uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= tokens || column >= OMM_HIDDEN) return;
    uint32_t ids[OMM_TOPK];
#pragma unroll
    for (uint32_t s = 0u; s < OMM_TOPK; ++s) ids[s] = expert_ids[token * OMM_TOPK + s];
    uint32_t used = 0u;
    float total = 0.0f;
#pragma unroll
    for (uint32_t position = 0u; position < OMM_TOPK; ++position) {
        uint32_t best = OMM_TOPK;
#pragma unroll
        for (uint32_t s = 0u; s < OMM_TOPK; ++s) {
            if (((used >> s) & 1u) != 0u) continue;
            if (best == OMM_TOPK || ids[s] < ids[best]) best = s;
        }
        used |= 1u << best;
        total = __fadd_rn(total, __bfloat162float(slot[(uint64_t)(token * OMM_TOPK + best) * OMM_HIDDEN + column]));
    }
    const uint64_t out = (uint64_t)token * OMM_HIDDEN + column;
    if (addend != nullptr) total = __fadd_rn(total, addend[out]);
    delta_f32[out] = total;
    delta_bf16[out] = __float2bfloat16_rn(total);
}

// ---- The indexer over a chunk: the scores on the FP4 tensor cores.
//
// opt_bulk_fp4_pack: re-encodes FP4 round-tripped f32 values (an E2M1 value times a power of two
//   per 32, as fp4_round_trip_bf16_to_f32 writes them) as E2M1 codes in 8-bit containers (bits
//   5:2, the kind::mxf8f6f4 operand form) plus one ue8m0 per 32. With the scale 2^(E - 2), E the
//   exponent of the block's largest magnitude, every value is a code times the scale exactly (the
//   E2M1 grid is closed under doubling up to 6), so the MMA multiplies the reference's operands.
//   One thread per 32-value block.
extern "C" __global__ void __launch_bounds__(256) opt_bulk_fp4_pack(
    const float *__restrict__ src, uint8_t *__restrict__ codes, uint8_t *__restrict__ scales,
    const uint32_t blocks) {
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
    uint32_t packed[8];
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        uint32_t word = 0u;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t nibble = spark_fp4_e2m1_encode_fallback(x[4 * w + b] * inv);
            word |= (nibble << 2) << (8 * b);
        }
        packed[w] = word;
    }
    uint4 *out = reinterpret_cast<uint4 *>(codes + (uint64_t)blk * 32u);
    out[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    out[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
    scales[blk] = (uint8_t)code;
}

__device__ __forceinline__ void om_mma_e2m1(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1,
                                            uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0),
          "h"((uint16_t)0), "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

// opt_mma_index_scores: the index scores of a chunk's rows against the compressed keys. Row y is
//   position p0 + y and scores keys 0 .. (p0 + y + 1) / ratio - 1 (its reach). Per (row, key), the
//   reference's chain: per head, raw = bf16(dot(q_head, key)); if raw > 0 the head adds
//   bf16(raw * weight); the heads are folded in ascending order with rounded adds and the sum is
//   rounded to bf16. The dots are the MMA's (exact products, f32 accumulation in its own order);
//   everything after them is the reference's arithmetic.
//   Operands from opt_bulk_fp4_pack: q [tokens][32 heads][128], keys [n][128], 4 ue8m0 per 128.
//   Grid (ceil(max reach / 128), ceil(tokens / 4)), 256 threads: a block owns 4 rows x 128 keys;
//   warp w computes row w / 2 against keys 64 (w % 2) .. + 63 (2 x 8 MMA tiles over K = 128),
//   then folds its terms 16 keys at a time through shared memory, one lane per key.
#define OIS_KEYS 128u
#define OIS_DIM 128u
#define OIS_HEADS 32u
extern "C" __global__ void __launch_bounds__(OM_THREADS) opt_mma_index_scores(
    const uint8_t *__restrict__ q_codes, const uint8_t *__restrict__ q_scales,
    const uint8_t *__restrict__ k_codes, const uint8_t *__restrict__ k_scales,
    const float *__restrict__ weights, float *__restrict__ scores, const uint32_t tokens,
    const uint32_t p0, const uint32_t ratio, const uint32_t score_stride) {
    OPT_PDL_ENTRY();
    __shared__ __align__(128) uint8_t s_k[OIS_KEYS * OIS_DIM];
    __shared__ __align__(16) uint8_t s_ks[OIS_KEYS * 4u];
    __shared__ float s_t[8][16][OIS_HEADS + 1u];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t y0 = blockIdx.y * 4u;
    const uint32_t key0 = blockIdx.x * OIS_KEYS;
    if (y0 >= tokens) return;
    const uint32_t y_last = min(y0 + 3u, tokens - 1u);
    const uint32_t reach_max = (p0 + y_last + 1u) / ratio;
    if (key0 >= reach_max) return;
    const uint32_t n_keys = min(OIS_KEYS, reach_max - key0);

    // Keys: 128 rows of 8 chunks; chunk c of key r lives at c ^ (r % 8), so the eight keys an MMA
    // fragment load reads fall in eight bank groups.
#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) {
        const uint32_t idx = tid + i * OM_THREADS;
        const uint32_t key = idx >> 3;
        const uint32_t c = idx & 7u;
        const bool valid = key < n_keys;
        const uint8_t *src = valid ? k_codes + (uint64_t)(key0 + key) * OIS_DIM + c * 16u : k_codes;
        om_cp16(om_smem(&s_k[key * OIS_DIM + ((c ^ (key & 7u)) << 4)]), src, valid);
    }
    om_commit();
    if (tid < OIS_KEYS) {
        *reinterpret_cast<uint32_t *>(&s_ks[tid * 4u]) =
            tid < n_keys ? __ldg(reinterpret_cast<const uint32_t *>(k_scales + (uint64_t)(key0 + tid) * 4u)) : 0u;
    }
    om_wait<0>();
    __syncthreads();

    const uint32_t y = y0 + (warp >> 1);
    const uint32_t n_row = y < tokens ? (p0 + y + 1u) / ratio : 0u;
    const uint32_t kh0 = (warp & 1u) * 64u;
    if (key0 + kh0 >= n_row) return;   // no later barrier: the warps below only use their own slots

    const uint32_t g = lane >> 2;
    const uint32_t q = lane & 3u;
    const uint8_t *qrow = q_codes + (uint64_t)y * OIS_HEADS * OIS_DIM;
    uint32_t a[2][4][4];
    uint32_t sfa[2][4];
#pragma unroll
    for (uint32_t mt = 0; mt < 2u; ++mt) {
        const uint32_t h0 = mt * 16u + g;
#pragma unroll
        for (uint32_t kb = 0; kb < 4u; ++kb) {
            a[mt][kb][0] = __ldg(reinterpret_cast<const uint32_t *>(qrow + h0 * OIS_DIM + kb * 32u + 4u * q));
            a[mt][kb][1] = __ldg(reinterpret_cast<const uint32_t *>(qrow + (h0 + 8u) * OIS_DIM + kb * 32u + 4u * q));
            a[mt][kb][2] = __ldg(reinterpret_cast<const uint32_t *>(qrow + h0 * OIS_DIM + kb * 32u + 16u + 4u * q));
            a[mt][kb][3] = __ldg(reinterpret_cast<const uint32_t *>(qrow + (h0 + 8u) * OIS_DIM + kb * 32u + 16u + 4u * q));
            sfa[mt][kb] = __ldg(q_scales + ((uint64_t)y * OIS_HEADS + mt * 16u + (lane >> 2) + 8u * (lane & 1u)) * 4u + kb);
        }
    }
    float acc[8][2][4];
#pragma unroll
    for (uint32_t nt = 0; nt < 8u; ++nt)
#pragma unroll
        for (uint32_t mt = 0; mt < 2u; ++mt)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) acc[nt][mt][e] = 0.0f;
#pragma unroll
    for (uint32_t nt = 0; nt < 8u; ++nt) {
        const uint32_t key = kh0 + nt * 8u + g;
#pragma unroll
        for (uint32_t kb = 0; kb < 4u; ++kb) {
            const uint32_t b0 = *reinterpret_cast<const uint32_t *>(&s_k[key * OIS_DIM + (((2u * kb) ^ (key & 7u)) << 4) + 4u * q]);
            const uint32_t b1 = *reinterpret_cast<const uint32_t *>(&s_k[key * OIS_DIM + (((2u * kb + 1u) ^ (key & 7u)) << 4) + 4u * q]);
            const uint32_t sfb = s_ks[key * 4u + kb];
#pragma unroll
            for (uint32_t mt = 0; mt < 2u; ++mt) om_mma_e2m1(acc[nt][mt], a[mt][kb], b0, b1, sfa[mt][kb], sfb);
        }
    }

    // The lane's four heads: g, g + 8, 16 + g, 24 + g (acc[.][mt][e]: head 16 mt + g + 8 (e / 2)).
    float w[2][2];
#pragma unroll
    for (uint32_t mt = 0; mt < 2u; ++mt) {
        w[mt][0] = __ldg(weights + (uint64_t)y * OIS_HEADS + mt * 16u + g);
        w[mt][1] = __ldg(weights + (uint64_t)y * OIS_HEADS + mt * 16u + g + 8u);
    }
#pragma unroll
    for (uint32_t round = 0; round < 4u; ++round) {
#pragma unroll
        for (uint32_t j = 0; j < 2u; ++j) {
            const uint32_t nt = 2u * round + j;
#pragma unroll
            for (uint32_t mt = 0; mt < 2u; ++mt) {
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    const uint32_t head = mt * 16u + g + 8u * (e >> 1);
                    const uint32_t keyl = j * 8u + 2u * q + (e & 1u);
                    const float raw = __bfloat162float(__float2bfloat16_rn(acc[nt][mt][e]));
                    s_t[warp][keyl][head] =
                        raw > 0.0f ? __bfloat162float(__float2bfloat16_rn(__fmul_rn(raw, w[mt][e >> 1]))) : 0.0f;
                }
            }
        }
        __syncwarp();
        const uint32_t key = key0 + kh0 + round * 16u + lane;
        if (lane < 16u && key < n_row) {
            float sum = 0.0f;
#pragma unroll
            for (uint32_t h = 0; h < OIS_HEADS; ++h) sum = __fadd_rn(sum, s_t[warp][lane][h]);
            scores[(uint64_t)y * score_stride + key] = __bfloat162float(__float2bfloat16_rn(sum));
        }
        __syncwarp();
    }
}

// ---------------------------------------------------------------------------------------------
// opt_bulk_attn: the chunk's sparse attention (opt_rows_attn's keys and softmax) with the rank's 16
//   heads as the M of bf16 m16n8k16 MMAs. One block per row y (position p0 + y): the window keys
//   (E4M3 x ue8m0, the layer's linear ring) then the selected compressed keys (E2M1 x E4M3), 32 at
//   a time decoded to bf16 in shared memory (both decodes are exact in bf16), an online softmax
//   over the tiles with the probabilities rounded to bf16 for P x V, and the sink in the
//   denominator. The next tile's raw bytes load into registers under the current tile's MMAs.
//   Grid (rows), 256 threads. Not bit-identical to opt_rows_attn (the sums run in MMA order).
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

__device__ __forceinline__ float oba_e2m1(uint32_t nibble) {
    const uint32_t e = (nibble >> 1) & 3u;
    const float half_m = (nibble & 1u) != 0u ? 0.5f : 0.0f;
    const float m = e == 0u ? half_m : __uint_as_float((e + 126u) << 23) * (1.0f + half_m);
    return (nibble & 8u) != 0u ? -m : m;
}

extern "C" __global__ void __launch_bounds__(256) opt_bulk_attn(
    const __nv_bfloat16 *__restrict__ q_all,   // [rows][16][512]
    const uint8_t *__restrict__ kv_fp4, const uint8_t *__restrict__ kv_scales,
    const uint8_t *__restrict__ ring_codes, const uint8_t *__restrict__ ring_scales,
    const uint32_t *__restrict__ selected_all, const float *__restrict__ sink,
    __nv_bfloat16 *__restrict__ out_all,       // [rows][16][512]
    const uint32_t p0, const uint32_t ratio, const uint32_t topk, const uint32_t selected_stride,
    const uint32_t ring_rows) {
    OPT_PDL_ENTRY();
    __shared__ __align__(128) uint8_t s_k[OBA_TILE * OBA_LATENT * 2u];
    __shared__ float s_s[2][OBA_HEADS][OBA_TILE];
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

    // S = Q K^T: warp w scores keys 8 (w % 4) .. + 7 over dims 256 (w / 4) .. + 255.
    const uint32_t nt_s = warp & 3u, kh = warp >> 2;
    const __nv_bfloat16 *qrow = q_all + (uint64_t)y * OBA_HEADS * OBA_LATENT;
    uint32_t qa[16][4];
#pragma unroll
    for (uint32_t j = 0; j < 16u; ++j) {
        const uint32_t k0 = kh * 256u + j * 16u + 2u * q;
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

    // The lane's raw bytes of its four keys (warp + 8 i) of a tile: dims 16 lane .. + 15.
    uint4 raw[4];
    uint32_t raw_scale[4], raw_kind[4];
    __syncthreads();   // s_cand
    auto fetch = [&](uint32_t tile) {
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
    };
    auto store = [&]() {
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
                const uint32_t w2[2] = {raw[i].x, raw[i].y};
#pragma unroll
                for (uint32_t j = 0; j < 2u; ++j) {
#pragma unroll
                    for (uint32_t b = 0; b < 4u; ++b) {
                        const uint32_t byte = (w2[j] >> (8u * b)) & 0xffu;
                        v[4u * j + b] = ob_pack(__float2bfloat16_rn(oba_e2m1(byte & 0xfu) * scale),
                                               __float2bfloat16_rn(oba_e2m1(byte >> 4) * scale));
                    }
                }
            } else {
#pragma unroll
                for (uint32_t j = 0; j < 8u; ++j) v[j] = 0u;
            }
            *reinterpret_cast<uint4 *>(&s_k[oba_k_off(key, 2u * lane)]) = make_uint4(v[0], v[1], v[2], v[3]);
            *reinterpret_cast<uint4 *>(&s_k[oba_k_off(key, 2u * lane + 1u)]) = make_uint4(v[4], v[5], v[6], v[7]);
            if (lane == 0u) s_valid[key] = raw_kind[i] != 0u;
        }
    };

    if (tiles > 0u) fetch(0);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        store();
        __syncthreads();
        if (tile + 1u < tiles) fetch(tile + 1u);
        {
            float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const uint32_t key = nt_s * 8u + (lane & 7u);
#pragma unroll
            for (uint32_t j2 = 0; j2 < 8u; ++j2) {
                const uint32_t chunk = kh * 32u + j2 * 4u + (lane >> 3);
                uint32_t b0, b1, b2, b3;
                om_ldsm4(om_smem(&s_k[oba_k_off(key, chunk)]), b0, b1, b2, b3);
                ob_mma(c, qa[2u * j2], b0, b1);
                ob_mma(c, qa[2u * j2 + 1u], b2, b3);
            }
            s_s[kh][g][nt_s * 8u + 2u * q] = c[0];
            s_s[kh][g][nt_s * 8u + 2u * q + 1u] = c[1];
            s_s[kh][g + 8u][nt_s * 8u + 2u * q] = c[2];
            s_s[kh][g + 8u][nt_s * 8u + 2u * q + 1u] = c[3];
        }
        __syncthreads();
        // Online softmax: warp w takes heads 2 w and 2 w + 1, lane = key.
#pragma unroll
        for (uint32_t hh = 0; hh < 2u; ++hh) {
            const uint32_t h = 2u * warp + hh;
            const bool valid = s_valid[lane] != 0u;
            const float score = valid ? (s_s[0][h][lane] + s_s[1][h][lane]) * OBA_SCORE_SCALE : -3.402823466e38f;
            float tile_max = score;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, o));
            const float m_old = s_m[h];
            const float m_new = fmaxf(m_old, tile_max);
            const float p = valid ? expf(score - m_new) : 0.0f;
            s_p[h * OBA_P_LD + lane] = __float2bfloat16_rn(p);
            float sum = p;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (lane == 0u) {
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

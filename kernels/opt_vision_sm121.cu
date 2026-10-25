// kernels/opt_vision_sm121.cu
//
// The vision tower (DECISIONS.md D59): the checkpoint's vision.py ViT and aligner over one image,
// and the copy of an image span's rows into the embedding gather. Rank 3 runs all of it; the image
// placeholder's embedding row is in its vocabulary slice, so the span rows go into its own gather
// slot and the exchange that follows hands them to the other ranks.
//
// Every tensor is bf16 with f32 arithmetic, rounded where the reference's torch ops round: a
// linear layer adds its bias to the f32 accumulator and rounds once; a residual adds the rounded
// projection to the rounded stream and rounds again; SiLU, GELU, RoPE and RMSNorm compute in f32
// and round their outputs. The GEMMs and the attention sum in their own order, so the tower
// matches torch to accumulation-order rounding, not bit for bit.
//
// opt_vision_patchify: the canvas (u8 RGB, rows top to bottom) to the ViT patches, bf16
//   [patches][608]: feature (c, py, px) at c * 196 + py * 14 + px is load_image's
//   (v / 255 - 0.5) / 0.5, and features 588.. are zero (K padded to a multiple of 32). Patches are
//   in row-major (vh, vw) order. One thread per element.
// opt_vision_gemm: out[m][n] = A[m][:k] . W[n][:k] (+ bias[n]), W row-major bf16 [n][k], with
//   mma.sync m16n8k16 bf16 into f32. flags: 1 adds the result to out (the residual), 2 applies the
//   erf GELU to the rounded result. k is a multiple of 32, n of 8, lda and ldo of 8. Grid
//   (ceil(n / 128), ceil(m / 128)), 256 threads: a block owns 128 rows x 128 columns and walks K 32
//   at a time through a double-buffered cp.async pipeline; warp (wm, wn) computes 64 rows x 32
//   columns.
// opt_vision_rmsnorm: out = bf16(w * (x * rsqrt(mean(x^2) + 1e-6))) over rows of 1024, one warp
//   per row, grid ceil(m / 8), 256 threads.
// opt_vision_rope: the 2D rotary embedding on the q and k thirds of the qkv rows, in place. Pair
//   (i, i + 32) of a head turns by hpos * f_i for i < 16 and wpos * f_(i - 16) for 16 <= i < 32,
//   f_j = 1 / 10000^(2j / 32). One thread per pair, grid m * 4, 256 threads.
// opt_vision_attn: bidirectional attention over the image's patches, scale 1/8, one head per
//   blockIdx.y. Grid (ceil(n / 64), 16), 128 threads: a block owns 64 queries (16 per warp) and
//   walks the keys 64 at a time with an online softmax, K and V double-buffered through cp.async.
// opt_vision_swiglu: out[m][j] = bf16(bf16(silu(g[m][j])) * g[m][2816 + j]), one thread per two
//   outputs.
// opt_vision_unfold: the aligner's 3 x 3 unfold of the ViT rows, [rows][9216] with feature
//   c * 9 + ky * 3 + kx, zero past the patch grid's right and bottom edges. One thread per element.
// opt_vision_span_rows: copies 5120-value bf16 rows into the gather slot: pair i of the table is
//   (destination row, source address). Grid count, 256 threads.
// opt_vision_route: the image tokens' expert choice. The reference's gate picks an image-span
//   token's experts with bias_vl instead of bias (model.py Gate); the router kernels have already
//   routed every row with bias, so this re-routes the listed rows: the same stage two as
//   opt_mma_router_topk (biased picks, ties to the lower id, unbiased scores normalised in selection
//   order + 1e-20, x 1.5) with bias_vl, over the row's 384 values in src. raw = 1: src holds the
//   raw gate dots (the bulk router's) and the score is sqrt(softplus(dot)) as there; raw = 0: src
//   holds the scores (the rows router's). One warp per listed row, grid ceil(count / 8), 256
//   threads.

#include "opt_pdl_sm121.cuh"
#include <cuda_bf16.h>
#include <stdint.h>

#define OV_DIM 1024u
#define OV_QKV (3u * OV_DIM)
#define OV_PATCH 14u
#define OV_PATCH_K 588u
#define OV_PATCH_KP 608u
#define OV_INTER 2816u
#define OV_UNFOLD (9u * OV_DIM)
#define OV_ROW 5120u

#define OV_GEMM_RESIDUAL 1u
#define OV_GEMM_GELU 2u

#define OV_BM 128u
#define OV_BN 128u
#define OV_BK 32u
#define OV_LD 40u    // bf16 per shared GEMM row: 32 + 8 padding (80 B), so ldmatrix phases do not conflict

#define OV_AQ 64u    // queries per attention block
#define OV_AK 64u    // keys per attention tile
#define OV_ALD 72u   // bf16 per shared attention row: 64 + 8 padding (144 B)

__device__ __forceinline__ uint32_t ov_smem(const void *p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

// 16 bytes global -> shared, zero-filled when !valid (the source is then not read).
__device__ __forceinline__ void ov_cp16(uint32_t dst, const void *src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void ov_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void ov_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ __forceinline__ void ov_ldsm4(uint32_t addr, uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ov_ldsm4_t(uint32_t addr, uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ov_mma(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t ov_pack(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&v);
}

// F.gelu (erf) on a bf16 value, in f32.
__device__ __forceinline__ float ov_gelu(float x) {
    return x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
}

// F.silu on a bf16 value, in f32.
__device__ __forceinline__ float ov_silu(float x) {
    return __fdiv_rn(x, 1.0f + expf(-x));
}

extern "C" __global__ void __launch_bounds__(256) opt_vision_patchify(
    const uint8_t *__restrict__ canvas, __nv_bfloat16 *__restrict__ out, const uint32_t n_vit_w,
    const uint32_t width, const uint32_t patches) {
    OPT_PDL_ENTRY();
    const uint64_t t = (uint64_t)blockIdx.x * 256u + threadIdx.x;
    const uint64_t patch = t / OV_PATCH_KP;
    const uint32_t f = (uint32_t)(t % OV_PATCH_KP);
    if (patch >= patches) return;
    float v = 0.0f;
    if (f < OV_PATCH_K) {
        const uint32_t c = f / (OV_PATCH * OV_PATCH);
        const uint32_t r = f % (OV_PATCH * OV_PATCH);
        const uint64_t y = (patch / n_vit_w) * OV_PATCH + r / OV_PATCH;
        const uint64_t x = (patch % n_vit_w) * OV_PATCH + r % OV_PATCH;
        const float byte = (float)canvas[(y * width + x) * 3u + c];
        v = __fdiv_rn(__fsub_rn(__fdiv_rn(byte, 255.0f), 0.5f), 0.5f);
    }
    out[t] = __float2bfloat16_rn(v);
}

extern "C" __global__ void __launch_bounds__(256) opt_vision_gemm(
    const __nv_bfloat16 *__restrict__ a, const __nv_bfloat16 *__restrict__ w,
    const __nv_bfloat16 *__restrict__ bias, __nv_bfloat16 *out, const uint32_t m, const uint32_t n,
    const uint32_t k, const uint64_t lda, const uint64_t ldo, const uint32_t flags) {
    OPT_PDL_ENTRY();
    __shared__ __align__(16) __nv_bfloat16 s_a[2][OV_BM * OV_LD];
    __shared__ __align__(16) __nv_bfloat16 s_w[2][OV_BN * OV_LD];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t n0 = blockIdx.x * OV_BN;
    const uint32_t m0 = blockIdx.y * OV_BM;
    const uint32_t k_steps = k / OV_BK;

    // Each thread moves two 16-byte chunks of the A tile and two of the W tile per stage.
    auto load = [&](uint32_t stage, uint32_t ks) {
#pragma unroll
        for (uint32_t i = 0; i < 2u; ++i) {
            const uint32_t idx = tid + i * 256u;
            const uint32_t row = idx >> 2;
            const uint32_t chunk = idx & 3u;
            const uint64_t col = (uint64_t)ks * OV_BK + chunk * 8u;
            const bool a_live = m0 + row < m;
            ov_cp16(ov_smem(&s_a[stage][row * OV_LD + chunk * 8u]),
                    a + (a_live ? (uint64_t)(m0 + row) * lda + col : 0u), a_live);
            const bool w_live = n0 + row < n;
            ov_cp16(ov_smem(&s_w[stage][row * OV_LD + chunk * 8u]),
                    w + (w_live ? (uint64_t)(n0 + row) * k + col : 0u), w_live);
        }
    };

    const uint32_t wm = warp & 1u;   // rows 64 wm .. 64 wm + 63
    const uint32_t wn = warp >> 1;   // columns 32 wn .. 32 wn + 31
    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    load(0u, 0u);
    ov_commit();
    for (uint32_t ks = 0; ks < k_steps; ++ks) {
        const uint32_t stage = ks & 1u;
        if (ks + 1u < k_steps) {
            load(stage ^ 1u, ks + 1u);
            ov_commit();
            ov_wait<1>();
        } else {
            ov_wait<0>();
        }
        __syncthreads();
        const __nv_bfloat16 *sa = s_a[stage];
        const __nv_bfloat16 *sw = s_w[stage];
#pragma unroll
        for (uint32_t kk = 0; kk < 2u; ++kk) {
            const uint32_t k0 = kk * 16u;
            uint32_t b[4][2];
#pragma unroll
            for (uint32_t p = 0; p < 2u; ++p) {
                const uint32_t col = wn * 32u + p * 16u + (lane & 7u) + 8u * (lane >> 4);
                const uint32_t at = k0 + 8u * ((lane >> 3) & 1u);
                ov_ldsm4(ov_smem(&sw[col * OV_LD + at]), b[2 * p][0], b[2 * p][1], b[2 * p + 1][0], b[2 * p + 1][1]);
            }
#pragma unroll
            for (uint32_t mt = 0; mt < 4u; ++mt) {
                const uint32_t row = wm * 64u + mt * 16u + (lane & 15u);
                const uint32_t at = k0 + 8u * (lane >> 4);
                uint32_t af[4];
                ov_ldsm4(ov_smem(&sa[row * OV_LD + at]), af[0], af[1], af[2], af[3]);
#pragma unroll
                for (uint32_t nt = 0; nt < 4u; ++nt) ov_mma(acc[mt][nt], af, b[nt][0], b[nt][1]);
            }
        }
        __syncthreads();
    }

    const uint32_t g = lane >> 2;
    const uint32_t q = lane & 3u;
#pragma unroll
    for (uint32_t mt = 0; mt < 4u; ++mt) {
#pragma unroll
        for (uint32_t half = 0; half < 2u; ++half) {
            const uint32_t row = m0 + wm * 64u + mt * 16u + g + 8u * half;
            if (row >= m) continue;
#pragma unroll
            for (uint32_t nt = 0; nt < 4u; ++nt) {
                const uint32_t col = n0 + wn * 32u + nt * 8u + 2u * q;
                if (col >= n) continue;
                float v0 = acc[mt][nt][2u * half];
                float v1 = acc[mt][nt][2u * half + 1u];
                if (bias != nullptr) {
                    v0 += __bfloat162float(bias[col]);
                    v1 += __bfloat162float(bias[col + 1u]);
                }
                __nv_bfloat162 t = __floats2bfloat162_rn(v0, v1);
                if ((flags & OV_GEMM_GELU) != 0u) {
                    const float2 f = __bfloat1622float2(t);
                    t = __floats2bfloat162_rn(ov_gelu(f.x), ov_gelu(f.y));
                }
                __nv_bfloat162 *dst = reinterpret_cast<__nv_bfloat162 *>(out + (uint64_t)row * ldo + col);
                if ((flags & OV_GEMM_RESIDUAL) != 0u) {
                    const float2 x = __bfloat1622float2(*dst);
                    const float2 y = __bfloat1622float2(t);
                    t = __floats2bfloat162_rn(x.x + y.x, x.y + y.y);
                }
                *dst = t;
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) opt_vision_rmsnorm(
    const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ out, const uint32_t m) {
    OPT_PDL_ENTRY();
    const uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= m) return;
    const uint4 *src = reinterpret_cast<const uint4 *>(x + (uint64_t)row * OV_DIM);
    float v[32];
    float squares = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) {
        const uint4 u = src[lane + 32u * i];
        const uint32_t words[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&words[j]));
            v[8u * i + 2u * j] = f.x;
            v[8u * i + 2u * j + 1u] = f.y;
            squares = __fadd_rn(squares, __fmul_rn(f.x, f.x));
            squares = __fadd_rn(squares, __fmul_rn(f.y, f.y));
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) squares += __shfl_xor_sync(0xffffffffu, squares, offset);
    const float r = rsqrtf(squares / (float)OV_DIM + 1e-6f);
    const uint4 *wsrc = reinterpret_cast<const uint4 *>(weight);
    uint4 *dst = reinterpret_cast<uint4 *>(out + (uint64_t)row * OV_DIM);
#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) {
        const uint4 wu = wsrc[lane + 32u * i];
        const uint32_t words[4] = {wu.x, wu.y, wu.z, wu.w};
        uint32_t packed[4];
#pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const float2 wf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&words[j]));
            packed[j] = ov_pack(__fmul_rn(wf.x, __fmul_rn(v[8u * i + 2u * j], r)),
                                __fmul_rn(wf.y, __fmul_rn(v[8u * i + 2u * j + 1u], r)));
        }
        dst[lane + 32u * i] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    }
}

extern "C" __global__ void __launch_bounds__(256) opt_vision_rope(
    __nv_bfloat16 *__restrict__ qkv, const uint32_t m, const uint32_t n_w) {
    OPT_PDL_ENTRY();
    // Thread t is pair i of head slot s of row r: t = (r * 32 + s) * 32 + i; slots 0..15 are the q
    // heads and 16..31 the k heads, so slot s starts at column 64 s.
    const uint64_t t = (uint64_t)blockIdx.x * 256u + threadIdx.x;
    const uint64_t row = t >> 10;
    if (row >= m) return;
    const uint32_t slot = (uint32_t)(t >> 5) & 31u;
    const uint32_t i = (uint32_t)t & 31u;
    const uint32_t j = i & 15u;
    const float freq = __fdiv_rn(1.0f, powf(10000.0f, __fdiv_rn((float)(2u * j), 32.0f)));
    const float pos = (float)(i < 16u ? row / n_w : row % n_w);
    const float angle = __fmul_rn(pos, freq);
    const float c = cosf(angle);
    const float s = sinf(angle);
    __nv_bfloat16 *head = qkv + row * OV_QKV + slot * 64u;
    const float x1 = __bfloat162float(head[i]);
    const float x2 = __bfloat162float(head[i + 32u]);
    head[i] = __float2bfloat16_rn(__fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, s)));
    head[i + 32u] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(x2, c), __fmul_rn(x1, s)));
}

extern "C" __global__ void __launch_bounds__(128) opt_vision_attn(
    const __nv_bfloat16 *__restrict__ qkv, __nv_bfloat16 *__restrict__ out, const uint32_t n) {
    OPT_PDL_ENTRY();
    __shared__ __align__(16) __nv_bfloat16 s_q[OV_AQ * OV_ALD];
    __shared__ __align__(16) __nv_bfloat16 s_k[2][OV_AK * OV_ALD];
    __shared__ __align__(16) __nv_bfloat16 s_v[2][OV_AK * OV_ALD];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t head = blockIdx.y;
    const uint32_t q0 = blockIdx.x * OV_AQ;
    const uint32_t g = lane >> 2;
    const uint32_t q = lane & 3u;

    // 64 rows x 8 chunks of 16 bytes: four per thread.
#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) {
        const uint32_t idx = tid + i * 128u;
        const uint32_t row = idx >> 3;
        const uint32_t chunk = idx & 7u;
        const bool live = q0 + row < n;
        ov_cp16(ov_smem(&s_q[row * OV_ALD + chunk * 8u]),
                qkv + (live ? (uint64_t)(q0 + row) * OV_QKV + head * 64u + chunk * 8u : 0u), live);
    }
    auto load_kv = [&](uint32_t stage, uint32_t tile) {
#pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) {
            const uint32_t idx = tid + i * 128u;
            const uint32_t row = idx >> 3;
            const uint32_t chunk = idx & 7u;
            const uint32_t key = tile * OV_AK + row;
            const bool live = key < n;
            const uint64_t at = live ? (uint64_t)key * OV_QKV + head * 64u + chunk * 8u : 0u;
            ov_cp16(ov_smem(&s_k[stage][row * OV_ALD + chunk * 8u]), qkv + (live ? at + OV_DIM : 0u), live);
            ov_cp16(ov_smem(&s_v[stage][row * OV_ALD + chunk * 8u]), qkv + (live ? at + 2u * OV_DIM : 0u), live);
        }
    };
    load_kv(0u, 0u);
    ov_commit();

    const uint32_t tiles = (n + OV_AK - 1u) / OV_AK;
    // softmax(s / 8) through exp2: the scores are scaled by log2(e) / 8.
    const float scale = 0.125f * 1.4426950408889634f;
    uint32_t qf[4][4];
    float o[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[i][e] = 0.0f;
    // Rows g and g + 8 of the warp's 16: running max (scaled) and this thread's share of the sum.
    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.0f, 0.0f};

    for (uint32_t tile = 0; tile < tiles; ++tile) {
        const uint32_t stage = tile & 1u;
        if (tile + 1u < tiles) {
            load_kv(stage ^ 1u, tile + 1u);
            ov_commit();
            ov_wait<1>();
        } else {
            ov_wait<0>();
        }
        __syncthreads();
        if (tile == 0u) {
#pragma unroll
            for (uint32_t ks = 0; ks < 4u; ++ks) {
                const uint32_t row = warp * 16u + (lane & 15u);
                ov_ldsm4(ov_smem(&s_q[row * OV_ALD + ks * 16u + 8u * (lane >> 4)]), qf[ks][0], qf[ks][1], qf[ks][2],
                         qf[ks][3]);
            }
        }
        const __nv_bfloat16 *sk = s_k[stage];
        const __nv_bfloat16 *sv = s_v[stage];
        float s[8][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int e = 0; e < 4; ++e) s[i][e] = 0.0f;
#pragma unroll
        for (uint32_t ks = 0; ks < 4u; ++ks) {
#pragma unroll
            for (uint32_t p = 0; p < 4u; ++p) {
                const uint32_t key = p * 16u + (lane & 7u) + 8u * (lane >> 4);
                const uint32_t dim = ks * 16u + 8u * ((lane >> 3) & 1u);
                uint32_t b0, b1, b2, b3;
                ov_ldsm4(ov_smem(&sk[key * OV_ALD + dim]), b0, b1, b2, b3);
                ov_mma(s[2u * p], qf[ks], b0, b1);
                ov_mma(s[2u * p + 1u], qf[ks], b2, b3);
            }
        }
        if ((tile + 1u) * OV_AK > n) {
#pragma unroll
            for (uint32_t nt = 0; nt < 8u; ++nt)
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e)
                    if (tile * OV_AK + nt * 8u + 2u * q + (e & 1u) >= n) s[nt][e] = -INFINITY;
        }
#pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            float top = -INFINITY;
#pragma unroll
            for (uint32_t nt = 0; nt < 8u; ++nt) top = fmaxf(top, fmaxf(s[nt][2u * r], s[nt][2u * r + 1u]));
            top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
            top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
            const float next = fmaxf(row_max[r], top * scale);
            const float alpha = exp2f(row_max[r] - next);
            row_max[r] = next;
            float sum = 0.0f;
#pragma unroll
            for (uint32_t nt = 0; nt < 8u; ++nt) {
                s[nt][2u * r] = exp2f(s[nt][2u * r] * scale - next);
                s[nt][2u * r + 1u] = exp2f(s[nt][2u * r + 1u] * scale - next);
                sum += s[nt][2u * r] + s[nt][2u * r + 1u];
                o[nt][2u * r] *= alpha;
                o[nt][2u * r + 1u] *= alpha;
            }
            row_sum[r] = row_sum[r] * alpha + sum;
        }
#pragma unroll
        for (uint32_t kk = 0; kk < 4u; ++kk) {
            const uint32_t pa[4] = {ov_pack(s[2u * kk][0], s[2u * kk][1]), ov_pack(s[2u * kk][2], s[2u * kk][3]),
                                    ov_pack(s[2u * kk + 1u][0], s[2u * kk + 1u][1]),
                                    ov_pack(s[2u * kk + 1u][2], s[2u * kk + 1u][3])};
#pragma unroll
            for (uint32_t dp = 0; dp < 4u; ++dp) {
                const uint32_t key = kk * 16u + (lane & 7u) + 8u * ((lane >> 3) & 1u);
                const uint32_t dim = dp * 16u + 8u * (lane >> 4);
                uint32_t b0, b1, b2, b3;
                ov_ldsm4_t(ov_smem(&sv[key * OV_ALD + dim]), b0, b1, b2, b3);
                ov_mma(o[2u * dp], pa, b0, b1);
                ov_mma(o[2u * dp + 1u], pa, b2, b3);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t r = 0; r < 2u; ++r) {
        float sum = row_sum[r];
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        const float inv = 1.0f / sum;
        const uint32_t row = q0 + warp * 16u + g + 8u * r;
        if (row >= n) continue;
        __nv_bfloat16 *dst = out + (uint64_t)row * OV_DIM + head * 64u + 2u * q;
#pragma unroll
        for (uint32_t nt = 0; nt < 8u; ++nt)
            *reinterpret_cast<__nv_bfloat162 *>(dst + nt * 8u) =
                __floats2bfloat162_rn(o[nt][2u * r] * inv, o[nt][2u * r + 1u] * inv);
    }
}

extern "C" __global__ void __launch_bounds__(256) opt_vision_swiglu(
    const __nv_bfloat16 *__restrict__ g, __nv_bfloat16 *__restrict__ out, const uint32_t m) {
    OPT_PDL_ENTRY();
    const uint64_t t = (uint64_t)blockIdx.x * 256u + threadIdx.x;
    const uint64_t row = t / (OV_INTER / 2u);
    if (row >= m) return;
    const uint32_t j = (uint32_t)(t % (OV_INTER / 2u)) * 2u;
    const __nv_bfloat16 *src = g + row * (2u * OV_INTER) + j;
    const float2 gate = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(src));
    const float2 up = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(src + OV_INTER));
    const float2 act = __bfloat1622float2(__floats2bfloat162_rn(ov_silu(gate.x), ov_silu(gate.y)));
    *reinterpret_cast<__nv_bfloat162 *>(out + row * OV_INTER + j) =
        __floats2bfloat162_rn(__fmul_rn(act.x, up.x), __fmul_rn(act.y, up.y));
}

extern "C" __global__ void __launch_bounds__(256) opt_vision_unfold(
    const __nv_bfloat16 *__restrict__ x, __nv_bfloat16 *__restrict__ out, const uint32_t n_vit_h,
    const uint32_t n_vit_w, const uint32_t n_llm_w, const uint32_t rows) {
    OPT_PDL_ENTRY();
    const uint64_t t = (uint64_t)blockIdx.x * 256u + threadIdx.x;
    const uint64_t row = t / OV_UNFOLD;
    if (row >= rows) return;
    const uint32_t f = (uint32_t)(t % OV_UNFOLD);
    const uint32_t c = f / 9u;
    const uint32_t k = f % 9u;
    const uint64_t y = (row / n_llm_w) * 3u + k / 3u;
    const uint64_t xx = (row % n_llm_w) * 3u + k % 3u;
    out[t] = y < n_vit_h && xx < n_vit_w ? x[(y * n_vit_w + xx) * OV_DIM + c] : __float2bfloat16_rn(0.0f);
}

extern "C" __global__ void __launch_bounds__(256) opt_vision_span_rows(
    const uint64_t *__restrict__ table, __nv_bfloat16 *__restrict__ dst) {
    OPT_PDL_ENTRY();
    const uint64_t row = table[2u * blockIdx.x];
    const uint4 *src = reinterpret_cast<const uint4 *>(table[2u * blockIdx.x + 1u]);
    uint4 *to = reinterpret_cast<uint4 *>(dst + row * OV_ROW);
    for (uint32_t i = threadIdx.x; i < OV_ROW / 8u; i += 256u) to[i] = src[i];
}

#define OV_EXPERTS 384u
#define OV_TOPK 6u
extern "C" __global__ void __launch_bounds__(256) opt_vision_route(
    const float *__restrict__ src, const uint32_t raw, const float *__restrict__ bias,
    const uint32_t *__restrict__ rows, const uint32_t count, uint32_t *__restrict__ ids,
    float *__restrict__ weights) {
    OPT_PDL_ENTRY();
    constexpr uint32_t PER_LANE = OV_EXPERTS / 32u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t i = blockIdx.x * 8u + (threadIdx.x >> 5);
    if (i >= count) return;
    const uint64_t t = rows[i];
    float sc[PER_LANE], sig[PER_LANE];
#pragma unroll
    for (uint32_t j = 0; j < PER_LANE; ++j) {
        const uint32_t e = lane + 32u * j;
        const float value = src[t * OV_EXPERTS + e];
        if (raw) {
            const float sp = value > 20.0f ? value : log1pf(expf(value));
            sc[j] = sqrtf(sp);
        } else {
            sc[j] = value;
        }
        sig[j] = __fadd_rn(sc[j], bias[e]);
    }
    uint32_t chosen[OV_TOPK];
    float picked[OV_TOPK];
#pragma unroll
    for (uint32_t k = 0; k < OV_TOPK; ++k) {
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
        for (uint32_t s = 0; s < OV_TOPK; ++s) sum = __fadd_rn(sum, picked[s]);
        const float den = __fadd_rn(sum, 1e-20f);
#pragma unroll
        for (uint32_t s = 0; s < OV_TOPK; ++s) {
            ids[t * OV_TOPK + s] = chosen[s];
            weights[t * OV_TOPK + s] = __fmul_rn(picked[s] / den, 1.5f);
        }
    }
}

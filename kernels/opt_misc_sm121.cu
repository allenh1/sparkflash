// kernels/opt_misc_sm121.cu
//
// Small decode-step kernels that replace launch-per-item loops.
//
// opt_rope_heads_bf16: the q rope of every head of the rank in one launch, in place. Grid (heads,
//   tokens), one warp per head; the arithmetic is rope_rotate_bf16's (adjacent pairs, rounded
//   multiplies and adds, one bf16 rounding), so the result is bit-identical to the per-head launch.

#include "opt_pdl_sm121.cuh"
#include <cuda_bf16.h>
#include <stdint.h>

#include "m4_common_sm121.cuh"
#include "m4_activation_quant_sm121.cuh"

extern "C" __global__ void opt_rope_heads_bf16(
    __nv_bfloat16 *__restrict__ rows,   // [tokens][heads][head_width], rotated in place
    const float *__restrict__ cos,      // [tokens][pair_stride] (this position's cos at token 0)
    const float *__restrict__ sin,
    const uint32_t head_width,
    const uint32_t rope_head_dim,
    const uint32_t heads,
    const uint32_t pair_stride) {
    OPT_PDL_ENTRY();       // floats between consecutive tokens' cos rows
    const uint32_t head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    __nv_bfloat16 *row = rows + ((uint64_t)token * heads + head) * head_width;
    const float *c = cos + (uint64_t)token * pair_stride;
    const float *s = sin + (uint64_t)token * pair_stride;
    const uint32_t start = head_width - rope_head_dim;
    const uint32_t pairs = rope_head_dim >> 1u;
    for (uint32_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
        const float a = __bfloat162float(row[start + 2u * pair]);
        const float b = __bfloat162float(row[start + 2u * pair + 1u]);
        const float first = __fadd_rn(__fmul_rn(a, c[pair]), -__fmul_rn(b, s[pair]));
        const float second = __fadd_rn(__fmul_rn(a, s[pair]), __fmul_rn(b, c[pair]));
        row[start + 2u * pair] = __float2bfloat16_rn(first);
        row[start + 2u * pair + 1u] = __float2bfloat16_rn(second);
    }
}

// opt_rmsnorm_bf16: rmsnorm_sm121's ABI with one block of 256 per row instead of one thread. The
//   per-element expressions are the reference's (rounded square-and-add, inv = 1/sqrt(mean + eps),
//   (x * inv) * w, bf16 RNE); the sum of squares is a block tree instead of a serial scan.
__device__ __forceinline__ float opt_m_bf16lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float opt_m_bf16hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

extern "C" __global__ void __launch_bounds__(256) opt_rmsnorm_bf16(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ out,
    const uint32_t rows,
    const uint32_t cols,
    const float eps,
    void *trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    OPT_PDL_ENTRY();
    (void)trace_ring; (void)layer_id; (void)step_seq;
    __shared__ float s_part[8];
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16 *x_row = x + (uint64_t)row * cols;
    __nv_bfloat16 *out_row = out + (uint64_t)row * cols;
    const uint32_t t = threadIdx.x;
    float sum = 0.0f;
    const bool vectorised = (cols & 7u) == 0u;
    if (vectorised) {
        const uint4 *xv = reinterpret_cast<const uint4 *>(x_row);
        for (uint32_t c = t; c < (cols >> 3); c += 256u) {
            const uint4 v = __ldg(xv + c);
            const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float lo = opt_m_bf16lo(w[k]), hi = opt_m_bf16hi(w[k]);
                sum = __fadd_rn(sum, __fmul_rn(lo, lo));
                sum = __fadd_rn(sum, __fmul_rn(hi, hi));
            }
        }
    } else {
        for (uint32_t c = t; c < cols; c += 256u) {
            const float v = __bfloat162float(x_row[c]);
            sum = __fadd_rn(sum, __fmul_rn(v, v));
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
    if ((t & 31u) == 0u) s_part[t >> 5] = sum;
    __syncthreads();
    float total = s_part[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) total = __fadd_rn(total, s_part[w]);
    const float mean = __fdiv_rn(total, (float)cols);
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
    if (vectorised) {
        const uint4 *xv = reinterpret_cast<const uint4 *>(x_row);
        const uint4 *wv = reinterpret_cast<const uint4 *>(weight);
        uint4 *ov = reinterpret_cast<uint4 *>(out_row);
        for (uint32_t c = t; c < (cols >> 3); c += 256u) {
            const uint4 v = __ldg(xv + c);
            const uint4 g = __ldg(wv + c);
            const uint32_t a[4] = {v.x, v.y, v.z, v.w};
            const uint32_t b[4] = {g.x, g.y, g.z, g.w};
            uint32_t o[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float lo = __fmul_rn(__fmul_rn(opt_m_bf16lo(a[k]), inv), opt_m_bf16lo(b[k]));
                const float hi = __fmul_rn(__fmul_rn(opt_m_bf16hi(a[k]), inv), opt_m_bf16hi(b[k]));
                const __nv_bfloat162 pair = __floats2bfloat162_rn(lo, hi);
                o[k] = *reinterpret_cast<const uint32_t *>(&pair);
            }
            ov[c] = make_uint4(o[0], o[1], o[2], o[3]);
        }
    } else {
        for (uint32_t c = t; c < cols; c += 256u) {
            out_row[c] = __float2bfloat16_rn(
                __fmul_rn(__fmul_rn(__bfloat162float(x_row[c]), inv), __bfloat162float(weight[c])));
        }
    }
}

// opt_lm_head: cert_lm_head_sm121's ABI and grid (one block per 1024-entry chunk) computing every
//   logit exactly - warp per row, sixteen-byte weight loads, the hidden row in shared memory - so no
//   entry is pruned (certified is all zero and the count stays at the caller's zero). The chunk's
//   max/argmax keeps the reference tie rule (lower entry wins).
#define OPT_LM_HIDDEN 5120u
#define OPT_LM_THREADS 512u
#define OPT_LM_WARPS (OPT_LM_THREADS / 32u)
#define OPT_LM_CHUNK 1024u

__device__ __forceinline__ uint4 opt_m_ldw(const void *p) {
    uint4 v;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ float opt_m_dot8(uint4 w, uint4 h, float acc) {
    const uint32_t a[4] = {w.x, w.y, w.z, w.w};
    const uint32_t b[4] = {h.x, h.y, h.z, h.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        acc = fmaf(opt_m_bf16lo(a[k]), opt_m_bf16lo(b[k]), acc);
        acc = fmaf(opt_m_bf16hi(a[k]), opt_m_bf16hi(b[k]), acc);
    }
    return acc;
}

extern "C" __global__ void __launch_bounds__(OPT_LM_THREADS) opt_lm_head(
    const __nv_bfloat16 *__restrict__ hidden,
    const __nv_bfloat16 *__restrict__ w,
    const __nv_bfloat16 *__restrict__ w_block_max,
    float *__restrict__ logits,
    uint8_t *__restrict__ certified,
    unsigned int *__restrict__ certified_count,
    float *__restrict__ block_max,
    uint32_t *__restrict__ block_argmax,
    const uint32_t vocab,
    void *trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    OPT_PDL_ENTRY();
    (void)w_block_max; (void)certified_count; (void)trace_ring; (void)layer_id; (void)step_seq;
    __shared__ uint4 s_h[OPT_LM_HIDDEN / 8u];
    __shared__ float s_best[OPT_LM_WARPS];
    __shared__ uint32_t s_best_i[OPT_LM_WARPS];
    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31u;
    const uint32_t warp = t >> 5;
    for (uint32_t i = t; i < OPT_LM_HIDDEN / 8u; i += OPT_LM_THREADS) s_h[i] = reinterpret_cast<const uint4 *>(hidden)[i];
    __syncthreads();
    const uint32_t base = blockIdx.x * OPT_LM_CHUNK;
    float best = -3.402823466e38f;
    uint32_t best_i = 0xffffffffu;
    for (uint32_t r = warp; r < OPT_LM_CHUNK; r += OPT_LM_WARPS) {
        const uint32_t entry = base + r;
        if (entry >= vocab) break;
        const uint4 *row = reinterpret_cast<const uint4 *>(w + (uint64_t)entry * OPT_LM_HIDDEN);
        uint4 wv[20];
#pragma unroll
        for (uint32_t i = 0; i < 20u; ++i) wv[i] = opt_m_ldw(row + lane + 32u * i);
        float acc = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < 20u; ++i) acc = opt_m_dot8(wv[i], s_h[lane + 32u * i], acc);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0u) { logits[entry] = acc; certified[entry] = 0u; }
        if (best_i == 0xffffffffu || acc > best) { best = acc; best_i = entry; }
    }
    if (lane == 0u) { s_best[warp] = best; s_best_i[warp] = best_i; }
    __syncthreads();
    if (t == 0u) {
        float v = s_best[0];
        uint32_t idx = s_best_i[0];
        for (uint32_t k = 1; k < OPT_LM_WARPS; ++k) {
            const uint32_t oi = s_best_i[k];
            if (oi == 0xffffffffu) continue;
            if (idx == 0xffffffffu || s_best[k] > v || (s_best[k] == v && oi < idx)) { v = s_best[k]; idx = oi; }
        }
        block_max[blockIdx.x] = v;
        block_argmax[blockIdx.x] = idx == 0xffffffffu ? 0u : idx;
    }
}

// opt_engram_wkv: engram_wkv_sm121's ABI and grid (warp per output row, 8 rows per block) with
//   sixteen-byte loads: lane l reads codes [16 l, 16 l + 16) of each 512-code step, the lane pair
//   folds one 32-code block, and the block's dot is scaled (dot * act_scale) * weight_scale as the
//   reference does before it joins the row sum. Only the summation order differs.
#include <cuda_fp16.h>

__device__ __forceinline__ void opt_m_e4m3x4(uint32_t w, float *f) {
    uint32_t h01, h23;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h01) : "h"((uint16_t)(w & 0xffffu)));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h23) : "h"((uint16_t)(w >> 16)));
    const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&h01));
    const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&h23));
    f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}


extern "C" __global__ void __launch_bounds__(256) opt_engram_wkv(
    const uint8_t *__restrict__ act_codes,
    const uint8_t *__restrict__ act_scales,
    const uint8_t *__restrict__ weight,
    const uint8_t *__restrict__ weight_scales,
    __nv_bfloat16 *__restrict__ out,
    const uint32_t out_width,
    const uint32_t inner,
    void *trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    OPT_PDL_ENTRY();
    (void)trace_ring; (void)layer_id; (void)step_seq;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5);
    if (row >= out_width) return;
    const uint32_t k_blocks = inner / 32u;
    const uint8_t *weight_row = weight + (uint64_t)row * inner;
    const uint8_t *scale_row = weight_scales + (uint64_t)(row / 32u) * k_blocks;
    float acc = 0.0f;
#pragma unroll 4
    for (uint32_t base = 0; base < inner; base += 512u) {
        const uint32_t col = base + 16u * lane;
        float d = 0.0f;
        if (col < inner) {
            const uint4 wv = opt_m_ldw(weight_row + col);
            const uint4 av = __ldg(reinterpret_cast<const uint4 *>(act_codes + col));
            const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
            const uint32_t aa[4] = {av.x, av.y, av.z, av.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float wf[4], af[4];
                opt_m_e4m3x4(ww[k], wf);
                opt_m_e4m3x4(aa[k], af);
#pragma unroll
                for (int j = 0; j < 4; ++j) d = fmaf(af[j], wf[j], d);
            }
        }
        d += __shfl_xor_sync(0xffffffffu, d, 1);
        if ((lane & 1u) == 0u && col < inner) {
            const uint32_t kb = col / 32u;
            acc = __fadd_rn(acc, __fmul_rn(__fmul_rn(d, spark_ue8m0_decode(act_scales[kb])), spark_ue8m0_decode(scale_row[kb])));
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0u) out[row] = __float2bfloat16_rn(acc);
}

// opt_kv_norm_rope: kv_norm_rope_bf16's ABI (one row) with a block of 256 instead of one thread. The
//   per-value expressions are the reference's; the sum of squares is a block tree.
extern "C" __global__ void __launch_bounds__(256) opt_kv_norm_rope(
    const __nv_bfloat16 *__restrict__ row,
    const __nv_bfloat16 *__restrict__ norm,
    const float eps,
    const float *__restrict__ cos,
    const float *__restrict__ sin,
    __nv_bfloat16 *__restrict__ out,
    const uint32_t head_width,
    const uint32_t rope_head_dim) {
    OPT_PDL_ENTRY();
    __shared__ float s_part[8];
    const uint32_t t = threadIdx.x;
    const uint32_t pairs = head_width >> 1u;
    const uint32_t *row2 = reinterpret_cast<const uint32_t *>(row);
    float sum = 0.0f;
    for (uint32_t p = t; p < pairs; p += 256u) {
        const uint32_t w = __ldg(row2 + p);
        const float a = opt_m_bf16lo(w), b = opt_m_bf16hi(w);
        sum = __fadd_rn(sum, __fmul_rn(a, a));
        sum = __fadd_rn(sum, __fmul_rn(b, b));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
    if ((t & 31u) == 0u) s_part[t >> 5] = sum;
    __syncthreads();
    float total = s_part[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) total = __fadd_rn(total, s_part[w]);
    const float mean = __fdiv_rn(total, (float)head_width);
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
    const uint32_t start_pair = (head_width - rope_head_dim) >> 1u;
    const uint32_t *norm2 = reinterpret_cast<const uint32_t *>(norm);
    uint32_t *out2 = reinterpret_cast<uint32_t *>(out);
    for (uint32_t p = t; p < pairs; p += 256u) {
        const uint32_t w = __ldg(row2 + p), n = __ldg(norm2 + p);
        float a = __fmul_rn(__fmul_rn(opt_m_bf16lo(w), inv), opt_m_bf16lo(n));
        float b = __fmul_rn(__fmul_rn(opt_m_bf16hi(w), inv), opt_m_bf16hi(n));
        if (p >= start_pair) {
            const uint32_t q = p - start_pair;
            const float c = __ldg(cos + q), s = __ldg(sin + q);
            const float first = __fadd_rn(__fmul_rn(a, c), -__fmul_rn(b, s));
            const float second = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
            a = first;
            b = second;
        }
        const __nv_bfloat162 o2 = __floats2bfloat162_rn(a, b);
        out2[p] = *reinterpret_cast<const uint32_t *>(&o2);
    }
}

// opt_engram_gate_write: engram_gate_write_sm121's ABI with one block of 256 per hc copy (grid hc).
//   A copy's output only needs its own gate, so the copies are independent; the three sums are block
//   trees instead of a serial scan. Gate arithmetic is the reference's.
extern "C" __global__ void __launch_bounds__(256) opt_engram_gate_write(
    const __nv_bfloat16 *__restrict__ kv,
    const __nv_bfloat16 *__restrict__ stream,
    const float *__restrict__ product,
    __nv_bfloat16 *__restrict__ out,
    float *__restrict__ gates,
    const uint32_t dim,
    const uint32_t hc,
    const float eps,
    void *trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    OPT_PDL_ENTRY();
    (void)trace_ring; (void)layer_id; (void)step_seq;
    __shared__ float s_part[3][8];
    __shared__ float s_gate;
    const uint32_t copy = blockIdx.x;
    if (copy >= hc) return;
    const uint32_t t = threadIdx.x;
    const uint32_t *h2 = reinterpret_cast<const uint32_t *>(stream + (uint64_t)copy * dim);
    const uint32_t *k2 = reinterpret_cast<const uint32_t *>(kv + (uint64_t)copy * dim);
    const float2 *p2 = reinterpret_cast<const float2 *>(product + (uint64_t)copy * dim);
    float hs = 0.0f, ks = 0.0f, dot = 0.0f;
    for (uint32_t p = t; p < (dim >> 1); p += 256u) {
        const uint32_t hw = __ldg(h2 + p), kw = __ldg(k2 + p);
        const float2 pr = __ldg(p2 + p);
        const float h0 = opt_m_bf16lo(hw), h1 = opt_m_bf16hi(hw);
        const float k0 = opt_m_bf16lo(kw), k1 = opt_m_bf16hi(kw);
        hs = __fadd_rn(__fadd_rn(hs, __fmul_rn(h0, h0)), __fmul_rn(h1, h1));
        ks = __fadd_rn(__fadd_rn(ks, __fmul_rn(k0, k0)), __fmul_rn(k1, k1));
        dot = __fadd_rn(__fadd_rn(dot, __fmul_rn(__fmul_rn(h0, pr.x), k0)), __fmul_rn(__fmul_rn(h1, pr.y), k1));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        hs = __fadd_rn(hs, __shfl_xor_sync(0xffffffffu, hs, o));
        ks = __fadd_rn(ks, __shfl_xor_sync(0xffffffffu, ks, o));
        dot = __fadd_rn(dot, __shfl_xor_sync(0xffffffffu, dot, o));
    }
    if ((t & 31u) == 0u) { s_part[0][t >> 5] = hs; s_part[1][t >> 5] = ks; s_part[2][t >> 5] = dot; }
    __syncthreads();
    if (t == 0u) {
        float a = s_part[0][0], b = s_part[1][0], d = s_part[2][0];
#pragma unroll
        for (int w = 1; w < 8; ++w) { a = __fadd_rn(a, s_part[0][w]); b = __fadd_rn(b, s_part[1][w]); d = __fadd_rn(d, s_part[2][w]); }
        const float dim_f = (float)dim;
        const float stream_inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(a, dim_f), eps)));
        const float key_inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(b, dim_f), eps)));
        const float rstd = __fmul_rn(stream_inv, key_inv);
        const float scale = __fdiv_rn(1.0f, __fsqrt_rn(dim_f));
        const float scaled_dot = __fmul_rn(__fmul_rn(d, rstd), scale);
        const float magnitude = fabsf(scaled_dot);
        const float clamped = magnitude > 1.0e-6f ? magnitude : 1.0e-6f;
        const float root = __fsqrt_rn(clamped);
        const float signed_root = scaled_dot < 0.0f ? -root : root;
        const float gate = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-signed_root)));
        s_gate = gate;
        gates[copy] = gate;
    }
    __syncthreads();
    const float gate = s_gate;
    const uint32_t *v2 = reinterpret_cast<const uint32_t *>(kv + (uint64_t)hc * dim);
    uint32_t *o2 = reinterpret_cast<uint32_t *>(out + (uint64_t)copy * dim);
    for (uint32_t p = t; p < (dim >> 1); p += 256u) {
        const uint32_t hw = __ldg(h2 + p), vw = __ldg(v2 + p);
        const __nv_bfloat162 r = __floats2bfloat162_rn(
            __fadd_rn(opt_m_bf16lo(hw), __fmul_rn(gate, opt_m_bf16lo(vw))),
            __fadd_rn(opt_m_bf16hi(hw), __fmul_rn(gate, opt_m_bf16hi(vw))));
        o2[p] = *reinterpret_cast<const uint32_t *>(&r);
    }
}

// opt_sum_ranks_bf16: the one-shot all-reduce's local fold. out[j] = bf16(sum over r = 0..world-1 in
//   rank order of row_r[j]), where row_r is `own` for this rank and gather[r * count ..] otherwise.
//   Every rank folds the same rows in the same order, so every rank gets the same bits.
extern "C" __global__ void __launch_bounds__(256) opt_sum_ranks_bf16(
    const __nv_bfloat16 *__restrict__ gather,
    const __nv_bfloat16 *__restrict__ own,
    __nv_bfloat16 *__restrict__ out,
    const uint32_t count,
    const uint32_t rank,
    const uint32_t world) {
    OPT_PDL_ENTRY();
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (2u * p >= count) return;
    const uint32_t *own2 = reinterpret_cast<const uint32_t *>(own);
    const uint32_t *g2 = reinterpret_cast<const uint32_t *>(gather);
    float a = 0.0f, b = 0.0f;
    for (uint32_t r = 0; r < world; ++r) {
        const uint32_t w = r == rank ? own2[p] : g2[(uint64_t)r * (count >> 1) + p];
        a = __fadd_rn(a, opt_m_bf16lo(w));
        b = __fadd_rn(b, opt_m_bf16hi(w));
    }
    const __nv_bfloat162 o = __floats2bfloat162_rn(a, b);
    reinterpret_cast<uint32_t *>(out)[p] = *reinterpret_cast<const uint32_t *>(&o);
}

// opt_pool_rows: pool_group_f32 over a chunk's completed compressor groups at once. Group i pools
//   rows i * ratio .. i * ratio + ratio - 1 of kv / score ([rows][head_dim] f32) into dest[i]; the
//   arithmetic per element is pool_group_f32's (the max, the ascending exp sum, the ascending
//   weighted sum with a division per row). Grid (ceil(head_dim / 128), groups), 128 threads.
extern "C" __global__ void __launch_bounds__(128) opt_pool_rows(
    const float *__restrict__ kv, const float *__restrict__ score, float *__restrict__ dest,
    const uint32_t ratio, const uint32_t head_dim) {
    OPT_PDL_ENTRY();
    const uint32_t element = blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= head_dim) return;
    const uint64_t base = (uint64_t)blockIdx.y * ratio * head_dim;
    kv += base;
    score += base;
    float *out = dest + (uint64_t)blockIdx.y * head_dim;
    float max = -__builtin_huge_valf();
    for (uint32_t row = 0; row < ratio; ++row) max = fmaxf(max, score[(uint64_t)row * head_dim + element]);
    if (!isfinite(max)) {
        out[element] = 0.0f;
        return;
    }
    float total = 0.0f;
    float acc = 0.0f;
    for (uint32_t row = 0; row < ratio; ++row) total += expf(score[(uint64_t)row * head_dim + element] - max);
    for (uint32_t row = 0; row < ratio; ++row) {
        const float weight = expf(score[(uint64_t)row * head_dim + element] - max);
        const float normalised = total > 0.0f ? weight / total : 0.0f;
        acc += normalised * kv[(uint64_t)row * head_dim + element];
    }
    out[element] = acc;
}

// opt_k_norm_rows: k_norm_f32_to_bf16 over `rows` key rows, one thread per row with that kernel's
//   sequential sums, so every row is bit-identical to the single-row launch. Grid ceil(rows / 128).
extern "C" __global__ void __launch_bounds__(128) opt_k_norm_rows(
    const float *__restrict__ x, const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ out, const uint32_t cols, const float eps, const uint32_t rows) {
    OPT_PDL_ENTRY();
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || cols == 0u) return;
    const float *xr = x + (uint64_t)r * cols;
    __nv_bfloat16 *o = out + (uint64_t)r * cols;
    float sum = 0.0f;
    for (uint32_t c = 0u; c < cols; ++c) sum = __fadd_rn(sum, __fmul_rn(xr[c], xr[c]));
    const float mean = __fdiv_rn(sum, static_cast<float>(cols));
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
    for (uint32_t c = 0u; c < cols; ++c) o[c] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(xr[c], inv), __bfloat162float(weight[c])));
}

// opt_bulk_engram_embed: engram_embed_sm121 over a chunk's rows at once. Row r's arena entry is
//   entries + r * stride (its scales at + scales_at); a thread re-quantises one 32-value block, which
//   shares one source scale, with that kernel's arithmetic, so every row is bit-identical to the
//   single-row launch. Grid (rows), 256 threads.
extern "C" __global__ void __launch_bounds__(256) opt_bulk_engram_embed(
    const uint8_t *__restrict__ entries, const uint64_t stride, const uint64_t scales_at,
    uint8_t *__restrict__ act_codes, uint8_t *__restrict__ act_scales, const uint32_t inner) {
    OPT_PDL_ENTRY();
    const uint32_t blocks = inner / SPARK_M4_QUANT_BLOCK;
    const uint8_t *codes = entries + (uint64_t)blockIdx.x * stride;
    const uint8_t *scales = codes + scales_at;
    uint8_t *out_codes = act_codes + (uint64_t)blockIdx.x * inner;
    uint8_t *out_scales = act_scales + (uint64_t)blockIdx.x * blocks;
    for (uint32_t block = threadIdx.x; block < blocks; block += blockDim.x) {
        const uint4 *src = reinterpret_cast<const uint4 *>(codes + block * SPARK_M4_QUANT_BLOCK);
        uint4 raw[2] = {__ldg(src), __ldg(src + 1)};
        const uint8_t *in = reinterpret_cast<const uint8_t *>(raw);
        const float source_scale = spark_ue8m0_decode(__ldg(scales + block));
        float values[SPARK_M4_QUANT_BLOCK];
        float amax = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < SPARK_M4_QUANT_BLOCK; ++i) {
            const float value = __bfloat162float(
                __float2bfloat16_rn(__fmul_rn(spark_fp8_e4m3_decode(in[i]), source_scale)));
            values[i] = value;
            const float magnitude = value < 0.0f ? -value : value;
            amax = magnitude > amax ? magnitude : amax;
        }
        const uint8_t scale_code = spark_act_quant_scale_code(amax);
        out_scales[block] = scale_code;
        const float scale = spark_ue8m0_decode(scale_code);
        uint4 packed[2];
        uint8_t *out = reinterpret_cast<uint8_t *>(packed);
#pragma unroll
        for (uint32_t i = 0; i < SPARK_M4_QUANT_BLOCK; ++i) {
            float scaled = __fdiv_rn(values[i], scale);
            if (scaled > 448.0f) {
                scaled = 448.0f;
            }
            if (scaled < -448.0f) {
                scaled = -448.0f;
            }
            out[i] = spark_fp8_e4m3_encode(scaled);
        }
        uint4 *dst = reinterpret_cast<uint4 *>(out_codes + block * SPARK_M4_QUANT_BLOCK);
        dst[0] = packed[0];
        dst[1] = packed[1];
    }
}

// opt_bulk_engram_post: the Engram stage after the wkv over a chunk's rows at once: the narrow,
//   opt_engram_gate_write's gates, the widen and engram_write_bf16 of every row in one launch. Row
//   r's stream ([hc][dim] f32) becomes bf16(stream + gate[c] * value), its kv row at kv_all + r *
//   kv_stride (key [hc][dim] then value [dim], bf16); a row with live[r] == 0 is left alone. The
//   sums are block trees over the 256 threads (not the single-row launch's 8-warp order), so a gate
//   can differ from it in the last bits. Grid (rows), 256 threads; hc <= 4, dim even.
extern "C" __global__ void __launch_bounds__(256) opt_bulk_engram_post(
    float *__restrict__ stream_all, const __nv_bfloat16 *__restrict__ kv_all,
    const float *__restrict__ product, const uint8_t *__restrict__ live, const uint32_t dim,
    const uint32_t hc, const float eps, const uint32_t kv_stride) {
    OPT_PDL_ENTRY();
    const uint32_t row = blockIdx.x;
    if (live[row] == 0u || hc > 4u) return;
    __shared__ float s_part[4][3][8];
    __shared__ float s_gate[4];
    const uint32_t t = threadIdx.x;
    const uint32_t pairs = dim >> 1u;
    float *stream = stream_all + (uint64_t)row * hc * dim;
    const __nv_bfloat16 *kv = kv_all + (uint64_t)row * kv_stride;
    for (uint32_t copy = 0u; copy < hc; ++copy) {
        const float2 *s2 = reinterpret_cast<const float2 *>(stream + (uint64_t)copy * dim);
        const uint32_t *k2 = reinterpret_cast<const uint32_t *>(kv + (uint64_t)copy * dim);
        const float2 *p2 = reinterpret_cast<const float2 *>(product + (uint64_t)copy * dim);
        float hs = 0.0f, ks = 0.0f, dot = 0.0f;
        for (uint32_t p = t; p < pairs; p += 256u) {
            const float2 sv = s2[p];
            const uint32_t kw = __ldg(k2 + p);
            const float2 pr = __ldg(p2 + p);
            const float h0 = __bfloat162float(__float2bfloat16_rn(sv.x));
            const float h1 = __bfloat162float(__float2bfloat16_rn(sv.y));
            const float k0 = opt_m_bf16lo(kw), k1 = opt_m_bf16hi(kw);
            hs = __fadd_rn(__fadd_rn(hs, __fmul_rn(h0, h0)), __fmul_rn(h1, h1));
            ks = __fadd_rn(__fadd_rn(ks, __fmul_rn(k0, k0)), __fmul_rn(k1, k1));
            dot = __fadd_rn(__fadd_rn(dot, __fmul_rn(__fmul_rn(h0, pr.x), k0)),
                            __fmul_rn(__fmul_rn(h1, pr.y), k1));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            hs = __fadd_rn(hs, __shfl_xor_sync(0xffffffffu, hs, o));
            ks = __fadd_rn(ks, __shfl_xor_sync(0xffffffffu, ks, o));
            dot = __fadd_rn(dot, __shfl_xor_sync(0xffffffffu, dot, o));
        }
        if ((t & 31u) == 0u) {
            s_part[copy][0][t >> 5] = hs;
            s_part[copy][1][t >> 5] = ks;
            s_part[copy][2][t >> 5] = dot;
        }
    }
    __syncthreads();
    if (t < hc) {
        float a = s_part[t][0][0], b = s_part[t][1][0], d = s_part[t][2][0];
#pragma unroll
        for (int w = 1; w < 8; ++w) {
            a = __fadd_rn(a, s_part[t][0][w]);
            b = __fadd_rn(b, s_part[t][1][w]);
            d = __fadd_rn(d, s_part[t][2][w]);
        }
        const float dim_f = (float)dim;
        const float stream_inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(a, dim_f), eps)));
        const float key_inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(b, dim_f), eps)));
        const float rstd = __fmul_rn(stream_inv, key_inv);
        const float scale = __fdiv_rn(1.0f, __fsqrt_rn(dim_f));
        const float scaled_dot = __fmul_rn(__fmul_rn(d, rstd), scale);
        const float magnitude = fabsf(scaled_dot);
        const float clamped = magnitude > 1.0e-6f ? magnitude : 1.0e-6f;
        const float root = __fsqrt_rn(clamped);
        const float signed_root = scaled_dot < 0.0f ? -root : root;
        s_gate[t] = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-signed_root)));
    }
    __syncthreads();
    const uint32_t *v2 = reinterpret_cast<const uint32_t *>(kv + (uint64_t)hc * dim);
    for (uint32_t copy = 0u; copy < hc; ++copy) {
        const float gate = s_gate[copy];
        float2 *s2 = reinterpret_cast<float2 *>(stream + (uint64_t)copy * dim);
        for (uint32_t p = t; p < pairs; p += 256u) {
            const float2 sv = s2[p];
            const uint32_t vw = __ldg(v2 + p);
            float2 o;
            o.x = __bfloat162float(__float2bfloat16_rn(__fadd_rn(sv.x, __fmul_rn(gate, opt_m_bf16lo(vw)))));
            o.y = __bfloat162float(__float2bfloat16_rn(__fadd_rn(sv.y, __fmul_rn(gate, opt_m_bf16hi(vw)))));
            s2[p] = o;
        }
    }
}

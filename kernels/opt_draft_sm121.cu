// kernels/opt_draft_sm121.cu
//
// The on-device DSpark drafter's own kernels (crates/sparkflash-cli/src/tp4_forward/draft.rs). The
// drafter does not have to match a reference bit for bit: only its acceptance length matters, so
// these follow the reference's operations (forward/mtp.rs, attention.rs dspark_attention_forward)
// without the per-expression rounding contract of the verify kernels.
//
//   opt_draft_taps        the hc-mean of a row's stream into one tap slot (grid 20 x rows).
//   opt_draft_norm_quant  RMSNorm of f32 rows by a bf16 weight, then the MX FP8 activation codes
//                         (grid rows).
//   opt_draft_attn        the block attention: every row of the block attends the stage ring's last
//                         min(d, 128) positions and all of the block's own rows (grid heads x rows).
//   opt_draft_logits      the head logits of up to 8 rows over the rank's vocabulary slice.
//   opt_draft_head_fp8    the head slice as E4M3 codes + ue8m0 per 32 (once, at the first draft).
//   opt_draft_logits_fp8  opt_draft_logits over that copy: half the bytes of the bf16 head.
//   opt_draft_markov      one sequential draft step: base logits of row i plus the Markov bias of
//                         ids[i], chunk maxima for opt_rows_argmax_fold and each chunk's softmax mass.
//   opt_draft_fold_prob   the chunk maxima and masses of one step folded into this rank's slot.
//   opt_draft_pick_prob   the cross-rank pick (opt_rows_argmax_pick's rule) and the pick's softmax
//                         probability, which the host uses to choose the verify depth.

#include "opt_pdl_sm121.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "m4_common_sm121.cuh"
#include "opt_sample_sm121.cuh"
#include "m4_activation_quant_sm121.cuh"

#define OD_HIDDEN 5120u
#define OD_HC 4u
#define OD_TAPS 3u

__device__ __forceinline__ float od_bf16lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float od_bf16hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

__device__ __forceinline__ uint4 od_ldw(const void *p) {
    uint4 v;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(p));
    return v;
}

__device__ __forceinline__ float od_dot8(uint4 w, uint4 h, float acc) {
    const uint32_t a[4] = {w.x, w.y, w.z, w.w};
    const uint32_t b[4] = {h.x, h.y, h.z, h.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        acc = fmaf(od_bf16lo(a[k]), od_bf16lo(b[k]), acc);
        acc = fmaf(od_bf16hi(a[k]), od_bf16hi(b[k]), acc);
    }
    return acc;
}

__device__ __forceinline__ float od_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Row y: taps_all + y * 3 * 5120 + tap * 5120 = mean over the hc copies of stream_all + y * 4 * 5120.
extern "C" __global__ void __launch_bounds__(256) opt_draft_taps(
    const float *__restrict__ stream_all,  // [rows][4][5120]
    float *__restrict__ taps_all,          // [rows][3][5120]
    const uint32_t tap) {
    OPT_PDL_ENTRY();
    const uint32_t column = blockIdx.x * 256u + threadIdx.x;
    if (column >= OD_HIDDEN) return;
    const uint32_t y = blockIdx.y;
    const float *stream = stream_all + (uint64_t)y * OD_HC * OD_HIDDEN;
    float sum = 0.0f;
#pragma unroll
    for (uint32_t c = 0; c < OD_HC; ++c) sum += stream[c * OD_HIDDEN + column];
    taps_all[(uint64_t)y * OD_TAPS * OD_HIDDEN + tap * OD_HIDDEN + column] = sum / (float)OD_HC;
}

// Row y: x_all + y * 5120 normalised by weight, rounded to bf16, then quantised in 32-value blocks
// (UE8M0 scale, E4M3 codes) into codes_all + y * 5120 and scales_all + y * 160.
extern "C" __global__ void __launch_bounds__(256) opt_draft_norm_quant(
    const float *__restrict__ x_all,
    const __nv_bfloat16 *__restrict__ weight,
    const float eps,
    uint8_t *__restrict__ codes_all,
    uint8_t *__restrict__ scales_all) {
    OPT_PDL_ENTRY();
    __shared__ float s_part[8];
    const uint32_t y = blockIdx.x;
    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31u;
    const uint32_t warp = t >> 5;
    const float *x = x_all + (uint64_t)y * OD_HIDDEN;
    float sum = 0.0f;
    for (uint32_t c = t; c < OD_HIDDEN; c += 256u) sum = fmaf(x[c], x[c], sum);
    sum = od_warp_sum(sum);
    if (lane == 0u) s_part[warp] = sum;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) total += s_part[w];
    const float inv = rsqrtf(total / (float)OD_HIDDEN + eps);
    uint8_t *codes = codes_all + (uint64_t)y * OD_HIDDEN;
    uint8_t *scales = scales_all + (uint64_t)y * (OD_HIDDEN / 32u);
    for (uint32_t qb = warp; qb < OD_HIDDEN / 32u; qb += 8u) {
        const uint32_t c = qb * 32u + lane;
        const float v = __bfloat162float(__float2bfloat16_rn(x[c] * inv * __bfloat162float(weight[c])));
        float amax = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const uint8_t scale = spark_act_quant_scale_code(amax);
        const float inv_scale = 1.0f / spark_ue8m0_decode(scale);
        codes[c] = spark_fp8_e4m3_encode(v * inv_scale);
        if (lane == 0u) scales[qb] = scale;
    }
}

// ---------------------------------------------------------------------------------------------
// Block attention. The stage ring is linear (slot = position): the main rows' KV at their own
// positions, and the block's rows written at d .. d + rows - 1 just before this launch, so the keys
// of every row are the one contiguous range [max(d, 128) - 128, d + rows). The block is not causal
// inside itself (model.py:1021-1029).
// ---------------------------------------------------------------------------------------------
#define OD_A_LATENT 512u
#define OD_A_THREADS 256u
#define OD_A_WARPS 8u
#define OD_A_WINDOW 128u
#define OD_A_MAX_KEYS 160u
#define OD_A_WSCALES 16u

__device__ __forceinline__ void od_e4m3x4(uint32_t w, float *f) {
    uint32_t h01, h23;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h01) : "h"((uint16_t)(w & 0xffffu)));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h23) : "h"((uint16_t)(w >> 16)));
    const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&h01));
    const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&h23));
    f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}

__device__ __forceinline__ void od_row16(const uint8_t *__restrict__ codes_base,
                                         const uint8_t *__restrict__ scales_base, uint32_t slot,
                                         uint32_t lane, float k[16]) {
    const uint4 codes = __ldg(reinterpret_cast<const uint4 *>(codes_base + (uint64_t)slot * OD_A_LATENT) + lane);
    const float scale = spark_ue8m0_decode(scales_base[(uint64_t)slot * OD_A_WSCALES + (lane >> 1)]);
    od_e4m3x4(codes.x, k);
    od_e4m3x4(codes.y, k + 4);
    od_e4m3x4(codes.z, k + 8);
    od_e4m3x4(codes.w, k + 12);
#pragma unroll
    for (int i = 0; i < 16; ++i) k[i] = __bfloat162float(__float2bfloat16_rn(k[i] * scale));
}

extern "C" __global__ void __launch_bounds__(OD_A_THREADS) opt_draft_attn(
    const __nv_bfloat16 *__restrict__ q_all,  // [rows][heads][512], rotated
    const uint8_t *__restrict__ ring_codes,   // the stage ring, slot = position
    const uint8_t *__restrict__ ring_scales,
    const float *__restrict__ sink,           // [heads] of this rank
    __nv_bfloat16 *__restrict__ out_all,      // [rows][heads][512]
    const uint32_t d) {
    OPT_PDL_ENTRY();
    const uint32_t head = blockIdx.x;
    const uint32_t heads = gridDim.x;
    const uint32_t y = blockIdx.y;
    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31u;
    const uint32_t warp = t >> 5;
    const uint32_t start = d > OD_A_WINDOW ? d - OD_A_WINDOW : 0u;
    uint32_t keys = d + gridDim.y - start;
    keys = keys < OD_A_MAX_KEYS ? keys : OD_A_MAX_KEYS;
    const uint8_t *codes = ring_codes + (uint64_t)start * OD_A_LATENT;
    const uint8_t *scales = ring_scales + (uint64_t)start * OD_A_WSCALES;
    const __nv_bfloat16 *q = q_all + ((uint64_t)y * heads + head) * OD_A_LATENT;
    __nv_bfloat16 *out = out_all + ((uint64_t)y * heads + head) * OD_A_LATENT;

    __shared__ float s_scores[OD_A_MAX_KEYS];
    __shared__ __align__(16) float s_q[OD_A_LATENT];
    __shared__ __align__(16) float s_red[OD_A_WARPS][OD_A_LATENT];
    __shared__ float s_wmax[OD_A_WARPS];
    __shared__ float s_wsum[OD_A_WARPS];
    s_q[t] = __bfloat162float(q[t]);
    s_q[t + OD_A_THREADS] = __bfloat162float(q[t + OD_A_THREADS]);
    __syncthreads();
    float qv[16];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 v = reinterpret_cast<const float4 *>(s_q)[4u * lane + j];
        qv[4 * j] = v.x; qv[4 * j + 1] = v.y; qv[4 * j + 2] = v.z; qv[4 * j + 3] = v.w;
    }
    float wmax = -3.402823466e38f;
    for (uint32_t slot = warp; slot < keys; slot += OD_A_WARPS) {
        float k[16];
        od_row16(codes, scales, slot, lane, k);
        float partial = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) partial = fmaf(qv[i], k[i], partial);
        const float score = od_warp_sum(partial) * SPARK_M4_MLA_SCORE_SCALE;
        if (lane == 0u) s_scores[slot] = score;
        wmax = score > wmax ? score : wmax;
    }
    if (lane == 0u) s_wmax[warp] = wmax;
    __syncthreads();
    float smax = s_wmax[0];
#pragma unroll
    for (uint32_t w = 1; w < OD_A_WARPS; ++w) smax = s_wmax[w] > smax ? s_wmax[w] : smax;
    float lsum = 0.0f;
    for (uint32_t slot = t; slot < keys; slot += OD_A_THREADS) {
        const float probability = spark_exp(s_scores[slot] - smax);
        s_scores[slot] = __bfloat162float(__float2bfloat16_rn(probability));
        lsum += probability;
    }
    lsum = od_warp_sum(lsum);
    if (lane == 0u) s_wsum[warp] = lsum;
    __syncthreads();
    float ssum = spark_exp(sink[head] - smax);
#pragma unroll
    for (uint32_t w = 0; w < OD_A_WARPS; ++w) ssum += s_wsum[w];
    ssum = ssum > 0.0f ? ssum : 1.0f;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    for (uint32_t slot = warp; slot < keys; slot += OD_A_WARPS) {
        const float prob = s_scores[slot];
        float v[16];
        od_row16(codes, scales, slot, lane, v);
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] = fmaf(prob, v[i], acc[i]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        reinterpret_cast<float4 *>(s_red[warp])[4u * lane + j] =
            make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]);
    }
    __syncthreads();
#pragma unroll
    for (uint32_t r = 0; r < 2; ++r) {
        const uint32_t i = t + r * OD_A_THREADS;
        float total = s_red[0][i];
#pragma unroll
        for (uint32_t w = 1; w < OD_A_WARPS; ++w) total += s_red[w][i];
        out[i] = __float2bfloat16_rn(total / ssum);
    }
}

// ---------------------------------------------------------------------------------------------
// Head logits and the Markov step.
// ---------------------------------------------------------------------------------------------
#define OD_LM_THREADS 512u
#define OD_LM_WARPS (OD_LM_THREADS / 32u)
#define OD_LM_CHUNK 1024u
#define OD_LM_MAXR 8u
#define OD_MARKOV_RANK 256u

// Warp w of block x scores vocabulary entry x * 16 + w of the slice, then every gridDim.x * 16
// entries on, for every row: logits_all + y * vocab + entry. An entry's dot product is one warp's
// whatever the grid, so the grid only sets how many SMs stream the head (one block per SM).
extern "C" __global__ void __launch_bounds__(OD_LM_THREADS) opt_draft_logits(
    const __nv_bfloat16 *__restrict__ hidden_all,  // [rows][5120]
    const __nv_bfloat16 *__restrict__ w,           // [vocab][5120], the rank's slice
    float *__restrict__ logits_all,                // [rows][vocab]
    const uint32_t vocab,
    const uint32_t rows) {
    OPT_PDL_ENTRY();
    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31u;
    const uint32_t warp = t >> 5;
    const uint32_t per = OD_HIDDEN / 8u;
    const uint4 *hidden4 = reinterpret_cast<const uint4 *>(hidden_all);
    for (uint32_t entry = blockIdx.x * OD_LM_WARPS + warp; entry < vocab; entry += gridDim.x * OD_LM_WARPS) {
        const uint4 *row = reinterpret_cast<const uint4 *>(w + (uint64_t)entry * OD_HIDDEN);
        uint4 wv[20];
#pragma unroll
        for (uint32_t i = 0; i < 20u; ++i) wv[i] = od_ldw(row + lane + 32u * i);
#pragma unroll
        for (uint32_t y = 0; y < OD_LM_MAXR; ++y) {
            if (y < rows) {
                const uint4 *h = hidden4 + y * per;
                float acc = 0.0f;
#pragma unroll
                for (uint32_t i = 0; i < 20u; ++i) acc = od_dot8(wv[i], __ldg(h + lane + 32u * i), acc);
                acc = od_warp_sum(acc);
                if (lane == 0u) logits_all[(uint64_t)y * vocab + entry] = acc;
            }
        }
    }
}

// Row v of w (bf16 [vocab][5120]) as E4M3 codes [vocab][5120] and ue8m0 scales [vocab][160], the
// activation quantiser's rule per 32-value block. Grid vocab, 256 threads (warp per block of 32).
extern "C" __global__ void __launch_bounds__(256) opt_draft_head_fp8(
    const __nv_bfloat16 *__restrict__ w, uint8_t *__restrict__ codes, uint8_t *__restrict__ scales) {
    // It reads nothing the previous kernel writes, but it must still wait: logits_fp8 waits only on
    // this grid, so skipping the wait here would let it run ahead of the rmsnorm that fills lm_normed.
    OPT_PDL_ENTRY();
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint64_t row = blockIdx.x;
    for (uint32_t b = warp; b < OD_HIDDEN / 32u; b += 8u) {
        const uint64_t i = row * OD_HIDDEN + b * 32u + lane;
        const float v = __bfloat162float(w[i]);
        float amax = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const uint8_t scale = spark_act_quant_scale_code(amax);
        codes[i] = spark_fp8_e4m3_encode(v * (1.0f / spark_ue8m0_decode(scale)));
        if (lane == 0u) scales[row * (OD_HIDDEN / 32u) + b] = scale;
    }
}

// 16 E4M3 codes dotted with 16 bf16 values (two uint4), sequential FMAs from zero.
__device__ __forceinline__ float od_dot16_fp8(uint4 w, uint4 h0, uint4 h1) {
    const uint32_t c[4] = {w.x, w.y, w.z, w.w};
    const uint32_t h[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const __nv_fp8x2_storage_t pair = (__nv_fp8x2_storage_t)((c[k >> 1] >> (16u * (k & 1))) & 0xffffu);
        const float2 f = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3)));
        acc = fmaf(f.x, od_bf16lo(h[k]), acc);
        acc = fmaf(f.y, od_bf16hi(h[k]), acc);
    }
    return acc;
}

// opt_draft_logits over the E4M3 copy (the same entry walk): lane l reads 16-byte chunks l + 32 i
// (K-block chunk / 2).
extern "C" __global__ void __launch_bounds__(OD_LM_THREADS) opt_draft_logits_fp8(
    const __nv_bfloat16 *__restrict__ hidden_all,  // [rows][5120]
    const uint8_t *__restrict__ codes,             // [vocab][5120]
    const uint8_t *__restrict__ scales,            // [vocab][160]
    float *__restrict__ logits_all,                // [rows][vocab]
    const uint32_t vocab,
    const uint32_t rows) {
    OPT_PDL_ENTRY();
    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31u;
    const uint32_t warp = t >> 5;
    const uint32_t per = OD_HIDDEN / 8u;
    const uint4 *hidden4 = reinterpret_cast<const uint4 *>(hidden_all);
    for (uint32_t entry = blockIdx.x * OD_LM_WARPS + warp; entry < vocab; entry += gridDim.x * OD_LM_WARPS) {
        const uint4 *row = reinterpret_cast<const uint4 *>(codes + (uint64_t)entry * OD_HIDDEN);
        const uint8_t *srow = scales + (uint64_t)entry * (OD_HIDDEN / 32u);
        uint4 wv[10];
        float sv[10];
#pragma unroll
        for (uint32_t i = 0; i < 10u; ++i) wv[i] = od_ldw(row + lane + 32u * i);
#pragma unroll
        for (uint32_t i = 0; i < 10u; ++i) sv[i] = spark_ue8m0_decode(__ldg(srow + ((lane + 32u * i) >> 1)));
#pragma unroll
        for (uint32_t y = 0; y < OD_LM_MAXR; ++y) {
            if (y < rows) {
                const uint4 *h = hidden4 + y * per;
                float acc = 0.0f;
#pragma unroll
                for (uint32_t i = 0; i < 10u; ++i) {
                    const uint32_t chunk = lane + 32u * i;
                    acc = fmaf(od_dot16_fp8(wv[i], __ldg(h + 2u * chunk), __ldg(h + 2u * chunk + 1u)), sv[i], acc);
                }
                acc = od_warp_sum(acc);
                if (lane == 0u) logits_all[(uint64_t)y * vocab + entry] = acc;
            }
        }
    }
}

// Draft step i: logit(v) = logits_all[i][v] + markov_head[v] . markov_embed[ids[i]] over the rank's
// slice (markov_head already offset to the slice's first row), block maxima over 1024 entries, ties
// to the lower index (the reference's argmax). With inv_t != 0 the value is logit * inv_t plus the
// serving sampler's noise at the position the step drafts (pos0 + step), opt_rows_lm_head's rule,
// so the draft is a sample under the same noise the verify head will use.
extern "C" __global__ void __launch_bounds__(OD_LM_THREADS) opt_draft_markov(
    const float *__restrict__ logits_all,          // [rows][vocab]
    const uint32_t *__restrict__ ids,              // [block + 1] global ids
    const uint32_t step,
    const __nv_bfloat16 *__restrict__ markov_head, // [vocab][256], the rank's slice
    const __nv_bfloat16 *__restrict__ markov_embed,// [full vocab][256]
    float *__restrict__ block_max,                 // [chunks]
    uint32_t *__restrict__ block_arg,              // [chunks], slice-local index
    const uint32_t vocab,
    float *__restrict__ block_sum,                 // [chunks], sum of exp(logit - block_max)
    const float inv_t,
    const uint64_t seed,
    const uint32_t pos0,
    const uint32_t id_base) {
    OPT_PDL_ENTRY();
    __shared__ float s_best[OD_LM_WARPS];
    __shared__ float s_sum[OD_LM_WARPS];
    __shared__ uint32_t s_best_i[OD_LM_WARPS];
    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31u;
    const uint32_t warp = t >> 5;
    const uint32_t token = ids[step];
    const uint4 e = __ldg(reinterpret_cast<const uint4 *>(markov_embed + (uint64_t)token * OD_MARKOV_RANK) + lane);
    const float *logits = logits_all + (uint64_t)step * vocab;
    const uint32_t base = blockIdx.x * OD_LM_CHUNK;
    float best = -3.402823466e38f;
    float sum = 0.0f;  // sum of exp(value - best) over the warp's entries so far
    uint32_t best_i = 0xffffffffu;
    for (uint32_t r = warp; r < OD_LM_CHUNK; r += OD_LM_WARPS) {
        const uint32_t entry = base + r;
        if (entry >= vocab) break;
        const uint4 h = od_ldw(reinterpret_cast<const uint4 *>(markov_head + (uint64_t)entry * OD_MARKOV_RANK) + lane);
        const float bias = od_warp_sum(od_dot8(h, e, 0.0f));
        const float logit = logits[entry] + bias;
        const float value =
            inv_t == 0.0f ? logit : fmaf(logit, inv_t, opt_sample_gumbel(seed, pos0 + step, id_base + entry));
        if (best_i == 0xffffffffu) {
            best = value; best_i = entry; sum = 1.0f;
        } else if (value > best) {
            sum = sum * __expf(best - value) + 1.0f; best = value; best_i = entry;
        } else {
            sum += __expf(value - best);
        }
    }
    if (lane == 0u) { s_best[warp] = best; s_sum[warp] = sum; s_best_i[warp] = best_i; }
    __syncthreads();
    if (t == 0u) {
        float v = s_best[0];
        uint32_t idx = s_best_i[0];
        for (uint32_t k = 1; k < OD_LM_WARPS; ++k) {
            const uint32_t oi = s_best_i[k];
            if (oi == 0xffffffffu) continue;
            if (idx == 0xffffffffu || s_best[k] > v || (s_best[k] == v && oi < idx)) { v = s_best[k]; idx = oi; }
        }
        float mass = 0.0f;
        for (uint32_t k = 0; k < OD_LM_WARPS; ++k) {
            if (s_best_i[k] != 0xffffffffu) mass += s_sum[k] * __expf(s_best[k] - v);
        }
        block_max[blockIdx.x] = v;
        block_arg[blockIdx.x] = idx == 0xffffffffu ? 0u : idx;
        block_sum[blockIdx.x] = mass;
    }
}

// One step's chunk maxima and masses into this rank's slot of the gather: [value, index, mass],
// the value and index by opt_rows_argmax_fold's rule (strictly greater replaces), the mass relative
// to the value. One thread.
extern "C" __global__ void opt_draft_fold_prob(
    const float *__restrict__ block_max,
    const uint32_t *__restrict__ block_arg,
    const float *__restrict__ block_sum,
    const uint32_t chunks,
    uint32_t *__restrict__ slot) {
    OPT_PDL_ENTRY();
    if (threadIdx.x != 0u) return;
    float best = block_max[0];
    uint32_t index = block_arg[0];
    for (uint32_t c = 1; c < chunks; ++c) {
        if (block_max[c] > best) { best = block_max[c]; index = block_arg[c]; }
    }
    float mass = 0.0f;
    for (uint32_t c = 0; c < chunks; ++c) mass += block_sum[c] * __expf(block_max[c] - best);
    slot[0] = __float_as_uint(best);
    slot[1] = index;
    slot[2] = __float_as_uint(mass);
}

// The cross-rank pick over the exchanged slots ([world][value, index, mass]): opt_rows_argmax_pick's
// winner (the first rank holding the greatest value), ids_out[0] its global id, probs_out[0] its
// softmax probability over the whole vocabulary. One thread.
extern "C" __global__ void opt_draft_pick_prob(
    const uint32_t *__restrict__ gather,
    const uint32_t world,
    const uint32_t vocab,
    uint32_t *__restrict__ ids_out,
    float *__restrict__ probs_out) {
    OPT_PDL_ENTRY();
    if (threadIdx.x != 0u) return;
    float best = __uint_as_float(gather[0]);
    uint32_t index = gather[1];
    for (uint32_t k = 1; k < world; ++k) {
        const float v = __uint_as_float(gather[3u * k]);
        if (v > best) { best = v; index = gather[3u * k + 1u] + k * vocab; }
    }
    float mass = 0.0f;
    for (uint32_t k = 0; k < world; ++k) {
        mass += __uint_as_float(gather[3u * k + 2u]) * __expf(__uint_as_float(gather[3u * k]) - best);
    }
    ids_out[0] = index;
    probs_out[0] = 1.0f / mass;
}

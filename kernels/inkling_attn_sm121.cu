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

// kernels/inkling_attn_sm121.cu
//
// Inkling-2-Small's grouped-query attention: no rotary embedding, a learned low-rank
// relative-position bias, an attention sink, a sigmoid output gate, per-head q/k and
// relative-state L2 clipping, an unweighted RMSNorm on the value rows, optional log
// scaling (tau), and a 1,024-token sliding window. The GQA group count is a runtime
// argument; the released release is 48 query heads over 8 KV heads (group 6).
//
// One warp owns one (query head, row). The decode entry is the same body with one row, so
// the two entry points cannot disagree about a score. The body keeps the online softmax in
// registers (running maximum, running denominator, four value accumulators a lane) which is
// what lets a full-attention layer walk a context of up to a million positions without a
// logits buffer.
//
// The relative bias is the released checkpoint's low-rank form: the per-token projection
// rel_logits_proj produces d_rel numbers a head (rel_proj, [heads][d_rel]), and the learned
// bank (rel_bank, [heads][rel_extent][d_rel], BF16) holds one d_rel profile per backward
// distance. The bias of a key at distance d is dot(rel_proj[head], rel_bank[head][d]), zero
// for d >= rel_extent. This is the "r @ proj over 1,024 distances" rule of the model note.
//
// ABI (all pointers are device pointers; every nullable pointer means "not present"):
//
//   q             [rows][heads][head_dim]        f32, post-projection, pre-clip
//   gate          [rows][heads][head_dim]        f32, pre-sigmoid output gate
//   keys          [positions][kv_heads][head_dim] BF16 cache
//   values        [positions][kv_heads][head_dim] BF16 cache
//   rel_proj      [rows][heads][d_rel]          f32, per-token relative profile
//   rel_bank      [heads][rel_extent][d_rel]    BF16 learned relative profiles
//   sink          [heads]                       f32, nullable
//   q_scale       [heads]                       f32 learned query scale, nullable
//   k_scale       [kv_heads]                    f32 learned key scale, nullable
//   out           [rows][heads][head_dim]        BF16
//
//   position / first_position   absolute position of the token (decode) or of row 0
//   window                       0 for a full layer, otherwise the sliding window
//   cache_rows                   0 for a linear (absolute-position) cache, otherwise the rows the
//     layer's cache holds and the ring length: an absolute slot addresses cache row
//     slot % cache_rows. A sliding layer's cache is exactly its window, so 40 of the 48 layers
//     hold 1,024 rows instead of one row a context position - the difference between 41 GiB and
//     8 GiB at a million tokens. Over the slots a window can address, the modulo is injective, so
//     a ring and a linear cache read the same rows.
//   qk_clip / rel_clip           per-head L2 clip thresholds, <=0 or not finite disables
//   tau                          log-scaling factor applied to q and rel_proj, <=0 means 1
//   value_norm                   non-zero to RMS-normalise each value row before mixing
//
// The arithmetic order is the host reference's: the q and k norms are warp reductions over
// the head, the clip is applied to the f32 vector, the learned scales follow the clip, tau
// multiplies query and relative state, the score is the clipped dot over sqrt(head_dim) plus
// the relative bias, the sink enters the denominator only, and the gate is a sigmoid on the
// output. Values are BF16-rounded on load because that is what the cache stores.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define INKLING_MAX_HEAD_DIM 128
#define INKLING_MAX_D_REL 32

__device__ __forceinline__ float inkling_attn_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float inkling_attn_sigmoid(float value) {
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-value)));
}

// The shared body: one warp, one (head, row). q_row, gate_row and out_row already point at
// the row's head block; the cache is indexed by absolute position.
__device__ __forceinline__ void inkling_attn_one(
    const float* __restrict__ q_row,
    const float* __restrict__ gate_row,
    const __nv_bfloat16* __restrict__ keys,
    const __nv_bfloat16* __restrict__ values,
    const float* __restrict__ rel_proj,
    const __nv_bfloat16* __restrict__ rel_bank,
    const float* __restrict__ sink,
    const float* __restrict__ q_scale,
    const float* __restrict__ k_scale,
    const float* __restrict__ rel_scale,
    __nv_bfloat16* __restrict__ out_row,
    int head,
    int kv_heads,
    int head_dim,
    int groups,
    int position,
    int window,
    int cache_rows,
    int rel_extent,
    int d_rel,
    float qk_clip,
    float rel_clip,
    float tau,
    int value_norm,
    float eps) {
    const unsigned int lane = threadIdx.x & 31u;
    if (head_dim <= 0 || head_dim > INKLING_MAX_HEAD_DIM || d_rel < 0 || d_rel > INKLING_MAX_D_REL ||
        groups <= 0) {
        for (int dim = (int)lane; dim < head_dim; dim += 32) {
            out_row[dim] = __float2bfloat16_rn(0.0f);
        }
        return;
    }
    const int kv_head = head / groups;
    const float inv_dim = 1.0f / sqrtf((float)head_dim);
    const float use_tau = tau > 0.0f ? tau : 1.0f;

    // Query: learned scale, then clip, then tau. The released q_l2_clip.scale is
    // [heads, head_dim], a gain per element, and the clip is on the *scaled* vector - clipping
    // the raw vector first would clip a different vector, because the gains are not uniform.
    float qv[INKLING_MAX_HEAD_DIM / 32];
    float q_norm = 0.0f;
#pragma unroll
    for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
        const int dim = (int)lane + 32 * step;
        const float gain = dim < head_dim && q_scale != nullptr
                               ? q_scale[(uint64_t)head * head_dim + dim]
                               : 1.0f;
        qv[step] = dim < head_dim ? __fmul_rn(q_row[dim], gain) : 0.0f;
        q_norm = __fadd_rn(q_norm, __fmul_rn(qv[step], qv[step]));
    }
    q_norm = sqrtf(inkling_attn_warp_sum(q_norm));
    float q_gain = 1.0f;
    if (qk_clip > 0.0f && q_norm > qk_clip) {
        q_gain = qk_clip / q_norm;
    }
    q_gain *= use_tau;
#pragma unroll
    for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
        qv[step] = __fmul_rn(qv[step], q_gain);
    }

    // Relative state: one d_rel value a lane for the first d_rel lanes. Clip then tau.
    float rv = lane < (unsigned int)d_rel ? rel_proj[(uint64_t)head * d_rel + lane] : 0.0f;
    if (rel_scale != nullptr && lane < (unsigned int)d_rel) {
        rv = __fmul_rn(rv, rel_scale[(uint64_t)head * d_rel + (int)lane]);
    }
    const float r_norm = sqrtf(inkling_attn_warp_sum(__fmul_rn(rv, rv)));
    float r_gain = 1.0f;
    if (rel_clip > 0.0f && r_norm > rel_clip) {
        r_gain = rel_clip / r_norm;
    }
    rv = __fmul_rn(rv, __fmul_rn(r_gain, use_tau));

    const uint64_t kv_stride = (uint64_t)kv_heads * head_dim;
    const uint64_t kv_offset = (uint64_t)kv_head * head_dim;
    const int64_t first = window > 0
                              ? (position + 1 > window ? position + 1 - window : 0)
                              : 0;

    // The sink is a value-less term of the denominator, so it starts the running softmax.
    float maximum = -INFINITY;
    float denominator = 0.0f;
    if (sink != nullptr) {
        maximum = sink[head];
        denominator = 1.0f;
    }
    float acc[INKLING_MAX_HEAD_DIM / 32];
#pragma unroll
    for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
        acc[step] = 0.0f;
    }

    for (int64_t slot = first; slot <= (int64_t)position; ++slot) {
        const int64_t row = cache_rows > 0 ? slot % (int64_t)cache_rows : slot;
        const __nv_bfloat16* krow = keys + (uint64_t)row * kv_stride + kv_offset;
        const __nv_bfloat16* vrow = values + (uint64_t)row * kv_stride + kv_offset;
        float kv[INKLING_MAX_HEAD_DIM / 32];
        float vv[INKLING_MAX_HEAD_DIM / 32];
        float k_norm = 0.0f;
        float v_norm = 0.0f;
#pragma unroll
        for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
            const int dim = (int)lane + 32 * step;
            // The key carries its own learned gain, and its clip is on the scaled vector too.
            const float gain = dim < head_dim && k_scale != nullptr
                                   ? k_scale[(uint64_t)kv_head * head_dim + dim]
                                   : 1.0f;
            kv[step] = dim < head_dim ? __fmul_rn(__bfloat162float(krow[dim]), gain) : 0.0f;
            vv[step] = dim < head_dim ? __bfloat162float(vrow[dim]) : 0.0f;
            k_norm = __fadd_rn(k_norm, __fmul_rn(kv[step], kv[step]));
            v_norm = __fadd_rn(v_norm, __fmul_rn(vv[step], vv[step]));
        }
        k_norm = sqrtf(inkling_attn_warp_sum(k_norm));
        float k_gain = 1.0f;
        if (qk_clip > 0.0f && k_norm > qk_clip) {
            k_gain = qk_clip / k_norm;
        }
        float dot = 0.0f;
#pragma unroll
        for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
            dot = __fadd_rn(dot, __fmul_rn(qv[step], __fmul_rn(kv[step], k_gain)));
        }
        dot = inkling_attn_warp_sum(dot);
        float score = __fmul_rn(dot, inv_dim);

        if (d_rel > 0 && rel_extent > 0) {
            const int64_t distance = (int64_t)position - slot;
            if (distance >= 0 && distance < rel_extent) {
                float bias = 0.0f;
                if (lane < (unsigned int)d_rel) {
                    // The released bank is [d_rel, rel_extent]: a row per relative dimension,
                    // one column per distance. Indexing it as [head][distance][d_rel] would read
                    // a different weight and still stay in bounds, so nothing would catch it.
                    const uint64_t index = (uint64_t)lane * rel_extent + (uint64_t)distance;
                    bias = __fmul_rn(rv, __bfloat162float(rel_bank[index]));
                }
                score = __fadd_rn(score, inkling_attn_warp_sum(bias));
            }
        }

        if (score > maximum) {
            const float rescale = expf(maximum - score);
            denominator = __fadd_rn(__fmul_rn(denominator, rescale), 1.0f);
#pragma unroll
            for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
                acc[step] = __fmul_rn(acc[step], rescale);
            }
            maximum = score;
        } else {
            denominator = __fadd_rn(denominator, expf(score - maximum));
        }
        const float weight = expf(score - maximum);

        if (value_norm != 0) {
            const float mean = __fdiv_rn(inkling_attn_warp_sum(v_norm), (float)head_dim);
            const float inv = __fdiv_rn(1.0f, sqrtf(__fadd_rn(mean, eps)));
#pragma unroll
            for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
                vv[step] = __fmul_rn(vv[step], inv);
            }
        }
#pragma unroll
        for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
            acc[step] = __fadd_rn(acc[step], __fmul_rn(weight, vv[step]));
        }
    }

    const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
#pragma unroll
    for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
        const int dim = (int)lane + 32 * step;
        if (dim < head_dim) {
            const float gated = __fmul_rn(__fmul_rn(acc[step], inverse),
                                           inkling_attn_sigmoid(gate_row[dim]));
            out_row[dim] = __float2bfloat16_rn(gated);
        }
    }
}

extern "C" __global__ void __launch_bounds__(32) inkling_attn_decode(
    const float* __restrict__ q,
    const float* __restrict__ gate,
    const __nv_bfloat16* __restrict__ keys,
    const __nv_bfloat16* __restrict__ values,
    const float* __restrict__ rel_proj,
    const __nv_bfloat16* __restrict__ rel_bank,
    const float* __restrict__ sink,
    const float* __restrict__ q_scale,
    const float* __restrict__ k_scale,
    const float* __restrict__ rel_scale,
    __nv_bfloat16* __restrict__ out,
    int heads,
    int kv_heads,
    int head_dim,
    int groups,
    int position,
    int window,
    int cache_rows,
    int rel_extent,
    int d_rel,
    float qk_clip,
    float rel_clip,
    float tau,
    int value_norm,
    float eps) {
    const int head = blockIdx.x;
    if (head >= heads) {
        return;
    }
    const uint64_t offset = (uint64_t)head * head_dim;
    inkling_attn_one(q + offset, gate + offset, keys, values, rel_proj, rel_bank, sink,
                      q_scale, k_scale, rel_scale, out + offset, head, kv_heads, head_dim, groups,
                      position, window, cache_rows, rel_extent, d_rel, qk_clip, rel_clip, tau,
                      value_norm, eps);
}

// Prefill: grid (heads, rows). Row r is at absolute position first_position + r and sees the
// cache rows 0..=position; a chunked prefill therefore passes the chunk's cache prefix in
// keys/values, and the same body gives the same bits as decoding those rows one at a time.
//
// Three operands are per row and are offset here, not by the body: the query and its gate by
// `row * heads * head_dim`, the relative profile by `row * heads * d_rel`, and the sink by
// `row * heads` - the sink is a projection of the row's own input, one value a head, so a chunk
// that passed row 0's sinks for every row would give a plausible wrong answer in bounds.
extern "C" __global__ void __launch_bounds__(32) inkling_attn_prefill(
    const float* __restrict__ q,
    const float* __restrict__ gate,
    const __nv_bfloat16* __restrict__ keys,
    const __nv_bfloat16* __restrict__ values,
    const float* __restrict__ rel_proj,
    const __nv_bfloat16* __restrict__ rel_bank,
    const float* __restrict__ sink,
    const float* __restrict__ q_scale,
    const float* __restrict__ k_scale,
    const float* __restrict__ rel_scale,
    __nv_bfloat16* __restrict__ out,
    int heads,
    int kv_heads,
    int head_dim,
    int groups,
    int rows,
    int first_position,
    int window,
    int cache_rows,
    int rel_extent,
    int d_rel,
    float qk_clip,
    float rel_clip,
    float tau,
    int value_norm,
    float eps,
    const float* __restrict__ tau_rows) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    if (head >= heads || row >= rows) {
        return;
    }
    // A full layer's log scaling is a function of the absolute position, so a chunk's rows
    // differ. The table is computed on the host by the same rule the row-at-a-time walk uses,
    // which is what keeps the two paths' scores identical rather than merely close.
    const float row_tau = tau_rows != nullptr ? tau_rows[row] : tau;
    const uint64_t head_offset = (uint64_t)head * head_dim;
    const uint64_t row_offset = (uint64_t)row * heads * head_dim;
    inkling_attn_one(q + row_offset + head_offset, gate + row_offset + head_offset, keys,
                      values, rel_proj + (uint64_t)row * heads * d_rel, rel_bank,
                      sink + (uint64_t)row * heads, q_scale, k_scale, rel_scale,
                      out + row_offset + head_offset, head, kv_heads, head_dim, groups,
                      first_position + row, window, cache_rows, rel_extent, d_rel, qk_clip,
                      rel_clip, row_tau, value_norm, eps);
}

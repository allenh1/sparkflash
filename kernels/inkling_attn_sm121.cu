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
//     slot % cache_rows. A sliding layer's cache is its window plus the widest chunk less one, so
//     40 of the 48 layers hold ~1,024 rows (a few thousand for a wide chunk) instead of one row a
//     context position - the difference between 41 GiB and 8 GiB at a million tokens. A chunk
//     stores all its keys before any of its rows attends, and that ring length is what keeps a
//     later row's key off a slot an earlier row still reads; over the slots a chunk's windows
//     address, the modulo is then injective, so a ring and a linear cache read the same rows.
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

#include "opt_pdl_sm121.cuh"

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

// The query side of one (head, row), in the body's lane layout: the learned gain, the clip of
// the scaled vector, then tau, and the clipped relative state (one d_rel value a lane).
__device__ __forceinline__ void inkling_attn_query(
    const float* __restrict__ q_row,
    const float* __restrict__ rel_proj,
    const float* __restrict__ q_scale,
    const float* __restrict__ rel_scale,
    int head,
    int head_dim,
    int d_rel,
    float qk_clip,
    float rel_clip,
    float tau,
    float (&qv)[INKLING_MAX_HEAD_DIM / 32],
    float& rv) {
    const unsigned int lane = threadIdx.x & 31u;
    const float use_tau = tau > 0.0f ? tau : 1.0f;

    // Query: learned scale, then clip, then tau. The released q_l2_clip.scale is
    // [heads, head_dim], a gain per element, and the clip is on the *scaled* vector - clipping
    // the raw vector first would clip a different vector, because the gains are not uniform.
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
    rv = lane < (unsigned int)d_rel ? rel_proj[(uint64_t)head * d_rel + lane] : 0.0f;
    if (rel_scale != nullptr && lane < (unsigned int)d_rel) {
        rv = __fmul_rn(rv, rel_scale[(uint64_t)head * d_rel + (int)lane]);
    }
    const float r_norm = sqrtf(inkling_attn_warp_sum(__fmul_rn(rv, rv)));
    float r_gain = 1.0f;
    if (rel_clip > 0.0f && r_norm > rel_clip) {
        r_gain = rel_clip / r_norm;
    }
    rv = __fmul_rn(rv, __fmul_rn(r_gain, use_tau));
}

// The score of one key row against a prepared query: the key's gain and clip, the dot over
// sqrt(head_dim), and the relative bias of a key `distance` positions back.
__device__ __forceinline__ float inkling_attn_score(
    const float (&qv)[INKLING_MAX_HEAD_DIM / 32],
    float rv,
    const __nv_bfloat16* __restrict__ krow,
    const __nv_bfloat16* __restrict__ rel_bank,
    const float* __restrict__ k_scale,
    int kv_head,
    int head_dim,
    int64_t distance,
    int rel_extent,
    int d_rel,
    float qk_clip,
    float inv_dim) {
    const unsigned int lane = threadIdx.x & 31u;
    float kv[INKLING_MAX_HEAD_DIM / 32];
    float k_norm = 0.0f;
#pragma unroll
    for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
        const int dim = (int)lane + 32 * step;
        // The key carries its own learned gain, and its clip is on the scaled vector too.
        const float gain = dim < head_dim && k_scale != nullptr
                               ? k_scale[(uint64_t)kv_head * head_dim + dim]
                               : 1.0f;
        kv[step] = dim < head_dim ? __fmul_rn(__bfloat162float(krow[dim]), gain) : 0.0f;
        k_norm = __fadd_rn(k_norm, __fmul_rn(kv[step], kv[step]));
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

    if (d_rel > 0 && rel_extent > 0 && distance >= 0 && distance < rel_extent) {
        float bias = 0.0f;
        if (lane < (unsigned int)d_rel) {
            // The released bank is [d_rel, rel_extent]: a row per relative dimension, one
            // column per distance. Indexing it as [head][distance][d_rel] would read a different
            // weight and still stay in bounds, so nothing would catch it.
            const uint64_t index = (uint64_t)lane * rel_extent + (uint64_t)distance;
            bias = __fmul_rn(rv, __bfloat162float(rel_bank[index]));
        }
        score = __fadd_rn(score, inkling_attn_warp_sum(bias));
    }
    return score;
}

// The RMS scale of one value row: 1 / sqrt(mean(v^2) + eps), the sum grouped as the body
// groups it (a lane's dims in step order, then the warp).
__device__ __forceinline__ float inkling_attn_value_inverse(
    const __nv_bfloat16* __restrict__ vrow, int head_dim, float eps) {
    const unsigned int lane = threadIdx.x & 31u;
    float v_norm = 0.0f;
#pragma unroll
    for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
        const int dim = (int)lane + 32 * step;
        const float value = dim < head_dim ? __bfloat162float(vrow[dim]) : 0.0f;
        v_norm = __fadd_rn(v_norm, __fmul_rn(value, value));
    }
    const float mean = __fdiv_rn(inkling_attn_warp_sum(v_norm), (float)head_dim);
    return __fdiv_rn(1.0f, sqrtf(__fadd_rn(mean, eps)));
}

// The first cache slot a row at `position` sees.
__device__ __forceinline__ int64_t inkling_attn_first_slot(int position, int window) {
    return window > 0 ? (position + 1 > window ? position + 1 - window : 0) : 0;
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

    float qv[INKLING_MAX_HEAD_DIM / 32];
    float rv;
    inkling_attn_query(q_row, rel_proj, q_scale, rel_scale, head, head_dim, d_rel, qk_clip,
                       rel_clip, tau, qv, rv);

    const uint64_t kv_stride = (uint64_t)kv_heads * head_dim;
    const uint64_t kv_offset = (uint64_t)kv_head * head_dim;
    const int64_t first = inkling_attn_first_slot(position, window);

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
        const float score = inkling_attn_score(qv, rv, krow, rel_bank, k_scale, kv_head, head_dim,
                                               (int64_t)position - slot, rel_extent, d_rel,
                                               qk_clip, inv_dim);

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

        float vv[INKLING_MAX_HEAD_DIM / 32];
#pragma unroll
        for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
            const int dim = (int)lane + 32 * step;
            vv[step] = dim < head_dim ? __bfloat162float(vrow[dim]) : 0.0f;
        }
        if (value_norm != 0) {
            const float inv = inkling_attn_value_inverse(vrow, head_dim, eps);
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
    OPT_PDL_ENTRY();
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
    OPT_PDL_ENTRY();
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

// ---------------------------------------------------------------------------------------------
// The split walk: the same bits as inkling_attn_one, in three launches.
//
// The serial body spends a context position's whole latency - two row loads and four warp
// reductions - inside the softmax recurrence, so a 1,024-key layer costs a millisecond on one
// warp. Only the running maximum, the denominator and the accumulators are a recurrence; a key's
// score and its value row's RMS scale depend on nothing before them. The split walk therefore
// computes every key's clip gain and value scale once (inkling_attn_key_scales), scores every key
// of a pass in parallel (inkling_attn_scores), and then runs the recurrence alone
// (inkling_attn_mix).
//
// Every number is the serial body's. The per-lane partial sums are the body's expressions in
// the body's lane layout, and the cross-lane sums pair lanes exactly as inkling_attn_warp_sum
// does (offset 16, then 8, 4, 2, 1); inkling_attn_warp_sums only reduces several sums at once,
// each pair-sum being the same two operands (IEEE addition commutes exactly). A key's clip gain
// and value scale depend on the key alone, so computing them once a key instead of once a
// (row, head) changes no bit.
//
// The mix keeps the recurrence's every rounding. Its one rewrite is the running maximum: the
// serial walk replaces the maximum only on a strictly greater score, which keeps the first of
// equal scores, and "the first maximal element" is associative, so a warp scan over 32 scores
// with a carried prefix gives each key the maximum the serial walk holds before and after it.
// With those, the rescale and weight exponentials of 32 keys are independent and run a lane
// each; what stays serial is the denominator's and the accumulators' multiply-adds, in key
// order. A NaN breaks the associativity (a NaN is never "greater", so where it sits matters), so
// a tile holding one replays the serial rule key by key instead.
//
// A pass covers slot indices [pass_first, pass_first + pass_slots) of each row's own range
// (index 0 is the row's first visible slot), and the slots its rows touch are
// [scale_first, scale_first + pass_slots + rows - 1): row 0's first visible slot plus
// pass_first is scale_first, and a later row's first slot is at most rows - 1 further on. The
// mix carries the running state between passes in `state`, so a full layer at a million
// positions needs no score buffer a million wide.
//
//   key_gains, value_scales   [kv_heads][scale_stride] f32, slot scale_first + i at i
//   scores                    [rows][heads][pass_slots] f32, a row's index pass_first + i at i
//   state                     [rows][heads][2 + INKLING_MAX_HEAD_DIM] f32: maximum,
//                             denominator, accumulators
// ---------------------------------------------------------------------------------------------

#define INKLING_SCORE_RUN 4
#define INKLING_SCORE_WARPS 4
#define INKLING_MAX_GROUPS 8
#define INKLING_STATE_WIDTH (2 + INKLING_MAX_HEAD_DIM)

// K sums at once (K a power of two, at most 32), each paired across lanes exactly as
// inkling_attn_warp_sum pairs one: offset 16 first, then 8, 4, 2, 1. A level that still holds
// several sums hands half of them to the partner lane and keeps the other half, so each sum
// costs one shuffle a level until one is left. Lane l returns sum (l >> (5 - log2 K)).
template <int K>
__device__ __forceinline__ float inkling_attn_warp_sums(float (&value)[K]) {
    const unsigned int lane = threadIdx.x & 31u;
    int offset = 16;
#pragma unroll
    for (int half = K / 2; half >= 1; half >>= 1) {
        const bool upper = (lane & (unsigned int)offset) != 0;
#pragma unroll
        for (int i = 0; i < half; ++i) {
            const float send = upper ? value[i] : value[i + half];
            const float keep = upper ? value[i + half] : value[i];
            value[i] = keep + __shfl_xor_sync(0xffffffffu, send, offset);
        }
        offset >>= 1;
    }
    float total = value[0];
#pragma unroll
    for (; offset > 0; offset >>= 1) {
        total += __shfl_xor_sync(0xffffffffu, total, offset);
    }
    return total;
}

// The cache row of an absolute slot: the ring index for a sliding layer.
__device__ __forceinline__ uint64_t inkling_attn_cache_row(int64_t slot, int cache_rows) {
    return cache_rows > 0 ? (uint64_t)((uint32_t)slot % (uint32_t)cache_rows) : (uint64_t)slot;
}

// grid (ceil(span / (RUN * WARPS)), kv_heads), block 32 * INKLING_SCORE_WARPS: slots
// scale_first .. scale_first + span.
extern "C" __global__ void __launch_bounds__(32 * INKLING_SCORE_WARPS) inkling_attn_key_scales(
    const __nv_bfloat16* __restrict__ keys,
    const __nv_bfloat16* __restrict__ values,
    const float* __restrict__ k_scale,
    float* __restrict__ key_gains,
    float* __restrict__ value_scales,
    int kv_heads,
    int head_dim,
    int cache_rows,
    float qk_clip,
    int value_norm,
    float eps,
    long long scale_first,
    int span,
    int scale_stride) {
    OPT_PDL_ENTRY();
    const int kv_head = blockIdx.y;
    if (kv_head >= kv_heads || head_dim <= 0 || head_dim > INKLING_MAX_HEAD_DIM) {
        return;
    }
    const unsigned int lane = threadIdx.x & 31u;
    const int warp = threadIdx.x >> 5;
    const uint64_t kv_stride = (uint64_t)kv_heads * head_dim;
    const uint64_t kv_offset = (uint64_t)kv_head * head_dim;
    const int begin = (blockIdx.x * INKLING_SCORE_WARPS + warp) * INKLING_SCORE_RUN;
#pragma unroll
    for (int i = 0; i < INKLING_SCORE_RUN; ++i) {
        const int index = begin + i;
        if (index < span) {
            const uint64_t row = inkling_attn_cache_row(scale_first + index, cache_rows);
            const __nv_bfloat16* krow = keys + row * kv_stride + kv_offset;
            const __nv_bfloat16* vrow = values + row * kv_stride + kv_offset;
            float sums[2] = {0.0f, 0.0f};
#pragma unroll
            for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
                const int dim = (int)lane + 32 * step;
                const float gain = dim < head_dim && k_scale != nullptr
                                       ? k_scale[(uint64_t)kv_head * head_dim + dim]
                                       : 1.0f;
                const float key = dim < head_dim ? __fmul_rn(__bfloat162float(krow[dim]), gain)
                                                 : 0.0f;
                const float value = dim < head_dim ? __bfloat162float(vrow[dim]) : 0.0f;
                sums[0] = __fadd_rn(sums[0], __fmul_rn(key, key));
                sums[1] = __fadd_rn(sums[1], __fmul_rn(value, value));
            }
            // Lanes 0-15 hold the key's sum, lanes 16-31 the value's.
            const float total = inkling_attn_warp_sums<2>(sums);
            if (lane == 0) {
                const float k_norm = sqrtf(total);
                float k_gain = 1.0f;
                if (qk_clip > 0.0f && k_norm > qk_clip) {
                    k_gain = qk_clip / k_norm;
                }
                key_gains[(uint64_t)kv_head * scale_stride + index] = k_gain;
            } else if (lane == 16) {
                float inverse = 1.0f;
                if (value_norm != 0) {
                    const float mean = __fdiv_rn(total, (float)head_dim);
                    inverse = __fdiv_rn(1.0f, sqrtf(__fadd_rn(mean, eps)));
                }
                value_scales[(uint64_t)kv_head * scale_stride + index] = inverse;
            }
        }
    }
}

// grid (blocks, kv_heads, rows), block 32 * INKLING_SCORE_WARPS: a warp scores runs of RUN keys
// for every query head of one KV group (groups <= INKLING_MAX_GROUPS), so a key row is loaded
// once a group. Warp w of block b takes the run at (b * WARPS + w) * RUN and every
// gridDim.x * WARPS * RUN keys after it, so any grid covers the pass; the host caps gridDim.x
// (SCORE_GRID in device.rs) and a long pass's warps then walk many runs. A warp prepares its
// queries once, and the next run's key loads are in flight while it scores the current one.
// Every score is the same arithmetic whatever the grid: only which warp computes it changes.
extern "C" __global__ void __launch_bounds__(32 * INKLING_SCORE_WARPS) inkling_attn_scores(
    const float* __restrict__ q,
    const __nv_bfloat16* __restrict__ keys,
    const float* __restrict__ rel_proj,
    const __nv_bfloat16* __restrict__ rel_bank,
    const float* __restrict__ q_scale,
    const float* __restrict__ k_scale,
    const float* __restrict__ rel_scale,
    const float* __restrict__ key_gains,
    float* __restrict__ scores,
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
    const float* __restrict__ tau_rows,
    long long pass_first,
    int pass_slots,
    long long scale_first,
    int scale_stride) {
    OPT_PDL_ENTRY();
    const int kv_head = blockIdx.y;
    const int row = blockIdx.z;
    if (kv_head >= kv_heads || row >= rows || head_dim <= 0 || head_dim > INKLING_MAX_HEAD_DIM ||
        d_rel < 0 || d_rel > INKLING_MAX_D_REL || groups <= 0 || groups > INKLING_MAX_GROUPS) {
        return;
    }
    const unsigned int lane = threadIdx.x & 31u;
    const int warp = threadIdx.x >> 5;
    const int position = first_position + row;
    const int64_t first = inkling_attn_first_slot(position, window);
    const int64_t count = (int64_t)position - first + 1;
    const int64_t end = pass_first + pass_slots < count ? pass_first + pass_slots : count;
    const int64_t stride = (int64_t)gridDim.x * INKLING_SCORE_WARPS * INKLING_SCORE_RUN;
    int64_t begin =
        pass_first + ((int64_t)blockIdx.x * INKLING_SCORE_WARPS + warp) * INKLING_SCORE_RUN;
    if (begin >= end) {
        return;
    }
    const float row_tau = tau_rows != nullptr ? tau_rows[row] : tau;
    const uint64_t row_offset = (uint64_t)row * heads * head_dim;
    float qv[INKLING_MAX_GROUPS][INKLING_MAX_HEAD_DIM / 32];
    float rv[INKLING_MAX_GROUPS];
#pragma unroll
    for (int g = 0; g < INKLING_MAX_GROUPS; ++g) {
        const int head = kv_head * groups + g;
        if (g < groups) {
            inkling_attn_query(q + row_offset + (uint64_t)head * head_dim,
                               rel_proj + (uint64_t)row * heads * d_rel, q_scale, rel_scale, head,
                               head_dim, d_rel, qk_clip, rel_clip, row_tau, qv[g], rv[g]);
        } else {
#pragma unroll
            for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
                qv[g][step] = 0.0f;
            }
            rv[g] = 0.0f;
        }
    }
    const float inv_dim = 1.0f / sqrtf((float)head_dim);
    const uint64_t kv_stride = (uint64_t)kv_heads * head_dim;
    const uint64_t kv_offset = (uint64_t)kv_head * head_dim;
    const float* gains = key_gains + (uint64_t)kv_head * scale_stride - scale_first;
    float gain[INKLING_MAX_HEAD_DIM / 32];
#pragma unroll
    for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
        const int dim = (int)lane + 32 * step;
        gain[step] =
            dim < head_dim && k_scale != nullptr ? k_scale[(uint64_t)kv_head * head_dim + dim] : 1.0f;
    }
    // loaded[i][step]: this lane's BF16 of key `at + i`, dim lane + 32 step.
    __nv_bfloat16 loaded[INKLING_SCORE_RUN][INKLING_MAX_HEAD_DIM / 32];
    const auto load = [&](int64_t at) {
#pragma unroll
        for (int i = 0; i < INKLING_SCORE_RUN; ++i) {
            if (at + i < end) {
                const __nv_bfloat16* krow =
                    keys + inkling_attn_cache_row(first + at + i, cache_rows) * kv_stride +
                    kv_offset;
#pragma unroll
                for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
                    const int dim = (int)lane + 32 * step;
                    loaded[i][step] = dim < head_dim ? krow[dim] : __float2bfloat16(0.0f);
                }
            }
        }
    };
    load(begin);
    for (;;) {
        __nv_bfloat16 run[INKLING_SCORE_RUN][INKLING_MAX_HEAD_DIM / 32];
#pragma unroll
        for (int i = 0; i < INKLING_SCORE_RUN; ++i) {
#pragma unroll
            for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
                run[i][step] = loaded[i][step];
            }
        }
        const int64_t next = begin + stride;
        if (next < end) {
            load(next);
        }
#pragma unroll
        for (int i = 0; i < INKLING_SCORE_RUN; ++i) {
            const int64_t index = begin + i;
            if (index < end) {
                const int64_t slot = first + index;
                const float k_gain = gains[slot];
                float kv[INKLING_MAX_HEAD_DIM / 32];
#pragma unroll
                for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
                    const int dim = (int)lane + 32 * step;
                    kv[step] =
                        dim < head_dim ? __fmul_rn(__bfloat162float(run[i][step]), gain[step]) : 0.0f;
                    kv[step] = __fmul_rn(kv[step], k_gain);
                }
                const int64_t distance = (int64_t)position - slot;
                const bool biased =
                    d_rel > 0 && rel_extent > 0 && distance >= 0 && distance < rel_extent;
                const float bank =
                    biased && lane < (unsigned int)d_rel
                        ? __bfloat162float(
                              rel_bank[(uint64_t)lane * rel_extent + (uint64_t)distance])
                        : 0.0f;
                // Sums 0-7: a head's dot; 8-15: its relative bias.
                float sums[2 * INKLING_MAX_GROUPS];
#pragma unroll
                for (int g = 0; g < INKLING_MAX_GROUPS; ++g) {
                    float dot = 0.0f;
#pragma unroll
                    for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
                        dot = __fadd_rn(dot, __fmul_rn(qv[g][step], kv[step]));
                    }
                    sums[g] = dot;
                    sums[INKLING_MAX_GROUPS + g] =
                        lane < (unsigned int)d_rel ? __fmul_rn(rv[g], bank) : 0.0f;
                }
                // Lane 2g holds head g's dot, lane 16 + 2g its bias.
                const float total = inkling_attn_warp_sums<2 * INKLING_MAX_GROUPS>(sums);
                const float bias = __shfl_sync(0xffffffffu, total, (lane + 16u) & 31u);
                const int g = (int)(lane >> 1);
                if ((lane & 1u) == 0 && g < groups) {
                    float score = __fmul_rn(total, inv_dim);
                    if (biased) {
                        score = __fadd_rn(score, bias);
                    }
                    const int head = kv_head * groups + g;
                    scores[((uint64_t)row * heads + head) * pass_slots + (index - pass_first)] =
                        score;
                }
            }
        }
        if (next >= end) {
            break;
        }
        begin = next;
    }
}

// The serial walk's "first maximal element": `later` replaces `earlier` only when greater.
__device__ __forceinline__ float inkling_attn_first_max(float earlier, float later) {
    return later > earlier ? later : earlier;
}

// One tile of the mix's loads: 32 keys' scores and value scales (a lane a key) and this
// thread's value at each of them.
struct InklingMixTile {
    float score;
    float scale;
    float value[32];
};

// Loads the tile at index `tile` of a row whose range starts at slot `first`. The ring index
// steps by one a key and wraps at most once a tile when the ring holds 32 rows or more; a
// smaller ring takes the modulo a key.
__device__ __forceinline__ void inkling_mix_load(
    InklingMixTile& out,
    const float* __restrict__ score_in,
    const float* __restrict__ scale_in,
    const __nv_bfloat16* __restrict__ column,
    int64_t first,
    int64_t tile,
    int64_t end,
    uint64_t kv_stride,
    int cache_rows,
    bool owns_dim) {
    const unsigned int lane = threadIdx.x & 31u;
    const int64_t index = tile + lane;
    const bool valid = index < end;
    out.score = valid ? score_in[index] : -INFINITY;
    out.scale = valid ? scale_in[first + index] : 1.0f;
    const int n = end - tile < 32 ? (int)(end - tile) : 32;
    uint32_t row = (uint32_t)inkling_attn_cache_row(first + tile, cache_rows);
    const __nv_bfloat16* at = column + (uint64_t)row * kv_stride;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        out.value[i] = i < n && owns_dim ? __bfloat162float(*at) : 0.0f;
        if (cache_rows >= 32) {
            ++row;
            at += kv_stride;
            if (row == (uint32_t)cache_rows) {
                row = 0;
                at = column;
            }
        } else {
            row = (uint32_t)inkling_attn_cache_row(first + tile + i + 1, cache_rows);
            at = column + (uint64_t)row * kv_stride;
        }
    }
}

// grid (heads, rows), block INKLING_MAX_HEAD_DIM (a warp a body step: thread 32w + l owns dim
// l + 32w, the dim the serial body's lane l holds at step w).
extern "C" __global__ void __launch_bounds__(INKLING_MAX_HEAD_DIM) inkling_attn_mix(
    const float* __restrict__ gate,
    const __nv_bfloat16* __restrict__ values,
    const float* __restrict__ sink,
    const float* __restrict__ scores,
    const float* __restrict__ value_scales,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ out,
    int heads,
    int kv_heads,
    int head_dim,
    int groups,
    int rows,
    int first_position,
    int window,
    int cache_rows,
    int value_norm,
    long long pass_first,
    int pass_slots,
    long long scale_first,
    int scale_stride,
    int first_pass,
    int last_pass) {
    OPT_PDL_ENTRY();
    // A key's terms, a warp's own copy: (grows, factor, weight, value scale).
    __shared__ float4 terms[INKLING_MAX_HEAD_DIM / 32][32];
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    if (head >= heads || row >= rows) {
        return;
    }
    const unsigned int lane = threadIdx.x & 31u;
    const int warp = threadIdx.x >> 5;
    const int dim = threadIdx.x;
    const uint64_t row_head = (uint64_t)row * heads + head;
    __nv_bfloat16* out_row = out + row_head * head_dim;
    if (head_dim <= 0 || head_dim > INKLING_MAX_HEAD_DIM || groups <= 0) {
        if (dim < head_dim) {
            out_row[dim] = __float2bfloat16_rn(0.0f);
        }
        return;
    }
    const int position = first_position + row;
    const int64_t first = inkling_attn_first_slot(position, window);
    const int64_t count = (int64_t)position - first + 1;
    const int kv_head = head / groups;
    const uint64_t kv_stride = (uint64_t)kv_heads * head_dim;
    const bool owns_dim = dim < head_dim;
    const __nv_bfloat16* column =
        values + (uint64_t)kv_head * head_dim + (owns_dim ? dim : 0);
    float* row_state = state + row_head * INKLING_STATE_WIDTH;

    float maximum;
    float denominator;
    float acc;
    if (first_pass != 0) {
        // The sink is a value-less term of the denominator, so it starts the running softmax.
        maximum = -INFINITY;
        denominator = 0.0f;
        if (sink != nullptr) {
            maximum = sink[row_head];
            denominator = 1.0f;
        }
        acc = 0.0f;
    } else {
        maximum = row_state[0];
        denominator = row_state[1];
        acc = row_state[2 + dim];
    }

    const float* score_in = scores + row_head * pass_slots - pass_first;
    const float* scale_in = value_scales + (uint64_t)kv_head * scale_stride - scale_first;
    const int64_t end = pass_first + pass_slots < count ? pass_first + pass_slots : count;
    InklingMixTile next;
    if (pass_first < end) {
        inkling_mix_load(next, score_in, scale_in, column, first, pass_first, end, kv_stride,
                         cache_rows, owns_dim);
    }
    for (int64_t tile = pass_first; tile < end; tile += 32) {
        const InklingMixTile current = next;
        // The next tile's loads are in flight while this one's recurrence runs.
        if (tile + 32 < end) {
            inkling_mix_load(next, score_in, scale_in, column, first, tile + 32, end, kv_stride,
                             cache_rows, owns_dim);
        }
        const bool valid = tile + lane < end;
        const int n = end - tile < 32 ? (int)(end - tile) : 32;
        const float score = current.score;

        // Each key's rule from the maximum the serial walk holds before and after it: `grows`
        // (a new maximum), `factor` (the rescale when it grows, else the denominator's term)
        // and `weight` (exp(score - maximum after)).
        bool grows;
        float factor;
        float weight;
        const bool any_nan = __any_sync(0xffffffffu, valid && isnan(score)) || isnan(maximum);
        if (!any_nan) {
            float scan = score;
#pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                const float earlier = __shfl_up_sync(0xffffffffu, scan, offset);
                if ((int)lane >= offset) {
                    scan = inkling_attn_first_max(earlier, scan);
                }
            }
            const float after = inkling_attn_first_max(maximum, scan);
            float before = __shfl_up_sync(0xffffffffu, after, 1);
            if (lane == 0) {
                before = maximum;
            }
            grows = score > before;
            factor = grows ? expf(before - score) : expf(score - before);
            weight = expf(score - after);
            maximum = __shfl_sync(0xffffffffu, after, n - 1);
        } else {
            grows = false;
            factor = 0.0f;
            weight = 0.0f;
            float running = maximum;
            for (int i = 0; i < n; ++i) {
                const float key = __shfl_sync(0xffffffffu, score, i);
                const bool key_grows = key > running;
                const float key_factor = key_grows ? expf(running - key) : expf(key - running);
                if (key_grows) {
                    running = key;
                }
                const float key_weight = expf(key - running);
                if ((int)lane == i) {
                    grows = key_grows;
                    factor = key_factor;
                    weight = key_weight;
                }
            }
            maximum = running;
        }
        terms[warp][lane] = make_float4(grows ? 1.0f : 0.0f, factor, weight, current.scale);
        __syncwarp();

        // The recurrence, in key order. Both arms of the serial walk's branch are evaluated and
        // one kept, which is the same arithmetic without a branch a key.
        auto step = [&](int i) {
            const float4 key = terms[warp][i];
            const bool key_grows = key.x != 0.0f;
            const float vv = value_norm != 0 ? __fmul_rn(current.value[i], key.w)
                                             : current.value[i];
            const float grown = __fadd_rn(__fmul_rn(denominator, key.y), 1.0f);
            const float kept = __fadd_rn(denominator, key.y);
            denominator = key_grows ? grown : kept;
            const float scaled = key_grows ? __fmul_rn(acc, key.y) : acc;
            acc = __fadd_rn(scaled, __fmul_rn(key.z, vv));
        };
        if (n == 32) {
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                step(i);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                if (i < n) {
                    step(i);
                }
            }
        }
        __syncwarp();
    }

    if (last_pass == 0) {
        if (threadIdx.x == 0) {
            row_state[0] = maximum;
            row_state[1] = denominator;
        }
        row_state[2 + dim] = acc;
        return;
    }
    if (owns_dim) {
        const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
        const float gated = __fmul_rn(__fmul_rn(acc, inverse),
                                      inkling_attn_sigmoid(gate[row_head * head_dim + dim]));
        out_row[dim] = __float2bfloat16_rn(gated);
    }
}

// ---------------------------------------------------------------------------------------------
// The split-key decode walk: the split walk's key scales and scores, then the softmax split over
// runs of INKLING_PART_KEYS keys ("flash-decoding", the partials-then-merge order Dory's
// dory_attn_split and dory_attn_combine take).
//
// Not the exact walks' bits. inkling_attn_mix runs a head's recurrence key by key in one block,
// so a row's attention is as long as its context: at 93k positions a full layer is 2,900
// dependent tiles on 12 blocks. Here block (kv head, run, row) takes one run of a row's keys for
// every query head of the group, so a value row is read once a group, and writes a partial a
// head: the run's largest score, its sum of exp(score - largest) and the value sums weighted by
// those terms and the keys' value scales. inkling_attn_merge then merges a head's runs in run
// order with the sink, divides and applies the gate. Every score, clip gain and value scale is
// the exact walks' (the same inkling_attn_scores and inkling_attn_key_scales launches); what
// changes is the order of the denominator's and the value sums' additions and that each run's
// terms are taken against its own maximum. The order is fixed by the row's key count alone, so
// a row always reduces the same way.
//
//   parts   [rows][heads][part_stride][2 + head_dim] f32: a run's maximum, its sum, its values
// ---------------------------------------------------------------------------------------------

#define INKLING_PART_KEYS 256
#define INKLING_PART_THREADS 128
// The scores a thread holds a head: keys t, t + THREADS, ... of the run.
#define INKLING_PART_HELD (INKLING_PART_KEYS / INKLING_PART_THREADS)
// A thread of the value sum owns 8 consecutive dims (one 16-byte load a value row); 16 of them
// cover a 128-wide head, so the block takes 8 keys at once.
#define INKLING_PART_DIMS 8
#define INKLING_PART_LANES (INKLING_MAX_HEAD_DIM / INKLING_PART_DIMS)
#define INKLING_PART_STRIDE (INKLING_PART_THREADS / INKLING_PART_LANES)

// The largest of `value` across the block, for each of the first `count` entries.
template <int K>
__device__ __forceinline__ void inkling_part_block_max(float (&value)[K], float (&shared)[4][K],
                                                       int count) {
    const unsigned int lane = threadIdx.x & 31u;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int g = 0; g < K; ++g) {
        if (g < count) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                value[g] = fmaxf(value[g], __shfl_xor_sync(0xffffffffu, value[g], offset));
            }
            if (lane == 0) {
                shared[warp][g] = value[g];
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < K; ++g) {
        if (g < count) {
            value[g] = fmaxf(fmaxf(shared[0][g], shared[1][g]), fmaxf(shared[2][g], shared[3][g]));
        }
    }
    __syncthreads();
}

// The sum of `value` across the block, the same way.
template <int K>
__device__ __forceinline__ void inkling_part_block_sum(float (&value)[K], float (&shared)[4][K],
                                                       int count) {
    const unsigned int lane = threadIdx.x & 31u;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int g = 0; g < K; ++g) {
        if (g < count) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                value[g] += __shfl_xor_sync(0xffffffffu, value[g], offset);
            }
            if (lane == 0) {
                shared[warp][g] = value[g];
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < K; ++g) {
        if (g < count) {
            value[g] = (shared[0][g] + shared[1][g]) + (shared[2][g] + shared[3][g]);
        }
    }
    __syncthreads();
}

// grid (kv_heads, rows, ceil(pass span / INKLING_PART_KEYS)), INKLING_PART_THREADS threads: the
// runs of this pass's keys [pass_first, pass_first + pass_slots) of each row's own range, whose
// scores inkling_attn_scores wrote. pass_first is a multiple of INKLING_PART_KEYS, so run
// pass_first / INKLING_PART_KEYS + blockIdx.z is the run's place in the row. The rows sit
// beside each other in the grid, so a verify's rows read a run's values together and all but
// the first find them in L2; which block takes a (row, run) never changes its bits. Needs head_dim a
// multiple of 8 and at most INKLING_MAX_HEAD_DIM (the host refuses any other shape).
extern "C" __global__ void __launch_bounds__(INKLING_PART_THREADS) inkling_attn_part(
    const __nv_bfloat16* __restrict__ values,
    const float* __restrict__ scores,
    const float* __restrict__ value_scales,
    float* __restrict__ parts,
    int heads,
    int kv_heads,
    int head_dim,
    int groups,
    int rows,
    int first_position,
    int window,
    int cache_rows,
    int value_norm,
    long long pass_first,
    int pass_slots,
    long long scale_first,
    int scale_stride,
    int part_stride) {
    OPT_PDL_ENTRY();
    __shared__ float weights[INKLING_MAX_GROUPS][INKLING_PART_KEYS];
    __shared__ float scales[INKLING_PART_KEYS];
    __shared__ float block[4][INKLING_MAX_GROUPS];
    __shared__ float sums[INKLING_PART_STRIDE][INKLING_MAX_GROUPS][INKLING_MAX_HEAD_DIM];
    const int kv_head = blockIdx.x;
    const int row = blockIdx.y;
    if (kv_head >= kv_heads || row >= rows || head_dim <= 0 || head_dim > INKLING_MAX_HEAD_DIM ||
        head_dim % INKLING_PART_DIMS != 0 || groups <= 0 || groups > INKLING_MAX_GROUPS) {
        return;
    }
    const int position = first_position + row;
    const int64_t first = inkling_attn_first_slot(position, window);
    const int64_t count = (int64_t)position - first + 1;
    const int64_t begin = pass_first + (int64_t)blockIdx.z * INKLING_PART_KEYS;
    int64_t end = begin + INKLING_PART_KEYS;
    end = end < pass_first + pass_slots ? end : pass_first + pass_slots;
    end = end < count ? end : count;
    if (begin >= end) {
        return;
    }
    const int n = (int)(end - begin);
    const int head0 = kv_head * groups;

    // The run's largest score a head, then each key's term against it. A thread reads its keys'
    // scores (t and t + INKLING_PART_THREADS) once, for the maximum and the terms both.
    float held[INKLING_MAX_GROUPS][INKLING_PART_HELD];
    float largest[INKLING_MAX_GROUPS];
#pragma unroll
    for (int g = 0; g < INKLING_MAX_GROUPS; ++g) {
        largest[g] = -INFINITY;
        if (g < groups) {
            const float* score_row = scores + ((uint64_t)row * heads + head0 + g) * pass_slots +
                                     (begin - pass_first);
#pragma unroll
            for (int k = 0; k < INKLING_PART_HELD; ++k) {
                const int i = threadIdx.x + k * INKLING_PART_THREADS;
                held[g][k] = i < n ? score_row[i] : -INFINITY;
                if (i < n) {
                    largest[g] = fmaxf(largest[g], held[g][k]);
                }
            }
        }
    }
    inkling_part_block_max(largest, block, groups);
    const float* scale_row = value_scales + (uint64_t)kv_head * scale_stride - scale_first + first;
    for (int i = threadIdx.x; i < n; i += INKLING_PART_THREADS) {
        scales[i] = value_norm != 0 ? scale_row[begin + i] : 1.0f;
    }
    float total[INKLING_MAX_GROUPS];
#pragma unroll
    for (int g = 0; g < INKLING_MAX_GROUPS; ++g) {
        total[g] = 0.0f;
        if (g < groups) {
            // An all -inf run (no key can carry weight) adds nothing; exp(-inf - -inf) is NaN.
            const bool live = largest[g] != -INFINITY;
#pragma unroll
            for (int k = 0; k < INKLING_PART_HELD; ++k) {
                const int i = threadIdx.x + k * INKLING_PART_THREADS;
                if (i < n) {
                    const float term = live ? expf(held[g][k] - largest[g]) : 0.0f;
                    weights[g][i] = term;
                    total[g] += term;
                }
            }
        }
    }
    inkling_part_block_sum(total, block, groups);

    // The weighted value sums: thread t takes dims 8 (t % 16) .. + 7 of keys t / 16, + 8, ...
    const int lane = threadIdx.x % INKLING_PART_LANES;
    const int key_lane = threadIdx.x / INKLING_PART_LANES;
    const int dim0 = lane * INKLING_PART_DIMS;
    const bool owns = dim0 < head_dim;
    const uint64_t kv_stride = (uint64_t)kv_heads * head_dim;
    const __nv_bfloat16* column = values + (uint64_t)kv_head * head_dim + (owns ? dim0 : 0);
    float acc[INKLING_MAX_GROUPS][INKLING_PART_DIMS];
#pragma unroll
    for (int g = 0; g < INKLING_MAX_GROUPS; ++g) {
#pragma unroll
        for (int j = 0; j < INKLING_PART_DIMS; ++j) {
            acc[g][j] = 0.0f;
        }
    }
    if (owns) {
#pragma unroll 4
        for (int i = key_lane; i < n; i += INKLING_PART_STRIDE) {
            const uint64_t cache_row = inkling_attn_cache_row(first + begin + i, cache_rows);
            const uint4 packed = *reinterpret_cast<const uint4*>(column + cache_row * kv_stride);
            const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
            float value[INKLING_PART_DIMS];
#pragma unroll
            for (int pair = 0; pair < INKLING_PART_DIMS / 2; ++pair) {
                const float2 two = __bfloat1622float2(pairs[pair]);
                value[2 * pair] = two.x * scales[i];
                value[2 * pair + 1] = two.y * scales[i];
            }
#pragma unroll
            for (int g = 0; g < INKLING_MAX_GROUPS; ++g) {
                if (g < groups) {
                    const float weight = weights[g][i];
#pragma unroll
                    for (int j = 0; j < INKLING_PART_DIMS; ++j) {
                        acc[g][j] = fmaf(weight, value[j], acc[g][j]);
                    }
                }
            }
        }
#pragma unroll
        for (int g = 0; g < INKLING_MAX_GROUPS; ++g) {
            if (g < groups) {
#pragma unroll
                for (int j = 0; j < INKLING_PART_DIMS; ++j) {
                    sums[key_lane][g][dim0 + j] = acc[g][j];
                }
            }
        }
    }
    __syncthreads();
    const uint64_t run = (uint64_t)(pass_first / INKLING_PART_KEYS) + blockIdx.z;
    for (int index = threadIdx.x; index < groups * head_dim; index += INKLING_PART_THREADS) {
        const int g = index / head_dim;
        const int dim = index - g * head_dim;
        float value = 0.0f;
#pragma unroll
        for (int k = 0; k < INKLING_PART_STRIDE; ++k) {
            value += sums[k][g][dim];
        }
        float* out = parts + (((uint64_t)row * heads + head0 + g) * part_stride + run) *
                                 (2 + (uint64_t)head_dim);
        if (dim == 0) {
            out[0] = largest[g];
            out[1] = total[g];
        }
        out[2 + dim] = value;
    }
}

// The runs one merge block holds factors for in shared memory: a 1M-key row's 4,096 runs of
// INKLING_PART_KEYS. A row with more takes the one-thread-a-dim walk.
#define INKLING_MERGE_RUNS 4096

// grid (heads, rows), INKLING_MAX_HEAD_DIM threads (thread d owns dim d): merges a head's runs
// in run order, the sink opening the softmax as it opens the exact walks', divides and applies
// the gate.
//
// The block shares what every dim needs: the threads take the runs' maxima congruent to their
// index (fmaxf returns one of its operands, so the largest is the same in any order) and then
// each run's factor exp(largest_r - largest), into shared memory. Each dim's thread then folds
// the denominator and its value sum over the runs in run order with those factors, the fmaf
// chain the one-thread walk ran.
extern "C" __global__ void __launch_bounds__(INKLING_MAX_HEAD_DIM) inkling_attn_merge(
    const float* __restrict__ gate,
    const float* __restrict__ sink,
    const float* __restrict__ parts,
    __nv_bfloat16* __restrict__ out,
    int heads,
    int head_dim,
    int rows,
    int first_position,
    int window,
    int part_stride) {
    OPT_PDL_ENTRY();
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int dim = threadIdx.x;
    if (head >= heads || row >= rows) {
        return;
    }
    const uint64_t row_head = (uint64_t)row * heads + head;
    const int position = first_position + row;
    const int64_t first = inkling_attn_first_slot(position, window);
    const int64_t count = (int64_t)position - first + 1;
    const int runs = (int)((count + INKLING_PART_KEYS - 1) / INKLING_PART_KEYS);
    const uint64_t width = 2 + (uint64_t)head_dim;
    const float* part = parts + row_head * part_stride * width;
    const bool sunk = sink != nullptr;
    if (runs > INKLING_MERGE_RUNS) {
        if (dim >= head_dim) {
            return;
        }
        float largest = sunk ? sink[row_head] : -INFINITY;
        for (int r = 0; r < runs; ++r) {
            largest = fmaxf(largest, part[r * width]);
        }
        float denominator = 0.0f;
        float acc = 0.0f;
        if (largest != -INFINITY) {
            denominator = sunk ? expf(sink[row_head] - largest) : 0.0f;
            for (int r = 0; r < runs; ++r) {
                const float* at = part + r * width;
                const float factor = at[0] != -INFINITY ? expf(at[0] - largest) : 0.0f;
                denominator = fmaf(at[1], factor, denominator);
                acc = fmaf(at[2 + dim], factor, acc);
            }
        }
        const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
        const float gated = __fmul_rn(__fmul_rn(acc, inverse),
                                      inkling_attn_sigmoid(gate[row_head * head_dim + dim]));
        out[row_head * head_dim + dim] = __float2bfloat16_rn(gated);
        return;
    }
    __shared__ float factors[INKLING_MERGE_RUNS];
    __shared__ float warp_largest[INKLING_MAX_HEAD_DIM / 32];
    float mine = -INFINITY;
    for (int r = threadIdx.x; r < runs; r += blockDim.x) {
        mine = fmaxf(mine, part[r * width]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        mine = fmaxf(mine, __shfl_xor_sync(0xFFFFFFFFu, mine, offset));
    }
    if ((threadIdx.x & 31) == 0) {
        warp_largest[threadIdx.x >> 5] = mine;
    }
    __syncthreads();
    float largest = sunk ? sink[row_head] : -INFINITY;
    for (int w = 0; w < (int)((blockDim.x + 31) >> 5); ++w) {
        largest = fmaxf(largest, warp_largest[w]);
    }
    if (largest != -INFINITY) {
        for (int r = threadIdx.x; r < runs; r += blockDim.x) {
            const float top = part[r * width];
            factors[r] = top != -INFINITY ? expf(top - largest) : 0.0f;
        }
    }
    __syncthreads();
    if (dim >= head_dim) {
        return;
    }
    float denominator = 0.0f;
    float acc = 0.0f;
    if (largest != -INFINITY) {
        denominator = sunk ? expf(sink[row_head] - largest) : 0.0f;
#pragma unroll 8
        for (int r = 0; r < runs; ++r) {
            const float* at = part + r * width;
            const float factor = factors[r];
            denominator = fmaf(at[1], factor, denominator);
            acc = fmaf(at[2 + dim], factor, acc);
        }
    }
    const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    const float gated = __fmul_rn(__fmul_rn(acc, inverse),
                                  inkling_attn_sigmoid(gate[row_head * head_dim + dim]));
    out[row_head * head_dim + dim] = __float2bfloat16_rn(gated);
}

// ---------------------------------------------------------------------------------------------
// The tensor-core prompt walk: the shared prefill attention body (shared_attn_mma_sm121.cuh, the
// loop Dory's dory_attn_mma runs) with Inkling's arithmetic as its policy.
//
// Not the exact walks' bits. The scores and the value sum are MMA products in the tile's order:
// q and the softmax weights are f32 carried as a BF16 head and the BF16 of its remainder, the
// keys and values the cache's own BF16, the sums f32. Every other number is the exact walks':
//   - the query side is inkling_attn_query itself (inkling_attn_prepare writes what it returns),
//   - each key's clip gain and value scale are inkling_attn_key_scales', computed once a chunk,
//   - the relative bias of a key `distance` back is the dot of the row's prepared relative state
//     with the bank's column `distance` (zero at rel_extent and past it), here a 16 x 16 x 48
//     MMA a warp a tile within rel_extent of the block's rows, the f32 state split as q is,
//   - the sink opens the softmax (its maximum, and one in the denominator of one lane a quad),
//   - the output is acc / denominator times the sigmoid of the gate, as inkling_attn_one ends.
// The score is dot / sqrt(head_dim) times the key's gain plus the bias, and P V multiplies each
// weight by its key's value scale (the denominator sums the weight itself). A decode row takes
// the split-key walk above instead.
//
//   q_ready      [rows][heads][INKLING_MAX_HEAD_DIM] f32: the query after its gain, clip and tau
//   rel_ready    [rows][heads][INKLING_MMA_D_REL] f32: the clipped, tau-scaled relative state,
//                zero past d_rel
//   key_gains, value_scales   [kv_heads][scale_stride] f32, slot scale_first + i at i, over the
//                slots the chunk's rows see
// ---------------------------------------------------------------------------------------------

#include "shared_attn_mma_sm121.cuh"

// The relative state's width the bias MMA takes (one k-step; d_rel past it is refused), and a
// row's f32 stride in a warp's bias tile (47 distances used; 52 keeps its reads to two-way bank
// conflicts).
#define INKLING_MMA_D_REL 16
#define INKLING_MMA_REL_STRIDE 52
// The policy's shared bytes past the body's tiles: the key gains, the value scales, then a bias
// tile a warp.
#define INKLING_MMA_GAINS 0u
#define INKLING_MMA_SCALES (4u * SHARED_ATTN_KEYS)
#define INKLING_MMA_REL (8u * SHARED_ATTN_KEYS)
#define INKLING_MMA_REL_BYTES (4u * SHARED_ATTN_Q * INKLING_MMA_REL_STRIDE)

// grid (heads, rows), 32 threads: the query side of the tensor-core walk, inkling_attn_query's
// own result for every (row, head).
extern "C" __global__ void __launch_bounds__(32) inkling_attn_prepare(
    const float* __restrict__ q,
    const float* __restrict__ rel_proj,
    const float* __restrict__ q_scale,
    const float* __restrict__ rel_scale,
    float* __restrict__ q_ready,
    float* __restrict__ rel_ready,
    int heads,
    int head_dim,
    int rows,
    int d_rel,
    float qk_clip,
    float rel_clip,
    float tau,
    const float* __restrict__ tau_rows) {
    OPT_PDL_ENTRY();
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    if (head >= heads || row >= rows || head_dim <= 0 || head_dim > INKLING_MAX_HEAD_DIM ||
        d_rel < 0 || d_rel > INKLING_MMA_D_REL) {
        return;
    }
    const unsigned int lane = threadIdx.x & 31u;
    const float row_tau = tau_rows != nullptr ? tau_rows[row] : tau;
    const uint64_t offset = ((uint64_t)row * heads + head) * head_dim;
    float qv[INKLING_MAX_HEAD_DIM / 32];
    float rv;
    inkling_attn_query(q + offset, rel_proj + (uint64_t)row * heads * d_rel, q_scale, rel_scale,
                       head, head_dim, d_rel, qk_clip, rel_clip, row_tau, qv, rv);
#pragma unroll
    for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
        const int dim = (int)lane + 32 * step;
        if (dim < head_dim) {
            q_ready[offset + dim] = qv[step];
        }
    }
    if (lane < INKLING_MMA_D_REL) {
        rel_ready[((uint64_t)row * heads + head) * INKLING_MMA_D_REL + lane] =
            lane < (unsigned int)d_rel ? rv : 0.0f;
    }
}

__device__ __forceinline__ float inkling_mma_lds(uint32_t address) {
    float value;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(value) : "r"(address));
    return value;
}

// The bank's raw BF16 at (j, distance), zero outside [0, d_rel) x [0, rel_extent).
__device__ __forceinline__ uint32_t inkling_mma_bank(const __nv_bfloat16* __restrict__ bank, int j,
                                                     int distance, int d_rel, int rel_extent) {
    return j < d_rel && distance >= 0 && distance < rel_extent
               ? (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(bank) +
                                 (uint64_t)j * rel_extent + distance)
               : 0u;
}

struct InklingAttnPolicy {
    const float* __restrict__ rel_ready;
    const __nv_bfloat16* __restrict__ rel_bank;
    const float* __restrict__ sink;
    const float* __restrict__ key_gains;
    const float* __restrict__ value_scales;
    const float* __restrict__ gate;
    __nv_bfloat16* __restrict__ out;
    int heads;
    int d_rel;
    int rel_extent;
    int scale_stride;
    long long scale_first;
    // This thread's rows' relative state as the bias MMA's A fragment (head and remainder), and
    // whether the current tile holds a key within rel_extent of any of the block's rows.
    uint32_t rel_hi[4];
    uint32_t rel_lo[4];
    bool band;

    __device__ __forceinline__ void per_key(uint32_t dst, const float* __restrict__ table,
                                            int kv_head, int base, int last) const {
        const int i = threadIdx.x;
        if (i < SHARED_ATTN_KEYS) {
            const int key = base + i;
            const bool valid = key <= last;
            om_cp4(dst + 4u * i,
                   valid ? table + (uint64_t)kv_head * scale_stride + (key - scale_first) : table,
                   valid);
        }
    }
    __device__ __forceinline__ void keys(uint32_t extra, int kv_head, int base, int last) const {
        per_key(extra + INKLING_MMA_GAINS, key_gains, kv_head, base, last);
    }
    __device__ __forceinline__ void values(uint32_t extra, int kv_head, int base, int last) const {
        per_key(extra + INKLING_MMA_SCALES, value_scales, kv_head, base, last);
    }
    __device__ __forceinline__ void start(int head, int qbase, int count_q, int, int g, int t4) {
        float r[2][4];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = g + 8 * half;
            const float* state =
                rel_ready + ((uint64_t)(qbase + row) * heads + head) * INKLING_MMA_D_REL;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int j = 2 * t4 + (i & 1) + 8 * (i >> 1);
                r[half][i] = row < count_q ? state[j] : 0.0f;
            }
        }
        // a0: row g, k 2 t4 (+1); a1: row g + 8; a2: row g, k 2 t4 + 8 (+1); a3: row g + 8.
#pragma unroll
        for (int reg = 0; reg < 4; ++reg) {
            const float* pair = &r[reg & 1][2 * (reg >> 1)];
            const __nv_bfloat16 h0 = __float2bfloat16_rn(pair[0]);
            const __nv_bfloat16 h1 = __float2bfloat16_rn(pair[1]);
            rel_hi[reg] = ob_pack(h0, h1);
            rel_lo[reg] = ob_pack(__float2bfloat16_rn(pair[0] - __bfloat162float(h0)),
                                  __float2bfloat16_rn(pair[1] - __bfloat162float(h1)));
        }
        band = false;
    }
    __device__ __forceinline__ float open_max(int head, int qbase, int count_q, int row) const {
        return sink != nullptr && row < count_q ? sink[(uint64_t)(qbase + row) * heads + head]
                                                : -INFINITY;
    }
    __device__ __forceinline__ float open_sum(int, int, int count_q, int row, int t4) const {
        return sink != nullptr && row < count_q && t4 == 0 ? 1.0f : 0.0f;
    }
    // The bias of every (row, key) of the warp's 16 x 32 tile: distance d0 + d' for d' = row - key +
    // SHARED_ATTN_KEYS - 1 in [0, 47), so one 16 x 48 product a tile holds them all.
    __device__ __forceinline__ void tile(uint32_t extra, int warp, int pos0, int base, int g,
                                         int t4) {
        band = d_rel > 0 && pos0 - (base + SHARED_ATTN_KEYS - 1) < rel_extent;
        if (!band) {
            return;
        }
        const int d0 = pos0 - base - (SHARED_ATTN_KEYS - 1);
        const uint32_t at = extra + INKLING_MMA_REL + (uint32_t)warp * INKLING_MMA_REL_BYTES;
#pragma unroll
        for (int nt = 0; nt < (SHARED_ATTN_Q + SHARED_ATTN_KEYS) / 8; ++nt) {
            const int distance = d0 + 8 * nt + g;
            const uint32_t b0 =
                inkling_mma_bank(rel_bank, 2 * t4, distance, d_rel, rel_extent) |
                (inkling_mma_bank(rel_bank, 2 * t4 + 1, distance, d_rel, rel_extent) << 16);
            const uint32_t b1 =
                inkling_mma_bank(rel_bank, 2 * t4 + 8, distance, d_rel, rel_extent) |
                (inkling_mma_bank(rel_bank, 2 * t4 + 9, distance, d_rel, rel_extent) << 16);
            float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            ob_mma(c, rel_hi, b0, b1);
            ob_mma(c, rel_lo, b0, b1);
            const uint32_t column = (uint32_t)(8 * nt + 2 * t4);
            asm volatile("st.shared.v2.f32 [%0], {%1, %2};" ::"r"(
                             at + 4u * ((uint32_t)g * INKLING_MMA_REL_STRIDE + column)),
                         "f"(c[0]), "f"(c[1]));
            asm volatile("st.shared.v2.f32 [%0], {%1, %2};" ::"r"(
                             at + 4u * ((uint32_t)(g + 8) * INKLING_MMA_REL_STRIDE + column)),
                         "f"(c[2]), "f"(c[3]));
        }
        __syncwarp();
    }
    __device__ __forceinline__ float score(float scaled, uint32_t extra, int warp, int n, int e,
                                           int g, int t4) const {
        const int key = 8 * n + 2 * t4 + (e & 1);
        float logit = scaled * inkling_mma_lds(extra + INKLING_MMA_GAINS + 4u * key);
        if (band) {
            const int row = g + 8 * (e >> 1);
            logit += inkling_mma_lds(extra + INKLING_MMA_REL + (uint32_t)warp * INKLING_MMA_REL_BYTES +
                                     4u * (uint32_t)(row * INKLING_MMA_REL_STRIDE + row - key +
                                                     SHARED_ATTN_KEYS - 1));
        }
        return logit;
    }
    __device__ __forceinline__ float weight(float p, uint32_t extra, int n, int e, int t4) const {
        return p * inkling_mma_lds(extra + INKLING_MMA_SCALES + 4u * (8 * n + 2 * t4 + (e & 1)));
    }
    __device__ __forceinline__ void store(unsigned long long token, int head, int n, int t4,
                                          float v0, float v1, float denominator) const {
        const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
        const uint64_t at =
            (token * heads + head) * INKLING_MAX_HEAD_DIM + (uint64_t)(8 * n + 2 * t4);
        const float g0 = __fmul_rn(__fmul_rn(v0, inverse), inkling_attn_sigmoid(gate[at]));
        const float g1 = __fmul_rn(__fmul_rn(v1, inverse), inkling_attn_sigmoid(gate[at + 1]));
        *reinterpret_cast<__nv_bfloat162*>(out + at) = __floats2bfloat162_rn(g0, g1);
    }
};

// Grid (ceil(rows / SHARED_ATTN_Q), kv_heads), groups x 32 threads, inkling_attn_mma_shared()
// bytes of dynamic shared memory. head_dim is INKLING_MAX_HEAD_DIM; the host refuses the walk
// for any other width, a group past INKLING_MAX_GROUPS or d_rel past INKLING_MMA_D_REL.
extern "C" __global__ void __launch_bounds__(INKLING_MAX_GROUPS * 32, 1) inkling_attn_mma(
    const float* __restrict__ q_ready,
    const __nv_bfloat16* __restrict__ keys,
    const __nv_bfloat16* __restrict__ values,
    const float* __restrict__ rel_ready,
    const __nv_bfloat16* __restrict__ rel_bank,
    const float* __restrict__ sink,
    const float* __restrict__ key_gains,
    const float* __restrict__ value_scales,
    const float* __restrict__ gate,
    __nv_bfloat16* __restrict__ out,
    int heads,
    int kv_heads,
    int rows,
    int first_position,
    int window,
    int cache_rows,
    int rel_extent,
    int d_rel,
    long long scale_first,
    int scale_stride) {
    OPT_PDL_ENTRY();
    if (kv_heads <= 0 || heads % kv_heads != 0 || heads / kv_heads > INKLING_MAX_GROUPS ||
        (int)blockDim.x != heads / kv_heads * 32 || d_rel < 0 || d_rel > INKLING_MMA_D_REL) {
        return;
    }
    InklingAttnPolicy policy{rel_ready, rel_bank,  sink,    key_gains,   value_scales,
                             gate,      out,       heads,   d_rel,       rel_extent,
                             scale_stride, scale_first, {}, {}, false};
    shared_attn_mma_body<INKLING_MAX_HEAD_DIM>(q_ready, keys, values, first_position, rows, heads,
                                               kv_heads, window, heads * INKLING_MAX_HEAD_DIM,
                                               cache_rows, policy);
}

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

#include <cooperative_groups.h>
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

// What inkling_attn_query reads for one (head, row), in the lane layout: the query and its
// learned gain, the relative state and its scale (1 where there is none). A KV group loads all
// its heads before preparing any, so their reads are in flight together.
struct InklingQueryLoads {
    float q[INKLING_MAX_HEAD_DIM / 32];
    float gain[INKLING_MAX_HEAD_DIM / 32];
    float rel;
    float rel_gain;
};

__device__ __forceinline__ void inkling_attn_query_load(
    const float* __restrict__ q_row,
    const float* __restrict__ rel_proj,
    const float* __restrict__ q_scale,
    const float* __restrict__ rel_scale,
    int head,
    int head_dim,
    int d_rel,
    InklingQueryLoads& loads) {
    const unsigned int lane = threadIdx.x & 31u;
#pragma unroll
    for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
        const int dim = (int)lane + 32 * step;
        loads.gain[step] = dim < head_dim && q_scale != nullptr
                               ? q_scale[(uint64_t)head * head_dim + dim]
                               : 1.0f;
        loads.q[step] = dim < head_dim ? q_row[dim] : 0.0f;
    }
    const bool relative = lane < (unsigned int)d_rel;
    loads.rel = relative ? rel_proj[(uint64_t)head * d_rel + lane] : 0.0f;
    loads.rel_gain =
        relative && rel_scale != nullptr ? rel_scale[(uint64_t)head * d_rel + (int)lane] : 1.0f;
}

// The query side from its loads: the learned gain, the clip of the scaled vector, then tau,
// and the clipped relative state (one d_rel value a lane).
__device__ __forceinline__ void inkling_attn_query_finish(
    const InklingQueryLoads& loads,
    const float* __restrict__ rel_scale,
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
        qv[step] = dim < head_dim ? __fmul_rn(loads.q[step], loads.gain[step]) : 0.0f;
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
    rv = loads.rel;
    if (rel_scale != nullptr && lane < (unsigned int)d_rel) {
        rv = __fmul_rn(rv, loads.rel_gain);
    }
    const float r_norm = sqrtf(inkling_attn_warp_sum(__fmul_rn(rv, rv)));
    float r_gain = 1.0f;
    if (rel_clip > 0.0f && r_norm > rel_clip) {
        r_gain = rel_clip / r_norm;
    }
    rv = __fmul_rn(rv, __fmul_rn(r_gain, use_tau));
}

// The query side of one (head, row), in the body's lane layout.
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
    InklingQueryLoads loads;
    inkling_attn_query_load(q_row, rel_proj, q_scale, rel_scale, head, head_dim, d_rel, loads);
    inkling_attn_query_finish(loads, rel_scale, head_dim, d_rel, qk_clip, rel_clip, tau, qv, rv);
}

// A key's clip gain from its scaled values in the lane layout (the key after its learned gain,
// zero past head_dim): the norm a lane's dims in step order then the warp, against qk_clip.
// Every lane returns it.
__device__ __forceinline__ float inkling_attn_key_clip(const float (&kv)[INKLING_MAX_HEAD_DIM / 32],
                                                       float qk_clip) {
    float k_norm = 0.0f;
#pragma unroll
    for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
        k_norm = __fadd_rn(k_norm, __fmul_rn(kv[step], kv[step]));
    }
    k_norm = sqrtf(inkling_attn_warp_sum(k_norm));
    return qk_clip > 0.0f && k_norm > qk_clip ? qk_clip / k_norm : 1.0f;
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
#pragma unroll
    for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
        const int dim = (int)lane + 32 * step;
        // The key carries its own learned gain, and its clip is on the scaled vector too.
        const float gain = dim < head_dim && k_scale != nullptr
                               ? k_scale[(uint64_t)kv_head * head_dim + dim]
                               : 1.0f;
        kv[step] = dim < head_dim ? __fmul_rn(__bfloat162float(krow[dim]), gain) : 0.0f;
    }
    const float k_gain = inkling_attn_key_clip(kv, qk_clip);
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

// A KV group's prepared queries, inkling_attn_query for each of its heads in the lane layout
// (every head's loads, then each one's arithmetic) and zeros for the slots past `groups` (their
// sums are never stored). MG >= groups.
template <int MG>
__device__ __forceinline__ void inkling_group_queries(
    const float* __restrict__ q,
    const float* __restrict__ rel_proj,
    const float* __restrict__ q_scale,
    const float* __restrict__ rel_scale,
    int row,
    int heads,
    int head0,
    int head_dim,
    int groups,
    int d_rel,
    float qk_clip,
    float rel_clip,
    float tau,
    float (&qv)[MG][INKLING_MAX_HEAD_DIM / 32],
    float (&rv)[MG]) {
    const uint64_t row_offset = (uint64_t)row * heads * head_dim;
    InklingQueryLoads loads[MG];
#pragma unroll
    for (int g = 0; g < MG; ++g) {
        const int head = head0 + g;
        if (g < groups) {
            inkling_attn_query_load(q + row_offset + (uint64_t)head * head_dim,
                                    rel_proj + (uint64_t)row * heads * d_rel, q_scale, rel_scale,
                                    head, head_dim, d_rel, loads[g]);
        }
    }
#pragma unroll
    for (int g = 0; g < MG; ++g) {
        if (g < groups) {
            inkling_attn_query_finish(loads[g], rel_scale, head_dim, d_rel, qk_clip, rel_clip, tau,
                                      qv[g], rv[g]);
        } else {
#pragma unroll
            for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
                qv[g][step] = 0.0f;
            }
            rv[g] = 0.0f;
        }
    }
}

// A KV head's learned key gain for the lane's dims.
__device__ __forceinline__ void inkling_key_gain(const float* __restrict__ k_scale, int kv_head,
                                                 int head_dim,
                                                 float (&gain)[INKLING_MAX_HEAD_DIM / 32]) {
    const unsigned int lane = threadIdx.x & 31u;
#pragma unroll
    for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
        const int dim = (int)lane + 32 * step;
        gain[step] =
            dim < head_dim && k_scale != nullptr ? k_scale[(uint64_t)kv_head * head_dim + dim] : 1.0f;
    }
}

// Whether a key `distance` positions back carries a relative bias.
__device__ __forceinline__ bool inkling_attn_biased(int64_t distance, int rel_extent, int d_rel) {
    return d_rel > 0 && rel_extent > 0 && distance >= 0 && distance < rel_extent;
}

// The lane's BF16 of keys `index` .. `index + RUN - 1` of a row's range (those before `end`),
// the row's first visible slot `first`, and the lane's relative bank weight at each key's
// distance back from `position` (zero where it carries no bias).
__device__ __forceinline__ void inkling_score_load(
    const __nv_bfloat16* __restrict__ keys,
    const __nv_bfloat16* __restrict__ rel_bank,
    int64_t first,
    int64_t index,
    int64_t end,
    int position,
    int cache_rows,
    uint64_t kv_stride,
    uint64_t kv_offset,
    int head_dim,
    int rel_extent,
    int d_rel,
    __nv_bfloat16 (&loaded)[INKLING_SCORE_RUN][INKLING_MAX_HEAD_DIM / 32],
    float (&banked)[INKLING_SCORE_RUN]) {
    const unsigned int lane = threadIdx.x & 31u;
#pragma unroll
    for (int i = 0; i < INKLING_SCORE_RUN; ++i) {
        if (index + i < end) {
            const int64_t slot = first + index + i;
            const __nv_bfloat16* krow =
                keys + inkling_attn_cache_row(slot, cache_rows) * kv_stride + kv_offset;
#pragma unroll
            for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
                const int dim = (int)lane + 32 * step;
                loaded[i][step] = dim < head_dim ? krow[dim] : __float2bfloat16(0.0f);
            }
            const int64_t distance = (int64_t)position - slot;
            banked[i] = inkling_attn_biased(distance, rel_extent, d_rel) && lane < (unsigned int)d_rel
                            ? __bfloat162float(
                                  rel_bank[(uint64_t)lane * rel_extent + (uint64_t)distance])
                            : 0.0f;
        }
    }
}

// One key's score for every head of the group: the key's learned gain and its clip gain
// (`k_gain`, inkling_attn_key_scales'), the dot over sqrt(head_dim) and the relative bias of a key
// `distance` positions back (`bank` the lane's weight there, inkling_score_load's). Lane 2g
// returns head g's score.
template <int MG>
__device__ __forceinline__ float inkling_group_score(
    const __nv_bfloat16 (&key)[INKLING_MAX_HEAD_DIM / 32],
    const float (&gain)[INKLING_MAX_HEAD_DIM / 32],
    float k_gain,
    const float (&qv)[MG][INKLING_MAX_HEAD_DIM / 32],
    const float (&rv)[MG],
    float bank,
    int64_t distance,
    int rel_extent,
    int d_rel,
    int head_dim,
    float inv_dim) {
    const unsigned int lane = threadIdx.x & 31u;
    float kv[INKLING_MAX_HEAD_DIM / 32];
#pragma unroll
    for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
        const int dim = (int)lane + 32 * step;
        kv[step] = dim < head_dim ? __fmul_rn(__bfloat162float(key[step]), gain[step]) : 0.0f;
        kv[step] = __fmul_rn(kv[step], k_gain);
    }
    const bool biased = inkling_attn_biased(distance, rel_extent, d_rel);
    // Sums 0-7: a head's dot; 8-15: its relative bias.
    float sums[2 * INKLING_MAX_GROUPS];
#pragma unroll
    for (int g = 0; g < INKLING_MAX_GROUPS; ++g) {
        sums[g] = 0.0f;
        sums[INKLING_MAX_GROUPS + g] = 0.0f;
    }
#pragma unroll
    for (int g = 0; g < MG; ++g) {
        float dot = 0.0f;
#pragma unroll
        for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
            dot = __fadd_rn(dot, __fmul_rn(qv[g][step], kv[step]));
        }
        sums[g] = dot;
        sums[INKLING_MAX_GROUPS + g] = lane < (unsigned int)d_rel ? __fmul_rn(rv[g], bank) : 0.0f;
    }
    // Lane 2g holds head g's dot, lane 16 + 2g its bias.
    const float total = inkling_attn_warp_sums<2 * INKLING_MAX_GROUPS>(sums);
    const float bias = __shfl_sync(0xffffffffu, total, (lane + 16u) & 31u);
    float score = __fmul_rn(total, inv_dim);
    if (biased) {
        score = __fadd_rn(score, bias);
    }
    return score;
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
    float qv[INKLING_MAX_GROUPS][INKLING_MAX_HEAD_DIM / 32];
    float rv[INKLING_MAX_GROUPS];
    inkling_group_queries<INKLING_MAX_GROUPS>(q, rel_proj, q_scale, rel_scale, row, heads,
                                              kv_head * groups, head_dim, groups, d_rel, qk_clip,
                                              rel_clip, row_tau, qv, rv);
    const float inv_dim = 1.0f / sqrtf((float)head_dim);
    const uint64_t kv_stride = (uint64_t)kv_heads * head_dim;
    const uint64_t kv_offset = (uint64_t)kv_head * head_dim;
    const float* gains = key_gains + (uint64_t)kv_head * scale_stride - scale_first;
    float gain[INKLING_MAX_HEAD_DIM / 32];
    inkling_key_gain(k_scale, kv_head, head_dim, gain);
    // loaded[i][step]: this lane's BF16 of key `begin + i`, dim lane + 32 step.
    __nv_bfloat16 loaded[INKLING_SCORE_RUN][INKLING_MAX_HEAD_DIM / 32];
    float banked[INKLING_SCORE_RUN];
    inkling_score_load(keys, rel_bank, first, begin, end, position, cache_rows, kv_stride,
                       kv_offset, head_dim, rel_extent, d_rel, loaded, banked);
    for (;;) {
        __nv_bfloat16 run[INKLING_SCORE_RUN][INKLING_MAX_HEAD_DIM / 32];
        float bank[INKLING_SCORE_RUN];
#pragma unroll
        for (int i = 0; i < INKLING_SCORE_RUN; ++i) {
#pragma unroll
            for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
                run[i][step] = loaded[i][step];
            }
            bank[i] = banked[i];
        }
        const int64_t next = begin + stride;
        if (next < end) {
            inkling_score_load(keys, rel_bank, first, next, end, position, cache_rows, kv_stride,
                               kv_offset, head_dim, rel_extent, d_rel, loaded, banked);
        }
#pragma unroll
        for (int i = 0; i < INKLING_SCORE_RUN; ++i) {
            const int64_t index = begin + i;
            if (index < end) {
                const int64_t slot = first + index;
                const float score = inkling_group_score<INKLING_MAX_GROUPS>(
                    run[i], gain, gains[slot], qv, rv, bank[i], (int64_t)position - slot,
                    rel_extent, d_rel, head_dim, inv_dim);
                const int g = (int)(lane >> 1);
                if ((lane & 1u) == 0 && g < groups) {
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

// The largest of `value` across the first four warps of the block, for each of the first
// `count` entries. Every thread of the block keeps the barriers; only `active` ones (the first
// INKLING_PART_THREADS, warp-uniform) take part.
template <int K>
__device__ __forceinline__ void inkling_part_block_max(float (&value)[K], float (&shared)[4][K],
                                                       int count, bool active) {
    const unsigned int lane = threadIdx.x & 31u;
    const int warp = threadIdx.x >> 5;
    if (active) {
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
    }
    __syncthreads();
    if (active) {
#pragma unroll
        for (int g = 0; g < K; ++g) {
            if (g < count) {
                value[g] =
                    fmaxf(fmaxf(shared[0][g], shared[1][g]), fmaxf(shared[2][g], shared[3][g]));
            }
        }
    }
    __syncthreads();
}

// The sum of `value` across the same threads, the same way.
template <int K>
__device__ __forceinline__ void inkling_part_block_sum(float (&value)[K], float (&shared)[4][K],
                                                       int count, bool active) {
    const unsigned int lane = threadIdx.x & 31u;
    const int warp = threadIdx.x >> 5;
    if (active) {
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
    }
    __syncthreads();
    if (active) {
#pragma unroll
        for (int g = 0; g < K; ++g) {
            if (g < count) {
                value[g] = (shared[0][g] + shared[1][g]) + (shared[2][g] + shared[3][g]);
            }
        }
    }
    __syncthreads();
}

// What a run's partial needs in shared memory, for MG heads a group. The value sums have their
// own place (ALIASED false): nothing is still reading theirs while they are written.
template <int MG>
struct InklingPartShared {
    static constexpr bool ALIASED = false;
    float weights[MG][INKLING_PART_KEYS];
    float scales[INKLING_PART_KEYS];
    float block[4][MG];
    float sums[INKLING_PART_STRIDE][MG][INKLING_MAX_HEAD_DIM];
};

// A run's value rows in the cache: key i's row of the KV head (`values` at its first dim) at
// cache row slot + i, the ring's index for a sliding layer.
struct InklingCacheRows {
    const __nv_bfloat16* values;
    int64_t slot;
    int cache_rows;
    uint64_t kv_stride;
    __device__ __forceinline__ const __nv_bfloat16* row(int i) const {
        return values + inkling_attn_cache_row(slot + i, cache_rows) * kv_stride;
    }
};

// A run's value scales into shared memory: key i's at scale_row[i], 1 without the value norm.
template <class Shared>
__device__ __forceinline__ void inkling_part_scales(Shared& shared,
                                                    const float* __restrict__ scale_row,
                                                    int value_norm, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        shared.scales[i] = value_norm != 0 ? scale_row[i] : 1.0f;
    }
}

// A run's partial from the scores its first INKLING_PART_THREADS threads hold (keys t and
// t + INKLING_PART_THREADS of the run on thread t, read once for the maximum and the terms):
// the run's largest score a head, each key's term against it, their sum and the value sums
// weighted by the terms and the keys' value scales (already in shared.scales), written to `out`
// (the run's partial of the group's first head; a head's is `head_stride` floats on). The run
// is its n keys' value rows at rows.row(0) .. rows.row(n - 1). The value sums take QUARTERS
// blocks of INKLING_PART_THREADS threads, quarter u heads u, u + QUARTERS, ...; every head's
// sum is the same fmaf chain whichever thread runs it, and threads past the ones a phase uses
// keep the block's barriers only, so a block of any width computes the same bits. FULL says
// groups is MG, so the value sums take every head without a test a key (a head past groups would
// sum unwritten weights, though never store them).
template <int MG, int UNROLL, int QUARTERS, bool FULL = false, class Shared, class Rows>
__device__ __forceinline__ void inkling_part_run(
    Shared& shared,
    const float (&held)[MG][INKLING_PART_HELD],
    int n,
    int groups,
    const Rows& rows,
    int head_dim,
    float* __restrict__ out,
    uint64_t head_stride) {
    const bool active = threadIdx.x < INKLING_PART_THREADS;
    float largest[MG];
#pragma unroll
    for (int g = 0; g < MG; ++g) {
        largest[g] = -INFINITY;
        if (g < groups && active) {
#pragma unroll
            for (int k = 0; k < INKLING_PART_HELD; ++k) {
                const int i = threadIdx.x + k * INKLING_PART_THREADS;
                if (i < n) {
                    largest[g] = fmaxf(largest[g], held[g][k]);
                }
            }
        }
    }
    inkling_part_block_max(largest, shared.block, groups, active);
    float total[MG];
#pragma unroll
    for (int g = 0; g < MG; ++g) {
        total[g] = 0.0f;
        if (g < groups && active) {
            // An all -inf run (no key can carry weight) adds nothing; exp(-inf - -inf) is NaN.
            const bool live = largest[g] != -INFINITY;
#pragma unroll
            for (int k = 0; k < INKLING_PART_HELD; ++k) {
                const int i = threadIdx.x + k * INKLING_PART_THREADS;
                if (i < n) {
                    const float term = live ? expf(held[g][k] - largest[g]) : 0.0f;
                    shared.weights[g][i] = term;
                    total[g] += term;
                }
            }
        }
    }
    inkling_part_block_sum(total, shared.block, groups, active);

    // The weighted value sums: thread t of quarter u takes dims 8 (t % 16) .. + 7 of keys
    // t / 16, + 8, ... for its heads.
    constexpr int SHARE = (MG + QUARTERS - 1) / QUARTERS;
    const int quarter = threadIdx.x / INKLING_PART_THREADS;
    const int within = threadIdx.x % INKLING_PART_THREADS;
    const int lane = within % INKLING_PART_LANES;
    const int key_lane = within / INKLING_PART_LANES;
    const int dim0 = lane * INKLING_PART_DIMS;
    const bool owns = quarter < QUARTERS && dim0 < head_dim;
    float acc[SHARE][INKLING_PART_DIMS];
#pragma unroll
    for (int u = 0; u < SHARE; ++u) {
#pragma unroll
        for (int j = 0; j < INKLING_PART_DIMS; ++j) {
            acc[u][j] = 0.0f;
        }
    }
    if (owns) {
#pragma unroll(UNROLL)
        for (int i = key_lane; i < n; i += INKLING_PART_STRIDE) {
            const uint4 packed = *reinterpret_cast<const uint4*>(rows.row(i) + dim0);
            const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
            float value[INKLING_PART_DIMS];
#pragma unroll
            for (int pair = 0; pair < INKLING_PART_DIMS / 2; ++pair) {
                const float2 two = __bfloat1622float2(pairs[pair]);
                value[2 * pair] = two.x * shared.scales[i];
                value[2 * pair + 1] = two.y * shared.scales[i];
            }
#pragma unroll
            for (int u = 0; u < SHARE; ++u) {
                const int g = quarter + QUARTERS * u;
                if ((QUARTERS * (u + 1) <= MG || g < MG) && (FULL || g < groups)) {
                    const float weight = shared.weights[g][i];
#pragma unroll
                    for (int j = 0; j < INKLING_PART_DIMS; ++j) {
                        acc[u][j] = fmaf(weight, value[j], acc[u][j]);
                    }
                }
            }
        }
    }
    if constexpr (Shared::ALIASED) {
        // The sums take the place the value rows were read from.
        __syncthreads();
    }
    if (owns) {
#pragma unroll
        for (int u = 0; u < SHARE; ++u) {
            const int g = quarter + QUARTERS * u;
            if (g < groups) {
#pragma unroll
                for (int j = 0; j < INKLING_PART_DIMS; ++j) {
                    shared.sums[key_lane][g][dim0 + j] = acc[u][j];
                }
            }
        }
    }
    __syncthreads();
    for (int index = threadIdx.x; index < groups * head_dim; index += blockDim.x) {
        const int g = index / head_dim;
        const int dim = index - g * head_dim;
        float value = 0.0f;
#pragma unroll
        for (int k = 0; k < INKLING_PART_STRIDE; ++k) {
            value += shared.sums[k][g][dim];
        }
        out[(uint64_t)g * head_stride + 2 + dim] = value;
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int g = 0; g < MG; ++g) {
            if (g < groups) {
                out[(uint64_t)g * head_stride] = largest[g];
                out[(uint64_t)g * head_stride + 1] = total[g];
            }
        }
    }
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
    __shared__ InklingPartShared<INKLING_MAX_GROUPS> shared;
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
    float held[INKLING_MAX_GROUPS][INKLING_PART_HELD];
#pragma unroll
    for (int g = 0; g < INKLING_MAX_GROUPS; ++g) {
        const float* score_row =
            scores + ((uint64_t)row * heads + head0 + g) * pass_slots + (begin - pass_first);
#pragma unroll
        for (int k = 0; k < INKLING_PART_HELD; ++k) {
            const int i = threadIdx.x + k * INKLING_PART_THREADS;
            held[g][k] = g < groups && i < n ? score_row[i] : -INFINITY;
        }
    }
    const uint64_t run = (uint64_t)(pass_first / INKLING_PART_KEYS) + blockIdx.z;
    const uint64_t head_stride = (uint64_t)part_stride * (2 + (uint64_t)head_dim);
    inkling_part_scales(
        shared, value_scales + (uint64_t)kv_head * scale_stride - scale_first + first + begin,
        value_norm, n);
    const InklingCacheRows cache{values + (uint64_t)kv_head * head_dim, first + begin, cache_rows,
                                 (uint64_t)kv_heads * head_dim};
    inkling_part_run<INKLING_MAX_GROUPS, 4, 1>(
        shared, held, n, groups, cache, head_dim,
        parts + ((uint64_t)row * heads + head0) * head_stride + run * (2 + (uint64_t)head_dim),
        head_stride);
}

// The scored partial: inkling_attn_scores and inkling_attn_part of one run in one block, the
// scores kept in shared memory. A long row's runs are many more blocks than the SMs hold, and
// there the scores' round trip through DRAM and the second launch are what the split pays for:
// at 93k keys a full layer's two launches move its keys and values at 184 GB/s, this one at
// 205. Every number is the two kernels' own: the scores are inkling_group_score's (below, a key
// a quad), and inkling_part_run takes the run from them as inkling_attn_part reads them. A short
// row is a few blocks, so the host keeps the two launches there (the scores spread over many
// more blocks).
#define INKLING_SCORED_WARPS 8
#define INKLING_SCORED_GROUPS 6

// The scored partial scores a key a quad (four threads) where inkling_group_score takes a warp:
// thread u of the quad holds the key's dims u + 4m + 32 step (m < 8), which are lane u + 4m's
// dims of the lane layout, so its partial dot for lane u + 4m is that lane's chain of products in
// step order. The warp's tree pairs lanes differing in bit 4, then 3, 2, 1, 0 (offset 16 first);
// lanes u + 4m differ in m's bits 2, 1, 0, so the first three levels pair a thread's own partials
// (m with m + 4, then + 2, then + 1) and the last two pair the quad's threads (u ^ 2, then u ^ 1):
// every sum adds the same two operands at every node, the bias's too (its partials past d_rel the
// same zeros). A warp at a time takes the dot's 32 lanes' work for 8 keys instead of one, and the
// tree's 31 adds a sum instead of 5 shuffle levels of 16 sums.
//
// A warp stages its 8 keys' rows through shared memory (16-byte loads along each row) so the
// quad's threads read their own dims as words: word k of thread u holds dims 8k + u and
// 8k + 4 + u, so word (m >> 1) + 4 step, half m & 1, is dim u + 4m + 32 step. A thread's 16 words
// sit at a stride of INKLING_QUAD_PAD words, so a quarter warp's 16-byte reads (two keys' quads)
// and a store's 32 words fall in distinct banks.
#define INKLING_QUAD_PAD 20

// What the scored partial needs in shared memory: inkling_part_run's, with the keys' stage in the
// value sums' place (the scores are done with it, past a barrier, before a sum is written), and
// the quads' operands: thread u's lane u + 4m prepared query at queries[u][m][g][step], relative
// query at rel[u][m * 8 + g], key gain at dim_gains[u][m][step] (each u's row 4 words longer, for
// the banks).
template <int MG>
struct InklingScoredShared {
    static constexpr bool ALIASED = false;
    float weights[MG][INKLING_PART_KEYS];
    float scales[INKLING_PART_KEYS];
    float block[4][MG];
    union {
        float sums[INKLING_PART_STRIDE][MG][INKLING_MAX_HEAD_DIM];
        alignas(16) uint32_t stage[INKLING_SCORED_WARPS][8][4][INKLING_QUAD_PAD];
    };
    alignas(16) float queries[4][8 * MG * 4 + 4];
    alignas(16) float rel[4][8 * 8 + 4];
    alignas(16) float dim_gains[4][8 * 4 + 4];
};

// A warp's 8 keys from `base` of a run whose key i is at cache slot slot0 + i: lane t loads the
// 16-byte pieces t, t + 32, ... (key base + (t >> 4) + 2h, dims 8 (t & 15) ..), zeros for a key at
// n or past and for dims past head_dim.
__device__ __forceinline__ void inkling_quad_fetch(const __nv_bfloat16* __restrict__ keys,
                                                   int64_t slot0, int base, int n, int cache_rows,
                                                   uint64_t kv_stride, uint64_t kv_offset,
                                                   int head_dim, uint4 (&pieces)[4]) {
    const unsigned int lane = threadIdx.x & 31u;
    const int k = (int)(lane & 15u);
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const int i = base + (int)(lane >> 4) + 2 * h;
        pieces[h] = make_uint4(0u, 0u, 0u, 0u);
        if (i < n && 8 * k < head_dim) {
            pieces[h] = *reinterpret_cast<const uint4*>(
                keys + inkling_attn_cache_row(slot0 + i, cache_rows) * kv_stride + kv_offset +
                8 * k);
        }
    }
}

// The fetched pieces into the warp's stage: piece (r, k) holds dims 8k .. 8k + 7 of key r as BF16
// pairs x = (8k, 8k + 1), y, z, w, and thread u's word k pairs dims 8k + u and 8k + 4 + u.
__device__ __forceinline__ void inkling_quad_stage(uint32_t (&stage)[8][4][INKLING_QUAD_PAD],
                                                   const uint4 (&pieces)[4]) {
    const unsigned int lane = threadIdx.x & 31u;
    const int k = (int)(lane & 15u);
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const int r = (int)(lane >> 4) + 2 * h;
        stage[r][0][k] = __byte_perm(pieces[h].x, pieces[h].z, 0x5410);
        stage[r][1][k] = __byte_perm(pieces[h].x, pieces[h].z, 0x7632);
        stage[r][2][k] = __byte_perm(pieces[h].y, pieces[h].w, 0x5410);
        stage[r][3][k] = __byte_perm(pieces[h].y, pieces[h].w, 0x7632);
    }
}

// Word `i` (compile-time after unrolling) of four.
__device__ __forceinline__ uint32_t inkling_quad_word(const uint4& four, int i) {
    return i == 0 ? four.x : i == 1 ? four.y : i == 2 ? four.z : four.w;
}

// The warp tree's last three levels of one sum on a thread's eight partials (lanes u + 4m).
__device__ __forceinline__ float inkling_quad_tree(const float (&leaf)[8]) {
    const float a0 = __fadd_rn(leaf[0], leaf[4]);
    const float a2 = __fadd_rn(leaf[2], leaf[6]);
    const float a1 = __fadd_rn(leaf[1], leaf[5]);
    const float a3 = __fadd_rn(leaf[3], leaf[7]);
    return __fadd_rn(__fadd_rn(a0, a2), __fadd_rn(a1, a3));
}

// The tree's first two levels across the quad: every thread of it returns the sum.
__device__ __forceinline__ float inkling_quad_sum(float value) {
    value = __fadd_rn(value, __shfl_xor_sync(0xffffffffu, value, 2));
    return __fadd_rn(value, __shfl_xor_sync(0xffffffffu, value, 1));
}

// The scored partial's scores for keys [0, n) of the run whose key i is at cache slot slot0 + i,
// into shared.weights, once the queries, relative queries and key gains are in shared memory.
// `pieces` holds the warp's first keys (inkling_quad_fetch from base 8 * warp) on entry. Every
// lane of a warp runs the same iterations (a quad past n computes and stores nothing).
template <int MG>
__device__ __forceinline__ void inkling_quad_scores(
    InklingScoredShared<MG>& shared,
    uint4 (&pieces)[4],
    const __nv_bfloat16* __restrict__ keys,
    const __nv_bfloat16* __restrict__ rel_bank,
    const float* __restrict__ gains,
    int64_t slot0,
    int n,
    int position,
    int cache_rows,
    uint64_t kv_stride,
    uint64_t kv_offset,
    int head_dim,
    int groups,
    int rel_extent,
    int d_rel,
    float inv_dim) {
    const unsigned int lane = threadIdx.x & 31u;
    const int warp = (int)(threadIdx.x >> 5);
    const int u = (int)(lane & 3u);
    const int quad = (int)(lane >> 2);
    uint32_t(&stage)[8][4][INKLING_QUAD_PAD] = shared.stage[warp];
    for (int base = 8 * warp; base < n; base += 8 * INKLING_SCORED_WARPS) {
        __syncwarp();
        inkling_quad_stage(stage, pieces);
        __syncwarp();
        uint4 words[4];
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            words[step] = *reinterpret_cast<const uint4*>(&stage[quad][u][4 * step]);
        }
        if (base + 8 * INKLING_SCORED_WARPS < n) {
            inkling_quad_fetch(keys, slot0, base + 8 * INKLING_SCORED_WARPS, n, cache_rows,
                               kv_stride, kv_offset, head_dim, pieces);
        }
        const int i = base + quad;
        const bool valid = i < n;
        const int64_t slot = slot0 + i;
        const float k_gain = valid ? gains[slot] : 1.0f;
        // inkling_group_score's kv for thread u's dims; the stage's zeros past head_dim (gain 1)
        // give its zeros.
        float kv[8][4];
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            const float4 gain = *reinterpret_cast<const float4*>(&shared.dim_gains[u][4 * m]);
            const float gains4[4] = {gain.x, gain.y, gain.z, gain.w};
#pragma unroll
            for (int step = 0; step < 4; ++step) {
                const uint32_t word = inkling_quad_word(words[step], m >> 1);
                const float key = (m & 1) != 0 ? __uint_as_float(word & 0xffff0000u)
                                               : __uint_as_float(word << 16);
                kv[m][step] = __fmul_rn(__fmul_rn(key, gains4[step]), k_gain);
            }
        }
        float dots[MG];
#pragma unroll
        for (int g = 0; g < MG; ++g) {
            dots[g] = 0.0f;
            if (g < groups) {
                float leaf[8];
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    const float4 q4 =
                        *reinterpret_cast<const float4*>(&shared.queries[u][(m * MG + g) * 4]);
                    float dot = 0.0f;
                    dot = __fadd_rn(dot, __fmul_rn(q4.x, kv[m][0]));
                    dot = __fadd_rn(dot, __fmul_rn(q4.y, kv[m][1]));
                    dot = __fadd_rn(dot, __fmul_rn(q4.z, kv[m][2]));
                    dot = __fadd_rn(dot, __fmul_rn(q4.w, kv[m][3]));
                    leaf[m] = dot;
                }
                dots[g] = inkling_quad_sum(inkling_quad_tree(leaf));
            }
        }
        const int64_t distance = (int64_t)position - slot;
        const bool biased = valid && inkling_attn_biased(distance, rel_extent, d_rel);
        float biases[MG];
#pragma unroll
        for (int g = 0; g < MG; ++g) {
            biases[g] = 0.0f;
        }
        if (__any_sync(0xffffffffu, biased)) {
            float bank[8];
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                const int dim = u + 4 * m;
                bank[m] = biased && dim < d_rel
                              ? __bfloat162float(
                                    rel_bank[(uint64_t)dim * rel_extent + (uint64_t)distance])
                              : 0.0f;
            }
#pragma unroll
            for (int g = 0; g < MG; ++g) {
                if (g < groups) {
                    float leaf[8];
#pragma unroll
                    for (int m = 0; m < 8; ++m) {
                        leaf[m] = u + 4 * m < d_rel ? __fmul_rn(shared.rel[u][m * 8 + g], bank[m])
                                                    : 0.0f;
                    }
                    biases[g] = inkling_quad_sum(inkling_quad_tree(leaf));
                }
            }
        }
#pragma unroll
        for (int g = 0; g < MG; ++g) {
            if (valid && g < groups && (g & 3) == u) {
                float score = __fmul_rn(dots[g], inv_dim);
                if (biased) {
                    score = __fadd_rn(score, biases[g]);
                }
                shared.weights[g][i] = score;
            }
        }
    }
}

// grid (kv_heads, rows, ceil(pass span / INKLING_PART_KEYS)), 32 * INKLING_SCORED_WARPS threads;
// the arguments are inkling_attn_scores' and inkling_attn_part's. groups <=
// INKLING_SCORED_GROUPS (the host refuses wider groups the launch).
extern "C" __global__ void __launch_bounds__(32 * INKLING_SCORED_WARPS) inkling_attn_scored_part(
    const float* __restrict__ q,
    const __nv_bfloat16* __restrict__ keys,
    const __nv_bfloat16* __restrict__ values,
    const float* __restrict__ rel_proj,
    const __nv_bfloat16* __restrict__ rel_bank,
    const float* __restrict__ q_scale,
    const float* __restrict__ k_scale,
    const float* __restrict__ rel_scale,
    const float* __restrict__ key_gains,
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
    int rel_extent,
    int d_rel,
    float qk_clip,
    float rel_clip,
    float tau,
    const float* __restrict__ tau_rows,
    int value_norm,
    long long pass_first,
    int pass_slots,
    long long scale_first,
    int scale_stride,
    int part_stride) {
    constexpr int MG = INKLING_SCORED_GROUPS;
    static_assert(MG <= INKLING_SCORED_WARPS && MG <= 8, "a warp prepares each head's queries");
    OPT_PDL_ENTRY();
    __shared__ InklingScoredShared<MG> shared;
    const int kv_head = blockIdx.x;
    const int row = blockIdx.y;
    if (kv_head >= kv_heads || row >= rows || head_dim <= 0 || head_dim > INKLING_MAX_HEAD_DIM ||
        head_dim % INKLING_PART_DIMS != 0 || d_rel < 0 || d_rel > INKLING_MAX_D_REL ||
        groups <= 0 || groups > MG) {
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
    const unsigned int lane = threadIdx.x & 31u;
    const int warp = threadIdx.x >> 5;
    const uint64_t kv_stride = (uint64_t)kv_heads * head_dim;
    const uint64_t kv_offset = (uint64_t)kv_head * head_dim;

    // The scores, a warp's first keys in flight while the queries are prepared: warp g prepares
    // head g's (inkling_group_queries' arithmetic, a head at a time) and its lane u + 4m writes
    // them where quad thread u reads them.
    uint4 pieces[4];
    if (8 * warp < n) {
        inkling_quad_fetch(keys, first + begin, 8 * warp, n, cache_rows, kv_stride, kv_offset,
                           head_dim, pieces);
    }
    if (warp < groups) {
        const int head = head0 + warp;
        float qv[INKLING_MAX_HEAD_DIM / 32];
        float rv;
        inkling_attn_query(q + ((uint64_t)row * heads + head) * head_dim,
                           rel_proj + (uint64_t)row * heads * d_rel, q_scale, rel_scale, head,
                           head_dim, d_rel, qk_clip, rel_clip,
                           tau_rows != nullptr ? tau_rows[row] : tau, qv, rv);
        const int u = (int)(lane & 3u);
        const int m = (int)(lane >> 2);
        *reinterpret_cast<float4*>(&shared.queries[u][(m * MG + warp) * 4]) =
            make_float4(qv[0], qv[1], qv[2], qv[3]);
        shared.rel[u][m * 8 + warp] = rv;
    }
    for (int dim = (int)threadIdx.x; dim < INKLING_MAX_HEAD_DIM; dim += (int)blockDim.x) {
        // inkling_key_gain's.
        shared.dim_gains[dim & 3][4 * ((dim >> 2) & 7) + (dim >> 5)] =
            dim < head_dim && k_scale != nullptr ? k_scale[(uint64_t)kv_head * head_dim + dim]
                                                 : 1.0f;
    }
    __syncthreads();
    const float inv_dim = 1.0f / sqrtf((float)head_dim);
    inkling_quad_scores<MG>(shared, pieces, keys, rel_bank,
                            key_gains + (uint64_t)kv_head * scale_stride - scale_first,
                            first + begin, n, position, cache_rows, kv_stride, kv_offset, head_dim,
                            groups, rel_extent, d_rel, inv_dim);
    inkling_part_scales(
        shared, value_scales + (uint64_t)kv_head * scale_stride - scale_first + first + begin,
        value_norm, n);
    __syncthreads();

    // The partial, the scores read as inkling_attn_part reads them.
    float held[MG][INKLING_PART_HELD];
#pragma unroll
    for (int g = 0; g < MG; ++g) {
#pragma unroll
        for (int k = 0; k < INKLING_PART_HELD; ++k) {
            const int i = (int)threadIdx.x + k * INKLING_PART_THREADS;
            held[g][k] = g < groups && i < n ? shared.weights[g][i] : -INFINITY;
        }
    }
    const uint64_t run = (uint64_t)(pass_first / INKLING_PART_KEYS) + blockIdx.z;
    const uint64_t head_stride = (uint64_t)part_stride * (2 + (uint64_t)head_dim);
    const InklingCacheRows cache{values + kv_offset, first + begin, cache_rows, kv_stride};
    // The value sums take both halves of the block, a half every other head (the same bits as
    // one half taking them all); with one, half the block waits out the slowest phase.
    constexpr int QUARTERS = 32 * INKLING_SCORED_WARPS / INKLING_PART_THREADS;
    float* const out =
        parts + ((uint64_t)row * heads + head0) * head_stride + run * (2 + (uint64_t)head_dim);
    if (groups == MG) {
        inkling_part_run<MG, 8, QUARTERS, true>(shared, held, n, groups, cache, head_dim, out,
                                                head_stride);
    } else {
        inkling_part_run<MG, 8, QUARTERS>(shared, held, n, groups, cache, head_dim, out,
                                          head_stride);
    }
}

// The runs one merge block holds factors for in shared memory: a 1M-key row's 4,096 runs of
// INKLING_PART_KEYS. A row with more takes the one-thread-a-dim walk.
#define INKLING_MERGE_RUNS 4096

// The merge's arithmetic, shared by inkling_attn_merge and inkling_attn_window: the sink's term
// against the largest of the maxima (the denominator's first), a run's factor against it, and
// the output from the folded sums - acc / denominator times the gate's sigmoid, in BF16.
__device__ __forceinline__ float inkling_merge_open(const float* __restrict__ sink,
                                                    uint64_t row_head, float largest) {
    return sink != nullptr ? expf(sink[row_head] - largest) : 0.0f;
}

__device__ __forceinline__ float inkling_merge_factor(float top, float largest) {
    return top != -INFINITY ? expf(top - largest) : 0.0f;
}

__device__ __forceinline__ __nv_bfloat16 inkling_merge_out(float acc, float denominator,
                                                           float gate) {
    const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    return __float2bfloat16_rn(__fmul_rn(__fmul_rn(acc, inverse), inkling_attn_sigmoid(gate)));
}

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
            denominator = inkling_merge_open(sink, row_head, largest);
            for (int r = 0; r < runs; ++r) {
                const float* at = part + r * width;
                const float factor = inkling_merge_factor(at[0], largest);
                denominator = fmaf(at[1], factor, denominator);
                acc = fmaf(at[2 + dim], factor, acc);
            }
        }
        out[row_head * head_dim + dim] =
            inkling_merge_out(acc, denominator, gate[row_head * head_dim + dim]);
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
            factors[r] = inkling_merge_factor(part[r * width], largest);
        }
    }
    __syncthreads();
    if (dim >= head_dim) {
        return;
    }
    float denominator = 0.0f;
    float acc = 0.0f;
    if (largest != -INFINITY) {
        denominator = inkling_merge_open(sink, row_head, largest);
#pragma unroll 8
        for (int r = 0; r < runs; ++r) {
            const float* at = part + r * width;
            const float factor = factors[r];
            denominator = fmaf(at[1], factor, denominator);
            acc = fmaf(at[2 + dim], factor, acc);
        }
    }
    out[row_head * head_dim + dim] =
        inkling_merge_out(acc, denominator, gate[row_head * head_dim + dim]);
}

// Twin of inkling_attn_merge that can write BF16-rounded f32 directly (eliminating
// inkling_widen_f32 on full-attention layers) while leaving inkling_attn_merge's ABI intact.
extern "C" __global__ void __launch_bounds__(INKLING_MAX_HEAD_DIM) inkling_attn_merge_f32(
    const float* __restrict__ gate,
    const float* __restrict__ sink,
    const float* __restrict__ parts,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ out_f32,
    int heads,
    int head_dim,
    int rows,
    int first_position,
    int window,
    int part_stride,
    int run_keys) {
    OPT_PDL_ENTRY();
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int dim = threadIdx.x;
    if (head >= heads || row >= rows || run_keys == 0) {
        return;
    }
    const uint64_t row_head = (uint64_t)row * heads + head;
    const int position = first_position + row;
    const int64_t first = inkling_attn_first_slot(position, window);
    const int64_t count = (int64_t)position - first + 1;
    const int runs = run_keys < 0 ? -run_keys : (int)((count + run_keys - 1) / run_keys);
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
            denominator = inkling_merge_open(sink, row_head, largest);
            for (int r = 0; r < runs; ++r) {
                const float* at = part + r * width;
                const float factor = inkling_merge_factor(at[0], largest);
                denominator = fmaf(at[1], factor, denominator);
                acc = fmaf(at[2 + dim], factor, acc);
            }
        }
        const __nv_bfloat16 val =
            inkling_merge_out(acc, denominator, gate[row_head * head_dim + dim]);
        if (out != nullptr) {
            out[row_head * head_dim + dim] = val;
        }
        if (out_f32 != nullptr) {
            out_f32[row_head * head_dim + dim] = __bfloat162float(val);
        }
        return;
    }
    __shared__ float factors[INKLING_MERGE_RUNS];
    __shared__ float warp_largest[INKLING_MAX_HEAD_DIM / 32];
    float mine = -INFINITY;
    for (int r = threadIdx.x; r < runs; r += blockDim.x) {
        const float top = part[r * width];
        factors[r] = top;
        mine = fmaxf(mine, top);
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
            factors[r] = inkling_merge_factor(factors[r], largest);
        }
    }
    __syncthreads();
    if (dim >= head_dim) {
        return;
    }
    float denominator = 0.0f;
    float acc = 0.0f;
    if (largest != -INFINITY) {
        denominator = inkling_merge_open(sink, row_head, largest);
#pragma unroll 8
        for (int r = 0; r < runs; ++r) {
            const float* at = part + r * width;
            const float factor = factors[r];
            denominator = fmaf(at[1], factor, denominator);
            acc = fmaf(at[2 + dim], factor, acc);
        }
    }
    const __nv_bfloat16 val =
        inkling_merge_out(acc, denominator, gate[row_head * head_dim + dim]);
    if (out != nullptr) {
        out[row_head * head_dim + dim] = val;
    }
    if (out_f32 != nullptr) {
        out_f32[row_head * head_dim + dim] = __bfloat162float(val);
    }
}

// ---------------------------------------------------------------------------------------------
// A sliding layer's split-key walk in one launch. A sliding row sees at most its window, at
// most INKLING_WINDOW_RUNS runs of INKLING_PART_KEYS, so a cluster of that many blocks (one a
// run) holds a (KV head, row)'s whole walk, and the four launches of the split walk (the key
// scales, the scores, the partials, the merge) become one whose scores, clip gains, value
// scales and partials never leave the cluster. Every number is those kernels':
//   - a block stages its run's value rows in shared memory, in flight while it scores the keys,
//   - warp w scores the run's sub-runs of INKLING_SCORE_RUN keys w, w + WARPS, ... with
//     inkling_group_score, each key's clip gain from the key it loaded (inkling_attn_key_clip,
//     inkling_attn_key_scales' arithmetic),
//   - each value row's scale is inkling_attn_value_inverse of the staged row (again
//     inkling_attn_key_scales'),
//   - inkling_part_run takes the run's partial into shared memory, and
//   - the cluster's blocks merge the partials over distributed shared memory in run order,
//     inkling_attn_merge's arithmetic, each block a quarter of the group's outputs.
// A decode step's or a verify's rows each take a cluster; a row past the window's first runs
// leaves the later blocks of its cluster empty.
// ---------------------------------------------------------------------------------------------

#define INKLING_WINDOW_RUNS 4
#define INKLING_WINDOW_WARPS 16
#define INKLING_WINDOW_GROUPS 6

// The window's shared memory: the run's staged value rows, whose place the value sums take once
// they are read (ALIASED), what inkling_part_run needs besides, and the run's partial the
// cluster merges.
template <int MG>
struct InklingWindowShared {
    static constexpr bool ALIASED = true;
    union {
        uint4 staged[INKLING_PART_KEYS][INKLING_MAX_HEAD_DIM / 8];
        float sums[INKLING_PART_STRIDE][MG][INKLING_MAX_HEAD_DIM];
    };
    float weights[MG][INKLING_PART_KEYS];
    float scales[INKLING_PART_KEYS];
    float block[4][MG];
    float part[MG][2 + INKLING_MAX_HEAD_DIM];
};

// device.rs launches the window with WINDOW_SHARED dynamic bytes.
static_assert(sizeof(InklingWindowShared<INKLING_WINDOW_GROUPS>) == 75920,
              "WINDOW_SHARED in crates/sparkflash-model-inkling2/src/device.rs");

// A run's staged value rows: key i's at staged[i].
struct InklingStagedRows {
    const uint4* staged;
    __device__ __forceinline__ const __nv_bfloat16* row(int i) const {
        return reinterpret_cast<const __nv_bfloat16*>(staged + i * (INKLING_MAX_HEAD_DIM / 8));
    }
};

// grid (kv_heads, rows, INKLING_WINDOW_RUNS), clusters of the z blocks, 32 * INKLING_WINDOW_WARPS
// threads, sizeof(InklingWindowShared<INKLING_WINDOW_GROUPS>) dynamic shared bytes; the
// arguments are the split walk's. head_dim INKLING_MAX_HEAD_DIM, groups <= INKLING_WINDOW_GROUPS
// and window <= INKLING_WINDOW_RUNS * INKLING_PART_KEYS (the host refuses the launch otherwise).
extern "C" __global__ void __cluster_dims__(1, 1, INKLING_WINDOW_RUNS)
    __launch_bounds__(32 * INKLING_WINDOW_WARPS, 1) inkling_attn_window(
        const float* __restrict__ q,
        const __nv_bfloat16* __restrict__ keys,
        const __nv_bfloat16* __restrict__ values,
        const float* __restrict__ rel_proj,
        const __nv_bfloat16* __restrict__ rel_bank,
        const float* __restrict__ q_scale,
        const float* __restrict__ k_scale,
        const float* __restrict__ rel_scale,
        const float* __restrict__ gate,
        const float* __restrict__ sink,
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
        const float* __restrict__ tau_rows,
        int value_norm,
        float eps) {
    constexpr int MG = INKLING_WINDOW_GROUPS;
    constexpr int THREADS = 32 * INKLING_WINDOW_WARPS;
    constexpr int CHUNKS = INKLING_MAX_HEAD_DIM / 8;
    OPT_PDL_ENTRY();
    extern __shared__ __align__(16) unsigned char inkling_window_raw[];
    InklingWindowShared<MG>& shared = *reinterpret_cast<InklingWindowShared<MG>*>(inkling_window_raw);
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    const int kv_head = blockIdx.x;
    const int row = blockIdx.y;
    const int run = blockIdx.z;
    // Every block of a cluster takes the same branch: the cluster spans z alone.
    if (kv_head >= kv_heads || row >= rows || head_dim != INKLING_MAX_HEAD_DIM || d_rel < 0 ||
        d_rel > INKLING_MAX_D_REL || groups <= 0 || groups > MG || window <= 0 ||
        window > INKLING_WINDOW_RUNS * INKLING_PART_KEYS || gridDim.z != INKLING_WINDOW_RUNS) {
        return;
    }
    const int position = first_position + row;
    const int64_t first = inkling_attn_first_slot(position, window);
    const int count = (int)((int64_t)position - first + 1);
    const int runs = (count + INKLING_PART_KEYS - 1) / INKLING_PART_KEYS;
    const int begin = run * INKLING_PART_KEYS;
    const int n = count <= begin ? 0 : min(count - begin, INKLING_PART_KEYS);
    const int head0 = kv_head * groups;
    const unsigned int lane = threadIdx.x & 31u;
    const int warp = threadIdx.x >> 5;
    const uint64_t kv_stride = (uint64_t)kv_heads * head_dim;
    const uint64_t kv_offset = (uint64_t)kv_head * head_dim;
    const InklingStagedRows staged{&shared.staged[0][0]};

    if (n > 0) {
        // A warp's first keys, then the run's value rows into shared memory, both in flight while
        // the queries are prepared.
        constexpr int STRIDE = INKLING_WINDOW_WARPS * INKLING_SCORE_RUN;
        int at = warp * INKLING_SCORE_RUN;
        __nv_bfloat16 loaded[INKLING_SCORE_RUN][INKLING_MAX_HEAD_DIM / 32];
        float banked[INKLING_SCORE_RUN];
        if (at < n) {
            inkling_score_load(keys, rel_bank, first + begin, at, n, position, cache_rows,
                               kv_stride, kv_offset, head_dim, rel_extent, d_rel, loaded, banked);
        }
        for (int c = threadIdx.x; c < n * CHUNKS; c += THREADS) {
            const int i = c / CHUNKS;
            const int chunk = c - i * CHUNKS;
            const __nv_bfloat16* source = values +
                                          inkling_attn_cache_row(first + begin + i, cache_rows) *
                                              kv_stride +
                                          kv_offset + chunk * 8;
            const uint32_t target = (uint32_t)__cvta_generic_to_shared(&shared.staged[i][chunk]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(target), "l"(source));
        }
        asm volatile("cp.async.commit_group;");
        float qv[MG][INKLING_MAX_HEAD_DIM / 32];
        float rv[MG];
        inkling_group_queries<MG>(q, rel_proj, q_scale, rel_scale, row, heads, head0, head_dim,
                                  groups, d_rel, qk_clip, rel_clip,
                                  tau_rows != nullptr ? tau_rows[row] : tau, qv, rv);
        const float inv_dim = 1.0f / sqrtf((float)head_dim);
        float gain[INKLING_MAX_HEAD_DIM / 32];
        inkling_key_gain(k_scale, kv_head, head_dim, gain);
        for (; at < n; at += STRIDE) {
            __nv_bfloat16 sub[INKLING_SCORE_RUN][INKLING_MAX_HEAD_DIM / 32];
            float bank[INKLING_SCORE_RUN];
#pragma unroll
            for (int i = 0; i < INKLING_SCORE_RUN; ++i) {
#pragma unroll
                for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
                    sub[i][step] = loaded[i][step];
                }
                bank[i] = banked[i];
            }
            if (at + STRIDE < n) {
                inkling_score_load(keys, rel_bank, first + begin, at + STRIDE, n, position,
                                   cache_rows, kv_stride, kv_offset, head_dim, rel_extent, d_rel,
                                   loaded, banked);
            }
#pragma unroll
            for (int i = 0; i < INKLING_SCORE_RUN; ++i) {
                const int index = at + i;
                if (index < n) {
                    float kv[INKLING_MAX_HEAD_DIM / 32];
#pragma unroll
                    for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
                        kv[step] = __fmul_rn(__bfloat162float(sub[i][step]), gain[step]);
                    }
                    const int64_t slot = first + begin + index;
                    const float score = inkling_group_score<MG>(
                        sub[i], gain, inkling_attn_key_clip(kv, qk_clip), qv, rv, bank[i],
                        (int64_t)position - slot, rel_extent, d_rel, head_dim, inv_dim);
                    const int g = (int)(lane >> 1);
                    if ((lane & 1u) == 0 && g < groups) {
                        shared.weights[g][index] = score;
                    }
                }
            }
        }
        asm volatile("cp.async.wait_all;" ::: "memory");
    }
    __syncthreads();

    // Each value row's scale from its staged row, four rows a warp at once.
    if (value_norm != 0) {
        for (int base = warp; base < n; base += 4 * INKLING_WINDOW_WARPS) {
            float inverse[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int i = base + u * INKLING_WINDOW_WARPS;
                inverse[u] = i < n ? inkling_attn_value_inverse(staged.row(i), head_dim, eps) : 1.0f;
            }
            if (lane == 0) {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int i = base + u * INKLING_WINDOW_WARPS;
                    if (i < n) {
                        shared.scales[i] = inverse[u];
                    }
                }
            }
        }
    } else {
        for (int i = threadIdx.x; i < n; i += THREADS) {
            shared.scales[i] = 1.0f;
        }
    }

    // The run's partial into shared memory, the scores read as inkling_attn_part reads them.
    float held[MG][INKLING_PART_HELD];
#pragma unroll
    for (int g = 0; g < MG; ++g) {
#pragma unroll
        for (int k = 0; k < INKLING_PART_HELD; ++k) {
            const int i = (int)threadIdx.x + k * INKLING_PART_THREADS;
            held[g][k] = g < groups && i < n ? shared.weights[g][i] : -INFINITY;
        }
    }
    if (groups == MG) {
        inkling_part_run<MG, 4, THREADS / INKLING_PART_THREADS, true>(
            shared, held, n, groups, staged, head_dim, &shared.part[0][0], 2 + INKLING_MAX_HEAD_DIM);
    } else {
        inkling_part_run<MG, 4, THREADS / INKLING_PART_THREADS>(
            shared, held, n, groups, staged, head_dim, &shared.part[0][0], 2 + INKLING_MAX_HEAD_DIM);
    }
    cluster.sync();

    // The merge of the cluster's partials in run order, block r taking outputs r, r + RUNS, ...
    const float* partials[INKLING_WINDOW_RUNS];
#pragma unroll
    for (int r = 0; r < INKLING_WINDOW_RUNS; ++r) {
        partials[r] = cluster.map_shared_rank(&shared.part[0][0], r);
    }
    constexpr int WIDTH = 2 + INKLING_MAX_HEAD_DIM;
    for (int index = run + INKLING_WINDOW_RUNS * (int)threadIdx.x; index < groups * head_dim;
         index += INKLING_WINDOW_RUNS * THREADS) {
        const int g = index / head_dim;
        const int dim = index - g * head_dim;
        const uint64_t row_head = (uint64_t)row * heads + head0 + g;
        float tops[INKLING_WINDOW_RUNS];
        float largest = sink != nullptr ? sink[row_head] : -INFINITY;
#pragma unroll
        for (int r = 0; r < INKLING_WINDOW_RUNS; ++r) {
            tops[r] = r < runs ? partials[r][g * WIDTH] : -INFINITY;
            if (r < runs) {
                largest = fmaxf(largest, tops[r]);
            }
        }
        float denominator = 0.0f;
        float acc = 0.0f;
        if (largest != -INFINITY) {
            denominator = inkling_merge_open(sink, row_head, largest);
#pragma unroll
            for (int r = 0; r < INKLING_WINDOW_RUNS; ++r) {
                if (r < runs) {
                    const float factor = inkling_merge_factor(tops[r], largest);
                    denominator = fmaf(partials[r][g * WIDTH + 1], factor, denominator);
                    acc = fmaf(partials[r][g * WIDTH + 2 + dim], factor, acc);
                }
            }
        }
        out[row_head * head_dim + dim] =
            inkling_merge_out(acc, denominator, gate[row_head * head_dim + dim]);
    }
    // No block leaves while another still reads its partial.
    cluster.sync();
}

// Single-run non-cluster sliding-window attention specialised for groups == 3 (Inkling-2-Mini)
// when count <= INKLING_PART_KEYS (256): eliminates 3 empty cluster blocks per (kv_head, row) and
// cluster.sync(), halves MG from 6 to 3, and writes f32 directly (eliminating inkling_widen_f32).
#define INKLING_WINDOW_G3 3
#define INKLING_WINDOW_G3_R1_WARPS 16
static_assert(sizeof(InklingWindowShared<INKLING_WINDOW_G3>) == 71248,
              "WINDOW_G3_SHARED in crates/sparkflash-model-inkling2/src/device.rs");

extern "C" __global__ void __launch_bounds__(32 * INKLING_WINDOW_G3_R1_WARPS, 1)
    inkling_attn_window_g3_r1(
        const float* __restrict__ q,
        const __nv_bfloat16* __restrict__ keys,
        const __nv_bfloat16* __restrict__ values,
        const float* __restrict__ rel_proj,
        const __nv_bfloat16* __restrict__ rel_bank,
        const float* __restrict__ q_scale,
        const float* __restrict__ k_scale,
        const float* __restrict__ rel_scale,
        const float* __restrict__ gate,
        const float* __restrict__ sink,
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ out_f32,
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
        int value_norm,
        float eps) {
    constexpr int MG = INKLING_WINDOW_G3;
    constexpr int WARPS = INKLING_WINDOW_G3_R1_WARPS;
    constexpr int THREADS = 32 * WARPS;
    constexpr int CHUNKS = INKLING_MAX_HEAD_DIM / 8;
    OPT_PDL_ENTRY();
    extern __shared__ __align__(16) unsigned char inkling_window_g3_raw[];
    InklingWindowShared<MG>& shared =
        *reinterpret_cast<InklingWindowShared<MG>*>(inkling_window_g3_raw);
    const int kv_head = blockIdx.x;
    const int row = blockIdx.y;
    if (kv_head >= kv_heads || row >= rows || head_dim != INKLING_MAX_HEAD_DIM || d_rel < 0 ||
        d_rel > INKLING_MAX_D_REL || groups != MG || window <= 0) {
        return;
    }
    const int position = first_position + row;
    const int64_t first = inkling_attn_first_slot(position, window);
    const int n = (int)((int64_t)position - first + 1);
    if (n <= 0 || n > INKLING_PART_KEYS) {
        return;
    }
    const int head0 = kv_head * MG;
    const unsigned int lane = threadIdx.x & 31u;
    const int warp = threadIdx.x >> 5;
    const uint64_t kv_stride = (uint64_t)kv_heads * head_dim;
    const uint64_t kv_offset = (uint64_t)kv_head * head_dim;
    const InklingStagedRows staged{&shared.staged[0][0]};

    constexpr int STRIDE = WARPS * INKLING_SCORE_RUN;
    int at = warp * INKLING_SCORE_RUN;
    __nv_bfloat16 loaded[INKLING_SCORE_RUN][INKLING_MAX_HEAD_DIM / 32];
    float banked[INKLING_SCORE_RUN];
    if (at < n) {
        inkling_score_load(keys, rel_bank, first, at, n, position, cache_rows, kv_stride,
                           kv_offset, head_dim, rel_extent, d_rel, loaded, banked);
    }
    for (int c = threadIdx.x; c < n * CHUNKS; c += THREADS) {
        const int i = c / CHUNKS;
        const int chunk = c - i * CHUNKS;
        const __nv_bfloat16* source =
            values + inkling_attn_cache_row(first + i, cache_rows) * kv_stride + kv_offset +
            chunk * 8;
        const uint32_t target = (uint32_t)__cvta_generic_to_shared(&shared.staged[i][chunk]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(target), "l"(source));
    }
    asm volatile("cp.async.commit_group;");
    float qv[MG][INKLING_MAX_HEAD_DIM / 32];
    float rv[MG];
    inkling_group_queries<MG>(q, rel_proj, q_scale, rel_scale, row, heads, head0, head_dim, MG,
                              d_rel, qk_clip, rel_clip, tau_rows != nullptr ? tau_rows[row] : tau,
                              qv, rv);
    const float inv_dim = 1.0f / sqrtf((float)head_dim);
    float gain[INKLING_MAX_HEAD_DIM / 32];
    inkling_key_gain(k_scale, kv_head, head_dim, gain);
    for (; at < n; at += STRIDE) {
        __nv_bfloat16 sub[INKLING_SCORE_RUN][INKLING_MAX_HEAD_DIM / 32];
        float bank[INKLING_SCORE_RUN];
#pragma unroll
        for (int i = 0; i < INKLING_SCORE_RUN; ++i) {
#pragma unroll
            for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
                sub[i][step] = loaded[i][step];
            }
            bank[i] = banked[i];
        }
        if (at + STRIDE < n) {
            inkling_score_load(keys, rel_bank, first, at + STRIDE, n, position, cache_rows,
                               kv_stride, kv_offset, head_dim, rel_extent, d_rel, loaded, banked);
        }
#pragma unroll
        for (int i = 0; i < INKLING_SCORE_RUN; ++i) {
            const int index = at + i;
            if (index < n) {
                float kv[INKLING_MAX_HEAD_DIM / 32];
#pragma unroll
                for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
                    kv[step] = __fmul_rn(__bfloat162float(sub[i][step]), gain[step]);
                }
                const int64_t slot = first + index;
                const float score = inkling_group_score<MG>(
                    sub[i], gain, inkling_attn_key_clip(kv, qk_clip), qv, rv, bank[i],
                    (int64_t)position - slot, rel_extent, d_rel, head_dim, inv_dim);
                const int g = (int)(lane >> 1);
                if ((lane & 1u) == 0 && g < MG) {
                    shared.weights[g][index] = score;
                }
            }
        }
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();

    if (value_norm != 0) {
        for (int base = warp; base < n; base += 4 * WARPS) {
            float inverse[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int i = base + u * WARPS;
                inverse[u] = i < n ? inkling_attn_value_inverse(staged.row(i), head_dim, eps) : 1.0f;
            }
            if (lane == 0) {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int i = base + u * WARPS;
                    if (i < n) {
                        shared.scales[i] = inverse[u];
                    }
                }
            }
        }
    } else {
        for (int i = threadIdx.x; i < n; i += THREADS) {
            shared.scales[i] = 1.0f;
        }
    }

    float held[MG][INKLING_PART_HELD];
#pragma unroll
    for (int g = 0; g < MG; ++g) {
#pragma unroll
        for (int k = 0; k < INKLING_PART_HELD; ++k) {
            const int i = (int)threadIdx.x + k * INKLING_PART_THREADS;
            held[g][k] = i < n ? shared.weights[g][i] : -INFINITY;
        }
    }
    inkling_part_run<MG, 4, THREADS / INKLING_PART_THREADS, true>(
        shared, held, n, MG, staged, head_dim, &shared.part[0][0], 2 + INKLING_MAX_HEAD_DIM);
    __syncthreads();

    for (int index = (int)threadIdx.x; index < MG * head_dim; index += THREADS) {
        const int g = index / head_dim;
        const int dim = index - g * head_dim;
        const uint64_t row_head = (uint64_t)row * heads + head0 + g;
        const float top = shared.part[g][0];
        float largest = sink != nullptr ? sink[row_head] : -INFINITY;
        largest = fmaxf(largest, top);
        float denominator = 0.0f;
        float acc = 0.0f;
        if (largest != -INFINITY) {
            denominator = inkling_merge_open(sink, row_head, largest);
            const float factor = inkling_merge_factor(top, largest);
            denominator = fmaf(shared.part[g][1], factor, denominator);
            acc = fmaf(shared.part[g][2 + dim], factor, acc);
        }
        const __nv_bfloat16 val =
            inkling_merge_out(acc, denominator, gate[row_head * head_dim + dim]);
        if (out != nullptr) {
            out[row_head * head_dim + dim] = val;
        }
        if (out_f32 != nullptr) {
            out_f32[row_head * head_dim + dim] = __bfloat162float(val);
        }
    }
}

// 4-run cluster sliding-window attention specialised for groups == 3 (Inkling-2-Mini) for
// count > 256: halves MG from 6 to 3 and writes f32 directly (eliminating inkling_widen_f32).
extern "C" __global__ void __cluster_dims__(1, 1, INKLING_WINDOW_RUNS)
    __launch_bounds__(32 * INKLING_WINDOW_WARPS, 1) inkling_attn_window_g3(
        const float* __restrict__ q,
        const __nv_bfloat16* __restrict__ keys,
        const __nv_bfloat16* __restrict__ values,
        const float* __restrict__ rel_proj,
        const __nv_bfloat16* __restrict__ rel_bank,
        const float* __restrict__ q_scale,
        const float* __restrict__ k_scale,
        const float* __restrict__ rel_scale,
        const float* __restrict__ gate,
        const float* __restrict__ sink,
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ out_f32,
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
        int value_norm,
        float eps) {
    constexpr int MG = INKLING_WINDOW_G3;
    constexpr int THREADS = 32 * INKLING_WINDOW_WARPS;
    constexpr int CHUNKS = INKLING_MAX_HEAD_DIM / 8;
    OPT_PDL_ENTRY();
    extern __shared__ __align__(16) unsigned char inkling_window_raw[];
    InklingWindowShared<MG>& shared = *reinterpret_cast<InklingWindowShared<MG>*>(inkling_window_raw);
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    const int kv_head = blockIdx.x;
    const int row = blockIdx.y;
    const int run = blockIdx.z;
    if (kv_head >= kv_heads || row >= rows || head_dim != INKLING_MAX_HEAD_DIM || d_rel < 0 ||
        d_rel > INKLING_MAX_D_REL || groups != MG || window <= 0 ||
        window > INKLING_WINDOW_RUNS * INKLING_PART_KEYS || gridDim.z != INKLING_WINDOW_RUNS) {
        return;
    }
    const int position = first_position + row;
    const int64_t first = inkling_attn_first_slot(position, window);
    const int count = (int)((int64_t)position - first + 1);
    const int runs = (count + INKLING_PART_KEYS - 1) / INKLING_PART_KEYS;
    const int begin = run * INKLING_PART_KEYS;
    const int n = count <= begin ? 0 : min(count - begin, INKLING_PART_KEYS);
    const int head0 = kv_head * MG;
    const unsigned int lane = threadIdx.x & 31u;
    const int warp = threadIdx.x >> 5;
    const uint64_t kv_stride = (uint64_t)kv_heads * head_dim;
    const uint64_t kv_offset = (uint64_t)kv_head * head_dim;
    const InklingStagedRows staged{&shared.staged[0][0]};

    if (n > 0) {
        constexpr int STRIDE = INKLING_WINDOW_WARPS * INKLING_SCORE_RUN;
        int at = warp * INKLING_SCORE_RUN;
        __nv_bfloat16 loaded[INKLING_SCORE_RUN][INKLING_MAX_HEAD_DIM / 32];
        float banked[INKLING_SCORE_RUN];
        if (at < n) {
            inkling_score_load(keys, rel_bank, first + begin, at, n, position, cache_rows,
                               kv_stride, kv_offset, head_dim, rel_extent, d_rel, loaded, banked);
        }
        for (int c = threadIdx.x; c < n * CHUNKS; c += THREADS) {
            const int i = c / CHUNKS;
            const int chunk = c - i * CHUNKS;
            const __nv_bfloat16* source = values +
                                          inkling_attn_cache_row(first + begin + i, cache_rows) *
                                              kv_stride +
                                          kv_offset + chunk * 8;
            const uint32_t target = (uint32_t)__cvta_generic_to_shared(&shared.staged[i][chunk]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(target), "l"(source));
        }
        asm volatile("cp.async.commit_group;");
        float qv[MG][INKLING_MAX_HEAD_DIM / 32];
        float rv[MG];
        inkling_group_queries<MG>(q, rel_proj, q_scale, rel_scale, row, heads, head0, head_dim,
                                  MG, d_rel, qk_clip, rel_clip,
                                  tau_rows != nullptr ? tau_rows[row] : tau, qv, rv);
        const float inv_dim = 1.0f / sqrtf((float)head_dim);
        float gain[INKLING_MAX_HEAD_DIM / 32];
        inkling_key_gain(k_scale, kv_head, head_dim, gain);
        for (; at < n; at += STRIDE) {
            __nv_bfloat16 sub[INKLING_SCORE_RUN][INKLING_MAX_HEAD_DIM / 32];
            float bank[INKLING_SCORE_RUN];
#pragma unroll
            for (int i = 0; i < INKLING_SCORE_RUN; ++i) {
#pragma unroll
                for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
                    sub[i][step] = loaded[i][step];
                }
                bank[i] = banked[i];
            }
            if (at + STRIDE < n) {
                inkling_score_load(keys, rel_bank, first + begin, at + STRIDE, n, position,
                                   cache_rows, kv_stride, kv_offset, head_dim, rel_extent, d_rel,
                                   loaded, banked);
            }
#pragma unroll
            for (int i = 0; i < INKLING_SCORE_RUN; ++i) {
                const int index = at + i;
                if (index < n) {
                    float kv[INKLING_MAX_HEAD_DIM / 32];
#pragma unroll
                    for (int step = 0; step < INKLING_MAX_HEAD_DIM / 32; ++step) {
                        kv[step] = __fmul_rn(__bfloat162float(sub[i][step]), gain[step]);
                    }
                    const int64_t slot = first + begin + index;
                    const float score = inkling_group_score<MG>(
                        sub[i], gain, inkling_attn_key_clip(kv, qk_clip), qv, rv, bank[i],
                        (int64_t)position - slot, rel_extent, d_rel, head_dim, inv_dim);
                    const int g = (int)(lane >> 1);
                    if ((lane & 1u) == 0 && g < MG) {
                        shared.weights[g][index] = score;
                    }
                }
            }
        }
        asm volatile("cp.async.wait_all;" ::: "memory");
    }
    __syncthreads();

    if (value_norm != 0) {
        for (int base = warp; base < n; base += 4 * INKLING_WINDOW_WARPS) {
            float inverse[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int i = base + u * INKLING_WINDOW_WARPS;
                inverse[u] = i < n ? inkling_attn_value_inverse(staged.row(i), head_dim, eps) : 1.0f;
            }
            if (lane == 0) {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int i = base + u * INKLING_WINDOW_WARPS;
                    if (i < n) {
                        shared.scales[i] = inverse[u];
                    }
                }
            }
        }
    } else {
        for (int i = threadIdx.x; i < n; i += THREADS) {
            shared.scales[i] = 1.0f;
        }
    }

    float held[MG][INKLING_PART_HELD];
#pragma unroll
    for (int g = 0; g < MG; ++g) {
#pragma unroll
        for (int k = 0; k < INKLING_PART_HELD; ++k) {
            const int i = (int)threadIdx.x + k * INKLING_PART_THREADS;
            held[g][k] = i < n ? shared.weights[g][i] : -INFINITY;
        }
    }
    inkling_part_run<MG, 4, THREADS / INKLING_PART_THREADS, true>(
        shared, held, n, MG, staged, head_dim, &shared.part[0][0], 2 + INKLING_MAX_HEAD_DIM);
    cluster.sync();

    const float* partials[INKLING_WINDOW_RUNS];
#pragma unroll
    for (int r = 0; r < INKLING_WINDOW_RUNS; ++r) {
        partials[r] = cluster.map_shared_rank(&shared.part[0][0], r);
    }
    constexpr int WIDTH = 2 + INKLING_MAX_HEAD_DIM;
    for (int index = run + INKLING_WINDOW_RUNS * (int)threadIdx.x; index < MG * head_dim;
         index += INKLING_WINDOW_RUNS * THREADS) {
        const int g = index / head_dim;
        const int dim = index - g * head_dim;
        const uint64_t row_head = (uint64_t)row * heads + head0 + g;
        float tops[INKLING_WINDOW_RUNS];
        float largest = sink != nullptr ? sink[row_head] : -INFINITY;
#pragma unroll
        for (int r = 0; r < INKLING_WINDOW_RUNS; ++r) {
            tops[r] = r < runs ? partials[r][g * WIDTH] : -INFINITY;
            if (r < runs) {
                largest = fmaxf(largest, tops[r]);
            }
        }
        float denominator = 0.0f;
        float acc = 0.0f;
        if (largest != -INFINITY) {
            denominator = inkling_merge_open(sink, row_head, largest);
#pragma unroll
            for (int r = 0; r < INKLING_WINDOW_RUNS; ++r) {
                if (r < runs) {
                    const float factor = inkling_merge_factor(tops[r], largest);
                    denominator = fmaf(partials[r][g * WIDTH + 1], factor, denominator);
                    acc = fmaf(partials[r][g * WIDTH + 2 + dim], factor, acc);
                }
            }
        }
        const __nv_bfloat16 val =
            inkling_merge_out(acc, denominator, gate[row_head * head_dim + dim]);
        if (out != nullptr) {
            out[row_head * head_dim + dim] = val;
        }
        if (out_f32 != nullptr) {
            out_f32[row_head * head_dim + dim] = __bfloat162float(val);
        }
    }
    cluster.sync();
}

// ---------------------------------------------------------------------------------------------
// Multi-row shared-KV full and sliding attention for groups == 3 (Inkling-2-Mini, rows <= 8,
// head_dim == 128). Reads K and V from DRAM ONCE across all verify rows in a single block per
// (kv_head, run), preserving exact bit-for-bit parity with inkling_attn_scored_part and
// inkling_attn_window_g3.
// ---------------------------------------------------------------------------------------------
#define INKLING_SM_MAX_ROWS 8
#define INKLING_SM_MG 3
#define INKLING_SM_Q (INKLING_SM_MAX_ROWS * INKLING_SM_MG)
#define INKLING_SM_WARPS 16
#define INKLING_SM_THREADS (32 * INKLING_SM_WARPS)
#define INKLING_SM_VTILE 64
#define INKLING_WM_VTILE 66
#define INKLING_WM_MAX_SLAB (INKLING_PART_KEYS + INKLING_SM_MAX_ROWS)

struct InklingScoredMultiShared {
    float weights[INKLING_SM_Q][INKLING_PART_KEYS];
    float scales[INKLING_PART_KEYS];
    float largest[INKLING_SM_Q];
    float total[INKLING_SM_Q];
    union {
        struct {
            alignas(16) uint32_t stage[INKLING_SM_WARPS][8][4][INKLING_QUAD_PAD];
            alignas(16) float queries[4][8 * INKLING_SM_Q * 4 + 4];
            alignas(16) float rel[4][8 * INKLING_SM_Q + 4];
            alignas(16) float dim_gains[4][8 * 4 + 4];
        } k;
        struct {
            alignas(16) uint4 v_stage[2][INKLING_SM_VTILE][INKLING_MAX_HEAD_DIM / 8];
        } v;
        float v_sums[4][INKLING_PART_STRIDE][INKLING_SM_MG][INKLING_MAX_HEAD_DIM];
    };
};

static_assert(sizeof(InklingScoredMultiShared) == 82816,
              "SCORED_G3_MULTI_SHARED in crates/sparkflash-model-inkling2/src/device.rs");

// Grid (kv_heads, runs, 1), 512 threads, SCORED_G3_MULTI_SHARED dynamic shared bytes.
extern "C" __global__ void __launch_bounds__(INKLING_SM_THREADS, 1)
    inkling_attn_scored_part_g3_multi(
        const float* __restrict__ q,
        const __nv_bfloat16* __restrict__ keys,
        const __nv_bfloat16* __restrict__ values,
        const float* __restrict__ rel_proj,
        const __nv_bfloat16* __restrict__ rel_bank,
        const float* __restrict__ q_scale,
        const float* __restrict__ k_scale,
        const float* __restrict__ rel_scale,
        const float* __restrict__ key_gains,
        const float* __restrict__ value_scales,
        float* __restrict__ parts,
        int heads,
        int kv_heads,
        int head_dim,
        int groups,
        int rows,
        int first_position,
        int cache_rows,
        int rel_extent,
        int d_rel,
        float qk_clip,
        float rel_clip,
        float tau,
        const float* __restrict__ tau_rows,
        int value_norm,
        long long scale_first,
        int scale_stride,
        int part_stride) {
    OPT_PDL_ENTRY();
    extern __shared__ __align__(16) unsigned char smem_raw[];
    InklingScoredMultiShared& shared = *reinterpret_cast<InklingScoredMultiShared*>(smem_raw);

    const int kv_head = blockIdx.x;
    const int run = blockIdx.y;
    if (kv_head >= kv_heads || rows <= 0 || rows > INKLING_SM_MAX_ROWS ||
        head_dim != INKLING_MAX_HEAD_DIM || groups != INKLING_SM_MG || d_rel < 0 ||
        d_rel > INKLING_MAX_D_REL) {
        return;
    }
    const int64_t begin = (int64_t)run * INKLING_PART_KEYS;
    const int64_t max_count = (int64_t)first_position + rows;
    if (begin >= max_count) {
        return;
    }
    const int n_max = (int)((max_count - begin) < INKLING_PART_KEYS ? (max_count - begin)
                                                                    : INKLING_PART_KEYS);
    const int head0 = kv_head * INKLING_SM_MG;
    const int Q = rows * INKLING_SM_MG;
    const unsigned int lane = threadIdx.x & 31u;
    const int warp = (int)(threadIdx.x >> 5);
    const uint64_t kv_stride = (uint64_t)kv_heads * head_dim;
    const uint64_t kv_offset = (uint64_t)kv_head * head_dim;

    uint4 pieces[4];
    if (8 * warp < n_max) {
        inkling_quad_fetch(keys, begin, 8 * warp, n_max, cache_rows, kv_stride, kv_offset,
                           head_dim, pieces);
    }

    for (int qi = warp; qi < Q; qi += INKLING_SM_WARPS) {
        const int r = qi / INKLING_SM_MG;
        const int g = qi - r * INKLING_SM_MG;
        const int head = head0 + g;
        float qv[INKLING_MAX_HEAD_DIM / 32];
        float rv;
        inkling_attn_query(q + ((uint64_t)r * heads + head) * head_dim,
                           rel_proj + (uint64_t)r * heads * d_rel, q_scale, rel_scale, head,
                           head_dim, d_rel, qk_clip, rel_clip,
                           tau_rows != nullptr ? tau_rows[r] : tau, qv, rv);
        const int u = (int)(lane & 3u);
        const int m = (int)(lane >> 2);
        *reinterpret_cast<float4*>(&shared.k.queries[u][(m * INKLING_SM_Q + qi) * 4]) =
            make_float4(qv[0], qv[1], qv[2], qv[3]);
        shared.k.rel[u][m * INKLING_SM_Q + qi] = rv;
    }
    if (k_scale != nullptr && (int)threadIdx.x < INKLING_MAX_HEAD_DIM) {
        const int dim = (int)threadIdx.x;
        shared.k.dim_gains[dim & 3][4 * ((dim >> 2) & 7) + (dim >> 5)] =
            k_scale[(uint64_t)kv_head * head_dim + dim];
    }
    if ((int)threadIdx.x < n_max) {
        const float* scale_row =
            value_scales + (uint64_t)kv_head * scale_stride - scale_first + begin;
        shared.scales[threadIdx.x] = value_norm != 0 ? scale_row[threadIdx.x] : 1.0f;
    }
    __syncthreads();

    const float inv_dim = 1.0f / sqrtf((float)head_dim);
    const float* gains = key_gains + (uint64_t)kv_head * scale_stride - scale_first;
    const int u = (int)(lane & 3u);
    const int quad = (int)(lane >> 2);
    uint32_t(&stage)[8][4][INKLING_QUAD_PAD] = shared.k.stage[warp];
    const bool run_may_bias =
        (d_rel > 0 && rel_extent > 0) &&
        ((int64_t)first_position - (begin + n_max - 1) < (int64_t)rel_extent);
    const bool has_k_scale = (k_scale != nullptr);

    for (int base = 8 * warp; base < n_max; base += 8 * INKLING_SM_WARPS) {
        __syncwarp();
        inkling_quad_stage(stage, pieces);
        __syncwarp();
        uint4 words[4];
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            words[step] = *reinterpret_cast<const uint4*>(&stage[quad][u][4 * step]);
        }
        if (base + 8 * INKLING_SM_WARPS < n_max) {
            inkling_quad_fetch(keys, begin, base + 8 * INKLING_SM_WARPS, n_max, cache_rows,
                               kv_stride, kv_offset, head_dim, pieces);
        }
        const int i = base + quad;
        const bool valid = i < n_max;
        const int64_t slot = begin + i;
        const float k_gain = valid ? gains[slot] : 1.0f;
        float kv[8][4];
        if (!has_k_scale) {
#pragma unroll
            for (int m = 0; m < 8; ++m) {
#pragma unroll
                for (int step = 0; step < 4; ++step) {
                    const uint32_t word = inkling_quad_word(words[step], m >> 1);
                    const float key = (m & 1) != 0 ? __uint_as_float(word & 0xffff0000u)
                                                   : __uint_as_float(word << 16);
                    kv[m][step] = __fmul_rn(key, k_gain);
                }
            }
        } else {
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                const float4 gain4 = *reinterpret_cast<const float4*>(&shared.k.dim_gains[u][4 * m]);
                const float gs[4] = {gain4.x, gain4.y, gain4.z, gain4.w};
#pragma unroll
                for (int step = 0; step < 4; ++step) {
                    const uint32_t word = inkling_quad_word(words[step], m >> 1);
                    const float key = (m & 1) != 0 ? __uint_as_float(word & 0xffff0000u)
                                                   : __uint_as_float(word << 16);
                    kv[m][step] = __fmul_rn(__fmul_rn(key, gs[step]), k_gain);
                }
            }
        }
        if (!run_may_bias) {
            for (int r = 0; r < rows; ++r) {
                float dots[INKLING_SM_MG];
#pragma unroll
                for (int g = 0; g < INKLING_SM_MG; ++g) {
                    const int qi = r * INKLING_SM_MG + g;
                    float leaf[8];
#pragma unroll
                    for (int m = 0; m < 8; ++m) {
                        const float4 q4 = *reinterpret_cast<const float4*>(
                            &shared.k.queries[u][(m * INKLING_SM_Q + qi) * 4]);
                        float dot = __fmul_rn(q4.x, kv[m][0]);
                        dot = __fadd_rn(dot, __fmul_rn(q4.y, kv[m][1]));
                        dot = __fadd_rn(dot, __fmul_rn(q4.z, kv[m][2]));
                        dot = __fadd_rn(dot, __fmul_rn(q4.w, kv[m][3]));
                        leaf[m] = dot;
                    }
                    dots[g] = inkling_quad_sum(inkling_quad_tree(leaf));
                }
                if (valid && u < INKLING_SM_MG) {
                    const float d = u == 0 ? dots[0] : (u == 1 ? dots[1] : dots[2]);
                    shared.weights[r * INKLING_SM_MG + u][i] = __fmul_rn(d, inv_dim);
                }
            }
        } else {
            for (int r = 0; r < rows; ++r) {
                const int position = first_position + r;
                const int64_t distance = (int64_t)position - slot;
                const bool row_valid = valid && distance >= 0;
                float dots[INKLING_SM_MG];
#pragma unroll
                for (int g = 0; g < INKLING_SM_MG; ++g) {
                    const int qi = r * INKLING_SM_MG + g;
                    float leaf[8];
#pragma unroll
                    for (int m = 0; m < 8; ++m) {
                        const float4 q4 = *reinterpret_cast<const float4*>(
                            &shared.k.queries[u][(m * INKLING_SM_Q + qi) * 4]);
                        float dot = __fmul_rn(q4.x, kv[m][0]);
                        dot = __fadd_rn(dot, __fmul_rn(q4.y, kv[m][1]));
                        dot = __fadd_rn(dot, __fmul_rn(q4.z, kv[m][2]));
                        dot = __fadd_rn(dot, __fmul_rn(q4.w, kv[m][3]));
                        leaf[m] = dot;
                    }
                    dots[g] = inkling_quad_sum(inkling_quad_tree(leaf));
                }
                float biases[INKLING_SM_MG] = {0.0f, 0.0f, 0.0f};
                const bool biased = row_valid && inkling_attn_biased(distance, rel_extent, d_rel);
                if (__any_sync(0xffffffffu, biased)) {
                    float bank[8];
#pragma unroll
                    for (int m = 0; m < 8; ++m) {
                        const int dim = u + 4 * m;
                        bank[m] = biased && dim < d_rel
                                      ? __bfloat162float(
                                            rel_bank[(uint64_t)dim * rel_extent + (uint64_t)distance])
                                      : 0.0f;
                    }
#pragma unroll
                    for (int g = 0; g < INKLING_SM_MG; ++g) {
                        const int qi = r * INKLING_SM_MG + g;
                        float leaf[8];
#pragma unroll
                        for (int m = 0; m < 8; ++m) {
                            leaf[m] = u + 4 * m < d_rel
                                          ? __fmul_rn(shared.k.rel[u][m * INKLING_SM_Q + qi], bank[m])
                                          : 0.0f;
                        }
                        biases[g] = inkling_quad_sum(inkling_quad_tree(leaf));
                    }
                }
                if (valid && u < INKLING_SM_MG) {
                    float score = -INFINITY;
                    if (row_valid) {
                        const float d = u == 0 ? dots[0] : (u == 1 ? dots[1] : dots[2]);
                        score = __fmul_rn(d, inv_dim);
                        if (biased) {
                            const float b = u == 0 ? biases[0] : (u == 1 ? biases[1] : biases[2]);
                            score = __fadd_rn(score, b);
                        }
                    }
                    shared.weights[r * INKLING_SM_MG + u][i] = score;
                }
            }
        }
    }
    __syncthreads();

    auto issue_vtile = [&](int buf, int t0) {
#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int c = (int)threadIdx.x + pass * INKLING_SM_THREADS;
            const int i = c >> 4;
            const int chunk = c & 15;
            const int key_idx = t0 + i;
            if (key_idx < n_max) {
                const __nv_bfloat16* src =
                    values + inkling_attn_cache_row(begin + key_idx, cache_rows) * kv_stride +
                    kv_offset + chunk * 8;
                const uint32_t dst =
                    (uint32_t)__cvta_generic_to_shared(&shared.v.v_stage[buf][i][chunk]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
            }
        }
        asm volatile("cp.async.commit_group;");
    };

    const int num_tiles = (n_max + INKLING_SM_VTILE - 1) / INKLING_SM_VTILE;
    if (num_tiles > 0) {
        issue_vtile(0, 0);
    }

    for (int qi = warp; qi < Q; qi += INKLING_SM_WARPS) {
        const int r = qi / INKLING_SM_MG;
        const int64_t count_r = (int64_t)(first_position + r) + 1;
        const int n_r = count_r <= begin
                            ? 0
                            : (int)((count_r - begin) < INKLING_PART_KEYS ? (count_r - begin)
                                                                          : INKLING_PART_KEYS);
        float vals[8];
        float lmax = -INFINITY;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int i0 = s * 32 + (int)lane;
            const int i1 = i0 + 128;
            vals[s] = i0 < n_r ? shared.weights[qi][i0] : -INFINITY;
            vals[s + 4] = i1 < n_r ? shared.weights[qi][i1] : -INFINITY;
            lmax = fmaxf(lmax, fmaxf(vals[s], vals[s + 4]));
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, offset));
        }
        const bool live = lmax != -INFINITY;
        float wsum[4];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int i0 = s * 32 + (int)lane;
            const int i1 = i0 + 128;
            float t0 = 0.0f;
            float t1 = 0.0f;
            if (i0 < n_r) {
                t0 = live ? expf(vals[s] - lmax) : 0.0f;
                shared.weights[qi][i0] = t0;
            }
            if (i1 < n_r) {
                t1 = live ? expf(vals[s + 4] - lmax) : 0.0f;
                shared.weights[qi][i1] = t1;
            }
            float sum = t0 + t1;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, offset);
            }
            wsum[s] = sum;
        }
        if (lane == 0) {
            shared.largest[qi] = lmax;
            shared.total[qi] = (wsum[0] + wsum[1]) + (wsum[2] + wsum[3]);
        }
    }

    const int quarter = (int)(threadIdx.x >> 7);
    const int within = (int)(threadIdx.x & 127u);
    const int v_lane = within & 15;
    const int key_lane = within >> 4;
    const int dim0 = v_lane * INKLING_PART_DIMS;
    const int r0 = quarter;
    const int r1 = quarter + 4;
    const bool has_r0 = r0 < rows;
    const bool has_r1 = r1 < rows;
    const int n_r0 = has_r0 ? max(0, min((int)((int64_t)(first_position + r0 + 1) - begin),
                                         INKLING_PART_KEYS))
                            : 0;
    const int n_r1 = has_r1 ? max(0, min((int)((int64_t)(first_position + r1 + 1) - begin),
                                         INKLING_PART_KEYS))
                            : 0;

    float acc0[INKLING_SM_MG][INKLING_PART_DIMS];
    float acc1[INKLING_SM_MG][INKLING_PART_DIMS];
#pragma unroll
    for (int g = 0; g < INKLING_SM_MG; ++g) {
#pragma unroll
        for (int j = 0; j < INKLING_PART_DIMS; ++j) {
            acc0[g][j] = 0.0f;
            acc1[g][j] = 0.0f;
        }
    }

    int buf = 0;
    for (int tile = 0; tile < num_tiles; ++tile) {
        const int t0 = tile * INKLING_SM_VTILE;
        const int next_tile = tile + 1;
        if (next_tile < num_tiles) {
            issue_vtile(buf ^ 1, next_tile * INKLING_SM_VTILE);
            asm volatile("cp.async.wait_group 1;" ::: "memory");
        } else {
            asm volatile("cp.async.wait_all;" ::: "memory");
        }
        __syncthreads();
        const int tile_n = min(INKLING_SM_VTILE, n_max - t0);
        if (has_r0 || has_r1) {
#pragma unroll 4
            for (int i = key_lane; i < tile_n; i += INKLING_PART_STRIDE) {
                const int key_idx = t0 + i;
                if ((has_r0 && key_idx < n_r0) || (has_r1 && key_idx < n_r1)) {
                    const uint4 packed = shared.v.v_stage[buf][i][v_lane];
                    const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
                    const float sc = shared.scales[key_idx];
                    float val[INKLING_PART_DIMS];
#pragma unroll
                    for (int pair = 0; pair < INKLING_PART_DIMS / 2; ++pair) {
                        const float2 two = __bfloat1622float2(pairs[pair]);
                        val[2 * pair] = two.x * sc;
                        val[2 * pair + 1] = two.y * sc;
                    }
                    if (has_r0 && key_idx < n_r0) {
#pragma unroll
                        for (int g = 0; g < INKLING_SM_MG; ++g) {
                            const float w = shared.weights[r0 * INKLING_SM_MG + g][key_idx];
#pragma unroll
                            for (int j = 0; j < INKLING_PART_DIMS; ++j) {
                                acc0[g][j] = fmaf(w, val[j], acc0[g][j]);
                            }
                        }
                    }
                    if (has_r1 && key_idx < n_r1) {
#pragma unroll
                        for (int g = 0; g < INKLING_SM_MG; ++g) {
                            const float w = shared.weights[r1 * INKLING_SM_MG + g][key_idx];
#pragma unroll
                            for (int j = 0; j < INKLING_PART_DIMS; ++j) {
                                acc1[g][j] = fmaf(w, val[j], acc1[g][j]);
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
        buf ^= 1;
    }

    const uint64_t head_stride = (uint64_t)part_stride * (2 + (uint64_t)head_dim);
    auto flush_half = [&](int base_r, const float (&acc)[INKLING_SM_MG][INKLING_PART_DIMS],
                          bool has_r, int n_r) {
        if (has_r && n_r > 0) {
#pragma unroll
            for (int g = 0; g < INKLING_SM_MG; ++g) {
#pragma unroll
                for (int j = 0; j < INKLING_PART_DIMS; ++j) {
                    shared.v_sums[quarter][key_lane][g][dim0 + j] = acc[g][j];
                }
            }
        }
        __syncthreads();
        if (has_r && n_r > 0) {
            const int r = base_r + quarter;
            float* out_base = parts + ((uint64_t)r * heads + head0) * head_stride +
                              (uint64_t)run * (2 + (uint64_t)head_dim);
            const int dim = within;
#pragma unroll
            for (int g = 0; g < INKLING_SM_MG; ++g) {
                float sum = 0.0f;
#pragma unroll
                for (int k = 0; k < INKLING_PART_STRIDE; ++k) {
                    sum += shared.v_sums[quarter][k][g][dim];
                }
                out_base[(uint64_t)g * head_stride + 2 + dim] = sum;
            }
            if (within == 0) {
#pragma unroll
                for (int g = 0; g < INKLING_SM_MG; ++g) {
                    out_base[(uint64_t)g * head_stride] = shared.largest[r * INKLING_SM_MG + g];
                    out_base[(uint64_t)g * head_stride + 1] = shared.total[r * INKLING_SM_MG + g];
                }
            }
        }
        __syncthreads();
    };
    flush_half(0, acc0, has_r0, n_r0);
    if (rows > 4) {
        flush_half(4, acc1, has_r1, n_r1);
    }
}

// K2 macro-run (512 keys/block = 2 sub-runs of 256) shared-KV full attention for long contexts
// (>= 8k keys): keeps queries resident across both 256-key sub-runs and merges them before
// writing `parts`, halving `parts` size (38 MB -> 19 MB at 97.6k rows=8) so it stays L2-resident
// and eliminates the L2-spill cliff in both scored_part and merge_f32.
struct InklingScoredMultiSharedK2 {
    float weights[INKLING_SM_Q][INKLING_PART_KEYS];
    float scales[INKLING_PART_KEYS];
    float largest[INKLING_SM_Q];
    float total[INKLING_SM_Q];
    alignas(16) float queries[4][8 * INKLING_SM_Q * 4 + 4];
    alignas(16) float rel[4][8 * INKLING_SM_Q + 4];
    alignas(16) float dim_gains[4][8 * 4 + 4];
    float m_max[INKLING_SM_Q];
    float m_sum[INKLING_SM_Q];
    float m_acc[INKLING_SM_Q][INKLING_MAX_HEAD_DIM];
    union {
        alignas(16) uint32_t stage[INKLING_SM_WARPS][8][4][INKLING_QUAD_PAD];
        alignas(16) uint4 v_stage[2][INKLING_SM_VTILE][INKLING_MAX_HEAD_DIM / 8];
        float v_sums[4][INKLING_PART_STRIDE][INKLING_MAX_HEAD_DIM];
    };
};

static_assert(sizeof(InklingScoredMultiSharedK2) == 95296,
              "SCORED_G3_MULTI_K2_SHARED in crates/sparkflash-model-inkling2/src/device.rs");

extern "C" __global__ void __launch_bounds__(INKLING_SM_THREADS, 1)
    inkling_attn_scored_part_g3_multi_k2(
        const float* __restrict__ q,
        const __nv_bfloat16* __restrict__ keys,
        const __nv_bfloat16* __restrict__ values,
        const float* __restrict__ rel_proj,
        const __nv_bfloat16* __restrict__ rel_bank,
        const float* __restrict__ q_scale,
        const float* __restrict__ k_scale,
        const float* __restrict__ rel_scale,
        const float* __restrict__ key_gains,
        const float* __restrict__ value_scales,
        float* __restrict__ parts,
        int heads,
        int kv_heads,
        int head_dim,
        int groups,
        int rows,
        int first_position,
        int cache_rows,
        int rel_extent,
        int d_rel,
        float qk_clip,
        float rel_clip,
        float tau,
        const float* __restrict__ tau_rows,
        int value_norm,
        long long scale_first,
        int scale_stride,
        int part_stride) {
    OPT_PDL_ENTRY();
    extern __shared__ __align__(16) unsigned char smem_k2_raw[];
    InklingScoredMultiSharedK2& shared = *reinterpret_cast<InklingScoredMultiSharedK2*>(smem_k2_raw);

    const int kv_head = blockIdx.x;
    const int macro_run = blockIdx.y;
    if (kv_head >= kv_heads || rows <= 0 || rows > INKLING_SM_MAX_ROWS ||
        head_dim != INKLING_MAX_HEAD_DIM || groups != INKLING_SM_MG || d_rel < 0 ||
        d_rel > INKLING_MAX_D_REL) {
        return;
    }
    const int64_t macro_begin = (int64_t)macro_run * (2 * INKLING_PART_KEYS);
    const int64_t max_count = (int64_t)first_position + rows;
    if (macro_begin >= max_count) {
        return;
    }

    const int head0 = kv_head * INKLING_SM_MG;
    const int Q = rows * INKLING_SM_MG;
    const unsigned int lane = threadIdx.x & 31u;
    const int warp = (int)(threadIdx.x >> 5);
    const uint64_t kv_stride = (uint64_t)kv_heads * head_dim;
    const uint64_t kv_offset = (uint64_t)kv_head * head_dim;
    const float inv_dim = 1.0f / sqrtf((float)head_dim);
    const float* gains = key_gains + (uint64_t)kv_head * scale_stride - scale_first;
    const int u = (int)(lane & 3u);
    const int quad = (int)(lane >> 2);
    const bool has_k_scale = (k_scale != nullptr);

    const int n0_max = (int)min(max_count - macro_begin, (int64_t)INKLING_PART_KEYS);
    uint4 pieces[4];
    if (8 * warp < n0_max) {
        inkling_quad_fetch(keys, macro_begin, 8 * warp, n0_max, cache_rows, kv_stride, kv_offset,
                           head_dim, pieces);
    }
    for (int qi = warp; qi < Q; qi += INKLING_SM_WARPS) {
        const int r = qi / INKLING_SM_MG;
        const int g = qi - r * INKLING_SM_MG;
        const int head = head0 + g;
        float qv[INKLING_MAX_HEAD_DIM / 32];
        float rv;
        inkling_attn_query(q + ((uint64_t)r * heads + head) * head_dim,
                           rel_proj + (uint64_t)r * heads * d_rel, q_scale, rel_scale, head,
                           head_dim, d_rel, qk_clip, rel_clip,
                           tau_rows != nullptr ? tau_rows[r] : tau, qv, rv);
        const int u0 = (int)(lane & 3u);
        const int m0 = (int)(lane >> 2);
        *reinterpret_cast<float4*>(&shared.queries[u0][(m0 * INKLING_SM_Q + qi) * 4]) =
            make_float4(qv[0], qv[1], qv[2], qv[3]);
        shared.rel[u0][m0 * INKLING_SM_Q + qi] = rv;
    }
    if (has_k_scale && (int)threadIdx.x < INKLING_MAX_HEAD_DIM) {
        const int dim = (int)threadIdx.x;
        shared.dim_gains[dim & 3][4 * ((dim >> 2) & 7) + (dim >> 5)] =
            k_scale[(uint64_t)kv_head * head_dim + dim];
    }
    __syncthreads();

    const int quarter = (int)(threadIdx.x >> 7);
    const int within = (int)(threadIdx.x & 127u);
    const int v_lane = within & 15;
    const int key_lane = within >> 4;
    const int dim0 = v_lane * INKLING_PART_DIMS;
    const int r0 = quarter;
    const int r1 = quarter + 4;
    const bool has_r0 = r0 < rows;
    const bool has_r1 = r1 < rows;
    const uint64_t head_stride = (uint64_t)part_stride * (2 + (uint64_t)head_dim);
    const int sub_runs = (macro_begin + INKLING_PART_KEYS < max_count) ? 2 : 1;

    for (int sr = 0; sr < sub_runs; ++sr) {
        const int64_t begin = macro_begin + (int64_t)sr * INKLING_PART_KEYS;
        const int n_max = (int)min(max_count - begin, (int64_t)INKLING_PART_KEYS);
        if (sr > 0 && 8 * warp < n_max) {
            inkling_quad_fetch(keys, begin, 8 * warp, n_max, cache_rows, kv_stride, kv_offset,
                               head_dim, pieces);
        }
        if ((int)threadIdx.x < n_max) {
            const float* scale_row =
                value_scales + (uint64_t)kv_head * scale_stride - scale_first + begin;
            shared.scales[threadIdx.x] = value_norm != 0 ? scale_row[threadIdx.x] : 1.0f;
        }
        uint32_t(&stage)[8][4][INKLING_QUAD_PAD] = shared.stage[warp];
        const bool run_may_bias =
            (d_rel > 0 && rel_extent > 0) &&
            ((int64_t)first_position - (begin + n_max - 1) < (int64_t)rel_extent);

        for (int base = 8 * warp; base < n_max; base += 8 * INKLING_SM_WARPS) {
            __syncwarp();
            inkling_quad_stage(stage, pieces);
            __syncwarp();
            uint4 words[4];
#pragma unroll
            for (int step = 0; step < 4; ++step) {
                words[step] = *reinterpret_cast<const uint4*>(&stage[quad][u][4 * step]);
            }
            if (base + 8 * INKLING_SM_WARPS < n_max) {
                inkling_quad_fetch(keys, begin, base + 8 * INKLING_SM_WARPS, n_max, cache_rows,
                                   kv_stride, kv_offset, head_dim, pieces);
            }
            const int i = base + quad;
            const bool valid = i < n_max;
            const int64_t slot = begin + i;
            const float k_gain = valid ? gains[slot] : 1.0f;
            float kv[8][4];
            if (!has_k_scale) {
#pragma unroll
                for (int m = 0; m < 8; ++m) {
#pragma unroll
                    for (int step = 0; step < 4; ++step) {
                        const uint32_t word = inkling_quad_word(words[step], m >> 1);
                        const float key = (m & 1) != 0 ? __uint_as_float(word & 0xffff0000u)
                                                       : __uint_as_float(word << 16);
                        kv[m][step] = __fmul_rn(key, k_gain);
                    }
                }
            } else {
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    const float4 gain4 =
                        *reinterpret_cast<const float4*>(&shared.dim_gains[u][4 * m]);
                    const float gs[4] = {gain4.x, gain4.y, gain4.z, gain4.w};
#pragma unroll
                    for (int step = 0; step < 4; ++step) {
                        const uint32_t word = inkling_quad_word(words[step], m >> 1);
                        const float key = (m & 1) != 0 ? __uint_as_float(word & 0xffff0000u)
                                                       : __uint_as_float(word << 16);
                        kv[m][step] = __fmul_rn(__fmul_rn(key, gs[step]), k_gain);
                    }
                }
            }
            if (!run_may_bias) {
                for (int r = 0; r < rows; ++r) {
                    float dots[INKLING_SM_MG];
#pragma unroll
                    for (int g = 0; g < INKLING_SM_MG; ++g) {
                        const int qi = r * INKLING_SM_MG + g;
                        float leaf[8];
#pragma unroll
                        for (int m = 0; m < 8; ++m) {
                            const float4 q4 = *reinterpret_cast<const float4*>(
                                &shared.queries[u][(m * INKLING_SM_Q + qi) * 4]);
                            float dot = __fmul_rn(q4.x, kv[m][0]);
                            dot = __fadd_rn(dot, __fmul_rn(q4.y, kv[m][1]));
                            dot = __fadd_rn(dot, __fmul_rn(q4.z, kv[m][2]));
                            dot = __fadd_rn(dot, __fmul_rn(q4.w, kv[m][3]));
                            leaf[m] = dot;
                        }
                        dots[g] = inkling_quad_sum(inkling_quad_tree(leaf));
                    }
                    if (valid && u < INKLING_SM_MG) {
                        const float d = u == 0 ? dots[0] : (u == 1 ? dots[1] : dots[2]);
                        shared.weights[r * INKLING_SM_MG + u][i] = __fmul_rn(d, inv_dim);
                    }
                }
            } else {
                for (int r = 0; r < rows; ++r) {
                    const int position = first_position + r;
                    const int64_t distance = (int64_t)position - slot;
                    const bool row_valid = valid && distance >= 0;
                    float dots[INKLING_SM_MG];
#pragma unroll
                    for (int g = 0; g < INKLING_SM_MG; ++g) {
                        const int qi = r * INKLING_SM_MG + g;
                        float leaf[8];
#pragma unroll
                        for (int m = 0; m < 8; ++m) {
                            const float4 q4 = *reinterpret_cast<const float4*>(
                                &shared.queries[u][(m * INKLING_SM_Q + qi) * 4]);
                            float dot = __fmul_rn(q4.x, kv[m][0]);
                            dot = __fadd_rn(dot, __fmul_rn(q4.y, kv[m][1]));
                            dot = __fadd_rn(dot, __fmul_rn(q4.z, kv[m][2]));
                            dot = __fadd_rn(dot, __fmul_rn(q4.w, kv[m][3]));
                            leaf[m] = dot;
                        }
                        dots[g] = inkling_quad_sum(inkling_quad_tree(leaf));
                    }
                    float biases[INKLING_SM_MG] = {0.0f, 0.0f, 0.0f};
                    const bool biased =
                        row_valid && inkling_attn_biased(distance, rel_extent, d_rel);
                    if (__any_sync(0xffffffffu, biased)) {
                        float bank[8];
#pragma unroll
                        for (int m = 0; m < 8; ++m) {
                            const int dim = u + 4 * m;
                            bank[m] = biased && dim < d_rel
                                          ? __bfloat162float(
                                                rel_bank[(uint64_t)dim * rel_extent +
                                                         (uint64_t)distance])
                                          : 0.0f;
                        }
#pragma unroll
                        for (int g = 0; g < INKLING_SM_MG; ++g) {
                            const int qi = r * INKLING_SM_MG + g;
                            float leaf[8];
#pragma unroll
                            for (int m = 0; m < 8; ++m) {
                                leaf[m] =
                                    u + 4 * m < d_rel
                                        ? __fmul_rn(shared.rel[u][m * INKLING_SM_Q + qi], bank[m])
                                        : 0.0f;
                            }
                            biases[g] = inkling_quad_sum(inkling_quad_tree(leaf));
                        }
                    }
                    if (valid && u < INKLING_SM_MG) {
                        float score = -INFINITY;
                        if (row_valid) {
                            const float d = u == 0 ? dots[0] : (u == 1 ? dots[1] : dots[2]);
                            score = __fmul_rn(d, inv_dim);
                            if (biased) {
                                const float b = u == 0 ? biases[0] : (u == 1 ? biases[1] : biases[2]);
                                score = __fadd_rn(score, b);
                            }
                        }
                        shared.weights[r * INKLING_SM_MG + u][i] = score;
                    }
                }
            }
        }
        __syncthreads();

        auto issue_vtile = [&](int buf, int t0) {
#pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                const int c = (int)threadIdx.x + pass * INKLING_SM_THREADS;
                const int i = c >> 4;
                const int chunk = c & 15;
                const int key_idx = t0 + i;
                if (key_idx < n_max) {
                    const __nv_bfloat16* src =
                        values + inkling_attn_cache_row(begin + key_idx, cache_rows) * kv_stride +
                        kv_offset + chunk * 8;
                    const uint32_t dst =
                        (uint32_t)__cvta_generic_to_shared(&shared.v_stage[buf][i][chunk]);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
                }
            }
            asm volatile("cp.async.commit_group;");
        };

        const int num_tiles = (n_max + INKLING_SM_VTILE - 1) / INKLING_SM_VTILE;
        if (num_tiles > 0) {
            issue_vtile(0, 0);
        }

        for (int qi = warp; qi < Q; qi += INKLING_SM_WARPS) {
            const int r = qi / INKLING_SM_MG;
            const int64_t count_r = (int64_t)(first_position + r) + 1;
            const int n_r =
                count_r <= begin ? 0 : (int)min(count_r - begin, (int64_t)INKLING_PART_KEYS);
            float vals[8];
            float lmax = -INFINITY;
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int i0 = s * 32 + (int)lane;
                const int i1 = i0 + 128;
                vals[s] = i0 < n_r ? shared.weights[qi][i0] : -INFINITY;
                vals[s + 4] = i1 < n_r ? shared.weights[qi][i1] : -INFINITY;
                lmax = fmaxf(lmax, fmaxf(vals[s], vals[s + 4]));
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, offset));
            }
            const bool live = lmax != -INFINITY;
            float wsum[4];
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int i0 = s * 32 + (int)lane;
                const int i1 = i0 + 128;
                float t0 = 0.0f;
                float t1 = 0.0f;
                if (i0 < n_r) {
                    t0 = live ? expf(vals[s] - lmax) : 0.0f;
                    shared.weights[qi][i0] = t0;
                }
                if (i1 < n_r) {
                    t1 = live ? expf(vals[s + 4] - lmax) : 0.0f;
                    shared.weights[qi][i1] = t1;
                }
                float sum = t0 + t1;
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    sum += __shfl_xor_sync(0xffffffffu, sum, offset);
                }
                wsum[s] = sum;
            }
            if (lane == 0) {
                shared.largest[qi] = lmax;
                shared.total[qi] = (wsum[0] + wsum[1]) + (wsum[2] + wsum[3]);
            }
        }

        const int n_r0 = has_r0 ? max(0, min((int)((int64_t)(first_position + r0 + 1) - begin),
                                             INKLING_PART_KEYS))
                                : 0;
        const int n_r1 = has_r1 ? max(0, min((int)((int64_t)(first_position + r1 + 1) - begin),
                                             INKLING_PART_KEYS))
                                : 0;

        float acc0[INKLING_SM_MG][INKLING_PART_DIMS];
        float acc1[INKLING_SM_MG][INKLING_PART_DIMS];
#pragma unroll
        for (int g = 0; g < INKLING_SM_MG; ++g) {
#pragma unroll
            for (int j = 0; j < INKLING_PART_DIMS; ++j) {
                acc0[g][j] = 0.0f;
                acc1[g][j] = 0.0f;
            }
        }

        int buf = 0;
        for (int tile = 0; tile < num_tiles; ++tile) {
            const int t0 = tile * INKLING_SM_VTILE;
            const int next_tile = tile + 1;
            if (next_tile < num_tiles) {
                issue_vtile(buf ^ 1, next_tile * INKLING_SM_VTILE);
                asm volatile("cp.async.wait_group 1;" ::: "memory");
            } else {
                asm volatile("cp.async.wait_all;" ::: "memory");
            }
            __syncthreads();
            const int tile_n = min(INKLING_SM_VTILE, n_max - t0);
            if (has_r0 || has_r1) {
#pragma unroll 4
                for (int i = key_lane; i < tile_n; i += INKLING_PART_STRIDE) {
                    const int key_idx = t0 + i;
                    if ((has_r0 && key_idx < n_r0) || (has_r1 && key_idx < n_r1)) {
                        const uint4 packed = shared.v_stage[buf][i][v_lane];
                        const __nv_bfloat162* pairs =
                            reinterpret_cast<const __nv_bfloat162*>(&packed);
                        const float sc = shared.scales[key_idx];
                        float val[INKLING_PART_DIMS];
#pragma unroll
                        for (int pair = 0; pair < INKLING_PART_DIMS / 2; ++pair) {
                            const float2 two = __bfloat1622float2(pairs[pair]);
                            val[2 * pair] = two.x * sc;
                            val[2 * pair + 1] = two.y * sc;
                        }
                        if (has_r0 && key_idx < n_r0) {
#pragma unroll
                            for (int g = 0; g < INKLING_SM_MG; ++g) {
                                const float w = shared.weights[r0 * INKLING_SM_MG + g][key_idx];
#pragma unroll
                                for (int j = 0; j < INKLING_PART_DIMS; ++j) {
                                    acc0[g][j] = fmaf(w, val[j], acc0[g][j]);
                                }
                            }
                        }
                        if (has_r1 && key_idx < n_r1) {
#pragma unroll
                            for (int g = 0; g < INKLING_SM_MG; ++g) {
                                const float w = shared.weights[r1 * INKLING_SM_MG + g][key_idx];
#pragma unroll
                                for (int j = 0; j < INKLING_PART_DIMS; ++j) {
                                    acc1[g][j] = fmaf(w, val[j], acc1[g][j]);
                                }
                            }
                        }
                    }
                }
            }
            __syncthreads();
            buf ^= 1;
        }

        const bool is_last = (sr == sub_runs - 1);
        auto flush_half = [&](int base_r, const float (&acc)[INKLING_SM_MG][INKLING_PART_DIMS],
                              bool has_r, int n_r) {
#pragma unroll
            for (int g = 0; g < INKLING_SM_MG; ++g) {
                if (has_r && n_r > 0) {
#pragma unroll
                    for (int j = 0; j < INKLING_PART_DIMS; ++j) {
                        shared.v_sums[quarter][key_lane][dim0 + j] = acc[g][j];
                    }
                }
                __syncthreads();
                if (has_r) {
                    const int r = base_r + quarter;
                    const int qi = r * INKLING_SM_MG + g;
                    const int dim = within;
                    float sum = 0.0f;
                    if (n_r > 0) {
#pragma unroll
                        for (int k = 0; k < INKLING_PART_STRIDE; ++k) {
                            sum += shared.v_sums[quarter][k][dim];
                        }
                    }
                    if (!is_last) {
                        shared.m_acc[qi][dim] = sum;
                        if (dim == 0) {
                            shared.m_max[qi] = n_r > 0 ? shared.largest[qi] : -INFINITY;
                            shared.m_sum[qi] = n_r > 0 ? shared.total[qi] : 0.0f;
                        }
                    } else {
                        const int64_t count_r = (int64_t)(first_position + r) + 1;
                        if (macro_begin < count_r) {
                            float* out_base = parts +
                                              ((uint64_t)r * heads + head0 + g) * head_stride +
                                              (uint64_t)macro_run * (2 + (uint64_t)head_dim);
                            if (sr == 0) {
                                out_base[2 + dim] = sum;
                                if (dim == 0) {
                                    out_base[0] = shared.largest[qi];
                                    out_base[1] = shared.total[qi];
                                }
                            } else {
                                const float m0 = shared.m_max[qi];
                                const float m1 = n_r > 0 ? shared.largest[qi] : -INFINITY;
                                const float M = fmaxf(m0, m1);
                                float out_v = 0.0f;
                                if (M != -INFINITY) {
                                    const float f0 = m0 != -INFINITY ? expf(m0 - M) : 0.0f;
                                    const float f1 = m1 != -INFINITY ? expf(m1 - M) : 0.0f;
                                    out_v = fmaf(sum, f1, shared.m_acc[qi][dim] * f0);
                                    if (dim == 0) {
                                        out_base[0] = M;
                                        out_base[1] = fmaf(n_r > 0 ? shared.total[qi] : 0.0f, f1,
                                                           shared.m_sum[qi] * f0);
                                    }
                                } else if (dim == 0) {
                                    out_base[0] = -INFINITY;
                                    out_base[1] = 0.0f;
                                }
                                out_base[2 + dim] = out_v;
                            }
                        }
                    }
                }
                __syncthreads();
            }
        };
        flush_half(0, acc0, has_r0, n_r0);
        if (rows > 4) {
            flush_half(4, acc1, has_r1, n_r1);
        }
    }
}

struct InklingWindowMultiShared {
    float weights[INKLING_SM_Q][INKLING_PART_KEYS];
    float scales[INKLING_WM_MAX_SLAB];
    float largest[INKLING_SM_Q];
    float total[INKLING_SM_Q];
    float part[INKLING_SM_Q][2 + INKLING_MAX_HEAD_DIM];
    union {
        struct {
            alignas(16) uint32_t stage[INKLING_SM_WARPS][8][4][INKLING_QUAD_PAD];
            alignas(16) float queries[4][8 * INKLING_SM_Q * 4 + 4];
            alignas(16) float rel[4][8 * INKLING_SM_Q + 4];
            alignas(16) float dim_gains[4][8 * 4 + 4];
            alignas(16) int64_t row_run_begin[INKLING_SM_MAX_ROWS];
            alignas(16) int64_t row_run_end[INKLING_SM_MAX_ROWS];
        } k;
        struct {
            alignas(16) uint4 v_stage[2][INKLING_WM_VTILE][INKLING_MAX_HEAD_DIM / 8];
        } v;
        float v_sums[4][INKLING_PART_STRIDE][INKLING_SM_MG][INKLING_MAX_HEAD_DIM];
    };
};

static_assert(sizeof(InklingWindowMultiShared) == 95456,
              "WINDOW_G3_MULTI_SHARED in crates/sparkflash-model-inkling2/src/device.rs");

// Grid (kv_heads, 1, INKLING_WINDOW_RUNS), 512 threads, WINDOW_G3_MULTI_SHARED dynamic bytes.
extern "C" __global__ void __cluster_dims__(1, 1, INKLING_WINDOW_RUNS)
    __launch_bounds__(INKLING_SM_THREADS, 1) inkling_attn_window_g3_multi(
        const float* __restrict__ q,
        const __nv_bfloat16* __restrict__ keys,
        const __nv_bfloat16* __restrict__ values,
        const float* __restrict__ rel_proj,
        const __nv_bfloat16* __restrict__ rel_bank,
        const float* __restrict__ q_scale,
        const float* __restrict__ k_scale,
        const float* __restrict__ rel_scale,
        const float* __restrict__ gate,
        const float* __restrict__ sink,
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ out_f32,
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
        int value_norm,
        float eps) {
    OPT_PDL_ENTRY();
    extern __shared__ __align__(16) unsigned char smem_raw[];
    InklingWindowMultiShared& shared = *reinterpret_cast<InklingWindowMultiShared*>(smem_raw);
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();

    const int kv_head = blockIdx.x;
    const int run = blockIdx.z;
    if (kv_head >= kv_heads || rows <= 0 || rows > INKLING_SM_MAX_ROWS ||
        head_dim != INKLING_MAX_HEAD_DIM || groups != INKLING_SM_MG || d_rel < 0 ||
        d_rel > INKLING_MAX_D_REL || window <= 0 ||
        window > INKLING_WINDOW_RUNS * INKLING_PART_KEYS || gridDim.z != INKLING_WINDOW_RUNS) {
        return;
    }

    const int64_t first0 = inkling_attn_first_slot(first_position, window);
    const int64_t first_last = inkling_attn_first_slot(first_position + rows - 1, window);
    const int64_t begin_off = (int64_t)run * INKLING_PART_KEYS;
    const int64_t slab_first = first0 + begin_off;
    const int64_t max_pos_end = (int64_t)first_position + rows;
    const int64_t slab_end = min(first_last + begin_off + INKLING_PART_KEYS, max_pos_end);
    const int slab_n = slab_first < slab_end ? (int)(slab_end - slab_first) : 0;

    const int head0 = kv_head * INKLING_SM_MG;
    const int Q = rows * INKLING_SM_MG;
    const unsigned int lane = threadIdx.x & 31u;
    const int warp = (int)(threadIdx.x >> 5);
    const uint64_t kv_stride = (uint64_t)kv_heads * head_dim;
    const uint64_t kv_offset = (uint64_t)kv_head * head_dim;

    if (slab_n > 0) {
        uint4 pieces[4];
        if (8 * warp < slab_n) {
            inkling_quad_fetch(keys, slab_first, 8 * warp, slab_n, cache_rows, kv_stride, kv_offset,
                               head_dim, pieces);
        }
        for (int qi = warp; qi < Q; qi += INKLING_SM_WARPS) {
            const int r = qi / INKLING_SM_MG;
            const int g = qi - r * INKLING_SM_MG;
            const int head = head0 + g;
            float qv[INKLING_MAX_HEAD_DIM / 32];
            float rv;
            inkling_attn_query(q + ((uint64_t)r * heads + head) * head_dim,
                               rel_proj + (uint64_t)r * heads * d_rel, q_scale, rel_scale, head,
                               head_dim, d_rel, qk_clip, rel_clip,
                               tau_rows != nullptr ? tau_rows[r] : tau, qv, rv);
            const int u = (int)(lane & 3u);
            const int m = (int)(lane >> 2);
            *reinterpret_cast<float4*>(&shared.k.queries[u][(m * INKLING_SM_Q + qi) * 4]) =
                make_float4(qv[0], qv[1], qv[2], qv[3]);
            shared.k.rel[u][m * INKLING_SM_Q + qi] = rv;
        }
        if (k_scale != nullptr && (int)threadIdx.x < INKLING_MAX_HEAD_DIM) {
            const int dim = (int)threadIdx.x;
            shared.k.dim_gains[dim & 3][4 * ((dim >> 2) & 7) + (dim >> 5)] =
                k_scale[(uint64_t)kv_head * head_dim + dim];
        }
        if ((int)threadIdx.x < rows) {
            const int r = (int)threadIdx.x;
            const int position = first_position + r;
            const int64_t first_r = inkling_attn_first_slot(position, window);
            const int64_t rb = first_r + begin_off;
            shared.k.row_run_begin[r] = rb;
            shared.k.row_run_end[r] = min(rb + INKLING_PART_KEYS, (int64_t)position + 1);
        }
        __syncthreads();

        const float inv_dim = 1.0f / sqrtf((float)head_dim);
        const int u = (int)(lane & 3u);
        const int quad = (int)(lane >> 2);
        uint32_t(&stage)[8][4][INKLING_QUAD_PAD] = shared.k.stage[warp];
        const bool has_k_scale = (k_scale != nullptr);

        for (int base = 8 * warp; base < slab_n; base += 8 * INKLING_SM_WARPS) {
            __syncwarp();
            inkling_quad_stage(stage, pieces);
            __syncwarp();
            uint4 words[4];
#pragma unroll
            for (int step = 0; step < 4; ++step) {
                words[step] = *reinterpret_cast<const uint4*>(&stage[quad][u][4 * step]);
            }
            if (base + 8 * INKLING_SM_WARPS < slab_n) {
                inkling_quad_fetch(keys, slab_first, base + 8 * INKLING_SM_WARPS, slab_n,
                                   cache_rows, kv_stride, kv_offset, head_dim, pieces);
            }
            const int s = base + quad;
            const bool valid = s < slab_n;
            const int64_t slot = slab_first + s;
            float kv[8][4];
            float norm_leaf[8];
            if (!has_k_scale) {
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    float kn = 0.0f;
#pragma unroll
                    for (int step = 0; step < 4; ++step) {
                        const uint32_t word = inkling_quad_word(words[step], m >> 1);
                        const float key = (m & 1) != 0 ? __uint_as_float(word & 0xffff0000u)
                                                       : __uint_as_float(word << 16);
                        kv[m][step] = key;
                        kn = __fadd_rn(kn, __fmul_rn(key, key));
                    }
                    norm_leaf[m] = kn;
                }
            } else {
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    const float4 gain4 =
                        *reinterpret_cast<const float4*>(&shared.k.dim_gains[u][4 * m]);
                    const float gs[4] = {gain4.x, gain4.y, gain4.z, gain4.w};
                    float kn = 0.0f;
#pragma unroll
                    for (int step = 0; step < 4; ++step) {
                        const uint32_t word = inkling_quad_word(words[step], m >> 1);
                        const float key = (m & 1) != 0 ? __uint_as_float(word & 0xffff0000u)
                                                       : __uint_as_float(word << 16);
                        const float k_sc = __fmul_rn(key, gs[step]);
                        kv[m][step] = k_sc;
                        kn = __fadd_rn(kn, __fmul_rn(k_sc, k_sc));
                    }
                    norm_leaf[m] = kn;
                }
            }
            const float k_norm = sqrtf(inkling_quad_sum(inkling_quad_tree(norm_leaf)));
            const float k_gain = (qk_clip > 0.0f && k_norm > qk_clip) ? (qk_clip / k_norm) : 1.0f;
#pragma unroll
            for (int m = 0; m < 8; ++m) {
#pragma unroll
                for (int step = 0; step < 4; ++step) {
                    kv[m][step] = __fmul_rn(kv[m][step], k_gain);
                }
            }
            for (int r = 0; r < rows; ++r) {
                const int position = first_position + r;
                const int64_t run_begin_r = shared.k.row_run_begin[r];
                const int64_t run_end_r = shared.k.row_run_end[r];
                const bool in_row = valid && slot >= run_begin_r && slot < run_end_r;
                if (!__any_sync(0xffffffffu, in_row)) {
                    continue;
                }
                const int local_i = (int)(slot - run_begin_r);
                const int64_t distance = (int64_t)position - slot;
                float dots[INKLING_SM_MG];
#pragma unroll
                for (int g = 0; g < INKLING_SM_MG; ++g) {
                    const int qi = r * INKLING_SM_MG + g;
                    float leaf[8];
#pragma unroll
                    for (int m = 0; m < 8; ++m) {
                        const float4 q4 = *reinterpret_cast<const float4*>(
                            &shared.k.queries[u][(m * INKLING_SM_Q + qi) * 4]);
                        float dot = __fmul_rn(q4.x, kv[m][0]);
                        dot = __fadd_rn(dot, __fmul_rn(q4.y, kv[m][1]));
                        dot = __fadd_rn(dot, __fmul_rn(q4.z, kv[m][2]));
                        dot = __fadd_rn(dot, __fmul_rn(q4.w, kv[m][3]));
                        leaf[m] = dot;
                    }
                    dots[g] = inkling_quad_sum(inkling_quad_tree(leaf));
                }
                float biases[INKLING_SM_MG] = {0.0f, 0.0f, 0.0f};
                const bool biased = in_row && inkling_attn_biased(distance, rel_extent, d_rel);
                if (__any_sync(0xffffffffu, biased)) {
                    float bank[8];
#pragma unroll
                    for (int m = 0; m < 8; ++m) {
                        const int dim = u + 4 * m;
                        bank[m] = biased && dim < d_rel
                                      ? __bfloat162float(
                                            rel_bank[(uint64_t)dim * rel_extent + (uint64_t)distance])
                                      : 0.0f;
                    }
#pragma unroll
                    for (int g = 0; g < INKLING_SM_MG; ++g) {
                        const int qi = r * INKLING_SM_MG + g;
                        float leaf[8];
#pragma unroll
                        for (int m = 0; m < 8; ++m) {
                            leaf[m] = u + 4 * m < d_rel
                                          ? __fmul_rn(shared.k.rel[u][m * INKLING_SM_Q + qi], bank[m])
                                          : 0.0f;
                        }
                        biases[g] = inkling_quad_sum(inkling_quad_tree(leaf));
                    }
                }
                if (in_row && u < INKLING_SM_MG) {
                    const float d = u == 0 ? dots[0] : (u == 1 ? dots[1] : dots[2]);
                    float score = __fmul_rn(d, inv_dim);
                    if (biased) {
                        const float b = u == 0 ? biases[0] : (u == 1 ? biases[1] : biases[2]);
                        score = __fadd_rn(score, b);
                    }
                    shared.weights[r * INKLING_SM_MG + u][local_i] = score;
                }
            }
        }
        __syncthreads();

        auto issue_vtile = [&](int buf, int t0) {
            for (int c = (int)threadIdx.x; c < INKLING_WM_VTILE * 16; c += INKLING_SM_THREADS) {
                const int i = c >> 4;
                const int chunk = c & 15;
                const int s = t0 + i;
                if (s < slab_n) {
                    const __nv_bfloat16* src =
                        values + inkling_attn_cache_row(slab_first + s, cache_rows) * kv_stride +
                        kv_offset + chunk * 8;
                    const uint32_t dst =
                        (uint32_t)__cvta_generic_to_shared(&shared.v.v_stage[buf][i][chunk]);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
                }
            }
            asm volatile("cp.async.commit_group;");
        };

        const int num_tiles = (slab_n + INKLING_WM_VTILE - 1) / INKLING_WM_VTILE;
        if (num_tiles > 0) {
            issue_vtile(0, 0);
        }

        for (int qi = warp; qi < Q; qi += INKLING_SM_WARPS) {
            const int r = qi / INKLING_SM_MG;
            const int position = first_position + r;
            const int64_t first_r = inkling_attn_first_slot(position, window);
            const int count_r = (int)((int64_t)position - first_r + 1);
            const int n_r =
                count_r <= begin_off ? 0 : min(count_r - (int)begin_off, INKLING_PART_KEYS);
            float vals[8];
            float lmax = -INFINITY;
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int i0 = s * 32 + (int)lane;
                const int i1 = i0 + 128;
                vals[s] = i0 < n_r ? shared.weights[qi][i0] : -INFINITY;
                vals[s + 4] = i1 < n_r ? shared.weights[qi][i1] : -INFINITY;
                lmax = fmaxf(lmax, fmaxf(vals[s], vals[s + 4]));
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, offset));
            }
            const bool live = lmax != -INFINITY;
            float wsum[4];
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int i0 = s * 32 + (int)lane;
                const int i1 = i0 + 128;
                float t0 = 0.0f;
                float t1 = 0.0f;
                if (i0 < n_r) {
                    t0 = live ? expf(vals[s] - lmax) : 0.0f;
                    shared.weights[qi][i0] = t0;
                }
                if (i1 < n_r) {
                    t1 = live ? expf(vals[s + 4] - lmax) : 0.0f;
                    shared.weights[qi][i1] = t1;
                }
                float sum = t0 + t1;
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    sum += __shfl_xor_sync(0xffffffffu, sum, offset);
                }
                wsum[s] = sum;
            }
            if (lane == 0) {
                shared.largest[qi] = lmax;
                shared.total[qi] = (wsum[0] + wsum[1]) + (wsum[2] + wsum[3]);
            }
        }

        const int quarter = (int)(threadIdx.x >> 7);
        const int within = (int)(threadIdx.x & 127u);
        const int v_lane = within & 15;
        const int key_lane = within >> 4;
        const int dim0 = v_lane * INKLING_PART_DIMS;
        const int r0 = quarter;
        const int r1 = quarter + 4;
        const bool has_r0 = r0 < rows;
        const bool has_r1 = r1 < rows;

        int shift_r0 = 0, n_r0 = 0;
        if (has_r0) {
            const int pos0 = first_position + r0;
            const int64_t fr0 = inkling_attn_first_slot(pos0, window);
            shift_r0 = (int)(fr0 - first0);
            const int c0 = (int)((int64_t)pos0 - fr0 + 1);
            n_r0 = c0 <= begin_off ? 0 : min(c0 - (int)begin_off, INKLING_PART_KEYS);
        }
        int shift_r1 = 0, n_r1 = 0;
        if (has_r1) {
            const int pos1 = first_position + r1;
            const int64_t fr1 = inkling_attn_first_slot(pos1, window);
            shift_r1 = (int)(fr1 - first0);
            const int c1 = (int)((int64_t)pos1 - fr1 + 1);
            n_r1 = c1 <= begin_off ? 0 : min(c1 - (int)begin_off, INKLING_PART_KEYS);
        }

        float acc0[INKLING_SM_MG][INKLING_PART_DIMS];
        float acc1[INKLING_SM_MG][INKLING_PART_DIMS];
#pragma unroll
        for (int g = 0; g < INKLING_SM_MG; ++g) {
#pragma unroll
            for (int j = 0; j < INKLING_PART_DIMS; ++j) {
                acc0[g][j] = 0.0f;
                acc1[g][j] = 0.0f;
            }
        }

        int buf = 0;
        for (int tile = 0; tile < num_tiles; ++tile) {
            const int t0 = tile * INKLING_WM_VTILE;
            const int next_tile = tile + 1;
            if (next_tile < num_tiles) {
                issue_vtile(buf ^ 1, next_tile * INKLING_WM_VTILE);
                asm volatile("cp.async.wait_group 1;" ::: "memory");
            } else {
                asm volatile("cp.async.wait_all;" ::: "memory");
            }
            __syncthreads();
            const int tile_n = min(INKLING_WM_VTILE, slab_n - t0);
            if (value_norm != 0) {
                float inv[5];
#pragma unroll
                for (int u = 0; u < 5; ++u) {
                    const int i = warp + u * INKLING_SM_WARPS;
                    inv[u] = i < tile_n
                                 ? inkling_attn_value_inverse(
                                       reinterpret_cast<const __nv_bfloat16*>(
                                           &shared.v.v_stage[buf][i][0]),
                                       head_dim, eps)
                                 : 1.0f;
                }
                if (lane == 0) {
#pragma unroll
                    for (int u = 0; u < 5; ++u) {
                        const int i = warp + u * INKLING_SM_WARPS;
                        if (i < tile_n) {
                            shared.scales[t0 + i] = inv[u];
                        }
                    }
                }
            } else {
                if ((int)threadIdx.x < tile_n) {
                    shared.scales[t0 + threadIdx.x] = 1.0f;
                }
            }
            __syncthreads();

            if (has_r0 && n_r0 > 0) {
                const int lo = max(0, t0 - shift_r0);
                const int hi = min(n_r0, t0 + tile_n - shift_r0);
                int start = lo + ((key_lane - (lo & 7)) & 7);
                for (int local_i = start; local_i < hi; local_i += INKLING_PART_STRIDE) {
                    const int s = shift_r0 + local_i;
                    const uint4 packed = shared.v.v_stage[buf][s - t0][v_lane];
                    const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
                    const float sc = shared.scales[s];
                    float val[INKLING_PART_DIMS];
#pragma unroll
                    for (int pair = 0; pair < INKLING_PART_DIMS / 2; ++pair) {
                        const float2 two = __bfloat1622float2(pairs[pair]);
                        val[2 * pair] = two.x * sc;
                        val[2 * pair + 1] = two.y * sc;
                    }
#pragma unroll
                    for (int g = 0; g < INKLING_SM_MG; ++g) {
                        const float w = shared.weights[r0 * INKLING_SM_MG + g][local_i];
#pragma unroll
                        for (int j = 0; j < INKLING_PART_DIMS; ++j) {
                            acc0[g][j] = fmaf(w, val[j], acc0[g][j]);
                        }
                    }
                }
            }
            if (has_r1 && n_r1 > 0) {
                const int lo = max(0, t0 - shift_r1);
                const int hi = min(n_r1, t0 + tile_n - shift_r1);
                int start = lo + ((key_lane - (lo & 7)) & 7);
                for (int local_i = start; local_i < hi; local_i += INKLING_PART_STRIDE) {
                    const int s = shift_r1 + local_i;
                    const uint4 packed = shared.v.v_stage[buf][s - t0][v_lane];
                    const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
                    const float sc = shared.scales[s];
                    float val[INKLING_PART_DIMS];
#pragma unroll
                    for (int pair = 0; pair < INKLING_PART_DIMS / 2; ++pair) {
                        const float2 two = __bfloat1622float2(pairs[pair]);
                        val[2 * pair] = two.x * sc;
                        val[2 * pair + 1] = two.y * sc;
                    }
#pragma unroll
                    for (int g = 0; g < INKLING_SM_MG; ++g) {
                        const float w = shared.weights[r1 * INKLING_SM_MG + g][local_i];
#pragma unroll
                        for (int j = 0; j < INKLING_PART_DIMS; ++j) {
                            acc1[g][j] = fmaf(w, val[j], acc1[g][j]);
                        }
                    }
                }
            }
            __syncthreads();
            buf ^= 1;
        }

        auto flush_part = [&](int base_r, const float (&acc)[INKLING_SM_MG][INKLING_PART_DIMS],
                              bool has_r, int n_r) {
            if (has_r && n_r > 0) {
#pragma unroll
                for (int g = 0; g < INKLING_SM_MG; ++g) {
#pragma unroll
                    for (int j = 0; j < INKLING_PART_DIMS; ++j) {
                        shared.v_sums[quarter][key_lane][g][dim0 + j] = acc[g][j];
                    }
                }
            }
            __syncthreads();
            if (has_r) {
                const int r = base_r + quarter;
                const int dim = within;
                if (n_r > 0) {
#pragma unroll
                    for (int g = 0; g < INKLING_SM_MG; ++g) {
                        float sum = 0.0f;
#pragma unroll
                        for (int k = 0; k < INKLING_PART_STRIDE; ++k) {
                            sum += shared.v_sums[quarter][k][g][dim];
                        }
                        shared.part[r * INKLING_SM_MG + g][2 + dim] = sum;
                    }
                    if (within == 0) {
#pragma unroll
                        for (int g = 0; g < INKLING_SM_MG; ++g) {
                            shared.part[r * INKLING_SM_MG + g][0] =
                                shared.largest[r * INKLING_SM_MG + g];
                            shared.part[r * INKLING_SM_MG + g][1] =
                                shared.total[r * INKLING_SM_MG + g];
                        }
                    }
                } else {
#pragma unroll
                    for (int g = 0; g < INKLING_SM_MG; ++g) {
                        shared.part[r * INKLING_SM_MG + g][2 + dim] = 0.0f;
                    }
                    if (within == 0) {
#pragma unroll
                        for (int g = 0; g < INKLING_SM_MG; ++g) {
                            shared.part[r * INKLING_SM_MG + g][0] = -INFINITY;
                            shared.part[r * INKLING_SM_MG + g][1] = 0.0f;
                        }
                    }
                }
            }
            __syncthreads();
        };
        flush_part(0, acc0, has_r0, n_r0);
        if (rows > 4) {
            flush_part(4, acc1, has_r1, n_r1);
        }
    } else {
        for (int idx = (int)threadIdx.x; idx < Q * (2 + INKLING_MAX_HEAD_DIM);
             idx += INKLING_SM_THREADS) {
            const int qi = idx / (2 + INKLING_MAX_HEAD_DIM);
            const int c = idx - qi * (2 + INKLING_MAX_HEAD_DIM);
            shared.part[qi][c] = (c == 0) ? -INFINITY : 0.0f;
        }
        __syncthreads();
    }

    cluster.sync();

    const float* partials[INKLING_WINDOW_RUNS];
#pragma unroll
    for (int rk = 0; rk < INKLING_WINDOW_RUNS; ++rk) {
        partials[rk] = cluster.map_shared_rank(&shared.part[0][0], rk);
    }
    constexpr int WIDTH = 2 + INKLING_MAX_HEAD_DIM;
    for (int index = run + INKLING_WINDOW_RUNS * (int)threadIdx.x; index < Q * head_dim;
         index += INKLING_WINDOW_RUNS * INKLING_SM_THREADS) {
        const int qi = index / head_dim;
        const int dim = index - qi * head_dim;
        const int r = qi / INKLING_SM_MG;
        const int g = qi - r * INKLING_SM_MG;
        const int position = first_position + r;
        const int64_t first_r = inkling_attn_first_slot(position, window);
        const int count_r = (int)((int64_t)position - first_r + 1);
        const int runs_r = (count_r + INKLING_PART_KEYS - 1) / INKLING_PART_KEYS;
        const uint64_t row_head = (uint64_t)r * heads + head0 + g;
        float tops[INKLING_WINDOW_RUNS];
        float largest = sink != nullptr ? sink[row_head] : -INFINITY;
#pragma unroll
        for (int rk = 0; rk < INKLING_WINDOW_RUNS; ++rk) {
            tops[rk] = rk < runs_r ? partials[rk][qi * WIDTH] : -INFINITY;
            if (rk < runs_r) {
                largest = fmaxf(largest, tops[rk]);
            }
        }
        float denominator = 0.0f;
        float acc = 0.0f;
        if (largest != -INFINITY) {
            denominator = inkling_merge_open(sink, row_head, largest);
#pragma unroll
            for (int rk = 0; rk < INKLING_WINDOW_RUNS; ++rk) {
                if (rk < runs_r) {
                    const float factor = inkling_merge_factor(tops[rk], largest);
                    denominator = fmaf(partials[rk][qi * WIDTH + 1], factor, denominator);
                    acc = fmaf(partials[rk][qi * WIDTH + 2 + dim], factor, acc);
                }
            }
        }
        const __nv_bfloat16 val =
            inkling_merge_out(acc, denominator, gate[row_head * head_dim + dim]);
        if (out != nullptr) {
            out[row_head * head_dim + dim] = val;
        }
        if (out_f32 != nullptr) {
            out_f32[row_head * head_dim + dim] = __bfloat162float(val);
        }
    }
    cluster.sync();
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

// grid (ceil(heads / warps_per_block), rows), blockDim.x = 32 * warps_per_block (32..256):
// the query side of the tensor-core walk, inkling_attn_query's own result for every (row, head).
extern "C" __global__ void __launch_bounds__(256) inkling_attn_prepare(
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
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int head = (int)blockIdx.x * warps_per_block + (int)(threadIdx.x >> 5);
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

// ---------------------------------------------------------------------------------------------
// High-occupancy tensor-core prompt walk specialized for groups == 3, head_dim == 128
// (Inkling-2-Mini): 4 Q-tiles (64 query rows) x 3 query heads = 12 warps (384 threads) per block,
// KEYS = 64 keys per tile, q_hi resident in registers and q_lo in SMEM, relative bias scratch
// split into 2 phases of 6 warps (3 aliased onto k_tile + 3 in tail = 98,560 bytes total SMEM),
// exact expf online softmax with rescale-skip, and optional fused f32 output (eliminating
// inkling_widen_f32).
// Grid: (ceil(rows / 64), kv_heads, 1), block: 384 threads, dynamic SMEM: 98,560 bytes.
// ---------------------------------------------------------------------------------------------

#define INKLING_MMA_G3_NQ 4
#define INKLING_MMA_G3_Q (INKLING_MMA_G3_NQ * 16)
#define INKLING_MMA_G3_WARPS (INKLING_MMA_G3_NQ * 3)
#define INKLING_MMA_G3_KEYS 64
#define INKLING_MMA_G3_REL_STRIDE 84
#define INKLING_MMA_G3_REL_BYTES (4u * 16u * INKLING_MMA_G3_REL_STRIDE)

extern "C" __global__ void __launch_bounds__(INKLING_MMA_G3_WARPS * 32, 1) inkling_attn_mma_g3(
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
    float* __restrict__ out_f32,
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
    constexpr int D = INKLING_MAX_HEAD_DIM;
    constexpr int CH = D / 8;
    constexpr int KS = D / 16;
    constexpr int ROW = D * 2;
    constexpr int KEYS = INKLING_MMA_G3_KEYS;
    constexpr int WARPS = INKLING_MMA_G3_WARPS;
    constexpr int K_NT = KEYS / 8;
    constexpr int REL_SLOTS = 6;
    constexpr int REL_PHASES = 2;

    if (kv_heads <= 0 || heads != kv_heads * 3 || d_rel < 0 || d_rel > INKLING_MMA_D_REL) {
        return;
    }

    extern __shared__ __align__(128) uint8_t smem[];
    const int kv_head = blockIdx.y;
    const int block_qbase = (gridDim.x - 1 - blockIdx.x) * INKLING_MMA_G3_Q;
    const int block_count_q = min(INKLING_MMA_G3_Q, rows - block_qbase);
    if (block_count_q <= 0) {
        return;
    }
    const int block_pos0 = first_position + block_qbase;
    const int block_last = block_pos0 + block_count_q - 1;
    const int block_first = (window > 0 && block_pos0 + 1 > window) ? block_pos0 + 1 - window : 0;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int qt = warp / 3;
    const int gh = warp - qt * 3;
    const int head = kv_head * 3 + gh;
    const int qbase = block_qbase + qt * 16;
    const int count_q = max(0, min(16, rows - qbase));
    const int pos0 = first_position + qbase;
    const unsigned long long kv_row = (unsigned long long)kv_heads * D;
    const float score_scale = rsqrtf((float)D);

    uint32_t ptr = om_smem(smem);
    const uint32_t q_lo_smem = ptr;
    ptr += WARPS * 16u * ROW;           // 49,152 B
    const uint32_t k_tile = ptr;
    ptr += KEYS * ROW;                  // 16,384 B
    const uint32_t v_tile = ptr;
    ptr += KEYS * ROW;                  // 16,384 B
    const uint32_t gains_smem = ptr;
    ptr += 4u * KEYS;                   // 256 B
    const uint32_t scales_smem = ptr;
    ptr += 4u * KEYS;                   // 256 B
    const uint32_t rel_tail = ptr;      // 3 * 5,376 = 16,128 B (total = 98,560 B)

    auto load_kv = [&](uint32_t tile, const __nv_bfloat16* cache, int base) {
#pragma unroll
        for (int index = threadIdx.x; index < KEYS * CH; index += WARPS * 32) {
            const int row = index >> 4;
            const int chunk = index & 15;
            const int key = base + row;
            const bool valid = key <= block_last;
            const int row_index = cache_rows > 0 ? key % cache_rows : key;
            const __nv_bfloat16* source =
                valid ? cache + (unsigned long long)row_index * kv_row +
                            (unsigned long long)kv_head * D + chunk * 8
                      : cache;
            om_cp16(tile + shared_attn_off<D>(row, chunk), source, valid);
        }
    };
    auto load_per_key = [&](uint32_t dst, const float* __restrict__ table, int base) {
        if ((int)threadIdx.x < KEYS) {
            const int key = base + (int)threadIdx.x;
            const bool valid = key <= block_last;
            om_cp4(dst + 4u * threadIdx.x,
                   valid ? table + (uint64_t)kv_head * scale_stride + (key - scale_first) : table,
                   valid);
        }
    };

    load_kv(k_tile, keys, block_first);
    load_per_key(gains_smem, key_gains, block_first);
    om_commit();

    const int g = lane >> 2;
    const int t4 = lane & 3;
    const int matrix = lane >> 3;
    const int mrow = lane & 7;

    // Stage q_lo (BF16 remainder) into SMEM.
    for (int index = threadIdx.x; index < WARPS * 16 * CH; index += WARPS * 32) {
        const int row = index >> 4;
        const int chunk = index & 15;
        const int w = row >> 4;
        const int t = row & 15;
        const int w_qt = w / 3;
        const int w_gh = w - w_qt * 3;
        const int row_tok = block_qbase + w_qt * 16 + t;
        const int row_head = kv_head * 3 + w_gh;
        float x[8];
        if (row_tok < rows) {
            const float4* src = reinterpret_cast<const float4*>(
                q_ready + ((uint64_t)row_tok * heads + row_head) * D + chunk * 8);
            const float4 a = src[0];
            const float4 b = src[1];
            x[0] = a.x; x[1] = a.y; x[2] = a.z; x[3] = a.w;
            x[4] = b.x; x[5] = b.y; x[6] = b.z; x[7] = b.w;
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                x[e] = 0.0f;
            }
        }
        uint32_t lo[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const __nv_bfloat16 h0 = __float2bfloat16_rn(x[2 * e]);
            const __nv_bfloat16 h1 = __float2bfloat16_rn(x[2 * e + 1]);
            lo[e] = ob_pack(__float2bfloat16_rn(x[2 * e] - __bfloat162float(h0)),
                            __float2bfloat16_rn(x[2 * e + 1] - __bfloat162float(h1)));
        }
        const uint32_t offset = shared_attn_off<D>(row, chunk);
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(q_lo_smem + offset),
                     "r"(lo[0]), "r"(lo[1]), "r"(lo[2]), "r"(lo[3]));
    }

    // Load q_hi (BF16 head) directly into m16n8k16 A-fragment registers.
    uint32_t qh_reg[KS][4];
    {
        const float* p0 =
            g < count_q ? q_ready + ((uint64_t)(qbase + g) * heads + head) * D : nullptr;
        const float* p1 =
            (g + 8) < count_q ? q_ready + ((uint64_t)(qbase + g + 8) * heads + head) * D : nullptr;
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            const int k0 = ks * 16 + 2 * t4;
            const float2 v[4] = {
                p0 ? __ldg(reinterpret_cast<const float2*>(p0 + k0)) : make_float2(0.0f, 0.0f),
                p1 ? __ldg(reinterpret_cast<const float2*>(p1 + k0)) : make_float2(0.0f, 0.0f),
                p0 ? __ldg(reinterpret_cast<const float2*>(p0 + k0 + 8)) : make_float2(0.0f, 0.0f),
                p1 ? __ldg(reinterpret_cast<const float2*>(p1 + k0 + 8)) : make_float2(0.0f, 0.0f),
            };
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                qh_reg[ks][r] = ob_pack(__float2bfloat16_rn(v[r].x), __float2bfloat16_rn(v[r].y));
            }
        }
    }

    float o[CH][4];
#pragma unroll
    for (int n = 0; n < CH; ++n) {
        o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    }
    float m0 = (sink != nullptr && g < count_q)
                   ? sink[(uint64_t)(qbase + g) * heads + head]
                   : -INFINITY;
    float m1 = (sink != nullptr && g + 8 < count_q)
                   ? sink[(uint64_t)(qbase + g + 8) * heads + head]
                   : -INFINITY;
    float l0 = (sink != nullptr && g < count_q && t4 == 0) ? 1.0f : 0.0f;
    float l1 = (sink != nullptr && g + 8 < count_q && t4 == 0) ? 1.0f : 0.0f;

    const uint32_t a_row = (uint32_t)(warp * 16 + (lane & 15));
    const uint32_t a_chunk = (uint32_t)(lane >> 4);
    const int warp_last = pos0 + count_q - 1;
    const int warp_first = (window > 0 && pos0 + 1 > window) ? pos0 + 1 - window : 0;

    for (int base = block_first; base <= block_last; base += KEYS) {
        load_kv(v_tile, values, base);
        load_per_key(scales_smem, value_scales, base);
        om_commit();
        om_wait<1>();
        __syncthreads();

        const bool warp_active =
            count_q > 0 && base <= warp_last && (base + KEYS - 1) >= warp_first;
        float s[K_NT][4];
        if (warp_active) {
#pragma unroll
            for (int n = 0; n < K_NT; ++n) {
                s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
            }
#pragma unroll
            for (int ks = 0; ks < KS; ++ks) {
                uint32_t al[4];
                om_ldsm4(q_lo_smem + shared_attn_off<D>(a_row, 2u * ks + a_chunk),
                         al[0], al[1], al[2], al[3]);
#pragma unroll
                for (int np = 0; np < K_NT / 2; ++np) {
                    uint32_t b0, b1, b2, b3;
                    om_ldsm4(k_tile + shared_attn_off<D>(16u * np + 8u * (matrix >> 1) + mrow,
                                                         2u * ks + (matrix & 1)),
                             b0, b1, b2, b3);
                    ob_mma(s[2 * np], qh_reg[ks], b0, b1);
                    ob_mma(s[2 * np + 1], qh_reg[ks], b2, b3);
                    ob_mma(s[2 * np], al, b0, b1);
                    ob_mma(s[2 * np + 1], al, b2, b3);
                }
            }
        }

        const bool any_band = d_rel > 0 && block_pos0 - (base + KEYS - 1) < rel_extent;
        if (!any_band) {
            if (warp_active) {
                const bool interior = (count_q == 16) && (base + KEYS - 1 <= pos0) &&
                                      (window == 0 || base + window > pos0 + 15);
                float x0 = -INFINITY, x1 = -INFINITY;
                if (interior) {
#pragma unroll
                    for (int n = 0; n < K_NT; ++n) {
                        const int k0 = 8 * n + 2 * t4;
                        const float gn0 = inkling_mma_lds(gains_smem + 4u * k0);
                        const float gn1 = inkling_mma_lds(gains_smem + 4u * (k0 + 1));
                        s[n][0] = (s[n][0] * score_scale) * gn0;
                        s[n][1] = (s[n][1] * score_scale) * gn1;
                        s[n][2] = (s[n][2] * score_scale) * gn0;
                        s[n][3] = (s[n][3] * score_scale) * gn1;
                        x0 = fmaxf(x0, fmaxf(s[n][0], s[n][1]));
                        x1 = fmaxf(x1, fmaxf(s[n][2], s[n][3]));
                    }
                } else {
#pragma unroll
                    for (int n = 0; n < K_NT; ++n) {
                        const int k0 = 8 * n + 2 * t4;
                        const float gn0 = inkling_mma_lds(gains_smem + 4u * k0);
                        const float gn1 = inkling_mma_lds(gains_smem + 4u * (k0 + 1));
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const int row = g + 8 * (e >> 1);
                            const int key = base + k0 + (e & 1);
                            const int query = pos0 + row;
                            const bool live = row < count_q && key <= query &&
                                              (window == 0 || key + window > query);
                            s[n][e] = live ? (s[n][e] * score_scale) * ((e & 1) ? gn1 : gn0)
                                           : -INFINITY;
                        }
                        x0 = fmaxf(x0, fmaxf(s[n][0], s[n][1]));
                        x1 = fmaxf(x1, fmaxf(s[n][2], s[n][3]));
                    }
                }
                x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1));
                x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
                x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1));
                x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
                const float next0 = fmaxf(m0, x0);
                const float next1 = fmaxf(m1, x1);
                const float shift0 = next0 == -INFINITY ? 0.0f : next0;
                const float shift1 = next1 == -INFINITY ? 0.0f : next1;
                const float rescale0 = (m0 == next0) ? 1.0f : expf(m0 - shift0);
                const float rescale1 = (m1 == next1) ? 1.0f : expf(m1 - shift1);
                float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
                for (int n = 0; n < K_NT; ++n) {
                    s[n][0] = expf(s[n][0] - shift0);
                    s[n][1] = expf(s[n][1] - shift0);
                    s[n][2] = expf(s[n][2] - shift1);
                    s[n][3] = expf(s[n][3] - shift1);
                    sum0 += s[n][0] + s[n][1];
                    sum1 += s[n][2] + s[n][3];
                }
                l0 = l0 * rescale0 + sum0;
                l1 = l1 * rescale1 + sum1;
                m0 = next0;
                m1 = next1;
                if (__any_sync(0xffffffffu, rescale0 != 1.0f || rescale1 != 1.0f)) {
#pragma unroll
                    for (int n = 0; n < CH; ++n) {
                        o[n][0] *= rescale0;
                        o[n][1] *= rescale0;
                        o[n][2] *= rescale1;
                        o[n][3] *= rescale1;
                    }
                }
            }
        } else {
            // All warps finish reading k_tile before slots 0..2 reuse k_tile for rel bias.
            __syncthreads();
            uint32_t rel_hi[4], rel_lo[4];
            if (warp_active) {
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
#pragma unroll
                for (int reg = 0; reg < 4; ++reg) {
                    const float* pair = &r[reg & 1][2 * (reg >> 1)];
                    const __nv_bfloat16 h0 = __float2bfloat16_rn(pair[0]);
                    const __nv_bfloat16 h1 = __float2bfloat16_rn(pair[1]);
                    rel_hi[reg] = ob_pack(h0, h1);
                    rel_lo[reg] = ob_pack(__float2bfloat16_rn(pair[0] - __bfloat162float(h0)),
                                          __float2bfloat16_rn(pair[1] - __bfloat162float(h1)));
                }
            }
            const int my_phase = warp / REL_SLOTS;
            const int my_slot = warp - my_phase * REL_SLOTS;
            for (int ph_idx = 0; ph_idx < REL_PHASES; ++ph_idx) {
                if (ph_idx > 0) {
                    __syncthreads();
                }
                if (my_phase == ph_idx && warp_active) {
                    const bool band = pos0 - (base + KEYS - 1) < rel_extent;
                    const uint32_t rel_base =
                        my_slot < 3
                            ? k_tile + (uint32_t)my_slot * INKLING_MMA_G3_REL_BYTES
                            : rel_tail + (uint32_t)(my_slot - 3) * INKLING_MMA_G3_REL_BYTES;
                    if (band) {
                        const int d0 = pos0 - base - (KEYS - 1);
#pragma unroll
                        for (int nt = 0; nt < (16 + KEYS) / 8; ++nt) {
                            const int distance = d0 + 8 * nt + g;
                            const uint32_t b0 =
                                inkling_mma_bank(rel_bank, 2 * t4, distance, d_rel, rel_extent) |
                                (inkling_mma_bank(rel_bank, 2 * t4 + 1, distance, d_rel,
                                                  rel_extent)
                                 << 16);
                            const uint32_t b1 =
                                inkling_mma_bank(rel_bank, 2 * t4 + 8, distance, d_rel,
                                                 rel_extent) |
                                (inkling_mma_bank(rel_bank, 2 * t4 + 9, distance, d_rel,
                                                  rel_extent)
                                 << 16);
                            float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                            ob_mma(c, rel_hi, b0, b1);
                            ob_mma(c, rel_lo, b0, b1);
                            const uint32_t column = (uint32_t)(8 * nt + 2 * t4);
                            asm volatile("st.shared.v2.f32 [%0], {%1, %2};" ::"r"(
                                             rel_base +
                                             4u * ((uint32_t)g * INKLING_MMA_G3_REL_STRIDE +
                                                   column)),
                                         "f"(c[0]), "f"(c[1]));
                            asm volatile("st.shared.v2.f32 [%0], {%1, %2};" ::"r"(
                                             rel_base +
                                             4u * ((uint32_t)(g + 8) * INKLING_MMA_G3_REL_STRIDE +
                                                   column)),
                                         "f"(c[2]), "f"(c[3]));
                        }
                        __syncwarp();
                    }
                    float x0 = -INFINITY, x1 = -INFINITY;
#pragma unroll
                    for (int n = 0; n < K_NT; ++n) {
                        const int k0 = 8 * n + 2 * t4;
                        const float gn0 = inkling_mma_lds(gains_smem + 4u * k0);
                        const float gn1 = inkling_mma_lds(gains_smem + 4u * (k0 + 1));
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const int row = g + 8 * (e >> 1);
                            const int key_off = k0 + (e & 1);
                            const int key = base + key_off;
                            const int query = pos0 + row;
                            const bool live = row < count_q && key <= query &&
                                              (window == 0 || key + window > query);
                            if (live) {
                                float logit = (s[n][e] * score_scale) * ((e & 1) ? gn1 : gn0);
                                if (band) {
                                    logit += inkling_mma_lds(
                                        rel_base +
                                        4u * (uint32_t)(row * INKLING_MMA_G3_REL_STRIDE + row -
                                                        key_off + KEYS - 1));
                                }
                                s[n][e] = logit;
                            } else {
                                s[n][e] = -INFINITY;
                            }
                        }
                        x0 = fmaxf(x0, fmaxf(s[n][0], s[n][1]));
                        x1 = fmaxf(x1, fmaxf(s[n][2], s[n][3]));
                    }
                    x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1));
                    x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
                    x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1));
                    x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
                    const float next0 = fmaxf(m0, x0);
                    const float next1 = fmaxf(m1, x1);
                    const float shift0 = next0 == -INFINITY ? 0.0f : next0;
                    const float shift1 = next1 == -INFINITY ? 0.0f : next1;
                    const float rescale0 = (m0 == next0) ? 1.0f : expf(m0 - shift0);
                    const float rescale1 = (m1 == next1) ? 1.0f : expf(m1 - shift1);
                    float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
                    for (int n = 0; n < K_NT; ++n) {
                        s[n][0] = expf(s[n][0] - shift0);
                        s[n][1] = expf(s[n][1] - shift0);
                        s[n][2] = expf(s[n][2] - shift1);
                        s[n][3] = expf(s[n][3] - shift1);
                        sum0 += s[n][0] + s[n][1];
                        sum1 += s[n][2] + s[n][3];
                    }
                    l0 = l0 * rescale0 + sum0;
                    l1 = l1 * rescale1 + sum1;
                    m0 = next0;
                    m1 = next1;
                    if (__any_sync(0xffffffffu, rescale0 != 1.0f || rescale1 != 1.0f)) {
#pragma unroll
                        for (int n = 0; n < CH; ++n) {
                            o[n][0] *= rescale0;
                            o[n][1] *= rescale0;
                            o[n][2] *= rescale1;
                            o[n][3] *= rescale1;
                        }
                    }
                }
            }
        }

        om_wait<0>();
        __syncthreads();
        if (base + KEYS <= block_last) {
            load_kv(k_tile, keys, base + KEYS);
            load_per_key(gains_smem, key_gains, base + KEYS);
        }
        om_commit();

        if (warp_active) {
#pragma unroll
            for (int kk = 0; kk < KEYS / 16; ++kk) {
                uint32_t ph[4], pl[4];
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int n = 2 * kk + half;
                    const int k0 = 8 * n + 2 * t4;
                    const float vs0 = inkling_mma_lds(scales_smem + 4u * k0);
                    const float vs1 = inkling_mma_lds(scales_smem + 4u * (k0 + 1));
                    const float* p = s[n];
#pragma unroll
                    for (int pair = 0; pair < 2; ++pair) {
                        const float w0 = p[2 * pair] * vs0;
                        const float w1 = p[2 * pair + 1] * vs1;
                        const __nv_bfloat16 h0 = __float2bfloat16_rn(w0);
                        const __nv_bfloat16 h1 = __float2bfloat16_rn(w1);
                        ph[2 * half + pair] = ob_pack(h0, h1);
                        pl[2 * half + pair] =
                            ob_pack(__float2bfloat16_rn(w0 - __bfloat162float(h0)),
                                    __float2bfloat16_rn(w1 - __bfloat162float(h1)));
                    }
                }
#pragma unroll
                for (int c = 0; c < CH / 2; ++c) {
                    uint32_t b0, b1, b2, b3;
                    om_ldsm4_t(v_tile + shared_attn_off<D>(16u * kk + 8u * (matrix & 1) + mrow,
                                                           2u * c + (matrix >> 1)),
                               b0, b1, b2, b3);
                    ob_mma(o[2 * c], ph, b0, b1);
                    ob_mma(o[2 * c + 1], ph, b2, b3);
                    ob_mma(o[2 * c], pl, b0, b1);
                    ob_mma(o[2 * c + 1], pl, b2, b3);
                }
            }
        }
        __syncthreads();
    }
    om_wait<0>();

    if (count_q <= 0) {
        return;
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int row = g + 8 * half;
        if (row >= count_q) {
            continue;
        }
        const float denominator = half == 0 ? l0 : l1;
        const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
        const unsigned long long token = (unsigned long long)(qbase + row);
#pragma unroll
        for (int n = 0; n < CH; ++n) {
            const uint64_t at = (token * heads + head) * D + (uint64_t)(8 * n + 2 * t4);
            const float g0 =
                __fmul_rn(__fmul_rn(o[n][2 * half], inverse), inkling_attn_sigmoid(gate[at]));
            const float g1 = __fmul_rn(__fmul_rn(o[n][2 * half + 1], inverse),
                                       inkling_attn_sigmoid(gate[at + 1]));
            const __nv_bfloat162 packed = __floats2bfloat162_rn(g0, g1);
            if (out != nullptr) {
                *reinterpret_cast<__nv_bfloat162*>(out + at) = packed;
            }
            if (out_f32 != nullptr) {
                *reinterpret_cast<float2*>(out_f32 + at) = __bfloat1622float2(packed);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Tensor-core split-KV full decode attention for groups == 3, head_dim == 128, rows <= 8.
// Uses 2-plane BF16 Q (q_hi + q_lo) and 2-plane BF16 P (p_hi + p_lo) via m16n8k16 MMA so
// accuracy matches FP32 (~3.8e-6 max abs err) while staying 100% DRAM-bandwidth-bound across
// rows = 1..8 (eliminating scalar register spills and scaling flat from 1 to 8 verify rows).
// 50,688 B dynamic SMEM fits 2 blocks/SM on sm_121a (101,376 B / SM) with 0 spills.
// ---------------------------------------------------------------------------------------------
#define INKLING_TC_KEYS 64
#define INKLING_TC_WARPS 8
#define INKLING_TC_THREADS (INKLING_TC_WARPS * 32)
#define INKLING_TC_MAX_ROWS 8
#define INKLING_TC_MAX_Q 24

struct InklingScoredTcShared {
    alignas(16) uint4 kv[2][INKLING_TC_KEYS][16];             // 32,768 B
    alignas(16) uint4 q_hi[INKLING_TC_MAX_Q][16];             //  6,144 B
    alignas(16) uint4 q_lo[INKLING_TC_MAX_Q][16];             //  6,144 B
    union {
        alignas(16) float s_f32[16][INKLING_TC_KEYS];         //  4,096 B
        struct {
            alignas(16) __nv_bfloat16 p_hi[16][INKLING_TC_KEYS]; // 2,048 B
            alignas(16) __nv_bfloat16 p_lo[16][INKLING_TC_KEYS]; // 2,048 B
        };
    };
    alignas(16) float k_gains[INKLING_TC_KEYS];               //    256 B
    alignas(16) float v_scales[INKLING_TC_KEYS];              //    256 B
    alignas(16) float m_state[INKLING_TC_MAX_Q];              //     96 B
    alignas(16) float l_state[INKLING_TC_MAX_Q];              //     96 B
    alignas(16) float rescale[16];                            //     64 B
};

static_assert(sizeof(InklingScoredTcShared) == 49920,
              "SCORED_G3_TC_SHARED in crates/sparkflash-model-inkling2/src/device.rs");

__device__ __forceinline__ void inkling_tc_mma_m16n8k16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ int inkling_tc_swz(int row, int col_vec) {
    return col_vec ^ (row & 7);
}

// grid (kv_heads, ceil(widest / macro_keys), 1), INKLING_TC_THREADS threads,
// sizeof(InklingScoredTcShared) dynamic shared memory.
extern "C" __global__ void __launch_bounds__(INKLING_TC_THREADS, 2)
    inkling_attn_scored_part_g3_tc(
        const float* __restrict__ q,
        const __nv_bfloat16* __restrict__ keys,
        const __nv_bfloat16* __restrict__ values,
        const float* __restrict__ rel_proj,
        const __nv_bfloat16* __restrict__ rel_bank,
        const float* __restrict__ q_scale,
        const float* __restrict__ k_scale,
        const float* __restrict__ rel_scale,
        const float* __restrict__ key_gains,
        const float* __restrict__ value_scales,
        float* __restrict__ parts,
        int heads,
        int kv_heads,
        int head_dim,
        int groups,
        int rows,
        int first_position,
        int cache_rows,
        int rel_extent,
        int d_rel,
        float qk_clip,
        float rel_clip,
        float tau,
        const float* __restrict__ tau_rows,
        int value_norm,
        long long scale_first,
        int scale_stride,
        int macro_keys,
        int part_stride) {
    OPT_PDL_ENTRY();
    extern __shared__ __align__(16) unsigned char raw_smem[];
    InklingScoredTcShared& sm = *reinterpret_cast<InklingScoredTcShared*>(raw_smem);

    const int kv_head = blockIdx.x;
    const int macro_run = blockIdx.y;
    if (kv_head >= kv_heads || rows <= 0 || rows > INKLING_TC_MAX_ROWS || head_dim != 128 ||
        groups != 3 || d_rel < 0 || d_rel > 16 || macro_keys <= 0) {
        return;
    }

    const int64_t macro_begin = (int64_t)macro_run * macro_keys;
    const int64_t max_count = (int64_t)first_position + rows;
    if (macro_begin >= max_count) {
        return;
    }
    const int macro_n = (int)min(max_count - macro_begin, (int64_t)macro_keys);

    const int Q = rows * 3;
    const int head0 = kv_head * 3;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const uint64_t kv_stride = (uint64_t)kv_heads * 128;
    const uint64_t kv_offset = (uint64_t)kv_head * 128;
    const float inv_dim = 1.0f / sqrtf(128.0f);

    auto issue_kv = [&](int buf, const __nv_bfloat16* base_ptr, int k0, int n_tile) {
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int idx = tid + it * INKLING_TC_THREADS;
            const int r = idx >> 4;
            const int c = idx & 15;
            const uint32_t dst =
                (uint32_t)__cvta_generic_to_shared(&sm.kv[buf][r][inkling_tc_swz(r, c)]);
            if (r < n_tile) {
                const __nv_bfloat16* src =
                    base_ptr +
                    inkling_attn_cache_row(macro_begin + k0 + r, cache_rows) * kv_stride +
                    kv_offset + c * 8;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
            } else {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;" ::"r"(dst),
                             "l"(base_ptr));
            }
        }
        asm volatile("cp.async.commit_group;");
    };

    issue_kv(0, keys, 0, min(INKLING_TC_KEYS, macro_n));

    for (int qi = warp; qi < Q; qi += INKLING_TC_WARPS) {
        const int r = qi / 3;
        const int g = qi - r * 3;
        const int head = head0 + g;
        float qv[4];
        float rv;
        inkling_attn_query(q + ((uint64_t)r * heads + head) * 128,
                           rel_proj + (uint64_t)r * heads * d_rel, q_scale, rel_scale, head, 128,
                           d_rel, qk_clip, rel_clip, tau_rows != nullptr ? tau_rows[r] : tau, qv,
                           rv);
        if (k_scale != nullptr) {
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                qv[s] *= k_scale[(uint64_t)kv_head * 128 + lane + s * 32];
            }
        }
        __nv_bfloat16* dst_hi = reinterpret_cast<__nv_bfloat16*>(&sm.q_hi[qi][0]);
        __nv_bfloat16* dst_lo = reinterpret_cast<__nv_bfloat16*>(&sm.q_lo[qi][0]);
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int d = lane + s * 32;
            const __nv_bfloat16 hi = __float2bfloat16_rn(qv[s]);
            const __nv_bfloat16 lo = __float2bfloat16_rn(qv[s] - __bfloat162float(hi));
            dst_hi[inkling_tc_swz(qi, d >> 3) * 8 + (d & 7)] = hi;
            dst_lo[inkling_tc_swz(qi, d >> 3) * 8 + (d & 7)] = lo;
        }
    }
    if (tid < INKLING_TC_MAX_Q) {
        sm.m_state[tid] = -INFINITY;
        sm.l_state[tid] = 0.0f;
    }
    __syncthreads();

    float O0[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    float O1[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    const int num_tiles = (macro_n + INKLING_TC_KEYS - 1) / INKLING_TC_KEYS;
    const int n_mtiles = (Q > 16) ? 2 : 1;

    const float* kg_base = key_gains + (uint64_t)kv_head * scale_stride - scale_first + macro_begin;
    const float* vs_base =
        value_scales + (uint64_t)kv_head * scale_stride - scale_first + macro_begin;
    const int grp = lane >> 2;
    const int t4 = lane & 3;
    const bool macro_may_bias =
        (d_rel > 0 && rel_extent > 0) &&
        ((int64_t)first_position - (macro_begin + macro_n - 1) < (int64_t)rel_extent);

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int k0 = tile * INKLING_TC_KEYS;
        const int tile_n = min(INKLING_TC_KEYS, macro_n - k0);
        issue_kv(1, values, k0, tile_n);
        if (tid < tile_n) {
            sm.k_gains[tid] = kg_base[k0 + tid] * inv_dim;
            sm.v_scales[tid] = value_norm != 0 ? vs_base[k0 + tid] : 1.0f;
        }
        asm volatile("cp.async.wait_group 1;" ::: "memory");
        __syncthreads();

        const bool tile_may_bias =
            macro_may_bias &&
            ((int64_t)first_position - (macro_begin + k0 + tile_n - 1) < (int64_t)rel_extent);

        for (int mt = 0; mt < n_mtiles; ++mt) {
            const int q0 = mt * 16;
            const int q_cnt = min(16, Q - q0);

            float s_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float s_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const int k_row = warp * 8 + (lane & 7);
            const int q_r = q0 + (lane & 15);
            const int q_r_safe = q_r < Q ? q_r : 0;

#pragma unroll
            for (int kk = 0; kk < 128; kk += 16) {
                const int cv = (kk >> 3) + ((lane >> 4) & 1);
                uint32_t a_hi[4], a_lo[4], b_k[2];
                const uint32_t addr_hi = (uint32_t)__cvta_generic_to_shared(
                    &sm.q_hi[q_r_safe][inkling_tc_swz(q_r_safe, cv)]);
                const uint32_t addr_lo = (uint32_t)__cvta_generic_to_shared(
                    &sm.q_lo[q_r_safe][inkling_tc_swz(q_r_safe, cv)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                             : "r"(addr_hi));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                             : "r"(addr_lo));
                const int k_cv = (kk >> 3) + ((lane >> 3) & 1);
                const uint32_t addr_k = (uint32_t)__cvta_generic_to_shared(
                    &sm.kv[0][k_row][inkling_tc_swz(k_row, k_cv)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                             : "=r"(b_k[0]), "=r"(b_k[1])
                             : "r"(addr_k));
                inkling_tc_mma_m16n8k16(s_hi, a_hi[0], a_hi[1], a_hi[2], a_hi[3], b_k[0], b_k[1]);
                inkling_tc_mma_m16n8k16(s_lo, a_lo[0], a_lo[1], a_lo[2], a_lo[3], b_k[0], b_k[1]);
            }

            const int kc0 = warp * 8 + 2 * t4;
            const int kc1 = kc0 + 1;
            const float g0 = kc0 < tile_n ? sm.k_gains[kc0] : 0.0f;
            const float g1 = kc1 < tile_n ? sm.k_gains[kc1] : 0.0f;
            if (grp < q_cnt) {
                sm.s_f32[grp][kc0] = (s_hi[0] + s_lo[0]) * g0;
                sm.s_f32[grp][kc1] = (s_hi[1] + s_lo[1]) * g1;
            }
            if (grp + 8 < q_cnt) {
                sm.s_f32[grp + 8][kc0] = (s_hi[2] + s_lo[2]) * g0;
                sm.s_f32[grp + 8][kc1] = (s_hi[3] + s_lo[3]) * g1;
            }
            __syncthreads();

            if (mt == n_mtiles - 1 && tile + 1 < num_tiles) {
                issue_kv(0, keys, (tile + 1) * INKLING_TC_KEYS,
                         min(INKLING_TC_KEYS, macro_n - (tile + 1) * INKLING_TC_KEYS));
            }

            const int i0 = lane;
            const int i1 = lane + 32;
            const float vs0 = i0 < tile_n ? sm.v_scales[i0] : 0.0f;
            const float vs1 = i1 < tile_n ? sm.v_scales[i1] : 0.0f;
            float wp0[2] = {0.0f, 0.0f};
            float wp1[2] = {0.0f, 0.0f};
#pragma unroll
            for (int it = 0; it < 2; ++it) {
                const int qr = warp + it * INKLING_TC_WARPS;
                if (qr < q_cnt) {
                    const int qi = q0 + qr;
                    const int r_row = qi / 3;
                    const int64_t pos_r = (int64_t)(first_position + r_row);
                    const int64_t s0 = macro_begin + k0 + i0;
                    const int64_t s1 = macro_begin + k0 + i1;
                    const bool ok0 = i0 < tile_n && s0 <= pos_r;
                    const bool ok1 = i1 < tile_n && s1 <= pos_r;
                    float v0 = ok0 ? sm.s_f32[qr][i0] : -INFINITY;
                    float v1 = ok1 ? sm.s_f32[qr][i1] : -INFINITY;
                    if (tile_may_bias) {
                        const int head_qi = head0 + (qi - r_row * 3);
                        const float tau_r = tau_rows != nullptr ? tau_rows[r_row] : tau;
                        const float use_tau = tau_r > 0.0f ? tau_r : 1.0f;
                        float rv_lane =
                            lane < d_rel
                                ? rel_proj[((uint64_t)r_row * heads + head_qi) * d_rel + lane]
                                : 0.0f;
                        if (rel_scale != nullptr && lane < d_rel) {
                            rv_lane = __fmul_rn(rv_lane,
                                                rel_scale[(uint64_t)head_qi * d_rel + lane]);
                        }
                        const float r_norm = sqrtf(inkling_attn_warp_sum(__fmul_rn(rv_lane, rv_lane)));
                        float r_gain = 1.0f;
                        if (rel_clip > 0.0f && r_norm > rel_clip) {
                            r_gain = rel_clip / r_norm;
                        }
                        rv_lane = __fmul_rn(rv_lane, __fmul_rn(r_gain, use_tau));
                        const int64_t dist0 = pos_r - s0;
                        const int64_t dist1 = pos_r - s1;
                        const bool b0_ok = ok0 && inkling_attn_biased(dist0, rel_extent, d_rel);
                        const bool b1_ok = ok1 && inkling_attn_biased(dist1, rel_extent, d_rel);
                        float b0 = 0.0f;
                        float b1 = 0.0f;
                        for (int d = 0; d < d_rel; ++d) {
                            const float qrv = __shfl_sync(0xffffffffu, rv_lane, d);
                            if (b0_ok) {
                                b0 = fmaf(
                                    qrv,
                                    __bfloat162float(__ldg(
                                        &rel_bank[(uint64_t)d * rel_extent + (uint64_t)dist0])),
                                    b0);
                            }
                            if (b1_ok) {
                                b1 = fmaf(
                                    qrv,
                                    __bfloat162float(__ldg(
                                        &rel_bank[(uint64_t)d * rel_extent + (uint64_t)dist1])),
                                    b1);
                            }
                        }
                        if (b0_ok) {
                            v0 += b0;
                        }
                        if (b1_ok) {
                            v1 += b1;
                        }
                    }
                    float tmax = fmaxf(v0, v1);
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, off));
                    }
                    const float m_old = sm.m_state[qi];
                    const float m_new = fmaxf(m_old, tmax);
                    const float alpha =
                        (m_old == -INFINITY) ? 0.0f : (m_new == m_old ? 1.0f : expf(m_old - m_new));
                    const float p0 = (ok0 && m_new != -INFINITY) ? expf(v0 - m_new) : 0.0f;
                    const float p1 = (ok1 && m_new != -INFINITY) ? expf(v1 - m_new) : 0.0f;
                    float psum = p0 + p1;
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        psum += __shfl_xor_sync(0xffffffffu, psum, off);
                    }
                    if (lane == 0) {
                        sm.m_state[qi] = m_new;
                        sm.l_state[qi] = sm.l_state[qi] * alpha + psum;
                        sm.rescale[qr] = alpha;
                    }
                    wp0[it] = p0 * vs0;
                    wp1[it] = p1 * vs1;
                }
            }

            if (mt == 0) {
                if (n_mtiles == 1 && tile + 1 < num_tiles) {
                    asm volatile("cp.async.wait_group 1;" ::: "memory");
                } else {
                    asm volatile("cp.async.wait_all;" ::: "memory");
                }
            }
            __syncthreads();

#pragma unroll
            for (int it = 0; it < 2; ++it) {
                const int qr = warp + it * INKLING_TC_WARPS;
                if (qr < q_cnt) {
                    const __nv_bfloat16 h0 = __float2bfloat16_rn(wp0[it]);
                    const __nv_bfloat16 h1 = __float2bfloat16_rn(wp1[it]);
                    sm.p_hi[qr][i0] = h0;
                    sm.p_hi[qr][i1] = h1;
                    sm.p_lo[qr][i0] = __float2bfloat16_rn(wp0[it] - __bfloat162float(h0));
                    sm.p_lo[qr][i1] = __float2bfloat16_rn(wp1[it] - __bfloat162float(h1));
                }
            }
            __syncthreads();

            float (&O)[2][4] = (mt == 0) ? O0 : O1;
            const float a_top = grp < q_cnt ? sm.rescale[grp] : 0.0f;
            const float a_bot = (grp + 8) < q_cnt ? sm.rescale[grp + 8] : 0.0f;
            if (a_top != 1.0f) {
                O[0][0] *= a_top;
                O[0][1] *= a_top;
                O[1][0] *= a_top;
                O[1][1] *= a_top;
            }
            if (a_bot != 1.0f) {
                O[0][2] *= a_bot;
                O[0][3] *= a_bot;
                O[1][2] *= a_bot;
                O[1][3] *= a_bot;
            }

#pragma unroll
            for (int kk = 0; kk < INKLING_TC_KEYS; kk += 16) {
                const int pr = lane & 15;
                const int pc = kk + ((lane >> 4) & 1) * 8;
                uint32_t a_hi[4], a_lo[4];
                const uint32_t addr_hi = (uint32_t)__cvta_generic_to_shared(&sm.p_hi[pr][pc]);
                const uint32_t addr_lo = (uint32_t)__cvta_generic_to_shared(&sm.p_lo[pr][pc]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                             : "r"(addr_hi));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                             : "r"(addr_lo));
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int vr = kk + (lane & 15);
                    const int vcv = 2 * warp + nt;
                    uint32_t b_v[2];
                    const uint32_t addr_v = (uint32_t)__cvta_generic_to_shared(
                        &sm.kv[1][vr][inkling_tc_swz(vr, vcv)]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                                 : "=r"(b_v[0]), "=r"(b_v[1])
                                 : "r"(addr_v));
                    inkling_tc_mma_m16n8k16(O[nt], a_hi[0], a_hi[1], a_hi[2], a_hi[3], b_v[0],
                                            b_v[1]);
                    inkling_tc_mma_m16n8k16(O[nt], a_lo[0], a_lo[1], a_lo[2], a_lo[3], b_v[0],
                                            b_v[1]);
                }
            }
            __syncthreads();
        }
    }

    const uint64_t head_stride = (uint64_t)part_stride * 130;
    auto store_tile = [&](int mt, const float (&O)[2][4]) {
        const int q0 = mt * 16;
        const int q_cnt = min(16, Q - q0);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int c0 = warp * 16 + nt * 8 + 2 * t4;
            if (grp < q_cnt) {
                const int qi = q0 + grp;
                const int r = qi / 3;
                const int g = qi - r * 3;
                float* ob = parts + ((uint64_t)r * heads + head0 + g) * head_stride +
                            (uint64_t)macro_run * 130;
                ob[2 + c0] = O[nt][0];
                ob[2 + c0 + 1] = O[nt][1];
                if (warp == 0 && nt == 0 && t4 == 0) {
                    ob[0] = sm.m_state[qi];
                    ob[1] = sm.l_state[qi];
                }
            }
            if (grp + 8 < q_cnt) {
                const int qi = q0 + grp + 8;
                const int r = qi / 3;
                const int g = qi - r * 3;
                float* ob = parts + ((uint64_t)r * heads + head0 + g) * head_stride +
                            (uint64_t)macro_run * 130;
                ob[2 + c0] = O[nt][2];
                ob[2 + c0 + 1] = O[nt][3];
                if (warp == 0 && nt == 0 && t4 == 0) {
                    ob[0] = sm.m_state[qi];
                    ob[1] = sm.l_state[qi];
                }
            }
        }
    };
    store_tile(0, O0);
    if (n_mtiles > 1) {
        store_tile(1, O1);
    }
}

// ---------------------------------------------------------------------------------------------
// Tensor-core aligned 64-key sliding-window partial attention for groups == 3, head_dim == 128,
// rows = 1..8. Each block processes one globally aligned 64-key tile [64*(first_macro+blockIdx.y),
// +64) for one KV head across all rows, computing k_gains and v_scales on the fly from staged
// SMEM and writing standard (m, l, O) partials for inkling_attn_merge_f32 (with run_keys =
// -num_macros). Global 64-key alignment guarantees exact row-count invariance (rows == 1 produces
// bit-identical results to rows == 2..8 for every token).
// Reuses InklingScoredTcShared (49,920 B -> 2 blocks/SM on sm_121a).
// Grid (kv_heads, num_macros, 1), 256 threads, sizeof(InklingScoredTcShared) dynamic SMEM.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(INKLING_TC_THREADS, 2)
    inkling_attn_window_g3_tc(
        const float* __restrict__ q,
        const __nv_bfloat16* __restrict__ keys,
        const __nv_bfloat16* __restrict__ values,
        const float* __restrict__ rel_proj,
        const __nv_bfloat16* __restrict__ rel_bank,
        const float* __restrict__ q_scale,
        const float* __restrict__ k_scale,
        const float* __restrict__ rel_scale,
        float* __restrict__ parts,
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
        int value_norm,
        float eps,
        int part_stride) {
    OPT_PDL_ENTRY();
    extern __shared__ __align__(16) unsigned char raw_smem[];
    InklingScoredTcShared& sm = *reinterpret_cast<InklingScoredTcShared*>(raw_smem);

    const int kv_head = blockIdx.x;
    const int macro_idx = blockIdx.y;
    if (kv_head >= kv_heads || rows <= 0 || rows > INKLING_TC_MAX_ROWS || head_dim != 128 ||
        groups != 3 || d_rel < 0 || d_rel > 16 || window <= 0) {
        return;
    }

    const int64_t first0 = inkling_attn_first_slot(first_position, window);
    const int64_t first_macro = first0 >> 6;
    const int64_t tile_begin = (first_macro + macro_idx) << 6;
    const int64_t max_pos_end = (int64_t)first_position + rows;

    const int Q = rows * 3;
    const int head0 = kv_head * 3;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp = lane >> 2;
    const int t4 = lane & 3;
    const uint64_t kv_stride = (uint64_t)kv_heads * 128;
    const uint64_t kv_offset = (uint64_t)kv_head * 128;
    const float inv_dim = 1.0f / sqrtf(128.0f);

    auto issue_kv = [&](int buf, const __nv_bfloat16* base_ptr) {
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int idx = tid + it * INKLING_TC_THREADS;
            const int r = idx >> 4;
            const int c = idx & 15;
            const int64_t pos = tile_begin + r;
            const uint32_t dst =
                (uint32_t)__cvta_generic_to_shared(&sm.kv[buf][r][inkling_tc_swz(r, c)]);
            if (pos >= first0 && pos < max_pos_end) {
                const __nv_bfloat16* src =
                    base_ptr + inkling_attn_cache_row(pos, cache_rows) * kv_stride + kv_offset +
                    c * 8;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
            } else {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;" ::"r"(dst),
                             "l"(base_ptr));
            }
        }
        asm volatile("cp.async.commit_group;");
    };

    // Issue K then V immediately so both overlap with Q load and K-MMA.
    issue_kv(0, keys);
    issue_kv(1, values);

    for (int qi = warp; qi < Q; qi += INKLING_TC_WARPS) {
        const int r = qi / 3;
        const int g = qi - r * 3;
        const int head = head0 + g;
        float qv[4];
        float rv;
        inkling_attn_query(q + ((uint64_t)r * heads + head) * 128,
                           rel_proj + (uint64_t)r * heads * d_rel, q_scale, rel_scale, head, 128,
                           d_rel, qk_clip, rel_clip, tau_rows != nullptr ? tau_rows[r] : tau, qv,
                           rv);
        if (k_scale != nullptr) {
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                qv[s] *= k_scale[(uint64_t)kv_head * 128 + lane + s * 32];
            }
        }
        __nv_bfloat16* dst_hi = reinterpret_cast<__nv_bfloat16*>(&sm.q_hi[qi][0]);
        __nv_bfloat16* dst_lo = reinterpret_cast<__nv_bfloat16*>(&sm.q_lo[qi][0]);
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int d = lane + s * 32;
            const __nv_bfloat16 hi = __float2bfloat16_rn(qv[s]);
            const __nv_bfloat16 lo = __float2bfloat16_rn(qv[s] - __bfloat162float(hi));
            dst_hi[inkling_tc_swz(qi, d >> 3) * 8 + (d & 7)] = hi;
            dst_lo[inkling_tc_swz(qi, d >> 3) * 8 + (d & 7)] = lo;
        }
    }

    // Wait for K tile (group 0); V tile (group 1) continues transferring in flight.
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncthreads();

    // Compute per-key qk_clip gains on the fly from sm.kv[0] (8 keys per warp).
    float ks_lane[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    if (k_scale != nullptr) {
        const float* ks = k_scale + (uint64_t)kv_head * 128 + lane * 4;
        ks_lane[0] = ks[0];
        ks_lane[1] = ks[1];
        ks_lane[2] = ks[2];
        ks_lane[3] = ks[3];
    }
    for (int k_it = 0; k_it < 8; ++k_it) {
        const int kr = warp * 8 + k_it;
        const uint2 w2 =
            reinterpret_cast<const uint2*>(&sm.kv[0][kr][inkling_tc_swz(kr, lane >> 1)])[lane & 1];
        const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w2.x));
        const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w2.y));
        const float k0 = f0.x * ks_lane[0];
        const float k1 = f0.y * ks_lane[1];
        const float k2 = f1.x * ks_lane[2];
        const float k3 = f1.y * ks_lane[3];
        float sq = fmaf(k0, k0, fmaf(k1, k1, fmaf(k2, k2, k3 * k3)));
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sq += __shfl_xor_sync(0xffffffffu, sq, off);
        }
        if (lane == 0) {
            const float kn = sqrtf(sq);
            const float kg = (qk_clip > 0.0f && kn > qk_clip) ? (qk_clip / kn) : 1.0f;
            sm.k_gains[kr] = kg * inv_dim;
        }
    }
    __syncthreads();

    const int n_mtiles = (Q > 16) ? 2 : 1;
    const uint64_t head_stride = (uint64_t)part_stride * 130;

    // Wait for V tile once and compute v_scales once before the mtile loop.
    // To overlap Q@K^T with V arrival, run mt=0 Q@K^T before waiting on V when n_mtiles==1,
    // or wait on V right before softmax of mt=0.
    bool v_ready = false;

    for (int mt = 0; mt < n_mtiles; ++mt) {
        const int q0 = mt * 16;
        const int q_cnt = min(16, Q - q0);

        float s_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float s_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const int k_row = warp * 8 + (lane & 7);
        const int q_r = q0 + (lane & 15);
        const int q_r_safe = q_r < Q ? q_r : 0;

#pragma unroll
        for (int kk = 0; kk < 128; kk += 16) {
            const int cv = (kk >> 3) + ((lane >> 4) & 1);
            uint32_t a_hi[4], a_lo[4], b_k[2];
            const uint32_t addr_hi = (uint32_t)__cvta_generic_to_shared(
                &sm.q_hi[q_r_safe][inkling_tc_swz(q_r_safe, cv)]);
            const uint32_t addr_lo = (uint32_t)__cvta_generic_to_shared(
                &sm.q_lo[q_r_safe][inkling_tc_swz(q_r_safe, cv)]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                         : "r"(addr_hi));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                         : "r"(addr_lo));
            const int k_cv = (kk >> 3) + ((lane >> 3) & 1);
            const uint32_t addr_k = (uint32_t)__cvta_generic_to_shared(
                &sm.kv[0][k_row][inkling_tc_swz(k_row, k_cv)]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                         : "=r"(b_k[0]), "=r"(b_k[1])
                         : "r"(addr_k));
            inkling_tc_mma_m16n8k16(s_hi, a_hi[0], a_hi[1], a_hi[2], a_hi[3], b_k[0], b_k[1]);
            inkling_tc_mma_m16n8k16(s_lo, a_lo[0], a_lo[1], a_lo[2], a_lo[3], b_k[0], b_k[1]);
        }

        const int kc0 = warp * 8 + 2 * t4;
        const int kc1 = kc0 + 1;
        const float g0 = sm.k_gains[kc0];
        const float g1 = sm.k_gains[kc1];
        if (grp < q_cnt) {
            sm.s_f32[grp][kc0] = (s_hi[0] + s_lo[0]) * g0;
            sm.s_f32[grp][kc1] = (s_hi[1] + s_lo[1]) * g1;
        }
        if (grp + 8 < q_cnt) {
            sm.s_f32[grp + 8][kc0] = (s_hi[2] + s_lo[2]) * g0;
            sm.s_f32[grp + 8][kc1] = (s_hi[3] + s_lo[3]) * g1;
        }

        if (!v_ready) {
            asm volatile("cp.async.wait_all;" ::: "memory");
            __syncthreads();
            if (value_norm != 0) {
                for (int k_it = 0; k_it < 8; ++k_it) {
                    const int kr = warp * 8 + k_it;
                    const uint2 w2 = reinterpret_cast<const uint2*>(
                        &sm.kv[1][kr][inkling_tc_swz(kr, lane >> 1)])[lane & 1];
                    const float2 f0 =
                        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w2.x));
                    const float2 f1 =
                        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w2.y));
                    float sq = fmaf(f0.x, f0.x, fmaf(f0.y, f0.y, fmaf(f1.x, f1.x, f1.y * f1.y)));
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        sq += __shfl_xor_sync(0xffffffffu, sq, off);
                    }
                    if (lane == 0) {
                        sm.v_scales[kr] = rsqrtf(sq * (1.0f / 128.0f) + eps);
                    }
                }
            } else if (tid < INKLING_TC_KEYS) {
                sm.v_scales[tid] = 1.0f;
            }
            __syncthreads();
            v_ready = true;
        } else {
            __syncthreads();
        }

        const int i0 = lane;
        const int i1 = lane + 32;
        const float vs0 = sm.v_scales[i0];
        const float vs1 = sm.v_scales[i1];
        float wp0[2] = {0.0f, 0.0f};
        float wp1[2] = {0.0f, 0.0f};
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int qr = warp + it * INKLING_TC_WARPS;
            if (qr < q_cnt) {
                const int qi = q0 + qr;
                const int r_row = qi / 3;
                const int64_t pos_r = (int64_t)(first_position + r_row);
                const int64_t first_r = inkling_attn_first_slot((int)pos_r, window);
                const int64_t s0 = tile_begin + i0;
                const int64_t s1 = tile_begin + i1;
                const bool ok0 = s0 >= first_r && s0 <= pos_r;
                const bool ok1 = s1 >= first_r && s1 <= pos_r;
                float v0 = ok0 ? sm.s_f32[qr][i0] : -INFINITY;
                float v1 = ok1 ? sm.s_f32[qr][i1] : -INFINITY;
                if (d_rel > 0 && rel_extent > 0) {
                    const int head_qi = head0 + (qi - r_row * 3);
                    const float tau_r = tau_rows != nullptr ? tau_rows[r_row] : tau;
                    const float use_tau = tau_r > 0.0f ? tau_r : 1.0f;
                    float rv_lane =
                        lane < d_rel
                            ? rel_proj[((uint64_t)r_row * heads + head_qi) * d_rel + lane]
                            : 0.0f;
                    if (rel_scale != nullptr && lane < d_rel) {
                        rv_lane =
                            __fmul_rn(rv_lane, rel_scale[(uint64_t)head_qi * d_rel + lane]);
                    }
                    const float r_norm =
                        sqrtf(inkling_attn_warp_sum(__fmul_rn(rv_lane, rv_lane)));
                    float r_gain = 1.0f;
                    if (rel_clip > 0.0f && r_norm > rel_clip) {
                        r_gain = rel_clip / r_norm;
                    }
                    rv_lane = __fmul_rn(rv_lane, __fmul_rn(r_gain, use_tau));
                    const int64_t dist0 = pos_r - s0;
                    const int64_t dist1 = pos_r - s1;
                    const bool b0_ok = ok0 && inkling_attn_biased(dist0, rel_extent, d_rel);
                    const bool b1_ok = ok1 && inkling_attn_biased(dist1, rel_extent, d_rel);
                    float b0 = 0.0f;
                    float b1 = 0.0f;
                    for (int d = 0; d < d_rel; ++d) {
                        const float qrv = __shfl_sync(0xffffffffu, rv_lane, d);
                        if (b0_ok) {
                            b0 = fmaf(qrv,
                                      __bfloat162float(__ldg(
                                          &rel_bank[(uint64_t)d * rel_extent + (uint64_t)dist0])),
                                      b0);
                        }
                        if (b1_ok) {
                            b1 = fmaf(qrv,
                                      __bfloat162float(__ldg(
                                          &rel_bank[(uint64_t)d * rel_extent + (uint64_t)dist1])),
                                      b1);
                        }
                    }
                    if (b0_ok) {
                        v0 += b0;
                    }
                    if (b1_ok) {
                        v1 += b1;
                    }
                }
                float tmax = fmaxf(v0, v1);
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, off));
                }
                const float p0 = (ok0 && tmax != -INFINITY) ? expf(v0 - tmax) : 0.0f;
                const float p1 = (ok1 && tmax != -INFINITY) ? expf(v1 - tmax) : 0.0f;
                float psum = p0 + p1;
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    psum += __shfl_xor_sync(0xffffffffu, psum, off);
                }
                if (lane == 0) {
                    sm.m_state[qi] = tmax;
                    sm.l_state[qi] = psum;
                }
                wp0[it] = p0 * vs0;
                wp1[it] = p1 * vs1;
            }
        }
        __syncthreads();

#pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int qr = warp + it * INKLING_TC_WARPS;
            if (qr < q_cnt) {
                const __nv_bfloat16 h0 = __float2bfloat16_rn(wp0[it]);
                const __nv_bfloat16 h1 = __float2bfloat16_rn(wp1[it]);
                sm.p_hi[qr][i0] = h0;
                sm.p_hi[qr][i1] = h1;
                sm.p_lo[qr][i0] = __float2bfloat16_rn(wp0[it] - __bfloat162float(h0));
                sm.p_lo[qr][i1] = __float2bfloat16_rn(wp1[it] - __bfloat162float(h1));
            }
        }
        __syncthreads();

        float O[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
        for (int kk = 0; kk < INKLING_TC_KEYS; kk += 16) {
            const int pr = lane & 15;
            const int pc = kk + ((lane >> 4) & 1) * 8;
            uint32_t a_hi[4], a_lo[4];
            const uint32_t addr_hi = (uint32_t)__cvta_generic_to_shared(&sm.p_hi[pr][pc]);
            const uint32_t addr_lo = (uint32_t)__cvta_generic_to_shared(&sm.p_lo[pr][pc]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                         : "r"(addr_hi));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                         : "r"(addr_lo));
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int vr = kk + (lane & 15);
                const int vcv = 2 * warp + nt;
                uint32_t b_v[2];
                const uint32_t addr_v = (uint32_t)__cvta_generic_to_shared(
                    &sm.kv[1][vr][inkling_tc_swz(vr, vcv)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                             : "=r"(b_v[0]), "=r"(b_v[1])
                             : "r"(addr_v));
                inkling_tc_mma_m16n8k16(O[nt], a_hi[0], a_hi[1], a_hi[2], a_hi[3], b_v[0], b_v[1]);
                inkling_tc_mma_m16n8k16(O[nt], a_lo[0], a_lo[1], a_lo[2], a_lo[3], b_v[0], b_v[1]);
            }
        }
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int c0 = warp * 16 + nt * 8 + 2 * t4;
            if (grp < q_cnt) {
                const int qi = q0 + grp;
                const int r = qi / 3;
                const int g = qi - r * 3;
                float* ob = parts + ((uint64_t)r * heads + head0 + g) * head_stride +
                            (uint64_t)macro_idx * 130;
                ob[2 + c0] = O[nt][0];
                ob[2 + c0 + 1] = O[nt][1];
                if (warp == 0 && nt == 0 && t4 == 0) {
                    ob[0] = sm.m_state[qi];
                    ob[1] = sm.l_state[qi];
                }
            }
            if (grp + 8 < q_cnt) {
                const int qi = q0 + grp + 8;
                const int r = qi / 3;
                const int g = qi - r * 3;
                float* ob = parts + ((uint64_t)r * heads + head0 + g) * head_stride +
                            (uint64_t)macro_idx * 130;
                ob[2 + c0] = O[nt][2];
                ob[2 + c0 + 1] = O[nt][3];
                if (warp == 0 && nt == 0 && t4 == 0) {
                    ob[0] = sm.m_state[qi];
                    ob[1] = sm.l_state[qi];
                }
            }
        }
        if (mt + 1 < n_mtiles) {
            __syncthreads();
        }
    }
}

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

// DFlash 2 drafter kernels: Qwen3.8-Flash-Next's at TP=1, and GLM-5.3-Flash's on every rank of
// its tensor-parallel world.
//
// The drafter is a five-layer bf16 transformer: its projections, norms, rotary and SwiGLU are the
// token image's q4e_* kernels, launched on the drafter's own weights. This image holds only what
// DFlash 2 adds on top of them:
//
//   - q4d_capture: one target layer's completed rows, averaged over the hyper-connection
//     streams, written into that layer's slice of the feature rows the drafter reads.
//   - q4d_block_embed: the draft block's input rows, the anchor token's embedding and then the
//     mask token's, from the target's embedding table.
//   - q4d_conv: the grouped causal convolution across one draft block, run on either side of
//     attention and of the MLP with taps projected from the sublayer's input.
//   - q4d_kv_ring: the context keys and values of committed positions, appended to a layer's
//     ring.
//   - q4d_attn: the block's non-causal sliding-window GQA over the ring and its own keys, and
//     q4d_attn_span_chunk / q4d_attn_span_merge, the same over one span of the ring for every row
//     (Inkling's drafter), with a key head's span read once a block for all its queries.
//   - q4d_topk_partial / q4d_topk_final: the target head's top candidates for each draft row, and
//     at TP the merge of the ranks' candidates for their slices of the vocabulary.
//   - q4d_select: the candidate selector's greedy walk through the candidates' transition
//     scores.
//
// Attribution: block-diffusion drafting is the DFlash method of Chen, Liang and Liu ("DFlash:
// Block Diffusion for Flash Speculative Decoding", ICML 2026; github.com/z-lab/dflash, MIT,
// Copyright (c) 2026 Z Lab). The DFlash 2 drafter - the grouped convolutions, the candidate
// selector and the capture of target layers 5/15/25/35/45 as the mean of the four streams - is
// Inco AI's release for Qwen3.8-Flash-Next (huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2,
// Apache-2.0). The arithmetic follows SGLang's DFlash support (github.com/sgl-project/sglang,
// Apache-2.0: python/sglang/srt/models/dflash.py and python/sglang/srt/speculative/dflash_*.py);
// no source is copied. See the root NOTICE file.

#include <cuda_bf16.h>

#include "q4e_common_sm121.cuh"

// The shared xor butterfly, under the name the drafter used.
#define q4d_warp_sum q4e_warp_sum_all
#include <math_constants.h>

// The drafter's head width; q4d_attn holds one query head in a warp, four floats a lane.
#define Q4D_HEAD_DIM 128
// The most warps q4d_attn's cross-warp merge stages.
#define Q4D_ATTN_WARPS 8
// The values one thread holds while a block picks its top candidates.
#define Q4D_TOPK_PER_THREAD 8
// The most candidates a row keeps, and the most selector rank.
#define Q4D_MAX_TOPK 32
#define Q4D_MAX_RANK 1024


// Whether (value, id) ranks above (other_value, other_id): the larger value, and on a tie the
// lower id, so the order is the same however the candidates were split between blocks.
__device__ __forceinline__ bool q4d_ranks_above(float value, int id, float other_value,
                                                int other_id) {
    return value > other_value || (value == other_value && id < other_id);
}

// target layer `slot`'s completed rows, the mean of `streams` streams of `hidden` floats each,
// into features[row * feature_stride + slot * hidden ...]. Grid (hidden / 256, rows).
extern "C" __global__ void q4d_capture(const float* __restrict__ hyper,
                                       float* __restrict__ features, int hidden, int streams,
                                       int feature_stride, int slot) {
    const int row = blockIdx.y;
    const float* source = hyper + (unsigned long long)row * streams * hidden;
    float* into = features + (unsigned long long)row * feature_stride + slot * hidden;
    for (int column = blockIdx.x * blockDim.x + threadIdx.x; column < hidden;
         column += blockDim.x * gridDim.x) {
        float total = 0.0f;
        for (int stream = 0; stream < streams; ++stream) {
            total += source[stream * hidden + column];
        }
        into[column] = total / (float)streams;
    }
}

// Row 0 is the anchor's embedding and every later row the mask token's. Grid (hidden / 256, rows).
extern "C" __global__ void q4d_block_embed(const __nv_bfloat16* __restrict__ table,
                                           float* __restrict__ out, int anchor, int mask,
                                           int hidden) {
    const int row = blockIdx.y;
    const __nv_bfloat16* source =
        table + (unsigned long long)(row == 0 ? anchor : mask) * hidden;
    float* into = out + (unsigned long long)row * hidden;
    for (int column = blockIdx.x * blockDim.x + threadIdx.x; column < hidden;
         column += blockDim.x * gridDim.x) {
        into[column] = __bfloat162float(source[column]);
    }
}

// The grouped causal convolution across a block of `block` rows, on side `side` (0 before the
// sublayer, 1 after it):
//
//   out[n][c] = sum over tap < taps with n % block >= tap of
//               (base[side][tap][c] + coefficients[n][side][tap][c / group_size]) * x[n - tap][c]
//
// coefficients is the sublayer input's projection, [rows][2][taps][hidden / group_size] floats
// `coefficient_stride` apart; base is the checkpoint's [2][taps][hidden] bf16 kernel. out must
// not alias x. Grid (hidden / 256, rows).
extern "C" __global__ void q4d_conv(const float* __restrict__ x,
                                    const float* __restrict__ coefficients,
                                    const __nv_bfloat16* __restrict__ base,
                                    float* __restrict__ out, int hidden, int group_size, int taps,
                                    int side, int coefficient_stride, int block) {
    const int row = blockIdx.y;
    const int within = row % block;
    const int groups = hidden / group_size;
    const float* taps_row = coefficients + (unsigned long long)row * coefficient_stride;
    for (int column = blockIdx.x * blockDim.x + threadIdx.x; column < hidden;
         column += blockDim.x * gridDim.x) {
        const int group = column / group_size;
        float total = 0.0f;
        for (int tap = 0; tap < taps && tap <= within; ++tap) {
            const int kernel = side * taps + tap;
            const float weight = __bfloat162float(base[(unsigned long long)kernel * hidden + column]) +
                                 taps_row[kernel * groups + group];
            total += weight * x[(unsigned long long)(row - tap) * hidden + column];
        }
        out[(unsigned long long)row * hidden + column] = total;
    }
}

// Append `rows` committed positions' keys and values (kv_width floats each) to a layer's ring of
// `ring` positions, position p at slot p % ring. Grid (kv_width / 256, rows).
extern "C" __global__ void q4d_kv_ring(float* __restrict__ key_ring, float* __restrict__ value_ring,
                                       const float* __restrict__ keys,
                                       const float* __restrict__ values, int kv_width,
                                       int position, int ring) {
    const int row = blockIdx.y;
    const unsigned long long slot = (unsigned long long)((position + row) % ring);
    for (int column = blockIdx.x * blockDim.x + threadIdx.x; column < kv_width;
         column += blockDim.x * gridDim.x) {
        key_ring[slot * kv_width + column] = keys[(unsigned long long)row * kv_width + column];
        value_ring[slot * kv_width + column] = values[(unsigned long long)row * kv_width + column];
    }
}

// The walks' arithmetic, pinned: the contraction the walk compiled to before it was shared (a
// key's score fma(q.w, k.w, fma(q.z, k.z, fma(q.x, k.x, q.y * k.y))), the running sum
// fma(correction, sum, weight), a dim fma(weight, v, acc * correction)), written out so no kernel
// that inlines the walk can contract the mul-adds the other way and move its bits.
__device__ __forceinline__ float q4d_attn_dot(const float4 query, const float4 k) {
    return __fmaf_rn(query.w, k.w,
                     __fmaf_rn(query.z, k.z, __fmaf_rn(query.x, k.x, __fmul_rn(query.y, k.y))));
}

// The keys a walk loads and scores before it folds them in, in order: their loads, dots and warp
// sums are independent, so they overlap, while the fold, the walk's only chain from key to key,
// stays one key at a time.
#define Q4D_ATTN_AHEAD 8

// q4d_attn_walk's batches. kStep: the ring slot of a batch's key j is its first key's plus
// j * stride, wrapped once, which holds while (Q4D_ATTN_AHEAD - 1) * stride < ring; otherwise
// every key takes its slot modulo the ring. Either way a key reads the same slot.
template <bool kStep>
__device__ __forceinline__ void q4d_attn_walk_batches(
    const float4 query, const float* __restrict__ key_ring, const float* __restrict__ value_ring,
    const float* __restrict__ block_keys, const float* __restrict__ block_values, int kv_width,
    int offset, int lowest, int ring, int context, int total, int first, int stride, float scale,
    float& running_max, float& running_sum, float (&acc)[4]) {
    for (int base_key = first; base_key < total; base_key += stride * Q4D_ATTN_AHEAD) {
        const int base_slot = base_key < context ? (lowest + base_key) % ring : 0;
        float4 k[Q4D_ATTN_AHEAD];
        float4 v[Q4D_ATTN_AHEAD];
#pragma unroll
        for (int j = 0; j < Q4D_ATTN_AHEAD; ++j) {
            // A key past the end loads the batch's first key, which is in range, and is not folded.
            const int key = base_key + j * stride < total ? base_key + j * stride : base_key;
            const bool in_ring = key < context;
            int slot;
            if (kStep) {
                slot = base_slot + (key - base_key);
                slot = slot >= ring ? slot - ring : slot;
            } else {
                slot = in_ring ? (lowest + key) % ring : 0;
            }
            const unsigned long long at =
                (unsigned long long)(in_ring ? slot : key - context) * kv_width + offset;
            k[j] = *reinterpret_cast<const float4*>((in_ring ? key_ring : block_keys) + at);
            v[j] = *reinterpret_cast<const float4*>((in_ring ? value_ring : block_values) + at);
        }
        float score[Q4D_ATTN_AHEAD];
#pragma unroll
        for (int j = 0; j < Q4D_ATTN_AHEAD; ++j) {
            score[j] = q4d_attn_dot(query, k[j]);
        }
        // q4d_warp_sum, the batch's keys side by side.
#pragma unroll
        for (int width = 16; width > 0; width >>= 1) {
#pragma unroll
            for (int j = 0; j < Q4D_ATTN_AHEAD; ++j) {
                score[j] += __shfl_xor_sync(0xFFFFFFFFu, score[j], width);
            }
        }
#pragma unroll
        for (int j = 0; j < Q4D_ATTN_AHEAD; ++j) {
            if (base_key + j * stride < total) {
                const float scaled = __fmul_rn(score[j], scale);
                const float next_max = fmaxf(running_max, scaled);
                const float correction = __expf(running_max - next_max);
                const float weight = __expf(scaled - next_max);
                running_sum = __fmaf_rn(correction, running_sum, weight);
                acc[0] = __fmaf_rn(weight, v[j].x, __fmul_rn(acc[0], correction));
                acc[1] = __fmaf_rn(weight, v[j].y, __fmul_rn(acc[1], correction));
                acc[2] = __fmaf_rn(weight, v[j].z, __fmul_rn(acc[2], correction));
                acc[3] = __fmaf_rn(weight, v[j].w, __fmul_rn(acc[3], correction));
                running_max = next_max;
            }
        }
    }
}

// One query's online softmax over keys first, first + stride, ... of `total`: key k below
// `context` is the ring's committed position lowest + k, the rest the block's own rows. A lane holds
// four of the head's dims, at `offset` floats into a key or value row of kv_width. A walk that sees
// no key leaves max -inf, sum 0 and acc 0.
__device__ __forceinline__ void q4d_attn_walk(const float4 query,
                                              const float* __restrict__ key_ring,
                                              const float* __restrict__ value_ring,
                                              const float* __restrict__ block_keys,
                                              const float* __restrict__ block_values,
                                              int kv_width, int offset, int lowest, int ring,
                                              int context, int total, int first, int stride,
                                              float scale, float& running_max, float& running_sum,
                                              float (&acc)[4]) {
    if ((Q4D_ATTN_AHEAD - 1) * stride < ring) {
        q4d_attn_walk_batches<true>(query, key_ring, value_ring, block_keys, block_values,
                                    kv_width, offset, lowest, ring, context, total, first, stride,
                                    scale, running_max, running_sum, acc);
    } else {
        q4d_attn_walk_batches<false>(query, key_ring, value_ring, block_keys, block_values,
                                     kv_width, offset, lowest, ring, context, total, first, stride,
                                     scale, running_max, running_sum, acc);
    }
}

// One query's `parts` walks merged, in order, into its output: walk p's max and sum at
// maxes[p * scalar_stride] and sums[p * scalar_stride], its acc at accs + p * acc_stride (this
// lane's four dims at lane * 4); `into` is the lane's four output dims.
__device__ __forceinline__ void q4d_attn_merge(const float* maxes, const float* sums,
                                               int scalar_stride, const float* accs,
                                               int acc_stride, int parts, int lane, float* into) {
    float merged_max = -CUDART_INF_F;
    for (int other = 0; other < parts; ++other) {
        merged_max = fmaxf(merged_max, maxes[other * scalar_stride]);
    }
    float merged_sum = 0.0f;
    float merged[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int other = 0; other < parts; ++other) {
        // A walk that saw no key holds -inf, and its share is exactly zero.
        const float share = __expf(maxes[other * scalar_stride] - merged_max);
        merged_sum = __fmaf_rn(sums[other * scalar_stride], share, merged_sum);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            merged[i] = __fmaf_rn(accs[other * acc_stride + lane * 4 + i], share, merged[i]);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        into[i] = __fdiv_rn(merged[i], merged_sum);
    }
}

// One query row of the draft block's attention: row `row` attends, without a causal mask, to the
// committed positions [lowest, base) (read from the ring) and to all `block_rows` of the block's
// own keys. GQA: query head h reads key head h / (heads / kv_heads). blockDim a multiple of 32, at
// most Q4D_ATTN_WARPS warps, each walking every warps-th key with an online softmax, merged at the
// end.
__device__ __forceinline__ void q4d_attn_row(const float* __restrict__ queries,
                                             const float* __restrict__ key_ring,
                                             const float* __restrict__ value_ring,
                                             const float* __restrict__ block_keys,
                                             const float* __restrict__ block_values,
                                             float* __restrict__ out, int heads, int kv_heads,
                                             int head_dim, int base, int lowest, int ring,
                                             int block_rows, float scale, int head, int row) {
    __shared__ float s_max[Q4D_ATTN_WARPS];
    __shared__ float s_sum[Q4D_ATTN_WARPS];
    __shared__ float s_acc[Q4D_ATTN_WARPS][Q4D_HEAD_DIM];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    if (head_dim != Q4D_HEAD_DIM || warps > Q4D_ATTN_WARPS || heads % kv_heads != 0) {
        return;
    }
    const int key_head = head / (heads / kv_heads);
    const int q_width = heads * head_dim;
    const int kv_width = kv_heads * head_dim;
    const float4 query = *reinterpret_cast<const float4*>(
        queries + (unsigned long long)row * q_width + head * head_dim + lane * 4);
    const int context = max(0, base - lowest);
    const int total = context + block_rows;

    float running_max = -CUDART_INF_F;
    float running_sum = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    q4d_attn_walk(query, key_ring, value_ring, block_keys, block_values, kv_width,
                  key_head * head_dim + lane * 4, lowest, ring, context, total, warp, warps, scale,
                  running_max, running_sum, acc);
    if (lane == 0) {
        s_max[warp] = running_max;
        s_sum[warp] = running_sum;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        s_acc[warp][lane * 4 + i] = acc[i];
    }
    __syncthreads();
    if (warp != 0) {
        return;
    }
    q4d_attn_merge(s_max, s_sum, 1, &s_acc[0][0], Q4D_HEAD_DIM, warps, lane,
                   out + (unsigned long long)row * q_width + head * head_dim + lane * 4);
}

// The draft block's attention: query row r sits at position base + r and attends to every
// committed position p with base + r - p <= window_left and to all `block_rows` of the block's own
// keys. Grid (heads, block_rows); see q4d_attn_row.
extern "C" __global__ void q4d_attn(const float* __restrict__ queries,
                                    const float* __restrict__ key_ring,
                                    const float* __restrict__ value_ring,
                                    const float* __restrict__ block_keys,
                                    const float* __restrict__ block_values,
                                    float* __restrict__ out, int heads, int kv_heads, int head_dim,
                                    int base, int window_left, int ring, int block_rows,
                                    float scale) {
    const int row = blockIdx.y;
    q4d_attn_row(queries, key_ring, value_ring, block_keys, block_values, out, heads, kv_heads,
                 head_dim, base, max(0, base + row - window_left), ring, block_rows, scale,
                 blockIdx.x, row);
}

// The floats one chunk of q4d_attn_span_chunk leaves: its acc (Q4D_HEAD_DIM), max and sum, padded
// to a float4.
#define Q4D_ATTN_PART (Q4D_HEAD_DIM + 4)
// A q4d_attn_span_chunk warp's queries and the keys it scores at a time: one (query, key) pair a
// lane.
#define Q4D_SPAN_QUERIES 8
#define Q4D_SPAN_KEYS 4

// The draft block's attention over one span of committed positions for every row (Inkling's
// drafter, which anchors its windows at the block, not at each row): each row attends to
// [lowest, base) and to all `block_rows` of the block's own keys, in two stages.
//
// Stage one: block (key_head, chunk) walks the chunk-th of gridDim.y contiguous runs of those
// keys, read once a block for every query of the key head's group. Warp w holds queries
// w * Q4D_SPAN_QUERIES .. + Q4D_SPAN_QUERIES - 1, numbered row-major over (row, head of the group);
// blockDim covers the group's queries. A batch's 32 (query, key) dot products are summed by a
// transpose reduction (31 shuffles; lane L ends with the pair L = (L / Q4D_SPAN_KEYS,
// L % Q4D_SPAN_KEYS)); each query's softmax steps once over the batch's keys, and each weight is
// taken once by its pair's lane and broadcast. At Inkling's TP4 shape (48 queries of a key head)
// this walk measured 1.06 ms over a 93,159-key span against 2.70 ms for a warp a query reading
// the span once per four queries, and 20.5 against 28.5 us over a local layer's 1,032 keys. The
// chunk's acc, max and sum go to part[((row * heads + head) * gridDim.y + chunk) * Q4D_ATTN_PART].
// Grid (kv_heads, chunks).
extern "C" __global__ void q4d_attn_span_chunk(const float* __restrict__ queries,
                                               const float* __restrict__ key_ring,
                                               const float* __restrict__ value_ring,
                                               const float* __restrict__ block_keys,
                                               const float* __restrict__ block_values,
                                               float* __restrict__ part, int heads, int kv_heads,
                                               int head_dim, int base, int lowest, int ring,
                                               int block_rows, float scale) {
    static_assert(Q4D_SPAN_QUERIES * Q4D_SPAN_KEYS == 32, "one (query, key) pair a lane");
    if (head_dim != Q4D_HEAD_DIM || heads % kv_heads != 0) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int group = heads / kv_heads;
    const int queries_total = group * block_rows;
    const int key_head = blockIdx.x;
    const int chunk = blockIdx.y;
    const int chunks = gridDim.y;
    const int context = max(0, base - lowest);
    const int total = context + block_rows;
    const int per = (total + chunks - 1) / chunks;
    const int first = min(total, chunk * per);
    const int last = min(total, first + per);
    const int kv_width = kv_heads * head_dim;
    const int offset = key_head * head_dim + lane * 4;
    const int own_query = lane / Q4D_SPAN_KEYS;
    const int own_key = lane % Q4D_SPAN_KEYS;
    float4 query[Q4D_SPAN_QUERIES];
    float running_max[Q4D_SPAN_QUERIES];
    float running_sum[Q4D_SPAN_QUERIES];
    float acc[Q4D_SPAN_QUERIES][4];
#pragma unroll
    for (int i = 0; i < Q4D_SPAN_QUERIES; ++i) {
        // A warp's queries past the group's repeat its last one, and are not written.
        const int index = min(warp * Q4D_SPAN_QUERIES + i, queries_total - 1);
        const int row = index / group;
        const int head = key_head * group + index % group;
        query[i] = *reinterpret_cast<const float4*>(
            queries + (unsigned long long)row * heads * head_dim + head * head_dim + lane * 4);
        running_max[i] = -CUDART_INF_F;
        running_sum[i] = 0.0f;
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    }
    for (int base_key = first; base_key < last; base_key += Q4D_SPAN_KEYS) {
        float4 k[Q4D_SPAN_KEYS];
        float4 v[Q4D_SPAN_KEYS];
#pragma unroll
        for (int j = 0; j < Q4D_SPAN_KEYS; ++j) {
            // A key past the run's end loads the batch's first key, which is in range, and gets
            // no weight.
            const int key = base_key + j < last ? base_key + j : base_key;
            const bool in_ring = key < context;
            const int slot = in_ring ? (lowest + key) % ring : key - context;
            const unsigned long long at = (unsigned long long)slot * kv_width + offset;
            k[j] = *reinterpret_cast<const float4*>((in_ring ? key_ring : block_keys) + at);
            v[j] = *reinterpret_cast<const float4*>((in_ring ? value_ring : block_values) + at);
        }
        float dot[32];
#pragma unroll
        for (int i = 0; i < Q4D_SPAN_QUERIES; ++i) {
#pragma unroll
            for (int j = 0; j < Q4D_SPAN_KEYS; ++j) {
                dot[i * Q4D_SPAN_KEYS + j] = q4d_attn_dot(query[i], k[j]);
            }
        }
        // Each round halves the sums a lane holds: it keeps the half its lane bit names and adds
        // its partner's share of it.
#pragma unroll
        for (int width = 16; width >= 1; width >>= 1) {
            const bool upper = (lane & width) != 0;
#pragma unroll
            for (int t = 0; t < width; ++t) {
                const float send = upper ? dot[t] : dot[t + width];
                const float keep = upper ? dot[t + width] : dot[t];
                dot[t] = __fadd_rn(keep, __shfl_xor_sync(0xFFFFFFFFu, send, width));
            }
        }
        const float scaled =
            base_key + own_key < last ? __fmul_rn(dot[0], scale) : -CUDART_INF_F;
        float batch_max = scaled;
#pragma unroll
        for (int width = Q4D_SPAN_KEYS / 2; width >= 1; width >>= 1) {
            batch_max = fmaxf(batch_max, __shfl_xor_sync(0xFFFFFFFFu, batch_max, width));
        }
        float own_max = 0.0f;
#pragma unroll
        for (int i = 0; i < Q4D_SPAN_QUERIES; ++i) {
            const float next_max = fmaxf(
                running_max[i], __shfl_sync(0xFFFFFFFFu, batch_max, i * Q4D_SPAN_KEYS));
            const float correction = __expf(running_max[i] - next_max);
            running_sum[i] = __fmul_rn(running_sum[i], correction);
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                acc[i][d] = __fmul_rn(acc[i][d], correction);
            }
            running_max[i] = next_max;
            own_max = own_query == i ? next_max : own_max;
        }
        const float own_weight = __expf(scaled - own_max);
#pragma unroll
        for (int i = 0; i < Q4D_SPAN_QUERIES; ++i) {
#pragma unroll
            for (int j = 0; j < Q4D_SPAN_KEYS; ++j) {
                const float weight =
                    __shfl_sync(0xFFFFFFFFu, own_weight, i * Q4D_SPAN_KEYS + j);
                running_sum[i] = __fadd_rn(running_sum[i], weight);
                acc[i][0] = __fmaf_rn(weight, v[j].x, acc[i][0]);
                acc[i][1] = __fmaf_rn(weight, v[j].y, acc[i][1]);
                acc[i][2] = __fmaf_rn(weight, v[j].z, acc[i][2]);
                acc[i][3] = __fmaf_rn(weight, v[j].w, acc[i][3]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < Q4D_SPAN_QUERIES; ++i) {
        const int index = warp * Q4D_SPAN_QUERIES + i;
        if (index >= queries_total) {
            break;
        }
        const int row = index / group;
        const int head = key_head * group + index % group;
        float* at =
            part + ((unsigned long long)(row * heads + head) * chunks + chunk) * Q4D_ATTN_PART;
        *reinterpret_cast<float4*>(at + lane * 4) =
            make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        if (lane == 0) {
            at[Q4D_HEAD_DIM] = running_max[i];
            at[Q4D_HEAD_DIM + 1] = running_sum[i];
        }
    }
}

// Stage two: query (row, head)'s `chunks` parts merged, in chunk order, into its output. Grid
// (heads, block_rows), one warp.
extern "C" __global__ void q4d_attn_span_merge(const float* __restrict__ part,
                                               float* __restrict__ out, int heads, int head_dim,
                                               int chunks) {
    const int lane = threadIdx.x & 31;
    if (head_dim != Q4D_HEAD_DIM || threadIdx.x >= 32) {
        return;
    }
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const float* at = part + (unsigned long long)(row * heads + head) * chunks * Q4D_ATTN_PART;
    q4d_attn_merge(at + Q4D_HEAD_DIM, at + Q4D_HEAD_DIM + 1, Q4D_ATTN_PART, at, Q4D_ATTN_PART,
                   chunks, lane,
                   out + (unsigned long long)row * heads * head_dim + head * head_dim + lane * 4);
}

// The block's top k of `count` (value, id) pairs, best first, into out_values / out_ids. The pairs
// come in groups of `group`, `stride` apart: pair i is at (i / group) * stride + i % group. ids is
// null when a pair's id is first_id plus its index. Every thread holds up to Q4D_TOPK_PER_THREAD
// pairs; each round the block agrees on the best one left and its holder retires it.
__device__ void q4d_block_topk(const float* __restrict__ values, const int* __restrict__ ids,
                               int count, int group, unsigned long long stride, int first_id, int k,
                               float* __restrict__ out_values, int* __restrict__ out_ids) {
    __shared__ float s_value[32];
    __shared__ int s_id[32];
    float held_value[Q4D_TOPK_PER_THREAD];
    int held_id[Q4D_TOPK_PER_THREAD];
#pragma unroll
    for (int j = 0; j < Q4D_TOPK_PER_THREAD; ++j) {
        const int index = threadIdx.x + j * blockDim.x;
        if (index < count) {
            const unsigned long long at = (index / group) * stride + index % group;
            const float value = values[at];
            // A NaN never wins a round: it compares as neither larger nor equal.
            held_value[j] = isnan(value) ? -CUDART_INF_F : value;
            held_id[j] = ids != nullptr ? ids[at] : first_id + index;
        } else {
            held_value[j] = -CUDART_INF_F;
            held_id[j] = 0x7FFFFFFF;
        }
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = (blockDim.x + 31) >> 5;
    for (int round = 0; round < k; ++round) {
        float best_value = -CUDART_INF_F;
        int best_id = 0x7FFFFFFF;
#pragma unroll
        for (int j = 0; j < Q4D_TOPK_PER_THREAD; ++j) {
            if (q4d_ranks_above(held_value[j], held_id[j], best_value, best_id)) {
                best_value = held_value[j];
                best_id = held_id[j];
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_value = __shfl_xor_sync(0xFFFFFFFFu, best_value, offset);
            const int other_id = __shfl_xor_sync(0xFFFFFFFFu, best_id, offset);
            if (q4d_ranks_above(other_value, other_id, best_value, best_id)) {
                best_value = other_value;
                best_id = other_id;
            }
        }
        if (lane == 0) {
            s_value[warp] = best_value;
            s_id[warp] = best_id;
        }
        __syncthreads();
        best_value = s_value[0];
        best_id = s_id[0];
        for (int other = 1; other < warps; ++other) {
            if (q4d_ranks_above(s_value[other], s_id[other], best_value, best_id)) {
                best_value = s_value[other];
                best_id = s_id[other];
            }
        }
        // Every thread has read this round's winners before the next round writes them.
        __syncthreads();
        if (threadIdx.x == 0) {
            out_values[round] = best_value;
            out_ids[round] = best_id;
        }
#pragma unroll
        for (int j = 0; j < Q4D_TOPK_PER_THREAD; ++j) {
            if (held_id[j] == best_id && held_value[j] == best_value) {
                held_value[j] = -CUDART_INF_F;
                held_id[j] = 0x7FFFFFFF;
            }
        }
    }
}

// Stage one of a row's top k over `vocab` logits, whose ids start at `first_id` (a rank's slice
// of the vocabulary): block x takes `chunk` consecutive ids (at most blockDim.x *
// Q4D_TOPK_PER_THREAD) and leaves its k best at part[row][x]. Grid (chunks, rows).
extern "C" __global__ void q4d_topk_partial(const float* __restrict__ logits,
                                            float* __restrict__ part_values,
                                            int* __restrict__ part_ids, int vocab, int chunk,
                                            int k, int first_id) {
    const int row = blockIdx.y;
    const int first = blockIdx.x * chunk;
    const int count = min(chunk, vocab - first);
    const unsigned long long part = (unsigned long long)row * gridDim.x + blockIdx.x;
    q4d_block_topk(logits + (unsigned long long)row * vocab + first, nullptr, count, count, 0,
                   first_id + first, k, part_values + part * k, part_ids + part * k);
}

// Stage two: a row's k best of `parts` parts of k candidates, row r's part p at
// r * row_stride + p * part_stride: the chunks' parts ([rows][parts][k]: parts * k and k), or the
// ranks' candidates for their slices gathered [ranks][rows][k] (k and rows * k), whose merge is
// the top k of the whole vocabulary because (value, id) orders every pair. Grid (rows).
extern "C" __global__ void q4d_topk_final(const float* __restrict__ part_values,
                                          const int* __restrict__ part_ids,
                                          float* __restrict__ values, int* __restrict__ ids,
                                          int parts, int k, unsigned long long row_stride,
                                          unsigned long long part_stride) {
    const int row = blockIdx.x;
    const unsigned long long first = (unsigned long long)row * row_stride;
    q4d_block_topk(part_values + first, part_ids + first, parts * k, k, part_stride, 0, k,
                   values + (unsigned long long)row * k, ids + (unsigned long long)row * k);
}

// The selector's greedy walk over `edges` draft rows of k candidates each. The score of
// candidate c at row e, coming from token `previous`, is
//
//   unary[e][c] + sum over r of predecessor[previous][r] * projected[e][r] * successor[cand[e][c]][r]
//
// where `previous` is the anchor at row 0 and row e - 1's chosen token after it. Each row takes
// its best-scoring candidate (the first on a tie) and drafts[e] receives its token, and
// confidences[e] the chosen candidate's share of the softmax over the row's scores: what the
// verify depth weighs the draft by. One block.
extern "C" __global__ void q4d_select(const int* __restrict__ candidates,
                                      const float* __restrict__ unary,
                                      const float* __restrict__ projected,
                                      const __nv_bfloat16* __restrict__ predecessor,
                                      const __nv_bfloat16* __restrict__ successor,
                                      int* __restrict__ drafts,
                                      float* __restrict__ confidences, int anchor, int edges,
                                      int k, int rank) {
    __shared__ float s_weighted[Q4D_MAX_RANK];
    __shared__ float s_score[Q4D_MAX_TOPK];
    __shared__ int s_previous;
    if (k > Q4D_MAX_TOPK || rank > Q4D_MAX_RANK) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    if (threadIdx.x == 0) {
        s_previous = anchor;
    }
    __syncthreads();
    for (int edge = 0; edge < edges; ++edge) {
        const __nv_bfloat16* from = predecessor + (unsigned long long)s_previous * rank;
        for (int r = threadIdx.x; r < rank; r += blockDim.x) {
            s_weighted[r] = __bfloat162float(from[r]) * projected[edge * rank + r];
        }
        __syncthreads();
        for (int c = warp; c < k; c += warps) {
            const __nv_bfloat16* to =
                successor + (unsigned long long)candidates[edge * k + c] * rank;
            float total = 0.0f;
            for (int r = lane; r < rank; r += 32) {
                total += s_weighted[r] * __bfloat162float(to[r]);
            }
            total = q4d_warp_sum(total);
            if (lane == 0) {
                s_score[c] = unary[edge * k + c] + total;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int best = 0;
            for (int c = 1; c < k; ++c) {
                if (s_score[c] > s_score[best]) {
                    best = c;
                }
            }
            const int token = candidates[edge * k + best];
            drafts[edge] = token;
            float total = 0.0f;
            for (int c = 0; c < k; ++c) {
                total += __expf(s_score[c] - s_score[best]);
            }
            confidences[edge] = 1.0f / total;
            s_previous = token;
        }
        __syncthreads();
    }
}

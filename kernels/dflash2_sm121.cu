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
//   - q4d_attn: the block's non-causal sliding-window GQA over the ring and its own keys.
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

// The draft block's attention: query row r sits at position base + r and attends, without a
// causal mask, to every committed position p with base + r - p <= window_left (read from the
// ring) and to all `block_rows` of the block's own keys. GQA: query head h reads key head
// h / (heads / kv_heads). Grid (heads, block_rows); blockDim a multiple of 32, at most
// Q4D_ATTN_WARPS warps, each walking every warps-th key with an online softmax, merged at the end.
extern "C" __global__ void q4d_attn(const float* __restrict__ queries,
                                    const float* __restrict__ key_ring,
                                    const float* __restrict__ value_ring,
                                    const float* __restrict__ block_keys,
                                    const float* __restrict__ block_values,
                                    float* __restrict__ out, int heads, int kv_heads, int head_dim,
                                    int base, int window_left, int ring, int block_rows,
                                    float scale) {
    __shared__ float s_max[Q4D_ATTN_WARPS];
    __shared__ float s_sum[Q4D_ATTN_WARPS];
    __shared__ float s_acc[Q4D_ATTN_WARPS][Q4D_HEAD_DIM];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    if (head_dim != Q4D_HEAD_DIM || warps > Q4D_ATTN_WARPS || heads % kv_heads != 0) {
        return;
    }
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int key_head = head / (heads / kv_heads);
    const int q_width = heads * head_dim;
    const int kv_width = kv_heads * head_dim;
    const float4 query = *reinterpret_cast<const float4*>(
        queries + (unsigned long long)row * q_width + head * head_dim + lane * 4);
    const int lowest = max(0, base + row - window_left);
    const int context = max(0, base - lowest);
    const int total = context + block_rows;

    float running_max = -CUDART_INF_F;
    float running_sum = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int key = warp; key < total; key += warps) {
        const float* key_row;
        const float* value_row;
        if (key < context) {
            const unsigned long long slot = (unsigned long long)((lowest + key) % ring);
            key_row = key_ring + slot * kv_width;
            value_row = value_ring + slot * kv_width;
        } else {
            const unsigned long long own = (unsigned long long)(key - context);
            key_row = block_keys + own * kv_width;
            value_row = block_values + own * kv_width;
        }
        const float4 k = *reinterpret_cast<const float4*>(key_row + key_head * head_dim + lane * 4);
        float score = query.x * k.x + query.y * k.y + query.z * k.z + query.w * k.w;
        score = q4d_warp_sum(score) * scale;
        const float next_max = fmaxf(running_max, score);
        const float correction = __expf(running_max - next_max);
        const float weight = __expf(score - next_max);
        const float4 v =
            *reinterpret_cast<const float4*>(value_row + key_head * head_dim + lane * 4);
        running_sum = running_sum * correction + weight;
        acc[0] = acc[0] * correction + weight * v.x;
        acc[1] = acc[1] * correction + weight * v.y;
        acc[2] = acc[2] * correction + weight * v.z;
        acc[3] = acc[3] * correction + weight * v.w;
        running_max = next_max;
    }
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
    float merged_max = -CUDART_INF_F;
    for (int other = 0; other < warps; ++other) {
        merged_max = fmaxf(merged_max, s_max[other]);
    }
    float merged_sum = 0.0f;
    float merged[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int other = 0; other < warps; ++other) {
        // A warp that saw no key holds -inf, and its share is exactly zero.
        const float share = __expf(s_max[other] - merged_max);
        merged_sum += s_sum[other] * share;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            merged[i] += s_acc[other][lane * 4 + i] * share;
        }
    }
    float* into = out + (unsigned long long)row * q_width + head * head_dim + lane * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        into[i] = merged[i] / merged_sum;
    }
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

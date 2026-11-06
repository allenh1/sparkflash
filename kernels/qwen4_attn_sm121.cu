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

// Qwen3.8-Flash-Next (Qwen4-Exp) attention, sparse-indexer and KV-cache token kernels, TP=1.
//
// This is the Qwen attention family that used to sit in qwen4_exp_sm121.cu. It is its own
// translation unit and cubin (Image::Qwen4Attn) so an arm that only needs attention does not
// carry the token, norm, dense and hyper kernels:
//
//   - q4e_cache_bf16 / q4e_cache_rows_bf16: the KV cache's BF16 narrowing append.
//   - q4e_qsa_pool / q4e_qsa_scores / q4e_qsa_pick: the sparse indexer's pooled keys, block
//     scores and budget selection.
//   - q4e_attn / q4e_attn_group / q4e_attn_group6: grouped-query attention over the selected
//     keys with the sigmoid output gate.
//
// The reductions and the indexer's block selection are the shared headers' (q4e_common and
// q4e_gemv); none of them is an exported symbol, so the cubins do not disagree over a global.
//
// Attribution (as in qwen4_exp_sm121.cu):
//   - Model architecture (Qwen4ExpForConditionalGeneration: Qwen Sparse Attention with a
//     4-token block-compressed indexer) originates from Qwen/Qwen3.8-Flash-Next by the Qwen team
//     at Alibaba Cloud and the vLLM reference implementation
//     (https://github.com/vllm-project/vllm, Apache-2.0).
//   - SM120/SM121 (DGX Spark GB10) kernel design and local-inference execution patterns draw on
//     prior art from b12x (https://github.com/local-inference-lab/b12x,
//     flashinfer/experimental/b12x, Apache-2.0; Luke Alonso @lukealonso, @original-el8, and
//     contributors) and NVIDIA Flare (https://github.com/NVIDIA/flare-local-inference, Apache-2.0).
//     See the root NOTICE file for full provenance details.
//
// The arithmetic mirrors src/forward.rs, which is the host reference oracle. Each kernel is named
// q4e_* and is loaded by name from the cubin, so a missing symbol is a load error rather than a
// silent wrong launch.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Every q4e kernel opens with OPT_PDL_ENTRY: the host launches them with programmatic stream
// serialization, so the next kernel's launch overlaps this one's run and stream order holds.
#include "opt_pdl_sm121.cuh"
#include "q4e_common_sm121.cuh"
#include "q4e_gemv_sm121.cuh"

#define Q4E_BLOCK 256

// The selected tokens q4e_attn stages in shared memory.
#define Q4E_MAX_SELECTED 2304


// Widen one token's BF16 embedding row and replicate it across width = hc * hidden. With a token
// array, grid row y widens tokens[y] into out + y * width instead, so a token the device picked
// is embedded without the host reading it.

// dst[i] = BF16(src[i]), the KV cache's narrowing append: the projections and their rotary
// stay f32, and the cache keeps what the reference keeps, the model dtype.
extern "C" __global__ void q4e_cache_bf16(__nv_bfloat16* __restrict__ dst,
                                          const float* __restrict__ src, int count) {
    OPT_PDL_ENTRY();
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += blockDim.x * gridDim.x) {
        dst[index] = __float2bfloat16_rn(src[index]);
    }
}

// dst[row][0..width] = BF16(src[row][0..width]) for rows rows, with independent strides: the
// indexer's per-token row into its BF16 key cache.
extern "C" __global__ void q4e_cache_rows_bf16(__nv_bfloat16* __restrict__ dst,
                                               const float* __restrict__ src, int width,
                                               int dst_stride, int src_stride) {
    OPT_PDL_ENTRY();
    const int row = blockIdx.y;
    __nv_bfloat16* d = dst + (unsigned long long)row * dst_stride;
    const float* s = src + (unsigned long long)row * src_stride;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < width;
         index += blockDim.x * gridDim.x) {
        d[index] = __float2bfloat16_rn(s[index]);
    }
}

// One indexer block's pooled key: the mean of its ratio keys, RMS-normed with k_weight and
// rotated at the block's first position. It depends on the block's keys alone, so a forward pools
// each block once for every row that scores it, where the score kernel used to pool a block again
// for each row. blockIdx.x is the block; launch 128 threads.
extern "C" __global__ void q4e_qsa_pool(const __nv_bfloat16* index_keys,
                                        const __nv_bfloat16* k_weight, float* pooled_keys,
                                        int index_dim, int ratio, int rotary, float theta,
                                        float eps) {
    OPT_PDL_ENTRY();
    const int block = blockIdx.x;
    if (index_dim > 128 || blockDim.x < 128) {
        return;
    }
    __shared__ float pooled[128];
    __shared__ float scratch[128];
    const int dim = threadIdx.x;
    const bool live = dim < index_dim;
    float value = 0.0f;
    if (live) {
        float sum = 0.0f;
        for (int token = 0; token < ratio; ++token) {
            sum += __bfloat162float(
                index_keys[(unsigned long long)(block * ratio + token) * index_dim + dim]);
        }
        value = sum / (float)ratio;
    }
    const float scale = rsqrtf(q4e_block_sum(value * value, scratch) / (float)index_dim + eps);
    if (live) {
        pooled[dim] = value * scale * (1.0f + __bfloat162float(k_weight[dim]));
    }
    __syncthreads();
    const int half = rotary < index_dim ? rotary / 2 : index_dim / 2;
    if (dim < half) {
        const float frequency = powf(theta, -2.0f * (float)dim / (float)rotary);
        const float angle = (float)(block * ratio) * frequency;
        const float cosine = __cosf(angle);
        const float sine = __sinf(angle);
        const float a = pooled[dim];
        const float b = pooled[dim + half];
        pooled[dim] = a * cosine - b * sine;
        pooled[dim + half] = a * sine + b * cosine;
    }
    __syncthreads();
    if (live) {
        pooled_keys[(unsigned long long)block * index_dim + dim] = pooled[dim];
    }
}

#define Q4E_QSA_TILE 64
#define Q4E_QSA_MAX_HEADS 8

// The indexer's per-block score over the pooled keys: gridDim.x covers the compressed blocks of
// the chunk's last token Q4E_QSA_TILE at a time, gridDim.y the tokens of the chunk. A token's
// score row is score_stride floats into `scores`, so a chunk keeps one row per token.
//
// A warp scores a block, and its sum takes the order the 128-thread block tree took before it
// (channel t + 64 into t, t + 32 into t, then halving), so the scores are bit-identical. The
// products are rounded on their own (__fmul_rn), as they were when the tree read them from
// shared memory, rather than fused into the first adds.
extern "C" __global__ void q4e_qsa_scores(const float* __restrict__ index_q,
                                          const float* __restrict__ pooled_keys,
                                          float* __restrict__ scores, int position,
                                          int index_heads, int index_dim, int ratio,
                                          int score_stride) {
    OPT_PDL_ENTRY();
    const int token_index = blockIdx.y;
    const int blocks = (position + token_index + 1) / ratio;
    const int first = blockIdx.x * Q4E_QSA_TILE;
    if (first >= blocks || index_dim > 128 || index_heads > Q4E_QSA_MAX_HEADS) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const float* index_q_t = index_q + (unsigned long long)token_index * index_heads * index_dim;
    float* scores_t = scores + (unsigned long long)token_index * score_stride;
    float query[Q4E_QSA_MAX_HEADS][4];
#pragma unroll
    for (int head = 0; head < Q4E_QSA_MAX_HEADS; ++head) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            const int dim = lane + 32 * step;
            query[head][step] =
                head < index_heads && dim < index_dim ? index_q_t[head * index_dim + dim] : 0.0f;
        }
    }
    const int last = first + Q4E_QSA_TILE < blocks ? first + Q4E_QSA_TILE : blocks;
    for (int block = first + warp; block < last; block += warps) {
        const float* key_row = pooled_keys + (unsigned long long)block * index_dim;
        float key[4];
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            const int dim = lane + 32 * step;
            key[step] = dim < index_dim ? key_row[dim] : 0.0f;
        }
        float score = 0.0f;
#pragma unroll
        for (int head = 0; head < Q4E_QSA_MAX_HEADS; ++head) {
            if (head < index_heads) {
                const float low = __fadd_rn(__fmul_rn(query[head][0], key[0]),
                                            __fmul_rn(query[head][2], key[2]));
                const float high = __fadd_rn(__fmul_rn(query[head][1], key[1]),
                                             __fmul_rn(query[head][3], key[3]));
                score += fmaxf(q4e_warp_sum(__fadd_rn(low, high)), 0.0f);
            }
        }
        if (lane == 0) {
            scores_t[block] = score / sqrtf((float)index_dim);
        }
    }
}

extern "C" __global__ void q4e_qsa_pick(const float* scores, int* selected, int* count,
                                        int position, int ratio, int budget, int score_stride,
                                        int select_stride) {
    OPT_PDL_ENTRY();
    q4e_pick_pools(scores, selected, count, position, ratio, budget, score_stride, select_stride,
                   false);
}

// GQA attention over the selected keys, with the sigmoid output gate and f32 softmax. A null
// gate leaves the output ungated (GLM-5.3-Flash's DSA).
//
// blockIdx.x is the head and blockIdx.y the token, so one launch covers the whole chunk. The
// kernel is warp-per-slot: warp w owns slots w, w + warps, ... and reduces the head's channels
// with a shuffle instead of a block tree, which is what removes the nine __syncthreads() the
// previous block-per-slot shape spent on every single selected key. At 672 tokens that was about
// six thousand barriers per head per token, and it is why a long prompt measured slower than a
// short one.
//
// The softmax is three block-wide reductions over the whole slot array (max, sum) with the
// normalising divide folded into the value loop, so the only barriers left are one per reduction.
extern "C" __global__ void q4e_attn(const float* __restrict__ q, const float* __restrict__ gate,
                                    const __nv_bfloat16* __restrict__ keys,
                                    const __nv_bfloat16* __restrict__ values,
                                    const int* __restrict__ selected,
                                    const int* __restrict__ selected_count,
                                    float* __restrict__ out, int heads, int kv_heads, int head_dim,
                                    int groups, int select_stride) {
    OPT_PDL_ENTRY();
    __shared__ float logits[Q4E_MAX_SELECTED];
    __shared__ float reduce[Q4E_BLOCK];
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int kv_head = head / groups;
    const int* sel = selected + (unsigned long long)token * select_stride;
    const int nsel = selected_count[token];
    if (head_dim <= 0 || head_dim > Q4E_BLOCK) {
        return;
    }
    const float* qh = q + ((unsigned long long)token * heads + head) * head_dim;
    const float* gh =
        gate != nullptr ? gate + ((unsigned long long)token * heads + head) * head_dim : nullptr;
    float* outh = out + ((unsigned long long)token * heads + head) * head_dim;
    if (nsel <= 0) {
        // Before the indexer has a full block there is nothing to attend over, and the reference
        // keeps the row at zero rather than leaving the buffer untouched.
        for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
            outh[dim] = 0.0f;
        }
        return;
    }
    if (nsel > Q4E_MAX_SELECTED) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const float scale = sqrtf((float)head_dim);

    for (int slot = warp; slot < nsel; slot += warps) {
        const __nv_bfloat16* krow =
            keys + (unsigned long long)sel[slot] * kv_heads * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int dim = lane; dim < head_dim; dim += 32) {
            dot += qh[dim] * __bfloat162float(krow[dim]);
        }
        dot = q4e_warp_sum(dot);
        if (lane == 0) {
            logits[slot] = dot / scale;
        }
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int slot = threadIdx.x; slot < nsel; slot += blockDim.x) {
        local_max = fmaxf(local_max, logits[slot]);
    }
    const float maximum = q4e_block_max(local_max, reduce);
    float local_sum = 0.0f;
    for (int slot = threadIdx.x; slot < nsel; slot += blockDim.x) {
        const float weight = __expf(logits[slot] - maximum);
        logits[slot] = weight;
        local_sum += weight;
    }
    const float total = q4e_block_sum(local_sum, reduce);

    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        float accumulator = 0.0f;
        for (int slot = 0; slot < nsel; ++slot) {
            accumulator += logits[slot] *
                           __bfloat162float(
                               values[(unsigned long long)sel[slot] * kv_heads * head_dim +
                                      kv_head * head_dim + dim]);
        }
        outh[dim] = gh != nullptr ? accumulator / total / (1.0f + __expf(-gh[dim]))
                                  : accumulator / total;
    }
}

// Block reductions of HEADS values at once, each over the same tree as q4e_block_max and
// q4e_block_sum, so each head's result is bit-identical to reducing it alone; scratch holds
// HEADS * blockDim.x floats. Reusable back to back, as those are.
template <int HEADS>
__device__ void q4e_block_max_heads(float* value, float* scratch) {
    const int thread = threadIdx.x;
    const int width = blockDim.x;
#pragma unroll
    for (int head = 0; head < HEADS; ++head) {
        scratch[head * width + thread] = value[head];
    }
    __syncthreads();
    for (int stride = width >> 1; stride > 0; stride >>= 1) {
        if (thread < stride) {
#pragma unroll
            for (int head = 0; head < HEADS; ++head) {
                float* at = scratch + head * width + thread;
                at[0] = fmaxf(at[0], at[stride]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int head = 0; head < HEADS; ++head) {
        value[head] = scratch[head * width];
    }
    __syncthreads();
}

template <int HEADS>
__device__ void q4e_block_sum_heads(float* value, float* scratch) {
    const int thread = threadIdx.x;
    const int width = blockDim.x;
#pragma unroll
    for (int head = 0; head < HEADS; ++head) {
        scratch[head * width + thread] = value[head];
    }
    __syncthreads();
    for (int stride = width >> 1; stride > 0; stride >>= 1) {
        if (thread < stride) {
#pragma unroll
            for (int head = 0; head < HEADS; ++head) {
                float* at = scratch + head * width + thread;
                at[0] += at[stride];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int head = 0; head < HEADS; ++head) {
        value[head] = scratch[head * width];
    }
    __syncthreads();
}

// The query heads q4e_attn_group computes in one CTA; it divides the released configuration's
// twelve query heads a KV head. q4e_attn_group6 takes six, the share of a KV head's twelve each rank
// of a four-rank group holds.
#define Q4E_ATTN_GROUP 4
// The selected tokens q4e_attn_group stages in shared memory: the released configuration's 2,051
// (512 blocks of four and the incomplete tail) rounded up to a multiple of four. The host sends a
// configuration that can select more to q4e_attn.
#define Q4E_ATTN_GROUP_SELECTED 2052
// Keys a warp has in flight in the key pass, and values a thread has in flight in the value pass.
#define Q4E_ATTN_GROUP_KEYS 2
#define Q4E_ATTN_GROUP_VALUES 16

// q4e_attn for HEADS query heads of one KV head at once (q4e_attn_group: Q4E_ATTN_GROUP of them).
//
// Every head is computed exactly as q4e_attn computes it alone: the same lane takes the same
// channels of a key in the same order, the same warp and block trees reduce them, and the value
// sum runs over the slots in order, so the outputs are bit-identical. What changes is the
// reading: a token's selected keys and values, about 1 MiB each at the full budget, are read once for
// the group instead of once a head, and twelve query heads share a KV head. The loads are batched
// ahead of the arithmetic that consumes them, which reorders reads and nothing else.
//
// blockIdx.x is the head group (its first head is blockIdx.x * HEADS) and blockIdx.y the token.
// Dynamic shared memory holds a row of Q4E_ATTN_GROUP_SELECTED logits a head, then a block of
// reduction floats a head, then the token's selection.
template <int HEADS>
__device__ __forceinline__ void q4e_attn_group_heads(const float* __restrict__ q,
                                                     const float* __restrict__ gate,
                                                     const __nv_bfloat16* __restrict__ keys,
                                                     const __nv_bfloat16* __restrict__ values,
                                                     const int* __restrict__ selected,
                                                     const int* __restrict__ selected_count,
                                                     float* __restrict__ out, int heads,
                                                     int kv_heads, int head_dim, int groups,
                                                     int select_stride) {
    constexpr int IN = Q4E_ATTN_GROUP_KEYS;
    constexpr int UNROLL = Q4E_ATTN_GROUP_VALUES;
    extern __shared__ float q4e_attn_shared[];
    float* logits = q4e_attn_shared;
    float* reduce = logits + HEADS * Q4E_ATTN_GROUP_SELECTED;
    int* sel = (int*)(reduce + HEADS * Q4E_BLOCK);
    const int first = blockIdx.x * HEADS;
    const int token = blockIdx.y;
    const int kv_head = first / groups;
    const int nsel = selected_count[token];
    if (head_dim <= 0 || head_dim > Q4E_BLOCK) {
        return;
    }
    const unsigned long long row = ((unsigned long long)token * heads + first) * head_dim;
    if (nsel <= 0) {
        for (int index = threadIdx.x; index < HEADS * head_dim; index += blockDim.x) {
            out[row + index] = 0.0f;
        }
        return;
    }
    if (nsel > Q4E_ATTN_GROUP_SELECTED) {
        return;
    }
    const unsigned long long kv_stride = (unsigned long long)kv_heads * head_dim;
    const unsigned long long kv_offset = (unsigned long long)kv_head * head_dim;
    {
        const int* source = selected + (unsigned long long)token * select_stride;
        for (int slot = threadIdx.x; slot < nsel; slot += blockDim.x) {
            sel[slot] = source[slot];
        }
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const float scale = sqrtf((float)head_dim);

    // The lane's channels of every head's query, held across its slots.
    float query[HEADS][Q4E_BLOCK / 32];
#pragma unroll
    for (int head = 0; head < HEADS; ++head) {
#pragma unroll
        for (int step = 0; step < Q4E_BLOCK / 32; ++step) {
            const int dim = lane + 32 * step;
            query[head][step] = dim < head_dim ? q[row + head * head_dim + dim] : 0.0f;
        }
    }
    __syncthreads();
    // Warp w still owns slots w, w + warps, ...; it loads IN of them before scoring the first. A
    // slot past the selection reloads the base slot and is not written.
    for (int base = warp; base < nsel; base += IN * warps) {
        float key[IN][Q4E_BLOCK / 32];
#pragma unroll
        for (int k = 0; k < IN; ++k) {
            const int slot = base + k * warps < nsel ? base + k * warps : base;
            const __nv_bfloat16* krow =
                keys + (unsigned long long)sel[slot] * kv_stride + kv_offset;
#pragma unroll
            for (int step = 0; step < Q4E_BLOCK / 32; ++step) {
                const int dim = lane + 32 * step;
                key[k][step] =
                    dim < head_dim ? __bfloat162float(krow[dim]) : 0.0f;
            }
        }
#pragma unroll
        for (int k = 0; k < IN; ++k) {
            const int slot = base + k * warps;
            float dot[HEADS];
#pragma unroll
            for (int head = 0; head < HEADS; ++head) {
                dot[head] = 0.0f;
            }
#pragma unroll
            for (int step = 0; step < Q4E_BLOCK / 32; ++step) {
                if (lane + 32 * step < head_dim) {
#pragma unroll
                    for (int head = 0; head < HEADS; ++head) {
                        dot[head] += query[head][step] * key[k][step];
                    }
                }
            }
#pragma unroll
            for (int head = 0; head < HEADS; ++head) {
                dot[head] = q4e_warp_sum(dot[head]);
            }
            if (lane == 0 && slot < nsel) {
#pragma unroll
                for (int head = 0; head < HEADS; ++head) {
                    logits[head * Q4E_ATTN_GROUP_SELECTED + slot] = dot[head] / scale;
                }
            }
        }
    }
    __syncthreads();

    float maximum[HEADS];
#pragma unroll
    for (int head = 0; head < HEADS; ++head) {
        maximum[head] = -INFINITY;
        for (int slot = threadIdx.x; slot < nsel; slot += blockDim.x) {
            maximum[head] = fmaxf(maximum[head], logits[head * Q4E_ATTN_GROUP_SELECTED + slot]);
        }
    }
    q4e_block_max_heads<HEADS>(maximum, reduce);
    float total[HEADS];
#pragma unroll
    for (int head = 0; head < HEADS; ++head) {
        total[head] = 0.0f;
        for (int slot = threadIdx.x; slot < nsel; slot += blockDim.x) {
            const float weight =
                __expf(logits[head * Q4E_ATTN_GROUP_SELECTED + slot] - maximum[head]);
            logits[head * Q4E_ATTN_GROUP_SELECTED + slot] = weight;
            total[head] += weight;
        }
    }
    q4e_block_sum_heads<HEADS>(total, reduce);

    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        float accumulator[HEADS];
#pragma unroll
        for (int head = 0; head < HEADS; ++head) {
            accumulator[head] = 0.0f;
        }
        const __nv_bfloat16* column = values + kv_offset + dim;
        int slot = 0;
        for (; slot + UNROLL <= nsel; slot += UNROLL) {
            float value[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                value[u] = __bfloat162float(
                    column[(unsigned long long)sel[slot + u] * kv_stride]);
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
                for (int head = 0; head < HEADS; ++head) {
                    accumulator[head] +=
                        logits[head * Q4E_ATTN_GROUP_SELECTED + slot + u] * value[u];
                }
            }
        }
        for (; slot < nsel; ++slot) {
            const float value =
                __bfloat162float(column[(unsigned long long)sel[slot] * kv_stride]);
#pragma unroll
            for (int head = 0; head < HEADS; ++head) {
                accumulator[head] += logits[head * Q4E_ATTN_GROUP_SELECTED + slot] * value;
            }
        }
#pragma unroll
        for (int head = 0; head < HEADS; ++head) {
            out[row + head * head_dim + dim] =
                accumulator[head] / total[head] /
                (1.0f + __expf(-gate[row + head * head_dim + dim]));
        }
    }
}

extern "C" __global__ void q4e_attn_group(const float* __restrict__ q,
                                          const float* __restrict__ gate,
                                          const __nv_bfloat16* __restrict__ keys,
                                          const __nv_bfloat16* __restrict__ values,
                                          const int* __restrict__ selected,
                                          const int* __restrict__ selected_count,
                                          float* __restrict__ out, int heads, int kv_heads,
                                          int head_dim, int groups, int select_stride) {
    OPT_PDL_ENTRY();
    q4e_attn_group_heads<Q4E_ATTN_GROUP>(q, gate, keys, values, selected, selected_count, out,
                                         heads, kv_heads, head_dim, groups, select_stride);
}

extern "C" __global__ void q4e_attn_group6(const float* __restrict__ q,
                                           const float* __restrict__ gate,
                                           const __nv_bfloat16* __restrict__ keys,
                                           const __nv_bfloat16* __restrict__ values,
                                           const int* __restrict__ selected,
                                           const int* __restrict__ selected_count,
                                           float* __restrict__ out, int heads, int kv_heads,
                                           int head_dim, int groups, int select_stride) {
    OPT_PDL_ENTRY();
    q4e_attn_group_heads<6>(q, gate, keys, values, selected, selected_count, out, heads,
                            kv_heads, head_dim, groups, select_stride);
}

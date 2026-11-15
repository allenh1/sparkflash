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

// Dory1.0-4B token kernels, TP=1, sm_121a.
//
// A decode step's dense BF16 projections and the embedding are the shared Qwen token kernels
// (q4e_gemv_bf16, q4e_embed); a prompt chunk's projections are dory_mma_wide, an entry on the
// shared wide tensor-core GEMM body. This translation unit owns the arithmetic pieces that are
// Dory's own:
//
//   - dory_rope_norm / dory_rope_norm_lanes: rope (the shared q4e_rope_pair), the nGPT per-head
//     L2 norm with the sqk scale squared on q, and the K/V cache append, for a prompt chunk's
//     rows or a decode pass's lanes.
//   - dory_attn_split / dory_attn_combine (decode) and dory_attn_mma (prefill): causal GQA with
//     the sliding window and the sigmoid output gate; dory_justnorm then takes the L2 norm across
//     all query heads before o_proj.
//   - dory_mlp_epilogue: the suv-scaled SiLU product and its L2 norm before down_proj.
//   - dory_slerp / dory_loop_combine / dory_justnorm / dory_logit_scale: the normalized residual
//     interpolation, the input reinjection at the start of each loop, the state norm and the
//     vocabulary logit scale.
//
// The arithmetic mirrors sparkflash-model-dory/src/forward.rs, the host reference oracle.
//
// Every kernel here opens with OPT_PDL_ENTRY, and the host launches the dory_* symbols with
// programmatic stream serialization like the q4e kernels between them: a launch overlaps the run
// of the kernel before it rather than paying a gap after it, and each kernel still waits for the
// whole previous grid before it reads or writes anything, so stream order holds.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "opt_mma_bf16_sm121.cuh"
#include "opt_pdl_sm121.cuh"
#include "q4e_common_sm121.cuh"
#include "shared_attn_mma_sm121.cuh"
#include "shared_prefill_mma_sm121.cuh"

#define DORY_BLOCK 256

// Dory's prompt-chunk projections: f32 activation rows against the checkpoint's row-major BF16
// weights on the shared wide tile (shared_prefill_wide_body, GLM's bulk-prefill BF16 GEMM), 128
// tokens x 256 output columns a block. Each activation is split into a BF16 head and the BF16 of
// its remainder, two MMAs a fragment accumulated in f32, head then remainder at each 16-K step in
// ascending K. It replaced a 128 x 128 tile on the opt_mma_bf16_body with the same chain per
// output, and so the same bits: reading each f32 activation tile once a 256 columns rather than
// a 128 made prompt projections 23% faster.
// Grid (ceil(rows / 256), ceil(tokens / 128)), 256 threads, GLM_WIDE_SHARED bytes of dynamic
// shared memory; a contiguous [tokens][inner] activation and [tokens][rows] output.
extern "C" __global__ void __launch_bounds__(OM_THREADS) dory_mma_wide(
    const __nv_bfloat16* __restrict__ w, const float* __restrict__ a, float* __restrict__ out,
    uint32_t cols, uint32_t inner, uint32_t tokens) {
    OPT_PDL_ENTRY();
    shared_prefill_wide_body<true>(w, a, out, cols, inner, tokens);
}

// dory_mma_wide over up to four matrices of one activation in one launch: q, k, v and g, or
// gate and up. Matrix j has cols_j output columns (a multiple of 32; zero for an unused slot)
// and its own output; the grid's x is the matrices' 256-column tiles end to end, so a block's
// tile is the one dory_mma_wide gives it and every output keeps that kernel's chain. Separate
// launches leave each one's last wave part empty: at a 1,024-row chunk k and v are 32 blocks
// on 48 SMs, a whole wave each.
// Grid (sum of ceil(cols_j / 256), ceil(tokens / 128)), 256 threads, GLM_WIDE_SHARED bytes.
extern "C" __global__ void __launch_bounds__(OM_THREADS) dory_mma_wide4(
    const __nv_bfloat16* __restrict__ w0, const __nv_bfloat16* __restrict__ w1,
    const __nv_bfloat16* __restrict__ w2, const __nv_bfloat16* __restrict__ w3,
    const float* __restrict__ a, float* __restrict__ out0, float* __restrict__ out1,
    float* __restrict__ out2, float* __restrict__ out3, uint32_t cols0, uint32_t cols1,
    uint32_t cols2, uint32_t cols3, uint32_t inner, uint32_t tokens) {
    OPT_PDL_ENTRY();
    const uint2 place = glm_raster();
    const uint32_t tiles0 = (cols0 + GLM_WIDE_BN - 1u) / GLM_WIDE_BN;
    const uint32_t tiles1 = (cols1 + GLM_WIDE_BN - 1u) / GLM_WIDE_BN;
    const uint32_t tiles2 = (cols2 + GLM_WIDE_BN - 1u) / GLM_WIDE_BN;
    uint32_t tile = place.x;
    const __nv_bfloat16* w = w0;
    float* out = out0;
    uint32_t cols = cols0;
    if (tile >= tiles0) {
        tile -= tiles0;
        w = w1, out = out1, cols = cols1;
        if (tile >= tiles1) {
            tile -= tiles1;
            w = w2, out = out2, cols = cols2;
            if (tile >= tiles2) {
                tile -= tiles2;
                w = w3, out = out3, cols = cols3;
            }
        }
    }
    shared_prefill_wide_tile<true>(w, a, out, cols, inner, tokens, tile * GLM_WIDE_BN,
                                   place.y * GLM_BM);
}

#define DORY_EPSILON 1e-6f

__device__ __forceinline__ float dory_block_sum(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ __forceinline__ float dory_block_max(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return scratch[0];
}

// A decode pass's lanes. Each of up to DORY_LANES rows of a batched decode pass is a sequence of
// its own, with its own position and cache, and one launch serves them all: grid z (or y) is the
// lane, and each lane's numbers arrive as launch parameters, so a pass needs no table upload.
// Lane i's word is its position (low 32 bits; 0xffffffff for a lane that sits the pass out) and
// dory_attn_split's run length (high 32 bits); its cache pointer is the K plane of its cache for
// the block, the V plane values_offset elements after it. A lane's arithmetic is the one-lane
// launch's whichever lanes share the launch.
#define DORY_LANES 8
#define DORY_LANE_IDLE 0xffffffffu
#define DORY_LANE_ARGS(name)                                                                     \
    unsigned long long name##0, unsigned long long name##1, unsigned long long name##2,         \
        unsigned long long name##3, unsigned long long name##4, unsigned long long name##5,     \
        unsigned long long name##6, unsigned long long name##7
#define DORY_LANE_PICK(name, i)                                                                  \
    ((i) == 0   ? name##0                                                                        \
     : (i) == 1 ? name##1                                                                        \
     : (i) == 2 ? name##2                                                                        \
     : (i) == 3 ? name##3                                                                        \
     : (i) == 4 ? name##4                                                                        \
     : (i) == 5 ? name##5                                                                        \
     : (i) == 6 ? name##6                                                                        \
                : name##7)

// The keys [first, position] a decode row attends to, cut into runs of `span`: the run count.
__device__ __forceinline__ int dory_decode_first(int position, int window) {
    return (window > 0 && position + 1 > window) ? position + 1 - window : 0;
}
__device__ __forceinline__ int dory_decode_splits(int position, int window, int span) {
    const int keys = position + 1 - dory_decode_first(position, window);
    return (keys + span - 1) / span;
}

// The rope, nGPT per-head L2 norm and cache append of a token's attention rows, one block a
// head (DORY_BLOCK threads). The head is rotated through q4e_rope_pair, q4e_rope's arithmetic,
// then normalized; a query head also takes (sqk * qk_factor)^2, and a kv head writes its K and
// its V into the BF16 cache row position % cache_rows, which wraps for a sliding block.
// It stands for q4e_rope on q and on k, a per-head norm kernel and the two cache appends: every
// value and sum is theirs, in their order, so the bits are too. head_dim even and at most
// DORY_HEAD_MAX; k, v and the cache planes are kv_stride floats a row.
#define DORY_HEAD_MAX 256

__device__ __forceinline__ void dory_rope_norm_row(
    float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v,
    __nv_bfloat16* __restrict__ keys, __nv_bfloat16* __restrict__ values,
    const __nv_bfloat16* __restrict__ sqk, int head, int q_heads, int head_dim, int position,
    float theta, float qk_factor, int kv_stride, int cache_rows, float* scratch, float* rotated) {
    const int is_query = head < q_heads;
    const int local_head = is_query ? head : head - q_heads;
    const unsigned long long offset = (unsigned long long)local_head * head_dim;
    float* base = is_query ? q + offset : k + offset;
    const float position_f = (float)position;
    const int half = head_dim / 2;
    for (int index = threadIdx.x; index < half; index += blockDim.x) {
        q4e_rope_pair(base, rotated, index, half, head_dim, position_f, theta);
    }
    __syncthreads();
    float sum = 0.0f;
    for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
        float value = rotated[index];
        sum += value * value;
    }
    const float scale = rsqrtf(dory_block_sum(sum, scratch) + DORY_EPSILON);
    if (is_query) {
        for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
            float value = rotated[index] * scale;
            float factor = __bfloat162float(sqk[offset + index]) * qk_factor;
            value *= factor * factor;
            base[index] = value;
        }
        return;
    }
    const unsigned long long row =
        (unsigned long long)(position % cache_rows) * kv_stride + offset;
    const float* v_row = v + offset;
    for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
        const float value = rotated[index] * scale;
        base[index] = value;
        keys[row + index] = __float2bfloat16_rn(value);
        values[row + index] = __float2bfloat16_rn(v_row[index]);
    }
}

// A prompt chunk's rows into one cache. Grid (q_heads + kv_heads, tokens); token t is at
// position + t; q rows are q_stride floats apart and k and v rows kv_stride.
extern "C" __global__ void __launch_bounds__(DORY_BLOCK) dory_rope_norm(
    float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v,
    __nv_bfloat16* __restrict__ keys, __nv_bfloat16* __restrict__ values,
    const __nv_bfloat16* __restrict__ sqk, int q_heads, int head_dim, int position,
    float theta, float qk_factor, int q_stride, int kv_stride, int cache_rows) {
    OPT_PDL_ENTRY();
    __shared__ float scratch[DORY_BLOCK];
    __shared__ float rotated[DORY_HEAD_MAX];
    const int token = blockIdx.y;
    dory_rope_norm_row(q + (unsigned long long)token * q_stride,
                       k + (unsigned long long)token * kv_stride,
                       v + (unsigned long long)token * kv_stride, keys, values, sqk, blockIdx.x,
                       q_heads, head_dim, position + token, theta, qk_factor, kv_stride,
                       cache_rows, scratch, rotated);
}

// A decode pass's rows, row i lane i's at its own position into its own cache (see DORY_LANES).
// Grid (q_heads + kv_heads, lanes).
extern "C" __global__ void __launch_bounds__(DORY_BLOCK) dory_rope_norm_lanes(
    float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v,
    const __nv_bfloat16* __restrict__ sqk, int q_heads, int head_dim, float theta,
    float qk_factor, int q_stride, int kv_stride, int cache_rows,
    unsigned long long values_offset, DORY_LANE_ARGS(lane), DORY_LANE_ARGS(cache)) {
    OPT_PDL_ENTRY();
    __shared__ float scratch[DORY_BLOCK];
    __shared__ float rotated[DORY_HEAD_MAX];
    const int row = blockIdx.y;
    const unsigned int position = (unsigned int)DORY_LANE_PICK(lane, row);
    if (position == DORY_LANE_IDLE) {
        return;
    }
    __nv_bfloat16* keys = reinterpret_cast<__nv_bfloat16*>(DORY_LANE_PICK(cache, row));
    dory_rope_norm_row(q + (unsigned long long)row * q_stride,
                       k + (unsigned long long)row * kv_stride,
                       v + (unsigned long long)row * kv_stride, keys, keys + values_offset, sqk,
                       blockIdx.x, q_heads, head_dim, (int)position, theta, qk_factor, kv_stride,
                       cache_rows, scratch, rotated);
}

// Decode attention, split over the keys ("flash-decoding"): one token row of causal GQA over the
// BF16 cache, with the sliding window, the ring index and the sigmoid output gate. (The L2 norm
// across all query heads before o_proj is dory_justnorm's.)
//
// dory_attn_split cuts the row's keys [first, position] into `splits` runs of `span` keys. Block
// (kv head, split) reads its run's K and V rows once for every query head that shares the kv
// head, so a row is read once rather than once a query head, and the grid covers the GPU however
// short the window is. A lane owns 8 consecutive head dimensions (one 16-byte load a row); warp w
// takes keys start + w, start + w + 8, ..., and keeps an online softmax a query head: the score
// is the lane products summed by an xor butterfly, so every lane holds it. The block then merges
// its eight warps and writes one partial a query head and split: the running maximum, the sum of
// exponentials and the head_dim weighted value sums. dory_attn_combine merges a head's splits in
// split order, divides and applies the gate.
//
// The order is fixed by (position, window, span): the same row always reduces the same way.
// Both run a decode pass's lanes in one launch (see DORY_LANES): lane i's q, gate and output are
// row i (q_stride floats a row), its partials lane_partials floats from lane i - 1's.
// dory_attn_split's grid is (kv_heads, the lanes' most splits, lanes), and a block past its
// lane's splits does nothing; DORY_BLOCK threads; needs head_dim % 8 == 0, head_dim <= 256 and at
// most DORY_GQA_MAX query heads a kv head (the host refuses any other shape).
#define DORY_GQA_MAX 4
#define DORY_SPLIT_WARPS (DORY_BLOCK / 32)
#define DORY_PARTIAL_STRIDE(head_dim) ((head_dim) + 2)

__device__ __forceinline__ float dory_warp_allsum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ void dory_load8(const __nv_bfloat16* __restrict__ row, float* out) {
    const uint4 packed = *reinterpret_cast<const uint4*>(row);
    const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const float2 values = __bfloat1622float2(pairs[pair]);
        out[2 * pair] = values.x;
        out[2 * pair + 1] = values.y;
    }
}

extern "C" __global__ void __launch_bounds__(DORY_BLOCK) dory_attn_split(
    const float* __restrict__ q, float* __restrict__ partial, int heads, int kv_heads,
    int head_dim, int window, int cache_rows, int q_stride, unsigned long long values_offset,
    unsigned long long lane_partials, DORY_LANE_ARGS(lane), DORY_LANE_ARGS(cache)) {
    OPT_PDL_ENTRY();
    __shared__ float warp_max[DORY_SPLIT_WARPS][DORY_GQA_MAX];
    __shared__ float warp_sum[DORY_SPLIT_WARPS][DORY_GQA_MAX];
    __shared__ float warp_acc[DORY_SPLIT_WARPS][DORY_GQA_MAX][DORY_HEAD_MAX];
    const int row_lane = blockIdx.z;
    const unsigned long long word = DORY_LANE_PICK(lane, row_lane);
    if ((unsigned int)word == DORY_LANE_IDLE) {
        return;
    }
    const int position = (int)(unsigned int)word;
    const int span = (int)(word >> 32);
    const int splits = dory_decode_splits(position, window, span);
    const int split = blockIdx.y;
    if (split >= splits) {
        return;
    }
    const __nv_bfloat16* keys =
        reinterpret_cast<const __nv_bfloat16*>(DORY_LANE_PICK(cache, row_lane));
    const __nv_bfloat16* values = keys + values_offset;
    q += (unsigned long long)row_lane * q_stride;
    partial += (unsigned long long)row_lane * lane_partials;
    const int kv_head = blockIdx.x;
    const int group = heads / kv_heads;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int dim0 = lane * 8;
    const bool active = dim0 < head_dim;
    const float score_scale = rsqrtf((float)head_dim);
    const unsigned long long kv_row = (unsigned long long)kv_heads * head_dim;
    const int first = dory_decode_first(position, window);
    const int start = first + split * span;
    const int end = min(start + span, position + 1);

    float query[DORY_GQA_MAX][8];
    float maximum[DORY_GQA_MAX];
    float denominator[DORY_GQA_MAX];
    float accumulator[DORY_GQA_MAX][8];
#pragma unroll
    for (int h = 0; h < DORY_GQA_MAX; ++h) {
        const float* q_head = q + (unsigned long long)(kv_head * group + h) * head_dim + dim0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            query[h][i] = (h < group && active) ? q_head[i] : 0.0f;
            accumulator[h][i] = 0.0f;
        }
        maximum[h] = -INFINITY;
        denominator[h] = 0.0f;
    }
    for (int key = start + warp; key < end; key += DORY_SPLIT_WARPS) {
        const int row_index = cache_rows > 0 ? key % cache_rows : key;
        const unsigned long long offset = (unsigned long long)row_index * kv_row +
                                          (unsigned long long)kv_head * head_dim + dim0;
        float key_row[8];
        float value_row[8];
        if (active) {
            dory_load8(keys + offset, key_row);
            dory_load8(values + offset, value_row);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                key_row[i] = 0.0f;
                value_row[i] = 0.0f;
            }
        }
#pragma unroll
        for (int h = 0; h < DORY_GQA_MAX; ++h) {
            if (h >= group) {
                continue;
            }
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                dot += query[h][i] * key_row[i];
            }
            const float score = dory_warp_allsum(dot) * score_scale;
            const float next = fmaxf(maximum[h], score);
            const float rescale = expf(maximum[h] - next);
            const float weight = expf(score - next);
            denominator[h] = denominator[h] * rescale + weight;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accumulator[h][i] = accumulator[h][i] * rescale + weight * value_row[i];
            }
            maximum[h] = next;
        }
    }
#pragma unroll
    for (int h = 0; h < DORY_GQA_MAX; ++h) {
        if (h >= group) {
            continue;
        }
        if (lane == 0) {
            warp_max[warp][h] = maximum[h];
            warp_sum[warp][h] = denominator[h];
        }
        if (active) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                warp_acc[warp][h][dim0 + i] = accumulator[h][i];
            }
        }
    }
    __syncthreads();
    for (int index = threadIdx.x; index < group * head_dim; index += blockDim.x) {
        const int h = index / head_dim;
        const int dim = index - h * head_dim;
        float merged = -INFINITY;
#pragma unroll
        for (int w = 0; w < DORY_SPLIT_WARPS; ++w) {
            merged = fmaxf(merged, warp_max[w][h]);
        }
        float sum = 0.0f;
        float value = 0.0f;
#pragma unroll
        for (int w = 0; w < DORY_SPLIT_WARPS; ++w) {
            // A warp past the run's end saw no key: its maximum is -inf and its weight 0.
            const float weight = expf(warp_max[w][h] - merged);
            sum += warp_sum[w][h] * weight;
            value += warp_acc[w][h][dim] * weight;
        }
        float* out = partial + ((unsigned long long)(kv_head * group + h) * splits + split) *
                                   DORY_PARTIAL_STRIDE(head_dim);
        if (dim == 0) {
            out[0] = merged;
            out[1] = sum;
        }
        out[2 + dim] = value;
    }
}

// Merges one head's split partials in split order, normalizes and applies the sigmoid gate.
// Grid (heads, lanes), DORY_BLOCK threads.
extern "C" __global__ void __launch_bounds__(DORY_BLOCK) dory_attn_combine(
    const float* __restrict__ partial, const float* __restrict__ gate,
    const __nv_bfloat16* __restrict__ s_gate, float* __restrict__ out, int head_dim,
    float gate_factor, int window, int q_stride, unsigned long long lane_partials,
    DORY_LANE_ARGS(lane)) {
    OPT_PDL_ENTRY();
    const int row_lane = blockIdx.y;
    const unsigned long long word = DORY_LANE_PICK(lane, row_lane);
    if ((unsigned int)word == DORY_LANE_IDLE) {
        return;
    }
    const int splits = dory_decode_splits((int)(unsigned int)word, window, (int)(word >> 32));
    gate += (unsigned long long)row_lane * q_stride;
    out += (unsigned long long)row_lane * q_stride;
    const int head = blockIdx.x;
    const float* base = partial + (unsigned long long)row_lane * lane_partials +
                        (unsigned long long)head * splits * DORY_PARTIAL_STRIDE(head_dim);
    float merged = -INFINITY;
    for (int split = 0; split < splits; ++split) {
        merged = fmaxf(merged, base[(unsigned long long)split * DORY_PARTIAL_STRIDE(head_dim)]);
    }
    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        float sum = 0.0f;
        float value = 0.0f;
        for (int split = 0; split < splits; ++split) {
            const float* run = base + (unsigned long long)split * DORY_PARTIAL_STRIDE(head_dim);
            const float weight = expf(run[0] - merged);
            sum += run[1] * weight;
            value += run[2 + dim] * weight;
        }
        const int local = head * head_dim + dim;
        const float sg = __bfloat162float(s_gate[local]) * gate_factor;
        out[local] = (value / sum) / (1.0f + expf(-gate[local] * sg * sg));
    }
}

// Prefill attention on the tensor cores: the shared body (shared_attn_mma_sm121.cuh), whose
// geometry, staging, MMAs and online softmax are this kernel's own lifted out unchanged so
// Inkling's prompt attention runs the same loop. Dory's part is the epilogue: the sigmoid gate
// with the s_gate scale squared, into an f32 output. Its scores are the dot over sqrt(D), its
// weights the softmax's own, and its softmax opens empty, so every other hook is the identity.
// Grid (ceil(tokens / SHARED_ATTN_Q), kv_heads), group x 32 threads (group <= DORY_GQA_MAX),
// dynamic shared memory (2 group SHARED_ATTN_Q + 2 SHARED_ATTN_KEYS) x D x 2 bytes.
template <int D>
struct DoryAttnPolicy {
    const float* __restrict__ gate;
    const __nv_bfloat16* __restrict__ s_gate;
    float* __restrict__ out;
    float gate_factor;
    int q_stride;
    int out_stride;

    __device__ __forceinline__ void keys(uint32_t, int, int, int) const {}
    __device__ __forceinline__ void values(uint32_t, int, int, int) const {}
    __device__ __forceinline__ void start(int, int, int, int, int, int) const {}
    __device__ __forceinline__ float open_max(int, int, int, int) const { return -INFINITY; }
    __device__ __forceinline__ float open_sum(int, int, int, int, int) const { return 0.0f; }
    __device__ __forceinline__ void tile(uint32_t, int, int, int, int, int) const {}
    __device__ __forceinline__ float score(float scaled, uint32_t, int, int, int, int,
                                           int) const {
        return scaled;
    }
    __device__ __forceinline__ float weight(float p, uint32_t, int, int, int) const { return p; }
    __device__ __forceinline__ void store(unsigned long long token, int head, int n, int t4,
                                          float v0, float v1, float denominator) const {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int local = head * D + 8 * n + 2 * t4 + e;
            const float sg = __bfloat162float(s_gate[local]) * gate_factor;
            const float value = e == 0 ? v0 : v1;
            out[token * out_stride + local] =
                (value / denominator) / (1.0f + expf(-gate[token * q_stride + local] * sg * sg));
        }
    }
};

#define DORY_ATTN_MMA(D)                                                                          \
    extern "C" __global__ void __launch_bounds__(DORY_GQA_MAX * 32, 1) dory_attn_mma##D(          \
        const float* __restrict__ q, const __nv_bfloat16* __restrict__ keys,                      \
        const __nv_bfloat16* __restrict__ values, const float* __restrict__ gate,                 \
        const __nv_bfloat16* __restrict__ s_gate, float* __restrict__ out, int position,          \
        int tokens, int heads, int kv_heads, int window, float gate_factor, int q_stride,         \
        int out_stride, int cache_rows) {                                                         \
        OPT_PDL_ENTRY();                                                                          \
        DoryAttnPolicy<D> policy{gate, s_gate, out, gate_factor, q_stride, out_stride};          \
        shared_attn_mma_body<D>(q, keys, values, position, tokens, heads, kv_heads, window,       \
                                q_stride, cache_rows, policy);                                    \
    }
DORY_ATTN_MMA(16)
DORY_ATTN_MMA(64)
DORY_ATTN_MMA(128)
DORY_ATTN_MMA(256)

// The norm kernels below are one block a row. A thread takes the elements index = thread +
// k * blockDim.x in ascending k, sums their squares in that order and reduces through
// dory_block_sum, then scales each. When a row fits DORY_HELD_HIDDEN (or, for the MLP's
// intermediate, DORY_HELD_INNER) elements a thread, the thread loads all of its elements at once
// and keeps them in registers for the scale, rather than one dependent load an iteration and a
// second read of what it wrote: the sums, products and their order are the loop's, so are the
// bits. A wider row takes the loop.
#define DORY_HELD_HIDDEN 16
#define DORY_HELD_INNER 32

__device__ __forceinline__ float dory_silu_product(float gate, float up, float suv_gate,
                                                   float suv_up, float factor) {
    const float g = gate * suv_gate * factor;
    const float u = up * suv_up * factor;
    return (g / (1.0f + expf(-g))) * u;
}

// out = justnorm(silu(gate * suv_gate * factor) * (up * suv_up * factor)), one block.
extern "C" __global__ void dory_mlp_epilogue(const float* __restrict__ gate,
                                             const float* __restrict__ up,
                                             const __nv_bfloat16* __restrict__ suv_gate,
                                             const __nv_bfloat16* __restrict__ suv_up,
                                             float* __restrict__ out, int width, float factor) {
    OPT_PDL_ENTRY();
    __shared__ float scratch[DORY_BLOCK];
    const int token = blockIdx.y;
    gate += (unsigned long long)token * width;
    up += (unsigned long long)token * width;
    out += (unsigned long long)token * width;
    float sum = 0.0f;
    if (width <= DORY_HELD_INNER * (int)blockDim.x) {
        // Every operand is loaded before the first product. Each product's IEEE division has a
        // slow-path call, and the compiler issues no load past one, so loads taken inside the
        // product loop waited a full L2 round trip an element.
        float held[DORY_HELD_INNER];
        float ups[DORY_HELD_INNER];
        __nv_bfloat16 gate_scales[DORY_HELD_INNER];
        __nv_bfloat16 up_scales[DORY_HELD_INNER];
#pragma unroll
        for (int k = 0; k < DORY_HELD_INNER; ++k) {
            const int index = threadIdx.x + k * blockDim.x;
            if (index < width) {
                held[k] = gate[index];
                ups[k] = up[index];
                gate_scales[k] = suv_gate[index];
                up_scales[k] = suv_up[index];
            }
        }
#pragma unroll
        for (int k = 0; k < DORY_HELD_INNER; ++k) {
            const int index = threadIdx.x + k * blockDim.x;
            if (index < width) {
                held[k] = dory_silu_product(held[k], ups[k], __bfloat162float(gate_scales[k]),
                                            __bfloat162float(up_scales[k]), factor);
                sum += held[k] * held[k];
            }
        }
        const float scale = rsqrtf(dory_block_sum(sum, scratch) + DORY_EPSILON);
#pragma unroll
        for (int k = 0; k < DORY_HELD_INNER; ++k) {
            const int index = threadIdx.x + k * blockDim.x;
            if (index < width) {
                out[index] = held[k] * scale;
            }
        }
        return;
    }
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        const float value =
            dory_silu_product(gate[index], up[index], __bfloat162float(suv_gate[index]),
                              __bfloat162float(suv_up[index]), factor);
        out[index] = value;
        sum += value * value;
    }
    const float scale = rsqrtf(dory_block_sum(sum, scratch) + DORY_EPSILON);
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        out[index] *= scale;
    }
}

// out = justnorm(justnorm(a) + |alpha * scale| * (justnorm(b) - justnorm(a))), one block.
extern "C" __global__ void dory_slerp(const float* __restrict__ a, const float* __restrict__ b,
                                     const __nv_bfloat16* __restrict__ alpha,
                                     float* __restrict__ out, int width, float alpha_scale) {
    OPT_PDL_ENTRY();
    __shared__ float scratch_a[DORY_BLOCK];
    __shared__ float scratch_b[DORY_BLOCK];
    __shared__ float scratch[DORY_BLOCK];
    const int token = blockIdx.y;
    a += (unsigned long long)token * width;
    b += (unsigned long long)token * width;
    out += (unsigned long long)token * width;
    float sum_a = 0.0f;
    float sum_b = 0.0f;
    if (width <= DORY_HELD_HIDDEN * (int)blockDim.x) {
        float held_a[DORY_HELD_HIDDEN];
        float held_b[DORY_HELD_HIDDEN];
#pragma unroll
        for (int k = 0; k < DORY_HELD_HIDDEN; ++k) {
            const int index = threadIdx.x + k * blockDim.x;
            if (index < width) {
                held_a[k] = a[index];
                held_b[k] = b[index];
                sum_a += held_a[k] * held_a[k];
                sum_b += held_b[k] * held_b[k];
            }
        }
        const float scale_a = rsqrtf(dory_block_sum(sum_a, scratch_a) + DORY_EPSILON);
        const float scale_b = rsqrtf(dory_block_sum(sum_b, scratch_b) + DORY_EPSILON);
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < DORY_HELD_HIDDEN; ++k) {
            const int index = threadIdx.x + k * blockDim.x;
            if (index < width) {
                const float ya = held_a[k] * scale_a;
                const float yb = held_b[k] * scale_b;
                const float lr = fabsf(__bfloat162float(alpha[index]) * alpha_scale);
                held_a[k] = ya + lr * (yb - ya);
                sum += held_a[k] * held_a[k];
            }
        }
        const float scale = rsqrtf(dory_block_sum(sum, scratch) + DORY_EPSILON);
#pragma unroll
        for (int k = 0; k < DORY_HELD_HIDDEN; ++k) {
            const int index = threadIdx.x + k * blockDim.x;
            if (index < width) {
                out[index] = held_a[k] * scale;
            }
        }
        return;
    }
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        const float av = a[index];
        const float bv = b[index];
        sum_a += av * av;
        sum_b += bv * bv;
    }
    const float scale_a = rsqrtf(dory_block_sum(sum_a, scratch_a) + DORY_EPSILON);
    const float scale_b = rsqrtf(dory_block_sum(sum_b, scratch_b) + DORY_EPSILON);
    float sum = 0.0f;
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        const float ya = a[index] * scale_a;
        const float yb = b[index] * scale_b;
        const float lr = fabsf(__bfloat162float(alpha[index]) * alpha_scale);
        const float value = ya + lr * (yb - ya);
        out[index] = value;
        sum += value * value;
    }
    const float scale = rsqrtf(dory_block_sum(sum, scratch) + DORY_EPSILON);
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        out[index] *= scale;
    }
}

// out = justnorm(state + input), the input reinjection at the start of each recurrent loop.
extern "C" __global__ void dory_loop_combine(const float* __restrict__ state,
                                             const float* __restrict__ input,
                                             float* __restrict__ out, int width) {
    OPT_PDL_ENTRY();
    __shared__ float scratch[DORY_BLOCK];
    const int token = blockIdx.y;
    state += (unsigned long long)token * width;
    input += (unsigned long long)token * width;
    out += (unsigned long long)token * width;
    float sum = 0.0f;
    if (width <= DORY_HELD_HIDDEN * (int)blockDim.x) {
        float held[DORY_HELD_HIDDEN];
#pragma unroll
        for (int k = 0; k < DORY_HELD_HIDDEN; ++k) {
            const int index = threadIdx.x + k * blockDim.x;
            if (index < width) {
                held[k] = state[index] + input[index];
                sum += held[k] * held[k];
            }
        }
        const float scale = rsqrtf(dory_block_sum(sum, scratch) + DORY_EPSILON);
#pragma unroll
        for (int k = 0; k < DORY_HELD_HIDDEN; ++k) {
            const int index = threadIdx.x + k * blockDim.x;
            if (index < width) {
                out[index] = held[k] * scale;
            }
        }
        return;
    }
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        const float value = state[index] + input[index];
        out[index] = value;
        sum += value * value;
    }
    const float scale = rsqrtf(dory_block_sum(sum, scratch) + DORY_EPSILON);
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        out[index] *= scale;
    }
}

// In-place justnorm over one vector.
extern "C" __global__ void dory_justnorm(float* __restrict__ values, int width) {
    OPT_PDL_ENTRY();
    __shared__ float scratch[DORY_BLOCK];
    values += (unsigned long long)blockIdx.y * width;
    float sum = 0.0f;
    if (width <= DORY_HELD_HIDDEN * (int)blockDim.x) {
        float held[DORY_HELD_HIDDEN];
#pragma unroll
        for (int k = 0; k < DORY_HELD_HIDDEN; ++k) {
            const int index = threadIdx.x + k * blockDim.x;
            if (index < width) {
                held[k] = values[index];
                sum += held[k] * held[k];
            }
        }
        const float scale = rsqrtf(dory_block_sum(sum, scratch) + DORY_EPSILON);
#pragma unroll
        for (int k = 0; k < DORY_HELD_HIDDEN; ++k) {
            const int index = threadIdx.x + k * blockDim.x;
            if (index < width) {
                values[index] = held[k] * scale;
            }
        }
        return;
    }
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        const float value = values[index];
        sum += value * value;
    }
    const float scale = rsqrtf(dory_block_sum(sum, scratch) + DORY_EPSILON);
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        values[index] *= scale;
    }
}

// logits[i] *= logit_scale[i] / divisor.
extern "C" __global__ void dory_logit_scale(const __nv_bfloat16* __restrict__ scale,
                                            float* __restrict__ logits, int width,
                                            float divisor) {
    OPT_PDL_ENTRY();
    // One vocabulary row a grid row: a batched decode pass leaves a row a lane.
    logits += (size_t)blockIdx.y * width;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < width;
         index += blockDim.x * gridDim.x) {
        logits[index] *= __bfloat162float(scale[index]) / divisor;
    }
}

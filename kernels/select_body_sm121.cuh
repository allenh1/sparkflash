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

// kernels/select_body_sm121.cuh
//
// The candidate-select body ref_select_sm121 and opt_select_sm121 had two copies of, and the two
// helpers it reads. The reference's score pass is one position a thread, the dot ascending inside
// a head (the host's order, O(n_pos^2) and deliberately exact); the optimized entry is one warp a
// position with the same head order and a warp tree inside a head, or scores an earlier pass
// already wrote. WARP picks the pass; every later phase - the HSI block mask, the reach and mask
// application, the rank and the mapped emission - is one statement here.

#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace spark_select {

constexpr int MAX_POS = 4096;
constexpr int MAX_BLOCKS = 4096;
constexpr int BLOCK = 256;
constexpr int MAX_MASK_WORDS = MAX_POS / 32;

__device__ __forceinline__ float round_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

// true when (score_a, id_a) stands before (score_b, id_b): score descending, then id ascending.
__device__ __forceinline__ bool before(float score_a, uint32_t id_a, float score_b, uint32_t id_b) {
    if (score_a > score_b) {
        return true;
    }
    if (score_a < score_b) {
        return false;
    }
    return id_a < id_b;
}

// WARP false: the reference's serial score pass. WARP true: the warp pass, or the pre_scores
// array when the caller already ran one. The rest of the body is identical in both entries.
template <bool WARP>
__device__ __forceinline__ void body(
    const float* __restrict__ pre_scores, const float* __restrict__ q_index,
    const float* __restrict__ keys, const float* __restrict__ weights,
    uint32_t* __restrict__ out_ids, float* __restrict__ out_scores,
    const uint32_t* __restrict__ mask_in, uint32_t* __restrict__ mask_out, uint32_t n_pos,
    uint32_t heads, uint32_t head_dim, uint32_t reach, uint32_t topk, uint32_t offset,
    uint32_t mask_mode, uint32_t topk_blocks, uint32_t block_size, uint32_t no_index,
    uint32_t flags) {
    (void)flags;
    static_assert(BLOCK % 32 == 0, "one whole warp per score-generation step");
    static_assert(MAX_BLOCKS >= MAX_POS, "the block table covers every position");
    __shared__ float s_score[MAX_POS];
    __shared__ uint32_t s_mask[MAX_MASK_WORDS];
    __shared__ union {
        float f[MAX_BLOCKS];
        uint32_t u[MAX_BLOCKS];
    } scratch;

    const uint32_t tid = threadIdx.x;
    const uint32_t n = n_pos < MAX_POS ? n_pos : MAX_POS;

    if (mask_mode == 2u) {
        for (uint32_t word = tid; word < MAX_MASK_WORDS; word += BLOCK) {
            s_mask[word] = 0u;
        }
    } else if (mask_mode == 1u) {
        for (uint32_t word = tid; word < MAX_MASK_WORDS; word += BLOCK) {
            s_mask[word] = mask_in[word];
        }
    }
    __syncthreads();

    if constexpr (WARP) {
        if (pre_scores != nullptr) {
            for (uint32_t position = tid; position < n; position += BLOCK) {
                s_score[position] = pre_scores[position];
            }
        } else {
            // Scores: warp per position, lanes over head_dim; the heads are folded in ascending
            // order as the reference does, each head's dot is a warp tree.
            const uint32_t lane = tid & 31u;
            const uint32_t warp = tid >> 5;
            for (uint32_t position = warp; position < n; position += BLOCK / 32u) {
                const float* key = keys + static_cast<size_t>(position) * head_dim;
                float sum = 0.0f;
                for (uint32_t head = 0; head < heads; ++head) {
                    const float* q = q_index + static_cast<size_t>(head) * head_dim;
                    float acc = 0.0f;
                    for (uint32_t d = lane; d < head_dim; d += 32u) {
                        acc = fmaf(q[d], key[d], acc);
                    }
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
                    const float raw = round_bf16(acc);
                    if (raw > 0.0f) {
                        sum = __fadd_rn(sum, round_bf16(__fmul_rn(raw, weights[head])));
                    }
                }
                if (lane == 0u) s_score[position] = round_bf16(sum);
            }
        }
    } else {
        // Scores: one position per thread pass, heads ascending and the dot ascending inside a
        // head. This is the host's order and is what the exactness claim rests on.
        for (uint32_t position = tid; position < n; position += BLOCK) {
            float sum = 0.0f;
            const float* key = keys + static_cast<size_t>(position) * head_dim;
            for (uint32_t head = 0; head < heads; ++head) {
                const float* q = q_index + static_cast<size_t>(head) * head_dim;
                float acc = 0.0f;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    acc = __fadd_rn(acc, __fmul_rn(q[d], key[d]));
                }
                const float raw = round_bf16(acc);
                if (raw > 0.0f) {
                    sum = __fadd_rn(sum, round_bf16(__fmul_rn(raw, weights[head])));
                }
            }
            s_score[position] = round_bf16(sum);
        }
    }
    __syncthreads();

    // Level one: the HSI candidate block mask. Block scores live in the scratch union while this
    // phase runs; the mask is then written one whole word per thread, so no atomic is needed.
    if (mask_mode == 2u && block_size != 0u && n != 0u) {
        const uint32_t num_blocks = (n + block_size - 1u) / block_size;
        for (uint32_t block = tid; block < num_blocks; block += BLOCK) {
            const uint32_t start = block * block_size;
            const uint32_t stop = (start + block_size) < n ? (start + block_size) : n;
            float best = -__builtin_huge_valf();
            for (uint32_t position = start; position < stop; ++position) {
                best = s_score[position] > best ? s_score[position] : best;
            }
            scratch.f[block] = best;
        }
        __syncthreads();
        if (reach > 0u) {
            const uint32_t last = (reach - 1u) / block_size;
            if (last < num_blocks && tid == 0u) {
                scratch.f[last] = __builtin_huge_valf();
            }
        }
        __syncthreads();
        const uint32_t keep = topk_blocks < num_blocks ? topk_blocks : num_blocks;
        for (uint32_t word = tid; word < MAX_MASK_WORDS; word += BLOCK) {
            uint32_t bits = 0u;
            for (uint32_t bit = 0; bit < 32u; ++bit) {
                const uint32_t position = word * 32u + bit;
                if (position < n) {
                    const uint32_t block = position / block_size;
                    uint32_t rank = 0u;
                    for (uint32_t other = 0; other < num_blocks; ++other) {
                        if (other != block &&
                            before(scratch.f[other], other, scratch.f[block], block)) {
                            rank += 1u;
                        }
                    }
                    if (rank < keep && scratch.f[block] > -__builtin_huge_valf()) {
                        bits |= 1u << bit;
                    }
                }
            }
            s_mask[word] = bits;
        }
        __syncthreads();
        if (mask_out != nullptr) {
            for (uint32_t word = tid; word < MAX_MASK_WORDS; word += BLOCK) {
                mask_out[word] = s_mask[word];
            }
        }
        __syncthreads();
    }
    // The masks on this query's own scores: reach first, then a consumed HSI mask.
    for (uint32_t position = tid; position < n; position += BLOCK) {
        if (position >= reach) {
            s_score[position] = -__builtin_huge_valf();
        }
        if (mask_mode == 1u && (s_mask[position >> 5] & (1u << (position & 31u))) == 0u) {
            s_score[position] = -__builtin_huge_valf();
        }
    }
    __syncthreads();

    if (out_scores != nullptr) {
        for (uint32_t position = tid; position < n; position += BLOCK) {
            out_scores[position] = s_score[position];
        }
    }

    // Rank by the host's total order; the chosen positions are those with rank < topk. Then the
    // chosen ones are ordered ascending by position and mapped into the shared index space.
    const uint32_t keep = topk < n ? topk : n;
    for (uint32_t position = tid; position < n; position += BLOCK) {
        uint32_t rank = 0u;
        for (uint32_t other = 0; other < n; ++other) {
            if (other != position) {
                const bool better = before(s_score[other], other, s_score[position], position);
                rank += better ? 1u : 0u;
            }
        }
        scratch.u[position] = rank;
    }
    __syncthreads();
    for (uint32_t position = tid; position < n; position += BLOCK) {
        if (scratch.u[position] < keep) {
            uint32_t before_it = 0u;
            for (uint32_t other = 0; other < n; ++other) {
                if (other != position && scratch.u[other] < keep && other < position) {
                    before_it += 1u;
                }
            }
            out_ids[before_it] = position < reach ? position + offset : no_index;
        }
    }
}

}  // namespace spark_select

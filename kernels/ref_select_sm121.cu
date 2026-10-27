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

// kernels/ref_select_sm121.cu
//
// The reference candidate select: index scores, the reach and HSI block masks, the top-k by
// (score descending, compressing position ascending) and the shared-index-space ids the attention
// body's selected list reads. One CTA per query; a decode step is one query.
//
// THE CHAIN IS THE ENGINE'S, statement by statement:
//   score        crates/sparkflash-engine/src/forward/attention.rs:1476-1506 (index_scores):
//                per head, ascending head order, raw = bf16_round(dot(q_head, key)) with the dot
//                accumulated ascending in f32; if raw > 0 then sum += bf16_round(raw * weight);
//                after the heads sum = bf16_round(sum).
//   reach mask   attention.rs:1610-1616: positions >= (start_pos + query + 1) / ratio become -inf.
//   block mask   attention.rs:1619-1650 with select_candidate_blocks (attention.rs:1704-1734):
//                per block the best position's score; the block holding the newest reachable
//                position is pinned to +inf; the best topk_blocks blocks by (score desc, block asc)
//                are kept; a kept block whose score is -inf is dropped; kept positions set bits.
//   top-k        attention.rs:1660-1686: order every position by score descending and, on a tie,
//                position ascending; take the first min(topk, n_pos); re-sort those ascending; map
//                position < reach to position + offset (window_rows) and anything else to NO_INDEX.
//
// EXACTNESS BEFORE SPEED. The selection here is computed as a rank, not a sort: a position's rank
// is the number of positions that beat it in the same total order, so the chosen set and its order
// are the host's by construction and no sort comparator can disagree with it. That is O(n_pos^2)
// comparisons per CTA and it is the deliberate first form; a bitonic or radix retile is a later
// change that must keep this file's rank definition, not redefine it.
//
// NO TREE REDUCTION, NO REASSOCIATION. The dot, the head sum and the f32->bf16 rounds are the
// engine's exact operations (__fmul_rn, __fadd_rn, RNE narrowing); a score is a single chain of
// them, one per (head, position), never a reduced batch.
//
// LIMITS. MAX_POS = 4096 compressed positions (one CTA's score table); a caller with more refuses
// before the launch, the kernel has no growing path. head_dim and heads are runtime words.
//
// NO TRACE RING: this kernel writes no trace slot, so its launch carries no ring pointer.

#include "m4_common_sm121.cuh"

namespace {
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
}  // namespace

extern "C" __global__ void ref_select_sm121(
    const float* __restrict__ q_index,
    const float* __restrict__ keys,
    const float* __restrict__ weights,
    uint32_t* __restrict__ out_ids,
    float* __restrict__ out_scores,
    const uint32_t* __restrict__ mask_in,
    uint32_t* __restrict__ mask_out,
    uint32_t n_pos,
    uint32_t heads,
    uint32_t head_dim,
    uint32_t reach,
    uint32_t topk,
    uint32_t offset,
    uint32_t mask_mode,
    uint32_t topk_blocks,
    uint32_t block_size,
    uint32_t no_index,
    uint32_t flags) {
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

    // Scores: one position per thread pass, heads ascending and the dot ascending inside a head.
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
                        if (other != block && before(scratch.f[other], other, scratch.f[block], block)) {
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
    // The masks on this query's own scores: reach first (attention.rs:1610-1616), then a consumed
    // HSI mask (attention.rs:1640-1647).
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
                const bool better =
                    before(s_score[other], other, s_score[position], position);
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

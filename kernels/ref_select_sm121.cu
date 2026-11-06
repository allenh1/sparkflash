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
#include "select_body_sm121.cuh"

// The exact entry: the serial score pass, then the shared body's masks, rank and emission.
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
    spark_select::body<false>(nullptr, q_index, keys, weights, out_ids, out_scores, mask_in,
                              mask_out, n_pos, heads, head_dim, reach, topk, offset, mask_mode,
                              topk_blocks, block_size, no_index, flags);
}

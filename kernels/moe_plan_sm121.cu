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

// kernels/moe_plan_sm121.cu
//
// The routed-expert plan every arm shares: count the routed slots by expert, order the slots by
// expert, and cut each expert's run into items of a fixed row count. Qwen's token path
// (q4e_moe_permute), GLM's bulk prefill (glm_moe_plan), Qwen's bulk prefill (q4p_moe_plan) and
// DeepSeek's bulk prefill (opt_mma_moe_plan) each carried their own copy of this algorithm; this
// is that algorithm once, with what differed between them as arguments.
//
// What the three wrote, and what this writes:
//
//   sorted[rank]        the slot (or rank, for a caller whose ids are already rank-ordered) that
//                        sorts into `rank`. Ordered by expert id, ties in ascending slot order.
//   items[4n + 0..3]     (expert, first_rank, count, n) for n in 0..*item_count. `first_rank`
//                        indexes `sorted`; a consumer reads item n's slot list as
//                        sorted[items[4n+1] + i] for i in 0..count. Cut every `item_rows`
//                        slots, or one item per expert when item_rows is 0.
//   run_bounds[2 at]     (first, last) of the contiguous run of `sorted` positions `at` falls
//                        in, indexed BY POSITION in the sorted order (not by slot): the expert
//                        kernels launch one block per position and read this to learn whether
//                        they lead their run and where it ends. Null for a caller that cuts
//                        items instead.
//   *item_count          how many items were written, clipped to max_items.
//
// A caller may pass null for any output it does not read, and 0 for item_rows/max_items.
//
// Determinism: the order is stable (expert, then slot), which the counting pass alone does not
// give. Each expert's positions are filled by one thread walking the ranks in ascending order,
// so the layout a run produces is the same on every launch and on every rank. Two of the four
// predecessors used an atomic cursor and were not stable; the routing is bit-neutral either way
// (every slot's row is written independently), but a stable plan is a plan two traces can be
// diffed against, and it costs one pass over the rank list per expert.
//
// Segments: a bulk prefill may run several sequences in one launch and needs only the ranks of
// live tokens counted and ordered. `seg_tokens` > 0 names the token stride of up to four
// contiguous segments and seg0..seg3 their widths; a rank whose token falls outside every
// segment is left out of the counts, the items and `sorted`. seg_tokens = 0 makes every rank
// live, which is what a single chunk and every token path pass.
//
// An id outside [0, experts) is one of three things, and `invalid` names which:
//
//   MOE_PLAN_INVALID_FIRST (0)  the rank sorts into a bucket of its own ahead of expert 0, so it
//                               is one run whose item names a negative expert - Qwen's token
//                               path, whose expert kernels branch on a negative id.
//   MOE_PLAN_CLAMP (1)          the rank counts as expert 0, so a corrupt plan routes somewhere
//                               rather than past the end of the counters - DeepSeek's bulk plan.
//   MOE_PLAN_DROP (2)           the rank is left out of the counts, the items and `sorted`. It is
//                               only safe for a consumer that never reads a slot it did not get
//                               an item for; a consumer that sums every slot's row must also pass
//                               `zero_rows` (Qwen's bulk prefill does both).
//
// The reserved bucket is index 0, and it is reserved by *not* being an expert: with the shift in
// place every expert e sorts to e + 1. An invalid rank that returned the shift itself would land
// in expert 0's bucket and be indistinguishable from it, which is a merged run no consumer can
// read correctly - the token path zeroes a run whose first id is negative, so the merge would
// either drop the valid expert-0 slots with it or compute the dropped ones as expert 0.
//
// Bounds: `rows` is the caller's slot count and may exceed any block's shared-memory budget -
// the counters are per expert, not per slot. An out-of-range `experts` (above
// MOE_PLAN_MAX_EXPERTS) or a non-positive `rows` writes the identity order, one run per slot,
// and zero items, so a caller that reads the plan anyway launches one block per slot instead of
// reading stale memory.

#include <cstdint>

#include <cuda_runtime.h>

#include "opt_pdl_sm121.cuh"

// The most experts the counting pass holds. 512 covers Qwen (512), DeepSeek (384) and GLM (288).
#define MOE_PLAN_MAX_EXPERTS 512
// The segments one launch may filter by.
#define MOE_PLAN_SEGS 4
// How an id outside [0, experts) is treated.
#define MOE_PLAN_INVALID_FIRST 0
#define MOE_PLAN_CLAMP 1
#define MOE_PLAN_DROP 2

extern "C" __global__ void moe_plan(
    const int *__restrict__ ids, int *__restrict__ sorted, int *__restrict__ items,
    int *__restrict__ item_count, int *__restrict__ run_bounds, int rows, int experts,
    int topk, int item_rows, int max_items, int seg_tokens, int invalid, int seg0, int seg1,
    int seg2, int seg3, float *__restrict__ zero_rows, int zero_stride) {
    OPT_PDL_ENTRY();
    __shared__ int s_count[MOE_PLAN_MAX_EXPERTS + 1];
    __shared__ int s_base[MOE_PLAN_MAX_EXPERTS + 1];
    __shared__ int s_used;

    const int tid = (int)threadIdx.x;
    const int span = (int)blockDim.x;
    const int stride = topk > 0 ? topk : 1;
    const int yield = seg_tokens > 0 ? seg_tokens : 0;
    const int width[MOE_PLAN_SEGS] = {seg0, seg1, seg2, seg3};
    // MOE_PLAN_INVALID_FIRST reserves bucket 0 for the ids no expert owns, so those ranks sort
    // first as one run, the group the token path's expert kernel zeroes.
    const int shift = invalid == MOE_PLAN_INVALID_FIRST ? 1 : 0;
    const int buckets = experts + shift;
    // Both predicates are kernel-wide (one `invalid` per launch), so they are evaluated once here
    // rather than per rank: read inside `bucket_of`, the extra branch made the compiler emit both
    // halves of every caller's loop and the cubin grew from 50,824 to 75,632 bytes.
    const bool drop = invalid == MOE_PLAN_DROP;
    const bool clamp = invalid == MOE_PLAN_CLAMP;

    // A token outside the named segments contributes no slots.
    auto live_token = [&](int token) {
        if (yield == 0) return true;
        const int seg = token / yield;
        return seg < MOE_PLAN_SEGS && (token - seg * yield) < width[seg];
    };
    auto live = [&](int rank) { return live_token(rank / stride); };
    // The bucket a rank sorts into, or -1 when the caller drops it. The reserved bucket is 0, so
    // an id no expert owns is never merged into expert 0's run: INVALID_FIRST keeps it in the
    // reserved bucket, CLAMP gives it expert 0's bucket, DROP leaves it out.
    auto bucket_of = [&](int rank) {
        const int expert = ids[rank];
        if (expert >= 0 && expert < experts) return expert + shift;
        if (drop) return -1;
        return clamp ? shift : 0;
    };

    if (rows <= 0 || experts <= 0 || experts > MOE_PLAN_MAX_EXPERTS) {
        for (int rank = tid; rank < rows; rank += span) {
            sorted[rank] = rank;
            if (run_bounds != nullptr) {
                run_bounds[2 * rank] = rank;
                run_bounds[2 * rank + 1] = rank;
            }
        }
        if (items != nullptr) {
            for (int item = tid; item < max_items; item += span) {
                items[4 * item] = 0;
                items[4 * item + 1] = 0;
                items[4 * item + 2] = 0;
                items[4 * item + 3] = item;
            }
        }
        if (tid == 0 && item_count != nullptr) *item_count = 0;
        return;
    }

    for (int bucket = tid; bucket < buckets; bucket += span) s_count[bucket] = 0;
    // A run bound is written at sorted positions, so clear the table first: a position no run
    // covers (a rank the caller drops) then reads as an empty run rather than stale memory.
    if (run_bounds != nullptr) {
        for (int at = tid; at < rows; at += span) {
            run_bounds[2 * at] = 0;
            run_bounds[2 * at + 1] = -1;
        }
    }
    __syncthreads();
    for (int rank = tid; rank < rows; rank += span) {
        if (!live(rank)) continue;
        const int bucket = bucket_of(rank);
        if (bucket >= 0) atomicAdd(&s_count[bucket], 1);
    }
    // A caller whose combine reads every slot's row names that row here, so a slot no expert
    // owns contributes zeros rather than whatever the buffer held.
    if (zero_rows != nullptr) {
        for (int rank = tid; rank < rows; rank += span) {
            if (!live(rank)) continue;
            const int expert = ids[rank];
            if (expert >= 0 && expert < experts) continue;
            float *row = zero_rows + (long long)rank * zero_stride;
            for (int column = 0; column < zero_stride; ++column) row[column] = 0.0f;
        }
    }
    __syncthreads();

    // One thread lays the buckets out in ascending order and cuts the items: the offsets every
    // later pass indexes, and a plan whose item list does not depend on which thread got here
    // first. Bucket 0 is an expert unless the caller reserved it for the invalid ids.
    if (tid == 0) {
        int offset = 0;
        int item = 0;
        for (int bucket = 0; bucket < buckets; ++bucket) {
            const int base = offset;
            s_base[bucket] = base;
            const int count = s_count[bucket];
            offset += count;
            if (items == nullptr || count <= 0) continue;
            // One item per run when the caller names no row count (`per` is then the count).
            const int per = item_rows > 0 ? item_rows : count;
            for (int first = 0; first < count; first += per) {
                const int left = count - first;
                if (item < max_items) {
                    items[4 * item] = bucket - shift;
                    items[4 * item + 1] = base + first;
                    items[4 * item + 2] = left < per ? left : per;
                    items[4 * item + 3] = item;
                }
                ++item;
            }
        }
        s_used = item < max_items ? item : max_items;
        if (item_count != nullptr) *item_count = s_used;
        // Every position of a bucket's run names the same (first, last): the block at that
        // position leads the run when its position is the first.
        if (run_bounds != nullptr) {
            for (int bucket = 0; bucket < buckets; ++bucket) {
                const int first = s_base[bucket];
                const int last = first + s_count[bucket] - 1;
                for (int at = first; at <= last; ++at) {
                    run_bounds[2 * at] = first;
                    run_bounds[2 * at + 1] = last;
                }
            }
        }
    }
    __syncthreads();

    // Items past the last are written with a zero count, so a launch over max_items runs nothing
    // for them instead of reading whatever the buffer held.
    if (items != nullptr) {
        for (int item = s_used + tid; item < max_items; item += span) {
            items[4 * item] = 0;
            items[4 * item + 1] = 0;
            items[4 * item + 2] = 0;
            items[4 * item + 3] = item;
        }
    }

    // The stable half: one thread per bucket walks the ranks in ascending order and fills that
    // bucket's positions. The token is tracked incrementally so the inner loop divides nothing.
    for (int bucket = tid; bucket < buckets; bucket += span) {
        const int count = s_count[bucket];
        if (count <= 0) continue;
        int at = s_base[bucket];
        int token = 0;
        int lane = 0;
        bool is_live = live_token(0);
        for (int rank = 0; rank < rows; ++rank) {
            if (lane == 0) is_live = live_token(token);
            if (is_live && bucket_of(rank) == bucket) sorted[at++] = rank;
            if (++lane == stride) {
                lane = 0;
                ++token;
            }
        }
    }
}

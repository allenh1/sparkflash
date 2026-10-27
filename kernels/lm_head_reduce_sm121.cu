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

// kernels/lm_head_reduce_sm121.cu
//
// The certified LM head's device-side reduction: the last host assembly step between the four
// cert_lm_head_sm121 shard launches and the token. The screen kernel publishes, per rank, 32,320
// stored logits (a certificate bound where certified == 1), 32,320 certified flags, 32 block
// maxima and 32 block argmax indices; the chain then reconstructs the 129,280-way argmax on the
// host. This kernel does that reconstruction on the device so the token can come from a launch:
//
//   * the 128 blocks are the m16 layout, rank-major: global block r * 32 + b covers rank r's
//     entries [b * 1024, min((b + 1) * 1024, 32,320)), so the last block of each rank has 576
//     entries, exactly as the screen's grid does;
//   * each block's maximum and argmax are computed with the screen's rule: strict greater-than,
//     and on a tie the LOWEST global entry index wins. The screen's own per-thread and per-block
//     reductions use the same rule (kernels/cert_lm_head_sm121.cu), so the global answer is the
//     same sequence of comparisons the host assembly performs;
//   * the 129,280-way argmax is the lowest-index tie-break over the 128 block answers in ascending
//     global block order; the value written is the stored value at that entry;
//   * the top-5 is sorted by value descending and then by ascending global index, matching
//     src/lm_head_ref.rs / src/m16's top_indices. It is CLAIMED only when no entry is certified
//     (certified_count == 0), because a certified entry stores an upper bound rather than its
//     exact value: the same rule m16's host-side claims_top5 applies before it will compare the
//     set with a reference. When any entry is certified the top-5 values are still written for
//     diagnostics but out_status does NOT carry SPARK_LM_REDUCE_STATUS_TOP5_CLAIMED;
//   * a certified entry is never silently valued as if it were exact: if the argmax entry is
//     certified, out_status carries SPARK_LM_REDUCE_STATUS_WINNER_CERTIFIED and the caller must
//     treat the token as not claimable (refuse, do not assert). out_token is still written for
//     diagnostics in that case. On a sound certificate this cannot happen: a certified entry
//     satisfies bound < running maximum, so it cannot be the strict argmax;
//   * no arithmetic is performed on a logit. The kernel only compares and moves f32 bit patterns,
//     so it cannot introduce a value delta: the rounding class is inherited from
//     cert_lm_head_sm121, the only arithmetic upstream. Determinism follows from fixed scan order
//     and one launch with one block.
//
// The four shard ranges are UNIFORM: rank r owns exactly the 32,320 consecutive vocabulary entries
// [r * 32,320, (r + 1) * 32,320). The non-uniform numbers sometimes quoted for the concentrated
// certificate case (1,062 / 1,205 / 1,215 / 1,417) are that case's CERTIFIED-FLAG COUNTS per rank,
// not shard sizes or ranges; do not re-import them as shard geometry.
//
// One launch, one block of 256 threads (8 warps x 16 chunks), so the whole reduction is one
// kernel with no grid-wide rendezvous: the block answers go through shared memory and thread 0
// folds them in ascending block order. That is deliberately not a performance claim: the stage is
// a correctness seam, and the 4 shard buffers are already device-resident, so the launch reads
// ~2 MiB and adds no host round-trip.
//
// No trace event: specification 5.8's 16 trace scope bits are all assigned and none describes the
// head reduction, so the trace arguments are carried for ABI uniformity and deliberately unused
// (the same convention as kernels/rmsnorm_sm121.cu).

#include "m4_common_sm121.cuh"

#define SPARK_LM_REDUCE_RANKS 4u
#define SPARK_LM_REDUCE_VOCAB_PER_RANK 32320u
#define SPARK_LM_REDUCE_VOCAB 129280u
#define SPARK_LM_REDUCE_CHUNK 1024u
#define SPARK_LM_REDUCE_CHUNKS_PER_RANK 32u
#define SPARK_LM_REDUCE_BLOCKS (SPARK_LM_REDUCE_RANKS * SPARK_LM_REDUCE_CHUNKS_PER_RANK)
#define SPARK_LM_REDUCE_THREADS 256u
#define SPARK_LM_REDUCE_WARPS 8u
#define SPARK_LM_REDUCE_LANES 32u
#define SPARK_LM_REDUCE_TOP 5u
#define SPARK_LM_REDUCE_STATUS_WINNER_CERTIFIED 1u
#define SPARK_LM_REDUCE_STATUS_TOP5_CLAIMED 2u
#define SPARK_LM_REDUCE_STATUS_ANY_CERTIFIED 4u

static_assert(SPARK_LM_REDUCE_VOCAB == 129280u, "the released vocabulary is 129,280 entries");
static_assert(SPARK_LM_REDUCE_VOCAB == SPARK_LM_REDUCE_RANKS * SPARK_LM_REDUCE_VOCAB_PER_RANK,
              "the vocabulary shards evenly over four ranks");
static_assert(SPARK_LM_REDUCE_VOCAB_PER_RANK == SPARK_M4_VOCAB_PER_RANK,
              "the per-rank vocabulary is the screen's per-rank vocabulary");
static_assert(SPARK_LM_REDUCE_CHUNKS_PER_RANK * SPARK_LM_REDUCE_CHUNK >= SPARK_LM_REDUCE_VOCAB_PER_RANK,
              "the per-rank chunks cover the rank");
static_assert((SPARK_LM_REDUCE_CHUNKS_PER_RANK - 1u) * SPARK_LM_REDUCE_CHUNK < SPARK_LM_REDUCE_VOCAB_PER_RANK,
              "the last block is the only partial block");
static_assert(SPARK_LM_REDUCE_BLOCKS == 128u, "the m16 layout has 128 blocks");
static_assert(SPARK_LM_REDUCE_THREADS == SPARK_LM_REDUCE_WARPS * SPARK_LM_REDUCE_LANES,
              "eight full warps");
static_assert(SPARK_LM_REDUCE_BLOCKS % SPARK_LM_REDUCE_WARPS == 0u,
              "each warp visits whole blocks");
static_assert(SPARK_LM_REDUCE_TOP == 5u, "the top-5 has five entries");
static_assert(SPARK_LM_REDUCE_STATUS_WINNER_CERTIFIED == 1u &&
              SPARK_LM_REDUCE_STATUS_TOP5_CLAIMED == 2u &&
              SPARK_LM_REDUCE_STATUS_ANY_CERTIFIED == 4u,
              "the status bits are the pinned contract");

// -FLT_MAX, the running maximum the screen's per-thread scan starts from.
#define SPARK_LM_REDUCE_MIN (-3.402823466e38f)

// The rank's logits pointer. A four-way select, not an array of pointers, so the compiler keeps it
// in registers; the rank is uniform per block chunk.
__device__ __forceinline__ const float *spark_lm_reduce_logits(const uint32_t rank,
                                                               const float *r0, const float *r1,
                                                               const float *r2, const float *r3) {
    switch (rank) {
    case 0u:
        return r0;
    case 1u:
        return r1;
    case 2u:
        return r2;
    default:
        return r3;
    }
}

// The rank's certified-flag pointer, the same four-way select.
__device__ __forceinline__ const uint8_t *spark_lm_reduce_certified(const uint32_t rank,
                                                                   const uint8_t *r0, const uint8_t *r1,
                                                                   const uint8_t *r2, const uint8_t *r3) {
    switch (rank) {
    case 0u:
        return r0;
    case 1u:
        return r1;
    case 2u:
        return r2;
    default:
        return r3;
    }
}

// One pair under the pinned order: greater value wins, equal value goes to the lower index.
__device__ __forceinline__ bool spark_lm_reduce_better(const float value, const uint32_t index,
                                                       const float held, const uint32_t at) {
    return value > held || (value == held && index < at);
}

// Inserts (value, index) into a descending top-5, value then ascending index. The slots start at
// -FLT_MAX with index 0xFFFFFFFF so every real entry enters once, and an index already present is
// not inserted again: a tie is one entry, not two copies of it.
__device__ __forceinline__ void spark_lm_reduce_top5_insert(const float value, const uint32_t index,
                                                            float *top_value, uint32_t *top_index) {
    if (!spark_lm_reduce_better(value, index, top_value[SPARK_LM_REDUCE_TOP - 1u],
                                top_index[SPARK_LM_REDUCE_TOP - 1u])) {
        return;
    }
#pragma unroll
    for (uint32_t slot = 0u; slot < SPARK_LM_REDUCE_TOP; ++slot) {
        if (top_index[slot] == index) {
            return;
        }
    }
    if (spark_lm_reduce_better(value, index, top_value[0], top_index[0])) {
        top_value[4] = top_value[3];
        top_index[4] = top_index[3];
        top_value[3] = top_value[2];
        top_index[3] = top_index[2];
        top_value[2] = top_value[1];
        top_index[2] = top_index[1];
        top_value[1] = top_value[0];
        top_index[1] = top_index[0];
        top_value[0] = value;
        top_index[0] = index;
    } else if (spark_lm_reduce_better(value, index, top_value[1], top_index[1])) {
        top_value[4] = top_value[3];
        top_index[4] = top_index[3];
        top_value[3] = top_value[2];
        top_index[3] = top_index[2];
        top_value[2] = top_value[1];
        top_index[2] = top_index[1];
        top_value[1] = value;
        top_index[1] = index;
    } else if (spark_lm_reduce_better(value, index, top_value[2], top_index[2])) {
        top_value[4] = top_value[3];
        top_index[4] = top_index[3];
        top_value[3] = top_value[2];
        top_index[3] = top_index[2];
        top_value[2] = value;
        top_index[2] = index;
    } else if (spark_lm_reduce_better(value, index, top_value[3], top_index[3])) {
        top_value[4] = top_value[3];
        top_index[4] = top_index[3];
        top_value[3] = value;
        top_index[3] = index;
    } else {
        top_value[4] = value;
        top_index[4] = index;
    }
}

// Warp-wide best under the same order, using the fixed shuffle tree so the result does not depend
// on scheduler order.
__device__ __forceinline__ void spark_lm_reduce_warp_best(float *value, uint32_t *index) {
    for (uint32_t offset = SPARK_LM_REDUCE_LANES / 2u; offset > 0u; offset >>= 1u) {
        const float other_value = __shfl_down_sync(0xffffffffu, *value, offset);
        const uint32_t other_index = __shfl_down_sync(0xffffffffu, *index, offset);
        if (spark_lm_reduce_better(other_value, other_index, *value, *index)) {
            *value = other_value;
            *index = other_index;
        }
    }
}

extern "C" __global__ void lm_head_reduce_sm121(
    const float *__restrict__ logits_r0,        // [32,320] rank 0 stored logits (bound where certified)
    const float *__restrict__ logits_r1,        // [32,320] rank 1
    const float *__restrict__ logits_r2,        // [32,320] rank 2
    const float *__restrict__ logits_r3,        // [32,320] rank 3
    const uint8_t *__restrict__ certified_r0,   // [32,320] rank 0 certified flags
    const uint8_t *__restrict__ certified_r1,   // [32,320] rank 1
    const uint8_t *__restrict__ certified_r2,   // [32,320] rank 2
    const uint8_t *__restrict__ certified_r3,   // [32,320] rank 3
    unsigned int *__restrict__ token,           // [1] global argmax entry (diagnostic when refused)
    float *__restrict__ value,                  // [1] stored value at that entry
    float *__restrict__ block_max,              // [128] the m16 block maxima
    uint32_t *__restrict__ block_argmax,        // [128] the global entry that scored each block
    float *__restrict__ top5_value,             // [5] descending, ties lower index
    uint32_t *__restrict__ top5_index,          // [5] the global entries
    unsigned int *__restrict__ status,          // [1] the pinned status bits
    unsigned int *__restrict__ certified_count, // [1] total certified flags
    GpuTraceRing *__restrict__ trace_ring,      // carried for ABI uniformity, deliberately unused
    const uint32_t layer_id,                    // carried for ABI uniformity, deliberately unused
    const uint64_t step_seq) {                  // carried for ABI uniformity, deliberately unused
    __shared__ float s_block_max[SPARK_LM_REDUCE_BLOCKS];
    __shared__ uint32_t s_block_argmax[SPARK_LM_REDUCE_BLOCKS];
    __shared__ unsigned int s_certified[SPARK_LM_REDUCE_THREADS];
    __shared__ float s_top_value[SPARK_LM_REDUCE_THREADS * SPARK_LM_REDUCE_TOP];
    __shared__ uint32_t s_top_index[SPARK_LM_REDUCE_THREADS * SPARK_LM_REDUCE_TOP];

    const uint32_t lane = threadIdx.x % SPARK_LM_REDUCE_LANES;
    const uint32_t warp = threadIdx.x / SPARK_LM_REDUCE_LANES;

    float lane_top_value[SPARK_LM_REDUCE_TOP];
    uint32_t lane_top_index[SPARK_LM_REDUCE_TOP];
    for (uint32_t slot = 0u; slot < SPARK_LM_REDUCE_TOP; ++slot) {
        lane_top_value[slot] = SPARK_LM_REDUCE_MIN;
        lane_top_index[slot] = 0xffffffffu;
    }
    unsigned int lane_certified = 0u;

    // One warp visits whole blocks; 128 blocks over 8 warps is 16 blocks per warp.
    for (uint32_t chunk = warp; chunk < SPARK_LM_REDUCE_BLOCKS; chunk += SPARK_LM_REDUCE_WARPS) {
        const uint32_t rank = chunk / SPARK_LM_REDUCE_CHUNKS_PER_RANK;
        const uint32_t block = chunk % SPARK_LM_REDUCE_CHUNKS_PER_RANK;
        const uint32_t base = block * SPARK_LM_REDUCE_CHUNK;
        const uint32_t remaining = SPARK_LM_REDUCE_VOCAB_PER_RANK - base;
        const uint32_t count = remaining < SPARK_LM_REDUCE_CHUNK ? remaining : SPARK_LM_REDUCE_CHUNK;
        const float *logits = spark_lm_reduce_logits(rank, logits_r0, logits_r1, logits_r2, logits_r3);
        const uint8_t *certified =
            spark_lm_reduce_certified(rank, certified_r0, certified_r1, certified_r2, certified_r3);
        float best_value = SPARK_LM_REDUCE_MIN;
        uint32_t best_index = 0xffffffffu;
        for (uint32_t step = 0u; step < SPARK_LM_REDUCE_CHUNK / SPARK_LM_REDUCE_LANES; ++step) {
            const uint32_t local = lane + step * SPARK_LM_REDUCE_LANES;
            if (local >= count) {
                break;
            }
            const float entry_value = logits[base + local];
            const uint32_t global = rank * SPARK_LM_REDUCE_VOCAB_PER_RANK + base + local;
            if (spark_lm_reduce_better(entry_value, global, best_value, best_index)) {
                best_value = entry_value;
                best_index = global;
            }
            if (certified[base + local] != 0u) {
                lane_certified += 1u;
            }
            spark_lm_reduce_top5_insert(entry_value, global, lane_top_value, lane_top_index);
        }
        spark_lm_reduce_warp_best(&best_value, &best_index);
        if (lane == 0u) {
            s_block_max[chunk] = best_value;
            s_block_argmax[chunk] = best_index;
        }
    }

    for (uint32_t slot = 0u; slot < SPARK_LM_REDUCE_TOP; ++slot) {
        s_top_value[threadIdx.x * SPARK_LM_REDUCE_TOP + slot] = lane_top_value[slot];
        s_top_index[threadIdx.x * SPARK_LM_REDUCE_TOP + slot] = lane_top_index[slot];
    }
    s_certified[threadIdx.x] = lane_certified;
    __syncthreads();

    // The finalisation is one thread: 128 block answers, the 128 certified sums and the 1,280
    // top-5 candidates are all small, and doing it in one fixed order is what makes the answer
    // deterministic without a second launch.
    if (threadIdx.x == 0u) {
        float best_value = SPARK_LM_REDUCE_MIN;
        uint32_t best_index = 0xffffffffu;
        for (uint32_t block = 0u; block < SPARK_LM_REDUCE_BLOCKS; ++block) {
            if (spark_lm_reduce_better(s_block_max[block], s_block_argmax[block], best_value, best_index)) {
                best_value = s_block_max[block];
                best_index = s_block_argmax[block];
            }
        }
        unsigned int total_certified = 0u;
        for (uint32_t thread = 0u; thread < SPARK_LM_REDUCE_THREADS; ++thread) {
            total_certified += s_certified[thread];
        }
        float top_value[SPARK_LM_REDUCE_TOP];
        uint32_t top_index[SPARK_LM_REDUCE_TOP];
        for (uint32_t slot = 0u; slot < SPARK_LM_REDUCE_TOP; ++slot) {
            top_value[slot] = SPARK_LM_REDUCE_MIN;
            top_index[slot] = 0xffffffffu;
        }
        for (uint32_t thread = 0u; thread < SPARK_LM_REDUCE_THREADS; ++thread) {
            for (uint32_t slot = 0u; slot < SPARK_LM_REDUCE_TOP; ++slot) {
                spark_lm_reduce_top5_insert(s_top_value[thread * SPARK_LM_REDUCE_TOP + slot],
                                            s_top_index[thread * SPARK_LM_REDUCE_TOP + slot], top_value,
                                            top_index);
            }
        }
        const uint32_t winner_rank = best_index / SPARK_LM_REDUCE_VOCAB_PER_RANK;
        const uint32_t winner_local = best_index % SPARK_LM_REDUCE_VOCAB_PER_RANK;
        const uint8_t *winner_certified_flags = spark_lm_reduce_certified(
            winner_rank, certified_r0, certified_r1, certified_r2, certified_r3);
        const bool winner_certified = winner_certified_flags[winner_local] != 0u;
        unsigned int status_bits = 0u;
        if (total_certified != 0u) {
            status_bits |= SPARK_LM_REDUCE_STATUS_ANY_CERTIFIED;
        }
        if (winner_certified) {
            status_bits |= SPARK_LM_REDUCE_STATUS_WINNER_CERTIFIED;
        } else if (total_certified == 0u) {
            status_bits |= SPARK_LM_REDUCE_STATUS_TOP5_CLAIMED;
        }
        *token = best_index;
        *value = best_value;
        for (uint32_t block = 0u; block < SPARK_LM_REDUCE_BLOCKS; ++block) {
            block_max[block] = s_block_max[block];
            block_argmax[block] = s_block_argmax[block];
        }
        for (uint32_t slot = 0u; slot < SPARK_LM_REDUCE_TOP; ++slot) {
            top5_value[slot] = top_value[slot];
            top5_index[slot] = top_index[slot];
        }
        *status = status_bits;
        *certified_count = total_certified;
    }
    (void)trace_ring;
    (void)layer_id;
    (void)step_seq;
}
// K8 batched reduction: one launch reduces SIX rows. The body is the scalar kernel's block
// algorithm run once per row, in the same thread/chunk/shuffle/top-5 order and with the same
// single-thread finalisation, so each row is BYTE-IDENTICAL to the scalar kernel's row at any
// vocabulary length. Batch-major report arrays: token/value/status/certified_count [6], block
// arrays [6][128], top-5 [6][5]. The block is reused across rows with a barrier between rows.
#define SPARK_LM_REDUCE_BATCH_ROWS 6u
static_assert(SPARK_LM_REDUCE_BATCH_ROWS == 6u, "the verifier batch is six rows");

extern "C" __global__ void lm_head_reduce_batched_sm121(

    const float *__restrict__ logits_r0_all,        // [32,320] rank 0 stored logits (bound where certified)
    const float *__restrict__ logits_r1_all,        // [32,320] rank 1
    const float *__restrict__ logits_r2_all,        // [32,320] rank 2
    const float *__restrict__ logits_r3_all,        // [32,320] rank 3
    const uint8_t *__restrict__ certified_r0_all,   // [32,320] rank 0 certified flags
    const uint8_t *__restrict__ certified_r1_all,   // [32,320] rank 1
    const uint8_t *__restrict__ certified_r2_all,   // [32,320] rank 2
    const uint8_t *__restrict__ certified_r3_all,   // [32,320] rank 3
    unsigned int *__restrict__ token_all,           // [1] global argmax entry (diagnostic when refused)
    float *__restrict__ value_all,                  // [1] stored value at that entry
    float *__restrict__ block_max_all,              // [128] the m16 block maxima
    uint32_t *__restrict__ block_argmax_all,        // [128] the global entry that scored each block
    float *__restrict__ top5_value_all,             // [5] descending, ties lower index
    uint32_t *__restrict__ top5_index_all,          // [5] the global entries
    unsigned int *__restrict__ status_all,          // [1] the pinned status bits
    unsigned int *__restrict__ certified_count_all, // [1] total certified flags
    GpuTraceRing *__restrict__ trace_ring,      // carried for ABI uniformity, deliberately unused
    const uint32_t layer_id,                    // carried for ABI uniformity, deliberately unused
    const uint64_t step_seq) {

    for (uint32_t row = 0u; row < 6u; ++row) {
        const float *logits_r0 = logits_r0_all + (uint64_t)row * SPARK_LM_REDUCE_VOCAB_PER_RANK;
        const float *logits_r1 = logits_r1_all + (uint64_t)row * SPARK_LM_REDUCE_VOCAB_PER_RANK;
        const float *logits_r2 = logits_r2_all + (uint64_t)row * SPARK_LM_REDUCE_VOCAB_PER_RANK;
        const float *logits_r3 = logits_r3_all + (uint64_t)row * SPARK_LM_REDUCE_VOCAB_PER_RANK;
        const uint8_t *certified_r0 = certified_r0_all + (uint64_t)row * SPARK_LM_REDUCE_VOCAB_PER_RANK;
        const uint8_t *certified_r1 = certified_r1_all + (uint64_t)row * SPARK_LM_REDUCE_VOCAB_PER_RANK;
        const uint8_t *certified_r2 = certified_r2_all + (uint64_t)row * SPARK_LM_REDUCE_VOCAB_PER_RANK;
        const uint8_t *certified_r3 = certified_r3_all + (uint64_t)row * SPARK_LM_REDUCE_VOCAB_PER_RANK;
        unsigned int *token = token_all + row;
        float *value = value_all + row;
        float *block_max = block_max_all + (uint64_t)row * SPARK_LM_REDUCE_BLOCKS;
        uint32_t *block_argmax = block_argmax_all + (uint64_t)row * SPARK_LM_REDUCE_BLOCKS;
        float *top5_value = top5_value_all + (uint64_t)row * SPARK_LM_REDUCE_TOP;
        uint32_t *top5_index = top5_index_all + (uint64_t)row * SPARK_LM_REDUCE_TOP;
        unsigned int *status = status_all + row;
        unsigned int *certified_count = certified_count_all + row;
                  // carried for ABI uniformity, deliberately unused
    __shared__ float s_block_max[SPARK_LM_REDUCE_BLOCKS];
    __shared__ uint32_t s_block_argmax[SPARK_LM_REDUCE_BLOCKS];
    __shared__ unsigned int s_certified[SPARK_LM_REDUCE_THREADS];
    __shared__ float s_top_value[SPARK_LM_REDUCE_THREADS * SPARK_LM_REDUCE_TOP];
    __shared__ uint32_t s_top_index[SPARK_LM_REDUCE_THREADS * SPARK_LM_REDUCE_TOP];

    const uint32_t lane = threadIdx.x % SPARK_LM_REDUCE_LANES;
    const uint32_t warp = threadIdx.x / SPARK_LM_REDUCE_LANES;

    float lane_top_value[SPARK_LM_REDUCE_TOP];
    uint32_t lane_top_index[SPARK_LM_REDUCE_TOP];
    for (uint32_t slot = 0u; slot < SPARK_LM_REDUCE_TOP; ++slot) {
        lane_top_value[slot] = SPARK_LM_REDUCE_MIN;
        lane_top_index[slot] = 0xffffffffu;
    }
    unsigned int lane_certified = 0u;

    // One warp visits whole blocks; 128 blocks over 8 warps is 16 blocks per warp.
    for (uint32_t chunk = warp; chunk < SPARK_LM_REDUCE_BLOCKS; chunk += SPARK_LM_REDUCE_WARPS) {
        const uint32_t rank = chunk / SPARK_LM_REDUCE_CHUNKS_PER_RANK;
        const uint32_t block = chunk % SPARK_LM_REDUCE_CHUNKS_PER_RANK;
        const uint32_t base = block * SPARK_LM_REDUCE_CHUNK;
        const uint32_t remaining = SPARK_LM_REDUCE_VOCAB_PER_RANK - base;
        const uint32_t count = remaining < SPARK_LM_REDUCE_CHUNK ? remaining : SPARK_LM_REDUCE_CHUNK;
        const float *logits = spark_lm_reduce_logits(rank, logits_r0, logits_r1, logits_r2, logits_r3);
        const uint8_t *certified =
            spark_lm_reduce_certified(rank, certified_r0, certified_r1, certified_r2, certified_r3);
        float best_value = SPARK_LM_REDUCE_MIN;
        uint32_t best_index = 0xffffffffu;
        for (uint32_t step = 0u; step < SPARK_LM_REDUCE_CHUNK / SPARK_LM_REDUCE_LANES; ++step) {
            const uint32_t local = lane + step * SPARK_LM_REDUCE_LANES;
            if (local >= count) {
                break;
            }
            const float entry_value = logits[base + local];
            const uint32_t global = rank * SPARK_LM_REDUCE_VOCAB_PER_RANK + base + local;
            if (spark_lm_reduce_better(entry_value, global, best_value, best_index)) {
                best_value = entry_value;
                best_index = global;
            }
            if (certified[base + local] != 0u) {
                lane_certified += 1u;
            }
            spark_lm_reduce_top5_insert(entry_value, global, lane_top_value, lane_top_index);
        }
        spark_lm_reduce_warp_best(&best_value, &best_index);
        if (lane == 0u) {
            s_block_max[chunk] = best_value;
            s_block_argmax[chunk] = best_index;
        }
    }

    for (uint32_t slot = 0u; slot < SPARK_LM_REDUCE_TOP; ++slot) {
        s_top_value[threadIdx.x * SPARK_LM_REDUCE_TOP + slot] = lane_top_value[slot];
        s_top_index[threadIdx.x * SPARK_LM_REDUCE_TOP + slot] = lane_top_index[slot];
    }
    s_certified[threadIdx.x] = lane_certified;
    __syncthreads();

    // The finalisation is one thread: 128 block answers, the 128 certified sums and the 1,280
    // top-5 candidates are all small, and doing it in one fixed order is what makes the answer
    // deterministic without a second launch.
    if (threadIdx.x == 0u) {
        float best_value = SPARK_LM_REDUCE_MIN;
        uint32_t best_index = 0xffffffffu;
        for (uint32_t block = 0u; block < SPARK_LM_REDUCE_BLOCKS; ++block) {
            if (spark_lm_reduce_better(s_block_max[block], s_block_argmax[block], best_value, best_index)) {
                best_value = s_block_max[block];
                best_index = s_block_argmax[block];
            }
        }
        unsigned int total_certified = 0u;
        for (uint32_t thread = 0u; thread < SPARK_LM_REDUCE_THREADS; ++thread) {
            total_certified += s_certified[thread];
        }
        float top_value[SPARK_LM_REDUCE_TOP];
        uint32_t top_index[SPARK_LM_REDUCE_TOP];
        for (uint32_t slot = 0u; slot < SPARK_LM_REDUCE_TOP; ++slot) {
            top_value[slot] = SPARK_LM_REDUCE_MIN;
            top_index[slot] = 0xffffffffu;
        }
        for (uint32_t thread = 0u; thread < SPARK_LM_REDUCE_THREADS; ++thread) {
            for (uint32_t slot = 0u; slot < SPARK_LM_REDUCE_TOP; ++slot) {
                spark_lm_reduce_top5_insert(s_top_value[thread * SPARK_LM_REDUCE_TOP + slot],
                                            s_top_index[thread * SPARK_LM_REDUCE_TOP + slot], top_value,
                                            top_index);
            }
        }
        const uint32_t winner_rank = best_index / SPARK_LM_REDUCE_VOCAB_PER_RANK;
        const uint32_t winner_local = best_index % SPARK_LM_REDUCE_VOCAB_PER_RANK;
        const uint8_t *winner_certified_flags = spark_lm_reduce_certified(
            winner_rank, certified_r0, certified_r1, certified_r2, certified_r3);
        const bool winner_certified = winner_certified_flags[winner_local] != 0u;
        unsigned int status_bits = 0u;
        if (total_certified != 0u) {
            status_bits |= SPARK_LM_REDUCE_STATUS_ANY_CERTIFIED;
        }
        if (winner_certified) {
            status_bits |= SPARK_LM_REDUCE_STATUS_WINNER_CERTIFIED;
        } else if (total_certified == 0u) {
            status_bits |= SPARK_LM_REDUCE_STATUS_TOP5_CLAIMED;
        }
        *token = best_index;
        *value = best_value;
        for (uint32_t block = 0u; block < SPARK_LM_REDUCE_BLOCKS; ++block) {
            block_max[block] = s_block_max[block];
            block_argmax[block] = s_block_argmax[block];
        }
        for (uint32_t slot = 0u; slot < SPARK_LM_REDUCE_TOP; ++slot) {
            top5_value[slot] = top_value[slot];
            top5_index[slot] = top_index[slot];
        }
        *status = status_bits;
        *certified_count = total_certified;
    }
    (void)trace_ring;
    (void)layer_id;
    (void)step_seq;

        __syncthreads();
    }
    (void)trace_ring;
    (void)layer_id;
    (void)step_seq;
}

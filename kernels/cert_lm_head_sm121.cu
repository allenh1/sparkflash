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

// kernels/cert_lm_head_sm121.cu
//
// The certified LM head of specification 2.1/6.2: 129,280 vocabulary entries over 4 ranks means
// 32,320 logits per rank against a 5,120-wide hidden state. Computing all of them exactly is the
// most expensive single step of the decode loop, so this kernel computes a cheap upper bound per
// entry first and skips the exact dot product when the bound proves the entry cannot be the
// maximum.
//
// The certificate, stated precisely, because this is the whole point of the kernel:
//
//   bound_j = sum over the 40 hidden blocks b of ( sum_{i in b} |h_i| ) * ( max_{i in b} |w_ji| )
//
// The per-block row maxima are a load-time precompute (w_block_max, [vocab][40] BF16). The
// checkpoint weights are BF16, so each maximum is a selection of a weight already in the row; the
// precomputed value is bit-for-bit the value the old in-kernel scan produced, and the bound below
// is the same arithmetic in the same block order. That is why the certified flags, the pruned
// logits and certified_count are bit-identical to the scan version, not merely within a tolerance.
//
//   For every entry j,  logits_j <= bound_j, since |sum_i h_i w_ji| <= sum_i |h_i| |w_ji| <=
//   sum_b S_b M_jb. Each thread walks its entries in strictly ascending index order against its own
//   running maximum, and skips an entry only when bound_j < running_max. Because running_max never
//   decreases outside a skip, every skipped entry satisfies logits_j <= bound_j < running_max
//   <= the final maximum, so no skipped entry can be the strict argmax and the returned maximum is
//   exact. Ties are preserved because the test is strict.
//
// The reference in src/lm_head_ref.rs implements exactly this loop, prunes nothing on its own, and
// the tests check three things: that entries with certified == 0 are bit-identical to the exact
// head, that entries with certified == 1 carry a bound which is still an upper bound, and that the
// maximum and the argmax survive the pruning. A mutation that replaces |w_ji| with w_ji (not a
// bound) must break the certificate on at least one input, which is how the tests show the
// pruning argument is not vacuous.
//
// Grid-wide finalisation is deliberately NOT here: each block publishes its own maximum and
// argmax, and the 32-value reduction across blocks is a separate step. That avoids a last-block
// rendezvous, and therefore any question about which fence an observing block would need.
//
// Verification status: never executed. nvcc 12.9.1 compiles it for sm_121a, clang parses it
// against the stub headers, src/m4_abi.rs pins the signature and geometry, src/lm_head_ref.rs
// defines and tests the numerics and the certificate on the CPU. No tensor-core path, no latency,
// and no claim that the bound's cost model beats the exact dot on hardware: on scalar code the
// bound costs a max-reduce per 128 values while the exact path costs a MAC, and the win belongs to
// an int8 tensor-core screen. What is proven here is the correctness of the pruning, not its speed.

#include "m4_common_sm121.cuh"

// The screen tile of specification 2.1, and one thread per entry of a 128-block chunk.
#define SPARK_M4_LM_HEAD_THREADS 128u
#define SPARK_M4_LM_BLOCKS (SPARK_M4_HIDDEN / SPARK_M4_LM_SCREEN_TILE)
#define SPARK_M4_LM_ENTRIES_PER_THREAD 8u
#define SPARK_M4_LM_CHUNK (SPARK_M4_LM_HEAD_THREADS * SPARK_M4_LM_ENTRIES_PER_THREAD)
// The argmax rule this file implements: when several entries tie at the maximum, the LOWEST entry
// index wins, in the per-thread scan and again in the block reduction. src/lm_head_ref.rs
// implements the same rule, and src/m4_abi.rs asserts the tie-break clause is present in both
// reductions here, so the rule cannot be dropped by an edit that only compares values.
#define SPARK_M4_LM_TIE_BREAK_BY_INDEX 1u
static_assert(SPARK_M4_LM_TIE_BREAK_BY_INDEX == 1u, "ties are broken by the lower entry index");

static_assert(SPARK_M4_LM_SCREEN_TILE == 128u, "the screen tile is 128 hidden values");
static_assert(SPARK_M4_HIDDEN % SPARK_M4_LM_SCREEN_TILE == 0u, "the hidden dimension tiles by 128");
static_assert(SPARK_M4_LM_BLOCKS == 40u, "5120 hidden values are 40 screen blocks");
static_assert(SPARK_M4_LM_HEAD_THREADS == SPARK_M4_LM_SCREEN_TILE, "one entry per thread per pass");
static_assert(SPARK_M4_LM_CHUNK == 1024u, "a block owns 1024 vocabulary entries");
static_assert(SPARK_M4_VOCAB_PER_RANK / SPARK_M4_LM_CHUNK >= 31u, "32,320 entries are 32 chunks");
static_assert(SPARK_M4_VOCAB % SPARK_M4_VOCAB_PER_RANK == 0u, "the vocabulary shards across ranks");

extern "C" __global__ void cert_lm_head_sm121(
    const __nv_bfloat16 *__restrict__ hidden,   // [hidden]
    const __nv_bfloat16 *__restrict__ w,        // [vocab][hidden] bf16, this rank's shard
    const __nv_bfloat16 *__restrict__ w_block_max, // [vocab][40] bf16, exact block row maxima
    float *__restrict__ logits,                 // [vocab] exact where certified == 0, else the bound
    uint8_t *__restrict__ certified,            // [vocab] 1 = pruned, logits holds an upper bound
    unsigned int *__restrict__ certified_count, // [1] total pruned entries, atomically accumulated
    float *__restrict__ block_max,              // [gridDim.x] this block's maximum
    uint32_t *__restrict__ block_argmax,        // [gridDim.x] the entry that scored it
    const uint32_t vocab,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t lane = threadIdx.x;
    __shared__ float s_h_abs[SPARK_M4_LM_BLOCKS];
    __shared__ float s_thread_max[SPARK_M4_LM_HEAD_THREADS];
    __shared__ uint32_t s_thread_argmax[SPARK_M4_LM_HEAD_THREADS];

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    // One block of the hidden state's absolute sums, computed cooperatively once per block: every
    // thread accumulates its share and the reduction folds them. This is the only shared input the
    // bound needs, which is why the screen is cheap compared with the exact dot.
    for (uint32_t block = 0; block < (uint32_t)SPARK_M4_LM_BLOCKS; ++block) {
        float local = 0.0f;
        for (uint32_t i = lane; i < (uint32_t)SPARK_M4_LM_SCREEN_TILE; i += SPARK_M4_LM_HEAD_THREADS) {
            const float value = __bfloat162float(hidden[block * SPARK_M4_LM_SCREEN_TILE + i]);
            local += value < 0.0f ? -value : value;
        }
        local = spark_block_sum(local, lane, SPARK_M4_LM_HEAD_THREADS);
        if (lane == 0u) {
            s_h_abs[block] = local;
        }
    }
    __syncthreads();

    // The block's first entry index; the last block may have fewer than CHUNK entries left.
    const uint32_t base = blockIdx.x * SPARK_M4_LM_CHUNK;
    float running_max = -3.402823466e38f;
    uint32_t running_argmax = 0u;
    unsigned int pruned = 0u;
    for (uint32_t step = 0; step < SPARK_M4_LM_ENTRIES_PER_THREAD; ++step) {
        // Strictly ascending within the thread: base + lane, base + lane + 128, ...
        const uint32_t entry = base + step * SPARK_M4_LM_HEAD_THREADS + lane;
        if (entry >= vocab) {
            continue;
        }
        const __nv_bfloat16 *row = w + (uint64_t)entry * SPARK_M4_HIDDEN;
        float bound = 0.0f;
#pragma unroll 8
        for (uint32_t block = 0; block < (uint32_t)SPARK_M4_LM_BLOCKS; ++block) {
            // The load-time precompute selected this exact BF16 weight as the block's largest
            // magnitude, so this replaces the old 128-value scan without changing the value.
            const float row_max = __bfloat162float(
                w_block_max[(uint64_t)entry * SPARK_M4_LM_BLOCKS + block]);
            bound += s_h_abs[block] * row_max;
        }
        if (bound < running_max) {
            // Certified: the exact dot cannot beat the running maximum.
            logits[entry] = bound;
            certified[entry] = 1u;
            pruned += 1u;
        } else {
            float acc = 0.0f;
#pragma unroll 8
            for (uint32_t i = 0; i < (uint32_t)SPARK_M4_HIDDEN; ++i) {
                acc += __bfloat162float(hidden[i]) * __bfloat162float(row[i]);
            }
            logits[entry] = acc;
            certified[entry] = 0u;
            // Ties go to the LOWEST entry index, which is the rule src/lm_head_ref.rs implements
            // and the rule a sequential scan gives. Comparing values alone is not enough here: the
            // entries are dealt round-robin, so entry 300 belongs to lane 44 and entry 100 to lane
            // 100, and a value-only reduction would return 300 for a tie between them because lane
            // 44 is visited first. The index comparison below is what makes the two agree.
            if (acc > running_max || (acc == running_max && entry < running_argmax)) {
                running_max = acc;
                running_argmax = entry;
            }
        }
    }

    // Block reduction, in ascending lane order, so the result does not depend on scheduler order.
    // The index tie-break repeats the per-thread rule, so the block's answer is the lowest entry
    // index among ties and not the lowest lane that happened to hold one.
    s_thread_max[lane] = running_max;
    s_thread_argmax[lane] = running_argmax;
    __syncthreads();
    if (lane == 0u) {
        float best = s_thread_max[0];
        uint32_t best_index = s_thread_argmax[0];
        for (uint32_t i = 1u; i < SPARK_M4_LM_HEAD_THREADS; ++i) {
            if (s_thread_max[i] > best ||
                (s_thread_max[i] == best && s_thread_argmax[i] < best_index)) {
                best = s_thread_max[i];
                best_index = s_thread_argmax[i];
            }
        }
        block_max[blockIdx.x] = best;
        block_argmax[blockIdx.x] = best_index;
        const uint32_t meta = SPARK_M4_LM_CHUNK * 4u;
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, meta, t_enter);
        (void)step_seq;
    }
    // The count is the number of ALL certified entries, so every thread contributes its own prunes.
    // Until D31 the atomic lived inside the lane-0 block above and added lane 0's count alone, which
    // undercounted the pruning path (4,899 flags against 34 published in windows b21/b23). The count
    // is a sum of integers, so the accumulation order does not matter and the value is deterministic
    // across runs; a thread with no prunes skips the atomic, so the all-exact path is unchanged.
    if (pruned > 0u) {
        atomicAdd(certified_count, pruned);
    }
    (void)t_enter;
}
// K8 batched certified head: one launch over the six-row verify batch. The weights and the exact
// BF16 row maxima are read once and reused across the six hidden rows; each row keeps its own
// running maximum, pruning flags, logits and block reductions. The arithmetic of every row is the
// scalar kernel's arithmetic in the same order (bound blocks ascending, exact dot K ascending,
// strict '<' pruning, strict '>' maximum with the lower-index tie break), so every per-row output
// is BIT-IDENTICAL to cert_lm_head_sm121.
#define SPARK_M4_LM_BATCH_ROWS 6u

static_assert(SPARK_M4_LM_BATCH_ROWS == 6u, "the six-row verify batch is the fixed shape");

extern "C" __global__ void cert_lm_head_batched_sm121(
    const __nv_bfloat16 *__restrict__ hidden,   // [rows][hidden]
    const __nv_bfloat16 *__restrict__ w,        // [vocab][hidden] bf16, this rank's shard
    const __nv_bfloat16 *__restrict__ w_block_max_t, // [40][vocab] bf16, coalesced bound reads
    float *__restrict__ logits,                 // [rows][vocab]
    uint8_t *__restrict__ certified,            // [rows][vocab]
    unsigned int *__restrict__ certified_count, // [rows] total pruned entries per row
    float *__restrict__ block_max,              // [rows][gridDim.x]
    uint32_t *__restrict__ block_argmax,        // [rows][gridDim.x]
    const uint32_t vocab,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t lane = threadIdx.x;
    __shared__ float s_h_abs[SPARK_M4_LM_BATCH_ROWS][SPARK_M4_LM_BLOCKS];
    __shared__ float s_thread_max[SPARK_M4_LM_BATCH_ROWS][SPARK_M4_LM_HEAD_THREADS];
    __shared__ uint32_t s_thread_argmax[SPARK_M4_LM_BATCH_ROWS][SPARK_M4_LM_HEAD_THREADS];

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    for (uint32_t r = 0u; r < SPARK_M4_LM_BATCH_ROWS; ++r) {
        const __nv_bfloat16 *hidden_row = hidden + (uint64_t)r * SPARK_M4_HIDDEN;
        for (uint32_t block = 0u; block < (uint32_t)SPARK_M4_LM_BLOCKS; ++block) {
            float local = 0.0f;
            for (uint32_t i = lane; i < (uint32_t)SPARK_M4_LM_SCREEN_TILE;
                 i += SPARK_M4_LM_HEAD_THREADS) {
                const float value =
                    __bfloat162float(hidden_row[block * SPARK_M4_LM_SCREEN_TILE + i]);
                local += value < 0.0f ? -value : value;
            }
            local = spark_block_sum(local, lane, SPARK_M4_LM_HEAD_THREADS);
            if (lane == 0u) {
                s_h_abs[r][block] = local;
            }
        }
    }
    __syncthreads();

    const uint32_t base = blockIdx.x * SPARK_M4_LM_CHUNK;
    float running_max[SPARK_M4_LM_BATCH_ROWS];
    uint32_t running_argmax[SPARK_M4_LM_BATCH_ROWS];
    unsigned int pruned[SPARK_M4_LM_BATCH_ROWS];
#pragma unroll
    for (uint32_t r = 0u; r < SPARK_M4_LM_BATCH_ROWS; ++r) {
        running_max[r] = -3.402823466e38f;
        running_argmax[r] = 0u;
        pruned[r] = 0u;
    }
    for (uint32_t step = 0; step < SPARK_M4_LM_ENTRIES_PER_THREAD; ++step) {
        const uint32_t entry = base + step * SPARK_M4_LM_HEAD_THREADS + lane;
        if (entry >= vocab) {
            continue;
        }
        const __nv_bfloat16 *row = w + (uint64_t)entry * SPARK_M4_HIDDEN;
        float bound[SPARK_M4_LM_BATCH_ROWS];
#pragma unroll
        for (uint32_t r = 0u; r < SPARK_M4_LM_BATCH_ROWS; ++r) {
            bound[r] = 0.0f;
        }
#pragma unroll 8
        for (uint32_t block = 0u; block < (uint32_t)SPARK_M4_LM_BLOCKS; ++block) {
            const float row_max =
                __bfloat162float(w_block_max_t[(uint64_t)block * vocab + entry]);
#pragma unroll
            for (uint32_t r = 0u; r < SPARK_M4_LM_BATCH_ROWS; ++r) {
                bound[r] += s_h_abs[r][block] * row_max;
            }
        }
        bool needs_exact[SPARK_M4_LM_BATCH_ROWS];
        bool any_exact = false;
#pragma unroll
        for (uint32_t r = 0u; r < SPARK_M4_LM_BATCH_ROWS; ++r) {
            needs_exact[r] = !(bound[r] < running_max[r]);
            any_exact = any_exact || needs_exact[r];
        }
        float acc[SPARK_M4_LM_BATCH_ROWS];
#pragma unroll
        for (uint32_t r = 0u; r < SPARK_M4_LM_BATCH_ROWS; ++r) {
            acc[r] = 0.0f;
        }
        if (any_exact) {
#pragma unroll 4
            for (uint32_t i = 0u; i < (uint32_t)SPARK_M4_HIDDEN; ++i) {
                const float w_value = __bfloat162float(row[i]);
#pragma unroll
                for (uint32_t r = 0u; r < SPARK_M4_LM_BATCH_ROWS; ++r) {
                    if (needs_exact[r]) {
                        acc[r] += __bfloat162float(hidden[(uint64_t)r * SPARK_M4_HIDDEN + i]) *
                                  w_value;
                    }
                }
            }
        }
#pragma unroll
        for (uint32_t r = 0u; r < SPARK_M4_LM_BATCH_ROWS; ++r) {
            const uint64_t out_index = (uint64_t)r * vocab + entry;
            if (!needs_exact[r]) {
                logits[out_index] = bound[r];
                certified[out_index] = 1u;
                pruned[r] += 1u;
            } else {
                logits[out_index] = acc[r];
                certified[out_index] = 0u;
                if (acc[r] > running_max[r] ||
                    (acc[r] == running_max[r] && entry < running_argmax[r])) {
                    running_max[r] = acc[r];
                    running_argmax[r] = entry;
                }
            }
        }
    }

#pragma unroll
    for (uint32_t r = 0u; r < SPARK_M4_LM_BATCH_ROWS; ++r) {
        s_thread_max[r][lane] = running_max[r];
        s_thread_argmax[r][lane] = running_argmax[r];
    }
    __syncthreads();
    if (lane == 0u) {
#pragma unroll
        for (uint32_t r = 0u; r < SPARK_M4_LM_BATCH_ROWS; ++r) {
            float best = s_thread_max[r][0];
            uint32_t best_index = s_thread_argmax[r][0];
            for (uint32_t i = 1u; i < SPARK_M4_LM_HEAD_THREADS; ++i) {
                if (s_thread_max[r][i] > best ||
                    (s_thread_max[r][i] == best && s_thread_argmax[r][i] < best_index)) {
                    best = s_thread_max[r][i];
                    best_index = s_thread_argmax[r][i];
                }
            }
            block_max[(uint64_t)r * gridDim.x + blockIdx.x] = best;
            block_argmax[(uint64_t)r * gridDim.x + blockIdx.x] = best_index;
        }
        const uint32_t meta = SPARK_M4_LM_CHUNK * 4u;
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, meta, t_enter);
        (void)step_seq;
    }
#pragma unroll
    for (uint32_t r = 0u; r < SPARK_M4_LM_BATCH_ROWS; ++r) {
        if (pruned[r] > 0u) {
            atomicAdd(certified_count + r, pruned[r]);
        }
    }
    (void)t_enter;
}

// kernels/flash_mla_csa2_sm121.cu
//
// FlashMLA with CSA2 (candidate-selected attention, specification 2.1/2.3): one MLA head per
// block, 512-wide latent Main KV stored as FP4 E2M1 with an E4M3 scale per 16 channels, a
// 128-token sliding window, and a candidate set selected from the 2,048 x 8 high-speed-index
// pool (16,384 positions, index_n_heads 32, index_head_dim 128, index_topk 512).
//
// Two stages live here:
//   1. indexer scoring: for this head, score every HSI block with a 128-wide dot product
//      against the block's index keys and record the block's best token. This is the cheap
//      pruning that CSA2's candidate selection consumes.
//   2. attention over the selected candidates: online softmax in f32 over the candidate list,
//      then the weighted sum of the dequantized 512-wide latent.
//
// Deliberate scope limits, so nothing here reads as more than it is:
//   * the candidates arrive already selected (a [selected_count] list of position ids). The full
//     top-512 merge over the 2,048 block candidates is a separate reduction step in the pipeline
//     and is NOT implemented in this file; this kernel defines the attention and indexer scoring
//     numerics, which is what a CPU reference can pin without a GPU.
//   * rotary position embedding of the 64 qk_rope dimensions is assumed already applied by the
//     caller, so this kernel consumes rotated q and k. Nothing here claims to implement rope.
//   * the FP4 latent is dequantized in the scalar path. A tensor-core variant would accumulate
//     in the hardware's order and is not claimed to be bit-identical to flash_mla_ref.rs.
//   * the attention sink is one additive logit per head (`layers.N.attn.attn_sink`, F32 [64])
//     and enters the softmax DENOMINATOR only, after the maximum, exactly as the released
//     implementation does (kernel.py:382-383): a sink above the maximum shrinks an output, it
//     never adds a value to it. This is the same arithmetic as mla_ref::attention_with_sink.
//
// Verification status: executed on node0 (GB10, sm_121a) on 2026-10-17. nvcc 13.0.3
// compiles it for sm_121a, clang parses it against the stub headers, src/m4_abi.rs pins the
// signature and geometry, and src/mla_ref.rs defines and tests the numerics (including that the
// online softmax equals a two-pass softmax and that the sliding window predicate is applied
// exactly once). The device run: stage 2 of examples/m7_layer0.rs, with the checkpoint sink
// consumed on both sides, PASSed 5/5 positions - 163,838/163,840 outputs (99.9988%) bit-exact
// against the bf16 rounding of the f32 reference, max abs delta 9.765625e-4 (one bf16 ULP), max
// ULP distance 1. That is internal consistency of the sink arithmetic against our reference on
// real checkpoint bytes, not a claim of bit-identity with the released kernel.

#include "m4_common_sm121.cuh"

#define SPARK_M4_MLA_THREADS 256u
#define SPARK_M4_MLA_LATENT SPARK_M4_KV_LORA_RANK
#define SPARK_M4_MLA_FP4_PER_BYTE 2u
// One score per selected position lives in shared memory, which is what bounds the candidate
// count: 512 positions of f32 plus the running statistics.
#define SPARK_M4_MLA_MAX_SELECTED 512u

static_assert(SPARK_M4_MLA_LATENT % SPARK_M4_KV_SCALE_CHANNELS == 0u,
              "the latent divides into 16-channel scale groups");
static_assert(SPARK_M4_MLA_LATENT % SPARK_M4_MLA_THREADS == 0u, "the latent tiles across the block");
static_assert(SPARK_M4_MLA_LATENT / SPARK_M4_MLA_THREADS == 2u, "two latent values per thread");
static_assert(SPARK_M4_MLA_MAX_SELECTED == SPARK_M4_INDEX_TOPK, "the candidate list is index_topk");
static_assert(SPARK_M4_KV_SCALE_CHANNELS == 16u, "an E4M3 scale per 16 channels");
static_assert(SPARK_M4_SWA_WINDOW == 128u, "the sliding window is 128 tokens");
static_assert(SPARK_M4_MLA_THREADS == SPARK_M4_REDUCE_SLOTS, "the block reduction needs 256 threads");
// The query heads and the indexer heads are two different index spaces: 64 query heads, 32 indexer
// heads (8 per rank), so one indexer head serves two query heads and the indexer query must be
// addressed as head / SPARK_M4_MLA_HEADS_PER_INDEX_HEAD. Using the query head directly would read
// past the end of a [32][128] index query for half the blocks and give the wrong head's score for
// the rest. SPARK_M4_MLA_SCORE_SCALE is the MLA score scale, defined once in the shared header.
static_assert(SPARK_M4_HEADS == 64u, "64 query heads");
static_assert(SPARK_M4_INDEX_HEADS == 32u, "32 indexer heads");
static_assert(SPARK_M4_MLA_HEADS_PER_INDEX_HEAD == 2u, "two query heads per indexer head");
// No C cast in a static_assert: the Rust reader evaluates these expressions itself, and a
// parenthesised type name would be an identifier it cannot resolve.
static_assert(SPARK_M4_HSI_BLOCKS * SPARK_M4_HSI_BLOCK_TOKENS == SPARK_M4_HSI_POSITIONS,
              "the HSI pool is 2048 blocks of 8 tokens");

// One warp scores one HSI block. The indexer's dot is 128 values wide, four per lane, so a
// five-step __shfl_down_sync tree reduces it with no shared memory and no block barrier. The old
// shape reduced the same 128 lanes inside a 256-thread shared-memory tree: nine __syncthreads()
// per slot-dot and half the threads contributing nothing, which made the indexer a barrier chain
// rather than arithmetic. Eight warps now score eight independent HSI blocks at once, and the
// blockIdx.y slice strides in units of eight. The shuffle is declared by the clang syntax stub as
// well, so the host-side ABI probe still parses.
#define SPARK_M4_MLA_WARP_LANES 32u
#define SPARK_M4_MLA_WARPS (SPARK_M4_MLA_THREADS / SPARK_M4_MLA_WARP_LANES)
static_assert(SPARK_M4_MLA_THREADS % SPARK_M4_MLA_WARP_LANES == 0u,
              "the block divides into whole warps");
static_assert(SPARK_M4_INDEX_HEAD_DIM == 4u * SPARK_M4_MLA_WARP_LANES,
              "four indexer dimensions per lane");

__device__ __forceinline__ float spark_mla_warp_sum(float value) {
    value += __shfl_down_sync(0xffffffffu, value, 16u);
    value += __shfl_down_sync(0xffffffffu, value, 8u);
    value += __shfl_down_sync(0xffffffffu, value, 4u);
    value += __shfl_down_sync(0xffffffffu, value, 2u);
    value += __shfl_down_sync(0xffffffffu, value, 1u);
    return value;
}

// A position is attendable when it is inside the sliding window behind the query. The predicate
// is a function of the query position and the candidate id, so the CPU reference can test it
// directly instead of duplicating the comparison.
__device__ __forceinline__ uint32_t spark_mla_in_window(uint32_t query_position, uint32_t candidate) {
    return query_position >= candidate && (query_position - candidate) < (uint32_t)SPARK_M4_SWA_WINDOW;
}

__device__ __forceinline__ float spark_mla_load_latent(const uint8_t *row, uint32_t index) {
    const uint8_t packed = row[index / SPARK_M4_MLA_FP4_PER_BYTE];
    const uint8_t nibble = (index % SPARK_M4_MLA_FP4_PER_BYTE) == 0u ? (uint8_t)(packed & 0x0Fu)
                                                                    : (uint8_t)(packed >> 4);
    return spark_fp4_e2m1_decode(nibble);
}

// The E4M3 scale that covers latent channel index: one per 16 channels, stored as bytes.
__device__ __forceinline__ float spark_mla_scale(const uint8_t *scales, uint32_t index) {
    return spark_fp8_e4m3_decode(scales[index / SPARK_M4_KV_SCALE_CHANNELS]);
}

extern "C" __global__ void flash_mla_csa2_sm121(
    const __nv_bfloat16 *__restrict__ q,             // [heads][latent] query, rope already applied
    const uint8_t *__restrict__ kv_fp4,              // [positions][latent/2] packed E2M1 latent
    const uint8_t *__restrict__ kv_scales,           // [positions][latent/16] E4M3 scales
    const uint32_t *__restrict__ selected,           // [selected_count] candidate position ids
    const __nv_bfloat16 *__restrict__ index_key,     // [hsi_blocks][8][index_head_dim]
    const __nv_bfloat16 *__restrict__ index_query,   // [heads][index_head_dim]
    const float *__restrict__ sink,                  // [heads] additive per-head attention sink logit
    float *__restrict__ block_scores,                // [hsi_blocks] out: best score in the block
    uint32_t *__restrict__ block_best,               // [hsi_blocks] out: the token that scored it
    __nv_bfloat16 *__restrict__ out,                 // [heads][latent] attention output
    const uint32_t position,
    const uint32_t selected_count,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t head = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t heads = (uint32_t)SPARK_M4_HEADS;
    // The indexer head that serves this query head: the two are different index spaces (64 query
    // heads, 32 indexer heads), so this division is required, not cosmetic.
    const uint32_t index_head = head / (uint32_t)SPARK_M4_MLA_HEADS_PER_INDEX_HEAD;

    __shared__ float s_scores[SPARK_M4_MLA_MAX_SELECTED];
    __shared__ float s_max;
    __shared__ float s_sum;

    uint64_t t_enter = 0u;
    // The trace is recorded by one block (blockIdx.x == 0, thread 0) and the slice is the extra
    // dimension the indexer added. Guarding the entry here keeps the enter/exit pair balanced
    // with the early return below: only slice 0 enters and only slice 0 exits.
    if (blockIdx.y == 0u) {
        SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_CSA2_ATTN, t_enter);
    }

    // Stage 1: score the HSI blocks for this head. Each block owns only its own block slot, so
    // there is no cross-block communication and nothing to acquire here.
    if (lane == 0u) {
        s_max = -3.402823466e38f;
        s_sum = 1.0f;
    }
    __syncthreads();
    // The two query heads of one indexer head compute the same 128-wide dots over the same keys:
    // one indexer head serves both, and the index query is addressed by it, not by the query head.
    // The pair therefore splits the pool instead of duplicating it. The even head takes the even
    // HSI blocks and the odd head the odd ones, so every block is still written, by exactly one
    // head of the pair, and the full-head launch (grid.x = SPARK_M4_HEADS) covers all 2,048. A
    // partial launch would leave the other parity stale; that is a property of the pairing, and
    // the fixed grid is what makes it safe.
    const __nv_bfloat16 *query_row =
        index_query + (uint64_t)index_head * SPARK_M4_INDEX_HEAD_DIM;
    const uint32_t lane_in_warp = lane % SPARK_M4_MLA_WARP_LANES;
    const uint32_t warp = lane / SPARK_M4_MLA_WARP_LANES;
    const uint32_t pair = head % (uint32_t)SPARK_M4_MLA_HEADS_PER_INDEX_HEAD;
    const uint32_t pair_stride = (uint32_t)SPARK_M4_MLA_HEADS_PER_INDEX_HEAD;
    const uint32_t blocks_per_slice = SPARK_M4_MLA_WARPS * pair_stride;
    const uint32_t block_stride = gridDim.y * blocks_per_slice;
    const float q0 = __bfloat162float(query_row[lane_in_warp]);
    const float q1 = __bfloat162float(query_row[lane_in_warp + SPARK_M4_MLA_WARP_LANES]);
    const float q2 = __bfloat162float(query_row[lane_in_warp + 2u * SPARK_M4_MLA_WARP_LANES]);
    const float q3 = __bfloat162float(query_row[lane_in_warp + 3u * SPARK_M4_MLA_WARP_LANES]);
    for (uint32_t block = blockIdx.y * blocks_per_slice + warp * pair_stride + pair;
         block < (uint32_t)SPARK_M4_HSI_BLOCKS;
         block += block_stride) {
        float best = -3.402823466e38f;
        uint32_t best_token = 0u;
        for (uint32_t slot = 0; slot < (uint32_t)SPARK_M4_HSI_BLOCK_TOKENS; ++slot) {
            const __nv_bfloat16 *key_row =
                index_key + ((uint64_t)block * SPARK_M4_HSI_BLOCK_TOKENS + slot) *
                                SPARK_M4_INDEX_HEAD_DIM;
            float score = q0 * __bfloat162float(key_row[lane_in_warp]) +
                          q1 * __bfloat162float(key_row[lane_in_warp + SPARK_M4_MLA_WARP_LANES]) +
                          q2 * __bfloat162float(key_row[lane_in_warp + 2u * SPARK_M4_MLA_WARP_LANES]) +
                          q3 * __bfloat162float(key_row[lane_in_warp + 3u * SPARK_M4_MLA_WARP_LANES]);
            // Reduce the four partial products of each lane inside the warp.
            score = spark_mla_warp_sum(score);
            if (lane_in_warp == 0u && score > best) {
                best = score;
                best_token = block * SPARK_M4_HSI_BLOCK_TOKENS + slot;
            }
        }
        if (lane_in_warp == 0u) {
            block_scores[block] = best;
            block_best[block] = best_token;
        }
    }

    // blockIdx.y exists only to slice the indexer's HSI pool, but stage 2 used to run unguarded
    // in every slice: each blockIdx.y re-ran the whole attention for the same head and wrote the
    // same out[] values. At a grid of 64 that performed 64 times the attention work for one
    // result, and the launch paid about 1.47 ms for it at 128 candidates (measured on node0
    // as the --tokens 130 minus --tokens 1 difference at grid.y 64, before this guard). One
    // execution of that attention is 0.068 ms, so the redundant copies overlapped rather than
    // serialising; the work was still real. Only slice 0 runs the attention now.
    if (blockIdx.y != 0u) {
        return;
    }

    // Stage 2: attention over the selected candidates. Pass one computes every score and the
    // running maximum, pass two the normalising sum, pass three the weighted latent sum. The
    // candidate count is bounded by SPARK_M4_MLA_MAX_SELECTED; the caller guarantees that.
    for (uint32_t slot = lane; slot < selected_count; slot += SPARK_M4_MLA_THREADS) {
        const uint32_t candidate = selected[slot];
        float score = 0.0f;
        if (spark_mla_in_window(position, candidate) != 0u) {
            for (uint32_t i = 0; i < SPARK_M4_MLA_LATENT; ++i) {
                const float key = spark_mla_load_latent(kv_fp4 + (uint64_t)candidate * (SPARK_M4_MLA_LATENT / 2u), i) *
                                  spark_mla_scale(kv_scales + (uint64_t)candidate * (SPARK_M4_MLA_LATENT / SPARK_M4_KV_SCALE_CHANNELS), i);
                score += __bfloat162float(q[(uint64_t)head * SPARK_M4_MLA_LATENT + i]) * key;
            }
            // The MLA score scale, 1/sqrt(512) over the latent width. Specification 2.3 fixes the
            // selection rule but not this constant, so it is named once in m4_common_sm121.cuh and
            // src/mla_ref.rs carries the same literal, compared bit for bit by a test there.
            score *= SPARK_M4_MLA_SCORE_SCALE;
        } else {
            score = -3.402823466e38f; // outside the window: excluded, not merely down-weighted
        }
        s_scores[slot] = score;
    }
    __syncthreads();
    // The maximum is taken per thread and then reduced as a tree. It must not be accumulated into
    // the shared s_max by every thread: that read-modify-write is a data race, lost updates leave
    // s_max below the true maximum, and the softmax below then overflows to infinity and divides
    // to NaN. spark_block_max is a rendezvous with a fixed tree order, so every thread reaches it
    // and the result is the same on every run, which is what the CPU reference can pin.
    float local_max = -3.402823466e38f;
    for (uint32_t slot = lane; slot < selected_count; slot += SPARK_M4_MLA_THREADS) {
        local_max = s_scores[slot] > local_max ? s_scores[slot] : local_max;
    }
    local_max = spark_block_max(local_max, lane, SPARK_M4_MLA_THREADS);
    if (lane == 0u) {
        s_max = local_max;
    }
    __syncthreads();
    // The sink contributes probability mass but no value, so it is one term of the denominator
    // and nothing else. Adding it after the maximum is the released order (kernel.py:382-383);
    // thread 0 owns the term so the reduction order stays fixed run to run.
    float local_sum = lane == 0u ? spark_exp(sink[head] - s_max) : 0.0f;
    for (uint32_t slot = lane; slot < selected_count; slot += SPARK_M4_MLA_THREADS) {
        const float probability = spark_exp(s_scores[slot] - s_max);
        s_scores[slot] = probability;
        local_sum += probability;
    }
    local_sum = spark_block_sum(local_sum, lane, SPARK_M4_MLA_THREADS);
    if (lane == 0u) {
        s_sum = local_sum > 0.0f ? local_sum : 1.0f;
    }
    // The publish of s_sum needs its own rendezvous, for the same reason s_max does: without it a
    // warp that loses the race reads the prologue's 1.0f and divides the latent sum by one instead
    // of by the normaliser. That is not a rounding error - it scales the head's output by s_sum and
    // it is nondeterministic, so the CPU reference cannot agree with it. Stage 2 of the layer-0
    // harness caught exactly that: identical input scored 96.1% exact on one run and 87.5% on the
    // next, and every over-tolerance value was a head that had divided by the wrong denominator.
    __syncthreads();
    // Each thread owns two latent channels and walks the candidate list in ascending slot order.
    // That order is part of the contract, so this deliberately does not use shared-memory atomics:
    // an atomic sum would make the rounding depend on warp scheduling and no CPU reference could
    // pin it. Out-of-window candidates contribute exactly zero, because exp(-FLT_MAX - max) is zero
    // and the window test below also skips them; the reference does the same.
    for (uint32_t i = lane; i < SPARK_M4_MLA_LATENT; i += SPARK_M4_MLA_THREADS) {
        float acc = 0.0f;
        for (uint32_t slot = 0; slot < selected_count; ++slot) {
            const uint32_t candidate = selected[slot];
            if (spark_mla_in_window(position, candidate) != 0u) {
                const float value = spark_mla_load_latent(kv_fp4 + (uint64_t)candidate * (SPARK_M4_MLA_LATENT / 2u), i) *
                                    spark_mla_scale(kv_scales + (uint64_t)candidate * (SPARK_M4_MLA_LATENT / SPARK_M4_KV_SCALE_CHANNELS), i);
                acc += s_scores[slot] * value;
            }
        }
        out[(uint64_t)head * SPARK_M4_MLA_LATENT + i] = __float2bfloat16_rn(acc / s_sum);
    }

    if (lane == 0u) {
        const uint32_t meta = (uint32_t)(SPARK_M4_MLA_LATENT * 2u);
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_CSA2_ATTN, layer_id, meta, t_enter);
        (void)heads;
        (void)step_seq;
    }
    (void)t_enter;
}

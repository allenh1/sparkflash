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
//
// Verification status: never executed. nvcc 12.9.1 compiles it for sm_121a, clang parses it
// against the stub headers, src/m4_abi.rs pins the signature and geometry, and
// src/mla_ref.rs defines and tests the numerics (including that the online softmax equals a
// two-pass softmax and that the sliding window predicate is applied exactly once).

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
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_CSA2_ATTN, t_enter);

    // Stage 1: score the HSI blocks for this head. Each block owns only its own block slot, so
    // there is no cross-block communication and nothing to acquire here.
    if (lane == 0u) {
        s_max = -3.402823466e38f;
        s_sum = 1.0f;
    }
    __syncthreads();
    for (uint32_t block = blockIdx.y; block < (uint32_t)SPARK_M4_HSI_BLOCKS; block += gridDim.y) {
        float best = -3.402823466e38f;
        uint32_t best_token = 0u;
        for (uint32_t slot = 0; slot < (uint32_t)SPARK_M4_HSI_BLOCK_TOKENS; ++slot) {
            float score = 0.0f;
            for (uint32_t i = lane; i < (uint32_t)SPARK_M4_INDEX_HEAD_DIM; i += SPARK_M4_MLA_THREADS) {
                score += __bfloat162float(index_query[(uint64_t)index_head * SPARK_M4_INDEX_HEAD_DIM + i]) *
                         __bfloat162float(
                             index_key[((uint64_t)block * SPARK_M4_HSI_BLOCK_TOKENS + slot) *
                                           SPARK_M4_INDEX_HEAD_DIM + i]);
            }
            // Reduce the partial dot product across the block.
            score = spark_block_sum(score, lane, SPARK_M4_MLA_THREADS);
            if (lane == 0u && score > best) {
                best = score;
                best_token = block * SPARK_M4_HSI_BLOCK_TOKENS + slot;
            }
        }
        if (lane == 0u) {
            block_scores[block] = best;
            block_best[block] = best_token;
        }
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
    float local_sum = 0.0f;
    for (uint32_t slot = lane; slot < selected_count; slot += SPARK_M4_MLA_THREADS) {
        const float probability = spark_exp(s_scores[slot] - s_max);
        s_scores[slot] = probability;
        local_sum += probability;
    }
    local_sum = spark_block_sum(local_sum, lane, SPARK_M4_MLA_THREADS);
    if (lane == 0u) {
        s_sum = local_sum > 0.0f ? local_sum : 1.0f;
    }
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

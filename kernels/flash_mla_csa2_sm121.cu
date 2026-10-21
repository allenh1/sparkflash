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
//   * the kernel reads a second Main-KV format when asked: the engine's released window format
//     (E4M3 code per channel, E8M0 scale per 32 channels, bf16-storage round trip). The format is
//     an explicit runtime argument (0 = historical FP4 path, unchanged; 1 = engine window format);
//     there is no host-side validation: the kernel entry guard below refuses any other value. The
//     released engine's own bf16
//     arithmetic is not claimed - the engine reference forward keeps f32 dequant, and that seam is
//     documented - only the format and its bf16 storage are reproduced.
//   * format 2 (CSA union) is the released attention's actual candidate set: the supplied engine
//     window rows are all attended, oldest first, then the selected compressed candidates are
//     attended when candidate <= position (the reference's causal reachability mask; a candidate
//     inside the window is NOT excluded). One f32 softmax runs over that union with the sink in
//     the denominator after the maximum. Out-of-window selected ids therefore contribute, which is
//     the Criterion-1 gap the window-only formats cannot express. Formats 0 and 1 keep their own
//     exclusion rules unchanged.
//
// Verification status: executed on node0 (GB10, sm_121a) on 2026-10-17. nvcc 13.0.3
// compiles it for sm_121a, clang parses it against the stub headers, src/m4_abi.rs pins the
// signature and geometry, and src/mla_ref.rs defines and tests the numerics (including that the
// online softmax equals a two-pass softmax and that the sliding window predicate is applied
// exactly once). The device run: stage 2 of examples/m7_layer0.rs, with the checkpoint sink
// consumed on both sides, PASSed 5/5 positions - 163,838/163,840 outputs (99.9988%) bit-exact
// against the bf16 rounding of the f32 reference, max abs delta 9.765625e-4 (one bf16 ULP), max
// ULP distance 1. That is internal consistency of the sink arithmetic against our reference on
// real checkpoint bytes, not a claim of bit-identity with the released kernel. Formats 0 and 1 are
// those measured columns. Format 2 is reference-tested by CPU tests only: src/mla_ref.rs's union
// twin is compared bit for bit with sparkflash_engine::forward::attention::sparse_attention over the
// concatenated window ++ compressed reference KV, its 640-slot bound is exercised at exactly 640 and
// at 641 (this kernel's entry guard poisons 641 with NaN), and src/m4_abi.rs pins the dispatch text.
// Format 2 WAS first executed on a device on 2026-10-18 in a Lead-sequenced window: the
// m14_flash_mla harness ran this cubin on real checkpoint bytes and PASSed at 8 and 130 query
// positions within the declared stage-2 tolerance class (not bit-identity; the m14 record in
// docs/build/phase3-flash-mla.md holds the measured numbers). That execution does not turn the
// class into a bit-identity claim, and the released kernel was not compared.

#include "m4_common_sm121.cuh"

#define SPARK_M4_MLA_THREADS 256u
#define SPARK_M4_MLA_LATENT SPARK_M4_KV_LORA_RANK
#define SPARK_M4_MLA_FP4_PER_BYTE 2u
// One score per selected position lives in shared memory, which is what bounds the candidate
// count: 512 positions of f32 plus the running statistics. In union format the window rows are
// attended in the same array, so its capacity is one candidate list plus one full window. The
// capacity is written as that expression (no C cast) so the Rust reader evaluates the same sum.
#define SPARK_M4_MLA_MAX_SELECTED 512u
#define SPARK_M4_MLA_MAX_ATTENDED (SPARK_M4_MLA_MAX_SELECTED + SPARK_M4_SWA_WINDOW)

static_assert(SPARK_M4_MLA_LATENT % SPARK_M4_KV_SCALE_CHANNELS == 0u,
              "the latent divides into 16-channel scale groups");
static_assert(SPARK_M4_MLA_LATENT % SPARK_M4_MLA_THREADS == 0u, "the latent tiles across the block");
static_assert(SPARK_M4_MLA_LATENT / SPARK_M4_MLA_THREADS == 2u, "two latent values per thread");
static_assert(SPARK_M4_MLA_MAX_SELECTED == SPARK_M4_INDEX_TOPK, "the candidate list is index_topk");
static_assert(SPARK_M4_MLA_MAX_ATTENDED == 640u,
              "the union capacity is 512 candidates plus the 128-token window");
static_assert(SPARK_M4_MLA_MAX_ATTENDED >= SPARK_M4_MLA_MAX_SELECTED,
              "the union array must hold every historical candidate list");
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

// The engine's released window-KV format (model.py:700-707, kernel.py:98-124): one E4M3 code per
// latent channel with one E8M0 power-of-two scale per 32 channels, stored after the engine's
// in-place quantise/dequantise round trip, which returns the input dtype and therefore stores bf16.
// The format is an explicit runtime argument, never inferred from another argument's value:
// 0 keeps the historical FP4 latent path byte-identical, 1 reads the engine window arrays, and
// 2 (CSA union) attends the engine window rows plus the causal FP4 compressed candidates in one
// softmax. Callers pass only 0, 1 or 2; any other value is poisoned at kernel entry. The format
// VALUE cannot be a static_assert because it is a runtime parameter; only the layout constants
// below are compile-time facts.
#define SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4 0u
#define SPARK_M4_MLA_WINDOW_KV_FORMAT_ENGINE_FP8 1u
#define SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION 2u
#define SPARK_M4_MLA_WINDOW_QUANT_BLOCK 32u
#define SPARK_M4_MLA_WINDOW_SCALES (SPARK_M4_MLA_LATENT / SPARK_M4_MLA_WINDOW_QUANT_BLOCK)
static_assert(SPARK_M4_MLA_LATENT % SPARK_M4_MLA_WINDOW_QUANT_BLOCK == 0u,
              "the engine window row is a whole number of 32-channel scale blocks");
static_assert(SPARK_M4_MLA_WINDOW_SCALES == 16u,
              "one E8M0 scale per 32 channels is sixteen bytes for a 512-wide row");

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

// One latent channel of one engine-format window row: the E4M3 code times its 32-channel E8M0
// power-of-two scale, in f32, rounded to bf16. The round is load-bearing: the engine's window cache
// holds bf16(dequant(code, scale)) because model.py:707 quantises in place and returns the input
// dtype, and this kernel and src/mla_ref.rs both apply the same round so they consume the same
// stored values.
__device__ __forceinline__ float spark_mla_load_window_stored(const uint8_t *window_codes,
                                                              const uint8_t *window_scales,
                                                              uint32_t row,
                                                              uint32_t index) {
    const float code = spark_fp8_e4m3_decode(window_codes[(uint64_t)row * SPARK_M4_MLA_LATENT + index]);
    const float scale = spark_ue8m0_decode(
        window_scales[(uint64_t)row * SPARK_M4_MLA_WINDOW_SCALES + index / SPARK_M4_MLA_WINDOW_QUANT_BLOCK]);
    return __bfloat162float(__float2bfloat16_rn(code * scale));
}

// Whether one selected id is attendable in the selected format. FP4 candidates are token positions
// filtered by the sliding-window predicate; engine-format candidates are row indices into the
// supplied window rows (oldest first), so any id at or past window_rows is out of window and gets
// the same excluded score, which makes its contribution exactly zero in every pass.
__device__ __forceinline__ uint32_t spark_mla_attends(uint32_t window_kv_format,
                                                      uint32_t position,
                                                      uint32_t candidate,
                                                      uint32_t window_rows) {
    return window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4
               ? spark_mla_in_window(position, candidate)
               : (candidate < window_rows ? 1u : 0u);
}

// Union format: the window rows are attended unconditionally (the caller supplies only rows at or
// behind the query, oldest first), and a selected compressed id is attended when it is not in the
// future. The reference's reachability mask is the same comparison for a ratio-1 compressed store;
// a ratio-2 store reaches this kernel as a per-position expanded view (row p = store row p/ratio),
// so an absolute candidate id is still a row index here. A candidate inside the sliding window is
// deliberately NOT filtered out: the union is the released attention's whole candidate set.
__device__ __forceinline__ uint32_t spark_mla_union_attends(uint32_t position, uint32_t candidate) {
    return candidate <= position ? 1u : 0u;
}

extern "C" __global__ void flash_mla_csa2_sm121(
    const __nv_bfloat16 *__restrict__ q,             // [heads][latent] query, rope already applied
    const uint8_t *__restrict__ kv_fp4,              // [positions][latent/2] packed E2M1 latent
    const uint8_t *__restrict__ kv_scales,           // [positions][latent/16] E4M3 scales
    const uint8_t *__restrict__ window_codes,        // engine format: [window_rows][latent] E4M3 codes
    const uint8_t *__restrict__ window_scales,       // engine format: [window_rows][latent/32] E8M0 scales
    const uint32_t window_kv_format,                 // 0 = FP4 window, 1 = engine FP8 window, 2 = CSA union
    const uint32_t window_rows,                      // engine format: window rows supplied, oldest first
    const uint32_t *__restrict__ selected,           // [selected_count] candidate position ids (2: absolute token positions)
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

    __shared__ float s_scores[SPARK_M4_MLA_MAX_ATTENDED];
    __shared__ float s_q[SPARK_M4_MLA_LATENT];
    __shared__ uint32_t s_candidates[SPARK_M4_MLA_MAX_SELECTED];
    __shared__ float s_max;
    __shared__ float s_sum;

    // The format is a runtime value, so it is guarded at run time rather than in a static_assert.
    // There is no host-side validation (cuda_harness::flash_mla_params does not check it), so this
    // guard is the contract: a caller that passes anything other than 0, 1 or 2 gets NaN out rather
    // than a half-formatted read, and the trace pair is not entered. This path is not a run: it is
    // here so that no unknown format can be silently treated as one of the real ones.
    if (window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION) {
        for (uint32_t i = lane; i < SPARK_M4_MLA_LATENT; i += SPARK_M4_MLA_THREADS) {
            out[(uint64_t)head * SPARK_M4_MLA_LATENT + i] =
                __float2bfloat16_rn(__uint_as_float(0x7FC00000u));
        }
        return;
    }
    // Union format needs room for the implicit window rows as well as the candidate list. The
    // caller supplies both; this guard is what keeps a direct launch from indexing past s_scores,
    // so the kernel refuses it the same way it refuses an unknown format.
    if (window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION &&
        (window_rows > (uint32_t)SPARK_M4_SWA_WINDOW ||
         selected_count > (uint32_t)SPARK_M4_MLA_MAX_SELECTED ||
         window_rows + selected_count > (uint32_t)SPARK_M4_MLA_MAX_ATTENDED)) {
        for (uint32_t i = lane; i < SPARK_M4_MLA_LATENT; i += SPARK_M4_MLA_THREADS) {
            out[(uint64_t)head * SPARK_M4_MLA_LATENT + i] =
                __float2bfloat16_rn(__uint_as_float(0x7FC00000u));
        }
        return;
    }

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

    // Format 2: the CSA union. The score array holds the supplied window rows first (all attended,
    // oldest first, engine format) and then the selected compressed candidates (absolute token
    // positions, attended when candidate <= position, FP4 latent). One f32 three-pass softmax runs
    // over that union: maximum, normalising sum plus the sink after the maximum, weighted values.
    // This is the released attention's concatenation (window_indices then compressed_ids) and the
    // reason a selected id older than the window must contribute. Formats 0 and 1 do not enter this
    // branch and their arithmetic below is unchanged.
    if (window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION) {
        const uint32_t union_total = window_rows + selected_count;
        const __nv_bfloat16 *q_head = q + (uint64_t)head * SPARK_M4_MLA_LATENT;
        s_q[lane] = __bfloat162float(q_head[lane]);
        s_q[lane + SPARK_M4_MLA_THREADS] = __bfloat162float(q_head[lane + SPARK_M4_MLA_THREADS]);
        for (uint32_t idx = lane; idx < selected_count; idx += SPARK_M4_MLA_THREADS) {
            s_candidates[idx] = selected[idx];
        }
        __syncthreads();
        // Pass one: every union slot's score, accumulated in ascending channel order with scale
        // decodes hoisted per scale group and the running maximum kept in a register.
        float local_max = -3.402823466e38f;
        for (uint32_t slot = lane; slot < union_total; slot += SPARK_M4_MLA_THREADS) {
            float score = 0.0f;
            if (slot < window_rows) {
                for (uint32_t i = 0; i < SPARK_M4_MLA_LATENT; ++i) {
                    score += s_q[i] *
                             spark_mla_load_window_stored(window_codes, window_scales, slot, i);
                }
                score *= SPARK_M4_MLA_SCORE_SCALE;
            } else {
                const uint32_t candidate = s_candidates[slot - window_rows];
                if (spark_mla_union_attends(position, candidate) != 0u) {
                    const uint8_t *row_fp4 =
                        kv_fp4 + (uint64_t)candidate * (SPARK_M4_MLA_LATENT / SPARK_M4_MLA_FP4_PER_BYTE);
                    const uint8_t *row_scales =
                        kv_scales + (uint64_t)candidate *
                                        (SPARK_M4_MLA_LATENT / SPARK_M4_KV_SCALE_CHANNELS);
                    for (uint32_t grp = 0u; grp < (SPARK_M4_MLA_LATENT / SPARK_M4_KV_SCALE_CHANNELS); ++grp) {
                        const float scale = spark_fp8_e4m3_decode(row_scales[grp]);
                        const uint32_t byte_base = grp * (SPARK_M4_KV_SCALE_CHANNELS / SPARK_M4_MLA_FP4_PER_BYTE);
                        const uint32_t ch_base = grp * SPARK_M4_KV_SCALE_CHANNELS;
#pragma unroll
                        for (uint32_t b = 0u; b < (SPARK_M4_KV_SCALE_CHANNELS / SPARK_M4_MLA_FP4_PER_BYTE); ++b) {
                            const uint8_t packed = row_fp4[byte_base + b];
                            const uint32_t i0 = ch_base + 2u * b;
                            const float k0 = spark_fp4_e2m1_decode((uint8_t)(packed & 0x0Fu)) * scale;
                            score += s_q[i0] * k0;
                            const float k1 = spark_fp4_e2m1_decode((uint8_t)(packed >> 4)) * scale;
                            score += s_q[i0 + 1u] * k1;
                        }
                    }
                    score *= SPARK_M4_MLA_SCORE_SCALE;
                } else {
                    score = -3.402823466e38f;
                }
            }
            s_scores[slot] = score;
            local_max = score > local_max ? score : local_max;
        }
        local_max = spark_block_max(local_max, lane, SPARK_M4_MLA_THREADS);
        if (lane == 0u) {
            s_max = local_max;
        }
        __syncthreads();
        // The sink is a denominator term only, added after the maximum as in the released order.
        float local_sum = lane == 0u ? spark_exp(sink[head] - s_max) : 0.0f;
        for (uint32_t slot = lane; slot < union_total; slot += SPARK_M4_MLA_THREADS) {
            const float probability = spark_exp(s_scores[slot] - s_max);
            s_scores[slot] = probability;
            local_sum += probability;
        }
        local_sum = spark_block_sum(local_sum, lane, SPARK_M4_MLA_THREADS);
        if (lane == 0u) {
            s_sum = local_sum > 0.0f ? local_sum : 1.0f;
        }
        __syncthreads();
        // Each thread owns two latent channels (lane, lane+256) and walks the union once in
        // ascending slot order: window rows first, then the selected list as supplied.
        const uint32_t i0 = lane;
        const uint32_t i1 = lane + SPARK_M4_MLA_THREADS;
        const uint32_t w_scale_off0 = i0 / SPARK_M4_MLA_WINDOW_QUANT_BLOCK;
        const uint32_t w_scale_off1 = i1 / SPARK_M4_MLA_WINDOW_QUANT_BLOCK;
        const uint32_t fp4_byte_off0 = i0 / SPARK_M4_MLA_FP4_PER_BYTE;
        const uint32_t fp4_byte_off1 = i1 / SPARK_M4_MLA_FP4_PER_BYTE;
        const uint32_t fp4_shift = (i0 % SPARK_M4_MLA_FP4_PER_BYTE) * 4u;
        const uint32_t kv_scale_off0 = i0 / SPARK_M4_KV_SCALE_CHANNELS;
        const uint32_t kv_scale_off1 = i1 / SPARK_M4_KV_SCALE_CHANNELS;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t slot = 0u; slot < window_rows; ++slot) {
            const float prob = s_scores[slot];
            const uint8_t *row_codes = window_codes + (uint64_t)slot * SPARK_M4_MLA_LATENT;
            const uint8_t *row_scales = window_scales + (uint64_t)slot * SPARK_M4_MLA_WINDOW_SCALES;
            const float v0 = __bfloat162float(__float2bfloat16_rn(
                spark_fp8_e4m3_decode(row_codes[i0]) * spark_ue8m0_decode(row_scales[w_scale_off0])));
            const float v1 = __bfloat162float(__float2bfloat16_rn(
                spark_fp8_e4m3_decode(row_codes[i1]) * spark_ue8m0_decode(row_scales[w_scale_off1])));
            acc0 += prob * v0;
            acc1 += prob * v1;
        }
        for (uint32_t idx = 0u; idx < selected_count; ++idx) {
            const uint32_t candidate = s_candidates[idx];
            if (spark_mla_union_attends(position, candidate) != 0u) {
                const float prob = s_scores[window_rows + idx];
                const uint8_t *row_fp4 =
                    kv_fp4 + (uint64_t)candidate * (SPARK_M4_MLA_LATENT / SPARK_M4_MLA_FP4_PER_BYTE);
                const uint8_t *row_scales =
                    kv_scales + (uint64_t)candidate * (SPARK_M4_MLA_LATENT / SPARK_M4_KV_SCALE_CHANNELS);
                const uint8_t nib0 = (uint8_t)((row_fp4[fp4_byte_off0] >> fp4_shift) & 0x0Fu);
                const uint8_t nib1 = (uint8_t)((row_fp4[fp4_byte_off1] >> fp4_shift) & 0x0Fu);
                const float v0 = spark_fp4_e2m1_decode(nib0) * spark_fp8_e4m3_decode(row_scales[kv_scale_off0]);
                const float v1 = spark_fp4_e2m1_decode(nib1) * spark_fp8_e4m3_decode(row_scales[kv_scale_off1]);
                acc0 += prob * v0;
                acc1 += prob * v1;
            }
        }
        out[(uint64_t)head * SPARK_M4_MLA_LATENT + i0] = __float2bfloat16_rn(acc0 / s_sum);
        out[(uint64_t)head * SPARK_M4_MLA_LATENT + i1] = __float2bfloat16_rn(acc1 / s_sum);
        if (lane == 0u) {
            const uint32_t meta = (uint32_t)(SPARK_M4_MLA_LATENT * 2u);
            SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_CSA2_ATTN, layer_id, meta, t_enter);
            (void)heads;
            (void)step_seq;
        }
        (void)t_enter;
        return;
    }

    // Stage 2: attention over the selected candidates. Pass one computes every score and the
    // running maximum, pass two the normalising sum, pass three the weighted latent sum. The
    // candidate count is bounded by SPARK_M4_MLA_MAX_SELECTED; the caller guarantees that.
    for (uint32_t slot = lane; slot < selected_count; slot += SPARK_M4_MLA_THREADS) {
        const uint32_t candidate = selected[slot];
        float score = 0.0f;
        if (spark_mla_attends(window_kv_format, position, candidate, window_rows) != 0u) {
            for (uint32_t i = 0; i < SPARK_M4_MLA_LATENT; ++i) {
                const float key = window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4
                                      ? spark_mla_load_latent(
                                            kv_fp4 + (uint64_t)candidate * (SPARK_M4_MLA_LATENT / 2u), i) *
                                            spark_mla_scale(
                                                kv_scales + (uint64_t)candidate *
                                                                (SPARK_M4_MLA_LATENT / SPARK_M4_KV_SCALE_CHANNELS),
                                                i)
                                      : spark_mla_load_window_stored(window_codes, window_scales, candidate, i);
                score += __bfloat162float(q[(uint64_t)head * SPARK_M4_MLA_LATENT + i]) * key;
            }
            // The MLA score scale, 1/sqrt(512) over the latent width. Specification 2.3 fixes the
            // selection rule but not this constant, so it is named once in m4_common_sm121.cuh and
            // src/mla_ref.rs carries the same literal, compared bit for bit by a test there.
            score *= SPARK_M4_MLA_SCORE_SCALE;
        } else {
            // Outside the format's window: excluded by score, not merely down-weighted. In engine
            // format an id at or past window_rows gets this exact value too, so its contribution is
            // exactly zero in pass three; that zeroing is a requirement and is tested by injection.
            score = -3.402823466e38f;
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
            if (spark_mla_attends(window_kv_format, position, candidate, window_rows) != 0u) {
                const float value = window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4
                                        ? spark_mla_load_latent(
                                              kv_fp4 + (uint64_t)candidate * (SPARK_M4_MLA_LATENT / 2u), i) *
                                              spark_mla_scale(
                                                  kv_scales + (uint64_t)candidate *
                                                                  (SPARK_M4_MLA_LATENT / SPARK_M4_KV_SCALE_CHANNELS),
                                                  i)
                                        : spark_mla_load_window_stored(window_codes, window_scales, candidate, i);
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

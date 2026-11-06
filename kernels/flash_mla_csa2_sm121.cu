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
// Verification status: executed on spark-0 (GB10, sm_121a) on 2026-10-17. nvcc 13.0.3
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
// docs/archive/build/phase3-flash-mla.md holds the measured numbers). That execution does not turn the
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
#include "flash_mla_csa2_common_sm121.cuh"
static_assert(SPARK_M4_MLA_LATENT % SPARK_M4_MLA_WINDOW_QUANT_BLOCK == 0u,
              "the engine window row is a whole number of 32-channel scale blocks");
static_assert(SPARK_M4_MLA_WINDOW_SCALES == 16u,
              "one E8M0 scale per 32 channels is sixteen bytes for a 512-wide row");

// A position is attendable when it is inside the sliding window behind the query. The predicate
// is a function of the query position and the candidate id, so the CPU reference can test it
// directly instead of duplicating the comparison.


// The E4M3 scale that covers latent channel index: one per 16 channels, stored as bytes.

// One latent channel of one engine-format window row: the E4M3 code times its 32-channel E8M0
// power-of-two scale, in f32, rounded to bf16. The round is load-bearing: the engine's window cache
// holds bf16(dequant(code, scale)) because model.py:707 quantises in place and returns the input
// dtype, and this kernel and src/mla_ref.rs both apply the same round so they consume the same
// stored values.

// Whether one selected id is attendable in the selected format. FP4 candidates are token positions
// filtered by the sliding-window predicate; engine-format candidates are row indices into the
// supplied window rows (oldest first), so any id at or past window_rows is out of window and gets
// the same excluded score, which makes its contribution exactly zero in every pass.

// Union format: the window rows are attended unconditionally (the caller supplies only rows at or
// behind the query, oldest first), and a selected compressed id is attended when it is not in the
// future. The reference's reachability mask is the same comparison for a ratio-1 compressed store;
// a ratio-2 store reaches this kernel as a per-position expanded view (row p = store row p/ratio),
// so an absolute candidate id is still a row index here. A candidate inside the sliding window is
// deliberately NOT filtered out: the union is the released attention's whole candidate set.

extern "C" __global__ void flash_mla_csa2_sm121(
    const __nv_bfloat16 *__restrict__ q,
    const uint8_t *__restrict__ kv_fp4,
    const uint8_t *__restrict__ kv_scales,
    const uint8_t *__restrict__ window_codes,
    const uint8_t *__restrict__ window_scales,
    const uint32_t window_kv_format,
    const uint32_t window_rows,
    const uint32_t *__restrict__ selected,
    const __nv_bfloat16 *__restrict__ index_key,
    const __nv_bfloat16 *__restrict__ index_query,
    const float *__restrict__ sink,
    float *__restrict__ block_scores,
    uint32_t *__restrict__ block_best,
    __nv_bfloat16 *__restrict__ out,
    const uint32_t position,
    const uint32_t selected_count,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    spark_mla_attention_body<SparkMlaBase>(
        q, kv_fp4, kv_scales, window_codes, window_scales, window_kv_format, window_rows,
        (uint32_t)SPARK_M4_SWA_WINDOW, 1u, selected, index_key, index_query, sink, block_scores,
        block_best, out, position, selected_count, trace_ring, layer_id, step_seq);
}

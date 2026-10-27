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

// crates/sparkflash-kernels/pending_kernels/routed_gate_up_swiglu_splitk_sm121.cu
//
// PENDING SOURCE - NOT YET IN kernels/ AND NOT YET PINNED.
//
// This file is the split-K / grouped gate+up kernel of task D51-A (docs/DECISIONS.md D51, m2
// execution plan lever "merge the two gate_up launches" / "fuse the weighted SwiGLU + quantise").
// It lives in the kernels crate while it is unpinned so that the CPU twin, the ABI pins and the
// tolerance harness can be reviewed and tested WITHOUT a CUDA host; the pin job
// (scripts/regen-kernel-baseline.sh + the four-place list in the crate's
// routed_gate_up_swiglu_splitk_abi.rs PIN_CHECKLIST) is what moves it to kernels/, adds it to
// build.rs PROBED_KERNELS, the baseline manifest and the check script's kernel list, and records
// the cubin bytes/sha256. Do not rename it in place: the pin names the file.
//
// What it replaces, and why. At HEAD the routed experts' gate/up path is 12 launches per layer:
// 8 x routed_gate_up_sm121 (w1 and w3, four ranks) plus 4 x swiglu_mxfp8_quant_weighted_sm121.
// This kernel is ONE launch per rank: it reads the token's activation row once, walks all
// SPARK_M4_MOE_TOP_K selected experts for that token (grouped GEMV: the routing gather happens
// inside the CTA, one CTA per (token, 32-column block)), fuses w1 and w3, and runs the weighted
// SwiGLU + the activation MX quantiser in the epilogue. 12 -> 4 launches per layer (480 -> 160 per
// decode position), and the two intermediate bf16 [6][640] gate/up tensors are never written.
//
// ARITHMETIC CONTRACT (D51 amendment A). The quantisation of D39 rule 1 is unchanged: E4M3
// activations, packed E2M1 weights, ue8m0 scales, f32 accumulation, the same SwiGLU, the same
// bf16 product rounding and the same activation quantiser. What changes is the reduction ORDER of
// the f32 K-sum, which D51 permits behind a measured per-kernel tolerance against the bit-exact
// twin (crates/sparkflash-kernels/src/routed_gate_up_swiglu_splitk_ref.rs is the twin arm and the
// tolerance record; the pinned kernels are untouched and stay the reference arm).
//
// The order, stated exactly:
//   * K is 160 32-value MX blocks. Warp w of the CTA owns the shard {b : b % 8 == w}: 20 blocks,
//     folded serially in ASCENDING block order into that shard's accumulator
//     (block = w + 8 * j, j = 0..19).
//   * The eight shard totals are combined with a three-level warp-shuffle butterfly
//     (__shfl_xor_sync offsets 1, 2, 4) whose value is
//         ((t0 + t1) + (t2 + t3)) + ((t4 + t5) + (t6 + t7)),
//     every add a single __fadd_rn. That is a reassociation of the twin's serial 160-term fold.
//   * Each block term is the twin's: (32-value f32 dot) * act_scale * w_scale, with the two
//     scale multiplies after the dot and each a separate __fmul_rn (no contraction, no FMA).
//   * The epilogue keeps the chain's rounding points: each projection is rounded to bf16 once
//     (the intermediate tensor the incumbent chain stores is bf16, so fusing must not skip it),
//     then SwiGLU (asymmetric clamp, limit), then ONE bf16 rounding of the routing-weighted
//     product, then the released activation quantiser (amax over the 32 columns, ue8m0 ceil scale,
//     E4M3 encode). Columns 576..639 are the ABI's padding and quantise as zeros, exactly as the
//     incumbent chain's zero-filled gate/up tail does.
//
// GEOMETRY. grid = (tokens, SPARK_SPLITK_TILES = 20) x 256 threads. One CTA owns one token's 32
// output columns for ALL six experts and BOTH projections: 6 * 2 * 32 weight rows of K = 5120.
// The activation row (5120 E4M3 bytes + 160 ue8m0 bytes) is staged into shared memory once per
// CTA; every expert and both projections reuse it, which is the grouping's reason to exist (the
// incumbent re-reads it in eight separate launches). A 32-column tile is exactly one MX block, so
// the epilogue's block amax is a single warp reduction over the 32 lanes.
//
// HONEST LIMITATION, stated in the source rather than discovered on the device: 20 CTAs per decode
// token is a launch-count fusion, not a bandwidth-saturating geometry (20 of ~148 SMs). The GPU
// verification plan (routed_gate_up_swiglu_splitk_abi.rs::GPU_VERIFICATION_PLAN) makes the
// occupancy sweep - SPLITK_TILES per CTA, i.e. 1, 2 or 4 MX blocks per CTA, and a second grid axis
// over expert slots - the first device experiment, and the harness reports achieved GB/s against
// the 273 GB/s per-node floor instead of a time alone.
//
// TRACE. TRACE_SCOPE_MOE_MXFP4, the routed-expert consumer domain the incumbent gate/up kernel and
// the combine kernel record under; the shipped cubin is the trace-disabled build, so the harness
// passes a null ring and the macros are no-ops. payload_meta counts the bytes the reporting block
// wrote: 6 slots * (32 codes + 1 scale).

#include "m4_common_sm121.cuh"
#include "m4_activation_quant_sm121.cuh"

#define SPARK_SPLITK_TILE 32u
#define SPARK_SPLITK_TILES (SPARK_M4_MOE_WIDTH_PADDED / SPARK_SPLITK_TILE)
#define SPARK_SPLITK_WARPS 8u
#define SPARK_SPLITK_THREADS (SPARK_SPLITK_WARPS * 32u)
#define SPARK_SPLITK_K_BLOCKS (SPARK_M4_HIDDEN / SPARK_M4_QUANT_BLOCK)
#define SPARK_SPLITK_BLOCKS_PER_SHARD (SPARK_SPLITK_K_BLOCKS / SPARK_SPLITK_WARPS)
#define SPARK_SPLITK_SHARDS SPARK_SPLITK_WARPS
#define SPARK_SPLITK_COLS_PER_GROUP (SPARK_SPLITK_TILE / SPARK_SPLITK_WARPS)
#define SPARK_SPLITK_ROWS (SPARK_M4_MOE_TOP_K * 2u)
#define SPARK_SPLITK_PACKED_ROW (SPARK_M4_HIDDEN / 2u)
#define SPARK_SPLITK_SCALE_ROW (SPARK_M4_HIDDEN / SPARK_M4_QUANT_BLOCK)
#define SPARK_SPLITK_OUT_SCALE_ROW (SPARK_M4_MOE_WIDTH_PADDED / SPARK_M4_QUANT_BLOCK)
#define SPARK_SPLITK_TRACE_META (SPARK_M4_MOE_TOP_K * (SPARK_SPLITK_TILE + 1u))

static_assert(SPARK_M4_HIDDEN % SPARK_M4_QUANT_BLOCK == 0u, "the hidden size tiles by 32");
static_assert(SPARK_SPLITK_K_BLOCKS == 160u, "5120 values are 160 MX blocks");
static_assert(SPARK_SPLITK_BLOCKS_PER_SHARD == 20u, "160 blocks split over eight warps");
static_assert(SPARK_SPLITK_K_BLOCKS == SPARK_SPLITK_BLOCKS_PER_SHARD * SPARK_SPLITK_WARPS,
              "the K shards cover every block exactly once");
static_assert(SPARK_SPLITK_TILES * SPARK_SPLITK_TILE == SPARK_M4_MOE_WIDTH_PADDED,
              "the tile grid covers the padded routed width exactly");
static_assert(SPARK_SPLITK_COLS_PER_GROUP * SPARK_SPLITK_WARPS == SPARK_SPLITK_TILE,
              "the fold's warp groups cover the 32-column tile");
static_assert(SPARK_M4_MOE_WIDTH % SPARK_SPLITK_TILE == 0u, "the real width tiles by 32");
static_assert(SPARK_SPLITK_THREADS == 256u, "the block is eight warps");
static_assert(SPARK_SPLITK_ROWS == 12u, "six slots, gate and up");
static_assert(SPARK_SPLITK_PACKED_ROW == 2560u, "one 5120-wide FP4 row is 2560 bytes");
static_assert(SPARK_SPLITK_SCALE_ROW == 160u, "one 5120-wide row has 160 ue8m0 scales");
static_assert(SPARK_SPLITK_OUT_SCALE_ROW == 20u, "one 640-wide row has 20 ue8m0 scales");
static_assert(SPARK_M4_MOE_TOP_K == 6u, "six routed slots per token");

// The E2M1 value at one index of a packed weight row: low nibble even, high nibble odd. The
// FP4 -> FP8 -> f32 round trip is the identity on all sixteen codes and is kept because it is
// what the reference does (the twin enumerates the codes rather than assuming the identity).
__device__ __forceinline__ float spark_splitk_fp4_load(const uint8_t *row, uint32_t index) {
    const uint8_t packed = row[index / 2u];
    const uint8_t nibble = (index % 2u) == 0u ? (uint8_t)(packed & 0x0Fu) : (uint8_t)(packed >> 4);
    return spark_fp4_e2m1_decode(nibble);
}

// One 32-value MX block's scaled contribution to one output element, in the twin's order: the f32
// dot with one rounding per multiply/add pair, then the two power-of-two scales on the PARTIAL.
__device__ __forceinline__ float spark_splitk_block_term(const uint8_t *act_block,
                                                         const uint8_t *w_row,
                                                         uint32_t base,
                                                         float act_scale,
                                                         float w_scale) {
    float partial = 0.0f;
    for (uint32_t i = 0; i < SPARK_M4_QUANT_BLOCK; ++i) {
        const uint32_t index = base + i;
        const float act_value = spark_fp8_e4m3_decode(act_block[index]);
        const float w_value =
            spark_fp8_e4m3_decode(spark_fp8_e4m3_encode(spark_splitk_fp4_load(w_row, index)));
        partial = __fadd_rn(partial, __fmul_rn(act_value, w_value));
    }
    return __fmul_rn(__fmul_rn(partial, act_scale), w_scale);
}

extern "C" __global__ void routed_gate_up_swiglu_splitk_sm121(
    const uint8_t *__restrict__ act_fp8,       // [tokens][hidden] E4M3, one byte per value
    const uint8_t *__restrict__ act_scales,    // [tokens][hidden / 32] ue8m0
    const uint8_t *__restrict__ w1_fp4,        // [experts][width][hidden / 2] packed E2M1
    const uint8_t *__restrict__ w1_scales,     // [experts][width][hidden / 32] ue8m0
    const uint8_t *__restrict__ w3_fp4,        // [experts][width][hidden / 2] packed E2M1
    const uint8_t *__restrict__ w3_scales,     // [experts][width][hidden / 32] ue8m0
    const uint32_t *__restrict__ expert_ids,   // [tokens * top_k] the routing selection
    const float *__restrict__ routing_weights, // [tokens * top_k] the router's weights
    uint8_t *__restrict__ out_fp8,             // [tokens * top_k][width_padded] E4M3
    uint8_t *__restrict__ out_scales,          // [tokens * top_k][width_padded / 32] ue8m0
    const uint32_t tokens,
    const float limit,                         // swiglu_limit, 10.0 in the released config
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t token = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const bool live_token = token < tokens;
    const uint32_t column = blockIdx.y * SPARK_SPLITK_TILE + lane;
    const bool live_column = column < (uint32_t)SPARK_M4_MOE_WIDTH;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_MOE_MXFP4, t_enter);

    // The token's activation row is shared by all six experts and both projections, so it is
    // staged once per CTA; the incumbent re-reads it in eight separate launches per layer.
    __shared__ uint8_t act_s[SPARK_M4_HIDDEN];
    __shared__ uint8_t act_scale_s[SPARK_SPLITK_SCALE_ROW];
    // One accumulator per (shard, row, column): 8 * 12 * 32 f32 = 12 KiB.
    __shared__ float shard_acc[SPARK_SPLITK_SHARDS][SPARK_SPLITK_ROWS][SPARK_SPLITK_TILE];
    // The shuffle fold's result: 12 * 32 f32 = 1.5 KiB.
    __shared__ float folded[SPARK_SPLITK_ROWS][SPARK_SPLITK_TILE];

    for (uint32_t i = tid; i < SPARK_M4_HIDDEN; i += SPARK_SPLITK_THREADS) {
        act_s[i] = live_token ? act_fp8[(uint64_t)token * SPARK_M4_HIDDEN + (uint64_t)i]
                              : (uint8_t)0u;
    }
    for (uint32_t i = tid; i < SPARK_SPLITK_SCALE_ROW; i += SPARK_SPLITK_THREADS) {
        act_scale_s[i] =
            live_token ? act_scales[(uint64_t)token * SPARK_SPLITK_SCALE_ROW + (uint64_t)i]
                       : (uint8_t)0u;
    }
    __syncthreads();

    // This warp's K shard, folded serially in ascending block order. block = warp + 8 * j, so the
    // shard's 20 terms are visited in ascending block order exactly once.
    float acc[SPARK_SPLITK_ROWS];
#pragma unroll
    for (uint32_t r = 0; r < SPARK_SPLITK_ROWS; ++r) {
        acc[r] = 0.0f;
    }
    if (live_token && live_column) {
        for (uint32_t slot = 0; slot < (uint32_t)SPARK_M4_MOE_TOP_K; ++slot) {
            const uint32_t expert = expert_ids[(uint64_t)token * SPARK_M4_MOE_TOP_K + slot];
            const uint32_t weight_row = expert * (uint32_t)SPARK_M4_MOE_WIDTH + column;
            const uint8_t *w1_row = w1_fp4 + (uint64_t)weight_row * SPARK_SPLITK_PACKED_ROW;
            const uint8_t *s1_row = w1_scales + (uint64_t)weight_row * SPARK_SPLITK_SCALE_ROW;
            const uint8_t *w3_row = w3_fp4 + (uint64_t)weight_row * SPARK_SPLITK_PACKED_ROW;
            const uint8_t *s3_row = w3_scales + (uint64_t)weight_row * SPARK_SPLITK_SCALE_ROW;
            for (uint32_t j = 0; j < SPARK_SPLITK_BLOCKS_PER_SHARD; ++j) {
                const uint32_t block = warp + SPARK_SPLITK_SHARDS * j;
                const uint32_t base = block * (uint32_t)SPARK_M4_QUANT_BLOCK;
                float act_scale = spark_ue8m0_decode(act_scale_s[block]);
                acc[slot * 2u] = __fadd_rn(
                    acc[slot * 2u],
                    spark_splitk_block_term(act_s, w1_row, base, act_scale,
                                            spark_ue8m0_decode(s1_row[block])));
                acc[slot * 2u + 1u] = __fadd_rn(
                    acc[slot * 2u + 1u],
                    spark_splitk_block_term(act_s, w3_row, base, act_scale,
                                            spark_ue8m0_decode(s3_row[block])));
            }
        }
    }
#pragma unroll
    for (uint32_t r = 0; r < SPARK_SPLITK_ROWS; ++r) {
        shard_acc[warp][r][lane] = acc[r];
    }
    __syncthreads();

    // The split-K reduction across warps, as a warp-shuffle butterfly. Warp g owns columns
    // [4g, 4g+4); lane l holds column 4g + (l >> 3) and shard (l & 7). Offsets 1, 2, 4 therefore
    // fold exactly the eight shard totals of one column, giving
    // ((t0 + t1) + (t2 + t3)) + ((t4 + t5) + (t6 + t7)) in every lane of the group.
    const uint32_t fold_column = warp * SPARK_SPLITK_COLS_PER_GROUP + (lane >> 3u);
    const uint32_t fold_shard = lane & 7u;
    float value[SPARK_SPLITK_ROWS];
#pragma unroll
    for (uint32_t r = 0; r < SPARK_SPLITK_ROWS; ++r) {
        value[r] = shard_acc[fold_shard][r][fold_column];
    }
#pragma unroll
    for (uint32_t offset = 1u; offset < SPARK_SPLITK_SHARDS; offset <<= 1u) {
#pragma unroll
        for (uint32_t r = 0; r < SPARK_SPLITK_ROWS; ++r) {
            value[r] = __fadd_rn(value[r], __shfl_xor_sync(0xFFFFFFFFu, value[r], offset));
        }
    }
    if (fold_shard == 0u) {
#pragma unroll
        for (uint32_t r = 0; r < SPARK_SPLITK_ROWS; ++r) {
            folded[r][fold_column] = value[r];
        }
    }
    __syncthreads();

    // The fused epilogue: one warp owns the 32-column tile, which is exactly one MX block, so the
    // block amax is one warp-shuffle reduction - the same shape swiglu_mxfp8_quant_weighted_sm121
    // uses, and the same quantiser rule.
    if (warp == 0u) {
        const uint32_t out_column = column;
        for (uint32_t slot = 0; slot < (uint32_t)SPARK_M4_MOE_TOP_K; ++slot) {
            const uint32_t row = token * (uint32_t)SPARK_M4_MOE_TOP_K + slot;
            // The chain stores each projection as bf16 before the SwiGLU reads it; fusing must not
            // drop that rounding, so it happens here on the f32 accumulator.
            float gate = live_column ? folded[slot * 2u][lane] : 0.0f;
            float up = live_column ? folded[slot * 2u + 1u][lane] : 0.0f;
            gate = __bfloat162float(__float2bfloat16_rn(gate));
            up = __bfloat162float(__float2bfloat16_rn(up));
            // The reference's asymmetric clamp (gate upper side only, up both sides).
            const float clamped_gate = fminf(gate, limit);
            const float clamped_up = fminf(fmaxf(up, -limit), limit);
            const float activated = (clamped_gate / (1.0f + __expf(-clamped_gate))) * clamped_up;
            // The routing weight multiplies BEFORE the quantiser, and the product is rounded to
            // bf16 once (model.py:849-851; the weighted epilogue kernel's order).
            const float weighted =
                __bfloat162float(__float2bfloat16_rn(activated * routing_weights[row]));
            float amax = fabsf(weighted);
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, offset));
            }
            const uint8_t scale = spark_act_quant_scale_code(amax);
            if (lane == 0u) {
                out_scales[(uint64_t)row * SPARK_SPLITK_OUT_SCALE_ROW + (uint64_t)blockIdx.y] =
                    scale;
            }
            out_fp8[(uint64_t)row * SPARK_M4_MOE_WIDTH_PADDED + (uint64_t)out_column] =
                spark_fp8_e4m3_encode(weighted / spark_ue8m0_decode(scale));
        }
    }
#if defined(SPARKFLASH_ENABLE_GPU_TRACE)
    __syncthreads();
#endif
    if (tid == 0u) {
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_MOE_MXFP4, layer_id, SPARK_SPLITK_TRACE_META,
                             t_enter);
        (void)step_seq;
    }
    (void)t_enter;
    (void)live_token;
}

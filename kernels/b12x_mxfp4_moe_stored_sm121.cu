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

// kernels/b12x_mxfp4_moe_stored_sm121.cu
//
// Attribution: The "b12x" naming prefix and SM120/SM121 block-scaled MXFP8/MXFP4 micro-tile
// layout concepts originate from b12x (https://github.com/local-inference-lab/b12x, now in
// FlashInfer under flashinfer/experimental/b12x, Apache-2.0; Luke Alonso @lukealonso,
// @original-el8, and contributors) and NVIDIA Flare
// (https://github.com/NVIDIA/flare-local-inference, Apache-2.0). See the root NOTICE file.
//
// The stored-layout variant of the routed-expert down projection: the same tensor-parallel rank
// partial GEMM as b12x_mxfp4_moe_sm121 (specification 2.1/6.2, decision D39), reading the released
// plan's real 576-wide rows instead of the A8 chain's 640-padded rows. This file is a NEW symbol
// rather than a runtime layout flag: the incumbent kernel and every pin of the A8 path stay exactly
// as they are, and the chain selects one cubin per bound plan layout before the first launch.
//
// Why a second kernel exists at all: the released per-rank plan stores each expert's w2 at its real
// width (576 K values per rank: config.json moe_inter_dim 2304 / TP 4), so one output row is 288
// packed FP4 code bytes and 18 ue8m0 scales. The incumbent kernel's row stride is 320 packed bytes
// and a 20-byte scale grid, so launching it on the plan's arena would misindex every row after the
// first. Materialising the padded rows at boot was measured and rejected (+25.2 GiB/rank on a 121 GiB
// node); reading the stored rows directly is the only other option that keeps zero per-token H2D.
//
// WHAT THIS KERNEL READS, EXACTLY (confirmed against the released plan before implementing - the
// design page told the implementer to check rather than assume, and the check changed the row
// contract):
//
//   w_fp4     [experts][5120][288] packed E2M1 bytes, row stride 288 = 576 / FP4_PER_BYTE
//   w_scales  [experts][5120][ 18] ue8m0 bytes,        row stride  18 = 576 / QUANT_BLOCK
//
// The two tensors are SEPARATE, not interleaved into one 306-byte row. The released plan's shards are
// layers.N.ffn.experts.E.w2.weight as [5120, 288] and its own w2.scale as [5120, 18] (92,160 B),
// and the rank's arena slices are those two contiguous buffers; the plan's shape test in
// crates/sparkflash-cli/src/resident_weights.rs pins exactly that. 288 + 18 = 306 is therefore the
// COST of one output row across the two tensors, and it is not a stride: the scale grid advances 18
// bytes per output row, one byte per 32-value block. A loader that emitted an interleaved 306-byte
// row would be rejected by src/mxfp4_ref.rs::check_stored_moe_layout.
//
// The A operand is UNCHANGED from the A8 path, because the chain's quantiser produces it before the
// weight layout is known: act_fp8 is [tokens * top_k][640] E4M3 and act_scales is
// [tokens * top_k][20] ue8m0, the same operand b12x_mxfp4_moe_sm121 reads. This kernel walks the 18
// REAL blocks only, which is what makes the layout switch a weight-pointer switch in the chain - an
// activation re-staging would be a second, silent ABI change. It never addresses columns 576..639,
// so it needs no padding mask; the incumbent's mask has nothing to mask here.
//
// Arithmetic and exactness (the contract the occupancy rewrite established, implemented here in full
// so this file does not depend on that branch being merged):
//
//   t_b = __fmul_rn(__fmul_rn(partial_b, act_scale_b), w_scale_b)  for b = 0..17 ascending,
//   acc = __fadd_rn(acc, t_b) from +0.0, then the unrounded f32 add into the rank slot accumulator.
//
// The 32-value partial is the reference's FP8xFP8 block product: the FP4 weight decodes through the
// exact E2M1 -> E4M3 round trip and the two ue8m0 scales (powers of two) are applied to the PARTIAL
// PRODUCT, exactly as kernel.py:552-553. One f32 term per 32-value block, folded in ascending block
// order, is the reference's only ordered arithmetic, so the split-K schedule below is bit-identical
// to a serial 18-block loop by construction: the block partials are independent, and the reducer adds
// them in the same ascending order the serial loop would.
//
// This variant CANNOT be bit-compared against the incumbent on the same weight bytes, and does not
// claim to be: the two kernels read different K widths (576 vs 640 padded with two masked blocks).
// Its exactness claim is stated against the host twin on the SAME two-tensor stored layout
// (src/mxfp4_ref.rs::moe_combine_rank_stored), and the de-interleave test proves the stride handling
// by rebuilding the padded rows (append 64 zero code bytes and 4 zero scales per row) and requiring
// the padded twin and the stored twin to produce bit-identical f32 accumulators and finalised rows.
//
// Schedule (the occupancy stage's shape): 32 output columns x 8 parallel K groups = 256 threads per
// block, one reducer thread per (slot, column), one block per (token, 32-column hidden tile), grid
// (tokens, 160). Shared memory: float terms[top_k][18][32] = 13,824 B. The 18 blocks split as
// block = group, group + 8, group + 16, so each group owns 2 or 3 blocks.
//
// Verification status: the cubin is pinned in baseline/kernel_baseline.txt and the declaration in
// src/mxfp4_moe_stored_abi.rs; the host twin and the stride test live in src/mxfp4_ref.rs; the
// device harness is examples/m25_moe_stored.rs and has NOT run yet (no GPU window in this session),
// so no device claim, no timing and no chain wiring is made here. The router is still host-side and
// the shared expert is still host arithmetic; nothing in this file changes that.

#include "m4_common_sm121.cuh"

// ---------------------------------------------------------------------------------------
// Stored 576-K row layout contract
// ---------------------------------------------------------------------------------------
// The per-rank real width and its two row strides. Every number is derived from the shared header's
// model geometry, so a geometry change fails the build instead of silently changing the layout.
#define SPARK_M4_STORED_K_VALUES SPARK_M4_MOE_WIDTH
#define SPARK_M4_STORED_FP4_PER_BYTE 2u
#define SPARK_M4_STORED_CODE_ROW (SPARK_M4_STORED_K_VALUES / SPARK_M4_STORED_FP4_PER_BYTE)
#define SPARK_M4_STORED_SCALE_ROW (SPARK_M4_STORED_K_VALUES / SPARK_M4_QUANT_BLOCK)
// The cost of one output row across the two tensors. NOT a stride: the tensors are separate.
#define SPARK_M4_STORED_ROW_TOTAL (SPARK_M4_STORED_CODE_ROW + SPARK_M4_STORED_SCALE_ROW)
#define SPARK_M4_STORED_K_BLOCKS SPARK_M4_STORED_SCALE_ROW

// The A operand is the A8 chain layout, unchanged: 640 E4M3 bytes (one per padded width value) and a
// 20-byte ue8m0 scale grid per routed row. Only blocks 0..17 are read.
#define SPARK_M4_STORED_ACT_ROW SPARK_M4_MOE_WIDTH_PADDED
#define SPARK_M4_STORED_ACT_SCALE_ROW (SPARK_M4_MOE_WIDTH_PADDED / SPARK_M4_QUANT_BLOCK)

// One block per (token, 32-value hidden tile): 32 output columns x 8 parallel K groups.
#define SPARK_M4_STORED_TILE_D 32u
#define SPARK_M4_STORED_K_GROUPS 8u
#define SPARK_M4_STORED_THREADS (SPARK_M4_STORED_TILE_D * SPARK_M4_STORED_K_GROUPS)
// Hidden tiles: 5120 / 32 = 160 blocks along grid.y.
#define SPARK_M4_STORED_GRID_Y (SPARK_M4_HIDDEN / SPARK_M4_STORED_TILE_D)

static_assert(SPARK_M4_STORED_K_VALUES % SPARK_M4_QUANT_BLOCK == 0u,
              "the stored rank width tiles by 32-value MX blocks");
static_assert(SPARK_M4_STORED_K_BLOCKS == 18u, "576 real values are 18 MX blocks");
static_assert(SPARK_M4_STORED_CODE_ROW == 288u, "576 FP4 values pack to 288 code bytes");
static_assert(SPARK_M4_STORED_CODE_ROW * SPARK_M4_STORED_FP4_PER_BYTE == SPARK_M4_STORED_K_VALUES,
              "the code row is exactly one byte per two K values");
static_assert(SPARK_M4_STORED_SCALE_ROW == 18u, "the stored scale row is 18 ue8m0 bytes");
static_assert(SPARK_M4_STORED_ROW_TOTAL == 306u,
              "one stored output row costs 288 code + 18 scale bytes across the two tensors");
static_assert(SPARK_M4_STORED_ACT_ROW == 640u,
              "the A operand is the A8 chain layout: 640 E4M3 bytes per routed row");
static_assert(SPARK_M4_STORED_ACT_SCALE_ROW == 20u,
              "the A scale grid is 20 ue8m0 bytes per routed row");
static_assert(SPARK_M4_STORED_K_BLOCKS <= SPARK_M4_STORED_ACT_SCALE_ROW,
              "the kernel reads only the real blocks of the padded A scale grid");
static_assert(SPARK_M4_HIDDEN % SPARK_M4_STORED_TILE_D == 0u,
              "the hidden dimension tiles evenly by 32 output columns");
static_assert(SPARK_M4_STORED_GRID_Y == 160u, "5120 hidden values are 160 tiles of 32 columns");
static_assert(SPARK_M4_STORED_THREADS == 256u,
              "the block is 32 columns x 8 parallel K groups");
static_assert(SPARK_M4_STORED_TILE_D * SPARK_M4_STORED_K_GROUPS == SPARK_M4_STORED_THREADS,
              "one thread per (column, K group)");
static_assert(SPARK_M4_MOE_TOP_K * SPARK_M4_STORED_TILE_D <= SPARK_M4_STORED_THREADS,
              "one reducer thread per (slot, column)");
static_assert(SPARK_M4_MOE_TOP_K * SPARK_M4_QUANT_BLOCK == 192u, "six slots of 32 values");
static_assert(SPARK_B12X_KTILE == SPARK_M4_QUANT_BLOCK, "an MX scale covers one 32-value block");

// The E2M1 value stored in the low nibble of byte (index / 2) at even indices, high nibble at odd.
// The helpers carry a stored_ prefix so this file can be compiled standalone AND in build.rs's
// combined probe, where b12x_mxfp4_moe_sm121.cu already defines the unprefixed versions.
__device__ __forceinline__ uint8_t spark_stored_fp4_low_nibble(uint8_t packed) { return (uint8_t)(packed & 0x0Fu); }
__device__ __forceinline__ uint8_t spark_stored_fp4_high_nibble(uint8_t packed) { return (uint8_t)(packed >> 4); }

__device__ __forceinline__ float spark_stored_fp4_load(const uint8_t *row, uint32_t index) {
    const uint8_t packed = row[index / 2u];
    const uint8_t nibble = (index % 2u) == 0u ? spark_stored_fp4_low_nibble(packed)
                                              : spark_stored_fp4_high_nibble(packed);
    return spark_fp4_e2m1_decode(nibble);
}

extern "C" __global__ void b12x_mxfp4_moe_stored_sm121(
    const uint8_t *__restrict__ act_fp8,        // [tokens * top_k][640] E4M3, A8 chain layout
    const uint8_t *__restrict__ act_scales,     // [tokens * top_k][20] ue8m0, A8 chain layout
    const uint8_t *__restrict__ w_fp4,          // [experts][d][288] packed E2M1, stored layout
    const uint8_t *__restrict__ w_scales,       // [experts][d][ 18] ue8m0,      stored layout
    const uint32_t *__restrict__ expert_ids,    // [tokens][top_k] expert index per slot
    float *__restrict__ slot_acc,               // [tokens][top_k][d] f32 IN/OUT: rank partials, added
    const uint32_t tokens,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t token = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t col_local = tid % SPARK_M4_STORED_TILE_D;
    const uint32_t group = tid / SPARK_M4_STORED_TILE_D;
    const uint32_t column = blockIdx.y * SPARK_M4_STORED_TILE_D + col_local;
    const bool live = token < tokens;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_MOE_MXFP4, t_enter);

    // This K group's scaled block partials: [slot][block][column-within-tile]. A 32-value block
    // partial is independent of every other block's, so splitting the 18 ascending blocks across the
    // eight groups removes the per-thread serial chain without moving a rounding: the reducer below
    // still folds the 18 scaled partials in strictly ascending block order, with the same two exact
    // power-of-two multiplies and one rounded add per block.
    __shared__ float terms[SPARK_M4_MOE_TOP_K][SPARK_M4_STORED_K_BLOCKS][SPARK_M4_STORED_TILE_D];

    if (live) {
        const uint64_t routed_row = (uint64_t)token * SPARK_M4_MOE_TOP_K;
        for (uint32_t slot = 0; slot < (uint32_t)SPARK_M4_MOE_TOP_K; ++slot) {
            const uint32_t expert = expert_ids[routed_row + slot];
            // The slot's own activation row, A8 chain layout: the routing weight was multiplied into
            // the activation before quantisation (model.py:849-851).
            const uint8_t *act_row = act_fp8 + (routed_row + slot) * SPARK_M4_STORED_ACT_ROW;
            const uint8_t *act_scale_row =
                act_scales + (routed_row + slot) * SPARK_M4_STORED_ACT_SCALE_ROW;
            // The stored weight row: one output element of one expert. The code row and the scale row
            // start at the same (expert, column) index but live in two different tensors with their
            // own strides - this is the whole layout difference and it stays on these two lines.
            const uint64_t weight_row = (uint64_t)expert * SPARK_M4_HIDDEN + column;
            const uint8_t *w_row = w_fp4 + weight_row * SPARK_M4_STORED_CODE_ROW;
            const uint8_t *w_scale_row = w_scales + weight_row * SPARK_M4_STORED_SCALE_ROW;
            for (uint32_t block = group; block < (uint32_t)SPARK_M4_STORED_K_BLOCKS;
                 block += (uint32_t)SPARK_M4_STORED_K_GROUPS) {
                const uint32_t base = block * SPARK_M4_QUANT_BLOCK;
                const float act_scale = spark_ue8m0_decode(act_scale_row[block]);
                const float w_scale = spark_ue8m0_decode(w_scale_row[block]);
                float partial = 0.0f;
                // Every index is a REAL K value (base + i < 576), so there is no padded region to
                // mask: the stored width is the arithmetic width in this kernel.
#pragma unroll
                for (uint32_t b = 0u; b < SPARK_M4_QUANT_BLOCK / 2u; ++b) {
                    const uint32_t i0 = base + 2u * b;
                    const uint8_t packed = w_row[i0 / 2u];
                    const float a0 = spark_fp8_e4m3_decode(act_row[i0]);
                    const float w0 = spark_fp4_e2m1_decode(spark_stored_fp4_low_nibble(packed));
                    partial += a0 * w0;
                    const float a1 = spark_fp8_e4m3_decode(act_row[i0 + 1u]);
                    const float w1 = spark_fp4_e2m1_decode(spark_stored_fp4_high_nibble(packed));
                    partial += a1 * w1;
                }
                // The reference's block term, with the explicit intrinsics that pin its
                // separate-rounding contract independently of nvcc's -fmad default:
                // t_b = (partial * act_scale) * w_scale, both multiplies exact (powers of two).
                terms[slot][block][col_local] =
                    __fmul_rn(__fmul_rn(partial, act_scale), w_scale);
            }
        }
    }
    __syncthreads();

    // One reducer thread per (slot, output element) folds the slot's 18 scaled block partials in
    // strictly ascending block order - the reference's only ordered arithmetic - and adds the rank
    // partial into the f32 slot accumulator, unrounded. One reducer owns each (slot, column), so the
    // accumulator is still written by exactly one thread.
    const uint32_t reducers = (uint32_t)SPARK_M4_MOE_TOP_K * SPARK_M4_STORED_TILE_D;
    if (tid < reducers) {
        const uint32_t slot = tid / SPARK_M4_STORED_TILE_D;
        const uint32_t col = tid % SPARK_M4_STORED_TILE_D;
        const uint32_t out_column = blockIdx.y * SPARK_M4_STORED_TILE_D + col;
        if (live && out_column < (uint32_t)SPARK_M4_HIDDEN) {
            float acc = 0.0f;
            for (uint32_t block = 0; block < (uint32_t)SPARK_M4_STORED_K_BLOCKS; ++block) {
                acc = __fadd_rn(acc, terms[slot][block][col]);
            }
            slot_acc[((uint64_t)token * SPARK_M4_MOE_TOP_K + slot) * SPARK_M4_HIDDEN + out_column] += acc;
        }
    }
    __syncthreads();

    if (tid == 0u) {
        // payload_meta: one f32 per (slot, output element) of this block's tile.
        const uint32_t meta = SPARK_M4_STORED_TILE_D * SPARK_M4_MOE_TOP_K * 4u;
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_MOE_MXFP4, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}

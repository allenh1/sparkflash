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

// kernels/routed_gate_up_sm121.cu
//
// The W4A8 routed-expert gate/up projection of roadmap 6.3 item 5: one FP8 E4M3 x packed-FP4 E2M1
// GEMM of the layer's ffn block input against one expert's w1 (or w3) slice, per routed (token, slot)
// row. It is the kernel the chained stage-7 harness prints as the missing seam ("no device kernel in
// the six-cubin set reduces K=5120 against packed FP4 weights") and the one b12x_mxfp4_moe_sm121.cu
// deliberately does NOT contain: that kernel is the down/combine projection only.
//
// The operands are the same D14 W4A8 pair the combine kernel reads, with the same provenance
// (docs/DECISIONS.md D14; docs/architecture/03-moe.md:118-134 for the checkpoint's grid):
//
//   * A - the ffn block input, as the reference's own act_quant leaves it (kernel.py:41-124):
//     FP8 E4M3, one byte per value, with one ue8m0 scale per 32 values along K. The producer is
//     mega_mhc_sm121, which already emits X_hat_fp8 and its ue8m0 grid for the sublayer input
//     (stage 1 prints "xhat_fp8 5120/5120 exact" and "ue8m0 scales 160/160 exact"). K is the whole
//     hidden size, 5120, because this GEMM reduces the block input, not the routed width.
//   * B - one expert's w1 or w3 slice for THIS RANK: FP4 E2M1 packed two per byte (low nibble first),
//     one ue8m0 scale per output row per 32 values along K, i.e. the checkpoint's [out, in/32] grid.
//     There are 576 output rows per rank: this is D30's implemented routed-expert layout, TP=4 over
//     the intermediate width (moe_inter_dim 2304 / four ranks), padded to 640 in the activation and
//     output rows. D2's reference rule shards whole experts at full 2304 width and is corrected by
//     D30, so 576-per-rank is the implemented shape, not the reference's.
//   * out - bf16, [tokens * top_k][640]: the 576 real columns plus 64 zero columns. The padding is
//     part of the ABI because the pinned SwiGLU routed shape is 640 wide and the combine kernel masks
//     the last 64 of its A operand; this kernel therefore WRITES zeros to columns 576..639 rather
//     than leaving the tail to a caller's memset, so a the-padding-is-zero invariant is a property
//     of the launch and not a precondition on the loader.
//
// What it computes, exactly: out[r][c] = bf16(sum over the 160 ascending 32-value K blocks of
// (sum over the block of fp8_e4m3_decode(a) * fp4_e2m1_decode(b)) * ue8m0(a_scale) * ue8m0(b_scale)).
// Both scales are powers of two, so scaling the block PARTIAL PRODUCT and scaling the operands are
// the same real arithmetic; what the ordering fixes is one rounded f32 add per block, which is what
// src/routed_gate_up_ref.rs mirrors and pins. Each multiply/add pair is a separate rounding
// (__fmul_rn then __fadd_rn), never a contracted FMA, so the device's f32 intermediates are the
// twin's. The f32 value is rounded to bf16 exactly once at the end. There is NO residual, NO routing
// weight and NO SwiGLU here:
//
//   * the residual belongs to the combine projection (b12x_mxfp4_moe_sm121.cu adds it);
//   * the routing weight multiplies the SwiGLU activation BEFORE the bf16 cast the activation
//     quantiser reads (model.py:849-851), so it is part of the down projection's A operand and
//     folding it into this GEMM would be a different arithmetic (mxfp4_ref.rs pins that difference
//     with a mutation). This kernel's contract is therefore the unrouted gate/up tensor; the
//     activation stage is the pinned swiglu_mxfp8_quant_sm121 kernel, and a weighted variant of that
//     activation stage is a separate, named gap (docs/archive/build/phase3-gate-up.md);
//   * the shared expert does not use this kernel at all: its w1/w3 are a dense FP8 [2304, 5120]
//     projection on the checkpoint's 32x32 grid, already served by b12x_mxfp8_dense_sm121 with
//     runtime inner/cols (stage-7 K13/K14 run at K=5120 -> 2,304 and match bit for bit).
//
// Geometry and launch: one block per (routed row, output tile of 256 columns); 256 threads, one
// output element per thread (the same per-thread serial-K shape as the combine kernel, so the
// accumulation order is the reference's and the comparison against the twin is exact rather than
// statistical). grid = (tokens * top_k, ceil(640 / 256) = 3, 1). The padding columns are covered by
// the third tile and written zero. No tensor-core path, no TMA, no prefetch and no latency claim:
// this kernel exists to close the seam with exact arithmetic first, and the trace scope is
// TRACE_SCOPE_MOE_MXFP4 - the routed-expert consumer domain the combine kernel and the router
// already record under, so the 16-bit scope table stays untouched (the shipped cubin is the
// trace-disabled build, so the launch passes a null ring and the macros are no-ops).
//
// Verification status (the class of the claim, stated honestly): the source is pinned, the cubin pin
// is recorded in src/routed_gate_up_abi.rs, and src/routed_gate_up_ref.rs is the CPU twin that
// defines the arithmetic - host-tested against a naive loop with an independently written
// E2M1/E4M3/ue8m0 decoder and bf16 rounder, against a golden block-fold vector pinned at f32
// precision, against one-hot hand arithmetic and on padding. examples/m10_gate_up.rs additionally
// compares the twin bit for bit
// against the shared host implementation on released layer-0 expert bytes, then compares the device
// against the twin, and reports an unquantised f32 control separately as a bounded gap. NO DEVICE RUN
// EXISTS YET: the harness refuses to run unless the Lead stages the pinned cubin in a window, and it
// prints its pass bar before any number.

#include "m4_common_sm121.cuh"

// One output column per reducer thread and eight parallel K groups per block: 32 columns x 8
// groups. The eight groups split the 160 ascending block partials the reference folds serially; the
// reducer re-adds them in ascending block order, so the split changes the schedule, not the value.
#define SPARK_GATE_UP_TILE 32u
#define SPARK_GATE_UP_K_GROUPS 8u
#define SPARK_GATE_UP_THREADS (SPARK_GATE_UP_TILE * SPARK_GATE_UP_K_GROUPS)
// K is the whole hidden size: the block input, not the routed width.
#define SPARK_GATE_UP_K_BLOCKS (SPARK_M4_HIDDEN / SPARK_M4_QUANT_BLOCK)
// Packed FP4 weight bytes per output row: 5120 values, two per byte.
#define SPARK_GATE_UP_PACKED_ROW (SPARK_M4_HIDDEN / 2u)
// ue8m0 scale bytes per output row and per activation row: one per 32 values along K.
#define SPARK_GATE_UP_SCALE_ROW (SPARK_M4_HIDDEN / SPARK_M4_QUANT_BLOCK)
// The output tile count that covers the padded routed width.
#define SPARK_GATE_UP_TILES ((SPARK_M4_MOE_WIDTH_PADDED + SPARK_GATE_UP_TILE - 1u) / SPARK_GATE_UP_TILE)

static_assert(SPARK_M4_HIDDEN % SPARK_M4_QUANT_BLOCK == 0u, "the hidden size tiles by 32");
static_assert(SPARK_GATE_UP_K_BLOCKS == 160u, "5120 values are 160 MX blocks");
static_assert(SPARK_M4_MOE_WIDTH % SPARK_M4_QUANT_BLOCK == 0u, "the routed width tiles by 32");
static_assert(SPARK_M4_MOE_WIDTH_PADDED % SPARK_M4_QUANT_BLOCK == 0u,
              "the padded routed width tiles by 32");
static_assert(SPARK_GATE_UP_TILES * SPARK_GATE_UP_TILE >= SPARK_M4_MOE_WIDTH_PADDED,
              "the tile grid covers the padded width");
static_assert(SPARK_GATE_UP_THREADS == 256u, "the block is 32 columns x 8 parallel K groups");
static_assert(SPARK_GATE_UP_TILE * SPARK_GATE_UP_K_GROUPS == SPARK_GATE_UP_THREADS,
              "one compute thread per (column, K group)");
static_assert(SPARK_M4_MOE_TOP_K == 6u, "six routed slots per token");
static_assert(SPARK_GATE_UP_PACKED_ROW == 2560u, "one 5120-wide FP4 row is 2560 bytes");
static_assert(SPARK_GATE_UP_SCALE_ROW == 160u, "one 5120-wide row has 160 ue8m0 scales");

// The E2M1 value stored in the low nibble of byte (index / 2) at even indices, high nibble at odd.
__device__ __forceinline__ uint8_t spark_gate_up_fp4_low(uint8_t packed) { return (uint8_t)(packed & 0x0Fu); }
__device__ __forceinline__ uint8_t spark_gate_up_fp4_high(uint8_t packed) { return (uint8_t)(packed >> 4); }

__device__ __forceinline__ float spark_gate_up_fp4_load(const uint8_t *row, uint32_t index) {
    const uint8_t packed = row[index / 2u];
    const uint8_t nibble = (index % 2u) == 0u ? spark_gate_up_fp4_low(packed)
                                              : spark_gate_up_fp4_high(packed);
    return spark_fp4_e2m1_decode(nibble);
}

extern "C" __global__ void routed_gate_up_sm121(
    const uint8_t *__restrict__ act_fp8,        // [tokens][hidden] E4M3, one byte per value
    const uint8_t *__restrict__ act_scales,     // [tokens][hidden / 32] ue8m0
    const uint8_t *__restrict__ w_fp4,          // [experts][width][hidden / 2] packed E2M1
    const uint8_t *__restrict__ w_scales,       // [experts][width][hidden / 32] ue8m0
    const uint32_t *__restrict__ expert_ids,    // [tokens * top_k] the routing selection
    __nv_bfloat16 *__restrict__ out,            // [tokens * top_k][width_padded] bf16
    const uint32_t tokens,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t routed_row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t col_local = tid % SPARK_GATE_UP_TILE;
    const uint32_t group = tid / SPARK_GATE_UP_TILE;
    const uint32_t column = blockIdx.y * SPARK_GATE_UP_TILE + col_local;
    const uint32_t live_rows = tokens * (uint32_t)SPARK_M4_MOE_TOP_K;
    const bool live = routed_row < live_rows;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_MOE_MXFP4, t_enter);

    // This K group's scaled block partials for one routed row: [block][column-within-tile]. The 160
    // K blocks are independent partial products, so eight groups compute them in parallel and the
    // reducer folds them in ascending block order, which is the reference's serial order.
    __shared__ float terms[SPARK_GATE_UP_K_BLOCKS][SPARK_GATE_UP_TILE];

    if (live && column < (uint32_t)SPARK_M4_MOE_WIDTH) {
        const uint32_t token = routed_row / (uint32_t)SPARK_M4_MOE_TOP_K;
        const uint32_t expert = expert_ids[routed_row];
        // The activation row is the token's row: the routing weight was folded into the activation
        // before quantisation (model.py:849-851), so every one of the token's slots multiplies the
        // same quantised E4M3 row and nothing here multiplies by routing again.
        const uint8_t *act_row = act_fp8 + (uint64_t)token * SPARK_M4_HIDDEN;
        const uint8_t *act_scale_row = act_scales + (uint64_t)token * SPARK_GATE_UP_SCALE_ROW;
        // The weight row is this rank's slice of the expert's gate (or up) matrix, [576, 5120] FP4
        // for one rank: the caller stages the slice and passes its rows contiguously.
        const uint32_t weight_row = expert * (uint32_t)SPARK_M4_MOE_WIDTH + column;
        const uint8_t *w_row = w_fp4 + (uint64_t)weight_row * SPARK_GATE_UP_PACKED_ROW;
        const uint8_t *w_scale_row =
            w_scales + spark_b12x_row_scale_index(weight_row, 0u, (uint32_t)SPARK_M4_HIDDEN);
        for (uint32_t block = group; block < (uint32_t)SPARK_GATE_UP_K_BLOCKS;
             block += (uint32_t)SPARK_GATE_UP_K_GROUPS) {
            const uint32_t base = block * SPARK_M4_QUANT_BLOCK;
            const float act_scale = spark_ue8m0_decode(act_scale_row[block]);
            const float w_scale = spark_ue8m0_decode(w_scale_row[block]);
            float partial = 0.0f;
#pragma unroll
            for (uint32_t b = 0u; b < SPARK_M4_QUANT_BLOCK / 2u; ++b) {
                const uint32_t i0 = base + 2u * b;
                const uint8_t packed = w_row[i0 / 2u];
                const float a0 = spark_fp8_e4m3_decode(act_row[i0]);
                const float w0 = spark_fp4_e2m1_decode(spark_gate_up_fp4_low(packed));
                partial = __fadd_rn(partial, __fmul_rn(a0, w0));
                const float a1 = spark_fp8_e4m3_decode(act_row[i0 + 1u]);
                const float w1 = spark_fp4_e2m1_decode(spark_gate_up_fp4_high(packed));
                partial = __fadd_rn(partial, __fmul_rn(a1, w1));
            }
            // The reference's block term: both scales are powers of two, so these two multiplies are
            // exact and this is the same value the serial kernel's block term had. The explicit
            // intrinsics keep the twin's separate-rounding contract independent of -fmad.
            terms[block][col_local] = __fmul_rn(__fmul_rn(partial, act_scale), w_scale);
        }
    }
    __syncthreads();

    // One reducer thread per output column: the 160-add ascending chain is the reference's order,
    // and it is the only ordered arithmetic in this kernel.
    if (tid < (uint32_t)SPARK_GATE_UP_TILE) {
        const uint32_t out_column = blockIdx.y * SPARK_GATE_UP_TILE + tid;
        if (live && out_column < (uint32_t)SPARK_M4_MOE_WIDTH_PADDED) {
            __nv_bfloat16 *out_row = out + (uint64_t)routed_row * SPARK_M4_MOE_WIDTH_PADDED;
            if (out_column < (uint32_t)SPARK_M4_MOE_WIDTH) {
                float acc = 0.0f;
                for (uint32_t block = 0; block < (uint32_t)SPARK_GATE_UP_K_BLOCKS; ++block) {
                    acc = __fadd_rn(acc, terms[block][tid]);
                }
                out_row[out_column] = __float2bfloat16_rn(acc);
            } else {
                // Columns 576..639 are the ABI's padding: this kernel writes them zero so the
                // pinned SwiGLU routed shape is fed deterministically whatever a caller allocated.
                out_row[out_column] = __float2bfloat16_rn(0.0f);
            }
        }
    }
    __syncthreads();

    // payload_meta: one bf16 per output element of this block's tile.
    if (tid == 0u) {
        const uint32_t meta = SPARK_GATE_UP_TILE * 2u;
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_MOE_MXFP4, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}

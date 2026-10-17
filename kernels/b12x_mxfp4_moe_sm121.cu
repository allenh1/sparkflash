// kernels/b12x_mxfp4_moe_sm121.cu
//
// The W4A8 mixture-of-experts combine of specification 2.1/6.2: 384 routed experts with top-6
// routing and one shared expert, whose width is 576 per rank padded to 640, FP4 E2M1 weights with
// one ue8m0 scale per 32 values along the reduction dimension, against FP8 E4M3 activations with
// per-32 ue8m0 scales. The activation format is decision D14 (docs/DECISIONS.md), and the three
// artefacts that settle it are:
//
//   * the checkpoint declares it - config.json:12-21 is quant_method "fp8", activation_scheme
//     "dynamic", weight_block_size [32, 32], scale_fmt "ue8m0", expert_dtype "fp4" - and
//     expert_dtype names the routed WEIGHT format only, since no activation tensor is stored;
//   * the reference is typed W4A8: model.py:181-183 says "Quantized weights need a quantized
//     activation, and both fp4 and fp8 weights take an fp8 one", and the fp4 branch (186-195)
//     makes the same act_quant(x, 32, "ue8m0", e8m0fnu) call as the fp8 branch (196-205);
//   * its fp4 GEMM is the FP8xFP8 path: "FP8 act x FP4 weight GEMM ... cast FP4 to FP8 via float,
//     then do FP8xFP8 GEMM. Apply activation and weight scales to the accumulator"
//     (kernel.py:486-499), with A: [M, K] FP8 and B: [N, K] FP4 (kernel.py:511-515).
//
// The spec's only word on this is the MOE_MXFP4 telemetry row's "W4A4" (spec line 1130), and that
// row is the spec's error: the spec contradicts it first at line 86, where the fused mHC kernel
// emits this block input as dynamic FP8 on a 32x32 ue8m0 grid. "MXFP4" in this filename and in the
// milestone list names the weight format. An earlier version of this file took packed E2M1
// activations and justified it with specification 6.2, which makes no activation claim at all.
//
// What it computes, exactly: out[t][j] = residual + the sum over the six slots of that slot's
// FP8xFP8 block sums, accumulated in f32 in ascending slot order and, inside a slot, in ascending
// 32-value K block order, with both ue8m0 scales applied to each block's PARTIAL PRODUCT exactly as
// kernel.py:552-553 applies them to the accumulator, and one rounding to bf16 at the very end. The
// routing weight is deliberately NOT folded in here: the reference multiplies it into the SwiGLU
// activation before the bf16 cast that the activation quantiser reads (model.py:849-851), so it is
// part of the A operand, and folding it into this GEMM would quantise an unrouted tensor instead -
// a different arithmetic, which src/mxfp4_ref.rs pins with a mutation. This kernel is the
// down/combine projection: the up/gate projections that produce the activation are the same MXFP4
// GEMM and are not duplicated here.
//
// The A operand is one E4M3 byte per value, [tokens * top_k][width_padded], with per-32 ue8m0
// scales [tokens * top_k][width_padded / 32]. The per-slot rows are the consequence of the routing
// rule above. SPARK_M4_MOE_ACT_BYTES_PER_VALUE states that byte width and a static_assert pins it,
// so a packed E2M1 activation operand (half a byte per value, which is what this kernel read before
// D14) cannot come back quietly. The FP4 weight keeps the checkpoint's [out, in/32] grid, one ue8m0
// scale per output row per 32 values along K, addressed by the spark_b12x_row_scale_* helpers
// (docs/architecture/03-moe.md:118-134).
//
// The 576 -> 640 padding is real work, not a comment: the kernel MASKS the padded region of the
// activation row to exactly zero (see the index test in the inner loop), so the 64 extra MACs per
// output cannot change a result even if a caller leaves garbage in the padding. The mask is what
// makes the invariant a property of this kernel rather than a precondition on its caller; the CPU
// reference masks identically, and it still COUNTS the wasted MACs so the cost of the padding is
// visible rather than assumed away.
//
// Verification status: executed on node0 (GB10, sm_121a) on 2026-10-17. nvcc 13.0.3
// compiles it for sm_121a, clang parses it against the stub headers, the signature and geometry
// are pinned by src/m4_abi.rs, and src/mxfp4_ref.rs defines and tests the numerics on the CPU.
// The device run: stage 3 of examples/m7_layer0.rs with 8 resident experts on real layer-0 bytes
// PASSed - the combine output 20,480/20,480 bit-exact against the bf16 rounding of the
// quantised reference, max ULP distance 0. The router is still the host seam and the shared
// expert is still host arithmetic, as the harness prints; no latency, no expert-parallel
// transport and no tensor-core path is claimed. The dispatch handshake that feeds this kernel is
// not in this file and does not poll any peer-written word.

#include "m4_common_sm121.cuh"

// ---------------------------------------------------------------------------------------
// Loader ABI (A8): the pre-padded row layout this kernel reads
// ---------------------------------------------------------------------------------------
// D14 (docs/DECISIONS.md) fixes the activation operand as FP8 E4M3 with per-32 ue8m0 scales and
// makes the PADDED width part of the kernel ABI. This kernel reads the pre-padded layout and never
// the reference's whole-expert grid: K = SPARK_M4_MOE_WIDTH_PADDED = 640, not the 576-wide per-rank
// slice and not the reference's moe_inter_dim = 2304. A loader must emit, per routed (token, slot)
// row:
//   * act_fp8     [tokens * top_k][640] E4M3 bytes     (activation row stride = 640);
//   * act_scales  [tokens * top_k][20]  ue8m0 bytes    (activation scale row stride = 20);
// and per output row (expert, hidden index):
//   * w_fp4       [experts][d][320] packed E2M1 bytes  (weight row stride = 320);
//   * w_scales    [experts][d][20]  ue8m0 bytes        (weight scale row stride = 20).
// Index >= SPARK_M4_MOE_WIDTH (576) is masked to zero on the activation side, so the 64 padded
// values cannot change a result; the kernel still ADDRESSES them, which is why the strides are a
// contract and not an optimisation. A loader that produced the reference's [out, 72] grid
// (2304 / 32) or a 576-wide [out, 18] grid (576 / 32) would misindex every row after the first.
// Producing this layout is the checkpoint loader's job, which is NOT in milestone 4; it lands with
// the milestone-5 integration that loads real weights and prefetches them.
// src/mxfp4_ref.rs::check_moe_layout is the Rust-side rejection of any other layout.

// One block per (token, d-tile); one output element per thread.
#define SPARK_M4_MOE_TILE_D 256u
#define SPARK_M4_MOE_THREADS 256u
// The reduction walks the padded width in whole MX blocks.
#define SPARK_M4_MOE_K_BLOCKS (SPARK_M4_MOE_WIDTH_PADDED / SPARK_M4_QUANT_BLOCK)
// The A operand is FP8 E4M3, one byte per value (D14). This is the pin that keeps a packed E2M1
// activation operand - half a byte per value, which is what this kernel read before D14 - from
// returning quietly: the row stride below is derived from it, and the static_asserts fail the build
// if it changes. m4_abi.rs pins the same text against a mutation, and src/mxfp4_ref.rs rejects an
// E2M1-sized operand with an assertion of its own.
#define SPARK_M4_MOE_ACT_BYTES_PER_VALUE 1u
#define SPARK_M4_MOE_ACT_BYTES_PER_ROW (SPARK_M4_MOE_WIDTH_PADDED * SPARK_M4_MOE_ACT_BYTES_PER_VALUE)
// FP4 packs two values per byte, low nibble first. It now describes the WEIGHTS only: the weight
// still costs half a byte per value while the activation costs a whole one.
#define SPARK_M4_FP4_PER_BYTE 2u

static_assert(SPARK_M4_MOE_WIDTH_PADDED % SPARK_M4_QUANT_BLOCK == 0u, "the padded width tiles by 32");
static_assert(SPARK_M4_MOE_K_BLOCKS == 20u, "640 values are 20 MX blocks");
static_assert(SPARK_M4_HIDDEN % SPARK_M4_MOE_TILE_D == 0u, "the hidden dimension tiles evenly");
static_assert(SPARK_M4_MOE_THREADS == SPARK_M4_MOE_TILE_D, "one output element per thread");
static_assert(SPARK_M4_MOE_TOP_K * SPARK_M4_QUANT_BLOCK == 192u, "six slots of 32 values");
// The fp4 expert scale grid is the checkpoint's [out, in/32]: one scale per output row per 32
// values along K, which is SPARK_B12X_KTILE values per scale, so the row stride must be exactly
// width_padded / 32 or every row after the first would read the previous row's scales.
static_assert(SPARK_B12X_KTILE == SPARK_M4_QUANT_BLOCK, "an MX scale covers one 32-value block");
static_assert(SPARK_M4_MOE_WIDTH_PADDED / SPARK_M4_QUANT_BLOCK == 20u,
              "the expert scale row is 20 ue8m0 bytes wide");
// The activation operand's format, in the build itself: one E4M3 byte per padded width value, with
// the ue8m0 scales as a separate per-32 grid. A packed E2M1 activation would halve the first of
// these and fail here.
static_assert(SPARK_M4_MOE_ACT_BYTES_PER_VALUE == 1u,
              "the A operand is one E4M3 byte per value; packed E2M1 half-bytes are D14's defect");
static_assert(SPARK_M4_MOE_ACT_BYTES_PER_ROW == SPARK_M4_MOE_WIDTH_PADDED,
              "an activation row is one byte per padded width value, scales aside");
static_assert(SPARK_M4_MOE_ACT_BYTES_PER_VALUE * SPARK_M4_FP4_PER_BYTE == 2u,
              "an E4M3 activation costs twice the bytes of a packed E2M1 weight");

// The E2M1 value stored in the low nibble of byte (index / 2) at even indices, high nibble at odd.
__device__ __forceinline__ uint8_t spark_fp4_low_nibble(uint8_t packed) { return (uint8_t)(packed & 0x0Fu); }
__device__ __forceinline__ uint8_t spark_fp4_high_nibble(uint8_t packed) { return (uint8_t)(packed >> 4); }

__device__ __forceinline__ float spark_fp4_load(const uint8_t *row, uint32_t index) {
    const uint8_t packed = row[index / 2u];
    const uint8_t nibble = (index % 2u) == 0u ? spark_fp4_low_nibble(packed) : spark_fp4_high_nibble(packed);
    return spark_fp4_e2m1_decode(nibble);
}

extern "C" __global__ void b12x_mxfp4_moe_sm121(
    const __nv_bfloat16 *__restrict__ hidden,   // [tokens][d] pre-combine input, kept for the residual
    const uint8_t *__restrict__ act_fp8,        // [tokens * top_k][width_padded] E4M3, one byte per value
    const uint8_t *__restrict__ act_scales,     // [tokens * top_k][width_padded / 32] ue8m0
    const uint8_t *__restrict__ w_fp4,          // [experts][d][width_padded / 2] packed E2M1
    const uint8_t *__restrict__ w_scales,       // [experts][d][width_padded / 32] ue8m0
    const uint32_t *__restrict__ expert_ids,    // [tokens][top_k] expert index per slot
    __nv_bfloat16 *__restrict__ combined,       // [tokens][d] the routed sum over slots
    const uint32_t tokens,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t token = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t column = blockIdx.y * SPARK_M4_MOE_TILE_D + lane;
    const uint32_t packed_width = SPARK_M4_MOE_WIDTH_PADDED / SPARK_M4_FP4_PER_BYTE;
    const uint32_t scale_width = SPARK_M4_MOE_WIDTH_PADDED / SPARK_M4_QUANT_BLOCK;
    const uint64_t act_row_bytes = SPARK_M4_MOE_ACT_BYTES_PER_ROW;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_MOE_MXFP4, t_enter);

    if (token < tokens && column < SPARK_M4_HIDDEN) {
        float acc = 0.0f;
        const uint64_t routed_row = ((uint64_t)token * SPARK_M4_MOE_TOP_K);
        for (uint32_t slot = 0; slot < (uint32_t)SPARK_M4_MOE_TOP_K; ++slot) {
            const uint32_t expert = expert_ids[routed_row + slot];
            // The slot's own activation row: the routing weight was multiplied into the activation
            // before this operand was quantised (model.py:849-851), so each slot reads its own E4M3
            // row and its own ue8m0 scales. Nothing here multiplies by routing, which is what makes
            // this GEMM the reference's FP8xFP8 GEMM rather than a different arithmetic.
            const uint8_t *act_row = act_fp8 + (routed_row + slot) * act_row_bytes;
            const uint8_t *act_scale_row = act_scales + (routed_row + slot) * scale_width;
            const uint8_t *w_row = w_fp4 + ((uint64_t)expert * SPARK_M4_HIDDEN + column) * packed_width;
            const uint8_t *w_scale_row = w_scales + spark_b12x_row_scale_index(
                                                       expert * SPARK_M4_HIDDEN + column, 0u,
                                                       SPARK_M4_MOE_WIDTH_PADDED);
            for (uint32_t block = 0; block < SPARK_M4_MOE_K_BLOCKS; ++block) {
                const uint32_t base = block * SPARK_M4_QUANT_BLOCK;
                const float act_scale = spark_ue8m0_decode(act_scale_row[block]);
                const float w_scale = spark_ue8m0_decode(w_scale_row[block]);
                // The FP8xFP8 block partial over this 32-value K block, which is the reference's MMA
                // (kernel.py:550): the FP4 weight is cast to FP8 through float first
                // (kernel.py:538-540), and the two scales are applied to the PARTIAL PRODUCT rather
                // than to the operands (kernel.py:552-553).
                float partial = 0.0f;
                for (uint32_t i = 0; i < SPARK_M4_QUANT_BLOCK; ++i) {
                    // The padded region of the activation contributes zero by construction: the
                    // mask below, not the caller, is what enforces it.
                    const uint32_t index = base + i;
                    const float act_value = index < (uint32_t)SPARK_M4_MOE_WIDTH
                                                ? spark_fp8_e4m3_decode(act_row[index])
                                                : 0.0f;
                    // The FP4 -> FP8 cast is exact on all sixteen E2M1 codes, so this round trip is
                    // the identity; it is written out because it is what kernel.py:538-540 does, and
                    // src/mxfp4_ref.rs enumerates the codes to prove the identity rather than assume
                    // it. The cast produces the value the MMA consumes, exactly as the reference.
                    const float w_value = spark_fp8_e4m3_decode(
                        spark_fp8_e4m3_encode(spark_fp4_load(w_row, index)));
                    partial += act_value * w_value;
                }
                acc += (partial * act_scale) * w_scale;
            }
        }
        // The residual is added in f32 and rounded once, which is the order mxfp4_ref.rs pins.
        const float residual = __bfloat162float(hidden[(uint64_t)token * SPARK_M4_HIDDEN + column]);
        combined[(uint64_t)token * SPARK_M4_HIDDEN + column] = __float2bfloat16_rn(residual + acc);
    }
    __syncthreads();

    if (lane == 0u) {
        // payload_meta: one bf16 per output element of this block's tile.
        const uint32_t meta = SPARK_M4_MOE_TILE_D * 2u;
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_MOE_MXFP4, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}

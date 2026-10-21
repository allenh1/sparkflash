// The WEIGHTED SwiGLU + MX-quantise epilogue: the routed experts' activation operand with the
// routing weight multiplied in BEFORE the quantiser, which is the model's order.
//
// WHY THIS FILE EXISTS. b12x_mxfp4_moe_sm121 consumes a weighted activation operand: one E4M3+/ue8m0
// pair per routed row, where the row is the expert's SwiGLU result times that slot's routing weight.
// swiglu_mxfp8_quant_sm121 produces the UNWEIGHTED activation, so the m17a chain assembled the
// combine operand on the host: host SwiGLU, host multiplication by the routing weight, host
// quantise. That host step is the F6 seam (DECISIONS D32). This kernel removes it by taking one
// per-row routing weight operand and doing exactly the reference's multiplication and rounding on
// the device:
//
//     activation = swiglu(gate, up, limit)                 // f32, same asymmetric clamp as the epilogue
//     product    = bf16(activation * weight)               // one rounding, round-nearest-even
//     code/scale = act_quant(product)                      // the activation rule, 32-value blocks
//
// The reference is crates/sparkflash-kernels/src/mxfp4_ref.rs::routed_activation_operand, which maps
// every value through m4_quant::bf16_bits(value * routing) and then m4_quant::act_quant_fp8_row.
// That bf16 rounding is part of the class: quantising the unrounded f32 product would differ from the
// engine's operand on values whose product needs more than eight mantissa bits, so it is reproduced
// explicitly. Multiplying already-quantised codes would quantise twice and is NOT what this kernel
// does.
//
// SHAPE AND LAYOUT. One warp owns one 32-value MX block, exactly as the unweighted epilogue: gate/up
// are [rows][cols] bf16 with cols a multiple of 32 (640 for the routed path: 576 real columns padded
// to 20 blocks), weights is [rows] f32 (one per routed row, row = token * top_k + slot), and the
// outputs are [rows][cols] E4M3 plus [rows][cols / 32] ue8m0 - the operand b12x_mxfp4_moe_sm121
// expects, including the zero padding, because the padded gate/up bytes quantise to zero codes.
//
// TRACE. The scope follows the launch shape as in the unweighted epilogue: cols = 640 is the routed
// MXFP4 path (TRACE_SCOPE_MOE_MXFP4); any other width keeps the dense scope so a future shared-expert
// caller cannot record under a scope its operands do not belong to. payload_meta counts one E4M3
// byte per value the reporting block actually quantised (active warps times 32), the same rule as the
// unweighted entry.
//
// The activation clamp is deliberately asymmetric, exactly as the engine's moe.rs does it
// (gate = min(gate, limit) upper side only; up = clamp(up, -limit, limit)): silu is not odd, so
// clamping the gate on both sides would change the operand for negative gates beyond the limit. The
// host test in src/swiglu_ref.rs keeps both epilogues' clamp shape pinned in the source text; the
// weighted path's product rounding is pinned by quantise_row_weighted's tests.

#include "m4_common_sm121.cuh"
#include "m4_activation_quant_sm121.cuh"

// One MX activation block's scale and one value's code, for a warp that owns the whole 32-value
// block. Kept separate from the unweighted file's helper of the same shape so the combined probe can
// include both translation units without a redefinition.
__device__ __forceinline__ uint8_t spark_weighted_activation_quantize_block(
    float value,
    uint32_t lane,
    uint8_t *__restrict__ scale_out) {
    float amax = fabsf(value);
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, offset));
    }
    const uint8_t scale = spark_act_quant_scale_code(amax);
    if (lane == 0u) {
        *scale_out = scale;
    }
    const float inv_scale = 1.0f / spark_ue8m0_decode(scale);
    return spark_fp8_e4m3_encode(value * inv_scale);
}

extern "C" __global__ void swiglu_mxfp8_quant_weighted_sm121(
    const __nv_bfloat16 *__restrict__ gate,   // [rows][cols] bf16, w1 x
    const __nv_bfloat16 *__restrict__ up,     // [rows][cols] bf16, w3 x
    const float *__restrict__ weights,        // [rows] one routing weight per routed row
    uint8_t *__restrict__ out_fp8,            // [rows][cols] E4M3, row stride cols
    uint8_t *__restrict__ out_scales,         // [rows][cols / 32] ue8m0, row stride cols / 32
    const uint32_t rows,
    const uint32_t cols,
    const float limit,                        // swiglu_limit, 10.0 in the released config
    GpuTraceRing *__restrict__ trace_ring,    // scope follows cols, see trace_scope below
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint64_t warp = (uint64_t)blockIdx.x * warps_per_block + (uint64_t)(threadIdx.x >> 5u);
    const uint32_t blocks_per_row = cols >> 5u;  // cols is a multiple of 32 by contract
    const uint64_t total_blocks = (uint64_t)rows * (uint64_t)blocks_per_row;

    const uint64_t trace_scope = cols == SPARK_M4_MOE_WIDTH_PADDED
                                     ? TRACE_SCOPE_MOE_MXFP4
                                     : TRACE_SCOPE_DENSE_MXFP8;
    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, trace_scope, t_enter);

    // Guard instead of early return, so every thread of the block reaches the trace barrier below.
    if (warp < total_blocks) {
        const uint32_t row = (uint32_t)(warp / blocks_per_row);
        const uint32_t block = (uint32_t)(warp % blocks_per_row);
        const uint64_t index =
            (uint64_t)row * (uint64_t)cols + (uint64_t)block * 32u + (uint64_t)lane;

        const float g = __bfloat162float(gate[index]);
        const float u = __bfloat162float(up[index]);
        const float clamped_gate = fminf(g, limit);               // upper side only
        const float clamped_up = fminf(fmaxf(u, -limit), limit);  // both sides
        const float activated = (clamped_gate / (1.0f + __expf(-clamped_gate))) * clamped_up;
        // The reference's order: multiply, round the product to bf16, THEN quantise. The rounded
        // value is widened back to f32 for the same block-scale/encode helpers the unweighted
        // epilogue uses, so the two kernels cannot disagree about the quantiser.
        const float weighted = __bfloat162float(__float2bfloat16_rn(activated * weights[row]));

        out_fp8[index] = spark_weighted_activation_quantize_block(
            weighted,
            lane,
            &out_scales[(uint64_t)row * (uint64_t)blocks_per_row + (uint64_t)block]);
    }
#if defined(SPARKFLASH_ENABLE_GPU_TRACE)
    __syncthreads();
#endif
    const uint64_t first_warp = (uint64_t)blockIdx.x * warps_per_block;
    const uint64_t remaining = total_blocks > first_warp ? total_blocks - first_warp : 0u;
    const uint32_t active_warps = remaining < warps_per_block ? (uint32_t)remaining : warps_per_block;
    const uint32_t meta = active_warps * 32u;
    if (lane == 0u) {
        SPARK_GPU_TRACE_EXIT(trace_ring, trace_scope, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
    (void)trace_scope;
    (void)meta;
}

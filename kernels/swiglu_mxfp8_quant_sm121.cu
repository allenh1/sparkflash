// The SwiGLU + MX-quantise epilogue that b12x_mxfp8_dense_sm121 does not have, and the standalone
// activation quantiser the shared expert needs beside it.
//
// SEVENTH PROBED KERNEL, STILL OUTSIDE THE FROZEN m4_abi KERNELS ARRAY. Specification 6.2 names six
// Milestone 4 kernels, and m4_abi's KERNELS array plus its per-kernel declaration pins are those
// six; adding an entry there would be an ABI revision, so these two entry points stay a layer-level
// addition outside that array (the Lead's decision on task-35). The build script's kernel set and
// the pinned baseline DO include this file: it is compiled on its own and in the combined probe by
// both the nvcc and the clang-fallback paths, its enabled trace macros are type-checked there, and
// baseline/kernel_baseline.txt records its cubin as the seventh entry. The host reference is
// src/swiglu_ref.rs.
//
// TRACE. Both entry points record under the scope of the arithmetic they feed, not of the file they
// live in. swiglu_mxfp8_quant_sm121 is launched for TWO shapes and the scope follows the shape: the
// routed-expert epilogue (cols = 640, SPARK_M4_MOE_WIDTH_PADDED) records under
// TRACE_SCOPE_MOE_MXFP4 because its operands are exactly what b12x_mxfp4_moe_sm121 consumes, and
// the shared-expert path (cols = 2304) records under TRACE_SCOPE_DENSE_MXFP8 because its operands
// feed the dense W8A8 GEMMs. The scope is picked from `cols` once per launch and passed to both
// macros, so no launch can carry a scope that contradicts its output. mxfp8_quantize_activations_sm121
// is always TRACE_SCOPE_DENSE_MXFP8. payload_meta is one E4M3 byte per value the block ACTUALLY
// quantised (active warps times 32, not blockDim.x: a short grid's last block owns fewer than
// warps_per_block whole blocks and idle warps quantise nothing). The instrumentation is the
// trace_globaltimer_sm121.cuh macro pattern the M4 kernels use; the barrier its end timestamp needs
// is compiled only when SPARKFLASH_ENABLE_GPU_TRACE is defined.
//
// TWO ENTRY POINTS, TWO DIFFERENT WEIGHT FORMATS, AND THE SHAPES ARE NOT INTERCHANGEABLE.
//
// 1. swiglu_mxfp8_quant_sm121 - the ROUTED experts' epilogue: gate = w1 x and up = w3 x, both bf16,
//    then the activation, then quantised into exactly the operand b12x_mxfp4_moe_sm121 consumes:
//    [rows][cols] E4M3 and [rows][cols / 32] ue8m0, with rows = tokens * top_k and cols = 640
//    (576 padded, 20 MX blocks). The FP4 expert path.
//
// 2. The same entry also serves the SHARED expert's post-SwiGLU quantise, which is the same
//    activation at a wider width: the shared expert's w1/w3 are F8_E4M3 [2304, 5120] with 32x32
//    block scales, so its intermediate is 2304 wide, and at cols = 2304 rows = tokens this emits
//    [tokens][2304] E4M3 with [tokens][72] ue8m0 - the 32-wide-block operand b12x_mxfp8_dense_sm121
//    wants for its w2. The width is the caller's argument, never baked in.
//
//    THE ACTIVATION IS ROUNDED TO BF16 BEFORE THE QUANTISER. Native computes the shared expert's
//    SwiGLU, casts that f32 activation to BF16, and only then lets w2's act_quant read it
//    (model.py:848-851). The operand is therefore quantise(bf16(swiglu)) and not
//    quantise(swiglu): quantising the unrounded f32 would keep up to eight mantissa bits the
//    released model never sees. This file rounds in the epilogue (one __float2bfloat16_rn per
//    value) and the host reference swiglu_ref::quantise_row rounds with m4_quant::bf16_bits, so
//    the two stay bit-exact.
//
// 3. mxfp8_quantize_activations_sm121 - the shared expert's INPUT quantiser, a different shape and a
//    different job: bf16 [rows][cols] to E4M3 [rows][cols] with [rows][cols / 32] ue8m0 scales over
//    the 5120-wide hidden, i.e. [tokens][160], because the shared expert's gate-up GEMMs are FP8
//    W8A8 and take a 5120-wide quantised activation. This is mega_mhc_sm121.cu's recipe, which
//    quantises the same 5120-wide vector on its way out of the pre-norm, extracted as its own entry
//    so the shared expert can use it without the norm fused in front of it.
//
// Shipping the [640]/[20] contracted operand into a GEMM expecting a [160]-scaled 5120-wide one is
// the mismatch that passes a shape check and fails a result, which is why the entry points are
// separate and each documents the format it serves.
//
// THE ACTIVATION IS THE REFERENCE'S, NOT THE TEXTBOOK'S. crates/sparkflash-engine/src/forward/moe.rs
// lines 681-686 clamp the two branches ASYMMETRICALLY:
//     gate = min(gate, limit)                  // upper side only
//     up   = clamp(up, -limit, limit)          // both sides
//     value = silu(gate) * up
// Clamping both branches on both sides is the plausible reading and it is wrong: silu is not odd,
// so the two readings differ for a negative gate beyond the limit. The asymmetry is reproduced here
// deliberately, and a host test in src/swiglu_ref.rs fails if the kernel text loses it.
//
// THE QUANTISE RECIPE IS THE REFERENCE'S ACTIVATION RULE, NOT THE WEIGHT-GRID RULE. Both output
// operands are ACTIVATIONS (the routed experts' SwiGLU result consumed by w2, and the shared
// expert's input consumed by w1/w3), so the scale is the reference's act_quant rule
// (kernel.py:36-37, 76-80; engine quant::e8m0_encode_ceil): amax over the 32-value block floored at
// 1e-4, times 1/448 in f32, exponent rounded UP to the next power of two, then
// spark_fp8_e4m3_encode(value / scale). The checkpoint weight-grid rule
// (spark_ue8m0_from_amax, floor(log2(amax)) - emax) is for stored weights only: on an activation it
// under-scales every block whose amax lies above 1.75 * 2^floor(log2 amax), and the operand then
// differs from the one the engine's act_quant produced.
//
// LAYOUT. One warp owns one 32-value block, so a block scale is a shuffle reduction and no shared
// memory is needed. Every lane of a warp is active because a block is exactly 32 wide.

#include "m4_common_sm121.cuh"
#include "m4_activation_quant_sm121.cuh"

// One MX activation block's scale and one value's code, for a warp that owns the whole 32-value
// block: the amax is a butterfly reduction across the warp, lane 0 writes the scale, and every lane
// encodes its own value against the reference's activation scale.
__device__ __forceinline__ uint8_t spark_activation_quantize_block(
    float value,
    uint32_t lane,
    uint8_t *__restrict__ scale_out) {
    float amax = fabsf(value);
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, offset));
    }
    // THE ACTIVATION RULE: the reference floors amax at 1e-4 and rounds the scale exponent up.
    const uint8_t scale = spark_act_quant_scale_code(amax);
    if (lane == 0u) {
        *scale_out = scale;
    }
    return spark_fp8_e4m3_encode(value / spark_ue8m0_decode(scale));
}

extern "C" __global__ void swiglu_mxfp8_quant_sm121(
    const __nv_bfloat16 *__restrict__ gate,   // [rows][cols] bf16, w1 x
    const __nv_bfloat16 *__restrict__ up,     // [rows][cols] bf16, w3 x
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

    // The scope is a property of the launch shape, not of this file: cols = 640 is the routed-expert
    // epilogue whose outputs b12x_mxfp4_moe_sm121 consumes, and cols = 2304 is the shared-expert
    // path whose outputs feed the dense W8A8 GEMMs. Both macros below take this one expression, so a
    // launch cannot open under one scope and close under another.
    const uint64_t trace_scope = cols == SPARK_M4_MOE_WIDTH_PADDED
                                     ? TRACE_SCOPE_MOE_MXFP4
                                     : TRACE_SCOPE_DENSE_MXFP8;
    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, trace_scope, t_enter);

    // The early return becomes a guard instead: the block's thread 0 has to stay alive to close the
    // trace event, and a barrier is only well defined if every non-exited thread reaches it. The
    // guard is per warp, not per block, which is legal here because nothing after it is
    // block-uniform.
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
        // Native rounds the unweighted activation to BF16 before w2's act_quant (model.py:848-851):
        // the shared expert's operand is quantise(bf16(swiglu)), not quantise(swiglu). The weighted
        // routed epilogue rounds the product for the same reason; here there is no product, so the
        // activation itself is the value that is cast.
        const float rounded = __bfloat162float(__float2bfloat16_rn(activated));

        out_fp8[index] = spark_activation_quantize_block(
            rounded,
            lane,
            &out_scales[(uint64_t)row * (uint64_t)blocks_per_row + (uint64_t)block]);
    }
    // The exit timestamp has to cover the whole block, so the barrier is taken after the block's
    // work. Unlike the b12x kernels' unconditional barrier, this one is compiled ONLY when the trace
    // is, the same condition that turns the macros below from no-ops into device code: the shipped
    // kernel pays nothing for a ring it was not built to write to. The lane guard is the same shape
    // b12x_mxfp8_wo_a_sm121.cu and b12x_mxfp8_dense_sm121.cu use.
#if defined(SPARKFLASH_ENABLE_GPU_TRACE)
    __syncthreads();
#endif
    // payload_meta: one E4M3 activation byte per value the reporting block quantised. Only block 0's
    // thread 0 records an event (the trace macro's guard), and for the real launches it is fully
    // live: 4x640 is 80 blocks over 10 grids and 2x2304 is 144 over 18, 8 warps each, so the count
    // is 8 warps = 256 values. The clamp below only changes a grid whose total block count is not a
    // multiple of warps_per_block (a one-block grid); there blockDim.x would over-report idle warps.
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

extern "C" __global__ void mxfp8_quantize_activations_sm121(
    const __nv_bfloat16 *__restrict__ values,  // [rows][cols] bf16 activation
    uint8_t *__restrict__ out_fp8,             // [rows][cols] E4M3, row stride cols
    uint8_t *__restrict__ out_scales,          // [rows][cols / 32] ue8m0, row stride cols / 32
    const uint32_t rows,
    const uint32_t cols,
    GpuTraceRing *__restrict__ trace_ring,    // TRACE_SCOPE_DENSE_MXFP8, see the event below
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint64_t warp = (uint64_t)blockIdx.x * warps_per_block + (uint64_t)(threadIdx.x >> 5u);
    const uint32_t blocks_per_row = cols >> 5u;
    const uint64_t total_blocks = (uint64_t)rows * (uint64_t)blocks_per_row;

    uint64_t t_enter = 0u;
    // The shared expert's input quantiser feeds the W8A8 gate-up GEMMs, which are the dense MXFP8
    // path's arithmetic, so it records under DENSE_MXFP8 rather than the FP4 expert scope.
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    // Guard instead of early return, so every thread of the block reaches the barrier below and the
    // exit timestamp covers the block.
    if (warp < total_blocks) {
        const uint32_t row = (uint32_t)(warp / blocks_per_row);
        const uint32_t block = (uint32_t)(warp % blocks_per_row);
        const uint64_t index =
            (uint64_t)row * (uint64_t)cols + (uint64_t)block * 32u + (uint64_t)lane;

        out_fp8[index] = spark_activation_quantize_block(
            __bfloat162float(values[index]),
            lane,
            &out_scales[(uint64_t)row * (uint64_t)blocks_per_row + (uint64_t)block]);
    }
    // Trace-only barrier, for the same reason as the entry above.
#if defined(SPARKFLASH_ENABLE_GPU_TRACE)
    __syncthreads();
#endif
    // payload_meta: the same active-warp rule as the entry above: block 0's 8 warps = 256 values for
    // the real grid, and the clamp only changes a grid whose total block count is not a multiple of
    // warps_per_block (a one-block grid).
    const uint64_t first_warp = (uint64_t)blockIdx.x * warps_per_block;
    const uint64_t remaining = total_blocks > first_warp ? total_blocks - first_warp : 0u;
    const uint32_t active_warps = remaining < warps_per_block ? (uint32_t)remaining : warps_per_block;
    const uint32_t meta = active_warps * 32u;
    if (lane == 0u) {
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
    (void)meta;
}

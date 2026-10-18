// The SwiGLU + MX-quantise epilogue that b12x_mxfp8_dense_sm121 does not have, and the standalone
// activation quantiser the shared expert needs beside it.
//
// NOT part of the six-kernel M4 ABI. m4_abi's KERNELS array and the six-cubin baseline are pinned,
// and adding an entry there would be an ABI revision; these are layer-level additions outside that
// array, by the Lead's decision on task-35. The crate's clang probe therefore still reports six
// kernels and does NOT cover this file: nvcc is this file's only compile proof.
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
    GpuTraceRing *__restrict__ trace_ring,    // no scope is defined for this kernel yet: the trace
    const uint32_t layer_id,                  // macros reduce to (void)(ring) when the build does
    const uint64_t step_seq) {                // not define SPARKFLASH_ENABLE_GPU_TRACE
    (void)trace_ring;
    (void)layer_id;
    (void)step_seq;

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint64_t warp = (uint64_t)blockIdx.x * warps_per_block + (uint64_t)(threadIdx.x >> 5u);
    const uint32_t blocks_per_row = cols >> 5u;  // cols is a multiple of 32 by contract
    const uint64_t total_blocks = (uint64_t)rows * (uint64_t)blocks_per_row;
    if (warp >= total_blocks) {
        return;
    }

    const uint32_t row = (uint32_t)(warp / blocks_per_row);
    const uint32_t block = (uint32_t)(warp % blocks_per_row);
    const uint64_t index = (uint64_t)row * (uint64_t)cols + (uint64_t)block * 32u + (uint64_t)lane;

    const float g = __bfloat162float(gate[index]);
    const float u = __bfloat162float(up[index]);
    const float clamped_gate = fminf(g, limit);               // upper side only
    const float clamped_up = fminf(fmaxf(u, -limit), limit);  // both sides
    const float activated = (clamped_gate / (1.0f + __expf(-clamped_gate))) * clamped_up;

    out_fp8[index] = spark_activation_quantize_block(
        activated,
        lane,
        &out_scales[(uint64_t)row * (uint64_t)blocks_per_row + (uint64_t)block]);
}

extern "C" __global__ void mxfp8_quantize_activations_sm121(
    const __nv_bfloat16 *__restrict__ values,  // [rows][cols] bf16 activation
    uint8_t *__restrict__ out_fp8,             // [rows][cols] E4M3, row stride cols
    uint8_t *__restrict__ out_scales,          // [rows][cols / 32] ue8m0, row stride cols / 32
    const uint32_t rows,
    const uint32_t cols,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    (void)trace_ring;
    (void)layer_id;
    (void)step_seq;

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint64_t warp = (uint64_t)blockIdx.x * warps_per_block + (uint64_t)(threadIdx.x >> 5u);
    const uint32_t blocks_per_row = cols >> 5u;
    const uint64_t total_blocks = (uint64_t)rows * (uint64_t)blocks_per_row;
    if (warp >= total_blocks) {
        return;
    }

    const uint32_t row = (uint32_t)(warp / blocks_per_row);
    const uint32_t block = (uint32_t)(warp % blocks_per_row);
    const uint64_t index = (uint64_t)row * (uint64_t)cols + (uint64_t)block * 32u + (uint64_t)lane;

    out_fp8[index] = spark_activation_quantize_block(
        __bfloat162float(values[index]),
        lane,
        &out_scales[(uint64_t)row * (uint64_t)blocks_per_row + (uint64_t)block]);
}

// kernels/engram_wkv_sm121.cu
//
// The Engram wkv projection, fp8 W8A8 (task-9, audit F4; released inference/model.py:345 and the
// released fp8 GEMM in inference/kernel.py:208-300).
//
// ONE TOKEN, ONE OUTPUT PER WARP LANE PAIR. The GEMM is [out_width][inner] fp8 weights over a single
// activation row, so the output is one vector and there is no M to tile: one warp owns one output
// element's dot product, one lane per column of a 32-column K block, and the block's 32 products are
// reduced by the five-step shuffle tree. That keeps the weight reads coalesced - lane k of a warp
// reads byte k of each 32-byte K block, so a warp-load is one 32-byte sector of the row - and it
// costs one launch, grid ceil(out_width / warps_per_block) blocks of 256 threads.
//
// THE REFERENCE'S ARITHMETIC, per output row n (kernel.py:227-272):
//
//     for each K block kb of 32 columns:
//         C_local     = dot(act_codes[kb], w_codes[n][kb])       // fp8 x fp8, f32 accumulate
//         C_accum    += (C_local * act_scale[kb]) * w_scale[n/32][kb]
//     out[n] = bf16(C_accum)
//
// The scales are applied to the BLOCK DOT, not to the elements: dequantising each operand element
// and summing the dequantised products rounds in different places. The weight scale grid is the
// checkpoint's [out_width/32][inner/32] byte grid, addressed directly - no repack.
//
// ACCUMULATION ORDER. The block dot is the five-step __shfl_down_sync tree (offsets 16, 8, 4, 2, 1),
// so the order is fixed and the CPU twin engram_native_ref::wkv_block_dot reproduces it op for op;
// the per-block scale multiply and the accumulation are single f32 operations in the reference's
// order. The released tensor-core reduction order is not observable from the source, so "the
// kernel's order" is the pinned one, not a claim about the hardware's.
//
// ARITHMETIC INTRINSICS. Every multiply, add and scale is an explicit _rn intrinsic so nvcc cannot
// contract a multiply-add: an FMA rounds once where the reference rounds twice.
//
// TRACE. ABI-uniform arguments, no event: no protocol scope describes wkv.

#include "m4_common_sm121.cuh"

#define SPARK_ENGRAM_WKV_WARPS 8u
#define SPARK_ENGRAM_WKV_THREADS (SPARK_ENGRAM_WKV_WARPS * 32u)

// One 32-column K block's dot product, in the shuffle tree's order. Lane 0's result is the block dot.
__device__ __forceinline__ float spark_engram_block_dot(const uint8_t *__restrict__ act_codes,
                                                       const uint8_t *__restrict__ weight_row,
                                                       uint32_t k_block, uint32_t lane) {
    const uint32_t column = k_block * SPARK_M4_QUANT_BLOCK + lane;
    float partial = __fmul_rn(spark_fp8_e4m3_decode(act_codes[column]),
                              spark_fp8_e4m3_decode(weight_row[column]));
    partial = __fadd_rn(partial, __shfl_down_sync(0xFFFFFFFFu, partial, 16u));
    partial = __fadd_rn(partial, __shfl_down_sync(0xFFFFFFFFu, partial, 8u));
    partial = __fadd_rn(partial, __shfl_down_sync(0xFFFFFFFFu, partial, 4u));
    partial = __fadd_rn(partial, __shfl_down_sync(0xFFFFFFFFu, partial, 2u));
    partial = __fadd_rn(partial, __shfl_down_sync(0xFFFFFFFFu, partial, 1u));
    return partial;
}

extern "C" __global__ void engram_wkv_sm121(
    const uint8_t *__restrict__ act_codes,     // [inner] E4M3 activation codes
    const uint8_t *__restrict__ act_scales,    // [inner / 32] E8M0 activation scales
    const uint8_t *__restrict__ weight,        // [out_width][inner] E4M3, checkpoint order
    const uint8_t *__restrict__ weight_scales, // [out_width / 32][inner / 32] E8M0
    __nv_bfloat16 *__restrict__ out,           // [out_width] the Linear's bf16 output
    const uint32_t out_width,
    const uint32_t inner,
    GpuTraceRing *__restrict__ trace_ring,     // ABI uniformity only; see TRACE above
    const uint32_t layer_id,
    const uint64_t step_seq) {
    (void)trace_ring;
    (void)layer_id;
    (void)step_seq;

    const uint32_t lane = threadIdx.x % 32u;
    const uint32_t warp = threadIdx.x / 32u;
    const uint32_t row = blockIdx.x * SPARK_ENGRAM_WKV_WARPS + warp;
    if (row >= out_width) {
        return;
    }
    const uint32_t k_blocks = inner / SPARK_M4_QUANT_BLOCK;
    const uint8_t *weight_row = weight + (uint64_t)row * (uint64_t)inner;
    const uint8_t *scale_row = weight_scales + (uint64_t)(row / SPARK_M4_QUANT_BLOCK) * k_blocks;

    float accumulator = 0.0f;
    for (uint32_t kb = 0; kb < k_blocks; ++kb) {
        const float block_dot = spark_engram_block_dot(act_codes, weight_row, kb, lane);
        if (lane == 0u) {
            const float act_scale = spark_ue8m0_decode(act_scales[kb]);
            const float weight_scale = spark_ue8m0_decode(scale_row[kb]);
            accumulator = __fadd_rn(accumulator, __fmul_rn(__fmul_rn(block_dot, act_scale), weight_scale));
        }
    }
    if (lane == 0u) {
        out[row] = __float2bfloat16_rn(accumulator);
    }
}

// The standalone final RMSNorm of the serving tail: out[row][c] = bf16((x * inv) * w[c]) with
// inv = 1 / sqrt(mean(x^2) + eps) over the row. This is the model's final_norm (the norm between the
// collapsed mHC residual and the LM head), which the engine currently runs on the host
// (crates/sparkflash-engine/src/forward/mod.rs:1092 -> crates/sparkflash-engine/src/linalg.rs:318),
// so c4c can put the last host op on the device.
//
// WHY A STANDALONE KERNEL. mega_mhc_sm121 computes an RMS-style norm internally for the residual
// seam, but it is not exported as a norm stage and its arithmetic is not the tail's: it normalises a
// different operand inside the mHC pass. m16's head already consumes a BF16 activation ABI, so the
// tail's norm output must be BF16 [rows][cols] and the released norm.weight is BF16 [cols].
//
// ARITHMETIC ORDER (this is the whole contract; it is not a tolerance class). The engine's
// linalg::rms_norm is the reference and it performs, per row, in this exact order:
//
//     sum  = sum over columns ascending of x[c] * x[c]     // f32, one rounding per multiply and add
//     mean = sum / cols                                    // f32 division
//     inv  = 1.0 / sqrt(mean + eps)                        // f32 add, f32 sqrt, f32 reciprocal
//     out  = bf16((x[c] * inv) * w[c])                     // f32 mul, f32 mul, one bf16 rounding
//
// This kernel reproduces those operations one for one. It uses the rounding intrinsics
// (__fmul_rn, __fadd_rn, __fsqrt_rn, __fdiv_rn) rather than plain operators so nvcc's default
// contraction cannot fuse the square-and-add into an FMA or replace the divide/sqrt with an
// approximate one: an FMA would round once where the engine rounds twice, and rsqrt would not be
// the engine's sqrt-then-reciprocal. The host twin is rmsnorm_ref.rs, which calls the engine's own
// linalg::rms_norm, so the reference is the engine code and not a second hand copy of it.
//
// ORDER OF OPERATIONS ACROSS COLUMNS. One thread owns one whole row and walks it in ascending column
// order. That is deliberately not a block or warp reduction: a tree reduction sums in a different
// order and would not be bit-identical to the engine's sequential sum. The row-serial order is what
// makes the BF16 result bit-exact, and it costs nothing at the tail's token counts (the tail is the
// decode path: tokens is small). The kernel is one launch, grid ceil(rows / 256) blocks of 256
// threads.
//
// LOADS. With cols a multiple of 8 (5120 in the released config) the row is read as 128-bit chunks -
// four u32 words = eight BF16 values, 16-byte aligned because cols * 2 is a multiple of 16 and the
// buffers are driver allocations. The load is explicit PTX (ld.global.nc.v4.u32) because the
// compiler otherwise narrows it to eight 16-bit loads once it sees only BF16 halves used: the
// pinned cubin's PTX has the v4 load, and the clang probe (no nvcc) takes the plain uint4
// dereference instead. The eight values are folded in ascending lane order. Any other width takes
// the scalar path; the odd-width test in the harness exercises it (513 columns). Output stores are
// scalar __nv_bfloat16 writes: the rounding contract is the conversion intrinsic, and packing eight
// of them into a vector would need bit punning for an eight-value tail that is not the bottleneck.
//
// INPUTS. x is [rows][cols] BF16. weight is [cols] BF16 - the released norm.weight, whose bytes
// decode exactly to the engine's f32 weight (bf16 -> f32 is exact), so the device multiplies by the
// same values the host does. eps is the config's rms_norm_eps cast to f32 (the released config is
// 1e-20, not 1e-6). rows and cols are u32; a zero row or column is a caller error, refused by the
// typed launcher.
//
// TRACE. The signature carries the trace ring, layer id and step sequence for ABI uniformity with
// the other kernels, but this kernel does not emit trace events: the protocol's 16 scope bits are
// all assigned (kernels/trace_globaltimer_sm121.cuh, specification 5.8) and none of them describes a
// tail norm, so recording under a dense or mHC scope would misattribute the event. A future scope
// would be a protocol revision; the arguments are here so that revision does not change the ABI.
//
// This file is compiled by build.rs's clang probe against cuda_stub.h on a CPU-only host, and by the
// pinned nvcc 13.0.3 to the cubin the baseline pins. It has no shared symbol with another kernel
// (the combined probe compiles every kernel in one translation unit), so every helper is
// spark_rmsnorm_-prefixed.

#include "m4_common_sm121.cuh"

// One 128-bit read of eight BF16 values. nvcc gets the explicit vector load; the clang syntax probe
// has no inline-asm validation for PTX and takes the equivalent alignment-trusted dereference.
__device__ __forceinline__ uint4 spark_rmsnorm_load128(const __nv_bfloat16 *pointer) {
#if defined(SPARKFLASH_CUDA_SYNTAX_STUB)
    return *reinterpret_cast<const uint4 *>(pointer);
#else
    // The pointer parameter is generic; cvta.to.global makes the state space explicit before the
    // vector load rather than relying on a generic address being numerically global.
    uint4 value;
    asm volatile(
        "{ .reg .u64 spark_rmsnorm_address;\n"
        "  cvta.to.global.u64 spark_rmsnorm_address, %4;\n"
        "  ld.global.nc.v4.u32 {%0, %1, %2, %3}, [spark_rmsnorm_address]; }"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(pointer));
    return value;
#endif
}

// The k-th BF16 value of a 128-bit chunk (k = 0..7), as the exact f32 the engine would hold after
// decoding the released BF16 operand.
__device__ __forceinline__ float spark_rmsnorm_chunk_value(const uint4 &chunk, int k) {
    const uint32_t words[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
    const uint32_t word = words[k >> 1];
    const unsigned short bits = (unsigned short)((word >> (16 * (k & 1))) & 0xFFFFu);
    return __bfloat162float(__ushort_as_bfloat16(bits));
}

// The engine's x*x + sum, one rounding per operation, ascending columns.
__device__ __forceinline__ float spark_rmsnorm_accumulate(float sum, float value) {
    return __fadd_rn(sum, __fmul_rn(value, value));
}

// The engine's (x * inv) * w order, one rounding per multiply.
__device__ __forceinline__ float spark_rmsnorm_scale(float value, float inv, float weight) {
    return __fmul_rn(__fmul_rn(value, inv), weight);
}

extern "C" __global__ void rmsnorm_sm121(
    const __nv_bfloat16 *__restrict__ x,       // [rows][cols] BF16 input
    const __nv_bfloat16 *__restrict__ weight,  // [cols] BF16 released norm.weight
    __nv_bfloat16 *__restrict__ out,           // [rows][cols] BF16 output
    const uint32_t rows,
    const uint32_t cols,
    const float eps,                           // config rms_norm_eps as f32
    GpuTraceRing *__restrict__ trace_ring,     // ABI uniformity only; see TRACE above
    const uint32_t layer_id,
    const uint64_t step_seq) {
    (void)trace_ring;
    (void)layer_id;
    (void)step_seq;

    const uint64_t row = (uint64_t)blockIdx.x * blockDim.x + (uint64_t)threadIdx.x;
    if (row >= (uint64_t)rows) {
        return;
    }
    const __nv_bfloat16 *x_row = x + row * (uint64_t)cols;
    __nv_bfloat16 *out_row = out + row * (uint64_t)cols;

    float sum = 0.0f;
    const bool vectorised = (cols & 7u) == 0u;
    if (vectorised) {
        const uint32_t count = cols >> 3u;
        for (uint32_t chunk = 0; chunk < count; ++chunk) {
            const uint4 values =
                spark_rmsnorm_load128(x_row + (uint64_t)chunk * 8u);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                sum = spark_rmsnorm_accumulate(sum, spark_rmsnorm_chunk_value(values, k));
            }
        }
    } else {
        for (uint32_t column = 0; column < cols; ++column) {
            sum = spark_rmsnorm_accumulate(sum, __bfloat162float(x_row[column]));
        }
    }

    const float mean = __fdiv_rn(sum, (float)cols);
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));

    if (vectorised) {
        const uint32_t count = cols >> 3u;
        for (uint32_t chunk = 0; chunk < count; ++chunk) {
            const uint4 values =
                spark_rmsnorm_load128(x_row + (uint64_t)chunk * 8u);
            const uint4 weights =
                spark_rmsnorm_load128(weight + (uint64_t)chunk * 8u);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float scaled = spark_rmsnorm_scale(
                    spark_rmsnorm_chunk_value(values, k),
                    inv,
                    spark_rmsnorm_chunk_value(weights, k));
                out_row[(uint64_t)chunk * 8u + (uint64_t)k] = __float2bfloat16_rn(scaled);
            }
        }
    } else {
        for (uint32_t column = 0; column < cols; ++column) {
            const float scaled = spark_rmsnorm_scale(
                __bfloat162float(x_row[column]),
                inv,
                __bfloat162float(weight[column]));
            out_row[column] = __float2bfloat16_rn(scaled);
        }
    }
}

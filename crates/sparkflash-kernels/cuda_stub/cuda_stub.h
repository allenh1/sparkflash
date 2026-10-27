// crates/sparkflash-kernels/cuda_stub/cuda_stub.h
//
// Minimal stand-in for the CUDA device headers, used ONLY by the clang syntax probe in
// crates/sparkflash-kernels/build.rs on hosts that have no CUDA toolkit. It lets clang's CUDA
// frontend parse kernels/*.cu (keywords, types, static_asserts, macro expansion) without an
// nvcc installation.
//
// This file is NOT an ABI reference and its helpers carry no semantics: __bfloat162float and
// __float2bfloat16_rn below return constants. The probe therefore proves that the kernel
// parses and type-checks; it says nothing about numerics, PTX or hardware behaviour. nvcc
// never includes this file (rocenante_oneshot_sm121.cuh only reaches it through
// SPARKFLASH_CUDA_SYNTAX_STUB).
#pragma once

// CUDA's crt/host_defines.h keyword definitions, reduced to the ones the kernels use.
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __forceinline__ inline __attribute__((always_inline))
#define __restrict__ __restrict
#define __align__(n) __attribute__((aligned(n)))

#include <stddef.h>
#include <stdint.h>

#define INFINITY __builtin_inff()

// cuda_bf16.h: __nv_bfloat16 is a 2-byte, 2-byte-aligned wrapper over the raw bits, and the
// conversion intrinsics are device functions.
struct __align__(2) __nv_bfloat16 {
    unsigned short __x;
};

__device__ __forceinline__ float __bfloat162float(__nv_bfloat16 value) {
    return (float)value.__x;
}

__device__ __forceinline__ __nv_bfloat16 __float2bfloat16_rn(float value) {
    __nv_bfloat16 out;
    out.__x = (unsigned short)((unsigned int)value >> 16);
    return out;
}

__device__ __forceinline__ __nv_bfloat16 __ushort_as_bfloat16(unsigned short bits) {
    __nv_bfloat16 out;
    out.__x = bits;
    return out;
}

// cuda_bf16.h: the raw-bits accessor the packed-store path uses (kernels/rmsnorm_sm121.cu).
__device__ __forceinline__ unsigned short __bfloat16_as_ushort(__nv_bfloat16 value) {
    return value.__x;
}

// Built-in variables and device runtime functions that CUDA declares in its headers. clang
// already knows __syncthreads() as a builtin, so it is deliberately absent here.
struct uint4 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
};

struct uint3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

struct uint2 {
    unsigned int x;
    unsigned int y;
};

struct float2 {
    float x;
    float y;
};

struct float4 {
    float x;
    float y;
    float z;
    float w;
};

// cuda_fp16.h: the half types and the two conversions the opt kernels use.
struct __align__(2) __half {
    unsigned short __x;
};

// The raw pair cuda_fp8.h's packed conversions return.
struct __align__(4) __half2_raw {
    unsigned short x;
    unsigned short y;
};

struct __align__(4) __half2 {
    __half x;
    __half y;

    __half2() = default;
    __device__ __forceinline__ explicit __half2(unsigned short raw) {
        x.__x = raw;
        y.__x = raw;
    }
    __device__ __forceinline__ __half2(const __half2_raw &raw) {
        x.__x = raw.x;
        y.__x = raw.y;
    }
};

__device__ __forceinline__ float __half2float(__half value) {
    return (float)value.__x;
}

__device__ __forceinline__ __half __ushort_as_half(unsigned short bits) {
    __half out;
    out.__x = bits;
    return out;
}

__device__ __forceinline__ __half __float2half(float value) {
    __half out;
    out.__x = (unsigned short)value;
    return out;
}

__device__ __forceinline__ float2 __half22float2(__half2 value) {
    float2 out;
    out.x = __half2float(value.x);
    out.y = __half2float(value.y);
    return out;
}

__device__ __forceinline__ __half2 __hmul2(__half2 a, __half2 b) {
    __half2 out;
    out.x.__x = (unsigned short)(a.x.__x * b.x.__x);
    out.y.__x = (unsigned short)(a.y.__x * b.y.__x);
    return out;
}

// CUDA's launch-bounds annotation is an attribute on the kernel; the syntax probe does not need it.
#define __launch_bounds__(...)

// vector_functions.h: make_uint4 is a device function, not a type, so the probe needs it declared
// too (kernels/rmsnorm_sm121.cu packs four bf16 pairs into one 128-bit store through it).
__device__ __forceinline__ uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) {
    uint4 out;
    out.x = x;
    out.y = y;
    out.z = z;
    out.w = w;
    return out;
}

__device__ __forceinline__ float2 make_float2(float x, float y) {
    float2 out;
    out.x = x;
    out.y = y;
    return out;
}

__device__ __forceinline__ uint2 make_uint2(unsigned int x, unsigned int y) {
    uint2 out;
    out.x = x;
    out.y = y;
    return out;
}

__device__ __forceinline__ float4 make_float4(float x, float y, float z, float w) {
    float4 out;
    out.x = x;
    out.y = y;
    out.z = z;
    out.w = w;
    return out;
}

__device__ __forceinline__ float fmaf(float a, float b, float c) {
    return a * b + c;
}

extern "C" __device__ const uint3 threadIdx;
extern "C" __device__ const uint3 blockIdx;
extern "C" __device__ const uint3 blockDim;
extern "C" __device__ const uint3 gridDim;

__device__ __forceinline__ void __threadfence(void) {}
__device__ __forceinline__ void __threadfence_system(void) {}
__device__ __forceinline__ void __nanosleep(unsigned int ns) { (void)ns; }

// Warp shuffle, used by the flash_mla indexer to reduce a 128-wide dot inside one warp and by the
// SwiGLU quantiser to reduce a 32-wide MX block's amax. clang's CUDA frontend carries the shuffles
// as builtins but only declares them through cuda_runtime.h, which this probe deliberately does
// not include. A bare declaration is enough for the -fsyntax-only probe; nvcc never sees this file.
__device__ float __shfl_down_sync(unsigned int mask, float value, unsigned int delta, int width = 32);
__device__ unsigned int __shfl_down_sync(unsigned int mask, unsigned int value, unsigned int delta, int width = 32);
__device__ float __shfl_xor_sync(unsigned int mask, float value, unsigned int lane_mask, int width = 32);

// The shared trace header's macros call atomicAdd and __ffsll, which CUDA's own headers
// provide. Declaring them here lets the ABI probe compile the macros with the trace enabled,
// which is the only way this host can type-check a kernel's trace call at all. The return
// values are meaningless; the probe discards them.
__device__ unsigned int atomicAdd(unsigned int *address, unsigned int val);
__device__ float atomicAdd(float *address, float val);
__device__ unsigned long long atomicAdd(unsigned long long *address, unsigned long long val);
__device__ int __ffsll(long long value);

// cuda_fp8.h / cuda_fp4.h expose their conversions as plain unsigned char values. The M4 kernels
// store FP8 as uint8_t and call these intrinsics under nvcc; the clang probe instead compiles
// the bit-arithmetic fallback in m4_common_sm121.cuh, so no fp8 type is needed here.

// Device math and bit-cast intrinsics the M4 kernels use. The math ones carry no semantics: the
// probe is -fsyntax-only, so nothing here is ever executed or numerically checked.
__device__ __forceinline__ unsigned int __float_as_uint(float value) {
    return (unsigned int)value;
}

__device__ __forceinline__ float __uint_as_float(unsigned int bits) {
    return (float)bits;
}

__device__ __forceinline__ float expf(float value) {
    (void)value;
    return 1.0f;
}

// The __-prefixed fast intrinsics and the C99 float math functions the SwiGLU quantiser uses.
// Like expf above they carry no semantics: -fsyntax-only never calls them. They are declared
// because the build script's probe now covers kernels/swiglu_mxfp8_quant_sm121.cu, which CUDA's
// own headers would normally provide.
__device__ __forceinline__ float __expf(float value) {
    (void)value;
    return 1.0f;
}

__device__ __forceinline__ float fabsf(float value) {
    return value < 0.0f ? -value : value;
}

__device__ __forceinline__ float fmaxf(float a, float b) {
    return a > b ? a : b;
}

// The S1 compressor pooling's empty-group test. CUDA's math.h provides this as a builtin; the
// stub declares it so the clang probe can type-check `!isfinite(max)`. No semantics are needed.
__device__ __forceinline__ bool isfinite(float value) {
    return __builtin_isfinite(value);
}

__device__ __forceinline__ float fminf(float a, float b) {
    return a < b ? a : b;
}

__device__ __forceinline__ float sqrtf(float value) {
    (void)value;
    return value;
}

__device__ __forceinline__ float rsqrtf(float value) {
    return value;
}

__device__ __forceinline__ float logf(float value) {
    return value;
}

template <typename T>
__device__ __forceinline__ T min(T a, T b) {
    return a < b ? a : b;
}

template <typename T>
__device__ __forceinline__ T max(T a, T b) {
    return a > b ? a : b;
}

__device__ __forceinline__ float frexpf(float value, int *exponent) {
    *exponent = 0;
    return value;
}

__device__ __forceinline__ float __shfl_sync(unsigned int mask, float value, int lane, int width = 32) {
    (void)mask;
    (void)lane;
    (void)width;
    return value;
}

__device__ __forceinline__ unsigned int __shfl_sync(unsigned int mask, unsigned int value, int lane, int width = 32) {
    (void)mask;
    (void)lane;
    (void)width;
    return value;
}

__device__ __forceinline__ unsigned int __shfl_up_sync(unsigned int mask, unsigned int value, unsigned int delta, int width = 32) {
    (void)mask;
    (void)delta;
    (void)width;
    return value;
}

__device__ __forceinline__ bool __all_sync(unsigned int mask, bool predicate) {
    (void)mask;
    return predicate;
}

__device__ __forceinline__ bool __syncthreads_or(bool predicate) {
    return predicate;
}

__device__ __forceinline__ unsigned int __reduce_add_sync(unsigned int mask, unsigned int value) {
    (void)mask;
    return value;
}

__device__ __forceinline__ unsigned int __ballot_sync(unsigned int mask, bool predicate) {
    (void)mask;
    return predicate ? 0xffffffffu : 0u;
}

__device__ __forceinline__ unsigned int __match_any_sync(unsigned int mask, unsigned int value) {
    (void)value;
    return mask;
}

__device__ __forceinline__ int __popc(unsigned int value) {
    return __builtin_popcount(value);
}

__device__ __forceinline__ int __ffs(int value) {
    return __builtin_ffs(value);
}

__device__ __forceinline__ unsigned int atomicOr(unsigned int *address, unsigned int value) {
    const unsigned int old = *address;
    *address |= value;
    return old;
}

// cuda_fp8.h's packed conversion path (the draft and MoE kernels): 16-bit storage in, a raw half
// pair out.
using __nv_fp8x2_storage_t = unsigned short;
using __nv_fp4x2_storage_t = unsigned char;
enum __nv_fp8_interpretation_t { __NV_E4M3 = 0, __NV_E5M2 = 1, __NV_E2M1 = 2 };
__device__ __forceinline__ __half2_raw __nv_cvt_fp8x2_to_halfraw2(
    __nv_fp8x2_storage_t value,
    __nv_fp8_interpretation_t interpretation
) {
    (void)interpretation;
    return __half2_raw{value, value};
}

__device__ __forceinline__ __half2_raw __nv_cvt_fp4x2_to_halfraw2(
    __nv_fp4x2_storage_t value,
    __nv_fp8_interpretation_t interpretation
) {
    (void)interpretation;
    return __half2_raw{value, value};
}


// The device router's score function: sqrt(softplus(z)) with PyTorch's threshold. nvcc provides
// expf/log1pf/sqrtf and the two rounding intrinsics; the clang probe only needs the declarations.
__device__ __forceinline__ float log1pf(float value) {
    (void)value;
    return 0.0f;
}

__device__ __forceinline__ float __fmul_rn(float a, float b) {
    return a * b;
}

__device__ __forceinline__ float __fadd_rn(float a, float b) {
    return a + b;
}

// The final RMSNorm kernel's correctly-rounded sqrt and divide (CUDA device_functions.h). The
// declarations exist only so the -fsyntax-only probe can type-check the kernel; they carry no
// semantics here and nvcc uses its own definitions.
__device__ __forceinline__ float __fsqrt_rn(float value) {
    (void)value;
    return value;
}

__device__ __forceinline__ float __fdiv_rn(float a, float b) {
    return a / b;
}

__device__ __forceinline__ float __fsub_rn(float a, float b) {
    return a - b;
}

__device__ __forceinline__ float erff(float value) {
    return value;
}

__device__ __forceinline__ float powf(float base, float exponent) {
    (void)base;
    (void)exponent;
    return 1.0f;
}

__device__ __forceinline__ float cosf(float value) {
    (void)value;
    return 1.0f;
}

__device__ __forceinline__ float sinf(float value) {
    (void)value;
    return 0.0f;
}

__device__ __forceinline__ float exp2f(float value) {
    (void)value;
    return 1.0f;
}

// The decode GEMV's contracted multiply-add and its warp barrier. clang treats __fmaf_rn and
// __syncwarp as CUDA builtins only when it is compiling against the real device headers, which
// the -nocudainc probe is not, so the stub declares them like the other intrinsics above.
__device__ __forceinline__ float __fmaf_rn(float a, float b, float c) {
    return a * b + c;
}

__device__ __forceinline__ unsigned int __activemask(void) {
    return 0xffffffffu;
}

template <typename T>
__device__ __forceinline__ unsigned long long __cvta_generic_to_shared(T *pointer) {
    return reinterpret_cast<unsigned long long>(pointer);
}

// Read-only load. CUDA declares one overload per vector type; the syntax probe only needs a
// declaration that type-checks the dereference.
template <typename T>
__device__ __forceinline__ T __ldg(const T *address) {
    return *address;
}

template <typename T>
__device__ __forceinline__ T __ldcg(const T *address) {
    return *address;
}

template <typename T>
__device__ __forceinline__ T __ldcs(const T *address) {
    return *address;
}

__device__ __forceinline__ void __syncwarp(unsigned int mask = 0xffffffffu) {
    (void)mask;
}

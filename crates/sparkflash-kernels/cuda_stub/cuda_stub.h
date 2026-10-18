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

// Built-in variables and device runtime functions that CUDA declares in its headers. clang
// already knows __syncthreads() as a builtin, so it is deliberately absent here.
struct uint3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

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

__device__ __forceinline__ float fminf(float a, float b) {
    return a < b ? a : b;
}

__device__ __forceinline__ float sqrtf(float value) {
    (void)value;
    return value;
}


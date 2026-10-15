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

// The shared trace header's macros call atomicAdd and __ffsll, which CUDA's own headers
// provide. Declaring them here lets the ABI probe compile the macros with the trace enabled,
// which is the only way this host can type-check a kernel's trace call at all. The return
// values are meaningless; the probe discards them.
__device__ unsigned int atomicAdd(unsigned int *address, unsigned int val);
__device__ int __ffsll(long long value);

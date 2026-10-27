// clang-probe stand-in for CUDA's <cuda_bf16.h>.
//
// The kernels include this header directly; on a host without a CUDA toolkit the build's clang
// probe puts this directory on the include path and sets SPARKFLASH_CUDA_SYNTAX_STUB. The
// definitions below are syntax-only, exactly like cuda_stub.h.
#pragma once
#include "cuda_stub.h"

struct __align__(4) __nv_bfloat162 {
    __nv_bfloat16 x;
    __nv_bfloat16 y;
};

__device__ __forceinline__ __nv_bfloat162 __floats2bfloat162_rn(float low, float high) {
    __nv_bfloat162 out;
    out.x = __float2bfloat16_rn(low);
    out.y = __float2bfloat16_rn(high);
    return out;
}

__device__ __forceinline__ float2 __bfloat1622float2(__nv_bfloat162 value) {
    float2 out;
    out.x = __bfloat162float(value.x);
    out.y = __bfloat162float(value.y);
    return out;
}

__device__ __forceinline__ float2 __bfloat1622float2(__nv_bfloat16 value) {
    float2 out;
    out.x = __bfloat162float(value);
    out.y = out.x;
    return out;
}

// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// kernels/b12x_common_sm121.cuh
//
// The four device helpers the b12x token kernels had byte-identical copies of: the 16-byte
// streaming load, the round-to-nearest warp sum, the byte and u16 accessors into a 16-byte
// register window, and the E4M3x2 widening. Each copy is the same code under a different prefix
// (spark_m4_dense_, spark_woa_, spark_m4_gemv_, spark_hw_gemv_); one home is enough.

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

// 16 bytes through the read-only path. Not an L1 line: these are one-pass streams.
__device__ __forceinline__ uint4 spark_b12x_load128(const void *p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// A warp's tree reduction with explicit round-to-nearest adds.
__device__ __forceinline__ float spark_b12x_warp_sum(float value) {
    value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 16u));
    value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 8u));
    value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 4u));
    value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 2u));
    return __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 1u));
}

// Byte i of a 16-byte register window.
__device__ __forceinline__ uint8_t spark_b12x_byte(uint4 v, uint32_t i) {
    const uint32_t word = i >> 2;
    const uint32_t shift = (i & 3u) * 8u;
    const uint32_t w = word == 0u ? v.x : word == 1u ? v.y : word == 2u ? v.z : v.w;
    return (uint8_t)((w >> shift) & 0xffu);
}

// u16 i of a 16-byte register window.
__device__ __forceinline__ uint16_t spark_b12x_u16_pair(uint4 v, uint32_t i) {
    const uint32_t word = i < 4u ? v.x : i < 8u ? v.y : i < 12u ? v.z : v.w;
    return (uint16_t)((word >> ((i & 3u) * 8u)) & 0xffffu);
}

// Two E4M3 codes (the low byte first) widened to f32 through f16, which holds every code exactly.
__device__ __forceinline__ void spark_b12x_fp8x2_to_f32(uint16_t two, float &lo, float &hi) {
    asm("{\n\t"
        ".reg .b32 h2;\n\t"
        ".reg .b16 lo, hi;\n\t"
        "cvt.rn.f16x2.e4m3x2 h2, %2;\n\t"
        "mov.b32 {lo, hi}, h2;\n\t"
        "cvt.f32.f16 %0, lo;\n\t"
        "cvt.f32.f16 %1, hi;\n\t"
        "}"
        : "=f"(lo), "=f"(hi)
        : "h"(two));
}

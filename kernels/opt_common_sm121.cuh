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

// kernels/opt_common_sm121.cuh
//
// The device helpers the opt_* images and the DeepSeek layer seams had copied: the E4M3x4
// widening, the two warp reductions, the bf16 round and sigmoid, the mixed-precision FMA pair and
// dot, the streaming 16-byte load (cached and no-allocate), the transposed ldmatrix, and the two
// bf16 narrowing bit patterns. Each body below was byte-identical in every copy (name and local
// parameter names aside), so the copies said the same thing once per translation unit.
//
// This header is included by the image that used the helper and its call sites keep their names
// through the defines at the bottom of each file; the code the compiler sees is the same.

#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Four E4M3 codes (low 16 bits first) widened through f16, which holds every code exactly.
__device__ __forceinline__ void opt_e4m3x4(uint32_t w, float *f) {
    uint32_t h01, h23;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h01) : "h"((uint16_t)(w & 0xffffu)));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h23) : "h"((uint16_t)(w >> 16)));
    const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&h01));
    const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&h23));
    f[0] = a.x;
    f[1] = a.y;
    f[2] = b.x;
    f[3] = b.y;
}

// A warp's xor butterfly sum: every lane holds the total.
__device__ __forceinline__ float opt_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// The same fold with explicit round-to-nearest adds, which is the reference order the seams
// reproduce.
__device__ __forceinline__ float opt_warp_sum_rn(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(value, __shfl_down_sync(0xFFFFFFFFu, value, offset));
    }
    return value;
}

// The reference sigmoid, evaluated at x >= 0 and e^x / (1 + e^x) below.
__device__ __forceinline__ float opt_sigmoid(float value) {
    if (value >= 0.0f) {
        return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-value)));
    }
    const float exponential = expf(value);
    return __fdiv_rn(exponential, __fadd_rn(1.0f, exponential));
}

// One f32 -> bf16 round-to-nearest-even, kept in an f32 register with the low half zero.
__device__ __forceinline__ float opt_bf16_round(float value) {
    const uint32_t bits = __float_as_uint(value);
    if (value != value) {
        return __uint_as_float((((bits >> 16) | 0x0040u)) << 16);
    }
    const uint32_t round = 0x7FFFu + ((bits >> 16) & 1u);
    return __uint_as_float(((bits + round) >> 16) << 16);
}

// The two halves of a packed bf16 pair, as the f32 register their bits spell.
__device__ __forceinline__ float opt_bf16lo(uint32_t w) { return __uint_as_float(w << 16); }

__device__ __forceinline__ float opt_bf16hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// Two bf16 pairs multiplied and accumulated in f32. The product is exact in f32, so each step
// rounds exactly as fmaf on the widened values does, without the widening.
__device__ __forceinline__ float opt_fma2_bf16(uint32_t a, uint32_t b, float c) {
    asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(c) : "h"((uint16_t)(a & 0xffffu)), "h"((uint16_t)(b & 0xffffu)));
    asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(c) : "h"((uint16_t)(a >> 16)), "h"((uint16_t)(b >> 16)));
    return c;
}

__device__ __forceinline__ float opt_fma2_f16(uint32_t a, uint32_t b, float c) {
    asm("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(c) : "h"((uint16_t)(a & 0xffffu)), "h"((uint16_t)(b & 0xffffu)));
    asm("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(c) : "h"((uint16_t)(a >> 16)), "h"((uint16_t)(b >> 16)));
    return c;
}

// Eight bf16 weights (four packed pairs) against their activations.
__device__ __forceinline__ float opt_dot8_bf16(uint4 w, uint4 h, float acc) {
    const uint32_t a[4] = {w.x, w.y, w.z, w.w};
    const uint32_t b[4] = {h.x, h.y, h.z, h.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        acc = fmaf(opt_bf16lo(a[k]), opt_bf16lo(b[k]), acc);
        acc = fmaf(opt_bf16hi(a[k]), opt_bf16hi(b[k]), acc);
    }
    return acc;
}

__device__ __forceinline__ float opt_dot8_fma(uint4 w, uint4 h, float acc) {
    acc = opt_fma2_bf16(w.x, h.x, acc);
    acc = opt_fma2_bf16(w.y, h.y, acc);
    acc = opt_fma2_bf16(w.z, h.z, acc);
    acc = opt_fma2_bf16(w.w, h.w, acc);
    return acc;
}

// 16 bytes read once: no L1 line, so the stream a lane reads is not turned into a cache fill.
__device__ __forceinline__ uint4 opt_ldw(const void *p) {
    uint4 v;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 opt_ldw_cs(const void *p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// ldmatrix's transposed x4 form: four 8x8 b16 tiles from one shared-memory address.
__device__ __forceinline__ void opt_ldsm4_t(uint32_t addr, uint32_t &r0, uint32_t &r1,
                                             uint32_t &r2, uint32_t &r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

// A f32 as the bf16 bits the routed combine rounds to, NaN keeping its payload.
__device__ __forceinline__ uint32_t opt_round_slot_bits(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return (bits >> 16) | 0x0040u;
    }
    return (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
}

__device__ __forceinline__ uint16_t opt_narrow_bits(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (uint16_t)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

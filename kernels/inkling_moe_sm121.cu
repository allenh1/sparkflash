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

// kernels/inkling_moe_sm121.cu
//
// Inkling-2-Small's routed-experts arithmetic: the two weight formats the release uses and its
// sigmoid router. The shared plan, grouped GEMM and combine kernels stay in their own images.
//
// The formats differ from DeepSeek's by what rides on a scale:
//   * layers 0-39 are NVFP4 (ModelOpt): packed E2M1 nibbles, one E4M3 multiplier a 16 columns,
//     and a per-expert scale_2 = amax / (6 * outer_scale_denominator);
//   * layers 40-47 are MXFP8: an E4M3 byte a column, one ue8m0 power of two a 32 columns, and
//     no per-expert factor.
// Because both scale factors are per row (or per block) and the product is linear, an expert's
// scale_2 is applied to that expert's output: out = scale_2 * (W x). That is what lets one GEMM
// body serve every expert instead of a kernel per amax.
//
// The decoders are copies of quant_decode_sm121.cuh under this pack's names. Including that
// header here would collide in the combined probe, which already pulls it in through
// m4_common_sm121.cuh, and m4_common is sha256-pinned by the baseline so it cannot grow a guard.
// The bodies are identical: E4M3FN (bias 7, no infinity, NaN at 0x7F/0xFF) and the sixteen-code
// E2M1 table.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "inkling_softcap_sm121.cuh"
#include "opt_mma_bf16_sm121.cuh"
#include "opt_mma_common_sm121.cuh"
#include "opt_pdl_sm121.cuh"

#ifndef INKLING_MOE_BLOCK
#define INKLING_MOE_BLOCK 256
#endif

__device__ __forceinline__ float inkling_e4m3(unsigned char bits) {
    const unsigned int b = (unsigned int)bits;
    const unsigned int sign = (b & 0x80u) << 24;
    const unsigned int exponent = (b >> 3) & 0x0Fu;
    const unsigned int mantissa = b & 0x07u;
    const float power = __uint_as_float((exponent + 120u) << 23);
    const float normal = power * (1.0f + (float)mantissa * 0.125f);
    const float subnormal = (float)mantissa * 0.001953125f;
    const float magnitude = exponent == 0u ? subnormal : normal;
    const unsigned int finite = __float_as_uint(magnitude) | sign;
    return __uint_as_float((b & 0x7Fu) == 0x7Fu ? (sign | 0x7FC00000u) : finite);
}

__device__ __forceinline__ float inkling_e2m1(unsigned char nibble) {
    const unsigned int b = (unsigned int)nibble;
    const unsigned int sign = (b & 0x8u) << 28;
    const unsigned int shift = (b << 2) & 0x1Cu;
    const float magnitude = (float)((0xC8643210u >> shift) & 0xFu) * 0.5f;
    return __uint_as_float(__float_as_uint(magnitude) | sign);
}

// A ue8m0 code is the power of two 2^(code - 127), so the biased exponent is the code itself.
__device__ __forceinline__ float inkling_ue8m0(unsigned char code) {
    return __uint_as_float((unsigned int)code << 23);
}

// Warp-wide sum of one value a lane, then a block-wide reduction through shared memory.
__device__ __forceinline__ float inkling_moe_row_sum(float value, float* scratch) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(value, __shfl_down_sync(0xFFFFFFFFu, value, offset));
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = (blockDim.x + 31) >> 5;
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();
    float total = threadIdx.x < (unsigned int)warps ? scratch[threadIdx.x] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            total = __fadd_rn(total, __shfl_down_sync(0xFFFFFFFFu, total, offset));
        }
    }
    return total;
}

// One NVFP4 row's dot product with x, before the expert's scale_2: every thread walks the columns
// congruent to its index in order, and the block sums the threads' partials. Both NVFP4 entries
// below call this one body, so a row's bits do not depend on which entry computed it.
__device__ __forceinline__ float inkling_nvfp4_row(
    const unsigned char* __restrict__ row_codes,
    const unsigned char* __restrict__ row_scales,
    const float* __restrict__ x,
    int columns,
    float* scratch) {
    float sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const unsigned char byte = row_codes[column >> 1];
        const unsigned char nibble = (column & 1) ? (byte >> 4) : (byte & 0x0F);
        const float weight = inkling_e2m1(nibble);
        const float block = inkling_e4m3(row_scales[column >> 4]);
        sum = __fadd_rn(sum, __fmul_rn(__fmul_rn(weight, block), x[column]));
    }
    return inkling_moe_row_sum(sum, scratch);
}

// One MXFP8 row's dot product with x, the MXFP8 entries' shared body.
__device__ __forceinline__ float inkling_mxfp8_row(
    const unsigned char* __restrict__ row_codes,
    const unsigned char* __restrict__ row_scales,
    const float* __restrict__ x,
    int columns,
    int block,
    float* scratch) {
    float sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float weight = inkling_e4m3(row_codes[column]);
        const float scale = inkling_ue8m0(row_scales[column / block]);
        sum = __fadd_rn(sum, __fmul_rn(__fmul_rn(weight, scale), x[column]));
    }
    return inkling_moe_row_sum(sum, scratch);
}

// One expert's NVFP4 matrix-vector product: one block a row, out[row] = scale_2 * (W x).
extern "C" __global__ void __launch_bounds__(INKLING_MOE_BLOCK) inkling_nvfp4_expert_matvec(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int columns,
    float scale_2,
    const int* __restrict__ plan) {
    OPT_PDL_ENTRY();
    const int row = blockIdx.x;
    if (row >= rows || columns <= 0) {
        return;
    }
    // A chunk selects its own experts a (row, band slot) pair, so one launch carries every pair
    // that picked this expert: `plan` holds those pairs and blockIdx.y indexes it.
    //
    // The two offsets are not the same index and this is where they part: the *input* row of the
    // gate/up projection is the token's own row of the stream, while the *output* goes to the
    // pair's band slot. The down projection's input is that band slot instead, which is why the
    // plan carries both numbers rather than one. A decode step passes no plan, which leaves one
    // pair and the launch's own buffer - the shape this entry had before a chunk existed.
    const int entry = blockIdx.y;
    const int input_row = plan != nullptr ? plan[2 * entry] : entry;
    const int output_slot = plan != nullptr ? plan[2 * entry + 1] : entry;
    x += (uint64_t)input_row * columns;
    out += (uint64_t)output_slot * rows;
    const int packed = (columns + 1) / 2;
    const int scale_columns = (columns + 15) / 16;
    __shared__ float scratch[INKLING_MOE_BLOCK / 32 + 1];
    const float total = inkling_nvfp4_row(
        codes + (uint64_t)row * packed, scales + (uint64_t)row * scale_columns, x, columns, scratch);
    if (threadIdx.x == 0) {
        out[row] = __fmul_rn(total, scale_2);
    }
}

// The decode band's grouped walk. The row-a-block body above spends ~30 instructions a weight on
// byte loads and bit-twiddled decodes, so it is issue-bound at a third of the DRAM rate. Here one
// physical thread carries sixteen of that body's threads: logical thread t = 16 q + i (i < 16)
// of a row's 256 lives in physical lane q of a half-warp, so its columns t + 256 k form one run
// of sixteen adjacent columns a step - one 8-byte (NVFP4) or 16-byte (MXFP8) load of codes, one
// scale code, and the hardware's pair converts for the codes.
//
// The arithmetic is still inkling_nvfp4_row's / inkling_mxfp8_row's, term for term: logical
// thread t sums its columns in ascending k, each term __fmul_rn(__fmul_rn(weight, scale), x),
// and the row's 256 partials meet in inkling_moe_row_sum's tree. That tree is replayed exactly:
// a logical warp's 32 lanes are two physical lanes (q even: lanes 0-15, q odd: 16-31), so its
// shuffle-down offsets 16, 8, 4, 2, 1 become one exchange between the pair and then additions in
// registers in the same pairing; the eight warp sums then go through the block tree's 32-lane
// pass, whose first two levels add the +0.0 of the lanes past the eighth warp, replayed here as
// the same two additions. The codes decode to the same floats: E2M1 and E4M3 values are exact
// in f16, so cvt to f16 and then to f32 is the table's value, and the E4M3 *scale* still comes
// through inkling_e4m3 (a table the block fills with it), so a NaN scale keeps its bits too.
// A row a half-warp, so a block of INKLING_MOE_BLOCK physical threads computes 16 rows of a slot.
#define INKLING_GROUP 16
#define INKLING_GROUP_ROWS (INKLING_MOE_BLOCK / INKLING_GROUP)

// Eight E2M1 codes (one 32-bit word, low nibble first) as eight floats, exact.
__device__ __forceinline__ void inkling_e2m1x8(unsigned int word, float (&out)[8]) {
    unsigned int pairs[4];
    asm("{\n\t.reg .b8 c0, c1, c2, c3;\n\t"
        "mov.b32 {c0, c1, c2, c3}, %4;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, c0;\n\t"
        "cvt.rn.f16x2.e2m1x2 %1, c1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %2, c2;\n\t"
        "cvt.rn.f16x2.e2m1x2 %3, c3;\n\t}"
        : "=r"(pairs[0]), "=r"(pairs[1]), "=r"(pairs[2]), "=r"(pairs[3])
        : "r"(word));
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const unsigned short lo = (unsigned short)(pairs[p] & 0xFFFFu);
        const unsigned short hi = (unsigned short)(pairs[p] >> 16);
        float a, b;
        asm("cvt.f32.f16 %0, %1;" : "=f"(a) : "h"(lo));
        asm("cvt.f32.f16 %0, %1;" : "=f"(b) : "h"(hi));
        out[2 * p] = a;
        out[2 * p + 1] = b;
    }
}

// Four E4M3 codes (one 32-bit word, low byte first) as four floats: the pair convert, exact for
// every finite code, with a code of 0x7F/0xFF given inkling_e4m3's NaN bits.
__device__ __forceinline__ void inkling_e4m3x4(unsigned int word, float (&out)[4]) {
    unsigned int pairs[2];
    asm("{\n\t.reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, %2;\n\t"
        "cvt.rn.f16x2.e4m3x2 %0, lo;\n\t"
        "cvt.rn.f16x2.e4m3x2 %1, hi;\n\t}"
        : "=r"(pairs[0]), "=r"(pairs[1])
        : "r"(word));
    const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&pairs[0]));
    const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&pairs[1]));
    out[0] = f0.x;
    out[1] = f0.y;
    out[2] = f1.x;
    out[3] = f1.y;
    // A byte whose low seven bits are all set is the format's NaN.
    const unsigned int marked = (word & 0x7F7F7F7Fu) ^ 0x7F7F7F7Fu;
    if (__builtin_expect(((marked - 0x01010101u) & ~marked & 0x80808080u) != 0u, 0)) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            out[j] = inkling_e4m3((unsigned char)(word >> (8 * j)));
        }
    }
}

// One row's sixteen partials of a physical lane through inkling_moe_row_sum's tree; the row's
// total lands in physical lane 0 of the half-warp (the other lanes return garbage).
__device__ __forceinline__ float inkling_group_row_sum(float (&v)[INKLING_GROUP]) {
    const int q = threadIdx.x & (INKLING_GROUP - 1);
    // Offset 16 of the logical warp: lane l (< 16) adds lane l + 16, which is the odd partner's
    // partial i = l. The even lane keeps the sums; the odd lane's become don't-cares.
    float a[INKLING_GROUP];
#pragma unroll
    for (int i = 0; i < INKLING_GROUP; ++i) {
        a[i] = __fadd_rn(v[i], __shfl_down_sync(0xFFFFFFFFu, v[i], 1));
    }
    // Offsets 8, 4, 2, 1, lane l adding lane l + offset, all inside the even lane.
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
#pragma unroll
        for (int i = 0; i < offset; ++i) {
            a[i] = __fadd_rn(a[i], a[i + offset]);
        }
    }
    // a[0] of even lane q is warp q/2's sum. The block tree: lanes 0..7 hold the warp sums and
    // lanes 8..31 zero; offsets 16 and 8 add those zeros, then 4, 2, 1 pair lane l with l + offset.
    float w[8];
#pragma unroll
    for (int warp = 0; warp < 8; ++warp) {
        w[warp] = __shfl_sync(0xFFFFFFFFu, a[0], (threadIdx.x & ~(INKLING_GROUP - 1)) + 2 * warp);
    }
#pragma unroll
    for (int warp = 0; warp < 8; ++warp) {
        w[warp] = __fadd_rn(__fadd_rn(w[warp], 0.0f), 0.0f);
    }
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
#pragma unroll
        for (int i = 0; i < offset; ++i) {
            w[i] = __fadd_rn(w[i], w[i + offset]);
        }
    }
    (void)q;
    return w[0];
}

// One E4M3 scale byte via hardware pair convert, exact for every finite code and matching
// inkling_e4m3's NaN bits for 0x7F/0xFF (no per-block shared LUT fill needed).
__device__ __forceinline__ float inkling_e4m3_hw(unsigned char bits) {
    if ((bits & 0x7Fu) == 0x7Fu) {
        return __uint_as_float((((unsigned int)bits & 0x80u) << 24) | 0x7FC00000u);
    }
    const unsigned short pair = (unsigned short)bits;
    unsigned int f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"(pair));
    const unsigned short lo = (unsigned short)(f16x2 & 0xFFFFu);
    float out;
    asm("cvt.f32.f16 %0, %1;" : "=f"(out) : "h"(lo));
    return out;
}

// Pre-decode one E4M3 scale byte into a duplicated FP16x2 pair for inkling_nvfp4_decode8_scaled.
__device__ __forceinline__ __half2 inkling_e4m3_scale_f16x2(unsigned char sb) {
    const unsigned short sp = (unsigned short)sb | ((unsigned short)sb << 8);
    unsigned int sc_f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sc_f16x2) : "h"(sp));
    return *reinterpret_cast<const __half2*>(&sc_f16x2);
}

// Eight E2M1 codes (one 32-bit word, low nibble first) multiplied by a pre-decoded E4M3 scale
// via hardware cvt + exact FP16x2 __hmul2 -> eight floats. Exact for every finite E4M3 scale
// (max product 6 * 448 = 2688 << 65504, min nonzero 2^-10 >> 2^-14, <= 6 mantissa bits <= 11);
// NaN scale bytes (0x7F/0xFF) preserve inkling_e4m3's NaN bits.
__device__ __forceinline__ void inkling_nvfp4_decode8_scaled(unsigned int word, __half2 sc2,
                                                             unsigned char sb,
                                                             float (&scaled)[8]) {
    if (__builtin_expect((sb & 0x7Fu) == 0x7Fu, 0)) {
        const float nanv = __uint_as_float((((unsigned int)sb & 0x80u) << 24) | 0x7FC00000u);
        float tmp[8];
        inkling_e2m1x8(word, tmp);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            scaled[j] = __fmul_rn(tmp[j], nanv);
        }
        return;
    }
    unsigned int pairs[4];
    asm("{\n\t.reg .b8 c0, c1, c2, c3;\n\t"
        "mov.b32 {c0, c1, c2, c3}, %4;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, c0;\n\t"
        "cvt.rn.f16x2.e2m1x2 %1, c1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %2, c2;\n\t"
        "cvt.rn.f16x2.e2m1x2 %3, c3;\n\t}"
        : "=r"(pairs[0]), "=r"(pairs[1]), "=r"(pairs[2]), "=r"(pairs[3])
        : "r"(word));
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const __half2 prod = __hmul2(*reinterpret_cast<const __half2*>(&pairs[p]), sc2);
        const float2 f2 = __half22float2(prod);
        scaled[2 * p] = f2.x;
        scaled[2 * p + 1] = f2.y;
    }
}

// 16-lane half-warp sum for one row's dot product: identical across band, items and pairs so
// 1-row solo decode, 2..8-row verify/fused lanes and >8-row chunk walks agree bit for bit.
__device__ __forceinline__ float inkling_halfwarp_sum(float s) {
    s += __shfl_down_sync(0xFFFFFFFFu, s, 8);
    s += __shfl_down_sync(0xFFFFFFFFu, s, 4);
    s += __shfl_down_sync(0xFFFFFFFFu, s, 2);
    s += __shfl_down_sync(0xFFFFFFFFu, s, 1);
    return s;
}

// Emit one row (or fused interleaved gate+up softcapped SwiGLU pair within a warp: lane 0 holds
// gate = row 2*c and lane 16 holds up = row 2*c+1).
template <bool kSwiGlu>
__device__ __forceinline__ void inkling_moe_emit(float total, float* __restrict__ out, int slot,
                                                 int row, int rows, int q, float gate_cap,
                                                 float up_cap) {
    if constexpr (kSwiGlu) {
        const float up_val = __shfl_down_sync(0xFFFFFFFFu, total, 16);
        if ((threadIdx.x & 31) == 0 && (row + 1) < rows) {
            out[(uint64_t)slot * (rows >> 1) + (row >> 1)] =
                inkling_softcapped_swiglu_value(total, up_val, gate_cap, up_cap);
        }
    } else {
        if (q == 0 && row < rows) {
            out[(uint64_t)slot * rows + row] = total;
        }
    }
}

// One-slot NVFP4 body shared by band and items (count == 1): zero SMEM, zero barriers.
template <bool kSwiGlu>
__device__ __forceinline__ void inkling_nvfp4_one_slot(
    const unsigned char* __restrict__ row_codes, const unsigned char* __restrict__ row_scales,
    const float* __restrict__ xr, float* __restrict__ out, int slot, int row, int rows,
    int columns, int q, float s2, float gate_cap, float up_cap) {
    float acc = 0.0f;
#pragma unroll 2
    for (int base = INKLING_GROUP * q; base < columns; base += INKLING_MOE_BLOCK) {
        const uint2 word = *reinterpret_cast<const uint2*>(row_codes + (base >> 1));
        const unsigned char sb = row_scales[base >> 4];
        const __half2 sc2 = inkling_e4m3_scale_f16x2(sb);
        const float4* __restrict__ in = reinterpret_cast<const float4*>(xr + base);
        float sc[8];
        inkling_nvfp4_decode8_scaled(word.x, sc2, sb, sc);
        const float4 a0 = __ldg(in);
        const float4 a1 = __ldg(in + 1);
        acc = fmaf(sc[0], a0.x, acc);
        acc = fmaf(sc[1], a0.y, acc);
        acc = fmaf(sc[2], a0.z, acc);
        acc = fmaf(sc[3], a0.w, acc);
        acc = fmaf(sc[4], a1.x, acc);
        acc = fmaf(sc[5], a1.y, acc);
        acc = fmaf(sc[6], a1.z, acc);
        acc = fmaf(sc[7], a1.w, acc);
        inkling_nvfp4_decode8_scaled(word.y, sc2, sb, sc);
        const float4 a2 = __ldg(in + 2);
        const float4 a3 = __ldg(in + 3);
        acc = fmaf(sc[0], a2.x, acc);
        acc = fmaf(sc[1], a2.y, acc);
        acc = fmaf(sc[2], a2.z, acc);
        acc = fmaf(sc[3], a2.w, acc);
        acc = fmaf(sc[4], a3.x, acc);
        acc = fmaf(sc[5], a3.y, acc);
        acc = fmaf(sc[6], a3.z, acc);
        acc = fmaf(sc[7], a3.w, acc);
    }
    const float total = inkling_halfwarp_sum(acc) * s2;
    inkling_moe_emit<kSwiGlu>(total, out, slot, row, rows, q, gate_cap, up_cap);
}

// One-slot MXFP8 body shared by band and items (count == 1): zero SMEM, zero barriers.
template <bool kSwiGlu>
__device__ __forceinline__ void inkling_mxfp8_one_slot(
    const unsigned char* __restrict__ row_codes, const unsigned char* __restrict__ row_scales,
    const float* __restrict__ xr, float* __restrict__ out, int slot, int row, int rows,
    int columns, int block, int q, float gate_cap, float up_cap) {
    float acc = 0.0f;
#pragma unroll 2
    for (int base = INKLING_GROUP * q; base < columns; base += INKLING_MOE_BLOCK) {
        const uint4 word = *reinterpret_cast<const uint4*>(row_codes + base);
        const float scale = inkling_ue8m0(row_scales[base / block]);
        const float4* __restrict__ in = reinterpret_cast<const float4*>(xr + base);
        const unsigned int words[4] = {word.x, word.y, word.z, word.w};
#pragma unroll
        for (int f = 0; f < 4; ++f) {
            float weight[4];
            inkling_e4m3x4(words[f], weight);
            const float4 x4 = __ldg(in + f);
            acc = fmaf(weight[0] * scale, x4.x, acc);
            acc = fmaf(weight[1] * scale, x4.y, acc);
            acc = fmaf(weight[2] * scale, x4.z, acc);
            acc = fmaf(weight[3] * scale, x4.w, acc);
        }
    }
    const float total = inkling_halfwarp_sum(acc);
    inkling_moe_emit<kSwiGlu>(total, out, slot, row, rows, q, gate_cap, up_cap);
}

// A decode band's NVFP4 projection with the router's choice read on the device, so no host ever
// waits for it: blockIdx.x is a group of INKLING_GROUP_ROWS rows and blockIdx.y the band slot.
// grid (ceil(rows / INKLING_GROUP_ROWS), slots), block INKLING_MOE_BLOCK.
extern "C" __global__ void __launch_bounds__(INKLING_MOE_BLOCK) inkling_nvfp4_expert_band(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int columns,
    const float* __restrict__ scale_2,
    const int* __restrict__ indices,
    int first,
    int owned,
    int row_slots) {
    OPT_PDL_ENTRY();
    const int slot = blockIdx.y;
    const int local = indices[slot] - first;
    if (local < 0 || local >= owned || columns <= 0) {
        return;
    }
    const int row = blockIdx.x * INKLING_GROUP_ROWS + (threadIdx.x / INKLING_GROUP);
    const int q = threadIdx.x & (INKLING_GROUP - 1);
    const int packed = (columns + 1) / 2;
    const int scale_columns = (columns + 15) / 16;
    const int live = row < rows ? row : rows - 1;
    const unsigned char* __restrict__ row_codes =
        codes + ((uint64_t)local * rows + live) * packed;
    const unsigned char* __restrict__ row_scales =
        scales + ((uint64_t)local * rows + live) * scale_columns;
    const float* __restrict__ xr =
        x + (uint64_t)(slot / (row_slots > 0 ? row_slots : 1)) * columns;
    inkling_nvfp4_one_slot<false>(row_codes, row_scales, xr, out, slot, row, rows, columns, q,
                                  scale_2[local], 0.0f, 0.0f);
}

// Fused gate_up + softcapped SwiGLU variant of inkling_nvfp4_expert_band.
extern "C" __global__ void __launch_bounds__(INKLING_MOE_BLOCK) inkling_nvfp4_expert_band_swiglu(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int columns,
    const float* __restrict__ scale_2,
    const int* __restrict__ indices,
    int first,
    int owned,
    int row_slots,
    float gate_cap,
    float up_cap) {
    OPT_PDL_ENTRY();
    const int slot = blockIdx.y;
    const int local = indices[slot] - first;
    if (local < 0 || local >= owned || columns <= 0) {
        return;
    }
    const int row = blockIdx.x * INKLING_GROUP_ROWS + (threadIdx.x / INKLING_GROUP);
    const int q = threadIdx.x & (INKLING_GROUP - 1);
    const int packed = (columns + 1) / 2;
    const int scale_columns = (columns + 15) / 16;
    const int live = row < rows ? row : rows - 1;
    const unsigned char* __restrict__ row_codes =
        codes + ((uint64_t)local * rows + live) * packed;
    const unsigned char* __restrict__ row_scales =
        scales + ((uint64_t)local * rows + live) * scale_columns;
    const float* __restrict__ xr =
        x + (uint64_t)(slot / (row_slots > 0 ? row_slots : 1)) * columns;
    inkling_nvfp4_one_slot<true>(row_codes, row_scales, xr, out, slot, row, rows, columns, q,
                                 scale_2[local], gate_cap, up_cap);
}

// One expert's MXFP8 matrix-vector product.
extern "C" __global__ void __launch_bounds__(INKLING_MOE_BLOCK) inkling_mxfp8_expert_matvec(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int columns,
    int block,
    const int* __restrict__ plan) {
    OPT_PDL_ENTRY();
    const int row = blockIdx.x;
    if (row >= rows || columns <= 0 || block <= 0) {
        return;
    }
    // The input row of a pair is not always its band slot: the gate/up projection reads the
    // token's row of the stream and writes the pair's slot, while the down projection reads the
    // slot it was activated in. Both numbers travel in the plan, in this order.
    const int entry = blockIdx.y;
    const int input_row = plan != nullptr ? plan[2 * entry] : entry;
    const int output_slot = plan != nullptr ? plan[2 * entry + 1] : entry;
    x += (uint64_t)input_row * columns;
    out += (uint64_t)output_slot * rows;
    const int scale_columns = (columns + block - 1) / block;
    __shared__ float scratch[INKLING_MOE_BLOCK / 32 + 1];
    const float total = inkling_mxfp8_row(
        codes + (uint64_t)row * columns, scales + (uint64_t)row * scale_columns, x, columns, block,
        scratch);
    if (threadIdx.x == 0) {
        out[row] = total;
    }
}

// inkling_nvfp4_expert_band for the MXFP8 layers: no per-expert factor, one ue8m0 code a block
// of `block` columns (a multiple of 16, so a run of sixteen never straddles two).
extern "C" __global__ void __launch_bounds__(INKLING_MOE_BLOCK) inkling_mxfp8_expert_band(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int columns,
    int block,
    const int* __restrict__ indices,
    int first,
    int owned,
    int row_slots) {
    OPT_PDL_ENTRY();
    const int slot = blockIdx.y;
    const int local = indices[slot] - first;
    if (local < 0 || local >= owned || columns <= 0 || block <= 0) {
        return;
    }
    const int row = blockIdx.x * INKLING_GROUP_ROWS + (threadIdx.x / INKLING_GROUP);
    const int q = threadIdx.x & (INKLING_GROUP - 1);
    const int scale_columns = (columns + block - 1) / block;
    const int live = row < rows ? row : rows - 1;
    const unsigned char* __restrict__ row_codes =
        codes + ((uint64_t)local * rows + live) * columns;
    const unsigned char* __restrict__ row_scales =
        scales + ((uint64_t)local * rows + live) * scale_columns;
    const float* __restrict__ xr =
        x + (uint64_t)(slot / (row_slots > 0 ? row_slots : 1)) * columns;
    inkling_mxfp8_one_slot<false>(row_codes, row_scales, xr, out, slot, row, rows, columns, block,
                                  q, 0.0f, 0.0f);
}

// Fused gate_up + softcapped SwiGLU variant of inkling_mxfp8_expert_band.
extern "C" __global__ void __launch_bounds__(INKLING_MOE_BLOCK) inkling_mxfp8_expert_band_swiglu(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int columns,
    int block,
    const int* __restrict__ indices,
    int first,
    int owned,
    int row_slots,
    float gate_cap,
    float up_cap) {
    OPT_PDL_ENTRY();
    const int slot = blockIdx.y;
    const int local = indices[slot] - first;
    if (local < 0 || local >= owned || columns <= 0 || block <= 0) {
        return;
    }
    const int row = blockIdx.x * INKLING_GROUP_ROWS + (threadIdx.x / INKLING_GROUP);
    const int q = threadIdx.x & (INKLING_GROUP - 1);
    const int scale_columns = (columns + block - 1) / block;
    const int live = row < rows ? row : rows - 1;
    const unsigned char* __restrict__ row_codes =
        codes + ((uint64_t)local * rows + live) * columns;
    const unsigned char* __restrict__ row_scales =
        scales + ((uint64_t)local * rows + live) * scale_columns;
    const float* __restrict__ xr =
        x + (uint64_t)(slot / (row_slots > 0 ? row_slots : 1)) * columns;
    inkling_mxfp8_one_slot<true>(row_codes, row_scales, xr, out, slot, row, rows, columns, block, q,
                                 gate_cap, up_cap);
}

// A step's inputs for a pair group, staged in shared memory: the 256 columns of a step for each
// pair, as sixteen runs of sixteen floats padded to twenty, so the sixteen lanes reading their
// runs with 128-bit loads land on distinct banks.
#define INKLING_STAGE_RUN 20
#define INKLING_STAGE (INKLING_MOE_BLOCK / INKLING_GROUP * INKLING_STAGE_RUN)

// One float accumulator per pair lets kPairs = 8 fit in 64 registers (0 spills, 4 blocks/SM),
// so a verify block of up to 8 tokens reads and decodes each shared expert's weights exactly once.
#define INKLING_PAIRS 8
#define INKLING_ITEM_PAIRS 8

// kSlots picks how `plan` names a pair: false, interleaved (input row, output slot) entries (the
// host's grouping); true, a list of band slots, slot s reading input row s / row_slots (moe_plan's
// sorted slots). Every slot computes the exact same 16-lane fmaf order and half-warp reduction as
// inkling_{nvfp4,mxfp8}_one_slot, so band, items and pairs are bit-identical to each other.
template <bool kNvfp4, bool kSlots, bool kSwiGlu, int kPairs = INKLING_PAIRS>
__device__ __forceinline__ void inkling_expert_pairs(const unsigned char* __restrict__ codes,
                                                     const unsigned char* __restrict__ scales,
                                                     const float* __restrict__ x,
                                                     float* __restrict__ out, int rows,
                                                     int columns, int block, float scale_2,
                                                     const int* __restrict__ plan, int count,
                                                     int row_slots, float gate_cap, float up_cap) {
    auto input_row = [&](int pair) { return kSlots ? plan[pair] / row_slots : plan[2 * pair]; };
    auto output_slot = [&](int pair) { return kSlots ? plan[pair] : plan[2 * pair + 1]; };
    const int row = blockIdx.x * INKLING_GROUP_ROWS + (threadIdx.x / INKLING_GROUP);
    const int q = threadIdx.x & (INKLING_GROUP - 1);
    const int live = row < rows ? row : rows - 1;
    const int code_row = kNvfp4 ? (columns + 1) / 2 : columns;
    const int scale_row = kNvfp4 ? (columns + 15) / 16 : (columns + block - 1) / block;
    const unsigned char* __restrict__ row_codes = codes + (uint64_t)live * code_row;
    const unsigned char* __restrict__ row_scales = scales + (uint64_t)live * scale_row;

    // Single-slot fast path: zero SMEM staging, zero barriers, identical arithmetic.
    if (count == 1) {
        const float* __restrict__ xr = x + (uint64_t)input_row(0) * columns;
        const int slot = output_slot(0);
        if constexpr (kNvfp4) {
            inkling_nvfp4_one_slot<kSwiGlu>(row_codes, row_scales, xr, out, slot, row, rows,
                                            columns, q, scale_2, gate_cap, up_cap);
        } else {
            inkling_mxfp8_one_slot<kSwiGlu>(row_codes, row_scales, xr, out, slot, row, rows,
                                            columns, block, q, gate_cap, up_cap);
        }
        return;
    }

    constexpr int kStagePieces = kPairs * INKLING_MOE_BLOCK / 4;
    constexpr int kStageLoads = (kStagePieces + INKLING_MOE_BLOCK - 1) / INKLING_MOE_BLOCK;
    __shared__ __align__(16) float staged[2][kPairs * INKLING_STAGE];
    const int steps = (columns + INKLING_MOE_BLOCK - 1) / INKLING_MOE_BLOCK;
    const int col_off = 4 * ((int)threadIdx.x & (INKLING_MOE_BLOCK / 4 - 1));
    const int pair_base = (int)threadIdx.x / (INKLING_MOE_BLOCK / 4);
    const int target_base =
        (col_off / INKLING_GROUP) * INKLING_STAGE_RUN + (col_off % INKLING_GROUP);
    for (int group = 0; group < count; group += kPairs) {
        const int here = count - group < kPairs ? count - group : kPairs;
        int in_row[kStageLoads];
#pragma unroll
        for (int l = 0; l < kStageLoads; ++l) {
            const int pair = pair_base + l * 4;
            in_row[l] = pair < here ? input_row(group + pair) : -1;
        }
        auto fetch = [&](int step, float4 (&value)[kStageLoads]) {
            const int column = step * INKLING_MOE_BLOCK + col_off;
#pragma unroll
            for (int l = 0; l < kStageLoads; ++l) {
                value[l] = (in_row[l] >= 0 && column < columns)
                               ? *reinterpret_cast<const float4*>(
                                     x + (uint64_t)in_row[l] * columns + column)
                               : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        };
        auto stage = [&](int buffer, const float4 (&value)[kStageLoads]) {
#pragma unroll
            for (int l = 0; l < kStageLoads; ++l) {
                const int pair = pair_base + l * 4;
                if (pair < kPairs) {
                    *reinterpret_cast<float4*>(
                        &staged[buffer][pair * INKLING_STAGE + target_base]) = value[l];
                }
            }
        };
        float4 pending[kStageLoads];
        fetch(0, pending);
        stage(0, pending);
        __syncthreads();
        float acc[kPairs];
#pragma unroll
        for (int p = 0; p < kPairs; ++p) {
            acc[p] = 0.0f;
        }
        for (int step = 0; step < steps; ++step) {
            const int current = step & 1;
            if (step + 1 < steps) {
                fetch(step + 1, pending);
            }
            const int base = step * INKLING_MOE_BLOCK + INKLING_GROUP * q;
            if (base < columns) {
                if constexpr (kNvfp4) {
                    const uint2 word = __ldg(reinterpret_cast<const uint2*>(row_codes + (base >> 1)));
                    const unsigned char sb = __ldg(row_scales + (base >> 4));
                    const __half2 sc2 = inkling_e4m3_scale_f16x2(sb);
                    float sc[8];
                    inkling_nvfp4_decode8_scaled(word.x, sc2, sb, sc);
#pragma unroll
                    for (int p = 0; p < kPairs; ++p) {
                        if (p < here) {
                            const float* in =
                                &staged[current][p * INKLING_STAGE + q * INKLING_STAGE_RUN];
                            const float4 a0 = *reinterpret_cast<const float4*>(in);
                            const float4 a1 = *reinterpret_cast<const float4*>(in + 4);
                            float s = acc[p];
                            s = fmaf(sc[0], a0.x, s);
                            s = fmaf(sc[1], a0.y, s);
                            s = fmaf(sc[2], a0.z, s);
                            s = fmaf(sc[3], a0.w, s);
                            s = fmaf(sc[4], a1.x, s);
                            s = fmaf(sc[5], a1.y, s);
                            s = fmaf(sc[6], a1.z, s);
                            s = fmaf(sc[7], a1.w, s);
                            acc[p] = s;
                        }
                    }
                    inkling_nvfp4_decode8_scaled(word.y, sc2, sb, sc);
#pragma unroll
                    for (int p = 0; p < kPairs; ++p) {
                        if (p < here) {
                            const float* in =
                                &staged[current][p * INKLING_STAGE + q * INKLING_STAGE_RUN];
                            const float4 a2 = *reinterpret_cast<const float4*>(in + 8);
                            const float4 a3 = *reinterpret_cast<const float4*>(in + 12);
                            float s = acc[p];
                            s = fmaf(sc[0], a2.x, s);
                            s = fmaf(sc[1], a2.y, s);
                            s = fmaf(sc[2], a2.z, s);
                            s = fmaf(sc[3], a2.w, s);
                            s = fmaf(sc[4], a3.x, s);
                            s = fmaf(sc[5], a3.y, s);
                            s = fmaf(sc[6], a3.z, s);
                            s = fmaf(sc[7], a3.w, s);
                            acc[p] = s;
                        }
                    }
                } else {
                    const uint4 word = __ldg(reinterpret_cast<const uint4*>(row_codes + base));
                    const int scale_idx = block == 32 ? (base >> 5) : (base / block);
                    const float scale = inkling_ue8m0(__ldg(row_scales + scale_idx));
                    const unsigned int words[4] = {word.x, word.y, word.z, word.w};
#pragma unroll
                    for (int f = 0; f < 4; ++f) {
                        float weight[4];
                        inkling_e4m3x4(words[f], weight);
                        const float s0 = weight[0] * scale;
                        const float s1 = weight[1] * scale;
                        const float s2 = weight[2] * scale;
                        const float s3 = weight[3] * scale;
#pragma unroll
                        for (int p = 0; p < kPairs; ++p) {
                            if (p < here) {
                                const float* in =
                                    &staged[current][p * INKLING_STAGE + q * INKLING_STAGE_RUN];
                                const float4 x4 = *reinterpret_cast<const float4*>(in + 4 * f);
                                float s = acc[p];
                                s = fmaf(s0, x4.x, s);
                                s = fmaf(s1, x4.y, s);
                                s = fmaf(s2, x4.z, s);
                                s = fmaf(s3, x4.w, s);
                                acc[p] = s;
                            }
                        }
                    }
                }
            }
            if (step + 1 < steps) {
                stage(current ^ 1, pending);
            }
            __syncthreads();
        }
#pragma unroll
        for (int p = 0; p < kPairs; ++p) {
            if (p < here) {
                const float total =
                    kNvfp4 ? inkling_halfwarp_sum(acc[p]) * scale_2 : inkling_halfwarp_sum(acc[p]);
                inkling_moe_emit<kSwiGlu>(total, out, output_slot(group + p), row, rows, q,
                                          gate_cap, up_cap);
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(INKLING_MOE_BLOCK) inkling_nvfp4_expert_pairs(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int columns,
    float scale_2,
    const int* __restrict__ plan,
    int count) {
    OPT_PDL_ENTRY();
    if (columns <= 0 || count <= 0) {
        return;
    }
    inkling_expert_pairs<true, false, false>(codes, scales, x, out, rows, columns, 16, scale_2,
                                             plan, count, 1, 0.0f, 0.0f);
}

extern "C" __global__ void __launch_bounds__(INKLING_MOE_BLOCK) inkling_mxfp8_expert_pairs(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int columns,
    int block,
    const int* __restrict__ plan,
    int count) {
    OPT_PDL_ENTRY();
    if (columns <= 0 || count <= 0 || block <= 0) {
        return;
    }
    inkling_expert_pairs<false, false, false>(codes, scales, x, out, rows, columns, block, 1.0f,
                                              plan, count, 1, 0.0f, 0.0f);
}

// The pair walk over moe_plan's items, for a band of a few rows (a verify, a few fused lanes) with
// the router's choice left on the device.
extern "C" __global__ void __launch_bounds__(INKLING_MOE_BLOCK) inkling_nvfp4_expert_items(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int columns,
    const float* __restrict__ scale_2,
    const int* __restrict__ sorted,
    const int* __restrict__ items,
    const int* __restrict__ item_count,
    int row_slots) {
    OPT_PDL_ENTRY();
    const int n = blockIdx.y;
    if (columns <= 0 || row_slots <= 0 || n >= *item_count) {
        return;
    }
    const int local = items[4 * n];
    const int first = items[4 * n + 1];
    const int count = items[4 * n + 2];
    if (local < 0 || count <= 0) {
        return;
    }
    const uint64_t packed = (uint64_t)(columns + 1) / 2;
    const uint64_t scale_columns = (uint64_t)(columns + 15) / 16;
    inkling_expert_pairs<true, true, false, INKLING_ITEM_PAIRS>(
        codes + (uint64_t)local * rows * packed, scales + (uint64_t)local * rows * scale_columns, x,
        out, rows, columns, 16, scale_2[local], sorted + first, count, row_slots, 0.0f, 0.0f);
}

// Fused gate_up + softcapped SwiGLU variant of inkling_nvfp4_expert_items.
extern "C" __global__ void __launch_bounds__(INKLING_MOE_BLOCK) inkling_nvfp4_expert_items_swiglu(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int columns,
    const float* __restrict__ scale_2,
    const int* __restrict__ sorted,
    const int* __restrict__ items,
    const int* __restrict__ item_count,
    int row_slots,
    float gate_cap,
    float up_cap) {
    OPT_PDL_ENTRY();
    const int n = blockIdx.y;
    if (columns <= 0 || row_slots <= 0 || n >= *item_count) {
        return;
    }
    const int local = items[4 * n];
    const int first = items[4 * n + 1];
    const int count = items[4 * n + 2];
    if (local < 0 || count <= 0) {
        return;
    }
    const uint64_t packed = (uint64_t)(columns + 1) / 2;
    const uint64_t scale_columns = (uint64_t)(columns + 15) / 16;
    inkling_expert_pairs<true, true, true, INKLING_ITEM_PAIRS>(
        codes + (uint64_t)local * rows * packed, scales + (uint64_t)local * rows * scale_columns, x,
        out, rows, columns, 16, scale_2[local], sorted + first, count, row_slots, gate_cap, up_cap);
}

// inkling_nvfp4_expert_items for the MXFP8 layers: one ue8m0 code a block of `block` columns.
extern "C" __global__ void __launch_bounds__(INKLING_MOE_BLOCK) inkling_mxfp8_expert_items(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int columns,
    int block,
    const int* __restrict__ sorted,
    const int* __restrict__ items,
    const int* __restrict__ item_count,
    int row_slots) {
    OPT_PDL_ENTRY();
    const int n = blockIdx.y;
    if (columns <= 0 || block <= 0 || row_slots <= 0 || n >= *item_count) {
        return;
    }
    const int local = items[4 * n];
    const int first = items[4 * n + 1];
    const int count = items[4 * n + 2];
    if (local < 0 || count <= 0) {
        return;
    }
    const uint64_t scale_columns = (uint64_t)(columns + block - 1) / block;
    inkling_expert_pairs<false, true, false, INKLING_ITEM_PAIRS>(
        codes + (uint64_t)local * rows * columns, scales + (uint64_t)local * rows * scale_columns, x,
        out, rows, columns, block, 1.0f, sorted + first, count, row_slots, 0.0f, 0.0f);
}

// Fused gate_up + softcapped SwiGLU variant of inkling_mxfp8_expert_items.
extern "C" __global__ void __launch_bounds__(INKLING_MOE_BLOCK) inkling_mxfp8_expert_items_swiglu(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int columns,
    int block,
    const int* __restrict__ sorted,
    const int* __restrict__ items,
    const int* __restrict__ item_count,
    int row_slots,
    float gate_cap,
    float up_cap) {
    OPT_PDL_ENTRY();
    const int n = blockIdx.y;
    if (columns <= 0 || block <= 0 || row_slots <= 0 || n >= *item_count) {
        return;
    }
    const int local = items[4 * n];
    const int first = items[4 * n + 1];
    const int count = items[4 * n + 2];
    if (local < 0 || count <= 0) {
        return;
    }
    const uint64_t scale_columns = (uint64_t)(columns + block - 1) / block;
    inkling_expert_pairs<false, true, true, INKLING_ITEM_PAIRS>(
        codes + (uint64_t)local * rows * columns, scales + (uint64_t)local * rows * scale_columns, x,
        out, rows, columns, block, 1.0f, sorted + first, count, row_slots, gate_cap, up_cap);
}

// The mix's share a band slot: the router's weight for a slot this rank computed, zero for one it
// did not (its partial is another rank's, and the all-reduce after the mix brings it in). One
// thread a slot, the grid striding over the slots; the band's replacement for the host's copy of
// the shares. `local`, when not null, gets the slot's expert as this rank numbers its experts,
// -1 for another rank's: the ids the shared plan (moe_plan, MOE_PLAN_DROP) sorts.
extern "C" __global__ void __launch_bounds__(64) inkling_band_shares(
    const int* __restrict__ indices,
    const float* __restrict__ shares,
    float* __restrict__ out,
    int slots,
    int first,
    int owned,
    int* __restrict__ local) {
    OPT_PDL_ENTRY();
    for (int slot = blockIdx.x * blockDim.x + threadIdx.x; slot < slots;
         slot += gridDim.x * blockDim.x) {
        const int expert = indices[slot] - first;
        const bool mine = expert >= 0 && expert < owned;
        out[slot] = mine ? shares[slot] : 0.0f;
        if (local != nullptr) {
            local[slot] = mine ? expert : -1;
        }
    }
}

// The sigmoid router: a score an expert, the learned bias on the selection, and the selected
// experts' shares. The block shares the walk: each thread scans the experts congruent to its index
// in ascending order and keeps the first best it meets, and the block then keeps the best of those,
// the lower index on a tie. That is the sequential scan's answer exactly (the first index holding
// the greatest biased score, NaN and -inf never chosen), so the schedule is still a function of the
// scores alone, and the shares are summed by one thread in slot order as before.
//
// A thread holds its experts' biased scores in registers and marks the ones already chosen in a
// mask, so a pick rescans nothing in shared memory; every thread then takes the block's best from
// the warps' bests itself, one barrier a pick (the warps' bests alternate between two buffers, so
// a pick's writes never meet the previous pick's reads). The bias is a weight and is loaded before
// the previous kernel ends.
#define INKLING_ROUTER_THREADS 128
// The experts a thread holds: the capacity over the block, one bit of the chosen mask each.
#define INKLING_ROUTER_HELD 32
extern "C" __global__ void __launch_bounds__(INKLING_ROUTER_THREADS) inkling_router_topk(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int* __restrict__ indices,
    float* __restrict__ weights,
    int experts,
    int top_k,
    float route_scale) {
    OPT_PDL_LAUNCH();
    const int stride = blockDim.x;
    if (experts <= 0 || top_k <= 0 || top_k > experts || top_k > 64 ||
        experts > INKLING_ROUTER_HELD * stride) {
        return;
    }
    // One block a row: a prefill chunk routes every row of the chunk, a decode step its one row
    // (grid 1, which leaves the three offsets below at zero).
    const int row = blockIdx.x;
    logits += (uint64_t)row * experts;
    indices += (uint64_t)row * top_k;
    weights += (uint64_t)row * top_k;
    const int capacity = 4096;
    __shared__ float scores[capacity];
    __shared__ int chosen[64];
    // Fast 1-warp register-only path for experts <= 128 && top_k <= 16 (Inkling-2-Mini: 128/8):
    // zero __syncthreads(), zero local stack frame, bit-identical tie-breaking and slot-order sum.
    if (experts <= 128 && top_k <= 16) {
        if (threadIdx.x >= 32) {
            OPT_PDL_WAIT();
            return;
        }
        const int lane = (int)threadIdx.x;
        float rk[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int expert = lane + 32 * k;
            rk[k] = (bias != nullptr && expert < experts) ? bias[expert] : 0.0f;
        }
        OPT_PDL_WAIT();
        float sc[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int expert = lane + 32 * k;
            if (expert < experts) {
                sc[k] = 1.0f / (1.0f + expf(-logits[expert]));
                rk[k] = __fadd_rn(sc[k], rk[k]);
            } else {
                sc[k] = 0.0f;
                rk[k] = -INFINITY;
            }
        }
        unsigned int taken = 0u;
        for (int slot = 0; slot < top_k; ++slot) {
            int best = -1;
            float best_score = -INFINITY;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int expert = lane + 32 * k;
                if (expert < experts && ((taken >> k) & 1u) == 0u) {
                    if (best < 0 || rk[k] > best_score ||
                        (rk[k] == best_score && expert < best)) {
                        best_score = rk[k];
                        best = expert;
                    }
                }
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other_score = __shfl_xor_sync(0xFFFFFFFFu, best_score, offset);
                const int other = __shfl_xor_sync(0xFFFFFFFFu, best, offset);
                if (other >= 0 && (best < 0 || other_score > best_score ||
                                   (other_score == best_score && other < best))) {
                    best_score = other_score;
                    best = other;
                }
            }
            if (best < 0) {
                return;
            }
            float mine = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (lane + 32 * k == best) {
                    taken |= 1u << k;
                    mine = sc[k];
                }
            }
            const float win_score = __shfl_sync(0xFFFFFFFFu, mine, best & 31);
            if (lane == 0) {
                chosen[slot] = best;
                scores[slot] = win_score;
            }
        }
        if (lane == 0) {
            float sum = 0.0f;
            for (int slot = 0; slot < top_k; ++slot) {
                sum = __fadd_rn(sum, scores[slot]);
            }
            const float denominator = __fadd_rn(sum, 1e-20f);
            for (int slot = 0; slot < top_k; ++slot) {
                indices[slot] = chosen[slot];
                weights[slot] = __fmul_rn(__fdiv_rn(scores[slot], denominator), route_scale);
            }
        }
        return;
    }

    if (experts > capacity) {
        return;
    }
    __shared__ float best_scores[2][INKLING_ROUTER_THREADS / 32];
    __shared__ int best_experts[2][INKLING_ROUTER_THREADS / 32];
    float ranked[INKLING_ROUTER_HELD];
#pragma unroll
    for (int k = 0; k < INKLING_ROUTER_HELD; ++k) {
        const int expert = threadIdx.x + k * stride;
        if (expert >= experts) {
            break;
        }
        ranked[k] = bias != nullptr ? bias[expert] : 0.0f;
    }
    OPT_PDL_WAIT();
#pragma unroll
    for (int k = 0; k < INKLING_ROUTER_HELD; ++k) {
        const int expert = threadIdx.x + k * stride;
        if (expert >= experts) {
            break;
        }
        const float score = 1.0f / (1.0f + expf(-logits[expert]));
        scores[expert] = score;
        ranked[k] = __fadd_rn(score, ranked[k]);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = (stride + 31) >> 5;
    if (experts <= stride) {
        bool avail = (int)threadIdx.x < experts;
        const float r0 = avail ? ranked[0] : -INFINITY;
        const int my_id = (int)threadIdx.x;
        for (int slot = 0; slot < top_k; ++slot) {
            int best = avail ? my_id : -1;
            float best_score = avail ? r0 : -INFINITY;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other_score = __shfl_down_sync(0xFFFFFFFFu, best_score, offset);
                const int other = __shfl_down_sync(0xFFFFFFFFu, best, offset);
                if (other >= 0 && (best < 0 || other_score > best_score ||
                                   (other_score == best_score && other < best))) {
                    best_score = other_score;
                    best = other;
                }
            }
            const int buffer = slot & 1;
            if (lane == 0) {
                best_scores[buffer][warp] = best_score;
                best_experts[buffer][warp] = best;
            }
            __syncthreads();
            int winner = -1;
            float winner_score = -INFINITY;
#pragma unroll
            for (int index = 0; index < INKLING_ROUTER_THREADS / 32; ++index) {
                const int other = best_experts[buffer][index];
                const float other_score = best_scores[buffer][index];
                if (other >= 0 && (winner < 0 || other_score > winner_score ||
                                   (other_score == winner_score && other < winner))) {
                    winner_score = other_score;
                    winner = other;
                }
            }
            if (winner < 0) {
                return;
            }
            if (threadIdx.x == 0) {
                chosen[slot] = winner;
            }
            if (winner == my_id) {
                avail = false;
            }
        }
    } else {
        unsigned int taken = 0;
        for (int slot = 0; slot < top_k; ++slot) {
            int best = -1;
            float best_score = -INFINITY;
#pragma unroll
            for (int k = 0; k < INKLING_ROUTER_HELD; ++k) {
                const int expert = threadIdx.x + k * stride;
                if (expert >= experts) {
                    break;
                }
                if (!((taken >> k) & 1u) && ranked[k] > best_score) {
                    best_score = ranked[k];
                    best = expert;
                }
            }
            // A candidate beats another on a greater score, or on an equal score at a lower index;
            // "none" is index -1 at -inf, which no real candidate loses to.
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other_score = __shfl_down_sync(0xFFFFFFFFu, best_score, offset);
                const int other = __shfl_down_sync(0xFFFFFFFFu, best, offset);
                if (other >= 0 && (best < 0 || other_score > best_score ||
                                   (other_score == best_score && other < best))) {
                    best_score = other_score;
                    best = other;
                }
            }
            const int buffer = slot & 1;
            if (lane == 0) {
                best_scores[buffer][warp] = best_score;
                best_experts[buffer][warp] = best;
            }
            __syncthreads();
            int winner = -1;
            float winner_score = -INFINITY;
            for (int index = 0; index < warps; ++index) {
                const int other = best_experts[buffer][index];
                const float other_score = best_scores[buffer][index];
                if (other >= 0 && (winner < 0 || other_score > winner_score ||
                                   (other_score == winner_score && other < winner))) {
                    winner_score = other_score;
                    winner = other;
                }
            }
            if (winner < 0) {
                return;
            }
            if (threadIdx.x == 0) {
                chosen[slot] = winner;
            }
            if (winner % stride == (int)threadIdx.x) {
                taken |= 1u << (winner / stride);
            }
        }
    }
    if (threadIdx.x != 0) {
        return;
    }
    // Thread 0 wrote every chosen slot and every score it reads below is behind the last barrier.
    float sum = 0.0f;
    for (int slot = 0; slot < top_k; ++slot) {
        sum = __fadd_rn(sum, scores[chosen[slot]]);
    }
    const float denominator = __fadd_rn(sum, 1e-20f);
    for (int slot = 0; slot < top_k; ++slot) {
        indices[slot] = chosen[slot];
        weights[slot] = __fmul_rn(__fdiv_rn(scores[chosen[slot]], denominator), route_scale);
    }
}

// Fused 1-block sigmoid router + expert plan for TP=1 decode/verify (rows <= 8, experts <= 128,
// top_k <= 16): warps 0..rows-1 each run the exact 1-warp inkling_router_topk body in parallel,
// then threads 0..127 run the exact 4-warp moe_plan prefix sum + item emit + warp-0 scatter.
// Produces bit-identical indices, weights, sorted, items, and item_count in one launch.
// Grid (1, 1, 1), Block 256 threads.
extern "C" __global__ void __launch_bounds__(256) inkling_router_plan_small(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int* __restrict__ indices,
    float* __restrict__ weights,
    int* __restrict__ sorted,
    int* __restrict__ items,
    int* __restrict__ item_count,
    int rows,
    int experts,
    int top_k,
    float route_scale,
    int max_items) {
    OPT_PDL_LAUNCH();
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (rows <= 0 || rows > 8 || experts <= 0 || experts > 128 || top_k <= 0 || top_k > 16 ||
        top_k > experts) {
        OPT_PDL_WAIT();
        return;
    }
    __shared__ int s_count[128];
    __shared__ int s_base[128];
    __shared__ int s_ids[128];
    __shared__ int s_chosen[8][16];
    __shared__ float s_scores[8][16];
    __shared__ int w_tot_c[4];
    __shared__ int w_tot_a[4];
    __shared__ int s_used;

    if (tid < experts) {
        s_count[tid] = 0;
    }
    float rk[4];
    if (warp < rows) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int expert = lane + 32 * k;
            rk[k] = (bias != nullptr && expert < experts) ? bias[expert] : 0.0f;
        }
    }
    OPT_PDL_WAIT();
    __syncthreads();

    if (warp < rows) {
        const float* row_logits = logits + (uint64_t)warp * experts;
        int* row_indices = indices + (uint64_t)warp * top_k;
        float* row_weights = weights + (uint64_t)warp * top_k;
        float sc[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int expert = lane + 32 * k;
            if (expert < experts) {
                sc[k] = 1.0f / (1.0f + expf(-row_logits[expert]));
                rk[k] = __fadd_rn(sc[k], rk[k]);
            } else {
                sc[k] = 0.0f;
                rk[k] = -INFINITY;
            }
        }
        unsigned int taken = 0u;
        for (int slot = 0; slot < top_k; ++slot) {
            int best = -1;
            float best_score = -INFINITY;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int expert = lane + 32 * k;
                if (expert < experts && ((taken >> k) & 1u) == 0u) {
                    if (best < 0 || rk[k] > best_score ||
                        (rk[k] == best_score && expert < best)) {
                        best_score = rk[k];
                        best = expert;
                    }
                }
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other_score = __shfl_xor_sync(0xFFFFFFFFu, best_score, offset);
                const int other = __shfl_xor_sync(0xFFFFFFFFu, best, offset);
                if (other >= 0 && (best < 0 || other_score > best_score ||
                                   (other_score == best_score && other < best))) {
                    best_score = other_score;
                    best = other;
                }
            }
            if (best < 0) {
                break;
            }
            float mine = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (lane + 32 * k == best) {
                    taken |= 1u << k;
                    mine = sc[k];
                }
            }
            const float win_score = __shfl_sync(0xFFFFFFFFu, mine, best & 31);
            if (lane == 0) {
                s_chosen[warp][slot] = best;
                s_scores[warp][slot] = win_score;
            }
        }
        if (lane == 0) {
            float sum = 0.0f;
            for (int slot = 0; slot < top_k; ++slot) {
                sum = __fadd_rn(sum, s_scores[warp][slot]);
            }
            const float denominator = __fadd_rn(sum, 1e-20f);
            const int base_slot = warp * top_k;
            for (int slot = 0; slot < top_k; ++slot) {
                const int best = s_chosen[warp][slot];
                row_indices[slot] = best;
                row_weights[slot] =
                    __fmul_rn(__fdiv_rn(s_scores[warp][slot], denominator), route_scale);
                s_ids[base_slot + slot] = best;
                atomicAdd(&s_count[best], 1);
            }
        }
    }
    __syncthreads();

    if (tid < 128) {
        const int c = tid < experts ? s_count[tid] : 0;
        const int act = c > 0 ? 1 : 0;
        int sc = c;
        int sa = act;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int pc = __shfl_up_sync(0xFFFFFFFFu, sc, off);
            const int pa = __shfl_up_sync(0xFFFFFFFFu, sa, off);
            if (lane >= off) {
                sc += pc;
                sa += pa;
            }
        }
        if (lane == 31) {
            w_tot_c[warp] = sc;
            w_tot_a[warp] = sa;
        }
        __syncthreads();
        int off_c = 0;
        int off_a = 0;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (w < warp) {
                off_c += w_tot_c[w];
                off_a += w_tot_a[w];
            }
        }
        const int base = off_c + sc - c;
        const int item_idx = off_a + sa - act;
        if (tid < experts) {
            s_base[tid] = base;
            if (c > 0 && item_idx < max_items) {
                items[4 * item_idx] = tid;
                items[4 * item_idx + 1] = base;
                items[4 * item_idx + 2] = c;
                items[4 * item_idx + 3] = item_idx;
            }
        }
        if (tid == 127) {
            const int total_items = off_a + sa;
            s_used = total_items < max_items ? total_items : max_items;
            if (item_count != nullptr) {
                *item_count = s_used;
            }
        }
    } else {
        __syncthreads();
    }
    __syncthreads();

    for (int item = s_used + tid; item < max_items; item += 256) {
        items[4 * item] = 0;
        items[4 * item + 1] = 0;
        items[4 * item + 2] = 0;
        items[4 * item + 3] = item;
    }
    if (tid < 32) {
        const int slots = rows * top_k;
        const unsigned int below = (1u << tid) - 1u;
        for (int first = 0; first < slots; first += 32) {
            const int rank = first + tid;
            const int bucket = rank < slots ? s_ids[rank] : -1;
            const unsigned int same = __match_any_sync(0xFFFFFFFFu, bucket);
            const int at = bucket >= 0 ? s_base[bucket] + __popc(same & below) : 0;
            __syncwarp();
            if (bucket >= 0) {
                if ((same & below) == 0u) {
                    s_base[bucket] = at + __popc(same);
                }
                sorted[at] = rank;
            }
            __syncwarp();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// High-occupancy BM=64, BN=128, BK=64 grouped MoE tensor-core prefill (2 blocks/SM on SM121).
//
// NVFP4 uses hardware cvt.rn.f16x2.e2m1x2 + cvt.rn.f16x2.e4m3x2 + __hmul2 + f16 mma.sync over
// 2-plane FP16 activations (head + remainder, clamped to +-65504 so no Inf-Inf NaN occurs):
// every NVFP4 weight E2M1 * E4M3 is exact in FP16 (max 2688 << 65504, min nonzero 2^-10 >> 2^-14,
// at most 6 significant bits <= 11), eliminating the 256-entry SMEM LUT and its bank conflicts
// while fitting 2 blocks/SM (42,240 B SMEM, <= 122 regs, 0 spills).
//
// MXFP8 keeps BF16 MMA (preserving exact ue8m0 exponent range and bit-exact sums with
// q4p_moe_*_mx8) at BM=64, BN=128, 2 stages (50,432 B SMEM -> 2 blocks/SM, 0 spills).
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ void inkling_f16_mma(float d[4], const uint32_t a[4], uint32_t b0,
                                                uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void inkling_cp4(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ uint32_t inkling_f16x2_mul(uint32_t x, uint32_t y) {
    const __half2 r =
        __hmul2(*reinterpret_cast<const __half2*>(&x), *reinterpret_cast<const __half2*>(&y));
    return *reinterpret_cast<const uint32_t*>(&r);
}

__device__ __forceinline__ void inkling_scale_pairs_f16(uint16_t two_bytes, uint32_t& s0,
                                                        uint32_t& s1) {
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(two_bytes));
    s0 = (h & 0xffffu) | ((h & 0xffffu) << 16);
    s1 = (h & 0xffff0000u) | (h >> 16);
}

__device__ __forceinline__ void inkling_e2m1x8_f16(uint32_t word, uint32_t out[4]) {
    asm("{\n\t.reg .b8 c0, c1, c2, c3;\n\t"
        "mov.b32 {c0, c1, c2, c3}, %4;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, c0;\n\t"
        "cvt.rn.f16x2.e2m1x2 %1, c1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %2, c2;\n\t"
        "cvt.rn.f16x2.e2m1x2 %3, c3;\n\t}"
        : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
        : "r"(word));
}

__device__ __forceinline__ uint32_t inkling_e4m3x2_scaled_bf16(uint32_t pair, float scale) {
    const __half2_raw raw =
        __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(pair & 0xffffu), __NV_E4M3);
    const float2 v = __half22float2(*reinterpret_cast<const __half2*>(&raw));
    const __nv_bfloat162 r = __floats2bfloat162_rn(v.x * scale, v.y * scale);
    return *reinterpret_cast<const uint32_t*>(&r);
}

__device__ __forceinline__ void inkling_split2_f16(float a, float b, __half2& head,
                                                   __half2& rest) {
    const float ca = fminf(fmaxf(a, -65504.0f), 65504.0f);
    const float cb = fminf(fmaxf(b, -65504.0f), 65504.0f);
    head = __floats2half2_rn(ca, cb);
    const float2 hf = __half22float2(head);
    rest = __floats2half2_rn(a - hf.x, b - hf.y);
}

__device__ __forceinline__ void inkling_split2_bf16(float a, float b, __nv_bfloat162& head,
                                                    __nv_bfloat162& rest) {
    head = __floats2bfloat162_rn(a, b);
    rest = __floats2bfloat162_rn(a - __low2float(head), b - __high2float(head));
}

// f32 -> two f16 planes (head at y[0..count), remainder at y[count..2*count)).
extern "C" __global__ void inkling_split_f16(const float* __restrict__ x,
                                             __half* __restrict__ y,
                                             unsigned long long count) {
    OPT_PDL_ENTRY();
    const unsigned long long quads = count >> 2;
    for (unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < quads; index += (unsigned long long)gridDim.x * blockDim.x) {
        const float4 v = reinterpret_cast<const float4*>(x)[index];
        __half2 h0, r0, h1, r1;
        inkling_split2_f16(v.x, v.y, h0, r0);
        inkling_split2_f16(v.z, v.w, h1, r1);
        uint2 hp = make_uint2(*reinterpret_cast<const uint32_t*>(&h0),
                              *reinterpret_cast<const uint32_t*>(&h1));
        uint2 rp = make_uint2(*reinterpret_cast<const uint32_t*>(&r0),
                              *reinterpret_cast<const uint32_t*>(&r1));
        reinterpret_cast<uint2*>(y)[index] = hp;
        reinterpret_cast<uint2*>(y + count)[index] = rp;
    }
    const unsigned long long tail_start = quads << 2;
    if (blockIdx.x == 0 && threadIdx.x < 4) {
        const unsigned long long i = tail_start + threadIdx.x;
        if (i < count) {
            const float v = x[i];
            const __half h = __float2half_rn(fminf(fmaxf(v, -65504.0f), 65504.0f));
            y[i] = h;
            y[count + i] = __float2half_rn(v - __half2float(h));
        }
    }
}

template <bool GATE_UP>
__device__ __forceinline__ void inkling_moe_f16_64_body(
    const unsigned long long* __restrict__ table, const float* __restrict__ scales,
    const int* __restrict__ sorted, const int4* __restrict__ items,
    const int* __restrict__ item_count, const __half* __restrict__ a_src,
    unsigned long long a_plane, const float* __restrict__ route_weights, void* __restrict__ out,
    unsigned long long out_plane, int topk, int hidden, int intermediate, int weight_stride,
    float gate_cap, float up_cap) {
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int WM = 2;
    constexpr int MT = 2;
    constexpr int NT_HALF = 2;
    constexpr int NJ = 4;
    struct Stage {
        __half a[2][BM * 64];
        uint32_t b[BN * 8];
        uint32_t s[BN];
    };
    extern __shared__ __align__(16) uint8_t inkling_moe64_smem[];
    Stage* stages = reinterpret_cast<Stage*>(inkling_moe64_smem);
    int* s_slot = reinterpret_cast<int*>(stages + 2);
    if ((int)blockIdx.y >= *item_count) {
        return;
    }
    const int4 item = items[blockIdx.y];
    const int expert = item.x;
    const int first = item.y;
    const int count = item.z;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp % WM;
    const int wn = warp / WM;
    const int inner = GATE_UP ? hidden : intermediate;
    const int n0 = blockIdx.x * (GATE_UP ? 64 : 128);
    const int packed_row = inner >> 1;
    const int scale_row = inner >> 4;
    if (tid < BM) {
        s_slot[tid] = tid < count ? sorted[first + tid] : -1;
    }
    __syncthreads();

    const int a_chunk = tid & 7;
    const __half* a_base[2];
    bool a_valid[2];
    uint32_t a_dst[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int r = (tid >> 3) + i * 32;
        const int slot = s_slot[r];
        a_valid[i] = slot >= 0;
        if (slot >= 0) {
            a_base[i] = GATE_UP ? a_src + (unsigned long long)(slot / topk) * hidden
                                : a_src + (unsigned long long)slot * intermediate;
        } else {
            a_base[i] = a_src;
        }
        a_dst[i] = (uint32_t)(r * 64 + ((a_chunk ^ ((r & 1) << 2)) * 8));
    }
    const int b_row = tid >> 1;
    const int b_half = tid & 1;
    const unsigned char* b_base =
        GATE_UP ? (const unsigned char*)table[expert * 6 + (b_row < 64 ? 0 : 2)] +
                      (unsigned long long)(n0 + (b_row & 63)) * (2 * packed_row)
                : (const unsigned char*)table[expert * 6 + 4] +
                      (unsigned long long)(n0 + b_row) * packed_row;
    const uint32_t b_dst = (uint32_t)(b_row * 8 + ((b_half ^ ((b_row >> 2) & 1)) * 4));
    const unsigned char* sc_base = nullptr;
    if (tid < 128) {
        if constexpr (GATE_UP) {
            sc_base = (const unsigned char*)table[expert * 6 + (tid < 64 ? 1 : 3)] +
                      (unsigned long long)(n0 + (tid & 63)) * (2 * scale_row);
        } else {
            sc_base = (const unsigned char*)table[expert * 6 + 5] +
                      (unsigned long long)(n0 + tid) * scale_row;
        }
    }
    auto load = [&](int buf, int ks) {
        Stage& st = stages[buf];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const __half* at = a_base[i] + ks * 64 + a_chunk * 8;
            om_cp16(om_smem(&st.a[0][a_dst[i]]), at, a_valid[i]);
            om_cp16(om_smem(&st.a[1][a_dst[i]]), at + a_plane, a_valid[i]);
        }
        om_cp16(om_smem(&st.b[b_dst]), b_base + ks * 32 + b_half * 16, true);
        if (sc_base != nullptr) {
            inkling_cp4(om_smem(&st.s[tid]), sc_base + ks * 4);
        }
    };

    const int g = lane >> 2;
    const int t = lane & 3;
    float acc[MT][NJ][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    const int ksteps = inner / 64;
    if (ksteps > 0) {
        load(0, 0);
    }
    om_commit();
    const int a_sw = (g & 1) << 2;
    const int b_sw = ((g >> 2) & 1) << 2;
    const int mt_live = min(MT, max(0, (count - wm * MT * 16 + 15) >> 4));
    for (int ks = 0; ks < ksteps; ++ks) {
        om_wait<0>();
        __syncthreads();
        if (ks + 1 < ksteps) {
            load((ks + 1) & 1, ks + 1);
        }
        om_commit();
        const Stage& st = stages[ks & 1];
        uint32_t bf[NJ][4][2];
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int row = GATE_UP ? (64 * (j >= NT_HALF ? 1 : 0) +
                                       (wn * NT_HALF + (j >= NT_HALF ? j - NT_HALF : j)) * 8 + g)
                                    : ((wn * NJ + j) * 8 + g);
            const uint32_t w0 = st.b[row * 8 + (t ^ b_sw)];
            const uint32_t w1 = st.b[row * 8 + ((t + 4) ^ b_sw)];
            const uint32_t sw = st.s[row];
            const uint16_t pair_bytes =
                (uint16_t)(((sw >> (8u * (t >> 1))) & 0xffu) |
                           (((sw >> (8u * (2u + (t >> 1)))) & 0xffu) << 8));
            uint32_t s_lo, s_hi;
            inkling_scale_pairs_f16(pair_bytes, s_lo, s_hi);
            uint32_t d0[4], d1[4];
            inkling_e2m1x8_f16(w0, d0);
            inkling_e2m1x8_f16(w1, d1);
#pragma unroll
            for (int kb = 0; kb < 4; ++kb) {
                bf[j][kb][0] = inkling_f16x2_mul(d0[kb], s_lo);
                bf[j][kb][1] = inkling_f16x2_mul(d1[kb], s_hi);
            }
        }
#pragma unroll
        for (int plane = 0; plane < 2; ++plane) {
            const __half* a = st.a[plane];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                if (mt >= mt_live) {
                    break;
                }
                const int r0 = wm * MT * 16 + mt * 16 + g;
                const uint4 lo0 = *reinterpret_cast<const uint4*>(&a[r0 * 64 + ((t ^ a_sw) * 8)]);
                const uint4 hi0 =
                    *reinterpret_cast<const uint4*>(&a[r0 * 64 + (((t + 4) ^ a_sw) * 8)]);
                const uint4 lo8 =
                    *reinterpret_cast<const uint4*>(&a[(r0 + 8) * 64 + ((t ^ a_sw) * 8)]);
                const uint4 hi8 =
                    *reinterpret_cast<const uint4*>(&a[(r0 + 8) * 64 + (((t + 4) ^ a_sw) * 8)]);
                const uint32_t l0[4] = {lo0.x, lo0.y, lo0.z, lo0.w};
                const uint32_t h0[4] = {hi0.x, hi0.y, hi0.z, hi0.w};
                const uint32_t l8[4] = {lo8.x, lo8.y, lo8.z, lo8.w};
                const uint32_t h8[4] = {hi8.x, hi8.y, hi8.z, hi8.w};
#pragma unroll
                for (int kb = 0; kb < 4; ++kb) {
                    const uint32_t af[4] = {l0[kb], l8[kb], h0[kb], h8[kb]};
#pragma unroll
                    for (int j = 0; j < NJ; ++j) {
                        inkling_f16_mma(acc[mt][j], af, bf[j][kb][0], bf[j][kb][1]);
                    }
                }
            }
        }
    }
    om_wait<0>();

    if constexpr (GATE_UP) {
        const float g2 = scales[expert * 3 + 0];
        const float u2 = scales[expert * 3 + 1];
        __half* activated = reinterpret_cast<__half*>(out);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = wm * MT * 16 + mt * 16 + g + 8 * h;
                if (r >= count) {
                    continue;
                }
                const int slot = s_slot[r];
#pragma unroll
                for (int jsub = 0; jsub < NT_HALF; ++jsub) {
                    const int col = n0 + (wn * NT_HALF + jsub) * 8 + 2 * t;
                    const float a0 = inkling_softcapped_swiglu_value(
                        acc[mt][jsub][2 * h] * g2, acc[mt][NT_HALF + jsub][2 * h] * u2, gate_cap,
                        up_cap);
                    const float a1 = inkling_softcapped_swiglu_value(
                        acc[mt][jsub][2 * h + 1] * g2, acc[mt][NT_HALF + jsub][2 * h + 1] * u2,
                        gate_cap, up_cap);
                    __half2 head, rest;
                    inkling_split2_f16(a0, a1, head, rest);
                    const unsigned long long at = (unsigned long long)slot * intermediate + col;
                    *reinterpret_cast<__half2*>(&activated[at]) = head;
                    *reinterpret_cast<__half2*>(&activated[out_plane + at]) = rest;
                }
            }
        }
    } else {
        const float d2 = scales[expert * 3 + 2];
        float* partials = reinterpret_cast<float*>(out);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = wm * MT * 16 + mt * 16 + g + 8 * h;
                if (r >= count) {
                    continue;
                }
                const int slot = s_slot[r];
                const float route =
                    route_weights[(unsigned long long)(slot / topk) * weight_stride + slot % topk];
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    const int col = n0 + (wn * NJ + j) * 8 + 2 * t;
                    *reinterpret_cast<float2*>(&partials[(unsigned long long)slot * hidden + col]) =
                        make_float2(route * (acc[mt][j][2 * h] * d2),
                                    route * (acc[mt][j][2 * h + 1] * d2));
                }
            }
        }
    }
}

template <bool GATE_UP>
__device__ __forceinline__ void inkling_moe_mx8_64_body(
    const unsigned long long* __restrict__ table, const float* __restrict__ scales,
    const int* __restrict__ sorted, const int4* __restrict__ items,
    const int* __restrict__ item_count, const __nv_bfloat16* __restrict__ a_src,
    unsigned long long a_plane, const float* __restrict__ route_weights, void* __restrict__ out,
    unsigned long long out_plane, int topk, int hidden, int intermediate, int weight_stride,
    float gate_cap, float up_cap) {
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int WM = 2;
    constexpr int MT = 2;
    constexpr int NT_HALF = 2;
    constexpr int NJ = 4;
    struct Stage {
        __nv_bfloat16 a[2][BM * 64];
        uint32_t b[BN * 16];
        uint32_t s[BN];
    };
    extern __shared__ __align__(16) uint8_t inkling_moe64_smem[];
    Stage* stages = reinterpret_cast<Stage*>(inkling_moe64_smem);
    int* s_slot = reinterpret_cast<int*>(stages + 2);
    if ((int)blockIdx.y >= *item_count) {
        return;
    }
    const int4 item = items[blockIdx.y];
    const int expert = item.x;
    const int first = item.y;
    const int count = item.z;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp % WM;
    const int wn = warp / WM;
    const int inner = GATE_UP ? hidden : intermediate;
    const int n0 = blockIdx.x * (GATE_UP ? 64 : 128);
    const int packed_row = inner;
    const int scale_row = inner >> 5;
    if (tid < BM) {
        s_slot[tid] = tid < count ? sorted[first + tid] : -1;
    }
    __syncthreads();

    const int a_chunk = tid & 7;
    const __nv_bfloat16* a_base[2];
    bool a_valid[2];
    uint32_t a_dst[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int r = (tid >> 3) + i * 32;
        const int slot = s_slot[r];
        a_valid[i] = slot >= 0;
        if (slot >= 0) {
            a_base[i] = GATE_UP ? a_src + (unsigned long long)(slot / topk) * hidden
                                : a_src + (unsigned long long)slot * intermediate;
        } else {
            a_base[i] = a_src;
        }
        a_dst[i] = (uint32_t)(r * 64 + ((a_chunk ^ ((r & 1) << 2)) * 8));
    }
    const int b_row = tid >> 1;
    const int b_half = tid & 1;
    const unsigned char* b_base =
        GATE_UP ? (const unsigned char*)table[expert * 6 + (b_row < 64 ? 0 : 2)] +
                      (unsigned long long)(n0 + (b_row & 63)) * (2 * packed_row)
                : (const unsigned char*)table[expert * 6 + 4] +
                      (unsigned long long)(n0 + b_row) * packed_row;
    const int mx_swizzle = ((b_row >> 1) & 1) << 1;
    const uint32_t mx_dst0 = (uint32_t)(b_row * 16 + (((2 * b_half) ^ mx_swizzle) * 4));
    const uint32_t mx_dst1 = (uint32_t)(b_row * 16 + (((2 * b_half + 1) ^ mx_swizzle) * 4));
    auto scale_base = [&](int row) -> const unsigned char* {
        if constexpr (GATE_UP) {
            return (const unsigned char*)table[expert * 6 + (row < 64 ? 1 : 3)] +
                   (unsigned long long)(n0 + (row & 63)) * (2 * scale_row);
        }
        return (const unsigned char*)table[expert * 6 + 5] +
               (unsigned long long)(n0 + row) * scale_row;
    };
    const unsigned char* sc_base = tid < 128 ? scale_base(tid) : nullptr;
    auto load = [&](int buf, int ks) {
        Stage& st = stages[buf];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const __nv_bfloat16* at = a_base[i] + ks * 64 + a_chunk * 8;
            om_cp16(om_smem(&st.a[0][a_dst[i]]), at, a_valid[i]);
            om_cp16(om_smem(&st.a[1][a_dst[i]]), at + a_plane, a_valid[i]);
        }
        const unsigned char* src = b_base + ks * 64 + b_half * 32;
        om_cp16(om_smem(&st.b[mx_dst0]), src, true);
        om_cp16(om_smem(&st.b[mx_dst1]), src + 16, true);
        if (sc_base != nullptr) {
            inkling_cp4(om_smem(&st.s[tid]),
                        (const void*)((uintptr_t)(sc_base + 2 * ks) & ~(uintptr_t)3));
        }
    };

    const int g = lane >> 2;
    const int t = lane & 3;
    float acc[MT][NJ][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    const int ksteps = inner / 64;
    if (ksteps > 0) {
        load(0, 0);
    }
    om_commit();
    const int a_sw = (g & 1) << 2;
    const int mx_sw = ((g >> 1) & 1) << 1;
    int mx_half[NJ];
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const int row = GATE_UP ? (64 * (j >= NT_HALF ? 1 : 0) +
                                   (wn * NT_HALF + (j >= NT_HALF ? j - NT_HALF : j)) * 8 + g)
                                : ((wn * NJ + j) * 8 + g);
        mx_half[j] = (int)(((uintptr_t)scale_base(row) >> 1) & 1);
    }
    const int mt_live = min(MT, max(0, (count - wm * MT * 16 + 15) >> 4));
    for (int ks = 0; ks < ksteps; ++ks) {
        om_wait<0>();
        __syncthreads();
        if (ks + 1 < ksteps) {
            load((ks + 1) & 1, ks + 1);
        }
        om_commit();
        const Stage& st = stages[ks & 1];
        uint32_t bf[NJ][4][2];
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int row = GATE_UP ? (64 * (j >= NT_HALF ? 1 : 0) +
                                       (wn * NT_HALF + (j >= NT_HALF ? j - NT_HALF : j)) * 8 + g)
                                    : ((wn * NJ + j) * 8 + g);
            const uint32_t* br = &st.b[row * 16];
            const uint2 lo =
                *reinterpret_cast<const uint2*>(&br[(((t >> 1) ^ mx_sw) * 4) + (t & 1) * 2]);
            const uint2 hi =
                *reinterpret_cast<const uint2*>(&br[(((2 + (t >> 1)) ^ mx_sw) * 4) + (t & 1) * 2]);
            const uint32_t sw = st.s[row] >> (16 * ((mx_half[j] + ks) & 1));
            const float s_lo = inkling_ue8m0((unsigned char)(sw & 0xffu));
            const float s_hi = inkling_ue8m0((unsigned char)((sw >> 8) & 0xffu));
#pragma unroll
            for (int kb = 0; kb < 4; ++kb) {
                const uint32_t wl = kb < 2 ? lo.x : lo.y;
                const uint32_t wh = kb < 2 ? hi.x : hi.y;
                bf[j][kb][0] = inkling_e4m3x2_scaled_bf16(wl >> (16 * (kb & 1)), s_lo);
                bf[j][kb][1] = inkling_e4m3x2_scaled_bf16(wh >> (16 * (kb & 1)), s_hi);
            }
        }
#pragma unroll
        for (int plane = 0; plane < 2; ++plane) {
            const __nv_bfloat16* a = st.a[plane];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                if (mt >= mt_live) {
                    break;
                }
                const int r0 = wm * MT * 16 + mt * 16 + g;
                const uint4 lo0 = *reinterpret_cast<const uint4*>(&a[r0 * 64 + ((t ^ a_sw) * 8)]);
                const uint4 hi0 =
                    *reinterpret_cast<const uint4*>(&a[r0 * 64 + (((t + 4) ^ a_sw) * 8)]);
                const uint4 lo8 =
                    *reinterpret_cast<const uint4*>(&a[(r0 + 8) * 64 + ((t ^ a_sw) * 8)]);
                const uint4 hi8 =
                    *reinterpret_cast<const uint4*>(&a[(r0 + 8) * 64 + (((t + 4) ^ a_sw) * 8)]);
                const uint32_t l0[4] = {lo0.x, lo0.y, lo0.z, lo0.w};
                const uint32_t h0[4] = {hi0.x, hi0.y, hi0.z, hi0.w};
                const uint32_t l8[4] = {lo8.x, lo8.y, lo8.z, lo8.w};
                const uint32_t h8[4] = {hi8.x, hi8.y, hi8.z, hi8.w};
#pragma unroll
                for (int kb = 0; kb < 4; ++kb) {
                    const uint32_t af[4] = {l0[kb], l8[kb], h0[kb], h8[kb]};
#pragma unroll
                    for (int j = 0; j < NJ; ++j) {
                        ob_mma(acc[mt][j], af, bf[j][kb][0], bf[j][kb][1]);
                    }
                }
            }
        }
    }
    om_wait<0>();

    if constexpr (GATE_UP) {
        const float g2 = scales[expert * 3 + 0];
        const float u2 = scales[expert * 3 + 1];
        __nv_bfloat16* activated = reinterpret_cast<__nv_bfloat16*>(out);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = wm * MT * 16 + mt * 16 + g + 8 * h;
                if (r >= count) {
                    continue;
                }
                const int slot = s_slot[r];
#pragma unroll
                for (int jsub = 0; jsub < NT_HALF; ++jsub) {
                    const int col = n0 + (wn * NT_HALF + jsub) * 8 + 2 * t;
                    const float a0 = inkling_softcapped_swiglu_value(
                        acc[mt][jsub][2 * h] * g2, acc[mt][NT_HALF + jsub][2 * h] * u2, gate_cap,
                        up_cap);
                    const float a1 = inkling_softcapped_swiglu_value(
                        acc[mt][jsub][2 * h + 1] * g2, acc[mt][NT_HALF + jsub][2 * h + 1] * u2,
                        gate_cap, up_cap);
                    __nv_bfloat162 head, rest;
                    inkling_split2_bf16(a0, a1, head, rest);
                    const unsigned long long at = (unsigned long long)slot * intermediate + col;
                    *reinterpret_cast<__nv_bfloat162*>(&activated[at]) = head;
                    *reinterpret_cast<__nv_bfloat162*>(&activated[out_plane + at]) = rest;
                }
            }
        }
    } else {
        const float d2 = scales[expert * 3 + 2];
        float* partials = reinterpret_cast<float*>(out);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = wm * MT * 16 + mt * 16 + g + 8 * h;
                if (r >= count) {
                    continue;
                }
                const int slot = s_slot[r];
                const float route =
                    route_weights[(unsigned long long)(slot / topk) * weight_stride + slot % topk];
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    const int col = n0 + (wn * NJ + j) * 8 + 2 * t;
                    *reinterpret_cast<float2*>(&partials[(unsigned long long)slot * hidden + col]) =
                        make_float2(route * (acc[mt][j][2 * h] * d2),
                                    route * (acc[mt][j][2 * h + 1] * d2));
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 2) void inkling_moe_gate_up_f16_64(
    const unsigned long long* __restrict__ table, const float* __restrict__ scales,
    const int* __restrict__ sorted, const int4* __restrict__ items,
    const int* __restrict__ item_count, const __half* __restrict__ x, unsigned long long x_plane,
    __half* __restrict__ activated, unsigned long long activated_plane, int topk, int hidden,
    int intermediate, float gate_cap, float up_cap) {
    OPT_PDL_ENTRY();
    inkling_moe_f16_64_body<true>(table, scales, sorted, items, item_count, x, x_plane, nullptr,
                                  activated, activated_plane, topk, hidden, intermediate, 0,
                                  gate_cap, up_cap);
}

extern "C" __global__ __launch_bounds__(256, 2) void inkling_moe_down_f16_64(
    const unsigned long long* __restrict__ table, const float* __restrict__ scales,
    const int* __restrict__ sorted, const int4* __restrict__ items,
    const int* __restrict__ item_count, const __half* __restrict__ activated,
    unsigned long long activated_plane, const float* __restrict__ route_weights,
    float* __restrict__ partials, int topk, int hidden, int intermediate, int weight_stride) {
    OPT_PDL_ENTRY();
    inkling_moe_f16_64_body<false>(table, scales, sorted, items, item_count, activated,
                                   activated_plane, route_weights, partials, 0, topk, hidden,
                                   intermediate, weight_stride, 0.0f, 0.0f);
}

extern "C" __global__ __launch_bounds__(256, 2) void inkling_moe_gate_up_mx8_64(
    const unsigned long long* __restrict__ table, const float* __restrict__ scales,
    const int* __restrict__ sorted, const int4* __restrict__ items,
    const int* __restrict__ item_count, const __nv_bfloat16* __restrict__ x,
    unsigned long long x_plane, __nv_bfloat16* __restrict__ activated,
    unsigned long long activated_plane, int topk, int hidden, int intermediate, float gate_cap,
    float up_cap) {
    OPT_PDL_ENTRY();
    inkling_moe_mx8_64_body<true>(table, scales, sorted, items, item_count, x, x_plane, nullptr,
                                  activated, activated_plane, topk, hidden, intermediate, 0,
                                  gate_cap, up_cap);
}

extern "C" __global__ __launch_bounds__(256, 2) void inkling_moe_down_mx8_64(
    const unsigned long long* __restrict__ table, const float* __restrict__ scales,
    const int* __restrict__ sorted, const int4* __restrict__ items,
    const int* __restrict__ item_count, const __nv_bfloat16* __restrict__ activated,
    unsigned long long activated_plane, const float* __restrict__ route_weights,
    float* __restrict__ partials, int topk, int hidden, int intermediate, int weight_stride) {
    OPT_PDL_ENTRY();
    inkling_moe_mx8_64_body<false>(table, scales, sorted, items, item_count, activated,
                                   activated_plane, route_weights, partials, 0, topk, hidden,
                                   intermediate, weight_stride, 0.0f, 0.0f);
}

// Pure BF16 x BF16 -> F32 dense MMA (BM=128, BN=256, BK=64, 2 stages, 98,304 B SMEM):
// reads BF16 activations directly (e.g. scratch.attn_bf16 from inkling_attn_mma_g3), skipping
// Op::Widen and halving attention output DRAM traffic while running at ~82 TFLOP/s on GB10.
// Grid (ceil(cols / 256), ceil(tokens / 128), 1), 256 threads, 98,304 B dynamic shared memory.
extern "C" __global__ __launch_bounds__(256, 1) void inkling_dense_bf16_mma(
    const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ a,
    float* __restrict__ out, uint32_t cols, uint32_t inner, uint32_t tokens) {
    OPT_PDL_ENTRY();
    constexpr uint32_t BM = 128u;
    constexpr uint32_t BN = 256u;
    constexpr uint32_t BK = 64u;
    constexpr uint32_t WM = 2u;
    constexpr uint32_t MT = 4u;
    constexpr uint32_t NT = 8u;
    constexpr uint32_t CHUNKS = 8u;
    constexpr uint32_t A_ITERS = (BM * CHUNKS) / 256u;
    constexpr uint32_t W_ITERS = (BN * CHUNKS) / 256u;

    struct Stage {
        __nv_bfloat16 a[BM * BK];
        __nv_bfloat16 w[BN * BK];
    };
    extern __shared__ __align__(128) uint8_t inkling_dense_smem[];
    Stage* stages = reinterpret_cast<Stage*>(inkling_dense_smem);

    // 8-wide rasterization along X for L2 reuse of weights and activations.
    constexpr uint32_t kGroup = 8u;
    const uint32_t linear = blockIdx.y * gridDim.x + blockIdx.x;
    const uint32_t per_strip = gridDim.y * kGroup;
    const uint32_t strip = linear / per_strip;
    const uint32_t rem = linear - strip * per_strip;
    const uint32_t width = min(kGroup, gridDim.x - strip * kGroup);
    const uint32_t n0 = (strip * kGroup + rem % width) * BN;
    const uint32_t m0 = (rem / width) * BM;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t wm = warp % WM;
    const uint32_t wn = warp / WM;
    const uint32_t ksteps = inner / BK;

    auto swiz = [](uint32_t row, uint32_t chunk) -> uint32_t {
        return (row * 8u + (chunk ^ (row & 7u))) * 8u;
    };

    auto load = [&](uint32_t buf, uint32_t ks) {
        Stage& st = stages[buf];
        const uint32_t k0 = ks * BK;
#pragma unroll
        for (uint32_t i = 0; i < A_ITERS; ++i) {
            const uint32_t idx = tid + i * 256u;
            const uint32_t row = idx >> 3;
            const uint32_t ck = idx & 7u;
            const bool valid = m0 + row < tokens;
            const __nv_bfloat16* src = valid ? a + (uint64_t)(m0 + row) * inner + k0 + ck * 8u : a;
            om_cp16(om_smem(&st.a[swiz(row, ck)]), src, valid);
        }
#pragma unroll
        for (uint32_t i = 0; i < W_ITERS; ++i) {
            const uint32_t idx = tid + i * 256u;
            const uint32_t row = idx >> 3;
            const uint32_t ck = idx & 7u;
            const bool valid = n0 + row < cols;
            const __nv_bfloat16* src = valid ? w + (uint64_t)(n0 + row) * inner + k0 + ck * 8u : w;
            om_cp16(om_smem(&st.w[swiz(row, ck)]), src, valid);
        }
    };

    float acc[MT][NT][4];
#pragma unroll
    for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) acc[i][j][e] = 0.0f;

    if (ksteps > 0u) {
        load(0u, 0u);
    }
    om_commit();
    for (uint32_t ks = 0u; ks < ksteps; ++ks) {
        om_wait<0>();
        __syncthreads();
        if (ks + 1u < ksteps) {
            load((ks + 1u) & 1u, ks + 1u);
        }
        om_commit();
        const Stage& st = stages[ks & 1u];
#pragma unroll
        for (uint32_t ksub = 0u; ksub < 4u; ++ksub) {
            const uint32_t ck0 = ksub * 2u;
            uint32_t b[NT][2];
#pragma unroll
            for (uint32_t p = 0u; p < NT / 2u; ++p) {
                const uint32_t n = wn * (NT * 8u) + p * 16u + (lane & 7u) + 8u * (lane >> 4);
                const uint32_t ck = ck0 + ((lane >> 3) & 1u);
                om_ldsm4(om_smem(&st.w[swiz(n, ck)]), b[2u * p][0], b[2u * p][1],
                         b[2u * p + 1u][0], b[2u * p + 1u][1]);
            }
#pragma unroll
            for (uint32_t mt = 0u; mt < MT; ++mt) {
                const uint32_t r = wm * (MT * 16u) + mt * 16u + (lane & 15u);
                const uint32_t ck = ck0 + (lane >> 4);
                uint32_t af[4];
                om_ldsm4(om_smem(&st.a[swiz(r, ck)]), af[0], af[1], af[2], af[3]);
#pragma unroll
                for (uint32_t nt = 0u; nt < NT; ++nt) {
                    ob_mma(acc[mt][nt], af, b[nt][0], b[nt][1]);
                }
            }
        }
    }
    om_wait<0>();
    const uint32_t g = lane >> 2;
    const uint32_t q = lane & 3u;
#pragma unroll
    for (uint32_t mt = 0u; mt < MT; ++mt) {
#pragma unroll
        for (uint32_t half = 0u; half < 2u; ++half) {
            const uint32_t token = m0 + wm * (MT * 16u) + mt * 16u + g + 8u * half;
            if (token >= tokens) {
                continue;
            }
            const uint64_t base = (uint64_t)token * cols + n0 + wn * (NT * 8u) + 2u * q;
#pragma unroll
            for (uint32_t nt = 0u; nt < NT; ++nt) {
                const uint32_t col = n0 + wn * (NT * 8u) + nt * 8u + 2u * q;
                if (col < cols) {
                    *reinterpret_cast<float2*>(out + base + nt * 8u) =
                        make_float2(acc[mt][nt][2u * half], acc[mt][nt][2u * half + 1u]);
                }
            }
        }
    }
}

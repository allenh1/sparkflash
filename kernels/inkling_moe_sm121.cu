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

#include <cuda_runtime.h>

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
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const unsigned short lo = (unsigned short)(pairs[p] & 0xFFFFu);
        const unsigned short hi = (unsigned short)(pairs[p] >> 16);
        float a, b;
        asm("cvt.f32.f16 %0, %1;" : "=f"(a) : "h"(lo));
        asm("cvt.f32.f16 %0, %1;" : "=f"(b) : "h"(hi));
        out[2 * p] = a;
        out[2 * p + 1] = b;
    }
    // A byte whose low seven bits are all set is the format's NaN.
    const unsigned int marked = (word & 0x7F7F7F7Fu) ^ 0x7F7F7F7Fu;
    if (((marked - 0x01010101u) & ~marked & 0x80808080u) != 0u) {
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

// A decode band's NVFP4 projection with the router's choice read on the device, so no host ever
// waits for it: blockIdx.x is a group of INKLING_GROUP_ROWS rows and blockIdx.y the band slot.
// indices[slot] is the global expert; this rank holds the contiguous run [first, first + owned)
// of them, indexed from zero, and a block whose slot names an expert outside the run exits (the
// mix gives that slot no share). The input is one row for every slot (the gate/up projection
// reads the token's stream) or a row a slot (the down projection reads the slot's activation):
// `per_slot` picks which. columns is a multiple of 16 (the arm refuses any other width).
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
    int per_slot) {
    const int slot = blockIdx.y;
    const int local = indices[slot] - first;
    if (local < 0 || local >= owned || columns <= 0) {
        return;
    }
    __shared__ float e4m3[256];
    for (int code = threadIdx.x; code < 256; code += blockDim.x) {
        e4m3[code] = inkling_e4m3((unsigned char)code);
    }
    __syncthreads();
    const int row = blockIdx.x * INKLING_GROUP_ROWS + (threadIdx.x / INKLING_GROUP);
    const int q = threadIdx.x & (INKLING_GROUP - 1);
    const int packed = (columns + 1) / 2;
    const int scale_columns = (columns + 15) / 16;
    const int live = row < rows ? row : rows - 1;
    const unsigned char* __restrict__ row_codes =
        codes + ((uint64_t)local * rows + live) * packed;
    const unsigned char* __restrict__ row_scales =
        scales + ((uint64_t)local * rows + live) * scale_columns;
    x += per_slot ? (uint64_t)slot * columns : 0;
    float v[INKLING_GROUP];
#pragma unroll
    for (int i = 0; i < INKLING_GROUP; ++i) {
        v[i] = 0.0f;
    }
#pragma unroll 2
    for (int base = INKLING_GROUP * q; base < columns; base += INKLING_MOE_BLOCK) {
        const uint2 word = *reinterpret_cast<const uint2*>(row_codes + (base >> 1));
        const float block = e4m3[row_scales[base >> 4]];
        const float4* __restrict__ in = reinterpret_cast<const float4*>(x + base);
        float weight[INKLING_GROUP];
        float half_weights[8];
        inkling_e2m1x8(word.x, half_weights);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            weight[j] = half_weights[j];
        }
        inkling_e2m1x8(word.y, half_weights);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            weight[8 + j] = half_weights[j];
        }
#pragma unroll
        for (int f = 0; f < 4; ++f) {
            const float4 x4 = in[f];
            v[4 * f + 0] = __fadd_rn(v[4 * f + 0], __fmul_rn(__fmul_rn(weight[4 * f + 0], block), x4.x));
            v[4 * f + 1] = __fadd_rn(v[4 * f + 1], __fmul_rn(__fmul_rn(weight[4 * f + 1], block), x4.y));
            v[4 * f + 2] = __fadd_rn(v[4 * f + 2], __fmul_rn(__fmul_rn(weight[4 * f + 2], block), x4.z));
            v[4 * f + 3] = __fadd_rn(v[4 * f + 3], __fmul_rn(__fmul_rn(weight[4 * f + 3], block), x4.w));
        }
    }
    const float total = inkling_group_row_sum(v);
    if (q == 0 && row < rows) {
        out[(uint64_t)slot * rows + row] = __fmul_rn(total, scale_2[local]);
    }
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
// of `block` columns (a multiple of 16, so a run of sixteen never straddles two), and
// inkling_mxfp8_row's arithmetic for every row.
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
    int per_slot) {
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
    x += per_slot ? (uint64_t)slot * columns : 0;
    float v[INKLING_GROUP];
#pragma unroll
    for (int i = 0; i < INKLING_GROUP; ++i) {
        v[i] = 0.0f;
    }
#pragma unroll 2
    for (int base = INKLING_GROUP * q; base < columns; base += INKLING_MOE_BLOCK) {
        const uint4 word = *reinterpret_cast<const uint4*>(row_codes + base);
        const float scale = inkling_ue8m0(row_scales[base / block]);
        const float4* __restrict__ in = reinterpret_cast<const float4*>(x + base);
        const unsigned int words[4] = {word.x, word.y, word.z, word.w};
#pragma unroll
        for (int f = 0; f < 4; ++f) {
            float weight[4];
            inkling_e4m3x4(words[f], weight);
            const float4 x4 = in[f];
            v[4 * f + 0] = __fadd_rn(v[4 * f + 0], __fmul_rn(__fmul_rn(weight[0], scale), x4.x));
            v[4 * f + 1] = __fadd_rn(v[4 * f + 1], __fmul_rn(__fmul_rn(weight[1], scale), x4.y));
            v[4 * f + 2] = __fadd_rn(v[4 * f + 2], __fmul_rn(__fmul_rn(weight[2], scale), x4.z));
            v[4 * f + 3] = __fadd_rn(v[4 * f + 3], __fmul_rn(__fmul_rn(weight[3], scale), x4.w));
        }
    }
    const float total = inkling_group_row_sum(v);
    if (q == 0 && row < rows) {
        out[(uint64_t)slot * rows + row] = total;
    }
}

// A chunk's expert projection over the (input row, output slot) pairs that chose one expert:
// the prefill twin of the band kernels above. inkling_*_expert_matvec computes a pair a block
// row, so every pair re-reads the expert's weights; here a block owns INKLING_GROUP_ROWS rows and
// walks every pair of the expert INKLING_PAIRS at a time, so a weight run is loaded and decoded
// once a group and its product with the scale feeds each pair's term. A (row, pair) sum is still
// the band kernels' - and so inkling_*_row's - term for term and tree for tree, which keeps the
// chunked prefill's bits the row-at-a-time walk's. grid (ceil(rows / INKLING_GROUP_ROWS)), block
// INKLING_MOE_BLOCK; plan holds `count` pairs, [input row, output slot] each.
#ifndef INKLING_PAIRS
#define INKLING_PAIRS 4
#endif
#ifndef INKLING_PAIRS_UNROLL
#define INKLING_PAIRS_UNROLL 2
#endif
// The column steps a pair walk keeps in flight together (an unroll pragma takes a constant).
constexpr int kInklingPairsUnroll = INKLING_PAIRS_UNROLL;

// The sixteen weight-times-scale products of one run of an NVFP4 row, inkling_nvfp4_row's
// __fmul_rn(weight, block) for each.
__device__ __forceinline__ void inkling_nvfp4_run(const unsigned char* __restrict__ row_codes,
                                                  const unsigned char* __restrict__ row_scales,
                                                  const float* __restrict__ e4m3, int base,
                                                  float (&scaled)[INKLING_GROUP]) {
    const uint2 word = *reinterpret_cast<const uint2*>(row_codes + (base >> 1));
    const float block = e4m3[row_scales[base >> 4]];
    float weight[8];
    inkling_e2m1x8(word.x, weight);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        scaled[j] = __fmul_rn(weight[j], block);
    }
    inkling_e2m1x8(word.y, weight);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        scaled[8 + j] = __fmul_rn(weight[j], block);
    }
}

// The same sixteen products for an MXFP8 row, inkling_mxfp8_row's __fmul_rn(weight, scale).
__device__ __forceinline__ void inkling_mxfp8_run(const unsigned char* __restrict__ row_codes,
                                                  const unsigned char* __restrict__ row_scales,
                                                  int block, int base,
                                                  float (&scaled)[INKLING_GROUP]) {
    const uint4 word = *reinterpret_cast<const uint4*>(row_codes + base);
    const float scale = inkling_ue8m0(row_scales[base / block]);
    const unsigned int words[4] = {word.x, word.y, word.z, word.w};
#pragma unroll
    for (int f = 0; f < 4; ++f) {
        float weight[4];
        inkling_e4m3x4(words[f], weight);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            scaled[4 * f + j] = __fmul_rn(weight[j], scale);
        }
    }
}

// A step's inputs for a pair group, staged in shared memory: the 256 columns of a step for each
// pair, as sixteen runs of sixteen floats padded to twenty, so the sixteen lanes reading their
// runs with 128-bit loads land on distinct banks (a quarter-warp's runs start at banks 20 q mod
// 32: 0, 20, 8, 28, 16, 4, 24, 12). Read straight from global memory the same loads sit 64 bytes
// apart and cost eight L1 wavefronts each, which is what held the walk to a third of its rate.
#define INKLING_STAGE_RUN 20
#define INKLING_STAGE (INKLING_MOE_BLOCK / INKLING_GROUP * INKLING_STAGE_RUN)
// The 128-bit pieces one stage holds, and how many a thread loads.
#define INKLING_STAGE_PIECES (INKLING_PAIRS * INKLING_MOE_BLOCK / 4)
#define INKLING_STAGE_LOADS ((INKLING_STAGE_PIECES + INKLING_MOE_BLOCK - 1) / INKLING_MOE_BLOCK)

template <bool kNvfp4>
__device__ __forceinline__ void inkling_expert_pairs(const unsigned char* __restrict__ codes,
                                                     const unsigned char* __restrict__ scales,
                                                     const float* __restrict__ x,
                                                     float* __restrict__ out, int rows,
                                                     int columns, int block, float scale_2,
                                                     const int* __restrict__ plan, int count,
                                                     const float* __restrict__ e4m3) {
    __shared__ __align__(16) float staged[2][INKLING_PAIRS * INKLING_STAGE];
    const int row = blockIdx.x * INKLING_GROUP_ROWS + (threadIdx.x / INKLING_GROUP);
    const int q = threadIdx.x & (INKLING_GROUP - 1);
    const int live = row < rows ? row : rows - 1;
    const int code_row = kNvfp4 ? (columns + 1) / 2 : columns;
    const int scale_row = kNvfp4 ? (columns + 15) / 16 : (columns + block - 1) / block;
    const unsigned char* __restrict__ row_codes = codes + (uint64_t)live * code_row;
    const unsigned char* __restrict__ row_scales = scales + (uint64_t)live * scale_row;
    const int steps = (columns + INKLING_MOE_BLOCK - 1) / INKLING_MOE_BLOCK;
    for (int group = 0; group < count; group += INKLING_PAIRS) {
        const int here = count - group < INKLING_PAIRS ? count - group : INKLING_PAIRS;
        // This thread's pieces of every stage: piece = pair * 64 + run * 4 + quarter.
        const float* source[INKLING_STAGE_LOADS];
        int offset[INKLING_STAGE_LOADS];
        int target[INKLING_STAGE_LOADS];
        bool loads[INKLING_STAGE_LOADS];
#pragma unroll
        for (int l = 0; l < INKLING_STAGE_LOADS; ++l) {
            const int piece = threadIdx.x + l * INKLING_MOE_BLOCK;
            const int pair = piece / (INKLING_MOE_BLOCK / 4);
            offset[l] = 4 * (piece % (INKLING_MOE_BLOCK / 4));
            loads[l] = piece < INKLING_STAGE_PIECES && pair < here;
            source[l] = x + (uint64_t)(loads[l] ? plan[2 * (group + pair)] : 0) * columns;
            target[l] = pair * INKLING_STAGE + (offset[l] / INKLING_GROUP) * INKLING_STAGE_RUN +
                        offset[l] % INKLING_GROUP;
        }
        auto fetch = [&](int step, float4 (&value)[INKLING_STAGE_LOADS]) {
#pragma unroll
            for (int l = 0; l < INKLING_STAGE_LOADS; ++l) {
                const int column = step * INKLING_MOE_BLOCK + offset[l];
                value[l] = loads[l] && column < columns
                               ? *reinterpret_cast<const float4*>(source[l] + column)
                               : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        };
        auto stage = [&](int buffer, const float4 (&value)[INKLING_STAGE_LOADS]) {
#pragma unroll
            for (int l = 0; l < INKLING_STAGE_LOADS; ++l) {
                if (threadIdx.x + l * INKLING_MOE_BLOCK < INKLING_STAGE_PIECES) {
                    *reinterpret_cast<float4*>(&staged[buffer][target[l]]) = value[l];
                }
            }
        };
        float4 pending[INKLING_STAGE_LOADS];
        fetch(0, pending);
        stage(0, pending);
        __syncthreads();
        float v[INKLING_PAIRS][INKLING_GROUP];
#pragma unroll
        for (int p = 0; p < INKLING_PAIRS; ++p) {
#pragma unroll
            for (int i = 0; i < INKLING_GROUP; ++i) {
                v[p][i] = 0.0f;
            }
        }
        for (int step = 0; step < steps; ++step) {
            const int current = step & 1;
            if (step + 1 < steps) {
                fetch(step + 1, pending);
            }
            const int base = step * INKLING_MOE_BLOCK + INKLING_GROUP * q;
            if (base < columns) {
                float scaled[INKLING_GROUP];
                if (kNvfp4) {
                    inkling_nvfp4_run(row_codes, row_scales, e4m3, base, scaled);
                } else {
                    inkling_mxfp8_run(row_codes, row_scales, block, base, scaled);
                }
#pragma unroll
                for (int p = 0; p < INKLING_PAIRS; ++p) {
                    if (p < here) {
                        const float* in = &staged[current][p * INKLING_STAGE + q * INKLING_STAGE_RUN];
#pragma unroll
                        for (int f = 0; f < 4; ++f) {
                            const float4 x4 = *reinterpret_cast<const float4*>(in + 4 * f);
                            v[p][4 * f + 0] = __fadd_rn(v[p][4 * f + 0], __fmul_rn(scaled[4 * f + 0], x4.x));
                            v[p][4 * f + 1] = __fadd_rn(v[p][4 * f + 1], __fmul_rn(scaled[4 * f + 1], x4.y));
                            v[p][4 * f + 2] = __fadd_rn(v[p][4 * f + 2], __fmul_rn(scaled[4 * f + 2], x4.z));
                            v[p][4 * f + 3] = __fadd_rn(v[p][4 * f + 3], __fmul_rn(scaled[4 * f + 3], x4.w));
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
        for (int p = 0; p < INKLING_PAIRS; ++p) {
            const float total = inkling_group_row_sum(v[p]);
            if (q == 0 && row < rows && p < here) {
                const int slot = plan[2 * (group + p) + 1];
                out[(uint64_t)slot * rows + row] = kNvfp4 ? __fmul_rn(total, scale_2) : total;
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
    if (columns <= 0 || count <= 0) {
        return;
    }
    __shared__ float e4m3[256];
    for (int code = threadIdx.x; code < 256; code += blockDim.x) {
        e4m3[code] = inkling_e4m3((unsigned char)code);
    }
    __syncthreads();
    inkling_expert_pairs<true>(codes, scales, x, out, rows, columns, 16, scale_2, plan, count, e4m3);
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
    if (columns <= 0 || count <= 0 || block <= 0) {
        return;
    }
    inkling_expert_pairs<false>(codes, scales, x, out, rows, columns, block, 1.0f, plan, count,
                                nullptr);
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
#define INKLING_ROUTER_THREADS 128
extern "C" __global__ void __launch_bounds__(INKLING_ROUTER_THREADS) inkling_router_topk(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int* __restrict__ indices,
    float* __restrict__ weights,
    int experts,
    int top_k,
    float route_scale) {
    if (experts <= 0 || top_k <= 0 || top_k > experts || top_k > 64) {
        return;
    }
    // One block a row: a prefill chunk routes every row of the chunk, a decode step its one row
    // (grid 1, which leaves the three offsets below at zero).
    const int row = blockIdx.x;
    logits += (uint64_t)row * experts;
    indices += (uint64_t)row * top_k;
    weights += (uint64_t)row * top_k;
    const int capacity = 4096;
    if (experts > capacity) {
        return;
    }
    __shared__ float scores[capacity];
    __shared__ float ranked[capacity];
    __shared__ float best_scores[INKLING_ROUTER_THREADS / 32];
    __shared__ int best_experts[INKLING_ROUTER_THREADS / 32];
    __shared__ int chosen[64];
    for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        const float score = 1.0f / (1.0f + expf(-logits[expert]));
        scores[expert] = score;
        ranked[expert] = __fadd_rn(score, bias != nullptr ? bias[expert] : 0.0f);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = (blockDim.x + 31) >> 5;
    for (int slot = 0; slot < top_k; ++slot) {
        int best = -1;
        float best_score = -INFINITY;
        for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
            bool taken = false;
            for (int earlier = 0; earlier < slot; ++earlier) {
                taken = taken || chosen[earlier] == expert;
            }
            const float value = ranked[expert];
            if (!taken && value > best_score) {
                best_score = value;
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
        if (lane == 0) {
            best_scores[warp] = best_score;
            best_experts[warp] = best;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int winner = -1;
            float winner_score = -INFINITY;
            for (int index = 0; index < warps; ++index) {
                const int other = best_experts[index];
                const float other_score = best_scores[index];
                if (other >= 0 && (winner < 0 || other_score > winner_score ||
                                   (other_score == winner_score && other < winner))) {
                    winner_score = other_score;
                    winner = other;
                }
            }
            chosen[slot] = winner;
        }
        __syncthreads();
        if (chosen[slot] < 0) {
            return;
        }
    }
    if (threadIdx.x != 0) {
        return;
    }
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

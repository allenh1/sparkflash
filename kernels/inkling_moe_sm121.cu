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

// One expert's NVFP4 matrix-vector product: one block a row, out[row] = scale_2 * (W x).
extern "C" __global__ void __launch_bounds__(INKLING_MOE_BLOCK) inkling_nvfp4_expert_matvec(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int columns,
    float scale_2) {
    const int row = blockIdx.x;
    if (row >= rows || columns <= 0) {
        return;
    }
    const int packed = (columns + 1) / 2;
    const int scale_columns = (columns + 15) / 16;
    const unsigned char* row_codes = codes + (uint64_t)row * packed;
    const unsigned char* row_scales = scales + (uint64_t)row * scale_columns;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const unsigned char byte = row_codes[column >> 1];
        const unsigned char nibble = (column & 1) ? (byte >> 4) : (byte & 0x0F);
        const float weight = inkling_e2m1(nibble);
        const float block = inkling_e4m3(row_scales[column >> 4]);
        sum = __fadd_rn(sum, __fmul_rn(__fmul_rn(weight, block), x[column]));
    }
    __shared__ float scratch[INKLING_MOE_BLOCK / 32 + 1];
    const float total = inkling_moe_row_sum(sum, scratch);
    if (threadIdx.x == 0) {
        out[row] = __fmul_rn(total, scale_2);
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
    int block) {
    const int row = blockIdx.x;
    if (row >= rows || columns <= 0 || block <= 0) {
        return;
    }
    const int scale_columns = (columns + block - 1) / block;
    const unsigned char* row_codes = codes + (uint64_t)row * columns;
    const unsigned char* row_scales = scales + (uint64_t)row * scale_columns;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float weight = inkling_e4m3(row_codes[column]);
        const float scale = inkling_ue8m0(row_scales[column / block]);
        sum = __fadd_rn(sum, __fmul_rn(__fmul_rn(weight, scale), x[column]));
    }
    __shared__ float scratch[INKLING_MOE_BLOCK / 32 + 1];
    const float total = inkling_moe_row_sum(sum, scratch);
    if (threadIdx.x == 0) {
        out[row] = total;
    }
}

// The sigmoid router: a score an expert, the learned bias on the selection, and the selected
// experts' shares. One thread walks the 384 experts, so the order is the host's order exactly
// (ties keep the lower index) and the schedule is a function of the scores alone.
extern "C" __global__ void __launch_bounds__(32) inkling_router_topk(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int* __restrict__ indices,
    float* __restrict__ weights,
    int experts,
    int top_k,
    float route_scale) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || experts <= 0 || top_k <= 0 || top_k > experts) {
        return;
    }
    const int capacity = 4096;
    __shared__ float scores[capacity];
    __shared__ unsigned char taken[capacity];
    for (int expert = 0; expert < experts; ++expert) {
        scores[expert] = 1.0f / (1.0f + expf(-logits[expert]));
        taken[expert] = 0;
    }
    float selected[64];
    int chosen[64];
    for (int slot = 0; slot < top_k; ++slot) {
        int best = -1;
        float best_score = -INFINITY;
        for (int expert = 0; expert < experts; ++expert) {
            if (taken[expert]) {
                continue;
            }
            const float ranked = __fadd_rn(scores[expert], bias != nullptr ? bias[expert] : 0.0f);
            if (ranked > best_score) {
                best_score = ranked;
                best = expert;
            }
        }
        if (best < 0) {
            return;
        }
        taken[best] = 1;
        chosen[slot] = best;
        selected[slot] = scores[best];
    }
    float sum = 0.0f;
    for (int slot = 0; slot < top_k; ++slot) {
        sum = __fadd_rn(sum, selected[slot]);
    }
    const float denominator = __fadd_rn(sum, 1e-20f);
    for (int slot = 0; slot < top_k; ++slot) {
        indices[slot] = chosen[slot];
        weights[slot] = __fmul_rn(__fdiv_rn(selected[slot], denominator), route_scale);
    }
}

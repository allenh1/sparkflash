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

// kernels/inkling_epilogue_sm121.cu
//
// The two Inkling-2-Small epilogues the shared stack does not have, and the NVFP4
// four-over-six activation quantiser that sits beside the shared MX quantisers.
//
// The routed experts' gate/up pair is the shared grouped MMA; only the activation differs.
// The released model caps the gate at tanh(g / 4) and the up projection at tanh(u / 25),
// so the epilogue is
//
//     out = tanh(gate / gate_cap) * sigmoid(gate) * tanh(up / up_cap)
//
// with gate_cap = 4 and up_cap = 25. Unlike a plain SwiGLU the gate's first factor is the
// softcap itself, not silu's g * sigmoid(g); the sigmoid is the same. A non-positive or
// non-finite cap disables the corresponding tanh, which is how the arm passes a layer that
// declares no cap.
//
// The final head reduces as usual; this image only applies the logit softcap
// tanh(logit / 30) * 30 and copies the real 200,058 entries out of the padded 201,024-row
// vocabulary, in BF16, so the sampler never sees a padded row.
//
// The four-over-six quantiser is the released activation rule for the layers whose experts
// are stored NVFP4: one E4M3 block scale a 16-element block, chosen between the candidate
// scales amax / 4 and amax / 6 by the smaller mean absolute error after the E2M1 round trip.
// It is a thin addition beside the shared quantisers: it emits the same packed E2M1 codes
// and one E4M3 scale code a block, and the caller's per-expert outer scale is applied after
// this kernel (this image does not pretend to know the checkpoint's amax denominator).
//
// ABI:
//
//   inkling_softcapped_swiglu(gate_up, out, width, gate_cap, up_cap)
//     gate_up  [rows][2 * width] f32, gate row first then up
//     out      [rows][width] f32
//
//   inkling_logit_softcap(logits, out, rows, padded_vocab, unpadded_vocab, cap)
//     logits   [rows][padded_vocab] f32
//     out      [rows][unpadded_vocab] BF16
//
//   inkling_nvfp4_quantize_4_6(x, packed, scales, rows, columns)
//     x        [rows][columns] f32, columns a multiple of 16
//     packed   [rows][columns / 2] uint8, two E2M1 codes a byte, low nibble first
//     scales   [rows][columns / 16] uint8, E4M3 block scales
//
// The two seam kernels the arm's schedule needs and no shared image has. Both are exact: a
// round-to-nearest-wide conversion and an addition, so their parity is a bit comparison rather
// than a tolerance.
//
//   inkling_kv_store(key, value, key_cache, value_cache, rows, position, width)
//     key, value        [rows][width] f32, the projected rows
//     key_cache, value_cache  [positions][width] BF16, written at position + row
//     The attention body reads BF16 cache rows because that is what the released model stores;
//     the projection that feeds it is f32, so the narrowing is its own launch and not a silent
//     argument to the attention kernel.
//
//   inkling_residual_add(residual, delta, out, rows, width)
//     residual, delta, out  [rows][width] f32; out may alias either input
//     The two residual topologies the block-attention stream needs: the additive residual and,
//     at a block boundary, the same expression with the summary stack as the first term.
//
//   inkling_widen_f32(src, dst, count)
//     src  BF16, dst f32; the widening is a bit insertion, so it rounds nothing
//     The attention body writes BF16 (the released cache and output precision) and the output
//     projection that follows reads f32, so this is the one conversion between them. It is the
//     inverse of what inkling_kv_store does on the key/value path.
//
//   inkling_expert_mix(out, partials, scales, slots, width)
//     partials  [rows][slots][width] f32, one selected expert's projection a slot
//     scales    [rows][slots] f32, the expert's share of the mix
//     out       [rows][width] f32
//     The three expert formats differ only in where their per-expert factor comes from, so the
//     mix is one body: the arm hands it the router weight times whatever scale the format stores
//     on the expert, and the sum is the layer's MLP output. It is a plain sum of scaled rows, in
//     slot order, so the host's sequential f32 sum and this kernel agree to the last bit.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "m4_common_sm121.cuh"

#define INKLING_EPI_BLOCK 256

// tanh through exp, so the epilogue depends only on expf and the same expression is available
// to the clang syntax probe's stub. Saturating the input keeps the exponent finite.
__device__ __forceinline__ float inkling_tanh(float value) {
    if (value > 20.0f) {
        return 1.0f;
    }
    if (value < -20.0f) {
        return -1.0f;
    }
    const float exponential = expf(2.0f * value);
    return __fdiv_rn(__fadd_rn(exponential, -1.0f), __fadd_rn(exponential, 1.0f));
}

__device__ __forceinline__ float inkling_sigmoid(float value) {
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-value)));
}

extern "C" __global__ void __launch_bounds__(INKLING_EPI_BLOCK) inkling_softcapped_swiglu(
    const float* __restrict__ gate_up,
    float* __restrict__ out,
    int width,
    float gate_cap,
    float up_cap) {
    const int row = blockIdx.y;
    const int column = blockIdx.x * INKLING_EPI_BLOCK + threadIdx.x;
    if (column >= width) {
        return;
    }
    const float* source = gate_up + (uint64_t)row * 2 * width;
    const float gate = source[column];
    const float up = source[width + column];
    // softcap(x) = tanh(x / cap) * cap: the cap multiplies back, or the activation would be
    // the tanh itself and the branch would lose its scale.
    const float gated = gate_cap > 0.0f ? inkling_tanh(gate / gate_cap) * gate_cap : gate;
    const float capped_up = up_cap > 0.0f ? inkling_tanh(up / up_cap) * up_cap : up;
    out[(uint64_t)row * width + column] =
        __fmul_rn(__fmul_rn(gated, inkling_sigmoid(gate)), capped_up);
}

// The final softcap and the unpadded slice in one pass.
extern "C" __global__ void __launch_bounds__(INKLING_EPI_BLOCK) inkling_logit_softcap(
    const float* __restrict__ logits,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int padded_vocab,
    int unpadded_vocab,
    float cap) {
    const int row = blockIdx.x;
    if (row >= rows || unpadded_vocab > padded_vocab) {
        return;
    }
    const float* source = logits + (uint64_t)row * padded_vocab;
    __nv_bfloat16* target = out + (uint64_t)row * unpadded_vocab;
    for (int entry = threadIdx.x; entry < unpadded_vocab; entry += INKLING_EPI_BLOCK) {
        const float value = source[entry];
        target[entry] = __float2bfloat16_rn(cap > 0.0f ? cap * inkling_tanh(value / cap) : value);
    }
}

// The mean absolute error of one E2M1 round trip at a candidate scale.
__device__ __forceinline__ float inkling_four_over_six_error(float value, float scale,
                                                             unsigned int lane) {
    if (lane >= 16u) {
        return 0.0f;
    }
    const float decoded = spark_fp4_e2m1_decode(spark_fp4_e2m1_encode(value / scale)) * scale;
    return fabsf(__fadd_rn(value, -decoded));
}

// The f32 projections become the BF16 cache rows the attention body reads.
extern "C" __global__ void __launch_bounds__(INKLING_EPI_BLOCK) inkling_kv_store(
    const float* __restrict__ key,
    const float* __restrict__ value,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    int rows,
    int position,
    int width) {
    const int row = blockIdx.y;
    if (row >= rows || width <= 0) {
        return;
    }
    const uint64_t destination = (uint64_t)(position + row) * width;
    const uint64_t source = (uint64_t)row * width;
    for (int column = blockIdx.x * INKLING_EPI_BLOCK + threadIdx.x; column < width;
         column += gridDim.x * INKLING_EPI_BLOCK) {
        key_cache[destination + column] = __float2bfloat16_rn(key[source + column]);
        value_cache[destination + column] = __float2bfloat16_rn(value[source + column]);
    }
}

// out = residual + delta, elementwise. out may alias residual (the stream is updated in place).
extern "C" __global__ void __launch_bounds__(INKLING_EPI_BLOCK) inkling_residual_add(
    const float* __restrict__ residual,
    const float* __restrict__ delta,
    float* __restrict__ out,
    int rows,
    int width) {
    const int row = blockIdx.y;
    if (row >= rows || width <= 0) {
        return;
    }
    const uint64_t offset = (uint64_t)row * width;
    for (int column = blockIdx.x * INKLING_EPI_BLOCK + threadIdx.x; column < width;
         column += gridDim.x * INKLING_EPI_BLOCK) {
        out[offset + column] =
            __fadd_rn(residual[offset + column], delta[offset + column]);
    }
}

// BF16 -> f32: exact, a bit insertion.
extern "C" __global__ void __launch_bounds__(INKLING_EPI_BLOCK) inkling_widen_f32(
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ dst,
    int count) {
    for (int index = blockIdx.x * INKLING_EPI_BLOCK + threadIdx.x; index < count;
         index += gridDim.x * INKLING_EPI_BLOCK) {
        dst[index] = __bfloat162float(src[index]);
    }
}

// The routed experts' mix: out = sum over the selected slots of scales[slot] * partials[slot].
extern "C" __global__ void __launch_bounds__(INKLING_EPI_BLOCK) inkling_expert_mix(
    float* __restrict__ out,
    const float* __restrict__ partials,
    const float* __restrict__ scales,
    int rows,
    int slots,
    int width) {
    const int row = blockIdx.y;
    if (row >= rows || slots <= 0 || width <= 0) {
        return;
    }
    const float* band = partials + (uint64_t)row * slots * width;
    const float* shares = scales + (uint64_t)row * slots;
    float* target = out + (uint64_t)row * width;
    for (int column = blockIdx.x * INKLING_EPI_BLOCK + threadIdx.x; column < width;
         column += gridDim.x * INKLING_EPI_BLOCK) {
        float value = 0.0f;
        for (int slot = 0; slot < slots; ++slot) {
            value = __fadd_rn(value, __fmul_rn(shares[slot], band[(uint64_t)slot * width + column]));
        }
        target[column] = value;
    }
}

extern "C" __global__ void __launch_bounds__(128) inkling_nvfp4_quantize_4_6(
    const float* __restrict__ x,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ scales,
    int rows,
    int columns) {
    const int row = blockIdx.x;
    if (row >= rows || (columns & 15) != 0) {
        return;
    }
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int warps = blockDim.x >> 5;
    const int blocks = columns >> 4;
    const float* source = x + (uint64_t)row * columns;
    uint8_t* codes = packed + (uint64_t)row * (columns >> 1);
    uint8_t* block_scales = scales + (uint64_t)row * blocks;
    for (int block = (int)warp; block < blocks; block += (int)warps) {
        const float value = lane < 16u ? source[block * 16 + lane] : 0.0f;
        float amax = fabsf(value);
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
        }
        const float base = amax > 0.0f ? amax : 1.0f;
        const uint8_t code4 = spark_fp8_e4m3_encode(base / 4.0f);
        const uint8_t code6 = spark_fp8_e4m3_encode(base / 6.0f);
        const float scale4 = spark_fp8_e4m3_decode(code4);
        const float scale6 = spark_fp8_e4m3_decode(code6);
        float error4 = inkling_four_over_six_error(value, scale4, lane);
        float error6 = inkling_four_over_six_error(value, scale6, lane);
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            error4 += __shfl_xor_sync(0xffffffffu, error4, offset);
            error6 += __shfl_xor_sync(0xffffffffu, error6, offset);
        }
        const bool take_six = error6 < error4;
        uint8_t chosen = take_six ? code6 : code4;
        float scale = take_six ? scale6 : scale4;
        // A block whose amax is below E4M3's smallest subnormal would otherwise divide by a
        // zero scale; such a block quantises to all-zero codes under the unscaled identity.
        if (!(scale > 0.0f)) {
            chosen = spark_fp8_e4m3_encode(1.0f);
            scale = 1.0f;
        }
        const unsigned int code = spark_fp4_e2m1_encode(value / scale);
        const unsigned int neighbour = __shfl_down_sync(0xffffffffu, code, 1u);
        if (lane < 16u && (lane & 1u) == 0u) {
            codes[(block * 16 + (int)lane) >> 1] = (uint8_t)(code | (neighbour << 4));
        }
        if (lane == 0u) {
            block_scales[block] = chosen;
        }
    }
}

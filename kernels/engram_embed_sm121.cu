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

// kernels/engram_embed_sm121.cu
//
// The Engram table row reader and activation quantiser, in the released dtype rule (task-9, audit F4).
//
// WHAT THE RELEASED CODE DOES (inference/model.py:311-320, 345-353):
//
//     values = embed(indices).float() * scales.float()      // f32 dequantise, one scale per 32 columns
//     values = values.to(torch.bfloat16)                    // THE bf16 CAST (model.py:318-319)
//     kv     = wkv(values.flatten(-2))                      // fp8 W8A8, Linear at the default dtype
//
// so the W8A8 GEMM's A operand is the bf16-rounded row, quantised by act_quant(block 32, ue8m0,
// scale_fmt="ue8m0", kernel.py:41-112): amax floored at 1e-4, scale = 2^ceil(log2(amax/448)), one
// E8M0 code per 32 values, then clamp(x/scale, +/-448) to E4M3 RNE.
//
// This kernel emits both tensors the next stage consumes, and it emits the bf16 row as its own
// output on purpose: the cast is the native rule's whole difference from the chain's f32 seam, so
// it must be observable and comparable instead of folded into the quantiser.
//
// ARITHMETIC ORDER (not a tolerance class):
//   dequant: f32 multiply of the E4M3 value by the E8M0 power of two (exact), then one bf16 RNE cast;
//   amax:    over the 32 bf16 values of the block, as f32 (bf16 -> f32 is exact);
//   scale:   the reference act-quant scale code (spark_act_quant_scale_code);
//   code:    fdiv by the decoded power of two (exact), clamp to +/-448, E4M3 RNE saturating.
//
// One block of kEngramEmbedThreads threads. The quantiser's per-block work is 32 values per block,
// so the kernel hands one whole block to one thread rather than reducing across lanes: the amax and
// the codes then cost no barrier and the order is the engine's ascending index order.
//
// TRACE. The signature carries the ring/layer/step words for ABI uniformity; this kernel emits no
// trace event (the protocol's 16 scopes are assigned and none describes an embed row read).

#include "m4_activation_quant_sm121.cuh"

#define SPARK_ENGRAM_EMBED_THREADS 256u

// One dequantised E4M3 value times its E8M0 block scale, as f32 (exact: the scale is a power of two).
__device__ __forceinline__ float spark_engram_dequantise(uint8_t code, uint8_t scale) {
    return __fmul_rn(spark_fp8_e4m3_decode(code), spark_ue8m0_decode(scale));
}

extern "C" __global__ void engram_embed_sm121(
    const uint8_t *__restrict__ row_codes,    // [rows_cols][head_dim] E4M3, checkpoint order
    const uint8_t *__restrict__ row_scales,   // [rows_cols][head_dim / 32] E8M0
    __nv_bfloat16 *__restrict__ fetched,      // [rows_cols * head_dim] the bf16 row (model.py:318)
    uint8_t *__restrict__ act_codes,          // [rows_cols * head_dim] E4M3 activation codes
    uint8_t *__restrict__ act_scales,         // [rows_cols * head_dim / 32] E8M0 activation scales
    const uint32_t rows_cols,                 // 24 for the released model: (max_ngram_size-1)*heads
    const uint32_t head_dim,                  // 256 for the released model
    GpuTraceRing *__restrict__ trace_ring,    // ABI uniformity only; see TRACE above
    const uint32_t layer_id,
    const uint64_t step_seq) {
    (void)trace_ring;
    (void)layer_id;
    (void)step_seq;

    const uint32_t total = rows_cols * head_dim;
    const uint32_t row_blocks = head_dim / SPARK_M4_QUANT_BLOCK;
    for (uint32_t index = threadIdx.x; index < total; index += blockDim.x) {
        const uint32_t row = index / head_dim;
        const uint32_t column = index % head_dim;
        const uint8_t scale = row_scales[row * row_blocks + column / SPARK_M4_QUANT_BLOCK];
        fetched[index] = __float2bfloat16_rn(spark_engram_dequantise(row_codes[index], scale));
    }
    __syncthreads();

    const uint32_t blocks = total / SPARK_M4_QUANT_BLOCK;
    for (uint32_t block = threadIdx.x; block < blocks; block += blockDim.x) {
        const uint32_t base = block * SPARK_M4_QUANT_BLOCK;
        // Load the 32-value block once into registers (it was read from global memory twice).
        float values[SPARK_M4_QUANT_BLOCK];
        float amax = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < SPARK_M4_QUANT_BLOCK; ++i) {
            const float value = __bfloat162float(fetched[base + i]);
            values[i] = value;
            const float magnitude = value < 0.0f ? -value : value;
            amax = magnitude > amax ? magnitude : amax;
        }
        const uint8_t scale_code = spark_act_quant_scale_code(amax);
        act_scales[block] = scale_code;
        const float scale = spark_ue8m0_decode(scale_code);
#pragma unroll
        for (uint32_t i = 0; i < SPARK_M4_QUANT_BLOCK; ++i) {
            float scaled = __fdiv_rn(values[i], scale);
            if (scaled > 448.0f) {
                scaled = 448.0f;
            }
            if (scaled < -448.0f) {
                scaled = -448.0f;
            }
            act_codes[base + i] = spark_fp8_e4m3_encode(scaled);
        }
    }
}

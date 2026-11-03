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

// Qwen3.8-Flash-Next (Qwen4-Exp) token & batched-prefill kernels, TP=1.
//
// Attribution:
//   - Model architecture (Qwen4ExpForConditionalGeneration: hybrid 3:1 Gated DeltaNet linear
//     attention + Qwen Sparse Attention with 4-token block-compressed indexer, 4-stream gated
//     hyper-connections, 512-expert top-10 MoE + sigmoid-gated shared expert, and Layer-2 16-head
//     bigram/trigram Per-Layer Embedding with dilated depthwise conv) originates from
//     Qwen/Qwen3.8-Flash-Next by the Qwen team at Alibaba Cloud and the vLLM reference
//     implementation (https://github.com/vllm-project/vllm, Apache-2.0).
//   - NVFP4 W4A4 block-scaled quantization layout (packed E2M1 nibbles, per-16-column FP8 E4M3
//     block scales, FP32 weight_scale_2, and 128-shard FP8 E4M3 PLE table) originates from
//     nvidia/Qwen3.8-Flash-Next-NVFP4 (NVIDIA TensorRT ModelOpt 0.46).
//   - Kimi Delta Attention's per-channel decay (q4e_kda_decay and q4e_gdn_recur's channel form)
//     is the method of Kimi Team et al., "Kimi Linear: An Expressive, Efficient Attention
//     Architecture" (arXiv:2510.26692), as Z.ai's GLM-5.3-Flash release (MIT License) uses it.
//   - The block-scaled FP8 GEMV (q4e_gemv_tiled_e4m3_block: E4M3 weights with one f32
//     weight_scale_inv a 128 x 128 block, activations quantised per 128 columns) reads the format
//     of Z.ai's GLM-5.3-Flash release (MIT License), the one Hugging Face Transformers'
//     fine-grained FP8 integration reads (Apache-2.0); no source is copied.
//   - The GLM-5.3-Flash section at the end of this file (q4e_mhc_*, q4e_swiglu_limited,
//     q4e_noaux_router, q4e_layer_norm, q4e_glm_pool, q4e_glm_index_scores, q4e_glm_pick and
//     q4e_expert_gemv_e4m3_block) computes Z.ai's GLM-5.3-Flash release (MIT License) as its
//     reference implementation in Hugging Face Transformers (Apache-2.0) does, read for the
//     arithmetic; no source is copied. Its hyper-connection mixers are mHC (Xie et al.,
//     arXiv:2512.24880) and its indexer is DeepSeek Sparse Attention's (DeepSeek-V3.2,
//     arXiv:2512.02556) with GLM-5.3-Flash's k-pool.
//   - SM120/SM121 (DGX Spark GB10) kernel design and local-inference execution patterns draw on
//     prior art from b12x (https://github.com/local-inference-lab/b12x,
//     flashinfer/experimental/b12x, Apache-2.0; Luke Alonso @lukealonso, @original-el8, and
//     contributors) and NVIDIA Flare (https://github.com/NVIDIA/flare-local-inference, Apache-2.0).
//     See the root NOTICE file for full provenance details.
//
// The arithmetic mirrors src/forward.rs, which is the host reference oracle. Each kernel is named
// q4e_* and is loaded by name from the cubin, so a missing symbol is a load error rather than a
// silent wrong launch.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// The E4M3 / E2M1 decoders, from the workspace's shared header (see
// kernels/quant_decode_sm121.cuh).
#include "prefetch_sm121.cuh"
#include "quant_decode_sm121.cuh"
// Every q4e kernel opens with OPT_PDL_ENTRY: the host launches them with programmatic stream
// serialization, so the next kernel's launch overlaps this one's run and stream order holds.
#include "opt_pdl_sm121.cuh"

#define Q4E_BLOCK 256

// Static shared-memory and register-tiling bounds.
//
// The driver launch path in sparkflash-kernels passes a dynamic shared-memory size of zero, so
// every shared buffer is a fixed array sized for the largest geometry this pack can serve, with a
// guard that makes an oversized request a no-op rather than a corruption. The host side checks the
// same bounds and refuses first.
#define Q4E_MAX_LOWRANK 1024
#define Q4E_MAX_INTERMEDIATE 1024
#define Q4E_MAX_SELECTED 2304
#define Q4E_MAX_EXPERTS 512
#define Q4E_MAX_TOPK 64
// The widest matrix row any NVFP4 expert projection stages in shared memory.
#define Q4E_MAX_ROW 2560
// Padding each 32-float group by 1 float (stride 33, coprime to 32) eliminates the 32-way
// shared-memory bank conflict when a warp's 32 lanes read 32-float chunks simultaneously.
#define Q4E_SMEM_STRIDE 33
#define Q4E_MAX_ROW_PADDED ((Q4E_MAX_ROW >> 5) * Q4E_SMEM_STRIDE + 32)
// Maximum tokens processed in one layer-major prefill chunk so dense BF16 weights are read once
// per chunk while keeping all per-token accumulators in registers with zero stack spills.
#define Q4E_MAX_BATCH 8

// The decoders are the shared header's; these aliases keep the kernel bodies readable.
#define q4e_e4m3 spark_fp8_e4m3_decode
#define q4e_e2m1 spark_fp4_e2m1_decode

#ifdef SPARKFLASH_CUDA_SYNTAX_STUB
template <typename T>
__device__ __forceinline__ T q4e_ldg(const T* ptr) {
    return *ptr;
}
#else
template <typename T>
__device__ __forceinline__ T q4e_ldg(const T* ptr) {
    return __ldg(ptr);
}
#endif

__device__ __forceinline__ int q4e_smem_idx(int col) {
    return col + (col >> 5);
}

// The sixteen E2M1 codes in a block-shared table, filled once before a kernel's dots run.
//
// Decoding one nibble through it costs a shift, a mask and a shared-memory load, where the scalar
// decoder needs a shift, a mask, a 32-bit table shift, an int-to-float, a scale and a sign-bit
// insert. The routed experts walk 4.9M packed weights per (layer, slot), so that difference
// dominates their issue-slot budget. Every code is exactly representable, so the table holds the
// scalar decoder's own results.
//
// It is shared memory rather than a warp-shuffle register table on purpose: the row-dot loops start
// each lane at a different chunk offset, so on a row that is not a whole number of 32-chunk strides
// the lanes leave the loop at different counts, and a lane-indexed shuffle would then name lanes
// that are no longer participating.
__device__ __forceinline__ void q4e_e2m1_table_fill(float* table) {
    for (int code = threadIdx.x; code < 16; code += blockDim.x) {
        table[code] = q4e_e2m1((unsigned char)code);
    }
}

// One nibble of a packed word, decoded through the block-shared table.
__device__ __forceinline__ float q4e_e2m1_at(unsigned int nibble, const float* table) {
    return table[nibble & 0x0Fu];
}

// The same table indexed by a whole packed byte: entry `code` holds the low nibble's value in
// `.x` and the high nibble's in `.y`, so one 64-bit shared load decodes two weights. The entry
// width is what matters as much as the count - entry `code` occupies words 2*code and 2*code+1,
// so sixteen distinct codes span all thirty-two banks, where the sixteen-entry table packs every
// distinct nibble into a 64-byte window and the gathers of a warp run at up to eight-way conflict.
__device__ __forceinline__ void q4e_e2m1_table_pair_fill(float2* table) {
    for (int code = threadIdx.x; code < 256; code += blockDim.x) {
        table[code] = make_float2(q4e_e2m1((unsigned char)(code & 15)),
                                  q4e_e2m1((unsigned char)(code >> 4)));
    }
}

__device__ __forceinline__ float2 q4e_e2m1_pair_at(unsigned int word, const float2* table) {
    return table[word & 0xFFu];
}

__device__ __forceinline__ float q4e_silu(float value) {
    return value / (1.0f + __expf(-value));
}

// sparkflash_weights::linalg::sigmoid, which GLM-5.3-Flash's host twin uses everywhere: 1 / (1 +
// e^-x) at x >= 0 and e^x / (1 + e^x) below, with the accurate exponential.
__device__ __forceinline__ float q4e_sigmoid_ref(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    }
    const float e = expf(x);
    return e / (1.0f + e);
}

// One row per warp, one lane per 32 columns: the packed weights are read coalesced.
__device__ __forceinline__ float q4e_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float q4e_dot_pair(unsigned int word, float x0, float x1) {
    const float low = __bfloat162float(__ushort_as_bfloat16((unsigned short)(word & 0xFFFFu)));
    const float high = __bfloat162float(__ushort_as_bfloat16((unsigned short)(word >> 16)));
    return low * x0 + high * x1;
}

__device__ __forceinline__ float q4e_dot_word(unsigned int word, const float* __restrict__ x,
                                              int base) {
    return q4e_dot_pair(word, x[base], x[base + 1]);
}

// One 16-byte chunk of a BF16 row: eight weights against eight activations loaded as two float4s.
__device__ __forceinline__ float q4e_dot_chunk(const uint4 value, const float* __restrict__ x,
                                               int base) {
    const float4 x0 = q4e_ldg(reinterpret_cast<const float4*>(x + base));
    const float4 x1 = q4e_ldg(reinterpret_cast<const float4*>(x + base + 4));
    return q4e_dot_pair(value.x, x0.x, x0.y) + q4e_dot_pair(value.y, x0.z, x0.w) +
           q4e_dot_pair(value.z, x1.x, x1.y) + q4e_dot_pair(value.w, x1.z, x1.w);
}

// One warp's partial dot product of a BF16 row against an f32 vector; reduce with q4e_warp_sum.
__device__ __forceinline__ float q4e_row_dot_bf16(const __nv_bfloat16* __restrict__ row,
                                                  const float* __restrict__ x, int columns,
                                                  int lane) {
    float total = 0.0f;
    if ((columns & 7) == 0 && ((reinterpret_cast<unsigned long long>(row) & 15u) == 0) &&
        ((reinterpret_cast<unsigned long long>(x) & 15u) == 0)) {
        const int chunks = columns >> 3;
        int chunk = lane;
        for (; chunk + 96 < chunks; chunk += 128) {
            const uint4 a = q4e_ldg(reinterpret_cast<const uint4*>(row + chunk * 8));
            const uint4 b = q4e_ldg(reinterpret_cast<const uint4*>(row + (chunk + 32) * 8));
            const uint4 c = q4e_ldg(reinterpret_cast<const uint4*>(row + (chunk + 64) * 8));
            const uint4 d = q4e_ldg(reinterpret_cast<const uint4*>(row + (chunk + 96) * 8));
            total += q4e_dot_chunk(a, x, chunk * 8) + q4e_dot_chunk(b, x, (chunk + 32) * 8) +
                     q4e_dot_chunk(c, x, (chunk + 64) * 8) +
                     q4e_dot_chunk(d, x, (chunk + 96) * 8);
        }
        if (chunk + 32 < chunks) {
            const uint4 a = q4e_ldg(reinterpret_cast<const uint4*>(row + chunk * 8));
            const uint4 b = q4e_ldg(reinterpret_cast<const uint4*>(row + (chunk + 32) * 8));
            total += q4e_dot_chunk(a, x, chunk * 8);
            total += q4e_dot_chunk(b, x, (chunk + 32) * 8);
            chunk += 64;
        }
        for (; chunk < chunks; chunk += 32) {
            const uint4 value = q4e_ldg(reinterpret_cast<const uint4*>(row + chunk * 8));
            total += q4e_dot_chunk(value, x, chunk * 8);
        }
        return total;
    }
#pragma unroll 1
    for (int column = lane; column < columns; column += 32) {
        total += __bfloat162float(row[column]) * x[column];
    }
    return total;
}

// Batched BF16 row dot: reads each weight chunk once and accumulates across batch token rows in
// registers (batch <= Q4E_MAX_BATCH).
//
// Each token's sum is grouped exactly as q4e_row_dot_bf16 groups it - four chunks added together
// before they meet the total, then a pair, then one at a time - so a row of a batch is
// bit-identical to the same row alone. A speculative verify has to compute as the one-token steps
// it stands in for, and a different association moved its logits by ~1e-6, enough to flip a
// near-tie and make greedy output depend on whether a drafter ran.
__device__ __forceinline__ void q4e_row_dot_bf16_batch(
    const __nv_bfloat16* __restrict__ row, const float* __restrict__ x, int columns, int lane,
    int batch, float totals[Q4E_MAX_BATCH]) {
#pragma unroll
    for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
        totals[t] = 0.0f;
    }
    if ((columns & 7) == 0 && ((reinterpret_cast<unsigned long long>(row) & 15u) == 0) &&
        ((reinterpret_cast<unsigned long long>(x) & 15u) == 0)) {
        const int chunks = columns >> 3;
        int chunk = lane;
        for (; chunk + 96 < chunks; chunk += 128) {
            const uint4 a = q4e_ldg(reinterpret_cast<const uint4*>(row + chunk * 8));
            const uint4 b = q4e_ldg(reinterpret_cast<const uint4*>(row + (chunk + 32) * 8));
            const uint4 c = q4e_ldg(reinterpret_cast<const uint4*>(row + (chunk + 64) * 8));
            const uint4 d = q4e_ldg(reinterpret_cast<const uint4*>(row + (chunk + 96) * 8));
#pragma unroll
            for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
                if (t < batch) {
                    const float* xt = x + (unsigned long long)t * columns;
                    totals[t] += q4e_dot_chunk(a, xt, chunk * 8) +
                                 q4e_dot_chunk(b, xt, (chunk + 32) * 8) +
                                 q4e_dot_chunk(c, xt, (chunk + 64) * 8) +
                                 q4e_dot_chunk(d, xt, (chunk + 96) * 8);
                }
            }
        }
        if (chunk + 32 < chunks) {
            const uint4 a = q4e_ldg(reinterpret_cast<const uint4*>(row + chunk * 8));
            const uint4 b = q4e_ldg(reinterpret_cast<const uint4*>(row + (chunk + 32) * 8));
#pragma unroll
            for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
                if (t < batch) {
                    const float* xt = x + (unsigned long long)t * columns;
                    totals[t] += q4e_dot_chunk(a, xt, chunk * 8);
                    totals[t] += q4e_dot_chunk(b, xt, (chunk + 32) * 8);
                }
            }
            chunk += 64;
        }
        for (; chunk < chunks; chunk += 32) {
            const uint4 value = q4e_ldg(reinterpret_cast<const uint4*>(row + chunk * 8));
#pragma unroll
            for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
                if (t < batch) {
                    const float* xt = x + (unsigned long long)t * columns;
                    totals[t] += q4e_dot_chunk(value, xt, chunk * 8);
                }
            }
        }
        return;
    }
#pragma unroll 1
    for (int column = lane; column < columns; column += 32) {
        const float w = __bfloat162float(row[column]);
#pragma unroll
        for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
            if (t < batch) {
                totals[t] += w * x[(unsigned long long)t * columns + column];
            }
        }
    }
}

// Paired BF16 row dot (gate and up sharing the same input vector x).
__device__ __forceinline__ void q4e_row_dot_bf16_pair(
    const __nv_bfloat16* __restrict__ g_row, const __nv_bfloat16* __restrict__ u_row,
    const float* __restrict__ x, int columns, int lane, float* g_out, float* u_out) {
    float g_total = 0.0f;
    float u_total = 0.0f;
    if ((columns & 7) == 0 && ((reinterpret_cast<unsigned long long>(g_row) & 15u) == 0) &&
        ((reinterpret_cast<unsigned long long>(u_row) & 15u) == 0) &&
        ((reinterpret_cast<unsigned long long>(x) & 15u) == 0)) {
        const int chunks = columns >> 3;
        int chunk = lane;
        for (; chunk + 96 < chunks; chunk += 128) {
            const uint4 ga = q4e_ldg(reinterpret_cast<const uint4*>(g_row + chunk * 8));
            const uint4 gb = q4e_ldg(reinterpret_cast<const uint4*>(g_row + (chunk + 32) * 8));
            const uint4 gc = q4e_ldg(reinterpret_cast<const uint4*>(g_row + (chunk + 64) * 8));
            const uint4 gd = q4e_ldg(reinterpret_cast<const uint4*>(g_row + (chunk + 96) * 8));
            const uint4 ua = q4e_ldg(reinterpret_cast<const uint4*>(u_row + chunk * 8));
            const uint4 ub = q4e_ldg(reinterpret_cast<const uint4*>(u_row + (chunk + 32) * 8));
            const uint4 uc = q4e_ldg(reinterpret_cast<const uint4*>(u_row + (chunk + 64) * 8));
            const uint4 ud = q4e_ldg(reinterpret_cast<const uint4*>(u_row + (chunk + 96) * 8));
            g_total += q4e_dot_chunk(ga, x, chunk * 8) + q4e_dot_chunk(gb, x, (chunk + 32) * 8) +
                       q4e_dot_chunk(gc, x, (chunk + 64) * 8) +
                       q4e_dot_chunk(gd, x, (chunk + 96) * 8);
            u_total += q4e_dot_chunk(ua, x, chunk * 8) + q4e_dot_chunk(ub, x, (chunk + 32) * 8) +
                       q4e_dot_chunk(uc, x, (chunk + 64) * 8) +
                       q4e_dot_chunk(ud, x, (chunk + 96) * 8);
        }
        if (chunk + 32 < chunks) {
            const uint4 ga = q4e_ldg(reinterpret_cast<const uint4*>(g_row + chunk * 8));
            const uint4 gb = q4e_ldg(reinterpret_cast<const uint4*>(g_row + (chunk + 32) * 8));
            const uint4 ua = q4e_ldg(reinterpret_cast<const uint4*>(u_row + chunk * 8));
            const uint4 ub = q4e_ldg(reinterpret_cast<const uint4*>(u_row + (chunk + 32) * 8));
            g_total += q4e_dot_chunk(ga, x, chunk * 8);
            g_total += q4e_dot_chunk(gb, x, (chunk + 32) * 8);
            u_total += q4e_dot_chunk(ua, x, chunk * 8);
            u_total += q4e_dot_chunk(ub, x, (chunk + 32) * 8);
            chunk += 64;
        }
        for (; chunk < chunks; chunk += 32) {
            const uint4 gv = q4e_ldg(reinterpret_cast<const uint4*>(g_row + chunk * 8));
            const uint4 uv = q4e_ldg(reinterpret_cast<const uint4*>(u_row + chunk * 8));
            g_total += q4e_dot_chunk(gv, x, chunk * 8);
            u_total += q4e_dot_chunk(uv, x, chunk * 8);
        }
        *g_out = g_total;
        *u_out = u_total;
        return;
    }
    *g_out = q4e_row_dot_bf16(g_row, x, columns, lane);
    *u_out = q4e_row_dot_bf16(u_row, x, columns, lane);
}

// Keep the higher value, breaking ties toward the lower expert index.
__device__ __forceinline__ void q4e_take_better(float other_value, int other_index, float* value,
                                                int* index) {
    if (other_value > *value || (other_value == *value && other_index < *index)) {
        *value = other_value;
        *index = other_index;
    }
}

// Block argmax over (value, index) pairs, with the same tie-break as the host sort.
__device__ __forceinline__ void q4e_block_best(float* value, int* index, float* shared_value,
                                               int* shared_index) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = (blockDim.x + 31) >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        q4e_take_better(__shfl_down_sync(0xFFFFFFFFu, *value, offset),
                        __shfl_down_sync(0xFFFFFFFFu, *index, offset), value, index);
    }
    if (lane == 0) {
        shared_value[warp] = *value;
        shared_index[warp] = *index;
    }
    __syncthreads();
    if (warp == 0) {
        float best_value = lane < warps ? shared_value[lane] : -INFINITY;
        int best_index = lane < warps ? shared_index[lane] : 0x7FFFFFFF;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            q4e_take_better(__shfl_down_sync(0xFFFFFFFFu, best_value, offset),
                            __shfl_down_sync(0xFFFFFFFFu, best_index, offset), &best_value,
                            &best_index);
        }
        if (lane == 0) {
            shared_value[0] = best_value;
            shared_index[0] = best_index;
        }
    }
    __syncthreads();
    *value = shared_value[0];
    *index = shared_index[0];
}

// Four packed bytes (one 32-bit word): eight E2M1 weights in one 16-column scale group.
__device__ __forceinline__ float q4e_dot_packed32(unsigned int word, const float* lut,
                                                   const float* x_chunk) {
    return q4e_e2m1_at(word, lut) * x_chunk[0] +
           q4e_e2m1_at(word >> 4, lut) * x_chunk[1] +
           q4e_e2m1_at(word >> 8, lut) * x_chunk[2] +
           q4e_e2m1_at(word >> 12, lut) * x_chunk[3] +
           q4e_e2m1_at(word >> 16, lut) * x_chunk[4] +
           q4e_e2m1_at(word >> 20, lut) * x_chunk[5] +
           q4e_e2m1_at(word >> 24, lut) * x_chunk[6] +
           q4e_e2m1_at(word >> 28, lut) * x_chunk[7];
}

__device__ __forceinline__ void q4e_dot_packed32_pair(unsigned int g_word, unsigned int u_word,
                                                      const float* lut, const float* x_chunk,
                                                      float* g_out, float* u_out) {
    const float x0 = x_chunk[0], x1 = x_chunk[1], x2 = x_chunk[2], x3 = x_chunk[3];
    const float x4 = x_chunk[4], x5 = x_chunk[5], x6 = x_chunk[6], x7 = x_chunk[7];
    *g_out = q4e_e2m1_at(g_word, lut) * x0 +
             q4e_e2m1_at(g_word >> 4, lut) * x1 +
             q4e_e2m1_at(g_word >> 8, lut) * x2 +
             q4e_e2m1_at(g_word >> 12, lut) * x3 +
             q4e_e2m1_at(g_word >> 16, lut) * x4 +
             q4e_e2m1_at(g_word >> 20, lut) * x5 +
             q4e_e2m1_at(g_word >> 24, lut) * x6 +
             q4e_e2m1_at(g_word >> 28, lut) * x7;
    *u_out = q4e_e2m1_at(u_word, lut) * x0 +
             q4e_e2m1_at(u_word >> 4, lut) * x1 +
             q4e_e2m1_at(u_word >> 8, lut) * x2 +
             q4e_e2m1_at(u_word >> 12, lut) * x3 +
             q4e_e2m1_at(u_word >> 16, lut) * x4 +
             q4e_e2m1_at(u_word >> 20, lut) * x5 +
             q4e_e2m1_at(u_word >> 24, lut) * x6 +
             q4e_e2m1_at(u_word >> 28, lut) * x7;
}

// Sixteen packed bytes (one uint4): 32 weights across two 16-column scale groups.
// x_chunk points to the 32 contiguous floats for chunk in padded shared memory
// (source + chunk * Q4E_SMEM_STRIDE), so all 32 lanes hit 32 distinct banks.
__device__ __forceinline__ float q4e_dot_packed_chunk(const uint4 packed, const float* lut,
                                                      const unsigned char* scale_row, float scale2,
                                                      const float* x_chunk, int group) {
    const float low = q4e_e4m3(q4e_ldg(&scale_row[group])) * scale2;
    const float high = q4e_e4m3(q4e_ldg(&scale_row[group + 1])) * scale2;
    return low * (q4e_dot_packed32(packed.x, lut, x_chunk) +
                  q4e_dot_packed32(packed.y, lut, x_chunk + 8)) +
           high * (q4e_dot_packed32(packed.z, lut, x_chunk + 16) +
                   q4e_dot_packed32(packed.w, lut, x_chunk + 24));
}

__device__ __forceinline__ void q4e_dot_packed_chunk_pair(
    const uint4 g_packed, const unsigned char* g_scale_row, float g_scale2,
    const uint4 u_packed, const unsigned char* u_scale_row, float u_scale2, const float* lut,
    const float* x_chunk, int group, float* g_total, float* u_total) {
    const float g_low = q4e_e4m3(q4e_ldg(&g_scale_row[group])) * g_scale2;
    const float g_high = q4e_e4m3(q4e_ldg(&g_scale_row[group + 1])) * g_scale2;
    const float u_low = q4e_e4m3(q4e_ldg(&u_scale_row[group])) * u_scale2;
    const float u_high = q4e_e4m3(q4e_ldg(&u_scale_row[group + 1])) * u_scale2;
    float g0, u0, g1, u1, g2, u2, g3, u3;
    q4e_dot_packed32_pair(g_packed.x, u_packed.x, lut, x_chunk, &g0, &u0);
    q4e_dot_packed32_pair(g_packed.y, u_packed.y, lut, x_chunk + 8, &g1, &u1);
    q4e_dot_packed32_pair(g_packed.z, u_packed.z, lut, x_chunk + 16, &g2, &u2);
    q4e_dot_packed32_pair(g_packed.w, u_packed.w, lut, x_chunk + 24, &g3, &u3);
    *g_total += g_low * (g0 + g1) + g_high * (g2 + g3);
    *u_total += u_low * (u0 + u1) + u_high * (u2 + u3);
}

// One warp's partial dot product of a row of an NVFP4 matrix against padded shared memory x.
__device__ __forceinline__ float q4e_nvfp4_row_dot(const unsigned char* packed_row,
                                                   const unsigned char* scale_row, float scale2,
                                                   const float* lut, const float* x, int columns,
                                                   int lane) {
    float total = 0.0f;
    if ((columns & 31) == 0 && ((reinterpret_cast<unsigned long long>(packed_row) & 15u) == 0)) {
        const int chunks = columns >> 5;
        int chunk = lane;
        for (; chunk + 32 < chunks; chunk += 64) {
            const uint4 first = q4e_ldg(reinterpret_cast<const uint4*>(packed_row + chunk * 16));
            const uint4 second =
                q4e_ldg(reinterpret_cast<const uint4*>(packed_row + (chunk + 32) * 16));
            total += q4e_dot_packed_chunk(first, lut, scale_row, scale2,
                                          x + chunk * Q4E_SMEM_STRIDE, chunk * 2);
            total += q4e_dot_packed_chunk(second, lut, scale_row, scale2,
                                          x + (chunk + 32) * Q4E_SMEM_STRIDE, (chunk + 32) * 2);
        }
        for (; chunk < chunks; chunk += 32) {
            const uint4 packed = q4e_ldg(reinterpret_cast<const uint4*>(packed_row + chunk * 16));
            total += q4e_dot_packed_chunk(packed, lut, scale_row, scale2,
                                          x + chunk * Q4E_SMEM_STRIDE, chunk * 2);
        }
        return total;
    }
    int packed = (columns + 1) >> 1;
#pragma unroll 1
    for (int index = lane; index < packed; index += 32) {
        const unsigned char byte = q4e_ldg(&packed_row[index]);
        const float scale = q4e_e4m3(q4e_ldg(&scale_row[index >> 3])) * scale2;
        const int column = index << 1;
        const float pair = (column + 1 < columns) ? x[q4e_smem_idx(column + 1)] : 0.0f;
        total += (q4e_e2m1(byte & 0x0F) * x[q4e_smem_idx(column)] + q4e_e2m1(byte >> 4) * pair) *
                 scale;
    }
    return total;
}

// Simultaneous gate and up NVFP4 row dot products sharing conflict-free x in shared memory.
__device__ __forceinline__ void q4e_nvfp4_row_dot_pair(
    const unsigned char* g_row, const unsigned char* g_scale, float g_scale2,
    const unsigned char* u_row, const unsigned char* u_scale, float u_scale2, const float* lut,
    const float* x, int columns, int lane, float* g_out, float* u_out) {
    if ((columns & 31) == 0 && ((reinterpret_cast<unsigned long long>(g_row) & 15u) == 0) &&
        ((reinterpret_cast<unsigned long long>(u_row) & 15u) == 0)) {
        float g_total = 0.0f;
        float u_total = 0.0f;
        const int chunks = columns >> 5;
        int chunk = lane;
        for (; chunk + 32 < chunks; chunk += 64) {
            const uint4 g0 = q4e_ldg(reinterpret_cast<const uint4*>(g_row + chunk * 16));
            const uint4 g1 = q4e_ldg(reinterpret_cast<const uint4*>(g_row + (chunk + 32) * 16));
            const uint4 u0 = q4e_ldg(reinterpret_cast<const uint4*>(u_row + chunk * 16));
            const uint4 u1 = q4e_ldg(reinterpret_cast<const uint4*>(u_row + (chunk + 32) * 16));
            q4e_dot_packed_chunk_pair(g0, g_scale, g_scale2, u0, u_scale, u_scale2, lut,
                                      x + chunk * Q4E_SMEM_STRIDE, chunk * 2, &g_total, &u_total);
            q4e_dot_packed_chunk_pair(g1, g_scale, g_scale2, u1, u_scale, u_scale2, lut,
                                      x + (chunk + 32) * Q4E_SMEM_STRIDE, (chunk + 32) * 2,
                                      &g_total, &u_total);
        }
        for (; chunk < chunks; chunk += 32) {
            const uint4 g0 = q4e_ldg(reinterpret_cast<const uint4*>(g_row + chunk * 16));
            const uint4 u0 = q4e_ldg(reinterpret_cast<const uint4*>(u_row + chunk * 16));
            q4e_dot_packed_chunk_pair(g0, g_scale, g_scale2, u0, u_scale, u_scale2, lut,
                                      x + chunk * Q4E_SMEM_STRIDE, chunk * 2, &g_total, &u_total);
        }
        *g_out = g_total;
        *u_out = u_total;
        return;
    }
    *g_out = q4e_nvfp4_row_dot(g_row, g_scale, g_scale2, lut, x, columns, lane);
    *u_out = q4e_nvfp4_row_dot(u_row, u_scale, u_scale2, lut, x, columns, lane);
}

// How many of a run's members a leader block stages at once. The decode - a table load and a
// nibble extract per weight - is half the dot product's instructions, and every member of a run
// reuses the same weight row, so decoding once per group of members instead of once per member
// measured a third of the routed-expert stage. Four rows of 2672 floats is 43 KiB of staging,
// which keeps the block inside the 48 KiB of static shared memory every launch may assume.
#define Q4E_MEMBER_GROUP 2

// Rows one warp owns. Every block that partitions an expert's rows stages the run's activations
// itself, so the staging traffic divides by the rows a block covers: at one row per warp a run's
// members were copied into shared memory once per sixteen-row panel, which came to as many bytes
// as the expert's weights. Two rows also put two weight-row streams in flight per warp.
#define Q4E_ROWS_PER_WARP 2

// One 32-column chunk of a gate/up row pair, decoded once and multiplied against the staged rows
// of up to Q4E_MEMBER_GROUP run members.
__device__ __forceinline__ void q4e_dot_packed_chunk_pair_group(
    const uint4 g_packed, const uint4 u_packed, float g_low, float g_high, float u_low,
    float u_high, const float2* lut, const float* base, int count, float* g_total,
    float* u_total) {
    const unsigned int g_words[4] = {g_packed.x, g_packed.y, g_packed.z, g_packed.w};
    const unsigned int u_words[4] = {u_packed.x, u_packed.y, u_packed.z, u_packed.w};
    float g_lo[Q4E_MEMBER_GROUP] = {};
    float g_hi[Q4E_MEMBER_GROUP] = {};
    float u_lo[Q4E_MEMBER_GROUP] = {};
    float u_hi[Q4E_MEMBER_GROUP] = {};
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        float gate[8];
        float up[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 g_pair = q4e_e2m1_pair_at(g_words[word] >> (8 * i), lut);
            const float2 u_pair = q4e_e2m1_pair_at(u_words[word] >> (8 * i), lut);
            gate[2 * i] = g_pair.x;
            gate[2 * i + 1] = g_pair.y;
            up[2 * i] = u_pair.x;
            up[2 * i + 1] = u_pair.y;
        }
#pragma unroll
        for (int m = 0; m < Q4E_MEMBER_GROUP; ++m) {
            if (m < count) {
                const float* x_chunk = base + m * Q4E_MAX_ROW_PADDED + word * 8;
                float g = 0.0f;
                float u = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    g += gate[i] * x_chunk[i];
                    u += up[i] * x_chunk[i];
                }
                if (word < 2) {
                    g_lo[m] += g;
                    u_lo[m] += u;
                } else {
                    g_hi[m] += g;
                    u_hi[m] += u;
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < Q4E_MEMBER_GROUP; ++m) {
        if (m < count) {
            g_total[m] += g_low * g_lo[m] + g_high * g_hi[m];
            u_total[m] += u_low * u_lo[m] + u_high * u_hi[m];
        }
    }
}

// The same for a single matrix, as the down pass has.
__device__ __forceinline__ void q4e_dot_packed_chunk_group(const uint4 packed, float low,
                                                           float high, const float2* lut,
                                                           const float* base, int count,
                                                           float* out) {
    const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
    float lo[Q4E_MEMBER_GROUP] = {};
    float hi[Q4E_MEMBER_GROUP] = {};
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        float weight[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 pair = q4e_e2m1_pair_at(words[word] >> (8 * i), lut);
            weight[2 * i] = pair.x;
            weight[2 * i + 1] = pair.y;
        }
#pragma unroll
        for (int m = 0; m < Q4E_MEMBER_GROUP; ++m) {
            if (m < count) {
                const float* x_chunk = base + m * Q4E_MAX_ROW_PADDED + word * 8;
                float dot = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    dot += weight[i] * x_chunk[i];
                }
                if (word < 2) {
                    lo[m] += dot;
                } else {
                    hi[m] += dot;
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < Q4E_MEMBER_GROUP; ++m) {
        if (m < count) {
            out[m] += low * lo[m] + high * hi[m];
        }
    }
}

// One warp's partial dot products of a gate/up row pair against the staged rows of `count` run
// members, `base` being the member-major staging.
__device__ __forceinline__ void q4e_nvfp4_row_dot_pair_group(
    const unsigned char* g_row, const unsigned char* g_scale, float g_scale2,
    const unsigned char* u_row, const unsigned char* u_scale, float u_scale2, const float* lut,
    const float2* lut_pair, const float* base, int columns, int lane, int count, float* g_out,
    float* u_out) {
#pragma unroll
    for (int m = 0; m < Q4E_MEMBER_GROUP; ++m) {
        g_out[m] = 0.0f;
        u_out[m] = 0.0f;
    }
    if ((columns & 31) == 0 && ((reinterpret_cast<unsigned long long>(g_row) & 15u) == 0) &&
        ((reinterpret_cast<unsigned long long>(u_row) & 15u) == 0)) {
        const int chunks = columns >> 5;
        for (int chunk = lane; chunk < chunks; chunk += 32) {
            const uint4 g_packed = q4e_ldg(reinterpret_cast<const uint4*>(g_row + chunk * 16));
            const uint4 u_packed = q4e_ldg(reinterpret_cast<const uint4*>(u_row + chunk * 16));
            const int group = chunk * 2;
            const float g_low = q4e_e4m3(q4e_ldg(&g_scale[group])) * g_scale2;
            const float g_high = q4e_e4m3(q4e_ldg(&g_scale[group + 1])) * g_scale2;
            const float u_low = q4e_e4m3(q4e_ldg(&u_scale[group])) * u_scale2;
            const float u_high = q4e_e4m3(q4e_ldg(&u_scale[group + 1])) * u_scale2;
            q4e_dot_packed_chunk_pair_group(g_packed, u_packed, g_low, g_high, u_low, u_high,
                                            lut_pair, base + chunk * Q4E_SMEM_STRIDE, count, g_out,
                                            u_out);
        }
        return;
    }
    for (int m = 0; m < count; ++m) {
        g_out[m] = q4e_nvfp4_row_dot(g_row, g_scale, g_scale2, lut,
                                      base + m * Q4E_MAX_ROW_PADDED, columns, lane);
        u_out[m] = q4e_nvfp4_row_dot(u_row, u_scale, u_scale2, lut,
                                      base + m * Q4E_MAX_ROW_PADDED, columns, lane);
    }
}

// And for the down pass's single matrix.
__device__ __forceinline__ void q4e_nvfp4_row_dot_group(const unsigned char* packed_row,
                                                        const unsigned char* scale_row,
                                                        float scale2, const float* lut,
                                                        const float2* lut_pair, const float* base,
                                                        int columns, int lane, int count,
                                                        float* out) {
#pragma unroll
    for (int m = 0; m < Q4E_MEMBER_GROUP; ++m) {
        out[m] = 0.0f;
    }
    if ((columns & 31) == 0 && ((reinterpret_cast<unsigned long long>(packed_row) & 15u) == 0)) {
        const int chunks = columns >> 5;
        for (int chunk = lane; chunk < chunks; chunk += 32) {
            const uint4 packed = q4e_ldg(reinterpret_cast<const uint4*>(packed_row + chunk * 16));
            const int group = chunk * 2;
            const float low = q4e_e4m3(q4e_ldg(&scale_row[group])) * scale2;
            const float high = q4e_e4m3(q4e_ldg(&scale_row[group + 1])) * scale2;
            q4e_dot_packed_chunk_group(packed, low, high, lut_pair,
                                       base + chunk * Q4E_SMEM_STRIDE, count, out);
        }
        return;
    }
    for (int m = 0; m < count; ++m) {
        out[m] = q4e_nvfp4_row_dot(packed_row, scale_row, scale2, lut,
                                    base + m * Q4E_MAX_ROW_PADDED, columns, lane);
    }
}

// Rows the down pass's warps own: an expert's down row is a fifth of a gate/up pair's, so a warp
// holds twice the rows to keep as many weight bytes in flight.
#define Q4E_DOWN_ROWS_PER_WARP 4

// The warp's first `valid` (at most ROWS) of ROWS consecutive down rows at once, rows `packed`
// bytes and their scales `groups` bytes apart: each chunk's loads for every row are issued before
// any of them is decoded, so a lane has ROWS loads in flight instead of one. Each row accumulates
// its chunks in the one-row form's order, so its partials are q4e_nvfp4_row_dot_group's. (A
// gate/up pair holds twice the loads per row and spills at ROWS 2 under four blocks an SM.)
template <int ROWS>
__device__ __forceinline__ void q4e_nvfp4_rows_dot_group(
    const unsigned char* row, const unsigned char* scale_row, float scale2, int packed, int groups,
    int valid, const float* lut, const float2* lut_pair, const float* base, int columns, int lane,
    int count, float (*out)[Q4E_MEMBER_GROUP]) {
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int m = 0; m < Q4E_MEMBER_GROUP; ++m) {
            out[r][m] = 0.0f;
        }
    }
    if ((columns & 31) != 0 || (packed & 15) != 0 ||
        ((reinterpret_cast<unsigned long long>(row) & 15u) != 0)) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r < valid) {
                q4e_nvfp4_row_dot_group(row + r * packed, scale_row + r * groups, scale2, lut,
                                        lut_pair, base, columns, lane, count, out[r]);
            }
        }
        return;
    }
    const int chunks = columns >> 5;
    for (int chunk = lane; chunk < chunks; chunk += 32) {
        uint4 weights[ROWS];
        unsigned short codes[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r < valid) {
                const int scale = r * groups + chunk * 2;
                weights[r] = q4e_ldg(reinterpret_cast<const uint4*>(row + r * packed + chunk * 16));
                codes[r] = q4e_ldg(&scale_row[scale]) | (q4e_ldg(&scale_row[scale + 1]) << 8);
            }
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r < valid) {
                q4e_dot_packed_chunk_group(weights[r], q4e_e4m3(codes[r] & 0xFF) * scale2,
                                           q4e_e4m3(codes[r] >> 8) * scale2, lut_pair,
                                           base + chunk * Q4E_SMEM_STRIDE, count, out[r]);
            }
        }
    }
}

// Block reduction over shared memory; scratch must hold blockDim.x floats.
//
// Every thread has read the result before any returns, so a caller may reduce into the same
// scratch again straight away. Without the closing barrier a fast thread 0 would store its next
// value over scratch[0] while a slower warp had yet to read this one: q4e_attn's max then sum
// handed some warps a partial sum as the maximum, which skewed their keys' softmax weights by a
// timing-dependent factor, and greedy output differed from run to run.
__device__ float q4e_block_sum(float value, float* scratch) {
    int thread = threadIdx.x;
    scratch[thread] = value;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (thread < stride) {
            scratch[thread] += scratch[thread + stride];
        }
        __syncthreads();
    }
    const float total = scratch[0];
    __syncthreads();
    return total;
}

// Block maximum over shared memory; scratch must hold blockDim.x floats. Reusable back to back,
// as q4e_block_sum.
__device__ float q4e_block_max(float value, float* scratch) {
    int thread = threadIdx.x;
    scratch[thread] = value;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (thread < stride) {
            scratch[thread] = fmaxf(scratch[thread], scratch[thread + stride]);
        }
        __syncthreads();
    }
    const float maximum = scratch[0];
    __syncthreads();
    return maximum;
}

// Widen one token's BF16 embedding row and replicate it across width = hc * hidden. With a token
// array, grid row y widens tokens[y] into out + y * width instead, so a token the device picked
// is embedded without the host reading it.
extern "C" __global__ void q4e_embed(const __nv_bfloat16* __restrict__ table,
                                     float* __restrict__ out, int token, int hidden, int width,
                                     const int* __restrict__ tokens) {
    OPT_PDL_ENTRY();
    if (tokens != nullptr) {
        token = tokens[blockIdx.y];
        out += (unsigned long long)blockIdx.y * width;
    }
    const __nv_bfloat16* row = table + (unsigned long long)token * hidden;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < width;
         index += blockDim.x * gridDim.x) {
        out[index] = __bfloat162float(row[index % hidden]);
    }
}

// Row blockIdx.x's greatest logit's index into tokens[blockIdx.x], over logits a tensor-parallel
// gather leaves as [world][rows][local] (index rank * local + j). It is the host argmax's answer:
// the lower index on a tie, a NaN never greater, and index 0 when the row's first logit is a NaN.
extern "C" __global__ void q4e_argmax_slices(const float* __restrict__ logits,
                                             int* __restrict__ tokens, int world, int rows,
                                             int local) {
    OPT_PDL_ENTRY();
    __shared__ float shared_value[32];
    __shared__ int shared_index[32];
    const int row = blockIdx.x;
    float value = -INFINITY;
    int index = 0x7FFFFFFF;
    for (int rank = 0; rank < world; ++rank) {
        const float* slice = logits + ((unsigned long long)rank * rows + row) * local;
#pragma unroll 4
        for (int j = threadIdx.x; j < local; j += blockDim.x) {
            q4e_take_better(slice[j], rank * local + j, &value, &index);
        }
    }
    q4e_block_best(&value, &index, shared_value, shared_index);
    if (threadIdx.x == 0) {
        tokens[row] = isnan(logits[(unsigned long long)row * local]) ? 0 : index;
    }
}

// y = W x, with W in BF16 [rows, columns] and x/y in f32 ([batch, columns] -> [batch, rows]).
extern "C" __global__ void q4e_gemv_bf16(const __nv_bfloat16* __restrict__ w,
                                         const float* __restrict__ x, float* __restrict__ y,
                                         int rows, int columns, int batch) {
    OPT_PDL_ENTRY();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int stride = gridDim.x * warps;
    if (batch <= 1) {
        for (int row = blockIdx.x * warps + warp; row < rows; row += stride) {
            float total =
                q4e_row_dot_bf16(w + (unsigned long long)row * columns, x, columns, lane);
            total = q4e_warp_sum(total);
            if (lane == 0) {
                y[row] = total;
            }
        }
        return;
    }
    if (batch > Q4E_MAX_BATCH) {
        return;
    }
    for (int row = blockIdx.x * warps + warp; row < rows; row += stride) {
        float totals[Q4E_MAX_BATCH];
        q4e_row_dot_bf16_batch(w + (unsigned long long)row * columns, x, columns, lane, batch,
                               totals);
#pragma unroll
        for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
            if (t < batch) {
                float sum = q4e_warp_sum(totals[t]);
                if (lane == 0) {
                    y[(unsigned long long)t * rows + row] = sum;
                }
            }
        }
    }
}

// Fused four-output BF16 projection sharing the same input x ([batch, columns]):
//   - Linear layers: (in_qkv, in_z, in_b, in_a)
//   - Sparse layers: (q_proj, k_proj, v_proj, index_qk)
extern "C" __global__ void q4e_gemv4_bf16(
    const __nv_bfloat16* __restrict__ w0, const __nv_bfloat16* __restrict__ w1,
    const __nv_bfloat16* __restrict__ w2, const __nv_bfloat16* __restrict__ w3,
    const float* __restrict__ x, float* __restrict__ y0, float* __restrict__ y1,
    float* __restrict__ y2, float* __restrict__ y3, int r0, int r1, int r2, int r3, int columns,
    int batch) {
    OPT_PDL_ENTRY();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int stride = gridDim.x * warps;
    const int r01 = r0 + r1;
    const int r012 = r01 + r2;
    const int total_rows = r012 + r3;
    if (batch <= 1) {
        for (int global_row = blockIdx.x * warps + warp; global_row < total_rows;
             global_row += stride) {
            const __nv_bfloat16* w_row;
            float* out;
            int local_row;
            if (global_row < r0) {
                local_row = global_row;
                w_row = w0 + (unsigned long long)local_row * columns;
                out = y0;
            } else if (global_row < r01) {
                local_row = global_row - r0;
                w_row = w1 + (unsigned long long)local_row * columns;
                out = y1;
            } else if (global_row < r012) {
                local_row = global_row - r01;
                w_row = w2 + (unsigned long long)local_row * columns;
                out = y2;
            } else {
                local_row = global_row - r012;
                w_row = w3 + (unsigned long long)local_row * columns;
                out = y3;
            }
            float total = q4e_row_dot_bf16(w_row, x, columns, lane);
            total = q4e_warp_sum(total);
            if (lane == 0) {
                out[local_row] = total;
            }
        }
        return;
    }
    if (batch > Q4E_MAX_BATCH) {
        return;
    }
    for (int global_row = blockIdx.x * warps + warp; global_row < total_rows;
         global_row += stride) {
        const __nv_bfloat16* w_row;
        float* out;
        int local_row;
        int out_rows;
        if (global_row < r0) {
            local_row = global_row;
            out_rows = r0;
            w_row = w0 + (unsigned long long)local_row * columns;
            out = y0;
        } else if (global_row < r01) {
            local_row = global_row - r0;
            out_rows = r1;
            w_row = w1 + (unsigned long long)local_row * columns;
            out = y1;
        } else if (global_row < r012) {
            local_row = global_row - r01;
            out_rows = r2;
            w_row = w2 + (unsigned long long)local_row * columns;
            out = y2;
        } else {
            local_row = global_row - r012;
            out_rows = r3;
            w_row = w3 + (unsigned long long)local_row * columns;
            out = y3;
        }
        float totals[Q4E_MAX_BATCH];
        q4e_row_dot_bf16_batch(w_row, x, columns, lane, batch, totals);
#pragma unroll
        for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
            if (t < batch) {
                float sum = q4e_warp_sum(totals[t]);
                if (lane == 0) {
                    out[(unsigned long long)t * out_rows + local_row] = sum;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Tiled decode GEMV: four weight rows a warp, the activations staged in shared memory.
// ---------------------------------------------------------------------------------------------
//
// The one-warp-a-row GEMV above reads every token's whole activation row again for each output
// row. At one token that row sits in L1; at a verify's eight it is 80 KiB of a hyper stream
// (320 KiB in all) and comes from L2 once per output, so the hyper passes and the 6,144-wide
// output projections fall to half the weight bandwidth past four tokens.
//
// Here a block owns a tile of 32 weight rows, four for each warp. The inner dimension goes in
// rounds of 256 columns: lane l reads the 16 bytes at column 8l of each of its warp's four rows
// (512 contiguous bytes of a row per load, as the one-row GEMV reads them), and the round's
// activations for every token are staged in shared memory once for the block, so a token's row
// crosses L2 once per block and each staged value serves four rows. A matrix with fewer tiles
// than the GPU wants blocks splits its rounds across gridDim.y blocks, and the tile's last block
// to finish adds the splits.
//
// The same body reads the drafter's E4M3 matrices: sixteen one-byte weights to a lane's 16-byte
// chunk, so a round is 512 columns, and the row's scale applied once to the finished sum.
//
// It also reads GLM-5.3-Flash's FP8 projections, E4M3 with one scale a 128 x 128 block, whose
// activations the checkpoint quantises per 128 columns. A tile's 32 rows lie in one band of
// blocks, so as a warp stages its run of 128 columns it quantises them, scales them back and
// multiplies them by the band's block scale; the weights then fold as the drafter's do.
//
// Every token's sum is one chain whatever the batch: per lane, an fmaf a weight in ascending
// column order over the block's rounds; the lanes folded by a fixed xor butterfly; the split
// partials added in split order by whichever block of the tile arrives last. The split count is
// the host's function of the shape alone (tiled_gemv_splits), so a verify row is bit-identical
// to the one-token step.
#define Q4E_TG_WARPS 8
#define Q4E_TG_THREADS (Q4E_TG_WARPS * 32)
#define Q4E_TG_ROWS 4
#define Q4E_TG_TILE (Q4E_TG_WARPS * Q4E_TG_ROWS)

// acc + the eight BF16 weights of `word4` against lo (columns 0..4) and hi (4..8).
__device__ __forceinline__ float q4e_tg_fma8(const uint4 word4, const float4 lo, const float4 hi,
                                             float acc) {
    acc = fmaf(__uint_as_float(word4.x << 16), lo.x, acc);
    acc = fmaf(__uint_as_float(word4.x & 0xFFFF0000u), lo.y, acc);
    acc = fmaf(__uint_as_float(word4.y << 16), lo.z, acc);
    acc = fmaf(__uint_as_float(word4.y & 0xFFFF0000u), lo.w, acc);
    acc = fmaf(__uint_as_float(word4.z << 16), hi.x, acc);
    acc = fmaf(__uint_as_float(word4.z & 0xFFFF0000u), hi.y, acc);
    acc = fmaf(__uint_as_float(word4.w << 16), hi.z, acc);
    acc = fmaf(__uint_as_float(word4.w & 0xFFFF0000u), hi.w, acc);
    return acc;
}

// Two E4M3 codes (the low byte first) widened through f16, which holds every code exactly.
__device__ __forceinline__ float2 q4e_e4m3x2(unsigned int pair) {
    const __half2_raw raw =
        __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(pair & 0xFFFFu), __NV_E4M3);
    return __half22float2(__half2(raw));
}

// The weight formats the tiled body reads. A lane takes one 16-byte chunk of each of its rows a
// round, `kColumns` weights, against `x`: the chunk's staged activations, four to a float4.
// `kBlockScaled` formats stage their activations through q4e_tg_fp8_activation.
struct Q4eTgBf16 {
    using Weight = __nv_bfloat16;
    static constexpr int kColumns = 8;
    static constexpr bool kBlockScaled = false;
    __device__ static float fma(const uint4 word4, const float4* x, float acc) {
        return q4e_tg_fma8(word4, x[0], x[1], acc);
    }
};

struct Q4eTgE4m3 {
    using Weight = unsigned char;
    static constexpr int kColumns = 16;
    static constexpr bool kBlockScaled = false;
    __device__ static float fma(const uint4 word4, const float4* x, float acc) {
        const unsigned int words[4] = {word4.x, word4.y, word4.z, word4.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 low = q4e_e4m3x2(words[i]);
            const float2 high = q4e_e4m3x2(words[i] >> 16);
            acc = fmaf(low.x, x[i].x, acc);
            acc = fmaf(low.y, x[i].y, acc);
            acc = fmaf(high.x, x[i].z, acc);
            acc = fmaf(high.y, x[i].w, acc);
        }
        return acc;
    }
};

// E4M3 with one f32 scale a 128 x 128 block (GLM-5.3-Flash's FP8 projections).
struct Q4eTgE4m3Block : Q4eTgE4m3 {
    static constexpr bool kBlockScaled = true;
};

#define Q4E_FP8_BLOCK 128
static_assert(Q4E_FP8_BLOCK % Q4E_TG_TILE == 0, "a tile's rows lie in one band of blocks");

// GLM-5.3-Flash's dynamic activation quantisation of one run of 128 columns, a warp's four values
// a lane, as its host forward does it (quant::quantize_activation_fp8_tiles, then the dequant):
// the run's largest magnitude over 448 is its scale, each value over the scale rounds to the
// nearest E4M3 code, and the code times the scale is the value the matmul takes; a run whose
// scale is zero is zero. `block` is the weights' scale for these columns, applied after.
__device__ __forceinline__ float4 q4e_tg_fp8_activation(const float4 value, const float block) {
    float largest =
        fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)), fmaxf(fabsf(value.z), fabsf(value.w)));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        largest = fmaxf(largest, __shfl_xor_sync(0xFFFFFFFFu, largest, offset));
    }
    const float scale = largest / 448.0f;
    if (scale == 0.0f) {
        return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const unsigned int codes[4] = {
        __nv_cvt_float_to_fp8(value.x / scale, __NV_SATFINITE, __NV_E4M3),
        __nv_cvt_float_to_fp8(value.y / scale, __NV_SATFINITE, __NV_E4M3),
        __nv_cvt_float_to_fp8(value.z / scale, __NV_SATFINITE, __NV_E4M3),
        __nv_cvt_float_to_fp8(value.w / scale, __NV_SATFINITE, __NV_E4M3)};
    const float2 low = q4e_e4m3x2(codes[0] | codes[1] << 8);
    const float2 high = q4e_e4m3x2(codes[2] | codes[3] << 8);
    return make_float4(low.x * scale * block, low.y * scale * block, high.x * scale * block,
                       high.y * scale * block);
}

__device__ __forceinline__ float q4e_tg_fold(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xFFFFFFFFu, value, offset);
    }
    return value;
}

// The body every tiled entry shares. `row_of(tile, warp, r)` names row r of the warp's four
// (nullptr past the matrix). Lane l of each warp then holds the sum of row l / 8 for token l % 8,
// and `finish(tile, warp, r, t, sum)` is called by the whole warp (lanes whose token is past the
// batch included, with t >= batch), so an epilogue may exchange values across it. `partials`
// holds gridDim.y x Q4E_MAX_BATCH x (gridDim.x * 32) floats and `counters` gridDim.x zeros when
// gridDim.y > 1; the tile's last block puts its counter back to zero. columns is a whole number
// of the format's chunks (of Q4E_FP8_BLOCK for a block-scaled one, whose `row_of` also has
// `block_scale(tile, column)`), and the rows and x are 16-byte aligned. Token t reads row t of x,
// or row x_rows[t] when there is a row map.
//
// The weights and their scales are constant, so an entry whose `row_of` reads nothing an earlier
// kernel writes may open with OPT_PDL_LAUNCH() alone: the body issues its first round's weight
// loads and only then waits, before it reads x, the row map or anything else. (An entry that
// waited already, as the routed experts' must before they read their picks, loses nothing.)
template <typename Format, typename RowOf, typename Finish>
__device__ __forceinline__ void q4e_tiled_gemv(const RowOf& row_of, const Finish& finish,
                                               const float* __restrict__ x, int columns,
                                               int batch, float* __restrict__ partials,
                                               unsigned int* __restrict__ counters,
                                               const int* x_rows = nullptr) {
    constexpr int kQuads = Format::kColumns / 4;
    constexpr int kRound = Format::kColumns * 32;
    // [token][quad][lane] float4s: quad j of lane l is its chunk's columns 4j..4j+4, so a warp's
    // reads of any quad are consecutive 16-byte words.
    __shared__ __align__(16) float4 staged[Q4E_MAX_BATCH * kQuads * 32];
    __shared__ int last;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int splits = gridDim.y;
    const int rounds = (columns + kRound - 1) / kRound;
    const int share = (rounds + splits - 1) / splits;
    const int begin = blockIdx.y * share;
    const int end = min(rounds, begin + share);
    const typename Format::Weight* rows[Q4E_TG_ROWS];
#pragma unroll
    for (int r = 0; r < Q4E_TG_ROWS; ++r) {
        rows[r] = row_of(tile, warp, r);
    }

    float acc[Q4E_TG_ROWS][Q4E_MAX_BATCH];
#pragma unroll
    for (int r = 0; r < Q4E_TG_ROWS; ++r) {
#pragma unroll
        for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
            acc[r][t] = 0.0f;
        }
    }
    for (int round = begin; round < end; ++round) {
        const int base = round * kRound;
        const int width = min(kRound, columns - base);
        const bool in = lane * Format::kColumns < width;
        // The round's weights are in flight while its activations are staged.
        uint4 w[Q4E_TG_ROWS];
#pragma unroll
        for (int r = 0; r < Q4E_TG_ROWS; ++r) {
            if (in && rows[r] != nullptr) {
                w[r] = q4e_ldg(
                    reinterpret_cast<const uint4*>(rows[r] + base + lane * Format::kColumns));
            }
        }
        if (round == begin) {
            // The first round's weights stream while the previous kernel finishes.
            OPT_PDL_WAIT();
        }
        const int vectors = width >> 2;
        __syncthreads();
        for (int index = threadIdx.x; index < batch * vectors; index += Q4E_TG_THREADS) {
            const int t = index / vectors;
            const int v = index - t * vectors;
            const unsigned long long x_row = x_rows != nullptr ? x_rows[t] : t;
            float4 value = q4e_ldg(
                reinterpret_cast<const float4*>(x + x_row * columns + base) + v);
            if constexpr (Format::kBlockScaled) {
                // vectors is a multiple of 32, so a warp holds one token's whole run.
                const float block = row_of.block_scale(tile, base + v * 4);
                value = row_of.quantise ? q4e_tg_fp8_activation(value, block)
                                        : make_float4(value.x * block, value.y * block,
                                                      value.z * block, value.w * block);
            }
            staged[(t * kQuads + v % kQuads) * 32 + v / kQuads] = value;
        }
        __syncthreads();
        if (in) {
#pragma unroll
            for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
                if (t < batch) {
                    float4 quads[kQuads];
#pragma unroll
                    for (int j = 0; j < kQuads; ++j) {
                        quads[j] = staged[(t * kQuads + j) * 32 + lane];
                    }
#pragma unroll
                    for (int r = 0; r < Q4E_TG_ROWS; ++r) {
                        if (rows[r] != nullptr) {
                            acc[r][t] = Format::fma(w[r], quads, acc[r][t]);
                        }
                    }
                }
            }
        }
    }
    // A block with no rounds waits here, before it writes; for the others it is a no-op.
    OPT_PDL_WAIT();
    // Lane l keeps row l / 8, token l % 8.
    const int mine_r = lane >> 3;
    const int mine_t = lane & 7;
    float sum = 0.0f;
#pragma unroll
    for (int r = 0; r < Q4E_TG_ROWS; ++r) {
#pragma unroll
        for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
            if (t < batch) {
                const float folded = q4e_tg_fold(acc[r][t]);
                if (r == mine_r && t == mine_t) {
                    sum = folded;
                }
            }
        }
    }
    if (splits > 1) {
        const unsigned long long slots = (unsigned long long)gridDim.x * Q4E_TG_TILE;
        const unsigned long long slot =
            (unsigned long long)tile * Q4E_TG_TILE + warp * Q4E_TG_ROWS + mine_r;
        if (mine_t < batch) {
            partials[((unsigned long long)blockIdx.y * Q4E_MAX_BATCH + mine_t) * slots + slot] =
                sum;
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            last = atomicAdd(&counters[tile], 1u) == (unsigned int)(splits - 1);
        }
        __syncthreads();
        if (!last) {
            return;
        }
        __threadfence();
        if (mine_t < batch) {
            sum = 0.0f;
            for (int split = 0; split < splits; ++split) {
                sum += __ldcg(&partials[((unsigned long long)split * Q4E_MAX_BATCH + mine_t) *
                                            slots +
                                        slot]);
            }
        }
        if (threadIdx.x == 0) {
            counters[tile] = 0u;
        }
    }
    finish(tile, warp, mine_r, mine_t, sum);
}

// Rows tile * 32 + warp * 4 + r of a [rows, columns] matrix.
template <typename Weight>
struct Q4eTgRows {
    const Weight* w;
    int rows;
    int columns;
    __device__ const Weight* operator()(int tile, int warp, int r) const {
        const int row = tile * Q4E_TG_TILE + warp * Q4E_TG_ROWS + r;
        return row < rows ? w + (unsigned long long)row * columns : nullptr;
    }
};

// Q4eTgRows of an E4M3 matrix with its block scales, [ceil(rows / 128)][ceil(columns / 128)].
// `quantise` is the release's dynamic activation scheme; clear, the f32 activations multiply the
// block scale unquantised (a configuration built in code, never a release's).
struct Q4eTgBlockRows : Q4eTgRows<unsigned char> {
    const float* scales;
    int quantise;
    __device__ float block_scale(int tile, int column) const {
        const int blocks = (columns + Q4E_FP8_BLOCK - 1) / Q4E_FP8_BLOCK;
        return q4e_ldg(&scales[(tile * Q4E_TG_TILE / Q4E_FP8_BLOCK) * blocks +
                               column / Q4E_FP8_BLOCK]);
    }
};

// `scales`, when there are any, holds one factor a row for the finished sum.
struct Q4eTgStore {
    float* y;
    const float* scales;
    int rows;
    int batch;
    __device__ void operator()(int tile, int warp, int r, int t, float sum) const {
        const int row = tile * Q4E_TG_TILE + warp * Q4E_TG_ROWS + r;
        if (t < batch && row < rows) {
            y[(unsigned long long)t * rows + row] = scales != nullptr ? sum * scales[row] : sum;
        }
    }
};

// Q4eTgStore for a matrix of up to four parts stacked by rows, which start at 0, second, third and
// fourth (0 <= second <= third <= fourth <= rows): part p's outputs land as one [batch][its rows]
// block after the blocks of the parts before it, so each part of a fused projection is the next
// kernel's contiguous input at any batch. One token's outputs are the plain row order, and every
// boundary at `rows` is Q4eTgStore at any batch. A grouped launch (q4e_tg_group) stores its
// group's `count` tokens from `first` on into the layout of all `batch`.
struct Q4eTgPartStore {
    float* y;
    int rows;
    int batch;
    int second;
    int third;
    int fourth;
    int first;
    int count;
    __device__ void operator()(int tile, int warp, int r, int t, float sum) const {
        const int row = tile * Q4E_TG_TILE + warp * Q4E_TG_ROWS + r;
        if (t < count && row < rows) {
            const int start = row < second ? 0 : row < third ? second : row < fourth ? third : fourth;
            const int end = row < second ? second : row < third ? third : row < fourth ? fourth : rows;
            y[(unsigned long long)batch * start +
              (unsigned long long)(first + t) * (end - start) + row - start] = sum;
        }
    }
};

// A grouped tiled launch: blockIdx.z is a group of Q4E_MAX_BATCH tokens, each group the body of a
// launch of its own over its tokens with its own split partials and tile counters (`partials` and
// `counters` hold gridDim.z of a launch's each), so a token's sum is the chain of an ungrouped
// launch - a prefill chunk runs the decode GEMV's arithmetic in one launch instead of one an
// eight rows. Moves x, partials and counters to the group's, and returns its first token and count
// (zero or less past the batch).
__device__ __forceinline__ int2 q4e_tg_group(const float* __restrict__& x,
                                             float* __restrict__& partials,
                                             unsigned int* __restrict__& counters, int columns,
                                             int batch) {
    const int group = blockIdx.z;
    const int first = group * Q4E_MAX_BATCH;
    x += (unsigned long long)first * columns;
    partials += (unsigned long long)group * gridDim.y * Q4E_MAX_BATCH * gridDim.x * Q4E_TG_TILE;
    counters += (unsigned long long)group * gridDim.x;
    return make_int2(first, min(Q4E_MAX_BATCH, batch - first));
}

// y = W x through the tiled body: grid (ceil(rows / 32), splits), Q4E_TG_THREADS threads.
extern "C" __global__ void __launch_bounds__(Q4E_TG_THREADS, 3)
    q4e_gemv_tiled_bf16(const __nv_bfloat16* __restrict__ w, const float* __restrict__ x,
                        float* __restrict__ y, int rows, int columns, int batch,
                        float* __restrict__ partials, unsigned int* __restrict__ counters) {
    OPT_PDL_LAUNCH();
    if (batch > Q4E_MAX_BATCH) {
        OPT_PDL_WAIT();
        return;
    }
    q4e_tiled_gemv<Q4eTgBf16>(Q4eTgRows<__nv_bfloat16>{w, rows, columns},
                              Q4eTgStore{y, nullptr, rows, batch}, x, columns, batch, partials,
                              counters);
}

// q4e_gemv_tiled_bf16 with its outputs in Q4eTgPartStore's layout: GLM-5.3-Flash's fused BF16
// projections (KDA's q, k, v, f_a, g_a and b; the indexer's wk, compress gate and weights_proj).
// Grid (ceil(rows / 32), splits, ceil(batch / Q4E_MAX_BATCH)): q4e_tg_group.
extern "C" __global__ void __launch_bounds__(Q4E_TG_THREADS, 3)
    q4e_gemv_tiled_bf16_parts(const __nv_bfloat16* __restrict__ w, const float* __restrict__ x,
                              float* __restrict__ y, int rows, int columns, int batch,
                              float* __restrict__ partials, unsigned int* __restrict__ counters,
                              int second, int third, int fourth) {
    OPT_PDL_LAUNCH();
    const int2 group = q4e_tg_group(x, partials, counters, columns, batch);
    if (batch > Q4E_MAX_BATCH * (int)gridDim.z || group.y <= 0) {
        OPT_PDL_WAIT();
        return;
    }
    q4e_tiled_gemv<Q4eTgBf16>(
        Q4eTgRows<__nv_bfloat16>{w, rows, columns},
        Q4eTgPartStore{y, rows, batch, second, third, fourth, group.x, group.y}, x, columns,
        group.y, partials, counters);
}

// y = (W x) * scales[row] for an E4M3 W with one f32 scale a row (the drafter's matrices, which
// q4e_quantize_e4m3_rows writes): grid (ceil(rows / 32), splits), Q4E_TG_THREADS threads, and
// columns a multiple of 16.
extern "C" __global__ void __launch_bounds__(Q4E_TG_THREADS, 3)
    q4e_gemv_tiled_e4m3(const unsigned char* __restrict__ w, const float* __restrict__ scales,
                        const float* __restrict__ x, float* __restrict__ y, int rows, int columns,
                        int batch, float* __restrict__ partials,
                        unsigned int* __restrict__ counters) {
    OPT_PDL_LAUNCH();
    if (batch > Q4E_MAX_BATCH) {
        OPT_PDL_WAIT();
        return;
    }
    q4e_tiled_gemv<Q4eTgE4m3>(Q4eTgRows<unsigned char>{w, rows, columns},
                              Q4eTgStore{y, scales, rows, batch}, x, columns, batch, partials,
                              counters);
}

// y = W x_q for GLM-5.3-Flash's FP8 projections: W is E4M3 with one f32 scale a 128 x 128 block
// (weight_scale_inv, [ceil(rows / 128)][ceil(columns / 128)]) and x_q is x quantised as the
// checkpoint's dynamic activation scheme does it, per 128 columns (q4e_tg_fp8_activation), when
// `quantise` is set, and is x unquantised when it is clear. The outputs are in Q4eTgPartStore's
// layout (the MLA's fused q_a and kv_a). Grid (ceil(rows / 32), splits, ceil(batch /
// Q4E_MAX_BATCH)) as q4e_tg_group, Q4E_TG_THREADS threads, and columns a multiple of 128.
extern "C" __global__ void __launch_bounds__(Q4E_TG_THREADS, 3)
    q4e_gemv_tiled_e4m3_block(const unsigned char* __restrict__ w,
                              const float* __restrict__ scales, const float* __restrict__ x,
                              float* __restrict__ y, int rows, int columns, int batch,
                              float* __restrict__ partials, unsigned int* __restrict__ counters,
                              int quantise, int second, int third, int fourth) {
    OPT_PDL_LAUNCH();
    const int2 group = q4e_tg_group(x, partials, counters, columns, batch);
    if (batch > Q4E_MAX_BATCH * (int)gridDim.z || group.y <= 0 ||
        columns % Q4E_FP8_BLOCK != 0) {
        OPT_PDL_WAIT();
        return;
    }
    q4e_tiled_gemv<Q4eTgE4m3Block>(
        Q4eTgBlockRows{{w, rows, columns}, scales, quantise},
        Q4eTgPartStore{y, rows, batch, second, third, fourth, group.x, group.y}, x, columns,
        group.y, partials, counters);
}

// L2 prefetch of up to four weight spans (address, bytes; a zero length skips the span), each split
// into SPARK_PREFETCH_CHUNK runs over the block's threads. It runs on a side stream while the run's
// stream waits in an exchange, so the next kernels' weights arrive while DRAM would otherwise idle.
// A hint: nothing waits on it and nothing it reads is written during the run.
extern "C" __global__ void q4e_prefetch_l2(const uint64_t a0, const uint64_t n0, const uint64_t a1,
                                           const uint64_t n1, const uint64_t a2, const uint64_t n2,
                                           const uint64_t a3, const uint64_t n3) {
    const uint64_t address[4] = {a0, a1, a2, a3};
    const uint64_t length[4] = {n0, n1, n2, n3};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        for (uint64_t at = threadIdx.x * (uint64_t)SPARK_PREFETCH_CHUNK; at < length[s];
             at += blockDim.x * (uint64_t)SPARK_PREFETCH_CHUNK) {
            const uint64_t left = length[s] - at;
            spark_prefetch_l2(reinterpret_cast<const unsigned char*>(address[s]) + at,
                              left < SPARK_PREFETCH_CHUNK ? left : SPARK_PREFETCH_CHUNK);
        }
    }
}

// A BF16 [rows, columns] matrix as E4M3 with one f32 scale a row, a warp a row: the scale maps the
// row's largest magnitude to 448, E4M3's largest finite value, and each weight over it rounds to
// the nearest code. A row of zeros takes the scale 1.
extern "C" __global__ void q4e_quantize_e4m3_rows(const __nv_bfloat16* __restrict__ w,
                                                  unsigned char* __restrict__ out,
                                                  float* __restrict__ scales, int rows,
                                                  int columns) {
    OPT_PDL_ENTRY();
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) {
        return;
    }
    const __nv_bfloat16* source = w + (unsigned long long)row * columns;
    float largest = 0.0f;
    for (int column = lane; column < columns; column += 32) {
        largest = fmaxf(largest, fabsf(__bfloat162float(source[column])));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        largest = fmaxf(largest, __shfl_xor_sync(0xFFFFFFFFu, largest, offset));
    }
    const float scale = largest > 0.0f ? largest / 448.0f : 1.0f;
    if (lane == 0) {
        scales[row] = scale;
    }
    unsigned char* target = out + (unsigned long long)row * columns;
    for (int column = lane; column < columns; column += 32) {
        target[column] =
            __nv_cvt_float_to_fp8(__bfloat162float(source[column]) / scale, __NV_SATFINITE,
                                  __NV_E4M3);
    }
}

// The low-rank rows, then the injection rows when there are any.
struct Q4eTgHyperDownRows {
    const __nv_bfloat16* down;
    const __nv_bfloat16* inject;
    int lowrank;
    int rows;
    int width;
    __device__ const __nv_bfloat16* operator()(int tile, int warp, int r) const {
        const int row = tile * Q4E_TG_TILE + warp * Q4E_TG_ROWS + r;
        if (row >= rows) {
            return nullptr;
        }
        return row < lowrank ? down + (unsigned long long)row * width
                             : inject + (unsigned long long)(row - lowrank) * width;
    }
};

struct Q4eTgHyperDownFinish {
    float* low;
    float* injection;
    int hc;
    int lowrank;
    int rows;
    int batch;
    __device__ void operator()(int tile, int warp, int r, int t, float sum) const {
        const int row = tile * Q4E_TG_TILE + warp * Q4E_TG_ROWS + r;
        if (t >= batch) {
            return;
        }
        if (row < lowrank) {
            low[(unsigned long long)t * lowrank + row] = q4e_silu(sum / (float)hc);
        } else if (row < rows) {
            injection[(unsigned long long)t * hc + (row - lowrank)] =
                2.0f / (1.0f + __expf(-sum / (float)hc));
        }
    }
};

// q4e_hyper_down through the tiled body: grid (ceil((lowrank [+ hc]) / 32), splits).
extern "C" __global__ void __launch_bounds__(Q4E_TG_THREADS, 3)
    q4e_hyper_down_tiled(const float* __restrict__ normed, const __nv_bfloat16* __restrict__ down,
                         const __nv_bfloat16* __restrict__ inject, float* __restrict__ low,
                         float* __restrict__ injection, int hc, int width, int lowrank, int batch,
                         float* __restrict__ partials, unsigned int* __restrict__ counters) {
    OPT_PDL_LAUNCH();
    if (batch > Q4E_MAX_BATCH) {
        OPT_PDL_WAIT();
        return;
    }
    const int rows = inject != nullptr ? lowrank + hc : lowrank;
    q4e_tiled_gemv<Q4eTgBf16>(Q4eTgHyperDownRows{down, inject, lowrank, rows, width},
                              Q4eTgHyperDownFinish{low, injection, hc, lowrank, rows, batch},
                              normed, width, batch, partials, counters);
}

// Warp w of tile c mixes hidden column 8c + w, its four rows the column's four streams.
struct Q4eTgHyperUpRows {
    const __nv_bfloat16* up;
    int hidden;
    int lowrank;
    __device__ const __nv_bfloat16* operator()(int tile, int warp, int r) const {
        const int column = tile * Q4E_TG_WARPS + warp;
        if (column >= hidden) {
            return nullptr;
        }
        return up + (unsigned long long)(r * hidden + column) * lowrank;
    }
};

struct Q4eTgHyperUpFinish {
    const float* normed;
    float* mixed;
    int hidden;
    int batch;
    __device__ void operator()(int tile, int warp, int r, int t, float sum) const {
        const int column = tile * Q4E_TG_WARPS + warp;
        const bool owner = r == 0 && t < batch && column < hidden;
        // Summed from zero a stream at a time, as q4e_hyper_mix sums the prefill's streams.
        float accumulator = 0.0f;
#pragma unroll
        for (int stream = 0; stream < Q4E_TG_ROWS; ++stream) {
            const float mix = __shfl_sync(0xFFFFFFFFu, sum, stream * 8 + t);
            if (owner) {
                accumulator += (1.0f / (1.0f + __expf(-mix))) *
                               normed[(unsigned long long)t * Q4E_TG_ROWS * hidden +
                                      (unsigned long long)stream * hidden + column];
            }
        }
        if (owner) {
            mixed[(unsigned long long)t * hidden + column] = accumulator / (float)Q4E_TG_ROWS;
        }
    }
};

// q4e_hyper_up through the tiled body for four streams: grid (ceil(hidden / 8), 1). The inner
// dimension is the low rank, short enough that it is never split.
extern "C" __global__ void __launch_bounds__(Q4E_TG_THREADS, 3)
    q4e_hyper_up_tiled(const float* __restrict__ normed, const __nv_bfloat16* __restrict__ up,
                       const float* __restrict__ low, float* __restrict__ mixed, int hc,
                       int hidden, int lowrank, int batch) {
    OPT_PDL_LAUNCH();
    if (batch > Q4E_MAX_BATCH || hc != Q4E_TG_ROWS || gridDim.y != 1) {
        OPT_PDL_WAIT();
        return;
    }
    q4e_tiled_gemv<Q4eTgBf16>(Q4eTgHyperUpRows{up, hidden, lowrank},
                              Q4eTgHyperUpFinish{normed, mixed, hidden, batch}, low, lowrank,
                              batch, nullptr, nullptr);
}

// y[token][row] = sum over columns of w[row][column] * x[token][column].
//
// The decode GEMV keeps one register accumulator per token, which caps it at Q4E_MAX_BATCH (8)
// tokens per weight read; a prefill chunk of a hundred tokens would therefore stream the whole
// dense stack a dozen times. This kernel stages the chunk's activations and a 64-row weight tile
// in shared memory and walks the inner dimension once, so a weight byte is read once per chunk
// however many tokens it serves. Each thread owns an 8 x 8 (row, token) tile of the output, which
// is the tile size that balances shared-memory loads against fused multiply-adds.
//
// The K stages are double buffered through cp.async: the copy of the next 32 columns is issued
// into the other shared buffer before the arithmetic on the current 32 begins, so the global read
// latency of a stage overlaps the stage before it. A short or out-of-range 16-byte slot is
// zero-filled by the copy itself (src-size), which is what the synchronous path wrote by hand.
//
// The arithmetic is the GEMV's: bf16 widens to f32 exactly, one f32 product per weight, summed in
// ascending inner-dimension order - which is the host reference's order.
#define QG_THREADS 128
#define QG_ROWS 64
#define QG_KB 32
#define QG_TOKENS 128
// The activation tile is padded by four floats a row. A row stride of exactly 32 words puts every
// row in the same bank, so the sixteen token groups reading column `k` at once serialise into a
// sixteen-way conflict and the tile costs more than the arithmetic it feeds. The padding is four
// words rather than one so a row stays sixteen-byte aligned for cp.async: the bank of row t is
// then 4t mod 32, which splits the sixteen groups over eight banks.
#define QG_XSTRIDE (QG_KB + 4)

__device__ __forceinline__ uint32_t q4e_smem_addr(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

// 16 bytes of global memory into shared, zero-filling the tail when `bytes` is short. `bytes == 0`
// does not read the source at all, which covers rows past the end of the matrix.
__device__ __forceinline__ void q4e_cp16_stream(uint32_t dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void q4e_cp16_cached(uint32_t dst, const void* src, int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void q4e_cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void q4e_cp_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// Issue the cp.async copy of one K stage into `s_w` / `s_x`.
__device__ __forceinline__ void q4e_gemm_stage_async(__nv_bfloat16 (*s_w)[QG_KB],
                                                     float (*s_x)[QG_XSTRIDE],
                                                     const __nv_bfloat16* __restrict__ w,
                                                     const float* __restrict__ x, int rows,
                                                     int columns, int batch, int row0, int k0,
                                                     int tx) {
    // The weight tile is (blockDim.x / 2) rows of 32 bf16 = 4 slots of eight per row; the row
    // tile shrinks with the block, so a matrix with few rows can be cut into more blocks.
    for (int index = tx; index < (blockDim.x >> 1) * (QG_KB / 8); index += blockDim.x) {
        const int row = index / (QG_KB / 8);
        const int slot = (index - row * (QG_KB / 8)) * 8;
        const int column = k0 + slot;
        const int global_row = row0 + row;
        const int bytes =
            (global_row < rows) ? 2 * min(8, max(0, columns - column)) : 0;
        const __nv_bfloat16* src = w + (unsigned long long)global_row * columns + column;
        q4e_cp16_stream(q4e_smem_addr(&s_w[row][slot]), src, bytes);
    }
    // The activation tile is QG_TOKENS rows of 32 floats = 8 slots of four per row.
    for (int index = tx; index < QG_TOKENS * (QG_KB / 4); index += blockDim.x) {
        const int token = index / (QG_KB / 4);
        const int slot = (index - token * (QG_KB / 4)) * 4;
        const int column = k0 + slot;
        const int bytes = (token < batch) ? 4 * min(4, max(0, columns - column)) : 0;
        const float* src = x + (unsigned long long)token * columns + column;
        q4e_cp16_cached(q4e_smem_addr(&s_x[token][slot]), src, bytes);
    }
}

__device__ __forceinline__ void q4e_gemm_compute(const __nv_bfloat16 (*s_w)[QG_KB],
                                                  const float (*s_x)[QG_XSTRIDE],
                                                  float accumulators[8][8], int row_in_block,
                                                  int token_group) {
    // The thread's eight tokens are strided by one group, not contiguous: sixteen lane groups
    // reading a column at once then land on eight banks instead of one.
    const int token_base = token_group;
    const int token_step = QG_TOKENS / 8;
#pragma unroll
    for (int k = 0; k < QG_KB; ++k) {
        float weight[8];
        float activation[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            weight[i] = __bfloat162float(s_w[row_in_block + i][k]);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            activation[j] = s_x[token_base + j * token_step][k];
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                accumulators[i][j] += weight[i] * activation[j];
            }
        }
    }
}

// `epilogue` is applied as the tile is written: 1 = silu(v / scale), 2 = sigmoid of v / scale
// scaled to two - the hyper-connection's two nonlinearities - and 0 leaves the dot product.
//
// `segments` splits the inner dimension across blockIdx.y, each block summing its own contiguous
// slice into `partials[segment]`; `q4e_gemm_reduce` then adds the slices in order and applies the
// epilogue. A narrow matrix leaves only (rows / 8) * (batch / 8) threads of work - 640 for the
// hyper-connection's 320-row down projection, a fortieth of the GPU - and no tile shape fixes
// that without splitting K. With one segment the block writes `y` itself, as before.
//
// blockIdx.z picks one of gridDim.z runs of `batch` tokens laid end to end (a prefill chunk's
// 128-token tiles): its tokens, outputs and partials sit z runs on. A token's sum depends only
// on `columns` and `segments`, so it has the bits its own launch would give it.
extern "C" __global__ void q4e_gemm_bf16(const __nv_bfloat16* __restrict__ w,
                                          const float* __restrict__ x, float* __restrict__ y,
                                          int rows, int columns, int batch, int epilogue,
                                          float scale, int segments, float* __restrict__ partials) {
    OPT_PDL_ENTRY();
    __shared__ __nv_bfloat16 s_w[2][QG_ROWS][QG_KB];
    __shared__ float s_x[2][QG_TOKENS][QG_XSTRIDE];
    if (columns <= 0 || batch <= 0 || batch > QG_TOKENS) {
        return;
    }
    {
        const unsigned long long run = blockIdx.z;
        x += run * batch * columns;
        y += run * batch * rows;
        if (segments > 1) {
            partials += run * segments * batch * rows;
        }
    }
    const int tx = threadIdx.x;
    const int row_in_block = (tx >> 4) * 8;
    const int token_group = tx & 15;
    const int token_step = QG_TOKENS / 8;
    const int row0 = blockIdx.x * (blockDim.x >> 1);
    // The slice of the inner dimension this block owns, rounded up to a whole K stage.
    const int segment = (segments > 1) ? (int)blockIdx.y : 0;
    int k_begin = 0;
    int k_end = columns;
    if (segments > 1) {
        const int per = ((columns + segments - 1) / segments + QG_KB - 1) / QG_KB * QG_KB;
        k_begin = min(columns, segment * per);
        k_end = min(columns, k_begin + per);
    }
    float accumulators[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            accumulators[i][j] = 0.0f;
        }
    }
    if ((columns & 7) == 0) {
        // A 16-byte slot is aligned in both matrices only when the row stride is: eight bf16 for
        // the weights, four floats for the activations.
        q4e_gemm_stage_async(s_w[0], s_x[0], w, x, rows, columns, batch, row0, k_begin, tx);
        q4e_cp_commit();
        int buffer = 0;
        for (int k0 = k_begin; k0 < k_end; k0 += QG_KB) {
            if (k0 + QG_KB < k_end) {
                q4e_gemm_stage_async(s_w[buffer ^ 1], s_x[buffer ^ 1], w, x, rows, columns, batch,
                                      row0, k0 + QG_KB, tx);
            }
            // Committed even when the prefetch was skipped: the empty group keeps one wait_group
            // depth right for every iteration.
            q4e_cp_commit();
            // At most the prefetch may still be in flight; the buffer about to be read has landed.
            q4e_cp_wait<1>();
            __syncthreads();
            q4e_gemm_compute(s_w[buffer], s_x[buffer], accumulators, row_in_block, token_group);
            // No thread may still be reading this buffer when the next stage copies into it.
            __syncthreads();
            buffer ^= 1;
        }
    } else {
        for (int k0 = k_begin; k0 < k_end; k0 += QG_KB) {
            for (int index = tx; index < (blockDim.x >> 1) * QG_KB; index += blockDim.x) {
                const int row = index / QG_KB;
                const int slot = index - row * QG_KB;
                const int column = k0 + slot;
                const int global_row = row0 + row;
                s_w[0][row][slot] = (global_row < rows && column < columns)
                                        ? w[(unsigned long long)global_row * columns + column]
                                        : __float2bfloat16(0.0f);
            }
            for (int index = tx; index < QG_TOKENS * QG_KB; index += blockDim.x) {
                const int token = index / QG_KB;
                const int slot = index - token * QG_KB;
                const int column = k0 + slot;
                s_x[0][token][slot] = (token < batch && column < columns)
                                          ? x[(unsigned long long)token * columns + column]
                                          : 0.0f;
            }
            __syncthreads();
            q4e_gemm_compute(s_w[0], s_x[0], accumulators, row_in_block, token_group);
            __syncthreads();
        }
    }
    // A split block writes its raw slice sum; the epilogue belongs to the reduction, which sees
    // the whole inner dimension.
    float* out = (segments > 1)
                     ? partials + (unsigned long long)segment * batch * rows
                     : y;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int global_row = row0 + row_in_block + i;
        if (global_row >= rows) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int token = token_group + j * token_step;
            if (token < batch) {
                float value = accumulators[i][j];
                if (segments == 1) {
                    if (epilogue == 1) {
                        value = q4e_silu(value / scale);
                    } else if (epilogue == 2) {
                        value = 2.0f / (1.0f + __expf(-value / scale));
                    }
                }
                out[(unsigned long long)token * rows + global_row] = value;
            }
        }
    }
}

// Add a split GEMM's segment sums in order and write the epilogue once. blockIdx.y picks one of
// gridDim.y runs of `count` outputs, each with its own `segments` slices, as q4e_gemm_bf16's
// blockIdx.z lays them out.
extern "C" __global__ void q4e_gemm_reduce(const float* __restrict__ partials,
                                            float* __restrict__ y, int count, int segments,
                                            int epilogue, float scale) {
    OPT_PDL_ENTRY();
    partials += (unsigned long long)blockIdx.y * segments * count;
    y += (unsigned long long)blockIdx.y * count;
    for (int index = (int)(blockIdx.x * blockDim.x + threadIdx.x); index < count;
         index += (int)(gridDim.x * blockDim.x)) {
        float value = partials[index];
        for (int segment = 1; segment < segments; ++segment) {
            value += partials[(unsigned long long)segment * count + index];
        }
        if (epilogue == 1) {
            value = q4e_silu(value / scale);
        } else if (epilogue == 2) {
            value = 2.0f / (1.0f + __expf(-value / scale));
        }
        y[index] = value;
    }
}

// Fused shared-expert gate_proj + up_proj + SwiGLU + shared_expert_gate scalar projection.
extern "C" __global__ void q4e_shared_gate_up(
    const __nv_bfloat16* __restrict__ gate_w, const __nv_bfloat16* __restrict__ up_w,
    const __nv_bfloat16* __restrict__ shared_gate_w, const float* __restrict__ x,
    float* __restrict__ out_swiglu, float* __restrict__ out_gate, int intermediate, int hidden,
    int batch) {
    OPT_PDL_ENTRY();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int row = blockIdx.x * warps + warp;
    if (row > intermediate || batch > Q4E_MAX_BATCH) {
        return;
    }
    if (row == intermediate) {
        if (batch <= 1) {
            float g = q4e_row_dot_bf16(shared_gate_w, x, hidden, lane);
            g = q4e_warp_sum(g);
            if (lane == 0) {
                out_gate[0] = g;
            }
            return;
        }
        float totals[Q4E_MAX_BATCH];
        q4e_row_dot_bf16_batch(shared_gate_w, x, hidden, lane, batch, totals);
#pragma unroll
        for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
            if (t < batch) {
                float sum = q4e_warp_sum(totals[t]);
                if (lane == 0) {
                    out_gate[t] = sum;
                }
            }
        }
        return;
    }
    const __nv_bfloat16* g_row = gate_w + (unsigned long long)row * hidden;
    const __nv_bfloat16* u_row = up_w + (unsigned long long)row * hidden;
    if (batch <= 1) {
        float g, u;
        q4e_row_dot_bf16_pair(g_row, u_row, x, hidden, lane, &g, &u);
        g = q4e_warp_sum(g);
        u = q4e_warp_sum(u);
        if (lane == 0) {
            out_swiglu[row] = q4e_silu(g) * u;
        }
        return;
    }
    float g_totals[Q4E_MAX_BATCH];
    float u_totals[Q4E_MAX_BATCH];
    q4e_row_dot_bf16_batch(g_row, x, hidden, lane, batch, g_totals);
    q4e_row_dot_bf16_batch(u_row, x, hidden, lane, batch, u_totals);
#pragma unroll
    for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
        if (t < batch) {
            float g = q4e_warp_sum(g_totals[t]);
            float u = q4e_warp_sum(u_totals[t]);
            if (lane == 0) {
                out_swiglu[(unsigned long long)t * intermediate + row] = q4e_silu(g) * u;
            }
        }
    }
}

// out = x * rsqrt(mean(x^2) per group + eps) * (zero_centred ? 1 + w : w).
// weight_per_group == 0 reads a width-element weight per token; otherwise the weight repeats every
// group.
extern "C" __global__ void q4e_rmsnorm(const float* x, const __nv_bfloat16* __restrict__ weight,
                                       float* out, int width, int group, float eps,
                                       int zero_centred, int weight_per_group) {
    OPT_PDL_ENTRY();
    int start = blockIdx.x * group;
    float sum = 0.0f;
    for (int offset = threadIdx.x; offset < group; offset += blockDim.x) {
        float value = x[start + offset];
        sum += value * value;
    }
    __shared__ float scratch[Q4E_BLOCK];
    float mean = q4e_block_sum(sum, scratch) / group;
    float scale = rsqrtf(mean + eps);
    for (int offset = threadIdx.x; offset < group; offset += blockDim.x) {
        int index = start + offset;
        float gain = __bfloat162float(weight[weight_per_group ? offset : (index % width)]);
        out[index] = x[index] * scale * (zero_centred ? (1.0f + gain) : gain);
    }
}

// RMSNorm over a row-strided buffer: blockIdx.x is the group inside the row, blockIdx.y the row.
// The dense q4e_rmsnorm numbers its groups consecutively over one flat buffer; this one keeps a
// source and a destination stride, which is what the indexer needs (its query occupies the first
// index_heads * index_dim floats of a wider per-token row).
extern "C" __global__ void q4e_rmsnorm_rows(const float* x, const __nv_bfloat16* __restrict__ weight,
                                            float* out, int width, int group, float eps,
                                            int zero_centred, int weight_per_group,
                                            int src_stride, int dst_stride) {
    OPT_PDL_ENTRY();
    const int row = blockIdx.y;
    const int dst_row = row * dst_stride;
    const int src_row = row * src_stride;
    const int group_index = blockIdx.x;
    const int src_start = src_row + group_index * group;
    const int dst_start = dst_row + group_index * group;
    float sum = 0.0f;
    for (int offset = threadIdx.x; offset < group; offset += blockDim.x) {
        float value = x[src_start + offset];
        sum += value * value;
    }
    __shared__ float scratch[Q4E_BLOCK];
    float mean = q4e_block_sum(sum, scratch) / group;
    float scale = rsqrtf(mean + eps);
    for (int offset = threadIdx.x; offset < group; offset += blockDim.x) {
        int index = dst_start + offset;
        int weight_index = weight_per_group ? offset : (group_index * group + offset) % width;
        float gain = __bfloat162float(weight[weight_index]);
        out[index] = x[src_start + offset] * scale * (zero_centred ? (1.0f + gain) : gain);
    }
}

// dst[i] = BF16(src[i]), the KV cache's narrowing append: the projections and their rotary
// stay f32, and the cache keeps what the reference keeps, the model dtype.
extern "C" __global__ void q4e_cache_bf16(__nv_bfloat16* __restrict__ dst,
                                          const float* __restrict__ src, int count) {
    OPT_PDL_ENTRY();
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += blockDim.x * gridDim.x) {
        dst[index] = __float2bfloat16_rn(src[index]);
    }
}

// dst[row][0..width] = BF16(src[row][0..width]) for rows rows, with independent strides: the
// indexer's per-token row into its BF16 key cache.
extern "C" __global__ void q4e_cache_rows_bf16(__nv_bfloat16* __restrict__ dst,
                                               const float* __restrict__ src, int width,
                                               int dst_stride, int src_stride) {
    OPT_PDL_ENTRY();
    const int row = blockIdx.y;
    __nv_bfloat16* d = dst + (unsigned long long)row * dst_stride;
    const float* s = src + (unsigned long long)row * src_stride;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < width;
         index += blockDim.x * gridDim.x) {
        d[index] = __float2bfloat16_rn(s[index]);
    }
}

// destination[i] += source[i].
extern "C" __global__ void q4e_zero(float* __restrict__ destination, int width) {
    OPT_PDL_ENTRY();
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < width;
         index += blockDim.x * gridDim.x) {
        destination[index] = 0.0f;
    }
}

extern "C" __global__ void q4e_add(float* __restrict__ destination,
                                   const float* __restrict__ source, int width) {
    OPT_PDL_ENTRY();
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < width;
         index += blockDim.x * gridDim.x) {
        destination[index] += source[index];
    }
}

// Hyper-connection read pass one: bottleneck projection low = silu((down * normed) / hc) plus
// optional fused injection gate injection = 2 * sigmoid((inject * normed) / hc) when inject is
// non-null.
extern "C" __global__ void q4e_hyper_down(const float* __restrict__ normed,
                                          const __nv_bfloat16* __restrict__ down,
                                          const __nv_bfloat16* __restrict__ inject,
                                          float* __restrict__ low,
                                          float* __restrict__ injection, int hc, int width,
                                          int lowrank, int batch) {
    OPT_PDL_ENTRY();
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    int row = blockIdx.x * warps + warp;
    int total_rows = inject != nullptr ? (lowrank + hc) : lowrank;
    if (row >= total_rows || batch > Q4E_MAX_BATCH) {
        return;
    }
    const bool is_inject = row >= lowrank;
    const int local_row = is_inject ? (row - lowrank) : row;
    const __nv_bfloat16* w_row =
        (is_inject ? inject : down) + (unsigned long long)local_row * width;
    if (batch <= 1) {
        float accumulator = q4e_row_dot_bf16(w_row, normed, width, lane);
        accumulator = q4e_warp_sum(accumulator);
        if (lane == 0) {
            if (is_inject) {
                injection[local_row] = 2.0f / (1.0f + __expf(-accumulator / (float)hc));
            } else {
                low[local_row] = q4e_silu(accumulator / (float)hc);
            }
        }
        return;
    }
    float totals[Q4E_MAX_BATCH];
    q4e_row_dot_bf16_batch(w_row, normed, width, lane, batch, totals);
#pragma unroll
    for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
        if (t < batch) {
            float sum = q4e_warp_sum(totals[t]);
            if (lane == 0) {
                if (is_inject) {
                    injection[(unsigned long long)t * hc + local_row] =
                        2.0f / (1.0f + __expf(-sum / (float)hc));
                } else {
                    low[(unsigned long long)t * lowrank + local_row] =
                        q4e_silu(sum / (float)hc);
                }
            }
        }
    }
}

// Hyper-connection read pass two: one warp per hidden column mixes the hc streams.
extern "C" __global__ void q4e_hyper_up(const float* __restrict__ normed,
                                        const __nv_bfloat16* __restrict__ up,
                                        const float* __restrict__ low,
                                        float* __restrict__ mixed, int hc, int hidden, int lowrank,
                                        int batch) {
    OPT_PDL_ENTRY();
    if (lowrank > Q4E_MAX_LOWRANK || batch > Q4E_MAX_BATCH) {
        return;
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    int column = blockIdx.x * warps + warp;
    if (column >= hidden) {
        return;
    }
    int width = hc * hidden;
    if (batch <= 1) {
        if (hc == 4 && (lowrank & 7) == 0 &&
            ((reinterpret_cast<unsigned long long>(up) & 15u) == 0)) {
            const __nv_bfloat16* s0 = up + (unsigned long long)(0 * hidden + column) * lowrank;
            const __nv_bfloat16* s1 = up + (unsigned long long)(1 * hidden + column) * lowrank;
            const __nv_bfloat16* s2 = up + (unsigned long long)(2 * hidden + column) * lowrank;
            const __nv_bfloat16* s3 = up + (unsigned long long)(3 * hidden + column) * lowrank;
            float m0 = q4e_row_dot_bf16(s0, low, lowrank, lane);
            float m1 = q4e_row_dot_bf16(s1, low, lowrank, lane);
            float m2 = q4e_row_dot_bf16(s2, low, lowrank, lane);
            float m3 = q4e_row_dot_bf16(s3, low, lowrank, lane);
            m0 = q4e_warp_sum(m0);
            m1 = q4e_warp_sum(m1);
            m2 = q4e_warp_sum(m2);
            m3 = q4e_warp_sum(m3);
            // Summed from zero a stream at a time, as the batched pass below and q4e_hyper_mix sum
            // it. One expression of four products contracted into a different pair of FMAs, so a
            // verify row's mix differed from the one-token step's in the last bit.
            if (lane == 0) {
                float acc = 0.0f;
                acc += (1.0f / (1.0f + __expf(-m0))) * normed[0 * hidden + column];
                acc += (1.0f / (1.0f + __expf(-m1))) * normed[1 * hidden + column];
                acc += (1.0f / (1.0f + __expf(-m2))) * normed[2 * hidden + column];
                acc += (1.0f / (1.0f + __expf(-m3))) * normed[3 * hidden + column];
                mixed[column] = acc / (float)hc;
            }
            return;
        }
        float accumulator = 0.0f;
        for (int stream = 0; stream < hc; ++stream) {
            const __nv_bfloat16* source =
                up + (unsigned long long)(stream * hidden + column) * lowrank;
            float mix = q4e_row_dot_bf16(source, low, lowrank, lane);
            mix = q4e_warp_sum(mix);
            if (lane == 0) {
                accumulator +=
                    (1.0f / (1.0f + __expf(-mix))) * normed[stream * hidden + column];
            }
        }
        if (lane == 0) {
            mixed[column] = accumulator / (float)hc;
        }
        return;
    }
    if (column >= hidden) {
        return;
    }
    float acc[Q4E_MAX_BATCH];
#pragma unroll
    for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
        acc[t] = 0.0f;
    }
    for (int stream = 0; stream < hc; ++stream) {
        const __nv_bfloat16* source =
            up + (unsigned long long)(stream * hidden + column) * lowrank;
        float totals[Q4E_MAX_BATCH];
        q4e_row_dot_bf16_batch(source, low, lowrank, lane, batch, totals);
#pragma unroll
        for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
            if (t < batch) {
                float mix = q4e_warp_sum(totals[t]);
                if (lane == 0) {
                    acc[t] += (1.0f / (1.0f + __expf(-mix))) *
                              normed[(unsigned long long)t * width + stream * hidden + column];
                }
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
            if (t < batch) {
                mixed[(unsigned long long)t * hidden + column] = acc[t] / (float)hc;
            }
        }
    }
}

// The mixing half of hyper_up, after the GEMM has produced every stream's raw dot product:
// mixed[t][column] = mean over streams of sigmoid(raw[t][s][column]) * normed[t][s][column].
extern "C" __global__ void q4e_hyper_mix(const float* __restrict__ raw,
                                          const float* __restrict__ normed,
                                          float* __restrict__ mixed, int hidden, int hc, int batch) {
    OPT_PDL_ENTRY();
    const int total = batch * hidden;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
         index += blockDim.x * gridDim.x) {
        const int t = index / hidden;
        const int column = index - t * hidden;
        const unsigned long long base = (unsigned long long)t * hc * hidden + column;
        float accumulator = 0.0f;
        for (int stream = 0; stream < hc; ++stream) {
            const unsigned long long at = base + (unsigned long long)stream * hidden;
            accumulator += (1.0f / (1.0f + __expf(-raw[at]))) * normed[at];
        }
        mixed[index] = accumulator / (float)hc;
    }
}

// Standalone hyper-connection injection gate, kept for ABI completeness.
extern "C" __global__ void q4e_hyper_inject(const float* __restrict__ normed,
                                            const __nv_bfloat16* __restrict__ inject,
                                            float* __restrict__ out, int hc, int width) {
    OPT_PDL_ENTRY();
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    for (int stream = warp; stream < hc; stream += warps) {
        float accumulator =
            q4e_row_dot_bf16(inject + (unsigned long long)stream * width, normed, width, lane);
        accumulator = q4e_warp_sum(accumulator);
        if (lane == 0) {
            out[stream] = 2.0f / (1.0f + __expf(-accumulator / (float)hc));
        }
    }
}

// Inject a branch output back into the widened residual streams (gridDim.y = batch, a thread a
// column for every stream).
//
// With world > 1 the branch is this rank's partial of a tensor-parallel sum, folded here as
// q4e_mhc_post folds it: in ascending rank order from 0.0, this rank's own from branch and every
// peer's from its slot of slots ([world][batch][hidden], this rank's slot unread), so every rank
// injects the same bits.
extern "C" __global__ void q4e_inject(float* __restrict__ hyper, const float* __restrict__ branch,
                                      const float* __restrict__ injection, int hc, int hidden,
                                      const float* __restrict__ slots, int rank, int world) {
    OPT_PDL_ENTRY();
    const int b = blockIdx.y;
    float* hyper_b = hyper + (unsigned long long)b * hc * hidden;
    const float* injection_b = injection + (unsigned long long)b * hc;
    const unsigned long long slot = (unsigned long long)gridDim.y * hidden;
    for (int column = blockIdx.x * blockDim.x + threadIdx.x; column < hidden;
         column += blockDim.x * gridDim.x) {
        const unsigned long long at = (unsigned long long)b * hidden + column;
        float y = branch[at];
        if (world > 1) {
            float sum = 0.0f;
            for (int r = 0; r < world; ++r) {
                sum = __fadd_rn(sum, r == rank ? y : slots[r * slot + at]);
            }
            y = sum;
        }
        for (int stream = 0; stream < hc; ++stream) {
            hyper_b[(unsigned long long)stream * hidden + column] += y * injection_b[stream];
        }
    }
}

// out[i] = silu(gate[i]) * up[i].
extern "C" __global__ void q4e_swiglu(const float* __restrict__ gate, const float* __restrict__ up,
                                      float* __restrict__ out, int width) {
    OPT_PDL_ENTRY();
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < width;
         index += blockDim.x * gridDim.x) {
        out[index] = q4e_silu(gate[index]) * up[index];
    }
}

// Softmax over router logits and top-k selection with renormalisation (blockIdx.x = batch token).
// Note: logits and probabilities may alias the same buffer.
extern "C" __global__ void q4e_router(const float* logits, float* probabilities, int* indices,
                                      int experts, int topk) {
    OPT_PDL_ENTRY();
    __shared__ float scratch[Q4E_BLOCK];
    __shared__ int scratch_index[Q4E_BLOCK];
    __shared__ float probability[Q4E_MAX_EXPERTS];
    __shared__ int picked[Q4E_MAX_TOPK];
    __shared__ float chosen[Q4E_MAX_TOPK];
    if (experts <= 0 || experts > Q4E_MAX_EXPERTS || topk <= 0 || topk > Q4E_MAX_TOPK ||
        topk > experts) {
        return;
    }
    int b = blockIdx.x;
    const float* logits_b = logits + (unsigned long long)b * experts;
    float* probs_b = probabilities + (unsigned long long)b * experts;
    int* indices_b = indices + (unsigned long long)b * topk;
    int thread = threadIdx.x;
    float local_max = -INFINITY;
    for (int expert = thread; expert < experts; expert += blockDim.x) {
        local_max = fmaxf(local_max, logits_b[expert]);
    }
    // The helper's closing barrier is what lets the sum below reuse scratch: read inline, a slow
    // warp could take thread 0's partial sum for the maximum and route on skewed probabilities.
    const float maximum = q4e_block_max(local_max, scratch);

    float local_sum = 0.0f;
    for (int expert = thread; expert < experts; expert += blockDim.x) {
        float value = expf(logits_b[expert] - maximum);
        probability[expert] = value;
        local_sum += value;
    }
    float sum = q4e_block_sum(local_sum, scratch);
    for (int expert = thread; expert < experts; expert += blockDim.x) {
        float value = probability[expert] / sum;
        probability[expert] = value;
        probs_b[expert] = value;
    }
    __syncthreads();

    for (int slot = 0; slot < topk; ++slot) {
        float best_value = -2.0f;
        int best = 0x7FFFFFFF;
        for (int expert = thread; expert < experts; expert += blockDim.x) {
            if (probability[expert] > -0.5f) {
                q4e_take_better(probability[expert], expert, &best_value, &best);
            }
        }
        q4e_block_best(&best_value, &best, scratch, scratch_index);
        if (thread == 0) {
            picked[slot] = best;
            chosen[slot] = best_value;
            probability[best] = -1.0f;
        }
        __syncthreads();
    }
    if (thread == 0) {
        float normaliser = 0.0f;
        for (int slot = 0; slot < topk; ++slot) {
            normaliser += chosen[slot];
        }
        if (normaliser <= 0.0f) {
            normaliser = 1.0f;
        }
        for (int slot = 0; slot < topk; ++slot) {
            probs_b[slot] = chosen[slot] / normaliser;
            indices_b[slot] = picked[slot];
        }
    }
}

// Test seam exposing E4M3 decode across all 256 codes.
extern "C" __global__ void q4e_decode_f8e4m3(const unsigned char* codes, float* out, int count) {
    OPT_PDL_ENTRY();
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += blockDim.x * gridDim.x) {
        out[index] = q4e_e4m3(codes[index]);
    }
}

// The routed-expert block order: perm[rank] is the (token, slot) index that sorts into rank,
// ordered by expert id with ties kept in slot order. run_bounds[2r] and run_bounds[2r + 1] are
// the first and last rank of the contiguous run of ranks that share rank r's expert.
//
// The expert kernels used to build this inline, with every block ranking the whole list. That is
// O(slots^2) per block, and a prefill chunk keeps ten slots per token in flight, so at a batch of
// 128 it was a hundred and sixty billion comparisons across the launch - the reason a 128-token
// chunk measured slower than a 35-token one. One block per layer is enough; the ordering is what
// lets one block serve a whole run, and the run bounds are what tell it the run's extent.
#define Q4E_MAX_PERMUTE 4096

extern "C" __global__ void q4e_moe_permute(const int* __restrict__ indices, int* __restrict__ perm,
                                            int* __restrict__ run_bounds, int total_slots) {
    OPT_PDL_ENTRY();
    __shared__ int s_indices[Q4E_MAX_PERMUTE];
    __shared__ int s_experts[Q4E_MAX_PERMUTE];
    for (int i = threadIdx.x; i < total_slots; i += blockDim.x) {
        perm[i] = i;
    }
    if (total_slots <= 0 || total_slots > Q4E_MAX_PERMUTE) {
        return;
    }
    for (int i = threadIdx.x; i < total_slots; i += blockDim.x) {
        s_indices[i] = indices[i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < total_slots; i += blockDim.x) {
        const int mine = s_indices[i];
        int rank = 0;
        for (int j = 0; j < total_slots; ++j) {
            const int theirs = s_indices[j];
            if (theirs < mine || (theirs == mine && j < i)) {
                ++rank;
            }
        }
        perm[rank] = i;
    }
    __syncthreads();
    for (int r = threadIdx.x; r < total_slots; r += blockDim.x) {
        s_experts[r] = s_indices[perm[r]];
    }
    __syncthreads();
    // Runs are short - a chunk's slots outnumber its distinct experts about five to two - so the
    // two walks are a handful of shared-memory reads each, not a second sort.
    for (int r = threadIdx.x; r < total_slots; r += blockDim.x) {
        const int expert = s_experts[r];
        int begin = r;
        while (begin > 0 && s_experts[begin - 1] == expert) {
            --begin;
        }
        int end = r;
        while (end + 1 < total_slots && s_experts[end + 1] == expert) {
            ++end;
        }
        run_bounds[2 * r] = begin;
        run_bounds[2 * r + 1] = end;
    }
}

// Routed experts pass one: simultaneous gate + up NVFP4 row dot products with conflict-free
// shared-memory activation staging (blockIdx.y = slot, blockIdx.z = batch token).
//
// One block per expert run, not per slot. Every slot of a run reads the same expert's weight
// rows, so the block that sorts first in the run recomputes the run's other tokens itself and the
// rest of the run's blocks exit before their staging loop. The expert's rows are then read from
// DRAM once per chunk and land in L2 for the run's later members instead of being fetched again.
extern "C" __global__ __launch_bounds__(256, 4) void q4e_expert_gate_up(
    const unsigned long long* __restrict__ table, const float* __restrict__ scales,
    const int* __restrict__ indices, const int* __restrict__ perm,
    const int* __restrict__ run_bounds, const float* __restrict__ x,
    float* __restrict__ activated, int slots, int hidden, int intermediate) {
    OPT_PDL_ENTRY();
    __shared__ __align__(16) float source[Q4E_MEMBER_GROUP * Q4E_MAX_ROW_PADDED];
    __shared__ float lut[16];
    __shared__ float2 lut_pair[256];
    if (hidden <= 0 || hidden > Q4E_MAX_ROW) {
        return;
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int rows = (blockDim.x >> 5) * Q4E_ROWS_PER_WARP;
    int first_row = blockIdx.x * rows + warp * Q4E_ROWS_PER_WARP;
    if (blockIdx.y >= slots) {
        return;
    }
    // blockIdx.y + blockIdx.z * slots is the rank; the run bounds say where its run starts.
    const int rank = (int)blockIdx.z * slots + (int)blockIdx.y;
    const int begin = run_bounds[2 * rank];
    if (rank != begin) {
        return;
    }
    const int members = run_bounds[2 * rank + 1] - begin + 1;
    const int expert = indices[perm[begin]];
    int groups = (hidden + 15) / 16;
    int packed = (hidden + 1) / 2;
    // The warp's first gate/up rows and how many of its rows the matrix has.
    const unsigned char* grow = nullptr;
    const unsigned char* gsrow = nullptr;
    const unsigned char* urow = nullptr;
    const unsigned char* usrow = nullptr;
    const int valid = max(0, min(Q4E_ROWS_PER_WARP, intermediate - first_row));
    float g2 = 0.0f;
    float u2 = 0.0f;
    if (expert >= 0) {
        const unsigned long long row = (unsigned long long)first_row;
        grow = (const unsigned char*)table[expert * 6 + 0] + row * packed;
        gsrow = (const unsigned char*)table[expert * 6 + 1] + row * groups;
        urow = (const unsigned char*)table[expert * 6 + 2] + row * packed;
        usrow = (const unsigned char*)table[expert * 6 + 3] + row * groups;
        g2 = scales[expert * 3 + 0];
        u2 = scales[expert * 3 + 1];
        // The block's panel of both matrices, fetched while the block stages its members.
        if (threadIdx.x == 0) {
            const unsigned long long panel = (unsigned long long)blockIdx.x * rows;
            const long long panel_rows = min((long long)rows, (long long)intermediate - (long long)panel);
            if (panel_rows > 0) {
                for (int matrix = 0; matrix < 4; ++matrix) {
                    const unsigned long long stride = (matrix & 1) ? groups : packed;
                    spark_prefetch_l2((const unsigned char*)table[expert * 6 + matrix] + panel * stride,
                                      panel_rows * stride);
                }
            }
        }
    }
    q4e_e2m1_table_fill(lut);
    q4e_e2m1_table_pair_fill(lut_pair);
    for (int group = 0; group < members; group += Q4E_MEMBER_GROUP) {
        const int count = min(Q4E_MEMBER_GROUP, members - group);
        // The block order comes from q4e_moe_permute: the run's members are consecutive ranks,
        // and each writes the result back to its own (token, slot) row.
        for (int m = 0; m < count; ++m) {
            const int total_slot = perm[begin + group + m];
            const int b = total_slot / slots;
            const float* x_b = x + (unsigned long long)b * hidden;
            float* staged = source + m * Q4E_MAX_ROW_PADDED;
            for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
                staged[q4e_smem_idx(column)] = x_b[column];
            }
        }
        __syncthreads();
        if (expert < 0) {
#pragma unroll
            for (int r = 0; r < Q4E_ROWS_PER_WARP; ++r) {
                const int row = first_row + r;
                if (row < intermediate && lane == 0) {
                    for (int m = 0; m < count; ++m) {
                        const int total_slot = perm[begin + group + m];
                        activated[(unsigned long long)total_slot * intermediate + row] = 0.0f;
                    }
                }
            }
        } else {
#pragma unroll
            for (int r = 0; r < Q4E_ROWS_PER_WARP; ++r) {
                const int row = first_row + r;
                if (r >= valid) {
                    continue;
                }
                float g[Q4E_MEMBER_GROUP];
                float u[Q4E_MEMBER_GROUP];
                q4e_nvfp4_row_dot_pair_group(grow + r * packed, gsrow + r * groups, g2,
                                             urow + r * packed, usrow + r * groups, u2, lut,
                                             lut_pair, source, hidden, lane, count, g, u);
#pragma unroll
                for (int m = 0; m < Q4E_MEMBER_GROUP; ++m) {
                    if (m < count) {
                        const float gate = q4e_warp_sum(g[m]);
                        const float up = q4e_warp_sum(u[m]);
                        if (lane == 0) {
                            const int total_slot = perm[begin + group + m];
                            activated[(unsigned long long)total_slot * intermediate + row] =
                                q4e_silu(gate) * up;
                        }
                    }
                }
            }
        }
        // No thread may still be reading the staging when the next group lands.
        __syncthreads();
    }
}

// Routed experts pass two: down projection with conflict-free shared-memory staging
// (blockIdx.y = slot, blockIdx.z = batch token). One block per expert run, as in pass one.
extern "C" __global__ __launch_bounds__(256, 4) void q4e_expert_down(
    const unsigned long long* __restrict__ table, const float* __restrict__ scales,
    const int* __restrict__ indices, const int* __restrict__ perm,
    const int* __restrict__ run_bounds, const float* __restrict__ weights,
    const float* __restrict__ activated, float* __restrict__ partials, int slots, int hidden,
    int intermediate, int weight_stride) {
    OPT_PDL_ENTRY();
    __shared__ __align__(16) float source[Q4E_MEMBER_GROUP * Q4E_MAX_ROW_PADDED];
    __shared__ float lut[16];
    __shared__ float2 lut_pair[256];
    if (intermediate <= 0 || intermediate > Q4E_MAX_ROW) {
        return;
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int rows = (blockDim.x >> 5) * Q4E_DOWN_ROWS_PER_WARP;
    int first_row = blockIdx.x * rows + warp * Q4E_DOWN_ROWS_PER_WARP;
    if (blockIdx.y >= slots) {
        return;
    }
    // blockIdx.y + blockIdx.z * slots is the rank; the run bounds say where its run starts.
    const int rank = (int)blockIdx.z * slots + (int)blockIdx.y;
    const int begin = run_bounds[2 * rank];
    if (rank != begin) {
        return;
    }
    const int members = run_bounds[2 * rank + 1] - begin + 1;
    const int expert = indices[perm[begin]];
    int groups = (intermediate + 15) / 16;
    int packed = (intermediate + 1) / 2;
    // The warp's first down row and how many of its rows the matrix has.
    const unsigned char* drow = nullptr;
    const unsigned char* dsrow = nullptr;
    const int valid = max(0, min(Q4E_DOWN_ROWS_PER_WARP, hidden - first_row));
    float d2 = 0.0f;
    if (expert >= 0) {
        const unsigned long long row = (unsigned long long)first_row;
        drow = (const unsigned char*)table[expert * 6 + 4] + row * packed;
        dsrow = (const unsigned char*)table[expert * 6 + 5] + row * groups;
        d2 = scales[expert * 3 + 2];
    }
    q4e_e2m1_table_fill(lut);
    q4e_e2m1_table_pair_fill(lut_pair);
    for (int group = 0; group < members; group += Q4E_MEMBER_GROUP) {
        const int count = min(Q4E_MEMBER_GROUP, members - group);
        for (int m = 0; m < count; ++m) {
            const int total_slot = perm[begin + group + m];
            const float* slot_activated =
                activated + (unsigned long long)total_slot * intermediate;
            float* staged = source + m * Q4E_MAX_ROW_PADDED;
            for (int column = threadIdx.x; column < intermediate; column += blockDim.x) {
                staged[q4e_smem_idx(column)] = slot_activated[column];
            }
        }
        __syncthreads();
        float value[Q4E_DOWN_ROWS_PER_WARP][Q4E_MEMBER_GROUP];
        if (expert >= 0) {
            q4e_nvfp4_rows_dot_group<Q4E_DOWN_ROWS_PER_WARP>(drow, dsrow, d2, packed, groups, valid,
                                                             lut, lut_pair, source, intermediate,
                                                             lane, count, value);
        }
#pragma unroll
        for (int r = 0; r < Q4E_DOWN_ROWS_PER_WARP; ++r) {
            const int row = first_row + r;
            if (row >= hidden) {
                continue;
            }
#pragma unroll
            for (int m = 0; m < Q4E_MEMBER_GROUP; ++m) {
                if (m < count) {
                    const float sum = expert >= 0 ? q4e_warp_sum(value[r][m]) : 0.0f;
                    const int total_slot = perm[begin + group + m];
                    if (lane == 0) {
                        const int b = total_slot / slots;
                        const int slot = total_slot % slots;
                        const float w = weights[(unsigned long long)b * weight_stride + slot];
                        partials[(unsigned long long)total_slot * hidden + row] = w * sum;
                    }
                }
            }
        }
        __syncthreads();
    }
}

// The MTP layer's routed experts are 128x128-block-scaled FP8 rather than NVFP4: one byte a
// weight and one BF16 inverse scale per 128x128 tile, with a full-precision activation. The
// passes are otherwise the same shape as the quantised pair above - one block per expert run,
// the run's members staged in shared memory - so the bodies mirror them with a different decode:
// no nibble LUT, no weight_scale_2, and the scale changes once per 128 columns rather than
// twice per 32.
__device__ __forceinline__ void q4e_fp8_chunk_group(const uint4 packed, float scale,
                                                     const float* base, int count, float* out) {
    const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
    float weight[16];
#pragma unroll
    for (int word = 0; word < 4; ++word) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            weight[4 * word + i] =
                q4e_e4m3((unsigned char)((words[word] >> (8 * i)) & 0xFFu)) * scale;
        }
    }
#pragma unroll
    for (int m = 0; m < Q4E_MEMBER_GROUP; ++m) {
        if (m < count) {
            const float* x_chunk = base + m * Q4E_MAX_ROW_PADDED;
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                dot += weight[i] * x_chunk[i];
            }
            out[m] += dot;
        }
    }
}

// One warp's partial dots of a single FP8 row against the staged rows of `count` members.
__device__ __forceinline__ void q4e_fp8_row_dot_group(const unsigned char* row,
                                                       const __nv_bfloat16* scale_row,
                                                       const float* base, int columns, int lane,
                                                       int count, float* out) {
#pragma unroll
    for (int m = 0; m < Q4E_MEMBER_GROUP; ++m) {
        out[m] = 0.0f;
    }
    // A lane owns sixteen consecutive values, so a 128-column scale tile covers whole lanes and
    // every load is 16-byte aligned: no straddle, no masked tail, for either 2560 or 640.
    const int chunks = columns >> 4;
    for (int chunk = lane; chunk < chunks; chunk += 32) {
        const int column = chunk * 16;
        const uint4 packed = q4e_ldg(reinterpret_cast<const uint4*>(row + column));
        const float scale = __bfloat162float(q4e_ldg(&scale_row[column >> 7]));
        q4e_fp8_chunk_group(packed, scale, base + q4e_smem_idx(column), count, out);
    }
}

// Pass one of the MTP routed experts: gate and up, SwiGLU'd, into the slot's activated row.
extern "C" __global__ __launch_bounds__(256, 4) void q4e_expert_gate_up_fp8(
    const unsigned long long* __restrict__ table, const int* __restrict__ indices,
    const int* __restrict__ perm, const int* __restrict__ run_bounds,
    const float* __restrict__ x, float* __restrict__ activated, int slots, int hidden,
    int intermediate) {
    OPT_PDL_ENTRY();
    __shared__ __align__(16) float source[Q4E_MEMBER_GROUP * Q4E_MAX_ROW_PADDED];
    if (hidden <= 0 || hidden > Q4E_MAX_ROW || (hidden & 15) != 0) {
        return;
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int rows = (blockDim.x >> 5) * Q4E_ROWS_PER_WARP;
    int first_row = blockIdx.x * rows + warp * Q4E_ROWS_PER_WARP;
    if (blockIdx.y >= slots) {
        return;
    }
    const int rank = (int)blockIdx.z * slots + (int)blockIdx.y;
    const int begin = run_bounds[2 * rank];
    if (rank != begin) {
        return;
    }
    const int members = run_bounds[2 * rank + 1] - begin + 1;
    const int expert = indices[perm[begin]];
    // 128x128 tiles: the checkpoint's weight_scale_inv is [rows / 128, columns / 128], so one
    // scale row serves 128 weight rows and holds columns / 128 BF16 values.
    const int scale_columns = hidden >> 7;
    const unsigned char* grow[Q4E_ROWS_PER_WARP];
    const __nv_bfloat16* gsrow[Q4E_ROWS_PER_WARP];
    const unsigned char* urow[Q4E_ROWS_PER_WARP];
    const __nv_bfloat16* usrow[Q4E_ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < Q4E_ROWS_PER_WARP; ++r) {
        grow[r] = nullptr;
        gsrow[r] = nullptr;
        urow[r] = nullptr;
        usrow[r] = nullptr;
    }
    if (expert >= 0) {
        const unsigned char* gw = (const unsigned char*)table[expert * 6 + 0];
        const __nv_bfloat16* gs = (const __nv_bfloat16*)table[expert * 6 + 1];
        const unsigned char* uw = (const unsigned char*)table[expert * 6 + 2];
        const __nv_bfloat16* us = (const __nv_bfloat16*)table[expert * 6 + 3];
#pragma unroll
        for (int r = 0; r < Q4E_ROWS_PER_WARP; ++r) {
            const unsigned long long row = (unsigned long long)(first_row + r);
            grow[r] = gw + row * hidden;
            gsrow[r] = gs + (row >> 7) * scale_columns;
            urow[r] = uw + row * hidden;
            usrow[r] = us + (row >> 7) * scale_columns;
        }
    }
    for (int group = 0; group < members; group += Q4E_MEMBER_GROUP) {
        const int count = min(Q4E_MEMBER_GROUP, members - group);
        for (int m = 0; m < count; ++m) {
            const int total_slot = perm[begin + group + m];
            const int b = total_slot / slots;
            const float* x_b = x + (unsigned long long)b * hidden;
            float* staged = source + m * Q4E_MAX_ROW_PADDED;
            for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
                staged[q4e_smem_idx(column)] = x_b[column];
            }
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < Q4E_ROWS_PER_WARP; ++r) {
            const int row = first_row + r;
            if (row >= intermediate) {
                continue;
            }
            if (expert < 0) {
                if (lane == 0) {
                    for (int m = 0; m < count; ++m) {
                        const int total_slot = perm[begin + group + m];
                        activated[(unsigned long long)total_slot * intermediate + row] = 0.0f;
                    }
                }
            } else {
                float g[Q4E_MEMBER_GROUP];
                float u[Q4E_MEMBER_GROUP];
                q4e_fp8_row_dot_group(grow[r], gsrow[r], source, hidden, lane, count, g);
                q4e_fp8_row_dot_group(urow[r], usrow[r], source, hidden, lane, count, u);
#pragma unroll
                for (int m = 0; m < Q4E_MEMBER_GROUP; ++m) {
                    if (m < count) {
                        const float gate = q4e_warp_sum(g[m]);
                        const float up = q4e_warp_sum(u[m]);
                        if (lane == 0) {
                            const int total_slot = perm[begin + group + m];
                            activated[(unsigned long long)total_slot * intermediate + row] =
                                q4e_silu(gate) * up;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

// Pass two of the MTP routed experts: the down projection, scattered by the router weight.
extern "C" __global__ __launch_bounds__(256, 4) void q4e_expert_down_fp8(
    const unsigned long long* __restrict__ table, const int* __restrict__ indices,
    const int* __restrict__ perm, const int* __restrict__ run_bounds,
    const float* __restrict__ weights, const float* __restrict__ activated,
    float* __restrict__ partials, int slots, int hidden, int intermediate, int weight_stride) {
    OPT_PDL_ENTRY();
    __shared__ __align__(16) float source[Q4E_MEMBER_GROUP * Q4E_MAX_ROW_PADDED];
    if (intermediate <= 0 || intermediate > Q4E_MAX_ROW || (intermediate & 15) != 0) {
        return;
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int rows = (blockDim.x >> 5) * Q4E_ROWS_PER_WARP;
    int first_row = blockIdx.x * rows + warp * Q4E_ROWS_PER_WARP;
    if (blockIdx.y >= slots) {
        return;
    }
    const int rank = (int)blockIdx.z * slots + (int)blockIdx.y;
    const int begin = run_bounds[2 * rank];
    if (rank != begin) {
        return;
    }
    const int members = run_bounds[2 * rank + 1] - begin + 1;
    const int expert = indices[perm[begin]];
    const int scale_columns = intermediate >> 7;
    const unsigned char* drow[Q4E_ROWS_PER_WARP];
    const __nv_bfloat16* dsrow[Q4E_ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < Q4E_ROWS_PER_WARP; ++r) {
        drow[r] = nullptr;
        dsrow[r] = nullptr;
    }
    if (expert >= 0) {
        const unsigned char* dw = (const unsigned char*)table[expert * 6 + 4];
        const __nv_bfloat16* ds = (const __nv_bfloat16*)table[expert * 6 + 5];
#pragma unroll
        for (int r = 0; r < Q4E_ROWS_PER_WARP; ++r) {
            const unsigned long long row = (unsigned long long)(first_row + r);
            drow[r] = dw + row * intermediate;
            dsrow[r] = ds + (row >> 7) * scale_columns;
        }
    }
    for (int group = 0; group < members; group += Q4E_MEMBER_GROUP) {
        const int count = min(Q4E_MEMBER_GROUP, members - group);
        for (int m = 0; m < count; ++m) {
            const int total_slot = perm[begin + group + m];
            const float* slot_activated = activated + (unsigned long long)total_slot * intermediate;
            float* staged = source + m * Q4E_MAX_ROW_PADDED;
            for (int column = threadIdx.x; column < intermediate; column += blockDim.x) {
                staged[q4e_smem_idx(column)] = slot_activated[column];
            }
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < Q4E_ROWS_PER_WARP; ++r) {
            const int row = first_row + r;
            if (row >= hidden) {
                continue;
            }
            float value[Q4E_MEMBER_GROUP];
            if (expert >= 0) {
                q4e_fp8_row_dot_group(drow[r], dsrow[r], source, intermediate, lane, count, value);
#pragma unroll
                for (int m = 0; m < Q4E_MEMBER_GROUP; ++m) {
                    if (m < count) {
                        value[m] = q4e_warp_sum(value[m]);
                    }
                }
            } else {
                for (int m = 0; m < count; ++m) {
                    value[m] = 0.0f;
                }
            }
#pragma unroll
            for (int m = 0; m < Q4E_MEMBER_GROUP; ++m) {
                if (m < count) {
                    const int total_slot = perm[begin + group + m];
                    if (lane == 0) {
                        const int b = total_slot / slots;
                        const int slot = total_slot % slots;
                        const float w = weights[(unsigned long long)b * weight_stride + slot];
                        partials[(unsigned long long)total_slot * hidden + row] = w * value[m];
                    }
                }
            }
        }
        __syncthreads();
    }
}

// The MTP head's fold of the widened residual: the mean of the normed streams, one row a token.
extern "C" __global__ void q4e_fold_streams(const float* __restrict__ normed,
                                             float* __restrict__ folded, int hidden, int hc,
                                             int batch, float scale) {
    OPT_PDL_ENTRY();
    const int total = batch * hidden;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
         index += blockDim.x * gridDim.x) {
        const int t = index / hidden;
        const int column = index - t * hidden;
        const unsigned long long base = (unsigned long long)t * hc * hidden + column;
        float accumulator = 0.0f;
        for (int stream = 0; stream < hc; ++stream) {
            accumulator += normed[base + (unsigned long long)stream * hidden];
        }
        folded[index] = accumulator * scale;
    }
}

// Widen one collapsed row back onto the model's stream count, one block group a token's row.
extern "C" __global__ void q4e_widen_streams(const float* __restrict__ collapsed,
                                              float* __restrict__ widened, int hidden, int hc) {
    OPT_PDL_ENTRY();
    const int b = blockIdx.y;
    const float* source = collapsed + (unsigned long long)b * hidden;
    float* destination = widened + (unsigned long long)b * hc * hidden;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < hc * hidden;
         index += blockDim.x * gridDim.x) {
        destination[index] = source[index % hidden];
    }
}

// Sum the slots routed-expert partials and add the sigmoid-gated shared expert, plus optional
// fused residual stream injection when hyper is non-null (blockIdx.y = batch token). A null
// shared_gate adds the shared expert ungated, GLM-5.3-Flash's form.
extern "C" __global__ void q4e_moe_combine(
    float* __restrict__ out, const float* __restrict__ partials, const float* __restrict__ shared,
    const float* __restrict__ shared_gate, float* __restrict__ hyper,
    const float* __restrict__ injection, int slots, int hidden, int hc) {
    OPT_PDL_ENTRY();
    int b = blockIdx.y;
    const float* partials_b = partials + (unsigned long long)b * slots * hidden;
    const float* shared_b = shared + (unsigned long long)b * hidden;
    float* out_b = out + (unsigned long long)b * hidden;
    float gate = shared_gate != nullptr ? 1.0f / (1.0f + __expf(-shared_gate[b])) : 1.0f;
    float* hyper_b = hyper != nullptr ? (hyper + (unsigned long long)b * hc * hidden) : nullptr;
    const float* injection_b =
        injection != nullptr ? (injection + (unsigned long long)b * hc) : nullptr;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < hidden;
         index += blockDim.x * gridDim.x) {
        float expert = 0.0f;
        for (int slot = 0; slot < slots; ++slot) {
            expert += partials_b[(unsigned long long)slot * hidden + index];
        }
        float combined = expert + gate * shared_b[index];
        out_b[index] = combined;
        if (hyper_b != nullptr && injection_b != nullptr) {
            for (int s = 0; s < hc; ++s) {
                hyper_b[s * hidden + index] += combined * injection_b[s];
            }
        }
    }
}

// Causal depthwise conv1d of width taps with SiLU activation, updating the history state in
// place across batch token steps.
extern "C" __global__ void q4e_gdn_conv(const float* __restrict__ raw, float* __restrict__ state,
                                        const __nv_bfloat16* __restrict__ weight,
                                        float* __restrict__ out, int width, int taps, int batch,
                                        float* __restrict__ snapshot, int snapshot_rows,
                                        long long region_stride) {
    OPT_PDL_ENTRY();
    int history = taps - 1;
    for (int column = blockIdx.x * blockDim.x + threadIdx.x; column < width;
         column += blockDim.x * gridDim.x) {
        for (int t = 0; t < batch; ++t) {
            float src = raw[(unsigned long long)t * width + column];
            float accumulator = 0.0f;
            for (int tap = 0; tap < taps; ++tap) {
                int back = history - tap;
                float source = (back == 0) ? src : state[(history - back) * width + column];
                accumulator += __bfloat162float(weight[column * taps + tap]) * source;
            }
            out[(unsigned long long)t * width + column] = q4e_silu(accumulator);
            for (int row = 0; row < history - 1; ++row) {
                state[row * width + column] = state[(row + 1) * width + column];
            }
            if (history > 0) {
                state[(history - 1) * width + column] = src;
            }
            // A speculative verify has to give back the rows past the ones the model agreed with,
            // so the state after each of its first rows is copied out while it is still live.
            if (snapshot != nullptr && t < snapshot_rows) {
                float* into = snapshot + (unsigned long long)t * region_stride;
                for (int row = 0; row < history; ++row) {
                    into[row * width + column] = state[row * width + column];
                }
            }
        }
    }
}

// Copies a head's `cells` floats of recurrent state with the whole block, sixteen bytes a thread
// and eight loads in flight. The scalar loop this replaces kept one four-byte load outstanding
// per thread, and with one block per head the copy in and out was most of a one-token step.
__device__ __forceinline__ void q4e_copy_state(float* __restrict__ to,
                                               const float* __restrict__ from, int cells) {
    int start = 0;
    if (((reinterpret_cast<unsigned long long>(to) | reinterpret_cast<unsigned long long>(from)) &
         15u) == 0) {
        const int vectors = cells >> 2;
        float4* to4 = reinterpret_cast<float4*>(to);
        const float4* from4 = reinterpret_cast<const float4*>(from);
        const int stride = blockDim.x;
        int index = threadIdx.x;
        for (; index + 7 * stride < vectors; index += 8 * stride) {
            float4 held[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                held[u] = from4[index + u * stride];
            }
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                to4[index + u * stride] = held[u];
            }
        }
        for (; index < vectors; index += stride) {
            to4[index] = from4[index];
        }
        start = vectors << 2;
    }
    for (int index = start + threadIdx.x; index < cells; index += blockDim.x) {
        to[index] = from[index];
    }
}

// q4e_copy_state for a slice of `width` value columns of a head's `rows` key rows: row r's floats
// go from `from + r * from_pitch` to `to + r * to_pitch`, sixteen bytes a thread and eight loads in
// flight. A slice as wide as both pitches is the whole state, which q4e_copy_state copies. Both
// slices are 16-byte aligned and width is a multiple of 4.
__device__ __forceinline__ void q4e_copy_columns(float* __restrict__ to, int to_pitch,
                                                 const float* __restrict__ from, int from_pitch,
                                                 int rows, int width) {
    if (to_pitch == width && from_pitch == width) {
        q4e_copy_state(to, from, rows * width);
        return;
    }
    const int quads = width >> 2;
    const int vectors = rows * quads;
    const int stride = blockDim.x;
    const auto source = [&](int vector) {
        return reinterpret_cast<const float4*>(from + (vector / quads) * from_pitch) +
               vector % quads;
    };
    const auto target = [&](int vector) {
        return reinterpret_cast<float4*>(to + (vector / quads) * to_pitch) + vector % quads;
    };
    int index = threadIdx.x;
    for (; index + 7 * stride < vectors; index += 8 * stride) {
        float4 held[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            held[u] = *source(index + u * stride);
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            *target(index + u * stride) = held[u];
        }
    }
    for (; index < vectors; index += stride) {
        *target(index) = *source(index);
    }
}

// One token's delta-rule step for this thread's value columns of a head: decay the state, recall
// what k reads from it, write beta * (v - recall) along k, and read the result with q into
// head_out. With kChannel the decay is one factor a key channel (`channel`, Kimi Delta Attention);
// without it, one factor for the head (Gated DeltaNet), in exactly the arithmetic it had before
// the channel form existed. The block holds `width` of the head's value columns, key row k's at
// `local + k * pitch`; `v` and `head_out` start at the first of them. Each column is its own
// chain, so a slice of the columns computes exactly what the whole head computes for them.
template <bool kChannel>
__device__ __forceinline__ void q4e_gdn_step(float* local, int pitch, const float* sq,
                                             const float* sk, const float* channel,
                                             float head_factor, float q_scale, float k_scale,
                                             const float* __restrict__ v, float beta,
                                             float* head_out, int key_dim, int width) {
    for (int value_index = threadIdx.x; value_index < width; value_index += blockDim.x) {
        float memory = 0.0f;
#pragma unroll 4
        for (int key_index = 0; key_index < key_dim; ++key_index) {
            float c = local[key_index * pitch + value_index] *
                      (kChannel ? channel[key_index] : head_factor);
            memory += c * sk[key_index] * k_scale;
        }
        float delta = (v[value_index] - memory) * beta;
        float output = 0.0f;
#pragma unroll 4
        for (int key_index = 0; key_index < key_dim; ++key_index) {
            float* cell = &local[key_index * pitch + value_index];
            float c = (*cell) * (kChannel ? channel[key_index] : head_factor);
            c += sk[key_index] * k_scale * delta;
            *cell = c;
            output += c * sq[key_index] * q_scale;
        }
        head_out[value_index] = output;
    }
}

// Kimi Delta Attention's decay factors, one a key channel:
// exp(lower_bound * sigmoid(exp(a_log[head]) * (forget + dt_bias))), as
// linear_attention::kda_channel_decay computes them with linalg's sigmoid. `forget` and `decay`
// are [tokens][heads * key_dim]; `a_log` is one f32 a head and `dt_bias` one a channel, as
// GLM-5.3-Flash stores them. The factors are q4e_gdn_recur's `channel_decay`.
extern "C" __global__ void q4e_kda_decay(const float* __restrict__ forget,
                                         const float* __restrict__ a_log,
                                         const float* __restrict__ dt_bias,
                                         float* __restrict__ decay, int heads, int key_dim,
                                         float lower_bound, int tokens) {
    OPT_PDL_ENTRY();
    const int width = heads * key_dim;
    const unsigned long long total = (unsigned long long)tokens * width;
    for (unsigned long long index = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
         index < total; index += (unsigned long long)blockDim.x * gridDim.x) {
        const int channel = (int)(index % width);
        const float x = expf(a_log[channel / key_dim]) * (forget[index] + dt_bias[channel]);
        decay[index] = expf(lower_bound * q4e_sigmoid_ref(x));
    }
}

// Gated DeltaNet recurrence, one block per value head, stepping across batch tokens.
//
// The head's state is 128x128 floats, and every token reads it once to form the delta, then
// reads it again (after the decay) to update it and project the output: three 64 KiB trips
// through a 1 MiB-per-layer state for every token. With `stage_bytes` set the block copies its
// head's slice into dynamic shared memory once, runs the whole chunk against it, and writes it
// back once - the traffic is per chunk instead of per token. The host sizes the reservation and
// passes 0 when the state does not fit.
//
// The block reads its head's state from `state` and leaves the state after the chunk's last row
// in `state_out`: the same buffer for a decode step or a prefill, the other buffer of a pair for a
// speculative verify, which leaves the state it read in place for a step that keeps only some of
// its rows. Those rows are always a prefix, and replaying the prefix through this same code from
// the same state is bit for bit the state they left behind; the replay needs no output, so it
// passes `out` as nullptr. Where a snapshot after every row was 3 MiB a layer a row of writes, a
// verify whose rows are all kept writes the state once, and one that keeps fewer replays them.
// Without the staged reservation the block works in `state_out` itself.
//
// With `channel_decay` set ([batch][heads][key_dim] factors, q4e_kda_decay's) the state decays one
// factor a key channel, Kimi Delta Attention's form, and `a`, `a_log` and `dt_bias` are not read;
// nullptr keeps Gated DeltaNet's one factor a head from them.
//
// A launch stages its rows' q, k and decay a chunk at a time, and one warp a row takes the row's
// scales with q4e_block_sum's tree (q4e_warp_square_sum), so no row waits on a block reduction or
// a global load of its own: each value column's chain runs through the chunk's rows back to back.
// With `out` the rows' raw outputs go to `out` and are gated there, a warp a row, after the last.
//
// q4e_gdn_recur_columns splits each head's value columns over `parts` blocks (grid (heads,
// parts)): every column's chain reads only its own column of the state, so a block that holds a
// slice computes those columns exactly as the whole head does, and a head's state moves through
// `parts` SMs instead of one. The gate needs all of a head's columns, so the head's last block to
// finish (its `counters` entry, zero between launches) gates every row, with the same tree, and
// `out` is bit for bit q4e_gdn_recur's.
#define Q4E_RECUR_CHUNK 8

// q4e_block_sum's total over a block of blockDim.x threads (a power of two from 32 to Q4E_BLOCK),
// taken by one warp: lane l holds the partials of threads l, l + 32, ... in `partial`, `per_lane`
// of them. The levels above a warp add them pairwise as the block's strides do and q4e_warp_sum
// adds the last five as its threads do, so every lane returns the block's bits.
__device__ __forceinline__ float q4e_warp_block_sum(float (&partial)[Q4E_BLOCK / 32],
                                                    int per_lane) {
#pragma unroll
    for (int half = Q4E_BLOCK / 64; half > 0; half >>= 1) {
        if (half < per_lane) {
#pragma unroll
            for (int j = 0; j < half; ++j) {
                partial[j] += partial[j + half];
            }
        }
    }
    return __shfl_sync(0xFFFFFFFFu, q4e_warp_sum(partial[0]), 0);
}

// The bits q4e_block_sum gives for the sum of squares of `count` values, thread t of the block
// adding values t, t + blockDim.x, ... in order, taken by one warp; `load(i)` reads value i.
template <typename Load>
__device__ __forceinline__ float q4e_warp_square_sum(Load load, int count) {
    float partial[Q4E_BLOCK / 32];
    const int lane = threadIdx.x & 31;
    const int per_lane = blockDim.x >> 5;
#pragma unroll
    for (int j = 0; j < Q4E_BLOCK / 32; ++j) {
        float sum = 0.0f;
        if (j < per_lane) {
            for (int index = lane + 32 * j; index < count; index += blockDim.x) {
                float value = load(index);
                sum += value * value;
            }
        }
        partial[j] = sum;
    }
    return q4e_warp_block_sum(partial, per_lane);
}

// One row's gated RMSNorm of a head's outputs, by one warp, in place: `out_t` holds the raw
// outputs, written by this block or, behind a fence, another, and takes the gated ones.
__device__ __forceinline__ void q4e_gdn_gate_row(float* out_t, const float* __restrict__ z_t,
                                                 const __nv_bfloat16* __restrict__ norm,
                                                 int value_dim, float eps, int sigmoid_gate) {
    const float sum = q4e_warp_square_sum([&](int index) { return __ldcg(out_t + index); },
                                          value_dim);
    float out_scale = rsqrtf(sum / (float)value_dim + eps);
    for (int value_index = threadIdx.x & 31; value_index < value_dim; value_index += 32) {
        float normed = __ldcg(out_t + value_index) * out_scale * __bfloat162float(norm[value_index]);
        float gated = z_t[value_index];
        float activated = sigmoid_gate ? (1.0f / (1.0f + __expf(-gated))) : q4e_silu(gated);
        out_t[value_index] = normed * activated;
    }
}

__device__ __forceinline__ void q4e_gdn_recur_body(
    const float* __restrict__ mixed, const float* __restrict__ z, const float* __restrict__ b,
    const float* __restrict__ a, const __nv_bfloat16* __restrict__ a_log,
    const __nv_bfloat16* __restrict__ dt_bias, const __nv_bfloat16* __restrict__ norm,
    const float* state, float* state_out, float* __restrict__ out, int key_width, int heads,
    int key_dim, int value_dim, int groups, float eps, int sigmoid_gate, int batch,
    int stage_bytes, const float* __restrict__ channel_decay, int parts, int part,
    unsigned int* __restrict__ counters) {
    __shared__ float head_out[Q4E_MAX_INTERMEDIATE];
    __shared__ float sq[Q4E_RECUR_CHUNK * 256];
    __shared__ float sk[Q4E_RECUR_CHUNK * 256];
    __shared__ float sd[Q4E_RECUR_CHUNK * 256];
    __shared__ float q_scales[Q4E_RECUR_CHUNK];
    __shared__ float k_scales[Q4E_RECUR_CHUNK];
    __shared__ float betas[Q4E_RECUR_CHUNK];
    __shared__ float factors[Q4E_RECUR_CHUNK];
    __shared__ int last;
    extern __shared__ float dynamic[];
    const int threads = blockDim.x;
    if (value_dim <= 0 || value_dim > Q4E_MAX_INTERMEDIATE || key_dim <= 0 || key_dim > 256 ||
        groups <= 0 || threads < 32 || threads > Q4E_BLOCK || (threads & (threads - 1)) != 0 ||
        parts <= 0 || value_dim % parts != 0 ||
        (parts > 1 && ((value_dim / parts) % 4 != 0 || (out != nullptr && counters == nullptr)))) {
        return;
    }
    int head = blockIdx.x;
    int key_head = head / groups;
    int conv_width = key_width * 2 + heads * value_dim;
    int val_width = heads * value_dim;
    const int width = value_dim / parts;
    const int first = part * width;
    const unsigned long long slice = (unsigned long long)head * key_dim * value_dim + first;
    const float* recurrent = state + slice;
    float* recurrent_out = state_out + slice;
    const int staged = stage_bytes >= key_dim * width * (int)sizeof(float);
    float* local = staged ? dynamic : recurrent_out;
    const int pitch = staged ? width : value_dim;
    // The state's copy and the first chunk's loads are in flight together; the chunk's barrier
    // is the copy's.
    if (staged) {
        q4e_copy_columns(dynamic, width, recurrent, value_dim, key_dim, width);
    } else if (recurrent_out != recurrent) {
        q4e_copy_columns(recurrent_out, value_dim, recurrent, value_dim, key_dim, width);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = threads >> 5;

    for (int t0 = 0; t0 < batch; t0 += Q4E_RECUR_CHUNK) {
        const int rows = min(Q4E_RECUR_CHUNK, batch - t0);
        if (t0 > 0) {
            // The last chunk's steps have read its rows.
            __syncthreads();
        }
        for (int index = threadIdx.x; index < rows * key_dim; index += threads) {
            const int t = index / key_dim;
            const int key = index - t * key_dim;
            const float* mixed_t = mixed + (unsigned long long)(t0 + t) * conv_width;
            sq[index] = mixed_t[key_head * key_dim + key];
            sk[index] = mixed_t[key_width + key_head * key_dim + key];
            if (channel_decay != nullptr) {
                sd[index] =
                    channel_decay[((unsigned long long)(t0 + t) * heads + head) * key_dim + key];
            }
        }
        __syncthreads();
        for (int t = warp; t < rows; t += warps) {
            const unsigned long long token = t0 + t;
            const float q_sum =
                q4e_warp_square_sum([&](int key) { return sq[t * key_dim + key]; }, key_dim);
            const float k_sum =
                q4e_warp_square_sum([&](int key) { return sk[t * key_dim + key]; }, key_dim);
            if (lane == 0) {
                q_scales[t] = rsqrtf(q_sum + 1e-6f) * rsqrtf((float)key_dim);
                k_scales[t] = rsqrtf(k_sum + 1e-6f);
                betas[t] = 1.0f / (1.0f + __expf(-b[token * heads + head]));
                if (channel_decay == nullptr) {
                    float decay = -__expf(__bfloat162float(a_log[head])) *
                                  logf(1.0f + __expf(a[token * heads + head] +
                                                     __bfloat162float(dt_bias[head])));
                    factors[t] = __expf(decay);
                }
            }
        }
        __syncthreads();
        for (int t = 0; t < rows; ++t) {
            const unsigned long long token = t0 + t;
            const float* v = mixed + token * conv_width + key_width * 2 + head * value_dim + first;
            float* into =
                out == nullptr ? head_out : out + token * val_width + head * value_dim + first;
            if (channel_decay != nullptr) {
                q4e_gdn_step<true>(local, pitch, sq + t * key_dim, sk + t * key_dim,
                                   sd + t * key_dim, 0.0f, q_scales[t], k_scales[t], v, betas[t],
                                   into, key_dim, width);
            } else {
                q4e_gdn_step<false>(local, pitch, sq + t * key_dim, sk + t * key_dim, nullptr,
                                    factors[t], q_scales[t], k_scales[t], v, betas[t], into,
                                    key_dim, width);
            }
        }
    }
    __syncthreads();
    if (staged) {
        q4e_copy_columns(recurrent_out, value_dim, dynamic, width, key_dim, width);
    }
    if (out == nullptr) {
        return;
    }
    if (parts > 1) {
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            last = atomicAdd(&counters[head], 1u) == (unsigned int)(parts - 1);
        }
        __syncthreads();
        if (!last) {
            return;
        }
        __threadfence();
    }
    for (int t = warp; t < batch; t += warps) {
        const unsigned long long row = (unsigned long long)t * val_width + head * value_dim;
        q4e_gdn_gate_row(out + row, z + row, norm, value_dim, eps, sigmoid_gate);
    }
    if (parts > 1 && threadIdx.x == 0) {
        counters[head] = 0u;
    }
}

extern "C" __global__ void q4e_gdn_recur(
    const float* __restrict__ mixed, const float* __restrict__ z, const float* __restrict__ b,
    const float* __restrict__ a, const __nv_bfloat16* __restrict__ a_log,
    const __nv_bfloat16* __restrict__ dt_bias, const __nv_bfloat16* __restrict__ norm,
    // `state` and `state_out` are the same buffer for a decode step, so neither is restrict.
    const float* state, float* state_out, float* __restrict__ out, int key_width, int heads,
    int key_dim, int value_dim, int groups, float eps, int sigmoid_gate, int batch,
    int stage_bytes, const float* __restrict__ channel_decay) {
    OPT_PDL_ENTRY();
    q4e_gdn_recur_body(mixed, z, b, a, a_log, dt_bias, norm, state, state_out, out, key_width,
                       heads, key_dim, value_dim, groups, eps, sigmoid_gate, batch, stage_bytes,
                       channel_decay, 1, 0, nullptr);
}

// q4e_gdn_recur over grid (heads, parts), `stage_bytes` sized for one block's key_dim x
// value_dim / parts slice; value_dim / parts is a multiple of 4.
extern "C" __global__ void q4e_gdn_recur_columns(
    const float* __restrict__ mixed, const float* __restrict__ z, const float* __restrict__ b,
    const float* __restrict__ a, const __nv_bfloat16* __restrict__ a_log,
    const __nv_bfloat16* __restrict__ dt_bias, const __nv_bfloat16* __restrict__ norm,
    const float* state, float* state_out, float* __restrict__ out, int key_width, int heads,
    int key_dim, int value_dim, int groups, float eps, int sigmoid_gate, int batch,
    int stage_bytes, const float* __restrict__ channel_decay, unsigned int* __restrict__ counters) {
    OPT_PDL_ENTRY();
    q4e_gdn_recur_body(mixed, z, b, a, a_log, dt_bias, norm, state, state_out, out, key_width,
                       heads, key_dim, value_dim, groups, eps, sigmoid_gate, batch, stage_bytes,
                       channel_decay, gridDim.y, blockIdx.y, counters);
}

// Rotary embedding over a whole chunk: blockIdx.x is the head inside a row, blockIdx.y is the
// token. A row is token_stride floats apart, so the same kernel rotates a [tokens][heads][dim]
// projection, a [tokens][kv_heads][dim] KV staging row, or a [tokens][kv_width] slice of the KV
// cache itself (where successive tokens are exactly one kv_width apart). The absolute position of
// token t is position + t.
extern "C" __global__ void q4e_rope(float* rows, int head_dim, int rotary, int position, float theta,
                                    int token_stride, int tokens) {
    OPT_PDL_ENTRY();
    int head = blockIdx.x;
    int token = blockIdx.y;
    if (token >= tokens) {
        return;
    }
    float* base = rows + (unsigned long long)token * token_stride + head * head_dim;
    const float position_f = (float)(position + token);
    // A rotary wider than the head is the indexer's case in some fixtures: clamp the pair width
    // to the head so the upper half never names the next head's first element.
    const int half = (rotary < head_dim ? rotary : head_dim) / 2;
    for (int index = threadIdx.x; index < half; index += blockDim.x) {
        float frequency = powf(theta, -2.0f * (float)index / (float)rotary);
        float angle = position_f * frequency;
        float cosine = __cosf(angle);
        float sine = __sinf(angle);
        float a = base[index];
        float b = base[index + half];
        base[index] = a * cosine - b * sine;
        base[index + half] = a * sine + b * cosine;
    }
}

#define Q4E_MAX_BUDGET 1024

// One indexer block's pooled key: the mean of its ratio keys, RMS-normed with k_weight and
// rotated at the block's first position. It depends on the block's keys alone, so a forward pools
// each block once for every row that scores it, where the score kernel used to pool a block again
// for each row. blockIdx.x is the block; launch 128 threads.
extern "C" __global__ void q4e_qsa_pool(const __nv_bfloat16* index_keys,
                                        const __nv_bfloat16* k_weight, float* pooled_keys,
                                        int index_dim, int ratio, int rotary, float theta,
                                        float eps) {
    OPT_PDL_ENTRY();
    const int block = blockIdx.x;
    if (index_dim > 128 || blockDim.x < 128) {
        return;
    }
    __shared__ float pooled[128];
    __shared__ float scratch[128];
    const int dim = threadIdx.x;
    const bool live = dim < index_dim;
    float value = 0.0f;
    if (live) {
        float sum = 0.0f;
        for (int token = 0; token < ratio; ++token) {
            sum += __bfloat162float(
                index_keys[(unsigned long long)(block * ratio + token) * index_dim + dim]);
        }
        value = sum / (float)ratio;
    }
    const float scale = rsqrtf(q4e_block_sum(value * value, scratch) / (float)index_dim + eps);
    if (live) {
        pooled[dim] = value * scale * (1.0f + __bfloat162float(k_weight[dim]));
    }
    __syncthreads();
    const int half = rotary < index_dim ? rotary / 2 : index_dim / 2;
    if (dim < half) {
        const float frequency = powf(theta, -2.0f * (float)dim / (float)rotary);
        const float angle = (float)(block * ratio) * frequency;
        const float cosine = __cosf(angle);
        const float sine = __sinf(angle);
        const float a = pooled[dim];
        const float b = pooled[dim + half];
        pooled[dim] = a * cosine - b * sine;
        pooled[dim + half] = a * sine + b * cosine;
    }
    __syncthreads();
    if (live) {
        pooled_keys[(unsigned long long)block * index_dim + dim] = pooled[dim];
    }
}

#define Q4E_QSA_TILE 64
#define Q4E_QSA_MAX_HEADS 8

// The indexer's per-block score over the pooled keys: gridDim.x covers the compressed blocks of
// the chunk's last token Q4E_QSA_TILE at a time, gridDim.y the tokens of the chunk. A token's
// score row is score_stride floats into `scores`, so a chunk keeps one row per token.
//
// A warp scores a block, and its sum takes the order the 128-thread block tree took before it
// (channel t + 64 into t, t + 32 into t, then halving), so the scores are bit-identical. The
// products are rounded on their own (__fmul_rn), as they were when the tree read them from
// shared memory, rather than fused into the first adds.
extern "C" __global__ void q4e_qsa_scores(const float* __restrict__ index_q,
                                          const float* __restrict__ pooled_keys,
                                          float* __restrict__ scores, int position,
                                          int index_heads, int index_dim, int ratio,
                                          int score_stride) {
    OPT_PDL_ENTRY();
    const int token_index = blockIdx.y;
    const int blocks = (position + token_index + 1) / ratio;
    const int first = blockIdx.x * Q4E_QSA_TILE;
    if (first >= blocks || index_dim > 128 || index_heads > Q4E_QSA_MAX_HEADS) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const float* index_q_t = index_q + (unsigned long long)token_index * index_heads * index_dim;
    float* scores_t = scores + (unsigned long long)token_index * score_stride;
    float query[Q4E_QSA_MAX_HEADS][4];
#pragma unroll
    for (int head = 0; head < Q4E_QSA_MAX_HEADS; ++head) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            const int dim = lane + 32 * step;
            query[head][step] =
                head < index_heads && dim < index_dim ? index_q_t[head * index_dim + dim] : 0.0f;
        }
    }
    const int last = first + Q4E_QSA_TILE < blocks ? first + Q4E_QSA_TILE : blocks;
    for (int block = first + warp; block < last; block += warps) {
        const float* key_row = pooled_keys + (unsigned long long)block * index_dim;
        float key[4];
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            const int dim = lane + 32 * step;
            key[step] = dim < index_dim ? key_row[dim] : 0.0f;
        }
        float score = 0.0f;
#pragma unroll
        for (int head = 0; head < Q4E_QSA_MAX_HEADS; ++head) {
            if (head < index_heads) {
                const float low = __fadd_rn(__fmul_rn(query[head][0], key[0]),
                                            __fmul_rn(query[head][2], key[2]));
                const float high = __fadd_rn(__fmul_rn(query[head][1], key[1]),
                                             __fmul_rn(query[head][3], key[3]));
                score += fmaxf(q4e_warp_sum(__fadd_rn(low, high)), 0.0f);
            }
        }
        if (lane == 0) {
            scores_t[block] = score / sqrtf((float)index_dim);
        }
    }
}

// A score as an unsigned key in the float order: -0 keys as +0, as the float compare has them
// equal, and negative scores below every positive one.
__device__ __forceinline__ unsigned int q4e_score_key(float score) {
    unsigned int bits = __float_as_uint(score);
    if (bits == 0x80000000u) {
        bits = 0u;
    }
    return (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u;
}

// Exclusive prefix sum of one int a thread over the block, with the block's total in *total;
// scratch holds 32 ints. Reusable back to back.
__device__ int q4e_block_scan(int value, int* scratch, int* total) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = (blockDim.x + 31) >> 5;
    int inclusive = value;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xFFFFFFFFu, inclusive, offset);
        if (lane >= offset) {
            inclusive += other;
        }
    }
    if (lane == 31) {
        scratch[warp] = inclusive;
    }
    __syncthreads();
    if (warp == 0) {
        int running = lane < warps ? scratch[lane] : 0;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int other = __shfl_up_sync(0xFFFFFFFFu, running, offset);
            if (lane >= offset) {
                running += other;
            }
        }
        scratch[lane] = running;
    }
    __syncthreads();
    const int before = (warp > 0 ? scratch[warp - 1] : 0) + inclusive - value;
    *total = scratch[warps - 1];
    __syncthreads();
    return before;
}

// The top-budget selection and the emission, from the scores above.
//
// blockIdx.x is the token of the chunk, so one launch does the whole chunk. The kept set is the
// budget best blocks, a higher score first and on a tie the lower block: what a heap fed the
// blocks in order keeps, since a later block never displaces an equal one. A radix select over
// the scores' keys finds the budget-th best key a byte at a time, so the set takes four passes
// over the scores instead of a single thread's walk through the heap (300 ms of a 4,096-row
// forward at 8k context). Each thread then walks a contiguous run of blocks, so the kept blocks
// are emitted ascending with two block scans and no sort.
//
// `tail_alone` is whether a token that keeps no block still attends over its tail: Qwen's
// reference attends over nothing before the first whole block, GLM-5.3-Flash's over the tail.
__device__ __forceinline__ void q4e_pick_pools(const float* scores, int* selected, int* count,
                                               int position, int ratio, int budget,
                                               int score_stride, int select_stride,
                                               bool tail_alone) {
    __shared__ unsigned int histogram[256];
    __shared__ int scan_scratch[32];
    __shared__ unsigned int chosen_prefix;
    __shared__ int chosen_remaining;
    const int token_index = blockIdx.x;
    const int token_position = position + token_index;
    const float* scores_t = scores + (unsigned long long)token_index * score_stride;
    int* selected_t = selected + (unsigned long long)token_index * select_stride;
    const int blocks = (token_position + 1) / ratio;
    const int thread = threadIdx.x;
    int keep = budget < blocks ? budget : blocks;
    if (keep > Q4E_MAX_BUDGET) {
        keep = Q4E_MAX_BUDGET;
    }
    if (keep <= 0) {
        const int tail = tail_alone ? token_position + 1 - blocks * ratio : 0;
        for (int index = thread; index < tail; index += blockDim.x) {
            selected_t[index] = blocks * ratio + index;
        }
        if (thread == 0) {
            count[token_index] = tail;
        }
        return;
    }
    // Every block past the budget keeps all; otherwise the keys above `threshold` and the first
    // `ties` keys equal to it, in block order.
    const bool all = blocks <= keep;
    unsigned int threshold = 0u;
    int ties = 0;
    if (!all) {
        unsigned int prefix = 0u;
        unsigned int mask = 0u;
        int remaining = keep;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int bin = thread; bin < 256; bin += blockDim.x) {
                histogram[bin] = 0u;
            }
            __syncthreads();
            for (int block = thread; block < blocks; block += blockDim.x) {
                const unsigned int key = q4e_score_key(scores_t[block]);
                if ((key & mask) == prefix) {
                    atomicAdd(&histogram[(key >> shift) & 255u], 1u);
                }
            }
            __syncthreads();
            if (thread == 0) {
                int above = 0;
                int bin = 255;
                for (; bin > 0; --bin) {
                    const int here = (int)histogram[bin];
                    if (above + here >= remaining) {
                        break;
                    }
                    above += here;
                }
                chosen_prefix = prefix | ((unsigned int)bin << shift);
                chosen_remaining = remaining - above;
            }
            __syncthreads();
            prefix = chosen_prefix;
            remaining = chosen_remaining;
            mask |= 255u << shift;
            __syncthreads();
        }
        threshold = prefix;
        ties = remaining;
    }
    const int per = (blocks + blockDim.x - 1) / blockDim.x;
    const int lo = thread * per < blocks ? thread * per : blocks;
    const int hi = lo + per < blocks ? lo + per : blocks;
    int equal = 0;
    if (!all) {
        for (int block = lo; block < hi; ++block) {
            equal += q4e_score_key(scores_t[block]) == threshold ? 1 : 0;
        }
    }
    int equal_total = 0;
    const int equal_before = q4e_block_scan(equal, scan_scratch, &equal_total);
    int kept_here = 0;
    {
        int tie = equal_before;
        for (int block = lo; block < hi; ++block) {
            bool kept = all;
            if (!all) {
                const unsigned int key = q4e_score_key(scores_t[block]);
                kept = key > threshold || (key == threshold && tie++ < ties);
            }
            kept_here += kept ? 1 : 0;
        }
    }
    int kept_total = 0;
    int write_at = q4e_block_scan(kept_here, scan_scratch, &kept_total);
    {
        int tie = equal_before;
        for (int block = lo; block < hi; ++block) {
            bool kept = all;
            if (!all) {
                const unsigned int key = q4e_score_key(scores_t[block]);
                kept = key > threshold || (key == threshold && tie++ < ties);
            }
            if (kept) {
                for (int token = 0; token < ratio; ++token) {
                    selected_t[write_at * ratio + token] = block * ratio + token;
                }
                ++write_at;
            }
        }
    }
    const int tail_start = blocks * ratio;
    const int tail = token_position + 1 - tail_start;
    for (int index = thread; index < tail; index += blockDim.x) {
        selected_t[keep * ratio + index] = tail_start + index;
    }
    if (thread == 0) {
        count[token_index] = keep * ratio + (tail > 0 ? tail : 0);
    }
}

extern "C" __global__ void q4e_qsa_pick(const float* scores, int* selected, int* count,
                                        int position, int ratio, int budget, int score_stride,
                                        int select_stride) {
    OPT_PDL_ENTRY();
    q4e_pick_pools(scores, selected, count, position, ratio, budget, score_stride, select_stride,
                   false);
}

// q4e_qsa_pick for GLM-5.3-Flash's indexer (topk::top_pools with the tail always kept).
extern "C" __global__ void q4e_glm_pick(const float* scores, int* selected, int* count,
                                        int position, int ratio, int budget, int score_stride,
                                        int select_stride) {
    OPT_PDL_ENTRY();
    q4e_pick_pools(scores, selected, count, position, ratio, budget, score_stride, select_stride,
                   true);
}

// GQA attention over the selected keys, with the sigmoid output gate and f32 softmax. A null
// gate leaves the output ungated (GLM-5.3-Flash's DSA).
//
// blockIdx.x is the head and blockIdx.y the token, so one launch covers the whole chunk. The
// kernel is warp-per-slot: warp w owns slots w, w + warps, ... and reduces the head's channels
// with a shuffle instead of a block tree, which is what removes the nine __syncthreads() the
// previous block-per-slot shape spent on every single selected key. At 672 tokens that was about
// six thousand barriers per head per token, and it is why a long prompt measured slower than a
// short one.
//
// The softmax is three block-wide reductions over the whole slot array (max, sum) with the
// normalising divide folded into the value loop, so the only barriers left are one per reduction.
extern "C" __global__ void q4e_attn(const float* __restrict__ q, const float* __restrict__ gate,
                                    const __nv_bfloat16* __restrict__ keys,
                                    const __nv_bfloat16* __restrict__ values,
                                    const int* __restrict__ selected,
                                    const int* __restrict__ selected_count,
                                    float* __restrict__ out, int heads, int kv_heads, int head_dim,
                                    int groups, int select_stride) {
    OPT_PDL_ENTRY();
    __shared__ float logits[Q4E_MAX_SELECTED];
    __shared__ float reduce[Q4E_BLOCK];
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int kv_head = head / groups;
    const int* sel = selected + (unsigned long long)token * select_stride;
    const int nsel = selected_count[token];
    if (head_dim <= 0 || head_dim > Q4E_BLOCK) {
        return;
    }
    const float* qh = q + ((unsigned long long)token * heads + head) * head_dim;
    const float* gh =
        gate != nullptr ? gate + ((unsigned long long)token * heads + head) * head_dim : nullptr;
    float* outh = out + ((unsigned long long)token * heads + head) * head_dim;
    if (nsel <= 0) {
        // Before the indexer has a full block there is nothing to attend over, and the reference
        // keeps the row at zero rather than leaving the buffer untouched.
        for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
            outh[dim] = 0.0f;
        }
        return;
    }
    if (nsel > Q4E_MAX_SELECTED) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const float scale = sqrtf((float)head_dim);

    for (int slot = warp; slot < nsel; slot += warps) {
        const __nv_bfloat16* krow =
            keys + (unsigned long long)sel[slot] * kv_heads * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int dim = lane; dim < head_dim; dim += 32) {
            dot += qh[dim] * __bfloat162float(krow[dim]);
        }
        dot = q4e_warp_sum(dot);
        if (lane == 0) {
            logits[slot] = dot / scale;
        }
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int slot = threadIdx.x; slot < nsel; slot += blockDim.x) {
        local_max = fmaxf(local_max, logits[slot]);
    }
    const float maximum = q4e_block_max(local_max, reduce);
    float local_sum = 0.0f;
    for (int slot = threadIdx.x; slot < nsel; slot += blockDim.x) {
        const float weight = __expf(logits[slot] - maximum);
        logits[slot] = weight;
        local_sum += weight;
    }
    const float total = q4e_block_sum(local_sum, reduce);

    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        float accumulator = 0.0f;
        for (int slot = 0; slot < nsel; ++slot) {
            accumulator += logits[slot] *
                           __bfloat162float(
                               values[(unsigned long long)sel[slot] * kv_heads * head_dim +
                                      kv_head * head_dim + dim]);
        }
        outh[dim] = gh != nullptr ? accumulator / total / (1.0f + __expf(-gh[dim]))
                                  : accumulator / total;
    }
}


// ---------------------------------------------------------------------------------------------
// GLM-5.3-Flash's MLA over the compressed latent, instead of over the expanded per-head keys and
// values q4e_attn read.
//
// The expanded form ran kv_b on the normed latent and cached heads * (nope + value) floats a
// position: 8,192 at the release's geometry on one rank of four, against the 512 the latent it
// expands already holds. Caching the latent instead, and folding kv_b into the two matmuls
// attention already makes, is MLA absorption (DeepSeek-V2's MLA, arXiv:2405.04434, of which
// GLM-5.3-Flash releases the NoPE form):
//
//   k[h][j] = sum_c kv_b[h * (nope + value) + j][c] * c_kv[c]           j < nope
//   v[h][j] = sum_c kv_b[h * (nope + value) + nope + j][c] * c_kv[c]    j < value
//
// so a score is
//
//   sum_j q[h][j] k[h][j] = (sum_j q[h][j] kv_b[h * (nope + value) + j][c]) . c_kv
//
// the query projected into the latent space once a row, and a value is
//
//   sum_slot p[slot] v[h][slot] = kv_b[h * (nope + value) + nope + j] . (sum_slot p[slot] c_kv[slot])
//
// the weighted latent sum projected once a row. Both absorbed operands are this rank's own kv_b
// rows, dequantised once at load, so no kernel here reads the checkpoint's FP8 codes.
//
// The cache is the release's own E4M3 activation: 512 codes and one f32 scale a 128-column block a
// position a layer, 528 bytes, against the expanded form's 2,048. The quantisation is the one the
// FP8 GEMV applied to the latent, so the products are exactly the expansion's - the codes and
// scales are the values it would have multiplied. A configuration without dynamic activation
// quantisation (a BF16 release) caches the latent as f32 instead, and the attention reads either.
#define Q4E_GLM_MAX_LATENT 512

// One row's normed latent into an f32 cache, for a configuration whose FP8 projections do not
// quantise their activations (a BF16 release). grid (rows).
extern "C" __global__ void q4e_glm_mla_cache(const float* __restrict__ latent,
                                             float* __restrict__ cache, int base, int rows,
                                             int width) {
    OPT_PDL_ENTRY();
    const int row = blockIdx.x;
    if (row >= rows || width > Q4E_GLM_MAX_LATENT) {
        return;
    }
    const float* src = latent + (unsigned long long)row * width;
    float* dst = cache + (unsigned long long)(base + row) * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// One row's normed latent into the E4M3 cache: per 128-column block the release's quantisation,
// the largest magnitude over 448 as the scale, each value over it rounded to the nearest code, and
// the scale kept beside the codes. grid (rows), Q4E_FP8_BLOCK threads: a warp a block, four values
// a lane.
extern "C" __global__ void q4e_glm_mla_cache_fp8(const float* __restrict__ latent,
                                                 unsigned char* __restrict__ codes,
                                                 float* __restrict__ scales, int base, int rows,
                                                 int width) {
    OPT_PDL_ENTRY();
    const int row = blockIdx.x;
    if (row >= rows || width > Q4E_GLM_MAX_LATENT || width % Q4E_FP8_BLOCK != 0) {
        return;
    }
    const float* src = latent + (unsigned long long)row * width;
    unsigned char* dst = codes + (unsigned long long)(base + row) * width;
    float* scl = scales + (unsigned long long)(base + row) * (width / Q4E_FP8_BLOCK);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    for (int start = warp * Q4E_FP8_BLOCK; start < width; start += warps * Q4E_FP8_BLOCK) {
        const float4 value = *reinterpret_cast<const float4*>(src + start + lane * 4);
        float largest = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                              fmaxf(fabsf(value.z), fabsf(value.w)));
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            largest = fmaxf(largest, __shfl_xor_sync(0xFFFFFFFFu, largest, offset));
        }
        const float scale = largest / 448.0f;
        if (lane == 0) {
            scl[start / Q4E_FP8_BLOCK] = scale;
        }
        unsigned int c0 = 0u, c1 = 0u, c2 = 0u, c3 = 0u;
        if (scale != 0.0f) {
            c0 = __nv_cvt_float_to_fp8(value.x / scale, __NV_SATFINITE, __NV_E4M3);
            c1 = __nv_cvt_float_to_fp8(value.y / scale, __NV_SATFINITE, __NV_E4M3);
            c2 = __nv_cvt_float_to_fp8(value.z / scale, __NV_SATFINITE, __NV_E4M3);
            c3 = __nv_cvt_float_to_fp8(value.w / scale, __NV_SATFINITE, __NV_E4M3);
        }
        *reinterpret_cast<uchar4*>(dst + start + lane * 4) =
            make_uchar4((unsigned char)c0, (unsigned char)c1, (unsigned char)c2,
                        (unsigned char)c3);
    }
}

// One cached latent, decoded as the release's activation: the E4M3 code times its 128-column
// block's scale, or the f32 cache when the configuration does not quantise.
__device__ __forceinline__ float q4e_glm_mla_at(const float* __restrict__ latent,
                                                const unsigned char* __restrict__ codes,
                                                const float* __restrict__ scales, int width,
                                                int row, int column, int quantise) {
    if (quantise) {
        return q4e_e4m3(codes[(unsigned long long)row * width + column]) *
               scales[(unsigned long long)row * (width / Q4E_FP8_BLOCK) +
                      column / Q4E_FP8_BLOCK];
    }
    return latent[(unsigned long long)row * width + column];
}

// The operands the absorbed MLA's serial sums load at once before they fold them in, in order:
// the sums are the same, and a thread has that many loads in flight instead of one.
#define Q4E_GLM_MLA_AHEAD 8

// The query half of the absorption: q_abs[row][head][c] = sum_j q[row][head][j] * kv_b[h][j][c],
// kv_b's key rows, [nope][latent] a head. grid (heads, rows, latent / blockDim.x rounded up):
// thread threadIdx.x of z-block z owns latent column z * blockDim.x + threadIdx.x, so a warp's
// reads of the weight are consecutive words and its reads of the query one broadcast.
extern "C" __global__ void q4e_glm_mla_absorb(const float* __restrict__ q,
                                              const float* __restrict__ kv_b,
                                              float* __restrict__ q_abs, int base, int nope,
                                              int value, int latent) {
    OPT_PDL_ENTRY();
    const int head = blockIdx.x;
    const int row = base + blockIdx.y;
    const int c = blockIdx.z * blockDim.x + threadIdx.x;
    if (latent > Q4E_GLM_MAX_LATENT || c >= latent) {
        return;
    }
    const float* qh = q + ((unsigned long long)row * gridDim.x + head) * nope;
    const float* wc = kv_b + (unsigned long long)head * (nope + value) * latent + c;
    float acc = 0.0f;
    int j = 0;
    for (; j + Q4E_GLM_MLA_AHEAD <= nope; j += Q4E_GLM_MLA_AHEAD) {
        float w[Q4E_GLM_MLA_AHEAD];
#pragma unroll
        for (int k = 0; k < Q4E_GLM_MLA_AHEAD; ++k) {
            w[k] = wc[(unsigned long long)(j + k) * latent];
        }
#pragma unroll
        for (int k = 0; k < Q4E_GLM_MLA_AHEAD; ++k) {
            acc = fmaf(w[k], qh[j + k], acc);
        }
    }
    for (; j < nope; ++j) {
        acc = fmaf(wc[(unsigned long long)j * latent], qh[j], acc);
    }
    q_abs[((unsigned long long)row * gridDim.x + head) * latent + c] = acc;
}

// The attention and the value half together: a f32 softmax over the selected latents, then the
// weighted latent sum projected by kv_b's value rows, which a head keeps transposed after its key
// rows ([latent][value]) so a warp's reads of them are consecutive words. grid (heads, rows),
// Q4E_BLOCK threads. The score and softmax phases are q4e_attn's, over the latent instead of an
// expanded key. The decode step runs it split four ways (q4e_glm_mla_scores below); this kernel
// is the bits the split is held to (tests/glm_kernels.rs).
extern "C" __global__ void q4e_glm_mla_attend(const float* __restrict__ q_abs,
                                              const float* __restrict__ latent_cache,
                                              const unsigned char* __restrict__ codes,
                                              const float* __restrict__ scales,
                                              const float* __restrict__ kv_b,
                                              const int* __restrict__ selected,
                                              const int* __restrict__ selected_count,
                                              float* __restrict__ out, int base, int nope,
                                              int value, int latent, int select_stride,
                                              int quantise) {
    OPT_PDL_ENTRY();
    __shared__ float logits[Q4E_MAX_SELECTED];
    __shared__ float reduce[Q4E_BLOCK];
    __shared__ float mixed[Q4E_GLM_MAX_LATENT];
    const int head = blockIdx.x;
    const int row = base + blockIdx.y;
    const int* sel = selected + (unsigned long long)row * select_stride;
    const int nsel = selected_count[row];
    if (latent > Q4E_GLM_MAX_LATENT) {
        return;
    }
    const float* qh = q_abs + ((unsigned long long)row * gridDim.x + head) * latent;
    const float* wv = kv_b + (unsigned long long)head * (nope + value) * latent +
                      (unsigned long long)nope * latent;
    float* oh = out + ((unsigned long long)row * gridDim.x + head) * value;
    if (nsel <= 0) {
        // Before the indexer has a full block there is nothing to attend over, and the reference
        // keeps the row at zero rather than leaving the buffer untouched.
        for (int j = threadIdx.x; j < value; j += blockDim.x) {
            oh[j] = 0.0f;
        }
        return;
    }
    if (nsel > Q4E_MAX_SELECTED) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const float scale = sqrtf((float)nope);

    for (int slot = warp; slot < nsel; slot += warps) {
        const int at = sel[slot];
        float dot = 0.0f;
        for (int c = lane; c < latent; c += 32) {
            dot += qh[c] * q4e_glm_mla_at(latent_cache, codes, scales, latent, at, c, quantise);
        }
        dot = q4e_warp_sum(dot);
        if (lane == 0) {
            logits[slot] = dot / scale;
        }
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int slot = threadIdx.x; slot < nsel; slot += blockDim.x) {
        local_max = fmaxf(local_max, logits[slot]);
    }
    const float maximum = q4e_block_max(local_max, reduce);
    float local_sum = 0.0f;
    for (int slot = threadIdx.x; slot < nsel; slot += blockDim.x) {
        const float weight = __expf(logits[slot] - maximum);
        logits[slot] = weight;
        local_sum += weight;
    }
    const float total = q4e_block_sum(local_sum, reduce);

    for (int c = threadIdx.x; c < latent; c += blockDim.x) {
        float accumulator = 0.0f;
        int slot = 0;
        for (; slot + Q4E_GLM_MLA_AHEAD <= nsel; slot += Q4E_GLM_MLA_AHEAD) {
            float x[Q4E_GLM_MLA_AHEAD];
#pragma unroll
            for (int k = 0; k < Q4E_GLM_MLA_AHEAD; ++k) {
                x[k] = q4e_glm_mla_at(latent_cache, codes, scales, latent, sel[slot + k], c,
                                      quantise);
            }
#pragma unroll
            for (int k = 0; k < Q4E_GLM_MLA_AHEAD; ++k) {
                accumulator = fmaf(logits[slot + k], x[k], accumulator);
            }
        }
        for (; slot < nsel; ++slot) {
            accumulator =
                fmaf(logits[slot],
                     q4e_glm_mla_at(latent_cache, codes, scales, latent, sel[slot], c, quantise),
                     accumulator);
        }
        mixed[c] = accumulator / total;
    }
    __syncthreads();

    for (int j = threadIdx.x; j < value; j += blockDim.x) {
        const float* wj = wv + j;
        float accumulator = 0.0f;
        int c = 0;
        for (; c + Q4E_GLM_MLA_AHEAD <= latent; c += Q4E_GLM_MLA_AHEAD) {
            float w[Q4E_GLM_MLA_AHEAD];
#pragma unroll
            for (int k = 0; k < Q4E_GLM_MLA_AHEAD; ++k) {
                w[k] = wj[(unsigned long long)(c + k) * value];
            }
#pragma unroll
            for (int k = 0; k < Q4E_GLM_MLA_AHEAD; ++k) {
                accumulator = fmaf(w[k], mixed[c + k], accumulator);
            }
        }
        for (; c < latent; ++c) {
            accumulator = fmaf(wj[(unsigned long long)c * value], mixed[c], accumulator);
        }
        oh[j] = accumulator;
    }
}

// q4e_glm_mla_attend's attention in four launches, so a full selection spreads over the GPU. The
// one kernel runs a block a head and a row: 16 of 48 SMs at a one-row step, each gathering its
// 2,048 selected latents one after another, twice, 1.3-1.8 ms a launch past 2k positions. Every sum
// here keeps the order it has there and every reduction its tree, so the outputs are its bits:
//
//   q4e_glm_mla_scores   a warp a selected position: its latent is read once for every head;
//   q4e_glm_mla_softmax  the attend kernel's softmax, the weights over the logits in place;
//   q4e_glm_mla_mix      the weighted latent sum, a lane a latent column and a warp a head;
//   q4e_glm_mla_project  the value projection, every row of a run over one read of kv_b.
//
// The logits are [row][head][Q4E_MAX_SELECTED], the mixed latents [row][head][latent], and the
// softmax sums [row][head].
#define Q4E_GLM_MLA_HEADS 16

// q4e_glm_mla_at in two halves, so a thread's gathers are all in flight before the first is
// decoded: the load of a cached latent's word (its E4M3 code, or the f32 cache's bits) and block
// scale, then the decode, the same product. The position and column are clamped in range by the
// caller, so every load is unconditional.
__device__ __forceinline__ void q4e_glm_mla_load(const float* __restrict__ latent_cache,
                                                 const unsigned char* __restrict__ codes,
                                                 const float* __restrict__ scales, int width,
                                                 int row, int column, int quantise,
                                                 unsigned int& word, float& scale) {
    if (quantise) {
        word = codes[(unsigned long long)row * width + column];
        scale = scales[(unsigned long long)row * (width / Q4E_FP8_BLOCK) +
                       column / Q4E_FP8_BLOCK];
    } else {
        word = __float_as_uint(latent_cache[(unsigned long long)row * width + column]);
        scale = 1.0f;
    }
}

__device__ __forceinline__ float q4e_glm_mla_decode(unsigned int word, float scale,
                                                    int quantise) {
    return quantise ? q4e_e4m3((unsigned char)word) * scale : __uint_as_float(word);
}
#define Q4E_GLM_MLA_SCORE_SLOTS 64
#define Q4E_GLM_MLA_MIX_TILE 128
#define Q4E_GLM_MLA_MIX_HEADS 8
#define Q4E_GLM_MLA_PROJECT_ROWS 8
#define Q4E_GLM_MLA_PROJECT_AHEAD 32

// The logits: grid (Q4E_MAX_SELECTED / Q4E_GLM_MLA_SCORE_SLOTS, rows), Q4E_BLOCK threads, at most
// Q4E_GLM_MLA_HEADS heads. A lane sums its columns lane, lane + 32, ... and the warp folds them as
// the attend kernel's warp does.
extern "C" __global__ void q4e_glm_mla_scores(const float* __restrict__ q_abs,
                                              const float* __restrict__ latent_cache,
                                              const unsigned char* __restrict__ codes,
                                              const float* __restrict__ scales,
                                              const int* __restrict__ selected,
                                              const int* __restrict__ selected_count,
                                              float* __restrict__ logits, int base, int heads,
                                              int nope, int latent, int select_stride,
                                              int quantise) {
    OPT_PDL_ENTRY();
    __shared__ float query[Q4E_GLM_MLA_HEADS * Q4E_GLM_MAX_LATENT];
    const int row = base + blockIdx.y;
    const int nsel = selected_count[row];
    const int first = blockIdx.x * Q4E_GLM_MLA_SCORE_SLOTS;
    if (heads > Q4E_GLM_MLA_HEADS || latent > Q4E_GLM_MAX_LATENT || nsel > Q4E_MAX_SELECTED ||
        first >= nsel) {
        return;
    }
    // Every load of the staging in flight at once, and each warp's positions in one load.
    const float4* q = reinterpret_cast<const float4*>(q_abs + (unsigned long long)row * heads *
                                                                   latent);
    float4* staged = reinterpret_cast<float4*>(query);
#pragma unroll 8
    for (int i = threadIdx.x; i < heads * latent / 4; i += blockDim.x) {
        staged[i] = q[i];
    }
    const int* sel = selected + (unsigned long long)row * select_stride;
    float* out = logits + (unsigned long long)row * heads * Q4E_MAX_SELECTED;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const float scale = sqrtf((float)nope);
    const int last = min(first + Q4E_GLM_MLA_SCORE_SLOTS, nsel);
    const int mine = first + warp + warps * lane;
    const int positions = mine < last ? sel[mine] : 0;
    __syncthreads();
    // A warp scores GROUP positions at once, so each staged query word it reads serves all of
    // them; each (head, position) sum is still a lane's columns in order, then the warp's fold.
    constexpr int SPAN = Q4E_GLM_MAX_LATENT / 32;
    constexpr int GROUP = 4;
    for (int group = 0; first + warp + warps * group < last; group += GROUP) {
        float x[GROUP][SPAN];
#pragma unroll
        for (int g = 0; g < GROUP; ++g) {
            const int at = __shfl_sync(0xFFFFFFFFu, positions, group + g);
            unsigned int word[SPAN];
            float block[SPAN];
#pragma unroll
            for (int k = 0; k < SPAN; ++k) {
                q4e_glm_mla_load(latent_cache, codes, scales, latent, at,
                                 min(lane + 32 * k, latent - 1), quantise, word[k], block[k]);
            }
#pragma unroll
            for (int k = 0; k < SPAN; ++k) {
                x[g][k] = lane + 32 * k < latent
                              ? q4e_glm_mla_decode(word[k], block[k], quantise)
                              : 0.0f;
            }
        }
        for (int head = 0; head < heads; ++head) {
            const float* qh = query + head * latent;
            float dot[GROUP];
#pragma unroll
            for (int g = 0; g < GROUP; ++g) {
                dot[g] = 0.0f;
            }
#pragma unroll
            for (int k = 0; k < SPAN; ++k) {
                const int c = lane + 32 * k;
                if (c < latent) {
                    const float qc = qh[c];
#pragma unroll
                    for (int g = 0; g < GROUP; ++g) {
                        dot[g] = fmaf(qc, x[g][k], dot[g]);
                    }
                }
            }
#pragma unroll
            for (int g = 0; g < GROUP; ++g) {
                dot[g] = q4e_warp_sum(dot[g]);
                const int slot = first + warp + warps * (group + g);
                if (lane == 0 && slot < last) {
                    out[head * Q4E_MAX_SELECTED + slot] = dot[g] / scale;
                }
            }
        }
    }
}

// The softmax: grid (heads, rows), Q4E_BLOCK threads, the attend kernel's block max and sum. Each
// logit becomes its unnormalised weight, and the sum goes to total.
extern "C" __global__ void q4e_glm_mla_softmax(float* __restrict__ logits,
                                               const int* __restrict__ selected_count,
                                               float* __restrict__ total, int base) {
    OPT_PDL_ENTRY();
    __shared__ float reduce[Q4E_BLOCK];
    const int head = blockIdx.x;
    const int heads = gridDim.x;
    const int row = base + blockIdx.y;
    const int nsel = selected_count[row];
    if (nsel <= 0 || nsel > Q4E_MAX_SELECTED) {
        return;
    }
    float* weights = logits + ((unsigned long long)row * heads + head) * Q4E_MAX_SELECTED;
    float local_max = -INFINITY;
    for (int slot = threadIdx.x; slot < nsel; slot += blockDim.x) {
        local_max = fmaxf(local_max, weights[slot]);
    }
    const float maximum = q4e_block_max(local_max, reduce);
    float local_sum = 0.0f;
    for (int slot = threadIdx.x; slot < nsel; slot += blockDim.x) {
        const float weight = __expf(weights[slot] - maximum);
        weights[slot] = weight;
        local_sum += weight;
    }
    const float sum = q4e_block_sum(local_sum, reduce);
    if (threadIdx.x == 0) {
        total[(unsigned long long)row * heads + head] = sum;
    }
}

// The weighted latent sum over the selection, in its order, over the softmax's sum: grid
// (latent / 32, heads / Q4E_GLM_MLA_MIX_HEADS, rows), 32 * Q4E_GLM_MLA_MIX_HEADS threads. A tile
// of positions' 32 columns and its weights load into registers while the block folds the tile
// before it, so a warp is never waiting on one position's gather.
extern "C" __global__ void q4e_glm_mla_mix(const float* __restrict__ latent_cache,
                                           const unsigned char* __restrict__ codes,
                                           const float* __restrict__ scales,
                                           const int* __restrict__ selected,
                                           const int* __restrict__ selected_count,
                                           const float* __restrict__ weights,
                                           const float* __restrict__ total,
                                           float* __restrict__ mixed, int base, int heads,
                                           int latent, int select_stride, int quantise) {
    OPT_PDL_ENTRY();
    constexpr int TILE = Q4E_GLM_MLA_MIX_TILE;
    constexpr int WARPS = Q4E_GLM_MLA_MIX_HEADS;
    __shared__ int at[Q4E_MAX_SELECTED];
    __shared__ float xs[TILE][32];
    __shared__ float ws[WARPS][TILE];
    const int row = base + blockIdx.z;
    const int nsel = selected_count[row];
    if (nsel <= 0 || nsel > Q4E_MAX_SELECTED || latent > Q4E_GLM_MAX_LATENT ||
        blockDim.x != 32 * WARPS) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int column = blockIdx.x * 32 + lane;
    const int head = blockIdx.y * WARPS + warp;
    const int* sel = selected + (unsigned long long)row * select_stride;
#pragma unroll
    for (int i = 0; i < (Q4E_MAX_SELECTED + 32 * WARPS - 1) / (32 * WARPS); ++i) {
        const int slot = threadIdx.x + 32 * WARPS * i;
        if (slot < nsel) {
            at[slot] = sel[slot];
        }
    }
    __syncthreads();
    const float* wh = weights + ((unsigned long long)row * heads + head) * Q4E_MAX_SELECTED;
    unsigned int word[TILE / WARPS];
    float block[TILE / WARPS];
    float w[TILE / 32];
    const int read = min(column, latent - 1);
    auto fetch = [&](int first) {
#pragma unroll
        for (int i = 0; i < TILE / WARPS; ++i) {
            q4e_glm_mla_load(latent_cache, codes, scales, latent,
                             at[min(first + warp + WARPS * i, nsel - 1)], read, quantise,
                             word[i], block[i]);
        }
#pragma unroll
        for (int i = 0; i < TILE / 32; ++i) {
            w[i] = head < heads ? wh[min(first + lane + 32 * i, nsel - 1)] : 0.0f;
        }
    };
    fetch(0);
    float accumulator = 0.0f;
    for (int first = 0; first < nsel; first += TILE) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < TILE / WARPS; ++i) {
            xs[warp + WARPS * i][lane] = q4e_glm_mla_decode(word[i], block[i], quantise);
        }
#pragma unroll
        for (int i = 0; i < TILE / 32; ++i) {
            ws[warp][lane + 32 * i] = w[i];
        }
        __syncthreads();
        if (first + TILE < nsel) {
            fetch(first + TILE);
        }
        const int count = min(TILE, nsel - first);
        if (count == TILE) {
#pragma unroll 16
            for (int slot = 0; slot < TILE; ++slot) {
                accumulator = fmaf(ws[warp][slot], xs[slot][lane], accumulator);
            }
        } else {
            for (int slot = 0; slot < count; ++slot) {
                accumulator = fmaf(ws[warp][slot], xs[slot][lane], accumulator);
            }
        }
    }
    if (column < latent && head < heads) {
        mixed[((unsigned long long)row * heads + head) * latent + column] =
            accumulator / total[(unsigned long long)row * heads + head];
    }
}

// The value projection, kv_b's value rows ([latent][value] a head, after its key rows) over the
// mixed latent: grid (heads, value / blockDim.x rounded up), rows base .. base + rows at once, so
// the weights are read once a run. A row with nothing selected is zero, as the attend kernel
// leaves it.
extern "C" __global__ void q4e_glm_mla_project(const float* __restrict__ mixed,
                                               const float* __restrict__ kv_b,
                                               const int* __restrict__ selected_count,
                                               float* __restrict__ out, int base, int rows,
                                               int nope, int value, int latent) {
    OPT_PDL_ENTRY();
    constexpr int ROWS = Q4E_GLM_MLA_PROJECT_ROWS;
    constexpr int AHEAD = Q4E_GLM_MLA_PROJECT_AHEAD;
    __shared__ float ms[ROWS][Q4E_GLM_MAX_LATENT];
    const int head = blockIdx.x;
    const int heads = gridDim.x;
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (latent > Q4E_GLM_MAX_LATENT) {
        return;
    }
    const float* wj = kv_b + (unsigned long long)head * (nope + value) * latent +
                      (unsigned long long)nope * latent + j;
    for (int first = 0; first < rows; first += ROWS) {
        const int count = min(ROWS, rows - first);
        __syncthreads();
        for (int r = 0; r < count; ++r) {
            const float* m =
                mixed + ((unsigned long long)(base + first + r) * heads + head) * latent;
#pragma unroll 8
            for (int c = threadIdx.x; c < latent; c += blockDim.x) {
                ms[r][c] = m[c];
            }
        }
        __syncthreads();
        if (j >= value) {
            continue;
        }
        float accumulator[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            accumulator[r] = 0.0f;
        }
        int c = 0;
        for (; c + AHEAD <= latent; c += AHEAD) {
            float w[AHEAD];
#pragma unroll
            for (int k = 0; k < AHEAD; ++k) {
                w[k] = wj[(unsigned long long)(c + k) * value];
            }
#pragma unroll
            for (int k = 0; k < AHEAD; ++k) {
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    accumulator[r] = fmaf(w[k], ms[r][c + k], accumulator[r]);
                }
            }
        }
        for (; c < latent; ++c) {
            const float wc = wj[(unsigned long long)c * value];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                accumulator[r] = fmaf(wc, ms[r][c], accumulator[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int at = base + first + r;
            if (r < count) {
                const int nsel = selected_count[at];
                if (nsel <= Q4E_MAX_SELECTED) {
                    out[((unsigned long long)at * heads + head) * value + j] =
                        nsel > 0 ? accumulator[r] : 0.0f;
                }
            }
        }
    }
}

// Block reductions of HEADS values at once, each over the same tree as q4e_block_max and
// q4e_block_sum, so each head's result is bit-identical to reducing it alone; scratch holds
// HEADS * blockDim.x floats. Reusable back to back, as those are.
template <int HEADS>
__device__ void q4e_block_max_heads(float* value, float* scratch) {
    const int thread = threadIdx.x;
    const int width = blockDim.x;
#pragma unroll
    for (int head = 0; head < HEADS; ++head) {
        scratch[head * width + thread] = value[head];
    }
    __syncthreads();
    for (int stride = width >> 1; stride > 0; stride >>= 1) {
        if (thread < stride) {
#pragma unroll
            for (int head = 0; head < HEADS; ++head) {
                float* at = scratch + head * width + thread;
                at[0] = fmaxf(at[0], at[stride]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int head = 0; head < HEADS; ++head) {
        value[head] = scratch[head * width];
    }
    __syncthreads();
}

template <int HEADS>
__device__ void q4e_block_sum_heads(float* value, float* scratch) {
    const int thread = threadIdx.x;
    const int width = blockDim.x;
#pragma unroll
    for (int head = 0; head < HEADS; ++head) {
        scratch[head * width + thread] = value[head];
    }
    __syncthreads();
    for (int stride = width >> 1; stride > 0; stride >>= 1) {
        if (thread < stride) {
#pragma unroll
            for (int head = 0; head < HEADS; ++head) {
                float* at = scratch + head * width + thread;
                at[0] += at[stride];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int head = 0; head < HEADS; ++head) {
        value[head] = scratch[head * width];
    }
    __syncthreads();
}

// The query heads q4e_attn_group computes in one CTA; it divides the released configuration's
// twelve query heads a KV head. q4e_attn_group6 takes six, the share of a KV head's twelve each rank
// of a four-rank group holds.
#define Q4E_ATTN_GROUP 4
// The selected tokens q4e_attn_group stages in shared memory: the released configuration's 2,051
// (512 blocks of four and the incomplete tail) rounded up to a multiple of four. The host sends a
// configuration that can select more to q4e_attn.
#define Q4E_ATTN_GROUP_SELECTED 2052
// Keys a warp has in flight in the key pass, and values a thread has in flight in the value pass.
#define Q4E_ATTN_GROUP_KEYS 2
#define Q4E_ATTN_GROUP_VALUES 16

// q4e_attn for HEADS query heads of one KV head at once (q4e_attn_group: Q4E_ATTN_GROUP of them).
//
// Every head is computed exactly as q4e_attn computes it alone: the same lane takes the same
// channels of a key in the same order, the same warp and block trees reduce them, and the value
// sum runs over the slots in order, so the outputs are bit-identical. What changes is the
// reading: a token's selected keys and values, about 1 MiB each at the full budget, are read once for
// the group instead of once a head, and twelve query heads share a KV head. The loads are batched
// ahead of the arithmetic that consumes them, which reorders reads and nothing else.
//
// blockIdx.x is the head group (its first head is blockIdx.x * HEADS) and blockIdx.y the token.
// Dynamic shared memory holds a row of Q4E_ATTN_GROUP_SELECTED logits a head, then a block of
// reduction floats a head, then the token's selection.
template <int HEADS>
__device__ __forceinline__ void q4e_attn_group_heads(const float* __restrict__ q,
                                                     const float* __restrict__ gate,
                                                     const __nv_bfloat16* __restrict__ keys,
                                                     const __nv_bfloat16* __restrict__ values,
                                                     const int* __restrict__ selected,
                                                     const int* __restrict__ selected_count,
                                                     float* __restrict__ out, int heads,
                                                     int kv_heads, int head_dim, int groups,
                                                     int select_stride) {
    constexpr int IN = Q4E_ATTN_GROUP_KEYS;
    constexpr int UNROLL = Q4E_ATTN_GROUP_VALUES;
    extern __shared__ float q4e_attn_shared[];
    float* logits = q4e_attn_shared;
    float* reduce = logits + HEADS * Q4E_ATTN_GROUP_SELECTED;
    int* sel = (int*)(reduce + HEADS * Q4E_BLOCK);
    const int first = blockIdx.x * HEADS;
    const int token = blockIdx.y;
    const int kv_head = first / groups;
    const int nsel = selected_count[token];
    if (head_dim <= 0 || head_dim > Q4E_BLOCK) {
        return;
    }
    const unsigned long long row = ((unsigned long long)token * heads + first) * head_dim;
    if (nsel <= 0) {
        for (int index = threadIdx.x; index < HEADS * head_dim; index += blockDim.x) {
            out[row + index] = 0.0f;
        }
        return;
    }
    if (nsel > Q4E_ATTN_GROUP_SELECTED) {
        return;
    }
    const unsigned long long kv_stride = (unsigned long long)kv_heads * head_dim;
    const unsigned long long kv_offset = (unsigned long long)kv_head * head_dim;
    {
        const int* source = selected + (unsigned long long)token * select_stride;
        for (int slot = threadIdx.x; slot < nsel; slot += blockDim.x) {
            sel[slot] = source[slot];
        }
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const float scale = sqrtf((float)head_dim);

    // The lane's channels of every head's query, held across its slots.
    float query[HEADS][Q4E_BLOCK / 32];
#pragma unroll
    for (int head = 0; head < HEADS; ++head) {
#pragma unroll
        for (int step = 0; step < Q4E_BLOCK / 32; ++step) {
            const int dim = lane + 32 * step;
            query[head][step] = dim < head_dim ? q[row + head * head_dim + dim] : 0.0f;
        }
    }
    __syncthreads();
    // Warp w still owns slots w, w + warps, ...; it loads IN of them before scoring the first. A
    // slot past the selection reloads the base slot and is not written.
    for (int base = warp; base < nsel; base += IN * warps) {
        float key[IN][Q4E_BLOCK / 32];
#pragma unroll
        for (int k = 0; k < IN; ++k) {
            const int slot = base + k * warps < nsel ? base + k * warps : base;
            const __nv_bfloat16* krow =
                keys + (unsigned long long)sel[slot] * kv_stride + kv_offset;
#pragma unroll
            for (int step = 0; step < Q4E_BLOCK / 32; ++step) {
                const int dim = lane + 32 * step;
                key[k][step] =
                    dim < head_dim ? __bfloat162float(krow[dim]) : 0.0f;
            }
        }
#pragma unroll
        for (int k = 0; k < IN; ++k) {
            const int slot = base + k * warps;
            float dot[HEADS];
#pragma unroll
            for (int head = 0; head < HEADS; ++head) {
                dot[head] = 0.0f;
            }
#pragma unroll
            for (int step = 0; step < Q4E_BLOCK / 32; ++step) {
                if (lane + 32 * step < head_dim) {
#pragma unroll
                    for (int head = 0; head < HEADS; ++head) {
                        dot[head] += query[head][step] * key[k][step];
                    }
                }
            }
#pragma unroll
            for (int head = 0; head < HEADS; ++head) {
                dot[head] = q4e_warp_sum(dot[head]);
            }
            if (lane == 0 && slot < nsel) {
#pragma unroll
                for (int head = 0; head < HEADS; ++head) {
                    logits[head * Q4E_ATTN_GROUP_SELECTED + slot] = dot[head] / scale;
                }
            }
        }
    }
    __syncthreads();

    float maximum[HEADS];
#pragma unroll
    for (int head = 0; head < HEADS; ++head) {
        maximum[head] = -INFINITY;
        for (int slot = threadIdx.x; slot < nsel; slot += blockDim.x) {
            maximum[head] = fmaxf(maximum[head], logits[head * Q4E_ATTN_GROUP_SELECTED + slot]);
        }
    }
    q4e_block_max_heads<HEADS>(maximum, reduce);
    float total[HEADS];
#pragma unroll
    for (int head = 0; head < HEADS; ++head) {
        total[head] = 0.0f;
        for (int slot = threadIdx.x; slot < nsel; slot += blockDim.x) {
            const float weight =
                __expf(logits[head * Q4E_ATTN_GROUP_SELECTED + slot] - maximum[head]);
            logits[head * Q4E_ATTN_GROUP_SELECTED + slot] = weight;
            total[head] += weight;
        }
    }
    q4e_block_sum_heads<HEADS>(total, reduce);

    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        float accumulator[HEADS];
#pragma unroll
        for (int head = 0; head < HEADS; ++head) {
            accumulator[head] = 0.0f;
        }
        const __nv_bfloat16* column = values + kv_offset + dim;
        int slot = 0;
        for (; slot + UNROLL <= nsel; slot += UNROLL) {
            float value[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                value[u] = __bfloat162float(
                    column[(unsigned long long)sel[slot + u] * kv_stride]);
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
                for (int head = 0; head < HEADS; ++head) {
                    accumulator[head] +=
                        logits[head * Q4E_ATTN_GROUP_SELECTED + slot + u] * value[u];
                }
            }
        }
        for (; slot < nsel; ++slot) {
            const float value =
                __bfloat162float(column[(unsigned long long)sel[slot] * kv_stride]);
#pragma unroll
            for (int head = 0; head < HEADS; ++head) {
                accumulator[head] += logits[head * Q4E_ATTN_GROUP_SELECTED + slot] * value;
            }
        }
#pragma unroll
        for (int head = 0; head < HEADS; ++head) {
            out[row + head * head_dim + dim] =
                accumulator[head] / total[head] /
                (1.0f + __expf(-gate[row + head * head_dim + dim]));
        }
    }
}

extern "C" __global__ void q4e_attn_group(const float* __restrict__ q,
                                          const float* __restrict__ gate,
                                          const __nv_bfloat16* __restrict__ keys,
                                          const __nv_bfloat16* __restrict__ values,
                                          const int* __restrict__ selected,
                                          const int* __restrict__ selected_count,
                                          float* __restrict__ out, int heads, int kv_heads,
                                          int head_dim, int groups, int select_stride) {
    OPT_PDL_ENTRY();
    q4e_attn_group_heads<Q4E_ATTN_GROUP>(q, gate, keys, values, selected, selected_count, out,
                                         heads, kv_heads, head_dim, groups, select_stride);
}

extern "C" __global__ void q4e_attn_group6(const float* __restrict__ q,
                                           const float* __restrict__ gate,
                                           const __nv_bfloat16* __restrict__ keys,
                                           const __nv_bfloat16* __restrict__ values,
                                           const int* __restrict__ selected,
                                           const int* __restrict__ selected_count,
                                           float* __restrict__ out, int heads, int kv_heads,
                                           int head_dim, int groups, int select_stride) {
    OPT_PDL_ENTRY();
    q4e_attn_group_heads<6>(q, gate, keys, values, selected, selected_count, out, heads,
                            kv_heads, head_dim, groups, select_stride);
}

// The PLE gate and its post-gate norm (blockIdx.y = batch token).
extern "C" __global__ void q4e_ple_gate(const float* __restrict__ key,
                                        const float* __restrict__ value,
                                        const float* __restrict__ query,
                                        const __nv_bfloat16* __restrict__ conv_weight,
                                        float* __restrict__ gated, float* __restrict__ normed,
                                        int hc, int hidden, float eps) {
    OPT_PDL_ENTRY();
    int width = hc * hidden;
    int b = blockIdx.y;
    const float* key_b = key + (unsigned long long)b * width;
    const float* value_b = value + (unsigned long long)b * hidden;
    const float* query_b = query + (unsigned long long)b * width;
    float* gated_b = gated + (unsigned long long)b * width;
    float* normed_b = normed + (unsigned long long)b * width;
    __shared__ float scratch[Q4E_BLOCK];
    for (int group = blockIdx.x; group < hc; group += gridDim.x) {
        int base = group * hidden;
        float dot = 0.0f;
        for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
            dot += key_b[base + column] * query_b[base + column];
        }
        float gate = q4e_block_sum(dot, scratch) / sqrtf((float)hidden);
        float magnitude = sqrtf(fmaxf(fabsf(gate), 1e-6f));
        float gain = copysignf(magnitude, gate);
        float g = 1.0f / (1.0f + __expf(-gain));
        float sum = 0.0f;
        for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
            float v = g * value_b[column];
            gated_b[base + column] = v;
            sum += v * v;
        }
        float scale = rsqrtf(q4e_block_sum(sum, scratch) / (float)hidden + eps);
        for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
            int index = base + column;
            normed_b[index] =
                gated_b[index] * scale * (1.0f + __bfloat162float(conv_weight[index]));
        }
    }
}

// Dilated depthwise convolution with SiLU, the PLE short convolution, across batch token steps.
extern "C" __global__ void q4e_ple_conv(const float* __restrict__ normed, float* __restrict__ state,
                                        const __nv_bfloat16* __restrict__ weight,
                                        float* __restrict__ out, int width, int taps, int dilation,
                                        int batch, float* __restrict__ snapshot,
                                        int snapshot_rows, long long region_stride) {
    OPT_PDL_ENTRY();
    int state_len = (taps - 1) * dilation;
    int last = state_len - 1;
    for (int column = blockIdx.x * blockDim.x + threadIdx.x; column < width;
         column += blockDim.x * gridDim.x) {
        for (int t = 0; t < batch; ++t) {
            float src = normed[(unsigned long long)t * width + column];
            float accumulator = 0.0f;
            for (int tap = 0; tap < taps; ++tap) {
                int back = (taps - 1 - tap) * dilation;
                float source = (back == 0) ? src : state[(state_len - back) * width + column];
                accumulator += __bfloat162float(weight[column * taps + tap]) * source;
            }
            out[(unsigned long long)t * width + column] = q4e_silu(accumulator);
            for (int row = 0; row < last; ++row) {
                state[row * width + column] = state[(row + 1) * width + column];
            }
            if (state_len > 0) {
                state[last * width + column] = src;
            }
            if (snapshot != nullptr && t < snapshot_rows) {
                float* into = snapshot + (unsigned long long)t * region_stride;
                for (int row = 0; row < state_len; ++row) {
                    into[row * width + column] = state[row * width + column];
                }
            }
        }
    }
}

// Split the interleaved [token][head][query | gate] projection into two contiguous buffers, one
// blockIdx.y per token.
extern "C" __global__ void q4e_deinterleave(const float* __restrict__ interleaved,
                                            float* __restrict__ query, float* __restrict__ gate,
                                            int heads, int head_dim) {
    OPT_PDL_ENTRY();
    const int width = heads * head_dim;
    const int token = blockIdx.y;
    const float* src = interleaved + (unsigned long long)token * 2 * width;
    float* q = query + (unsigned long long)token * width;
    float* g = gate + (unsigned long long)token * width;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < width;
         index += blockDim.x * gridDim.x) {
        int head = index / head_dim;
        int dim = index % head_dim;
        q[index] = src[head * 2 * head_dim + dim];
        g[index] = src[head * 2 * head_dim + head_dim + dim];
    }
}

// dst[i] = src[i] for count floats: the KV append of one chunk's staging rows into the cache,
// which is contiguous on both sides because the cache is row-major in token order.
extern "C" __global__ void q4e_copy_flat(float* __restrict__ dst, const float* __restrict__ src,
                                          int count) {
    OPT_PDL_ENTRY();
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += blockDim.x * gridDim.x) {
        dst[index] = src[index];
    }
}

// ---------------------------------------------------------------------------------------------
// GLM-5.3-Flash: the kernels its device arm (sparkflash-model-glm5-next) adds to the ones above.
//
// The host twin is that crate's forward.rs with sparkflash_weights' mhc, linalg and topk. Where a
// comment says a kernel mirrors the host, it takes the host's operations in the host's order, each
// rounded on its own (__fmul_rn / __fadd_rn, so nvcc does not fuse them), and the accurate expf.
// ---------------------------------------------------------------------------------------------

// The most mHC copies the coefficient kernel holds: one comb cell a lane of one warp.
#define Q4E_MHC_MAX_HC 5
// The most columns of every copy a thread of q4e_mhc_in holds: the hidden width is a multiple of
// the block, at most this many of them (the host refuses others).
#define Q4E_MHC_COLUMNS 16

// a / d by the sequence nvcc emits for `/` when FCHK finds the operands in range: an approximate
// reciprocal, a Newton step on it, and an fma correction of the quotient. Where q4e_div_in_range
// holds, that is the path `/` takes, so the quotient rounds as `/`'s does (7.6e11 sampled pairs,
// none apart from __fdiv_rn). It leaves out `/`'s branch to its slow path, which in a dependent
// chain of divisions costs some 250 cycles each.
__device__ __forceinline__ float q4e_div_fast(float a, float d) {
    float reciprocal;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(reciprocal) : "f"(d));
    const float error = __fmaf_rn(-d, reciprocal, 1.0f);
    const float refined = __fmaf_rn(reciprocal, error, reciprocal);
    const float quotient = __fmaf_rn(a, refined, 0.0f);
    return __fmaf_rn(refined, __fmaf_rn(-d, quotient, a), quotient);
}

// Whether q4e_div_fast(a, d) is a / d: both operands normal, of magnitude in [2^-60, 2^61), so no
// step of the quotient leaves the normal range.
__device__ __forceinline__ bool q4e_div_in_range(float a, float d) {
    const unsigned a_exponent = (__float_as_uint(a) >> 23) & 0xFFu;
    const unsigned d_exponent = (__float_as_uint(d) >> 23) & 0xFFu;
    return a_exponent - 67u <= 120u && d_exponent - 67u <= 120u;
}

// sqrtf(x) by the path nvcc emits for it where q4e_sqrt_in_range holds (its own test): an
// approximate reciprocal root, then one correction. It leaves out sqrtf's branch to its slow path.
__device__ __forceinline__ float q4e_sqrt_fast(float x) {
    float reciprocal;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(reciprocal) : "f"(x));
    const float root = __fmul_rn(x, reciprocal);
    return __fmaf_rn(__fmaf_rn(-root, root, x), __fmul_rn(reciprocal, 0.5f), root);
}

__device__ __forceinline__ bool q4e_sqrt_in_range(float x) {
    return __float_as_uint(x) - 0x0D000000u <= 0x727FFFFFu;
}

// a / d and sqrtf(x): `/` and sqrtf themselves, or with FAST their fast paths, clearing `exact`
// where those may not round as they do.
template <bool FAST>
__device__ __forceinline__ float q4e_mhc_div(float a, float d, bool& exact) {
    if constexpr (FAST) {
        exact = exact && q4e_div_in_range(a, d);
        return q4e_div_fast(a, d);
    }
    return a / d;
}

template <bool FAST>
__device__ __forceinline__ float q4e_mhc_sqrt(float x, bool& exact) {
    if constexpr (FAST) {
        exact = exact && q4e_sqrt_in_range(x);
        return q4e_sqrt_fast(x);
    }
    return sqrtf(x);
}

// 1 / sqrt(sum / count + eps): the RMS scale of mhc::hc_mixes_from_projection and linalg::rms_norm.
template <bool FAST>
__device__ __forceinline__ float q4e_mhc_rms_scale(float sum, float count, float eps,
                                                   bool& exact) {
    const float mean = q4e_mhc_div<FAST>(sum, count, exact);
    return q4e_mhc_div<FAST>(1.0f, q4e_mhc_sqrt<FAST>(__fadd_rn(mean, eps), exact), exact);
}

// What lane l of q4e_mhc_in's warp 0 reads of a token's fn projection and the mixer's
// parameters: gate l's (l < 2 hc) and comb cell l's (l < hc * hc), zero past them.
struct Q4eMhcLane {
    float gate_raw, gate_scale, gate_base;
    float cell_raw, cell_scale, cell_base;
};

// The parameters of the lane's gate (l < gates) and, with comb, its comb cell: before the wait on
// the launch before, which never writes them.
__device__ __forceinline__ Q4eMhcLane q4e_mhc_lane(const float* __restrict__ scale,
                                                   const float* __restrict__ base, int hc,
                                                   int gates, bool comb) {
    const int lane = threadIdx.x;
    Q4eMhcLane in = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (lane < gates) {
        in.gate_scale = scale[lane < hc ? 0 : 1];
        in.gate_base = base[lane];
    }
    if (comb && lane < hc * hc) {
        in.cell_scale = scale[2];
        in.cell_base = base[2 * hc + lane];
    }
    return in;
}

// Output i of token t's fn projection: raw_t[i], its row of the tiled GEMV's output, or with
// parts_t the [splits][mixes] partials q4e_mhc_post worked out in the GEMV's place, added from 0.0
// in split order as the GEMV's last block adds its own.
__device__ __forceinline__ float q4e_mhc_raw(const float* __restrict__ raw_t,
                                             const float* __restrict__ parts_t, int splits,
                                             int mixes, int i) {
    if (parts_t == nullptr) {
        return raw_t[i];
    }
    float sum = 0.0f;
#pragma unroll 16
    for (int split = 0; split < splits; ++split) {
        sum += parts_t[(unsigned long long)split * mixes + i];
    }
    return sum;
}

// The lane's share of token t's fn projection ([batch][(2 + hc) hc]), after the wait.
__device__ __forceinline__ void q4e_mhc_lane_raw(Q4eMhcLane& in, const float* __restrict__ raw_t,
                                                 const float* __restrict__ parts_t, int splits,
                                                 int hc, int gates, bool comb) {
    const int lane = threadIdx.x;
    const int mixes = (2 + hc) * hc;
    if (lane < gates) {
        in.gate_raw = q4e_mhc_raw(raw_t, parts_t, splits, mixes, lane);
    }
    if (comb && lane < hc * hc) {
        in.cell_raw = q4e_mhc_raw(raw_t, parts_t, splits, mixes, 2 * hc + lane);
    }
}

// A token's streams [copy][hidden] at x: a thread loads its columns j = threadIdx.x + k blockDim.x
// of every copy at once into held[copy][k] and adds their squares in the streams' index order
// (copy, then column), and q4e_block_sum reduces the block's. `scratch` holds Q4E_BLOCK floats.
__device__ __forceinline__ float q4e_mhc_hold(const float* __restrict__ x, int hidden, int hc,
                                              int columns,
                                              float (&held)[Q4E_MHC_MAX_HC][Q4E_MHC_COLUMNS],
                                              float* scratch) {
#pragma unroll
    for (int copy = 0; copy < Q4E_MHC_MAX_HC; ++copy) {
#pragma unroll
        for (int k = 0; k < Q4E_MHC_COLUMNS; ++k) {
            held[copy][k] = copy < hc && k < columns
                                ? x[(unsigned long long)copy * hidden + threadIdx.x +
                                    k * blockDim.x]
                                : 0.0f;
        }
    }
    float sum = 0.0f;
#pragma unroll
    for (int copy = 0; copy < Q4E_MHC_MAX_HC; ++copy) {
#pragma unroll
        for (int k = 0; k < Q4E_MHC_COLUMNS; ++k) {
            if (copy < hc && k < columns) {
                sum = __fadd_rn(sum, __fmul_rn(held[copy][k], held[copy][k]));
            }
        }
    }
    return q4e_block_sum(sum, scratch);
}

// Warp 0's mHC coefficients of a token, as mhc::hc_mixes_from_projection and split_sinkhorn: lane
// l < gates (hc for pre alone, 2 hc for pre and post) gets its gate, and with `comb` lane row * hc +
// col < hc * hc gets comb[row][col]. Every row or column sum reads its hc cells through shuffles in
// the host's order, so the 2 * iters normalisations round as the host's do. With FAST every
// division and square root takes its fast path, and `exact` clears where one the lane returns
// depends on may not have rounded as `/` and sqrtf do.
template <bool FAST>
__device__ __forceinline__ void q4e_mhc_mixes(const Q4eMhcLane& in, float sum_squares, int hc,
                                              int width, int gates, bool comb, int iters,
                                              float hc_eps, float norm_eps, float& gate,
                                              float& cell, bool& exact) {
    const int lane = threadIdx.x & 31;
    const float rsqrt = q4e_mhc_rms_scale<FAST>(sum_squares, (float)width, norm_eps, exact);
    // linalg::sigmoid: 1 / (1 + e^-x) at x >= 0 and e^x / (1 + e^x) below.
    const float x =
        __fadd_rn(__fmul_rn(__fmul_rn(in.gate_raw, rsqrt), in.gate_scale), in.gate_base);
    const bool positive = x >= 0.0f;
    const float power = expf(positive ? -x : x);
    bool gate_exact = true;
    const float sigmoid = q4e_mhc_div<FAST>(positive ? 1.0f : power, 1.0f + power, gate_exact);
    gate = lane < hc ? __fadd_rn(sigmoid, hc_eps) : __fmul_rn(2.0f, sigmoid);
    exact = exact && (gate_exact || lane >= gates);
    if (!comb) {
        return;
    }
    const bool live = lane < hc * hc;
    const int row = live ? lane / hc : 0;
    const int col = live ? lane - row * hc : 0;
    float value =
        live ? __fadd_rn(__fmul_rn(__fmul_rn(in.cell_raw, rsqrt), in.cell_scale), in.cell_base)
             : 0.0f;
    // The row softmax (linalg::softmax), then hc_eps.
    float maximum = -INFINITY;
#pragma unroll
    for (int k = 0; k < Q4E_MHC_MAX_HC; ++k) {
        const float other = __shfl_sync(0xFFFFFFFFu, value, row * hc + k);
        if (k < hc) {
            maximum = fmaxf(maximum, other);
        }
    }
    const float e = expf(value - maximum);
    float total = 0.0f;
#pragma unroll
    for (int k = 0; k < Q4E_MHC_MAX_HC; ++k) {
        const float other = __shfl_sync(0xFFFFFFFFu, e, row * hc + k);
        if (k < hc) {
            total = __fadd_rn(total, other);
        }
    }
    bool cell_exact = true;
    const float share = q4e_mhc_div<FAST>(e, total, cell_exact);
    cell_exact = cell_exact || !(isfinite(maximum) && total > 0.0f);
    value = isfinite(maximum) ? (total > 0.0f ? share : e) : 0.0f;
    value = __fadd_rn(value, hc_eps);
    for (int pass = 0; pass < iters; ++pass) {
        if (pass > 0) {
            float row_sum = 0.0f;
#pragma unroll
            for (int k = 0; k < Q4E_MHC_MAX_HC; ++k) {
                const float other = __shfl_sync(0xFFFFFFFFu, value, row * hc + k);
                if (k < hc) {
                    row_sum = __fadd_rn(row_sum, other);
                }
            }
            value = q4e_mhc_div<FAST>(value, __fadd_rn(row_sum, hc_eps), cell_exact);
        }
        float column_sum = 0.0f;
#pragma unroll
        for (int k = 0; k < Q4E_MHC_MAX_HC; ++k) {
            const float other = __shfl_sync(0xFFFFFFFFu, value, k * hc + col);
            if (k < hc) {
                column_sum = __fadd_rn(column_sum, other);
            }
        }
        value = q4e_mhc_div<FAST>(value, __fadd_rn(column_sum, hc_eps), cell_exact);
    }
    cell = value;
    exact = exact && (cell_exact || !live);
}

// q4e_mhc_mixes on the fast paths, and again through `/` and sqrtf when one of the warp's may not
// have rounded as they do.
__device__ __forceinline__ void q4e_mhc_warp_mixes(const Q4eMhcLane& in, float sum_squares, int hc,
                                                   int width, int gates, bool comb, int iters,
                                                   float hc_eps, float norm_eps, float& gate,
                                                   float& cell) {
    bool exact = true;
    q4e_mhc_mixes<true>(in, sum_squares, hc, width, gates, comb, iters, hc_eps, norm_eps, gate,
                        cell, exact);
    if (__any_sync(0xFFFFFFFFu, !exact)) {
        q4e_mhc_mixes<false>(in, sum_squares, hc, width, gates, comb, iters, hc_eps, norm_eps,
                             gate, cell, exact);
    }
}

// A sublayer's mixer: grid (batch, 2), Q4E_BLOCK threads, two blocks a token that run side by
// side. Block (t, 1) writes token t's coefficients [batch][pre (hc) | post (hc) | comb (hc x hc)]
// from its fn projection and its streams' sum of squares. Block (t, 0) works out only the pre
// coefficients, and from them the normed sublayer input `out`: sum_c pre[c] x[c][j] from zero in
// copy order (mhc::hc_pre), then RMSNorm with the sublayer's BF16 weight (linalg::rms_norm). So
// the input never waits on the comb's 2 * iters normalisations. Only the sums of squares are
// reassociated (block trees).
//
// A thread keeps the stream values it squares (q4e_mhc_hold): they are the very columns its pre
// sums read. The mixer's parameters and the norm weight it reads before waiting on the launch
// before. Warp 0 takes the divisions' and square roots' fast paths (q4e_mhc_warp_mixes), as does
// the input's RMS scale, each falling back to `/` and sqrtf where those may round otherwise.
//
// The fn projection is `raw`, or with `partials` the split partials the q4e_mhc_post before it
// left there ([batch][hc hidden / Q4E_BLOCK][(2 + hc) hc]); `raw` is then unread.
extern "C" __global__ void q4e_mhc_in(const float* __restrict__ raw,
                                      const float* __restrict__ streams,
                                      const float* __restrict__ scale,
                                      const float* __restrict__ base, float* __restrict__ coeffs,
                                      const __nv_bfloat16* __restrict__ weight,
                                      float* __restrict__ out, int hidden, int hc, int iters,
                                      float hc_eps, float norm_eps, float eps,
                                      const float* __restrict__ partials) {
    OPT_PDL_LAUNCH();
    __shared__ float scratch[Q4E_BLOCK];
    __shared__ float pre[Q4E_MHC_MAX_HC];
    const int columns = hidden / (int)blockDim.x;
    if (hc <= 0 || hc > Q4E_MHC_MAX_HC || columns * (int)blockDim.x != hidden ||
        columns > Q4E_MHC_COLUMNS) {
        return;
    }
    const int t = blockIdx.x;
    const int lane = threadIdx.x;
    const int mixes = (2 + hc) * hc;
    const float* raw_t = raw + (unsigned long long)t * mixes;
    const int splits = hc * hidden / Q4E_BLOCK;
    const float* parts_t =
        partials != nullptr ? partials + (unsigned long long)t * splits * mixes : nullptr;
    const float* x = streams + (unsigned long long)t * hc * hidden;
    float held[Q4E_MHC_MAX_HC][Q4E_MHC_COLUMNS];
    float gate = 0.0f;
    float cell = 0.0f;
    if (blockIdx.y != 0) {
        Q4eMhcLane in = q4e_mhc_lane(scale, base, hc, 2 * hc, true);
        OPT_PDL_WAIT();
        q4e_mhc_lane_raw(in, raw_t, parts_t, splits, hc, 2 * hc, true);
        const float sum_squares = q4e_mhc_hold(x, hidden, hc, columns, held, scratch);
        if (threadIdx.x >= 32) {
            return;
        }
        q4e_mhc_warp_mixes(in, sum_squares, hc, hc * hidden, 2 * hc, true, iters, hc_eps,
                           norm_eps, gate, cell);
        float* o = coeffs + (unsigned long long)t * mixes;
        if (lane < 2 * hc) {
            o[lane] = gate;
        }
        if (lane < hc * hc) {
            o[2 * hc + lane] = cell;
        }
        return;
    }
    Q4eMhcLane in = q4e_mhc_lane(scale, base, hc, hc, false);
    __nv_bfloat16 norm[Q4E_MHC_COLUMNS];
#pragma unroll
    for (int k = 0; k < Q4E_MHC_COLUMNS; ++k) {
        norm[k] = k < columns ? weight[threadIdx.x + k * blockDim.x] : __float2bfloat16(0.0f);
    }
    OPT_PDL_WAIT();
    q4e_mhc_lane_raw(in, raw_t, parts_t, splits, hc, hc, false);
    const float sum_squares = q4e_mhc_hold(x, hidden, hc, columns, held, scratch);
    if (threadIdx.x < 32) {
        q4e_mhc_warp_mixes(in, sum_squares, hc, hc * hidden, hc, false, iters, hc_eps, norm_eps,
                           gate, cell);
        if (lane < hc) {
            pre[lane] = gate;
        }
    }
    __syncthreads();
    float p[Q4E_MHC_MAX_HC];
#pragma unroll
    for (int copy = 0; copy < Q4E_MHC_MAX_HC; ++copy) {
        p[copy] = copy < hc ? pre[copy] : 0.0f;
    }
    float mixed[Q4E_MHC_COLUMNS];
    float squares = 0.0f;
#pragma unroll
    for (int k = 0; k < Q4E_MHC_COLUMNS; ++k) {
        float accumulated = 0.0f;
#pragma unroll
        for (int copy = 0; copy < Q4E_MHC_MAX_HC; ++copy) {
            if (copy < hc) {
                accumulated = __fadd_rn(accumulated, __fmul_rn(p[copy], held[copy][k]));
            }
        }
        mixed[k] = accumulated;
        if (k < columns) {
            squares = __fadd_rn(squares, __fmul_rn(accumulated, accumulated));
        }
    }
    const float total = q4e_block_sum(squares, scratch);
    bool exact = true;
    float inverse = q4e_mhc_rms_scale<true>(total, (float)hidden, eps, exact);
    if (__any_sync(0xFFFFFFFFu, !exact)) {
        inverse = q4e_mhc_rms_scale<false>(total, (float)hidden, eps, exact);
    }
    float* y = out + (unsigned long long)t * hidden;
#pragma unroll
    for (int k = 0; k < Q4E_MHC_COLUMNS; ++k) {
        if (k < columns) {
            y[threadIdx.x + k * blockDim.x] =
                __fmul_rn(__fmul_rn(mixed[k], inverse), __bfloat162float(norm[k]));
        }
    }
}

// The streams after a sublayer, in place: copy k becomes post[k] y + sum_s comb[s][k] x[s], in
// mhc::hc_post's order. A thread owns one column of every copy, so reading the old streams and
// writing the new ones never crosses threads. grid (ceil(hidden / Q4E_BLOCK), batch).
//
// With world > 1, y is the TP all-reduce of the ranks' partials, Wire::F32's sum in ascending
// rank order from 0.0: this rank's own from branch, every peer's from its slot of slots
// ([world][batch][hidden], this rank's slot unread).
//
// With next_fn, the next mixer's [mixes][hc hidden] BF16 fn projection, a hidden of whole blocks
// and every split of that projection's tiled GEMV one 256-column round (the host's
// tiled_gemv_splits says when), the block also works out the GEMV's partials of the splits it
// holds. Split s = copy (hidden / Q4E_BLOCK) + blockIdx.x is the block's columns of `copy`: lane l
// takes its columns 8l..8l+8 through q4e_tg_fma8 from 0.0 and q4e_tg_fold folds the lanes, as the
// GEMV's round does. They land in next_partials ([batch][splits][mixes]) for q4e_mhc_in to add in
// split order from 0.0, as the GEMV's last block adds its own, so the projection is the GEMV's to
// the bit and the GEMV does not run. A warp takes the (copy, row) pairs warp + 8 i, whose weights
// it loads before the wait.
#define Q4E_MHC_PAIRS 12
static_assert(Q4E_BLOCK == Q4E_TG_THREADS && Q4E_BLOCK == 32 * Q4eTgBf16::kColumns,
              "a mixer block's columns of a copy are one round of the tiled BF16 GEMV");
extern "C" __global__ void q4e_mhc_post(float* __restrict__ streams,
                                        const float* __restrict__ branch,
                                        const float* __restrict__ coeffs, int hidden, int hc,
                                        const float* __restrict__ slots, int rank, int world,
                                        const __nv_bfloat16* __restrict__ next_fn,
                                        float* __restrict__ next_partials, int mixes) {
    OPT_PDL_LAUNCH();
    __shared__ __align__(16) float staged[Q4E_MHC_MAX_HC][Q4E_BLOCK];
    const int t = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const bool fused = next_fn != nullptr;
    const int pairs = hc * mixes;
    if (hc <= 0 || hc > Q4E_MHC_MAX_HC ||
        (fused && (blockDim.x != Q4E_BLOCK || hidden % Q4E_BLOCK != 0 ||
                   pairs > Q4E_TG_WARPS * Q4E_MHC_PAIRS))) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    uint4 word[Q4E_MHC_PAIRS];
#pragma unroll
    for (int i = 0; i < Q4E_MHC_PAIRS; ++i) {
        const int p = warp + Q4E_TG_WARPS * i;
        word[i] = make_uint4(0u, 0u, 0u, 0u);
        if (fused && p < pairs) {
            const int copy = p / mixes;
            const int row = p - copy * mixes;
            word[i] = q4e_ldg(reinterpret_cast<const uint4*>(
                next_fn + ((unsigned long long)row * hc + copy) * hidden + blockIdx.x * Q4E_BLOCK +
                8 * lane));
        }
    }
    OPT_PDL_WAIT();
    if (j >= hidden) {
        return;
    }
    float* x = streams + (unsigned long long)t * hc * hidden;
    const float* post = coeffs + (unsigned long long)t * (2 + hc) * hc + hc;
    const float* comb = post + hc;
    const unsigned long long at = (unsigned long long)t * hidden + j;
    float y = branch[at];
    if (world > 1) {
        const unsigned long long slot = (unsigned long long)gridDim.y * hidden;
        float sum = 0.0f;
        for (int r = 0; r < world; ++r) {
            sum = __fadd_rn(sum, r == rank ? y : slots[r * slot + at]);
        }
        y = sum;
    }
    float residual[Q4E_MHC_MAX_HC];
#pragma unroll
    for (int source = 0; source < Q4E_MHC_MAX_HC; ++source) {
        residual[source] = source < hc ? x[(unsigned long long)source * hidden + j] : 0.0f;
    }
#pragma unroll
    for (int copy = 0; copy < Q4E_MHC_MAX_HC; ++copy) {
        if (copy < hc) {
            float accumulated = __fmul_rn(post[copy], y);
#pragma unroll
            for (int source = 0; source < Q4E_MHC_MAX_HC; ++source) {
                if (source < hc) {
                    accumulated =
                        __fadd_rn(accumulated, __fmul_rn(comb[source * hc + copy], residual[source]));
                }
            }
            x[(unsigned long long)copy * hidden + j] = accumulated;
            staged[copy][threadIdx.x] = accumulated;
        }
    }
    if (!fused) {
        return;
    }
    __syncthreads();
    const int splits = hc * hidden / Q4E_BLOCK;
    float* out = next_partials + (unsigned long long)t * splits * mixes;
#pragma unroll
    for (int i = 0; i < Q4E_MHC_PAIRS; ++i) {
        const int p = warp + Q4E_TG_WARPS * i;
        if (p < pairs) {
            const int copy = p / mixes;
            const int row = p - copy * mixes;
            const float4* held = reinterpret_cast<const float4*>(&staged[copy][8 * lane]);
            const float sum = q4e_tg_fold(q4e_tg_fma8(word[i], held[0], held[1], 0.0f));
            if (lane == 0) {
                out[(unsigned long long)(copy * (hidden / Q4E_BLOCK) + blockIdx.x) * mixes + row] =
                    sum;
            }
        }
    }
}

// The two-shot exchange's sum: this rank's chunk of the TP all-reduce in place, q4e_mhc_post's
// Wire::F32 fold in ascending rank order from 0.0, this rank's own partial from `own` and every
// peer's from its slot of `slots` ([world][count], this rank's slot unread). The chunks are then
// shared, so every element is the one-shot exchange's sum. count is a multiple of 4. grid
// (ceil(count / 4 / Q4E_BLOCK)).
extern "C" __global__ void q4e_sum_slots(float* __restrict__ own, const float* __restrict__ slots,
                                         long long count, int rank, int world) {
    OPT_PDL_ENTRY();
    const long long at = 4 * ((long long)blockIdx.x * blockDim.x + threadIdx.x);
    if (at >= count) {
        return;
    }
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int r = 0; r < world; ++r) {
        const float4 term = *reinterpret_cast<const float4*>(r == rank ? own + at
                                                                        : slots + r * count + at);
        sum = make_float4(__fadd_rn(sum.x, term.x), __fadd_rn(sum.y, term.y),
                          __fadd_rn(sum.z, term.z), __fadd_rn(sum.w, term.w));
    }
    *reinterpret_cast<float4*>(own + at) = sum;
}

// GLM's clamped SwiGLU over rows of [gate (width) | up (width)]: silu(min(gate, limit)) *
// clamp(up, -limit, limit), with linalg's silu. grid (ceil(width / Q4E_BLOCK), rows).
extern "C" __global__ void q4e_swiglu_limited(const float* __restrict__ gate_up,
                                              float* __restrict__ out, int width, float limit) {
    OPT_PDL_ENTRY();
    const int row = blockIdx.y;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width) {
        return;
    }
    const float* source = gate_up + (unsigned long long)row * 2 * width;
    const float gate = fminf(source[index], limit);
    const float up = fminf(fmaxf(source[width + index], -limit), limit);
    out[(unsigned long long)row * width + index] =
        __fmul_rn(__fmul_rn(gate, q4e_sigmoid_ref(gate)), up);
}

// GLM's sigmoid noaux_tc routing over the router's logits ([batch][experts]): each score is
// sigmoid(logit); the top-k is taken on score + bias, the lower expert on a tie; the weights are
// the plain scores, renormalised by their sum (in pick order) plus 1e-20 when `renormalise` and
// topk > 1, then times `scaling` (topk::noaux_top_k). The slots are written in ascending expert
// order, the order the host combines them in. grid (batch), one warp: lane l holds experts
// l, l + 32, ..., and each pick is a butterfly over the lanes' best untaken experts, so it costs
// no block barrier.
extern "C" __global__ void q4e_noaux_router(const float* __restrict__ logits,
                                            const float* __restrict__ bias,
                                            int* __restrict__ indices, float* __restrict__ weights,
                                            int experts, int topk, int renormalise,
                                            float scaling) {
    OPT_PDL_ENTRY();
    constexpr int kEach = Q4E_MAX_EXPERTS / 32;
    static_assert(kEach <= 32, "a lane's taken experts are one word's bits");
    if (experts <= 0 || experts > Q4E_MAX_EXPERTS || topk <= 0 || topk > Q4E_MAX_TOPK ||
        topk > experts || threadIdx.x >= 32) {
        return;
    }
    const int b = blockIdx.x;
    const int lane = threadIdx.x;
    // Every load first, so they are in flight together.
    float score[kEach];
    float signal[kEach];
#pragma unroll
    for (int k = 0; k < kEach; ++k) {
        const int expert = lane + 32 * k;
        score[k] = expert < experts ? logits[(unsigned long long)b * experts + expert] : 0.0f;
        signal[k] = expert < experts ? bias[expert] : 0.0f;
    }
#pragma unroll
    for (int k = 0; k < kEach; ++k) {
        if (lane + 32 * k < experts) {
            score[k] = q4e_sigmoid_ref(score[k]);
            signal[k] = __fadd_rn(score[k], signal[k]);
        }
    }
    unsigned int taken = 0;
    int picked[Q4E_MAX_TOPK];
    float chosen[Q4E_MAX_TOPK];
    for (int slot = 0; slot < topk; ++slot) {
        float best_value = -INFINITY;
        int best = 0x7FFFFFFF;
#pragma unroll
        for (int k = 0; k < kEach; ++k) {
            if (lane + 32 * k < experts && (taken >> k & 1u) == 0) {
                q4e_take_better(signal[k], lane + 32 * k, &best_value, &best);
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            q4e_take_better(__shfl_xor_sync(0xFFFFFFFFu, best_value, offset),
                            __shfl_xor_sync(0xFFFFFFFFu, best, offset), &best_value, &best);
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < kEach; ++k) {
            if (lane + 32 * k == best) {
                taken |= 1u << k;
                mine = score[k];
            }
        }
        picked[slot] = best;
        chosen[slot] = __shfl_sync(0xFFFFFFFFu, mine, best & 31);
    }
    if (lane == 0) {
        float total = 0.0f;
        for (int slot = 0; slot < topk; ++slot) {
            total = __fadd_rn(total, chosen[slot]);
        }
        const float denominator = __fadd_rn(total, 1e-20f);
        for (int slot = 0; slot < topk; ++slot) {
            const float weight = renormalise && topk > 1 ? chosen[slot] / denominator : chosen[slot];
            chosen[slot] = __fmul_rn(weight, scaling);
        }
        // Ascending expert order; the experts are distinct, so the order is total.
        for (int slot = 1; slot < topk; ++slot) {
            const int expert = picked[slot];
            const float weight = chosen[slot];
            int at = slot;
            for (; at > 0 && picked[at - 1] > expert; --at) {
                picked[at] = picked[at - 1];
                chosen[at] = chosen[at - 1];
            }
            picked[at] = expert;
            chosen[at] = weight;
        }
        for (int slot = 0; slot < topk; ++slot) {
            indices[(unsigned long long)b * topk + slot] = picked[slot];
            weights[(unsigned long long)b * topk + slot] = chosen[slot];
        }
    }
}

// LayerNorm with a BF16 weight and bias over rows of `columns` (linalg::layer_norm: the biased
// variance, eps under the root): row r of `x` into row r of `out`. grid (rows), Q4E_BLOCK threads.
extern "C" __global__ void q4e_layer_norm(const float* __restrict__ x,
                                          const __nv_bfloat16* __restrict__ weight,
                                          const __nv_bfloat16* __restrict__ bias,
                                          float* __restrict__ out, int columns, float eps) {
    OPT_PDL_ENTRY();
    __shared__ float scratch[Q4E_BLOCK];
    const float* source = x + (unsigned long long)blockIdx.x * columns;
    float* destination = out + (unsigned long long)blockIdx.x * columns;
    float sum = 0.0f;
    for (int index = threadIdx.x; index < columns; index += blockDim.x) {
        sum = __fadd_rn(sum, source[index]);
    }
    const float mean = q4e_block_sum(sum, scratch) / (float)columns;
    float squares = 0.0f;
    for (int index = threadIdx.x; index < columns; index += blockDim.x) {
        const float delta = __fadd_rn(source[index], -mean);
        squares = __fadd_rn(squares, __fmul_rn(delta, delta));
    }
    const float variance = q4e_block_sum(squares, scratch) / (float)columns;
    const float scale = 1.0f / sqrtf(__fadd_rn(variance, eps));
    for (int index = threadIdx.x; index < columns; index += blockDim.x) {
        const float delta = __fadd_rn(source[index], -mean);
        destination[index] = __fadd_rn(
            __fmul_rn(__fmul_rn(delta, scale), __bfloat162float(weight[index])),
            __bfloat162float(bias[index]));
    }
}

// The longest pool GLM's indexer softmaxes over in registers.
#define Q4E_GLM_MAX_POOL 16

// GLM's k-pool: when the token at `position` completes a pool of `pool` tokens, channel c of the
// pooled key is the softmax over the pool's tokens of gate + ape (BF16 [pool][dim]) weighting
// their keys, summed in token order (forward.rs's Indexer::select). `keys` and `gates` are the
// layer's per-position caches ([positions][dim]); the pooled key lands at row
// (position + 1) / pool - 1 of `pooled`. A token that completes no pool is a no-op, so a step
// launches this whatever its position. grid (positions): block b pools for `position` + b, threads
// over the channels; a pool reads only cached keys and gates, so a run's pools are one launch.
extern "C" __global__ void q4e_glm_pool(const float* __restrict__ keys,
                                        const float* __restrict__ gates,
                                        const __nv_bfloat16* __restrict__ ape,
                                        float* __restrict__ pooled, int dim, int pool,
                                        int position) {
    OPT_PDL_ENTRY();
    position += blockIdx.x;
    if (pool <= 0 || pool > Q4E_GLM_MAX_POOL || (position + 1) % pool != 0) {
        return;
    }
    const int first = position + 1 - pool;
    float* destination = pooled + (unsigned long long)((position + 1) / pool - 1) * dim;
    for (int channel = threadIdx.x; channel < dim; channel += blockDim.x) {
        float weight[Q4E_GLM_MAX_POOL];
        float maximum = -INFINITY;
#pragma unroll
        for (int offset = 0; offset < Q4E_GLM_MAX_POOL; ++offset) {
            if (offset < pool) {
                weight[offset] =
                    __fadd_rn(gates[(unsigned long long)(first + offset) * dim + channel],
                              __bfloat162float(ape[offset * dim + channel]));
                maximum = fmaxf(maximum, weight[offset]);
            }
        }
        float total = 0.0f;
#pragma unroll
        for (int offset = 0; offset < Q4E_GLM_MAX_POOL; ++offset) {
            if (offset < pool) {
                weight[offset] = expf(weight[offset] - maximum);
                total = __fadd_rn(total, weight[offset]);
            }
        }
        float accumulated = 0.0f;
#pragma unroll
        for (int offset = 0; offset < Q4E_GLM_MAX_POOL; ++offset) {
            if (offset < pool) {
                const float share = !isfinite(maximum) ? 0.0f
                                    : total > 0.0f    ? weight[offset] / total
                                                      : weight[offset];
                accumulated = __fadd_rn(
                    accumulated,
                    __fmul_rn(share, keys[(unsigned long long)(first + offset) * dim + channel]));
            }
        }
        destination[channel] = accumulated;
    }
}

#define Q4E_GLM_MAX_INDEX_HEADS 32
#define Q4E_GLM_MAX_INDEX_DIM 128

// GLM's indexer score of every complete pool for the query at `position`:
// sum_h w_h max(q_h . k_pool * score_scale, 0) with w_h = head_weights[h] * head_scale, summed in
// head order (forward.rs's Indexer::select). The query heads are staged in shared memory and a
// warp scores a pool, Q4E_QSA_TILE pools a block, the layout q4e_glm_pick reads.
// grid (ceil(pools / Q4E_QSA_TILE) of the last row, rows), Q4E_BLOCK threads: blockIdx.y is the
// row at `position` + y, its query heads and head weights the y-th of the run's and its scores
// `score_stride` floats on from the previous row's.
extern "C" __global__ void q4e_glm_index_scores(const float* __restrict__ query,
                                                const float* __restrict__ head_weights,
                                                const float* __restrict__ pooled,
                                                float* __restrict__ scores, int position,
                                                int heads, int dim, int pool, float score_scale,
                                                float head_scale, int score_stride) {
    OPT_PDL_ENTRY();
    __shared__ float staged[Q4E_GLM_MAX_INDEX_HEADS * Q4E_GLM_MAX_INDEX_DIM];
    __shared__ float weight[Q4E_GLM_MAX_INDEX_HEADS];
    const int row = blockIdx.y;
    position += row;
    query += (unsigned long long)row * heads * dim;
    head_weights += (unsigned long long)row * heads;
    scores += (unsigned long long)row * score_stride;
    const int pools = (position + 1) / pool;
    const int first = blockIdx.x * Q4E_QSA_TILE;
    if (first >= pools || heads > Q4E_GLM_MAX_INDEX_HEADS || dim > Q4E_GLM_MAX_INDEX_DIM) {
        return;
    }
    for (int index = threadIdx.x; index < heads * dim; index += blockDim.x) {
        staged[index] = query[index];
    }
    for (int head = threadIdx.x; head < heads; head += blockDim.x) {
        weight[head] = __fmul_rn(head_weights[head], head_scale);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int warps = blockDim.x >> 5;
    const int last = min(first + Q4E_QSA_TILE, pools);
    for (int block = first + (threadIdx.x >> 5); block < last; block += warps) {
        const float* key_row = pooled + (unsigned long long)block * dim;
        float key[Q4E_GLM_MAX_INDEX_DIM / 32];
#pragma unroll
        for (int step = 0; step < Q4E_GLM_MAX_INDEX_DIM / 32; ++step) {
            const int channel = lane + 32 * step;
            key[step] = channel < dim ? key_row[channel] : 0.0f;
        }
        float score = 0.0f;
        for (int head = 0; head < heads; ++head) {
            float dot = 0.0f;
#pragma unroll
            for (int step = 0; step < Q4E_GLM_MAX_INDEX_DIM / 32; ++step) {
                const int channel = lane + 32 * step;
                if (channel < dim) {
                    dot = fmaf(staged[head * dim + channel], key[step], dot);
                }
            }
            dot = q4e_warp_sum(dot);
            score = __fadd_rn(score,
                              __fmul_rn(weight[head], fmaxf(__fmul_rn(dot, score_scale), 0.0f)));
        }
        if (lane == 0) {
            scores[block] = score;
        }
    }
}

// The threads below this one whose `flag` is set, and the block's count in `total`: a ballot a
// warp and the warps' counts in `counts` (one a warp). Every thread of the block calls it.
__device__ __forceinline__ int q4e_block_rank(bool flag, int* counts, int* total) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const unsigned int ballot = __ballot_sync(0xFFFFFFFFu, flag);
    if (lane == 0) {
        counts[warp] = __popc(ballot);
    }
    __syncthreads();
    int before = __popc(ballot & ((1u << lane) - 1u));
    int all = 0;
    for (int other = 0; other < (int)(blockDim.x >> 5); ++other) {
        before += other < warp ? counts[other] : 0;
        all += counts[other];
    }
    __syncthreads();
    *total = all;
    return before;
}

// The routed experts' block-scaled FP8 GEMV through the tiled body, over the router's [batch]
// [topk] picks: `table` holds each expert's E4M3 [rows, columns] weight and f32 [rows / 128]
// [columns / 128] scales ([experts][2] device pointers). Pair p reads x row p / topk (`x_per_slot`
// 0: gate_up on the token's normed input) or row p (1: down on the pair's activation), and writes
// y row p, times its routing weight when `slot_weights` is set - the product the host's combine
// rounds before it adds. `quantise` is q4e_gemv_tiled_e4m3_block's.
//
// The pairs that pick one expert share its weights: blockIdx.z is the z-th distinct expert in
// pair order, and its blocks run every pair that picked it (a token picks an expert once, so at
// most Q4E_MAX_BATCH) as one batch of the tiled body, so a step reads each expert once. A pair's
// sum is the chain a batch of one gives it. grid (ceil(rows / 32), 1, batch * topk), Q4E_TG_THREADS
// threads, batch * topk at most Q4E_TG_THREADS; blocks past the distinct experts return.
struct Q4eTgPairStore {
    float* y;
    int rows;
    const int* pairs;
    const float* slot_weights;
    int count;
    __device__ void operator()(int tile, int warp, int r, int t, float sum) const {
        const int row = tile * Q4E_TG_TILE + warp * Q4E_TG_ROWS + r;
        if (t < count && row < rows) {
            const int pair = pairs[t];
            y[(unsigned long long)pair * rows + row] =
                slot_weights != nullptr ? __fmul_rn(sum, slot_weights[pair]) : sum;
        }
    }
};

extern "C" __global__ void __launch_bounds__(Q4E_TG_THREADS, 3)
    q4e_expert_gemv_e4m3_block(const unsigned long long* __restrict__ table,
                               const int* __restrict__ indices,
                               const float* __restrict__ slot_weights,
                               const float* __restrict__ x, float* __restrict__ y, int rows,
                               int columns, int topk, int x_per_slot, int quantise) {
    OPT_PDL_ENTRY();
    __shared__ int first_of[Q4E_MAX_EXPERTS];
    __shared__ int counts[Q4E_TG_WARPS];
    __shared__ int chosen;
    __shared__ int group_pairs[Q4E_MAX_BATCH];
    __shared__ int group_rows[Q4E_MAX_BATCH];
    const int pairs = gridDim.z;
    if (columns % Q4E_FP8_BLOCK != 0 || topk <= 0 || pairs > Q4E_TG_THREADS) {
        return;
    }
    for (int expert = threadIdx.x; expert < Q4E_MAX_EXPERTS; expert += blockDim.x) {
        first_of[expert] = 0x7FFFFFFF;
    }
    const int pair = threadIdx.x;
    int expert = pair < pairs ? indices[pair] : -1;
    expert = expert < Q4E_MAX_EXPERTS ? expert : -1;
    __syncthreads();
    if (expert >= 0) {
        atomicMin(&first_of[expert], pair);
    }
    __syncthreads();
    int groups;
    const bool opens = expert >= 0 && first_of[expert] == pair;
    const int group = q4e_block_rank(opens, counts, &groups);
    if (opens && group == (int)blockIdx.z) {
        chosen = expert;
    }
    __syncthreads();
    if ((int)blockIdx.z >= groups) {
        return;
    }
    const int mine = chosen;
    int count;
    const bool member = expert == mine;
    const int slot = q4e_block_rank(member, counts, &count);
    if (member && slot < Q4E_MAX_BATCH) {
        group_pairs[slot] = pair;
        group_rows[slot] = x_per_slot ? pair : pair / topk;
    }
    __syncthreads();
    const unsigned char* w = reinterpret_cast<const unsigned char*>(table[2 * mine]);
    const float* scales = reinterpret_cast<const float*>(table[2 * mine + 1]);
    count = min(count, Q4E_MAX_BATCH);
    q4e_tiled_gemv<Q4eTgE4m3Block>(Q4eTgBlockRows{{w, rows, columns}, scales, quantise},
                                   Q4eTgPairStore{y, rows, group_pairs, slot_weights, count}, x,
                                   columns, count, nullptr, nullptr, group_rows);
}

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
// The Qwen attention family (grouped-query attention, the indexer's pool/score/pick kernels and
// the KV-cache append) lives in its own translation unit, kernels/qwen4_attn_sm121.cu, and its
// own cubin. The decode/token routed-expert family (router, permutation, NVFP4 and FP8 expert
// passes and the combine) lives in kernels/qwen4_moe_sm121.cu, and its own cubin. The
// GLM-5.3-Flash decode/token family (hyper-connection mixers, clamped SwiGLU, noaux router,
// LayerNorm, indexer pool/score/pick and the absorbed MLA) and the tiled decode GEMV wrappers
// GLM loads (the fused-parts BF16 one, the block-scaled FP8 one, the routed-expert one and the L2
// prefetch) live in kernels/glm_decode_sm121.cu, and its own cubin.
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
//   - The block-scaled FP8 form of the tiled GEMV (E4M3 weights with one f32 weight_scale_inv a
//     128 x 128 block, activations quantised per 128 columns) reads the format of Z.ai's
//     GLM-5.3-Flash release (MIT License), the one Hugging Face Transformers' fine-grained FP8
//     integration reads (Apache-2.0); no source is copied. Its wrappers (the fused-parts BF16,
//     block-scaled FP8 and routed-expert entries) and the decode/token kernels (mHC mixers, the
//     indexer and the absorbed MLA) are in
//     kernels/glm_decode_sm121.cu, which carries its own inline copy of the tiled body; this file
//     keeps the copy Qwen instantiates for q4e_gemv_tiled_bf16 and q4e_gemv_tiled_e4m3. The
//     mixers are mHC (Xie et al., arXiv:2512.24880) and the indexer is DeepSeek Sparse
//     Attention's (DeepSeek-V3.2, arXiv:2512.02556) with GLM-5.3-Flash's k-pool.
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
#include "q4e_common_sm121.cuh"
#include "shared_swiglu_sm121.cuh"
#include "q4e_gemv_sm121.cuh"

#define Q4E_BLOCK 256

// Static shared-memory and register-tiling bounds.
//
// The driver launch path in sparkflash-kernels passes a dynamic shared-memory size of zero, so
// every shared buffer is a fixed array sized for the largest geometry this pack can serve, with a
// guard that makes an oversized request a no-op rather than a corruption. The host side checks the
// same bounds and refuses first.
#define Q4E_MAX_LOWRANK 1024
// The widest linear-attention head output q4e_gdn_recur holds (linear_value_head_dim). It is
// not the MoE intermediate: the routed experts staged their width dynamically and carry no cap
// here. The served value head is 128, so 1024 is a superset of every geometry the scan takes;
// a wider head would be a scan geometry, not a router/expert bound.
#define Q4E_MAX_INTERMEDIATE 1024
#define Q4E_MAX_SELECTED 2304
#define Q4E_MAX_TOPK 64

// The decoders are the shared header's; these aliases keep the kernel bodies readable.
#define q4e_e4m3 spark_fp8_e4m3_decode
#define q4e_e2m1 spark_fp4_e2m1_decode


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

// Batched BF16 row dot over R weight rows of one warp: reads each weight chunk once and each
// activation chunk once for all R rows, accumulating across batch token rows in registers
// (batch <= Q4E_MAX_BATCH).
//
// Each (row, token) sum is grouped exactly as q4e_row_dot_bf16 groups it - four chunks added
// together before they meet the total, then a pair, then one at a time - so a row of a batch is
// bit-identical to the same row alone, and to the same row in a warp of any other R. A
// speculative verify has to compute as the one-token steps it stands in for, and a different
// association moved its logits by ~1e-6, enough to flip a near-tie and make greedy output depend
// on whether a drafter ran. A row the 16-byte walk cannot read takes the one-row column walk, as
// it would alone.
template <int R>
__device__ __forceinline__ void q4e_rows_dot_bf16_batch(const __nv_bfloat16* const* rows,
                                                        const float* __restrict__ x,
                                                        int columns, int lane, int batch,
                                                        float (*totals)[Q4E_MAX_BATCH]) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
            totals[r][t] = 0.0f;
        }
    }
    bool vector = (columns & 7) == 0 && ((reinterpret_cast<unsigned long long>(x) & 15u) == 0);
#pragma unroll
    for (int r = 0; r < R; ++r) {
        vector = vector && ((reinterpret_cast<unsigned long long>(rows[r]) & 15u) == 0);
    }
    if (vector) {
        const int chunks = columns >> 3;
        int chunk = lane;
        for (; chunk + 96 < chunks; chunk += 128) {
            uint4 a[R], b[R], c[R], d[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                a[r] = q4e_ldg(reinterpret_cast<const uint4*>(rows[r] + chunk * 8));
                b[r] = q4e_ldg(reinterpret_cast<const uint4*>(rows[r] + (chunk + 32) * 8));
                c[r] = q4e_ldg(reinterpret_cast<const uint4*>(rows[r] + (chunk + 64) * 8));
                d[r] = q4e_ldg(reinterpret_cast<const uint4*>(rows[r] + (chunk + 96) * 8));
            }
#pragma unroll
            for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
                if (t < batch) {
                    const float* xt = x + (unsigned long long)t * columns;
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        totals[r][t] += q4e_dot_chunk(a[r], xt, chunk * 8) +
                                        q4e_dot_chunk(b[r], xt, (chunk + 32) * 8) +
                                        q4e_dot_chunk(c[r], xt, (chunk + 64) * 8) +
                                        q4e_dot_chunk(d[r], xt, (chunk + 96) * 8);
                    }
                }
            }
        }
        if (chunk + 32 < chunks) {
            uint4 a[R], b[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                a[r] = q4e_ldg(reinterpret_cast<const uint4*>(rows[r] + chunk * 8));
                b[r] = q4e_ldg(reinterpret_cast<const uint4*>(rows[r] + (chunk + 32) * 8));
            }
#pragma unroll
            for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
                if (t < batch) {
                    const float* xt = x + (unsigned long long)t * columns;
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        totals[r][t] += q4e_dot_chunk(a[r], xt, chunk * 8);
                        totals[r][t] += q4e_dot_chunk(b[r], xt, (chunk + 32) * 8);
                    }
                }
            }
            chunk += 64;
        }
        for (; chunk < chunks; chunk += 32) {
            uint4 value[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                value[r] = q4e_ldg(reinterpret_cast<const uint4*>(rows[r] + chunk * 8));
            }
#pragma unroll
            for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
                if (t < batch) {
                    const float* xt = x + (unsigned long long)t * columns;
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        totals[r][t] += q4e_dot_chunk(value[r], xt, chunk * 8);
                    }
                }
            }
        }
        return;
    }
    if constexpr (R > 1) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            q4e_rows_dot_bf16_batch<1>(rows + r, x, columns, lane, batch, totals + r);
        }
    } else {
#pragma unroll 1
        for (int column = lane; column < columns; column += 32) {
            const float w = __bfloat162float(rows[0][column]);
#pragma unroll
            for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
                if (t < batch) {
                    totals[0][t] += w * x[(unsigned long long)t * columns + column];
                }
            }
        }
    }
}

// q4e_rows_dot_bf16_batch over one row.
__device__ __forceinline__ void q4e_row_dot_bf16_batch(
    const __nv_bfloat16* __restrict__ row, const float* __restrict__ x, int columns, int lane,
    int batch, float totals[Q4E_MAX_BATCH]) {
    const __nv_bfloat16* rows[1] = {row};
    q4e_rows_dot_bf16_batch<1>(rows, x, columns, lane, batch,
                               reinterpret_cast<float (*)[Q4E_MAX_BATCH]>(totals));
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

// Block argmax over (value, index) pairs, with the same tie-break as the host sort.


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
    // Two rows a warp: each activation load feeds both rows, which halves the L1 reads a weight
    // byte costs once several tokens share the pass (eight tokens' activations are sixteen times
    // the weight bytes). A block covers sixteen rows, so the grid's upper half has none.
    for (int row = (blockIdx.x * warps + warp) * 2; row < rows; row += stride * 2) {
        const bool second = row + 1 < rows;
        const __nv_bfloat16* pair[2] = {
            w + (unsigned long long)row * columns,
            w + (unsigned long long)(second ? row + 1 : row) * columns};
        float totals[2][Q4E_MAX_BATCH];
        q4e_rows_dot_bf16_batch<2>(pair, x, columns, lane, batch, totals);
#pragma unroll
        for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
            if (t < batch) {
                const float first_sum = q4e_warp_sum(totals[0][t]);
                const float second_sum = q4e_warp_sum(totals[1][t]);
                if (lane == 0) {
                    y[(unsigned long long)t * rows + row] = first_sum;
                    if (second) {
                        y[(unsigned long long)t * rows + row + 1] = second_sum;
                    }
                }
            }
        }
    }
}

// Where q4e_gemv4_bf16's row `global_row` of the four matrices stacked lands: its weight row,
// its output, its row there and that output's row count.
__device__ __forceinline__ void q4e_gemv4_locate(
    int global_row, const __nv_bfloat16* __restrict__ w0, const __nv_bfloat16* __restrict__ w1,
    const __nv_bfloat16* __restrict__ w2, const __nv_bfloat16* __restrict__ w3, float* y0,
    float* y1, float* y2, float* y3, int r0, int r1, int r2, int r3, int columns,
    const __nv_bfloat16** w_row, float** out, int* local_row, int* out_rows) {
    const int r01 = r0 + r1;
    const int r012 = r01 + r2;
    if (global_row < r0) {
        *local_row = global_row;
        *out_rows = r0;
        *w_row = w0 + (unsigned long long)*local_row * columns;
        *out = y0;
    } else if (global_row < r01) {
        *local_row = global_row - r0;
        *out_rows = r1;
        *w_row = w1 + (unsigned long long)*local_row * columns;
        *out = y1;
    } else if (global_row < r012) {
        *local_row = global_row - r01;
        *out_rows = r2;
        *w_row = w2 + (unsigned long long)*local_row * columns;
        *out = y2;
    } else {
        *local_row = global_row - r012;
        *out_rows = r3;
        *w_row = w3 + (unsigned long long)*local_row * columns;
        *out = y3;
    }
}

// Fused four-output BF16 projection sharing the same input x ([batch, columns]):
//   - Linear layers: (in_qkv, in_z, in_b, in_a)
//   - Sparse layers: (q_proj, k_proj, v_proj, index_qk)
// A batch takes two rows a warp, as q4e_gemv_bf16 does; the two may sit in different matrices.
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
    const int total_rows = r0 + r1 + r2 + r3;
    if (batch <= 1) {
        for (int global_row = blockIdx.x * warps + warp; global_row < total_rows;
             global_row += stride) {
            const __nv_bfloat16* w_row;
            float* out;
            int local_row;
            int out_rows;
            q4e_gemv4_locate(global_row, w0, w1, w2, w3, y0, y1, y2, y3, r0, r1, r2, r3, columns,
                             &w_row, &out, &local_row, &out_rows);
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
    for (int global_row = (blockIdx.x * warps + warp) * 2; global_row < total_rows;
         global_row += stride * 2) {
        const bool second = global_row + 1 < total_rows;
        const __nv_bfloat16* pair[2];
        float* out[2];
        int local_row[2];
        int out_rows[2];
        q4e_gemv4_locate(global_row, w0, w1, w2, w3, y0, y1, y2, y3, r0, r1, r2, r3, columns,
                         &pair[0], &out[0], &local_row[0], &out_rows[0]);
        q4e_gemv4_locate(second ? global_row + 1 : global_row, w0, w1, w2, w3, y0, y1, y2, y3, r0,
                         r1, r2, r3, columns, &pair[1], &out[1], &local_row[1], &out_rows[1]);
        float totals[2][Q4E_MAX_BATCH];
        q4e_rows_dot_bf16_batch<2>(pair, x, columns, lane, batch, totals);
#pragma unroll
        for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
            if (t < batch) {
                const float first_sum = q4e_warp_sum(totals[0][t]);
                const float second_sum = q4e_warp_sum(totals[1][t]);
                if (lane == 0) {
                    out[0][(unsigned long long)t * out_rows[0] + local_row[0]] = first_sum;
                    if (second) {
                        out[1][(unsigned long long)t * out_rows[1] + local_row[1]] = second_sum;
                    }
                }
            }
        }
    }
}

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

// y = W x for the Qwen3.5/3.6 FP8 release's dense projections: W is E4M3 with one BF16 inverse
// scale a 128 x 128 tile (weight_scale_inv, [ceil(rows / 128)][columns / 128]), and x stays f32 -
// the W8A16 product of the dequantised weight, as the host oracle and the FP8 experts' token
// kernels compute it, not the checkpoint's dynamic activation scheme. Grid (ceil(rows / 32),
// splits), Q4E_TG_THREADS threads, columns a multiple of 128 and the operands 16-byte aligned.
extern "C" __global__ void __launch_bounds__(Q4E_TG_THREADS, 3)
    q4e_gemv_tiled_e4m3_tile(const unsigned char* __restrict__ w,
                             const __nv_bfloat16* __restrict__ scales,
                             const float* __restrict__ x, float* __restrict__ y, int rows,
                             int columns, int batch, float* __restrict__ partials,
                             unsigned int* __restrict__ counters) {
    OPT_PDL_LAUNCH();
    if (batch > Q4E_MAX_BATCH || columns % Q4E_FP8_BLOCK != 0) {
        OPT_PDL_WAIT();
        return;
    }
    q4e_tiled_gemv<Q4eTgE4m3Block>(
        Q4eTgBlockRowsOf<__nv_bfloat16>{{w, rows, columns}, scales, 0},
        Q4eTgStore{y, nullptr, rows, batch}, x, columns, batch, partials, counters);
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
    int hc;
    __device__ const __nv_bfloat16* operator()(int tile, int warp, int r) const {
        const int column = tile * Q4E_TG_WARPS + warp;
        if (column >= hidden || r >= hc) {
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
    int hc;
    __device__ void operator()(int tile, int warp, int r, int t, float sum) const {
        const int column = tile * Q4E_TG_WARPS + warp;
        const bool owner = r == 0 && t < batch && column < hidden;
        // Summed from zero a stream at a time, as q4e_hyper_mix sums the prefill's streams.
        float accumulator = 0.0f;
        for (int stream = 0; stream < hc; ++stream) {
            const float mix = __shfl_sync(0xFFFFFFFFu, sum, stream * 8 + t);
            if (owner) {
                accumulator += (1.0f / (1.0f + __expf(-mix))) *
                               normed[(unsigned long long)t * hc * hidden +
                                      (unsigned long long)stream * hidden + column];
            }
        }
        if (owner) {
            mixed[(unsigned long long)t * hidden + column] = accumulator / (float)hc;
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
    if (batch > Q4E_MAX_BATCH || hc <= 0 || hc > Q4E_TG_ROWS || gridDim.y != 1) {
        OPT_PDL_WAIT();
        return;
    }
    q4e_tiled_gemv<Q4eTgBf16>(Q4eTgHyperUpRows{up, hidden, lowrank, hc},
                              Q4eTgHyperUpFinish{normed, mixed, hidden, batch, hc}, low, lowrank,
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
// the mHC post mixer folds it: in ascending rank order from 0.0, this rank's own from branch and every
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
        out[index] = shared_swiglu_value<false, false>(gate[index], up[index], 0.0f);
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
        q4e_rope_pair(base, base, index, half, rotary, position_f, theta);
    }
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

// The two-shot exchange's sum: this rank's chunk of the TP all-reduce in place, the mHC post's
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

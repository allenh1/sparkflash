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

// Qwen3.8-Flash-Next (Qwen4-Exp) routed-expert token kernels, TP=1.
//
// This is the decode/token MoE family that used to sit in qwen4_exp_sm121.cu. It is its own
// translation unit and cubin (Image::Qwen4Moe) so the token image carries only the attention,
// embedding, norm and dense kernels. The shared attention and dense symbols stay in
// qwen4_exp_sm121.cu; the arithmetic here is unchanged.
//
// The family is: router and top-k, the routed-slot permutation and its run bounds, the NVFP4
// gate/up and down passes, the MTP layer's 128x128-block-scaled FP8 gate/up and down passes, and
// the combine that sums the routed partials with the sigmoid-gated shared expert.
//
// Attribution (as in qwen4_exp_sm121.cu):
//   - Model architecture (Qwen4ExpForConditionalGeneration: 512-expert top-10 MoE + sigmoid-gated
//     shared expert) originates from Qwen/Qwen3.8-Flash-Next by the Qwen team at Alibaba Cloud
//     and the vLLM reference implementation (https://github.com/vllm-project/vllm, Apache-2.0).
//   - NVFP4 W4A4 block-scaled quantization layout (packed E2M1 nibbles, per-16-column FP8 E4M3
//     block scales, FP32 weight_scale_2) originates from NVIDIA's
//     Qwen3.8-Flash-Next-NVFP4 (NVIDIA TensorRT ModelOpt 0.46).
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

#include "prefetch_sm121.cuh"
#include "quant_decode_sm121.cuh"
#include "opt_pdl_sm121.cuh"
#include "q4e_common_sm121.cuh"
#include "q4e_gemv_sm121.cuh"

#define Q4E_BLOCK 256

// Padding each 32-float group by 1 float (stride 33, coprime to 32) eliminates the 32-way
// shared-memory bank conflict when a warp's 32 lanes read 32-float chunks simultaneously. The
// activation staging itself is dynamic shared memory, sized from the projection's width by the
// host launch, so a geometry is no longer capped at a compile-time row width.
#define Q4E_SMEM_STRIDE 33
#define Q4E_MAX_EXPERTS 512
#define Q4E_MAX_TOPK 64

// The decoders are the shared header's; these aliases keep the kernel bodies readable.
#define q4e_e4m3 spark_fp8_e4m3_decode
#define q4e_e2m1 spark_fp4_e2m1_decode

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


// Block argmax over (value, index) pairs, with the same tie-break as the host sort.

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
    float u_high, const float2* lut, const float* base, int count, int pad, float* g_total,
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
                const float* x_chunk = base + m * pad + word * 8;
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
                                                           const float* base, int count, int pad,
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
                const float* x_chunk = base + m * pad + word * 8;
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
    const float2* lut_pair, const float* base, int columns, int lane, int count, int pad,
    float* g_out, float* u_out) {
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
                                            lut_pair, base + chunk * Q4E_SMEM_STRIDE, count, pad,
                                            g_out, u_out);
        }
        return;
    }
    for (int m = 0; m < count; ++m) {
        g_out[m] = q4e_nvfp4_row_dot(g_row, g_scale, g_scale2, lut, base + m * pad, columns, lane);
        u_out[m] = q4e_nvfp4_row_dot(u_row, u_scale, u_scale2, lut, base + m * pad, columns, lane);
    }
}

// And for the down pass's single matrix.
__device__ __forceinline__ void q4e_nvfp4_row_dot_group(const unsigned char* packed_row,
                                                        const unsigned char* scale_row,
                                                        float scale2, const float* lut,
                                                        const float2* lut_pair, const float* base,
                                                        int columns, int lane, int count, int pad,
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
                                       base + chunk * Q4E_SMEM_STRIDE, count, pad, out);
        }
        return;
    }
    for (int m = 0; m < count; ++m) {
        out[m] = q4e_nvfp4_row_dot(packed_row, scale_row, scale2, lut, base + m * pad, columns,
                                    lane);
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
    int count, int pad, float (*out)[Q4E_MEMBER_GROUP]) {
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
                                        lut_pair, base, columns, lane, count, pad, out[r]);
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
                                           base + chunk * Q4E_SMEM_STRIDE, count, pad, out[r]);
            }
        }
    }
}

// Softmax over router logits and top-k selection with renormalisation (blockIdx.x = batch token).
// Note: logits and probabilities may alias the same buffer.
extern "C" __global__ void q4e_router(const float* logits, float* probabilities, int* indices,
                                      int experts, int topk) {
    OPT_PDL_ENTRY();
    shared_moe_router_body<Q4E_ROUTER_SOFTMAX, false, false, true, Q4E_MAX_EXPERTS,
                           Q4E_MAX_TOPK, Q4E_BLOCK>(logits, nullptr, probabilities, indices,
                                                    probabilities, experts, topk, 1, 1.0f);
}


// Routed-experts pass one: simultaneous gate + up row dot products with conflict-free
// shared-memory activation staging (blockIdx.y = slot, blockIdx.z = batch token).
//
// One block per expert run, not per slot. Every slot of a run reads the same expert's weight
// rows, so the block that sorts first in the run recomputes the run's other tokens itself and the
// rest of the run's blocks exit before their staging loop. The expert's rows are then read from
// DRAM once per chunk and land in L2 for the run's later members instead of being fetched again.
//
// QUANT selects the stored expert format and nothing else: the NVFP4 release (packed E2M1 with an
// E4M3 scale a 16-column group and two per-tensor scales) and the FP8 release (E4M3 with one
// BF16 inverse scale a 128x128 tile). The run, the staging, the SwiGLU and the write-back are
// one body; only the row strides and the dot product differ.
#define Q4E_QUANT_NVFP4 0
#define Q4E_QUANT_FP8 1

// The bytes one FP8 tile scale takes: an E4M3 weight is a byte and its 128x128 tile scale is a
// BF16, while the NVFP4 grid's per-tile scale is a single byte. The expert table holds byte
// addresses for both, so the FP8 branch has to stride its scale rows by two.
#define Q4E_FP8_SCALE_BYTES 2

// One warp's partial dots of a single FP8 row against the staged rows of 'count' members. A lane
// owns sixteen consecutive values, so a 128-column scale tile covers whole lanes and every load
// is 16-byte aligned: no straddle and no masked tail, for either 2560 or 640.
__device__ __forceinline__ void q4e_fp8_chunk_group(const uint4 packed, float scale,
                                                     const float* base, int count, int pad,
                                                     float* out) {
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
            const float* x_chunk = base + m * pad;
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                dot += weight[i] * x_chunk[i];
            }
            out[m] += dot;
        }
    }
}

__device__ __forceinline__ void q4e_fp8_row_dot_group(const unsigned char* row,
                                                       const __nv_bfloat16* scale_row,
                                                       const float* base, int columns, int lane,
                                                       int count, int pad, float* out) {
#pragma unroll
    for (int m = 0; m < Q4E_MEMBER_GROUP; ++m) {
        out[m] = 0.0f;
    }
    const int chunks = columns >> 4;
    for (int chunk = lane; chunk < chunks; chunk += 32) {
        const int column = chunk * 16;
        const uint4 packed = q4e_ldg(reinterpret_cast<const uint4*>(row + column));
        const float scale = __bfloat162float(q4e_ldg(&scale_row[column >> 7]));
        q4e_fp8_chunk_group(packed, scale, base + q4e_smem_idx(column), count, pad, out);
    }
}

template <int QUANT>
__device__ __forceinline__ void q4e_expert_gate_up_body(
    const unsigned long long* __restrict__ table, const float* __restrict__ scales,
    const int* __restrict__ indices, const int* __restrict__ perm,
    const int* __restrict__ run_bounds, const float* __restrict__ x,
    float* __restrict__ activated, int slots, int hidden, int intermediate) {
    constexpr bool FP8 = QUANT == Q4E_QUANT_FP8;
    extern __shared__ __align__(16) float source[];
    __shared__ float lut[16];
    __shared__ float2 lut_pair[256];
    if (hidden <= 0 || (FP8 && (hidden & 15) != 0)) {
        return;
    }
    // The per-member stride: one padded float a column group, the conflict-free layout the
    // staging writes. The host sizes the dynamic reservation from the same expression.
    const int pad = ((hidden + 31) >> 5) * Q4E_SMEM_STRIDE;
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
    // NVFP4: 16-column scale groups and packed nibbles. FP8: one byte a weight and one BF16
    // scale row per 128 weight rows.
    int groups = FP8 ? (hidden >> 7) : ((hidden + 15) / 16);
    int packed = FP8 ? hidden : ((hidden + 1) / 2);
    const int valid = max(0, min(Q4E_ROWS_PER_WARP, intermediate - first_row));
    const unsigned char* grow[Q4E_ROWS_PER_WARP];
    const void* gsrow[Q4E_ROWS_PER_WARP];
    const unsigned char* urow[Q4E_ROWS_PER_WARP];
    const void* usrow[Q4E_ROWS_PER_WARP];
    float g2 = 1.0f;
    float u2 = 1.0f;
#pragma unroll
    for (int r = 0; r < Q4E_ROWS_PER_WARP; ++r) {
        grow[r] = nullptr;
        gsrow[r] = nullptr;
        urow[r] = nullptr;
        usrow[r] = nullptr;
    }
    if (expert >= 0) {
        const unsigned char* gw = (const unsigned char*)table[expert * 6 + 0];
        const unsigned char* gs = (const unsigned char*)table[expert * 6 + 1];
        const unsigned char* uw = (const unsigned char*)table[expert * 6 + 2];
        const unsigned char* us = (const unsigned char*)table[expert * 6 + 3];
#pragma unroll
        for (int r = 0; r < Q4E_ROWS_PER_WARP; ++r) {
            const unsigned long long row = (unsigned long long)(first_row + r);
            grow[r] = gw + row * packed;
            urow[r] = uw + row * packed;
            if constexpr (FP8) {
                // The table carries byte addresses: an FP8 tile scale is a BF16, so a scale row
                // advances two bytes an element where the NVFP4 grid's scales are one.
                gsrow[r] = gs + (row >> 7) * groups * Q4E_FP8_SCALE_BYTES;
                usrow[r] = us + (row >> 7) * groups * Q4E_FP8_SCALE_BYTES;
            } else {
                gsrow[r] = gs + row * groups;
                usrow[r] = us + row * groups;
            }
        }
        if constexpr (FP8) {
            g2 = 1.0f;
            u2 = 1.0f;
        } else {
            g2 = scales[expert * 3 + 0];
            u2 = scales[expert * 3 + 1];
        }
        // The block's panel of both matrices, fetched while the block stages its members.
        if (!FP8 && threadIdx.x == 0) {
            const unsigned long long panel = (unsigned long long)blockIdx.x * rows;
            const long long panel_rows =
                min((long long)rows, (long long)intermediate - (long long)panel);
            if (panel_rows > 0) {
                for (int matrix = 0; matrix < 4; ++matrix) {
                    const unsigned long long stride = (matrix & 1) ? groups : packed;
                    spark_prefetch_l2((const unsigned char*)table[expert * 6 + matrix] +
                                          panel * stride,
                                      panel_rows * stride);
                }
            }
        }
    }
    if constexpr (!FP8) {
        q4e_e2m1_table_fill(lut);
        q4e_e2m1_table_pair_fill(lut_pair);
    }
    for (int group = 0; group < members; group += Q4E_MEMBER_GROUP) {
        const int count = min(Q4E_MEMBER_GROUP, members - group);
        // The block order comes from the shared plan (kernels/moe_plan_sm121.cu): the run's
        // members are consecutive ranks, and each writes the result back to its own (token, slot).
        for (int m = 0; m < count; ++m) {
            const int total_slot = perm[begin + group + m];
            const int b = total_slot / slots;
            const float* x_b = x + (unsigned long long)b * hidden;
            float* staged = source + m * pad;
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
                if constexpr (FP8) {
                    q4e_fp8_row_dot_group(grow[r],
                                           reinterpret_cast<const __nv_bfloat16*>(gsrow[r]),
                                           source, hidden, lane, count, pad, g);
                    q4e_fp8_row_dot_group(urow[r],
                                           reinterpret_cast<const __nv_bfloat16*>(usrow[r]),
                                           source, hidden, lane, count, pad, u);
                } else {
                    q4e_nvfp4_row_dot_pair_group(
                        grow[r], (const unsigned char*)gsrow[r], g2, urow[r],
                        (const unsigned char*)usrow[r], u2, lut, lut_pair, source, hidden, lane,
                        count, pad, g, u);
                }
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

extern "C" __global__ __launch_bounds__(256, 4) void q4e_expert_gate_up(
    const unsigned long long* __restrict__ table, const float* __restrict__ scales,
    const int* __restrict__ indices, const int* __restrict__ perm,
    const int* __restrict__ run_bounds, const float* __restrict__ x,
    float* __restrict__ activated, int slots, int hidden, int intermediate) {
    OPT_PDL_ENTRY();
    q4e_expert_gate_up_body<Q4E_QUANT_NVFP4>(table, scales, indices, perm, run_bounds, x, activated,
                                             slots, hidden, intermediate);
}

extern "C" __global__ __launch_bounds__(256, 4) void q4e_expert_gate_up_fp8(
    const unsigned long long* __restrict__ table, const int* __restrict__ indices,
    const int* __restrict__ perm, const int* __restrict__ run_bounds,
    const float* __restrict__ x, float* __restrict__ activated, int slots, int hidden,
    int intermediate) {
    OPT_PDL_ENTRY();
    q4e_expert_gate_up_body<Q4E_QUANT_FP8>(table, nullptr, indices, perm, run_bounds, x, activated,
                                           slots, hidden, intermediate);
}

// Routed-experts pass two: down projection with conflict-free shared-memory staging
// (blockIdx.y = slot, blockIdx.z = batch token). One block per expert run, as in pass one.
template <int QUANT>
__device__ __forceinline__ void q4e_expert_down_body(
    const unsigned long long* __restrict__ table, const float* __restrict__ scales,
    const int* __restrict__ indices, const int* __restrict__ perm,
    const int* __restrict__ run_bounds, const float* __restrict__ weights,
    const float* __restrict__ activated, float* __restrict__ partials, int slots, int hidden,
    int intermediate, int weight_stride) {
    constexpr bool FP8 = QUANT == Q4E_QUANT_FP8;
    extern __shared__ __align__(16) float source[];
    __shared__ float lut[16];
    __shared__ float2 lut_pair[256];
    if (intermediate <= 0 || (FP8 && (intermediate & 15) != 0)) {
        return;
    }
    // As in pass one, from the intermediate width the down pass stages.
    const int pad = ((intermediate + 31) >> 5) * Q4E_SMEM_STRIDE;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int rows = (blockDim.x >> 5) * Q4E_DOWN_ROWS_PER_WARP;
    int first_row = blockIdx.x * rows + warp * Q4E_DOWN_ROWS_PER_WARP;
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
    int groups = FP8 ? (intermediate >> 7) : ((intermediate + 15) / 16);
    int packed = FP8 ? intermediate : ((intermediate + 1) / 2);
    const int valid = max(0, min(Q4E_DOWN_ROWS_PER_WARP, hidden - first_row));
    const unsigned char* drow[Q4E_DOWN_ROWS_PER_WARP];
    const void* dsrow[Q4E_DOWN_ROWS_PER_WARP];
    float d2 = 1.0f;
#pragma unroll
    for (int r = 0; r < Q4E_DOWN_ROWS_PER_WARP; ++r) {
        drow[r] = nullptr;
        dsrow[r] = nullptr;
    }
    if (expert >= 0) {
        const unsigned char* dw = (const unsigned char*)table[expert * 6 + 4];
        const unsigned char* ds = (const unsigned char*)table[expert * 6 + 5];
#pragma unroll
        for (int r = 0; r < Q4E_DOWN_ROWS_PER_WARP; ++r) {
            const unsigned long long row = (unsigned long long)(first_row + r);
            drow[r] = dw + row * packed;
            if constexpr (FP8) {
                // As in pass one: an FP8 tile scale is a BF16 addressed by byte.
                dsrow[r] = ds + (row >> 7) * groups * Q4E_FP8_SCALE_BYTES;
            } else {
                dsrow[r] = ds + row * groups;
            }
        }
        if constexpr (!FP8) {
            d2 = scales[expert * 3 + 2];
        }
    }
    if constexpr (!FP8) {
        q4e_e2m1_table_fill(lut);
        q4e_e2m1_table_pair_fill(lut_pair);
    }
    for (int group = 0; group < members; group += Q4E_MEMBER_GROUP) {
        const int count = min(Q4E_MEMBER_GROUP, members - group);
        for (int m = 0; m < count; ++m) {
            const int total_slot = perm[begin + group + m];
            const float* slot_activated =
                activated + (unsigned long long)total_slot * intermediate;
            float* staged = source + m * pad;
            for (int column = threadIdx.x; column < intermediate; column += blockDim.x) {
                staged[q4e_smem_idx(column)] = slot_activated[column];
            }
        }
        __syncthreads();
        float value[Q4E_DOWN_ROWS_PER_WARP][Q4E_MEMBER_GROUP];
        if (expert >= 0) {
            if constexpr (FP8) {
#pragma unroll
                for (int r = 0; r < Q4E_DOWN_ROWS_PER_WARP; ++r) {
                    if (r < valid) {
                        q4e_fp8_row_dot_group(
                            drow[r], reinterpret_cast<const __nv_bfloat16*>(dsrow[r]), source,
                            intermediate, lane, count, pad, value[r]);
                    }
                }
            } else {
                q4e_nvfp4_rows_dot_group<Q4E_DOWN_ROWS_PER_WARP>(
                    drow[0], (const unsigned char*)dsrow[0], d2, packed, groups, valid, lut,
                    lut_pair, source, intermediate, lane, count, pad, value);
            }
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

extern "C" __global__ __launch_bounds__(256, 4) void q4e_expert_down(
    const unsigned long long* __restrict__ table, const float* __restrict__ scales,
    const int* __restrict__ indices, const int* __restrict__ perm,
    const int* __restrict__ run_bounds, const float* __restrict__ weights,
    const float* __restrict__ activated, float* __restrict__ partials, int slots, int hidden,
    int intermediate, int weight_stride) {
    OPT_PDL_ENTRY();
    q4e_expert_down_body<Q4E_QUANT_NVFP4>(table, scales, indices, perm, run_bounds, weights,
                                          activated, partials, slots, hidden, intermediate,
                                          weight_stride);
}

extern "C" __global__ __launch_bounds__(256, 4) void q4e_expert_down_fp8(
    const unsigned long long* __restrict__ table, const int* __restrict__ indices,
    const int* __restrict__ perm, const int* __restrict__ run_bounds,
    const float* __restrict__ weights, const float* __restrict__ activated,
    float* __restrict__ partials, int slots, int hidden, int intermediate, int weight_stride) {
    OPT_PDL_ENTRY();
    q4e_expert_down_body<Q4E_QUANT_FP8>(table, nullptr, indices, perm, run_bounds, weights,
                                        activated, partials, slots, hidden, intermediate,
                                        weight_stride);
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

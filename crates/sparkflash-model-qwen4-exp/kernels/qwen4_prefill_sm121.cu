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

// Qwen3.8-Flash-Next (Qwen4-Exp) tensor-core prefill kernels, TP=1.
//
// A prefill chunk is wide enough that the token kernels in qwen4_exp_sm121.cu - CUDA-core f32
// dot products, one block per expert run - leave the tensor cores idle: at a 128-token chunk the
// routed experts alone were two thirds of the chunk. This image holds the chunk-wide forms:
//
//   - q4p_dense_mma: a BF16 projection over the whole chunk, the workspace's shared
//     opt_mma_bf16_body with the f32 activations split into a bf16 head and remainder.
//   - q4p_moe_plan / q4p_moe_gate_up / q4p_moe_down: the routed experts as grouped W4A16 MMA.
//     The plan sorts the chunk's (token, slot) pairs by expert into items of up to 32 rows; each
//     CTA multiplies one item by one panel of an expert's NVFP4 weights, decoding E2M1 x E4M3 to
//     bf16 in registers (the product is exact in bf16) and applying weight_scale_2 in f32. The
//     activations - the expert input and the SwiGLU rows between the passes - are a bf16 head and
//     remainder, two MMAs a fragment, as the dense projections.
//   - q4p_gdn_conv / q4p_gdn_conv_state: the causal depthwise conv, parallel over the chunk's
//     tokens instead of stepping through them.
//   - q4p_gdn_prep / q4p_gdn_scan / q4p_gdn_gate: the Gated DeltaNet recurrence with the state in
//     registers, so a token step is two short FMA chains and two shuffle reductions instead of
//     three block-wide reductions.
//   - q4p_kda_prep / q4p_kda_scan: the same recurrence with Kimi Delta Attention's decay, one
//     factor a key channel (GLM-5.3-Flash), from the token cubin's q4e_kda_decay.
//
// The token kernels remain the decode path and the f32 reference these are tested against.
// The MMA forms carry each activation as a bf16 head and remainder (16 significant bits) and
// accumulate in f32, so a chunk through this image matches the reference to a tolerance, not
// bit for bit; a decode step never runs here.
//
// Attribution:
//   - Model architecture (Qwen4ExpForConditionalGeneration: Gated DeltaNet linear attention,
//     512-expert top-10 MoE) originates from Qwen/Qwen3.8-Flash-Next by the Qwen team at Alibaba
//     Cloud and the vLLM reference implementation (https://github.com/vllm-project/vllm,
//     Apache-2.0). The Gated DeltaNet recurrence is that of Yang, Kautz and Hatamizadeh, "Gated
//     Delta Networks: Improving Mamba2 with Delta Rule" (ICLR 2025), computed here as a
//     sequential scan in the same order as this crate's host oracle (src/forward.rs).
//   - Kimi Delta Attention's per-channel decay (q4p_kda_*) is the method of Kimi Team et al.,
//     "Kimi Linear: An Expressive, Efficient Attention Architecture" (arXiv:2510.26692), as Z.ai's
//     GLM-5.3-Flash release (MIT License) uses it; it is checked against the shared host delta
//     rule in crates/sparkflash-weights/src/linear_attention.rs.
//   - NVFP4 block-scaled weight layout (packed E2M1 nibbles, per-16-column FP8 E4M3 block
//     scales, FP32 weight_scale_2) originates from nvidia/Qwen3.8-Flash-Next-NVFP4 (NVIDIA
//     TensorRT ModelOpt 0.46).
//   - SM120/SM121 (DGX Spark GB10) mma.sync / cp.async tiling patterns draw on prior art from
//     b12x (https://github.com/local-inference-lab/b12x, flashinfer/experimental/b12x,
//     Apache-2.0; Luke Alonso @lukealonso, @original-el8, and contributors) and NVIDIA Flare
//     (https://github.com/NVIDIA/flare-local-inference, Apache-2.0), through the shared
//     kernels/opt_mma_*_sm121.cuh headers. See the root NOTICE file for full provenance details.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// opt_mma_bf16_body, ob_mma and the cp.async helpers, shared with the DeepSeek prefill.
#include "opt_mma_bf16_sm121.cuh"
// The E4M3 / E2M1 decoders (kernels/quant_decode_sm121.cuh).
#include "quant_decode_sm121.cuh"

__device__ __forceinline__ float q4p_silu(float value) {
    return value / (1.0f + __expf(-value));
}

// ---------------------------------------------------------------------------------------------
// Dense projections.
// ---------------------------------------------------------------------------------------------

// out[t][c] = sum_i a[t][i] * w[c][i] for a row-major BF16 [cols][inner] matrix and f32
// activations. The grid is (cols / 128, tokens / 128, segments) rounded up, 256 threads; `cols` and
// `inner / segments` must be multiples of 32 and both operands 16-byte aligned (the host checks).
//
// A narrow matrix is one 128-column tile wide - the router's 512 rows are four - so a 512-token
// chunk is sixteen CTAs on forty-eight SMs. `segments` > 1 splits the inner dimension over
// blockIdx.z instead: slice s sums its inner / segments columns into out[s][t][c], and
// q4e_gemm_reduce adds the slices in order, as it does for the token GEMM's split.
extern "C" __global__ __launch_bounds__(256) void q4p_dense_mma(
    const __nv_bfloat16* __restrict__ w, const float* __restrict__ a, float* __restrict__ out,
    unsigned int cols, unsigned int inner, unsigned int tokens, unsigned int segments) {
    const unsigned int slice = inner / segments;
    opt_mma_bf16_body<OB_A_F32, OB_W_ROWK, true>(
        a, reinterpret_cast<const uint8_t*>(w), nullptr, out, cols, slice, tokens, inner, cols,
        slice, (unsigned long long)slice * 2ull, 0, (unsigned long long)tokens * cols, inner >> 5);
}

// The token GEMM's two epilogues, applied in place after the MMA has written the raw dot
// products: 1 = silu(v / scale), 2 = sigmoid(v / scale) scaled to two.
extern "C" __global__ void q4p_epilogue(float* __restrict__ y, unsigned long long count,
                                        int epilogue, float scale) {
    for (unsigned long long index = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
         index < count; index += (unsigned long long)gridDim.x * blockDim.x) {
        const float value = y[index];
        y[index] = epilogue == 1 ? q4p_silu(value / scale) : 2.0f / (1.0f + __expf(-value / scale));
    }
}

// The bf16 head of a pair of f32 values and what is left of them, both round to nearest even.
__device__ __forceinline__ void q4p_split2(float a, float b, __nv_bfloat162& head,
                                           __nv_bfloat162& rest) {
    head = __floats2bfloat162_rn(a, b);
    rest = __floats2bfloat162_rn(a - __low2float(head), b - __high2float(head));
}

// f32 -> two bf16 planes: y[i] the head of x[i], y[count + i] the remainder. The routed experts'
// activation operand.
extern "C" __global__ void q4p_split_bf16(const float* __restrict__ x,
                                          __nv_bfloat16* __restrict__ y,
                                          unsigned long long count) {
    const unsigned long long pairs = count >> 1;
    for (unsigned long long index = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
         index < pairs; index += (unsigned long long)gridDim.x * blockDim.x) {
        const float2 v = reinterpret_cast<const float2*>(x)[index];
        __nv_bfloat162 head;
        __nv_bfloat162 rest;
        q4p_split2(v.x, v.y, head, rest);
        reinterpret_cast<__nv_bfloat162*>(y)[index] = head;
        reinterpret_cast<__nv_bfloat162*>(y + count)[index] = rest;
    }
    if ((count & 1ull) != 0ull && blockIdx.x == 0 && threadIdx.x == 0) {
        const float v = x[count - 1];
        const __nv_bfloat16 head = __float2bfloat16_rn(v);
        y[count - 1] = head;
        y[2 * count - 1] = __float2bfloat16_rn(v - __bfloat162float(head));
    }
}

// ---------------------------------------------------------------------------------------------
// Routed experts.
// ---------------------------------------------------------------------------------------------

// Routed (token, slot) rows per item: two m16 tiles. At a 512-token chunk an expert averages ten
// rows, so most experts are one item and their weights are read once per chunk.
#define Q4P_MOE_ROWS 32
#define Q4P_MAX_EXPERTS 512
// K per pipeline stage: four 16-column E4M3 scale groups.
#define Q4P_BK 64
// Weight rows (output columns) per CTA: 64 gate + 64 up, or 128 down.
#define Q4P_BN 128
#define Q4P_STAGES 3

// The counting-sort plan, one block of 1024 threads per layer, as the DeepSeek prefill's
// opt_mma_moe_plan: `sorted` lists the flat slot indices (token * topk + slot) grouped by expert,
// and `items` the (expert, first sorted index, rows) runs of at most Q4P_MOE_ROWS. The order of
// rows inside an expert is whatever the atomics produce; no row's result depends on which other
// rows share its item, so it does not reach the output.
//
// A slot routed to no expert (an index outside [0, experts)) has its partial row zeroed here, so
// the combine still sums ten rows per token.
extern "C" __global__ __launch_bounds__(1024) void q4p_moe_plan(
    const int* __restrict__ indices, int total, int experts, int* __restrict__ sorted,
    int4* __restrict__ items, int* __restrict__ item_count, float* __restrict__ partials,
    int hidden) {
    __shared__ int s_count[Q4P_MAX_EXPERTS];
    __shared__ int s_start[Q4P_MAX_EXPERTS];
    __shared__ int s_item[Q4P_MAX_EXPERTS];
    const int tid = threadIdx.x;
    if (experts <= 0 || experts > Q4P_MAX_EXPERTS) {
        if (tid == 0) {
            *item_count = 0;
        }
        return;
    }
    for (int e = tid; e < experts; e += blockDim.x) {
        s_count[e] = 0;
    }
    __syncthreads();
    for (int i = tid; i < total; i += blockDim.x) {
        const int e = indices[i];
        if (e >= 0 && e < experts) {
            atomicAdd(&s_count[e], 1);
        }
    }
    __syncthreads();
    // One warp scans both the row counts and the item counts: lane l owns a contiguous range.
    if (tid < 32) {
        const int per = (experts + 31) / 32;
        const int lo = min(experts, tid * per);
        const int hi = min(experts, lo + per);
        int rows = 0;
        int runs = 0;
        for (int e = lo; e < hi; ++e) {
            rows += s_count[e];
            runs += (s_count[e] + Q4P_MOE_ROWS - 1) / Q4P_MOE_ROWS;
        }
        int rows_scan = rows;
        int runs_scan = runs;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int r = __shfl_up_sync(0xffffffffu, rows_scan, offset);
            const int n = __shfl_up_sync(0xffffffffu, runs_scan, offset);
            if (tid >= offset) {
                rows_scan += r;
                runs_scan += n;
            }
        }
        int row_at = rows_scan - rows;
        int item_at = runs_scan - runs;
        for (int e = lo; e < hi; ++e) {
            s_start[e] = row_at;
            s_item[e] = item_at;
            row_at += s_count[e];
            item_at += (s_count[e] + Q4P_MOE_ROWS - 1) / Q4P_MOE_ROWS;
        }
        if (tid == 31) {
            *item_count = runs_scan;
        }
    }
    __syncthreads();
    for (int e = tid; e < experts; e += blockDim.x) {
        const int n = s_count[e];
        int at = s_item[e];
        for (int first = 0; first < n; first += Q4P_MOE_ROWS, ++at) {
            items[at] = make_int4(e, s_start[e] + first, min(Q4P_MOE_ROWS, n - first), 0);
        }
        // The count becomes the expert's scatter cursor; this thread is its only reader.
        s_count[e] = s_start[e];
    }
    __syncthreads();
    for (int i = tid; i < total; i += blockDim.x) {
        const int e = indices[i];
        if (e >= 0 && e < experts) {
            sorted[atomicAdd(&s_count[e], 1)] = i;
        } else if (partials != nullptr) {
            float* row = partials + (unsigned long long)i * hidden;
            for (int c = 0; c < hidden; ++c) {
                row[c] = 0.0f;
            }
        }
    }
}

// One pipeline stage of an expert CTA.
//
// a: the item's 32 activation rows x 64 bf16, head plane then remainder plane, each 128-byte row
//    as eight 16-byte chunks with the chunk index XOR 4 on odd rows, so the eight rows a
//    quarter-warp's LDS.128 touches land on distinct banks.
// b: 128 weight rows x 32 packed bytes (64 E2M1 codes), the two 16-byte halves swapped on rows
//    with bit 2 set, so a warp's 32 word loads (8 rows x 4 words) are conflict-free.
// s: the four E4M3 block scales of each weight row for this stage, one word a row.
struct Q4pStage {
    __nv_bfloat16 a[2][Q4P_MOE_ROWS * Q4P_BK];
    uint32_t b[Q4P_BN * 8];
    uint32_t s[Q4P_BN];
};

__device__ __forceinline__ void q4p_cp4(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src));
}

// bf16x2 of one packed byte's two E2M1 codes: low nibble (the even column) in the low half.
__device__ __forceinline__ uint32_t q4p_e2m1x2_bf16(unsigned int code) {
    const __nv_bfloat16 lo = __float2bfloat16_rn(spark_fp4_e2m1_decode((unsigned char)(code & 15u)));
    const __nv_bfloat16 hi = __float2bfloat16_rn(spark_fp4_e2m1_decode((unsigned char)(code >> 4)));
    return ob_pack(lo, hi);
}

__device__ __forceinline__ uint32_t q4p_bf16x2_mul(uint32_t x, uint32_t y) {
    const __nv_bfloat162 r = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&x),
                                     *reinterpret_cast<const __nv_bfloat162*>(&y));
    return *reinterpret_cast<const uint32_t*>(&r);
}

// The E4M3 block scale in byte `which` of `word`, as a bf16 pair (exact: three mantissa bits).
__device__ __forceinline__ uint32_t q4p_scale_pair(uint32_t word, unsigned int which) {
    const __nv_bfloat16 s =
        __float2bfloat16_rn(spark_fp8_e4m3_decode((unsigned char)((word >> (8u * which)) & 0xffu)));
    return ob_pack(s, s);
}

// One CTA: an item's rows (up to 32) times one 128-row panel of NVFP4 weights, K in stages of 64.
//
// Warp w owns panel rows 8w..8w+7 and 64+8w..64+8w+7 - for gate_up the same 8 intermediate
// columns of the gate and the up matrix, so SwiGLU pairs them in registers; for down 16 output
// columns. Each lane decodes exactly its own m16n8k16 B fragments from the packed stage: lane
// (g, t) takes words t and t+4 of row g, i.e. packed columns 8t..8t+7 and 32+8t..32+8t+7, and the
// MMA's k index is permuted to match (block kb, k 2t+e <-> column 8t+2kb+e, k 2t+8+e <-> column
// 32+8t+2kb+e). The A fragments use the same permutation, read straight from the staged rows as
// 16-byte chunks, so the dot product is over the same pairs and no shuffle of either operand is
// needed. The head and remainder planes of A run through the same B fragments into one
// accumulator.
//
// `a_plane` is the element offset of a_src's remainder plane; `out_plane` that of the SwiGLU
// rows' remainder plane gate_up writes.
template <bool GATE_UP>
__device__ __forceinline__ void q4p_moe_body(
    const unsigned long long* __restrict__ table, const float* __restrict__ scales,
    const int* __restrict__ sorted, const int4* __restrict__ items,
    const int* __restrict__ item_count, const __nv_bfloat16* __restrict__ a_src,
    unsigned long long a_plane, const float* __restrict__ route_weights, void* __restrict__ out,
    unsigned long long out_plane, int topk, int hidden, int intermediate, int weight_stride) {
    __shared__ __align__(16) Q4pStage stages[Q4P_STAGES];
    __shared__ uint32_t lut[256];
    __shared__ int s_slot[Q4P_MOE_ROWS];
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
    // K and the output width of this pass.
    const int inner = GATE_UP ? hidden : intermediate;
    const int n0 = blockIdx.x * (GATE_UP ? Q4P_BN / 2 : Q4P_BN);
    const int packed_row = inner >> 1;
    const int scale_row = inner >> 4;
    for (int code = tid; code < 256; code += blockDim.x) {
        lut[code] = q4p_e2m1x2_bf16((unsigned int)code);
    }
    if (tid < Q4P_MOE_ROWS) {
        s_slot[tid] = tid < count ? sorted[first + tid] : -1;
    }
    __syncthreads();

    // The rows this thread copies every stage.
    const int a_row = tid >> 3;
    const int a_chunk = tid & 7;
    const int a_slot = s_slot[a_row];
    const __nv_bfloat16* a_base = a_src;
    if (a_slot >= 0) {
        a_base = GATE_UP ? a_src + (unsigned long long)(a_slot / topk) * hidden
                         : a_src + (unsigned long long)a_slot * intermediate;
    }
    const uint32_t a_dst = (uint32_t)(a_row * Q4P_BK + ((a_chunk ^ ((a_row & 1) << 2)) * 8));
    const int b_row = tid >> 1;
    const int b_half = tid & 1;
    const unsigned char* b_base;
    if (GATE_UP) {
        const int which = b_row < Q4P_BN / 2 ? 0 : 2;
        const unsigned long long row = (unsigned long long)(n0 + (b_row & (Q4P_BN / 2 - 1)));
        b_base = (const unsigned char*)table[expert * 6 + which] + row * packed_row;
    } else {
        b_base = (const unsigned char*)table[expert * 6 + 4] +
                 (unsigned long long)(n0 + b_row) * packed_row;
    }
    const uint32_t b_dst = (uint32_t)(b_row * 8 + ((b_half ^ ((b_row >> 2) & 1)) * 4));
    // Scale rows: threads 0..127 each own one weight row's word.
    const unsigned char* sc_base = nullptr;
    if (tid < Q4P_BN) {
        if (GATE_UP) {
            const int which = tid < Q4P_BN / 2 ? 1 : 3;
            const unsigned long long row = (unsigned long long)(n0 + (tid & (Q4P_BN / 2 - 1)));
            sc_base = (const unsigned char*)table[expert * 6 + which] + row * scale_row;
        } else {
            sc_base = (const unsigned char*)table[expert * 6 + 5] +
                      (unsigned long long)(n0 + tid) * scale_row;
        }
    }
    auto load = [&](int buffer, int ks) {
        Q4pStage& st = stages[buffer];
        om_cp16(om_smem(&st.a[0][a_dst]), a_base + ks * Q4P_BK + a_chunk * 8, a_slot >= 0);
        om_cp16(om_smem(&st.a[1][a_dst]), a_base + a_plane + ks * Q4P_BK + a_chunk * 8,
                a_slot >= 0);
        om_cp16(om_smem(&st.b[b_dst]), b_base + ks * (Q4P_BK / 2) + b_half * 16, true);
        if (sc_base != nullptr) {
            q4p_cp4(om_smem(&st.s[tid]), sc_base + ks * (Q4P_BK / 16));
        }
    };

    const int g = lane >> 2;
    const int t = lane & 3;
    float acc[2][2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    const int ksteps = inner / Q4P_BK;
#pragma unroll
    for (int s = 0; s < Q4P_STAGES - 1; ++s) {
        if (s < ksteps) {
            load(s, s);
        }
        om_commit();
    }
    // Both m16 tiles' rows have the parity of g, so one swizzle serves all four row reads.
    const int a_sw = (g & 1) << 2;
    const int b_sw = ((g >> 2) & 1) << 2;
    for (int ks = 0; ks < ksteps; ++ks) {
        om_wait<Q4P_STAGES - 2>();
        __syncthreads();
        const int next = ks + Q4P_STAGES - 1;
        if (next < ksteps) {
            load(next % Q4P_STAGES, next);
        }
        om_commit();
        const Q4pStage& st = stages[ks % Q4P_STAGES];
        uint32_t bf[2][4][2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = 64 * j + 8 * warp + g;
            const uint32_t w0 = st.b[row * 8 + (t ^ b_sw)];
            const uint32_t w1 = st.b[row * 8 + ((t + 4) ^ b_sw)];
            const uint32_t sw = st.s[row];
            const uint32_t s_lo = q4p_scale_pair(sw, (unsigned int)(t >> 1));
            const uint32_t s_hi = q4p_scale_pair(sw, 2u + (unsigned int)(t >> 1));
#pragma unroll
            for (int kb = 0; kb < 4; ++kb) {
                bf[j][kb][0] = q4p_bf16x2_mul(lut[(w0 >> (8 * kb)) & 0xffu], s_lo);
                bf[j][kb][1] = q4p_bf16x2_mul(lut[(w1 >> (8 * kb)) & 0xffu], s_hi);
            }
        }
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int r0 = mt * 16 + g;
#pragma unroll
            for (int plane = 0; plane < 2; ++plane) {
                const __nv_bfloat16* a = st.a[plane];
                const uint4 lo0 = *reinterpret_cast<const uint4*>(&a[r0 * Q4P_BK + ((t ^ a_sw) * 8)]);
                const uint4 hi0 =
                    *reinterpret_cast<const uint4*>(&a[r0 * Q4P_BK + (((t + 4) ^ a_sw) * 8)]);
                const uint4 lo8 =
                    *reinterpret_cast<const uint4*>(&a[(r0 + 8) * Q4P_BK + ((t ^ a_sw) * 8)]);
                const uint4 hi8 = *reinterpret_cast<const uint4*>(
                    &a[(r0 + 8) * Q4P_BK + (((t + 4) ^ a_sw) * 8)]);
                const uint32_t l0[4] = {lo0.x, lo0.y, lo0.z, lo0.w};
                const uint32_t h0[4] = {hi0.x, hi0.y, hi0.z, hi0.w};
                const uint32_t l8[4] = {lo8.x, lo8.y, lo8.z, lo8.w};
                const uint32_t h8[4] = {hi8.x, hi8.y, hi8.z, hi8.w};
#pragma unroll
                for (int kb = 0; kb < 4; ++kb) {
                    const uint32_t af[4] = {l0[kb], l8[kb], h0[kb], h8[kb]};
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        ob_mma(acc[mt][j], af, bf[j][kb][0], bf[j][kb][1]);
                    }
                }
            }
        }
    }
    om_wait<0>();

    // Epilogue. Accumulator (mt, j) element 2h + e is item row mt*16 + g + 8h, panel row
    // 64j + 8w + 2t + e.
    if (GATE_UP) {
        const float g2 = scales[expert * 3 + 0];
        const float u2 = scales[expert * 3 + 1];
        __nv_bfloat16* activated = reinterpret_cast<__nv_bfloat16*>(out);
        const int column = n0 + 8 * warp + 2 * t;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mt * 16 + g + 8 * h;
                if (r >= count) {
                    continue;
                }
                const int slot = s_slot[r];
                const float a0 = q4p_silu(acc[mt][0][2 * h] * g2) * (acc[mt][1][2 * h] * u2);
                const float a1 =
                    q4p_silu(acc[mt][0][2 * h + 1] * g2) * (acc[mt][1][2 * h + 1] * u2);
                __nv_bfloat162 head;
                __nv_bfloat162 rest;
                q4p_split2(a0, a1, head, rest);
                const unsigned long long at = (unsigned long long)slot * intermediate + column;
                *reinterpret_cast<__nv_bfloat162*>(&activated[at]) = head;
                *reinterpret_cast<__nv_bfloat162*>(&activated[out_plane + at]) = rest;
            }
        }
    } else {
        const float d2 = scales[expert * 3 + 2];
        float* partials = reinterpret_cast<float*>(out);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mt * 16 + g + 8 * h;
                if (r >= count) {
                    continue;
                }
                const int slot = s_slot[r];
                const float route =
                    route_weights[(unsigned long long)(slot / topk) * weight_stride + slot % topk];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int column = n0 + 64 * j + 8 * warp + 2 * t;
                    *reinterpret_cast<float2*>(&partials[(unsigned long long)slot * hidden + column]) =
                        make_float2(route * (acc[mt][j][2 * h] * d2),
                                    route * (acc[mt][j][2 * h + 1] * d2));
                }
            }
        }
    }
}

// Pass one: gate and up for 64 intermediate columns of one item, SwiGLU'd into the two bf16
// planes of activated[slot][intermediate], the remainder `activated_plane` elements on. `x` is
// q4p_split_bf16's pair of [token][hidden] planes, `x_plane` apart. Grid (intermediate / 64, max
// items), 256 threads.
extern "C" __global__ __launch_bounds__(256, 2) void q4p_moe_gate_up(
    const unsigned long long* __restrict__ table, const float* __restrict__ scales,
    const int* __restrict__ sorted, const int4* __restrict__ items,
    const int* __restrict__ item_count, const __nv_bfloat16* __restrict__ x,
    unsigned long long x_plane, __nv_bfloat16* __restrict__ activated,
    unsigned long long activated_plane, int topk, int hidden, int intermediate) {
    q4p_moe_body<true>(table, scales, sorted, items, item_count, x, x_plane, nullptr, activated,
                       activated_plane, topk, hidden, intermediate, 0);
}

// Pass two: down for 128 output columns of one item, scaled by the routing weight into f32
// partials[slot][hidden] for q4e_moe_combine. Grid (hidden / 128, max items), 256 threads.
extern "C" __global__ __launch_bounds__(256, 2) void q4p_moe_down(
    const unsigned long long* __restrict__ table, const float* __restrict__ scales,
    const int* __restrict__ sorted, const int4* __restrict__ items,
    const int* __restrict__ item_count, const __nv_bfloat16* __restrict__ activated,
    unsigned long long activated_plane, const float* __restrict__ route_weights,
    float* __restrict__ partials, int topk, int hidden, int intermediate, int weight_stride) {
    q4p_moe_body<false>(table, scales, sorted, items, item_count, activated, activated_plane,
                        route_weights, partials, 0, topk, hidden, intermediate, weight_stride);
}

// ---------------------------------------------------------------------------------------------
// Gated DeltaNet.
// ---------------------------------------------------------------------------------------------

#define Q4P_GDN_KEY 128
// Tokens staged per pipeline step. Eight keeps a head's block (both buffers) at 26 KiB, so all
// forty-eight heads are resident at once - the scan is latency-bound, and a second wave would
// double it.
#define Q4P_GDN_TILE 8
// Floats per staged token: q then k, each 128 keys with four floats of padding after every 32,
// so the four key quarters a warp reads at one offset sit on different banks.
#define Q4P_GDN_HALF 144
#define Q4P_GDN_ROW (2 * Q4P_GDN_HALF)
// The value head width the scan is compiled for (the released configuration's): a compile-time
// stride keeps the 32 state cells a lane owns at immediate offsets from one base pointer, where a
// runtime stride had ptxas hold 32 hoisted 64-bit addresses and spill.
#define Q4P_GDN_VALUE 128

__device__ __forceinline__ int q4p_gdn_pad(int key) {
    return key + (key >> 5) * 4;
}

// The causal depthwise conv history a chunk carries (taps - 1 rows); the released configuration
// has four taps.
#define Q4P_CONV_HISTORY 8

// The conv's raw input row j of the chunk, or for j < 0 the carried history row history + j.
__device__ __forceinline__ float q4p_conv_input(const float* __restrict__ raw,
                                                const float* __restrict__ state, int j,
                                                int history, int width, int column) {
    return j >= 0 ? raw[(unsigned long long)j * width + column]
                  : state[(unsigned long long)(history + j) * width + column];
}

// The causal depthwise conv1d with SiLU over a whole chunk, one thread per (column, token): row
// t reads raw rows t - history .. t, and rows before the chunk from the carried history. The
// token kernel q4e_gdn_conv walks the chunk one token at a time per column and shifts the history
// through global memory at every step; this one has no step, and computes each output with the
// same products in the same order. Grid (width / 256, tokens), 256 threads; q4p_gdn_conv_state
// advances the history afterwards.
extern "C" __global__ void q4p_gdn_conv(const float* __restrict__ raw,
                                        const float* __restrict__ state,
                                        const __nv_bfloat16* __restrict__ weight,
                                        float* __restrict__ out, int width, int taps) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (column >= width) {
        return;
    }
    const int history = taps - 1;
    float accumulator = 0.0f;
    for (int tap = 0; tap < taps; ++tap) {
        const float source = q4p_conv_input(raw, state, t - (history - tap), history, width, column);
        accumulator += __bfloat162float(weight[column * taps + tap]) * source;
    }
    out[(unsigned long long)t * width + column] = q4p_silu(accumulator);
}

// Advance the conv history past a chunk of `batch` rows: row r becomes input row
// batch - history + r. When snapshot is non-null, region t of it (region_stride floats apart)
// receives the history as it stood after row t, for t < snapshot_rows, as q4e_gdn_conv writes it. Grid
// (width / 256), 256 threads; taps - 1 is at most Q4P_CONV_HISTORY (the host checks).
extern "C" __global__ void q4p_gdn_conv_state(const float* __restrict__ raw,
                                              float* __restrict__ state, int width, int taps,
                                              int batch, float* __restrict__ snapshot,
                                              int snapshot_rows, long long region_stride) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int history = taps - 1;
    if (column >= width || history <= 0 || history > Q4P_CONV_HISTORY) {
        return;
    }
    float carried[Q4P_CONV_HISTORY];
#pragma unroll
    for (int r = 0; r < Q4P_CONV_HISTORY; ++r) {
        if (r < history) {
            carried[r] = state[(unsigned long long)r * width + column];
        }
    }
    // Input row j, where the carried rows are j = -history .. -1.
    auto input = [&](int j) {
        float value = 0.0f;
#pragma unroll
        for (int r = 0; r < Q4P_CONV_HISTORY; ++r) {
            if (r < history && j == r - history) {
                value = carried[r];
            }
        }
        return j >= 0 ? raw[(unsigned long long)j * width + column] : value;
    };
    if (snapshot != nullptr) {
        for (int at = 0; at < snapshot_rows && at < batch; ++at) {
            float* into = snapshot + (unsigned long long)at * region_stride;
            for (int r = 0; r < history; ++r) {
                into[(unsigned long long)r * width + column] = input(at - history + 1 + r);
            }
        }
    }
    for (int r = 0; r < history; ++r) {
        state[(unsigned long long)r * width + column] = input(batch - history + r);
    }
}

// Per (token, key head): the L2-normalised query (also scaled by 1/sqrt(key_dim)) and key, laid
// out as the scan stages them; per (token, value head of the group): the decay factor and beta.
// With kChannel (Kimi Delta Attention) the decay is one factor a key channel, which the scan
// reads from q4e_kda_decay's buffer, so the head's factor is written as 1 and a, a_log and
// dt_bias are not read.
template <bool kChannel>
__device__ __forceinline__ void q4p_gdn_prep_body(
    const float* __restrict__ mixed, const float* __restrict__ a, const float* __restrict__ b,
    const __nv_bfloat16* __restrict__ a_log, const __nv_bfloat16* __restrict__ dt_bias,
    float* __restrict__ qk, float2* __restrict__ decay_beta, int key_width, int key_heads,
    int heads, int value_dim, int groups) {
    __shared__ float reduce[2][4];
    const int kh = blockIdx.x;
    const int token = blockIdx.y;
    const int key = threadIdx.x;
    const int conv_width = key_width * 2 + heads * value_dim;
    const float* row = mixed + (unsigned long long)token * conv_width;
    const float q = row[kh * Q4P_GDN_KEY + key];
    const float k = row[key_width + kh * Q4P_GDN_KEY + key];
    float q_sum = q * q;
    float k_sum = k * k;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        q_sum += __shfl_xor_sync(0xffffffffu, q_sum, offset);
        k_sum += __shfl_xor_sync(0xffffffffu, k_sum, offset);
    }
    if ((key & 31) == 0) {
        reduce[0][key >> 5] = q_sum;
        reduce[1][key >> 5] = k_sum;
    }
    __syncthreads();
    q_sum = (reduce[0][0] + reduce[0][1]) + (reduce[0][2] + reduce[0][3]);
    k_sum = (reduce[1][0] + reduce[1][1]) + (reduce[1][2] + reduce[1][3]);
    const float q_norm = rsqrtf(q_sum + 1e-6f);
    const float k_norm = rsqrtf(k_sum + 1e-6f);
    float* out = qk + ((unsigned long long)token * key_heads + kh) * Q4P_GDN_ROW;
    out[q4p_gdn_pad(key)] = q * q_norm * rsqrtf((float)Q4P_GDN_KEY);
    out[Q4P_GDN_HALF + q4p_gdn_pad(key)] = k * k_norm;
    if (key < 16) {
        // The padding is never read; zero it so the staged rows are fully defined.
        out[(key >> 2) * 36 + 32 + (key & 3)] = 0.0f;
        out[Q4P_GDN_HALF + (key >> 2) * 36 + 32 + (key & 3)] = 0.0f;
    }
    if (key < groups) {
        const int head = kh * groups + key;
        float factor = 1.0f;
        if constexpr (!kChannel) {
            const float decay =
                -__expf(__bfloat162float(a_log[head])) *
                logf(1.0f + __expf(a[(unsigned long long)token * heads + head] +
                                   __bfloat162float(dt_bias[head])));
            factor = __expf(decay);
        }
        const float beta = 1.0f / (1.0f + __expf(-b[(unsigned long long)token * heads + head]));
        decay_beta[(unsigned long long)token * heads + head] = make_float2(factor, beta);
    }
}

// Gated DeltaNet's prep: grid (key_heads, tokens), 128 threads (one per key).
extern "C" __global__ __launch_bounds__(128) void q4p_gdn_prep(
    const float* __restrict__ mixed, const float* __restrict__ a, const float* __restrict__ b,
    const __nv_bfloat16* __restrict__ a_log, const __nv_bfloat16* __restrict__ dt_bias,
    float* __restrict__ qk, float2* __restrict__ decay_beta, int key_width, int key_heads,
    int heads, int value_dim, int groups) {
    q4p_gdn_prep_body<false>(mixed, a, b, a_log, dt_bias, qk, decay_beta, key_width, key_heads,
                             heads, value_dim, groups);
}

// Kimi Delta Attention's prep, for q4p_kda_scan: the same grid, beta only beside the q and k.
extern "C" __global__ __launch_bounds__(128) void q4p_kda_prep(
    const float* __restrict__ mixed, const float* __restrict__ b, float* __restrict__ qk,
    float2* __restrict__ decay_beta, int key_width, int key_heads, int heads, int value_dim,
    int groups) {
    q4p_gdn_prep_body<true>(mixed, nullptr, b, nullptr, nullptr, qk, decay_beta, key_width,
                            key_heads, heads, value_dim, groups);
}

// The recurrence, state in registers. Grid (heads), 4 * Q4P_GDN_VALUE threads: warp w owns value
// columns 8w..8w+7, and the four lanes of a column split the 128 keys into quarters of 32, so a
// column's dot products are 32-term chains plus two shuffles. The state is read once and written
// once per chunk; `out` gets the raw (pre-norm) head output [token][heads * value_dim].
//
// The arithmetic is the host oracle's per element: S *= decay; memory = S . k; delta =
// (v - memory) * beta; S += k * delta; out = S . q. When snapshot is non-null, region t of it
// (region_stride floats apart) receives the state after row t, for t < snapshot_rows, as
// q4e_gdn_recur writes it.
//
// With kChannel (Kimi Delta Attention) S decays by one factor a key channel instead: `channel`
// is [batch][heads][Q4P_GDN_KEY], q4e_kda_decay's factors, staged with the keys' padding so a
// lane's quarter reads them as it reads k.
template <bool kChannel>
__device__ __forceinline__ void q4p_gdn_scan_body(
    const float* __restrict__ qk, const float2* __restrict__ decay_beta,
    const float* __restrict__ channel, const float* __restrict__ mixed,
    float* __restrict__ state, float* __restrict__ out, int key_width, int key_heads, int heads,
    int value_dim, int groups, int batch, float* __restrict__ snapshot, int snapshot_rows,
    long long region_stride) {
    __shared__ __align__(16) float s_qk[2][Q4P_GDN_TILE * Q4P_GDN_ROW];
    __shared__ __align__(16) float s_v[2][Q4P_GDN_TILE * Q4P_GDN_VALUE];
    __shared__ __align__(16) float2 s_db[2][Q4P_GDN_TILE];
    __shared__ __align__(16) float s_cd[2][kChannel ? Q4P_GDN_TILE * Q4P_GDN_HALF : 4];
    if (value_dim != Q4P_GDN_VALUE || (int)blockDim.x != 4 * Q4P_GDN_VALUE) {
        return;
    }
    const int head = blockIdx.x;
    const int kh = head / groups;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int column = (tid >> 5) * 8 + (lane >> 2);
    const int quarter = lane & 3;
    const int conv_width = key_width * 2 + heads * Q4P_GDN_VALUE;
    const int val_width = heads * Q4P_GDN_VALUE;
    const unsigned long long cells = (unsigned long long)Q4P_GDN_KEY * Q4P_GDN_VALUE;
    // This lane's 32 cells: keys quarter*32 + i of `column`, Q4P_GDN_VALUE floats apart.
    const unsigned long long lane_cell = (unsigned long long)quarter * 32 * Q4P_GDN_VALUE + column;
    float* recurrent = state + (unsigned long long)head * cells + lane_cell;

    float s[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        s[i] = recurrent[i * Q4P_GDN_VALUE];
    }

    const int v_chunks = Q4P_GDN_VALUE >> 2;
    auto stage = [&](int buffer, int t0) {
        const int rows = min(Q4P_GDN_TILE, batch - t0);
        // q / k: Q4P_GDN_ROW floats a token, in 16-byte chunks.
        for (int c = tid; c < rows * (Q4P_GDN_ROW / 4); c += blockDim.x) {
            const int r = c / (Q4P_GDN_ROW / 4);
            const int within = c - r * (Q4P_GDN_ROW / 4);
            const float* src =
                qk + ((unsigned long long)(t0 + r) * key_heads + kh) * Q4P_GDN_ROW + within * 4;
            om_cp16(om_smem(&s_qk[buffer][r * Q4P_GDN_ROW + within * 4]), src, true);
        }
        for (int c = tid; c < rows * v_chunks; c += blockDim.x) {
            const int r = c / v_chunks;
            const int within = c - r * v_chunks;
            const float* src = mixed + (unsigned long long)(t0 + r) * conv_width + key_width * 2 +
                               head * Q4P_GDN_VALUE + within * 4;
            om_cp16(om_smem(&s_v[buffer][r * Q4P_GDN_VALUE + within * 4]), src, true);
        }
        if (tid < rows) {
            s_db[buffer][tid] = decay_beta[(unsigned long long)(t0 + tid) * heads + head];
        }
        if constexpr (kChannel) {
            for (int c = tid; c < rows * (Q4P_GDN_KEY / 4); c += blockDim.x) {
                const int r = c / (Q4P_GDN_KEY / 4);
                const int within = c - r * (Q4P_GDN_KEY / 4);
                const float* src = channel +
                                   ((unsigned long long)(t0 + r) * heads + head) * Q4P_GDN_KEY +
                                   within * 4;
                om_cp16(om_smem(&s_cd[buffer][r * Q4P_GDN_HALF + q4p_gdn_pad(within * 4)]), src,
                        true);
            }
        }
    };

    stage(0, 0);
    om_commit();
    int buffer = 0;
    for (int t0 = 0; t0 < batch; t0 += Q4P_GDN_TILE) {
        if (t0 + Q4P_GDN_TILE < batch) {
            stage(buffer ^ 1, t0 + Q4P_GDN_TILE);
        }
        om_commit();
        om_wait<1>();
        __syncthreads();
        const int rows = min(Q4P_GDN_TILE, batch - t0);
        for (int r = 0; r < rows; ++r) {
            const float* q = &s_qk[buffer][r * Q4P_GDN_ROW + quarter * 36];
            const float* k = q + Q4P_GDN_HALF;
            const float2 db = s_db[buffer][r];
            const float v = s_v[buffer][r * Q4P_GDN_VALUE + column];
            float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int i = 0; i < 32; i += 4) {
                const float4 kk = *reinterpret_cast<const float4*>(&k[i]);
                if constexpr (kChannel) {
                    const float4 dd = *reinterpret_cast<const float4*>(
                        &s_cd[buffer][r * Q4P_GDN_HALF + quarter * 36 + i]);
                    s[i] *= dd.x;
                    s[i + 1] *= dd.y;
                    s[i + 2] *= dd.z;
                    s[i + 3] *= dd.w;
                } else {
                    s[i] *= db.x;
                    s[i + 1] *= db.x;
                    s[i + 2] *= db.x;
                    s[i + 3] *= db.x;
                }
                m[0] += s[i] * kk.x;
                m[1] += s[i + 1] * kk.y;
                m[2] += s[i + 2] * kk.z;
                m[3] += s[i + 3] * kk.w;
            }
            float memory = (m[0] + m[1]) + (m[2] + m[3]);
            memory += __shfl_xor_sync(0xffffffffu, memory, 1);
            memory += __shfl_xor_sync(0xffffffffu, memory, 2);
            const float delta = (v - memory) * db.y;
            float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int i = 0; i < 32; i += 4) {
                const float4 kk = *reinterpret_cast<const float4*>(&k[i]);
                const float4 qq = *reinterpret_cast<const float4*>(&q[i]);
                s[i] += kk.x * delta;
                s[i + 1] += kk.y * delta;
                s[i + 2] += kk.z * delta;
                s[i + 3] += kk.w * delta;
                o[0] += s[i] * qq.x;
                o[1] += s[i + 1] * qq.y;
                o[2] += s[i + 2] * qq.z;
                o[3] += s[i + 3] * qq.w;
            }
            float output = (o[0] + o[1]) + (o[2] + o[3]);
            output += __shfl_xor_sync(0xffffffffu, output, 1);
            output += __shfl_xor_sync(0xffffffffu, output, 2);
            const int token = t0 + r;
            if (quarter == 0) {
                out[(unsigned long long)token * val_width + head * Q4P_GDN_VALUE + column] = output;
            }
            if (snapshot != nullptr && token < snapshot_rows) {
                float* destination = snapshot + (unsigned long long)token * region_stride +
                                     (unsigned long long)head * cells + lane_cell;
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    destination[i * Q4P_GDN_VALUE] = s[i];
                }
            }
        }
        // No thread may still be reading this buffer when the stage after next lands in it.
        __syncthreads();
        buffer ^= 1;
    }
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        recurrent[i * Q4P_GDN_VALUE] = s[i];
    }
}

// Gated DeltaNet's scan: one factor a head, from q4p_gdn_prep's decay_beta.
extern "C" __global__ __launch_bounds__(512) void q4p_gdn_scan(
    const float* __restrict__ qk, const float2* __restrict__ decay_beta,
    const float* __restrict__ mixed, float* __restrict__ state, float* __restrict__ out,
    int key_width, int key_heads, int heads, int value_dim, int groups, int batch,
    float* __restrict__ snapshot, int snapshot_rows, long long region_stride) {
    q4p_gdn_scan_body<false>(qk, decay_beta, nullptr, mixed, state, out, key_width, key_heads,
                             heads, value_dim, groups, batch, snapshot, snapshot_rows,
                             region_stride);
}

// Kimi Delta Attention's scan: one factor a key channel from `channel_decay`, beta from
// q4p_kda_prep's decay_beta. The same grid and threads as q4p_gdn_scan.
extern "C" __global__ __launch_bounds__(512) void q4p_kda_scan(
    const float* __restrict__ qk, const float2* __restrict__ decay_beta,
    const float* __restrict__ channel_decay, const float* __restrict__ mixed,
    float* __restrict__ state, float* __restrict__ out, int key_width, int key_heads, int heads,
    int value_dim, int groups, int batch, float* __restrict__ snapshot, int snapshot_rows,
    long long region_stride) {
    q4p_gdn_scan_body<true>(qk, decay_beta, channel_decay, mixed, state, out, key_width,
                            key_heads, heads, value_dim, groups, batch, snapshot, snapshot_rows,
                            region_stride);
}

// The gated RMSNorm on the scan's raw head outputs, as q4e_gdn_recur's tail computes it.
// Grid (heads, tokens), value_dim threads (a multiple of 32, at most 1024).
extern "C" __global__ void q4p_gdn_gate(const float* __restrict__ raw,
                                        const float* __restrict__ z,
                                        const __nv_bfloat16* __restrict__ norm,
                                        float* __restrict__ out, int heads, int value_dim,
                                        float eps, int sigmoid_gate) {
    __shared__ float reduce[32];
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int c = threadIdx.x;
    const unsigned long long at = ((unsigned long long)token * heads + head) * value_dim + c;
    const float value = raw[at];
    float sum = value * value;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    if ((c & 31) == 0) {
        reduce[c >> 5] = sum;
    }
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < (value_dim >> 5); ++w) {
        total += reduce[w];
    }
    const float out_scale = rsqrtf(total / (float)value_dim + eps);
    const float normed = value * out_scale * __bfloat162float(norm[c]);
    const float gated = z[at];
    const float activated = sigmoid_gate ? (1.0f / (1.0f + __expf(-gated))) : q4p_silu(gated);
    out[at] = normed * activated;
}

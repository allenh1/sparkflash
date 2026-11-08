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
//   - q4p_moe_gate_up / q4p_moe_down: the routed experts as grouped W4A16 MMA. The shared plan
//     (kernels/moe_plan_sm121.cu) sorts the chunk's (token, slot) pairs by expert into items of
//     up to 32 rows ([MOE_ROWS]); each
//     CTA multiplies one item by one panel of an expert's NVFP4 weights, decoding E2M1 x E4M3 to
//     bf16 in registers (the product is exact in bf16) and applying weight_scale_2 in f32. The
//     activations - the expert input and the SwiGLU rows between the passes - are a bf16 head and
//     remainder, two MMAs a fragment, as the dense projections.
//   - q4p_gdn_conv / q4p_gdn_conv_state, q4p_gdn_prep / q4p_gdn_scan / q4p_gdn_gate and
//     q4p_kda_prep / q4p_kda_scan (Gated DeltaNet and Kimi Delta Attention) live in their own
//     image, kernels/qwen4_scan_sm121.cu, not in this cubin.
//
// The token kernels remain the f32 reference these are tested against. The MMA forms carry each
// activation as a bf16 head and remainder (16 significant bits) and accumulate in f32, so a chunk
// through this image matches the reference to a tolerance, not bit for bit. A decode step and a
// speculative verify run here too (SPARKFLASH_DECODE_MMA, on by default): every form computes a
// row from that row alone, in an order fixed by the matrix, so a row gets the same bits at every
// row count, and a continued conversation resumes on the state a cold prefill of it leaves.
//
// Attribution:
//   - Model architecture (Qwen4ExpForConditionalGeneration: Gated DeltaNet linear attention,
//     512-expert top-10 MoE) originates from Qwen/Qwen3.8-Flash-Next by the Qwen team at Alibaba
//     Cloud and the vLLM reference implementation (https://github.com/vllm-project/vllm,
//     Apache-2.0). The Gated DeltaNet recurrence is that of Yang, Kautz and Hatamizadeh, "Gated
//     Delta Networks: Improving Mamba2 with Delta Rule" (ICLR 2025), computed here as a
//     sequential scan in the same order as this crate's host oracle (src/forward.rs).
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
// Every q4p kernel opens with OPT_PDL_ENTRY, or splits it around its first weight loads: the host
// launches them with programmatic stream serialization, as it does the token cubin's.
#include "opt_pdl_sm121.cuh"
#include "q4e_common_sm121.cuh"
// The 128 x 128 bulk-prefill tile body GLM's dense and routed-expert images launch: Qwen's
// block-scaled FP8 experts run their grouped prefill MMA on the same body (D95).
#include "shared_prefill_mma_sm121.cuh"

// The shared header's gate, under the name this image used.
#define q4p_silu q4e_silu


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
    OPT_PDL_ENTRY();
    const unsigned int slice = inner / segments;
    opt_mma_bf16_body<OB_A_F32, OB_W_ROWK, true>(
        a, reinterpret_cast<const uint8_t*>(w), nullptr, out, cols, slice, tokens, inner, cols,
        slice, (unsigned long long)slice * 2ull, 0, (unsigned long long)tokens * cols, inner >> 5);
}

// out[t][c] = sum_i a[t][i] * w[c][i] for the Qwen3.5/3.6 FP8 release's dense projections: an
// E4M3 [cols][inner] matrix with one BF16 inverse scale a 128 x 128 tile ([cols / 128][inner /
// 128]), on the shared bulk-prefill tile GLM's FP8 projections run (the scale typed, D88). The
// grid is (cols / 128, tokens / 128) rounded up, 256 threads; cols must be a multiple of 32, inner
// of 128, and the operands 16-byte aligned (the host checks). A row's sum is that row's alone, in
// an order fixed by the matrix, so it has the same bits at any row count.
extern "C" __global__ __launch_bounds__(OM_THREADS) void q4p_dense_fp8(
    const uint8_t* __restrict__ w, const __nv_bfloat16* __restrict__ ws,
    const float* __restrict__ a, float* __restrict__ out, unsigned int cols, unsigned int inner,
    unsigned int tokens) {
    OPT_PDL_ENTRY();
    shared_prefill_mma_body<SPM_W_FP8, false, true, __nv_bfloat16>(
        a, w, ws, nullptr, nullptr, out, cols, inner, tokens, inner, cols, 0, 0);
}

// A weight load for a stream read once: no L1 line, and the L2 fetches the 256 bytes around it,
// so a lane's 16 bytes and its neighbours' come from DRAM as one burst.
__device__ __forceinline__ uint4 q4p_stream16(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
    return r;
}

// Word `i` of four, without indexing registers.
__device__ __forceinline__ uint32_t q4p_word(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3, unsigned int i) {
    const uint32_t low = (i & 1u) ? w1 : w0;
    const uint32_t high = (i & 1u) ? w3 : w2;
    return (i & 2u) ? high : low;
}

// The B fragments of a 32-column K block from four lanes' 16-byte reads of one weight row. Lane q
// of the quad holds column pairs 4q..4q+3; an m16n8k16 B fragment wants pairs q and q + 4 for the
// first k16 step and q + 8 and q + 12 for the second, the registers ldmatrix gives it. b[i] is
// word q of quad lane i: a 4 x 4 transpose, four shuffles between two lane-dependent selections.
__device__ __forceinline__ void q4p_quad_transpose(const uint4 w, unsigned int q, uint32_t b[4]) {
    uint32_t v[4];
#pragma unroll
    for (unsigned int j = 0; j < 4u; ++j) {
        const uint32_t send = q4p_word(w.x, w.y, w.z, w.w, (q - j) & 3u);
        v[j] = __shfl_sync(0xffffffffu, send, (threadIdx.x & 28u) | ((q + j) & 3u));
    }
#pragma unroll
    for (unsigned int i = 0; i < 4u; ++i) b[i] = q4p_word(v[0], v[1], v[2], v[3], (i - q) & 3u);
}

// The token GEMM's epilogues, as q4e_gemm_reduce and the tile's own store apply them to a finished
// sum: 1 = silu(v / scale), 2 = sigmoid(v / scale) scaled to two, otherwise the sum itself.
__device__ __forceinline__ float q4p_finish(float value, int epilogue, float scale) {
    if (epilogue == 1) {
        return q4p_silu(value / scale);
    }
    if (epilogue == 2) {
        return 2.0f / (1.0f + __expf(-value / scale));
    }
    return value;
}

// Whether this CTA is the last of a split tile's `segments` to write its slice, the tiled GEMV's
// arrival count (q4e_gemv_tiled_bf16): one counter a tile, zero between launches, which the last
// CTA resets. The last one then reads the others' slices past L1 (__ldcg) and adds them in order,
// so a sum is q4e_gemm_reduce's whichever slice lands last.
__device__ __forceinline__ bool q4p_last_slice(unsigned int* ticket, unsigned int segments) {
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(ticket, 1u) == segments - 1u;
        if (last) {
            *ticket = 0u;
        }
    }
    __syncthreads();
    if (last) {
        __threadfence();
    }
    return last;
}

// Weights a q4p_gemm_rows CTA stages a pass: 16 KB over its rows.
#define Q4P_GEMM_ROWS_STAGE 8192

// The token cubin's q4e_gemm_bf16 for a few rows, to the same bits: the projections whose shape
// the MMA tile does not take (a row count off the 32 grid - the hyper-connection's four-row
// injection, the GDN gates, the shared-expert gate) run there at every row count. An output is
// one f32 FMA chain over its slice of the inner dimension, in order - the slice q4e_gemm_bf16's
// blockIdx.y owns, `per` columns rounded up to its 32-column stage - so one thread an output
// gives it the bits a thread of the 64 x 128 tile does, without the tile's other 8,191 outputs.
// The CTA's 128 threads first stage the slice of every row its outputs read in shared memory,
// 16 bytes a load, so a chain waits on one round trip rather than one a load; a matrix whose rows
// are not whole 16-byte loads reads its weights directly. The grid is (rows * batch / 128 rounded
// up, segments, runs), 128 threads; the arguments and the partials are q4e_gemm_bf16's, and a
// split's last CTA adds the slices in order and applies the epilogue as q4e_gemm_reduce does,
// with `tickets` a zeroed counter for each of the launch's (rows * batch / 128) * runs tiles.
//
// q4p_gemm_rows_cta is one CTA of that grid, (bx, by, bz) of `gx` along x, its weights staged in
// `staged`: q4p_dense_rows runs it for the matrices of its input that its tile does not take.
__device__ __forceinline__ void q4p_gemm_rows_cta(
    const __nv_bfloat16* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int rows,
    int columns, int batch, int epilogue, float scale, int segments, float* __restrict__ partials,
    unsigned int* __restrict__ tickets, unsigned int bx, unsigned int by, unsigned int bz,
    unsigned int gx, __nv_bfloat16* __restrict__ staged) {
    const unsigned long long run = bz;
    x += run * batch * columns;
    y += run * batch * rows;
    const unsigned long long count = (unsigned long long)rows * batch;
    if (segments > 1) {
        partials += run * segments * count;
    }
    const int first = (int)(bx * blockDim.x);
    if (columns <= 0 || first >= rows * batch) {
        return;
    }
    const int index = first + (int)threadIdx.x;
    const bool mine = index < rows * batch;
    const int row = mine ? index / batch : 0;
    const int token = mine ? index - row * batch : 0;
    const int segment = segments > 1 ? (int)by : 0;
    int k_begin = 0;
    int k_end = columns;
    if (segments > 1) {
        // q4e_gemm_bf16's QG_KB-rounded slice.
        const int per = ((columns + segments - 1) / segments + 31) / 32 * 32;
        k_begin = min(columns, segment * per);
        k_end = min(columns, k_begin + per);
    }
    const float* activations = x + (unsigned long long)token * columns;
    float value = 0.0f;
    if ((columns & 7) == 0 && (reinterpret_cast<unsigned long long>(w) & 15ull) == 0ull) {
        // The CTA's rows, from its first output's to its last's, each staged at a stride of whole
        // 16-byte parts plus one, so the eight lanes of a quarter warp reading a part each of
        // eight rows find their banks apart.
        const int low_row = first / batch;
        const int rows_here = (min(rows * batch, first + (int)blockDim.x) - 1) / batch - low_row + 1;
        const int width = max(8, Q4P_GEMM_ROWS_STAGE / rows_here - 8) & ~7;
        for (int k0 = k_begin; k0 < k_end; k0 += width) {
            const int span = min(width, k_end - k0);
            const int parts = span >> 3;
            const int stride = span + 8;
            const int total = rows_here * parts;
            __syncthreads();
            for (int base = (int)threadIdx.x; base < total; base += 4 * (int)blockDim.x) {
                uint4 v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = base + j * (int)blockDim.x;
                    const int r = c / parts;
                    v[j] = c < total ? __ldg(reinterpret_cast<const uint4*>(
                                           w + (unsigned long long)(low_row + r) * columns + k0 +
                                           8 * (c - r * parts)))
                                     : make_uint4(0u, 0u, 0u, 0u);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = base + j * (int)blockDim.x;
                    const int r = c / parts;
                    if (c < total) {
                        *reinterpret_cast<uint4*>(staged + r * stride + 8 * (c - r * parts)) = v[j];
                    }
                }
            }
            __syncthreads();
            if (mine) {
                const __nv_bfloat16* weights = staged + (row - low_row) * stride;
                for (int part = 0; part < parts; ++part) {
                    const uint4 packed = *reinterpret_cast<const uint4*>(weights + 8 * part);
                    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&packed);
                    const float* at = activations + k0 + 8 * part;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        value = fmaf(__bfloat162float(h[j]), at[j], value);
                    }
                }
            }
        }
    } else if (mine) {
        const __nv_bfloat16* weights = w + (unsigned long long)row * columns;
#pragma unroll 8
        for (int k = k_begin; k < k_end; ++k) {
            value = fmaf(__bfloat162float(weights[k]), activations[k], value);
        }
    }
    if (segments == 1) {
        if (mine) {
            y[(unsigned long long)token * rows + row] = q4p_finish(value, epilogue, scale);
        }
        return;
    }
    if (mine) {
        partials[(unsigned long long)segment * count + (unsigned long long)token * rows + row] = value;
    }
    if (!q4p_last_slice(tickets + run * gx + bx, (unsigned int)segments)) {
        return;
    }
    if (mine) {
        const unsigned long long at = (unsigned long long)token * rows + row;
        float total = __ldcg(partials + at);
        for (int other = 1; other < segments; ++other) {
            total += __ldcg(partials + (unsigned long long)other * count + at);
        }
        y[at] = q4p_finish(total, epilogue, scale);
    }
}

extern "C" __global__ __launch_bounds__(128) void q4p_gemm_rows(
    const __nv_bfloat16* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int rows,
    int columns, int batch, int epilogue, float scale, int segments, float* __restrict__ partials,
    unsigned int* __restrict__ tickets) {
    __shared__ __align__(16) __nv_bfloat16 staged[Q4P_GEMM_ROWS_STAGE];
    OPT_PDL_ENTRY();
    q4p_gemm_rows_cta(w, x, y, rows, columns, batch, epilogue, scale, segments, partials, tickets,
                      blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, staged);
}

// K blocks a q4p_dense_rows round reads: one 512-byte load of each of a warp's eight rows.
#define Q4P_ROWS_ROUND 8u
// A staged row's bytes: the round's 512 and 16 more, so ldmatrix's eight rows of a K block start
// in different banks.
#define Q4P_ROWS_STRIDE (Q4P_ROWS_ROUND * 64u + 16u)

// One warp's m16 x n8 tile of q4p_dense_rows (described below) is weight rows n0 to n0 + 7 of `w`
// against activation rows m0 to m0 + 15 of `a`, over the `k_blocks` 32-column K blocks from column
// `k_at`. A kernel that runs one starts the warp's weight stream at q4p_rows_row, requests the
// first round with q4p_rows_fetch, waits for the previous grid (OPT_PDL_WAIT: the weights are
// constant, the activations are not), and then runs q4p_rows_tile. Every kernel that takes a tile
// runs this one chain, so a tile has the same bits in each.
//
// Lane l's 16 bytes of a row's 512 a round are part l % 4 of the round's K block l / 4; a warp that
// is not `live` reads nothing and computes zeros.
__device__ __forceinline__ const __nv_bfloat16* q4p_rows_row(const __nv_bfloat16* w,
                                                            unsigned int inner, unsigned int n0,
                                                            bool live, unsigned long long k_at) {
    return w + (unsigned long long)(live ? n0 : 0u) * inner + k_at + 8u * (threadIdx.x & 31u);
}

// The round of K blocks from kb0: 512 bytes of each of the warp's eight rows, zeros past the slice.
__device__ __forceinline__ void q4p_rows_fetch(const __nv_bfloat16* row, unsigned int inner,
                                               unsigned int kb0, unsigned int k_blocks, bool live,
                                               uint4 next[8]) {
    const unsigned int lane = threadIdx.x & 31u;
#pragma unroll
    for (unsigned int i = 0; i < 8u; ++i) {
        next[i] = live && kb0 + (lane >> 2) < k_blocks
                      ? q4p_stream16(row + (unsigned long long)i * inner + kb0 * 32u)
                      : make_uint4(0u, 0u, 0u, 0u);
    }
}

// The tile's MMA chain from the first round in `next`, into the C fragment `acc` - rows g and
// g + 8, columns 2q and 2q + 1. `tile` is the warp's own 8 x Q4P_ROWS_STRIDE shared stage.
__device__ __forceinline__ void q4p_rows_tile(const __nv_bfloat16* row, const float* __restrict__ a,
                                              unsigned int inner, unsigned int tokens, bool live,
                                              unsigned int m0, unsigned long long k_at,
                                              unsigned int k_blocks, unsigned char* tile,
                                              uint4 next[8], float acc[4]) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int q = lane & 3u;
    const bool low = m0 + g < tokens;
    const bool high = m0 + g + 8u < tokens;
    const float* a_low = a + (unsigned long long)(m0 + g) * inner + k_at + 2u * q;
    const float* a_high = a_low + 8ull * inner;
    // ldmatrix's row addresses: lane l points its matrix l / 8 - the 16-byte part j of a K block,
    // columns 8j to 8j + 7 - at staged row l % 8, so b[j] is word q of row g's part j.
    const uint32_t parts = om_smem(tile + (lane & 7u) * Q4P_ROWS_STRIDE + 16u * (lane >> 3));
#pragma unroll
    for (unsigned int r = 0; r < 4u; ++r) acc[r] = 0.0f;
    for (unsigned int kb0 = 0; kb0 < k_blocks; kb0 += Q4P_ROWS_ROUND) {
        __syncwarp();
#pragma unroll
        for (unsigned int i = 0; i < 8u; ++i) {
            *reinterpret_cast<uint4*>(tile + i * Q4P_ROWS_STRIDE + 16u * lane) = next[i];
        }
        __syncwarp();
        if (kb0 + Q4P_ROWS_ROUND < k_blocks) {
            q4p_rows_fetch(row, inner, kb0 + Q4P_ROWS_ROUND, k_blocks, live, next);
        }
#pragma unroll
        for (unsigned int u = 0; u < Q4P_ROWS_ROUND; ++u) {
            if (kb0 + u >= k_blocks) break;
            uint32_t b[4];
            om_ldsm4(parts + u * 64u, b[0], b[1], b[2], b[3]);
#pragma unroll
            for (unsigned int ks = 0; ks < 2u; ++ks) {
                // The m16n8k16 A fragment, rows g and g + 8, columns 2q and 2q + 8 of the step,
                // split as opt_mma_bf16_body's store() splits it.
                const unsigned int k = (kb0 + u) * 32u + ks * 16u;
                const float2 zero = make_float2(0.0f, 0.0f);
                const float2 v[4] = {
                    low ? __ldg(reinterpret_cast<const float2*>(a_low + k)) : zero,
                    high ? __ldg(reinterpret_cast<const float2*>(a_high + k)) : zero,
                    low ? __ldg(reinterpret_cast<const float2*>(a_low + k + 8u)) : zero,
                    high ? __ldg(reinterpret_cast<const float2*>(a_high + k + 8u)) : zero,
                };
                uint32_t head[4];
                uint32_t rest[4];
#pragma unroll
                for (unsigned int r = 0; r < 4u; ++r) {
                    const __nv_bfloat16 h0 = __float2bfloat16_rn(v[r].x);
                    const __nv_bfloat16 h1 = __float2bfloat16_rn(v[r].y);
                    head[r] = ob_pack(h0, h1);
                    rest[r] = ob_pack(__float2bfloat16_rn(v[r].x - __bfloat162float(h0)),
                                      __float2bfloat16_rn(v[r].y - __bfloat162float(h1)));
                }
                ob_mma(acc, head, b[2u * ks], b[2u * ks + 1u]);
                ob_mma(acc, rest, b[2u * ks], b[2u * ks + 1u]);
            }
        }
    }
}

// q4p_dense_mma for a few rows - a decode step, a speculative verify, a short chunk - to the same
// bits. An output is the same chain of MMAs in both: its slice's 32-column K blocks in order, each
// two 16-column steps of a head MMA and then a remainder MMA, into an f32 accumulator that starts
// at zero, with the head and remainder rounded as opt_mma_bf16_body rounds them, and every weight
// and activation in the MMA slot it has there. Only the tile differs. The 128-row tile computes
// 127 zero rows for one and keeps one K block in flight; here a warp is one m16 tile of rows by 8
// columns, and a round is one 512-byte load of each of its eight weight rows - a lane 16 bytes,
// eight K blocks, the stream a plain GEMV reads - staged in the warp's own shared tile and taken
// back out with ldmatrix into the B fragments the 128-row tile's ldmatrix fills. The next round
// streams while this one computes, and the activations come from L1.
//
// One launch takes up to four matrices of the same inner dimension over the same rows - a layer's
// projections of one input - laid end to end along x, `count` of them: matrix m is `w[m]`, its
// `cols[m]` columns split `segments[m]` ways, its slices `out[m]` and its finished outputs `y[m]`,
// and a CTA computes one K slice of 32 columns of one of them exactly as a launch of that matrix
// alone would. Matrix m takes (cols[m] / 32 rounded up) * segments[m] CTAs along x, a tile's
// slices next to each other, so matrices of different splits (the attention's wide query beside
// its narrow keys, values and indexer) share a launch. The grid is (sum of those, tokens / 16, 1)
// rounded up, 128 threads, under q4p_dense_mma's constraints.
//
// One slice (`segments[m]` 1) stores finish(sum) to `out[m]`. A split stores slice s's sums to
// out[m][s] as q4p_dense_mma does, and the tile's last CTA to land adds the slices in order into
// `y[m]` and finishes them, as q4e_gemm_reduce would: `tickets` holds a zeroed counter for each of
// the launch's split tiles, matrix by matrix, each split matrix's (cols[m] / 32) * (tokens / 16)
// of them; an unsplit matrix takes none.
//
// The matrices of the same input whose shape the tile does not take - the GDN gates beside the
// in-projections, the injection beside the hyper-connection's down projection, the shared-expert
// gate beside the router - ride along instead of taking a launch each: up to two `riders`, each
// q4p_gemm_rows's arguments over `a` (`inner` its columns, `tokens` its batch, its own split,
// epilogue, partials and tickets). The CTAs past the `dense_x` along x of every y are numbered in
// order and run rider 0's grid (rows * tokens / 128 rounded up, split, 1), then rider 1's, each
// CTA exactly as q4p_gemm_rows runs it.
extern "C" __global__ __launch_bounds__(128) void q4p_dense_rows(
    const float* __restrict__ a, unsigned int inner, unsigned int tokens,
    unsigned int* __restrict__ tickets, int epilogue, float scale, unsigned int count,
    const __nv_bfloat16* w0, float* out0, float* y0, unsigned int cols0, unsigned int segments0,
    const __nv_bfloat16* w1, float* out1, float* y1, unsigned int cols1, unsigned int segments1,
    const __nv_bfloat16* w2, float* out2, float* y2, unsigned int cols2, unsigned int segments2,
    const __nv_bfloat16* w3, float* out3, float* y3, unsigned int cols3, unsigned int segments3,
    unsigned int dense_x, unsigned int riders, const __nv_bfloat16* rw0, float* ry0, int rows0,
    int epilogue0, float scale0, int split0, float* partials0, unsigned int* tickets0,
    const __nv_bfloat16* rw1, float* ry1, int rows1, int epilogue1, float scale1, int split1,
    float* partials1, unsigned int* tickets1) {
    __shared__ __align__(16) unsigned char stage[4][8u * Q4P_ROWS_STRIDE];
    static_assert(sizeof(stage) >= Q4P_GEMM_ROWS_STAGE * sizeof(__nv_bfloat16),
                  "a rider stages its weights in the dense tile's stage");
    if (blockIdx.x >= dense_x) {
        unsigned int cta = blockIdx.x - dense_x + (gridDim.x - dense_x) * blockIdx.y;
        OPT_PDL_ENTRY();
        const auto ride = [&](const __nv_bfloat16* rw, float* ry, int rows, int rider_epilogue,
                              float rider_scale, int split, float* partials,
                              unsigned int* rider_tickets) {
            const unsigned int blocks = ((unsigned int)rows * tokens + 127u) / 128u;
            if (cta >= blocks * (unsigned int)split) {
                cta -= blocks * (unsigned int)split;
                return false;
            }
            q4p_gemm_rows_cta(rw, a, ry, rows, (int)inner, (int)tokens, rider_epilogue,
                              rider_scale, split, partials, rider_tickets, cta % blocks,
                              cta / blocks, 0u, blocks, reinterpret_cast<__nv_bfloat16*>(stage));
            return true;
        };
        if (riders > 0u && !ride(rw0, ry0, rows0, epilogue0, scale0, split0, partials0, tickets0) &&
            riders > 1u) {
            ride(rw1, ry1, rows1, epilogue1, scale1, split1, partials1, tickets1);
        }
        return;
    }
    // This CTA's matrix: the one whose run of CTAs along x holds blockIdx.x. `first` counts the
    // CTAs before it, `counted` the split tiles before it along one y (its tickets' offset).
    const __nv_bfloat16* w = w0;
    float* out = out0;
    float* y = y0;
    unsigned int cols = cols0;
    unsigned int segments = segments0;
    unsigned int first = 0;
    unsigned int counted = 0;
    unsigned int across = (cols0 + 31u) / 32u;
    const auto pick = [&](unsigned int m, const __nv_bfloat16* wm, float* om, float* ym,
                          unsigned int cm, unsigned int sm) {
        if (count > m && blockIdx.x >= first + across * segments) {
            first += across * segments;
            counted += segments > 1u ? across : 0u;
            w = wm;
            out = om;
            y = ym;
            cols = cm;
            segments = sm;
            across = (cm + 31u) / 32u;
        }
    };
    pick(1u, w1, out1, y1, cols1, segments1);
    pick(2u, w2, out2, y2, cols2, segments2);
    pick(3u, w3, out3, y3, cols3, segments3);
    const unsigned int bx = (blockIdx.x - first) / segments;
    const unsigned int z = blockIdx.x - first - bx * segments;
    // The weights are constant: the first round streams while the previous kernel finishes.
    OPT_PDL_LAUNCH();
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int g = lane >> 2;
    const unsigned int q = lane & 3u;
    const unsigned int n0 = (bx * 4u + warp) * 8u;
    const bool live = n0 < cols;
    const unsigned int m0 = blockIdx.y * 16u;
    const unsigned int slice = inner / segments;
    const unsigned int k_blocks = slice >> 5;
    const unsigned long long k_at = (unsigned long long)z * slice;
    const __nv_bfloat16* row = q4p_rows_row(w, inner, n0, live, k_at);
    uint4 next[8];
    q4p_rows_fetch(row, inner, 0u, k_blocks, live, next);
    OPT_PDL_WAIT();
    float acc[4];
    q4p_rows_tile(row, a, inner, tokens, live, m0, k_at, k_blocks, stage[warp], next, acc);
    const unsigned long long plane = (unsigned long long)tokens * cols;
#pragma unroll
    for (unsigned int half = 0; half < 2u; ++half) {
        const unsigned int token = m0 + g + 8u * half;
        if (!live || token >= tokens) continue;
        float2 pair = make_float2(acc[2u * half], acc[2u * half + 1u]);
        if (segments == 1u) {
            pair = make_float2(q4p_finish(pair.x, epilogue, scale), q4p_finish(pair.y, epilogue, scale));
        }
        *reinterpret_cast<float2*>(out + z * plane + (unsigned long long)token * cols + n0 + 2u * q) =
            pair;
    }
    if (segments == 1u ||
        !q4p_last_slice(tickets + counted * gridDim.y + blockIdx.y * across + bx, segments)) {
        return;
    }
    // The tile's 32 columns of its tokens, four outputs a thread.
    for (unsigned int e = threadIdx.x; e < 16u * 32u; e += blockDim.x) {
        const unsigned int token = m0 + e / 32u;
        const unsigned int column = bx * 32u + e % 32u;
        if (token >= tokens || column >= cols) continue;
        const unsigned long long at = (unsigned long long)token * cols + column;
        float value = __ldcg(out + at);
        for (unsigned int segment = 1; segment < segments; ++segment) {
            value += __ldcg(out + segment * plane + at);
        }
        y[at] = q4p_finish(value, epilogue, scale);
    }
}

// The hyper-connection's up projection and stream mix for a few rows, in one launch: the raw sums
// a q4p_dense_rows launch of the up projection gives (one slice, no epilogue), mixed as
// q4e_hyper_mix mixes them, to the same bits. Warp s of CTA b takes the n8 tile of stream s's rows
// s * hidden + 8b to 8b + 7 - one of the tiles the plain launch runs, the same MMA chain - and the
// CTA then mixes its eight columns of every stream: summed from zero a stream at a time,
// sigmoid(raw) * normed, over hc. The grid is (hidden / 8, tokens / 16, 1) rounded up, 128
// threads, hc at most 4, `lowrank` whole 32-column K blocks; `low` is the down projection's
// [tokens][lowrank] and `normed` the [tokens][hc][hidden] streams.
extern "C" __global__ __launch_bounds__(128) void q4p_hyper_up_rows(
    const float* __restrict__ low, const __nv_bfloat16* __restrict__ up,
    const float* __restrict__ normed, float* __restrict__ mixed, unsigned int lowrank,
    unsigned int hidden, unsigned int hc, unsigned int tokens) {
    __shared__ __align__(16) unsigned char stage[4][8u * Q4P_ROWS_STRIDE];
    __shared__ float raw[4][16][8];
    OPT_PDL_LAUNCH();
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int g = lane >> 2;
    const unsigned int q = lane & 3u;
    const unsigned int column0 = blockIdx.x * 8u;
    const unsigned int m0 = blockIdx.y * 16u;
    const bool live = warp < hc && column0 < hidden;
    const unsigned int k_blocks = lowrank >> 5;
    const __nv_bfloat16* row = q4p_rows_row(up, lowrank, warp * hidden + column0, live, 0ull);
    uint4 next[8];
    q4p_rows_fetch(row, lowrank, 0u, k_blocks, live, next);
    OPT_PDL_WAIT();
    float acc[4];
    q4p_rows_tile(row, low, lowrank, tokens, live, m0, 0ull, k_blocks, stage[warp], next, acc);
    raw[warp][g][2u * q] = acc[0];
    raw[warp][g][2u * q + 1u] = acc[1];
    raw[warp][g + 8u][2u * q] = acc[2];
    raw[warp][g + 8u][2u * q + 1u] = acc[3];
    __syncthreads();
    const unsigned int t = threadIdx.x >> 3;
    const unsigned int c = threadIdx.x & 7u;
    const unsigned int token = m0 + t;
    const unsigned int column = column0 + c;
    if (token >= tokens || column >= hidden) return;
    const unsigned long long base = (unsigned long long)token * hc * hidden + column;
    float accumulator = 0.0f;
    for (unsigned int stream = 0; stream < hc; ++stream) {
        const unsigned long long at = base + (unsigned long long)stream * hidden;
        accumulator += (1.0f / (1.0f + __expf(-raw[stream][t][c]))) * normed[at];
    }
    mixed[(unsigned long long)token * hidden + column] = accumulator / (float)hc;
}

// The token GEMM's two epilogues, applied in place after the MMA has written the raw dot
// products: 1 = silu(v / scale), 2 = sigmoid(v / scale) scaled to two.
extern "C" __global__ void q4p_epilogue(float* __restrict__ y, unsigned long long count,
                                        int epilogue, float scale) {
    OPT_PDL_ENTRY();
    for (unsigned long long index = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
         index < count; index += (unsigned long long)gridDim.x * blockDim.x) {
        y[index] = q4p_finish(y[index], epilogue, scale);
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
    OPT_PDL_ENTRY();
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

// The two passes' outputs from their accumulators, shared by both MoE forms so a row is the same
// expression in each: gate_up's SwiGLU of the gate and up sums under their weight_scale_2, and
// down's sum under its weight_scale_2 and the slot's routing weight.
__device__ __forceinline__ float q4p_swiglu(float gate, float up, float g2, float u2) {
    return q4p_silu(gate * g2) * (up * u2);
}
__device__ __forceinline__ float q4p_routed(float sum, float route, float d2) {
    return route * (sum * d2);
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
                const float a0 = q4p_swiglu(acc[mt][0][2 * h], acc[mt][1][2 * h], g2, u2);
                const float a1 = q4p_swiglu(acc[mt][0][2 * h + 1], acc[mt][1][2 * h + 1], g2, u2);
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
                        make_float2(q4p_routed(acc[mt][j][2 * h], route, d2),
                                    q4p_routed(acc[mt][j][2 * h + 1], route, d2));
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
    OPT_PDL_ENTRY();
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
    OPT_PDL_ENTRY();
    q4p_moe_body<false>(table, scales, sorted, items, item_count, activated, activated_plane,
                        route_weights, partials, 0, topk, hidden, intermediate, weight_stride);
}

// K-stage pairs a q4p_moe_rows lane has in flight: 16 bytes of a packed row each (128 columns).
#define Q4P_MOE_ROWS_UNROLL 4

// q4p_moe_body for a few rows - a decode step, a speculative verify - to the same bits. An output
// is the same chain of MMAs in both: its K stages of 64 columns in order, each four k16 steps of
// the head plane and then four of the remainder, with the same permuted k order, the same B
// fragments (E2M1 pairs from the same table times their E4M3 block scale in bf16) and the same
// epilogue. Only the tile differs. The 128-row panel stages 32 bytes of every weight row a K stage
// through shared memory; here a warp is two groups of 8 weight rows - for gate_up its 8
// intermediate columns of the gate and the up matrix, for down 16 output columns - read straight
// into the B fragments: lane (g, t) reads 16 bytes of row g, two K stages, a quad transpose gives
// it words t and t + 4 of each stage as the panel's layout does, and the activations come from L1.
// An item of 16 rows or fewer skips the second m16 tile. Grid (intermediate / 32, max items) for
// gate_up and (hidden / 64, max items) for down, 128 threads; both K dimensions are multiples of
// 128.
template <bool GATE_UP>
__device__ __forceinline__ void q4p_moe_rows_body(
    const unsigned long long* __restrict__ table, const float* __restrict__ scales,
    const int* __restrict__ sorted, const int4* __restrict__ items,
    const int* __restrict__ item_count, const __nv_bfloat16* __restrict__ a_src,
    unsigned long long a_plane, const float* __restrict__ route_weights, void* __restrict__ out,
    unsigned long long out_plane, int topk, int hidden, int intermediate, int weight_stride) {
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
    const int g = lane >> 2;
    const int t = lane & 3;
    const int inner = GATE_UP ? hidden : intermediate;
    const int n0 = GATE_UP ? blockIdx.x * 32 + 8 * warp : blockIdx.x * 64 + 16 * warp;
    const int packed_row = inner >> 1;
    const int scale_row = inner >> 4;
    for (int code = tid; code < 256; code += blockDim.x) {
        lut[code] = q4p_e2m1x2_bf16((unsigned int)code);
    }
    if (tid < Q4P_MOE_ROWS) {
        s_slot[tid] = tid < count ? sorted[first + tid] : -1;
    }
    __syncthreads();
    // Weight row g of each 8-row group, and its scale row.
    const unsigned char* w_row[2];
    const unsigned char* s_row[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const unsigned long long row = GATE_UP ? (unsigned long long)(n0 + g) : (unsigned long long)(n0 + 8 * j + g);
        const int packed = GATE_UP ? (j == 0 ? 0 : 2) : 4;
        w_row[j] = (const unsigned char*)table[expert * 6 + packed] + row * packed_row + 16 * t;
        s_row[j] = (const unsigned char*)table[expert * 6 + packed + 1] + row * scale_row;
    }
    // The item rows this lane's A fragments read: g, g + 8, 16 + g, 24 + g.
    const __nv_bfloat16* a_row[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int slot = s_slot[g + 8 * r];
        a_row[r] = slot < 0 ? nullptr
                            : GATE_UP ? a_src + (unsigned long long)(slot / topk) * hidden + 8 * t
                                      : a_src + (unsigned long long)slot * intermediate + 8 * t;
    }
    const int tiles = count > 16 ? 2 : 1;
    float acc[2][2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    const int pairs = inner >> 7;
    uint4 next[Q4P_MOE_ROWS_UNROLL][2];
    auto fetch = [&](int p0) {
#pragma unroll
        for (int u = 0; u < Q4P_MOE_ROWS_UNROLL; ++u)
#pragma unroll
            for (int j = 0; j < 2; ++j)
                next[u][j] = p0 + u < pairs ? q4p_stream16(w_row[j] + (p0 + u) * 64) : make_uint4(0u, 0u, 0u, 0u);
    };
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    fetch(0);
    for (int p0 = 0; p0 < pairs; p0 += Q4P_MOE_ROWS_UNROLL) {
        uint4 round[Q4P_MOE_ROWS_UNROLL][2];
#pragma unroll
        for (int u = 0; u < Q4P_MOE_ROWS_UNROLL; ++u)
#pragma unroll
            for (int j = 0; j < 2; ++j) round[u][j] = next[u][j];
        if (p0 + Q4P_MOE_ROWS_UNROLL < pairs) fetch(p0 + Q4P_MOE_ROWS_UNROLL);
#pragma unroll
        for (int u = 0; u < Q4P_MOE_ROWS_UNROLL; ++u) {
            if (p0 + u >= pairs) break;
            uint32_t words[2][4];
#pragma unroll
            for (int j = 0; j < 2; ++j) q4p_quad_transpose(round[u][j], (unsigned int)t, words[j]);
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int ks = 2 * (p0 + u) + half;
                uint32_t bf[2][4][2];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const uint32_t w0 = words[j][2 * half];
                    const uint32_t w1 = words[j][2 * half + 1];
                    const uint32_t sw = __ldg(reinterpret_cast<const uint32_t*>(s_row[j] + ks * (Q4P_BK / 16)));
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
                    if (mt >= tiles) break;
#pragma unroll
                    for (int plane = 0; plane < 2; ++plane) {
                        const unsigned long long at = plane * a_plane + ks * Q4P_BK;
                        const __nv_bfloat16* r0 = a_row[2 * mt];
                        const __nv_bfloat16* r8 = a_row[2 * mt + 1];
                        const uint4 lo0 = r0 ? __ldg(reinterpret_cast<const uint4*>(r0 + at)) : zero;
                        const uint4 hi0 = r0 ? __ldg(reinterpret_cast<const uint4*>(r0 + at + 32)) : zero;
                        const uint4 lo8 = r8 ? __ldg(reinterpret_cast<const uint4*>(r8 + at)) : zero;
                        const uint4 hi8 = r8 ? __ldg(reinterpret_cast<const uint4*>(r8 + at + 32)) : zero;
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
        }
    }
    // Epilogue, as q4p_moe_body's. Accumulator (mt, j) element 2h + e is item row
    // mt*16 + g + 8h and column 2t + e of the warp's group j.
    if (GATE_UP) {
        const float g2 = scales[expert * 3 + 0];
        const float u2 = scales[expert * 3 + 1];
        __nv_bfloat16* activated = reinterpret_cast<__nv_bfloat16*>(out);
        const int column = n0 + 2 * t;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mt * 16 + g + 8 * h;
                if (r >= count) {
                    continue;
                }
                const int slot = s_slot[r];
                const float a0 = q4p_swiglu(acc[mt][0][2 * h], acc[mt][1][2 * h], g2, u2);
                const float a1 = q4p_swiglu(acc[mt][0][2 * h + 1], acc[mt][1][2 * h + 1], g2, u2);
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
                    const int column = n0 + 8 * j + 2 * t;
                    *reinterpret_cast<float2*>(&partials[(unsigned long long)slot * hidden + column]) =
                        make_float2(q4p_routed(acc[mt][j][2 * h], route, d2),
                                    q4p_routed(acc[mt][j][2 * h + 1], route, d2));
                }
            }
        }
    }
}

// q4p_moe_gate_up and q4p_moe_down for a few rows: the same arguments, q4p_moe_rows_body's grid.
extern "C" __global__ __launch_bounds__(128) void q4p_moe_rows_gate_up(
    const unsigned long long* __restrict__ table, const float* __restrict__ scales,
    const int* __restrict__ sorted, const int4* __restrict__ items,
    const int* __restrict__ item_count, const __nv_bfloat16* __restrict__ x,
    unsigned long long x_plane, __nv_bfloat16* __restrict__ activated,
    unsigned long long activated_plane, int topk, int hidden, int intermediate) {
    OPT_PDL_ENTRY();
    q4p_moe_rows_body<true>(table, scales, sorted, items, item_count, x, x_plane, nullptr, activated,
                            activated_plane, topk, hidden, intermediate, 0);
}

extern "C" __global__ __launch_bounds__(128) void q4p_moe_rows_down(
    const unsigned long long* __restrict__ table, const float* __restrict__ scales,
    const int* __restrict__ sorted, const int4* __restrict__ items,
    const int* __restrict__ item_count, const __nv_bfloat16* __restrict__ activated,
    unsigned long long activated_plane, const float* __restrict__ route_weights,
    float* __restrict__ partials, int topk, int hidden, int intermediate, int weight_stride) {
    OPT_PDL_ENTRY();
    q4p_moe_rows_body<false>(table, scales, sorted, items, item_count, activated, activated_plane,
                             route_weights, partials, 0, topk, hidden, intermediate, weight_stride);
}

// ---------------------------------------------------------------------------------------------
// Block-scaled FP8 routed experts on the shared 128 x 128 tile (D95).
//
// Qwen's FP8 release stores each expert as an E4M3 gate, up and down matrix with one BF16
// inverse scale a 128 x 128 tile. The token path reads them one expert run at a time on CUDA
// cores; a prefill chunk runs the grouped plan's items through the shared bulk-prefill tile,
// exactly as GLM's FP8 experts do, so a prefill multiplies 128 rows x 128 columns a block on the
// tensor cores. The weight/scale pointer pairs come from the expert table
// [gate w, gate s, up w, up s, down w, down s], typed: a BF16 scale is read as a BF16 (D88).
//
// The activation stays f32. Gate and up read each item's token row and write its slot row of one
// [slots][2 * intermediate] f32 buffer (sorted[first + token], so a plan permutation follows the
// row it names), the SwiGLU kernel maps those rows one to one into [slots][intermediate], and the
// down pass applies the routing weight - probabilities[(slot / topk) * topk + slot % topk] - on
// the way to q4e_moe_combine's [slot][hidden] partials. This is a grouped accumulation order, so
// it is a D67 numerics note against the scalar token path, not a bit-identity claim.
extern "C" __global__ __launch_bounds__(OM_THREADS) void q4p_moe_gate_up_fp8(
    const unsigned long long* __restrict__ table, const int4* __restrict__ items,
    const float* __restrict__ a, float* __restrict__ out, unsigned int cols,
    unsigned int inner, const int* __restrict__ sorted, unsigned int topk, int matrix) {
    OPT_PDL_ENTRY();
    if (matrix == 0) {
        shared_prefill_mma_body<SPM_W_FP8, true, true, __nv_bfloat16, 6, 0, false, true>(
            a, nullptr, nullptr, table, items, out, cols, inner, Q4P_MOE_ROWS, inner,
            2ull * cols, 0ull, (unsigned long long)Q4P_MOE_ROWS * 2ull * cols, 0, 0, sorted,
            topk);
    } else {
        shared_prefill_mma_body<SPM_W_FP8, true, true, __nv_bfloat16, 6, 2, false, true>(
            a, nullptr, nullptr, table, items, out, cols, inner, Q4P_MOE_ROWS, inner,
            2ull * cols, 0ull, (unsigned long long)Q4P_MOE_ROWS * 2ull * cols, 0, 0, sorted,
            topk);
    }
}

// gate_up's [items][32][2 * intermediate] f32 rows -> the SwiGLU'd [slots][intermediate] f32
// rows the down pass takes. One slot row per blockIdx.y.
extern "C" __global__ void q4p_moe_swiglu_fp8(const float* __restrict__ gate_up,
                                               float* __restrict__ activated, int slots,
                                               int intermediate) {
    OPT_PDL_ENTRY();
    const int slot = blockIdx.y;
    if (slot >= slots) return;
    const float* src = gate_up + (unsigned long long)slot * 2u * intermediate;
    float* dst = activated + (unsigned long long)slot * intermediate;
    for (int column = blockIdx.x * blockDim.x + threadIdx.x; column < intermediate;
         column += blockDim.x * gridDim.x) {
        dst[column] = q4p_silu(src[column]) * src[intermediate + column];
    }
}

extern "C" __global__ __launch_bounds__(OM_THREADS) void q4p_moe_down_fp8(
    const unsigned long long* __restrict__ table, const int4* __restrict__ items,
    const float* __restrict__ a, const float* __restrict__ route, float* __restrict__ out,
    unsigned int cols, unsigned int inner, const int* __restrict__ sorted, unsigned int topk,
    unsigned int weight_stride) {
    OPT_PDL_ENTRY();
    shared_prefill_mma_body<SPM_W_FP8, true, true, __nv_bfloat16, 6, 4, true>(
        a, nullptr, nullptr, table, items, out, cols, inner, Q4P_MOE_ROWS, inner, cols, 0ull,
        0ull, 0, 0, sorted, topk, route, weight_stride);
}


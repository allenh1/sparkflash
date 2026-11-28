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

// kernels/inkling_message_sm121.cu
//
// Inkling's loop message path: the four cross-pass reads that open a core layer's pass and
// the per-stream gates that mix them into the running residual.
//
// The producer's `Inkling2LoopMessageGate` (modeling_inkling2_loop.py:81-99) normalises the
// hidden row and the message row separately (an unweighted RMSNorm), concatenates them,
// projects the pair down to the gate rank, and mixes the two rows with
// `(1 - tanh(forget)) * source + tanh(gain) * message`. `Inkling2LoopMessageAttentionResidual`
// (:147-176) normalises the previous pass's summary rows and final hidden row the same way, then
// scores them with a learned query bank, scales by `hidden**-0.5`, softmaxes over the stack (not
// causal) and returns one weighted hidden row a query.
//
// ABI:
//
//   inkling_message_gate(source, destination, message, down, forget, gain, hidden, rank, epsilon,
//                        source_stride, source_shift, destination_stride, destination_shift,
//                        message_stride, message_shift)
//     source       [rows][source_stride] f32: row r's hidden slice starts at
//                  `r * source_stride + source_shift` (a sliding loop layer shares one
//                  attention_input row across its query, key and value gates, so the gate writes
//                  to a distinct destination rather than overwriting `source` in place)
//     destination  [rows][destination_stride] f32: row r's output slice starts at
//                  `r * destination_stride + destination_shift` (may alias `source` when safe)
//     message      [rows][message_stride] f32: row r's hidden slice starts at
//                  `r * message_stride + message_shift`
//     down         [rank][2 * hidden] f32: the concatenated pair to the bottleneck
//     forget       [hidden][rank] f32: the forget gate
//     gain         [hidden][rank] f32: the gain gate
//     rank         at most INKLING_MESSAGE_MAX_RANK (64), the shared-memory bottleneck's bound
//
//   inkling_cross_pass(summaries, prev_hidden, queries, kv, out, rows, start_block, count,
//                      summary_stride, core, hidden, epsilon)
//     summaries      [rows][summary_stride] f32: each row's block-summary stack, where block b of
//                    row r starts at `r * summary_stride + b * hidden` (for rows == 1,
//                    summary_stride may be 0)
//     prev_hidden    [rows][hidden] f32: the previous pass's final hidden row
//     queries        [2 * core][hidden] f32: the query and MLP stream bank
//     kv             [2 * core][hidden] f32: the key and value stream bank
//     out            [rows][core][4][hidden] f32: four messages a core layer a row
//     start_block    first summary block of the previous pass (`num_pre_loop_blocks + 1`)
//     count          summary rows the previous pass left (at most INKLING_MESSAGE_MAX_STATES - 1)
//     summary_stride floats between consecutive rows in `summaries`
//     core           the core layers the walk loops
//     The state stack is `summaries[row][start_block .. start_block + count]` and then
//     `prev_hidden[row]`, each unweighted-RMS-normalised. Stream s of layer l reads bank row
//     `2 * l + (s / 2)` of `queries` for s in {0, 3} and of `kv` for s in {1, 2}.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "opt_mma_bf16_sm121.cuh"
#include "opt_pdl_sm121.cuh"

#define INKLING_MESSAGE_BLOCK 256
#define INKLING_MESSAGE_MAX_RANK 64
#define INKLING_MESSAGE_MAX_STATES 72
#define INKLING_MSG_DOWN_BLOCK 128
#define INKLING_MSG_UP_BLOCK 256
#define INKLING_MSG_BATCH 8

__device__ __forceinline__ float inkling_message_block_sum(float value, float* scratch) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    if (lane == 0u) {
        scratch[warp] = value;
    }
    __syncthreads();
    float total = scratch[0];
#pragma unroll
    for (int index = 1; index < (int)(INKLING_MESSAGE_BLOCK / 32); ++index) {
        total += scratch[index];
    }
    return total;
}

// The two-tanh gate: one block a row, the bottleneck kept in shared memory.
extern "C" __global__ void __launch_bounds__(INKLING_MESSAGE_BLOCK) inkling_message_gate(
    const float* source,
    float* destination,
    const float* __restrict__ message,
    const float* __restrict__ down,
    const float* __restrict__ forget,
    const float* __restrict__ gain,
    int hidden,
    int rank,
    float epsilon,
    int source_stride,
    int source_shift,
    int destination_stride,
    int destination_shift,
    int message_stride,
    int message_shift) {
    OPT_PDL_ENTRY();
    __shared__ float scratch[INKLING_MESSAGE_BLOCK / 32];
    __shared__ float bottleneck[INKLING_MESSAGE_MAX_RANK];
    const int row = blockIdx.x;
    const float* value = source + (size_t)row * source_stride + source_shift;
    float* out = destination + (size_t)row * destination_stride + destination_shift;
    const float* msg = message + (size_t)row * message_stride + message_shift;
    if (rank > INKLING_MESSAGE_MAX_RANK) {
        return;
    }
    float squares_value = 0.0f;
    float squares_message = 0.0f;
    for (int index = threadIdx.x; index < hidden; index += INKLING_MESSAGE_BLOCK) {
        const float v = value[index];
        const float other = msg[index];
        squares_value += v * v;
        squares_message += other * other;
    }
    squares_value = inkling_message_block_sum(squares_value, scratch);
    __syncthreads();
    squares_message = inkling_message_block_sum(squares_message, scratch);
    __syncthreads();
    const float scale_value = rsqrtf(squares_value / (float)hidden + epsilon);
    const float scale_message = rsqrtf(squares_message / (float)hidden + epsilon);
    for (int index = threadIdx.x; index < rank; index += INKLING_MESSAGE_BLOCK) {
        const float* weights = down + (size_t)index * 2 * hidden;
        float total = 0.0f;
        for (int column = 0; column < hidden; ++column) {
            total += weights[column] * value[column] * scale_value
                + weights[hidden + column] * msg[column] * scale_message;
        }
        bottleneck[index] = total;
    }
    __syncthreads();
    for (int index = threadIdx.x; index < hidden; index += INKLING_MESSAGE_BLOCK) {
        const float* forget_row = forget + (size_t)index * rank;
        const float* gain_row = gain + (size_t)index * rank;
        float forget_value = 0.0f;
        float gain_value = 0.0f;
        for (int column = 0; column < rank; ++column) {
            forget_value += forget_row[column] * bottleneck[column];
            gain_value += gain_row[column] * bottleneck[column];
        }
        out[index] = (1.0f - tanhf(forget_value)) * value[index]
            + tanhf(gain_value) * msg[index];
    }
}

template <int KRANK>
__device__ __forceinline__ void inkling_message_gate_down_one(
    const float* __restrict__ x_base,
    const unsigned short* __restrict__ down_half,
    float* __restrict__ partials,
    int hidden,
    int rank,
    int rows,
    float epsilon,
    int x_stride,
    int x_shift,
    int stream_slot,
    int half) {
    const int k0 = (int)blockIdx.x * KRANK;
    if (k0 >= rank) {
        return;
    }
    const int kcount = (rank - k0 < KRANK) ? (rank - k0) : KRANK;
    const int r0 = (int)blockIdx.z * INKLING_MSG_BATCH;
    const int rcount = (rows - r0 < INKLING_MSG_BATCH) ? (rows - r0) : INKLING_MSG_BATCH;
    if (rcount <= 0) {
        return;
    }
    float dot[KRANK][INKLING_MSG_BATCH];
    float sq[INKLING_MSG_BATCH];
#pragma unroll
    for (int t = 0; t < INKLING_MSG_BATCH; ++t) {
        sq[t] = 0.0f;
#pragma unroll
        for (int kr = 0; kr < KRANK; ++kr) {
            dot[kr][t] = 0.0f;
        }
    }

    const float* __restrict__ x0 = x_base + (size_t)r0 * x_stride + x_shift;
    const bool vec_ok = ((hidden & 7) == 0) && ((x_stride & 3) == 0) && ((x_shift & 3) == 0)
        && (((uintptr_t)down_half & 15u) == 0u) && (((uintptr_t)x0 & 15u) == 0u);
    if (vec_ok) {
        for (int c = (int)threadIdx.x * 8; c < hidden; c += INKLING_MSG_DOWN_BLOCK * 8) {
            uint4 wraw[KRANK];
#pragma unroll
            for (int kr = 0; kr < KRANK; ++kr) {
                if (kr < kcount) {
                    const unsigned short* __restrict__ wk =
                        down_half + (size_t)(k0 + kr) * 2 * (size_t)hidden;
                    wraw[kr] = __ldg(reinterpret_cast<const uint4*>(wk + c));
                }
            }
            if (rcount == 1) {
                const float4 a = *reinterpret_cast<const float4*>(x0 + c);
                const float4 b = *reinterpret_cast<const float4*>(x0 + c + 4);
                sq[0] += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w
                    + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
#pragma unroll
                for (int kr = 0; kr < KRANK; ++kr) {
                    if (kr < kcount) {
                        const float2 w01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wraw[kr].x));
                        const float2 w23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wraw[kr].y));
                        const float2 w45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wraw[kr].z));
                        const float2 w67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wraw[kr].w));
                        dot[kr][0] += w01.x * a.x + w01.y * a.y + w23.x * a.z + w23.y * a.w
                            + w45.x * b.x + w45.y * b.y + w67.x * b.z + w67.y * b.w;
                    }
                }
            } else {
                float4 ax[INKLING_MSG_BATCH];
                float4 bx[INKLING_MSG_BATCH];
#pragma unroll
                for (int t = 0; t < INKLING_MSG_BATCH; ++t) {
                    if (t < rcount) {
                        const float* __restrict__ xt = x0 + (size_t)t * x_stride + c;
                        ax[t] = *reinterpret_cast<const float4*>(xt);
                        bx[t] = *reinterpret_cast<const float4*>(xt + 4);
                        sq[t] += ax[t].x * ax[t].x + ax[t].y * ax[t].y + ax[t].z * ax[t].z + ax[t].w * ax[t].w
                            + bx[t].x * bx[t].x + bx[t].y * bx[t].y + bx[t].z * bx[t].z + bx[t].w * bx[t].w;
                    }
                }
#pragma unroll
                for (int kr = 0; kr < KRANK; ++kr) {
                    if (kr < kcount) {
                        const float2 w01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wraw[kr].x));
                        const float2 w23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wraw[kr].y));
                        const float2 w45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wraw[kr].z));
                        const float2 w67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wraw[kr].w));
#pragma unroll
                        for (int t = 0; t < INKLING_MSG_BATCH; ++t) {
                            if (t < rcount) {
                                dot[kr][t] += w01.x * ax[t].x + w01.y * ax[t].y + w23.x * ax[t].z + w23.y * ax[t].w
                                    + w45.x * bx[t].x + w45.y * bx[t].y + w67.x * bx[t].z + w67.y * bx[t].w;
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (int c = (int)threadIdx.x; c < hidden; c += INKLING_MSG_DOWN_BLOCK) {
            for (int t = 0; t < rcount; ++t) {
                const float xv = x0[(size_t)t * x_stride + c];
                sq[t] += xv * xv;
                for (int kr = 0; kr < kcount; ++kr) {
                    const unsigned short* __restrict__ wk =
                        down_half + (size_t)(k0 + kr) * 2 * (size_t)hidden;
                    const float wf = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(wk + c));
                    dot[kr][t] += wf * xv;
                }
            }
        }
    }

    __shared__ float smem_sq[INKLING_MSG_DOWN_BLOCK / 32][INKLING_MSG_BATCH];
    __shared__ float smem_dot[INKLING_MSG_DOWN_BLOCK / 32][KRANK][INKLING_MSG_BATCH];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
#pragma unroll
    for (int t = 0; t < INKLING_MSG_BATCH; ++t) {
        if (t < rcount) {
            float s = sq[t];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                s += __shfl_xor_sync(0xffffffffu, s, offset);
            }
            if (lane == 0u) {
                smem_sq[warp][t] = s;
            }
#pragma unroll
            for (int kr = 0; kr < KRANK; ++kr) {
                if (kr < kcount) {
                    float d = dot[kr][t];
#pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        d += __shfl_xor_sync(0xffffffffu, d, offset);
                    }
                    if (lane == 0u) {
                        smem_dot[warp][kr][t] = d;
                    }
                }
            }
        }
    }
    __syncthreads();
    const int idx = (int)threadIdx.x;
    if (idx < rcount * kcount) {
        const int t = idx / kcount;
        const int kr = idx - t * kcount;
        float total_s = 0.0f;
        float total_d = 0.0f;
#pragma unroll
        for (int w_idx = 0; w_idx < INKLING_MSG_DOWN_BLOCK / 32; ++w_idx) {
            total_s += smem_sq[w_idx][t];
            total_d += smem_dot[w_idx][kr][t];
        }
        const float scale = rsqrtf(total_s / (float)hidden + epsilon);
        const size_t dst =
            (((size_t)stream_slot * rows + (r0 + t)) * 2 + half) * (size_t)rank + (k0 + kr);
        partials[dst] = total_d * scale;
    }
}

// Multi-SM BF16 down-projection for 1 message-gate stream: grid = (ceil(rank/2), 2, ceil(rows/8)).
extern "C" __global__ void __launch_bounds__(INKLING_MSG_DOWN_BLOCK) inkling_message_gate_down_bf16(
    const float* __restrict__ source,
    const float* __restrict__ message,
    const unsigned short* __restrict__ down,
    float* __restrict__ partials,
    int hidden,
    int rank,
    int rows,
    float epsilon,
    int source_stride,
    int source_shift,
    int message_stride,
    int message_shift) {
    OPT_PDL_ENTRY();
    const int half = (int)blockIdx.y;
    const float* __restrict__ x_base = half == 0 ? source : message;
    const int x_stride = half == 0 ? source_stride : message_stride;
    const int x_shift = half == 0 ? source_shift : message_shift;
    const unsigned short* __restrict__ down_half = down + (half ? hidden : 0);
    inkling_message_gate_down_one<2>(
        x_base, down_half, partials, hidden, rank, rows, epsilon, x_stride, x_shift, 0, half);
}

// Multi-SM BF16 down-projection batched across all 3 attention streams (Q/K/V):
// grid = (ceil(rank/4), 6, ceil(rows/8)), where blockIdx.y >> 1 is the stream (0..2) and & 1 is half.
extern "C" __global__ void __launch_bounds__(INKLING_MSG_DOWN_BLOCK) inkling_message_gate_down3_bf16(
    const float* __restrict__ source,
    const float* __restrict__ message,
    const unsigned short* __restrict__ down0,
    const unsigned short* __restrict__ down1,
    const unsigned short* __restrict__ down2,
    float* __restrict__ partials,
    int hidden,
    int rank,
    int rows,
    float epsilon,
    int source_stride,
    int source_shift0,
    int source_shift1,
    int source_shift2,
    int message_stride,
    int message_shift0,
    int message_shift1,
    int message_shift2) {
    OPT_PDL_ENTRY();
    const int stream = (int)blockIdx.y >> 1;
    const int half = (int)blockIdx.y & 1;
    const unsigned short* __restrict__ down = stream == 0 ? down0 : (stream == 1 ? down1 : down2);
    const int src_shift = stream == 0 ? source_shift0 : (stream == 1 ? source_shift1 : source_shift2);
    const int msg_shift = stream == 0 ? message_shift0 : (stream == 1 ? message_shift1 : message_shift2);
    const float* __restrict__ x_base = half == 0 ? source : message;
    const int x_stride = half == 0 ? source_stride : message_stride;
    const int x_shift = half == 0 ? src_shift : msg_shift;
    const unsigned short* __restrict__ down_half = down + (half ? hidden : 0);
    inkling_message_gate_down_one<4>(
        x_base, down_half, partials, hidden, rank, rows, epsilon, x_stride, x_shift, stream, half);
}

__device__ __forceinline__ void inkling_message_gate_up_one(
    const float* source,
    float* destination,
    const float* __restrict__ message,
    const unsigned short* __restrict__ forget,
    const unsigned short* __restrict__ gain,
    const float* __restrict__ partials,
    int hidden,
    int rank,
    int rows,
    int source_stride,
    int source_shift,
    int destination_stride,
    int destination_shift,
    int message_stride,
    int message_shift,
    int stream_slot) {
    // Overlap constant weight loads (forget/gain) with the preceding down-projection tail via PDL.
    OPT_PDL_LAUNCH();
    const int r0 = (int)blockIdx.z * INKLING_MSG_BATCH;
    const int rcount = (rows - r0 < INKLING_MSG_BATCH) ? (rows - r0) : INKLING_MSG_BATCH;
    if (rcount <= 0 || rank > INKLING_MESSAGE_MAX_RANK) {
        OPT_PDL_WAIT();
        return;
    }
    const int local_row = (int)(threadIdx.x >> 3);
    const int sublane = (int)(threadIdx.x & 7u);
    const int h = (int)blockIdx.x * 32 + local_row;
    const bool fast_rank = (rank == 64);
    uint4 fraw = {0u, 0u, 0u, 0u};
    uint4 graw = {0u, 0u, 0u, 0u};
    if (fast_rank && h < hidden) {
        fraw = *reinterpret_cast<const uint4*>(forget + (size_t)h * 64 + sublane * 8);
        graw = *reinterpret_cast<const uint4*>(gain + (size_t)h * 64 + sublane * 8);
    }
    OPT_PDL_WAIT();

    __shared__ float bn[INKLING_MSG_BATCH][INKLING_MESSAGE_MAX_RANK];
    const int total_bn = rcount * rank;
    for (int idx = (int)threadIdx.x; idx < total_bn; idx += INKLING_MSG_UP_BLOCK) {
        const int t = idx / rank;
        const int k = idx - t * rank;
        const size_t base = ((size_t)stream_slot * rows + (r0 + t)) * 2 * (size_t)rank;
        bn[t][k] = partials[base + k] + partials[base + rank + k];
    }
    __syncthreads();

    float fv[INKLING_MSG_BATCH];
    float gv[INKLING_MSG_BATCH];
#pragma unroll
    for (int t = 0; t < INKLING_MSG_BATCH; ++t) {
        fv[t] = 0.0f;
        gv[t] = 0.0f;
    }

    if (h < hidden) {
        if (fast_rank) {
            const float2 f01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&fraw.x));
            const float2 f23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&fraw.y));
            const float2 f45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&fraw.z));
            const float2 f67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&fraw.w));
            const float2 g01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.x));
            const float2 g23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.y));
            const float2 g45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.z));
            const float2 g67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.w));
            const int base_k = sublane * 8;
            if (rcount == 1) {
                const float4 b0 = *reinterpret_cast<const float4*>(&bn[0][base_k]);
                const float4 b1 = *reinterpret_cast<const float4*>(&bn[0][base_k + 4]);
                fv[0] = f01.x * b0.x + f01.y * b0.y + f23.x * b0.z + f23.y * b0.w
                    + f45.x * b1.x + f45.y * b1.y + f67.x * b1.z + f67.y * b1.w;
                gv[0] = g01.x * b0.x + g01.y * b0.y + g23.x * b0.z + g23.y * b0.w
                    + g45.x * b1.x + g45.y * b1.y + g67.x * b1.z + g67.y * b1.w;
            } else {
#pragma unroll
                for (int t = 0; t < INKLING_MSG_BATCH; ++t) {
                    if (t < rcount) {
                        const float4 b0 = *reinterpret_cast<const float4*>(&bn[t][base_k]);
                        const float4 b1 = *reinterpret_cast<const float4*>(&bn[t][base_k + 4]);
                        fv[t] = f01.x * b0.x + f01.y * b0.y + f23.x * b0.z + f23.y * b0.w
                            + f45.x * b1.x + f45.y * b1.y + f67.x * b1.z + f67.y * b1.w;
                        gv[t] = g01.x * b0.x + g01.y * b0.y + g23.x * b0.z + g23.y * b0.w
                            + g45.x * b1.x + g45.y * b1.y + g67.x * b1.z + g67.y * b1.w;
                    }
                }
            }
        } else {
            const unsigned short* __restrict__ frow = forget + (size_t)h * rank;
            const unsigned short* __restrict__ grow = gain + (size_t)h * rank;
            for (int k = sublane; k < rank; k += 8) {
                const float wf = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(frow + k));
                const float wg = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(grow + k));
                for (int t = 0; t < rcount; ++t) {
                    const float b = bn[t][k];
                    fv[t] += wf * b;
                    gv[t] += wg * b;
                }
            }
        }
    }

    for (int t = 0; t < rcount; ++t) {
        float f = fv[t];
        float g = gv[t];
        f += __shfl_xor_sync(0xffffffffu, f, 4);
        g += __shfl_xor_sync(0xffffffffu, g, 4);
        f += __shfl_xor_sync(0xffffffffu, f, 2);
        g += __shfl_xor_sync(0xffffffffu, g, 2);
        f += __shfl_xor_sync(0xffffffffu, f, 1);
        g += __shfl_xor_sync(0xffffffffu, g, 1);
        if (sublane == 0 && h < hidden) {
            const size_t row = (size_t)(r0 + t);
            const float v = source[row * source_stride + source_shift + h];
            const float m = message[row * message_stride + message_shift + h];
            destination[row * destination_stride + destination_shift + h] =
                (1.0f - tanhf(f)) * v + tanhf(g) * m;
        }
    }
}

// Multi-SM BF16 up-projection + two-tanh mix for 1 stream: grid = (ceil(hidden/32), 1, ceil(rows/8)).
extern "C" __global__ void __launch_bounds__(INKLING_MSG_UP_BLOCK) inkling_message_gate_up_bf16(
    const float* source,
    float* destination,
    const float* __restrict__ message,
    const unsigned short* __restrict__ forget,
    const unsigned short* __restrict__ gain,
    const float* __restrict__ partials,
    int hidden,
    int rank,
    int rows,
    int source_stride,
    int source_shift,
    int destination_stride,
    int destination_shift,
    int message_stride,
    int message_shift) {
    inkling_message_gate_up_one(
        source, destination, message, forget, gain, partials, hidden, rank, rows,
        source_stride, source_shift, destination_stride, destination_shift,
        message_stride, message_shift, 0);
}

// Multi-SM BF16 up-projection + two-tanh mix batched across 3 attention streams (Q/K/V):
// grid = (ceil(hidden/32), 3, ceil(rows/8)).
extern "C" __global__ void __launch_bounds__(INKLING_MSG_UP_BLOCK) inkling_message_gate_up3_bf16(
    const float* source,
    float* dst0,
    float* dst1,
    float* dst2,
    const float* __restrict__ message,
    const unsigned short* __restrict__ forget0,
    const unsigned short* __restrict__ forget1,
    const unsigned short* __restrict__ forget2,
    const unsigned short* __restrict__ gain0,
    const unsigned short* __restrict__ gain1,
    const unsigned short* __restrict__ gain2,
    const float* __restrict__ partials,
    int hidden,
    int rank,
    int rows,
    int source_stride,
    int source_shift0,
    int source_shift1,
    int source_shift2,
    int destination_stride,
    int destination_shift0,
    int destination_shift1,
    int destination_shift2,
    int message_stride,
    int message_shift0,
    int message_shift1,
    int message_shift2) {
    const int stream = (int)blockIdx.y;
    float* dst = stream == 0 ? dst0 : (stream == 1 ? dst1 : dst2);
    const unsigned short* __restrict__ forget =
        stream == 0 ? forget0 : (stream == 1 ? forget1 : forget2);
    const unsigned short* __restrict__ gain =
        stream == 0 ? gain0 : (stream == 1 ? gain1 : gain2);
    const int src_shift =
        stream == 0 ? source_shift0 : (stream == 1 ? source_shift1 : source_shift2);
    const int dst_shift =
        stream == 0 ? destination_shift0 : (stream == 1 ? destination_shift1 : destination_shift2);
    const int msg_shift =
        stream == 0 ? message_shift0 : (stream == 1 ? message_shift1 : message_shift2);
    inkling_message_gate_up_one(
        source, dst, message, forget, gain, partials, hidden, rank, rows,
        source_stride, src_shift, destination_stride, dst_shift,
        message_stride, msg_shift, stream);
}

// ---------------------------------------------------------------------------------------------
// Tensor-core prefill message gate (rows >= 32, hidden % 32 == 0, rank <= 64, rank % 8 == 0):
// BF16 head + remainder MMA (16 significant bits) accumulated in f32.
// ---------------------------------------------------------------------------------------------

#define INKLING_MSG_MMA_THREADS 256u
#define INKLING_MSG_MMA_BM 64u
#define INKLING_MSG_MMA_LD32 40u
#define INKLING_MSG_MMA_LD64 72u

__device__ __forceinline__ void inkling_msg_split4(
    float4 v, uint2& hi, uint2& lo, float& sq) {
    sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    const __nv_bfloat16 h0 = __float2bfloat16_rn(v.x);
    const __nv_bfloat16 h1 = __float2bfloat16_rn(v.y);
    const __nv_bfloat16 h2 = __float2bfloat16_rn(v.z);
    const __nv_bfloat16 h3 = __float2bfloat16_rn(v.w);
    const __nv_bfloat16 l0 = __float2bfloat16_rn(v.x - __bfloat162float(h0));
    const __nv_bfloat16 l1 = __float2bfloat16_rn(v.y - __bfloat162float(h1));
    const __nv_bfloat16 l2 = __float2bfloat16_rn(v.z - __bfloat162float(h2));
    const __nv_bfloat16 l3 = __float2bfloat16_rn(v.w - __bfloat162float(h3));
    hi = make_uint2(ob_pack(h0, h1), ob_pack(h2, h3));
    lo = make_uint2(ob_pack(l0, l1), ob_pack(l2, l3));
}

__device__ __forceinline__ void inkling_msg_down_half_mma(
    const float* __restrict__ x_base,
    const __nv_bfloat16* __restrict__ w_half,
    int hidden,
    int rank,
    int r0,
    int rows,
    float epsilon,
    int x_stride,
    int x_shift,
    __nv_bfloat16* s_ah,
    __nv_bfloat16* s_al,
    __nv_bfloat16* s_w,
    float* s_scale,
    float acc[2][2][4]) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t wm = warp & 1u;
    const uint32_t wn = warp >> 1;
    const uint32_t a_row = tid >> 2;
    const uint32_t a_chunk = tid & 3u;
    const uint32_t w_row = tid >> 2;
    const uint32_t w_chunk = tid & 3u;
    const bool a_valid = (r0 + (int)a_row) < rows;
    const bool w_valid = (int)w_row < rank;
    const float* __restrict__ x_ptr =
        a_valid ? x_base + (size_t)(r0 + (int)a_row) * x_stride + x_shift + a_chunk * 8u : nullptr;
    const __nv_bfloat16* __restrict__ w_ptr =
        w_valid ? w_half + (size_t)w_row * (2u * (size_t)hidden) + w_chunk * 8u : nullptr;

#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[mt][nt][e] = 0.0f;
            }
        }
    }

    const uint32_t k_blocks = (uint32_t)hidden >> 5;
    float4 ra0, ra1;
    uint4 rw;
    float sq = 0.0f;
    auto load = [&](uint32_t kb) {
        if (a_valid) {
            ra0 = __ldg(reinterpret_cast<const float4*>(x_ptr + kb * 32u));
            ra1 = __ldg(reinterpret_cast<const float4*>(x_ptr + kb * 32u + 4u));
        } else {
            ra0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            ra1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        if (w_valid) {
            rw = __ldg(reinterpret_cast<const uint4*>(w_ptr + kb * 32u));
        } else {
            rw = make_uint4(0u, 0u, 0u, 0u);
        }
    };
    auto store = [&]() {
        uint2 hi0, lo0, hi1, lo1;
        inkling_msg_split4(ra0, hi0, lo0, sq);
        inkling_msg_split4(ra1, hi1, lo1, sq);
        *reinterpret_cast<uint4*>(&s_ah[a_row * INKLING_MSG_MMA_LD32 + a_chunk * 8u]) =
            make_uint4(hi0.x, hi0.y, hi1.x, hi1.y);
        *reinterpret_cast<uint4*>(&s_al[a_row * INKLING_MSG_MMA_LD32 + a_chunk * 8u]) =
            make_uint4(lo0.x, lo0.y, lo1.x, lo1.y);
        *reinterpret_cast<uint4*>(&s_w[w_row * INKLING_MSG_MMA_LD32 + w_chunk * 8u]) = rw;
    };

    load(0u);
    for (uint32_t kb = 0u; kb < k_blocks; ++kb) {
        store();
        __syncthreads();
        if (kb + 1u < k_blocks) {
            load(kb + 1u);
        }
#pragma unroll
        for (uint32_t ks = 0u; ks < 2u; ++ks) {
            const uint32_t k0 = ks * 16u;
            uint32_t b0, b1, b2, b3;
            const uint32_t bn = wn * 16u + (lane & 7u) + 8u * (lane >> 4);
            const uint32_t bk = k0 + 8u * ((lane >> 3) & 1u);
            om_ldsm4(om_smem(&s_w[bn * INKLING_MSG_MMA_LD32 + bk]), b0, b1, b2, b3);
#pragma unroll
            for (uint32_t mt = 0u; mt < 2u; ++mt) {
                const uint32_t row = wm * 32u + mt * 16u + (lane & 15u);
                const uint32_t ak = k0 + 8u * (lane >> 4);
                uint32_t af[4];
                om_ldsm4(om_smem(&s_ah[row * INKLING_MSG_MMA_LD32 + ak]),
                         af[0], af[1], af[2], af[3]);
                ob_mma(acc[mt][0], af, b0, b1);
                ob_mma(acc[mt][1], af, b2, b3);
                om_ldsm4(om_smem(&s_al[row * INKLING_MSG_MMA_LD32 + ak]),
                         af[0], af[1], af[2], af[3]);
                ob_mma(acc[mt][0], af, b0, b1);
                ob_mma(acc[mt][1], af, b2, b3);
            }
        }
        __syncthreads();
    }
    sq += __shfl_xor_sync(0xffffffffu, sq, 1);
    sq += __shfl_xor_sync(0xffffffffu, sq, 2);
    if (a_chunk == 0u) {
        s_scale[a_row] = rsqrtf(sq / (float)hidden + epsilon);
    }
    __syncthreads();
}

__device__ __forceinline__ void inkling_message_gate_down_mma_one(
    const float* __restrict__ source,
    const float* __restrict__ message,
    const unsigned short* __restrict__ down,
    float* __restrict__ bottleneck,
    int hidden,
    int rank,
    int rows,
    float epsilon,
    int source_stride,
    int source_shift,
    int message_stride,
    int message_shift,
    int stream_slot) {
    const int r0 = (int)(blockIdx.y * INKLING_MSG_MMA_BM);
    if (r0 >= rows || rank <= 0 || rank > INKLING_MESSAGE_MAX_RANK) {
        return;
    }
    __shared__ __align__(16) __nv_bfloat16 s_ah[INKLING_MSG_MMA_BM * INKLING_MSG_MMA_LD32];
    __shared__ __align__(16) __nv_bfloat16 s_al[INKLING_MSG_MMA_BM * INKLING_MSG_MMA_LD32];
    __shared__ __align__(16) __nv_bfloat16 s_w[INKLING_MESSAGE_MAX_RANK * INKLING_MSG_MMA_LD32];
    __shared__ float s_scale[INKLING_MSG_MMA_BM];

    const __nv_bfloat16* __restrict__ down_bf16 = reinterpret_cast<const __nv_bfloat16*>(down);
    float acc_v[2][2][4];
    inkling_msg_down_half_mma(source, down_bf16, hidden, rank, r0, rows, epsilon,
                              source_stride, source_shift, s_ah, s_al, s_w, s_scale, acc_v);

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t wm = warp & 1u;
    const uint32_t wn = warp >> 1;
    const uint32_t g = lane >> 2;
    const uint32_t q = lane & 3u;

    float out_acc[2][2][4];
#pragma unroll
    for (uint32_t mt = 0u; mt < 2u; ++mt) {
        const float sv0 = s_scale[wm * 32u + mt * 16u + g];
        const float sv1 = s_scale[wm * 32u + mt * 16u + g + 8u];
#pragma unroll
        for (uint32_t nt = 0u; nt < 2u; ++nt) {
            out_acc[mt][nt][0] = acc_v[mt][nt][0] * sv0;
            out_acc[mt][nt][1] = acc_v[mt][nt][1] * sv0;
            out_acc[mt][nt][2] = acc_v[mt][nt][2] * sv1;
            out_acc[mt][nt][3] = acc_v[mt][nt][3] * sv1;
        }
    }
    __syncthreads();

    float acc_m[2][2][4];
    inkling_msg_down_half_mma(message, down_bf16 + hidden, hidden, rank, r0, rows, epsilon,
                              message_stride, message_shift, s_ah, s_al, s_w, s_scale, acc_m);

#pragma unroll
    for (uint32_t mt = 0u; mt < 2u; ++mt) {
        const float sm0 = s_scale[wm * 32u + mt * 16u + g];
        const float sm1 = s_scale[wm * 32u + mt * 16u + g + 8u];
#pragma unroll
        for (uint32_t half = 0u; half < 2u; ++half) {
            const int row = r0 + (int)(wm * 32u + mt * 16u + g + 8u * half);
            if (row >= rows) {
                continue;
            }
            const float sm = half == 0u ? sm0 : sm1;
            float* __restrict__ dst =
                bottleneck + ((size_t)stream_slot * rows + (size_t)row) * (size_t)rank;
#pragma unroll
            for (uint32_t nt = 0u; nt < 2u; ++nt) {
                const int col = (int)(wn * 16u + nt * 8u + 2u * q);
                if (col + 1 < rank) {
                    const float v0 = out_acc[mt][nt][2u * half] + acc_m[mt][nt][2u * half] * sm;
                    const float v1 =
                        out_acc[mt][nt][2u * half + 1u] + acc_m[mt][nt][2u * half + 1u] * sm;
                    *reinterpret_cast<float2*>(dst + col) = make_float2(v0, v1);
                }
            }
        }
    }
}

// Tensor-core BF16 down-projection for 1 stream: grid = (1, ceil(rows / 64), 1), 256 threads.
extern "C" __global__ void __launch_bounds__(INKLING_MSG_MMA_THREADS, 2)
inkling_message_gate_down_mma(
    const float* __restrict__ source,
    const float* __restrict__ message,
    const unsigned short* __restrict__ down,
    float* __restrict__ bottleneck,
    int hidden,
    int rank,
    int rows,
    float epsilon,
    int source_stride,
    int source_shift,
    int message_stride,
    int message_shift) {
    OPT_PDL_ENTRY();
    inkling_message_gate_down_mma_one(
        source, message, down, bottleneck, hidden, rank, rows, epsilon,
        source_stride, source_shift, message_stride, message_shift, 0);
}

// Tensor-core BF16 down-projection for 3 attention streams: grid = (3, ceil(rows / 64), 1).
extern "C" __global__ void __launch_bounds__(INKLING_MSG_MMA_THREADS, 2)
inkling_message_gate_down3_mma(
    const float* __restrict__ source,
    const float* __restrict__ message,
    const unsigned short* __restrict__ down0,
    const unsigned short* __restrict__ down1,
    const unsigned short* __restrict__ down2,
    float* __restrict__ bottleneck,
    int hidden,
    int rank,
    int rows,
    float epsilon,
    int source_stride,
    int source_shift0,
    int source_shift1,
    int source_shift2,
    int message_stride,
    int message_shift0,
    int message_shift1,
    int message_shift2) {
    OPT_PDL_ENTRY();
    const int stream = (int)blockIdx.x;
    const unsigned short* __restrict__ down = stream == 0 ? down0 : (stream == 1 ? down1 : down2);
    const int src_shift = stream == 0 ? source_shift0 : (stream == 1 ? source_shift1 : source_shift2);
    const int msg_shift = stream == 0 ? message_shift0 : (stream == 1 ? message_shift1 : message_shift2);
    inkling_message_gate_down_mma_one(
        source, message, down, bottleneck, hidden, rank, rows, epsilon,
        source_stride, src_shift, message_stride, msg_shift, stream);
}

__device__ __forceinline__ void inkling_message_gate_up_mma_one(
    const float* source,
    float* destination,
    const float* __restrict__ message,
    const unsigned short* __restrict__ forget,
    const unsigned short* __restrict__ gain,
    const float* __restrict__ bottleneck,
    int hidden,
    int rank,
    int rows,
    int source_stride,
    int source_shift,
    int destination_stride,
    int destination_shift,
    int message_stride,
    int message_shift,
    int stream_slot) {
    const int n0 = (int)(blockIdx.x * 128u);
    const int r0 = (int)(blockIdx.z * INKLING_MSG_MMA_BM);
    if (r0 >= rows || n0 >= hidden || rank <= 0 || rank > INKLING_MESSAGE_MAX_RANK) {
        return;
    }
    __shared__ __align__(16) __nv_bfloat16 s_ah[INKLING_MSG_MMA_BM * INKLING_MSG_MMA_LD64];
    __shared__ __align__(16) __nv_bfloat16 s_al[INKLING_MSG_MMA_BM * INKLING_MSG_MMA_LD64];
    __shared__ __align__(16) __nv_bfloat16 s_w[128u * INKLING_MSG_MMA_LD64];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t wm = warp & 1u;
    const uint32_t wn = warp >> 1;

    // Stage bottleneck [64][rank] f32 -> BF16 head + remainder in SMEM (rank <= 64, padded with 0).
#pragma unroll
    for (uint32_t i = 0u; i < 2u; ++i) {
        const uint32_t idx = tid + i * INKLING_MSG_MMA_THREADS;
        const uint32_t row = idx >> 3;
        const uint32_t chunk = idx & 7u;  // 8 floats per chunk, 8 chunks = 64
        const int gr = r0 + (int)row;
        const int gk = (int)(chunk * 8u);
        float4 a0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 a1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (gr < rows && gk < rank) {
            const float* __restrict__ bp =
                bottleneck + ((size_t)stream_slot * rows + (size_t)gr) * (size_t)rank + gk;
            if (gk + 8 <= rank) {
                a0 = __ldg(reinterpret_cast<const float4*>(bp));
                a1 = __ldg(reinterpret_cast<const float4*>(bp + 4));
            } else {
                float tmp[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                for (int e = 0; e < 8 && gk + e < rank; ++e) {
                    tmp[e] = bp[e];
                }
                a0 = make_float4(tmp[0], tmp[1], tmp[2], tmp[3]);
                a1 = make_float4(tmp[4], tmp[5], tmp[6], tmp[7]);
            }
        }
        float dummy_sq = 0.0f;
        uint2 hi0, lo0, hi1, lo1;
        inkling_msg_split4(a0, hi0, lo0, dummy_sq);
        inkling_msg_split4(a1, hi1, lo1, dummy_sq);
        *reinterpret_cast<uint4*>(&s_ah[row * INKLING_MSG_MMA_LD64 + chunk * 8u]) =
            make_uint4(hi0.x, hi0.y, hi1.x, hi1.y);
        *reinterpret_cast<uint4*>(&s_al[row * INKLING_MSG_MMA_LD64 + chunk * 8u]) =
            make_uint4(lo0.x, lo0.y, lo1.x, lo1.y);
    }

    const uint32_t k_steps = ((uint32_t)rank + 15u) >> 4;
    auto run_w = [&](const unsigned short* __restrict__ w_raw, float acc[2][4][4]) {
        const __nv_bfloat16* __restrict__ w_bf16 = reinterpret_cast<const __nv_bfloat16*>(w_raw);
#pragma unroll
        for (uint32_t i = 0u; i < 4u; ++i) {
            const uint32_t idx = tid + i * INKLING_MSG_MMA_THREADS;
            const uint32_t row = idx >> 3;
            const uint32_t chunk = idx & 7u;
            const int gh = n0 + (int)row;
            const int gk = (int)(chunk * 8u);
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (gh < hidden && gk < rank) {
                if (gk + 8 <= rank) {
                    val = __ldg(reinterpret_cast<const uint4*>(
                        w_bf16 + (size_t)gh * (size_t)rank + gk));
                } else {
                    unsigned short tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    for (int e = 0; e < 8 && gk + e < rank; ++e) {
                        tmp[e] = w_raw[(size_t)gh * (size_t)rank + gk + e];
                    }
                    val = *reinterpret_cast<const uint4*>(tmp);
                }
            }
            *reinterpret_cast<uint4*>(&s_w[row * INKLING_MSG_MMA_LD64 + chunk * 8u]) = val;
        }
        __syncthreads();
#pragma unroll
        for (uint32_t mt = 0u; mt < 2u; ++mt) {
#pragma unroll
            for (uint32_t nt = 0u; nt < 4u; ++nt) {
#pragma unroll
                for (uint32_t e = 0u; e < 4u; ++e) {
                    acc[mt][nt][e] = 0.0f;
                }
            }
        }
        for (uint32_t ks = 0u; ks < k_steps; ++ks) {
            const uint32_t k0 = ks * 16u;
            uint32_t b[4][2];
#pragma unroll
            for (uint32_t p = 0u; p < 2u; ++p) {
                const uint32_t bn = wn * 32u + p * 16u + (lane & 7u) + 8u * (lane >> 4);
                const uint32_t bk = k0 + 8u * ((lane >> 3) & 1u);
                om_ldsm4(om_smem(&s_w[bn * INKLING_MSG_MMA_LD64 + bk]),
                         b[2u * p][0], b[2u * p][1], b[2u * p + 1u][0], b[2u * p + 1u][1]);
            }
#pragma unroll
            for (uint32_t mt = 0u; mt < 2u; ++mt) {
                const uint32_t row = wm * 32u + mt * 16u + (lane & 15u);
                const uint32_t ak = k0 + 8u * (lane >> 4);
                uint32_t af[4];
                om_ldsm4(om_smem(&s_ah[row * INKLING_MSG_MMA_LD64 + ak]),
                         af[0], af[1], af[2], af[3]);
#pragma unroll
                for (uint32_t nt = 0u; nt < 4u; ++nt) {
                    ob_mma(acc[mt][nt], af, b[nt][0], b[nt][1]);
                }
                om_ldsm4(om_smem(&s_al[row * INKLING_MSG_MMA_LD64 + ak]),
                         af[0], af[1], af[2], af[3]);
#pragma unroll
                for (uint32_t nt = 0u; nt < 4u; ++nt) {
                    ob_mma(acc[mt][nt], af, b[nt][0], b[nt][1]);
                }
            }
        }
    };

    float acc_f[2][4][4];
    float acc_g[2][4][4];
    run_w(forget, acc_f);
    __syncthreads();
    run_w(gain, acc_g);

    const uint32_t g = lane >> 2;
    const uint32_t q = lane & 3u;
#pragma unroll
    for (uint32_t mt = 0u; mt < 2u; ++mt) {
#pragma unroll
        for (uint32_t half = 0u; half < 2u; ++half) {
            const int row = r0 + (int)(wm * 32u + mt * 16u + g + 8u * half);
            if (row >= rows) {
                continue;
            }
            const float* src_row = source + (size_t)row * source_stride + source_shift;
            const float* msg_row = message + (size_t)row * message_stride + message_shift;
            float* dst_row = destination + (size_t)row * destination_stride + destination_shift;
#pragma unroll
            for (uint32_t nt = 0u; nt < 4u; ++nt) {
                const int col = n0 + (int)(wn * 32u + nt * 8u + 2u * q);
                if (col + 1 < hidden) {
                    const float2 v = *reinterpret_cast<const float2*>(src_row + col);
                    const float2 m = *reinterpret_cast<const float2*>(msg_row + col);
                    const float f0 = acc_f[mt][nt][2u * half];
                    const float f1 = acc_f[mt][nt][2u * half + 1u];
                    const float g0 = acc_g[mt][nt][2u * half];
                    const float g1 = acc_g[mt][nt][2u * half + 1u];
                    *reinterpret_cast<float2*>(dst_row + col) = make_float2(
                        (1.0f - tanhf(f0)) * v.x + tanhf(g0) * m.x,
                        (1.0f - tanhf(f1)) * v.y + tanhf(g1) * m.y);
                }
            }
        }
    }
}

// Tensor-core BF16 up-projection + two-tanh mix for 1 stream:
// grid = (ceil(hidden / 128), 1, ceil(rows / 64)), 256 threads.
extern "C" __global__ void __launch_bounds__(INKLING_MSG_MMA_THREADS, 2)
inkling_message_gate_up_mma(
    const float* source,
    float* destination,
    const float* __restrict__ message,
    const unsigned short* __restrict__ forget,
    const unsigned short* __restrict__ gain,
    const float* __restrict__ bottleneck,
    int hidden,
    int rank,
    int rows,
    int source_stride,
    int source_shift,
    int destination_stride,
    int destination_shift,
    int message_stride,
    int message_shift) {
    OPT_PDL_ENTRY();
    inkling_message_gate_up_mma_one(
        source, destination, message, forget, gain, bottleneck, hidden, rank, rows,
        source_stride, source_shift, destination_stride, destination_shift,
        message_stride, message_shift, 0);
}

// Tensor-core BF16 up-projection + two-tanh mix for 3 attention streams:
// grid = (ceil(hidden / 128), 3, ceil(rows / 64)), 256 threads.
extern "C" __global__ void __launch_bounds__(INKLING_MSG_MMA_THREADS, 2)
inkling_message_gate_up3_mma(
    const float* source,
    float* dst0,
    float* dst1,
    float* dst2,
    const float* __restrict__ message,
    const unsigned short* __restrict__ forget0,
    const unsigned short* __restrict__ forget1,
    const unsigned short* __restrict__ forget2,
    const unsigned short* __restrict__ gain0,
    const unsigned short* __restrict__ gain1,
    const unsigned short* __restrict__ gain2,
    const float* __restrict__ bottleneck,
    int hidden,
    int rank,
    int rows,
    int source_stride,
    int source_shift0,
    int source_shift1,
    int source_shift2,
    int destination_stride,
    int destination_shift0,
    int destination_shift1,
    int destination_shift2,
    int message_stride,
    int message_shift0,
    int message_shift1,
    int message_shift2) {
    OPT_PDL_ENTRY();
    const int stream = (int)blockIdx.y;
    float* dst = stream == 0 ? dst0 : (stream == 1 ? dst1 : dst2);
    const unsigned short* __restrict__ forget =
        stream == 0 ? forget0 : (stream == 1 ? forget1 : forget2);
    const unsigned short* __restrict__ gain =
        stream == 0 ? gain0 : (stream == 1 ? gain1 : gain2);
    const int src_shift =
        stream == 0 ? source_shift0 : (stream == 1 ? source_shift1 : source_shift2);
    const int dst_shift =
        stream == 0 ? destination_shift0 : (stream == 1 ? destination_shift1 : destination_shift2);
    const int msg_shift =
        stream == 0 ? message_shift0 : (stream == 1 ? message_shift1 : message_shift2);
    inkling_message_gate_up_mma_one(
        source, dst, message, forget, gain, bottleneck, hidden, rank, rows,
        source_stride, src_shift, destination_stride, dst_shift,
        message_stride, msg_shift, stream);
}

// The cross-pass read: one block a (row, core layer, stream) query.
extern "C" __global__ void __launch_bounds__(INKLING_MESSAGE_BLOCK) inkling_cross_pass(
    const float* __restrict__ summaries,
    const float* __restrict__ prev_hidden,
    const float* __restrict__ queries,
    const float* __restrict__ kv,
    float* __restrict__ out,
    int rows,
    int start_block,
    int count,
    int summary_stride,
    int core,
    int hidden,
    float epsilon) {
    OPT_PDL_ENTRY();
    __shared__ float scratch[INKLING_MESSAGE_BLOCK / 32];
    __shared__ float scores[INKLING_MESSAGE_MAX_STATES];
    __shared__ float norms[INKLING_MESSAGE_MAX_STATES];
    const int states = count + 1;
    if (states > INKLING_MESSAGE_MAX_STATES) {
        return;
    }
    const int query_index = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= rows) {
        return;
    }
    const int layer = query_index >> 2;
    const int stream = query_index & 3;
    const float* bank = (stream == 0 || stream == 3) ? queries : kv;
    const int sub = (stream == 2 || stream == 3) ? 1 : 0;
    const float* query = bank + (size_t)(2 * layer + sub) * hidden;
    const float scale = rsqrtf((float)hidden);
    const size_t row_floats = (size_t)hidden;
    const float* row_summaries = summaries + (size_t)row * summary_stride;
    for (int state = 0; state < states; ++state) {
        const float* values;
        if (state < count) {
            values = row_summaries + (size_t)(start_block + state) * row_floats;
        } else {
            values = prev_hidden + (size_t)row * row_floats;
        }
        float squares = 0.0f;
        float dot = 0.0f;
        for (int index = threadIdx.x; index < hidden; index += INKLING_MESSAGE_BLOCK) {
            const float value = values[index];
            squares += value * value;
            dot += value * query[index];
        }
        squares = inkling_message_block_sum(squares, scratch);
        __syncthreads();
        dot = inkling_message_block_sum(dot, scratch);
        __syncthreads();
        if (threadIdx.x == 0) {
            const float norm = rsqrtf(squares / (float)hidden + epsilon);
            norms[state] = norm;
            scores[state] = dot * norm * scale;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float largest = scores[0];
        for (int state = 1; state < states; ++state) {
            largest = fmaxf(largest, scores[state]);
        }
        float total = 0.0f;
        for (int state = 0; state < states; ++state) {
            scores[state] = expf(scores[state] - largest);
            total += scores[state];
        }
        for (int state = 0; state < states; ++state) {
            scores[state] /= total;
        }
    }
    __syncthreads();
    float* destination = out + ((size_t)row * core * 4 + query_index) * row_floats;
    for (int index = threadIdx.x; index < hidden; index += INKLING_MESSAGE_BLOCK) {
        float total = 0.0f;
        for (int state = 0; state < states; ++state) {
            const float* values = state < count
                ? row_summaries + (size_t)(start_block + state) * row_floats
                : prev_hidden + (size_t)row * row_floats;
            total += scores[state] * norms[state] * values[index];
        }
        destination[index] = total;
    }
}

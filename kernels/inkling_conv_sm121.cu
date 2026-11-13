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

// kernels/inkling_conv_sm121.cu
//
// Inkling-2-Small's four causal short convolutions a layer and its block-attention residual.
//
// The short convolutions are depthwise: channel c of a row is a kernel-tap filter of channel
// c of the current and the previous kernel-1 rows, with the taps learned a channel. The
// static form is the one the model note writes as (conv(x) + x) / sqrt(kernel + 1); it serves
// the sliding layers' K/V and every layer's attention and MLP outputs. The dynamic form is
// the full-attention K/V one: a per-head, input-projected tap vector is added to the static
// taps before the filter, there is no residual, and the output divides by
// sqrt(2 * kernel + 1). Decode keeps the history in a [channels][kernel - 1] ring ordered
// oldest to newest in the caller's state, which the decode entries update in place.
//
// The block-attention residual is the other residual topology of the released model: the
// argument to a sublayer is not the additive stream but the output of a softmax attention
// over the stack of block summaries (and, at a block boundary, the partial summary), read by
// one learned query on a sliding layer and three on a full layer. The stack and the partial
// row are unweighted RMS-normalised before the dot products; the output is nq rows of
// hidden, which the arm uses as the sublayer input and appends to the next summary.
//
// ABI:
//
//   inkling_short_conv(x, taps, out, rows, channels, kernel)
//     x, out   [rows][channels] f32 (a row-major matrix)
//     taps     [channels][kernel] BF16, channel-major
//
//   inkling_short_conv_decode(x, taps, state, out, channels, kernel)
//     x, out   [channels] f32
//     state    [channels][kernel - 1] f32, oldest first; updated in place
//
//   inkling_short_conv_dynamic(x, static_taps, dynamic_taps, out, rows, channels, kernel,
//                               head_dim)
//     x, out          [rows][channels] f32
//     static_taps     [channels][kernel] BF16
//     dynamic_taps    [rows][channels / head_dim][kernel] f32
//
//   inkling_short_conv_dynamic_decode(x, static_taps, dynamic_taps, state, out, channels,
//                                      kernel, head_dim)
//     x, out          [channels] f32
//     dynamic_taps    [channels / head_dim][kernel] f32
//     state            [channels][kernel - 1] f32, updated in place
//
//   inkling_block_residual(summaries, partial, queries, out, blocks, hidden, nq, eps)
//     summaries  [blocks][hidden] f32
//     partial    [hidden] f32, nullable
//     queries    [nq][hidden] f32 learned
//     out        [nq][hidden] f32
//
// The kernel width is 4 in the released checkpoint; the entries accept 1..INKLING_MAX_KERNEL
// so the same body can be brought up against a different release.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define INKLING_MAX_KERNEL 8
#define INKLING_MAX_BLOCKS 16
#define INKLING_BLOCK 256

__device__ __forceinline__ float inkling_conv_block_sum(float value, float* scratch) {
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
    for (int index = 1; index < (int)(INKLING_BLOCK / 32); ++index) {
        total += scratch[index];
    }
    return total;
}

// The static multi-row convolution: out = (conv(x) + x) / sqrt(kernel + 1).
extern "C" __global__ void __launch_bounds__(INKLING_BLOCK) inkling_short_conv(
    const float* __restrict__ x,
    const __nv_bfloat16* __restrict__ taps,
    float* __restrict__ out,
    int rows,
    int channels,
    int kernel) {
    const int row = blockIdx.y;
    if (row >= rows) {
        return;
    }
    if (kernel < 1 || kernel > INKLING_MAX_KERNEL) {
        return;
    }
    const float scale = 1.0f / sqrtf((float)(kernel + 1));
    for (int channel = blockIdx.x * INKLING_BLOCK + threadIdx.x; channel < channels;
         channel += gridDim.x * INKLING_BLOCK) {
        float convolution = 0.0f;
        for (int tap = 0; tap < kernel; ++tap) {
            const int source_row = row - tap;
            if (source_row >= 0) {
                // The released taps are a conv1d kernel, so index k multiplies the input at
                // distance kernel - 1 - k: index 0 is the oldest tap. `tap` here is the
                // backward distance, which is why the index runs backwards.
                convolution = __fadd_rn(
                    convolution,
                    __fmul_rn(__bfloat162float(taps[(uint64_t)channel * kernel + (kernel - 1 - tap)]),
                              x[(uint64_t)source_row * channels + channel]));
            }
        }
        out[(uint64_t)row * channels + channel] = __fmul_rn(
            __fadd_rn(convolution, x[(uint64_t)row * channels + channel]), scale);
    }
}

// One decode step of the static form, with the history update.
extern "C" __global__ void __launch_bounds__(INKLING_BLOCK) inkling_short_conv_decode(
    const float* __restrict__ x,
    const __nv_bfloat16* __restrict__ taps,
    float* __restrict__ state,
    float* __restrict__ out,
    int channels,
    int kernel) {
    if (kernel < 1 || kernel > INKLING_MAX_KERNEL) {
        return;
    }
    const float scale = 1.0f / sqrtf((float)(kernel + 1));
    const int history = kernel - 1;
    for (int channel = blockIdx.x * INKLING_BLOCK + threadIdx.x; channel < channels;
         channel += gridDim.x * INKLING_BLOCK) {
        const float current = x[channel];
        float convolution = 0.0f;
        for (int tap = 0; tap < history; ++tap) {
            convolution = __fadd_rn(
                convolution,
                __fmul_rn(__bfloat162float(taps[(uint64_t)channel * kernel + tap]),
                          state[(uint64_t)channel * history + tap]));
        }
        convolution = __fadd_rn(
            convolution,
            __fmul_rn(__bfloat162float(taps[(uint64_t)channel * kernel + history]), current));
        out[channel] = __fmul_rn(__fadd_rn(convolution, current), scale);
        for (int tap = 0; tap + 1 < history; ++tap) {
            state[(uint64_t)channel * history + tap] =
                state[(uint64_t)channel * history + tap + 1];
        }
        if (history > 0) {
            state[(uint64_t)channel * history + history - 1] = current;
        }
    }
}

// The dynamic multi-row convolution: the effective tap is static[c][t] + dynamic[r][head][t],
// head = c / head_dim, and out = conv / sqrt(2 * kernel + 1).
extern "C" __global__ void __launch_bounds__(INKLING_BLOCK) inkling_short_conv_dynamic(
    const float* __restrict__ x,
    const __nv_bfloat16* __restrict__ static_taps,
    const float* __restrict__ dynamic_taps,
    float* __restrict__ out,
    int rows,
    int channels,
    int kernel,
    int head_dim) {
    const int row = blockIdx.y;
    if (row >= rows || head_dim <= 0) {
        return;
    }
    if (kernel < 1 || kernel > INKLING_MAX_KERNEL) {
        return;
    }
    const float scale = 1.0f / sqrtf((float)(2 * kernel + 1));
    const int heads = channels / head_dim;
    for (int channel = blockIdx.x * INKLING_BLOCK + threadIdx.x; channel < channels;
         channel += gridDim.x * INKLING_BLOCK) {
        const int head = channel / head_dim;
        if (head >= heads) {
            continue;
        }
        float convolution = 0.0f;
        for (int tap = 0; tap < kernel; ++tap) {
            const int source_row = row - tap;
            if (source_row < 0) {
                continue;
            }
            // Both tap tables are in conv1d index order, so index k covers distance
            // kernel - 1 - k: static_w is [kernel, channels] and the projection's rows are
            // [heads, kernel] in that same order.
            const int k = kernel - 1 - tap;
            const float weight =
                __fadd_rn(__bfloat162float(static_taps[(uint64_t)k * channels + channel]),
                          dynamic_taps[((uint64_t)row * heads + head) * kernel + k]);
            convolution = __fadd_rn(
                convolution, __fmul_rn(weight, x[(uint64_t)source_row * channels + channel]));
        }
        out[(uint64_t)row * channels + channel] = __fmul_rn(convolution, scale);
    }
}

// One decode step of the dynamic form, with the history update.
extern "C" __global__ void __launch_bounds__(INKLING_BLOCK) inkling_short_conv_dynamic_decode(
    const float* __restrict__ x,
    const __nv_bfloat16* __restrict__ static_taps,
    const float* __restrict__ dynamic_taps,
    float* __restrict__ state,
    float* __restrict__ out,
    int channels,
    int kernel,
    int head_dim) {
    if (kernel < 1 || kernel > INKLING_MAX_KERNEL || head_dim <= 0) {
        return;
    }
    const float scale = 1.0f / sqrtf((float)(2 * kernel + 1));
    const int history = kernel - 1;
    const int heads = channels / head_dim;
    for (int channel = blockIdx.x * INKLING_BLOCK + threadIdx.x; channel < channels;
         channel += gridDim.x * INKLING_BLOCK) {
        const int head = channel / head_dim;
        if (head >= heads) {
            continue;
        }
        const float current = x[channel];
        float convolution = 0.0f;
        for (int tap = 0; tap < history; ++tap) {
            // state[tap] is the sample at distance history - tap, whose conv1d index is
            // kernel - 1 - (history - tap) = tap. So both tables are read at tap, and
            // static_w is [kernel, channels].
            const float weight =
                __fadd_rn(__bfloat162float(static_taps[(uint64_t)tap * channels + channel]),
                          dynamic_taps[(uint64_t)head * kernel + tap]);
            convolution = __fadd_rn(
                convolution, __fmul_rn(weight, state[(uint64_t)channel * history + tap]));
        }
        const float last_weight =
            __fadd_rn(__bfloat162float(static_taps[(uint64_t)history * channels + channel]),
                      dynamic_taps[(uint64_t)head * kernel + history]);
        convolution = __fadd_rn(convolution, __fmul_rn(last_weight, current));
        out[channel] = __fmul_rn(convolution, scale);
        for (int tap = 0; tap + 1 < history; ++tap) {
            state[(uint64_t)channel * history + tap] =
                state[(uint64_t)channel * history + tap + 1];
        }
        if (history > 0) {
            state[(uint64_t)channel * history + history - 1] = current;
        }
    }
}

// The block-attention residual: one block a query. The summary stack plus the optional
// partial row is RMS-normalised (no gamma), each normalised row is scored against the query,
// the scores are softmaxed and the row mix is the query's output. The learned query is the
// only thing that differs between the one-query and three-query schedules; nq is a launch
// parameter.
extern "C" __global__ void __launch_bounds__(INKLING_BLOCK) inkling_block_residual(
    const float* __restrict__ summaries,
    const float* __restrict__ partial,
    const float* __restrict__ queries,
    float* __restrict__ out,
    int blocks,
    int hidden,
    int nq,
    float eps) {
    const int query_index = blockIdx.x;
    if (query_index >= nq || blocks < 0 || blocks > INKLING_MAX_BLOCKS || hidden <= 0) {
        return;
    }
    const int rows = blocks + (partial != nullptr ? 1 : 0);
    if (rows <= 0) {
        return;
    }
    __shared__ float inv_norm[INKLING_MAX_BLOCKS + 1];
    __shared__ float scores[INKLING_MAX_BLOCKS + 1];
    __shared__ float scratch[INKLING_BLOCK];
    __shared__ float probabilities[INKLING_MAX_BLOCKS + 1];

    // The RMS of every row, and the normalised dot of every row against the query.
    for (int row = 0; row < rows; ++row) {
        const float* source = row < blocks ? summaries + (uint64_t)row * hidden : partial;
        float squares = 0.0f;
        for (int index = threadIdx.x; index < hidden; index += INKLING_BLOCK) {
            const float value = source[index];
            squares = __fadd_rn(squares, __fmul_rn(value, value));
        }
        const float total = inkling_conv_block_sum(squares, scratch);
        const float mean = __fdiv_rn(total, (float)hidden);
        const float inverse = __fdiv_rn(1.0f, sqrtf(__fadd_rn(mean, eps)));
        float dot = 0.0f;
        for (int index = threadIdx.x; index < hidden; index += INKLING_BLOCK) {
            dot = __fadd_rn(
                dot, __fmul_rn(queries[(uint64_t)query_index * hidden + index],
                               __fmul_rn(source[index], inverse)));
        }
        const float reduced = inkling_conv_block_sum(dot, scratch);
        if (threadIdx.x == 0) {
            inv_norm[row] = inverse;
            scores[row] = __fmul_rn(reduced, 1.0f / sqrtf((float)hidden));
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float maximum = -INFINITY;
        for (int row = 0; row < rows; ++row) {
            maximum = fmaxf(maximum, scores[row]);
        }
        float denominator = 0.0f;
        for (int row = 0; row < rows; ++row) {
            probabilities[row] = expf(scores[row] - maximum);
            denominator += probabilities[row];
        }
        const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
        for (int row = 0; row < rows; ++row) {
            probabilities[row] *= inverse;
        }
    }
    __syncthreads();

    for (int index = threadIdx.x; index < hidden; index += INKLING_BLOCK) {
        float mixed = 0.0f;
        for (int row = 0; row < blocks; ++row) {
            mixed = __fadd_rn(
                mixed, __fmul_rn(probabilities[row],
                                 __fmul_rn(summaries[(uint64_t)row * hidden + index],
                                           inv_norm[row])));
        }
        if (partial != nullptr) {
            mixed = __fadd_rn(
                mixed, __fmul_rn(probabilities[blocks],
                                 __fmul_rn(partial[index], inv_norm[blocks])));
        }
        out[(uint64_t)query_index * hidden + index] = mixed;
    }
}

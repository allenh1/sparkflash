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
//   inkling_short_conv(x, taps, out, rows, channels, kernel, lead)
//     x, out   [rows][channels] f32 (a row-major matrix)
//     taps     [channels][kernel] BF16, channel-major
//     lead     rows of history physically in front of x, so `x - lead * channels` is the
//              sample one row back. A chunk that continues a sequence passes its kernel - 1
//              rows of history this way and needs no branch on the chunk's first rows; a cold
//              start passes that many zero rows, which is the release's "no such sample".
//
//   inkling_short_conv_decode(x, taps, state, out, channels, kernel)
//     x, out   [channels] f32
//     state    [channels][kernel - 1] f32, oldest first; updated in place
//
//   inkling_short_conv_dynamic(x, static_taps, dynamic_taps, out, rows, channels, kernel,
//                               head_dim, lead)
//     x, out          [rows][channels] f32
//     static_taps     [channels][kernel] BF16
//     dynamic_taps    [rows][channels / head_dim][kernel] f32, one row of taps a row of x:
//                      the dynamic half is a projection of the row's own input, never of the
//                      history, so a lead needs no taps of its own.
//     lead             as in inkling_short_conv
//
//   inkling_short_conv_dynamic_decode(x, static_taps, dynamic_taps, state, out, channels,
//                                      kernel, head_dim)
//     x, out          [channels] f32
//     dynamic_taps    [channels / head_dim][kernel] f32
//     state            [channels][kernel - 1] f32, updated in place
//
//   inkling_conv_history_copy(source, destination, channels, kernel, to_pad)
//     The two layouts of a convolution's history: the decode form keeps a channel's taps
//     together ([channels][kernel - 1], oldest first) and the row form keeps a row's channels
//     together ([kernel - 1][channels], farthest first). `to_pad` reads the first and writes
//     the second; zero does the reverse. A chunk reads its history in the row form and writes
//     the next decode step's back in the channel form.
//
//   inkling_block_residual(summaries, partial, queries, out, blocks, hidden, nq, eps,
//                          summary_stride)
//     summaries  [rows][summary_stride] f32, of which row r's stack is the first
//                `blocks * hidden` floats; a single-row walk passes stride 0
//     partial    [rows][hidden] f32, nullable: each row's current sublayer input
//     queries    [nq][hidden] f32 learned, shared by every row
//     out        [rows][nq][hidden] f32
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
    int kernel,
    int lead) {
    const int row = blockIdx.y;
    if (row >= rows) {
        return;
    }
    if (kernel < 1 || kernel > INKLING_MAX_KERNEL) {
        return;
    }
    const float scale = 1.0f / sqrtf((float)(kernel + 1));
    // `lead` history rows sit in front of the first row of x, so a chunk's first rows read the
    // rows the previous launch ended on instead of a zero that would drop a tap.
    const int current_row = row + lead;
    for (int channel = blockIdx.x * INKLING_BLOCK + threadIdx.x; channel < channels;
         channel += gridDim.x * INKLING_BLOCK) {
        float convolution = 0.0f;
        // The taps are accumulated in the decode entry's order - oldest sample first, the current
        // sample last - and not merely over the same terms: f32 addition is not associative, and
        // a chunk's row has to be the bits its own step would give it, or the two walks' residual
        // streams part company a layer later. `tap` is the distance back, so the oldest sample is
        // the largest tap and the weight index runs from 0 up.
        for (int tap = kernel - 1; tap >= 0; --tap) {
            const int source_row = current_row - tap;
            if (source_row >= 0) {
                convolution = __fadd_rn(
                    convolution,
                    __fmul_rn(__bfloat162float(taps[(uint64_t)channel * kernel + (kernel - 1 - tap)]),
                              x[(uint64_t)source_row * channels + channel]));
            }
        }
        out[(uint64_t)row * channels + channel] = __fmul_rn(
            __fadd_rn(convolution, x[(uint64_t)current_row * channels + channel]), scale);
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

// The dynamic multi-row convolution: the effective tap is static[t][c] + dynamic[r][head][t],
// head = c / head_dim, t is the distance back (index 0 weights the current token), and
// out = (conv + x) / sqrt(2 * kernel + 1) - a residual convolution, like the static one.
extern "C" __global__ void __launch_bounds__(INKLING_BLOCK) inkling_short_conv_dynamic(
    const float* __restrict__ x,
    const __nv_bfloat16* __restrict__ static_taps,
    const float* __restrict__ dynamic_taps,
    float* __restrict__ out,
    int rows,
    int channels,
    int kernel,
    int head_dim,
    int lead) {
    const int row = blockIdx.y;
    if (row >= rows || head_dim <= 0) {
        return;
    }
    if (kernel < 1 || kernel > INKLING_MAX_KERNEL) {
        return;
    }
    const float scale = 1.0f / sqrtf((float)(2 * kernel + 1));
    const int heads = channels / head_dim;
    const int current_row = row + lead;
    for (int channel = blockIdx.x * INKLING_BLOCK + threadIdx.x; channel < channels;
         channel += gridDim.x * INKLING_BLOCK) {
        const int head = channel / head_dim;
        if (head >= heads) {
            continue;
        }
        float convolution = 0.0f;
        // The decode entry accumulates the stored history newest first and the current sample
        // last; the terms here are the same ones in that order, for the same reason the static
        // convolution does: a chunk's row must be the row's own bits.
        for (int lag = 1; lag < kernel; ++lag) {
            const int source_row = current_row - lag;
            if (source_row < 0) {
                continue;
            }
            // Both tap tables count the distance back, so index lag is the sample lag steps ago:
            // static_w is [kernel, channels] and the projection's rows are [heads, kernel].
            const float weight =
                __fadd_rn(__bfloat162float(static_taps[(uint64_t)lag * channels + channel]),
                          dynamic_taps[((uint64_t)row * heads + head) * kernel + lag]);
            convolution = __fadd_rn(
                convolution, __fmul_rn(weight, x[(uint64_t)source_row * channels + channel]));
        }
        {
            const int source_row = current_row;
            const float weight =
                __fadd_rn(__bfloat162float(static_taps[channel]),
                          dynamic_taps[((uint64_t)row * heads + head) * kernel]);
            convolution = __fadd_rn(
                convolution, __fmul_rn(weight, x[(uint64_t)source_row * channels + channel]));
        }
        // The input comes back before the scale: the release's dynamic convolution is a residual
        // convolution.
        const float residual = x[(uint64_t)current_row * channels + channel];
        out[(uint64_t)row * channels + channel] =
            __fmul_rn(__fadd_rn(convolution, residual), scale);
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
        // The history is oldest first and the taps count backwards, so lag 1 - the sample before
        // this one - is the newest stored row.
        for (int lag = 1; lag < kernel; ++lag) {
            const float weight =
                __fadd_rn(__bfloat162float(static_taps[(uint64_t)lag * channels + channel]),
                          dynamic_taps[(uint64_t)head * kernel + lag]);
            convolution = __fadd_rn(
                convolution,
                __fmul_rn(weight, state[(uint64_t)channel * history + (history - lag)]));
        }
        const float current_weight =
            __fadd_rn(__bfloat162float(static_taps[channel]), dynamic_taps[(uint64_t)head * kernel]);
        convolution = __fadd_rn(convolution, __fmul_rn(current_weight, current));
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
    float eps,
    int summary_stride) {
    const int query_index = blockIdx.x;
    const int sequence_row = blockIdx.z;
    if (query_index >= nq || blocks < 0 || blocks > INKLING_MAX_BLOCKS || hidden <= 0) {
        return;
    }
    // A chunk's rows each carry their own stack; a single-row walk passes stride 0 and row 0, so
    // the three offsets below are no-ops.
    summaries += (uint64_t)sequence_row * summary_stride;
    if (partial != nullptr) {
        partial += (uint64_t)sequence_row * hidden;
    }
    out += (uint64_t)sequence_row * nq * hidden;
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

// The convolution history in the other layout, either way.
//
// The decode entry keeps one channel's taps together, because a decode step advances every
// channel independently; a chunk reads its history as rows, because its first rows need the
// samples the previous launch ended on. Both numbers are the same history, and this is the
// conversion between them: no arithmetic, so nothing can round.
extern "C" __global__ void __launch_bounds__(INKLING_BLOCK) inkling_conv_history_copy(
    const float* __restrict__ source,
    float* __restrict__ destination,
    int channels,
    int kernel,
    int to_pad) {
    if (kernel < 1 || kernel > INKLING_MAX_KERNEL || channels <= 0) {
        return;
    }
    const int history = kernel - 1;
    for (int index = blockIdx.x * INKLING_BLOCK + threadIdx.x; index < channels * history;
         index += gridDim.x * INKLING_BLOCK) {
        const int channel = index / history;
        // Both layouts are oldest first, so this is a transpose and not a reversal: tap j of a
        // channel and column j of a row are the same sample.
        const int tap = index - channel * history;
        if (to_pad != 0) {
            destination[(uint64_t)tap * channels + channel] = source[index];
        } else {
            destination[index] = source[(uint64_t)tap * channels + channel];
        }
    }
}

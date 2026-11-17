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
//   inkling_conv_history_restore(table, entries, kernel, row)
//     table    [entries][3] u64: a row-form history's base ([rows][channels] f32), the
//              channel-form history it restores, and its channels
//     row      the base row each restored history starts at
//     Every entry's channel-form history from its row-form rows row .. row + kernel - 2, entry
//     y on the grid's y: what inkling_conv_history_copy with to_pad zero does for one entry, for
//     all of a verify's histories in one launch.
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

#include "opt_pdl_sm121.cuh"

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
    OPT_PDL_ENTRY();
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
    OPT_PDL_ENTRY();
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
    OPT_PDL_ENTRY();
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
    OPT_PDL_ENTRY();
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
// One row of the stack: its RMS inverse and its normalised dot against `query`, scaled by
// 1 / sqrt(hidden). Every thread of the block takes part; the results are thread 0's to keep.
__device__ __forceinline__ void inkling_residual_row(const float* __restrict__ source,
                                                     const float* __restrict__ query,
                                                     int hidden,
                                                     float eps,
                                                     float* scratch,
                                                     float* inverse_out,
                                                     float* score_out) {
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
        dot = __fadd_rn(dot, __fmul_rn(query[index], __fmul_rn(source[index], inverse)));
    }
    const float reduced = inkling_conv_block_sum(dot, scratch);
    *inverse_out = inverse;
    *score_out = __fmul_rn(reduced, 1.0f / sqrtf((float)hidden));
}

// The rows' scores softmaxed into `probabilities`, on one thread.
__device__ __forceinline__ void inkling_residual_softmax(const float* scores,
                                                         float* probabilities,
                                                         int rows) {
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

// Element `index` of the query's mix: the normalised rows weighted in stack order, the partial
// row last.
__device__ __forceinline__ float inkling_residual_mix(const float* __restrict__ summaries,
                                                      const float* __restrict__ partial,
                                                      const float* probabilities,
                                                      const float* inv_norm,
                                                      int blocks,
                                                      int hidden,
                                                      int index) {
    float mixed = 0.0f;
    for (int row = 0; row < blocks; ++row) {
        mixed = __fadd_rn(
            mixed, __fmul_rn(probabilities[row],
                             __fmul_rn(summaries[(uint64_t)row * hidden + index], inv_norm[row])));
    }
    if (partial != nullptr) {
        mixed = __fadd_rn(mixed, __fmul_rn(probabilities[blocks],
                                           __fmul_rn(partial[index], inv_norm[blocks])));
    }
    return mixed;
}

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
    OPT_PDL_ENTRY();
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
        float inverse;
        float score;
        inkling_residual_row(source, queries + (uint64_t)query_index * hidden, hidden, eps,
                             scratch, &inverse, &score);
        if (threadIdx.x == 0) {
            inv_norm[row] = inverse;
            scores[row] = score;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        inkling_residual_softmax(scores, probabilities, rows);
    }
    __syncthreads();

    for (int index = threadIdx.x; index < hidden; index += INKLING_BLOCK) {
        out[(uint64_t)query_index * hidden + index] = inkling_residual_mix(
            summaries, partial, probabilities, inv_norm, blocks, hidden, index);
    }
}

// inkling_block_residual in two launches, its rows side by side. The one-block form walks the
// stack's rows one after another, two block reductions each, which is most of a decode step's
// residual time; here a block takes one row's statistics, and the mix reads them back. Every
// number is the one-block form's: the same row body, softmax and mix, in the same orders.
//
// Statistics: grid (rows, nq, sequence rows), where rows is the stack plus the partial row.
// stats is [sequence rows][nq][INKLING_MAX_BLOCKS + 1][2]: each row's RMS inverse and score.
//
// The query is a weight, so a thread loads its share of it before the previous grid ends; the
// thread then holds its share of the row too, so the dot pass reads nothing again. The sums are
// inkling_residual_row's, the same additions in the same order. A row wider than
// INKLING_RESIDUAL_HELD a thread takes inkling_residual_row itself.
#define INKLING_RESIDUAL_HELD 32
extern "C" __global__ void __launch_bounds__(INKLING_BLOCK) inkling_block_residual_stats(
    const float* __restrict__ summaries,
    const float* __restrict__ partial,
    const float* __restrict__ queries,
    float* __restrict__ stats,
    int blocks,
    int hidden,
    int nq,
    float eps,
    int summary_stride) {
    OPT_PDL_LAUNCH();
    const int row = blockIdx.x;
    const int query_index = blockIdx.y;
    const int sequence_row = blockIdx.z;
    if (query_index >= nq || blocks < 0 || blocks > INKLING_MAX_BLOCKS || hidden <= 0) {
        return;
    }
    const int rows = blocks + (partial != nullptr ? 1 : 0);
    if (row >= rows) {
        return;
    }
    summaries += (uint64_t)sequence_row * summary_stride;
    if (partial != nullptr) {
        partial += (uint64_t)sequence_row * hidden;
    }
    __shared__ float scratch[INKLING_BLOCK];
    const float* source = row < blocks ? summaries + (uint64_t)row * hidden : partial;
    const float* query = queries + (uint64_t)query_index * hidden;
    float inverse;
    float score;
    if (hidden > INKLING_RESIDUAL_HELD * INKLING_BLOCK) {
        OPT_PDL_WAIT();
        inkling_residual_row(source, query, hidden, eps, scratch, &inverse, &score);
    } else {
        // Predicated, not a break: every load of a pass is in flight at once.
        float held_query[INKLING_RESIDUAL_HELD];
#pragma unroll
        for (int k = 0; k < INKLING_RESIDUAL_HELD; ++k) {
            const int index = threadIdx.x + k * INKLING_BLOCK;
            held_query[k] = index < hidden ? query[index] : 0.0f;
        }
        OPT_PDL_WAIT();
        float held[INKLING_RESIDUAL_HELD];
#pragma unroll
        for (int k = 0; k < INKLING_RESIDUAL_HELD; ++k) {
            const int index = threadIdx.x + k * INKLING_BLOCK;
            held[k] = index < hidden ? source[index] : 0.0f;
        }
        float squares = 0.0f;
#pragma unroll
        for (int k = 0; k < INKLING_RESIDUAL_HELD; ++k) {
            if (threadIdx.x + k * INKLING_BLOCK < hidden) {
                squares = __fadd_rn(squares, __fmul_rn(held[k], held[k]));
            }
        }
        const float total = inkling_conv_block_sum(squares, scratch);
        // Every warp has read the first sum before any writes the second into the same scratch:
        // the dot pass no longer waits on loads that kept the two apart.
        __syncthreads();
        const float mean = __fdiv_rn(total, (float)hidden);
        inverse = __fdiv_rn(1.0f, sqrtf(__fadd_rn(mean, eps)));
        float dot = 0.0f;
#pragma unroll
        for (int k = 0; k < INKLING_RESIDUAL_HELD; ++k) {
            if (threadIdx.x + k * INKLING_BLOCK < hidden) {
                dot = __fadd_rn(dot, __fmul_rn(held_query[k], __fmul_rn(held[k], inverse)));
            }
        }
        const float reduced = inkling_conv_block_sum(dot, scratch);
        score = __fmul_rn(reduced, 1.0f / sqrtf((float)hidden));
    }
    if (threadIdx.x == 0) {
        float* cell =
            stats + (((uint64_t)sequence_row * nq + query_index) * (INKLING_MAX_BLOCKS + 1) + row) * 2;
        cell[0] = inverse;
        cell[1] = score;
    }
}

// Mix: grid (ceil(hidden / INKLING_BLOCK), nq, sequence rows). Each block softmaxes the scores
// itself, so every block holds the same probabilities.
extern "C" __global__ void __launch_bounds__(INKLING_BLOCK) inkling_block_residual_mix(
    const float* __restrict__ summaries,
    const float* __restrict__ partial,
    const float* __restrict__ stats,
    float* __restrict__ out,
    int blocks,
    int hidden,
    int nq,
    int summary_stride) {
    OPT_PDL_ENTRY();
    const int query_index = blockIdx.y;
    const int sequence_row = blockIdx.z;
    if (query_index >= nq || blocks < 0 || blocks > INKLING_MAX_BLOCKS || hidden <= 0) {
        return;
    }
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
    __shared__ float probabilities[INKLING_MAX_BLOCKS + 1];
    if ((int)threadIdx.x < rows) {
        const float* cell =
            stats + (((uint64_t)sequence_row * nq + query_index) * (INKLING_MAX_BLOCKS + 1) +
                     threadIdx.x) * 2;
        inv_norm[threadIdx.x] = cell[0];
        scores[threadIdx.x] = cell[1];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        inkling_residual_softmax(scores, probabilities, rows);
    }
    __syncthreads();
    for (int index = blockIdx.x * INKLING_BLOCK + threadIdx.x; index < hidden;
         index += gridDim.x * INKLING_BLOCK) {
        out[(uint64_t)query_index * hidden + index] = inkling_residual_mix(
            summaries, partial, probabilities, inv_norm, blocks, hidden, index);
    }
}

// The chunk walk's block residual: one block a sequence row for all of a call's queries, and the
// next call's scores against the summary rows carried forward.
//
// The one-block form reads a row's stack twice for every query, once for the statistics and once
// for the mix, so a three-query call reads it six times. A summary row does not change once it is
// pushed, and its RMS inverse and its score against a query depend on nothing else. So the mix
// pass, which reads every summary element anyway, also sums the next call's dots over the summary
// rows; the next call takes the statistics of the rows it is handed and reads only the rest (its
// partial row, and a row pushed since) before it reads its stack once for the mix.
//
// Every number is the one-block form's: a row's squares and dots are summed per thread over
// index = threadIdx.x + k * INKLING_BLOCK in ascending k, a warp's sum by the same xor shuffle and
// the warps' in warp order as inkling_conv_block_sum, each score scaled as inkling_residual_row
// scales it, and the softmax and the mix are the same arithmetic in the same orders.
//
//   carry          [sequence rows][carry_stride] f32: the RMS inverse of stack rows
//                  0..INKLING_MAX_BLOCKS, then INKLING_CHAIN_QUERIES runs of as many scores.
//                  Rows 0..carried-1 are read as this call's queries' (the caller's promise that
//                  the previous call scored them); with NEXT > 0, summary rows 0..blocks-1 are
//                  written against next_queries after the block's own reads.
//   next_queries   [NEXT][hidden] f32, the next call's learned queries; null with NEXT 0.
//
// Grid (sequence rows), INKLING_BLOCK threads. A call's query count NQ and the next call's NEXT are
// compile-time, one entry a pair (inkling_block_residual_chain_q<NQ>_n<NEXT>), and the stack holds
// at most INKLING_CHAIN_SUMMARIES summary rows: a thread keeps NEXT dots a summary row across its
// whole pass, so the bound is what keeps those in registers and several blocks on an SM.
#define INKLING_CHAIN_QUERIES 3
#define INKLING_CHAIN_SUMMARIES 8
#define INKLING_CHAIN_WARPS (INKLING_BLOCK / 32)

__device__ __forceinline__ float inkling_chain_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

// The warps' sums of one value in warp order, as inkling_conv_block_sum adds them.
__device__ __forceinline__ float inkling_chain_warps_total(const float* slots) {
    float total = slots[0];
#pragma unroll
    for (int warp = 1; warp < INKLING_CHAIN_WARPS; ++warp) {
        total += slots[warp];
    }
    return total;
}

template <int NQ, int NEXT>
__device__ __forceinline__ void inkling_block_residual_chain_body(
    const float* __restrict__ summaries,
    const float* __restrict__ partial,
    const float* __restrict__ queries,
    float* __restrict__ out,
    float* __restrict__ carry,
    const float* __restrict__ next_queries,
    int blocks,
    int hidden,
    float eps,
    int summary_stride,
    int carried,
    int carry_stride) {
    static_assert(NQ >= 1 && NQ <= INKLING_CHAIN_QUERIES && NEXT >= 0 &&
                      NEXT <= INKLING_CHAIN_QUERIES,
                  "a call scores one to three queries and the next zero to three");
    const int sequence_row = blockIdx.x;
    if (blocks < 0 || blocks > INKLING_CHAIN_SUMMARIES || hidden <= 0 || carried < 0 ||
        carried > blocks ||
        carry_stride < (INKLING_MAX_BLOCKS + 1) * (1 + INKLING_CHAIN_QUERIES)) {
        return;
    }
    summaries += (uint64_t)sequence_row * summary_stride;
    if (partial != nullptr) {
        partial += (uint64_t)sequence_row * hidden;
    }
    out += (uint64_t)sequence_row * NQ * hidden;
    carry += (uint64_t)sequence_row * carry_stride;
    const int rows = blocks + (partial != nullptr ? 1 : 0);
    if (rows <= 0) {
        return;
    }
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    __shared__ float inv_norm[INKLING_CHAIN_SUMMARIES + 1];
    __shared__ float scores[NQ][INKLING_CHAIN_SUMMARIES + 1];
    __shared__ float probabilities[NQ][INKLING_CHAIN_SUMMARIES + 1];
    __shared__ float square_slots[INKLING_CHAIN_WARPS];
    __shared__ float dot_slots[NQ][INKLING_CHAIN_WARPS];
    __shared__ float next_slots[(NEXT > 0 ? NEXT : 1) * INKLING_CHAIN_SUMMARIES][INKLING_CHAIN_WARPS];

    if ((int)threadIdx.x < carried) {
        inv_norm[threadIdx.x] = carry[threadIdx.x];
#pragma unroll
        for (int query = 0; query < NQ; ++query) {
            scores[query][threadIdx.x] =
                carry[(INKLING_MAX_BLOCKS + 1) * (1 + query) + threadIdx.x];
        }
    }
    const float scale = 1.0f / sqrtf((float)hidden);

    // The statistics of the rows not handed over, as inkling_residual_row forms them, every
    // query's dot reduced at once.
    for (int row = carried; row < rows; ++row) {
        const float* source = row < blocks ? summaries + (uint64_t)row * hidden : partial;
        float squares = 0.0f;
        for (int index = threadIdx.x; index < hidden; index += INKLING_BLOCK) {
            const float value = source[index];
            squares = __fadd_rn(squares, __fmul_rn(value, value));
        }
        squares = inkling_chain_warp_sum(squares);
        if (lane == 0u) {
            square_slots[warp] = squares;
        }
        __syncthreads();
        const float mean = __fdiv_rn(inkling_chain_warps_total(square_slots), (float)hidden);
        const float inverse = __fdiv_rn(1.0f, sqrtf(__fadd_rn(mean, eps)));
        float dots[NQ];
#pragma unroll
        for (int query = 0; query < NQ; ++query) {
            dots[query] = 0.0f;
        }
        for (int index = threadIdx.x; index < hidden; index += INKLING_BLOCK) {
            const float value = source[index];
#pragma unroll
            for (int query = 0; query < NQ; ++query) {
                dots[query] =
                    __fadd_rn(dots[query], __fmul_rn(queries[(uint64_t)query * hidden + index],
                                                     __fmul_rn(value, inverse)));
            }
        }
#pragma unroll
        for (int query = 0; query < NQ; ++query) {
            const float sum = inkling_chain_warp_sum(dots[query]);
            if (lane == 0u) {
                dot_slots[query][warp] = sum;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            inv_norm[row] = inverse;
#pragma unroll
            for (int query = 0; query < NQ; ++query) {
                scores[query][row] = __fmul_rn(inkling_chain_warps_total(dot_slots[query]), scale);
            }
        }
    }
    __syncthreads();
    if ((int)threadIdx.x < NQ) {
        inkling_residual_softmax(scores[threadIdx.x], probabilities[threadIdx.x], rows);
    }
    __syncthreads();

    // The mix, every query's from one read of each element, and the next call's dots over the
    // summary rows from the same reads.
    float next_dots[NEXT > 0 ? NEXT : 1][INKLING_CHAIN_SUMMARIES];
#pragma unroll
    for (int query = 0; query < NEXT; ++query) {
#pragma unroll
        for (int row = 0; row < INKLING_CHAIN_SUMMARIES; ++row) {
            next_dots[query][row] = 0.0f;
        }
    }
    for (int index = threadIdx.x; index < hidden; index += INKLING_BLOCK) {
        float mixed[NQ];
        float next_query[NEXT > 0 ? NEXT : 1];
#pragma unroll
        for (int query = 0; query < NQ; ++query) {
            mixed[query] = 0.0f;
        }
#pragma unroll
        for (int query = 0; query < NEXT; ++query) {
            next_query[query] = next_queries[(uint64_t)query * hidden + index];
        }
#pragma unroll
        for (int row = 0; row < INKLING_CHAIN_SUMMARIES; ++row) {
            if (row < blocks) {
                const float value =
                    __fmul_rn(summaries[(uint64_t)row * hidden + index], inv_norm[row]);
#pragma unroll
                for (int query = 0; query < NQ; ++query) {
                    mixed[query] =
                        __fadd_rn(mixed[query], __fmul_rn(probabilities[query][row], value));
                }
#pragma unroll
                for (int query = 0; query < NEXT; ++query) {
                    next_dots[query][row] =
                        __fadd_rn(next_dots[query][row], __fmul_rn(next_query[query], value));
                }
            }
        }
        if (partial != nullptr) {
            const float value = __fmul_rn(partial[index], inv_norm[blocks]);
#pragma unroll
            for (int query = 0; query < NQ; ++query) {
                mixed[query] =
                    __fadd_rn(mixed[query], __fmul_rn(probabilities[query][blocks], value));
            }
        }
#pragma unroll
        for (int query = 0; query < NQ; ++query) {
            out[(uint64_t)query * hidden + index] = mixed[query];
        }
    }
    if constexpr (NEXT > 0) {
#pragma unroll
        for (int query = 0; query < NEXT; ++query) {
#pragma unroll
            for (int row = 0; row < INKLING_CHAIN_SUMMARIES; ++row) {
                if (row < blocks) {
                    const float sum = inkling_chain_warp_sum(next_dots[query][row]);
                    if (lane == 0u) {
                        next_slots[query * INKLING_CHAIN_SUMMARIES + row][warp] = sum;
                    }
                }
            }
        }
        __syncthreads();
        // Every read of the carry above is done: its rows are written for the next call.
        if ((int)threadIdx.x < blocks) {
            carry[threadIdx.x] = inv_norm[threadIdx.x];
        }
        for (int cell = threadIdx.x; cell < NEXT * blocks; cell += INKLING_BLOCK) {
            const int query = cell / blocks;
            const int row = cell - query * blocks;
            carry[(INKLING_MAX_BLOCKS + 1) * (1 + query) + row] = __fmul_rn(
                inkling_chain_warps_total(next_slots[query * INKLING_CHAIN_SUMMARIES + row]),
                scale);
        }
    }
}

#define INKLING_CHAIN_ENTRY(NQ, NEXT)                                                             \
    extern "C" __global__ void __launch_bounds__(INKLING_BLOCK)                                   \
        inkling_block_residual_chain_q##NQ##_n##NEXT(                                             \
            const float* __restrict__ summaries, const float* __restrict__ partial,               \
            const float* __restrict__ queries, float* __restrict__ out,                           \
            float* __restrict__ carry, const float* __restrict__ next_queries, int blocks,        \
            int hidden, float eps, int summary_stride, int carried, int carry_stride) {           \
        OPT_PDL_ENTRY();                                                                          \
        inkling_block_residual_chain_body<NQ, NEXT>(summaries, partial, queries, out, carry,      \
                                                    next_queries, blocks, hidden, eps,            \
                                                    summary_stride, carried, carry_stride);       \
    }

// The pairs Inkling-2-Small's walk makes: a sliding layer's attention (one query) and a full
// layer's (three), each layer's MLP (one), the output (one), and the next call's of each.
INKLING_CHAIN_ENTRY(1, 0)
INKLING_CHAIN_ENTRY(1, 1)
INKLING_CHAIN_ENTRY(1, 3)
INKLING_CHAIN_ENTRY(3, 0)
INKLING_CHAIN_ENTRY(3, 1)
INKLING_CHAIN_ENTRY(3, 3)

// One history's conversion between the two layouts (inkling_conv_history_copy), this thread's
// elements first, first + step, ... of its channels * history.
__device__ __forceinline__ void inkling_conv_history_move(const float* __restrict__ source,
                                                          float* __restrict__ destination,
                                                          int channels, int history, int to_pad,
                                                          int first, int step) {
    for (int index = first; index < channels * history; index += step) {
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
    OPT_PDL_ENTRY();
    if (kernel < 1 || kernel > INKLING_MAX_KERNEL || channels <= 0) {
        return;
    }
    inkling_conv_history_move(source, destination, channels, kernel - 1, to_pad,
                              blockIdx.x * INKLING_BLOCK + threadIdx.x, gridDim.x * INKLING_BLOCK);
}

extern "C" __global__ void __launch_bounds__(INKLING_BLOCK) inkling_conv_history_restore(
    const unsigned long long* __restrict__ table,
    int entries,
    int kernel,
    int row) {
    OPT_PDL_ENTRY();
    if (kernel < 1 || kernel > INKLING_MAX_KERNEL || (int)blockIdx.y >= entries || row < 0) {
        return;
    }
    const unsigned long long* entry = table + 3ull * blockIdx.y;
    const int channels = (int)entry[2];
    if (channels <= 0) {
        return;
    }
    inkling_conv_history_move(reinterpret_cast<const float*>(entry[0]) + (uint64_t)row * channels,
                              reinterpret_cast<float*>(entry[1]), channels, kernel - 1, 0,
                              blockIdx.x * INKLING_BLOCK + threadIdx.x, gridDim.x * INKLING_BLOCK);
}

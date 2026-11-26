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
//   inkling_message_gate(values, message, down, forget, gain, hidden, rank, epsilon,
//                        value_stride, value_shift, message_stride, message_shift)
//     values    [rows][value_stride] f32, updated in place: row r's hidden slice starts at
//               `r * value_stride + value_shift` (a chunk's sublayer inputs are interleaved
//               rows, a decode walk's are contiguous)
//     message   [rows][message_stride] f32: row r's hidden slice starts at
//               `r * message_stride + message_shift`
//     down     [rank][2 * hidden] f32: the concatenated pair to the bottleneck
//     forget   [hidden][rank] f32: the forget gate
//     gain     [hidden][rank] f32: the gain gate
//     rank     at most INKLING_MESSAGE_MAX_RANK (64), the shared-memory bottleneck's bound
//
//   inkling_cross_pass(prev, prev_hidden, queries, kv, out, rows, count, core, hidden,
//                      epsilon)
//     prev          [count][rows][hidden] f32: the previous pass's block summaries
//     prev_hidden   [rows][hidden] f32: the previous pass's final hidden row
//     queries       [2 * core][hidden] f32: the query and MLP stream bank
//     kv            [2 * core][hidden] f32: the key and value stream bank
//     out           [rows][core][4][hidden] f32: four messages a core layer a row
//     core          the core layers the walk loops
//     count         the summary rows the previous pass left (at most INKLING_MESSAGE_MAX_STATES)
//     The state stack is `prev`'s rows and then `prev_hidden`, each unweighted-RMS-normalised.
//     Stream s of layer l reads bank row `2 * l + s / 2` of `queries` for s in {0, 3} and of
//     `kv` for s in {1, 2}.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "opt_pdl_sm121.cuh"

#define INKLING_MESSAGE_BLOCK 256
#define INKLING_MESSAGE_MAX_RANK 64
#define INKLING_MESSAGE_MAX_STATES 72

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
    float* __restrict__ values,
    const float* __restrict__ message,
    const float* __restrict__ down,
    const float* __restrict__ forget,
    const float* __restrict__ gain,
    int hidden,
    int rank,
    float epsilon,
    int value_stride,
    int value_shift,
    int message_stride,
    int message_shift) {
    OPT_PDL_ENTRY();
    __shared__ float scratch[INKLING_MESSAGE_BLOCK / 32];
    __shared__ float bottleneck[INKLING_MESSAGE_MAX_RANK];
    const int row = blockIdx.x;
    float* value = values + (size_t)row * value_stride + value_shift;
    const float* msg = message + (size_t)row * message_stride + message_shift;
    if (rank > INKLING_MESSAGE_MAX_RANK) {
        return;
    }
    float squares_value = 0.0f;
    float squares_message = 0.0f;
    for (int index = threadIdx.x; index < hidden; index += INKLING_MESSAGE_BLOCK) {
        const float source = value[index];
        const float other = msg[index];
        squares_value += source * source;
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
        value[index] = (1.0f - tanhf(forget_value)) * value[index]
            + tanhf(gain_value) * msg[index];
    }
}

// The cross-pass read: one block a (row, core layer, stream) query.
extern "C" __global__ void __launch_bounds__(INKLING_MESSAGE_BLOCK) inkling_cross_pass(
    const float* __restrict__ prev,
    const float* __restrict__ prev_hidden,
    const float* __restrict__ queries,
    const float* __restrict__ kv,
    float* __restrict__ out,
    int rows,
    int count,
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
    const int layer = query_index >> 2;
    const int stream = query_index & 3;
    const float* bank = (stream == 0 || stream == 3) ? queries : kv;
    const int sub = (stream == 2 || stream == 3) ? 1 : 0;
    const float* query = bank + (size_t)(2 * layer + sub) * hidden;
    const float scale = rsqrtf((float)hidden);
    const size_t row_bytes = (size_t)hidden;
    for (int state = 0; state < states; ++state) {
        const float* values;
        if (state < count) {
            values = prev + ((size_t)state * rows + row) * row_bytes;
        } else {
            values = prev_hidden + (size_t)row * row_bytes;
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
    float* destination = out + ((size_t)row * core * 4 + query_index) * row_bytes;
    for (int index = threadIdx.x; index < hidden; index += INKLING_MESSAGE_BLOCK) {
        float total = 0.0f;
        for (int state = 0; state < states; ++state) {
            const float* values = state < count
                ? prev + ((size_t)state * rows + row) * row_bytes
                : prev_hidden + (size_t)row * row_bytes;
            total += scores[state] * norms[state] * values[index];
        }
        destination[index] = total;
    }
}

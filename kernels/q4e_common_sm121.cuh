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

// kernels/q4e_common_sm121.cuh
//
// The warp and block reductions every Qwen4-Exp image uses, and the GLM decode image with them:
// a row dot summed across a warp, a block maximum or sum over shared scratch, and the coalesced
// load. They were copied into four translation units (qwen4_exp, qwen4_attn, qwen4_moe,
// glm_decode) and the copies differed only in their comments; the arithmetic is one statement
// each that four images must not answer differently.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>


// The coalesced read: the read-only path on a device, a plain load where the stub headers
// stand in for the CUDA ones.
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

// sparkflash_weights::linalg::sigmoid, which GLM-5.3-Flash's host twin uses everywhere: 1 / (1 +
// e^-x) at x >= 0 and e^x / (1 + e^x) below, with the accurate exponential. It lives here because
// the shared SwiGLU epilogue and the token image both need it.
__device__ __forceinline__ float q4e_sigmoid_ref(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    }
    const float e = expf(x);
    return e / (1.0f + e);
}

// One rotary pair of a head at absolute position `position`: elements index and index + half of
// `src`, rotated by the frequency theta^(-2 index / rotary) (the accurate power) through the fast
// cosine and sine, into the same elements of `dst` (which may be `src`). q4e_rope and Dory's
// dory_rope_norm both rotate through it, so they cannot answer a position differently; q4e_rope's
// SASS is byte for byte what its inline rotation compiled to.
__device__ __forceinline__ void q4e_rope_pair(const float* src, float* dst, int index, int half,
                                              int rotary, float position, float theta) {
    const float frequency = powf(theta, -2.0f * (float)index / (float)rotary);
    const float angle = position * frequency;
    const float cosine = __cosf(angle);
    const float sine = __sinf(angle);
    const float a = src[index];
    const float b = src[index + half];
    dst[index] = a * cosine - b * sine;
    dst[index + half] = a * sine + b * cosine;
}

// One row per warp, one lane per 32 columns: the packed weights are read coalesced.
__device__ __forceinline__ float q4e_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
    }
    return value;
}

// A warp's xor butterfly sum: every lane holds the total (the lane-0 form above leaves only
// lane 0 valid). Used where a whole warp needs the result, as the tiled GEMV and the drafter do.
__device__ __forceinline__ float q4e_warp_sum_all(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xFFFFFFFFu, value, offset);
    }
    return value;
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



// Keep the higher value, breaking ties toward the lower expert index.
__device__ __forceinline__ void q4e_take_better(float other_value, int other_index, float* value,
                                                int* index) {
    if (other_value > *value || (other_value == *value && other_index < *index)) {
        *value = other_value;
        *index = other_index;
    }
}

// The reference SwiGLU gate: x / (1 + e^-x) with the fast exponential. One expression serves every
// token and prefill image so a gate computes the same value wherever it runs.
__device__ __forceinline__ float q4e_silu(float value) {
    return value / (1.0f + __expf(-value));
}

// Block argmax over (value, index) pairs, with the same tie-break as the host sort: the higher
// value wins and an equal value prefers the lower index. shared_value and shared_index hold
// blockDim.x / 32 entries each.
__device__ __forceinline__ void q4e_block_best(float* value, int* index, float* shared_value,
                                               int* shared_index) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = (blockDim.x + 31) >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        q4e_take_better(__shfl_down_sync(0xFFFFFFFFu, *value, offset),
                        (int)__shfl_down_sync(0xFFFFFFFFu, (unsigned int)*index, offset), value,
                        index);
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
            q4e_take_better(
                __shfl_down_sync(0xFFFFFFFFu, best_value, offset),
                (int)__shfl_down_sync(0xFFFFFFFFu, (unsigned int)best_index, offset), &best_value,
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


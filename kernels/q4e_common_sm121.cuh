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

// One row per warp, one lane per 32 columns: the packed weights are read coalesced.
__device__ __forceinline__ float q4e_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
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


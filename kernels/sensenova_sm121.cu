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

// SenseNova-Loop's own arithmetic: YaRN rotary, the interleaved query/gate split, and the
// two-stream manifold-constrained hyper-connection. Everything else is the shared Qwen token,
// attention and scan images.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define SENSENOVA_BLOCK 256

__device__ __forceinline__ float sensenova_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

extern "C" __global__ void sensenova_rope(float* __restrict__ values, int head_dim, int rotated,
                                          int position, const float* __restrict__ inv_freq,
                                          float scale, int row_stride, int batch) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    if (head >= (int)blockDim.x || row >= batch || rotated < 2 || (rotated & 1)) {
        return;
    }
    const int half = rotated >> 1;
    float* base = values + (unsigned long long)row * row_stride + head * head_dim;
    for (int pair = threadIdx.x; pair < half; pair += blockDim.x) {
        const float angle = position * inv_freq[pair];
        float sine, cosine;
        sincosf(angle, &sine, &cosine);
        const float sin_scaled = sine * scale;
        const float cos_scaled = cosine * scale;
        const float left = base[pair];
        const float right = base[pair + half];
        base[pair] = left * cos_scaled - right * sin_scaled;
        base[pair + half] = left * sin_scaled + right * cos_scaled;
    }
}

// Splits Qwen3.5's per-head [query | gate] projection into two contiguous rows.
extern "C" __global__ void sensenova_qgate(const float* __restrict__ interleaved,
                                           float* __restrict__ query, float* __restrict__ gate,
                                           int head_dim, int heads, int batch) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    if (head >= heads || row >= batch) {
        return;
    }
    const int width = heads * head_dim;
    const float* source =
        interleaved + (unsigned long long)row * 2 * width + head * 2 * head_dim;
    float* q = query + (unsigned long long)row * width + head * head_dim;
    float* g = gate + (unsigned long long)row * width + head * head_dim;
    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        q[dim] = source[dim];
        g[dim] = source[head_dim + dim];
    }
}

// One token's mHC router: normalized two-stream RMS, eight fp32 logits, pre/post sigmoids and
// the 2x2 log-space Sinkhorn. `coeffs` is [batch][pre (2) | post (2) | mix (4)].
extern "C" __global__ void sensenova_mhc_pre(const float* __restrict__ streams,
                                             const float* __restrict__ weight,
                                             const float* __restrict__ bias,
                                             float* __restrict__ branch,
                                             float* __restrict__ coeffs, int hidden, int streams_n,
                                             int iterations, float epsilon) {
    __shared__ float squares[SENSENOVA_BLOCK];
    __shared__ float logits[8];
    __shared__ float pre[2];
    __shared__ float post[2];
    __shared__ float mix[4];
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const int width = streams_n * hidden;
    const float* flat = streams + (unsigned long long)row * width;
    float sum = 0.0f;
    for (int index = lane; index < width; index += blockDim.x) {
        const float value = flat[index];
        sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    squares[lane] = sum;
    __syncthreads();
    for (int stride = SENSENOVA_BLOCK / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            squares[lane] = __fadd_rn(squares[lane], squares[lane + stride]);
        }
        __syncthreads();
    }
    const float mean = __fdiv_rn(squares[0], (float)width);
    const float scale = rsqrtf(__fadd_rn(mean, epsilon));
    if (lane < 8) {
        float accumulator = 0.0f;
        for (int index = 0; index < width; ++index) {
            accumulator = __fadd_rn(
                accumulator,
                __fmul_rn(weight[(unsigned long long)lane * width + index],
                          __fmul_rn(flat[index], scale)));
        }
        logits[lane] = __fadd_rn(accumulator, bias[lane]);
    }
    __syncthreads();
    if (lane == 0) {
        pre[0] = sensenova_sigmoid(logits[0]);
        pre[1] = sensenova_sigmoid(logits[1]);
        post[0] = 2.0f * sensenova_sigmoid(logits[2]);
        post[1] = 2.0f * sensenova_sigmoid(logits[3]);
        float cells[4] = {logits[4], logits[5], logits[6], logits[7]};
        for (int pass = 0; pass < iterations; ++pass) {
            for (int r = 0; r < 2; ++r) {
                const float maximum = fmaxf(cells[r * 2], cells[r * 2 + 1]);
                const float e0 = expf(cells[r * 2] - maximum);
                const float e1 = expf(cells[r * 2 + 1] - maximum);
                const float total = __fadd_rn(e0, e1);
                cells[r * 2] = __fsub_rn(cells[r * 2], __fadd_rn(maximum, logf(total)));
                cells[r * 2 + 1] =
                    __fsub_rn(cells[r * 2 + 1], __fadd_rn(maximum, logf(total)));
            }
            for (int c = 0; c < 2; ++c) {
                const float maximum = fmaxf(cells[c], cells[2 + c]);
                const float e0 = expf(cells[c] - maximum);
                const float e1 = expf(cells[2 + c] - maximum);
                const float total = __fadd_rn(e0, e1);
                cells[c] = __fsub_rn(cells[c], __fadd_rn(maximum, logf(total)));
                cells[2 + c] = __fsub_rn(cells[2 + c], __fadd_rn(maximum, logf(total)));
            }
        }
        mix[0] = expf(cells[0]);
        mix[1] = expf(cells[1]);
        mix[2] = expf(cells[2]);
        mix[3] = expf(cells[3]);
    }
    __syncthreads();
    float* out = branch + (unsigned long long)row * hidden;
    const float* first = flat;
    const float* second = flat + hidden;
    for (int column = lane; column < hidden; column += blockDim.x) {
        out[column] = __fadd_rn(__fmul_rn(pre[0], first[column]),
                                __fmul_rn(pre[1], second[column]));
    }
    if (lane == 0) {
        float* c = coeffs + (unsigned long long)row * 8;
        c[0] = pre[0];
        c[1] = pre[1];
        c[2] = post[0];
        c[3] = post[1];
        c[4] = mix[0];
        c[5] = mix[1];
        c[6] = mix[2];
        c[7] = mix[3];
    }
}

// The merge side of the same router: each stream becomes mix[k] @ streams + post[k] * branch.
extern "C" __global__ void sensenova_mhc_post(float* __restrict__ streams,
                                               const float* __restrict__ branch,
                                               const float* __restrict__ coeffs, int hidden) {
    const int row = blockIdx.x;
    const int offset = blockIdx.y * blockDim.x + threadIdx.x;
    if (offset >= hidden) {
        return;
    }
    const float* c = coeffs + (unsigned long long)row * 8;
    float* first = streams + (unsigned long long)row * 2 * hidden;
    float* second = first + hidden;
    const float a = first[offset];
    const float b = second[offset];
    const float branch_value = branch[(unsigned long long)row * hidden + offset];
    first[offset] = __fadd_rn(__fadd_rn(__fmul_rn(c[4], a), __fmul_rn(c[5], b)),
                               __fmul_rn(c[2], branch_value));
    second[offset] = __fadd_rn(__fadd_rn(__fmul_rn(c[6], a), __fmul_rn(c[7], b)),
                                __fmul_rn(c[3], branch_value));
}

// Collapses the two streams at the last mHC layer.
extern "C" __global__ void sensenova_mhc_leave(const float* __restrict__ streams,
                                                 float* __restrict__ out, int hidden) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= hidden) {
        return;
    }
    out[column] = 0.5f * __fadd_rn(streams[column], streams[hidden + column]);
}

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
#include <cuda_fp8.h>
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
    // One warp a router row: the released loop ran lane row over all width columns in order,
    // which is 8192 serial dependent FMAs on eight lanes and cost 0.4 ms a router. A warp holds
    // the same row, so the sum is a tree over the lanes and the row streams coalesced.
    {
        const int warp = threadIdx.x >> 5;
        const int sub = threadIdx.x & 31;
        if (warp < 8) {
            const unsigned long long base = (unsigned long long)warp * width;
            float accumulator = 0.0f;
            for (int index = sub; index < width; index += 32) {
                accumulator = __fadd_rn(
                    accumulator, __fmul_rn(weight[base + index], __fmul_rn(flat[index], scale)));
            }
#pragma unroll
            for (int stride = 16; stride > 0; stride >>= 1) {
                accumulator = __fadd_rn(
                    accumulator, __shfl_down_sync(0xFFFFFFFFu, accumulator, stride));
            }
            if (sub == 0) {
                logits[warp] = __fadd_rn(accumulator, bias[warp]);
            }
        }
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

// The 8-bit K/V cache: one E4M3 code a value with one f32 inverse scale a (token, KV head).
// The scale maps the head's largest magnitude to 448, E4M3's largest finite value; a zero head
// takes scale 1 so every code is 0.
__device__ __forceinline__ float sensenova_e4m3(unsigned char code) {
    return __half2float(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3));
}

extern "C" __global__ void sensenova_cache_q8(const float* __restrict__ source,
                                               unsigned char* __restrict__ codes,
                                               float* __restrict__ scales, int base, int rows,
                                               int kv_heads, int head_dim) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= rows || head >= kv_heads) {
        return;
    }
    const int width = kv_heads * head_dim;
    const float* src = source + (unsigned long long)row * width + head * head_dim;
    unsigned char* dst = codes + (unsigned long long)(base + row) * width + head * head_dim;
    float largest = 0.0f;
    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        largest = fmaxf(largest, fabsf(src[dim]));
    }
    __shared__ float scratch[256];
    scratch[threadIdx.x] = largest;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float scale = scratch[0] > 0.0f ? scratch[0] / 448.0f : 1.0f;
    if (threadIdx.x == 0) {
        scales[(unsigned long long)(base + row) * kv_heads + head] = scale;
    }
    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        dst[dim] = __nv_cvt_float_to_fp8(src[dim] / scale, __NV_SATFINITE, __NV_E4M3);
    }
}

__device__ __forceinline__ float sensenova_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float sensenova_block_sum(float value, float* scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = sensenova_warp_sum(value);
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = lane < (int)(blockDim.x >> 5) ? scratch[lane] : 0.0f;
        value = sensenova_warp_sum(value);
        if (lane == 0) {
            scratch[0] = value;
        }
    }
    __syncthreads();
    return scratch[0];
}

__device__ __forceinline__ float sensenova_block_max(float value, float* scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFu, value, offset));
    }
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = lane < (int)(blockDim.x >> 5) ? scratch[lane] : -INFINITY;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFu, value, offset));
        }
        if (lane == 0) {
            scratch[0] = value;
        }
    }
    __syncthreads();
    return scratch[0];
}

#define SENSENOVA_FULL_TILE 256
#define SENSENOVA_FULL_SPLITS 16

__device__ __forceinline__ int sensenova_full_splits(int visible) {
    return min(SENSENOVA_FULL_SPLITS, (visible + SENSENOVA_FULL_TILE - 1) / SENSENOVA_FULL_TILE);
}

// q4e_attn_full over an 8-bit cache: same split, tiling, online softmax and partial layout,
// with K and V decoded as code * per-(token, head) scale.
extern "C" __global__ void sensenova_attn_full_q8(const float* __restrict__ q,
                                                   const unsigned char* __restrict__ keys,
                                                   const unsigned char* __restrict__ values,
                                                   const float* __restrict__ k_scales,
                                                   const float* __restrict__ v_scales,
                                                   float* __restrict__ partials, int position,
                                                   int heads, int kv_heads, int head_dim,
                                                   int groups) {
    __shared__ float logits[SENSENOVA_FULL_TILE];
    __shared__ float reduce[256];
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int split = blockIdx.z;
    const int visible = position + row + 1;
    const int splits = sensenova_full_splits(visible);
    if (split >= splits || head_dim <= 0 || head_dim > 256) {
        return;
    }
    const int span = (visible + splits - 1) / splits;
    const int first = split * span;
    const int last = min(visible, first + span);
    const int kv_head = head / groups;
    const float* qh = q + ((unsigned long long)row * heads + head) * head_dim;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const float scale = sqrtf((float)head_dim);
    const unsigned long long kv_row = (unsigned long long)kv_heads * head_dim;
    float maximum = -INFINITY;
    float sum = 0.0f;
    float accumulator = 0.0f;
    for (int base = first; base < last; base += SENSENOVA_FULL_TILE) {
        const int count = min(SENSENOVA_FULL_TILE, last - base);
        for (int slot = warp; slot < count; slot += warps) {
            const int at = base + slot;
            const float k_scale = k_scales[(unsigned long long)at * kv_heads + kv_head];
            const unsigned char* krow = keys + (unsigned long long)at * kv_row + kv_head * head_dim;
            float dot = 0.0f;
            for (int dim = lane; dim < head_dim; dim += 32) {
                dot += qh[dim] * (sensenova_e4m3(krow[dim]) * k_scale);
            }
            dot = sensenova_warp_sum(dot);
            if (lane == 0) {
                logits[slot] = dot / scale;
            }
        }
        __syncthreads();
        float local_max = -INFINITY;
        for (int slot = threadIdx.x; slot < count; slot += blockDim.x) {
            local_max = fmaxf(local_max, logits[slot]);
        }
        const float next = fmaxf(maximum, sensenova_block_max(local_max, reduce));
        const float rescale = __expf(maximum - next);
        float local_sum = 0.0f;
        for (int slot = threadIdx.x; slot < count; slot += blockDim.x) {
            const float weight = __expf(logits[slot] - next);
            logits[slot] = weight;
            local_sum += weight;
        }
        sum = sum * rescale + sensenova_block_sum(local_sum, reduce);
        if (threadIdx.x < head_dim) {
            float tile = 0.0f;
            for (int slot = 0; slot < count; ++slot) {
                const int at = base + slot;
                const float v_scale = v_scales[(unsigned long long)at * kv_heads + kv_head];
                const unsigned char* vrow =
                    values + (unsigned long long)at * kv_row + kv_head * head_dim;
                tile += logits[slot] * (sensenova_e4m3(vrow[threadIdx.x]) * v_scale);
            }
            accumulator = accumulator * rescale + tile;
        }
        maximum = next;
        __syncthreads();
    }
    float* part = partials +
                  (((unsigned long long)row * heads + head) * SENSENOVA_FULL_SPLITS + split) *
                      (head_dim + 2);
    if (threadIdx.x == 0) {
        part[0] = maximum;
        part[1] = sum;
    }
    if (threadIdx.x < head_dim) {
        part[2 + threadIdx.x] = accumulator;
    }
}

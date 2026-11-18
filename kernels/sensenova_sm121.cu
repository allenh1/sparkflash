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

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define SENSENOVA_BLOCK 256

__device__ __forceinline__ float sensenova_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// The position is a device word so the launch list does not change a token: a captured CUDA
// graph replays the arguments it was captured with, and only the buffer's contents advance.
extern "C" __global__ void sensenova_rope(float* __restrict__ values, int head_dim, int rotated,
                                          const int* __restrict__ position,
                                          const float* __restrict__ inv_freq, float scale,
                                          int row_stride, int batch) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    if (head >= (int)blockDim.x || row >= batch || rotated < 2 || (rotated & 1)) {
        return;
    }
    const int at = *position + row;
    const int half = rotated >> 1;
    float* base = values + (unsigned long long)row * row_stride + head * head_dim;
    for (int pair = threadIdx.x; pair < half; pair += blockDim.x) {
        const float angle = at * inv_freq[pair];
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

// The enter side: a row's residual becomes both streams. One block a row, so a chunk is one
// launch instead of two copies a row.
extern "C" __global__ void sensenova_mhc_enter(const float* __restrict__ hidden,
                                                float* __restrict__ streams, int width, int rows) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const float* source = hidden + (unsigned long long)row * width;
    float* first = streams + (unsigned long long)row * 2 * width;
    float* second = first + width;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = source[column];
        first[column] = value;
        second[column] = value;
    }
}

// Collapses the two streams at the last mHC layer.
extern "C" __global__ void sensenova_mhc_leave(const float* __restrict__ streams,
                                                 float* __restrict__ out, int hidden, int rows) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const float* pair = streams + (unsigned long long)row * 2 * hidden;
    float* target = out + (unsigned long long)row * hidden;
    for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
        target[column] = 0.5f * __fadd_rn(pair[column], pair[hidden + column]);
    }
}

// The 8-bit K/V cache: one E4M3 code a value with one f32 inverse scale a (token, KV head).
// The scale maps the head's largest magnitude to 448, E4M3's largest finite value; a zero head
// takes scale 1 so every code is 0.
__device__ __forceinline__ float sensenova_e4m3(unsigned char code) {
    return __half2float(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3));
}

extern "C" __global__ void sensenova_cache_q8(const float* __restrict__ source,
                                               unsigned char* __restrict__ codes,
                                               float* __restrict__ scales,
                                               const int* __restrict__ position, int rows,
                                               int kv_heads, int head_dim) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= rows || head >= kv_heads) {
        return;
    }
    const int width = kv_heads * head_dim;
    const float* src = source + (unsigned long long)row * width + head * head_dim;
    unsigned char* dst =
        codes + (unsigned long long)(*position + row) * width + head * head_dim;
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
        scales[(unsigned long long)(*position + row) * kv_heads + head] = scale;
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
                                                   float* __restrict__ partials,
                                                   const int* __restrict__ position, int heads,
                                                   int kv_heads, int head_dim, int groups) {
    __shared__ float logits[SENSENOVA_FULL_TILE];
    __shared__ float reduce[256];
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int split = blockIdx.z;
    const int visible = *position + row + 1;
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

// q4e_attn_full_combine over a device position: the shared kernel takes the token's index as an
// argument, which would make every replayed launch a different graph node argument.
extern "C" __global__ void sensenova_attn_combine(const float* __restrict__ partials,
                                                   const float* __restrict__ gate,
                                                   float* __restrict__ out,
                                                   const int* __restrict__ position,
                                                   int heads, int head_dim) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int splits = sensenova_full_splits(*position + row + 1);
    const int stride = head_dim + 2;
    const float* part =
        partials + ((unsigned long long)row * heads + head) * SENSENOVA_FULL_SPLITS * stride;
    float maximum = -INFINITY;
    for (int split = 0; split < splits; ++split) {
        maximum = fmaxf(maximum, part[split * stride]);
    }
    float total = 0.0f;
    for (int split = 0; split < splits; ++split) {
        total += part[split * stride + 1] * __expf(part[split * stride] - maximum);
    }
    const unsigned long long at = ((unsigned long long)row * heads + head) * head_dim;
    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        float accumulator = 0.0f;
        for (int split = 0; split < splits; ++split) {
            accumulator += part[split * stride + 2 + dim] * __expf(part[split * stride] - maximum);
        }
        const float value = accumulator / total;
        out[at + dim] = gate != nullptr ? value / (1.0f + __expf(-gate[at + dim])) : value;
    }
}


// ---------------------------------------------------------------------------------------------
// The prefill GEMM: NVFP4 weights and activations through the mxf4nvf4 block-scaled MMA.
//
// The 5080 (sm_120a) runs one m16n8k64 e2m1 MMA every 4 SM cycles - 2048 MAC a cycle, twice the
// FP8 form and four times the BF16 body the first prefill used - and carries the block scales in
// the instruction, so the accumulator needs no per-tile fold. The operands are ModelOpt NVFP4:
// two E2M1 nibbles a byte (the low nibble the even column) with one E4M3 scale a 16 values, which
// is exactly the MMA's own block, so a scale never has to be applied by hand.
//
// The fragment layout (measured on the part, not assumed): lane l holds A row l/4 and l/4+8, its
// K 16*(l%4) block as eight contiguous nibbles - the low four in the first register, the high
// four in the third for A (second for B) - so a lane reads one 8-byte run a row and needs no
// ldmatrix. Lane 4a+2b supplies the A scale of row a+8b and lane 4c the B scale of column c, a
// u32 of the four 16-value blocks the stage covers.

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 1200

// D += A x B over one 64-value K block, E2M1 x E2M1 with ue4m3 block scales.
__device__ __forceinline__ void sensenova_mma_fp4(float d[4], const uint32_t a[4], uint32_t b0,
                                                  uint32_t b1, uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa),
          "h"((uint16_t)0), "h"((uint16_t)0), "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

__device__ __forceinline__ uint32_t sensenova_smem(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void sensenova_cp16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void sensenova_cp4(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void sensenova_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void sensenova_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// The E2M1 magnitude of |v| / scale: the eight codes 0, 0.5, 1, 1.5, 2, 3, 4, 6, nearest first.
__device__ __forceinline__ uint32_t sensenova_e2m1(float value) {
    const float magnitude = fabsf(value);
    const uint32_t code = magnitude <= 0.25f   ? 0u
                          : magnitude <= 0.75f ? 1u
                          : magnitude <= 1.25f ? 2u
                          : magnitude <= 1.75f ? 3u
                          : magnitude <= 2.50f ? 4u
                          : magnitude <= 3.50f ? 5u
                          : magnitude <= 5.00f ? 6u
                                               : 7u;
    return code | (value < 0.0f ? 8u : 0u);
}

// How far a lane's 16-value K block sits from the start of its row: the low nibble first, so a
// block is eight bytes.
__device__ __forceinline__ uint32_t sensenova_pair_off(uint32_t row, uint32_t pair) {
    return row * 32u + ((pair ^ (row & 1u)) << 4);
}

// x[rows][inner] f32 -> codes[rows][inner / 2] (two nibbles a byte) and scales[rows][inner / 16]
// E4M3. One thread a 16-value block: its own amax, its own scale, its own eight packed bytes.
extern "C" __global__ void sensenova_quant_fp4(const float* __restrict__ x,
                                               uint8_t* __restrict__ codes,
                                               uint8_t* __restrict__ scales, unsigned int rows,
                                               unsigned int inner) {
    const unsigned int blocks = inner >> 4;
    const unsigned int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= rows * blocks) return;
    const float* source = x + (unsigned long long)at * 16u;
    float values[16];
    float largest = 0.0f;
#pragma unroll
    for (unsigned int i = 0; i < 16u; ++i) {
        values[i] = source[i];
        largest = fmaxf(largest, fabsf(values[i]));
    }
    float scale = largest / 6.0f;
    __nv_fp8_storage_t storage = __nv_cvt_float_to_fp8(scale, __NV_SATFINITE, __NV_E4M3);
    float stored = sensenova_e4m3(storage);
    if (stored == 0.0f) stored = 1.0f;
    const float inverse = 1.0f / stored;
    scales[at] = (uint8_t)storage;
    uint64_t packed = 0;
#pragma unroll
    for (unsigned int i = 0; i < 16u; ++i) {
        packed |= (uint64_t)sensenova_e2m1(values[i] * inverse) << (4u * i);
    }
    *reinterpret_cast<uint64_t*>(codes + (unsigned long long)at * 8u) = packed;
}

// An FP8 [cols][inner] matrix with a BF16 inverse scale a 128x128 tile -> the same NVFP4 form:
// codes[cols][inner / 2] and scales[cols][inner / 16]. One thread a 16-value block.
extern "C" __global__ void sensenova_quant_weights_fp4(const uint8_t* __restrict__ w,
                                                       const __nv_bfloat16* __restrict__ ws,
                                                       uint8_t* __restrict__ codes,
                                                       uint8_t* __restrict__ scales,
                                                       unsigned int cols, unsigned int inner) {
    const unsigned int blocks = inner >> 4;
    const unsigned int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= cols * blocks) return;
    const unsigned int col = at / blocks;
    const unsigned int block = at % blocks;
    const float tile = __bfloat162float(ws[(col >> 7) * (inner >> 7) + (block >> 3)]);
    const uint8_t* source = w + (unsigned long long)col * inner + block * 16u;
    float values[16];
    float largest = 0.0f;
#pragma unroll
    for (unsigned int i = 0; i < 16u; ++i) {
        const __nv_fp8_storage_t storage = source[i];
        values[i] = (float)(*reinterpret_cast<const __nv_fp8_e4m3*>(&storage)) * tile;
        largest = fmaxf(largest, fabsf(values[i]));
    }
    float scale = largest / 6.0f;
    __nv_fp8_storage_t storage = __nv_cvt_float_to_fp8(scale, __NV_SATFINITE, __NV_E4M3);
    float stored = sensenova_e4m3(storage);
    if (stored == 0.0f) stored = 1.0f;
    const float inverse = 1.0f / stored;
    scales[at] = (uint8_t)storage;
    uint64_t packed = 0;
#pragma unroll
    for (unsigned int i = 0; i < 16u; ++i) {
        packed |= (uint64_t)sensenova_e2m1(values[i] * inverse) << (4u * i);
    }
    *reinterpret_cast<uint64_t*>(codes + (unsigned long long)at * 8u) = packed;
}

// out[t][c] = a[t][:] . w[c][:] over NVFP4 operands. BM rows x BN columns a block, eight warps,
// the K dimension in 64-value stages through STAGES cp.async buffers.
template <int BM, int BN, int STAGES, int WM, int WN>
__device__ __forceinline__ void sensenova_dense_fp4_body(const uint8_t* __restrict__ w,
                                                         const uint8_t* __restrict__ ws,
                                                         const uint8_t* __restrict__ a,
                                                         const uint8_t* __restrict__ as,
                                                         float* __restrict__ out,
                                                         unsigned int cols, unsigned int inner,
                                                         unsigned int tokens) {
    constexpr int MT = BM / WM / 16;
    constexpr int NT = BN / WN / 8;
    constexpr unsigned STAGE_BYTES = BM * 32u + BN * 32u + BM * 4u + BN * 4u;
    extern __shared__ __align__(128) uint8_t sensenova_fp4_shared[];
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    const unsigned int n0 = blockIdx.x * BN;
    const unsigned int m0 = blockIdx.y * BM;
    const unsigned int stages = inner / 64u;
    const unsigned int packed = inner >> 1;
    auto s_a = [&](unsigned int buf) { return sensenova_fp4_shared + buf * STAGE_BYTES; };
    auto s_w = [&](unsigned int buf) { return s_a(buf) + BM * 32u; };
    auto s_as = [&](unsigned int buf) { return s_w(buf) + BN * 32u; };
    auto s_ws = [&](unsigned int buf) { return s_as(buf) + BM * 4u; };

    // A stage is BM + BN rows of 32 bytes: two 16-byte runs a row, 256 threads covering the
    // larger of the two tiles two rows at a time.
    auto issue = [&](unsigned int stage, unsigned int buf) {
#pragma unroll
        for (unsigned int i = 0; i < 2u; ++i) {
            const unsigned int at = tid + i * 256u;
            const unsigned int row = at >> 1;
            const unsigned int pair = at & 1u;
            if (row < BM) {
                sensenova_cp16(sensenova_smem(s_a(buf) + sensenova_pair_off(row, pair)),
                               a + (unsigned long long)(m0 + row) * packed + stage * 32u +
                                   pair * 16u);
            }
            if (row < BN) {
                sensenova_cp16(sensenova_smem(s_w(buf) + sensenova_pair_off(row, pair)),
                               w + (unsigned long long)(n0 + row) * packed + stage * 32u +
                                   pair * 16u);
            }
        }
        if (tid < BM) {
            sensenova_cp4(sensenova_smem(s_as(buf) + tid * 4u),
                          as + (unsigned long long)(m0 + tid) * (inner >> 4) + stage * 4u);
        }
        if (tid < BN) {
            sensenova_cp4(sensenova_smem(s_ws(buf) + tid * 4u),
                          ws + (unsigned long long)(n0 + tid) * (inner >> 4) + stage * 4u);
        }
    };

    const unsigned int wm = warp / WN, wn = warp % WN;
    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    const unsigned int chunk = lane & 3u;
    const unsigned int gr = lane >> 2;
    const bool scale_a_lane = (lane & 3u) < 2u;
    const bool scale_b_lane = (lane & 3u) == 0u;

#pragma unroll
    for (int stage = 0; stage + 1 < STAGES; ++stage) {
        if (stage < (int)stages) issue(stage, stage);
        sensenova_commit();
    }
    for (unsigned int stage = 0; stage < stages; ++stage) {
        sensenova_wait<STAGES - 2>();
        __syncthreads();
        const unsigned int next = stage + STAGES - 1u;
        if (next < stages) issue(next, next % STAGES);
        sensenova_commit();
        const unsigned int buf = stage % STAGES;
        uint32_t af[MT][4], bf[NT][2], sfa[MT], sfb[NT];
        const uint8_t* a_tile = s_a(buf);
        const uint8_t* w_tile = s_w(buf);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const unsigned int row = wm * (BM / WM) + mt * 16u + gr;
            const uint2 lo = *reinterpret_cast<const uint2*>(a_tile + sensenova_pair_off(row, chunk));
            const uint2 hi =
                *reinterpret_cast<const uint2*>(a_tile + sensenova_pair_off(row + 8u, chunk));
            af[mt][0] = lo.x;
            af[mt][1] = hi.x;
            af[mt][2] = lo.y;
            af[mt][3] = hi.y;
            sfa[mt] = scale_a_lane
                          ? reinterpret_cast<const uint32_t*>(s_as(buf))[wm * (BM / WM) + mt * 16u +
                                                                        gr + 8u * (lane & 1u)]
                          : 0u;
        }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const unsigned int col = wn * (BN / WN) + nt * 8u + gr;
            const uint2 v = *reinterpret_cast<const uint2*>(w_tile + sensenova_pair_off(col, chunk));
            bf[nt][0] = v.x;
            bf[nt][1] = v.y;
            sfb[nt] = scale_b_lane
                          ? reinterpret_cast<const uint32_t*>(s_ws(buf))[wn * (BN / WN) + nt * 8u + gr]
                          : 0u;
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                sensenova_mma_fp4(acc[mt][nt], af[mt], bf[nt][0], bf[nt][1], sfa[mt], sfb[nt]);
    }

    const unsigned int q = lane & 3u;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const unsigned int row = m0 + wm * (BM / WM) + mt * 16u + gr + 8u * half;
            if (row >= tokens) continue;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const unsigned long long base =
                    (unsigned long long)row * cols + n0 + wn * (BN / WN) + nt * 8u + 2u * q;
                *reinterpret_cast<float2*>(out + base) =
                    make_float2(acc[mt][nt][2 * half], acc[mt][nt][2 * half + 1]);
            }
        }
    }
}

// A 128-row x 256-column block, eight warps of 64 x 64: the shape a 12,288-wide projection
// wants. 55,296 B of shared memory (STAGES = 4).
extern "C" __global__ void __launch_bounds__(256) sensenova_dense_fp4_wide(
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws, const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ as, float* __restrict__ out, unsigned int cols,
    unsigned int inner, unsigned int tokens) {
    sensenova_dense_fp4_body<128, 256, 4, 2, 4>(w, ws, a, as, out, cols, inner, tokens);
}

// A 256-row x 128-column block, eight warps of 64 x 64: the tall form. Its 256 rows halve the
// column tiles a chunk reads the weights for, and it reaches 300 TFLOPS on a 4,096-wide
// projection where the 128-row tile's 32 columns stop at 262.
extern "C" __global__ void __launch_bounds__(256) sensenova_dense_fp4_tall(
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws, const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ as, float* __restrict__ out, unsigned int cols,
    unsigned int inner, unsigned int tokens) {
    sensenova_dense_fp4_body<256, 128, 6, 4, 2>(w, ws, a, as, out, cols, inner, tokens);
}

// A 128-row x 128-column block, eight warps of 64 x 32: the narrow form, whose 32 column tiles
// fill the card where the wide one's 16 would not.
extern "C" __global__ void __launch_bounds__(256) sensenova_dense_fp4_narrow(
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws, const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ as, float* __restrict__ out, unsigned int cols,
    unsigned int inner, unsigned int tokens) {
    sensenova_dense_fp4_body<128, 128, 6, 2, 4>(w, ws, a, as, out, cols, inner, tokens);
}
#endif


// ---------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------
// The Gated DeltaNet recurrence over a prefill chunk.
//
// The first form gave a head one block of 128 threads, one a value column, and walked the rows
// one at a time: each column's chain is 128 dependent shared-memory reads for the recall and 128
// more for the update, so a block spent thousands of cycles a row and 42 visits of a 512-row
// chunk cost 120 ms, a fifth of the whole prefill.
//
// Here a block owns 32 of a head's 128 value columns as four warps, and each column's key
// dimension is split four ways inside the warp: lane (r, c) holds keys r, r + 4, ... of column
// c, so the recall is 32 reads and the update 32 more, and the two cross-lane sums are warp
// shuffles. The state slice lives in shared memory for the whole chunk, read and written once.
// The row's q, k, beta and decay do not depend on the state, so a separate pass computes them
// for every row ahead of the scan.

// The per-row scalars the scan reads: q's and k's RMS scales, beta, and the decay factor.
extern "C" __global__ void sensenova_gdn_params(const float* __restrict__ mixed,
                                                const float* __restrict__ b, int b_stride,
                                                const float* __restrict__ a, int a_stride,
                                                const __nv_bfloat16* __restrict__ a_log,
                                                const __nv_bfloat16* __restrict__ dt_bias,
                                                float* __restrict__ q_scale,
                                                float* __restrict__ k_scale,
                                                float* __restrict__ beta,
                                                float* __restrict__ factor, int key_width,
                                                int heads, int key_dim, int value_dim,
                                                int groups, int rows) {
    const int head = blockIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const int key_head = head / groups;
    const int conv_width = key_width * 2 + heads * value_dim;
    const int lane = threadIdx.x;
    const float* row_ptr = mixed + (unsigned long long)row * conv_width;
    const float* q = row_ptr + key_head * key_dim;
    const float* k = row_ptr + key_width + key_head * key_dim;
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (int key = lane; key < key_dim; key += 32) {
        q_sum += q[key] * q[key];
        k_sum += k[key] * k[key];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        q_sum += __shfl_xor_sync(0xFFFFFFFFu, q_sum, offset);
        k_sum += __shfl_xor_sync(0xFFFFFFFFu, k_sum, offset);
    }
    if (lane == 0) {
        const unsigned long long at = (unsigned long long)row * heads + head;
        q_scale[at] = rsqrtf(q_sum + 1e-6f) * rsqrtf((float)key_dim);
        k_scale[at] = rsqrtf(k_sum + 1e-6f);
        beta[at] = 1.0f / (1.0f + expf(-b[(unsigned long long)row * b_stride + head]));
        const float decay =
            -expf(__bfloat162float(a_log[head])) *
            logf(1.0f + expf(a[(unsigned long long)row * a_stride + head] +
                             __bfloat162float(dt_bias[head])));
        factor[at] = expf(decay);
    }
}

// A block's 32 value columns of one head's state: lane (r, c) owns the four keys
// 16 j + 4 r of column c and keeps them in registers for the whole chunk, so a row costs no
// state traffic at all - only the row's k and q, read out of shared memory. The state is loaded
// from global once and written back once.
#define SENSENOVA_SCAN_ROWS 12
#define SENSENOVA_SCAN_COLUMNS 32

extern "C" __global__ void sensenova_gdn_scan(
    const float* __restrict__ mixed, const float* __restrict__ q_scale,
    const float* __restrict__ k_scale, const float* __restrict__ beta,
    const float* __restrict__ factor, const float* __restrict__ state, float* __restrict__ state_out,
    float* __restrict__ out, int key_width, int heads, int key_dim, int value_dim, int groups,
    int rows) {
    __shared__ float sh_q[SENSENOVA_SCAN_ROWS][128];
    __shared__ float sh_k[SENSENOVA_SCAN_ROWS][128];
    __shared__ float sh_v[SENSENOVA_SCAN_ROWS][SENSENOVA_SCAN_COLUMNS];
    __shared__ float sh_scale[4][SENSENOVA_SCAN_ROWS];
    const int head = blockIdx.x;
    const int part = blockIdx.y;
    const int key_head = head / groups;
    const int conv_width = key_width * 2 + heads * value_dim;
    const int first = part * SENSENOVA_SCAN_COLUMNS;
    const int lane = threadIdx.x & 31u;
    const int warp = threadIdx.x >> 5u;
    const int group = lane & 3u;    // the four-key run this lane holds of every 16
    const int column = lane >> 2u;  // 0..7 within the warp
    const int mine = warp * 8 + column;
    if (value_dim % SENSENOVA_SCAN_COLUMNS != 0 || key_dim != 128) return;
    const unsigned long long base = (unsigned long long)head * key_dim * value_dim;
    const int column_index = first + mine;

    // The lane's 32 state values: keys 16 j + 4 group + i.
    float held[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            held[j][i] =
                state[base + (unsigned long long)(16 * j + 4 * group + i) * value_dim + column_index];
        }
    }

    for (int t0 = 0; t0 < rows; t0 += SENSENOVA_SCAN_ROWS) {
        const int slice = min(SENSENOVA_SCAN_ROWS, rows - t0);
        for (int t = warp; t < slice; t += blockDim.x >> 5) {
            const float* row_ptr = mixed + (unsigned long long)(t0 + t) * conv_width;
#pragma unroll
            for (int key = lane; key < 128; key += 32) {
                sh_q[t][key] = row_ptr[key_head * 128 + key];
                sh_k[t][key] = row_ptr[key_width + key_head * 128 + key];
            }
        }
        for (int index = threadIdx.x; index < slice * SENSENOVA_SCAN_COLUMNS; index += blockDim.x) {
            const int t = index / SENSENOVA_SCAN_COLUMNS;
            const int col = index - t * SENSENOVA_SCAN_COLUMNS;
            sh_v[t][col] = mixed[(unsigned long long)(t0 + t) * conv_width + key_width * 2 +
                                 head * value_dim + first + col];
        }
        for (int index = threadIdx.x; index < slice; index += blockDim.x) {
            const unsigned long long at = (unsigned long long)(t0 + index) * heads + head;
            sh_scale[0][index] = q_scale[at];
            sh_scale[1][index] = k_scale[at];
            sh_scale[2][index] = beta[at];
            sh_scale[3][index] = factor[at];
        }
        __syncthreads();
        for (int t = 0; t < slice; ++t) {
            const float qq = sh_scale[0][t];
            const float kk = sh_scale[1][t];
            const float scale_beta = sh_scale[2][t];
            const float decay = sh_scale[3][t];
            const float* k_row = sh_k[t];
            const float* q_row = sh_q[t];
            float recall0 = 0.0f, recall1 = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float4 k4 = *reinterpret_cast<const float4*>(k_row + 16 * j + 4 * group);
                recall0 += held[j][0] * k4.x + held[j][1] * k4.y;
                recall1 += held[j][2] * k4.z + held[j][3] * k4.w;
            }
            float recall = ((recall0 + recall1) * decay);
            recall = __shfl_xor_sync(0xFFFFFFFFu, recall, 1);
            recall = __shfl_xor_sync(0xFFFFFFFFu, recall, 2);
            const float delta = (sh_v[t][mine] - recall * kk) * scale_beta;
            const float kd = kk * delta;
            float out0 = 0.0f, out1 = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float4 k4 = *reinterpret_cast<const float4*>(k_row + 16 * j + 4 * group);
                const float4 q4 = *reinterpret_cast<const float4*>(q_row + 16 * j + 4 * group);
                held[j][0] = held[j][0] * decay + k4.x * kd;
                held[j][1] = held[j][1] * decay + k4.y * kd;
                held[j][2] = held[j][2] * decay + k4.z * kd;
                held[j][3] = held[j][3] * decay + k4.w * kd;
                out0 += held[j][0] * q4.x + held[j][1] * q4.y;
                out1 += held[j][2] * q4.z + held[j][3] * q4.w;
            }
            float output = (out0 + out1) * qq;
            output = __shfl_xor_sync(0xFFFFFFFFu, output, 1);
            output = __shfl_xor_sync(0xFFFFFFFFu, output, 2);
            if (group == 0) {
                out[(unsigned long long)(t0 + t) * heads * value_dim + head * value_dim +
                    column_index] = output;
            }
        }
        if (t0 + slice < rows) __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            state_out[base + (unsigned long long)(16 * j + 4 * group + i) * value_dim +
                      column_index] = held[j][i];
        }
    }
}
// The head's gated RMSNorm of its raw outputs, in place: one block a (row, head), a warp a
// reduction level.
extern "C" __global__ void sensenova_gdn_gate(float* __restrict__ out, const float* __restrict__ z,
                                               const __nv_bfloat16* __restrict__ norm,
                                               int heads, int value_dim, float eps,
                                               int sigmoid_gate, int rows) {
    __shared__ float partials[32];
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= rows) return;
    const unsigned long long at = (unsigned long long)row * heads * value_dim +
                                  (unsigned long long)head * value_dim;
    float* values = out + at;
    const float* gate = z + at;
    float sum = 0.0f;
    for (int index = threadIdx.x; index < value_dim; index += blockDim.x) {
        sum += values[index] * values[index];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, offset);
    if ((threadIdx.x & 31u) == 0u) partials[threadIdx.x >> 5u] = sum;
    __syncthreads();
    const int warps = (blockDim.x + 31u) >> 5u;
    float total = 0.0f;
    for (int index = 0; index < warps; ++index) total += partials[index];
    const float scale = rsqrtf(total / (float)value_dim + eps);
    for (int index = threadIdx.x; index < value_dim; index += blockDim.x) {
        const float activated = sigmoid_gate
                                    ? (1.0f / (1.0f + expf(-gate[index])))
                                    : (gate[index] / (1.0f + expf(-gate[index])));
        values[index] = values[index] * scale * __bfloat162float(norm[index]) * activated;
    }
}


// Copies NVFP4 rows from one matrix into another's row range: how a layer's z, b and a become the
// single 4,224-column projection the prefill issues. Zeroed first, so the pad rows read as zeros.
extern "C" __global__ void sensenova_fp4_append(const uint8_t* __restrict__ src_codes,
                                                 const uint8_t* __restrict__ src_scales,
                                                 uint8_t* __restrict__ dst_codes,
                                                 uint8_t* __restrict__ dst_scales, int rows,
                                                 int inner, int dst_offset) {
    const int packed = inner >> 1;
    const int blocks = inner >> 4;
    const int total = rows * blocks;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += gridDim.x * blockDim.x) {
        const int row = index / blocks;
        const int block = index - row * blocks;
        const uint64_t value = *reinterpret_cast<const uint64_t*>(src_codes +
                                                                  ((unsigned long long)row * packed +
                                                                   block * 8u));
        *reinterpret_cast<uint64_t*>(dst_codes + ((unsigned long long)(dst_offset + row) * packed +
                                                  block * 8u)) = value;
        dst_scales[(unsigned long long)(dst_offset + row) * blocks + block] =
            src_scales[(unsigned long long)row * blocks + block];
    }
}


// The Gated DeltaNet convolution over a chunk: one thread a channel, its taps in registers, and
// the rows split over blocks - row t reads raw[t-1..t-3], so every block after the first starts
// from the raw stream rather than the serial chain the old kernel walked. Only the last row block
// writes the convolution state back.
extern "C" __global__ void sensenova_gdn_conv(const float* __restrict__ raw,
                                               float* __restrict__ state,
                                               const __nv_bfloat16* __restrict__ weight,
                                               float* __restrict__ out, int width, int taps,
                                               int rows, int row_block) {
    const int history = taps - 1;
    if (history < 0 || history > 7) return;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= width) return;
    const int first = blockIdx.y * row_block;
    if (first >= rows) return;
    const int last = min(rows, first + row_block);
    float held[8];
    for (int back = 1; back <= history; ++back) {
        const int at = first - back;
        held[back - 1] = at >= 0 ? raw[(unsigned long long)at * width + column]
                                 : state[(history - back) * width + column];
    }
    const __nv_bfloat16* channel = weight + (unsigned long long)column * taps;
    float w[8];
    for (int tap = 0; tap < taps; ++tap) w[tap] = __bfloat162float(channel[tap]);
    for (int t = first; t < last; ++t) {
        const float src = raw[(unsigned long long)t * width + column];
        // held[history-1] is the previous row, held[0] the oldest.
        float accumulator = 0.0f;
        for (int tap = 0; tap < taps; ++tap) {
            const int back = history - tap;
            const float source = back == 0 ? src : held[history - back];
            accumulator += w[tap] * source;
        }
        out[(unsigned long long)t * width + column] = accumulator / (1.0f + expf(-accumulator));
        for (int index = 0; index + 1 < history; ++index) held[index] = held[index + 1];
        if (history > 0) held[history - 1] = src;
    }
    if (last == rows) {
        for (int back = 1; back <= history; ++back) {
            state[(history - back) * width + column] = held[history - back];
        }
    }
}


// ---------------------------------------------------------------------------------------------
// Full attention over a prefill chunk.
//
// sensenova_attn_full_q8 gives one block to a (head, row, split), so the 8-bit K and V are
// decoded once per query row and 131k blocks run for a 512-row chunk: 17 ms of a 100 ms chunk.
// Here a block is a head's eight query rows at once, a warp a row, and the K/V tile a 64-key
// slice staged once in shared for all eight: the decode is amortised eight ways and the blocks
// fall to a few thousand. The softmax is online against a running maximum, as the split form's is.
#define SENSENOVA_PREFILL_ROWS 8
#define SENSENOVA_PREFILL_TILE 64

extern "C" __global__ void sensenova_attn_prefill(
    const float* __restrict__ q, const float* __restrict__ gate,
    const unsigned char* __restrict__ keys, const unsigned char* __restrict__ values,
    const float* __restrict__ k_scales, const float* __restrict__ v_scales,
    float* __restrict__ out, const int* __restrict__ position, int heads, int kv_heads,
    int head_dim, int groups, int rows) {
    __shared__ unsigned char sh_k[SENSENOVA_PREFILL_TILE][256];
    __shared__ unsigned char sh_v[SENSENOVA_PREFILL_TILE][256];
    __shared__ float sh_ks[SENSENOVA_PREFILL_TILE];
    __shared__ float sh_vs[SENSENOVA_PREFILL_TILE];
    __shared__ float sh_q[SENSENOVA_PREFILL_ROWS][256];
    __shared__ float sh_logits[SENSENOVA_PREFILL_ROWS][SENSENOVA_PREFILL_TILE];
    const int head = blockIdx.x;
    const int first_row = blockIdx.y * SENSENOVA_PREFILL_ROWS;
    const int lane = threadIdx.x & 31u;
    const int warp = threadIdx.x >> 5u;
    if (head_dim != 256 || warp >= SENSENOVA_PREFILL_ROWS) return;
    const int kv_head = head / (groups > 0 ? groups : 1);
    const int row = first_row + warp;
    const bool active = row < rows;
    const int last_row = first_row + min(SENSENOVA_PREFILL_ROWS, rows - first_row);
    const int limit = *position + last_row;   // the last key any row of the group sees
    const int kv_row = kv_heads * 256;
    const float scale = 1.0f / sqrtf(256.0f);
    const int dims = lane * 8;   // this lane's eight head dimensions

    for (int index = threadIdx.x; index < SENSENOVA_PREFILL_ROWS * 256; index += blockDim.x) {
        const int r = index / 256;
        const int d = index - r * 256;
        const int at = first_row + r;
        sh_q[r][d] = at < rows ? q[((unsigned long long)at * heads + head) * 256 + d] : 0.0f;
    }
    __syncthreads();

    float held[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) held[i] = active ? sh_q[warp][dims + i] : 0.0f;
    float accumulator[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) accumulator[i] = 0.0f;
    float maximum = -INFINITY;
    float sum = 0.0f;
    const int visible = active ? *position + row + 1 : 0;

    for (int base = 0; base < limit; base += SENSENOVA_PREFILL_TILE) {
        const int count = min(SENSENOVA_PREFILL_TILE, limit - base);
        __syncthreads();
        for (int index = threadIdx.x; index < count * 256; index += blockDim.x) {
            const int slot = index / 256;
            const int d = index - slot * 256;
            const unsigned long long src = (unsigned long long)(base + slot) * kv_row + kv_head * 256 + d;
            sh_k[slot][d] = keys[src];
            sh_v[slot][d] = values[src];
        }
        if (threadIdx.x < count) {
            sh_ks[threadIdx.x] = k_scales[(unsigned long long)(base + threadIdx.x) * kv_heads + kv_head];
            sh_vs[threadIdx.x] = v_scales[(unsigned long long)(base + threadIdx.x) * kv_heads + kv_head];
        }
        __syncthreads();
        if (!active) continue;
        // This row's logits for the tile: a lane takes the slots it can see and reduces the
        // eight-dimension product across the warp, so every lane then holds each slot's sum.
        float tile_max = -INFINITY;
        for (int slot = lane; slot < count; slot += 32) {
            const int at = base + slot;
            float dot = 0.0f;
            if (at < visible) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    dot += held[i] * sensenova_e4m3(sh_k[slot][dims + i]);
                }
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    dot += __shfl_xor_sync(0xFFFFFFFFu, dot, offset);
                }
                dot *= sh_ks[slot] * scale;
            } else {
                dot = -INFINITY;
            }
            sh_logits[warp][slot] = dot;
            tile_max = fmaxf(tile_max, dot);
        }
        __syncwarp();
        // The tile's logits are all in shared: fold them into the running maximum and sum.
        float slot_max = -INFINITY;
        for (int slot = 0; slot < count; ++slot) slot_max = fmaxf(slot_max, sh_logits[warp][slot]);
        const float next = fmaxf(maximum, slot_max);
        const float rescale = __expf(maximum - next);
        float tile_sum = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const float weight = __expf(sh_logits[warp][slot] - next);
            sh_logits[warp][slot] = weight;
            tile_sum += weight;
        }
        sum = sum * rescale + tile_sum;
        maximum = next;
#pragma unroll
        for (int i = 0; i < 8; ++i) accumulator[i] *= rescale;
        for (int slot = 0; slot < count; ++slot) {
            const float weight = sh_logits[warp][slot] * sh_vs[slot];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accumulator[i] += weight * sensenova_e4m3(sh_v[slot][dims + i]);
            }
        }
        __syncwarp();
    }
    if (!active) return;
    const unsigned long long at = ((unsigned long long)row * heads + head) * 256;
    const float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int i = 0; i < 8; ++i) {
        const float value = accumulator[i] * inverse;
        const float g = gate[at + dims + i];
        out[at + dims + i] = value / (1.0f + __expf(-g));
    }
}


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

// Qwen3.8-Flash-Next (Qwen4-Exp) linear-attention scan kernels, TP=1.
//
// The Gated DeltaNet and Kimi Delta Attention recurrences, split out of
// kernels/qwen4_prefill_sm121.cu into their own image so a model arm that only needs the scan
// loads only the scan:
//
//   - q4p_gdn_conv / q4p_gdn_conv_state: the causal depthwise conv, parallel over the chunk's
//     tokens instead of stepping through them.
//   - q4p_gdn_prep / q4p_gdn_scan / q4p_gdn_gate: the Gated DeltaNet recurrence with the state in
//     registers, so a token step is two short FMA chains and two shuffle reductions instead of
//     three block-wide reductions.
//   - q4p_kda_prep / q4p_kda_scan: the same recurrence with Kimi Delta Attention's decay, one
//     factor a key channel (GLM-5.3-Flash), from the token cubin's q4e_kda_decay.
//
// This translation unit compiles on its own: it copies q4p_silu from the prefill image and
// includes the shared cp.async helpers and OPT_PDL_ENTRY instead of including qwen4_prefill.
//
// Attribution:
//   - Model architecture (Qwen4ExpForConditionalGeneration: Gated DeltaNet linear attention)
//     originates from Qwen/Qwen3.8-Flash-Next by the Qwen team at Alibaba Cloud and the vLLM
//     reference implementation (https://github.com/vllm-project/vllm, Apache-2.0). The Gated
//     DeltaNet recurrence is that of Yang, Kautz and Hatamizadeh, "Gated Delta Networks:
//     Improving Mamba2 with Delta Rule" (ICLR 2025), computed here as a sequential scan in the
//     same order as this crate's host oracle (src/forward.rs).
//   - Kimi Delta Attention's per-channel decay (q4p_kda_*) is the method of Kimi Team et al.,
//     "Kimi Linear: An Expressive, Efficient Attention Architecture" (arXiv:2510.26692), as Z.ai's
//     GLM-5.3-Flash release (MIT License) uses it; it is checked against the shared host delta
//     rule in crates/sparkflash-weights/src/linear_attention.rs.
//   - SM120/SM121 (DGX Spark GB10) cp.async tiling patterns draw on prior art from b12x
//     (https://github.com/local-inference-lab/b12x, flashinfer/experimental/b12x, Apache-2.0;
//     Luke Alonso @lukealonso, @original-el8, and contributors) and NVIDIA Flare
//     (https://github.com/NVIDIA/flare-local-inference, Apache-2.0), through the shared
//     kernels/opt_mma_*_sm121.cuh headers. See the root NOTICE file for full provenance details.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// om_smem, om_cp16, om_commit and om_wait: the cp.async stage copy the scan pipelines with.
#include "opt_mma_common_sm121.cuh"
// Every q4p kernel opens with OPT_PDL_ENTRY: the host launches them with programmatic stream
// serialization, as it does the token cubin's.
#include "opt_pdl_sm121.cuh"
#include "q4e_common_sm121.cuh"

// The shared header's gate, under the name this image used.
#define q4p_silu q4e_silu


// ---------------------------------------------------------------------------------------------
// Gated DeltaNet.
// ---------------------------------------------------------------------------------------------

#define Q4P_GDN_KEY 128
// Tokens staged per pipeline step. Eight keeps a head's block (both buffers) at 26 KiB, so all
// forty-eight heads are resident at once - the scan is latency-bound, and a second wave would
// double it.
#define Q4P_GDN_TILE 8
// Floats per staged token: q then k, each 128 keys with four floats of padding after every 32,
// so the four key quarters a warp reads at one offset sit on different banks.
#define Q4P_GDN_HALF 144
#define Q4P_GDN_ROW (2 * Q4P_GDN_HALF)
// The value head width the scan is compiled for (the released configuration's): a compile-time
// stride keeps the 32 state cells a lane owns at immediate offsets from one base pointer, where a
// runtime stride had ptxas hold 32 hoisted 64-bit addresses and spill.
#define Q4P_GDN_VALUE 128

__device__ __forceinline__ int q4p_gdn_pad(int key) {
    return key + (key >> 5) * 4;
}

// The causal depthwise conv history a chunk carries (taps - 1 rows); the released configuration
// has four taps.
#define Q4P_CONV_HISTORY 8

// The conv's raw input row j of the chunk, or for j < 0 the carried history row history + j.
__device__ __forceinline__ float q4p_conv_input(const float* __restrict__ raw,
                                                const float* __restrict__ state, int j,
                                                int history, int width, int column) {
    return j >= 0 ? raw[(unsigned long long)j * width + column]
                  : state[(unsigned long long)(history + j) * width + column];
}

// The causal depthwise conv1d with SiLU over a whole chunk, one thread per (column, token): row
// t reads raw rows t - history .. t, and rows before the chunk from the carried history. The
// token kernel q4e_gdn_conv walks the chunk one token at a time per column and shifts the history
// through global memory at every step; this one has no step, and computes each output with the
// same products in the same order. Grid (width / 256, tokens), 256 threads; q4p_gdn_conv_state
// advances the history afterwards.
extern "C" __global__ void q4p_gdn_conv(const float* __restrict__ raw,
                                        const float* __restrict__ state,
                                        const __nv_bfloat16* __restrict__ weight,
                                        float* __restrict__ out, int width, int taps) {
    OPT_PDL_ENTRY();
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (column >= width) {
        return;
    }
    const int history = taps - 1;
    float accumulator = 0.0f;
    for (int tap = 0; tap < taps; ++tap) {
        const float source = q4p_conv_input(raw, state, t - (history - tap), history, width, column);
        accumulator += __bfloat162float(weight[column * taps + tap]) * source;
    }
    out[(unsigned long long)t * width + column] = q4p_silu(accumulator);
}

// Advance the conv history past a chunk of `batch` rows: row r becomes input row
// batch - history + r. When snapshot is non-null, region t of it (region_stride floats apart)
// receives the history as it stood after row t, for t < snapshot_rows, as q4e_gdn_conv writes it. Grid
// (width / 256), 256 threads; taps - 1 is at most Q4P_CONV_HISTORY (the host checks).
extern "C" __global__ void q4p_gdn_conv_state(const float* __restrict__ raw,
                                              float* __restrict__ state, int width, int taps,
                                              int batch, float* __restrict__ snapshot,
                                              int snapshot_rows, long long region_stride) {
    OPT_PDL_ENTRY();
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int history = taps - 1;
    if (column >= width || history <= 0 || history > Q4P_CONV_HISTORY) {
        return;
    }
    float carried[Q4P_CONV_HISTORY];
#pragma unroll
    for (int r = 0; r < Q4P_CONV_HISTORY; ++r) {
        if (r < history) {
            carried[r] = state[(unsigned long long)r * width + column];
        }
    }
    // Input row j, where the carried rows are j = -history .. -1.
    auto input = [&](int j) {
        float value = 0.0f;
#pragma unroll
        for (int r = 0; r < Q4P_CONV_HISTORY; ++r) {
            if (r < history && j == r - history) {
                value = carried[r];
            }
        }
        return j >= 0 ? raw[(unsigned long long)j * width + column] : value;
    };
    if (snapshot != nullptr) {
        for (int at = 0; at < snapshot_rows && at < batch; ++at) {
            float* into = snapshot + (unsigned long long)at * region_stride;
            for (int r = 0; r < history; ++r) {
                into[(unsigned long long)r * width + column] = input(at - history + 1 + r);
            }
        }
    }
    for (int r = 0; r < history; ++r) {
        state[(unsigned long long)r * width + column] = input(batch - history + r);
    }
}

// Per (token, key head): the L2-normalised query (also scaled by 1/sqrt(key_dim)) and key, laid
// out as the scan stages them; per (token, value head of the group): the decay factor and beta.
// With kChannel (Kimi Delta Attention) the decay is one factor a key channel, which the scan
// reads from q4e_kda_decay's buffer, so the head's factor is written as 1 and a, a_log and
// dt_bias are not read.
template <bool kChannel>
__device__ __forceinline__ void q4p_gdn_prep_body(
    const float* __restrict__ mixed, const float* __restrict__ a, const float* __restrict__ b,
    const __nv_bfloat16* __restrict__ a_log, const __nv_bfloat16* __restrict__ dt_bias,
    float* __restrict__ qk, float2* __restrict__ decay_beta, int key_width, int key_heads,
    int heads, int value_dim, int groups) {
    __shared__ float reduce[2][4];
    const int kh = blockIdx.x;
    const int token = blockIdx.y;
    const int key = threadIdx.x;
    const int conv_width = key_width * 2 + heads * value_dim;
    const float* row = mixed + (unsigned long long)token * conv_width;
    const float q = row[kh * Q4P_GDN_KEY + key];
    const float k = row[key_width + kh * Q4P_GDN_KEY + key];
    float q_sum = q * q;
    float k_sum = k * k;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        q_sum += __shfl_xor_sync(0xffffffffu, q_sum, offset);
        k_sum += __shfl_xor_sync(0xffffffffu, k_sum, offset);
    }
    if ((key & 31) == 0) {
        reduce[0][key >> 5] = q_sum;
        reduce[1][key >> 5] = k_sum;
    }
    __syncthreads();
    q_sum = (reduce[0][0] + reduce[0][1]) + (reduce[0][2] + reduce[0][3]);
    k_sum = (reduce[1][0] + reduce[1][1]) + (reduce[1][2] + reduce[1][3]);
    const float q_norm = rsqrtf(q_sum + 1e-6f);
    const float k_norm = rsqrtf(k_sum + 1e-6f);
    float* out = qk + ((unsigned long long)token * key_heads + kh) * Q4P_GDN_ROW;
    out[q4p_gdn_pad(key)] = q * q_norm * rsqrtf((float)Q4P_GDN_KEY);
    out[Q4P_GDN_HALF + q4p_gdn_pad(key)] = k * k_norm;
    if (key < 16) {
        // The padding is never read; zero it so the staged rows are fully defined.
        out[(key >> 2) * 36 + 32 + (key & 3)] = 0.0f;
        out[Q4P_GDN_HALF + (key >> 2) * 36 + 32 + (key & 3)] = 0.0f;
    }
    if (key < groups) {
        const int head = kh * groups + key;
        float factor = 1.0f;
        if constexpr (!kChannel) {
            const float decay =
                -__expf(__bfloat162float(a_log[head])) *
                logf(1.0f + __expf(a[(unsigned long long)token * heads + head] +
                                   __bfloat162float(dt_bias[head])));
            factor = __expf(decay);
        }
        const float beta = 1.0f / (1.0f + __expf(-b[(unsigned long long)token * heads + head]));
        decay_beta[(unsigned long long)token * heads + head] = make_float2(factor, beta);
    }
}

// Gated DeltaNet's prep: grid (key_heads, tokens), 128 threads (one per key).
extern "C" __global__ __launch_bounds__(128) void q4p_gdn_prep(
    const float* __restrict__ mixed, const float* __restrict__ a, const float* __restrict__ b,
    const __nv_bfloat16* __restrict__ a_log, const __nv_bfloat16* __restrict__ dt_bias,
    float* __restrict__ qk, float2* __restrict__ decay_beta, int key_width, int key_heads,
    int heads, int value_dim, int groups) {
    OPT_PDL_ENTRY();
    q4p_gdn_prep_body<false>(mixed, a, b, a_log, dt_bias, qk, decay_beta, key_width, key_heads,
                             heads, value_dim, groups);
}

// Kimi Delta Attention's prep, for q4p_kda_scan: the same grid, beta only beside the q and k.
extern "C" __global__ __launch_bounds__(128) void q4p_kda_prep(
    const float* __restrict__ mixed, const float* __restrict__ b, float* __restrict__ qk,
    float2* __restrict__ decay_beta, int key_width, int key_heads, int heads, int value_dim,
    int groups) {
    OPT_PDL_ENTRY();
    q4p_gdn_prep_body<true>(mixed, nullptr, b, nullptr, nullptr, qk, decay_beta, key_width,
                            key_heads, heads, value_dim, groups);
}

// The recurrence, state in registers. Grid (heads, splits), 4 * Q4P_GDN_VALUE / splits threads (a
// multiple of 32): blockIdx.y owns 1 / splits of the head's value columns, warp w of it columns
// 8w..8w+7 of those, and the four lanes of a column split the 128 keys into quarters of 32, so a
// column's dot products are 32-term chains plus two shuffles. A value column's recurrence reads no
// other column, so the split changes no bit; it gives the SMs work when a rank holds few heads.
// The state is read once from `state` and written once per chunk to `state_out`, which may be
// `state` itself; a speculative verify leaves the state it read as it found it and its last row's
// in another buffer. `out` gets the raw (pre-norm) head output [token][heads * value_dim].
//
// The arithmetic is the host oracle's per element: S *= decay; memory = S . k; delta =
// (v - memory) * beta; S += k * delta; out = S . q. When snapshot is non-null, region t of it
// (region_stride floats apart) receives the state after row t, for t < snapshot_rows, as
// q4e_gdn_recur writes it.
//
// With kChannel (Kimi Delta Attention) S decays by one factor a key channel instead: `channel`
// is [batch][heads][Q4P_GDN_KEY], q4e_kda_decay's factors, staged with the keys' padding so a
// lane's quarter reads them as it reads k.
template <bool kChannel>
__device__ __forceinline__ void q4p_gdn_scan_body(
    const float* __restrict__ qk, const float2* __restrict__ decay_beta,
    const float* __restrict__ channel, const float* __restrict__ mixed, const float* state,
    float* state_out, float* __restrict__ out, int key_width, int key_heads, int heads,
    int value_dim, int groups, int batch, float* __restrict__ snapshot, int snapshot_rows,
    long long region_stride) {
    __shared__ __align__(16) float s_qk[2][Q4P_GDN_TILE * Q4P_GDN_ROW];
    __shared__ __align__(16) float s_v[2][Q4P_GDN_TILE * Q4P_GDN_VALUE];
    __shared__ __align__(16) float2 s_db[2][Q4P_GDN_TILE];
    __shared__ __align__(16) float s_cd[2][kChannel ? Q4P_GDN_TILE * Q4P_GDN_HALF : 4];
    if (value_dim != Q4P_GDN_VALUE || (int)(blockDim.x * gridDim.y) != 4 * Q4P_GDN_VALUE ||
        (blockDim.x & 31u) != 0u) {
        return;
    }
    const int head = blockIdx.x;
    const int kh = head / groups;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    // This block's value columns start here; there are blockDim.x / 4 of them.
    const int first_column = blockIdx.y * (blockDim.x >> 2);
    const int column = first_column + (tid >> 5) * 8 + (lane >> 2);
    const int quarter = lane & 3;
    const int conv_width = key_width * 2 + heads * Q4P_GDN_VALUE;
    const int val_width = heads * Q4P_GDN_VALUE;
    const unsigned long long cells = (unsigned long long)Q4P_GDN_KEY * Q4P_GDN_VALUE;
    // This lane's 32 cells: keys quarter*32 + i of `column`, Q4P_GDN_VALUE floats apart.
    const unsigned long long lane_cell = (unsigned long long)quarter * 32 * Q4P_GDN_VALUE + column;
    const float* recurrent = state + (unsigned long long)head * cells + lane_cell;

    float s[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        s[i] = recurrent[i * Q4P_GDN_VALUE];
    }

    // This block's value columns in 16-byte chunks.
    const int v_chunks = blockDim.x >> 4;
    auto stage = [&](int buffer, int t0) {
        const int rows = min(Q4P_GDN_TILE, batch - t0);
        // q / k: Q4P_GDN_ROW floats a token, in 16-byte chunks.
        for (int c = tid; c < rows * (Q4P_GDN_ROW / 4); c += blockDim.x) {
            const int r = c / (Q4P_GDN_ROW / 4);
            const int within = c - r * (Q4P_GDN_ROW / 4);
            const float* src =
                qk + ((unsigned long long)(t0 + r) * key_heads + kh) * Q4P_GDN_ROW + within * 4;
            om_cp16(om_smem(&s_qk[buffer][r * Q4P_GDN_ROW + within * 4]), src, true);
        }
        for (int c = tid; c < rows * v_chunks; c += blockDim.x) {
            const int r = c / v_chunks;
            const int within = c - r * v_chunks;
            const float* src = mixed + (unsigned long long)(t0 + r) * conv_width + key_width * 2 +
                               head * Q4P_GDN_VALUE + first_column + within * 4;
            om_cp16(om_smem(&s_v[buffer][r * Q4P_GDN_VALUE + first_column + within * 4]), src,
                    true);
        }
        if (tid < rows) {
            s_db[buffer][tid] = decay_beta[(unsigned long long)(t0 + tid) * heads + head];
        }
        if constexpr (kChannel) {
            for (int c = tid; c < rows * (Q4P_GDN_KEY / 4); c += blockDim.x) {
                const int r = c / (Q4P_GDN_KEY / 4);
                const int within = c - r * (Q4P_GDN_KEY / 4);
                const float* src = channel +
                                   ((unsigned long long)(t0 + r) * heads + head) * Q4P_GDN_KEY +
                                   within * 4;
                om_cp16(om_smem(&s_cd[buffer][r * Q4P_GDN_HALF + q4p_gdn_pad(within * 4)]), src,
                        true);
            }
        }
    };

    stage(0, 0);
    om_commit();
    int buffer = 0;
    for (int t0 = 0; t0 < batch; t0 += Q4P_GDN_TILE) {
        if (t0 + Q4P_GDN_TILE < batch) {
            stage(buffer ^ 1, t0 + Q4P_GDN_TILE);
        }
        om_commit();
        om_wait<1>();
        __syncthreads();
        const int rows = min(Q4P_GDN_TILE, batch - t0);
        for (int r = 0; r < rows; ++r) {
            const float* q = &s_qk[buffer][r * Q4P_GDN_ROW + quarter * 36];
            const float* k = q + Q4P_GDN_HALF;
            const float2 db = s_db[buffer][r];
            const float v = s_v[buffer][r * Q4P_GDN_VALUE + column];
            float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int i = 0; i < 32; i += 4) {
                const float4 kk = *reinterpret_cast<const float4*>(&k[i]);
                if constexpr (kChannel) {
                    const float4 dd = *reinterpret_cast<const float4*>(
                        &s_cd[buffer][r * Q4P_GDN_HALF + quarter * 36 + i]);
                    s[i] *= dd.x;
                    s[i + 1] *= dd.y;
                    s[i + 2] *= dd.z;
                    s[i + 3] *= dd.w;
                } else {
                    s[i] *= db.x;
                    s[i + 1] *= db.x;
                    s[i + 2] *= db.x;
                    s[i + 3] *= db.x;
                }
                m[0] += s[i] * kk.x;
                m[1] += s[i + 1] * kk.y;
                m[2] += s[i + 2] * kk.z;
                m[3] += s[i + 3] * kk.w;
            }
            float memory = (m[0] + m[1]) + (m[2] + m[3]);
            memory += __shfl_xor_sync(0xffffffffu, memory, 1);
            memory += __shfl_xor_sync(0xffffffffu, memory, 2);
            const float delta = (v - memory) * db.y;
            float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int i = 0; i < 32; i += 4) {
                const float4 kk = *reinterpret_cast<const float4*>(&k[i]);
                const float4 qq = *reinterpret_cast<const float4*>(&q[i]);
                s[i] += kk.x * delta;
                s[i + 1] += kk.y * delta;
                s[i + 2] += kk.z * delta;
                s[i + 3] += kk.w * delta;
                o[0] += s[i] * qq.x;
                o[1] += s[i + 1] * qq.y;
                o[2] += s[i + 2] * qq.z;
                o[3] += s[i + 3] * qq.w;
            }
            float output = (o[0] + o[1]) + (o[2] + o[3]);
            output += __shfl_xor_sync(0xffffffffu, output, 1);
            output += __shfl_xor_sync(0xffffffffu, output, 2);
            const int token = t0 + r;
            if (quarter == 0) {
                out[(unsigned long long)token * val_width + head * Q4P_GDN_VALUE + column] = output;
            }
            if (snapshot != nullptr && token < snapshot_rows) {
                float* destination = snapshot + (unsigned long long)token * region_stride +
                                     (unsigned long long)head * cells + lane_cell;
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    destination[i * Q4P_GDN_VALUE] = s[i];
                }
            }
        }
        // No thread may still be reading this buffer when the stage after next lands in it.
        __syncthreads();
        buffer ^= 1;
    }
    float* recurrent_out = state_out + (unsigned long long)head * cells + lane_cell;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        recurrent_out[i * Q4P_GDN_VALUE] = s[i];
    }
}

// Gated DeltaNet's scan: one factor a head, from q4p_gdn_prep's decay_beta, from the state in
// `state` to the state in `state_out`.
extern "C" __global__ __launch_bounds__(512) void q4p_gdn_scan(
    const float* __restrict__ qk, const float2* __restrict__ decay_beta,
    const float* __restrict__ mixed, const float* state, float* state_out,
    float* __restrict__ out, int key_width, int key_heads, int heads, int value_dim, int groups,
    int batch, float* __restrict__ snapshot, int snapshot_rows, long long region_stride) {
    OPT_PDL_ENTRY();
    q4p_gdn_scan_body<false>(qk, decay_beta, nullptr, mixed, state, state_out, out, key_width,
                             key_heads, heads, value_dim, groups, batch, snapshot, snapshot_rows,
                             region_stride);
}

// Kimi Delta Attention's scan: one factor a key channel from `channel_decay`, beta from
// q4p_kda_prep's decay_beta. The same grid and threads as q4p_gdn_scan.
extern "C" __global__ __launch_bounds__(512) void q4p_kda_scan(
    const float* __restrict__ qk, const float2* __restrict__ decay_beta,
    const float* __restrict__ channel_decay, const float* __restrict__ mixed,
    float* __restrict__ state, float* __restrict__ out, int key_width, int key_heads, int heads,
    int value_dim, int groups, int batch, float* __restrict__ snapshot, int snapshot_rows,
    long long region_stride) {
    OPT_PDL_ENTRY();
    q4p_gdn_scan_body<true>(qk, decay_beta, channel_decay, mixed, state, state, out, key_width,
                            key_heads, heads, value_dim, groups, batch, snapshot, snapshot_rows,
                            region_stride);
}

// The gated RMSNorm on the scan's raw head outputs, as q4e_gdn_recur's tail computes it.
// Grid (heads, tokens), value_dim threads (a multiple of 32, at most 1024).
extern "C" __global__ void q4p_gdn_gate(const float* __restrict__ raw,
                                        const float* __restrict__ z,
                                        const __nv_bfloat16* __restrict__ norm,
                                        float* __restrict__ out, int heads, int value_dim,
                                        float eps, int sigmoid_gate) {
    OPT_PDL_ENTRY();
    __shared__ float reduce[32];
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int c = threadIdx.x;
    const unsigned long long at = ((unsigned long long)token * heads + head) * value_dim + c;
    const float value = raw[at];
    float sum = value * value;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    if ((c & 31) == 0) {
        reduce[c >> 5] = sum;
    }
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < (value_dim >> 5); ++w) {
        total += reduce[w];
    }
    const float out_scale = rsqrtf(total / (float)value_dim + eps);
    const float normed = value * out_scale * __bfloat162float(norm[c]);
    const float gated = z[at];
    const float activated = sigmoid_gate ? (1.0f / (1.0f + __expf(-gated))) : q4p_silu(gated);
    out[at] = normed * activated;
}

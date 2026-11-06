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

// GLM-5.3-Flash (Z.ai) decode/token kernels, TP=1.
//
// These kernels used to sit at the end of kernels/qwen4_exp_sm121.cu, which the GLM arm loaded
// with the Qwen token image. They now have their own translation unit and cubin so the Qwen token
// image carries no GLM decode kernel: the hyper-connection mixers (q4e_mhc_*), the clamped SwiGLU,
// the noaux_tc router, LayerNorm, the indexer's pool/score/pick, the absorbed MLA over the
// compressed latent, and the tiled decode GEMV wrappers GLM loads (q4e_gemv_tiled_bf16_parts,
// q4e_gemv_tiled_e4m3_block, q4e_expert_gemv_e4m3_block and q4e_prefetch_l2). The shared token
// kernels (embedding, argmax, BF16/FP8 GEMV, norm, scan, combine) stay in the Qwen image, which
// also keeps the tiled GEMV body Qwen instantiates; this file carries an inline copy of that body
// for its wrappers, so no external symbol is defined twice.
//
// Attribution:
//   - These kernels compute Z.ai's GLM-5.3-Flash release (MIT License) as its reference
//     implementation in Hugging Face Transformers (Apache-2.0) does, read for the arithmetic; no
//     source is copied. The hyper-connection mixers are mHC (Xie et al., arXiv:2512.24880) and
//     the indexer is DeepSeek Sparse Attention's (DeepSeek-V3.2, arXiv:2512.02556) with
//     GLM-5.3-Flash's k-pool. The absorbed MLA is DeepSeek-V2's MLA (arXiv:2405.04434), of which
//     GLM-5.3-Flash releases the NoPE form.
//   - SM120/SM121 (DGX Spark GB10) kernel design and local-inference execution patterns draw on
//     prior art from b12x (https://github.com/local-inference-lab/b12x,
//     flashinfer/experimental/b12x, Apache-2.0; Luke Alonso @lukealonso, @original-el8, and
//     contributors) and NVIDIA Flare (https://github.com/NVIDIA/flare-local-inference, Apache-2.0).
//     See the root NOTICE file for full provenance details.
//
// The arithmetic mirrors the GLM arm's forward.rs, which is the host reference oracle. Each kernel
// is named q4e_* and is loaded by name from the cubin, so a missing symbol is a load error rather
// than a silent wrong launch.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "prefetch_sm121.cuh"
#include "quant_decode_sm121.cuh"
#include "opt_pdl_sm121.cuh"
#include "q4e_common_sm121.cuh"
#include "q4e_gemv_sm121.cuh"

#define Q4E_BLOCK 256
#define Q4E_MAX_SELECTED 2304
#define Q4E_MAX_EXPERTS 512
#define Q4E_MAX_TOPK 64

#define q4e_e4m3 spark_fp8_e4m3_decode

// The budget selection for GLM-5.3-Flash's indexer (topk::top_pools with the tail always kept);
// q4e_qsa_pick's variant stays in the attention image.
extern "C" __global__ void q4e_glm_pick(const float* scores, int* selected, int* count,
                                        int position, int ratio, int budget, int score_stride,
                                        int select_stride) {
    OPT_PDL_ENTRY();
    q4e_pick_pools(scores, selected, count, position, ratio, budget, score_stride, select_stride,
                   true);
}

// ---------------------------------------------------------------------------------------------
// GLM-5.3-Flash's MLA over the compressed latent, instead of over the expanded per-head keys and
// values q4e_attn read.
//
// The expanded form ran kv_b on the normed latent and cached heads * (nope + value) floats a
// position: 8,192 at the release's geometry on one rank of four, against the 512 the latent it
// expands already holds. Caching the latent instead, and folding kv_b into the two matmuls
// attention already makes, is MLA absorption (DeepSeek-V2's MLA, arXiv:2405.04434, of which
// GLM-5.3-Flash releases the NoPE form):
//
//   k[h][j] = sum_c kv_b[h * (nope + value) + j][c] * c_kv[c]           j < nope
//   v[h][j] = sum_c kv_b[h * (nope + value) + nope + j][c] * c_kv[c]    j < value
//
// so a score is
//
//   sum_j q[h][j] k[h][j] = (sum_j q[h][j] kv_b[h * (nope + value) + j][c]) . c_kv
//
// the query projected into the latent space once a row, and a value is
//
//   sum_slot p[slot] v[h][slot] = kv_b[h * (nope + value) + nope + j] . (sum_slot p[slot] c_kv[slot])
//
// the weighted latent sum projected once a row. Both absorbed operands are this rank's own kv_b
// rows, dequantised once at load, so no kernel here reads the checkpoint's FP8 codes.
//
// The cache is the release's own E4M3 activation: 512 codes and one f32 scale a 128-column block a
// position a layer, 528 bytes, against the expanded form's 2,048. The quantisation is the one the
// FP8 GEMV applied to the latent, so the products are exactly the expansion's - the codes and
// scales are the values it would have multiplied. A configuration without dynamic activation
// quantisation (a BF16 release) caches the latent as f32 instead, and the attention reads either.
#define Q4E_GLM_MAX_LATENT 512

// One row's normed latent into an f32 cache, for a configuration whose FP8 projections do not
// quantise their activations (a BF16 release). grid (rows).
extern "C" __global__ void q4e_glm_mla_cache(const float* __restrict__ latent,
                                             float* __restrict__ cache, int base, int rows,
                                             int width) {
    OPT_PDL_ENTRY();
    const int row = blockIdx.x;
    if (row >= rows || width > Q4E_GLM_MAX_LATENT) {
        return;
    }
    const float* src = latent + (unsigned long long)row * width;
    float* dst = cache + (unsigned long long)(base + row) * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// One row's normed latent into the E4M3 cache: per 128-column block the release's quantisation,
// the largest magnitude over 448 as the scale, each value over it rounded to the nearest code, and
// the scale kept beside the codes. grid (rows), Q4E_FP8_BLOCK threads: a warp a block, four values
// a lane.
extern "C" __global__ void q4e_glm_mla_cache_fp8(const float* __restrict__ latent,
                                                 unsigned char* __restrict__ codes,
                                                 float* __restrict__ scales, int base, int rows,
                                                 int width) {
    OPT_PDL_ENTRY();
    const int row = blockIdx.x;
    if (row >= rows || width > Q4E_GLM_MAX_LATENT || width % Q4E_FP8_BLOCK != 0) {
        return;
    }
    const float* src = latent + (unsigned long long)row * width;
    unsigned char* dst = codes + (unsigned long long)(base + row) * width;
    float* scl = scales + (unsigned long long)(base + row) * (width / Q4E_FP8_BLOCK);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    for (int start = warp * Q4E_FP8_BLOCK; start < width; start += warps * Q4E_FP8_BLOCK) {
        const float4 value = *reinterpret_cast<const float4*>(src + start + lane * 4);
        float largest = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                              fmaxf(fabsf(value.z), fabsf(value.w)));
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            largest = fmaxf(largest, __shfl_xor_sync(0xFFFFFFFFu, largest, offset));
        }
        const float scale = largest / 448.0f;
        if (lane == 0) {
            scl[start / Q4E_FP8_BLOCK] = scale;
        }
        unsigned int c0 = 0u, c1 = 0u, c2 = 0u, c3 = 0u;
        if (scale != 0.0f) {
            c0 = __nv_cvt_float_to_fp8(value.x / scale, __NV_SATFINITE, __NV_E4M3);
            c1 = __nv_cvt_float_to_fp8(value.y / scale, __NV_SATFINITE, __NV_E4M3);
            c2 = __nv_cvt_float_to_fp8(value.z / scale, __NV_SATFINITE, __NV_E4M3);
            c3 = __nv_cvt_float_to_fp8(value.w / scale, __NV_SATFINITE, __NV_E4M3);
        }
        *reinterpret_cast<uchar4*>(dst + start + lane * 4) =
            make_uchar4((unsigned char)c0, (unsigned char)c1, (unsigned char)c2,
                        (unsigned char)c3);
    }
}

// One cached latent, decoded as the release's activation: the E4M3 code times its 128-column
// block's scale, or the f32 cache when the configuration does not quantise.
__device__ __forceinline__ float q4e_glm_mla_at(const float* __restrict__ latent,
                                                const unsigned char* __restrict__ codes,
                                                const float* __restrict__ scales, int width,
                                                int row, int column, int quantise) {
    if (quantise) {
        return q4e_e4m3(codes[(unsigned long long)row * width + column]) *
               scales[(unsigned long long)row * (width / Q4E_FP8_BLOCK) +
                      column / Q4E_FP8_BLOCK];
    }
    return latent[(unsigned long long)row * width + column];
}

// The operands the absorbed MLA's serial sums load at once before they fold them in, in order:
// the sums are the same, and a thread has that many loads in flight instead of one.
#define Q4E_GLM_MLA_AHEAD 8

// The query half of the absorption: q_abs[row][head][c] = sum_j q[row][head][j] * kv_b[h][j][c],
// kv_b's key rows, [nope][latent] a head. grid (heads, rows, latent / blockDim.x rounded up):
// thread threadIdx.x of z-block z owns latent column z * blockDim.x + threadIdx.x, so a warp's
// reads of the weight are consecutive words and its reads of the query one broadcast.
extern "C" __global__ void q4e_glm_mla_absorb(const float* __restrict__ q,
                                              const float* __restrict__ kv_b,
                                              float* __restrict__ q_abs, int base, int nope,
                                              int value, int latent) {
    OPT_PDL_ENTRY();
    const int head = blockIdx.x;
    const int row = base + blockIdx.y;
    const int c = blockIdx.z * blockDim.x + threadIdx.x;
    if (latent > Q4E_GLM_MAX_LATENT || c >= latent) {
        return;
    }
    const float* qh = q + ((unsigned long long)row * gridDim.x + head) * nope;
    const float* wc = kv_b + (unsigned long long)head * (nope + value) * latent + c;
    float acc = 0.0f;
    int j = 0;
    for (; j + Q4E_GLM_MLA_AHEAD <= nope; j += Q4E_GLM_MLA_AHEAD) {
        float w[Q4E_GLM_MLA_AHEAD];
#pragma unroll
        for (int k = 0; k < Q4E_GLM_MLA_AHEAD; ++k) {
            w[k] = wc[(unsigned long long)(j + k) * latent];
        }
#pragma unroll
        for (int k = 0; k < Q4E_GLM_MLA_AHEAD; ++k) {
            acc = fmaf(w[k], qh[j + k], acc);
        }
    }
    for (; j < nope; ++j) {
        acc = fmaf(wc[(unsigned long long)j * latent], qh[j], acc);
    }
    q_abs[((unsigned long long)row * gridDim.x + head) * latent + c] = acc;
}

// The attention and the value half together: a f32 softmax over the selected latents, then the
// weighted latent sum projected by kv_b's value rows, which a head keeps transposed after its key
// rows ([latent][value]) so a warp's reads of them are consecutive words. grid (heads, rows),
// Q4E_BLOCK threads. The score and softmax phases are q4e_attn's, over the latent instead of an
// expanded key. The decode step runs it split four ways (q4e_glm_mla_scores below); this kernel
// is the bits the split is held to (tests/glm_kernels.rs).
extern "C" __global__ void q4e_glm_mla_attend(const float* __restrict__ q_abs,
                                              const float* __restrict__ latent_cache,
                                              const unsigned char* __restrict__ codes,
                                              const float* __restrict__ scales,
                                              const float* __restrict__ kv_b,
                                              const int* __restrict__ selected,
                                              const int* __restrict__ selected_count,
                                              float* __restrict__ out, int base, int nope,
                                              int value, int latent, int select_stride,
                                              int quantise) {
    OPT_PDL_ENTRY();
    __shared__ float logits[Q4E_MAX_SELECTED];
    __shared__ float reduce[Q4E_BLOCK];
    __shared__ float mixed[Q4E_GLM_MAX_LATENT];
    const int head = blockIdx.x;
    const int row = base + blockIdx.y;
    const int* sel = selected + (unsigned long long)row * select_stride;
    const int nsel = selected_count[row];
    if (latent > Q4E_GLM_MAX_LATENT) {
        return;
    }
    const float* qh = q_abs + ((unsigned long long)row * gridDim.x + head) * latent;
    const float* wv = kv_b + (unsigned long long)head * (nope + value) * latent +
                      (unsigned long long)nope * latent;
    float* oh = out + ((unsigned long long)row * gridDim.x + head) * value;
    if (nsel <= 0) {
        // Before the indexer has a full block there is nothing to attend over, and the reference
        // keeps the row at zero rather than leaving the buffer untouched.
        for (int j = threadIdx.x; j < value; j += blockDim.x) {
            oh[j] = 0.0f;
        }
        return;
    }
    if (nsel > Q4E_MAX_SELECTED) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const float scale = sqrtf((float)nope);

    for (int slot = warp; slot < nsel; slot += warps) {
        const int at = sel[slot];
        float dot = 0.0f;
        for (int c = lane; c < latent; c += 32) {
            dot += qh[c] * q4e_glm_mla_at(latent_cache, codes, scales, latent, at, c, quantise);
        }
        dot = q4e_warp_sum(dot);
        if (lane == 0) {
            logits[slot] = dot / scale;
        }
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int slot = threadIdx.x; slot < nsel; slot += blockDim.x) {
        local_max = fmaxf(local_max, logits[slot]);
    }
    const float maximum = q4e_block_max(local_max, reduce);
    float local_sum = 0.0f;
    for (int slot = threadIdx.x; slot < nsel; slot += blockDim.x) {
        const float weight = __expf(logits[slot] - maximum);
        logits[slot] = weight;
        local_sum += weight;
    }
    const float total = q4e_block_sum(local_sum, reduce);

    for (int c = threadIdx.x; c < latent; c += blockDim.x) {
        float accumulator = 0.0f;
        int slot = 0;
        for (; slot + Q4E_GLM_MLA_AHEAD <= nsel; slot += Q4E_GLM_MLA_AHEAD) {
            float x[Q4E_GLM_MLA_AHEAD];
#pragma unroll
            for (int k = 0; k < Q4E_GLM_MLA_AHEAD; ++k) {
                x[k] = q4e_glm_mla_at(latent_cache, codes, scales, latent, sel[slot + k], c,
                                      quantise);
            }
#pragma unroll
            for (int k = 0; k < Q4E_GLM_MLA_AHEAD; ++k) {
                accumulator = fmaf(logits[slot + k], x[k], accumulator);
            }
        }
        for (; slot < nsel; ++slot) {
            accumulator =
                fmaf(logits[slot],
                     q4e_glm_mla_at(latent_cache, codes, scales, latent, sel[slot], c, quantise),
                     accumulator);
        }
        mixed[c] = accumulator / total;
    }
    __syncthreads();

    for (int j = threadIdx.x; j < value; j += blockDim.x) {
        const float* wj = wv + j;
        float accumulator = 0.0f;
        int c = 0;
        for (; c + Q4E_GLM_MLA_AHEAD <= latent; c += Q4E_GLM_MLA_AHEAD) {
            float w[Q4E_GLM_MLA_AHEAD];
#pragma unroll
            for (int k = 0; k < Q4E_GLM_MLA_AHEAD; ++k) {
                w[k] = wj[(unsigned long long)(c + k) * value];
            }
#pragma unroll
            for (int k = 0; k < Q4E_GLM_MLA_AHEAD; ++k) {
                accumulator = fmaf(w[k], mixed[c + k], accumulator);
            }
        }
        for (; c < latent; ++c) {
            accumulator = fmaf(wj[(unsigned long long)c * value], mixed[c], accumulator);
        }
        oh[j] = accumulator;
    }
}

// q4e_glm_mla_attend's attention in four launches, so a full selection spreads over the GPU. The
// one kernel runs a block a head and a row: 16 of 48 SMs at a one-row step, each gathering its
// 2,048 selected latents one after another, twice, 1.3-1.8 ms a launch past 2k positions. Every sum
// here keeps the order it has there and every reduction its tree, so the outputs are its bits:
//
//   q4e_glm_mla_scores   a warp a selected position: its latent is read once for every head;
//   q4e_glm_mla_softmax  the attend kernel's softmax, the weights over the logits in place;
//   q4e_glm_mla_mix      the weighted latent sum, a lane a latent column and a warp a head;
//   q4e_glm_mla_project  the value projection, every row of a run over one read of kv_b.
//
// The logits are [row][head][Q4E_MAX_SELECTED], the mixed latents [row][head][latent], and the
// softmax sums [row][head].
#define Q4E_GLM_MLA_HEADS 16

// q4e_glm_mla_at in two halves, so a thread's gathers are all in flight before the first is
// decoded: the load of a cached latent's word (its E4M3 code, or the f32 cache's bits) and block
// scale, then the decode, the same product. The position and column are clamped in range by the
// caller, so every load is unconditional.
__device__ __forceinline__ void q4e_glm_mla_load(const float* __restrict__ latent_cache,
                                                 const unsigned char* __restrict__ codes,
                                                 const float* __restrict__ scales, int width,
                                                 int row, int column, int quantise,
                                                 unsigned int& word, float& scale) {
    if (quantise) {
        word = codes[(unsigned long long)row * width + column];
        scale = scales[(unsigned long long)row * (width / Q4E_FP8_BLOCK) +
                       column / Q4E_FP8_BLOCK];
    } else {
        word = __float_as_uint(latent_cache[(unsigned long long)row * width + column]);
        scale = 1.0f;
    }
}

__device__ __forceinline__ float q4e_glm_mla_decode(unsigned int word, float scale,
                                                    int quantise) {
    return quantise ? q4e_e4m3((unsigned char)word) * scale : __uint_as_float(word);
}
#define Q4E_GLM_MLA_SCORE_SLOTS 64
#define Q4E_GLM_MLA_MIX_TILE 128
#define Q4E_GLM_MLA_MIX_HEADS 8
#define Q4E_GLM_MLA_PROJECT_ROWS 8
#define Q4E_GLM_MLA_PROJECT_AHEAD 32

// The logits: grid (Q4E_MAX_SELECTED / Q4E_GLM_MLA_SCORE_SLOTS, rows), Q4E_BLOCK threads, at most
// Q4E_GLM_MLA_HEADS heads. A lane sums its columns lane, lane + 32, ... and the warp folds them as
// the attend kernel's warp does.
extern "C" __global__ void q4e_glm_mla_scores(const float* __restrict__ q_abs,
                                              const float* __restrict__ latent_cache,
                                              const unsigned char* __restrict__ codes,
                                              const float* __restrict__ scales,
                                              const int* __restrict__ selected,
                                              const int* __restrict__ selected_count,
                                              float* __restrict__ logits, int base, int heads,
                                              int nope, int latent, int select_stride,
                                              int quantise) {
    OPT_PDL_ENTRY();
    __shared__ float query[Q4E_GLM_MLA_HEADS * Q4E_GLM_MAX_LATENT];
    const int row = base + blockIdx.y;
    const int nsel = selected_count[row];
    const int first = blockIdx.x * Q4E_GLM_MLA_SCORE_SLOTS;
    if (heads > Q4E_GLM_MLA_HEADS || latent > Q4E_GLM_MAX_LATENT || nsel > Q4E_MAX_SELECTED ||
        first >= nsel) {
        return;
    }
    // Every load of the staging in flight at once, and each warp's positions in one load.
    const float4* q = reinterpret_cast<const float4*>(q_abs + (unsigned long long)row * heads *
                                                                   latent);
    float4* staged = reinterpret_cast<float4*>(query);
#pragma unroll 8
    for (int i = threadIdx.x; i < heads * latent / 4; i += blockDim.x) {
        staged[i] = q[i];
    }
    const int* sel = selected + (unsigned long long)row * select_stride;
    float* out = logits + (unsigned long long)row * heads * Q4E_MAX_SELECTED;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const float scale = sqrtf((float)nope);
    const int last = min(first + Q4E_GLM_MLA_SCORE_SLOTS, nsel);
    const int mine = first + warp + warps * lane;
    const int positions = mine < last ? sel[mine] : 0;
    __syncthreads();
    // A warp scores GROUP positions at once, so each staged query word it reads serves all of
    // them; each (head, position) sum is still a lane's columns in order, then the warp's fold.
    constexpr int SPAN = Q4E_GLM_MAX_LATENT / 32;
    constexpr int GROUP = 4;
    for (int group = 0; first + warp + warps * group < last; group += GROUP) {
        float x[GROUP][SPAN];
#pragma unroll
        for (int g = 0; g < GROUP; ++g) {
            const int at = __shfl_sync(0xFFFFFFFFu, positions, group + g);
            unsigned int word[SPAN];
            float block[SPAN];
#pragma unroll
            for (int k = 0; k < SPAN; ++k) {
                q4e_glm_mla_load(latent_cache, codes, scales, latent, at,
                                 min(lane + 32 * k, latent - 1), quantise, word[k], block[k]);
            }
#pragma unroll
            for (int k = 0; k < SPAN; ++k) {
                x[g][k] = lane + 32 * k < latent
                              ? q4e_glm_mla_decode(word[k], block[k], quantise)
                              : 0.0f;
            }
        }
        for (int head = 0; head < heads; ++head) {
            const float* qh = query + head * latent;
            float dot[GROUP];
#pragma unroll
            for (int g = 0; g < GROUP; ++g) {
                dot[g] = 0.0f;
            }
#pragma unroll
            for (int k = 0; k < SPAN; ++k) {
                const int c = lane + 32 * k;
                if (c < latent) {
                    const float qc = qh[c];
#pragma unroll
                    for (int g = 0; g < GROUP; ++g) {
                        dot[g] = fmaf(qc, x[g][k], dot[g]);
                    }
                }
            }
#pragma unroll
            for (int g = 0; g < GROUP; ++g) {
                dot[g] = q4e_warp_sum(dot[g]);
                const int slot = first + warp + warps * (group + g);
                if (lane == 0 && slot < last) {
                    out[head * Q4E_MAX_SELECTED + slot] = dot[g] / scale;
                }
            }
        }
    }
}

// The softmax: grid (heads, rows), Q4E_BLOCK threads, the attend kernel's block max and sum. Each
// logit becomes its unnormalised weight, and the sum goes to total.
extern "C" __global__ void q4e_glm_mla_softmax(float* __restrict__ logits,
                                               const int* __restrict__ selected_count,
                                               float* __restrict__ total, int base) {
    OPT_PDL_ENTRY();
    __shared__ float reduce[Q4E_BLOCK];
    const int head = blockIdx.x;
    const int heads = gridDim.x;
    const int row = base + blockIdx.y;
    const int nsel = selected_count[row];
    if (nsel <= 0 || nsel > Q4E_MAX_SELECTED) {
        return;
    }
    float* weights = logits + ((unsigned long long)row * heads + head) * Q4E_MAX_SELECTED;
    float local_max = -INFINITY;
    for (int slot = threadIdx.x; slot < nsel; slot += blockDim.x) {
        local_max = fmaxf(local_max, weights[slot]);
    }
    const float maximum = q4e_block_max(local_max, reduce);
    float local_sum = 0.0f;
    for (int slot = threadIdx.x; slot < nsel; slot += blockDim.x) {
        const float weight = __expf(weights[slot] - maximum);
        weights[slot] = weight;
        local_sum += weight;
    }
    const float sum = q4e_block_sum(local_sum, reduce);
    if (threadIdx.x == 0) {
        total[(unsigned long long)row * heads + head] = sum;
    }
}

// The weighted latent sum over the selection, in its order, over the softmax's sum: grid
// (latent / 32, heads / Q4E_GLM_MLA_MIX_HEADS, rows), 32 * Q4E_GLM_MLA_MIX_HEADS threads. A tile
// of positions' 32 columns and its weights load into registers while the block folds the tile
// before it, so a warp is never waiting on one position's gather.
extern "C" __global__ void q4e_glm_mla_mix(const float* __restrict__ latent_cache,
                                           const unsigned char* __restrict__ codes,
                                           const float* __restrict__ scales,
                                           const int* __restrict__ selected,
                                           const int* __restrict__ selected_count,
                                           const float* __restrict__ weights,
                                           const float* __restrict__ total,
                                           float* __restrict__ mixed, int base, int heads,
                                           int latent, int select_stride, int quantise) {
    OPT_PDL_ENTRY();
    constexpr int TILE = Q4E_GLM_MLA_MIX_TILE;
    constexpr int WARPS = Q4E_GLM_MLA_MIX_HEADS;
    __shared__ int at[Q4E_MAX_SELECTED];
    __shared__ float xs[TILE][32];
    __shared__ float ws[WARPS][TILE];
    const int row = base + blockIdx.z;
    const int nsel = selected_count[row];
    if (nsel <= 0 || nsel > Q4E_MAX_SELECTED || latent > Q4E_GLM_MAX_LATENT ||
        blockDim.x != 32 * WARPS) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int column = blockIdx.x * 32 + lane;
    const int head = blockIdx.y * WARPS + warp;
    const int* sel = selected + (unsigned long long)row * select_stride;
#pragma unroll
    for (int i = 0; i < (Q4E_MAX_SELECTED + 32 * WARPS - 1) / (32 * WARPS); ++i) {
        const int slot = threadIdx.x + 32 * WARPS * i;
        if (slot < nsel) {
            at[slot] = sel[slot];
        }
    }
    __syncthreads();
    const float* wh = weights + ((unsigned long long)row * heads + head) * Q4E_MAX_SELECTED;
    unsigned int word[TILE / WARPS];
    float block[TILE / WARPS];
    float w[TILE / 32];
    const int read = min(column, latent - 1);
    auto fetch = [&](int first) {
#pragma unroll
        for (int i = 0; i < TILE / WARPS; ++i) {
            q4e_glm_mla_load(latent_cache, codes, scales, latent,
                             at[min(first + warp + WARPS * i, nsel - 1)], read, quantise,
                             word[i], block[i]);
        }
#pragma unroll
        for (int i = 0; i < TILE / 32; ++i) {
            w[i] = head < heads ? wh[min(first + lane + 32 * i, nsel - 1)] : 0.0f;
        }
    };
    fetch(0);
    float accumulator = 0.0f;
    for (int first = 0; first < nsel; first += TILE) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < TILE / WARPS; ++i) {
            xs[warp + WARPS * i][lane] = q4e_glm_mla_decode(word[i], block[i], quantise);
        }
#pragma unroll
        for (int i = 0; i < TILE / 32; ++i) {
            ws[warp][lane + 32 * i] = w[i];
        }
        __syncthreads();
        if (first + TILE < nsel) {
            fetch(first + TILE);
        }
        const int count = min(TILE, nsel - first);
        if (count == TILE) {
#pragma unroll 16
            for (int slot = 0; slot < TILE; ++slot) {
                accumulator = fmaf(ws[warp][slot], xs[slot][lane], accumulator);
            }
        } else {
            for (int slot = 0; slot < count; ++slot) {
                accumulator = fmaf(ws[warp][slot], xs[slot][lane], accumulator);
            }
        }
    }
    if (column < latent && head < heads) {
        mixed[((unsigned long long)row * heads + head) * latent + column] =
            accumulator / total[(unsigned long long)row * heads + head];
    }
}

// The value projection, kv_b's value rows ([latent][value] a head, after its key rows) over the
// mixed latent: grid (heads, value / blockDim.x rounded up), rows base .. base + rows at once, so
// the weights are read once a run. A row with nothing selected is zero, as the attend kernel
// leaves it.
extern "C" __global__ void q4e_glm_mla_project(const float* __restrict__ mixed,
                                               const float* __restrict__ kv_b,
                                               const int* __restrict__ selected_count,
                                               float* __restrict__ out, int base, int rows,
                                               int nope, int value, int latent) {
    OPT_PDL_ENTRY();
    constexpr int ROWS = Q4E_GLM_MLA_PROJECT_ROWS;
    constexpr int AHEAD = Q4E_GLM_MLA_PROJECT_AHEAD;
    __shared__ float ms[ROWS][Q4E_GLM_MAX_LATENT];
    const int head = blockIdx.x;
    const int heads = gridDim.x;
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (latent > Q4E_GLM_MAX_LATENT) {
        return;
    }
    const float* wj = kv_b + (unsigned long long)head * (nope + value) * latent +
                      (unsigned long long)nope * latent + j;
    for (int first = 0; first < rows; first += ROWS) {
        const int count = min(ROWS, rows - first);
        __syncthreads();
        for (int r = 0; r < count; ++r) {
            const float* m =
                mixed + ((unsigned long long)(base + first + r) * heads + head) * latent;
#pragma unroll 8
            for (int c = threadIdx.x; c < latent; c += blockDim.x) {
                ms[r][c] = m[c];
            }
        }
        __syncthreads();
        if (j >= value) {
            continue;
        }
        float accumulator[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            accumulator[r] = 0.0f;
        }
        int c = 0;
        for (; c + AHEAD <= latent; c += AHEAD) {
            float w[AHEAD];
#pragma unroll
            for (int k = 0; k < AHEAD; ++k) {
                w[k] = wj[(unsigned long long)(c + k) * value];
            }
#pragma unroll
            for (int k = 0; k < AHEAD; ++k) {
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    accumulator[r] = fmaf(w[k], ms[r][c + k], accumulator[r]);
                }
            }
        }
        for (; c < latent; ++c) {
            const float wc = wj[(unsigned long long)c * value];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                accumulator[r] = fmaf(wc, ms[r][c], accumulator[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int at = base + first + r;
            if (r < count) {
                const int nsel = selected_count[at];
                if (nsel <= Q4E_MAX_SELECTED) {
                    out[((unsigned long long)at * heads + head) * value + j] =
                        nsel > 0 ? accumulator[r] : 0.0f;
                }
            }
        }
    }
}


// The most mHC copies the coefficient kernel holds: one comb cell a lane of one warp.
#define Q4E_MHC_MAX_HC 5
// The most columns of every copy a thread of q4e_mhc_in holds: the hidden width is a multiple of
// the block, at most this many of them (the host refuses others).
#define Q4E_MHC_COLUMNS 16

// a / d by the sequence nvcc emits for `/` when FCHK finds the operands in range: an approximate
// reciprocal, a Newton step on it, and an fma correction of the quotient. Where q4e_div_in_range
// holds, that is the path `/` takes, so the quotient rounds as `/`'s does (7.6e11 sampled pairs,
// none apart from __fdiv_rn). It leaves out `/`'s branch to its slow path, which in a dependent
// chain of divisions costs some 250 cycles each.
__device__ __forceinline__ float q4e_div_fast(float a, float d) {
    float reciprocal;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(reciprocal) : "f"(d));
    const float error = __fmaf_rn(-d, reciprocal, 1.0f);
    const float refined = __fmaf_rn(reciprocal, error, reciprocal);
    const float quotient = __fmaf_rn(a, refined, 0.0f);
    return __fmaf_rn(refined, __fmaf_rn(-d, quotient, a), quotient);
}

// Whether q4e_div_fast(a, d) is a / d: both operands normal, of magnitude in [2^-60, 2^61), so no
// step of the quotient leaves the normal range.
__device__ __forceinline__ bool q4e_div_in_range(float a, float d) {
    const unsigned a_exponent = (__float_as_uint(a) >> 23) & 0xFFu;
    const unsigned d_exponent = (__float_as_uint(d) >> 23) & 0xFFu;
    return a_exponent - 67u <= 120u && d_exponent - 67u <= 120u;
}

// sqrtf(x) by the path nvcc emits for it where q4e_sqrt_in_range holds (its own test): an
// approximate reciprocal root, then one correction. It leaves out sqrtf's branch to its slow path.
__device__ __forceinline__ float q4e_sqrt_fast(float x) {
    float reciprocal;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(reciprocal) : "f"(x));
    const float root = __fmul_rn(x, reciprocal);
    return __fmaf_rn(__fmaf_rn(-root, root, x), __fmul_rn(reciprocal, 0.5f), root);
}

__device__ __forceinline__ bool q4e_sqrt_in_range(float x) {
    return __float_as_uint(x) - 0x0D000000u <= 0x727FFFFFu;
}

// a / d and sqrtf(x): `/` and sqrtf themselves, or with FAST their fast paths, clearing `exact`
// where those may not round as they do.
template <bool FAST>
__device__ __forceinline__ float q4e_mhc_div(float a, float d, bool& exact) {
    if constexpr (FAST) {
        exact = exact && q4e_div_in_range(a, d);
        return q4e_div_fast(a, d);
    }
    return a / d;
}

template <bool FAST>
__device__ __forceinline__ float q4e_mhc_sqrt(float x, bool& exact) {
    if constexpr (FAST) {
        exact = exact && q4e_sqrt_in_range(x);
        return q4e_sqrt_fast(x);
    }
    return sqrtf(x);
}

// 1 / sqrt(sum / count + eps): the RMS scale of mhc::hc_mixes_from_projection and linalg::rms_norm.
template <bool FAST>
__device__ __forceinline__ float q4e_mhc_rms_scale(float sum, float count, float eps,
                                                   bool& exact) {
    const float mean = q4e_mhc_div<FAST>(sum, count, exact);
    return q4e_mhc_div<FAST>(1.0f, q4e_mhc_sqrt<FAST>(__fadd_rn(mean, eps), exact), exact);
}

// What lane l of q4e_mhc_in's warp 0 reads of a token's fn projection and the mixer's
// parameters: gate l's (l < 2 hc) and comb cell l's (l < hc * hc), zero past them.
struct Q4eMhcLane {
    float gate_raw, gate_scale, gate_base;
    float cell_raw, cell_scale, cell_base;
};

// The parameters of the lane's gate (l < gates) and, with comb, its comb cell: before the wait on
// the launch before, which never writes them.
__device__ __forceinline__ Q4eMhcLane q4e_mhc_lane(const float* __restrict__ scale,
                                                   const float* __restrict__ base, int hc,
                                                   int gates, bool comb) {
    const int lane = threadIdx.x;
    Q4eMhcLane in = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (lane < gates) {
        in.gate_scale = scale[lane < hc ? 0 : 1];
        in.gate_base = base[lane];
    }
    if (comb && lane < hc * hc) {
        in.cell_scale = scale[2];
        in.cell_base = base[2 * hc + lane];
    }
    return in;
}

// Output i of token t's fn projection: raw_t[i], its row of the tiled GEMV's output, or with
// parts_t the [splits][mixes] partials q4e_mhc_post worked out in the GEMV's place, added from 0.0
// in split order as the GEMV's last block adds its own.
__device__ __forceinline__ float q4e_mhc_raw(const float* __restrict__ raw_t,
                                             const float* __restrict__ parts_t, int splits,
                                             int mixes, int i) {
    if (parts_t == nullptr) {
        return raw_t[i];
    }
    float sum = 0.0f;
#pragma unroll 16
    for (int split = 0; split < splits; ++split) {
        sum += parts_t[(unsigned long long)split * mixes + i];
    }
    return sum;
}

// The lane's share of token t's fn projection ([batch][(2 + hc) hc]), after the wait.
__device__ __forceinline__ void q4e_mhc_lane_raw(Q4eMhcLane& in, const float* __restrict__ raw_t,
                                                 const float* __restrict__ parts_t, int splits,
                                                 int hc, int gates, bool comb) {
    const int lane = threadIdx.x;
    const int mixes = (2 + hc) * hc;
    if (lane < gates) {
        in.gate_raw = q4e_mhc_raw(raw_t, parts_t, splits, mixes, lane);
    }
    if (comb && lane < hc * hc) {
        in.cell_raw = q4e_mhc_raw(raw_t, parts_t, splits, mixes, 2 * hc + lane);
    }
}

// A token's streams [copy][hidden] at x: a thread loads its columns j = threadIdx.x + k blockDim.x
// of every copy at once into held[copy][k] and adds their squares in the streams' index order
// (copy, then column), and q4e_block_sum reduces the block's. `scratch` holds Q4E_BLOCK floats.
__device__ __forceinline__ float q4e_mhc_hold(const float* __restrict__ x, int hidden, int hc,
                                              int columns,
                                              float (&held)[Q4E_MHC_MAX_HC][Q4E_MHC_COLUMNS],
                                              float* scratch) {
#pragma unroll
    for (int copy = 0; copy < Q4E_MHC_MAX_HC; ++copy) {
#pragma unroll
        for (int k = 0; k < Q4E_MHC_COLUMNS; ++k) {
            held[copy][k] = copy < hc && k < columns
                                ? x[(unsigned long long)copy * hidden + threadIdx.x +
                                    k * blockDim.x]
                                : 0.0f;
        }
    }
    float sum = 0.0f;
#pragma unroll
    for (int copy = 0; copy < Q4E_MHC_MAX_HC; ++copy) {
#pragma unroll
        for (int k = 0; k < Q4E_MHC_COLUMNS; ++k) {
            if (copy < hc && k < columns) {
                sum = __fadd_rn(sum, __fmul_rn(held[copy][k], held[copy][k]));
            }
        }
    }
    return q4e_block_sum(sum, scratch);
}

// Warp 0's mHC coefficients of a token, as mhc::hc_mixes_from_projection and split_sinkhorn: lane
// l < gates (hc for pre alone, 2 hc for pre and post) gets its gate, and with `comb` lane row * hc +
// col < hc * hc gets comb[row][col]. Every row or column sum reads its hc cells through shuffles in
// the host's order, so the 2 * iters normalisations round as the host's do. With FAST every
// division and square root takes its fast path, and `exact` clears where one the lane returns
// depends on may not have rounded as `/` and sqrtf do.
template <bool FAST>
__device__ __forceinline__ void q4e_mhc_mixes(const Q4eMhcLane& in, float sum_squares, int hc,
                                              int width, int gates, bool comb, int iters,
                                              float hc_eps, float norm_eps, float& gate,
                                              float& cell, bool& exact) {
    const int lane = threadIdx.x & 31;
    const float rsqrt = q4e_mhc_rms_scale<FAST>(sum_squares, (float)width, norm_eps, exact);
    // linalg::sigmoid: 1 / (1 + e^-x) at x >= 0 and e^x / (1 + e^x) below.
    const float x =
        __fadd_rn(__fmul_rn(__fmul_rn(in.gate_raw, rsqrt), in.gate_scale), in.gate_base);
    const bool positive = x >= 0.0f;
    const float power = expf(positive ? -x : x);
    bool gate_exact = true;
    const float sigmoid = q4e_mhc_div<FAST>(positive ? 1.0f : power, 1.0f + power, gate_exact);
    gate = lane < hc ? __fadd_rn(sigmoid, hc_eps) : __fmul_rn(2.0f, sigmoid);
    exact = exact && (gate_exact || lane >= gates);
    if (!comb) {
        return;
    }
    const bool live = lane < hc * hc;
    const int row = live ? lane / hc : 0;
    const int col = live ? lane - row * hc : 0;
    float value =
        live ? __fadd_rn(__fmul_rn(__fmul_rn(in.cell_raw, rsqrt), in.cell_scale), in.cell_base)
             : 0.0f;
    // The row softmax (linalg::softmax), then hc_eps.
    float maximum = -INFINITY;
#pragma unroll
    for (int k = 0; k < Q4E_MHC_MAX_HC; ++k) {
        const float other = __shfl_sync(0xFFFFFFFFu, value, row * hc + k);
        if (k < hc) {
            maximum = fmaxf(maximum, other);
        }
    }
    const float e = expf(value - maximum);
    float total = 0.0f;
#pragma unroll
    for (int k = 0; k < Q4E_MHC_MAX_HC; ++k) {
        const float other = __shfl_sync(0xFFFFFFFFu, e, row * hc + k);
        if (k < hc) {
            total = __fadd_rn(total, other);
        }
    }
    bool cell_exact = true;
    const float share = q4e_mhc_div<FAST>(e, total, cell_exact);
    cell_exact = cell_exact || !(isfinite(maximum) && total > 0.0f);
    value = isfinite(maximum) ? (total > 0.0f ? share : e) : 0.0f;
    value = __fadd_rn(value, hc_eps);
    for (int pass = 0; pass < iters; ++pass) {
        if (pass > 0) {
            float row_sum = 0.0f;
#pragma unroll
            for (int k = 0; k < Q4E_MHC_MAX_HC; ++k) {
                const float other = __shfl_sync(0xFFFFFFFFu, value, row * hc + k);
                if (k < hc) {
                    row_sum = __fadd_rn(row_sum, other);
                }
            }
            value = q4e_mhc_div<FAST>(value, __fadd_rn(row_sum, hc_eps), cell_exact);
        }
        float column_sum = 0.0f;
#pragma unroll
        for (int k = 0; k < Q4E_MHC_MAX_HC; ++k) {
            const float other = __shfl_sync(0xFFFFFFFFu, value, k * hc + col);
            if (k < hc) {
                column_sum = __fadd_rn(column_sum, other);
            }
        }
        value = q4e_mhc_div<FAST>(value, __fadd_rn(column_sum, hc_eps), cell_exact);
    }
    cell = value;
    exact = exact && (cell_exact || !live);
}

// q4e_mhc_mixes on the fast paths, and again through `/` and sqrtf when one of the warp's may not
// have rounded as they do.
__device__ __forceinline__ void q4e_mhc_warp_mixes(const Q4eMhcLane& in, float sum_squares, int hc,
                                                   int width, int gates, bool comb, int iters,
                                                   float hc_eps, float norm_eps, float& gate,
                                                   float& cell) {
    bool exact = true;
    q4e_mhc_mixes<true>(in, sum_squares, hc, width, gates, comb, iters, hc_eps, norm_eps, gate,
                        cell, exact);
    if (__any_sync(0xFFFFFFFFu, !exact)) {
        q4e_mhc_mixes<false>(in, sum_squares, hc, width, gates, comb, iters, hc_eps, norm_eps,
                             gate, cell, exact);
    }
}

// A sublayer's mixer: grid (batch, 2), Q4E_BLOCK threads, two blocks a token that run side by
// side. Block (t, 1) writes token t's coefficients [batch][pre (hc) | post (hc) | comb (hc x hc)]
// from its fn projection and its streams' sum of squares. Block (t, 0) works out only the pre
// coefficients, and from them the normed sublayer input `out`: sum_c pre[c] x[c][j] from zero in
// copy order (mhc::hc_pre), then RMSNorm with the sublayer's BF16 weight (linalg::rms_norm). So
// the input never waits on the comb's 2 * iters normalisations. Only the sums of squares are
// reassociated (block trees).
//
// A thread keeps the stream values it squares (q4e_mhc_hold): they are the very columns its pre
// sums read. The mixer's parameters and the norm weight it reads before waiting on the launch
// before. Warp 0 takes the divisions' and square roots' fast paths (q4e_mhc_warp_mixes), as does
// the input's RMS scale, each falling back to `/` and sqrtf where those may round otherwise.
//
// The fn projection is `raw`, or with `partials` the split partials the q4e_mhc_post before it
// left there ([batch][hc hidden / Q4E_BLOCK][(2 + hc) hc]); `raw` is then unread.
extern "C" __global__ void q4e_mhc_in(const float* __restrict__ raw,
                                      const float* __restrict__ streams,
                                      const float* __restrict__ scale,
                                      const float* __restrict__ base, float* __restrict__ coeffs,
                                      const __nv_bfloat16* __restrict__ weight,
                                      float* __restrict__ out, int hidden, int hc, int iters,
                                      float hc_eps, float norm_eps, float eps,
                                      const float* __restrict__ partials) {
    OPT_PDL_LAUNCH();
    __shared__ float scratch[Q4E_BLOCK];
    __shared__ float pre[Q4E_MHC_MAX_HC];
    const int columns = hidden / (int)blockDim.x;
    if (hc <= 0 || hc > Q4E_MHC_MAX_HC || columns * (int)blockDim.x != hidden ||
        columns > Q4E_MHC_COLUMNS) {
        return;
    }
    const int t = blockIdx.x;
    const int lane = threadIdx.x;
    const int mixes = (2 + hc) * hc;
    const float* raw_t = raw + (unsigned long long)t * mixes;
    const int splits = hc * hidden / Q4E_BLOCK;
    const float* parts_t =
        partials != nullptr ? partials + (unsigned long long)t * splits * mixes : nullptr;
    const float* x = streams + (unsigned long long)t * hc * hidden;
    float held[Q4E_MHC_MAX_HC][Q4E_MHC_COLUMNS];
    float gate = 0.0f;
    float cell = 0.0f;
    if (blockIdx.y != 0) {
        Q4eMhcLane in = q4e_mhc_lane(scale, base, hc, 2 * hc, true);
        OPT_PDL_WAIT();
        q4e_mhc_lane_raw(in, raw_t, parts_t, splits, hc, 2 * hc, true);
        const float sum_squares = q4e_mhc_hold(x, hidden, hc, columns, held, scratch);
        if (threadIdx.x >= 32) {
            return;
        }
        q4e_mhc_warp_mixes(in, sum_squares, hc, hc * hidden, 2 * hc, true, iters, hc_eps,
                           norm_eps, gate, cell);
        float* o = coeffs + (unsigned long long)t * mixes;
        if (lane < 2 * hc) {
            o[lane] = gate;
        }
        if (lane < hc * hc) {
            o[2 * hc + lane] = cell;
        }
        return;
    }
    Q4eMhcLane in = q4e_mhc_lane(scale, base, hc, hc, false);
    __nv_bfloat16 norm[Q4E_MHC_COLUMNS];
#pragma unroll
    for (int k = 0; k < Q4E_MHC_COLUMNS; ++k) {
        norm[k] = k < columns ? weight[threadIdx.x + k * blockDim.x] : __float2bfloat16(0.0f);
    }
    OPT_PDL_WAIT();
    q4e_mhc_lane_raw(in, raw_t, parts_t, splits, hc, hc, false);
    const float sum_squares = q4e_mhc_hold(x, hidden, hc, columns, held, scratch);
    if (threadIdx.x < 32) {
        q4e_mhc_warp_mixes(in, sum_squares, hc, hc * hidden, hc, false, iters, hc_eps, norm_eps,
                           gate, cell);
        if (lane < hc) {
            pre[lane] = gate;
        }
    }
    __syncthreads();
    float p[Q4E_MHC_MAX_HC];
#pragma unroll
    for (int copy = 0; copy < Q4E_MHC_MAX_HC; ++copy) {
        p[copy] = copy < hc ? pre[copy] : 0.0f;
    }
    float mixed[Q4E_MHC_COLUMNS];
    float squares = 0.0f;
#pragma unroll
    for (int k = 0; k < Q4E_MHC_COLUMNS; ++k) {
        float accumulated = 0.0f;
#pragma unroll
        for (int copy = 0; copy < Q4E_MHC_MAX_HC; ++copy) {
            if (copy < hc) {
                accumulated = __fadd_rn(accumulated, __fmul_rn(p[copy], held[copy][k]));
            }
        }
        mixed[k] = accumulated;
        if (k < columns) {
            squares = __fadd_rn(squares, __fmul_rn(accumulated, accumulated));
        }
    }
    const float total = q4e_block_sum(squares, scratch);
    bool exact = true;
    float inverse = q4e_mhc_rms_scale<true>(total, (float)hidden, eps, exact);
    if (__any_sync(0xFFFFFFFFu, !exact)) {
        inverse = q4e_mhc_rms_scale<false>(total, (float)hidden, eps, exact);
    }
    float* y = out + (unsigned long long)t * hidden;
#pragma unroll
    for (int k = 0; k < Q4E_MHC_COLUMNS; ++k) {
        if (k < columns) {
            y[threadIdx.x + k * blockDim.x] =
                __fmul_rn(__fmul_rn(mixed[k], inverse), __bfloat162float(norm[k]));
        }
    }
}

// The streams after a sublayer, in place: copy k becomes post[k] y + sum_s comb[s][k] x[s], in
// mhc::hc_post's order. A thread owns one column of every copy, so reading the old streams and
// writing the new ones never crosses threads. grid (ceil(hidden / Q4E_BLOCK), batch).
//
// With world > 1, y is the TP all-reduce of the ranks' partials, Wire::F32's sum in ascending
// rank order from 0.0: this rank's own from branch, every peer's from its slot of slots
// ([world][batch][hidden], this rank's slot unread).
//
// With next_fn, the next mixer's [mixes][hc hidden] BF16 fn projection, a hidden of whole blocks
// and every split of that projection's tiled GEMV one 256-column round (the host's
// tiled_gemv_splits says when), the block also works out the GEMV's partials of the splits it
// holds. Split s = copy (hidden / Q4E_BLOCK) + blockIdx.x is the block's columns of `copy`: lane l
// takes its columns 8l..8l+8 through q4e_tg_fma8 from 0.0 and q4e_tg_fold folds the lanes, as the
// GEMV's round does. They land in next_partials ([batch][splits][mixes]) for q4e_mhc_in to add in
// split order from 0.0, as the GEMV's last block adds its own, so the projection is the GEMV's to
// the bit and the GEMV does not run. A warp takes the (copy, row) pairs warp + 8 i, whose weights
// it loads before the wait.
#define Q4E_MHC_PAIRS 12
static_assert(Q4E_BLOCK == Q4E_TG_THREADS && Q4E_BLOCK == 32 * 8,
              "a mixer block's columns of a copy are one round of the tiled BF16 GEMV");
extern "C" __global__ void q4e_mhc_post(float* __restrict__ streams,
                                        const float* __restrict__ branch,
                                        const float* __restrict__ coeffs, int hidden, int hc,
                                        const float* __restrict__ slots, int rank, int world,
                                        const __nv_bfloat16* __restrict__ next_fn,
                                        float* __restrict__ next_partials, int mixes) {
    OPT_PDL_LAUNCH();
    __shared__ __align__(16) float staged[Q4E_MHC_MAX_HC][Q4E_BLOCK];
    const int t = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const bool fused = next_fn != nullptr;
    const int pairs = hc * mixes;
    if (hc <= 0 || hc > Q4E_MHC_MAX_HC ||
        (fused && (blockDim.x != Q4E_BLOCK || hidden % Q4E_BLOCK != 0 ||
                   pairs > Q4E_TG_WARPS * Q4E_MHC_PAIRS))) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    uint4 word[Q4E_MHC_PAIRS];
#pragma unroll
    for (int i = 0; i < Q4E_MHC_PAIRS; ++i) {
        const int p = warp + Q4E_TG_WARPS * i;
        word[i] = make_uint4(0u, 0u, 0u, 0u);
        if (fused && p < pairs) {
            const int copy = p / mixes;
            const int row = p - copy * mixes;
            word[i] = q4e_ldg(reinterpret_cast<const uint4*>(
                next_fn + ((unsigned long long)row * hc + copy) * hidden + blockIdx.x * Q4E_BLOCK +
                8 * lane));
        }
    }
    OPT_PDL_WAIT();
    if (j >= hidden) {
        return;
    }
    float* x = streams + (unsigned long long)t * hc * hidden;
    const float* post = coeffs + (unsigned long long)t * (2 + hc) * hc + hc;
    const float* comb = post + hc;
    const unsigned long long at = (unsigned long long)t * hidden + j;
    float y = branch[at];
    if (world > 1) {
        const unsigned long long slot = (unsigned long long)gridDim.y * hidden;
        float sum = 0.0f;
        for (int r = 0; r < world; ++r) {
            sum = __fadd_rn(sum, r == rank ? y : slots[r * slot + at]);
        }
        y = sum;
    }
    float residual[Q4E_MHC_MAX_HC];
#pragma unroll
    for (int source = 0; source < Q4E_MHC_MAX_HC; ++source) {
        residual[source] = source < hc ? x[(unsigned long long)source * hidden + j] : 0.0f;
    }
#pragma unroll
    for (int copy = 0; copy < Q4E_MHC_MAX_HC; ++copy) {
        if (copy < hc) {
            float accumulated = __fmul_rn(post[copy], y);
#pragma unroll
            for (int source = 0; source < Q4E_MHC_MAX_HC; ++source) {
                if (source < hc) {
                    accumulated =
                        __fadd_rn(accumulated, __fmul_rn(comb[source * hc + copy], residual[source]));
                }
            }
            x[(unsigned long long)copy * hidden + j] = accumulated;
            staged[copy][threadIdx.x] = accumulated;
        }
    }
    if (!fused) {
        return;
    }
    __syncthreads();
    const int splits = hc * hidden / Q4E_BLOCK;
    float* out = next_partials + (unsigned long long)t * splits * mixes;
#pragma unroll
    for (int i = 0; i < Q4E_MHC_PAIRS; ++i) {
        const int p = warp + Q4E_TG_WARPS * i;
        if (p < pairs) {
            const int copy = p / mixes;
            const int row = p - copy * mixes;
            const float4* held = reinterpret_cast<const float4*>(&staged[copy][8 * lane]);
            const float sum = q4e_tg_fold(q4e_tg_fma8(word[i], held[0], held[1], 0.0f));
            if (lane == 0) {
                out[(unsigned long long)(copy * (hidden / Q4E_BLOCK) + blockIdx.x) * mixes + row] =
                    sum;
            }
        }
    }
}

// GLM's clamped SwiGLU over rows of [gate (width) | up (width)]: silu(min(gate, limit)) *
// clamp(up, -limit, limit), with linalg's silu. grid (ceil(width / Q4E_BLOCK), rows).
extern "C" __global__ void q4e_swiglu_limited(const float* __restrict__ gate_up,
                                              float* __restrict__ out, int width, float limit) {
    OPT_PDL_ENTRY();
    const int row = blockIdx.y;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width) {
        return;
    }
    const float* source = gate_up + (unsigned long long)row * 2 * width;
    const float gate = fminf(source[index], limit);
    const float up = fminf(fmaxf(source[width + index], -limit), limit);
    out[(unsigned long long)row * width + index] =
        __fmul_rn(__fmul_rn(gate, q4e_sigmoid_ref(gate)), up);
}

// GLM's sigmoid noaux_tc routing over the router's logits ([batch][experts]): each score is
// sigmoid(logit); the top-k is taken on score + bias, the lower expert on a tie; the weights are
// the plain scores, renormalised by their sum (in pick order) plus 1e-20 when `renormalise` and
// topk > 1, then times `scaling` (topk::noaux_top_k). The slots are written in ascending expert
// order, the order the host combines them in. grid (batch), one warp: lane l holds experts
// l, l + 32, ..., and each pick is a butterfly over the lanes' best untaken experts, so it costs
// no block barrier.
extern "C" __global__ void q4e_noaux_router(const float* __restrict__ logits,
                                            const float* __restrict__ bias,
                                            int* __restrict__ indices, float* __restrict__ weights,
                                            int experts, int topk, int renormalise,
                                            float scaling) {
    OPT_PDL_ENTRY();
    constexpr int kEach = Q4E_MAX_EXPERTS / 32;
    static_assert(kEach <= 32, "a lane's taken experts are one word's bits");
    if (experts <= 0 || experts > Q4E_MAX_EXPERTS || topk <= 0 || topk > Q4E_MAX_TOPK ||
        topk > experts || threadIdx.x >= 32) {
        return;
    }
    const int b = blockIdx.x;
    const int lane = threadIdx.x;
    // Every load first, so they are in flight together.
    float score[kEach];
    float signal[kEach];
#pragma unroll
    for (int k = 0; k < kEach; ++k) {
        const int expert = lane + 32 * k;
        score[k] = expert < experts ? logits[(unsigned long long)b * experts + expert] : 0.0f;
        signal[k] = expert < experts ? bias[expert] : 0.0f;
    }
#pragma unroll
    for (int k = 0; k < kEach; ++k) {
        if (lane + 32 * k < experts) {
            score[k] = q4e_sigmoid_ref(score[k]);
            signal[k] = __fadd_rn(score[k], signal[k]);
        }
    }
    unsigned int taken = 0;
    int picked[Q4E_MAX_TOPK];
    float chosen[Q4E_MAX_TOPK];
    for (int slot = 0; slot < topk; ++slot) {
        float best_value = -INFINITY;
        int best = 0x7FFFFFFF;
#pragma unroll
        for (int k = 0; k < kEach; ++k) {
            if (lane + 32 * k < experts && (taken >> k & 1u) == 0) {
                q4e_take_better(signal[k], lane + 32 * k, &best_value, &best);
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            q4e_take_better(__shfl_xor_sync(0xFFFFFFFFu, best_value, offset),
                            __shfl_xor_sync(0xFFFFFFFFu, best, offset), &best_value, &best);
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < kEach; ++k) {
            if (lane + 32 * k == best) {
                taken |= 1u << k;
                mine = score[k];
            }
        }
        picked[slot] = best;
        chosen[slot] = __shfl_sync(0xFFFFFFFFu, mine, best & 31);
    }
    if (lane == 0) {
        float total = 0.0f;
        for (int slot = 0; slot < topk; ++slot) {
            total = __fadd_rn(total, chosen[slot]);
        }
        const float denominator = __fadd_rn(total, 1e-20f);
        for (int slot = 0; slot < topk; ++slot) {
            const float weight = renormalise && topk > 1 ? chosen[slot] / denominator : chosen[slot];
            chosen[slot] = __fmul_rn(weight, scaling);
        }
        // Ascending expert order; the experts are distinct, so the order is total.
        for (int slot = 1; slot < topk; ++slot) {
            const int expert = picked[slot];
            const float weight = chosen[slot];
            int at = slot;
            for (; at > 0 && picked[at - 1] > expert; --at) {
                picked[at] = picked[at - 1];
                chosen[at] = chosen[at - 1];
            }
            picked[at] = expert;
            chosen[at] = weight;
        }
        for (int slot = 0; slot < topk; ++slot) {
            indices[(unsigned long long)b * topk + slot] = picked[slot];
            weights[(unsigned long long)b * topk + slot] = chosen[slot];
        }
    }
}

// LayerNorm with a BF16 weight and bias over rows of `columns` (linalg::layer_norm: the biased
// variance, eps under the root): row r of `x` into row r of `out`. grid (rows), Q4E_BLOCK threads.
extern "C" __global__ void q4e_layer_norm(const float* __restrict__ x,
                                          const __nv_bfloat16* __restrict__ weight,
                                          const __nv_bfloat16* __restrict__ bias,
                                          float* __restrict__ out, int columns, float eps) {
    OPT_PDL_ENTRY();
    __shared__ float scratch[Q4E_BLOCK];
    const float* source = x + (unsigned long long)blockIdx.x * columns;
    float* destination = out + (unsigned long long)blockIdx.x * columns;
    float sum = 0.0f;
    for (int index = threadIdx.x; index < columns; index += blockDim.x) {
        sum = __fadd_rn(sum, source[index]);
    }
    const float mean = q4e_block_sum(sum, scratch) / (float)columns;
    float squares = 0.0f;
    for (int index = threadIdx.x; index < columns; index += blockDim.x) {
        const float delta = __fadd_rn(source[index], -mean);
        squares = __fadd_rn(squares, __fmul_rn(delta, delta));
    }
    const float variance = q4e_block_sum(squares, scratch) / (float)columns;
    const float scale = 1.0f / sqrtf(__fadd_rn(variance, eps));
    for (int index = threadIdx.x; index < columns; index += blockDim.x) {
        const float delta = __fadd_rn(source[index], -mean);
        destination[index] = __fadd_rn(
            __fmul_rn(__fmul_rn(delta, scale), __bfloat162float(weight[index])),
            __bfloat162float(bias[index]));
    }
}

// The longest pool GLM's indexer softmaxes over in registers.
#define Q4E_GLM_MAX_POOL 16

// GLM's k-pool: when the token at `position` completes a pool of `pool` tokens, channel c of the
// pooled key is the softmax over the pool's tokens of gate + ape (BF16 [pool][dim]) weighting
// their keys, summed in token order (forward.rs's Indexer::select). `keys` and `gates` are the
// layer's per-position caches ([positions][dim]); the pooled key lands at row
// (position + 1) / pool - 1 of `pooled`. A token that completes no pool is a no-op, so a step
// launches this whatever its position. grid (positions): block b pools for `position` + b, threads
// over the channels; a pool reads only cached keys and gates, so a run's pools are one launch.
extern "C" __global__ void q4e_glm_pool(const float* __restrict__ keys,
                                        const float* __restrict__ gates,
                                        const __nv_bfloat16* __restrict__ ape,
                                        float* __restrict__ pooled, int dim, int pool,
                                        int position) {
    OPT_PDL_ENTRY();
    position += blockIdx.x;
    if (pool <= 0 || pool > Q4E_GLM_MAX_POOL || (position + 1) % pool != 0) {
        return;
    }
    const int first = position + 1 - pool;
    float* destination = pooled + (unsigned long long)((position + 1) / pool - 1) * dim;
    for (int channel = threadIdx.x; channel < dim; channel += blockDim.x) {
        float weight[Q4E_GLM_MAX_POOL];
        float maximum = -INFINITY;
#pragma unroll
        for (int offset = 0; offset < Q4E_GLM_MAX_POOL; ++offset) {
            if (offset < pool) {
                weight[offset] =
                    __fadd_rn(gates[(unsigned long long)(first + offset) * dim + channel],
                              __bfloat162float(ape[offset * dim + channel]));
                maximum = fmaxf(maximum, weight[offset]);
            }
        }
        float total = 0.0f;
#pragma unroll
        for (int offset = 0; offset < Q4E_GLM_MAX_POOL; ++offset) {
            if (offset < pool) {
                weight[offset] = expf(weight[offset] - maximum);
                total = __fadd_rn(total, weight[offset]);
            }
        }
        float accumulated = 0.0f;
#pragma unroll
        for (int offset = 0; offset < Q4E_GLM_MAX_POOL; ++offset) {
            if (offset < pool) {
                const float share = !isfinite(maximum) ? 0.0f
                                    : total > 0.0f    ? weight[offset] / total
                                                      : weight[offset];
                accumulated = __fadd_rn(
                    accumulated,
                    __fmul_rn(share, keys[(unsigned long long)(first + offset) * dim + channel]));
            }
        }
        destination[channel] = accumulated;
    }
}

#define Q4E_GLM_MAX_INDEX_HEADS 32
#define Q4E_GLM_MAX_INDEX_DIM 128

// GLM's indexer score of every complete pool for the query at `position`:
// sum_h w_h max(q_h . k_pool * score_scale, 0) with w_h = head_weights[h] * head_scale, summed in
// head order (forward.rs's Indexer::select). The query heads are staged in shared memory and a
// warp scores a pool, Q4E_QSA_TILE pools a block, the layout q4e_glm_pick reads.
#define Q4E_QSA_TILE 64
// grid (ceil(pools / Q4E_QSA_TILE) of the last row, rows), Q4E_BLOCK threads: blockIdx.y is the
// row at `position` + y, its query heads and head weights the y-th of the run's and its scores
// `score_stride` floats on from the previous row's.
extern "C" __global__ void q4e_glm_index_scores(const float* __restrict__ query,
                                                const float* __restrict__ head_weights,
                                                const float* __restrict__ pooled,
                                                float* __restrict__ scores, int position,
                                                int heads, int dim, int pool, float score_scale,
                                                float head_scale, int score_stride) {
    OPT_PDL_ENTRY();
    __shared__ float staged[Q4E_GLM_MAX_INDEX_HEADS * Q4E_GLM_MAX_INDEX_DIM];
    __shared__ float weight[Q4E_GLM_MAX_INDEX_HEADS];
    const int row = blockIdx.y;
    position += row;
    query += (unsigned long long)row * heads * dim;
    head_weights += (unsigned long long)row * heads;
    scores += (unsigned long long)row * score_stride;
    const int pools = (position + 1) / pool;
    const int first = blockIdx.x * Q4E_QSA_TILE;
    if (first >= pools || heads > Q4E_GLM_MAX_INDEX_HEADS || dim > Q4E_GLM_MAX_INDEX_DIM) {
        return;
    }
    for (int index = threadIdx.x; index < heads * dim; index += blockDim.x) {
        staged[index] = query[index];
    }
    for (int head = threadIdx.x; head < heads; head += blockDim.x) {
        weight[head] = __fmul_rn(head_weights[head], head_scale);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int warps = blockDim.x >> 5;
    const int last = min(first + Q4E_QSA_TILE, pools);
    for (int block = first + (threadIdx.x >> 5); block < last; block += warps) {
        const float* key_row = pooled + (unsigned long long)block * dim;
        float key[Q4E_GLM_MAX_INDEX_DIM / 32];
#pragma unroll
        for (int step = 0; step < Q4E_GLM_MAX_INDEX_DIM / 32; ++step) {
            const int channel = lane + 32 * step;
            key[step] = channel < dim ? key_row[channel] : 0.0f;
        }
        float score = 0.0f;
        for (int head = 0; head < heads; ++head) {
            float dot = 0.0f;
#pragma unroll
            for (int step = 0; step < Q4E_GLM_MAX_INDEX_DIM / 32; ++step) {
                const int channel = lane + 32 * step;
                if (channel < dim) {
                    dot = fmaf(staged[head * dim + channel], key[step], dot);
                }
            }
            dot = q4e_warp_sum(dot);
            score = __fadd_rn(score,
                              __fmul_rn(weight[head], fmaxf(__fmul_rn(dot, score_scale), 0.0f)));
        }
        if (lane == 0) {
            scores[block] = score;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// GLM-5.3-Flash: the tiled decode GEMV wrappers this arm loads from its own image.
//
// q4e_tiled_gemv, its formats and Q4eTgRows are the shared header's; what follows is this
// arm's block-scaled format, its grouped launch and the epilogues that read it. The Qwen token
// image instantiates the same body for q4e_gemv_tiled_bf16 and q4e_gemv_tiled_e4m3, so a
// retile reaches both without a second copy here.
// ---------------------------------------------------------------------------------------------

// E4M3 with one f32 scale a 128 x 128 block (GLM-5.3-Flash's FP8 projections).
struct Q4eTgE4m3Block : Q4eTgE4m3 {
    static constexpr bool kBlockScaled = true;
};

// Q4eTgRows of an E4M3 matrix with its block scales, [ceil(rows / 128)][ceil(columns / 128)].
// `quantise` is the release's dynamic activation scheme; clear, the f32 activations multiply the
// block scale unquantised (a configuration built in code, never a release's).
struct Q4eTgBlockRows : Q4eTgRows<unsigned char> {
    const float* scales;
    int quantise;
    __device__ float block_scale(int tile, int column) const {
        const int blocks = (columns + Q4E_FP8_BLOCK - 1) / Q4E_FP8_BLOCK;
        return q4e_ldg(&scales[(tile * Q4E_TG_TILE / Q4E_FP8_BLOCK) * blocks +
                               column / Q4E_FP8_BLOCK]);
    }
};

// Q4eTgStore for a matrix of up to four parts stacked by rows, which start at 0, second, third and
// fourth (0 <= second <= third <= fourth <= rows): part p's outputs land as one [batch][its rows]
// block after the blocks of the parts before it, so each part of a fused projection is the next
// kernel's contiguous input at any batch. One token's outputs are the plain row order, and every
// boundary at `rows` is the plain row order at any batch. A grouped launch (q4e_tg_group) stores its
// group's `count` tokens from `first` on into the layout of all `batch`.
struct Q4eTgPartStore {
    float* y;
    int rows;
    int batch;
    int second;
    int third;
    int fourth;
    int first;
    int count;
    __device__ void operator()(int tile, int warp, int r, int t, float sum) const {
        const int row = tile * Q4E_TG_TILE + warp * Q4E_TG_ROWS + r;
        if (t < count && row < rows) {
            const int start = row < second ? 0 : row < third ? second : row < fourth ? third : fourth;
            const int end = row < second ? second : row < third ? third : row < fourth ? fourth : rows;
            y[(unsigned long long)batch * start +
              (unsigned long long)(first + t) * (end - start) + row - start] = sum;
        }
    }
};

// A grouped tiled launch: blockIdx.z is a group of Q4E_MAX_BATCH tokens, each group the body of a
// launch of its own over its tokens with its own split partials and tile counters (`partials` and
// `counters` hold gridDim.z of a launch's each), so a token's sum is the chain of an ungrouped
// launch - a prefill chunk runs the decode GEMV's arithmetic in one launch instead of one an
// eight rows. Moves x, partials and counters to the group's, and returns its first token and count
// (zero or less past the batch).
__device__ __forceinline__ int2 q4e_tg_group(const float* __restrict__& x,
                                             float* __restrict__& partials,
                                             unsigned int* __restrict__& counters, int columns,
                                             int batch) {
    const int group = blockIdx.z;
    const int first = group * Q4E_MAX_BATCH;
    x += (unsigned long long)first * columns;
    partials += (unsigned long long)group * gridDim.y * Q4E_MAX_BATCH * gridDim.x * Q4E_TG_TILE;
    counters += (unsigned long long)group * gridDim.x;
    return make_int2(first, min(Q4E_MAX_BATCH, batch - first));
}

// q4e_gemv_tiled_bf16 with its outputs in Q4eTgPartStore's layout: GLM-5.3-Flash's fused BF16
// projections (KDA's q, k, v, f_a, g_a and b; the indexer's wk, compress gate and weights_proj).
// Grid (ceil(rows / 32), splits, ceil(batch / Q4E_MAX_BATCH)): q4e_tg_group.
extern "C" __global__ void __launch_bounds__(Q4E_TG_THREADS, 3)
    q4e_gemv_tiled_bf16_parts(const __nv_bfloat16* __restrict__ w, const float* __restrict__ x,
                              float* __restrict__ y, int rows, int columns, int batch,
                              float* __restrict__ partials, unsigned int* __restrict__ counters,
                              int second, int third, int fourth) {
    OPT_PDL_LAUNCH();
    const int2 group = q4e_tg_group(x, partials, counters, columns, batch);
    if (batch > Q4E_MAX_BATCH * (int)gridDim.z || group.y <= 0) {
        OPT_PDL_WAIT();
        return;
    }
    q4e_tiled_gemv<Q4eTgBf16>(
        Q4eTgRows<__nv_bfloat16>{w, rows, columns},
        Q4eTgPartStore{y, rows, batch, second, third, fourth, group.x, group.y}, x, columns,
        group.y, partials, counters);
}

// y = W x_q for GLM-5.3-Flash's FP8 projections: W is E4M3 with one f32 scale a 128 x 128 block
// (weight_scale_inv, [ceil(rows / 128)][ceil(columns / 128)]) and x_q is x quantised as the
// checkpoint's dynamic activation scheme does it, per 128 columns (q4e_tg_fp8_activation), when
// `quantise` is set, and is x unquantised when it is clear. The outputs are in Q4eTgPartStore's
// layout (the MLA's fused q_a and kv_a). Grid (ceil(rows / 32), splits, ceil(batch /
// Q4E_MAX_BATCH)) as q4e_tg_group, Q4E_TG_THREADS threads, and columns a multiple of 128.
extern "C" __global__ void __launch_bounds__(Q4E_TG_THREADS, 3)
    q4e_gemv_tiled_e4m3_block(const unsigned char* __restrict__ w,
                              const float* __restrict__ scales, const float* __restrict__ x,
                              float* __restrict__ y, int rows, int columns, int batch,
                              float* __restrict__ partials, unsigned int* __restrict__ counters,
                              int quantise, int second, int third, int fourth) {
    OPT_PDL_LAUNCH();
    const int2 group = q4e_tg_group(x, partials, counters, columns, batch);
    if (batch > Q4E_MAX_BATCH * (int)gridDim.z || group.y <= 0 ||
        columns % Q4E_FP8_BLOCK != 0) {
        OPT_PDL_WAIT();
        return;
    }
    q4e_tiled_gemv<Q4eTgE4m3Block>(
        Q4eTgBlockRows{{w, rows, columns}, scales, quantise},
        Q4eTgPartStore{y, rows, batch, second, third, fourth, group.x, group.y}, x, columns,
        group.y, partials, counters);
}

// L2 prefetch of up to four weight spans (address, bytes; a zero length skips the span), each split
// into SPARK_PREFETCH_CHUNK runs over the block's threads. It runs on a side stream while the run's
// stream waits in an exchange, so the next kernels' weights arrive while DRAM would otherwise idle.
// A hint: nothing waits on it and nothing it reads is written during the run.
extern "C" __global__ void q4e_prefetch_l2(const uint64_t a0, const uint64_t n0, const uint64_t a1,
                                           const uint64_t n1, const uint64_t a2, const uint64_t n2,
                                           const uint64_t a3, const uint64_t n3) {
    const uint64_t address[4] = {a0, a1, a2, a3};
    const uint64_t length[4] = {n0, n1, n2, n3};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        for (uint64_t at = threadIdx.x * (uint64_t)SPARK_PREFETCH_CHUNK; at < length[s];
             at += blockDim.x * (uint64_t)SPARK_PREFETCH_CHUNK) {
            const uint64_t left = length[s] - at;
            spark_prefetch_l2(reinterpret_cast<const unsigned char*>(address[s]) + at,
                              left < SPARK_PREFETCH_CHUNK ? left : SPARK_PREFETCH_CHUNK);
        }
    }
}

// The threads below this one whose `flag` is set, and the block's count in `total`: a ballot a
// warp and the warps' counts in `counts` (one a warp). Every thread of the block calls it.
__device__ __forceinline__ int q4e_block_rank(bool flag, int* counts, int* total) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const unsigned int ballot = __ballot_sync(0xFFFFFFFFu, flag);
    if (lane == 0) {
        counts[warp] = __popc(ballot);
    }
    __syncthreads();
    int before = __popc(ballot & ((1u << lane) - 1u));
    int all = 0;
    for (int other = 0; other < (int)(blockDim.x >> 5); ++other) {
        before += other < warp ? counts[other] : 0;
        all += counts[other];
    }
    __syncthreads();
    *total = all;
    return before;
}

// The routed experts' block-scaled FP8 GEMV through the tiled body, over the router's [batch]
// [topk] picks: `table` holds each expert's E4M3 [rows, columns] weight and f32 [rows / 128]
// [columns / 128] scales ([experts][2] device pointers). Pair p reads x row p / topk (`x_per_slot`
// 0: gate_up on the token's normed input) or row p (1: down on the pair's activation), and writes
// y row p, times its routing weight when `slot_weights` is set - the product the host's combine
// rounds before it adds. `quantise` is q4e_gemv_tiled_e4m3_block's.
//
// The pairs that pick one expert share its weights: blockIdx.z is the z-th distinct expert in
// pair order, and its blocks run every pair that picked it (a token picks an expert once, so at
// most Q4E_MAX_BATCH) as one batch of the tiled body, so a step reads each expert once. A pair's
// sum is the chain a batch of one gives it. grid (ceil(rows / 32), 1, batch * topk), Q4E_TG_THREADS
// threads, batch * topk at most Q4E_TG_THREADS; blocks past the distinct experts return.
struct Q4eTgPairStore {
    float* y;
    int rows;
    const int* pairs;
    const float* slot_weights;
    int count;
    __device__ void operator()(int tile, int warp, int r, int t, float sum) const {
        const int row = tile * Q4E_TG_TILE + warp * Q4E_TG_ROWS + r;
        if (t < count && row < rows) {
            const int pair = pairs[t];
            y[(unsigned long long)pair * rows + row] =
                slot_weights != nullptr ? __fmul_rn(sum, slot_weights[pair]) : sum;
        }
    }
};

extern "C" __global__ void __launch_bounds__(Q4E_TG_THREADS, 3)
    q4e_expert_gemv_e4m3_block(const unsigned long long* __restrict__ table,
                               const int* __restrict__ indices,
                               const float* __restrict__ slot_weights,
                               const float* __restrict__ x, float* __restrict__ y, int rows,
                               int columns, int topk, int x_per_slot, int quantise) {
    OPT_PDL_ENTRY();
    __shared__ int first_of[Q4E_MAX_EXPERTS];
    __shared__ int counts[Q4E_TG_WARPS];
    __shared__ int chosen;
    __shared__ int group_pairs[Q4E_MAX_BATCH];
    __shared__ int group_rows[Q4E_MAX_BATCH];
    const int pairs = gridDim.z;
    if (columns % Q4E_FP8_BLOCK != 0 || topk <= 0 || pairs > Q4E_TG_THREADS) {
        return;
    }
    for (int expert = threadIdx.x; expert < Q4E_MAX_EXPERTS; expert += blockDim.x) {
        first_of[expert] = 0x7FFFFFFF;
    }
    const int pair = threadIdx.x;
    int expert = pair < pairs ? indices[pair] : -1;
    expert = expert < Q4E_MAX_EXPERTS ? expert : -1;
    __syncthreads();
    if (expert >= 0) {
        atomicMin(&first_of[expert], pair);
    }
    __syncthreads();
    int groups;
    const bool opens = expert >= 0 && first_of[expert] == pair;
    const int group = q4e_block_rank(opens, counts, &groups);
    if (opens && group == (int)blockIdx.z) {
        chosen = expert;
    }
    __syncthreads();
    if ((int)blockIdx.z >= groups) {
        return;
    }
    const int mine = chosen;
    int count;
    const bool member = expert == mine;
    const int slot = q4e_block_rank(member, counts, &count);
    if (member && slot < Q4E_MAX_BATCH) {
        group_pairs[slot] = pair;
        group_rows[slot] = x_per_slot ? pair : pair / topk;
    }
    __syncthreads();
    const unsigned char* w = reinterpret_cast<const unsigned char*>(table[2 * mine]);
    const float* scales = reinterpret_cast<const float*>(table[2 * mine + 1]);
    count = min(count, Q4E_MAX_BATCH);
    q4e_tiled_gemv<Q4eTgE4m3Block>(Q4eTgBlockRows{{w, rows, columns}, scales, quantise},
                                   Q4eTgPairStore{y, rows, group_pairs, slot_weights, count}, x,
                                   columns, count, nullptr, nullptr, group_rows);
}

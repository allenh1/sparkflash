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

// kernels/q4e_gemv_sm121.cuh
//
// The tiled decode GEMV every Qwen4-Exp and GLM image runs, the weight formats it specialises
// over, and the block selection the decode and indexer kernels share: q4e_tiled_gemv and the
// BF16 / E4M3 / block-scaled formats it takes, q4e_tg_fma8, q4e_tg_fold, q4e_e4m3x2,
// q4e_tg_fp8_activation, and q4e_score_key with q4e_block_scan and q4e_pick_pools. They were
// copied into up to three translation units (qwen4_exp, qwen4_attn, glm_decode) and the copies
// were byte-identical. q4e_sigmoid_ref moved on to q4e_common_sm121.cuh, where the shared SwiGLU
// epilogue can reach it without depending on this header.
//
// The geometry the body is tuned to - eight warps, four rows a warp, eight tokens - is a
// property of the shapes it reads, so it is stated once here: a retile is one edit instead of
// copies that drift, and a token's sum is one arithmetic chain in every image rather than four
// answers to the same question. Everything below is an inline device template, as the copies
// were, so no image gains or loses an exported symbol.

#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "opt_pdl_sm121.cuh"
#include "q4e_common_sm121.cuh"

// The shared xor butterfly, under the name the body used.
#define q4e_tg_fold q4e_warp_sum_all

// Maximum tokens processed in one launch (a layer-major prefill chunk): dense BF16 weights are
// read once per chunk while every per-token accumulator stays in registers with no stack spill.
#define Q4E_MAX_BATCH 8

// ---------------------------------------------------------------------------------------------
// One token MoE router: softmax, sigmoid or sqrt-softplus scores, an optional noaux_tc
// selection bias, and norm_topk/routed_scaling. Qwen's q4e_router and GLM's q4e_noaux_router
// are thin instantiations, so a selection rule has one statement instead of two that can drift.
// ---------------------------------------------------------------------------------------------
#define Q4E_ROUTER_SOFTMAX 0
#define Q4E_ROUTER_SIGMOID 1
#define Q4E_ROUTER_SQRT_SOFTPLUS 2

// `kScoreKind` picks the score: a softmax over the logits (Qwen), or sigmoid / sqrt-softplus of
// each logit independently (GLM's noaux_tc and DeepSeek's form). `kHasBias` adds the
// selection bias to the ranking signal only; `kAscending` writes the slots in ascending expert
// order (GLM's host combine) rather than in pick order (Qwen's). `kWriteProbabilities` stores
// every score (Qwen's probe and its full row) before the top-k weights overwrite the first
// slots of the same buffer.
//
// The softmax form keeps the block-wide order q4e_router has always used; the independent
// forms keep q4e_noaux_router's one-warp form, so both images' bits are unchanged.
template <int kScoreKind, bool kHasBias, bool kAscending, bool kWriteProbabilities,
          int kMaxExperts, int kMaxTopk, int kBlock>
__device__ __forceinline__ void shared_moe_router_body(
    const float* __restrict__ logits, const float* __restrict__ bias,
    float* __restrict__ probabilities, int* __restrict__ indices, float* __restrict__ weights,
    int experts, int topk, int renormalise, float scaling) {
    constexpr int kEach = kMaxExperts / 32;
    static_assert(kEach <= 32, "a lane's taken experts are one word's bits");
    if (experts <= 0 || experts > kMaxExperts || topk <= 0 || topk > kMaxTopk || topk > experts) {
        return;
    }
    __shared__ float probability[kMaxExperts];
    __shared__ float scratch[kBlock];
    __shared__ int scratch_index[kBlock];
    __shared__ int picked[kMaxTopk];
    __shared__ float chosen[kMaxTopk];
    const int b = blockIdx.x;
    const float* logits_b = logits + (unsigned long long)b * experts;
    float* probs_b =
        probabilities != nullptr ? probabilities + (unsigned long long)b * experts : nullptr;
    int* indices_b = indices + (unsigned long long)b * topk;
    const int thread = threadIdx.x;

    if constexpr (kScoreKind == Q4E_ROUTER_SOFTMAX) {
        // The softmax and the block-wide top-k, in the order q4e_router has always used.
        float local_max = -INFINITY;
        for (int expert = thread; expert < experts; expert += blockDim.x) {
            local_max = fmaxf(local_max, logits_b[expert]);
        }
        const float maximum = q4e_block_max(local_max, scratch);
        float local_sum = 0.0f;
        for (int expert = thread; expert < experts; expert += blockDim.x) {
            const float value = expf(logits_b[expert] - maximum);
            probability[expert] = value;
            local_sum += value;
        }
        const float sum = q4e_block_sum(local_sum, scratch);
        for (int expert = thread; expert < experts; expert += blockDim.x) {
            const float value = probability[expert] / sum;
            probability[expert] = value;
            if constexpr (kWriteProbabilities) probs_b[expert] = value;
        }
        __syncthreads();
        for (int slot = 0; slot < topk; ++slot) {
            float best_value = -2.0f;
            int best = 0x7FFFFFFF;
            for (int expert = thread; expert < experts; expert += blockDim.x) {
                if (probability[expert] > -0.5f) {
                    q4e_take_better(probability[expert], expert, &best_value, &best);
                }
            }
            q4e_block_best(&best_value, &best, scratch, scratch_index);
            if (thread == 0) {
                picked[slot] = best;
                chosen[slot] = best_value;
                probability[best] = -1.0f;
            }
            __syncthreads();
        }
        if (thread == 0) {
            float normaliser = 0.0f;
            for (int slot = 0; slot < topk; ++slot) normaliser += chosen[slot];
            if (normaliser <= 0.0f) normaliser = 1.0f;
            for (int slot = 0; slot < topk; ++slot) {
                const float route = chosen[slot] / normaliser;
                if constexpr (kWriteProbabilities) {
                    probs_b[slot] = route;
                } else {
                    weights[(unsigned long long)b * topk + slot] = route;
                }
                indices_b[slot] = picked[slot];
            }
        }
        return;
    }

    if (thread >= 32) return;
    const int lane = thread;
    // Every load first, so they are in flight together.
    float score[kEach];
    float signal[kEach];
#pragma unroll
    for (int k = 0; k < kEach; ++k) {
        const int expert = lane + 32 * k;
        const float raw = expert < experts ? logits_b[expert] : 0.0f;
        if constexpr (kScoreKind == Q4E_ROUTER_SIGMOID) {
            score[k] = expert < experts ? q4e_sigmoid_ref(raw) : 0.0f;
        } else {
            const float softplus = raw > 20.0f ? raw : log1pf(expf(raw));
            score[k] = expert < experts ? sqrtf(softplus) : 0.0f;
        }
        signal[k] = kHasBias && expert < experts ? bias[expert] : 0.0f;
    }
#pragma unroll
    for (int k = 0; k < kEach; ++k) {
        if (lane + 32 * k < experts) signal[k] = __fadd_rn(score[k], signal[k]);
    }
    unsigned int taken = 0;
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
        const float winner = __shfl_sync(0xFFFFFFFFu, mine, best & 31);
        // Every lane holds the same winner; one store keeps the shared arrays race-free.
        if (lane == 0) {
            picked[slot] = best;
            chosen[slot] = winner;
        }
    }
    if (lane == 0) {
        float total = 0.0f;
        for (int slot = 0; slot < topk; ++slot) total = __fadd_rn(total, chosen[slot]);
        const float denominator = __fadd_rn(total, 1e-20f);
        for (int slot = 0; slot < topk; ++slot) {
            const float weight = renormalise && topk > 1 ? chosen[slot] / denominator : chosen[slot];
            chosen[slot] = __fmul_rn(weight, scaling);
        }
        if constexpr (kAscending) {
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
        }
        for (int slot = 0; slot < topk; ++slot) {
            indices_b[slot] = picked[slot];
            weights[(unsigned long long)b * topk + slot] = chosen[slot];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Tiled decode GEMV: four weight rows a warp, the activations staged in shared memory.
// ---------------------------------------------------------------------------------------------
//
// The one-warp-a-row GEMV qwen4_exp keeps reads every token's whole activation row again for each output
// row. At one token that row sits in L1; at a verify's eight it is 80 KiB of a hyper stream
// (320 KiB in all) and comes from L2 once per output, so the hyper passes and the 6,144-wide
// output projections fall to half the weight bandwidth past four tokens.
//
// Here a block owns a tile of 32 weight rows, four for each warp. The inner dimension goes in
// rounds of 256 columns: lane l reads the 16 bytes at column 8l of each of its warp's four rows
// (512 contiguous bytes of a row per load, as the one-row GEMV reads them), and the round's
// activations for every token are staged in shared memory once for the block, so a token's row
// crosses L2 once per block and each staged value serves four rows. A matrix with fewer tiles
// than the GPU wants blocks splits its rounds across gridDim.y blocks, and the tile's last block
// to finish adds the splits.
//
// The same body reads the drafter's E4M3 matrices: sixteen one-byte weights to a lane's 16-byte
// chunk, so a round is 512 columns, and the row's scale applied once to the finished sum.
//
// It also reads GLM-5.3-Flash's FP8 projections, E4M3 with one scale a 128 x 128 block, whose
// activations the checkpoint quantises per 128 columns. A tile's 32 rows lie in one band of
// blocks, so as a warp stages its run of 128 columns it quantises them, scales them back and
// multiplies them by the band's block scale; the weights then fold as the drafter's do.
//
// Every token's sum is one chain whatever the batch: per lane, an fmaf a weight in ascending
// column order over the block's rounds; the lanes folded by a fixed xor butterfly; the split
// partials added in split order by whichever block of the tile arrives last. The split count is
// the host's function of the shape alone (tiled_gemv_splits), so a verify row is bit-identical
// to the one-token step.
#define Q4E_TG_WARPS 8
#define Q4E_TG_THREADS (Q4E_TG_WARPS * 32)
#define Q4E_TG_ROWS 4
#define Q4E_TG_TILE (Q4E_TG_WARPS * Q4E_TG_ROWS)

// acc + the eight BF16 weights of `word4` against lo (columns 0..4) and hi (4..8).
__device__ __forceinline__ float q4e_tg_fma8(const uint4 word4, const float4 lo, const float4 hi,
                                             float acc) {
    acc = fmaf(__uint_as_float(word4.x << 16), lo.x, acc);
    acc = fmaf(__uint_as_float(word4.x & 0xFFFF0000u), lo.y, acc);
    acc = fmaf(__uint_as_float(word4.y << 16), lo.z, acc);
    acc = fmaf(__uint_as_float(word4.y & 0xFFFF0000u), lo.w, acc);
    acc = fmaf(__uint_as_float(word4.z << 16), hi.x, acc);
    acc = fmaf(__uint_as_float(word4.z & 0xFFFF0000u), hi.y, acc);
    acc = fmaf(__uint_as_float(word4.w << 16), hi.z, acc);
    acc = fmaf(__uint_as_float(word4.w & 0xFFFF0000u), hi.w, acc);
    return acc;
}

// Two E4M3 codes (the low byte first) widened through f16, which holds every code exactly.
__device__ __forceinline__ float2 q4e_e4m3x2(unsigned int pair) {
    const __half2_raw raw =
        __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(pair & 0xFFFFu), __NV_E4M3);
    return __half22float2(__half2(raw));
}

// The weight formats the tiled body reads. A lane takes one 16-byte chunk of each of its rows a
// round, `kColumns` weights, against `x`: the chunk's staged activations, four to a float4.
// `kBlockScaled` formats stage their activations through q4e_tg_fp8_activation.
struct Q4eTgBf16 {
    using Weight = __nv_bfloat16;
    static constexpr int kColumns = 8;
    static constexpr bool kBlockScaled = false;
    __device__ static float fma(const uint4 word4, const float4* x, float acc) {
        return q4e_tg_fma8(word4, x[0], x[1], acc);
    }
};

struct Q4eTgE4m3 {
    using Weight = unsigned char;
    static constexpr int kColumns = 16;
    static constexpr bool kBlockScaled = false;
    __device__ static float fma(const uint4 word4, const float4* x, float acc) {
        const unsigned int words[4] = {word4.x, word4.y, word4.z, word4.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 low = q4e_e4m3x2(words[i]);
            const float2 high = q4e_e4m3x2(words[i] >> 16);
            acc = fmaf(low.x, x[i].x, acc);
            acc = fmaf(low.y, x[i].y, acc);
            acc = fmaf(high.x, x[i].z, acc);
            acc = fmaf(high.y, x[i].w, acc);
        }
        return acc;
    }
};

#define Q4E_FP8_BLOCK 128
static_assert(Q4E_FP8_BLOCK % Q4E_TG_TILE == 0, "a tile's rows lie in one band of blocks");

// GLM-5.3-Flash's dynamic activation quantisation of one run of 128 columns, a warp's four values
// a lane, as its host forward does it (quant::quantize_activation_fp8_tiles, then the dequant):
// the run's largest magnitude over 448 is its scale, each value over the scale rounds to the
// nearest E4M3 code, and the code times the scale is the value the matmul takes; a run whose
// scale is zero is zero. `block` is the weights' scale for these columns, applied after.
__device__ __forceinline__ float4 q4e_tg_fp8_activation(const float4 value, const float block) {
    float largest =
        fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)), fmaxf(fabsf(value.z), fabsf(value.w)));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        largest = fmaxf(largest, __shfl_xor_sync(0xFFFFFFFFu, largest, offset));
    }
    const float scale = largest / 448.0f;
    if (scale == 0.0f) {
        return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const unsigned int codes[4] = {
        __nv_cvt_float_to_fp8(value.x / scale, __NV_SATFINITE, __NV_E4M3),
        __nv_cvt_float_to_fp8(value.y / scale, __NV_SATFINITE, __NV_E4M3),
        __nv_cvt_float_to_fp8(value.z / scale, __NV_SATFINITE, __NV_E4M3),
        __nv_cvt_float_to_fp8(value.w / scale, __NV_SATFINITE, __NV_E4M3)};
    const float2 low = q4e_e4m3x2(codes[0] | codes[1] << 8);
    const float2 high = q4e_e4m3x2(codes[2] | codes[3] << 8);
    return make_float4(low.x * scale * block, low.y * scale * block, high.x * scale * block,
                       high.y * scale * block);
}


// The body every tiled entry shares. `row_of(tile, warp, r)` names row r of the warp's four
// (nullptr past the matrix). Lane l of each warp then holds the sum of row l / 8 for token l % 8,
// and `finish(tile, warp, r, t, sum)` is called by the whole warp (lanes whose token is past the
// batch included, with t >= batch), so an epilogue may exchange values across it. `partials`
// holds gridDim.y x Q4E_MAX_BATCH x (gridDim.x * 32) floats and `counters` gridDim.x zeros when
// gridDim.y > 1; the tile's last block puts its counter back to zero. columns is a whole number
// of the format's chunks (of Q4E_FP8_BLOCK for a block-scaled one, whose `row_of` also has
// `block_scale(tile, column)`), and the rows and x are 16-byte aligned. Token t reads row t of x,
// or row x_rows[t] when there is a row map.
//
// The weights and their scales are constant, so an entry whose `row_of` reads nothing an earlier
// kernel writes may open with OPT_PDL_LAUNCH() alone: the body issues its first round's weight
// loads and only then waits, before it reads x, the row map or anything else. (An entry that
// waited already, as the routed experts' must before they read their picks, loses nothing.)
template <typename Format, typename RowOf, typename Finish>
__device__ __forceinline__ void q4e_tiled_gemv(const RowOf& row_of, const Finish& finish,
                                               const float* __restrict__ x, int columns,
                                               int batch, float* __restrict__ partials,
                                               unsigned int* __restrict__ counters,
                                               const int* x_rows = nullptr) {
    constexpr int kQuads = Format::kColumns / 4;
    constexpr int kRound = Format::kColumns * 32;
    // [token][quad][lane] float4s: quad j of lane l is its chunk's columns 4j..4j+4, so a warp's
    // reads of any quad are consecutive 16-byte words.
    __shared__ __align__(16) float4 staged[Q4E_MAX_BATCH * kQuads * 32];
    __shared__ int last;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int splits = gridDim.y;
    const int rounds = (columns + kRound - 1) / kRound;
    const int share = (rounds + splits - 1) / splits;
    const int begin = blockIdx.y * share;
    const int end = min(rounds, begin + share);
    const typename Format::Weight* rows[Q4E_TG_ROWS];
#pragma unroll
    for (int r = 0; r < Q4E_TG_ROWS; ++r) {
        rows[r] = row_of(tile, warp, r);
    }

    float acc[Q4E_TG_ROWS][Q4E_MAX_BATCH];
#pragma unroll
    for (int r = 0; r < Q4E_TG_ROWS; ++r) {
#pragma unroll
        for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
            acc[r][t] = 0.0f;
        }
    }
    for (int round = begin; round < end; ++round) {
        const int base = round * kRound;
        const int width = min(kRound, columns - base);
        const bool in = lane * Format::kColumns < width;
        // The round's weights are in flight while its activations are staged.
        uint4 w[Q4E_TG_ROWS];
#pragma unroll
        for (int r = 0; r < Q4E_TG_ROWS; ++r) {
            if (in && rows[r] != nullptr) {
                w[r] = q4e_ldg(
                    reinterpret_cast<const uint4*>(rows[r] + base + lane * Format::kColumns));
            }
        }
        if (round == begin) {
            // The first round's weights stream while the previous kernel finishes.
            OPT_PDL_WAIT();
        }
        const int vectors = width >> 2;
        __syncthreads();
        for (int index = threadIdx.x; index < batch * vectors; index += Q4E_TG_THREADS) {
            const int t = index / vectors;
            const int v = index - t * vectors;
            const unsigned long long x_row = x_rows != nullptr ? x_rows[t] : t;
            float4 value = q4e_ldg(
                reinterpret_cast<const float4*>(x + x_row * columns + base) + v);
            if constexpr (Format::kBlockScaled) {
                // vectors is a multiple of 32, so a warp holds one token's whole run.
                const float block = row_of.block_scale(tile, base + v * 4);
                value = row_of.quantise ? q4e_tg_fp8_activation(value, block)
                                        : make_float4(value.x * block, value.y * block,
                                                      value.z * block, value.w * block);
            }
            staged[(t * kQuads + v % kQuads) * 32 + v / kQuads] = value;
        }
        __syncthreads();
        if (in) {
#pragma unroll
            for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
                if (t < batch) {
                    float4 quads[kQuads];
#pragma unroll
                    for (int j = 0; j < kQuads; ++j) {
                        quads[j] = staged[(t * kQuads + j) * 32 + lane];
                    }
#pragma unroll
                    for (int r = 0; r < Q4E_TG_ROWS; ++r) {
                        if (rows[r] != nullptr) {
                            acc[r][t] = Format::fma(w[r], quads, acc[r][t]);
                        }
                    }
                }
            }
        }
    }
    // A block with no rounds waits here, before it writes; for the others it is a no-op.
    OPT_PDL_WAIT();
    // Lane l keeps row l / 8, token l % 8.
    const int mine_r = lane >> 3;
    const int mine_t = lane & 7;
    float sum = 0.0f;
#pragma unroll
    for (int r = 0; r < Q4E_TG_ROWS; ++r) {
#pragma unroll
        for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
            if (t < batch) {
                const float folded = q4e_tg_fold(acc[r][t]);
                if (r == mine_r && t == mine_t) {
                    sum = folded;
                }
            }
        }
    }
    if (splits > 1) {
        const unsigned long long slots = (unsigned long long)gridDim.x * Q4E_TG_TILE;
        const unsigned long long slot =
            (unsigned long long)tile * Q4E_TG_TILE + warp * Q4E_TG_ROWS + mine_r;
        if (mine_t < batch) {
            partials[((unsigned long long)blockIdx.y * Q4E_MAX_BATCH + mine_t) * slots + slot] =
                sum;
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            last = atomicAdd(&counters[tile], 1u) == (unsigned int)(splits - 1);
        }
        __syncthreads();
        if (!last) {
            return;
        }
        __threadfence();
        if (mine_t < batch) {
            sum = 0.0f;
            for (int split = 0; split < splits; ++split) {
                sum += __ldcg(&partials[((unsigned long long)split * Q4E_MAX_BATCH + mine_t) *
                                            slots +
                                        slot]);
            }
        }
        if (threadIdx.x == 0) {
            counters[tile] = 0u;
        }
    }
    finish(tile, warp, mine_r, mine_t, sum);
}

// Rows tile * 32 + warp * 4 + r of a [rows, columns] matrix.
template <typename Weight>
struct Q4eTgRows {
    const Weight* w;
    int rows;
    int columns;
    __device__ const Weight* operator()(int tile, int warp, int r) const {
        const int row = tile * Q4E_TG_TILE + warp * Q4E_TG_ROWS + r;
        return row < rows ? w + (unsigned long long)row * columns : nullptr;
    }
};

// ---------------------------------------------------------------------------------------------
// The indexer's block selection: the score key, the block scan, and the top-budget emission.
// ---------------------------------------------------------------------------------------------
#define Q4E_MAX_BUDGET 1024

// A score as an unsigned key in the float order: -0 keys as +0, as the float compare has them
// equal, and negative scores below every positive one.
__device__ __forceinline__ unsigned int q4e_score_key(float score) {
    unsigned int bits = __float_as_uint(score);
    if (bits == 0x80000000u) {
        bits = 0u;
    }
    return (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u;
}

// Exclusive prefix sum of one int a thread over the block, with the block's total in *total;
// scratch holds 32 ints. Reusable back to back.
__device__ int q4e_block_scan(int value, int* scratch, int* total) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = (blockDim.x + 31) >> 5;
    int inclusive = value;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xFFFFFFFFu, inclusive, offset);
        if (lane >= offset) {
            inclusive += other;
        }
    }
    if (lane == 31) {
        scratch[warp] = inclusive;
    }
    __syncthreads();
    if (warp == 0) {
        int running = lane < warps ? scratch[lane] : 0;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int other = __shfl_up_sync(0xFFFFFFFFu, running, offset);
            if (lane >= offset) {
                running += other;
            }
        }
        scratch[lane] = running;
    }
    __syncthreads();
    const int before = (warp > 0 ? scratch[warp - 1] : 0) + inclusive - value;
    *total = scratch[warps - 1];
    __syncthreads();
    return before;
}

// The top-budget selection and the emission, from the scores above.
//
// blockIdx.x is the token of the chunk, so one launch does the whole chunk. The kept set is the
// budget best blocks, a higher score first and on a tie the lower block: what a heap fed the
// blocks in order keeps, since a later block never displaces an equal one. A radix select over
// the scores' keys finds the budget-th best key a byte at a time, so the set takes four passes
// over the scores instead of a single thread's walk through the heap (300 ms of a 4,096-row
// forward at 8k context). Each thread then walks a contiguous run of blocks, so the kept blocks
// are emitted ascending with two block scans and no sort.
//
// `tail_alone` is whether a token that keeps no block still attends over its tail: Qwen's
// reference attends over nothing before the first whole block, GLM-5.3-Flash's over the tail.
__device__ __forceinline__ void q4e_pick_pools(const float* scores, int* selected, int* count,
                                               int position, int ratio, int budget,
                                               int score_stride, int select_stride,
                                               bool tail_alone) {
    __shared__ unsigned int histogram[256];
    __shared__ int scan_scratch[32];
    __shared__ unsigned int chosen_prefix;
    __shared__ int chosen_remaining;
    const int token_index = blockIdx.x;
    const int token_position = position + token_index;
    const float* scores_t = scores + (unsigned long long)token_index * score_stride;
    int* selected_t = selected + (unsigned long long)token_index * select_stride;
    const int blocks = (token_position + 1) / ratio;
    const int thread = threadIdx.x;
    int keep = budget < blocks ? budget : blocks;
    if (keep > Q4E_MAX_BUDGET) {
        keep = Q4E_MAX_BUDGET;
    }
    if (keep <= 0) {
        const int tail = tail_alone ? token_position + 1 - blocks * ratio : 0;
        for (int index = thread; index < tail; index += blockDim.x) {
            selected_t[index] = blocks * ratio + index;
        }
        if (thread == 0) {
            count[token_index] = tail;
        }
        return;
    }
    // Every block past the budget keeps all; otherwise the keys above `threshold` and the first
    // `ties` keys equal to it, in block order.
    const bool all = blocks <= keep;
    unsigned int threshold = 0u;
    int ties = 0;
    if (!all) {
        unsigned int prefix = 0u;
        unsigned int mask = 0u;
        int remaining = keep;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int bin = thread; bin < 256; bin += blockDim.x) {
                histogram[bin] = 0u;
            }
            __syncthreads();
            for (int block = thread; block < blocks; block += blockDim.x) {
                const unsigned int key = q4e_score_key(scores_t[block]);
                if ((key & mask) == prefix) {
                    atomicAdd(&histogram[(key >> shift) & 255u], 1u);
                }
            }
            __syncthreads();
            if (thread == 0) {
                int above = 0;
                int bin = 255;
                for (; bin > 0; --bin) {
                    const int here = (int)histogram[bin];
                    if (above + here >= remaining) {
                        break;
                    }
                    above += here;
                }
                chosen_prefix = prefix | ((unsigned int)bin << shift);
                chosen_remaining = remaining - above;
            }
            __syncthreads();
            prefix = chosen_prefix;
            remaining = chosen_remaining;
            mask |= 255u << shift;
            __syncthreads();
        }
        threshold = prefix;
        ties = remaining;
    }
    const int per = (blocks + blockDim.x - 1) / blockDim.x;
    const int lo = thread * per < blocks ? thread * per : blocks;
    const int hi = lo + per < blocks ? lo + per : blocks;
    int equal = 0;
    if (!all) {
        for (int block = lo; block < hi; ++block) {
            equal += q4e_score_key(scores_t[block]) == threshold ? 1 : 0;
        }
    }
    int equal_total = 0;
    const int equal_before = q4e_block_scan(equal, scan_scratch, &equal_total);
    int kept_here = 0;
    {
        int tie = equal_before;
        for (int block = lo; block < hi; ++block) {
            bool kept = all;
            if (!all) {
                const unsigned int key = q4e_score_key(scores_t[block]);
                kept = key > threshold || (key == threshold && tie++ < ties);
            }
            kept_here += kept ? 1 : 0;
        }
    }
    int kept_total = 0;
    int write_at = q4e_block_scan(kept_here, scan_scratch, &kept_total);
    {
        int tie = equal_before;
        for (int block = lo; block < hi; ++block) {
            bool kept = all;
            if (!all) {
                const unsigned int key = q4e_score_key(scores_t[block]);
                kept = key > threshold || (key == threshold && tie++ < ties);
            }
            if (kept) {
                for (int token = 0; token < ratio; ++token) {
                    selected_t[write_at * ratio + token] = block * ratio + token;
                }
                ++write_at;
            }
        }
    }
    const int tail_start = blocks * ratio;
    const int tail = token_position + 1 - tail_start;
    for (int index = thread; index < tail; index += blockDim.x) {
        selected_t[keep * ratio + index] = tail_start + index;
    }
    if (thread == 0) {
        count[token_index] = keep * ratio + (tail > 0 ? tail : 0);
    }
}

// ---------------------------------------------------------------------------------------------
// The sparse indexer's pool and score bodies, beside q4e_pick_pools. Qwen's q4e_qsa_pool /
// q4e_qsa_scores and GLM's q4e_glm_pool / q4e_glm_index_scores are thin instantiations: a pool
// is one expression and a score is one selection rule, and the two arms had written each twice.
// ---------------------------------------------------------------------------------------------

// One cached indexer key as f32: Qwen caches BF16 keys, GLM caches f32 keys.
template <typename CacheT>
__device__ __forceinline__ float q4e_indexer_key(const CacheT value) {
    if constexpr (sizeof(CacheT) == 2) {
        return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&value));
    } else {
        return value;
    }
}

// One indexer pool's key. kGated=false is Qwen's form: the mean of the pool's BF16 keys,
// RMS-normed by k_weight and rotated at the pool's first position. kGated=true is GLM's k-pool:
// a softmax over the pool's tokens of gate + ape, weighting their f32 keys in token order.
// `keys` is already at the pool's first token; `destination` at the pool's row.
template <typename CacheT, bool kGated>
__device__ __forceinline__ void q4e_indexer_pool_body(
    const CacheT* __restrict__ keys, const float* __restrict__ gates,
    const __nv_bfloat16* __restrict__ ape, const __nv_bfloat16* __restrict__ k_weight,
    float* __restrict__ destination, int dim, int pool, int rotary, float theta, float eps,
    int first) {
    __shared__ float pooled[128];
    __shared__ float scratch[128];
    constexpr int kGatedMax = 16;
    // The shared arrays hold 128 channels and the block reduction one float a thread: a wider
    // block, or one narrower than the head, would write past them or leave a channel unwritten.
    // The gated form uses no block reduction, so GLM's 256-thread launch is still in bounds.
    if (dim <= 0 || dim > 128 || blockDim.x < (unsigned)dim) return;
    if constexpr (!kGated) {
        if (blockDim.x > 128) return;
    }
    const int channel = threadIdx.x;
    const bool live = channel < dim;
    if constexpr (!kGated) {
        float value = 0.0f;
        if (live) {
            float sum = 0.0f;
            for (int token = 0; token < pool; ++token) {
                sum += q4e_indexer_key(keys[(unsigned long long)token * dim + channel]);
            }
            value = sum / (float)pool;
        }
        const float scale = rsqrtf(q4e_block_sum(value * value, scratch) / (float)dim + eps);
        if (live) {
            pooled[channel] =
                value * scale * (1.0f + __bfloat162float(k_weight[channel]));
        }
        __syncthreads();
        const int half = rotary < dim ? rotary / 2 : dim / 2;
        if (channel < half) {
            const float frequency = powf(theta, -2.0f * (float)channel / (float)rotary);
            const float angle = (float)first * frequency;
            const float cosine = __cosf(angle);
            const float sine = __sinf(angle);
            const float a = pooled[channel];
            const float b = pooled[channel + half];
            pooled[channel] = a * cosine - b * sine;
            pooled[channel + half] = a * sine + b * cosine;
        }
        __syncthreads();
        if (live) destination[channel] = pooled[channel];
        return;
    }
    if (!live) return;
    float weight[kGatedMax];
    float maximum = -INFINITY;
#pragma unroll
    for (int offset = 0; offset < kGatedMax; ++offset) {
        if (offset < pool) {
            weight[offset] = __fadd_rn(
                gates[(unsigned long long)offset * dim + channel],
                __bfloat162float(ape[offset * dim + channel]));
            maximum = fmaxf(maximum, weight[offset]);
        }
    }
    float total = 0.0f;
#pragma unroll
    for (int offset = 0; offset < kGatedMax; ++offset) {
        if (offset < pool) {
            weight[offset] = expf(weight[offset] - maximum);
            total = __fadd_rn(total, weight[offset]);
        }
    }
    float accumulated = 0.0f;
#pragma unroll
    for (int offset = 0; offset < kGatedMax; ++offset) {
        if (offset < pool) {
            const float share = !isfinite(maximum) ? 0.0f
                                : total > 0.0f    ? weight[offset] / total
                                                  : weight[offset];
            accumulated = __fadd_rn(
                accumulated,
                __fmul_rn(share, q4e_indexer_key(
                                     keys[(unsigned long long)offset * dim + channel])));
        }
    }
    destination[channel] = accumulated;
}

// One pool's indexer score for one lane's warp. kWeightedRelu=false is Qwen's: the sum over
// heads of max(dot, 0), divided by sqrt(dim); kWeightedRelu=true is GLM's: the sum over heads of
// w_h max(dot * score_scale, 0), the caller's head order. Both keep the dot-product order and
// rounding of the kernel they came from.
template <typename CacheT, bool kWeightedRelu>
__device__ __forceinline__ float q4e_indexer_score_body(
    const float* __restrict__ query, const float* __restrict__ head_weights,
    const CacheT* __restrict__ key_row, int heads, int dim, float score_scale,
    float head_scale, int lane) {
    float key[4];
#pragma unroll
    for (int step = 0; step < 4; ++step) {
        const int channel = lane + 32 * step;
        key[step] = channel < dim ? q4e_indexer_key(key_row[channel]) : 0.0f;
    }
    float score = 0.0f;
    for (int head = 0; head < heads; ++head) {
        const float* qh = query + (unsigned long long)head * dim;
        if constexpr (kWeightedRelu) {
            float dot = 0.0f;
#pragma unroll
            for (int step = 0; step < 4; ++step) {
                const int channel = lane + 32 * step;
                if (channel < dim) dot = fmaf(qh[channel], key[step], dot);
            }
            dot = q4e_warp_sum(dot);
            const float weight = __fmul_rn(head_weights[head], head_scale);
            score = __fadd_rn(score,
                              __fmul_rn(weight, fmaxf(__fmul_rn(dot, score_scale), 0.0f)));
        } else {
            float low = 0.0f;
            float high = 0.0f;
            if (lane < dim) {
                low = __fmul_rn(qh[lane], key[0]);
                if (lane + 64 < dim) low = __fadd_rn(low, __fmul_rn(qh[lane + 64], key[2]));
            }
            if (lane + 32 < dim) {
                high = __fmul_rn(qh[lane + 32], key[1]);
                if (lane + 96 < dim) high = __fadd_rn(high, __fmul_rn(qh[lane + 96], key[3]));
            }
            score += fmaxf(q4e_warp_sum(__fadd_rn(low, high)), 0.0f);
        }
    }
    return (kWeightedRelu ? score : score / sqrtf((float)dim));
}

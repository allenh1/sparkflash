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

// kernels/shared_attn_mma_sm121.cuh
//
// The one prompt-attention body on the tensor cores: causal grouped-query attention over a BF16
// cache, with a sliding window and a ring cache. Dory's prefill (dory_attn_mma) and Inkling's
// (inkling_attn_mma) instantiate it; each model's own arithmetic around the softmax is a policy
// the body calls at fixed points, so neither carries a second staging loop.
//
// A block is one kv head and SHARED_ATTN_Q query positions; warp w runs query head
// kv_head * group + w over those positions, so one shared K/V tile feeds every head of the group.
// The scores Q K^T and the value sum P V are mma.sync m16n8k16 BF16 products accumulated in f32:
// K and V are the cache's own BF16, while q and the softmax weights are f32 carried as a BF16
// head and the BF16 of its remainder (16 significant bits, two MMAs a fragment), as the shared
// wide tile carries every f32 activation. The online softmax is f32 with expf: a row's tiles in
// key order, a tile's maximum before its weights, and each thread's partial row sum folded
// across its quad once at the end. This is the FlashAttention-2 recurrence (Dao, "FlashAttention-2:
// Faster Attention with Better Parallelism and Work Partitioning", 2023) on the mma.sync,
// ldmatrix and cp.async primitives of opt_mma_common_sm121.cuh.
//
// Shared memory: q's head and remainder for group x SHARED_ATTN_Q rows, one tile of
// SHARED_ATTN_KEYS keys and one of values, all rows of D BF16 with their 16-byte chunks swizzled,
// then the policy's own bytes (`extra`, 16-byte aligned). The values for a tile land while its
// scores run, and the next tile's keys while its values are summed.
// Grid (ceil(tokens / SHARED_ATTN_Q), kv_heads), group x 32 threads.
//
// The policy (a struct the entry builds from its arguments) supplies:
//   keys(extra, kv_head, base, last)     cp.async of its per-key data for keys [base, base +
//                                        SHARED_ATTN_KEYS) into `extra`, in the key tile's group
//   values(extra, kv_head, base, last)   the same, in the value tile's group
//   start(head, qbase, count_q, pos0, g, t4)
//                                        a thread's own setup for its rows g and g + 8
//   open_max(head, qbase, count_q, row), open_sum(head, qbase, count_q, row, t4)
//                                        the softmax's opening maximum and this thread's share
//                                        of the opening denominator for block row `row`
//   tile(extra, warp, pos0, base, g, t4) once a key tile's scores are in, before any is read
//   score(scaled, extra, warp, n, e, g, t4)
//                                        the logit of score s[n][e] (rows g + 8 (e / 2), key
//                                        8 n + 2 t4 + e % 2), given the dot over sqrt(D)
//   weight(p, extra, n, e, t4)           what P V multiplies key (n, e)'s value row by, given its
//                                        softmax weight p (the denominator sums p itself)
//   store(token, head, n, t4, v0, v1, denominator)
//                                        output dims 8 n + 2 t4 and + 1 of (token, head)
// A policy that leaves a hook as the identity compiles to the body without it.

#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "opt_mma_bf16_sm121.cuh"
#include "opt_mma_common_sm121.cuh"

#define SHARED_ATTN_Q 16
#define SHARED_ATTN_KEYS 32

// The byte offset of 16-byte chunk `chunk` of row `row` in a tile of D-wide BF16 rows, the chunks
// swizzled by the row's low bits (as many as a row has chunks, up to three).
template <int D>
__device__ __forceinline__ uint32_t shared_attn_off(uint32_t row, uint32_t chunk) {
    constexpr uint32_t mask = D / 8 >= 8 ? 7u : (uint32_t)(D / 8 - 1);
    return row * (uint32_t)(D * 2) + ((chunk ^ (row & mask)) << 4);
}

// One tile of SHARED_ATTN_KEYS cache rows of kv head `kv_head` from absolute position `base`, the
// rows past `last` zero-filled.
template <int D>
__device__ __forceinline__ void shared_attn_load_tile(uint32_t tile, const __nv_bfloat16* cache,
                                                      int base, int last, int cache_rows,
                                                      unsigned long long kv_row, int kv_head) {
    constexpr int CH = D / 8;
    for (int index = threadIdx.x; index < SHARED_ATTN_KEYS * CH; index += blockDim.x) {
        const int row = index / CH;
        const int chunk = index - row * CH;
        const int key = base + row;
        const bool valid = key <= last;
        const int row_index = cache_rows > 0 ? key % cache_rows : key;
        const __nv_bfloat16* source =
            valid ? cache + (unsigned long long)row_index * kv_row +
                        (unsigned long long)kv_head * D + chunk * 8
                  : cache;
        om_cp16(tile + shared_attn_off<D>(row, chunk), source, valid);
    }
}

template <int D, class Policy>
__device__ __forceinline__ void shared_attn_mma_body(
    const float* __restrict__ q, const __nv_bfloat16* __restrict__ keys,
    const __nv_bfloat16* __restrict__ values, int position, int tokens, int heads, int kv_heads,
    int window, int q_stride, int cache_rows, Policy& policy) {
    constexpr int CH = D / 8;   // 16-byte chunks of a row, and n-tiles of a head's output
    constexpr int KS = D / 16;  // k-steps of a score
    constexpr int ROW = D * 2;  // bytes of a BF16 row
    extern __shared__ __align__(128) uint8_t shared_attn_smem[];

    const int group = heads / kv_heads;
    const int kv_head = blockIdx.y;
    // The latest positions see the most keys: they go first.
    const int qbase = (gridDim.x - 1 - blockIdx.x) * SHARED_ATTN_Q;
    const int count_q = min(SHARED_ATTN_Q, tokens - qbase);
    const int pos0 = position + qbase;
    const int last = pos0 + count_q - 1;
    const int first = (window > 0 && pos0 + 1 > window) ? pos0 + 1 - window : 0;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int head = kv_head * group + warp;
    const unsigned long long kv_row = (unsigned long long)kv_heads * D;
    const float score_scale = rsqrtf((float)D);

    const uint32_t q_hi = om_smem(shared_attn_smem);
    const uint32_t q_lo = q_hi + group * SHARED_ATTN_Q * ROW;
    const uint32_t k_tile = q_lo + group * SHARED_ATTN_Q * ROW;
    const uint32_t v_tile = k_tile + SHARED_ATTN_KEYS * ROW;
    const uint32_t extra = v_tile + SHARED_ATTN_KEYS * ROW;

    shared_attn_load_tile<D>(k_tile, keys, first, last, cache_rows, kv_row, kv_head);
    policy.keys(extra, kv_head, first, last);
    om_commit();

    // q as BF16 head and remainder, rows (warp, position) of the group.
    for (int index = threadIdx.x; index < group * SHARED_ATTN_Q * CH; index += blockDim.x) {
        const int row = index / CH;
        const int chunk = index - row * CH;
        const int t = row % SHARED_ATTN_Q;
        const int row_head = kv_head * group + row / SHARED_ATTN_Q;
        float x[8];
        if (t < count_q) {
            const float4* source = reinterpret_cast<const float4*>(
                q + (unsigned long long)(qbase + t) * q_stride +
                (unsigned long long)row_head * D + chunk * 8);
            const float4 a = source[0];
            const float4 b = source[1];
            x[0] = a.x; x[1] = a.y; x[2] = a.z; x[3] = a.w;
            x[4] = b.x; x[5] = b.y; x[6] = b.z; x[7] = b.w;
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                x[e] = 0.0f;
            }
        }
        uint32_t hi[4];
        uint32_t lo[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const __nv_bfloat16 h0 = __float2bfloat16_rn(x[2 * e]);
            const __nv_bfloat16 h1 = __float2bfloat16_rn(x[2 * e + 1]);
            hi[e] = ob_pack(h0, h1);
            lo[e] = ob_pack(__float2bfloat16_rn(x[2 * e] - __bfloat162float(h0)),
                            __float2bfloat16_rn(x[2 * e + 1] - __bfloat162float(h1)));
        }
        const uint32_t offset = shared_attn_off<D>(row, chunk);
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(q_hi + offset), "r"(hi[0]),
                     "r"(hi[1]), "r"(hi[2]), "r"(hi[3]));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(q_lo + offset), "r"(lo[0]),
                     "r"(lo[1]), "r"(lo[2]), "r"(lo[3]));
    }

    const int g = lane >> 2;
    const int t4 = lane & 3;
    const int matrix = lane >> 3;
    const int mrow = lane & 7;
    const uint32_t a_row = (uint32_t)(warp * SHARED_ATTN_Q + (lane & 15));
    const uint32_t a_chunk = (uint32_t)(lane >> 4);

    float o[CH][4];
#pragma unroll
    for (int n = 0; n < CH; ++n) {
        o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    }
    policy.start(head, qbase, count_q, pos0, g, t4);
    float m0 = policy.open_max(head, qbase, count_q, g),
          m1 = policy.open_max(head, qbase, count_q, g + 8);
    float l0 = policy.open_sum(head, qbase, count_q, g, t4),
          l1 = policy.open_sum(head, qbase, count_q, g + 8, t4);

    for (int base = first; base <= last; base += SHARED_ATTN_KEYS) {
        shared_attn_load_tile<D>(v_tile, values, base, last, cache_rows, kv_row, kv_head);
        policy.values(extra, kv_head, base, last);
        om_commit();
        om_wait<1>();
        __syncthreads();

        float s[4][4];
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
        }
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            uint32_t ah[4], al[4];
            const uint32_t a_offset = shared_attn_off<D>(a_row, 2u * ks + a_chunk);
            om_ldsm4(q_hi + a_offset, ah[0], ah[1], ah[2], ah[3]);
            om_ldsm4(q_lo + a_offset, al[0], al[1], al[2], al[3]);
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                uint32_t b0, b1, b2, b3;
                om_ldsm4(k_tile + shared_attn_off<D>(16u * np + 8u * (matrix >> 1) + mrow,
                                                     2u * ks + (matrix & 1)),
                         b0, b1, b2, b3);
                ob_mma(s[2 * np], ah, b0, b1);
                ob_mma(s[2 * np], al, b0, b1);
                ob_mma(s[2 * np + 1], ah, b2, b3);
                ob_mma(s[2 * np + 1], al, b2, b3);
            }
        }
        policy.tile(extra, warp, pos0, base, g, t4);

        // Scale and mask: thread rows g and g + 8, keys 8n + 2 t4 + {0, 1}.
        float x0 = -INFINITY, x1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = g + 8 * (e >> 1);
                const int key = base + 8 * n + 2 * t4 + (e & 1);
                const int query = pos0 + row;
                const bool live = row < count_q && key <= query &&
                                  (window == 0 || key + window > query);
                s[n][e] = live ? policy.score(s[n][e] * score_scale, extra, warp, n, e, g, t4)
                               : -INFINITY;
            }
            x0 = fmaxf(x0, fmaxf(s[n][0], s[n][1]));
            x1 = fmaxf(x1, fmaxf(s[n][2], s[n][3]));
        }
        x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1));
        x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
        x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1));
        x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
        const float next0 = fmaxf(m0, x0);
        const float next1 = fmaxf(m1, x1);
        // A row with no live key yet keeps zero weights and a zero state.
        const float shift0 = next0 == -INFINITY ? 0.0f : next0;
        const float shift1 = next1 == -INFINITY ? 0.0f : next1;
        const float rescale0 = expf(m0 - shift0);
        const float rescale1 = expf(m1 - shift1);
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            s[n][0] = expf(s[n][0] - shift0);
            s[n][1] = expf(s[n][1] - shift0);
            s[n][2] = expf(s[n][2] - shift1);
            s[n][3] = expf(s[n][3] - shift1);
            sum0 += s[n][0] + s[n][1];
            sum1 += s[n][2] + s[n][3];
        }
        l0 = l0 * rescale0 + sum0;
        l1 = l1 * rescale1 + sum1;
        m0 = next0;
        m1 = next1;
#pragma unroll
        for (int n = 0; n < CH; ++n) {
            o[n][0] *= rescale0;
            o[n][1] *= rescale0;
            o[n][2] *= rescale1;
            o[n][3] *= rescale1;
        }

        om_wait<0>();
        __syncthreads();
        if (base + SHARED_ATTN_KEYS <= last) {
            shared_attn_load_tile<D>(k_tile, keys, base + SHARED_ATTN_KEYS, last, cache_rows,
                                     kv_row, kv_head);
            policy.keys(extra, kv_head, base + SHARED_ATTN_KEYS, last);
        }
        om_commit();

#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            uint32_t ph[4], pl[4];
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const float* p = s[2 * kk + half];
#pragma unroll
                for (int pair = 0; pair < 2; ++pair) {
                    const float w0 = policy.weight(p[2 * pair], extra, 2 * kk + half, 2 * pair, t4);
                    const float w1 =
                        policy.weight(p[2 * pair + 1], extra, 2 * kk + half, 2 * pair + 1, t4);
                    const __nv_bfloat16 h0 = __float2bfloat16_rn(w0);
                    const __nv_bfloat16 h1 = __float2bfloat16_rn(w1);
                    ph[2 * half + pair] = ob_pack(h0, h1);
                    pl[2 * half + pair] = ob_pack(__float2bfloat16_rn(w0 - __bfloat162float(h0)),
                                                  __float2bfloat16_rn(w1 - __bfloat162float(h1)));
                }
            }
#pragma unroll
            for (int c = 0; c < CH / 2; ++c) {
                uint32_t b0, b1, b2, b3;
                om_ldsm4_t(v_tile + shared_attn_off<D>(16u * kk + 8u * (matrix & 1) + mrow,
                                                       2u * c + (matrix >> 1)),
                           b0, b1, b2, b3);
                ob_mma(o[2 * c], ph, b0, b1);
                ob_mma(o[2 * c], pl, b0, b1);
                ob_mma(o[2 * c + 1], ph, b2, b3);
                ob_mma(o[2 * c + 1], pl, b2, b3);
            }
        }
        __syncthreads();
    }
    om_wait<0>();

    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int row = g + 8 * half;
        if (row >= count_q) {
            continue;
        }
        const float denominator = half == 0 ? l0 : l1;
        const unsigned long long token = (unsigned long long)(qbase + row);
#pragma unroll
        for (int n = 0; n < CH; ++n) {
            policy.store(token, head, n, t4, o[n][2 * half], o[n][2 * half + 1], denominator);
        }
    }
}

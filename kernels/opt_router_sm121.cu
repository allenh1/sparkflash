// kernels/opt_router_sm121.cu
//
// The decode router in one launch: scores and top-k for up to OPT_R_MAX_TOKENS tokens.
//
// Stage one is warp-per-expert with split K: lane l reads 8 bf16 of the gate row at k = 8 * (l + 32 i),
// so a warp reads 512 contiguous bytes per step, and the 32 lane partials are folded with a xor
// butterfly. That changes the summation order against the serial-K reference (a reassociation of an
// f32 dot), not the operation set: score = sqrt(softplus(dot)), the biased signal picks, the
// unbiased score is normalised (sum in selection order + 1e-20) and scaled by 1.5.
//
// Stage two (top-k, one warp per token) runs in the last block to finish (threadfence + atomic
// ticket), so the router is one launch and the ticket is reset by that block for the next launch
// (graph-replay safe).

#include "opt_pdl_sm121.cuh"
#include <cuda_bf16.h>
#include <stdint.h>

#define OPT_R_HIDDEN 5120u
#define OPT_R_WARPS 8u
#define OPT_R_THREADS (OPT_R_WARPS * 32u)
#define OPT_R_MAX_TOKENS 8u
#define OPT_R_ROUTE_SCALE 1.5f
#define OPT_R_NORM_EPS 1e-20f

static_assert(OPT_R_HIDDEN % 256u == 0u, "a warp step is 256 values");

__device__ __forceinline__ float opt_r_score(float value) {
    const float sp = value > 20.0f ? value : log1pf(expf(value));
    return sqrtf(sp);
}

__device__ __forceinline__ void opt_r_bf16x8(const uint4 v, float out[8]) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[2 * i] = __uint_as_float(w[i] << 16);
        out[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}

template <uint32_t OPT_R_EXPERTS, uint32_t OPT_R_TOPK>
__device__ __forceinline__ void opt_router_body(
    const __nv_bfloat16 *__restrict__ x,     // [tokens][5120]
    const __nv_bfloat16 *__restrict__ gate,  // [experts][5120]
    const float *__restrict__ bias,          // [experts]
    uint32_t *__restrict__ ids,              // [tokens][topk]
    float *__restrict__ weights,             // [tokens][topk]
    float *__restrict__ scores,              // [tokens][experts]
    const uint32_t tokens,
    uint32_t *__restrict__ ticket) {
    static_assert(OPT_R_EXPERTS % OPT_R_WARPS == 0u, "whole blocks of experts");
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t expert = blockIdx.x * OPT_R_WARPS + warp;
    const uint32_t live = tokens < OPT_R_MAX_TOKENS ? tokens : OPT_R_MAX_TOKENS;

    {
        float acc[OPT_R_MAX_TOKENS];
#pragma unroll
        for (uint32_t t = 0; t < OPT_R_MAX_TOKENS; ++t) acc[t] = 0.0f;
        const uint4 *g = reinterpret_cast<const uint4 *>(gate + (uint64_t)expert * OPT_R_HIDDEN);
#pragma unroll 4
        for (uint32_t i = 0; i < OPT_R_HIDDEN / 256u; ++i) {
            const uint32_t v = lane + 32u * i;
            float wf[8];
            opt_r_bf16x8(__ldg(g + v), wf);
#pragma unroll
            for (uint32_t t = 0; t < OPT_R_MAX_TOKENS; ++t) {
                if (t < live) {
                    float xf[8];
                    opt_r_bf16x8(__ldg(reinterpret_cast<const uint4 *>(x + (uint64_t)t * OPT_R_HIDDEN) + v), xf);
#pragma unroll
                    for (int j = 0; j < 8; ++j) acc[t] = __fmaf_rn(xf[j], wf[j], acc[t]);
                }
            }
        }
#pragma unroll
        for (uint32_t t = 0; t < OPT_R_MAX_TOKENS; ++t) {
            if (t < live) {
                float s = acc[t];
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
                if (lane == 0u) scores[(uint64_t)t * OPT_R_EXPERTS + expert] = opt_r_score(s);
            }
        }
    }

    __shared__ uint32_t s_last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0u) {
        s_last = atomicAdd(ticket, 1u) == gridDim.x - 1u ? 1u : 0u;
    }
    __syncthreads();
    if (s_last == 0u) return;
    __threadfence();

    // One warp per token: lane l holds experts l, l + 32, ... in registers, and each pick is a warp
    // argmax (ties keep the lower id, the reference's strict > scan), so tokens finish in parallel
    // and no pick waits on a block barrier.
    static_assert(OPT_R_EXPERTS % 32u == 0u, "whole warps of experts");
    constexpr uint32_t PER_LANE = OPT_R_EXPERTS / 32u;
    for (uint32_t t = warp; t < live; t += OPT_R_WARPS) {
        float sc[PER_LANE], sig[PER_LANE];
#pragma unroll
        for (uint32_t j = 0; j < PER_LANE; ++j) {
            const uint32_t e = lane + 32u * j;
            sc[j] = __ldcg(scores + (uint64_t)t * OPT_R_EXPERTS + e);
            sig[j] = __fadd_rn(sc[j], bias[e]);
        }
        uint32_t chosen[OPT_R_TOPK];
        float picked[OPT_R_TOPK];
#pragma unroll
        for (uint32_t k = 0; k < OPT_R_TOPK; ++k) {
            float bv = sig[0];
            uint32_t bi = lane;
#pragma unroll
            for (uint32_t j = 1; j < PER_LANE; ++j) {
                if (sig[j] > bv) { bv = sig[j]; bi = lane + 32u * j; }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            float own = 0.0f;
#pragma unroll
            for (uint32_t j = 0; j < PER_LANE; ++j) {
                if (lane + 32u * j == bi) { own = sc[j]; sig[j] = -INFINITY; }
            }
            chosen[k] = bi;
            picked[k] = __shfl_sync(0xffffffffu, own, bi & 31u);
        }
        if (lane == 0u) {
            float sum = 0.0f;
#pragma unroll
            for (uint32_t s = 0; s < OPT_R_TOPK; ++s) sum = __fadd_rn(sum, picked[s]);
            const float den = __fadd_rn(sum, OPT_R_NORM_EPS);
#pragma unroll
            for (uint32_t s = 0; s < OPT_R_TOPK; ++s) {
                ids[t * OPT_R_TOPK + s] = chosen[s];
                weights[t * OPT_R_TOPK + s] = __fmul_rn(picked[s] / den, OPT_R_ROUTE_SCALE);
            }
        }
    }
    if (threadIdx.x == 0u) *ticket = 0u;
}

#define OPT_ROUTER_ENTRY(name, EXPERTS, TOPK)                                                      \
    extern "C" __global__ void __launch_bounds__(OPT_R_THREADS) name(                              \
        const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ gate,              \
        const float *__restrict__ bias, uint32_t *__restrict__ ids, float *__restrict__ weights,  \
        float *__restrict__ scores, const uint32_t tokens, uint32_t *__restrict__ ticket) {       \
        OPT_PDL_ENTRY();                                                                          \
        opt_router_body<EXPERTS, TOPK>(x, gate, bias, ids, weights, scores, tokens, ticket);      \
    }

OPT_ROUTER_ENTRY(opt_router_sm121, 384u, 6u)
// The DSpark stages: 128 experts, top-3, the backbone's score function and route scale.
OPT_ROUTER_ENTRY(opt_router_e128k3_sm121, 128u, 3u)

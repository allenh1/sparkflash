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

// kernels/moe_router_sm121.cu
//
// The device MoE router of roadmap 6.3 item 4 (task-37): the W4A8 gate projection in BF16, the
// released sqrtsoftplus scoring, the noaux_tc selection bias, top-6 and norm_topk_prob. It is the
// first Phase 3 kernel that is not a variant of an M4 kernel, so its ABI is pinned by
// src/router_abi.rs (a separate, non-frozen module) and not by m4_abi.rs.
//
// What the reference does (docs/architecture/03-moe.md section 2, from model.py:809-827):
//
//   scores = linear(x.float(), weight.float()) / gate_temp        # gate_temp = 1.0
//   scores = F.softplus(scores).sqrt()                            # sqrtsoftplus
//   indices = (scores + bias).topk(6, dim=-1)[1]                  # SELECTION uses biased scores
//   weights = scores.gather(1, indices)                           # WEIGHTING uses raw scores
//   weights = weights / (weights.sum(-1) + 1e-20)                 # norm_topk_prob
//   weights = weights * 1.5                                       # route_scale
//
// This kernel is the RELEASED CONFIGURATION, not a general router: 384 experts, 5120 values,
// top-6, gate_temp 1.0, softplus threshold 20, 1e-20 norm epsilon and 1.5 route scale are
// compile-time constants with static_asserts. A caller that needs different values needs a
// different kernel; a runtime field here would be a route by which the released arithmetic could
// silently change.
//
// Arithmetic order is part of the contract, because the CPU twin in src/router_ref.rs is compared
// to the engine's host path bit for bit:
//
//   * one output score per (token, expert) is accumulated in ASCENDING k over 0..5119 with one
//     rounding per multiply and one per add (__fmul_rn then __fadd_rn, never an FMA), which is
//     exactly what the engine's linalg::matmul does for this geometry (2.0M multiplies is below
//     its thread-pool gate, and the serial loop is `acc += x[col] * w[col]`);
//   * the score function is the engine's: z > 20 ? z : exp(z).ln_1p(), then sqrt;
//   * selection is by scores[e] + bias[e], strict > scanning e ascending, so a tie keeps the LOWER
//     expert id (the engine's rule; model.py uses torch.topk, whose tie rule the engine already
//     does not reproduce and does not claim to);
//   * the normalisation sums the raw scores in selection order (descending selection signal), adds
//     1e-20, divides, then multiplies by the route scale - two roundings per weight, in that order.
//
// The gate weight is BF16 and the activations are BF16 (the device path hands this kernel the
// post-attention hidden in the storage format the mHC quantiser already produced); both widen to
// f32 exactly, so the GEMM operands are the same f32 values the host reference uses after its
// `x.float()` / `weight.float()` casts. No weight decoding happens here beyond that exact widening.
//
// Geometry: stage one is a grid of (384 / 8 = 48, ceil(tokens / 8)) blocks of 256 threads. One
// warp owns one expert; lanes 0..7 of the warp own one token each and run that token's 5,120-value
// dot serially in ascending k, so a score is the incumbent serial sequence of f32 roundings, not a
// tree over lane partials. The eight token lanes share one gate-row read per k. Stage two is a
// separate launch with one thread per token: it re-scans the 384 unbiased scores with the 6 passes
// of the top-6 scan and writes the ids and weights. Nothing about the result depends on the launch
// order: every score is one deterministic serial sequence, and the selection is a deterministic
// scan.
//
// A NULL scores pointer is not accepted: scores are an output of this kernel (the harness reads
// them back and compares them against the CPU twin inside the declared score bar), and a caller
// that does not need them passes a scratch allocation. Making it optional would turn an ABI
// mismatch into a silent no-op.
//
// Verification status: compiled by nvcc 13.0.3 for sm_121a (the committed baseline records the
// cubin length, sha256 and PTX membar count; docs/archive/build/phase3-router.md records the window recipe
// and the Lead's measured runs, and states that the pre-softplus GEMM is not separately asserted),
// parsed by the clang frontend through the stub headers on this host, and its ABI and constants are
// pinned by src/router_abi.rs. The measured comparison class is device-versus-twin, with the host
// path bit-identical to the twin (asserted in examples/m9_router.rs); there is no separate
// device-versus-host comparison, and no device number is claimed by this file.

#include "m4_common_sm121.cuh"

// The released router geometry and constants (docs/architecture/03-moe.md section 1 and 2).
#define SPARK_ROUTER_N_ROUTED 384u
#define SPARK_ROUTER_HIDDEN SPARK_M4_HIDDEN
#define SPARK_ROUTER_TOP_K 6u
#define SPARK_ROUTER_GATE_TEMP 1.0f
#define SPARK_ROUTER_SOFTPLUS_THRESHOLD 20.0f
#define SPARK_ROUTER_NORM_EPS 1e-20f
#define SPARK_ROUTER_ROUTE_SCALE 1.5f
// One warp owns one expert, and lanes 0..7 of that warp own one token each. Every active lane
// accumulates its token's whole 5,120-value dot in ASCENDING k with one f32 accumulator and the
// __fmul_rn / __fadd_rn pair, which is the engine's serial acc += x[col] * w[col] order. The
// lanes are not a tree: a lane that owns a token never exchanges partial sums with another lane,
// so a score is bit-identical to the incumbent serial kernel. Token-block grid.y lets the same
// 3.93 MB of gate weight serve eight tokens per launch: the gate value is loaded once per k per
// expert and reused by the eight token lanes. The 48 blocks cover the 384 experts and give the
// gate GEMM enough blocks to fill the GPU while keeping one expert's row in one cache window.
#define SPARK_ROUTER_EXPERTS_PER_BLOCK 8u
#define SPARK_ROUTER_THREADS 256u
#define SPARK_ROUTER_LANES 32u
#define SPARK_ROUTER_WARPS (SPARK_ROUTER_THREADS / SPARK_ROUTER_LANES)
#define SPARK_ROUTER_TOKENS_PER_BLOCK 8u
// The hidden width divides into whole MX blocks of 32, the quantiser's own block size; stage one
// no longer tiles the reduction, but the geometry constant stays pinned.
#define SPARK_ROUTER_K_TILE 32u
// Stage two is a 256-thread block with one thread per token.
#define SPARK_ROUTER_TOPK_THREADS 256u

static_assert(SPARK_ROUTER_N_ROUTED == 384u, "the released router has 384 experts");
static_assert(SPARK_ROUTER_HIDDEN == 5120u, "the released hidden width is 5120");
static_assert(SPARK_ROUTER_TOP_K == 6u, "the released router activates 6 experts");
static_assert(SPARK_ROUTER_TOP_K <= SPARK_ROUTER_N_ROUTED, "top-k fits the expert count");
static_assert(SPARK_ROUTER_EXPERTS_PER_BLOCK == SPARK_ROUTER_WARPS,
              "one warp owns one expert of the block");
static_assert(SPARK_ROUTER_N_ROUTED % SPARK_ROUTER_EXPERTS_PER_BLOCK == 0u,
              "the expert count divides into whole blocks");
static_assert(SPARK_ROUTER_THREADS == SPARK_ROUTER_WARPS * SPARK_ROUTER_LANES,
              "the block is whole warps");
static_assert(SPARK_ROUTER_TOKENS_PER_BLOCK <= SPARK_ROUTER_LANES,
              "one lane can own one token of the block");
static_assert(SPARK_ROUTER_HIDDEN % SPARK_ROUTER_K_TILE == 0u, "the reduction tiles evenly");
static_assert(SPARK_ROUTER_K_TILE == SPARK_M4_QUANT_BLOCK, "a K tile is one quantiser block");

// The softplus threshold is the reference's (model.py:816, PyTorch F.softplus(beta=1, threshold=20)).
__device__ __forceinline__ float spark_router_softplus(float value) {
    if (value > SPARK_ROUTER_SOFTPLUS_THRESHOLD) {
        return value;
    }
    return log1pf(expf(value));
}

__device__ __forceinline__ float spark_router_score(float value) {
    return sqrtf(spark_router_softplus(value / SPARK_ROUTER_GATE_TEMP));
}

extern "C" __global__ void moe_router_sm121(
    const __nv_bfloat16 *__restrict__ x,        // [tokens][hidden] BF16 activations
    const __nv_bfloat16 *__restrict__ gate,     // [n_routed][hidden] BF16 gate weight
    const float *__restrict__ bias,             // [n_routed] F32 selection bias (text or image)
    uint32_t *__restrict__ ids,                 // [tokens][top_k] selected expert ids (stage 2)
    float *__restrict__ weights,                // [tokens][top_k] normalised weights (stage 2)
    float *__restrict__ scores,                 // [tokens][n_routed] unbiased sqrtsoftplus scores
    const uint32_t tokens,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    // Stage one: grid (384 / 8 = 48, ceil(tokens / 8)). Each warp owns one expert and lanes 0..7
    // own one token each; every active lane runs the whole 5,120-value dot in ascending k with
    // one f32 accumulator, the same one-rounding-per-multiply / one-per-add sequence as the
    // engine's serial loop. The gate row is read once per k and reused by the eight token lanes.
    // Active lane t writes scores[token][expert]; ids, bias and weights are only touched by stage
    // two and stay in the declaration so the ABI is the incumbent one.
    const uint32_t lane = threadIdx.x % SPARK_ROUTER_LANES;
    const uint32_t warp = threadIdx.x / SPARK_ROUTER_LANES;
    const uint32_t expert = blockIdx.x * SPARK_ROUTER_EXPERTS_PER_BLOCK + warp;
    const uint32_t token_base = blockIdx.y * SPARK_ROUTER_TOKENS_PER_BLOCK;
    if (expert >= SPARK_ROUTER_N_ROUTED) {
        return;
    }

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_MOE_MXFP4, t_enter);

    const uint32_t slot = lane;
    const uint32_t token = token_base + slot;
    const __nv_bfloat16 *gate_row = gate + (uint64_t)expert * SPARK_ROUTER_HIDDEN;
    if (slot < SPARK_ROUTER_TOKENS_PER_BLOCK && token < tokens) {
        const __nv_bfloat16 *x_row = x + (uint64_t)token * SPARK_ROUTER_HIDDEN;
        float acc = 0.0f;
#pragma unroll 4
        for (uint32_t k = 0u; k < SPARK_ROUTER_HIDDEN; ++k) {
            const float weight = __bfloat162float(gate_row[k]);
            const float activation = __bfloat162float(x_row[k]);
            acc = __fadd_rn(acc, __fmul_rn(activation, weight));
        }
        scores[(uint64_t)token * SPARK_ROUTER_N_ROUTED + expert] = spark_router_score(acc);
    }

    if (threadIdx.x == 0u) {
        // payload_meta: the bytes this block writes for its live tokens: the 384 unbiased scores.
        const uint32_t remaining = tokens > token_base ? tokens - token_base : 0u;
        const uint32_t live = remaining < SPARK_ROUTER_TOKENS_PER_BLOCK
                                  ? remaining
                                  : SPARK_ROUTER_TOKENS_PER_BLOCK;
        const uint32_t meta = live * SPARK_ROUTER_N_ROUTED * 4u;
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_MOE_MXFP4, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}

extern "C" __global__ void moe_router_topk_sm121(
    const float *__restrict__ bias,             // [n_routed] F32 selection bias
    const float *__restrict__ scores,           // [tokens][n_routed] stage-one scores
    uint32_t *__restrict__ ids,                 // [tokens][top_k] selected expert ids
    float *__restrict__ weights,                // [tokens][top_k] normalised weights * route_scale
    const uint32_t tokens,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    // Stage two: one thread per token, one launch. It re-scans the full 384 scores with the same
    // strict > scan as the incumbent kernel, so the tie rule (lower id) and the selection order are
    // unchanged, then applies norm_topk_prob. The separate launch is the visibility barrier for the
    // forty-eight stage-one score slices.
    const uint32_t token = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_MOE_MXFP4, t_enter);

    __shared__ float s_bias[SPARK_ROUTER_N_ROUTED];
    for (uint32_t e = threadIdx.x; e < SPARK_ROUTER_N_ROUTED; e += blockDim.x) {
        s_bias[e] = bias[e];
    }
    __syncthreads();

    if (token < tokens) {
        const float *row = scores + (uint64_t)token * SPARK_ROUTER_N_ROUTED;
        uint32_t chosen[SPARK_ROUTER_TOP_K];
        uint32_t taken_count = 0u;
        for (uint32_t slot = 0u; slot < SPARK_ROUTER_TOP_K; ++slot) {
            uint32_t best = 0u;
            float best_signal = 0.0f;
            bool have = false;
            for (uint32_t expert = 0u; expert < SPARK_ROUTER_N_ROUTED; ++expert) {
                bool already = false;
#pragma unroll
                for (uint32_t s = 0u; s < SPARK_ROUTER_TOP_K; ++s) {
                    if (s < taken_count && chosen[s] == expert) {
                        already = true;
                    }
                }
                if (already) {
                    continue;
                }
                // The selection signal is the biased score; the stored weight is the unbiased one.
                const float signal = __fadd_rn(row[expert], s_bias[expert]);
                // Strict > keeps the lower expert id on a tie, which is the engine's rule.
                if (!have || signal > best_signal) {
                    best = expert;
                    best_signal = signal;
                    have = true;
                }
            }
            chosen[taken_count++] = best;
        }

        // norm_topk_prob: sum the raw scores in selection order, add 1e-20, divide, then scale.
        float sum = 0.0f;
        for (uint32_t s = 0u; s < SPARK_ROUTER_TOP_K; ++s) {
            sum = __fadd_rn(sum, row[chosen[s]]);
        }
        const float denominator = __fadd_rn(sum, SPARK_ROUTER_NORM_EPS);
        const uint64_t out_base = (uint64_t)token * SPARK_ROUTER_TOP_K;
        for (uint32_t s = 0u; s < SPARK_ROUTER_TOP_K; ++s) {
            ids[out_base + s] = chosen[s];
            const float normalised = row[chosen[s]] / denominator;
            weights[out_base + s] = __fmul_rn(normalised, SPARK_ROUTER_ROUTE_SCALE);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0u) {
        const uint32_t block_token = blockIdx.x * blockDim.x;
        const uint32_t remaining = tokens > block_token ? tokens - block_token : 0u;
        const uint32_t live = remaining < blockDim.x ? remaining : blockDim.x;
        const uint32_t meta = live * SPARK_ROUTER_TOP_K * 8u;
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_MOE_MXFP4, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}
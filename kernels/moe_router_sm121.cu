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
// Geometry: one thread block routes SPARK_ROUTER_TOKENS_PER_BLOCK tokens. The block walks the
// 5120-value reduction in K tiles of 32, staging the tile of the gate weight in shared memory so
// that all eight tokens share one read of it, and accumulates 12 experts per thread. A second
// phase stores the 384 unbiased scores of each token in shared memory; one thread per token then
// performs the 6 passes of the top-6 scan and writes the ids and weights. Nothing about the result
// depends on the launch order: every token's score vector is produced by one deterministic
// sequence of f32 roundings, and the selection is a deterministic scan.
//
// A NULL scores pointer is not accepted: scores are an output of this kernel (the harness compares
// them against the host reference to separate a GEMM difference from a score-function difference),
// and a caller that does not need them passes a scratch allocation. Making it optional would turn
// an ABI mismatch into a silent no-op.
//
// Verification status: compiled by nvcc 13.0.3 for sm_121a (PTX and cubin recorded in
// docs/build/phase3-router.md), parsed by the clang frontend through the stub headers on this
// host, and its ABI and constants are pinned by src/router_abi.rs. The device-versus-host run is
// examples/m9_router.rs on node0; no device number is claimed by this file.

#include "m4_common_sm121.cuh"

// The released router geometry and constants (docs/architecture/03-moe.md section 1 and 2).
#define SPARK_ROUTER_N_ROUTED 384u
#define SPARK_ROUTER_HIDDEN SPARK_M4_HIDDEN
#define SPARK_ROUTER_TOP_K 6u
#define SPARK_ROUTER_GATE_TEMP 1.0f
#define SPARK_ROUTER_SOFTPLUS_THRESHOLD 20.0f
#define SPARK_ROUTER_NORM_EPS 1e-20f
#define SPARK_ROUTER_ROUTE_SCALE 1.5f
// One thread block routes eight tokens over the full 384-expert gate. THREADS / TOKENS is 32
// threads per token, and each of those 32 threads owns 12 consecutive experts, so a block holds
// 8 * 384 = 3072 accumulators in 256 threads.
#define SPARK_ROUTER_TOKENS_PER_BLOCK 8u
#define SPARK_ROUTER_THREADS 256u
#define SPARK_ROUTER_THREADS_PER_TOKEN (SPARK_ROUTER_THREADS / SPARK_ROUTER_TOKENS_PER_BLOCK)
#define SPARK_ROUTER_EXPERTS_PER_THREAD (SPARK_ROUTER_N_ROUTED / SPARK_ROUTER_THREADS_PER_TOKEN)
// The reduction walks the 5120 values in whole MX blocks of 32, the quantiser's own block size.
#define SPARK_ROUTER_K_TILE 32u

static_assert(SPARK_ROUTER_N_ROUTED == 384u, "the released router has 384 experts");
static_assert(SPARK_ROUTER_HIDDEN == 5120u, "the released hidden width is 5120");
static_assert(SPARK_ROUTER_TOP_K == 6u, "the released router activates 6 experts");
static_assert(SPARK_ROUTER_TOP_K <= SPARK_ROUTER_N_ROUTED, "top-k fits the expert count");
static_assert(SPARK_ROUTER_THREADS == SPARK_ROUTER_TOKENS_PER_BLOCK * SPARK_ROUTER_THREADS_PER_TOKEN,
              "the block is laid out as one fixed set of threads per token");
static_assert(SPARK_ROUTER_THREADS_PER_TOKEN * SPARK_ROUTER_EXPERTS_PER_THREAD == SPARK_ROUTER_N_ROUTED,
              "the block covers every expert exactly once per token");
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
    uint32_t *__restrict__ ids,                 // [tokens][top_k] selected expert ids
    float *__restrict__ weights,                // [tokens][top_k] normalised weights * route_scale
    float *__restrict__ scores,                 // [tokens][n_routed] unbiased sqrtsoftplus scores
    const uint32_t tokens,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t lane = threadIdx.x;
    const uint32_t token_in_block = lane / SPARK_ROUTER_THREADS_PER_TOKEN;
    const uint32_t expert_lane = lane % SPARK_ROUTER_THREADS_PER_TOKEN;
    const uint32_t expert_base = expert_lane * SPARK_ROUTER_EXPERTS_PER_THREAD;

    // The gate tile is staged per K tile: 384 experts x 32 values x 2 B = 24,576 B, plus the
    // activation tile (8 x 32 f32 = 1,024 B) and the score tile (8 x 384 f32 = 12,288 B): 37,888 B
    // of arrays. The cubin reports .nv.shared = 38,912 B - the extra 1 KiB is the driver's reserved
    // static shared memory - still under the 48 KiB static limit (cuobjdump -res-usage, V13.0.88).
    __shared__ __nv_bfloat16 gate_tile[SPARK_ROUTER_N_ROUTED][SPARK_ROUTER_K_TILE];
    __shared__ float act_tile[SPARK_ROUTER_TOKENS_PER_BLOCK][SPARK_ROUTER_K_TILE];
    __shared__ float score_tile[SPARK_ROUTER_TOKENS_PER_BLOCK][SPARK_ROUTER_N_ROUTED];

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_MOE_MXFP4, t_enter);

    float acc[SPARK_ROUTER_EXPERTS_PER_THREAD];
    for (uint32_t j = 0u; j < SPARK_ROUTER_EXPERTS_PER_THREAD; ++j) {
        acc[j] = 0.0f;
    }

    for (uint32_t k0 = 0u; k0 < SPARK_ROUTER_HIDDEN; k0 += SPARK_ROUTER_K_TILE) {
        // Stage the gate block: one weight row is 32 contiguous BF16 values, so consecutive
        // threads read consecutive values along K, which is the coalesced direction.
        for (uint32_t index = lane; index < SPARK_ROUTER_N_ROUTED * SPARK_ROUTER_K_TILE;
             index += SPARK_ROUTER_THREADS) {
            const uint32_t expert = index / SPARK_ROUTER_K_TILE;
            const uint32_t k = index % SPARK_ROUTER_K_TILE;
            gate_tile[expert][k] = gate[(uint64_t)expert * SPARK_ROUTER_HIDDEN + k0 + k];
        }
        // Stage the activation block. A token past the end of the batch contributes zeros, so the
        // padding tokens of the last block cannot read outside the caller's allocation.
        for (uint32_t index = lane; index < SPARK_ROUTER_TOKENS_PER_BLOCK * SPARK_ROUTER_K_TILE;
             index += SPARK_ROUTER_THREADS) {
            const uint32_t slot = index / SPARK_ROUTER_K_TILE;
            const uint32_t k = index % SPARK_ROUTER_K_TILE;
            const uint32_t slot_token = blockIdx.x * SPARK_ROUTER_TOKENS_PER_BLOCK + slot;
            act_tile[slot][k] = slot_token < tokens
                                    ? __bfloat162float(x[(uint64_t)slot_token * SPARK_ROUTER_HIDDEN + k0 + k])
                                    : 0.0f;
        }
        __syncthreads();

        // Ascending K inside the tile, one rounding per multiply and one per add, no FMA: the same
        // order and the same two roundings as the host reference's serial accumulation.
        for (uint32_t k = 0u; k < SPARK_ROUTER_K_TILE; ++k) {
            const float activation = act_tile[token_in_block][k];
            for (uint32_t j = 0u; j < SPARK_ROUTER_EXPERTS_PER_THREAD; ++j) {
                const float weight = __bfloat162float(gate_tile[expert_base + j][k]);
                acc[j] = __fadd_rn(acc[j], __fmul_rn(activation, weight));
            }
        }
        __syncthreads();
    }

    for (uint32_t j = 0u; j < SPARK_ROUTER_EXPERTS_PER_THREAD; ++j) {
        score_tile[token_in_block][expert_base + j] = spark_router_score(acc[j]);
    }
    __syncthreads();

    // The second phase re-cuts the block: lane 0..7 is one token each, because the top-6 scan is
    // per token and a serial scan cannot be split across the 32 accumulation threads of a token.
    const uint32_t slot_token = blockIdx.x * SPARK_ROUTER_TOKENS_PER_BLOCK + lane;
    if (lane < SPARK_ROUTER_TOKENS_PER_BLOCK && slot_token < tokens) {
        const float *row = score_tile[lane];
        uint32_t chosen[SPARK_ROUTER_TOP_K];
        uint32_t taken_count = 0u;
        for (uint32_t slot = 0u; slot < SPARK_ROUTER_TOP_K; ++slot) {
            uint32_t best = 0u;
            float best_signal = 0.0f;
            bool have = false;
            for (uint32_t expert = 0u; expert < SPARK_ROUTER_N_ROUTED; ++expert) {
                bool already = false;
                for (uint32_t s = 0u; s < taken_count; ++s) {
                    if (chosen[s] == expert) {
                        already = true;
                        break;
                    }
                }
                if (already) {
                    continue;
                }
                // The selection signal is the biased score; the stored weight is the unbiased one.
                const float signal = __fadd_rn(row[expert], bias[expert]);
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
        const uint64_t out_base = (uint64_t)slot_token * SPARK_ROUTER_TOP_K;
        for (uint32_t s = 0u; s < SPARK_ROUTER_TOP_K; ++s) {
            ids[out_base + s] = chosen[s];
            const float normalised = row[chosen[s]] / denominator;
            weights[out_base + s] = __fmul_rn(normalised, SPARK_ROUTER_ROUTE_SCALE);
        }
        const uint64_t scores_base = (uint64_t)slot_token * SPARK_ROUTER_N_ROUTED;
        for (uint32_t expert = 0u; expert < SPARK_ROUTER_N_ROUTED; ++expert) {
            scores[scores_base + expert] = row[expert];
        }
    }
    __syncthreads();

    if (lane == 0u) {
        // payload_meta: the bytes this block writes for a full token block: top-k ids and weights
        // plus the unbiased score vector per token.
        const uint32_t meta = SPARK_ROUTER_TOKENS_PER_BLOCK
                              * (SPARK_ROUTER_TOP_K * 8u + SPARK_ROUTER_N_ROUTED * 4u);
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_MOE_MXFP4, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}

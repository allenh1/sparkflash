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

// kernels/inkling_draft_sm121.cu
//
// Inkling-2-Small's DSpark drafter (the checkpoint's mtp.safetensors; the forward is written up in
// docs/models/inkling2-small-drafter.md). The five-layer block's projections run on the token
// image's GEMV and the arm's prompt projections, and its attention on DFlash 2's block attention
// (kernels/dflash2_sm121.cu, q4d_attn_span_part / _merge), on the drafter's own weights. This
// file holds the rest:
//
//   - inkling_draft_capture: one of the eight target taps, a stream row rounded to BF16.
//   - inkling_draft_mix_norm: a fusion row's weighted sum of a row's eight taps, RMS-normed.
//   - inkling_draft_masks: the block's mask rows.
//   - inkling_draft_round: rows rounded to BF16 in place.
//   - inkling_draft_norm / _add_round / _swiglu: the BF16 module's RMSNorm, residual add and
//     MLP activation, rounded where the reference rounds.
//   - inkling_draft_rope: the drafter's NeoX rotary.
//   - inkling_draft_context: committed rows' context keys and values into a layer's rings.
//   - inkling_draft_markov_input / _gates / _logits / _local / _last: one step of the rnn_hidden
//     Markov head, and the greedy pick across the ranks' slices of the vocabulary.
//
// Every BF16 rounding the reference makes between its BF16 tensors is made here at the same
// place, on f32 buffers that hold BF16 values: the drafter only proposes, but proposals that drift
// from the trained arithmetic are accepted less often.
//
// Attribution: the forward follows SGLang's Inkling 2 drafter (github.com/sgl-project/sglang,
// branch inkling-2-small-support at 778fb834e745bffd9a303ff388435fb44b644746, Apache-2.0,
// Copyright 2023-2024 SGLang Team: python/sglang/srt/models/inkling2_mtp.py and
// python/sglang/kernels/ops/inkling2/mtp.py); no source is copied. The method combines DFlash's
// block-diffusion drafting (Chen, Liang and Liu, ICML 2026; github.com/z-lab/dflash, MIT,
// Copyright (c) 2026 Z Lab), DFlare's adaptive layer fusion and context projections (Zhang et
// al., arXiv:2606.02091) and DSpark's semi-autoregressive RNN head (Cheng et al.,
// arXiv:2607.05147). See the root NOTICE file.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "opt_pdl_sm121.cuh"
#include "opt_sample_sm121.cuh"
#include <math_constants.h>

// The drafter's head width: one rotary pair a thread of a half-head.
#define IKD_HEAD_DIM 128
// The most hidden values one mix-norm thread holds: hidden <= blockDim.x * this.
#define IKD_MAX_PER_THREAD 32
// The logits each thread of a step-logits block scores.
#define IKD_LOGIT_PER_THREAD 8
// The floats a rank's slot of the cross-rank pick takes: (value, id, mass) and a zero, so the
// group's sum moves whole groups of four. The mass is the rank's sum of exp(logit - value).
#define IKD_PICK_SLOT 4

__device__ __forceinline__ float ikd_bf16(float value) {
    return __bfloat162float(__float2bfloat16(value));
}

// Whether (value, id) ranks above (other, other_id): the larger value, the lower id on a tie, so
// the pick is the same however the vocabulary was split.
__device__ __forceinline__ bool ikd_above(float value, int id, float other, int other_id) {
    return value > other || (value == other && id < other_id);
}

// The block's sum, the same on every thread: each warp's xor butterfly, then the warps' partials
// in warp order. blockDim.x a multiple of 32; scratch holds 32 floats and is reused back to back.
__device__ float ikd_block_sum(float value, float* scratch) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();
    float total = 0.0f;
    for (int other = 0; other < warps; ++other) {
        total += scratch[other];
    }
    return total;
}

// The block's best (value, id), the same on every thread.
__device__ void ikd_block_best(float& value, int& id, float* s_value, int* s_id) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, value, offset);
        const int other_id = __shfl_xor_sync(0xffffffffu, id, offset);
        if (ikd_above(other, other_id, value, id)) {
            value = other;
            id = other_id;
        }
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) {
        s_value[warp] = value;
        s_id[warp] = id;
    }
    __syncthreads();
    value = s_value[0];
    id = s_id[0];
    for (int other = 1; other < warps; ++other) {
        if (ikd_above(s_value[other], s_id[other], value, id)) {
            value = s_value[other];
            id = s_id[other];
        }
    }
}

// The best of the ranks' (value, id) slots after the group's sum: the logit, or under a sampled
// request's noise (`noisy`) the perturbed score in word 3.
__device__ __forceinline__ int ikd_pick(const float* __restrict__ slots, int world, int noisy) {
    const int field = noisy ? 3 : 0;
    float best = slots[field];
    int id = (int)slots[1];
    for (int rank = 1; rank < world; ++rank) {
        const float value = slots[rank * IKD_PICK_SLOT + field];
        const int other = (int)slots[rank * IKD_PICK_SLOT + 1];
        if (ikd_above(value, other, best, id)) {
            best = value;
            id = other;
        }
    }
    return id;
}

// The probability the step's softmax over every rank's logits gives its pick, from the slots
// after the group's sum: each rank's mass is relative to its own best, so it is rescaled to the
// group's best, whose own term is the pick's 1. Every rank reads the same slots and so the same
// probability, which is what lets each choose the same verify depth from it.
__device__ __forceinline__ float ikd_confidence(const float* __restrict__ slots, int world) {
    float best = slots[0];
    for (int rank = 1; rank < world; ++rank) {
        best = fmaxf(best, slots[rank * IKD_PICK_SLOT]);
    }
    float total = 0.0f;
    for (int rank = 0; rank < world; ++rank) {
        const float value = slots[rank * IKD_PICK_SLOT];
        // A rank whose ids are all masked holds no mass.
        if (value > -CUDART_INF_F) {
            total += slots[rank * IKD_PICK_SLOT + 2] * expf(value - best);
        }
    }
    return total > 0.0f ? 1.0f / total : 0.0f;
}

// One NeoX rotate-half pair of a head of `dim` at `position`, rounded to BF16: the reference's
// fp32 inverse frequency and angle, its cosine and sine cache in BF16, and an accurate sincosf
// (the fast intrinsics lose the angle at long positions).
__device__ __forceinline__ void ikd_rope_pair(float* head, int index, int dim, float position,
                                              float theta) {
    const int half = dim / 2;
    const float inverse = 1.0f / powf(theta, (float)(2 * index) / (float)dim);
    float sine;
    float cosine;
    sincosf(position * inverse, &sine, &cosine);
    sine = ikd_bf16(sine);
    cosine = ikd_bf16(cosine);
    const float a = ikd_bf16(head[index]);
    const float b = ikd_bf16(head[index + half]);
    head[index] = ikd_bf16(a * cosine - b * sine);
    head[index + half] = ikd_bf16(b * cosine + a * sine);
}

// Tap `slot` of `rows` rows: taps[row][slot] = bf16(stream[row]), the stream rows `stride`
// floats apart and a row's taps `taps_per_row * hidden` values. Grid (hidden / 256, rows).
extern "C" __global__ void inkling_draft_capture(const float* __restrict__ stream,
                                                 __nv_bfloat16* __restrict__ taps, int hidden,
                                                 int stride, int taps_per_row, int slot) {
    OPT_PDL_ENTRY();
    const int row = blockIdx.y;
    const float* from = stream + (unsigned long long)row * stride;
    __nv_bfloat16* into = taps + ((unsigned long long)row * taps_per_row + slot) * hidden;
    for (int column = blockIdx.x * blockDim.x + threadIdx.x; column < hidden;
         column += blockDim.x * gridDim.x) {
        into[column] = __float2bfloat16(from[column]);
    }
}

// out[row] = bf16(m * rsqrt(mean(m^2) + eps) * weight), m = sum over j of alpha[j] * taps[row][j]
// in tap order, in fp32: DFlare's fused feature of a row's taps under one fusion row `alpha`
// (BF16 softmax weights). Grid (rows), blockDim a multiple of 32.
extern "C" __global__ void inkling_draft_mix_norm(const __nv_bfloat16* __restrict__ taps,
                                                  const __nv_bfloat16* __restrict__ alpha,
                                                  const __nv_bfloat16* __restrict__ weight,
                                                  float* __restrict__ out, int hidden,
                                                  int taps_per_row, float eps) {
    OPT_PDL_ENTRY();
    __shared__ float scratch[32];
    if (hidden > (int)blockDim.x * IKD_MAX_PER_THREAD) {
        return;
    }
    const int row = blockIdx.x;
    const __nv_bfloat16* base = taps + (unsigned long long)row * taps_per_row * hidden;
    float mixed[IKD_MAX_PER_THREAD];
    float square = 0.0f;
#pragma unroll
    for (int i = 0; i < IKD_MAX_PER_THREAD; ++i) {
        const int column = threadIdx.x + i * blockDim.x;
        float total = 0.0f;
        if (column < hidden) {
            for (int tap = 0; tap < taps_per_row; ++tap) {
                total += __bfloat162float(alpha[tap]) *
                         __bfloat162float(base[(unsigned long long)tap * hidden + column]);
            }
        }
        mixed[i] = total;
        square += total * total;
    }
    const float inverse = rsqrtf(ikd_block_sum(square, scratch) / (float)hidden + eps);
    float* into = out + (unsigned long long)row * hidden;
#pragma unroll
    for (int i = 0; i < IKD_MAX_PER_THREAD; ++i) {
        const int column = threadIdx.x + i * blockDim.x;
        if (column < hidden) {
            into[column] = ikd_bf16(mixed[i] * inverse * __bfloat162float(weight[column]));
        }
    }
}

// Rows 1.. of the draft block take the raw mask embedding (no embedding norm). Grid
// (hidden / 256, rows - 1).
extern "C" __global__ void inkling_draft_masks(const __nv_bfloat16* __restrict__ mask,
                                               float* __restrict__ block, int hidden) {
    OPT_PDL_ENTRY();
    float* into = block + (unsigned long long)(blockIdx.y + 1) * hidden;
    for (int column = blockIdx.x * blockDim.x + threadIdx.x; column < hidden;
         column += blockDim.x * gridDim.x) {
        into[column] = __bfloat162float(mask[column]);
    }
}

// values = bf16(values), in place.
extern "C" __global__ void inkling_draft_round(float* __restrict__ values, int count) {
    OPT_PDL_ENTRY();
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += blockDim.x * gridDim.x) {
        values[index] = ikd_bf16(values[index]);
    }
}

// An RMSNorm as the reference's BF16 module computes it: each group of `group` values (one block
// a group, groups back to back) is read as BF16, normed in fp32, rounded, scaled by the BF16
// weight (one weight a group's element) and rounded again: out = bf16(bf16(x * r) * w). A row of
// the stream is one group of the hidden width; a per-head norm is a group a head. In place is
// fine. blockDim a multiple of 32.
extern "C" __global__ void inkling_draft_norm(const float* __restrict__ x,
                                              const __nv_bfloat16* __restrict__ weight,
                                              float* __restrict__ out, int group, float eps) {
    OPT_PDL_ENTRY();
    __shared__ float scratch[32];
    const unsigned long long start = (unsigned long long)blockIdx.x * group;
    float square = 0.0f;
    for (int offset = threadIdx.x; offset < group; offset += blockDim.x) {
        const float value = ikd_bf16(x[start + offset]);
        square += value * value;
    }
    const float inverse = rsqrtf(ikd_block_sum(square, scratch) / (float)group + eps);
    for (int offset = threadIdx.x; offset < group; offset += blockDim.x) {
        const float normed = ikd_bf16(ikd_bf16(x[start + offset]) * inverse);
        out[start + offset] = ikd_bf16(normed * __bfloat162float(weight[offset]));
    }
}

// A residual add of a projection's sums: dst = bf16(dst + bf16(src)).
extern "C" __global__ void inkling_draft_add_round(float* __restrict__ dst,
                                                   const float* __restrict__ src, int count) {
    OPT_PDL_ENTRY();
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += blockDim.x * gridDim.x) {
        dst[index] = ikd_bf16(dst[index] + ikd_bf16(src[index]));
    }
}

// The drafter's MLP activation, without the target's softcaps: out = bf16(bf16(silu(g)) * u),
// g and u the gate and up sums rounded to BF16.
extern "C" __global__ void inkling_draft_swiglu(const float* __restrict__ gate,
                                                const float* __restrict__ up,
                                                float* __restrict__ out, int count) {
    OPT_PDL_ENTRY();
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += blockDim.x * gridDim.x) {
        const float g = ikd_bf16(gate[index]);
        const float u = ikd_bf16(up[index]);
        const float silu = ikd_bf16(g / (1.0f + expf(-g)));
        out[index] = ikd_bf16(silu * u);
    }
}

// The drafter's rotary over a block's rows: head blockIdx.x of row blockIdx.y, rows `stride`
// floats apart and `dim` wide heads, row t at position + t. Grid (heads, rows), dim / 2 threads.
extern "C" __global__ void inkling_draft_rope(float* __restrict__ rows, int dim, int position,
                                              float theta, int stride) {
    OPT_PDL_ENTRY();
    float* head = rows + (unsigned long long)blockIdx.y * stride + blockIdx.x * dim;
    const float at = (float)(position + (int)blockIdx.y);
    for (int index = threadIdx.x; index < dim / 2; index += blockDim.x) {
        ikd_rope_pair(head, index, dim, at, theta);
    }
}

// Committed rows' context keys and values into one draft layer's rings: row r of `keys` and
// `values` ([rows][kv_heads * 128], the context projections' sums) is position + r and lands at
// ring slot (position + r) % ring. A key is rounded to BF16, normed per head by `norm` as
// inkling_draft_norm norms (bf16(bf16(x * r) * w)) and rotated at its position (rounded again); a
// value is rounded. Grid (kv_heads, rows), 128 threads.
extern "C" __global__ void inkling_draft_context(const float* __restrict__ keys,
                                                 const float* __restrict__ values,
                                                 const __nv_bfloat16* __restrict__ norm,
                                                 float* __restrict__ key_ring,
                                                 float* __restrict__ value_ring, int kv_heads,
                                                 int position, int ring, float eps, float theta) {
    OPT_PDL_ENTRY();
    __shared__ float head[IKD_HEAD_DIM];
    __shared__ float scratch[32];
    if (blockDim.x != IKD_HEAD_DIM) {
        return;
    }
    const int kv_head = blockIdx.x;
    const int row = blockIdx.y;
    const int dim = threadIdx.x;
    const int width = kv_heads * IKD_HEAD_DIM;
    const unsigned long long from = (unsigned long long)row * width + kv_head * IKD_HEAD_DIM + dim;
    const float key = ikd_bf16(keys[from]);
    const float inverse = rsqrtf(ikd_block_sum(key * key, scratch) / IKD_HEAD_DIM + eps);
    head[dim] = ikd_bf16(ikd_bf16(key * inverse) * __bfloat162float(norm[dim]));
    __syncthreads();
    if (dim < IKD_HEAD_DIM / 2) {
        ikd_rope_pair(head, dim, IKD_HEAD_DIM, (float)(position + row), theta);
    }
    __syncthreads();
    const unsigned long long slot = (unsigned long long)((position + row) % ring);
    const unsigned long long into = slot * width + kv_head * IKD_HEAD_DIM + dim;
    key_ring[into] = head[dim];
    value_ring[into] = ikd_bf16(values[from]);
}

// Same as inkling_draft_context, storing the already-BF16-rounded context keys and values in
// BF16 rings (halving drafter KV DRAM footprint and bandwidth while keeping bit-identical floats).
extern "C" __global__ void inkling_draft_context_bf16(const float* __restrict__ keys,
                                                      const float* __restrict__ values,
                                                      const __nv_bfloat16* __restrict__ norm,
                                                      __nv_bfloat16* __restrict__ key_ring,
                                                      __nv_bfloat16* __restrict__ value_ring,
                                                      int kv_heads, int position, int ring,
                                                      float eps, float theta) {
    OPT_PDL_ENTRY();
    __shared__ float head[IKD_HEAD_DIM];
    __shared__ float scratch[32];
    if (blockDim.x != IKD_HEAD_DIM) {
        return;
    }
    const int kv_head = blockIdx.x;
    const int row = blockIdx.y;
    const int dim = threadIdx.x;
    const int width = kv_heads * IKD_HEAD_DIM;
    const unsigned long long from = (unsigned long long)row * width + kv_head * IKD_HEAD_DIM + dim;
    const float key = ikd_bf16(keys[from]);
    const float inverse = rsqrtf(ikd_block_sum(key * key, scratch) / IKD_HEAD_DIM + eps);
    head[dim] = ikd_bf16(ikd_bf16(key * inverse) * __bfloat162float(norm[dim]));
    __syncthreads();
    if (dim < IKD_HEAD_DIM / 2) {
        ikd_rope_pair(head, dim, IKD_HEAD_DIM, (float)(position + row), theta);
    }
    __syncthreads();
    const unsigned long long slot = (unsigned long long)((position + row) % ring);
    const unsigned long long into = slot * width + kv_head * IKD_HEAD_DIM + dim;
    key_ring[into] = __float2bfloat16_rn(head[dim]);
    value_ring[into] = __float2bfloat16_rn(values[from]);
}

// The Markov head's step input z = [s; W1[x]; h; f] (rank, rank, hidden, hidden values), BF16
// values: x is `anchor` at step 0 and otherwise the pick of the last step's slots (`noisy` as
// ikd_pick), which block 0 records at tokens[step] with its probability at confidence[step]. h is
// the step's normed block row, rounded here; s and f already hold BF16 values. Grid (width / 256).
extern "C" __global__ void inkling_draft_markov_input(float* __restrict__ z,
                                                      const float* __restrict__ state,
                                                      const __nv_bfloat16* __restrict__ embed,
                                                      unsigned* __restrict__ tokens, int anchor,
                                                      const float* __restrict__ slots, int world,
                                                      int step, const float* __restrict__ row,
                                                      const float* __restrict__ fused, int rank,
                                                      int hidden, float* __restrict__ confidence,
                                                      int noisy) {
    OPT_PDL_ENTRY();
    int token;
    if (step == 0) {
        token = anchor;
    } else {
        token = ikd_pick(slots, world, noisy);
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            tokens[step] = (unsigned)token;
            confidence[step] = ikd_confidence(slots, world);
        }
    }
    const int width = 2 * rank + 2 * hidden;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < width;
         index += blockDim.x * gridDim.x) {
        float value;
        if (index < rank) {
            value = state[index];
        } else if (index < 2 * rank) {
            value = __bfloat162float(embed[(unsigned long long)token * rank + (index - rank)]);
        } else if (index < 2 * rank + hidden) {
            value = ikd_bf16(row[index - 2 * rank]);
        } else {
            value = fused[index - 2 * rank - hidden];
        }
        z[index] = value;
    }
}

// The step's gates from the joint projection's sums `projected` ([gate | candidate | out], rank
// each) and its bias, every op rounded to BF16 as the reference's eager BF16 ops: the state
// s = g * s + (1 - g) * c in place, and the output latent tanh(out). One block.
extern "C" __global__ void inkling_draft_markov_gates(const float* __restrict__ projected,
                                                      const __nv_bfloat16* __restrict__ bias,
                                                      float* __restrict__ state,
                                                      float* __restrict__ latent, int rank) {
    OPT_PDL_ENTRY();
    for (int index = threadIdx.x; index < rank; index += blockDim.x) {
        const float gate_in = ikd_bf16(projected[index] + __bfloat162float(bias[index]));
        const float candidate_in =
            ikd_bf16(projected[rank + index] + __bfloat162float(bias[rank + index]));
        const float out_in =
            ikd_bf16(projected[2 * rank + index] + __bfloat162float(bias[2 * rank + index]));
        const float gate = ikd_bf16(1.0f / (1.0f + expf(-gate_in)));
        const float candidate = ikd_bf16(tanhf(candidate_in));
        const float kept = ikd_bf16(gate * state[index]);
        const float fresh = ikd_bf16(ikd_bf16(1.0f - gate) * candidate);
        state[index] = ikd_bf16(kept + fresh);
        latent[index] = ikd_bf16(tanhf(out_in));
    }
}

// One step's logits over this rank's `count` ids from first_id, each block's best, and each
// block's mass, the sum of exp(v - best) over its ids (the softmax's normaliser relative to the
// best, a masked id adding nothing):
// v = bf16(bf16(U) + bf16(b_f + W_f l)), softcapped as the target's (bf16(cap * tanh(v / cap)),
// none when cap is 0), ids at or past `valid` masked. `base` is the block row's head sums U and
// `product` the fold's sums W_f l. Grid (count / (blockDim.x * IKD_LOGIT_PER_THREAD)).
//
// A sampled request (`inv_t` > 0) picks by the serving sampler's Gumbel-max rule instead, over
// the full tempered distribution: each block's pick is its best fmaf(v, inv_t, G(seed, position,
// id)), into part_ids with its score in part_scores, where `position` is the sequence index the
// draft lands at; part_values keeps the block's largest logit, which the mass is relative to. The
// target perturbs its row with the same noise, so a draft from a distribution close to the
// target's tends to be the token the target samples.
extern "C" __global__ void inkling_draft_markov_logits(const float* __restrict__ base,
                                                       const float* __restrict__ product,
                                                       const __nv_bfloat16* __restrict__ bias,
                                                       int count, int first_id, int valid,
                                                       float cap, float* __restrict__ part_values,
                                                       int* __restrict__ part_ids,
                                                       float* __restrict__ part_sums, float inv_t,
                                                       unsigned long long seed, int position,
                                                       float* __restrict__ part_scores) {
    OPT_PDL_ENTRY();
    __shared__ float s_value[32];
    __shared__ int s_id[32];
    float values[IKD_LOGIT_PER_THREAD];
    float best = -CUDART_INF_F;
    int best_id = 0x7fffffff;
    float score = -CUDART_INF_F;
    int score_id = 0x7fffffff;
    const int start = blockIdx.x * blockDim.x * IKD_LOGIT_PER_THREAD;
#pragma unroll
    for (int i = 0; i < IKD_LOGIT_PER_THREAD; ++i) {
        const int index = start + i * blockDim.x + threadIdx.x;
        values[i] = -CUDART_INF_F;
        if (index < count) {
            const float head = ikd_bf16(base[index]);
            const float shift = ikd_bf16(__bfloat162float(bias[index]) + product[index]);
            float value = ikd_bf16(head + shift);
            if (cap > 0.0f) {
                value = ikd_bf16(cap * tanhf(value / cap));
            }
            const int id = first_id + index;
            if (id >= valid) {
                value = -CUDART_INF_F;
            }
            values[i] = value;
            if (ikd_above(value, id, best, best_id)) {
                best = value;
                best_id = id;
            }
            if (inv_t > 0.0f && value > -CUDART_INF_F) {
                const float noisy = fmaf(value, inv_t,
                                         opt_sample_gumbel(seed, (uint32_t)position, (uint32_t)id));
                if (ikd_above(noisy, id, score, score_id)) {
                    score = noisy;
                    score_id = id;
                }
            }
        }
    }
    ikd_block_best(best, best_id, s_value, s_id);
    if (inv_t > 0.0f) {
        ikd_block_best(score, score_id, s_value, s_id);
    }
    float mass = 0.0f;
#pragma unroll
    for (int i = 0; i < IKD_LOGIT_PER_THREAD; ++i) {
        if (values[i] > -CUDART_INF_F) {
            mass += expf(values[i] - best);
        }
    }
    mass = ikd_block_sum(mass, s_value);
    if (threadIdx.x == 0) {
        part_values[blockIdx.x] = best;
        part_ids[blockIdx.x] = inv_t > 0.0f ? score_id : best_id;
        part_sums[blockIdx.x] = mass;
        if (inv_t > 0.0f) {
            part_scores[blockIdx.x] = score;
        }
    }
}

// This rank's best of `parts` block bests into its slot of `world` pick slots ((value, id, mass,
// 0) each), every other slot zero, so the group's sum leaves each rank's best in its own slot on
// every rank. The mass is the blocks' masses (`part_sums`) rescaled to the rank's best. Under a
// sampled request's noise (`part_scores` not null) the id is the blocks' best perturbed pick and
// word 3 its score, while the value stays the rank's largest logit for the mass. One block.
extern "C" __global__ void inkling_draft_markov_local(const float* __restrict__ part_values,
                                                      const int* __restrict__ part_ids, int parts,
                                                      float* __restrict__ slots, int rank,
                                                      int world,
                                                      const float* __restrict__ part_sums,
                                                      const float* __restrict__ part_scores) {
    OPT_PDL_ENTRY();
    __shared__ float s_value[32];
    __shared__ int s_id[32];
    float best = -CUDART_INF_F;
    int best_id = 0x7fffffff;
    float score = -CUDART_INF_F;
    int score_id = 0x7fffffff;
    for (int part = threadIdx.x; part < parts; part += blockDim.x) {
        if (ikd_above(part_values[part], part_ids[part], best, best_id)) {
            best = part_values[part];
            best_id = part_ids[part];
        }
        if (part_scores != nullptr && ikd_above(part_scores[part], part_ids[part], score, score_id)) {
            score = part_scores[part];
            score_id = part_ids[part];
        }
    }
    ikd_block_best(best, best_id, s_value, s_id);
    if (part_scores != nullptr) {
        ikd_block_best(score, score_id, s_value, s_id);
        best_id = score_id;
    }
    float mass = 0.0f;
    for (int part = threadIdx.x; part < parts; part += blockDim.x) {
        if (part_values[part] > -CUDART_INF_F) {
            mass += part_sums[part] * expf(part_values[part] - best);
        }
    }
    mass = ikd_block_sum(mass, s_value);
    for (int index = threadIdx.x; index < world * IKD_PICK_SLOT; index += blockDim.x) {
        const int owner = index / IKD_PICK_SLOT;
        const int field = index % IKD_PICK_SLOT;
        float value = 0.0f;
        if (owner == rank && field == 0) {
            value = best;
        } else if (owner == rank && field == 1) {
            value = (float)best_id;
        } else if (owner == rank && field == 2) {
            value = mass;
        } else if (owner == rank && field == 3 && part_scores != nullptr) {
            value = score;
        }
        slots[index] = value;
    }
}

// The last step's pick (`noisy` as ikd_pick), into tokens[step], and its probability, into
// confidence[step]. One thread.
extern "C" __global__ void inkling_draft_markov_last(const float* __restrict__ slots, int world,
                                                     unsigned* __restrict__ tokens, int step,
                                                     float* __restrict__ confidence, int noisy) {
    OPT_PDL_ENTRY();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        tokens[step] = (unsigned)ikd_pick(slots, world, noisy);
        confidence[step] = ikd_confidence(slots, world);
    }
}

// ---------------------------------------------------------------------------------------------
// Per-row NVFP4 quantization and Tensor Core batched GEMV for the DSpark drafter.
// The drafter only proposes candidate tokens (the target verifies every token with exact
// weights), so quantizing the drafter's BF16 projections and LM/Markov heads at load time cuts
// drafter DRAM bandwidth by ~2.7x while preserving >0.9995 cosine similarity.
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ unsigned char ikd_q_e2m1(float v) {
    const unsigned char sign = (v < 0.0f) ? 8u : 0u;
    const float a = fabsf(v);
    const unsigned char code = (a < 0.25f) ? 0u : (a < 0.75f) ? 1u : (a < 1.25f) ? 2u :
                               (a < 1.75f) ? 3u : (a < 2.50f) ? 4u : (a < 3.50f) ? 5u :
                               (a < 5.00f) ? 6u : 7u;
    return sign | code;
}

__device__ __forceinline__ unsigned char ikd_f32_to_e4m3(float v) {
    v = fminf(fmaxf(v, 0.0f), 448.0f);
    unsigned short pair;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(pair) : "f"(0.0f), "f"(v));
    return (unsigned char)(pair & 0xFFu);
}

__device__ __forceinline__ float ikd_e4m3_hw(unsigned char bits) {
    const unsigned short pair = (unsigned short)bits;
    unsigned int f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"(pair));
    const unsigned short lo = (unsigned short)(f16x2 & 0xFFFFu);
    float out;
    asm("cvt.f32.f16 %0, %1;" : "=f"(out) : "h"(lo));
    return out;
}

__device__ __forceinline__ __half2 ikd_e4m3_scale_f16x2(unsigned char sb) {
    const unsigned short sp = (unsigned short)sb | ((unsigned short)sb << 8);
    unsigned int sc_f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sc_f16x2) : "h"(sp));
    return *reinterpret_cast<const __half2*>(&sc_f16x2);
}

__device__ __forceinline__ void ikd_mma_m16n8k16_f16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ikd_e2m1x4_scaled(unsigned short two_bytes, __half2 sc2,
                                                  uint32_t& out0, uint32_t& out1) {
    uint32_t p0, p1;
    asm("{\n\t.reg .b8 c0, c1;\n\t"
        "mov.b16 {c0, c1}, %2;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, c0;\n\t"
        "cvt.rn.f16x2.e2m1x2 %1, c1;\n\t}"
        : "=r"(p0), "=r"(p1)
        : "h"(two_bytes));
    const __half2 h0 = __hmul2(*reinterpret_cast<const __half2*>(&p0), sc2);
    const __half2 h1 = __hmul2(*reinterpret_cast<const __half2*>(&p1), sc2);
    out0 = *reinterpret_cast<const uint32_t*>(&h0);
    out1 = *reinterpret_cast<const uint32_t*>(&h1);
}

__device__ __forceinline__ void ikd_nvfp4_decode8_scaled(unsigned int word, __half2 sc2,
                                                         float (&scaled)[8]) {
    unsigned int pairs[4];
    asm("{\n\t.reg .b8 c0, c1, c2, c3;\n\t"
        "mov.b32 {c0, c1, c2, c3}, %4;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, c0;\n\t"
        "cvt.rn.f16x2.e2m1x2 %1, c1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %2, c2;\n\t"
        "cvt.rn.f16x2.e2m1x2 %3, c3;\n\t}"
        : "=r"(pairs[0]), "=r"(pairs[1]), "=r"(pairs[2]), "=r"(pairs[3])
        : "r"(word));
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const __half2 prod = __hmul2(*reinterpret_cast<const __half2*>(&pairs[p]), sc2);
        const float2 f2 = __half22float2(prod);
        scaled[2 * p] = f2.x;
        scaled[2 * p + 1] = f2.y;
    }
}

// Quantize BF16 [rows, cols] (cols % 64 == 0) into pre-swizzled MMA-tiled NVFP4.
// Grid: (ceil(rows / 16), 1, 1), Block: 256 threads.
extern "C" __global__ void __launch_bounds__(256) inkling_draft_quantize_nvfp4_tiled(
    const __nv_bfloat16* __restrict__ w,
    uint4* __restrict__ codes_tiled,
    unsigned int* __restrict__ scales_tiled,
    float* __restrict__ row_scale,
    int rows,
    int cols) {
    OPT_PDL_ENTRY();
    const int rt = blockIdx.x;
    const int r0 = rt * 16;
    if (r0 >= rows || (cols & 63) != 0) {
        return;
    }
    const int local_r = threadIdx.x >> 4;
    const int q = threadIdx.x & 15;
    const int row = r0 + local_r;
    const bool valid = row < rows;

    float amax = 0.0f;
    if (valid) {
        const __nv_bfloat16* src = w + (unsigned long long)row * cols;
        for (int c = q; c < cols; c += 16) {
            amax = fmaxf(amax, fabsf(__bfloat162float(src[c])));
        }
    }
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
    }
    __shared__ float s_rscale[16];
    if (q == 0) {
        const float rs = (valid && amax > 0.0f) ? (amax / (6.0f * 448.0f)) : 1.0f;
        s_rscale[local_r] = rs;
        if (valid) {
            row_scale[row] = rs;
        }
    }
    __syncthreads();

    const int k_tiles = cols >> 6;
    __shared__ unsigned char s_codes[16][32];
    __shared__ unsigned char s_scales[16][4];
    const int lane = threadIdx.x & 31;
    const int grp = lane >> 2;
    const int t4 = lane & 3;

    for (int kt = 0; kt < k_tiles; ++kt) {
        if (q < 4) {
            const int sub = q;
            const int col0 = (kt << 6) + sub * 16;
            float vals[16];
            float bmax = 0.0f;
            if (valid) {
                const __nv_bfloat16* src = w + (unsigned long long)row * cols + col0;
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    vals[j] = __bfloat162float(src[j]);
                    bmax = fmaxf(bmax, fabsf(vals[j]));
                }
            } else {
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    vals[j] = 0.0f;
                }
            }
            const float rs = s_rscale[local_r];
            const unsigned char sb = ikd_f32_to_e4m3(bmax / (6.0f * rs));
            const float bs = ikd_e4m3_hw(sb) * rs;
            const float inv = bs > 0.0f ? (1.0f / bs) : 0.0f;
            s_scales[local_r][sub] = sb;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned char lo = ikd_q_e2m1(vals[2 * j] * inv);
                const unsigned char hi = ikd_q_e2m1(vals[2 * j + 1] * inv);
                s_codes[local_r][sub * 8 + j] = lo | (hi << 4);
            }
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            unsigned int cw[4];
#pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                const int coff = sub * 8 + t4;
                const unsigned int b0 = s_codes[grp][coff];
                const unsigned int b1 = s_codes[grp][coff + 4];
                const unsigned int b2 = s_codes[grp + 8][coff];
                const unsigned int b3 = s_codes[grp + 8][coff + 4];
                cw[sub] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
            }
            codes_tiled[((unsigned long long)rt * k_tiles + kt) * 32 + lane] =
                make_uint4(cw[0], cw[1], cw[2], cw[3]);
            if (lane < 16) {
                const unsigned int s4 = (unsigned int)s_scales[lane][0] |
                                        ((unsigned int)s_scales[lane][1] << 8) |
                                        ((unsigned int)s_scales[lane][2] << 16) |
                                        ((unsigned int)s_scales[lane][3] << 24);
                scales_tiled[((unsigned long long)rt * k_tiles + kt) * 16 + lane] = s4;
            }
        }
        __syncthreads();
    }
}

// Quantize BF16 [rows, cols] (cols % 16 == 0) into row-major per-row-scaled NVFP4 (for fold_weight).
// Grid: (rows, 1, 1), Block: 256 threads.
extern "C" __global__ void __launch_bounds__(256) inkling_draft_quantize_nvfp4_row(
    const __nv_bfloat16* __restrict__ w,
    unsigned char* __restrict__ codes,
    unsigned char* __restrict__ scales,
    float* __restrict__ row_scale,
    int rows,
    int cols) {
    OPT_PDL_ENTRY();
    const int row = blockIdx.x;
    if (row >= rows || (cols & 15) != 0) {
        return;
    }
    const __nv_bfloat16* src = w + (unsigned long long)row * cols;
    float amax = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        amax = fmaxf(amax, fabsf(__bfloat162float(src[c])));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
    }
    __shared__ float smax[8];
    if ((threadIdx.x & 31) == 0) {
        smax[threadIdx.x >> 5] = amax;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = 0.0f;
        for (int i = 0; i < 8; ++i) {
            m = fmaxf(m, smax[i]);
        }
        smax[0] = (m > 0.0f) ? (m / (6.0f * 448.0f)) : 1.0f;
        row_scale[row] = smax[0];
    }
    __syncthreads();
    const float rscale = smax[0];
    const int nblocks = cols >> 4;
    unsigned char* dst_c = codes + (unsigned long long)row * (cols >> 1);
    unsigned char* dst_s = scales + (unsigned long long)row * nblocks;
    for (int b = threadIdx.x; b < nblocks; b += blockDim.x) {
        float vals[16];
        float bmax = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            vals[j] = __bfloat162float(src[b * 16 + j]);
            bmax = fmaxf(bmax, fabsf(vals[j]));
        }
        const unsigned char sb = ikd_f32_to_e4m3(bmax / (6.0f * rscale));
        const float bs = ikd_e4m3_hw(sb) * rscale;
        const float inv = bs > 0.0f ? (1.0f / bs) : 0.0f;
        dst_s[b] = sb;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned char lo = ikd_q_e2m1(vals[2 * j] * inv);
            const unsigned char hi = ikd_q_e2m1(vals[2 * j + 1] * inv);
            dst_c[b * 8 + j] = lo | (hi << 4);
        }
    }
}

#define IKD_TC_K_CHUNK 512

// Chunked-K Tensor Core NVFP4 batched GEMV (batch 1..8, cols % 64 == 0, 8 KB SMEM -> 4+ blocks/SM).
// Grid: (ceil(rows / rows_per_block), 1, 1).
extern "C" __global__ void __launch_bounds__(256, 4) inkling_draft_nvfp4_gemv(
    const uint4* __restrict__ codes_tiled,
    const unsigned int* __restrict__ scales_tiled,
    const float* __restrict__ row_scale,
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int columns,
    int batch,
    int rt_per_warp) {
    OPT_PDL_ENTRY();
    __shared__ __align__(16) __half s_x[8][IKD_TC_K_CHUNK];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;
    const int lane = tid & 31;
    const int grp = lane >> 2;
    const int t4 = lane & 3;
    const int k_tiles = columns >> 6;
    const bool two_rt = rt_per_warp >= 2;
    const int rt0 = blockIdx.x * (two_rt ? (warps * 2) : warps) + warp;
    const int rt1 = rt0 + warps;
    const bool has0 = (rt0 * 16) < rows;
    const bool has1 = two_rt && ((rt1 * 16) < rows);

    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k0 = 0; k0 < columns; k0 += IKD_TC_K_CHUNK) {
        const int kn = min(IKD_TC_K_CHUNK, columns - k0);
        for (int idx = tid * 4; idx < 8 * IKD_TC_K_CHUNK; idx += (int)blockDim.x * 4) {
            const int p = idx / IKD_TC_K_CHUNK;
            const int c = idx - p * IKD_TC_K_CHUNK;
            if (p < batch && c < kn) {
                const float4 v = *reinterpret_cast<const float4*>(
                    x + (unsigned long long)p * columns + k0 + c);
                *reinterpret_cast<__half2*>(&s_x[p][c]) = __float22half2_rn(make_float2(v.x, v.y));
                *reinterpret_cast<__half2*>(&s_x[p][c + 2]) =
                    __float22half2_rn(make_float2(v.z, v.w));
            } else {
                *reinterpret_cast<uint2*>(&s_x[p][c]) = make_uint2(0, 0);
            }
        }
        __syncthreads();

        const int kt_begin = k0 >> 6;
        const int kt_n = kn >> 6;
        auto run_rt = [&](int rt, float (&acc)[4]) {
            const uint4* __restrict__ cp =
                codes_tiled + ((unsigned long long)rt * k_tiles + kt_begin) * 32 + lane;
            const unsigned int* __restrict__ sp =
                scales_tiled + ((unsigned long long)rt * k_tiles + kt_begin) * 16;
            for (int i = 0; i < kt_n; ++i) {
                const uint4 cv = __ldg(cp + (unsigned long long)i * 32);
                const unsigned int sa4 = __ldg(sp + (unsigned long long)i * 16 + grp);
                const unsigned int sb4 = __ldg(sp + (unsigned long long)i * 16 + grp + 8);
                const unsigned int cw[4] = {cv.x, cv.y, cv.z, cv.w};
                const int k_local = i << 6;
#pragma unroll
                for (int sub = 0; sub < 4; ++sub) {
                    const __half2 sc_a = ikd_e4m3_scale_f16x2((unsigned char)(sa4 >> (8 * sub)));
                    const __half2 sc_b = ikd_e4m3_scale_f16x2((unsigned char)(sb4 >> (8 * sub)));
                    uint32_t a0, a1, a2, a3;
                    ikd_e2m1x4_scaled((unsigned short)(cw[sub] & 0xFFFFu), sc_a, a0, a2);
                    ikd_e2m1x4_scaled((unsigned short)(cw[sub] >> 16), sc_b, a1, a3);
                    const int ks = k_local + sub * 16;
                    const uint32_t b0 =
                        *reinterpret_cast<const uint32_t*>(&s_x[grp][ks + 2 * t4]);
                    const uint32_t b1 =
                        *reinterpret_cast<const uint32_t*>(&s_x[grp][ks + 2 * t4 + 8]);
                    ikd_mma_m16n8k16_f16(acc, a0, a1, a2, a3, b0, b1);
                }
            }
        };
        if (has0) {
            run_rt(rt0, acc0);
        }
        if (has1) {
            run_rt(rt1, acc1);
        }
        __syncthreads();
    }

    auto store_rt = [&](int rt, const float (&acc)[4]) {
        const int r0 = rt * 16;
        const int ra = r0 + grp;
        const int rb = r0 + grp + 8;
        if (ra < rows) {
            const float rs = row_scale[ra];
            if (2 * t4 < batch) {
                out[(unsigned long long)(2 * t4) * rows + ra] = acc[0] * rs;
            }
            if (2 * t4 + 1 < batch) {
                out[(unsigned long long)(2 * t4 + 1) * rows + ra] = acc[1] * rs;
            }
        }
        if (rb < rows) {
            const float rs = row_scale[rb];
            if (2 * t4 < batch) {
                out[(unsigned long long)(2 * t4) * rows + rb] = acc[2] * rs;
            }
            if (2 * t4 + 1 < batch) {
                out[(unsigned long long)(2 * t4 + 1) * rows + rb] = acc[3] * rs;
            }
        }
    };
    if (has0) {
        store_rt(rt0, acc0);
    }
    if (has1) {
        store_rt(rt1, acc1);
    }
}

// Fused 2- or 3-matrix Tensor Core NVFP4 batched GEMV sharing s_x (for QKV and GateUp).
// Each matrix's rows must be a multiple of 16 so row-tiles never straddle matrices.
// Grid: (ceil((rt0 + rt1 + rt2) / 8), 1, 1), Block: 256 threads (128 rows/block).
extern "C" __global__ void __launch_bounds__(256, 4) inkling_draft_nvfp4_gemv3(
    const uint4* __restrict__ c0,
    const unsigned int* __restrict__ s0,
    const float* __restrict__ rs0,
    float* __restrict__ o0,
    int r0_rows,
    const uint4* __restrict__ c1,
    const unsigned int* __restrict__ s1,
    const float* __restrict__ rs1,
    float* __restrict__ o1,
    int r1_rows,
    const uint4* __restrict__ c2,
    const unsigned int* __restrict__ s2,
    const float* __restrict__ rs2,
    float* __restrict__ o2,
    int r2_rows,
    const float* __restrict__ x,
    int columns,
    int batch) {
    OPT_PDL_ENTRY();
    __shared__ __align__(16) __half s_x[8][IKD_TC_K_CHUNK];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp = lane >> 2;
    const int t4 = lane & 3;
    const int k_tiles = columns >> 6;

    const int nrt0 = (r0_rows + 15) >> 4;
    const int nrt1 = (r1_rows + 15) >> 4;
    const int nrt2 = (r2_rows + 15) >> 4;
    const int global_rt = blockIdx.x * 8 + warp;

    const uint4* codes_tiled = nullptr;
    const unsigned int* scales_tiled = nullptr;
    const float* row_scale = nullptr;
    float* out = nullptr;
    int local_rt = 0;
    int m_rows = 0;
    if (global_rt < nrt0) {
        codes_tiled = c0;
        scales_tiled = s0;
        row_scale = rs0;
        out = o0;
        local_rt = global_rt;
        m_rows = r0_rows;
    } else if (global_rt < nrt0 + nrt1) {
        codes_tiled = c1;
        scales_tiled = s1;
        row_scale = rs1;
        out = o1;
        local_rt = global_rt - nrt0;
        m_rows = r1_rows;
    } else if (global_rt < nrt0 + nrt1 + nrt2) {
        codes_tiled = c2;
        scales_tiled = s2;
        row_scale = rs2;
        out = o2;
        local_rt = global_rt - nrt0 - nrt1;
        m_rows = r2_rows;
    }

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k0 = 0; k0 < columns; k0 += IKD_TC_K_CHUNK) {
        const int kn = min(IKD_TC_K_CHUNK, columns - k0);
        for (int idx = tid * 4; idx < 8 * IKD_TC_K_CHUNK; idx += 256 * 4) {
            const int p = idx / IKD_TC_K_CHUNK;
            const int c = idx - p * IKD_TC_K_CHUNK;
            if (p < batch && c < kn) {
                const float4 v = *reinterpret_cast<const float4*>(
                    x + (unsigned long long)p * columns + k0 + c);
                *reinterpret_cast<__half2*>(&s_x[p][c]) = __float22half2_rn(make_float2(v.x, v.y));
                *reinterpret_cast<__half2*>(&s_x[p][c + 2]) =
                    __float22half2_rn(make_float2(v.z, v.w));
            } else {
                *reinterpret_cast<uint2*>(&s_x[p][c]) = make_uint2(0, 0);
            }
        }
        __syncthreads();

        if (codes_tiled != nullptr) {
            const int kt_begin = k0 >> 6;
            const int kt_n = kn >> 6;
            const uint4* __restrict__ cp =
                codes_tiled + ((unsigned long long)local_rt * k_tiles + kt_begin) * 32 + lane;
            const unsigned int* __restrict__ sp =
                scales_tiled + ((unsigned long long)local_rt * k_tiles + kt_begin) * 16;
            for (int i = 0; i < kt_n; ++i) {
                const uint4 cv = __ldg(cp + (unsigned long long)i * 32);
                const unsigned int sa4 = __ldg(sp + (unsigned long long)i * 16 + grp);
                const unsigned int sb4 = __ldg(sp + (unsigned long long)i * 16 + grp + 8);
                const unsigned int cw[4] = {cv.x, cv.y, cv.z, cv.w};
                const int k_local = i << 6;
#pragma unroll
                for (int sub = 0; sub < 4; ++sub) {
                    const __half2 sc_a = ikd_e4m3_scale_f16x2((unsigned char)(sa4 >> (8 * sub)));
                    const __half2 sc_b = ikd_e4m3_scale_f16x2((unsigned char)(sb4 >> (8 * sub)));
                    uint32_t a0, a1, a2, a3;
                    ikd_e2m1x4_scaled((unsigned short)(cw[sub] & 0xFFFFu), sc_a, a0, a2);
                    ikd_e2m1x4_scaled((unsigned short)(cw[sub] >> 16), sc_b, a1, a3);
                    const int ks = k_local + sub * 16;
                    const uint32_t b0 =
                        *reinterpret_cast<const uint32_t*>(&s_x[grp][ks + 2 * t4]);
                    const uint32_t b1 =
                        *reinterpret_cast<const uint32_t*>(&s_x[grp][ks + 2 * t4 + 8]);
                    ikd_mma_m16n8k16_f16(acc, a0, a1, a2, a3, b0, b1);
                }
            }
        }
        __syncthreads();
    }

    if (codes_tiled != nullptr) {
        const int r0 = local_rt * 16;
        const int ra = r0 + grp;
        const int rb = r0 + grp + 8;
        if (ra < m_rows) {
            const float rs = row_scale[ra];
            if (2 * t4 < batch) {
                out[(unsigned long long)(2 * t4) * m_rows + ra] = acc[0] * rs;
            }
            if (2 * t4 + 1 < batch) {
                out[(unsigned long long)(2 * t4 + 1) * m_rows + ra] = acc[1] * rs;
            }
        }
        if (rb < m_rows) {
            const float rs = row_scale[rb];
            if (2 * t4 < batch) {
                out[(unsigned long long)(2 * t4) * m_rows + rb] = acc[2] * rs;
            }
            if (2 * t4 + 1 < batch) {
                out[(unsigned long long)(2 * t4 + 1) * m_rows + rb] = acc[3] * rs;
            }
        }
    }
}

// Fused NVFP4 fold_weight @ latent + base_logits + fold_bias + softcap + Gumbel + block argmax
// + softmax mass in one launch (zero self.product DRAM traffic).
// Supports rank <= 256, rank % 16 == 0. Grid: (ceil(count / 256), 1, 1), Block: 256 threads.
extern "C" __global__ void __launch_bounds__(256) inkling_draft_nvfp4_markov_logits(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ row_scale,
    const float* __restrict__ latent,
    const float* __restrict__ base_logits,
    const __nv_bfloat16* __restrict__ bias,
    int count,
    int rank,
    int first_id,
    int valid,
    float cap,
    float* __restrict__ part_values,
    int* __restrict__ part_ids,
    float* __restrict__ part_sums,
    float inv_t,
    unsigned long long seed,
    int position,
    float* __restrict__ part_scores) {
    OPT_PDL_ENTRY();
    __shared__ float s_lat[256];
    s_lat[threadIdx.x] = (int)threadIdx.x < rank ? latent[threadIdx.x] : 0.0f;
    __syncthreads();

    const int q = threadIdx.x & 15;
    const int hw = threadIdx.x >> 4;
    const int rank_half = rank >> 1;
    const int rank_blocks = rank >> 4;
    const bool lane_active = (q * 16) < rank;
    float4 l0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 l1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 l2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 l3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (lane_active) {
        l0 = *reinterpret_cast<const float4*>(&s_lat[q * 16]);
        l1 = *reinterpret_cast<const float4*>(&s_lat[q * 16 + 4]);
        l2 = *reinterpret_cast<const float4*>(&s_lat[q * 16 + 8]);
        l3 = *reinterpret_cast<const float4*>(&s_lat[q * 16 + 12]);
    }

    const int block_row0 = blockIdx.x * 256;
    __shared__ float s_vals[256];
    for (int it = 0; it < 16; ++it) {
        const int local_r = it * 16 + hw;
        const int row = block_row0 + local_r;
        const int live = row < count ? row : count - 1;
        float acc = 0.0f;
        if (lane_active) {
            const uint2 word =
                *reinterpret_cast<const uint2*>(codes + (unsigned long long)live * rank_half + q * 8);
            const unsigned char sb = scales[(unsigned long long)live * rank_blocks + q];
            const __half2 sc2 = ikd_e4m3_scale_f16x2(sb);
            float sc[8];
            ikd_nvfp4_decode8_scaled(word.x, sc2, sc);
            acc = fmaf(sc[0], l0.x, acc);
            acc = fmaf(sc[1], l0.y, acc);
            acc = fmaf(sc[2], l0.z, acc);
            acc = fmaf(sc[3], l0.w, acc);
            acc = fmaf(sc[4], l1.x, acc);
            acc = fmaf(sc[5], l1.y, acc);
            acc = fmaf(sc[6], l1.z, acc);
            acc = fmaf(sc[7], l1.w, acc);
            ikd_nvfp4_decode8_scaled(word.y, sc2, sc);
            acc = fmaf(sc[0], l2.x, acc);
            acc = fmaf(sc[1], l2.y, acc);
            acc = fmaf(sc[2], l2.z, acc);
            acc = fmaf(sc[3], l2.w, acc);
            acc = fmaf(sc[4], l3.x, acc);
            acc = fmaf(sc[5], l3.y, acc);
            acc = fmaf(sc[6], l3.z, acc);
            acc = fmaf(sc[7], l3.w, acc);
        }
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 8);
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 4);
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 2);
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 1);
        if (q == 0) {
            float v = -CUDART_INF_F;
            if (row < count && first_id + row < valid) {
                const float prod = acc * row_scale[live];
                const float head = ikd_bf16(base_logits[row]);
                const float shift = ikd_bf16(__bfloat162float(bias[row]) + prod);
                v = ikd_bf16(head + shift);
                if (cap > 0.0f) {
                    v = ikd_bf16(cap * tanhf(v / cap));
                }
            }
            s_vals[local_r] = v;
        }
    }
    __syncthreads();

    __shared__ float s_value[32];
    __shared__ int s_id[32];
    const float value = s_vals[threadIdx.x];
    const int id = first_id + block_row0 + (int)threadIdx.x;
    float best = value;
    int best_id = (value > -CUDART_INF_F) ? id : 0x7fffffff;
    ikd_block_best(best, best_id, s_value, s_id);

    float score = -CUDART_INF_F;
    int score_id = 0x7fffffff;
    if (inv_t > 0.0f) {
        if (value > -CUDART_INF_F) {
            score = fmaf(value, inv_t, opt_sample_gumbel(seed, (uint32_t)position, (uint32_t)id));
            score_id = id;
        }
        ikd_block_best(score, score_id, s_value, s_id);
    }
    float mass = (value > -CUDART_INF_F) ? expf(value - best) : 0.0f;
    mass = ikd_block_sum(mass, s_value);
    if (threadIdx.x == 0) {
        part_values[blockIdx.x] = best;
        part_ids[blockIdx.x] = inv_t > 0.0f ? score_id : best_id;
        part_sums[blockIdx.x] = mass;
        if (inv_t > 0.0f) {
            part_scores[blockIdx.x] = score;
        }
    }
}

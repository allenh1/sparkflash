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
#include <cuda_runtime.h>

#include "opt_pdl_sm121.cuh"
#include <math_constants.h>

// The drafter's head width: one rotary pair a thread of a half-head.
#define IKD_HEAD_DIM 128
// The most hidden values one mix-norm thread holds: hidden <= blockDim.x * this.
#define IKD_MAX_PER_THREAD 32
// The logits each thread of a step-logits block scores.
#define IKD_LOGIT_PER_THREAD 8
// The floats a rank's slot of the cross-rank pick takes: (value, id) and two zeros, so the
// group's sum moves whole groups of four.
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

// The best of the ranks' (value, id) slots after the group's sum.
__device__ __forceinline__ int ikd_pick(const float* __restrict__ slots, int world) {
    float best = slots[0];
    int id = (int)slots[1];
    for (int rank = 1; rank < world; ++rank) {
        const float value = slots[rank * IKD_PICK_SLOT];
        const int other = (int)slots[rank * IKD_PICK_SLOT + 1];
        if (ikd_above(value, other, best, id)) {
            best = value;
            id = other;
        }
    }
    return id;
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

// The Markov head's step input z = [s; W1[x]; h; f] (rank, rank, hidden, hidden values), BF16
// values: x is `anchor` at step 0 and otherwise the pick of the last step's slots, which block 0
// records at tokens[step]. h is the step's normed block row, rounded here; s and f already hold
// BF16 values. Grid (width / 256).
extern "C" __global__ void inkling_draft_markov_input(float* __restrict__ z,
                                                      const float* __restrict__ state,
                                                      const __nv_bfloat16* __restrict__ embed,
                                                      unsigned* __restrict__ tokens, int anchor,
                                                      const float* __restrict__ slots, int world,
                                                      int step, const float* __restrict__ row,
                                                      const float* __restrict__ fused, int rank,
                                                      int hidden) {
    OPT_PDL_ENTRY();
    int token;
    if (step == 0) {
        token = anchor;
    } else {
        token = ikd_pick(slots, world);
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            tokens[step] = (unsigned)token;
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

// One step's logits over this rank's `count` ids from first_id, and each block's best:
// v = bf16(bf16(U) + bf16(b_f + W_f l)), softcapped as the target's (bf16(cap * tanh(v / cap)),
// none when cap is 0), ids at or past `valid` masked. `base` is the block row's head sums U and
// `product` the fold's sums W_f l. Grid (count / (blockDim.x * IKD_LOGIT_PER_THREAD)).
extern "C" __global__ void inkling_draft_markov_logits(const float* __restrict__ base,
                                                       const float* __restrict__ product,
                                                       const __nv_bfloat16* __restrict__ bias,
                                                       int count, int first_id, int valid,
                                                       float cap, float* __restrict__ part_values,
                                                       int* __restrict__ part_ids) {
    OPT_PDL_ENTRY();
    __shared__ float s_value[32];
    __shared__ int s_id[32];
    float best = -CUDART_INF_F;
    int best_id = 0x7fffffff;
    const int start = blockIdx.x * blockDim.x * IKD_LOGIT_PER_THREAD;
#pragma unroll
    for (int i = 0; i < IKD_LOGIT_PER_THREAD; ++i) {
        const int index = start + i * blockDim.x + threadIdx.x;
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
            if (ikd_above(value, id, best, best_id)) {
                best = value;
                best_id = id;
            }
        }
    }
    ikd_block_best(best, best_id, s_value, s_id);
    if (threadIdx.x == 0) {
        part_values[blockIdx.x] = best;
        part_ids[blockIdx.x] = best_id;
    }
}

// This rank's best of `parts` block bests into its slot of `world` pick slots ((value, id, 0, 0)
// each), every other slot zero, so the group's sum leaves each rank's best in its own slot on
// every rank. One block.
extern "C" __global__ void inkling_draft_markov_local(const float* __restrict__ part_values,
                                                      const int* __restrict__ part_ids, int parts,
                                                      float* __restrict__ slots, int rank,
                                                      int world) {
    OPT_PDL_ENTRY();
    __shared__ float s_value[32];
    __shared__ int s_id[32];
    float best = -CUDART_INF_F;
    int best_id = 0x7fffffff;
    for (int part = threadIdx.x; part < parts; part += blockDim.x) {
        if (ikd_above(part_values[part], part_ids[part], best, best_id)) {
            best = part_values[part];
            best_id = part_ids[part];
        }
    }
    ikd_block_best(best, best_id, s_value, s_id);
    for (int index = threadIdx.x; index < world * IKD_PICK_SLOT; index += blockDim.x) {
        const int owner = index / IKD_PICK_SLOT;
        const int field = index % IKD_PICK_SLOT;
        float value = 0.0f;
        if (owner == rank && field == 0) {
            value = best;
        } else if (owner == rank && field == 1) {
            value = (float)best_id;
        }
        slots[index] = value;
    }
}

// The last step's pick, into tokens[step]. One thread.
extern "C" __global__ void inkling_draft_markov_last(const float* __restrict__ slots, int world,
                                                     unsigned* __restrict__ tokens, int step) {
    OPT_PDL_ENTRY();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        tokens[step] = (unsigned)ikd_pick(slots, world);
    }
}

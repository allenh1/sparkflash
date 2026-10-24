// kernels/opt_select_sm121.cu: ref_select_sm121 with a warp-per-position score pass (generated copy).
// kernels/ref_select_sm121.cu
//
// The reference candidate select: index scores, the reach and HSI block masks, the top-k by
// (score descending, compressing position ascending) and the shared-index-space ids the attention
// body's selected list reads. One CTA per query; a decode step is one query.
//
// THE CHAIN IS THE ENGINE'S, statement by statement:
//   score        crates/sparkflash-engine/src/forward/attention.rs:1476-1506 (index_scores):
//                per head, ascending head order, raw = bf16_round(dot(q_head, key)) with the dot
//                accumulated ascending in f32; if raw > 0 then sum += bf16_round(raw * weight);
//                after the heads sum = bf16_round(sum).
//   reach mask   attention.rs:1610-1616: positions >= (start_pos + query + 1) / ratio become -inf.
//   block mask   attention.rs:1619-1650 with select_candidate_blocks (attention.rs:1704-1734):
//                per block the best position's score; the block holding the newest reachable
//                position is pinned to +inf; the best topk_blocks blocks by (score desc, block asc)
//                are kept; a kept block whose score is -inf is dropped; kept positions set bits.
//   top-k        attention.rs:1660-1686: order every position by score descending and, on a tie,
//                position ascending; take the first min(topk, n_pos); re-sort those ascending; map
//                position < reach to position + offset (window_rows) and anything else to NO_INDEX.
//
// EXACTNESS BEFORE SPEED. The selection here is computed as a rank, not a sort: a position's rank
// is the number of positions that beat it in the same total order, so the chosen set and its order
// are the host's by construction and no sort comparator can disagree with it. That is O(n_pos^2)
// comparisons per CTA and it is the deliberate first form; a bitonic or radix retile is a later
// change that must keep this file's rank definition, not redefine it.
//
// NO TREE REDUCTION, NO REASSOCIATION. The dot, the head sum and the f32->bf16 rounds are the
// engine's exact operations (__fmul_rn, __fadd_rn, RNE narrowing); a score is a single chain of
// them, one per (head, position), never a reduced batch.
//
// LIMITS. MAX_POS = 4096 compressed positions (one CTA's score table); a caller with more refuses
// before the launch, the kernel has no growing path. head_dim and heads are runtime words.
//
// NO TRACE RING: this kernel writes no trace slot, so its launch carries no ring pointer.

#include "opt_pdl_sm121.cuh"
#include "m4_common_sm121.cuh"

// A named namespace around the whole file, kernels included (extern "C" keeps their symbols): the
// build's combined probe compiles this beside ref_select_sm121.cu, whose helpers share these names.
namespace opt_select {
constexpr int MAX_POS = 4096;
constexpr int MAX_BLOCKS = 4096;
constexpr int BLOCK = 256;
constexpr int MAX_MASK_WORDS = MAX_POS / 32;

__device__ __forceinline__ float round_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

// true when (score_a, id_a) stands before (score_b, id_b): score descending, then id ascending.
__device__ __forceinline__ bool before(float score_a, uint32_t id_a, float score_b, uint32_t id_b) {
    if (score_a > score_b) {
        return true;
    }
    if (score_a < score_b) {
        return false;
    }
    return id_a < id_b;
}

__device__ __forceinline__ void opt_ref_select_body(
    const float* __restrict__ pre_scores,
    const float* __restrict__ q_index,
    const float* __restrict__ keys,
    const float* __restrict__ weights,
    uint32_t* __restrict__ out_ids,
    float* __restrict__ out_scores,
    const uint32_t* __restrict__ mask_in,
    uint32_t* __restrict__ mask_out,
    uint32_t n_pos,
    uint32_t heads,
    uint32_t head_dim,
    uint32_t reach,
    uint32_t topk,
    uint32_t offset,
    uint32_t mask_mode,
    uint32_t topk_blocks,
    uint32_t block_size,
    uint32_t no_index,
    uint32_t flags) {
    (void)flags;
    static_assert(BLOCK % 32 == 0, "one whole warp per score-generation step");
    static_assert(MAX_BLOCKS >= MAX_POS, "the block table covers every position");
    __shared__ float s_score[MAX_POS];
    __shared__ uint32_t s_mask[MAX_MASK_WORDS];
    __shared__ union {
        float f[MAX_BLOCKS];
        uint32_t u[MAX_BLOCKS];
    } scratch;

    const uint32_t tid = threadIdx.x;
    const uint32_t n = n_pos < MAX_POS ? n_pos : MAX_POS;

    if (mask_mode == 2u) {
        for (uint32_t word = tid; word < MAX_MASK_WORDS; word += BLOCK) {
            s_mask[word] = 0u;
        }
    } else if (mask_mode == 1u) {
        for (uint32_t word = tid; word < MAX_MASK_WORDS; word += BLOCK) {
            s_mask[word] = mask_in[word];
        }
    }
    __syncthreads();

    // Scores: one position per thread pass, heads ascending and the dot ascending inside a head.
    // Scores: warp per position, lanes over head_dim; the heads are folded in ascending order as
    // the reference does, each head's dot is a warp tree instead of a serial scan.
    if (pre_scores != nullptr) {
        for (uint32_t position = tid; position < n; position += BLOCK) s_score[position] = pre_scores[position];
    } else {
        const uint32_t lane = tid & 31u;
        const uint32_t warp = tid >> 5;
        for (uint32_t position = warp; position < n; position += BLOCK / 32u) {
            const float* key = keys + static_cast<size_t>(position) * head_dim;
            float sum = 0.0f;
            for (uint32_t head = 0; head < heads; ++head) {
                const float* q = q_index + static_cast<size_t>(head) * head_dim;
                float acc = 0.0f;
                for (uint32_t d = lane; d < head_dim; d += 32u) {
                    acc = fmaf(q[d], key[d], acc);
                }
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
                const float raw = round_bf16(acc);
                if (raw > 0.0f) {
                    sum = __fadd_rn(sum, round_bf16(__fmul_rn(raw, weights[head])));
                }
            }
            if (lane == 0u) s_score[position] = round_bf16(sum);
        }
    }
    __syncthreads();

    // Level one: the HSI candidate block mask. Block scores live in the scratch union while this
    // phase runs; the mask is then written one whole word per thread, so no atomic is needed.
    if (mask_mode == 2u && block_size != 0u && n != 0u) {
        const uint32_t num_blocks = (n + block_size - 1u) / block_size;
        for (uint32_t block = tid; block < num_blocks; block += BLOCK) {
            const uint32_t start = block * block_size;
            const uint32_t stop = (start + block_size) < n ? (start + block_size) : n;
            float best = -__builtin_huge_valf();
            for (uint32_t position = start; position < stop; ++position) {
                best = s_score[position] > best ? s_score[position] : best;
            }
            scratch.f[block] = best;
        }
        __syncthreads();
        if (reach > 0u) {
            const uint32_t last = (reach - 1u) / block_size;
            if (last < num_blocks && tid == 0u) {
                scratch.f[last] = __builtin_huge_valf();
            }
        }
        __syncthreads();
        const uint32_t keep = topk_blocks < num_blocks ? topk_blocks : num_blocks;
        for (uint32_t word = tid; word < MAX_MASK_WORDS; word += BLOCK) {
            uint32_t bits = 0u;
            for (uint32_t bit = 0; bit < 32u; ++bit) {
                const uint32_t position = word * 32u + bit;
                if (position < n) {
                    const uint32_t block = position / block_size;
                    uint32_t rank = 0u;
                    for (uint32_t other = 0; other < num_blocks; ++other) {
                        if (other != block && before(scratch.f[other], other, scratch.f[block], block)) {
                            rank += 1u;
                        }
                    }
                    if (rank < keep && scratch.f[block] > -__builtin_huge_valf()) {
                        bits |= 1u << bit;
                    }
                }
            }
            s_mask[word] = bits;
        }
        __syncthreads();
        if (mask_out != nullptr) {
            for (uint32_t word = tid; word < MAX_MASK_WORDS; word += BLOCK) {
                mask_out[word] = s_mask[word];
            }
        }
        __syncthreads();
    }
    // The masks on this query's own scores: reach first (attention.rs:1610-1616), then a consumed
    // HSI mask (attention.rs:1640-1647).
    for (uint32_t position = tid; position < n; position += BLOCK) {
        if (position >= reach) {
            s_score[position] = -__builtin_huge_valf();
        }
        if (mask_mode == 1u && (s_mask[position >> 5] & (1u << (position & 31u))) == 0u) {
            s_score[position] = -__builtin_huge_valf();
        }
    }
    __syncthreads();

    if (out_scores != nullptr) {
        for (uint32_t position = tid; position < n; position += BLOCK) {
            out_scores[position] = s_score[position];
        }
    }

    // Rank by the host's total order; the chosen positions are those with rank < topk. Then the
    // chosen ones are ordered ascending by position and mapped into the shared index space.
    const uint32_t keep = topk < n ? topk : n;
    for (uint32_t position = tid; position < n; position += BLOCK) {
        uint32_t rank = 0u;
        for (uint32_t other = 0; other < n; ++other) {
            if (other != position) {
                const bool better =
                    before(s_score[other], other, s_score[position], position);
                rank += better ? 1u : 0u;
            }
        }
        scratch.u[position] = rank;
    }
    __syncthreads();
    for (uint32_t position = tid; position < n; position += BLOCK) {
        if (scratch.u[position] < keep) {
            uint32_t before_it = 0u;
            for (uint32_t other = 0; other < n; ++other) {
                if (other != position && scratch.u[other] < keep && other < position) {
                    before_it += 1u;
                }
            }
            out_ids[before_it] = position < reach ? position + offset : no_index;
        }
    }
}

#define OPT_SEL_PARAMS                                                                              \
    const float* __restrict__ q_index, const float* __restrict__ keys,                            \
        const float* __restrict__ weights, uint32_t* __restrict__ out_ids,                         \
        float* __restrict__ out_scores, const uint32_t* __restrict__ mask_in,                      \
        uint32_t* __restrict__ mask_out, uint32_t n_pos, uint32_t heads, uint32_t head_dim,        \
        uint32_t reach, uint32_t topk, uint32_t offset, uint32_t mask_mode, uint32_t topk_blocks,  \
        uint32_t block_size, uint32_t no_index, uint32_t flags
#define OPT_SEL_ARGS                                                                                \
    q_index, keys, weights, out_ids, out_scores, mask_in, mask_out, n_pos, heads, head_dim, reach,  \
        topk, offset, mask_mode, topk_blocks, block_size, no_index, flags

extern "C" __global__ void __launch_bounds__(BLOCK) opt_ref_select(OPT_SEL_PARAMS) {
    OPT_PDL_ENTRY();
    opt_ref_select_body(nullptr, OPT_SEL_ARGS);
}

// The 18 ref_select words plus the scores opt_select_scores wrote.
extern "C" __global__ void __launch_bounds__(BLOCK) opt_ref_select_pre(OPT_SEL_PARAMS,
                                                                      const float* __restrict__ pre) {
    OPT_PDL_ENTRY();
    opt_ref_select_body(pre, OPT_SEL_ARGS);
}

// One block per compressed position: warp w scores heads w, w+8, ... (a warp tree per dot, the
// reference's bf16 rounds), then thread 0 folds the heads in ascending order. Grid (n_pos), 256.
extern "C" __global__ void __launch_bounds__(BLOCK) opt_select_scores(
    const float* __restrict__ q_index, const float* __restrict__ keys,
    const float* __restrict__ weights, float* __restrict__ scores, uint32_t n_pos, uint32_t heads,
    uint32_t head_dim) {
    OPT_PDL_ENTRY();
    __shared__ float s_c[128];
    const uint32_t position = blockIdx.x;
    if (position >= n_pos || position >= (uint32_t)MAX_POS) return;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const float* key = keys + static_cast<size_t>(position) * head_dim;
    for (uint32_t head = warp; head < heads && head < 128u; head += BLOCK / 32u) {
        const float* q = q_index + static_cast<size_t>(head) * head_dim;
        float acc = 0.0f;
        for (uint32_t d = lane; d < head_dim; d += 32u) acc = fmaf(q[d], key[d], acc);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0u) {
            const float raw = round_bf16(acc);
            s_c[head] = raw > 0.0f ? round_bf16(__fmul_rn(raw, weights[head])) : 0.0f;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
        float sum = 0.0f;
        const uint32_t h_end = heads < 128u ? heads : 128u;
        for (uint32_t head = 0; head < h_end; ++head) sum = __fadd_rn(sum, s_c[head]);
        scores[position] = round_bf16(sum);
    }
}

}  // namespace opt_select

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
        if (mask_mode == 1u && (mask_in[position >> 5] & (1u << (position & 31u))) == 0u) {
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

// ---------------------------------------------------------------------------------------------
// opt_ref_select_fast: opt_ref_select_pre's selection as a radix select instead of an O(n^2) rank.
//
// The rank definition is unchanged: a position (or a candidate block) is chosen when fewer than
// `keep` others stand before it in the total order (score descending, id ascending). Under a
// strict total order the chosen set is exactly the `keep` largest (key, -id) pairs, so it is found
// here as a threshold instead: a four-pass 8-bit radix select finds the keep-th largest score key
// T and how many of the ids holding T are taken (`need`), and an id is chosen when its key exceeds
// T, or equals T and fewer than `need` lower ids hold T. The chosen ids are then written in
// ascending id order by a block scan. Scores map to keys by their float order, with -0 folded into
// +0 because the float compare ties them.
//
// LIMITS. Up to FAST_MAX_POS compressed positions, not the rank form's MAX_POS: the keys and the
// chosen flags live in shared memory up to MAX_POS positions and in `scratch` (n_pos * 5 bytes,
// global) beyond. The masks are read and written in place (global): the mask written is
// max(MAX_MASK_WORDS, ceil(n_pos / 32)) words. Grid (1), FAST_BLOCK threads; the 18 ref_select
// words, then the scores opt_select_scores wrote and the scratch.
// ---------------------------------------------------------------------------------------------
constexpr int FAST_BLOCK = 1024;
// The model's 1M-token context at ratio 1, with room for the verify block.
constexpr uint32_t FAST_MAX_POS = 1u << 21;

__device__ __forceinline__ uint32_t order_key(float value) {
    uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7FFFFFFFu) == 0u) bits = 0u;
    return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

// The block-wide exclusive prefix of `value` in thread order; `s_warp` holds 33 words, the last
// of which receives the total. Every thread of the block calls it.
__device__ __forceinline__ uint32_t block_exclusive(uint32_t value, uint32_t* s_warp) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    uint32_t inclusive = value;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t up = __shfl_up_sync(0xFFFFFFFFu, inclusive, o);
        if (lane >= static_cast<uint32_t>(o)) inclusive += up;
    }
    if (lane == 31u) s_warp[warp] = inclusive;
    __syncthreads();
    if (warp == 0u) {
        const uint32_t mine = lane < FAST_BLOCK / 32u ? s_warp[lane] : 0u;
        uint32_t scan = mine;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const uint32_t up = __shfl_up_sync(0xFFFFFFFFu, scan, o);
            if (lane >= static_cast<uint32_t>(o)) scan += up;
        }
        if (lane < FAST_BLOCK / 32u) s_warp[lane] = scan - mine;
        if (lane == 31u) s_warp[32] = scan;
    }
    __syncthreads();
    const uint32_t result = s_warp[warp] + inclusive - value;
    __syncthreads();
    return result;
}

// Flags the `keep` largest keys of keys[0..n) by (key descending, id ascending) into chosen[]
// (1 or 0). keep <= n. Thread t owns ids [t * per, (t + 1) * per) for the scans. `hist` holds 256
// words, `s_var` 2, `s_warp` 33.
__device__ __forceinline__ void select_largest(const uint32_t* keys, uint32_t n, uint32_t keep,
                                               uint8_t* chosen, uint32_t* hist, uint32_t* s_var,
                                               uint32_t* s_warp) {
    const uint32_t tid = threadIdx.x;
    if (keep >= n || keep == 0u) {
        const uint8_t all = keep >= n ? 1u : 0u;
        for (uint32_t i = tid; i < n; i += FAST_BLOCK) chosen[i] = all;
        __syncthreads();
        return;
    }
    if (tid == 0u) {
        s_var[0] = 0u;
        s_var[1] = keep;
    }
    uint32_t prefix_mask = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint32_t i = tid; i < 256u; i += FAST_BLOCK) hist[i] = 0u;
        __syncthreads();
        const uint32_t prefix = s_var[0];
        for (uint32_t i = tid; i < n; i += FAST_BLOCK) {
            const uint32_t key = keys[i];
            if ((key & prefix_mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid < 32u) {
            // Lane l holds digits 255 - 8l down to 248 - 8l: the scan runs from the top digit.
            uint32_t local[8];
            uint32_t sum = 0u;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                local[j] = hist[255u - 8u * tid - static_cast<uint32_t>(j)];
                sum += local[j];
            }
            uint32_t inclusive = sum;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const uint32_t up = __shfl_up_sync(0xFFFFFFFFu, inclusive, o);
                if (tid >= static_cast<uint32_t>(o)) inclusive += up;
            }
            const uint32_t remaining = s_var[1];
            uint32_t above = inclusive - sum;
            __syncwarp();
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (above < remaining && above + local[j] >= remaining) {
                    const uint32_t digit = 255u - 8u * tid - static_cast<uint32_t>(j);
                    s_var[0] = prefix | (digit << shift);
                    s_var[1] = remaining - above;
                }
                above += local[j];
            }
        }
        __syncthreads();
        prefix_mask |= 255u << shift;
    }
    const uint32_t threshold = s_var[0];
    const uint32_t need = s_var[1];
    // The ids holding the threshold, in ascending order: the first `need` of them are taken.
    const uint32_t per = (n + FAST_BLOCK - 1u) / FAST_BLOCK;
    const uint32_t base = tid * per;
    const uint32_t stop = base + per < n ? base + per : n;
    uint32_t ties = 0u;
    for (uint32_t i = base; i < stop; ++i) ties += keys[i] == threshold ? 1u : 0u;
    uint32_t before_ties = block_exclusive(ties, s_warp);
    for (uint32_t i = base; i < stop; ++i) {
        const uint32_t key = keys[i];
        uint8_t take = key > threshold ? 1u : 0u;
        if (key == threshold) {
            take = before_ties < need ? 1u : 0u;
            before_ties += 1u;
        }
        chosen[i] = take;
    }
    __syncthreads();
}

// One query's select over its scores `pre` (the body of both kernels below). A chosen position p
// is written as p * id_scale + offset when it is reachable (no_index otherwise).
__device__ __forceinline__ void fast_select(const float* __restrict__ pre, uint32_t* __restrict__ scratch,
                                            uint32_t* __restrict__ out_ids, float* __restrict__ out_scores,
                                            const uint32_t* __restrict__ mask_in,
                                            uint32_t* __restrict__ mask_out, uint32_t n_pos,
                                            uint32_t reach, uint32_t topk, uint32_t offset,
                                            uint32_t id_scale, uint32_t mask_mode,
                                            uint32_t topk_blocks, uint32_t block_size,
                                            uint32_t no_index) {
    __shared__ uint32_t s_key_small[MAX_POS];
    __shared__ uint8_t s_chosen_small[MAX_POS];
    __shared__ uint32_t s_hist[256];
    __shared__ uint32_t s_warp[33];
    __shared__ uint32_t s_var[2];

    const uint32_t tid = threadIdx.x;
    const uint32_t n = n_pos < FAST_MAX_POS ? n_pos : FAST_MAX_POS;
    const bool small = n <= static_cast<uint32_t>(MAX_POS);
    uint32_t* s_key = small ? s_key_small : scratch;
    uint8_t* s_chosen = small ? s_chosen_small : reinterpret_cast<uint8_t*>(scratch + n);
    const uint32_t n_words = (n + 31u) / 32u;
    const uint32_t out_words = n_words > static_cast<uint32_t>(MAX_MASK_WORDS) ? n_words : MAX_MASK_WORDS;

    // Level one: the HSI candidate block mask (opt_ref_select_body's mask_mode 2).
    if (mask_mode == 2u && block_size != 0u && n != 0u) {
        // The blocks' keys live in s_key until the positions' keys replace them below.
        const uint32_t num_blocks = (n + block_size - 1u) / block_size;
        const uint32_t empty = order_key(-__builtin_huge_valf());
        for (uint32_t block = tid; block < num_blocks; block += FAST_BLOCK) {
            const uint32_t start = block * block_size;
            const uint32_t stop = (start + block_size) < n ? (start + block_size) : n;
            float best = -__builtin_huge_valf();
            for (uint32_t position = start; position < stop; ++position) {
                best = pre[position] > best ? pre[position] : best;
            }
            s_key[block] = order_key(best);
        }
        __syncthreads();
        if (reach > 0u && tid == 0u) {
            const uint32_t last = (reach - 1u) / block_size;
            if (last < num_blocks) s_key[last] = order_key(__builtin_huge_valf());
        }
        __syncthreads();
        const uint32_t keep_blocks = topk_blocks < num_blocks ? topk_blocks : num_blocks;
        select_largest(s_key, num_blocks, keep_blocks, s_chosen, s_hist, s_var, s_warp);
        if (mask_out != nullptr) {
            for (uint32_t word = tid; word < out_words; word += FAST_BLOCK) {
                uint32_t bits = 0u;
                for (uint32_t bit = 0; bit < 32u; ++bit) {
                    const uint32_t position = word * 32u + bit;
                    if (position < n) {
                        const uint32_t block = position / block_size;
                        if (s_chosen[block] != 0u && s_key[block] != empty) bits |= 1u << bit;
                    }
                }
                mask_out[word] = bits;
            }
        }
        __syncthreads();
    }
    for (uint32_t position = tid; position < n; position += FAST_BLOCK) {
        float score = pre[position];
        if (position >= reach) score = -__builtin_huge_valf();
        if (mask_mode == 1u && (mask_in[position >> 5] & (1u << (position & 31u))) == 0u) {
            score = -__builtin_huge_valf();
        }
        if (out_scores != nullptr) out_scores[position] = score;
        s_key[position] = order_key(score);
    }
    __syncthreads();

    const uint32_t keep = topk < n ? topk : n;
    select_largest(s_key, n, keep, s_chosen, s_hist, s_var, s_warp);
    const uint32_t per = (n + FAST_BLOCK - 1u) / FAST_BLOCK;
    const uint32_t base = tid * per;
    const uint32_t stop = base + per < n ? base + per : n;
    uint32_t mine = 0u;
    for (uint32_t i = base; i < stop; ++i) mine += s_chosen[i] != 0u ? 1u : 0u;
    uint32_t slot = block_exclusive(mine, s_warp);
    for (uint32_t position = base; position < stop; ++position) {
        if (s_chosen[position] != 0u) {
            out_ids[slot] = position < reach ? position * id_scale + offset : no_index;
            slot += 1u;
        }
    }
}

extern "C" __global__ void __launch_bounds__(FAST_BLOCK) opt_ref_select_fast(OPT_SEL_PARAMS,
                                                                            const float* __restrict__ pre,
                                                                            uint32_t* __restrict__ scratch) {
    OPT_PDL_ENTRY();
    (void)q_index;
    (void)keys;
    (void)weights;
    (void)heads;
    (void)head_dim;
    (void)flags;
    fast_select(pre, scratch, out_ids, out_scores, mask_in, mask_out, n_pos, reach, topk, offset, 1u,
                mask_mode, topk_blocks, block_size, no_index);
}

// The selects of a chunk's rows: row y is position p0 + y, which reaches the first
// (p0 + y + 1) / ratio compressed positions (all of its scores). Its chosen positions land in
// selected[y] ascending as expanded store rows (position * ratio, what compressed_selected_fill
// writes), its candidate mask in mask_in/out + y * mask_stride. Grid (blocks), 1024 threads: the
// blocks stride over the rows, each with its own `scratch_words` of scratch (n + n / 4 words for
// the most compressed positions n a row reaches).
extern "C" __global__ void __launch_bounds__(FAST_BLOCK) opt_select_rows(
    const float* __restrict__ scores, uint32_t* __restrict__ selected,
    const uint32_t* __restrict__ mask_in, uint32_t* __restrict__ mask_out,
    uint32_t* __restrict__ scratch, uint32_t tokens, uint32_t p0, uint32_t ratio, uint32_t topk,
    uint32_t mask_mode, uint32_t topk_blocks, uint32_t block_size, uint32_t score_stride,
    uint32_t selected_stride, uint32_t mask_stride, uint32_t scratch_words) {
    OPT_PDL_ENTRY();
    uint32_t* mine = scratch + static_cast<size_t>(blockIdx.x) * scratch_words;
    for (uint32_t y = blockIdx.x; y < tokens; y += gridDim.x) {
        const uint32_t n = (p0 + y + 1u) / ratio;
        if (n == 0u) continue;
        const size_t mask_row = static_cast<size_t>(y) * mask_stride;
        fast_select(scores + static_cast<size_t>(y) * score_stride, mine,
                    selected + static_cast<size_t>(y) * selected_stride, nullptr,
                    mask_mode == 1u ? mask_in + mask_row : nullptr,
                    mask_mode == 2u ? mask_out + mask_row : nullptr, n, n, topk, 0u, ratio,
                    mask_mode, topk_blocks, block_size, 0xFFFFFFFFu);
        __syncthreads();
    }
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
    if (position >= n_pos || position >= FAST_MAX_POS) return;
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

// The scores of several rows' selects, opt_select_scores' chains exactly: row y (position p0 + y)
// scores the first (p0 + y + 1) / ratio compressed positions into scores + y * score_stride, the
// query and head weights of row y at q_index + y * heads * SR_DIM and weights + y * heads. Grid
// (blocks, rows), 256 threads: a block holds its row's query in registers and strides over tiles
// of SR_TILE positions, warp w scoring each position of the tile (SR_PAIR at a time) against the
// heads w, w + 8, ... with the same lane-strided fma chain per dot and the same warp-tree sums;
// then thread t folds position t's heads in ascending order. heads <= 64, head_dim SR_DIM.
//
// The tree's three upper levels run on all eight dots in 7 shuffles: at levels 16, 8 and 4 a lane
// keeps the low half of its dots and sends the high half, and lane L's registers hold the heads in
// the order r ^ ((L >> 2) & 7), so the half it sends is exactly the half its partner keeps. Every
// add is the tree's own + partner's, and lane L ends with the dot of head slot (L >> 2) & 7.
constexpr uint32_t SR_TILE = 16;
constexpr uint32_t SR_PAIR = 2;
constexpr uint32_t SR_HPW = 8;
constexpr uint32_t SR_DIM = 128;
constexpr uint32_t SR_DPL = SR_DIM / 32;
extern "C" __global__ void __launch_bounds__(BLOCK) opt_select_scores_rows(
    const float* __restrict__ q_index, const float* __restrict__ keys,
    const float* __restrict__ weights, float* __restrict__ scores, uint32_t p0, uint32_t ratio,
    uint32_t score_stride, uint32_t heads) {
    OPT_PDL_ENTRY();
    __shared__ float s_c[SR_TILE][SR_HPW * (BLOCK / 32) + 1];
    const uint32_t row = blockIdx.y;
    const uint32_t reach = (p0 + row + 1u) / ratio;
    const uint32_t n_pos = reach < FAST_MAX_POS ? reach : FAST_MAX_POS;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t slot = (lane >> 2) & 7u;
    const float* q_row = q_index + static_cast<size_t>(row) * heads * SR_DIM;
    float q[SR_HPW][SR_DPL];
#pragma unroll
    for (uint32_t r = 0; r < SR_HPW; ++r) {
        const uint32_t head = warp + (r ^ slot) * (BLOCK / 32u);
#pragma unroll
        for (uint32_t i = 0; i < SR_DPL; ++i) {
            q[r][i] = head < heads ? q_row[head * SR_DIM + lane + 32u * i] : 0.0f;
        }
    }
    const uint32_t my_head = warp + slot * (BLOCK / 32u);
    const bool finisher = (lane & 3u) == 0u && my_head < heads;
    const float my_weight = finisher ? weights[static_cast<size_t>(row) * heads + my_head] : 0.0f;
    float* out = scores + static_cast<size_t>(row) * score_stride;
    for (uint32_t base = blockIdx.x * SR_TILE; base < n_pos; base += gridDim.x * SR_TILE) {
        const uint32_t tile = n_pos - base < SR_TILE ? n_pos - base : SR_TILE;
        for (uint32_t local = 0; local < tile; local += SR_PAIR) {
            float k[SR_PAIR][SR_DPL];
#pragma unroll
            for (uint32_t u = 0; u < SR_PAIR; ++u) {
                // A tile's odd last position scores its predecessor's key again, unstored.
                const uint32_t at = base + (local + u < tile ? local + u : local);
                const float* key = keys + static_cast<size_t>(at) * SR_DIM;
#pragma unroll
                for (uint32_t i = 0; i < SR_DPL; ++i) k[u][i] = key[lane + 32u * i];
            }
            float acc[SR_PAIR][SR_HPW];
#pragma unroll
            for (uint32_t u = 0; u < SR_PAIR; ++u) {
#pragma unroll
                for (uint32_t r = 0; r < SR_HPW; ++r) {
                    acc[u][r] = 0.0f;
#pragma unroll
                    for (uint32_t i = 0; i < SR_DPL; ++i) acc[u][r] = fmaf(q[r][i], k[u][i], acc[u][r]);
                }
            }
#pragma unroll
            for (uint32_t o = 16u, n = SR_HPW / 2u; n >= 1u; o >>= 1, n >>= 1) {
#pragma unroll
                for (uint32_t u = 0; u < SR_PAIR; ++u) {
#pragma unroll
                    for (uint32_t i = 0; i < n; ++i) {
                        acc[u][i] = acc[u][i] + __shfl_xor_sync(0xffffffffu, acc[u][n + i], o);
                    }
                }
            }
#pragma unroll
            for (uint32_t u = 0; u < SR_PAIR; ++u) {
                float dot = acc[u][0] + __shfl_xor_sync(0xffffffffu, acc[u][0], 2);
                dot += __shfl_xor_sync(0xffffffffu, dot, 1);
                if (finisher && local + u < tile) {
                    const float raw = round_bf16(dot);
                    s_c[local + u][my_head] = raw > 0.0f ? round_bf16(__fmul_rn(raw, my_weight)) : 0.0f;
                }
            }
        }
        __syncthreads();
        if (threadIdx.x < tile) {
            float sum = 0.0f;
            for (uint32_t head = 0; head < heads; ++head) sum = __fadd_rn(sum, s_c[threadIdx.x][head]);
            out[base + threadIdx.x] = round_bf16(sum);
        }
        __syncthreads();
    }
}

}  // namespace opt_select

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
// here as a threshold instead: a radix select (radix_threshold) finds the keep-th largest score
// key T and how many of the ids holding T are taken (`need`), and an id is chosen when its key
// exceeds T, or equals T and fewer than `need` lower ids hold T. The chosen ids are then written in
// ascending id order by a warp sweep. Scores map to keys by their float order, with -0 folded into
// +0 because the float compare ties them.
//
// LIMITS. Up to FAST_MAX_POS compressed positions, not the rank form's MAX_POS: the block keys
// (mask_mode 2) and the gathered candidates (mask_mode 1) live in shared memory up to MAX_POS
// positions and in `scratch` (n_pos * 5 bytes, global) beyond. The masks are read and written in
// place (global): the mask written is
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

constexpr uint32_t FAST_WARPS = FAST_BLOCK / 32;
// The radix select's widest digit: its first pass counts 12 bits.
constexpr uint32_t RADIX_BINS = 4096;

// true when an order key is a bf16 value's: its low 16 bits are then fixed by its top bit (0 for
// a non-negative value's key, all ones for a negative value's).
__device__ __forceinline__ bool bf16_key(uint32_t key) {
    return (key & 0xFFFFu) == ((key & 0x80000000u) != 0u ? 0u : 0xFFFFu);
}

// A stored score as f32: f32 scores as they are, bf16 ones (the bulk scorer's) widened exactly.
__device__ __forceinline__ float load_score(const float* p) { return *p; }
__device__ __forceinline__ float load_score(const uint16_t* p) {
    return __uint_as_float(static_cast<uint32_t>(*p) << 16);
}

// The key of id i: a query's score with the reach and a consumed candidate mask applied
// (opt_ref_select_body's two masks).
template <typename T>
struct ScoreKey {
    const T* __restrict__ pre;
    const uint32_t* __restrict__ mask;
    uint32_t reach;
    __device__ __forceinline__ float score(uint32_t i) const {
        float value = load_score(pre + i);
        if (i >= reach) value = -__builtin_huge_valf();
        if (mask != nullptr && (mask[i >> 5] & (1u << (i & 31u))) == 0u) value = -__builtin_huge_valf();
        return value;
    }
    __device__ __forceinline__ uint32_t operator()(uint32_t i) const { return order_key(score(i)); }
};

// The key of id i, stored (by this block, so read coherently).
struct StoredKey {
    const uint32_t* keys;
    __device__ __forceinline__ uint32_t operator()(uint32_t i) const { return keys[i]; }
};

// The keep-th largest key T of key(0 .. n) by (key descending, id ascending), 0 < keep < n, and
// how many of the ids holding T are chosen (`need`): an id is chosen when its key exceeds T, or
// equals T and fewer than `need` lower ids hold T. Under a strict total order that is exactly the
// rank form's chosen set. A radix select from the top: each pass counts the keys under the prefix
// found so far by their next digit (12, 4, 8 and 8 bits) and takes the digit holding the keep-th.
// When every key is a bf16 value's (bf16_key), the 16 bits under the first two digits are fixed
// by the top bit, so the last two passes are not run. `hist` holds RADIX_BINS words, `s_var` 2,
// `s_warp` 33. Every thread of the block calls it and gets the same T and need.
template <typename Key>
__device__ __forceinline__ void radix_threshold(const Key& key, uint32_t n, uint32_t keep,
                                                uint32_t* hist, uint32_t* s_var, uint32_t* s_warp,
                                                uint32_t& threshold, uint32_t& need) {
    const uint32_t tid = threadIdx.x;
    uint32_t prefix = 0u, prefix_mask = 0u, remaining = keep;
    bool all_bf16 = false;
#pragma unroll 1
    for (uint32_t pass = 0; pass < 4u; ++pass) {
        const uint32_t width = pass == 0u ? 12u : pass == 1u ? 4u : 8u;
        const uint32_t shift = pass == 0u ? 20u : pass == 1u ? 16u : pass == 2u ? 8u : 0u;
        const uint32_t bins = 1u << width;
        for (uint32_t i = tid; i < bins; i += FAST_BLOCK) hist[i] = 0u;
        __syncthreads();
        bool odd = false;
        for (uint32_t i0 = tid; i0 < n; i0 += 4u * FAST_BLOCK) {
            uint32_t k[4];
#pragma unroll
            for (uint32_t u = 0; u < 4u; ++u) {
                const uint32_t i = i0 + u * FAST_BLOCK;
                k[u] = i < n ? key(i) : 0u;
            }
#pragma unroll
            for (uint32_t u = 0; u < 4u; ++u) {
                if (i0 + u * FAST_BLOCK < n) {
                    odd |= !bf16_key(k[u]);
                    if ((k[u] & prefix_mask) == prefix) atomicAdd(&hist[(k[u] >> shift) & (bins - 1u)], 1u);
                }
            }
        }
        if (pass == 0u) {
            all_bf16 = __syncthreads_or(odd ? 1 : 0) == 0;
        } else {
            __syncthreads();
        }
        // The digits top down, `per` to a thread: the keep-th key's digit is the one where the
        // count of keys above it first reaches `remaining`.
        const uint32_t per = bins > FAST_BLOCK ? bins / FAST_BLOCK : 1u;
        const bool owns = tid * per < bins;
        const uint32_t top = owns ? bins - tid * per : 0u;
        uint32_t sum = 0u;
        if (owns) {
            for (uint32_t j = 0; j < per; ++j) sum += hist[top - 1u - j];
        }
        uint32_t above = block_exclusive(sum, s_warp);
        if (owns) {
            for (uint32_t j = 0; j < per; ++j) {
                const uint32_t count = hist[top - 1u - j];
                if (above < remaining && above + count >= remaining) {
                    s_var[0] = top - 1u - j;
                    s_var[1] = remaining - above;
                }
                above += count;
            }
        }
        __syncthreads();
        prefix |= s_var[0] << shift;
        remaining = s_var[1];
        prefix_mask |= (bins - 1u) << shift;
        if (pass == 1u && all_bf16) {
            prefix |= (prefix & 0x80000000u) != 0u ? 0u : 0xFFFFu;
            break;
        }
    }
    threshold = prefix;
    need = remaining;
}

// The chosen ids of key(0 .. n) given radix_threshold's T and need, ascending: visit(id, key,
// slot) runs once per chosen id, slot being its rank among them. Warp w sweeps the ids
// [w * span, (w + 1) * span) a lane per id, twice: the first sweep counts its keys above T and
// at T, so the second knows the ties and the chosen ids of every lower warp. `s_cnt` holds
// 2 * FAST_WARPS words.
template <typename Key, typename Visit>
__device__ __forceinline__ void choose_sweep(const Key& key, uint32_t n, uint32_t threshold,
                                             uint32_t need, uint32_t* s_cnt, Visit visit) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t span = ((n + FAST_WARPS - 1u) / FAST_WARPS + 31u) & ~31u;
    const uint32_t lo = warp * span < n ? warp * span : n;
    const uint32_t hi = n - lo < span ? n : lo + span;
    uint32_t above = 0u, ties = 0u;
    for (uint32_t i = lo + lane; i < hi; i += 32u) {
        const uint32_t k = key(i);
        above += k > threshold ? 1u : 0u;
        ties += k == threshold ? 1u : 0u;
    }
    above = __reduce_add_sync(0xFFFFFFFFu, above);
    ties = __reduce_add_sync(0xFFFFFFFFu, ties);
    if (lane == 0u) {
        s_cnt[warp] = above;
        s_cnt[FAST_WARPS + warp] = ties;
    }
    __syncthreads();
    uint32_t ties_before = 0u, above_before = 0u;
    for (uint32_t w = 0; w < warp; ++w) {
        above_before += s_cnt[w];
        ties_before += s_cnt[FAST_WARPS + w];
    }
    uint32_t chosen = above_before + (ties_before < need ? ties_before : need);
    const uint32_t lower = (1u << lane) - 1u;
    for (uint32_t base = lo; base < hi; base += 32u) {
        const uint32_t i = base + lane;
        const bool valid = i < hi;
        const uint32_t k = valid ? key(i) : 0u;
        const bool tie = valid && k == threshold;
        const uint32_t tie_bits = __ballot_sync(0xFFFFFFFFu, tie);
        const bool take = valid && (k > threshold || (tie && ties_before + __popc(tie_bits & lower) < need));
        const uint32_t take_bits = __ballot_sync(0xFFFFFFFFu, take);
        if (take) visit(i, k, chosen + __popc(take_bits & lower));
        ties_before += __popc(tie_bits);
        chosen += __popc(take_bits);
    }
    __syncthreads();
}

// The `keep` largest of key(0 .. n), visited as choose_sweep does.
template <typename Key, typename Visit>
__device__ __forceinline__ void choose_largest(const Key& key, uint32_t n, uint32_t keep,
                                               uint32_t* hist, uint32_t* s_var, uint32_t* s_warp,
                                               uint32_t* s_cnt, Visit visit) {
    if (keep == 0u) return;
    if (keep >= n) {
        for (uint32_t i = threadIdx.x; i < n; i += FAST_BLOCK) visit(i, key(i), i);
        __syncthreads();
        return;
    }
    uint32_t threshold, need;
    radix_threshold(key, n, keep, hist, s_var, s_warp, threshold, need);
    choose_sweep(key, n, threshold, need, s_cnt, visit);
}

// One query's select over its scores `pre` (the body of both kernels below). A chosen position p
// is written as p * id_scale + offset when it is reachable (no_index otherwise).
//
// A consumed candidate mask (mask_mode 1) leaves most positions at -inf: when more than `keep`
// positions are set, the set ones are gathered in ascending order (key and position, in
// `scratch`) and the select runs over those. A set position's key is above -inf's whenever the
// keep-th largest of them is, and then no cleared position can be chosen, so the chosen set is
// the full select's; otherwise the full select runs.
template <typename T>
__device__ __forceinline__ void fast_select(const T* __restrict__ pre, uint32_t* __restrict__ scratch,
                                            uint32_t* __restrict__ out_ids, float* __restrict__ out_scores,
                                            const uint32_t* __restrict__ mask_in,
                                            uint32_t* __restrict__ mask_out, uint32_t n_pos,
                                            uint32_t reach, uint32_t topk, uint32_t offset,
                                            uint32_t id_scale, uint32_t mask_mode,
                                            uint32_t topk_blocks, uint32_t block_size,
                                            uint32_t no_index) {
    __shared__ uint32_t s_store_small[MAX_POS];
    __shared__ uint32_t s_hist[RADIX_BINS];
    __shared__ uint32_t s_warp[33];
    __shared__ uint32_t s_var[2];
    __shared__ uint32_t s_cnt[2 * FAST_WARPS];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t n = n_pos < FAST_MAX_POS ? n_pos : FAST_MAX_POS;
    const bool small = n <= static_cast<uint32_t>(MAX_POS);
    uint32_t* store = small ? s_store_small : scratch;
    const uint32_t capacity = small ? static_cast<uint32_t>(MAX_POS) : n + n / 4u;
    const uint32_t n_words = (n + 31u) / 32u;
    const uint32_t out_words = n_words > static_cast<uint32_t>(MAX_MASK_WORDS) ? n_words : MAX_MASK_WORDS;
    const uint32_t empty = order_key(-__builtin_huge_valf());

    // Level one: the HSI candidate block mask (opt_ref_select_body's mask_mode 2).
    if (mask_mode == 2u && block_size != 0u && n != 0u && mask_out != nullptr) {
        const uint32_t num_blocks = (n + block_size - 1u) / block_size;
        for (uint32_t block = tid; block < num_blocks; block += FAST_BLOCK) {
            const uint32_t start = block * block_size;
            const uint32_t stop = (start + block_size) < n ? (start + block_size) : n;
            float best = -__builtin_huge_valf();
            for (uint32_t position = start; position < stop; ++position) {
                const float value = load_score(pre + position);
                best = value > best ? value : best;
            }
            store[block] = order_key(best);
        }
        for (uint32_t word = tid; word < out_words; word += FAST_BLOCK) mask_out[word] = 0u;
        __syncthreads();
        if (reach > 0u && tid == 0u) {
            const uint32_t last = (reach - 1u) / block_size;
            if (last < num_blocks) store[last] = order_key(__builtin_huge_valf());
        }
        __syncthreads();
        const uint32_t keep_blocks = topk_blocks < num_blocks ? topk_blocks : num_blocks;
        // A chosen block whose best score is -inf is dropped; a kept block sets its positions.
        choose_largest(StoredKey{store}, num_blocks, keep_blocks, s_hist, s_var, s_warp, s_cnt,
                       [&](uint32_t block, uint32_t key, uint32_t) {
                           if (key == empty) return;
                           const uint32_t start = block * block_size;
                           const uint32_t stop = (start + block_size) < n ? (start + block_size) : n;
                           for (uint32_t p = start; p < stop;) {
                               const uint32_t end = ((p >> 5) + 1u) * 32u < stop ? ((p >> 5) + 1u) * 32u : stop;
                               const uint32_t count = end - p;
                               const uint32_t bits = (count == 32u ? 0xFFFFFFFFu : (1u << count) - 1u) << (p & 31u);
                               atomicOr(&mask_out[p >> 5], bits);
                               p = end;
                           }
                       });
    }

    const ScoreKey<T> score_key{pre, mask_mode == 1u ? mask_in : nullptr, reach};
    if (out_scores != nullptr) {
        for (uint32_t position = tid; position < n; position += FAST_BLOCK) {
            out_scores[position] = score_key.score(position);
        }
    }
    const uint32_t keep = topk < n ? topk : n;
    const auto emit = [&](uint32_t position, uint32_t slot) {
        out_ids[slot] = position < reach ? position * id_scale + offset : no_index;
    };
    if (score_key.mask != nullptr && keep != 0u && keep < n) {
        // The set positions, counted by warp over the mask words, then gathered in order.
        const uint32_t span = ((n_words + FAST_WARPS - 1u) / FAST_WARPS + 31u) & ~31u;
        const uint32_t lo = warp * span < n_words ? warp * span : n_words;
        const uint32_t hi = n_words - lo < span ? n_words : lo + span;
        const auto set_bits = [&](uint32_t word) {
            const uint32_t bits = mask_in[word];
            return (word + 1u) * 32u <= n ? bits : bits & ((1u << (n - word * 32u)) - 1u);
        };
        uint32_t count = 0u;
        for (uint32_t word = lo + lane; word < hi; word += 32u) count += __popc(set_bits(word));
        count = __reduce_add_sync(0xFFFFFFFFu, count);
        if (lane == 0u) s_cnt[warp] = count;
        __syncthreads();
        uint32_t total = 0u, at = 0u;
        for (uint32_t w = 0; w < FAST_WARPS; ++w) {
            at += w < warp ? s_cnt[w] : 0u;
            total += s_cnt[w];
        }
        __syncthreads();
        if (total > keep && 2u * total <= capacity) {
            uint32_t* cand_pos = store + total;
            for (uint32_t base = lo; base < hi; base += 32u) {
                const uint32_t word = base + lane;
                uint32_t bits = word < hi ? set_bits(word) : 0u;
                const uint32_t mine = __popc(bits);
                uint32_t inclusive = mine;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const uint32_t up = __shfl_up_sync(0xFFFFFFFFu, inclusive, o);
                    if (lane >= static_cast<uint32_t>(o)) inclusive += up;
                }
                uint32_t slot = at + inclusive - mine;
                while (bits != 0u) {
                    const uint32_t position = word * 32u + static_cast<uint32_t>(__ffs(bits) - 1);
                    bits &= bits - 1u;
                    store[slot] = score_key(position);
                    cand_pos[slot] = position;
                    slot += 1u;
                }
                at += __shfl_sync(0xFFFFFFFFu, inclusive, 31);
            }
            __syncthreads();
            uint32_t threshold, need;
            radix_threshold(StoredKey{store}, total, keep, s_hist, s_var, s_warp, threshold, need);
            if (threshold > empty) {
                choose_sweep(StoredKey{store}, total, threshold, need, s_cnt,
                             [&](uint32_t i, uint32_t, uint32_t slot) { emit(cand_pos[i], slot); });
                return;
            }
        }
    }
    choose_largest(score_key, n, keep, s_hist, s_var, s_warp, s_cnt,
                   [&](uint32_t position, uint32_t, uint32_t slot) { emit(position, slot); });
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
// the most compressed positions n a row reaches). opt_select_rows reads f32 scores,
// opt_select_rows_bf16 the bulk scorer's bf16 ones (the same values, so the same choice).
// With `row_tab` (a step over several sequences' rows, DeviceLayer::upload_row_table's
// {lane offset, position} per row) row y's position is row_tab[2 y + 1] instead of p0 + y.
template <typename T>
__device__ __forceinline__ void select_rows_body(
    const T* __restrict__ scores, const uint64_t* __restrict__ row_tab,
    uint32_t* __restrict__ selected, const uint32_t* __restrict__ mask_in,
    uint32_t* __restrict__ mask_out, uint32_t* __restrict__ scratch, uint32_t tokens, uint32_t p0,
    uint32_t ratio, uint32_t topk, uint32_t mask_mode, uint32_t topk_blocks, uint32_t block_size,
    uint32_t score_stride, uint32_t selected_stride, uint32_t mask_stride, uint32_t scratch_words) {
    uint32_t* mine = scratch + static_cast<size_t>(blockIdx.x) * scratch_words;
    for (uint32_t y = blockIdx.x; y < tokens; y += gridDim.x) {
        const uint32_t position =
            row_tab != nullptr ? static_cast<uint32_t>(row_tab[2u * y + 1u]) : p0 + y;
        const uint32_t n = (position + 1u) / ratio;
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

#define OPT_SELECT_ROWS_PARAMS                                                                     \
    uint32_t *__restrict__ selected, const uint32_t *__restrict__ mask_in,                         \
        uint32_t *__restrict__ mask_out, uint32_t *__restrict__ scratch, uint32_t tokens,          \
        uint32_t p0, uint32_t ratio, uint32_t topk, uint32_t mask_mode, uint32_t topk_blocks,      \
        uint32_t block_size, uint32_t score_stride, uint32_t selected_stride, uint32_t mask_stride, \
        uint32_t scratch_words
#define OPT_SELECT_ROWS_ARGS                                                                        \
    selected, mask_in, mask_out, scratch, tokens, p0, ratio, topk, mask_mode, topk_blocks,         \
        block_size, score_stride, selected_stride, mask_stride, scratch_words

extern "C" __global__ void __launch_bounds__(FAST_BLOCK) opt_select_rows(
    const float* __restrict__ scores, OPT_SELECT_ROWS_PARAMS) {
    OPT_PDL_ENTRY();
    select_rows_body(scores, static_cast<const uint64_t*>(nullptr), OPT_SELECT_ROWS_ARGS);
}

extern "C" __global__ void __launch_bounds__(FAST_BLOCK) opt_select_rows_bf16(
    const uint16_t* __restrict__ scores, OPT_SELECT_ROWS_PARAMS) {
    OPT_PDL_ENTRY();
    select_rows_body(scores, static_cast<const uint64_t*>(nullptr), OPT_SELECT_ROWS_ARGS);
}

// opt_select_rows over a step's rows from several sequences, each row's position from `row_tab`
// (p0 unused).
extern "C" __global__ void __launch_bounds__(FAST_BLOCK) opt_select_rows_tab(
    const float* __restrict__ scores, const uint64_t* __restrict__ row_tab,
    OPT_SELECT_ROWS_PARAMS) {
    OPT_PDL_ENTRY();
    select_rows_body(scores, row_tab, OPT_SELECT_ROWS_ARGS);
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

// opt_select_scores_rows' scores, bit for bit, for a 32-head indexer, with the rows of one segment
// (one sequence's positions p0 .. p0 + rows - 1) sharing each key tile: the keys are read once a
// segment, not once a row. Grid (blocks), SG_SLICE * rows * slices threads (rows <= SG_ROWS): a
// row slice is 64 threads, lane octet 8g .. 8g + 7 holding heads g, g + 8, g + 16 and g + 24's
// queries in registers, and slice z scores row z % rows at the tile's positions z / rows,
// z / rows + slices, .... The next tile streams in (cp.async, a second buffer) while this one is
// scored.
//
// A dot is the same 32 lane chains (lane L: dims L, L + 32, L + 64, L + 96, fma ascending) and the
// same xor tree. The octet splits the chains by L % 8: the tree's levels 16 and 8 keep L % 8, so
// each thread folds its 4 chains a head without leaving the thread, and levels 4, 2 and 1 are the
// octet's shuffles, transposed as opt_select_scores_rows' upper levels are (a thread keeps half
// its heads and sends the other half to its partner, which keeps those). Each key value a thread
// reads feeds its four heads (a shared-memory broadcast costs about four cycles a warp, so a key
// value must feed four fmas). Then a thread per (row, position) folds the heads in ascending order.
// A head's term is a bf16 value (or 0), so the terms wait in bf16.
//
// mask_in (mask_mode 1, a consumed candidate mask; row y's at mask_in + y * mask_stride words): a
// position whose bit is clear in a row's mask is not scored for that row (the select overwrites
// its score with -inf), and a tile no row needs is not read.
constexpr uint32_t SG_HEADS = 32;
constexpr uint32_t SG_TILE = 32;
constexpr uint32_t SG_ROWS = 8;
constexpr uint32_t SG_SLICE = 64;
constexpr uint32_t SG_CHAINS = 4;
constexpr uint32_t SG_GROUP = 4;

__device__ __forceinline__ void sg_copy4(float* dst, const float* src) {
    const uint32_t to = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(to), "l"(src) : "memory");
}

// The first tile at or after `base` (stepping `step`) that some row needs, with the union of the
// rows' mask words and this row's; n_top when none is left.
__device__ __forceinline__ uint32_t sg_next(const uint32_t* __restrict__ mask_in, uint32_t base,
                                            uint32_t step, uint32_t n_top, uint32_t rows,
                                            uint32_t mask_stride, uint32_t row, uint32_t& need,
                                            uint32_t& mine) {
    need = 0xFFFFFFFFu;
    mine = 0xFFFFFFFFu;
    if (mask_in == nullptr) return base < n_top ? base : n_top;
    for (; base < n_top; base += step) {
        need = 0u;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t word = mask_in[static_cast<size_t>(r) * mask_stride + base / 32u];
            need |= word;
            mine = r == row ? word : mine;
        }
        if (need != 0u) return base;
    }
    return n_top;
}

// The scores of one segment's rows from block bx of `blocks`, with `slices` row slices (threads
// past rows * slices slices fetch and fold but score nothing).
__device__ __forceinline__ void scores_seg_body(
    const float* __restrict__ q_index, const float* __restrict__ keys,
    const float* __restrict__ weights, float* __restrict__ scores,
    const uint32_t* __restrict__ mask_in, uint32_t p0, uint32_t ratio, uint32_t score_stride,
    uint32_t mask_stride, uint32_t rows, uint32_t slices, uint32_t bx, uint32_t blocks) {
    // Two key tiles, dim 32 i + L at [i * 32 + (L & 7) * 4 + (L >> 3)]: a thread's chains in order.
    __shared__ __align__(16) float s_k[2][SG_TILE][SR_DIM];
    // The head terms, [row][position][head].
    __shared__ __align__(16) __nv_bfloat16 s_c[SG_ROWS][SG_TILE][SG_HEADS];
    const uint32_t tid = threadIdx.x;
    const uint32_t z = tid / SG_SLICE;
    const uint32_t row = z % rows, slice = z / rows;
    const uint32_t group = (tid >> 3) & 7u, eighth = tid & 7u;
    const uint32_t reach_top = (p0 + rows) / ratio;
    const uint32_t n_top = reach_top < FAST_MAX_POS ? reach_top : FAST_MAX_POS;
    const uint32_t reach = (p0 + row + 1u) / ratio;
    const uint32_t n_row = reach < FAST_MAX_POS ? reach : FAST_MAX_POS;
    const uint32_t step = blocks * SG_TILE;
    // q[g][i][m]: dim 32 i + 8 m + eighth of head group + 8 g of this row.
    float q[SG_GROUP][SR_DPL][SG_CHAINS];
#pragma unroll
    for (uint32_t g = 0; g < SG_GROUP; ++g) {
        const float* q_head =
            q_index + (static_cast<size_t>(row) * SG_HEADS + group + 8u * g) * SR_DIM + eighth;
#pragma unroll
        for (uint32_t i = 0; i < SR_DPL; ++i) {
#pragma unroll
            for (uint32_t m = 0; m < SG_CHAINS; ++m) q[g][i][m] = q_head[32u * i + 8u * m];
        }
    }
    // After the transposed levels lane (eighth) holds head group + 8 j, j = 2 * bit 2 + bit 1;
    // the even lane of each pair finishes it.
    const uint32_t hi = (eighth >> 2) & 1u, mid = (eighth >> 1) & 1u;
    const uint32_t my_head = group + 8u * (2u * hi + mid);
    const float weight = weights[static_cast<size_t>(row) * SG_HEADS + my_head];
    // Thread t copies the words t, t + blockDim, ... of a tile's SG_TILE keys.
    auto fetch = [&](uint32_t base, uint32_t need, uint32_t buf) {
        const uint32_t tile = n_top - base < SG_TILE ? n_top - base : SG_TILE;
        for (uint32_t c = tid; c < SG_TILE * SR_DIM; c += blockDim.x) {
            const uint32_t p = c / SR_DIM, d = c % SR_DIM, l = d & 31u;
            if (p < tile && ((need >> p) & 1u) != 0u) {
                sg_copy4(&s_k[buf][p][(d & ~31u) + (l & 7u) * 4u + (l >> 3)],
                         keys + static_cast<size_t>(base + p) * SR_DIM + d);
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };
    uint32_t need, mine;
    uint32_t base = sg_next(mask_in, bx * SG_TILE, step, n_top, rows, mask_stride, row, need, mine);
    uint32_t buf = 0u;
    if (base < n_top) fetch(base, need, buf);
    while (base < n_top) {
        uint32_t need_next, mine_next;
        const uint32_t next =
            sg_next(mask_in, base + step, step, n_top, rows, mask_stride, row, need_next, mine_next);
        if (next < n_top) {
            fetch(next, need_next, buf ^ 1u);
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        }
        __syncthreads();
        const uint32_t tile = n_top - base < SG_TILE ? n_top - base : SG_TILE;
        for (uint32_t p = slice; slice < slices && p < tile; p += slices) {
            if (base + p >= n_row || ((mine >> p) & 1u) == 0u) continue;
            const float* k = &s_k[buf][p][eighth * 4u];
            float c[SG_GROUP][SG_CHAINS];
#pragma unroll
            for (uint32_t i = 0; i < SR_DPL; ++i) {
                const float4 kv = *reinterpret_cast<const float4*>(k + 32u * i);
#pragma unroll
                for (uint32_t g = 0; g < SG_GROUP; ++g) {
                    c[g][0] = fmaf(q[g][i][0], kv.x, i == 0u ? 0.0f : c[g][0]);
                    c[g][1] = fmaf(q[g][i][1], kv.y, i == 0u ? 0.0f : c[g][1]);
                    c[g][2] = fmaf(q[g][i][2], kv.z, i == 0u ? 0.0f : c[g][2]);
                    c[g][3] = fmaf(q[g][i][3], kv.w, i == 0u ? 0.0f : c[g][3]);
                }
            }
            // Chain m is lane L = 8 m + eighth: levels 16 and 8 pair m with m ^ 2 and m ^ 1.
            float t[SG_GROUP];
#pragma unroll
            for (uint32_t g = 0; g < SG_GROUP; ++g) {
                const float a = c[g][0] + c[g][2], b = c[g][1] + c[g][3];
                t[g] = a + b;
            }
            // Level 4 (lane ^ 4): keep heads 2 hi, 2 hi + 1, send the other two.
            const float k0 = hi ? t[2] : t[0], k1 = hi ? t[3] : t[1];
            const float r0 = __shfl_xor_sync(0xffffffffu, hi ? t[0] : t[2], 4);
            const float r1 = __shfl_xor_sync(0xffffffffu, hi ? t[1] : t[3], 4);
            const float u0 = k0 + r0, u1 = k1 + r1;
            // Level 2 (lane ^ 2): keep head 2 hi + mid; level 1 (lane ^ 1).
            const float v = (mid ? u1 : u0) + __shfl_xor_sync(0xffffffffu, mid ? u0 : u1, 2);
            const float dot = v + __shfl_xor_sync(0xffffffffu, v, 1);
            if ((eighth & 1u) == 0u) {
                const float raw = round_bf16(dot);
                s_c[row][p][my_head] =
                    __float2bfloat16_rn(raw > 0.0f ? round_bf16(__fmul_rn(raw, weight)) : 0.0f);
            }
        }
        __syncthreads();
        for (uint32_t item = tid; item < rows * SG_TILE; item += blockDim.x) {
            const uint32_t r = item / SG_TILE, p = item % SG_TILE;
            const uint32_t reach_r = (p0 + r + 1u) / ratio;
            const uint32_t word =
                mask_in != nullptr ? mask_in[static_cast<size_t>(r) * mask_stride + base / 32u] : 0xFFFFFFFFu;
            if (p < tile && base + p < reach_r && ((word >> p) & 1u) != 0u) {
                const uint4* terms = reinterpret_cast<const uint4*>(&s_c[r][p][0]);
                float sum = 0.0f;
#pragma unroll
                for (uint32_t v = 0; v < SG_HEADS / 8u; ++v) {
                    const uint4 w4 = terms[v];
                    const uint32_t w[4] = {w4.x, w4.y, w4.z, w4.w};
#pragma unroll
                    for (uint32_t h = 0; h < 4u; ++h) {
                        sum = __fadd_rn(sum, __uint_as_float(w[h] << 16));
                        sum = __fadd_rn(sum, __uint_as_float(w[h] & 0xFFFF0000u));
                    }
                }
                scores[static_cast<size_t>(r) * score_stride + base + p] = round_bf16(sum);
            }
        }
        base = next;
        need = need_next;
        mine = mine_next;
        buf ^= 1u;
    }
}

extern "C" __global__ void __launch_bounds__(SG_SLICE * SG_ROWS) opt_select_scores_seg(
    const float* __restrict__ q_index, const float* __restrict__ keys,
    const float* __restrict__ weights, float* __restrict__ scores,
    const uint32_t* __restrict__ mask_in, uint32_t p0, uint32_t ratio, uint32_t score_stride,
    uint32_t mask_stride, uint32_t rows) {
    OPT_PDL_ENTRY();
    scores_seg_body(q_index, keys, weights, scores, mask_in, p0, ratio, score_stride, mask_stride,
                    rows, blockDim.x / (SG_SLICE * rows), blockIdx.x, gridDim.x);
}

// opt_select_scores_seg over a step's rows from several sequences, in one launch: block row y
// takes piece y of `pieces` ({row0, rows} words, rows <= SG_ROWS, one sequence's consecutive
// positions), whose first position and lane offset are row_tab's at row0
// (DeviceLayer::upload_row_table) and whose keys are at keys + that offset. A piece's scores are
// opt_select_scores_seg's over its rows (its rows' at their step rows), with SG_ROWS / rows slices.
extern "C" __global__ void __launch_bounds__(SG_SLICE * SG_ROWS) opt_select_scores_segs(
    const float* __restrict__ q_index, const float* __restrict__ keys,
    const float* __restrict__ weights, float* __restrict__ scores,
    const uint32_t* __restrict__ mask_in, const uint64_t* __restrict__ row_tab,
    const uint32_t* __restrict__ pieces, uint32_t ratio, uint32_t score_stride,
    uint32_t mask_stride) {
    OPT_PDL_ENTRY();
    const uint32_t row0 = pieces[2u * blockIdx.y], rows = pieces[2u * blockIdx.y + 1u];
    if (rows == 0u || rows > SG_ROWS) return;
    const uint64_t offset = row_tab[2u * row0];
    const uint32_t p0 = static_cast<uint32_t>(row_tab[2u * row0 + 1u]);
    scores_seg_body(q_index + static_cast<size_t>(row0) * SG_HEADS * SR_DIM,
                    reinterpret_cast<const float*>(reinterpret_cast<const char*>(keys) + offset),
                    weights + static_cast<size_t>(row0) * SG_HEADS,
                    scores + static_cast<size_t>(row0) * score_stride,
                    mask_in != nullptr ? mask_in + static_cast<size_t>(row0) * mask_stride : nullptr,
                    p0, ratio, score_stride, mask_stride, rows, SG_ROWS / rows, blockIdx.x,
                    gridDim.x);
}

}  // namespace opt_select

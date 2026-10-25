// crates/sparkflash-kernels/scripts/opt_select_fast_check.cu
//
// A device check, not a cargo test: opt_ref_select_fast (the radix select) against a host
// transcription of the rank rules on random selects up to FAST_MAX_POS (2^21) positions, and against
// opt_ref_select_pre (the O(n^2) rank form it replaces) where that one reaches (4096 positions).
// The inputs have heavy score ties, -0/+0, -inf, the reach cut, all three mask modes and
// block-mask cuts. Ids, the produced mask and the masked scores must match bit for bit. Run on a
// Spark from the repository root:
//
//   nvcc -std=c++17 -arch=sm_121a -O2 -I kernels \
//       crates/sparkflash-kernels/scripts/opt_select_fast_check.cu -o /tmp/opt_select_fast_check
//   /tmp/opt_select_fast_check
//
// It prints the case counts, the mismatches and the kernels' mean times, and exits non-zero on a
// mismatch.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "opt_select_sm121.cu"
using namespace opt_select;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("cuda %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

struct Case { uint32_t n, topk, reach, mode, block_size, topk_blocks; };

static bool before_host(float a, uint32_t ia, float b, uint32_t ib) {
    if (a > b) return true;
    if (a < b) return false;
    return ia < ib;
}

// The rank rules of kernels/opt_select_sm121.cu's header, on the host.
static void host_select(const Case& c, const std::vector<float>& pre, const std::vector<uint32_t>& mask_in,
                        std::vector<uint32_t>& ids, std::vector<uint32_t>& mask_out, std::vector<float>& scores) {
    const uint32_t n = c.n, words = std::max<uint32_t>(MAX_MASK_WORDS, (n + 31) / 32);
    std::vector<uint32_t> mask(words, 0u);
    if (c.mode == 1) for (uint32_t w = 0; w < (n + 31) / 32; ++w) mask[w] = mask_in[w];
    if (c.mode == 2 && c.block_size != 0) {
        const uint32_t nb = (n + c.block_size - 1) / c.block_size;
        std::vector<float> best(nb, -__builtin_huge_valf());
        for (uint32_t p = 0; p < n; ++p) best[p / c.block_size] = std::max(best[p / c.block_size], pre[p]);
        if (c.reach > 0 && (c.reach - 1) / c.block_size < nb) best[(c.reach - 1) / c.block_size] = __builtin_huge_valf();
        std::vector<uint32_t> order(nb);
        for (uint32_t b = 0; b < nb; ++b) order[b] = b;
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return before_host(best[a], a, best[b], b); });
        std::vector<char> kept(nb, 0);
        for (uint32_t i = 0; i < std::min(c.topk_blocks, nb); ++i) kept[order[i]] = 1;
        for (uint32_t p = 0; p < n; ++p) {
            const uint32_t b = p / c.block_size;
            if (kept[b] && best[b] > -__builtin_huge_valf()) mask[p >> 5] |= 1u << (p & 31);
        }
        mask_out = mask;
    }
    scores.assign(n, 0.0f);
    for (uint32_t p = 0; p < n; ++p) {
        float s = pre[p];
        if (p >= c.reach) s = -__builtin_huge_valf();
        if (c.mode == 1 && (mask[p >> 5] & (1u << (p & 31))) == 0) s = -__builtin_huge_valf();
        scores[p] = s;
    }
    std::vector<uint32_t> order(n);
    for (uint32_t p = 0; p < n; ++p) order[p] = p;
    const uint32_t keep = std::min(c.topk, n);
    std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                      [&](uint32_t a, uint32_t b) { return before_host(scores[a], a, scores[b], b); });
    std::sort(order.begin(), order.begin() + keep);
    ids.assign(keep, 0);
    for (uint32_t i = 0; i < keep; ++i) ids[i] = order[i] < c.reach ? order[i] + 128u : 0xFFFFFFFFu;
}

int main() {
    std::mt19937 rng(1234);
    const uint32_t cap = FAST_MAX_POS, words_cap = cap / 32;
    float *d_pre, *d_scores; uint32_t *d_ids, *d_mask_in, *d_mask, *d_scratch;
    CK(cudaMalloc(&d_pre, cap * 4)); CK(cudaMalloc(&d_scores, cap * 4)); CK(cudaMalloc(&d_ids, cap * 4));
    CK(cudaMalloc(&d_mask_in, words_cap * 4)); CK(cudaMalloc(&d_mask, words_cap * 4)); CK(cudaMalloc(&d_scratch, cap * 8));
    int fails = 0, cases = 0, rank_cases = 0;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float ms_rank = 0, ms_fast_small = 0, ms_fast_big = 0; int small = 0, big = 0;
    for (int it = 0; it < 2000; ++it) {
        Case c;
        // Most large cases stay below 65536 so the host reference keeps the run short; every tenth
        // reaches the whole capacity.
        c.n = (it % 2 == 0) ? 1 + rng() % 4096 : 1 + rng() % (it % 10 == 1 ? cap : 65536);
        if (it % 7 == 0) c.n = 2052; if (it % 11 == 0) c.n = 4096; if (it % 13 == 0) c.n = 4097; if (it % 17 == 0) c.n = cap;
        if (it % 19 == 0) c.n = 32768; if (it % 23 == 0) c.n = (1u << 20) + 7;
        c.topk = (it % 3 == 0) ? 512 : 1 + rng() % 1200;
        c.reach = (it % 4 == 0) ? c.n : 1 + rng() % c.n;
        c.mode = rng() % 3;
        c.block_size = (it % 5 == 0) ? 1 + rng() % 16 : 8;
        c.topk_blocks = (it % 2 == 0) ? 2048 : 1 + rng() % 600;
        const int levels = (it % 3 == 0) ? 4 : (it % 3 == 1 ? 64 : 1 << 20);
        std::vector<float> pre(c.n); std::vector<uint32_t> mask_in(words_cap);
        for (uint32_t i = 0; i < c.n; ++i) {
            const int v = (int)(rng() % levels) - levels / 3;
            float f = (float)v / 7.0f;
            if (rng() % 50 == 0) f = -__builtin_huge_valf();
            if (rng() % 97 == 0) f = -0.0f;
            if (rng() % 89 == 0) f = 0.0f;
            pre[i] = __bfloat162float(__float2bfloat16_rn(f));
        }
        for (auto& w : mask_in) w = rng() | rng();
        CK(cudaMemcpy(d_pre, pre.data(), c.n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_mask_in, mask_in.data(), words_cap * 4, cudaMemcpyHostToDevice));
        std::vector<uint32_t> want_ids, want_mask; std::vector<float> want_scores;
        host_select(c, pre, mask_in, want_ids, want_mask, want_scores);
        const uint32_t out_words = std::max<uint32_t>(MAX_MASK_WORDS, (c.n + 31) / 32);
        const uint32_t keep = std::min(c.topk, c.n);
        for (int kernel = 0; kernel < 2; ++kernel) {
            if (kernel == 1 && c.n > (uint32_t)MAX_POS) continue;
            CK(cudaMemset(d_ids, 0xEE, cap * 4)); CK(cudaMemset(d_mask, 0xEE, words_cap * 4));
            const uint32_t* mi = c.mode == 1 ? d_mask_in : nullptr;
            uint32_t* mo = c.mode == 2 ? d_mask : nullptr;
            cudaEventRecord(e0);
            if (kernel == 0) {
                opt_ref_select_fast<<<1, FAST_BLOCK>>>(nullptr, nullptr, nullptr, d_ids, d_scores, mi, mo, c.n, 32, 128, c.reach,
                                                       c.topk, 128, c.mode, c.topk_blocks, c.block_size, 0xFFFFFFFFu, 0, d_pre, d_scratch);
            } else {
                opt_ref_select_pre<<<1, 256>>>(nullptr, nullptr, nullptr, d_ids, d_scores, mi, mo, c.n, 32, 128, c.reach,
                                               c.topk, 128, c.mode, c.topk_blocks, c.block_size, 0xFFFFFFFFu, 0, d_pre);
            }
            cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float t; cudaEventElapsedTime(&t, e0, e1);
            if (kernel == 1) { ms_rank += t; ++rank_cases; }
            else if (c.n <= (uint32_t)MAX_POS) { ms_fast_small += t; ++small; } else { ms_fast_big += t; ++big; }
            std::vector<uint32_t> ids(keep), mask(out_words); std::vector<float> scores(c.n);
            CK(cudaMemcpy(ids.data(), d_ids, keep * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(mask.data(), d_mask, out_words * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(scores.data(), d_scores, c.n * 4, cudaMemcpyDeviceToHost));
            const bool ok_ids = ids == want_ids;
            const bool ok_mask = c.mode != 2 || mask == want_mask;
            const bool ok_scores = memcmp(scores.data(), want_scores.data(), c.n * 4) == 0;
            ++cases;
            if (!(ok_ids && ok_mask && ok_scores)) {
                if (fails < 8) printf("MISMATCH kernel=%s it=%d n=%u topk=%u reach=%u mode=%u bs=%u tb=%u ids=%d mask=%d scores=%d\n",
                                      kernel ? "rank" : "fast", it, c.n, c.topk, c.reach, c.mode, c.block_size, c.topk_blocks,
                                      ok_ids, ok_mask, ok_scores);
                ++fails;
            }
        }
    }
    printf("%d comparisons, %d mismatches; rank %.1f us/call, fast %.1f us/call (<= 4096), fast %.1f us/call (> 4096)\n",
           cases, fails, 1e3 * ms_rank / rank_cases, 1e3 * ms_fast_small / small, 1e3 * ms_fast_big / big);
    return fails != 0;
}

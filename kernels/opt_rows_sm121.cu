// kernels/opt_rows_sm121.cu
//
// The multi-row (speculative verify) twins of the decode step's per-row kernels. Row r of a launch
// is position p0 + r; blockIdx.y (or blockIdx.x where noted) selects the row, and every row's
// arithmetic is the single-row kernel's, expression for expression, so a row computed here is
// bit-identical to the same position computed by the M = 1 path.
//
// Row buffers are contiguous [rows][natural width] unless a stride argument says otherwise.
//
//   opt_rows_mhc_seam     opt_mhc_seam per row (grid 80 x rows): per-row partials, row_buf, ticket.
//   opt_rows_mhc_post     opt_mhc_post per row (grid 40 x rows).
//   opt_rows_kv           opt_kv_norm_rope + window_quant_fp8_bf16 + kv_append_bf16 per row (grid rows).
//   opt_rows_attn         opt_flash_mla_csa2_seq without the HSI prologue (grid heads x rows); the
//                         window and the selected count are derived from the row's position.
//   opt_rows_inv_rope     rope_rotate_rows_bf16 with sin_sign = -1 (grid heads x rows).
//   opt_rows_lm_head      opt_lm_head per row (grid chunks x rows).
//   opt_rows_engram_wkv   opt_engram_wkv over up to 8 rows with one weight read.
//   opt_rows_hostw_*      b12x_hostw_gemv_m1 per row (grid cols/32 x rows).
//   opt_rows_argmax_pick  the cross-rank argmax over exchanged folds (the host gather's rule).
//   opt_rows_topc         a row's slice statistics for the truncated (top-k / top-p) sampler.
//   opt_rows_embed_*      the rows' embedding streams from device ids, over the exchange.

#include "opt_pdl_sm121.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "m4_common_sm121.cuh"
#include "m4_activation_quant_sm121.cuh"
#include "opt_sample_sm121.cuh"

// ---------------------------------------------------------------------------------------------
// mHC seam and post
// ---------------------------------------------------------------------------------------------
#define OR_S_HIDDEN 5120u
#define OR_S_HC 4u
#define OR_S_WIDTH (OR_S_HC * OR_S_HIDDEN)
#define OR_S_ROWS 24u
#define OR_S_THREADS 256u
#define OR_S_WARPS (OR_S_THREADS / 32u)
#define OR_S_COLS 64u
#define OR_S_BLOCKS (OR_S_HIDDEN / OR_S_COLS)
#define OR_S_PART 26u
#define OR_S_PART_STRIDE 32u
#define OR_S_QBLOCKS (OR_S_HIDDEN / 32u)

__device__ __forceinline__ float or_bf16_round(float value) {
    const uint32_t bits = __float_as_uint(value);
    if (value != value) {
        return __uint_as_float((((bits >> 16) | 0x0040u)) << 16);
    }
    const uint32_t round = 0x7FFFu + ((bits >> 16) & 1u);
    return __uint_as_float(((bits + round) >> 16) << 16);
}

__device__ __forceinline__ float or_sigmoid(float value) {
    if (value >= 0.0f) {
        return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-value)));
    }
    const float exponential = expf(value);
    return __fdiv_rn(exponential, __fadd_rn(1.0f, exponential));
}

__device__ __forceinline__ float or_warp_sum_rn(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(value, __shfl_down_sync(0xFFFFFFFFu, value, offset));
    }
    return value;
}

__device__ __forceinline__ void or_normalise_columns(float comb[OR_S_HC][OR_S_HC], float hc_eps) {
    for (uint32_t column = 0; column < OR_S_HC; ++column) {
        float sum = hc_eps;
        for (uint32_t row = 0; row < OR_S_HC; ++row) sum = __fadd_rn(sum, comb[row][column]);
        const float inverse = __fdiv_rn(1.0f, sum);
        for (uint32_t row = 0; row < OR_S_HC; ++row) comb[row][column] = __fmul_rn(comb[row][column], inverse);
    }
}

__device__ void or_coefficient_tail(const float *raw, float sum_squares, const float *hc_scale,
                                    const float *hc_base, float norm_eps, float hc_eps,
                                    uint32_t iterations, float *mix_table, float *pre_mix_out) {
    const uint32_t hc = OR_S_HC;
    const float mean_square = __fdiv_rn(sum_squares, (float)OR_S_WIDTH);
    const float rsqrt = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean_square, norm_eps)));
    float mix[OR_S_ROWS];
#pragma unroll
    for (uint32_t row = 0; row < OR_S_ROWS; ++row) mix[row] = __fmul_rn(raw[row], rsqrt);
    for (uint32_t c = 0; c < hc; ++c) {
        const float pre = __fadd_rn(or_sigmoid(__fadd_rn(__fmul_rn(mix[c], hc_scale[0]), hc_base[c])), hc_eps);
        mix_table[c] = pre;
        if (pre_mix_out != nullptr) pre_mix_out[c] = pre;
    }
    for (uint32_t c = 0; c < hc; ++c) {
        const uint32_t row = hc + c;
        mix_table[row] = __fmul_rn(2.0f, or_sigmoid(__fadd_rn(__fmul_rn(mix[row], hc_scale[1]), hc_base[row])));
    }
    float comb[OR_S_HC][OR_S_HC];
    for (uint32_t row = 0; row < hc; ++row) {
        for (uint32_t column = 0; column < hc; ++column) {
            const uint32_t k = 2u * hc + row * hc + column;
            comb[row][column] = __fadd_rn(__fmul_rn(mix[k], hc_scale[2]), hc_base[k]);
        }
    }
    for (uint32_t row = 0; row < hc; ++row) {
        float maximum = comb[row][0];
        for (uint32_t column = 1; column < hc; ++column) maximum = comb[row][column] > maximum ? comb[row][column] : maximum;
        float total = 0.0f;
        for (uint32_t column = 0; column < hc; ++column) {
            const float exponential = expf(__fadd_rn(comb[row][column], -maximum));
            comb[row][column] = exponential;
            total = __fadd_rn(total, exponential);
        }
        for (uint32_t column = 0; column < hc; ++column) comb[row][column] = __fadd_rn(__fdiv_rn(comb[row][column], total), hc_eps);
    }
    or_normalise_columns(comb, hc_eps);
    for (uint32_t iteration = 1; iteration < iterations; ++iteration) {
        for (uint32_t row = 0; row < hc; ++row) {
            float sum = 0.0f;
            for (uint32_t column = 0; column < hc; ++column) sum = __fadd_rn(sum, comb[row][column]);
            const float inverse = __fdiv_rn(1.0f, __fadd_rn(sum, hc_eps));
            for (uint32_t column = 0; column < hc; ++column) comb[row][column] = __fmul_rn(comb[row][column], inverse);
        }
        or_normalise_columns(comb, hc_eps);
    }
    for (uint32_t row = 0; row < hc; ++row)
        for (uint32_t column = 0; column < hc; ++column) mix_table[2u * hc + row * hc + column] = comb[row][column];
}

// or_coefficient_tail on one warp: lane i holds comb[(i & 15) / 4][i % 4] and gathers its row or
// column with shuffles. Every max, sum and reciprocal takes the serial tail's operands in the serial
// tail's order, so the table is bit-identical, while the twenty Sinkhorn rounds run 16 wide.
__device__ void or_coefficient_tail_warp(const float *raw, float sum_squares, const float *hc_scale,
                                         const float *hc_base, float norm_eps, float hc_eps,
                                         uint32_t iterations, float *mix_table, float *pre_mix_out) {
    const uint32_t hc = OR_S_HC;
    const uint32_t lane = threadIdx.x & 31u;
    const float mean_square = __fdiv_rn(sum_squares, (float)OR_S_WIDTH);
    const float rsqrt = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean_square, norm_eps)));
    if (lane < hc) {
        const float pre =
            __fadd_rn(or_sigmoid(__fadd_rn(__fmul_rn(__fmul_rn(raw[lane], rsqrt), hc_scale[0]), hc_base[lane])), hc_eps);
        mix_table[lane] = pre;
        if (pre_mix_out != nullptr) pre_mix_out[lane] = pre;
    } else if (lane < 2u * hc) {
        mix_table[lane] =
            __fmul_rn(2.0f, or_sigmoid(__fadd_rn(__fmul_rn(__fmul_rn(raw[lane], rsqrt), hc_scale[1]), hc_base[lane])));
    }
    const uint32_t e = lane & 15u;
    const uint32_t r = e >> 2;
    const uint32_t c = e & 3u;
    const uint32_t k = 2u * hc + e;
    float v = __fadd_rn(__fmul_rn(__fmul_rn(raw[k], rsqrt), hc_scale[2]), hc_base[k]);
    {
        float row[OR_S_HC];
#pragma unroll
        for (uint32_t j = 0; j < hc; ++j) row[j] = __shfl_sync(0xFFFFFFFFu, v, r * hc + j);
        float maximum = row[0];
#pragma unroll
        for (uint32_t j = 1; j < hc; ++j) maximum = row[j] > maximum ? row[j] : maximum;
        float total = 0.0f;
        float own = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < hc; ++j) {
            const float exponential = expf(__fadd_rn(row[j], -maximum));
            if (j == c) own = exponential;
            total = __fadd_rn(total, exponential);
        }
        v = __fadd_rn(__fdiv_rn(own, total), hc_eps);
    }
    auto normalise_column = [&]() {
        float sum = hc_eps;
#pragma unroll
        for (uint32_t j = 0; j < hc; ++j) sum = __fadd_rn(sum, __shfl_sync(0xFFFFFFFFu, v, j * hc + c));
        v = __fmul_rn(v, __fdiv_rn(1.0f, sum));
    };
    normalise_column();
    for (uint32_t iteration = 1; iteration < iterations; ++iteration) {
        float sum = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < hc; ++j) sum = __fadd_rn(sum, __shfl_sync(0xFFFFFFFFu, v, r * hc + j));
        v = __fmul_rn(v, __fdiv_rn(1.0f, __fadd_rn(sum, hc_eps)));
        normalise_column();
    }
    if (lane < 16u) mix_table[k] = v;
}

// The seam partials of one row's column block bx (64 columns x the 4 copies) against the block's
// fn_w slice w: the 24 mix dots and the stream's sum of squares, then the row_buf columns (the
// in_pre mix) and their sum of squares, into partials[bx * 32 ..]. Ends on a barrier.
__device__ __forceinline__ void or_seam_partials(const float *in_pre,
                                                 const float (&w)[OR_S_ROWS], float x, float *__restrict__ hc_rows,
                                                 float *partials, float *row_buf, uint32_t bx,
                                                 uint32_t native_bf16, float *xs,
                                                 float (*red)[OR_S_PART_STRIDE]) {
    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31u;
    const uint32_t warp = t >> 5;
    const uint32_t index = (t / OR_S_COLS) * OR_S_HIDDEN + bx * OR_S_COLS + (t % OR_S_COLS);
    xs[t] = x;
    if (hc_rows != nullptr) hc_rows[index] = x;
#pragma unroll
    for (uint32_t r = 0; r < OR_S_ROWS; ++r) {
        const float s = or_warp_sum_rn(__fmul_rn(w[r], x));
        if (lane == 0u) red[warp][r] = s;
    }
    const float sq = or_warp_sum_rn(__fmul_rn(x, x));
    if (lane == 0u) red[warp][24] = sq;
    __syncthreads();
    if (t < OR_S_COLS) {
        float accumulated = 0.0f;
#pragma unroll
        for (uint32_t c = 0; c < OR_S_HC; ++c) {
            accumulated = __fadd_rn(__fmul_rn(in_pre[c], xs[c * OR_S_COLS + t]), accumulated);
        }
        if (native_bf16 != 0u) accumulated = or_bf16_round(accumulated);
        row_buf[bx * OR_S_COLS + t] = accumulated;
        const float rs = or_warp_sum_rn(__fmul_rn(accumulated, accumulated));
        if (lane == 0u) red[warp][25] = rs;
    }
    __syncthreads();
    if (t < OR_S_PART) {
        const uint32_t warps = t == 25u ? OR_S_COLS / 32u : OR_S_WARPS;
        float s = red[0][t];
        for (uint32_t k = 1; k < warps; ++k) s = __fadd_rn(s, red[k][t]);
        partials[bx * OR_S_PART_STRIDE + t] = s;
    }
    __syncthreads();
}

// The seam tail of one row once every column block's partials are in: the totals, the coefficient
// tail on the last warp (mix_table, pre_mix_out), and the norm (and the FP8 quantise) of row_buf on
// the others. `tail_warp_done` runs on the last warp after its tail (the ticket reset).
template <typename Done>
__device__ __forceinline__ void or_seam_tail(const float *partials, const float *row_buf, const float *__restrict__ hc_base,
                                             const float *__restrict__ hc_scale, const uint8_t *__restrict__ norm_w,
                                             float *mix_table, float *pre_mix_out, float *__restrict__ normed,
                                             uint8_t *__restrict__ codes, uint8_t *__restrict__ scales,
                                             __nv_bfloat16 *__restrict__ x_bf16, uint32_t norm_bf16, uint32_t native_bf16,
                                             uint32_t quant_narrow, uint32_t sinkhorn_iterations, float norm_eps,
                                             float hc_eps, float *tot, uint32_t blocks, Done tail_warp_done) {
    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31u;
    const uint32_t warp = t >> 5;
    for (uint32_t v = warp; v < OR_S_PART; v += OR_S_WARPS) {
        float s = 0.0f;
        for (uint32_t b = lane; b < blocks; b += 32u) s = __fadd_rn(s, __ldcg(partials + b * OR_S_PART_STRIDE + v));
        s = or_warp_sum_rn(s);
        if (lane == 0u) tot[v] = s;
    }
    __syncthreads();

    if (warp == OR_S_WARPS - 1u) {
#ifdef OR_SERIAL_TAIL
        if (lane == 0u)
            or_coefficient_tail(tot, tot[24], hc_scale, hc_base, norm_eps, hc_eps, sinkhorn_iterations,
                                mix_table, pre_mix_out);
#else
        or_coefficient_tail_warp(tot, tot[24], hc_scale, hc_base, norm_eps, hc_eps, sinkhorn_iterations,
                                 mix_table, pre_mix_out);
#endif
        tail_warp_done(lane);
        return;
    }
    const float mean = __fdiv_rn(tot[25], (float)OR_S_HIDDEN);
    const float inverse = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, norm_eps)));
    constexpr uint32_t NORM_WARPS = OR_S_WARPS - 1u;
    constexpr uint32_t PER_WARP = (OR_S_QBLOCKS + NORM_WARPS - 1u) / NORM_WARPS;
    float rows_v[PER_WARP], weights_v[PER_WARP];
#pragma unroll
    for (uint32_t i = 0; i < PER_WARP; ++i) {
        const uint32_t qb = warp + i * NORM_WARPS;
        if (qb < OR_S_QBLOCKS) {
            const uint32_t column = qb * 32u + lane;
            rows_v[i] = __ldcg(row_buf + column);
            weights_v[i] = norm_bf16 != 0u
                               ? __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(norm_w)[column])
                               : reinterpret_cast<const float *>(norm_w)[column];
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < PER_WARP; ++i) {
        const uint32_t qb = warp + i * NORM_WARPS;
        if (qb >= OR_S_QBLOCKS) break;
        const uint32_t column = qb * 32u + lane;
        float value = __fmul_rn(__fmul_rn(rows_v[i], inverse), weights_v[i]);
        if (native_bf16 != 0u) value = or_bf16_round(value);
        normed[column] = value;
        if (x_bf16 != nullptr) x_bf16[column] = __float2bfloat16_rn(value);
        if (codes != nullptr) {
            const float q = quant_narrow != 0u ? __bfloat162float(__float2bfloat16_rn(value)) : value;
            float amax = q < 0.0f ? -q : q;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other = __shfl_xor_sync(0xFFFFFFFFu, amax, offset);
                amax = other > amax ? other : amax;
            }
            const uint8_t scale = spark_act_quant_scale_code(amax);
            const float inv_scale = __fdiv_rn(1.0f, spark_ue8m0_decode(scale));
            codes[column] = spark_fp8_e4m3_encode(__fmul_rn(q, inv_scale));
            if (lane == 0u) scales[qb] = scale;
        }
    }
}

// Row y: stream + y*WIDTH, in_pre + y*in_pre_stride, mix_table + y*mix_stride, pre_mix_out + y*hc,
// hc_rows + y*WIDTH, normed + y*dim, codes + y*dim, scales + y*dim/32, x_bf16 + y*dim,
// partials + y*80*32, row_buf + y*dim, ticket + y.
extern "C" __global__ void __launch_bounds__(OR_S_THREADS) opt_rows_mhc_seam(
    const float *__restrict__ stream_all,
    const float *in_pre_all,
    const float *__restrict__ fn_w,
    const float *__restrict__ hc_base,
    const float *__restrict__ hc_scale,
    const uint8_t *__restrict__ norm_w,
    float *mix_table_all,
    float *pre_mix_out_all,
    float *__restrict__ hc_rows_all,
    float *__restrict__ normed_all,
    uint8_t *__restrict__ codes_all,
    uint8_t *__restrict__ scales_all,
    __nv_bfloat16 *__restrict__ x_bf16_all,
    float *partials_all,
    float *row_buf_all,
    uint32_t *ticket_all,
    const uint32_t norm_bf16,
    const uint32_t native_bf16,
    const uint32_t quant_narrow,
    const uint32_t sinkhorn_iterations,
    const float norm_eps,
    const float hc_eps,
    const uint32_t in_pre_stride,
    const uint32_t mix_stride) {
    OPT_PDL_LAUNCH();
    const uint32_t y = blockIdx.y;
    const float *stream = stream_all + (uint64_t)y * OR_S_WIDTH;
    const float *in_pre = in_pre_all + (uint64_t)y * in_pre_stride;
    float *mix_table = mix_table_all + (uint64_t)y * mix_stride;
    float *pre_mix_out = pre_mix_out_all != nullptr ? pre_mix_out_all + (uint64_t)y * OR_S_HC : nullptr;
    float *hc_rows = hc_rows_all != nullptr ? hc_rows_all + (uint64_t)y * OR_S_WIDTH : nullptr;
    float *normed = normed_all + (uint64_t)y * OR_S_HIDDEN;
    uint8_t *codes = codes_all != nullptr ? codes_all + (uint64_t)y * OR_S_HIDDEN : nullptr;
    uint8_t *scales = scales_all != nullptr ? scales_all + (uint64_t)y * OR_S_QBLOCKS : nullptr;
    __nv_bfloat16 *x_bf16 = x_bf16_all != nullptr ? x_bf16_all + (uint64_t)y * OR_S_HIDDEN : nullptr;
    float *partials = partials_all + (uint64_t)y * OR_S_BLOCKS * OR_S_PART_STRIDE;
    float *row_buf = row_buf_all + (uint64_t)y * OR_S_HIDDEN;
    uint32_t *ticket = ticket_all + y;

    __shared__ float xs[OR_S_THREADS];
    __shared__ float red[OR_S_WARPS][OR_S_PART_STRIDE];
    __shared__ float tot[OR_S_PART_STRIDE];
    __shared__ uint32_t s_last;
    const uint32_t t = threadIdx.x;

    {
        const uint32_t index = (t / OR_S_COLS) * OR_S_HIDDEN + blockIdx.x * OR_S_COLS + (t % OR_S_COLS);
        // fn_w is constant: it loads under the previous kernel's tail, and the stream after the wait.
        float w[OR_S_ROWS];
#pragma unroll
        for (uint32_t r = 0; r < OR_S_ROWS; ++r) w[r] = __ldg(fn_w + (uint64_t)r * OR_S_WIDTH + index);
        OPT_PDL_WAIT();
        const float x = stream[index];
        or_seam_partials(in_pre, w, x, hc_rows, partials, row_buf, blockIdx.x, native_bf16, xs, red);
    }

    __threadfence();
    __syncthreads();
    if (t == 0u) s_last = atomicAdd(ticket, 1u) == gridDim.x - 1u ? 1u : 0u;
    __syncthreads();
    if (s_last == 0u) return;
    __threadfence();

    or_seam_tail(partials, row_buf, hc_base, hc_scale, norm_w, mix_table, pre_mix_out, normed, codes, scales,
                 x_bf16, norm_bf16, native_bf16, quant_narrow, sinkhorn_iterations, norm_eps, hc_eps, tot, OR_S_BLOCKS,
                 [&](uint32_t lane) {
                     if (lane == 0u) *ticket = 0u;
                 });
}

// The bulk chunk's seam, in two launches (not bit-identical to opt_rows_mhc_seam: the partial
// sums run in another order). opt_bulk_mhc_partials: grid (40, ceil(rows / 32)), 128 threads;
// thread t owns column 128 blockIdx.x + t over the four copies, holds its 96 fn_w values, and per
// row of the block's 32 writes hc_rows, row_buf and the block's 26 partials (partials + y * 80 * 32
// + blockIdx.x * 32, the first 40 of the row's 80 slots). opt_bulk_mhc_tail: grid (rows), 256
// threads; or_seam_tail per row over those 40 partials.
#define OR_BULK_SEAM_ROWS 32u
#define OR_BULK_SEAM_THREADS 128u
#define OR_BULK_SEAM_BLOCKS (OR_S_HIDDEN / OR_BULK_SEAM_THREADS)
// The partials pass of the bulk seam over rows y0.. of the block. With POST, row y's stream column
// is first computed by opt_rows_mhc_post's arithmetic (no shared row) from residual (src_all), the
// row's mix table and delta, written to stream_out_all, and the partials read it from registers.
template <bool POST>
__device__ __forceinline__ void or_bulk_partials(
    const float *src_all, const float *mix_all, const __nv_bfloat16 *delta_all, float *stream_out_all,
    const uint32_t mix_stride, const float *in_pre_all, const float *__restrict__ fn_w,
    float *__restrict__ hc_rows_all, float *partials_all, float *row_buf_all, const uint32_t rows,
    const uint32_t native_bf16, const uint32_t in_pre_stride) {
    constexpr uint32_t WARPS = OR_BULK_SEAM_THREADS / 32u;
    __shared__ float red[2][WARPS][OR_S_PART_STRIDE];
    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31u;
    const uint32_t warp = t >> 5;
    const uint32_t column = blockIdx.x * OR_BULK_SEAM_THREADS + t;
    float w[OR_S_ROWS][OR_S_HC];
#pragma unroll
    for (uint32_t r = 0; r < OR_S_ROWS; ++r)
#pragma unroll
        for (uint32_t c = 0; c < OR_S_HC; ++c) w[r][c] = __ldg(fn_w + (uint64_t)r * OR_S_WIDTH + c * OR_S_HIDDEN + column);
    const uint32_t y0 = blockIdx.y * OR_BULK_SEAM_ROWS;
    const uint32_t y1 = y0 + OR_BULK_SEAM_ROWS < rows ? y0 + OR_BULK_SEAM_ROWS : rows;
    float x[OR_S_HC];
    float d = 0.0f;
    if (y0 < y1) {
#pragma unroll
        for (uint32_t c = 0; c < OR_S_HC; ++c) x[c] = src_all[(uint64_t)y0 * OR_S_WIDTH + c * OR_S_HIDDEN + column];
        if (POST) d = __bfloat162float(delta_all[(uint64_t)y0 * OR_S_HIDDEN + column]);
    }
    for (uint32_t y = y0; y < y1; ++y) {
        float next[OR_S_HC];
        float next_d = 0.0f;
#pragma unroll
        for (uint32_t c = 0; c < OR_S_HC; ++c)
            next[c] = y + 1u < y1 ? src_all[(uint64_t)(y + 1u) * OR_S_WIDTH + c * OR_S_HIDDEN + column] : 0.0f;
        if (POST && y + 1u < y1) next_d = __bfloat162float(delta_all[(uint64_t)(y + 1u) * OR_S_HIDDEN + column]);
        if (POST) {
            const float *mix_table = mix_all + (uint64_t)y * mix_stride;
            float value[OR_S_HC];
#pragma unroll
            for (uint32_t c = 0; c < OR_S_HC; ++c) {
                const float post = mix_table[OR_S_HC + c];
                if (native_bf16 != 0u) {
                    float comb_sum = 0.0f;
#pragma unroll
                    for (uint32_t s = 0; s < OR_S_HC; ++s)
                        comb_sum = __fadd_rn(comb_sum, __fmul_rn(mix_table[2u * OR_S_HC + s * OR_S_HC + c], x[s]));
                    value[c] = or_bf16_round(__fadd_rn(__fmul_rn(post, d), comb_sum));
                } else {
                    float accumulated = __fmul_rn(post, d);
#pragma unroll
                    for (uint32_t s = 0; s < OR_S_HC; ++s)
                        accumulated = __fadd_rn(accumulated, __fmul_rn(mix_table[2u * OR_S_HC + s * OR_S_HC + c], x[s]));
                    value[c] = accumulated;
                }
            }
#pragma unroll
            for (uint32_t c = 0; c < OR_S_HC; ++c) {
                x[c] = value[c];
                stream_out_all[(uint64_t)y * OR_S_WIDTH + c * OR_S_HIDDEN + column] = value[c];
            }
        }
        if (hc_rows_all != nullptr) {
#pragma unroll
            for (uint32_t c = 0; c < OR_S_HC; ++c) hc_rows_all[(uint64_t)y * OR_S_WIDTH + c * OR_S_HIDDEN + column] = x[c];
        }
        const float *in_pre = in_pre_all + (uint64_t)y * in_pre_stride;
        float part[OR_S_PART];
#pragma unroll
        for (uint32_t r = 0; r < OR_S_ROWS; ++r) {
            float dot = 0.0f;
#pragma unroll
            for (uint32_t c = 0; c < OR_S_HC; ++c) dot = fmaf(w[r][c], x[c], dot);
            part[r] = dot;
        }
        float sq = 0.0f, mixed = 0.0f;
#pragma unroll
        for (uint32_t c = 0; c < OR_S_HC; ++c) {
            sq = fmaf(x[c], x[c], sq);
            mixed = __fadd_rn(__fmul_rn(in_pre[c], x[c]), mixed);
        }
        if (native_bf16 != 0u) mixed = or_bf16_round(mixed);
        row_buf_all[(uint64_t)y * OR_S_HIDDEN + column] = mixed;
        part[24] = sq;
        part[25] = mixed * mixed;
#pragma unroll
        for (uint32_t v = 0; v < OR_S_PART; ++v) {
            float s = part[v];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
            part[v] = s;
        }
        const uint32_t buf = y & 1u;
        if (lane < OR_S_PART) {
            float mine = part[0];
#pragma unroll
            for (uint32_t v = 1; v < OR_S_PART; ++v) mine = lane == v ? part[v] : mine;
            red[buf][warp][lane] = mine;
        }
        __syncthreads();
        if (t < OR_S_PART) {
            float s = red[buf][0][t];
#pragma unroll
            for (uint32_t k = 1; k < WARPS; ++k) s += red[buf][k][t];
            partials_all[(uint64_t)y * OR_S_BLOCKS * OR_S_PART_STRIDE + blockIdx.x * OR_S_PART_STRIDE + t] = s;
        }
#pragma unroll
        for (uint32_t c = 0; c < OR_S_HC; ++c) x[c] = next[c];
        d = next_d;
    }
}

extern "C" __global__ void __launch_bounds__(OR_BULK_SEAM_THREADS) opt_bulk_mhc_partials(
    const float *__restrict__ stream_all, const float *in_pre_all, const float *__restrict__ fn_w,
    float *__restrict__ hc_rows_all, float *partials_all, float *row_buf_all, const uint32_t rows,
    const uint32_t native_bf16, const uint32_t in_pre_stride) {
    OPT_PDL_ENTRY();
    or_bulk_partials<false>(stream_all, nullptr, nullptr, nullptr, 0u, in_pre_all, fn_w, hc_rows_all, partials_all,
                            row_buf_all, rows, native_bf16, in_pre_stride);
}

// opt_bulk_mhc_post_partials: opt_rows_mhc_post into stream_out (which may be residual: a thread
// reads its column of a row before writing it) fused with the next seam's partials, so the stream
// is read once for both. Grid and block as opt_bulk_mhc_partials; no hc_rows copy.
extern "C" __global__ void __launch_bounds__(OR_BULK_SEAM_THREADS) opt_bulk_mhc_post_partials(
    const float *residual_all, const float *mix_all, const __nv_bfloat16 *__restrict__ delta_all,
    float *stream_out_all, const uint32_t mix_stride, const float *in_pre_all, const float *__restrict__ fn_w,
    float *partials_all, float *row_buf_all, const uint32_t rows, const uint32_t native_bf16,
    const uint32_t in_pre_stride) {
    OPT_PDL_ENTRY();
    or_bulk_partials<true>(residual_all, mix_all, delta_all, stream_out_all, mix_stride, in_pre_all, fn_w, nullptr,
                           partials_all, row_buf_all, rows, native_bf16, in_pre_stride);
}

extern "C" __global__ void __launch_bounds__(OR_S_THREADS) opt_bulk_mhc_tail(
    const float *__restrict__ hc_base, const float *__restrict__ hc_scale, const uint8_t *__restrict__ norm_w,
    float *mix_table_all, float *pre_mix_out_all, float *__restrict__ normed_all, uint8_t *__restrict__ codes_all,
    uint8_t *__restrict__ scales_all, __nv_bfloat16 *__restrict__ x_bf16_all, const float *partials_all,
    const float *row_buf_all, const uint32_t norm_bf16, const uint32_t native_bf16, const uint32_t quant_narrow,
    const uint32_t sinkhorn_iterations, const float norm_eps, const float hc_eps, const uint32_t mix_stride) {
    OPT_PDL_ENTRY();
    __shared__ float tot[OR_S_PART_STRIDE];
    const uint32_t y = blockIdx.x;
    or_seam_tail(partials_all + (uint64_t)y * OR_S_BLOCKS * OR_S_PART_STRIDE, row_buf_all + (uint64_t)y * OR_S_HIDDEN,
                 hc_base, hc_scale, norm_w, mix_table_all + (uint64_t)y * mix_stride,
                 pre_mix_out_all != nullptr ? pre_mix_out_all + (uint64_t)y * OR_S_HC : nullptr,
                 normed_all + (uint64_t)y * OR_S_HIDDEN, codes_all != nullptr ? codes_all + (uint64_t)y * OR_S_HIDDEN : nullptr,
                 scales_all != nullptr ? scales_all + (uint64_t)y * OR_S_QBLOCKS : nullptr,
                 x_bf16_all != nullptr ? x_bf16_all + (uint64_t)y * OR_S_HIDDEN : nullptr, norm_bf16, native_bf16,
                 quant_narrow, sinkhorn_iterations, norm_eps, hc_eps, tot, OR_BULK_SEAM_BLOCKS, [](uint32_t) {});
}

// Row y: residual + y*WIDTH, mix + y*mix_stride, delta + y*dim, shared + y*dim, out + y*WIDTH.
extern "C" __global__ void __launch_bounds__(128) opt_rows_mhc_post(
    const float *residual_all,
    const float *__restrict__ mix_all,
    const __nv_bfloat16 *__restrict__ delta_all,
    const float *__restrict__ shared_all,
    float *stream_out_all,
    const uint32_t native_bf16,
    const uint32_t mix_stride) {
    OPT_PDL_ENTRY();
    const uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= OR_S_HIDDEN) return;
    const uint32_t y = blockIdx.y;
    const float *residual = residual_all + (uint64_t)y * OR_S_WIDTH;
    const float *mix_table = mix_all + (uint64_t)y * mix_stride;
    const __nv_bfloat16 *delta = delta_all + (uint64_t)y * OR_S_HIDDEN;
    const float *shared = shared_all != nullptr ? shared_all + (uint64_t)y * OR_S_HIDDEN : nullptr;
    float *stream_out = stream_out_all + (uint64_t)y * OR_S_WIDTH;
    const uint32_t hc = OR_S_HC;
    float x = __bfloat162float(delta[column]);
    if (shared != nullptr) x = __fadd_rn(x, shared[column]);
    float r[OR_S_HC];
#pragma unroll
    for (uint32_t s = 0; s < hc; ++s) r[s] = residual[s * OR_S_HIDDEN + column];
#pragma unroll
    for (uint32_t c = 0; c < hc; ++c) {
        const float post = mix_table[hc + c];
        float value;
        if (native_bf16 != 0u) {
            float comb_sum = 0.0f;
#pragma unroll
            for (uint32_t s = 0; s < hc; ++s) comb_sum = __fadd_rn(comb_sum, __fmul_rn(mix_table[2u * hc + s * hc + c], r[s]));
            value = or_bf16_round(__fadd_rn(__fmul_rn(post, x), comb_sum));
        } else {
            float accumulated = __fmul_rn(post, x);
#pragma unroll
            for (uint32_t s = 0; s < hc; ++s) accumulated = __fadd_rn(accumulated, __fmul_rn(mix_table[2u * hc + s * hc + c], r[s]));
            value = accumulated;
        }
        stream_out[c * OR_S_HIDDEN + column] = value;
    }
}

// ---------------------------------------------------------------------------------------------
// kv path: norm + rope (opt_kv_norm_rope), window quantise (window_quant_fp8_bf16) into the ring
// slot of the row's position, and the bf16 cache append (kv_append_bf16). One block per row.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float or_bf16lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float or_bf16hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

extern "C" __global__ void __launch_bounds__(256) opt_rows_kv(
    const __nv_bfloat16 *__restrict__ latent_all,   // [rows][head_width]
    const __nv_bfloat16 *__restrict__ norm,
    const float eps,
    const float *__restrict__ cos_table,            // [max_positions][pairs]
    const float *__restrict__ sin_table,
    __nv_bfloat16 *__restrict__ rotated_all,         // [rows][head_width]
    uint8_t *__restrict__ ring_codes,                // the layer ring's codes base (slot 0)
    uint8_t *__restrict__ ring_scales,               // the layer ring's scales base (slot 0)
    __nv_bfloat16 *__restrict__ cache,               // the layer's bf16 cache [cache_rows][2*head_width]
    const uint32_t cache_rows,
    const uint32_t ring_rows,
    const uint32_t head_width,
    const uint32_t rope_head_dim,
    const uint32_t p0) {
    OPT_PDL_ENTRY();
    __shared__ float s_part[8];
    const uint32_t y = blockIdx.x;
    const uint32_t position = p0 + y;
    const uint32_t t = threadIdx.x;
    const uint32_t pairs = head_width >> 1u;
    const uint32_t rope_pairs = rope_head_dim >> 1u;
    const uint32_t *row2 = reinterpret_cast<const uint32_t *>(latent_all + (uint64_t)y * head_width);
    const float *cos = cos_table + (uint64_t)position * rope_pairs;
    const float *sin = sin_table + (uint64_t)position * rope_pairs;
    __nv_bfloat16 *out = rotated_all + (uint64_t)y * head_width;
    float sum = 0.0f;
    for (uint32_t p = t; p < pairs; p += 256u) {
        const uint32_t w = __ldg(row2 + p);
        const float a = or_bf16lo(w), b = or_bf16hi(w);
        sum = __fadd_rn(sum, __fmul_rn(a, a));
        sum = __fadd_rn(sum, __fmul_rn(b, b));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
    if ((t & 31u) == 0u) s_part[t >> 5] = sum;
    __syncthreads();
    float total = s_part[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) total = __fadd_rn(total, s_part[w]);
    const float mean = __fdiv_rn(total, (float)head_width);
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
    const uint32_t start_pair = (head_width - rope_head_dim) >> 1u;
    const uint32_t *norm2 = reinterpret_cast<const uint32_t *>(norm);
    uint32_t *out2 = reinterpret_cast<uint32_t *>(out);
    for (uint32_t p = t; p < pairs; p += 256u) {
        const uint32_t w = __ldg(row2 + p), n = __ldg(norm2 + p);
        float a = __fmul_rn(__fmul_rn(or_bf16lo(w), inv), or_bf16lo(n));
        float b = __fmul_rn(__fmul_rn(or_bf16hi(w), inv), or_bf16hi(n));
        if (p >= start_pair) {
            const uint32_t q = p - start_pair;
            const float c = __ldg(cos + q), s = __ldg(sin + q);
            const float first = __fadd_rn(__fmul_rn(a, c), -__fmul_rn(b, s));
            const float second = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
            a = first;
            b = second;
        }
        const __nv_bfloat162 o2 = __floats2bfloat162_rn(a, b);
        out2[p] = *reinterpret_cast<const uint32_t *>(&o2);
    }
    __syncthreads();
    // Window quantiser: one thread per 32-value block, the reference's serial amax and encode.
    const uint32_t blocks = head_width / 32u;
    const uint32_t ring_slot = position % ring_rows;
    uint8_t *codes = ring_codes + (uint64_t)ring_slot * head_width;
    uint8_t *scales = ring_scales + (uint64_t)ring_slot * blocks;
    if (t < blocks) {
        const uint32_t base = t * 32u;
        float amax = 0.0f;
        for (uint32_t i = 0; i < SPARK_M4_QUANT_BLOCK; ++i) {
            const float value = __bfloat162float(out[base + i]);
            const float magnitude = value < 0.0f ? -value : value;
            amax = magnitude > amax ? magnitude : amax;
        }
        const uint8_t scale_code = spark_act_quant_scale_code(amax);
        scales[t] = scale_code;
        const float scale = spark_ue8m0_decode(scale_code);
        for (uint32_t i = 0; i < SPARK_M4_QUANT_BLOCK; ++i) {
            const float value = __bfloat162float(out[base + i]);
            float scaled = __fdiv_rn(value, scale);
            scaled = scaled > SPARK_M4_ACT_E4M3_MAX ? SPARK_M4_ACT_E4M3_MAX : scaled;
            scaled = scaled < -SPARK_M4_ACT_E4M3_MAX ? -SPARK_M4_ACT_E4M3_MAX : scaled;
            codes[base + i] = spark_fp8_e4m3_encode(scaled);
        }
    }
    if (cache != nullptr) {
        const uint32_t slot = position % cache_rows;
        __nv_bfloat16 *const k_out = cache + (size_t)slot * 2u * head_width;
        __nv_bfloat16 *const v_out = k_out + head_width;
        for (uint32_t index = t; index < head_width; index += 256u) {
            k_out[index] = out[index];
            v_out[index] = out[index];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Attention body (opt_flash_mla_csa2_seq without the HSI prologue)
// ---------------------------------------------------------------------------------------------
#define OR_A_LATENT 512u
#define OR_A_THREADS 256u
#define OR_A_WARPS 8u
#define OR_A_MAX_SELECTED 512u
#define OR_A_MAX_ATTENDED 640u
#define OR_A_WINDOW 128u
#define OR_A_WSCALES 16u
#define OR_A_KSCALES 32u

__device__ __forceinline__ void or_e4m3x4(uint32_t w, float *f) {
    uint32_t h01, h23;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h01) : "h"((uint16_t)(w & 0xffffu)));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h23) : "h"((uint16_t)(w >> 16)));
    const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&h01));
    const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&h23));
    f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}

__device__ __forceinline__ float or_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Eight E2M1 values (the nibbles of `word`, low nibble first) as four f16 pairs, by the hardware
// conversion: every E2M1 value is exact in f16, so each is spark_fp4_e2m1_decode's float, -0
// included.
__device__ __forceinline__ void or_e2m1x8(uint32_t word, uint32_t halves[4]) {
    asm("{\n.reg .b8 b0, b1, b2, b3;\nmov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\ncvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\ncvt.rn.f16x2.e2m1x2 %3, b3;\n}"
        : "=r"(halves[0]), "=r"(halves[1]), "=r"(halves[2]), "=r"(halves[3])
        : "r"(word));
}

// A union slot's sixteen key/value channels [16 * lane, 16 * lane + 16), in two halves so a warp can
// have several slots' loads in flight: or_load16 issues a slot's loads (the window's 16 B of E4M3
// codes and its ue8m0 scale, or the store's 8 B of E2M1 and its E4M3 scale), and or_decode16 turns
// them into the reference's key values (window: bf16(e4m3 * ue8m0), store: e2m1 * e4m3 scale).
__device__ __forceinline__ void or_load16(uint32_t slot, uint32_t window_rows, uint32_t lane,
                                          const uint8_t *__restrict__ window_codes,
                                          const uint8_t *__restrict__ window_scales,
                                          const uint8_t *__restrict__ kv_fp4,
                                          const uint8_t *__restrict__ kv_scales,
                                          const uint32_t *s_candidates, uint4 &raw, uint32_t &scale) {
    if (slot < window_rows) {
        raw = __ldg(reinterpret_cast<const uint4 *>(window_codes + (uint64_t)slot * OR_A_LATENT) + lane);
        scale = window_scales[(uint64_t)slot * OR_A_WSCALES + (lane >> 1)];
    } else {
        const uint32_t candidate = s_candidates[slot - window_rows];
        const uint2 packed = __ldg(reinterpret_cast<const uint2 *>(kv_fp4 + (uint64_t)candidate * (OR_A_LATENT / 2u)) + lane);
        raw = make_uint4(packed.x, packed.y, 0u, 0u);
        scale = kv_scales[(uint64_t)candidate * OR_A_KSCALES + lane];
    }
}

__device__ __forceinline__ void or_decode16(bool window, const uint4 raw, uint32_t scale_byte, float k[16]) {
    if (window) {
        const float scale = spark_ue8m0_decode((uint8_t)scale_byte);
        or_e4m3x4(raw.x, k);
        or_e4m3x4(raw.y, k + 4);
        or_e4m3x4(raw.z, k + 8);
        or_e4m3x4(raw.w, k + 12);
#pragma unroll
        for (int i = 0; i < 16; ++i) k[i] = __bfloat162float(__float2bfloat16_rn(k[i] * scale));
    } else {
        const float scale = spark_fp8_e4m3_decode((uint8_t)scale_byte);
        uint32_t halves[8];
        or_e2m1x8(raw.x, halves);
        or_e2m1x8(raw.y, halves + 4);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float2 pair = __half22float2(*reinterpret_cast<const __half2 *>(&halves[i]));
            k[2 * i] = pair.x * scale;
            k[2 * i + 1] = pair.y * scale;
        }
    }
}

// Slots a warp loads at once: warp w still visits its slots w, w + 8, w + 16, ... in order, so
// every score, every exp and every value accumulation is the one-slot loop's, in its order.
#define OR_A_UNROLL 4u

// Row y is position p0 + y: window_rows = min(position + 1, 128) ending at the position in the
// layer's linear ring (codes at ring_codes + slot * 512, scales at ring_scales + slot * 16); the
// selected count is min(topk, (position + 1) / ratio) when ratio > 0, else 0; the row's selected
// list is selected_all + y * selected_stride.
extern "C" __global__ void __launch_bounds__(OR_A_THREADS) opt_rows_attn(
    const __nv_bfloat16 *__restrict__ q_all,          // [rows][heads][512]
    const uint8_t *__restrict__ kv_fp4,
    const uint8_t *__restrict__ kv_scales,
    const uint8_t *__restrict__ ring_codes,
    const uint8_t *__restrict__ ring_scales,
    const uint32_t *__restrict__ selected_all,
    const float *__restrict__ sink,
    __nv_bfloat16 *__restrict__ out_all,              // [rows][heads][512]
    const uint32_t p0,
    const uint32_t ratio,
    const uint32_t topk,
    const uint32_t selected_stride,
    const uint32_t ring_rows) {
    OPT_PDL_ENTRY();
    const uint32_t head = blockIdx.x;
    const uint32_t heads = gridDim.x;
    const uint32_t y = blockIdx.y;
    const uint32_t position = p0 + y;
    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31u;
    const uint32_t warp = t >> 5;
    const uint32_t window_rows = position + 1u < OR_A_WINDOW ? position + 1u : OR_A_WINDOW;
    const uint32_t window_start = position + 1u - window_rows;
    uint32_t selected_count = 0u;
    if (ratio > 0u) {
        const uint32_t n_pos = (position + 1u) / ratio;
        selected_count = n_pos < topk ? n_pos : topk;
    }
    const uint32_t window_slot = window_start % ring_rows;
    const uint8_t *window_codes = ring_codes + (uint64_t)window_slot * OR_A_LATENT;
    const uint8_t *window_scales = ring_scales + (uint64_t)window_slot * OR_A_WSCALES;
    const uint32_t *selected = selected_all + (uint64_t)y * selected_stride;
    const __nv_bfloat16 *q = q_all + (uint64_t)y * heads * OR_A_LATENT;
    __nv_bfloat16 *out = out_all + (uint64_t)y * heads * OR_A_LATENT;
    if (selected_count > OR_A_MAX_SELECTED || window_rows + selected_count > OR_A_MAX_ATTENDED) {
        for (uint32_t i = t; i < OR_A_LATENT; i += OR_A_THREADS)
            out[(uint64_t)head * OR_A_LATENT + i] = __float2bfloat16_rn(__uint_as_float(0x7FC00000u));
        return;
    }
    __shared__ float s_scores[OR_A_MAX_ATTENDED];
    __shared__ __align__(16) float s_q[OR_A_LATENT];
    __shared__ uint32_t s_candidates[OR_A_MAX_SELECTED];
    __shared__ __align__(16) float s_red[OR_A_WARPS][OR_A_LATENT];
    __shared__ float s_wmax[OR_A_WARPS];
    __shared__ float s_wsum[OR_A_WARPS];

    const uint32_t union_total = window_rows + selected_count;
    const __nv_bfloat16 *q_head = q + (uint64_t)head * OR_A_LATENT;
    s_q[t] = __bfloat162float(q_head[t]);
    s_q[t + OR_A_THREADS] = __bfloat162float(q_head[t + OR_A_THREADS]);
    for (uint32_t i = t; i < selected_count; i += OR_A_THREADS) s_candidates[i] = selected[i];
    __syncthreads();

    float qv[16];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 v = reinterpret_cast<const float4 *>(s_q)[4u * lane + j];
        qv[4 * j] = v.x; qv[4 * j + 1] = v.y; qv[4 * j + 2] = v.z; qv[4 * j + 3] = v.w;
    }

    // A union slot is live unless it is a selected store entry past the row's position.
    const auto live = [&](uint32_t slot) {
        return slot < union_total && !(slot >= window_rows && s_candidates[slot - window_rows] > position);
    };
    float wmax = -3.402823466e38f;
    for (uint32_t base = warp; base < union_total; base += OR_A_WARPS * OR_A_UNROLL) {
        uint4 raw[OR_A_UNROLL];
        uint32_t scale[OR_A_UNROLL];
#pragma unroll
        for (uint32_t u = 0; u < OR_A_UNROLL; ++u) {
            const uint32_t slot = base + u * OR_A_WARPS;
            if (live(slot))
                or_load16(slot, window_rows, lane, window_codes, window_scales, kv_fp4, kv_scales, s_candidates,
                          raw[u], scale[u]);
        }
#pragma unroll
        for (uint32_t u = 0; u < OR_A_UNROLL; ++u) {
            const uint32_t slot = base + u * OR_A_WARPS;
            if (slot >= union_total) break;
            float score;
            if (!live(slot)) {
                score = -3.402823466e38f;
            } else {
                float k[16];
                or_decode16(slot < window_rows, raw[u], scale[u], k);
                float partial = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) partial = fmaf(qv[i], k[i], partial);
                score = or_warp_sum(partial) * SPARK_M4_MLA_SCORE_SCALE;
            }
            if (lane == 0u) s_scores[slot] = score;
            wmax = score > wmax ? score : wmax;
        }
    }
    if (lane == 0u) s_wmax[warp] = wmax;
    __syncthreads();
    float smax = s_wmax[0];
#pragma unroll
    for (uint32_t w = 1; w < OR_A_WARPS; ++w) smax = s_wmax[w] > smax ? s_wmax[w] : smax;

    float lsum = 0.0f;
    for (uint32_t slot = t; slot < union_total; slot += OR_A_THREADS) {
        const float probability = spark_exp(s_scores[slot] - smax);
        s_scores[slot] = __bfloat162float(__float2bfloat16_rn(probability));
        lsum += probability;
    }
    lsum = or_warp_sum(lsum);
    if (lane == 0u) s_wsum[warp] = lsum;
    __syncthreads();
    float ssum = spark_exp(sink[head] - smax);
#pragma unroll
    for (uint32_t w = 0; w < OR_A_WARPS; ++w) ssum += s_wsum[w];
    ssum = ssum > 0.0f ? ssum : 1.0f;

    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    for (uint32_t base = warp; base < union_total; base += OR_A_WARPS * OR_A_UNROLL) {
        uint4 raw[OR_A_UNROLL];
        uint32_t scale[OR_A_UNROLL];
#pragma unroll
        for (uint32_t u = 0; u < OR_A_UNROLL; ++u) {
            const uint32_t slot = base + u * OR_A_WARPS;
            if (live(slot))
                or_load16(slot, window_rows, lane, window_codes, window_scales, kv_fp4, kv_scales, s_candidates,
                          raw[u], scale[u]);
        }
#pragma unroll
        for (uint32_t u = 0; u < OR_A_UNROLL; ++u) {
            const uint32_t slot = base + u * OR_A_WARPS;
            if (!live(slot)) continue;
            const float prob = s_scores[slot];
            float v[16];
            or_decode16(slot < window_rows, raw[u], scale[u], v);
#pragma unroll
            for (int i = 0; i < 16; ++i) acc[i] = fmaf(prob, v[i], acc[i]);
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        reinterpret_cast<float4 *>(s_red[warp])[4u * lane + j] =
            make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]);
    }
    __syncthreads();
#pragma unroll
    for (uint32_t r = 0; r < 2; ++r) {
        const uint32_t i = t + r * OR_A_THREADS;
        float total = s_red[0][i];
#pragma unroll
        for (uint32_t w = 1; w < OR_A_WARPS; ++w) total += s_red[w][i];
        out[(uint64_t)head * OR_A_LATENT + i] = __float2bfloat16_rn(total / ssum);
    }
}

// opt_rows_attn in three kernels for 16 heads a rank, bit-identical to it. opt_rows_attn's block
// (head, row) loads and decodes every slot once per head and pass; here each slot is decoded once
// per four heads (scores) or per 64 channels (values):
//  - opt_rows_attn_scores: every score is opt_rows_attn's 16-channel fma chains and xor butterfly.
//    A warp holds four heads' q and reduces four slots' sixteen partials together: at each
//    butterfly level a lane keeps half its values and trades the other half with its partner, so
//    every add is the butterfly's (own + partner's) and the 16 scores take 16 shuffles, not 80.
//  - opt_rows_attn_softmax: opt_rows_attn's max, bf16 probabilities and sums, once per (row, head).
//  - opt_rows_attn_values: opt_rows_attn's eight warps' value passes over 64 channels a block, the
//    probabilities read from shared memory.
#define OR_A3_HEADS 16u
#define OR_A3_CHUNK 32u
#define OR_A3_UV 8u  // value slots a warp loads at once

// e4m3 * 2^(e - 127) is exact in f32 and so already bf16 when the product is normal: 2^-9 (the
// least e4m3) * 2^(e - 127) >= 2^-126 and 448 * 2^(e - 127) < 2^128. Zero stays zero, and an
// e4m3 NaN gives a NaN either way, which the fma chains take the same.
__device__ __forceinline__ bool or_a3_window_exact(uint32_t e) { return e >= 10u && e <= 246u; }

// Window codes (16 e4m3) times their ue8m0 scale, as or_decode16's window branch.
__device__ __forceinline__ void or_a3_window16(const uint4 raw, uint32_t scale_byte, float k[16]) {
    const float scale = spark_ue8m0_decode((uint8_t)scale_byte);
    or_e4m3x4(raw.x, k);
    or_e4m3x4(raw.y, k + 4);
    or_e4m3x4(raw.z, k + 8);
    or_e4m3x4(raw.w, k + 12);
    if (or_a3_window_exact(scale_byte)) {
#pragma unroll
        for (int i = 0; i < 16; ++i) k[i] *= scale;
    } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) k[i] = __bfloat162float(__float2bfloat16_rn(k[i] * scale));
    }
}

// spark_fp8_e4m3_decode by the hardware conversion: every e4m3 value is exact in f16, -0 included;
// the NaN codes give a NaN, which every use (a product) turns into the same canonical NaN.
__device__ __forceinline__ float or_a3_e4m3(uint32_t byte) {
    uint32_t h01;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h01) : "h"((uint16_t)(byte & 0xffu)));
    return __half2float(__ushort_as_half((unsigned short)(h01 & 0xffffu)));
}

// Sixteen E2M1 values times their e4m3 scale, as or_decode16's store branch.
__device__ __forceinline__ void or_a3_fp4x16(const uint4 raw, uint32_t scale_byte, float k[16]) {
    const float scale = or_a3_e4m3(scale_byte);
    uint32_t halves[8];
    or_e2m1x8(raw.x, halves);
    or_e2m1x8(raw.y, halves + 4);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float2 pair = __half22float2(*reinterpret_cast<const __half2 *>(&halves[i]));
        k[2 * i] = pair.x * scale;
        k[2 * i + 1] = pair.y * scale;
    }
}

// The transpose reduction of 16 partials: on return lane L holds the butterfly sum of partial
// (L >> 1) & 15 (both lanes of a pair hold it).
__device__ __forceinline__ float or_a3_reduce16(float v[16], uint32_t lane) {
#pragma unroll
    for (uint32_t o = 16u, n = 8u; n >= 1u; o >>= 1, n >>= 1) {
        const bool up = (lane & o) != 0u;
#pragma unroll
        for (uint32_t i = 0; i < n; ++i) {
            const float send = up ? v[i] : v[n + i];
            const float keep = up ? v[n + i] : v[i];
            v[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
    }
    return v[0] + __shfl_xor_sync(0xffffffffu, v[0], 1);
}

// Scores: block (chunk, row), 8 warps; warp w scores heads 4 (w & 3) .. + 3 over the chunk's slots
// first + 16 (w >> 2) .. + 15, four at a time, into scores[row][head][640] (-FLT_MAX where the
// slot is not live).
extern "C" __global__ void __launch_bounds__(256, 2) opt_rows_attn_scores(
    const __nv_bfloat16 *__restrict__ q_all, const uint8_t *__restrict__ kv_fp4,
    const uint8_t *__restrict__ kv_scales, const uint8_t *__restrict__ ring_codes,
    const uint8_t *__restrict__ ring_scales, const uint32_t *__restrict__ selected_all,
    float *__restrict__ scores_all, const uint32_t p0, const uint32_t ratio, const uint32_t topk,
    const uint32_t selected_stride, const uint32_t ring_rows) {
    OPT_PDL_ENTRY();
    const uint32_t y = blockIdx.y;
    const uint32_t position = p0 + y;
    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31u;
    const uint32_t warp = t >> 5;
    const uint32_t window_rows = position + 1u < OR_A_WINDOW ? position + 1u : OR_A_WINDOW;
    const uint32_t window_start = position + 1u - window_rows;
    uint32_t selected_count = 0u;
    if (ratio > 0u) {
        const uint32_t n_pos = (position + 1u) / ratio;
        selected_count = n_pos < topk ? n_pos : topk;
    }
    if (selected_count > OR_A_MAX_SELECTED || window_rows + selected_count > OR_A_MAX_ATTENDED) return;
    const uint32_t union_total = window_rows + selected_count;
    const uint32_t first = blockIdx.x * OR_A3_CHUNK;
    if (first >= union_total) return;
    const uint32_t window_slot = window_start % ring_rows;
    const uint8_t *window_codes = ring_codes + (uint64_t)window_slot * OR_A_LATENT;
    const uint8_t *window_scales = ring_scales + (uint64_t)window_slot * OR_A_WSCALES;
    const uint32_t *selected = selected_all + (uint64_t)y * selected_stride;

    __shared__ uint32_t s_cand[OR_A3_CHUNK];
    if (t < OR_A3_CHUNK) {
        const uint32_t slot = first + t;
        s_cand[t] = slot >= window_rows && slot < union_total ? selected[slot - window_rows] : 0u;
    }
    const uint32_t h0 = (warp & 3u) * 4u;
    float qv[4][16];
#pragma unroll
    for (uint32_t h = 0; h < 4; ++h) {
        const uint4 *src = reinterpret_cast<const uint4 *>(q_all + ((uint64_t)y * OR_A3_HEADS + h0 + h) * OR_A_LATENT) + 2u * lane;
        const uint4 a = __ldg(src), b = __ldg(src + 1);
        const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qv[h][2 * i] = __uint_as_float(w[i] << 16);
            qv[h][2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
        }
    }
    __syncthreads();
    float *scores = scores_all + (uint64_t)y * OR_A3_HEADS * OR_A_MAX_ATTENDED;
    const uint32_t base = first + (warp >> 2) * 16u;
#pragma unroll 1
    for (uint32_t it = 0; it < 4u; ++it) {
        const uint32_t s0 = base + it * 4u;
        if (s0 >= union_total) break;
        uint4 raw[4];
        uint32_t scale[4];
        bool live[4];
#pragma unroll
        for (uint32_t u = 0; u < 4; ++u) {
            const uint32_t slot = s0 + u;
            const uint32_t candidate = s_cand[slot - first];
            live[u] = slot < union_total && !(slot >= window_rows && candidate > position);
            raw[u] = make_uint4(0u, 0u, 0u, 0u);
            scale[u] = 127u;
            if (live[u]) {
                if (slot < window_rows) {
                    raw[u] = __ldg(reinterpret_cast<const uint4 *>(window_codes + (uint64_t)slot * OR_A_LATENT) + lane);
                    scale[u] = window_scales[(uint64_t)slot * OR_A_WSCALES + (lane >> 1)];
                } else {
                    const uint2 packed = __ldg(reinterpret_cast<const uint2 *>(kv_fp4 + (uint64_t)candidate * (OR_A_LATENT / 2u)) + lane);
                    raw[u] = make_uint4(packed.x, packed.y, 0u, 0u);
                    scale[u] = kv_scales[(uint64_t)candidate * OR_A_KSCALES + lane];
                }
            }
        }
        float v[16];
#pragma unroll
        for (uint32_t u = 0; u < 4; ++u) {
            float k[16];
            if (s0 + u < window_rows) or_a3_window16(raw[u], scale[u], k);
            else or_a3_fp4x16(raw[u], scale[u], k);
#pragma unroll
            for (uint32_t h = 0; h < 4; ++h) {
                float partial = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) partial = fmaf(qv[h][i], k[i], partial);
                v[u * 4u + h] = partial;
            }
        }
        const float sum = or_a3_reduce16(v, lane);
        const uint32_t j = (lane >> 1) & 15u, u = j >> 2, h = j & 3u;
        const uint32_t slot = s0 + u;
        if ((lane & 1u) == 0u && slot < union_total)
            scores[(h0 + h) * OR_A_MAX_ATTENDED + slot] = live[u] ? sum * SPARK_M4_MLA_SCORE_SCALE : -3.402823466e38f;
    }
}

// Softmax: block (head, row). opt_rows_attn's max over the slots, probabilities p = exp(score -
// max) (thread t over slots t, t + 256, ...), stored bf16-rounded as floats at probs[row][slot][head],
// and its sum: exp(sink - max) plus the warps' butterfly sums of the unrounded p, in warp order.
extern "C" __global__ void __launch_bounds__(256) opt_rows_attn_softmax(
    const float *__restrict__ scores_all, const float *__restrict__ sink, float *__restrict__ probs_all,
    float *__restrict__ ssum_all, const uint32_t p0, const uint32_t ratio, const uint32_t topk) {
    OPT_PDL_ENTRY();
    constexpr uint32_t SPT = (OR_A_MAX_ATTENDED + 255u) / 256u;
    const uint32_t h = blockIdx.x, y = blockIdx.y;
    const uint32_t position = p0 + y;
    const uint32_t t = threadIdx.x, lane = t & 31u, warp = t >> 5;
    const uint32_t window_rows = position + 1u < OR_A_WINDOW ? position + 1u : OR_A_WINDOW;
    uint32_t selected_count = 0u;
    if (ratio > 0u) {
        const uint32_t n_pos = (position + 1u) / ratio;
        selected_count = n_pos < topk ? n_pos : topk;
    }
    if (selected_count > OR_A_MAX_SELECTED || window_rows + selected_count > OR_A_MAX_ATTENDED) return;
    const uint32_t union_total = window_rows + selected_count;
    const float *scores = scores_all + ((uint64_t)y * OR_A3_HEADS + h) * OR_A_MAX_ATTENDED;
    __shared__ float s_max[OR_A_WARPS], s_sum[OR_A_WARPS];
    float sc[SPT];
    float m = -3.402823466e38f;
#pragma unroll
    for (uint32_t k = 0; k < SPT; ++k) {
        const uint32_t slot = t + 256u * k;
        sc[k] = slot < union_total ? __ldcg(scores + slot) : -3.402823466e38f;
        m = sc[k] > m ? sc[k] : m;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, m, o);
        m = other > m ? other : m;
    }
    if (lane == 0u) s_max[warp] = m;
    __syncthreads();
    float smax = s_max[0];
#pragma unroll
    for (uint32_t w = 1; w < OR_A_WARPS; ++w) smax = s_max[w] > smax ? s_max[w] : smax;
    float *probs = probs_all + (uint64_t)y * OR_A_MAX_ATTENDED * OR_A3_HEADS + h;
    float lsum = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < SPT; ++k) {
        const uint32_t slot = t + 256u * k;
        if (slot < union_total) {
            const float probability = spark_exp(sc[k] - smax);
            probs[(uint64_t)slot * OR_A3_HEADS] = __bfloat162float(__float2bfloat16_rn(probability));
            lsum += probability;
        }
    }
    lsum = or_warp_sum(lsum);
    if (lane == 0u) s_sum[warp] = lsum;
    __syncthreads();
    if (t == 0u) {
        float ssum = spark_exp(sink[h] - smax);
#pragma unroll
        for (uint32_t w = 0; w < OR_A_WARPS; ++w) ssum += s_sum[w];
        ssum_all[y * OR_A3_HEADS + h] = ssum > 0.0f ? ssum : 1.0f;
    }
}

// One slot's value accumulation for a lane's two channels c0, c0 + 1 over all sixteen heads.
template <bool WINDOW>
__device__ __forceinline__ void or_a3_value_step(uint32_t raw, uint32_t scale, const float *prob, float acc[OR_A3_HEADS][2]) {
    float v0, v1;
    if constexpr (WINDOW) {
        const float s = spark_ue8m0_decode((uint8_t)scale);
        uint32_t h01;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h01) : "h"((uint16_t)raw));
        const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&h01));
        if (or_a3_window_exact(scale)) {
            v0 = a.x * s;
            v1 = a.y * s;
        } else {
            v0 = __bfloat162float(__float2bfloat16_rn(a.x * s));
            v1 = __bfloat162float(__float2bfloat16_rn(a.y * s));
        }
    } else {
        const float s = or_a3_e4m3(scale);
        uint32_t h01;
        asm("{\n.reg .b8 b0, b1, b2, b3;\nmov.b32 {b0, b1, b2, b3}, %1;\ncvt.rn.f16x2.e2m1x2 %0, b0;\n}" : "=r"(h01) : "r"(raw));
        const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&h01));
        v0 = a.x * s;
        v1 = a.y * s;
    }
    const float4 *p4 = reinterpret_cast<const float4 *>(prob);
#pragma unroll
    for (uint32_t q = 0; q < 4; ++q) {
        const float4 p = p4[q];
        const float pv[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            acc[4 * q + i][0] = fmaf(pv[i], v0, acc[4 * q + i][0]);
            acc[4 * q + i][1] = fmaf(pv[i], v1, acc[4 * q + i][1]);
        }
    }
}

// Values: block (split, row), 8 warps = opt_rows_attn's 8 warps; a lane holds channels
// 64 split + 2 lane, + 1 for all sixteen heads. Warp w accumulates its slots w, w + 8, ... in
// order; the window slots (all live) come first, then the selected ones.
extern "C" __global__ void __launch_bounds__(256) opt_rows_attn_values(
    const uint8_t *__restrict__ kv_fp4, const uint8_t *__restrict__ kv_scales,
    const uint8_t *__restrict__ ring_codes, const uint8_t *__restrict__ ring_scales,
    const uint32_t *__restrict__ selected_all, const float *__restrict__ probs_all,
    const float *__restrict__ ssum_all, __nv_bfloat16 *__restrict__ out_all, const uint32_t p0,
    const uint32_t ratio, const uint32_t topk, const uint32_t selected_stride, const uint32_t ring_rows) {
    OPT_PDL_ENTRY();
    constexpr uint32_t CPB = 64u;
    const uint32_t y = blockIdx.y;
    const uint32_t position = p0 + y;
    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31u;
    const uint32_t warp = t >> 5;
    const uint32_t window_rows = position + 1u < OR_A_WINDOW ? position + 1u : OR_A_WINDOW;
    const uint32_t window_start = position + 1u - window_rows;
    uint32_t selected_count = 0u;
    if (ratio > 0u) {
        const uint32_t n_pos = (position + 1u) / ratio;
        selected_count = n_pos < topk ? n_pos : topk;
    }
    __nv_bfloat16 *out = out_all + (uint64_t)y * OR_A3_HEADS * OR_A_LATENT;
    const uint32_t cb = blockIdx.x * CPB;
    if (selected_count > OR_A_MAX_SELECTED || window_rows + selected_count > OR_A_MAX_ATTENDED) {
        for (uint32_t i = t; i < OR_A3_HEADS * CPB; i += 256u)
            out[(uint64_t)(i / CPB) * OR_A_LATENT + cb + i % CPB] = __float2bfloat16_rn(__uint_as_float(0x7FC00000u));
        return;
    }
    const uint32_t union_total = window_rows + selected_count;
    const uint32_t window_slot = window_start % ring_rows;
    const uint8_t *window_codes = ring_codes + (uint64_t)window_slot * OR_A_LATENT;
    const uint8_t *window_scales = ring_scales + (uint64_t)window_slot * OR_A_WSCALES;
    const uint32_t *selected = selected_all + (uint64_t)y * selected_stride;

    // The probabilities, then (after the value pass) the warp partials.
    __shared__ __align__(16) float s_buf[OR_A_MAX_ATTENDED * OR_A3_HEADS];
    __shared__ uint32_t s_candidates[OR_A_MAX_SELECTED];
    {
        const float4 *src = reinterpret_cast<const float4 *>(probs_all + (uint64_t)y * OR_A_MAX_ATTENDED * OR_A3_HEADS);
        float4 *dst = reinterpret_cast<float4 *>(s_buf);
        for (uint32_t i = t; i < union_total * (OR_A3_HEADS / 4u); i += 256u) dst[i] = __ldcg(src + i);
        for (uint32_t i = t; i < selected_count; i += 256u) s_candidates[i] = selected[i];
    }
    __syncthreads();

    const uint32_t c0 = cb + lane * 2u;
    float acc[OR_A3_HEADS][2];
#pragma unroll
    for (uint32_t h = 0; h < OR_A3_HEADS; ++h) acc[h][0] = acc[h][1] = 0.0f;

    // Window slots warp, warp + 8, ... < window_rows.
    uint32_t slot = warp;
    for (; slot + (OR_A3_UV - 1u) * OR_A_WARPS < window_rows; slot += OR_A3_UV * OR_A_WARPS) {
        uint32_t raw[OR_A3_UV], scale[OR_A3_UV];
#pragma unroll
        for (uint32_t u = 0; u < OR_A3_UV; ++u) {
            const uint32_t s = slot + u * OR_A_WARPS;
            raw[u] = __ldg(reinterpret_cast<const uint16_t *>(window_codes + (uint64_t)s * OR_A_LATENT + c0));
            scale[u] = window_scales[(uint64_t)s * OR_A_WSCALES + c0 / 32u];
        }
#pragma unroll
        for (uint32_t u = 0; u < OR_A3_UV; ++u) or_a3_value_step<true>(raw[u], scale[u], s_buf + (slot + u * OR_A_WARPS) * OR_A3_HEADS, acc);
    }
    for (; slot < window_rows; slot += OR_A_WARPS) {
        const uint32_t raw = __ldg(reinterpret_cast<const uint16_t *>(window_codes + (uint64_t)slot * OR_A_LATENT + c0));
        or_a3_value_step<true>(raw, window_scales[(uint64_t)slot * OR_A_WSCALES + c0 / 32u], s_buf + slot * OR_A3_HEADS, acc);
    }
    // Selected slots: a non-live one (candidate past the position) is skipped, as in opt_rows_attn.
    bool all_live = true;
    for (uint32_t s = slot + lane * OR_A_WARPS; s < union_total + 31u * OR_A_WARPS; s += 32u * OR_A_WARPS)
        if (s < union_total && s_candidates[s - window_rows] > position) all_live = false;
    all_live = __all_sync(0xffffffffu, all_live);
    if (all_live && slot + (OR_A3_UV - 1u) * OR_A_WARPS < union_total) {
        // Software-pipelined: the next group's loads are in flight while this group accumulates.
        uint32_t raw[OR_A3_UV], scale[OR_A3_UV], nraw[OR_A3_UV], nscale[OR_A3_UV];
        const auto load = [&](uint32_t s0, uint32_t (&r)[OR_A3_UV], uint32_t (&sc)[OR_A3_UV]) {
#pragma unroll
            for (uint32_t u = 0; u < OR_A3_UV; ++u) {
                const uint32_t candidate = s_candidates[s0 + u * OR_A_WARPS - window_rows];
                r[u] = __ldg(kv_fp4 + (uint64_t)candidate * (OR_A_LATENT / 2u) + c0 / 2u);
                sc[u] = kv_scales[(uint64_t)candidate * OR_A_KSCALES + c0 / 16u];
            }
        };
        load(slot, nraw, nscale);
        for (; slot + (OR_A3_UV - 1u) * OR_A_WARPS < union_total; slot += OR_A3_UV * OR_A_WARPS) {
#pragma unroll
            for (uint32_t u = 0; u < OR_A3_UV; ++u) { raw[u] = nraw[u]; scale[u] = nscale[u]; }
            const uint32_t next = slot + OR_A3_UV * OR_A_WARPS;
            if (next + (OR_A3_UV - 1u) * OR_A_WARPS < union_total) load(next, nraw, nscale);
#pragma unroll
            for (uint32_t u = 0; u < OR_A3_UV; ++u) or_a3_value_step<false>(raw[u], scale[u], s_buf + (slot + u * OR_A_WARPS) * OR_A3_HEADS, acc);
        }
    }
    for (; slot < union_total; slot += OR_A_WARPS) {
        const uint32_t candidate = s_candidates[slot - window_rows];
        if (candidate > position) continue;
        const uint32_t raw = __ldg(kv_fp4 + (uint64_t)candidate * (OR_A_LATENT / 2u) + c0 / 2u);
        or_a3_value_step<false>(raw, kv_scales[(uint64_t)candidate * OR_A_KSCALES + c0 / 16u], s_buf + slot * OR_A3_HEADS, acc);
    }

    // The warp partials, summed in warp order per channel, over the sum.
    __syncthreads();
    float (*s_red)[OR_A3_HEADS][CPB] = reinterpret_cast<float (*)[OR_A3_HEADS][CPB]>(s_buf);
#pragma unroll
    for (uint32_t h = 0; h < OR_A3_HEADS; ++h) {
        s_red[warp][h][lane * 2u] = acc[h][0];
        s_red[warp][h][lane * 2u + 1u] = acc[h][1];
    }
    __syncthreads();
    const float *ssum = ssum_all + y * OR_A3_HEADS;
    for (uint32_t i = t; i < OR_A3_HEADS * CPB; i += 256u) {
        const uint32_t h = i / CPB, c = i % CPB;
        float total = s_red[0][h][c];
#pragma unroll
        for (uint32_t w = 1; w < OR_A_WARPS; ++w) total += s_red[w][h][c];
        out[(uint64_t)h * OR_A_LATENT + cb + c] = __float2bfloat16_rn(total / __ldcg(ssum + h));
    }
}

// rope_rotate_rows_bf16 with sin_sign = -1 for head row (y, head) at position p0 + y, in place.
extern "C" __global__ void opt_rows_inv_rope(
    __nv_bfloat16 *__restrict__ rows_all,      // [rows][heads][head_width]
    const float *__restrict__ cos_table,       // [max_positions][pairs]
    const float *__restrict__ sin_table,
    const uint32_t head_width,
    const uint32_t rope_head_dim,
    const uint32_t p0) {
    OPT_PDL_ENTRY();
    const uint32_t head = blockIdx.x;
    const uint32_t heads = gridDim.x;
    const uint32_t y = blockIdx.y;
    const uint32_t pairs = rope_head_dim / 2u;
    const uint32_t tail = head_width - rope_head_dim;
    __nv_bfloat16 *row = rows_all + ((uint64_t)y * heads + head) * head_width;
    const float *cos = cos_table + (uint64_t)(p0 + y) * pairs;
    const float *sin = sin_table + (uint64_t)(p0 + y) * pairs;
    const float sign = -1.0f;
    for (uint32_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
        const float a = __bfloat162float(row[tail + 2u * pair]);
        const float b = __bfloat162float(row[tail + 2u * pair + 1u]);
        const float first = __fadd_rn(__fmul_rn(a, cos[pair]),
                                       -__fmul_rn(sign, __fmul_rn(b, sin[pair])));
        const float second = __fadd_rn(__fmul_rn(sign, __fmul_rn(a, sin[pair])),
                                        __fmul_rn(b, cos[pair]));
        row[tail + 2u * pair] = __float2bfloat16_rn(first);
        row[tail + 2u * pair + 1u] = __float2bfloat16_rn(second);
    }
}

// ---------------------------------------------------------------------------------------------
// LM head per row (opt_lm_head arithmetic): block (chunk, y) scores 1024 entries of row y.
// ---------------------------------------------------------------------------------------------
#define OR_LM_HIDDEN 5120u
#define OR_LM_THREADS 512u
#define OR_LM_WARPS (OR_LM_THREADS / 32u)
#define OR_LM_CHUNK 1024u

__device__ __forceinline__ uint4 or_ldw(const void *p) {
    uint4 v;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(p));
    return v;
}

__device__ __forceinline__ float or_dot8(uint4 w, uint4 h, float acc) {
    const uint32_t a[4] = {w.x, w.y, w.z, w.w};
    const uint32_t b[4] = {h.x, h.y, h.z, h.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        acc = fmaf(or_bf16lo(a[k]), or_bf16lo(b[k]), acc);
        acc = fmaf(or_bf16hi(a[k]), or_bf16hi(b[k]), acc);
    }
    return acc;
}

// Up to OR_LM_MAXR rows share one pass over the weight: every warp row is loaded once and dotted
// against each row's hidden (read through L1: 8 rows of 10 KB exceed the static shared limit), so
// the per-row value is opt_lm_head's (same FMA order per row).
//
// Sampling: with inv_t == 0 the block maxima are the raw logits' (greedy, unchanged). Otherwise the
// score of entry v on row y is logit * inv_t + G(seed, pos0 + y, id_base + v) (opt_sample_sm121.cuh),
// so the maxima's argmax is a sample of softmax(logits / T) at the position row y produces. With
// logits_out non-null every raw logit is also written ([rows][vocab]) for opt_rows_topc.
#define OR_LM_MAXR 8u
extern "C" __global__ void __launch_bounds__(OR_LM_THREADS) opt_rows_lm_head(
    const __nv_bfloat16 *__restrict__ hidden_all,    // [rows][5120]
    const __nv_bfloat16 *__restrict__ w,
    float *__restrict__ block_max_all,               // [rows][chunks]
    uint32_t *__restrict__ block_argmax_all,         // [rows][chunks]
    const uint32_t vocab,
    const uint32_t rows,
    float *__restrict__ logits_out,                  // [rows][vocab] or null
    const float inv_t,
    const uint64_t seed,
    const uint32_t pos0,
    const uint32_t id_base,
    const uint32_t *__restrict__ row_params) {     // [rows][4] or null (see below)
    OPT_PDL_ENTRY();
    __shared__ float s_best[OR_LM_MAXR][OR_LM_WARPS];
    __shared__ uint32_t s_best_i[OR_LM_MAXR][OR_LM_WARPS];
    // Each row's noise: (inv_t, seed, position). Without row_params every row takes the scalar
    // arguments at position pos0 + y; with them (rows of different requests) row y reads
    // {inv_t bits, position, seed low, seed high} from row_params[4y..].
    __shared__ float s_inv_t[OR_LM_MAXR];
    __shared__ uint64_t s_seed[OR_LM_MAXR];
    __shared__ uint32_t s_pos[OR_LM_MAXR];
    if (threadIdx.x < OR_LM_MAXR) {
        const uint32_t y = threadIdx.x;
        if (row_params != nullptr && y < rows) {
            s_inv_t[y] = __uint_as_float(row_params[4u * y]);
            s_pos[y] = row_params[4u * y + 1u];
            s_seed[y] = (uint64_t)row_params[4u * y + 2u] | ((uint64_t)row_params[4u * y + 3u] << 32);
        } else {
            s_inv_t[y] = inv_t;
            s_pos[y] = pos0 + y;
            s_seed[y] = seed;
        }
    }
    __syncthreads();
    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31u;
    const uint32_t warp = t >> 5;
    const uint32_t per = OR_LM_HIDDEN / 8u;
    const uint4 *hidden4 = reinterpret_cast<const uint4 *>(hidden_all);
    const uint32_t base = blockIdx.x * OR_LM_CHUNK;
    float best[OR_LM_MAXR];
    uint32_t best_i[OR_LM_MAXR];
#pragma unroll
    for (uint32_t y = 0; y < OR_LM_MAXR; ++y) { best[y] = -3.402823466e38f; best_i[y] = 0xffffffffu; }
    for (uint32_t r = warp; r < OR_LM_CHUNK; r += OR_LM_WARPS) {
        const uint32_t entry = base + r;
        if (entry >= vocab) break;
        const uint4 *row = reinterpret_cast<const uint4 *>(w + (uint64_t)entry * OR_LM_HIDDEN);
        uint4 wv[20];
#pragma unroll
        for (uint32_t i = 0; i < 20u; ++i) wv[i] = or_ldw(row + lane + 32u * i);
#pragma unroll
        for (uint32_t y = 0; y < OR_LM_MAXR; ++y) {
            if (y < rows) {
                const uint4 *h = hidden4 + y * per;
                float acc = 0.0f;
#pragma unroll
                for (uint32_t i = 0; i < 20u; ++i) acc = or_dot8(wv[i], __ldg(h + lane + 32u * i), acc);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
                if (logits_out != nullptr && lane == 0u) logits_out[(uint64_t)y * vocab + entry] = acc;
                const float row_inv_t = s_inv_t[y];
                const float score = row_inv_t == 0.0f
                    ? acc
                    : fmaf(acc, row_inv_t, opt_sample_gumbel(s_seed[y], s_pos[y], id_base + entry));
                if (best_i[y] == 0xffffffffu || score > best[y]) { best[y] = score; best_i[y] = entry; }
            }
        }
    }
    if (lane == 0u) {
#pragma unroll
        for (uint32_t y = 0; y < OR_LM_MAXR; ++y) { s_best[y][warp] = best[y]; s_best_i[y][warp] = best_i[y]; }
    }
    __syncthreads();
    if (t < rows) {
        const uint32_t y = t;
        float v = s_best[y][0];
        uint32_t idx = s_best_i[y][0];
        for (uint32_t k = 1; k < OR_LM_WARPS; ++k) {
            const uint32_t oi = s_best_i[y][k];
            if (oi == 0xffffffffu) continue;
            if (idx == 0xffffffffu || s_best[y][k] > v || (s_best[y][k] == v && oi < idx)) { v = s_best[y][k]; idx = oi; }
        }
        block_max_all[(uint64_t)y * gridDim.x + blockIdx.x] = v;
        block_argmax_all[(uint64_t)y * gridDim.x + blockIdx.x] = idx == 0xffffffffu ? 0u : idx;
    }
}

// Per row: fold the chunk maxima (lower chunk wins ties, as the host fold does: strictly greater
// replaces) into one (value, local index) pair. Grid rows, 32 threads.
extern "C" __global__ void opt_rows_argmax_fold(
    const float *__restrict__ block_max_all,
    const uint32_t *__restrict__ block_argmax_all,
    const uint32_t chunks,
    float *__restrict__ out_value,      // [rows]
    uint32_t *__restrict__ out_index) { // [rows]
    OPT_PDL_ENTRY();
    if (threadIdx.x != 0u) return;
    const uint32_t y = blockIdx.x;
    float best = 0.0f;
    uint32_t index = 0u;
    bool have = false;
    for (uint32_t c = 0; c < chunks; ++c) {
        const float v = block_max_all[(uint64_t)y * chunks + c];
        if (!have || v > best) { best = v; index = block_argmax_all[(uint64_t)y * chunks + c]; have = true; }
    }
    out_value[y] = best;
    out_index[y] = index;
}

// The cross-rank argmax over the exchanged folds: gather is [world][values rows | indices rows]
// (each rank's opt_rows_argmax_fold output, rank-major). Row r's winner is the first rank holding
// the greatest value (strictly greater replaces, as the host gather does), its index offset by
// rank * vocab. ids_out[r] = global index, values_out[r] (when given) = value. One thread per row.
extern "C" __global__ void opt_rows_argmax_pick(
    const uint32_t *__restrict__ gather,
    const uint32_t rows,
    const uint32_t world,
    const uint32_t vocab,
    uint32_t *__restrict__ ids_out,
    float *__restrict__ values_out) {
    OPT_PDL_ENTRY();
    const uint32_t r = threadIdx.x;
    if (r >= rows) return;
    float best = 0.0f;
    uint32_t index = 0u;
    bool have = false;
    for (uint32_t k = 0; k < world; ++k) {
        const float v = __uint_as_float(gather[(uint64_t)k * 2u * rows + r]);
        if (!have || v > best) {
            best = v;
            index = gather[(uint64_t)k * 2u * rows + rows + r] + k * vocab;
            have = true;
        }
    }
    ids_out[r] = index;
    if (values_out != nullptr) values_out[r] = best;
}

// ---------------------------------------------------------------------------------------------
// Truncated sampling (top-k / top-p): one row's slice statistics for the host cut.
// ---------------------------------------------------------------------------------------------
#define OR_TOPC 128u
#define OR_TOPC_THREADS 1024u
#define OR_TOPC_WARPS (OR_TOPC_THREADS / 32u)
#define OR_TOPC_WORDS (2u + 2u * OR_TOPC)

// The order-preserving key of a logit: a larger float has a larger key.
__device__ __forceinline__ uint32_t or_topc_key(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Row y's slot ([OR_TOPC_WORDS] u32): [0] the slice's largest logit, [1] the sum over the slice of
// exp((logit - largest) * inv_t), then OR_TOPC (logit bits, global id) pairs: the slice's OR_TOPC
// largest logits, ties to the lower id, in no particular order. The C-th largest key comes from a
// four-pass byte radix select; thread t owns the contiguous ids [t * per, (t + 1) * per), so a
// block prefix over the per-thread tie counts orders the tied ids ascending. Every reduction runs
// in one fixed order, so the slot is the same bytes on every run. Grid rows, block 1024.
extern "C" __global__ void __launch_bounds__(OR_TOPC_THREADS) opt_rows_topc(
    const float *__restrict__ logits_all,  // [rows][vocab]
    const uint32_t vocab,
    const float inv_t_all,
    const uint32_t id_base,
    uint32_t *__restrict__ slot_all,       // [rows][OR_TOPC_WORDS]
    const uint32_t *__restrict__ row_params) {  // opt_rows_lm_head's, or null: inv_t_all
    OPT_PDL_ENTRY();
    __shared__ uint32_t s_hist[256];
    __shared__ float s_red[OR_TOPC_WARPS];
    __shared__ uint32_t s_scan[OR_TOPC_WARPS];
    __shared__ uint32_t s_prefix;
    __shared__ uint32_t s_want;
    __shared__ uint32_t s_written;
    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31u;
    const uint32_t warp = t >> 5;
    const uint32_t y = blockIdx.x;
    const float inv_t = row_params != nullptr ? __uint_as_float(row_params[4u * y]) : inv_t_all;
    const float *logits = logits_all + (uint64_t)y * vocab;
    uint32_t *slot = slot_all + (uint64_t)y * OR_TOPC_WORDS;
    const uint32_t per = (vocab + OR_TOPC_THREADS - 1u) / OR_TOPC_THREADS;
    const uint32_t lo = t * per < vocab ? t * per : vocab;
    const uint32_t hi = lo + per < vocab ? lo + per : vocab;

    // The slice maximum.
    float m = -3.402823466e38f;
    for (uint32_t v = lo; v < hi; ++v) m = fmaxf(m, logits[v]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0u) s_red[warp] = m;
    __syncthreads();
    m = s_red[0];
    for (uint32_t k = 1; k < OR_TOPC_WARPS; ++k) m = fmaxf(m, s_red[k]);
    __syncthreads();

    // The slice normaliser relative to the maximum, at the request's temperature.
    float z = 0.0f;
    for (uint32_t v = lo; v < hi; ++v) z += expf((logits[v] - m) * inv_t);
    for (int o = 16; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    if (lane == 0u) s_red[warp] = z;
    __syncthreads();
    if (t == 0u) {
        float total = 0.0f;
        for (uint32_t k = 0; k < OR_TOPC_WARPS; ++k) total += s_red[k];
        slot[0] = __float_as_uint(m);
        slot[1] = __float_as_uint(total);
        s_prefix = 0u;
        s_want = OR_TOPC;
        s_written = 0u;
    }
    __syncthreads();

    // The OR_TOPC-th largest key, one byte per pass from the top.
    uint32_t mask = 0u;
    for (int pass = 3; pass >= 0; --pass) {
        const uint32_t shift = (uint32_t)pass * 8u;
        for (uint32_t b = t; b < 256u; b += OR_TOPC_THREADS) s_hist[b] = 0u;
        __syncthreads();
        const uint32_t prefix = s_prefix;
        for (uint32_t v = lo; v < hi; ++v) {
            const uint32_t key = or_topc_key(logits[v]);
            if ((key & mask) == prefix) atomicAdd(&s_hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (t == 0u) {
            uint32_t want = s_want;
            for (int b = 255; b >= 0; --b) {
                const uint32_t count = s_hist[b];
                if (count >= want) {
                    s_prefix = prefix | ((uint32_t)b << shift);
                    break;
                }
                want -= count;
            }
            s_want = want;
        }
        mask |= 255u << shift;
        __syncthreads();
    }
    const uint32_t kth = s_prefix;
    const uint32_t ties_wanted = s_want;
    const uint32_t above = OR_TOPC - ties_wanted;

    // The keys above the cut, in any order, then the tied keys in ascending id order.
    uint32_t ties = 0u;
    for (uint32_t v = lo; v < hi; ++v) {
        const float value = logits[v];
        const uint32_t key = or_topc_key(value);
        if (key > kth) {
            const uint32_t at = atomicAdd(&s_written, 1u);
            slot[2u + 2u * at] = __float_as_uint(value);
            slot[3u + 2u * at] = id_base + v;
        } else if (key == kth) {
            ++ties;
        }
    }
    uint32_t before = ties;
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t other = __shfl_up_sync(0xffffffffu, before, o);
        if (lane >= (uint32_t)o) before += other;
    }
    if (lane == 31u) s_scan[warp] = before;
    __syncthreads();
    uint32_t base = 0u;
    for (uint32_t k = 0; k < warp; ++k) base += s_scan[k];
    uint32_t rank = base + before - ties;
    for (uint32_t v = lo; v < hi && rank < ties_wanted; ++v) {
        const float value = logits[v];
        if (or_topc_key(value) == kth) {
            const uint32_t at = above + rank;
            slot[2u + 2u * at] = __float_as_uint(value);
            slot[3u + 2u * at] = id_base + v;
            ++rank;
        }
    }
}

// L2 prefetch of up to four weight spans (address, bytes; a zero length skips the span) as 32 KB
// bulk prefetches from one warp. It runs on the side stream while the run's stream waits in an
// all-reduce, so the next kernels' weights arrive while DRAM would otherwise idle. A hint: nothing
// waits on it and nothing it reads is written during the run.
extern "C" __global__ void opt_rows_prefetch_l2(
    const uint64_t a0, const uint64_t n0, const uint64_t a1, const uint64_t n1,
    const uint64_t a2, const uint64_t n2, const uint64_t a3, const uint64_t n3) {
    const uint64_t address[4] = {a0, a1, a2, a3};
    const uint64_t length[4] = {n0, n1, n2, n3};
    constexpr uint64_t CHUNK = 32768u;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        // Rounded inward to 16 bytes, so the prefetch never touches bytes outside the span.
        const uint64_t begin = (address[s] + 15ull) & ~15ull;
        const uint64_t end = (address[s] + length[s]) & ~15ull;
        if (end <= begin) continue;
        for (uint64_t at = begin + threadIdx.x * CHUNK; at < end; at += blockDim.x * CHUNK) {
            const uint32_t bytes = (uint32_t)(end - at < CHUNK ? end - at : CHUNK);
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(at), "r"(bytes) : "memory");
        }
    }
}

// dst[0] = value: a host-chosen id written in stream order, without a host copy.
extern "C" __global__ void opt_rows_store_u32(uint32_t *__restrict__ dst, const uint32_t value) {
    OPT_PDL_ENTRY();
    if (threadIdx.x == 0u) dst[0] = value;
}

// ---------------------------------------------------------------------------------------------
// Embedding rows from device ids (the host all-gather of the owners' bf16 rows, on the device).
// ---------------------------------------------------------------------------------------------
#define OR_E_WORDS (OR_S_HIDDEN * 2u / 16u)

// Row r of `slot` is the embedding row of ids[r] when this rank owns it (ids[r] / vocab == rank);
// the rows other ranks own come from their slots. grid rows, block 256.
extern "C" __global__ void opt_rows_embed_own(const uint4 *__restrict__ embed, const uint32_t *__restrict__ ids, const uint32_t rank, const uint32_t vocab, uint4 *__restrict__ slot) {
    OPT_PDL_ENTRY();
    const uint32_t r = blockIdx.x;
    const uint32_t token = ids[r];
    if (token / vocab != rank) return;
    const uint4 *src = embed + (uint64_t)(token % vocab) * OR_E_WORDS;
    uint4 *dst = slot + (uint64_t)r * OR_E_WORDS;
    for (uint32_t i = threadIdx.x; i < OR_E_WORDS; i += blockDim.x) dst[i] = src[i];
}

// Row r's stream: rows below n are ids[r]'s row from its owner's slot of the exchanged gather
// (`[world][n][hidden]` bf16), rows from n on are `noise` (f32); each is widened to f32 and copied
// over the hc axis, and the row's pre mix is the identity seed (begin_position's layout).
// grid rows, block 256.
extern "C" __global__ void opt_rows_embed_stream(const uint2 *__restrict__ gather, const uint32_t *__restrict__ ids, const uint32_t n, const uint32_t vocab, const float4 *__restrict__ noise, float4 *__restrict__ stream, float *__restrict__ pre_mix) {
    OPT_PDL_ENTRY();
    const uint32_t r = blockIdx.x;
    const uint32_t quads = OR_S_HIDDEN / 4u;
    float4 *dst = stream + (uint64_t)r * OR_S_HC * quads;
    const uint2 *src = nullptr;
    if (r < n) src = gather + ((uint64_t)(ids[r] / vocab) * n + r) * quads;
    for (uint32_t q = threadIdx.x; q < quads; q += blockDim.x) {
        float4 v;
        if (src != nullptr) {
            const uint2 w = src[q];
            v = make_float4(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xffff0000u), __uint_as_float(w.y << 16), __uint_as_float(w.y & 0xffff0000u));
        } else {
            v = noise[q];
        }
#pragma unroll
        for (uint32_t c = 0; c < OR_S_HC; ++c) dst[c * quads + q] = v;
    }
    if (threadIdx.x < OR_S_HC) pre_mix[r * OR_S_HC + threadIdx.x] = threadIdx.x == 0u ? 1.0f : 0.0f;
}

// ---------------------------------------------------------------------------------------------
// Engram wkv over up to 8 rows (opt_engram_wkv arithmetic per row, one weight read).
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ uint4 or_ldw_nc(const uint8_t *p) {
    uint4 v;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(p));
    return v;
}

// Outputs row_base .. row_end of out_width (a rank's slice when split), written at
// out_all[y * out_stride + (row - row_base)].
#define OR_E_MAXR 8
extern "C" __global__ void __launch_bounds__(256) opt_rows_engram_wkv(
    const uint8_t *__restrict__ act_codes_all,     // [rows][inner]
    const uint8_t *__restrict__ act_scales_all,    // [rows][inner/32]
    const uint8_t *__restrict__ weight,
    const uint8_t *__restrict__ weight_scales,
    __nv_bfloat16 *__restrict__ out_all,           // [rows][out_stride]
    const uint32_t out_width,
    const uint32_t inner,
    const uint32_t rows,
    const uint32_t row_base,
    const uint32_t row_end,
    const uint32_t out_stride) {
    OPT_PDL_ENTRY();
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = row_base + blockIdx.x * 8u + (threadIdx.x >> 5);
    if (row >= row_end || row >= out_width) return;
    const uint32_t k_blocks = inner / 32u;
    const uint8_t *weight_row = weight + (uint64_t)row * inner;
    const uint8_t *scale_row = weight_scales + (uint64_t)(row / 32u) * k_blocks;
    float acc[OR_E_MAXR];
#pragma unroll
    for (int y = 0; y < OR_E_MAXR; ++y) acc[y] = 0.0f;
#pragma unroll 2
    for (uint32_t base = 0; base < inner; base += 512u) {
        const uint32_t col = base + 16u * lane;
        float wf[16];
        bool live = col < inner;
        if (live) {
            const uint4 wv = or_ldw_nc(weight_row + col);
            const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) or_e4m3x4(ww[k], wf + 4 * k);
        }
        const uint32_t kb = col / 32u;
        const float wscale = live ? spark_ue8m0_decode(scale_row[kb]) : 0.0f;
#pragma unroll
        for (int y = 0; y < OR_E_MAXR; ++y) {
            if ((uint32_t)y >= rows) break;
            float d = 0.0f;
            if (live) {
                const uint4 av = __ldg(reinterpret_cast<const uint4 *>(act_codes_all + (uint64_t)y * inner + col));
                const uint32_t aa[4] = {av.x, av.y, av.z, av.w};
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float af[4];
                    or_e4m3x4(aa[k], af);
#pragma unroll
                    for (int j = 0; j < 4; ++j) d = fmaf(af[j], wf[4 * k + j], d);
                }
            }
            d += __shfl_xor_sync(0xffffffffu, d, 1);
            if ((lane & 1u) == 0u && live) {
                acc[y] = __fadd_rn(acc[y], __fmul_rn(__fmul_rn(d, spark_ue8m0_decode(act_scales_all[(uint64_t)y * k_blocks + kb])), wscale));
            }
        }
    }
#pragma unroll
    for (int y = 0; y < OR_E_MAXR; ++y) {
        if ((uint32_t)y >= rows) break;
        float a = acc[y];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0u) out_all[(uint64_t)y * out_stride + (row - row_base)] = __float2bfloat16_rn(a);
    }
}

// [world][rows][chunk] (the split wkv's gathered slices) -> [rows][world * chunk]; 16-byte moves.
extern "C" __global__ void __launch_bounds__(256) opt_rows_rank_permute(
    const uint4 *__restrict__ src, uint4 *__restrict__ dst, const uint32_t rows,
    const uint32_t chunk_vecs, const uint32_t world) {
    OPT_PDL_ENTRY();
    const uint32_t total = world * rows * chunk_vecs;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const uint32_t v = i % chunk_vecs;
        const uint32_t y = (i / chunk_vecs) % rows;
        const uint32_t r = i / (chunk_vecs * rows);
        dst[((uint64_t)y * world + r) * chunk_vecs + v] = src[i];
    }
}

// ---------------------------------------------------------------------------------------------
// Activation quantiser and shared SwiGLU (layer_seams' kernels with the PDL entry)
// ---------------------------------------------------------------------------------------------
// w8a8_prologue_bf16_e4m3 over [rows][cols] bf16, a warp per MX block instead of a thread: lane i
// reads the block's value i, the warp takes the amax by shuffles (a max, so the same value in any
// order; a NaN never wins a > compare in either form, and 0 seeds both), and each lane encodes its
// own value with the block's scale. Grid ceil(rows * cols / 32 / 8), 256 threads.
extern "C" __global__ void __launch_bounds__(256) opt_rows_prologue_bf16(
    const __nv_bfloat16 *__restrict__ normed, uint8_t *__restrict__ codes,
    uint8_t *__restrict__ scales, const uint32_t cols, const uint32_t rows) {
    OPT_PDL_ENTRY();
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t block = blockIdx.x * (blockDim.x / 32u) + (threadIdx.x >> 5);
    const uint32_t blocks_per_row = cols / SPARK_M4_QUANT_BLOCK;
    if (block >= blocks_per_row * rows) return;
    const uint32_t row = block / blocks_per_row;
    const uint32_t within = block % blocks_per_row;
    const uint64_t at = (uint64_t)row * cols + (uint64_t)within * SPARK_M4_QUANT_BLOCK + lane;
    const float value = __bfloat162float(normed[at]);
    const float magnitude = value < 0.0f ? -value : value;
    float amax = magnitude > 0.0f ? magnitude : 0.0f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, amax, o);
        amax = other > amax ? other : amax;
    }
    const uint8_t scale = spark_act_quant_scale_code(amax);
    const float inverse = __fdiv_rn(1.0f, spark_ue8m0_decode(scale));
    if (lane == 0u) scales[(uint64_t)row * blocks_per_row + within] = scale;
    codes[at] = spark_fp8_e4m3_encode(__fmul_rn(value, inverse));
}

// shared_swiglu_f32 with the PDL entry: out = silu(clamped gate) * clamped up, per element.
extern "C" __global__ void __launch_bounds__(256) opt_rows_shared_swiglu(
    const float *__restrict__ gate, const float *__restrict__ up, float *__restrict__ out,
    const float limit, const uint32_t inter) {
    OPT_PDL_ENTRY();
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= inter) return;
    float g = gate[index];
    float u = up[index];
    if (limit > 0.0f) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    const float silu = __fdiv_rn(g, __fadd_rn(1.0f, expf(-g)));
    out[index] = __fmul_rn(silu, u);
}

// ---------------------------------------------------------------------------------------------
// Host-weight GEMVs over rows
// ---------------------------------------------------------------------------------------------
// b12x_hostw_gemv_m1's body once per row (grid cols/32 x rows, one warp per block): row y reads
// a_all + y * a_stride and writes out_all + y * out_stride elements, and its chain is the M = 1
// kernel's own, so each row is bit-identical to that row's single-row launch.
#include "b12x_hostw_gemv_m1_sm121.cu"

template <typename W, bool k_out_f32>
__device__ __forceinline__ void or_hostw_rows(const float *a_all, const W *w, void *out_all,
                                              uint32_t cols, uint32_t inner, uint32_t a_stride,
                                              uint32_t out_stride) {
    const uint64_t y = blockIdx.y;
    uint8_t *out = reinterpret_cast<uint8_t *>(out_all) + y * out_stride * (k_out_f32 ? 4u : 2u);
    spark_hw_gemv_m1_body<W, k_out_f32>(a_all + y * a_stride, w, out, 1u, cols, inner, nullptr, 0u,
                                        0u);
}

extern "C" __global__ void __launch_bounds__(32) opt_rows_hostw_bf16w(
    const float *a_all, const __nv_bfloat16 *__restrict__ w, void *out_all, const uint32_t cols,
    const uint32_t inner, const uint32_t a_stride, const uint32_t out_stride) {
    OPT_PDL_ENTRY();
    or_hostw_rows<__nv_bfloat16, false>(a_all, w, out_all, cols, inner, a_stride, out_stride);
}

extern "C" __global__ void __launch_bounds__(32) opt_rows_hostw_bf16w_f32o(
    const float *a_all, const __nv_bfloat16 *__restrict__ w, void *out_all, const uint32_t cols,
    const uint32_t inner, const uint32_t a_stride, const uint32_t out_stride) {
    OPT_PDL_ENTRY();
    or_hostw_rows<__nv_bfloat16, true>(a_all, w, out_all, cols, inner, a_stride, out_stride);
}

extern "C" __global__ void __launch_bounds__(32) opt_rows_hostw_f32w_f32o(
    const float *a_all, const float *__restrict__ w, void *out_all, const uint32_t cols,
    const uint32_t inner, const uint32_t a_stride, const uint32_t out_stride) {
    OPT_PDL_ENTRY();
    or_hostw_rows<float, true>(a_all, w, out_all, cols, inner, a_stride, out_stride);
}

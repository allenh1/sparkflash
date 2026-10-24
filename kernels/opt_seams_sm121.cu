// kernels/opt_seams_sm121.cu
//
// The mHC seams of the decode step spread over the GPU instead of one block.
//
// opt_mhc_seam: the entry (attention) or FFN seam in one launch. Each of the 80 blocks owns 64
//   columns of the 5,120-wide hidden across all four hc copies: it reads its 256 stream values,
//   the 24 predictor rows over them, folds the 24 partial dots and the stream sum of squares, and
//   collapses its columns with the INPUT pre mix (hc_pre) into the row buffer with that row's sum of
//   squares. The last block to finish (threadfence + ticket) folds the 80 partials in block order,
//   runs the coefficient tail on one lane of the last warp while the other warps apply the RMSNorm
//   and, when codes is set, quantise the normed row (E4M3 + ue8m0 per 32; quant_narrow selects the
//   attention prologue's BF16-narrowed rule over the FFN's raw f32 rule). The per-element
//   expressions are layer_seams_sm121.cu's; the reduction order of the three sums is a different
//   tree (the D54 tolerance class those kernels already declare).
//
// opt_mhc_post: hc_post for the attention residual or the FFN epilogue, one thread per column.

#include "opt_pdl_sm121.cuh"
#include <cuda_bf16.h>
#include <stdint.h>

#include "m4_common_sm121.cuh"
#include "m4_activation_quant_sm121.cuh"

#define OPT_S_HIDDEN 5120u
#define OPT_S_HC 4u
#define OPT_S_WIDTH (OPT_S_HC * OPT_S_HIDDEN)
#define OPT_S_ROWS 24u
#define OPT_S_THREADS 256u
#define OPT_S_WARPS (OPT_S_THREADS / 32u)
#define OPT_S_COLS 64u
#define OPT_S_BLOCKS (OPT_S_HIDDEN / OPT_S_COLS)
#define OPT_S_PART 26u          // 24 raw dots, the stream sum of squares, the row sum of squares
#define OPT_S_PART_STRIDE 32u
#define OPT_S_QBLOCKS (OPT_S_HIDDEN / 32u)

static_assert(OPT_S_COLS * OPT_S_HC == OPT_S_THREADS, "one stream value per thread");

__device__ __forceinline__ float opt_s_bf16_round(float value) {
    const uint32_t bits = __float_as_uint(value);
    if (value != value) {
        return __uint_as_float((((bits >> 16) | 0x0040u)) << 16);
    }
    const uint32_t round = 0x7FFFu + ((bits >> 16) & 1u);
    return __uint_as_float(((bits + round) >> 16) << 16);
}

__device__ __forceinline__ float opt_s_sigmoid(float value) {
    if (value >= 0.0f) {
        return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-value)));
    }
    const float exponential = expf(value);
    return __fdiv_rn(exponential, __fadd_rn(1.0f, exponential));
}

__device__ __forceinline__ float opt_s_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(value, __shfl_down_sync(0xFFFFFFFFu, value, offset));
    }
    return value;
}

__device__ __forceinline__ void opt_s_normalise_columns(float comb[OPT_S_HC][OPT_S_HC], float hc_eps) {
    for (uint32_t column = 0; column < OPT_S_HC; ++column) {
        float sum = hc_eps;
        for (uint32_t row = 0; row < OPT_S_HC; ++row) sum = __fadd_rn(sum, comb[row][column]);
        const float inverse = __fdiv_rn(1.0f, sum);
        for (uint32_t row = 0; row < OPT_S_HC; ++row) comb[row][column] = __fmul_rn(comb[row][column], inverse);
    }
}

// layer_seams_sm121.cu's coefficient tail, one thread.
__device__ void opt_s_coefficient_tail(const float *raw, float sum_squares, const float *hc_scale,
                                       const float *hc_base, float norm_eps, float hc_eps,
                                       uint32_t iterations, float *mix_table, float *pre_mix_out) {
    const uint32_t hc = OPT_S_HC;
    const float mean_square = __fdiv_rn(sum_squares, (float)OPT_S_WIDTH);
    const float rsqrt = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean_square, norm_eps)));
    float mix[OPT_S_ROWS];
#pragma unroll
    for (uint32_t row = 0; row < OPT_S_ROWS; ++row) mix[row] = __fmul_rn(raw[row], rsqrt);
    for (uint32_t c = 0; c < hc; ++c) {
        const float pre = __fadd_rn(opt_s_sigmoid(__fadd_rn(__fmul_rn(mix[c], hc_scale[0]), hc_base[c])), hc_eps);
        mix_table[c] = pre;
        if (pre_mix_out != nullptr) pre_mix_out[c] = pre;
    }
    for (uint32_t c = 0; c < hc; ++c) {
        const uint32_t row = hc + c;
        mix_table[row] = __fmul_rn(2.0f, opt_s_sigmoid(__fadd_rn(__fmul_rn(mix[row], hc_scale[1]), hc_base[row])));
    }
    float comb[OPT_S_HC][OPT_S_HC];
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
    opt_s_normalise_columns(comb, hc_eps);
    for (uint32_t iteration = 1; iteration < iterations; ++iteration) {
        for (uint32_t row = 0; row < hc; ++row) {
            float sum = 0.0f;
            for (uint32_t column = 0; column < hc; ++column) sum = __fadd_rn(sum, comb[row][column]);
            const float inverse = __fdiv_rn(1.0f, __fadd_rn(sum, hc_eps));
            for (uint32_t column = 0; column < hc; ++column) comb[row][column] = __fmul_rn(comb[row][column], inverse);
        }
        opt_s_normalise_columns(comb, hc_eps);
    }
    for (uint32_t row = 0; row < hc; ++row)
        for (uint32_t column = 0; column < hc; ++column) mix_table[2u * hc + row * hc + column] = comb[row][column];
}

extern "C" __global__ void __launch_bounds__(OPT_S_THREADS) opt_mhc_seam(
    const float *__restrict__ stream,       // [hc][dim]
    const float *in_pre,                    // [hc] the input pre mix (hc_pre's coefficients)
    const float *__restrict__ fn_w,         // [24][hc*dim]
    const float *__restrict__ hc_base,      // [24]
    const float *__restrict__ hc_scale,     // [3]
    const uint8_t *__restrict__ norm_w,     // [dim] f32 or bf16
    float *mix_table,                       // [24] out
    float *pre_mix_out,                     // [hc] out or null
    float *__restrict__ hc_rows,            // [hc][dim] out copy of the stream or null
    float *__restrict__ normed,             // [dim] out
    uint8_t *__restrict__ codes,            // [dim] out or null
    uint8_t *__restrict__ scales,           // [dim/32] out or null
    __nv_bfloat16 *__restrict__ x_bf16,     // [dim] out or null
    float *partials,                        // [80][32] scratch
    float *row_buf,                         // [dim] scratch: the hc_pre row
    uint32_t *ticket,
    const uint32_t norm_bf16,
    const uint32_t native_bf16,
    const uint32_t quant_narrow,
    const uint32_t sinkhorn_iterations,
    const float norm_eps,
    const float hc_eps) {
    OPT_PDL_ENTRY();
    __shared__ float xs[OPT_S_THREADS];
    __shared__ float red[OPT_S_WARPS][OPT_S_PART_STRIDE];
    __shared__ float tot[OPT_S_PART_STRIDE];
    __shared__ uint32_t s_last;
    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31u;
    const uint32_t warp = t >> 5;

    {
        const uint32_t copy = t / OPT_S_COLS;
        const uint32_t column = blockIdx.x * OPT_S_COLS + (t % OPT_S_COLS);
        const uint32_t index = copy * OPT_S_HIDDEN + column;
        const float x = stream[index];
        xs[t] = x;
        if (hc_rows != nullptr) hc_rows[index] = x;
        float w[OPT_S_ROWS];
#pragma unroll
        for (uint32_t r = 0; r < OPT_S_ROWS; ++r) w[r] = __ldg(fn_w + (uint64_t)r * OPT_S_WIDTH + index);
#pragma unroll
        for (uint32_t r = 0; r < OPT_S_ROWS; ++r) {
            const float s = opt_s_warp_sum(__fmul_rn(w[r], x));
            if (lane == 0u) red[warp][r] = s;
        }
        const float sq = opt_s_warp_sum(__fmul_rn(x, x));
        if (lane == 0u) red[warp][24] = sq;
        __syncthreads();
        if (t < OPT_S_COLS) {
            float accumulated = 0.0f;
#pragma unroll
            for (uint32_t c = 0; c < OPT_S_HC; ++c) {
                accumulated = __fadd_rn(__fmul_rn(in_pre[c], xs[c * OPT_S_COLS + t]), accumulated);
            }
            if (native_bf16 != 0u) accumulated = opt_s_bf16_round(accumulated);
            row_buf[blockIdx.x * OPT_S_COLS + t] = accumulated;
            const float rs = opt_s_warp_sum(__fmul_rn(accumulated, accumulated));
            if (lane == 0u) red[warp][25] = rs;
        }
        __syncthreads();
        if (t < OPT_S_PART) {
            const uint32_t warps = t == 25u ? OPT_S_COLS / 32u : OPT_S_WARPS;
            float s = red[0][t];
            for (uint32_t k = 1; k < warps; ++k) s = __fadd_rn(s, red[k][t]);
            partials[blockIdx.x * OPT_S_PART_STRIDE + t] = s;
        }
    }

    __threadfence();
    __syncthreads();
    if (t == 0u) s_last = atomicAdd(ticket, 1u) == gridDim.x - 1u ? 1u : 0u;
    __syncthreads();
    if (s_last == 0u) return;
    __threadfence();

    // Fold the 80 block partials of each value, block order fixed per lane then a warp tree.
    for (uint32_t v = warp; v < OPT_S_PART; v += OPT_S_WARPS) {
        float s = 0.0f;
        for (uint32_t b = lane; b < OPT_S_BLOCKS; b += 32u) s = __fadd_rn(s, __ldcg(partials + b * OPT_S_PART_STRIDE + v));
        s = opt_s_warp_sum(s);
        if (lane == 0u) tot[v] = s;
    }
    __syncthreads();

    if (warp == OPT_S_WARPS - 1u) {
        if (lane == 0u) {
            opt_s_coefficient_tail(tot, tot[24], hc_scale, hc_base, norm_eps, hc_eps, sinkhorn_iterations,
                                   mix_table, pre_mix_out);
            *ticket = 0u;
        }
        return;
    }
    const float mean = __fdiv_rn(tot[25], (float)OPT_S_HIDDEN);
    const float inverse = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, norm_eps)));
    // Every load of this warp's columns is issued before any arithmetic: one L2 round trip, not
    // one per 32-column block.
    constexpr uint32_t NORM_WARPS = OPT_S_WARPS - 1u;
    constexpr uint32_t PER_WARP = (OPT_S_QBLOCKS + NORM_WARPS - 1u) / NORM_WARPS;
    float rows_v[PER_WARP], weights_v[PER_WARP];
#pragma unroll
    for (uint32_t i = 0; i < PER_WARP; ++i) {
        const uint32_t qb = warp + i * NORM_WARPS;
        if (qb < OPT_S_QBLOCKS) {
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
        if (qb >= OPT_S_QBLOCKS) break;
        const uint32_t column = qb * 32u + lane;
        float value = __fmul_rn(__fmul_rn(rows_v[i], inverse), weights_v[i]);
        if (native_bf16 != 0u) value = opt_s_bf16_round(value);
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

// hc_post: out[c][j] = post[c] * x[j] + sum_s comb[s][c] * residual[s][j] (or the bf16 arm), with
// x = delta[j] (+ shared[j] when shared is set). In place is safe: column j reads only column j.
extern "C" __global__ void __launch_bounds__(128) opt_mhc_post(
    const float *residual,                  // [hc][dim]
    const float *__restrict__ mix_table,    // [24]
    const __nv_bfloat16 *__restrict__ delta,// [dim]
    const float *__restrict__ shared,       // [dim] f32 or null
    float *stream_out,                      // [hc][dim]
    const uint32_t native_bf16) {
    OPT_PDL_ENTRY();
    const uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= OPT_S_HIDDEN) return;
    const uint32_t hc = OPT_S_HC;
    float x = __bfloat162float(delta[column]);
    if (shared != nullptr) x = __fadd_rn(x, shared[column]);
    float r[OPT_S_HC];
#pragma unroll
    for (uint32_t s = 0; s < hc; ++s) r[s] = residual[s * OPT_S_HIDDEN + column];
#pragma unroll
    for (uint32_t c = 0; c < hc; ++c) {
        const float post = mix_table[hc + c];
        float value;
        if (native_bf16 != 0u) {
            float comb_sum = 0.0f;
#pragma unroll
            for (uint32_t s = 0; s < hc; ++s) comb_sum = __fadd_rn(comb_sum, __fmul_rn(mix_table[2u * hc + s * hc + c], r[s]));
            value = opt_s_bf16_round(__fadd_rn(__fmul_rn(post, x), comb_sum));
        } else {
            float accumulated = __fmul_rn(post, x);
#pragma unroll
            for (uint32_t s = 0; s < hc; ++s) accumulated = __fadd_rn(accumulated, __fmul_rn(mix_table[2u * hc + s * hc + c], r[s]));
            value = accumulated;
        }
        stream_out[c * OPT_S_HIDDEN + column] = value;
    }
}

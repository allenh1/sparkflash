// kernels/mega_mhc_sm121.cu
//
// The fused Mega-mHC kernel of specification 2.4 (single-pass manifold-constrained
// hyper-connections, DeepSeek-V4.1-Flash). One kernel per layer does, in one sweep over the
// hidden dimension:
//
//   1. X_l <- B_{l-1} X_{l-1} + C_{l-1} Y_{l-1}   (updated in registers, written back once)
//   2. X_hat_l = A_{l-1} X_l                       (from those same registers: the shift of the
//                                                   input-mixing coefficients by one layer is
//                                                   the point of 2.4 - no second pass)
//   3. tile dot products and sum-of-squares for H(X_l), then 20 Sinkhorn-Knopp iterations
//   4. input RMS pre-norm and dynamic 32x32 ue8m0 FP8 quantization of X_hat_l
//
// Write set: exactly X_l (n*d bf16) plus the quantized block input (d FP8 + d/32 scales). The
// traffic bound (2n+2)d of specification 2.4 is checked by the CPU reference in
// crates/sparkflash-kernels/src/mega_mhc_ref.rs, which counts every element the reference
// touches per stage; see that module for the model and for the deliberately multi-pass variant
// that proves the bound is not vacuous.
//
// H() interpretation, stated plainly because the specification does not fix its internals
// (2.4 only says "RMSNorm weights folded offline into the predictor projection"): the folded
// predictor is a (n*n + 2n) x d matrix applied to the stream mean of X_l. Its first n*n rows
// are the raw stream-mixing logits (exponentiated and Sinkhorn-normalised into B), the next n
// rows are the input-mixing logits (softmax into A), and the last n rows are C. The predictor
// input is the mean across the n streams, so H() costs (n*n + 2n)*d multiply-accumulates, and
// the 25 accumulated scalars per tile are what the grid reduces.
//
// Verification status: this kernel has never been executed - there is no GPU on the development
// host. It is compiled for sm_121a by nvcc 12.9.1 through the build-script probe (PTX and
// cubin), parsed by the clang CUDA frontend against stub headers when nvcc is absent, and its
// ABI is pinned by static asserts here plus parsed-declaration tests in
// crates/sparkflash-kernels/src/m4_abi.rs. The numerics are defined and tested by
// mega_mhc_ref.rs on the CPU. Runtime behaviour - the grid rendezvous in particular - is NOT
// verified and must not be claimed until it runs on hardware.

#include "m4_common_sm121.cuh"

// The hidden dimension is swept as four 1280-element tiles, one block each: four tiles covering
// d = 5120, which is the "4 x 5120 hidden-dim tiles" reading of 2.4.
#define SPARK_M4_MHC_TILE_ELEMS 1280u
#define SPARK_M4_MHC_TILES (SPARK_M4_HIDDEN / SPARK_M4_MHC_TILE_ELEMS)
#define SPARK_M4_MHC_THREADS 128u
// One dot product per predictor row plus the sum of squares of the stream mean.
#define SPARK_M4_MHC_STATS (SPARK_M4_HC_PREDICTOR_ROWS + 1u)
// hc_sync[0] counts finished tiles, hc_sync[1] is the coefficients-ready flag.
#define SPARK_M4_MHC_SYNC_WORDS 2u

static_assert(SPARK_M4_MHC_TILES == 4u, "the sweep is four tiles");
static_assert(SPARK_M4_MHC_TILE_ELEMS * SPARK_M4_MHC_TILES == SPARK_M4_HIDDEN,
              "the tiles must cover the hidden dimension exactly");
static_assert(SPARK_M4_MHC_TILE_ELEMS % SPARK_M4_QUANT_BLOCK == 0u,
              "a tile boundary must not split an MX scale block");
static_assert(SPARK_M4_MHC_TILES * SPARK_M4_MHC_THREADS >= SPARK_M4_MHC_STATS,
              "the block reduction needs at least one thread per statistic");
static_assert(SPARK_M4_MHC_THREADS % 32u == 0u, "a block is a whole number of warps");
static_assert(SPARK_M4_MHC_STATS == 25u, "24 predictor rows plus one sum of squares");

extern "C" __global__ void mega_mhc_fused_bf16_sm121(
    __nv_bfloat16 *__restrict__ x,              // [n][d] residual streams, updated in place
    const __nv_bfloat16 *__restrict__ y_prev,   // [d] previous layer block output Y_{l-1}
    const float *__restrict__ b_prev,           // [n][n] row-major stream mixing of layer l-1
    const float *__restrict__ c_prev,           // [n] block-output weights of layer l-1
    const float *__restrict__ a_prev,           // [n] input-mixing weights of layer l-1
    const float *__restrict__ predictor,        // [n*n + 2n][d] folded H() predictor
    const float *__restrict__ rms_weight,       // [d] folded input RMS pre-norm weight
    uint8_t *__restrict__ xhat_fp8,             // [d] E4M3 block input, one byte per element
    uint8_t *__restrict__ xhat_scales,          // [d / 32] ue8m0 scales, one per MX block
    float *__restrict__ hc_stats,               // [25] grid scratch: dot products, sum of squares
    uint32_t *__restrict__ hc_sync,             // [2] grid scratch: tiles done, ready flag
    float *__restrict__ hc_coeffs,              // [24] out: B (n*n), A (n), C (n) for layer l
    float *__restrict__ hc_rms_out,             // [1] out: 1/rms(X_hat_l) used by the pre-norm
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t tile = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t base = tile * SPARK_M4_MHC_TILE_ELEMS;
    const uint32_t n = (uint32_t)SPARK_M4_HC_MULT;
    const uint32_t d = (uint32_t)SPARK_M4_HIDDEN;

    __shared__ float s_xhat[SPARK_M4_MHC_TILE_ELEMS];
    __shared__ float s_b[SPARK_M4_HC_MULT * SPARK_M4_HC_MULT];
    __shared__ float s_c[SPARK_M4_HC_MULT];
    __shared__ float s_a[SPARK_M4_HC_MULT];
    __shared__ float s_stats[SPARK_M4_MHC_STATS];
    __shared__ float s_reduce[SPARK_M4_MHC_THREADS];
    __shared__ float s_inv_rms;
    __shared__ uint32_t s_failed;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_MEGA_MHC, t_enter);

    // Layer l-1's coefficients are 24 scalars: read them once per block into shared memory.
    if (tid == 0u) {
        s_failed = 0u;
        s_inv_rms = 1.0f;
        for (uint32_t i = 0; i < SPARK_M4_HC_MULT * SPARK_M4_HC_MULT; ++i) {
            s_b[i] = b_prev[i];
        }
        for (uint32_t i = 0; i < SPARK_M4_HC_MULT; ++i) {
            s_c[i] = c_prev[i];
            s_a[i] = a_prev[i];
        }
        for (uint32_t i = 0; i < SPARK_M4_MHC_STATS; ++i) {
            s_stats[i] = 0.0f;
        }
    }
    __syncthreads();

    // Stages 1 and 2 of 2.4, fused: each element of X_{l-1} is read exactly once, the update is
    // formed in registers, X_l is written back, and X_hat is taken from those same registers.
    float local[SPARK_M4_MHC_STATS];
    for (uint32_t p = 0; p < SPARK_M4_MHC_STATS; ++p) {
        local[p] = 0.0f;
    }
    for (uint32_t i = tid; i < SPARK_M4_MHC_TILE_ELEMS; i += SPARK_M4_MHC_THREADS) {
        const uint32_t index = base + i;
        const float y = __bfloat162float(y_prev[index]);
        float updated[SPARK_M4_HC_MULT];
        for (uint32_t s = 0; s < n; ++s) {
            float acc = s_c[s] * y;
            for (uint32_t t = 0; t < n; ++t) {
                acc += s_b[s * n + t] * __bfloat162float(x[t * d + index]);
            }
            updated[s] = acc;
        }
        // Write X_l once. The in-place update is safe because element index of every stream is
        // read and written by exactly this thread, and nothing else depends on it.
        for (uint32_t s = 0; s < n; ++s) {
            x[s * d + index] = __float2bfloat16_rn(updated[s]);
        }
        // X_hat_l = A_{l-1} X_l from registers: no second traversal of the residual stream.
        float xhat = 0.0f;
        float mean = 0.0f;
        for (uint32_t s = 0; s < n; ++s) {
            xhat += s_a[s] * updated[s];
            mean += updated[s];
        }
        mean *= 1.0f / (float)n;
        s_xhat[i] = xhat;
        for (uint32_t p = 0; p < (uint32_t)SPARK_M4_HC_PREDICTOR_ROWS; ++p) {
            local[p] += mean * predictor[p * d + index];
        }
        local[SPARK_M4_HC_PREDICTOR_ROWS] += mean * mean;
    }

    // Block-reduce the 25 statistics, then hand them to the grid.
    for (uint32_t p = 0; p < SPARK_M4_MHC_STATS; ++p) {
        s_reduce[tid] = local[p];
        __syncthreads();
        for (uint32_t stride = SPARK_M4_MHC_THREADS / 2u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                s_reduce[tid] += s_reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            s_stats[p] = s_reduce[0];
        }
        __syncthreads();
    }

    if (tid == 0u) {
        for (uint32_t p = 0; p < SPARK_M4_MHC_STATS; ++p) {
            atomicAdd(&hc_stats[p], s_stats[p]);
        }
        // The fence makes this tile's statistics visible before the counter that counts it, the
        // same publish order the RoCEnante kernel uses for its doorbell.
        __threadfence();
        const uint32_t done = atomicAdd(&hc_sync[0], 1u) + 1u;
        if (done == (uint32_t)SPARK_M4_MHC_TILES) {
            // Last tile of the sweep: solve H(X_l) and publish the coefficients for layer l.
            float mixing[SPARK_M4_HC_MULT][SPARK_M4_HC_MULT];
            float input[SPARK_M4_HC_MULT];
            float output[SPARK_M4_HC_MULT];
            for (uint32_t r = 0; r < n; ++r) {
                for (uint32_t c = 0; c < n; ++c) {
                    mixing[r][c] = spark_exp(hc_stats[r * n + c]);
                }
            }
            spark_sinkhorn_knopp(mixing, (uint32_t)SPARK_M4_HC_SINKHORN_ITERS,
                                 (float)SPARK_M4_HC_EPS);
            // A is a softmax over the n input-mixing logits.
            float amax = hc_stats[n * n];
            for (uint32_t s = 1; s < n; ++s) {
                const float candidate = hc_stats[n * n + s];
                amax = candidate > amax ? candidate : amax;
            }
            float asum = 0.0f;
            for (uint32_t s = 0; s < n; ++s) {
                input[s] = spark_exp(hc_stats[n * n + s] - amax);
                asum += input[s];
            }
            const float ainverse = 1.0f / asum;
            for (uint32_t s = 0; s < n; ++s) {
                input[s] *= ainverse;
                output[s] = hc_stats[2u * n * n + s];
            }
            for (uint32_t r = 0; r < n; ++r) {
                for (uint32_t c = 0; c < n; ++c) {
                    hc_coeffs[r * n + c] = mixing[r][c];
                }
                hc_coeffs[n * n + r] = input[r];
                hc_coeffs[n * n + n + r] = output[r];
            }
            // The RMS pre-norm of X_hat_l over the whole hidden dimension. eps keeps an
            // all-zero block input finite; the CPU reference uses the same expression.
            const float mean_square = hc_stats[SPARK_M4_HC_PREDICTOR_ROWS] / (float)d;
            *hc_rms_out = 1.0f / spark_sqrt(mean_square + (float)SPARK_M4_HC_EPS);
            __threadfence();
            hc_sync[1] = 1u;
        }
    }
    __syncthreads();

    // Every other tile waits for those coefficients, bounded by a %globaltimer deadline.
    if (tid == 0u) {
        const uint64_t deadline = spark_read_globaltimer_ns() + (uint64_t)SPARK_M4_SYNC_TIMEOUT_NS;
        // The acquiring poll: the fence lives inside the helper, in the same expansion as the poll,
        // so this call site cannot lose it to an edit here. The publisher keeps the raw
        // spark_wait_for_flag, which is a wait without an acquire.
        const uint32_t ready = spark_wait_for_flag_acquire(&hc_sync[1], deadline);
        if (ready == 0u) {
            s_failed = 1u;
        } else {
            s_inv_rms = *hc_rms_out;
        }
    }
    __syncthreads();

    const float inverse_rms = s_inv_rms;
    const uint32_t failed = s_failed;
    const uint32_t blocks = SPARK_M4_MHC_TILE_ELEMS / SPARK_M4_QUANT_BLOCK;
    for (uint32_t b = tid; b < blocks; b += SPARK_M4_MHC_THREADS) {
        const uint32_t element = b * SPARK_M4_QUANT_BLOCK;
        const uint32_t index = base + element;
        float amax = 0.0f;
        for (uint32_t k = 0; k < SPARK_M4_QUANT_BLOCK; ++k) {
            const float value = s_xhat[element + k] * rms_weight[index + k] * inverse_rms;
            const float magnitude = value < 0.0f ? -value : value;
            amax = magnitude > amax ? magnitude : amax;
        }
        const uint8_t scale = spark_ue8m0_from_amax(amax, 8u);
        xhat_scales[index / SPARK_M4_QUANT_BLOCK] = scale;
        const float inverse_scale = 1.0f / spark_ue8m0_decode(scale);
        for (uint32_t k = 0; k < SPARK_M4_QUANT_BLOCK; ++k) {
            const float value = s_xhat[element + k] * rms_weight[index + k] * inverse_rms;
            xhat_fp8[index + k] =
                failed != 0u ? (uint8_t)0x7Fu : spark_fp8_e4m3_encode(value * inverse_scale);
        }
    }

    if (tid == 0u) {
        // payload_meta: the bytes this layer writes per token - X_l (n*d*2), the block input
        // (d) and its scales (d/32). The trace ring is the only place that number is reported.
        const uint32_t meta = (uint32_t)(n * d * 2u + d + d / SPARK_M4_QUANT_BLOCK);
        const uint32_t reported = failed != 0u ? (meta | 0x80000000u) : meta;
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_MEGA_MHC, layer_id, reported, t_enter);
        // The trace macros expand to nothing when the scope mask is compiled out, so keep these
        // reads alive explicitly instead of depending on the macro's expansion.
        (void)reported;
        (void)step_seq;
    }
    (void)t_enter;
}

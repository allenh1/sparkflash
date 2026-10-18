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
//   4. input RMS pre-norm and the reference's dynamic activation quantisation of X_hat_l: one
//      ue8m0 scale per 32 values from the act_quant ceil rule (spark_act_quant_scale_code), the
//      rule the engine's quant::quantize_activation_fp8 uses for every GEMM's activation operand.
//      The checkpoint weight-grid rule (spark_ue8m0_from_amax) is the other rule and is wrong here
//
// Write set: X_l (n*d bf16), the f32 pre-normed block input ready for quantization (d f32 - the
// B0 boundary capture, so a comparison can separate the norm from the quantizer), and the
// quantized block input (d FP8 + d/32 scales). The traffic bound (2n+2)d of specification 2.4 is
// over the residual streams (read X_l once, write X_{l+1}); the CPU reference in
// crates/sparkflash-kernels/src/mega_mhc_ref.rs counts every element the reference touches per
// stage and keeps the stream traffic and the block-input representations separate. See that
// module for the model and for the deliberately multi-pass variant that proves the bound is not
// vacuous.
//
// H() interpretation, stated plainly because the specification does not fix its internals: the
// predictor is the RELEASED (n*n + 2n) x (n*d) matrix, whose rows are the checkpoint's
// pre(0..n) / post(n..2n) / comb(2n..2n+n*n) rows permuted into this kernel's coefficient order
// B(0..n*n) / A(n*n..n*n+n) / C(n*n+n..) (the "RMSNorm weights folded offline into the predictor
// projection" of 2.4 is how the release stores the projection; it is not a d-wide fold). Every row
// projects all n copies of the flattened stream (model.py:951-954). The released arithmetic is
// applied to the accumulated values, in the released order (inference/kernel.py:426-458):
//
//   mixes   = projection * rsqrt(mean(X_l^2) + norm_eps)   over the flattened n*d stream
//   A_j     = sigmoid(mixes_j * hc_scale[0] + hc_base_j) + hc_eps
//   C_j     = 2 * sigmoid(mixes_{j+n} * hc_scale[1] + hc_base_{j+n})
//   comb    = released split-Sinkhorn(mixes_{comb} * hc_scale[2] + hc_base_{comb})
//
// with hc_base in this kernel's row order and hc_scale in the released [pre, post, comb] order.
// The two different RMS statistics the release needs are both accumulated per tile: the sum of
// squares of X_hat over d (the block input's RMSNorm) and the sum of squares of the four streams
// over the flattened n*d input (the projection's rsqrt). Together with the 24 projection rows
// that is 26 accumulated scalars per tile.
//
// Predictor fold, removed by B0 (the Phase-1 device chain, 2026-10): earlier revisions of this
// kernel and its CPU reference took the project's [24, 5120] fold - each released row averaged over
// the four copies and applied to the stream mean - because that is what a d-wide single pass can
// compute. That fold is NOT the released projection: with equal copies it computes the released
// projection divided by the copy count, which is a semantic difference, not a rounding one. This
// kernel now takes the released [24, 20480] matrix (1,966,080 bytes of f32 predictor buffer, the
// predictor argument below) and performs the full projection over all four copies, so the device
// column of the Phase-1 boundary table can be compared with the host reference. The pre-B0 stage-1
// device runs measured the folded semantics; those numbers are retired, and the post-B0 run is the
// first release-semantics stage-1 check. mega_mhc_ref.rs no longer exposes a fold helper.
//
// Verification status: compiled for sm_121a by nvcc through the build-script probe (PTX and
// cubin), parsed by the clang CUDA frontend against stub headers when nvcc is absent, and its ABI
// is pinned by static asserts here plus parsed-declaration tests in
// crates/sparkflash-kernels/src/m4_abi.rs. The numerics are defined by mega_mhc_ref.rs on the CPU.
//
// Earlier revisions of this kernel (the pre-released-arithmetic ABI) ran on hardware: then
// examples/m7_layer0.rs stage 1 matched the CPU reference on four layer-0 tokens, and stage 4
// launched the same input ten times with byte-identical output. That old PASS compared the folded
// semantics on both sides and is retired. The release-semantics device run IS the post-B0 run
// (2026-10-18: stage 0 matches the engine's captured row-0 coefficients to 5.96e-8, stage 1 PASSes
// 4 of 4 tokens at layers 0, 1 and 4 against the released-projection reference; logs
// /home/user/phase1-evidence/phase1-b0-* and phase1-d8-*). That run is also the regression test
// for the reduction below: the four block partials go to slots fixed by blockIdx and are combined
// in tile order in binary64, so the order the blocks reach the rendezvous cannot change hc_stats.
// The old fp32 atomicAdd sequence could and did.

#include "m4_common_sm121.cuh"
#include "m4_activation_quant_sm121.cuh"

// The hidden dimension is swept as four 1280-element tiles, one block each: four tiles covering
// d = 5120, which is the "4 x 5120 hidden-dim tiles" reading of 2.4.
#define SPARK_M4_MHC_TILE_ELEMS 1280u
#define SPARK_M4_MHC_TILES (SPARK_M4_HIDDEN / SPARK_M4_MHC_TILE_ELEMS)
#define SPARK_M4_MHC_THREADS 128u
// One dot product per predictor row, then the two different sums of squares the released model
// needs: one over X_hat for the block RMS pre-norm, one over the four streams for the
// coefficient projection's rsqrt. Keeping them apart is not a refinement: they are different
// functions of the input, and the old single "sum of squares of the stream mean" was neither.
#define SPARK_M4_MHC_STATS (SPARK_M4_HC_PREDICTOR_ROWS + 2u)
// hc_sync[0] counts finished tiles, hc_sync[1] is the coefficients-ready flag.
#define SPARK_M4_MHC_SYNC_WORDS 2u
// hc_stats is one final statistics vector plus one block-reduced partial vector per tile:
//
//   hc_stats[0 .. STATS)                         the 26 final statistics (published output)
//   hc_stats[(tile + 1) * STATS .. + STATS)      tile's partials (grid scratch)
//
// The launcher must allocate SPARK_M4_MHC_GRID_STATS floats, not STATS.
#define SPARK_M4_MHC_GRID_STATS (SPARK_M4_MHC_STATS * (SPARK_M4_MHC_TILES + 1u))

static_assert(SPARK_M4_MHC_TILES == 4u, "the sweep is four tiles");
static_assert(SPARK_M4_MHC_TILE_ELEMS * SPARK_M4_MHC_TILES == SPARK_M4_HIDDEN,
              "the tiles must cover the hidden dimension exactly");
static_assert(SPARK_M4_MHC_TILE_ELEMS % SPARK_M4_QUANT_BLOCK == 0u,
              "a tile boundary must not split an MX scale block");
static_assert(SPARK_M4_MHC_TILES * SPARK_M4_MHC_THREADS >= SPARK_M4_MHC_STATS,
              "the block reduction needs at least one thread per statistic");
static_assert(SPARK_M4_MHC_THREADS % 32u == 0u, "a block is a whole number of warps");
static_assert(SPARK_M4_MHC_STATS == 26u,
              "24 predictor rows plus the block RMS and the coefficient RMS statistics");

extern "C" __global__ void mega_mhc_fused_bf16_sm121(
    __nv_bfloat16 *__restrict__ x,              // [n][d] residual streams, updated in place
    const __nv_bfloat16 *__restrict__ y_prev,   // [d] previous layer block output Y_{l-1}
    const float *__restrict__ b_prev,           // [n][n] row-major stream mixing of layer l-1
    const float *__restrict__ c_prev,           // [n] block-output weights of layer l-1
    const float *__restrict__ a_prev,           // [n] input-mixing weights of layer l-1
    const float *__restrict__ predictor,        // [n*n + 2n][n * d] released H() predictor, kernel row order
    const float *__restrict__ hc_scale,         // [3] released per-group scale: pre, post, comb
    const float *__restrict__ hc_base,          // [n*n + 2n] released per-row bias, kernel row order
    const float *__restrict__ rms_weight,       // [d] folded input RMS pre-norm weight
    uint8_t *__restrict__ xhat_fp8,             // [d] E4M3 block input, one byte per element
    uint8_t *__restrict__ xhat_scales,          // [d / 32] ue8m0 scales, one per MX block
    float *__restrict__ hc_stats,               // [GRID_STATS] final 26 + one partial vector per tile
    uint32_t *__restrict__ hc_sync,             // [2] grid scratch: tiles done, ready flag
    float *__restrict__ hc_coeffs,              // [24] out: B (n*n), A (n), C (n) for layer l
    float *__restrict__ hc_rms_out,             // [1] out: 1/rms(X_hat_l) used by the pre-norm
    float *__restrict__ xhat_normed,            // [d] out: f32 X_hat_l * rms_weight * inv_rms
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
        for (uint32_t s = 0; s < n; ++s) {
            xhat += s_a[s] * updated[s];
        }
        s_xhat[i] = xhat;
        // The released predictor is [n*n + 2n][n*d]: every row projects the FLATTENED four-stream
        // input, not the stream mean (model.py:951-954). The launcher has already permuted the
        // released pre/post/comb rows into this kernel's B/A/C row order; the projection is the
        // same "accumulate first, then rstd and scale and bias in the last-tile solve" chain.
        for (uint32_t p = 0; p < (uint32_t)SPARK_M4_HC_PREDICTOR_ROWS; ++p) {
            const float *row = predictor + (uint64_t)p * (uint64_t)(n * d) + (uint64_t)index;
            float projection = 0.0f;
            for (uint32_t s = 0; s < n; ++s) {
                projection += updated[s] * row[(uint64_t)s * (uint64_t)d];
            }
            local[p] += projection;
        }
        // The block RMSNorm's statistic is over X_hat (released model.py:984/991), and the
        // projection's rsqrt statistic is over the flattened four-stream input (model.py:953-954).
        local[SPARK_M4_HC_PREDICTOR_ROWS] += xhat * xhat;
        float stream_squares = 0.0f;
        for (uint32_t s = 0; s < n; ++s) {
            stream_squares += updated[s] * updated[s];
        }
        local[SPARK_M4_HC_PREDICTOR_ROWS + 1u] += stream_squares;
    }

    // Block-reduce the 26 statistics, then hand them to the grid.
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
        // Deterministic grid reduction. The slot a tile writes is fixed by blockIdx, so the buffer
        // does not depend on arrival order; the fence makes this tile's 26 partials visible before
        // the counter that counts it, the same publish order the RoCEnante kernel uses.
        for (uint32_t p = 0; p < SPARK_M4_MHC_STATS; ++p) {
            hc_stats[(tile + 1u) * SPARK_M4_MHC_STATS + p] = s_stats[p];
        }
        __threadfence();
        const uint32_t done = atomicAdd(&hc_sync[0], 1u) + 1u;
        if (done == (uint32_t)SPARK_M4_MHC_TILES) {
            // All four partial vectors are visible. Combine them in TILE order in binary64: every
            // partial is a binary32 value, and the sum of four of them is exact in binary64 (its
            // significand needs at most 26 bits against binary64's 53) for any realistic exponent
            // spread, so the cast below performs the one rounding the fp32 atomicAdd sequence used
            // to perform four times in an order the host could not fix. The result is therefore
            // independent of which tile arrives last, and repeated launches of the same input
            // publish byte-identical hc_stats.
            float total[SPARK_M4_MHC_STATS];
            for (uint32_t p = 0; p < SPARK_M4_MHC_STATS; ++p) {
                double sum = 0.0;
                for (uint32_t b = 0u; b < (uint32_t)SPARK_M4_MHC_TILES; ++b) {
                    sum += (double)hc_stats[(b + 1u) * SPARK_M4_MHC_STATS + p];
                }
                total[p] = (float)sum;
                hc_stats[p] = total[p];
            }
            // Last tile of the sweep: solve H(X_l) and publish the coefficients for layer l.
            // Released order: the projection takes its rsqrt factor first (model.py:953-954),
            // then each group is scaled and biased (inference/kernel.py:426-431).
            const float coefficient_rstd =
                1.0f / spark_sqrt(total[SPARK_M4_HC_PREDICTOR_ROWS + 1u] / (float)(n * d) +
                                  (float)SPARK_M4_NORM_EPS);
            // mixing is in the RELEASED [source][destination] orientation, the orientation the
            // released schedule's row softmax and column passes assume (D3).
            float mixing[SPARK_M4_HC_MULT][SPARK_M4_HC_MULT];
            float input[SPARK_M4_HC_MULT];
            float output[SPARK_M4_HC_MULT];
            for (uint32_t source = 0; source < n; ++source) {
                for (uint32_t destination = 0; destination < n; ++destination) {
                    // This kernel's row order is B[destination][source]; the released comb is
                    // [source][destination].
                    const uint32_t row = destination * n + source;
                    mixing[source][destination] =
                        total[row] * coefficient_rstd * hc_scale[2] + hc_base[row];
                }
            }
            spark_released_split_sinkhorn(mixing, (uint32_t)SPARK_M4_HC_SINKHORN_ITERS,
                                          (float)SPARK_M4_HC_EPS);
            // A_j = sigmoid(mix_j * scale[0] + base_j) + hc_eps (kernel.py:427).
            for (uint32_t j = 0; j < n; ++j) {
                const float mix = total[n * n + j] * coefficient_rstd;
                input[j] = spark_sigmoid(mix * hc_scale[0] + hc_base[n * n + j]) +
                           (float)SPARK_M4_HC_EPS;
            }
            // C_j = 2 * sigmoid(mix_{j+n} * scale[1] + base_{j+n}) (kernel.py:429). The post rows are
            // the n rows after the pre rows, i.e. n*n + j, not 2*n*n + j: this kernel's row order is
            // B(0..n*n)/A(n*n..n*n+n)/C(n*n+n..n*n+2n), and the stat that carries each group is the
            // same row index, so C's two statistics are total[n*n+n+j] and hc_base[n*n+n+j].
            for (uint32_t j = 0; j < n; ++j) {
                const float mix = total[n * n + n + j] * coefficient_rstd;
                output[j] = 2.0f * spark_sigmoid(mix * hc_scale[1] + hc_base[n * n + n + j]);
            }
            for (uint32_t destination = 0; destination < n; ++destination) {
                for (uint32_t source = 0; source < n; ++source) {
                    hc_coeffs[destination * n + source] = mixing[source][destination];
                }
                hc_coeffs[n * n + destination] = input[destination];
                hc_coeffs[n * n + n + destination] = output[destination];
            }
            // The RMS pre-norm of X_hat_l over the hidden dimension, with the released norm_eps.
            const float mean_square = total[SPARK_M4_HC_PREDICTOR_ROWS] / (float)d;
            *hc_rms_out = 1.0f / spark_sqrt(mean_square + (float)SPARK_M4_NORM_EPS);
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
            xhat_normed[index + k] = value;
            const float magnitude = value < 0.0f ? -value : value;
            amax = magnitude > amax ? magnitude : amax;
        }
        // X_hat^FP8 is an ACTIVATION operand, so its scale is the reference's act_quant rule
        // (smallest power of two at or above amax/448 after the 1e-4 floor), never the checkpoint
        // weight-grid rule: the engine quantises this same vector with quant::quantize_activation_fp8.
        const uint8_t scale = spark_act_quant_scale_code(amax);
        xhat_scales[index / SPARK_M4_QUANT_BLOCK] = scale;
        const float inverse_scale = 1.0f / spark_ue8m0_decode(scale);
        for (uint32_t k = 0; k < SPARK_M4_QUANT_BLOCK; ++k) {
            const float value = s_xhat[element + k] * rms_weight[index + k] * inverse_rms;
            xhat_fp8[index + k] =
                failed != 0u ? (uint8_t)0x7Fu : spark_fp8_e4m3_encode(value * inverse_scale);
        }
    }

    if (tid == 0u) {
        // payload_meta: the bytes this layer writes per token - X_l (n*d*2), the quantized block
        // input (d) and its scales (d/32), and the f32 pre-normed block input (d*4). The trace ring
        // is the only place that number is reported.
        const uint32_t meta =
            (uint32_t)(n * d * 2u + d + d / SPARK_M4_QUANT_BLOCK + d * 4u);
        const uint32_t reported = failed != 0u ? (meta | 0x80000000u) : meta;
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_MEGA_MHC, layer_id, reported, t_enter);
        // The trace macros expand to nothing when the scope mask is compiled out, so keep these
        // reads alive explicitly instead of depending on the macro's expansion.
        (void)reported;
        (void)step_seq;
    }
    (void)t_enter;
}

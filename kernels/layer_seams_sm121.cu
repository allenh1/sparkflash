// kernels/layer_seams_sm121.cu
//
// The eight device kernels of the whole-layer device port's seam image
// (docs/archive/build/ws-whole-layer-device-port-20261022.md,
// task/layer-seams), in the contract order of crates/sparkflash-cli/src/layer_device.rs:
//
//   1. mhc_entry_bf16  - the mHC entry of one token: the released [2n + n^2, n*d] coefficient
//                        projection over the four stream copies, the coefficient tail (rsqrt,
//                        sigmoid pre/post, split Sinkhorn comb), the hc_pre collapse and the input
//                        RMS pre-norm. Replaces the host seam that ran hc_projection_transposed /
//                        hc_mixes / hc_transpose_predictor plus hc_pre and rms_norm on the host.
//   2. mhc_ffn_seams   - the FFN entry seams of one token: the same projection and tail on the FFN
//                        predictor, hc_pre and the FFN RMSNorm, and the activation quantiser inside
//                        the same launch, so the FFN's E4M3 code row never exists on the host.
//   3. w8a8_prologue_e4m3 - the attention projection's activation quantiser as a device pass over
//                        the normed row, so no host quantise sits in front of the projections.
//   4. mhc_attn_residual - the attention sublayer's hc_post: the reduced attention delta mixed back
//                        onto the hc stream with the entry's post mixes and Sinkhorn comb matrix.
//   5. mhc_ffn_epilogue - the shared expert added onto the routed MoE delta and the FFN's hc_post
//                        onto the post-attention stream, in one launch.
//
// ARITHMETIC ORDER. The reference is the engine's host path, function by function:
//   projection       crates/sparkflash-engine/src/forward/mhc.rs:250-300 (hc_projection): one
//                    accumulator per coefficient row, ascending k, one f32 rounding per multiply
//                    and one per add (no FMA, no reassociation).
//   coefficient tail mhc.rs:430-480 (hc_mixes_from_projection): mean_square = sum_squares / width,
//                    rsqrt = 1 / sqrt(mean_square + norm_eps), mix = raw * rsqrt;
//                    pre = sigmoid(mix * scale[0] + base[..hc]) + hc_eps;
//                    post = 2 * sigmoid(mix * scale[1] + base[hc..2hc]);
//                    comb = mix * scale[2] + base[2hc..2hc + hc^2].
//   Sinkhorn         mhc.rs:590-608 (split_sinkhorn) with linalg.rs:1136-1155 (softmax) and
//                    mhc.rs normalise_columns: row softmax, then + hc_eps, then a column pass, then
//                    hc_sinkhorn_iters - 1 pairs of a row pass (denominator row_sum + hc_eps) and a
//                    column pass (denominator column_sum + hc_eps).
//   hc_pre           mhc.rs:622-664 (hc_pre): out[j] = sum_c pre_mix[c] * x[c*dim + j], ascending c.
//   hc_post          mhc.rs:690-760 (hc_post): out[c][j] = post[c] * x[j] +
//                    sum_source comb[source][c] * residual[source][j], the comb terms ascending
//                    source; mhc.rs:823-852 (hc_post_bf16) accumulates that sum from zero, adds the
//                    post term last and rounds once. No pre-mix term exists in either.
//   RMS pre-norm     linalg.rs:1103-1130 (rms_norm): mean = sum(x*2) / cols,
//                    inv = 1 / sqrt(mean + norm_eps), out = (x * inv) * weight.
//   BF16 boundaries  quant.rs:157-171 (bf16_round): round-to-nearest-even f32 -> bf16 -> f32,
//                    NaN kept quiet. Applied where the released block rounds: the hc_pre row, the
//                    RMSNorm row and, for mhc_ffn_seams, the quantiser's input row (the caller's
//                    stored seam row is already rounded), when native_bf16 is set.
//   quantiser        m4_activation_quant_sm121.cuh (the activation ceil rule) plus
//                    quant.rs:469-500/763-800: one ue8m0 scale per 32 values from the block amax
//                    floored at 1e-4, then E4M3 round-to-nearest-even with saturation. The per-block
//                    amax is a maximum, so its reduction order cannot change the code; the
//                    per-element encoder is the same RNE rule the engine's bisection implements.
//
// TOLERANCE CLASS (D54). The per-element expressions above are the engine's, spelled with the
// rounding intrinsics so nvcc cannot contract them. What this kernel changes is the REDUCTION
// ORDER of the sums: the 24 dot products, the projection's sum of squares and the RMSNorm's sum of
// squares are split across the block's lanes and combined as a tree (see PARALLELISM). f32 addition
// is not associative, so those sums can differ from the engine's sequential ones in the last bits;
// the declared comparison class is therefore the D54 tolerance arm, not bit identity:
//   * coefficient tables (pre/post/comb): max abs delta <= 1e-4 (the bar the mHC device slice and
//     mega_mhc_ref already declare for the same coefficient tail);
//   * the normed/seam rows and 1/rms: last-bits agreement, max abs delta <= 3e-4 in the 1/rms
//     factor, which is mega_mhc_ref's declared bar;
//   * the E4M3 codes and ue8m0 scales: exact (a maximum and an order-independent function of it).
// The host twin that carries this class is crates/sparkflash-kernels/src/layer_seams_ref.rs; its
// tests pin the twin to the engine's own hmhc functions bit for bit, so the class above is the
// whole distance between the device arm and the host arm.
//
// PARALLELISM. One block of 256 threads owns the token. The 24 coefficient rows are spread over the
// block's eight warps (three rows per warp); inside a row every lane walks one 32-wide column stripe
// and the warp reduces its 32 lanes as a tree, which is what turns the released model's 20,480-long
// dependent add chain per row into ~640 iterations per lane with three independent accumulators
// (the three rows). The two sums of squares and the row reductions use the same block tree. The
// coefficient tail runs in one thread (it is 24 sigmoids and a 4x4 Sinkhorn) and broadcasts its pre
// mix through shared memory.
//
// SHARED MEMORY. The hc_pre row is kept in 20 KiB of shared memory between the collapse and the
// norm (it is the block input the norm consumes), and the coefficient scratch is a few floats, so
// the launch needs no dynamic shared-memory argument (the driver's launch ABI has none).
//
// TRACE. The signatures carry the trace ring, layer id and step sequence for ABI uniformity with the
// other kernels. The steady-state loop passes no ring at all, so the contract is that a null ring
// writes NOTHING: every store through the ring sits inside an explicit `if (trace_ring != nullptr)`
// guard, and the pin test in crates/sparkflash-kernels/src/layer_seams_launch.rs refuses a kernel
// whose guard disappears. These seams have no scope bit of their own yet, so an armed ring gets the
// parent scope's record: MEGA_MHC for the four mHC seam kernels and DENSE_MXFP8 for the prologue;
// shared_swiglu_f32 carries no ring at all, because it is not a layer seam - it is the shared
// expert's activation and it is launched by moe_shared.
// whose quantised row is the dense projections' input. A record's `payload_meta` word carries the
// launch's STAGE CODE (the SPARK_SEAM_STAGE_* macros below), which is what lets a reader attribute
// a record to a stage even when four stages share a scope: MEGA_MHC alone names no stage. The Rust
// side (crates/sparkflash-kernels/src/device_trace.rs) decodes the codes and pins them against the
// macros here; the layer loop's per-stage counter stays the primary observation, and a trace record
// places the launch in a window.

// This image compiles the in-graph trace hooks IN, for every build, unlike the other baseline
// kernels (the shipped cubins of routed_gate_up_launch/router_launch are trace-disabled builds and
// say so). The device arm's per-stage timing is the trace ring, and it must be readable from the
// pinned image: if the hooks compiled out, arming the ring would silently record nothing. The cost
// is two %globaltimer reads and one 32-byte store per launch, inside the `trace_ring != nullptr`
// guard, which is zero when the steady-state loop passes no ring. The macro must be defined before
// the first include: trace_globaltimer_sm121.cuh branches on it at include time.
#define SPARKFLASH_ENABLE_GPU_TRACE 1
#include "m4_activation_quant_sm121.cuh"

// Per-launch stage codes for the trace record's `payload_meta` word. The Rust mirror is
// crates/sparkflash-kernels/src/device_trace.rs (STAGE_*), and its test pins both halves to this
// block: one macro, one stage, one EXIT call carrying it.
#define SPARK_SEAM_STAGE_MHC_ENTRY 1U
#define SPARK_SEAM_STAGE_FFN_SEAMS 2U
#define SPARK_SEAM_STAGE_PROLOGUE 3U
#define SPARK_SEAM_STAGE_PROLOGUE_BF16 4U
#define SPARK_SEAM_STAGE_ATTN_RESIDUAL 5U
#define SPARK_SEAM_STAGE_FFN_EPILOGUE 6U
#define SPARK_SEAM_STAGE_ROPE_ROWS 7U


#define SPARK_SEAM_THREADS 256u
#define SPARK_SEAM_WARPS (SPARK_SEAM_THREADS / 32u)
/// The released coefficient-row count (2 * n + n^2 for n = 4); the launcher refuses any other.
#define SPARK_SEAM_ROWS SPARK_M4_HC_PREDICTOR_ROWS
/// The rows one warp owns: SPARK_SEAM_ROWS over SPARK_SEAM_WARPS.
#define SPARK_SEAM_ROWS_PER_WARP (SPARK_SEAM_ROWS / SPARK_SEAM_WARPS)
static_assert(SPARK_SEAM_ROWS % SPARK_SEAM_WARPS == 0u,
              "the coefficient rows must split evenly over the block's warps");
static_assert(SPARK_M4_HIDDEN % SPARK_SEAM_THREADS == 0u,
              "the row loops stride the block size into the hidden width");

// The released BF16 boundary: round to nearest even, NaN kept quiet (quant.rs:157-171).
__device__ __forceinline__ float spark_seam_bf16_round(float value) {
    const uint32_t bits = spark_float_bits(value);
    if (value != value) {
        return __uint_as_float((((bits >> 16) | 0x0040u)) << 16);
    }
    const uint32_t round = 0x7FFFu + ((bits >> 16) & 1u);
    return __uint_as_float(((bits + round) >> 16) << 16);
}

// linalg::sigmoid: 1 / (1 + exp(-x)), with the negative branch written the reference's way so the
// exponential cannot overflow (linalg.rs:1160-1167).
__device__ __forceinline__ float spark_seam_sigmoid(float value) {
    if (value >= 0.0f) {
        return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-value)));
    }
    const float exponential = expf(value);
    return __fdiv_rn(exponential, __fadd_rn(1.0f, exponential));
}

// One warp's tree reduction of 32 lane values.
__device__ __forceinline__ float spark_seam_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(value, __shfl_down_sync(0xFFFFFFFFu, value, offset));
    }
    return value;
}

// The whole block's tree reduction, broadcast to every thread through the shared scratch slot.
__device__ __forceinline__ float spark_seam_block_sum(float value, float *scratch) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    value = spark_seam_warp_sum(value);
    if (lane == 0u) {
        scratch[warp] = value;
    }
    __syncthreads();
    if (warp == 0u) {
        float total = lane < SPARK_SEAM_WARPS ? scratch[lane] : 0.0f;
        total = spark_seam_warp_sum(total);
        if (lane == 0u) {
            scratch[0] = total;
        }
    }
    __syncthreads();
    return scratch[0];
}

// The column normalisation of the Sinkhorn pair (mhc.rs normalise_columns): hc_eps inside every
// denominator.
__device__ __forceinline__ void spark_seam_normalise_columns(float comb[SPARK_M4_HC_MULT]
                                                                  [SPARK_M4_HC_MULT],
                                                             float hc_eps) {
    for (uint32_t column = 0; column < SPARK_M4_HC_MULT; ++column) {
        float sum = hc_eps;
        for (uint32_t row = 0; row < SPARK_M4_HC_MULT; ++row) {
            sum = __fadd_rn(sum, comb[row][column]);
        }
        const float inverse = __fdiv_rn(1.0f, sum);
        for (uint32_t row = 0; row < SPARK_M4_HC_MULT; ++row) {
            comb[row][column] = __fmul_rn(comb[row][column], inverse);
        }
    }
}

// The released split Sinkhorn (mhc.rs:590-608): row softmax, + hc_eps, a column pass, then
// hc_sinkhorn_iters - 1 row/column pairs. The softmax is linalg::softmax on the hc-wide row.
__device__ __forceinline__ void spark_seam_split_sinkhorn(float comb[SPARK_M4_HC_MULT]
                                                                [SPARK_M4_HC_MULT],
                                                          uint32_t iterations, float hc_eps) {
    for (uint32_t row = 0; row < SPARK_M4_HC_MULT; ++row) {
        float maximum = comb[row][0];
        for (uint32_t column = 1; column < SPARK_M4_HC_MULT; ++column) {
            maximum = comb[row][column] > maximum ? comb[row][column] : maximum;
        }
        float total = 0.0f;
        for (uint32_t column = 0; column < SPARK_M4_HC_MULT; ++column) {
            const float exponential = expf(__fadd_rn(comb[row][column], -maximum));
            comb[row][column] = exponential;
            total = __fadd_rn(total, exponential);
        }
        for (uint32_t column = 0; column < SPARK_M4_HC_MULT; ++column) {
            comb[row][column] = __fadd_rn(__fdiv_rn(comb[row][column], total), hc_eps);
        }
    }
    spark_seam_normalise_columns(comb, hc_eps);
    for (uint32_t iteration = 1; iteration < iterations; ++iteration) {
        for (uint32_t row = 0; row < SPARK_M4_HC_MULT; ++row) {
            float sum = 0.0f;
            for (uint32_t column = 0; column < SPARK_M4_HC_MULT; ++column) {
                sum = __fadd_rn(sum, comb[row][column]);
            }
            const float inverse = __fdiv_rn(1.0f, __fadd_rn(sum, hc_eps));
            for (uint32_t column = 0; column < SPARK_M4_HC_MULT; ++column) {
                comb[row][column] = __fmul_rn(comb[row][column], inverse);
            }
        }
        spark_seam_normalise_columns(comb, hc_eps);
    }
}

// The 24 raw dot products and the projection's sum of squares, in the engine's per-element order
// and the block's tree reduction order. raw[row] trails through the shared row table; return is the
// flattened-stream sum of squares (warp 0 covers the whole width, so no other warp's copy is
// summed).
__device__ __forceinline__ float spark_seam_project(
    const float *__restrict__ stream,  // [SPARK_M4_HC_MULT * SPARK_M4_HIDDEN]
    const float *__restrict__ fn_w,    // [SPARK_SEAM_ROWS][SPARK_M4_HC_MULT * SPARK_M4_HIDDEN]
    float *raw_smem,                   // [SPARK_SEAM_ROWS]
    float *scratch)                    // [32]
{
    const uint32_t width = SPARK_M4_HC_MULT * SPARK_M4_HIDDEN;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    float raw[SPARK_SEAM_ROWS_PER_WARP] = {0.0f, 0.0f, 0.0f};
    float sum_squares = 0.0f;
    const uint32_t iterations = width / 32u;
    for (uint32_t stripe = 0; stripe < iterations; ++stripe) {
        const uint32_t column = stripe * 32u + lane;
        const float x = stream[column];
#pragma unroll
        for (uint32_t slot = 0; slot < SPARK_SEAM_ROWS_PER_WARP; ++slot) {
            const uint32_t row = warp + slot * SPARK_SEAM_WARPS;
            // The engine's project(): one multiply, one add, ascending k (mhc.rs:216-232).
            raw[slot] = __fadd_rn(__fmul_rn(fn_w[(uint64_t)row * width + column], x), raw[slot]);
        }
        sum_squares = __fadd_rn(sum_squares, __fmul_rn(x, x));
    }
#pragma unroll
    for (uint32_t slot = 0; slot < SPARK_SEAM_ROWS_PER_WARP; ++slot) {
        const uint32_t row = warp + slot * SPARK_SEAM_WARPS;
        const float reduced = spark_seam_warp_sum(raw[slot]);
        if (lane == 0u) {
            raw_smem[row] = reduced;
        }
    }
    if (warp == 0u) {
        const float reduced = spark_seam_warp_sum(sum_squares);
        if (lane == 0u) {
            scratch[31] = reduced;
        }
    }
    __syncthreads();
    return scratch[31];
}

// The coefficient tail. Thread 0 alone computes the 24 coefficients and writes the three tables; the
// pre set is the table's own pre region (mix_table[0..hc]), so there is no second output to alias and
// no cross-lane read of a value another lane wrote. (The earlier form broadcast the pre set through
// a shared float[4] and read it back under a lane guard with no barrier between the write and the
// read: lanes 1-3 could read uninitialised shared memory, and the entry passed the same span as
// mix_table and pre_mix, so the stale lane would have corrupted exactly the four values the FFN seam
// latches as its hc_pre input. The entry no longer takes a pre_mix pointer at all - its pre set is
// the mix table's pre region - so the two aliased __restrict__ parameters are gone with the race.)
__device__ __forceinline__ void spark_seam_coefficient_tail(
    const float *raw_smem,       // [SPARK_SEAM_ROWS]
    float sum_squares,           // flattened-stream sum of squares
    const float *hc_scale,       // [3]
    const float *hc_base,        // [SPARK_SEAM_ROWS]
    float norm_eps,
    float hc_eps,
    uint32_t sinkhorn_iterations,
    float *mix_table)            // [SPARK_SEAM_ROWS] out: the pre, post, comb tables
{
    if (threadIdx.x != 0u) {
        return;
    }
    const uint32_t hc = SPARK_M4_HC_MULT;
    const uint32_t width = hc * SPARK_M4_HIDDEN;
    const float mean_square = __fdiv_rn(sum_squares, (float)width);
    const float rsqrt = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean_square, norm_eps)));
    float mix[SPARK_SEAM_ROWS];
#pragma unroll
    for (uint32_t row = 0; row < SPARK_SEAM_ROWS; ++row) {
        mix[row] = __fmul_rn(raw_smem[row], rsqrt);
    }
    for (uint32_t coefficient = 0; coefficient < hc; ++coefficient) {
        const float pre = __fadd_rn(
            spark_seam_sigmoid(__fadd_rn(
                __fmul_rn(mix[coefficient], hc_scale[0]), hc_base[coefficient])),
            hc_eps);
        mix_table[coefficient] = pre;
    }
    for (uint32_t coefficient = 0; coefficient < hc; ++coefficient) {
        const uint32_t row = hc + coefficient;
        mix_table[row] = __fmul_rn(
            2.0f,
            spark_seam_sigmoid(__fadd_rn(__fmul_rn(mix[row], hc_scale[1]), hc_base[row])));
    }
    float comb[SPARK_M4_HC_MULT][SPARK_M4_HC_MULT];
    for (uint32_t row = 0; row < hc; ++row) {
        for (uint32_t column = 0; column < hc; ++column) {
            const uint32_t coefficient = 2u * hc + row * hc + column;
            comb[row][column] =
                __fadd_rn(__fmul_rn(mix[coefficient], hc_scale[2]), hc_base[coefficient]);
        }
    }
    spark_seam_split_sinkhorn(comb, sinkhorn_iterations, hc_eps);
    for (uint32_t row = 0; row < hc; ++row) {
        for (uint32_t column = 0; column < hc; ++column) {
            mix_table[2u * hc + row * hc + column] = comb[row][column];
        }
    }
}

// hc_pre and the RMS pre-norm of one token: shared row -> normed, with the released BF16 boundaries
// when native_bf16 is set. The pre mix it reads is the INPUT set (the previous sublayer's), latched
// by the caller; the stream is read from global.
//
// x_bf16, when not null, is the FFN block's input AFTER its RMS pre-norm, narrowed to BF16 with
// RNE. That is the host arm's pre_ffn_hidden (rms_norm output, whose router input is its BF16 RNE
// round - docs/archive/build/moe-stage-contract.md H3b), NOT the pre-norm hc_pre collapse: the router routes
// on the normalized hidden. The write therefore lives in the norm loop, beside normed[], so the row
// cannot be produced before the value it narrows exists. A null pointer writes nothing (the
// attention entry has no consumer for it).
__device__ __forceinline__ void spark_seam_hc_pre_and_norm(
    const float *__restrict__ stream,
    const float *in_pre,
    const uint8_t *__restrict__ norm_w,
    uint32_t norm_bf16,
    uint32_t native_bf16,
    float norm_eps,
    float *row_smem,          // [SPARK_M4_HIDDEN]
    __nv_bfloat16 *x_bf16,    // [SPARK_M4_HIDDEN] optional: the normed row, RNE-narrowed
    float *normed,            // [SPARK_M4_HIDDEN]
    float *scratch)           // [32]
{
    const uint32_t dim = SPARK_M4_HIDDEN;
    float sum_squares = 0.0f;
    for (uint32_t column = threadIdx.x; column < dim; column += SPARK_SEAM_THREADS) {
        float accumulated = 0.0f;
        for (uint32_t copy = 0; copy < SPARK_M4_HC_MULT; ++copy) {
            accumulated = __fadd_rn(
                __fmul_rn(in_pre[copy], stream[(uint64_t)copy * dim + column]),
                accumulated);
        }
        if (native_bf16 != 0u) {
            accumulated = spark_seam_bf16_round(accumulated);
        }
        row_smem[column] = accumulated;
        sum_squares = __fadd_rn(sum_squares, __fmul_rn(accumulated, accumulated));
    }
    const float total = spark_seam_block_sum(sum_squares, scratch);
    const float mean = __fdiv_rn(total, (float)dim);
    const float inverse = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, norm_eps)));
    for (uint32_t column = threadIdx.x; column < dim; column += SPARK_SEAM_THREADS) {
        const float weight = norm_bf16 != 0u
                                 ? __bfloat162float(
                                       reinterpret_cast<const __nv_bfloat16 *>(norm_w)[column])
                                 : reinterpret_cast<const float *>(norm_w)[column];
        float value = __fmul_rn(__fmul_rn(row_smem[column], inverse), weight);
        if (native_bf16 != 0u) {
            value = spark_seam_bf16_round(value);
        }
        normed[column] = value;
        if (x_bf16 != nullptr) {
            // The host arm's pre_ffn_hidden is this value in f32 and the router consumes its BF16
            // RNE round. Under native_bf16 the released boundary already made it bf16-representable
            // (the narrowing is idempotent), so one rule covers both arms.
            x_bf16[column] = __float2bfloat16_rn(value);
        }
    }
}

// The activation quantiser's per-block rule over one stored row: one ue8m0 code from the block amax
// and one E4M3 code per value (kernel.py rule; see the header comment).
__device__ __forceinline__ void spark_seam_quantise_row(
    const float *__restrict__ row,
    uint8_t *__restrict__ codes,
    uint8_t *__restrict__ scales,
    uint32_t dim) {
    const uint32_t blocks = dim / SPARK_M4_QUANT_BLOCK;
    for (uint32_t block = threadIdx.x; block < blocks; block += SPARK_SEAM_THREADS) {
        const float *source = row + (uint64_t)block * SPARK_M4_QUANT_BLOCK;
        float amax = 0.0f;
        for (uint32_t index = 0; index < SPARK_M4_QUANT_BLOCK; ++index) {
            const float magnitude = source[index] < 0.0f ? -source[index] : source[index];
            amax = magnitude > amax ? magnitude : amax;
        }
        const uint8_t scale = spark_act_quant_scale_code(amax);
        const float inverse = __fdiv_rn(1.0f, spark_ue8m0_decode(scale));
        scales[block] = scale;
        for (uint32_t index = 0; index < SPARK_M4_QUANT_BLOCK; ++index) {
            codes[(uint64_t)block * SPARK_M4_QUANT_BLOCK + index] =
                spark_fp8_e4m3_encode(__fmul_rn(source[index], inverse));
        }
    }
}

// The released geometry the seam kernels are compiled for; the launcher validates the arguments
// against it and a mismatched launch writes nothing rather than a wrong row.
__device__ __forceinline__ bool spark_seam_geometry_ok(uint32_t dim, uint32_t hc_mult,
                                                       uint32_t rows) {
    return dim == SPARK_M4_HIDDEN && hc_mult == SPARK_M4_HC_MULT && rows == SPARK_SEAM_ROWS;
}

// ---------------------------------------------------------------------------------------------
// 1. mhc_entry_bf16: the layer's mHC entry.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void mhc_entry_bf16(
    const float *__restrict__ residual,      // [hc*dim] the hc stream the layer reads
    const float *__restrict__ attn_pre_mix,  // [hc] previous layer's FFN pre mix / identity seed
    const float *__restrict__ fn_w,          // [rows][hc*dim] the released attn predictor
    const float *__restrict__ hc_base,       // [rows]
    const float *__restrict__ hc_scale,      // [3] pre, post, comb
    const uint8_t *__restrict__ norm_w,      // [dim] attn_norm weight (f32 or bf16 per norm_bf16)
    float *__restrict__ mix_table,           // [rows] out: pre, post, comb
    float *__restrict__ hc_rows,             // [hc*dim] out: the entry's read of the stream
    float *__restrict__ normed,              // [dim] out: the normalized block input
    const uint32_t dim,
    const uint32_t hc_mult,
    const uint32_t rows,
    const uint32_t norm_bf16,
    const uint32_t native_bf16,
    const uint32_t sinkhorn_iterations,
    const float norm_eps,
    const float hc_eps,
    GpuTraceRing *__restrict__ trace_ring,   // guarded: null means write nothing (see TRACE)
    const uint32_t layer_id,
    const uint64_t step_seq) {
    // TRACE. No store through the ring runs before this guard, so a null ring (the steady state:
    // the loop passes no evidence ring) writes nothing at all; the layer-seam pin test refuses a
    // kernel whose guard disappears. A ring that IS passed gets one scope record per launch.
    uint64_t trace_enter = 0U;
    if (trace_ring != nullptr) {
        SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_MEGA_MHC, trace_enter);
    }
    (void)step_seq;
    if (!spark_seam_geometry_ok(dim, hc_mult, rows)) {
        return;
    }
    __shared__ float row_smem[SPARK_M4_HIDDEN];
    __shared__ float in_pre_smem[SPARK_M4_HC_MULT];
    __shared__ float raw_smem[SPARK_SEAM_ROWS];
    __shared__ float scratch[32];

    // hc_pre collapses the stream with the INPUT pre mix - the previous sublayer's pre set (the
    // identity seed at position start) - while this seam's own pre set is what the FFN block's
    // hc_pre will consume (model.py:954-966: hc_pre(pre_mix_in) at the attention seam and
    // hc_pre(attn pre) at the FFN). The two sets are different, so the input is latched into its
    // own shared buffer before the coefficient tail overwrites the working one, and the caller may
    // still hand the same span as the pre-mix output.
    if (threadIdx.x < SPARK_M4_HC_MULT) {
        in_pre_smem[threadIdx.x] = attn_pre_mix[threadIdx.x];
    }
    __syncthreads();

    const float sum_squares = spark_seam_project(residual, fn_w, raw_smem, scratch);
    __syncthreads();
    spark_seam_coefficient_tail(raw_smem, sum_squares, hc_scale, hc_base, norm_eps, hc_eps,
                                sinkhorn_iterations, mix_table);
    __syncthreads();

    spark_seam_hc_pre_and_norm(residual, in_pre_smem, norm_w, norm_bf16, native_bf16, norm_eps,
                               row_smem, nullptr, normed, scratch);

    const uint32_t width = SPARK_M4_HC_MULT * SPARK_M4_HIDDEN;
    for (uint32_t element = threadIdx.x; element < width; element += SPARK_SEAM_THREADS) {
        hc_rows[element] = residual[element];
    }
    if (trace_ring != nullptr) {
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_MEGA_MHC, layer_id, SPARK_SEAM_STAGE_MHC_ENTRY, trace_enter);
    }
}

// ---------------------------------------------------------------------------------------------
// 2. mhc_ffn_seams: the FFN's mHC entry, its RMSNorm and its activation quantiser.
// ---------------------------------------------------------------------------------------------
// The x_bf16 out word is the host arm's pre_ffn_hidden: the FFN block's RMS-normed input in f32,
// narrowed to BF16 with RNE - the device router's operand (docs/archive/build/moe-stage-contract.md H3b).
// It is NOT the pre-norm hc_pre collapse; the norm loop writes it beside seam_row, which is the same
// value. Under native_bf16 the released boundary already made the normed value bf16-representable,
// so the narrowing is idempotent and one rule covers both arms. It is the ABI's one optional
// pointer: null writes nothing, so a launcher can pass a null word until the span exists.
extern "C" __global__ void mhc_ffn_seams(
    const float *__restrict__ stream,        // [hc*dim] the post-attention hc stream
    const float *__restrict__ attn_pre_mix,  // [hc] the entry's pre mix (this block's hc_pre)
    const float *__restrict__ fn_w,          // [rows][hc*dim] the released FFN predictor
    const float *__restrict__ hc_base,       // [rows]
    const float *__restrict__ hc_scale,      // [3]
    const uint8_t *__restrict__ norm_w,      // [dim] ffn_norm weight (f32 or bf16)
    float *__restrict__ mix_table,           // [rows] out: the FFN pre, post, comb
    float *__restrict__ pre_mix,             // [hc] out: the FFN pre mix (the next layer's input)
    float *__restrict__ seam_row,            // [dim] out: the FFN block input, normalized
    uint8_t *__restrict__ act_codes,         // [dim] out: the FFN activation E4M3 row
    uint8_t *__restrict__ act_scales,        // [dim / 32] out: its ue8m0 block scales
    __nv_bfloat16 *__restrict__ x_bf16,      // [dim] optional out: the normed FFN input row, RNE
                                             // narrowed (the router's input; null writes nothing)
    const uint32_t dim,
    const uint32_t hc_mult,
    const uint32_t rows,
    const uint32_t norm_bf16,
    const uint32_t native_bf16,
    const uint32_t sinkhorn_iterations,
    const float norm_eps,
    const float hc_eps,
    GpuTraceRing *__restrict__ trace_ring,   // guarded: null means write nothing (see TRACE)
    const uint32_t layer_id,
    const uint64_t step_seq) {
    // TRACE. No store through the ring runs before this guard, so a null ring (the steady state:
    // the loop passes no evidence ring) writes nothing at all; the layer-seam pin test refuses a
    // kernel whose guard disappears. A ring that IS passed gets one scope record per launch.
    uint64_t trace_enter = 0U;
    if (trace_ring != nullptr) {
        SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_MEGA_MHC, trace_enter);
    }
    (void)step_seq;
    if (!spark_seam_geometry_ok(dim, hc_mult, rows)) {
        return;
    }
    __shared__ float row_smem[SPARK_M4_HIDDEN];
    __shared__ float in_pre_smem[SPARK_M4_HC_MULT];
    __shared__ float raw_smem[SPARK_SEAM_ROWS];
    __shared__ float scratch[32];

    // The FFN block's hc_pre consumes the attention seam's pre set (the INPUT here), and this
    // seam's own pre set is the next layer's input: the reference's two different sets, so the
    // input is latched before the coefficient tail overwrites the working buffer.
    if (threadIdx.x < SPARK_M4_HC_MULT) {
        in_pre_smem[threadIdx.x] = attn_pre_mix[threadIdx.x];
    }
    __syncthreads();

    const float sum_squares = spark_seam_project(stream, fn_w, raw_smem, scratch);
    __syncthreads();
    spark_seam_coefficient_tail(raw_smem, sum_squares, hc_scale, hc_base, norm_eps, hc_eps,
                                sinkhorn_iterations, mix_table);
    // The next layer's hc_pre input is this seam's pre set: the table's pre region, copied by the
    // same thread that wrote it, so no shared broadcast and no cross-lane read exist.
    if (threadIdx.x == 0u) {
        for (uint32_t copy = 0; copy < SPARK_M4_HC_MULT; ++copy) {
            pre_mix[copy] = mix_table[copy];
        }
    }
    __syncthreads();

    spark_seam_hc_pre_and_norm(stream, in_pre_smem, norm_w, norm_bf16, native_bf16, norm_eps,
                               row_smem, x_bf16, seam_row, scratch);

    __syncthreads();
    spark_seam_quantise_row(seam_row, act_codes, act_scales, SPARK_M4_HIDDEN);
    if (trace_ring != nullptr) {
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_MEGA_MHC, layer_id, SPARK_SEAM_STAGE_FFN_SEAMS, trace_enter);
    }
}

// ---------------------------------------------------------------------------------------------
// 3. w8a8_prologue_e4m3: the attention projection's activation quantiser as a device pass.
// ---------------------------------------------------------------------------------------------
// One MX block per thread: the block's 32 values, its ue8m0 scale from the floored amax, and 32 E4M3
// codes. The grid is ceil(rows * cols / 32 / 256) blocks of 256 threads, so a 5,120-wide row is 160
// independent lanes inside ONE block (the launcher's grid, not a block per MX block); the launch is
// tiny either way and the semantics do not depend on the split because the block value set is fixed
// by the layout.
// The BF16-input variant of the quantiser. The host's q_norm output is a bf16 tensor
// (attention.rs:1318-1341), so the reference quantises the values it sees *after* that rounding:
// widening the bf16 row here is the reference's operand, while quantising an unrounded f32 row
// would encode values the reference never had. Same MX-block arithmetic as the f32 variant, one
// widening per read. (It exists because query_path passes the bf16 q_normed span: reading it as f32
// took 5,120 B out of a 2,560 B span and encoded wrong bytes.)
extern "C" __global__ void w8a8_prologue_bf16_e4m3(
    const __nv_bfloat16 *__restrict__ normed,  // [rows][cols] the bf16 activation row
    uint8_t *__restrict__ codes,        // [rows][cols] out
    uint8_t *__restrict__ scales,       // [rows][cols / 32] out
    const uint32_t cols,
    const uint32_t rows,
    GpuTraceRing *__restrict__ trace_ring,  // guarded: null means write nothing (see TRACE)
    const uint32_t layer_id,
    const uint64_t step_seq) {
    uint64_t trace_enter = 0U;
    if (trace_ring != nullptr) {
        SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, trace_enter);
    }
    (void)step_seq;
    const uint32_t blocks_per_row = cols / SPARK_M4_QUANT_BLOCK;
    const uint32_t total = blocks_per_row * rows;
    const uint32_t block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= total) {
        return;
    }
    const uint32_t row = block / blocks_per_row;
    const uint32_t within = block % blocks_per_row;
    const __nv_bfloat16 *source = normed + (uint64_t)row * cols + (uint64_t)within * SPARK_M4_QUANT_BLOCK;
    float amax = 0.0f;
    for (uint32_t index = 0; index < SPARK_M4_QUANT_BLOCK; ++index) {
        const float value = __bfloat162float(source[index]);
        const float magnitude = value < 0.0f ? -value : value;
        amax = magnitude > amax ? magnitude : amax;
    }
    const uint8_t scale = spark_act_quant_scale_code(amax);
    const float inverse = __fdiv_rn(1.0f, spark_ue8m0_decode(scale));
    scales[(uint64_t)row * blocks_per_row + within] = scale;
    uint8_t *destination = codes + (uint64_t)row * cols + (uint64_t)within * SPARK_M4_QUANT_BLOCK;
    for (uint32_t index = 0; index < SPARK_M4_QUANT_BLOCK; ++index) {
        destination[index] = spark_fp8_e4m3_encode(
            __fmul_rn(__bfloat162float(source[index]), inverse));
    }
    if (trace_ring != nullptr) {
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, SPARK_SEAM_STAGE_PROLOGUE, trace_enter);
    }
}

extern "C" __global__ void w8a8_prologue_e4m3(
    const float *__restrict__ normed,   // [rows][cols] the f32 activation row
    uint8_t *__restrict__ codes,        // [rows][cols] out
    uint8_t *__restrict__ scales,       // [rows][cols / 32] out
    const uint32_t cols,
    const uint32_t rows,
    GpuTraceRing *__restrict__ trace_ring,  // guarded: null means write nothing (see TRACE)
    const uint32_t layer_id,
    const uint64_t step_seq) {
    // TRACE. No store through the ring runs before this guard, so a null ring (the steady state:
    // the loop passes no evidence ring) writes nothing at all; the layer-seam pin test refuses a
    // kernel whose guard disappears. A ring that IS passed gets one scope record per launch.
    uint64_t trace_enter = 0U;
    if (trace_ring != nullptr) {
        SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, trace_enter);
    }
    (void)step_seq;
    const uint32_t blocks_per_row = cols / SPARK_M4_QUANT_BLOCK;
    const uint32_t total = blocks_per_row * rows;
    const uint32_t block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= total) {
        return;
    }
    const uint32_t row = block / blocks_per_row;
    const uint32_t within = block % blocks_per_row;
    const float *source = normed + (uint64_t)row * cols + (uint64_t)within * SPARK_M4_QUANT_BLOCK;
    float amax = 0.0f;
    for (uint32_t index = 0; index < SPARK_M4_QUANT_BLOCK; ++index) {
        const float magnitude = source[index] < 0.0f ? -source[index] : source[index];
        amax = magnitude > amax ? magnitude : amax;
    }
    const uint8_t scale = spark_act_quant_scale_code(amax);
    const float inverse = __fdiv_rn(1.0f, spark_ue8m0_decode(scale));
    scales[(uint64_t)row * blocks_per_row + within] = scale;
    uint8_t *destination = codes + (uint64_t)row * cols + (uint64_t)within * SPARK_M4_QUANT_BLOCK;
    for (uint32_t index = 0; index < SPARK_M4_QUANT_BLOCK; ++index) {
        destination[index] = spark_fp8_e4m3_encode(__fmul_rn(source[index], inverse));
    }
    if (trace_ring != nullptr) {
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, SPARK_SEAM_STAGE_PROLOGUE_BF16, trace_enter);
    }
}

// ---------------------------------------------------------------------------------------------
// 4. mhc_attn_residual: the attention sublayer's hc_post.
// ---------------------------------------------------------------------------------------------
// The released hc_post (mhc.rs:690-760): the reduced attention delta is mixed back onto the hc
// residual stream with the entry's post mixes and its Sinkhorn comb matrix,
//   out[c][j] = post[c] * delta[j] + sum_source comb[source][c] * residual[source][j],
// and the hc_post_bf16 arm (mhc.rs:823-852) is the same expression with the comb sum accumulated
// from zero, the post term added last, and the result rounded to BF16. There is no pre-mix term in
// either: hc_pre consumes a pre set, hc_post never does, so the third input here is the entry's
// MIX TABLE (post at +hc, comb at +2hc), not the pre mix the stage contract sketch called it.
//
// The residual the reference hands hc_post is the hc stream as the entry consumed it, which is the
// entry's own copy: the kernel reads residual.hc_rows and writes residual.stream, so the in-place
// update cannot race itself. hc_post mixes the four copies at a fixed j and never across j, so one
// thread reading all four sources for its j and then writing the four outputs is the whole order.
extern "C" __global__ void mhc_attn_residual(
    const float *__restrict__ hc_rows,       // [hc*dim] the hc stream the entry consumed
    const float *__restrict__ mix_table,     // [rows] the entry's pre, post, comb
    const void *__restrict__ reduced_bf16,   // [dim] bf16, the all-reduced attention delta
    float *__restrict__ stream_out,          // [hc*dim] out: the updated hc stream
    const uint32_t dim,
    const uint32_t hc_mult,
    const uint32_t rows,
    const uint32_t native_bf16,
    GpuTraceRing *__restrict__ trace_ring,   // guarded: null means write nothing (see TRACE)
    const uint32_t layer_id,
    const uint64_t step_seq) {
    // TRACE. No store through the ring runs before this guard, so a null ring (the steady state:
    // the loop passes no evidence ring) writes nothing at all; the layer-seam pin test refuses a
    // kernel whose guard disappears. A ring that IS passed gets one scope record per launch.
    uint64_t trace_enter = 0U;
    if (trace_ring != nullptr) {
        SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_MEGA_MHC, trace_enter);
    }
    (void)step_seq;
    if (!spark_seam_geometry_ok(dim, hc_mult, rows)) {
        return;
    }
    const uint32_t hc = SPARK_M4_HC_MULT;
    const __nv_bfloat16 *delta = reinterpret_cast<const __nv_bfloat16 *>(reduced_bf16);
    float post[SPARK_M4_HC_MULT];
    float comb[SPARK_M4_HC_MULT][SPARK_M4_HC_MULT];
    for (uint32_t copy = 0; copy < hc; ++copy) {
        post[copy] = mix_table[hc + copy];
    }
    for (uint32_t source = 0; source < hc; ++source) {
        for (uint32_t copy = 0; copy < hc; ++copy) {
            comb[source][copy] = mix_table[2u * hc + source * hc + copy];
        }
    }
    for (uint32_t column = threadIdx.x; column < SPARK_M4_HIDDEN; column += SPARK_SEAM_THREADS) {
        const float x = __bfloat162float(delta[column]);
        float residual[SPARK_M4_HC_MULT];
        for (uint32_t source = 0; source < hc; ++source) {
            residual[source] = hc_rows[(uint64_t)source * dim + column];
        }
        for (uint32_t copy = 0; copy < hc; ++copy) {
            float value;
            if (native_bf16 != 0u) {
                // hc_post_bf16: the comb sum from zero, the post term last, one BF16 rounding.
                float comb_sum = 0.0f;
                for (uint32_t source = 0; source < hc; ++source) {
                    comb_sum = __fadd_rn(
                        comb_sum, __fmul_rn(comb[source][copy], residual[source]));
                }
                const float post_term = __fmul_rn(post[copy], x);
                value = spark_seam_bf16_round(__fadd_rn(post_term, comb_sum));
            } else {
                // hc_post: post * x first, then the comb terms in ascending source order.
                float accumulated = __fmul_rn(post[copy], x);
                for (uint32_t source = 0; source < hc; ++source) {
                    accumulated = __fadd_rn(
                        accumulated, __fmul_rn(comb[source][copy], residual[source]));
                }
                value = accumulated;
            }
            stream_out[(uint64_t)copy * dim + column] = value;
        }
    }
    if (trace_ring != nullptr) {
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_MEGA_MHC, layer_id, SPARK_SEAM_STAGE_ATTN_RESIDUAL, trace_enter);
    }
}

// ---------------------------------------------------------------------------------------------
// 5. mhc_ffn_epilogue: the shared-expert add and the FFN's hc_post.
// ---------------------------------------------------------------------------------------------
// The released epilogue is two elementwise steps on one row: the shared expert's row is added to the
// widened routed delta in f32 (the default arm) -- or the native arm's BF16 shared addend joins the
// routed sum and the MoE output is rounded once -- and that FFN output row goes through the FFN
// seam's hc_post with the post mixes and comb matrix, like the attention residual above
// (forward/mod.rs:1950-2038). The routed delta arrives as the bf16 row the second collective wrote
// (workspace.moe_reduced_bf16); the epilogue adds no reduce of its own. The shared row is f32 in the
// default arm and bf16 under --native-rounding shared (forward/mod.rs:1948-1975), selected by
// shared_bf16. shared_present says whether the arm HAS a shared-expert row at all: the whole-layer
// port's weight set carries no shared-expert weight yet, so the layer loop passes 0 and the FFN
// output is the widened routed delta alone -- the arm's known gap, declared in the ABI rather than
// guessed from a null pointer, and one word to flip when the shared expert lands. The stream is the
// post-attention hc stream and the update is in place, safe for the same copy-wise reason as
// mhc_attn_residual.
extern "C" __global__ void mhc_ffn_epilogue(
    const float *__restrict__ stream,        // [hc*dim] the post-attention hc stream
    const void *__restrict__ moe_delta,      // [dim] bf16, the reduced routed MoE delta
    const void *__restrict__ shared_row,     // [dim] f32 or bf16, the shared expert's row
    const float *__restrict__ mix_table,     // [rows] the FFN pre, post, comb
    float *__restrict__ stream_out,          // [hc*dim] out: the next layer's hc stream
    const uint32_t dim,
    const uint32_t hc_mult,
    const uint32_t rows,
    const uint32_t shared_present,
    const uint32_t shared_bf16,
    const uint32_t native_bf16,
    GpuTraceRing *__restrict__ trace_ring,   // guarded: null means write nothing (see TRACE)
    const uint32_t layer_id,
    const uint64_t step_seq) {
    // TRACE. No store through the ring runs before this guard, so a null ring (the steady state:
    // the loop passes no evidence ring) writes nothing at all; the layer-seam pin test refuses a
    // kernel whose guard disappears. A ring that IS passed gets one scope record per launch.
    uint64_t trace_enter = 0U;
    if (trace_ring != nullptr) {
        SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_MEGA_MHC, trace_enter);
    }
    (void)step_seq;
    if (!spark_seam_geometry_ok(dim, hc_mult, rows)) {
        return;
    }
    const uint32_t hc = SPARK_M4_HC_MULT;
    const __nv_bfloat16 *delta = reinterpret_cast<const __nv_bfloat16 *>(moe_delta);
    const __nv_bfloat16 *shared_bf = reinterpret_cast<const __nv_bfloat16 *>(shared_row);
    const float *shared_f = reinterpret_cast<const float *>(shared_row);
    float post[SPARK_M4_HC_MULT];
    float comb[SPARK_M4_HC_MULT][SPARK_M4_HC_MULT];
    for (uint32_t copy = 0; copy < hc; ++copy) {
        post[copy] = mix_table[hc + copy];
    }
    for (uint32_t source = 0; source < hc; ++source) {
        for (uint32_t copy = 0; copy < hc; ++copy) {
            comb[source][copy] = mix_table[2u * hc + source * hc + copy];
        }
    }
    for (uint32_t column = threadIdx.x; column < SPARK_M4_HIDDEN; column += SPARK_SEAM_THREADS) {
        const float shared = shared_present == 0u
                                 ? 0.0f
                                 : (shared_bf16 != 0u ? __bfloat162float(shared_bf[column])
                                                      : shared_f[column]);
        // The epilogue's own add: the widened routed delta plus the shared row, in f32. With no
        // shared row in the arm the addend is a true zero, so the row stays the routed delta.
        const float ffn_out = __fadd_rn(__bfloat162float(delta[column]), shared);
        float residual[SPARK_M4_HC_MULT];
        for (uint32_t source = 0; source < hc; ++source) {
            residual[source] = stream[(uint64_t)source * dim + column];
        }
        for (uint32_t copy = 0; copy < hc; ++copy) {
            float value;
            if (native_bf16 != 0u) {
                float comb_sum = 0.0f;
                for (uint32_t source = 0; source < hc; ++source) {
                    comb_sum = __fadd_rn(
                        comb_sum, __fmul_rn(comb[source][copy], residual[source]));
                }
                const float post_term = __fmul_rn(post[copy], ffn_out);
                value = spark_seam_bf16_round(__fadd_rn(post_term, comb_sum));
            } else {
                float accumulated = __fmul_rn(post[copy], ffn_out);
                for (uint32_t source = 0; source < hc; ++source) {
                    accumulated = __fadd_rn(
                        accumulated, __fmul_rn(comb[source][copy], residual[source]));
                }
                value = accumulated;
            }
            stream_out[(uint64_t)copy * dim + column] = value;
        }
    }
    if (trace_ring != nullptr) {
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_MEGA_MHC, layer_id, SPARK_SEAM_STAGE_FFN_EPILOGUE, trace_enter);
    }
}

// ---------------------------------------------------------------------------------------------
// 6. shared_swiglu_f32: the shared expert's activation, for the default (f32-reference) arm.
// ---------------------------------------------------------------------------------------------
// The host seam's arithmetic (tp4_forward.rs:7441-7462) with the clamps where the host has them:
// when limit > 0, gate is min(gate, limit) and up is clamped to [-limit, limit]; then
//   out = silu(gate) * up,  silu(x) = x / (1 + exp(-x)).
//
// DECLARED DEVIATIONS, two of them, both inside the D54 f32-reference tolerance arm:
//   * the exponential. The host silu calls Rust's exp (glibc expf on this host); the kernel calls
//     CUDA's expf, a different implementation of the same function, not bit-identical for every
//     input.
//   * the division shape. The host silu is x * sigmoid(x) with sigmoid = 1 / (1 + exp(-x))
//     (tp4_forward.rs:7441-7462, linalg::silu at linalg.rs:1171-1173): a reciprocal rounded, then a
//     multiply rounded. This kernel divides once, x / (1 + exp(-x)) (__fdiv_rn), so even with an
//     identical exponential the two round at different points.
// This row therefore belongs to f32-reference arithmetic with a D54 tolerance, not to a
// bit-identical arm - the first by-construction difference in the default FFN arm, declared here
// rather than discovered by a logits bar.
//
// No trace: the seam kernels carry a ring because they are layer stages; this kernel is one
// activation inside moe_shared, which records the stage's record itself.
extern "C" __global__ void shared_swiglu_f32(
    const float *__restrict__ gate,   // [inter] the gate projection's f32 row
    const float *__restrict__ up,     // [inter] the up projection's f32 row
    float *__restrict__ out,          // [inter] out: silu(gate) * up
    const float limit,                // the clamp from the checkpoint (0 disables it)
    const uint32_t inter) {           // the row width
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= inter) {
        return;
    }
    float g = gate[index];
    float u = up[index];
    if (limit > 0.0f) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    // One true divide (the second declared deviation: the host rounds a reciprocal, then a multiply).
    const float silu = __fdiv_rn(g, __fadd_rn(1.0f, expf(-g)));
    out[index] = __fmul_rn(silu, u);
}

// ---------------------------------------------------------------------------------------------
// 8. rope_rotate_rows_bf16: the attention output's inverse RoPE over the per-rank head rows.
// ---------------------------------------------------------------------------------------------
// The moved-out rotation the host loop still does in Rust: after the csa2 body writes the BF16 out
// rows, the reference rotates each head's rope tail BACK before wo_a (model.py:772; the host path is
// linalg::rope_rotate_inverse at tp4_forward.rs:3515-3540). This kernel is that rotation for a batch
// of head rows, with the sign carried as a WORD instead of a second table class: sin_sign = 1 is the
// forward rotation (the same pairs, the same tables) and sin_sign = -1 the inverse. One pair of
// tables therefore serves both directions; a negated table copy would state one bit twice.
//
// Arithmetic, and why it is claimed bit-identical: adjacent pairs (x[2i], x[2i+1]) ->
// (a cos - b sin, a sin + b cos), the engine's convention (linalg.rs:1197-1218), NOT the half-split
// NeoX form. The stored rows are BF16, widened to f32 exactly; each product is rounded
// (__fmul_rn), each sum rounded (__fadd_rn/__fsub_rn) in the host's order, and the result rounded to
// BF16 once - the host does the same widening, plain f32 products and sums (no FMA contraction in
// Rust), and one narrowing when it writes the activation row. The head before the tail is copied
// unchanged. DECLARED DEVIATION: none in the arithmetic; the only difference from the host is that
// this is a device launch. A twin in layer_seams_ref.rs asserts both sign arms bit for bit against a
// Rust transcription of rope_rotate_with on the same bytes.
//
// Geometry: grid.x = rows (one block per head row), SPARK_SEAM_THREADS threads; head_width 512 and
// rope_head_dim 64 in the released config, but both are runtime words here because the head width
// is the wq_b output the launch already knows. rows = 0 is refused by the entry guard (a launch
// with an empty grid would silently record a stage that did nothing). out MAY alias rows: each pair
// reads both its values before writing either, and no thread touches another pair's values.
extern "C" __global__ void rope_rotate_rows_bf16(
    const __nv_bfloat16 *__restrict__ src,   // [rows][head_width] the stored head rows
    const float *__restrict__ cos,            // [rope_head_dim / 2] this position's cos
    const float *__restrict__ sin,            // [rope_head_dim / 2] this position's sin
    __nv_bfloat16 *__restrict__ out,          // [rows][head_width] the rotated rows
    const uint32_t rows,                      // grid.x; the head count of this rank
    const uint32_t head_width,                // 512 for the MLA latent
    const uint32_t rope_head_dim,             // 64 in the released config
    const int32_t sin_sign,                   // 1: rope_rotate, -1: rope_rotate_inverse
    GpuTraceRing *__restrict__ trace_ring,    // guarded: null means write nothing (see TRACE)
    const uint32_t layer_id,
    const uint64_t step_seq) {
    uint64_t trace_enter = 0U;
    if (trace_ring != nullptr) {
        SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, trace_enter);
    }
    (void)step_seq;
    const uint32_t head = blockIdx.x;
    if (rows == 0u || head >= rows || head_width == 0u || rope_head_dim == 0u ||
        rope_head_dim > head_width || (rope_head_dim & 1u) != 0u ||
        (sin_sign != 1 && sin_sign != -1)) {
        return;
    }
    const uint32_t lane = threadIdx.x;
    const uint32_t tail = head_width - rope_head_dim;
    const uint32_t pairs = rope_head_dim / 2u;
    const __nv_bfloat16 *row = src + (uint64_t)head * head_width;
    __nv_bfloat16 *dst = out + (uint64_t)head * head_width;
    for (uint32_t column = lane; column < tail; column += blockDim.x) {
        dst[column] = row[column];
    }
    const float sign = sin_sign < 0 ? -1.0f : 1.0f;
    for (uint32_t pair = lane; pair < pairs; pair += blockDim.x) {
        const float a = __bfloat162float(row[tail + 2u * pair]);
        const float b = __bfloat162float(row[tail + 2u * pair + 1u]);
        const float first = __fadd_rn(__fmul_rn(a, cos[pair]),
                                       -__fmul_rn(sign, __fmul_rn(b, sin[pair])));
        const float second = __fadd_rn(__fmul_rn(sign, __fmul_rn(a, sin[pair])),
                                        __fmul_rn(b, cos[pair]));
        dst[tail + 2u * pair] = __float2bfloat16_rn(first);
        dst[tail + 2u * pair + 1u] = __float2bfloat16_rn(second);
    }
    if (trace_ring != nullptr) {
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, SPARK_SEAM_STAGE_ROPE_ROWS, trace_enter);
    }
}

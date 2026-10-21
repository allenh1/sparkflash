// kernels/b12x_mxfp8_gemv_m1_sm121.cu
//
// The dedicated decode-step (M=1) MXFP8 GEMV of WS-D: a bandwidth-first kernel, not a retile of the
// prefill GEMM. It is added, not substituted: the pinned b12x_mxfp8_dense_sm121 kernel and its
// cubin stay the reference and the frozen ABI id, and this file compiles into its own cubin with
// its own symbols, its own baseline entry and its own launch module.
//
// Why this shape. The B12x weight layout is a grid of 32x32 micro-tiles: one tile is 32 outputs x
// 32 K values, contiguous 1024 bytes, and the tile row for output n is the 32 bytes at
// +32*(n%32) inside it (m4_common_sm121.cuh's spark_b12x_weight_index). That is exactly the
// granularity a decode GEMV wants: one warp per micro-tile - one lane per output column - and each
// lane's 32 K-bytes for a K-block are contiguous and 32-byte aligned, so a warp reads the whole
// tile with two 128-bit loads per lane (1024 coalesced bytes per K-block).
//
// ONE KERNEL BODY, THREE INSTANTIATIONS. The body is templated on two compile-time switches, so the
// three entry points differ only in the two operations the seam they serve is pinned to, never in
// the tiling, the prefetch schedule or the ownership of an output:
//
//   b12x_mxfp8_gemv_m1_sm121        k_fused = true,  k_coded_activation = true
//       The dense projections (wq_a, wkv, wq_b, wo_b). acc = fma(a_value, w_value, acc).
//
//   b12x_mxfp8_gemv_m1_nf_sm121     k_fused = false, k_coded_activation = true
//       The shared expert. Its host reference is sparkflash-engine's native_linear ->
//       linalg::matmul_f8_scaled -> coded_matmul, whose inner loop is the Rust expression
//       "acc += value * element(row, col)": a ROUNDED f32 multiply and then a ROUNDED f32 add, two
//       roundings, never contracted. A device twin of that row therefore has to spell the same two
//       operations out: acc = __fadd_rn(acc, __fmul_rn(a_value, w_value)). Values and order are
//       already the same as the fused mode (w_value is decode(code) * scale, one rounded multiply;
//       a_value is one ue8m0 activation scale times one exact E4M3 decode) - only the rounding
//       sequence of the accumulate differs, and it has to, or the shared expert would stop being
//       byte-identical to the reference row it replaces.
//
//   b12x_mxfp8_gemv_m1_nf_f32a_sm121   k_fused = false, k_coded_activation = false
//       The compressor seam (ref_seam). Its host reference is attention.rs's compressor step calling
//       linalg::matmul_hot_rows with an f32 activation row and f32 weights decoded once per layer:
//       again a rounded multiply then a rounded add, but with NO activation quantisation, so this
//       mode stages the f32 row into shared memory unchanged (a float4 copy) instead of decoding
//       E4M3 codes and scales. The weights stay F8 coded on the device: w_value is decoded in-kernel
//       with the same table arithmetic the host's one-time decode used, so the value is identical
//       and the traffic stays at the coded weight bytes rather than four times them. The activation
//       pointer is passed in the a_fp8 word, reinterpreted as const float*, and a_scales is unused.
//
// BIT IDENTITY. Lane j owns one output column and keeps the only accumulator for it, so the sum is
// exactly the pinned contract: acc starts at 0, K ascends in 32-value MX blocks and ascends inside
// each block, each element is one accumulate of a_value * w_value with a_value = the activation
// value for that K (decoded E4M3 x ue8m0 in the coded modes, the f32 row itself in the f32 mode) and
// w_value = decode(w) * w_scale (one ue8m0 weight scale per 32x32 block, the same byte for the whole
// warp), and the single __float2bfloat16_rn happens at the end. There is no split-K, no tree
// reduction and no reassociation here: this kernel changes WHICH THREAD owns an output, never the
// order of its sum. The coded activation decode is done once per launch into shared memory in f32
// (E4M3 codes are exactly representable in f16, so the cvt.e4m3x2 -> f16 -> f32 round trip is the
// identical f32 the scalar decoder produces, and the multiply by the ue8m0 scale is the same single
// rounded f32 multiply).
//
// BANDWIDTH. The launch is one warp per 32 outputs: wq_a 40 warps, wkv 16, wq_b 256, wo_b 160, the
// shared expert's w1/w3 72 warps each and its w2 160. SM count is not the binding resource - bytes
// in flight are. Each lane prefetches SPARK_M4_GEMV_PIPELINE K-blocks (6 x 32 bytes) of its own row
// ahead of the accumulate loop, i.e. 6 KB per warp, which at a 600-cycle HBM latency is ~288 GB/s of
// capability per 16-warp launch: enough to reach the 60% of peak bar without splitting K. The
// accumulate loop costs ~130 issued instructions per 1024 bytes of weight (2 loads + 1 scale load +
// 16 packed decodes + 32 f16->f32 converts + 32 f32 multiplies + 32 fused multiply-adds), so issue
// rate is below the memory time for every shape.
//
// Shape contract: rows == 1 and inner a multiple of 32. The launch wrapper refuses anything else.
//
// What is proven here: nvcc compiles it for sm_121a (the build script probes every kernel file and
// scripts/check-kernel-baseline.sh recompiles it against the committed baseline); its signature,
// geometry and pipeline depth are pinned by parsed-declaration tests in src/mxfp8_gemv_launch.rs;
// and the tiling model it implements is compared bit for bit against the pinned arithmetic through
// src/mxfp8_ref.rs's dense_gemm (fused mode) and against the rounded-multiply-then-rounded-add
// model of the engine's coded_matmul (non-fused mode) in tests/mxfp8_gemv_oracle.rs for every shape
// it serves. What is NOT proven here: it has never executed on a GPU on this branch, and its
// achieved bandwidth is measured by the window row, not by this file.

#include "m4_common_sm121.cuh"

// One warp per B12x micro-tile: 32 output columns, 32 threads, one lane per column.
#define SPARK_M4_GEMV_OUTPUTS_PER_TILE 32u
#define SPARK_M4_GEMV_THREADS 32u
// K-blocks whose 32 weight bytes per lane are held in registers while the previous ones accumulate.
#define SPARK_M4_GEMV_PIPELINE 6u
// The longest reduction the shared activation row can hold: 5120 f32 = 20 KB of static shared
// memory, which covers every dense shape the decode step serves (hidden 5120 is the widest). The
// launch wrapper refuses a wider shape instead of letting the decode truncate.
#define SPARK_M4_GEMV_MAX_INNER 5120u
// One lane's slice of a K-block: 32 contiguous E4M3 bytes, one per K value.
#define SPARK_M4_GEMV_LANE_BYTES 32u
#define SPARK_M4_GEMV_TILE_BYTES (SPARK_B12X_NTILE * SPARK_B12X_KTILE)

static_assert(SPARK_M4_GEMV_THREADS == SPARK_M4_GEMV_OUTPUTS_PER_TILE,
              "one lane per output column of the tile");
static_assert(SPARK_M4_GEMV_THREADS == 32u, "one warp");
static_assert(SPARK_M4_GEMV_OUTPUTS_PER_TILE == SPARK_B12X_NTILE,
              "a warp covers exactly one B12x micro-tile's outputs");
static_assert(SPARK_M4_GEMV_LANE_BYTES == SPARK_B12X_KTILE,
              "a lane reads one whole K block of its column");
static_assert(SPARK_M4_GEMV_PIPELINE >= 1u, "at least one block in flight");
static_assert(SPARK_M4_GEMV_MAX_INNER % SPARK_M4_QUANT_BLOCK == 0u,
              "the widest reduction is a whole number of MX blocks");

__device__ __forceinline__ uint4 spark_m4_gemv_load128(const uint8_t *p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// The 16-bit pair of E4M3 codes covering elements 2*pair and 2*pair+1 of a K block.
__device__ __forceinline__ uint16_t spark_m4_gemv_pair(const uint4 lo, const uint4 hi,
                                                       const uint32_t pair) {
    const uint4 v = pair < 8u ? lo : hi;
    const uint32_t index = pair & 7u;
    const uint32_t word = index >> 1;
    const uint32_t shift = (index & 1u) * 16u;
    const uint32_t bits = word == 0u ? v.x : word == 1u ? v.y : word == 2u ? v.z : v.w;
    return (uint16_t)((bits >> shift) & 0xffffu);
}

// E4M3x2 -> f16x2 -> two f32. Every E4M3 value is exactly representable in f16 (3 mantissa bits,
// exponent range -9..8), so these are the exact f32 values the scalar decoder in the pinned kernel
// produces, obtained with one packed instruction per pair instead of a branchy scit.
__device__ __forceinline__ void spark_m4_gemv_pair_to_f32(const uint16_t two, float &lo, float &hi) {
    asm("{\n\t"
        ".reg .b32 h2;\n\t"
        ".reg .b16 l16, h16;\n\t"
        "cvt.rn.f16x2.e4m3x2 h2, %2;\n\t"
        "mov.b32 {l16, h16}, h2;\n\t"
        "cvt.f32.f16 %0, l16;\n\t"
        "cvt.f32.f16 %1, h16;\n\t"
        "}"
        : "=f"(lo), "=f"(hi)
        : "h"(two));
}

// One K block for this lane's output column: 32 elements appended to acc in ascending K order, each
// with the pinned multiply order (w_value = decode * w_scale, rounded once, then accumulated).
// k_fused picks the accumulate the seam's reference is written with: one contracted fma for the
// dense family, a rounded multiply followed by a rounded add for the engine's coded_matmul twin.
template <bool k_fused>
__device__ __forceinline__ void spark_m4_gemv_block(const uint4 w_lo, const uint4 w_hi,
                                                    const float *__restrict__ a_values,
                                                    const uint8_t w_scale_code, float &acc) {
    const float w_scale = spark_ue8m0_decode(w_scale_code);
#pragma unroll
    for (uint32_t pair = 0u; pair < 16u; ++pair) {
        float w0 = 0.0f;
        float w1 = 0.0f;
        spark_m4_gemv_pair_to_f32(spark_m4_gemv_pair(w_lo, w_hi, pair), w0, w1);
        if (k_fused) {
            acc = __fmaf_rn(a_values[2u * pair], w0 * w_scale, acc);
            acc = __fmaf_rn(a_values[2u * pair + 1u], w1 * w_scale, acc);
        } else {
            acc = __fadd_rn(acc, __fmul_rn(a_values[2u * pair], w0 * w_scale));
            acc = __fadd_rn(acc, __fmul_rn(a_values[2u * pair + 1u], w1 * w_scale));
        }
    }
}

// The body of every entry point. k_coded_activation picks how the 5120-wide (or narrower) operand
// row reaches shared memory: decoded once from E4M3 codes and ue8m0 scales, or copied unchanged from
// an f32 row for the seam whose host reference never quantises its activation.
template <bool k_fused, bool k_coded_activation>
__device__ __forceinline__ void spark_m4_gemv_m1_body(
    const uint8_t *__restrict__ a_fp8,     // [rows][inner] E4M3 activations, row 0 only
    const uint8_t *__restrict__ a_scales,  // [rows][inner / 32] ue8m0, one per MX block
    const uint8_t *__restrict__ w_fp8,     // B12x tiled E4M3 weights
    const uint8_t *__restrict__ w_scales,  // B12x tiled ue8m0 weight scales
    __nv_bfloat16 *__restrict__ out,       // [rows][cols] bf16 result
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t inner,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t lane = threadIdx.x;
    const uint32_t col = blockIdx.x * SPARK_M4_GEMV_OUTPUTS_PER_TILE + lane;
    const uint32_t k_blocks = inner / SPARK_M4_QUANT_BLOCK;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    // Shape contract, re-checked here so a wrong launch cannot read past the activation row or
    // past the shared buffer: one row, a whole number of MX blocks, no wider than the buffer.
    if (rows != 1u || inner % SPARK_M4_QUANT_BLOCK != 0u || inner > SPARK_M4_GEMV_MAX_INNER) {
        if (lane == 0u) {
            SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, 0u, t_enter);
            (void)step_seq;
        }
        return;
    }

    // The operand row: the same values for every lane and every warp, so it is built once
    // cooperatively in shared memory and then broadcast-read by the accumulate loop.
    __shared__ __align__(16) float a_values[SPARK_M4_GEMV_MAX_INNER];
    if (k_coded_activation) {
        const uint32_t pairs = inner / 2u;
        for (uint32_t g = lane; g < pairs; g += SPARK_M4_GEMV_THREADS) {
            const uint32_t k = 2u * g;
            const float a_scale = spark_ue8m0_decode(a_scales[k / SPARK_M4_QUANT_BLOCK]);
            const uint16_t two = (uint16_t)a_fp8[k] | ((uint16_t)a_fp8[k + 1u] << 8);
            float lo = 0.0f;
            float hi = 0.0f;
            spark_m4_gemv_pair_to_f32(two, lo, hi);
            a_values[k] = lo * a_scale;
            a_values[k + 1u] = hi * a_scale;
        }
    } else {
        // The f32 row arrives already rounded the way the host reference holds it; copying it is
        // the whole adaptation, and the float4 copy keeps the staging off the critical path.
        const float *__restrict__ a_row = reinterpret_cast<const float *>(a_fp8);
        const uint4 *__restrict__ source = reinterpret_cast<const uint4 *>(a_row);
        uint4 *__restrict__ destination = reinterpret_cast<uint4 *>(a_values);
        const uint32_t vectors = inner / 4u;
        for (uint32_t g = lane; g < vectors; g += SPARK_M4_GEMV_THREADS) {
            destination[g] = source[g];
        }
    }
    __syncwarp(0xffffffffu);

    const bool active = col < cols;
    if (active) {
        const uint8_t *w_row = w_fp8;
        float acc = 0.0f;

        // Prologue: put the first SPARK_M4_GEMV_PIPELINE K blocks of this lane's column in flight.
        uint4 w_lo[SPARK_M4_GEMV_PIPELINE];
        uint4 w_hi[SPARK_M4_GEMV_PIPELINE];
#pragma unroll
        for (uint32_t p = 0u; p < SPARK_M4_GEMV_PIPELINE; ++p) {
            if (p < k_blocks) {
                const uint64_t index = spark_b12x_weight_index(col, p * SPARK_M4_QUANT_BLOCK, inner);
                w_lo[p] = spark_m4_gemv_load128(w_row + index);
                w_hi[p] = spark_m4_gemv_load128(w_row + index + 16u);
            }
        }

        uint32_t block = 0u;
        for (; block + SPARK_M4_GEMV_PIPELINE <= k_blocks; block += SPARK_M4_GEMV_PIPELINE) {
#pragma unroll
            for (uint32_t p = 0u; p < SPARK_M4_GEMV_PIPELINE; ++p) {
                const uint32_t current = block + p;
                const uint8_t w_scale_code =
                    w_scales[spark_b12x_scale_index(col, current * SPARK_M4_QUANT_BLOCK, inner)];
                spark_m4_gemv_block<k_fused>(w_lo[p], w_hi[p],
                                             a_values + current * SPARK_M4_QUANT_BLOCK,
                                             w_scale_code, acc);
                // Refill this slot with the block that will be needed SPARK_M4_GEMV_PIPELINE
                // iterations from now, so the load is in flight while the next blocks accumulate.
                const uint32_t next = current + SPARK_M4_GEMV_PIPELINE;
                if (next < k_blocks) {
                    const uint64_t index =
                        spark_b12x_weight_index(col, next * SPARK_M4_QUANT_BLOCK, inner);
                    w_lo[p] = spark_m4_gemv_load128(w_row + index);
                    w_hi[p] = spark_m4_gemv_load128(w_row + index + 16u);
                }
            }
        }
        // Tail: the last k_blocks % SPARK_M4_GEMV_PIPELINE blocks, loaded as they are consumed. The
        // order is still ascending K; only the prefetch depth differs.
        for (; block < k_blocks; ++block) {
            const uint64_t index =
                spark_b12x_weight_index(col, block * SPARK_M4_QUANT_BLOCK, inner);
            const uint4 lo = spark_m4_gemv_load128(w_row + index);
            const uint4 hi = spark_m4_gemv_load128(w_row + index + 16u);
            const uint8_t w_scale_code =
                w_scales[spark_b12x_scale_index(col, block * SPARK_M4_QUANT_BLOCK, inner)];
            spark_m4_gemv_block<k_fused>(lo, hi, a_values + block * SPARK_M4_QUANT_BLOCK,
                                         w_scale_code, acc);
        }

        out[col] = __float2bfloat16_rn(acc);
    }
    __syncwarp(0xffffffffu);

    if (lane == 0u) {
        // payload_meta: one bf16 per output element of this block's tile (M=1: one row x 32 cols).
        const uint32_t meta = SPARK_M4_GEMV_OUTPUTS_PER_TILE * 2u;
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}

// The dense projections: contracted accumulate, coded activation, the pinned dense-family order.
extern "C" __global__ void b12x_mxfp8_gemv_m1_sm121(
    const uint8_t *__restrict__ a_fp8, const uint8_t *__restrict__ a_scales,
    const uint8_t *__restrict__ w_fp8, const uint8_t *__restrict__ w_scales,
    __nv_bfloat16 *__restrict__ out, const uint32_t rows, const uint32_t cols,
    const uint32_t inner, GpuTraceRing *__restrict__ trace_ring, const uint32_t layer_id,
    const uint64_t step_seq) {
    spark_m4_gemv_m1_body<true, true>(a_fp8, a_scales, w_fp8, w_scales, out, rows, cols, inner,
                                      trace_ring, layer_id, step_seq);
}

// The shared expert: the engine's rounded-multiply-then-rounded-add twin, coded activation.
extern "C" __global__ void b12x_mxfp8_gemv_m1_nf_sm121(
    const uint8_t *__restrict__ a_fp8, const uint8_t *__restrict__ a_scales,
    const uint8_t *__restrict__ w_fp8, const uint8_t *__restrict__ w_scales,
    __nv_bfloat16 *__restrict__ out, const uint32_t rows, const uint32_t cols,
    const uint32_t inner, GpuTraceRing *__restrict__ trace_ring, const uint32_t layer_id,
    const uint64_t step_seq) {
    spark_m4_gemv_m1_body<false, true>(a_fp8, a_scales, w_fp8, w_scales, out, rows, cols, inner,
                                       trace_ring, layer_id, step_seq);
}

// The compressor seam: the engine's rounded-multiply-then-rounded-add twin, f32 activation row.
extern "C" __global__ void b12x_mxfp8_gemv_m1_nf_f32a_sm121(
    const uint8_t *__restrict__ a_fp8, const uint8_t *__restrict__ a_scales,
    const uint8_t *__restrict__ w_fp8, const uint8_t *__restrict__ w_scales,
    __nv_bfloat16 *__restrict__ out, const uint32_t rows, const uint32_t cols,
    const uint32_t inner, GpuTraceRing *__restrict__ trace_ring, const uint32_t layer_id,
    const uint64_t step_seq) {
    spark_m4_gemv_m1_body<false, false>(a_fp8, a_scales, w_fp8, w_scales, out, rows, cols, inner,
                                        trace_ring, layer_id, step_seq);
}

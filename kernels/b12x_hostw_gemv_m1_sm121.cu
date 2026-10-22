// kernels/b12x_hostw_gemv_m1_sm121.cu
//
// The decode-step (M=1) GEMV whose weight is the checkpoint's OWN values - widened bf16 or plain
// f32 - rather than MXFP8 codes with a ue8m0 grid. It is an addition, not a substitution: the pinned
// b12x_mxfp8_gemv_m1_sm121 cubin, its four symbols and its digest are untouched, and this file
// compiles into its own cubin with its own symbols, its own baseline entry and its own launch
// module.
//
// Why a second GEMV. The ref_seam block's remaining host statements are the compressor's wkv and
// wgate Linears, the indexer's key projection (wk) and its per-query weights projection
// (weights_proj). None of them has an MXFP8 weight: compressor.wkv/wgate are f32 at ratio > 1 and
// bf16 at ratio == 1, indexer.wk and indexer.weights_proj are bf16 (docs/build/
// native-arithmetic-audit.md, model.py:446-449, 461-463, 513, 540). Quantising those weights to
// E4M3 to reuse the pinned coded GEMV would change their values, so the host reference's operands
// have to reach the device as they are. This kernel is that load path.
//
// THE CHAIN IS THE ENGINE'S. attention.rs's compressor step calls linalg::matmul_hot_rows and the
// indexer calls WeightView::matmul; both run matmul_checked's inner loop, whose only expression is
// "acc += xb[col] * wr[col]": a ROUNDED f32 multiply and then a ROUNDED f32 add, ascending col, one
// accumulator per output. So this kernel stages the f32 activation row unchanged and spells the
// same two operations out per element - __fadd_rn(acc, __fmul_rn(a, w)) - with one accumulator per
// output column, K ascending in 32-value blocks and ascending inside a block. There is no fma, no
// split-K, no tree reduction and no reassociation: which thread owns an output changes, the order of
// its sum does not. The weight value is the exact widening of the checkpoint's own value (bf16's
// 8-bit significand is inside f32's, so the widening is exact; f32 is already f32), which is exactly
// what quant::decode does on the host before its matmul.
//
// THE OUTPUT ROUNDING IS THE CALLER'S. weights_proj hands its bf16-module output on and rounds it to
// bf16 on the host, so its symbol stores __float2bfloat16_rn(acc). wk feeds rms_norm in f32 and the
// compressor's wkv/wgate rows are pooled in f32, so those symbols store the f32 accumulator itself.
// A bf16 store there would move the values; the f32 store is a separate symbol, not a flag, for the
// same reason the coded GEMV keeps four symbols.
//
// LAYOUT. The B12x tiling is a grid of 32x32 micro-tiles, here of the weight's own element type: one
// tile is 32 outputs x 32 K values, so a lane's 32 values for one K block are contiguous (64 bytes
// for bf16, 128 for f32) and a warp reads a whole tile as one coalesced run. The encoded form of the
// tile is the same permutation the coded GEMV uses - tile = (n/32) * k_blocks + (k/32), byte offset
// (n%32)*32 + (k%32) within it - just scaled by the element size. The host stages it once per layer:
// a pure permutation, no value is touched.
//
// BIT IDENTITY is by construction (above) and is pinned by tests/hostw_gemv_oracle.rs against the
// engine's own host rules at every shape the seam serves. What is NOT proven here: it has never
// executed on a GPU on this branch, and its achieved bandwidth is the window row's measurement, not
// this file's.

#include "m4_common_sm121.cuh"

// One warp per B12x micro-tile: 32 output columns, 32 threads, one lane per column.
#define SPARK_HW_GEMV_OUTPUTS_PER_TILE 32u
#define SPARK_HW_GEMV_THREADS 32u
// K blocks whose 64/128 bytes per lane are held in registers while the previous ones accumulate.
#define SPARK_HW_GEMV_PIPELINE 4u
// The longest reduction the shared activation row can hold: 5120 f32 = 20 KB, the widest dense
// shape the decode step serves. The launch wrapper refuses anything wider.
#define SPARK_HW_GEMV_MAX_INNER 5120u
// One micro-tile: 32 outputs x 32 K values of the weight's own element type.
#define SPARK_HW_GEMV_TILE_VALUES 1024u
// One lane's values of a K block and their byte count, for the widest element type (f32).
#define SPARK_HW_GEMV_LANE_VALUES 32u
#define SPARK_HW_GEMV_LANE_WORDS_MAX 8u

static_assert(SPARK_HW_GEMV_THREADS == SPARK_HW_GEMV_OUTPUTS_PER_TILE,
              "one lane per output column of the tile");
static_assert(SPARK_HW_GEMV_THREADS == 32u, "one warp");
static_assert(SPARK_HW_GEMV_OUTPUTS_PER_TILE == SPARK_B12X_NTILE,
              "a warp covers exactly one B12x micro-tile's outputs");
static_assert(SPARK_HW_GEMV_LANE_VALUES == SPARK_B12X_KTILE,
              "a lane reads one whole K block of its column");
static_assert(SPARK_HW_GEMV_PIPELINE >= 1u, "at least one block in flight");
static_assert(SPARK_HW_GEMV_MAX_INNER % SPARK_M4_QUANT_BLOCK == 0u,
              "the widest reduction is a whole number of 32-value blocks");
static_assert(SPARK_HW_GEMV_TILE_VALUES == SPARK_B12X_NTILE * SPARK_B12X_KTILE,
              "the tile is 32 outputs x 32 K values");

// One 16-byte load of the lane's contiguous weight values.
__device__ __forceinline__ uint4 spark_hw_gemv_load128(const void *p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// One K block for this lane's output column: 32 weight values appended to acc in ascending K order,
// each as a rounded multiply followed by a rounded add - the engine matmul's own expression.
// W is __nv_bfloat16 (widened exactly; a bf16 is a truncated f32) or float (the checkpoint's f32
// weights). No scale is involved: these weights carry no MX grid.
template <typename W>
__device__ __forceinline__ void spark_hw_gemv_block(const uint4 *block,
                                                    const float *__restrict__ a_values,
                                                    float &acc) {
    const uint32_t *words = reinterpret_cast<const uint32_t *>(block);
    if constexpr (sizeof(W) == 4u) {
#pragma unroll
        for (uint32_t i = 0u; i < SPARK_HW_GEMV_LANE_VALUES; ++i) {
            acc = __fadd_rn(acc, __fmul_rn(a_values[i], __uint_as_float(words[i])));
        }
    } else {
#pragma unroll
        for (uint32_t i = 0u; i < SPARK_HW_GEMV_LANE_VALUES / 2u; ++i) {
            const uint32_t word = words[i];
            const float lo = spark_bits_float((word & 0xffffu) << 16);
            const float hi = spark_bits_float(word & 0xffff0000u);
            acc = __fadd_rn(acc, __fmul_rn(a_values[2u * i], lo));
            acc = __fadd_rn(acc, __fmul_rn(a_values[2u * i + 1u], hi));
        }
    }
}

// The body of every entry point. The weight's element type decides the load width; k_out_f32
// decides whether the accumulator is rounded to bf16 or written as f32.
template <typename W, bool k_out_f32>
__device__ __forceinline__ void spark_hw_gemv_m1_body(
    const float *__restrict__ a_row,       // [rows][inner] f32 activation, row 0 only
    const W *__restrict__ w,               // B12x tiled weights of the checkpoint's own type
    void *__restrict__ out,                // [rows][cols] bf16, or f32 when k_out_f32
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t inner,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t lane = threadIdx.x;
    const uint32_t col = blockIdx.x * SPARK_HW_GEMV_OUTPUTS_PER_TILE + lane;
    const uint32_t k_blocks = inner / SPARK_M4_QUANT_BLOCK;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    // Shape contract, re-checked here so a wrong launch cannot read past the activation row, past a
    // tile or past the shared buffer.
    if (rows != 1u || inner % SPARK_M4_QUANT_BLOCK != 0u ||
        inner > SPARK_HW_GEMV_MAX_INNER || cols % SPARK_HW_GEMV_OUTPUTS_PER_TILE != 0u) {
        if (lane == 0u) {
            SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, 0u, t_enter);
            (void)step_seq;
        }
        return;
    }

    // The operand row: the same values for every lane and every warp, so it is staged once
    // cooperatively in shared memory and then read by the accumulate loop.
    __shared__ __align__(16) float a_values[SPARK_HW_GEMV_MAX_INNER];
    {
        const uint4 *__restrict__ source = reinterpret_cast<const uint4 *>(a_row);
        uint4 *__restrict__ destination = reinterpret_cast<uint4 *>(a_values);
        const uint32_t vectors = inner / 4u;
        for (uint32_t g = lane; g < vectors; g += SPARK_HW_GEMV_THREADS) {
            destination[g] = source[g];
        }
    }
    __syncwarp(0xffffffffu);

    const uint32_t n_blocks = cols / SPARK_HW_GEMV_OUTPUTS_PER_TILE;
    if (col < cols) {
        const uint32_t n_local = col % SPARK_HW_GEMV_OUTPUTS_PER_TILE;
        // The tile the lane's column belongs to, then the lane's own 32 values inside it. The tile
        // order is the coded GEMV's: tile = (n/32) * k_blocks + (k/32).
        const W *column = w + (uint64_t)(col / SPARK_HW_GEMV_OUTPUTS_PER_TILE) * k_blocks *
                                  (uint64_t)SPARK_HW_GEMV_TILE_VALUES +
                          (uint64_t)n_local * SPARK_HW_GEMV_LANE_VALUES;
        const uint32_t words = (SPARK_HW_GEMV_LANE_VALUES * (uint32_t)sizeof(W)) / 16u;
        float acc = 0.0f;

        // Prologue: the first SPARK_HW_GEMV_PIPELINE K blocks in flight.
        uint4 pre[SPARK_HW_GEMV_PIPELINE][SPARK_HW_GEMV_LANE_WORDS_MAX];
#pragma unroll
        for (uint32_t p = 0u; p < SPARK_HW_GEMV_PIPELINE; ++p) {
            if (p < k_blocks) {
                const uint4 *src =
                    reinterpret_cast<const uint4 *>(column + (uint64_t)p * SPARK_HW_GEMV_TILE_VALUES);
#pragma unroll
                for (uint32_t v = 0u; v < SPARK_HW_GEMV_LANE_WORDS_MAX; ++v) {
                    if (v < words) {
                        pre[p][v] = spark_hw_gemv_load128(src + v);
                    }
                }
            }
        }

        uint32_t block = 0u;
        for (; block + SPARK_HW_GEMV_PIPELINE <= k_blocks; block += SPARK_HW_GEMV_PIPELINE) {
#pragma unroll
            for (uint32_t p = 0u; p < SPARK_HW_GEMV_PIPELINE; ++p) {
                const uint32_t current = block + p;
                spark_hw_gemv_block<W>(pre[p],
                                       a_values + current * SPARK_M4_QUANT_BLOCK, acc);
                const uint32_t next = current + SPARK_HW_GEMV_PIPELINE;
                if (next < k_blocks) {
                    const uint4 *src = reinterpret_cast<const uint4 *>(
                        column + (uint64_t)next * SPARK_HW_GEMV_TILE_VALUES);
#pragma unroll
                    for (uint32_t v = 0u; v < SPARK_HW_GEMV_LANE_WORDS_MAX; ++v) {
                        if (v < words) {
                            pre[p][v] = spark_hw_gemv_load128(src + v);
                        }
                    }
                }
            }
        }
        // Tail: the last k_blocks % SPARK_HW_GEMV_PIPELINE blocks. The order is still ascending K;
        // only the prefetch depth differs.
        for (; block < k_blocks; ++block) {
            uint4 tail[SPARK_HW_GEMV_LANE_WORDS_MAX];
            const uint4 *src =
                reinterpret_cast<const uint4 *>(column + (uint64_t)block * SPARK_HW_GEMV_TILE_VALUES);
#pragma unroll
            for (uint32_t v = 0u; v < SPARK_HW_GEMV_LANE_WORDS_MAX; ++v) {
                if (v < words) {
                    tail[v] = spark_hw_gemv_load128(src + v);
                }
            }
            spark_hw_gemv_block<W>(tail, a_values + block * SPARK_M4_QUANT_BLOCK, acc);
        }

        if constexpr (k_out_f32) {
            reinterpret_cast<float *>(out)[col] = acc;
        } else {
            reinterpret_cast<__nv_bfloat16 *>(out)[col] = __float2bfloat16_rn(acc);
        }
    }
    __syncwarp(0xffffffffu);

    if (lane == 0u) {
        // payload_meta: one output word per element of this block's tile (M=1: one row x 32 cols);
        // the word is bf16 except in the f32-store mode, where it is the f32 accumulator.
        const uint32_t meta = SPARK_HW_GEMV_OUTPUTS_PER_TILE * (k_out_f32 ? 4u : 2u);
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
    (void)n_blocks;
}

// bf16 weights, bf16 output: the indexer's per-query weights projection.
extern "C" __global__ void b12x_hostw_gemv_m1_bf16w_sm121(
    const float *__restrict__ a_row, const __nv_bfloat16 *__restrict__ w,
    void *__restrict__ out, const uint32_t rows, const uint32_t cols, const uint32_t inner,
    GpuTraceRing *__restrict__ trace_ring, const uint32_t layer_id, const uint64_t step_seq) {
    spark_hw_gemv_m1_body<__nv_bfloat16, false>(a_row, w, out, rows, cols, inner, trace_ring,
                                                layer_id, step_seq);
}

// bf16 weights, f32 accumulators: the indexer's key projection and the ratio-1 compressor wkv.
extern "C" __global__ void b12x_hostw_gemv_m1_bf16w_f32o_sm121(
    const float *__restrict__ a_row, const __nv_bfloat16 *__restrict__ w,
    void *__restrict__ out, const uint32_t rows, const uint32_t cols, const uint32_t inner,
    GpuTraceRing *__restrict__ trace_ring, const uint32_t layer_id, const uint64_t step_seq) {
    spark_hw_gemv_m1_body<__nv_bfloat16, true>(a_row, w, out, rows, cols, inner, trace_ring,
                                               layer_id, step_seq);
}

// f32 weights, f32 accumulators: the compressor's wkv and wgate at ratio > 1.
extern "C" __global__ void b12x_hostw_gemv_m1_f32w_f32o_sm121(
    const float *__restrict__ a_row, const float *__restrict__ w, void *__restrict__ out,
    const uint32_t rows, const uint32_t cols, const uint32_t inner,
    GpuTraceRing *__restrict__ trace_ring, const uint32_t layer_id, const uint64_t step_seq) {
    spark_hw_gemv_m1_body<float, true>(a_row, w, out, rows, cols, inner, trace_ring, layer_id,
                                       step_seq);
}

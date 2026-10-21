// kernels/b12x_mxfp8_dense_sm121.cu
//
// The dense MXFP8 GEMM of specification 4.1 (B12x weight layout, per-32 ue8m0 shared scales,
// f32 accumulation, bf16 output). This is the kernel that every dense projection in the model
// uses: the attention projections, the MLP up/gate halves, and (through the wo_a variant in
// b12x_mxfp8_wo_a_sm121.cu) the grouped output projection.
//
// What is proven on this host: it parses under clang's CUDA frontend against the stub headers,
// nvcc 12.9.1 compiles it for sm_121a (a per-kernel PTX probe and the combined Milestone 4 probe
// are compiled into build.rs's OUT_DIR), its
// signature and geometry are pinned by parsed-declaration tests in src/m4_abi.rs, and the
// numerics of exactly this arithmetic order are defined and tested by src/mxfp8_ref.rs on the
// CPU. What is NOT proven: it has never executed on a GPU, there is no latency measurement, and
// the tensor-core fast path does not exist yet.
//
// Scope, stated plainly: the loop below is the SCALAR reference path, one output element per
// thread, accumulating over K in ascending 32-value blocks and, inside a block, in ascending
// element order. That order is the contract mxfp8_ref.rs implements. A future sm_121a variant
// using the block-scaled FP8 tensor-core MMA would accumulate in the hardware's own order and
// cannot be claimed bit-identical to this: it must be validated against the reference with a
// tolerance on hardware, or its order must be pinned there. Nothing in this file claims the MMA
// numerics.
//
// Layout note (specification 4.1 does not give byte offsets, so this is this project's documented
// reading, and it is the checkpoint's own grid rather than a private one): an fp8 dense weight has
// ONE ue8m0 scale per 32x32 block, the [ceil(out/32), ceil(in/32)] scale grid recorded in
// docs/architecture/03-moe.md:120, with the weights themselves in the matching 32x32 micro-tiles.
// The helpers in m4_common_sm121.cuh are the only places that know the byte offsets, the CPU
// reference uses the same formulas, tests/b12x_index_parity.rs compares the two implementations
// value for value, and mxfp8_ref.rs pins the layout with a bijection test. The FP4 routed-expert
// weights use a DIFFERENT scale layout, [out, in/32] one per output row, which is why the header
// has separate spark_b12x_row_* helpers for them.

#include "m4_common_sm121.cuh"

// The B12x micro-tile shape and the two index helpers live in m4_common_sm121.cuh, because
// b12x_mxfp8_wo_a_sm121.cu indexes the same layout. One block computes a 64x64 output tile, and
// each thread owns a contiguous 4x4 sub-tile of it, so 256 threads cover the tile exactly.
#define SPARK_M4_DENSE_TILE_M 64u
#define SPARK_M4_DENSE_TILE_N 64u
#define SPARK_M4_DENSE_THREADS 256u
#define SPARK_M4_DENSE_ROWS_PER_THREAD 4u
#define SPARK_M4_DENSE_COLS_PER_THREAD 4u
#define SPARK_M4_DENSE_SUBTILES_PER_ROW (SPARK_M4_DENSE_TILE_N / SPARK_M4_DENSE_COLS_PER_THREAD)

static_assert(SPARK_B12X_KTILE == SPARK_M4_QUANT_BLOCK, "a B12x tile is one MX scale block wide");
static_assert(SPARK_M4_DENSE_TILE_N % SPARK_B12X_NTILE == 0u, "the N tile is whole micro-tiles");
static_assert(SPARK_M4_DENSE_TILE_M * SPARK_M4_DENSE_TILE_N ==
                  SPARK_M4_DENSE_THREADS * SPARK_M4_DENSE_ROWS_PER_THREAD *
                      SPARK_M4_DENSE_COLS_PER_THREAD,
              "the 64x64 tile is exactly covered by 256 threads of 4x4");
static_assert(SPARK_M4_DENSE_TILE_N % SPARK_M4_DENSE_COLS_PER_THREAD == 0u,
              "the sub-tiles tile the N dimension");
static_assert(SPARK_M4_DENSE_TILE_M % SPARK_M4_DENSE_ROWS_PER_THREAD == 0u,
              "the sub-tiles tile the M dimension");
static_assert(SPARK_M4_DENSE_THREADS % 32u == 0u, "a block is a whole number of warps");

extern "C" __global__ void b12x_mxfp8_dense_sm121(
    const uint8_t *__restrict__ a_fp8,      // [rows][inner] E4M3 activations
    const uint8_t *__restrict__ a_scales,   // [rows][inner / 32] ue8m0, one per MX block
    const uint8_t *__restrict__ w_fp8,      // B12x tiled E4M3 weights
    const uint8_t *__restrict__ w_scales,   // B12x tiled ue8m0 weight scales
    __nv_bfloat16 *__restrict__ out,        // [rows][cols] bf16 result
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t inner,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row_base = blockIdx.x * SPARK_M4_DENSE_TILE_M +
                              (lane / SPARK_M4_DENSE_SUBTILES_PER_ROW) *
                                  SPARK_M4_DENSE_ROWS_PER_THREAD;
    const uint32_t col_base =
        blockIdx.y * SPARK_M4_DENSE_TILE_N +
        (lane % SPARK_M4_DENSE_SUBTILES_PER_ROW) * SPARK_M4_DENSE_COLS_PER_THREAD;
    const uint32_t k_blocks = inner / SPARK_M4_QUANT_BLOCK;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    // The accumulation order per output is the contract mxfp8_ref.rs implements: K ascending in
    // 32-value blocks and ascending within a block. The 4x4 sub-tile only decides which outputs a
    // thread owns; it never reorders a sum, so the reference does not have to model the tiling.
    for (uint32_t di = 0; di < SPARK_M4_DENSE_ROWS_PER_THREAD; ++di) {
        const uint32_t row = row_base + di;
        if (row >= rows) {
            continue;
        }
        const uint8_t *a_row = a_fp8 + (uint64_t)row * inner;
        const uint8_t *a_scale_row = a_scales + (uint64_t)row * k_blocks;
        float acc[SPARK_M4_DENSE_COLS_PER_THREAD];
#pragma unroll
        for (uint32_t dj = 0; dj < SPARK_M4_DENSE_COLS_PER_THREAD; ++dj) {
            acc[dj] = 0.0f;
        }
        for (uint32_t block = 0; block < k_blocks; ++block) {
            const uint32_t base = block * SPARK_M4_QUANT_BLOCK;
            const float a_scale = spark_ue8m0_decode(a_scale_row[block]);
            float w_scale[SPARK_M4_DENSE_COLS_PER_THREAD];
#pragma unroll
            for (uint32_t dj = 0; dj < SPARK_M4_DENSE_COLS_PER_THREAD; ++dj) {
                const uint32_t col = col_base + dj;
                w_scale[dj] =
                    col < cols ? spark_ue8m0_decode(w_scales[spark_b12x_scale_index(col, base, inner)])
                               : 0.0f;
            }
#pragma unroll 8
            for (uint32_t i = 0; i < SPARK_M4_QUANT_BLOCK; ++i) {
                const float a_value = spark_fp8_e4m3_decode(a_row[base + i]) * a_scale;
#pragma unroll
                for (uint32_t dj = 0; dj < SPARK_M4_DENSE_COLS_PER_THREAD; ++dj) {
                    const uint32_t col = col_base + dj;
                    if (col < cols) {
                        const float w_value =
                            spark_fp8_e4m3_decode(
                                w_fp8[spark_b12x_weight_index(col, base + i, inner)]) *
                            w_scale[dj];
                        acc[dj] += a_value * w_value;
                    }
                }
            }
        }
#pragma unroll
        for (uint32_t dj = 0; dj < SPARK_M4_DENSE_COLS_PER_THREAD; ++dj) {
            const uint32_t col = col_base + dj;
            if (col < cols) {
                out[(uint64_t)row * cols + col] = __float2bfloat16_rn(acc[dj]);
            }
        }
    }
    __syncthreads();

    if (lane == 0u) {
        // payload_meta: one bf16 per output element of this block's tile.
        const uint32_t meta = SPARK_M4_DENSE_TILE_M * SPARK_M4_DENSE_TILE_N * 2u;
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}

// ---------------------------------------------------------------------------------------------
// K1 warp-per-column dense path (new symbol; the scalar symbol above stays the reference oracle
// and the pin for the incumbent harnesses).
//
// One warp owns one output column and walks K in 512-value groups: lane l loads 16 consecutive
// E4M3 bytes of the activation row and 16 consecutive E4M3 bytes of that column's B12x row, so a
// group covers 16 MX blocks with one ld.global.nc.v4.u32 per lane and one 5-step shuffle tree per
// output row. The block has eight warps (eight columns), and grid.y chunks the rows so one weight
// row is read once for up to SPARK_M4_DENSE_WARP_ROWS rows. The scalar accumulation order is the
// pinned oracle, so this path declares a tolerance instead of bit-identity: the tree folds the 32
// lane partials, which is a different f32 summation order.
#define SPARK_M4_DENSE_WARP_THREADS 256u
#define SPARK_M4_DENSE_WARP_LANES 32u
#define SPARK_M4_DENSE_WARP_WARPS (SPARK_M4_DENSE_WARP_THREADS / SPARK_M4_DENSE_WARP_LANES)
#define SPARK_M4_DENSE_WARP_ROWS 8u
#define SPARK_M4_DENSE_WARP_GROUP 512u
#define SPARK_M4_DENSE_WARP_BYTES_PER_LANE 16u

static_assert(SPARK_M4_DENSE_WARP_THREADS == SPARK_M4_DENSE_WARP_WARPS * SPARK_M4_DENSE_WARP_LANES,
              "the warp block is whole warps");
static_assert(SPARK_M4_DENSE_WARP_GROUP ==
                  SPARK_M4_DENSE_WARP_LANES * SPARK_M4_DENSE_WARP_BYTES_PER_LANE,
              "one group is one vector load per lane");
static_assert(SPARK_M4_DENSE_WARP_BYTES_PER_LANE == 16u, "a lane load is 16 E4M3 bytes");
static_assert(SPARK_M4_DENSE_WARP_GROUP % SPARK_M4_QUANT_BLOCK == 0u,
              "a group ends on an MX block boundary");

__device__ __forceinline__ uint4 spark_m4_dense_load128(const uint8_t *p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ float spark_m4_dense_warp_sum(float value) {
    value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 16u));
    value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 8u));
    value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 4u));
    value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 2u));
    value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 1u));
    return value;
}

__device__ __forceinline__ uint8_t spark_m4_byte(const uint4 v, uint32_t i) {
    const uint32_t word = i >> 2;
    const uint32_t shift = (i & 3u) * 8u;
    const uint32_t w = word == 0u ? v.x : word == 1u ? v.y : word == 2u ? v.z : v.w;
    return (uint8_t)((w >> shift) & 0xffu);
}

extern "C" __global__ void b12x_mxfp8_dense_warp_sm121(
    const uint8_t *__restrict__ a_fp8,
    const uint8_t *__restrict__ a_scales,
    const uint8_t *__restrict__ w_fp8,
    const uint8_t *__restrict__ w_scales,
    __nv_bfloat16 *__restrict__ out,
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t inner,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t lane = threadIdx.x % SPARK_M4_DENSE_WARP_LANES;
    const uint32_t warp = threadIdx.x / SPARK_M4_DENSE_WARP_LANES;
    const uint32_t col = blockIdx.x * SPARK_M4_DENSE_WARP_WARPS + warp;
    const uint32_t row_base = blockIdx.y * SPARK_M4_DENSE_WARP_ROWS;
    const uint32_t k_blocks = inner / SPARK_M4_QUANT_BLOCK;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);
    if (col >= cols) {
        if (threadIdx.x == 0u) {
            SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, 0u, t_enter);
        }
        return;
    }

    float acc[SPARK_M4_DENSE_WARP_ROWS];
#pragma unroll
    for (uint32_t r = 0u; r < SPARK_M4_DENSE_WARP_ROWS; ++r) {
        acc[r] = 0.0f;
    }

    const uint32_t groups = (inner + SPARK_M4_DENSE_WARP_GROUP - 1u) / SPARK_M4_DENSE_WARP_GROUP;
    for (uint32_t group = 0u; group < groups; ++group) {
        const uint32_t base = group * SPARK_M4_DENSE_WARP_GROUP +
                              lane * SPARK_M4_DENSE_WARP_BYTES_PER_LANE;
        const bool valid = base < inner;
        uint4 w_vec = {0u, 0u, 0u, 0u};
        uint32_t block = 0u;
        if (valid) {
            block = base / SPARK_M4_QUANT_BLOCK;
            w_vec = spark_m4_dense_load128(w_fp8 + spark_b12x_weight_index(col, base, inner));
        }
        const float w_scale = valid ? spark_ue8m0_decode(
                                          w_scales[spark_b12x_scale_index(col, base, inner)])
                                    : 0.0f;
        for (uint32_t rr = 0u; rr < SPARK_M4_DENSE_WARP_ROWS; ++rr) {
            const uint32_t row = row_base + rr;
            if (row >= rows) {
                continue;
            }
            float local = 0.0f;
            if (valid) {
                const uint4 a_vec =
                    spark_m4_dense_load128(a_fp8 + (uint64_t)row * inner + base);
                const float a_scale = spark_ue8m0_decode(
                    a_scales[(uint64_t)row * k_blocks + block]);
                for (uint32_t i = 0u; i < SPARK_M4_DENSE_WARP_BYTES_PER_LANE; ++i) {
                    const float a_value =
                        spark_fp8_e4m3_decode(spark_m4_byte(a_vec, i)) * a_scale;
                    const float w_value =
                        spark_fp8_e4m3_decode(spark_m4_byte(w_vec, i)) * w_scale;
                    local = __fadd_rn(local, __fmul_rn(a_value, w_value));
                }
            }
            const float total = spark_m4_dense_warp_sum(local);
            if (lane == 0u) {
                acc[rr] = __fadd_rn(acc[rr], total);
            }
        }
    }

    if (lane == 0u) {
        for (uint32_t rr = 0u; rr < SPARK_M4_DENSE_WARP_ROWS; ++rr) {
            const uint32_t row = row_base + rr;
            if (row < rows) {
                out[(uint64_t)row * cols + col] = __float2bfloat16_rn(acc[rr]);
            }
        }
    }
    if (threadIdx.x == 0u) {
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, 0u, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}
__device__ __forceinline__ uint16_t spark_m4_u16_pair(const uint4 v, uint32_t i) {
    const uint32_t word = i < 4u ? v.x : i < 8u ? v.y : i < 12u ? v.z : v.w;
    return (uint16_t)((word >> ((i & 3u) * 8u)) & 0xffffu);
}

__device__ __forceinline__ void spark_m4_fp8x2_to_f32(uint16_t two, float &lo, float &hi) {
    asm("{\n\t"
        ".reg .b32 h2;\n\t"
        ".reg .b16 lo, hi;\n\t"
        "cvt.rn.f16x2.e4m3x2 h2, %2;\n\t"
        "mov.b32 {lo, hi}, h2;\n\t"
        "cvt.f32.f16 %0, lo;\n\t"
        "cvt.f32.f16 %1, hi;\n\t"
        "}"
        : "=f"(lo), "=f"(hi)
        : "h"(two));
}

// A-reuse tile with one thread per (row, column): 1024 threads = 8 rows x 128 columns. The
// activation stage is still shared and read once per block; every thread accumulates its own
// output in the scalar K-ascending order. The weight row is loaded per thread but the eight row
// threads of a column hit the same 32-byte line in L1, so DRAM weight traffic stays one read.
#define SPARK_M4_DENSE_TILE2_THREADS 1024u
#define SPARK_M4_DENSE_TILE2_COLUMNS 128u
#define SPARK_M4_DENSE_TILE2_ROWS 8u
#define SPARK_M4_DENSE_TILE2_STAGE 4096u

static_assert(SPARK_M4_DENSE_TILE2_THREADS == SPARK_M4_DENSE_TILE2_COLUMNS * SPARK_M4_DENSE_TILE2_ROWS,
              "one thread per row-column output of the tile");

extern "C" __global__ void b12x_mxfp8_dense_tile_sm121(
    const uint8_t *__restrict__ a_fp8,
    const uint8_t *__restrict__ a_scales,
    const uint8_t *__restrict__ w_fp8,
    const uint8_t *__restrict__ w_scales,
    __nv_bfloat16 *__restrict__ out,
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t inner,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t col = blockIdx.x * SPARK_M4_DENSE_TILE2_COLUMNS +
                         (threadIdx.x % SPARK_M4_DENSE_TILE2_COLUMNS);
    const uint32_t row_base = blockIdx.y * SPARK_M4_DENSE_TILE2_ROWS;
    const uint32_t row = row_base + threadIdx.x / SPARK_M4_DENSE_TILE2_COLUMNS;
    const uint32_t rows_in_chunk = rows - row_base < SPARK_M4_DENSE_TILE2_ROWS
                                       ? rows - row_base
                                       : SPARK_M4_DENSE_TILE2_ROWS;
    const uint32_t k_blocks = inner / SPARK_M4_QUANT_BLOCK;
    const bool live = row < row_base + rows_in_chunk && col < cols;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    __shared__ __align__(16) uint8_t a_tile[SPARK_M4_DENSE_TILE2_ROWS][SPARK_M4_DENSE_TILE2_STAGE];
    __shared__ uint8_t a_scale_tile[SPARK_M4_DENSE_TILE2_ROWS]
                                    [SPARK_M4_DENSE_TILE2_STAGE / SPARK_M4_QUANT_BLOCK];

    float acc = 0.0f;
    const uint32_t vectors_per_row = SPARK_M4_DENSE_TILE2_STAGE / 16u;
    for (uint32_t stage_base = 0u; stage_base < inner;
         stage_base += SPARK_M4_DENSE_TILE2_STAGE) {
        const uint32_t stage_end = stage_base + SPARK_M4_DENSE_TILE2_STAGE < inner
                                       ? stage_base + SPARK_M4_DENSE_TILE2_STAGE
                                       : inner;
        const uint32_t stage_bytes = stage_end - stage_base;
        const uint32_t stage_vectors = stage_bytes / 16u;
        const uint32_t stage_k_blocks = stage_bytes / SPARK_M4_QUANT_BLOCK;
        for (uint32_t t = threadIdx.x; t < rows_in_chunk * vectors_per_row;
             t += SPARK_M4_DENSE_TILE2_THREADS) {
            const uint32_t r = t / vectors_per_row;
            const uint32_t off = (t % vectors_per_row) * 16u;
            if (off < stage_vectors * 16u) {
                const uint4 v = spark_m4_dense_load128(
                    a_fp8 + (uint64_t)(row_base + r) * inner + stage_base + off);
                *reinterpret_cast<uint4 *>(&a_tile[r][off]) = v;
            }
        }
        for (uint32_t t = threadIdx.x; t < rows_in_chunk * stage_k_blocks;
             t += SPARK_M4_DENSE_TILE2_THREADS) {
            const uint32_t r = t / stage_k_blocks;
            const uint32_t b = t % stage_k_blocks;
            a_scale_tile[r][b] = a_scales[(uint64_t)(row_base + r) * k_blocks +
                                          stage_base / SPARK_M4_QUANT_BLOCK + b];
        }
        __syncthreads();

        for (uint32_t b = 0u; b < stage_k_blocks; ++b) {
            const uint32_t global_k = stage_base + b * SPARK_M4_QUANT_BLOCK;
            const uint32_t local_row = row - row_base;
            const float a_scale =
                live ? spark_ue8m0_decode(a_scale_tile[local_row][b]) : 0.0f;
            float w_scale = 0.0f;
            uint4 w0 = {0u, 0u, 0u, 0u};
            uint4 w1 = {0u, 0u, 0u, 0u};
            if (live) {
                const uint64_t w_at = spark_b12x_weight_index(col, global_k, inner);
                w0 = spark_m4_dense_load128(w_fp8 + w_at);
                w1 = spark_m4_dense_load128(w_fp8 + w_at + 16u);
                w_scale = spark_ue8m0_decode(
                    w_scales[spark_b12x_scale_index(col, global_k, inner)]);
            }
            if (live) {
#pragma unroll
                for (uint32_t i = 0u; i < 32u; i += 2u) {
                    const uint16_t a_two = *reinterpret_cast<const uint16_t *>(
                        &a_tile[local_row][b * SPARK_M4_QUANT_BLOCK + i]);
                    const uint16_t w_two = spark_m4_u16_pair(i < 16u ? w0 : w1, i & 15u);
                    float a_lo = 0.0f;
                    float a_hi = 0.0f;
                    float w_lo = 0.0f;
                    float w_hi = 0.0f;
                    spark_m4_fp8x2_to_f32(a_two, a_lo, a_hi);
                    spark_m4_fp8x2_to_f32(w_two, w_lo, w_hi);
                    acc = __fadd_rn(acc, __fmul_rn(a_lo * a_scale, w_lo * w_scale));
                    acc = __fadd_rn(acc, __fmul_rn(a_hi * a_scale, w_hi * w_scale));
                }
            }
        }
        __syncthreads();
    }

    if (live) {
        out[(uint64_t)row * cols + col] = __float2bfloat16_rn(acc);
    }
    if (threadIdx.x == 0u) {
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, 0u, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}// ---------------------------------------------------------------------------------------------
// K1-c(a) parallel-K dense tile. One block owns 8 rows x 32 output columns and FOUR K slices: 1024
// threads = 8 rows x 32 columns x 4 slices. Each slice walks a contiguous quarter of the reduction
// in ascending K order, so each partial is the scalar order over its range; the four partials are
// then combined with a deterministic tree ((p0+p1)+(p2+p3)). This is a declared tolerance path,
// NOT bit-identical: the reduction order differs from the scalar oracle by construction.
#define SPARK_M4_DENSE_PK_ROWS 8u
#define SPARK_M4_DENSE_PK_COLUMNS 32u
#define SPARK_M4_DENSE_PK_SPLIT 4u
#define SPARK_M4_DENSE_PK_THREADS (SPARK_M4_DENSE_PK_ROWS * SPARK_M4_DENSE_PK_COLUMNS * SPARK_M4_DENSE_PK_SPLIT)
#define SPARK_M4_DENSE_PK_STAGE 4096u

static_assert(SPARK_M4_DENSE_PK_THREADS == 1024u, "the parallel-K block is 1024 threads");
static_assert(SPARK_M4_DENSE_PK_STAGE % SPARK_M4_QUANT_BLOCK == 0u, "stages are whole scale blocks");

extern "C" __global__ void b12x_mxfp8_dense_par_k_sm121(
    const uint8_t *__restrict__ a_fp8,
    const uint8_t *__restrict__ a_scales,
    const uint8_t *__restrict__ w_fp8,
    const uint8_t *__restrict__ w_scales,
    __nv_bfloat16 *__restrict__ out,
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t inner,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t tid = threadIdx.x;
    const uint32_t slice = tid & (SPARK_M4_DENSE_PK_SPLIT - 1u);
    const uint32_t col_lane = (tid >> 2) & (SPARK_M4_DENSE_PK_COLUMNS - 1u);
    const uint32_t row_lane = tid >> 7;
    const uint32_t row_base = blockIdx.y * SPARK_M4_DENSE_PK_ROWS;
    const uint32_t row = row_base + row_lane;
    const uint32_t col = blockIdx.x * SPARK_M4_DENSE_PK_COLUMNS + col_lane;
    const uint32_t rows_in_chunk = rows - row_base < SPARK_M4_DENSE_PK_ROWS
                                       ? rows - row_base
                                       : SPARK_M4_DENSE_PK_ROWS;
    const bool live = row < row_base + rows_in_chunk && col < cols;
    const uint32_t k_blocks = inner / SPARK_M4_QUANT_BLOCK;
    const uint32_t slice_k_blocks = k_blocks / SPARK_M4_DENSE_PK_SPLIT;
    const uint32_t slice_first_block = slice * slice_k_blocks;
    const uint32_t slice_last_block = slice_first_block + slice_k_blocks;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    __shared__ __align__(16) uint8_t a_tile[SPARK_M4_DENSE_PK_ROWS][SPARK_M4_DENSE_PK_STAGE];
    __shared__ uint8_t a_scale_tile[SPARK_M4_DENSE_PK_ROWS]
                                   [SPARK_M4_DENSE_PK_STAGE / SPARK_M4_QUANT_BLOCK];
    __shared__ float partials[SPARK_M4_DENSE_PK_ROWS][SPARK_M4_DENSE_PK_COLUMNS]
                              [SPARK_M4_DENSE_PK_SPLIT];

    float acc = 0.0f;
    const uint32_t vectors_per_row = SPARK_M4_DENSE_PK_STAGE / 16u;
    for (uint32_t stage_base = 0u; stage_base < inner;
         stage_base += SPARK_M4_DENSE_PK_STAGE) {
        const uint32_t stage_end = stage_base + SPARK_M4_DENSE_PK_STAGE < inner
                                       ? stage_base + SPARK_M4_DENSE_PK_STAGE
                                       : inner;
        const uint32_t stage_bytes = stage_end - stage_base;
        const uint32_t stage_vectors = stage_bytes / 16u;
        const uint32_t stage_k_blocks = stage_bytes / SPARK_M4_QUANT_BLOCK;
        for (uint32_t t = tid; t < rows_in_chunk * vectors_per_row;
             t += SPARK_M4_DENSE_PK_THREADS) {
            const uint32_t r = t / vectors_per_row;
            const uint32_t off = (t % vectors_per_row) * 16u;
            if (off < stage_vectors * 16u) {
                const uint4 v = spark_m4_dense_load128(
                    a_fp8 + (uint64_t)(row_base + r) * inner + stage_base + off);
                *reinterpret_cast<uint4 *>(&a_tile[r][off]) = v;
            }
        }
        for (uint32_t t = tid; t < rows_in_chunk * stage_k_blocks;
             t += SPARK_M4_DENSE_PK_THREADS) {
            const uint32_t r = t / stage_k_blocks;
            const uint32_t b = t % stage_k_blocks;
            a_scale_tile[r][b] = a_scales[(uint64_t)(row_base + r) * k_blocks +
                                          stage_base / SPARK_M4_QUANT_BLOCK + b];
        }
        __syncthreads();

        const uint32_t stage_first_block = stage_base / SPARK_M4_QUANT_BLOCK;
        const uint32_t first = slice_first_block > stage_first_block
                                   ? slice_first_block
                                   : stage_first_block;
        const uint32_t last = slice_last_block < stage_first_block + stage_k_blocks
                                  ? slice_last_block
                                  : stage_first_block + stage_k_blocks;
        for (uint32_t b = first; b < last; ++b) {
            const uint32_t local_b = b - stage_first_block;
            const float a_scale = live ? spark_ue8m0_decode(a_scale_tile[row_lane][local_b]) : 0.0f;
            float w_scale = 0.0f;
            uint4 w0 = {0u, 0u, 0u, 0u};
            uint4 w1 = {0u, 0u, 0u, 0u};
            if (live) {
                const uint32_t global_k = b * SPARK_M4_QUANT_BLOCK;
                const uint64_t w_at = spark_b12x_weight_index(col, global_k, inner);
                w0 = spark_m4_dense_load128(w_fp8 + w_at);
                w1 = spark_m4_dense_load128(w_fp8 + w_at + 16u);
                w_scale = spark_ue8m0_decode(
                    w_scales[spark_b12x_scale_index(col, global_k, inner)]);
            }
            if (live) {
#pragma unroll
                for (uint32_t i = 0u; i < 32u; i += 2u) {
                    const uint16_t a_two = *reinterpret_cast<const uint16_t *>(
                        &a_tile[row_lane][local_b * SPARK_M4_QUANT_BLOCK + i]);
                    const uint16_t w_two = spark_m4_u16_pair(i < 16u ? w0 : w1, i & 15u);
                    float a_lo = 0.0f;
                    float a_hi = 0.0f;
                    float w_lo = 0.0f;
                    float w_hi = 0.0f;
                    spark_m4_fp8x2_to_f32(a_two, a_lo, a_hi);
                    spark_m4_fp8x2_to_f32(w_two, w_lo, w_hi);
                    acc = __fadd_rn(acc, __fmul_rn(a_lo * a_scale, w_lo * w_scale));
                    acc = __fadd_rn(acc, __fmul_rn(a_hi * a_scale, w_hi * w_scale));
                }
            }
        }
        __syncthreads();
    }

    partials[row_lane][col_lane][slice] = acc;
    __syncthreads();
    if (slice == 0u && live) {
        const float p0 = partials[row_lane][col_lane][0];
        const float p1 = partials[row_lane][col_lane][1];
        const float p2 = partials[row_lane][col_lane][2];
        const float p3 = partials[row_lane][col_lane][3];
        const float sum = __fadd_rn(__fadd_rn(p0, p1), __fadd_rn(p2, p3));
        out[(uint64_t)row * cols + col] = __float2bfloat16_rn(sum);
    }
    if (tid == 0u) {
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, 0u, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}
// ---------------------------------------------------------------------------------------------
// K1-c(b) split-K dense: each block owns 8 rows x 32 columns and ONE contiguous K quarter
// (blockIdx.z in 0..3), writing an f32 partial. A second tiny kernel combines the four partials in
// ascending z with a deterministic f32 chain and rounds once. Declared tolerance path: the K range
// is contiguous per block, but the final sum is (z0+z1+z2+z3) in order, not the scalar K-ascending
// chain, so this is NOT bit-identical to the oracle.
#define SPARK_M4_DENSE_SK_ROWS 8u
#define SPARK_M4_DENSE_SK_COLUMNS 32u
#define SPARK_M4_DENSE_SK_SPLIT 4u
#define SPARK_M4_DENSE_SK_THREADS (SPARK_M4_DENSE_SK_ROWS * SPARK_M4_DENSE_SK_COLUMNS)
#define SPARK_M4_DENSE_SK_STAGE 4096u

static_assert(SPARK_M4_DENSE_SK_THREADS == 256u, "the split-K partial block is 256 threads");
static_assert(SPARK_M4_DENSE_SK_STAGE % SPARK_M4_QUANT_BLOCK == 0u, "stages are whole scale blocks");

extern "C" __global__ void b12x_mxfp8_dense_split_k_sm121(
    const uint8_t *__restrict__ a_fp8,
    const uint8_t *__restrict__ a_scales,
    const uint8_t *__restrict__ w_fp8,
    const uint8_t *__restrict__ w_scales,
    float *__restrict__ partials, // [split][rows][cols] f32
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t inner,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t tid = threadIdx.x;
    const uint32_t col_lane = tid & (SPARK_M4_DENSE_SK_COLUMNS - 1u);
    const uint32_t row_lane = tid >> 5;
    const uint32_t row_base = blockIdx.y * SPARK_M4_DENSE_SK_ROWS;
    const uint32_t row = row_base + row_lane;
    const uint32_t col = blockIdx.x * SPARK_M4_DENSE_SK_COLUMNS + col_lane;
    const uint32_t rows_in_chunk = rows - row_base < SPARK_M4_DENSE_SK_ROWS
                                       ? rows - row_base
                                       : SPARK_M4_DENSE_SK_ROWS;
    const bool live = row < row_base + rows_in_chunk && col < cols;
    const uint32_t k_blocks = inner / SPARK_M4_QUANT_BLOCK;
    const uint32_t split = blockIdx.z;
    const uint32_t split_blocks = k_blocks / SPARK_M4_DENSE_SK_SPLIT;
    const uint32_t first_block = split * split_blocks;
    const uint32_t last_block = first_block + split_blocks;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    __shared__ __align__(16) uint8_t a_tile[SPARK_M4_DENSE_SK_ROWS][SPARK_M4_DENSE_SK_STAGE];
    __shared__ uint8_t a_scale_tile[SPARK_M4_DENSE_SK_ROWS]
                                   [SPARK_M4_DENSE_SK_STAGE / SPARK_M4_QUANT_BLOCK];
    float acc = 0.0f;
    const uint32_t vectors_per_row = SPARK_M4_DENSE_SK_STAGE / 16u;
    for (uint32_t stage_base = 0u; stage_base < inner; stage_base += SPARK_M4_DENSE_SK_STAGE) {
        const uint32_t stage_end = stage_base + SPARK_M4_DENSE_SK_STAGE < inner
                                       ? stage_base + SPARK_M4_DENSE_SK_STAGE
                                       : inner;
        const uint32_t stage_bytes = stage_end - stage_base;
        const uint32_t stage_vectors = stage_bytes / 16u;
        const uint32_t stage_k_blocks = stage_bytes / SPARK_M4_QUANT_BLOCK;
        for (uint32_t t = tid; t < rows_in_chunk * vectors_per_row;
             t += SPARK_M4_DENSE_SK_THREADS) {
            const uint32_t r = t / vectors_per_row;
            const uint32_t off = (t % vectors_per_row) * 16u;
            if (off < stage_vectors * 16u) {
                const uint4 v = spark_m4_dense_load128(
                    a_fp8 + (uint64_t)(row_base + r) * inner + stage_base + off);
                *reinterpret_cast<uint4 *>(&a_tile[r][off]) = v;
            }
        }
        for (uint32_t t = tid; t < rows_in_chunk * stage_k_blocks;
             t += SPARK_M4_DENSE_SK_THREADS) {
            const uint32_t r = t / stage_k_blocks;
            const uint32_t b = t % stage_k_blocks;
            a_scale_tile[r][b] = a_scales[(uint64_t)(row_base + r) * k_blocks +
                                          stage_base / SPARK_M4_QUANT_BLOCK + b];
        }
        __syncthreads();
        const uint32_t stage_first_block = stage_base / SPARK_M4_QUANT_BLOCK;
        const uint32_t first = first_block > stage_first_block ? first_block : stage_first_block;
        const uint32_t last = last_block < stage_first_block + stage_k_blocks
                                  ? last_block
                                  : stage_first_block + stage_k_blocks;
        for (uint32_t b = first; b < last; ++b) {
            const uint32_t local_b = b - stage_first_block;
            if (!live) {
                continue;
            }
            const float a_scale = spark_ue8m0_decode(a_scale_tile[row_lane][local_b]);
            const uint32_t global_k = b * SPARK_M4_QUANT_BLOCK;
            const uint64_t w_at = spark_b12x_weight_index(col, global_k, inner);
            const uint4 w0 = spark_m4_dense_load128(w_fp8 + w_at);
            const uint4 w1 = spark_m4_dense_load128(w_fp8 + w_at + 16u);
            const float w_scale =
                spark_ue8m0_decode(w_scales[spark_b12x_scale_index(col, global_k, inner)]);
#pragma unroll
            for (uint32_t i = 0u; i < 32u; i += 2u) {
                const uint16_t a_two = *reinterpret_cast<const uint16_t *>(
                    &a_tile[row_lane][local_b * SPARK_M4_QUANT_BLOCK + i]);
                const uint16_t w_two = spark_m4_u16_pair(i < 16u ? w0 : w1, i & 15u);
                float a_lo = 0.0f;
                float a_hi = 0.0f;
                float w_lo = 0.0f;
                float w_hi = 0.0f;
                spark_m4_fp8x2_to_f32(a_two, a_lo, a_hi);
                spark_m4_fp8x2_to_f32(w_two, w_lo, w_hi);
                acc = __fadd_rn(acc, __fmul_rn(a_lo * a_scale, w_lo * w_scale));
                acc = __fadd_rn(acc, __fmul_rn(a_hi * a_scale, w_hi * w_scale));
            }
        }
        __syncthreads();
    }
    if (live) {
        partials[((uint64_t)split * rows + row) * cols + col] = acc;
    }
    if (tid == 0u) {
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, 0u, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}

extern "C" __global__ void b12x_mxfp8_dense_split_k_combine_sm121(
    const float *__restrict__ partials,
    __nv_bfloat16 *__restrict__ out,
    const uint32_t rows,
    const uint32_t cols) {
    const uint32_t total = rows * cols;
    for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
         index += gridDim.x * blockDim.x) {
        const float p0 = partials[index];
        const float p1 = partials[(uint64_t)rows * cols + index];
        const float p2 = partials[2u * (uint64_t)rows * cols + index];
        const float p3 = partials[3u * (uint64_t)rows * cols + index];
        out[index] = __float2bfloat16_rn(__fadd_rn(__fadd_rn(p0, p1), __fadd_rn(p2, p3)));
    }
}

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
        for (uint32_t dj = 0; dj < SPARK_M4_DENSE_COLS_PER_THREAD; ++dj) {
            const uint32_t col = col_base + dj;
            if (col >= cols) {
                continue;
            }
            float acc = 0.0f;
            for (uint32_t block = 0; block < k_blocks; ++block) {
                const uint32_t base = block * SPARK_M4_QUANT_BLOCK;
                const float a_scale = spark_ue8m0_decode(a_scales[(uint64_t)row * k_blocks + block]);
                const float w_scale =
                    spark_ue8m0_decode(w_scales[spark_b12x_scale_index(col, base, inner)]);
                for (uint32_t i = 0; i < SPARK_M4_QUANT_BLOCK; ++i) {
                    const float a_value =
                        spark_fp8_e4m3_decode(a_fp8[(uint64_t)row * inner + base + i]) * a_scale;
                    const float w_value = spark_fp8_e4m3_decode(
                                              w_fp8[spark_b12x_weight_index(col, base + i, inner)]) *
                                          w_scale;
                    acc += a_value * w_value;
                }
            }
            out[(uint64_t)row * cols + col] = __float2bfloat16_rn(acc);
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

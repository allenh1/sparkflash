// kernels/b12x_bf16a_mxfp8w_dense_sm121.cu
//
// The BF16-activation variant of the dense MXFP8 GEMM: same weights, same layout, same tile and
// grid geometry, same trace ABI as b12x_mxfp8_dense_sm121.cu, with the ACTIVATION operand in
// __nv_bfloat16 instead of E4M3 + ue8m0 block scales. This is the D37 precision-first path for
// the attention projections (docs/DECISIONS.md D37, docs/build/phase3-attn-precision.md): the
// activation quantiser is removed from the GEMM's input, while the weights stay exactly the
// B12x FP8 codes and ue8m0 scales the checkpoint ships.
//
// Arithmetic contract, stated as the twin implements it:
//
//     out[row][col] = __float2bfloat16_rn(
//         sum over k ascending in 32-value blocks, ascending inside a block,
//             of __bfloat162float(a_bf16[row][k]) * (fp8(w[col][k]) * ue8m0(w_scale)))
//
// The bf16 activation widens to f32 exactly, so there is no activation rounding inside this
// kernel; each product is one f32 multiply, each block contributes one f32 add, and the single
// rounding is the final __float2bfloat16_rn. That is the same f32 accumulation order as
// b12x_mxfp8_dense_sm121.cu; src/bf16a_mxfp8w_ref.rs is the definition of that order and
// examples/m21_bf16a_projections.rs measures the device against it on the same packed bytes.
//
// Layout, geometry and trace are shared with the MXFP8 twin and are not restated here: the weight
// byte layout is m4_common_sm121.cuh's B12x helpers (one ue8m0 scale per 32x32 block), a block
// computes one 64x64 output tile with 256 threads owning 4x4 sub-tiles, and the trace record uses
// TRACE_SCOPE_DENSE_MXFP8 with the same payload_meta (tile elements * 2 bytes). The only changed
// operand is a_bf16: [rows][inner] BF16, no activation scale buffer.
//
// Verification status at this commit: nvcc V13.0.88 compiles it for sm_121a and the compiled form
// is pinned in baseline/kernel_baseline.txt; clang parses it against the stub headers; the typed
// launch module pins the declaration; the twin and its tests are in src/bf16a_mxfp8w_ref.rs. It
// has NOT yet executed on a GPU: examples/m21_bf16a_projections.rs is the harness that does that
// in a Lead-owned window, comparing every output bit against the twin. Nothing here claims a
// tensor-core path or a latency.

#include "m4_common_sm121.cuh"

// The B12x micro-tile shape and the two index helpers live in m4_common_sm121.cuh. The geometry
// macros are named for THIS kernel so the combined build.rs probe can compile it together with
// the MXFP8 twin without redefining the twin's macros to different bodies.
#define SPARK_M4_BF16A_DENSE_TILE_M 64u
#define SPARK_M4_BF16A_DENSE_TILE_N 64u
#define SPARK_M4_BF16A_DENSE_THREADS 256u
#define SPARK_M4_BF16A_DENSE_ROWS_PER_THREAD 4u
#define SPARK_M4_BF16A_DENSE_COLS_PER_THREAD 4u
#define SPARK_M4_BF16A_DENSE_SUBTILES_PER_ROW \
    (SPARK_M4_BF16A_DENSE_TILE_N / SPARK_M4_BF16A_DENSE_COLS_PER_THREAD)

static_assert(SPARK_B12X_KTILE == SPARK_M4_QUANT_BLOCK, "a B12x tile is one MX scale block wide");
static_assert(SPARK_M4_BF16A_DENSE_TILE_N % SPARK_B12X_NTILE == 0u,
              "the N tile is whole micro-tiles");
static_assert(SPARK_M4_BF16A_DENSE_TILE_M * SPARK_M4_BF16A_DENSE_TILE_N ==
                  SPARK_M4_BF16A_DENSE_THREADS * SPARK_M4_BF16A_DENSE_ROWS_PER_THREAD *
                      SPARK_M4_BF16A_DENSE_COLS_PER_THREAD,
              "the 64x64 tile is exactly covered by 256 threads of 4x4");
static_assert(SPARK_M4_BF16A_DENSE_TILE_N % SPARK_M4_BF16A_DENSE_COLS_PER_THREAD == 0u,
              "the sub-tiles tile the N dimension");
static_assert(SPARK_M4_BF16A_DENSE_TILE_M % SPARK_M4_BF16A_DENSE_ROWS_PER_THREAD == 0u,
              "the sub-tiles tile the M dimension");
static_assert(SPARK_M4_BF16A_DENSE_THREADS % 32u == 0u, "a block is a whole number of warps");

extern "C" __global__ void b12x_bf16a_mxfp8w_dense_sm121(
    const __nv_bfloat16 *__restrict__ a_bf16,  // [rows][inner] BF16 activations
    const uint8_t *__restrict__ w_fp8,         // B12x tiled E4M3 weights
    const uint8_t *__restrict__ w_scales,      // B12x tiled ue8m0 weight scales
    __nv_bfloat16 *__restrict__ out,           // [rows][cols] bf16 result
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t inner,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row_base = blockIdx.x * SPARK_M4_BF16A_DENSE_TILE_M +
                              (lane / SPARK_M4_BF16A_DENSE_SUBTILES_PER_ROW) *
                                  SPARK_M4_BF16A_DENSE_ROWS_PER_THREAD;
    const uint32_t col_base =
        blockIdx.y * SPARK_M4_BF16A_DENSE_TILE_N +
        (lane % SPARK_M4_BF16A_DENSE_SUBTILES_PER_ROW) * SPARK_M4_BF16A_DENSE_COLS_PER_THREAD;
    const uint32_t k_blocks = inner / SPARK_M4_QUANT_BLOCK;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    // The accumulation order per output is the contract src/bf16a_mxfp8w_ref.rs implements: K
    // ascending in 32-value blocks and ascending within a block. The 4x4 sub-tile only decides
    // which outputs a thread owns; it never reorders a sum, so the reference does not model the
    // tiling.
    for (uint32_t di = 0; di < SPARK_M4_BF16A_DENSE_ROWS_PER_THREAD; ++di) {
        const uint32_t row = row_base + di;
        if (row >= rows) {
            continue;
        }
        for (uint32_t dj = 0; dj < SPARK_M4_BF16A_DENSE_COLS_PER_THREAD; ++dj) {
            const uint32_t col = col_base + dj;
            if (col >= cols) {
                continue;
            }
            float acc = 0.0f;
            for (uint32_t block = 0; block < k_blocks; ++block) {
                const uint32_t base = block * SPARK_M4_QUANT_BLOCK;
                const float w_scale =
                    spark_ue8m0_decode(w_scales[spark_b12x_scale_index(col, base, inner)]);
                for (uint32_t i = 0; i < SPARK_M4_QUANT_BLOCK; ++i) {
                    // bf16 -> f32 is exact; the activation enters the sum unrounded.
                    const float a_value =
                        __bfloat162float(a_bf16[(uint64_t)row * inner + base + i]);
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
        // payload_meta: one bf16 per output element of this block's tile, as in the MXFP8 twin.
        const uint32_t meta = SPARK_M4_BF16A_DENSE_TILE_M * SPARK_M4_BF16A_DENSE_TILE_N * 2u;
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}

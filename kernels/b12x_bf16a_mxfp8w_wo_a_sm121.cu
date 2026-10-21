// kernels/b12x_bf16a_mxfp8w_wo_a_sm121.cu
//
// The BF16-activation variant of the grouped output projection wo_a: the grouped twin of
// b12x_bf16a_mxfp8w_dense_sm121.cu and the D37 precision-first path for K6 (docs/DECISIONS.md
// D37, docs/build/phase3-attn-precision.md). Weights are exactly the B12x FP8 codes and ue8m0
// scales that b12x_mxfp8_wo_a_sm121.cu reads; the activation operand is __nv_bfloat16 instead of
// E4M3 codes plus per-32 ue8m0 scales.
//
// Arithmetic contract, stated as the twin implements it:
//
//     lo_out[group][col] = __float2bfloat16_rn(
//         sum over k ascending in 32-value blocks, ascending inside a block,
//             of __bfloat162float(attn_bf16[group][k]) * (fp8(w[col][k]) * ue8m0(w_scale)))
//
// The group is a grid dimension, not a GEMM shape: each group still has its own weight buffer
// [o_lora_rank][group_inner] and its own scale buffer, whose sizes differ and therefore keep two
// different group strides. The reduction width is 4096 (8 heads x 512 head dim), o_lora_rank is
// 1024, one block computes one 256-wide output-channel tile with 256 threads, and the trace record
// is TRACE_SCOPE_DENSE_MXFP8 with payload_meta = tile * 2 - all identical to the MXFP8 wo_a
// kernel, because this variant changes only the activation dtype.
//
// What is NOT claimed: GPU execution has not happened at this commit. nvcc V13.0.88 compiles it
// for sm_121a and the compiled form is pinned in baseline/kernel_baseline.txt; the typed launch
// module pins the declaration; src/bf16a_mxfp8w_ref.rs defines the arithmetic and
// examples/m21_bf16a_projections.rs measures the device against it in a Lead-owned window. No
// tensor-core path and no latency claim.

#include "m4_common_sm121.cuh"

#define SPARK_M4_BF16A_WOA_THREADS 256u
#define SPARK_M4_BF16A_WOA_TILE 256u
// Heads per group: 64 heads over 8 groups.
#define SPARK_M4_BF16A_WOA_HEADS_PER_GROUP (SPARK_M4_HEADS / SPARK_M4_O_GROUPS)
// The reduction width of one group: its heads' 512-dim outputs concatenated.
#define SPARK_M4_BF16A_WOA_GROUP_INNER \
    (SPARK_M4_BF16A_WOA_HEADS_PER_GROUP * SPARK_M4_HEAD_DIM)

static_assert(SPARK_M4_BF16A_WOA_HEADS_PER_GROUP == 8u, "64 heads over 8 groups is 8 heads per group");
static_assert(SPARK_M4_BF16A_WOA_GROUP_INNER == 4096u, "the group reduction width is 4096");
static_assert(SPARK_M4_BF16A_WOA_GROUP_INNER % SPARK_M4_QUANT_BLOCK == 0u, "the width tiles by 32");
static_assert(SPARK_M4_O_LORA_RANK % SPARK_M4_BF16A_WOA_TILE == 0u,
              "1024 output channels tile by 256");
static_assert(SPARK_M4_BF16A_WOA_THREADS == SPARK_M4_BF16A_WOA_TILE,
              "one output element per thread");
static_assert(SPARK_M4_O_GROUPS == 8u && SPARK_M4_O_GROUPS_PER_RANK == 2u, "8 groups, 2 per rank");
static_assert(SPARK_M4_O_LORA_RANK % SPARK_B12X_NTILE == 0u,
              "the output rank is whole B12x micro-tile rows, so nothing is padded away");
static_assert(SPARK_M4_BF16A_WOA_GROUP_INNER % SPARK_B12X_KTILE == 0u,
              "the group reduction width is whole B12x micro-tiles");
// The same two-stride pin as the MXFP8 wo_a kernel: a group's scale buffer is 32 times smaller
// than its weight buffer, so one shared stride would walk a scale buffer off its end.
static_assert(SPARK_M4_O_LORA_RANK / SPARK_B12X_NTILE == 32u,
              "1024 output channels are 32 micro-tile rows of scales");
static_assert(SPARK_M4_BF16A_WOA_GROUP_INNER / SPARK_B12X_KTILE == 128u,
              "4096 reduction values are 128 scale blocks along K");
static_assert(SPARK_B12X_TILE_BYTES == 1024u, "a 32x32 block is 1024 weight bytes, one scale");

extern "C" __global__ void b12x_bf16a_mxfp8w_wo_a_sm121(
    const __nv_bfloat16 *__restrict__ attn_bf16,  // [groups][group_inner] BF16 attention outputs
    const uint8_t *__restrict__ w_fp8,            // B12x tiled [groups][o_lora_rank][group_inner]
    const uint8_t *__restrict__ w_scales,         // B12x tiled ue8m0 scales
    __nv_bfloat16 *__restrict__ lo_out,           // [groups][o_lora_rank] bf16 result
    const uint32_t groups,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t group = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t column = blockIdx.y * SPARK_M4_BF16A_WOA_TILE + lane;
    const uint32_t k_blocks = SPARK_M4_BF16A_WOA_GROUP_INNER / SPARK_M4_QUANT_BLOCK;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    __shared__ __nv_bfloat16 s_a_row[SPARK_M4_BF16A_WOA_GROUP_INNER];
    if (group < groups) {
        const __nv_bfloat16 *a_row =
            attn_bf16 + (uint64_t)group * SPARK_M4_BF16A_WOA_GROUP_INNER;
        for (uint32_t k = lane; k < (uint32_t)SPARK_M4_BF16A_WOA_GROUP_INNER;
             k += SPARK_M4_BF16A_WOA_THREADS) {
            s_a_row[k] = a_row[k];
        }
    }
    __syncthreads();

    if (group < groups && column < (uint32_t)SPARK_M4_O_LORA_RANK) {
        // Each group owns a weight buffer AND a scale buffer, and the two are different sizes, so
        // there are two group strides. The offsets inside each buffer come from the same helpers the
        // dense kernel uses.
        const uint64_t group_weight_base = (uint64_t)group * spark_b12x_buffer_bytes(
                                                           SPARK_M4_O_LORA_RANK,
                                                           SPARK_M4_BF16A_WOA_GROUP_INNER);
        const uint64_t group_scale_base = (uint64_t)group * spark_b12x_scale_bytes(
                                                          SPARK_M4_O_LORA_RANK,
                                                          SPARK_M4_BF16A_WOA_GROUP_INNER);
        float acc = 0.0f;
        for (uint32_t block = 0; block < k_blocks; ++block) {
            const uint32_t base = block * SPARK_M4_QUANT_BLOCK;
            const float w_scale = spark_ue8m0_decode(
                w_scales[group_scale_base +
                         spark_b12x_scale_index(column, base, SPARK_M4_BF16A_WOA_GROUP_INNER)]);
#pragma unroll 8
            for (uint32_t i = 0; i < SPARK_M4_QUANT_BLOCK; ++i) {
                // bf16 -> f32 is exact; the activation enters the sum unrounded.
                const float a_value = __bfloat162float(s_a_row[base + i]);
                const float w_value =
                    spark_fp8_e4m3_decode(
                        w_fp8[group_weight_base +
                              spark_b12x_weight_index(column, base + i,
                                                      SPARK_M4_BF16A_WOA_GROUP_INNER)]) *
                    w_scale;
                acc += a_value * w_value;
            }
        }
        lo_out[(uint64_t)group * SPARK_M4_O_LORA_RANK + column] = __float2bfloat16_rn(acc);
    }
    __syncthreads();

    if (lane == 0u) {
        const uint32_t meta = SPARK_M4_BF16A_WOA_TILE * 2u;
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}

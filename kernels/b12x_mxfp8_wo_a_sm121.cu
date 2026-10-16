// kernels/b12x_mxfp8_wo_a_sm121.cu
//
// The grouped output projection wo_a of specification 2.1/6.2, MXFP8 like the dense twin:
// 8 output groups (2 per rank) with o_lora_rank 1024, each group consuming the concatenated
// 512-dim outputs of its 8 attention heads (8 * 512 = 4096 reduction width), weights in the
// B12x layout with one ue8m0 scale per 32 values and an f32 accumulator.
//
// Why this is a separate kernel from b12x_mxfp8_dense_sm121.cu rather than a call into it: the
// group is a grid dimension, not a GEMM shape, and each group has its own weight matrix width. The
// arithmetic is deliberately identical, so mxfp8_ref.rs's accumulation contract covers both, and
// mxfp8_ref.rs's wo_a test asserts the grouped result equals a per-group dense GEMM over the same
// slice. The CUDA index helpers themselves are compared against their Rust twins value for value
// by tests/b12x_index_parity.rs, which compiles the helper block out of the shared header.
//
// Each group has TWO buffers with different sizes: weights of [o_lora_rank][group_inner] and scales
// of one ue8m0 byte per 32x32 block, ceil(o_lora_rank/32) * (group_inner/32). The two group
// strides are therefore different numbers and this kernel must not share one between them; see the
// group_weight_base and group_scale_base below.
//
// Verification status: never executed. nvcc 12.9.1 compiles it for sm_121a, clang parses it
// against the stub headers, src/m4_abi.rs pins the signature, the group geometry and the shared
// index helpers, and src/mxfp8_ref.rs defines and tests the numerics on the CPU. No tensor-core
// path is claimed.

#include "m4_common_sm121.cuh"

#define SPARK_M4_WOA_THREADS 256u
#define SPARK_M4_WOA_TILE 256u
// Heads per group: 64 heads over 8 groups.
#define SPARK_M4_WOA_HEADS_PER_GROUP (SPARK_M4_HEADS / SPARK_M4_O_GROUPS)
// The reduction width of one group: its heads' 512-dim outputs concatenated.
#define SPARK_M4_WOA_GROUP_INNER (SPARK_M4_WOA_HEADS_PER_GROUP * SPARK_M4_HEAD_DIM)

static_assert(SPARK_M4_WOA_HEADS_PER_GROUP == 8u, "64 heads over 8 groups is 8 heads per group");
static_assert(SPARK_M4_WOA_GROUP_INNER == 4096u, "the group reduction width is 4096");
static_assert(SPARK_M4_WOA_GROUP_INNER % SPARK_M4_QUANT_BLOCK == 0u, "the width tiles by 32");
static_assert(SPARK_M4_O_LORA_RANK % SPARK_M4_WOA_TILE == 0u, "1024 output channels tile by 256");
static_assert(SPARK_M4_WOA_THREADS == SPARK_M4_WOA_TILE, "one output element per thread");
static_assert(SPARK_M4_O_GROUPS == 8u && SPARK_M4_O_GROUPS_PER_RANK == 2u, "8 groups, 2 per rank");
static_assert(SPARK_M4_O_LORA_RANK % SPARK_B12X_NTILE == 0u,
              "the output rank is whole B12x micro-tile rows, so nothing is padded away");
static_assert(SPARK_M4_WOA_GROUP_INNER % SPARK_B12X_KTILE == 0u,
              "the group reduction width is whole B12x micro-tiles");
// The scale grid of one group is ceil(o_lora_rank / 32) * (group_inner / 32) bytes, which is 32
// times smaller than the weight block: a group stride shared between the two would overrun the
// scale buffer by 31 group strides, which is the defect this pin exists to prevent returning.
static_assert(SPARK_M4_O_LORA_RANK / SPARK_B12X_NTILE == 32u,
              "1024 output channels are 32 micro-tile rows of scales");
static_assert(SPARK_M4_WOA_GROUP_INNER / SPARK_B12X_KTILE == 128u,
              "4096 reduction values are 128 scale blocks along K");
static_assert(SPARK_B12X_TILE_BYTES == 1024u, "a 32x32 block is 1024 weight bytes, one scale");

extern "C" __global__ void b12x_mxfp8_wo_a_sm121(
    const uint8_t *__restrict__ attn_fp8,     // [groups][group_inner] E4M3 attention outputs
    const uint8_t *__restrict__ attn_scales,  // [groups][group_inner / 32] ue8m0
    const uint8_t *__restrict__ w_fp8,        // B12x tiled [groups][o_lora_rank][group_inner]
    const uint8_t *__restrict__ w_scales,     // B12x tiled ue8m0 scales
    __nv_bfloat16 *__restrict__ lo_out,       // [groups][o_lora_rank] bf16 result
    const uint32_t groups,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    const uint32_t group = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t column = blockIdx.y * SPARK_M4_WOA_TILE + lane;
    const uint32_t k_blocks = SPARK_M4_WOA_GROUP_INNER / SPARK_M4_QUANT_BLOCK;

    uint64_t t_enter = 0u;
    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_DENSE_MXFP8, t_enter);

    if (group < groups && column < (uint32_t)SPARK_M4_O_LORA_RANK) {
        const uint8_t *a_row = attn_fp8 + (uint64_t)group * SPARK_M4_WOA_GROUP_INNER;
        const uint8_t *a_scale_row = attn_scales + (uint64_t)group * k_blocks;
        // Each group owns a weight buffer AND a scale buffer, and the two are different sizes, so
        // there are two group strides. The offsets inside each buffer come from the same helpers the
        // dense kernel uses.
        const uint64_t group_weight_base =
            (uint64_t)group * spark_b12x_buffer_bytes(SPARK_M4_O_LORA_RANK, SPARK_M4_WOA_GROUP_INNER);
        const uint64_t group_scale_base =
            (uint64_t)group * spark_b12x_scale_bytes(SPARK_M4_O_LORA_RANK, SPARK_M4_WOA_GROUP_INNER);
        float acc = 0.0f;
        for (uint32_t block = 0; block < k_blocks; ++block) {
            const uint32_t base = block * SPARK_M4_QUANT_BLOCK;
            const float a_scale = spark_ue8m0_decode(a_scale_row[block]);
            const float w_scale =
                spark_ue8m0_decode(w_scales[group_scale_base +
                                            spark_b12x_scale_index(column, base, SPARK_M4_WOA_GROUP_INNER)]);
            for (uint32_t i = 0; i < SPARK_M4_QUANT_BLOCK; ++i) {
                const float a_value = spark_fp8_e4m3_decode(a_row[base + i]) * a_scale;
                const float w_value =
                    spark_fp8_e4m3_decode(
                        w_fp8[group_weight_base +
                              spark_b12x_weight_index(column, base + i, SPARK_M4_WOA_GROUP_INNER)]) *
                    w_scale;
                acc += a_value * w_value;
            }
        }
        lo_out[(uint64_t)group * SPARK_M4_O_LORA_RANK + column] = __float2bfloat16_rn(acc);
    }
    __syncthreads();

    if (lane == 0u) {
        const uint32_t meta = SPARK_M4_WOA_TILE * 2u;
        SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, meta, t_enter);
        (void)step_seq;
    }
    (void)t_enter;
}

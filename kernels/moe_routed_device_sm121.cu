// kernels/moe_routed_device_sm121.cu
//
// The two pieces the layer port's device-only `moe_routed` stage was missing. The expert arithmetic
// already runs on device through five landed launches (routed_gate_up_sm121 twice,
// swiglu_mxfp8_quant_weighted_sm121, b12x_mxfp4_moe_stored_sm121), but those launches consume
// CONTIGUOUS per-slot slices and they leave the last two steps on the host:
//
//   * staging: the five landed launches index their weight operand by the staged slot
//     (`weight_row = expert * width + column`), so the selected experts' resident arena slices must
//     be gathered into slot order first. The host arm does it with per-slot cuMemcpyDtoD, which is
//     a PCIe-free but host-issued copy per layer per step; on the device-only path the ids live in
//     `workspace.moe_ids` and the host must not read them, so the gather has to be a kernel.
//   * combine: after the stored down projection the f32 per-slot accumulators are still on device,
//     and the host arm reads them back (`copy_dtoh`) to run `mxfp4_ref::moe_combine_routed_f32`
//     plus its bf16 narrow. That host combine is the P3 blocker. Both arithmetic steps are
//     transcribed here.
//
// This file adds NO new numeric rule beyond the two references it transcribes:
//
// 1. `moe_expert_stage_sm121` - one (layer, matrix) expert slice per routed row, gathered from a
//    device-resident pointer table of (weight, scales) addresses. Pure byte copy: the same bytes the
//    host arm's cuMemcpyDtoD moves. It also zeroes the f32 slot accumulator on request, because
//    b12x_mxfp4_moe_stored_sm121 ADDS into it and the host arm allocates a pooled buffer whose
//    contents are undefined.
//
// 2. `moe_routed_combine_sm121` - the host arm's post-stored-combine epilogue, on device:
//      for each token, order the six routed slots by ascending GLOBAL expert id (a selection sort
//      over the ids the router wrote, stable on equal ids: the first minimum in ascending slot
//      order wins);
//      round each slot's f32 rank accumulator to bf16 - the expert's w2 output dtype,
//      model.py:851-852 returns `self.w2(x.to(dtype))` - and widen it back;
//      add the six values in that order in f32 starting from 0.0f, with __fadd_rn;
//      write the f32 total (the routed sum the shared expert add consumes, "moe_delta") and its
//      bf16 narrow (the pre-reduce payload the bf16 all-reduce consumes).
//    The two roundings are the references' own bit arithmetic, transcribed below; the per-slot round
//    is `mxfp4_ref::round_expert_row` (which quiets a NaN) and the payload narrow is the host arm's
//    `bf16_le_bytes` (which does not special-case a NaN total). They are different formulas only on
//    a NaN, and both are reproduced exactly rather than approximated with the CUDA intrinsic, so the
//    device row is bit-identical to the f32 reference row and to the payload the host arm shipped.
//
// The routing weight is NOT applied in either kernel: the host arm multiplies it into the activation
// BEFORE the quantiser (swiglu_mxfp8_quant_weighted_sm121), which is the model's order, and this
// path keeps that operand. Nothing here re-weights a combined row.
//
// Out of scope by task split: the 640/576 column padding of the routed gate/up/SwiGLU operands is
// the S4 concern. This file reproduces the incumbent ABI exactly - gate/up emit 640 bf16 columns of
// which 576..639 are zero, SwiGLU consumes all 640, the stored down projection reads the 640-wide
// code row but only its 20 real scale blocks - so a later padding change moves the landed kernels'
// geometry, not this staging or this combine.

#include "m4_common_sm121.cuh"

// 256 threads: the staging kernel copies one row per block and the combine kernel owns one 256-wide
// column tile per block. Both are launch-geometry constants, pinned by the Rust ABI module.
#define SPARK_MOE_ROUTED_THREADS 256u
// The column tile the combine kernel's grid uses: hidden / tile blocks per token.
#define SPARK_MOE_ROUTED_TILE_COLS 256u
// The pointer table stores (weight, scales) per (layer, matrix, expert), so one entry is two words.
#define SPARK_MOE_ROUTED_TABLE_STRIDE 2u
// A bf16 payload word is two bytes.
#define SPARK_MOE_ROUTED_BF16_BYTES 2u
// The copy fast path moves 16 bytes per thread iteration, which requires 16-byte alignment: every
// arena tensor is 256-byte aligned (sparkflash_device::arena::ARENA_ALIGNMENT) and every staged slice
// is a multiple of 16 bytes at the released geometry, so the fast path is the only path the release
// takes. The two fallbacks keep a geometry change from becoming a misaligned access.
#define SPARK_MOE_ROUTED_COPY_ALIGN 16u
// The routed-row ceiling one stage call may gather, and the size of the host-side staging plan. It is
// a capacity bound, not a model constant: the released decode is one token (six rows) and a batch of
// `MAX_ROUTED_ROWS / 6` tokens still fits.
#define SPARK_MOE_ROUTED_MAX_ROWS 256u

static_assert(SPARK_M4_MOE_TOP_K == 6u,
              "moe_routed_combine_sm121 orders the router's six routed slots by expert id");
static_assert(SPARK_M4_HIDDEN % SPARK_MOE_ROUTED_TILE_COLS == 0u,
              "the combine kernel tiles the hidden row by its block width");
static_assert(SPARK_MOE_ROUTED_MAX_ROWS >= SPARK_M4_MOE_TOP_K,
              "at least one token's routed rows fit the staging plan");
static_assert(SPARK_MOE_ROUTED_COPY_ALIGN % 4u == 0u,
              "the typed copy fallback is a whole number of 32-bit words");

// The reference's per-slot round to bf16, transcribed from mxfp4_ref::round_expert_row: round to
// nearest even via the half-up-plus-lsb trick, and a NaN is quieted with its sign and top payload bit
// kept (half::bf16::from_f32's rule). For every finite value this equals __float2bfloat16_rn; the
// integer form is kept because it is the one the reference states, and the NaN arm of the reference
// is not the intrinsic's.
__device__ __forceinline__ uint32_t spark_moe_routed_round_slot_bits(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return (bits >> 16) | 0x0040u;
    }
    return (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
}

// The host arm's bf16 narrow of the fused f32 total (crates/sparkflash-cli's bf16_le_bytes): the same
// integer round as above WITHOUT the NaN special case, because the payload path spells
// `(bits + 0x7fff + lsb) & 0xffff_0000` and that is what the reference computes for every value,
// including a NaN total. Untyped as a bf16 word so no dtype conversion can reorder the bits.
__device__ __forceinline__ uint16_t spark_moe_routed_narrow_bits(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (uint16_t)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

// Copies one staged slice with the widest typed path its size and alignment allow. `bytes == 0` is a
// no-op (a zero-sized tensor would otherwise loop on a zero count, which is still correct but the
// guard makes the intent explicit).
__device__ __forceinline__ void spark_moe_routed_copy(const uint8_t *__restrict__ src,
                                                     uint8_t *__restrict__ dst,
                                                     uint32_t bytes) {
    const uint32_t tid = threadIdx.x;
    const uint32_t stride = blockDim.x;
    if (bytes == 0u) {
        return;
    }
    if ((bytes & (SPARK_MOE_ROUTED_COPY_ALIGN - 1u)) == 0u) {
        const uint4 *source = reinterpret_cast<const uint4 *>(src);
        uint4 *target = reinterpret_cast<uint4 *>(dst);
        const uint32_t words = bytes / SPARK_MOE_ROUTED_COPY_ALIGN;
        for (uint32_t index = tid; index < words; index += stride) {
            target[index] = source[index];
        }
    } else if ((bytes & 3u) == 0u) {
        const uint32_t *source = reinterpret_cast<const uint32_t *>(src);
        uint32_t *target = reinterpret_cast<uint32_t *>(dst);
        const uint32_t words = bytes / 4u;
        for (uint32_t index = tid; index < words; index += stride) {
            target[index] = source[index];
        }
    } else {
        for (uint32_t index = tid; index < bytes; index += stride) {
            dst[index] = src[index];
        }
    }
}

// Gathers one expert slice per routed row from the resident pointer table.
//
// The pointer table is device memory the caller built at open from the resolved routed-expert arena:
// for each (layer, matrix, expert) it holds two 64-bit addresses, the weight slice and the scale grid
// slice. `table` here is the (layer, matrix) slice's base, so entry `id` is `table[2 * id]`
// (weight) and `table[2 * id + 1]` (scales). The table is why this kernel is on device: the host
// never learns an address or an id.
//
// `expert_ids` is `[rows]` in routed-row order (row = token * top_k + slot), which is
// `workspace.moe_ids` at the released top-6 geometry. Every id is a global expert index in
// `[0, experts)`; the pinned router (kernels/moe_router_sm121.cu, `moe_router_topk_sm121`) selects
// six DISTINCT ids from `[0, n_routed)` by scanning that range, so an out-of-range id is a corrupted
// workspace rather than a routing outcome. The clamp to 0 is therefore unreachable for the pinned
// producer; it exists because `table[id]` is an address, and a corrupted id must not become an
// arbitrary device read. The Rust launcher refuses `experts` different from the table's expert count.
//
// `slot_acc` is zeroed when `acc_row_floats != 0`, one row of `acc_row_floats` floats per routed row.
// The caller passes the f32 width of the down projection (hidden) on the launch that precedes the
// stored combine and 0 on the two gate/up staging launches, so the zeroing costs nothing where it is
// not needed and cannot race the accumulator.
extern "C" __global__ void moe_expert_stage_sm121(
    const uint64_t *__restrict__ table,       // [experts][2] (weight, scales) for THIS (layer, matrix)
    const uint32_t *__restrict__ expert_ids,  // [rows] global expert ids, routed-row order
    uint8_t *__restrict__ dst_weight,         // [rows][weight_bytes] staged packed weights
    uint8_t *__restrict__ dst_scales,         // [rows][scale_bytes] staged ue8m0 grid
    float *__restrict__ slot_acc,             // [rows][acc_row_floats], zeroed when nonzero
    const uint32_t rows,
    const uint32_t experts,
    const uint32_t weight_bytes,
    const uint32_t scale_bytes,
    const uint32_t acc_row_floats) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const uint32_t id = expert_ids[row] < experts ? expert_ids[row] : 0u;
    const uint64_t entry = (uint64_t)id * SPARK_MOE_ROUTED_TABLE_STRIDE;
    const uint8_t *src_weight =
        reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(table[entry]));
    const uint8_t *src_scales =
        reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(table[entry + 1u]));
    spark_moe_routed_copy(src_weight, dst_weight + (uint64_t)row * weight_bytes, weight_bytes);
    spark_moe_routed_copy(src_scales, dst_scales + (uint64_t)row * scale_bytes, scale_bytes);
    if (acc_row_floats != 0u) {
        float *acc = slot_acc + (uint64_t)row * acc_row_floats;
        for (uint32_t index = threadIdx.x; index < acc_row_floats; index += blockDim.x) {
            acc[index] = 0.0f;
        }
    }
}

// The routed combine epilogue on device: the host arm's `mxfp4_ref::moe_combine_routed_f32` followed
// by its `bf16_le_bytes` narrow.
//
// `slot_acc` is `[tokens * top_k][hidden]` in routed-row order and `expert_ids` is `[tokens * top_k]`
// of GLOBAL expert ids. The host arm feeds the reference the SLOT indices instead, but its slots are
// numbered in ascending expert-id order (`moe_slots` sorts and dedups the selection before mapping),
// so ordering by slot index and ordering by global id are the same permutation. This kernel takes the
// ids because the device path never builds the host slot map, and sorts by the id directly - the same
// six rows in the same order, hence the same sequence of f32 additions.
//
// One thread owns one hidden column of one token, so the six-deep ordered add chain per column is the
// reference's order and nothing is reassociated. Both outputs are written from the same `total`: the
// f32 row (the pre-shared-expert routed sum, "moe_delta") and its bf16 payload word row.
extern "C" __global__ void moe_routed_combine_sm121(
    const float *__restrict__ slot_acc,      // [tokens * top_k][hidden] f32 rank accumulators
    const uint32_t *__restrict__ expert_ids, // [tokens * top_k] global ids from the router
    float *__restrict__ delta_f32,           // [tokens][hidden] the routed sum, unrounded f32
    uint16_t *__restrict__ delta_bf16,       // [tokens][hidden] the pre-reduce bf16 payload
    const uint32_t tokens,
    const uint32_t hidden) {
    const uint32_t token = blockIdx.x;
    const uint32_t column = blockIdx.y * blockDim.x + threadIdx.x;
    if (token >= tokens || column >= hidden) {
        return;
    }
    uint32_t ids[SPARK_M4_MOE_TOP_K];
#pragma unroll
    for (uint32_t slot = 0u; slot < (uint32_t)SPARK_M4_MOE_TOP_K; ++slot) {
        ids[slot] = expert_ids[(uint64_t)token * SPARK_M4_MOE_TOP_K + slot];
    }
    // Selection sort by ascending expert id, first minimum in ascending slot order: exactly
    // mxfp4_ref::moe_combine_slot_order. A bitmask is the "used" set, so the whole loop stays in
    // registers.
    uint32_t used = 0u;
    float total = 0.0f;
#pragma unroll
    for (uint32_t position = 0u; position < (uint32_t)SPARK_M4_MOE_TOP_K; ++position) {
        uint32_t best = SPARK_M4_MOE_TOP_K;
#pragma unroll
        for (uint32_t slot = 0u; slot < (uint32_t)SPARK_M4_MOE_TOP_K; ++slot) {
            if (((used >> slot) & 1u) != 0u) {
                continue;
            }
            if (best == (uint32_t)SPARK_M4_MOE_TOP_K || ids[slot] < ids[best]) {
                best = slot;
            }
        }
        used |= 1u << best;
        const uint64_t index = ((uint64_t)token * SPARK_M4_MOE_TOP_K + best) * hidden + column;
        // The expert's bf16 output, widened: the reference adds the widened bf16 value.
        const float expert_output =
            __uint_as_float(spark_moe_routed_round_slot_bits(slot_acc[index]) << 16);
        total = __fadd_rn(total, expert_output);
    }
    const uint64_t out_index = (uint64_t)token * hidden + column;
    delta_f32[out_index] = total;
    delta_bf16[out_index] = spark_moe_routed_narrow_bits(total);
}

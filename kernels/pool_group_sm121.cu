// kernels/pool_group_sm121.cu
//
// `pool_group_f32`: the compressor's ratio-axis softmax pooling, the host rule of
// crates/sparkflash-engine/src/forward/attention.rs:985-1020 (`pool_group`), moved one launch at a
// time onto the device so layers 2..39 stay device-only.
//
// WHY IT EXISTS. Above ratio 1 the released compressor computes two projections over the same rows -
// the KV latent (`compressor.wkv`) and the pooling gate (`compressor.wgate`) - and pools the first by
// a per-element softmax of the second over the ratio axis (`model.py:475, 482`). That pooling was the
// last host transcendental in the seam: it is one launch with one thread per latent element.
//
// EXACTNESS. The host rule and this kernel perform the same four steps in the same order per element:
// the maximum over the group, `exp(score - max)`, the ascending sum into `total`, and an ascending
// weighted sum of the kv rows by `weight / total` (division per row, never a reciprocal). The only
// difference is the exponential itself: `expf` on the device is not bit-identical to the host libm's,
// so the pooling is a DECLARED DEVIATION gated by D54, not a bit-identical oracle - the same class
// the certified head's sigmoid exponential already carries (docs/build/s1-result.md §7.1(3)). An
// all-`-inf` group (an empty slot) contributes zero, as the host rule states.
//
// GEOMETRY. One thread per latent element (`head_dim` threads), `ceil(head_dim / 128)` blocks of 128.
// Each thread reads the same element of every row: `score[row * head_dim + element]` and
// `kv[row * head_dim + element]`, ascending row, touching only its own output element. No allocation,
// no readback, no synchronisation, no trace ring (it is not a stage; the launching stage records the
// trace).
//
// PIN. No cubin bytes or digest live here: registration in baseline/kernel_baseline.txt is the Lead's
// step and lands with the stage that launches it
// (crates/sparkflash-kernels/src/pool_group_launch.rs resolves the symbol by name).

#include "m4_common_sm121.cuh"

// CUDART_INF_F, the +-inf constant the empty-group rule compares against, comes from
// <math_constants.h> under nvcc. The clang syntax probe runs -nocudainc against cuda_stub/, which
// has no such header, so the stub branch spells the same value with the compiler builtin.
#if defined(SPARKFLASH_CUDA_SYNTAX_STUB)
#define CUDART_INF_F __builtin_inff()
#else
#include <math_constants.h>
#endif

// Threads per block: the launcher's THREADS and this constant must agree.
#define SPARK_POOL_GROUP_THREADS 128u

extern "C" __global__ void pool_group_f32(
    const float *__restrict__ kv,
    const float *__restrict__ score,
    float *__restrict__ dest,
    const int ratio,
    const int head_dim) {
  const int element = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (element >= head_dim) {
    return;
  }
  float max = -CUDART_INF_F;
  for (int row = 0; row < ratio; ++row) {
    const float value = score[static_cast<size_t>(row) * head_dim + element];
    max = fmaxf(max, value);
  }
  if (!isfinite(max)) {
    // An empty group: the host rule writes zero, and so does this kernel.
    dest[element] = 0.0f;
    return;
  }
  float total = 0.0f;
  float acc = 0.0f;
  // Two ascending passes: the host rule stores the weights, sums them, then sums the weighted rows.
  // Recomputing the exponential in the second pass yields the same value (same inputs, same order).
  for (int row = 0; row < ratio; ++row) {
    total += expf(score[static_cast<size_t>(row) * head_dim + element] - max);
  }
  for (int row = 0; row < ratio; ++row) {
    const float weight =
        expf(score[static_cast<size_t>(row) * head_dim + element] - max);
    const float normalised = total > 0.0f ? weight / total : 0.0f;
    acc += normalised * kv[static_cast<size_t>(row) * head_dim + element];
  }
  dest[element] = acc;
}

// Indexer k_norm (attention.rs:2897-2900): takes f32 keys [cols] from indexer.wk
// (Mode::Bf16WeightF32Stores), bf16 weight [cols] (`indexer.k_norm.weight`), computes
// sequential ascending sum of squares (__fmul_rn, __fadd_rn), mean = __fdiv_rn(sum, cols),
// inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps))), and writes
// out[c] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x[c], inv), __bfloat162float(weight[c]))).
extern "C" __global__ void k_norm_f32_to_bf16(
    const float *__restrict__ x,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ out,
    const uint32_t cols,
    const float eps) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || cols == 0u) {
    return;
  }
  float sum = 0.0f;
  for (uint32_t c = 0u; c < cols; ++c) {
    const float val = x[c];
    sum = __fadd_rn(sum, __fmul_rn(val, val));
  }
  const float mean = __fdiv_rn(sum, static_cast<float>(cols));
  const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
  for (uint32_t c = 0u; c < cols; ++c) {
    const float w = __bfloat162float(weight[c]);
    out[c] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x[c], inv), w));
  }
}

// E8M0 ceiling scale code for a positive normal product (quant.rs:435-454).
__device__ __forceinline__ uint8_t spark_pool_e8m0_encode_ceil(float value) {
  if (!(value > 0.0f)) {
    return 0u;
  }
  const uint32_t bits = spark_float_bits(value);
  const uint32_t field = (bits >> 23) & 0xFFu;
  if (field == 0xFFu) {
    return 255u;
  }
  const uint32_t mantissa = bits & 0x007FFFFFu;
  int32_t exponent = static_cast<int32_t>(field) - 127 + (mantissa != 0u ? 1 : 0);
  int32_t code = exponent + 127;
  if (code < 0) {
    code = 0;
  }
  if (code > 254) {
    code = 254;
  }
  return static_cast<uint8_t>(code);
}

// Indexer FP4 E2M1 + UE8M0 power-of-two round trip (attention.rs:215-267, 1571-1577, 2916-2922):
// one thread per 32-value block (`blocks` blocks total).
extern "C" __global__ void fp4_round_trip_bf16_to_f32(
    const __nv_bfloat16 *__restrict__ src,
    float *__restrict__ dst,
    const uint32_t blocks) {
  const uint32_t blk = blockIdx.x * blockDim.x + threadIdx.x;
  if (blk >= blocks) {
    return;
  }
  const uint32_t base = blk * 32u;
  float amax = 0.0f;
  for (uint32_t i = 0u; i < 32u; ++i) {
    const float v = fabsf(__bfloat162float(src[base + i]));
    amax = v > amax ? v : amax;
  }
  const float floor_val = 6.0f * spark_bits_float(0x00800000u);
  const float floored = amax > floor_val ? amax : floor_val;
  const float max_inv = 1.0f / 6.0f;
  const uint8_t code = spark_pool_e8m0_encode_ceil(__fmul_rn(floored, max_inv));
  const float scale = spark_ue8m0_decode(code);
  for (uint32_t i = 0u; i < 32u; ++i) {
    const float v = __bfloat162float(src[base + i]);
    const float scaled = fminf(fmaxf(__fdiv_rn(v, scale), -6.0f), 6.0f);
    const uint8_t nibble = spark_fp4_e2m1_encode_fallback(scaled);
    dst[base + i] = __fmul_rn(spark_fp4_e2m1_decode(nibble), scale);
  }
}

// Indexer per-head weight scaling (attention.rs:2815-2818):
// dst[i] = bf16_round(bf16_to_f32(src[i]) * factor).
extern "C" __global__ void scale_index_weights_bf16_to_f32(
    const __nv_bfloat16 *__restrict__ src,
    float *__restrict__ dst,
    const float factor,
    const uint32_t count) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  const float val = __bfloat162float(src[idx]);
  dst[idx] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(val, factor)));
}

// Device tail hc_pre collapse (tp4_forward.rs:8504-8527):
// collapses [hc_mult][dim] f32 stream with [hc_mult] f32 pre_mix in ascending copy order
// and narrows to [dim] bf16, while zeroing *certified_count on thread 0.
extern "C" __global__ void hc_pre_collapse_bf16(
    const float *__restrict__ pre_mix,
    const float *__restrict__ stream,
    __nv_bfloat16 *__restrict__ out_bf16,
    uint32_t *__restrict__ certified_count,
    const uint32_t hc_mult,
    const uint32_t dim) {
  const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col == 0u && certified_count != nullptr) {
    *certified_count = 0u;
  }
  if (col >= dim) {
    return;
  }
  float acc = 0.0f;
  for (uint32_t copy = 0u; copy < hc_mult; ++copy) {
    acc = __fadd_rn(
        acc, __fmul_rn(pre_mix[copy], stream[static_cast<size_t>(copy) * dim + col]));
  }
  out_bf16[col] = __float2bfloat16_rn(acc);
}

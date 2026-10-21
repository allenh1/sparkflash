// kernels/engram_gate_write_sm121.cu
//
// The Engram gate and residual write, in the released dtype rule (task-9, audit F4; released
// inference/model.py:355-365).
//
// THE REFERENCE, per token:
//
//     key = kv[:hc*dim].float(); value = kv[hc*dim:].float()        // kv is the bf16 wkv output
//     weight = q_weight.float() * k_weight.float()                  // only used as a product
//     h = x.float()                                                 // x is the bf16 residual
//     rstd = rsqrt(h.square().mean(-1) + eps) * rsqrt(key.square().mean(-1) + eps)
//     dot  = (h * weight * key).sum(-1) * rstd * dim**-0.5
//     gate = sigmoid(copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
//     out  = (h + gate.unsqueeze(-1) * value).to(x.dtype)           // one bf16 cast
//
// The stream operand is the bf16 residual, so the engine's host seam must round its f32
// representation before this kernel sees it: the kernel takes the bf16 tensor the released code
// computes on, and the f32 representation is the caller's conversion (bf16 -> f32 is exact).
//
// ORDER. The three per-copy sums walk the dimension ascending and accumulate with one f32 rounding
// per multiply and add (explicit _rn intrinsics, no contraction). The means, the two reciprocals of
// square roots, the rstd product, the dot scaling, the clamp, the signed square root and the write
// are the reference's operations in the reference's order. The gate's sigmoid is f32 expf, which is
// the one place this kernel and a host twin can differ in the last mantissa bit: the audit's F8
// class, and it is recorded as such in the twin test rather than claimed bit-exact.
//
// PARALLELISM. One thread per hc copy owns that copy's whole dot product (hc is 4 in the released
// model), so the reduction order is one ascending sequential sum, not a tree: a tree would sum in a
// different order and could change the bf16 result. After the barrier every thread writes a strided
// slice of the hc*dim output, one element per lane, each an independent f32 add and one bf16 cast.
//
// TRACE. ABI-uniform arguments, no event: no protocol scope describes the Engram write.

#include "m4_common_sm121.cuh"

#define SPARK_ENGRAM_MAX_HC 8u
#define SPARK_ENGRAM_CLAMP 1.0e-6f
#define SPARK_ENGRAM_GATE_THREADS 256u

// 1 / (1 + exp(-x)) in f32, the reference's logistic function (the F8 class noted above).
__device__ __forceinline__ float spark_engram_sigmoid(float value) {
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-value)));
}

extern "C" __global__ void engram_gate_write_sm121(
    const __nv_bfloat16 *__restrict__ kv,      // [hc*dim + dim] bf16 wkv output: key then value
    const __nv_bfloat16 *__restrict__ stream,  // [hc*dim] the bf16 residual x
    const float *__restrict__ product,         // [hc*dim] q_weight * k_weight, f32
    __nv_bfloat16 *__restrict__ out,           // [hc*dim] bf16 write result
    float *__restrict__ gates,                 // [hc] the gates, for the caller's record
    const uint32_t dim,
    const uint32_t hc,
    const float eps,
    GpuTraceRing *__restrict__ trace_ring,     // ABI uniformity only; see TRACE above
    const uint32_t layer_id,
    const uint64_t step_seq) {
    (void)trace_ring;
    (void)layer_id;
    (void)step_seq;

    __shared__ float gate_shared[SPARK_ENGRAM_MAX_HC];
    const uint32_t copy = threadIdx.x;
    if (copy < hc && copy < SPARK_ENGRAM_MAX_HC) {
        float stream_square = 0.0f;
        float key_square = 0.0f;
        float dot = 0.0f;
        for (uint32_t column = 0; column < dim; ++column) {
            const float h = __bfloat162float(stream[copy * dim + column]);
            const float k = __bfloat162float(kv[copy * dim + column]);
            stream_square = __fadd_rn(stream_square, __fmul_rn(h, h));
            key_square = __fadd_rn(key_square, __fmul_rn(k, k));
            dot = __fadd_rn(dot, __fmul_rn(__fmul_rn(h, product[copy * dim + column]), k));
        }
        const float dim_f = (float)dim;
        const float stream_inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(stream_square, dim_f), eps)));
        const float key_inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(key_square, dim_f), eps)));
        const float rstd = __fmul_rn(stream_inv, key_inv);
        const float scale = __fdiv_rn(1.0f, __fsqrt_rn(dim_f));
        const float scaled_dot = __fmul_rn(__fmul_rn(dot, rstd), scale);
        const float magnitude = scaled_dot < 0.0f ? -scaled_dot : scaled_dot;
        const float clamped = magnitude > SPARK_ENGRAM_CLAMP ? magnitude : SPARK_ENGRAM_CLAMP;
        const float root = __fsqrt_rn(clamped);
        const float signed_root = (spark_float_bits(scaled_dot) & 0x80000000u) != 0u ? -root : root;
        const float gate = spark_engram_sigmoid(signed_root);
        gate_shared[copy] = gate;
        gates[copy] = gate;
    }
    __syncthreads();

    const __nv_bfloat16 *value_row = kv + (uint64_t)hc * dim;
    for (uint32_t column = threadIdx.x; column < dim; column += blockDim.x) {
        const float value = __bfloat162float(value_row[column]);
        for (uint32_t which = 0u; which < hc && which < SPARK_ENGRAM_MAX_HC; ++which) {
            const uint32_t index = which * dim + column;
            const float h = __bfloat162float(stream[index]);
            out[index] = __float2bfloat16_rn(__fadd_rn(h, __fmul_rn(gate_shared[which], value)));
        }
    }
}

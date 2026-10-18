// kernels/m4_activation_quant_sm121.cuh
//
// The ACTIVATION quantiser's block scale: the reference's act_quant rule, which is a different
// rule from the checkpoint weight-grid scale in m4_common_sm121.cuh's spark_ue8m0_from_amax.
//
// WHY THIS IS NOT IN THE COMMON HEADER. The common header's bytes are the input to the pinned
// measured-text evidence (crates/sparkflash-kernels/src/m4_abi.rs, measured_texts): every stored
// text is one substitution of the shipped helper body, and every one of them is a compiled
// translation unit with a measured cubin quoted in that module's documentation. Changing one byte
// of the common header therefore invalidates that whole table. A helper that only the activation
// paths need lives here instead, so the common header's text stays frozen.
//
// THE RULE (kernel.py:36-37, 76-80; engine quant::e8m0_encode_ceil and
// quant::quantize_activation_fp8): floor the block's amax at 1e-4, multiply by 1/448 in f32, and
// round the exponent UP - one higher when the product's mantissa is nonzero - which is the
// smallest power of two at or above amax/448, so a scaled activation block never needs the
// encoder's saturation. The checkpoint weight-grid rule (floor(log2(amax)) - element_emax) is for
// stored weights only: on an activation it under-scales every block whose amax lies above
// 1.75 * 2^floor(log2 amax), and that block's E4M3 operand then differs from the engine's.
#pragma once

#include "m4_common_sm121.cuh"

// The largest finite E4M3 magnitude (kernel.py:44-46) and the reference's amax floor
// (kernel.py:76), spelled as the engine's quant.rs spells them.
#define SPARK_M4_ACT_E4M3_MAX 448.0f
#define SPARK_M4_ACT_QUANT_AMAX_FLOOR 1.0e-4f

// The ue8m0 code of the reference activation scale for one block whose largest magnitude is amax.
// The amax floor is inside this helper on purpose: a caller that forgot it would quantise an
// all-zero block into NaNs, which is the failure the floor exists to prevent.
__device__ __forceinline__ uint8_t spark_act_quant_scale_code(float amax) {
    const float floored =
        amax > SPARK_M4_ACT_QUANT_AMAX_FLOOR ? amax : SPARK_M4_ACT_QUANT_AMAX_FLOOR;
    const float product = floored * (1.0f / SPARK_M4_ACT_E4M3_MAX);
    if (!(product > 0.0f)) {
        return (uint8_t)SPARK_M4_UE8M0_BIAS; // unreachable with the 1e-4 floor, kept total
    }
    const uint32_t bits = spark_float_bits(product);
    const uint32_t exponent_field = (bits >> 23) & 0xFFu;
    if (exponent_field == 0xFFu) {
        return (uint8_t)SPARK_M4_UE8M0_MAX;
    }
    // Ceiling of log2 by exponent field plus one when the mantissa is nonzero: the reference's
    // fast_log2_ceil (kernel.py:22-27). The clamp spells the reachable ue8m0 range.
    int32_t exponent = (int32_t)exponent_field - 127 + ((bits & 0x7FFFFFu) != 0u ? 1 : 0);
    const int32_t ceiling = (int32_t)SPARK_M4_UE8M0_MAX - (int32_t)SPARK_M4_UE8M0_BIAS;
    const int32_t floor_code = -(int32_t)SPARK_M4_UE8M0_BIAS;
    if (exponent < floor_code) {
        exponent = floor_code;
    }
    if (exponent > ceiling) {
        exponent = ceiling;
    }
    return (uint8_t)(exponent + (int32_t)SPARK_M4_UE8M0_BIAS);
}

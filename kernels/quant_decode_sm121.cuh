// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Shared scalar decoders for the formats more than one model pack reads on the device:
//
//   * FP8 E4M3 (FN, no infinities; 0x7F and 0xFF are NaN, and exponent 15 with mantissa 0..6 is
//     a finite number up to 448). Used for MXFP8 weights, the DeepSeek Engram table, the Qwen4-Exp
//     PLE n-gram shards, and the 16-column block scales of NVFP4 weights.
//   * FP4 E2M1 (the sixteen-entry codebook of MXFP4 and NVFP4).
//
// Attribution:
//   - Block-scaled NVFP4 / MXFP4 / MXFP8 dequantization conventions on SM120/SM121 (GB10) follow
//     NVIDIA ModelOpt (nvidia/Qwen3.8-Flash-Next-NVFP4), b12x
//     (https://github.com/local-inference-lab/b12x, flashinfer/experimental/b12x, Apache-2.0;
//     Luke Alonso @lukealonso, @original-el8, and contributors), and NVIDIA Flare
//     (https://github.com/NVIDIA/flare-local-inference, Apache-2.0). See the root NOTICE file.
//
// Both decoders are branch-free and call-free: the power of two is built by placing the biased
// exponent straight into the IEEE-754 f32 exponent field instead of calling ldexpf. Keep this
// header self-contained (no shared-memory declarations, no kernel symbols) so any pack can include
// it without pulling in another model's ABI.
//
// Note on m4_common_sm121.cuh: that header predates the second model pack and its callers are
// sha256-pinned in crates/sparkflash-kernels/baseline/kernel_baseline.txt. Touching it requires a
// CUDA 13.0.88 baseline refresh on the target node; until that refresh is scheduled, the V4.1
// kernels keep their in-header copy and every new pack includes this file instead.

#ifndef SPARKFLASH_QUANT_DECODE_SM121_CUH
#define SPARKFLASH_QUANT_DECODE_SM121_CUH

// E4M3FN: bias 7, no infinity, NaN at (exponent == 15 && mantissa == 7).
__device__ __forceinline__ float spark_fp8_e4m3_decode(unsigned char bits) {
    const unsigned int b = (unsigned int)bits;
    const unsigned int sign = (b & 0x80u) << 24;
    const unsigned int exponent = (b >> 3) & 0x0Fu;
    const unsigned int mantissa = b & 0x07u;
    // 2^(exponent - 7) by exponent-field insertion; the exponent == 0 result is unused.
    const float power = __uint_as_float((exponent + 120u) << 23);
    const float normal = power * (1.0f + (float)mantissa * 0.125f);
    const float subnormal = (float)mantissa * 0.001953125f;
    const float magnitude = exponent == 0u ? subnormal : normal;
    const unsigned int finite = __float_as_uint(magnitude) | sign;
    return __uint_as_float((b & 0x7Fu) == 0x7Fu ? (sign | 0x7FC00000u) : finite);
}

// E2M1: the sixteen codes are 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives.
// The eight unsigned magnitudes are 0.5 * {0, 1, 2, 3, 4, 6, 8, 12}, packed as 4-bit nibbles in
// 0xC8643210u so decoding a nibble is a shift, mask, int-to-float, * 0.5f and sign-bit OR with
// zero branches or predicates.
__device__ __forceinline__ float spark_fp4_e2m1_decode(unsigned char nibble) {
    const unsigned int b = (unsigned int)nibble;
    const unsigned int sign = (b & 0x8u) << 28;
    const unsigned int shift = (b << 2) & 0x1Cu;
    const float magnitude = (float)((0xC8643210u >> shift) & 0xFu) * 0.5f;
    return __uint_as_float(__float_as_uint(magnitude) | sign);
}

#endif

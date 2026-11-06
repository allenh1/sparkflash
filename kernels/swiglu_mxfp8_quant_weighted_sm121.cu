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

// The WEIGHTED SwiGLU + MX-quantise epilogue: the routed experts' activation operand with the
// routing weight multiplied in BEFORE the quantiser, which is the model's order. This removes the
// host combine step the m17a chain carried (DECISIONS D32): the reference maps every value through
// bf16(value * routing) and then the activation quantiser, and that rounding is part of the class.
//
// The body, the asymmetric clamp, the block scale rule and the trace scope are
// kernels/swiglu_quant_sm121.cuh's; this file only supplies the routing-weight operand.

#include "swiglu_quant_sm121.cuh"

extern "C" __global__ void swiglu_mxfp8_quant_weighted_sm121(
    const __nv_bfloat16 *__restrict__ gate,   // [rows][cols] bf16, w1 x
    const __nv_bfloat16 *__restrict__ up,     // [rows][cols] bf16, w3 x
    const float *__restrict__ weights,        // [rows] one routing weight per routed row
    uint8_t *__restrict__ out_fp8,            // [rows][cols] E4M3, row stride cols
    uint8_t *__restrict__ out_scales,         // [rows][cols / 32] ue8m0, row stride cols / 32
    const uint32_t rows,
    const uint32_t cols,
    const float limit,
    GpuTraceRing *__restrict__ trace_ring,
    const uint32_t layer_id,
    const uint64_t step_seq) {
    swiglu_quant_body<true>(gate, up, weights, out_fp8, out_scales, rows, cols, limit, trace_ring,
                            layer_id, step_seq);
}

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

// kernels/inkling_softcap_sm121.cuh
//
// Inkling-2-Small's softcapped SwiGLU, the one expression both of its expert activations
// evaluate: the f32 epilogue kernel (inkling_softcapped_swiglu, kernels/inkling_epilogue_sm121.cu)
// and the grouped W4A16 MMA's gate_up epilogue (q4p_moe_gate_up_capped,
// kernels/qwen4_prefill_sm121.cu). It is self-contained - expf and the IEEE intrinsics, no
// shared memory, no kernel symbols - so either image includes it without the other's ABI.
//
//     out = softcap(gate, gate_cap) * sigmoid(gate) * softcap(up, up_cap)
//     softcap(x, cap) = tanh(x / cap) * cap, or x itself when cap is not positive

#ifndef SPARKFLASH_INKLING_SOFTCAP_SM121_CUH
#define SPARKFLASH_INKLING_SOFTCAP_SM121_CUH

// tanh through exp, so the epilogue depends only on expf and the same expression is available
// to the clang syntax probe's stub. Saturating the input keeps the exponent finite.
__device__ __forceinline__ float inkling_tanh(float value) {
    if (value > 20.0f) {
        return 1.0f;
    }
    if (value < -20.0f) {
        return -1.0f;
    }
    const float exponential = expf(2.0f * value);
    return __fdiv_rn(__fadd_rn(exponential, -1.0f), __fadd_rn(exponential, 1.0f));
}

__device__ __forceinline__ float inkling_sigmoid(float value) {
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-value)));
}

// softcap(x) = tanh(x / cap) * cap: the cap multiplies back, or the activation would be the tanh
// itself and the branch would lose its scale.
__device__ __forceinline__ float inkling_softcapped_swiglu_value(float gate, float up,
                                                                 float gate_cap, float up_cap) {
    const float gated = gate_cap > 0.0f ? inkling_tanh(gate / gate_cap) * gate_cap : gate;
    const float capped_up = up_cap > 0.0f ? inkling_tanh(up / up_cap) * up_cap : up;
    return __fmul_rn(__fmul_rn(gated, inkling_sigmoid(gate)), capped_up);
}

#endif

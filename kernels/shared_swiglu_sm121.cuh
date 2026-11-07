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

// kernels/shared_swiglu_sm121.cuh
//
// One SwiGLU epilogue: silu(gate) * up, with an optional clamp and an optional dynamic-q8
// output. The token image (q4e_swiglu), GLM's decode epilogue (q4e_swiglu_limited) and GLM's
// bulk prefill epilogues (glm_moe_swiglu and glm_moe_swiglu_q8) used to state the same
// expression four times. They keep their own indexing and quantisation, and call this once.
//
// Two activation spellings are preserved because they round differently: the single division
// `gate / (1 + e^-gate)` (Qwen's and the bulk prefill's) and `gate * sigmoid_ref(gate)`
// (GLM's decode epilogue, the linalg::silu form). A template flag picks the spelling, so
// unifying the body is not an arithmetic change.

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "q4e_common_sm121.cuh"

// The epilogue's value: clamp when the caller's form does, then the gate's silu times up.
// `kSigmoidRef` selects gate * sigmoid_ref(gate); otherwise gate / (1 + e^-gate).
template <bool kClamped, bool kSigmoidRef>
__device__ __forceinline__ float shared_swiglu_value(float gate, float up, float limit) {
    if constexpr (kClamped) {
        gate = fminf(gate, limit);
        up = fminf(fmaxf(up, -limit), limit);
    }
    const float activated =
        kSigmoidRef ? __fmul_rn(gate, q4e_sigmoid_ref(gate))
                    : gate / (1.0f + __expf(-gate));
    return __fmul_rn(activated, up);
}

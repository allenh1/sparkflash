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


// kernels/dense_format_sm121.cu
//
// DeepSeek's dense projections in every dense class the released MXFP8 bodies do not read, on the
// one shared bulk-prefill tile (shared_prefill_mma_sm121.cuh's dense mode, the body GLM's and
// Qwen's FP8 and BF16 projections run on). The body, the tile and each format's decode live in
// the header; this file only names the entries the base's projections launch:
//
//   * `a` is [groups][tokens][inner] f32 rows (a_tok_stride floats a token, a_grp_stride a
//     group), the seam's unquantised rows. The activation enters the tile as its BF16 head and
//     the BF16 of what the head leaves (kLow), as the host reference multiplies an unquantised
//     row.
//   * `w` is [groups][cols][inner] row-major weight bytes and `ws` its scale plane: one inverse
//     scale a 128 x 128 tile for the FP8 classes (f32 or BF16 words), none for BF16.
//   * `out` is [groups][tokens][cols] f32 (out_tok_stride floats a token).
//
// Grid (ceil(cols / GLM_BN), ceil(tokens / GLM_BM), groups), OM_THREADS threads. `cols` and
// `inner` are multiples of 32; the FP8 classes' scale plane is indexed by 128-wide tiles.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "opt_mma_common_sm121.cuh"
#include "opt_pdl_sm121.cuh"
#include "shared_prefill_mma_sm121.cuh"

// One dense projection: rows of `AT` (f32, or BF16 as stored) against the stored class, rows of
// `OutT` out (the f32 accumulator, or it rounded once to BF16). f32 rows are read as a BF16 head
// and remainder, so they lose nothing a BF16 row would keep. Group z (a grouped projection's
// head) is a*_grp_stride / out_grp_stride elements and w_grp_stride bytes / ws_grp_stride scales
// on.
template <int kFormat, typename ScaleT, typename AT, typename OutT>
__device__ __forceinline__ void df_project(const AT *__restrict__ a, const uint8_t *__restrict__ w,
                                           const void *__restrict__ ws, OutT *__restrict__ out,
                                           uint32_t cols, uint32_t inner, uint32_t tokens,
                                           uint64_t a_tok_stride, uint64_t out_tok_stride,
                                           uint64_t a_grp_stride, uint64_t out_grp_stride,
                                           uint64_t w_grp_stride, uint64_t ws_grp_stride) {
    constexpr bool kLow = std::is_same<AT, float>::value;
    shared_prefill_mma_body<kFormat, false, kLow, ScaleT, 2, 0, false, false, 0, AT, OutT>(
        a, w, ws, nullptr, nullptr, out, cols, inner, tokens, a_tok_stride, out_tok_stride,
        a_grp_stride, out_grp_stride, w_grp_stride, ws_grp_stride);
}

#define DF_ENTRY(name, format, scale_t, a_t, out_t)                                            \
    extern "C" __global__ __launch_bounds__(OM_THREADS) void name(                             \
        const a_t *__restrict__ a, const uint8_t *__restrict__ w,                              \
        const void *__restrict__ ws, out_t *__restrict__ out, uint32_t cols, uint32_t inner,   \
        uint32_t tokens, uint64_t a_tok_stride, uint64_t out_tok_stride,                       \
        uint64_t a_grp_stride, uint64_t out_grp_stride, uint64_t w_grp_stride,                 \
        uint64_t ws_grp_stride) {                                                              \
        OPT_PDL_ENTRY();                                                                       \
        df_project<format, scale_t, a_t, out_t>(a, w, ws, out, cols, inner, tokens,            \
                                                a_tok_stride, out_tok_stride, a_grp_stride,    \
                                                out_grp_stride, w_grp_stride, ws_grp_stride);  \
    }

// The four row forms (f32 or BF16 in, f32 or BF16 out) of each stored class: FP8 on 128 x 128
// tiles with an f32 or a BF16 inverse scale, and BF16 row-major.
#define DF_CLASS(io, a_t, out_t)                                                               \
    DF_ENTRY(dense_format_##io##_fp8_sm121, SPM_W_FP8, float, a_t, out_t)                       \
    DF_ENTRY(dense_format_##io##_fp8_bf16_sm121, SPM_W_FP8, __nv_bfloat16, a_t, out_t)          \
    DF_ENTRY(dense_format_##io##_bf16_sm121, SPM_W_BF16, float, a_t, out_t)

DF_CLASS(f32a_f32o, float, float)
DF_CLASS(f32a_bf16o, float, __nv_bfloat16)
DF_CLASS(bf16a_f32o, __nv_bfloat16, float)
DF_CLASS(bf16a_bf16o, __nv_bfloat16, __nv_bfloat16)

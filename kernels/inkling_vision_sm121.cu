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

// kernels/inkling_vision_sm121.cu
//
// Inkling-2-Small's vision tower on the device: the four stages its ViT does not share with
// DeepSeek's, composed with DeepSeek's vision GEMM and attention (kernels/opt_vision_sm121.cu,
// included below and reused unchanged: the tower is 1024 wide with 16 heads of 64, which is
// DeepSeek's geometry). The arithmetic is the host reference's
// (crates/sparkflash-model-inkling2/src/vision.rs), which follows the released processor and
// model; every value is bf16 with f32 arithmetic, rounded where the released ops round.
//
// inkling_vision_patchify(canvas, out, height, width, cells_w, patches)
//   canvas  u8 RGB, height x width, rows top to bottom
//   out     bf16 [patches][1216]: patch p in merge-block order (cell p / 4 in raster order of a
//           cells_w-wide grid, then row and column in the cell), feature f < 1200 at
//           (py, px, c) = (f / 60, f / 3 % 20, f % 3), features 1200.. zero (K padded to a
//           multiple of 32 for the GEMM). A sample past the image's right or bottom edge is the
//           processor's -1 pad. Each is (x - 127.5) / 127.5 in f32, rounded once. One thread an
//           element.
// inkling_vision_layernorm(x, weight, bias, out, m, eps)
//   LayerNorm over rows of 1024 with a bias, the statistics in f32 (two passes); out may be x.
//   One warp a row, grid ceil(m / 8), 256 threads.
// inkling_vision_rope(qkv, cos, sin, m, cells_w)
//   The axial rope on the q and k thirds of rows 1..m of [m][3072], in place; row 0 (the class
//   token) is at (0, 0), where the rotation is the identity, and is left alone. Pair (2i, 2i + 1)
//   of a head turns by the patch's column (counted from one) at frequency i for i < 16 and by its
//   row at frequency i - 16 after, through the host's f32 tables cos / sin [position][16]:
//   (x0 c - x1 s, x1 c + x0 s) in f32, rounded once. One thread a pair, grid m * 4, 256 threads.
// inkling_vision_scale_add(x, gamma, h, m)
//   The layer scale's residual over rows of 1024: x = bf16(x + bf16(gamma * h)). One thread two
//   values, grid ceil(m * 512 / 256), 256 threads.

// opt_vision_sm121.cu takes its model shapes as instance arguments (kernels/instances.toml,
// docs/engine/06-base-reads-description.md D.2). Inkling composes the same 1024-wide, 16-head,
// 64-wide DeepSeek bodies and its cubin has always carried their symbols, so it names the
// released DeepSeek instance's values here. The tokens are identical to the released instance's
// defines, which is why the composed cubin keeps its bytes and full symbol set.
#define SPARK_INSTANCE_OPT_VISION_PATCHIFY_HIDDEN 1024u
#define SPARK_INSTANCE_OPT_VISION_PATCHIFY_PATCH 14u
#define SPARK_INSTANCE_OPT_VISION_PATCHIFY_16_HIDDEN 1024u
#define SPARK_INSTANCE_OPT_VISION_PATCHIFY_16_PATCH 16u
#define SPARK_INSTANCE_OPT_VISION_GEMM_HIDDEN 1024u
#define SPARK_INSTANCE_OPT_VISION_RMSNORM_HIDDEN 1024u
#define SPARK_INSTANCE_OPT_VISION_ROPE_HIDDEN 1024u
#define SPARK_INSTANCE_OPT_VISION_ROPE_HEADS 16u
#define SPARK_INSTANCE_OPT_VISION_ATTN_HIDDEN 1024u
#define SPARK_INSTANCE_OPT_VISION_ATTN_HEADS 16u
#define SPARK_INSTANCE_OPT_VISION_SWIGLU_HIDDEN 1024u
#define SPARK_INSTANCE_OPT_VISION_SWIGLU_INTERMEDIATE 2816u
#define SPARK_INSTANCE_OPT_VISION_UNFOLD_HIDDEN 1024u
#define SPARK_INSTANCE_OPT_VISION_SPAN_ROWS_HIDDEN 1024u
#define SPARK_INSTANCE_OPT_VISION_SPAN_ROWS_OUT_HIDDEN 5120u
#define SPARK_INSTANCE_OPT_VISION_ROUTE_HIDDEN 1024u
#define SPARK_INSTANCE_OPT_VISION_ROUTE_EXPERTS 384u
#define SPARK_INSTANCE_OPT_VISION_ROUTE_TOP_K 6u

#include "opt_vision_sm121.cu"

// Inkling's own bodies below are written against the shared tower's width names. The shared
// source undefines its per-entry shape macros after each entry, so these come after the include.
#define OV_DIM SPARK_INSTANCE_OPT_VISION_RMSNORM_HIDDEN
#define OV_QKV (3u * OV_DIM)

#define IV_PATCH 20u
#define IV_K (3u * IV_PATCH * IV_PATCH)
#define IV_KP ((IV_K + 31u) / 32u * 32u)

// Where patch `patch` of a merge-block-ordered grid of cells_w cells across sits.
__device__ __forceinline__ void iv_patch_position(const uint32_t patch, const uint32_t cells_w,
                                                  uint32_t &row, uint32_t &col) {
    const uint32_t cell = patch >> 2;
    const uint32_t inner = patch & 3u;
    row = (cell / cells_w) * 2u + (inner >> 1);
    col = (cell % cells_w) * 2u + (inner & 1u);
}

extern "C" __global__ void __launch_bounds__(256) inkling_vision_patchify(
    const uint8_t *__restrict__ canvas, __nv_bfloat16 *__restrict__ out, const uint32_t height,
    const uint32_t width, const uint32_t cells_w, const uint32_t patches) {
    OPT_PDL_ENTRY();
    const uint64_t t = (uint64_t)blockIdx.x * 256u + threadIdx.x;
    const uint64_t patch = t / IV_KP;
    const uint32_t f = (uint32_t)(t % IV_KP);
    if (patch >= patches) return;
    float v = 0.0f;
    if (f < IV_K) {
        uint32_t row, col;
        iv_patch_position((uint32_t)patch, cells_w, row, col);
        const uint64_t y = (uint64_t)row * IV_PATCH + f / (3u * IV_PATCH);
        const uint64_t x = (uint64_t)col * IV_PATCH + (f / 3u) % IV_PATCH;
        const float sample = y < height && x < width ? (float)canvas[(y * width + x) * 3u + f % 3u] : -1.0f;
        v = __fdiv_rn(__fsub_rn(sample, 127.5f), 127.5f);
    }
    out[t] = __float2bfloat16_rn(v);
}

extern "C" __global__ void __launch_bounds__(256) inkling_vision_layernorm(
    const __nv_bfloat16 *x, const __nv_bfloat16 *__restrict__ weight,
    const __nv_bfloat16 *__restrict__ bias, __nv_bfloat16 *out, const uint32_t m, const float eps) {
    OPT_PDL_ENTRY();
    const uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= m) return;
    const uint4 *src = reinterpret_cast<const uint4 *>(x + (uint64_t)row * OV_DIM);
    float v[32];
    float sum = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) {
        const uint4 u = src[lane + 32u * i];
        const uint32_t words[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&words[j]));
            v[8u * i + 2u * j] = f.x;
            v[8u * i + 2u * j + 1u] = f.y;
            sum += f.x + f.y;
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    const float mean = sum / (float)OV_DIM;
    float squares = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < 32u; ++i) {
        const float d = v[i] - mean;
        squares += d * d;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) squares += __shfl_xor_sync(0xffffffffu, squares, offset);
    const float r = rsqrtf(squares / (float)OV_DIM + eps);
    const uint4 *wsrc = reinterpret_cast<const uint4 *>(weight);
    const uint4 *bsrc = reinterpret_cast<const uint4 *>(bias);
    uint4 *dst = reinterpret_cast<uint4 *>(out + (uint64_t)row * OV_DIM);
#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) {
        const uint4 wu = wsrc[lane + 32u * i];
        const uint4 bu = bsrc[lane + 32u * i];
        const uint32_t wwords[4] = {wu.x, wu.y, wu.z, wu.w};
        const uint32_t bwords[4] = {bu.x, bu.y, bu.z, bu.w};
        uint32_t packed[4];
#pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const float2 wf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&wwords[j]));
            const float2 bf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&bwords[j]));
            packed[j] = ov_pack((v[8u * i + 2u * j] - mean) * r * wf.x + bf.x,
                                (v[8u * i + 2u * j + 1u] - mean) * r * wf.y + bf.y);
        }
        dst[lane + 32u * i] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    }
}

extern "C" __global__ void __launch_bounds__(256) inkling_vision_rope(
    __nv_bfloat16 *__restrict__ qkv, const float *__restrict__ cos_table,
    const float *__restrict__ sin_table, const uint32_t m, const uint32_t cells_w) {
    OPT_PDL_ENTRY();
    // Thread t is pair i of head slot s of row r: t = (r * 32 + s) * 32 + i; slots 0..15 are the q
    // heads and 16..31 the k heads, so slot s starts at column 64 s.
    const uint64_t t = (uint64_t)blockIdx.x * 256u + threadIdx.x;
    const uint64_t row = t >> 10;
    if (row >= m || row == 0u) return;
    const uint32_t slot = (uint32_t)(t >> 5) & 31u;
    const uint32_t i = (uint32_t)t & 31u;
    uint32_t grid_row, grid_col;
    iv_patch_position((uint32_t)(row - 1u), cells_w, grid_row, grid_col);
    const uint32_t position = i < 16u ? grid_col + 1u : grid_row + 1u;
    const uint32_t at = position * 16u + (i & 15u);
    const float c = cos_table[at];
    const float s = sin_table[at];
    __nv_bfloat16 *pair = qkv + row * OV_QKV + slot * 64u + 2u * i;
    const float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(pair));
    *reinterpret_cast<__nv_bfloat162 *>(pair) =
        __floats2bfloat162_rn(__fadd_rn(__fmul_rn(x.x, c), __fmul_rn(-x.y, s)),
                              __fadd_rn(__fmul_rn(x.y, c), __fmul_rn(x.x, s)));
}

extern "C" __global__ void __launch_bounds__(256) inkling_vision_scale_add(
    __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ gamma,
    const __nv_bfloat16 *__restrict__ h, const uint32_t m) {
    OPT_PDL_ENTRY();
    const uint64_t t = (uint64_t)blockIdx.x * 256u + threadIdx.x;
    const uint64_t row = t / (OV_DIM / 2u);
    if (row >= m) return;
    const uint32_t j = (uint32_t)(t % (OV_DIM / 2u)) * 2u;
    const float2 g = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(gamma + j));
    const float2 hv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(h + row * OV_DIM + j));
    __nv_bfloat162 *dst = reinterpret_cast<__nv_bfloat162 *>(x + row * OV_DIM + j);
    const float2 xv = __bfloat1622float2(*dst);
    const float2 scaled = __bfloat1622float2(__floats2bfloat162_rn(__fmul_rn(g.x, hv.x), __fmul_rn(g.y, hv.y)));
    *dst = __floats2bfloat162_rn(__fadd_rn(xv.x, scaled.x), __fadd_rn(xv.y, scaled.y));
}

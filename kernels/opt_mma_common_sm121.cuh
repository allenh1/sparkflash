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

// kernels/opt_mma_common_sm121.cuh
//
// The tensor-core prefill's shared primitives, lifted unchanged out of kernels/opt_mma_sm121.cu so a
// second model pack runs the same instructions instead of growing its own copy: the bulk tile
// geometry, the ue8m0 decode, the cp.async stage copy and the ldmatrix load.
//
// Attribution:
//   - The SM120/SM121 (GB10) mma.sync / ldmatrix / cp.async tiling patterns draw on prior art from
//     b12x (https://github.com/local-inference-lab/b12x, flashinfer/experimental/b12x, Apache-2.0;
//     Luke Alonso @lukealonso, @original-el8, and contributors) and NVIDIA Flare
//     (https://github.com/NVIDIA/flare-local-inference, Apache-2.0). See the root NOTICE file.
//
// opt_mma_sm121.cu includes this header at the place the definitions used to stand, so its token
// stream - and its sha256-pinned cubin (crates/sparkflash-kernels/baseline/kernel_baseline.txt) - is
// what it was. Anything that would change the generated code of an existing instantiation belongs in
// a new helper, not an edit here.

#ifndef SPARKFLASH_OPT_MMA_COMMON_SM121_CUH
#define SPARKFLASH_OPT_MMA_COMMON_SM121_CUH

#include <cuda_bf16.h>
#include <stdint.h>

#define OM_THREADS 256u
#define OM_BM 128u
#define OM_BN 128u
#define OM_KB 2u                 // MX blocks per stage
#define OM_A_ROW (OM_KB * 32u)   // activation bytes per row per stage (E4M3)
#define OM_W_TILES ((OM_BN / 32u) * OM_KB)

// 2^(code - 127) as f32; code 0 is 2^-127 (subnormal).
__device__ __forceinline__ float opt_ue8m0_f32(uint8_t code) {
    return code == 0u ? __uint_as_float(0x00400000u) : __uint_as_float((uint32_t)code << 23);
}

__device__ __forceinline__ uint32_t om_smem(const void *p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

// 16 bytes global -> shared, zero-filled when !valid (the source is then not read).
__device__ __forceinline__ void om_cp16(uint32_t dst, const void *src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void om_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void om_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ __forceinline__ void om_ldsm4(uint32_t addr, uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

#endif  // SPARKFLASH_OPT_MMA_COMMON_SM121_CUH

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

// kernels/inkling_sm121.cu
//
// The Image::Inkling cubin: Inkling-2-Small's relative attention, short convolution and
// block-attention residual, and its softcapped epilogues in one translation unit. The three
// bodies stay in their own sources (they are also probed on their own), so this file is only
// the image's composition. Every symbol is loaded by name; a missing one is a load error.
//
// The shared token, GEMV, RMSNorm and routed-MoE plan kernels stay in their own images and are
// reused unchanged; the expert's own weight format and the sigmoid router are this pack's and
// live in inkling_moe_sm121.cu. The direct-RDMA exchange's stage and collect kernels are the
// DeepSeek arm's own (kernels/opt_exchange_sm121.cuh, which opt_misc_sm121.cu also includes), so
// the Inkling collectives ring the same doorbell with the same code. The vision tower's own
// stages (inkling_vision_sm121.cu) bring DeepSeek's vision GEMM and attention with them
// (opt_vision_sm121.cu), which rank 0 runs on an image before its prompt. The DSpark drafter's
// own stages (inkling_draft_sm121.cu) run beside DFlash 2's block attention, which stays in its
// image.

#include "inkling_attn_sm121.cu"
#include "inkling_conv_sm121.cu"
#include "inkling_draft_sm121.cu"
#include "inkling_epilogue_sm121.cu"
#include "inkling_moe_sm121.cu"
#include "inkling_vision_sm121.cu"
#include "opt_exchange_sm121.cuh"

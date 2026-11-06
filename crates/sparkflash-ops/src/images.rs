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

//! The images the shared kernel stack builds, re-exported for every arm.

pub use sparkflash_kernels::images::{
    Image, dflash2_cubin, glm_attention_cubin, glm_decode_cubin, glm_moe_cubin, glm_prefill_cubin,
    qwen4_attn_cubin, qwen4_exp_cubin, qwen4_moe_cubin, qwen4_prefill_cubin, qwen4_scan_cubin,
};

/// Whether this build carries the Qwen3.8 attention image.
#[must_use]
pub fn qwen4_attn_available() -> bool {
    Image::Qwen4Attn.available()
}

/// Whether this build carries the Qwen3.8 token image.
#[must_use]
pub fn qwen4_available() -> bool {
    Image::Qwen4Exp.available()
}

/// Whether this build carries the Qwen3.8 tensor-core prefill image.
#[must_use]
pub fn qwen4_prefill_available() -> bool {
    Image::Qwen4Prefill.available()
}

/// Whether this build carries the Qwen3.8 decode/token routed-expert image.
#[must_use]
pub fn qwen4_moe_available() -> bool {
    Image::Qwen4Moe.available()
}

/// Whether this build carries the Qwen3.8 linear-attention scan image.
#[must_use]
pub fn qwen4_scan_available() -> bool {
    Image::Qwen4Scan.available()
}

/// Whether this build carries the DFlash 2 drafter image.
#[must_use]
pub fn dflash2_available() -> bool {
    Image::Dflash2.available()
}

/// Whether this build carries the GLM-5.3 bulk-prefill image.
#[must_use]
pub fn glm_prefill_available() -> bool {
    Image::GlmPrefill.available()
}

/// Whether this build carries the GLM-5.3 sparse MLA attention image.
#[must_use]
pub fn glm_attention_available() -> bool {
    Image::GlmAttention.available()
}

/// Whether this build carries the GLM-5.3 routed-expert (MoE) image.
#[must_use]
pub fn glm_moe_available() -> bool {
    Image::GlmMoe.available()
}

/// Whether this build carries the GLM-5.3 decode/token image.
#[must_use]
pub fn glm_decode_available() -> bool {
    Image::GlmDecode.available()
}

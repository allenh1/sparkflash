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

//! The shared kernel images: every cubin the arms launch, behind one accessor each.
//!
//! Before the one-stack move, the Qwen3.8 and GLM-5.3 sources were compiled by
//! sparkflash-model-qwen4-exp's build script and every other arm reached them through that model
//! crate. The build script here compiles all ten images from kernels/ and exports the paths, so
//! a model crate carries no CUDA and no kernel build script and no model crate depends on another
//! for a kernel.

/// The CUDA architecture this build's kernels target (`SPARKFLASH_CUDA_ARCH`, default `sm_121a`).
///
/// The numerics registry keys every configuration by it: the same source compiled for another
/// architecture is a different arithmetic until a reference says otherwise.
pub const CUDA_ARCH: &str = env!("SPARKFLASH_BUILT_CUDA_ARCH");

/// One image the shared stack builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Image {
    /// The Qwen3.8 token, norm, dense and hyper kernels.
    Qwen4Exp,
    /// The Qwen3.8 attention, sparse-indexer and KV-cache token kernels.
    Qwen4Attn,
    /// The Qwen3.8 decode/token routed-expert (MoE) kernels.
    Qwen4Moe,
    /// The Qwen3.8 tensor-core bulk prefill.
    Qwen4Prefill,
    /// The Qwen3.8 Gated DeltaNet / Kimi Delta Attention scan.
    Qwen4Scan,
    /// The DFlash 2 block-drafter kernels.
    Dflash2,
    /// The GLM-5.3 block-scaled FP8 bulk prefill.
    GlmPrefill,
    /// The GLM-5.3 sparse MLA attention and its per-head projections.
    GlmAttention,
    /// The GLM-5.3 bulk routed-expert (MoE) kernels.
    GlmMoe,
    /// The GLM-5.3 decode/token kernels: hyper-connection mixers, router, indexer and absorbed MLA.
    GlmDecode,
    /// The routed-expert plan every arm shares.
    MoePlan,
    /// The DeepSeek kernels loaded from the shared PTX set.
    Deepseek,
    /// The RoCEnante one-shot collective.
    Rocenante,
}

#[cfg(sparkflash_model_cubin)]
const QWEN4_EXP: &[u8] = include_bytes!(env!("SPARKFLASH_QWEN4_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const QWEN4_EXP: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const QWEN4_ATTN: &[u8] = include_bytes!(env!("SPARKFLASH_QWEN4_ATTN_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const QWEN4_ATTN: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const QWEN4_MOE: &[u8] = include_bytes!(env!("SPARKFLASH_QWEN4_MOE_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const QWEN4_MOE: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const QWEN4_PREFILL: &[u8] = include_bytes!(env!("SPARKFLASH_QWEN4_PREFILL_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const QWEN4_PREFILL: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const QWEN4_SCAN: &[u8] = include_bytes!(env!("SPARKFLASH_QWEN4_SCAN_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const QWEN4_SCAN: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const DFLASH2: &[u8] = include_bytes!(env!("SPARKFLASH_QWEN4_DFLASH2_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const DFLASH2: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const GLM_PREFILL: &[u8] = include_bytes!(env!("SPARKFLASH_GLM_PREFILL_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const GLM_PREFILL: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const GLM_ATTENTION: &[u8] = include_bytes!(env!("SPARKFLASH_GLM_ATTENTION_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const GLM_ATTENTION: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const GLM_MOE: &[u8] = include_bytes!(env!("SPARKFLASH_GLM_MOE_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const GLM_MOE: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const GLM_DECODE: &[u8] = include_bytes!(env!("SPARKFLASH_GLM_DECODE_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const GLM_DECODE: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const MOE_PLAN: &[u8] = include_bytes!(env!("SPARKFLASH_MOE_PLAN_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const MOE_PLAN: &[u8] = &[];

/// The Qwen3.8 token image, empty on a CPU-only build.
#[must_use]
pub fn qwen4_exp_cubin() -> &'static [u8] {
    QWEN4_EXP
}

/// The Qwen3.8 attention image, empty on a CPU-only build.
#[must_use]
pub fn qwen4_attn_cubin() -> &'static [u8] {
    QWEN4_ATTN
}

/// The Qwen3.8 MoE image, empty on a CPU-only build.
#[must_use]
pub fn qwen4_moe_cubin() -> &'static [u8] {
    QWEN4_MOE
}

/// The Qwen3.8 tensor-core prefill image, empty on a CPU-only build.
#[must_use]
pub fn qwen4_prefill_cubin() -> &'static [u8] {
    QWEN4_PREFILL
}

/// The Qwen3.8 scan image, empty on a CPU-only build.
#[must_use]
pub fn qwen4_scan_cubin() -> &'static [u8] {
    QWEN4_SCAN
}

/// The DFlash 2 drafter image, empty on a CPU-only build.
#[must_use]
pub fn dflash2_cubin() -> &'static [u8] {
    DFLASH2
}

/// The GLM-5.3 bulk-prefill image, empty on a CPU-only build.
#[must_use]
pub fn glm_prefill_cubin() -> &'static [u8] {
    GLM_PREFILL
}

/// The GLM-5.3 sparse MLA attention image, empty on a CPU-only build.
#[must_use]
pub fn glm_attention_cubin() -> &'static [u8] {
    GLM_ATTENTION
}

/// The GLM-5.3 routed-expert image, empty on a CPU-only build.
#[must_use]
pub fn glm_moe_cubin() -> &'static [u8] {
    GLM_MOE
}

/// The GLM-5.3 decode/token image, empty on a CPU-only build.
#[must_use]
pub fn glm_decode_cubin() -> &'static [u8] {
    GLM_DECODE
}

/// The routed-expert plan image every arm shares, empty on a CPU-only build.
#[must_use]
pub fn moe_plan_cubin() -> &'static [u8] {
    MOE_PLAN
}

impl Image {
    /// The image's bytes, empty when this build or this process carries none.
    #[must_use]
    pub fn bytes(self) -> &'static [u8] {
        match self {
            Self::Qwen4Exp => qwen4_exp_cubin(),
            Self::Qwen4Attn => qwen4_attn_cubin(),
            Self::Qwen4Moe => qwen4_moe_cubin(),
            Self::Qwen4Prefill => qwen4_prefill_cubin(),
            Self::Qwen4Scan => qwen4_scan_cubin(),
            Self::Dflash2 => dflash2_cubin(),
            Self::GlmPrefill => glm_prefill_cubin(),
            Self::GlmAttention => glm_attention_cubin(),
            Self::GlmMoe => glm_moe_cubin(),
            Self::GlmDecode => glm_decode_cubin(),
            Self::MoePlan => moe_plan_cubin(),
            // DeepSeek loads its kernels through the shared driver from the PTX set, and the
            // RoCEnante path compiles its own image at run time.
            Self::Deepseek | Self::Rocenante => &[],
        }
    }

    /// Whether this build embeds the image.
    #[must_use]
    pub fn available(self) -> bool {
        !self.bytes().is_empty()
    }

    /// The image's stable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Qwen4Exp => "qwen4_exp_sm121",
            Self::Qwen4Attn => "qwen4_attn_sm121",
            Self::Qwen4Moe => "qwen4_moe_sm121",
            Self::Qwen4Prefill => "qwen4_prefill_sm121",
            Self::Qwen4Scan => "qwen4_scan_sm121",
            Self::Dflash2 => "dflash2_sm121",
            Self::GlmPrefill => "glm_prefill_sm121",
            Self::GlmAttention => "glm_attention_sm121",
            Self::GlmMoe => "glm_moe_sm121",
            Self::GlmDecode => "glm_decode_sm121",
            Self::MoePlan => "moe_plan_sm121",
            Self::Deepseek => "deepseek_ptx",
            Self::Rocenante => "rocenante_oneshot_sm121",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_image_has_a_stable_name_and_a_consistent_availability_bit() {
        for image in [
            Image::Qwen4Exp,
            Image::Qwen4Attn,
            Image::Qwen4Moe,
            Image::Qwen4Prefill,
            Image::Qwen4Scan,
            Image::Dflash2,
            Image::GlmPrefill,
            Image::GlmAttention,
            Image::GlmMoe,
            Image::GlmDecode,
            Image::Deepseek,
            Image::Rocenante,
        ] {
            assert!(!image.name().is_empty());
            assert_eq!(image.available(), !image.bytes().is_empty());
        }
    }
}

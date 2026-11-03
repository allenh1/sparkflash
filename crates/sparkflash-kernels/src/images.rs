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
//! crate. The build script here compiles all four images from kernels/ and exports the paths, so
//! a model crate carries no CUDA and no kernel build script and no model crate depends on another
//! for a kernel.

/// One image the shared stack builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Image {
    /// The Qwen3.8 token, attention and expert kernels.
    Qwen4Exp,
    /// The Qwen3.8 tensor-core bulk prefill.
    Qwen4Prefill,
    /// The DFlash 2 block-drafter kernels.
    Dflash2,
    /// The GLM-5.3 block-scaled FP8 bulk prefill.
    GlmPrefill,
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
const QWEN4_PREFILL: &[u8] = include_bytes!(env!("SPARKFLASH_QWEN4_PREFILL_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const QWEN4_PREFILL: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const DFLASH2: &[u8] = include_bytes!(env!("SPARKFLASH_QWEN4_DFLASH2_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const DFLASH2: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const GLM_PREFILL: &[u8] = include_bytes!(env!("SPARKFLASH_GLM_PREFILL_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const GLM_PREFILL: &[u8] = &[];

/// The Qwen3.8 token image, empty on a CPU-only build.
#[must_use]
pub fn qwen4_exp_cubin() -> &'static [u8] {
    QWEN4_EXP
}

/// The Qwen3.8 tensor-core prefill image, empty on a CPU-only build.
#[must_use]
pub fn qwen4_prefill_cubin() -> &'static [u8] {
    QWEN4_PREFILL
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

impl Image {
    /// The image's bytes, empty when this build or this process carries none.
    #[must_use]
    pub fn bytes(self) -> &'static [u8] {
        match self {
            Self::Qwen4Exp => qwen4_exp_cubin(),
            Self::Qwen4Prefill => qwen4_prefill_cubin(),
            Self::Dflash2 => dflash2_cubin(),
            Self::GlmPrefill => glm_prefill_cubin(),
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
            Self::Qwen4Prefill => "qwen4_prefill_sm121",
            Self::Dflash2 => "dflash2_sm121",
            Self::GlmPrefill => "glm_prefill_sm121",
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
            Image::Qwen4Prefill,
            Image::Dflash2,
            Image::GlmPrefill,
            Image::Deepseek,
            Image::Rocenante,
        ] {
            assert!(!image.name().is_empty());
            assert_eq!(image.available(), !image.bytes().is_empty());
        }
    }
}

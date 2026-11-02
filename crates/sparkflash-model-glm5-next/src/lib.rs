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

//! The GLM-5.3-Flash (Glm5Next) pack, for Z.ai's FP8 release at TP=4.
//!
//! zai-org/GLM-5.3-Flash is a 321B-parameter (18B active) hybrid-attention MoE: 34 Kimi Delta
//! Attention layers and 11 DeepSeek Sparse Attention layers, every branch wrapped in
//! manifold-constrained hyper-connections. Z.ai releases it in BF16 (642.6 GB) and with its experts,
//! dense MLPs and MLA projections in block-scaled FP8 (328.3 GB); only the FP8 release fits four
//! 128 GB nodes, at 82 GB a rank. Its blocks are the ones this tree already serves for the other
//! two models: Qwen3.8's delta rule and pooled-block indexer, and V4.1's hyper-connections and
//! compressed attention cache. This crate owns the model definition:
//!
//! * [config] parses the released configuration and enforces the invariants the reference
//!   implementation relies on (NoPE MLA, the k-pool indexer, sigmoid `noaux_tc` routing, mHC,
//!   square FP8 tiles).
//! * [tensor] plans all 76,108 tensor names of the checkpoint and pins the byte identity of the
//!   complete plan against the released index total (and the BF16 release's), and splits every
//!   served tensor across a tensor-parallel world.
//! * [checkpoint] writes a synthetic checkpoint for any configuration in the released storage
//!   formats, so the tiny configuration loads the way the release does.
//! * [forward] is the f32 host reference forward, the oracle a device arm is compared against.
//! * [device] is that device arm: one tensor-parallel rank on one GPU, decoding with the
//!   Qwen4-Exp pack's kernels.
//! * [convert] writes a rank's weights as that arm uploads them to one file a rank, which the arm
//!   boots from with one parallel direct-I/O read.
//! * [codec] renders the released chat template over a Hugging Face tokenizer, and [serve] puts the
//!   device rank behind `ModelBackend`/`ModelSession` so the generic host driver can serve it.
//!
//! # Attribution
//!
//! * **Model architecture, weights & reference:**
//!   [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (Z.ai, MIT) and the
//!   `Glm5NextForConditionalGeneration` reference implementation in
//!   [Transformers](https://github.com/huggingface/transformers) (Apache-2.0), read for the
//!   arithmetic; no source is copied.
//! * **Methods:** Kimi Delta Attention from Kimi Linear (Kimi Team, arXiv:2510.26692), DeepSeek
//!   Sparse Attention from DeepSeek-V3.2 (arXiv:2512.02556), manifold-constrained
//!   hyper-connections (Xie et al., arXiv:2512.24880), and the MTP layer's multi-token prediction
//!   from DeepSeek-V3 (arXiv:2412.19437).
//! * **Block drafting:** the DFlash 2 drafter of
//!   [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (Inco AI,
//!   CC BY-NC-ND 4.0; loaded from the user's copy, not redistributed), the DFlash method
//!   ([z-lab/dflash](https://github.com/z-lab/dflash), MIT), and
//!   [SGLang](https://github.com/sgl-project/sglang)'s DFlash support and GLM-5.3-Flash capture
//!   (Apache-2.0), read for the arithmetic; no source is copied.
//! * **Kernels:** the device arm runs the Qwen3.8 pack's token cubin, whose SM120/SM121 execution
//!   patterns draw on [b12x](https://github.com/local-inference-lab/b12x) and
//!   [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference) (both Apache-2.0).
//!
//! See the root NOTICE file (sections 1, 2, 8 and 9) for full provenance details.

/// The canonical id of the released GLM-5.3-Flash checkpoint.
pub const MODEL_ID: &str = "zai-org/GLM-5.3-Flash";

/// The control-plane weight selector for the released GLM-5.3-Flash FP8 checkpoint.
pub const WEIGHTS_ID: &str = "glm5.3-flash-fp8";

/// The model family the descriptor reports and the control plane groups by.
pub const FAMILY: &str = "glm5-next";

/// The default context the serving arm allocates for one sequence.
///
/// The sparse layers cache the release's E4M3 latent, not an expansion of it
/// ([crate::device::DeviceRank::dsa_attend]): 512 codes and four f32 block scales a position a
/// layer, plus the indexer's 128-wide key and gate and its pooled key. That is 1.7 KiB a position
/// a layer, 18.2 KiB across the eleven sparse layers, so the model's full 1,048,576-position
/// window costs 18.2 GiB a rank - beside the 82 GB rank on a 121 GiB Spark, with about 18 GiB to
/// spare. The exact f32 cache ([crate::device::LatentCache::F32]) is 35 KiB a position; at the
/// full window it leaves 1-2 GiB, so a serve arm that selects it wants a smaller `--context`.
pub const DEFAULT_DEVICE_CONTEXT: usize = 1_048_576;

/// The output ceiling the serving arm advertises for one request.
pub const MAX_OUTPUT_TOKENS: usize = 32_768;

pub mod checkpoint;
pub mod codec;
pub mod config;
pub mod convert;
pub mod device;
pub mod forward;
pub mod serve;
pub mod tensor;

/// Whether this build embeds the device cubin the arm runs.
#[must_use]
pub fn cubin_available() -> bool {
    sparkflash_model_qwen4_exp::device::cubin_available()
}

pub use codec::GlmCodec;
pub use config::{ConfigError, Glm5NextConfig, IndexerKind, LayerKind, MlpKind, VisionConfig};
pub use device::{DFLASH2_STEP_COST, DeviceRank, DeviceSequence, MAX_ROWS};
pub use forward::{Glm5NextWeights, ReferenceState};
pub use serve::Glm5NextDeviceBackend;
pub use tensor::{Glm5NextTensorPlan, Role, Split};

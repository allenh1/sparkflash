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
//!   device rank behind `ModelBackend`/`ModelSession` so the generic host driver can serve it: a
//!   session reuses a kept prefix, and one that has been told to draft verifies the MTP layer's or
//!   the DFlash 2 drafter's chain with the shared rule (`sparkflash_model::spec`).
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
pub const MODEL_ID: &str = sparkflash_model::descriptor::GLM53_FLASH_ID;

/// The control-plane weight selector for the released GLM-5.3-Flash FP8 checkpoint.
pub const WEIGHTS_ID: &str = sparkflash_model::descriptor::GLM53_FLASH_WEIGHTS_ID;

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
pub const DEFAULT_DEVICE_CONTEXT: usize = sparkflash_model::descriptor::GLM53_CONTEXT_TOKENS;

/// The output ceiling the serving arm advertises for one request.
pub const MAX_OUTPUT_TOKENS: usize = sparkflash_model::descriptor::GLM53_MAX_OUTPUT_TOKENS;

pub mod architecture;
pub mod checkpoint;
pub mod codec;
pub mod config;
pub mod convert;
pub mod device;
pub mod forward;
pub mod serve;
pub mod tensor;

/// The KV cache a packed serve may spend before it must refuse more workers.
///
/// This is not a node limit: the 82 GB rank and its chunk scratch sit beside it. It is the slice
/// the session caches are allowed to claim at the default window, and it is sized so that exactly
/// one 1,048,576-position session fits ([DEFAULT_DEVICE_CONTEXT] costs 18.2 GiB at the FP8
/// release) and a second has to come from a shorter `--context`. A node with more room raises it
/// with `--kv-budget-gib`.
///
/// Before this constant, `--concurrency` multiplied a window nobody had costed: four workers at
/// the full window claim 72.8 GiB of caches beside the weights, and the node finds out one
/// request at a time.
pub const CONCURRENCY_KV_BUDGET_BYTES: u64 = 24 << 30;

/// The KV bytes a budget of `gib` gibibytes allows.
#[must_use]
pub const fn kv_budget_bytes(gib: usize) -> u64 {
    (gib as u64) << 30
}

/// The cache plan one session of `config` claims, from the configuration alone.
///
/// The two terms come from the geometry [DeviceRank::new_sequence] allocates: a sparse layer's
/// latent row (the release's E4M3 codes and their f32 block scales, or the f32 latent of a BF16
/// release), an indexer layer's key, gate and pooled key, the MTP layer's latent when the
/// checkpoint carries one, and the fixed KDA history, recurrence and verify spare of every linear
/// layer. The pooled key is counted as if every position added a whole one, so the plan is an
/// upper bound by at most one pool.
///
/// The representation follows [Glm5NextConfig::fp8_block], which is the same signal
/// [crate::device::LatentCache::Auto] reads on a loaded rank: a block-quantised checkpoint stores
/// the quantised activation, a BF16 one the f32 latent.
#[must_use]
pub fn kv_plan(config: &Glm5NextConfig) -> sparkflash_model::KvPlan {
    let width = config.kv_lora_rank as u64;
    let latent = if config.fp8_block.is_some() {
        // E4M3 codes, then one f32 scale a 128 codes.
        width + 4 * width.div_ceil(128)
    } else {
        4 * width
    };
    let index_dim = config.index_head_dim as u64;
    let pool = config.index_kpool.max(1) as u64;
    // The indexer's key and gate rows, and its pooled key amortised over one pool of positions.
    let indexed = 2 * index_dim * 4 + (index_dim * 4).div_ceil(pool);
    let heads = config.linear_num_heads as u64;
    let head = config.linear_head_dim as u64;
    let history = config.linear_conv_kernel_dim.saturating_sub(1) as u64;
    let mut per_token = 0u64;
    let mut fixed = 0u64;
    for (layer, kind) in config.layer_types.iter().enumerate() {
        match kind {
            LayerKind::Sparse => {
                per_token += latent;
                if config.indexer_types[layer] == IndexerKind::Full {
                    per_token += indexed;
                }
            }
            LayerKind::Linear => {
                // The convolution history the kernel keeps, then the recurrence and its spare.
                fixed += history * 3 * heads * head * 4 + 2 * heads * head * head * 4;
            }
        }
    }
    if config.num_nextn_predict_layers > 0 {
        per_token += latent;
    }
    sparkflash_model::KvPlan::new(per_token, fixed)
}

/// The cache bytes one session allocates at `context` positions.
#[must_use]
pub fn sequence_kv_bytes(config: &Glm5NextConfig, context: usize) -> u64 {
    kv_plan(config).sequence_bytes(context)
}

/// The cache bytes `concurrency` sessions allocate at `context` positions.
#[must_use]
pub fn concurrency_kv_bytes(config: &Glm5NextConfig, context: usize, concurrency: usize) -> u64 {
    kv_plan(config).concurrency_bytes(context, concurrency)
}

/// Whether this build embeds the device cubin the arm runs.
#[must_use]
pub fn cubin_available() -> bool {
    sparkflash_ops::images::qwen4_available() && sparkflash_ops::images::glm_decode_available()
}

pub use architecture::{architecture_from_carried, architecture_from_json, tiny_architecture};
pub use codec::GlmCodec;
pub use config::{ConfigError, Glm5NextConfig, IndexerKind, LayerKind, MlpKind, VisionConfig};
pub use device::{DFLASH2_STEP_COST, DeviceRank, DeviceSequence, MAX_ROWS};
pub use forward::{Glm5NextWeights, ReferenceState};
pub use serve::Glm5NextDeviceBackend;
pub use tensor::{Glm5NextTensorPlan, Role, Split};

#[cfg(test)]
mod tests {
    use super::*;

    /// The released FP8 checkpoint: the carried document is the BF16 release, so the block tiling
    /// is set to what the FP8 release declares.
    fn fp8_release() -> Glm5NextConfig {
        let mut config = Glm5NextConfig::glm53_flash().expect("the carried configuration");
        config.fp8_block = Some(128);
        config
    }

    #[test]
    fn the_released_window_costs_the_memory_the_arm_documents() {
        let config = fp8_release();
        let plan = kv_plan(&config);
        // 1.7 KiB a position across the eleven sparse layers, plus the MTP layer's latent.
        assert!(
            (18_000..20_500).contains(&plan.per_token()),
            "per-position claim {}",
            plan.per_token()
        );
        let one = plan.sequence_bytes(DEFAULT_DEVICE_CONTEXT);
        let gib = one as f64 / (1u64 << 30) as f64;
        assert!(
            (17.0..21.0).contains(&gib),
            "{gib:.1} GiB at the full window"
        );
    }

    #[test]
    fn the_kv_budget_admits_one_full_window_and_refuses_two() {
        let config = fp8_release();
        let plan = kv_plan(&config);
        let one = plan.sequence_bytes(DEFAULT_DEVICE_CONTEXT);
        assert!(one <= CONCURRENCY_KV_BUDGET_BYTES);
        let two = plan.concurrency_bytes(DEFAULT_DEVICE_CONTEXT, 2);
        assert!(two > CONCURRENCY_KV_BUDGET_BYTES);
        // Two sessions fit the same slice at half the window.
        let halves = plan.concurrency_bytes(DEFAULT_DEVICE_CONTEXT / 2, 2);
        assert!(halves <= CONCURRENCY_KV_BUDGET_BYTES);
        assert_eq!(kv_budget_bytes(24), CONCURRENCY_KV_BUDGET_BYTES);
        assert_eq!(
            concurrency_kv_bytes(&config, DEFAULT_DEVICE_CONTEXT, 2),
            two,
            "the free function is the plan's"
        );
    }

    #[test]
    fn a_bf16_release_claims_the_f32_latent() {
        let fp8 = fp8_release();
        let mut bf16 = fp8.clone();
        bf16.fp8_block = None;
        // Eleven sparse layers and the MTP layer each move from 528 E4M3 bytes a position to
        // 2,048 f32 ones.
        assert_eq!(
            kv_plan(&bf16).per_token(),
            kv_plan(&fp8).per_token() + 12 * (4 * 512 - 528)
        );
        assert_eq!(kv_plan(&bf16).fixed(), kv_plan(&fp8).fixed());
    }

    #[test]
    fn the_tiny_fixture_claims_a_negligible_window() {
        let config = Glm5NextConfig::tiny();
        let one = kv_plan(&config).sequence_bytes(config.max_position_embeddings);
        assert!(one < 64 << 20, "{one} bytes");
    }
}

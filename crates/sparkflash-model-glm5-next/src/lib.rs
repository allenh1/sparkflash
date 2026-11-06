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
/// a layer, and the MTP stage allocates the same latent and indexer (its device layer owns one),
/// so the full 1,048,576-position window costs 19.7 GiB a rank - beside the 82 GB rank on a
/// 121 GiB Spark, with about 19 GiB to spare. The exact f32 cache
/// ([crate::device::LatentCache::F32]) is 35 KiB a position; at the full window it leaves 1-2 GiB,
/// so a serve arm that selects it wants a smaller `--context`.
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
/// one 1,048,576-position session fits ([DEFAULT_DEVICE_CONTEXT] costs 19.7 GiB at the FP8
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
/// release), an indexer layer's key, gate and pooled key, the MTP stage's latent and its own
/// indexer when the checkpoint carries the stage, and the fixed KDA history, recurrence and verify
/// spare of every linear layer. The pooled key is counted as a whole row a position, which is what
/// the device's `capacity / pool` rows allocate.
///
/// The representation is explicit here, never re-derived inside the architecture:
/// `latent_fp8 = true` is the release's E4M3 latent, `false` the exact f32 one. The E4M3
/// choice needs a latent width that is whole 128-wide scale tiles; a narrower row is f32 in
/// both modes, the condition the device's latent-cache choice applies. A serving profile's
/// [sparkflash_model::profile::LatentCacheMode] chooses which to pass on the TP=4 path; the
/// single-node host fixes it with [SINGLE_RANK_LATENT_FP8].
#[must_use]
pub fn kv_plan_with_latent(config: &Glm5NextConfig, latent_fp8: bool) -> sparkflash_model::KvPlan {
    config
        .architecture
        .kv_plan(1, latent_fp8)
        .expect("the carried GLM-5.3 configuration carries no compressed attention")
}

/// The cache plan the TP=1 host both admits and allocates.
///
/// This is the single source the single-node admission check (`GlmArm` in the CLI) and the
/// single-node rank ([Glm5NextDeviceBackend::open_single_rank]) read, so the plan checked
/// before the upload is the plan the rank's session caches are sized by: the latent and
/// indexer rows and the linear-attention state. It is not a byte-for-byte ledger of every
/// device allocation; the shared [sparkflash_model::Architecture::kv_plan] documents the
/// caches it leaves outside (Qwen's GDN convolution history and PLE frames, which GLM's KDA
/// schedule does not have).
#[must_use]
pub fn single_rank_kv_plan(config: &Glm5NextConfig) -> sparkflash_model::KvPlan {
    kv_plan_with_latent(config, SINGLE_RANK_LATENT_FP8)
}

/// Whether the TP=1 host stores its latent as E4M3: it does not.
///
/// The single-node backend opens with the explicit [device::LatentCache::F32], and the
/// single-node admission check admits the matching 38,400-byte-class row
/// ([single_rank_kv_plan]). The single-node arm ignores the profile's
/// `attention.latent_cache`; that choice is the TP=4 arm's (`open_glm_tp4_rank`), which
/// passes the profile's mode to [device::DeviceRank::load_converted_with_cache].
///
/// The representation is deliberately not derived from [Glm5NextConfig::fp8_block]. That flag
/// is the checkpoint-wide FP8 tiling of the block-scaled projections; `kv_b_proj.weight` is
/// BF16 in the tensor plan ([crate::tensor]), and the device's [device::LatentCache::Auto]
/// reads the per-tensor `kv_b` block scales, not the checkpoint flag. Deriving admission from
/// `fp8_block` selected the 20,160-byte E4M3 row where the rank allocated the 38,400-byte f32
/// one, an 18,240-byte-a-token under-admission.
pub const SINGLE_RANK_LATENT_FP8: bool = false;

/// The [device::LatentCache] [SINGLE_RANK_LATENT_FP8] selects for the TP=1 host.
#[must_use]
pub const fn single_rank_latent() -> device::LatentCache {
    if SINGLE_RANK_LATENT_FP8 {
        device::LatentCache::Fp8
    } else {
        device::LatentCache::F32
    }
}

/// The checkpoint-wide estimate: E4M3 for a block-quantised release, f32 for a BF16 one.
///
/// This reads [Glm5NextConfig::fp8_block], so it describes the block-scaled projections and is
/// not the device's per-tensor [device::LatentCache::Auto]. The single-node host admits and
/// allocates [single_rank_kv_plan] instead.
#[must_use]
pub fn kv_plan(config: &Glm5NextConfig) -> sparkflash_model::KvPlan {
    kv_plan_with_latent(config, config.fp8_block.is_some())
}

/// The cache bytes one session allocates at `context` positions.
#[must_use]
pub fn sequence_kv_bytes(config: &Glm5NextConfig, context: usize) -> u64 {
    kv_plan(config).sequence_bytes(context)
}

/// [sequence_kv_bytes] under an explicit latent representation.
#[must_use]
pub fn sequence_kv_bytes_with_latent(
    config: &Glm5NextConfig,
    context: usize,
    latent_fp8: bool,
) -> u64 {
    kv_plan_with_latent(config, latent_fp8).sequence_bytes(context)
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
        // 1.7 KiB a position across the eleven sparse layers, plus the MTP stage's latent and
        // indexer (the device allocator builds the stage with an indexer).
        assert!(
            (19_000..21_000).contains(&plan.per_token()),
            "per-position claim {}",
            plan.per_token()
        );
        let one = plan.sequence_bytes(DEFAULT_DEVICE_CONTEXT);
        let gib = one as f64 / (1u64 << 30) as f64;
        assert!(
            (19.0..21.0).contains(&gib),
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
        bf16.architecture.quantization.dense_format =
            sparkflash_model::architecture::WeightFormat::Bf16;
        // Eleven sparse layers and the MTP layer each move from 528 E4M3 bytes a position (plus
        // the MTP indexer's 1,152) to 2,048 f32 ones plus the indexer.
        assert_eq!(
            kv_plan_with_latent(&fp8, true).per_token(),
            20_160,
            "the FP8 latent at TP1"
        );
        assert_eq!(
            kv_plan_with_latent(&fp8, false).per_token(),
            38_400,
            "the F32 latent at TP1"
        );
        assert_eq!(
            kv_plan(&fp8).per_token(),
            kv_plan_with_latent(&fp8, true).per_token(),
            "the checkpoint default is E4M3 for the FP8 release"
        );
        assert_eq!(kv_plan(&bf16).per_token(), 38_400, "the F32 latent at TP1");
        assert_eq!(
            kv_plan(&bf16).per_token(),
            kv_plan(&fp8).per_token() + 12 * (4 * 512 - 528)
        );
        assert_eq!(kv_plan(&bf16).fixed(), kv_plan(&fp8).fixed());
        assert_eq!(
            sequence_kv_bytes_with_latent(&fp8, 1, false),
            sequence_kv_bytes(&bf16, 1),
            "the explicit F32 plan sizes the same session as the BF16 Auto default"
        );
    }

    #[test]
    fn the_tiny_fixture_claims_a_negligible_window() {
        let config = Glm5NextConfig::tiny();
        let one = kv_plan(&config).sequence_bytes(config.max_position_embeddings);
        assert!(one < 64 << 20, "{one} bytes");
    }

    #[test]
    fn the_single_rank_host_admits_the_latent_its_rank_allocates() {
        let tiny = Glm5NextConfig::tiny();
        // The tiny fixture carries the checkpoint-wide FP8 tiling while its tensor plan stores
        // `kv_b_proj.weight` BF16: the distinction the old `config.fp8_block` admission missed.
        assert_eq!(tiny.fp8_block, Some(8));
        assert_eq!(single_rank_latent(), device::LatentCache::F32);
        let admitted = single_rank_kv_plan(&tiny);
        let f32_plan = kv_plan_with_latent(&tiny, false);
        let fp8_plan = kv_plan_with_latent(&tiny, true);
        assert_eq!(admitted, f32_plan, "the admitted plan is the allocated one");
        // The fixture's 16-wide latent is narrower than one 128-wide FP8 tile, so the
        // allocator's tile guard makes both choices f32; the pack test exercises the
        // tile-wide boundary where E4M3 and f32 differ.
        assert_eq!(fp8_plan, f32_plan, "a narrow latent is f32 in both modes");
        // Two backbone MLA layers and the MTP stage allocate the f32 latent; the first
        // backbone layer owns an indexer and the device's MTP stage always builds one, even
        // though the last backbone MLA layer shares a selection and declares none.
        assert_eq!(
            admitted.per_token(),
            3 * (16_u64 * 4) + 2 * (8_u64 * 8 + (8_u64 * 4).div_ceil(2)),
            "the plan charges the allocator's latent and both indexers"
        );

        // Auto is the per-tensor choice, and a latent width that is not a whole FP8 tile falls
        // back to f32 whatever the caller asks.
        assert!(!device::LatentCache::Fp8.quantises(&tiny, true));
        // `small` has a 128-wide latent, so its choice turns on kv_b alone.
        let small = Glm5NextConfig::small();
        assert_eq!(small.fp8_block, Some(128));
        assert!(
            !device::LatentCache::Auto.quantises(&small, false),
            "Auto reads the per-tensor kv_b, which is BF16, not the checkpoint-wide fp8 flag"
        );
        assert!(
            device::LatentCache::Auto.quantises(&small, true),
            "a block-scaled kv_b makes Auto the E4M3 latent"
        );
        assert!(!device::LatentCache::F32.quantises(&small, true));
        assert!(device::LatentCache::Fp8.quantises(&small, false));
    }
}

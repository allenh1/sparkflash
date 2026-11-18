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

//! `sparkflash serve --model ID`: the model packs that bring their own arm.
//!
//! A pack owns the arithmetic, the chat template and the token ids. The process around it is the
//! same for every pack and lives here once: resolve the checkpoint and its configuration, check
//! the window, build the prompt seam before any weight is uploaded, open the backend, start the
//! driver, serve the router. [`PackArm`] is the part that differs, one impl a pack, and [`Pack`]
//! is the list of packs `--model` can name.
//!
//! An arm serves on one node behind the generic host driver ([`serve_host`]), or as one process of
//! a tensor-parallel group whose rank 0 owns HTTP, the tokenizer, the chat template and streaming
//! ([`serve_group`]). A flag the path would not apply is refused when the options are parsed
//! ([`Pack::unapplied`]), not accepted and ignored.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use sparkflash_api::engine::InferenceEngine as ApiInferenceEngine;
use sparkflash_kernels::images::CUDA_ARCH;
use sparkflash_model::numerics::ConfigKey;
use sparkflash_model::{ModelBackend, ModelDescriptor, ModelProfile, PromptCodec, Target};
use sparkflash_model_glm5_next::Glm5NextConfig;
use sparkflash_model_qwen4_exp::Qwen4ExpConfig;
use sparkflash_ops::EngineBuilder;
use sparkflash_ops::engine_plan::EnginePlan;
use sparkflash_ops::parts::serving_language;
use sparkflash_ops::resolver::{Availability, KernelResolver};
use sparkflash_weights::safetensors::{INDEX_NAME, SafeTensorsSource};

use super::{CHECKPOINT_CONFIG_NAME, EXIT_REFUSED, ServeOptions, serve_with_seam};
#[cfg(feature = "tp4-serve")]
use super::{
    ClusterStatsSnapshot, DSparkStepStats, RequestTimingStats, RoceTransportStats,
    TP4_CONTROL_TIMEOUT, Tp4Driver, Tp4EngineSpec, Tp4RankApi, Tp4RankConfig, Tp4ServeOptions,
    default_engram_layers, empty_kv_snapshot, open_glm_lanes, open_glm_tp4_rank, open_qwen_lanes,
    open_qwen_tp4_rank, pack_tp4_worker, tp4_rank_config,
};

/// The variable naming a directory of Qwen3.8-Flash-Next rank files (`sparkflash convert`).
pub(super) use sparkflash_model_qwen4_exp::rank::CONVERTED_DIR_ENV as QWEN_CONVERTED_DIR_ENV;
/// Where an Inkling run reads its converted rank file, when one is staged.
pub(super) const INKLING_CONVERTED_DIR_ENV: &str = "SPARKFLASH_INKLING_CONVERTED_DIR";

/// The default DeepSeek arm's carried profile. The arm has no [PackArm] (the reference path
/// serves it), but its boot still reads the same typed profile as the packed arms.
pub(super) const DEEPSEEK_PROFILE_TOML: &str =
    include_str!("../../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.toml");

/// The carried profile of the Dory host-reference arm.
#[cfg(test)]
pub(super) const DORY_PROFILE_TOML: &str =
    include_str!("../../../sparkflash-model-dory/config/dory1_0_4b_bf16.toml");

/// The carried profile a checkpoint family serves, when this build carries one.
///
/// A dynamic checkpoint whose `model_type` names a released family still reads that family's
/// typed profile; a novel family gets `ModelProfile::default()` plus the declared environment
/// overrides, which is the documented fallback (issue #17 step 2.2).
pub(super) fn profile_toml_for_family(family: &str) -> Option<&'static str> {
    // The family spelling and the profile beside it are one fact in `ARMS`; a released pack's
    // `PackArm::PROFILE_TOML` is the same TOML the table carries (a test holds them equal).
    sparkflash_model::arm_for_family(family).and_then(|arm| arm.profile_toml())
}

/// Opens one rank of a pack's group stepping `lanes` fused lanes ([`PackArm::open_lanes`]).
///
/// `lanes` is the operator's `--concurrency` when given, else `None` and the arm's own
/// [sparkflash_runtime::Limits] names the width. The answer is rank 0's backend (a worker
/// answers `None` once its mirror ends) and the lane count the group actually opened.
#[cfg(feature = "tp4-serve")]
pub(super) type GroupLanesOpen =
    fn(&Tp4RankConfig, Option<usize>) -> Result<(Option<Box<dyn ModelBackend>>, usize), String>;

/// What one pack's serving arm supplies; [`serve_host`] and [`serve_group`] are the rest.
pub(super) trait PackArm {
    /// The pack's parsed `config.json`.
    type Config;
    /// The pack's single-node device backend.
    type Backend: ModelBackend + 'static;

    /// The id `/v1/models` reports and the hub cache resolves.
    const MODEL_ID: &'static str;
    /// The pack's carried typed profile (the TOML in its own crate).
    const PROFILE_TOML: &'static str;
    /// The name startup messages use.
    const NAME: &'static str;
    /// The tiny fixture's directory stem: `sparkflash-{FIXTURE}-fixture-{pid}`.
    const FIXTURE: &'static str;
    /// The pack's output ceiling; the CLI default belongs to the V4.1 surface and may be larger.
    const MAX_OUTPUT_TOKENS: usize;
    /// The window served without `--context`, capped by the checkpoint's own.
    const DEFAULT_CONTEXT: usize;
    /// Whether the single-node arm steps its sessions as one batch under `--fuse-batch`.
    const FUSED_BATCH: bool;
    /// Whether the single-node arm sizes its caches against `--kv-budget-gib`.
    const KV_BUDGET: bool;
    /// Opens this process's rank of a tensor-parallel group.
    #[cfg(feature = "tp4-serve")]
    const OPEN_RANK: fn(&Tp4RankConfig) -> Result<Box<dyn Tp4RankApi>, String>;
    /// Opens this process's rank of the group's fused lanes under `--fuse-batch`
    /// (sparkflash_runtime::GroupLanes): the operator's lane count, or the count the arm's
    /// [sparkflash_runtime::Limits] states when the operator gave none. Every pack arm opens
    /// lanes, so this has no default: a group either steps its arm's walk or says which world
    /// it cannot serve.
    ///
    /// # Errors
    /// A rank that cannot load the checkpoint, open its collective, or mirror rank 0's lanes.
    #[cfg(feature = "tp4-serve")]
    fn open_lanes(
        config: &Tp4RankConfig,
        lanes: Option<usize>,
    ) -> Result<(Option<Box<dyn ModelBackend>>, usize), String>;

    /// Writes the tiny fixture `--synthetic` serves into `directory`.
    fn write_fixture(directory: &Path) -> Result<Self::Config, String>;
    /// The configuration of the checkpoint in `directory`.
    fn read_config(directory: &Path) -> Result<Self::Config, String>;
    /// The shared architecture the typed configuration carries, with the producer correction the
    /// arm's bridge applies. The boot plan is built from this view, not from a raw re-parse.
    fn architecture(config: &Self::Config) -> &sparkflash_model::Architecture;
    /// The checkpoint's window, `max_position_embeddings`.
    fn window(config: &Self::Config) -> usize;
    /// What a group serves over `context` tokens, or the arm's refusal of a release it has no
    /// identity for.
    fn descriptor(config: &Self::Config, context: usize) -> Result<ModelDescriptor, String>;
    /// The released prompt seam from a tokenizer file and the checkpoint's parsed architecture.
    fn codec(tokenizer: &Path, config: &Self::Config) -> Result<Arc<dyn PromptCodec>, String>;
    /// Refusals the single-node arm makes before it uploads anything.
    fn check_host(
        options: &ServeOptions,
        config: &Self::Config,
        context: usize,
        profile: &ModelProfile,
    ) -> Result<(), String>;
    /// Opens the single-node backend over `context` tokens.
    fn open_host(
        directory: &Path,
        config: Self::Config,
        context: usize,
        profile: &ModelProfile,
        plan: Option<&EnginePlan>,
    ) -> Result<Self::Backend, String>;
    /// Pools `concurrency` sessions into one fused batch.
    fn fuse(backend: &mut Self::Backend, concurrency: usize) -> Result<(), String>;
    /// The pack's response dialect over the driver: its reasoning and tool channels.
    fn respond(driver: Arc<dyn ApiInferenceEngine>) -> Arc<dyn ApiInferenceEngine>;

    /// Refuses a typed profile field this arm reads but cannot honour.
    ///
    /// The check runs wherever the profile is read, so the single-node arm and the group agree:
    /// an unhonoured field is a boot refusal naming the switch, never a setting accepted and
    /// dropped. The default honours every field.
    ///
    /// # Errors
    /// A field this arm cannot serve.
    fn check_profile(_profile: &ModelProfile) -> Result<(), String> {
        Ok(())
    }

    /// Parse the carried profile and apply the process environment as overrides.
    ///
    /// The TOML is the source of truth; the environment only wins where an operator set one of
    /// [sparkflash_model::profile::ENV_OVERRIDES]. An out-of-domain value is refused by
    /// [ModelProfile::validate].
    fn profile_from(lookup: impl Fn(&str) -> Option<String>) -> Result<ModelProfile, String> {
        let mut profile = ModelProfile::from_toml(Self::PROFILE_TOML)
            .map_err(|error| format!("the {} profile: {error}", Self::NAME))?;
        profile
            .apply_env(lookup)
            .map_err(|error| format!("the {} profile: {error}", Self::NAME))?;
        Self::check_profile(&profile)
            .map_err(|error| format!("the {} profile: {error}", Self::NAME))?;
        Ok(profile)
    }

    /// [Self::profile_from] over the process environment.
    fn profile() -> Result<ModelProfile, String> {
        Self::profile_from(|name| std::env::var(name).ok())
    }
}

/// Qwen3.8-Flash-Next, the NVFP4 checkpoint.
pub(super) struct QwenArm;

impl PackArm for QwenArm {
    type Config = Qwen4ExpConfig;
    type Backend = sparkflash_model_qwen4_exp::device::Qwen4ExpDeviceBackend;

    const MODEL_ID: &'static str = sparkflash_model::descriptor::QWEN38_FLASH_NEXT_ID;
    const NAME: &'static str = sparkflash_model_qwen4_exp::rank::ARM_NAME;
    const FIXTURE: &'static str = "qwen4";
    const MAX_OUTPUT_TOKENS: usize = sparkflash_model::descriptor::QWEN38_MAX_OUTPUT_TOKENS;
    const DEFAULT_CONTEXT: usize = sparkflash_model::descriptor::QWEN38_CONTEXT_TOKENS;
    const PROFILE_TOML: &'static str =
        include_str!("../../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.toml");
    const FUSED_BATCH: bool = true;
    const KV_BUDGET: bool = true;
    #[cfg(feature = "tp4-serve")]
    const OPEN_RANK: fn(&Tp4RankConfig) -> Result<Box<dyn Tp4RankApi>, String> = open_qwen_tp4_rank;
    #[cfg(feature = "tp4-serve")]
    fn open_lanes(
        config: &Tp4RankConfig,
        lanes: Option<usize>,
    ) -> Result<(Option<Box<dyn ModelBackend>>, usize), String> {
        open_qwen_lanes(config, lanes)
    }

    fn write_fixture(directory: &Path) -> Result<Self::Config, String> {
        let config = Qwen4ExpConfig::tiny();
        sparkflash_model_qwen4_exp::write_synthetic_checkpoint(directory, &config, 7)?;
        Ok(config)
    }

    fn read_config(directory: &Path) -> Result<Self::Config, String> {
        Qwen4ExpConfig::from_checkpoint_dir(directory).map_err(|error| {
            format!(
                "{}: {error}",
                directory.join(CHECKPOINT_CONFIG_NAME).display()
            )
        })
    }

    fn architecture(config: &Self::Config) -> &sparkflash_model::Architecture {
        &config.architecture
    }

    fn window(config: &Self::Config) -> usize {
        config.max_position_embeddings
    }

    fn descriptor(config: &Self::Config, context: usize) -> Result<ModelDescriptor, String> {
        sparkflash_model_qwen4_exp::device::descriptor_for(config, context)
    }

    fn codec(tokenizer: &Path, config: &Self::Config) -> Result<Arc<dyn PromptCodec>, String> {
        Ok(Arc::new(
            sparkflash_model_qwen4_exp::QwenCodec::from_path_with_architecture(
                tokenizer,
                &config.architecture,
            )?,
        ))
    }

    fn check_host(
        options: &ServeOptions,
        config: &Self::Config,
        context: usize,
        _profile: &ModelProfile,
    ) -> Result<(), String> {
        admit(
            config
                .architecture
                .kv_plan(1, false)
                .map_err(|error| error.to_string())?,
            sparkflash_model_qwen4_exp::CONCURRENCY_KV_BUDGET_BYTES,
            options,
            context,
        )
    }

    fn open_host(
        directory: &Path,
        config: Self::Config,
        context: usize,
        _profile: &ModelProfile,
        plan: Option<&EnginePlan>,
    ) -> Result<Self::Backend, String> {
        use sparkflash_model_qwen4_exp::device::Qwen4ExpDeviceBackend;
        if let Some(plan) = plan {
            check_backend_plan(plan, config.family(), config.num_hidden_layers)?;
        }
        // A converted rank file is one slab read instead of 77.8 GiB of per-tensor uploads. The
        // checkpoint directory stays the configuration, tokenizer and host-paged PLE source.
        let converted =
            sparkflash_weights::rank_file::converted_rank_file(QWEN_CONVERTED_DIR_ENV, 0, 1)?;
        let backend = match (converted.as_deref(), plan) {
            // The plan-aware open is the arm's EngineBuilder, so the plan's target/world gate
            // and the shared rank boot check run before the arm uploads a weight. The converted
            // file is the same builder's arm-level storage path, not a second open.
            (converted, Some(plan)) => {
                let source = SafeTensorsSource::open(directory, &directory.join(INDEX_NAME))
                    .map_err(|error| error.to_string())?;
                match converted {
                    Some(path) => <Qwen4ExpDeviceBackend as EngineBuilder>::open_converted(
                        plan, &source, path, context,
                    ),
                    None => <Qwen4ExpDeviceBackend as EngineBuilder>::open(plan, &source, context),
                }
                .map(|backend| backend.into_arm())
                .map_err(|error| error.to_string())
            }
            (Some(path), None) => Qwen4ExpDeviceBackend::open_converted(directory, context, path)
                .map_err(|error| error.to_string()),
            (None, None) => {
                Qwen4ExpDeviceBackend::open(directory, context).map_err(|error| error.to_string())
            }
        }?;
        if let Some(path) = &converted {
            eprintln!(
                "sparkflash serve: {} weights from converted {}",
                Self::NAME,
                path.display()
            );
        }
        Ok(backend)
    }

    fn fuse(backend: &mut Self::Backend, concurrency: usize) -> Result<(), String> {
        backend
            .enable_fusion(concurrency)
            .map_err(|error| error.to_string())
    }

    fn respond(driver: Arc<dyn ApiInferenceEngine>) -> Arc<dyn ApiInferenceEngine> {
        sparkflash_api::qwen::QwenResponseEngine::layer(driver)
    }
}

/// Qwen3.8-Flash-Next, the official FP8 checkpoint: the same arm and profile with the FP8
/// release's served id, so --model qwen3.8-flash-next-fp8 resolves the FP8 snapshot rather than
/// the NVFP4 one. The checkpoint's own quantization_config selects the expert format.
pub(super) struct QwenFp8Arm;

impl PackArm for QwenFp8Arm {
    type Config = Qwen4ExpConfig;
    type Backend = sparkflash_model_qwen4_exp::device::Qwen4ExpDeviceBackend;

    const MODEL_ID: &'static str = sparkflash_model::descriptor::QWEN38_FLASH_NEXT_FP8_ID;
    const NAME: &'static str = "Qwen3.8-Flash-Next (FP8)";
    const FIXTURE: &'static str = QwenArm::FIXTURE;
    const MAX_OUTPUT_TOKENS: usize = QwenArm::MAX_OUTPUT_TOKENS;
    const DEFAULT_CONTEXT: usize = QwenArm::DEFAULT_CONTEXT;
    const PROFILE_TOML: &'static str = QwenArm::PROFILE_TOML;
    const FUSED_BATCH: bool = QwenArm::FUSED_BATCH;
    const KV_BUDGET: bool = QwenArm::KV_BUDGET;
    #[cfg(feature = "tp4-serve")]
    const OPEN_RANK: fn(&Tp4RankConfig) -> Result<Box<dyn Tp4RankApi>, String> = QwenArm::OPEN_RANK;
    #[cfg(feature = "tp4-serve")]
    fn open_lanes(
        config: &Tp4RankConfig,
        lanes: Option<usize>,
    ) -> Result<(Option<Box<dyn ModelBackend>>, usize), String> {
        <QwenArm as PackArm>::open_lanes(config, lanes)
    }

    fn write_fixture(directory: &Path) -> Result<Self::Config, String> {
        <QwenArm as PackArm>::write_fixture(directory)
    }

    fn read_config(directory: &Path) -> Result<Self::Config, String> {
        <QwenArm as PackArm>::read_config(directory)
    }

    fn architecture(config: &Self::Config) -> &sparkflash_model::Architecture {
        <QwenArm as PackArm>::architecture(config)
    }

    fn window(config: &Self::Config) -> usize {
        <QwenArm as PackArm>::window(config)
    }

    fn descriptor(config: &Self::Config, context: usize) -> Result<ModelDescriptor, String> {
        <QwenArm as PackArm>::descriptor(config, context)
    }

    fn codec(tokenizer: &Path, config: &Self::Config) -> Result<Arc<dyn PromptCodec>, String> {
        <QwenArm as PackArm>::codec(tokenizer, config)
    }

    fn check_host(
        options: &ServeOptions,
        config: &Self::Config,
        context: usize,
        profile: &ModelProfile,
    ) -> Result<(), String> {
        <QwenArm as PackArm>::check_host(options, config, context, profile)
    }

    fn open_host(
        directory: &Path,
        config: Self::Config,
        context: usize,
        profile: &ModelProfile,
        plan: Option<&EnginePlan>,
    ) -> Result<Self::Backend, String> {
        <QwenArm as PackArm>::open_host(directory, config, context, profile, plan)
    }

    fn fuse(backend: &mut Self::Backend, concurrency: usize) -> Result<(), String> {
        <QwenArm as PackArm>::fuse(backend, concurrency)
    }

    fn respond(driver: Arc<dyn ApiInferenceEngine>) -> Arc<dyn ApiInferenceEngine> {
        <QwenArm as PackArm>::respond(driver)
    }
}

/// Qwen3.6-35B-A3B, the official FP8 checkpoint: the Qwen arm's device backend over the
/// Qwen3.5-MoE family (plain residual, full gated attention, E4M3 128x128-tile dense and expert
/// weights). The family and stored formats select the kernels; [identity_for] refuses a release
/// of the family this build has no kernel or identity for.
///
/// [identity_for]: sparkflash_model_qwen4_exp::device::identity_for
pub(super) struct Qwen36Arm;

impl PackArm for Qwen36Arm {
    type Config = Qwen4ExpConfig;
    type Backend = sparkflash_model_qwen4_exp::device::Qwen4ExpDeviceBackend;

    const MODEL_ID: &'static str = sparkflash_model::descriptor::QWEN36_FLASH_35B_A3B_FP8_ID;
    const NAME: &'static str = "Qwen3.6-35B-A3B (FP8)";
    const FIXTURE: &'static str = "qwen36";
    const MAX_OUTPUT_TOKENS: usize = sparkflash_model::descriptor::QWEN36_MAX_OUTPUT_TOKENS;
    const DEFAULT_CONTEXT: usize = sparkflash_model::descriptor::QWEN36_CONTEXT_TOKENS;
    const PROFILE_TOML: &'static str =
        include_str!("../../../sparkflash-model-qwen4-exp/config/qwen36_35b_a3b_fp8.toml");
    const FUSED_BATCH: bool = QwenArm::FUSED_BATCH;
    const KV_BUDGET: bool = QwenArm::KV_BUDGET;
    #[cfg(feature = "tp4-serve")]
    const OPEN_RANK: fn(&Tp4RankConfig) -> Result<Box<dyn Tp4RankApi>, String> = QwenArm::OPEN_RANK;
    #[cfg(feature = "tp4-serve")]
    fn open_lanes(
        config: &Tp4RankConfig,
        lanes: Option<usize>,
    ) -> Result<(Option<Box<dyn ModelBackend>>, usize), String> {
        <QwenArm as PackArm>::open_lanes(config, lanes)
    }

    fn write_fixture(directory: &Path) -> Result<Self::Config, String> {
        let config = Qwen4ExpConfig::tiny_qwen35_moe(
            sparkflash_model_qwen4_exp::config::DenseQuant::Fp8Block,
        );
        sparkflash_model_qwen4_exp::write_synthetic_checkpoint(directory, &config, 7)?;
        Ok(config)
    }

    fn read_config(directory: &Path) -> Result<Self::Config, String> {
        <QwenArm as PackArm>::read_config(directory)
    }

    fn architecture(config: &Self::Config) -> &sparkflash_model::Architecture {
        <QwenArm as PackArm>::architecture(config)
    }

    fn window(config: &Self::Config) -> usize {
        <QwenArm as PackArm>::window(config)
    }

    fn descriptor(config: &Self::Config, context: usize) -> Result<ModelDescriptor, String> {
        <QwenArm as PackArm>::descriptor(config, context)
    }

    fn codec(tokenizer: &Path, config: &Self::Config) -> Result<Arc<dyn PromptCodec>, String> {
        <QwenArm as PackArm>::codec(tokenizer, config)
    }

    fn check_host(
        options: &ServeOptions,
        config: &Self::Config,
        context: usize,
        profile: &ModelProfile,
    ) -> Result<(), String> {
        <QwenArm as PackArm>::check_host(options, config, context, profile)
    }

    fn open_host(
        directory: &Path,
        config: Self::Config,
        context: usize,
        profile: &ModelProfile,
        plan: Option<&EnginePlan>,
    ) -> Result<Self::Backend, String> {
        <QwenArm as PackArm>::open_host(directory, config, context, profile, plan)
    }

    fn fuse(backend: &mut Self::Backend, concurrency: usize) -> Result<(), String> {
        <QwenArm as PackArm>::fuse(backend, concurrency)
    }

    fn respond(driver: Arc<dyn ApiInferenceEngine>) -> Arc<dyn ApiInferenceEngine> {
        <QwenArm as PackArm>::respond(driver)
    }
}

/// GLM-5.3-Flash, the FP8 checkpoint.
///
/// The release is 328.3 GB, so one node holds only the tiny fixture; the release is served by the
/// four-rank group.
pub(super) struct GlmArm;

impl PackArm for GlmArm {
    type Config = Glm5NextConfig;
    type Backend = sparkflash_model_glm5_next::Glm5NextDeviceBackend;

    const MODEL_ID: &'static str = sparkflash_model::descriptor::GLM53_FLASH_ID;
    const NAME: &'static str = sparkflash_model_glm5_next::rank::ARM_NAME;
    const FIXTURE: &'static str = "glm53";
    const MAX_OUTPUT_TOKENS: usize = sparkflash_model::descriptor::GLM53_MAX_OUTPUT_TOKENS;
    const DEFAULT_CONTEXT: usize = sparkflash_model::descriptor::GLM53_CONTEXT_TOKENS;
    const PROFILE_TOML: &'static str =
        include_str!("../../../sparkflash-model-glm5-next/config/glm53_flash.toml");
    const FUSED_BATCH: bool = true;
    const KV_BUDGET: bool = true;
    #[cfg(feature = "tp4-serve")]
    const OPEN_RANK: fn(&Tp4RankConfig) -> Result<Box<dyn Tp4RankApi>, String> = open_glm_tp4_rank;
    #[cfg(feature = "tp4-serve")]
    fn open_lanes(
        config: &Tp4RankConfig,
        lanes: Option<usize>,
    ) -> Result<(Option<Box<dyn ModelBackend>>, usize), String> {
        open_glm_lanes(config, lanes)
    }

    fn check_profile(profile: &ModelProfile) -> Result<(), String> {
        // The ledger's tensor-core decode row, the serving walk's capability and this check are
        // one constant ([sparkflash_model_glm5_next::TOKEN_DECODE]): GLM's row states the tile's
        // few-row body, so `decode.tensor_cores = true` is accepted and runs it. A row that ever
        // goes off refuses the switch with the row's own reason.
        let sparkflash_model::ledger::State::Off(why) = sparkflash_model_glm5_next::TOKEN_DECODE
        else {
            return Ok(());
        };
        if profile.decode.tensor_cores == Some(true) {
            return Err(format!(
                "the GLM arm ({}) cannot honour decode.tensor_cores = true: {why}. Unset \
                 SPARKFLASH_DECODE_MMA, or set it to off, to run the row GEMV the arm serves",
                Self::MODEL_ID
            ));
        }
        Ok(())
    }

    fn write_fixture(directory: &Path) -> Result<Self::Config, String> {
        let config = Glm5NextConfig::tiny();
        sparkflash_model_glm5_next::checkpoint::write_synthetic_checkpoint(
            directory, &config, 7, 2,
        )?;
        Ok(config)
    }

    /// The checkpoint's `config.json`. A checkpoint without one is refused rather than served
    /// from a compiled-in release document.
    fn read_config(directory: &Path) -> Result<Self::Config, String> {
        let path = directory.join(CHECKPOINT_CONFIG_NAME);
        if !path.is_file() {
            return Err(format!(
                "{} has no {CHECKPOINT_CONFIG_NAME}: a GLM-5.3 checkpoint is read from its own configuration, not the carried release",
                directory.display()
            ));
        }
        Glm5NextConfig::from_checkpoint_dir(directory)
            .map_err(|error| format!("{}: {error}", path.display()))
    }

    fn architecture(config: &Self::Config) -> &sparkflash_model::Architecture {
        &config.architecture
    }

    fn window(config: &Self::Config) -> usize {
        config.max_position_embeddings
    }

    fn descriptor(config: &Self::Config, context: usize) -> Result<ModelDescriptor, String> {
        Ok(sparkflash_model_glm5_next::serve::descriptor_for(
            config, context,
        ))
    }

    fn codec(tokenizer: &Path, config: &Self::Config) -> Result<Arc<dyn PromptCodec>, String> {
        Ok(Arc::new(
            sparkflash_model_glm5_next::GlmCodec::from_path_with_architecture(
                tokenizer,
                &config.architecture,
            )?,
        ))
    }

    fn check_host(
        options: &ServeOptions,
        config: &Self::Config,
        context: usize,
        _profile: &ModelProfile,
    ) -> Result<(), String> {
        // The FP8 release is 82 GB a rank over four nodes. One rank holding all four shards is
        // not the served shape, and the server says so before it tries a 328.3 GB upload.
        if config.num_hidden_layers >= 45 {
            return Err(format!(
                "{} is the released 321B-parameter FP8 checkpoint (328.3 GB): one TP=1 rank cannot hold it. Serve the tiny fixture at TP=1 (--synthetic), or serve the release with the four-rank group: build with --features tp4-serve and pass --tp4-rank/--tp4-world-size/--tp4-peers with --checkpoint.",
                Self::MODEL_ID
            ));
        }
        // Each worker opens its own window: at the 1M default that is 18.8 GiB of caches a
        // session, so a worker count the node cannot hold is refused before the upload rather
        // than at the first request.
        //
        // The admitted plan is the one the single-node rank sizes its session caches by:
        // [Glm5NextDeviceBackend] opens TP=1 with the explicit F32 latent, and
        // [single_rank_kv_plan] is the same choice. It is the shared plan's cache ledger, not
        // every device allocation (its documented Qwen GDN/PLE exclusions do not apply here).
        // The profile's `attention.latent_cache` belongs to the TP=4 arm; the single-node
        // backend does not read it. The checkpoint-wide `fp8_block` flag must not choose here:
        // it tiles the block-scaled projections, while `kv_b_proj.weight` is BF16 in the tensor
        // plan, so deriving E4M3 from it under-admits the f32 latent by 18,240 bytes a token at
        // the release geometry.
        admit(
            sparkflash_model_glm5_next::single_rank_kv_plan(config),
            sparkflash_model_glm5_next::CONCURRENCY_KV_BUDGET_BYTES,
            options,
            context,
        )
    }

    fn open_host(
        directory: &Path,
        config: Self::Config,
        context: usize,
        profile: &ModelProfile,
        plan: Option<&EnginePlan>,
    ) -> Result<Self::Backend, String> {
        if let Some(plan) = plan {
            check_backend_plan(plan, "glm5_next", config.num_hidden_layers)?;
        }
        let mut backend = match plan {
            // The plan-aware open is the arm's EngineBuilder: its plan gate and the shared rank
            // boot check run before the arm opens a device.
            Some(plan) => {
                let source = SafeTensorsSource::open(directory, &directory.join(INDEX_NAME))
                    .map_err(|error| error.to_string())?;
                <sparkflash_model_glm5_next::Glm5NextDeviceBackend as EngineBuilder>::open(
                    plan, &source, context,
                )
                .map(|backend| backend.into_arm())
                .map_err(|error| error.to_string())
            }
            None => {
                let cuda = sparkflash_kernels::cuda_driver::Cuda::load()
                    .map_err(|error| format!("CUDA: {error}"))?;
                let cuda = Arc::new(sparkflash_model_glm5_next::device::CudaHandle::new(cuda));
                sparkflash_model_glm5_next::Glm5NextDeviceBackend::open_single_rank(
                    cuda, config, directory, context,
                )
                .map_err(|error| error.to_string())
            }
        }?;
        apply_speculation(&mut backend, profile)?;
        Ok(backend)
    }

    fn fuse(backend: &mut Self::Backend, concurrency: usize) -> Result<(), String> {
        backend
            .enable_fusion(concurrency)
            .map_err(|error| error.to_string())
    }

    fn respond(driver: Arc<dyn ApiInferenceEngine>) -> Arc<dyn ApiInferenceEngine> {
        // The dialect splits the model reasoning block into reasoning_content and lifts its
        // <tool_call> XML into native tool calls (sparkflash_api::glm).
        sparkflash_api::glm::GlmResponseEngine::layer(driver)
    }
}

/// Whether the Qwen3.5-MoE checkpoint in `directory` is one the device arm serves.
///
/// The family is many releases and formats; the device arm takes the ones it has kernels and an
/// identity for ([identity_for]), and the rest (the BF16 release) stay on the dynamic host path
/// that served the family before the arm.
///
/// [identity_for]: sparkflash_model_qwen4_exp::device::identity_for
pub(super) fn qwen35_on_device(directory: &Path) -> bool {
    Qwen4ExpConfig::from_checkpoint_dir(directory)
        .ok()
        .is_some_and(|config| sparkflash_model_qwen4_exp::device::identity_for(&config).is_ok())
}

/// The lane count the options record before an arm answers, and the dynamic dense arm's own
/// default: four.
///
/// A pack group's real width is its arm's [sparkflash_runtime::Limits], read when the rank
/// opens ([PackArm::open_lanes]); this is what `ServeOptions` holds when the operator gave no
/// `--concurrency`, so flag validation has a number before any device is opened.
pub(super) const GROUP_LANES: usize = 4;

/// One typed pack: a [PackArm] this build carries, selected by its registry weights selector.
struct TypedPack {
    /// The weights selector that names this entry's [sparkflash_model::Arm].
    weights: &'static str,
    /// Whether the single-node arm steps its sessions as one batch under `--fuse-batch`.
    fused_batch: bool,
    /// Whether the single-node arm sizes its caches against `--kv-budget-gib`.
    kv_budget: bool,
    /// Serves this pack through [serve_arm] or [serve_arm_identity].
    serve: fn(&ServeOptions) -> Result<ExitCode, String>,
}

/// The typed packs this build serves, one entry a [PackArm].
///
/// A new family adds an [sparkflash_model::ARMS] entry and, when it brings its own arithmetic and
/// boot, one entry here plus its `PackArm` impl. The [Pack] identity is the descriptor's weights
/// selector, so a second release of one arm (Qwen's FP8 snapshot) is a selector, not a variant.
static TYPED_PACKS: &[TypedPack] = &[
    TypedPack {
        weights: sparkflash_model::descriptor::QWEN38_FLASH_NEXT_WEIGHTS_ID,
        fused_batch: QwenArm::FUSED_BATCH,
        kv_budget: QwenArm::KV_BUDGET,
        serve: serve_arm::<QwenArm>,
    },
    TypedPack {
        weights: sparkflash_model::descriptor::QWEN38_FLASH_NEXT_FP8_WEIGHTS_ID,
        fused_batch: QwenFp8Arm::FUSED_BATCH,
        kv_budget: QwenFp8Arm::KV_BUDGET,
        serve: serve_arm::<QwenFp8Arm>,
    },
    TypedPack {
        weights: sparkflash_model::descriptor::QWEN36_FLASH_35B_A3B_FP8_WEIGHTS_ID,
        fused_batch: Qwen36Arm::FUSED_BATCH,
        kv_budget: Qwen36Arm::KV_BUDGET,
        serve: serve_arm::<Qwen36Arm>,
    },
    TypedPack {
        weights: sparkflash_model::descriptor::GLM53_FLASH_WEIGHTS_ID,
        fused_batch: GlmArm::FUSED_BATCH,
        kv_budget: GlmArm::KV_BUDGET,
        serve: serve_arm::<GlmArm>,
    },
];

/// The typed pack a weights selector names, if this build carries its [PackArm].
fn typed_pack(weights: &str) -> Option<&'static TypedPack> {
    TYPED_PACKS
        .iter()
        .find(|entry| entry.weights.eq_ignore_ascii_case(weights))
}

/// How a pack's process is served.
///
/// The variants are serving shapes, not families: every family is a descriptor and an
/// [sparkflash_model::Arm] entry, and only the shape changes which code boots it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Route {
    /// The V4.1 reference/TP4 engine, which serves the default DeepSeek arm.
    Reference,
    /// A typed [PackArm] behind the generic host driver (the Qwen releases and GLM).
    Typed,
    /// The dynamic path: a checkpoint's own arm (Dory, Inkling, or a novel family).
    Dynamic,
}

/// The packs the model registry knows, including the default DeepSeek arm.
///
/// A pack is a registry descriptor and the [sparkflash_model::Arm] that answers for its weights;
/// every id, weights selector, model type, label, context and vocabulary is read from
/// [sparkflash_model::MODELS] and [sparkflash_model::ARMS], never from a second table. A new
/// family is an entry there, not a variant here.
#[derive(Clone, Copy)]
pub(super) struct Pack {
    descriptor: &'static ModelDescriptor,
    route: Route,
}

impl std::fmt::Debug for Pack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pack")
            .field("model", &self.model_id())
            .field("route", &self.route)
            .finish()
    }
}

impl PartialEq for Pack {
    fn eq(&self, other: &Self) -> bool {
        self.descriptor.weights_id == other.descriptor.weights_id
    }
}

impl Eq for Pack {}

impl Pack {
    /// This pack's descriptor in [sparkflash_model::MODELS].
    #[cfg(test)]
    pub(super) fn descriptor(self) -> &'static ModelDescriptor {
        self.descriptor
    }

    /// The [sparkflash_model::Arm] entry that answers for this pack's weights.
    #[cfg(test)]
    pub(super) fn arm(self) -> &'static dyn sparkflash_model::Arm {
        sparkflash_model::arm_for_weights(&self.descriptor.weights_id)
            .expect("a pack's weights name an ARMS entry")
    }

    /// The serving shape this pack's descriptor has.
    #[cfg(test)]
    pub(super) fn route(self) -> Route {
        self.route
    }

    /// The pack serving a registry descriptor, or `None` for a weights selector with no arm.
    fn from_descriptor(descriptor: &'static ModelDescriptor) -> Option<Self> {
        sparkflash_model::arm_for_weights(&descriptor.weights_id)?;
        let route = if descriptor.weights_id == sparkflash_model::descriptor::DEFAULT_WEIGHTS_ID {
            Route::Reference
        } else if typed_pack(&descriptor.weights_id).is_some() {
            Route::Typed
        } else {
            Route::Dynamic
        };
        Some(Self { descriptor, route })
    }

    /// The pack a checkpoint's own `config.json` names, if this build has its arm.
    ///
    /// This is what makes `--checkpoint DIR` enough: the released document names its family
    /// (`model_type`), and the registry maps that family to the identity and arm. This is not a
    /// substring scan over model names: the three carried released configs are recognized by
    /// the family their own architecture reports.
    pub(super) fn detect(directory: &Path) -> Option<Self> {
        let path = directory.join(CHECKPOINT_CONFIG_NAME);
        let text = std::fs::read_to_string(path).ok()?;
        let architecture = sparkflash_model::Architecture::from_json(&text).ok()?;
        let descriptor = sparkflash_model::find_model_by_type(&architecture.family)?;
        let pack = Self::from_descriptor(descriptor)?;
        if pack.has_weights(sparkflash_model::descriptor::QWEN36_FLASH_35B_A3B_FP8_WEIGHTS_ID)
            && !qwen35_on_device(directory)
        {
            return None;
        }
        Some(pack)
    }

    /// Whether this pack serves the registry selector `weights`.
    fn has_weights(self, weights: &str) -> bool {
        self.descriptor.weights_id == weights
    }

    /// The parts a checkpoint's own `config.json` names that this build cannot load.
    ///
    /// The model-side `Architecture` says what the checkpoint is; the ops catalogue says which
    /// image implements each part, and [sparkflash_ops::parts::missing_loadable] keeps only the
    /// parts with an image this build can actually load (an embedded image, or the DeepSeek arm's
    /// run-time cubin directory). An empty list is the whole bring-up condition.
    ///
    /// The question is about the *served* parts, not every part the document names. The Qwen and
    /// GLM packs do not stage a vision tower, so a document whose only missing image is the
    /// tower's DeepSeek cubin (absent without `SPARKFLASH_CUBINS`) must still parse and boot.
    /// The DeepSeek pack keeps the full set, because its device arm runs the tower.
    /// A document that does not parse names no parts here; [Self::detect] already refused it.
    pub(super) fn missing_parts(directory: &Path) -> Vec<sparkflash_model::Part> {
        let path = directory.join(CHECKPOINT_CONFIG_NAME);
        let Ok(text) = std::fs::read_to_string(path) else {
            return Vec::new();
        };
        let Ok(architecture) = sparkflash_model::Architecture::from_json(&text) else {
            return Vec::new();
        };
        let served = match Self::detect(directory) {
            // Without a Dory/Qwen cubin the dynamic route opens Dory's CPU host reference,
            // which needs no kernel image. With a device arm, the catalogue is the gate.
            Some(pack)
                if pack.has_weights(sparkflash_model::descriptor::DORY1_0_4B_BF16_WEIGHTS_ID)
                    && !sparkflash_model_dory::DoryDeviceBackend::available() =>
            {
                return Vec::new();
            }
            // The Inkling pack's host reference is the initial serving path; its device arm is
            // gated by the same catalogue once it lands.
            Some(pack)
                if pack.has_weights(sparkflash_model::descriptor::INKLING2_SMALL_WEIGHTS_ID) =>
            {
                return Vec::new();
            }
            Some(pack) if pack.route == Route::Typed => {
                sparkflash_ops::parts::serving_language(sparkflash_model::Target::Sm121)
            }
            Some(_) | None => sparkflash_ops::parts::serving(sparkflash_model::Target::Sm121),
        };
        sparkflash_ops::parts::missing_loadable(&architecture, sparkflash_model::Target::Sm121)
            .into_iter()
            .filter(|part| served.contains(*part))
            .collect()
    }

    /// Whether this pack has a typed [PackArm] (the DeepSeek arm is the reference path).
    pub(super) const fn has_pack_arm(self) -> bool {
        matches!(self.route, Route::Typed)
    }

    /// Whether this pack is served by the V4.1 reference/TP4 engine.
    pub(super) const fn is_reference(self) -> bool {
        matches!(self.route, Route::Reference)
    }

    /// Whether this pack is Inkling-2-Small, whose `--rank` device group is its own path.
    pub(super) fn is_inkling(self) -> bool {
        self.has_weights(sparkflash_model::descriptor::INKLING2_SMALL_WEIGHTS_ID)
    }

    /// Whether this pack is Dory1.0-4B, whose `--recurrence-rank` ring is its own path.
    pub(super) fn is_dory(self) -> bool {
        self.has_weights(sparkflash_model::descriptor::DORY1_0_4B_BF16_WEIGHTS_ID)
    }

    /// The served id of this pack.
    pub(super) fn model_id(self) -> &'static str {
        self.descriptor.id.as_str()
    }

    /// The context one sequence of this arm allocates when the operator gave no `--context`.
    ///
    /// It is the arm's own default from the registry, not another model's: the `--tp4-rank`
    /// group used to take GLM's window whatever arm it served.
    pub(super) fn default_context(self) -> usize {
        self.descriptor.context_tokens
    }

    /// The pack `model` names: a registry model id, weights id, family or id leaf, in any case.
    pub(super) fn named(model: &str) -> Option<Self> {
        let descriptor = sparkflash_model::find_model(model)?;
        Self::from_descriptor(descriptor)
    }

    /// Whether this pack's `--tp4-rank` group opens its ranks as fused lanes
    /// ([PackArm::open_lanes]). Every typed pack arm opens lanes; such a group steps the
    /// operator's `--concurrency` or the arm's [sparkflash_runtime::Limits] otherwise, and
    /// `--concurrency 1` serves one request at a time.
    pub(super) fn opens_group_lanes(self) -> bool {
        #[cfg(feature = "tp4-serve")]
        {
            self.has_pack_arm()
        }
        #[cfg(not(feature = "tp4-serve"))]
        {
            let _ = self;
            false
        }
    }

    /// The flags among those given that this pack's arm would not apply: on one node, the
    /// knobs its host arm lacks; in a group, which serves one request at a time over a window
    /// every rank allocates, `--concurrency` above 1, `--fuse-batch` and `--kv-budget-gib`.
    ///
    /// `device_group` is the `--rank` group of the Inkling device arm, which steps concurrent
    /// requests as fused lanes on every rank, as GLM's `--tp4-rank` group does: they take
    /// `--concurrency` with `--fuse-batch`, and still refuse `--concurrency` alone, which would
    /// interleave sequences in their lockstep.
    pub(super) fn unapplied(
        self,
        group: bool,
        device_group: bool,
        concurrency: usize,
        fuse_batch: bool,
        kv_budget: bool,
    ) -> Vec<&'static str> {
        let (fuses, budgets) = typed_pack(self.descriptor.weights_id.as_str())
            .map_or((false, false), |pack| (pack.fused_batch, pack.kv_budget));
        // The groups that step concurrent requests as fused lanes: the Inkling device group, and a
        // `--tp4-rank` group whose arm opens its ranks as lanes ([PackArm::open_lanes]).
        let tp4_lanes = !device_group && self.opens_group_lanes();
        let lanes = tp4_lanes || (device_group && self.is_inkling());
        let mut unapplied = Vec::new();
        if group && concurrency != 1 && !(lanes && fuse_batch) {
            unapplied.push("--concurrency");
        }
        if fuse_batch && if group { !lanes } else { !fuses } {
            unapplied.push("--fuse-batch");
        }
        if kv_budget && (group || !budgets) {
            unapplied.push("--kv-budget-gib");
        }
        unapplied
    }

    /// Serves this pack: its group arm under `--tp4-rank`, else its single-node arm.
    pub(super) fn serve(self, options: &ServeOptions) -> ExitCode {
        let served = match self.route {
            Route::Reference => {
                unreachable!("the DeepSeek pack is served by the default reference path")
            }
            Route::Typed => {
                let pack = typed_pack(self.descriptor.weights_id.as_str())
                    .expect("a typed pack is in TYPED_PACKS");
                (pack.serve)(options)
            }
            Route::Dynamic => {
                return super::dynamic::serve(options).unwrap_or_else(|message| {
                    eprintln!("sparkflash serve: {message}");
                    ExitCode::from(EXIT_REFUSED)
                });
            }
        };
        served.unwrap_or_else(|message| {
            eprintln!("sparkflash serve: {message}");
            ExitCode::from(EXIT_REFUSED)
        })
    }
}

fn serve_arm<A: PackArm>(options: &ServeOptions) -> Result<ExitCode, String> {
    #[cfg(feature = "tp4-serve")]
    if let Some(tp4) = options.tp4.as_ref() {
        return serve_group::<A>(options, tp4);
    }
    serve_host::<A>(options)
}

/// The directory a checkpoint path names: the path itself, or the directory holding the file.
///
/// The rule is the rank seam's ([sparkflash_serve::tp4::checkpoint_directory]), because an arm's
/// rank opener resolves the same path the single-node arm does.
pub(super) use sparkflash_serve::tp4::checkpoint_directory;

/// The configuration of the checkpoint in `directory`, with the pack named in a refusal.
pub(super) fn read_config<A: PackArm>(directory: &Path) -> Result<A::Config, String> {
    A::read_config(directory)
        .map_err(|error| format!("cannot read the {} configuration: {error}", A::NAME))
}

/// Turns the profile's speculation section into the GLM arm's drafter and depth.
///
/// The fields are the typed ones ([sparkflash_model::profile::SpeculationProfile]) and the
/// environment names that set them are declared in [sparkflash_model::profile::ENV_OVERRIDES], so
/// a switch here is a documented field rather than a new variable read at the point of use. The
/// drafts and their depth rule are [sparkflash_model_glm5_next::glm_drafting]'s, which the
/// four-rank group reads too.
fn apply_speculation(
    backend: &mut sparkflash_model_glm5_next::Glm5NextDeviceBackend,
    profile: &ModelProfile,
) -> Result<(), String> {
    let spec = &profile.speculation;
    if let Some(drafter) = spec.drafter.as_deref() {
        backend
            .load_block_drafter(Path::new(drafter))
            .map_err(|error| format!("the DFlash 2 drafter {drafter}: {error}"))?;
        eprintln!("sparkflash serve: GLM drafting with the DFlash 2 drafter {drafter}");
    }
    let (block, most) = backend
        .drafting_limits()
        .map_err(|error| error.to_string())?;
    let Some(drafting) = sparkflash_model_glm5_next::glm_drafting(spec, block, most)? else {
        return Ok(());
    };
    eprintln!("sparkflash serve: GLM speculating {}", drafting.label());
    backend
        .enable_speculation(drafting.drafts, Some(drafting.policy))
        .map_err(|error| error.to_string())
}

/// The one KV admission rule every single-node arm runs before it uploads a weight.
///
/// The pack supplies its own [KvPlan](sparkflash_model::KvPlan) out of its own geometry and the
/// budget it documents; the arithmetic, the announcement and the refusal's wording live in
/// [sparkflash_model::arm] so two arms cannot disagree about what a window costs.
pub(super) fn admit(
    plan: sparkflash_model::KvPlan,
    default_budget: u64,
    options: &ServeOptions,
    context: usize,
) -> Result<(), String> {
    let budget = options
        .kv_budget_gib
        .map_or(default_budget, sparkflash_model::kv_budget_bytes);
    let claim = plan
        .admit(context, options.concurrency, budget)
        .map_err(|refusal| refusal.message())?;
    eprintln!(
        "sparkflash serve: KV cache {} ({concurrency} x {context} tokens), budget {}",
        sparkflash_model::gib(claim.total),
        sparkflash_model::gib(claim.budget),
        concurrency = options.concurrency,
    );
    Ok(())
}

/// The window this run serves: `--context`, else the pack's default capped by the checkpoint's.
pub(super) fn pack_context<A: PackArm>(
    options: &ServeOptions,
    config: &A::Config,
) -> Result<usize, String> {
    let window = A::window(config);
    let context = options.context.unwrap_or(A::DEFAULT_CONTEXT.min(window));
    if context == 0 || context > window {
        return Err(format!("context limit {context} is outside 1..={window}"));
    }
    Ok(context)
}

/// The released prompt seam: `--tokenizer`, else the `tokenizer.json` beside the checkpoint.
///
/// It is built before the checkpoint is opened. Opening a released checkpoint uploads tens of
/// gigabytes; discovering afterwards that the tokenizer is missing (or names a vocabulary whose
/// special ids disagree with the configuration) would waste the upload and leave the GPU full.
fn pack_seam<A: PackArm>(
    options: &ServeOptions,
    directory: &Path,
    config: &A::Config,
) -> Result<(PathBuf, Arc<dyn PromptCodec>), String> {
    let tokenizer = options
        .tokenizer
        .clone()
        .unwrap_or_else(|| directory.join("tokenizer.json"));
    if !tokenizer.is_file() {
        return Err(format!(
            "--model {} needs --tokenizer FILE, or a tokenizer.json beside the checkpoint at {}",
            A::MODEL_ID,
            tokenizer.display()
        ));
    }
    let seam = A::codec(&tokenizer, config)?;
    Ok((tokenizer, seam))
}

/// The checkpoint a single-node arm serves: the tiny fixture under `--synthetic`, else
/// `--checkpoint`, else the hub cache's snapshot of the pack's id.
fn host_checkpoint<A: PackArm>(options: &ServeOptions) -> Result<(PathBuf, A::Config), String> {
    if options.synthetic {
        let directory = std::env::temp_dir().join(format!(
            "sparkflash-{}-fixture-{}",
            A::FIXTURE,
            std::process::id()
        ));
        let config = A::write_fixture(&directory).map_err(|error| {
            format!(
                "cannot write the tiny {} fixture to {}: {error}",
                A::NAME,
                directory.display()
            )
        })?;
        eprintln!(
            "sparkflash serve: serving the tiny {} fixture at {} with a byte codec: this exercises the serving path, it is not the released model",
            A::NAME,
            directory.display()
        );
        return Ok((directory, config));
    }
    let directory = match options.checkpoint.as_deref() {
        Some(path) => checkpoint_directory(path),
        None => {
            // The caller's id, not the arm's compile-time one: a cached fine-tune named on the
            // command line must load its own snapshot (issue #17). A canonical weight selector
            // still maps to its registry id, so the alias and the served id resolve alike.
            let requested = options.model.as_deref().unwrap_or(A::MODEL_ID);
            let id = sparkflash_model::find_model(requested)
                .map_or_else(|| requested.to_owned(), |model| model.id.clone());
            sparkflash_weights::hub::cached_model_dir(&id).ok_or_else(|| {
                format!(
                    "{id} is not in the Hugging Face cache; run hf download {id} or pass --checkpoint DIR"
                )
            })?
        }
    };
    let config = read_config::<A>(&directory)?;
    Ok((directory, config))
}

/// Refuse a backend whose typed configuration disagrees with the plan it was opened from.
///
/// The plan is the boot contract; the typed config is still the arm's weight recipe. This is the
/// seam where the two meet, so a family or layer-count drift is named before a weight is read.
fn check_backend_plan(plan: &EnginePlan, family: &str, layers: usize) -> Result<(), String> {
    // A text-only release names its family `<family>_text` at the document root while the arm
    // and its plan use the multimodal spelling. Fold the alias on both sides so a flat
    // text-only checkpoint is not refused after detection accepts it.
    if !sparkflash_model::canonical_model_type(plan.family())
        .eq_ignore_ascii_case(sparkflash_model::canonical_model_type(family))
    {
        return Err(format!(
            "the execution plan is {}, the backend is {family}",
            plan.family()
        ));
    }
    if plan.layers().len() != layers {
        return Err(format!(
            "the execution plan has {} layers, the backend has {layers}",
            plan.layers().len()
        ));
    }
    Ok(())
}

/// Check the checkpoint's execution plan against the kernel catalogue at `world`.
///
/// This is the boot-time use of the model-independent plan: every part instance the arm
/// executes is resolved to a loadable image before the arm opens a weight. The check is skipped
/// for the synthetic fixture, which exercises the serving path without claiming this build's
/// device images.
fn check_plan<A: PackArm>(
    directory: &Path,
    config: &A::Config,
    world: usize,
) -> Result<EnginePlan, String> {
    check_plan_with::<A>(directory, config, world, Availability::runtime_env())
}

/// [check_plan] under an explicit availability policy.
///
/// The gate is serving-aware: it checks the parts the arm executes ([serving]), not the whole
/// plan, because the Qwen and GLM packs do not load a vision tower (D84): DeepSeek loads and
/// runs the 14-patch tower, and those two arms do not. The catalogue's own
/// [KernelResolver::boot_check] still refuses the 16-patch tower on the plan itself, so this
/// does not weaken the catalogue's patch constraint.
fn check_plan_with<A: PackArm>(
    directory: &Path,
    config: &A::Config,
    world: usize,
    availability: Availability,
) -> Result<EnginePlan, String> {
    // Plan the architecture the arm parsed, not a second raw view of the document: the Qwen FP8
    // bridge corrects the producer's global fp8 flag, and the boot plan must see that correction.
    let plan =
        EnginePlan::for_architecture(A::architecture(config).clone(), Target::Sm121, world, 0)
            .map_err(|error| format!("cannot plan the {} checkpoint: {error}", A::NAME))?;
    let resolver = KernelResolver::new(Target::Sm121, availability);
    // Prove both passes: a plan that resolves only because a decode symbol was accepted for a
    // prefill request has not booted, it has half-booted.
    let failures = resolver
        .boot_check_serving(&plan, &serving_language(Target::Sm121))
        .err()
        .unwrap_or_default();
    if failures.is_empty() {
        eprintln!("sparkflash serve: {} plan {}", A::NAME, plan.boot_summary());
        return Ok(plan);
    }
    let reasons: Vec<String> = failures.iter().map(ToString::to_string).collect();
    Err(format!(
        "{} cannot execute {} at TP{world}: {}",
        A::NAME,
        directory.display(),
        reasons.join("; ")
    ))
}

/// A pack's single-node arm: its device backend behind the generic host driver.
fn serve_host<A: PackArm>(options: &ServeOptions) -> Result<ExitCode, String> {
    let (directory, config) = host_checkpoint::<A>(options)?;
    let context = pack_context::<A>(options, &config)?;
    let plan = if options.synthetic {
        None
    } else {
        Some(check_plan::<A>(&directory, &config, 1)?)
    };
    let profile = A::profile()?;
    sparkflash_model::install_active(profile.clone());
    eprintln!(
        "sparkflash serve: {} profile model={} family={} weights={}",
        A::NAME,
        profile.model.as_deref().unwrap_or("-"),
        profile.family.as_deref().unwrap_or("-"),
        profile.weights.as_deref().unwrap_or("-")
    );
    A::check_host(options, &config, context, &profile)?;
    let released_seam = if options.synthetic {
        None
    } else {
        Some(pack_seam::<A>(options, &directory, &config)?.1)
    };
    let mut backend =
        A::open_host(&directory, config, context, &profile, plan.as_ref()).map_err(|error| {
            format!(
                "cannot open {} at {}: {error}",
                A::NAME,
                directory.display()
            )
        })?;
    // One pooled session a concurrent request: the driver opens exactly `concurrency` sessions, so
    // a smaller pool would make a request wait for a slot it asked for at start-up.
    if options.fuse_batch {
        A::fuse(&mut backend, options.concurrency)
            .map_err(|error| format!("cannot fuse the batch: {error}"))?;
    }
    let descriptor = backend.descriptor().clone();
    let seam = released_seam
        .unwrap_or_else(|| Arc::new(sparkflash_ops::codec::ByteCodec::new(descriptor.vocab_size)));
    let banner = format!(
        "sparkflash serve: {} ({}) at TP=1 on {}, context {context}",
        descriptor.id,
        descriptor.weights_id,
        directory.display()
    );
    // The byte codec's output is arbitrary text, not the released template's channels, so the
    // fixture stays on the raw driver.
    let respond = (!options.synthetic).then_some(A::respond as Respond);
    serve_backend(
        options,
        Box::new(backend),
        seam,
        options.max_tokens.min(A::MAX_OUTPUT_TOKENS),
        1,
        respond,
        &banner,
        options.concurrency,
    )
}

/// An arm's response dialect over the raw driver ([`PackArm::respond`]).
type Respond = fn(Arc<dyn ApiInferenceEngine>) -> Arc<dyn ApiInferenceEngine>;

/// Serves `backend` behind the generic host driver ([`sparkflash_serve::HostDriver`]): one
/// session a concurrent request, `respond`'s dialect over it when given, and the numerics key of
/// a group of `world` ranks. Every arm that serves a [`ModelBackend`] serves it here, on one
/// node or as rank 0 of a group's fused lanes; `banner` announces what is served.
// One argument a piece of the boot line and its driver: keep the call sites shaped like the
// serving surface rather than packing them into a struct no caller reuses.
#[allow(clippy::too_many_arguments)]
fn serve_backend(
    options: &ServeOptions,
    backend: Box<dyn ModelBackend>,
    seam: Arc<dyn PromptCodec>,
    max_tokens: usize,
    world: usize,
    respond: Option<Respond>,
    banner: &str,
    concurrency: usize,
) -> Result<ExitCode, String> {
    let descriptor = backend.descriptor().clone();
    let driver = sparkflash_serve::HostDriver::start(
        backend,
        Arc::clone(&seam),
        sparkflash_serve::HostOptions {
            max_tokens,
            concurrency,
            ..sparkflash_serve::HostOptions::default()
        }
        .with_env()?,
    )?;
    let metrics = driver.metrics();
    println!("{banner}");
    let driver: Arc<dyn ApiInferenceEngine> = Arc::new(driver);
    let engine = match respond {
        Some(respond) => respond(driver),
        None => driver,
    };
    let numerics = ConfigKey::tp(descriptor.id.clone(), world, options.numerics, CUDA_ARCH);
    Ok(serve_with_seam(
        engine, metrics, seam, None, options, descriptor, numerics,
    ))
}

/// Wraps an arm's backend so the driver advertises a dynamically resolved identity.
///
/// The released arms hard-code their served id in their descriptor. A fine-tune in the cache is
/// the same arithmetic under another id: this wrapper keeps the backend's capabilities and
/// sessions and replaces only the identity the HTTP surface reports.
struct DescriptorBackend<B> {
    inner: B,
    descriptor: ModelDescriptor,
}

impl<B: ModelBackend> ModelBackend for DescriptorBackend<B> {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> sparkflash_model::Capabilities {
        self.inner.capabilities()
    }

    fn sizes_sessions(&self) -> bool {
        self.inner.sizes_sessions()
    }

    fn open_session(
        &self,
    ) -> Result<Box<dyn sparkflash_model::ModelSession>, sparkflash_model::BackendError> {
        self.inner.open_session()
    }

    fn open_session_for(
        &self,
        tokens: usize,
    ) -> Result<Box<dyn sparkflash_model::ModelSession>, sparkflash_model::BackendError> {
        // The identity is the only thing this wrapper replaces: a session that sizes its cache
        // for the request must be sized by the arm, not opened at the served context.
        self.inner.open_session_for(tokens)
    }
}

/// Serves a preset family's arm under a checkpoint's own identity.
///
/// This is the dynamic path for a cached fine-tune whose `model_type` is a released family: the
/// arm's arithmetic and profile are used, while the descriptor, context and tokenizer are the
/// checkpoint's. `respond` is the arm's response dialect (`A::respond`).
pub(super) fn serve_arm_identity<A: PackArm>(
    options: &ServeOptions,
    directory: PathBuf,
    descriptor: ModelDescriptor,
    seam: Arc<dyn PromptCodec>,
    context: usize,
    respond: fn(Arc<dyn ApiInferenceEngine>) -> Arc<dyn ApiInferenceEngine>,
) -> Result<ExitCode, String> {
    let config = read_config::<A>(&directory)?;
    let window = A::window(&config);
    if context == 0 || context > window {
        return Err(format!(
            "context limit {context} is outside 1..={window} for {}",
            descriptor.id
        ));
    }
    let plan = check_plan::<A>(&directory, &config, 1)?;
    let profile = A::profile()?;
    sparkflash_model::install_active(profile.clone());
    A::check_host(options, &config, context, &profile)?;
    let mut backend =
        A::open_host(&directory, config, context, &profile, Some(&plan)).map_err(|error| {
            format!(
                "cannot open {} at {}: {error}",
                A::NAME,
                directory.display()
            )
        })?;
    if options.fuse_batch {
        A::fuse(&mut backend, options.concurrency)
            .map_err(|error| format!("cannot fuse the batch: {error}"))?;
    }
    let banner = format!(
        "sparkflash serve: {} ({}) at TP=1 on {}, context {context}",
        descriptor.id,
        descriptor.weights_id,
        directory.display()
    );
    let backend = DescriptorBackend {
        inner: backend,
        descriptor,
    };
    serve_backend(
        options,
        Box::new(backend),
        seam,
        options.max_tokens.min(A::MAX_OUTPUT_TOKENS),
        1,
        Some(respond),
        &banner,
        options.concurrency,
    )
}

/// The cluster telemetry a pack's group can state: its fabric mode, not a node list or a rail.
#[cfg(feature = "tp4-serve")]
pub(super) fn pack_cluster_snapshot(model: &str, context: usize) -> ClusterStatsSnapshot {
    ClusterStatsSnapshot {
        model: model.to_owned(),
        fabric_mode: "tp4-tcp".to_owned(),
        nodes: Vec::new(),
        kv: empty_kv_snapshot(),
        engram: default_engram_layers().to_vec(),
        transport: RoceTransportStats {
            fabric_mode: "tp4-tcp".to_owned(),
            ..RoceTransportStats::default()
        },
        dspark: DSparkStepStats::default(),
        timing: RequestTimingStats::default(),
        max_seq_len: context,
    }
}

/// Runs one process of a pack's tensor-parallel group (`--tp4-world-size` 2 or 4).
///
/// Every rank holds its share of the weights ([`PackArm::OPEN_RANK`]) and steps through the same
/// collectives over the same window. Rank 0 also owns HTTP, the tokenizer, the chat template and
/// streaming; the other ranks run the control worker and never load a tokenizer. The control
/// channel and the request loop are the V4.1 group's.
#[cfg(feature = "tp4-serve")]
fn serve_group<A: PackArm>(
    options: &ServeOptions,
    tp4: &Tp4ServeOptions,
) -> Result<ExitCode, String> {
    // Every rank must resolve the same checkpoint. A `--checkpoint` names it directly; a
    // `--model` id resolves the one cached snapshot through the shared hub resolver, so
    // `--model`/the rank flags reach every worker as identically as `--checkpoint` did (D74).
    let directory = match options.checkpoint.as_deref() {
        Some(checkpoint) => checkpoint_directory(checkpoint),
        None => {
            let requested = options.model.as_deref().unwrap_or(A::MODEL_ID);
            let id = sparkflash_model::find_model(requested)
                .map_or_else(|| requested.to_owned(), |model| model.id.clone());
            sparkflash_weights::hub::cached_model_dir(&id).ok_or_else(|| {
                format!(
                    "--model {id} --tp4-rank needs --checkpoint DIR or a cached snapshot; run hf download {id}"
                )
            })?
        }
    };
    let config = read_config::<A>(&directory)?;
    // Every rank derives the window from the same configuration, so the group agrees on the
    // sequence width without rank 0 sending it.
    let context = pack_context::<A>(options, &config)?;
    check_plan::<A>(&directory, &config, tp4.world_size)?;
    // The pack's carried profile is the source of truth for this arm, exactly as it is on the
    // single-node path (serve_host installs it). Without this the four-rank arm read
    // ModelProfile::default() everywhere it asked active_profile(), so every setting the file
    // makes - the speculation depth, the prefill chunk, the plan's threads - was inert here and
    // only an environment override had any effect.
    let profile = A::profile()?;
    sparkflash_model::install_active(profile.clone());
    if tp4.rank == 0 {
        eprintln!(
            "sparkflash serve: {} profile model={} family={} weights={}",
            A::NAME,
            profile.model.as_deref().unwrap_or("-"),
            profile.family.as_deref().unwrap_or("-"),
            profile.weights.as_deref().unwrap_or("-")
        );
    }
    if options.fuse_batch {
        return serve_group_lanes::<A>(options, tp4, &directory, &config, context, A::open_lanes);
    }
    if tp4.rank != 0 {
        pack_tp4_worker(options, tp4, context, A::OPEN_RANK)
            .map_err(|message| format!("rank {}: {message}", tp4.rank))?;
        return Ok(ExitCode::SUCCESS);
    }
    // The prompt seam and the detokenizer are the same file; load it before the rank upload so a
    // missing tokenizer costs a startup message, not a load.
    let (tokenizer, seam) = pack_seam::<A>(options, &directory, &config)?;
    let descriptor = A::descriptor(&config, context)?;
    let mut rank_config = tp4_rank_config(options, tp4, TP4_CONTROL_TIMEOUT)?;
    rank_config.context = context;
    let spec = Tp4EngineSpec {
        model: A::MODEL_ID.to_owned(),
        vocab_size: descriptor.vocab_size,
        capabilities: descriptor.capabilities,
        cluster: pack_cluster_snapshot(A::MODEL_ID, context),
        tokenizer,
        open: A::OPEN_RANK,
    };
    let driver = Tp4Driver::start(
        spec,
        rank_config,
        options.max_tokens.min(A::MAX_OUTPUT_TOKENS),
    )?;
    println!(
        "sparkflash serve: {} TP={} rank 0 on {}; control {}; workers never detokenize",
        A::MODEL_ID,
        tp4.world_size,
        tp4.peers[0],
        tp4.control_port
    );
    let metrics = driver.metrics();
    let engine = A::respond(Arc::new(driver));
    let numerics = ConfigKey::tp(
        descriptor.id.clone(),
        tp4.world_size,
        options.numerics,
        CUDA_ARCH,
    );
    Ok(serve_with_seam(
        engine, metrics, seam, None, options, descriptor, numerics,
    ))
}

/// A pack's group under `--fuse-batch`: every rank opens the operator's lane count, or the
/// count its arm's [sparkflash_runtime::Limits] states when the operator gave none
/// ([`PackArm::open_lanes`]), rank 0 serves its backend through the host driver as one node
/// does, and each worker mirrors rank 0's lanes until the group ends.
#[cfg(feature = "tp4-serve")]
fn serve_group_lanes<A: PackArm>(
    options: &ServeOptions,
    tp4: &Tp4ServeOptions,
    directory: &Path,
    config: &A::Config,
    context: usize,
    open: GroupLanesOpen,
) -> Result<ExitCode, String> {
    let mut rank_config = tp4_rank_config(options, tp4, TP4_CONTROL_TIMEOUT)?;
    rank_config.context = context;
    // `--concurrency` names the lanes when the operator gave it; `None` asks the arm for the
    // width its Limits state, so the group is not stuck at one placeholder count.
    let requested = options.concurrency_set.then_some(options.concurrency);
    if tp4.rank != 0 {
        open(&rank_config, requested).map_err(|message| format!("rank {}: {message}", tp4.rank))?;
        return Ok(ExitCode::SUCCESS);
    }
    // The tokenizer before the rank upload, as the one-request group loads it.
    let (_, seam) = pack_seam::<A>(options, directory, config)?;
    let (backend, lanes) = open(&rank_config, requested)?;
    let backend = backend.ok_or_else(|| format!("{} rank 0 opened no lanes", A::NAME))?;
    let banner = format!(
        "sparkflash serve: {} TP={} rank 0 on {}; control {}; {} fused lanes; workers never \
         detokenize",
        A::MODEL_ID,
        tp4.world_size,
        tp4.peers[0],
        tp4.control_port,
        lanes
    );
    serve_backend(
        options,
        backend,
        seam,
        options.max_tokens.min(A::MAX_OUTPUT_TOKENS),
        tp4.world_size,
        Some(A::respond),
        &banner,
        lanes,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const GLM_CONFIG: &str =
        include_str!("../../../sparkflash-model-glm5-next/config/glm53_flash.hf.json");
    const QWEN_CONFIG: &str =
        include_str!("../../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.hf.json");
    const DEEPSEEK_CONFIG: &str = include_str!(
        "../../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json"
    );
    #[test]
    fn the_arms_table_carries_the_same_profile_as_the_pack() {
        for (family, toml) in [
            ("deepseek_v41", DEEPSEEK_PROFILE_TOML),
            ("qwen4_exp", QwenArm::PROFILE_TOML),
            ("qwen4_exp_text", QwenArm::PROFILE_TOML),
            ("qwen3_5_moe", Qwen36Arm::PROFILE_TOML),
            ("glm5_next", GlmArm::PROFILE_TOML),
            ("dory", DORY_PROFILE_TOML),
            ("inkling2", sparkflash_model_inkling2::PROFILE_TOML),
        ] {
            assert_eq!(
                profile_toml_for_family(family),
                Some(toml),
                "{family} moved out of ARMS"
            );
        }
        assert_eq!(profile_toml_for_family("a_dynamic_family"), None);
    }

    #[test]
    fn every_carried_profile_parses_and_keeps_its_registry_identity() {
        for (name, toml) in [
            ("DeepSeek-V4.1-Flash", DEEPSEEK_PROFILE_TOML),
            ("Qwen3.8-Flash-Next", QwenArm::PROFILE_TOML),
            ("GLM-5.3-Flash", GlmArm::PROFILE_TOML),
            ("Dory1.0-4B", DORY_PROFILE_TOML),
        ] {
            let profile = ModelProfile::from_toml(toml)
                .unwrap_or_else(|error| panic!("{name} profile: {error}"));
            let weights = profile
                .weights
                .as_deref()
                .unwrap_or_else(|| panic!("{name} names no weights"));
            let descriptor = sparkflash_model::find_model_by_weights(weights)
                .unwrap_or_else(|| panic!("{name} weights {weights} are not in the registry"));
            assert_eq!(profile.model.as_deref(), Some(descriptor.id.as_str()));
            // The two spellings a family is named in differ by punctuation alone: the descriptor
            // carries the release's dotted name and a profile may carry the architecture's, so
            // the comparison folds punctuation out rather than forcing one spelling into an arm
            // profile - an edit there is a served setting the numerics gate judges.
            let fold = |name: &str| {
                name.chars()
                    .filter(char::is_ascii_alphanumeric)
                    .map(|c| c.to_ascii_lowercase())
                    .collect::<String>()
            };
            assert_eq!(
                profile.family.as_deref().map(fold),
                Some(fold(descriptor.family.as_str())),
                "{name} names a different family than its weights"
            );
        }
    }

    #[test]
    fn a_profile_environment_override_wins_over_the_carried_toml() {
        let carried = ModelProfile::from_toml(QwenArm::PROFILE_TOML).expect("the carried profile");
        assert_eq!(carried.prefill.chunk, Some(4096));
        // The block drafter's whole width: seven drafts, with the adaptive policy choosing how
        // many of them a step verifies (the measured table is beside the setting).
        assert_eq!(carried.speculation.depth, Some(7));
        let overridden = QwenArm::profile_from(|name| match name {
            "SPARKFLASH_QWEN4_PREFILL_CHUNK" => Some("2048".to_owned()),
            "SPARKFLASH_DRAFT_DEPTH" => Some("5".to_owned()),
            _ => None,
        })
        .expect("an environment override is valid");
        assert_eq!(overridden.prefill.chunk, Some(2048));
        assert_eq!(overridden.speculation.depth, Some(5));
    }

    #[test]
    fn glm_accepts_tensor_core_decode_where_its_profile_is_read() {
        // The token row is the tile's few-row body, so the switch names what the arm runs.
        let profile = GlmArm::profile_from(|name| match name {
            "SPARKFLASH_DECODE_MMA" => Some("on".to_owned()),
            _ => None,
        })
        .expect("GLM runs the tile's few-row body");
        assert_eq!(profile.decode.tensor_cores, Some(true));
        // The carried profile leaves the field unset, which is the on default.
        let carried = GlmArm::profile_from(|_| None).expect("the carried GLM profile boots");
        assert_eq!(carried.decode.tensor_cores, None);
    }

    #[test]
    fn every_released_config_is_detected_by_its_own_family() {
        for (config, expected) in [
            (
                DEEPSEEK_CONFIG,
                sparkflash_model::descriptor::DEFAULT_MODEL_ID,
            ),
            (
                QWEN_CONFIG,
                sparkflash_model::descriptor::QWEN38_FLASH_NEXT_ID,
            ),
            (GLM_CONFIG, sparkflash_model::descriptor::GLM53_FLASH_ID),
        ] {
            let directory = tempfile::tempdir().expect("a temporary checkpoint");
            std::fs::write(directory.path().join(CHECKPOINT_CONFIG_NAME), config)
                .expect("the carried config is written");
            assert_eq!(
                Pack::detect(directory.path()).map(Pack::model_id),
                Some(expected)
            );
        }
    }

    #[test]
    fn a_flat_text_only_qwen_checkpoint_detects_as_the_qwen_pack() {
        // The released text-only Qwen writes the language configuration at the document root
        // and names itself `qwen4_exp_text`. Detection must fold that alias to the Qwen arm,
        // otherwise ServeOptions::parse gates it against the host part set and refuses it
        // before the Qwen dispatch can run.
        let nested: serde_json::Value = serde_json::from_str(QWEN_CONFIG).expect("the Qwen config");
        let mut flat = nested["text_config"].clone();
        flat["model_type"] = serde_json::json!("qwen4_exp_text");
        flat["architectures"] = serde_json::json!(["Qwen4ExpTextForCausalLM"]);
        flat["tie_word_embeddings"] = nested["tie_word_embeddings"].clone();
        flat["image_token_id"] = nested["image_token_id"].clone();
        flat["vision_start_token_id"] = nested["vision_start_token_id"].clone();
        flat["vision_end_token_id"] = nested["vision_end_token_id"].clone();
        let directory = tempfile::tempdir().expect("a temporary checkpoint");
        std::fs::write(
            directory.path().join(CHECKPOINT_CONFIG_NAME),
            flat.to_string(),
        )
        .expect("the text-only config is written");
        assert_eq!(
            Pack::detect(directory.path()).map(Pack::model_id),
            Some(sparkflash_model::descriptor::QWEN38_FLASH_NEXT_ID)
        );
        assert_eq!(
            Pack::named("qwen4_exp_text").map(Pack::model_id),
            Some(sparkflash_model::descriptor::QWEN38_FLASH_NEXT_ID)
        );
    }

    #[test]
    fn a_dory_checkpoint_detects_as_the_dynamic_host_pack() {
        const CONFIG: &str = include_str!(
            "../../../sparkflash-model-dory/config/nvidia_nemotron_labs_dory1_0_4b_bf16.hf.json"
        );
        let directory = tempfile::tempdir().expect("a temporary checkpoint");
        std::fs::write(directory.path().join(CHECKPOINT_CONFIG_NAME), CONFIG)
            .expect("the Dory config is written");
        assert_eq!(
            Pack::detect(directory.path()).map(Pack::model_id),
            Some(sparkflash_model::descriptor::DORY1_0_4B_BF16_ID)
        );
        assert!(
            !Pack::named(sparkflash_model::descriptor::DORY1_0_4B_BF16_ID)
                .expect("Dory is a pack")
                .has_pack_arm()
        );
        assert!(Pack::missing_parts(directory.path()).is_empty());
    }

    #[test]
    fn an_inkling_checkpoint_detects_as_the_inkling_pack() {
        let directory = tempfile::tempdir().expect("a temporary checkpoint");
        std::fs::write(
            directory.path().join(CHECKPOINT_CONFIG_NAME),
            sparkflash_model_inkling2::carried_config(),
        )
        .expect("the released config is written");
        std::fs::write(
            directory.path().join("hf_quant_config.json"),
            sparkflash_model_inkling2::carried_quant(),
        )
        .expect("the quant groups are written");
        assert_eq!(
            Pack::detect(directory.path()).map(Pack::model_id),
            Some(sparkflash_model::descriptor::INKLING2_SMALL_ID)
        );
        assert!(
            !Pack::named(sparkflash_model::descriptor::INKLING2_SMALL_ID)
                .expect("Inkling is a pack")
                .has_pack_arm()
        );
        assert!(Pack::missing_parts(directory.path()).is_empty());
    }

    #[test]
    fn every_registry_id_names_its_pack_or_the_dynamic_host() {
        for model in sparkflash_model::MODELS.iter() {
            for id in [model.id.as_str(), model.weights_id.as_str()] {
                let pack =
                    Pack::named(id).unwrap_or_else(|| panic!("the registry id {id} names no pack"));
                assert_eq!(pack.model_id(), model.id);
                assert_eq!(pack.descriptor().weights_id, model.weights_id);
            }
        }
        // Qwen3.6-35B-A3B FP8 is the Qwen arm's device backend over the Qwen3.5-MoE family.
        assert_eq!(
            Pack::named(sparkflash_model::descriptor::QWEN36_FLASH_35B_A3B_FP8_ID)
                .map(Pack::model_id),
            Some(sparkflash_model::descriptor::QWEN36_FLASH_35B_A3B_FP8_ID)
        );
        assert_eq!(Pack::named("example/no-arm"), None);
    }

    #[test]
    fn every_carried_arm_opens_a_group_at_world_two() {
        const QWEN36_FP8: &str =
            include_str!("../../../sparkflash-model/config/qwen36_flash_35b_a3b_fp8.hf.json");
        const DORY: &str = include_str!(
            "../../../sparkflash-model-dory/config/nvidia_nemotron_labs_dory1_0_4b_bf16.hf.json"
        );
        for arm in sparkflash_model::ARMS
            .iter()
            .filter(|arm| arm.profile_toml().is_some())
        {
            let pack =
                Pack::named(arm.model_id()).unwrap_or_else(|| panic!("{} names no pack", arm.id()));
            assert_eq!(pack.model_id(), arm.model_id());
            let architecture = match arm.id() {
                "DeepSeek" => sparkflash_model::Architecture::from_json(DEEPSEEK_CONFIG),
                "Qwen" | "Qwen (FP8)" => sparkflash_model::Architecture::from_json(QWEN_CONFIG),
                "Qwen3.6 (FP8)" => sparkflash_model::Architecture::from_json(QWEN36_FP8),
                "GLM" => sparkflash_model::Architecture::from_json(GLM_CONFIG),
                "Dory" => sparkflash_model::Architecture::from_json(DORY),
                "Inkling" => sparkflash_model::Architecture::from_json(
                    sparkflash_model_inkling2::carried_config(),
                ),
                other => panic!("{other} has a carried profile but no config in this test"),
            }
            .unwrap_or_else(|error| panic!("{}: {error}", arm.id()));
            // World 2 is a group the plan can lay out for every carried arm. The device open is
            // the arm's own; this is the construction the group's lanes stand on.
            let plan = EnginePlan::for_architecture(architecture, Target::Sm121, 2, 0)
                .unwrap_or_else(|error| panic!("{} at TP2: {error}", arm.id()));
            assert!(
                plan.boot_summary().contains("TP0/2"),
                "{}: {}",
                arm.id(),
                plan.boot_summary()
            );
            match pack.route() {
                Route::Typed => {
                    assert!(pack.has_pack_arm(), "{}", arm.id());
                    assert!(pack.opens_group_lanes(), "{}", arm.id());
                    assert!(
                        typed_pack(&pack.descriptor.weights_id).is_some(),
                        "{} has no typed opener",
                        arm.id()
                    );
                }
                Route::Reference => assert!(pack.is_reference(), "{}", arm.id()),
                Route::Dynamic => assert!(!pack.has_pack_arm(), "{}", arm.id()),
            }
        }
    }

    /// The group's inner path on a device: each pack arm opens its tiny checkpoint and its walk
    /// states the lane width a group with no `--concurrency` would step.
    #[test]
    #[ignore = "needs a CUDA device: opens each pack arm's tiny checkpoint"]
    fn the_qwen_pack_arms_group_lane_width_boots_its_runtime() {
        // The Qwen3.8 canonical drafter is looked up in the hub cache when the profile names
        // none; an empty cache keeps a released drafter away from the tiny fixture, which is a
        // different width.
        let cache = tempfile::tempdir().expect("an empty hub cache");
        unsafe { std::env::set_var("HF_HUB_CACHE", cache.path()) };
        let directory = tempfile::tempdir().expect("a temporary checkpoint");
        let config = <QwenArm as PackArm>::write_fixture(directory.path()).expect("the fixture");
        let context = <QwenArm as PackArm>::window(&config).min(4_096);
        let profile = <QwenArm as PackArm>::profile_from(|_| None).expect("the profile");
        let backend =
            <QwenArm as PackArm>::open_host(directory.path(), config, context, &profile, None)
                .expect("the Qwen fixture opens");
        let qwen = backend.lane_limit().expect("the Qwen lane width");
        assert!(qwen > 0, "Qwen states no lane width");

        let directory = tempfile::tempdir().expect("a temporary checkpoint");
        let config = <QwenFp8Arm as PackArm>::write_fixture(directory.path()).expect("the fixture");
        let context = <QwenFp8Arm as PackArm>::window(&config).min(4_096);
        let profile = <QwenFp8Arm as PackArm>::profile_from(|_| None).expect("the profile");
        let backend =
            <QwenFp8Arm as PackArm>::open_host(directory.path(), config, context, &profile, None)
                .expect("the FP8 fixture opens");
        assert_eq!(
            backend.lane_limit().expect("the FP8 lane width"),
            qwen,
            "the FP8 release is the same arm"
        );

        let directory = tempfile::tempdir().expect("a temporary checkpoint");
        let config = <Qwen36Arm as PackArm>::write_fixture(directory.path()).expect("the fixture");
        let context = <Qwen36Arm as PackArm>::window(&config).min(4_096);
        let profile = <Qwen36Arm as PackArm>::profile_from(|_| None).expect("the profile");
        let backend =
            <Qwen36Arm as PackArm>::open_host(directory.path(), config, context, &profile, None)
                .expect("the Qwen3.6 fixture opens");
        assert!(
            backend.lane_limit().expect("the Qwen3.6 lane width") > 0,
            "Qwen3.6 states no lane width"
        );

        // GLM's tiny fixture is not device-servable (its 32-wide hidden size is refused by the
        // mixer's 256-column threads); its group width is the world-2 plan above, and the
        // runtime's Limits are what `open_lanes` reads on the released checkpoint.
    }

    #[test]
    fn the_glm_host_admits_the_latent_the_single_rank_allocates() {
        // The fixture's checkpoint-wide FP8 tiling says nothing about `kv_b_proj.weight`, and
        // the single-node rank allocates the f32 latent. One whole FP8 tile of latent keeps
        // the E4M3 plan the old admission chose measurably smaller; the fixture's own 16-wide
        // latent is below a tile and falls back to f32 in both plans.
        let mut config = Glm5NextConfig::tiny();
        config.kv_lora_rank = 128;
        for layer in &mut config.architecture.layers {
            if let sparkflash_model::architecture::AttentionKind::Mla { kv_lora_rank, .. } =
                &mut layer.attention.kind
            {
                *kv_lora_rank = 128;
            }
        }
        assert_eq!(config.fp8_block, Some(8));
        let context = 10_000;
        let budget = sparkflash_model::kv_budget_bytes(1);
        let f32_plan = sparkflash_model_glm5_next::kv_plan_with_latent(&config, false);
        let fp8_plan = sparkflash_model_glm5_next::kv_plan_with_latent(&config, true);
        assert!(
            f32_plan.concurrency_bytes(context, 64) > budget,
            "the allocated f32 latent does not fit 1 GiB at 64 sessions"
        );
        assert!(
            fp8_plan.concurrency_bytes(context, 64) <= budget,
            "the old fp8_block-derived plan fit 1 GiB, the bug"
        );
        let options = |concurrency: &str| {
            ServeOptions::parse(vec![
                "--model".to_owned(),
                GlmArm::MODEL_ID.to_owned(),
                "--synthetic".to_owned(),
                "--context".to_owned(),
                context.to_string(),
                "--concurrency".to_owned(),
                concurrency.to_owned(),
                "--kv-budget-gib".to_owned(),
                "1".to_owned(),
            ])
            .expect("the GLM host options parse")
        };
        let profile = GlmArm::profile_from(|_| None).expect("the carried profile parses");
        let over = options("64");
        let error = GlmArm::check_host(&over, &config, context, &profile)
            .expect_err("the f32 latent the rank allocates is over 1 GiB at 64 sessions");
        assert!(error.contains("--concurrency"), "{error}");
        // One fewer worker fits the same window and budget.
        GlmArm::check_host(&options("63"), &config, context, &profile)
            .expect("63 sessions fit the budget");
        // The profile's choice belongs to the TP=4 arm: even an explicit Fp8 request is
        // admitted against the single-node f32 latent and is still refused here.
        let mut fp8_profile = profile.clone();
        fp8_profile.attention.latent_cache = Some(sparkflash_model::profile::LatentCacheMode::Fp8);
        let error = GlmArm::check_host(&over, &config, context, &fp8_profile)
            .expect_err("the single-node arm ignores the profile's latent cache");
        assert!(error.contains("--concurrency"), "{error}");
    }

    #[test]
    fn missing_parts_needs_an_image_this_build_can_load() {
        let directory = tempfile::tempdir().expect("a temporary checkpoint");
        std::fs::write(
            directory.path().join(CHECKPOINT_CONFIG_NAME),
            DEEPSEEK_CONFIG,
        )
        .expect("the carried config is written");
        let missing = Pack::missing_parts(directory.path());
        // The DeepSeek image is loaded from SPARKFLASH_CUBINS at run time. Unless that directory
        // is configured, a part only DeepSeek implements cannot be loaded, and the catalogue's
        // answer (which accepts any catalogued image) must not hide it.
        let discovered = sparkflash_ops::parts::missing(
            &sparkflash_model::Architecture::from_json(DEEPSEEK_CONFIG).expect("the config"),
            sparkflash_model::Target::Sm121,
        );
        assert!(discovered.is_empty(), "the catalogue carries the parts");
        if std::env::var_os(sparkflash_ops::parts::CUBIN_DIR_ENV).is_none() {
            assert!(
                missing.contains(&sparkflash_model::Part::Csa2Compression),
                "a run-time-only image is not loadable without its cubin directory: {missing:?}"
            );
        }
    }

    /// The CLI's real Qwen FP8 boot path: the serving constructor parses the released FP8
    /// document, the plan is built from its corrected architecture, and the serving boot check
    /// resolves every language part at TP1 and TP4. The catalogue's whole-plan check still
    /// refuses the 16-patch tower, so this does not weaken the patch rule.
    #[test]
    fn the_released_qwen_fp8_pack_boots_at_tp1_and_tp4() {
        const QWEN_FP8: &str = include_str!(
            "../../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_fp8.hf.json"
        );
        let directory = tempfile::tempdir().expect("a temporary checkpoint");
        std::fs::write(directory.path().join(CHECKPOINT_CONFIG_NAME), QWEN_FP8)
            .expect("the carried config is written");
        std::fs::write(
            directory.path().join("generation_config.json"),
            r#"{"eos_token_id": [248044]}"#,
        )
        .expect("the carried generation config is written");
        let config =
            QwenFp8Arm::read_config(directory.path()).expect("the FP8 serving configuration");
        for world in [1, 4] {
            let plan = check_plan_with::<QwenFp8Arm>(
                directory.path(),
                &config,
                world,
                Availability::Catalogue,
            )
            .unwrap_or_else(|error| panic!("Qwen FP8 TP{world}: {error}"));
            assert!(
                plan.requirements()
                    .any(|requirement| requirement.part == sparkflash_model::Part::Vision),
                "the released document still plans the tower"
            );
            assert_eq!(
                plan.precision().experts,
                sparkflash_ops::plan::Precision::Fp8E4M3,
                "TP{world}"
            );
            if world == 4 {
                let moe = plan
                    .requirements()
                    .find(|requirement| requirement.part == sparkflash_model::Part::Moe)
                    .expect("a Qwen FP8 MoE requirement");
                assert_eq!(moe.geometry.intermediate, 256, "rank 0's whole-tile share");
            }
            // The catalogue's own boot check (the whole plan) still refuses the tower.
            let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
            let errors = resolver
                .boot_check(&plan)
                .expect_err("the 16-patch tower is still refused");
            assert!(
                errors
                    .iter()
                    .all(|error| error.part == sparkflash_model::Part::Vision),
                "TP{world}: language parts must boot: {errors:?}"
            );
            assert!(
                errors
                    .iter()
                    .any(|error| error.to_string().contains("vision patch 16")),
                "TP{world}: {errors:?}"
            );
        }
    }

    /// The CLI's real Qwen3.6 boot path: the released FP8 document parses as the Qwen pack's
    /// configuration, and the serving boot check resolves every language part at TP1, the routed
    /// experts on the Qwen images at the second published routed shape.
    #[test]
    fn the_released_qwen36_fp8_pack_boots_at_tp1() {
        const QWEN36_FP8: &str =
            include_str!("../../../sparkflash-model/config/qwen36_flash_35b_a3b_fp8.hf.json");
        let directory = tempfile::tempdir().expect("a temporary checkpoint");
        std::fs::write(directory.path().join(CHECKPOINT_CONFIG_NAME), QWEN36_FP8)
            .expect("the carried config is written");
        std::fs::write(
            directory.path().join("generation_config.json"),
            r#"{"eos_token_id": [248046, 248044]}"#,
        )
        .expect("the carried generation config is written");
        let config =
            Qwen36Arm::read_config(directory.path()).expect("the Qwen3.6 serving configuration");
        sparkflash_model_qwen4_exp::device::identity_for(&config)
            .expect("the FP8 release has a device identity");
        let plan =
            check_plan_with::<Qwen36Arm>(directory.path(), &config, 1, Availability::Catalogue)
                .unwrap_or_else(|error| panic!("Qwen3.6 FP8 TP1: {error}"));
        assert_eq!(
            plan.precision().experts,
            sparkflash_ops::plan::Precision::Fp8E4M3
        );
        let moe = plan
            .requirements()
            .find(|requirement| requirement.part == sparkflash_model::Part::Moe)
            .expect("a Qwen3.6 MoE requirement");
        assert_eq!(moe.geometry.intermediate, 512, "TP1 holds the whole expert");
    }

    /// The plan-aware single-rank open is the arm's EngineBuilder, not the old direct
    /// open_with_plan/open_single_rank_with_plan call. A TP2 plan is refused by the builder's
    /// own world gate before either arm opens a device, so the test needs no GPU and fails if the
    /// direct path is restored (which would accept the plan and try the device).
    #[test]
    fn the_single_rank_serve_open_uses_the_engine_builder() {
        let qwen_directory = tempfile::tempdir().expect("a temporary Qwen checkpoint");
        let qwen_config = <QwenArm as PackArm>::write_fixture(qwen_directory.path())
            .expect("the Qwen fixture writes");
        let qwen_plan =
            EnginePlan::for_architecture(qwen_config.architecture.clone(), Target::Sm121, 2, 0)
                .expect("the tiny Qwen architecture plans at TP2");
        let profile = <QwenArm as PackArm>::profile_from(|_| None).expect("the Qwen profile");
        let error = <QwenArm as PackArm>::open_host(
            qwen_directory.path(),
            qwen_config,
            qwen_plan.architecture().max_position_embeddings,
            &profile,
            Some(&qwen_plan),
        )
        .expect_err("the builder refuses a TP2 plan");
        assert!(
            error.contains("single-rank builder opens TP1/rank0") && error.contains("TP2/rank0"),
            "the Qwen open did not reach the builder: {error}"
        );

        let glm_directory = tempfile::tempdir().expect("a temporary GLM checkpoint");
        let glm_config = <GlmArm as PackArm>::write_fixture(glm_directory.path())
            .expect("the GLM fixture writes");
        let glm_plan =
            EnginePlan::for_architecture(glm_config.architecture.clone(), Target::Sm121, 2, 0)
                .expect("the tiny GLM architecture plans at TP2");
        let profile = <GlmArm as PackArm>::profile_from(|_| None).expect("the GLM profile");
        let error = <GlmArm as PackArm>::open_host(
            glm_directory.path(),
            glm_config,
            glm_plan.architecture().max_position_embeddings,
            &profile,
            Some(&glm_plan),
        )
        .expect_err("the builder refuses a TP2 plan");
        assert!(
            error.contains("single-rank builder opens TP1/rank0") && error.contains("TP2/rank0"),
            "the GLM open did not reach the builder: {error}"
        );
    }

    /// The plan-aware Qwen open is the serve path's `--checkpoint` arm, not `--synthetic`: it
    /// receives the admitted `--context` and must open (and advertise) exactly that window. The
    /// builder used to reopen from `architecture.max_position_embeddings`, so a 32-token request
    /// allocated and advertised the fixture's whole 64-token window.
    #[test]
    fn the_plan_aware_qwen_open_keeps_the_requested_context() {
        if sparkflash_kernels::cuda_driver::Cuda::load().is_err()
            || !sparkflash_model_qwen4_exp::device::cubin_available()
        {
            eprintln!("skipped: no Qwen CUDA image");
            return;
        }
        let directory = tempfile::tempdir().expect("a temporary Qwen checkpoint");
        let config =
            <QwenArm as PackArm>::write_fixture(directory.path()).expect("the Qwen fixture writes");
        let window = <QwenArm as PackArm>::window(&config);
        let context = window / 2;
        assert!(context < window);
        let options = ServeOptions::parse(vec![
            "--model".to_owned(),
            QwenArm::MODEL_ID.to_owned(),
            "--checkpoint".to_owned(),
            directory.path().to_string_lossy().into_owned(),
            "--context".to_owned(),
            context.to_string(),
        ])
        .expect("the Qwen serve options parse");
        let admitted = pack_context::<QwenArm>(&options, &config).expect("the context is admitted");
        assert_eq!(admitted, context);
        let plan = EnginePlan::for_architecture(config.architecture.clone(), Target::Sm121, 1, 0)
            .expect("the tiny Qwen architecture plans");
        let profile = <QwenArm as PackArm>::profile_from(|_| None).expect("the Qwen profile");
        let backend = <QwenArm as PackArm>::open_host(
            directory.path(),
            config,
            admitted,
            &profile,
            Some(&plan),
        )
        .expect("the plan-aware Qwen open serves the fixture");
        assert_eq!(backend.descriptor().context_tokens, context);
        assert_eq!(backend.context_limit(), context);
    }

    /// The same regression on the GLM arm: its plan-aware open must carry the admitted window
    /// into `open_single_rank_with_plan`, not reopen at the architecture's full window.
    #[test]
    fn the_plan_aware_glm_open_keeps_the_requested_context() {
        if sparkflash_kernels::cuda_driver::Cuda::load().is_err()
            || !sparkflash_model_glm5_next::cubin_available()
        {
            eprintln!("skipped: no GLM CUDA image");
            return;
        }
        let directory = tempfile::tempdir().expect("a temporary GLM checkpoint");
        let config =
            <GlmArm as PackArm>::write_fixture(directory.path()).expect("the GLM fixture writes");
        let window = <GlmArm as PackArm>::window(&config);
        let context = window / 2;
        assert!(context < window);
        let options = ServeOptions::parse(vec![
            "--model".to_owned(),
            GlmArm::MODEL_ID.to_owned(),
            "--checkpoint".to_owned(),
            directory.path().to_string_lossy().into_owned(),
            "--context".to_owned(),
            context.to_string(),
        ])
        .expect("the GLM serve options parse");
        let admitted = pack_context::<GlmArm>(&options, &config).expect("the context is admitted");
        assert_eq!(admitted, context);
        let plan = EnginePlan::for_architecture(config.architecture.clone(), Target::Sm121, 1, 0)
            .expect("the tiny GLM architecture plans");
        let profile = <GlmArm as PackArm>::profile_from(|_| None).expect("the GLM profile");
        let backend = <GlmArm as PackArm>::open_host(
            directory.path(),
            config.clone(),
            admitted,
            &profile,
            Some(&plan),
        )
        .expect("the plan-aware GLM open serves the fixture");
        assert_eq!(backend.descriptor().context_tokens, context);
        assert_eq!(backend.context_limit(), context);
    }
}

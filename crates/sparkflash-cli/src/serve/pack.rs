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
use sparkflash_ops::engine_plan::EnginePlan;
use sparkflash_ops::resolver::{Availability, KernelResolver};
use sparkflash_weights::rank_file::{find_converted_rank_file, rank_file_name};

use super::{CHECKPOINT_CONFIG_NAME, EXIT_REFUSED, ServeOptions, serve_with_seam};
#[cfg(feature = "tp4-serve")]
use super::{
    ClusterStatsSnapshot, DSparkStepStats, RequestTimingStats, RoceTransportStats,
    TP4_CONTROL_TIMEOUT, Tp4Driver, Tp4EngineSpec, Tp4RankApi, Tp4RankConfig, Tp4ServeOptions,
    default_engram_layers, empty_kv_snapshot, open_glm_tp4_rank, open_qwen_tp4_rank,
    pack_tp4_worker, tp4_rank_config,
};

/// The variable naming a directory of Qwen3.8-Flash-Next rank files (`sparkflash convert`).
pub(super) const QWEN_CONVERTED_DIR_ENV: &str = "SPARKFLASH_QWEN_CONVERTED_DIR";

/// The default DeepSeek arm's carried profile. The arm has no [PackArm] (the reference path
/// serves it), but its boot still reads the same typed profile as the packed arms.
pub(super) const DEEPSEEK_PROFILE_TOML: &str =
    include_str!("../../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.toml");

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

    /// Writes the tiny fixture `--synthetic` serves into `directory`.
    fn write_fixture(directory: &Path) -> Result<Self::Config, String>;
    /// The configuration of the checkpoint in `directory`.
    fn read_config(directory: &Path) -> Result<Self::Config, String>;
    /// The checkpoint's window, `max_position_embeddings`.
    fn window(config: &Self::Config) -> usize;
    /// What a group serves over `context` tokens.
    fn descriptor(config: &Self::Config, context: usize) -> ModelDescriptor;
    /// The released prompt seam from a tokenizer file.
    fn codec(tokenizer: &Path) -> Result<Arc<dyn PromptCodec>, String>;
    /// Refusals the single-node arm makes before it uploads anything.
    fn check_host(
        options: &ServeOptions,
        config: &Self::Config,
        context: usize,
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
    const NAME: &'static str = "Qwen3.8-Flash-Next";
    const FIXTURE: &'static str = "qwen4";
    const MAX_OUTPUT_TOKENS: usize = sparkflash_model::descriptor::QWEN38_MAX_OUTPUT_TOKENS;
    const DEFAULT_CONTEXT: usize = sparkflash_model::descriptor::QWEN38_CONTEXT_TOKENS;
    const PROFILE_TOML: &'static str =
        include_str!("../../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.toml");
    const FUSED_BATCH: bool = true;
    const KV_BUDGET: bool = true;
    #[cfg(feature = "tp4-serve")]
    const OPEN_RANK: fn(&Tp4RankConfig) -> Result<Box<dyn Tp4RankApi>, String> = open_qwen_tp4_rank;

    fn write_fixture(directory: &Path) -> Result<Self::Config, String> {
        let config = Qwen4ExpConfig::tiny();
        sparkflash_model_qwen4_exp::write_synthetic_checkpoint(directory, &config, 7)?;
        Ok(config)
    }

    fn read_config(directory: &Path) -> Result<Self::Config, String> {
        let path = directory.join(CHECKPOINT_CONFIG_NAME);
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        Qwen4ExpConfig::from_json(&text).map_err(|error| format!("{}: {error}", path.display()))
    }

    fn window(config: &Self::Config) -> usize {
        config.max_position_embeddings
    }

    fn descriptor(config: &Self::Config, context: usize) -> ModelDescriptor {
        sparkflash_model_qwen4_exp::device::descriptor_for(config, context)
    }

    fn codec(tokenizer: &Path) -> Result<Arc<dyn PromptCodec>, String> {
        Ok(Arc::new(sparkflash_model_qwen4_exp::QwenCodec::from_path(
            tokenizer,
        )?))
    }

    fn check_host(
        options: &ServeOptions,
        _config: &Self::Config,
        context: usize,
    ) -> Result<(), String> {
        admit(
            sparkflash_model_qwen4_exp::kv_plan(),
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
            check_backend_plan(plan, "qwen4_exp", config.num_hidden_layers)?;
        }
        // A converted rank file is one slab read instead of 77.8 GiB of per-tensor uploads. The
        // checkpoint directory stays the configuration, tokenizer and host-paged PLE source.
        let converted = converted_rank_file(QWEN_CONVERTED_DIR_ENV, 0, 1)?;
        let backend = match (&converted, plan) {
            (Some(path), Some(plan)) => {
                Qwen4ExpDeviceBackend::open_converted_with_plan(directory, context, path, plan)
            }
            (Some(path), None) => Qwen4ExpDeviceBackend::open_converted(directory, context, path),
            (None, Some(plan)) => Qwen4ExpDeviceBackend::open_with_plan(directory, context, plan),
            (None, None) => Qwen4ExpDeviceBackend::open(directory, context),
        }
        .map_err(|error| error.to_string())?;
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

    fn write_fixture(directory: &Path) -> Result<Self::Config, String> {
        <QwenArm as PackArm>::write_fixture(directory)
    }

    fn read_config(directory: &Path) -> Result<Self::Config, String> {
        <QwenArm as PackArm>::read_config(directory)
    }

    fn window(config: &Self::Config) -> usize {
        <QwenArm as PackArm>::window(config)
    }

    fn descriptor(config: &Self::Config, context: usize) -> ModelDescriptor {
        <QwenArm as PackArm>::descriptor(config, context)
    }

    fn codec(tokenizer: &Path) -> Result<Arc<dyn PromptCodec>, String> {
        <QwenArm as PackArm>::codec(tokenizer)
    }

    fn check_host(
        options: &ServeOptions,
        config: &Self::Config,
        context: usize,
    ) -> Result<(), String> {
        <QwenArm as PackArm>::check_host(options, config, context)
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
    const NAME: &'static str = "GLM-5.3-Flash";
    const FIXTURE: &'static str = "glm53";
    const MAX_OUTPUT_TOKENS: usize = sparkflash_model::descriptor::GLM53_MAX_OUTPUT_TOKENS;
    const DEFAULT_CONTEXT: usize = sparkflash_model::descriptor::GLM53_CONTEXT_TOKENS;
    const PROFILE_TOML: &'static str =
        include_str!("../../../sparkflash-model-glm5-next/config/glm53_flash.toml");
    const FUSED_BATCH: bool = true;
    const KV_BUDGET: bool = true;
    #[cfg(feature = "tp4-serve")]
    const OPEN_RANK: fn(&Tp4RankConfig) -> Result<Box<dyn Tp4RankApi>, String> = open_glm_tp4_rank;

    fn write_fixture(directory: &Path) -> Result<Self::Config, String> {
        let config = Glm5NextConfig::tiny();
        sparkflash_model_glm5_next::checkpoint::write_synthetic_checkpoint(
            directory, &config, 7, 2,
        )?;
        Ok(config)
    }

    /// The checkpoint's `config.json`, or the carried release configuration when the directory
    /// has none.
    fn read_config(directory: &Path) -> Result<Self::Config, String> {
        let path = directory.join(CHECKPOINT_CONFIG_NAME);
        if !path.is_file() {
            return Glm5NextConfig::glm53_flash()
                .map_err(|error| format!("the carried release configuration: {error}"));
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        Glm5NextConfig::from_json(&text).map_err(|error| format!("{}: {error}", path.display()))
    }

    fn window(config: &Self::Config) -> usize {
        config.max_position_embeddings
    }

    fn descriptor(config: &Self::Config, context: usize) -> ModelDescriptor {
        sparkflash_model_glm5_next::serve::descriptor_for(config, context)
    }

    fn codec(tokenizer: &Path) -> Result<Arc<dyn PromptCodec>, String> {
        Ok(Arc::new(sparkflash_model_glm5_next::GlmCodec::from_path(
            tokenizer,
        )?))
    }

    fn check_host(
        options: &ServeOptions,
        config: &Self::Config,
        context: usize,
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
        admit(
            sparkflash_model_glm5_next::kv_plan(config),
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
        let cuda = sparkflash_kernels::cuda_driver::Cuda::load()
            .map_err(|error| format!("CUDA: {error}"))?;
        let cuda = Arc::new(sparkflash_model_glm5_next::device::CudaHandle::new(cuda));
        let mut backend = match plan {
            Some(plan) => {
                sparkflash_model_glm5_next::Glm5NextDeviceBackend::open_single_rank_with_plan(
                    cuda, config, directory, context, plan,
                )
            }
            None => sparkflash_model_glm5_next::Glm5NextDeviceBackend::open_single_rank(
                cuda, config, directory, context,
            ),
        }
        .map_err(|error| error.to_string())?;
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

/// The packs the model registry knows, including the default DeepSeek arm.
///
/// The variants are the registry's families; every id, weights selector, model type, label,
/// context and vocabulary is read from [sparkflash_model::MODELS], never from a second table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Pack {
    /// The default DeepSeek-V4.1-Flash arm, which the reference/TP4 path serves.
    Deepseek,
    Qwen,
    /// The official FP8 release of the same Qwen arm: one family, two weight selectors.
    QwenFp8,
    Glm,
}

impl Pack {
    /// This pack's descriptor in [sparkflash_model::MODELS].
    ///
    /// # Panics
    /// Only if the registry loses the family this binary was built to serve.
    fn descriptor(self) -> &'static ModelDescriptor {
        let weights = match self {
            Self::Deepseek => sparkflash_model::descriptor::DEFAULT_WEIGHTS_ID,
            Self::Qwen => sparkflash_model::descriptor::QWEN38_FLASH_NEXT_WEIGHTS_ID,
            Self::QwenFp8 => sparkflash_model::descriptor::QWEN38_FLASH_NEXT_FP8_WEIGHTS_ID,
            Self::Glm => sparkflash_model::descriptor::GLM53_FLASH_WEIGHTS_ID,
        };
        sparkflash_model::find_model_by_weights(weights)
            .expect("the model registry carries every pack's weight selector")
    }

    /// The variant serving a registry descriptor, or `None` for a family with no arm here.
    fn from_descriptor(descriptor: &ModelDescriptor) -> Option<Self> {
        match descriptor.weights_id.as_str() {
            sparkflash_model::descriptor::DEFAULT_WEIGHTS_ID => Some(Self::Deepseek),
            sparkflash_model::descriptor::QWEN38_FLASH_NEXT_WEIGHTS_ID => Some(Self::Qwen),
            sparkflash_model::descriptor::QWEN38_FLASH_NEXT_FP8_WEIGHTS_ID => Some(Self::QwenFp8),
            sparkflash_model::descriptor::GLM53_FLASH_WEIGHTS_ID => Some(Self::Glm),
            _ => None,
        }
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
        let descriptor = sparkflash_model::MODELS
            .iter()
            .find(|model| model.model_type.eq_ignore_ascii_case(&architecture.family))?;
        Self::from_descriptor(descriptor)
    }

    /// The parts a checkpoint's own `config.json` names that this build cannot load.
    ///
    /// The model-side `Architecture` says what the checkpoint is; the ops catalogue says which
    /// image implements each part, and [sparkflash_ops::parts::missing_loadable] keeps only the
    /// parts with an image this build can actually load (an embedded image, or the DeepSeek arm's
    /// run-time cubin directory). An empty list is the whole bring-up condition.
    /// A document that does not parse names no parts here; [Self::detect] already refused it.
    pub(super) fn missing_parts(directory: &Path) -> Vec<sparkflash_model::Part> {
        let path = directory.join(CHECKPOINT_CONFIG_NAME);
        let Ok(text) = std::fs::read_to_string(path) else {
            return Vec::new();
        };
        let Ok(architecture) = sparkflash_model::Architecture::from_json(&text) else {
            return Vec::new();
        };
        sparkflash_ops::parts::missing_loadable(&architecture, sparkflash_model::Target::Sm121)
    }

    /// Whether this variant has a [PackArm] (the DeepSeek arm is the default serving path).
    pub(super) const fn is_arm(self) -> bool {
        !matches!(self, Self::Deepseek)
    }

    /// The served id of this pack.
    pub(super) fn model_id(self) -> &'static str {
        self.descriptor().id.as_str()
    }

    /// The pack `model` names: a registry model id, weights id, family or id leaf, in any case.
    pub(super) fn named(model: &str) -> Option<Self> {
        let descriptor = sparkflash_model::find_model(model)?;
        Self::from_descriptor(descriptor)
    }

    /// The flags among those given that this pack's arm would not apply: on one node, the
    /// knobs its host arm lacks; in a group, which serves one request at a time over a window
    /// every rank allocates, `--concurrency` above 1, `--fuse-batch` and `--kv-budget-gib`.
    pub(super) fn unapplied(
        self,
        group: bool,
        concurrency: usize,
        fuse_batch: bool,
        kv_budget: bool,
    ) -> Vec<&'static str> {
        let (fuses, budgets) = match self {
            Self::Qwen | Self::QwenFp8 => (QwenArm::FUSED_BATCH, QwenArm::KV_BUDGET),
            Self::Glm => (GlmArm::FUSED_BATCH, GlmArm::KV_BUDGET),
            Self::Deepseek => (false, false),
        };
        let mut unapplied = Vec::new();
        if group && concurrency != 1 {
            unapplied.push("--concurrency");
        }
        if fuse_batch && (group || !fuses) {
            unapplied.push("--fuse-batch");
        }
        if kv_budget && (group || !budgets) {
            unapplied.push("--kv-budget-gib");
        }
        unapplied
    }

    /// Serves this pack: its group arm under `--tp4-rank`, else its single-node arm.
    pub(super) fn serve(self, options: &ServeOptions) -> ExitCode {
        let served = match self {
            Self::Deepseek => {
                unreachable!("the DeepSeek pack is served by the default reference path")
            }
            Self::Qwen => serve_arm::<QwenArm>(options),
            Self::QwenFp8 => serve_arm::<QwenFp8Arm>(options),
            Self::Glm => serve_arm::<GlmArm>(options),
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
pub(super) fn checkpoint_directory(checkpoint: &Path) -> PathBuf {
    if checkpoint.is_dir() {
        checkpoint.to_path_buf()
    } else {
        checkpoint.parent().unwrap_or(Path::new(".")).to_path_buf()
    }
}

/// The configuration of the checkpoint in `directory`, with the pack named in a refusal.
pub(super) fn read_config<A: PackArm>(directory: &Path) -> Result<A::Config, String> {
    A::read_config(directory)
        .map_err(|error| format!("cannot read the {} configuration: {error}", A::NAME))
}

/// The converted rank file for rank `rank` of `world` in the directory `var` names.
///
/// `None` when `var` is unset or empty. A configured directory that does not hold this rank's file
/// is a refusal, not a silent fallback to the checkpoint's tensors.
/// Turns the profile's speculation section into the GLM arm's drafter and depth.
///
/// The fields are the typed ones ([sparkflash_model::profile::SpeculationProfile]) and the
/// environment names that set them are declared in [sparkflash_model::profile::ENV_OVERRIDES], so
/// a switch here is a documented field rather than a new variable read at the point of use: a
/// block drafter's depth is adaptive (it reports a confidence a draft), the MTP layer's is fixed
/// (it reports none).
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
    if spec.enabled != Some(true) {
        return Ok(());
    }
    // Two drafts is what the former harness default was and what the MTP layer pays for; a block
    // drafter's block is longer, so a fixed depth is an operator's choice either way.
    let drafts = spec.depth.unwrap_or(2);
    let policy = spec.drafter.as_ref().map(|_| {
        sparkflash_model::DepthPolicy::adaptive(
            drafts,
            sparkflash_model_glm5_next::DFLASH2_STEP_COST,
        )
    });
    backend
        .enable_speculation(drafts, policy)
        .map_err(|error| error.to_string())?;
    eprintln!(
        "sparkflash serve: GLM speculating {drafts} draft(s) a step, {}",
        if spec.drafter.is_some() {
            "adaptive depth"
        } else {
            "fixed depth"
        }
    );
    Ok(())
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

pub(super) fn converted_rank_file(
    var: &str,
    rank: usize,
    world: usize,
) -> Result<Option<PathBuf>, String> {
    let Some(directory) = std::env::var_os(var).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    let directory = PathBuf::from(directory);
    find_converted_rank_file(&directory, rank, world)
        .map(Some)
        .ok_or_else(|| {
            format!(
                "{var}={} holds no {} for rank {rank} of {world}",
                directory.display(),
                rank_file_name(rank, world)
            )
        })
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
    let seam = A::codec(&tokenizer)?;
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
        None => sparkflash_weights::hub::cached_model_dir(A::MODEL_ID).ok_or_else(|| {
            format!(
                "{id} is not in the Hugging Face cache; run hf download {id} or pass --checkpoint DIR",
                id = A::MODEL_ID
            )
        })?,
    };
    let config = read_config::<A>(&directory)?;
    Ok((directory, config))
}

/// Refuse a backend whose typed configuration disagrees with the plan it was opened from.
///
/// The plan is the boot contract; the typed config is still the arm's weight recipe. This is the
/// seam where the two meet, so a family or layer-count drift is named before a weight is read.
fn check_backend_plan(plan: &EnginePlan, family: &str, layers: usize) -> Result<(), String> {
    if plan.family() != family {
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
/// This is the boot-time use of the model-independent plan: every part instance is resolved
/// to a loadable image before the arm opens a weight. The check is skipped for the synthetic
/// fixture, which exercises the serving path without claiming this build's device images.
fn check_plan<A: PackArm>(directory: &Path, world: usize) -> Result<EnginePlan, String> {
    let config = directory.join(CHECKPOINT_CONFIG_NAME);
    let plan = EnginePlan::from_path(&config, Target::Sm121, world, 0)
        .map_err(|error| format!("cannot plan the {} checkpoint: {error}", A::NAME))?;
    let resolver =
        KernelResolver::new(Target::Sm121, Availability::Runtime).with_family(plan.family());
    let failures = resolver.failures(&plan);
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
        Some(check_plan::<A>(&directory, 1)?)
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
    A::check_host(options, &config, context)?;
    let released_seam = if options.synthetic {
        None
    } else {
        Some(pack_seam::<A>(options, &directory)?.1)
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
    let driver = sparkflash_serve::HostDriver::start(
        Box::new(backend),
        Arc::clone(&seam),
        sparkflash_serve::HostOptions {
            max_tokens: options.max_tokens.min(A::MAX_OUTPUT_TOKENS),
            concurrency: options.concurrency,
            ..sparkflash_serve::HostOptions::default()
        },
    )?;
    let metrics = driver.metrics();
    println!(
        "sparkflash serve: {} ({}) at TP=1 on {}, context {context}",
        descriptor.id,
        descriptor.weights_id,
        directory.display()
    );
    // The byte codec's output is arbitrary text, not the released template's channels, so the
    // fixture stays on the raw driver.
    let engine: Arc<dyn ApiInferenceEngine> = if options.synthetic {
        Arc::new(driver)
    } else {
        A::respond(Arc::new(driver))
    };
    let numerics = ConfigKey::tp(descriptor.id.clone(), 1, options.numerics, CUDA_ARCH);
    Ok(serve_with_seam(
        engine, metrics, seam, None, options, descriptor, numerics,
    ))
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
    let checkpoint = options
        .checkpoint
        .as_deref()
        .ok_or_else(|| format!("--model {} --tp4-rank needs --checkpoint DIR", A::MODEL_ID))?;
    let directory = checkpoint_directory(checkpoint);
    let config = read_config::<A>(&directory)?;
    // Every rank derives the window from the same configuration, so the group agrees on the
    // sequence width without rank 0 sending it.
    let context = pack_context::<A>(options, &config)?;
    check_plan::<A>(&directory, tp4.world_size)?;
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
    if tp4.rank != 0 {
        pack_tp4_worker(options, tp4, context, A::OPEN_RANK)
            .map_err(|message| format!("rank {}: {message}", tp4.rank))?;
        return Ok(ExitCode::SUCCESS);
    }
    // The prompt seam and the detokenizer are the same file; load it before the rank upload so a
    // missing tokenizer costs a startup message, not a load.
    let (tokenizer, seam) = pack_seam::<A>(options, &directory)?;
    let descriptor = A::descriptor(&config, context);
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
    fn every_carried_profile_parses_and_keeps_its_registry_identity() {
        for (name, toml) in [
            ("DeepSeek-V4.1-Flash", DEEPSEEK_PROFILE_TOML),
            ("Qwen3.8-Flash-Next", QwenArm::PROFILE_TOML),
            ("GLM-5.3-Flash", GlmArm::PROFILE_TOML),
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
            "SPARKFLASH_QWEN4_SPEC_DEPTH" => Some("5".to_owned()),
            _ => None,
        })
        .expect("an environment override is valid");
        assert_eq!(overridden.prefill.chunk, Some(2048));
        assert_eq!(overridden.speculation.depth, Some(5));
    }

    #[test]
    fn every_released_config_is_detected_by_its_own_family() {
        for (config, expected) in [
            (DEEPSEEK_CONFIG, Pack::Deepseek),
            (QWEN_CONFIG, Pack::Qwen),
            (GLM_CONFIG, Pack::Glm),
        ] {
            let directory = tempfile::tempdir().expect("a temporary checkpoint");
            std::fs::write(directory.path().join(CHECKPOINT_CONFIG_NAME), config)
                .expect("the carried config is written");
            assert_eq!(Pack::detect(directory.path()), Some(expected));
        }
    }

    #[test]
    fn every_registry_id_and_weights_id_names_its_pack() {
        for model in sparkflash_model::MODELS.iter() {
            for id in [model.id.as_str(), model.weights_id.as_str()] {
                let pack =
                    Pack::named(id).unwrap_or_else(|| panic!("the registry id {id} names no pack"));
                assert_eq!(pack.model_id(), model.id);
                assert_eq!(pack.descriptor().weights_id, model.weights_id);
            }
        }
        assert_eq!(Pack::named("example/no-arm"), None);
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
}

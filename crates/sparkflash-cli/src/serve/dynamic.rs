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

//! The dynamic serving path: any Hugging Face cached checkpoint whose `config.json` plans and
//! resolves.
//!
//! This is the path the issue calls the zero-code one. A model id that is not in
//! [sparkflash_model::MODELS] is resolved through [sparkflash_weights::hub], parsed by the shared
//! [Architecture], planned by [EnginePlan] and checked against the [KernelResolver]. Its descriptor
//! and prompt dialect come from the checkpoint's own `config.json` and `tokenizer.json`; its
//! profile is the canonical preset when the family matches one, else the typed default with the
//! process environment applied.
//!
//! A resolved preset family is served by its existing device arm with the checkpoint's own
//! identity. Any other family is served by [super::dynamic_host::DynamicHostBackend], the CPU
//! reference path that runs [Architecture]'s parts. Phase 4's `EngineBuilder` is merged and the
//! released Qwen/GLM arms implement it, but a novel family has no arm to implement it; the host
//! path is the honest substitute the issue allows, and the limitation is stated in the boot
//! line and in docs/adding-a-model.md section 7.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use sparkflash_api::engine::InferenceEngine as ApiInferenceEngine;
use sparkflash_kernels::images::CUDA_ARCH;
use sparkflash_model::numerics::ConfigKey;
use sparkflash_model::prompt::PromptError;
use sparkflash_model::{
    Architecture, Capabilities, ModelBackend, ModelDescriptor, ModelProfile, PromptCodec, Target,
};
use sparkflash_ops::engine_plan::EnginePlan;
use sparkflash_ops::plan::KernelPhase;
use sparkflash_ops::resolver::{Availability, KernelResolver};

use super::dynamic_host::DynamicHostBackend;
use super::pack::PackArm;
use super::{ServeOptions, pack, serve_with_seam};

/// A checkpoint resolved from `--model` or `--checkpoint`, with its execution plan.
pub(crate) struct DynamicCheckpoint {
    /// The snapshot or directory the weights come from.
    pub directory: PathBuf,
    /// The id the run advertises.
    pub model_id: String,
    /// The checkpoint's shared architecture.
    pub architecture: Architecture,
    /// The plan at this target and world.
    pub plan: EnginePlan,
}

/// The prompt dialect a checkpoint's tokenizer declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dialect {
    /// GLM's `[gMASK]<sop>` and `<|user|>` turns.
    Glm,
    /// ChatML's `<|im_start|>`/`<|im_end|>` turns.
    Qwen,
    /// DeepSeek's own renderer, which its reference path normally serves.
    Deepseek,
}

/// Resolves `options` into a checkpoint and its plan.
///
/// # Errors
/// A missing model id, a cache miss, an unreadable configuration, or a plan that does not parse.
pub(crate) fn resolve(options: &ServeOptions) -> Result<DynamicCheckpoint, String> {
    let (directory, model_id) = resolve_directory(options)?;
    let architecture = Architecture::from_checkpoint_dir(&directory).map_err(|error| {
        format!(
            "--model {model_id}: {} is not a usable architecture: {error}",
            directory.join("config.json").display()
        )
    })?;
    let world = serving_world(options)?;
    let plan = EnginePlan::for_architecture(architecture.clone(), Target::Sm121, world, 0)
        .map_err(|error| format!("--model {model_id}: the execution plan: {error}"))?;
    Ok(DynamicCheckpoint {
        directory,
        model_id,
        architecture,
        plan,
    })
}

/// The world a dynamic run plans at: one node for the host path, and the group's world only when
/// the checkpoint belongs to a preset arm that has a group.
fn serving_world(options: &ServeOptions) -> Result<usize, String> {
    #[cfg(feature = "tp4-serve")]
    if let Some(tp4) = options.tp4.as_ref() {
        return Ok(tp4.world_size);
    }
    let _ = options;
    Ok(1)
}

/// The directory and served id an operator asked for.
///
/// `--checkpoint` wins when both are given: it names the weights unambiguously, and `--model`
/// then only sets the advertised identity. With `--model` alone the id is resolved through the
/// one hub resolver, after mapping a canonical weight selector to its registry id.
///
/// # Errors
/// The id is not cached, or is neither a model id nor a readable checkpoint.
pub(crate) fn resolve_directory(options: &ServeOptions) -> Result<(PathBuf, String), String> {
    if let Some(checkpoint) = options.checkpoint.as_deref() {
        let directory = super::pack::checkpoint_directory(checkpoint);
        let model_id = options
            .model
            .as_deref()
            .map(served_id_for)
            .unwrap_or_else(|| directory_id(&directory));
        return Ok((directory, model_id));
    }
    let model = options
        .model
        .as_deref()
        .ok_or_else(|| "no --model or --checkpoint to resolve".to_owned())?;
    let id = served_id_for(model);
    let cache = sparkflash_weights::hub::default_cache_dir();
    let directory = sparkflash_weights::hub::cached_model_dir(&id).ok_or_else(|| {
        match cache {
            Some(cache) => format!(
                "{id} is not in the Hugging Face cache at {}; run hf download {id} or pass --checkpoint DIR",
                cache.display()
            ),
            None => format!(
                "{id} is not in the Hugging Face cache (HF_HUB_CACHE is unset); run hf download {id} or pass --checkpoint DIR"
            ),
        }
    })?;
    Ok((directory, model.to_owned()))
}

pub(crate) fn served_id_for(model: &str) -> String {
    sparkflash_model::find_model(model)
        .map_or_else(|| model.to_owned(), |descriptor| descriptor.id.clone())
}

/// The id a checkpoint directory names: the hub repository folder when it is a cache snapshot,
/// else the directory's own name.
pub(crate) fn directory_id(directory: &Path) -> String {
    for ancestor in directory.ancestors() {
        if let Some(name) = ancestor.file_name().and_then(std::ffi::OsStr::to_str)
            && let Some(id) = sparkflash_weights::hub::folder_id(name)
        {
            return id;
        }
    }
    directory.file_name().map_or_else(
        || directory.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// Every unresolved requirement, each naming its part, rank-local geometry and precision.
///
/// The resolver's own [sparkflash_ops::resolver::ResolveError] carries no geometry, so this walks
/// the plan's requirements and asks each one in both passes. It is the boot gate and the
/// `EXIT_REFUSED` payload the issue asks for: never a silent fall-through to another model's
/// loader.
#[must_use]
pub(crate) fn failures(plan: &EnginePlan, availability: Availability) -> Vec<String> {
    let resolver = KernelResolver::new(Target::Sm121, availability);
    let mut failures = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for requirement in plan.requirements() {
        for phase in [KernelPhase::Prefill, KernelPhase::Decode] {
            if let Err(error) = resolver.resolve_for(requirement, phase) {
                let geometry = geometry_summary(&requirement.geometry);
                let reason = if error.reasons.is_empty() {
                    "no catalogue entry".to_owned()
                } else {
                    error.reasons.join("; ")
                };
                let line = format!(
                    "{} [{geometry}] at {} ({} pass): {reason}",
                    requirement.part,
                    requirement.precision.name(),
                    phase.name()
                );
                if seen.insert(line.clone()) {
                    failures.push(line);
                }
            }
        }
    }
    failures
}

fn geometry_summary(geometry: &sparkflash_ops::plan::PartGeometry) -> String {
    let mut fields = Vec::new();
    for (name, value) in [
        ("hidden", geometry.hidden),
        ("rows", geometry.rows),
        ("intermediate", geometry.intermediate),
        ("heads", geometry.heads),
        ("kv_heads", geometry.kv_heads),
        ("head_dim", geometry.head_dim),
        ("experts", geometry.experts),
        ("top_k", geometry.top_k),
        ("positions", geometry.positions),
        ("vocab", geometry.vocab),
        ("world", geometry.world),
        ("rank", geometry.rank),
    ] {
        if value != 0 {
            fields.push(format!("{name}={value}"));
        }
    }
    fields.join(", ")
}

/// The Qwen ChatML codec with Dory's own released template on top.
///
/// Dory shares ChatML's turn tokens and Qwen's tokenizer machinery, but not its prompt: the
/// released Dory template opens an empty system turn, never injects a reasoning-effort
/// instruction, and uses its own thinking markers on history. Delegating tokenization while
/// rendering Dory's template keeps both halves testable.
struct DoryCodec {
    inner: sparkflash_model_qwen4_exp::QwenCodec,
}

impl PromptCodec for DoryCodec {
    fn encode_chat(&self, request: &serde_json::Value) -> Result<Vec<u32>, PromptError> {
        let text = sparkflash_model_dory::prompt::render_chat(request)?;
        self.inner.encode_text(&text, false)
    }

    fn encode_text(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, PromptError> {
        self.inner.encode_text(text, add_special_tokens)
    }

    fn decode(&self, ids: &[u32]) -> Option<String> {
        self.inner.decode(ids)
    }

    fn token_str(&self, id: u32) -> Option<String> {
        self.inner.token_str(id)
    }

    fn vision_codec(&self) -> Option<Arc<dyn sparkflash_model::vision::VisionCodec>> {
        self.inner.vision_codec()
    }
}

/// The prompt codec and dialect the checkpoint's tokenizer declares.
fn codec(
    directory: &Path,
    architecture: &Architecture,
    configured: Option<&Path>,
) -> Result<(Arc<dyn PromptCodec>, Dialect), String> {
    // An explicit --tokenizer wins over the one beside the weights; a run without it reads the
    // checkpoint's own tokenizer.json.
    let tokenizer = configured.map_or_else(
        || directory.join("tokenizer.json"),
        std::path::Path::to_path_buf,
    );
    if !tokenizer.is_file() {
        return Err(match configured {
            Some(path) => format!(
                "--tokenizer {} does not exist (the checkpoint is {})",
                path.display(),
                directory.display()
            ),
            None => format!(
                "--model {} needs --tokenizer FILE, or a tokenizer.json beside the checkpoint at {}",
                architecture.family,
                tokenizer.display()
            ),
        });
    }
    let text = std::fs::read_to_string(&tokenizer)
        .map_err(|error| format!("cannot read {}: {error}", tokenizer.display()))?;
    if text.contains(sparkflash_model_glm5_next::codec::GMASK)
        && text.contains(sparkflash_model_glm5_next::codec::SOP)
    {
        let codec = sparkflash_model_glm5_next::GlmCodec::from_path_with_architecture(
            &tokenizer,
            architecture,
        )?;
        return Ok((Arc::new(codec), Dialect::Glm));
    }
    if text.contains(sparkflash_model_qwen4_exp::codec::IM_START)
        && text.contains(sparkflash_model_qwen4_exp::codec::IM_END)
    {
        // Dory's ChatML tokenizer has no Qwen <|endoftext|> or image delimiters, and its
        // released template is not Qwen's: it opens an empty system turn and never injects a
        // reasoning-effort instruction.
        if architecture.family == "dory" {
            let inner = sparkflash_model_qwen4_exp::QwenCodec::chatml_from_path_with_architecture(
                &tokenizer,
                architecture,
            )?;
            return Ok((Arc::new(DoryCodec { inner }), Dialect::Qwen));
        }
        let codec = sparkflash_model_qwen4_exp::QwenCodec::from_path_with_architecture(
            &tokenizer,
            architecture,
        )?;
        return Ok((Arc::new(codec), Dialect::Qwen));
    }
    if architecture.family == "deepseek_v41" {
        let seam = sparkflash_model_deepseek_v41::encoding::TokenizerSeam::from_path(&tokenizer)
            .map_err(|error| format!("{}: {error}", tokenizer.display()))?;
        return Ok((Arc::new(seam), Dialect::Deepseek));
    }
    Err(format!(
        "{} is not a tokenizer this path can name: it carries neither GLM's [gMASK]/<sop> nor ChatML's <|im_start|>/<|im_end|> markers, and the checkpoint family is {:?}. Pass --tokenizer FILE naming a supported tokenizer.json; the raw byte codec is no longer a silent fallback",
        tokenizer.display(),
        architecture.family
    ))
}

/// Whether a checkpoint family is served by a released device arm (rather than the host
/// reference), and therefore judged by the kernel catalogue. A Qwen3.5-MoE checkpoint is when
/// the device arm has its formats ([pack::qwen35_on_device]).
fn device_family(family: &str, directory: &Path) -> bool {
    match family {
        "qwen4_exp" | "qwen4_exp_text" | "glm5_next" => true,
        "qwen3_5_moe" | "qwen3_5_moe_text" => pack::qwen35_on_device(directory),
        // A Dory checkpoint is judged by the kernel catalogue only when this build carries the
        // Dory/Qwen cubins; otherwise the CPU host reference is its executor.
        "dory" => sparkflash_model_dory::DoryDeviceBackend::available(),
        _ => false,
    }
}

/// The typed profile a family serves: its carried preset when one exists, else the default with
/// the process environment's declared overrides.
fn profile(architecture: &Architecture) -> Result<ModelProfile, String> {
    let mut profile = match pack::profile_toml_for_family(&architecture.family) {
        Some(toml) => ModelProfile::from_toml(toml)
            .map_err(|error| format!("the {} profile: {error}", architecture.family))?,
        None => ModelProfile::default(),
    };
    profile
        .apply_process_env()
        .map_err(|error| format!("the {} profile: {error}", architecture.family))?;
    Ok(profile)
}

/// Serves a dynamically resolved checkpoint.
///
/// A preset family that this build has an arm for is served by that arm, with the checkpoint's
/// own id, window and tokenizer. Any other resolved family is served by the CPU reference host
/// backend, because the shared `EngineBuilder` backend is phase 4 and is not on this base.
///
/// # Errors
/// The resolution or plan failed, the plan has unresolved parts (the exact list is returned),
/// the tokenizer cannot be named, or the backend cannot open.
pub(crate) fn serve(options: &ServeOptions) -> Result<ExitCode, String> {
    #[cfg(feature = "tp4-serve")]
    if options.tp4.is_some() {
        return Err(
            "a dynamically resolved model serves at TP=1 on this base: the shared EngineBuilder is TP1 and a novel family has no device arm. Name a released preset with --model to use its group, or drop the --tp4-rank flags"
                .to_owned(),
        );
    }
    let checkpoint = resolve(options)?;
    let on_device = device_family(&checkpoint.architecture.family, &checkpoint.directory);
    let unresolved = if on_device {
        // The catalogue resolver sorts canonical entries first, so a Dory plan can resolve
        // through a Qwen image. Check the Dory image's own symbol list before advertising it.
        if checkpoint.architecture.family == "dory" {
            sparkflash_ops::parts::dynamic_device_arm(&checkpoint.architecture)?;
        }
        failures(&checkpoint.plan, Availability::Catalogue)
    } else {
        // The host reference is the executor for a novel family. The CUDA catalogue answers a
        // different question (whether a kernel image carries the part) and used to over-accept a
        // catalogue-shaped MLA while refusing a host-runnable GQA at another hidden width.
        let parts = crate::dynamic::unsupported_parts(&checkpoint.architecture);
        if parts.is_empty() {
            Vec::new()
        } else {
            vec![format!(
                "the host reference path does not execute: {}",
                parts
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )]
        }
    };
    if !unresolved.is_empty() {
        return Err(format!(
            "--model {} resolves to a plan this build cannot execute at TP1 on sm121: {}",
            checkpoint.model_id,
            unresolved.join("; ")
        ));
    }
    if options.recurrence.is_some() && checkpoint.plan.family() != "dory" {
        return Err(format!(
            "--recurrence-rank runs Dory's shared recurrent block a rank; --model {} is a {} \
             checkpoint, which has no latent recurrence to split",
            checkpoint.model_id,
            checkpoint.plan.family()
        ));
    }
    let window = checkpoint.architecture.max_position_embeddings;
    let context = options
        .context
        .unwrap_or(if window == 0 { 4096 } else { window });
    if context == 0 || (window != 0 && context > window) {
        return Err(format!(
            "context limit {context} is outside 1..={window} for {}",
            checkpoint.model_id
        ));
    }
    let (seam, dialect) = codec(
        &checkpoint.directory,
        &checkpoint.architecture,
        options.tokenizer.as_deref(),
    )?;
    // A canonical family's served descriptor takes its capabilities from the registry, so the
    // dynamic route does not re-advertise a control the arm's template does not consume.
    let capabilities = sparkflash_model::find_model_by_type(&checkpoint.architecture.family)
        .map_or(Capabilities::ALL, |model| model.capabilities);
    let descriptor = ModelDescriptor::from_architecture(
        checkpoint.model_id.clone(),
        &checkpoint.architecture,
        context,
        capabilities,
    );
    let profile = profile(&checkpoint.architecture)?;
    sparkflash_model::install_active(profile.clone());
    eprintln!(
        "sparkflash serve: {} profile model={} family={} weights={}",
        checkpoint.model_id,
        profile.model.as_deref().unwrap_or("-"),
        profile.family.as_deref().unwrap_or("-"),
        profile.weights.as_deref().unwrap_or("-")
    );
    eprintln!(
        "sparkflash serve: dynamic plan {}",
        checkpoint.plan.boot_summary()
    );
    match checkpoint.plan.family() {
        "qwen4_exp" | "qwen4_exp_text" => super::pack::serve_arm_identity::<super::pack::QwenArm>(
            options,
            checkpoint.directory,
            descriptor,
            seam,
            context,
            super::pack::QwenArm::respond,
        ),
        "qwen3_5_moe" | "qwen3_5_moe_text" if on_device => {
            super::pack::serve_arm_identity::<super::pack::Qwen36Arm>(
                options,
                checkpoint.directory,
                descriptor,
                seam,
                context,
                super::pack::Qwen36Arm::respond,
            )
        }
        "glm5_next" => super::pack::serve_arm_identity::<super::pack::GlmArm>(
            options,
            checkpoint.directory,
            descriptor,
            seam,
            context,
            super::pack::GlmArm::respond,
        ),
        "dory" => match options.recurrence.as_ref() {
            #[cfg(feature = "cuda")]
            Some(group) => super::recurrence::serve(
                checkpoint, descriptor, seam, dialect, context, options, group,
            ),
            #[cfg(not(feature = "cuda"))]
            Some(_) => Err(
                "--recurrence-rank runs the Dory device arm over NCCL, and this build has no cuda feature"
                    .to_owned(),
            ),
            None => serve_dory_host(checkpoint, descriptor, seam, dialect, context, options),
        },
        _ => serve_host(checkpoint, descriptor, seam, dialect, context, options),
    }
}

fn serve_host(
    checkpoint: DynamicCheckpoint,
    descriptor: ModelDescriptor,
    seam: Arc<dyn PromptCodec>,
    dialect: Dialect,
    context: usize,
    options: &ServeOptions,
) -> Result<ExitCode, String> {
    let source = sparkflash_weights::safetensors::open_checkpoint(&checkpoint.directory)
        .map_err(|error| format!("{}: {error}", checkpoint.directory.display()))?;
    let backend = DynamicHostBackend::open(
        descriptor.clone(),
        Arc::new(checkpoint.architecture),
        Arc::new(source),
        context,
    )
    .map_err(|error| format!("--model {}: {error}", checkpoint.model_id))?;
    let banner = format!(
        "{} ({}) at TP=1 on {}, context {context}, dynamic host reference",
        descriptor.id,
        descriptor.weights_id,
        checkpoint.directory.display()
    );
    serve_backend(
        Box::new(backend),
        descriptor,
        seam,
        dialect,
        options,
        &banner,
    )
}

/// Serves `backend` behind HTTP through the generic host driver, in `dialect`, under the TP=1
/// numerics key; `banner` is the boot line's description of what answers.
pub(crate) fn serve_backend(
    backend: Box<dyn ModelBackend>,
    descriptor: ModelDescriptor,
    seam: Arc<dyn PromptCodec>,
    dialect: Dialect,
    options: &ServeOptions,
    banner: &str,
) -> Result<ExitCode, String> {
    let driver = sparkflash_serve::HostDriver::start(
        backend,
        Arc::clone(&seam),
        sparkflash_serve::HostOptions {
            max_tokens: options.max_tokens,
            concurrency: options.concurrency,
            ..sparkflash_serve::HostOptions::default()
        },
    )?;
    let metrics = driver.metrics();
    let engine: Arc<dyn ApiInferenceEngine> = match dialect {
        Dialect::Glm => sparkflash_api::glm::GlmResponseEngine::layer(Arc::new(driver)),
        Dialect::Qwen => sparkflash_api::qwen::QwenResponseEngine::layer(Arc::new(driver)),
        Dialect::Deepseek => Arc::new(driver),
    };
    println!("sparkflash serve: {banner}");
    let numerics = ConfigKey::tp(descriptor.id.clone(), 1, options.numerics, CUDA_ARCH);
    Ok(serve_with_seam(
        engine, metrics, seam, None, options, descriptor, numerics,
    ))
}

/// Serves a Dory checkpoint through its device arm, or the host reference when this build has
/// no cubin.
fn serve_dory_host(
    checkpoint: DynamicCheckpoint,
    descriptor: ModelDescriptor,
    seam: Arc<dyn PromptCodec>,
    dialect: Dialect,
    context: usize,
    options: &ServeOptions,
) -> Result<ExitCode, String> {
    let config = sparkflash_model_dory::DoryConfig::from_checkpoint_dir(&checkpoint.directory)
        .map_err(|error| {
            format!(
                "--model {}: {}: {error}",
                checkpoint.model_id,
                checkpoint.directory.join("config.json").display()
            )
        })?;
    let device = sparkflash_model_dory::DoryDeviceBackend::available();
    let backend: Box<dyn ModelBackend> = if device {
        Box::new(
            sparkflash_model_dory::DoryDeviceBackend::open_with_descriptor(
                &checkpoint.directory,
                descriptor.clone(),
                context,
            )
            .map_err(|error| format!("--model {}: {error}", checkpoint.model_id))?,
        )
    } else {
        let source = sparkflash_weights::safetensors::open_checkpoint(&checkpoint.directory)
            .map_err(|error| format!("{}: {error}", checkpoint.directory.display()))?;
        Box::new(
            sparkflash_model_dory::DoryHostBackend::open(
                descriptor.clone(),
                config,
                &source,
                context,
            )
            .map_err(|error| format!("--model {}: {error}", checkpoint.model_id))?,
        )
    };
    let banner = format!(
        "{} ({}) at TP=1 on {}, context {context}, Dory {} reference",
        descriptor.id,
        descriptor.weights_id,
        checkpoint.directory.display(),
        if device { "device" } else { "host" }
    );
    serve_backend(backend, descriptor, seam, dialect, options, &banner)
}

/// Whether a non-registry `--model` names a DeepSeek checkpoint, which the existing reference
/// path serves rather than the dynamic host.
#[must_use]
pub(crate) fn names_deepseek(options: &ServeOptions) -> bool {
    resolve_directory(options)
        .ok()
        .and_then(|(directory, _)| Architecture::from_checkpoint_dir(&directory).ok())
        .is_some_and(|architecture| architecture.family == "deepseek_v41")
}

#[cfg(test)]
mod tests {
    use super::device_family;
    use sparkflash_model::Target;
    use sparkflash_ops::engine_plan::EnginePlan;

    #[test]
    fn the_qwen_text_alias_dispatches_to_the_qwen_arm() {
        let nowhere = std::path::Path::new("/nonexistent/checkpoint");
        assert!(device_family("qwen4_exp", nowhere));
        assert!(device_family("qwen4_exp_text", nowhere));
        assert!(device_family("glm5_next", nowhere));
        assert!(!device_family("acme_custom_flash", nowhere));
        // A Qwen3.5-MoE checkpoint is the device arm's only when its formats are.
        assert!(!device_family("qwen3_5_moe", nowhere));
    }

    #[test]
    fn dory_plans_at_tp1_and_names_the_host_path() {
        const CONFIG: &str = include_str!(
            "../../../sparkflash-model-dory/config/nvidia_nemotron_labs_dory1_0_4b_bf16.hf.json"
        );
        let plan =
            EnginePlan::from_json(CONFIG, Target::Sm121, 1, 0).expect("the Dory checkpoint plans");
        assert_eq!(plan.family(), "dory");
        assert_eq!(plan.architecture().layers.len(), 36);
        assert!(plan.architecture().looped.is_some());
        assert_eq!(
            super::super::pack::profile_toml_for_family("dory").map(str::len),
            Some(include_str!("../../../sparkflash-model-dory/config/dory1_0_4b_bf16.toml").len())
        );
    }

    #[test]
    fn a_qwen35_checkpoint_goes_to_the_device_by_its_formats() {
        use sparkflash_model_qwen4_exp::config::{DenseQuant, Qwen4ExpConfig};
        for (format, on_device) in [(DenseQuant::Fp8Block, true), (DenseQuant::Bf16, false)] {
            let directory = tempfile::tempdir().expect("a temporary checkpoint");
            let config = Qwen4ExpConfig::tiny_qwen35_moe(format);
            std::fs::write(directory.path().join("config.json"), config.to_json())
                .expect("the configuration is written");
            assert_eq!(
                device_family("qwen3_5_moe", directory.path()),
                on_device,
                "{format:?}"
            );
        }
    }
}

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
//! identity. Any other family that is the dense grouped-query decoder the shared dense arm
//! executes ([sparkflash_ops::parts::dense_device_refusal]) is served on the device by
//! [sparkflash_model_dense::DenseDeviceBackend], with no code of its own; the rest, and an
//! operator who opts out with [DENSE_DEVICE_ENV]`=0`, are served by
//! [super::dynamic_host::DynamicHostBackend], the CPU reference path that runs [Architecture]'s
//! parts. The boot line names which one answers and, on the host, why the device arm did not;
//! docs/adding-a-model.md section 7 states the limits. The dense arm serves on the shared
//! [sparkflash_runtime::TickEngine] with [dense_lanes] lanes; the host reference answers one
//! request at a time and refuses `--concurrency` and `--fuse-batch`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use sparkflash_api::engine::InferenceEngine as ApiInferenceEngine;
use sparkflash_model::prompt::PromptError;
use sparkflash_model::{
    Architecture, Arm as _, Capabilities, ModelBackend, ModelDescriptor, ModelProfile, PromptCodec,
    Target,
};
use sparkflash_ops::engine_plan::EnginePlan;
use sparkflash_ops::plan::KernelPhase;
use sparkflash_ops::resolver::{Availability, KernelResolver};

use super::dynamic_host::DynamicHostBackend;
use super::{ServeOptions, pack};

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

/// The prompt dialect a checkpoint serves in: its family's registration when that declares one,
/// else what the tokenizer's own markers say ([codec]).
pub(crate) use sparkflash_model::Dialect;

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
    if let Some(group) = options.group.as_ref() {
        return Ok(group.world);
    }
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

/// The prompt codec and dialect of a checkpoint.
///
/// The family's registration ([sparkflash_model::Arm::tokenizer]) declares the dialect of an arm
/// whose tokenizer carries no marker of its own (Inkling, DeepSeek) and Dory's template over
/// ChatML's turns; GLM's and ChatML's markers name themselves, which is also how a family with
/// no registration is read.
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
    let declared = sparkflash_model::arm_for_family(&architecture.family)
        .and_then(|arm| arm.tokenizer().dialect);
    if text.contains(sparkflash_model_glm5_next::codec::GMASK)
        && text.contains(sparkflash_model_glm5_next::codec::SOP)
    {
        let codec = sparkflash_model_glm5_next::GlmCodec::from_path_with_architecture(
            &tokenizer,
            architecture,
        )?;
        return Ok((Arc::new(codec), Dialect::Glm));
    }
    if declared == Some(Dialect::Inkling) {
        let codec = sparkflash_model_inkling2::InklingCodec::from_path(
            &tokenizer,
            architecture.tokens.bos_token_id,
            architecture.tokens.eos_token_ids.clone(),
        )
        .map_err(|error| format!("{}: {error}", tokenizer.display()))?;
        // The tower's geometry plans the image spans; a checkpoint without one serves text.
        let config =
            sparkflash_model_inkling2::config::Inkling2Config::from_checkpoint_dir(directory)
                .map_err(|error| format!("{}: {error}", directory.display()))?;
        let codec = match config.vision {
            Some(vision) => codec
                .with_vision(vision)
                .map_err(|error| format!("{}: {error}", tokenizer.display()))?,
            None => codec,
        };
        return Ok((Arc::new(codec), Dialect::Inkling));
    }
    if text.contains(sparkflash_model_qwen4_exp::codec::IM_START)
        && text.contains(sparkflash_model_qwen4_exp::codec::IM_END)
    {
        // Dory's ChatML tokenizer has no Qwen <|endoftext|> or image delimiters, and its
        // released template is not Qwen's: it opens an empty system turn and never injects a
        // reasoning-effort instruction.
        if declared == Some(Dialect::Dory) {
            let inner = sparkflash_model_qwen4_exp::QwenCodec::chatml_from_path_with_architecture(
                &tokenizer,
                architecture,
            )?;
            return Ok((Arc::new(DoryCodec { inner }), Dialect::Dory));
        }
        let codec = sparkflash_model_qwen4_exp::QwenCodec::from_path_with_architecture(
            &tokenizer,
            architecture,
        )?;
        return Ok((Arc::new(codec), Dialect::Qwen));
    }
    if declared == Some(Dialect::Deepseek) {
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
/// reference), and therefore judged by the kernel catalogue: a typed pack that takes the
/// checkpoint ([pack::typed_for_checkpoint]; a Qwen3.5-MoE checkpoint is one when the device arm
/// has its formats), or Dory with its images.
fn device_family(family: &str, directory: &Path) -> bool {
    // A Dory checkpoint is judged by the kernel catalogue only when this build carries the
    // Dory/Qwen cubins; otherwise the CPU host reference is its executor.
    pack::typed_for_checkpoint(family, directory).is_some()
        || (family == "dory" && sparkflash_model_dory::DoryDeviceBackend::available())
}

/// The typed profile a family serves: its carried preset when one exists, else the default with
/// the process environment's declared overrides.
fn profile(architecture: &Architecture) -> Result<ModelProfile, String> {
    profile_with(architecture, |name| std::env::var(name).ok())
}

/// [profile], with the environment read through `lookup` so a test can set one switch.
fn profile_with(
    architecture: &Architecture,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<ModelProfile, String> {
    let mut profile = match pack::profile_toml_for_family(&architecture.family) {
        Some(toml) => ModelProfile::from_toml(toml)
            .map_err(|error| format!("the {} profile: {error}", architecture.family))?,
        None => ModelProfile::default(),
    };
    profile
        .apply_env(lookup)
        .map_err(|error| format!("the {} profile: {error}", architecture.family))?;
    // An arm whose decode is the row GEMV refuses the tensor-core switch where its profile is
    // read, in the one message every such arm gives.
    if let Some(arm) = sparkflash_model::arm_for_family(&architecture.family) {
        sparkflash_model::arms::check_profile(arm, &profile)?;
    }
    Ok(profile)
}

/// Serves a dynamically resolved checkpoint.
///
/// A preset family that this build has an arm for is served by that arm, with the checkpoint's
/// own id, window and tokenizer. Any other resolved family is served by the shared dense device
/// arm when it takes the checkpoint, else by the CPU reference host backend ([serve_novel]).
///
/// # Errors
/// The resolution or plan failed, the plan has unresolved parts (the exact list is returned),
/// the tokenizer cannot be named, or the backend cannot open.
pub(crate) fn serve(options: &ServeOptions) -> Result<ExitCode, String> {
    let checkpoint = resolve(options)?;
    let on_device = device_family(&checkpoint.architecture.family, &checkpoint.directory);
    let unresolved = if checkpoint.architecture.family == "inkling2" {
        // Its host backend runs the new relative/short-conv/block arithmetic itself; the CUDA
        // catalogue is a device-arm gate, not this path's capability list.
        Vec::new()
    } else if on_device {
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
            "--model {} resolves to a plan this build cannot execute at TP{} on sm121: {}",
            checkpoint.model_id,
            serving_world(options)?,
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
    // Every rank of the device group allocates its KV at the admitted context, and a released
    // 1M window is not what a first server wants: the arm states its group's default
    // ([sparkflash_model::GroupDefaults::context]). An explicit --context still reaches the whole
    // window, and validation below still bounds it.
    let group_context = sparkflash_model::arm_for_family(&checkpoint.architecture.family)
        .and_then(|arm| arm.group().context);
    let context = match (options.group.is_some(), options.context, group_context) {
        (true, None, Some(admitted)) => context.min(admitted),
        _ => context,
    };
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
    // Only a family the match below sends to serve_novel steps lanes on this path (the shared
    // dense device arm); every other family keeps one request at a time here, as before.
    let family = checkpoint.plan.family();
    let typed = pack::typed_for_checkpoint(family, &checkpoint.directory);
    let novel = typed.is_none() && !matches!(family, "inkling2" | "dory");
    #[cfg(feature = "tp4-serve")]
    if options.tp4.is_some() && !novel {
        return Err(format!(
            "--model {} is a {family} checkpoint: a dynamically resolved model serves a --tp4-rank \
             group only on the shared dense device arm. Name the released preset with --model to \
             use its own group, or drop the --tp4-rank flags",
            checkpoint.model_id
        ));
    }
    if !novel
        && (options.concurrency != 1 || options.fuse_batch)
        && options.group.is_none()
        && options.recurrence.is_none()
    {
        return Err(format!(
            "--model {} is a {family} checkpoint, which this path serves one request at a time: \
             --concurrency and --fuse-batch step the shared dense device arm's lanes; name the \
             released preset for its own lanes",
            checkpoint.model_id
        ));
    }
    if let Some(typed) = typed {
        return (typed.serve_identity)(options, checkpoint.directory, descriptor, seam, context);
    }
    match checkpoint.plan.family() {
        "inkling2" if options.group.is_some() => {
            serve_inkling_group(options, checkpoint, descriptor, seam, dialect, context)
        }
        "inkling2" => serve_inkling_host(
            checkpoint,
            descriptor,
            seam,
            dialect,
            context,
            options,
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
        _ => serve_novel(checkpoint, descriptor, seam, dialect, context, options),
    }
}

/// The variable that chooses where a family no pack transcribes runs: unset prefers the shared
/// dense device arm and states why when the host reference answers instead, `1` requires the
/// arm (a checkpoint it cannot open is a boot refusal), and `0` keeps the host reference.
pub(crate) const DENSE_DEVICE_ENV: &str = "SPARKFLASH_DENSE_DEVICE";

/// What [DENSE_DEVICE_ENV] asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DenseDevice {
    /// The arm when it takes the checkpoint, else the host reference with the reason.
    Prefer,
    /// The arm, or a refusal.
    Require,
    /// The host reference.
    Off,
}

fn dense_device(value: Option<&str>) -> Result<DenseDevice, String> {
    match value.map(str::trim) {
        None | Some("") => Ok(DenseDevice::Prefer),
        Some("1" | "on" | "true") => Ok(DenseDevice::Require),
        Some("0" | "off" | "false") => Ok(DenseDevice::Off),
        Some(other) => Err(format!(
            "{DENSE_DEVICE_ENV}={other:?} is neither 1 (require the shared dense device arm) nor \
             0 (serve on the host reference); unset it to prefer the arm"
        )),
    }
}

/// Serves a family no pack transcribes: on the shared dense device arm when it takes the
/// checkpoint and [DENSE_DEVICE_ENV] allows it, else on the host reference, naming why.
fn serve_novel(
    checkpoint: DynamicCheckpoint,
    descriptor: ModelDescriptor,
    seam: Arc<dyn PromptCodec>,
    dialect: Dialect,
    context: usize,
    options: &ServeOptions,
) -> Result<ExitCode, String> {
    let wanted = dense_device(std::env::var(DENSE_DEVICE_ENV).ok().as_deref())?;
    #[cfg(feature = "tp4-serve")]
    if let Some(tp4) = options.tp4.as_ref() {
        return serve_dense_group(
            &checkpoint,
            descriptor,
            seam,
            dialect,
            context,
            options,
            tp4,
            wanted,
        );
    }
    let reason = if wanted == DenseDevice::Off {
        format!("{DENSE_DEVICE_ENV}=0")
    } else {
        let opened = sparkflash_ops::parts::dynamic_device_arm(&checkpoint.architecture)
            .and_then(|arm| {
                if arm == sparkflash_ops::parts::DENSE_DEVICE_ARM {
                    Ok(())
                } else {
                    Err(format!("{arm} is not this path's"))
                }
            })
            .and_then(|()| {
                let source =
                    sparkflash_weights::safetensors::open_checkpoint(&checkpoint.directory)
                        .map_err(|error| format!("{}: {error}", checkpoint.directory.display()))?;
                sparkflash_model_dense::DenseDeviceBackend::open(
                    descriptor.clone(),
                    &checkpoint.architecture,
                    &source,
                    context,
                )
            });
        match opened {
            Ok(mut backend) => {
                let lanes = dense_lanes(
                    options.concurrency_set.then_some(options.concurrency),
                    sparkflash_model::active_profile().runtime.concurrency,
                )?;
                backend
                    .enable_fusion(lanes)
                    .map_err(|error| format!("--model {}: {error}", checkpoint.model_id))?;
                let banner = format!(
                    "{} ({}) at TP=1 on {}, context {context}, shared dense device arm ({:.2} GiB \
                     of BF16 weights, {lanes} lanes)",
                    descriptor.id,
                    descriptor.weights_id,
                    checkpoint.directory.display(),
                    backend.weight_bytes() as f64 / f64::from(1u32 << 30)
                );
                let mut options = options.clone();
                options.concurrency = lanes;
                return serve_backend(
                    Box::new(backend),
                    descriptor,
                    seam,
                    dialect,
                    &options,
                    &banner,
                );
            }
            Err(reason) if wanted == DenseDevice::Require => {
                return Err(format!(
                    "--model {}: {DENSE_DEVICE_ENV}=1 requires the shared dense device arm: \
                     {reason}",
                    checkpoint.model_id
                ));
            }
            Err(reason) => reason,
        }
    };
    if options.concurrency != 1 || options.fuse_batch {
        return Err(format!(
            "--model {}: --concurrency and --fuse-batch step the shared dense device arm's lanes, \
             and the host reference, which serves one request at a time, answers instead: \
             {reason}",
            checkpoint.model_id
        ));
    }
    eprintln!(
        "sparkflash serve: {} runs on the host reference, not the shared dense device arm: \
         {reason}",
        checkpoint.model_id
    );
    serve_host(checkpoint, descriptor, seam, dialect, context, options)
}

/// Serves one rank of a `--tp4-rank` group on the shared dense device arm: each rank opens its
/// share of the checkpoint over the pack collective ([super::open_pack_collective]: NCCL by
/// default, the host seam with `SPARK_COMM_REDUCE=tcp`, both summing in rank order), rank 0 fuses
/// the group's lanes behind HTTP and the workers mirror them until the group ends. The host
/// reference runs alone, so a checkpoint the arm does not take is refused here rather than served
/// at another world than the one asked for.
#[cfg(feature = "tp4-serve")]
#[allow(clippy::too_many_arguments)]
fn serve_dense_group(
    checkpoint: &DynamicCheckpoint,
    descriptor: ModelDescriptor,
    seam: Arc<dyn PromptCodec>,
    dialect: Dialect,
    context: usize,
    options: &ServeOptions,
    tp4: &super::Tp4ServeOptions,
    wanted: DenseDevice,
) -> Result<ExitCode, String> {
    let model = &checkpoint.model_id;
    if wanted == DenseDevice::Off {
        return Err(format!(
            "--model {model}: {DENSE_DEVICE_ENV}=0 keeps the host reference, which runs on one \
             rank; drop the --tp4-rank flags or unset it"
        ));
    }
    if tp4.device.is_some() {
        return Err(format!(
            "--model {model}: --tp4-device configures DeepSeek's device harness; the shared dense \
             device arm's group takes none of it"
        ));
    }
    let arm =
        sparkflash_ops::parts::dynamic_device_arm(&checkpoint.architecture).map_err(|reason| {
            format!(
                "--model {model}: a --tp4-rank group needs the shared dense device arm: {reason}"
            )
        })?;
    if arm != sparkflash_ops::parts::DENSE_DEVICE_ARM {
        return Err(format!(
            "--model {model}: a --tp4-rank group needs the shared dense device arm, not {arm}"
        ));
    }
    let lanes = dense_lanes(
        options.concurrency_set.then_some(options.concurrency),
        sparkflash_model::active_profile().runtime.concurrency,
    )?;
    let mut rank_options = options.clone();
    rank_options.checkpoint = Some(checkpoint.directory.clone());
    let mut config = super::tp4_rank_config(&rank_options, tp4, super::TP4_CONTROL_TIMEOUT)?;
    config.context = context;
    let started = std::time::Instant::now();
    let cuda =
        sparkflash_kernels::cuda_driver::Cuda::load().map_err(|error| format!("CUDA: {error}"))?;
    let (tp, _gather) = super::open_pack_collective(
        &config,
        &cuda,
        "shared dense",
        checkpoint.architecture.hidden_size,
        &format!(
            "dense-serve;{model};{};{context}",
            checkpoint.directory.display()
        ),
    )?;
    let label = tp.label();
    let source = sparkflash_weights::safetensors::open_checkpoint(&checkpoint.directory)
        .map_err(|error| format!("{}: {error}", checkpoint.directory.display()))?;
    let mut backend = sparkflash_model_dense::DenseDeviceBackend::open_rank(
        descriptor.clone(),
        &checkpoint.architecture,
        &source,
        context,
        tp,
    )
    .map_err(|error| format!("--model {model} rank {}: {error}", tp4.rank))?;
    eprintln!(
        "sparkflash serve: {model} rank {}/{} loaded in {:.1} s on the shared dense device arm \
         ({:.2} GiB of weights); {label} collective",
        tp4.rank,
        tp4.world_size,
        started.elapsed().as_secs_f64(),
        backend.weight_bytes() as f64 / f64::from(1u32 << 30)
    );
    if tp4.rank != 0 {
        let mut control = super::tp4_control_worker(&config)?;
        match backend
            .mirror_group(&mut control, lanes)
            .map_err(|error| format!("rank {}: {error}", tp4.rank))? {}
    }
    let hub = sparkflash_comm::control::Tp4ControlHub::start(
        config.control_address(),
        config.world_size,
        config.timeout,
    )
    .map_err(|error| format!("rank 0 control channel: {error}"))?;
    backend
        .enable_group_fusion(lanes, hub)
        .map_err(|error| format!("--model {model}: cannot fuse the group's lanes: {error}"))?;
    let banner = format!(
        "{} ({}) at TP={} rank 0 on {}, context {context}, shared dense device arm ({lanes} \
         lanes; workers never detokenize)",
        descriptor.id,
        descriptor.weights_id,
        tp4.world_size,
        checkpoint.directory.display(),
    );
    let mut options = options.clone();
    options.concurrency = lanes;
    serve_backend_world(
        Box::new(backend),
        descriptor,
        seam,
        dialect,
        &options,
        &banner,
        tp4.world_size,
    )
}

/// The lanes the shared dense device arm steps: `--concurrency` when given, else the profile's
/// `runtime.concurrency` (`SPARKFLASH_LANES`), else the arm's own default
/// ([sparkflash_model::GroupDefaults::lanes]). `--concurrency 1` keeps one lane, which still
/// keeps its conversation between requests.
///
/// # Errors
/// A flag and a profile that ask for different lanes.
fn dense_lanes(flag: Option<usize>, profile: Option<usize>) -> Result<usize, String> {
    match (flag, profile) {
        (Some(flag), Some(profile)) if flag != profile => Err(format!(
            "--concurrency {flag} and the profile's runtime.concurrency = {profile} ask the shared \
             dense device arm for different lanes; give one"
        )),
        (Some(lanes), _) | (None, Some(lanes)) => Ok(lanes),
        (None, None) => Ok(sparkflash_model::Dense.group().lanes),
    }
}

/// Serves an Inkling checkpoint through its pack's f32 host reference.
fn serve_inkling_host(
    checkpoint: DynamicCheckpoint,
    descriptor: ModelDescriptor,
    seam: Arc<dyn PromptCodec>,
    dialect: Dialect,
    context: usize,
    options: &ServeOptions,
) -> Result<ExitCode, String> {
    let config =
        sparkflash_model_inkling2::Inkling2Config::from_checkpoint_dir(&checkpoint.directory)
            .map_err(|error| format!("--model {}: {error}", checkpoint.model_id))?;
    // A staged converted rank file is served as it is: the same plan, the same arithmetic, and a
    // load that is a walk over one file instead of a sharded safetensors seek per tensor.
    let layout = sparkflash_engine::parallel_tp::TpLayout {
        rank: 0,
        world_size: 1,
    };
    let converted = sparkflash_weights::rank_file::converted_rank_file(
        crate::serve::pack::INKLING_CONVERTED_DIR_ENV,
        layout.rank,
        layout.world_size,
    )?;
    let (source, where_from): (Arc<dyn sparkflash_weights::WeightSource>, String) = match &converted
    {
        Some(path) => (
            Arc::new(
                sparkflash_model_inkling2::RankFileSource::open(path, &config, layout)
                    .map_err(|error| format!("{}: {error}", path.display()))?,
            ),
            format!("converted {}", path.display()),
        ),
        None => (
            Arc::new(
                sparkflash_weights::safetensors::open_checkpoint(&checkpoint.directory)
                    .map_err(|error| format!("{}: {error}", checkpoint.directory.display()))?,
            ),
            checkpoint.directory.display().to_string(),
        ),
    };
    let backend = sparkflash_model_inkling2::Inkling2HostBackend::open(
        descriptor.clone(),
        config,
        source,
        context,
    )
    .map_err(|error| format!("--model {}: {error}", checkpoint.model_id))?;
    let banner = format!(
        "{} ({}) at TP=1 on {where_from}, context {context}, Inkling f32 host reference",
        descriptor.id, descriptor.weights_id
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

/// Serves an Inkling checkpoint through its four-rank device arm.
///
/// Rank 0 owns HTTP, the tokenizer and the control channel; every rank holds its own device state
/// and mirrors the token ids rank 0 admits. The dispatch is here rather than in the pack because
/// the device arm's group needs the resolved checkpoint, its tokenizer seam and the served
/// descriptor, all of which this module built.
fn serve_inkling_group(
    options: &ServeOptions,
    checkpoint: DynamicCheckpoint,
    descriptor: ModelDescriptor,
    seam: Arc<dyn PromptCodec>,
    dialect: Dialect,
    context: usize,
) -> Result<ExitCode, String> {
    #[cfg(feature = "tp4-serve")]
    {
        super::inkling_group::serve(options, checkpoint, descriptor, seam, dialect, context)
    }
    #[cfg(not(feature = "tp4-serve"))]
    {
        let _ = (options, checkpoint, descriptor, seam, dialect, context);
        Err(
            "--rank is the Inkling device arm's four-rank group: this build carries no tp4-serve feature, so rebuild with --features tp4-serve"
                .to_owned(),
        )
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

/// The response engine `dialect` reads replies with, over the raw driver: the one place a
/// dialect meets its family's reasoning and tool channels.
pub(crate) fn respond(
    dialect: Dialect,
    driver: Arc<dyn ApiInferenceEngine>,
) -> Arc<dyn ApiInferenceEngine> {
    match dialect {
        // Splits the model reasoning block into reasoning_content and lifts its <tool_call> XML
        // into native tool calls (sparkflash_api::glm).
        Dialect::Glm => sparkflash_api::glm::GlmResponseEngine::layer(driver),
        Dialect::Qwen | Dialect::Dory => sparkflash_api::qwen::QwenResponseEngine::layer(driver),
        Dialect::Deepseek => driver,
        Dialect::Inkling => sparkflash_api::inkling::InklingResponseEngine::layer(driver),
    }
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
    serve_backend_world(backend, descriptor, seam, dialect, options, banner, 1)
}

/// Serves `backend` behind HTTP in a group of `world` ranks, under that group's numerics key.
///
/// The single-process path is [serve_backend] with `world = 1`; a four-rank device arm names its
/// own world so the boot line's numerics identity is the configuration that actually answered.
/// The driver is the typed route's ([pack::serve_backend]), with the operator's own output cap
/// and concurrency.
pub(crate) fn serve_backend_world(
    backend: Box<dyn ModelBackend>,
    descriptor: ModelDescriptor,
    seam: Arc<dyn PromptCodec>,
    dialect: Dialect,
    options: &ServeOptions,
    banner: &str,
    world: usize,
) -> Result<ExitCode, String> {
    pack::serve_backend(
        options,
        backend,
        descriptor,
        seam,
        options.max_tokens,
        world,
        Some(dialect),
        &format!("sparkflash serve: {banner}"),
        options.concurrency,
    )
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
    // The single-process device arm serves on the shared tick engine, so its seams are the
    // engine's: the lanes below are the device's, and the host's reference keeps its own.
    let mut options = options.clone();
    let backend: Box<dyn ModelBackend> = if device {
        let mut backend = sparkflash_model_dory::DoryDeviceBackend::open_with_descriptor(
            &checkpoint.directory,
            descriptor.clone(),
            context,
        )
        .map_err(|error| format!("--model {}: {error}", checkpoint.model_id))?;
        let lanes = dense_lanes(
            options.concurrency_set.then_some(options.concurrency),
            sparkflash_model::active_profile().runtime.concurrency,
        )?;
        backend
            .enable_fusion(lanes)
            .map_err(|error| format!("--model {}: {error}", checkpoint.model_id))?;
        options.concurrency = lanes;
        Box::new(backend)
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
    let reference = if device {
        format!("device arm on {} lanes", options.concurrency)
    } else {
        "host reference".to_owned()
    };
    let banner = format!(
        "{} ({}) at TP=1 on {}, context {context}, Dory {reference}",
        descriptor.id,
        descriptor.weights_id,
        checkpoint.directory.display(),
    );
    serve_backend(backend, descriptor, seam, dialect, &options, &banner)
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
    use super::{DenseDevice, dense_device, dense_lanes, device_family, profile_with};
    use sparkflash_model::Target;
    use sparkflash_ops::engine_plan::EnginePlan;

    #[test]
    fn inkling_refuses_tensor_core_decode_where_its_profile_is_read() {
        let architecture =
            sparkflash_model::Architecture::from_json(sparkflash_model_inkling2::carried_config())
                .expect("the carried Inkling config parses");
        let error = profile_with(&architecture, |name| match name {
            "SPARKFLASH_DECODE_MMA" => Some("on".to_owned()),
            _ => None,
        })
        .expect_err("Inkling reads decode.tensor_cores and cannot honour it");
        assert!(error.contains("Inkling"), "{error}");
        assert!(error.contains("SPARKFLASH_DECODE_MMA"), "{error}");
        assert!(error.contains("row GEMV"), "{error}");
        assert!(
            error.contains("docs/numerics/inkling2/v2-clean-head.md"),
            "{error}"
        );
        profile_with(&architecture, |_| None).expect("the carried Inkling profile boots");
    }

    #[test]
    fn the_dense_arms_lanes_are_the_flag_the_profile_or_four() {
        assert_eq!(dense_lanes(None, None), Ok(sparkflash_model::GROUP_LANES));
        assert_eq!(dense_lanes(Some(1), None), Ok(1));
        assert_eq!(dense_lanes(None, Some(8)), Ok(8));
        assert_eq!(dense_lanes(Some(8), Some(8)), Ok(8));
        assert!(dense_lanes(Some(2), Some(8)).is_err());
    }

    #[test]
    fn the_dense_device_switch_prefers_requires_or_refuses() {
        assert_eq!(dense_device(None), Ok(DenseDevice::Prefer));
        assert_eq!(dense_device(Some(" ")), Ok(DenseDevice::Prefer));
        assert_eq!(dense_device(Some("1")), Ok(DenseDevice::Require));
        assert_eq!(dense_device(Some("0")), Ok(DenseDevice::Off));
        assert_eq!(dense_device(Some("off")), Ok(DenseDevice::Off));
        // A value it does not know is refused, not read as either answer.
        let refused = dense_device(Some("maybe")).expect_err("an unknown value is refused");
        assert!(refused.contains("SPARKFLASH_DENSE_DEVICE"), "{refused}");
    }

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

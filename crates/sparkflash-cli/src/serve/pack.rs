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
use sparkflash_model::{ModelBackend, ModelDescriptor, PromptCodec};
use sparkflash_model_glm5_next::Glm5NextConfig;
use sparkflash_model_qwen4_exp::Qwen4ExpConfig;
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

/// What one pack's serving arm supplies; [`serve_host`] and [`serve_group`] are the rest.
pub(super) trait PackArm {
    /// The pack's parsed `config.json`.
    type Config;
    /// The pack's single-node device backend.
    type Backend: ModelBackend + 'static;

    /// The id `/v1/models` reports and the hub cache resolves.
    const MODEL_ID: &'static str;
    /// The weights id `--model` also takes.
    const WEIGHTS_ID: &'static str;
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
    ) -> Result<Self::Backend, String>;
    /// Pools `concurrency` sessions into one fused batch.
    fn fuse(backend: &mut Self::Backend, concurrency: usize) -> Result<(), String>;
    /// The pack's response dialect over the driver: its reasoning and tool channels.
    fn respond(driver: Arc<dyn ApiInferenceEngine>) -> Arc<dyn ApiInferenceEngine>;
}

/// Qwen3.8-Flash-Next, the NVFP4 checkpoint.
pub(super) struct QwenArm;

impl PackArm for QwenArm {
    type Config = Qwen4ExpConfig;
    type Backend = sparkflash_model_qwen4_exp::device::Qwen4ExpDeviceBackend;

    const MODEL_ID: &'static str = sparkflash_model_qwen4_exp::MODEL_ID;
    const WEIGHTS_ID: &'static str = sparkflash_model_qwen4_exp::WEIGHTS_ID;
    const NAME: &'static str = "Qwen3.8-Flash-Next";
    const FIXTURE: &'static str = "qwen4";
    const MAX_OUTPUT_TOKENS: usize = sparkflash_model_qwen4_exp::MAX_OUTPUT_TOKENS;
    const DEFAULT_CONTEXT: usize = sparkflash_model_qwen4_exp::DEFAULT_DEVICE_CONTEXT;
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
        // Each worker opens its own full-window KV set. Refuse a worker count whose caches alone
        // would exceed the documented budget before uploading 77.8 GiB of weights, rather than
        // failing one request at a time when the node runs out of memory.
        let budget = options.kv_budget_gib.map_or(
            sparkflash_model_qwen4_exp::CONCURRENCY_KV_BUDGET_BYTES,
            sparkflash_model_qwen4_exp::kv_budget_bytes,
        );
        let kv_bytes =
            sparkflash_model_qwen4_exp::concurrency_kv_bytes(context, options.concurrency);
        let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        eprintln!(
            "sparkflash serve: KV cache {:.1} GiB ({concurrency} x {context} tokens), budget {:.0} GiB",
            gib(kv_bytes),
            gib(budget),
            concurrency = options.concurrency,
        );
        if kv_bytes > budget {
            return Err(format!(
                "--concurrency {} at --context {context} needs {:.1} GiB of KV caches, over the {:.0} GiB budget; lower --concurrency or --context, or raise --kv-budget-gib on a node with more room",
                options.concurrency,
                gib(kv_bytes),
                gib(budget),
            ));
        }
        Ok(())
    }

    fn open_host(
        directory: &Path,
        _config: Self::Config,
        context: usize,
    ) -> Result<Self::Backend, String> {
        use sparkflash_model_qwen4_exp::device::Qwen4ExpDeviceBackend;
        // A converted rank file is one slab read instead of 77.8 GiB of per-tensor uploads. The
        // checkpoint directory stays the configuration, tokenizer and host-paged PLE source.
        let converted = converted_rank_file(QWEN_CONVERTED_DIR_ENV, 0, 1)?;
        let backend = match &converted {
            Some(path) => Qwen4ExpDeviceBackend::open_converted(directory, context, path),
            None => Qwen4ExpDeviceBackend::open(directory, context),
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

/// GLM-5.3-Flash, the FP8 checkpoint.
///
/// The release is 328.3 GB, so one node holds only the tiny fixture; the release is served by the
/// four-rank group.
pub(super) struct GlmArm;

impl PackArm for GlmArm {
    type Config = Glm5NextConfig;
    type Backend = sparkflash_model_glm5_next::Glm5NextDeviceBackend;

    const MODEL_ID: &'static str = sparkflash_model_glm5_next::MODEL_ID;
    const WEIGHTS_ID: &'static str = sparkflash_model_glm5_next::WEIGHTS_ID;
    const NAME: &'static str = "GLM-5.3-Flash";
    const FIXTURE: &'static str = "glm53";
    const MAX_OUTPUT_TOKENS: usize = sparkflash_model_glm5_next::MAX_OUTPUT_TOKENS;
    const DEFAULT_CONTEXT: usize = sparkflash_model_glm5_next::DEFAULT_DEVICE_CONTEXT;
    const FUSED_BATCH: bool = false;
    const KV_BUDGET: bool = false;
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
        _options: &ServeOptions,
        config: &Self::Config,
        _context: usize,
    ) -> Result<(), String> {
        // The FP8 release is 82 GB a rank over four nodes. One rank holding all four shards is
        // not the served shape, and the server says so before it tries a 328.3 GB upload.
        if config.num_hidden_layers >= 45 {
            return Err(format!(
                "{} is the released 321B-parameter FP8 checkpoint (328.3 GB): one TP=1 rank cannot hold it. Serve the tiny fixture at TP=1 (--synthetic), or serve the release with the four-rank group: build with --features tp4-serve and pass --tp4-rank/--tp4-world-size/--tp4-peers with --checkpoint.",
                Self::MODEL_ID
            ));
        }
        Ok(())
    }

    fn open_host(
        directory: &Path,
        config: Self::Config,
        context: usize,
    ) -> Result<Self::Backend, String> {
        let cuda = sparkflash_kernels::cuda_driver::Cuda::load()
            .map_err(|error| format!("CUDA: {error}"))?;
        sparkflash_model_glm5_next::Glm5NextDeviceBackend::open_single_rank(
            Arc::new(sparkflash_model_glm5_next::device::CudaHandle::new(cuda)),
            config,
            directory,
            context,
        )
        .map_err(|error| error.to_string())
    }

    fn fuse(_backend: &mut Self::Backend, _concurrency: usize) -> Result<(), String> {
        Err(format!("the {} arm has no fused batch", Self::NAME))
    }

    fn respond(driver: Arc<dyn ApiInferenceEngine>) -> Arc<dyn ApiInferenceEngine> {
        // The dialect splits the model's <think> block into reasoning_content; the tool-call XML
        // is still streamed as content until a GLM tool parser lands.
        sparkflash_api::glm::GlmResponseEngine::layer(driver)
    }
}

/// The packs `--model` can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Pack {
    Qwen,
    Glm,
}

impl Pack {
    /// The pack `model` names: its model id or its weights id, in any case.
    pub(super) fn named(model: &str) -> Option<Self> {
        let names = |id: &str, weights: &str| {
            model.eq_ignore_ascii_case(id) || model.eq_ignore_ascii_case(weights)
        };
        if names(QwenArm::MODEL_ID, QwenArm::WEIGHTS_ID) {
            Some(Self::Qwen)
        } else if names(GlmArm::MODEL_ID, GlmArm::WEIGHTS_ID) {
            Some(Self::Glm)
        } else {
            None
        }
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
            Self::Qwen => (QwenArm::FUSED_BATCH, QwenArm::KV_BUDGET),
            Self::Glm => (GlmArm::FUSED_BATCH, GlmArm::KV_BUDGET),
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
            Self::Qwen => serve_arm::<QwenArm>(options),
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

/// A pack's single-node arm: its device backend behind the generic host driver.
fn serve_host<A: PackArm>(options: &ServeOptions) -> Result<ExitCode, String> {
    let (directory, config) = host_checkpoint::<A>(options)?;
    let context = pack_context::<A>(options, &config)?;
    A::check_host(options, &config, context)?;
    let released_seam = if options.synthetic {
        None
    } else {
        Some(pack_seam::<A>(options, &directory)?.1)
    };
    let mut backend = A::open_host(&directory, config, context).map_err(|error| {
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

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

//! The Dory1.0-4B device arm.
//!
//! One rank, one CUDA context and two images: the Qwen token image supplies the BF16 GEMV and
//! embedding gather kernels; kernels/dory_sm121.cu supplies the tensor-core projection, the rope
//! with the per-head norm and ring-cache append, the nGPT-specific norms, attention epilogue, MLP
//! epilogue, residual lerp and logit scale.
//!
//! Every projection is BF16 with f32 activations, accumulated in f32. The recurrence and the KV
//! slots are exactly the host reference's: per-loop caches in the default mode, one overwritten
//! cache in last-loop mode. The arm refuses at load when a cubin, the context or a tensor shape
//! is missing; it never falls back to another arithmetic.
//!
//! A process runs one [DoryStage] of the looped walk. The whole model is the single-node
//! [DoryDeviceBackend]; a latent-recurrence group ([crate::recurrence]) gives each rank a run of
//! loops, rank 0 the embedding and the input prefix, and the last rank the output suffix and the
//! vocabulary head. A stage uploads and caches only its own blocks, and every stage runs the same
//! forward body, so a row's bits do not depend on where its loops ran.

use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_kernels::images::{dory_cubin, qwen4_exp_cubin};
use sparkflash_model::ModelDescriptor;
use sparkflash_model::backend::{BackendError, ModelBackend, ModelSession, Sampling, StepOutput};
use sparkflash_ops::gemv::CudaHandle;
use sparkflash_ops::sampler::Sampler;
use sparkflash_weights::safetensors::open_checkpoint;
use sparkflash_weights::{DType, WeightSource};

use crate::config::{Block, DoryConfig, KvCacheMode, Partition};
use crate::tensor::DoryTensorPlan;

/// The host-facing name of a device refusal.
const DEVICE: &str = "dory_device";

/// The number of prompt tokens one layer's elementwise and attention work sees at once.
///
/// Projections over this chunk run `dory_mma_wide`'s 128-token x 256-column tensor-core block;
/// chunks of TOKEN_ROWS or fewer rows keep the token GEMV, whose eight accumulators are the token
/// kernel's bits. A prefill of one token keeps the token-at-a-time path.
pub(crate) const PREFILL_BATCH: usize = 4096;

/// One uploaded BF16 matrix.
#[derive(Debug, Clone, Copy)]
struct DeviceMat {
    pointer: Deviceptr,
    rows: usize,
    columns: usize,
}

/// One attention sublayer's device weights.
#[derive(Debug, Clone, Copy)]
struct DeviceAttention {
    q: DeviceMat,
    k: DeviceMat,
    v: DeviceMat,
    g: DeviceMat,
    o: DeviceMat,
    sqk: Deviceptr,
    s_gate: Deviceptr,
    alpha: Deviceptr,
}

/// One MLP sublayer's device weights.
#[derive(Debug, Clone, Copy)]
struct DeviceMlp {
    gate: DeviceMat,
    up: DeviceMat,
    down: DeviceMat,
    suv_gate: Deviceptr,
    suv_up: Deviceptr,
    alpha: Deviceptr,
}

/// One physical block's device weights.
#[derive(Debug, Clone, Copy)]
struct DeviceBlock {
    attention: DeviceAttention,
    mlp: DeviceMlp,
}

/// The device-resident weights of one stage: the embedding only on the stage that runs the
/// prefix, the head only on the one that runs the suffix, and the blocks it runs.
#[derive(Debug, Clone)]
struct DeviceWeights {
    embed: Option<DeviceMat>,
    lm_head: Option<DeviceMat>,
    logit_scale: Option<Deviceptr>,
    /// Indexed by physical block; `None` for a block another stage runs.
    blocks: Vec<Option<DeviceBlock>>,
}

/// The slice of the looped walk one process runs.
///
/// The whole model is the input prefix, every loop and the output suffix. A latent-recurrence
/// group splits the loops into one contiguous run a rank: rank 0 also runs the embedding and the
/// prefix, the last rank the suffix and the vocabulary head. Each stage keeps the KV cache of its
/// own blocks and loops only, so a four-rank group of the four-loop checkpoint holds a quarter
/// of the recurrent caches a rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoryStage {
    /// The recurrent loops this stage runs.
    pub loops: Range<usize>,
    /// Whether this stage runs the embedding and the input prefix.
    pub prefix: bool,
    /// Whether this stage runs the output suffix and the vocabulary head.
    pub suffix: bool,
    /// Whether this stage also holds a copy of the output suffix and the head, with caches of
    /// its own, for a draft exit over the latent it receives ([StageCore::draft_lanes]): the
    /// token the model would pick after the loops before this stage's.
    pub draft: bool,
}

impl DoryStage {
    /// The whole model on one process.
    #[must_use]
    pub fn whole(config: &DoryConfig) -> Self {
        Self {
            loops: 0..config.n_recurrent_loops,
            prefix: true,
            suffix: true,
            draft: false,
        }
    }

    /// This stage with a draft exit.
    ///
    /// # Errors
    /// A stage with the prefix, which receives no latent to exit from, or with the suffix,
    /// whose own output blocks would share the draft's caches.
    pub fn drafting(self) -> Result<Self, String> {
        if self.prefix || self.suffix {
            return Err(format!(
                "a draft exit runs the output suffix on the latent a middle recurrence rank \
                 receives; the stage of loops {}..{} {}",
                self.loops.start,
                self.loops.end,
                if self.prefix {
                    "runs the prefix and receives no latent"
                } else {
                    "runs the real suffix, whose caches the draft would share"
                }
            ));
        }
        Ok(Self {
            draft: true,
            ..self
        })
    }

    /// Rank `rank` of a `world`-rank latent-recurrence group.
    ///
    /// Every rank runs the same number of loops, in rank order, so the four-loop checkpoint on
    /// four ranks is one recurrence a rank.
    ///
    /// # Errors
    /// A rank outside the world, a world that does not divide the checkpoint's loops, or a
    /// last-loop cache split across ranks: in that mode every loop overwrites one cache, so a
    /// token's first loop reads rows the final loop of the earlier tokens wrote on another rank.
    pub fn of_rank(config: &DoryConfig, rank: usize, world: usize) -> Result<Self, String> {
        let loops = config.n_recurrent_loops;
        if world == 0 || rank >= world {
            return Err(format!(
                "recurrence rank {rank} is outside a group of {world}"
            ));
        }
        if !loops.is_multiple_of(world) {
            return Err(format!(
                "a recurrence group of {world} rank(s) cannot split the checkpoint's {loops} \
                 loop(s) evenly: every rank runs the same number of loops (n_recurrent_loops \
                 in config.json)"
            ));
        }
        if world > 1 && config.recurrent_kv_cache_mode == KvCacheMode::LastLoop {
            return Err(
                "recurrent_kv_cache_mode last_loop keeps one cache that every loop overwrites, \
                 so a token's first loop reads rows the last loop wrote on another rank; a \
                 recurrence group needs per_loop caches"
                    .to_owned(),
            );
        }
        let each = loops / world;
        Ok(Self {
            loops: rank * each..(rank + 1) * each,
            prefix: rank == 0,
            suffix: rank + 1 == world,
            draft: false,
        })
    }

    /// Whether this stage holds `block`, for its own walk or its draft exit.
    fn runs(&self, block: &Block) -> bool {
        match block.partition {
            Partition::Input => self.prefix,
            Partition::Recurrent => !self.loops.is_empty(),
            Partition::Output => self.suffix || self.draft,
        }
    }

    /// Whether this stage holds the vocabulary head.
    fn has_head(&self) -> bool {
        self.suffix || self.draft
    }

    /// Whether this stage keeps `block`'s cache slot `slot`.
    fn keeps_slot(&self, block: &Block, slot: usize) -> bool {
        match block.partition {
            Partition::Recurrent if block.cache_slots > 1 => self.loops.contains(&slot),
            _ => self.runs(block),
        }
    }
}

/// The kernels this arm launches, loaded once.
struct Kernels {
    embed: Kernel,
    gemv_bf16: Kernel,
    gemv4_bf16: Kernel,
    mma_wide: Kernel,
    mma_wide4: Kernel,
    rope_norm: Kernel,
    rope_norm_lanes: Kernel,
    attn_split: Kernel,
    attn_combine: Kernel,
    attn_mma: Kernel,
    mlp_epilogue: Kernel,
    slerp: Kernel,
    loop_combine: Kernel,
    justnorm: Kernel,
    logit_scale: Kernel,
}

// SAFETY: a Kernel is an opaque driver function word, and the handle only names it. The context
// that owns the module lives in the shared DeviceShared, which outlives the session.
unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

pub(crate) struct DeviceShared {
    cuda: CudaHandle,
    kernels: Kernels,
    weights: DeviceWeights,
    config: DoryConfig,
    stage: DoryStage,
    context: usize,
    allocated_bytes: usize,
}

impl Drop for DeviceShared {
    fn drop(&mut self) {
        let pointers = self.weights.pointers();
        let references: Vec<&Deviceptr> = pointers.iter().collect();
        self.cuda.free_buffers(&references);
    }
}

/// The Dory model on the device.
pub struct DoryDeviceBackend {
    descriptor: ModelDescriptor,
    shared: Arc<DeviceShared>,
}

impl DoryDeviceBackend {
    /// Whether this build embeds every cubin the arm needs.
    #[must_use]
    pub fn available() -> bool {
        !dory_cubin().is_empty() && !qwen4_exp_cubin().is_empty()
    }

    /// Opens the checkpoint and uploads every tensor.
    ///
    /// # Errors
    /// A missing cubin, a CUDA context that cannot open, a tensor whose shape disagrees with the
    /// plan, or a context past the checkpoint window.
    pub fn open(directory: &Path, context: usize) -> Result<Self, String> {
        Self::open_with_descriptor(directory, crate::descriptor(), context)
    }

    /// Opens the checkpoint under the descriptor the serving path resolved.
    ///
    /// # Errors
    /// As [Self::open].
    pub fn open_with_descriptor(
        directory: &Path,
        mut descriptor: sparkflash_model::ModelDescriptor,
        context: usize,
    ) -> Result<Self, String> {
        let shared = DeviceShared::open(directory, context, |config| Ok(DoryStage::whole(config)))?;
        descriptor.context_tokens = context;
        Ok(Self {
            descriptor,
            shared: Arc::new(shared),
        })
    }

    /// The device bytes the weights claimed, for boot diagnostics.
    #[must_use]
    pub fn weight_bytes(&self) -> usize {
        self.shared.allocated_bytes
    }
}

/// Refuses a typed profile whose values this arm cannot apply.
///
/// The device arm's prompt chunk, tensor-core selection, cache form and one-process
/// speculation are fixed by its kernels. Reading the installed profile here is what keeps an
/// environment override or a hand-edited TOML from being silently ignored: a field this arm
/// does not honour is refused by name rather than accepted and dropped.
pub(crate) fn check_profile() -> Result<(), String> {
    profile_refusal(sparkflash_model::profile::active())
}

/// The field-by-field rule [check_profile] applies, without reading the process-wide profile.
fn profile_refusal(profile: &sparkflash_model::profile::ModelProfile) -> Result<(), String> {
    if let Some(chunk) = profile.prefill.chunk
        && chunk != PREFILL_BATCH
    {
        return Err(format!(
            "the Dory device arm fixes the prefill chunk at {PREFILL_BATCH} rows, not {chunk}; \
             change prefill.chunk back, or serve without the device arm"
        ));
    }
    if profile.prefill.enabled == Some(false) {
        return Err(
            "the Dory device arm always runs its prompt through prefill, but the profile sets \
             prefill.enabled = false"
                .to_owned(),
        );
    }
    if profile.prefill.tensor_cores == Some(false) {
        return Err(
            "the Dory device arm's prompt projections run on the tensor cores, but the profile \
             disables prefill.tensor_cores"
                .to_owned(),
        );
    }
    if profile.prefill.exact == Some(true) {
        return Err(
            "the Dory device arm's batched prefill and token-at-a-time decode use different \
             attention kernels and are not bit-identical, but the profile sets prefill.exact = \
             true"
                .to_owned(),
        );
    }
    if profile.decode.tensor_cores == Some(true) {
        return Err(
            "the Dory device arm's decode attention is a warp-dot/FMA kernel, not a tensor-core \
             kernel, but the profile sets decode.tensor_cores = true"
                .to_owned(),
        );
    }
    if profile.speculation.enabled == Some(true) {
        return Err(
            "Dory's loop-1 drafting is selected by --recurrence-draft-lanes on a recurrence \
             group, not by speculation.enabled; leave the profile's speculation disabled"
                .to_owned(),
        );
    }
    if profile.sequence.parallel == Some(true) || profile.sequence.overlap == Some(true) {
        return Err(
            "the Dory device arm does not implement sequence parallelism or overlap; leave \
             sequence.parallel and sequence.overlap false"
                .to_owned(),
        );
    }
    if let Some(concurrency) = profile.runtime.concurrency
        && concurrency > 1
    {
        return Err(format!(
            "the Dory device arm's process holds one sequence; use --concurrency on a \
             recurrence group, not runtime.concurrency = {concurrency}"
        ));
    }
    if matches!(
        profile.attention.latent_cache,
        Some(
            sparkflash_model::profile::LatentCacheMode::F32
                | sparkflash_model::profile::LatentCacheMode::Fp8
        )
    ) {
        return Err(
            "the Dory device arm keeps its own BF16 KV caches; the profile's \
             attention.latent_cache does not select their form"
                .to_owned(),
        );
    }
    Ok(())
}

impl DeviceShared {
    /// Opens the checkpoint and uploads the tensors of the stage `stage` picks for its config.
    pub(crate) fn open(
        directory: &Path,
        context: usize,
        stage: impl FnOnce(&DoryConfig) -> Result<DoryStage, String>,
    ) -> Result<Self, String> {
        if !DoryDeviceBackend::available() {
            return Err(
                "this build has no Dory/Qwen cubin: rebuild on a host with nvcc (or set \
                 SPARKFLASH_NVCC) so build.rs compiles kernels/dory_sm121.cu and \
                 kernels/qwen4_exp_sm121.cu"
                    .to_owned(),
            );
        }
        let config = DoryConfig::from_checkpoint_dir(directory)
            .map_err(|error| format!("{}: {error}", directory.join("config.json").display()))?;
        if context == 0 || context > config.max_position_embeddings {
            return Err(format!(
                "context {context} is outside 1..={} for the checkpoint",
                config.max_position_embeddings
            ));
        }
        check_profile()?;
        attention_fits(&config)?;
        let stage = stage(&config)?;
        let cuda = CudaHandle::new(Cuda::load()?);
        // Stage copies use copy_dtod, which is queued only when forward async launches are on.
        // Without this switch every prefix/loop/suffix state copy is a blocking D2D copy in the
        // serving loop. Qwen enables the same switch at its open.
        sparkflash_kernels::cuda_driver::enable_forward_async_launches();
        let kernels = Kernels::load(&cuda, config.head_dim)?;
        let source = open_checkpoint(directory).map_err(|error| error.to_string())?;
        let mut total = 0usize;
        let weights = DeviceWeights::upload(&cuda, &config, &stage, &source, &mut total)?;
        Ok(Self {
            cuda,
            kernels,
            weights,
            config,
            stage,
            context,
            allocated_bytes: total,
        })
    }

    pub(crate) fn cuda(&self) -> &Cuda {
        &self.cuda
    }

    pub(crate) fn config(&self) -> &DoryConfig {
        &self.config
    }

    pub(crate) fn stage(&self) -> &DoryStage {
        &self.stage
    }

    pub(crate) fn context(&self) -> usize {
        self.context
    }

    pub(crate) fn weight_bytes(&self) -> usize {
        self.allocated_bytes
    }
}

impl Kernels {
    /// Every kernel, with the prefill attention instance for `head_dim` (one of
    /// PREFILL_HEAD_DIMS; attention_fits refuses the rest first).
    fn load(cuda: &Cuda, head_dim: usize) -> Result<Self, String> {
        let named = |image: &'static [u8], name: &'static str| {
            cuda.load_named(image, name).map(|(_, kernel)| kernel)
        };
        let attn_mma = match head_dim {
            16 => "dory_attn_mma16",
            64 => "dory_attn_mma64",
            128 => "dory_attn_mma128",
            256 => "dory_attn_mma256",
            other => return Err(format!("no dory_attn_mma instance for head_dim {other}")),
        };
        let shared = u32::try_from(prefill_attention_shared(head_dim))
            .map_err(|_| format!("{attn_mma}: shared memory past u32"))?;
        Ok(Self {
            embed: named(qwen4_exp_cubin(), "q4e_embed")?,
            gemv_bf16: named(qwen4_exp_cubin(), "q4e_gemv_bf16")?,
            gemv4_bf16: named(qwen4_exp_cubin(), "q4e_gemv4_bf16")?,
            mma_wide: cuda
                .load_named_shared(dory_cubin(), "dory_mma_wide", WIDE_SHARED)
                .map(|(_, kernel)| kernel)?,
            mma_wide4: cuda
                .load_named_shared(dory_cubin(), "dory_mma_wide4", WIDE_SHARED)
                .map(|(_, kernel)| kernel)?,
            rope_norm: named(dory_cubin(), "dory_rope_norm")?,
            rope_norm_lanes: named(dory_cubin(), "dory_rope_norm_lanes")?,
            attn_split: named(dory_cubin(), "dory_attn_split")?,
            attn_combine: named(dory_cubin(), "dory_attn_combine")?,
            attn_mma: cuda
                .load_named_shared(dory_cubin(), attn_mma, shared)
                .map(|(_, kernel)| kernel)?,
            mlp_epilogue: named(dory_cubin(), "dory_mlp_epilogue")?,
            slerp: named(dory_cubin(), "dory_slerp")?,
            loop_combine: named(dory_cubin(), "dory_loop_combine")?,
            justnorm: named(dory_cubin(), "dory_justnorm")?,
            logit_scale: named(dory_cubin(), "dory_logit_scale")?,
        })
    }
}

impl DeviceWeights {
    fn pointers(&self) -> Vec<Deviceptr> {
        let mut pointers: Vec<Deviceptr> = [
            self.embed.map(|embed| embed.pointer),
            self.lm_head.map(|head| head.pointer),
            self.logit_scale,
        ]
        .into_iter()
        .flatten()
        .collect();
        for block in self.blocks.iter().flatten() {
            pointers.extend([
                block.attention.q.pointer,
                block.attention.k.pointer,
                block.attention.v.pointer,
                block.attention.g.pointer,
                block.attention.o.pointer,
                block.attention.sqk,
                block.attention.s_gate,
                block.attention.alpha,
                block.mlp.gate.pointer,
                block.mlp.up.pointer,
                block.mlp.down.pointer,
                block.mlp.suv_gate,
                block.mlp.suv_up,
                block.mlp.alpha,
            ]);
        }
        pointers
    }

    fn upload(
        cuda: &Cuda,
        config: &DoryConfig,
        stage: &DoryStage,
        source: &dyn WeightSource,
        total: &mut usize,
    ) -> Result<Self, String> {
        let hidden = config.hidden_size;
        let query_width = config.num_attention_heads * config.head_dim;
        let kv_width = config.num_key_value_heads * config.head_dim;
        let intermediate = config.intermediate_size;
        DoryTensorPlan::served(config)
            .conformance(source.names())
            .map_err(|error| format!("the checkpoint does not match the Dory plan: {error}"))?;
        let embed = if stage.prefix {
            Some(upload_matrix(
                cuda,
                source,
                "backbone.embeddings.weight",
                config.vocab_size,
                hidden,
                total,
            )?)
        } else {
            None
        };
        let (lm_head, logit_scale) = if stage.has_head() {
            (
                Some(upload_matrix(
                    cuda,
                    source,
                    "lm_head.weight",
                    config.vocab_size,
                    hidden,
                    total,
                )?),
                Some(upload_vector(
                    cuda,
                    source,
                    "lm_head.logit_scale",
                    config.vocab_size,
                    total,
                )?),
            )
        } else {
            (None, None)
        };
        let mut blocks = Vec::with_capacity(config.num_blocks());
        for block in &config.blocks {
            if !stage.runs(block) {
                blocks.push(None);
                continue;
            }
            let attention_prefix = format!("backbone.layers.{}.mixer", block.attention_layer);
            let mlp_prefix = format!("backbone.layers.{}.mixer", block.mlp_layer);
            let attention = DeviceAttention {
                alpha: upload_vector(
                    cuda,
                    source,
                    &format!("backbone.layers.{}.attn_alpha", block.attention_layer),
                    hidden,
                    total,
                )?,
                q: upload_matrix(
                    cuda,
                    source,
                    &format!("{attention_prefix}.q_proj.weight"),
                    query_width,
                    hidden,
                    total,
                )?,
                k: upload_matrix(
                    cuda,
                    source,
                    &format!("{attention_prefix}.k_proj.weight"),
                    kv_width,
                    hidden,
                    total,
                )?,
                v: upload_matrix(
                    cuda,
                    source,
                    &format!("{attention_prefix}.v_proj.weight"),
                    kv_width,
                    hidden,
                    total,
                )?,
                g: upload_matrix(
                    cuda,
                    source,
                    &format!("{attention_prefix}.g_proj.weight"),
                    query_width,
                    hidden,
                    total,
                )?,
                o: upload_matrix(
                    cuda,
                    source,
                    &format!("{attention_prefix}.o_proj.weight"),
                    hidden,
                    query_width,
                    total,
                )?,
                sqk: upload_vector(
                    cuda,
                    source,
                    &format!("{attention_prefix}.sqk"),
                    query_width,
                    total,
                )?,
                s_gate: upload_vector(
                    cuda,
                    source,
                    &format!("{attention_prefix}.s_gate"),
                    query_width,
                    total,
                )?,
            };
            let mlp = DeviceMlp {
                alpha: upload_vector(
                    cuda,
                    source,
                    &format!("backbone.layers.{}.mlp_alpha", block.mlp_layer),
                    hidden,
                    total,
                )?,
                gate: upload_matrix(
                    cuda,
                    source,
                    &format!("{mlp_prefix}.gate_proj.weight"),
                    intermediate,
                    hidden,
                    total,
                )?,
                up: upload_matrix(
                    cuda,
                    source,
                    &format!("{mlp_prefix}.up_proj.weight"),
                    intermediate,
                    hidden,
                    total,
                )?,
                down: upload_matrix(
                    cuda,
                    source,
                    &format!("{mlp_prefix}.down_proj.weight"),
                    hidden,
                    intermediate,
                    total,
                )?,
                suv_gate: upload_vector(
                    cuda,
                    source,
                    &format!("{mlp_prefix}.suv_gate"),
                    intermediate,
                    total,
                )?,
                suv_up: upload_vector(
                    cuda,
                    source,
                    &format!("{mlp_prefix}.suv_up"),
                    intermediate,
                    total,
                )?,
            };
            blocks.push(Some(DeviceBlock { attention, mlp }));
        }
        Ok(Self {
            embed,
            lm_head,
            logit_scale,
            blocks,
        })
    }
}

fn upload_matrix(
    cuda: &Cuda,
    source: &dyn WeightSource,
    name: &str,
    rows: usize,
    columns: usize,
    total: &mut usize,
) -> Result<DeviceMat, String> {
    let spec = source.spec(name).map_err(|error| error.to_string())?;
    if spec.dtype != DType::BF16 || spec.shape != [rows, columns] {
        return Err(format!(
            "{name} is {:?} {:?}, expected BF16 [{rows}, {columns}]",
            spec.dtype, spec.shape
        ));
    }
    let bytes = source.read(name).map_err(|error| error.to_string())?;
    if bytes.len() != rows * columns * 2 {
        return Err(format!(
            "{name} decoded to {} bytes, expected {}",
            bytes.len(),
            rows * columns * 2
        ));
    }
    let pointer = cuda.alloc(bytes.len(), total)?;
    cuda.copy_htod(&pointer, &bytes)?;
    Ok(DeviceMat {
        pointer,
        rows,
        columns,
    })
}

fn upload_vector(
    cuda: &Cuda,
    source: &dyn WeightSource,
    name: &str,
    width: usize,
    total: &mut usize,
) -> Result<Deviceptr, String> {
    let spec = source.spec(name).map_err(|error| error.to_string())?;
    if spec.dtype != DType::BF16 || spec.shape != [width] {
        return Err(format!(
            "{name} is {:?} {:?}, expected BF16 [{width}]",
            spec.dtype, spec.shape
        ));
    }
    let bytes = source.read(name).map_err(|error| error.to_string())?;
    if bytes.len() != width * 2 {
        return Err(format!(
            "{name} decoded to {} bytes, expected {}",
            bytes.len(),
            width * 2
        ));
    }
    let pointer = cuda.alloc(bytes.len(), total)?;
    cuda.copy_htod(&pointer, &bytes)?;
    Ok(pointer)
}

impl ModelBackend for DoryDeviceBackend {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        let core = StageCore::open(Arc::clone(&self.shared))
            .map_err(|error| BackendError::new(DEVICE, error))?;
        Ok(Box::new(DoryDeviceSession {
            core,
            sampler: Sampler::new(),
            sampling: Sampling::default(),
        }))
    }
}

/// Per-session f32 scratch, sized for the checkpoint geometry.
/// The fewest keys one dory_attn_split block reads (the kernel's split run).
const SPLIT_KEYS: usize = 64;
/// The most runs dory_attn_split cuts a row's keys into; longer rows get longer runs.
const SPLIT_MAX: usize = 256;
/// dory_attn_split's lane layout: 8 head dimensions a lane, one warp a key.
const SPLIT_HEAD_MAX: usize = 256;
/// The query heads a dory_attn_split block serves from one kv head's rows.
const SPLIT_GQA_MAX: usize = 4;

/// dory_mma_wide's output columns a block (GLM_WIDE_BN) and its dynamic shared memory
/// (GLM_WIDE_SHARED): three cp.async stages of a 128-row f32 activation tile and a 256-row BF16
/// weight tile, 32 K each.
const WIDE_COLUMNS: usize = 256;
const WIDE_SHARED: u32 = 98_304;

/// The head widths dory_attn_mma is instantiated for.
const PREFILL_HEAD_DIMS: [usize; 4] = [16, 64, 128, 256];
/// dory_attn_mma's query positions a block and keys a shared tile (DORY_MMA_Q, DORY_MMA_KEYS).
const PREFILL_Q: usize = 16;
const PREFILL_KEYS: usize = 32;

/// dory_attn_mma's dynamic shared memory at `head_dim`: q's BF16 head and remainder for a full
/// group's rows, then one key tile and one value tile.
fn prefill_attention_shared(head_dim: usize) -> usize {
    (2 * SPLIT_GQA_MAX * PREFILL_Q + 2 * PREFILL_KEYS) * head_dim * 2
}

/// The attention kernels' shape limits, checked before anything is uploaded.
fn attention_fits(config: &DoryConfig) -> Result<(), String> {
    let heads = config.num_attention_heads;
    let kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    if kv_heads == 0 || !heads.is_multiple_of(kv_heads) || heads / kv_heads > SPLIT_GQA_MAX {
        return Err(format!(
            "the device arm's decode attention serves 1..={SPLIT_GQA_MAX} query heads a kv head; \
             this checkpoint has {heads} over {kv_heads}"
        ));
    }
    if !head_dim.is_multiple_of(8) || head_dim > SPLIT_HEAD_MAX {
        return Err(format!(
            "the device arm's decode attention needs head_dim a multiple of 8 up to \
             {SPLIT_HEAD_MAX}; this checkpoint has {head_dim}"
        ));
    }
    if !PREFILL_HEAD_DIMS.contains(&head_dim) {
        return Err(format!(
            "the device arm's prefill attention is built for head_dim {PREFILL_HEAD_DIMS:?}; \
             this checkpoint has {head_dim}"
        ));
    }
    // The wide projection kernel is a 32-column-step MMA. A width that is not a multiple of 32
    // would force the token GEMV over the chunk, whose association and rounding differ, so the
    // arm refuses the checkpoint at load instead of choosing a different arithmetic per launch.
    let q_width = heads
        .checked_mul(head_dim)
        .ok_or_else(|| "the query projection width overflows usize".to_owned())?;
    let kv_width = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "the key/value projection width overflows usize".to_owned())?;
    for (name, width) in [
        ("hidden_size", config.hidden_size),
        ("intermediate_size", config.intermediate_size),
        ("vocab_size", config.vocab_size),
        ("num_attention_heads * head_dim", q_width),
        ("num_key_value_heads * head_dim", kv_width),
    ] {
        if !width.is_multiple_of(32) {
            return Err(format!(
                "the device arm's wide projections need a multiple of 32, but {name} is {width}; \
                 the token GEMV would change a chunk's arithmetic"
            ));
        }
    }
    Ok(())
}

/// The lanes one dory_rope_norm_lanes, dory_attn_split or dory_attn_combine launch serves
/// (dory_sm121.cu's DORY_LANES): each lane's word and cache are launch parameters of their own.
const ATTENTION_LANES: usize = 8;
/// A lane word's position for a lane that sits a pass out (DORY_LANE_IDLE).
const LANE_IDLE: u64 = 0xffff_ffff;

/// dory_attn_split's partials for one lane, in floats: a record a query head and run.
fn lane_partials(config: &DoryConfig) -> usize {
    config.num_attention_heads * SPLIT_MAX * (config.head_dim + 2)
}

/// A decode lane's launch word: its position in the low half, dory_attn_split's run length in the
/// high half.
fn lane_word(position: usize, span: usize) -> Result<u64, BackendError> {
    match (u32::try_from(position), u32::try_from(span)) {
        (Ok(position), Ok(span)) if u64::from(position) != LANE_IDLE => {
            Ok(u64::from(position) | u64::from(span) << 32)
        }
        _ => Err(BackendError::new(
            DEVICE,
            format!("position {position} or run {span} does not fit a decode lane's word"),
        )),
    }
}

/// How dory_attn_split cuts the keys of the row at `position`: (first key, run length, runs).
fn decode_split(position: usize, window: Option<usize>) -> (usize, usize, usize) {
    // The kernel reads a window of 0 as full attention.
    let first = window
        .filter(|window| *window > 0)
        .map_or(0, |window| (position + 1).saturating_sub(window));
    let keys = position + 1 - first;
    let span = SPLIT_KEYS.max(keys.div_ceil(SPLIT_MAX));
    (first, span, keys.div_ceil(span))
}

struct Scratch {
    hidden: Deviceptr,
    input_h: Deviceptr,
    state: Deviceptr,
    q: Deviceptr,
    k: Deviceptr,
    v: Deviceptr,
    gate: Deviceptr,
    attention: Deviceptr,
    branch: Deviceptr,
    mlp_gate: Deviceptr,
    mlp_up: Deviceptr,
    mlp_hidden: Deviceptr,
    logits: Deviceptr,
    /// dory_attn_split's partials: a (max, sum, head_dim sums) record a query head and run.
    partials: Deviceptr,
    /// The received latent rows a draft exit keeps while its suffix uses `state`, on a stage
    /// with one ([DoryStage::draft]).
    latent: Option<Deviceptr>,
}

impl Scratch {
    fn pointers(&self) -> Vec<Deviceptr> {
        let mut pointers = vec![
            self.hidden,
            self.input_h,
            self.state,
            self.q,
            self.k,
            self.v,
            self.gate,
            self.attention,
            self.branch,
            self.mlp_gate,
            self.mlp_up,
            self.mlp_hidden,
            self.logits,
            self.partials,
        ];
        pointers.extend(self.latent);
        pointers
    }

    fn allocate(shared: &DeviceShared) -> Result<Self, String> {
        let mut total = 0usize;
        let mut alloc = |elements: usize| {
            shared
                .cuda
                .alloc(elements * 4, &mut total)
                .map_err(|error| format!("scratch: {error}"))
        };
        let config = &shared.config;
        let batch = PREFILL_BATCH;
        Ok(Self {
            hidden: alloc(config.hidden_size * batch)?,
            input_h: alloc(config.hidden_size * batch)?,
            state: alloc(config.hidden_size * batch)?,
            q: alloc(config.num_attention_heads * config.head_dim * batch)?,
            k: alloc(config.num_key_value_heads * config.head_dim * batch)?,
            v: alloc(config.num_key_value_heads * config.head_dim * batch)?,
            gate: alloc(config.num_attention_heads * config.head_dim * batch)?,
            attention: alloc(config.num_attention_heads * config.head_dim * batch)?,
            branch: alloc(config.hidden_size.max(config.intermediate_size) * batch)?,
            mlp_gate: alloc(config.intermediate_size * batch)?,
            mlp_up: alloc(config.intermediate_size * batch)?,
            mlp_hidden: alloc(config.intermediate_size * batch)?,
            logits: alloc(config.vocab_size * LANE_BATCH)?,
            partials: alloc(LANE_BATCH * lane_partials(config))?,
            latent: if shared.stage.draft {
                Some(alloc(config.hidden_size * batch)?)
            } else {
                None
            },
        })
    }
}

/// The rows a block's cache holds: the whole context for full attention, the window plus one
/// prefill chunk for a sliding ring (never more than the context).
fn cache_rows(block: &Block, context: usize) -> usize {
    block.window.map_or(context, |window| {
        window.saturating_add(PREFILL_BATCH).min(context)
    })
}

/// One cache per attention block and loop slot the stage runs, BF16; `None` for a slot another
/// stage keeps.
///
/// A slot holds K and V back to back, so it is twice one BF16 [rows, width] plane.
fn allocate_caches(shared: &DeviceShared) -> Result<Vec<Vec<Option<Deviceptr>>>, String> {
    let width = shared.config.num_key_value_heads * shared.config.head_dim;
    let mut total = 0usize;
    let mut caches = Vec::with_capacity(shared.config.blocks.len());
    for block in &shared.config.blocks {
        // A sliding block never reads a row older than its window, so its cache is a ring of
        // that many rows rather than the whole context. A full-attention block keeps every
        // row.
        let bytes = cache_rows(block, shared.context) * width * 2 * 2;
        let mut slots = Vec::with_capacity(block.cache_slots);
        for slot in 0..block.cache_slots {
            if !shared.stage.keeps_slot(block, slot) {
                slots.push(None);
                continue;
            }
            match shared.cuda.alloc(bytes, &mut total) {
                Ok(pointer) => slots.push(Some(pointer)),
                Err(error) => {
                    free_caches(shared, &caches);
                    free_caches(shared, &[slots]);
                    return Err(format!("KV cache: {error}"));
                }
            }
        }
        caches.push(slots);
    }
    Ok(caches)
}

fn free_caches(shared: &DeviceShared, caches: &[Vec<Option<Deviceptr>>]) {
    let pointers: Vec<Deviceptr> = caches.iter().flatten().flatten().copied().collect();
    let references: Vec<&Deviceptr> = pointers.iter().collect();
    shared.cuda.free_buffers(&references);
}

/// A block or cache slot the stage does not hold: a schedule bug, never a request error.
fn stage_refusal(block_index: usize, loop_index: usize) -> BackendError {
    BackendError::new(
        DEVICE,
        format!("block {block_index} loop {loop_index} belongs to another recurrence stage"),
    )
}

/// One stage's sequence: its scratch, its caches and the next position.
///
/// The forward body is [StageCore::forward]: the prefix (when the stage has it) over the rows
/// [StageCore::embed] wrote, the stage's loops over `state` and `input_h`, and the suffix and
/// head (when the stage has them). A stage without the prefix finds its rows in `state` and
/// `input_h` ([StageCore::carry]); a stage without the suffix leaves its rows there for the next.
pub(crate) struct StageCore {
    shared: Arc<DeviceShared>,
    scratch: Arc<OwnedScratch>,
    caches: Vec<Vec<Option<Deviceptr>>>,
    position: usize,
}

/// Scratch that frees itself: one sequence's, or the one set a rank's sequences share.
struct OwnedScratch {
    shared: Arc<DeviceShared>,
    buffers: Scratch,
}

impl Drop for OwnedScratch {
    fn drop(&mut self) {
        let pointers = self.buffers.pointers();
        let references: Vec<&Deviceptr> = pointers.iter().collect();
        self.shared.cuda.free_buffers(&references);
    }
}

impl std::ops::Deref for OwnedScratch {
    type Target = Scratch;

    fn deref(&self) -> &Scratch {
        &self.buffers
    }
}

impl Drop for StageCore {
    fn drop(&mut self) {
        free_caches(&self.shared, &self.caches);
    }
}

impl StageCore {
    pub(crate) fn open(shared: Arc<DeviceShared>) -> Result<Self, String> {
        shared.cuda.make_current()?;
        let scratch = Arc::new(OwnedScratch {
            buffers: Scratch::allocate(&shared)?,
            shared: Arc::clone(&shared),
        });
        Self::with_scratch(shared, scratch)
    }

    /// `count` sequences over one scratch set, each with its own caches and position.
    ///
    /// Only for sequences whose steps are queued on one stream, one step at a time: a step reads
    /// its rows from the scratch the step before it left, so the shared set holds no sequence's
    /// state between steps except the carried rows ([StageCore::carry]), which every transfer that
    /// reads them has consumed, in stream order, before the next step writes them.
    pub(crate) fn lanes(shared: &Arc<DeviceShared>, count: usize) -> Result<Vec<Self>, String> {
        shared.cuda.make_current()?;
        let scratch = Arc::new(OwnedScratch {
            buffers: Scratch::allocate(shared)?,
            shared: Arc::clone(shared),
        });
        (0..count)
            .map(|_| Self::with_scratch(Arc::clone(shared), Arc::clone(&scratch)))
            .collect()
    }

    fn with_scratch(shared: Arc<DeviceShared>, scratch: Arc<OwnedScratch>) -> Result<Self, String> {
        let caches = allocate_caches(&shared)?;
        Ok(Self {
            shared,
            scratch,
            caches,
            position: 0,
        })
    }

    pub(crate) fn shared(&self) -> &Arc<DeviceShared> {
        &self.shared
    }

    /// The next position this sequence writes.
    pub(crate) fn position(&self) -> usize {
        self.position
    }

    /// Starts a new sequence: the caches are overwritten from position 0.
    pub(crate) fn restart(&mut self) {
        self.position = 0;
    }

    /// The `(state, input_h)` rows a stage hands the next one: row-major f32, `hidden` wide.
    pub(crate) fn carry(&self) -> (Deviceptr, Deviceptr) {
        (self.scratch.state, self.scratch.input_h)
    }

    /// Row `row` of the vocabulary rows the head leaves on the device: [StageCore::forward]'s one
    /// row is row 0, and [StageCore::forward_lanes] leaves one a lane.
    pub(crate) fn logits_row(&self, row: usize) -> Deviceptr {
        self.scratch
            .logits
            .offset(row * self.shared.config.vocab_size * 4)
    }

    /// Refuses `rows` rows at the current position when they are not one chunk of the context.
    pub(crate) fn check_rows(&self, rows: usize) -> Result<(), BackendError> {
        if rows == 0 || rows > PREFILL_BATCH {
            return Err(BackendError::new(
                DEVICE,
                format!("a chunk of {rows} rows is outside 1..={PREFILL_BATCH}"),
            ));
        }
        if self.position + rows > self.shared.context {
            return Err(BackendError::new(
                DEVICE,
                format!(
                    "the sequence is past the served context {}",
                    self.shared.context
                ),
            ));
        }
        Ok(())
    }

    /// Gathers the embedding rows of `tokens` into `hidden`, on the stage that runs the prefix.
    pub(crate) fn embed(&self, tokens: &[u32]) -> Result<(), BackendError> {
        self.check_rows(tokens.len())?;
        embed_rows(&self.shared, &self.scratch, tokens)
    }

    /// Gathers `tokens[i]`'s embedding into row `i`, the row [StageCore::forward_lanes] runs for
    /// `lanes[i]`.
    pub(crate) fn embed_lanes(
        lanes: &[&mut StageCore],
        tokens: &[u32],
    ) -> Result<(), BackendError> {
        let scratch = batch_scratch(lanes, tokens.len())?;
        for lane in lanes {
            lane.check_rows(1)?;
        }
        embed_rows(&lanes[0].shared, &scratch, tokens)
    }

    /// Runs this stage's part of the walk over `rows` rows at the current position, and the head
    /// over the last row when `want_logits` (only on the stage with the suffix).
    ///
    /// One row takes the token kernels and a chunk the batched ones, on every stage alike, so a
    /// group's rows are the single process's bits. Only the launches are queued; the logits, when
    /// asked for, are left in [StageCore::logits_row] 0.
    pub(crate) fn forward(&mut self, rows: usize, want_logits: bool) -> Result<(), BackendError> {
        self.check_rows(rows)?;
        let position = self.position;
        walk(
            &self.shared,
            &self.scratch,
            rows,
            want_logits.then(|| rows - 1..rows),
            Pass::Stage,
            |block_index, block, loop_index| {
                self.run_block(block_index, block, loop_index, position, rows)
            },
        )?;
        self.position += rows;
        Ok(())
    }

    /// The draft exit without a guess, on a stage with one, split around the stage's own pass:
    /// [StageCore::keep] copies `rows` received latent rows aside before [StageCore::forward]
    /// overwrites `state`, and [StageCore::draft_kept] runs the output suffix over them, on
    /// caches of its own, once the rank has sent its rows on, so the next rank waits only on this
    /// stage's loops. The draft's caches get the rows at the positions the forward has since
    /// moved past, as [StageCore::draft_lanes] writes them; `state` is scratch afterwards.
    pub(crate) fn keep(&self, rows: usize) -> Result<(), BackendError> {
        self.check_rows(rows)?;
        keep_latent(&self.shared, &self.scratch, rows).map(drop)
    }

    /// The draft exit over the `rows` rows [StageCore::keep] kept, which the last
    /// [StageCore::forward] has run: see [StageCore::keep].
    pub(crate) fn draft_kept(&mut self, rows: usize) -> Result<(), BackendError> {
        let position = self.position.checked_sub(rows).ok_or_else(|| {
            BackendError::new(
                DEVICE,
                format!("{rows} kept rows at position {}", self.position),
            )
        })?;
        let latent = kept_latent(&self.scratch)?;
        restore_latent(&self.shared, &self.scratch, latent, rows)?;
        self.position = position;
        let walked = walk(
            &self.shared,
            &self.scratch,
            rows,
            None,
            Pass::Draft,
            |block_index, block, loop_index| {
                self.run_block(block_index, block, loop_index, position, rows)
            },
        );
        self.position = position + rows;
        walked
    }

    /// [StageCore::keep] for one decode row of each of `lanes`, before
    /// [StageCore::forward_lanes].
    pub(crate) fn keep_lanes(
        lanes: &[&mut StageCore],
        active: &[bool],
    ) -> Result<(), BackendError> {
        let scratch = batch_scratch(lanes, active.len())?;
        for (lane, _) in lanes.iter().zip(active).filter(|(_, active)| **active) {
            lane.check_rows(1)?;
        }
        keep_latent(&lanes[0].shared, &scratch, lanes.len()).map(drop)
    }

    /// [StageCore::draft_kept] for the rows [StageCore::keep_lanes] kept, after
    /// [StageCore::forward_lanes] has moved each active lane one row on.
    pub(crate) fn draft_kept_lanes(
        lanes: &mut [&mut StageCore],
        active: &[bool],
    ) -> Result<(), BackendError> {
        let scratch = batch_scratch(lanes, active.len())?;
        if lanes
            .iter()
            .zip(active)
            .any(|(lane, active)| *active && lane.position == 0)
        {
            return Err(BackendError::new(
                DEVICE,
                "a kept row of a lane that has not moved",
            ));
        }
        let shared = Arc::clone(&lanes[0].shared);
        let count = lanes.len();
        restore_latent(&shared, &scratch, kept_latent(&scratch)?, count)?;
        let moved = |lanes: &mut [&mut StageCore], step: fn(usize) -> usize| {
            for (lane, _) in lanes.iter_mut().zip(active).filter(|(_, active)| **active) {
                lane.position = step(lane.position);
            }
        };
        moved(lanes, |position| position - 1);
        let views: Vec<&StageCore> = lanes.iter().map(|lane| &**lane).collect();
        let walked = walk(
            &shared,
            &scratch,
            count,
            None,
            Pass::Draft,
            |block_index, block, loop_index| {
                Self::block_lanes(&views, active, block_index, block, loop_index)
            },
        );
        moved(lanes, |position| position + 1);
        walked
    }

    /// The draft exit over one received latent row of each of `lanes` at its current position,
    /// in one pass, as [StageCore::forward_lanes] runs them, on a stage with one: the output
    /// suffix, on caches of its own, then, when asked for, the head, which leaves lane `i`'s
    /// draft logits at [StageCore::logits_row] `i`. No lane moves and `state` is left as it came,
    /// so [StageCore::forward_lanes] then runs the stage's own loops on the same rows.
    pub(crate) fn draft_lanes(
        lanes: &mut [&mut StageCore],
        active: &[bool],
        want_logits: bool,
    ) -> Result<(), BackendError> {
        let scratch = batch_scratch(lanes, active.len())?;
        for (lane, _) in lanes.iter().zip(active).filter(|(_, active)| **active) {
            lane.check_rows(1)?;
        }
        let shared = Arc::clone(&lanes[0].shared);
        let count = lanes.len();
        let latent = keep_latent(&shared, &scratch, count)?;
        let views: Vec<&StageCore> = lanes.iter().map(|lane| &**lane).collect();
        walk(
            &shared,
            &scratch,
            count,
            want_logits.then_some(0..count),
            Pass::Draft,
            |block_index, block, loop_index| {
                Self::block_lanes(&views, active, block_index, block, loop_index)
            },
        )?;
        restore_latent(&shared, &scratch, latent, count)
    }

    /// Takes the sequence back to `position`, at or before where it stands: the rows from there
    /// on are written again by the steps that follow, and no step reads a row past its own.
    pub(crate) fn rewind(&mut self, position: usize) -> Result<(), BackendError> {
        if position > self.position {
            return Err(BackendError::new(
                DEVICE,
                format!(
                    "a rewind to position {position} is past the sequence's {}",
                    self.position
                ),
            ));
        }
        self.position = position;
        Ok(())
    }

    /// One decode row for each of `lanes`, each at its own position, in one pass; the head, when
    /// asked for, leaves lane `i`'s logits at [StageCore::logits_row] `i`.
    ///
    /// Row `i` of the lanes' shared scratch is `lanes[i]`'s: [StageCore::embed_lanes] wrote it,
    /// or the stage before sent it. The projections read each weight row once for the whole batch
    /// through the token GEMV, whose sum for a row is the one-row kernel's; rope, the cache append
    /// and attention run lane by lane, each on its own cache; the norms and lerps keep a block a
    /// row. So a lane's row has the bits a pass of its own gives it, whichever lanes share the
    /// pass.
    ///
    /// A lane `active` marks false keeps its row in the batch, since the rows travel between
    /// stages together, but appends no cache row and does not advance; nothing reads its results.
    pub(crate) fn forward_lanes(
        lanes: &mut [&mut StageCore],
        active: &[bool],
        want_logits: bool,
    ) -> Result<(), BackendError> {
        let scratch = batch_scratch(lanes, active.len())?;
        for (lane, _) in lanes.iter().zip(active).filter(|(_, active)| **active) {
            lane.check_rows(1)?;
        }
        let shared = Arc::clone(&lanes[0].shared);
        let count = lanes.len();
        let views: Vec<&StageCore> = lanes.iter().map(|lane| &**lane).collect();
        walk(
            &shared,
            &scratch,
            count,
            want_logits.then_some(0..count),
            Pass::Stage,
            |block_index, block, loop_index| {
                Self::block_lanes(&views, active, block_index, block, loop_index)
            },
        )?;
        for (lane, _) in lanes.iter_mut().zip(active).filter(|(_, active)| **active) {
            lane.position += 1;
        }
        Ok(())
    }

    /// Copies the first `rows` vocabulary rows the head left on the device to the host.
    pub(crate) fn read_logits_rows(&self, rows: usize) -> Result<Vec<Vec<f32>>, BackendError> {
        if rows > LANE_BATCH {
            return Err(BackendError::new(
                DEVICE,
                format!("the head leaves at most {LANE_BATCH} vocabulary rows, not {rows}"),
            ));
        }
        let vocab = self.shared.config.vocab_size;
        let mut bytes = vec![0u8; rows * vocab * 4];
        self.shared
            .cuda
            .copy_dtoh(&mut bytes, &self.scratch.logits)
            .map_err(|error| BackendError::new(DEVICE, error))?;
        Ok(bytes
            .chunks(vocab * 4)
            .map(|row| {
                row.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f32::from_le_bytes(*chunk))
                    .collect()
            })
            .collect())
    }

    /// Copies the vocabulary row [StageCore::forward] left on the device to the host.
    pub(crate) fn read_logits(&self) -> Result<Vec<f32>, BackendError> {
        Ok(self.read_logits_rows(1)?.swap_remove(0))
    }

    /// One block over `rows` rows: the token kernels for one row, the batched ones for a chunk.
    fn run_block(
        &self,
        block_index: usize,
        block: &Block,
        loop_index: usize,
        position: usize,
        rows: usize,
    ) -> Result<(), BackendError> {
        if rows == 1 {
            Self::block_lanes(&[self], &[true], block_index, block, loop_index)
        } else {
            self.block_batch(block_index, block, loop_index, position, rows)
        }
    }

    /// The cache this sequence keeps for `block` in loop `loop_index`: the keys, the values, the
    /// rows it holds, and the row count a sliding ring wraps at (0 for a full cache).
    fn cache(
        &self,
        block_index: usize,
        block: &Block,
        loop_index: usize,
    ) -> Result<(Deviceptr, Deviceptr, usize, usize), BackendError> {
        let config = &self.shared.config;
        let slot = if block.cache_slots == 1 {
            0
        } else {
            loop_index.min(block.cache_slots - 1)
        };
        let keys =
            self.caches[block_index][slot].ok_or_else(|| stage_refusal(block_index, loop_index))?;
        let rows = cache_rows(block, self.shared.context);
        let ring = if rows < self.shared.context { rows } else { 0 };
        let kv_width = config.num_key_value_heads * config.head_dim;
        Ok((keys, keys.offset(rows * kv_width * 2), rows, ring))
    }

    /// One decoder block over one row of each of `lanes`, each at its own position.
    ///
    /// The projections run once over all the rows; rope, the cache append and split attention run
    /// lane by lane, each on its own cache, skipping a lane `active` marks false. One lane is a
    /// decode step or a one-row prompt chunk.
    fn block_lanes(
        lanes: &[&StageCore],
        active: &[bool],
        block_index: usize,
        block: &Block,
        loop_index: usize,
    ) -> Result<(), BackendError> {
        let Some(first) = lanes.first() else {
            return Ok(());
        };
        let shared = &first.shared;
        let scratch = &first.scratch;
        let cuda = &shared.cuda;
        let config = &shared.config;
        let weights = shared.weights.blocks[block_index]
            .as_ref()
            .ok_or_else(|| stage_refusal(block_index, loop_index))?;
        let count = lanes.len();
        let hidden = config.hidden_size;
        let q_width = config.num_attention_heads * config.head_dim;
        let intermediate = config.intermediate_size;
        let alpha_scale = config.alpha_scale() as f32;
        let gate_factor = config.gate_scale_factor() as f32;
        let theta = block.rope_theta as f32;

        gemv_group(
            cuda,
            &shared.kernels,
            &[
                (&weights.attention.q, scratch.q),
                (&weights.attention.k, scratch.k),
                (&weights.attention.v, scratch.v),
                (&weights.attention.g, scratch.gate),
            ],
            scratch.hidden,
            count,
        )?;
        // Rope, the cache append and split attention run lane by lane, each on its own cache at
        // its own position, but in one launch apiece for all the lanes.
        let mut words = [LANE_IDLE; ATTENTION_LANES];
        let mut caches = [0u64; ATTENTION_LANES];
        let mut splits = 0;
        let (mut cache_rows, mut ring) = (0, 0);
        for (row, lane) in lanes.iter().enumerate() {
            if !active[row] {
                continue;
            }
            let (keys, _, rows, lane_ring) = lane.cache(block_index, block, loop_index)?;
            let (_, span, runs) = decode_split(lane.position, block.window);
            words[row] = lane_word(lane.position, span)?;
            caches[row] = keys.0;
            splits = splits.max(runs);
            (cache_rows, ring) = (rows, lane_ring);
        }
        if splits > 0 {
            let kv_width = config.num_key_value_heads * config.head_dim;
            let values_offset = (cache_rows * kv_width) as u64;
            let lane_partials = lane_partials(config) as u64;
            // Fixed-size argument arrays: this path runs every decode block, and a Vec per
            // launch would allocate in the serving loop.
            let mut args = [0u64; 28];
            args[..12].copy_from_slice(&[
                scratch.q.0,
                scratch.k.0,
                scratch.v.0,
                weights.attention.sqk.0,
                config.num_attention_heads as u64,
                config.head_dim as u64,
                u64::from(theta.to_bits()),
                u64::from((config.qk_scale_factor() as f32).to_bits()),
                q_width as u64,
                kv_width as u64,
                cache_rows as u64,
                values_offset,
            ]);
            args[12..20].copy_from_slice(&words);
            args[20..28].copy_from_slice(&caches);
            launch(
                cuda,
                shared.kernels.rope_norm_lanes,
                (
                    (config.num_attention_heads + config.num_key_value_heads) as u32,
                    count as u32,
                    1,
                ),
                (256, 1, 1),
                &args,
            )?;
            let mut args = [0u64; 26];
            args[..10].copy_from_slice(&[
                scratch.q.0,
                scratch.partials.0,
                config.num_attention_heads as u64,
                config.num_key_value_heads as u64,
                config.head_dim as u64,
                block.window.unwrap_or(0) as u64,
                ring as u64,
                q_width as u64,
                values_offset,
                lane_partials,
            ]);
            args[10..18].copy_from_slice(&words);
            args[18..26].copy_from_slice(&caches);
            launch(
                cuda,
                shared.kernels.attn_split,
                (
                    config.num_key_value_heads as u32,
                    splits as u32,
                    count as u32,
                ),
                (256, 1, 1),
                &args,
            )?;
            let mut args = [0u64; 17];
            args[..9].copy_from_slice(&[
                scratch.partials.0,
                scratch.gate.0,
                weights.attention.s_gate.0,
                scratch.attention.0,
                config.head_dim as u64,
                u64::from(gate_factor.to_bits()),
                block.window.unwrap_or(0) as u64,
                q_width as u64,
                lane_partials,
            ]);
            args[9..].copy_from_slice(&words);
            launch(
                cuda,
                shared.kernels.attn_combine,
                (config.num_attention_heads as u32, count as u32, 1),
                (256, 1, 1),
                &args,
            )?;
        }
        launch(
            cuda,
            shared.kernels.justnorm,
            (1, count as u32, 1),
            (256, 1, 1),
            &[scratch.attention.0, q_width as u64],
        )?;
        gemv_batch(
            cuda,
            &shared.kernels,
            &weights.attention.o,
            scratch.attention,
            scratch.branch,
            count,
        )?;
        launch(
            cuda,
            shared.kernels.slerp,
            (1, count as u32, 1),
            (256, 1, 1),
            &[
                scratch.hidden.0,
                scratch.branch.0,
                weights.attention.alpha.0,
                scratch.state.0,
                hidden as u64,
                alpha_scale.to_bits() as u64,
            ],
        )?;

        gemv_group(
            cuda,
            &shared.kernels,
            &[
                (&weights.mlp.gate, scratch.mlp_gate),
                (&weights.mlp.up, scratch.mlp_up),
            ],
            scratch.state,
            count,
        )?;
        launch(
            cuda,
            shared.kernels.mlp_epilogue,
            (1, count as u32, 1),
            (256, 1, 1),
            &[
                scratch.mlp_gate.0,
                scratch.mlp_up.0,
                weights.mlp.suv_gate.0,
                weights.mlp.suv_up.0,
                scratch.mlp_hidden.0,
                intermediate as u64,
                (config.mlp_scale_factor() as f32).to_bits() as u64,
            ],
        )?;
        gemv_batch(
            cuda,
            &shared.kernels,
            &weights.mlp.down,
            scratch.mlp_hidden,
            scratch.branch,
            count,
        )?;
        launch(
            cuda,
            shared.kernels.slerp,
            (1, count as u32, 1),
            (256, 1, 1),
            &[
                scratch.state.0,
                scratch.branch.0,
                weights.mlp.alpha.0,
                scratch.hidden.0,
                hidden as u64,
                alpha_scale.to_bits() as u64,
            ],
        )?;
        Ok(())
    }

    /// One decoder block over up to PREFILL_BATCH positions.
    ///
    /// The projections read each weight row once for every token in the chunk; the elementwise
    /// kernels keep a block per token, so every row's reduction is the one a single-token step
    /// computes. The cache rows for the whole chunk are written before attention, and a query
    /// still stops at its own position.
    fn block_batch(
        &self,
        block_index: usize,
        block: &Block,
        loop_index: usize,
        position: usize,
        batch: usize,
    ) -> Result<(), BackendError> {
        let shared = &self.shared;
        let cuda = &shared.cuda;
        let config = &shared.config;
        let weights = shared.weights.blocks[block_index]
            .as_ref()
            .ok_or_else(|| stage_refusal(block_index, loop_index))?;
        let hidden = config.hidden_size;
        let q_width = config.num_attention_heads * config.head_dim;
        let intermediate = config.intermediate_size;
        let alpha_scale = config.alpha_scale() as f32;
        let gate_factor = config.gate_scale_factor() as f32;
        let theta = block.rope_theta as f32;
        let (keys, values, rows, ring) = self.cache(block_index, block, loop_index)?;

        gemv_group(
            cuda,
            &shared.kernels,
            &[
                (&weights.attention.q, self.scratch.q),
                (&weights.attention.k, self.scratch.k),
                (&weights.attention.v, self.scratch.v),
                (&weights.attention.g, self.scratch.gate),
            ],
            self.scratch.hidden,
            batch,
        )?;
        rope_norm(
            cuda,
            &shared.kernels,
            config,
            &self.scratch,
            weights.attention.sqk,
            (keys, values, rows),
            position,
            theta,
            0..batch,
        )?;
        let group = config.num_attention_heads / config.num_key_value_heads;
        launch(
            cuda,
            shared.kernels.attn_mma,
            (
                batch.div_ceil(PREFILL_Q) as u32,
                config.num_key_value_heads as u32,
                1,
            ),
            ((group * 32) as u32, 1, 1),
            &[
                self.scratch.q.0,
                keys.0,
                values.0,
                self.scratch.gate.0,
                weights.attention.s_gate.0,
                self.scratch.attention.0,
                position as u64,
                batch as u64,
                config.num_attention_heads as u64,
                config.num_key_value_heads as u64,
                block.window.unwrap_or(0) as u64,
                gate_factor.to_bits() as u64,
                q_width as u64,
                q_width as u64,
                ring as u64,
            ],
        )?;
        launch(
            cuda,
            shared.kernels.justnorm,
            (1, batch as u32, 1),
            (256, 1, 1),
            &[self.scratch.attention.0, q_width as u64],
        )?;
        gemv_batch(
            cuda,
            &shared.kernels,
            &weights.attention.o,
            self.scratch.attention,
            self.scratch.branch,
            batch,
        )?;
        launch(
            cuda,
            shared.kernels.slerp,
            (1, batch as u32, 1),
            (256, 1, 1),
            &[
                self.scratch.hidden.0,
                self.scratch.branch.0,
                weights.attention.alpha.0,
                self.scratch.state.0,
                hidden as u64,
                alpha_scale.to_bits() as u64,
            ],
        )?;

        gemv_group(
            cuda,
            &shared.kernels,
            &[
                (&weights.mlp.gate, self.scratch.mlp_gate),
                (&weights.mlp.up, self.scratch.mlp_up),
            ],
            self.scratch.state,
            batch,
        )?;
        launch(
            cuda,
            shared.kernels.mlp_epilogue,
            (1, batch as u32, 1),
            (256, 1, 1),
            &[
                self.scratch.mlp_gate.0,
                self.scratch.mlp_up.0,
                weights.mlp.suv_gate.0,
                weights.mlp.suv_up.0,
                self.scratch.mlp_hidden.0,
                intermediate as u64,
                (config.mlp_scale_factor() as f32).to_bits() as u64,
            ],
        )?;
        gemv_batch(
            cuda,
            &shared.kernels,
            &weights.mlp.down,
            self.scratch.mlp_hidden,
            self.scratch.branch,
            batch,
        )?;
        launch(
            cuda,
            shared.kernels.slerp,
            (1, batch as u32, 1),
            (256, 1, 1),
            &[
                self.scratch.state.0,
                self.scratch.branch.0,
                weights.mlp.alpha.0,
                self.scratch.hidden.0,
                hidden as u64,
                alpha_scale.to_bits() as u64,
            ],
        )?;
        Ok(())
    }
}

/// The whole model's sequence, with the shared sampler over its logits.
struct DoryDeviceSession {
    core: StageCore,
    sampler: Sampler,
    sampling: Sampling,
}

impl DoryDeviceSession {
    /// One row or one chunk of the whole walk, returning the last row's logits when asked.
    fn run(&mut self, tokens: &[u32], want_logits: bool) -> Result<Vec<f32>, BackendError> {
        self.core.embed(tokens)?;
        self.core.forward(tokens.len(), want_logits)?;
        if want_logits {
            self.core.read_logits()
        } else {
            Ok(Vec::new())
        }
    }
}

/// The rows at or below which gemv_batch hands a chunk to the token GEMV rather than the batched
/// tensor-core GEMM.
const TOKEN_ROWS: usize = 8;

/// The most lanes one [StageCore::forward_lanes] pass carries: the token GEMV's rows, so every
/// lane's row keeps the one-row bits.
pub(crate) const LANE_BATCH: usize = TOKEN_ROWS;
const _: () = assert!(LANE_BATCH <= ATTENTION_LANES);

/// The prompt chunks of the exact per-loop cache mode, `width` rows each, or one token at a time
/// in last-loop mode.
///
/// With a cache per loop a whole chunk can run loop by loop: every query sees the same per-loop
/// rows it would see token by token. Last-loop's overwritten cache makes the rows a later query
/// sees depend on the tokens processed after the earlier ones, so that mode keeps the
/// token-at-a-time order.
///
/// A row's bits do not depend on the chunk it runs in, so long as every chunk starts on a query
/// tile of dory_attn_mma (a multiple of PREFILL_Q) and none is narrow enough for the token GEMV:
/// a last chunk of TOKEN_ROWS or fewer rows takes PREFILL_Q more from the chunk before. Any
/// width that is a multiple of PREFILL_Q then runs a prompt to the same bits, which is what lets
/// the recurrence group cut prompts finer than one process does.
pub(crate) fn prompt_chunks(mode: KvCacheMode, prompt: usize, width: usize) -> Vec<Range<usize>> {
    if mode != KvCacheMode::PerLoop {
        return (0..prompt).map(|start| start..start + 1).collect();
    }
    debug_assert!(
        width.is_multiple_of(PREFILL_Q) && width > PREFILL_Q + TOKEN_ROWS && width <= PREFILL_BATCH
    );
    let mut chunks: Vec<Range<usize>> = (0..prompt)
        .step_by(width)
        .map(|start| start..(start + width).min(prompt))
        .collect();
    if let [.., before, last] = chunks.as_mut_slice()
        && last.len() <= TOKEN_ROWS
    {
        before.end -= PREFILL_Q;
        last.start -= PREFILL_Q;
    }
    chunks
}

impl ModelSession for DoryDeviceSession {
    fn set_sampling(&mut self, sampling: Sampling) {
        // The penalties, temperature, top-k and top-p run on the host over the row every step
        // already copies back, so a non-greedy request is served rather than approximated.
        self.sampling = sampling;
        self.sampler.reset();
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::new(DEVICE, "the prompt has no tokens"));
        }
        let chunks = prompt_chunks(
            self.core.shared.config.recurrent_kv_cache_mode,
            tokens.len(),
            PREFILL_BATCH,
        );
        let mut logits = Vec::new();
        for (index, chunk) in chunks.iter().enumerate() {
            let last = index + 1 == chunks.len();
            logits = self.run(&tokens[chunk.clone()], last)?;
        }
        self.sampler.reset();
        self.sampler.record_prompt(tokens);
        let token = self.sampler.choose(&logits, &self.sampling);
        Ok(StepOutput {
            next_token: self.core.shared.config.next_token(token),
            logits,
            draft: Vec::new(),
        })
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        let logits = self.run(&[token], true)?;
        self.sampler.record(token);
        let chosen = self.sampler.choose(&logits, &self.sampling);
        Ok(StepOutput {
            next_token: self.core.shared.config.next_token(chosen),
            logits,
            draft: Vec::new(),
        })
    }

    fn reusable_prefix(&self) -> Option<&[u32]> {
        None
    }
}

/// The scratch `lanes` share, checked for one batched pass of `rows` rows: one row a lane, at
/// most [LANE_BATCH] lanes, every lane on the same scratch set.
fn batch_scratch(lanes: &[&mut StageCore], rows: usize) -> Result<Arc<OwnedScratch>, BackendError> {
    let Some(first) = lanes.first() else {
        return Err(BackendError::new(DEVICE, "a batched pass of no lanes"));
    };
    if lanes.len() != rows || rows > LANE_BATCH {
        return Err(BackendError::new(
            DEVICE,
            format!(
                "a batched pass takes one row for each of at most {LANE_BATCH} lanes, not {rows} \
                 rows for {} lanes",
                lanes.len()
            ),
        ));
    }
    if lanes
        .iter()
        .any(|lane| !Arc::ptr_eq(&lane.scratch, &first.scratch))
    {
        return Err(BackendError::new(
            DEVICE,
            "the lanes of a batched pass must share one scratch set",
        ));
    }
    Ok(Arc::clone(&first.scratch))
}

/// Copies the first `rows` received latent rows aside for a draft exit, whose suffix blocks use
/// `state` as their own scratch, and returns where they are kept.
fn keep_latent(
    shared: &DeviceShared,
    scratch: &Scratch,
    rows: usize,
) -> Result<Deviceptr, BackendError> {
    let latent = kept_latent(scratch)?;
    copy(
        &shared.cuda,
        latent,
        scratch.state,
        rows * shared.config.hidden_size * 4,
    )?;
    Ok(latent)
}

/// Where a stage with a draft exit keeps the received latent rows.
fn kept_latent(scratch: &Scratch) -> Result<Deviceptr, BackendError> {
    scratch
        .latent
        .ok_or_else(|| BackendError::new(DEVICE, "this recurrence stage holds no draft exit"))
}

/// Copies the first `rows` kept latent rows back into `state`.
fn restore_latent(
    shared: &DeviceShared,
    scratch: &Scratch,
    latent: Deviceptr,
    rows: usize,
) -> Result<(), BackendError> {
    copy(
        &shared.cuda,
        scratch.state,
        latent,
        rows * shared.config.hidden_size * 4,
    )
}

/// Gathers the embedding rows of `tokens` into `scratch.hidden` from row 0, on the stage that
/// runs the prefix.
fn embed_rows(
    shared: &DeviceShared,
    scratch: &Scratch,
    tokens: &[u32],
) -> Result<(), BackendError> {
    let cuda = &shared.cuda;
    cuda.make_current()
        .map_err(|error| BackendError::new(DEVICE, error))?;
    let config = &shared.config;
    let embed = shared
        .weights
        .embed
        .ok_or_else(|| BackendError::new(DEVICE, "the embedding belongs to recurrence rank 0"))?;
    for token in tokens {
        if *token as usize >= config.vocab_size {
            return Err(BackendError::new(
                DEVICE,
                format!(
                    "token {token} is outside the checkpoint vocabulary {}",
                    config.vocab_size
                ),
            ));
        }
    }
    let hidden = config.hidden_size;
    for (offset, token) in tokens.iter().enumerate() {
        launch(
            cuda,
            shared.kernels.embed,
            (hidden.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            &[
                embed.pointer.0,
                scratch.hidden.offset(offset * hidden * 4).0,
                *token as u64,
                hidden as u64,
                hidden as u64,
                0,
            ],
        )?;
    }
    Ok(())
}

/// Which part of the looped walk a pass runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pass {
    /// The stage's own part: the prefix, its loops and the suffix, as the stage has them.
    Stage,
    /// The draft exit: the output suffix alone, over the latent the stage received.
    Draft,
}

/// A stage's walk over `rows` rows of `scratch`: the input prefix (when the stage has it), its
/// loops and the output suffix (when it has it), with `block` run for each block pass, then the
/// vocabulary head over the rows `head` into the logits rows from 0. A [Pass::Draft] walk runs
/// only the suffix, over the rows in `state`.
fn walk(
    shared: &DeviceShared,
    scratch: &Scratch,
    rows: usize,
    head: Option<Range<usize>>,
    pass: Pass,
    mut block: impl FnMut(usize, &Block, usize) -> Result<(), BackendError>,
) -> Result<(), BackendError> {
    let cuda = &shared.cuda;
    cuda.make_current()
        .map_err(|error| BackendError::new(DEVICE, error))?;
    let config = &shared.config;
    let stage = &shared.stage;
    let (prefix, loops, suffix) = match pass {
        Pass::Stage => (stage.prefix, stage.loops.clone(), stage.suffix),
        Pass::Draft => (false, 0..0, true),
    };
    if pass == Pass::Draft && !stage.draft {
        return Err(BackendError::new(
            DEVICE,
            "this recurrence stage holds no draft exit",
        ));
    }
    let head_refusal = || {
        BackendError::new(
            DEVICE,
            "the vocabulary head belongs to the last recurrence rank",
        )
    };
    if head.is_some() && !suffix {
        return Err(head_refusal());
    }
    let hidden = config.hidden_size;
    let bytes = hidden * rows * 4;

    if prefix {
        for (block_index, input) in config.input_blocks().iter().enumerate() {
            block(block_index, input, 0)?;
        }
        copy(cuda, scratch.input_h, scratch.hidden, bytes)?;
        copy(cuda, scratch.state, scratch.hidden, bytes)?;
    }

    let start = config.input_blocks().len();
    for loop_index in loops {
        launch(
            cuda,
            shared.kernels.loop_combine,
            (1, rows as u32, 1),
            (256, 1, 1),
            &[
                scratch.state.0,
                scratch.input_h.0,
                scratch.hidden.0,
                hidden as u64,
            ],
        )?;
        for (offset, recurrent) in config.recurrent_blocks().iter().enumerate() {
            block(start + offset, recurrent, loop_index)?;
        }
        launch(
            cuda,
            shared.kernels.justnorm,
            (1, rows as u32, 1),
            (256, 1, 1),
            &[scratch.hidden.0, hidden as u64],
        )?;
        copy(cuda, scratch.state, scratch.hidden, bytes)?;
    }

    if suffix {
        copy(cuda, scratch.hidden, scratch.state, bytes)?;
        let start = config.input_blocks().len() + config.recurrent_blocks().len();
        for (offset, output) in config.output_blocks().iter().enumerate() {
            block(start + offset, output, 0)?;
        }
        if let Some(head) = head {
            let (Some(lm_head), Some(logit_scale)) =
                (shared.weights.lm_head, shared.weights.logit_scale)
            else {
                return Err(head_refusal());
            };
            if head.len() > LANE_BATCH || head.end > rows {
                return Err(BackendError::new(
                    DEVICE,
                    format!("a head over rows {head:?} of {rows}"),
                ));
            }
            gemv_batch(
                cuda,
                &shared.kernels,
                &lm_head,
                scratch.hidden.offset(head.start * hidden * 4),
                scratch.logits,
                head.len(),
            )?;
            launch(
                cuda,
                shared.kernels.logit_scale,
                (config.vocab_size.div_ceil(256) as u32, head.len() as u32, 1),
                (256, 1, 1),
                &[
                    logit_scale.0,
                    scratch.logits.0,
                    config.vocab_size as u64,
                    (config.logit_scale_divisor() as f32).to_bits() as u64,
                ],
            )?;
        }
    }
    Ok(())
}

fn copy(
    cuda: &Cuda,
    destination: Deviceptr,
    source: Deviceptr,
    bytes: usize,
) -> Result<(), BackendError> {
    cuda.copy_dtod(&destination, &source, bytes)
        .map_err(|error| BackendError::new(DEVICE, error))
}

/// `batch` rows of `input` share each read of `matrix`; `output` is `[batch][rows]`.
fn gemv_batch(
    cuda: &Cuda,
    kernels: &Kernels,
    matrix: &DeviceMat,
    input: Deviceptr,
    output: Deviceptr,
    batch: usize,
) -> Result<(), BackendError> {
    if batch > TOKEN_ROWS && matrix.rows.is_multiple_of(32) && matrix.columns.is_multiple_of(32) {
        // One 128-token x 256-column tensor-core block per grid point: the whole prompt chunk
        // is one launch, and the row-major checkpoint weights are loaded a tile at a time.
        // The eight-token path keeps q4e_gemv_bf16, whose per-token dot is the token kernel's
        // bits.
        return launch(
            cuda,
            kernels.mma_wide,
            (
                matrix.rows.div_ceil(WIDE_COLUMNS) as u32,
                batch.div_ceil(128) as u32,
                1,
            ),
            (256, 1, 1),
            &[
                matrix.pointer.0,
                input.0,
                output.0,
                matrix.rows as u64,
                matrix.columns as u64,
                batch as u64,
            ],
        );
    }
    if batch > TOKEN_ROWS {
        // The load-time shape check refuses this, but keep the launch honest if another caller
        // reaches it: the token GEMV's bits differ from the tensor-core chunk's, so silently
        // switching arithmetic here would make a prompt's logits depend on its width.
        return Err(BackendError::new(
            DEVICE,
            format!(
                "a {batch}-row projection of a {}x{} matrix needs both dimensions in 32-column \
                 steps for the wide kernel; the token GEMV would change the chunk's arithmetic",
                matrix.rows, matrix.columns
            ),
        ));
    }
    // The kernel covers two rows a warp for a batch (16 rows a block at 256 threads) and one row
    // a warp for a single row; the grid must match that mapping or half the blocks exit unused.
    let rows_per_block = if batch > 1 { 16 } else { 8 };
    launch(
        cuda,
        kernels.gemv_bf16,
        (matrix.rows.div_ceil(rows_per_block) as u32, 1, 1),
        (256, 1, 1),
        &[
            matrix.pointer.0,
            input.0,
            output.0,
            matrix.rows as u64,
            matrix.columns as u64,
            batch as u64,
        ],
    )
}

/// Up to four projections of one `input`, each into its own output.
///
/// Up to TOKEN_ROWS rows run as one q4e_gemv4_bf16 launch over all the matrices' rows. That
/// kernel gives every row q4e_gemv_bf16's dot (the same lane split, the same warp sum), so each
/// output has the bits its own launch gives it, and a decode step's four attention projections
/// and two MLP inputs cost two launches rather than six. A wider chunk runs one dory_mma_wide4
/// launch over all the matrices' column tiles, each tile dory_mma_wide's own, so the outputs keep
/// that kernel's bits and the matrices share their waves rather than each leaving its last one
/// part empty.
fn gemv_group(
    cuda: &Cuda,
    kernels: &Kernels,
    projections: &[(&DeviceMat, Deviceptr)],
    input: Deviceptr,
    batch: usize,
) -> Result<(), BackendError> {
    let columns = projections.first().map(|(matrix, _)| matrix.columns);
    let wide = batch > TOKEN_ROWS;
    if projections.len() > 4
        || projections.iter().any(|(matrix, _)| {
            Some(matrix.columns) != columns
                || (wide && !(matrix.rows.is_multiple_of(32) && matrix.columns.is_multiple_of(32)))
        })
    {
        for (matrix, output) in projections {
            gemv_batch(cuda, kernels, matrix, input, *output, batch)?;
        }
        return Ok(());
    }
    let Some(&(first, first_output)) = projections.first() else {
        return Ok(());
    };
    // A missing slot is zero rows of the first matrix: the kernel never reaches it.
    let slot = |index: usize| {
        projections
            .get(index)
            .map_or((first.pointer, 0, first_output), |(matrix, output)| {
                (matrix.pointer, matrix.rows, *output)
            })
    };
    let slots = [slot(0), slot(1), slot(2), slot(3)];
    if wide {
        let tiles: usize = slots
            .iter()
            .map(|(_, rows, _)| rows.div_ceil(WIDE_COLUMNS))
            .sum();
        return launch(
            cuda,
            kernels.mma_wide4,
            (tiles as u32, batch.div_ceil(128) as u32, 1),
            (256, 1, 1),
            &[
                slots[0].0.0,
                slots[1].0.0,
                slots[2].0.0,
                slots[3].0.0,
                input.0,
                slots[0].2.0,
                slots[1].2.0,
                slots[2].2.0,
                slots[3].2.0,
                slots[0].1 as u64,
                slots[1].1 as u64,
                slots[2].1 as u64,
                slots[3].1 as u64,
                first.columns as u64,
                batch as u64,
            ],
        );
    }
    let total: usize = slots.iter().map(|(_, rows, _)| rows).sum();
    let rows_per_block = if batch > 1 { 16 } else { 8 };
    launch(
        cuda,
        kernels.gemv4_bf16,
        (total.div_ceil(rows_per_block) as u32, 1, 1),
        (256, 1, 1),
        &[
            slots[0].0.0,
            slots[1].0.0,
            slots[2].0.0,
            slots[3].0.0,
            input.0,
            slots[0].2.0,
            slots[1].2.0,
            slots[2].2.0,
            slots[3].2.0,
            slots[0].1 as u64,
            slots[1].1 as u64,
            slots[2].1 as u64,
            slots[3].1 as u64,
            first.columns as u64,
            batch as u64,
        ],
    )
}

/// dory_rope_norm over the scratch rows `tokens`, the first at `position`: rope and the per-head
/// norm on q and k in place, and each row's BF16 K and V into `cache` (keys, values, rows), which
/// wraps at `rows`.
#[allow(clippy::too_many_arguments)]
fn rope_norm(
    cuda: &Cuda,
    kernels: &Kernels,
    config: &DoryConfig,
    scratch: &Scratch,
    sqk: Deviceptr,
    cache: (Deviceptr, Deviceptr, usize),
    position: usize,
    theta: f32,
    tokens: Range<usize>,
) -> Result<(), BackendError> {
    let (keys, values, rows) = cache;
    let qk_factor = config.qk_scale_factor() as f32;
    let q_width = config.num_attention_heads * config.head_dim;
    let kv_width = config.num_key_value_heads * config.head_dim;
    launch(
        cuda,
        kernels.rope_norm,
        (
            (config.num_attention_heads + config.num_key_value_heads) as u32,
            tokens.len() as u32,
            1,
        ),
        (256, 1, 1),
        &[
            scratch.q.offset(tokens.start * q_width * 4).0,
            scratch.k.offset(tokens.start * kv_width * 4).0,
            scratch.v.offset(tokens.start * kv_width * 4).0,
            keys.0,
            values.0,
            sqk.0,
            config.num_attention_heads as u64,
            config.head_dim as u64,
            position as u64,
            theta.to_bits() as u64,
            qk_factor.to_bits() as u64,
            q_width as u64,
            kv_width as u64,
            rows as u64,
        ],
    )
}

fn launch(
    cuda: &Cuda,
    kernel: Kernel,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    args: &[u64],
) -> Result<(), BackendError> {
    cuda.launch_async(kernel, grid, block, args)
        .map_err(|error| BackendError::new(DEVICE, format!("{}: {error}", kernel.name())))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        ATTENTION_LANES, DeviceMat, DeviceShared, DoryDeviceBackend, DoryStage, Kernels,
        LANE_BATCH, LANE_IDLE, PREFILL_BATCH, PREFILL_Q, SPLIT_MAX, StageCore, attention_fits,
        decode_split, lane_word, launch, profile_refusal, prompt_chunks,
    };
    use crate::config::{DoryConfig, KvCacheMode};
    use sparkflash_model::profile::ModelProfile;

    #[test]
    fn profile_fields_the_arm_does_not_honor_are_refused() {
        assert!(profile_refusal(&ModelProfile::default()).is_ok());
        let refused = |edit: fn(&mut ModelProfile)| {
            let mut profile = ModelProfile::default();
            edit(&mut profile);
            profile_refusal(&profile).expect_err("the unsupported profile is refused")
        };
        assert!(refused(|profile| profile.prefill.chunk = Some(2048)).contains("prefill chunk"));
        assert!(
            refused(|profile| profile.prefill.exact = Some(true)).contains("not bit-identical")
        );
        assert!(refused(|profile| profile.decode.tensor_cores = Some(true)).contains("warp-dot"));
        assert!(refused(|profile| profile.speculation.enabled = Some(true)).contains("drafting"));
        assert!(refused(|profile| profile.sequence.parallel = Some(true)).contains("sequence"));
        assert!(refused(|profile| profile.runtime.concurrency = Some(4)).contains("one sequence"));
    }

    #[test]
    fn a_shape_off_the_wide_tiles_is_refused_at_load() {
        let mut config = DoryConfig::tiny_ring();
        config.hidden_size = 65;
        let error = attention_fits(&config).expect_err("65 is not a 32-column step");
        assert!(error.contains("multiple of 32"), "{error}");
        attention_fits(&DoryConfig::tiny_ring()).expect("the ring fixture is 32-aligned");
    }

    #[test]
    #[ignore = "needs a Dory/Qwen cubin and a CUDA device"]
    fn a_batched_lane_pass_is_each_lane_alone_bit_for_bit() {
        if !DoryDeviceBackend::available() {
            eprintln!("skipping the batched lane check: this build has no Dory cubin");
            return;
        }
        let directory = tempfile::tempdir().expect("a temporary checkpoint");
        crate::fixture::write(directory.path(), &DoryConfig::tiny_ring_json())
            .expect("the fixture writes");
        let shared = match DeviceShared::open(directory.path(), 5000, |config| {
            Ok(DoryStage::whole(config))
        }) {
            Ok(shared) => Arc::new(shared),
            Err(error) => {
                eprintln!("skipping the batched lane check: {error}");
                return;
            }
        };
        // Prompts of different lengths, so the lanes decode at different positions: one past the
        // sliding ring's wrap, one inside the window, one a single row.
        let prompts: Vec<Vec<u32>> = [4200usize, 37, 1]
            .iter()
            .enumerate()
            .map(|(lane, len)| {
                (0..*len)
                    .map(|index| ((index * 7 + lane * 13) % 61 + 1) as u32)
                    .collect()
            })
            .collect();
        let mut alone: Vec<StageCore> = prompts
            .iter()
            .map(|_| StageCore::open(Arc::clone(&shared)).expect("a solo sequence"))
            .collect();
        let mut lanes = StageCore::lanes(&shared, prompts.len()).expect("the lanes");
        let mut tokens: Vec<u32> = Vec::new();
        for (index, prompt) in prompts.iter().enumerate() {
            for core in [&mut alone[index], &mut lanes[index]] {
                for chunk in prompt_chunks(KvCacheMode::PerLoop, prompt.len(), PREFILL_BATCH) {
                    core.embed(&prompt[chunk.clone()])
                        .expect("the chunk embeds");
                    core.forward(chunk.len(), true).expect("the chunk runs");
                }
            }
            let logits = alone[index].read_logits().expect("the logits");
            assert_eq!(
                logits,
                lanes[index].read_logits().expect("the lane's logits"),
                "lane {index}'s prompt"
            );
            tokens.push(argmax(&logits));
        }
        // The middle lane sits out the third step: it keeps its row in the batch but appends
        // nothing and stays where it was, so its later steps still match its solo sequence.
        for step in 0..6 {
            let active: Vec<bool> = (0..prompts.len())
                .map(|lane| !(step == 2 && lane == 1))
                .collect();
            let mut expected = Vec::new();
            for (index, core) in alone.iter_mut().enumerate() {
                if active[index] {
                    core.embed(&[tokens[index]]).expect("the token embeds");
                    core.forward(1, true).expect("the solo step runs");
                    expected.push(Some(core.read_logits().expect("the solo logits")));
                } else {
                    expected.push(None);
                }
            }
            let mut batch: Vec<&mut StageCore> = lanes.iter_mut().collect();
            StageCore::embed_lanes(&batch, &tokens).expect("the batch embeds");
            StageCore::forward_lanes(&mut batch, &active, true).expect("the batched step runs");
            let rows = batch[0]
                .read_logits_rows(prompts.len())
                .expect("the batched logits");
            for (index, (want, got)) in expected.iter().zip(&rows).enumerate() {
                let Some(want) = want else {
                    continue;
                };
                let differing = want
                    .iter()
                    .zip(got)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(
                    differing, 0,
                    "step {step} lane {index}: {differing} logits differ"
                );
                tokens[index] = argmax(want);
            }
            for (solo, lane) in alone.iter().zip(&lanes) {
                assert_eq!(solo.position(), lane.position(), "step {step}");
            }
        }
        // A batch is one row a lane, on one scratch, at most LANE_BATCH lanes.
        let mut other = StageCore::open(Arc::clone(&shared)).expect("a sequence");
        let mut mixed: Vec<&mut StageCore> = vec![&mut lanes[0], &mut other];
        let error = StageCore::forward_lanes(&mut mixed, &[true, true], false)
            .expect_err("two scratch sets");
        assert!(error.to_string().contains("one scratch"), "{error}");
        let mut many = StageCore::lanes(&shared, LANE_BATCH + 1).expect("the lanes");
        let mut many: Vec<&mut StageCore> = many.iter_mut().collect();
        let error = StageCore::forward_lanes(&mut many, &[true; LANE_BATCH + 1], false)
            .expect_err("too many lanes");
        assert!(error.to_string().contains("at most"), "{error}");
    }

    /// The draft exit rank 1 runs over the latent rank 0 sends is the one-loop model: its
    /// guesses' logits are a whole one-loop process's, bit for bit, whether the draft caches took
    /// a row before the rank's own loop (a row that asked for a guess) or after it was sent on (a
    /// prompt chunk of many rows or of one, a decode row that asked for none).
    #[test]
    #[ignore = "needs a Dory/Qwen cubin and a CUDA device"]
    fn the_draft_exit_is_the_one_loop_model_in_either_order() {
        if !DoryDeviceBackend::available() {
            eprintln!("skipping the draft exit check: this build has no Dory cubin");
            return;
        }
        let looped = tempfile::tempdir().expect("a temporary checkpoint");
        let json = DoryConfig::tiny_ring_json();
        crate::fixture::write(looped.path(), &json).expect("the fixture writes");
        let once = tempfile::tempdir().expect("a temporary checkpoint");
        let mut value: serde_json::Value = serde_json::from_str(&json).expect("the fixture JSON");
        value["n_recurrent_loops"] = serde_json::json!(1);
        crate::fixture::write(once.path(), &value.to_string()).expect("the one-loop fixture");
        let world = DoryConfig::tiny_ring().n_recurrent_loops;
        let opened = [
            DeviceShared::open(looped.path(), 5000, |config| {
                DoryStage::of_rank(config, 0, world)
            }),
            DeviceShared::open(looped.path(), 5000, |config| {
                DoryStage::of_rank(config, 1, world)?.drafting()
            }),
            DeviceShared::open(once.path(), 5000, |config| Ok(DoryStage::whole(config))),
        ];
        let [first, drafter, whole] = match opened {
            [Ok(first), Ok(drafter), Ok(whole)] => [first, drafter, whole].map(Arc::new),
            [first, drafter, whole] => {
                let error = [first.err(), drafter.err(), whole.err()]
                    .into_iter()
                    .flatten();
                eprintln!(
                    "skipping the draft exit check: {:?}",
                    error.collect::<Vec<_>>()
                );
                return;
            }
        };
        let mut rank0 = StageCore::open(Arc::clone(&first)).expect("rank 0's sequence");
        let mut rank1 = StageCore::open(Arc::clone(&drafter)).expect("rank 1's sequence");
        let mut solo = StageCore::open(whole).expect("the one-loop sequence");
        let hidden = first.config.hidden_size;
        // Rank 0's rows for `tokens`, handed to rank 1 as the ring would.
        let send = |rank0: &mut StageCore, rank1: &StageCore, tokens: &[u32]| {
            rank0.embed(tokens).expect("rank 0 embeds");
            rank0.forward(tokens.len(), false).expect("rank 0 runs");
            let (state, input_h) = rank0.carry();
            let (to_state, to_input_h) = rank1.carry();
            for (to, from) in [(to_state, state), (to_input_h, input_h)] {
                first
                    .cuda
                    .copy_dtod(&to, &from, tokens.len() * hidden * 4)
                    .expect("the rows cross");
            }
        };
        let prompt: Vec<u32> = (0..1500).map(|index| (index * 7 % 61 + 1) as u32).collect();
        for chunk in [&prompt[..1499], &prompt[1499..]] {
            send(&mut rank0, &rank1, chunk);
            rank1.keep(chunk.len()).expect("the chunk is kept");
            rank1.forward(chunk.len(), false).expect("rank 1 runs");
            rank1.draft_kept(chunk.len()).expect("the kept draft runs");
            solo.embed(chunk).expect("the one-loop prompt embeds");
            solo.forward(chunk.len(), true)
                .expect("the one-loop prompt runs");
        }
        let mut token = argmax(&solo.read_logits().expect("the one-loop logits"));
        for step in 0..9 {
            send(&mut rank0, &rank1, &[token]);
            solo.embed(&[token]).expect("the one-loop token embeds");
            solo.forward(1, true).expect("the one-loop step runs");
            let want = solo.read_logits().expect("the one-loop logits");
            let mut lane = [&mut rank1];
            if step % 3 == 2 {
                StageCore::keep_lanes(&lane, &[true]).expect("the row is kept");
                StageCore::forward_lanes(&mut lane, &[true], false).expect("rank 1 steps");
                StageCore::draft_kept_lanes(&mut lane, &[true]).expect("the kept draft steps");
            } else {
                StageCore::draft_lanes(&mut lane, &[true], true).expect("the draft guesses");
                let got = lane[0].read_logits_rows(1).expect("the draft logits");
                let differing = want
                    .iter()
                    .zip(&got[0])
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(differing, 0, "step {step}: {differing} draft logits differ");
                StageCore::forward_lanes(&mut lane, &[true], false).expect("rank 1 steps");
            }
            assert_eq!(rank1.position(), solo.position(), "step {step}");
            token = argmax(&want);
        }
    }

    fn argmax(logits: &[f32]) -> u32 {
        let mut best = 0;
        for (index, value) in logits.iter().enumerate() {
            if *value > logits[best] {
                best = index;
            }
        }
        best as u32
    }

    #[test]
    fn prompt_chunks_never_leave_a_token_kernel_tail() {
        let chunks = |prompt, width| prompt_chunks(KvCacheMode::PerLoop, prompt, width);
        assert_eq!(chunks(5, 1024), vec![0..5]);
        assert_eq!(chunks(1024, 1024), vec![0..1024]);
        assert_eq!(chunks(1033, 1024), vec![0..1024, 1024..1033]);
        assert_eq!(chunks(1029, 1024), vec![0..1008, 1008..1029]);
        assert_eq!(
            chunks(4100, PREFILL_BATCH),
            vec![
                0..PREFILL_BATCH - PREFILL_Q,
                PREFILL_BATCH - PREFILL_Q..4100
            ]
        );
        // The same prompt cut at two widths reaches the same kernels: every chunk past the
        // token kernels' rows, every start on a query tile.
        for prompt in [9, 1029, 4100, 8197, 9799] {
            for width in [1024, PREFILL_BATCH] {
                let cut = chunks(prompt, width);
                assert_eq!(cut.first().map(|chunk| chunk.start), Some(0));
                assert_eq!(cut.last().map(|chunk| chunk.end), Some(prompt));
                for pair in cut.windows(2) {
                    assert_eq!(pair[0].end, pair[1].start);
                }
                for chunk in &cut {
                    assert!(
                        chunk.len() > super::TOKEN_ROWS,
                        "{prompt} at {width}: {chunk:?}"
                    );
                    assert!(chunk.start.is_multiple_of(PREFILL_Q));
                }
            }
        }
        assert_eq!(
            prompt_chunks(KvCacheMode::LastLoop, 3, 1024),
            vec![0..1, 1..2, 2..3]
        );
    }
    use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};

    /// A deterministic stream of values in [-1, 1).
    struct Stream(u64);

    impl Stream {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn to_bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }

    fn from_bf16(value: u16) -> f64 {
        f64::from(f32::from_bits(u32::from(value) << 16))
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn u16_bytes(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    /// The query heads, kv heads and head width of the released checkpoint.
    const HEADS: usize = 16;
    const KV_HEADS: usize = 4;
    const HEAD_DIM: usize = 256;

    struct Row<'a> {
        q: &'a [f32],
        keys: &'a [u16],
        values: &'a [u16],
        gate: &'a [f32],
        s_gate: &'a [u16],
        gate_factor: f32,
    }

    /// One decode row in f64 over the same BF16 cache: the window's softmax, the value sum and
    /// the sigmoid gate.
    fn reference(row: &Row<'_>, position: usize, window: Option<usize>, ring: usize) -> Vec<f64> {
        let group = HEADS / KV_HEADS;
        let kv_row = KV_HEADS * HEAD_DIM;
        let (first, _, _) = decode_split(position, window);
        let mut out = vec![0.0f64; HEADS * HEAD_DIM];
        for head in 0..HEADS {
            let kv_head = head / group;
            let q = &row.q[head * HEAD_DIM..(head + 1) * HEAD_DIM];
            let offset = |key: usize| {
                let index = if ring > 0 { key % ring } else { key };
                index * kv_row + kv_head * HEAD_DIM
            };
            let scores: Vec<f64> = (first..=position)
                .map(|key| {
                    let base = offset(key);
                    let dot: f64 = (0..HEAD_DIM)
                        .map(|dim| f64::from(q[dim]) * from_bf16(row.keys[base + dim]))
                        .sum();
                    dot / (HEAD_DIM as f64).sqrt()
                })
                .collect();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
            let sum: f64 = weights.iter().sum();
            for dim in 0..HEAD_DIM {
                let value: f64 = (first..=position)
                    .zip(&weights)
                    .map(|(key, weight)| weight * from_bf16(row.values[offset(key) + dim]))
                    .sum();
                let local = head * HEAD_DIM + dim;
                let sg = from_bf16(row.s_gate[local]) * f64::from(row.gate_factor);
                out[local] = (value / sum) / (1.0 + (-f64::from(row.gate[local]) * sg * sg).exp());
            }
        }
        out
    }

    #[test]
    #[ignore = "needs a Dory/Qwen cubin and a CUDA device"]
    fn a_grouped_chunk_projection_is_each_matrix_alone_bit_for_bit() {
        if !DoryDeviceBackend::available() {
            eprintln!("skipping the grouped projection check: this build has no Dory cubin");
            return;
        }
        let cuda = match Cuda::load() {
            Ok(cuda) => cuda,
            Err(error) => {
                eprintln!("skipping the grouped projection check: {error}");
                return;
            }
        };
        cuda.make_current().expect("the context is current");
        let kernels = Kernels::load(&cuda, HEAD_DIM).expect("the Dory kernels load");
        let columns = 512;
        let most_rows = 300;
        let mut stream = Stream(11);
        let mut total = 0usize;
        let mut alloc = |bytes: usize| cuda.alloc(bytes, &mut total).expect("a test buffer");
        let input: Vec<f32> = (0..most_rows * columns).map(|_| stream.next()).collect();
        let input_buffer = alloc(input.len() * 4);
        cuda.copy_htod(&input_buffer, &f32_bytes(&input))
            .expect("the input");
        // q, k, v and g's shape (a matrix whose rows end inside a 256-column tile among them),
        // and gate and up's.
        for shape in [&[384usize, 96, 96, 288][..], &[544, 544]] {
            let matrices: Vec<DeviceMat> = shape
                .iter()
                .map(|&rows| {
                    let weights: Vec<u16> = (0..rows * columns)
                        .map(|_| to_bf16(stream.next() * 0.1))
                        .collect();
                    let pointer = alloc(weights.len() * 2);
                    cuda.copy_htod(&pointer, &u16_bytes(&weights))
                        .expect("a matrix");
                    DeviceMat {
                        pointer,
                        rows,
                        columns,
                    }
                })
                .collect();
            let grouped: Vec<Deviceptr> = shape
                .iter()
                .map(|rows| alloc(most_rows * rows * 4))
                .collect();
            let alone: Vec<Deviceptr> = shape
                .iter()
                .map(|rows| alloc(most_rows * rows * 4))
                .collect();
            for batch in [9, 128, 200, most_rows] {
                let projections: Vec<(&DeviceMat, Deviceptr)> =
                    matrices.iter().zip(grouped.iter().copied()).collect();
                super::gemv_group(&cuda, &kernels, &projections, input_buffer, batch)
                    .expect("the grouped launch");
                for (matrix, output) in matrices.iter().zip(&alone) {
                    super::gemv_batch(&cuda, &kernels, matrix, input_buffer, *output, batch)
                        .expect("a matrix's own launch");
                }
                for ((matrix, one), two) in matrices.iter().zip(&grouped).zip(&alone) {
                    let mut first = vec![0u8; batch * matrix.rows * 4];
                    let mut second = first.clone();
                    cuda.copy_dtoh(&mut first, one).expect("the grouped output");
                    cuda.copy_dtoh(&mut second, two).expect("the lone output");
                    assert!(
                        first == second,
                        "a {}-row matrix of {shape:?} at {batch} rows differs grouped",
                        matrix.rows
                    );
                    assert!(first.iter().any(|&byte| byte != 0));
                }
            }
        }
    }

    #[test]
    #[ignore = "needs a Dory/Qwen cubin and a CUDA device"]
    fn split_decode_attention_matches_an_f64_reference() {
        if !DoryDeviceBackend::available() {
            eprintln!("skipping the split decode attention check: this build has no Dory cubin");
            return;
        }
        let cuda = match Cuda::load() {
            Ok(cuda) => cuda,
            Err(error) => {
                eprintln!("skipping the split decode attention check: {error}");
                return;
            }
        };
        cuda.make_current().expect("the context is current");
        let kernels = Kernels::load(&cuda, HEAD_DIM).expect("the Dory kernels load");
        let q_width = HEADS * HEAD_DIM;
        let kv_width = KV_HEADS * HEAD_DIM;
        // (position, window, ring rows; 0 rows is a full-context cache): a lone key, a short and a
        // long full-attention row, a wrapped sliding ring, and a row long enough that every one
        // of the SPLIT_MAX runs is longer than the minimum.
        let cases = [
            (0, None, 0),
            (777, None, 0),
            (9000, None, 0),
            (9000, Some(1024), 1024 + super::PREFILL_BATCH),
            (70_000, None, 0),
        ];
        let rows = 70_001;
        let mut stream = Stream(7);
        let keys: Vec<u16> = (0..rows * kv_width)
            .map(|_| to_bf16(stream.next()))
            .collect();
        let values: Vec<u16> = (0..rows * kv_width)
            .map(|_| to_bf16(stream.next()))
            .collect();
        let gate: Vec<f32> = (0..q_width).map(|_| stream.next() * 2.0).collect();
        let s_gate: Vec<u16> = (0..q_width).map(|_| to_bf16(stream.next())).collect();
        let gate_factor = 1.5f32;
        let mut total = 0usize;
        let mut alloc = |bytes: usize| cuda.alloc(bytes, &mut total).expect("a test buffer");
        // K and V back to back, as a cache slot holds them.
        let key_buffer = alloc((keys.len() + values.len()) * 2);
        let value_buffer = key_buffer.offset(keys.len() * 2);
        let gate_buffer = alloc(q_width * 4);
        let s_gate_buffer = alloc(q_width * 2);
        let q_buffer = alloc(q_width * 4);
        let out_buffer = alloc(q_width * 4);
        let partials = alloc(HEADS * SPLIT_MAX * (HEAD_DIM + 2) * 4);
        cuda.copy_htod(&key_buffer, &u16_bytes(&keys))
            .expect("keys");
        cuda.copy_htod(&value_buffer, &u16_bytes(&values))
            .expect("values");
        cuda.copy_htod(&gate_buffer, &f32_bytes(&gate))
            .expect("gate");
        cuda.copy_htod(&s_gate_buffer, &u16_bytes(&s_gate))
            .expect("s_gate");
        let run = |position: usize, window: Option<usize>, ring: usize| -> Vec<f32> {
            let (_, span, splits) = decode_split(position, window);
            // One lane; the others sit the launch out.
            let mut words = [LANE_IDLE; ATTENTION_LANES];
            let mut caches = [0u64; ATTENTION_LANES];
            words[0] = lane_word(position, span).expect("the lane's word");
            caches[0] = key_buffer.0;
            let values_offset = keys.len() as u64;
            let mut args = vec![
                q_buffer.0,
                partials.0,
                HEADS as u64,
                KV_HEADS as u64,
                HEAD_DIM as u64,
                window.unwrap_or(0) as u64,
                ring as u64,
                q_width as u64,
                values_offset,
                0,
            ];
            args.extend(words);
            args.extend(caches);
            launch(
                &cuda,
                kernels.attn_split,
                (KV_HEADS as u32, splits as u32, 1),
                (256, 1, 1),
                &args,
            )
            .expect("the split launch");
            let mut args = vec![
                partials.0,
                gate_buffer.0,
                s_gate_buffer.0,
                out_buffer.0,
                HEAD_DIM as u64,
                u64::from(gate_factor.to_bits()),
                window.unwrap_or(0) as u64,
                q_width as u64,
                0,
            ];
            args.extend(words);
            launch(
                &cuda,
                kernels.attn_combine,
                (HEADS as u32, 1, 1),
                (256, 1, 1),
                &args,
            )
            .expect("the combine launch");
            let mut bytes = vec![0u8; q_width * 4];
            cuda.copy_dtoh(&mut bytes, &out_buffer).expect("the output");
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect()
        };
        let mut worst = 0.0f64;
        // A flat softmax and a sharp one (scores tens apart).
        for sharpness in [1.0f32, 24.0] {
            let q: Vec<f32> = (0..q_width).map(|_| stream.next() * sharpness).collect();
            cuda.copy_htod(&q_buffer, &f32_bytes(&q)).expect("q");
            let row = Row {
                q: &q,
                keys: &keys,
                values: &values,
                gate: &gate,
                s_gate: &s_gate,
                gate_factor,
            };
            for (position, window, ring) in cases {
                let device = run(position, window, ring);
                let expected = reference(&row, position, window, ring);
                let error = device
                    .iter()
                    .zip(&expected)
                    .map(|(device, expected)| (f64::from(*device) - expected).abs())
                    .fold(0.0f64, f64::max);
                assert!(
                    device.iter().all(|value| value.is_finite()),
                    "position {position}: a non-finite output"
                );
                eprintln!(
                    "split decode attention, sharpness {sharpness}, position {position}, window \
                     {window:?}: max |device - f64| = {error:.3e}"
                );
                worst = worst.max(error);
            }
        }
        let buffers: [&Deviceptr; 6] = [
            &key_buffer,
            &gate_buffer,
            &s_gate_buffer,
            &q_buffer,
            &out_buffer,
            &partials,
        ];
        cuda.free_buffers(&buffers);
        // The outputs are gated softmax averages of values in [-1, 1]. A flat softmax lands
        // within 1e-7 of f64 over 70,001 keys; a sharp one carries the f32 score's own rounding
        // (scores near 30, an ulp of 2e-6) into its weights, measured at 1.7e-6.
        assert!(
            worst < 1e-5,
            "max |device - f64| = {worst:.3e} exceeds 1e-5"
        );
    }

    #[test]
    #[ignore = "needs a Dory/Qwen cubin and a CUDA device"]
    fn mma_prefill_attention_matches_an_f64_reference() {
        if !DoryDeviceBackend::available() {
            eprintln!("skipping the prefill attention check: this build has no Dory cubin");
            return;
        }
        let cuda = match Cuda::load() {
            Ok(cuda) => cuda,
            Err(error) => {
                eprintln!("skipping the prefill attention check: {error}");
                return;
            }
        };
        cuda.make_current().expect("the context is current");
        let kernels = Kernels::load(&cuda, HEAD_DIM).expect("the Dory kernels load");
        let q_width = HEADS * HEAD_DIM;
        let kv_width = KV_HEADS * HEAD_DIM;
        let ring_rows = 1024 + super::PREFILL_BATCH;
        // (position, tokens, window, ring rows): a lone query, a causal chunk with a partial
        // last tile, a short chunk over a long context, a wrapped sliding ring, and chunks that
        // cross the window's edge inside themselves.
        let cases = [
            (0, 1, None, 0),
            (0, 300, None, 0),
            (5000, 37, None, 0),
            (9000, 50, Some(1024), ring_rows),
            (700, 200, Some(1024), ring_rows),
            (0, 1100, Some(1024), ring_rows),
        ];
        let rows = 10_000;
        let tokens_max = 1100;
        let mut stream = Stream(11);
        let keys: Vec<u16> = (0..rows * kv_width)
            .map(|_| to_bf16(stream.next()))
            .collect();
        let values: Vec<u16> = (0..rows * kv_width)
            .map(|_| to_bf16(stream.next()))
            .collect();
        let gate: Vec<f32> = (0..tokens_max * q_width)
            .map(|_| stream.next() * 2.0)
            .collect();
        let s_gate: Vec<u16> = (0..q_width).map(|_| to_bf16(stream.next())).collect();
        let gate_factor = 1.5f32;
        let mut total = 0usize;
        let mut alloc = |bytes: usize| cuda.alloc(bytes, &mut total).expect("a test buffer");
        let key_buffer = alloc(keys.len() * 2);
        let value_buffer = alloc(values.len() * 2);
        let gate_buffer = alloc(tokens_max * q_width * 4);
        let s_gate_buffer = alloc(q_width * 2);
        let q_buffer = alloc(tokens_max * q_width * 4);
        let out_buffer = alloc(tokens_max * q_width * 4);
        cuda.copy_htod(&key_buffer, &u16_bytes(&keys))
            .expect("keys");
        cuda.copy_htod(&value_buffer, &u16_bytes(&values))
            .expect("values");
        cuda.copy_htod(&gate_buffer, &f32_bytes(&gate))
            .expect("gate");
        cuda.copy_htod(&s_gate_buffer, &u16_bytes(&s_gate))
            .expect("s_gate");
        let run = |position: usize, tokens: usize, window: Option<usize>, ring: usize| {
            launch(
                &cuda,
                kernels.attn_mma,
                (tokens.div_ceil(PREFILL_Q) as u32, KV_HEADS as u32, 1),
                ((HEADS / KV_HEADS * 32) as u32, 1, 1),
                &[
                    q_buffer.0,
                    key_buffer.0,
                    value_buffer.0,
                    gate_buffer.0,
                    s_gate_buffer.0,
                    out_buffer.0,
                    position as u64,
                    tokens as u64,
                    HEADS as u64,
                    KV_HEADS as u64,
                    window.unwrap_or(0) as u64,
                    u64::from(gate_factor.to_bits()),
                    q_width as u64,
                    q_width as u64,
                    ring as u64,
                ],
            )
            .expect("the prefill attention launch");
            let mut bytes = vec![0u8; tokens * q_width * 4];
            cuda.copy_dtoh(&mut bytes, &out_buffer).expect("the output");
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect::<Vec<f32>>()
        };
        let mut worst = 0.0f64;
        for sharpness in [1.0f32, 24.0] {
            let q: Vec<f32> = (0..tokens_max * q_width)
                .map(|_| stream.next() * sharpness)
                .collect();
            cuda.copy_htod(&q_buffer, &f32_bytes(&q)).expect("q");
            for (position, tokens, window, ring) in cases {
                let device = run(position, tokens, window, ring);
                assert!(
                    device.iter().all(|value| value.is_finite()),
                    "position {position}: a non-finite output"
                );
                let mut error = 0.0f64;
                for (t, device) in device.chunks_exact(q_width).enumerate() {
                    let row = Row {
                        q: &q[t * q_width..(t + 1) * q_width],
                        keys: &keys,
                        values: &values,
                        gate: &gate[t * q_width..(t + 1) * q_width],
                        s_gate: &s_gate,
                        gate_factor,
                    };
                    let expected = reference(&row, position + t, window, ring);
                    for (device, expected) in device.iter().zip(&expected) {
                        error = error.max((f64::from(*device) - expected).abs());
                    }
                }
                eprintln!(
                    "prefill attention, sharpness {sharpness}, position {position}, {tokens} \
                     tokens, window {window:?}: max |device - f64| = {error:.3e}"
                );
                worst = worst.max(error);
            }
        }
        let buffers: [&Deviceptr; 6] = [
            &key_buffer,
            &value_buffer,
            &gate_buffer,
            &s_gate_buffer,
            &q_buffer,
            &out_buffer,
        ];
        cuda.free_buffers(&buffers);
        // q and the softmax weights reach the tensor cores as a BF16 head and remainder, 16
        // significant bits; the scores and sums accumulate in f32. A flat softmax lands within
        // 2e-6 of f64 and a sharp one (scores tens apart, where q's dropped bits move the
        // weights most) within 4.1e-5. The f32 CUDA-core tile this kernel replaced measured
        // 1.1e-7 and 2.1e-6 on the same rows.
        assert!(
            worst < 1e-4,
            "max |device - f64| = {worst:.3e} exceeds 1e-4"
        );
    }
}

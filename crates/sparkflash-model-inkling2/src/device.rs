// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The Inkling-2-Small device arm: the schedule of [crate::device_plan] executed on a GB10.
//!
//! The plan decides the launches and their order; this module owns the pointers, the state and
//! the argument words. Nothing here chooses a sequence of ops: it walks
//! [InklingDevicePlan::globals] and [InklingDevicePlan::layers] and dispatches each [Op] with its
//! [Target] to the entry the plan named, so a schedule change is a plan change rather than a
//! second walk kept in step by hand.
//!
//! # State
//!
//! A session holds the two key/value caches a layer, the four short-convolution histories a
//! layer, the block-summary stack and one scratch arena. A sliding layer's cache is its window
//! and the attention reads it as a ring, which is what makes the memory figure in the note a
//! computation: 8 GiB at a million tokens instead of 41.
//!
//! # Prefill
//!
//! The decode walk is the only one wired: [ModelSession::prefill] loops it a token at a time.
//! That is correct and slow, and it is what the plan's [crate::device_plan::Pass::Prefill]
//! entries are for - they name the chunked launches that replace the loop next.
//!
//! # Tensor parallelism
//!
//! One rank. The schedule already describes TP4, and the three [crate::device_plan::Collective]
//! steps are the only ones that are no-ops at world one, so opening a rank of a larger world is
//! refused rather than run without its exchanges: a rank that computed its own expert subset and
//! never reduced it would answer with a quarter of each token's MoE.

use std::sync::Arc;

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel, Module};
use sparkflash_kernels::images::{inkling_cubin, qwen4_exp_cubin};
use sparkflash_ops::gemv::CudaHandle;
use sparkflash_model::backend::{BackendError, ModelBackend, ModelSession, Sampling, StepOutput};
use sparkflash_model::{Capabilities, ModelDescriptor, TpPlan};
use sparkflash_weights::WeightSource;
use sparkflash_weights::quant::bf16_to_f32;

use crate::collective::{Exchange, Solo};
use crate::config::Inkling2Config;
use crate::device_plan::{
    BandOp, Collective, ConvSlot, ExpertBody, InklingDevicePlan, NormRole, Op, Pass, Projection,
    ResidualInput, Step, Target,
};
use crate::forward::trace_on;
use crate::host::greedy;
use crate::tensor::InklingTensorPlan;

/// The backend name the diagnostics use.
pub const DEVICE: &str = "inkling_device";

/// The threads a block of the token image's dense GEMV or RMSNorm runs.
const TOKEN_BLOCK: u32 = 256;
/// The threads the Inkling elementwise bodies run.
const ELEMENT_BLOCK: u32 = 256;
/// The threads the attention body runs: one warp a head.
const ATTENTION_BLOCK: u32 = 32;
/// The threads the router runs: one warp walks every expert in order.
const ROUTER_BLOCK: u32 = 32;
/// The column block a released MXFP8 expert stores one ue8m0 code for.
const MXFP8_BLOCK: usize = 32;

/// The blocks a flat elementwise launch takes: one a 256 elements, at least one and at most the
/// grid limit, so a launch never asks for a zero-sized grid.
fn flat(elements: usize) -> u32 {
    u32::try_from(elements.div_ceil(ELEMENT_BLOCK as usize).max(1))
        .unwrap_or(u32::MAX)
        .min(65_535)
}

/// The blocks a dense GEMV takes: the kernel strides rows over its warps times the grid's blocks.
fn gemv_grid(rows: usize) -> u32 {
    let warps = (TOKEN_BLOCK / 32) as usize;
    u32::try_from(rows.div_ceil(warps).max(1))
        .unwrap_or(u32::MAX)
        .min(4_096)
}

/// One f32 argument as the driver's one-word cell.
fn bits(value: f32) -> u64 {
    u64::from(value.to_bits())
}

/// A host vector of f32 as the little-endian bytes the driver copies.
fn row_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// The f32 values of a little-endian byte buffer.
fn from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

/// One loaded symbol.
#[derive(Debug, Clone, Copy)]
struct Entry {
    kernel: Kernel,
}

/// Every kernel the walk launches, loaded by the name the plan records.
struct Kernels {
    /// The modules the symbols came from. A dropped module unloads its symbols, so the backend
    /// keeps them for as long as it launches.
    _modules: Vec<Module>,
    embed: Entry,
    rmsnorm: Entry,
    gemv: Entry,
    block_residual: Entry,
    short_conv: Entry,
    short_conv_dynamic: Entry,
    kv_store: Entry,
    attention: Entry,
    router: Entry,
    expert_nvfp4: Entry,
    expert_mxfp8: Entry,
    swiglu: Entry,
    expert_mix: Entry,
    residual_add: Entry,
    logit_softcap: Entry,
    widen: Entry,
}

impl Kernels {
    fn load(cuda: &Cuda) -> Result<Self, String> {
        let inkling = inkling_cubin();
        let token = qwen4_exp_cubin();
        if inkling.is_empty() {
            return Err("this build has no Inkling cubin: rebuild on a host with nvcc (or set \
                        SPARKFLASH_NVCC) so build.rs compiles kernels/inkling_sm121.cu"
                .to_owned());
        }
        if token.is_empty() {
            return Err("this build has no token-image cubin: the arm's embedding, RMSNorm and \
                        dense projections are kernels/qwen4_exp_sm121.cu"
                .to_owned());
        }
        let mut modules = Vec::new();
        let mut load = |cubin: &'static [u8], name: &'static str| -> Result<Entry, String> {
            let (module, kernel) = cuda.load_named(cubin, name)?;
            modules.push(module);
            Ok(Entry { kernel })
        };
        Ok(Self {
            embed: load(token, "q4e_embed")?,
            rmsnorm: load(token, "q4e_rmsnorm")?,
            gemv: load(token, "q4e_gemv_bf16")?,
            block_residual: load(inkling, "inkling_block_residual")?,
            short_conv: load(inkling, "inkling_short_conv_decode")?,
            short_conv_dynamic: load(inkling, "inkling_short_conv_dynamic_decode")?,
            kv_store: load(inkling, "inkling_kv_store")?,
            attention: load(inkling, "inkling_attn_decode")?,
            router: load(inkling, "inkling_router_topk")?,
            expert_nvfp4: load(inkling, "inkling_nvfp4_expert_matvec")?,
            expert_mxfp8: load(inkling, "inkling_mxfp8_expert_matvec")?,
            swiglu: load(inkling, "inkling_softcapped_swiglu")?,
            expert_mix: load(inkling, "inkling_expert_mix")?,
            residual_add: load(inkling, "inkling_residual_add")?,
            logit_softcap: load(inkling, "inkling_logit_softcap")?,
            widen: load(inkling, "inkling_widen_f32")?,
            _modules: modules,
        })
    }
}

// SAFETY: a Kernel is a driver function handle plus its name. The driver serialises context
// access and every launch in this module happens on the process's default stream, so moving the
// handle between threads cannot race a launch; the same argument the Dory and GLM kernels sets
// make for their own handles.
unsafe impl Send for Kernels {}
// SAFETY: as above.
unsafe impl Sync for Kernels {}

/// One served slot on the device.
#[derive(Debug, Clone, Copy)]
struct Slot {
    pointer: Deviceptr,
    bytes: usize,
}

/// The rank's weights, in the served plan's order.
struct Weights {
    slots: Vec<Slot>,
    /// [hidden] zeros, for a launch that names a term this arm does not use.
    zeros: Deviceptr,
    total_bytes: usize,
}

impl Weights {
    fn get(&self, index: usize) -> Slot {
        self.slots[index]
    }
}

/// The two widened L2-clip gain vectors a layer's attention reads.
#[derive(Debug, Clone, Copy)]
struct ClipScales {
    query: Deviceptr,
    relative: Deviceptr,
}

/// The per-expert scale a stored format puts on an expert's own output.
///
/// An NVFP4 expert stores amax over its outer-scale denominator, and that factor is
/// linear, so it rides on the expert's output: the arm multiplies it into the mix's share and no
/// kernel needs a per-expert amax. A band without one records nothing.
struct ExpertScale {
    gate_up: Vec<f32>,
    down: Vec<f32>,
}

/// Everything one backend shares with its sessions.
///
/// The driver handle is the shared wrapper rather than the raw context: the context holds a
/// process-wide pointer that is neither Send nor Sync, and the backend has to be both because a
/// serving front end opens sessions from another thread.
struct Shared {
    cuda: CudaHandle,
    plan: InklingDevicePlan,
    config: Inkling2Config,
    kernels: Kernels,
    weights: Weights,
    clip: Vec<ClipScales>,
    clip_arena: Deviceptr,
    experts: Vec<Option<ExpertScale>>,
    context: usize,
    exchange: Box<dyn Exchange>,
}

impl Shared {
    fn hidden(&self) -> usize {
        self.config.hidden_size
    }

    fn kv_width(&self) -> usize {
        self.plan.rank_kv_heads() * self.config.head_dim
    }

    fn heads(&self) -> usize {
        self.plan.rank_heads()
    }

    fn kernel_width(&self) -> usize {
        self.config.sconv_kernel_size
    }

    fn launch(
        &self,
        entry: Entry,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), String> {
        self.cuda
            .launch(entry.kernel, grid, block, args)
            .map_err(|error| format!("{}: {error}", entry.kernel.name()))
    }

    /// One dense BF16 projection of one row.
    fn gemv(
        &self,
        weight: Deviceptr,
        input: Deviceptr,
        out: Deviceptr,
        rows: usize,
        columns: usize,
    ) -> Result<(), String> {
        self.launch(
            self.kernels.gemv,
            (gemv_grid(rows), 1, 1),
            (TOKEN_BLOCK, 1, 1),
            &[weight.0, input.0, out.0, rows as u64, columns as u64, 1],
        )
    }

    /// One RMSNorm of one row.
    fn rmsnorm(
        &self,
        input: Deviceptr,
        weight: Deviceptr,
        out: Deviceptr,
        rows: usize,
        width: usize,
    ) -> Result<(), String> {
        self.launch(
            self.kernels.rmsnorm,
            (rows as u32, 1, 1),
            (TOKEN_BLOCK, 1, 1),
            &[
                input.0,
                weight.0,
                out.0,
                width as u64,
                width as u64,
                bits(self.config.architecture.norm_epsilon as f32),
                0,
                0,
            ],
        )
    }

    /// The block-attention residual over the summary stack plus the current stream.
    fn block_residual(
        &self,
        summaries: Deviceptr,
        partial: Deviceptr,
        queries: Deviceptr,
        out: Deviceptr,
        blocks: usize,
        queries_count: usize,
    ) -> Result<(), String> {
        self.launch(
            self.kernels.block_residual,
            (queries_count as u32, 1, 1),
            (ELEMENT_BLOCK, 1, 1),
            &[
                summaries.0,
                partial.0,
                queries.0,
                out.0,
                blocks as u64,
                self.hidden() as u64,
                queries_count as u64,
                bits(self.config.architecture.norm_epsilon as f32),
            ],
        )
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        let mut pointers: Vec<Deviceptr> =
            self.weights.slots.iter().map(|slot| slot.pointer).collect();
        pointers.push(self.weights.zeros);
        pointers.push(self.clip_arena);
        let refs: Vec<&Deviceptr> = pointers.iter().collect();
        self.cuda.free_buffers(&refs);
    }
}

/// A device backend over one Inkling-2-Small rank.
pub struct InklingDeviceBackend {
    descriptor: ModelDescriptor,
    shared: Arc<Shared>,
}

impl std::fmt::Debug for InklingDeviceBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InklingDeviceBackend")
            .field("descriptor", &self.descriptor.id)
            .field("context", &self.shared.context)
            .field("weight_bytes", &self.shared.weights.total_bytes)
            .finish()
    }
}

impl InklingDeviceBackend {
    /// Whether this build carries both cubins the walk needs.
    #[must_use]
    pub fn available() -> bool {
        !inkling_cubin().is_empty() && !qwen4_exp_cubin().is_empty()
    }

    /// Open the arm over a checkpoint source, at rank zero of a world of one.
    ///
    /// # Errors
    /// A missing cubin, a context outside the model window, a checkpoint whose block residual is
    /// not normalised, or a weight the plan names and the source cannot produce.
    pub fn open(
        descriptor: ModelDescriptor,
        config: Inkling2Config,
        source: Arc<dyn WeightSource>,
        context: usize,
    ) -> Result<Self, String> {
        let tp = TpPlan::new(1, 0).map_err(|error| error.detail)?;
        Self::open_rank(descriptor, config, source, context, tp, Box::new(Solo))
    }

    /// Open one rank of a larger world, exchanging through the given collective.
    ///
    /// The three collectives the schedule names are the only steps whose behaviour depends on the
    /// world: the embedding row is gathered from its owner, the routed experts' mix is reduced
    /// across the ranks that computed it, and the logit slices are gathered before the softcap.
    ///
    /// # Errors
    /// As [Self::open], and a plan the world does not divide.
    pub fn open_rank(
        descriptor: ModelDescriptor,
        config: Inkling2Config,
        source: Arc<dyn WeightSource>,
        context: usize,
        tp: TpPlan,
        exchange: Box<dyn Exchange>,
    ) -> Result<Self, String> {
        if !Self::available() {
            return Err("this build carries no Inkling or token cubin: rebuild on a host with nvcc \
                        (or set SPARKFLASH_NVCC)"
                .to_owned());
        }
        let window = config.context_tokens();
        if context == 0 || (window != 0 && context > window) {
            return Err(format!(
                "context {context} is outside 1..={window} for {}",
                descriptor.id
            ));
        }
        if !config.attn_res_normalize_values {
            return Err("the released block residual normalises its summary stack; a checkpoint \
                        that clears attn_res_normalize_values has no kernel in this arm"
                .to_owned());
        }
        if exchange.world() != tp.world() || exchange.rank() != tp.rank() {
            return Err(format!(
                "the collective is rank {}/{} and the plan rank {}/{}",
                exchange.rank(),
                exchange.world(),
                tp.rank(),
                tp.world()
            ));
        }
        let plan = InklingDevicePlan::build(&config, tp, Pass::Decode, 1, context)?;
        let cuda = CudaHandle::new(Cuda::load()?);
        let kernels = Kernels::load(&cuda)?;
        let weights = upload(&cuda, source.as_ref(), plan.served())?;
        let (clip, clip_arena) = widen_clips(&cuda, &kernels, &plan, &weights)?;
        let experts = expert_scales(&cuda, &plan, &weights)?;
        if trace_on() {
            eprintln!("[inkling] device arm: {}", plan.boot_summary());
        }
        Ok(Self {
            descriptor,
            shared: Arc::new(Shared {
                cuda,
                plan,
                config,
                kernels,
                weights,
                clip,
                clip_arena,
                experts,
                context,
                exchange,
            }),
        })
    }

    /// The schedule this arm executes.
    #[must_use]
    pub fn plan(&self) -> &InklingDevicePlan {
        &self.shared.plan
    }

    /// The bytes the weights occupy on the device.
    #[must_use]
    pub fn weight_bytes(&self) -> usize {
        self.shared.weights.total_bytes
    }
}

impl ModelBackend for InklingDeviceBackend {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> Capabilities {
        // The text backbone runs. The tower and the drafter are in the checkpoint and in this
        // tree's host reference, but neither has a device path yet.
        Capabilities {
            vision: false,
            ..self.descriptor.capabilities
        }
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        // A CUDA context is per-thread: [Cuda::load] makes the primary context current on the
        // thread that opened the backend, and the generic host driver opens a sequence on its own
        // worker thread. Without this bind the first allocation answers CUDA_ERROR_INVALID_CONTEXT
        // (201) - which is exactly how the four-rank server failed its first request.
        self.shared
            .cuda
            .make_current()
            .map_err(|error| BackendError::new(DEVICE, error))?;
        let state =
            State::allocate(&self.shared).map_err(|error| BackendError::new(DEVICE, error))?;
        Ok(Box::new(InklingDeviceSession {
            shared: Arc::clone(&self.shared),
            state,
            history: Vec::new(),
            last_logits: Vec::new(),
            cached: 0,
            sampling: Sampling::default(),
        }))
    }
}

/// Upload every served slot, refusing a source whose bytes disagree with the plan.
fn upload(
    cuda: &Cuda,
    source: &dyn WeightSource,
    plan: &InklingTensorPlan,
) -> Result<Weights, String> {
    let specs = plan.plan().specs();
    let mut slots = Vec::with_capacity(specs.len());
    let mut total = 0usize;
    for index in 0..specs.len() {
        let spec = plan.rank_spec_at(index);
        let bytes = usize::try_from(spec.storage_bytes())
            .map_err(|_| format!("{}: {} bytes past usize", spec.name, spec.storage_bytes()))?;
        let data = source
            .read(&spec.name)
            .map_err(|error| format!("{}: {error}", spec.name))?;
        if data.len() != bytes {
            return Err(format!(
                "{}: the source holds {} bytes and the rank plan {bytes}",
                spec.name,
                data.len()
            ));
        }
        let pointer = cuda.alloc(bytes.max(1), &mut total)?;
        cuda.copy_htod(&pointer, &data)?;
        slots.push(Slot { pointer, bytes });
    }
    let hidden = plan
        .index_of("model.llm.norm.weight")
        .map_or(4096, |index| plan.rank_spec_at(index).shape[0]);
    let zeros_bytes = hidden.max(1) * std::mem::size_of::<f32>();
    let mut zeros_total = 0usize;
    let zeros = cuda.alloc(zeros_bytes, &mut zeros_total)?;
    cuda.memset_d8_async(zeros, 0, zeros_bytes)?;
    cuda.synchronize()?;
    let _ = zeros_bytes;
    Ok(Weights {
        slots,
        zeros,
        total_bytes: total + zeros_total,
    })
}

/// Widen every layer's two L2-clip gain vectors from the checkpoint's BF16 to the f32 the
/// attention body reads.
///
/// The released q_l2_clip.scale is [heads][head_dim] and rel_l2_clip.scale is [heads][d_rel],
/// both BF16; the kernel's arguments are float pointers because the gains multiply a vector that
/// is already f32. The widening is exact, so this is the checkpoint's own value and not a
/// re-quantisation.
fn widen_clips(
    cuda: &Cuda,
    kernels: &Kernels,
    plan: &InklingDevicePlan,
    weights: &Weights,
) -> Result<(Vec<ClipScales>, Deviceptr), String> {
    let heads = plan.rank_heads();
    let head_dim = plan.config().head_dim;
    let d_rel = plan.config().d_rel;
    let per_layer = heads * head_dim + heads * d_rel;
    let mut total = 0usize;
    let bytes = plan.layers().len().max(1) * per_layer * std::mem::size_of::<f32>();
    let arena = cuda.alloc(bytes, &mut total)?;
    cuda.memset_d8_async(arena, 0, bytes)?;
    cuda.synchronize()?;
    let mut clip = Vec::with_capacity(plan.layers().len());
    for layer in plan.layers() {
        let prefix = format!("model.llm.layers.{}.attn", layer.index);
        let query_name = format!("{prefix}.q_l2_clip.scale");
        let relative_name = format!("{prefix}.rel_l2_clip.scale");
        let query_index = plan
            .served()
            .index_of(&query_name)
            .ok_or_else(|| format!("{query_name} is not a served tensor"))?;
        let relative_index = plan
            .served()
            .index_of(&relative_name)
            .ok_or_else(|| format!("{relative_name} is not a served tensor"))?;
        let query = arena.offset(layer.index * per_layer * 4);
        let relative = arena.offset(layer.index * per_layer * 4 + heads * head_dim * 4);
        for (source, destination, count) in [
            (query_index, query, heads * head_dim),
            (relative_index, relative, heads * d_rel),
        ] {
            cuda.launch(
                kernels.widen.kernel,
                (flat(count), 1, 1),
                (ELEMENT_BLOCK, 1, 1),
                &[weights.get(source).pointer.0, destination.0, count as u64],
            )
            .map_err(|error| format!("widen {source}: {error}"))?;
        }
        clip.push(ClipScales { query, relative });
    }
    cuda.synchronize()?;
    Ok((clip, arena))
}

/// Read the released per-expert scales off the device once, so a token's band does not have to.
fn expert_scales(
    cuda: &Cuda,
    plan: &InklingDevicePlan,
    weights: &Weights,
) -> Result<Vec<Option<ExpertScale>>, String> {
    let mut table = Vec::with_capacity(plan.layers().len());
    for layer in plan.layers() {
        if layer.experts != ExpertBody::NvFp4 {
            table.push(None);
            continue;
        }
        let prefix = format!("model.llm.layers.{}.mlp.experts", layer.index);
        let read = |suffix: &str| -> Result<Vec<f32>, String> {
            let name = format!("{prefix}.{suffix}");
            let index = plan
                .served()
                .index_of(&name)
                .ok_or_else(|| format!("{name} is not a served tensor"))?;
            let slot = weights.get(index);
            let mut bytes = vec![0u8; slot.bytes];
            cuda.copy_dtoh(&mut bytes, &slot.pointer)
                .map_err(|error| format!("{name}: {error}"))?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect())
        };
        // ModelOpt's NVFP4 outer scale is amax over the outer-scale denominator, which already
        // carries the six times the E4M3 maximum: the same convention the host forward's
        // nvfp4_outer_scale records.
        let scale = |amax: Vec<f32>, denominator: Vec<f32>| -> Vec<f32> {
            let denominator = denominator
                .first()
                .copied()
                .filter(|value| *value != 0.0)
                .unwrap_or(1.0);
            amax.into_iter()
                .map(|value| value / denominator)
                .collect()
        };
        let gate_up = scale(
            read("w13_weight_amax")?,
            read("w13_weight_outer_scale_denominator")?,
        );
        let down = scale(
            read("w2_weight_amax")?,
            read("w2_weight_outer_scale_denominator")?,
        );
        table.push(Some(ExpertScale { gate_up, down }));
    }
    Ok(table)
}

/// The bytes one expert's stored codes occupy.
fn codes_bytes(rows: usize, columns: usize, body: ExpertBody) -> usize {
    match body {
        ExpertBody::Bf16 => rows * columns * 2,
        ExpertBody::NvFp4 => rows * columns.div_ceil(2),
        ExpertBody::MxFp8 => rows * columns,
    }
}

/// The bytes one expert's block scales occupy.
fn scales_bytes(rows: usize, columns: usize, body: ExpertBody) -> usize {
    match body {
        ExpertBody::Bf16 => 0,
        ExpertBody::NvFp4 => rows * columns.div_ceil(16),
        ExpertBody::MxFp8 => rows * columns.div_ceil(MXFP8_BLOCK),
    }
}

/// One session's device state.
struct State {
    keys: Vec<Deviceptr>,
    values: Vec<Deviceptr>,
    conv: Vec<[Deviceptr; 4]>,
    summaries: Deviceptr,
    summaries_bytes: usize,
    summary_rows: usize,
    blocks: usize,
    scratch: Scratch,
    position: usize,
}

impl State {
    fn allocate(shared: &Shared) -> Result<Self, String> {
        let plan = &shared.plan;
        let cuda = &shared.cuda;
        let hidden = shared.hidden();
        let mut total = 0usize;
        let kv_bytes = |rows: usize, width: usize| rows.max(1) * width * 2;
        let mut keys = Vec::with_capacity(plan.layers().len());
        let mut values = Vec::with_capacity(plan.layers().len());
        for layer in plan.layers() {
            let bytes = kv_bytes(layer.kv_rows, shared.kv_width());
            let key = cuda.alloc(bytes, &mut total)?;
            let value = cuda.alloc(bytes, &mut total)?;
            cuda.memset_d8_async(key, 0, bytes)?;
            cuda.memset_d8_async(value, 0, bytes)?;
            keys.push(key);
            values.push(value);
        }
        let history = shared.config.conv_history().max(1);
        let mut conv = Vec::with_capacity(plan.layers().len());
        for _ in 0..plan.layers().len() {
            let mut slots = [Deviceptr(0); 4];
            let widths = [
                shared.kv_width(),
                shared.kv_width(),
                hidden,
                hidden,
            ];
            for (slot, width) in widths.into_iter().enumerate() {
                let bytes = width.max(1) * history * std::mem::size_of::<f32>();
                let pointer = cuda.alloc(bytes, &mut total)?;
                cuda.memset_d8_async(pointer, 0, bytes)?;
                slots[slot] = pointer;
            }
            conv.push(slots);
        }
        // Eight summaries for the released block size, the count the plan's buffer layout uses.
        let summaries_rows = plan
            .layers()
            .iter()
            .filter(|layer| layer.index.is_multiple_of(shared.config.n_layers_per_attn_res_block.max(1)))
            .count()
            .max(1);
        let summaries_bytes = summaries_rows * hidden * std::mem::size_of::<f32>();
        let summaries = cuda.alloc(summaries_bytes, &mut total)?;
        cuda.memset_d8_async(summaries, 0, summaries_bytes)?;
        let scratch = Scratch::allocate(shared, &mut total)?;
        cuda.synchronize()?;
        Ok(Self {
            keys,
            values,
            conv,
            summaries,
            summaries_bytes,
            summary_rows: summaries_rows,
            blocks: 0,
            scratch,
            position: 0,
        })
    }

    fn reset(&mut self, shared: &Shared) -> Result<(), String> {
        for (index, (key, value)) in self.keys.iter().zip(&self.values).enumerate() {
            let bytes = shared.plan.layers()[index].kv_rows.max(1) * shared.kv_width() * 2;
            shared.cuda.memset_d8_async(*key, 0, bytes)?;
            shared.cuda.memset_d8_async(*value, 0, bytes)?;
        }
        let history = shared.config.conv_history().max(1);
        for slots in &self.conv {
            // The key/value histories are the cache width; the attention and MLP ones the hidden
            // width. Only the wider of the two matters: a memset of the larger size is safe for
            // both because each slot owns its own allocation.
            let widest = shared.hidden().max(shared.kv_width());
            let bytes = widest * history * std::mem::size_of::<f32>();
            for (slot, width) in [
                shared.kv_width(),
                shared.kv_width(),
                shared.hidden(),
                shared.hidden(),
            ]
            .into_iter()
            .enumerate()
            {
                let own = width * history * std::mem::size_of::<f32>();
                debug_assert!(own <= bytes);
                shared.cuda.memset_d8_async(slots[slot], 0, own)?;
            }
        }
        shared
            .cuda
            .memset_d8_async(self.summaries, 0, self.summaries_bytes)?;
        self.blocks = 0;
        self.position = 0;
        Ok(())
    }
}

/// The scratch arena one session allocates.
struct Scratch {
    x: Deviceptr,
    rows: Deviceptr,
    normed: Deviceptr,
    q: Deviceptr,
    gate: Deviceptr,
    attn_bf16: Deviceptr,
    attn: Deviceptr,
    k: Deviceptr,
    v: Deviceptr,
    k_next: Deviceptr,
    v_next: Deviceptr,
    taps: Deviceptr,
    rel: Deviceptr,
    sink: Deviceptr,
    delta: Deviceptr,
    router_logits: Deviceptr,
    indices: Deviceptr,
    shares: Deviceptr,
    band: Deviceptr,
    activation: Deviceptr,
    partials: Deviceptr,
    scales: Deviceptr,
    mixed: Deviceptr,
    logits: Deviceptr,
    logits_full: Deviceptr,
    logits_bf16: Deviceptr,
    final_row: Deviceptr,
}

impl Scratch {
    fn allocate(shared: &Shared, total: &mut usize) -> Result<Self, String> {
        let plan = &shared.plan;
        let config = &shared.config;
        let cuda = &shared.cuda;
        let hidden = shared.hidden();
        let heads = shared.heads();
        let head_dim = config.head_dim;
        let kv_width = shared.kv_width();
        let activation = config.moe_intermediate_size;
        let top_k = config.num_experts_per_tok;
        let vocab = plan.rank_vocab();
        let taps = plan.rank_kv_heads() * config.sconv_kernel_size;
        let mut cursor = 0usize;
        let mut take = |elements: usize, width: usize| {
            let at = cursor;
            cursor += (elements.max(1) * width).max(1);
            at
        };
        let x = take(hidden, 4);
        let rows = take(3 * hidden, 4);
        let normed = take(3 * hidden, 4);
        let q = take(heads * head_dim, 4);
        let gate = take(heads * head_dim, 4);
        let attn_bf16 = take(heads * head_dim, 2);
        let attn = take(heads * head_dim, 4);
        let k = take(kv_width, 4);
        let v = take(kv_width, 4);
        let k_next = take(kv_width, 4);
        let v_next = take(kv_width, 4);
        let taps_buffer = take(taps, 4);
        let rel = take(heads * config.d_rel, 4);
        let sink = take(heads, 4);
        let delta = take(hidden, 4);
        let router_logits = take(config.n_routed_experts, 4);
        let indices = take(top_k, 4);
        let shares = take(top_k, 4);
        let band = take(top_k * 2 * activation, 4);
        let activation_buffer = take(top_k * activation, 4);
        let partials = take(top_k * hidden, 4);
        let scales = take(top_k, 4);
        let mixed = take(hidden, 4);
        let logits = take(vocab.max(1), 4);
        let logits_full = take(vocab.max(1) * shared.exchange.world(), 4);
        // The softcap trims the whole gathered row, so the released bf16 row is as wide as the
        // gathered vocabulary and not as wide as this rank's slice of it.
        let logits_bf16 = take(vocab.max(1) * shared.exchange.world(), 2);
        let final_row = take(hidden, 4);
        let arena = cuda.alloc(cursor, total)?;
        cuda.memset_d8_async(arena, 0, cursor)?;
        let at = |offset: usize| arena.offset(offset);
        Ok(Self {
            x: at(x),
            rows: at(rows),
            normed: at(normed),
            q: at(q),
            gate: at(gate),
            attn_bf16: at(attn_bf16),
            attn: at(attn),
            k: at(k),
            v: at(v),
            k_next: at(k_next),
            v_next: at(v_next),
            taps: at(taps_buffer),
            rel: at(rel),
            sink: at(sink),
            delta: at(delta),
            router_logits: at(router_logits),
            indices: at(indices),
            shares: at(shares),
            band: at(band),
            activation: at(activation_buffer),
            partials: at(partials),
            scales: at(scales),
            mixed: at(mixed),
            logits: at(logits),
            logits_full: at(logits_full),
            logits_bf16: at(logits_bf16),
            final_row: at(final_row),
        })
    }
}

/// One serving session.
struct InklingDeviceSession {
    shared: Arc<Shared>,
    state: State,
    history: Vec<u32>,
    last_logits: Vec<f32>,
    cached: usize,
    sampling: Sampling,
}

impl ModelSession for InklingDeviceSession {
    fn set_sampling(&mut self, sampling: Sampling) {
        self.sampling = sampling;
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::new("request", "the prompt is empty"));
        }
        let start = if tokens.starts_with(&self.history) {
            self.history.len()
        } else {
            self.state
                .reset(&self.shared)
                .map_err(|error| BackendError::new(DEVICE, error))?;
            self.history.clear();
            self.last_logits.clear();
            0
        };
        self.cached = start;
        let remaining = tokens.len().saturating_sub(start);
        if self.state.position + remaining > self.shared.context {
            return Err(BackendError::new(
                "context",
                format!(
                    "the prompt needs {remaining} positions, {} remain",
                    self.shared.context.saturating_sub(self.state.position)
                ),
            ));
        }
        if remaining == 0 && !self.last_logits.is_empty() {
            return Ok(self.output(self.last_logits.clone()));
        }
        let mut logits = Vec::new();
        for token in &tokens[start..] {
            logits = self.step(*token)?;
            self.history.push(*token);
        }
        self.last_logits = logits.clone();
        Ok(self.output(logits))
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        if self.state.position >= self.shared.context {
            return Err(BackendError::new("context", "the context is exhausted"));
        }
        let logits = self.step(token)?;
        self.history.push(token);
        self.last_logits = logits.clone();
        self.cached = 0;
        Ok(self.output(logits))
    }

    fn reusable_prefix(&self) -> Option<&[u32]> {
        Some(&self.history)
    }

    fn cached_tokens(&self) -> usize {
        self.cached
    }
}

impl InklingDeviceSession {
    fn output(&self, logits: Vec<f32>) -> StepOutput {
        let next_token = greedy(&logits, &self.sampling);
        StepOutput {
            logits,
            next_token,
            draft: Vec::new(),
        }
    }

    /// A device refusal, with no borrow of the session: the walk mutates the state while it
    /// reports, so the error path must not hold a reference to it.
    fn fail(error: impl std::fmt::Display) -> BackendError {
        BackendError::new(DEVICE, error.to_string())
    }

    fn weight(&self, index: usize) -> Deviceptr {
        self.shared.weights.get(index).pointer
    }

    /// One all-reduce of a residual-width f32 vector, in place.
    ///
    /// Every collective this arm performs over the hidden stream is a sum of partial sums, so the
    /// exchange is the same staged round-trip whatever produced the partial: a download, the
    /// transport's sum, an upload.
    fn all_reduce(
        &self,
        shared: &Shared,
        pointer: Deviceptr,
        elements: usize,
    ) -> Result<(), BackendError> {
        if shared.exchange.world() == 1 {
            return Ok(());
        }
        let mut bytes = row_bytes(&vec![0f32; elements]);
        shared
            .cuda
            .copy_dtoh(&mut bytes, &pointer)
            .map_err(Self::fail)?;
        let mut values = from_bytes(&bytes);
        shared
            .exchange
            .all_reduce_f32(&mut values)
            .map_err(Self::fail)?;
        shared
            .cuda
            .copy_htod(&pointer, &row_bytes(&values))
            .map_err(Self::fail)?;
        Ok(())
    }

    /// The root-mean-square of one f32 buffer, for the trace. A debug read: it synchronises.
    fn rms(&self, shared: &Shared, label: &str, pointer: Deviceptr, elements: usize) -> Result<(), BackendError> {
        if !trace_on() {
            return Ok(());
        }
        let mut bytes = vec![0u8; elements * 4];
        shared
            .cuda
            .copy_dtoh(&mut bytes, &pointer)
            .map_err(Self::fail)?;
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
        let rms = (values.iter().map(|value| value * value).sum::<f32>() / elements as f32).sqrt();
        eprintln!("[inkling] {label} rms={rms:.5} first={:.5}", values[0]);
        Ok(())
    }

    /// One token: the whole schedule, in the plan's order.
    fn step(&mut self, token: u32) -> Result<Vec<f32>, BackendError> {
        let shared = Arc::clone(&self.shared);
        let config = &shared.config;
        let plan = &shared.plan;
        if token as usize >= config.vocab_size {
            return Err(Self::fail(format!(
                "token {token} is outside the vocabulary {}",
                config.vocab_size
            )));
        }
        let position = self.state.position;
        // The block-summary stack is this token's stream at each block boundary, not a history:
        // the previous token's summaries are a different residual chain.
        self.state.blocks = 0;
        let hidden = shared.hidden();
        let kv_width = shared.kv_width();
        let heads = shared.heads();
        let head_dim = config.head_dim;
        let groups = heads / plan.rank_kv_heads().max(1);
        let run = |result: Result<(), String>| result.map_err(Self::fail);

        for step in plan.prologue() {
            match (step.op, step.target) {
                (Op::Embed, _) => {
                    // The embedding is vocabulary-parallel: exactly one rank holds this token's
                    // row, and the gather below sums the row it widens with the zeros the others
                    // contribute.
                    let rank_vocab = plan.rank_vocab();
                    let first = plan.tp().rank() * rank_vocab;
                    let local = (token as usize).checked_sub(first);
                    let weight = self.weight(step.weights[0]);
                    match local {
                        Some(row) if row < rank_vocab => run(shared.launch(
                            shared.kernels.embed,
                            (flat(hidden), 1, 1),
                            (TOKEN_BLOCK, 1, 1),
                            &[
                                weight.0,
                                self.state.scratch.x.0,
                                row as u64,
                                hidden as u64,
                                hidden as u64,
                                0,
                            ],
                        ))?,
                        _ => run(shared
                            .cuda
                            .memset_d8_async(
                                self.state.scratch.x,
                                0,
                                hidden * std::mem::size_of::<f32>(),
                            )
                            .map_err(|error| error.to_string()))?,
                    }
                }
                (Op::RmsNorm, Target::Norm(NormRole::Embedding)) => {
                    let weight = self.weight(step.weights[0]);
                    run(shared.rmsnorm(self.state.scratch.x, weight, self.state.scratch.x, 1, hidden))?;
                    self.rms(&shared, "embed_norm", self.state.scratch.x, hidden)?;
                }
                (Op::Collective(Collective::EmbeddingRow), _) => {
                    let mut bytes = row_bytes(&vec![0f32; hidden]);
                    shared
                        .cuda
                        .copy_dtoh(&mut bytes, &self.state.scratch.x)
                        .map_err(Self::fail)?;
                    let mut row = from_bytes(&bytes);
                    shared.exchange.all_reduce_f32(&mut row).map_err(Self::fail)?;
                    shared
                        .cuda
                        .copy_htod(&self.state.scratch.x, &row_bytes(&row))
                        .map_err(Self::fail)?;
                    self.rms(&shared, "embed_row", self.state.scratch.x, hidden)?;
                }
                (Op::Collective(_), _) | (Op::SummaryPush, _) => {}
                (op, target) => {
                    return Err(Self::fail(format!(
                        "the schedule's prologue {op:?} at {target:?} has no device dispatch"
                    )));
                }
            }
        }


        for (index, schedule) in plan.layers().iter().enumerate() {
            let layer = schedule.index;
            if trace_on() {
                eprintln!("[inkling] layer {layer} sliding={} blocks={}", schedule.sliding, self.state.blocks);
            }
            let sliding = schedule.sliding;
            let cache_rows = schedule.kv_rows;
            let clips = shared.clip[index];
            let mut previous_conv: Option<ConvSlot> = None;
            let mut cursor = 0usize;
            while cursor < schedule.steps.len() {
                let step = &schedule.steps[cursor];
                if trace_on() {
                    eprintln!("[inkling]   {}/{:?}", step.op.name(), step.target);
                }
                match (step.op, step.target) {
                    (Op::BlockResidual, Target::Input(ResidualInput::Attention)) => {
                        let queries = self.weight(step.weights[0]);
                        run(shared.block_residual(
                            self.state.summaries,
                            self.state.scratch.x,
                            queries,
                            self.state.scratch.rows,
                            self.state.blocks,
                            if sliding { 1 } else { 3 },
                        ))?;
                    }
                    (Op::SummaryPush, _) => {
                        if self.state.blocks >= self.state.summary_rows {
                            return Err(Self::fail(format!(
                                "layer {layer}: the summary stack is full ({} rows) and the \
                                 plan pushes another",
                                self.state.summary_rows
                            )));
                        }
                        let destination = self
                            .state
                            .summaries
                            .offset(self.state.blocks * hidden * std::mem::size_of::<f32>());
                        run(shared
                            .cuda
                            .copy_dtod(
                                &destination,
                                &self.state.scratch.x,
                                hidden * std::mem::size_of::<f32>(),
                            )
                            .map_err(|error| error.to_string()))?;
                        self.state.blocks += 1;
                        // At a boundary the sublayer output *is* the stream: there is no additive
                        // residual and the pre-attention stream is now a summary.
                        run(shared
                            .cuda
                            .copy_dtod(
                                &self.state.scratch.x,
                                &self.state.scratch.delta,
                                hidden * std::mem::size_of::<f32>(),
                            )
                            .map_err(|error| error.to_string()))?;
                    }
                    (Op::RmsNorm, Target::Norm(role)) => {
                        let weight = self.weight(step.weights[0]);
                        let (source, destination) = match (role, sliding) {
                            (NormRole::Stream, _) => {
                                (self.state.scratch.x, self.state.scratch.normed)
                            }
                            (NormRole::Attention, _) => {
                                (self.state.scratch.rows, self.state.scratch.normed)
                            }
                            (NormRole::Query, false) => (
                                self.state.scratch.rows,
                                self.state.scratch.normed,
                            ),
                            (NormRole::Key, false) => (
                                self.state.scratch.rows.offset(hidden * 4),
                                self.state.scratch.normed.offset(hidden * 4),
                            ),
                            (NormRole::Value, false) => (
                                self.state.scratch.rows.offset(2 * hidden * 4),
                                self.state.scratch.normed.offset(2 * hidden * 4),
                            ),
                            (NormRole::Mlp, _) => {
                                (self.state.scratch.rows, self.state.scratch.normed)
                            }
                            (other, _) => {
                                return Err(Self::fail(format!(
                                    "layer {layer}: norm role {other:?} is not an input norm"
                                )));
                            }
                        };
                        run(shared.rmsnorm(source, weight, destination, 1, hidden))?;
                    }
                    (Op::Gemv, Target::Project(projection)) => {
                        let weight = self.weight(step.weights[0]);
                        let normed = self.state.scratch.normed;
                        let (input, out, rows, columns) = match projection {
                            Projection::Query => (
                                normed,
                                self.state.scratch.q,
                                heads * head_dim,
                                hidden,
                            ),
                            Projection::Gate => (
                                normed,
                                self.state.scratch.gate,
                                heads * head_dim,
                                hidden,
                            ),
                            Projection::Key => (
                                if sliding { normed } else { normed.offset(hidden * 4) },
                                self.state.scratch.k,
                                kv_width,
                                hidden,
                            ),
                            Projection::Value => (
                                if sliding {
                                    normed
                                } else {
                                    normed.offset(2 * hidden * 4)
                                },
                                self.state.scratch.v,
                                kv_width,
                                hidden,
                            ),
                            Projection::Relative => (
                                normed,
                                self.state.scratch.rel,
                                heads * config.d_rel,
                                hidden,
                            ),
                            Projection::Sink => (normed, self.state.scratch.sink, heads, hidden),
                            Projection::AttentionOut => (
                                self.state.scratch.attn,
                                self.state.scratch.delta,
                                hidden,
                                heads * head_dim,
                            ),
                            Projection::Router => (
                                normed,
                                self.state.scratch.router_logits,
                                config.n_routed_experts,
                                hidden,
                            ),
                            Projection::Head => {
                                return Err(Self::fail("the head is a global step"));
                            }
                        };
                        run(shared.gemv(weight, input, out, rows, columns))?;
                        if layer == 0 && matches!(projection, Projection::AttentionOut) {
                            self.rms(&shared, "attention_out", out, rows)?;
                        }
                    }
                    (Op::ShortConv, Target::Conv(slot)) => {
                        let taps = self.weight(step.weights[0]);
                        let (input, out, channels, state) = match slot {
                            ConvSlot::Key => (
                                self.state.scratch.k,
                                self.state.scratch.k_next,
                                kv_width,
                                self.state.conv[layer][0],
                            ),
                            ConvSlot::Value => (
                                self.state.scratch.v,
                                self.state.scratch.v_next,
                                kv_width,
                                self.state.conv[layer][1],
                            ),
                            ConvSlot::Attention => (
                                self.state.scratch.delta,
                                self.state.scratch.delta,
                                hidden,
                                self.state.conv[layer][2],
                            ),
                            ConvSlot::Mlp => (
                                self.state.scratch.mixed,
                                self.state.scratch.mixed,
                                hidden,
                                self.state.conv[layer][3],
                            ),
                        };
                        run(shared.launch(
                            shared.kernels.short_conv,
                            (flat(channels), 1, 1),
                            (ELEMENT_BLOCK, 1, 1),
                            &[
                                input.0,
                                taps.0,
                                state.0,
                                out.0,
                                channels as u64,
                                shared.kernel_width() as u64,
                            ],
                        ))?;
                        previous_conv = Some(slot);
                    }
                    (Op::ShortConvDynamic, Target::Conv(slot)) => {
                        let static_taps = self.weight(step.weights[0]);
                        let projection = self.weight(step.weights[1]);
                        let (input, out, state, source) = match slot {
                            ConvSlot::Key => (
                                self.state.scratch.k,
                                self.state.scratch.k_next,
                                self.state.conv[layer][0],
                                self.state.scratch.normed.offset(hidden * 4),
                            ),
                            ConvSlot::Value => (
                                self.state.scratch.v,
                                self.state.scratch.v_next,
                                self.state.conv[layer][1],
                                self.state.scratch.normed.offset(2 * hidden * 4),
                            ),
                            other => {
                                return Err(Self::fail(format!(
                                    "layer {layer}: a dynamic convolution has no {other:?} slot"
                                )));
                            }
                        };
                        // The dynamic taps are a projection of this slot's input, one kernel width
                        // a key/value head.
                        let taps = plan.rank_kv_heads() * shared.kernel_width();
                        run(shared.gemv(projection, source, self.state.scratch.taps, taps, hidden))?;
                        run(shared.launch(
                            shared.kernels.short_conv_dynamic,
                            (flat(kv_width), 1, 1),
                            (ELEMENT_BLOCK, 1, 1),
                            &[
                                input.0,
                                static_taps.0,
                                self.state.scratch.taps.0,
                                state.0,
                                out.0,
                                kv_width as u64,
                                shared.kernel_width() as u64,
                                head_dim as u64,
                            ],
                        ))?;
                        previous_conv = Some(slot);
                    }
                    (Op::Widen, _) => {
                        run(shared.launch(
                            shared.kernels.widen,
                            (flat(heads * head_dim), 1, 1),
                            (ELEMENT_BLOCK, 1, 1),
                            &[
                                self.state.scratch.attn_bf16.0,
                                self.state.scratch.attn.0,
                                (heads * head_dim) as u64,
                            ],
                        ))?;
                    }
                    (Op::KvStore, _) => {
                        if trace_on() {
                            let rank = plan.tp().rank();
                            let read = |pointer: Deviceptr, count: usize| -> Vec<f32> {
                                let mut bytes = vec![0u8; count * 4];
                                let _ = shared.cuda.copy_dtoh(&mut bytes, &pointer);
                                from_bytes(&bytes)
                            };
                            let q = read(self.state.scratch.q, 2);
                            let k = read(self.state.scratch.k_next, 2);
                            let v = read(self.state.scratch.v_next, 2);
                            let gate = read(self.state.scratch.gate, 1);
                            let sink = read(self.state.scratch.sink, 1);
                            let rel = read(self.state.scratch.rel, 1);
                            eprintln!(
                                "[inkling] device rank {rank} layer {layer} q={:.6} {:.6} g={:.6}                                  s={:.6} r={:.6} k={:.6} {:.6} v={:.6} {:.6}",
                                q[0], q[1], gate[0], sink[0], rel[0], k[0], k[1], v[0], v[1]
                            );
                        }
                        // A ring cache indexes by the position modulo its length; the attention
                        // body is given the absolute position and maps it itself.
                        let slot_position = position % cache_rows.max(1);
                        run(shared.launch(
                            shared.kernels.kv_store,
                            (flat(kv_width), 1, 1),
                            (ELEMENT_BLOCK, 1, 1),
                            &[
                                self.state.scratch.k_next.0,
                                self.state.scratch.v_next.0,
                                self.state.keys[index].0,
                                self.state.values[index].0,
                                1,
                                slot_position as u64,
                                kv_width as u64,
                            ],
                        ))?;
                    }
                    (Op::Attention, _) => {
                        let bank = self.weight(step.weights[0]);
                        run(shared.launch(
                            shared.kernels.attention,
                            (heads as u32, 1, 1),
                            (ATTENTION_BLOCK, 1, 1),
                            &[
                                self.state.scratch.q.0,
                                self.state.scratch.gate.0,
                                self.state.keys[index].0,
                                self.state.values[index].0,
                                self.state.scratch.rel.0,
                                bank.0,
                                self.state.scratch.sink.0,
                                clips.query.0,
                                0,
                                clips.relative.0,
                                self.state.scratch.attn_bf16.0,
                                heads as u64,
                                plan.rank_kv_heads() as u64,
                                head_dim as u64,
                                groups as u64,
                                position as u64,
                                if sliding { schedule.window as u64 } else { 0 },
                                // A sliding layer's cache is its window and is read as a ring; a
                                // full layer's is the context and is addressed absolutely.
                                if sliding { cache_rows as u64 } else { 0 },
                                schedule.rel_extent as u64,
                                config.d_rel as u64,
                                bits(config.qk_l2_clip_threshold as f32),
                                bits(config.rel_l2_clip_threshold as f32),
                                bits(schedule.tau.at(position)),
                                1,
                                bits(config.architecture.norm_epsilon as f32),
                            ],
                        ))?;
                    }
                    (Op::ResidualAdd, _) => {
                        if layer == 0 {
                            self.rms(&shared, "mixed", self.state.scratch.mixed, hidden)?;
                        }
                        let delta = match previous_conv {
                            Some(ConvSlot::Mlp) => self.state.scratch.mixed,
                            _ => self.state.scratch.delta,
                        };
                        run(shared.launch(
                            shared.kernels.residual_add,
                            (flat(hidden), 1, 1),
                            (ELEMENT_BLOCK, 1, 1),
                            &[
                                self.state.scratch.x.0,
                                delta.0,
                                self.state.scratch.x.0,
                                1,
                                hidden as u64,
                            ],
                        ))?;
                        previous_conv = None;
                    }
                    (Op::BlockResidual, Target::Input(ResidualInput::Mlp)) => {
                        let queries = self.weight(step.weights[0]);
                        run(shared.block_residual(
                            self.state.summaries,
                            self.state.scratch.x,
                            queries,
                            self.state.scratch.rows,
                            self.state.blocks,
                            1,
                        ))?;
                    }
                    (Op::Router, _) => {
                        let bias = self.weight(step.weights[0]);
                        run(shared.launch(
                            shared.kernels.router,
                            (1, 1, 1),
                            (ROUTER_BLOCK, 1, 1),
                            &[
                                self.state.scratch.router_logits.0,
                                bias.0,
                                self.state.scratch.indices.0,
                                self.state.scratch.shares.0,
                                config.n_routed_experts as u64,
                                config.num_experts_per_tok as u64,
                                bits(config.route_scale as f32),
                            ],
                        ))?;
                    }
                    (Op::ExpertGateUp, Target::Band(BandOp::GateUp)) => {
                        let activation = &schedule.steps[cursor + 1];
                        let down = &schedule.steps[cursor + 2];
                        self.run_band(&shared, index, step, activation, down)?;
                        // The band's three launches and its local mix; the all-reduce that
                        // follows them is the next step and stays the walk's.
                        cursor += 4;
                        continue;
                    }
                    (Op::Collective(Collective::AttentionDelta), _) => {
                        // The output projection sharded the heads, its contraction axis, so the
                        // dense result is a partial sum of every element of the layer's attention
                        // output. The convolution that follows reads a complete row.
                        self.all_reduce(&shared, self.state.scratch.delta, hidden)?;
                        if trace_on() {
                            let label = format!("attn_proj layer {layer}");
                            self.rms(&shared, &label, self.state.scratch.delta, hidden)?;
                        }
                    }
                    (Op::Collective(Collective::ExpertDelta), _) => {
                        // Each rank mixed its own experts and zeroed the slots it does not own,
                        // so the sum over ranks is the layer's whole expert contribution.
                        self.all_reduce(&shared, self.state.scratch.mixed, hidden)?;
                    }
                    (Op::Collective(_), _) => {}
                    (op, target) => {
                        return Err(Self::fail(format!(
                            "layer {layer}: the schedule's {op:?} at {target:?} has no device \
                             dispatch"
                        )));
                    }
                }
                cursor += 1;
            }
            if trace_on() {
                let mut probe = vec![0u8; hidden * 4];
                shared
                    .cuda
                    .copy_dtoh(&mut probe, &self.state.scratch.x)
                    .map_err(Self::fail)?;
                let values: Vec<f32> = probe
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| f32::from_le_bytes(*word))
                    .collect();
                let rms = (values.iter().map(|value| value * value).sum::<f32>()
                    / hidden as f32)
                    .sqrt();
                eprintln!(
                    "[inkling] device layer {layer} x0={:.5} x1={:.5} rms={rms:.5}",
                    values[0], values[1]
                );
            }
        }

        for step in plan.epilogue() {
            match (step.op, step.target) {
                (Op::BlockResidual, Target::Input(ResidualInput::Output)) => {
                    let queries = self.weight(step.weights[0]);
                    run(shared.block_residual(
                        self.state.summaries,
                        self.state.scratch.x,
                        queries,
                        self.state.scratch.final_row,
                        self.state.blocks,
                        1,
                    ))?;
                }
                (Op::RmsNorm, Target::Norm(NormRole::Final)) => {
                    self.rms(&shared, "final_input", self.state.scratch.final_row, hidden)?;
                    let weight = self.weight(step.weights[0]);
                    run(shared.rmsnorm(
                        self.state.scratch.final_row,
                        weight,
                        self.state.scratch.final_row,
                        1,
                        hidden,
                    ))?;
                }
                (Op::Gemv, Target::Project(Projection::Head)) => {
                    self.rms(&shared, "final_norm", self.state.scratch.final_row, hidden)?;
                    let weight = self.weight(step.weights[0]);
                    run(shared.gemv(
                        weight,
                        self.state.scratch.final_row,
                        self.state.scratch.logits,
                        plan.rank_vocab(),
                        hidden,
                    ))?;
                }
                (Op::Collective(Collective::LogitSlice), _) => {
                    let width = plan.rank_vocab();
                    let mut bytes = row_bytes(&vec![0f32; width]);
                    shared
                        .cuda
                        .copy_dtoh(&mut bytes, &self.state.scratch.logits)
                        .map_err(Self::fail)?;
                    let mut gathered = vec![0f32; width * shared.exchange.world()];
                    shared
                        .exchange
                        .all_gather_f32(&from_bytes(&bytes), &mut gathered)
                        .map_err(Self::fail)?;
                    shared
                        .cuda
                        .copy_htod(&self.state.scratch.logits_full, &row_bytes(&gathered))
                        .map_err(Self::fail)?;
                }
                (Op::Collective(_), _) => {}
                (Op::LogitSoftcap, _) => {
                    self.rms(&shared, "raw_logits", self.state.scratch.logits, 4)?;
                    // The gathered row is the whole padded vocabulary, so the released trim is
                    // against that width and not against one rank's slice of it.
                    let padded = plan.rank_vocab() * shared.exchange.world();
                    let unpadded = config.unpadded_vocab_size.min(padded).max(1);
                    run(shared.launch(
                        shared.kernels.logit_softcap,
                        (1, 1, 1),
                        (ELEMENT_BLOCK, 1, 1),
                        &[
                            self.state.scratch.logits_full.0,
                            self.state.scratch.logits_bf16.0,
                            1,
                            // The gathered row is the whole padded vocabulary, which is what
                            // the softcap trims to the released 200,058.
                            (plan.rank_vocab() * shared.exchange.world()) as u64,
                            unpadded as u64,
                            bits(config.final_logit_softcap.unwrap_or(0.0) as f32),
                        ],
                    ))?;
                }
                (op, target) => {
                    return Err(Self::fail(format!(
                        "the schedule's epilogue {op:?} at {target:?} has no device dispatch"
                    )));
                }
            }
        }

        self.state.position += 1;
        shared
            .cuda
            .synchronize()
            .map_err(Self::fail)?;
        let padded = plan.rank_vocab() * shared.exchange.world();
        let unpadded = config.unpadded_vocab_size.min(padded).max(1);
        let mut bytes = vec![0u8; unpadded * 2];
        shared
            .cuda
            .copy_dtoh(&mut bytes, &self.state.scratch.logits_bf16)
            .map_err(Self::fail)?;
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| bf16_to_f32(u16::from_le_bytes(*pair)))
            .collect())
    }

    /// The routed band: each selected expert's three launches, then the weighted mix.
    ///
    /// The band is a loop over the experts the router selected, not a step a expert: the plan
    /// records the three launches once and this runs them for every selected expert the rank
    /// owns.
    fn run_band(
        &mut self,
        shared: &Shared,
        layer: usize,
        gate_up: &Step,
        activation: &Step,
        down: &Step,
    ) -> Result<(), BackendError> {
        let config = &shared.config;
        let schedule = &shared.plan.layers()[layer];
        let top_k = config.num_experts_per_tok;
        let intermediate = config.moe_intermediate_size;
        let hidden = config.hidden_size;
        let mut index_bytes = vec![0u8; top_k * std::mem::size_of::<i32>()];
        let mut share_bytes = vec![0u8; top_k * std::mem::size_of::<f32>()];
        shared
            .cuda
            .copy_dtoh(&mut index_bytes, &self.state.scratch.indices)
            .map_err(Self::fail)?;
        shared
            .cuda
            .copy_dtoh(&mut share_bytes, &self.state.scratch.shares)
            .map_err(Self::fail)?;
        let selected: Vec<usize> = index_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| i32::from_le_bytes(*word).max(0) as usize)
            .collect();
        let shares: Vec<f32> = share_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
        let scales = shared.experts[layer].as_ref();
        let input = self.state.scratch.normed;
        let band = self.state.scratch.band;
        let activated = self.state.scratch.activation;
        let partials = self.state.scratch.partials;
        let mut scale_bytes = vec![0u8; top_k * 4];
        let gate_up_codes = self.weight(gate_up.weights[0]);
        let gate_up_scales = gate_up.weights.get(1).map(|index| self.weight(*index));
        let down_codes = self.weight(down.weights[0]);
        let down_scales = down.weights.get(1).map(|index| self.weight(*index));
        // The router is replicated and selects global experts; each rank holds one contiguous run
        // of them. A slot outside the run is not this rank's to compute - its share stays zero, so
        // the mix contributes nothing and the all-reduce that follows is the whole exchange.
        let (first, owned) = shared.plan.expert_range();
        for (slot, expert) in selected.iter().copied().enumerate() {
            if expert >= config.n_routed_experts || expert < first || expert >= first + owned {
                continue;
            }
            // The rank's own weights and per-expert scales are indexed from zero, not from the
            // released expert number.
            let local = expert - first;
            let gate_up_offset = local * codes_bytes(2 * intermediate, hidden, schedule.experts);
            let destination = band.offset(slot * 2 * intermediate * std::mem::size_of::<f32>());
            match schedule.experts {
                ExpertBody::Bf16 => {
                    shared
                        .gemv(
                            gate_up_codes.offset(gate_up_offset),
                            input,
                            destination,
                            2 * intermediate,
                            hidden,
                        )
                        .map_err(Self::fail)?;
                }
                ExpertBody::NvFp4 | ExpertBody::MxFp8 => {
                    let scale_offset =
                        local * scales_bytes(2 * intermediate, hidden, schedule.experts);
                    let scale_pointer = gate_up_scales
                        .map_or(Deviceptr(0), |pointer| pointer.offset(scale_offset));
                    let (entry, last) = match schedule.experts {
                        ExpertBody::NvFp4 => (
                            shared.kernels.expert_nvfp4,
                            bits(scales.map_or(1.0, |scale| scale.gate_up[local])),
                        ),
                        _ => (shared.kernels.expert_mxfp8, MXFP8_BLOCK as u64),
                    };
                    shared
                        .launch(
                            entry,
                            (2 * intermediate as u32, 1, 1),
                            (ELEMENT_BLOCK, 1, 1),
                            &[
                                gate_up_codes.offset(gate_up_offset).0,
                                scale_pointer.0,
                                input.0,
                                destination.0,
                                2 * intermediate as u64,
                                hidden as u64,
                                last,
                            ],
                        )
                        .map_err(Self::fail)?;
                }
            }
            let activation_out = activated.offset(slot * intermediate * 4);
            shared
                .launch(
                    shared.kernels.swiglu,
                    (flat(intermediate), 1, 1),
                    (ELEMENT_BLOCK, 1, 1),
                    &[
                        destination.0,
                        activation_out.0,
                        intermediate as u64,
                        bits(config.mlp_gate_softcap as f32),
                        bits(config.mlp_up_softcap as f32),
                    ],
                )
                .map_err(Self::fail)?;
            let down_offset = local * codes_bytes(hidden, intermediate, schedule.experts);
            let partial = partials.offset(slot * hidden * 4);
            match schedule.experts {
                ExpertBody::Bf16 => {
                    shared
                        .gemv(
                            down_codes.offset(down_offset),
                            activation_out,
                            partial,
                            hidden,
                            intermediate,
                        )
                        .map_err(Self::fail)?;
                }
                ExpertBody::NvFp4 | ExpertBody::MxFp8 => {
                    let scale_offset =
                        local * scales_bytes(hidden, intermediate, schedule.experts);
                    let scale_pointer = down_scales
                        .map_or(Deviceptr(0), |pointer| pointer.offset(scale_offset));
                    let (entry, last) = match schedule.experts {
                        ExpertBody::NvFp4 => (
                            shared.kernels.expert_nvfp4,
                            bits(scales.map_or(1.0, |scale| scale.down[local])),
                        ),
                        _ => (shared.kernels.expert_mxfp8, MXFP8_BLOCK as u64),
                    };
                    shared
                        .launch(
                            entry,
                            (hidden as u32, 1, 1),
                            (ELEMENT_BLOCK, 1, 1),
                            &[
                                down_codes.offset(down_offset).0,
                                scale_pointer.0,
                                activation_out.0,
                                partial.0,
                                hidden as u64,
                                intermediate as u64,
                                last,
                            ],
                        )
                        .map_err(Self::fail)?;
                }
            }
            // The share this expert's output carries into the mix, which is the router's weight
            // times whatever per-expert scale the stored format puts on its own output.
            let share = shares[slot];
            scale_bytes[slot * 4..slot * 4 + 4].copy_from_slice(&share.to_le_bytes());
        }
        let _ = activation;
        shared
            .cuda
            .copy_htod(&self.state.scratch.scales, &scale_bytes)
            .map_err(Self::fail)?;
        shared
            .launch(
                shared.kernels.expert_mix,
                (flat(hidden), 1, 1),
                (ELEMENT_BLOCK, 1, 1),
                &[
                    self.state.scratch.mixed.0,
                    partials.0,
                    self.state.scratch.scales.0,
                    1,
                    top_k as u64,
                    hidden as u64,
                ],
            )
            .map_err(Self::fail)?;
        Ok(())
    }
}


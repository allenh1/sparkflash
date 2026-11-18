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

//! The shared dense device arm: a family no pack transcribes, served on the device.
//!
//! A BF16 grouped-query decoder (embedding, RMSNorm, grouped-query attention with or without the
//! per-head output gate and q/k norms, plain rotate-half RoPE, a dense SwiGLU or routed experts,
//! a plain residual and the head) needs no code of its own: this arm reads the shared
//! [Architecture], resolves the checkpoint's own spelling through [sparkflash_ops::naming],
//! uploads every tensor through [DraftUploader] and launches the shared token-image bodies through
//! their typed [sparkflash_ops::calls]. Its prompt projections take the shared wide tile under
//! [WidePolicy::DENSE]; decode rows keep the row GEMV.
//!
//! A routed layer runs the parts the released arms run: the one router body (`q4e_route`, any
//! family's score, top-k, renormalisation and scale), the shared expert plan's run walk
//! ([MoePlanCall::runs]), the token path's expert passes in their BF16 form (one block a run of
//! an expert's rows, Qwen's NVFP4 and FP8 body) and the combine, with the shared expert and its
//! sigmoid gate on the same projections as a dense MLP.
//!
//! Which checkpoints it takes is [sparkflash_ops::parts::dense_device_refusal]: nothing the host
//! reference does not execute, and nothing the reference does that these kernels do not. Its
//! gate is a per-layer parity report against that reference (the CLI's `dense_parity` test),
//! not a recorded digest: the arm keeps its KV cache in BF16 where the reference keeps f32, so a
//! row agrees to a tolerance rather than to the bit.
//!
//! It serves on the shared runtime's contract ([DenseLanes], a [FusedArm]): fused lanes whose
//! decode rows share one walk, prompts a chunk at a time between the lanes' steps, kept
//! conversations and a rewind point to the end of each prompt, its caches on the shared lane
//! memory ([SequenceLanes]), and the ledger. It runs on one rank.

use std::sync::{Arc, Mutex};

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_kernels::images::{
    dory_cubin, moe_plan_cubin, qwen4_attn_cubin, qwen4_exp_cubin, qwen4_moe_cubin,
};
use sparkflash_kernels::lane_memory::{Layout, SequenceLane, SequenceLanes};
use sparkflash_model::architecture::{AttentionKind, Ffn};
use sparkflash_model::backend::{
    BackendError, ModelBackend, ModelSession, Sampling, Speculation, StepOutput,
};
use sparkflash_model::vision::VisionImage;
use sparkflash_model::{Architecture, Capabilities, ModelDescriptor};
use sparkflash_ops::calls::{self, Call, FullAttention, RmsNorm, Rope};
use sparkflash_ops::dflash::DraftUploader;
use sparkflash_ops::engine_plan::{EnginePlan, PlanError};
use sparkflash_ops::gemv::{CudaHandle, GEMV_MAX_BATCH, GpuMat};
use sparkflash_ops::moe::{Launch, MoePlanCall, combine_grid, router_score, token};
use sparkflash_ops::naming::{AttentionNames, ExpertNames, FfnNames, TensorNames};
use sparkflash_ops::projection::{Route, WIDE_SHARED, WidePolicy, WideTile, route_bf16};
use sparkflash_ops::sampler::Sampler;
use sparkflash_runtime::{FusedArm, FusedEngine, Ledger, Mode, Part, Resume, Strategy, resume};
use sparkflash_weights::{DType, WeightSource};

/// The host-facing name of a device refusal.
const DEVICE: &str = "dense_device";

/// The prompt rows one pass of the layer walk takes: two of the wide tile's 128-row blocks.
pub const CHUNK_ROWS: usize = 256;

/// One layer's attention and MLP shape.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Shape {
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    gated: bool,
    rotary: usize,
    theta: f32,
    /// The dense MLP's width, or a routed layer's shared expert's (zero without one).
    intermediate: usize,
    /// A routed layer's experts.
    routed: Option<Routed>,
}

/// A routed layer's rule: what `q4e_route` and the expert passes take.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Routed {
    experts: usize,
    top_k: usize,
    /// One expert's SwiGLU width.
    intermediate: usize,
    /// The router's score ([router_score]).
    score: u64,
    renormalise: bool,
    scaling: f32,
}

impl Shape {
    fn of(architecture: &Architecture, index: usize) -> Result<Self, String> {
        let layer = &architecture.layers[index];
        let AttentionKind::Gqa {
            kv_heads,
            output_gate,
        } = layer.attention.kind
        else {
            return Err(format!("layer {index} is not grouped-query attention"));
        };
        let (intermediate, routed) = match &layer.ffn {
            Ffn::Dense(dense) => (dense.intermediate, None),
            Ffn::Moe(moe) => (
                if moe.shared_experts > 0 {
                    moe.shared_intermediate
                } else {
                    0
                },
                Some(Routed {
                    experts: moe.routed_experts,
                    top_k: moe.active_experts,
                    intermediate: moe.intermediate,
                    score: router_score(moe.score),
                    renormalise: moe.norm_topk,
                    scaling: moe.routed_scaling as f32,
                }),
            ),
        };
        let head_dim = layer.attention.head_dim;
        Ok(Self {
            heads: layer.attention.heads,
            kv_heads,
            head_dim,
            gated: output_gate,
            rotary: layer.attention.rope.rotated_dims.min(head_dim),
            theta: layer.attention.rope.theta.max(1.0) as f32,
            intermediate,
            routed,
        })
    }

    const fn query_width(&self) -> usize {
        self.heads * self.head_dim
    }

    const fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// The query projection's rows: the queries, and the interleaved gates when gated.
    const fn query_rows(&self) -> usize {
        if self.gated {
            2 * self.query_width()
        } else {
            self.query_width()
        }
    }
}

/// One layer's device weights.
#[derive(Debug, Clone, Copy)]
struct Layer {
    shape: Shape,
    input_norm: Deviceptr,
    q: GpuMat,
    k: GpuMat,
    v: GpuMat,
    o: GpuMat,
    q_norm: Option<Deviceptr>,
    k_norm: Option<Deviceptr>,
    post_norm: Deviceptr,
    ffn: LayerFfn,
}

/// One SwiGLU MLP's projections: a dense layer's, or a routed layer's shared expert.
#[derive(Debug, Clone, Copy)]
struct Mlp {
    gate: GpuMat,
    up: GpuMat,
    down: GpuMat,
}

/// A layer's MLP on the device.
#[derive(Debug, Clone, Copy)]
enum LayerFfn {
    Dense(Mlp),
    Routed {
        /// The router, `[experts][hidden]`.
        router: GpuMat,
        /// Six addresses an expert, the token passes' table: gate, its (unread) scale, up, its
        /// scale, down, its scale.
        table: Deviceptr,
        shared: Option<Mlp>,
        /// The shared expert's sigmoid gate, one `[hidden]` row.
        shared_gate: Option<GpuMat>,
    },
}

/// The routed layers' kernels, loaded when a layer routes.
#[derive(Clone, Copy)]
struct MoeKernels {
    route: Kernel,
    plan: Kernel,
    gate_up: Kernel,
    down: Kernel,
    combine: Kernel,
}

/// The kernels this arm launches, loaded once.
#[derive(Clone, Copy)]
struct Kernels {
    add: Kernel,
    deinterleave: Kernel,
    embed: Kernel,
    gemv: Kernel,
    rmsnorm: Kernel,
    rope: Kernel,
    swiglu: Kernel,
    attention: Kernel,
    combine: Kernel,
    cache: Kernel,
    wide: Kernel,
    moe: Option<MoeKernels>,
}

// SAFETY: a Kernel is an opaque driver function word, and the handle only names it. The context
// that owns the module lives in the shared state, which outlives every session.
unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

impl Kernels {
    /// Loads every kernel, and the routed layers' when `stage` (the expert passes' staging
    /// reservation, [token::stage_bytes]) names one.
    fn load(cuda: &Cuda, stage: Option<u32>) -> Result<Self, String> {
        let named = |image: &'static [u8], name: &'static str| {
            cuda.load_named(image, name).map(|(_, kernel)| kernel)
        };
        let staged = |name: &'static str, bytes: u32| {
            cuda.load_named_shared(qwen4_moe_cubin(), name, bytes)
                .map(|(_, kernel)| kernel)
        };
        let moe = match stage {
            Some(bytes) => Some(MoeKernels {
                route: named(qwen4_moe_cubin(), "q4e_route")?,
                plan: named(moe_plan_cubin(), "moe_plan")?,
                gate_up: staged("q4e_expert_gate_up_bf16", bytes)?,
                down: staged("q4e_expert_down_bf16", bytes)?,
                combine: named(qwen4_moe_cubin(), "q4e_moe_combine")?,
            }),
            None => None,
        };
        Ok(Self {
            moe,
            add: named(qwen4_exp_cubin(), "q4e_add")?,
            deinterleave: named(qwen4_exp_cubin(), "q4e_deinterleave")?,
            embed: named(qwen4_exp_cubin(), "q4e_embed")?,
            gemv: named(qwen4_exp_cubin(), "q4e_gemv_bf16")?,
            rmsnorm: named(qwen4_exp_cubin(), "q4e_rmsnorm")?,
            rope: named(qwen4_exp_cubin(), "q4e_rope")?,
            swiglu: named(qwen4_exp_cubin(), "q4e_swiglu")?,
            attention: named(qwen4_attn_cubin(), "q4e_attn_full")?,
            combine: named(qwen4_attn_cubin(), "q4e_attn_full_combine")?,
            cache: named(qwen4_attn_cubin(), "q4e_cache_bf16")?,
            wide: cuda
                .load_named_shared(dory_cubin(), "dory_mma_wide", WIDE_SHARED)
                .map(|(_, kernel)| kernel)?,
        })
    }
}

/// The weights, kernels and geometry every session reads.
struct Shared {
    cuda: CudaHandle,
    kernels: Kernels,
    embed: GpuMat,
    head: GpuMat,
    norm: Deviceptr,
    layers: Vec<Layer>,
    hidden: usize,
    vocab: usize,
    eps: f32,
    zero_centred: bool,
    policy: Option<WidePolicy>,
    context: usize,
    /// The sequences' key/value caches' lanes, for sequences of `context` positions.
    lanes: Arc<SequenceLanes<LaneParts>>,
    /// Whether a sequence keeps a rewind point at the end of each prompt
    /// ([sparkflash_runtime::REWIND_ENV], on by default).
    rewind: bool,
    /// Every weight allocation, freed once.
    owned: Vec<Deviceptr>,
    weight_bytes: usize,
}

impl Drop for Shared {
    fn drop(&mut self) {
        // Every sequence holds the shared state, so each has given its lane back by now.
        self.lanes.drain(&self.cuda);
        let references: Vec<&Deviceptr> = self.owned.iter().collect();
        self.cuda.free_buffers(&references);
    }
}

/// A dense grouped-query checkpoint on the device.
pub struct DenseDeviceBackend {
    descriptor: ModelDescriptor,
    shared: Arc<Shared>,
    /// The fused engine its sessions open on once [Self::enable_fusion] has run.
    fused: Option<Arc<FusedEngine<DenseLanes>>>,
}

impl DenseDeviceBackend {
    /// Whether this build carries every kernel the arm launches.
    #[must_use]
    pub fn available() -> bool {
        sparkflash_ops::parts::dense_device_images().is_ok()
    }

    /// Opens `architecture`'s checkpoint from `source` under `descriptor`, for sequences of
    /// `context` tokens.
    ///
    /// # Errors
    /// A checkpoint the arm does not execute ([sparkflash_ops::parts::dense_device_refusal]), a
    /// missing kernel, a profile field the arm cannot honour, a [sparkflash_runtime::REWIND_ENV]
    /// that is not a switch, a tensor that is not BF16 at the shape the architecture names, a
    /// zero context or one past the checkpoint's window, a CUDA context that cannot open, or a
    /// device without the virtual memory management the lanes need.
    pub fn open(
        mut descriptor: ModelDescriptor,
        architecture: &Architecture,
        source: &dyn WeightSource,
        context: usize,
    ) -> Result<Self, String> {
        if let Some(refusal) = sparkflash_ops::parts::dense_device_refusal(architecture) {
            return Err(format!("the shared dense device arm refuses it: {refusal}"));
        }
        sparkflash_ops::parts::dense_device_images()?;
        let window = architecture.max_position_embeddings;
        if context == 0 || (window != 0 && context > window) {
            return Err(format!(
                "context {context} is outside 1..={window} for the checkpoint"
            ));
        }
        let profile = sparkflash_model::profile::active();
        let policy = profile_policy(profile)?;
        let rewind = sparkflash_runtime::rewind()?;
        let names = sparkflash_ops::naming::resolve(architecture, source)?;
        let cuda = CudaHandle::new(Cuda::load()?);
        sparkflash_kernels::cuda_driver::enable_forward_async_launches();
        let kernels = Kernels::load(&cuda, expert_stage(architecture)?)?;
        let mut owned = Vec::new();
        let uploaded = upload(&cuda, architecture, &names, source, &mut owned);
        let (embed, head, norm, layers, weight_bytes) = match uploaded {
            Ok(uploaded) => uploaded,
            Err(error) => {
                let references: Vec<&Deviceptr> = owned.iter().collect();
                cuda.free_buffers(&references);
                return Err(error);
            }
        };
        let lanes = SequenceLanes::plan(
            &cuda,
            context,
            0,
            profile.runtime.lane_memory_mb,
            |layout| LaneParts::plan(layout, &layers, context),
        );
        let lanes = match lanes {
            Ok(lanes) => Arc::new(lanes),
            Err(error) => {
                let references: Vec<&Deviceptr> = owned.iter().collect();
                cuda.free_buffers(&references);
                return Err(error);
            }
        };
        eprintln!(
            "[dense] sequences of {context} positions back their caches as they grow: {}",
            lanes.describe()
        );
        descriptor.context_tokens = context;
        Ok(Self {
            descriptor,
            fused: None,
            shared: Arc::new(Shared {
                cuda,
                kernels,
                embed,
                head,
                norm,
                layers,
                hidden: architecture.hidden_size,
                vocab: architecture.vocab_size,
                eps: architecture.norm_epsilon as f32,
                zero_centred: architecture.zero_centered_norms(),
                policy,
                context,
                lanes,
                rewind,
                owned,
                weight_bytes,
            }),
        })
    }

    /// The device bytes the weights claimed, for boot diagnostics.
    #[must_use]
    pub fn weight_bytes(&self) -> usize {
        self.shared.weight_bytes
    }

    /// A session of its concrete type, for a caller that reads its per-layer trace.
    ///
    /// # Errors
    /// The session buffers or caches do not fit on the device.
    pub fn session(&self) -> Result<DenseDeviceSession, String> {
        DenseDeviceSession::open(Arc::clone(&self.shared))
    }

    /// Serves this backend's sessions on the shared fused engine: `capacity` lanes whose decode
    /// rows share one walk, whose prompts run a part at a time between the lanes' steps, and which
    /// keep their conversations between requests. Call it before the backend is shared with the
    /// serving driver, with the driver's concurrency as the capacity.
    ///
    /// # Errors
    /// A zero capacity, a scratch or lane the device refuses, or a thread that cannot start.
    pub fn enable_fusion(&mut self, capacity: usize) -> Result<(), BackendError> {
        let arm = DenseLanes::new(Arc::clone(&self.shared), capacity)?;
        self.fused = Some(FusedEngine::start(Arc::new(arm), capacity)?);
        Ok(())
    }
}

impl sparkflash_ops::EngineBuilder for DenseDeviceBackend {
    type Backend = Self;

    fn open(
        plan: &EnginePlan,
        weights: &dyn WeightSource,
        context: usize,
    ) -> Result<Self::Backend, PlanError> {
        plan.single_rank_context()?;
        let context = plan.admitted_context(context)?;
        let architecture = plan.architecture();
        let descriptor = ModelDescriptor::from_architecture(
            architecture.family.clone(),
            architecture,
            context,
            Capabilities::ALL,
        );
        Self::open(descriptor, architecture, weights, context).map_err(PlanError::Invalid)
    }
}

impl ModelBackend for DenseDeviceBackend {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        if let Some(engine) = &self.fused {
            return Ok(Box::new(engine.open()?));
        }
        Ok(Box::new(
            self.session()
                .map_err(|error| BackendError::new(DEVICE, error))?,
        ))
    }
}

/// The wide-tile policy a typed profile asks of the arm, or the field it cannot honour.
///
/// The arm's chunk and its prompt-and-decode split are fixed by its kernels; a field it does not
/// honour is refused by name rather than accepted and dropped. It honours
/// `prefill.tensor_cores = false`, which keeps every projection on the row GEMV; its lanes
/// (`runtime.concurrency`) and lane memory (`runtime.lane_memory_mb`) are the serving engine's.
fn profile_policy(
    profile: &sparkflash_model::profile::ModelProfile,
) -> Result<Option<WidePolicy>, String> {
    use sparkflash_model::profile::LatentCacheMode;
    let refuse = |what: &str| {
        Err(format!(
            "the shared dense device arm {what}; change the profile, or serve the checkpoint on \
             the host reference (SPARKFLASH_DENSE_DEVICE=0)"
        ))
    };
    if let Some(chunk) = profile.prefill.chunk
        && chunk != CHUNK_ROWS
    {
        return refuse(&format!(
            "fixes its prompt chunk at {CHUNK_ROWS} rows, not prefill.chunk = {chunk}"
        ));
    }
    if profile.prefill.enabled == Some(false) {
        return refuse("always runs its prompt as a prefill, but prefill.enabled = false");
    }
    if profile.prefill.exact == Some(true) {
        return refuse(
            "runs its prompt projections on the wide tile and its decode rows on the row GEMV, \
             which are not bit-identical, but prefill.exact = true",
        );
    }
    if profile.decode.tensor_cores == Some(true) {
        return refuse("decodes on the row GEMV, but decode.tensor_cores = true");
    }
    if profile.speculation.enabled == Some(true) {
        return refuse("has no drafter, but speculation.enabled = true");
    }
    if profile.sequence.parallel == Some(true) || profile.sequence.overlap == Some(true) {
        return refuse("runs on one rank, without sequence parallelism or overlap");
    }
    if matches!(
        profile.attention.latent_cache,
        Some(LatentCacheMode::F32 | LatentCacheMode::Fp8)
    ) {
        return refuse("keeps a BF16 key/value cache, not attention.latent_cache's form");
    }
    Ok(if profile.prefill.tensor_cores == Some(false) {
        None
    } else {
        Some(WidePolicy::DENSE)
    })
}

/// The expert passes' staging reservation for the widest routed layer, or `None` when no layer
/// routes.
fn expert_stage(architecture: &Architecture) -> Result<Option<u32>, String> {
    let widest = architecture
        .layers
        .iter()
        .filter_map(|layer| match &layer.ffn {
            Ffn::Moe(moe) => Some(moe.intermediate),
            Ffn::Dense(_) => None,
        })
        .max();
    widest
        .map(|intermediate| token::stage_bytes(architecture.hidden_size, intermediate))
        .transpose()
}

type Uploaded = (GpuMat, GpuMat, Deviceptr, Vec<Layer>, usize);

/// One upload's tensors: each checked BF16 at its architecture shape, and every allocation in
/// `owned` as it is made, so a failure part way frees what was uploaded.
struct Uploads<'a> {
    uploader: DraftUploader<'a>,
    source: &'a dyn WeightSource,
    owned: &'a mut Vec<Deviceptr>,
    bytes: usize,
}

impl Uploads<'_> {
    fn keep(&mut self, pointer: Deviceptr, bytes: usize) -> Deviceptr {
        self.owned.push(pointer);
        self.bytes += bytes;
        pointer
    }

    fn mat(&mut self, name: &str, rows: usize, columns: usize) -> Result<GpuMat, String> {
        bf16(self.source, name, &[rows, columns])?;
        let mat = self.uploader.mat(name, rows, columns)?;
        self.keep(mat.pointer, rows * columns * 2);
        Ok(mat)
    }

    fn vector(&mut self, name: &str, width: usize) -> Result<Deviceptr, String> {
        bf16(self.source, name, &[width])?;
        let pointer = self.uploader.vector(name, width)?;
        Ok(self.keep(pointer, width * 2))
    }

    fn mlp(&mut self, names: &ExpertNames, width: usize, hidden: usize) -> Result<Mlp, String> {
        Ok(Mlp {
            gate: self.mat(&names.gate, width, hidden)?,
            up: self.mat(&names.up, width, hidden)?,
            down: self.mat(&names.down, hidden, width)?,
        })
    }

    /// A routed layer's router, experts (and their table), shared expert and its gate.
    fn routed(
        &mut self,
        index: usize,
        names: &FfnNames,
        shape: &Shape,
        routed: &Routed,
        hidden: usize,
    ) -> Result<LayerFfn, String> {
        let FfnNames::Moe {
            router,
            experts,
            shared,
            shared_gate,
        } = names
        else {
            return Err(format!("layer {index} names a dense MLP's tensors"));
        };
        if experts.len() != routed.experts {
            return Err(format!(
                "layer {index} names {} experts, the architecture {}",
                experts.len(),
                routed.experts
            ));
        }
        let router = self.mat(router, routed.experts, hidden)?;
        let mut table = Vec::with_capacity(routed.experts * 6 * 8);
        for expert in experts {
            let Mlp { gate, up, down } = self.mlp(expert, routed.intermediate, hidden)?;
            for address in [gate.pointer.0, 0, up.pointer.0, 0, down.pointer.0, 0] {
                table.extend(address.to_le_bytes());
            }
        }
        let table = self.uploader.bytes(&table)?;
        let table = self.keep(table, routed.experts * 6 * 8);
        let shared = match (shared, shape.intermediate) {
            (Some(names), width) if width > 0 => Some(self.mlp(names, width, hidden)?),
            (None, 0) => None,
            _ => {
                return Err(format!(
                    "layer {index}'s shared expert and the architecture's width {} disagree",
                    shape.intermediate
                ));
            }
        };
        let shared_gate = match shared_gate {
            Some(name) => {
                // A release stores the gate as `[1][hidden]` or as `[hidden]`.
                let spec = self.source.spec(name).map_err(|error| error.to_string())?;
                if spec.dtype != DType::BF16 || spec.elements() != hidden as u128 {
                    return Err(format!(
                        "{name} is {:?} {:?}; the shared dense device arm reads it as a BF16 \
                         {hidden}-wide gate",
                        spec.dtype, spec.shape
                    ));
                }
                let pointer = self.uploader.vector(name, hidden)?;
                Some(GpuMat {
                    pointer: self.keep(pointer, hidden * 2),
                    rows: 1,
                    columns: hidden,
                })
            }
            None => None,
        };
        if shared_gate.is_some() && shared.is_none() {
            return Err(format!(
                "layer {index} names a shared-expert gate without a shared expert"
            ));
        }
        Ok(LayerFfn::Routed {
            router,
            table,
            shared,
            shared_gate,
        })
    }
}

/// Uploads every tensor `names` resolves, each checked BF16 at its architecture shape; every
/// allocation lands in `owned` as it is made, so a failure part way frees what was uploaded.
fn upload(
    cuda: &Cuda,
    architecture: &Architecture,
    names: &TensorNames,
    source: &dyn WeightSource,
    owned: &mut Vec<Deviceptr>,
) -> Result<Uploaded, String> {
    let mut up = Uploads {
        uploader: DraftUploader::new(cuda, source),
        source,
        owned,
        bytes: 0,
    };
    let hidden = architecture.hidden_size;
    let vocab = architecture.vocab_size;
    let embed = up.mat(&names.embed, vocab, hidden)?;
    let head = if names.head_tied() {
        embed
    } else {
        up.mat(&names.head, vocab, hidden)?
    };
    let norm = up.vector(&names.norm, hidden)?;
    let mut layers = Vec::with_capacity(architecture.layers.len());
    for (index, layer_names) in names.layers.iter().enumerate() {
        let shape = Shape::of(architecture, index)?;
        let AttentionNames::Gqa {
            q,
            k,
            v,
            o,
            q_norm,
            k_norm,
            ..
        } = &layer_names.attention
        else {
            return Err(format!("layer {index} names another attention's tensors"));
        };
        let ffn = match (&shape.routed, &layer_names.ffn) {
            (None, FfnNames::Dense(names)) => {
                LayerFfn::Dense(up.mlp(names, shape.intermediate, hidden)?)
            }
            (None, _) => return Err(format!("layer {index} names a routed MLP's tensors")),
            (Some(routed), names) => up.routed(index, names, &shape, routed, hidden)?,
        };
        layers.push(Layer {
            shape,
            input_norm: up.vector(&layer_names.input_norm, hidden)?,
            q: up.mat(q, shape.query_rows(), hidden)?,
            k: up.mat(k, shape.kv_width(), hidden)?,
            v: up.mat(v, shape.kv_width(), hidden)?,
            o: up.mat(o, hidden, shape.query_width())?,
            q_norm: q_norm
                .as_deref()
                .map(|name| up.vector(name, shape.head_dim))
                .transpose()?,
            k_norm: k_norm
                .as_deref()
                .map(|name| up.vector(name, shape.head_dim))
                .transpose()?,
            post_norm: up.vector(&layer_names.post_norm, hidden)?,
            ffn,
        });
    }
    Ok((embed, head, norm, layers, up.bytes))
}

/// Refuses a tensor that is not BF16 at `shape`.
fn bf16(source: &dyn WeightSource, name: &str, shape: &[usize]) -> Result<(), String> {
    let spec = source.spec(name).map_err(|error| error.to_string())?;
    if spec.dtype != DType::BF16 || spec.shape != shape {
        return Err(format!(
            "{name} is {:?} {:?}; the shared dense device arm reads it as BF16 {shape:?}",
            spec.dtype, spec.shape
        ));
    }
    Ok(())
}

impl Shared {
    fn call<const N: usize>(&self, kernel: Kernel, call: Call<N>) -> Result<(), BackendError> {
        self.cuda
            .launch_async(kernel, call.grid, call.block, &call.args)
            .map_err(|error| BackendError::new(DEVICE, format!("{}: {error}", kernel.name())))
    }

    /// `output[rows][mat.rows] = input[rows][mat.columns] x matᵀ` on the kernel
    /// [route_bf16] picks under `policy` (`None` keeps the row GEMV).
    fn project(
        &self,
        mat: &GpuMat,
        input: u64,
        output: u64,
        rows: usize,
        policy: Option<WidePolicy>,
    ) -> Result<(), BackendError> {
        let gemv = |input: u64, output: u64, batch: usize| {
            self.call(
                self.kernels.gemv,
                calls::gemv_bf16(mat.pointer.0, input, output, (mat.rows, mat.columns), batch),
            )
        };
        match route_bf16(policy, mat.rows, mat.columns, rows) {
            Route::RowGemv => gemv(input, output, rows),
            Route::BatchedGemv => {
                for start in (0..rows).step_by(GEMV_MAX_BATCH) {
                    gemv(
                        input + (start * mat.columns * 4) as u64,
                        output + (start * mat.rows * 4) as u64,
                        GEMV_MAX_BATCH.min(rows - start),
                    )?;
                }
                Ok(())
            }
            Route::Wide => {
                let tile =
                    WideTile::call(mat.pointer.0, input, output, mat.rows, mat.columns, rows)
                        .map_err(|error| BackendError::new(DEVICE, error))?;
                self.cuda
                    .launch_async(self.kernels.wide, tile.grid, tile.block, &tile.args)
                    .map_err(|error| BackendError::new(DEVICE, format!("dory_mma_wide: {error}")))
            }
            route @ (Route::Skinny | Route::Refuse) => Err(BackendError::new(
                DEVICE,
                format!(
                    "the dense policy routed a {} x {} projection to {route:?}",
                    mat.rows, mat.columns
                ),
            )),
        }
    }

    fn norm(&self, plan: RmsNorm, rows: usize) -> Result<(), BackendError> {
        self.call(
            self.kernels.rmsnorm,
            plan.zero_centred(self.zero_centred).call(rows),
        )
    }

    fn launch(
        &self,
        kernel: Kernel,
        (grid, block): Launch,
        args: &[u64],
    ) -> Result<(), BackendError> {
        self.cuda
            .launch_async(kernel, grid, block, args)
            .map_err(|error| BackendError::new(DEVICE, format!("{}: {error}", kernel.name())))
    }

    /// `output = down(silu(gate(normed)) * up(normed))` over `rows`, in the MLP scratch.
    fn mlp(
        &self,
        mlp: &Mlp,
        b: &Buffers,
        output: Deviceptr,
        rows: usize,
        policy: Option<WidePolicy>,
    ) -> Result<(), BackendError> {
        self.project(&mlp.gate, b.normed.0, b.ffn_gate.0, rows, policy)?;
        self.project(&mlp.up, b.normed.0, b.ffn_up.0, rows, policy)?;
        self.call(
            self.kernels.swiglu,
            calls::swiglu(b.ffn_gate.0, b.ffn_up.0, b.ffn_up.0, rows * mlp.gate.rows),
        )?;
        self.project(&mlp.down, b.ffn_up.0, output.0, rows, policy)
    }

    /// A routed layer's MLP over `rows` into `b.projected`: the router on the row GEMV (so a
    /// prompt row picks the experts its decode row would), `q4e_route`, the shared plan's runs,
    /// the BF16 token passes, the shared expert and its gate, and the combine.
    fn routed(
        &self,
        layer: &Layer,
        b: &Buffers,
        rows: usize,
        policy: Option<WidePolicy>,
    ) -> Result<(), BackendError> {
        let (
            Some(routed),
            LayerFfn::Routed {
                router,
                table,
                shared,
                shared_gate,
            },
            Some(kernels),
        ) = (&layer.shape.routed, &layer.ffn, self.kernels.moe)
        else {
            return Err(BackendError::new(
                DEVICE,
                "a routed layer runs without its experts or the routed kernels",
            ));
        };
        let hidden = self.hidden;
        let (experts, top_k, intermediate) = (routed.experts, routed.top_k, routed.intermediate);
        self.project(router, b.normed.0, b.router.0, rows, None)?;
        self.launch(
            kernels.route,
            ((rows as u32, 1, 1), (256, 1, 1)),
            &[
                b.router.0,
                b.indices.0,
                b.weights.0,
                experts as u64,
                top_k as u64,
                routed.score,
                u64::from(routed.renormalise),
                u64::from(routed.scaling.to_bits()),
            ],
        )?;
        let plan = MoePlanCall::runs(
            b.indices.0,
            b.sorted.0,
            b.runs.0,
            (rows * top_k, experts, top_k),
        );
        self.launch(kernels.plan, MoePlanCall::launch(), &plan.args())?;
        self.launch(
            kernels.gate_up,
            token::gate_up_grid(intermediate, top_k, rows),
            &[
                table.0,
                b.indices.0,
                b.sorted.0,
                b.runs.0,
                b.normed.0,
                b.activated.0,
                top_k as u64,
                hidden as u64,
                intermediate as u64,
            ],
        )?;
        self.launch(
            kernels.down,
            token::down_grid(hidden, top_k, rows),
            &[
                table.0,
                b.indices.0,
                b.sorted.0,
                b.runs.0,
                b.weights.0,
                b.activated.0,
                b.expert_partials.0,
                top_k as u64,
                hidden as u64,
                intermediate as u64,
                top_k as u64,
            ],
        )?;
        match shared {
            Some(mlp) => self.mlp(mlp, b, b.shared_out, rows, policy)?,
            // The combine always adds a shared row; a layer without one adds zeros.
            None => self
                .cuda
                .memset_d8_async(b.shared_out, 0, rows * hidden * 4)
                .map_err(|error| BackendError::new(DEVICE, format!("the shared row: {error}")))?,
        }
        if let Some(gate) = shared_gate {
            self.project(gate, b.normed.0, b.shared_logit.0, rows, None)?;
        }
        self.launch(
            kernels.combine,
            combine_grid(hidden, rows),
            &[
                b.projected.0,
                b.expert_partials.0,
                b.shared_out.0,
                if shared_gate.is_some() {
                    b.shared_logit.0
                } else {
                    0
                },
                0,
                0,
                top_k as u64,
                hidden as u64,
                0,
            ],
        )
    }
}

/// Where each layer's key and value caches lie in a sequence's lane ([SequenceLanes], the lane
/// memory every other arm's sequences use): a `[capacity][kv_width]` BF16 run each, backed only as
/// far as the sequence has reached, from one pool the backend's sequences share.
#[derive(Debug)]
struct LaneParts {
    caches: Vec<[usize; 2]>,
}

impl LaneParts {
    fn plan(layout: &mut Layout, layers: &[Layer], capacity: usize) -> Self {
        Self {
            caches: layers
                .iter()
                .map(|layer| {
                    let row = layer.shape.kv_width() * 2;
                    [
                        layout.part(capacity * row, row, 0),
                        layout.part(capacity * row, row, 0),
                    ]
                })
                .collect(),
        }
    }
}

/// The f32 scratch one walk of [CHUNK_ROWS] rows takes, sized for the widest layer, and room for
/// `logit_rows` rows of logits.
struct Buffers {
    tokens: Deviceptr,
    x: Deviceptr,
    normed: Deviceptr,
    interleaved: Deviceptr,
    query: Deviceptr,
    gate: Deviceptr,
    key: Deviceptr,
    value: Deviceptr,
    partials: Deviceptr,
    attended: Deviceptr,
    projected: Deviceptr,
    ffn_gate: Deviceptr,
    ffn_up: Deviceptr,
    logits: Deviceptr,
    /// A routed layer's router logits, `[rows][experts]`.
    router: Deviceptr,
    /// The picked experts (i32) and their weights, `[rows][top_k]` each.
    indices: Deviceptr,
    weights: Deviceptr,
    /// The slots in expert order (i32), and each slot's run (two i32 a slot).
    sorted: Deviceptr,
    runs: Deviceptr,
    /// Every slot's SwiGLU row, `[rows * top_k][intermediate]`.
    activated: Deviceptr,
    /// Every slot's weighted down row, `[rows * top_k][hidden]`.
    expert_partials: Deviceptr,
    /// The shared expert's rows (zeros without one), and its gate's logit a row.
    shared_out: Deviceptr,
    shared_logit: Deviceptr,
    logit_rows: usize,
}

impl Buffers {
    fn all(&self) -> [&Deviceptr; 23] {
        [
            &self.tokens,
            &self.x,
            &self.normed,
            &self.interleaved,
            &self.query,
            &self.gate,
            &self.key,
            &self.value,
            &self.partials,
            &self.attended,
            &self.projected,
            &self.ffn_gate,
            &self.ffn_up,
            &self.logits,
            &self.router,
            &self.indices,
            &self.weights,
            &self.sorted,
            &self.runs,
            &self.activated,
            &self.expert_partials,
            &self.shared_out,
            &self.shared_logit,
        ]
    }
}

/// A walk's scratch on the device, freed with it. Nothing in it outlives a walk but the logits the
/// walk's caller reads back, so the slots of one engine share one.
struct Scratch {
    shared: Arc<Shared>,
    buffers: Buffers,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        self.shared.cuda.free_buffers(&self.buffers.all());
    }
}

impl Scratch {
    fn new(shared: Arc<Shared>, logit_rows: usize) -> Result<Self, String> {
        let widest = |of: fn(&Shape) -> usize| {
            shared
                .layers
                .iter()
                .map(|layer| of(&layer.shape))
                .max()
                .unwrap_or(0)
        };
        let rows = CHUNK_ROWS;
        let hidden = shared.hidden;
        let logit_rows = logit_rows.clamp(1, CHUNK_ROWS);
        let routed = |of: fn(&Routed) -> usize| {
            shared
                .layers
                .iter()
                .filter_map(|layer| layer.shape.routed.as_ref())
                .map(of)
                .max()
                .unwrap_or(0)
        };
        let (experts, slots) = (routed(|r| r.experts), rows * routed(|r| r.top_k));
        let any_routed = slots > 0;
        let floats = [
            rows,
            rows * hidden,
            rows * hidden,
            rows * widest(Shape::query_rows),
            rows * widest(Shape::query_width),
            rows * widest(Shape::query_width),
            rows * widest(Shape::kv_width),
            rows * widest(Shape::kv_width),
            widest(|shape| FullAttention::partial_floats(CHUNK_ROWS, shape.heads, shape.head_dim)),
            rows * widest(Shape::query_width),
            rows * hidden,
            rows * widest(|shape| shape.intermediate),
            rows * widest(|shape| shape.intermediate),
            logit_rows * shared.vocab,
            rows * experts,
            slots,
            slots,
            slots,
            2 * slots,
            slots * routed(|r| r.intermediate),
            slots * hidden,
            if any_routed { rows * hidden } else { 0 },
            if any_routed { rows } else { 0 },
        ];
        let mut pointers = Vec::with_capacity(floats.len());
        let mut total = 0usize;
        for count in floats {
            match shared.cuda.alloc(count.max(1) * 4, &mut total) {
                Ok(pointer) => pointers.push(pointer),
                Err(error) => {
                    let references: Vec<&Deviceptr> = pointers.iter().collect();
                    shared.cuda.free_buffers(&references);
                    return Err(error);
                }
            }
        }
        let p = |index: usize| pointers[index];
        let buffers = Buffers {
            tokens: p(0),
            x: p(1),
            normed: p(2),
            interleaved: p(3),
            query: p(4),
            gate: p(5),
            key: p(6),
            value: p(7),
            partials: p(8),
            attended: p(9),
            projected: p(10),
            ffn_gate: p(11),
            ffn_up: p(12),
            logits: p(13),
            router: p(14),
            indices: p(15),
            weights: p(16),
            sorted: p(17),
            runs: p(18),
            activated: p(19),
            expert_partials: p(20),
            shared_out: p(21),
            shared_logit: p(22),
            logit_rows,
        };
        Ok(Self { shared, buffers })
    }
}

/// One sequence on the device: its lane of BF16 key/value caches, the ids whose rows they hold,
/// the point it can go back to, and its sampler.
///
/// Every cache row is addressed by its position and attention reads only the rows before a
/// query's own, so the sequence goes back to any prefix of its history by moving its position:
/// the rewind point needs no copy of anything (as Dory's lanes, unlike the recurrent arms').
pub struct Sequence {
    shared: Arc<Shared>,
    lane: SequenceLane<LaneParts>,
    /// Each layer's keys and values in the lane.
    caches: Vec<(Deviceptr, Deviceptr)>,
    /// The ids whose rows the caches hold, in position order: the last prompt and every token
    /// stepped since. Its length is the sequence's position.
    history: Vec<u32>,
    /// The end of the last prompt, when [sparkflash_runtime::REWIND_ENV] keeps one.
    point: Option<usize>,
    /// How many of the last prompt's ids the sequence already held.
    cached: usize,
    /// A prompt running a part at a time, and the next id of it to forward.
    prompt: Option<(Vec<u32>, usize)>,
    sampler: Sampler,
    sampling: Sampling,
}

impl Drop for Sequence {
    fn drop(&mut self) {
        self.lane.free(&self.shared.cuda);
    }
}

impl Sequence {
    fn open(shared: Arc<Shared>) -> Result<Self, String> {
        let lane = SequenceLane::reserve(&shared.cuda, &shared.lanes)?;
        let base = lane.base();
        let caches = shared
            .lanes
            .parts()
            .caches
            .iter()
            .map(|[keys, values]| (base.offset(*keys), base.offset(*values)))
            .collect();
        Ok(Self {
            shared,
            lane,
            caches,
            history: Vec::new(),
            point: None,
            cached: 0,
            prompt: None,
            sampler: Sampler::new(),
            sampling: Sampling::default(),
        })
    }

    fn position(&self) -> usize {
        self.history.len()
    }

    /// Starts `tokens` as a prompt: past the history or the rewind point it extends
    /// ([sparkflash_runtime::resume]), else from the start, and runs its first part.
    fn begin(
        &mut self,
        b: &Buffers,
        tokens: &[u32],
        trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Part, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::new(DEVICE, "the prompt has no tokens"));
        }
        let start = match resume(&self.history, true, self.point, tokens) {
            Some(Resume::Extend(held) | Resume::Rewind(held)) => held,
            None => 0,
        };
        self.history.truncate(start);
        if start == 0 {
            self.point = None;
        }
        self.cached = start;
        self.prompt = Some((tokens.to_vec(), start));
        self.part(b, trace)
    }

    /// Runs the next [CHUNK_ROWS] ids of the prompt [Self::begin] started; after the last, the
    /// prompt's first step.
    fn part(
        &mut self,
        b: &Buffers,
        trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Part, BackendError> {
        let Some((tokens, next)) = self.prompt.take() else {
            return Err(BackendError::new(
                DEVICE,
                "the sequence has no prompt in parts",
            ));
        };
        let end = (next + CHUNK_ROWS).min(tokens.len());
        let shared = Arc::clone(&self.shared);
        walk(
            &shared,
            b,
            &mut [(&mut *self, &tokens[next..end])],
            shared.policy,
            trace,
        )?;
        if end < tokens.len() {
            self.prompt = Some((tokens, end));
            return Ok(Part::More);
        }
        let logits = head(&shared, b, end - next - 1, 1)?;
        self.sampler.reset();
        self.sampler.record_prompt(&tokens);
        let token = self.sampler.choose(&logits, &self.sampling);
        self.point = shared.rewind.then_some(tokens.len());
        Ok(Part::Done(StepOutput {
            next_token: Some(token),
            logits,
            draft: Vec::new(),
        }))
    }

    /// The whole of `tokens` as a prompt: [Self::begin], then every part.
    fn prefill(
        &mut self,
        b: &Buffers,
        tokens: &[u32],
        mut trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<StepOutput, BackendError> {
        let mut part = self.begin(b, tokens, trace.as_deref_mut())?;
        loop {
            match part {
                Part::Done(step) => return Ok(step),
                Part::More => part = self.part(b, trace.as_deref_mut())?,
            }
        }
    }
}

/// One walk over the segments' rows, laid out one after another: each sequence's ids at its own
/// position. The row-wise stages (norms, projections, the MLP, the residual adds) take every row
/// at once and the per-sequence stages (rotary, the cache appends, attention) each segment's
/// rows, so a row computes as it would alone wherever the projections take the same kernel: a
/// fused step passes no wide policy, and a row GEMV row is the same sum at any batch.
///
/// The residual of every row is left in `b.x`; each sequence's history takes its ids.
fn walk(
    shared: &Shared,
    b: &Buffers,
    segments: &mut [(&mut Sequence, &[u32])],
    policy: Option<WidePolicy>,
    mut trace: Option<&mut Vec<Vec<f32>>>,
) -> Result<(), BackendError> {
    let refuse = |message: String| Err(BackendError::new(DEVICE, message));
    let rows: usize = segments.iter().map(|(_, tokens)| tokens.len()).sum();
    if rows == 0 || rows > CHUNK_ROWS {
        return refuse(format!("a walk takes 1..={CHUNK_ROWS} rows, not {rows}"));
    }
    let mut ids = Vec::with_capacity(rows * 4);
    for (sequence, tokens) in segments.iter_mut() {
        if tokens.is_empty() {
            return refuse("a segment has no tokens".to_owned());
        }
        if let Some(token) = tokens.iter().find(|token| **token as usize >= shared.vocab) {
            return refuse(format!(
                "token {token} is outside the checkpoint vocabulary {}",
                shared.vocab
            ));
        }
        let reach = sequence.position() + tokens.len();
        if reach > shared.context {
            return refuse(format!(
                "the sequence is past the served context {}",
                shared.context
            ));
        }
        sequence
            .lane
            .back(&shared.cuda, reach)
            .map_err(|error| BackendError::new(DEVICE, format!("the lane memory: {error}")))?;
        ids.extend(tokens.iter().flat_map(|token| token.to_le_bytes()));
    }
    let kernels = shared.kernels;
    let hidden = shared.hidden;
    let device = |error: String| BackendError::new(DEVICE, error);
    shared.cuda.copy_htod(&b.tokens, &ids).map_err(device)?;
    shared.call(
        kernels.embed,
        calls::embed_rows(
            shared.embed.pointer.0,
            b.x.0,
            b.tokens.0,
            (hidden, hidden),
            rows,
        ),
    )?;
    for (index, layer) in shared.layers.iter().enumerate() {
        let shape = layer.shape;
        let (query_width, kv_width) = (shape.query_width(), shape.kv_width());
        shared.norm(
            RmsNorm::new(b.x.0, layer.input_norm.0, b.normed.0, hidden, shared.eps),
            rows,
        )?;
        let projected_query = if shape.gated { b.interleaved } else { b.query };
        shared.project(&layer.q, b.normed.0, projected_query.0, rows, policy)?;
        shared.project(&layer.k, b.normed.0, b.key.0, rows, policy)?;
        shared.project(&layer.v, b.normed.0, b.value.0, rows, policy)?;
        if shape.gated {
            shared.call(
                kernels.deinterleave,
                calls::deinterleave(
                    b.interleaved.0,
                    (b.query.0, b.gate.0),
                    (shape.heads, shape.head_dim),
                    rows,
                ),
            )?;
        }
        for (gains, values, width) in [
            (layer.q_norm, b.query, query_width),
            (layer.k_norm, b.key, kv_width),
        ] {
            if let Some(gains) = gains {
                shared.norm(
                    RmsNorm::new(values.0, gains.0, values.0, width, shared.eps)
                        .groups(shape.head_dim, true),
                    rows,
                )?;
            }
        }
        let mut first = 0usize;
        for (sequence, tokens) in segments.iter() {
            let count = tokens.len();
            let position = sequence.position();
            let row = |base: Deviceptr, width: usize| base.0 + (first * width * 4) as u64;
            if shape.rotary >= 2 {
                for (values, heads, width) in [
                    (b.query, shape.heads, query_width),
                    (b.key, shape.kv_heads, kv_width),
                ] {
                    let rope = Rope {
                        rows: row(values, width),
                        heads,
                        head_dim: shape.head_dim,
                        rotary: shape.rotary,
                        position,
                        theta: shape.theta,
                        row_stride: width,
                        tokens: count,
                    };
                    shared.call(kernels.rope, rope.call())?;
                }
            }
            let (keys, values) = sequence.caches[index];
            let at = (position * kv_width * 2) as u64;
            shared.call(
                kernels.cache,
                calls::cache_bf16(keys.0 + at, row(b.key, kv_width), count * kv_width),
            )?;
            shared.call(
                kernels.cache,
                calls::cache_bf16(values.0 + at, row(b.value, kv_width), count * kv_width),
            )?;
            // The partials are one segment's: the combine reads them before the next segment's
            // attention writes them, both on the one stream.
            let attention = FullAttention {
                query: row(b.query, query_width),
                keys: keys.0,
                values: values.0,
                partials: b.partials.0,
                position,
                rows: count,
                heads: shape.heads,
                kv_heads: shape.kv_heads,
                head_dim: shape.head_dim,
            };
            shared.call(kernels.attention, attention.call())?;
            let gate = if shape.gated {
                row(b.gate, query_width)
            } else {
                0
            };
            shared.call(
                kernels.combine,
                attention.combine(gate, row(b.attended, query_width)),
            )?;
            first += count;
        }
        shared.project(&layer.o, b.attended.0, b.projected.0, rows, policy)?;
        shared.call(kernels.add, calls::add(b.x.0, b.projected.0, rows * hidden))?;
        shared.norm(
            RmsNorm::new(b.x.0, layer.post_norm.0, b.normed.0, hidden, shared.eps),
            rows,
        )?;
        match &layer.ffn {
            LayerFfn::Dense(mlp) => shared.mlp(mlp, b, b.projected, rows, policy)?,
            LayerFfn::Routed { .. } => shared.routed(layer, b, rows, policy)?,
        }
        shared.call(kernels.add, calls::add(b.x.0, b.projected.0, rows * hidden))?;
        if let Some(trace) = trace.as_deref_mut() {
            let mut bytes = vec![0u8; rows * hidden * 4];
            shared.cuda.copy_dtoh(&mut bytes, &b.x).map_err(device)?;
            trace[index].extend(floats(&bytes));
        }
    }
    for (sequence, tokens) in segments.iter_mut() {
        sequence.history.extend_from_slice(tokens);
    }
    Ok(())
}

/// The logits of `count` residual rows of the last walk from row `first`, row after row: the
/// final norm and the head on the row GEMV.
fn head(
    shared: &Shared,
    b: &Buffers,
    first: usize,
    count: usize,
) -> Result<Vec<f32>, BackendError> {
    if count == 0 || count > b.logit_rows {
        return Err(BackendError::new(
            DEVICE,
            format!(
                "the scratch holds 1..={} rows of logits, not {count}",
                b.logit_rows
            ),
        ));
    }
    let hidden = shared.hidden;
    shared.norm(
        RmsNorm::new(
            b.x.0 + (first * hidden * 4) as u64,
            shared.norm.0,
            b.normed.0,
            hidden,
            shared.eps,
        ),
        count,
    )?;
    shared.project(&shared.head, b.normed.0, b.logits.0, count, None)?;
    let mut bytes = vec![0u8; count * shared.vocab * 4];
    shared
        .cuda
        .copy_dtoh(&mut bytes, &b.logits)
        .map_err(|error| BackendError::new(DEVICE, format!("the logits: {error}")))?;
    Ok(floats(&bytes))
}

/// One fused step: a row a member, each its own token, and each member's next.
fn step(
    shared: &Shared,
    b: &Buffers,
    members: &mut [(&mut Sequence, u32)],
    trace: Option<&mut Vec<Vec<f32>>>,
) -> Result<Vec<StepOutput>, BackendError> {
    let tokens: Vec<u32> = members.iter().map(|(_, token)| *token).collect();
    let mut segments: Vec<(&mut Sequence, &[u32])> = members
        .iter_mut()
        .zip(tokens.chunks(1))
        .map(|((sequence, _), token)| (&mut **sequence, token))
        .collect();
    walk(shared, b, &mut segments, None, trace)?;
    let logits = head(shared, b, 0, segments.len())?;
    Ok(segments
        .iter_mut()
        .zip(logits.chunks(shared.vocab))
        .map(|((sequence, token), logits)| {
            sequence.sampler.record(token[0]);
            let chosen = sequence.sampler.choose(logits, &sequence.sampling);
            StepOutput {
                next_token: Some(chosen),
                logits: logits.to_vec(),
                draft: Vec::new(),
            }
        })
        .collect())
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

/// One sequence with its own scratch, outside the fused engine: the parity gate's session, which
/// records each layer's residual.
pub struct DenseDeviceSession {
    sequence: Sequence,
    scratch: Scratch,
    /// Each layer's residual after it, every row in position order, while tracing.
    trace: Option<Vec<Vec<f32>>>,
}

impl DenseDeviceSession {
    fn open(shared: Arc<Shared>) -> Result<Self, String> {
        Ok(Self {
            scratch: Scratch::new(Arc::clone(&shared), 1)?,
            sequence: Sequence::open(shared)?,
            trace: None,
        })
    }

    /// Starts (or stops) recording each layer's residual after it, every row.
    pub fn trace(&mut self, on: bool) {
        self.trace = on.then(|| vec![Vec::new(); self.sequence.shared.layers.len()]);
    }

    /// The recorded residuals, a `[positions][hidden]` row-major run a layer, and a fresh record.
    #[must_use]
    pub fn take_trace(&mut self) -> Vec<Vec<f32>> {
        let layers = self.sequence.shared.layers.len();
        self.trace
            .as_mut()
            .map(|trace| std::mem::replace(trace, vec![Vec::new(); layers]))
            .unwrap_or_default()
    }

    /// Runs `tokens` from the session's position, [CHUNK_ROWS] rows a walk, and returns the last
    /// row's logits. The sampler does not see them.
    ///
    /// # Errors
    /// A token outside the vocabulary, a sequence past the served context, or a device failure.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, BackendError> {
        let shared = Arc::clone(&self.sequence.shared);
        let b = &self.scratch.buffers;
        let mut last = 0;
        for chunk in tokens.chunks(CHUNK_ROWS) {
            walk(
                &shared,
                b,
                &mut [(&mut self.sequence, chunk)],
                shared.policy,
                self.trace.as_mut(),
            )?;
            last = chunk.len();
        }
        if last == 0 {
            return Err(BackendError::new(DEVICE, "the step has no tokens"));
        }
        head(&shared, b, last - 1, 1)
    }
}

impl ModelSession for DenseDeviceSession {
    fn set_sampling(&mut self, sampling: Sampling) {
        // Penalties, temperature, top-k and top-p run on the host over the row every step copies
        // back, so a sampled request is served rather than approximated by the greedy pick.
        self.sequence.sampling = sampling;
        self.sequence.sampler.reset();
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        // A session outside the engine starts every prompt over.
        self.sequence.history.clear();
        self.sequence.point = None;
        self.sequence
            .prefill(&self.scratch.buffers, tokens, self.trace.as_mut())
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        let shared = Arc::clone(&self.sequence.shared);
        let mut steps = step(
            &shared,
            &self.scratch.buffers,
            &mut [(&mut self.sequence, token)],
            self.trace.as_mut(),
        )?;
        Ok(steps.remove(0))
    }

    fn reusable_prefix(&self) -> Option<&[u32]> {
        // Kept conversations are the fused engine's ([DenseLanes]).
        None
    }
}

/// The arm behind the shared fused engine ([FusedEngine]): a slot is one [Sequence], a step is
/// one walk over every member's row, and a prompt runs a [CHUNK_ROWS]-row part at a time
/// between the members' steps. The engine thread runs one job at a time, so the slots share one
/// scratch.
pub struct DenseLanes {
    shared: Arc<Shared>,
    scratch: Mutex<Scratch>,
    /// The most members one walk holds.
    rows: usize,
}

impl DenseLanes {
    fn new(shared: Arc<Shared>, capacity: usize) -> Result<Self, BackendError> {
        let rows = capacity.clamp(1, CHUNK_ROWS);
        let scratch = Scratch::new(Arc::clone(&shared), rows)
            .map_err(|error| BackendError::new(DEVICE, error))?;
        Ok(Self {
            shared,
            scratch: Mutex::new(scratch),
            rows,
        })
    }

    fn scratch(&self) -> Result<std::sync::MutexGuard<'_, Scratch>, BackendError> {
        self.scratch
            .lock()
            .map_err(|_| BackendError::new(DEVICE, "the walk's scratch is poisoned"))
    }
}

impl FusedArm for DenseLanes {
    type Slot = Sequence;

    fn open_slot(&self) -> Result<Sequence, BackendError> {
        Sequence::open(Arc::clone(&self.shared)).map_err(|error| BackendError::new(DEVICE, error))
    }

    fn holds_request(&self, slot: &Sequence) -> bool {
        !slot.history.is_empty() || slot.prompt.is_some()
    }

    fn renew(&self, slot: &mut Sequence) -> Result<(), BackendError> {
        let mut fresh = self.open_slot()?;
        // The request's sampling reached the old sequence before its prompt replaced it.
        fresh.sampling = slot.sampling;
        *slot = fresh;
        Ok(())
    }

    fn set_sampling(&self, slot: &mut Sequence, sampling: Sampling) {
        slot.sampling = sampling;
        slot.sampler.reset();
    }

    fn prefill(&self, slot: &mut Sequence, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        let scratch = self.scratch()?;
        slot.prefill(&scratch.buffers, tokens, None)
    }

    /// A prompt's [CHUNK_ROWS]-row chunks run one a part; the arm serves text, so a prompt with
    /// images is refused as [FusedArm::prefill_images] refuses it.
    fn begin_prompt(
        &self,
        slot: &mut Sequence,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<Part, BackendError> {
        if !images.is_empty() {
            return self.prefill_images(slot, tokens, images).map(Part::Done);
        }
        let scratch = self.scratch()?;
        slot.begin(&scratch.buffers, tokens, None)
    }

    fn next_part(&self, slot: &mut Sequence) -> Result<Part, BackendError> {
        let scratch = self.scratch()?;
        slot.part(&scratch.buffers, None)
    }

    fn step(&self, members: &mut [(Sequence, u32)]) -> Result<Vec<StepOutput>, BackendError> {
        let scratch = self.scratch()?;
        let mut members: Vec<(&mut Sequence, u32)> = members
            .iter_mut()
            .map(|(sequence, token)| (sequence, *token))
            .collect();
        step(&self.shared, &scratch.buffers, &mut members, None)
    }

    fn speculation(&self, _slot: &Sequence) -> Speculation {
        Speculation::default()
    }

    /// A slot keeps its sequence between requests: the next prompt that extends it forwards only
    /// the rest ([Sequence::begin]).
    fn history<'s>(&self, slot: &'s Sequence) -> Option<&'s [u32]> {
        Some(&slot.history)
    }

    /// The end of the last prompt, which a prompt that parts from the reply goes back to.
    fn rewind_prefix<'s>(&self, slot: &'s Sequence) -> Option<&'s [u32]> {
        slot.point
            .filter(|point| *point <= slot.history.len())
            .map(|point| &slot.history[..point])
    }

    fn cached_tokens(&self, slot: &Sequence) -> usize {
        slot.cached
    }

    /// The sequences' caches share one pool: a prompt that does not continue the slot's history
    /// starts over, so it counts the chunks the reset gives back.
    fn room(&self, slot: &Sequence, prompt: Option<&[u32]>) -> bool {
        match prompt {
            Some(tokens) => slot.lane.room(tokens.len(), !self.continues(slot, tokens)),
            None => slot.lane.room(slot.position() + 1, false),
        }
    }

    fn max_batch(&self) -> usize {
        self.rows
    }

    fn ledger(&self) -> Ledger {
        let mut ledger = Ledger::new("dense");
        ledger.set(
            Strategy::PromptParts,
            Mode::native(format!("{CHUNK_ROWS}-row chunks")),
        );
        ledger.drafting(None, false, false);
        let reuse = if self.shared.rewind {
            "a prompt that extends a lane's sequence, or goes back to the last prompt's end"
        } else {
            "a prompt that extends a lane's sequence; no rewind (SPARKFLASH_REWIND=off)"
        };
        ledger.set(Strategy::PrefixReuse, Mode::native(reuse));
        ledger
    }
}

#[cfg(test)]
mod tests {
    use super::{CHUNK_ROWS, profile_policy};
    use sparkflash_model::profile::ModelProfile;
    use sparkflash_ops::projection::WidePolicy;

    #[test]
    fn the_profile_fields_the_arm_does_not_honour_are_refused() {
        assert_eq!(
            profile_policy(&ModelProfile::default()),
            Ok(Some(WidePolicy::DENSE))
        );
        let mut row_gemv = ModelProfile::default();
        row_gemv.prefill.tensor_cores = Some(false);
        assert_eq!(profile_policy(&row_gemv), Ok(None));
        let refused = |edit: fn(&mut ModelProfile)| {
            let mut profile = ModelProfile::default();
            edit(&mut profile);
            profile_policy(&profile).expect_err("the field is refused")
        };
        assert!(refused(|profile| profile.prefill.chunk = Some(CHUNK_ROWS * 2)).contains("chunk"));
        assert!(refused(|profile| profile.prefill.enabled = Some(false)).contains("prefill"));
        assert!(refused(|profile| profile.prefill.exact = Some(true)).contains("exact"));
        assert!(refused(|profile| profile.decode.tensor_cores = Some(true)).contains("decode"));
        assert!(refused(|profile| profile.speculation.enabled = Some(true)).contains("drafter"));
        assert!(refused(|profile| profile.sequence.parallel = Some(true)).contains("one rank"));
        let mut lanes = ModelProfile::default();
        lanes.runtime.concurrency = Some(4);
        assert!(
            profile_policy(&lanes).is_ok(),
            "the engine's lanes are honoured"
        );
        let mut chunk = ModelProfile::default();
        chunk.prefill.chunk = Some(CHUNK_ROWS);
        assert!(
            profile_policy(&chunk).is_ok(),
            "the arm's own chunk is honoured"
        );
    }
}

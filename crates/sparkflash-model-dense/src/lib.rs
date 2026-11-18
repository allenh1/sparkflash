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
//! A dense BF16 grouped-query decoder (embedding, RMSNorm, grouped-query attention with or
//! without the per-head output gate and q/k norms, plain rotate-half RoPE, a dense SwiGLU, a
//! plain residual and the head) needs no code of its own: this arm reads the shared
//! [Architecture], resolves the checkpoint's own spelling through [sparkflash_ops::naming],
//! uploads every tensor through [DraftUploader] and launches the shared token-image bodies through
//! their typed [sparkflash_ops::calls]. Its prompt projections take the shared wide tile under
//! [WidePolicy::DENSE]; decode rows keep the row GEMV.
//!
//! Which checkpoints it takes is [sparkflash_ops::parts::dense_device_refusal]: nothing the host
//! reference does not execute, and nothing the reference does that these kernels do not. Its
//! gate is a per-layer parity report against that reference (the CLI's `dense_parity` test),
//! not a recorded digest: the arm keeps its KV cache in BF16 where the reference keeps f32, so a
//! row agrees to a tolerance rather than to the bit. It holds one sequence on one rank.

use std::sync::Arc;

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_kernels::images::{dory_cubin, qwen4_attn_cubin, qwen4_exp_cubin};
use sparkflash_model::architecture::{AttentionKind, Ffn};
use sparkflash_model::backend::{BackendError, ModelBackend, ModelSession, Sampling, StepOutput};
use sparkflash_model::{Architecture, Capabilities, ModelDescriptor};
use sparkflash_ops::calls::{self, Call, FullAttention, RmsNorm, Rope};
use sparkflash_ops::dflash::DraftUploader;
use sparkflash_ops::engine_plan::{EnginePlan, PlanError};
use sparkflash_ops::gemv::{CudaHandle, GEMV_MAX_BATCH, GpuMat};
use sparkflash_ops::naming::{AttentionNames, FfnNames, TensorNames};
use sparkflash_ops::projection::{Route, WIDE_SHARED, WidePolicy, WideTile, route_bf16};
use sparkflash_ops::sampler::Sampler;
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
    intermediate: usize,
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
        let Ffn::Dense(dense) = &layer.ffn else {
            return Err(format!("layer {index} routes experts"));
        };
        let head_dim = layer.attention.head_dim;
        Ok(Self {
            heads: layer.attention.heads,
            kv_heads,
            head_dim,
            gated: output_gate,
            rotary: layer.attention.rope.rotated_dims.min(head_dim),
            theta: layer.attention.rope.theta.max(1.0) as f32,
            intermediate: dense.intermediate,
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
    gate: GpuMat,
    up: GpuMat,
    down: GpuMat,
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
}

// SAFETY: a Kernel is an opaque driver function word, and the handle only names it. The context
// that owns the module lives in the shared state, which outlives every session.
unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

impl Kernels {
    fn load(cuda: &Cuda) -> Result<Self, String> {
        let named = |image: &'static [u8], name: &'static str| {
            cuda.load_named(image, name).map(|(_, kernel)| kernel)
        };
        Ok(Self {
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
    /// Every weight allocation, freed once.
    owned: Vec<Deviceptr>,
    weight_bytes: usize,
}

impl Drop for Shared {
    fn drop(&mut self) {
        let references: Vec<&Deviceptr> = self.owned.iter().collect();
        self.cuda.free_buffers(&references);
    }
}

/// A dense grouped-query checkpoint on the device.
pub struct DenseDeviceBackend {
    descriptor: ModelDescriptor,
    shared: Arc<Shared>,
}

impl DenseDeviceBackend {
    /// Whether this build carries every kernel the arm launches.
    #[must_use]
    pub fn available() -> bool {
        sparkflash_ops::parts::dense_device_images().is_ok()
    }

    /// Opens `architecture`'s checkpoint from `source` under `descriptor`, with caches for
    /// `context` tokens.
    ///
    /// # Errors
    /// A checkpoint the arm does not execute ([sparkflash_ops::parts::dense_device_refusal]), a
    /// missing kernel, a profile field the arm cannot honour, a tensor that is not BF16 at the
    /// shape the architecture names, a zero context or one past the checkpoint's window, or a
    /// CUDA context that cannot open.
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
        let policy = profile_policy(sparkflash_model::profile::active())?;
        let names = sparkflash_ops::naming::resolve(architecture, source)?;
        let cuda = CudaHandle::new(Cuda::load()?);
        sparkflash_kernels::cuda_driver::enable_forward_async_launches();
        let kernels = Kernels::load(&cuda)?;
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
        descriptor.context_tokens = context;
        Ok(Self {
            descriptor,
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
        Ok(Box::new(
            self.session()
                .map_err(|error| BackendError::new(DEVICE, error))?,
        ))
    }
}

/// The wide-tile policy a typed profile asks of the arm, or the field it cannot honour.
///
/// The arm's chunk, its prompt-and-decode split and its one sequence are fixed by its kernels;
/// a field it does not honour is refused by name rather than accepted and dropped. The one field
/// it does honour is `prefill.tensor_cores = false`, which keeps every projection on the row
/// GEMV.
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
    if let Some(concurrency) = profile.runtime.concurrency
        && concurrency > 1
    {
        return refuse(&format!(
            "holds one sequence a session, not runtime.concurrency = {concurrency}"
        ));
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

type Uploaded = (GpuMat, GpuMat, Deviceptr, Vec<Layer>, usize);

/// Uploads every tensor `names` resolves, each checked BF16 at its architecture shape; every
/// allocation lands in `owned` as it is made, so a failure part way frees what was uploaded.
fn upload(
    cuda: &Cuda,
    architecture: &Architecture,
    names: &TensorNames,
    source: &dyn WeightSource,
    owned: &mut Vec<Deviceptr>,
) -> Result<Uploaded, String> {
    let uploader = DraftUploader::new(cuda, source);
    let hidden = architecture.hidden_size;
    let vocab = architecture.vocab_size;
    let mut bytes = 0usize;
    let mut mat = |name: &str, rows: usize, columns: usize| -> Result<GpuMat, String> {
        bf16(source, name, &[rows, columns])?;
        let mat = uploader.mat(name, rows, columns)?;
        owned.push(mat.pointer);
        bytes += rows * columns * 2;
        Ok(mat)
    };
    let embed = mat(&names.embed, vocab, hidden)?;
    let head = if names.head_tied() {
        embed
    } else {
        mat(&names.head, vocab, hidden)?
    };
    let mut matrices = Vec::with_capacity(architecture.layers.len());
    for (index, layer_names) in names.layers.iter().enumerate() {
        let shape = Shape::of(architecture, index)?;
        let AttentionNames::Gqa { q, k, v, o, .. } = &layer_names.attention else {
            return Err(format!("layer {index} names another attention's tensors"));
        };
        let FfnNames::Dense(ffn) = &layer_names.ffn else {
            return Err(format!("layer {index} names a routed MLP's tensors"));
        };
        matrices.push((
            shape,
            [
                mat(q, shape.query_rows(), hidden)?,
                mat(k, shape.kv_width(), hidden)?,
                mat(v, shape.kv_width(), hidden)?,
                mat(o, hidden, shape.query_width())?,
                mat(&ffn.gate, shape.intermediate, hidden)?,
                mat(&ffn.up, shape.intermediate, hidden)?,
                mat(&ffn.down, hidden, shape.intermediate)?,
            ],
        ));
    }
    let mut vector = |name: &str, width: usize| -> Result<Deviceptr, String> {
        bf16(source, name, &[width])?;
        let pointer = uploader.vector(name, width)?;
        owned.push(pointer);
        bytes += width * 2;
        Ok(pointer)
    };
    let norm = vector(&names.norm, hidden)?;
    let mut layers = Vec::with_capacity(matrices.len());
    for ((shape, [q, k, v, o, gate, up, down]), layer_names) in
        matrices.into_iter().zip(&names.layers)
    {
        let AttentionNames::Gqa { q_norm, k_norm, .. } = &layer_names.attention else {
            unreachable!("checked above");
        };
        layers.push(Layer {
            shape,
            input_norm: vector(&layer_names.input_norm, hidden)?,
            q,
            k,
            v,
            o,
            q_norm: q_norm
                .as_deref()
                .map(|name| vector(name, shape.head_dim))
                .transpose()?,
            k_norm: k_norm
                .as_deref()
                .map(|name| vector(name, shape.head_dim))
                .transpose()?,
            post_norm: vector(&layer_names.post_norm, hidden)?,
            gate,
            up,
            down,
        });
    }
    Ok((embed, head, norm, layers, bytes))
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
    /// [route_bf16] picks under the arm's policy.
    fn project(
        &self,
        mat: &GpuMat,
        input: u64,
        output: u64,
        rows: usize,
    ) -> Result<(), BackendError> {
        let gemv = |input: u64, output: u64, batch: usize| {
            self.call(
                self.kernels.gemv,
                calls::gemv_bf16(mat.pointer.0, input, output, (mat.rows, mat.columns), batch),
            )
        };
        match route_bf16(self.policy, mat.rows, mat.columns, rows) {
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
}

/// The f32 scratch one pass of [CHUNK_ROWS] rows takes, sized for the widest layer.
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
}

impl Buffers {
    fn all(&self) -> [&Deviceptr; 14] {
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
        ]
    }
}

/// One sequence on the device: its scratch, its BF16 key/value caches and its sampler.
pub struct DenseDeviceSession {
    shared: Arc<Shared>,
    buffers: Buffers,
    /// Each layer's `[context][kv_width]` BF16 keys and values.
    caches: Vec<(Deviceptr, Deviceptr)>,
    position: usize,
    sampler: Sampler,
    sampling: Sampling,
    /// Each layer's residual after it, every row in position order, while tracing.
    trace: Option<Vec<Vec<f32>>>,
}

// SAFETY: the session's device pointers name allocations it alone writes; the shared weights are
// read-only, and the context handle is Send.
unsafe impl Send for DenseDeviceSession {}

impl Drop for DenseDeviceSession {
    fn drop(&mut self) {
        let mut references: Vec<&Deviceptr> = self.buffers.all().to_vec();
        for (keys, values) in &self.caches {
            references.push(keys);
            references.push(values);
        }
        self.shared.cuda.free_buffers(&references);
    }
}

impl DenseDeviceSession {
    fn open(shared: Arc<Shared>) -> Result<Self, String> {
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
            shared.vocab,
        ];
        let mut pointers = Vec::with_capacity(floats.len() + 2 * shared.layers.len());
        let mut total = 0usize;
        let allocated = (|| {
            for count in floats {
                pointers.push(shared.cuda.alloc(count.max(1) * 4, &mut total)?);
            }
            let mut caches = Vec::with_capacity(shared.layers.len());
            for layer in &shared.layers {
                let bytes = shared.context * layer.shape.kv_width() * 2;
                let keys = shared.cuda.alloc(bytes.max(4), &mut total)?;
                pointers.push(keys);
                let values = shared.cuda.alloc(bytes.max(4), &mut total)?;
                pointers.push(values);
                caches.push((keys, values));
            }
            Ok::<_, String>(caches)
        })();
        let caches = match allocated {
            Ok(caches) => caches,
            Err(error) => {
                let references: Vec<&Deviceptr> = pointers.iter().collect();
                shared.cuda.free_buffers(&references);
                return Err(error);
            }
        };
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
        };
        Ok(Self {
            shared,
            buffers,
            caches,
            position: 0,
            sampler: Sampler::new(),
            sampling: Sampling::default(),
            trace: None,
        })
    }

    /// Starts (or stops) recording each layer's residual after it, every row.
    pub fn trace(&mut self, on: bool) {
        self.trace = on.then(|| vec![Vec::new(); self.shared.layers.len()]);
    }

    /// The recorded residuals, a `[positions][hidden]` row-major run a layer, and a fresh record.
    #[must_use]
    pub fn take_trace(&mut self) -> Vec<Vec<f32>> {
        let layers = self.shared.layers.len();
        self.trace
            .as_mut()
            .map(|trace| std::mem::replace(trace, vec![Vec::new(); layers]))
            .unwrap_or_default()
    }

    /// Runs `tokens` from the session's position and returns the last row's logits.
    ///
    /// # Errors
    /// A token outside the vocabulary, a sequence past the served context, or a device failure.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, BackendError> {
        let refuse = |message: String| Err(BackendError::new(DEVICE, message));
        if tokens.is_empty() {
            return refuse("the step has no tokens".to_owned());
        }
        if let Some(token) = tokens
            .iter()
            .find(|token| **token as usize >= self.shared.vocab)
        {
            return refuse(format!(
                "token {token} is outside the checkpoint vocabulary {}",
                self.shared.vocab
            ));
        }
        if self.position + tokens.len() > self.shared.context {
            return refuse(format!(
                "the sequence is past the served context {}",
                self.shared.context
            ));
        }
        let mut last = 0;
        for chunk in tokens.chunks(CHUNK_ROWS) {
            self.pass(chunk)?;
            last = chunk.len();
        }
        let shared = Arc::clone(&self.shared);
        let b = &self.buffers;
        let hidden = shared.hidden;
        shared.norm(
            RmsNorm::new(
                b.x.0 + ((last - 1) * hidden * 4) as u64,
                shared.norm.0,
                b.normed.0,
                hidden,
                shared.eps,
            ),
            1,
        )?;
        shared.project(&shared.head, b.normed.0, b.logits.0, 1)?;
        let mut bytes = vec![0u8; shared.vocab * 4];
        shared
            .cuda
            .copy_dtoh(&mut bytes, &b.logits)
            .map_err(|error| BackendError::new(DEVICE, format!("the logits: {error}")))?;
        Ok(floats(&bytes))
    }

    /// One pass of at most [CHUNK_ROWS] rows through every layer.
    fn pass(&mut self, tokens: &[u32]) -> Result<(), BackendError> {
        let shared = Arc::clone(&self.shared);
        let kernels = shared.kernels;
        let b = &self.buffers;
        let rows = tokens.len();
        let position = self.position;
        let hidden = shared.hidden;
        let device = |error: String| BackendError::new(DEVICE, error);
        let ids: Vec<u8> = tokens
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect();
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
            shared.project(&layer.q, b.normed.0, projected_query.0, rows)?;
            shared.project(&layer.k, b.normed.0, b.key.0, rows)?;
            shared.project(&layer.v, b.normed.0, b.value.0, rows)?;
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
            if shape.rotary >= 2 {
                for (values, heads, width) in [
                    (b.query, shape.heads, query_width),
                    (b.key, shape.kv_heads, kv_width),
                ] {
                    let rope = Rope {
                        rows: values.0,
                        heads,
                        head_dim: shape.head_dim,
                        rotary: shape.rotary,
                        position,
                        theta: shape.theta,
                        row_stride: width,
                        tokens: rows,
                    };
                    shared.call(kernels.rope, rope.call())?;
                }
            }
            let (keys, values) = self.caches[index];
            let at = (position * kv_width * 2) as u64;
            shared.call(
                kernels.cache,
                calls::cache_bf16(keys.0 + at, b.key.0, rows * kv_width),
            )?;
            shared.call(
                kernels.cache,
                calls::cache_bf16(values.0 + at, b.value.0, rows * kv_width),
            )?;
            let attention = FullAttention {
                query: b.query.0,
                keys: keys.0,
                values: values.0,
                partials: b.partials.0,
                position,
                rows,
                heads: shape.heads,
                kv_heads: shape.kv_heads,
                head_dim: shape.head_dim,
            };
            shared.call(kernels.attention, attention.call())?;
            let gate = if shape.gated { b.gate.0 } else { 0 };
            shared.call(kernels.combine, attention.combine(gate, b.attended.0))?;
            shared.project(&layer.o, b.attended.0, b.projected.0, rows)?;
            shared.call(kernels.add, calls::add(b.x.0, b.projected.0, rows * hidden))?;
            shared.norm(
                RmsNorm::new(b.x.0, layer.post_norm.0, b.normed.0, hidden, shared.eps),
                rows,
            )?;
            shared.project(&layer.gate, b.normed.0, b.ffn_gate.0, rows)?;
            shared.project(&layer.up, b.normed.0, b.ffn_up.0, rows)?;
            let width = rows * shape.intermediate;
            shared.call(
                kernels.swiglu,
                calls::swiglu(b.ffn_gate.0, b.ffn_up.0, b.ffn_up.0, width),
            )?;
            shared.project(&layer.down, b.ffn_up.0, b.projected.0, rows)?;
            shared.call(kernels.add, calls::add(b.x.0, b.projected.0, rows * hidden))?;
            if let Some(trace) = self.trace.as_mut() {
                let mut bytes = vec![0u8; rows * hidden * 4];
                shared.cuda.copy_dtoh(&mut bytes, &b.x).map_err(device)?;
                trace[index].extend(floats(&bytes));
            }
        }
        self.position += rows;
        Ok(())
    }
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

impl ModelSession for DenseDeviceSession {
    fn set_sampling(&mut self, sampling: Sampling) {
        // Penalties, temperature, top-k and top-p run on the host over the row every step copies
        // back, so a sampled request is served rather than approximated by the greedy pick.
        self.sampling = sampling;
        self.sampler.reset();
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        let logits = self.forward(tokens)?;
        self.sampler.reset();
        self.sampler.record_prompt(tokens);
        let token = self.sampler.choose(&logits, &self.sampling);
        Ok(StepOutput {
            next_token: Some(token),
            logits,
            draft: Vec::new(),
        })
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        let logits = self.forward(&[token])?;
        self.sampler.record(token);
        let chosen = self.sampler.choose(&logits, &self.sampling);
        Ok(StepOutput {
            next_token: Some(chosen),
            logits,
            draft: Vec::new(),
        })
    }

    fn reusable_prefix(&self) -> Option<&[u32]> {
        // A fresh request opens a fresh session; the arm does not promise a prefix state yet.
        None
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
        assert!(refused(|profile| profile.runtime.concurrency = Some(4)).contains("concurrency"));
        let mut chunk = ModelProfile::default();
        chunk.prefill.chunk = Some(CHUNK_ROWS);
        assert!(
            profile_policy(&chunk).is_ok(),
            "the arm's own chunk is honoured"
        );
    }
}

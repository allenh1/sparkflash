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

//! The SenseNova-Loop device arm.
//!
//! The walk follows the host reference's execution plan exactly: each of the 56 visits owns a
//! KV cache or a GDN state, the physical layer's weights are uploaded once, and the two mHC
//! passes are the only model-specific launches. Every projection and every block body is the
//! shared Qwen token, attention or scan kernel. A prompt is fed one row at a time, so prefill
//! and decode use the same arithmetic and the same weight pass; the tensor-core prefill is the
//! next change on this plan.

use std::path::Path;
use std::sync::Arc;

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_kernels::images::{
    qwen4_attn_cubin, qwen4_exp_cubin, qwen4_prefill_cubin, qwen4_scan_cubin, sensenova_cubin,
};
use sparkflash_model::ModelDescriptor;
use sparkflash_model::backend::{BackendError, ModelBackend, ModelSession, Sampling, StepOutput};
use sparkflash_model::profile::{KvCacheMode, active};
use sparkflash_ops::gemv::{
    GEMV_MAX_BATCH, GpuMat, GpuMatE4m3Tile, TILED_GEMV_PARTIALS, gemv_e4m3_tile,
};
use sparkflash_ops::sampler::Sampler;
use sparkflash_weights::safetensors::open_checkpoint;
use sparkflash_weights::{DType, TensorSpec, WeightSource};

use crate::config::{LayerType, SenseNovaConfig};
use crate::forward::yarn_attention_scale;
use crate::tensor::SenseNovaTensorPlan;

const DEVICE: &str = "sensenova_device";
const BLOCK: u32 = 256;

/// The attention split grid: one block a 256-key tile, at most. The q8 kernel derives the splits
/// a row actually takes from the device position, so a constant grid is correct at every index
/// and is what lets the whole visit list be one replayed graph.
const FULL_SPLITS: u32 = 16;

/// The most rows one forward takes: a verify is the anchor plus drafts, and a prefill chunk is
/// the same shape. The shared GEMV's register tile (`GEMV_MAX_BATCH`) is the bound.
const ROWS: usize = 8;

/// The rows one prefill chunk takes. The tensor-core projection tiles 128 rows, so a chunk is
/// one tile row; the decode GEMV's eight-row tile stays for the verify.
const PREFILL_ROWS: usize = 512;

/// The dynamic shared memory `q4p_dense_fp8` asks for: GLM_Q8_SHARED in
/// shared_prefill_mma_sm121.cuh - four stages of a 128x32 activation tile, a 128x32 weight tile
/// and the rows' activation scales, then the rows' activation rows and the column tile's scales.
const DENSE_SHARED: u32 = 4 * (128 * 64 + 128 * 64 + 128 * 4) + 128 * 4 + 128 * 4;

/// Serialises graph capture: the driver's capture mode is process-wide, so two sessions must not
/// build their graphs at once.
static CAPTURE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Where a forward's time goes, one launch group at a time.
///
/// Under `SPARKFLASH_SENSENOVA_TIME=2` each mark drains the stream, so a mark's span is the work
/// the launches before it queued: the numbers are the device's own, at the cost of one round trip
/// a mark. The spans are summed by name and printed once at the end of the forward, which is what
/// makes a prefill's stage budget readable without a profiler.
#[derive(Debug)]
struct StageClock {
    on: bool,
    last: std::time::Instant,
    spans: Vec<(&'static str, f64)>,
}

impl StageClock {
    fn new(on: bool) -> Self {
        StageClock {
            on,
            last: std::time::Instant::now(),
            spans: Vec::new(),
        }
    }

    /// Closes the span since the last mark, named `label`, once the stream is drained.
    fn mark<E>(
        &mut self,
        label: &'static str,
        sync: impl FnOnce() -> Result<(), E>,
    ) -> Result<(), E> {
        if !self.on {
            return Ok(());
        }
        sync()?;
        let now = std::time::Instant::now();
        self.spans.push((label, (now - self.last).as_secs_f64() * 1e3));
        self.last = now;
        Ok(())
    }

    /// Prints the summed spans by name, largest first, then the total.
    fn report(&self, rows: usize) {
        if !self.on {
            return;
        }
        let mut totals: Vec<(&'static str, f64)> = Vec::new();
        for (label, ms) in &self.spans {
            match totals.iter_mut().find(|(name, _)| name == label) {
                Some(entry) => entry.1 += ms,
                None => totals.push((label, *ms)),
            }
        }
        totals.sort_by(|a, b| b.1.total_cmp(&a.1));
        let total: f64 = totals.iter().map(|(_, ms)| ms).sum();
        for (label, ms) in &totals {
            eprintln!(
                "sensenova stage: {label:<18} {ms:8.2} ms {:5.1}%",
                ms / total * 100.0
            );
        }
        eprintln!(
            "sensenova stage: total {total:.2} ms over {rows} rows ({:.1} tok/s)",
            rows as f64 / (total / 1e3)
        );
    }
}

/// One uploaded matrix, BF16 or E4M3 with its tile scales.
#[derive(Debug, Clone, Copy)]
struct DeviceMat {
    pointer: Deviceptr,
    scales: Option<Deviceptr>,
    rows: usize,
    columns: usize,
    /// The same E4M3 matrix as NVFP4: two E2M1 nibbles a byte and one E4M3 scale a 16 values.
    /// The prefill GEMM reads this form; the decode GEMV reads the bytes above.
    fp4: Option<(Deviceptr, Deviceptr)>,
}

/// One layer's uploaded weights.
#[derive(Debug, Clone, Copy)]
struct DeviceLayer {
    input_norm: Deviceptr,
    post_norm: Deviceptr,
    q: Option<DeviceMat>,
    k: Option<DeviceMat>,
    v: Option<DeviceMat>,
    o: Option<DeviceMat>,
    q_norm: Option<Deviceptr>,
    k_norm: Option<Deviceptr>,
    qkv: Option<DeviceMat>,
    z: Option<DeviceMat>,
    b: Option<DeviceMat>,
    a: Option<DeviceMat>,
    out: Option<DeviceMat>,
    conv: Option<Deviceptr>,
    a_log: Option<Deviceptr>,
    dt_bias: Option<Deviceptr>,
    gdn_norm: Option<Deviceptr>,
    gate: DeviceMat,
    up: DeviceMat,
    down: DeviceMat,
    mhc_attn: Option<(Deviceptr, Deviceptr)>,
    mhc_ffn: Option<(Deviceptr, Deviceptr)>,
}

struct Kernels {
    copy_flat: Kernel,
    gemv_bf16: Kernel,
    gemv_fp8: Kernel,
    rmsnorm: Kernel,
    swiglu: Kernel,
    add: Kernel,
    gdn_conv: Kernel,
    gdn_recur: Kernel,
    rope: Kernel,
    qgate: Kernel,
    cache_bf16: Kernel,
    attn_full: Kernel,
    attn_full_combine: Kernel,
    cache_q8: Kernel,
    attn_q8: Kernel,
    attn_combine: Kernel,
    mhc_pre: Kernel,
    mhc_post: Kernel,
    mhc_leave: Kernel,
    mhc_enter: Kernel,
    /// The tensor-core dense projection, when the prefill image carries it.
    dense_fp8: Option<Kernel>,
    /// The NVFP4 prefill path: the two operand quantisers and the wide and narrow GEMMs.
    quant_fp4: Option<Kernel>,
    quant_weights_fp4: Option<Kernel>,
    dense_fp4_wide: Option<Kernel>,
    dense_fp4_narrow: Option<Kernel>,
}

struct Weights {
    /// The BF16 embedding table on the host: one row is copied to the device a token. The full
    /// table is 2 GB, so it stays in host memory and only the token's row crosses the bus.
    host_embed: Vec<u8>,
    /// The BF16 LM head on the device. Projecting it on the host cost 100-200 ms a token; one
    /// device GEMV reads the same 2 GB at close to the card's bandwidth (2.3 ms).
    head: DeviceMat,
    norm: Deviceptr,
    layers: Vec<DeviceLayer>,
}

pub(crate) struct DeviceShared {
    cuda: CudaHandle,
    kernels: Kernels,
    weights: Weights,
    config: SenseNovaConfig,
    inv_freq: Deviceptr,
    rope_scale: f32,
    tiled_partials: Deviceptr,
    tiled_counters: Deviceptr,
    context: usize,
    kv_fp8: bool,
    allocated: usize,
    pointers: Vec<Deviceptr>,
}

// SAFETY: a Kernel is an opaque driver function word; the context that owns it lives in the
// shared backend and outlives every session.
unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

use sparkflash_ops::gemv::CudaHandle;

impl Drop for DeviceShared {
    fn drop(&mut self) {
        let references: Vec<&Deviceptr> = self.pointers.iter().collect();
        self.cuda.free_buffers(&references);
    }
}

/// The SenseNova-Loop model on the device.
pub struct SenseNovaDeviceBackend {
    descriptor: ModelDescriptor,
    shared: Arc<DeviceShared>,
}

impl SenseNovaDeviceBackend {
    /// Whether this build embeds the images the arm launches.
    #[must_use]
    pub fn available() -> bool {
        !sensenova_cubin().is_empty()
            && !qwen4_exp_cubin().is_empty()
            && !qwen4_attn_cubin().is_empty()
            && !qwen4_scan_cubin().is_empty()
    }

    /// Opens the checkpoint and uploads the text backbone.
    ///
    /// # Errors
    /// A missing image, a CUDA context that cannot open, or a tensor whose shape disagrees.
    pub fn open(
        directory: &Path,
        descriptor: ModelDescriptor,
        context: usize,
    ) -> Result<Self, String> {
        let kv_fp8 = active().attention.kv_cache.unwrap_or(KvCacheMode::Fp8) == KvCacheMode::Fp8;
        Self::open_with_cache(directory, descriptor, context, kv_fp8)
    }

    /// Opens the checkpoint with an explicit K/V cache precision.
    ///
    /// The serving profile picks this (`attention.kv_cache`); the parity tests name it because
    /// the cached E4M3 round is a property they measure, not one they inherit from the profile.
    ///
    /// # Errors
    /// As [`Self::open`].
    pub fn open_with_cache(
        directory: &Path,
        mut descriptor: ModelDescriptor,
        context: usize,
        kv_fp8: bool,
    ) -> Result<Self, String> {
        if !Self::available() {
            return Err(
                "this build has no SenseNova/Qwen cubin: rebuild on a host with nvcc, or set \
                 SPARKFLASH_NVCC"
                    .to_owned(),
            );
        }
        let config = SenseNovaConfig::from_checkpoint_dir(directory)
            .map_err(|error| format!("{}: {error}", directory.join("config.json").display()))?;
        if context == 0 || context > config.max_position_embeddings {
            return Err(format!(
                "context {context} is outside 1..={} for the checkpoint",
                config.max_position_embeddings
            ));
        }
        // The mHC kernel is the released two-stream, fixed-count router; a checkpoint that
        // changes the stream count, fixes the identity, or scales the branch is another kernel.
        if config.mhc.streams != 2 || config.mhc.fixed_identity || config.mhc.pre_scale != 1.0 {
            return Err(format!(
                "the SenseNova device arm implements two streams, a learned mix and pre_scale 1.0,                  not streams={}, fixed_identity={}, pre_scale={}",
                config.mhc.streams, config.mhc.fixed_identity, config.mhc.pre_scale
            ));
        }
        if config.head_dim > 256 {
            return Err(format!(
                "the shared attention body takes head_dim at most 256, not {}",
                config.head_dim
            ));
        }
        let shared = DeviceShared::open(directory, config, context, kv_fp8)?;
        descriptor.context_tokens = context;
        Ok(Self {
            descriptor,
            shared: Arc::new(shared),
        })
    }

    /// The device bytes the weights and scratch claimed.
    #[must_use]
    pub fn weight_bytes(&self) -> usize {
        self.shared.allocated
    }
}

impl DeviceShared {
    fn open(
        directory: &Path,
        config: SenseNovaConfig,
        context: usize,
        kv_fp8: bool,
    ) -> Result<Self, String> {
        let cuda = CudaHandle::new(Cuda::load()?);
        sparkflash_kernels::cuda_driver::enable_forward_async_launches();
        let gdn_shared = (config.linear.key_dim * config.linear.value_dim * 4) as u32;
        let kernels = Kernels::load(&cuda, gdn_shared)?;
        let source = open_checkpoint(directory).map_err(|error| error.to_string())?;
        let plan = SenseNovaTensorPlan::served(&config);
        // The released checkpoint is a superset of the text plan: it also carries the 27-layer
        // vision tower. Check that every served tensor is present; the extras are the tower the
        // first arm does not load (and `source.spec` would catch a missing one at upload anyway).
        for spec in plan.specs() {
            source.spec(&spec.name).map_err(|error| {
                format!("the checkpoint does not match the SenseNova plan: {error}")
            })?;
        }
        let mut pointers = Vec::new();
        let mut allocated = 0usize;
        let mut upload = |bytes: usize, data: &[u8]| -> Result<Deviceptr, String> {
            let pointer = cuda.alloc(bytes, &mut allocated)?;
            cuda.copy_htod(&pointer, data)?;
            pointers.push(pointer);
            Ok(pointer)
        };
        let mat = |source: &dyn WeightSource,
                   name: &str,
                   rows: usize,
                   columns: usize,
                   upload: &mut dyn FnMut(usize, &[u8]) -> Result<Deviceptr, String>|
         -> Result<DeviceMat, String> {
            let spec = source.spec(name).map_err(|error| error.to_string())?;
            if spec.shape != [rows, columns] {
                return Err(format!(
                    "{name} is {:?}, expected [{rows}, {columns}]",
                    spec.shape
                ));
            }
            match spec.dtype {
                DType::BF16 => {
                    let bytes = source.read(name).map_err(|error| error.to_string())?;
                    Ok(DeviceMat {
                        pointer: upload(rows * columns * 2, &bytes)?,
                        scales: None,
                        rows,
                        columns,
                        fp4: None,
                    })
                }
                DType::F8E4M3 => {
                    let bytes = source.read(name).map_err(|error| error.to_string())?;
                    let scale_name = format!("{name}_scale_inv");
                    let scale_spec = source
                        .spec(&scale_name)
                        .map_err(|error| error.to_string())?;
                    let expected = vec![rows.div_ceil(128), columns.div_ceil(128)];
                    if scale_spec.shape != expected {
                        return Err(format!(
                            "{scale_name} is {:?}, expected {expected:?}",
                            scale_spec.shape
                        ));
                    }
                    let scales = source
                        .read(&scale_name)
                        .map_err(|error| error.to_string())?;
                    Ok(DeviceMat {
                        pointer: upload(rows * columns, &bytes)?,
                        scales: Some(upload(
                            rows.div_ceil(128) * columns.div_ceil(128) * 2,
                            &scales,
                        )?),
                        rows,
                        columns,
                        // Quantised once the whole shared state exists (attach_fp4).
                        fp4: None,
                    })
                }
                DType::F32 => {
                    let bytes = source.read(name).map_err(|error| error.to_string())?;
                    Ok(DeviceMat {
                        pointer: upload(rows * columns * 4, &bytes)?,
                        scales: None,
                        rows,
                        columns,
                        fp4: None,
                    })
                }
                other => Err(format!("{name} is {other:?}, expected BF16, F32 or E4M3")),
            }
        };
        let vector = |source: &dyn WeightSource,
                      name: &str,
                      elements: usize,
                      upload: &mut dyn FnMut(usize, &[u8]) -> Result<Deviceptr, String>|
         -> Result<Deviceptr, String> {
            let spec: TensorSpec = source.spec(name).map_err(|error| error.to_string())?;
            if spec.elements() as usize != elements {
                return Err(format!(
                    "{name} is {:?}, expected {elements} values",
                    spec.shape
                ));
            }
            let bytes = source.read(name).map_err(|error| error.to_string())?;
            upload(bytes.len(), &bytes)
        };
        let vector_bf16 = |source: &dyn WeightSource,
                           name: &str,
                           elements: usize,
                           upload: &mut dyn FnMut(usize, &[u8]) -> Result<Deviceptr, String>|
         -> Result<Deviceptr, String> {
            let spec = source.spec(name).map_err(|error| error.to_string())?;
            if spec.elements() as usize != elements {
                return Err(format!(
                    "{name} is {:?}, expected {elements} values",
                    spec.shape
                ));
            }
            let bytes = source.read(name).map_err(|error| error.to_string())?;
            let converted: Vec<u8> = match spec.dtype {
                DType::BF16 => bytes,
                DType::F32 => bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|chunk| {
                        sparkflash_weights::quant::f32_to_bf16_bits(f32::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                        ]))
                        .to_le_bytes()
                    })
                    .collect(),
                other => return Err(format!("{name} is {other:?}, expected BF16 or F32")),
            };
            upload(converted.len(), &converted)
        };
        let host_embed = source
            .read("model.language_model.embed_tokens.weight")
            .map_err(|error| error.to_string())?;
        let host_head = source
            .read("lm_head.weight")
            .map_err(|error| error.to_string())?;
        if host_embed.len() != config.vocab_size * config.hidden_size * 2
            || host_head.len() != config.vocab_size * config.hidden_size * 2
        {
            return Err("the embedding and LM head must be BF16 [vocab, hidden]".to_owned());
        }
        let head = DeviceMat {
            pointer: upload(host_head.len(), &host_head)?,
            scales: None,
            rows: config.vocab_size,
            columns: config.hidden_size,
            fp4: None,
        };
        let norm = vector(
            &source,
            "model.language_model.norm.weight",
            config.hidden_size,
            &mut upload,
        )?;
        let hidden = config.hidden_size;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (index, kind) in config.layer_types.iter().enumerate() {
            let prefix = format!("model.language_model.layers.{index}");
            let mut layer = DeviceLayer {
                input_norm: vector(
                    &source,
                    &format!("{prefix}.input_layernorm.weight"),
                    hidden,
                    &mut upload,
                )?,
                post_norm: vector(
                    &source,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    hidden,
                    &mut upload,
                )?,
                q: None,
                k: None,
                v: None,
                o: None,
                q_norm: None,
                k_norm: None,
                qkv: None,
                z: None,
                b: None,
                a: None,
                out: None,
                conv: None,
                a_log: None,
                dt_bias: None,
                gdn_norm: None,
                gate: mat(
                    &source,
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    config.intermediate_size,
                    hidden,
                    &mut upload,
                )?,
                up: mat(
                    &source,
                    &format!("{prefix}.mlp.up_proj.weight"),
                    config.intermediate_size,
                    hidden,
                    &mut upload,
                )?,
                down: mat(
                    &source,
                    &format!("{prefix}.mlp.down_proj.weight"),
                    hidden,
                    config.intermediate_size,
                    &mut upload,
                )?,
                mhc_attn: None,
                mhc_ffn: None,
            };
            match kind {
                LayerType::Full => {
                    let heads = config.num_attention_heads;
                    layer.q = Some(mat(
                        &source,
                        &format!("{prefix}.self_attn.q_proj.weight"),
                        2 * heads * config.head_dim,
                        hidden,
                        &mut upload,
                    )?);
                    layer.k = Some(mat(
                        &source,
                        &format!("{prefix}.self_attn.k_proj.weight"),
                        config.kv_width(),
                        hidden,
                        &mut upload,
                    )?);
                    layer.v = Some(mat(
                        &source,
                        &format!("{prefix}.self_attn.v_proj.weight"),
                        config.kv_width(),
                        hidden,
                        &mut upload,
                    )?);
                    layer.o = Some(mat(
                        &source,
                        &format!("{prefix}.self_attn.o_proj.weight"),
                        hidden,
                        heads * config.head_dim,
                        &mut upload,
                    )?);
                    layer.q_norm = Some(vector(
                        &source,
                        &format!("{prefix}.self_attn.q_norm.weight"),
                        config.head_dim,
                        &mut upload,
                    )?);
                    layer.k_norm = Some(vector(
                        &source,
                        &format!("{prefix}.self_attn.k_norm.weight"),
                        config.head_dim,
                        &mut upload,
                    )?);
                }
                LayerType::Linear => {
                    let fused = config.linear_projection_width();
                    let value = config.linear_value_width();
                    layer.qkv = Some(mat(
                        &source,
                        &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                        fused,
                        hidden,
                        &mut upload,
                    )?);
                    layer.z = Some(mat(
                        &source,
                        &format!("{prefix}.linear_attn.in_proj_z.weight"),
                        value,
                        hidden,
                        &mut upload,
                    )?);
                    layer.b = Some(mat(
                        &source,
                        &format!("{prefix}.linear_attn.in_proj_b.weight"),
                        config.linear.value_heads,
                        hidden,
                        &mut upload,
                    )?);
                    layer.a = Some(mat(
                        &source,
                        &format!("{prefix}.linear_attn.in_proj_a.weight"),
                        config.linear.value_heads,
                        hidden,
                        &mut upload,
                    )?);
                    layer.out = Some(mat(
                        &source,
                        &format!("{prefix}.linear_attn.out_proj.weight"),
                        hidden,
                        value,
                        &mut upload,
                    )?);
                    layer.conv = Some(vector(
                        &source,
                        &format!("{prefix}.linear_attn.conv1d.weight"),
                        fused * config.linear.conv_kernel,
                        &mut upload,
                    )?);
                    layer.a_log = Some(vector_bf16(
                        &source,
                        &format!("{prefix}.linear_attn.A_log"),
                        config.linear.value_heads,
                        &mut upload,
                    )?);
                    layer.dt_bias = Some(vector_bf16(
                        &source,
                        &format!("{prefix}.linear_attn.dt_bias"),
                        config.linear.value_heads,
                        &mut upload,
                    )?);
                    layer.gdn_norm = Some(vector(
                        &source,
                        &format!("{prefix}.linear_attn.norm.weight"),
                        config.linear.value_dim,
                        &mut upload,
                    )?);
                }
            }
            if index >= config.mhc.start && index < config.mhc.end {
                let rows = 2 * config.mhc.streams
                    + if config.mhc.fixed_identity {
                        0
                    } else {
                        config.mhc.streams * config.mhc.streams
                    };
                let width = config.mhc.streams * hidden;
                let load = |name: &str,
                            upload: &mut dyn FnMut(usize, &[u8]) -> Result<Deviceptr, String>|
                 -> Result<(Deviceptr, Deviceptr), String> {
                    Ok((
                        mat(
                            &source,
                            &format!("{prefix}.{name}.weight"),
                            rows,
                            width,
                            upload,
                        )?
                        .pointer,
                        vector(&source, &format!("{prefix}.{name}.bias"), rows, upload)?,
                    ))
                };
                layer.mhc_attn = Some(load("mhc_attn", &mut upload)?);
                layer.mhc_ffn = Some(load("mhc_ffn", &mut upload)?);
            }
            layers.push(layer);
        }
        let inv = crate::forward::yarn_inv_freq(&config);
        let rope_scale = yarn_attention_scale(&config);
        let inv_bytes: Vec<u8> = inv
            .iter()
            .flat_map(|value| (*value as f32).to_le_bytes())
            .collect();
        let inv_freq = upload(inv_bytes.len(), &inv_bytes)?;
        let partials = cuda.alloc(TILED_GEMV_PARTIALS * 4, &mut allocated)?;
        let counters = cuda.alloc(TILED_GEMV_PARTIALS, &mut allocated)?;
        // The tiled GEMV's per-tile counters are atomically incremented from zero; device
        // allocations are not zeroed, so clear them before the first split launch.
        cuda.memset_d8_async(counters, 0, TILED_GEMV_PARTIALS)?;
        pointers.push(partials);
        pointers.push(counters);
        let mut shared = Self {
            cuda,
            kernels,
            weights: Weights {
                host_embed,
                head,
                norm,
                layers,
            },
            config,
            inv_freq,
            rope_scale,
            tiled_partials: partials,
            tiled_counters: counters,
            context,
            kv_fp8,
            allocated,
            pointers,
        };
        shared.attach_fp4()?;
        Ok(shared)
    }

    /// Quantises every E4M3 projection to the NVFP4 form the prefill GEMM reads.
    ///
    /// It runs once, after the checkpoint is resident: a 16-value block's amax decides its own
    /// E4M3 scale and the nibble the value rounds to, so the device reads the FP8 bytes rather
    /// than a host copy, and the bytes stay for the decode GEMV.
    fn attach_fp4(&mut self) -> Result<(), String> {
        let Self {
            cuda,
            kernels,
            weights,
            pointers,
            allocated,
            ..
        } = self;
        let Some(quantise) = kernels.quant_weights_fp4 else {
            return Ok(());
        };
        let mut quantised = 0usize;
        for layer in &mut weights.layers {
            let mut slots: Vec<&mut DeviceMat> = Vec::new();
            for optional in [
                &mut layer.q,
                &mut layer.k,
                &mut layer.v,
                &mut layer.o,
                &mut layer.qkv,
                &mut layer.z,
                &mut layer.b,
                &mut layer.a,
                &mut layer.out,
            ] {
                if let Some(matrix) = optional {
                    slots.push(matrix);
                }
            }
            slots.push(&mut layer.gate);
            slots.push(&mut layer.up);
            slots.push(&mut layer.down);
            for matrix in slots {
                let (Some(scales), None) = (matrix.scales, matrix.fp4) else {
                    continue;
                };
                let rows = matrix.rows;
                let columns = matrix.columns;
                let codes = cuda.alloc(rows * columns / 2, allocated)?;
                let block_scales = cuda.alloc(rows * (columns / 16), allocated)?;
                pointers.push(codes);
                pointers.push(block_scales);
                cuda.launch_async(
                    quantise,
                    (((rows * columns / 16).div_ceil(256)) as u32, 1, 1),
                    (BLOCK, 1, 1),
                    &[
                        matrix.pointer.0,
                        scales.0,
                        codes.0,
                        block_scales.0,
                        rows as u64,
                        columns as u64,
                    ],
                )?;
                matrix.fp4 = Some((codes, block_scales));
                quantised += rows * columns / 2 + rows * (columns / 16);
            }
        }
        *allocated += 0;
        if quantised == 0 {
            eprintln!("sensenova: no NVFP4 prefill weights (the image carries no quantiser)");
        } else {
            eprintln!(
                "sensenova: quantised {:.2} GiB of projections to NVFP4",
                quantised as f64 / (1u64 << 30) as f64
            );
        }
        cuda.synchronize()?;
        Ok(())
    }

    fn launch(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), BackendError> {
        self.cuda
            .launch_async(kernel, grid, block, args)
            .map_err(|error| BackendError::new(DEVICE, format!("{}: {error}", kernel.name())))
    }

    fn rmsnorm(
        &self,
        input: Deviceptr,
        weight: Deviceptr,
        output: Deviceptr,
        width: usize,
        group: usize,
        batch: usize,
    ) -> Result<(), BackendError> {
        self.launch(
            self.kernels.rmsnorm,
            (((batch * width) / group) as u32, 1, 1),
            (BLOCK, 1, 1),
            &[
                input.0,
                weight.0,
                output.0,
                width as u64,
                group as u64,
                (self.config.rms_norm_eps as f32).to_bits() as u64,
                1,
                0,
            ],
        )
    }

    /// Projects `rows` rows of `input` through `matrix` into `output`.
    ///
    /// Above the decode tile the whole chunk goes through the tensor-core kernel in one launch.
    /// A shape that is not a 128-row multiple (the GDN gate and decay vectors are 32 rows) still
    /// takes the decode GEMV, a tile at a time.
    fn project_matrix(
        &self,
        matrix: &DeviceMat,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
        quantize: (Deviceptr, Deviceptr),
    ) -> Result<(), BackendError> {
        // The prefill tile: a char-complete 128-row chunk through the NVFP4 GEMM. The activation
        // is quantised into the caller's scratch block first, one block scale a 16 values.
        if rows >= 128
            && rows % 128 == 0
            && matrix.rows % 128 == 0
            && matrix.columns % 128 == 0
            && let Some((codes, scales)) = matrix.fp4
            && let (Some(quantise), Some(wide), Some(narrow)) = (
                self.kernels.quant_fp4,
                self.kernels.dense_fp4_wide,
                self.kernels.dense_fp4_narrow,
            )
        {
            let (a_codes, a_scales) = quantize;
            self.launch(
                quantise,
                (((rows * matrix.columns / 16).div_ceil(256)) as u32, 1, 1),
                (BLOCK, 1, 1),
                &[
                    input.0,
                    a_codes.0,
                    a_scales.0,
                    rows as u64,
                    matrix.columns as u64,
                ],
            )?;
            // A 12,288-wide projection fills the card with 256-column tiles; a 4,096-wide one
            // needs the 128-column form to reach 32 blocks.
            let (kernel, columns_tile) = if matrix.rows >= 8192 {
                (wide, 256)
            } else {
                (narrow, 128)
            };
            return self.launch(
                kernel,
                (
                    (matrix.rows / columns_tile) as u32,
                    (rows / 128) as u32,
                    1,
                ),
                (256, 1, 1),
                &[
                    codes.0,
                    scales.0,
                    a_codes.0,
                    a_scales.0,
                    output.0,
                    matrix.rows as u64,
                    matrix.columns as u64,
                    rows as u64,
                ],
            );
        }
        if rows > GEMV_MAX_BATCH && matrix.scales.is_some() && matrix.rows % 128 == 0
            && matrix.columns % 128 == 0
            && let Some(kernel) = self.kernels.dense_fp8
        {
            return self.launch(
                kernel,
                ((matrix.rows / 128) as u32, rows.div_ceil(128) as u32, 1),
                (256, 1, 1),
                &[
                    matrix.pointer.0,
                    matrix.scales.expect("scales").0,
                    input.0,
                    output.0,
                    matrix.rows as u64,
                    matrix.columns as u64,
                    rows as u64,
                ],
            );
        }
        for base in (0..rows).step_by(GEMV_MAX_BATCH) {
            let count = GEMV_MAX_BATCH.min(rows - base);
            self.gemv(
                matrix,
                input.offset(base * matrix.columns * 4),
                output.offset(base * matrix.rows * 4),
                count,
            )?;
        }
        Ok(())
    }

    fn gemv(
        &self,
        matrix: &DeviceMat,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        if let Some(scales) = matrix.scales {
            for base in (0..rows).step_by(GEMV_MAX_BATCH) {
                let count = GEMV_MAX_BATCH.min(rows - base);
                gemv_e4m3_tile(
                    self,
                    self.kernels.gemv_fp8,
                    &GpuMatE4m3Tile {
                        weight: matrix.pointer,
                        scales,
                        rows: matrix.rows,
                        columns: matrix.columns,
                    },
                    input.offset(base * matrix.columns * 4),
                    output.offset(base * matrix.rows * 4),
                    count,
                )?;
            }
            return Ok(());
        }
        let rows_per_block = if rows > 1 { 16 } else { 8 };
        self.launch(
            self.kernels.gemv_bf16,
            (matrix.rows.div_ceil(rows_per_block) as u32, 1, 1),
            (BLOCK, 1, 1),
            &[
                matrix.pointer.0,
                input.0,
                output.0,
                matrix.rows as u64,
                matrix.columns as u64,
                rows as u64,
            ],
        )
    }
}

impl sparkflash_ops::gemv::GemvDevice for DeviceShared {
    fn launch(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), BackendError> {
        DeviceShared::launch(self, kernel, grid, block, args)
    }

    fn gemv(
        &self,
        matrix: &GpuMat,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        let matrix = DeviceMat {
            pointer: matrix.pointer,
            scales: None,
            rows: matrix.rows,
            columns: matrix.columns,
            fp4: None,
        };
        DeviceShared::gemv(self, &matrix, input, output, rows)
    }

    fn tiled_scratch(&self) -> (Deviceptr, Deviceptr) {
        (self.tiled_partials, self.tiled_counters)
    }
}

/// The dynamic shared memory the wide NVFP4 GEMM asks for: four stages of a 128-row and a
/// 256-row operand tile, each a 32-byte row, plus the rows' block scales.
const DENSE_FP4_WIDE_SHARED: u32 = 4 * (128 * 32 + 256 * 32 + 128 * 4 + 256 * 4);

/// The narrow engine's: six stages of two 128-row tiles and their scales.
const DENSE_FP4_NARROW_SHARED: u32 = 6 * (128 * 32 + 128 * 32 + 128 * 4 + 128 * 4);

/// A kernel of the image that may be absent (an older toolchain, or a build for an architecture
/// the block-scaled MMA does not exist on): the arm falls back rather than refusing to load.
fn optional(cuda: &Cuda, name: &'static str, shared: u32) -> Option<Kernel> {
    let loaded = if shared == 0 {
        cuda.load_named(sensenova_cubin(), name).map(|(_, kernel)| kernel)
    } else {
        cuda.load_named_shared(sensenova_cubin(), name, shared)
            .map(|(_, kernel)| kernel)
    };
    match loaded {
        Ok(kernel) => Some(kernel),
        Err(error) => {
            eprintln!("sensenova: no {name}: {error}");
            None
        }
    }
}

impl Kernels {
    fn load(cuda: &Cuda, gdn_shared: u32) -> Result<Self, String> {
        let load = |image: &'static [u8], name: &'static str| {
            cuda.load_named(image, name).map(|(_, kernel)| kernel)
        };
        // The bulk-prefill image holds the tensor-core dense projection and is optional: on a
        // toolchain that cannot build it the arm prefills through the decode GEMV instead.
        let dense = if qwen4_prefill_cubin().is_empty() {
            None
        } else {
            match cuda.load_named_shared(qwen4_prefill_cubin(), "q4p_dense_fp8", DENSE_SHARED) {
                Ok((_, kernel)) => Some(kernel),
                Err(error) => {
                    eprintln!("sensenova: no tensor-core dense projection: {error}");
                    None
                }
            }
        };
        Ok(Self {
            copy_flat: load(qwen4_exp_cubin(), "q4e_copy_flat")?,
            gemv_bf16: load(qwen4_exp_cubin(), "q4e_gemv_bf16")?,
            gemv_fp8: load(qwen4_exp_cubin(), "q4e_gemv_tiled_e4m3_tile")?,
            rmsnorm: load(qwen4_exp_cubin(), "q4e_rmsnorm")?,
            swiglu: load(qwen4_exp_cubin(), "q4e_swiglu")?,
            add: load(qwen4_exp_cubin(), "q4e_add")?,
            gdn_conv: load(qwen4_exp_cubin(), "q4e_gdn_conv")?,
            gdn_recur: cuda
                .load_named_shared(qwen4_exp_cubin(), "q4e_gdn_recur", gdn_shared)
                .map(|(_, kernel)| kernel)?,
            rope: load(sensenova_cubin(), "sensenova_rope")?,
            qgate: load(sensenova_cubin(), "sensenova_qgate")?,
            cache_bf16: load(qwen4_attn_cubin(), "q4e_cache_bf16")?,
            attn_full: load(qwen4_attn_cubin(), "q4e_attn_full")?,
            attn_full_combine: load(qwen4_attn_cubin(), "q4e_attn_full_combine")?,
            cache_q8: load(sensenova_cubin(), "sensenova_cache_q8")?,
            attn_q8: load(sensenova_cubin(), "sensenova_attn_full_q8")?,
            attn_combine: load(sensenova_cubin(), "sensenova_attn_combine")?,
            mhc_pre: load(sensenova_cubin(), "sensenova_mhc_pre")?,
            mhc_post: load(sensenova_cubin(), "sensenova_mhc_post")?,
            mhc_leave: load(sensenova_cubin(), "sensenova_mhc_leave")?,
            mhc_enter: load(sensenova_cubin(), "sensenova_mhc_enter")?,
            dense_fp8: dense,
            quant_fp4: optional(cuda, "sensenova_quant_fp4", 0),
            quant_weights_fp4: optional(cuda, "sensenova_quant_weights_fp4", 0),
            dense_fp4_wide: optional(cuda, "sensenova_dense_fp4_wide", DENSE_FP4_WIDE_SHARED),
            dense_fp4_narrow: optional(cuda, "sensenova_dense_fp4_narrow", DENSE_FP4_NARROW_SHARED),
        })
    }
}

struct Scratch {
    hidden: Deviceptr,
    branch: Deviceptr,
    normed: Deviceptr,
    attention: Deviceptr,
    ffn: Deviceptr,
    streams: Deviceptr,
    qkv: Deviceptr,
    z: Deviceptr,
    b: Deviceptr,
    a: Deviceptr,
    conv_out: Deviceptr,
    gdn_out: Deviceptr,
    q_interleaved: Deviceptr,
    query: Deviceptr,
    gate: Deviceptr,
    key: Deviceptr,
    value: Deviceptr,
    attn_out: Deviceptr,
    partials: Deviceptr,
    coeffs: Deviceptr,
    logits: Deviceptr,
    /// The FP4 activation of the projection in flight: nibbles, then one E4M3 scale a 16 values.
    fp4_codes: Deviceptr,
    fp4_scales: Deviceptr,
    /// The token index the kernels read. A device word, not an argument, so one captured graph
    /// serves every position.
    position: Deviceptr,
    pointers: Vec<Deviceptr>,
}

#[derive(Clone, Copy)]
struct VisitState {
    keys: Option<Deviceptr>,
    values: Option<Deviceptr>,
    k_scales: Option<Deviceptr>,
    v_scales: Option<Deviceptr>,
    conv: Option<Deviceptr>,
    recurrent: Option<Deviceptr>,
}

struct DeviceSession {
    shared: Arc<DeviceShared>,
    scratch: Scratch,
    visits: Vec<VisitState>,
    position: usize,
    /// The captured 56-visit forward, when the arm built one.
    graph: Option<sparkflash_kernels::cuda_driver::GraphId>,
    sampler: Sampler,
    sampling: Sampling,
}

impl Drop for DeviceSession {
    fn drop(&mut self) {
        let _ = self.shared.cuda.make_current();
        // The graph reads these buffers, so it goes first.
        if let Some(graph) = self.graph.take() {
            self.shared.cuda.graph_destroy(graph);
        }
        let mut references: Vec<&Deviceptr> = self.scratch.pointers.iter().collect();
        for visit in &self.visits {
            references.extend(visit.keys.as_ref());
            references.extend(visit.values.as_ref());
            references.extend(visit.k_scales.as_ref());
            references.extend(visit.v_scales.as_ref());
            references.extend(visit.conv.as_ref());
            references.extend(visit.recurrent.as_ref());
        }
        self.shared.cuda.free_buffers(&references);
    }
}

impl DeviceSession {
    fn open(shared: Arc<DeviceShared>) -> Result<Self, String> {
        // The generic host driver opens sessions on a serving worker, not on the thread that
        // loaded the context. A CUDA context is per-thread, so bind it before any allocation.
        shared.cuda.make_current()?;
        let config = &shared.config;
        let hidden = config.hidden_size;
        let query = config.num_attention_heads * config.head_dim;
        let kv = config.kv_width();
        let fused = config.linear_projection_width();
        let value = config.linear_value_width();
        let mut pointers = Vec::new();
        let mut allocated = shared.allocated;
        let mut alloc = |bytes: usize| -> Result<Deviceptr, String> {
            let pointer = shared.cuda.alloc(bytes, &mut allocated)?;
            pointers.push(pointer);
            Ok(pointer)
        };
        // Every activation buffer holds ROWS tokens: a verify or a prefill chunk runs the same
        // visit list over that many rows, which is what makes one weight pass serve them all.
        let mut wide = |count: usize| -> Result<Deviceptr, String> { alloc(PREFILL_ROWS * count * 4) };
        let fp4_inner = config.intermediate_size.max(config.hidden_size);
        let scratch = Scratch {
            hidden: wide(hidden)?,
            branch: wide(hidden.max(config.intermediate_size))?,
            normed: wide(hidden)?,
            attention: wide(hidden)?,
            ffn: wide(hidden.max(config.intermediate_size))?,
            streams: wide(2 * hidden)?,
            qkv: wide(fused)?,
            z: wide(value)?,
            b: wide(config.linear.value_heads)?,
            a: wide(config.linear.value_heads)?,
            conv_out: wide(fused)?,
            gdn_out: wide(value)?,
            q_interleaved: wide(2 * query)?,
            query: wide(query)?,
            gate: wide(query)?,
            key: wide(kv)?,
            value: wide(kv)?,
            attn_out: wide(query)?,
            partials: wide(config.num_attention_heads * 16 * (config.head_dim + 2))?,
            coeffs: wide(8)?,
            // A verify reads every row's logits; a prefill chunk only the last row's.
            logits: alloc(ROWS * config.vocab_size * 4)?,
            // The widest activation a prefill block quantises is the MLP's intermediate.
            fp4_codes: alloc(PREFILL_ROWS * fp4_inner / 2)?,
            fp4_scales: alloc(PREFILL_ROWS * (fp4_inner / 16))?,
            position: alloc(4)?,
            pointers,
        };
        let mut visits = Vec::with_capacity(config.visits());
        for physical in config.execution_plan() {
            let state = match config.layer_types[physical] {
                LayerType::Full => {
                    let row_bytes = if shared.kv_fp8 {
                        shared.context * kv
                    } else {
                        shared.context * kv * 2
                    };
                    let k_scale = if shared.kv_fp8 {
                        Some(shared.cuda.alloc(
                            shared.context * config.num_key_value_heads * 4,
                            &mut allocated,
                        )?)
                    } else {
                        None
                    };
                    let v_scale = if shared.kv_fp8 {
                        Some(shared.cuda.alloc(
                            shared.context * config.num_key_value_heads * 4,
                            &mut allocated,
                        )?)
                    } else {
                        None
                    };
                    VisitState {
                        keys: Some(shared.cuda.alloc(row_bytes, &mut allocated)?),
                        values: Some(shared.cuda.alloc(row_bytes, &mut allocated)?),
                        k_scales: k_scale,
                        v_scales: v_scale,
                        conv: None,
                        recurrent: None,
                    }
                }
                LayerType::Linear => VisitState {
                    keys: None,
                    values: None,
                    k_scales: None,
                    v_scales: None,
                    conv: Some(shared.cuda.alloc(
                        config.linear.conv_kernel.saturating_sub(1) * fused * 4,
                        &mut allocated,
                    )?),
                    recurrent: Some(shared.cuda.alloc(
                        config.linear.value_heads
                            * config.linear.key_dim
                            * config.linear.value_dim
                            * 4,
                        &mut allocated,
                    )?),
                },
            };
            visits.push(state);
        }
        // The convolution and recurrent states start at zero.
        for visit in &visits {
            if let Some(pointer) = visit.conv {
                shared.cuda.memset_d8_async(
                    pointer,
                    0,
                    config.linear.conv_kernel.saturating_sub(1) * fused * 4,
                )?;
            }
            if let Some(pointer) = visit.recurrent {
                shared.cuda.memset_d8_async(
                    pointer,
                    0,
                    config.linear.value_heads * config.linear.key_dim * config.linear.value_dim * 4,
                )?;
            }
        }
        let mut session = Self {
            shared,
            scratch,
            visits,
            position: 0,
            graph: None,
            sampler: Sampler::new(),
            sampling: Sampling::default(),
        };
        session.graph = session.capture_graph();
        Ok(session)
    }

    /// Captures the 56 visits and the LM head as one CUDA graph.
    ///
    /// The decode measured 22.5 ms a token for 14.1 GB of weights, and one 2 GB GEMV alone ran
    /// at ~870 GB/s against the visit list's ~600: the difference is the ~600 dependent launches
    /// a token and their tails, not the memory system. A replayed graph issues the same work with
    /// one host call, so the gap is the launch path itself.
    ///
    /// The launches read the position from a device word, so the argument list - and therefore
    /// the graph - is the same at every index. That is the one structural requirement: the split
    /// grid is constant and the cache, rope and attention take a pointer.
    ///
    /// Capture happens once, when the session opens. A graph bakes in the buffers it was
    /// captured with, so it is per-session; the serving pool opens a sequence only after the one
    /// it replaces is dropped, and [CAPTURE] keeps two opens from overlapping anyway.
    fn capture_graph(&mut self) -> Option<sparkflash_kernels::cuda_driver::GraphId> {
        if std::env::var("SPARKFLASH_SENSENOVA_GRAPH").as_deref() == Ok("0") {
            return None;
        }
        // The 8-bit path is the one whose launches are position-free; the BF16 fallback caches by
        // a host-computed pointer offset, which a replay would freeze at capture time.
        if !self.shared.kv_fp8 {
            return None;
        }
        if std::env::var("SPARKFLASH_SENSENOVA_DEBUG").is_ok() {
            return None;
        }
        // The per-visit timer synchronises between visits, which a capture cannot cross.
        if std::env::var("SPARKFLASH_SENSENOVA_TIME").as_deref() == Ok("2") {
            return None;
        }
        let shared = Arc::clone(&self.shared);
        let _guard = CAPTURE.lock().unwrap_or_else(|error| error.into_inner());
        if let Err(error) = shared.cuda.graph_capture_begin() {
            eprintln!("sensenova graph: capture refused: {error}");
            return None;
        }
        if let Err(error) = self.project(1, false, false) {
            shared.cuda.graph_capture_abort();
            eprintln!("sensenova graph: capture failed: {error}");
            return None;
        }
        match shared.cuda.graph_capture_end() {
            Ok((graph, records)) => {
                eprintln!(
                    "sensenova graph: captured {} launches, replaying one call a token",
                    records.len()
                );
                Some(graph)
            }
            Err(error) => {
                eprintln!("sensenova graph: capture ended badly: {error}");
                None
            }
        }
    }

    /// One forward over `tokens`, from the current position, returning the last row's logits.
    ///
    /// A verify and a prefill chunk are the same call. Every projection reads its weights once
    /// for all the rows, which is what makes a batch of eight cost little more than one token on
    /// a body that is bound by the weight pass.
    fn forward_rows(&mut self, tokens: &[u32]) -> Result<Vec<f32>, BackendError> {
        let shared = Arc::clone(&self.shared);
        shared
            .cuda
            .make_current()
            .map_err(|error| BackendError::new(DEVICE, error))?;
        let config = &shared.config;
        let hidden = config.hidden_size;
        let rows = tokens.len();
        if rows == 0 || rows > PREFILL_ROWS {
            return Err(BackendError::new(
                DEVICE,
                format!("a forward takes 1..={PREFILL_ROWS} rows, not {rows}"),
            ));
        }
        if self.position + rows > shared.context {
            return Err(BackendError::new(
                DEVICE,
                format!(
                    "{rows} rows from {} run past the served context {}",
                    self.position, shared.context
                ),
            ));
        }
        for token in tokens {
            if *token as usize >= config.vocab_size {
                return Err(BackendError::new(
                    DEVICE,
                    format!(
                        "token {token} is outside the vocabulary {}",
                        config.vocab_size
                    ),
                ));
            }
        }
        // The BF16 cache path addresses a row by a host-computed pointer offset, which the chunk
        // loop would have to know; the 8-bit path's kernels take the row index themselves.
        if !shared.kv_fp8 && rows > 1 {
            let mut logits = Vec::new();
            for token in tokens {
                logits = self.forward_rows(&[*token])?;
            }
            return Ok(logits);
        }
        let clock = std::env::var("SPARKFLASH_SENSENOVA_TIME");
        let timed = clock.is_ok();
        let per_visit = clock.as_deref() == Ok("2");
        let started = std::time::Instant::now();
        let position = self.position;
        let mut widened = Vec::with_capacity(rows * hidden * 4);
        for token in tokens {
            let row = *token as usize * hidden * 2;
            let embed = &shared.weights.host_embed[row..row + hidden * 2];
            for pair in embed.as_chunks::<2>().0 {
                let value =
                    sparkflash_weights::quant::bf16_to_f32(u16::from_le_bytes([pair[0], pair[1]]));
                widened.extend_from_slice(&value.to_le_bytes());
            }
        }
        shared
            .cuda
            .copy_htod(&self.scratch.hidden, &widened)
            .map_err(|error| BackendError::new(DEVICE, error))?;
        let after_embed = started.elapsed();
        let debug = std::env::var("SPARKFLASH_SENSENOVA_DEBUG").is_ok();
        if debug {
            self.report("embed", self.scratch.hidden, hidden)?;
        }
        // The launches read the position from the device, so a replayed graph is valid at any
        // index; the word is rewritten here, ahead of either path, on the same stream.
        shared
            .cuda
            .copy_htod(&self.scratch.position, &u32::try_from(position).unwrap_or(0).to_le_bytes())
            .map_err(|error| BackendError::new(DEVICE, error))?;
        // A graph bakes in its grids, so only the one-row forward replays one.
        match self.graph.filter(|_| rows == 1) {
            Some(graph) => shared
                .cuda
                .graph_launch(graph)
                .map_err(|error| BackendError::new(DEVICE, error))?,
            None => self.project(rows, debug, per_visit)?,
        }
        if timed {
            shared
                .cuda
                .synchronize()
                .map_err(|error| BackendError::new(DEVICE, error))?;
        }
        let after_device = started.elapsed();
        // A one-row step and a chunk above the tile read one row; a verify reads them all.
        let head_rows = if rows <= ROWS { rows } else { 1 };
        let mut logit_bytes = vec![0u8; head_rows * config.vocab_size * 4];
        shared
            .cuda
            .copy_dtoh(&mut logit_bytes, &self.scratch.logits)
            .map_err(|error| BackendError::new(DEVICE, error))?;
        let logits: Vec<f32> = logit_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        if timed {
            let total = started.elapsed();
            eprintln!(
                "sensenova time: embed {:.2} ms, device {:.2} ms, head {:.2} ms, total {:.2} ms",
                after_embed.as_secs_f64() * 1e3,
                (after_device - after_embed).as_secs_f64() * 1e3,
                (total - after_device).as_secs_f64() * 1e3,
                total.as_secs_f64() * 1e3
            );
        }
        self.position += rows;
        Ok(logits)
    }

    /// The token's 56 visits, the final norm and the LM head: the launches a graph captures.
    fn project(
        &mut self,
        rows: usize,
        debug: bool,
        per_visit: bool,
    ) -> Result<(), BackendError> {
        let shared = Arc::clone(&self.shared);
        let config = &shared.config;
        let hidden = config.hidden_size;
        let position = self.position;
        let mut visit_clock = std::time::Instant::now();
        let mut clock = StageClock::new(per_visit);
        let drain = || shared.cuda.synchronize().map_err(|error| BackendError::new(DEVICE, error));
        for (visit, physical) in config.execution_plan().into_iter().enumerate() {
            let layer = &shared.weights.layers[physical];
            let state = self.visits[visit];
            let uses_mhc = layer.mhc_attn.is_some();
            if uses_mhc {
                if physical == config.mhc.start {
                    shared.launch(
                        shared.kernels.mhc_enter,
                        (rows as u32, 1, 1),
                        (BLOCK, 1, 1),
                        &[
                            self.scratch.hidden.0,
                            self.scratch.streams.0,
                            hidden as u64,
                            rows as u64,
                        ],
                    )?;
                }
                let (weight, bias) = layer.mhc_attn.expect("router");
                shared.launch(
                    shared.kernels.mhc_pre,
                    (rows as u32, 1, 1),
                    (BLOCK, 1, 1),
                    &[
                        self.scratch.streams.0,
                        weight.0,
                        bias.0,
                        self.scratch.branch.0,
                        self.scratch.coeffs.0,
                        hidden as u64,
                        config.mhc.streams as u64,
                        config.mhc.iterations as u64,
                        (config.mhc.epsilon as f32).to_bits() as u64,
                    ],
                )?;
                shared.rmsnorm(
                    self.scratch.branch,
                    layer.input_norm,
                    self.scratch.normed,
                    hidden,
                    hidden,
                    rows,
                )?;
            } else {
                shared.rmsnorm(
                    self.scratch.hidden,
                    layer.input_norm,
                    self.scratch.normed,
                    hidden,
                    hidden,
                    rows,
                )?;
            }
            clock.mark("attn norm+mhc", drain)?;
            let attention = match config.layer_types[physical] {
                LayerType::Full => self.full_attention(layer, state, position, rows, &mut clock),
                LayerType::Linear => self.linear_attention(layer, state, rows, &mut clock),
            };
            attention?;
            clock.mark("attention", drain)?;
            if debug {
                self.report(
                    &format!("visit {visit} layer {physical} attention"),
                    self.scratch.attention,
                    hidden,
                )?;
            }
            if uses_mhc {
                shared.launch(
                    shared.kernels.mhc_post,
                    (rows as u32, hidden.div_ceil(256) as u32, 1),
                    (BLOCK, 1, 1),
                    &[
                        self.scratch.streams.0,
                        self.scratch.attention.0,
                        self.scratch.coeffs.0,
                        hidden as u64,
                    ],
                )?;
                clock.mark("attn merge", drain)?;
                let (weight, bias) = layer.mhc_ffn.expect("router");
                shared.launch(
                    shared.kernels.mhc_pre,
                    (rows as u32, 1, 1),
                    (BLOCK, 1, 1),
                    &[
                        self.scratch.streams.0,
                        weight.0,
                        bias.0,
                        self.scratch.branch.0,
                        self.scratch.coeffs.0,
                        hidden as u64,
                        config.mhc.streams as u64,
                        config.mhc.iterations as u64,
                        (config.mhc.epsilon as f32).to_bits() as u64,
                    ],
                )?;
                shared.rmsnorm(
                    self.scratch.branch,
                    layer.post_norm,
                    self.scratch.normed,
                    hidden,
                    hidden,
                    rows,
                )?;
            } else {
                shared.launch(
                    shared.kernels.add,
                    ((rows * hidden).div_ceil(256) as u32, 1, 1),
                    (BLOCK, 1, 1),
                    &[
                        self.scratch.hidden.0,
                        self.scratch.attention.0,
                        (rows * hidden) as u64,
                    ],
                )?;
                shared.rmsnorm(
                    self.scratch.hidden,
                    layer.post_norm,
                    self.scratch.normed,
                    hidden,
                    hidden,
                    rows,
                )?;
            }
            clock.mark("ffn norm+mhc", drain)?;
            self.mlp(layer, rows, &mut clock)?;
            clock.mark("mlp", drain)?;
            if debug {
                self.report(
                    &format!("device visit {visit} layer {physical} ffn"),
                    self.scratch.ffn,
                    hidden,
                )?;
            }
            if uses_mhc {
                shared.launch(
                    shared.kernels.mhc_post,
                    (rows as u32, hidden.div_ceil(256) as u32, 1),
                    (BLOCK, 1, 1),
                    &[
                        self.scratch.streams.0,
                        self.scratch.ffn.0,
                        self.scratch.coeffs.0,
                        hidden as u64,
                    ],
                )?;
                if physical == config.mhc.end - 1 {
                    shared.launch(
                        shared.kernels.mhc_leave,
                        (rows as u32, 1, 1),
                        (BLOCK, 1, 1),
                        &[
                            self.scratch.streams.0,
                            self.scratch.hidden.0,
                            hidden as u64,
                            rows as u64,
                        ],
                    )?;
                }
            } else {
                shared.launch(
                    shared.kernels.add,
                    ((rows * hidden).div_ceil(256) as u32, 1, 1),
                    (BLOCK, 1, 1),
                    &[
                        self.scratch.hidden.0,
                        self.scratch.ffn.0,
                        (rows * hidden) as u64,
                    ],
                )?;
            }
            if debug {
                self.report(
                    &format!("device hidden after visit {visit}"),
                    self.scratch.hidden,
                    hidden,
                )?;
            }
            if per_visit {
                shared
                    .cuda
                    .synchronize()
                    .map_err(|error| BackendError::new(DEVICE, error))?;
                let now = std::time::Instant::now();
                eprintln!(
                    "sensenova visit {visit:2} layer {physical:2} {} {:.2} ms",
                    match config.layer_types[physical] {
                        LayerType::Full => "full",
                        LayerType::Linear => "linear",
                    },
                    (now - visit_clock).as_secs_f64() * 1e3
                );
                visit_clock = now;
            }
        }
        shared.rmsnorm(
            self.scratch.hidden,
            shared.weights.norm,
            self.scratch.normed,
            hidden,
            hidden,
            rows,
        )?;
        if debug {
            self.report("device final hidden", self.scratch.normed, hidden)?;
        }
        clock.mark("ffn merge+leave", drain)?;
        // A verify needs every row's pick, so its head runs over all of them; a prefill chunk
        // needs the row that predicts the next token, and a one-row step is one row.
        let head_rows = if rows <= ROWS { rows } else { 1 };
        shared.gemv(
            &shared.weights.head,
            self.scratch
                .normed
                .offset((rows - head_rows) * hidden * 4),
            self.scratch.logits,
            head_rows,
        )?;
        clock.mark("final norm+head", drain)?;
        clock.report(rows);
        Ok(())
    }

    fn report(&self, label: &str, pointer: Deviceptr, width: usize) -> Result<(), BackendError> {
        let mut bytes = vec![0u8; width * 4];
        self.shared
            .cuda
            .copy_dtoh(&mut bytes, &pointer)
            .map_err(|error| BackendError::new(DEVICE, error))?;
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let finite = values.iter().filter(|value| value.is_finite()).count();
        let max = values
            .iter()
            .filter(|value| value.is_finite())
            .fold(0.0f32, |a, b| a.max(b.abs()));
        eprintln!("{label}: {finite}/{} finite, max {max}", values.len());
        Ok(())
    }

    fn mlp(
        &mut self,
        layer: &DeviceLayer,
        rows: usize,
        clock: &mut StageClock,
    ) -> Result<(), BackendError> {
        let shared = Arc::clone(&self.shared);
        let quant = (self.scratch.fp4_codes, self.scratch.fp4_scales);
        let intermediate = shared.config.intermediate_size;
        let debug = std::env::var("SPARKFLASH_SENSENOVA_DEBUG").is_ok();
        let drain = || shared.cuda.synchronize().map_err(|error| BackendError::new(DEVICE, error));
        shared.project_matrix(&layer.gate, self.scratch.normed, self.scratch.branch, rows, quant)?;
        clock.mark("  mlp gate", drain)?;
        if debug {
            self.report("device mlp gate", self.scratch.branch, rows * intermediate)?;
        }
        shared.project_matrix(&layer.up, self.scratch.normed, self.scratch.ffn, rows, quant)?;
        clock.mark("  mlp up", drain)?;
        if debug {
            self.report("device mlp up", self.scratch.ffn, rows * intermediate)?;
        }
        shared.launch(
            shared.kernels.swiglu,
            ((rows * intermediate).div_ceil(256) as u32, 1, 1),
            (BLOCK, 1, 1),
            &[
                self.scratch.branch.0,
                self.scratch.ffn.0,
                self.scratch.branch.0,
                (rows * intermediate) as u64,
            ],
        )?;
        clock.mark("  mlp swiglu", drain)?;
        if debug {
            self.report("device mlp hidden", self.scratch.branch, rows * intermediate)?;
        }
        shared.project_matrix(&layer.down, self.scratch.branch, self.scratch.ffn, rows, quant)?;
        clock.mark("  mlp down", drain)?;
        if debug {
            self.report("device mlp down", self.scratch.ffn, rows * intermediate)?;
        }
        Ok(())
    }

    fn full_attention(
        &mut self,
        layer: &DeviceLayer,
        state: VisitState,
        position: usize,
        rows: usize,
        clock: &mut StageClock,
    ) -> Result<(), BackendError> {
        let shared = Arc::clone(&self.shared);
        let config = &shared.config;
        let drain = || shared.cuda.synchronize().map_err(|error| BackendError::new(DEVICE, error));
        let quant = (self.scratch.fp4_codes, self.scratch.fp4_scales);
        let heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let query = heads * head_dim;
        let kv = kv_heads * head_dim;
        let (q, k, v, o) = (
            layer.q.expect("q"),
            layer.k.expect("k"),
            layer.v.expect("v"),
            layer.o.expect("o"),
        );
        shared.project_matrix(&q, self.scratch.normed, self.scratch.q_interleaved, rows, quant)?;
        clock.mark("  full q", drain)?;
        shared.launch(
            shared.kernels.qgate,
            (heads as u32, rows as u32, 1),
            (256, 1, 1),
            &[
                self.scratch.q_interleaved.0,
                self.scratch.query.0,
                self.scratch.gate.0,
                head_dim as u64,
                heads as u64,
                rows as u64,
            ],
        )?;
        shared.project_matrix(&k, self.scratch.normed, self.scratch.key, rows, quant)?;
        shared.project_matrix(&v, self.scratch.normed, self.scratch.value, rows, quant)?;
        clock.mark("  full kv", drain)?;
        shared.rmsnorm(
            self.scratch.query,
            layer.q_norm.expect("q norm"),
            self.scratch.query,
            head_dim,
            head_dim,
            rows * heads,
        )?;
        shared.rmsnorm(
            self.scratch.key,
            layer.k_norm.expect("k norm"),
            self.scratch.key,
            head_dim,
            head_dim,
            rows * kv_heads,
        )?;
        shared.launch(
            shared.kernels.rope,
            (heads as u32, rows as u32, 1),
            (128, 1, 1),
            &[
                self.scratch.query.0,
                head_dim as u64,
                config.rope.dims as u64,
                self.scratch.position.0,
                shared.inv_freq.0,
                u64::from(shared.rope_scale.to_bits()),
                query as u64,
                rows as u64,
            ],
        )?;
        shared.launch(
            shared.kernels.rope,
            (kv_heads as u32, rows as u32, 1),
            (128, 1, 1),
            &[
                self.scratch.key.0,
                head_dim as u64,
                config.rope.dims as u64,
                self.scratch.position.0,
                shared.inv_freq.0,
                u64::from(shared.rope_scale.to_bits()),
                kv as u64,
                rows as u64,
            ],
        )?;
        let keys = state.keys.expect("keys");
        let values = state.values.expect("values");
        let splits = ((position + 256) / 256).min(16) as u32;
        if shared.kv_fp8 {
            let k_scales = state.k_scales.expect("k scales");
            let v_scales = state.v_scales.expect("v scales");
            for (source, codes, scales) in [
                (self.scratch.key, keys, k_scales),
                (self.scratch.value, values, v_scales),
            ] {
                shared.launch(
                    shared.kernels.cache_q8,
                    (kv_heads as u32, rows as u32, 1),
                    (BLOCK, 1, 1),
                    &[
                        source.0,
                        codes.0,
                        scales.0,
                        self.scratch.position.0,
                        rows as u64,
                        kv_heads as u64,
                        head_dim as u64,
                    ],
                )?;
            }
            shared.launch(
                shared.kernels.attn_q8,
                (heads as u32, rows as u32, FULL_SPLITS),
                (BLOCK, 1, 1),
                &[
                    self.scratch.query.0,
                    keys.0,
                    values.0,
                    k_scales.0,
                    v_scales.0,
                    self.scratch.partials.0,
                    self.scratch.position.0,
                    heads as u64,
                    kv_heads as u64,
                    head_dim as u64,
                    (heads / kv_heads.max(1)) as u64,
                ],
            )?;
        } else {
            let cache_blocks = kv.div_ceil(256) as u32;
            shared.launch(
                shared.kernels.cache_bf16,
                (cache_blocks, 1, 1),
                (BLOCK, 1, 1),
                &[
                    keys.offset(position * kv * 2).0,
                    self.scratch.key.0,
                    kv as u64,
                ],
            )?;
            shared.launch(
                shared.kernels.cache_bf16,
                (cache_blocks, 1, 1),
                (BLOCK, 1, 1),
                &[
                    values.offset(position * kv * 2).0,
                    self.scratch.value.0,
                    kv as u64,
                ],
            )?;
            shared.launch(
                shared.kernels.attn_full,
                (heads as u32, 1, splits),
                (BLOCK, 1, 1),
                &[
                    self.scratch.query.0,
                    keys.0,
                    values.0,
                    self.scratch.partials.0,
                    position as u64,
                    heads as u64,
                    kv_heads as u64,
                    head_dim as u64,
                    (heads / kv_heads.max(1)) as u64,
                ],
            )?;
        }
        clock.mark("  full attention core", drain)?;
        if shared.kv_fp8 {
            shared.launch(
                shared.kernels.attn_combine,
                (heads as u32, rows as u32, 1),
                (BLOCK, 1, 1),
                &[
                    self.scratch.partials.0,
                    self.scratch.gate.0,
                    self.scratch.attn_out.0,
                    self.scratch.position.0,
                    heads as u64,
                    head_dim as u64,
                ],
            )?;
        } else {
            shared.launch(
                shared.kernels.attn_full_combine,
                (heads as u32, 1, 1),
                (BLOCK, 1, 1),
                &[
                    self.scratch.partials.0,
                    self.scratch.gate.0,
                    self.scratch.attn_out.0,
                    position as u64,
                    heads as u64,
                    head_dim as u64,
                ],
            )?;
        }
        clock.mark("  full combine", drain)?;
        shared.project_matrix(&o, self.scratch.attn_out, self.scratch.attention, rows, quant)?;
        clock.mark("  full o", drain)?;
        Ok(())
    }

    fn linear_attention(
        &mut self,
        layer: &DeviceLayer,
        state: VisitState,
        rows: usize,
        clock: &mut StageClock,
    ) -> Result<(), BackendError> {
        let shared = Arc::clone(&self.shared);
        let config = &shared.config;
        let drain = || shared.cuda.synchronize().map_err(|error| BackendError::new(DEVICE, error));
        let quant = (self.scratch.fp4_codes, self.scratch.fp4_scales);
        let key_width = config.linear.key_heads * config.linear.key_dim;
        let value_width = config.linear_value_width();
        let fused = 2 * key_width + value_width;
        let (qkv, z, b, a, out) = (
            layer.qkv.expect("qkv"),
            layer.z.expect("z"),
            layer.b.expect("b"),
            layer.a.expect("a"),
            layer.out.expect("out"),
        );
        let debug = std::env::var("SPARKFLASH_SENSENOVA_DEBUG").is_ok();
        shared.project_matrix(&qkv, self.scratch.normed, self.scratch.qkv, rows, quant)?;
        clock.mark("  gdn qkv", drain)?;
        if debug {
            self.report("gdn qkv", self.scratch.qkv, fused)?;
        }
        shared.project_matrix(&z, self.scratch.normed, self.scratch.z, rows, quant)?;
        clock.mark("  gdn z", drain)?;
        if debug {
            self.report("gdn z", self.scratch.z, value_width)?;
        }
        shared.gemv(&b, self.scratch.normed, self.scratch.b, rows)?;
        shared.gemv(&a, self.scratch.normed, self.scratch.a, rows)?;
        clock.mark("  gdn ba", drain)?;
        if debug {
            self.report("gdn b", self.scratch.b, config.linear.value_heads)?;
        }
        let conv = state.conv.expect("conv state");
        let recurrent = state.recurrent.expect("recurrent state");
        shared.launch(
            shared.kernels.gdn_conv,
            (fused.div_ceil(256) as u32, 1, 1),
            (BLOCK, 1, 1),
            &[
                self.scratch.qkv.0,
                conv.0,
                layer.conv.expect("conv weight").0,
                self.scratch.conv_out.0,
                fused as u64,
                config.linear.conv_kernel as u64,
                rows as u64,
                0,
                0,
                0,
            ],
        )?;
        clock.mark("  gdn conv", drain)?;
        if debug {
            self.report("gdn conv", self.scratch.conv_out, fused)?;
        }
        shared.launch(
            shared.kernels.gdn_recur,
            (config.linear.value_heads as u32, 1, 1),
            (128, 1, 1),
            &[
                self.scratch.conv_out.0,
                self.scratch.z.0,
                self.scratch.b.0,
                self.scratch.a.0,
                layer.a_log.expect("a log").0,
                layer.dt_bias.expect("dt bias").0,
                layer.gdn_norm.expect("gdn norm").0,
                recurrent.0,
                recurrent.0,
                self.scratch.gdn_out.0,
                key_width as u64,
                config.linear.value_heads as u64,
                config.linear.key_dim as u64,
                config.linear.value_dim as u64,
                (config.linear.value_heads / config.linear.key_heads.max(1)) as u64,
                (config.rms_norm_eps as f32).to_bits() as u64,
                0,
                rows as u64,
                (config.linear.key_dim * config.linear.value_dim * 4) as u64,
                0,
            ],
        )?;
        clock.mark("  gdn recur", drain)?;
        if debug {
            self.report("gdn recur", self.scratch.gdn_out, value_width)?;
        }
        shared.project_matrix(&out, self.scratch.gdn_out, self.scratch.attention, rows, quant)?;
        clock.mark("  gdn out", drain)?;
        if debug {
            self.report("gdn out", self.scratch.attention, config.hidden_size)?;
        }
        Ok(())
    }
}

impl ModelBackend for SenseNovaDeviceBackend {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        let session = DeviceSession::open(Arc::clone(&self.shared))
            .map_err(|error| BackendError::new(DEVICE, error))?;
        Ok(Box::new(session))
    }
}

impl ModelSession for DeviceSession {
    fn set_sampling(&mut self, sampling: Sampling) {
        self.sampling = sampling;
        self.sampler.reset();
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::new(DEVICE, "an empty prompt"));
        }
        // A prompt runs in chunks of ROWS: every projection then reads its weights once for the
        // whole chunk, and the causal attention the kernels implement makes the rows of one
        // chunk exactly the rows of the one-token steps they stand in for.
        let mut logits = Vec::new();
        for chunk in tokens.chunks(PREFILL_ROWS) {
            let all = self.forward_rows(chunk)?;
            // A chunk under the verify tile returns every row; only the last one predicts.
            logits = all[all.len() - self.shared.config.vocab_size..].to_vec();
        }
        self.sampler.reset();
        self.sampler.record_prompt(tokens);
        let chosen = self.sampler.choose(&logits, &self.sampling);
        Ok(StepOutput {
            next_token: Some(chosen as u32),
            logits,
            draft: Vec::new(),
        })
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        if self.position >= self.shared.context {
            return Err(BackendError::new(
                DEVICE,
                format!(
                    "the sequence is past the served context {}",
                    self.shared.context
                ),
            ));
        }
        let logits = self.forward_rows(&[token])?;
        self.sampler.record(token);
        let chosen = self.sampler.choose(&logits, &self.sampling);
        Ok(StepOutput {
            next_token: Some(chosen as u32),
            logits,
            draft: Vec::new(),
        })
    }

    fn reusable_prefix(&self) -> Option<&[u32]> {
        None
    }
}

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

//! The dynamic host backend: a model-independent reference forward over the parts the dynamic
//! path supports.
//!
//! Phase 4's `EngineBuilder` is merged, but only the released Qwen and GLM arms implement it
//! (`PartsDeviceBackend`), and a checkpoint that is not one of those presets has no device arm.
//! This module is the honest host path the issue allows when a full device serve is unavailable:
//! it runs [Architecture]'s layer walk over a [WeightSource] with f32 arithmetic, on the CPU. It
//! is a correctness and bring-up path, not a performance claim; every weight is read per step, so
//! it is meant for the tiny synthetic checkpoints the zero-code test writes.
//!
//! The executed parts are [crate::dynamic::DYNAMIC_PARTS]: embedding, RMSNorm, grouped-query
//! attention, Gated DeltaNet, dense SwiGLU, routed/shared MoE and the LM head, with a plain
//! residual. A checkpoint naming anything else is refused by [crate::dynamic::unsupported_parts]
//! before this backend opens.

use std::sync::Arc;

use sparkflash_model::architecture::{Activation, AttentionKind, Ffn, Score};
use sparkflash_model::{
    Architecture, BackendError, ModelBackend, ModelDescriptor, ModelSession, Sampling, StepOutput,
};
use sparkflash_weights::DType;
use sparkflash_weights::linalg::{ScaleGrid, apply_scales, decode_by_name};
use sparkflash_weights::linear_attention::{
    Decay, advance_conv_history, causal_conv, delta_rule_step, l2_normalise, rms_norm_gated,
};
use sparkflash_weights::weights::WeightSource;

use crate::dynamic::{ResolvedAttention, ResolvedFfn, ResolvedNames};

/// The largest per-tensor element count this host path materialises.
const ELEMENT_LIMIT: usize = 1 << 30;

/// A model-independent CPU backend.
pub(crate) struct DynamicHostBackend {
    descriptor: ModelDescriptor,
    architecture: Arc<Architecture>,
    names: Arc<ResolvedNames>,
    source: Arc<dyn WeightSource>,
    context: usize,
}

impl DynamicHostBackend {
    /// Opens the backend, resolving every tensor name the checkpoint actually carries.
    ///
    /// # Errors
    /// A part the dynamic path does not execute, or a tensor none of the canonical spellings
    /// names.
    pub(crate) fn open(
        descriptor: ModelDescriptor,
        architecture: Arc<Architecture>,
        source: Arc<dyn WeightSource>,
        context: usize,
    ) -> Result<Self, String> {
        // The document-level host-support gate (parts, rotary scaling, declared quantization)
        // runs before a weight is resolved or a socket is bound, so an unsupported checkpoint is
        // a boot refusal rather than a failure inside the first request's forward pass.
        if let Some(refusal) =
            sparkflash_ops::parts::host_reference_structure_refusal(&architecture)
        {
            return Err(refusal);
        }
        let names = crate::dynamic::resolve(&architecture, source.as_ref())?;
        validate_quantization(&architecture, source.as_ref())?;
        validate_shapes(&architecture, &names, source.as_ref())?;
        Ok(Self {
            descriptor,
            architecture,
            names: Arc::new(names),
            source,
            context,
        })
    }
}

/// Refuses a checkpoint whose storage the host reference cannot interpret.
///
/// The host path decodes one tensor at a time and never applies a block scale beside it. A
/// ModelOpt NVFP4 checkpoint stores its experts as `U8` beside `weight_scale` tensors, and an
/// FP8 checkpoint stores E4M3 codes beside tile scales; decoding only the codes would quietly
/// serve the wrong arithmetic. The declared-format half is shared with the web control plane, and
/// the storage half reads the same all-name header walk there.
fn validate_quantization(
    architecture: &Architecture,
    source: &dyn WeightSource,
) -> Result<(), String> {
    let quantization = &architecture.quantization;
    let declared = sparkflash_ops::parts::host_reference_quantization_refusal(architecture);
    let storage =
        sparkflash_ops::parts::host_reference_storage_quantization_refusal(architecture, source);
    // The storage check names the actual tensor and dtype; prefer it over the declaration when
    // both refuse, so an operator sees what the shard holds.
    if let Some(refusal) = storage.or(declared) {
        return Err(format!(
            "the checkpoint is quantized (method={:?}, dense={:?}, experts={:?}): {refusal}",
            quantization.method,
            quantization.dense_format.name(),
            quantization.expert_format.name(),
        ));
    }
    Ok(())
}

/// Decodes one host tensor, applying an E4M3 weight's 128x128 inverse-scale tile.
///
/// An FP8 checkpoint stores E4M3 codes beside `<weight>_scale_inv`, a grid of
/// `ceil(rows/128) x ceil(cols/128)` values that multiplies the decoded code. Every other
/// tensor is BF16/F32 and decodes as before. A scale-less E4M3 tensor is refused rather than
/// served as raw codes.
fn decode_host_tensor(
    source: &dyn WeightSource,
    name: &str,
    limit: usize,
) -> Result<Vec<f32>, sparkflash_weights::EngineError> {
    let spec = source.spec(name)?;
    let mut values = decode_by_name(source, name, limit)?;
    if spec.dtype != DType::F8E4M3 {
        return Ok(values);
    }
    let scale_name = format!("{name}_scale_inv");
    let scale_spec =
        source
            .spec(&scale_name)
            .map_err(|_| sparkflash_weights::EngineError::Shape {
                what: "fp8 weight".to_owned(),
                detail: format!("{name} is E4M3 but the checkpoint has no {scale_name}"),
            })?;
    let [rows, cols] = match spec.shape.as_slice() {
        [rows, cols] => [*rows, *cols],
        shape => {
            return Err(sparkflash_weights::EngineError::Shape {
                what: "fp8 weight".to_owned(),
                detail: format!("{name} is {shape:?}, not a two-dimensional weight"),
            });
        }
    };
    if scale_spec.shape != ScaleGrid::Tile(128).shape(rows, cols) {
        return Err(sparkflash_weights::EngineError::Shape {
            what: "fp8 scale grid".to_owned(),
            detail: format!(
                "{scale_name} is {:?}, expected {:?} for {rows}x{cols}",
                scale_spec.shape,
                ScaleGrid::Tile(128).shape(rows, cols)
            ),
        });
    }
    let scales = decode_by_name(source, &scale_name, limit)?;
    apply_scales(&mut values, rows, cols, &scales, ScaleGrid::Tile(128))?;
    Ok(values)
}

/// The shapes [DynamicHostBackend] allocates against, checked from headers before a weight is
/// decoded.
///
/// The forward reads whole tensors, and a bad row count used to reach a slice or an `assert!`
/// in a helper. A wrong shape is a checkpoint property, so it is refused here (and therefore
/// `EXIT_REFUSED`) rather than discovered inside a request.
fn validate_shapes(
    architecture: &Architecture,
    names: &ResolvedNames,
    source: &dyn WeightSource,
) -> Result<(), String> {
    let hidden = architecture.hidden_size;
    let check = |name: &str, expected: &[usize]| -> Result<(), String> {
        let spec = source.spec(name).map_err(|error| error.to_string())?;
        if spec.shape != expected {
            return Err(format!("{name} is {:?}, expected {expected:?}", spec.shape));
        }
        Ok(())
    };
    check(&names.embed, &[architecture.vocab_size, hidden])?;
    check(&names.norm, &[hidden])?;
    check(&names.head, &[architecture.vocab_size, hidden])?;
    for (layer_index, (layer, names)) in architecture.layers.iter().zip(&names.layers).enumerate() {
        check(&names.input_norm, &[hidden])?;
        check(&names.post_norm, &[hidden])?;
        match (&layer.attention.kind, &names.attention) {
            (
                AttentionKind::Gqa {
                    kv_heads,
                    output_gate,
                },
                ResolvedAttention::Gqa {
                    q,
                    k,
                    v,
                    o,
                    q_norm,
                    k_norm,
                },
            ) => {
                let heads = layer.attention.heads;
                let head_dim = layer.attention.head_dim;
                if heads == 0 || head_dim == 0 || *kv_heads == 0 {
                    return Err(format!(
                        "layer {layer_index} has an empty GQA geometry: heads={heads}, kv_heads={kv_heads}, head_dim={head_dim}"
                    ));
                }
                if !heads.is_multiple_of(*kv_heads) {
                    return Err(format!(
                        "layer {layer_index} has {heads} query heads and {kv_heads} key/value heads, which do not divide"
                    ));
                }
                let query_width = heads * head_dim * if *output_gate { 2 } else { 1 };
                let kv_width = *kv_heads * head_dim;
                check(q, &[query_width, hidden])?;
                check(k, &[kv_width, hidden])?;
                check(v, &[kv_width, hidden])?;
                check(o, &[hidden, heads * head_dim])?;
                for norm in [q_norm, k_norm].into_iter().flatten() {
                    check(norm, &[head_dim])?;
                }
            }
            (
                AttentionKind::Gdn(gdn),
                ResolvedAttention::Gdn {
                    qkv,
                    z,
                    b,
                    a,
                    out,
                    conv,
                    a_log,
                    dt_bias,
                    norm,
                },
            ) => {
                if gdn.key_heads == 0
                    || gdn.value_heads == 0
                    || gdn.key_dim == 0
                    || gdn.value_dim == 0
                {
                    return Err(format!("layer {layer_index} has an empty GDN geometry"));
                }
                if !gdn.value_heads.is_multiple_of(gdn.key_heads) {
                    return Err(format!(
                        "layer {layer_index} has {} value heads and {} key heads, which do not divide",
                        gdn.value_heads, gdn.key_heads
                    ));
                }
                let key_width = gdn.key_heads * gdn.key_dim;
                let value_width = gdn.value_heads * gdn.value_dim;
                check(qkv, &[2 * key_width + value_width, hidden])?;
                check(z, &[value_width, hidden])?;
                check(b, &[gdn.value_heads, hidden])?;
                check(a, &[gdn.value_heads, hidden])?;
                check(out, &[hidden, value_width])?;
                let spec = source.spec(conv).map_err(|error| error.to_string())?;
                let taps = gdn.conv_kernel;
                let width = 2 * key_width + value_width;
                let conv_ok = match spec.shape.as_slice() {
                    [channels, 1, kernel] => *channels == width && *kernel == taps,
                    [channels, kernel] => *channels == width && *kernel == taps,
                    _ => false,
                };
                if !conv_ok {
                    return Err(format!(
                        "{conv} is {:?}, expected [{width}, 1, {taps}]",
                        spec.shape
                    ));
                }
                check(a_log, &[gdn.value_heads])?;
                check(dt_bias, &[gdn.value_heads])?;
                check(norm, &[gdn.value_dim])?;
            }
            _ => {
                return Err(format!(
                    "layer {layer_index} resolves its tensors for a different attention family"
                ));
            }
        }
        match (&layer.ffn, &names.ffn) {
            (Ffn::Dense(dense), ResolvedFfn::Dense { gate, up, down }) => {
                check(gate, &[dense.intermediate, hidden])?;
                check(up, &[dense.intermediate, hidden])?;
                check(down, &[hidden, dense.intermediate])?;
            }
            (
                Ffn::Moe(moe),
                ResolvedFfn::Moe {
                    router,
                    experts,
                    shared,
                    shared_gate,
                },
            ) => {
                check(router, &[moe.routed_experts, hidden])?;
                if experts.len() != moe.routed_experts {
                    return Err(format!(
                        "layer {layer_index} resolves {} expert bodies for {} experts",
                        experts.len(),
                        moe.routed_experts
                    ));
                }
                for expert in experts {
                    check(&expert.gate, &[moe.intermediate, hidden])?;
                    check(&expert.up, &[moe.intermediate, hidden])?;
                    check(&expert.down, &[hidden, moe.intermediate])?;
                }
                if let Some(shared) = shared {
                    check(&shared.gate, &[moe.shared_intermediate, hidden])?;
                    check(&shared.up, &[moe.shared_intermediate, hidden])?;
                    check(&shared.down, &[hidden, moe.shared_intermediate])?;
                }
                if let Some(gate) = shared_gate {
                    let spec = source.spec(gate).map_err(|error| error.to_string())?;
                    if spec.elements() != hidden as u128 {
                        return Err(format!(
                            "{gate} is {:?}, expected a {hidden}-wide gate",
                            spec.shape
                        ));
                    }
                }
            }
            _ => {
                return Err(format!(
                    "layer {layer_index} resolves its tensors for a different FFN family"
                ));
            }
        }
    }
    Ok(())
}

impl ModelBackend for DynamicHostBackend {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        let mut kv = Vec::with_capacity(self.architecture.layers.len());
        let mut gdn = Vec::with_capacity(self.architecture.layers.len());
        for layer in &self.architecture.layers {
            match layer.attention.kind {
                AttentionKind::Gqa { .. } => {
                    kv.push(Some(KvCache::default()));
                    gdn.push(None);
                }
                AttentionKind::Gdn(_) => {
                    kv.push(None);
                    gdn.push(Some(GdnState::default()));
                }
                _ => {
                    return Err(BackendError::new(
                        "dynamic_host",
                        format!(
                            "layer {} runs an attention family this host path does not execute",
                            layer.index
                        ),
                    ));
                }
            }
        }
        Ok(Box::new(HostSession {
            backend: Arc::new(SessionShared {
                architecture: Arc::clone(&self.architecture),
                names: Arc::clone(&self.names),
                source: Arc::clone(&self.source),
                context: self.context,
            }),
            kv,
            gdn,
            position: 0,
            history: Vec::new(),
        }))
    }
}

struct SessionShared {
    architecture: Arc<Architecture>,
    names: Arc<ResolvedNames>,
    source: Arc<dyn WeightSource>,
    context: usize,
}

/// One attention layer's cached keys and values, appended a position at a time.
#[derive(Default)]
struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
}

/// One Gated DeltaNet layer's recurrent state.
#[derive(Default)]
struct GdnState {
    conv_history: Vec<Vec<f32>>,
    recurrent: Vec<Vec<f32>>,
}

struct HostSession {
    backend: Arc<SessionShared>,
    kv: Vec<Option<KvCache>>,
    gdn: Vec<Option<GdnState>>,
    position: usize,
    history: Vec<u32>,
}

impl HostSession {
    fn forward(&mut self, token: u32) -> Result<Vec<f32>, String> {
        let shared = &self.backend;
        let architecture = &shared.architecture;
        if token as usize >= architecture.vocab_size {
            return Err(format!(
                "token {token} is outside the checkpoint vocabulary {}",
                architecture.vocab_size
            ));
        }
        let hidden = architecture.hidden_size;
        let zero_centered = architecture.zero_centered_norms();
        let mut x = embedding_row(shared.source.as_ref(), &shared.names.embed, token, hidden)?;
        for (index, layer) in architecture.layers.iter().enumerate() {
            let names = &shared.names.layers[index];
            let norm = decode_host_tensor(shared.source.as_ref(), &names.input_norm, ELEMENT_LIMIT)
                .map_err(|error| error.to_string())?;
            let attention_input =
                rms_norm(&x, &norm, architecture.norm_epsilon as f32, zero_centered);
            let attention = match &layer.attention.kind {
                AttentionKind::Gqa {
                    kv_heads,
                    output_gate,
                } => {
                    let cache = self.kv[index]
                        .as_mut()
                        .ok_or_else(|| format!("layer {index} has no GQA cache"))?;
                    gqa_attention(
                        shared.source.as_ref(),
                        names,
                        &layer.attention,
                        *kv_heads,
                        *output_gate,
                        &attention_input,
                        cache,
                        self.position,
                        architecture.norm_epsilon as f32,
                        zero_centered,
                    )?
                }
                AttentionKind::Gdn(gdn) => {
                    let state = self.gdn[index]
                        .as_mut()
                        .ok_or_else(|| format!("layer {index} has no GDN state"))?;
                    gdn_attention(
                        shared.source.as_ref(),
                        names,
                        &layer.attention,
                        gdn,
                        &attention_input,
                        state,
                        architecture.norm_epsilon as f32,
                    )?
                }
                _ => {
                    return Err(format!(
                        "layer {index} runs an attention family this host path does not execute"
                    ));
                }
            };
            for (value, update) in x.iter_mut().zip(&attention) {
                *value += update;
            }
            let norm = decode_host_tensor(shared.source.as_ref(), &names.post_norm, ELEMENT_LIMIT)
                .map_err(|error| error.to_string())?;
            let ffn_input = rms_norm(&x, &norm, architecture.norm_epsilon as f32, zero_centered);
            let ffn = match &layer.ffn {
                Ffn::Dense(dense) => dense_ffn(
                    shared.source.as_ref(),
                    names,
                    dense.activation,
                    &ffn_input,
                    hidden,
                    dense.intermediate,
                )?,
                Ffn::Moe(moe) => moe_ffn(shared.source.as_ref(), names, &ffn_input, hidden, moe)?,
            };
            for (value, update) in x.iter_mut().zip(&ffn) {
                *value += update;
            }
        }
        let norm = decode_host_tensor(shared.source.as_ref(), &shared.names.norm, ELEMENT_LIMIT)
            .map_err(|error| error.to_string())?;
        let final_hidden = rms_norm(&x, &norm, architecture.norm_epsilon as f32, zero_centered);
        let (vocab, columns, head) =
            matrix(shared.source.as_ref(), &shared.names.head, &final_hidden)?;
        if columns != hidden || vocab != architecture.vocab_size {
            return Err(format!(
                "the LM head is {vocab}x{columns}, the checkpoint is {}x{hidden}",
                architecture.vocab_size
            ));
        }
        self.position += 1;
        self.history.push(token);
        if self.position > shared.context {
            return Err(format!(
                "the sequence is past the served context {}",
                shared.context
            ));
        }
        Ok(head)
    }
}

impl ModelSession for HostSession {
    fn set_sampling(&mut self, sampling: Sampling) {
        // The host path serves greedy decoding. A positive temperature would need a sampler with
        // a stable seed; the dynamic acceptance path only asks for a completed completion.
        let _ = sampling;
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::new(
                "dynamic_host",
                "the host path needs at least one prompt token",
            ));
        }
        let mut logits = Vec::new();
        for token in tokens {
            logits = self
                .forward(*token)
                .map_err(|error| BackendError::new("dynamic_host", error))?;
        }
        Ok(StepOutput {
            next_token: argmax(&logits).map(|index| index as u32),
            logits,
            draft: Vec::new(),
        })
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        let logits = self
            .forward(token)
            .map_err(|error| BackendError::new("dynamic_host", error))?;
        Ok(StepOutput {
            next_token: argmax(&logits).map(|index| index as u32),
            logits,
            draft: Vec::new(),
        })
    }

    fn reusable_prefix(&self) -> Option<&[u32]> {
        // A fresh request opens a fresh session; the host path does not promise a prefix state.
        None
    }
}

fn embedding_row(
    source: &dyn WeightSource,
    name: &str,
    token: u32,
    hidden: usize,
) -> Result<Vec<f32>, String> {
    let spec = source.spec(name).map_err(|error| error.to_string())?;
    if spec.shape.len() != 2 || spec.shape[1] != hidden {
        return Err(format!("{name} is {:?}, not [vocab, {hidden}]", spec.shape));
    }
    let values =
        decode_host_tensor(source, name, ELEMENT_LIMIT).map_err(|error| error.to_string())?;
    let start = token as usize * hidden;
    values
        .get(start..start + hidden)
        .map(<[f32]>::to_vec)
        .ok_or_else(|| format!("{name} has no row {token}"))
}

/// A `[rows, columns]` matrix times a `columns` vector.
fn matrix(
    source: &dyn WeightSource,
    name: &str,
    vector: &[f32],
) -> Result<(usize, usize, Vec<f32>), String> {
    let spec = source.spec(name).map_err(|error| error.to_string())?;
    if spec.shape.len() != 2 {
        return Err(format!("{name} is {:?}, not a matrix", spec.shape));
    }
    let (rows, columns) = (spec.shape[0], spec.shape[1]);
    if vector.len() != columns {
        return Err(format!(
            "{name} is {rows}x{columns}, the vector is {} wide",
            vector.len()
        ));
    }
    let values =
        decode_host_tensor(source, name, ELEMENT_LIMIT).map_err(|error| error.to_string())?;
    if values.len() != rows * columns {
        return Err(format!(
            "{name} decodes to {} values, its {rows}x{columns} shape needs {}",
            values.len(),
            rows * columns
        ));
    }
    let mut out = vec![0.0f32; rows];
    for (row, slot) in out.iter_mut().enumerate() {
        let row_values = &values[row * columns..(row + 1) * columns];
        *slot = row_values
            .iter()
            .zip(vector)
            .map(|(weight, input)| weight * input)
            .sum();
    }
    Ok((rows, columns, out))
}

fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32, zero_centered: bool) -> Vec<f32> {
    let width = input.len();
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / width as f32;
    let scale = 1.0 / (mean_square + epsilon).sqrt();
    input
        .iter()
        .zip(weight)
        .map(|(value, weight)| {
            let weight = if zero_centered { 1.0 + weight } else { *weight };
            value * scale * weight
        })
        .collect()
}

fn activate(activation: Activation, value: f32) -> f32 {
    match activation {
        Activation::Silu => silu(value),
        Activation::Gelu => 0.5 * value * (1.0 + erf(value * std::f32::consts::FRAC_1_SQRT_2)),
        Activation::GeluTanh => {
            0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * value.powi(3))).tanh())
        }
        Activation::Relu => value.max(0.0),
        Activation::Sigmoid => sigmoid(value),
    }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn softplus(value: f32) -> f32 {
    value.max(0.0) + (-value.abs()).exp().ln_1p()
}

/// Abramowitz and Stegun 7.1.26, accurate enough for a reference path that is not a numerics
/// claim.
fn erf(value: f32) -> f32 {
    let sign = value.signum();
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp());
    sign * y
}

#[allow(clippy::too_many_arguments)]
fn gqa_attention(
    source: &dyn WeightSource,
    names: &crate::dynamic::ResolvedLayer,
    attention: &sparkflash_model::architecture::Attention,
    kv_heads: usize,
    output_gate: bool,
    input: &[f32],
    cache: &mut KvCache,
    position: usize,
    epsilon: f32,
    zero_centered: bool,
) -> Result<Vec<f32>, String> {
    let ResolvedAttention::Gqa {
        q,
        k,
        v,
        o,
        q_norm,
        k_norm,
    } = &names.attention
    else {
        return Err("a GQA layer carries another attention's tensors".to_owned());
    };
    let heads = attention.heads;
    let head_dim = attention.head_dim;
    let (_, _, query) = matrix(source, q, input)?;
    let (_, _, key) = matrix(source, k, input)?;
    let (_, _, value) = matrix(source, v, input)?;
    let query_width = heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let expected_query = if output_gate {
        2 * query_width
    } else {
        query_width
    };
    if query.len() != expected_query || key.len() != kv_width || value.len() != kv_width {
        return Err(format!(
            "the GQA projections are {}/{}/{}, expected {expected_query}/{kv_width}/{kv_width}",
            query.len(),
            key.len(),
            value.len()
        ));
    }
    // Qwen's gated GQA projection interleaves query and gate per head:
    // [head0 query | head0 gate | head1 query | head1 gate | ...]. Splitting the projection
    // in half instead pairs head i's query with head (i + heads/2)'s gate.
    let (mut queries, gates) = if output_gate {
        let mut queries = vec![0.0f32; query_width];
        let mut gates = vec![0.0f32; query_width];
        for head in 0..heads {
            let source_head = head * 2 * head_dim;
            queries[head * head_dim..(head + 1) * head_dim]
                .copy_from_slice(&query[source_head..source_head + head_dim]);
            gates[head * head_dim..(head + 1) * head_dim]
                .copy_from_slice(&query[source_head + head_dim..source_head + 2 * head_dim]);
        }
        (queries, Some(gates))
    } else {
        (query, None)
    };
    let mut keys = key;
    if let Some(q_norm) = q_norm {
        let weights =
            decode_host_tensor(source, q_norm, ELEMENT_LIMIT).map_err(|error| error.to_string())?;
        if weights.len() != head_dim {
            return Err(format!(
                "{q_norm} holds {} values, expected {head_dim}",
                weights.len()
            ));
        }
        for head in queries.chunks_mut(head_dim) {
            let normalized = rms_norm(head, &weights, epsilon, zero_centered);
            head.copy_from_slice(&normalized);
        }
    }
    if let Some(k_norm) = k_norm {
        let weights =
            decode_host_tensor(source, k_norm, ELEMENT_LIMIT).map_err(|error| error.to_string())?;
        if weights.len() != head_dim {
            return Err(format!(
                "{k_norm} holds {} values, expected {head_dim}",
                weights.len()
            ));
        }
        for head in keys.chunks_mut(head_dim) {
            let normalized = rms_norm(head, &weights, epsilon, zero_centered);
            head.copy_from_slice(&normalized);
        }
    }
    apply_rope(&mut queries, heads, head_dim, attention, position)?;
    apply_rope(&mut keys, kv_heads, head_dim, attention, position)?;
    cache.k.extend_from_slice(&keys);
    cache.v.extend_from_slice(&value);
    let groups = heads / kv_heads.max(1);
    let scale = (head_dim as f32).sqrt().recip();
    let mut out = vec![0.0f32; query_width];
    for head in 0..heads {
        let kv_head = head / groups;
        let mut scores = Vec::with_capacity(cache.k.len() / kv_width);
        for position in 0..cache.k.len() / kv_width {
            let key = &cache.k[position * kv_width + kv_head * head_dim
                ..position * kv_width + (kv_head + 1) * head_dim];
            let query = &queries[head * head_dim..(head + 1) * head_dim];
            scores.push(
                query
                    .iter()
                    .zip(key)
                    .map(|(query, key)| query * key)
                    .sum::<f32>()
                    * scale,
            );
        }
        let probabilities = softmax(&scores);
        let target = &mut out[head * head_dim..(head + 1) * head_dim];
        for (position, probability) in probabilities.iter().enumerate() {
            let value = &cache.v[position * kv_width + kv_head * head_dim
                ..position * kv_width + (kv_head + 1) * head_dim];
            for (slot, value) in target.iter_mut().zip(value) {
                *slot += probability * value;
            }
        }
        if let Some(gates) = &gates {
            for (slot, gate) in target
                .iter_mut()
                .zip(&gates[head * head_dim..(head + 1) * head_dim])
            {
                *slot *= sigmoid(*gate);
            }
        }
    }
    let (_, _, projected) = matrix(source, o, &out)?;
    if projected.len() != input.len() {
        return Err(format!(
            "the GQA output projection is {} wide, the residual is {}",
            projected.len(),
            input.len()
        ));
    }
    Ok(projected)
}

/// Hugging Face's `rotate_half` RoPE: dimension `i` pairs with `i + rotated/2` inside each
/// head. A partial rotation rotates exactly the first `rope.rotated_dims` dimensions.
fn apply_rope(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    attention: &sparkflash_model::architecture::Attention,
    position: usize,
) -> Result<(), String> {
    let rotated = attention.rope.rotated_dims.min(head_dim);
    if rotated < 2 {
        return Ok(());
    }
    if !rotated.is_multiple_of(2) {
        return Err(format!(
            "the rope rotates {rotated} dimensions per head, which is not even"
        ));
    }
    let mut scaling = 1.0f32;
    if let Some(rope_scaling) = &attention.rope.scaling {
        match rope_scaling.kind.to_ascii_lowercase().as_str() {
            "default" | "" => {}
            "linear" => {
                if rope_scaling.factor <= 0.0 {
                    return Err(format!(
                        "rope.scaling linear declares factor {}",
                        rope_scaling.factor
                    ));
                }
                scaling = rope_scaling.factor as f32;
            }
            other => {
                return Err(format!(
                    "rope.scaling {other:?} is declared, which the dynamic host path does not apply; use a plain or linear rope, or serve with a device pack"
                ));
            }
        }
    }
    let half = rotated / 2;
    let pairs = half;
    let theta = attention.rope.theta.max(1.0) as f32;
    for head in 0..heads {
        let base = head * head_dim;
        for pair in 0..pairs {
            let frequency = 1.0 / theta.powf((2 * pair) as f32 / rotated as f32) / scaling;
            let angle = position as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            let left = values[base + pair];
            let right = values[base + pair + half];
            values[base + pair] = left * cos - right * sin;
            values[base + pair + half] = left * sin + right * cos;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn gdn_attention(
    source: &dyn WeightSource,
    names: &crate::dynamic::ResolvedLayer,
    _attention: &sparkflash_model::architecture::Attention,
    gdn: &sparkflash_model::architecture::Gdn,
    input: &[f32],
    state: &mut GdnState,
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    let ResolvedAttention::Gdn {
        qkv,
        z,
        b,
        a,
        out,
        conv,
        a_log,
        dt_bias,
        norm,
    } = &names.attention
    else {
        return Err("a GDN layer carries another attention's tensors".to_owned());
    };
    let key_width = gdn.key_heads * gdn.key_dim;
    let value_width = gdn.value_heads * gdn.value_dim;
    let width = 2 * key_width + value_width;
    let (_, _, raw) = matrix(source, qkv, input)?;
    if raw.len() != width {
        return Err(format!(
            "the GDN fused projection is {}, expected {width}",
            raw.len()
        ));
    }
    let conv_kernel =
        decode_host_tensor(source, conv, ELEMENT_LIMIT).map_err(|error| error.to_string())?;
    if conv_kernel.len() != width * gdn.conv_kernel {
        return Err(format!(
            "{conv} holds {} values, expected [{width}, 1, {}]",
            conv_kernel.len(),
            gdn.conv_kernel
        ));
    }
    let convolved = causal_conv(
        std::slice::from_ref(&raw),
        &state.conv_history,
        &conv_kernel,
        width,
        gdn.conv_kernel,
        silu,
    );
    let (_, _, z_values) = matrix(source, z, input)?;
    let (_, _, b_values) = matrix(source, b, input)?;
    let (_, _, a_values) = matrix(source, a, input)?;
    if z_values.len() != value_width {
        return Err(format!(
            "{z} is {} wide, expected {value_width}",
            z_values.len()
        ));
    }
    for (name, values) in [("in_proj_b", &b_values), ("in_proj_a", &a_values)] {
        if values.len() != gdn.value_heads {
            return Err(format!(
                "the GDN {name} projection is {} wide, expected {} value heads",
                values.len(),
                gdn.value_heads
            ));
        }
    }
    let (_, _, a_log_values) = matrix_or_vector(source, a_log)?;
    let (_, _, dt_values) = matrix_or_vector(source, dt_bias)?;
    let (_, _, norm_values) = matrix_or_vector(source, norm)?;
    if a_log_values.len() != gdn.value_heads || dt_values.len() != gdn.value_heads {
        return Err(format!(
            "the GDN decay vectors are {}/{} wide, expected {}",
            a_log_values.len(),
            dt_values.len(),
            gdn.value_heads
        ));
    }
    if norm_values.len() != gdn.value_dim {
        return Err(format!(
            "{norm} holds {} values, expected {}",
            norm_values.len(),
            gdn.value_dim
        ));
    }
    let mixed = &convolved[0];
    let groups = gdn.value_heads / gdn.key_heads.max(1);
    let mut queries = vec![0.0f32; gdn.value_heads * gdn.key_dim];
    let mut keys = vec![0.0f32; gdn.value_heads * gdn.key_dim];
    for head in 0..gdn.value_heads {
        let source_head = head / groups.max(1);
        for dim in 0..gdn.key_dim {
            queries[head * gdn.key_dim + dim] = mixed[source_head * gdn.key_dim + dim];
            keys[head * gdn.key_dim + dim] = mixed[key_width + source_head * gdn.key_dim + dim];
        }
    }
    let values = &mixed[2 * key_width..2 * key_width + value_width];
    let mut qn = vec![0.0f32; queries.len()];
    let mut kn = vec![0.0f32; keys.len()];
    let scale = (gdn.key_dim as f32).sqrt().recip();
    for ((q, k), (q_out, k_out)) in queries
        .chunks(gdn.key_dim)
        .zip(keys.chunks(gdn.key_dim))
        .zip(qn.chunks_mut(gdn.key_dim).zip(kn.chunks_mut(gdn.key_dim)))
    {
        l2_normalise(q, scale, q_out);
        l2_normalise(k, 1.0, k_out);
    }
    if state.recurrent.is_empty() {
        state.recurrent = (0..gdn.value_heads)
            .map(|_| vec![0.0f32; gdn.key_dim * gdn.value_dim])
            .collect();
    }
    let mut core = vec![0.0f32; value_width];
    for head in 0..gdn.value_heads {
        let a_log = a_log_values.get(head).copied().ok_or_else(|| {
            format!(
                "A_log holds {} values, {head} requested",
                a_log_values.len()
            )
        })?;
        let dt = dt_values
            .get(head)
            .copied()
            .ok_or_else(|| format!("dt_bias holds {} values, {head} requested", dt_values.len()))?;
        let decay =
            (-a_log.exp() * softplus(a_values.get(head).copied().unwrap_or(0.0) + dt)).exp();
        let beta = sigmoid(b_values.get(head).copied().unwrap_or(0.0));
        delta_rule_step(
            &mut state.recurrent[head],
            Decay::Head(decay),
            &kn[head * gdn.key_dim..(head + 1) * gdn.key_dim],
            &values[head * gdn.value_dim..(head + 1) * gdn.value_dim],
            beta,
            &qn[head * gdn.key_dim..(head + 1) * gdn.key_dim],
            &mut core[head * gdn.value_dim..(head + 1) * gdn.value_dim],
        );
    }
    let mut normed = vec![0.0f32; value_width];
    for head in 0..gdn.value_heads {
        let gated = rms_norm_gated(
            &core[head * gdn.value_dim..(head + 1) * gdn.value_dim],
            &norm_values,
            &z_values[head * gdn.value_dim..(head + 1) * gdn.value_dim],
            epsilon,
            activation_of(gdn.output_gate),
        );
        normed[head * gdn.value_dim..(head + 1) * gdn.value_dim].copy_from_slice(&gated);
    }
    advance_conv_history(&mut state.conv_history, vec![raw], gdn.conv_kernel);
    let (_, _, projected) = matrix(source, out, &normed)?;
    if projected.len() != input.len() {
        return Err(format!(
            "the GDN output projection is {} wide, the residual is {}",
            projected.len(),
            input.len()
        ));
    }
    Ok(projected)
}

fn activation_of(activation: Activation) -> impl Fn(f32) -> f32 {
    move |value| activate(activation, value)
}

fn matrix_or_vector(
    source: &dyn WeightSource,
    name: &str,
) -> Result<(usize, usize, Vec<f32>), String> {
    let spec = source.spec(name).map_err(|error| error.to_string())?;
    let values =
        decode_host_tensor(source, name, ELEMENT_LIMIT).map_err(|error| error.to_string())?;
    match spec.shape.len() {
        1 => Ok((1, spec.shape[0], values)),
        2 => Ok((spec.shape[0], spec.shape[1], values)),
        _ => Err(format!(
            "{name} is {:?}, not a vector or matrix",
            spec.shape
        )),
    }
}

fn dense_ffn(
    source: &dyn WeightSource,
    names: &crate::dynamic::ResolvedLayer,
    activation: Activation,
    input: &[f32],
    hidden: usize,
    intermediate: usize,
) -> Result<Vec<f32>, String> {
    let ResolvedFfn::Dense { gate, up, down } = &names.ffn else {
        return Err("a dense layer carries another FFN's tensors".to_owned());
    };
    let (_, _, gate_values) = matrix(source, gate, input)?;
    let (_, _, up_values) = matrix(source, up, input)?;
    let activated: Vec<f32> = gate_values
        .iter()
        .zip(&up_values)
        .map(|(gate, up)| activate(activation, *gate) * up)
        .collect();
    let (_, columns, out) = matrix(source, down, &activated)?;
    if columns != intermediate || out.len() != hidden {
        return Err(format!(
            "the dense down projection is {}x{columns}, expected {hidden}x{intermediate}",
            out.len()
        ));
    }
    Ok(out)
}

fn moe_ffn(
    source: &dyn WeightSource,
    names: &crate::dynamic::ResolvedLayer,
    input: &[f32],
    hidden: usize,
    moe: &sparkflash_model::architecture::Moe,
) -> Result<Vec<f32>, String> {
    let ResolvedFfn::Moe {
        router,
        experts,
        shared,
        shared_gate,
    } = &names.ffn
    else {
        return Err("an MoE layer carries another FFN's tensors".to_owned());
    };
    let (_, _, logits) = matrix(source, router, input)?;
    let probabilities = match moe.score {
        Score::Softmax => softmax(&logits),
        Score::Sigmoid => logits.iter().map(|value| sigmoid(*value)).collect(),
        Score::SqrtSoftPlus => logits
            .iter()
            .map(|value| softplus(*value).max(0.0).sqrt())
            .collect(),
    };
    if probabilities.len() != experts.len() {
        return Err(format!(
            "the router scores {} experts, the checkpoint stores {}",
            probabilities.len(),
            experts.len()
        ));
    }
    let mut selected: Vec<(usize, f32)> = probabilities.iter().copied().enumerate().collect();
    selected.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    selected.truncate(moe.active_experts);
    if moe.norm_topk {
        let total: f32 = selected.iter().map(|(_, weight)| weight).sum();
        if total > 0.0 {
            for (_, weight) in &mut selected {
                *weight /= total;
            }
        }
    }
    let mut out = vec![0.0f32; hidden];
    for (expert, weight) in selected {
        let names = &experts[expert];
        let gate = matrix(source, &names.gate, input)?.2;
        let up = matrix(source, &names.up, input)?.2;
        let activated: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| activate(moe.activation, *gate) * up)
            .collect();
        let body = matrix(source, &names.down, &activated)?.2;
        for (slot, value) in out.iter_mut().zip(&body) {
            *slot += weight * moe.routed_scaling as f32 * value;
        }
    }
    if let Some(shared) = shared {
        let gate = matrix(source, &shared.gate, input)?.2;
        let up = matrix(source, &shared.up, input)?.2;
        let activated: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| activate(moe.activation, *gate) * up)
            .collect();
        let body = matrix(source, &shared.down, &activated)?.2;
        let scalar = match shared_gate {
            Some(name) => {
                let (_, _, values) = matrix_or_vector(source, name)?;
                let score: f32 = values.iter().zip(input).map(|(w, x)| w * x).sum();
                sigmoid(score)
            }
            None => 1.0,
        };
        for (slot, value) in out.iter_mut().zip(&body) {
            *slot += scalar * value;
        }
    }
    Ok(out)
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = values.iter().map(|value| (value - max).exp()).collect();
    let total: f32 = out.iter().sum();
    if total > 0.0 {
        for value in &mut out {
            *value /= total;
        }
    }
    out
}

fn argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| {
            left.1
                .partial_cmp(right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use sparkflash_model::Part;
    use sparkflash_model::architecture::{Attention, AttentionKind, Rope};
    use sparkflash_weights::dtype::DType;
    use sparkflash_weights::spec::TensorSpec;
    use sparkflash_weights::weights::{WeightError, WeightSource};

    use super::*;

    struct Memory(BTreeMap<String, (TensorSpec, Vec<u8>)>);

    impl Memory {
        fn new() -> Self {
            Self(BTreeMap::new())
        }

        fn push(&mut self, name: &str, shape: &[usize], values: impl Fn(usize) -> f32) {
            let elements: usize = shape.iter().product();
            let data: Vec<u8> = (0..elements)
                .flat_map(|index| values(index).to_le_bytes())
                .collect();
            self.0.insert(
                name.to_owned(),
                (
                    TensorSpec {
                        name: name.to_owned(),
                        dtype: DType::F32,
                        shape: shape.to_vec(),
                    },
                    data,
                ),
            );
        }
    }

    impl WeightSource for Memory {
        fn names(&self) -> Vec<String> {
            self.0.keys().cloned().collect()
        }

        fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
            self.0
                .get(name)
                .map(|(spec, _)| spec.clone())
                .ok_or_else(|| WeightError::Missing {
                    name: name.to_owned(),
                })
        }

        fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
            self.0
                .get(name)
                .map(|(_, bytes)| bytes.clone())
                .ok_or_else(|| WeightError::Missing {
                    name: name.to_owned(),
                })
        }
    }

    #[test]
    fn an_e4m3_weight_is_scaled_by_its_inverse_tile() {
        let mut source = Memory::new();
        source.0.insert(
            "w.weight".to_owned(),
            (
                TensorSpec {
                    name: "w.weight".to_owned(),
                    dtype: DType::F8E4M3,
                    shape: vec![1, 2],
                },
                vec![0x38, 0x40], // E4M3 1.0, 2.0
            ),
        );
        source.push("w.weight_scale_inv", &[1, 1], |_| 2.0);
        let decoded = decode_host_tensor(&source, "w.weight", ELEMENT_LIMIT)
            .expect("a tile-scaled E4M3 weight decodes");
        assert_eq!(decoded, vec![2.0, 4.0]);
        // Without its scale the codes would be served as if they were the weights.
        source.0.remove("w.weight_scale_inv");
        assert!(decode_host_tensor(&source, "w.weight", ELEMENT_LIMIT).is_err());
    }

    fn values(source: &Memory, name: &str) -> Vec<f32> {
        let bytes = source.read(name).expect("tensor");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect()
    }

    fn matvec(matrix: &[f32], rows: usize, columns: usize, vector: &[f32]) -> Vec<f32> {
        (0..rows)
            .map(|row| {
                matrix[row * columns..(row + 1) * columns]
                    .iter()
                    .zip(vector)
                    .map(|(weight, input)| weight * input)
                    .sum()
            })
            .collect()
    }

    fn rms(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
        let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
        let scale = 1.0 / (mean_square + epsilon).sqrt();
        input
            .iter()
            .zip(weight)
            .map(|(value, weight)| value * scale * weight)
            .collect()
    }

    /// A reference written from Hugging Face's arithmetic, not from the implementation's loops:
    /// per-head interleaved q/gate, q/k RMSNorm, rotate_half RoPE, grouped softmax and the
    /// sigmoid output gate.
    fn reference(source: &Memory, steps: usize) -> Vec<f32> {
        let heads = 2usize;
        let kv_heads = 1usize;
        let head_dim = 4usize;
        let hidden = 8usize;
        let theta = 10_000.0f32;
        let epsilon = 1e-5f32;
        let q = values(source, "q");
        let k = values(source, "k");
        let v = values(source, "v");
        let o = values(source, "o");
        let q_norm = values(source, "q_norm");
        let k_norm = values(source, "k_norm");
        let inputs: Vec<Vec<f32>> = (0..steps)
            .map(|step| {
                (0..hidden)
                    .map(|index| 0.03 * (step * hidden + index) as f32 - 0.1)
                    .collect()
            })
            .collect();
        let mut cache_k: Vec<Vec<f32>> = Vec::new();
        let mut cache_v: Vec<Vec<f32>> = Vec::new();
        let mut out = Vec::new();
        for (position, input) in inputs.iter().enumerate() {
            let qg = matvec(&q, heads * 2 * head_dim, hidden, input);
            let mut queries = vec![0.0f32; heads * head_dim];
            let mut gates = vec![0.0f32; heads * head_dim];
            for head in 0..heads {
                for dim in 0..head_dim {
                    queries[head * head_dim + dim] = qg[head * 2 * head_dim + dim];
                    gates[head * head_dim + dim] = qg[head * 2 * head_dim + head_dim + dim];
                }
                let normalized = rms(
                    &queries[head * head_dim..(head + 1) * head_dim],
                    &q_norm,
                    epsilon,
                );
                queries[head * head_dim..(head + 1) * head_dim].copy_from_slice(&normalized);
            }
            let mut keys = matvec(&k, kv_heads * head_dim, hidden, input);
            let normalized = rms(&keys, &k_norm, epsilon);
            keys.copy_from_slice(&normalized);
            let values_out = matvec(&v, kv_heads * head_dim, hidden, input);
            for vector in [&mut queries, &mut keys] {
                for head in vector.chunks_mut(head_dim) {
                    let half = head_dim / 2;
                    for pair in 0..half {
                        let frequency = 1.0 / theta.powf((2 * pair) as f32 / head_dim as f32);
                        let (sin, cos) = (position as f32 * frequency).sin_cos();
                        let left = head[pair];
                        let right = head[pair + half];
                        head[pair] = left * cos - right * sin;
                        head[pair + half] = left * sin + right * cos;
                    }
                }
            }
            cache_k.push(keys);
            cache_v.push(values_out);
            let mut attended = vec![0.0f32; heads * head_dim];
            for head in 0..heads {
                let kv_head = head / (heads / kv_heads);
                let mut scores = Vec::new();
                for (cached_key, _) in cache_k.iter().zip(&cache_v) {
                    let scores_start = kv_head * head_dim;
                    scores.push(
                        queries[head * head_dim..(head + 1) * head_dim]
                            .iter()
                            .zip(&cached_key[scores_start..scores_start + head_dim])
                            .map(|(query, key)| query * key)
                            .sum::<f32>()
                            / (head_dim as f32).sqrt(),
                    );
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut probabilities: Vec<f32> =
                    scores.iter().map(|score| (score - max).exp()).collect();
                let total: f32 = probabilities.iter().sum();
                for probability in &mut probabilities {
                    *probability /= total;
                }
                for (probability, cached_value) in probabilities.iter().zip(&cache_v) {
                    let start = kv_head * head_dim;
                    for dim in 0..head_dim {
                        attended[head * head_dim + dim] += probability * cached_value[start + dim];
                    }
                }
                for dim in 0..head_dim {
                    attended[head * head_dim + dim] *= sigmoid(gates[head * head_dim + dim]);
                }
            }
            out = matvec(&o, hidden, heads * head_dim, &attended);
        }
        out
    }

    #[test]
    fn a_qwen_shaped_gqa_layer_matches_hugging_face_arithmetic() {
        let heads = 2usize;
        let kv_heads = 1usize;
        let head_dim = 4usize;
        let hidden = 8usize;
        let mut source = Memory::new();
        // Every tensor gets a distinct ramp so a wrong pairing, split or norm leaves a visible
        // difference.
        let mut seed = 0usize;
        let mut ramp = |elements: usize| {
            let start = seed;
            seed += elements;
            move |index: usize| 0.05 * ((start + index) % 19) as f32 - 0.45
        };
        source.push(
            "q",
            &[heads * 2 * head_dim, hidden],
            ramp(heads * 2 * head_dim * hidden),
        );
        source.push(
            "k",
            &[kv_heads * head_dim, hidden],
            ramp(kv_heads * head_dim * hidden),
        );
        source.push(
            "v",
            &[kv_heads * head_dim, hidden],
            ramp(kv_heads * head_dim * hidden),
        );
        source.push(
            "o",
            &[hidden, heads * head_dim],
            ramp(hidden * heads * head_dim),
        );
        source.push("q_norm", &[head_dim], ramp(head_dim));
        source.push("k_norm", &[head_dim], ramp(head_dim));
        source.push("input_norm", &[hidden], ramp(hidden));
        source.push("post_norm", &[hidden], ramp(hidden));
        source.push("gate", &[1, hidden], ramp(hidden));
        source.push("up", &[1, hidden], ramp(hidden));
        source.push("down", &[hidden, 1], ramp(hidden));

        let attention = Attention {
            kind: AttentionKind::Gqa {
                kv_heads,
                output_gate: true,
            },
            heads,
            head_dim,
            window: None,
            rope: Rope {
                theta: 10_000.0,
                rotated_dims: head_dim,
                scaling: None,
                mrope_section: None,
            },
            compression: None,
            indexer: None,
        };
        let layer = crate::dynamic::ResolvedLayer {
            input_norm: "input_norm".to_owned(),
            attention: crate::dynamic::ResolvedAttention::Gqa {
                q: "q".to_owned(),
                k: "k".to_owned(),
                v: "v".to_owned(),
                o: "o".to_owned(),
                q_norm: Some("q_norm".to_owned()),
                k_norm: Some("k_norm".to_owned()),
            },
            post_norm: "post_norm".to_owned(),
            ffn: crate::dynamic::ResolvedFfn::Dense {
                gate: "gate".to_owned(),
                up: "up".to_owned(),
                down: "down".to_owned(),
            },
        };
        let inputs: Vec<Vec<f32>> = (0..2)
            .map(|step| {
                (0..hidden)
                    .map(|index| 0.03 * (step * hidden + index) as f32 - 0.1)
                    .collect()
            })
            .collect();
        let mut cache = KvCache::default();
        let mut produced = Vec::new();
        for (position, input) in inputs.iter().enumerate() {
            produced = gqa_attention(
                &source, &layer, &attention, kv_heads, true, input, &mut cache, position, 1e-5,
                false,
            )
            .expect("gqa computes");
        }
        let expected = reference(&source, 2);
        assert_eq!(produced.len(), expected.len());
        for (index, (produced, expected)) in produced.iter().zip(&expected).enumerate() {
            assert!(
                (produced - expected).abs() < 1e-5,
                "element {index}: produced {produced}, Hugging Face reference {expected}"
            );
        }
    }

    #[test]
    fn the_host_capability_list_is_the_shared_dynamic_set() {
        assert_eq!(
            crate::dynamic::DYNAMIC_PARTS,
            sparkflash_ops::parts::HOST_REFERENCE_PARTS
        );
        assert!(!crate::dynamic::DYNAMIC_PARTS.contains(&Part::MlaAttention));
    }
}

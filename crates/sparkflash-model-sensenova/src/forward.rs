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

//! The f32 host reference forward for SenseNova-Loop-9B.
//!
//! The arithmetic follows the released `modeling_sensenova_loop.py`: the execution plan replays
//! the shared block [4, 28) twice, every visit has its own KV or GDN state, and the mHC router
//! normalises the flattened two-stream residual, predicts pre/post/comb, runs twenty log-space
//! Sinkhorn passes, and mixes the streams before each sublayer. The full-attention layers are
//! Qwen3.5 GQA with partial rotary and YaRN; the linear layers are Gated DeltaNet.

use sparkflash_weights::WeightSource;
use sparkflash_weights::linalg::{ScaleGrid, apply_scales, decode_by_name};
use sparkflash_weights::linear_attention::{
    Decay, advance_conv_history, causal_conv, delta_rule_step, l2_normalise, rms_norm_gated,
};

use crate::config::{LayerType, SenseNovaConfig};
use crate::tensor::TEXT_PREFIX;

/// The largest tensor this reference materialises.
pub const ELEMENT_LIMIT: usize = 1 << 30;

/// A row-major f32 matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    /// Rows.
    pub rows: usize,
    /// Columns.
    pub columns: usize,
    /// Values, row-major.
    pub data: Vec<f32>,
}

impl Mat {
    /// A matrix from row-major data.
    ///
    /// # Panics
    /// When the data length is not rows * columns.
    #[must_use]
    pub fn new(rows: usize, columns: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * columns, "matrix shape and data disagree");
        Self {
            rows,
            columns,
            data,
        }
    }

    /// y = W x.
    #[must_use]
    pub fn matvec(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.columns, "matvec input width");
        let mut out = vec![0.0; self.rows];
        for (row, slot) in out.iter_mut().enumerate() {
            let base = row * self.columns;
            let mut accumulator = 0.0f32;
            for (value, activation) in self.data[base..base + self.columns].iter().zip(input) {
                accumulator += value * activation;
            }
            *slot = accumulator;
        }
        out
    }
}

/// An mHC router.
#[derive(Debug, Clone, PartialEq)]
pub struct MhcWeights {
    /// The fp32 projection [rows, streams * hidden].
    pub weight: Mat,
    /// The fp32 bias [rows].
    pub bias: Vec<f32>,
}

/// One layer's attention weights.
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionWeights {
    /// Full grouped-query attention.
    Full {
        /// Query projection, with the per-head gate interleaved.
        q: Mat,
        /// Key projection.
        k: Mat,
        /// Value projection.
        v: Mat,
        /// Output projection.
        o: Mat,
        /// Per-head query norm.
        q_norm: Vec<f32>,
        /// Per-head key norm.
        k_norm: Vec<f32>,
    },
    /// Gated DeltaNet.
    Linear {
        /// Fused q/k/v projection.
        qkv: Mat,
        /// Output gate projection.
        z: Mat,
        /// Beta projection.
        b: Mat,
        /// Decay projection.
        a: Mat,
        /// Output projection.
        out: Mat,
        /// Depthwise convolution [width, 1, taps].
        conv: Vec<f32>,
        /// Log decay rate per value head.
        a_log: Vec<f32>,
        /// Decay bias per value head.
        dt_bias: Vec<f32>,
        /// Gated RMSNorm weight.
        norm: Vec<f32>,
    },
}

/// One physical layer's weights.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerWeights {
    /// Pre-attention norm.
    pub input_norm: Vec<f32>,
    /// Pre-FFN norm.
    pub post_norm: Vec<f32>,
    /// The attention block.
    pub attention: AttentionWeights,
    /// Gate projection.
    pub gate: Mat,
    /// Up projection.
    pub up: Mat,
    /// Down projection.
    pub down: Mat,
    /// The attention-side mHC router, when the layer carries one.
    pub mhc_attn: Option<MhcWeights>,
    /// The FFN-side mHC router, when the layer carries one.
    pub mhc_ffn: Option<MhcWeights>,
}

/// The complete host-reference weights.
#[derive(Debug, Clone, PartialEq)]
pub struct SenseNovaWeights {
    /// The configuration these weights were built for.
    pub config: SenseNovaConfig,
    /// The token embedding.
    pub embed: Mat,
    /// The LM head.
    pub lm_head: Mat,
    /// The final norm.
    pub norm: Vec<f32>,
    /// One body per physical layer.
    pub layers: Vec<LayerWeights>,
}

impl SenseNovaWeights {
    /// Load the text backbone.
    ///
    /// # Errors
    /// A missing or malformed tensor, or a shape that disagrees with the configuration.
    pub fn load(config: &SenseNovaConfig, source: &dyn WeightSource) -> Result<Self, String> {
        let hidden = config.hidden_size;
        let prefix = TEXT_PREFIX;
        let embed = matrix(
            source,
            &format!("{prefix}.embed_tokens.weight"),
            config.vocab_size,
            hidden,
        )?;
        let lm_head = matrix(source, "lm_head.weight", config.vocab_size, hidden)?;
        let norm = vector(source, &format!("{prefix}.norm.weight"), hidden)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (index, kind) in config.layer_types.iter().enumerate() {
            let layer = format!("{prefix}.layers.{index}");
            let attention = match kind {
                LayerType::Full => {
                    let heads = config.num_attention_heads;
                    let head_dim = config.head_dim;
                    AttentionWeights::Full {
                        q: matrix(
                            source,
                            &format!("{layer}.self_attn.q_proj.weight"),
                            2 * heads * head_dim,
                            hidden,
                        )?,
                        k: matrix(
                            source,
                            &format!("{layer}.self_attn.k_proj.weight"),
                            config.kv_width(),
                            hidden,
                        )?,
                        v: matrix(
                            source,
                            &format!("{layer}.self_attn.v_proj.weight"),
                            config.kv_width(),
                            hidden,
                        )?,
                        o: matrix(
                            source,
                            &format!("{layer}.self_attn.o_proj.weight"),
                            hidden,
                            heads * head_dim,
                        )?,
                        q_norm: vector(
                            source,
                            &format!("{layer}.self_attn.q_norm.weight"),
                            head_dim,
                        )?,
                        k_norm: vector(
                            source,
                            &format!("{layer}.self_attn.k_norm.weight"),
                            head_dim,
                        )?,
                    }
                }
                LayerType::Linear => {
                    let fused = config.linear_projection_width();
                    AttentionWeights::Linear {
                        qkv: matrix(
                            source,
                            &format!("{layer}.linear_attn.in_proj_qkv.weight"),
                            fused,
                            hidden,
                        )?,
                        z: matrix(
                            source,
                            &format!("{layer}.linear_attn.in_proj_z.weight"),
                            config.linear_value_width(),
                            hidden,
                        )?,
                        b: matrix(
                            source,
                            &format!("{layer}.linear_attn.in_proj_b.weight"),
                            config.linear.value_heads,
                            hidden,
                        )?,
                        a: matrix(
                            source,
                            &format!("{layer}.linear_attn.in_proj_a.weight"),
                            config.linear.value_heads,
                            hidden,
                        )?,
                        out: matrix(
                            source,
                            &format!("{layer}.linear_attn.out_proj.weight"),
                            hidden,
                            config.linear_value_width(),
                        )?,
                        conv: tensor_as_vector(
                            source,
                            &format!("{layer}.linear_attn.conv1d.weight"),
                            fused * config.linear.conv_kernel,
                        )?,
                        a_log: vector(
                            source,
                            &format!("{layer}.linear_attn.A_log"),
                            config.linear.value_heads,
                        )?,
                        dt_bias: vector(
                            source,
                            &format!("{layer}.linear_attn.dt_bias"),
                            config.linear.value_heads,
                        )?,
                        norm: vector(
                            source,
                            &format!("{layer}.linear_attn.norm.weight"),
                            config.linear.value_dim,
                        )?,
                    }
                }
            };
            let (mhc_attn, mhc_ffn) = if index >= config.mhc.start && index < config.mhc.end {
                let rows = 2 * config.mhc.streams
                    + if config.mhc.fixed_identity {
                        0
                    } else {
                        config.mhc.streams * config.mhc.streams
                    };
                let width = config.mhc.streams * hidden;
                let load = |name: &str| -> Result<MhcWeights, String> {
                    Ok(MhcWeights {
                        weight: matrix(source, &format!("{layer}.{name}.weight"), rows, width)?,
                        bias: vector(source, &format!("{layer}.{name}.bias"), rows)?,
                    })
                };
                (Some(load("mhc_attn")?), Some(load("mhc_ffn")?))
            } else {
                (None, None)
            };
            layers.push(LayerWeights {
                input_norm: vector(source, &format!("{layer}.input_layernorm.weight"), hidden)?,
                post_norm: vector(
                    source,
                    &format!("{layer}.post_attention_layernorm.weight"),
                    hidden,
                )?,
                attention,
                gate: matrix(
                    source,
                    &format!("{layer}.mlp.gate_proj.weight"),
                    config.intermediate_size,
                    hidden,
                )?,
                up: matrix(
                    source,
                    &format!("{layer}.mlp.up_proj.weight"),
                    config.intermediate_size,
                    hidden,
                )?,
                down: matrix(
                    source,
                    &format!("{layer}.mlp.down_proj.weight"),
                    hidden,
                    config.intermediate_size,
                )?,
                mhc_attn,
                mhc_ffn,
            });
        }
        Ok(Self {
            config: config.clone(),
            embed,
            lm_head,
            norm,
            layers,
        })
    }

    /// The logits one token produces, with the reference state updated.
    ///
    /// # Errors
    /// A token outside the vocabulary, or a state that is not at `position`.
    pub fn forward_token(
        &self,
        token: u32,
        position: usize,
        state: &mut SenseNovaState,
    ) -> Result<Vec<f32>, String> {
        let config = &self.config;
        if token as usize >= config.vocab_size {
            return Err(format!(
                "token {token} is outside the checkpoint vocabulary {}",
                config.vocab_size
            ));
        }
        if position != state.position {
            return Err(format!(
                "the state is at position {}, the caller asked for {position}",
                state.position
            ));
        }
        let hidden = config.hidden_size;
        let start = token as usize * hidden;
        let embed = self
            .embed
            .data
            .get(start..start + hidden)
            .ok_or_else(|| format!("the embedding has no row {token}"))?;
        let mut x = embed.to_vec();
        let debug = std::env::var("SPARKFLASH_SENSENOVA_DEBUG").is_ok();
        if debug {
            report("host embed", &x);
        }
        let plan = config.execution_plan();
        for (visit, physical) in plan.into_iter().enumerate() {
            let layer = &self.layers[physical];
            let visit_state = state
                .visits
                .get_mut(visit)
                .ok_or_else(|| format!("visit {visit} has no state"))?;
            let uses_mhc = layer.mhc_attn.is_some();
            let width = config.hidden_size;
            let mut streams: Vec<Vec<f32>> = if uses_mhc {
                if physical == config.mhc.start {
                    vec![x.clone(); config.mhc.streams]
                } else {
                    x.chunks(width).map(<[f32]>::to_vec).collect()
                }
            } else {
                vec![x.clone()]
            };

            let (branch, post, mix) = if let Some(router) = &layer.mhc_attn {
                let (branch, post, mix) = mhc_forward(config, router, &streams)?;
                (branch, Some(post), Some(mix))
            } else {
                (streams[0].clone(), None, None)
            };
            let attention_input = rms_norm(&branch, &layer.input_norm, config.rms_norm_eps as f32);
            let attention = match &layer.attention {
                AttentionWeights::Full {
                    q,
                    k,
                    v,
                    o,
                    q_norm,
                    k_norm,
                } => {
                    let cache = visit_state
                        .kv
                        .as_mut()
                        .ok_or_else(|| format!("layer {physical} has no KV cache"))?;
                    full_attention(
                        config,
                        q,
                        k,
                        v,
                        o,
                        q_norm,
                        k_norm,
                        &attention_input,
                        cache,
                        position,
                    )?
                }
                AttentionWeights::Linear {
                    qkv,
                    z,
                    b,
                    a,
                    out,
                    conv,
                    a_log,
                    dt_bias,
                    norm,
                } => {
                    let gdn = visit_state
                        .gdn
                        .as_mut()
                        .ok_or_else(|| format!("layer {physical} has no GDN state"))?;
                    linear_attention(
                        config,
                        qkv,
                        z,
                        b,
                        a,
                        out,
                        conv,
                        a_log,
                        dt_bias,
                        norm,
                        &attention_input,
                        gdn,
                    )?
                }
            };
            if debug {
                report(
                    &format!("host visit {visit} layer {physical} attention"),
                    &attention,
                );
            }
            if let (Some(post), Some(mix)) = (post, mix) {
                streams = mhc_merge(config, &streams, &attention, &post, &mix);
            } else {
                for (value, update) in streams[0].iter_mut().zip(&attention) {
                    *value += update;
                }
            }

            let (branch, post, mix) = if let Some(router) = &layer.mhc_ffn {
                let (branch, post, mix) = mhc_forward(config, router, &streams)?;
                (branch, Some(post), Some(mix))
            } else {
                (streams[0].clone(), None, None)
            };
            let ffn_input = rms_norm(&branch, &layer.post_norm, config.rms_norm_eps as f32);
            let ffn = mlp(&layer.gate, &layer.up, &layer.down, &ffn_input);
            if debug {
                report(&format!("host visit {visit} layer {physical} ffn"), &ffn);
            }
            if let (Some(post), Some(mix)) = (post, mix) {
                streams = mhc_merge(config, &streams, &ffn, &post, &mix);
            } else {
                for (value, update) in streams[0].iter_mut().zip(&ffn) {
                    *value += update;
                }
            }

            x = if uses_mhc && physical == config.mhc.end - 1 {
                (0..width)
                    .map(|column| {
                        streams.iter().map(|stream| stream[column]).sum::<f32>()
                            / streams.len() as f32
                    })
                    .collect()
            } else if uses_mhc {
                streams.into_iter().flatten().collect()
            } else {
                streams.pop().expect("one stream")
            };
            if debug {
                report(&format!("host hidden after visit {visit}"), &x);
            }
        }
        let final_hidden = rms_norm(&x, &self.norm, config.rms_norm_eps as f32);
        let logits = self.lm_head.matvec(&final_hidden);
        state.position += 1;
        Ok(logits)
    }

    /// The logits of the last token of a prompt, from a fresh state.
    ///
    /// # Errors
    /// As [Self::forward_token].
    pub fn forward_prompt(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let mut state = SenseNovaState::new(&self.config);
        let mut logits = Vec::new();
        for (position, token) in tokens.iter().enumerate() {
            logits = self.forward_token(*token, position, &mut state)?;
        }
        Ok(logits)
    }
}

/// One full-attention visit's cache.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KvCache {
    /// Keys, position * kv_width.
    pub k: Vec<f32>,
    /// Values, position * kv_width.
    pub v: Vec<f32>,
}

/// One GDN visit's recurrent state.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GdnState {
    /// Convolution history.
    pub conv_history: Vec<Vec<f32>>,
    /// One [key_dim][value_dim] state per value head.
    pub recurrent: Vec<Vec<f32>>,
}

/// One visit's state.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VisitState {
    /// The KV cache of a full-attention visit.
    pub kv: Option<KvCache>,
    /// The recurrent state of a linear visit.
    pub gdn: Option<GdnState>,
}

/// The reference state one sequence carries between tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct SenseNovaState {
    /// The next absolute position.
    pub position: usize,
    /// One state per execution visit, in plan order.
    pub visits: Vec<VisitState>,
}

impl SenseNovaState {
    /// A zeroed state for a configuration.
    #[must_use]
    pub fn new(config: &SenseNovaConfig) -> Self {
        let visits = config
            .execution_plan()
            .into_iter()
            .map(|index| match config.layer_types[index] {
                LayerType::Full => VisitState {
                    kv: Some(KvCache::default()),
                    gdn: None,
                },
                LayerType::Linear => VisitState {
                    kv: None,
                    gdn: Some(GdnState::default()),
                },
            })
            .collect();
        Self {
            position: 0,
            visits,
        }
    }
}

fn mhc_forward(
    config: &SenseNovaConfig,
    router: &MhcWeights,
    streams: &[Vec<f32>],
) -> Result<(Vec<f32>, Vec<f32>, Vec<Vec<f32>>), String> {
    let mhc = &config.mhc;
    let mut flat = Vec::with_capacity(mhc.streams * config.hidden_size);
    for stream in streams {
        flat.extend_from_slice(stream);
    }
    let mean_square = flat.iter().map(|value| value * value).sum::<f32>() / flat.len() as f32;
    let scale = 1.0 / (mean_square + mhc.epsilon as f32).sqrt();
    let normalized: Vec<f32> = flat.iter().map(|value| value * scale).collect();
    let logits = router.weight.matvec(&normalized);
    let mut out = Vec::with_capacity(mhc.streams * config.hidden_size);
    for (index, value) in logits.iter().enumerate() {
        out.push(value + router.bias[index]);
    }
    let streams_count = mhc.streams;
    let pre: Vec<f32> = out[..streams_count]
        .iter()
        .map(|value| sigmoid(*value))
        .collect();
    let post: Vec<f32> = out[streams_count..2 * streams_count]
        .iter()
        .map(|value| 2.0 * sigmoid(*value))
        .collect();
    let mix = if mhc.fixed_identity {
        (0..streams_count)
            .map(|row| {
                (0..streams_count)
                    .map(|column| if row == column { 1.0 } else { 0.0 })
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<_>>()
    } else {
        sinkhorn(&out[2 * streams_count..], streams_count, mhc.iterations)
    };
    let mut branch = vec![0.0f32; config.hidden_size];
    for (stream, weight) in streams.iter().zip(&pre) {
        for (slot, value) in branch.iter_mut().zip(stream) {
            *slot += weight * value;
        }
    }
    if mhc.pre_scale != 1.0 {
        let scale = mhc.pre_scale as f32;
        for value in &mut branch {
            *value *= scale;
        }
    }
    Ok((branch, post, mix))
}

fn sinkhorn(logits: &[f32], streams: usize, iterations: usize) -> Vec<Vec<f32>> {
    let mut values: Vec<f64> = logits.iter().map(|value| f64::from(*value)).collect();
    for _ in 0..iterations {
        for row in 0..streams {
            let max = (0..streams)
                .map(|column| values[row * streams + column])
                .fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = (0..streams)
                .map(|column| (values[row * streams + column] - max).exp())
                .sum();
            for column in 0..streams {
                values[row * streams + column] = (values[row * streams + column] - max) - sum.ln();
            }
        }
        for column in 0..streams {
            let max = (0..streams)
                .map(|row| values[row * streams + column])
                .fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = (0..streams)
                .map(|row| (values[row * streams + column] - max).exp())
                .sum();
            for row in 0..streams {
                values[row * streams + column] = (values[row * streams + column] - max) - sum.ln();
            }
        }
    }
    (0..streams)
        .map(|row| {
            (0..streams)
                .map(|column| values[row * streams + column].exp() as f32)
                .collect()
        })
        .collect()
}

fn mhc_merge(
    config: &SenseNovaConfig,
    streams: &[Vec<f32>],
    branch: &[f32],
    post: &[f32],
    mix: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let width = config.hidden_size;
    let mut next = vec![vec![0.0f32; width]; streams.len()];
    for (row, outputs) in next.iter_mut().enumerate() {
        for (source, stream) in streams.iter().enumerate() {
            let weight = mix[row][source];
            for (slot, value) in outputs.iter_mut().zip(stream) {
                *slot += weight * value;
            }
        }
        for (slot, value) in outputs.iter_mut().zip(branch) {
            *slot += post[row] * value;
        }
    }
    next
}

#[allow(clippy::too_many_arguments)]
fn full_attention(
    config: &SenseNovaConfig,
    q: &Mat,
    k: &Mat,
    v: &Mat,
    o: &Mat,
    q_norm: &[f32],
    k_norm: &[f32],
    x: &[f32],
    cache: &mut KvCache,
    position: usize,
) -> Result<Vec<f32>, String> {
    let heads = config.num_attention_heads;
    let kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let groups = heads / kv_heads.max(1);
    let query = q.matvec(x);
    let key = k.matvec(x);
    let value = v.matvec(x);
    let query_width = heads * head_dim;
    let mut queries = vec![0.0f32; query_width];
    let mut gates = vec![0.0f32; query_width];
    for head in 0..heads {
        let source = head * 2 * head_dim;
        queries[head * head_dim..(head + 1) * head_dim]
            .copy_from_slice(&query[source..source + head_dim]);
        gates[head * head_dim..(head + 1) * head_dim]
            .copy_from_slice(&query[source + head_dim..source + 2 * head_dim]);
        let head_slice = &mut queries[head * head_dim..(head + 1) * head_dim];
        let normalized = rms_norm(head_slice, q_norm, config.rms_norm_eps as f32);
        head_slice.copy_from_slice(&normalized);
    }
    let mut keys = key;
    let values = value;
    for head in 0..kv_heads {
        let head_slice = &mut keys[head * head_dim..(head + 1) * head_dim];
        let normalized = rms_norm(head_slice, k_norm, config.rms_norm_eps as f32);
        head_slice.copy_from_slice(&normalized);
    }
    apply_rope(&mut queries, heads, head_dim, config, position);
    apply_rope(&mut keys, kv_heads, head_dim, config, position);
    cache.k.extend_from_slice(&keys);
    cache.v.extend_from_slice(&values);
    let kv_width = kv_heads * head_dim;
    let scale = (head_dim as f32).sqrt().recip();
    let mut out = vec![0.0f32; query_width];
    for head in 0..heads {
        let kv_head = head / groups;
        let query_head = &queries[head * head_dim..(head + 1) * head_dim];
        let mut scores = Vec::with_capacity(position + 1);
        for key_position in 0..=position {
            let start = key_position * kv_width + kv_head * head_dim;
            let key = &cache.k[start..start + head_dim];
            scores.push(
                query_head
                    .iter()
                    .zip(key)
                    .map(|(query, key)| query * key)
                    .sum::<f32>()
                    * scale,
            );
        }
        let probabilities = softmax(&scores);
        let target = &mut out[head * head_dim..(head + 1) * head_dim];
        for (key_position, probability) in probabilities.iter().enumerate() {
            let start = key_position * kv_width + kv_head * head_dim;
            for (slot, value) in target.iter_mut().zip(&cache.v[start..start + head_dim]) {
                *slot += probability * value;
            }
        }
        for (slot, gate) in target
            .iter_mut()
            .zip(&gates[head * head_dim..(head + 1) * head_dim])
        {
            *slot *= sigmoid(*gate);
        }
    }
    let projected = o.matvec(&out);
    if projected.len() != x.len() {
        return Err(format!(
            "the output projection is {} wide, the residual is {}",
            projected.len(),
            x.len()
        ));
    }
    Ok(projected)
}

#[allow(clippy::too_many_arguments)]
fn linear_attention(
    config: &SenseNovaConfig,
    qkv: &Mat,
    z: &Mat,
    b: &Mat,
    a: &Mat,
    out: &Mat,
    conv: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    norm: &[f32],
    x: &[f32],
    state: &mut GdnState,
) -> Result<Vec<f32>, String> {
    let gdn = &config.linear;
    let key_width = gdn.key_heads * gdn.key_dim;
    let value_width = gdn.value_heads * gdn.value_dim;
    let width = 2 * key_width + value_width;
    let raw = qkv.matvec(x);
    if raw.len() != width {
        return Err(format!(
            "the GDN projection is {}, expected {width}",
            raw.len()
        ));
    }
    let convolved = causal_conv(
        std::slice::from_ref(&raw),
        &state.conv_history,
        conv,
        width,
        gdn.conv_kernel,
        silu,
    );
    let z_values = z.matvec(x);
    let b_values = b.matvec(x);
    let a_values = a.matvec(x);
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
        let decay = (-a_log[head].exp() * softplus(a_values[head] + dt_bias[head])).exp();
        let beta = sigmoid(b_values[head]);
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
            norm,
            &z_values[head * gdn.value_dim..(head + 1) * gdn.value_dim],
            config.rms_norm_eps as f32,
            silu,
        );
        normed[head * gdn.value_dim..(head + 1) * gdn.value_dim].copy_from_slice(&gated);
    }
    advance_conv_history(&mut state.conv_history, vec![raw], gdn.conv_kernel);
    Ok(out.matvec(&normed))
}

fn mlp(gate: &Mat, up: &Mat, down: &Mat, x: &[f32]) -> Vec<f32> {
    let debug = std::env::var("SPARKFLASH_SENSENOVA_DEBUG").is_ok();
    let gate_values = gate.matvec(x);
    let up_values = up.matvec(x);
    if debug {
        report("host mlp gate", &gate_values);
        report("host mlp up", &up_values);
    }
    let hidden: Vec<f32> = gate_values
        .iter()
        .zip(&up_values)
        .map(|(gate, up)| silu(*gate) * up)
        .collect();
    if debug {
        report("host mlp hidden", &hidden);
    }
    let out = down.matvec(&hidden);
    if debug {
        report("host mlp down", &out);
    }
    out
}

fn report(label: &str, values: &[f32]) {
    let finite = values.iter().filter(|value| value.is_finite()).count();
    let max = values
        .iter()
        .filter(|value| value.is_finite())
        .fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("{label}: {finite}/{} finite, max {max}", values.len());
}

fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
    let width = input.len();
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / width as f32;
    let scale = 1.0 / (mean_square + epsilon).sqrt();
    input
        .iter()
        .zip(weight)
        .map(|(value, weight)| value * scale * (1.0 + weight))
        .collect()
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponents: Vec<f32> = scores.iter().map(|score| (score - max).exp()).collect();
    let sum: f32 = exponents.iter().sum();
    exponents.iter().map(|value| value / sum).collect()
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn softplus(value: f32) -> f32 {
    value.max(0.0) + (-value.abs()).exp().ln_1p()
}

/// The YaRN-scaled inverse frequencies of the partial-rotary dimensions.
#[must_use]
pub fn yarn_inv_freq(config: &SenseNovaConfig) -> Vec<f64> {
    let rope = &config.rope;
    let dim = rope.dims.max(2);
    let half = dim / 2;
    let mut frequencies = Vec::with_capacity(half);
    for index in 0..half {
        frequencies.push(1.0 / rope.theta.powf((2 * index) as f64 / dim as f64));
    }
    if rope.factor <= 1.0 {
        return frequencies;
    }
    let theta = rope.theta.ln();
    let original = rope.original as f64;
    let low = (dim as f64 * (original / (rope.beta_fast * 2.0 * std::f64::consts::PI)).ln()
        / (2.0 * theta))
        .floor()
        .max(0.0) as usize;
    let high = (dim as f64 * (original / (rope.beta_slow * 2.0 * std::f64::consts::PI)).ln()
        / (2.0 * theta))
        .ceil()
        .min((half - 1) as f64) as usize;
    for (index, frequency) in frequencies.iter_mut().enumerate() {
        let ramp = if index < low {
            0.0
        } else if index >= high {
            1.0
        } else {
            (index - low) as f64 / (high - low) as f64
        };
        *frequency = (1.0 - ramp) * (*frequency / rope.factor) + ramp * *frequency;
    }
    frequencies
}

/// The YaRN attention scaling applied to cos/sin.
#[must_use]
pub fn yarn_attention_scale(config: &SenseNovaConfig) -> f32 {
    if config.rope.factor <= 1.0 {
        1.0
    } else {
        (0.1 * config.rope.factor.ln() + 1.0) as f32
    }
}

fn apply_rope(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    config: &SenseNovaConfig,
    position: usize,
) {
    let rotated = config.rope.dims.min(head_dim);
    if rotated < 2 {
        return;
    }
    let half = rotated / 2;
    let frequencies = yarn_inv_freq(config);
    let scale = yarn_attention_scale(config);
    for head in 0..heads {
        let base = head * head_dim;
        for pair in 0..half {
            let angle = position as f64 * frequencies[pair];
            let (sin, cos) = angle.sin_cos();
            let (sin, cos) = (sin as f32 * scale, cos as f32 * scale);
            let left = values[base + pair];
            let right = values[base + pair + half];
            values[base + pair] = left * cos - right * sin;
            values[base + pair + half] = left * sin + right * cos;
        }
    }
}

fn decode_weight(
    source: &dyn WeightSource,
    name: &str,
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, String> {
    let spec = source.spec(name).map_err(|error| error.to_string())?;
    if spec.shape != [rows, columns] {
        return Err(format!(
            "{name} is {:?}, expected [{rows}, {columns}]",
            spec.shape
        ));
    }
    let mut values =
        decode_by_name(source, name, ELEMENT_LIMIT).map_err(|error| error.to_string())?;
    if spec.dtype == sparkflash_weights::DType::F8E4M3 {
        let scale_name = format!("{name}_scale_inv");
        let scales = decode_by_name(source, &scale_name, ELEMENT_LIMIT)
            .map_err(|error| error.to_string())?;
        apply_scales(&mut values, rows, columns, &scales, ScaleGrid::Tile(128))
            .map_err(|error| error.to_string())?;
    }
    Ok(values)
}

fn matrix(
    source: &dyn WeightSource,
    name: &str,
    rows: usize,
    columns: usize,
) -> Result<Mat, String> {
    Ok(Mat::new(
        rows,
        columns,
        decode_weight(source, name, rows, columns)?,
    ))
}

fn vector(source: &dyn WeightSource, name: &str, width: usize) -> Result<Vec<f32>, String> {
    let spec = source.spec(name).map_err(|error| error.to_string())?;
    if spec.elements() != width as u128 {
        return Err(format!(
            "{name} is {:?}, expected {width} values",
            spec.shape
        ));
    }
    decode_by_name(source, name, ELEMENT_LIMIT).map_err(|error| error.to_string())
}

fn tensor_as_vector(
    source: &dyn WeightSource,
    name: &str,
    width: usize,
) -> Result<Vec<f32>, String> {
    vector(source, name, width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinkhorn_is_column_stochastic_in_log_space() {
        let mix = sinkhorn(&[0.1, 0.2, 0.3, 0.4], 2, 20);
        for column in 0..2 {
            let sum = mix[0][column] + mix[1][column];
            assert!((sum - 1.0).abs() < 1e-5, "column {column} sums to {sum}");
        }
    }

    #[test]
    fn yarn_divides_the_low_frequencies() {
        let config =
            SenseNovaConfig::from_json(include_str!("../config/sensenova_loop_9b.hf.json"))
                .expect("the carried config");
        let plain_theta = 10_000_000.0f64;
        let plain = 1.0 / plain_theta.powf(0.0);
        let scaled = yarn_inv_freq(&config);
        assert!((plain - 1.0).abs() < 1e-12);
        assert!(scaled[0] < 1.0);
    }
}

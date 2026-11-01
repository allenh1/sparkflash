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

//! The f32 host reference forward.
//!
//! This is the oracle a device arm is compared against, not a serving path. It runs the released
//! arithmetic (`Glm5NextTextModel.forward`) at any geometry, one token at a time, so the tiny
//! configuration exercises every branch: the mHC mixers, Kimi Delta Attention with its
//! per-projection convolutions and per-channel decay, NoPE MLA over the indexer's selection, the
//! k-pool indexer and the layer that shares its selection, the sigmoid `noaux_tc` router, and the
//! clamped SwiGLU of every MLP.
//!
//! # Layer order (`Glm5NextTextDecoderLayer.forward`)
//!
//! 1. `attn_hc` mixes the `hc_mult` streams: `(pre, post, comb)` from the normed, flattened
//!    streams, and the branch input `sum_j pre_j * stream_j`;
//! 2. `input_layernorm`, then the branch: KDA, or DSA with its own indexer or the previous layer's
//!    selection;
//! 3. `stream_k = post_k * branch + sum_j comb[j][k] * stream_j`;
//! 4. the same with `ffn_hc`, `post_attention_layernorm` and the MLP, dense or MoE.
//!
//! The embedding is copied onto every stream; after the last layer the streams are averaged,
//! normed and scored by the LM head.
//!
//! # Precision
//!
//! Everything is f32 over decoded weights. A block-scaled FP8 linear quantises its input the way
//! the released W8A8 GEMM does ([quant::quantize_activation_fp8_tiles]) and multiplies the
//! dequantised operand ([quant::dequantized_activation_fp8_tiles]); every other linear multiplies
//! its f32 input. Nothing is rounded to BF16 between operations, so a device arm is compared
//! against this within a tolerance, not bit for bit.
//!
//! # Tensor parallelism
//!
//! [Glm5NextWeights::load_tp] keeps one rank's slice of every tensor the plan splits
//! ([crate::tensor::Split]), and [Glm5NextWeights::forward_tp] runs that rank. Every sublayer's
//! branch ends in a column-split projection (KDA and DSA `o_proj`, every `down_proj`), so its
//! output is a partial sum that [TpRuntime] all-reduces once, and the LM head's row slice crosses
//! the exact [BitGatherHandle]. One rank over [SingleRank] is [Glm5NextWeights::forward], bit for
//! bit.

use sparkflash_weights::linalg::{self, EngineError, ScaleGrid};
use sparkflash_weights::linear_attention::{self, Decay};
use sparkflash_weights::mhc::{self, HcGeometry, Mixes};
use sparkflash_weights::tp::{BitGatherHandle, SingleRank, TpLayout, TpRuntime};
use sparkflash_weights::{DType, WeightSource, quant, topk};

use crate::config::{Glm5NextConfig, IndexerKind, LayerKind, MlpKind};
use crate::tensor::{Glm5NextTensorPlan, LM_PREFIX};

/// The epsilon of the indexer's key LayerNorm (`nn.LayerNorm(head_dim, eps=1e-6)`).
pub(crate) const INDEX_KEY_NORM_EPS: f32 = 1e-6;

fn shape_error(what: &str, detail: impl Into<String>) -> EngineError {
    EngineError::Shape {
        what: what.to_owned(),
        detail: detail.into(),
    }
}

/// `value ** -0.5`, in double precision and then rounded, as the reference's Python scalars are.
pub(crate) fn inverse_sqrt(value: usize) -> f32 {
    (1.0 / (value as f64).sqrt()) as f32
}

/// A decoded linear weight, `[rows, columns]` row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Linear {
    /// Output width.
    pub rows: usize,
    /// Input width.
    pub columns: usize,
    /// The dequantised weight.
    pub weight: Vec<f32>,
    /// For a block-scaled FP8 weight, the tile width its input is quantised over; `None` multiplies
    /// the f32 input.
    pub act_tile: Option<usize>,
}

impl Linear {
    /// `W x`, with the input quantised first when the weight is FP8.
    ///
    /// # Errors
    /// When the input is not `columns` wide, or not whole tiles.
    pub fn apply(&self, input: &[f32]) -> Result<Vec<f32>, EngineError> {
        let dequantised;
        let operand = match self.act_tile {
            None => input,
            Some(tile) => {
                dequantised = quant::dequantized_activation_fp8_tiles(input, tile)?;
                &dequantised
            }
        };
        let mut out = vec![0.0; self.rows];
        linalg::matmul(&mut out, operand, &self.weight, self.rows, self.columns, 1)?;
        Ok(out)
    }
}

/// One mHC mixer's parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Hyper {
    /// `fn`, `[(2 + hc) * hc, hc * hidden]`.
    pub projection: Vec<f32>,
    /// `base`, `[(2 + hc) * hc]`.
    pub base: Vec<f32>,
    /// `scale`, `[3]`: pre, post, comb.
    pub scale: Vec<f32>,
}

/// One KDA layer.
#[derive(Debug, Clone, PartialEq)]
pub struct Kda {
    /// The query, key and value projections, `[heads * head_dim, hidden]`.
    pub qkv: [Linear; 3],
    /// Their depthwise convolutions, `[heads * head_dim][kernel]` each.
    pub conv: [Vec<f32>; 3],
    /// The forget gate's low-rank pair.
    pub f_a: Linear,
    /// See `f_a`.
    pub f_b: Linear,
    /// The output gate's low-rank pair.
    pub g_a: Linear,
    /// See `g_a`.
    pub g_b: Linear,
    /// The write strength, `[heads, hidden]`.
    pub b: Linear,
    /// `[heads]`.
    pub a_log: Vec<f32>,
    /// `[heads * head_dim]`.
    pub dt_bias: Vec<f32>,
    /// The gated output norm's weight, `[head_dim]`.
    pub o_norm: Vec<f32>,
    /// `[hidden, heads * head_dim]`.
    pub o: Linear,
}

/// One DSA layer's own k-pool indexer.
#[derive(Debug, Clone, PartialEq)]
pub struct Indexer {
    /// `[index_heads * index_dim, q_lora]`.
    pub wq_b: Linear,
    /// `[index_dim, hidden]`.
    pub wk: Linear,
    /// The key LayerNorm's weight, `[index_dim]`.
    pub k_norm_weight: Vec<f32>,
    /// The key LayerNorm's bias, `[index_dim]`.
    pub k_norm_bias: Vec<f32>,
    /// `[index_heads, hidden]`.
    pub weights_proj: Linear,
    /// `index_kpool_compress_gate`, `[index_dim, hidden]`.
    pub gate: Linear,
    /// `index_kpool_compress_ape`, `[kpool, index_dim]`.
    pub ape: Vec<f32>,
}

/// One DSA layer.
#[derive(Debug, Clone, PartialEq)]
pub struct Dsa {
    /// `[q_lora, hidden]`.
    pub q_a: Linear,
    /// `[q_lora]`.
    pub q_a_norm: Vec<f32>,
    /// `[heads * nope, q_lora]`.
    pub q_b: Linear,
    /// `[kv_lora, hidden]`.
    pub kv_a: Linear,
    /// `[kv_lora]`.
    pub kv_a_norm: Vec<f32>,
    /// `[heads * (nope + v), kv_lora]`, each head's key half then its value half.
    pub kv_b: Linear,
    /// `[hidden, heads * v]`.
    pub o: Linear,
    /// The layer's own indexer, or `None` for a layer that shares the previous one's selection.
    pub indexer: Option<Indexer>,
}

/// One layer's attention branch.
#[derive(Debug, Clone, PartialEq)]
pub enum Branch {
    /// Kimi Delta Attention.
    Linear(Kda),
    /// DeepSeek Sparse Attention.
    Sparse(Dsa),
}

/// A clamped SwiGLU.
#[derive(Debug, Clone, PartialEq)]
pub struct Mlp {
    /// `[intermediate, hidden]`.
    pub gate: Linear,
    /// `[intermediate, hidden]`.
    pub up: Linear,
    /// `[hidden, intermediate]`.
    pub down: Linear,
}

/// The routed experts, their router and the shared expert.
#[derive(Debug, Clone, PartialEq)]
pub struct Moe {
    /// The router, `[experts, hidden]`, applied in f32.
    pub router: Linear,
    /// The selection bias, `[experts]`.
    pub bias: Vec<f32>,
    /// The shared expert.
    pub shared: Mlp,
    /// The routed experts.
    pub experts: Vec<Mlp>,
}

/// One layer's MLP.
#[derive(Debug, Clone, PartialEq)]
pub enum Ffn {
    /// A dense SwiGLU.
    Dense(Mlp),
    /// Routed experts plus the shared one.
    Moe(Moe),
}

/// One decoder layer.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    /// `input_layernorm`.
    pub input_norm: Vec<f32>,
    /// `post_attention_layernorm`.
    pub post_norm: Vec<f32>,
    /// The attention branch's mixer.
    pub attn_hc: Hyper,
    /// The MLP's mixer.
    pub ffn_hc: Hyper,
    /// The attention branch.
    pub branch: Branch,
    /// The MLP.
    pub ffn: Ffn,
}

/// The decoded text model, or one rank's slice of it: the shapes documented on each weight are
/// the whole model's, and a tensor the plan splits holds this rank's part.
#[derive(Debug, Clone, PartialEq)]
pub struct Glm5NextWeights {
    /// The configuration these weights were loaded for.
    pub config: Glm5NextConfig,
    /// Which rank of which world these weights are.
    pub layout: TpLayout,
    /// `[vocab, hidden]`.
    pub embed: Vec<f32>,
    /// The final norm, `[hidden]`.
    pub norm: Vec<f32>,
    /// `[vocab, hidden]`.
    pub lm_head: Linear,
    /// The decoder layers.
    pub layers: Vec<Layer>,
}

/// What one sequence carries between tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceState {
    /// The next token's position.
    pub position: usize,
    /// One state per decoder layer.
    pub layers: Vec<LayerState>,
}

/// One layer's carried state.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerState {
    /// A KDA layer's convolution histories and delta-rule memory.
    Linear(KdaState),
    /// A DSA layer's caches.
    Sparse(DsaState),
}

/// The state of one KDA layer.
#[derive(Debug, Clone, PartialEq)]
pub struct KdaState {
    /// The last `kernel - 1` pre-convolution rows of the query, key and value projections.
    pub conv: [Vec<Vec<f32>>; 3],
    /// Each head's `[head_dim][head_dim]` memory, row-major by key channel.
    pub recurrent: Vec<Vec<f32>>,
}

/// The caches of one DSA layer, one row per position.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DsaState {
    /// The expanded keys, `heads * nope` wide.
    pub keys: Vec<Vec<f32>>,
    /// The expanded values, `heads * v` wide.
    pub values: Vec<Vec<f32>>,
    /// The indexer's normed keys, `index_dim` wide; empty on a sharing layer.
    pub index_keys: Vec<Vec<f32>>,
    /// The indexer's pool-gate logits, `index_dim` wide; empty on a sharing layer.
    pub index_gates: Vec<Vec<f32>>,
    /// One pooled key per complete pool.
    pub pool_keys: Vec<Vec<f32>>,
}

/// Reads tensors by their released names, checks their shapes, and keeps this rank's slices.
struct Loader<'a, S: WeightSource + ?Sized> {
    source: &'a S,
    config: &'a Glm5NextConfig,
    limit: usize,
    /// Where each tensor's split comes from.
    plan: Glm5NextTensorPlan,
    layout: TpLayout,
}

impl<S: WeightSource + ?Sized> Loader<'_, S> {
    /// One tensor whole, in the shape the configuration gives it.
    fn whole(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, EngineError> {
        let spec = self.source.spec(name).map_err(|_| EngineError::Missing {
            name: name.to_owned(),
        })?;
        if spec.shape != shape {
            return Err(shape_error(
                name,
                format!("stored as {:?}, expected {shape:?}", spec.shape),
            ));
        }
        linalg::decode_by_name(self.source, name, self.limit)
    }

    /// This rank's slice of a decoded tensor, and its shape, by the split the plan gives its name.
    fn slice(
        &self,
        name: &str,
        shape: &[usize],
        values: Vec<f32>,
    ) -> Result<(Vec<usize>, Vec<f32>), EngineError> {
        let split = self
            .plan
            .split_of(name)
            .ok_or_else(|| shape_error(name, "a tensor the served plan does not name"))?;
        split
            .slice(shape, values, self.layout.rank, self.layout.world_size)
            .map_err(|detail| shape_error(name, detail))
    }

    /// This rank's slice of one tensor.
    fn tensor(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, EngineError> {
        let values = self.whole(name, shape)?;
        Ok(self.slice(name, shape, values)?.1)
    }

    /// This rank's slice of a dequantised weight.
    fn sliced(
        &self,
        name: &str,
        rows: usize,
        columns: usize,
        weight: Vec<f32>,
        act_tile: Option<usize>,
    ) -> Result<Linear, EngineError> {
        let (shape, weight) = self.slice(name, &[rows, columns], weight)?;
        Ok(Linear {
            rows: shape[0],
            columns: shape[1],
            weight,
            act_tile,
        })
    }

    /// A plain weight named in full, multiplied in f32.
    fn matrix(&self, name: &str, rows: usize, columns: usize) -> Result<Linear, EngineError> {
        let weight = self.whole(name, &[rows, columns])?;
        self.sliced(name, rows, columns, weight, None)
    }

    /// `{prefix}.weight`: block-scaled FP8 with its tiles when it is stored that way, plain
    /// otherwise. The whole weight is dequantised, then sliced, so a rank's slice is the same
    /// values the whole model multiplies. An FP8 weight quantises its input when the
    /// configuration's [Glm5NextConfig::dynamic_activations] says so.
    fn linear(&self, prefix: &str, rows: usize, columns: usize) -> Result<Linear, EngineError> {
        let name = format!("{prefix}.weight");
        let mut weight = self.whole(&name, &[rows, columns])?;
        let mut act_tile = None;
        if self.source.spec(&name)?.dtype == DType::F8E4M3 {
            let tile = self.config.fp8_block.ok_or_else(|| {
                shape_error(&name, "an FP8 weight with no FP8 tile in the config")
            })?;
            let grid = ScaleGrid::Tile(tile);
            let scales = self.whole(
                &format!("{prefix}.weight_scale_inv"),
                &grid.shape(rows, columns),
            )?;
            linalg::apply_scales(&mut weight, rows, columns, &scales, grid)?;
            act_tile = self.config.dynamic_activations.then_some(tile);
        }
        self.sliced(&name, rows, columns, weight, act_tile)
    }

    fn hyper(&self, prefix: &str) -> Result<Hyper, EngineError> {
        let geometry = HcGeometry::from(self.config);
        let mixes = geometry.hc_mix_count();
        Ok(Hyper {
            projection: self.tensor(&format!("{prefix}_fn"), &[mixes, geometry.hc_width()])?,
            base: self.tensor(&format!("{prefix}_base"), &[mixes])?,
            scale: self.tensor(&format!("{prefix}_scale"), &[3])?,
        })
    }

    fn kda(&self, prefix: &str) -> Result<Kda, EngineError> {
        let config = self.config;
        let (hidden, head, heads) = (
            config.hidden_size,
            config.linear_head_dim,
            config.linear_num_heads,
        );
        let width = config.linear_width();
        let conv = |name: &str| {
            self.tensor(
                &format!("{prefix}.{name}_conv1d.weight"),
                &[width, 1, config.linear_conv_kernel_dim],
            )
        };
        Ok(Kda {
            qkv: [
                self.linear(&format!("{prefix}.q_proj"), width, hidden)?,
                self.linear(&format!("{prefix}.k_proj"), width, hidden)?,
                self.linear(&format!("{prefix}.v_proj"), width, hidden)?,
            ],
            conv: [conv("q")?, conv("k")?, conv("v")?],
            f_a: self.linear(&format!("{prefix}.f_a_proj"), head, hidden)?,
            f_b: self.linear(&format!("{prefix}.f_b_proj"), width, head)?,
            g_a: self.linear(&format!("{prefix}.g_a_proj"), head, hidden)?,
            g_b: self.linear(&format!("{prefix}.g_b_proj"), width, head)?,
            b: self.linear(&format!("{prefix}.b_proj"), heads, hidden)?,
            a_log: self.tensor(&format!("{prefix}.A_log"), &[heads])?,
            dt_bias: self.tensor(&format!("{prefix}.dt_bias"), &[width])?,
            o_norm: self.tensor(&format!("{prefix}.o_norm.weight"), &[head])?,
            o: self.linear(&format!("{prefix}.o_proj"), hidden, width)?,
        })
    }

    fn indexer(&self, prefix: &str) -> Result<Indexer, EngineError> {
        let config = self.config;
        let (hidden, index_dim) = (config.hidden_size, config.index_head_dim);
        Ok(Indexer {
            wq_b: self.linear(
                &format!("{prefix}.wq_b"),
                config.index_n_heads * index_dim,
                config.q_lora_rank,
            )?,
            wk: self.linear(&format!("{prefix}.wk"), index_dim, hidden)?,
            k_norm_weight: self.tensor(&format!("{prefix}.k_norm.weight"), &[index_dim])?,
            k_norm_bias: self.tensor(&format!("{prefix}.k_norm.bias"), &[index_dim])?,
            weights_proj: self.linear(
                &format!("{prefix}.weights_proj"),
                config.index_n_heads,
                hidden,
            )?,
            gate: self.matrix(
                &format!("{prefix}.index_kpool_compress_gate"),
                index_dim,
                hidden,
            )?,
            ape: self.tensor(
                &format!("{prefix}.index_kpool_compress_ape"),
                &[config.index_kpool, index_dim],
            )?,
        })
    }

    fn dsa(&self, prefix: &str, own_indexer: bool) -> Result<Dsa, EngineError> {
        let config = self.config;
        let hidden = config.hidden_size;
        let heads = config.num_attention_heads;
        Ok(Dsa {
            q_a: self.linear(&format!("{prefix}.q_a_proj"), config.q_lora_rank, hidden)?,
            q_a_norm: self.tensor(
                &format!("{prefix}.q_a_layernorm.weight"),
                &[config.q_lora_rank],
            )?,
            q_b: self.linear(
                &format!("{prefix}.q_b_proj"),
                heads * config.qk_nope_head_dim,
                config.q_lora_rank,
            )?,
            kv_a: self.linear(
                &format!("{prefix}.kv_a_proj_with_mqa"),
                config.kv_lora_rank,
                hidden,
            )?,
            kv_a_norm: self.tensor(
                &format!("{prefix}.kv_a_layernorm.weight"),
                &[config.kv_lora_rank],
            )?,
            kv_b: self.linear(
                &format!("{prefix}.kv_b_proj"),
                heads * (config.qk_nope_head_dim + config.v_head_dim),
                config.kv_lora_rank,
            )?,
            o: self.linear(
                &format!("{prefix}.o_proj"),
                hidden,
                heads * config.v_head_dim,
            )?,
            indexer: if own_indexer {
                Some(self.indexer(&format!("{prefix}.indexer"))?)
            } else {
                None
            },
        })
    }

    fn mlp(&self, prefix: &str, intermediate: usize) -> Result<Mlp, EngineError> {
        let hidden = self.config.hidden_size;
        Ok(Mlp {
            gate: self.linear(&format!("{prefix}.gate_proj"), intermediate, hidden)?,
            up: self.linear(&format!("{prefix}.up_proj"), intermediate, hidden)?,
            down: self.linear(&format!("{prefix}.down_proj"), hidden, intermediate)?,
        })
    }

    fn moe(&self, prefix: &str) -> Result<Moe, EngineError> {
        let config = self.config;
        let experts = config.n_routed_experts;
        Ok(Moe {
            router: self.matrix(
                &format!("{prefix}.gate.weight"),
                experts,
                config.hidden_size,
            )?,
            bias: self.tensor(
                &format!("{prefix}.gate.e_score_correction_bias"),
                &[experts],
            )?,
            shared: self.mlp(
                &format!("{prefix}.shared_experts"),
                config.moe_intermediate_size * config.n_shared_experts,
            )?,
            experts: (0..experts)
                .map(|expert| {
                    self.mlp(
                        &format!("{prefix}.experts.{expert}"),
                        config.moe_intermediate_size,
                    )
                })
                .collect::<Result<_, _>>()?,
        })
    }

    fn layer(&self, layer: usize) -> Result<Layer, EngineError> {
        let config = self.config;
        let prefix = format!("{LM_PREFIX}.layers.{layer}");
        let hidden = config.hidden_size;
        Ok(Layer {
            input_norm: self.tensor(&format!("{prefix}.input_layernorm.weight"), &[hidden])?,
            post_norm: self.tensor(
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[hidden],
            )?,
            attn_hc: self.hyper(&format!("{prefix}.hc_attn"))?,
            ffn_hc: self.hyper(&format!("{prefix}.hc_ffn"))?,
            branch: match config.layer_types[layer] {
                LayerKind::Linear => Branch::Linear(self.kda(&format!("{prefix}.self_attn"))?),
                LayerKind::Sparse => Branch::Sparse(self.dsa(
                    &format!("{prefix}.self_attn"),
                    config.indexer_types[layer] == IndexerKind::Full,
                )?),
            },
            ffn: match config.mlp_layer_types[layer] {
                MlpKind::Dense => {
                    Ffn::Dense(self.mlp(&format!("{prefix}.mlp"), config.intermediate_size)?)
                }
                MlpKind::Moe => Ffn::Moe(self.moe(&format!("{prefix}.mlp"))?),
            },
        })
    }
}

impl Kda {
    /// One token through the layer: the convolved projections, the delta rule, the gated norm and
    /// the output projection.
    fn step(
        &self,
        config: &Glm5NextConfig,
        state: &mut KdaState,
        input: &[f32],
    ) -> Result<Vec<f32>, EngineError> {
        // This rank's heads' channels.
        let width = self.o.columns;
        let head_dim = config.linear_head_dim;
        let taps = config.linear_conv_kernel_dim;
        let mut convolved: Vec<Vec<f32>> = Vec::with_capacity(3);
        for ((projection, kernel), history) in self.qkv.iter().zip(&self.conv).zip(&mut state.conv)
        {
            let row = projection.apply(input)?;
            let mut out = linear_attention::causal_conv(
                std::slice::from_ref(&row),
                history,
                kernel,
                width,
                taps,
                linalg::silu,
            );
            linear_attention::advance_conv_history(history, vec![row], taps);
            convolved.push(out.swap_remove(0));
        }
        let forget = self.f_b.apply(&self.f_a.apply(input)?)?;
        let write = self.b.apply(input)?;
        let gate = self.g_b.apply(&self.g_a.apply(input)?)?;
        let lower_bound = config.linear_lower_bound as f32;
        let query_scale = inverse_sqrt(head_dim);
        let mut out = vec![0.0f32; width];
        let mut query = vec![0.0f32; head_dim];
        let mut key = vec![0.0f32; head_dim];
        for (head, memory) in state.recurrent.iter_mut().enumerate() {
            let channels = head * head_dim..(head + 1) * head_dim;
            linear_attention::l2_normalise(
                &convolved[0][channels.clone()],
                query_scale,
                &mut query,
            );
            linear_attention::l2_normalise(&convolved[1][channels.clone()], 1.0, &mut key);
            let rate = self.a_log[head].exp();
            let decay: Vec<f32> = channels
                .clone()
                .map(|channel| {
                    linear_attention::kda_channel_decay(
                        lower_bound,
                        rate,
                        forget[channel],
                        self.dt_bias[channel],
                        linalg::sigmoid,
                    )
                })
                .collect();
            linear_attention::delta_rule_step(
                memory,
                Decay::Channel(&decay),
                &key,
                &convolved[2][channels.clone()],
                linalg::sigmoid(write[head]),
                &query,
                &mut out[channels],
            );
        }
        let normed: Vec<f32> = out
            .chunks(head_dim)
            .zip(gate.chunks(head_dim))
            .flat_map(|(head, gate)| {
                linear_attention::rms_norm_gated(
                    head,
                    &self.o_norm,
                    gate,
                    config.rms_norm_eps as f32,
                    linalg::sigmoid,
                )
            })
            .collect();
        self.o.apply(&normed)
    }
}

impl Indexer {
    /// Caches this token's indexer key, pools a block it completes, and selects the tokens the
    /// query at `position` attends over.
    fn select(
        &self,
        config: &Glm5NextConfig,
        state: &mut DsaState,
        input: &[f32],
        query_residual: &[f32],
        position: usize,
    ) -> Result<Vec<usize>, EngineError> {
        let index_dim = config.index_head_dim;
        let pool = config.index_kpool;
        let mut key = self.wk.apply(input)?;
        linalg::layer_norm(
            &mut key,
            &self.k_norm_weight,
            &self.k_norm_bias,
            index_dim,
            INDEX_KEY_NORM_EPS,
        )?;
        state.index_keys.push(key);
        state.index_gates.push(self.gate.apply(input)?);
        if (position + 1).is_multiple_of(pool) {
            // A pooled key is a per-channel softmax over the pool's tokens of gate plus position.
            let first = position + 1 - pool;
            let mut pooled = vec![0.0f32; index_dim];
            let mut weights = vec![0.0f32; pool];
            for (channel, slot) in pooled.iter_mut().enumerate() {
                for (offset, weight) in weights.iter_mut().enumerate() {
                    *weight = state.index_gates[first + offset][channel]
                        + self.ape[offset * index_dim + channel];
                }
                linalg::softmax(&mut weights);
                *slot = weights
                    .iter()
                    .enumerate()
                    .map(|(offset, weight)| weight * state.index_keys[first + offset][channel])
                    .sum();
            }
            state.pool_keys.push(pooled);
        }
        let query = self.wq_b.apply(query_residual)?;
        let head_weight = inverse_sqrt(config.index_n_heads);
        let head_weights: Vec<f32> = self
            .weights_proj
            .apply(input)?
            .into_iter()
            .map(|weight| weight * head_weight)
            .collect();
        let score_scale = inverse_sqrt(index_dim);
        let scores: Vec<f32> = state
            .pool_keys
            .iter()
            .map(|pooled| {
                head_weights
                    .iter()
                    .zip(query.chunks(index_dim))
                    .map(|(weight, head_query)| {
                        let dot: f32 = head_query.iter().zip(pooled).map(|(q, k)| q * k).sum();
                        weight * (dot * score_scale).max(0.0)
                    })
                    .sum()
            })
            .collect();
        let mut selected = Vec::new();
        topk::top_pools(
            &scores,
            pool,
            config.index_pools(),
            position,
            config.index_kpool_always_select_tail,
            &mut selected,
        );
        Ok(selected)
    }
}

impl Dsa {
    /// One token through the layer. Returns the branch output and the selection it attended over,
    /// which the next layer reuses when it shares this layer's indexer.
    fn step(
        &self,
        config: &Glm5NextConfig,
        state: &mut DsaState,
        input: &[f32],
        position: usize,
        shared: Option<Vec<usize>>,
    ) -> Result<(Vec<f32>, Vec<usize>), EngineError> {
        let eps = config.rms_norm_eps as f32;
        let (nope, value_dim) = (config.qk_nope_head_dim, config.v_head_dim);
        // This rank's heads.
        let heads = self.q_b.rows / nope;
        let mut query_residual = self.q_a.apply(input)?;
        linalg::rms_norm(&mut query_residual, &self.q_a_norm, config.q_lora_rank, eps)?;
        let query = self.q_b.apply(&query_residual)?;
        let mut latent = self.kv_a.apply(input)?;
        linalg::rms_norm(&mut latent, &self.kv_a_norm, config.kv_lora_rank, eps)?;
        let expanded = self.kv_b.apply(&latent)?;
        let mut keys = Vec::with_capacity(heads * nope);
        let mut values = Vec::with_capacity(heads * value_dim);
        for head in expanded.chunks(nope + value_dim) {
            keys.extend_from_slice(&head[..nope]);
            values.extend_from_slice(&head[nope..]);
        }
        state.keys.push(keys);
        state.values.push(values);
        let selected = match (&self.indexer, shared) {
            (Some(indexer), _) => {
                indexer.select(config, state, input, &query_residual, position)?
            }
            (None, Some(selected)) => selected,
            (None, None) => {
                return Err(shape_error(
                    "dsa",
                    "a layer that shares an indexer has no selection from the layer before it",
                ));
            }
        };
        let scaling = inverse_sqrt(nope);
        let mut out = vec![0.0f32; heads * value_dim];
        let mut logits = vec![0.0f32; selected.len()];
        for head in 0..heads {
            let head_query = &query[head * nope..(head + 1) * nope];
            for (logit, token) in logits.iter_mut().zip(&selected) {
                let key = &state.keys[*token][head * nope..(head + 1) * nope];
                let dot: f32 = head_query.iter().zip(key).map(|(q, k)| q * k).sum();
                *logit = dot * scaling;
            }
            linalg::softmax(&mut logits);
            let head_out = &mut out[head * value_dim..(head + 1) * value_dim];
            for (weight, token) in logits.iter().zip(&selected) {
                let value = &state.values[*token][head * value_dim..(head + 1) * value_dim];
                for (slot, value) in head_out.iter_mut().zip(value) {
                    *slot += weight * value;
                }
            }
        }
        Ok((self.o.apply(&out)?, selected))
    }
}

impl Mlp {
    /// `down(silu(min(gate x, limit)) * clamp(up x, -limit, limit))`.
    fn apply(&self, input: &[f32], limit: f32) -> Result<Vec<f32>, EngineError> {
        let gate = self.gate.apply(input)?;
        let up = self.up.apply(input)?;
        let activated: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| linalg::silu(gate.min(limit)) * up.clamp(-limit, limit))
            .collect();
        self.down.apply(&activated)
    }
}

impl Moe {
    /// The routed experts in ascending expert order, each scaled by its weight, then the shared
    /// expert.
    fn apply(&self, config: &Glm5NextConfig, input: &[f32]) -> Result<Vec<f32>, EngineError> {
        let limit = config.swiglu_limit as f32;
        let scores: Vec<f32> = self
            .router
            .apply(input)?
            .into_iter()
            .map(linalg::sigmoid)
            .collect();
        let mut selected = Vec::with_capacity(config.num_experts_per_tok);
        topk::noaux_top_k(
            &scores,
            &self.bias,
            config.num_experts_per_tok,
            config.norm_topk_prob,
            config.routed_scaling_factor as f32,
            &mut selected,
        );
        selected.sort_unstable_by_key(|(expert, _)| *expert);
        let mut out = vec![0.0f32; config.hidden_size];
        for (expert, weight) in selected {
            let expert_out = self.experts[expert].apply(input, limit)?;
            for (slot, value) in out.iter_mut().zip(expert_out) {
                *slot += value * weight;
            }
        }
        for (slot, value) in out.iter_mut().zip(self.shared.apply(input, limit)?) {
            *slot += value;
        }
        Ok(out)
    }
}

/// One sublayer around the mHC streams: mix, norm, branch, reduce the branch's partial sum across
/// the ranks, and mix the output back.
fn hyper_sublayer(
    geometry: HcGeometry,
    tp: &TpRuntime,
    hyper: &Hyper,
    norm: &[f32],
    eps: f32,
    streams: &mut Vec<f32>,
    branch: impl FnOnce(&[f32]) -> Result<Vec<f32>, EngineError>,
) -> Result<(), EngineError> {
    let mut mixes = Mixes::for_config(geometry);
    mhc::hc_mixes(
        geometry,
        streams,
        &hyper.projection,
        &hyper.scale,
        &hyper.base,
        &mut mixes,
    )?;
    let mut collapsed = vec![0.0f32; geometry.dim];
    mhc::hc_pre(geometry, &mixes.pre, streams, &mut collapsed)?;
    linalg::rms_norm(&mut collapsed, norm, geometry.dim, eps)?;
    let mut out = branch(&collapsed)?;
    // Every branch ends in a column-split projection, the MoE's routed and shared experts already
    // summed, so this is the sublayer's one reduce; one rank's is a strict no-op.
    tp.all_reduce(&mut out)?;
    let mut mixed = vec![0.0f32; streams.len()];
    mhc::hc_post(
        geometry,
        &out,
        streams,
        &mixes.post,
        &mixes.comb,
        &mut mixed,
    )?;
    *streams = mixed;
    Ok(())
}

impl Glm5NextWeights {
    /// Load the text model from a checkpoint by its released names.
    ///
    /// `limit` is the largest tensor, in elements, the load will decode: the host reference is
    /// meant for small configurations, and [linalg::REFERENCE_TENSOR_LIMIT] refuses a released one.
    ///
    /// # Errors
    /// When a tensor is missing, stored in a shape the configuration does not give it, above the
    /// limit, or not decodable.
    pub fn load<S: WeightSource + ?Sized>(
        config: &Glm5NextConfig,
        source: &S,
        limit: usize,
    ) -> Result<Self, EngineError> {
        let whole = TpLayout {
            rank: 0,
            world_size: 1,
        };
        Self::load_tp(config, source, limit, whole)
    }

    /// Load rank `layout.rank`'s slice of the text model: every tensor the served plan splits keeps
    /// this rank's part, decoded and dequantised whole and then sliced.
    ///
    /// # Errors
    /// As [Glm5NextWeights::load], and when the layout is not a valid pair or the configuration
    /// does not split over its world ([Glm5NextConfig::check_world]).
    pub fn load_tp<S: WeightSource + ?Sized>(
        config: &Glm5NextConfig,
        source: &S,
        limit: usize,
        layout: TpLayout,
    ) -> Result<Self, EngineError> {
        layout.validate()?;
        config
            .check_world(layout.world_size)
            .map_err(|error| shape_error("TP world", error.to_string()))?;
        let loader = Loader {
            source,
            config,
            limit,
            plan: Glm5NextTensorPlan::served(config),
            layout,
        };
        let (vocab, hidden) = (config.vocab_size, config.hidden_size);
        Ok(Self {
            config: config.clone(),
            layout,
            embed: loader.tensor(
                &format!("{LM_PREFIX}.embed_tokens.weight"),
                &[vocab, hidden],
            )?,
            norm: loader.tensor(&format!("{LM_PREFIX}.norm.weight"), &[hidden])?,
            lm_head: loader.linear("lm_head", vocab, hidden)?,
            layers: (0..config.num_hidden_layers)
                .map(|layer| loader.layer(layer))
                .collect::<Result<_, _>>()?,
        })
    }

    /// A fresh sequence.
    #[must_use]
    pub fn new_state(&self) -> ReferenceState {
        let config = &self.config;
        let head_dim = config.linear_head_dim;
        ReferenceState {
            position: 0,
            layers: self
                .layers
                .iter()
                .map(|layer| match &layer.branch {
                    // One memory per head this rank holds.
                    Branch::Linear(kda) => LayerState::Linear(KdaState {
                        conv: Default::default(),
                        recurrent: vec![vec![0.0; head_dim * head_dim]; kda.a_log.len()],
                    }),
                    Branch::Sparse(_) => LayerState::Sparse(DsaState::default()),
                })
                .collect(),
        }
    }

    /// Run `tokens` through the model and return one row of logits per token.
    ///
    /// # Errors
    /// When the weights are one rank of a wider world, a token is outside the vocabulary, the chunk
    /// runs past the context window, the state belongs to another model, or an operation refuses
    /// its shapes.
    pub fn forward(
        &self,
        state: &mut ReferenceState,
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, EngineError> {
        let tp = TpRuntime::new(self.layout, Box::new(SingleRank))?;
        let gather = BitGatherHandle::new(Box::new(SingleRank));
        self.forward_tp(&tp, &gather, state, tokens)
    }

    /// Run `tokens` through this rank's slice and return one row of full-vocabulary logits per
    /// token, the same on every rank. Every rank of the world runs the same tokens in step: each
    /// sublayer reduces through `tp`, and each token's logits gather through `gather`.
    ///
    /// # Errors
    /// As [Glm5NextWeights::forward], and when `tp` or `gather` is not this rank of this world or a
    /// collective fails.
    pub fn forward_tp(
        &self,
        tp: &TpRuntime,
        gather: &BitGatherHandle,
        state: &mut ReferenceState,
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, EngineError> {
        let layout = self.layout;
        for (what, rank, world) in [
            ("TP collective", tp.rank(), tp.world()),
            ("bit gather", gather.rank(), gather.world()),
        ] {
            if (rank, world) != (layout.rank, layout.world_size) {
                return Err(shape_error(
                    "forward",
                    format!(
                        "the {what} is rank {rank} of {world}, the weights rank {} of {}",
                        layout.rank, layout.world_size
                    ),
                ));
            }
        }
        let config = &self.config;
        let hidden = config.hidden_size;
        if state.position + tokens.len() > config.max_position_embeddings {
            return Err(shape_error(
                "forward",
                format!(
                    "the chunk would end at {} past the context window {}",
                    state.position + tokens.len(),
                    config.max_position_embeddings
                ),
            ));
        }
        if let Some(token) = tokens
            .iter()
            .find(|token| **token as usize >= config.vocab_size)
        {
            return Err(shape_error(
                "forward",
                format!(
                    "token {token} is outside the vocabulary {}",
                    config.vocab_size
                ),
            ));
        }
        if state.layers.len() != self.layers.len() {
            return Err(shape_error(
                "forward",
                format!(
                    "a state for {} layers on a model of {}",
                    state.layers.len(),
                    self.layers.len()
                ),
            ));
        }
        let geometry = HcGeometry::from(config);
        let eps = config.rms_norm_eps as f32;
        let mut logits = Vec::with_capacity(tokens.len());
        for &token in tokens {
            let position = state.position;
            let row = token as usize * hidden;
            let mut streams = self.embed[row..row + hidden].repeat(config.hc_mult);
            // The selection a sparse layer computed, for a next layer that shares its indexer.
            let mut selection: Option<Vec<usize>> = None;
            for (layer, layer_state) in self.layers.iter().zip(&mut state.layers) {
                hyper_sublayer(
                    geometry,
                    tp,
                    &layer.attn_hc,
                    &layer.input_norm,
                    eps,
                    &mut streams,
                    |input| match (&layer.branch, layer_state) {
                        (Branch::Linear(kda), LayerState::Linear(kda_state)) => {
                            selection = None;
                            kda.step(config, kda_state, input)
                        }
                        (Branch::Sparse(dsa), LayerState::Sparse(dsa_state)) => {
                            let (out, selected) =
                                dsa.step(config, dsa_state, input, position, selection.take())?;
                            selection = Some(selected);
                            Ok(out)
                        }
                        _ => Err(shape_error(
                            "forward",
                            "a layer state of the other attention kind",
                        )),
                    },
                )?;
                hyper_sublayer(
                    geometry,
                    tp,
                    &layer.ffn_hc,
                    &layer.post_norm,
                    eps,
                    &mut streams,
                    |input| match &layer.ffn {
                        Ffn::Dense(mlp) => mlp.apply(input, config.swiglu_limit as f32),
                        Ffn::Moe(moe) => moe.apply(config, input),
                    },
                )?;
            }
            let copies = config.hc_mult as f32;
            let mut head: Vec<f32> = (0..hidden)
                .map(|column| streams.iter().skip(column).step_by(hidden).sum::<f32>() / copies)
                .collect();
            linalg::rms_norm(&mut head, &self.norm, hidden, eps)?;
            // This rank's vocabulary rows, gathered bit for bit in rank order.
            let mut row = vec![0.0f32; config.vocab_size];
            gather.gather_bits(&self.lm_head.apply(&head)?, &mut row)?;
            logits.push(row);
            state.position += 1;
        }
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use sparkflash_weights::linalg::REFERENCE_TENSOR_LIMIT;
    use sparkflash_weights::safetensors::{INDEX_NAME, SafeTensorsSource};
    use sparkflash_weights::tp::{EXACT_WIRE_REL_L2_BAR, LocalBus, LocalRank, Wire};

    use super::*;
    use crate::checkpoint::write_synthetic_checkpoint;

    /// Fourteen tokens: seven complete pools of two, past the tiny budget of four.
    const TOKENS: [u32; 14] = [3, 17, 42, 5, 63, 0, 28, 11, 50, 7, 33, 21, 9, 60];

    /// The tiny checkpoint, open, with the directory that keeps it.
    fn tiny_source() -> (tempfile::TempDir, SafeTensorsSource) {
        let directory = tempfile::tempdir().expect("a temporary directory");
        write_synthetic_checkpoint(directory.path(), &Glm5NextConfig::tiny(), 7, 2)
            .expect("the fixture writes");
        let source = SafeTensorsSource::open(directory.path(), &directory.path().join(INDEX_NAME))
            .expect("the fixture opens");
        (directory, source)
    }

    fn tiny_weights() -> Glm5NextWeights {
        let (_directory, source) = tiny_source();
        Glm5NextWeights::load(&Glm5NextConfig::tiny(), &source, REFERENCE_TENSOR_LIMIT)
            .expect("the fixture loads")
    }

    /// Rank 0's logits for `tokens` and the seams it saw, each rank on its own thread over one
    /// recording bus on `wire`, once every rank's agree with rank 0's bit for bit.
    fn sharded(
        source: &SafeTensorsSource,
        world: usize,
        wire: Wire,
        tokens: &[u32],
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let config = Glm5NextConfig::tiny();
        let ranks: Vec<Glm5NextWeights> = (0..world)
            .map(|rank| {
                let layout = TpLayout {
                    rank,
                    world_size: world,
                };
                Glm5NextWeights::load_tp(&config, source, REFERENCE_TENSOR_LIMIT, layout)
                    .expect("the fixture splits")
            })
            .collect();
        let bus = Arc::new(LocalBus::recording(world).with_wire(wire));
        let logits: Vec<Vec<Vec<f32>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = ranks
                .iter()
                .enumerate()
                .map(|(rank, weights)| {
                    let adapter = LocalRank {
                        rank,
                        bus: Arc::clone(&bus),
                    };
                    scope.spawn(move || {
                        let tp = TpRuntime::new(weights.layout, Box::new(adapter.clone()))
                            .expect("the adapter is this rank");
                        let gather = BitGatherHandle::new(Box::new(adapter));
                        let mut state = weights.new_state();
                        weights
                            .forward_tp(&tp, &gather, &mut state, tokens)
                            .expect("the rank runs")
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("a rank does not panic"))
                .collect()
        });
        let seams = bus.seams(0).expect("the bus records");
        for rank in 1..world {
            assert_eq!(
                bits(&logits[rank]),
                bits(&logits[0]),
                "rank {rank} of {world}"
            );
            assert_eq!(
                bits(&bus.seams(rank).expect("the bus records")),
                bits(&seams),
                "rank {rank} of {world}'s seams"
            );
        }
        (logits.into_iter().next().expect("rank 0"), seams)
    }

    fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
        rows.iter()
            .map(|row| row.iter().map(|value| value.to_bits()).collect())
            .collect()
    }

    fn sparse(layer: &Layer) -> &Dsa {
        match &layer.branch {
            Branch::Sparse(dsa) => dsa,
            Branch::Linear(_) => panic!("a KDA layer where a DSA one was expected"),
        }
    }

    #[test]
    fn the_tiny_forward_is_finite_deterministic_and_chunk_invariant() {
        let weights = tiny_weights();
        let mut state = weights.new_state();
        let logits = weights.forward(&mut state, &TOKENS).unwrap();
        assert_eq!(logits.len(), TOKENS.len());
        for row in &logits {
            assert_eq!(row.len(), weights.config.vocab_size);
            assert!(row.iter().all(|value| value.is_finite()), "{row:?}");
        }
        // Different tokens score differently: nothing collapsed to a constant.
        assert_ne!(logits[0], logits[1]);
        let mut again = weights.new_state();
        assert_eq!(weights.forward(&mut again, &TOKENS).unwrap(), logits);
        assert_eq!(again, state);
        let mut split = weights.new_state();
        let mut chunked = weights.forward(&mut split, &TOKENS[..5]).unwrap();
        chunked.extend(weights.forward(&mut split, &TOKENS[5..]).unwrap());
        assert_eq!(chunked, logits);
        assert_eq!(split.position, TOKENS.len());
    }

    #[test]
    fn the_loader_keeps_fp8_tiles_where_the_release_stores_them() {
        let weights = tiny_weights();
        let tile = weights.config.fp8_block;
        let Ffn::Dense(dense) = &weights.layers[0].ffn else {
            panic!("layer 0 of the tiny config is dense");
        };
        assert_eq!(dense.gate.act_tile, tile);
        let dsa = sparse(&weights.layers[1]);
        assert_eq!(dsa.q_a.act_tile, tile);
        assert_eq!(dsa.o.act_tile, tile);
        assert_eq!(dsa.kv_b.act_tile, None);
        assert!(dsa.indexer.is_some());
        assert!(sparse(&weights.layers[2]).indexer.is_none());
        let Ffn::Moe(moe) = &weights.layers[1].ffn else {
            panic!("layer 1 of the tiny config is MoE");
        };
        assert_eq!(moe.router.act_tile, None);
        assert_eq!(moe.experts[0].down.act_tile, tile);
        assert_eq!(weights.lm_head.act_tile, None);
    }

    #[test]
    fn the_indexer_keeps_its_budget_and_the_tail_and_the_next_layer_shares_it() {
        let weights = tiny_weights();
        let config = &weights.config;
        let mut state = weights.new_state();
        weights.forward(&mut state, &TOKENS).unwrap();
        let LayerState::Sparse(full) = &state.layers[1] else {
            panic!("layer 1 is sparse");
        };
        assert_eq!(full.index_keys.len(), TOKENS.len());
        assert_eq!(full.pool_keys.len(), TOKENS.len() / config.index_kpool);
        let LayerState::Sparse(shared) = &state.layers[2] else {
            panic!("layer 2 is sparse");
        };
        assert!(shared.index_keys.is_empty() && shared.pool_keys.is_empty());
        assert_eq!(shared.keys.len(), TOKENS.len());

        // One more token at position 14: seven complete pools, four kept, and the open tail.
        let position = TOKENS.len();
        let input: Vec<f32> = (0..config.hidden_size)
            .map(|index| (index as f32 * 0.37).sin())
            .collect();
        let mut full = full.clone();
        let (_, selected) = sparse(&weights.layers[1])
            .step(config, &mut full, &input, position, None)
            .unwrap();
        assert_eq!(
            selected.len(),
            config.index_pools() * config.index_kpool + 1,
            "{selected:?}"
        );
        assert_eq!(selected.last(), Some(&position));
        assert!(
            selected.windows(2).all(|pair| pair[0] < pair[1]),
            "{selected:?}"
        );
        let mut shared = shared.clone();
        let (_, reused) = sparse(&weights.layers[2])
            .step(
                config,
                &mut shared,
                &input,
                position,
                Some(selected.clone()),
            )
            .unwrap();
        assert_eq!(reused, selected);
        let error = sparse(&weights.layers[2])
            .step(config, &mut shared, &input, position + 1, None)
            .unwrap_err();
        assert!(error.to_string().contains("shares an indexer"), "{error}");
    }

    #[test]
    fn an_fp8_linear_multiplies_the_quantised_input() {
        let tile = 8;
        let mut identity = vec![0.0f32; tile * tile];
        for index in 0..tile {
            identity[index * tile + index] = 1.0;
        }
        let input: Vec<f32> = (0..tile).map(|index| 0.1 + index as f32 * 0.013).collect();
        let mut linear = Linear {
            rows: tile,
            columns: tile,
            weight: identity,
            act_tile: None,
        };
        assert_eq!(linear.apply(&input).unwrap(), input);
        linear.act_tile = Some(tile);
        let (codes, scales) = quant::quantize_activation_fp8_tiles(&input, tile).unwrap();
        let want: Vec<f32> = codes
            .iter()
            .map(|code| quant::f8e4m3_to_f32(*code) * scales[0])
            .collect();
        let got = linear.apply(&input).unwrap();
        assert_eq!(got, want);
        assert_ne!(
            got, input,
            "E4M3 has three mantissa bits; these inputs do not all fit"
        );
        assert!(linear.apply(&input[..tile - 1]).is_err());
    }

    #[test]
    fn the_forward_refuses_tokens_and_lengths_the_model_does_not_have() {
        let weights = tiny_weights();
        let config = &weights.config;
        let mut state = weights.new_state();
        let error = weights
            .forward(&mut state, &[config.vocab_size as u32])
            .unwrap_err();
        assert!(error.to_string().contains("vocabulary"), "{error}");
        let too_long = vec![1u32; config.max_position_embeddings + 1];
        let error = weights.forward(&mut state, &too_long).unwrap_err();
        assert!(error.to_string().contains("context window"), "{error}");
        assert_eq!(state.position, 0, "a refused chunk advances nothing");
    }

    /// The worst relative L2 over paired rows.
    fn worst(ours: &[Vec<f32>], theirs: &[Vec<f32>]) -> f64 {
        assert_eq!(ours.len(), theirs.len());
        ours.iter()
            .zip(theirs)
            .map(|(a, b)| linalg::relative_l2(a, b).unwrap())
            .fold(0.0, f64::max)
    }

    #[test]
    fn two_and_four_ranks_shard_the_one_rank_forward_over_an_exact_wire() {
        let config = Glm5NextConfig::tiny();
        let (_directory, source) = tiny_source();
        let whole = Glm5NextWeights::load(&config, &source, REFERENCE_TENSOR_LIMIT).unwrap();
        let reference = whole.forward(&mut whole.new_state(), &TOKENS).unwrap();
        let (one, one_seams) = sharded(&source, 1, Wire::F32, &TOKENS);
        assert_eq!(
            bits(&one),
            bits(&reference),
            "one rank is the plain forward"
        );
        // An attention and an MLP reduce per layer per token.
        assert_eq!(one_seams.len(), TOKENS.len() * 2 * config.num_hidden_layers);
        for world in [2, 4] {
            let (logits, seams) = sharded(&source, world, Wire::F32, &TOKENS);
            let seam = worst(&seams, &one_seams);
            let token = worst(&logits, &reference);
            println!("measured: world {world}: worst seam rel L2 {seam:e}, logits {token:e}");
            assert!(
                seam <= EXACT_WIRE_REL_L2_BAR,
                "world {world} seams: {seam:e}"
            );
            assert!(
                token <= EXACT_WIRE_REL_L2_BAR,
                "world {world} logits: {token:e}"
            );
            for (ours, theirs) in logits.iter().zip(&reference) {
                assert_eq!(
                    linalg::argmax(ours),
                    linalg::argmax(theirs),
                    "world {world}"
                );
            }
        }
    }

    /// The bf16 wire is the transport's, and the reference a device TP=4 run is held to: its ranks
    /// agree bit for bit and one rank is the plain forward. Its distance from the one-rank forward
    /// is the wire's rounding compounded through the fixture's random layers, so it is measured,
    /// not barred.
    #[test]
    fn the_bf16_wire_keeps_every_rank_bit_equal_and_one_rank_the_plain_forward() {
        let config = Glm5NextConfig::tiny();
        let (_directory, source) = tiny_source();
        let whole = Glm5NextWeights::load(&config, &source, REFERENCE_TENSOR_LIMIT).unwrap();
        let reference = whole.forward(&mut whole.new_state(), &TOKENS).unwrap();
        let (one, one_seams) = sharded(&source, 1, Wire::Bf16, &TOKENS);
        assert_eq!(
            bits(&one),
            bits(&reference),
            "one rank is the plain forward"
        );
        for world in [2, 4] {
            let (logits, seams) = sharded(&source, world, Wire::Bf16, &TOKENS);
            let flips = logits
                .iter()
                .zip(&reference)
                .filter(|(ours, theirs)| linalg::argmax(ours) != linalg::argmax(theirs))
                .count();
            println!(
                "measured: world {world}: first seam rel L2 {:e}, worst seam {:e}, worst logits \
                 {:e}, {flips} of {} argmax flips",
                linalg::relative_l2(&seams[0], &one_seams[0]).unwrap(),
                worst(&seams, &one_seams),
                worst(&logits, &reference),
                TOKENS.len()
            );
        }
    }

    #[test]
    fn a_rank_refuses_the_one_rank_forward_and_a_mismatched_adapter() {
        let config = Glm5NextConfig::tiny();
        let (_directory, source) = tiny_source();
        let layout = TpLayout {
            rank: 1,
            world_size: 4,
        };
        let rank = Glm5NextWeights::load_tp(&config, &source, REFERENCE_TENSOR_LIMIT, layout)
            .expect("the fixture splits");
        let error = rank.forward(&mut rank.new_state(), &TOKENS).unwrap_err();
        assert!(error.to_string().contains("world 1"), "{error}");
        let bus = Arc::new(LocalBus::new(4));
        let other = |rank| LocalRank {
            rank,
            bus: Arc::clone(&bus),
        };
        let tp = TpRuntime::new(layout, Box::new(other(1))).unwrap();
        let gather = BitGatherHandle::new(Box::new(other(2)));
        let error = rank
            .forward_tp(&tp, &gather, &mut rank.new_state(), &TOKENS)
            .unwrap_err();
        assert!(
            error.to_string().contains("bit gather is rank 2"),
            "{error}"
        );
        let three = TpLayout {
            rank: 0,
            world_size: 3,
        };
        assert!(Glm5NextWeights::load_tp(&config, &source, REFERENCE_TENSOR_LIMIT, three).is_err());
    }
}

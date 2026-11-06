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

//! The architecture a released Hugging Face `config.json` describes, as data.
//!
//! A model is a configuration and a list of [Part]s, not a Rust module. This parser reads the
//! released document - flat or nested under `text_config` - and turns its string fields
//! (`layer_types`, `mlp_layer_types`, `indexer_types`, `hidden_act`, `scoring_func`, ...)
//! into typed pieces. [Architecture::parts] then reports the pieces a run needs, so a checkpoint
//! whose pieces all exist is servable without new Rust.
//!
//! The parser is deliberately about *shape and selection*, not about a checkpoint's arithmetic.
//! Precision, layout and the kernel that runs a part belong to the shared profile and the typed
//! op plans; this document only says which parts a layer is made of.

use serde_json::Value;

use crate::config::{ConfigError, OneOrMany};
use crate::parts::{Part, PartSet, Target, built};

/// The architecture of one released checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct Architecture {
    /// The root `model_type` (`deepseek_v41`, `qwen4_exp`, `glm5_next`, ...).
    pub family: String,
    /// The root `architectures` list.
    pub architectures: Vec<String>,
    /// Residual-stream width.
    pub hidden_size: usize,
    /// Vocabulary rows.
    pub vocab_size: usize,
    /// The checkpoint's window.
    pub max_position_embeddings: usize,
    /// RMSNorm epsilon.
    pub norm_epsilon: f64,
    /// Whether the LM head shares the embedding table.
    pub embedding_tied_to_head: bool,
    /// Whether a projection in the attention path carries a bias.
    pub attention_bias: bool,
    /// The decoder layers, in forward order.
    pub layers: Vec<Layer>,
    /// The stored weight format, if the release declares one.
    pub quantization: Quantization,
    /// The vision tower, if the release carries one.
    pub vision: Option<Vision>,
    /// The MTP/DSpark stages, if the release carries any.
    pub mtp: Option<Mtp>,
}

/// One decoder layer.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    /// The layer's index in the checkpoint.
    pub index: usize,
    /// The attention the layer runs.
    pub attention: Attention,
    /// The feed-forward network the layer runs.
    pub ffn: Ffn,
    /// How the layer mixes its residual stream.
    pub residual: Residual,
    /// Lookups that land on this layer's input, such as Engram or the n-gram PLE.
    pub sides: Vec<Side>,
}

/// The attention arithmetic of one layer.
#[derive(Debug, Clone, PartialEq)]
pub struct Attention {
    /// The attention family.
    pub kind: AttentionKind,
    /// Query heads.
    pub heads: usize,
    /// The width of one query/key head.
    pub head_dim: usize,
    /// The sliding window, when the layer has one.
    pub window: Option<usize>,
    /// The rotary embedding.
    pub rope: Rope,
    /// CSA2 compression, when the layer owns or reads a compressed cache.
    pub compression: Option<Compression>,
    /// The sparse indexer, when the layer runs one.
    pub indexer: Option<Indexer>,
}

/// The attention family of a layer.
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionKind {
    /// Multi-head latent attention.
    Mla {
        /// The query LoRA rank, when the query path is low-rank.
        q_lora_rank: Option<usize>,
        /// The key/value latent width.
        kv_lora_rank: usize,
        /// The value head width.
        v_head_dim: usize,
        /// The grouped low-rank output, when the layer has one.
        groups: Option<OutputGroups>,
    },
    /// Grouped-query attention.
    Gqa {
        /// Key/value heads.
        kv_heads: usize,
        /// Whether the output is multiplied by a learned gate.
        output_gate: bool,
    },
    /// Gated DeltaNet.
    Gdn(Gdn),
    /// Kimi Delta Attention.
    Kda(Kda),
}

/// The grouped low-rank output of MLA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputGroups {
    /// Independent output groups.
    pub groups: usize,
    /// The per-group width.
    pub rank: usize,
}

/// Gated DeltaNet geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gdn {
    /// The causal convolution width.
    pub conv_kernel: usize,
    /// Key head count.
    pub key_heads: usize,
    /// Value head count.
    pub value_heads: usize,
    /// Key head width.
    pub key_dim: usize,
    /// Value head width.
    pub value_dim: usize,
    /// The output gate's activation.
    pub output_gate: Activation,
}

/// Kimi Delta Attention geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct Kda {
    /// The causal convolution width.
    pub conv_kernel: usize,
    /// Head count.
    pub heads: usize,
    /// Head width.
    pub head_dim: usize,
    /// The forget gate's lower bound.
    pub lower_bound: f64,
}

/// The rotary embedding a layer's attention uses.
#[derive(Debug, Clone, PartialEq)]
pub struct Rope {
    /// The base frequency.
    pub theta: f64,
    /// How many trailing dimensions rotate; zero for NoPE.
    pub rotated_dims: usize,
    /// The scaling, when the release declares one.
    pub scaling: Option<RopeScaling>,
    /// The mRoPE section split, when the release declares one.
    pub mrope_section: Option<[usize; 3]>,
}

/// A rotary scaling rule.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeScaling {
    /// The rule's name (`yarn`, `linear`, `default`).
    pub kind: String,
    /// The scale factor.
    pub factor: f64,
    /// The trained sequence length the rule is anchored to.
    pub original_seq_len: usize,
    /// The YaRN fast correction bound.
    pub beta_fast: f64,
    /// The YaRN slow correction bound.
    pub beta_slow: f64,
}

/// CSA2 compression on one layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compression {
    /// Tokens pooled into one compressed position.
    pub ratio: usize,
    /// Whether the layer owns the cache it compresses.
    pub owns_keys: bool,
    /// Whether the layer's selection is restricted by the HSI candidate pool.
    pub uses_candidate_pool: bool,
}

/// A sparse indexer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Indexer {
    /// Indexer query heads.
    pub heads: usize,
    /// Indexer head width.
    pub head_dim: usize,
    /// Entries kept per query.
    pub topk: usize,
    /// Whether the layer owns the keys it scores.
    pub owns_keys: bool,
    /// Whether the layer reuses another layer's selection instead of running its own.
    pub reuse: bool,
    /// Tokens pooled into one scored position.
    pub pool: usize,
    /// Whether the incomplete tail pool is always attended.
    pub always_select_tail: bool,
    /// The rectifier applied to the similarity scores.
    pub activation: Activation,
    /// The hierarchical candidate pool, when the indexer builds or reads one.
    pub candidate_pool: Option<CandidatePool>,
}

/// The HSI candidate pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidatePool {
    /// The layer that builds the pool.
    pub source_layer: usize,
    /// Blocks in the pool.
    pub blocks: usize,
    /// Tokens per block.
    pub block_size: usize,
}

/// The feed-forward network of one layer.
#[derive(Debug, Clone, PartialEq)]
pub enum Ffn {
    /// A dense SwiGLU.
    Dense(DenseMlp),
    /// Routed experts plus the shared expert.
    Moe(Moe),
}

/// A dense MLP.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseMlp {
    /// Intermediate width.
    pub intermediate: usize,
    /// The activation.
    pub activation: Activation,
    /// The SwiGLU clamp, when the release declares one.
    pub clamp: Option<f64>,
}

/// The MoE block.
#[derive(Debug, Clone, PartialEq)]
pub struct Moe {
    /// Routed experts.
    pub routed_experts: usize,
    /// Experts a token activates.
    pub active_experts: usize,
    /// One routed expert's intermediate width.
    pub intermediate: usize,
    /// Shared experts.
    pub shared_experts: usize,
    /// One shared expert's intermediate width.
    pub shared_intermediate: usize,
    /// Whether the shared expert is gated by a learned scalar.
    pub shared_gated: bool,
    /// The selection score.
    pub score: Score,
    /// The multiplier on the routed weights after normalisation.
    pub routed_scaling: f64,
    /// Whether the selected weights are renormalised.
    pub norm_topk: bool,
    /// Whether selection adds a learned correction bias.
    pub selection_bias: bool,
    /// The SwiGLU clamp, when the release declares one.
    pub clamp: Option<f64>,
    /// The activation.
    pub activation: Activation,
}

/// The routing score a MoE gate applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Score {
    /// Softmax over all experts.
    Softmax,
    /// Per-expert sigmoid.
    Sigmoid,
    /// `sqrt(softplus(z))`.
    SqrtSoftPlus,
}

/// An activation this tree can select by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// SiLU / swish.
    Silu,
    /// Exact GELU.
    Gelu,
    /// The tanh approximation of GELU.
    GeluTanh,
    /// ReLU.
    Relu,
    /// Sigmoid.
    Sigmoid,
}

/// How a layer mixes its residual stream.
#[derive(Debug, Clone, PartialEq)]
pub enum Residual {
    /// The ordinary additive residual.
    Plain,
    /// A widened residual stream with a data-dependent mix.
    Hyper(HyperConnection),
}

/// Hyper-connection geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperConnection {
    /// Residual streams.
    pub streams: usize,
    /// The input mixer's bottleneck rank, when it has one.
    pub lowrank: Option<usize>,
    /// Whether the mix is manifold-constrained.
    pub manifold: bool,
    /// Sinkhorn iterations used to normalise the mix.
    pub sinkhorn_iters: usize,
    /// The mix's epsilon.
    pub epsilon: f64,
}

/// A lookup that lands on a layer's input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Side {
    /// An Engram table.
    Engram(EngramGeometry),
    /// The n-gram PLE table.
    Ple(PleGeometry),
}

/// Engram geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngramGeometry {
    /// Rows in each table, one per Engram layer.
    pub tables: Vec<usize>,
    /// Longest n-gram the keys represent.
    pub max_ngram: usize,
    /// Hash heads.
    pub heads: usize,
    /// Hash head width.
    pub head_dim: usize,
    /// The declared vocabulary the keys are drawn from.
    pub vocab_size: usize,
    /// The compressed vocabulary.
    pub compressed_vocab: usize,
    /// The padding id the compressor skips.
    pub pad_id: u32,
}

/// n-gram PLE geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleGeometry {
    /// The embedding width.
    pub embed_dim: usize,
    /// The short-convolution width.
    pub conv_kernel: usize,
    /// Longest n-gram represented.
    pub ngram_size: usize,
    /// Hashed heads per n-gram order.
    pub heads_per_ngram: usize,
    /// Lower bound of every head's bucket count.
    pub vocab_base: u64,
    /// Seed of the hash multipliers.
    pub seed: u64,
    /// Checkpoint shards the table is stored in.
    pub split_parts: usize,
}

/// The MTP/DSpark stages a checkpoint carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mtp {
    /// Stages after the backbone.
    pub stages: usize,
    /// Tokens one stage drafts, when the release declares a block size.
    pub block_size: Option<usize>,
    /// Backbone layers whose hidden states feed the stages.
    pub target_layers: Vec<usize>,
    /// Routed experts per stage.
    pub routed_experts: Option<usize>,
    /// Experts a token activates per stage.
    pub active_experts: Option<usize>,
}

/// The stored weight format a release declares.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Quantization {
    /// The quantization method (`fp8`, `nvfp4`, ...).
    pub method: Option<String>,
    /// The block width of a weight scale.
    pub block: Option<usize>,
    /// The scale format (`ue8m0`, ...).
    pub scale: Option<String>,
    /// The activation scheme (`dynamic`, ...).
    pub activation: Option<String>,
    /// The expert weight format (`fp4`, ...).
    pub expert: Option<String>,
}

/// The vision tower's geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct Vision {
    /// Transformer blocks.
    pub depth: usize,
    /// Block width.
    pub hidden_size: usize,
    /// Attention heads.
    pub heads: usize,
    /// MLP width.
    pub intermediate: usize,
    /// Patch side.
    pub patch_size: usize,
    /// Spatial merge factor before the aligner.
    pub spatial_merge: usize,
    /// The width the aligner hands the language model.
    pub out_hidden_size: usize,
    /// Axial RoPE base.
    pub rope_theta: f64,
}

impl Architecture {
    /// Parse a released Hugging Face `config.json`.
    ///
    /// # Errors
    /// [ConfigError::Json] for a malformed document and [ConfigError::Invalid] for a document
    /// that does not describe a shape this parser can name.
    pub fn from_json(text: &str) -> Result<Self, ConfigError> {
        let root: Value =
            serde_json::from_str(text).map_err(|error| ConfigError::Json(error.to_string()))?;
        Self::from_value(&root)
    }

    /// Read a released Hugging Face `config.json` from a path.
    ///
    /// # Errors
    /// A read error as [ConfigError::Json], or as [Self::from_json].
    pub fn from_path(path: &std::path::Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| ConfigError::Json(format!("{}: {error}", path.display())))?;
        Self::from_json(&text)
    }

    /// Parse an already-decoded `config.json`.
    ///
    /// # Errors
    /// As [Self::from_json].
    pub fn from_value(root: &Value) -> Result<Self, ConfigError> {
        let text = object(root, "text_config").unwrap_or(root);
        let family = string(root, "model_type")
            .or_else(|| string(text, "model_type"))
            .ok_or_else(|| invalid("the configuration names no model_type"))?;
        let architectures = string_list(root.get("architectures"))?;
        let hidden_size = required_usize(text, "hidden_size", "text_config")?;
        let vocab_size = required_usize(text, "vocab_size", "text_config")?;
        let max_position_embeddings = optional_usize(text, "max_position_embeddings")?.unwrap_or(0);
        let norm_epsilon = optional_f64(text, "rms_norm_eps")?.unwrap_or(1e-6);
        let embedding_tied_to_head = optional_bool(text, "tie_word_embeddings")?.unwrap_or(false);
        let attention_bias = optional_bool(text, "attention_bias")?.unwrap_or(false);

        let layers = layers(text)?;
        let quantization = quantization(root)?;
        let vision = vision(root)?;
        let mtp = mtp(text, hidden_size)?;

        Ok(Self {
            family,
            architectures,
            hidden_size,
            vocab_size,
            max_position_embeddings,
            norm_epsilon,
            embedding_tied_to_head,
            attention_bias,
            layers,
            quantization,
            vision,
            mtp,
        })
    }

    /// The parts a run of this architecture needs.
    #[must_use]
    pub fn parts(&self) -> PartSet {
        let mut parts = PartSet::new();
        parts.insert(Part::Embedding);
        parts.insert(Part::RmsNorm);
        parts.insert(Part::LmHead);
        if self.vision.is_some() {
            parts.insert(Part::Vision);
        }
        for layer in &self.layers {
            match &layer.attention.kind {
                AttentionKind::Mla { .. } => {
                    parts.insert(Part::MlaAttention);
                }
                AttentionKind::Gqa { .. } => {
                    parts.insert(Part::GqaAttention);
                }
                AttentionKind::Gdn(_) => {
                    parts.insert(Part::GdnScan);
                }
                AttentionKind::Kda(_) => {
                    parts.insert(Part::KdaScan);
                }
            }
            if layer.attention.compression.is_some() {
                parts.insert(Part::Csa2Compression);
            }
            if layer.attention.indexer.is_some() {
                parts.insert(Part::SparseIndexer);
            }
            match &layer.ffn {
                Ffn::Dense(_) => {
                    parts.insert(Part::DenseMlp);
                }
                Ffn::Moe(_) => {
                    parts.insert(Part::Moe);
                }
            }
            if let Residual::Hyper(hyper) = &layer.residual {
                parts.insert(if hyper.manifold {
                    Part::ManifoldHyperConnection
                } else {
                    Part::HyperConnection
                });
            }
            for side in &layer.sides {
                parts.insert(match side {
                    Side::Engram(_) => Part::Engram,
                    Side::Ple(_) => Part::Ple,
                });
            }
        }
        if self.mtp.is_some() {
            parts.insert(Part::Mtp);
        }
        parts
    }

    /// The parts this architecture needs that `available` does not hold.
    #[must_use]
    pub fn missing(&self, available: &PartSet) -> Vec<Part> {
        self.parts().missing_from(available)
    }

    /// The parts this architecture needs that this build does not carry for `target`.
    #[must_use]
    pub fn missing_on(&self, target: Target) -> Vec<Part> {
        self.missing(&built(target))
    }

    /// The layer indexes whose attention runs a sparse indexer.
    #[must_use]
    pub fn indexer_layers(&self) -> Vec<usize> {
        self.layers
            .iter()
            .filter(|layer| layer.attention.indexer.is_some())
            .map(|layer| layer.index)
            .collect()
    }
}

fn layers(text: &Value) -> Result<Vec<Layer>, ConfigError> {
    let count = required_usize(text, "num_hidden_layers", "text_config")?;
    let layer_types = string_list(text.get("layer_types"))?;
    let ratios = usize_list(text.get("compress_ratios"))?;
    let indexer_types = string_list(text.get("indexer_types"))?;
    if layer_types.is_empty() && ratios.is_empty() {
        return Err(invalid(
            "the configuration has neither layer_types nor compress_ratios, so its decoder schedule is unknown",
        ));
    }
    let kv_sources = usize_list(text.get("kv_source_layer_ids"))?;
    let index_sources = usize_list(text.get("index_source_layer_ids"))?;
    let candidate_source = optional_usize(text, "candidate_source_layer_id")?;
    let candidate_blocks = optional_usize(text, "candidate_topk_blocks")?.unwrap_or(0);
    let candidate_block_size = optional_usize(text, "candidate_block_size")?.unwrap_or(0);
    let candidate_pool = candidate_source.map(|source_layer| CandidatePool {
        source_layer,
        blocks: candidate_blocks,
        block_size: candidate_block_size,
    });
    let first_dense = optional_usize(text, "first_k_dense_replace")?.unwrap_or(0);
    let mlp_types = string_list(text.get("mlp_layer_types"))?;
    let engram_layers = usize_list(text.get("engram_layer_ids"))?;
    let ple_layers = usize_list(text.get("ple_layer_ids"))?;
    let engram_geometry = engram_geometry(text)?;
    let ple_geometry = ple_geometry(text)?;
    let schedule = Schedule {
        kv_sources: &kv_sources,
        index_sources: &index_sources,
        indexer_types: &indexer_types,
        candidate_pool,
    };

    let mut layers = Vec::with_capacity(count);
    for index in 0..count {
        let kind = layer_types
            .get(index)
            .map_or("compressed_attention", String::as_str);
        let ratio = ratios.get(index).copied().unwrap_or(0);
        let attention = attention(text, index, kind, ratio, &schedule)?;
        let ffn = ffn(text, index, first_dense, &mlp_types)?;
        let residual = residual(text)?;
        let mut sides = Vec::new();
        if engram_layers.contains(&index) {
            sides.push(Side::Engram(engram_geometry.clone()));
        }
        if ple_layers
            .iter()
            .any(|layer| layer.saturating_sub(1) == index)
        {
            sides.push(Side::Ple(ple_geometry.clone()));
        }
        layers.push(Layer {
            index,
            attention,
            ffn,
            residual,
            sides,
        });
    }
    Ok(layers)
}

/// The per-layer selection lists a released document uses.
struct Schedule<'a> {
    kv_sources: &'a [usize],
    index_sources: &'a [usize],
    indexer_types: &'a [String],
    candidate_pool: Option<CandidatePool>,
}

fn attention(
    text: &Value,
    index: usize,
    layer_type: &str,
    ratio: usize,
    schedule: &Schedule<'_>,
) -> Result<Attention, ConfigError> {
    let heads = required_usize(text, "num_attention_heads", "text_config")?;
    let declared_head_dim = optional_usize(text, "head_dim")?.unwrap_or(0);
    let nope = optional_usize(text, "qk_nope_head_dim")?.unwrap_or(0);
    let rope_dim = optional_usize(text, "qk_rope_head_dim")?.unwrap_or(0);
    let head_dim = if declared_head_dim > 0 {
        declared_head_dim
    } else {
        nope + rope_dim
    };
    if head_dim == 0 {
        return Err(invalid(format!(
            "layer {index} has no head_dim and no qk_nope_head_dim"
        )));
    }
    let window = optional_usize(text, "sliding_window")?;
    let partial = optional_f64(text, "partial_rotary_factor")?.unwrap_or(0.0);
    let rotated_dims = if rope_dim > 0 {
        rope_dim
    } else if partial > 0.0 {
        ((head_dim as f64 * partial).round() as usize).min(head_dim)
    } else {
        0
    };
    let rope = rope(text, ratio, rotated_dims)?;

    let is_linear = layer_type.contains("linear");
    let is_sparse = layer_type.contains("sparse")
        || layer_type.contains("attention")
        || layer_type == "compressed_attention";
    let is_mla = optional_usize(text, "kv_lora_rank")?.is_some()
        || optional_usize(text, "qk_nope_head_dim")?.is_some()
        || optional_usize(text, "o_groups")?.is_some()
        || !usize_list(text.get("compress_ratios"))?.is_empty();

    let kind = if is_linear {
        if object(text, "linear_attn_config").is_some() {
            let linear = object(text, "linear_attn_config")
                .ok_or_else(|| invalid("linear_attn_config is not an object"))?;
            AttentionKind::Kda(Kda {
                conv_kernel: optional_usize(linear, "short_conv_kernel_size")?.unwrap_or(4),
                heads: optional_usize(linear, "num_heads")?.unwrap_or(0),
                head_dim: optional_usize(linear, "head_dim")?.unwrap_or(0),
                lower_bound: optional_f64(linear, "gate_lower_bound")?.unwrap_or(-5.0),
            })
        } else {
            AttentionKind::Gdn(Gdn {
                conv_kernel: optional_usize(text, "linear_conv_kernel_dim")?.unwrap_or(4),
                key_heads: optional_usize(text, "linear_num_key_heads")?.unwrap_or(0),
                value_heads: optional_usize(text, "linear_num_value_heads")?.unwrap_or(0),
                key_dim: optional_usize(text, "linear_key_head_dim")?.unwrap_or(0),
                value_dim: optional_usize(text, "linear_value_head_dim")?.unwrap_or(0),
                output_gate: activation(
                    optional_string(text, "output_gate_type")
                        .as_deref()
                        .unwrap_or("silu"),
                    "output_gate_type",
                )?,
            })
        }
    } else if is_mla {
        AttentionKind::Mla {
            q_lora_rank: optional_usize(text, "q_lora_rank")?,
            kv_lora_rank: optional_usize(text, "kv_lora_rank")?.unwrap_or(head_dim),
            v_head_dim: optional_usize(text, "v_head_dim")?.unwrap_or(head_dim),
            groups: match (
                optional_usize(text, "o_groups")?,
                optional_usize(text, "o_lora_rank")?,
            ) {
                (Some(groups), Some(rank)) => Some(OutputGroups { groups, rank }),
                _ => None,
            },
        }
    } else if is_sparse {
        AttentionKind::Gqa {
            kv_heads: optional_usize(text, "num_key_value_heads")?.unwrap_or(heads),
            output_gate: true,
        }
    } else {
        return Err(invalid(format!(
            "layer {index} has unknown layer_type {layer_type:?}"
        )));
    };

    let compression = (ratio > 0).then_some(Compression {
        ratio,
        owns_keys: true,
        uses_candidate_pool: schedule
            .candidate_pool
            .is_some_and(|pool| index >= pool.source_layer),
    });
    let indexer = indexer(text, index, &kind, ratio, schedule)?;
    Ok(Attention {
        kind,
        heads,
        head_dim,
        window,
        rope,
        compression,
        indexer,
    })
}

fn indexer(
    text: &Value,
    index: usize,
    kind: &AttentionKind,
    ratio: usize,
    schedule: &Schedule<'_>,
) -> Result<Option<Indexer>, ConfigError> {
    if matches!(kind, AttentionKind::Gdn(_) | AttentionKind::Kda(_)) {
        return Ok(None);
    }
    let configured = schedule
        .indexer_types
        .get(index)
        .map_or("full", String::as_str);
    let runs_here = if schedule.index_sources.is_empty() {
        // The GLM and Qwen releases select indexer layers by layer type and by the absence of a
        // shared-indexer marker.
        matches!(kind, AttentionKind::Gqa { .. }) || configured == "full"
    } else {
        schedule.index_sources.contains(&index)
    };
    if !runs_here || configured == "shared" {
        return Ok(None);
    }
    Ok(Some(Indexer {
        heads: optional_usize(text, "index_n_heads")?
            .or(optional_usize(text, "indexer_n_heads")?)
            .unwrap_or(0),
        head_dim: optional_usize(text, "index_head_dim")?
            .or(optional_usize(text, "indexer_head_dim")?)
            .unwrap_or(0),
        topk: optional_usize(text, "index_topk")?
            .or(optional_usize(text, "indexer_budget")?)
            .unwrap_or(0),
        owns_keys: schedule.kv_sources.is_empty() || schedule.kv_sources.contains(&index),
        reuse: false,
        pool: optional_usize(text, "index_kpool")?
            .or(optional_usize(text, "indexer_compress_ratio")?)
            .unwrap_or(1)
            .max(1),
        always_select_tail: optional_bool(text, "index_kpool_always_select_tail")?.unwrap_or(false),
        activation: Activation::Relu,
        candidate_pool: schedule
            .candidate_pool
            .filter(|pool| ratio > 0 && index >= pool.source_layer),
    }))
}

fn ffn(
    text: &Value,
    index: usize,
    first_dense: usize,
    mlp_types: &[String],
) -> Result<Ffn, ConfigError> {
    let activation = activation(
        optional_string(text, "hidden_act")
            .as_deref()
            .unwrap_or("silu"),
        "hidden_act",
    )?;
    let clamp = optional_f64(text, "swiglu_limit")?.filter(|limit| *limit > 0.0);
    let dense = || -> Result<DenseMlp, ConfigError> {
        Ok(DenseMlp {
            intermediate: required_usize(text, "intermediate_size", "text_config")?,
            activation,
            clamp,
        })
    };
    let routed = optional_usize(text, "n_routed_experts")?.or(optional_usize(text, "num_experts")?);
    let selected_by_name = mlp_types.get(index).map(String::as_str);
    let is_moe = match selected_by_name {
        Some("dense") => false,
        Some("sparse" | "moe") => true,
        Some(other) => {
            return Err(invalid(format!(
                "layer {index} has unknown mlp_layer_type {other:?}"
            )));
        }
        None => routed.is_some() && index >= first_dense,
    };
    if !is_moe {
        return Ok(Ffn::Dense(dense()?));
    }
    let routed_experts = routed.ok_or_else(|| {
        invalid(format!(
            "layer {index} is a MoE layer but the configuration names no routed-expert count"
        ))
    })?;
    let shared_experts = optional_usize(text, "n_shared_experts")?.unwrap_or_else(|| {
        // Qwen4-Exp names the shared expert by its width, not by a count.
        usize::from(
            optional_usize(text, "shared_expert_intermediate_size")
                .ok()
                .flatten()
                .is_some(),
        )
    });
    let shared_intermediate = optional_usize(text, "shared_expert_intermediate_size")?
        .or(optional_usize(text, "moe_intermediate_size")?)
        .ok_or_else(|| invalid(format!("layer {index} has no MoE intermediate width")))?;
    Ok(Ffn::Moe(Moe {
        routed_experts,
        active_experts: required_usize(text, "num_experts_per_tok", "text_config")?,
        intermediate: required_usize(text, "moe_intermediate_size", "text_config")?,
        shared_experts,
        shared_intermediate,
        // A release that gates its shared expert names no count; it names the expert by width.
        shared_gated: optional_usize(text, "n_shared_experts")?.is_none()
            && optional_usize(text, "shared_expert_intermediate_size")?.is_some(),
        score: score(optional_string(text, "scoring_func").as_deref())?,
        routed_scaling: optional_f64(text, "routed_scaling_factor")?.unwrap_or(1.0),
        norm_topk: optional_bool(text, "norm_topk_prob")?.unwrap_or(true),
        selection_bias: optional_string(text, "topk_method").as_deref() == Some("noaux_tc"),
        clamp,
        activation,
    }))
}

fn residual(text: &Value) -> Result<Residual, ConfigError> {
    let streams = optional_usize(text, "hc_mult")?
        .or(optional_usize(text, "hc_count")?)
        .unwrap_or(1);
    if streams <= 1 {
        return Ok(Residual::Plain);
    }
    Ok(Residual::Hyper(HyperConnection {
        streams,
        lowrank: optional_usize(text, "hc_lowrank")?,
        manifold: optional_bool(text, "mhc")?.unwrap_or(false),
        sinkhorn_iters: optional_usize(text, "hc_sinkhorn_iters")?.unwrap_or(0),
        epsilon: optional_f64(text, "hc_eps")?.unwrap_or(0.0),
    }))
}

fn rope(text: &Value, ratio: usize, rotated_dims: usize) -> Result<Rope, ConfigError> {
    let compress_theta = optional_f64(text, "compress_rope_theta")?;
    let theta = if ratio > 0 {
        compress_theta
            .or(optional_f64(text, "rope_theta")?)
            .unwrap_or(10_000.0)
    } else {
        optional_f64(text, "rope_theta")?.unwrap_or(10_000.0)
    };
    let parameters = object(text, "rope_parameters");
    let scaling_json = object(text, "rope_scaling").or(parameters);
    // DeepSeek's ratio-0 layers run plain RoPE; every other release's scaling is global.
    let scaling = if ratio == 0 && text.get("compress_ratios").is_some() {
        None
    } else {
        scaling_json
            .filter(|value| {
                optional_f64(value, "factor").ok().flatten().is_some()
                    || optional_usize(value, "original_max_position_embeddings")
                        .ok()
                        .flatten()
                        .is_some_and(|length| length > 0)
            })
            .map(|value| RopeScaling {
                kind: optional_string(value, "rope_type").unwrap_or_else(|| "default".to_owned()),
                factor: optional_f64(value, "factor").ok().flatten().unwrap_or(1.0),
                original_seq_len: optional_usize(value, "original_max_position_embeddings")
                    .ok()
                    .flatten()
                    .unwrap_or(0),
                beta_fast: optional_f64(value, "beta_fast")
                    .ok()
                    .flatten()
                    .unwrap_or(32.0),
                beta_slow: optional_f64(value, "beta_slow")
                    .ok()
                    .flatten()
                    .unwrap_or(1.0),
            })
    };
    let mrope_section = parameters
        .and_then(|value| usize_list(value.get("mrope_section")).ok())
        .and_then(|sections| <[usize; 3]>::try_from(sections.as_slice()).ok());
    Ok(Rope {
        theta,
        rotated_dims,
        scaling,
        mrope_section,
    })
}

fn engram_geometry(text: &Value) -> Result<EngramGeometry, ConfigError> {
    Ok(EngramGeometry {
        tables: usize_list(text.get("engram_num_embeddings"))?,
        max_ngram: optional_usize(text, "engram_max_ngram_size")?.unwrap_or(0),
        heads: optional_usize(text, "engram_n_heads")?.unwrap_or(0),
        head_dim: optional_usize(text, "engram_head_dim")?.unwrap_or(0),
        vocab_size: optional_usize(text, "engram_vocab_size")?.unwrap_or(0),
        compressed_vocab: optional_usize(text, "engram_compressed_vocab_size")?.unwrap_or(0),
        pad_id: optional_usize(text, "engram_pad_token_id")?.unwrap_or(0) as u32,
    })
}

fn ple_geometry(text: &Value) -> Result<PleGeometry, ConfigError> {
    Ok(PleGeometry {
        embed_dim: optional_usize(text, "ple_embed_dim")?.unwrap_or(0),
        conv_kernel: optional_usize(text, "ple_conv_kernel_size")?.unwrap_or(0),
        ngram_size: optional_usize(text, "ngram_size")?.unwrap_or(0),
        heads_per_ngram: optional_usize(text, "heads_per_ngram")?.unwrap_or(0),
        vocab_base: optional_usize(text, "ngram_vocab_size_base")?.unwrap_or(0) as u64,
        seed: optional_usize(text, "seed")?.unwrap_or(0) as u64,
        split_parts: optional_usize(text, "split_ngram_parts")?.unwrap_or(1),
    })
}

fn mtp(text: &Value, _hidden: usize) -> Result<Option<Mtp>, ConfigError> {
    let stages = optional_usize(text, "num_nextn_predict_layers")?
        .or(optional_usize(text, "mtp_num_hidden_layers")?)
        .or(optional_usize(text, "n_mtp_layers")?)
        .unwrap_or(0);
    if stages == 0 {
        return Ok(None);
    }
    Ok(Some(Mtp {
        stages,
        block_size: optional_usize(text, "dspark_block_size")?,
        target_layers: usize_list(text.get("dspark_target_layer_ids"))?,
        routed_experts: optional_usize(text, "dspark_n_routed_experts")?,
        active_experts: optional_usize(text, "dspark_num_experts_per_tok")?,
    }))
}

/// Whether a declared `quant_method` names the producer rather than the arithmetic.
///
/// ModelOpt writes its own name in `quant_method` (`"modelopt"`) and records the arithmetic
/// separately, in `quant_algo` and in the per-module `quantized_layers` table. Read as a format,
/// the producer name labels an NVFP4 release BF16.
#[must_use]
pub fn names_the_producer(method: &str) -> bool {
    matches!(
        method.to_ascii_lowercase().as_str(),
        "modelopt" | "nvidia-modelopt" | "ammo"
    )
}

/// Whether a ModelOpt module name belongs to the MTP stage rather than a decoder layer.
///
/// The released NVFP4 snapshot stores the backbone's experts as NVFP4 and the MTP stage's as
/// `FP8_PB_WO`, so the two must not be read as one format: [Quantization::expert] describes the
/// decoder layers' routed experts, and the MTP stage keeps its own.
fn is_mtp_module(name: &str) -> bool {
    name.starts_with("mtp.") || name.contains(".mtp.")
}

/// The routed experts' format as a ModelOpt `quantization_config` records it.
///
/// `quantized_layers` keys every quantized module to its own `quant_algo`, and the decoder
/// layers' routed experts are the entries under `.mlp.experts`. A release that omits that table
/// still names the same arithmetic in `config_groups`, as a bit width and a number type over a
/// group of targets.
fn modelopt_expert(config: &Value) -> Option<String> {
    if let Some(layers) = object(config, "quantized_layers").and_then(Value::as_object) {
        let mut expert: Option<String> = None;
        for (name, spec) in layers {
            if !name.contains(".mlp.experts") || is_mtp_module(name) {
                continue;
            }
            let Some(algo) = string(spec, "quant_algo") else {
                continue;
            };
            match &expert {
                // Experts that disagree are not one stored format, and the plan has no name for
                // that mix: report none rather than pick one of them.
                Some(first) if !first.eq_ignore_ascii_case(&algo) => return None,
                Some(_) => {}
                None => expert = Some(algo),
            }
        }
        if expert.is_some() {
            return expert;
        }
    }
    let groups = object(config, "config_groups")?.as_object()?;
    for group in groups.values() {
        let targeted = group
            .get("targets")
            .and_then(Value::as_array)
            .is_some_and(|targets| {
                targets.iter().any(|target| {
                    target
                        .as_str()
                        .is_some_and(|name| name.contains(".mlp.experts") && !is_mtp_module(name))
                })
            });
        if !targeted {
            continue;
        }
        let Some(weights) = object(group, "weights") else {
            continue;
        };
        if let Some(format) = group_format(weights) {
            return Some(format);
        }
    }
    None
}

/// The arithmetic a ModelOpt group's `weights` block describes, by width and number type.
fn group_format(weights: &Value) -> Option<String> {
    let bits = weights.get("num_bits").and_then(Value::as_u64)?;
    let kind = string(weights, "type")?;
    match (bits, kind.to_ascii_lowercase().as_str()) {
        (4, "float") => Some("NVFP4".to_owned()),
        (8, "float") => Some("FP8".to_owned()),
        _ => None,
    }
}

fn quantization(root: &Value) -> Result<Quantization, ConfigError> {
    let Some(config) = object(root, "quantization_config") else {
        return Ok(Quantization::default());
    };
    let block = usize_list(config.get("weight_block_size"))?
        .first()
        .copied();
    // Three releases spell the same facts differently: DeepSeek writes quant_method /
    // scale_fmt / expert_dtype, GLM writes quant_method / fmt, and Qwen nests the expert format
    // under expert_weight.quant_algo with a top-level quant_algo of MIXED_PRECISION. A ModelOpt
    // release writes neither the expert format nor a format in quant_method at all: the producer
    // goes in quant_method and the arithmetic in quant_algo / quantized_layers.
    let expert = optional_string(config, "expert_dtype")
        .or_else(|| object(config, "expert_weight").and_then(|value| string(value, "quant_algo")))
        .or_else(|| {
            object(config, "mtp_expert_weight").and_then(|value| string(value, "quant_algo"))
        })
        .or_else(|| modelopt_expert(config));
    // A producer name is not the arithmetic; ModelOpt records that in quant_algo.
    let method = match optional_string(config, "quant_method") {
        Some(name) if !names_the_producer(&name) => Some(name),
        _ => optional_string(config, "quant_algo"),
    };
    Ok(Quantization {
        method,
        block,
        scale: optional_string(config, "scale_fmt").or(optional_string(config, "fmt")),
        activation: optional_string(config, "activation_scheme"),
        expert,
    })
}

fn vision(root: &Value) -> Result<Option<Vision>, ConfigError> {
    let Some(config) = object(root, "vision_config") else {
        return Ok(None);
    };
    Ok(Some(Vision {
        depth: required_usize(config, "num_hidden_layers", "vision_config")
            .or_else(|_| required_usize(config, "depth", "vision_config"))?,
        hidden_size: required_usize(config, "hidden_size", "vision_config")?,
        heads: required_usize(config, "num_attention_heads", "vision_config")
            .or_else(|_| required_usize(config, "num_heads", "vision_config"))?,
        intermediate: required_usize(config, "intermediate_size", "vision_config")?,
        patch_size: optional_usize(config, "patch_size")?.unwrap_or(0),
        spatial_merge: optional_usize(config, "spatial_merge_size")?.unwrap_or(1),
        out_hidden_size: optional_usize(config, "out_hidden_size")?.unwrap_or(0),
        rope_theta: optional_f64(config, "rope_theta")?.unwrap_or(10_000.0),
    }))
}

fn score(name: Option<&str>) -> Result<Score, ConfigError> {
    match name {
        None | Some("softmax") => Ok(Score::Softmax),
        Some("sigmoid") => Ok(Score::Sigmoid),
        Some("sqrtsoftplus" | "sqrt_softplus") => Ok(Score::SqrtSoftPlus),
        Some(other) => Err(invalid(format!("unknown scoring_func {other:?}"))),
    }
}

fn activation(name: &str, field: &str) -> Result<Activation, ConfigError> {
    match name {
        "silu" | "swish" => Ok(Activation::Silu),
        "gelu" => Ok(Activation::Gelu),
        "gelu_pytorch_tanh" | "gelu_tanh" => Ok(Activation::GeluTanh),
        "relu" => Ok(Activation::Relu),
        "sigmoid" => Ok(Activation::Sigmoid),
        other => Err(invalid(format!("unknown {field} {other:?}"))),
    }
}

fn object<'a>(value: &'a Value, name: &str) -> Option<&'a Value> {
    value.get(name).filter(|value| value.is_object())
}

fn string(value: &Value, name: &str) -> Option<String> {
    value
        .get(name)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn optional_string(value: &Value, name: &str) -> Option<String> {
    string(value, name)
}

fn optional_usize(value: &Value, name: &str) -> Result<Option<usize>, ConfigError> {
    match value.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number
            .as_u64()
            .and_then(|number| usize::try_from(number).ok())
            .map(Some)
            .ok_or_else(|| invalid(format!("{name} is not a non-negative integer"))),
        Some(other) => Err(invalid(format!("{name} is {other}, not a number"))),
    }
}

fn required_usize(value: &Value, name: &str, context: &str) -> Result<usize, ConfigError> {
    optional_usize(value, name)?.ok_or_else(|| invalid(format!("{context} names no {name}")))
}

fn optional_f64(value: &Value, name: &str) -> Result<Option<f64>, ConfigError> {
    match value.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number
            .as_f64()
            .map(Some)
            .ok_or_else(|| invalid(format!("{name} is not a number"))),
        Some(other) => Err(invalid(format!("{name} is {other}, not a number"))),
    }
}

fn optional_bool(value: &Value, name: &str) -> Result<Option<bool>, ConfigError> {
    match value.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(flag)) => Ok(Some(*flag)),
        Some(other) => Err(invalid(format!("{name} is {other}, not a boolean"))),
    }
}

fn string_list(value: Option<&Value>) -> Result<Vec<String>, ConfigError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    match value {
        Value::String(item) => Ok(vec![item.clone()]),
        Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(ToOwned::to_owned)
                    .ok_or_else(|| invalid(format!("a string list holds {item}")))
            })
            .collect(),
        other => Err(invalid(format!("expected a string or list, found {other}"))),
    }
}

fn usize_list(value: Option<&Value>) -> Result<Vec<usize>, ConfigError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let one: OneOrMany<usize> = serde_json::from_value(value.clone())
        .map_err(|error| invalid(format!("expected a number or list: {error}")))?;
    Ok(one.into_vec())
}

fn invalid(message: impl Into<String>) -> ConfigError {
    ConfigError::Invalid(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEEPSEEK: &str =
        include_str!("../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json");
    const GLM: &str = include_str!("../../sparkflash-model-glm5-next/config/glm53_flash.hf.json");
    const QWEN: &str =
        include_str!("../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.hf.json");

    fn count<F: Fn(&Layer) -> bool>(architecture: &Architecture, predicate: F) -> usize {
        architecture
            .layers
            .iter()
            .filter(|layer| predicate(layer))
            .count()
    }

    #[test]
    fn a_modelopt_producer_name_does_not_hide_the_expert_format() {
        // The carried Qwen document is the released snapshot's own: the producer is in
        // quant_method, the arithmetic is MIXED_PRECISION, the backbone's 48 routed experts are
        // NVFP4, and the MTP stage's are FP8_PB_WO. Reading the producer as a format left the
        // plan reporting BF16 experts for an NVFP4 release; letting the MTP stage vote made the
        // two formats look like one contradiction.
        let architecture = Architecture::from_json(QWEN).expect("the released Qwen config");
        assert_eq!(
            architecture.quantization.method.as_deref(),
            Some("MIXED_PRECISION"),
            "the plan reads the arithmetic, not the producer's name"
        );
        assert_eq!(architecture.quantization.expert.as_deref(), Some("NVFP4"));
    }

    #[test]
    fn the_mtp_stage_keeps_its_own_expert_format() {
        // The same release with the MTP entry promoted to the only expert entry: the decoder
        // layers carry no routed experts there, so the plan must not report the MTP format.
        let mut document: Value = serde_json::from_str(QWEN).expect("the carried Qwen config");
        let mut layers = serde_json::Map::new();
        layers.insert(
            "mtp.layers.0.mlp.experts".to_owned(),
            serde_json::json!({"quant_algo": "FP8_PB_WO", "group_size": 128}),
        );
        document["quantization_config"]["quantized_layers"] = Value::Object(layers);
        document["quantization_config"]
            .as_object_mut()
            .expect("an object")
            .remove("config_groups");
        let architecture =
            Architecture::from_json(&document.to_string()).expect("the released Qwen config");
        assert_eq!(architecture.quantization.expert, None);
    }

    #[test]
    fn a_modelopt_group_still_names_the_experts_without_a_layer_table() {
        // The same release without quantized_layers: the group's width and number type say it.
        let mut document: Value = serde_json::from_str(QWEN).expect("the carried Qwen config");
        document["quantization_config"] = serde_json::json!({
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "config_groups": {"group_0": {
                "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                "targets": [
                    "model.language_model.layers.0.mlp.experts",
                    "model.language_model.layers.1.mlp.experts",
                ],
            }},
        });
        let architecture =
            Architecture::from_json(&document.to_string()).expect("the released Qwen config");
        assert_eq!(architecture.quantization.expert.as_deref(), Some("NVFP4"));
        // A group whose targets are dense projections is not an expert format.
        document["quantization_config"]["config_groups"]["group_0"]["targets"] =
            serde_json::json!(["model.language_model.layers.0.mlp.shared_expert"]);
        let architecture =
            Architecture::from_json(&document.to_string()).expect("the released Qwen config");
        assert_eq!(architecture.quantization.expert, None);
    }

    #[test]
    fn a_producer_name_is_not_a_format() {
        assert!(names_the_producer("modelopt"));
        assert!(names_the_producer("ModelOpt"));
        assert!(!names_the_producer("fp8"));
        assert!(!names_the_producer("nvfp4"));
        assert!(!names_the_producer(""));
    }

    #[test]
    fn deepseek_names_csa2_moe_engram_and_mtp() {
        let architecture = Architecture::from_json(DEEPSEEK).expect("DeepSeek config");
        assert_eq!(architecture.family, "deepseek_v41");
        assert_eq!(architecture.layers.len(), 40);
        assert_eq!(
            count(&architecture, |layer| layer.attention.compression.is_some()),
            38
        );
        assert_eq!(architecture.indexer_layers().len(), 8);
        assert_eq!(
            count(&architecture, |layer| matches!(layer.ffn, Ffn::Moe(_))),
            40
        );
        assert_eq!(
            count(&architecture, |layer| !layer.sides.is_empty()),
            2,
            "Engram lands on layers 1 and 14"
        );
        assert_eq!(architecture.mtp.as_ref().map(|mtp| mtp.stages), Some(3));
        let parts = architecture.parts();
        for part in [
            Part::MlaAttention,
            Part::Csa2Compression,
            Part::SparseIndexer,
            Part::Moe,
            Part::HyperConnection,
            Part::Engram,
            Part::Mtp,
        ] {
            assert!(parts.contains(part), "DeepSeek needs {part}");
        }
        assert!(!parts.contains(Part::GqaAttention));
        assert!(!parts.contains(Part::KdaScan));
    }

    #[test]
    fn glm_names_kda_mla_dense_and_manifold_connections() {
        let architecture = Architecture::from_json(GLM).expect("GLM config");
        assert_eq!(architecture.family, "glm5_next");
        assert_eq!(architecture.layers.len(), 45);
        assert_eq!(
            count(&architecture, |layer| matches!(
                layer.attention.kind,
                AttentionKind::Kda(_)
            )),
            34
        );
        assert_eq!(
            count(&architecture, |layer| matches!(
                layer.attention.kind,
                AttentionKind::Mla { .. }
            )),
            11
        );
        assert_eq!(
            count(&architecture, |layer| matches!(layer.ffn, Ffn::Dense(_))),
            3
        );
        assert_eq!(architecture.indexer_layers().len(), 11);
        let parts = architecture.parts();
        for part in [
            Part::KdaScan,
            Part::MlaAttention,
            Part::SparseIndexer,
            Part::DenseMlp,
            Part::Moe,
            Part::ManifoldHyperConnection,
            Part::Mtp,
        ] {
            assert!(parts.contains(part), "GLM needs {part}");
        }
        assert!(!parts.contains(Part::Csa2Compression));
    }

    #[test]
    fn qwen_names_gdn_sparse_gqa_moe_and_the_ngram_ple() {
        let architecture = Architecture::from_json(QWEN).expect("Qwen config");
        assert_eq!(architecture.family, "qwen4_exp");
        assert_eq!(architecture.layers.len(), 48);
        assert_eq!(
            count(&architecture, |layer| matches!(
                layer.attention.kind,
                AttentionKind::Gdn(_)
            )),
            36
        );
        assert_eq!(
            count(&architecture, |layer| matches!(
                layer.attention.kind,
                AttentionKind::Gqa { .. }
            )),
            12
        );
        assert_eq!(
            count(&architecture, |layer| matches!(layer.ffn, Ffn::Moe(_))),
            48
        );
        assert_eq!(
            count(&architecture, |layer| layer
                .sides
                .iter()
                .any(|side| matches!(side, Side::Ple(_)))),
            1
        );
        assert_eq!(architecture.mtp.as_ref().map(|mtp| mtp.stages), Some(1));
        let parts = architecture.parts();
        for part in [
            Part::GdnScan,
            Part::GqaAttention,
            Part::SparseIndexer,
            Part::Moe,
            Part::HyperConnection,
            Part::Ple,
            Part::Mtp,
        ] {
            assert!(parts.contains(part), "Qwen needs {part}");
        }
        assert!(!parts.contains(Part::ManifoldHyperConnection));
        assert!(!parts.contains(Part::Csa2Compression));
    }

    #[test]
    fn a_missing_part_is_named_rather_than_guessed() {
        let architecture = Architecture::from_json(GLM).expect("GLM config");
        let available = PartSet::from_iter([Part::MlaAttention, Part::Moe]);
        let missing = architecture.missing(&available);
        assert!(missing.contains(&Part::KdaScan));
        assert!(missing.contains(&Part::SparseIndexer));
        assert!(!missing.contains(&Part::Moe));
        // The kernels are compiled for sm121 today; a target with none of them needs every part.
        assert!(architecture.missing_on(Target::Sm121).is_empty());
        assert_eq!(
            architecture.missing_on(Target::Sm120).len(),
            architecture.parts().len()
        );
    }

    #[test]
    fn an_unknown_selection_string_is_refused() {
        let mut root: Value = serde_json::from_str(QWEN).expect("Qwen config");
        root["text_config"]["hidden_act"] = Value::String("mystery".to_owned());
        let error = Architecture::from_value(&root).expect_err("unknown activation");
        assert!(matches!(error, ConfigError::Invalid(_)));
    }

    #[test]
    fn a_document_without_a_schedule_is_refused() {
        let error = Architecture::from_json(r#"{"model_type":"empty","text_config":{"hidden_size":8,"vocab_size":8,"num_hidden_layers":1,"num_attention_heads":1,"head_dim":8}}"#)
            .expect_err("no layer_types");
        assert!(matches!(error, ConfigError::Invalid(_)));
    }
}

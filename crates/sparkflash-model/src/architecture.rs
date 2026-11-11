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
use crate::vision_geometry::VisionGeometry;

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
    /// The looped schedule, when the release applies a shared block more than once.
    pub looped: Option<LoopSchedule>,
    /// The stored weight format, if the release declares one.
    pub quantization: Quantization,
    /// The vision tower, if the release carries one.
    pub vision: Option<Vision>,
    /// The MTP/DSpark stages, if the release carries any.
    pub mtp: Option<Mtp>,
    /// The checkpoint's special-token ids, as `config.json` and the tokenizer declare them.
    pub tokens: Tokens,
}

/// The special-token ids a checkpoint names.
///
/// A release names these in `config.json` (root or `text_config`) and, for the end-of-generation
/// set, sometimes only in `generation_config.json`. [Architecture::from_checkpoint_dir] merges the
/// two so a serving path never falls back to a compiled-in spelling.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Tokens {
    /// Beginning-of-sequence id, when the checkpoint names one.
    pub bos_token_id: Option<u32>,
    /// Every id that ends a generation, in the checkpoint's order.
    pub eos_token_ids: Vec<u32>,
    /// Padding id, when the checkpoint names one.
    pub pad_token_id: Option<u32>,
    /// The image placeholder's id, when the checkpoint carries a vision tower.
    pub image_token_id: Option<u32>,
    /// The video placeholder's id, when the checkpoint carries one.
    pub video_token_id: Option<u32>,
    /// The delimiter before an image's merged rows.
    pub vision_start_token_id: Option<u32>,
    /// The delimiter after an image's merged rows.
    pub vision_end_token_id: Option<u32>,
}

/// A looped schedule: the stored blocks are split into an input prefix, a shared recurrent
/// block applied many times, and an output suffix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopSchedule {
    /// Sublayers before the recurrent block.
    pub input_sublayers: usize,
    /// Sublayers in the shared recurrent block.
    pub recurrent_sublayers: usize,
    /// Sublayers after the recurrent block.
    pub output_sublayers: usize,
    /// How many times the recurrent block runs.
    pub loops: usize,
    /// Whether each loop keeps its own KV cache.
    pub per_loop_kv: bool,
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
    /// Indexer key heads (the QSA release runs one).
    pub kv_heads: usize,
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
    /// The table's padded row count is a multiple of this.
    pub vocab_multiple: u64,
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
    /// Whether every rank allocates the stage's cache.
    ///
    /// A release that drafts from the stage at every world size declares true (the default).
    /// Qwen4-Exp clears it: at TP>1 its rank configuration sets `mtp_num_hidden_layers = 0` and
    /// drafts with DFlash, so only the TP1 rank carries the stage's cache.
    pub replicated: bool,
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
    /// The dense projections' stored format.
    pub dense_format: WeightFormat,
    /// The backbone decoder experts' stored format.
    pub expert_format: WeightFormat,
    /// The MTP stage's expert format, which a release may store differently from the backbone's.
    pub mtp_expert_format: Option<WeightFormat>,
}

/// The arithmetic a weight format uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScaleDType {
    /// A 32-bit float inverse scale.
    F32,
    /// A BF16 inverse scale.
    Bf16,
    /// An 8-bit E4M3 (or `ue8m0`) scale.
    #[default]
    E4M3,
}

/// The stored format of one class of weights.
///
/// This is the arithmetic, not the producer's name: a ModelOpt release writes
/// `quant_method: "modelopt"` and records the format in `quant_algo` / `quantized_layers`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightFormat {
    /// Unquantized BF16.
    #[default]
    Bf16,
    /// E4M3 codes with an inverse scale a tile.
    Fp8Tile {
        /// Rows a scale covers.
        block_m: usize,
        /// Columns a scale covers.
        block_n: usize,
        /// The scale's dtype.
        scale_dtype: ScaleDType,
        /// Whether activations are quantized per block at run time.
        dynamic_activations: bool,
    },
    /// E2M1 codes grouped with per-group E4M3 scales and a global scale.
    NvFp4 {
        /// Elements a scale covers.
        group_size: usize,
    },
    /// E4M3 codes with one power-of-two `ue8m0` scale a block.
    MxFp8 {
        /// Elements a scale covers.
        block: usize,
    },
    /// E2M1 codes with one power-of-two `ue8m0` scale a block.
    MxFp4 {
        /// Elements a scale covers.
        block: usize,
    },
}

impl WeightFormat {
    /// The arithmetic's released name, for a document that renders the plan back.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::Fp8Tile { .. } => "FP8",
            Self::NvFp4 { .. } => "NVFP4",
            Self::MxFp8 { .. } => "MXFP8",
            Self::MxFp4 { .. } => "MXFP4",
        }
    }

    /// The format a name and a scale declaration describe.
    ///
    /// The producer's spellings carry the arithmetic: `fp4` without a power-of-two scale is
    /// NVIDIA's 16-element group, and the same name with a `ue8m0` scale is MXFP4's 32-element
    /// block. `fp8` with `ue8m0` is MXFP8 rather than a tile-scaled E4M3 format, which is what
    /// the DeepSeek release declares. A name this parser does not know stays BF16 rather than
    /// guessing a narrower arithmetic.
    #[must_use]
    pub fn from_name(method: Option<&str>, block: Option<usize>, scale: Option<&str>) -> Self {
        let name = method.unwrap_or("").to_ascii_lowercase();
        let scale = scale.unwrap_or("").to_ascii_lowercase();
        if name.contains("nvfp4") {
            return Self::NvFp4 {
                group_size: block.unwrap_or(16),
            };
        }
        if name.contains("mxfp4") {
            return Self::MxFp4 {
                block: block.unwrap_or(32),
            };
        }
        if name.contains("mxfp8") {
            return Self::MxFp8 {
                block: block.unwrap_or(32),
            };
        }
        if name.contains("fp4") {
            // The releases spell their 4-bit arithmetic `fp4`, not `nvfp4` or `mxfp4`. The
            // scale identifies which one: a ue8m0 scale is an MX block, and anything else is
            // NVIDIA's per-16-element group.
            return if scale.contains("ue8m0") {
                Self::MxFp4 {
                    block: block.unwrap_or(32),
                }
            } else {
                Self::NvFp4 {
                    group_size: block.unwrap_or(16),
                }
            };
        }
        if name.contains("fp8") || name.contains("e4m3") {
            if scale.contains("ue8m0") {
                return Self::MxFp8 {
                    block: block.unwrap_or(32),
                };
            }
            let scale_dtype = if scale.contains("bf16") {
                ScaleDType::Bf16
            } else {
                ScaleDType::F32
            };
            return Self::Fp8Tile {
                block_m: block.unwrap_or(128),
                block_n: block.unwrap_or(128),
                scale_dtype,
                dynamic_activations: true,
            };
        }
        Self::Bf16
    }
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
    /// Frames one patch covers.
    pub temporal_patch_size: usize,
    /// Spatial merge factor before the aligner.
    pub spatial_merge: usize,
    /// The aligner's hidden width.
    pub projection_intermediate: usize,
    /// Image channels.
    pub in_channels: usize,
    /// Learned position-grid entries, or zero when the tower ropes only.
    pub num_position_embeddings: usize,
    /// The smallest image the tower plans, in pixels.
    pub min_pixels: usize,
    /// The largest image the tower plans, in pixels.
    pub max_pixels: usize,
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

        let layers = layers(text, &family)?;
        let looped = loop_schedule(text, &family)?;
        let quantization = quantization(root)?;
        let vision = vision(root, hidden_size)?;
        let mtp = mtp(text, hidden_size)?;
        let tokens = tokens(root, text)?;

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
            looped,
            quantization,
            vision,
            mtp,
            tokens,
        })
    }

    /// The declarations in a `config.json` that change the model's arithmetic but that this
    /// parser does not read, each named for the operator.
    ///
    /// [Self::from_value] is deliberately tolerant: it reads the fields it models and ignores the
    /// rest, so a Gemma-2 document parses into a plain GQA stack and a Mixtral document into a
    /// dense one. A caller deciding whether a checkpoint can run must not take that parse as the
    /// whole story; this is the list of what the parse dropped. An empty list means nothing this
    /// function knows to look for was declared, not that every field was modelled.
    ///
    /// It looks for logit soft-capping (attention and final), a routed-expert count under a
    /// spelling the parser does not read (`num_local_experts`), and a quantization method whose
    /// format the parser cannot name, which [Self::from_value] reads as BF16.
    #[must_use]
    pub fn unmodelled_features(root: &Value) -> Vec<String> {
        let text = object(root, "text_config").unwrap_or(root);
        let mut found = Vec::new();
        for field in ["attn_logit_softcapping", "final_logit_softcapping"] {
            let declared = text
                .get(field)
                .or_else(|| root.get(field))
                .and_then(Value::as_f64);
            if let Some(cap) = declared.filter(|cap| *cap > 0.0) {
                found.push(format!(
                    "{field} {cap} (logit soft-capping is not modelled)"
                ));
            }
        }
        let routed_named = ["n_routed_experts", "num_experts"]
            .iter()
            .any(|field| text.get(*field).is_some_and(|value| !value.is_null()));
        if !routed_named
            && let Some(experts) = text.get("num_local_experts").and_then(Value::as_u64)
        {
            found.push(format!(
                "num_local_experts {experts} (Mixtral-style routed experts are not read; the \
                 layers would parse as dense MLPs)"
            ));
        }
        if let Some(config) = object(root, "quantization_config") {
            let declared = optional_string(config, "quant_method")
                .filter(|name| !names_the_producer(name))
                .or_else(|| optional_string(config, "quant_algo"));
            let modelled = quantization(root).is_ok_and(|parsed| {
                parsed.dense_format != WeightFormat::Bf16
                    || parsed.expert_format != WeightFormat::Bf16
            });
            let plain = declared.as_deref().is_none_or(|name| {
                matches!(name.to_ascii_lowercase().as_str(), "" | "bf16" | "none")
            });
            if !modelled && !plain {
                found.push(format!(
                    "quantization method {:?} (not a format this build models; it would be read \
                     as BF16)",
                    declared.unwrap_or_default()
                ));
            }
        }
        found
    }

    /// Re-describe a Qwen FP8 release whose global `fp8` flag quantises only its routed experts.
    ///
    /// The official Qwen3.8-Flash-Next FP8 pack stores the 24,576 routed experts as E4M3 with
    /// one BF16 inverse scale a 128x128 tile and keeps every other tensor (attention, shared
    /// experts, router, hyper-connections, embedding, head, MTP non-expert) BF16, but its
    /// producer still writes one global `quant_method: "fp8"`. Reading that flag as the dense
    /// format made [crate::Part] requirements dense/attention FP8 when the pack stores BF16.
    ///
    /// This is the Qwen4-Exp arm's producer correction and only that arm's bridge calls it
    /// (`sparkflash-model-qwen4-exp`'s `architecture_from_json` and
    /// `Qwen4ExpConfig::from_json_with_architecture`); it is not a general rule about FP8
    /// documents. The guard is narrow but not a proof of the producer: the parsed dense and expert
    /// formats must both be the same [WeightFormat::Fp8Tile], which a genuinely all-FP8 release
    /// such as GLM-5.3-Flash also satisfies, and applying it there would silently move that
    /// document's dense class to BF16. The expert format is left exactly as the document declared
    /// it; only the dense class moves to BF16.
    pub fn experts_only_fp8(&mut self) {
        let quantization = &mut self.quantization;
        if quantization.dense_format == quantization.expert_format
            && matches!(quantization.dense_format, WeightFormat::Fp8Tile { .. })
        {
            quantization.dense_format = WeightFormat::Bf16;
        }
    }

    /// The Qwen device arm's arithmetic on a Qwen3.5/3.6 FP8 release: weight-only E4M3.
    ///
    /// The release declares `activation_scheme: "dynamic"`, which the shared parser reads as an
    /// [WeightFormat::Fp8Tile] with `dynamic_activations`. The Qwen arm never quantizes an
    /// activation: it multiplies the f32 activation by the dequantised weight (W8A16), as its
    /// host oracle does. This clears the flag on every E4M3 class so the execution plan describes
    /// that arithmetic. A grouped chunk is then the BF16-activation tile from 64 tokens, not the
    /// q8 MMA that GLM-5.3-Flash's identical stored formats select through its dynamic scheme.
    ///
    /// Like [Self::experts_only_fp8], this is one arm's correction, called by the Qwen arm's
    /// bridge (and the web fit, which plans the document as that bridge does), not a rule about
    /// FP8 documents. The stored formats (codes, tiles, scale dtype) are unchanged.
    pub fn weight_only_fp8(&mut self) {
        let quantization = &mut self.quantization;
        let formats = [
            Some(&mut quantization.dense_format),
            Some(&mut quantization.expert_format),
            quantization.mtp_expert_format.as_mut(),
        ];
        for format in formats.into_iter().flatten() {
            if let WeightFormat::Fp8Tile {
                dynamic_activations,
                ..
            } = format
            {
                *dynamic_activations = false;
            }
        }
    }

    /// Read a checkpoint's `config.json`, merging `generation_config.json` when it sits beside.
    ///
    /// A serving path calls this with the snapshot directory it resolved from the Hugging Face
    /// cache, so the architecture, its special-token ids and its end-of-generation set all come
    /// from the checkpoint rather than from a compiled-in document.
    ///
    /// # Errors
    /// A read error, a malformed document, or a shape [Self::from_json] refuses.
    pub fn from_checkpoint_dir(dir: &std::path::Path) -> Result<Self, ConfigError> {
        let mut architecture = Self::from_path(&dir.join("config.json"))?;
        let generation = dir.join("generation_config.json");
        if generation.is_file() {
            let text = std::fs::read_to_string(&generation)
                .map_err(|error| ConfigError::Json(format!("{}: {error}", generation.display())))?;
            let value: Value = serde_json::from_str(&text)
                .map_err(|error| ConfigError::Json(error.to_string()))?;
            // A generation config can add stop ids the model config does not name, but it
            // cannot remove the model's own: the config's order is the served order, and the
            // first id is the one the PLE segment rule resets on. Union generation-only ids
            // after the config's, then dedup.
            if value
                .get("eos_token_id")
                .is_some_and(|field| !field.is_null())
            {
                let mut ids = architecture.tokens.eos_token_ids.clone();
                ids.extend(eos_ids(&value)?);
                architecture.tokens.eos_token_ids = dedup(ids);
            }
            if let Some(id) = optional_u32(&value, "bos_token_id")? {
                architecture.tokens.bos_token_id.get_or_insert(id);
            }
            if let Some(id) = optional_u32(&value, "pad_token_id")? {
                architecture.tokens.pad_token_id.get_or_insert(id);
            }
        }
        Ok(architecture)
    }

    /// The cache claim this architecture makes at `world` ranks.
    ///
    /// The two terms are the ones a serving flag can move: [`KvPlan::per_token`] is every cache
    /// a position adds (attention K/V, an absorbed latent, an indexer's keys and pooled key), and
    /// [`KvPlan::fixed`] is every state a session allocates regardless of its window (a recurrent
    /// state and the spare a verify keeps beside it). The arithmetic is the same one the arms'
    /// hand-written plans used, now read from the parsed geometry.
    ///
    /// The fixed term is deliberately not uniform across attention families: a GDN layer counts
    /// its recurrent state and verify spare only, while a KDA layer includes its convolution
    /// history. Qwen's GDN convolution history and its PLE convolution frames are outside this
    /// plan, which is the convention the pinned Qwen TP=1 figure was taken under; adding them
    /// would recalibrate that figure and is a separate change.
    ///
    /// `per_token` is a full-window upper bound: it counts every cache as if the position were
    /// inside every layer's window and does not subtract a layer's per-layer sliding window.
    /// A caller sizing a shorter context therefore gets a safe over-count, not a wrong one.
    ///
    /// `latent_fp8` is the caller's explicit choice of the absorbed-MLA latent representation,
    /// not a value re-derived from the quantization block: true stores E4M3 codes plus one f32
    /// scale per 128-wide block, false stores the f32 latent. The E4M3 choice needs a latent
    /// width that is whole 128-wide scale tiles; a narrower row falls back to f32, the same
    /// condition the device's latent-cache choice applies. A GLM pack passes its
    /// [`crate::profile::LatentCacheMode`] choice and Qwen passes false. The latent row is
    /// `kv_lora_rank + rope.rotated_dims`, because the absorbed cache keeps the rope part
    /// beside the compressed latent.
    ///
    /// An indexer's key, gate and pooled key are counted only for a layer that
    /// [`Indexer::owns_keys`]; a shared indexer reads another layer's selection and allocates
    /// nothing. The pooled key is counted as `dim / pool` a position, which is exactly the
    /// device's `capacity / pool` rows. An MTP stage allocates the same caches as the last
    /// backbone attention layer, including an indexer (the GLM device allocator builds its MTP
    /// stage with `own_indexer = true`) whenever the architecture declares indexer geometry
    /// anywhere, even if that last layer shares another layer's selection. A stage whose
    /// [`Mtp::replicated`] is false is counted only at TP1.
    ///
    /// # Errors
    /// A checkpoint with CSA2 compression, whose compressed caches are not the one latent row a
    /// layer this plan models, is refused rather than described by a wrong number.
    pub fn kv_plan(&self, world: usize, latent_fp8: bool) -> Result<crate::KvPlan, ConfigError> {
        if self
            .layers
            .iter()
            .any(|layer| layer.attention.compression.is_some())
        {
            return Err(invalid(
                "kv_plan: the checkpoint uses CSA2 compression, whose compressed cache is not \
                 the one latent row this plan models; the compressed release needs its own plan",
            ));
        }
        let world = world.max(1) as u64;
        let mut per_token = 0u64;
        let mut fixed = 0u64;
        for layer in &self.layers {
            match &layer.attention.kind {
                AttentionKind::Gqa { kv_heads, .. } => {
                    let heads = ((*kv_heads as u64).div_ceil(world)).max(1);
                    let kv = 2u64
                        .saturating_mul(heads)
                        .saturating_mul(layer.attention.head_dim as u64)
                        .saturating_mul(2);
                    per_token = per_token.saturating_add(kv);
                    if let Some(indexer) = &layer.attention.indexer
                        && indexer.owns_keys
                    {
                        // QSA keeps one indexer key row per position.
                        per_token =
                            per_token.saturating_add((indexer.head_dim as u64).saturating_mul(2));
                    }
                }
                AttentionKind::Mla { kv_lora_rank, .. } => {
                    let width = (*kv_lora_rank as u64)
                        .saturating_add(layer.attention.rope.rotated_dims as u64);
                    per_token = per_token.saturating_add(latent_bytes(width, latent_fp8));
                    if let Some(indexer) = &layer.attention.indexer
                        && indexer.owns_keys
                    {
                        per_token = per_token.saturating_add(mla_indexer_bytes(indexer));
                    }
                }
                AttentionKind::Gdn(gdn) => {
                    fixed = fixed.saturating_add(gdn_state_bytes(gdn, world));
                }
                AttentionKind::Kda(kda) => {
                    let heads = ((kda.heads as u64).div_ceil(world)).max(1);
                    let head = kda.head_dim as u64;
                    let history = kda.conv_kernel.saturating_sub(1) as u64;
                    fixed = fixed.saturating_add(
                        history
                            .saturating_mul(3)
                            .saturating_mul(heads)
                            .saturating_mul(head)
                            .saturating_mul(4),
                    );
                    fixed = fixed.saturating_add(
                        heads
                            .saturating_mul(head)
                            .saturating_mul(head)
                            .saturating_mul(8),
                    );
                }
            }
        }
        if let Some(mtp) = &self.mtp
            && (mtp.replicated || world == 1)
        {
            per_token = per_token.saturating_add(
                self.mtp_per_token(world, latent_fp8)
                    .saturating_mul(mtp.stages as u64),
            );
        }
        if let Some(looped) = &self.looped
            && looped.per_loop_kv
            && looped.loops > 1
        {
            // Only the recurrent slice is replayed; the input and output blocks run once.
            let start = looped.input_sublayers / 2;
            let end = start + looped.recurrent_sublayers / 2;
            let extra = (looped.loops - 1) as u64;
            for layer in self.layers.get(start..end).unwrap_or_default() {
                match &layer.attention.kind {
                    AttentionKind::Gqa { kv_heads, .. } => {
                        let heads = ((*kv_heads as u64).div_ceil(world)).max(1);
                        let kv = 2u64
                            .saturating_mul(heads)
                            .saturating_mul(layer.attention.head_dim as u64)
                            .saturating_mul(2);
                        per_token = per_token.saturating_add(kv.saturating_mul(extra));
                    }
                    AttentionKind::Mla { kv_lora_rank, .. } => {
                        let width = (*kv_lora_rank as u64)
                            .saturating_add(layer.attention.rope.rotated_dims as u64);
                        per_token = per_token
                            .saturating_add(latent_bytes(width, latent_fp8).saturating_mul(extra));
                    }
                    AttentionKind::Gdn(_) | AttentionKind::Kda(_) => {}
                }
            }
        }
        Ok(crate::KvPlan::new(per_token, fixed))
    }

    /// The per-position cache one MTP stage adds.
    ///
    /// A stage reuses the backbone's last attention kind: an absorbed-MLA stage adds its latent
    /// and the indexer's key, gate and pooled key; a GQA stage adds its K/V rows and its
    /// indexer key. The device allocator builds both stages with their own indexer
    /// (`DeviceRank::mtp` passes `own_indexer = true`), so the plan counts one whenever the
    /// architecture declares indexer geometry anywhere, even when the last backbone attention
    /// layer shares another layer's selection and declares none of its own.
    fn mtp_per_token(&self, world: u64, latent_fp8: bool) -> u64 {
        let Some(last) = self.layers.iter().rev().find(|layer| {
            matches!(
                layer.attention.kind,
                AttentionKind::Gqa { .. } | AttentionKind::Mla { .. }
            )
        }) else {
            return 0;
        };
        // The device's MTP indexer reads its geometry from the checkpoint, not from the last
        // layer's schedule. Prefer that layer's own indexer, then any indexer the architecture
        // declares.
        let declared = self
            .layers
            .iter()
            .find_map(|layer| layer.attention.indexer.as_ref());
        match &last.attention.kind {
            AttentionKind::Gqa { kv_heads, .. } => {
                let heads = ((*kv_heads as u64).div_ceil(world)).max(1);
                let indexer = last
                    .attention
                    .indexer
                    .as_ref()
                    .filter(|indexer| indexer.owns_keys)
                    .or(declared)
                    .map_or(0, |indexer| (indexer.head_dim as u64) * 2);
                2u64.saturating_mul(heads)
                    .saturating_mul(last.attention.head_dim as u64)
                    .saturating_mul(2)
                    .saturating_add(indexer)
            }
            AttentionKind::Mla { kv_lora_rank, .. } => {
                let width =
                    (*kv_lora_rank as u64).saturating_add(last.attention.rope.rotated_dims as u64);
                let indexer = last
                    .attention
                    .indexer
                    .as_ref()
                    .filter(|indexer| indexer.owns_keys)
                    .or(declared)
                    .map_or(0, mla_indexer_bytes);
                latent_bytes(width, latent_fp8).saturating_add(indexer)
            }
            _ => 0,
        }
    }

    /// The vision tower's geometry in the shared structural form, when the checkpoint carries one.
    #[must_use]
    pub fn vision_geometry(&self) -> Option<VisionGeometry> {
        self.vision.as_ref().map(VisionGeometry::from_vision)
    }

    /// Whether the checkpoint's RMSNorm weights are zero-centred.
    ///
    /// Qwen's hybrid releases store `weight` around zero and compute `norm(x) * (1 + weight)`;
    /// a plain `weight` multiply would scale the residual stream to near zero. The Gated
    /// DeltaNet's gated norm is a separate, conventionally centred tensor and is not covered.
    ///
    /// This is a family fact the released document does not spell out: the Transformers config
    /// classes hard-code `Qwen3_5MoeRMSNorm`/`Qwen4ExpRMSNorm` with a zero-initialised weight.
    #[must_use]
    pub fn zero_centered_norms(&self) -> bool {
        matches!(
            self.family.as_str(),
            "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen4_exp" | "qwen4_exp_text"
        )
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

fn layers(text: &Value, family: &str) -> Result<Vec<Layer>, ConfigError> {
    if family == "dory" {
        return dory_layers(text);
    }
    let count = required_usize(text, "num_hidden_layers", "text_config")?;
    let mut layer_types = string_list(text.get("layer_types"))?;
    let ratios = usize_list(text.get("compress_ratios"))?;
    let indexer_types = string_list(text.get("indexer_types"))?;
    if layer_types.is_empty() && ratios.is_empty() {
        // A release that names only `full_attention_interval` uses the reference rule: every
        // Nth layer is full/sparse attention and the rest are linear. A Qwen-shaped release
        // that names linear-attention geometry but omits the interval gets the same default
        // Qwen4ExpConfig::from_json applies (4), so the two parsers cannot disagree about a
        // checkpoint that omits `layer_types`. A release with no schedule and no linear
        // geometry is the plain dense/GQA case: every layer is full attention.
        let interval = match optional_usize(text, "full_attention_interval")? {
            Some(0) => {
                return Err(invalid(
                    "full_attention_interval must not be zero".to_owned(),
                ));
            }
            Some(interval) => Some(interval),
            None if has_linear_attention(text) => Some(4),
            None => None,
        };
        layer_types = match interval {
            Some(interval) => (0..count)
                .map(|index| {
                    if (index + 1).is_multiple_of(interval) {
                        "full_attention".to_owned()
                    } else {
                        "linear_attention".to_owned()
                    }
                })
                .collect(),
            None => vec!["full_attention".to_owned(); count],
        };
    } else if layer_types.is_empty() {
        // A compressed release with ratios but no names: the reference names its ratio-zero
        // layers full attention and its compressed layers compressed attention.
        layer_types = (0..count)
            .map(|index| {
                if ratios.get(index).copied().unwrap_or(0) > 0 {
                    "compressed_attention".to_owned()
                } else {
                    "full_attention".to_owned()
                }
            })
            .collect();
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

/// The collapsed block schedule of a looped Dory checkpoint.
///
/// Dory stores each transformer block as two entries in layer_types: an attention sublayer and
/// an MLP sublayer. The shared layer walk wants one Layer per block, so the pair becomes one
/// layer whose attention and dense MLP are the pair's two stored sublayers.
fn dory_layers(text: &Value) -> Result<Vec<Layer>, ConfigError> {
    let count = required_usize(text, "num_hidden_layers", "text_config")?;
    if !count.is_multiple_of(2) {
        return Err(invalid(
            "a looped model stores attention and MLP sublayers in pairs, so num_hidden_layers must be even",
        ));
    }
    let layer_types = string_list(text.get("layer_types"))?;
    if layer_types.len() != count {
        return Err(invalid(format!(
            "layer_types has {} entries, expected num_hidden_layers = {count}",
            layer_types.len()
        )));
    }
    let schedule = Schedule {
        kv_sources: &[],
        index_sources: &[],
        indexer_types: &[],
        candidate_pool: None,
    };
    let mut layers = Vec::with_capacity(count / 2);
    for block in 0..count / 2 {
        let attention_type = layer_types[block * 2].as_str();
        if attention_type == "mlp" {
            return Err(invalid(format!(
                "block {block} starts with an MLP sublayer, expected an attention sublayer"
            )));
        }
        if layer_types[block * 2 + 1] != "mlp" {
            return Err(invalid(format!(
                "block {block} does not follow its attention sublayer with an MLP sublayer"
            )));
        }
        let attention = attention(text, block * 2, attention_type, 0, &schedule)?;
        let ffn = ffn(text, block * 2 + 1, 0, &[])?;
        if !matches!(ffn, Ffn::Dense(_)) {
            return Err(invalid(format!("block {block} is not a dense MLP block")));
        }
        layers.push(Layer {
            index: block,
            attention,
            ffn,
            residual: Residual::Plain,
            sides: Vec::new(),
        });
    }
    Ok(layers)
}

/// The loop metadata of a looped Dory checkpoint.
fn loop_schedule(text: &Value, family: &str) -> Result<Option<LoopSchedule>, ConfigError> {
    if family != "dory" {
        return Ok(None);
    }
    let input_sublayers = required_usize(text, "n_input_layers", "text_config")?;
    let output_sublayers = required_usize(text, "n_output_layers", "text_config")?;
    let count = required_usize(text, "num_hidden_layers", "text_config")?;
    let recurrent_sublayers = optional_usize(text, "n_recurrent_layers")?
        .unwrap_or(count.saturating_sub(input_sublayers + output_sublayers));
    if input_sublayers + recurrent_sublayers + output_sublayers != count {
        return Err(invalid(
            "the input, recurrent and output sublayers do not sum to num_hidden_layers",
        ));
    }
    for (name, sublayers) in [
        ("n_input_layers", input_sublayers),
        ("n_recurrent_layers", recurrent_sublayers),
        ("n_output_layers", output_sublayers),
    ] {
        if !sublayers.is_multiple_of(2) {
            return Err(invalid(format!(
                "{name} must be even: Dory stores attention and MLP sublayers in pairs"
            )));
        }
    }
    let loops = optional_usize(text, "n_recurrent_loops")?.unwrap_or(1);
    if loops == 0 {
        return Err(invalid("n_recurrent_loops must be positive"));
    }
    let per_loop_kv = match optional_string(text, "recurrent_kv_cache_mode")
        .as_deref()
        .unwrap_or("per_loop")
    {
        "per_loop" => true,
        "last_loop" => false,
        other => {
            return Err(invalid(format!(
                "recurrent_kv_cache_mode must be per_loop or last_loop, not {other:?}"
            )));
        }
    };
    Ok(Some(LoopSchedule {
        input_sublayers,
        recurrent_sublayers,
        output_sublayers,
        loops,
        per_loop_kv,
    }))
}

/// Whether a document carries the linear-attention geometry of a hybrid release.
fn has_linear_attention(text: &Value) -> bool {
    object(text, "linear_attn_config").is_some()
        || [
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
        ]
        .iter()
        .any(|field| text.get(*field).is_some())
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
    if heads == 0 {
        return Err(invalid(format!("layer {index} has zero query heads")));
    }
    let declared_head_dim = optional_usize(text, "head_dim")?.unwrap_or(0);
    let nope = optional_usize(text, "qk_nope_head_dim")?.unwrap_or(0);
    let declared_rope_dim = optional_usize(text, "qk_rope_head_dim")?;
    let rope_dim = declared_rope_dim.unwrap_or(0);
    let head_dim = if declared_head_dim > 0 {
        declared_head_dim
    } else if nope + rope_dim > 0 {
        nope + rope_dim
    } else {
        // A standard Llama/Qwen2 configuration names neither: the head width is the model's
        // width divided by its query heads, and only when the division is exact. A ragged
        // width is a refused shape, not a silent truncation.
        let hidden = required_usize(text, "hidden_size", "text_config")?;
        if !hidden.is_multiple_of(heads) {
            return Err(invalid(format!(
                "layer {index} has no head_dim and hidden_size {hidden} is not divisible by {heads} heads"
            )));
        }
        hidden / heads
    };
    if head_dim == 0 {
        return Err(invalid(format!(
            "layer {index} has no head_dim, no qk_nope_head_dim and no divisible hidden_size"
        )));
    }
    let rope_parameters = object(text, "rope_parameters");
    // A looped model keys its rotary config by attention kind; the selected block is the one
    // this layer runs. Every other release keeps the flat fields.
    let selected_rope = rope_parameters
        .and_then(|parameters| parameters.get(layer_type))
        .filter(|value| value.is_object());
    let window = if selected_rope.is_some() && layer_type == "full_attention" {
        None
    } else {
        optional_usize(text, "sliding_window")?
    };
    // Newer releases (Qwen4-Exp among them) nest both rotary fields under
    // `rope_parameters`; the top-level spelling is the older form and wins when both appear.
    let nested_partial = match selected_rope.or(rope_parameters) {
        Some(value) => optional_f64(value, "partial_rotary_factor")?,
        None => None,
    };
    let partial = optional_f64(text, "partial_rotary_factor")?
        .or(nested_partial)
        .unwrap_or(0.0);
    let rotated_dims = if let Some(dim) = declared_rope_dim {
        // An explicit `qk_rope_head_dim` wins, including an explicit zero (GLM's NoPE MLA).
        dim
    } else if partial > 0.0 {
        ((head_dim as f64 * partial).round() as usize).min(head_dim)
    } else {
        // No rope declaration is not a request for NoPE: the reference implementations rotate
        // the whole head unless a release says otherwise.
        head_dim
    };
    let rope = rope(text, ratio, rotated_dims, selected_rope)?;

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
        // A standard dense/GQA configuration carries no learned output gate; the Qwen and GLM
        // releases either spell one or run a QSA indexer whose gate is part of the arithmetic.
        let declares_indexer = [
            "index_n_heads",
            "indexer_n_heads",
            "index_head_dim",
            "indexer_head_dim",
            "indexer_compress_ratio",
            "index_kpool",
        ]
        .iter()
        .any(|field| text.get(field).is_some());
        let output_gate = optional_bool(text, "attn_output_gate")?
            .or(optional_bool(text, "use_output_gate")?)
            .unwrap_or_else(|| {
                optional_string(text, "output_gate_type").is_some()
                    || declares_indexer
                    || text.get("attention_output_gate_logit_scale_init").is_some()
            });
        AttentionKind::Gqa {
            kv_heads: optional_usize(text, "num_key_value_heads")?.unwrap_or(heads),
            output_gate,
        }
    } else {
        return Err(invalid(format!(
            "layer {index} has unknown layer_type {layer_type:?}"
        )));
    };

    let compression = (ratio > 0).then_some(Compression {
        ratio,
        // The same ownership rule as the indexer: a layer named in `kv_source_layer_ids`
        // reads another layer's compressed cache rather than allocating its own.
        owns_keys: schedule.kv_sources.is_empty() || schedule.kv_sources.contains(&index),
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
    // A standard GQA release declares no indexer arithmetic; an all-"full" schedule is not a
    // request to invent one.
    let declares_indexer = [
        "index_n_heads",
        "indexer_n_heads",
        "index_head_dim",
        "indexer_head_dim",
        "indexer_compress_ratio",
        "index_kpool",
        "index_topk",
        "indexer_budget",
        "index_source_layer_ids",
    ]
    .iter()
    .any(|field| text.get(field).is_some());
    if !declares_indexer {
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
        kv_heads: optional_usize(text, "indexer_kv_heads")?
            .or(optional_usize(text, "index_kv_heads")?)
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

fn rope(
    text: &Value,
    ratio: usize,
    rotated_dims: usize,
    selected: Option<&Value>,
) -> Result<Rope, ConfigError> {
    let compress_theta = optional_f64(text, "compress_rope_theta")?;
    let parameters = selected.or_else(|| object(text, "rope_parameters"));
    // Qwen4-Exp stores the base under `rope_parameters`; older releases keep it top-level.
    let nested_theta = match parameters {
        Some(value) => optional_f64(value, "rope_theta")?,
        None => None,
    };
    let top_theta = optional_f64(text, "rope_theta")?;
    let selected_theta = match selected {
        Some(value) => optional_f64(value, "rope_theta")?,
        None => None,
    };
    let theta = if ratio > 0 {
        compress_theta
            .or(top_theta)
            .or(nested_theta)
            .unwrap_or_else(|| default_rope_theta(text))
    } else {
        selected_theta
            .or(top_theta)
            .or(nested_theta)
            .unwrap_or_else(|| default_rope_theta(text))
    };
    let scaling_json = selected
        .or_else(|| object(text, "rope_scaling"))
        .or(parameters);
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
                // A historical document spells the rule as `type`; the nested
                // `rope_parameters` form uses `rope_type`. Treat them as one field, or a YaRN
                // checkpoint parses as "default" and the host serves it unscaled.
                kind: optional_string(value, "rope_type")
                    .or_else(|| optional_string(value, "type"))
                    .unwrap_or_else(|| "default".to_owned()),
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

/// The rotary base of a document that omits one.
///
/// Qwen4-Exp's hybrid linear-attention release documents use 10,000,000; a plain transformer
/// keeps the usual 10,000. This is the one default both the shared architecture and a typed
/// arm's parser read, so a fine-tune that omits `rope_theta` cannot load with two different
/// bases depending on which view parsed it.
fn default_rope_theta(text: &Value) -> f64 {
    if has_linear_attention(text) {
        10_000_000.0
    } else {
        10_000.0
    }
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
        vocab_multiple: optional_usize(text, "make_ngram_vocab_size_divisible_by")?.unwrap_or(0)
            as u64,
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
    // Qwen4-Exp's nested stage block says `hybrid: true` and the document also carries
    // `mtp_use_dedicated_embeddings`: at TP>1 its rank config turns the stage off and drafts
    // with DFlash, so every rank does not own the stage's cache. A nested `mtp.hybrid` alone
    // is not enough - a GLM-style nested stage block can say `hybrid` while still owning its
    // cache on every rank - so the Qwen marker (or an explicit `mtp.replicated`) decides, and
    // the default stays replicated.
    let replicated = match object(text, "mtp") {
        Some(nested) => {
            if let Some(replicated) = optional_bool(nested, "replicated")? {
                replicated
            } else {
                !(optional_bool(nested, "hybrid")?.unwrap_or(false)
                    && text.get("mtp_use_dedicated_embeddings").is_some())
            }
        }
        None => true,
    };
    Ok(Some(Mtp {
        stages,
        block_size: optional_usize(text, "dspark_block_size")?,
        target_layers: usize_list(text.get("dspark_target_layer_ids"))?,
        routed_experts: optional_usize(text, "dspark_n_routed_experts")?,
        active_experts: optional_usize(text, "dspark_num_experts_per_tok")?,
        replicated,
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

/// One stored format a ModelOpt table names, with the group width it declares.
///
/// The group width belongs to the arithmetic, not to `weight_block_size`: a ModelOpt group
/// records it in the same `weights` block or `quantized_layers` entry as the algorithm, and a
/// release that names no top-level block still has a real per-16-element group.
#[derive(Debug, Clone)]
struct DeclaredFormat {
    name: String,
    group_size: Option<usize>,
}

impl DeclaredFormat {
    fn named(name: String) -> Self {
        Self {
            name,
            group_size: None,
        }
    }

    /// The format this declaration describes, with its own group width when it has one.
    fn format(&self, block: Option<usize>, scale: Option<&str>) -> WeightFormat {
        let format = WeightFormat::from_name(Some(&self.name), block, scale);
        match (format, self.group_size) {
            (WeightFormat::NvFp4 { .. }, Some(group_size)) => WeightFormat::NvFp4 { group_size },
            (format, _) => format,
        }
    }
}

/// The routed experts' format as a ModelOpt `quantization_config` records it.
///
/// `quantized_layers` keys every quantized module to its own `quant_algo`, and the decoder
/// layers' routed experts are the entries under `.mlp.experts`. A release that omits that table
/// still names the same arithmetic in `config_groups`, as a bit width and a number type over a
/// group of targets.
fn modelopt_expert(config: &Value) -> Option<DeclaredFormat> {
    if let Some(layers) = object(config, "quantized_layers").and_then(Value::as_object) {
        let mut expert: Option<DeclaredFormat> = None;
        for (name, spec) in layers {
            if !name.contains(".mlp.experts") || is_mtp_module(name) {
                continue;
            }
            let Some(algo) = string(spec, "quant_algo") else {
                continue;
            };
            let group_size = optional_usize(spec, "group_size").ok().flatten();
            match &expert {
                // Experts that disagree are not one stored format, and the plan has no name for
                // that mix: report none rather than pick one of them.
                Some(first) if !first.name.eq_ignore_ascii_case(&algo) => return None,
                Some(_) => {}
                None => {
                    expert = Some(DeclaredFormat {
                        name: algo,
                        group_size,
                    });
                }
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
fn group_format(weights: &Value) -> Option<DeclaredFormat> {
    let bits = weights.get("num_bits").and_then(Value::as_u64)?;
    let kind = string(weights, "type")?;
    let name = match (bits, kind.to_ascii_lowercase().as_str()) {
        (4, "float") => "NVFP4".to_owned(),
        (8, "float") => "FP8".to_owned(),
        _ => return None,
    };
    Some(DeclaredFormat {
        name,
        group_size: optional_usize(weights, "group_size").ok().flatten(),
    })
}

/// The MTP stage's routed-expert format as ModelOpt's `quantized_layers` records it.
fn modelopt_mtp_expert(config: &Value) -> Option<DeclaredFormat> {
    let layers = object(config, "quantized_layers")?.as_object()?;
    let mut expert: Option<DeclaredFormat> = None;
    for (name, spec) in layers {
        if !name.contains(".mlp.experts") || !is_mtp_module(name) {
            continue;
        }
        let Some(algo) = string(spec, "quant_algo") else {
            continue;
        };
        match &expert {
            Some(first) if !first.name.eq_ignore_ascii_case(&algo) => return None,
            Some(_) => {}
            None => {
                expert = Some(DeclaredFormat {
                    name: algo,
                    group_size: optional_usize(spec, "group_size").ok().flatten(),
                });
            }
        }
    }
    expert
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
        .map(DeclaredFormat::named)
        .or_else(|| modelopt_expert(config));
    // A producer name is not the arithmetic; ModelOpt records that in quant_algo.
    let method = match optional_string(config, "quant_method") {
        Some(name) if !names_the_producer(&name) => Some(name),
        _ => optional_string(config, "quant_algo"),
    };
    let mtp_expert = object(config, "mtp_expert_weight")
        .and_then(|value| string(value, "quant_algo"))
        .or_else(|| optional_string(config, "mtp_expert_dtype"))
        .map(DeclaredFormat::named)
        .or_else(|| modelopt_mtp_expert(config));
    let scale = optional_string(config, "scale_fmt").or(optional_string(config, "fmt"));
    let dense_format = WeightFormat::from_name(method.as_deref(), block, scale.as_deref());
    // A release that names no separate expert format stores the experts in the dense format.
    // GLM-5.3-Flash is one: its single `fp8` / `e4m3` declaration governs the dense and the
    // routed weights alike, and reading the missing name as BF16 served FP8 experts as BF16.
    let expert_format = match &expert {
        Some(declared) => declared.format(block, scale.as_deref()),
        None => dense_format,
    };
    let mtp_expert_format = mtp_expert
        .as_ref()
        .map(|declared| declared.format(block, scale.as_deref()));
    Ok(Quantization {
        method,
        block,
        scale,
        activation: optional_string(config, "activation_scheme"),
        expert: expert.map(|declared| declared.name),
        dense_format,
        expert_format,
        mtp_expert_format,
    })
}

fn vision(root: &Value, language_hidden: usize) -> Result<Option<Vision>, ConfigError> {
    let Some(config) = object(root, "vision_config") else {
        return Ok(None);
    };
    // One structural parser for both the architecture and the arms' `VisionTowerConfig`. The
    // language width is passed in because a tower that names no merger output hands the
    // language model its own width, not the tower's block width.
    let geometry = VisionGeometry::from_value(config, language_hidden)?;
    geometry.validate()?;
    Ok(Some(Vision {
        depth: geometry.depth,
        hidden_size: geometry.hidden_size,
        heads: geometry.num_heads,
        intermediate: geometry.intermediate_size,
        patch_size: geometry.patch_size,
        temporal_patch_size: geometry.temporal_patch_size,
        spatial_merge: geometry.spatial_merge_size,
        projection_intermediate: geometry.projection_intermediate_size,
        in_channels: geometry.in_channels,
        num_position_embeddings: geometry.num_position_embeddings,
        min_pixels: geometry.min_pixels,
        max_pixels: geometry.max_pixels,
        out_hidden_size: geometry.out_hidden_size,
        rope_theta: geometry.rope_theta,
    }))
}

/// A DFlash 2 drafter checks its target by width, vocabulary and backbone depth.
impl From<&Architecture> for crate::dflash2::DraftTarget {
    fn from(architecture: &Architecture) -> Self {
        Self {
            hidden_size: architecture.hidden_size,
            vocab_size: architecture.vocab_size,
            num_hidden_layers: architecture.layers.len(),
        }
    }
}

/// One position's absorbed-MLA latent row, in the representation the caller selected.
///
/// The FP8 latent stores one E4M3 code a element plus one f32 inverse scale a 128-wide block;
/// the f32 latent stores four bytes a element. The choice is explicit (`latent_fp8`), never
/// re-derived from the weight format: the same checkpoint can be served with either cache.
///
/// The device's E4M3 latent is allocated in whole 128-wide scale tiles: a narrower
/// row falls back to f32 however the caller asks (`LatentCache::quantises` applies the same
/// guard), so a narrow latent has one plan in both modes.
fn latent_bytes(width: u64, latent_fp8: bool) -> u64 {
    if !latent_fp8 || !width.is_multiple_of(128) {
        return width.saturating_mul(4);
    }
    let block = 128u64;
    width.saturating_add(width.div_ceil(block).saturating_mul(4))
}

/// One position's indexer caches on an MLA layer: the key and gate rows, then the pooled key
/// amortised over one pool.
fn mla_indexer_bytes(indexer: &Indexer) -> u64 {
    let dim = indexer.head_dim as u64;
    let pool = indexer.pool.max(1) as u64;
    dim.saturating_mul(8)
        .saturating_add(dim.saturating_mul(4).div_ceil(pool))
}

/// The live recurrent state and the spare a verify keeps beside it, one GDN layer.
fn gdn_state_bytes(gdn: &Gdn, world: u64) -> u64 {
    let value_heads = ((gdn.value_heads as u64).div_ceil(world)).max(1);
    let key_dim = gdn.key_dim as u64;
    let value_dim = gdn.value_dim as u64;
    value_heads
        .saturating_mul(key_dim)
        .saturating_mul(value_dim)
        .saturating_mul(8)
}

/// The special-token ids a checkpoint names, preferring the language model's `text_config`.
fn tokens(root: &Value, text: &Value) -> Result<Tokens, ConfigError> {
    Ok(Tokens {
        bos_token_id: optional_u32(text, "bos_token_id")?.or(optional_u32(root, "bos_token_id")?),
        eos_token_ids: {
            let mut ids = eos_ids(text)?;
            ids.extend(eos_ids(root)?);
            dedup(ids)
        },
        pad_token_id: optional_u32(text, "pad_token_id")?.or(optional_u32(root, "pad_token_id")?),
        image_token_id: optional_u32(text, "image_token_id")?
            .or(optional_u32(root, "image_token_id")?),
        video_token_id: optional_u32(text, "video_token_id")?
            .or(optional_u32(root, "video_token_id")?),
        vision_start_token_id: optional_u32(text, "vision_start_token_id")?
            .or(optional_u32(root, "vision_start_token_id")?)
            .or(optional_u32(text, "image_start_token_id")?)
            .or(optional_u32(root, "image_start_token_id")?)
            .or(optional_u32(text, "video_start_token_id")?)
            .or(optional_u32(root, "video_start_token_id")?),
        vision_end_token_id: optional_u32(text, "vision_end_token_id")?
            .or(optional_u32(root, "vision_end_token_id")?)
            .or(optional_u32(text, "image_end_token_id")?)
            .or(optional_u32(root, "image_end_token_id")?)
            .or(optional_u32(text, "video_end_token_id")?)
            .or(optional_u32(root, "video_end_token_id")?),
    })
}

/// A token-id list with its first occurrence of each id kept, in declaration order.
fn dedup(ids: Vec<u32>) -> Vec<u32> {
    let mut unique = Vec::with_capacity(ids.len());
    for id in ids {
        if !unique.contains(&id) {
            unique.push(id);
        }
    }
    unique
}

/// Every end-of-generation id a document names, one or many.
fn eos_ids(value: &Value) -> Result<Vec<u32>, ConfigError> {
    let Some(field) = value.get("eos_token_id") else {
        return Ok(Vec::new());
    };
    if field.is_null() {
        return Ok(Vec::new());
    }
    let one: OneOrMany<u32> = serde_json::from_value(field.clone())
        .map_err(|error| invalid(format!("eos_token_id: {error}")))?;
    Ok(one.into_vec())
}

fn optional_u32(value: &Value, name: &str) -> Result<Option<u32>, ConfigError> {
    optional_usize(value, name)?
        .map(|number| {
            u32::try_from(number).map_err(|_| invalid(format!("{name} does not fit a token id")))
        })
        .transpose()
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

    #[test]
    fn weight_only_fp8_clears_only_the_activation_scheme() {
        let text = include_str!("../config/qwen36_flash_35b_a3b_fp8.hf.json");
        let declared = Architecture::from_json(text).expect("Qwen3.6 FP8");
        let tile = |dynamic_activations| WeightFormat::Fp8Tile {
            block_m: 128,
            block_n: 128,
            scale_dtype: ScaleDType::F32,
            dynamic_activations,
        };
        assert_eq!(declared.quantization.dense_format, tile(true));
        assert_eq!(declared.quantization.expert_format, tile(true));
        let mut computed = declared.clone();
        computed.weight_only_fp8();
        assert_eq!(computed.quantization.dense_format, tile(false));
        assert_eq!(computed.quantization.expert_format, tile(false));
        let mut rest = computed.clone();
        rest.quantization = declared.quantization.clone();
        assert_eq!(rest, declared, "nothing but the scheme moves");
        // Only the E4M3 classes move: Qwen3.8 NVFP4 keeps its BF16 dense and NVFP4 experts, and
        // only its FP8 MTP experts lose the scheme.
        let mut nvfp4 = Architecture::from_json(QWEN).expect("Qwen3.8 NVFP4");
        let mut before = nvfp4.clone();
        assert_eq!(before.quantization.mtp_expert_format, Some(tile(true)));
        nvfp4.weight_only_fp8();
        assert_eq!(nvfp4.quantization.mtp_expert_format, Some(tile(false)));
        before.quantization.mtp_expert_format = Some(tile(false));
        assert_eq!(nvfp4, before);
    }

    fn count<F: Fn(&Layer) -> bool>(architecture: &Architecture, predicate: F) -> usize {
        architecture
            .layers
            .iter()
            .filter(|layer| predicate(layer))
            .count()
    }

    const STANDARD: &str = r#"{
        "model_type": "acme_dense_v1",
        "architectures": ["AcmeForCausalLM"],
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 512,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": true,
        "eos_token_id": [2, 3],
        "bos_token_id": 1,
        "pad_token_id": 0
    }"#;

    #[test]
    fn a_standard_dense_gqa_config_needs_no_layer_types() {
        let architecture = Architecture::from_json(STANDARD).expect("a standard config");
        assert_eq!(architecture.layers.len(), 2);
        assert!(architecture.layers.iter().all(|layer| matches!(
            &layer.attention.kind,
            AttentionKind::Gqa {
                kv_heads: 2,
                output_gate: false
            }
        )));
        assert!(
            architecture
                .layers
                .iter()
                .all(|layer| matches!(&layer.ffn, Ffn::Dense(mlp) if mlp.intermediate == 256))
        );
        assert!(
            architecture
                .layers
                .iter()
                .all(|layer| layer.attention.indexer.is_none())
        );
        assert_eq!(architecture.tokens.eos_token_ids, [2, 3]);
        assert_eq!(architecture.tokens.bos_token_id, Some(1));
        assert_eq!(architecture.tokens.pad_token_id, Some(0));
    }

    #[test]
    fn an_output_gate_is_read_from_the_checkpoint_not_assumed() {
        let ungated = Architecture::from_json(STANDARD).expect("a standard config");
        assert!(matches!(
            ungated.layers[0].attention.kind,
            AttentionKind::Gqa {
                output_gate: false,
                ..
            }
        ));
        let gated = STANDARD.replace(
            "\"tie_word_embeddings\": true,",
            "\"tie_word_embeddings\": true, \"attn_output_gate\": true,",
        );
        let gated = Architecture::from_json(&gated).expect("a gated config");
        assert!(matches!(
            gated.layers[0].attention.kind,
            AttentionKind::Gqa {
                output_gate: true,
                ..
            }
        ));
    }

    #[test]
    fn a_checkpoint_directory_unions_generation_only_eos_ids() {
        let directory = std::env::temp_dir().join(format!(
            "sparkflash-architecture-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&directory).expect("a temp directory");
        std::fs::write(directory.join("config.json"), STANDARD).expect("config.json");
        std::fs::write(
            directory.join("generation_config.json"),
            r#"{"eos_token_id": [2, 9, 2], "bos_token_id": 1, "pad_token_id": 0}"#,
        )
        .expect("generation_config.json");
        let architecture =
            Architecture::from_checkpoint_dir(&directory).expect("the checkpoint directory");
        assert_eq!(
            architecture.tokens.eos_token_ids,
            [2, 3, 9],
            "the config's ids keep their order; generation_config only adds new ones"
        );
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn image_and_video_aliases_supply_the_vision_delimiters() {
        let architecture = Architecture::from_json(
            r#"{
                "model_type": "acme_vl",
                "text_config": {
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "vocab_size": 256,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "head_dim": 16
                },
                "image_token_id": 10,
                "image_start_token_id": 11,
                "image_end_token_id": 12
            }"#,
        )
        .expect("a GLM-shaped generic vision codec document");
        assert_eq!(architecture.tokens.image_token_id, Some(10));
        assert_eq!(architecture.tokens.vision_start_token_id, Some(11));
        assert_eq!(architecture.tokens.vision_end_token_id, Some(12));

        let video = Architecture::from_json(
            r#"{
                "model_type": "acme_video",
                "text_config": {
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "vocab_size": 256,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "head_dim": 16
                },
                "video_start_token_id": 21,
                "video_end_token_id": 22
            }"#,
        )
        .expect("a video-only delimiter document");
        assert_eq!(video.tokens.vision_start_token_id, Some(21));
        assert_eq!(video.tokens.vision_end_token_id, Some(22));
    }

    #[test]
    fn the_released_kv_plans_are_read_from_the_architecture() {
        let qwen = Architecture::from_json(QWEN).expect("the released Qwen config");
        assert_eq!(
            qwen.kv_plan(1, false)
                .expect("Qwen has no compressed cache")
                .sequence_bytes(262_144),
            8_078_229_504,
            "Qwen3.8-Flash-Next's released TP=1 claim"
        );
        let glm = Architecture::from_json(GLM).expect("the released GLM config");
        let fp8 = glm.kv_plan(1, true).expect("GLM has no compressed cache");
        assert_eq!(
            fp8.per_token(),
            20_160,
            "eleven sparse layers plus the MTP stage, each 528 latent + 1,152 indexer bytes"
        );
        // The f32 latent is 4,096 bytes a sparse row and the same in the MTP stage.
        let f32_latent = glm.kv_plan(1, false).expect("GLM has no compressed cache");
        assert_eq!(
            f32_latent.per_token(),
            38_400,
            "eleven sparse layers and the MTP stage each move to the f32 latent"
        );
        assert_eq!(
            f32_latent.per_token(),
            fp8.per_token() + 12 * (4 * 512 - 528)
        );
        assert_eq!(fp8.fixed(), f32_latent.fixed());
        // Two ranks split the two KV heads, so the per-position claim halves and the TP>1
        // Qwen rank turns the MTP stage off (DFlash drafts instead).
        assert!(
            qwen.kv_plan(2, false)
                .expect("Qwen has no compressed cache")
                .per_token()
                < qwen
                    .kv_plan(1, false)
                    .expect("Qwen has no compressed cache")
                    .per_token()
        );
    }

    #[test]
    fn the_released_kv_plan_is_exact_at_each_world_size() {
        let qwen = Architecture::from_json(QWEN).expect("the released Qwen config");
        assert!(
            !qwen.mtp.as_ref().expect("an MTP stage").replicated,
            "Qwen drafts with DFlash at TP>1 and clears the nested mtp.hybrid stage"
        );
        // At TP1 KV heads split across the world, the GDN value heads split the same way, and the
        // MTP stage follows the last GQA layer. At TP>1 the stage is absent, so the TP2/TP4
        // claim is the backbone's 15,360 bytes a position.
        for (world, per_token, fixed) in [
            (1usize, 29_952u64, 226_492_416u64),
            (2, 15_360, 113_246_208),
            (4, 15_360, 56_623_104),
        ] {
            let plan = qwen
                .kv_plan(world, false)
                .expect("Qwen has no compressed cache");
            assert_eq!(plan.per_token(), per_token, "Qwen TP{world} per-token");
            assert_eq!(plan.fixed(), fixed, "Qwen TP{world} fixed");
        }

        let glm = Architecture::from_json(GLM).expect("the released GLM config");
        assert!(
            glm.mtp.as_ref().expect("an MTP stage").replicated,
            "GLM keeps its MTP cache at every world size"
        );
        // The absorbed MLA latent and its indexer are replicated across ranks, so only the KDA
        // heads split.
        for (world, fixed) in [(1usize, 295_239_680u64), (2, 147_619_840), (4, 73_809_920)] {
            let plan = glm
                .kv_plan(world, true)
                .expect("GLM has no compressed cache");
            assert_eq!(plan.per_token(), 20_160, "GLM TP{world} per-token");
            assert_eq!(plan.fixed(), fixed, "GLM TP{world} fixed");
        }
    }

    #[test]
    fn a_glm_shaped_nested_mtp_block_stays_replicated() {
        // GLM's backbone is hybrid, so a nested stage block can carry `hybrid: true` without
        // Qwen's dedicated-embeddings marker. Only the marker means the stage's cache is not on
        // every rank; the bare hybrid flag must not flip the default.
        let mut document: Value = serde_json::from_str(GLM).expect("the carried GLM document");
        document
            .get_mut("text_config")
            .and_then(Value::as_object_mut)
            .expect("the GLM text configuration")
            .insert("mtp".to_owned(), serde_json::json!({"hybrid": true}));
        let glm = Architecture::from_json(&document.to_string()).expect("the nested GLM config");
        assert!(
            glm.mtp.as_ref().expect("an MTP stage").replicated,
            "a nested mtp block without Qwen's marker keeps the replicated default"
        );

        // The marker is Qwen's, so the same nested block with it clears the replication.
        document
            .get_mut("text_config")
            .and_then(Value::as_object_mut)
            .expect("the GLM text configuration")
            .insert(
                "mtp_use_dedicated_embeddings".to_owned(),
                Value::Bool(false),
            );
        let qwen_shaped =
            Architecture::from_json(&document.to_string()).expect("the marker-bearing config");
        assert!(
            !qwen_shaped.mtp.as_ref().expect("an MTP stage").replicated,
            "Qwen's marker identifies a stage every rank does not own"
        );

        // An explicit field wins over both heuristics.
        document
            .get_mut("text_config")
            .and_then(Value::as_object_mut)
            .expect("the GLM text configuration")
            .insert(
                "mtp".to_owned(),
                serde_json::json!({"hybrid": true, "replicated": true}),
            );
        let explicit =
            Architecture::from_json(&document.to_string()).expect("the explicit nested config");
        assert!(
            explicit.mtp.as_ref().expect("an MTP stage").replicated,
            "an explicit mtp.replicated=true is honored"
        );
    }

    #[test]
    fn an_mtp_stage_multiplies_by_the_release_stage_count() {
        let mut architecture = Architecture::from_json(GLM).expect("the released GLM config");
        let one = architecture
            .kv_plan(1, true)
            .expect("GLM has no compressed cache");
        architecture.mtp.as_mut().expect("an MTP stage").stages = 3;
        let three = architecture
            .kv_plan(1, true)
            .expect("GLM has no compressed cache");
        let per_stage = one.per_token() - 11 * (528 + 1_152);
        assert_eq!(three.per_token(), one.per_token() + 2 * per_stage);
        assert_eq!(three.fixed(), one.fixed());
    }

    #[test]
    fn a_compressed_cache_is_refused_rather_than_miscounted() {
        let deepseek = Architecture::from_json(DEEPSEEK).expect("the DeepSeek config");
        let error = deepseek
            .kv_plan(1, false)
            .expect_err("CSA2 is not one latent row");
        assert!(error.to_string().contains("CSA2"), "{error}");
    }

    #[test]
    fn an_indexer_that_reads_another_layers_keys_counts_nothing() {
        let shared = Architecture::from_json(
            r#"{
                "model_type": "acme_shared_indexer",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "vocab_size": 256,
                "max_position_embeddings": 64,
                "layer_types": ["full_attention", "full_attention"],
                "kv_source_layer_ids": [0],
                "index_n_heads": 2,
                "index_head_dim": 8,
                "index_topk": 4,
                "index_kpool": 2
            }"#,
        )
        .expect("a synthetic shared indexer");
        assert!(
            shared.layers[0]
                .attention
                .indexer
                .as_ref()
                .expect("layer 0 owns its indexer")
                .owns_keys
        );
        assert!(
            !shared.layers[1]
                .attention
                .indexer
                .as_ref()
                .expect("layer 1 reads layer 0's index")
                .owns_keys
        );
        // Two GQA layers at 128 bytes each, plus layer 0's 8-wide key at two bytes an element.
        assert_eq!(
            shared
                .kv_plan(1, false)
                .expect("a plain GQA cache")
                .per_token(),
            2 * 128 + 16
        );
    }

    #[test]
    fn a_compressed_layer_that_reads_another_layers_cache_does_not_own_keys() {
        let shared = Architecture::from_json(
            r#"{
                "model_type": "acme_shared_compression",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "vocab_size": 256,
                "max_position_embeddings": 64,
                "layer_types": ["compressed_attention", "compressed_attention"],
                "compress_ratios": [4, 4],
                "kv_source_layer_ids": [0]
            }"#,
        )
        .expect("a synthetic shared compression");
        assert!(
            shared.layers[0]
                .attention
                .compression
                .as_ref()
                .expect("layer 0 owns its cache")
                .owns_keys
        );
        assert!(
            !shared.layers[1]
                .attention
                .compression
                .as_ref()
                .expect("layer 1 reads layer 0's cache")
                .owns_keys
        );
    }

    #[test]
    fn an_mla_latent_counts_its_rope_width() {
        let mla = Architecture::from_json(
            r#"{
                "model_type": "acme_mla",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
                "vocab_size": 256,
                "max_position_embeddings": 64,
                "layer_types": ["full_attention"]
            }"#,
        )
        .expect("an MLA shape");
        assert_eq!(mla.layers[0].attention.rope.rotated_dims, 64);
        assert_eq!(
            mla.kv_plan(1, false)
                .expect("a plain MLA cache")
                .per_token(),
            4 * 576,
            "the f32 latent keeps the 64-wide rope part beside the 512-wide latent"
        );
    }

    #[test]
    fn a_vision_config_is_structural_not_a_producer_string() {
        let geometry = crate::vision_geometry::VisionGeometry::from_value(
            &serde_json::json!({
                "model_type": "unregistered_vision",
                "depth": 3,
                "hidden_size": 96,
                "num_heads": 4,
                "intermediate_size": 192,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "num_position_embeddings": 16,
                "rope_theta": 10000.0,
                "in_channels": 3
            }),
            128,
        )
        .expect("a structural tower");
        assert_eq!(
            geometry.out_hidden_size, 128,
            "a tower with no merger width hands the language model its own width"
        );
        assert_eq!(geometry.patch_size, 16);
        assert_eq!(geometry.depth, 3);
        assert!(geometry.validate().is_ok());
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
                "weights": {"num_bits": 4, "type": "float", "group_size": 32},
                "targets": [
                    "model.language_model.layers.0.mlp.experts",
                    "model.language_model.layers.1.mlp.experts",
                ],
            }},
        });
        let architecture =
            Architecture::from_json(&document.to_string()).expect("the released Qwen config");
        assert_eq!(architecture.quantization.expert.as_deref(), Some("NVFP4"));
        assert_eq!(
            architecture.quantization.expert_format,
            WeightFormat::NvFp4 { group_size: 32 },
            "the group's own group_size, not weight_block_size or a default 16"
        );
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
        let vision = architecture
            .vision
            .as_ref()
            .expect("a released vision tower");
        assert_eq!(
            vision.spatial_merge, 3,
            "DeepSeek's downsample_ratio is the spatial merge factor"
        );
        assert_eq!(
            vision.out_hidden_size, 5120,
            "a tower with no merger width hands the language model its width"
        );
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
    fn the_released_qwen_rope_is_read_from_rope_parameters() {
        let architecture = Architecture::from_json(QWEN).expect("Qwen config");
        let layer = &architecture.layers[0];
        assert_eq!(
            layer.attention.rope.theta, 10_000_000.0,
            "Qwen stores the base under rope_parameters"
        );
        assert_eq!(
            layer.attention.rope.rotated_dims, 64,
            "partial_rotary_factor 0.25 of a 256-wide head"
        );
    }

    #[test]
    fn a_legacy_type_key_selects_the_rope_rule() {
        let document = STANDARD.replace(
            "\"rope_theta\": 10000.0,",
            "\"rope_theta\": 10000.0, \"rope_scaling\": {\"type\": \"yarn\", \"factor\": 4.0, \"original_max_position_embeddings\": 2048},",
        );
        let architecture = Architecture::from_json(&document).expect("a legacy yarn document");
        let scaling = architecture.layers[0]
            .attention
            .rope
            .scaling
            .as_ref()
            .expect("the scaling is parsed");
        assert_eq!(scaling.kind, "yarn", "the historical type key is the rule");
        assert_eq!(scaling.factor, 4.0);
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
    fn a_document_without_a_schedule_defaults_to_full_attention() {
        // No layer_types, no compress_ratios and no full_attention_interval is the plain
        // dense/GQA release: every layer is full attention, and the parser says so rather than
        // refusing a document every Llama-shaped checkpoint uses.
        let architecture = Architecture::from_json(
            r#"{"model_type":"empty","text_config":{"hidden_size":8,"intermediate_size":16,"vocab_size":8,"num_hidden_layers":1,"num_attention_heads":1,"head_dim":8}}"#,
        )
        .expect("an all-full schedule");
        assert!(matches!(
            architecture.layers[0].attention.kind,
            AttentionKind::Gqa { .. }
        ));
    }

    #[test]
    fn a_qwen_shaped_document_defaults_its_interval_to_four() {
        // No layer_types and no full_attention_interval, but the linear-attention geometry is
        // present: this is the Qwen shape, and its default interval is 4. The shared parser and
        // Qwen4ExpConfig::from_json now agree instead of one saying all-full and the other 4.
        let mut root: Value = serde_json::from_str(QWEN).expect("Qwen config");
        root["text_config"]
            .as_object_mut()
            .expect("a text object")
            .remove("layer_types");
        root["text_config"]
            .as_object_mut()
            .expect("a text object")
            .remove("full_attention_interval");
        let architecture = Architecture::from_value(&root).expect("a Qwen-shaped schedule");
        assert_eq!(
            architecture
                .layers
                .iter()
                .filter(|layer| matches!(layer.attention.kind, AttentionKind::Gdn(_)))
                .count(),
            36,
            "three linear layers a block of four"
        );
        assert!(matches!(
            architecture.layers[3].attention.kind,
            AttentionKind::Gqa { .. }
        ));
    }

    #[test]
    fn a_full_attention_interval_derives_the_reference_schedule() {
        // The reference rule (also Qwen4ExpConfig's when it has to derive): one-based multiples
        // of the interval are full/sparse, every other layer is linear.
        let document = r#"{
            "model_type": "acme_hybrid_v1",
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 8,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 512,
            "max_position_embeddings": 64,
            "full_attention_interval": 4
        }"#;
        let architecture = Architecture::from_json(document).expect("an interval schedule");
        let full = architecture
            .layers
            .iter()
            .filter(|layer| matches!(layer.attention.kind, AttentionKind::Gqa { .. }))
            .map(|layer| layer.index)
            .collect::<Vec<_>>();
        assert_eq!(full, [3, 7], "every fourth layer is full attention");
        assert_eq!(
            architecture
                .layers
                .iter()
                .filter(|layer| matches!(layer.attention.kind, AttentionKind::Gdn(_)))
                .count(),
            6,
            "the rest are the reference linear attention, not a silent all-GQA schedule"
        );
    }

    #[test]
    fn a_standard_dense_config_derives_head_dim_and_rotates_the_whole_head() {
        // A Llama-3-shaped document: no head_dim, no qk_rope_head_dim, no
        // partial_rotary_factor.
        let document = r#"{
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 2,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "max_position_embeddings": 8192,
            "rope_theta": 500000.0,
            "eos_token_id": 128001
        }"#;
        let architecture = Architecture::from_json(document).expect("a Llama-shaped config");
        let layer = &architecture.layers[0];
        assert_eq!(layer.attention.head_dim, 128, "4096 / 32");
        assert_eq!(
            layer.attention.rope.rotated_dims, 128,
            "an undeclared rope width rotates the full head, not NoPE"
        );
        assert!(matches!(
            layer.attention.kind,
            AttentionKind::Gqa { kv_heads: 8, .. }
        ));
        assert!(matches!(layer.ffn, Ffn::Dense(_)));
    }

    #[test]
    fn the_declarations_the_parser_drops_are_named() {
        let llama: Value = serde_json::from_str(
            r#"{"model_type": "llama", "hidden_size": 64, "intermediate_size": 128,
                "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
                "vocab_size": 32}"#,
        )
        .expect("JSON");
        assert!(Architecture::unmodelled_features(&llama).is_empty());

        let mut gemma = llama.clone();
        gemma["attn_logit_softcapping"] = serde_json::json!(50.0);
        gemma["final_logit_softcapping"] = serde_json::json!(30.0);
        let found = Architecture::unmodelled_features(&gemma);
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(found[0].starts_with("attn_logit_softcapping 50"));

        let mut mixtral = llama.clone();
        mixtral["num_local_experts"] = serde_json::json!(8);
        mixtral["num_experts_per_tok"] = serde_json::json!(2);
        let found = Architecture::unmodelled_features(&mixtral);
        assert!(found[0].contains("num_local_experts 8"), "{found:?}");
        // A document that also names the count the parser reads is modelled.
        mixtral["num_experts"] = serde_json::json!(8);
        assert!(Architecture::unmodelled_features(&mixtral).is_empty());

        let mut awq = llama.clone();
        awq["quantization_config"] = serde_json::json!({"quant_method": "awq", "bits": 4});
        let found = Architecture::unmodelled_features(&awq);
        assert!(found[0].contains("\"awq\""), "{found:?}");

        let mut fp8 = llama;
        fp8["quantization_config"] =
            serde_json::json!({"quant_method": "fp8", "weight_block_size": [128, 128]});
        assert!(Architecture::unmodelled_features(&fp8).is_empty());
    }

    #[test]
    fn the_released_documents_declare_nothing_unmodelled() {
        for text in [
            include_str!("../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.hf.json"),
            include_str!("../../sparkflash-model-glm5-next/config/glm53_flash.hf.json"),
            include_str!(
                "../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json"
            ),
        ] {
            let root: Value = serde_json::from_str(text).expect("released JSON");
            assert!(
                Architecture::unmodelled_features(&root).is_empty(),
                "{:?}",
                Architecture::unmodelled_features(&root)
            );
        }
    }

    #[test]
    fn a_ragged_or_zero_head_count_refuses_the_derived_width() {
        let ragged = r#"{
            "model_type": "ragged",
            "hidden_size": 100,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 3,
            "vocab_size": 128,
            "max_position_embeddings": 16
        }"#;
        let error = Architecture::from_json(ragged).expect_err("100 / 3 is not a head width");
        assert!(error.to_string().contains("divisible"), "{error}");

        let zero = r#"{
            "model_type": "zero_heads",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 0,
            "head_dim": 16,
            "vocab_size": 128,
            "max_position_embeddings": 16
        }"#;
        let error = Architecture::from_json(zero).expect_err("zero query heads");
        assert!(error.to_string().contains("zero query heads"), "{error}");
    }

    #[test]
    fn the_released_weight_formats_are_typed_once() {
        let deepseek = Architecture::from_json(DEEPSEEK).expect("DeepSeek config");
        assert!(matches!(
            deepseek.quantization.dense_format,
            WeightFormat::MxFp8 { block: 32 }
        ));
        assert!(matches!(
            deepseek.quantization.expert_format,
            WeightFormat::MxFp4 { block: 32 }
        ));
        assert_eq!(deepseek.quantization.mtp_expert_format, None);

        let glm = Architecture::from_json(GLM).expect("GLM config");
        assert!(matches!(
            glm.quantization.dense_format,
            WeightFormat::Fp8Tile {
                block_m: 128,
                block_n: 128,
                ..
            }
        ));
        assert_eq!(
            glm.quantization.expert_format, glm.quantization.dense_format,
            "GLM names no expert format, so its experts are the dense FP8 format"
        );

        let qwen = Architecture::from_json(QWEN).expect("Qwen config");
        assert_eq!(qwen.quantization.dense_format, WeightFormat::Bf16);
        assert!(matches!(
            qwen.quantization.expert_format,
            WeightFormat::NvFp4 { group_size: 16 }
        ));
        assert!(matches!(
            qwen.quantization.mtp_expert_format,
            Some(WeightFormat::Fp8Tile { .. })
        ));
    }

    #[test]
    fn a_ue8m0_scale_selects_the_mx_formats() {
        assert!(matches!(
            WeightFormat::from_name(Some("fp8"), Some(32), Some("ue8m0")),
            WeightFormat::MxFp8 { block: 32 }
        ));
        assert!(matches!(
            WeightFormat::from_name(Some("fp4"), Some(32), Some("ue8m0")),
            WeightFormat::MxFp4 { block: 32 }
        ));
        assert!(matches!(
            WeightFormat::from_name(Some("fp4"), None, None),
            WeightFormat::NvFp4 { group_size: 16 }
        ));
        assert!(matches!(
            WeightFormat::from_name(Some("mxfp4"), Some(32), None),
            WeightFormat::MxFp4 { block: 32 }
        ));
        assert!(matches!(
            WeightFormat::from_name(Some("fp8"), Some(128), Some("e4m3")),
            WeightFormat::Fp8Tile {
                block_m: 128,
                block_n: 128,
                scale_dtype: ScaleDType::F32,
                ..
            }
        ));
    }

    #[test]
    fn a_vision_config_derives_its_merger_width_and_refuses_a_missing_patch() {
        let without_out = r#"{
            "model_type": "acme_vl",
            "text_config": {
                "hidden_size": 64,
                "intermediate_size": 128,
                "vocab_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "head_dim": 16
            },
            "vision_config": {
                "depth": 2,
                "hidden_size": 96,
                "num_heads": 4,
                "intermediate_size": 192,
                "patch_size": 16,
                "spatial_merge_size": 2
            }
        }"#;
        let architecture = Architecture::from_json(without_out).expect("a structural tower");
        let vision = architecture.vision.expect("a vision tower");
        assert_eq!(
            vision.out_hidden_size, 64,
            "defaults to the language model's hidden_size, not the tower's"
        );
        assert_eq!(vision.patch_size, 16);
        assert_eq!(vision.spatial_merge, 2);

        let without_patch = without_out.replace("\"patch_size\": 16,", "");
        let error = Architecture::from_json(&without_patch).expect_err("a missing patch side");
        assert!(error.to_string().contains("patch_size"), "{error}");
    }
}

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

//! The released `nvidia/Qwen3.8-Flash-Next-NVFP4` configuration.
//!
//! The checkpoint is a multimodal `Qwen4ExpForConditionalGeneration`. This pack serves the text
//! language model: the vision tower and the MTP stage are planned (so the byte identity against
//! the real index holds) but never loaded by the device arm.
//!
//! Every field here is read from the released `config.json`; the parser is deliberately strict
//! about the architecture invariants the reference implementation checks in
//! `Qwen4ExpTextConfig.validate_architecture` (`transformers/models/qwen4_exp/configuration_qwen4_exp.py`),
//! because a checkpoint that violates them cannot run correctly even if the shapes line up.
//!
//! # What the architecture is
//!
//! 48 decoder layers in 12 blocks of `3 x (Gated DeltaNet -> MoE) + 1 x (Qwen Sparse Attention -> MoE)`.
//! Every branch is wrapped in a **gated residual** ("hyper-connection"): the residual stream is
//! widened to `hc_count = 4` copies of the 2560-wide hidden state, each branch reads a
//! data-dependent mixture of the four streams, and writes back a per-branch scalar injection.
//! Layer 2 (one-indexed; index 1) additionally carries the **n-gram PLE**: 16 hashed n-gram
//! heads (8 bigram + 8 trigram) over a 320,001,536-row FP8 table, projected to a shared value and
//! one key per residual stream.

use serde::Deserialize;

/// The end-of-text id the chat template and the PLE segment rule use.
pub const ENDOFTEXT_TOKEN: u32 = 248_044;

/// `<|im_end|>`, appended to the stop set the way the Qwen pack does.
pub const IM_END_TOKEN: u32 = 248_046;

/// Which attention a decoder layer runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated DeltaNet (linear attention): depthwise causal convolution, then the delta rule.
    Linear,
    /// Qwen Sparse Attention: GQA with an output gate and the QSA block indexer.
    Sparse,
}

/// One n-gram head's bucket range in the PLE table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramHead {
    /// The bucket modulus: a prime above `ngram_vocab_size_base`.
    pub vocab_size: u64,
    /// The row offset of this head's bucket range in the concatenated table.
    pub offset: u64,
}

pub use sparkflash_model::ConfigError;

/// The stored format of the routed experts.
///
/// The released NVFP4 checkpoint stores them as packed E2M1 with E4M3 block scales and two
/// per-tensor F32 scales; the official FP8 checkpoint stores the same experts as E4M3 codes with
/// one BF16 scale a 128x128 tile. Every other tensor of the two checkpoints is BF16 (and the PLE
/// table is FP8 in both), so the expert format is the whole difference the loader sees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExpertQuant {
    /// ModelOpt NVFP4: two E2M1 nibbles a byte, one E4M3 scale a 16-column group.
    #[default]
    Nvfp4,
    /// E4M3 with one scale a 128x128 tile ([ScaleGrid::Tile]).
    Fp8Block,
}

impl ExpertQuant {
    /// The format the released quantisation metadata selects.
    ///
    /// # Errors
    /// A mix this pack has no loader for.
    pub fn of(quant: &QuantizationSpec) -> Result<Self, ConfigError> {
        // ModelOpt names the producer in `quant_method` and the arithmetic in `quant_algo`; the
        // released NVFP4 snapshot declares `quant_method: "modelopt"`, so reading the producer as
        // a format refused it before a weight was touched.
        let declared = quant.quant_method.as_deref().unwrap_or_default();
        let method = if declared.is_empty()
            || sparkflash_model::architecture::names_the_producer(declared)
        {
            quant.quant_algo.as_str()
        } else {
            declared
        };
        match method.to_ascii_lowercase().as_str() {
            "fp8" | "fp8_block_scales" => {
                let block = quant.weight_block_size.as_deref().unwrap_or(&[]);
                if block != [128, 128] {
                    return Err(ConfigError::Invalid(format!(
                        "the FP8 checkpoint must declare weight_block_size [128, 128], not {block:?}"
                    )));
                }
                Ok(Self::Fp8Block)
            }
            "" | "mixed_precision" | "nvfp4" => Ok(Self::Nvfp4),
            _ => Err(ConfigError::Invalid(format!(
                "the pack serves the NVFP4 and FP8 expert checkpoints, not {method:?}"
            ))),
        }
    }
}

/// The text-model configuration the device arm runs.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen4ExpConfig {
    /// Hidden width (2560).
    pub hidden_size: usize,
    /// Number of decoder layers (48).
    pub num_hidden_layers: usize,
    /// Query heads per sparse-attention layer (24).
    pub num_attention_heads: usize,
    /// Key/value heads per sparse-attention layer (2).
    pub num_key_value_heads: usize,
    /// Attention head width (256).
    pub head_dim: usize,
    /// Vocabulary size (248,320, padded).
    pub vocab_size: usize,
    /// RMSNorm epsilon (1e-6).
    pub rms_norm_eps: f64,
    /// RoPE base (10,000,000).
    pub rope_theta: f64,
    /// Fraction of each head that rotates (0.25 -> 64 of 256).
    pub partial_rotary_factor: f64,
    /// mRoPE section split of the rotated half-width (11/11/10).
    pub mrope_section: [usize; 3],
    /// Maximum positions (262,144).
    pub max_position_embeddings: usize,
    /// The attention kind of every layer, in order.
    pub layer_types: Vec<LayerKind>,
    /// The shared architecture this typed profile describes.
    ///
    /// [Self::from_json] stores the [sparkflash_model::Architecture] it parses alongside the
    /// typed fields, so every caller reads the one walk rather than re-deriving the schedule from
    /// `layer_types`. [Self::steps] is the layer walk the device arm builds from.
    pub architecture: sparkflash_model::Architecture,
    /// The stored format of the routed experts.
    pub expert_quant: ExpertQuant,
    /// Gated DeltaNet convolution width (4).
    pub linear_conv_kernel_dim: usize,
    /// Gated DeltaNet key head width (128).
    pub linear_key_head_dim: usize,
    /// Gated DeltaNet value head width (128).
    pub linear_value_head_dim: usize,
    /// Gated DeltaNet key heads (16).
    pub linear_num_key_heads: usize,
    /// Gated DeltaNet value heads (48).
    pub linear_num_value_heads: usize,
    /// Routed-expert intermediate width (640).
    pub moe_intermediate_size: usize,
    /// Shared-expert intermediate width (640).
    pub shared_expert_intermediate_size: usize,
    /// Routed experts per layer (512).
    pub num_experts: usize,
    /// Experts a token activates (10).
    pub num_experts_per_tok: usize,
    /// Whether the top-k probabilities are renormalised (true).
    pub norm_topk_prob: bool,
    /// The activation name (silu).
    pub hidden_act: String,
    /// Output gate activation of Gated DeltaNet (sigmoid).
    pub output_gate_type: String,
    /// Residual streams per hyper-connection (4).
    pub hc_count: usize,
    /// Bottleneck rank of the hyper-connection input mixer (320).
    pub hc_lowrank: usize,
    /// One-indexed decoder layers that carry the n-gram PLE ([2]).
    pub ple_layer_ids: Vec<usize>,
    /// PLE embedding width (2560).
    pub ple_embed_dim: usize,
    /// PLE depthwise convolution width (4).
    pub ple_conv_kernel_size: usize,
    /// Largest n-gram represented (3).
    pub ngram_size: usize,
    /// Hashed heads per n-gram order (8).
    pub heads_per_ngram: usize,
    /// Lower bound of every head's bucket count (20,000,000).
    pub ngram_vocab_size_base: u64,
    /// The table's padded vocabulary is a multiple of this (128).
    pub make_ngram_vocab_size_divisible_by: u64,
    /// Seed of the per-head hash multipliers (1234).
    pub seed: u64,
    /// Checkpoint shards the PLE table is stored in (128).
    pub split_ngram_parts: usize,
    /// QSA indexer query heads (4).
    pub indexer_n_heads: usize,
    /// QSA indexer key heads (1).
    pub indexer_kv_heads: usize,
    /// QSA indexer head width (128).
    pub indexer_head_dim: usize,
    /// QSA token budget (2048).
    pub indexer_budget: usize,
    /// Tokens averaged into one QSA block (4).
    pub indexer_compress_ratio: usize,
    /// Whether projections carry a bias (false).
    pub attention_bias: bool,
    /// Whether the LM head is tied to the embedding (false).
    pub tie_word_embeddings: bool,
    /// End-of-generation ids: the config's EOS plus `<|im_end|>`.
    pub eos_token_ids: Vec<u32>,
    /// Beginning-of-sequence id.
    pub bos_token_id: Option<u32>,
    /// Padding id (unset).
    pub pad_token_id: Option<u32>,
    /// Image placeholder id.
    pub image_token_id: u32,
    /// Vision span start id.
    pub vision_start_token_id: u32,
    /// Vision span end id.
    pub vision_end_token_id: u32,
    /// MTP layers in the checkpoint (1, not served).
    pub mtp_num_hidden_layers: usize,
    /// Whether the checkpoint carries a vision tower (true, not served).
    pub has_vision: bool,
    /// Vision tower depth (27).
    pub vision_depth: Option<usize>,
    /// Vision tower width (1152).
    pub vision_hidden_size: Option<usize>,
    /// Vision patch side in pixels (16).
    pub vision_patch_size: Option<usize>,
    /// Temporal patch depth (2; a still image duplicates its frame).
    pub vision_temporal_patch_size: Option<usize>,
    /// Spatial merge per side feeding the language model (2).
    pub vision_spatial_merge_size: Option<usize>,
    /// Vision attention heads (16).
    pub vision_num_heads: Option<usize>,
    /// Vision MLP intermediate width (4304).
    pub vision_intermediate_size: Option<usize>,
    /// Vision merger output width (2560).
    pub vision_out_hidden_size: Option<usize>,
    /// Learned position grid size (2304 = 48 x 48).
    pub vision_num_position_embeddings: Option<usize>,
    /// Vision axial RoPE base (10,000).
    pub vision_rope_theta: Option<f64>,
}

const CARRIED_CONFIG: &str = include_str!("../config/qwen38_flash_next_nvfp4.hf.json");
const CARRIED_FP8_CONFIG: &str = include_str!("../config/qwen38_flash_next_fp8.hf.json");

impl Qwen4ExpConfig {
    /// The released `nvidia/Qwen3.8-Flash-Next-NVFP4` configuration carried in this crate.
    ///
    /// # Errors
    /// [ConfigError] only if the carried document were edited into an unsupported shape, which the
    /// crate's own test rules out.
    pub fn qwen38_flash_next_nvfp4() -> Result<Self, ConfigError> {
        Self::from_json(CARRIED_CONFIG)
    }

    /// The released `Qwen/Qwen3.8-Flash-Next-FP8` configuration carried in this crate.
    ///
    /// The architecture is the NVFP4 checkpoint's; only the routed experts' storage differs
    /// (E4M3 codes with BF16 128x128 tiles rather than packed E2M1 with E4M3 groups).
    ///
    /// # Errors
    /// [ConfigError] only if the carried document were edited into an unsupported shape.
    pub fn qwen38_flash_next_fp8() -> Result<Self, ConfigError> {
        Self::from_json(CARRIED_FP8_CONFIG)
    }

    /// Parse a released HF `config.json`.
    ///
    /// # Errors
    /// [ConfigError::Json] for a malformed document and [ConfigError::Invalid] for a shape this
    /// pack cannot serve.
    pub fn from_json(text: &str) -> Result<Self, ConfigError> {
        let architecture = crate::architecture::architecture_from_json(text)?;
        Self::from_json_with_architecture(text, architecture)
    }

    /// Parse a released HF `config.json` whose shared architecture was already read.
    ///
    /// The serving path reads the architecture through
    /// [sparkflash_model::Architecture::from_checkpoint_dir], so `generation_config.json`
    /// contributes the stop set, and passes it here rather than letting this function parse a
    /// second, generation-config-less view.
    ///
    /// # Errors
    /// As [Self::from_json].
    pub fn from_json_with_architecture(
        text: &str,
        architecture: sparkflash_model::Architecture,
    ) -> Result<Self, ConfigError> {
        let mut architecture = architecture;
        let root: RootConfig =
            serde_json::from_str(text).map_err(|error| ConfigError::Json(error.to_string()))?;
        let text = root.text_config;
        let layer_types = match text.layer_types {
            Some(types) if !types.is_empty() => types
                .iter()
                .map(|kind| match kind.as_str() {
                    "linear_attention" => Ok(LayerKind::Linear),
                    // The released checkpoint spells the sparse layers `full_attention`; the
                    // reference config rewrites that spelling to `indexed_attention`.
                    "indexed_attention" | "full_attention" => Ok(LayerKind::Sparse),
                    other => Err(ConfigError::Invalid(format!(
                        "unknown layer type {other:?}; expected linear_attention or full_attention"
                    ))),
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => {
                let interval = text.full_attention_interval.unwrap_or(4);
                if interval == 0 {
                    return Err(ConfigError::Invalid(
                        "full_attention_interval must not be zero".to_owned(),
                    ));
                }
                (0..text.num_hidden_layers)
                    .map(|index| {
                        if (index + 1) % interval == 0 {
                            LayerKind::Sparse
                        } else {
                            LayerKind::Linear
                        }
                    })
                    .collect()
            }
        };
        let rope = text.rope_parameters;
        let partial_rotary_factor = rope
            .partial_rotary_factor
            .or(text.partial_rotary_factor)
            .unwrap_or(1.0);
        let mrope_section = rope.mrope_section.unwrap_or([11, 11, 10]);
        // A fine-tune may omit `rope_parameters.rope_theta` and inherit the release's base.
        // The shared architecture is the one parser that decides that default (10,000,000 for
        // Qwen's hybrid linear-attention documents, 10,000 otherwise); following it here keeps
        // the two views equal instead of parsing a second, silent default.
        let rope_theta = rope.rope_theta.unwrap_or_else(|| {
            architecture
                .layers
                .first()
                .map_or(10_000.0, |layer| layer.attention.rope.theta)
        });
        // One stop set: the checkpoint's. `Architecture::from_checkpoint_dir` has already
        // unioned `generation_config.json` onto the config's ids, and the config's order is
        // first. Qwen's own config names only <|endoftext|>; the pack also stops on
        // <|im_end|>, which the reference appends when the vocabulary can hold it. Push the
        // same list back into the shared architecture so the runtime and the codec read one
        // list rather than two stop sets that happen to agree today.
        let mut eos_token_ids = architecture.tokens.eos_token_ids.clone();
        if text.vocab_size > IM_END_TOKEN as usize && !eos_token_ids.contains(&IM_END_TOKEN) {
            eos_token_ids.push(IM_END_TOKEN);
        }
        architecture.tokens.eos_token_ids = eos_token_ids.clone();
        let quant = root.quantization_config.unwrap_or_default();
        let expert_quant = ExpertQuant::of(&quant)?;
        let mut ple_layer_ids = text.ple_layer_ids.unwrap_or_default();
        ple_layer_ids.sort_unstable();
        ple_layer_ids.dedup();
        let config = Self {
            hidden_size: text.hidden_size,
            num_hidden_layers: text.num_hidden_layers,
            num_attention_heads: text.num_attention_heads,
            num_key_value_heads: text.num_key_value_heads,
            head_dim: text.head_dim,
            vocab_size: text.vocab_size,
            rms_norm_eps: text.rms_norm_eps,
            rope_theta,
            partial_rotary_factor,
            mrope_section,
            max_position_embeddings: text.max_position_embeddings,
            layer_types,
            architecture,
            expert_quant,
            linear_conv_kernel_dim: text.linear_conv_kernel_dim,
            linear_key_head_dim: text.linear_key_head_dim,
            linear_value_head_dim: text.linear_value_head_dim,
            linear_num_key_heads: text.linear_num_key_heads,
            linear_num_value_heads: text.linear_num_value_heads,
            moe_intermediate_size: text.moe_intermediate_size,
            shared_expert_intermediate_size: text.shared_expert_intermediate_size,
            num_experts: text.num_experts,
            num_experts_per_tok: text.num_experts_per_tok,
            norm_topk_prob: text.norm_topk_prob,
            hidden_act: text.hidden_act,
            output_gate_type: text.output_gate_type.unwrap_or_else(|| "silu".to_owned()),
            hc_count: text.hc_count,
            hc_lowrank: text.hc_lowrank,
            ple_layer_ids,
            ple_embed_dim: text.ple_embed_dim.unwrap_or(text.hidden_size),
            ple_conv_kernel_size: text.ple_conv_kernel_size,
            ngram_size: text.ngram_size,
            heads_per_ngram: text.heads_per_ngram,
            ngram_vocab_size_base: text.ngram_vocab_size_base,
            make_ngram_vocab_size_divisible_by: text.make_ngram_vocab_size_divisible_by,
            seed: text.seed,
            split_ngram_parts: text.split_ngram_parts,
            indexer_n_heads: text.indexer_n_heads.unwrap_or(0),
            indexer_kv_heads: text.indexer_kv_heads.unwrap_or(0),
            indexer_head_dim: text.indexer_head_dim.unwrap_or(0),
            indexer_budget: text.indexer_budget.unwrap_or(0),
            indexer_compress_ratio: text.indexer_compress_ratio.unwrap_or(0),
            attention_bias: text.attention_bias,
            tie_word_embeddings: root.tie_word_embeddings,
            eos_token_ids,
            bos_token_id: text.bos_token_id,
            pad_token_id: text.pad_token_id,
            image_token_id: root.image_token_id,
            vision_start_token_id: root.vision_start_token_id,
            vision_end_token_id: root.vision_end_token_id,
            mtp_num_hidden_layers: text.mtp_num_hidden_layers,
            has_vision: root.vision_config.is_some(),
            vision_depth: root.vision_config.as_ref().map(|vision| vision.depth),
            vision_hidden_size: root.vision_config.as_ref().map(|vision| vision.hidden_size),
            vision_patch_size: root.vision_config.as_ref().map(|vision| vision.patch_size),
            vision_temporal_patch_size: root
                .vision_config
                .as_ref()
                .map(|vision| vision.temporal_patch_size),
            vision_spatial_merge_size: root
                .vision_config
                .as_ref()
                .map(|vision| vision.spatial_merge_size),
            vision_num_heads: root.vision_config.as_ref().map(|vision| vision.num_heads),
            vision_intermediate_size: root
                .vision_config
                .as_ref()
                .map(|vision| vision.intermediate_size),
            vision_out_hidden_size: root
                .vision_config
                .as_ref()
                .map(|vision| vision.out_hidden_size),
            vision_num_position_embeddings: root
                .vision_config
                .as_ref()
                .map(|vision| vision.num_position_embeddings),
            vision_rope_theta: root.vision_config.as_ref().map(|vision| vision.rope_theta),
        };
        config.validate()?;
        config.cross_check(&config.architecture)?;
        // The NVFP4 checkpoint is only usable through the quantisation metadata below; a
        // different mix (say an FP8 checkpoint) would need a different loader.
        quant.validate()?;
        Ok(config)
    }

    /// Parse a checkpoint directory, merging its `generation_config.json`.
    ///
    /// This is the serving path's constructor: the typed profile and the shared architecture
    /// both read the same merged stop set, so the codec's validation and the runtime's stop
    /// check cannot disagree about which ids end a generation.
    ///
    /// # Errors
    /// A read or parse failure, or a document this pack cannot serve.
    pub fn from_checkpoint_dir(directory: &std::path::Path) -> Result<Self, ConfigError> {
        let architecture = sparkflash_model::Architecture::from_checkpoint_dir(directory)?;
        let path = directory.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|error| ConfigError::Json(format!("{}: {error}", path.display())))?;
        Self::from_json_with_architecture(&text, architecture)
    }

    /// The tiny reference configuration, stored as the FP8 checkpoint's experts.
    ///
    /// The expert kernels read whole 128-column tiles, so the fixture's widths are the one
    /// geometry the tiny NVFP4 fixture cannot also supply. Everything else is [Self::tiny].
    ///
    /// # Panics
    /// Only if the rendered document cannot be parsed again, which its own test rules out.
    #[must_use]
    pub fn tiny_fp8() -> Self {
        let mut config = Self::tiny();
        config.hidden_size = 128;
        config.moe_intermediate_size = 128;
        config.shared_expert_intermediate_size = 128;
        config.ple_embed_dim = 128;
        config.expert_quant = ExpertQuant::Fp8Block;
        // The shared architecture must describe the geometry the typed fields now hold; it is
        // parsed from the document the typed fields render.
        config.architecture = crate::architecture::architecture_from_json(&config.to_json())
            .expect("the tiny FP8 document parses");
        config
    }

    /// A tiny two-layer configuration for the reference tests.
    ///
    /// Every width divides evenly, the first layer is Gated DeltaNet and the second sparse
    /// attention, and the PLE sits on the linear layer. The geometry is not the released one, so
    /// the arithmetic can be checked with small matrices.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            vocab_size: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.5,
            mrope_section: [1, 1, 0],
            max_position_embeddings: 64,
            layer_types: vec![LayerKind::Linear, LayerKind::Sparse],
            architecture: crate::architecture::tiny_architecture(),
            expert_quant: ExpertQuant::Nvfp4,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_num_key_heads: 2,
            linear_num_value_heads: 4,
            moe_intermediate_size: 8,
            shared_expert_intermediate_size: 8,
            num_experts: 8,
            num_experts_per_tok: 2,
            norm_topk_prob: true,
            hidden_act: "silu".to_owned(),
            output_gate_type: "sigmoid".to_owned(),
            hc_count: 2,
            hc_lowrank: 4,
            ple_layer_ids: vec![1],
            ple_embed_dim: 16,
            ple_conv_kernel_size: 2,
            ngram_size: 3,
            heads_per_ngram: 2,
            ngram_vocab_size_base: 1_000,
            make_ngram_vocab_size_divisible_by: 8,
            seed: 1234,
            split_ngram_parts: 2,
            indexer_n_heads: 2,
            indexer_kv_heads: 1,
            indexer_head_dim: 4,
            indexer_budget: 4,
            indexer_compress_ratio: 2,
            attention_bias: false,
            tie_word_embeddings: false,
            eos_token_ids: vec![2],
            bos_token_id: None,
            pad_token_id: None,
            image_token_id: 3,
            vision_start_token_id: 4,
            vision_end_token_id: 5,
            mtp_num_hidden_layers: 1,
            has_vision: false,
            vision_depth: None,
            vision_hidden_size: None,
            vision_patch_size: None,
            vision_temporal_patch_size: None,
            vision_spatial_merge_size: None,
            vision_num_heads: None,
            vision_intermediate_size: None,
            vision_out_hidden_size: None,
            vision_num_position_embeddings: None,
            vision_rope_theta: None,
        }
    }

    /// Render this configuration back to the released HF `config.json` schema.
    ///
    /// The round trip is exact: parsing the output reproduces the same [Qwen4ExpConfig], which is
    /// what the tests pin. The quantisation block is omitted because this pack's parser treats it
    /// as optional metadata; a fixture checkpoint therefore needs only the text geometry.
    #[must_use]
    pub fn to_json(&self) -> String {
        let layer_types: Vec<&str> = self
            .layer_types
            .iter()
            .map(|kind| match kind {
                LayerKind::Linear => "linear_attention",
                LayerKind::Sparse => "full_attention",
            })
            .collect();
        let mut text = serde_json::json!({
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_parameters": {
                "rope_theta": self.rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
                "mrope_section": self.mrope_section,
            },
            "max_position_embeddings": self.max_position_embeddings,
            "layer_types": layer_types,
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
            "moe_intermediate_size": self.moe_intermediate_size,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_size,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "norm_topk_prob": self.norm_topk_prob,
            "hidden_act": self.hidden_act,
            "output_gate_type": self.output_gate_type,
            "hc_count": self.hc_count,
            "hc_lowrank": self.hc_lowrank,
            "ple_layer_ids": self.ple_layer_ids,
            "ple_embed_dim": self.ple_embed_dim,
            "ple_conv_kernel_size": self.ple_conv_kernel_size,
            "ngram_size": self.ngram_size,
            "heads_per_ngram": self.heads_per_ngram,
            "ngram_vocab_size_base": self.ngram_vocab_size_base,
            "make_ngram_vocab_size_divisible_by": self.make_ngram_vocab_size_divisible_by,
            "seed": self.seed,
            "split_ngram_parts": self.split_ngram_parts,
            "indexer_n_heads": self.indexer_n_heads,
            "indexer_kv_heads": self.indexer_kv_heads,
            "indexer_head_dim": self.indexer_head_dim,
            "indexer_budget": self.indexer_budget,
            "indexer_compress_ratio": self.indexer_compress_ratio,
            "attention_bias": self.attention_bias,
            "eos_token_id": self.eos_token_ids,
            "mtp_num_hidden_layers": self.mtp_num_hidden_layers,
        });
        // The shared architecture reads the top-level key, so the rendered document carries it
        // or the round trip would describe a different rotary width.
        text["partial_rotary_factor"] = serde_json::json!(self.partial_rotary_factor);
        // The shared parser reads the stage's per-rank ownership from the nested `mtp.hybrid`
        // shape plus Qwen's `mtp_use_dedicated_embeddings` marker; render both or a round trip
        // would turn the Qwen stage back into a replicated one.
        if self
            .architecture
            .mtp
            .as_ref()
            .is_some_and(|mtp| !mtp.replicated)
        {
            text["mtp"] = serde_json::json!({
                "hybrid": true,
                "num_hidden_layers": self.mtp_num_hidden_layers,
            });
            text["mtp_use_dedicated_embeddings"] = serde_json::json!(false);
        }
        if let Some(bos) = self.bos_token_id {
            text["bos_token_id"] = serde_json::json!(bos);
        }
        if let Some(pad) = self.pad_token_id {
            text["pad_token_id"] = serde_json::json!(pad);
        }
        let mut root = serde_json::json!({
            "model_type": "qwen4_exp",
            // The shared architecture carries the released architecture name, so the rendered
            // document names it too and the round trip keeps both parsers in step.
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "text_config": text,
            "tie_word_embeddings": self.tie_word_embeddings,
            "image_token_id": self.image_token_id,
            "vision_start_token_id": self.vision_start_token_id,
            "vision_end_token_id": self.vision_end_token_id,
        });
        if let Some(video) = self.architecture.tokens.video_token_id {
            root["video_token_id"] = serde_json::json!(video);
        }
        // The document must round-trip the format its plan was built from, so the expert format
        // is written from the typed field rather than re-derived from the shared architecture.
        let mut quant = serde_json::Map::new();
        match self.expert_quant {
            ExpertQuant::Nvfp4 => {
                let quantization = &self.architecture.quantization;
                if let Some(method) = &quantization.method {
                    quant.insert("quant_algo".to_owned(), serde_json::json!(method));
                }
                if let Some(block) = quantization.block {
                    quant.insert(
                        "weight_block_size".to_owned(),
                        serde_json::json!([block, block]),
                    );
                }
                if let Some(scale) = &quantization.scale {
                    quant.insert("fmt".to_owned(), serde_json::json!(scale));
                }
                if let Some(expert) = &quantization.expert {
                    quant.insert(
                        "expert_weight".to_owned(),
                        serde_json::json!({ "quant_algo": expert }),
                    );
                }
                if let Some(mtp) = quantization.mtp_expert_format {
                    quant.insert(
                        "mtp_expert_weight".to_owned(),
                        serde_json::json!({ "quant_algo": mtp.name() }),
                    );
                }
            }
            ExpertQuant::Fp8Block => {
                quant.insert("quant_method".to_owned(), serde_json::json!("fp8"));
                quant.insert("activation_scheme".to_owned(), serde_json::json!("dynamic"));
                quant.insert(
                    "weight_block_size".to_owned(),
                    serde_json::json!([128, 128]),
                );
            }
        }
        if !quant.is_empty() {
            root["quantization_config"] = serde_json::Value::Object(quant);
        }
        if self.has_vision {
            root["vision_config"] = serde_json::json!({
                "hidden_size": self.vision_hidden_size.unwrap_or(0),
                "depth": self.vision_depth.unwrap_or(0),
                "patch_size": self.vision_patch_size.unwrap_or(16),
                "temporal_patch_size": self.vision_temporal_patch_size.unwrap_or(2),
                "spatial_merge_size": self.vision_spatial_merge_size.unwrap_or(2),
                "num_heads": self.vision_num_heads.unwrap_or(16),
                "intermediate_size": self.vision_intermediate_size.unwrap_or(4304),
                "out_hidden_size": self.vision_out_hidden_size.unwrap_or(2560),
                "num_position_embeddings": self.vision_num_position_embeddings.unwrap_or(2304),
                "rope_theta": self.vision_rope_theta.unwrap_or(10_000.0),
            });
        }
        serde_json::to_string_pretty(&root).expect("a config of plain scalars always renders")
    }

    /// The number of sparse-attention layers.
    #[must_use]
    pub fn sparse_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|kind| **kind == LayerKind::Sparse)
            .count()
    }

    /// The number of Gated DeltaNet layers.
    #[must_use]
    pub fn linear_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|kind| **kind == LayerKind::Linear)
            .count()
    }

    /// The width of the widened residual stream: `hc_count * hidden_size`.
    #[must_use]
    pub fn hyper_width(&self) -> usize {
        self.hc_count * self.hidden_size
    }

    /// Whether the Gated DeltaNet output gate is `sigmoid` rather than `silu`.
    ///
    /// The reference builds the GDN gated norm as
    /// `Qwen4ExpTextRMSNormGated(head_v_dim, eps, activation=config.output_gate_type or
    /// config.hidden_act)`. The released checkpoint names `"sigmoid"`, so the gate is
    /// `sigmoid(z)`; `hidden_act` is `"silu"` and governs the convolutions and the experts,
    /// not this gate.
    #[must_use]
    pub fn output_gate_is_sigmoid(&self) -> bool {
        self.output_gate_type == "sigmoid"
    }

    /// The rotated width of one attention head (64).
    #[must_use]
    pub fn rotary_dim(&self) -> usize {
        ((self.head_dim as f64) * self.partial_rotary_factor).round() as usize
    }

    /// The number of query heads per key/value head (12).
    #[must_use]
    pub fn query_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Sparse-attention query projection width, query and gate halves (2 * 24 * 256).
    #[must_use]
    pub fn q_proj_width(&self) -> usize {
        self.num_attention_heads * self.head_dim * 2
    }

    /// Sparse-attention key/value projection width (2 * 256).
    #[must_use]
    pub fn kv_proj_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Sparse-attention output projection input width (24 * 256).
    #[must_use]
    pub fn attn_out_width(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// Gated DeltaNet key width (16 * 128).
    #[must_use]
    pub fn linear_key_width(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    /// Gated DeltaNet value width (48 * 128).
    #[must_use]
    pub fn linear_value_width(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// Gated DeltaNet mixed QKV width: Q, K and V concatenated (2 * 2048 + 6144).
    #[must_use]
    pub fn linear_conv_width(&self) -> usize {
        self.linear_key_width() * 2 + self.linear_value_width()
    }

    /// Gated DeltaNet value heads per key head (3).
    #[must_use]
    pub fn value_groups(&self) -> usize {
        self.linear_num_value_heads / self.linear_num_key_heads
    }

    /// Hashed PLE heads per n-gram order (8).
    #[must_use]
    pub fn heads_per_order(&self) -> usize {
        self.heads_per_ngram
    }

    /// Hashed PLE heads across every order (16).
    #[must_use]
    pub fn ngram_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// The width of one hashed head's embedding row (160).
    ///
    /// # Panics
    /// Panics if `ple_embed_dim` is not divisible by the head count; [validate](Self::validate)
    /// rejects that configuration before any caller can reach this.
    #[must_use]
    pub fn ngram_head_dim(&self) -> usize {
        self.ple_embed_dim / self.ngram_heads()
    }

    /// The indexer's block budget: `indexer_budget / indexer_compress_ratio` (512).
    #[must_use]
    pub fn indexer_block_budget(&self) -> usize {
        self.indexer_budget / self.indexer_compress_ratio.max(1)
    }

    /// The most tokens the indexer selects for one query: the budget's complete blocks and the
    /// incomplete block at the end (2,051).
    #[must_use]
    pub fn max_selected_tokens(&self) -> usize {
        self.indexer_block_budget() * self.indexer_compress_ratio
            + self.indexer_compress_ratio.saturating_sub(1)
    }

    /// The padded row count of the PLE n-gram table (320,001,536).
    #[must_use]
    pub fn ple_table_rows(&self) -> u64 {
        crate::ngram::PleLayout::new(self).padded_rows
    }

    /// The one-indexed decoder layer ids that carry PLE, as zero-indexed layer indices.
    #[must_use]
    pub fn ple_layer_indices(&self) -> Vec<usize> {
        self.ple_layer_ids.iter().map(|id| id - 1).collect()
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if self.expert_quant == ExpertQuant::Fp8Block
            && (!self.hidden_size.is_multiple_of(128)
                || !self.moe_intermediate_size.is_multiple_of(128))
        {
            return Err(ConfigError::Invalid(format!(
                "FP8 experts need 128-wide tiles: hidden {} and intermediate {} must both be \
                 multiples of 128",
                self.hidden_size, self.moe_intermediate_size
            )));
        }
        let require = |condition: bool, message: &str| {
            if condition {
                Ok(())
            } else {
                Err(ConfigError::Invalid(message.to_owned()))
            }
        };
        require(self.hidden_size > 0, "hidden_size must be positive")?;
        require(
            self.num_hidden_layers > 0,
            "num_hidden_layers must be positive",
        )?;
        require(self.vocab_size > 0, "vocab_size must be positive")?;
        require(
            self.layer_types.len() == self.num_hidden_layers,
            "layer_types must have one entry per hidden layer",
        )?;
        require(self.hc_count > 1, "Qwen4-Exp requires hc_count > 1")?;
        require(self.hc_lowrank > 0, "hc_lowrank must be positive")?;
        require(
            self.moe_intermediate_size > 0 && self.shared_expert_intermediate_size > 0,
            "expert intermediate widths must be positive",
        )?;
        require(
            self.num_experts > 0
                && self.num_experts_per_tok > 0
                && self.num_experts_per_tok <= self.num_experts,
            "num_experts_per_tok must be in 1..=num_experts",
        )?;
        require(
            self.hidden_act == "silu",
            "this pack implements the silu expert activation",
        )?;
        require(
            matches!(self.output_gate_type.as_str(), "sigmoid" | "silu"),
            "the Gated DeltaNet output gate must be sigmoid or silu",
        )?;
        // The device router kernel renormalises the top-k probabilities unconditionally, so a
        // checkpoint that turns this off would be served with a silently different MoE weighting.
        // The host arm branches on it; the device arm cannot, so refuse rather than diverge.
        require(
            self.norm_topk_prob,
            "this pack implements the renormalised (norm_topk_prob = true) router",
        )?;
        require(
            !self.attention_bias,
            "this pack implements bias-free attention projections",
        )?;
        if self.sparse_layers() > 0 {
            require(
                self.num_attention_heads > 0 && self.num_key_value_heads > 0,
                "sparse-attention layers need positive head counts",
            )?;
            require(
                self.num_attention_heads
                    .is_multiple_of(self.num_key_value_heads),
                "num_attention_heads must be a multiple of num_key_value_heads",
            )?;
            let rotary = self.rotary_dim();
            require(
                rotary > 0 && rotary <= self.head_dim && rotary.is_multiple_of(2),
                "partial_rotary_factor must yield a positive even width no wider than head_dim",
            )?;
            require(
                self.mrope_section.iter().sum::<usize>() == rotary / 2,
                "mrope_section must sum to half the rotated head width",
            )?;
            let qsa = [
                self.indexer_n_heads,
                self.indexer_kv_heads,
                self.indexer_head_dim,
                self.indexer_budget,
                self.indexer_compress_ratio,
            ];
            require(
                qsa.iter().all(|value| *value > 0),
                "QSA is enabled by this configuration, so every indexer field must be positive",
            )?;
            require(
                self.indexer_kv_heads == 1,
                "Qwen4-Exp QSA requires indexer_kv_heads=1",
            )?;
            require(
                self.indexer_budget
                    .is_multiple_of(self.indexer_compress_ratio),
                "indexer_budget must be divisible by indexer_compress_ratio",
            )?;
            require(
                rotary <= self.indexer_head_dim,
                "the attention rotary width must fit inside the QSA index head",
            )?;
        }
        if self.linear_layers() > 0 {
            require(
                self.linear_conv_kernel_dim >= 1,
                "linear_conv_kernel_dim must be at least 1",
            )?;
            require(
                self.linear_num_key_heads > 0 && self.linear_num_value_heads > 0,
                "Gated DeltaNet layers need positive head counts",
            )?;
            require(
                self.linear_key_head_dim > 0 && self.linear_value_head_dim > 0,
                "Gated DeltaNet head widths must be positive",
            )?;
            require(
                self.linear_num_value_heads
                    .is_multiple_of(self.linear_num_key_heads),
                "linear_num_value_heads must be a multiple of linear_num_key_heads",
            )?;
        }
        if !self.ple_layer_ids.is_empty() {
            require(self.ngram_size >= 2, "the n-gram PLE needs ngram_size >= 2")?;
            require(
                self.ple_embed_dim > 0 && self.ple_embed_dim.is_multiple_of(self.ngram_heads()),
                "ple_embed_dim must be divisible by (ngram_size - 1) * heads_per_ngram",
            )?;
            require(
                self.ple_layer_ids
                    .iter()
                    .all(|id| *id >= 1 && *id <= self.num_hidden_layers),
                "ple_layer_ids must be one-indexed layers inside the stack",
            )?;
            for id in &self.ple_layer_ids {
                require(
                    self.layer_types[id - 1] == LayerKind::Linear,
                    "Qwen4-Exp PLE is only supported on linear-attention layers",
                )?;
            }
            require(
                !self.eos_token_ids.is_empty(),
                "eos_token_id must be set when PLE is enabled",
            )?;
            require(
                self.split_ngram_parts > 0,
                "split_ngram_parts must be positive",
            )?;
            require(
                self.make_ngram_vocab_size_divisible_by > 0,
                "make_ngram_vocab_size_divisible_by must be positive",
            )?;
        }
        require(
            self.rope_theta.is_finite() && self.rope_theta > 0.0,
            "rope_theta must be positive and finite",
        )?;
        require(
            self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0,
            "rms_norm_eps must be positive and finite",
        )?;
        require(
            self.max_position_embeddings > 0,
            "max_position_embeddings must be positive",
        )?;
        require(
            !self.eos_token_ids.is_empty(),
            "the configuration must name at least one end-of-generation token",
        )?;
        for token in &self.eos_token_ids {
            require(
                (*token as usize) < self.vocab_size,
                "every eos token must be inside the vocabulary",
            )?;
        }
        for (name, token) in [
            ("image_token_id", self.image_token_id),
            ("vision_start_token_id", self.vision_start_token_id),
            ("vision_end_token_id", self.vision_end_token_id),
        ] {
            require(
                (token as usize) < self.vocab_size,
                &format!("{name} must be inside the vocabulary"),
            )?;
        }
        if self.has_vision {
            let depth = self.vision_depth.unwrap_or(0);
            let hidden = self.vision_hidden_size.unwrap_or(0);
            let heads = self.vision_num_heads.unwrap_or(16);
            let patch = self.vision_patch_size.unwrap_or(16);
            let temporal = self.vision_temporal_patch_size.unwrap_or(2);
            let merge = self.vision_spatial_merge_size.unwrap_or(2);
            let intermediate = self.vision_intermediate_size.unwrap_or(4304);
            let out = self.vision_out_hidden_size.unwrap_or(2560);
            let positions = self.vision_num_position_embeddings.unwrap_or(2304);
            let rope = self.vision_rope_theta.unwrap_or(10_000.0);
            require(depth > 0, "vision depth must be positive")?;
            require(hidden > 0, "vision hidden_size must be positive")?;
            require(
                heads > 0 && hidden.is_multiple_of(heads),
                "vision hidden_size must be divisible by a positive num_heads",
            )?;
            require(patch > 0, "vision patch_size must be positive")?;
            require(temporal > 0, "vision temporal_patch_size must be positive")?;
            require(merge > 0, "vision spatial_merge_size must be positive")?;
            require(
                intermediate > 0 && out > 0,
                "vision intermediate and output widths must be positive",
            )?;
            require(
                out == self.hidden_size,
                "the vision merger output must match the text hidden_size",
            )?;
            let side = (positions as f64).sqrt() as usize;
            require(
                positions > 0 && side * side == positions,
                "vision num_position_embeddings must be a positive square",
            )?;
            require(
                rope.is_finite() && rope > 0.0,
                "vision rope_theta must be positive and finite",
            )?;
            require(
                patch.checked_mul(merge).is_some_and(|cell| cell > 0),
                "vision patch_size * spatial_merge_size must be a positive cell",
            )?;
        }
        Ok(())
    }
}

/// The quantisation metadata this pack requires of the checkpoint.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct QuantizationSpec {
    /// The mixture name (`MIXED_PRECISION`) of a ModelOpt NVFP4 checkpoint.
    #[serde(default)]
    pub quant_algo: String,
    /// The method a Qwen FP8 checkpoint declares (`fp8`).
    #[serde(default)]
    pub quant_method: Option<String>,
    /// The weight block a block-scaled FP8 checkpoint declares ([128, 128]).
    #[serde(default)]
    pub weight_block_size: Option<Vec<usize>>,
}

impl QuantizationSpec {
    fn validate(&self) -> Result<(), ConfigError> {
        ExpertQuant::of(self).map(|_| ())
    }
}

#[derive(Debug, Deserialize)]
struct RootConfig {
    text_config: TextConfig,
    #[serde(default)]
    vision_config: Option<VisionConfig>,
    #[serde(default)]
    quantization_config: Option<QuantizationSpec>,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default = "default_image_token")]
    image_token_id: u32,
    #[serde(default = "default_vision_start")]
    vision_start_token_id: u32,
    #[serde(default = "default_vision_end")]
    vision_end_token_id: u32,
}

#[derive(Debug, Deserialize)]
struct TextConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    #[serde(default)]
    rope_parameters: RopeConfig,
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
    #[serde(default)]
    max_position_embeddings: usize,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    full_attention_interval: Option<usize>,
    #[serde(default = "default_conv_kernel")]
    linear_conv_kernel_dim: usize,
    #[serde(default = "default_linear_key_dim")]
    linear_key_head_dim: usize,
    #[serde(default = "default_linear_value_dim")]
    linear_value_head_dim: usize,
    #[serde(default = "default_linear_key_heads")]
    linear_num_key_heads: usize,
    #[serde(default = "default_linear_value_heads")]
    linear_num_value_heads: usize,
    #[serde(default = "default_moe_intermediate")]
    moe_intermediate_size: usize,
    #[serde(default = "default_moe_intermediate")]
    shared_expert_intermediate_size: usize,
    #[serde(default = "default_experts")]
    num_experts: usize,
    #[serde(default = "default_topk")]
    num_experts_per_tok: usize,
    #[serde(default = "default_true")]
    norm_topk_prob: bool,
    #[serde(default = "default_hidden_act")]
    hidden_act: String,
    #[serde(default)]
    output_gate_type: Option<String>,
    #[serde(default = "default_hc_count")]
    hc_count: usize,
    #[serde(default = "default_hc_lowrank")]
    hc_lowrank: usize,
    #[serde(default)]
    ple_layer_ids: Option<Vec<usize>>,
    #[serde(default)]
    ple_embed_dim: Option<usize>,
    #[serde(default = "default_ple_conv")]
    ple_conv_kernel_size: usize,
    #[serde(default = "default_ngram_size")]
    ngram_size: usize,
    #[serde(default = "default_heads_per_ngram")]
    heads_per_ngram: usize,
    #[serde(default = "default_ngram_vocab_base")]
    ngram_vocab_size_base: u64,
    #[serde(default = "default_ngram_divisor")]
    make_ngram_vocab_size_divisible_by: u64,
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default = "default_split_parts")]
    split_ngram_parts: usize,
    #[serde(default)]
    indexer_n_heads: Option<usize>,
    #[serde(default)]
    indexer_kv_heads: Option<usize>,
    #[serde(default)]
    indexer_head_dim: Option<usize>,
    #[serde(default)]
    indexer_budget: Option<usize>,
    #[serde(default)]
    indexer_compress_ratio: Option<usize>,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    pad_token_id: Option<u32>,
    #[serde(default)]
    mtp_num_hidden_layers: usize,
}

#[derive(Debug, Default, Deserialize)]
struct RopeConfig {
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
    #[serde(default)]
    mrope_section: Option<[usize; 3]>,
}

#[derive(Debug, Deserialize)]
struct VisionConfig {
    hidden_size: usize,
    depth: usize,
    #[serde(default = "default_vision_patch")]
    patch_size: usize,
    #[serde(default = "default_vision_temporal")]
    temporal_patch_size: usize,
    #[serde(default = "default_vision_merge")]
    spatial_merge_size: usize,
    #[serde(default = "default_vision_heads")]
    num_heads: usize,
    #[serde(default = "default_vision_intermediate")]
    intermediate_size: usize,
    #[serde(default = "default_vision_out")]
    out_hidden_size: usize,
    #[serde(default = "default_vision_positions")]
    num_position_embeddings: usize,
    #[serde(default = "default_vision_rope_theta")]
    rope_theta: f64,
}

fn default_vision_patch() -> usize {
    16
}

fn default_vision_temporal() -> usize {
    2
}

fn default_vision_merge() -> usize {
    2
}

fn default_vision_heads() -> usize {
    16
}

fn default_vision_intermediate() -> usize {
    4304
}

fn default_vision_out() -> usize {
    2560
}

fn default_vision_positions() -> usize {
    2304
}

fn default_vision_rope_theta() -> f64 {
    10_000.0
}

fn default_conv_kernel() -> usize {
    4
}
fn default_linear_key_dim() -> usize {
    128
}
fn default_linear_value_dim() -> usize {
    128
}
fn default_linear_key_heads() -> usize {
    16
}
fn default_linear_value_heads() -> usize {
    32
}
fn default_moe_intermediate() -> usize {
    640
}
fn default_experts() -> usize {
    512
}
fn default_topk() -> usize {
    10
}
fn default_true() -> bool {
    true
}
fn default_hidden_act() -> String {
    "silu".to_owned()
}
fn default_hc_count() -> usize {
    4
}
fn default_hc_lowrank() -> usize {
    320
}
fn default_ple_conv() -> usize {
    4
}
fn default_ngram_size() -> usize {
    3
}
fn default_heads_per_ngram() -> usize {
    8
}
fn default_ngram_vocab_base() -> u64 {
    20_000_000
}
fn default_ngram_divisor() -> u64 {
    128
}
fn default_seed() -> u64 {
    1234
}
fn default_split_parts() -> usize {
    512
}
fn default_image_token() -> u32 {
    248_056
}
fn default_vision_start() -> u32 {
    248_053
}
fn default_vision_end() -> u32 {
    248_054
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The packed arm resolves a boot plan from the checkpoint's own `config.json` before it reads
    /// a weight (`check_plan`), so a release that plans at TP1 and not at TP4 is refused at boot
    /// with the missing part named. This is that check, on the carried FP8 document: the release
    /// plans at four ranks and every part resolves to a kernel image, which is what makes the
    /// four-rank FP8 serve a launch and not an experiment.
    #[test]
    fn the_fp8_release_plans_at_the_four_rank_width() {
        use sparkflash_model::Target;
        use sparkflash_ops::engine_plan::EnginePlan;
        use sparkflash_ops::resolver::{Availability, KernelResolver};
        for world in [1, 2, 4] {
            let plan = EnginePlan::from_json(CARRIED_FP8_CONFIG, Target::Sm121, world, 0)
                .unwrap_or_else(|error| panic!("the FP8 release at TP{world}: {error}"));
            let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue)
                .with_family(plan.family());
            let failures = resolver.failures(&plan);
            assert!(
                failures.is_empty(),
                "TP{world}: {}",
                failures
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; ")
            );
        }
    }

    #[test]
    fn the_released_config_parses_to_the_qwen4_exp_shape() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_hidden_layers, 48);
        assert_eq!(config.num_attention_heads, 24);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.vocab_size, 248_320);
        assert_eq!(config.linear_layers(), 36);
        assert_eq!(config.sparse_layers(), 12);
        assert_eq!(config.layer_types[0], LayerKind::Linear);
        assert_eq!(config.layer_types[3], LayerKind::Sparse);
        assert_eq!(config.rotary_dim(), 64);
        assert_eq!(config.query_groups(), 12);
        assert_eq!(config.value_groups(), 3);
        assert_eq!(config.hyper_width(), 10_240);
        assert_eq!(config.q_proj_width(), 12_288);
        assert_eq!(config.linear_conv_width(), 10_240);
        assert_eq!(config.ngram_heads(), 16);
        assert_eq!(config.ngram_head_dim(), 160);
        assert_eq!(config.indexer_block_budget(), 512);
        assert_eq!(config.max_selected_tokens(), 2_051);
        assert_eq!(config.ple_layer_indices(), vec![1]);
        assert_eq!(config.eos_token_ids, vec![ENDOFTEXT_TOKEN, IM_END_TOKEN]);
        assert_eq!(config.num_experts, 512);
        assert_eq!(config.num_experts_per_tok, 10);
        assert_eq!(config.moe_intermediate_size, 640);
        assert_eq!(config.hc_count, 4);
        assert_eq!(config.hc_lowrank, 320);
        assert!(config.has_vision);
        assert_eq!(config.vision_depth, Some(27));
        assert_eq!(config.mtp_num_hidden_layers, 1);
        // The released checkpoint gates the Gated DeltaNet output with sigmoid. `hidden_act` is
        // silu and governs the convolutions and the experts, not this gate: both arms formerly
        // applied silu here, which served token soup on the real checkpoint while device/host
        // parity stayed green, because the two arms shared the mistake.
        assert_eq!(config.hidden_act, "silu");
        assert_eq!(config.output_gate_type, "sigmoid");
        assert!(config.output_gate_is_sigmoid());
        assert!(config.norm_topk_prob);
    }

    #[test]
    fn the_checkpoint_directory_stop_set_is_one_source() {
        let directory = std::env::temp_dir().join(format!(
            "sparkflash-qwen-stop-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&directory).expect("a temp directory");
        std::fs::write(
            directory.join("config.json"),
            include_str!("../config/qwen38_flash_next_nvfp4.hf.json"),
        )
        .expect("config.json");
        std::fs::write(
            directory.join("generation_config.json"),
            r#"{"eos_token_id": [248044, 248099, 248044]}"#,
        )
        .expect("generation_config.json");
        let config = Qwen4ExpConfig::from_checkpoint_dir(&directory).expect("the checkpoint");
        assert_eq!(
            config.eos_token_ids,
            [248044, 248099, IM_END_TOKEN],
            "the config's id stays first, generation_config adds one, and im_end is appended"
        );
        assert_eq!(
            config.architecture.tokens.eos_token_ids, config.eos_token_ids,
            "the codec's and the runtime's stop sets are the same list"
        );
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn a_generation_config_cannot_move_the_first_eos() {
        // Qwen's real generation_config order is [248046, 248044]. Replacing the config's set
        // would put im_end first, change the PLE segment reset, and change served arithmetic.
        let directory = std::env::temp_dir().join(format!(
            "sparkflash-qwen-eos-order-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&directory).expect("a temp directory");
        std::fs::write(
            directory.join("config.json"),
            include_str!("../config/qwen38_flash_next_nvfp4.hf.json"),
        )
        .expect("config.json");
        std::fs::write(
            directory.join("generation_config.json"),
            r#"{"eos_token_id": [248046, 248044]}"#,
        )
        .expect("generation_config.json");
        let config = Qwen4ExpConfig::from_checkpoint_dir(&directory).expect("the checkpoint");
        assert_eq!(config.eos_token_ids.first(), Some(&ENDOFTEXT_TOKEN));
        assert_eq!(config.eos_token_ids, [ENDOFTEXT_TOKEN, IM_END_TOKEN]);
        assert_eq!(
            config.architecture.tokens.eos_token_ids, config.eos_token_ids,
            "the typed profile and the architecture share one list"
        );
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn a_qwen_document_that_omits_rope_theta_loads_with_one_base() {
        // A fine-tune may drop `rope_parameters.rope_theta` and inherit the hybrid release's
        // 10,000,000 base. Before the aligned default, the typed parser invented 10,000,000 while
        // the shared architecture invented 10,000, and the cross-check refused the document.
        let mut document: serde_json::Value =
            serde_json::from_str(include_str!("../config/qwen38_flash_next_nvfp4.hf.json"))
                .expect("the carried document");
        document
            .get_mut("text_config")
            .and_then(serde_json::Value::as_object_mut)
            .and_then(|text| text.get_mut("rope_parameters"))
            .and_then(serde_json::Value::as_object_mut)
            .expect("the released rope block")
            .remove("rope_theta");
        let config = Qwen4ExpConfig::from_json(&document.to_string()).expect("the fine-tune loads");
        assert_eq!(config.rope_theta, 10_000_000.0);
        assert_eq!(
            config.architecture.layers[0].attention.rope.theta, config.rope_theta,
            "the typed profile and the shared architecture agree on the inherited base"
        );
    }

    #[test]
    fn a_layer_with_no_kind_counts_as_a_validation_failure() {
        let mut config = Qwen4ExpConfig::tiny();
        config.layer_types.pop();
        let error = config.validate().expect_err("the layer count is wrong");
        assert!(error.to_string().contains("layer_types"));
    }

    #[test]
    fn a_ple_layer_on_a_sparse_layer_is_refused() {
        let mut config = Qwen4ExpConfig::tiny();
        config.ple_layer_ids = vec![2];
        let error = config
            .validate()
            .expect_err("PLE must sit on a linear layer");
        assert!(error.to_string().contains("linear-attention"));
    }

    #[test]
    fn malformed_vision_geometry_is_refused_at_parse_time() {
        let mut config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config");
        config.vision_num_heads = Some(0);
        let error = config.validate().expect_err("zero vision heads");
        assert!(error.to_string().contains("num_heads"), "{error}");

        let mut config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config");
        config.vision_out_hidden_size = Some(1);
        let error = config
            .validate()
            .expect_err("merger width disagrees with text");
        assert!(error.to_string().contains("merger"), "{error}");

        let mut config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config");
        config.vision_num_position_embeddings = Some(2305);
        let error = config
            .validate()
            .expect_err("the position grid is not square");
        assert!(error.to_string().contains("square"), "{error}");
    }

    #[test]
    fn a_special_image_id_outside_the_vocabulary_is_refused() {
        let mut config = Qwen4ExpConfig::tiny();
        config.vision_start_token_id = config.vocab_size as u32;
        let error = config
            .validate()
            .expect_err("the id is outside the vocabulary");
        assert!(
            error.to_string().contains("vision_start_token_id"),
            "{error}"
        );
    }

    #[test]
    fn a_misaligned_indexer_budget_is_refused() {
        let mut config = Qwen4ExpConfig::tiny();
        config.indexer_budget = 5;
        let error = config
            .validate()
            .expect_err("the budget is not a multiple of the block width");
        assert!(error.to_string().contains("indexer_budget"));
    }

    #[test]
    fn the_config_round_trips_through_its_own_json() {
        for config in [
            Qwen4ExpConfig::tiny(),
            Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses"),
            Qwen4ExpConfig::qwen38_flash_next_fp8().expect("the carried FP8 config parses"),
        ] {
            let rendered = config.to_json();
            let parsed = Qwen4ExpConfig::from_json(&rendered).unwrap_or_else(|error| {
                panic!("rendered config does not parse: {error}\n{rendered}")
            });
            assert_eq!(parsed, config);
        }
    }

    #[test]
    fn the_fp8_release_is_the_same_architecture_with_a_different_expert_format() {
        let nvfp4 = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
        let fp8 = Qwen4ExpConfig::qwen38_flash_next_fp8().expect("the carried FP8 config parses");
        assert_eq!(fp8.expert_quant, ExpertQuant::Fp8Block);
        assert_eq!(nvfp4.expert_quant, ExpertQuant::Nvfp4);
        assert_eq!(fp8.hidden_size, nvfp4.hidden_size);
        assert_eq!(fp8.num_hidden_layers, nvfp4.num_hidden_layers);
        assert_eq!(fp8.num_experts, nvfp4.num_experts);
        assert_eq!(fp8.moe_intermediate_size, nvfp4.moe_intermediate_size);
        let again =
            Qwen4ExpConfig::from_json(&fp8.to_json()).expect("the rendered document parses");
        assert_eq!(again.expert_quant, ExpertQuant::Fp8Block);
        // The served identity follows the stored format, so a card cannot report the NVFP4
        // checkpoint for an FP8 run.
        assert_eq!(
            crate::device::descriptor_for(&fp8, 4096).id,
            "Qwen/Qwen3.8-Flash-Next-FP8"
        );
        assert_eq!(
            crate::device::descriptor_for(&nvfp4, 4096).id,
            "nvidia/Qwen3.8-Flash-Next-NVFP4"
        );
    }

    #[test]
    fn the_released_modelopt_snapshot_opens_by_its_producer_name() {
        // The snapshot's own quantization_config, copied field for field: ModelOpt names the
        // producer, the mixture is MIXED_PRECISION, and every routed expert is NVFP4. The pack
        // refused this document before a weight was touched, and the carried config hid it by
        // omitting quant_method.
        let document = serde_json::json!({
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "producer": {"name": "modelopt", "version": "0.46.0.dev281+g73d778422"},
            "quantized_layers": {
                "model.language_model.layers.0.mlp.experts": {
                    "quant_algo": "NVFP4", "group_size": 16,
                },
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding": {
                    "quant_algo": "FP8",
                },
            },
            "config_groups": {"group_0": {
                "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                "targets": ["model.language_model.layers.0.mlp.experts"],
            }},
        });
        let spec: QuantizationSpec =
            serde_json::from_value(document).expect("the released spec parses");
        assert_eq!(
            ExpertQuant::of(&spec).expect("the released snapshot opens"),
            ExpertQuant::Nvfp4
        );
        // A producer name still cannot smuggle in an arithmetic the pack has no loader for.
        let spec = QuantizationSpec {
            quant_method: Some("modelopt".to_owned()),
            quant_algo: "AWQ".to_owned(),
            ..QuantizationSpec::default()
        };
        assert!(ExpertQuant::of(&spec).is_err(), "AWQ is still refused");
    }

    #[test]
    fn an_unknown_quantisation_is_refused_by_name() {
        let spec = QuantizationSpec {
            quant_algo: "AWQ".to_owned(),
            ..QuantizationSpec::default()
        };
        let error = ExpertQuant::of(&spec).expect_err("AWQ is not this pack's format");
        assert!(error.to_string().contains("AWQ"), "{error}");
        let spec = QuantizationSpec {
            quant_method: Some("fp8".to_owned()),
            weight_block_size: Some(vec![32, 32]),
            ..QuantizationSpec::default()
        };
        assert!(
            ExpertQuant::of(&spec).is_err(),
            "a 32-wide FP8 tile is not the release"
        );
    }

    #[test]
    fn the_tiny_config_is_self_consistent() {
        let config = Qwen4ExpConfig::tiny();
        config.validate().expect("the tiny configuration is valid");
        assert_eq!(config.ngram_heads(), 4);
        assert_eq!(config.ngram_head_dim(), 4);
        assert_eq!(config.hyper_width(), 64);
    }
}

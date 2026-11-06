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

//! The released `zai-org/GLM-5.3-Flash` configuration.
//!
//! The checkpoint is a multimodal `Glm5NextForConditionalGeneration` whose routed experts, dense and
//! shared MLPs and MLA projections are block-scaled FP8 ([sparkflash_weights::linalg::ScaleGrid]); the
//! BF16 release (`zai-org/GLM-5.3-Flash-BF16`) is the same configuration without its
//! `quantization_config`, and parses too. This pack serves the text
//! language model; the vision tower is planned (so the byte identity against the real index holds)
//! but not loaded.
//!
//! Every field is read from the released `config.json`. The parser is strict about what the
//! reference implementation (`transformers/models/glm5_next/modeling_glm5_next.py`) assumes,
//! because a checkpoint that breaks those assumptions cannot run correctly even if its shapes line
//! up: the MLA heads carry no rotary half (`qk_rope_head_dim = 0`), the indexer scores 4-token
//! pools, the router is a single-group sigmoid with `noaux_tc` selection, and every branch is
//! wrapped in mHC.
//!
//! # What the architecture is
//!
//! 45 decoder layers: 34 Kimi Delta Attention (linear attention) and 11 DeepSeek Sparse Attention
//! layers, one sparse layer after every three linear ones. Layers 0-2 carry a dense SwiGLU MLP and
//! the rest a 288-expert MoE with one shared expert. The residual stream is `hc_mult = 4` copies of
//! the 4,096-wide hidden state, mixed by manifold-constrained hyper-connections around each branch.
//! One MTP layer follows the decoder in the checkpoint.

use serde::Deserialize;
use sparkflash_model::config::OneOrMany;
use sparkflash_model::dflash2::DraftTarget;
use sparkflash_weights::mhc::HcGeometry;

/// Which attention a decoder layer runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Kimi Delta Attention: per-projection causal convolutions, then the delta rule with a decay
    /// per key channel.
    Linear,
    /// DeepSeek Sparse Attention: NoPE MLA over the latent cache, restricted to the indexer's
    /// selection.
    Sparse,
}

/// Which MLP a decoder layer runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpKind {
    /// A dense SwiGLU of `intermediate_size`.
    Dense,
    /// The routed experts plus the shared expert.
    Moe,
}

/// Whether a sparse layer runs its own indexer or reuses the previous sparse layer's selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexerKind {
    /// The layer runs its own indexer.
    Full,
    /// The layer reuses the top-k of the previous sparse layer that ran one.
    Shared,
}

pub use sparkflash_model::ConfigError;

/// The text-model configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Glm5NextConfig {
    /// The shared canonical description of the same checkpoint.
    ///
    /// [Glm5NextConfig::from_json] parses the released document through
    /// [sparkflash_model::Architecture] once and keeps it here; [Glm5NextConfig::steps] and the
    /// device uploader's layer selection read this walk rather than the typed lists.
    pub architecture: sparkflash_model::Architecture,
    /// Hidden width (4,096).
    pub hidden_size: usize,
    /// Decoder layers, the MTP layer excluded (45).
    pub num_hidden_layers: usize,
    /// Vocabulary rows of the embedding and the LM head (154,880).
    pub vocab_size: usize,
    /// RMSNorm epsilon, also the mHC input norm's (1e-5).
    pub rms_norm_eps: f64,
    /// Maximum positions (1,048,576).
    pub max_position_embeddings: usize,
    /// The attention kind of every decoder layer, in order.
    pub layer_types: Vec<LayerKind>,
    /// The MLP kind of every decoder layer, in order.
    pub mlp_layer_types: Vec<MlpKind>,
    /// Whether each layer runs its own indexer; only sparse layers consult it.
    pub indexer_types: Vec<IndexerKind>,
    /// KDA heads (64).
    pub linear_num_heads: usize,
    /// KDA head width, keys and values alike (128).
    pub linear_head_dim: usize,
    /// KDA causal convolution width (4).
    pub linear_conv_kernel_dim: usize,
    /// The forget gate's lower bound (-5): `g = bound * sigmoid(exp(A_log) * (f + dt_bias))`.
    pub linear_lower_bound: f64,
    /// MLA query heads (64).
    pub num_attention_heads: usize,
    /// MLA query LoRA rank (1,536).
    pub q_lora_rank: usize,
    /// MLA key/value latent width (512).
    pub kv_lora_rank: usize,
    /// MLA query/key head width; all of it is nope (256).
    pub qk_nope_head_dim: usize,
    /// MLA value head width (256).
    pub v_head_dim: usize,
    /// Indexer query heads (32).
    pub index_n_heads: usize,
    /// Indexer head width (128).
    pub index_head_dim: usize,
    /// Indexer token budget (2,048), spent in whole pools.
    pub index_topk: usize,
    /// Tokens in one indexer pool (4).
    pub index_kpool: usize,
    /// Whether the incomplete tail pool is always attended (true).
    pub index_kpool_always_select_tail: bool,
    /// Dense MLP width (12,288).
    pub intermediate_size: usize,
    /// Routed and shared expert width (2,048).
    pub moe_intermediate_size: usize,
    /// Routed experts per MoE layer (288).
    pub n_routed_experts: usize,
    /// Shared experts per MoE layer (1).
    pub n_shared_experts: usize,
    /// Experts a token activates (8).
    pub num_experts_per_tok: usize,
    /// Whether the selected weights are renormalised (true).
    pub norm_topk_prob: bool,
    /// The multiplier on the routed weights after renormalisation (2.5).
    pub routed_scaling_factor: f64,
    /// The SwiGLU clamp: `gate <= limit`, `|up| <= limit` (10).
    pub swiglu_limit: f64,
    /// Residual streams (4).
    pub hc_mult: usize,
    /// Sinkhorn normalisations of the comb matrix (20).
    pub hc_sinkhorn_iters: usize,
    /// mHC epsilon in the sigmoid and Sinkhorn steps (1e-6).
    pub hc_eps: f64,
    /// MTP layers in the checkpoint (1).
    pub num_nextn_predict_layers: usize,
    /// Whether the LM head is tied to the embedding (false).
    pub tie_word_embeddings: bool,
    /// End-of-generation ids (`<|endoftext|>`, `<|user|>`, `<|observation|>`).
    pub eos_token_ids: Vec<u32>,
    /// Padding id (`<|endoftext|>`).
    pub pad_token_id: Option<u32>,
    /// Image placeholder id.
    pub image_token_id: u32,
    /// Video placeholder id.
    pub video_token_id: u32,
    /// The vision tower, planned but not served; absent from a text-only configuration.
    pub vision: Option<VisionConfig>,
    /// The tile side of the block-scaled FP8 weights (128), or `None` for the BF16 release.
    pub fp8_block: Option<usize>,
    /// Whether an FP8 weight's input is quantised per `fp8_block` columns at run time, as the
    /// release's `activation_scheme: dynamic` says; every parsed configuration sets it. Only one
    /// built in code clears it, and then the host twin and the device arm both multiply the f32
    /// input: that holds every other kernel to the exact bar without the quantiser's rounding
    /// boundaries, where an association difference of one ulp can move a code (the GLM crate's
    /// device_forward test).
    pub dynamic_activations: bool,
}

/// The vision tower's geometry: enough to plan its tensors, nothing more yet.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct VisionConfig {
    /// Transformer blocks (24).
    pub depth: usize,
    /// Block width (1,024).
    pub hidden_size: usize,
    /// Block MLP width (4,096).
    pub intermediate_size: usize,
    /// Attention heads (16).
    pub num_heads: usize,
    /// The width the merger hands the language model (4,096).
    pub out_hidden_size: usize,
    /// The merger MLP width (10,240).
    pub projection_intermediate_size: usize,
    /// Image channels (3).
    pub in_channels: usize,
    /// Patch side (14).
    pub patch_size: usize,
    /// Frames per patch (2).
    pub temporal_patch_size: usize,
    /// Spatial merge factor before the merger (2).
    pub spatial_merge_size: usize,
}

impl VisionConfig {
    /// The width of one attention head (64).
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// This tower as the shared geometry the device kernels read
    /// ([sparkflash_model::VisionGeometry]).
    ///
    /// The released GLM configuration names no learned position grid, no RoPE base and no pixel
    /// window, so those keep the shared defaults: the tower ropes only, and the planner uses the
    /// serving recipe's window rather than one the checkpoint declares.
    #[must_use]
    pub fn geometry(&self) -> sparkflash_model::VisionGeometry {
        sparkflash_model::VisionGeometry {
            depth: self.depth,
            hidden_size: self.hidden_size,
            num_heads: self.num_heads,
            intermediate_size: self.intermediate_size,
            out_hidden_size: self.out_hidden_size,
            projection_intermediate_size: self.projection_intermediate_size,
            in_channels: self.in_channels,
            patch_size: self.patch_size,
            temporal_patch_size: self.temporal_patch_size,
            spatial_merge_size: self.spatial_merge_size,
            num_position_embeddings: 0,
            rope_theta: 10_000.0,
            min_pixels: 0,
            max_pixels: 0,
        }
    }
}

/// The `model_type` a released GLM-5.3 `config.json` names.
pub const MODEL_TYPE: &str = "glm5_next";

const CARRIED_CONFIG: &str = include_str!("../config/glm53_flash.hf.json");

impl Glm5NextConfig {
    /// The released `zai-org/GLM-5.3-Flash` configuration carried in this crate.
    ///
    /// # Errors
    /// [ConfigError] only if the carried document were edited into an unsupported shape, which the
    /// crate's own test rules out.
    pub fn glm53_flash() -> Result<Self, ConfigError> {
        Self::from_json(CARRIED_CONFIG)
    }

    /// Parse a released HF `config.json`.
    ///
    /// The same document is parsed through the shared
    /// [Architecture](sparkflash_model::Architecture) and cross-checked against this typed view, so
    /// a change that only one parser follows is refused at load.
    ///
    /// # Errors
    /// [ConfigError::Json] for a malformed document and [ConfigError::Invalid] for a shape this
    /// pack cannot serve, or a field on which the two views disagree.
    pub fn from_json(text: &str) -> Result<Self, ConfigError> {
        let json = text;
        let architecture = crate::architecture::architecture_from_json(json)?;
        let root: RootConfig =
            serde_json::from_str(text).map_err(|error| ConfigError::Json(error.to_string()))?;
        let fp8_block = root
            .quantization_config
            .as_ref()
            .map(QuantizationSpec::block)
            .transpose()?;
        let text = root.text_config;
        let layers = text.num_hidden_layers;
        let layer_types = text
            .layer_types
            .iter()
            .map(|kind| match kind.as_str() {
                "linear_attention" => Ok(LayerKind::Linear),
                "deepseek_sparse_attention" => Ok(LayerKind::Sparse),
                other => Err(ConfigError::Invalid(format!(
                    "unknown layer type {other:?}; expected linear_attention or \
                     deepseek_sparse_attention"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mlp_layer_types = match text.mlp_layer_types {
            Some(types) => types
                .iter()
                .map(|kind| match kind.as_str() {
                    "dense" => Ok(MlpKind::Dense),
                    "sparse" => Ok(MlpKind::Moe),
                    other => Err(ConfigError::Invalid(format!(
                        "unknown MLP type {other:?}; expected dense or sparse"
                    ))),
                })
                .collect::<Result<Vec<_>, _>>()?,
            None => (0..layers)
                .map(|layer| {
                    if layer < text.first_k_dense_replace {
                        MlpKind::Dense
                    } else {
                        MlpKind::Moe
                    }
                })
                .collect(),
        };
        let indexer_types = match text.indexer_types {
            Some(types) => types
                .iter()
                .map(|kind| match kind.as_str() {
                    "full" => Ok(IndexerKind::Full),
                    "shared" => Ok(IndexerKind::Shared),
                    other => Err(ConfigError::Invalid(format!(
                        "unknown indexer type {other:?}; expected full or shared"
                    ))),
                })
                .collect::<Result<Vec<_>, _>>()?,
            None => vec![IndexerKind::Full; layers],
        };
        // The official release spells the KDA geometry only inside `linear_attn_config`; the
        // NVFP4 release repeats it as flat keys. Either is accepted, and both must agree.
        let linear = text.linear_attn_config;
        let linear_num_heads = agree("linear_num_heads", text.linear_num_heads, linear.num_heads)?;
        let linear_head_dim = agree("linear_head_dim", text.linear_head_dim, linear.head_dim)?;
        let linear_conv_kernel_dim = agree(
            "linear_conv_kernel_dim",
            text.linear_conv_kernel_dim,
            linear.short_conv_kernel_size,
        )?;
        let linear_lower_bound = agree(
            "linear_lower_bound",
            text.linear_lower_bound,
            linear.gate_lower_bound,
        )?;
        if let Some(kda_layers) = &linear.kda_layers {
            let linear_layers: Vec<usize> = (0..layer_types.len())
                .filter(|&layer| layer_types[layer] == LayerKind::Linear)
                .collect();
            if *kda_layers != linear_layers {
                return Err(ConfigError::Invalid(
                    "linear_attn_config.kda_layers disagrees with layer_types".to_owned(),
                ));
            }
        }
        if text.qk_rope_head_dim != 0 || !text.mla_use_nope {
            return Err(ConfigError::Invalid(format!(
                "qk_rope_head_dim = {} with mla_use_nope = {}: this pack serves NoPE MLA only",
                text.qk_rope_head_dim, text.mla_use_nope
            )));
        }
        if text.attention_bias {
            return Err(ConfigError::Invalid(
                "attention_bias = true: the text attention projections are planned without biases"
                    .to_owned(),
            ));
        }
        let config = Self {
            architecture,
            hidden_size: text.hidden_size,
            num_hidden_layers: layers,
            vocab_size: text.vocab_size,
            rms_norm_eps: text.rms_norm_eps,
            max_position_embeddings: text.max_position_embeddings,
            layer_types,
            mlp_layer_types,
            indexer_types,
            linear_num_heads,
            linear_head_dim,
            linear_conv_kernel_dim,
            linear_lower_bound,
            num_attention_heads: text.num_attention_heads,
            q_lora_rank: text.q_lora_rank,
            kv_lora_rank: text.kv_lora_rank,
            qk_nope_head_dim: text.qk_nope_head_dim,
            v_head_dim: text.v_head_dim,
            index_n_heads: text.index_n_heads,
            index_head_dim: text.index_head_dim,
            index_topk: text.index_topk,
            index_kpool: text.index_kpool,
            index_kpool_always_select_tail: text.index_kpool_always_select_tail,
            intermediate_size: text.intermediate_size,
            moe_intermediate_size: text.moe_intermediate_size,
            n_routed_experts: text.n_routed_experts,
            n_shared_experts: text.n_shared_experts,
            num_experts_per_tok: text.num_experts_per_tok,
            norm_topk_prob: text.norm_topk_prob,
            routed_scaling_factor: text.routed_scaling_factor,
            swiglu_limit: text.swiglu_limit,
            hc_mult: text.hc_mult,
            hc_sinkhorn_iters: text.hc_sinkhorn_iters,
            hc_eps: text.hc_eps,
            num_nextn_predict_layers: text.num_nextn_predict_layers,
            tie_word_embeddings: root.tie_word_embeddings,
            eos_token_ids: text.eos_token_id.into_vec(),
            pad_token_id: text.pad_token_id,
            image_token_id: root.image_token_id,
            video_token_id: root.video_token_id,
            vision: root.vision_config,
            fp8_block,
            dynamic_activations: true,
        };
        if text.n_group != 1 || text.topk_group != 1 {
            return Err(ConfigError::Invalid(format!(
                "n_group = {}, topk_group = {}: only single-group routing is served",
                text.n_group, text.topk_group
            )));
        }
        if text.scoring_func != "sigmoid" || text.topk_method != "noaux_tc" {
            return Err(ConfigError::Invalid(format!(
                "router {} / {}: only sigmoid scores with noaux_tc selection are served",
                text.scoring_func, text.topk_method
            )));
        }
        if text.hidden_act != "silu" {
            return Err(ConfigError::Invalid(format!(
                "hidden_act {:?}: only silu is served",
                text.hidden_act
            )));
        }
        if !text.mhc {
            return Err(ConfigError::Invalid(
                "mhc = false: every layer of this pack runs hyper-connections".to_owned(),
            ));
        }
        config.validate()?;
        config.cross_check(&config.architecture)?;
        Ok(config)
    }

    /// A tiny configuration for the reference and device tests.
    ///
    /// Four layers so every kind appears: a linear layer, a sparse layer that runs its own indexer,
    /// the sparse layer right after it that reuses that selection, and a second linear layer; the
    /// first MLP is dense. The FP8 tiles are 8 wide, so every split width is whole tiles in a TP
    /// world of 2.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            architecture: crate::architecture::tiny_architecture(),
            hidden_size: 32,
            num_hidden_layers: 4,
            vocab_size: 64,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 128,
            layer_types: vec![
                LayerKind::Linear,
                LayerKind::Sparse,
                LayerKind::Sparse,
                LayerKind::Linear,
            ],
            mlp_layer_types: vec![MlpKind::Dense, MlpKind::Moe, MlpKind::Moe, MlpKind::Moe],
            indexer_types: vec![
                IndexerKind::Full,
                IndexerKind::Full,
                IndexerKind::Shared,
                IndexerKind::Full,
            ],
            linear_num_heads: 4,
            linear_head_dim: 8,
            linear_conv_kernel_dim: 4,
            linear_lower_bound: -5.0,
            num_attention_heads: 4,
            q_lora_rank: 16,
            kv_lora_rank: 16,
            qk_nope_head_dim: 8,
            v_head_dim: 8,
            index_n_heads: 2,
            index_head_dim: 8,
            index_topk: 8,
            index_kpool: 2,
            index_kpool_always_select_tail: true,
            intermediate_size: 64,
            moe_intermediate_size: 32,
            n_routed_experts: 8,
            n_shared_experts: 1,
            num_experts_per_tok: 2,
            norm_topk_prob: true,
            routed_scaling_factor: 2.5,
            swiglu_limit: 10.0,
            hc_mult: 4,
            hc_sinkhorn_iters: 20,
            hc_eps: 1e-6,
            num_nextn_predict_layers: 1,
            tie_word_embeddings: false,
            eos_token_ids: vec![2],
            pad_token_id: Some(2),
            image_token_id: 3,
            video_token_id: 4,
            vision: None,
            fp8_block: Some(8),
            dynamic_activations: true,
        }
    }

    /// The smallest configuration the device kernels take: [Glm5NextConfig::tiny]'s four layers
    /// at the release's tile geometry.
    ///
    /// The block-scaled FP8 GEMV multiplies whole 128-wide tiles, the KDA recurrence 128 x 128
    /// heads and the attention one 128-wide head a block, so every FP8 weight here is whole tiles
    /// and every width a TP=4 rank slices is a multiple of 128: one KDA head, one MLA head and a
    /// 128-wide slice of each MLP a rank. The k-pool budget is four 4-token pools, so a 24-token
    /// run keeps the best four of six and the tail.
    #[must_use]
    pub fn small() -> Self {
        Self {
            hidden_size: 256,
            vocab_size: 256,
            linear_head_dim: 128,
            q_lora_rank: 128,
            kv_lora_rank: 128,
            qk_nope_head_dim: 128,
            v_head_dim: 128,
            index_n_heads: 4,
            index_head_dim: 64,
            index_topk: 16,
            index_kpool: 4,
            intermediate_size: 512,
            moe_intermediate_size: 512,
            fp8_block: Some(128),
            ..Self::tiny()
        }
    }

    /// The decoder layers as the shared layer walk, in forward order.
    ///
    /// This is the plan the device uploader selects each layer's branch and FFN from; it reads
    /// [Glm5NextConfig::architecture] rather than [Glm5NextConfig::layer_types].
    #[must_use]
    pub fn steps(&self) -> Vec<sparkflash_model::LayerStep> {
        self.architecture.steps()
    }

    fn validate(&self) -> Result<(), ConfigError> {
        let invalid = |message: String| Err(ConfigError::Invalid(message));
        let layers = self.num_hidden_layers;
        if layers == 0 || self.hidden_size == 0 || self.vocab_size == 0 {
            return invalid("the layer count, hidden width and vocabulary must be nonzero".into());
        }
        for (name, length) in [
            ("layer_types", self.layer_types.len()),
            ("mlp_layer_types", self.mlp_layer_types.len()),
            ("indexer_types", self.indexer_types.len()),
        ] {
            if length != layers {
                return invalid(format!("{name} has {length} entries for {layers} layers"));
            }
        }
        // A shared indexer reuses the selection the layer right before it computed: the reference
        // hands a sparse layer's top-k only to the next layer, and only when that one is shared,
        // so the layer before a shared one must be sparse and run its own indexer.
        for layer in 0..layers {
            if self.layer_types[layer] != LayerKind::Sparse
                || self.indexer_types[layer] != IndexerKind::Shared
            {
                continue;
            }
            let previous = layer.checked_sub(1);
            let follows_full = previous.is_some_and(|previous| {
                self.layer_types[previous] == LayerKind::Sparse
                    && self.indexer_types[previous] == IndexerKind::Full
            });
            if !follows_full {
                return invalid(format!(
                    "sparse layer {layer} shares an indexer, but layer {} is not a sparse layer \
                     that runs one",
                    previous.map_or_else(|| "-1".to_owned(), |previous| previous.to_string())
                ));
            }
        }
        if self.linear_num_heads == 0 || self.linear_head_dim == 0 {
            return invalid("KDA heads and head width must be nonzero".into());
        }
        if self.linear_conv_kernel_dim < 2 {
            return invalid(format!(
                "linear_conv_kernel_dim = {} leaves no history to convolve",
                self.linear_conv_kernel_dim
            ));
        }
        if !(self.linear_lower_bound.is_finite() && self.linear_lower_bound < 0.0) {
            return invalid(format!(
                "linear_lower_bound = {}: the forget gate's log decay must have a negative bound",
                self.linear_lower_bound
            ));
        }
        if self.num_attention_heads == 0 || self.qk_nope_head_dim == 0 || self.v_head_dim == 0 {
            return invalid("MLA heads and head widths must be nonzero".into());
        }
        if self.q_lora_rank == 0 || self.kv_lora_rank == 0 {
            return invalid("this pack serves the LoRA MLA path only".into());
        }
        if self.index_kpool == 0 || !self.index_topk.is_multiple_of(self.index_kpool) {
            return invalid(format!(
                "index_topk = {} is not a whole number of {}-token pools",
                self.index_topk, self.index_kpool
            ));
        }
        if !self.index_kpool_always_select_tail {
            return invalid(
                "index_kpool_always_select_tail = false: a query before its first complete pool \
                 would attend over nothing"
                    .into(),
            );
        }
        if self.index_n_heads == 0 || self.index_head_dim == 0 {
            return invalid("indexer heads and head width must be nonzero".into());
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.n_routed_experts {
            return invalid(format!(
                "{} experts per token out of {}",
                self.num_experts_per_tok, self.n_routed_experts
            ));
        }
        if self.n_shared_experts != 1 {
            return invalid(format!(
                "n_shared_experts = {}: the plan names exactly one shared expert",
                self.n_shared_experts
            ));
        }
        if self.hc_mult == 0 || self.hc_sinkhorn_iters == 0 {
            return invalid("hc_mult and hc_sinkhorn_iters must be nonzero".into());
        }
        if self.fp8_block == Some(0) {
            return invalid("an FP8 tile of zero width".into());
        }
        if self.num_nextn_predict_layers > 1 {
            return invalid(format!(
                "num_nextn_predict_layers = {}: the plan names one MTP layer",
                self.num_nextn_predict_layers
            ));
        }
        Ok(())
    }

    /// The number of KDA layers (34).
    #[must_use]
    pub fn linear_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|kind| **kind == LayerKind::Linear)
            .count()
    }

    /// The number of DSA layers (11).
    #[must_use]
    pub fn sparse_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|kind| **kind == LayerKind::Sparse)
            .count()
    }

    /// The number of MoE layers (42).
    #[must_use]
    pub fn moe_layers(&self) -> usize {
        self.mlp_layer_types
            .iter()
            .filter(|kind| **kind == MlpKind::Moe)
            .count()
    }

    /// The KDA projection width: heads times head width (8,192).
    #[must_use]
    pub fn linear_width(&self) -> usize {
        self.linear_num_heads * self.linear_head_dim
    }

    /// Indexer pools the budget buys (512).
    #[must_use]
    pub fn index_pools(&self) -> usize {
        self.index_topk / self.index_kpool
    }

    /// The decoder layer index of the MTP layer (45).
    #[must_use]
    pub fn mtp_layer(&self) -> usize {
        self.num_hidden_layers
    }

    /// Check that a tensor-parallel world can split this configuration.
    ///
    /// Heads, MLP widths and the LM head rows are split. Every split side of a block-scaled FP8
    /// weight (the MLP widths, the MLA query rows and the MLA output columns) must stay whole tiles
    /// on every rank, so that a rank's scale grid is a slice of the released one.
    ///
    /// # Errors
    /// [ConfigError::Invalid] naming the first width the world does not divide.
    pub fn check_world(&self, world: usize) -> Result<(), ConfigError> {
        if world == 0 {
            return Err(ConfigError::Invalid("a TP world of zero ranks".into()));
        }
        for (name, width) in [
            ("linear_num_heads", self.linear_num_heads),
            ("num_attention_heads", self.num_attention_heads),
            ("intermediate_size", self.intermediate_size),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("vocab_size", self.vocab_size),
        ] {
            if !width.is_multiple_of(world) {
                return Err(ConfigError::Invalid(format!(
                    "{name} = {width} does not split across {world} ranks"
                )));
            }
        }
        let Some(block) = self.fp8_block else {
            return Ok(());
        };
        let heads = self.num_attention_heads;
        for (name, width) in [
            ("intermediate_size", self.intermediate_size),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("the MLA query rows", heads * self.qk_nope_head_dim),
            ("the MLA output columns", heads * self.v_head_dim),
        ] {
            if !width.is_multiple_of(world * block) {
                return Err(ConfigError::Invalid(format!(
                    "{name} = {width} does not split across {world} ranks in whole \
                     {block}-wide FP8 tiles"
                )));
            }
        }
        Ok(())
    }
}

fn agree<T: PartialEq + std::fmt::Debug>(
    name: &str,
    flat: Option<T>,
    nested: Option<T>,
) -> Result<T, ConfigError> {
    match (flat, nested) {
        (Some(flat), Some(nested)) if flat != nested => Err(ConfigError::Invalid(format!(
            "{name} = {flat:?} disagrees with linear_attn_config ({nested:?})"
        ))),
        (Some(value), _) | (None, Some(value)) => Ok(value),
        (None, None) => Err(ConfigError::Json(format!("missing {name}"))),
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
    image_token_id: u32,
    video_token_id: u32,
}

#[derive(Debug, Deserialize)]
struct TextConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    max_position_embeddings: usize,
    layer_types: Vec<String>,
    #[serde(default)]
    mlp_layer_types: Option<Vec<String>>,
    #[serde(default)]
    first_k_dense_replace: usize,
    #[serde(default)]
    indexer_types: Option<Vec<String>>,
    #[serde(default)]
    linear_attn_config: LinearAttnConfig,
    #[serde(default)]
    linear_num_heads: Option<usize>,
    #[serde(default)]
    linear_head_dim: Option<usize>,
    #[serde(default)]
    linear_conv_kernel_dim: Option<usize>,
    #[serde(default)]
    linear_lower_bound: Option<f64>,
    num_attention_heads: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    #[serde(default)]
    mla_use_nope: bool,
    #[serde(default)]
    attention_bias: bool,
    index_n_heads: usize,
    index_head_dim: usize,
    index_topk: usize,
    index_kpool: usize,
    #[serde(default)]
    index_kpool_always_select_tail: bool,
    intermediate_size: usize,
    moe_intermediate_size: usize,
    n_routed_experts: usize,
    n_shared_experts: usize,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
    #[serde(default = "default_group")]
    n_group: usize,
    #[serde(default = "default_group")]
    topk_group: usize,
    scoring_func: String,
    topk_method: String,
    hidden_act: String,
    swiglu_limit: f64,
    #[serde(default)]
    mhc: bool,
    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    hc_eps: f64,
    #[serde(default)]
    num_nextn_predict_layers: usize,
    eos_token_id: OneOrMany<u32>,
    #[serde(default)]
    pad_token_id: Option<u32>,
}

fn default_group() -> usize {
    1
}

#[derive(Debug, Default, Deserialize)]
struct LinearAttnConfig {
    #[serde(default)]
    num_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    short_conv_kernel_size: Option<usize>,
    #[serde(default)]
    gate_lower_bound: Option<f64>,
    #[serde(default)]
    kda_layers: Option<Vec<usize>>,
}

/// The quantisation block: the official release's block-scaled FP8, or absent for the BF16 one.
#[derive(Debug, Deserialize)]
struct QuantizationSpec {
    #[serde(default)]
    quant_method: Option<String>,
    #[serde(default)]
    fmt: Option<String>,
    #[serde(default)]
    activation_scheme: Option<String>,
    #[serde(default)]
    weight_block_size: Option<Vec<usize>>,
}

impl QuantizationSpec {
    /// The square tile side, once the rest of the block names the release's scheme: E4M3 weights
    /// with one scale per tile, and activations quantised at run time rather than by stored scales.
    fn block(&self) -> Result<usize, ConfigError> {
        let scheme = (
            self.quant_method.as_deref(),
            self.fmt.as_deref(),
            self.activation_scheme.as_deref(),
        );
        if scheme != (Some("fp8"), Some("e4m3"), Some("dynamic")) {
            return Err(ConfigError::Invalid(format!(
                "quantization {scheme:?}: this pack reads the official block-scaled FP8 release \
                 or the BF16 one"
            )));
        }
        match self.weight_block_size.as_deref() {
            Some(&[rows, columns]) if rows == columns && rows > 0 => Ok(rows),
            other => Err(ConfigError::Invalid(format!(
                "weight_block_size {other:?}: only square FP8 tiles are served"
            ))),
        }
    }
}

/// The target a DFlash 2 drafter checks itself against ([sparkflash_model::dflash2]).
impl From<&Glm5NextConfig> for DraftTarget {
    fn from(config: &Glm5NextConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            num_hidden_layers: config.num_hidden_layers,
        }
    }
}

/// The mHC geometry: V4.1's schedule ([sparkflash_weights::mhc]) with the stream's input norm at
/// `rms_norm_eps` (1e-5), where V4.1 uses 1e-20.
impl From<&Glm5NextConfig> for HcGeometry {
    fn from(config: &Glm5NextConfig) -> Self {
        Self {
            hc_mult: config.hc_mult,
            dim: config.hidden_size,
            hc_sinkhorn_iters: config.hc_sinkhorn_iters,
            hc_eps: config.hc_eps as f32,
            norm_eps: config.rms_norm_eps as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_released_configuration_parses_to_the_published_geometry() {
        let config = Glm5NextConfig::glm53_flash().expect("the carried config parses");
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 45);
        assert_eq!(config.vocab_size, 154_880);
        assert_eq!(config.linear_layers(), 34);
        assert_eq!(config.sparse_layers(), 11);
        assert_eq!(config.moe_layers(), 42);
        assert_eq!(config.linear_width(), 8192);
        assert_eq!(
            DraftTarget::from(&config),
            DraftTarget {
                hidden_size: 4096,
                vocab_size: 154_880,
                num_hidden_layers: 45,
            }
        );
        let hc = HcGeometry::from(&config);
        assert_eq!((hc.hc_width(), hc.hc_mix_count()), (16_384, 24));
        assert_eq!((hc.hc_eps, hc.norm_eps), (1e-6, 1e-5));
        assert_eq!(config.index_pools(), 512);
        assert_eq!(config.linear_lower_bound, -5.0);
        assert_eq!(config.eos_token_ids, vec![154_820, 154_827, 154_829]);
        assert_eq!(config.mtp_layer(), 45);
        assert_eq!(config.fp8_block, Some(128));
        assert!(
            config.dynamic_activations,
            "the release quantises activations at run time"
        );
        let vision = config
            .vision
            .as_ref()
            .expect("the release carries a vision tower");
        assert_eq!((vision.depth, vision.head_dim()), (24, 64));
        // One sparse layer after every three linear ones.
        for (layer, kind) in config.layer_types.iter().enumerate() {
            let sparse = layer % 4 == 3;
            assert_eq!(*kind == LayerKind::Sparse, sparse, "layer {layer}");
        }
        assert!(
            config.mlp_layer_types[..3]
                .iter()
                .all(|kind| *kind == MlpKind::Dense)
        );
        assert!(
            config
                .indexer_types
                .iter()
                .all(|kind| *kind == IndexerKind::Full)
        );
    }

    #[test]
    fn the_shared_walk_matches_the_typed_schedule() {
        use sparkflash_model::parts::Part;

        let config = Glm5NextConfig::glm53_flash().expect("the carried config parses");
        let steps = config.steps();
        assert_eq!(steps.len(), config.num_hidden_layers);
        for (step, kind) in steps.iter().zip(&config.layer_types) {
            let expected = match kind {
                LayerKind::Linear => Part::KdaScan,
                LayerKind::Sparse => Part::MlaAttention,
            };
            assert_eq!(step.attention, expected, "layer {}", step.index);
        }
        for (step, kind) in steps.iter().zip(&config.mlp_layer_types) {
            let expected = match kind {
                MlpKind::Dense => Part::DenseMlp,
                MlpKind::Moe => Part::Moe,
            };
            assert_eq!(step.ffn, expected, "layer {}", step.index);
        }
        assert_eq!(
            steps
                .iter()
                .filter(|step| step.attention == Part::KdaScan)
                .count(),
            config.linear_layers()
        );
        assert_eq!(
            steps
                .iter()
                .filter(|step| step.attention == Part::MlaAttention)
                .count(),
            config.sparse_layers()
        );
        let dense_configured = config
            .mlp_layer_types
            .iter()
            .filter(|kind| **kind == MlpKind::Dense)
            .count();
        assert_eq!(
            steps
                .iter()
                .filter(|step| step.ffn == Part::DenseMlp)
                .count(),
            dense_configured
        );
        assert_eq!(
            steps.iter().filter(|step| step.ffn == Part::Moe).count(),
            config.moe_layers()
        );
        assert!(
            steps[..3].iter().all(|step| step.ffn == Part::DenseMlp),
            "the three dense layers are the first three"
        );
        assert!(steps[3..].iter().all(|step| step.ffn == Part::Moe));
    }

    #[test]
    fn the_tiny_architecture_walks_its_typed_schedule() {
        use sparkflash_model::parts::Part;

        let config = Glm5NextConfig::tiny();
        let steps = config.steps();
        assert_eq!(steps.len(), config.num_hidden_layers);
        assert_eq!(
            steps
                .iter()
                .filter(|step| step.attention == Part::KdaScan)
                .count(),
            config.linear_layers()
        );
        assert_eq!(
            steps
                .iter()
                .filter(|step| step.attention == Part::MlaAttention)
                .count(),
            config.sparse_layers()
        );
        assert_eq!(
            steps
                .iter()
                .filter(|step| step.ffn == Part::DenseMlp)
                .count(),
            1
        );
        assert_eq!(
            steps.iter().filter(|step| step.ffn == Part::Moe).count(),
            config.moe_layers()
        );
        assert_eq!(steps[0].ffn, Part::DenseMlp);
        assert_eq!(
            steps
                .iter()
                .filter(|step| step.indexer == Some(Part::SparseIndexer))
                .count(),
            1,
            "only the sparse layer that runs its own indexer carries one"
        );
    }

    #[test]
    fn the_released_configuration_splits_four_ways_but_not_three() {
        let config = Glm5NextConfig::glm53_flash().expect("the carried config parses");
        config.check_world(1).expect("TP=1");
        config.check_world(2).expect("TP=2");
        config.check_world(4).expect("TP=4");
        assert!(config.check_world(3).is_err());
        Glm5NextConfig::tiny()
            .check_world(2)
            .expect("the tiny fixture splits in two");
        let small = Glm5NextConfig::small();
        small.validate().expect("the small fixture is valid");
        small
            .check_world(4)
            .expect("the small fixture splits four ways in whole tiles");
    }

    #[test]
    fn rotary_mla_is_refused() {
        let text = CARRIED_CONFIG.replace("\"qk_rope_head_dim\": 0", "\"qk_rope_head_dim\": 64");
        assert_ne!(
            text, CARRIED_CONFIG,
            "the carried config names qk_rope_head_dim"
        );
        assert!(matches!(
            Glm5NextConfig::from_json(&text),
            Err(ConfigError::Invalid(_))
        ));
    }

    #[test]
    fn a_shared_indexer_must_follow_a_sparse_layer_that_runs_one() {
        let config = Glm5NextConfig::tiny();
        config.validate().expect("the tiny fixture is valid");
        // Layer 1 follows a linear layer.
        let mut first = config.clone();
        first.indexer_types[1] = IndexerKind::Shared;
        assert!(first.validate().is_err());
        // Layer 2 would follow a shared layer, which hands nothing on.
        let mut chained = config.clone();
        chained.indexer_types[1] = IndexerKind::Shared;
        chained.layer_types[0] = LayerKind::Sparse;
        assert!(chained.validate().is_err());
        // A linear layer between the full and the shared layer breaks the hand-off.
        let mut gapped = config;
        gapped.layer_types.swap(2, 3);
        gapped.indexer_types.swap(2, 3);
        assert!(gapped.validate().is_err());
    }

    #[test]
    fn a_selection_without_the_tail_is_refused() {
        let mut config = Glm5NextConfig::tiny();
        config.index_kpool_always_select_tail = false;
        assert!(config.validate().is_err());
    }

    #[test]
    fn disagreeing_kda_geometry_is_refused() {
        let flat = |heads: usize| {
            CARRIED_CONFIG.replace(
                "\"linear_attn_config\": {",
                &format!("\"linear_num_heads\": {heads}, \"linear_attn_config\": {{"),
            )
        };
        assert_ne!(
            flat(64),
            CARRIED_CONFIG,
            "the carried config names linear_attn_config"
        );
        Glm5NextConfig::from_json(&flat(64)).expect("agreeing flat and nested keys parse");
        assert!(matches!(
            Glm5NextConfig::from_json(&flat(32)),
            Err(ConfigError::Invalid(_))
        ));
    }

    fn edited(edit: impl FnOnce(&mut serde_json::Value)) -> Result<Glm5NextConfig, ConfigError> {
        let mut root: serde_json::Value =
            serde_json::from_str(CARRIED_CONFIG).expect("the carried config is JSON");
        edit(&mut root);
        Glm5NextConfig::from_json(&root.to_string())
    }

    #[test]
    fn the_bf16_release_is_the_same_model_without_tiles() {
        let bf16 = edited(|root| {
            root.as_object_mut()
                .expect("an object")
                .remove("quantization_config");
        })
        .expect("the BF16 release's config parses");
        assert_eq!(bf16.fp8_block, None);
        assert_eq!(bf16.architecture.quantization.block, None);
        let fp8 = Glm5NextConfig::glm53_flash().expect("the carried config parses");
        assert_eq!(fp8.architecture.quantization.block, Some(128));
        let mut same = bf16;
        same.fp8_block = fp8.fp8_block;
        same.architecture = fp8.architecture.clone();
        assert_eq!(same, fp8);
    }

    #[test]
    fn another_quantization_is_refused() {
        let modelopt = edited(|root| {
            root["quantization_config"]["quant_method"] = "modelopt".into();
        });
        assert!(matches!(modelopt, Err(ConfigError::Invalid(_))));
        let static_activations = edited(|root| {
            root["quantization_config"]["activation_scheme"] = "static".into();
        });
        assert!(matches!(static_activations, Err(ConfigError::Invalid(_))));
        let oblong = edited(|root| {
            root["quantization_config"]["weight_block_size"] = serde_json::json!([128, 64]);
        });
        assert!(matches!(oblong, Err(ConfigError::Invalid(_))));
    }
}

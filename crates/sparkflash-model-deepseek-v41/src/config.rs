//! The model configuration: the flat runtime hyperparameters, checked.
//!
//! The engine reads the *runtime* configuration, not the released HF config. Those are different
//! documents: the HF config nests everything under `text_config`/`vision_config` and names a few
//! keys differently, while the runtime config is flat and is what the reference implementation
//! feeds straight into `ModelArgs(**json.load(...))` (inference/generate.py:121-123, and the
//! docstring at model.py:46 says the field names are exactly the JSON keys). This module uses the
//! runtime keys, so a value here and a value in the reference are the same value.
//!
//! [`ModelConfig::v41_flash`] parses the copy of that configuration carried in this crate at
//! `config/deepseek_v41_flash.runtime.json`. The file is a byte-for-byte semantic copy: the same 64
//! keys and values as the released runtime config, re-serialised. Its provenance, including the
//! source digest, is in the `README.md` beside it.
//!
//! Two fields are deliberately absent from the released file and carry the reference's own defaults:
//! `gate_temp` (1.0) and `norm_topk_prob` (true), both declared in `ModelArgs` at model.py:67-68.
//!
//! The checkpoint also ships the released HF `config.json` at its root, and that document is not the
//! runtime one: it nests everything under `text_config`/`vision_config` and renames several keys
//! (`hidden_size` for `dim`, `num_nextn_predict_layers` for `n_mtp_layers`,
//! `routed_scaling_factor` for `route_scale`, and more). [`ModelConfig::from_released_json`] maps it
//! field by field and [`ModelConfig::from_json`] accepts either document, so `sparkflash serve`
//! with `--checkpoint` reads the checkpoint as it was released instead of requiring its `inference/`
//! config to be extracted first. Every key is accounted for - mapped, checked against the value the rest of
//! the engine assumes, or explicitly ignored with its reason - and an unrecognised key is refused by
//! name rather than dropped; `tests/released_config.rs` pins the schema, so the next drift fails a
//! test.
//!
//! The released document carries no `first_k_dense_replace`, `n_group` or `topk_group`, and neither
//! does the runtime one: they are DeepSeek-V3-era fields, every V4.1 block is a MoE block
//! (model.py:929) and the gate is an ungrouped top-k over all routed experts (model.py:808-823), so
//! nothing in this crate needs them. `gate.bias_vl` is absent for a different reason: it is a tensor
//! rather than a config field, present only where the vision tower exists (model.py:807), and the
//! text path never reads it (forward/moe.rs:461).

use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

/// The runtime configuration carried in this crate.
pub const V41_FLASH_RUNTIME_CONFIG: &str =
    include_str!("../config/deepseek_v41_flash.runtime.json");

pub use sparkflash_weights::ConfigError;

/// The hyperparameters of one DeepSeek-V4.1-Flash-shaped model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Size of the token vocabulary.
    pub vocab_size: usize,
    /// Residual stream width.
    pub dim: usize,
    /// Intermediate width of one routed expert.
    pub moe_inter_dim: usize,
    /// Backbone blocks.
    pub n_layers: usize,
    /// DSpark MTP stages, which the checkpoint names `mtp.{0..n_mtp_layers-1}`.
    pub n_mtp_layers: usize,
    /// Tokens one DSpark block drafts.
    pub dspark_block_size: usize,
    /// The token a draft slot is seeded with when nothing better is known.
    pub dspark_noise_token_id: u32,
    /// The backbone layers whose hidden states feed the MTP stages.
    pub dspark_target_layer_ids: Vec<usize>,
    /// Width of the Markov head's embedding.
    pub dspark_markov_rank: usize,
    /// Routed experts per MTP stage.
    pub dspark_n_routed_experts: usize,
    /// Experts activated per token per MTP stage.
    pub dspark_n_activated_experts: usize,
    /// Query heads.
    pub n_heads: usize,
    /// Routed experts per backbone block.
    pub n_routed_experts: usize,
    /// Shared experts. The reference asserts this is exactly 1 (model.py:886).
    pub n_shared_experts: usize,
    /// Experts activated per token per backbone block.
    pub n_activated_experts: usize,
    /// The routing score function: `sqrtsoftplus`, `sigmoid` or `softmax`.
    pub score_func: String,
    /// Multiplier applied to the routed weights after normalisation.
    pub route_scale: f64,
    /// Clamp applied to the SwiGLU gate branch.
    pub swiglu_limit: f64,
    /// Query LoRA rank.
    pub q_lora_rank: usize,
    /// MLA latent width, including the rotated slice.
    pub head_dim: usize,
    /// The last `rope_head_dim` dims of `head_dim` are rotated.
    pub rope_head_dim: usize,
    /// RMSNorm epsilon.
    pub norm_eps: f64,
    /// Output projection groups.
    pub o_groups: usize,
    /// Per-group output width.
    pub o_lora_rank: usize,
    /// Sliding window width.
    pub window_size: usize,
    /// Layers that own the main KV and publish index keys.
    pub kv_source_layers: Vec<usize>,
    /// Layers that run an HSI indexer.
    pub index_source_layers: Vec<usize>,
    /// The trained sequence length, which sizes the RoPE scaling.
    pub original_seq_len: usize,
    /// RoPE base.
    pub rope_theta: f64,
    /// RoPE scaling factor.
    pub rope_factor: f64,
    /// RoPE correction bound (fast).
    pub beta_fast: f64,
    /// RoPE correction bound (slow).
    pub beta_slow: f64,
    /// Indexer heads.
    pub index_n_heads: usize,
    /// Indexer head width.
    pub index_head_dim: usize,
    /// Index entries kept per query.
    pub index_topk: usize,
    /// The layer that builds the HSI candidate pool.
    pub candidate_source_layer: usize,
    /// Blocks in that pool.
    pub candidate_topk_blocks: usize,
    /// Tokens per pool block.
    pub candidate_block_size: usize,
    /// mHC streams: the stream count `n`, giving `hc_dim = n * dim`.
    pub hc_mult: usize,
    /// Sinkhorn iterations used to normalise the mHC coefficients.
    pub hc_sinkhorn_iters: usize,
    /// mHC epsilon.
    pub hc_eps: f64,
    /// Layers carrying an Engram table.
    pub engram_layer_ids: Vec<usize>,
    /// The Engram vocabulary as the config declares it.
    pub engram_vocab_size: usize,
    /// Rows in each Engram table, one per layer in `engram_layer_ids`.
    pub engram_num_embeddings: Vec<usize>,
    /// Longest n-gram the Engram keys on.
    pub engram_max_ngram_size: usize,
    /// The padding token the Engram compressor skips.
    pub engram_pad_id: u32,
    /// The compressed vocabulary the Engram keys are drawn from.
    pub engram_compressed_vocab_size: usize,
    /// Engram heads.
    pub engram_n_heads: usize,
    /// Engram head width.
    pub engram_head_dim: usize,
    /// The dense weight format, as the config spells it (`fp8`).
    pub dtype: String,
    /// The expert weight format (`fp4`).
    pub expert_dtype: String,
    /// RoPE base for the compressor.
    pub compress_rope_theta: f64,
    /// Per-layer compression ratio, over `n_layers + n_mtp_layers` entries.
    pub compress_ratios: Vec<usize>,
    /// The image placeholder token.
    pub image_token_id: u32,
    /// Routing temperature. Absent from the released config; the reference default is 1.0.
    #[serde(default = "default_gate_temp")]
    pub gate_temp: f64,
    /// Whether the top-k routing weights are renormalised. Absent from the released config; the
    /// reference default is true.
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    /// Vision tower depth, or `None` when the config carries no vision tower.
    #[serde(default)]
    pub vision_n_layers: Option<usize>,
    /// Vision tower width.
    #[serde(default)]
    pub vision_dim: Option<usize>,
    /// Vision tower heads.
    #[serde(default)]
    pub vision_n_heads: Option<usize>,
    /// Vision MLP width.
    #[serde(default)]
    pub vision_inter_dim: Option<usize>,
    /// Vision patch size.
    #[serde(default)]
    pub vision_patch_size: Option<usize>,
    /// Vision patch downsample ratio.
    #[serde(default)]
    pub vision_downsample_ratio: Option<usize>,
    /// Most tokens one image may contribute.
    #[serde(default)]
    pub vision_max_n_token: Option<usize>,
    /// Fewest pixels an image may have.
    #[serde(default)]
    pub vision_min_pixels: Option<usize>,
    /// Aspect-ratio bound for image resizing.
    #[serde(default)]
    pub vision_max_wh_ratio: Option<f64>,
    /// Vision RoPE base.
    #[serde(default)]
    pub vision_rope_theta: Option<f64>,
}

fn default_gate_temp() -> f64 {
    1.0
}

fn default_norm_topk_prob() -> bool {
    true
}

impl ModelConfig {
    /// The released DeepSeek-V4.1-Flash runtime configuration, parsed from the copy in this crate.
    ///
    /// Parsing rather than constructing is deliberate: it puts the shipped constants through the
    /// same deserialiser a user's own config goes through, so a key this struct stops accepting is
    /// a test failure rather than a silent divergence between two constructors.
    pub fn v41_flash() -> Result<Self, ConfigError> {
        Self::from_json(V41_FLASH_RUNTIME_CONFIG)
    }

    /// Parse a configuration from JSON and check it.
    ///
    /// Accepts either of the two documents the checkpoint distribution carries: the flat runtime
    /// configuration ([the copy in this crate](V41_FLASH_RUNTIME_CONFIG)) and the released HF
    /// `config.json`, which [`ModelConfig::from_released_json`] maps. A `text_config` key selects
    /// the released reader; the two shapes cannot be confused.
    pub fn from_json(text: &str) -> Result<Self, ConfigError> {
        let value: serde_json::Value =
            serde_json::from_str(text).map_err(|e| ConfigError::Json(e.to_string()))?;
        let config = if value.get("text_config").is_some() {
            released::from_value(value)?
        } else {
            serde_json::from_value(value).map_err(|e| ConfigError::Json(e.to_string()))?
        };
        config.validate()?;
        Ok(config)
    }

    /// Parse the released HF `config.json` - the checkpoint's own configuration document.
    ///
    /// This is the document the checkpoint ships at its root, not the flat runtime copy in this crate:
    /// the same model, nested under `text_config`/`vision_config` with the HF key names. The
    /// mapping is explicit and total - a key the released document carries is assigned to a field,
    /// checked against the value the rest of the engine assumes, or ignored with its reason - and a
    /// key this loader was not written for is an error naming the key.
    ///
    /// # Errors
    ///
    /// [`ConfigError::Json`] when the document does not have the released schema, and
    /// [`ConfigError::Invalid`] when a value contradicts an assumption the engine makes elsewhere.
    pub fn from_released_json(text: &str) -> Result<Self, ConfigError> {
        let value: serde_json::Value =
            serde_json::from_str(text).map_err(|e| ConfigError::Json(e.to_string()))?;
        let config = released::from_value(value)?;
        config.validate()?;
        Ok(config)
    }

    /// The released configuration, parsed once and cached.
    ///
    /// The shipped configuration is a compile-time constant that a test parses on every run, so a
    /// failure here would already have failed the build's tests. Panicking is therefore honest: it
    /// means the crate and its own fixture disagree, not that a user supplied something bad.
    ///
    /// # Panics
    ///
    /// Panics if the shipped configuration does not parse or does not validate.
    pub fn v41_flash_cached() -> &'static ModelConfig {
        static CACHED: OnceLock<ModelConfig> = OnceLock::new();
        CACHED.get_or_init(|| {
            ModelConfig::v41_flash().expect("the shipped runtime configuration must parse")
        })
    }

    /// A structurally complete configuration with tiny shapes, for tests that must run without the
    /// checkpoint.
    ///
    /// Every relation the real model has is preserved - a real compressor layer, a real indexer
    /// layer, an Engram layer, an MTP stage - so a test that walks the plan exercises the same code
    /// paths at a size that fits in a few kilobytes.
    pub fn tiny() -> Self {
        ModelConfig {
            vocab_size: 128,
            dim: 64,
            moe_inter_dim: 32,
            n_layers: 2,
            n_mtp_layers: 1,
            dspark_block_size: 2,
            dspark_noise_token_id: 127,
            dspark_target_layer_ids: vec![1],
            dspark_markov_rank: 8,
            dspark_n_routed_experts: 2,
            dspark_n_activated_experts: 1,
            n_heads: 2,
            n_routed_experts: 4,
            n_shared_experts: 1,
            n_activated_experts: 2,
            score_func: "sqrtsoftplus".to_string(),
            route_scale: 1.5,
            swiglu_limit: 10.0,
            q_lora_rank: 32,
            head_dim: 16,
            rope_head_dim: 4,
            norm_eps: 1e-20,
            o_groups: 2,
            o_lora_rank: 8,
            window_size: 8,
            kv_source_layers: vec![0],
            index_source_layers: vec![0, 1],
            original_seq_len: 64,
            rope_theta: 10_000.0,
            rope_factor: 2.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            index_n_heads: 2,
            index_head_dim: 8,
            index_topk: 4,
            candidate_source_layer: 0,
            candidate_topk_blocks: 4,
            candidate_block_size: 2,
            hc_mult: 2,
            hc_sinkhorn_iters: 2,
            hc_eps: 1e-6,
            engram_layer_ids: vec![1],
            engram_vocab_size: 100,
            // Large enough for the whole bucket range: the reference indexes the table with the
            // hash id itself (model.py:309, 317), so the rows have to cover every offset plus every
            // one of the (orders - 1) * n_heads bucket primes, which for this vocabulary are drawn
            // just above engram_vocab_size. 64 rows was smaller than the sum of two such primes and
            // made tiny an invalid configuration the layout refused.
            engram_num_embeddings: vec![1024],
            engram_max_ngram_size: 2,
            engram_pad_id: 2,
            engram_compressed_vocab_size: 32,
            engram_n_heads: 2,
            engram_head_dim: 8,
            dtype: "fp8".to_string(),
            expert_dtype: "fp4".to_string(),
            compress_rope_theta: 160_000.0,
            compress_ratios: vec![2, 2, 0],
            image_token_id: 127,
            gate_temp: 1.0,
            norm_topk_prob: true,
            vision_n_layers: None,
            vision_dim: None,
            vision_n_heads: None,
            vision_inter_dim: None,
            vision_patch_size: None,
            vision_downsample_ratio: None,
            vision_max_n_token: None,
            vision_min_pixels: None,
            vision_max_wh_ratio: None,
            vision_rope_theta: None,
        }
    }

    /// Check the relations the architecture depends on.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let bad = |field: &'static str, detail: String| Err(ConfigError::Invalid { field, detail });
        if self.dim == 0 || self.moe_inter_dim == 0 || self.n_heads == 0 || self.head_dim == 0 {
            return bad(
                "dim",
                "the model shape must be non-zero in every dimension".into(),
            );
        }
        if self.head_dim < self.rope_head_dim {
            return bad(
                "rope_head_dim",
                format!(
                    "the rotated slice ({}) is wider than the latent ({})",
                    self.rope_head_dim, self.head_dim
                ),
            );
        }
        if !self.rope_head_dim.is_multiple_of(2) {
            return bad(
                "rope_head_dim",
                "RoPE rotates pairs, so this must be even".into(),
            );
        }
        if self.o_groups == 0 || self.o_lora_rank == 0 || self.q_lora_rank == 0 {
            return bad(
                "o_groups",
                "the attention projections must be non-zero".into(),
            );
        }
        if self.n_shared_experts != 1 {
            return bad(
                "n_shared_experts",
                format!(
                    "the reference asserts exactly 1 shared expert, this config has {}",
                    self.n_shared_experts
                ),
            );
        }
        if self.n_activated_experts == 0 || self.n_activated_experts > self.n_routed_experts {
            return bad(
                "n_activated_experts",
                "top-k must be in 1..=n_routed_experts".into(),
            );
        }
        if self.dspark_n_activated_experts == 0
            || self.dspark_n_activated_experts > self.dspark_n_routed_experts
        {
            return bad(
                "dspark_n_activated_experts",
                "the MTP top-k must be in 1..=dspark_n_routed_experts".into(),
            );
        }
        let expected = self.n_layers + self.n_mtp_layers;
        if self.compress_ratios.len() != expected {
            return bad(
                "compress_ratios",
                format!(
                    "expected one ratio per backbone layer and MTP stage ({expected}), found {}",
                    self.compress_ratios.len()
                ),
            );
        }
        for (field, ids) in [
            ("kv_source_layers", &self.kv_source_layers),
            ("index_source_layers", &self.index_source_layers),
        ] {
            if let Some(&id) = ids.iter().find(|&&id| id >= self.n_layers) {
                return bad(field, format!("layer {id} is not a backbone layer"));
            }
        }
        if self.candidate_source_layer >= self.n_layers {
            return bad(
                "candidate_source_layer",
                "the pool builder must be a backbone layer".into(),
            );
        }
        if self.engram_layer_ids.len() != self.engram_num_embeddings.len() {
            return bad(
                "engram_num_embeddings",
                format!(
                    "expected one row count per Engram layer ({}), found {}",
                    self.engram_layer_ids.len(),
                    self.engram_num_embeddings.len()
                ),
            );
        }
        if let Some(&id) = self
            .engram_layer_ids
            .iter()
            .find(|&&id| id >= self.n_layers)
        {
            return bad(
                "engram_layer_ids",
                format!("layer {id} is not a backbone layer"),
            );
        }
        if self.engram_n_heads == 0 || self.engram_head_dim == 0 {
            return bad(
                "engram_head_dim",
                "the Engram heads must be non-zero".into(),
            );
        }
        if self.dspark_target_layer_ids.len() != self.n_mtp_layers {
            return bad(
                "dspark_target_layer_ids",
                format!(
                    "expected one target layer per MTP stage ({}), found {}",
                    self.n_mtp_layers,
                    self.dspark_target_layer_ids.len()
                ),
            );
        }
        // Each target must NAME a backbone layer, not merely be a plausible count: the released
        // model targets 37, 38 and 39 of 40 layers (model.py:1111-1113). A configuration whose target
        // index is out of range describes a model that cannot exist, and the plan would happily
        // describe it, so it is refused here.
        for &id in &self.dspark_target_layer_ids {
            if id >= self.n_layers {
                return bad(
                    "dspark_target_layer_ids",
                    format!(
                        "target layer {id} is not a backbone layer of {}",
                        self.n_layers
                    ),
                );
            }
        }
        if self.window_size == 0 || self.hc_mult == 0 || self.hc_sinkhorn_iters == 0 {
            return bad(
                "window_size",
                "the attention and mHC widths must be non-zero".into(),
            );
        }
        Ok(())
    }

    /// Whether the configuration carries a vision tower.
    ///
    /// The tower is what makes `ffn.gate.bias_vl` exist at all (model.py:807 creates the bias inside
    /// the `vision_enabled` branch), so a plan must ask this rather than assume the released shape.
    ///
    /// Zero disables the vision path in exactly the way an absent field does: `vision_n_layers`
    /// defaults to 0 and the reference computes `self.vision_enabled = self.vision_n_layers > 0`
    /// (model.py:116-117, 138-140). Testing for `Some(_)` rather than a positive count would accept
    /// the reference's own disabled sentinel and plan 43 `ffn.gate.bias_vl` tensors that a
    /// vision-disabled checkpoint cannot contain.
    pub fn vision_enabled(&self) -> bool {
        self.vision_n_layers.is_some_and(|layers| layers > 0)
    }

    /// Whether a layer index addresses an MTP stage rather than a backbone block.
    pub fn is_mtp(&self, layer: usize) -> bool {
        layer >= self.n_layers
    }

    /// The routed and activated expert counts for a block, which differ between the backbone and the
    /// MTP stages (model.py:142-149, `get_moe_config`).
    pub fn moe_config(&self, layer: usize) -> (usize, usize) {
        if self.is_mtp(layer) {
            (
                self.dspark_n_routed_experts,
                self.dspark_n_activated_experts,
            )
        } else {
            (self.n_routed_experts, self.n_activated_experts)
        }
    }

    /// The compression ratio of a block, or `None` when the index is out of range.
    pub fn compress_ratio(&self, layer: usize) -> Option<usize> {
        self.compress_ratios.get(layer).copied()
    }

    /// The Engram table rows for a layer, or `None` when that layer has no table.
    pub fn engram_rows(&self, layer: usize) -> Option<usize> {
        self.engram_layer_ids
            .iter()
            .position(|&l| l == layer)
            .map(|position| self.engram_num_embeddings[position])
    }

    /// The number of mHC coefficient groups: `(2 + n) * n`.
    pub fn hc_mix_count(&self) -> usize {
        (2 + self.hc_mult) * self.hc_mult
    }

    /// The width of the mHC stream: `n * dim`.
    pub fn hc_width(&self) -> usize {
        self.hc_mult * self.dim
    }
}

/// The released HF `config.json`, mapped onto [`ModelConfig`].
///
/// The checkpoint ships two configuration documents and they are not the same file. This module reads
/// the released HF one - the `config.json` at the checkpoint root - and maps it onto the flat runtime
/// configuration. The mapping is explicit in both directions: every key of the released document is
/// named here, `deny_unknown_fields` refuses a key this loader was not written for, and a value the
/// rest of the engine depends on (the quantisation scheme, the activation, the KV-head count) is
/// checked rather than accepted silently.
mod released {
    use serde::Deserialize;

    use super::{ConfigError, ModelConfig, default_gate_temp};

    /// The released document's top-level object.
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Document {
        /// The HF architecture the checkpoint belongs to.
        architectures: Vec<String>,
        /// The HF model family.
        model_type: String,
        /// The safetensors storage dtype (`bfloat16`), which is not the runtime `dtype`.
        #[allow(dead_code)]
        dtype: String,
        /// Informational: which `transformers` wrote the file.
        #[allow(dead_code)]
        transformers_version: String,
        /// Tokenizer contract, not a runtime field.
        #[allow(dead_code)]
        bos_token_id: u32,
        /// Tokenizer contract, not a runtime field.
        #[allow(dead_code)]
        eos_token_id: u32,
        /// Tokenizer contract, not a runtime field.
        #[allow(dead_code)]
        pad_token_id: u32,
        /// The image placeholder token.
        image_token_id: u32,
        /// How the checkpoint stores its weights.
        quantization_config: Quantization,
        /// The language model.
        text_config: Text,
        /// The vision tower.
        vision_config: Vision,
    }

    /// `quantization_config`: checked because the plan and the decode assume it.
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Quantization {
        /// `fp8`: the dense weight format the plan describes.
        quant_method: String,
        /// Per-token dynamic activation scaling: a kernel strategy, not a model hyperparameter.
        #[allow(dead_code)]
        activation_scheme: String,
        /// The 32x32 block the UE8M0 scale grid assumes.
        weight_block_size: [usize; 2],
        /// `ue8m0`: the scale format `linalg` decodes.
        scale_fmt: String,
        /// `fp4`: the expert format the plan describes.
        expert_dtype: String,
    }

    /// `text_config`: the released names of the runtime hyperparameters.
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Text {
        /// `deepseek_v41_text`.
        model_type: String,
        /// Vocabulary rows.
        vocab_size: usize,
        /// Residual width; the runtime name is `dim`.
        hidden_size: usize,
        /// One routed expert's intermediate width.
        moe_intermediate_size: usize,
        /// Backbone blocks.
        num_hidden_layers: usize,
        /// DSpark stages; the runtime name is `n_mtp_layers`.
        num_nextn_predict_layers: usize,
        /// Query heads.
        num_attention_heads: usize,
        /// KV heads: the reference's MLA has exactly one.
        num_key_value_heads: usize,
        /// MLA latent width, including the rotated slice.
        head_dim: usize,
        /// The rotated slice of `head_dim`.
        qk_rope_head_dim: usize,
        /// Query LoRA rank.
        q_lora_rank: usize,
        /// Output LoRA rank.
        o_lora_rank: usize,
        /// Output projection groups.
        o_groups: usize,
        /// Routed experts per backbone block.
        n_routed_experts: usize,
        /// Shared experts per block.
        n_shared_experts: usize,
        /// Experts per token per block.
        num_experts_per_tok: usize,
        /// `sqrtsoftplus`, `sigmoid` or `softmax`.
        scoring_func: String,
        /// Multiplier after normalisation; the runtime name is `route_scale`.
        routed_scaling_factor: f64,
        /// SwiGLU gate clamp.
        swiglu_limit: f64,
        /// RMSNorm epsilon; the runtime name is `norm_eps`.
        rms_norm_eps: f64,
        /// RoPE base.
        rope_theta: f64,
        /// Sliding window width; the runtime name is `window_size`.
        sliding_window: usize,
        /// Per-layer compression ratio.
        compress_ratios: Vec<usize>,
        /// Compressor RoPE base.
        compress_rope_theta: f64,
        /// Layers that own the main KV.
        kv_source_layer_ids: Vec<usize>,
        /// Layers that run the indexer.
        index_source_layer_ids: Vec<usize>,
        /// Indexer heads.
        index_n_heads: usize,
        /// Indexer head width.
        index_head_dim: usize,
        /// Index entries kept per query.
        index_topk: usize,
        /// The layer that builds the HSI candidate pool.
        candidate_source_layer_id: usize,
        /// Blocks in the pool.
        candidate_topk_blocks: usize,
        /// Tokens per block.
        candidate_block_size: usize,
        /// mHC stream count.
        hc_mult: usize,
        /// Sinkhorn iterations.
        hc_sinkhorn_iters: usize,
        /// mHC epsilon.
        hc_eps: f64,
        /// Layers with an Engram table.
        engram_layer_ids: Vec<usize>,
        /// Rows in each Engram table.
        engram_num_embeddings: Vec<usize>,
        /// Longest Engram n-gram.
        engram_max_ngram_size: usize,
        /// Engram vocabulary.
        engram_vocab_size: usize,
        /// Engram heads.
        engram_n_heads: usize,
        /// Engram head width.
        engram_head_dim: usize,
        /// Token the Engram compressor skips; the runtime name is `engram_pad_id`.
        engram_pad_token_id: u32,
        /// Compressed vocabulary.
        engram_compressed_vocab_size: usize,
        /// Tokens one DSpark block drafts.
        dspark_block_size: usize,
        /// Draft seed token.
        dspark_noise_token_id: u32,
        /// Backbone layers feeding the MTP stages.
        dspark_target_layer_ids: Vec<usize>,
        /// Markov head width.
        dspark_markov_rank: usize,
        /// Routed experts per MTP stage.
        dspark_n_routed_experts: usize,
        /// Experts per token per MTP stage; the runtime name is `dspark_n_activated_experts`.
        dspark_num_experts_per_tok: usize,
        /// Whether the top-k weights are renormalised. The released document states this; the
        /// runtime document omits it and relies on the reference default.
        norm_topk_prob: bool,
        /// Activation function. Checked as `silu` rather than mapped: no runtime field carries it.
        hidden_act: String,
        /// Whether attention has a bias. The released MLA has none.
        attention_bias: bool,
        /// Dropout probability: inference-only, ignored.
        #[allow(dead_code)]
        attention_dropout: f64,
        /// Weight init range: training-only, ignored.
        #[allow(dead_code)]
        initializer_range: f64,
        /// The model's maximum positions. The runtime document carries no sequence-length field and
        /// the reference passes `max_seq_len` separately, so this is ignored.
        #[allow(dead_code)]
        max_position_embeddings: usize,
        /// Whether the forward returns the cache. The harness owns the KV cache, so this is ignored.
        #[allow(dead_code)]
        use_cache: bool,
        /// Whether the head shares the embedding table. The index stores `head.weight` separately, so
        /// the released value is checked false.
        tie_word_embeddings: bool,
        /// The expert-selection method. Checked as `noaux_tc`, which is what the gate implements.
        topk_method: String,
        /// RoPE scaling.
        rope_scaling: Scaling,
    }

    /// `vision_config`: the tower the released checkpoint carries.
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Vision {
        /// `deepseek_v41_vision`.
        model_type: String,
        /// Tower depth.
        num_hidden_layers: usize,
        /// Tower width.
        hidden_size: usize,
        /// Tower heads.
        num_attention_heads: usize,
        /// Tower MLP width.
        intermediate_size: usize,
        /// Patch edge.
        patch_size: usize,
        /// Tower RoPE base.
        rope_theta: f64,
        /// Patch downsample ratio.
        downsample_ratio: usize,
        /// Most tokens one image contributes.
        max_image_tokens: usize,
        /// Fewest pixels an image may have.
        min_pixels: usize,
        /// Aspect-ratio bound, `null` in the released file.
        max_wh_ratio: Option<f64>,
    }

    /// `rope_scaling`: the YaRN parameters.
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Scaling {
        /// `yarn`: the only scaling this loader accepts.
        rope_type: String,
        /// The scaling factor.
        factor: f64,
        /// Correction bound (fast).
        beta_fast: f64,
        /// Correction bound (slow).
        beta_slow: f64,
        /// The trained length the scaling is anchored to; the runtime name is `original_seq_len`.
        original_max_position_embeddings: usize,
    }

    /// Map the released document onto the runtime configuration.
    pub(super) fn from_value(value: serde_json::Value) -> Result<ModelConfig, ConfigError> {
        let document: Document =
            serde_json::from_value(value).map_err(|error| ConfigError::Json(error.to_string()))?;
        let Document {
            architectures,
            model_type,
            image_token_id,
            quantization_config: quant,
            text_config: text,
            vision_config: vision,
            ..
        } = document;

        if model_type != "deepseek_v41"
            || !architectures
                .iter()
                .any(|name| name == "DeepseekV41ForCausalLM")
        {
            return Err(ConfigError::Json(format!(
                "the document is not a DeepSeek-V4.1-Flash configuration: model_type {model_type:?}, \
                 architectures {architectures:?}"
            )));
        }
        if text.model_type != "deepseek_v41_text" {
            return Err(ConfigError::Invalid {
                field: "text_config.model_type",
                detail: format!(
                    "the released document says {:?}, not \"deepseek_v41_text\"",
                    text.model_type
                ),
            });
        }
        if vision.model_type != "deepseek_v41_vision" {
            return Err(ConfigError::Invalid {
                field: "vision_config.model_type",
                detail: format!(
                    "the released document says {:?}, not \"deepseek_v41_vision\"",
                    vision.model_type
                ),
            });
        }
        require("text_config.hidden_act", &text.hidden_act, "silu")?;
        require("text_config.topk_method", &text.topk_method, "noaux_tc")?;
        require(
            "text_config.rope_scaling.rope_type",
            &text.rope_scaling.rope_type,
            "yarn",
        )?;
        require(
            "quantization_config.quant_method",
            &quant.quant_method,
            "fp8",
        )?;
        require(
            "quantization_config.expert_dtype",
            &quant.expert_dtype,
            "fp4",
        )?;
        require("quantization_config.scale_fmt", &quant.scale_fmt, "ue8m0")?;
        require_flag("text_config.attention_bias", text.attention_bias, false)?;
        require_flag(
            "text_config.tie_word_embeddings",
            text.tie_word_embeddings,
            false,
        )?;
        if quant.weight_block_size != [32, 32] {
            return Err(ConfigError::Invalid {
                field: "quantization_config.weight_block_size",
                detail: format!(
                    "the plan describes 32x32 UE8M0 scales, the released document says {:?}",
                    quant.weight_block_size
                ),
            });
        }
        if text.num_key_value_heads != 1 {
            return Err(ConfigError::Invalid {
                field: "text_config.num_key_value_heads",
                detail: format!(
                    "the reference's MLA has exactly one KV head, the released document says {}",
                    text.num_key_value_heads
                ),
            });
        }

        Ok(ModelConfig {
            vocab_size: text.vocab_size,
            dim: text.hidden_size,
            moe_inter_dim: text.moe_intermediate_size,
            n_layers: text.num_hidden_layers,
            n_mtp_layers: text.num_nextn_predict_layers,
            dspark_block_size: text.dspark_block_size,
            dspark_noise_token_id: text.dspark_noise_token_id,
            dspark_target_layer_ids: text.dspark_target_layer_ids,
            dspark_markov_rank: text.dspark_markov_rank,
            dspark_n_routed_experts: text.dspark_n_routed_experts,
            dspark_n_activated_experts: text.dspark_num_experts_per_tok,
            n_heads: text.num_attention_heads,
            n_routed_experts: text.n_routed_experts,
            n_shared_experts: text.n_shared_experts,
            n_activated_experts: text.num_experts_per_tok,
            score_func: text.scoring_func,
            route_scale: text.routed_scaling_factor,
            swiglu_limit: text.swiglu_limit,
            q_lora_rank: text.q_lora_rank,
            head_dim: text.head_dim,
            rope_head_dim: text.qk_rope_head_dim,
            norm_eps: text.rms_norm_eps,
            o_groups: text.o_groups,
            o_lora_rank: text.o_lora_rank,
            window_size: text.sliding_window,
            kv_source_layers: text.kv_source_layer_ids,
            index_source_layers: text.index_source_layer_ids,
            original_seq_len: text.rope_scaling.original_max_position_embeddings,
            rope_theta: text.rope_theta,
            rope_factor: text.rope_scaling.factor,
            beta_fast: text.rope_scaling.beta_fast,
            beta_slow: text.rope_scaling.beta_slow,
            index_n_heads: text.index_n_heads,
            index_head_dim: text.index_head_dim,
            index_topk: text.index_topk,
            candidate_source_layer: text.candidate_source_layer_id,
            candidate_topk_blocks: text.candidate_topk_blocks,
            candidate_block_size: text.candidate_block_size,
            hc_mult: text.hc_mult,
            hc_sinkhorn_iters: text.hc_sinkhorn_iters,
            hc_eps: text.hc_eps,
            engram_layer_ids: text.engram_layer_ids,
            engram_vocab_size: text.engram_vocab_size,
            engram_num_embeddings: text.engram_num_embeddings,
            engram_max_ngram_size: text.engram_max_ngram_size,
            engram_pad_id: text.engram_pad_token_id,
            engram_compressed_vocab_size: text.engram_compressed_vocab_size,
            engram_n_heads: text.engram_n_heads,
            engram_head_dim: text.engram_head_dim,
            // Not the safetensors storage dtype: the plan needs the linear-weight format, which the
            // released document states in its quantisation block.
            dtype: quant.quant_method.clone(),
            expert_dtype: quant.expert_dtype.clone(),
            compress_rope_theta: text.compress_rope_theta,
            compress_ratios: text.compress_ratios,
            image_token_id,
            gate_temp: default_gate_temp(),
            norm_topk_prob: text.norm_topk_prob,
            vision_n_layers: Some(vision.num_hidden_layers),
            vision_dim: Some(vision.hidden_size),
            vision_n_heads: Some(vision.num_attention_heads),
            vision_inter_dim: Some(vision.intermediate_size),
            vision_patch_size: Some(vision.patch_size),
            vision_downsample_ratio: Some(vision.downsample_ratio),
            vision_max_n_token: Some(vision.max_image_tokens),
            vision_min_pixels: Some(vision.min_pixels),
            vision_max_wh_ratio: vision.max_wh_ratio,
            vision_rope_theta: Some(vision.rope_theta),
        })
    }

    /// Check a string field against the value the engine's other code assumes.
    fn require(field: &'static str, actual: &str, expected: &str) -> Result<(), ConfigError> {
        if actual == expected {
            Ok(())
        } else {
            Err(ConfigError::Invalid {
                field,
                detail: format!(
                    "the released document says {actual:?}, but the engine assumes {expected:?}"
                ),
            })
        }
    }

    /// Check a flag against the value the engine's other code assumes.
    fn require_flag(field: &'static str, actual: bool, expected: bool) -> Result<(), ConfigError> {
        if actual == expected {
            Ok(())
        } else {
            Err(ConfigError::Invalid {
                field,
                detail: format!(
                    "the released document says {actual}, but the engine assumes {expected}"
                ),
            })
        }
    }
}

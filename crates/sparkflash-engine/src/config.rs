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

use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

/// The runtime configuration carried in this crate.
pub const V41_FLASH_RUNTIME_CONFIG: &str =
    include_str!("../config/deepseek_v41_flash.runtime.json");

/// What is wrong with a configuration.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    /// The JSON could not be parsed, or a required key was absent.
    #[error("could not parse the configuration: {0}")]
    Json(String),
    /// A parsed value contradicts another.
    #[error("{field}: {detail}")]
    Invalid {
        /// The field the complaint is about.
        field: &'static str,
        /// What is wrong with it.
        detail: String,
    },
}

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
    pub fn from_json(text: &str) -> Result<Self, ConfigError> {
        let config: ModelConfig =
            serde_json::from_str(text).map_err(|e| ConfigError::Json(e.to_string()))?;
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
            compress_ratios: vec![2, 2, 1],
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

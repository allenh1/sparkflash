// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The typed Inkling-2-Small configuration.
//!
//! The released document is a Transformers config: a `text_config`, a `vision_config` and a
//! nested `mtp_config`. The shared [Architecture] parser already names the parts a run needs;
//! this type keeps the numbers the Inkling arithmetic reads directly (the relative bank, the
//! block schedule, the short-convolution width and the mixed-precision layer split) and refuses a
//! document whose geometry cannot be served.

use std::path::Path;

use serde_json::Value;
use sparkflash_model::architecture::{AttentionKind, Ffn, Residual, WeightFormat};
use sparkflash_model::{Architecture, ConfigError};

use crate::tensor::ExpertSplit;

/// The released model type.
pub const MODEL_TYPE: &str = "inkling2";
/// The canonical Hugging Face id.
pub const MODEL_ID: &str = "thinkingmachines/Inkling-2-Small";
/// The control-plane weight selector.
pub const WEIGHTS_ID: &str = "inkling2-small-nvfp4";

/// The released text stack.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2Config {
    /// The shared architecture, including the per-layer schedule and quant formats.
    pub architecture: Architecture,
    /// Residual width.
    pub hidden_size: usize,
    /// Padded vocabulary rows.
    pub vocab_size: usize,
    /// Scored vocabulary rows.
    pub unpadded_vocab_size: usize,
    /// Decoder layers.
    pub num_hidden_layers: usize,
    /// Query heads.
    pub num_attention_heads: usize,
    /// Key/value heads.
    pub num_key_value_heads: usize,
    /// Head width.
    pub head_dim: usize,
    /// Learned relative-bias width.
    pub d_rel: usize,
    /// Relative-bias extent.
    pub rel_extent: usize,
    /// Sliding-window width (0 for a full layer).
    pub sliding_window_size: usize,
    /// The sliding layers, ascending.
    pub local_layer_ids: Vec<usize>,
    /// The full-attention layers, ascending.
    pub full_layer_ids: Vec<usize>,
    /// Static short-convolution width.
    pub sconv_kernel_size: usize,
    /// Query/key L2 clip.
    pub qk_l2_clip_threshold: f64,
    /// Relative-state L2 clip.
    pub rel_l2_clip_threshold: f64,
    /// Query/relative log-scale floor.
    pub log_scaling_n_floor: usize,
    /// Query/relative log-scale coefficient.
    pub log_scaling_alpha: f64,
    /// Gate softcap.
    pub mlp_gate_softcap: f64,
    /// Up softcap.
    pub mlp_up_softcap: f64,
    /// Final logit softcap.
    pub final_logit_softcap: Option<f64>,
    /// Routed experts.
    pub n_routed_experts: usize,
    /// Active experts.
    pub num_experts_per_tok: usize,
    /// One expert's intermediate width.
    pub moe_intermediate_size: usize,
    /// The routed multiplier.
    pub route_scale: f64,
    /// Layers in one block-attention residual block.
    pub n_layers_per_attn_res_block: usize,
    /// Whether the block summary stack is normalized.
    pub attn_res_normalize_values: bool,
    /// The vision tower, when carried.
    pub vision: Option<InklingVisionConfig>,
    /// The DSpark/MTP stage, when carried.
    pub mtp: Option<InklingMtpConfig>,
    /// The mixed-precision split.
    pub quant: InklingQuantConfig,
    /// How the rank files divide the routed experts. Not a released field: the converter sets
    /// it and a rank file's load takes it from the file.
    pub expert_split: ExpertSplit,
}

/// The vision tower's released geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct InklingVisionConfig {
    /// Transformer blocks.
    pub depth: usize,
    /// Tower width.
    pub hidden_size: usize,
    /// Attention heads.
    pub heads: usize,
    /// Patch side.
    pub patch_size: usize,
    /// Frames per patch.
    pub temporal_patch_size: usize,
    /// Spatial merge before the adapter.
    pub adapter_merge_size: usize,
    /// Adapter hidden width.
    pub projection_hidden_dim: usize,
    /// Position-grid side.
    pub posemb_grid_size: usize,
    /// LayerNorm epsilon.
    pub layer_norm_eps: f64,
    /// Width handed the language model.
    pub out_hidden_size: usize,
}

/// The DSpark/MTP stage's released geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct InklingMtpConfig {
    /// Stage depth.
    pub num_hidden_layers: usize,
    /// Stage width.
    pub hidden_size: usize,
    /// Query heads.
    pub num_attention_heads: usize,
    /// Key/value heads.
    pub num_key_value_heads: usize,
    /// Head width.
    pub head_dim: usize,
    /// Dense MLP width.
    pub intermediate_size: usize,
    /// Stage sliding layers.
    pub local_layer_ids: Vec<usize>,
    /// Stage sliding window.
    pub sliding_window_size: usize,
    /// Draft block.
    pub block_size: usize,
    /// RNN Markov rank.
    pub markov_rank: usize,
    /// Backbone layers whose hidden states feed the stage.
    pub target_layer_ids: Vec<usize>,
    /// RoPE base.
    pub rope_theta: f64,
    /// The stage's RMSNorm epsilon.
    pub rms_norm_eps: f64,
}

/// The stored format of each class of weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InklingQuantConfig {
    /// Backbone expert format, per layer.
    pub layer_expert_formats: Vec<WeightFormat>,
    /// Dense projection format.
    pub dense_format: WeightFormat,
    /// The first NVFP4 group.
    pub nvfp4_group: usize,
    /// The MXFP8 block.
    pub mxfp8_block: usize,
    /// NVFP4 layers.
    pub nvfp4_layers: Vec<usize>,
    /// MXFP8 layers.
    pub mxfp8_layers: Vec<usize>,
    /// MTP storage format.
    pub mtp_format: WeightFormat,
}

impl Inkling2Config {
    /// Parse the released `config.json` and the optional `hf_quant_config.json`.
    ///
    /// # Errors
    /// A malformed document, a geometry outside the served shape, or a quant group whose
    /// arithmetic has no typed format.
    pub fn from_json(text: &str, quant: Option<&str>) -> Result<Self, ConfigError> {
        let root: Value =
            serde_json::from_str(text).map_err(|error| ConfigError::Json(error.to_string()))?;
        let text_config = object(&root, "text_config").unwrap_or(&root);
        let mut architecture = Architecture::from_value(&root)?;
        let quant_value = match quant {
            Some(text) => Some(
                serde_json::from_str::<Value>(text)
                    .map_err(|error| ConfigError::Json(error.to_string()))?,
            ),
            None => None,
        };
        if let Some(value) = &quant_value {
            architecture.apply_quant_config(value)?;
        }
        let parsed = Self::from_parts(text_config, &root, architecture, quant_value.as_ref())?;
        parsed.validate()?;
        Ok(parsed)
    }

    /// Read a checkpoint directory, including its external quant document.
    ///
    /// # Errors
    /// A missing or malformed document, or as [Self::from_json].
    pub fn from_checkpoint_dir(directory: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(directory.join("config.json"))
            .map_err(|error| ConfigError::Json(format!("config.json: {error}")))?;
        let quant_path = directory.join("hf_quant_config.json");
        let quant = if quant_path.is_file() {
            Some(
                std::fs::read_to_string(&quant_path)
                    .map_err(|error| ConfigError::Json(format!("hf_quant_config.json: {error}")))?,
            )
        } else {
            None
        };
        Self::from_json(&text, quant.as_deref())
    }

    fn from_parts(
        text: &Value,
        root: &Value,
        architecture: Architecture,
        quant: Option<&Value>,
    ) -> Result<Self, ConfigError> {
        let hidden_size = architecture.hidden_size;
        let vocab_size = architecture.vocab_size;
        let num_hidden_layers = architecture.layers.len();
        let num_attention_heads = required_usize(text, "num_attention_heads")?;
        let num_key_value_heads = required_usize(text, "num_key_value_heads")?;
        let head_dim = required_usize(text, "head_dim")?;
        let local_layer_ids = ascending(usize_list(text.get("local_layer_ids"))?);
        let full_layer_ids = (0..num_hidden_layers)
            .filter(|index| !local_layer_ids.contains(index))
            .collect();
        let inkling = architecture.inkling.clone().ok_or_else(|| {
            ConfigError::Invalid("the document is not an Inkling architecture".to_owned())
        })?;
        let n_routed_experts = match architecture.layers.first().map(|layer| &layer.ffn) {
            Some(Ffn::Moe(moe)) => moe.routed_experts,
            _ => {
                return Err(ConfigError::Invalid(
                    "the first layer is not MoE".to_owned(),
                ));
            }
        };
        let num_experts_per_tok = match architecture.layers.first().map(|layer| &layer.ffn) {
            Some(Ffn::Moe(moe)) => moe.active_experts,
            _ => 0,
        };
        let moe_intermediate_size = match architecture.layers.first().map(|layer| &layer.ffn) {
            Some(Ffn::Moe(moe)) => moe.intermediate,
            _ => 0,
        };
        let route_scale =
            optional_f64(text, "route_scale")?.unwrap_or_else(|| {
                match architecture.layers.first().map(|layer| &layer.ffn) {
                    Some(Ffn::Moe(moe)) => moe.routed_scaling,
                    _ => 1.0,
                }
            });
        let block = inkling.block_residual;
        let quant_config = parse_quant(quant, &inkling.layer_expert_formats, num_hidden_layers)?;
        let vision = parse_vision(root, hidden_size)?;
        let mtp = parse_mtp(root)?;
        Ok(Self {
            architecture,
            hidden_size,
            vocab_size,
            unpadded_vocab_size: inkling.unpadded_vocab_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            d_rel: inkling.d_rel,
            rel_extent: inkling.rel_extent,
            sliding_window_size: optional_usize(text, "sliding_window_size")?.unwrap_or(0),
            local_layer_ids,
            full_layer_ids,
            sconv_kernel_size: inkling.short_conv,
            qk_l2_clip_threshold: inkling.qk_l2_clip,
            rel_l2_clip_threshold: inkling.rel_l2_clip,
            log_scaling_n_floor: inkling.log_scaling.map_or(0, |scaling| scaling.n_floor),
            log_scaling_alpha: inkling.log_scaling.map_or(0.0, |scaling| scaling.alpha),
            mlp_gate_softcap: inkling.softcaps.gate,
            mlp_up_softcap: inkling.softcaps.up,
            final_logit_softcap: inkling.final_logit_softcap,
            n_routed_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            route_scale,
            n_layers_per_attn_res_block: block.map_or(0, |block| block.layers_per_block),
            attn_res_normalize_values: block.is_some_and(|block| block.normalize),
            vision,
            mtp,
            quant: quant_config,
            expert_split: ExpertSplit::default(),
        })
    }

    /// Refuse a shape the arithmetic cannot run.
    ///
    /// # Errors
    /// A zero or ragged dimension, a layer schedule that disagrees with the parser, or a quant
    /// split that names a layer outside the stack.
    pub fn validate(&self) -> Result<(), ConfigError> {
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("vocab_size", self.vocab_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("d_rel", self.d_rel),
            ("rel_extent", self.rel_extent),
            ("n_routed_experts", self.n_routed_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("sconv_kernel_size", self.sconv_kernel_size),
        ] {
            if value == 0 {
                return Err(ConfigError::Invalid(format!("{name} is zero")));
            }
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(ConfigError::Invalid(format!(
                "{} query heads do not divide {} key/value heads",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if !self.local_layer_ids.len().is_multiple_of(1)
            || self.local_layer_ids.len() + self.full_layer_ids.len() != self.num_hidden_layers
        {
            return Err(ConfigError::Invalid(
                "the sliding and full layer schedules do not partition the stack".to_owned(),
            ));
        }
        if self.sliding_window_size == 0 {
            return Err(ConfigError::Invalid(
                "sliding_window_size is zero".to_owned(),
            ));
        }
        if self.n_layers_per_attn_res_block == 0
            || !self
                .num_hidden_layers
                .is_multiple_of(self.n_layers_per_attn_res_block)
        {
            return Err(ConfigError::Invalid(format!(
                "{} layers do not divide into blocks of {}",
                self.num_hidden_layers, self.n_layers_per_attn_res_block
            )));
        }
        if self.quant.layer_expert_formats.len() != self.num_hidden_layers {
            return Err(ConfigError::Invalid(format!(
                "the quant split names {} layers, expected {}",
                self.quant.layer_expert_formats.len(),
                self.num_hidden_layers
            )));
        }
        // 201,024 is 64-aligned; at TP4 the 50,256-row rank slice is padded to 50,304 (a
        // 128-row tile) by the loader/head, not by the model's declared vocabulary.
        if !self.vocab_size.is_multiple_of(64) {
            return Err(ConfigError::Invalid(format!(
                "vocab_size {} is not a multiple of 64",
                self.vocab_size
            )));
        }
        if self.unpadded_vocab_size > self.vocab_size {
            return Err(ConfigError::Invalid(format!(
                "unpadded vocabulary {} exceeds the padded {}",
                self.unpadded_vocab_size, self.vocab_size
            )));
        }
        Ok(())
    }

    /// A two-layer synthetic release for parity tests.
    ///
    /// # Panics
    /// Only if the carried JSON stops parsing, which is a crate bug.
    #[must_use]
    pub fn tiny() -> Self {
        const TINY: &str = r#"{
          "model_type":"inkling2",
          "text_config":{
            "hidden_size":32,"num_hidden_layers":2,"vocab_size":256,"max_position_embeddings":64,
            "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
            "d_rel":4,"rel_extent":8,"qk_l2_clip_threshold":22.627416997969522,
            "rel_l2_clip_threshold":8.0,"log_scaling_n_floor":16,"log_scaling_alpha":0.1,
            "rms_norm_eps":1e-06,"use_value_norm_no_gamma":true,
            "local_layer_ids":[0],"sconv_kernel_size":4,"unpadded_vocab_size":250,
            "final_logit_softcapping":30.0,"sliding_window_size":8,
            "n_layers_per_attn_res_block":1,"attn_res_normalize_values":true,
            "n_routed_experts":2,"num_experts_per_tok":1,"n_shared_experts":0,
            "intermediate_size":16,"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,
            "route_scale":8.0,"use_gate_bias":true
          }
        }"#;
        Self::from_json(TINY, None).expect("the tiny Inkling config is valid")
    }

    /// A four-expert variant of [Self::tiny], for the tensor-parallel tests.
    ///
    /// Every axis a four-rank plan shards divides four here and does not in [Self::tiny]: four
    /// routed experts rather than two, and two active experts so a token's mix spans more than
    /// one rank.
    ///
    /// # Panics
    /// Only if the carried JSON stops parsing, which is a crate bug.
    #[must_use]
    pub fn tiny_world() -> Self {
        const TINY: &str = r#"{
          "model_type":"inkling2",
          "text_config":{
            "hidden_size":32,"num_hidden_layers":2,"vocab_size":256,"max_position_embeddings":64,
            "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
            "d_rel":4,"rel_extent":8,"qk_l2_clip_threshold":22.627416997969522,
            "rel_l2_clip_threshold":8.0,"log_scaling_n_floor":16,"log_scaling_alpha":0.1,
            "rms_norm_eps":1e-06,"use_value_norm_no_gamma":true,
            "local_layer_ids":[0],"sconv_kernel_size":4,"unpadded_vocab_size":250,
            "final_logit_softcapping":30.0,"sliding_window_size":8,
            "n_layers_per_attn_res_block":1,"attn_res_normalize_values":true,
            "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":0,
            "intermediate_size":16,"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,
            "route_scale":8.0,"use_gate_bias":true
          }
        }"#;
        Self::from_json(TINY, None).expect("the four-expert fixture is valid")
    }

    /// The shared architecture.
    #[must_use]
    pub fn architecture(&self) -> &Architecture {
        &self.architecture
    }

    /// The context window the model document declares.
    #[must_use]
    pub fn context_tokens(&self) -> usize {
        self.architecture.max_position_embeddings
    }

    /// The token a step hands back to serve: `chosen`, or `None` when it ends the turn.
    ///
    /// The released checkpoint names `<|content_model_end_sampling|>` (200006) as its only
    /// end-of-generation id, which is what the template appends after a model turn. Without this
    /// the host driver decodes past the end of the answer and the model opens a new turn.
    #[must_use]
    pub fn next_token(&self, chosen: u32) -> Option<u32> {
        Some(chosen).filter(|token| !self.architecture.tokens.eos_token_ids.contains(token))
    }

    /// The sliding layers' attention kind, or an error for a mis-shaped layer.
    #[must_use]
    pub fn is_sliding(&self, layer: usize) -> bool {
        matches!(
            self.architecture
                .layers
                .get(layer)
                .map(|layer| &layer.attention.kind),
            Some(AttentionKind::Inkling { .. })
        ) && self.local_layer_ids.contains(&layer)
    }

    /// The short-conv history width a layer retains.
    #[must_use]
    pub fn conv_history(&self) -> usize {
        self.sconv_kernel_size.saturating_sub(1)
    }

    /// The per-layer expert format.
    #[must_use]
    pub fn layer_format(&self, layer: usize) -> Option<WeightFormat> {
        self.quant.layer_expert_formats.get(layer).copied()
    }

    /// The residual kind named by the document.
    #[must_use]
    pub fn block_residual(&self) -> bool {
        self.architecture
            .layers
            .iter()
            .any(|layer| matches!(layer.residual, Residual::BlockAttention))
    }
}

fn parse_quant(
    quant: Option<&Value>,
    formats: &[WeightFormat],
    layers: usize,
) -> Result<InklingQuantConfig, ConfigError> {
    let mut layer_expert_formats = formats.to_vec();
    if layer_expert_formats.len() != layers {
        layer_expert_formats.resize(layers, WeightFormat::Bf16);
    }
    let mut nvfp4_layers = Vec::new();
    let mut mxfp8_layers = Vec::new();
    for (layer, format) in layer_expert_formats.iter().enumerate() {
        match format {
            WeightFormat::NvFp4 { .. } => nvfp4_layers.push(layer),
            WeightFormat::MxFp8 { .. } => mxfp8_layers.push(layer),
            _ => {}
        }
    }
    let nvfp4_group = layer_expert_formats
        .iter()
        .find_map(|format| match format {
            WeightFormat::NvFp4 { group_size } => Some(*group_size),
            _ => None,
        })
        .unwrap_or(16);
    let mxfp8_block = layer_expert_formats
        .iter()
        .find_map(|format| match format {
            WeightFormat::MxFp8 { block } => Some(*block),
            _ => None,
        })
        .unwrap_or(32);
    let mtp_format = quant
        .and_then(|value| object(value, "mtp"))
        .and_then(|mtp| mtp.get("parameter_groups"))
        .and_then(Value::as_array)
        .and_then(|groups| {
            groups.iter().find_map(|group| {
                let compute = string(group, "compute_dtype").unwrap_or_default();
                match compute.as_str() {
                    "nvfp4" => Some(WeightFormat::NvFp4 {
                        group_size: nvfp4_group,
                    }),
                    "mxfp8" => Some(WeightFormat::MxFp8 { block: mxfp8_block }),
                    _ => None,
                }
            })
        })
        .unwrap_or(WeightFormat::Bf16);
    // The dense tensors are BF16 in every release; each one's dtype is checked against the plan
    // on load, so the config's torch_dtype decides nothing here.
    let dense_format = WeightFormat::Bf16;
    Ok(InklingQuantConfig {
        layer_expert_formats,
        dense_format,
        nvfp4_group,
        mxfp8_block,
        nvfp4_layers,
        mxfp8_layers,
        mtp_format,
    })
}

fn parse_vision(
    root: &Value,
    language_hidden: usize,
) -> Result<Option<InklingVisionConfig>, ConfigError> {
    let Some(config) = object(root, "vision_config") else {
        return Ok(None);
    };
    let hidden = required_usize(config, "hidden_size")?;
    let patch = optional_usize(config, "native_patch_size")?
        .or(optional_usize(config, "patch_size")?)
        .ok_or_else(|| ConfigError::Invalid("vision_config names no patch size".to_owned()))?;
    let merge = optional_usize(config, "adapter_merge_size")?
        .or(optional_usize(config, "spatial_merge_size")?)
        .unwrap_or(1);
    let projection = optional_usize(config, "projection_hidden_dim")?
        .or(optional_usize(config, "projection_intermediate_size")?)
        .unwrap_or(0);
    let grid = optional_usize(config, "posemb_grid_size")?
        .or(optional_usize(config, "num_position_embeddings")?)
        .unwrap_or(0);
    Ok(Some(InklingVisionConfig {
        depth: required_usize(config, "num_hidden_layers")?,
        hidden_size: hidden,
        heads: required_usize(config, "num_attention_heads")?,
        patch_size: patch,
        temporal_patch_size: optional_usize(config, "temporal_patch_size")?.unwrap_or(1),
        adapter_merge_size: merge,
        projection_hidden_dim: projection,
        posemb_grid_size: grid,
        layer_norm_eps: optional_f64(config, "layer_norm_eps")?.unwrap_or(1e-5),
        out_hidden_size: optional_usize(config, "out_hidden_size")?.unwrap_or(language_hidden),
    }))
}

fn parse_mtp(root: &Value) -> Result<Option<InklingMtpConfig>, ConfigError> {
    let Some(config) = object(root, "mtp_config") else {
        return Ok(None);
    };
    Ok(Some(InklingMtpConfig {
        num_hidden_layers: required_usize(config, "num_hidden_layers")?,
        hidden_size: required_usize(config, "hidden_size")?,
        num_attention_heads: required_usize(config, "num_attention_heads")?,
        num_key_value_heads: required_usize(config, "num_key_value_heads")?,
        head_dim: required_usize(config, "head_dim")?,
        intermediate_size: required_usize(config, "intermediate_size")?,
        local_layer_ids: usize_list(config.get("local_layer_ids"))?,
        sliding_window_size: optional_usize(config, "sliding_window_size")?.unwrap_or(0),
        block_size: optional_usize(config, "block_size")?.unwrap_or(0),
        markov_rank: optional_usize(config, "markov_rank")?.unwrap_or(0),
        target_layer_ids: usize_list(config.get("target_layer_ids"))?,
        rope_theta: optional_f64(config, "rope_theta")?.unwrap_or(1_000_000.0),
        rms_norm_eps: optional_f64(config, "rms_norm_eps")?.unwrap_or(1e-6),
    }))
}

fn ascending(mut values: Vec<usize>) -> Vec<usize> {
    values.sort_unstable();
    values.dedup();
    values
}

fn object<'a>(value: &'a Value, name: &str) -> Option<&'a Value> {
    value.get(name).filter(|value| value.is_object())
}

fn string(value: &Value, name: &str) -> Option<String> {
    value.get(name).and_then(Value::as_str).map(str::to_owned)
}

fn optional_usize(value: &Value, name: &str) -> Result<Option<usize>, ConfigError> {
    match value.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .map(Some)
            .ok_or_else(|| ConfigError::Invalid(format!("{name} is not an integer"))),
    }
}

fn required_usize(value: &Value, name: &str) -> Result<usize, ConfigError> {
    optional_usize(value, name)?.ok_or_else(|| ConfigError::Invalid(format!("{name} is missing")))
}

fn optional_f64(value: &Value, name: &str) -> Result<Option<f64>, ConfigError> {
    match value.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_f64()
            .map(Some)
            .ok_or_else(|| ConfigError::Invalid(format!("{name} is not a number"))),
    }
}

fn usize_list(value: Option<&Value>) -> Result<Vec<usize>, ConfigError> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                item.as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| {
                        ConfigError::Invalid("a layer list holds a non-integer".to_owned())
                    })
            })
            .collect(),
        Some(_) => Err(ConfigError::Invalid(
            "a layer list is not an array".to_owned(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = include_str!("../config/inkling2_small.hf.json");
    const QUANT: &str = include_str!("../config/inkling2_small.hf_quant.json");

    #[test]
    fn the_released_document_parses_its_mixed_precision_split() {
        let config = Inkling2Config::from_json(CONFIG, Some(QUANT)).expect("the release parses");
        assert_eq!(config.num_hidden_layers, 48);
        assert_eq!(config.local_layer_ids.len(), 40);
        assert_eq!(config.full_layer_ids, vec![5, 11, 17, 23, 29, 35, 41, 47]);
        assert_eq!(config.quant.nvfp4_layers.len(), 40);
        assert_eq!(
            config.quant.mxfp8_layers,
            vec![40, 41, 42, 43, 44, 45, 46, 47]
        );
        assert_eq!(config.quant.nvfp4_group, 16);
        assert_eq!(config.quant.mxfp8_block, 32);
        assert_eq!(config.vision.as_ref().map(|v| v.patch_size), Some(20));
        assert_eq!(config.vision.as_ref().map(|v| v.depth), Some(23));
        assert_eq!(config.mtp.as_ref().map(|m| m.num_hidden_layers), Some(5));
        assert_eq!(config.mtp.as_ref().map(|m| m.block_size), Some(8));
    }
}

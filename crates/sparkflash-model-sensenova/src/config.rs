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

//! The released sensenova/SenseNova-Loop-9B configuration.
//!
//! The checkpoint is a Qwen3.5 hybrid decoder with two additions this arm has to execute
//! exactly: a shared-weight loop over physical layers `[4, 28)` applied twice, and two-stream
//! manifold-constrained hyper-connections over the same range. The released `config.json`
//! describes both; this parser turns them into a typed execution plan and refuses a document
//! whose plan cannot be run one visit at a time.

use std::path::Path;

use serde::Deserialize;
use sparkflash_model::ConfigError;
use sparkflash_model::config::OneOrMany;

/// The served model id.
pub const MODEL_ID: &str = sparkflash_model::descriptor::SENSENOVA_LOOP_9B_ID;

/// The control-plane weight selector.
pub const WEIGHTS_ID: &str = sparkflash_model::descriptor::SENSENOVA_LOOP_9B_WEIGHTS_ID;

/// The released `model_type`.
pub const MODEL_TYPE: &str = "sensenova_loop";

/// One physical layer's attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LayerType {
    /// Gated DeltaNet.
    Linear,
    /// Full grouped-query attention.
    Full,
}

/// The looped execution schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopSettings {
    /// First physical layer of the shared block.
    pub start: usize,
    /// One past the last physical layer of the shared block.
    pub end: usize,
    /// How many times the shared block runs.
    pub count: usize,
}

/// Two-stream manifold-constrained hyper-connections.
#[derive(Debug, Clone, PartialEq)]
pub struct MhcSettings {
    /// First physical layer that carries streams.
    pub start: usize,
    /// One past the last physical layer that carries streams.
    pub end: usize,
    /// Residual streams (the release uses two).
    pub streams: usize,
    /// Sinkhorn iterations.
    pub iterations: usize,
    /// Whether the mix is a fixed identity.
    pub fixed_identity: bool,
    /// The branch scale.
    pub pre_scale: f64,
    /// The router input's RMS epsilon.
    pub epsilon: f64,
}

/// The rotary rule of the full-attention layers.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeSettings {
    /// Base frequency.
    pub theta: f64,
    /// Dimensions rotated per head (partial rotary).
    pub dims: usize,
    /// mRoPE section split, when declared.
    pub mrope_section: Option<[usize; 3]>,
    /// YaRN scale factor, or 1.0 for plain RoPE.
    pub factor: f64,
    /// The window the factor was trained at.
    pub original: usize,
    /// YaRN fast correction bound.
    pub beta_fast: f64,
    /// YaRN slow correction bound.
    pub beta_slow: f64,
}

/// Gated DeltaNet geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearSettings {
    /// Convolution width.
    pub conv_kernel: usize,
    /// Key heads.
    pub key_heads: usize,
    /// Value heads.
    pub value_heads: usize,
    /// Key head width.
    pub key_dim: usize,
    /// Value head width.
    pub value_dim: usize,
}

/// The stored weight format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Storage {
    /// Unquantized BF16.
    #[default]
    Bf16,
    /// E4M3 with one BF16/F32 inverse scale a 128x128 tile.
    Fp8Tile128,
}

/// The vision tower's geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionSettings {
    /// Decoder blocks.
    pub depth: usize,
    /// Hidden width.
    pub hidden_size: usize,
    /// Patch side.
    pub patch_size: usize,
    /// Spatial merge factor.
    pub spatial_merge_size: usize,
    /// Temporal patch factor.
    pub temporal_patch_size: usize,
    /// Output width.
    pub out_hidden_size: usize,
}

/// The typed release configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SenseNovaConfig {
    /// The checkpoint's declared name.
    pub model_name: String,
    /// Vocabulary rows.
    pub vocab_size: usize,
    /// Residual width (one stream).
    pub hidden_size: usize,
    /// Dense MLP intermediate width.
    pub intermediate_size: usize,
    /// Physical decoder layers.
    pub num_hidden_layers: usize,
    /// Query heads.
    pub num_attention_heads: usize,
    /// Key/value heads.
    pub num_key_value_heads: usize,
    /// Head width.
    pub head_dim: usize,
    /// Maximum positions (512K with YaRN).
    pub max_position_embeddings: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// Whether a projection carries a bias.
    pub attention_bias: bool,
    /// Whether the LM head shares the embedding.
    pub tie_word_embeddings: bool,
    /// One entry per physical layer.
    pub layer_types: Vec<LayerType>,
    /// The GDN geometry.
    pub linear: LinearSettings,
    /// The full-attention rotary rule.
    pub rope: RopeSettings,
    /// The looped block.
    pub loop_settings: LoopSettings,
    /// The mHC range.
    pub mhc: MhcSettings,
    /// The vision tower, when the checkpoint carries one.
    pub vision: Option<VisionSettings>,
    /// The dense-projection storage.
    pub storage: Storage,
    /// Beginning-of-sequence id.
    pub bos_token_id: Option<u32>,
    /// End-of-generation ids.
    pub eos_token_ids: Vec<u32>,
    /// Padding id.
    pub pad_token_id: Option<u32>,
    /// Image placeholder id.
    pub image_token_id: Option<u32>,
    /// Video placeholder id.
    pub video_token_id: Option<u32>,
    /// Vision start delimiter.
    pub vision_start_token_id: Option<u32>,
    /// Vision end delimiter.
    pub vision_end_token_id: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    #[serde(default)]
    model_name: Option<String>,
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    loop_settings: Option<RawLoop>,
    #[serde(default)]
    mhc: Option<RawMhc>,
    #[serde(default)]
    vision_config: Option<RawVision>,
    #[serde(default)]
    quantization_config: Option<serde_json::Value>,
    #[serde(default)]
    text_config: RawText,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    eos_token_id: Option<OneOrMany<u32>>,
    #[serde(default)]
    pad_token_id: Option<u32>,
    #[serde(default)]
    image_token_id: Option<u32>,
    #[serde(default)]
    video_token_id: Option<u32>,
    #[serde(default)]
    vision_start_token_id: Option<u32>,
    #[serde(default)]
    vision_end_token_id: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
struct RawText {
    hidden_size: usize,
    vocab_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    layer_types: Vec<String>,
    linear_conv_kernel_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    #[serde(default)]
    rope_parameters: Option<RawRope>,
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
    #[serde(default)]
    eos_token_id: Option<OneOrMany<u32>>,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    pad_token_id: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
struct RawLoop {
    start: usize,
    end: usize,
    #[serde(default)]
    counts: Vec<usize>,
    #[serde(default)]
    eval_count: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
struct RawMhc {
    start: usize,
    end: usize,
    streams: usize,
    iterations: usize,
    #[serde(default)]
    fixed_identity: bool,
    #[serde(default = "one")]
    pre_scale: f64,
}

#[derive(Debug, Default, Deserialize)]
struct RawRope {
    #[serde(default)]
    rope_theta: f64,
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
    #[serde(default)]
    mrope_section: Option<Vec<usize>>,
    #[serde(default)]
    factor: Option<f64>,
    #[serde(default)]
    original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    beta_fast: Option<f64>,
    #[serde(default)]
    beta_slow: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
struct RawVision {
    depth: usize,
    hidden_size: usize,
    patch_size: usize,
    spatial_merge_size: usize,
    temporal_patch_size: usize,
    out_hidden_size: usize,
}

const fn one() -> f64 {
    1.0
}

impl SenseNovaConfig {
    /// Parse a released `config.json`.
    ///
    /// # Errors
    /// A malformed document, an execution plan this pack cannot describe, or a geometry it
    /// cannot serve.
    pub fn from_json(text: &str) -> Result<Self, ConfigError> {
        let raw: RawConfig =
            serde_json::from_str(text).map_err(|error| ConfigError::Json(error.to_string()))?;
        if raw
            .model_type
            .as_deref()
            .is_some_and(|kind| kind != MODEL_TYPE)
        {
            return Err(ConfigError::Invalid(format!(
                "model_type {:?} is not {MODEL_TYPE}",
                raw.model_type
            )));
        }
        let text = raw.text_config;
        let loop_settings = raw.loop_settings.ok_or_else(|| {
            ConfigError::Invalid("the document names no loop_settings".to_owned())
        })?;
        let mhc = raw
            .mhc
            .ok_or_else(|| ConfigError::Invalid("the document names no mhc settings".to_owned()))?;
        let count = match loop_settings.eval_count {
            Some(count) if loop_settings.counts.is_empty() || loop_settings.counts == [count] => {
                count
            }
            Some(count) if loop_settings.counts == [count] => count,
            _ => {
                return Err(ConfigError::Invalid(
                    "only a fixed-count loop checkpoint is supported".to_owned(),
                ));
            }
        };
        let layers = text.num_hidden_layers;
        if count == 0 || loop_settings.start >= loop_settings.end || loop_settings.end > layers {
            return Err(ConfigError::Invalid(format!(
                "invalid loop interval or count: start={}, end={}, count={count}, layers={layers}",
                loop_settings.start, loop_settings.end
            )));
        }
        if mhc.streams < 2 || mhc.iterations == 0 || mhc.start >= mhc.end || mhc.end > layers {
            return Err(ConfigError::Invalid(format!(
                "invalid mhc range: start={}, end={}, streams={}, iterations={}",
                mhc.start, mhc.end, mhc.streams, mhc.iterations
            )));
        }
        let inside_loop = |start: usize, end: usize| {
            (loop_settings.start..=loop_settings.end)
                .any(|boundary| boundary == start || boundary == end)
        };
        if !inside_loop(mhc.start, mhc.end) {
            return Err(ConfigError::Invalid(
                "the mHC range crosses a loop phase boundary".to_owned(),
            ));
        }
        let layer_types: Vec<LayerType> = text
            .layer_types
            .iter()
            .map(|kind| match kind.as_str() {
                "linear_attention" => Ok(LayerType::Linear),
                "full_attention" => Ok(LayerType::Full),
                other => Err(ConfigError::Invalid(format!(
                    "unknown layer type {other:?}"
                ))),
            })
            .collect::<Result<_, _>>()?;
        if layer_types.len() != layers {
            return Err(ConfigError::Invalid(format!(
                "layer_types has {} entries, expected {layers}",
                layer_types.len()
            )));
        }
        let rope = text.rope_parameters.unwrap_or_default();
        let partial = text
            .partial_rotary_factor
            .or(rope.partial_rotary_factor)
            .unwrap_or(0.0);
        let dims = ((text.head_dim as f64 * partial).round() as usize).min(text.head_dim);
        let mrope_section = rope
            .mrope_section
            .and_then(|sections| <[usize; 3]>::try_from(sections.as_slice()).ok());
        let storage = raw
            .quantization_config
            .as_ref()
            .and_then(|config| config.get("quant_method"))
            .and_then(serde_json::Value::as_str)
            .map_or(Storage::Bf16, |method| {
                if method == "fp8" {
                    Storage::Fp8Tile128
                } else {
                    Storage::Bf16
                }
            });
        let mut eos = raw
            .eos_token_id
            .map(OneOrMany::into_vec)
            .unwrap_or_default();
        if let Some(ids) = text.eos_token_id {
            eos.extend(ids.into_vec());
        }
        eos.sort_unstable();
        eos.dedup();
        Ok(Self {
            model_name: raw
                .model_name
                .unwrap_or_else(|| "SenseNova-Loop-9B".to_owned()),
            vocab_size: text.vocab_size,
            hidden_size: text.hidden_size,
            intermediate_size: text.intermediate_size,
            num_hidden_layers: text.num_hidden_layers,
            num_attention_heads: text.num_attention_heads,
            num_key_value_heads: text.num_key_value_heads,
            head_dim: text.head_dim,
            max_position_embeddings: text.max_position_embeddings,
            rms_norm_eps: text.rms_norm_eps,
            attention_bias: text.attention_bias,
            tie_word_embeddings: text.tie_word_embeddings,
            layer_types,
            linear: LinearSettings {
                conv_kernel: text.linear_conv_kernel_dim,
                key_heads: text.linear_num_key_heads,
                value_heads: text.linear_num_value_heads,
                key_dim: text.linear_key_head_dim,
                value_dim: text.linear_value_head_dim,
            },
            rope: RopeSettings {
                theta: rope.rope_theta,
                dims,
                mrope_section,
                factor: rope.factor.unwrap_or(1.0),
                original: rope
                    .original_max_position_embeddings
                    .unwrap_or(text.max_position_embeddings),
                beta_fast: rope.beta_fast.unwrap_or(32.0),
                beta_slow: rope.beta_slow.unwrap_or(1.0),
            },
            loop_settings: LoopSettings {
                start: loop_settings.start,
                end: loop_settings.end,
                count,
            },
            mhc: MhcSettings {
                start: mhc.start,
                end: mhc.end,
                streams: mhc.streams,
                iterations: mhc.iterations,
                fixed_identity: mhc.fixed_identity,
                pre_scale: mhc.pre_scale,
                epsilon: 1e-6,
            },
            vision: raw.vision_config.map(|vision| VisionSettings {
                depth: vision.depth,
                hidden_size: vision.hidden_size,
                patch_size: vision.patch_size,
                spatial_merge_size: vision.spatial_merge_size,
                temporal_patch_size: vision.temporal_patch_size,
                out_hidden_size: vision.out_hidden_size,
            }),
            storage,
            bos_token_id: raw.bos_token_id.or(text.bos_token_id),
            eos_token_ids: eos,
            pad_token_id: raw.pad_token_id.or(text.pad_token_id),
            image_token_id: raw.image_token_id,
            video_token_id: raw.video_token_id,
            vision_start_token_id: raw.vision_start_token_id,
            vision_end_token_id: raw.vision_end_token_id,
        })
    }

    /// Read a checkpoint's `config.json` and merge its `generation_config.json`.
    ///
    /// # Errors
    /// A read error or [Self::from_json].
    pub fn from_checkpoint_dir(directory: &Path) -> Result<Self, ConfigError> {
        let path = directory.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|error| ConfigError::Json(format!("{}: {error}", path.display())))?;
        let mut config = Self::from_json(&text)?;
        let generation = directory.join("generation_config.json");
        if generation.is_file() {
            let text = std::fs::read_to_string(&generation)
                .map_err(|error| ConfigError::Json(error.to_string()))?;
            let value: serde_json::Value = serde_json::from_str(&text)
                .map_err(|error| ConfigError::Json(error.to_string()))?;
            if let Some(ids) = value.get("eos_token_id") {
                let ids: OneOrMany<u32> = serde_json::from_value(ids.clone())
                    .map_err(|error| ConfigError::Json(error.to_string()))?;
                config.eos_token_ids.extend(ids.into_vec());
                config.eos_token_ids.sort_unstable();
                config.eos_token_ids.dedup();
            }
        }
        Ok(config)
    }

    /// The physical layers in forward order, exactly as the reference's `execution_plan` does.
    #[must_use]
    pub fn execution_plan(&self) -> Vec<usize> {
        let settings = &self.loop_settings;
        let mut plan: Vec<usize> = (0..settings.start).collect();
        for _ in 0..settings.count {
            plan.extend(settings.start..settings.end);
        }
        plan.extend(settings.end..self.num_hidden_layers);
        plan
    }

    /// Number of visits, i.e. layer passes per token.
    #[must_use]
    pub fn visits(&self) -> usize {
        self.execution_plan().len()
    }

    /// Key/value width of one position.
    #[must_use]
    pub fn kv_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Width of the fused GDN projection.
    #[must_use]
    pub fn linear_projection_width(&self) -> usize {
        2 * self.linear.key_heads * self.linear.key_dim
            + self.linear.value_heads * self.linear.value_dim
    }

    /// Width of the GDN output gate.
    #[must_use]
    pub fn linear_value_width(&self) -> usize {
        self.linear.value_heads * self.linear.value_dim
    }

    /// The number of full-attention visits, and so KV caches, at this plan.
    #[must_use]
    pub fn full_attention_visits(&self) -> usize {
        self.execution_plan()
            .into_iter()
            .filter(|index| self.layer_types[*index] == LayerType::Full)
            .count()
    }

    /// The number of GDN visits, and so recurrent state slots, at this plan.
    #[must_use]
    pub fn linear_attention_visits(&self) -> usize {
        self.execution_plan()
            .into_iter()
            .filter(|index| self.layer_types[*index] == LayerType::Linear)
            .count()
    }
}

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

//! The released nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 configuration.
//!
//! Dory is a looped normalized-GPT (nGPT): the 72 stored sublayers are 36 physical
//! transformer blocks, each an attention sublayer followed by an MLP sublayer. The first
//! n_input_layers sublayers run once, the next n_recurrent_layers sublayers form a shared
//! block that is applied n_recurrent_loops times, and the last n_output_layers sublayers
//! run once. The residual is not an addition: every sublayer writes a normalized interpolation
//! between its input and its branch output.
//!
//! The parser is strict about the invariants the reference implementation checks. A checkpoint
//! whose schedule cannot be described one block at a time is refused at load rather than served
//! with a silently different depth.

use std::collections::BTreeMap;

use serde::Deserialize;
use sparkflash_model::ConfigError;
use sparkflash_model::config::OneOrMany;

/// The served model id.
pub const MODEL_ID: &str = sparkflash_model::descriptor::DORY1_0_4B_BF16_ID;

/// The control-plane weight selector.
pub const WEIGHTS_ID: &str = sparkflash_model::descriptor::DORY1_0_4B_BF16_WEIGHTS_ID;

/// The released model_type.
pub const MODEL_TYPE: &str = "dory";

/// The nGPT normalization epsilon, hard-coded by the reference implementation.
pub const NGPT_EPSILON: f32 = 1e-6;

/// The attention kind of one physical transformer block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AttentionType {
    /// Sliding-window causal attention.
    Sliding,
    /// Full causal attention.
    Full,
}

impl AttentionType {
    /// The released spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sliding => "sliding_attention",
            Self::Full => "full_attention",
        }
    }
}

/// Which part of the looped schedule a block belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Partition {
    /// Runs once before the recurrent block.
    Input,
    /// The shared block, applied n_recurrent_loops times.
    Recurrent,
    /// Runs once after the recurrent block.
    Output,
}

/// How the recurrent block's KV cache is kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvCacheMode {
    /// A separate cache for every loop.
    #[default]
    PerLoop,
    /// One cache, overwritten by each loop.
    LastLoop,
}

/// One attention-MLP pair of the stored schedule.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    /// Zero-based block index (not the stored sublayer index).
    pub index: usize,
    /// The stored attention sublayer index.
    pub attention_layer: usize,
    /// The stored MLP sublayer index.
    pub mlp_layer: usize,
    /// The partition this block belongs to.
    pub partition: Partition,
    /// The attention arithmetic.
    pub attention: AttentionType,
    /// The sliding window, when the attention has one.
    pub window: Option<usize>,
    /// The rotary base of the attention.
    pub rope_theta: f64,
    /// How many times this block runs.
    pub loops: usize,
    /// How many independent KV caches this block owns.
    pub cache_slots: usize,
}

/// The typed configuration the host reference runs.
#[derive(Debug, Clone, PartialEq)]
pub struct DoryConfig {
    /// Vocabulary rows.
    pub vocab_size: usize,
    /// Residual-stream width.
    pub hidden_size: usize,
    /// Dense MLP intermediate width.
    pub intermediate_size: usize,
    /// Stored sublayers (attention and MLP entries together).
    pub num_hidden_layers: usize,
    /// Query heads.
    pub num_attention_heads: usize,
    /// Key/value heads.
    pub num_key_value_heads: usize,
    /// Head width.
    pub head_dim: usize,
    /// Maximum positions.
    pub max_position_embeddings: usize,
    /// Sliding-window width.
    pub sliding_window: usize,
    /// Sublayers before the recurrent block.
    pub n_input_layers: usize,
    /// Sublayers in the recurrent block.
    pub n_recurrent_layers: usize,
    /// Sublayers after the recurrent block.
    pub n_output_layers: usize,
    /// How many times the recurrent block runs.
    pub n_recurrent_loops: usize,
    /// The recurrent cache policy.
    pub recurrent_kv_cache_mode: KvCacheMode,
    /// nGPT alpha initializer; the residual lerp multiplies this by 1 / initializer_range.
    pub ngpt_alpha_init: f64,
    /// The output-gate logit-scale initializer, or None for sqrt(hidden_size).
    pub attention_output_gate_logit_scale_init: Option<f64>,
    /// The initializer range used by every learned nGPT scale.
    pub initializer_range: f64,
    /// The activation name.
    pub hidden_act: String,
    /// Whether the LM head shares the embedding.
    pub tie_word_embeddings: bool,
    /// Whether an attention projection carries a bias.
    pub attention_bias: bool,
    /// Whether an MLP projection carries a bias.
    pub mlp_bias: bool,
    /// The released physical schedule.
    pub layer_types: Vec<String>,
    /// The rotary base of each attention kind.
    pub rope_theta: BTreeMap<AttentionType, f64>,
    /// Beginning-of-sequence id.
    pub bos_token_id: Option<u32>,
    /// End-of-generation ids.
    pub eos_token_ids: Vec<u32>,
    /// Padding id.
    pub pad_token_id: Option<u32>,
    /// The derived block schedule.
    pub blocks: Vec<Block>,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    sliding_window: Option<usize>,
    n_input_layers: usize,
    n_recurrent_layers: Option<usize>,
    n_output_layers: usize,
    n_recurrent_loops: usize,
    recurrent_kv_cache_mode: Option<String>,
    ngpt_alpha_init: f64,
    attention_output_gate_logit_scale_init: Option<f64>,
    initializer_range: f64,
    hidden_act: String,
    tie_word_embeddings: bool,
    attention_bias: bool,
    mlp_bias: bool,
    layer_types: Vec<String>,
    rope_parameters: BTreeMap<String, RawRope>,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    eos_token_id: Option<OneOrMany<u32>>,
    #[serde(default)]
    pad_token_id: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct RawRope {
    rope_type: Option<String>,
    rope_theta: f64,
    partial_rotary_factor: Option<f64>,
}

impl DoryConfig {
    /// Parse a released config.json.
    ///
    /// # Errors
    /// A malformed document or a schedule this arm cannot describe.
    pub fn from_json(text: &str) -> Result<Self, ConfigError> {
        let raw: RawConfig =
            serde_json::from_str(text).map_err(|error| ConfigError::Json(error.to_string()))?;
        Self::from_raw(raw)
    }

    /// Read a released config.json from a path.
    ///
    /// # Errors
    /// A read error or [Self::from_json].
    pub fn from_path(path: &std::path::Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| ConfigError::Json(format!("{}: {error}", path.display())))?;
        Self::from_json(&text)
    }

    /// Read a checkpoint's config.json and merge its generation_config.json.
    ///
    /// # Errors
    /// A read error or [Self::from_json].
    pub fn from_checkpoint_dir(directory: &std::path::Path) -> Result<Self, ConfigError> {
        let mut config = Self::from_path(&directory.join("config.json"))?;
        let generation = directory.join("generation_config.json");
        if generation.is_file() {
            let text = std::fs::read_to_string(&generation)
                .map_err(|error| ConfigError::Json(format!("{}: {error}", generation.display())))?;
            let value: serde_json::Value = serde_json::from_str(&text)
                .map_err(|error| ConfigError::Json(error.to_string()))?;
            if let Some(ids) = value.get("eos_token_id") {
                let ids: OneOrMany<u32> = serde_json::from_value(ids.clone())
                    .map_err(|error| ConfigError::Json(error.to_string()))?;
                let mut merged = config.eos_token_ids.clone();
                merged.extend(ids.into_vec());
                merged.sort_unstable();
                merged.dedup();
                config.eos_token_ids = merged;
            }
            if let Some(id) = value
                .get("bos_token_id")
                .and_then(serde_json::Value::as_u64)
            {
                config.bos_token_id.get_or_insert(id as u32);
            }
            if let Some(id) = value
                .get("pad_token_id")
                .and_then(serde_json::Value::as_u64)
            {
                config.pad_token_id.get_or_insert(id as u32);
            }
        }
        Ok(config)
    }

    fn from_raw(raw: RawConfig) -> Result<Self, ConfigError> {
        if raw.vocab_size == 0
            || raw.hidden_size == 0
            || raw.intermediate_size == 0
            || raw.num_attention_heads == 0
            || raw.num_key_value_heads == 0
            || raw.head_dim == 0
        {
            return Err(invalid("every model dimension must be positive"));
        }
        if !raw
            .num_attention_heads
            .is_multiple_of(raw.num_key_value_heads)
        {
            return Err(invalid(
                "num_attention_heads must be divisible by num_key_value_heads",
            ));
        }
        if raw.initializer_range <= 0.0 {
            return Err(invalid("initializer_range must be positive"));
        }
        if raw.hidden_act != "silu" {
            return Err(invalid(format!(
                "Dory requires hidden_act = \"silu\", not {:?}",
                raw.hidden_act
            )));
        }
        if raw.tie_word_embeddings {
            return Err(invalid("Dory requires untied embeddings"));
        }
        if raw.attention_bias || raw.mlp_bias {
            return Err(invalid("Dory requires bias-free projections"));
        }
        if raw.layer_types.len() != raw.num_hidden_layers {
            return Err(invalid(format!(
                "layer_types has {} entries, expected num_hidden_layers = {}",
                raw.layer_types.len(),
                raw.num_hidden_layers
            )));
        }
        if !raw.num_hidden_layers.is_multiple_of(2) {
            return Err(invalid(
                "Dory stores attention and MLP sublayers in pairs, so num_hidden_layers must be even",
            ));
        }
        let n_recurrent_layers = raw.n_recurrent_layers.unwrap_or(
            raw.num_hidden_layers
                .saturating_sub(raw.n_input_layers + raw.n_output_layers),
        );
        let partitions = (raw.n_input_layers, n_recurrent_layers, raw.n_output_layers);
        if partitions.0 + partitions.1 + partitions.2 != raw.num_hidden_layers {
            return Err(invalid(format!(
                "input/recurrent/output layers {:?} do not sum to num_hidden_layers = {}",
                partitions, raw.num_hidden_layers
            )));
        }
        for (name, count) in [
            ("n_input_layers", partitions.0),
            ("n_recurrent_layers", partitions.1),
            ("n_output_layers", partitions.2),
        ] {
            if count == 0 || !count.is_multiple_of(2) {
                return Err(invalid(format!(
                    "{name} = {count} must be a positive even count of attention/MLP pairs"
                )));
            }
        }
        if raw.n_recurrent_loops == 0 {
            return Err(invalid("n_recurrent_loops must be positive"));
        }
        let mode = match raw.recurrent_kv_cache_mode.as_deref().unwrap_or("per_loop") {
            "per_loop" => KvCacheMode::PerLoop,
            "last_loop" => KvCacheMode::LastLoop,
            other => {
                return Err(invalid(format!(
                    "recurrent_kv_cache_mode must be per_loop or last_loop, not {other:?}"
                )));
            }
        };
        if raw
            .attention_output_gate_logit_scale_init
            .is_some_and(|value| value <= 0.0)
        {
            return Err(invalid(
                "attention_output_gate_logit_scale_init must be positive when set",
            ));
        }
        let sliding_window = raw.sliding_window.unwrap_or(0);
        let mut rope_theta = BTreeMap::new();
        for (name, kind) in [
            ("sliding_attention", AttentionType::Sliding),
            ("full_attention", AttentionType::Full),
        ] {
            let Some(rope) = raw.rope_parameters.get(name) else {
                continue;
            };
            if rope
                .rope_type
                .as_deref()
                .is_some_and(|kind| kind != "default")
            {
                return Err(invalid(format!(
                    "{name} rope_type must be default, not {:?}",
                    rope.rope_type
                )));
            }
            if rope.rope_theta <= 0.0 {
                return Err(invalid(format!("{name} rope_theta must be positive")));
            }
            if rope
                .partial_rotary_factor
                .is_some_and(|fraction| fraction != 1.0)
            {
                return Err(invalid(format!(
                    "{name} partial_rotary_factor must be 1.0 for this arm"
                )));
            }
            rope_theta.insert(kind, rope.rope_theta);
        }

        let input_blocks = raw.n_input_layers / 2;
        let recurrent_blocks = n_recurrent_layers / 2;
        let output_blocks = raw.n_output_layers / 2;
        let mut blocks = Vec::with_capacity(input_blocks + recurrent_blocks + output_blocks);
        for block_index in 0..raw.num_hidden_layers / 2 {
            let attention_layer = block_index * 2;
            let mlp_layer = attention_layer + 1;
            let attention = match raw.layer_types[attention_layer].as_str() {
                "sliding_attention" => AttentionType::Sliding,
                "full_attention" => AttentionType::Full,
                other => {
                    return Err(invalid(format!(
                        "block {block_index} attention sublayer is {other:?}, expected \
                         sliding_attention or full_attention"
                    )));
                }
            };
            if raw.layer_types[mlp_layer] != "mlp" {
                return Err(invalid(format!(
                    "block {block_index} MLP sublayer is {:?}, expected \"mlp\"",
                    raw.layer_types[mlp_layer]
                )));
            }
            if attention == AttentionType::Sliding && sliding_window == 0 {
                return Err(invalid(
                    "a sliding-attention block needs a positive sliding_window",
                ));
            }
            let theta = *rope_theta.get(&attention).ok_or_else(|| {
                invalid(format!(
                    "rope_parameters does not name {:?} for block {block_index}",
                    attention.name()
                ))
            })?;
            let partition = if block_index < input_blocks {
                Partition::Input
            } else if block_index < input_blocks + recurrent_blocks {
                Partition::Recurrent
            } else if block_index < input_blocks + recurrent_blocks + output_blocks {
                Partition::Output
            } else {
                return Err(invalid(format!(
                    "block {block_index} is outside the input/recurrent/output partition"
                )));
            };
            let loops = match partition {
                Partition::Recurrent => raw.n_recurrent_loops,
                Partition::Input | Partition::Output => 1,
            };
            let cache_slots = match (partition, mode) {
                (Partition::Recurrent, KvCacheMode::PerLoop) => raw.n_recurrent_loops,
                _ => 1,
            };
            blocks.push(Block {
                index: block_index,
                attention_layer,
                mlp_layer,
                partition,
                attention,
                window: (attention == AttentionType::Sliding).then_some(sliding_window),
                rope_theta: theta,
                loops,
                cache_slots,
            });
        }

        let eos_token_ids = raw
            .eos_token_id
            .map(OneOrMany::into_vec)
            .unwrap_or_default();
        Ok(Self {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            max_position_embeddings: raw.max_position_embeddings,
            sliding_window,
            n_input_layers: raw.n_input_layers,
            n_recurrent_layers,
            n_output_layers: raw.n_output_layers,
            n_recurrent_loops: raw.n_recurrent_loops,
            recurrent_kv_cache_mode: mode,
            ngpt_alpha_init: raw.ngpt_alpha_init,
            attention_output_gate_logit_scale_init: raw.attention_output_gate_logit_scale_init,
            initializer_range: raw.initializer_range,
            hidden_act: raw.hidden_act,
            tie_word_embeddings: raw.tie_word_embeddings,
            attention_bias: raw.attention_bias,
            mlp_bias: raw.mlp_bias,
            layer_types: raw.layer_types,
            rope_theta,
            bos_token_id: raw.bos_token_id,
            eos_token_ids,
            pad_token_id: raw.pad_token_id,
            blocks,
        })
    }

    /// The tiny fixture's configuration document.
    ///
    /// This is the same JSON a checkpoint would carry, so a device test can write it beside
    /// the fixture's safetensors file.
    #[must_use]
    pub fn tiny_json() -> String {
        serde_json::json!({
            "model_type": "dory",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 32,
            "sliding_window": 3,
            "n_input_layers": 2,
            "n_recurrent_layers": 4,
            "n_output_layers": 2,
            "n_recurrent_loops": 3,
            "recurrent_kv_cache_mode": "per_loop",
            "ngpt_alpha_init": 0.05,
            "attention_output_gate_logit_scale_init": null,
            "initializer_range": 0.02,
            "hidden_act": "silu",
            "tie_word_embeddings": false,
            "attention_bias": false,
            "mlp_bias": false,
            "layer_types": [
                "sliding_attention", "mlp",
                "sliding_attention", "mlp",
                "sliding_attention", "mlp",
                "full_attention", "mlp"
            ],
            "rope_parameters": {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 100.0,
                    "partial_rotary_factor": 1.0
                },
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0
                }
            }
        })
        .to_string()
    }

    /// The tiny fixture used by the host-reference tests.
    ///
    /// Eight stored sublayers: one input pair, two recurrent pairs, one output pair, applied
    /// three times. The geometry follows the reference test's fixture.
    #[must_use]
    pub fn tiny() -> Self {
        Self::from_json(&Self::tiny_json()).expect("the tiny fixture is valid")
    }

    /// The tiny fixture with a context long enough for a sliding cache to wrap.
    ///
    /// The device test runs a 4200-token prompt: a sliding block's ring is its 8-row window
    /// plus one 4096-token prefill chunk, so the tail chunks overwrite rows the first chunk
    /// wrote and the parity check covers the wrap.
    pub fn tiny_ring_json() -> String {
        let mut value: serde_json::Value =
            serde_json::from_str(&Self::tiny_json()).expect("the tiny fixture is JSON");
        value["max_position_embeddings"] = serde_json::json!(5000);
        value["sliding_window"] = serde_json::json!(8);
        // The tensor-core tile needs both sides in 32-column steps; 64 / 96 / 64 does that
        // while keeping the fixture small.
        value["vocab_size"] = serde_json::json!(64);
        value["hidden_size"] = serde_json::json!(64);
        value["intermediate_size"] = serde_json::json!(96);
        value["num_attention_heads"] = serde_json::json!(4);
        value["num_key_value_heads"] = serde_json::json!(2);
        value["head_dim"] = serde_json::json!(16);
        value.to_string()
    }

    /// The ring-exercising tiny fixture.
    pub fn tiny_ring() -> Self {
        Self::from_json(&Self::tiny_ring_json()).expect("the ring fixture is valid")
    }

    /// The token a step hands back to serve: `chosen`, or `None` when it ends the turn, so
    /// the host stops at `<|im_end|>` instead of decoding past it.
    #[must_use]
    pub fn next_token(&self, chosen: u32) -> Option<u32> {
        Some(chosen).filter(|token| !self.eos_token_ids.contains(token))
    }

    /// The number of physical transformer blocks.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// The blocks in the input partition.
    #[must_use]
    pub fn input_blocks(&self) -> &[Block] {
        &self.blocks[..self.n_input_layers / 2]
    }

    /// The blocks in the shared recurrent partition.
    #[must_use]
    pub fn recurrent_blocks(&self) -> &[Block] {
        let start = self.n_input_layers / 2;
        &self.blocks[start..start + self.n_recurrent_layers / 2]
    }

    /// The blocks in the output partition.
    #[must_use]
    pub fn output_blocks(&self) -> &[Block] {
        let start = self.n_input_layers / 2 + self.n_recurrent_layers / 2;
        &self.blocks[start..]
    }

    /// The per-block residual alpha scale, ngpt_alpha_init / initializer_range.
    #[must_use]
    pub fn alpha_scale(&self) -> f64 {
        self.ngpt_alpha_init / self.initializer_range
    }

    /// The q/k scale factor, sqrt(head_dim) / initializer_range.
    #[must_use]
    pub fn qk_scale_factor(&self) -> f64 {
        (self.head_dim as f64).sqrt() / self.initializer_range
    }

    /// The output-gate scale factor, sqrt(sqrt(hidden_size)) / initializer_range.
    #[must_use]
    pub fn gate_scale_factor(&self) -> f64 {
        let gate_init = self
            .attention_output_gate_logit_scale_init
            .unwrap_or_else(|| (self.hidden_size as f64).sqrt());
        gate_init.sqrt() / self.initializer_range
    }

    /// The MLP scale factor, sqrt(hidden_size).
    #[must_use]
    pub fn mlp_scale_factor(&self) -> f64 {
        (self.hidden_size as f64).sqrt()
    }

    /// The LM-head logit scale divisor, initializer_range.
    #[must_use]
    pub fn logit_scale_divisor(&self) -> f64 {
        self.initializer_range
    }
}

fn invalid(message: impl Into<String>) -> ConfigError {
    ConfigError::Invalid(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    const RELEASED: &str = include_str!("../config/nvidia_nemotron_labs_dory1_0_4b_bf16.hf.json");

    #[test]
    fn the_released_configuration_parses_into_36_blocks() {
        let config = DoryConfig::from_json(RELEASED).expect("the released configuration parses");
        assert_eq!(config.num_blocks(), 36);
        assert_eq!(config.input_blocks().len(), 8);
        assert_eq!(config.recurrent_blocks().len(), 20);
        assert_eq!(config.output_blocks().len(), 8);
        assert_eq!(config.n_recurrent_loops, 2);
        assert_eq!(config.recurrent_kv_cache_mode, KvCacheMode::PerLoop);
        assert_eq!(config.blocks[0].attention, AttentionType::Sliding);
        assert_eq!(config.blocks[3].attention, AttentionType::Full);
        assert_eq!(config.blocks[3].window, None);
        assert_eq!(config.blocks[0].rope_theta, 10_000.0);
        assert_eq!(config.blocks[3].rope_theta, 10_000_000.0);
        assert_eq!(config.recurrent_blocks()[0].cache_slots, 2);
    }

    #[test]
    fn an_odd_partition_is_refused() {
        let mut value: serde_json::Value = serde_json::from_str(RELEASED).expect("json");
        value["n_input_layers"] = serde_json::json!(15);
        value["n_recurrent_layers"] = serde_json::json!(41);
        let error = DoryConfig::from_json(&value.to_string()).expect_err("odd partition");
        assert!(error.to_string().contains("positive even"));
    }

    #[test]
    fn a_last_loop_cache_keeps_one_slot() {
        let mut value: serde_json::Value = serde_json::from_str(RELEASED).expect("json");
        value["recurrent_kv_cache_mode"] = serde_json::json!("last_loop");
        let config = DoryConfig::from_json(&value.to_string()).expect("last-loop config");
        assert!(
            config
                .recurrent_blocks()
                .iter()
                .all(|block| block.cache_slots == 1)
        );
        assert!(
            config
                .recurrent_blocks()
                .iter()
                .all(|block| block.loops == 2)
        );
    }
}

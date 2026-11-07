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

//! The DFlash 2 block drafter's configuration.
//!
//! DFlash drafts a whole block at once: the token the target just chose (the anchor) and
//! `block_size - 1` mask positions run through a small non-causal transformer whose keys and
//! values for the committed context come from the target's own hidden states, captured at a few
//! of its layers. DFlash 2 wraps each sublayer in a grouped causal convolution across the block
//! and, instead of reading one token per mask row, walks a low-rank selector through the target
//! head's top candidates for each row, so the drafted tokens agree with one another.
//!
//! Inco AI releases the same drafter architecture for several targets, so this module parses a
//! drafter's `config.json` for any of them and checks it against the target it would draft for
//! ([DraftTarget]): a draft trained for another model or another layout is refused at load rather
//! than drafting garbage. A pack's device arm runs it.
//!
//! # Attribution
//!
//! * Block-diffusion speculative drafting is the DFlash method of Jian Chen, Yesheng Liang and
//!   Zhijian Liu ("DFlash: Block Diffusion for Flash Speculative Decoding", ICML 2026), with its
//!   reference implementation in [z-lab/dflash](https://github.com/z-lab/dflash) (MIT,
//!   Copyright (c) 2026 Z Lab).
//! * The DFlash 2 drafters are Inco AI's:
//!   [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
//!   (Apache-2.0) and
//!   [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
//!   (CC BY-NC-ND 4.0, for research and evaluation). Their weights are loaded from the user's
//!   copy, not redistributed.
//! * The forward and the verify protocol follow SGLang's DFlash support
//!   ([sgl-project/sglang](https://github.com/sgl-project/sglang), Apache-2.0). No source is
//!   copied. See the root NOTICE file.

use serde::Deserialize;

use crate::config::ConfigError;

/// The most rows a draft block may have: the verify forward takes them all, and the decode
/// GEMV keeps one register accumulator per row up to this many.
pub const MAX_BLOCK_SIZE: usize = 8;

/// The head width the drafter's attention kernel holds in a warp (Q4D_HEAD_DIM).
pub const HEAD_DIM: usize = 128;

/// The most candidates per row and the widest selector rank the walk stages (Q4D_MAX_TOPK and
/// Q4D_MAX_RANK).
pub const MAX_TOP_K: usize = 32;
/// The widest selector rank the walk stages (Q4D_MAX_RANK).
pub const MAX_RANK: usize = 1024;

/// What a drafter must agree with in the model it drafts for: the width of the captured rows, the
/// vocabulary its head and mask token index, and the layers it may capture. A pack converts its
/// configuration into one (`From<&Config>`), so [Dflash2Config::check_target] takes either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraftTarget {
    /// The target's residual width, which a captured feature row repeats per captured layer.
    pub hidden_size: usize,
    /// The target's vocabulary: the drafter's head scores it and the mask token indexes it.
    pub vocab_size: usize,
    /// The target's layer count; every captured layer id must be below it.
    pub num_hidden_layers: usize,
}

/// A DFlash 2 drafter's shape, as the device arm runs it.
#[derive(Debug, Clone, PartialEq)]
pub struct Dflash2Config {
    /// The drafter's residual width, which must equal the target's.
    pub hidden_size: usize,
    /// The SwiGLU intermediate width.
    pub intermediate_size: usize,
    /// The drafter's own layer count, which is not the target's.
    pub num_hidden_layers: usize,
    /// Query heads a layer.
    pub num_attention_heads: usize,
    /// Key and value heads a layer; the query heads group over them.
    pub num_key_value_heads: usize,
    /// The head width, which the kernels fix at [HEAD_DIM].
    pub head_dim: usize,
    /// The RMSNorm epsilon every norm in the block uses.
    pub rms_norm_eps: f64,
    /// The rotary base, from `rope_parameters` or the top-level field.
    pub rope_theta: f64,
    /// The vocabulary the head scores, which must equal the target's.
    pub vocab_size: usize,
    /// How many committed positions before a query it sees, besides the block: the HF
    /// `sliding_window` counts the query itself, so this is one less.
    pub window_left: usize,
    /// The block's rows: the anchor and `block_size - 1` drafts.
    pub block_size: usize,
    /// Channels one grouped-convolution tap covers; the hidden width is a whole number of them.
    pub conv_group_size: usize,
    /// Taps a grouped convolution takes, which is at most the block's rows.
    pub conv_kernel_size: usize,
    /// The token a block's mask positions start from.
    pub mask_token_id: u32,
    /// The selector's low rank, which is the codebooks' row width.
    pub selector_rank: usize,
    /// Candidates a row's selector walks.
    pub selector_top_k: usize,
    /// The target layers (zero-based) whose completed outputs the drafter reads, in feature
    /// order.
    pub target_layer_ids: Vec<usize>,
}

#[derive(Deserialize)]
struct RootConfig {
    #[serde(default)]
    architectures: Vec<String>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    vocab_size: usize,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    is_causal: Option<bool>,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    layer_types: Vec<String>,
    sliding_window: Option<usize>,
    rope_parameters: Option<RopeConfig>,
    rope_theta: Option<f64>,
    dflash_config: DflashSection,
}

#[derive(Deserialize)]
struct RopeConfig {
    rope_theta: Option<f64>,
    rope_type: Option<String>,
}

#[derive(Deserialize)]
struct DflashSection {
    block_size: usize,
    #[serde(default)]
    conv_group_size: usize,
    #[serde(default)]
    conv_kernel_size: usize,
    mask_token_id: u32,
    selector_rank: usize,
    selector_top_k: usize,
    target_layer_ids: Vec<usize>,
    /// Options of other DFlash drafters whose arithmetic this arm does not implement; any of them
    /// set means the checkpoint is not the drafter this arm runs.
    #[serde(default)]
    attention_value_scale: Option<f64>,
    #[serde(default)]
    attention_sink_bias: Option<bool>,
    #[serde(default)]
    output_multiplier: Option<f64>,
    #[serde(default)]
    final_logit_softcapping: Option<f64>,
    #[serde(default)]
    projector_type: Option<String>,
}

impl Dflash2Config {
    /// Parse a drafter's `config.json`.
    ///
    /// # Errors
    /// [ConfigError::Json] for a malformed document and [ConfigError::Invalid] for a drafter this
    /// arm does not run: another architecture, causal or full attention, an option whose
    /// arithmetic is not implemented, or a shape past the kernels' limits.
    pub fn from_json(text: &str) -> Result<Self, ConfigError> {
        let root: RootConfig =
            serde_json::from_str(text).map_err(|error| ConfigError::Json(error.to_string()))?;
        let invalid = |message: String| Err(ConfigError::Invalid(message));
        if !root
            .architectures
            .iter()
            .any(|name| name == "DFlash2DraftModel")
        {
            return invalid(format!(
                "architectures {:?} do not name DFlash2DraftModel",
                root.architectures
            ));
        }
        if root.hidden_act.as_deref().unwrap_or("silu") != "silu" {
            return invalid(format!(
                "hidden_act {:?}; only silu is run",
                root.hidden_act
            ));
        }
        if root.is_causal != Some(false) {
            return invalid("the drafter must be non-causal (is_causal: false)".to_owned());
        }
        if root.attention_bias {
            return invalid("attention_bias is not implemented".to_owned());
        }
        if root.layer_types.len() != root.num_hidden_layers
            || root
                .layer_types
                .iter()
                .any(|kind| kind != "sliding_attention")
        {
            return invalid(format!(
                "layer_types {:?}: every layer must be sliding_attention",
                root.layer_types
            ));
        }
        let Some(sliding_window) = root.sliding_window.filter(|window| *window > 0) else {
            return invalid("sliding_attention layers need a sliding_window".to_owned());
        };
        let rope_type = root
            .rope_parameters
            .as_ref()
            .and_then(|rope| rope.rope_type.as_deref())
            .unwrap_or("default");
        if rope_type != "default" {
            return invalid(format!("rope_type {rope_type:?}; only default is run"));
        }
        let rope_theta = root
            .rope_parameters
            .as_ref()
            .and_then(|rope| rope.rope_theta)
            .or(root.rope_theta)
            .unwrap_or(10_000.0);
        let section = root.dflash_config;
        if section.attention_value_scale.is_some()
            || section.attention_sink_bias == Some(true)
            || section.output_multiplier.is_some_and(|value| value != 1.0)
            || section.final_logit_softcapping.is_some()
            || section.projector_type.is_some()
        {
            return invalid(
                "the checkpoint sets a DFlash option (value scale, attention sinks, output \
                 multiplier, logit softcap or projector) this arm does not implement"
                    .to_owned(),
            );
        }
        let config = Self {
            hidden_size: root.hidden_size,
            intermediate_size: root.intermediate_size,
            num_hidden_layers: root.num_hidden_layers,
            num_attention_heads: root.num_attention_heads,
            num_key_value_heads: root.num_key_value_heads,
            head_dim: root.head_dim,
            rms_norm_eps: root.rms_norm_eps,
            rope_theta,
            vocab_size: root.vocab_size,
            window_left: sliding_window - 1,
            block_size: section.block_size,
            conv_group_size: section.conv_group_size,
            conv_kernel_size: section.conv_kernel_size,
            mask_token_id: section.mask_token_id,
            selector_rank: section.selector_rank,
            selector_top_k: section.selector_top_k,
            target_layer_ids: section.target_layer_ids,
        };
        config.check_limits()?;
        Ok(config)
    }

    fn check_limits(&self) -> Result<(), ConfigError> {
        let invalid = |message: String| Err(ConfigError::Invalid(message));
        if !(2..=MAX_BLOCK_SIZE).contains(&self.block_size) {
            return invalid(format!(
                "block_size {} is outside 2..={MAX_BLOCK_SIZE}",
                self.block_size
            ));
        }
        if self.head_dim != HEAD_DIM {
            return invalid(format!("head_dim {} is not {HEAD_DIM}", self.head_dim));
        }
        if self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return invalid(format!(
                "{} query heads do not group over {} key heads",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        if self.conv_group_size == 0
            || !self.hidden_size.is_multiple_of(self.conv_group_size)
            || !(1..=self.block_size).contains(&self.conv_kernel_size)
        {
            return invalid(format!(
                "conv group {} over hidden {} with {} taps",
                self.conv_group_size, self.hidden_size, self.conv_kernel_size
            ));
        }
        if !(1..=MAX_TOP_K).contains(&self.selector_top_k)
            || !(1..=MAX_RANK).contains(&self.selector_rank)
        {
            return invalid(format!(
                "selector top-{} at rank {} is past the walk's {MAX_TOP_K} / {MAX_RANK}",
                self.selector_top_k, self.selector_rank
            ));
        }
        if self.target_layer_ids.is_empty() {
            return invalid("target_layer_ids is empty".to_owned());
        }
        Ok(())
    }

    /// Check the drafter against the target it would draft for.
    ///
    /// # Errors
    /// [ConfigError::Invalid] when the widths, the vocabulary or the captured layers do not match.
    pub fn check_target(&self, target: impl Into<DraftTarget>) -> Result<(), ConfigError> {
        let target: DraftTarget = target.into();
        let invalid = |message: String| Err(ConfigError::Invalid(message));
        if self.hidden_size != target.hidden_size {
            return invalid(format!(
                "drafter width {} is not the target's {}",
                self.hidden_size, target.hidden_size
            ));
        }
        if self.vocab_size != target.vocab_size {
            return invalid(format!(
                "drafter vocabulary {} is not the target's {}",
                self.vocab_size, target.vocab_size
            ));
        }
        if self.mask_token_id as usize >= target.vocab_size {
            return invalid(format!(
                "mask token {} is out of vocabulary",
                self.mask_token_id
            ));
        }
        if let Some(layer) = self
            .target_layer_ids
            .iter()
            .find(|layer| **layer >= target.num_hidden_layers)
        {
            return invalid(format!(
                "captured layer {layer} is past the target's {} layers",
                target.num_hidden_layers
            ));
        }
        Ok(())
    }

    /// Floats in one position's captured feature row: one target width per captured layer.
    #[must_use]
    pub fn feature_width(&self) -> usize {
        self.target_layer_ids.len() * self.hidden_size
    }

    /// Floats in one row's query vectors: a head_dim for every query head.
    #[must_use]
    pub fn q_width(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// Floats in one row's key and value vectors: a head_dim for every key head.
    #[must_use]
    pub fn kv_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Floats in one row's convolution taps: two sides, `conv_kernel_size` taps, one per group.
    #[must_use]
    pub fn conv_taps_width(&self) -> usize {
        2 * self.conv_kernel_size * (self.hidden_size / self.conv_group_size)
    }

    /// Positions a layer's context ring holds. A query at the block's last row reaches
    /// `window_left` positions back; a rejected verify may have appended up to `block_size`
    /// positions past where the sequence rolls back to, and those must land on slots no query
    /// can see, so the ring is that much longer than the window.
    #[must_use]
    pub fn ring_positions(&self) -> usize {
        self.window_left + 1 + self.block_size
    }

    /// The commits a forward's captured rows take into the context: `rows` rows at `position`
    /// onward, of a sequence whose next block follows position `end`. That block's anchor row
    /// reaches `window_left` positions back, so a row before `end - window_left` is never read
    /// and is skipped; the rest go in pieces of at most `step` rows, aligned to `step` from the
    /// forward's first row. Each is `(row, rows, position)`, the row counted from the forward's
    /// first. A step's forward and a prompt's chunk commit through this alike.
    pub fn window_commits(
        &self,
        (position, rows): (usize, usize),
        end: usize,
        step: usize,
    ) -> impl Iterator<Item = (usize, usize, usize)> {
        let step = step.max(1);
        let unread = end
            .saturating_sub(self.window_left)
            .saturating_sub(position)
            .min(rows);
        (0..rows).step_by(step).filter_map(move |at| {
            let first = at.max(unread);
            let last = (at + step).min(rows);
            (first < last).then(|| (first, last - first, position + first))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RELEASED: &str = r#"{
        "architectures": ["DFlash2DraftModel"],
        "attention_bias": false,
        "dflash_config": {
            "block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
            "mask_token_id": 248077, "selector_rank": 256, "selector_top_k": 16,
            "target_layer_ids": [5, 15, 25, 35, 45]
        },
        "head_dim": 128, "hidden_act": "silu", "hidden_size": 2560, "intermediate_size": 7680,
        "is_causal": false,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention"],
        "model_type": "qwen3", "num_attention_heads": 32, "num_hidden_layers": 5,
        "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
        "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
        "sliding_window": 2048, "vocab_size": 248320
    }"#;

    #[test]
    fn the_released_drafter_parses() {
        let config = Dflash2Config::from_json(RELEASED).expect("the released config parses");
        assert_eq!(config.block_size, 8);
        assert_eq!(config.window_left, 2047);
        assert_eq!(config.ring_positions(), 2056);
        assert_eq!(config.feature_width(), 12_800);
        assert_eq!(config.conv_taps_width(), 640);
        assert_eq!(config.q_width(), 4096);
        assert_eq!(config.kv_width(), 1024);
        assert_eq!(config.target_layer_ids, vec![5, 15, 25, 35, 45]);
        assert!((config.rope_theta - 10_000.0).abs() < f64::EPSILON);
    }

    /// A forward commits only the rows the next block can read, in pieces aligned to the step
    /// from the forward's first row, whether it is a step's rows or a prompt's chunk.
    #[test]
    fn a_forward_commits_only_the_rows_the_next_block_reads() {
        let config = Dflash2Config::from_json(RELEASED).expect("the released config parses");
        let commits = |at, end, step| -> Vec<_> { config.window_commits(at, end, step).collect() };
        // A short prompt: every row, a step's rows at a time.
        assert_eq!(
            commits((0, 40), 40, 16),
            [(0, 16, 0), (16, 16, 16), (32, 8, 32)]
        );
        // A long prompt's chunk: only its last window_left rows, the first piece cut at the window
        // and the rest still aligned to the chunk's first row.
        let window = config.window_left;
        assert_eq!(
            commits((4096, 4096), 8192, 16).first(),
            Some(&(4096 - window, window % 16, 8192 - window))
        );
        let rows: usize = commits((4096, 4096), 8192, 16).iter().map(|c| c.1).sum();
        assert_eq!(rows, window);
        // A chunk wholly before the window commits nothing.
        assert!(commits((0, 4096), 8192, 16).is_empty());
        // A step past a long prompt commits every row it ran, as one piece.
        assert_eq!(commits((9000, 3), 9003, 16), [(0, 3, 9000)]);
        // A step whose rows straddle the window's start keeps the ones inside it.
        assert_eq!(commits((100, 16), 100 + window + 4, 16), [(4, 12, 104)]);
    }

    /// The pieces are the ones the arms committed before they shared this: Qwen's chunk loop (a
    /// `step`-aligned piece each, its rows before the window cut) and GLM's step (one piece of the
    /// rows at or past the window), less the empty pieces, which committed nothing.
    #[test]
    fn the_window_commits_are_the_arms_former_commits() {
        let config = Dflash2Config::from_json(RELEASED).expect("the released config parses");
        let window = config.window_left;
        for start in [0_usize, 7, 2047, 2048, 4096, 5000] {
            for rows in [1, 15, 16, 17, 2047, 2048, 4096] {
                for extra in [0, 1, 40, 5000] {
                    let end = start + rows + extra;
                    let visible = end.saturating_sub(window);
                    for step in [8, 16] {
                        let former: Vec<_> = (0..rows)
                            .step_by(step)
                            .map(|at| {
                                let piece = step.min(rows - at);
                                let first = visible.saturating_sub(start + at).min(piece);
                                (at + first, piece - first, start + at + first)
                            })
                            .filter(|&(_, count, _)| count > 0)
                            .collect();
                        let shared: Vec<_> =
                            config.window_commits((start, rows), end, step).collect();
                        assert_eq!(shared, former, "start {start} rows {rows} end {end}");
                    }
                    if rows <= 16 {
                        let unseen = visible.saturating_sub(start).min(rows);
                        let former: Vec<_> = (unseen < rows)
                            .then(|| (unseen, rows - unseen, start + unseen))
                            .into_iter()
                            .collect();
                        let shared: Vec<_> =
                            config.window_commits((start, rows), end, 16).collect();
                        assert_eq!(shared, former, "a step at {start} of {rows} to {end}");
                    }
                }
            }
        }
    }

    /// The GLM-5.3-Flash drafter is the same architecture at GLM's geometry: its widths, mask
    /// token, vocabulary and captured layers, set into the Qwen3.8 drafter's document.
    #[test]
    fn the_glm_drafter_geometry_parses_and_fits_its_target() {
        let glm = RELEASED
            .replace("\"hidden_size\": 2560", "\"hidden_size\": 4096")
            .replace(
                "\"intermediate_size\": 7680",
                "\"intermediate_size\": 12288",
            )
            .replace("\"mask_token_id\": 248077", "\"mask_token_id\": 154856")
            .replace("[5, 15, 25, 35, 45]", "[5, 14, 24, 33, 42]")
            .replace("\"rms_norm_eps\": 1e-06", "\"rms_norm_eps\": 1e-05")
            .replace("\"vocab_size\": 248320", "\"vocab_size\": 154880");
        let config = Dflash2Config::from_json(&glm).expect("the GLM drafter parses");
        assert_eq!(config.feature_width(), 5 * 4096);
        assert_eq!(config.conv_taps_width(), 1024);
        let target = DraftTarget {
            hidden_size: 4096,
            vocab_size: 154_880,
            num_hidden_layers: 45,
        };
        config
            .check_target(target)
            .expect("the drafter fits GLM-5.3-Flash");
    }

    #[test]
    fn a_drafter_for_another_target_is_refused() {
        let config = Dflash2Config::from_json(RELEASED).expect("the released config parses");
        let target = DraftTarget {
            hidden_size: 2560,
            vocab_size: 248_320,
            num_hidden_layers: 48,
        };
        config.check_target(target).expect("the drafter fits");
        let narrow = DraftTarget {
            hidden_size: 2048,
            ..target
        };
        assert!(config.check_target(narrow).is_err());
        let shallow = DraftTarget {
            num_hidden_layers: 45,
            ..target
        };
        assert!(config.check_target(shallow).is_err());
    }

    #[test]
    fn a_causal_or_unimplemented_drafter_is_refused() {
        let causal = RELEASED.replace("\"is_causal\": false", "\"is_causal\": true");
        assert!(Dflash2Config::from_json(&causal).is_err());
        let capped = RELEASED.replace(
            "\"block_size\": 8,",
            "\"block_size\": 8, \"final_logit_softcapping\": 30.0,",
        );
        assert!(Dflash2Config::from_json(&capped).is_err());
        let wide = RELEASED.replace("\"block_size\": 8,", "\"block_size\": 16,");
        assert!(Dflash2Config::from_json(&wide).is_err());
    }
}

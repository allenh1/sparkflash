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

//! The f32 host reference forward for Dory.
//!
//! The arithmetic follows NVIDIA's reference implementation as carried by vLLM's Dory model:
//! every projection is bias-free BF16 (read as f32 here); q and k are rotated first, then
//! L2-normalized per head; q receives the learned sqk scale squared and k stays plain; the
//! attention output is multiplied by sigmoid(gate * sg), L2-normalized over all heads, and
//! projected; the MLP scales gate and up by the learned suv vectors, applies SiLU,
//! L2-normalizes the intermediate vector, and projects down; and every sublayer's residual is
//! justnorm(a) + alpha * (justnorm(b) - justnorm(a)), itself L2-normalized.
//!
//! The loop is part of the arithmetic, not a replay of the same cache: the input block's output
//! is reinjected at the start of every recurrent pass, and each pass writes its own KV cache
//! (or overwrites one shared cache in last-loop mode).

use crate::config::{Block, DoryConfig, NGPT_EPSILON};
use sparkflash_weights::WeightSource;
use sparkflash_weights::linalg::decode_by_name;

/// The largest tensor this reference materialises.
pub const ELEMENT_LIMIT: usize = 1 << 30;

/// A row-major f32 matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    /// Rows.
    pub rows: usize,
    /// Columns.
    pub columns: usize,
    /// Values, row-major.
    pub data: Vec<f32>,
}

impl Mat {
    /// A matrix from row-major data.
    ///
    /// # Panics
    /// When the data length is not rows * columns.
    #[must_use]
    pub fn new(rows: usize, columns: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * columns, "matrix shape and data disagree");
        Self {
            rows,
            columns,
            data,
        }
    }

    /// A matrix whose element (row, column) is f(row, column).
    #[must_use]
    pub fn from_fn(rows: usize, columns: usize, mut f: impl FnMut(usize, usize) -> f32) -> Self {
        let mut data = Vec::with_capacity(rows * columns);
        for row in 0..rows {
            for column in 0..columns {
                data.push(f(row, column));
            }
        }
        Self {
            rows,
            columns,
            data,
        }
    }

    /// y = W x.
    #[must_use]
    pub fn matvec(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.columns, "matvec input width");
        let mut out = vec![0.0; self.rows];
        for (row, slot) in out.iter_mut().enumerate() {
            let base = row * self.columns;
            let mut accumulator = 0.0f32;
            for (value, activation) in self.data[base..base + self.columns].iter().zip(input) {
                accumulator += value * activation;
            }
            *slot = accumulator;
        }
        out
    }
}

/// The weights of one attention sublayer.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionWeights {
    /// The residual lerp weight [hidden].
    pub alpha: Vec<f32>,
    /// Query projection [heads * head_dim, hidden].
    pub q_proj: Mat,
    /// Key projection [kv_heads * head_dim, hidden].
    pub k_proj: Mat,
    /// Value projection [kv_heads * head_dim, hidden].
    pub v_proj: Mat,
    /// Output-gate projection [heads * head_dim, hidden].
    pub g_proj: Mat,
    /// Output projection [hidden, heads * head_dim].
    pub o_proj: Mat,
    /// The learned q scale [heads * head_dim].
    pub sqk: Vec<f32>,
    /// The learned gate scale [heads * head_dim].
    pub s_gate: Vec<f32>,
}

/// The weights of one MLP sublayer.
#[derive(Debug, Clone, PartialEq)]
pub struct MlpWeights {
    /// The residual lerp weight [hidden].
    pub alpha: Vec<f32>,
    /// Gate projection [intermediate, hidden].
    pub gate_proj: Mat,
    /// Up projection [intermediate, hidden].
    pub up_proj: Mat,
    /// Down projection [hidden, intermediate].
    pub down_proj: Mat,
    /// The learned gate scale [intermediate].
    pub suv_gate: Vec<f32>,
    /// The learned up scale [intermediate].
    pub suv_up: Vec<f32>,
}

/// One physical block's weights: an attention sublayer and the MLP sublayer after it.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockWeights {
    /// The attention sublayer.
    pub attention: AttentionWeights,
    /// The MLP sublayer that follows it.
    pub mlp: MlpWeights,
}

/// The complete host-reference weights.
#[derive(Debug, Clone, PartialEq)]
pub struct DoryWeights {
    /// The configuration these weights were built for.
    pub config: DoryConfig,
    /// The token embedding [vocab, hidden].
    pub embed: Mat,
    /// The LM head [vocab, hidden].
    pub lm_head: Mat,
    /// The per-vocabulary logit scale [vocab].
    pub logit_scale: Vec<f32>,
    /// One body per physical block, in schedule order.
    pub blocks: Vec<BlockWeights>,
}

impl DoryWeights {
    /// Load the whole checkpoint through one weight source.
    ///
    /// # Errors
    /// A missing or malformed tensor, or a shape that disagrees with the configuration.
    pub fn load(config: &DoryConfig, source: &dyn WeightSource) -> Result<Self, String> {
        let hidden = config.hidden_size;
        let query_width = config.num_attention_heads * config.head_dim;
        let kv_width = config.num_key_value_heads * config.head_dim;
        let intermediate = config.intermediate_size;
        let embed = matrix(
            source,
            "backbone.embeddings.weight",
            config.vocab_size,
            hidden,
        )?;
        let lm_head = matrix(source, "lm_head.weight", config.vocab_size, hidden)?;
        let logit_scale = vector(source, "lm_head.logit_scale", config.vocab_size)?;
        let mut blocks = Vec::with_capacity(config.num_blocks());
        for block in &config.blocks {
            let attention_prefix = format!("backbone.layers.{}.mixer", block.attention_layer);
            let mlp_prefix = format!("backbone.layers.{}.mixer", block.mlp_layer);
            let attention = AttentionWeights {
                alpha: vector(
                    source,
                    &format!("backbone.layers.{}.attn_alpha", block.attention_layer),
                    hidden,
                )?,
                q_proj: matrix(
                    source,
                    &format!("{attention_prefix}.q_proj.weight"),
                    query_width,
                    hidden,
                )?,
                k_proj: matrix(
                    source,
                    &format!("{attention_prefix}.k_proj.weight"),
                    kv_width,
                    hidden,
                )?,
                v_proj: matrix(
                    source,
                    &format!("{attention_prefix}.v_proj.weight"),
                    kv_width,
                    hidden,
                )?,
                g_proj: matrix(
                    source,
                    &format!("{attention_prefix}.g_proj.weight"),
                    query_width,
                    hidden,
                )?,
                o_proj: matrix(
                    source,
                    &format!("{attention_prefix}.o_proj.weight"),
                    hidden,
                    query_width,
                )?,
                sqk: vector(source, &format!("{attention_prefix}.sqk"), query_width)?,
                s_gate: vector(source, &format!("{attention_prefix}.s_gate"), query_width)?,
            };
            let mlp = MlpWeights {
                alpha: vector(
                    source,
                    &format!("backbone.layers.{}.mlp_alpha", block.mlp_layer),
                    hidden,
                )?,
                gate_proj: matrix(
                    source,
                    &format!("{mlp_prefix}.gate_proj.weight"),
                    intermediate,
                    hidden,
                )?,
                up_proj: matrix(
                    source,
                    &format!("{mlp_prefix}.up_proj.weight"),
                    intermediate,
                    hidden,
                )?,
                down_proj: matrix(
                    source,
                    &format!("{mlp_prefix}.down_proj.weight"),
                    hidden,
                    intermediate,
                )?,
                suv_gate: vector(source, &format!("{mlp_prefix}.suv_gate"), intermediate)?,
                suv_up: vector(source, &format!("{mlp_prefix}.suv_up"), intermediate)?,
            };
            blocks.push(BlockWeights { attention, mlp });
        }
        Ok(Self {
            config: config.clone(),
            embed,
            lm_head,
            logit_scale,
            blocks,
        })
    }

    /// The logits one token produces, with the reference state updated.
    ///
    /// # Errors
    /// A token outside the vocabulary.
    pub fn forward_token(
        &self,
        token: u32,
        position: usize,
        state: &mut DoryState,
    ) -> Result<Vec<f32>, String> {
        let config = &self.config;
        if token as usize >= config.vocab_size {
            return Err(format!(
                "token {token} is outside the checkpoint vocabulary {}",
                config.vocab_size
            ));
        }
        if position != state.position {
            return Err(format!(
                "the state is at position {}, the caller asked for {position}",
                state.position
            ));
        }
        let mut x = self.embed_row(token)?;

        for (block_index, block) in config.input_blocks().iter().enumerate() {
            x = self.block_forward(block_index, block, &x, 0, position, state)?;
        }
        let input_h = x.clone();
        let mut recurrent_state = input_h.clone();
        for loop_index in 0..config.n_recurrent_loops {
            let mut h = justnorm(&add(&recurrent_state, &input_h));
            let start = config.input_blocks().len();
            for (offset, block) in config.recurrent_blocks().iter().enumerate() {
                let block_index = start + offset;
                h = self.block_forward(block_index, block, &h, loop_index, position, state)?;
            }
            recurrent_state = justnorm(&h);
        }
        let mut h = recurrent_state;
        let start = config.input_blocks().len() + config.recurrent_blocks().len();
        for (offset, block) in config.output_blocks().iter().enumerate() {
            let block_index = start + offset;
            h = self.block_forward(block_index, block, &h, 0, position, state)?;
        }
        let mut logits = self.lm_head.matvec(&h);
        let divisor = config.logit_scale_divisor() as f32;
        for (logit, scale) in logits.iter_mut().zip(&self.logit_scale) {
            *logit *= *scale / divisor;
        }
        state.position += 1;
        Ok(logits)
    }

    /// The logits of the last token of a prompt, from a fresh state.
    ///
    /// # Errors
    /// As [Self::forward_token].
    pub fn forward_prompt(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let mut state = DoryState::new(&self.config);
        let mut logits = Vec::new();
        for (position, token) in tokens.iter().enumerate() {
            logits = self.forward_token(*token, position, &mut state)?;
        }
        Ok(logits)
    }

    fn embed_row(&self, token: u32) -> Result<Vec<f32>, String> {
        let hidden = self.config.hidden_size;
        let start = token as usize * hidden;
        self.embed
            .data
            .get(start..start + hidden)
            .map(<[f32]>::to_vec)
            .ok_or_else(|| format!("the embedding has no row {token}"))
    }

    fn block_forward(
        &self,
        block_index: usize,
        block: &Block,
        x: &[f32],
        loop_index: usize,
        position: usize,
        state: &mut DoryState,
    ) -> Result<Vec<f32>, String> {
        let config = &self.config;
        let weights = &self.blocks[block_index];
        let slot = if block.cache_slots == 1 {
            0
        } else {
            loop_index.min(block.cache_slots - 1)
        };
        let cache = state
            .caches
            .get_mut(block_index)
            .and_then(|slots| slots.get_mut(slot))
            .ok_or_else(|| format!("block {block_index} has no attention cache"))?;
        let attention = attention_forward(config, block, &weights.attention, x, cache, position)?;
        let after_attention = slerp(
            x,
            &attention,
            &weights.attention.alpha,
            config.alpha_scale() as f32,
        );
        let mlp = mlp_forward(config, &weights.mlp, &after_attention);
        Ok(slerp(
            &after_attention,
            &mlp,
            &weights.mlp.alpha,
            config.alpha_scale() as f32,
        ))
    }
}

/// One attention layer's cached keys and values.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KvCache {
    /// Keys, position * kv_width.
    pub k: Vec<f32>,
    /// Values, position * kv_width.
    pub v: Vec<f32>,
}

impl KvCache {
    fn write(&mut self, position: usize, k: &[f32], v: &[f32]) {
        let width = k.len();
        let start = position * width;
        if self.k.len() < start + width {
            self.k.resize(start + width, 0.0);
            self.v.resize(start + width, 0.0);
        }
        self.k[start..start + width].copy_from_slice(k);
        self.v[start..start + width].copy_from_slice(v);
    }
}

/// The reference state one sequence carries between tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct DoryState {
    /// The next absolute position.
    pub position: usize,
    /// One cache per physical block, with this block's cache slots.
    pub caches: Vec<Vec<KvCache>>,
}

impl DoryState {
    /// A zeroed state for a configuration.
    #[must_use]
    pub fn new(config: &DoryConfig) -> Self {
        Self {
            position: 0,
            caches: config
                .blocks
                .iter()
                .map(|block| vec![KvCache::default(); block.cache_slots])
                .collect(),
        }
    }

    /// The cache slots one block owns.
    #[must_use]
    pub fn slots(&self, block: usize) -> usize {
        self.caches.get(block).map_or(0, Vec::len)
    }
}

fn attention_forward(
    config: &DoryConfig,
    block: &Block,
    weights: &AttentionWeights,
    x: &[f32],
    cache: &mut KvCache,
    position: usize,
) -> Result<Vec<f32>, String> {
    let heads = config.num_attention_heads;
    let kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let groups = heads / kv_heads;
    let mut q = weights.q_proj.matvec(x);
    let mut k = weights.k_proj.matvec(x);
    let v = weights.v_proj.matvec(x);
    let gate = weights.g_proj.matvec(x);
    let qk_factor = config.qk_scale_factor() as f32;
    let gate_factor = config.gate_scale_factor() as f32;
    for head in 0..heads {
        let start = head * head_dim;
        rope(&mut q[start..start + head_dim], position, block.rope_theta);
        let norm = justnorm(&q[start..start + head_dim]);
        for index in 0..head_dim {
            let scale = weights.sqk[start + index] * qk_factor;
            q[start + index] = norm[index] * scale * scale;
        }
    }
    for head in 0..kv_heads {
        let start = head * head_dim;
        rope(&mut k[start..start + head_dim], position, block.rope_theta);
        let norm = justnorm(&k[start..start + head_dim]);
        k[start..start + head_dim].copy_from_slice(&norm);
    }
    cache.write(position, &k, &v);
    let mut attended = vec![0.0f32; heads * head_dim];
    let window = block.window;
    for head in 0..heads {
        let kv_head = head / groups;
        let q_head = &q[head * head_dim..(head + 1) * head_dim];
        let mut scores = Vec::with_capacity(position + 1);
        for key_position in 0..=position {
            if window.is_some_and(|window| key_position + window <= position) {
                scores.push(f32::NEG_INFINITY);
                continue;
            }
            let start = key_position * kv_heads * head_dim + kv_head * head_dim;
            let key = &cache.k[start..start + head_dim];
            let score = q_head
                .iter()
                .zip(key)
                .map(|(query, key)| query * key)
                .sum::<f32>()
                / (head_dim as f32).sqrt();
            scores.push(score);
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for score in &mut scores {
            *score = (*score - max).exp();
            sum += *score;
        }
        let out = &mut attended[head * head_dim..(head + 1) * head_dim];
        for (key_position, weight) in scores.iter().enumerate() {
            let probability = weight / sum;
            let start = key_position * kv_heads * head_dim + kv_head * head_dim;
            for (index, value) in out.iter_mut().enumerate() {
                *value += probability * cache.v[start + index];
            }
        }
    }
    for index in 0..attended.len() {
        let scale = weights.s_gate[index] * gate_factor;
        attended[index] *= sigmoid(gate[index] * scale * scale);
    }
    let attended = justnorm(&attended);
    Ok(weights.o_proj.matvec(&attended))
}

fn mlp_forward(config: &DoryConfig, weights: &MlpWeights, x: &[f32]) -> Vec<f32> {
    let gate = weights.gate_proj.matvec(x);
    let up = weights.up_proj.matvec(x);
    let factor = config.mlp_scale_factor() as f32;
    let mut hidden = Vec::with_capacity(gate.len());
    for index in 0..gate.len() {
        let gate = gate[index] * weights.suv_gate[index] * factor;
        let up = up[index] * weights.suv_up[index] * factor;
        hidden.push(silu(gate) * up);
    }
    let hidden = justnorm(&hidden);
    weights.down_proj.matvec(&hidden)
}

/// L2-normalizes a vector with the reference epsilon.
#[must_use]
pub fn justnorm(values: &[f32]) -> Vec<f32> {
    let sum = values.iter().map(|value| value * value).sum::<f32>();
    let scale = 1.0 / (sum + NGPT_EPSILON).sqrt();
    values.iter().map(|value| value * scale).collect()
}

/// The nGPT residual lerp: justnorm(justnorm(a) + lr * (justnorm(b) - justnorm(a))).
#[must_use]
pub fn slerp(a: &[f32], b: &[f32], alpha: &[f32], alpha_scale: f32) -> Vec<f32> {
    let ya = justnorm(a);
    let yb = justnorm(b);
    let mut mixed = Vec::with_capacity(a.len());
    for index in 0..a.len() {
        let lr = (alpha[index] * alpha_scale).abs();
        mixed.push(ya[index] + lr * (yb[index] - ya[index]));
    }
    justnorm(&mixed)
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(a, b)| a + b).collect()
}

fn rope(values: &mut [f32], position: usize, theta: f64) {
    let half = values.len() / 2;
    for index in 0..half {
        let frequency = 1.0 / theta.powf(2.0 * index as f64 / values.len() as f64);
        let angle = position as f64 * frequency;
        let (sin, cos) = angle.sin_cos();
        let (sin, cos) = (sin as f32, cos as f32);
        let x0 = values[index];
        let x1 = values[index + half];
        values[index] = x0 * cos - x1 * sin;
        values[index + half] = x1 * cos + x0 * sin;
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn matrix(
    source: &dyn WeightSource,
    name: &str,
    rows: usize,
    columns: usize,
) -> Result<Mat, String> {
    let spec = source.spec(name).map_err(|error| error.to_string())?;
    if spec.shape != [rows, columns] {
        return Err(format!(
            "{name} is {:?}, expected [{rows}, {columns}]",
            spec.shape
        ));
    }
    let values = decode_by_name(source, name, ELEMENT_LIMIT).map_err(|error| error.to_string())?;
    Ok(Mat::new(rows, columns, values))
}

fn vector(source: &dyn WeightSource, name: &str, width: usize) -> Result<Vec<f32>, String> {
    let spec = source.spec(name).map_err(|error| error.to_string())?;
    if spec.shape != [width] {
        return Err(format!("{name} is {:?}, expected [{width}]", spec.shape));
    }
    decode_by_name(source, name, ELEMENT_LIMIT).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn justnorm_is_unit_length_except_for_the_epsilon() {
        let values = justnorm(&[3.0, 4.0]);
        assert!((values[0] - 0.6).abs() < 1e-6);
        assert!((values[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn slerp_interpolates_toward_the_branch_and_renormalizes() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let out = slerp(&a, &b, &[0.5, 0.5], 1.0);
        assert!((out[0] - out[1]).abs() < 1e-6);
        assert!((out[0] * out[0] + out[1] * out[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rope_rotates_the_first_pair_by_the_position_angle() {
        let mut values = vec![1.0, 0.0, 0.0, 0.0];
        rope(&mut values, 1, 10_000.0);
        assert!((values[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((values[2] - 1.0f32.sin()).abs() < 1e-6);
        let mut values = vec![1.0, 0.0, 0.0, 0.0];
        rope(&mut values, 1, 1.0);
        assert!((values[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((values[2] - 1.0f32.sin()).abs() < 1e-6);
    }
}

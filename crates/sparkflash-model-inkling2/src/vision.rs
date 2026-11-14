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

//! The Inkling-2-Small vision tower, as an f32 host reference.
//!
//! A patch-20 ViT of 23 blocks, a learned 32x32 position grid interpolated to the image's patch
//! grid, axial two-dimensional RoPE inside each head, the 2x2 merger and the two-layer adapter
//! that hands the language model its image tokens. The tensors are the released names
//! [crate::tensor] plans under `model.visual`, so the tower reads the checkpoint the converter
//! wrote rather than a second spelling of it.
//!
//! The release ships weights and a config but no modelling code, so the arithmetic here follows
//! the stored shapes and the published ViT conventions; the choices a config cannot express are
//! stated where they are made. This is the oracle a device tower will be held to, the same
//! relationship [crate::reference] has with the text kernels.

use std::sync::Arc;

use sparkflash_weights::WeightSource;
use sparkflash_weights::linalg::decode_by_name;

use crate::config::InklingVisionConfig;

/// The largest merged image-token count the host tower runs, as a guard on the canvas.
pub const MAX_MERGED_TOKENS: usize = 4_096;
/// The rope base of the axial tables.
pub const AXIAL_ROPE_THETA: f64 = 10_000.0;

/// The patch grid an image of `height` by `width` pixels makes.
///
/// The tower requires both sides to divide into `patch_size * merge_size` cells, and the
/// released guidance keeps an image between 40 and 4,096 pixels a side.
///
/// # Errors
/// A side outside the released range, or one that does not divide.
pub fn plan_grid(
    height: usize,
    width: usize,
    config: &InklingVisionConfig,
) -> Result<(usize, usize), String> {
    let cell = config.patch_size * config.adapter_merge_size;
    for (what, side) in [("height", height), ("width", width)] {
        if !(40..=4096).contains(&side) {
            return Err(format!(
                "the image's {what} is {side} pixels; the tower takes 40 to 4096"
            ));
        }
        if !side.is_multiple_of(cell) {
            return Err(format!(
                "the image's {what} is {side} pixels, not a multiple of the {cell}-pixel patch-and-merge cell"
            ));
        }
    }
    Ok((height / config.patch_size, width / config.patch_size))
}

/// One patch's flattened samples: three channels of `patch_size` squared.
#[must_use]
pub fn patch_values(config: &InklingVisionConfig) -> usize {
    3 * config.patch_size * config.patch_size
}

/// A row-major matrix read from the checkpoint.
struct Matrix {
    rows: usize,
    columns: usize,
    values: Vec<f32>,
}

impl Matrix {
    fn load(
        source: &dyn WeightSource,
        name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<Self, String> {
        let values =
            decode_by_name(source, name, 1 << 30).map_err(|error| format!("{name}: {error}"))?;
        if values.len() != rows * columns {
            return Err(format!(
                "{name} decoded {} values, expected {rows}x{columns}",
                values.len()
            ));
        }
        Ok(Self {
            rows,
            columns,
            values,
        })
    }

    fn row(&self, row: usize) -> &[f32] {
        &self.values[row * self.columns..(row + 1) * self.columns]
    }

    fn apply(&self, input: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
        let mut out = vec![0f32; self.rows];
        for (row, slot) in out.iter_mut().enumerate() {
            let mut value = 0.0f32;
            for (weight, input) in self.row(row).iter().zip(input) {
                value += weight * input;
            }
            *slot = value + bias.map_or(0.0, |bias| bias[row]);
        }
        out
    }
}

/// A per-channel affine pair.
struct Affine {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl Affine {
    fn load(
        source: &dyn WeightSource,
        prefix: &str,
        width: usize,
        with_bias: bool,
    ) -> Result<Self, String> {
        let weight = decode_by_name(source, &format!("{prefix}.weight"), 1 << 20)
            .map_err(|error| format!("{prefix}.weight: {error}"))?;
        if weight.len() != width {
            return Err(format!(
                "{prefix}.weight holds {} values, expected {width}",
                weight.len()
            ));
        }
        let bias = if with_bias {
            Some(
                decode_by_name(source, &format!("{prefix}.bias"), 1 << 20)
                    .map_err(|error| format!("{prefix}.bias: {error}"))?,
            )
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    fn norm(&self, input: &[f32], epsilon: f64) -> Vec<f32> {
        layer_norm(input, &self.weight, self.bias.as_deref(), epsilon)
    }
}

/// One transformer block.
struct Block {
    ln_1: Affine,
    ln_2: Affine,
    ls_1: Vec<f32>,
    ls_2: Vec<f32>,
    q: Matrix,
    q_bias: Vec<f32>,
    k: Matrix,
    k_bias: Vec<f32>,
    v: Matrix,
    v_bias: Vec<f32>,
    out: Matrix,
    out_bias: Vec<f32>,
    fc: Matrix,
    fc_bias: Vec<f32>,
    proj: Matrix,
    proj_bias: Vec<f32>,
}

/// The tower's weights.
pub struct InklingVision {
    config: InklingVisionConfig,
    patch: Matrix,
    position: Vec<f32>,
    class_embedding: Vec<f32>,
    cls_position: Vec<f32>,
    ln_pre: Affine,
    blocks: Vec<Block>,
    proj_in: Matrix,
    proj_out: Matrix,
}

impl std::fmt::Debug for InklingVision {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InklingVision")
            .field("depth", &self.config.depth)
            .field("hidden", &self.config.hidden_size)
            .finish()
    }
}

/// The width a block's gated MLP expands to.
#[must_use]
pub fn mlp_width(width: usize) -> usize {
    ((width as f64) * 4.0).round() as usize
}

impl InklingVision {
    /// Reads the tower's weights from a checkpoint source.
    ///
    /// # Errors
    /// A missing or mis-shaped tensor.
    pub fn load(config: &InklingVisionConfig, source: Arc<dyn WeightSource>) -> Result<Self, String> {
        let prefix = "model.visual";
        let width = config.hidden_size;
        let intermediate = mlp_width(width);
        let patch = Matrix::load(
            source.as_ref(),
            &format!("{prefix}.backbone.embeddings.patch_embedding.weight"),
            width,
            patch_values(config),
        )?;
        let position = decode_by_name(
            source.as_ref(),
            &format!("{prefix}.backbone.embeddings.position_embedding"),
            1 << 30,
        )
        .map_err(|error| format!("position_embedding: {error}"))?;
        if position.len() != config.posemb_grid_size * config.posemb_grid_size * width {
            return Err(format!(
                "position_embedding holds {} values, expected a {}x{} grid of {width}",
                position.len(),
                config.posemb_grid_size,
                config.posemb_grid_size
            ));
        }
        let class_embedding = decode_by_name(
            source.as_ref(),
            &format!("{prefix}.backbone.embeddings.class_embedding"),
            1 << 20,
        )
        .map_err(|error| format!("class_embedding: {error}"))?;
        let cls_position = decode_by_name(
            source.as_ref(),
            &format!("{prefix}.backbone.embeddings.cls_position_embedding"),
            1 << 20,
        )
        .map_err(|error| format!("cls_position_embedding: {error}"))?;
        let ln_pre = Affine::load(
            source.as_ref(),
            &format!("{prefix}.backbone.ln_pre"),
            width,
            true,
        )?;
        let mut blocks = Vec::with_capacity(config.depth);
        for layer in 0..config.depth {
            let p = format!("{prefix}.backbone.layers.{layer}");
            let attention = format!("{p}.self_attn");
            let projection = |name: &str, rows: usize, columns: usize| {
                Matrix::load(source.as_ref(), &format!("{attention}.{name}.weight"), rows, columns)
            };
            let bias = |name: &str| {
                decode_by_name(source.as_ref(), &format!("{attention}.{name}.bias"), 1 << 20)
                    .map_err(|error| format!("{name}.bias: {error}"))
            };
            blocks.push(Block {
                ln_1: Affine::load(source.as_ref(), &format!("{p}.ln_1"), width, true)?,
                ln_2: Affine::load(source.as_ref(), &format!("{p}.ln_2"), width, true)?,
                ls_1: decode_by_name(source.as_ref(), &format!("{p}.ls_1.gamma"), 1 << 20)
                    .map_err(|error| format!("ls_1: {error}"))?,
                ls_2: decode_by_name(source.as_ref(), &format!("{p}.ls_2.gamma"), 1 << 20)
                    .map_err(|error| format!("ls_2: {error}"))?,
                q: projection("q_proj", width, width)?,
                q_bias: bias("q_proj")?,
                k: projection("k_proj", width, width)?,
                k_bias: bias("k_proj")?,
                v: projection("v_proj", width, width)?,
                v_bias: bias("v_proj")?,
                out: projection("out_proj", width, width)?,
                out_bias: bias("out_proj")?,
                fc: Matrix::load(
                    source.as_ref(),
                    &format!("{p}.mlp.c_fc.weight"),
                    intermediate,
                    width,
                )?,
                fc_bias: decode_by_name(source.as_ref(), &format!("{p}.mlp.c_fc.bias"), 1 << 20)
                    .map_err(|error| format!("c_fc.bias: {error}"))?,
                proj: Matrix::load(
                    source.as_ref(),
                    &format!("{p}.mlp.c_proj.weight"),
                    width,
                    intermediate,
                )?,
                proj_bias: decode_by_name(source.as_ref(), &format!("{p}.mlp.c_proj.bias"), 1 << 20)
                    .map_err(|error| format!("c_proj.bias: {error}"))?,
            });
        }
        let merge = config.adapter_merge_size;
        let proj_in = Matrix::load(
            source.as_ref(),
            &format!("{prefix}.proj_in.weight"),
            config.projection_hidden_dim,
            width * merge * merge,
        )?;
        let proj_out = Matrix::load(
            source.as_ref(),
            &format!("{prefix}.proj_out.weight"),
            config.out_hidden_size,
            config.projection_hidden_dim,
        )?;
        Ok(Self {
            config: config.clone(),
            patch,
            position,
            class_embedding,
            cls_position,
            ln_pre,
            blocks,
            proj_in,
            proj_out,
        })
    }

    /// The tower's geometry.
    #[must_use]
    pub fn config(&self) -> &InklingVisionConfig {
        &self.config
    }

    /// Encodes a canvas into image tokens, `[merged, out_hidden]`.
    ///
    /// `pixels` is `[grid_h * grid_w, 3 * patch_size^2]`: one patch a row, channels first,
    /// row-major inside the patch, which is the order the released
    /// `patch_embedding.weight` expects.
    ///
    /// # Errors
    /// A grid that does not divide into merge blocks, a canvas of the wrong length, or a merged
    /// token count over [MAX_MERGED_TOKENS].
    pub fn encode(&self, pixels: &[f32], grid_h: usize, grid_w: usize) -> Result<Vec<f32>, String> {
        let config = &self.config;
        let width = config.hidden_size;
        let merge = config.adapter_merge_size;
        let epsilon = config.layer_norm_eps;
        if !grid_h.is_multiple_of(merge) || !grid_w.is_multiple_of(merge) {
            return Err(format!(
                "a {grid_h}x{grid_w} patch grid does not divide into {merge}x{merge} merge blocks"
            ));
        }
        let patches = grid_h * grid_w;
        let merged = patches / (merge * merge);
        if merged > MAX_MERGED_TOKENS {
            return Err(format!(
                "a {grid_h}x{grid_w} patch grid makes {merged} image tokens, over the tower's {MAX_MERGED_TOKENS}-token cap"
            ));
        }
        let patch_width = patch_values(config);
        if pixels.len() != patches * patch_width {
            return Err(format!(
                "the canvas holds {} samples, expected {patches} patches of {patch_width}",
                pixels.len()
            ));
        }
        let tokens = patches + 1;

        // The class token leads the patch tokens, so one position table covers the sequence.
        let mut hidden = vec![0f32; tokens * width];
        hidden[..width].copy_from_slice(&self.class_embedding);
        for (value, position) in hidden.iter_mut().zip(&self.cls_position) {
            *value += *position;
        }
        let side = config.posemb_grid_size;
        for patch in 0..patches {
            let encoded = self
                .patch
                .apply(&pixels[patch * patch_width..(patch + 1) * patch_width], None);
            let row = (patch + 1) * width;
            hidden[row..row + width].copy_from_slice(&encoded);
            let (rows, row_weights) = axis_taps(patch / grid_w, grid_h, side);
            let (columns, column_weights) = axis_taps(patch % grid_w, grid_w, side);
            for (grid_row, row_weight) in rows.iter().zip(&row_weights) {
                for (grid_column, column_weight) in columns.iter().zip(&column_weights) {
                    let weight = row_weight * column_weight;
                    let base = (grid_row * side + grid_column) * width;
                    for column in 0..width {
                        hidden[row + column] += weight * self.position[base + column];
                    }
                }
            }
        }

        for token in hidden.chunks_mut(width) {
            let normed = self.ln_pre.norm(token, epsilon);
            token.copy_from_slice(&normed);
        }

        let heads = config.heads;
        let head_dim = width / heads;
        // The class token leads the sequence and has no spatial position, so the tables carry
        // one identity row for it.
        let rope = AxialRope::new(grid_h, grid_w, head_dim, 1);
        for block in &self.blocks {
            // x = x + ls_1 * attention(ln_1(x))
            let mut q = vec![0f32; tokens * width];
            let mut k = vec![0f32; tokens * width];
            let mut v = vec![0f32; tokens * width];
            for token in 0..tokens {
                let normed = block
                    .ln_1
                    .norm(&hidden[token * width..(token + 1) * width], epsilon);
                let query = block.q.apply(&normed, Some(&block.q_bias));
                let key = block.k.apply(&normed, Some(&block.k_bias));
                let value = block.v.apply(&normed, Some(&block.v_bias));
                q[token * width..(token + 1) * width].copy_from_slice(&query);
                k[token * width..(token + 1) * width].copy_from_slice(&key);
                v[token * width..(token + 1) * width].copy_from_slice(&value);
            }
            rope.apply(&mut q, tokens, heads, width);
            rope.apply(&mut k, tokens, heads, width);
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut attention = vec![0f32; tokens * width];
            for head in 0..heads {
                for query in 0..tokens {
                    let mut scores = Vec::with_capacity(tokens);
                    for key in 0..tokens {
                        let mut dot = 0.0f32;
                        for dim in 0..head_dim {
                            dot += q[query * width + head * head_dim + dim]
                                * k[key * width + head * head_dim + dim];
                        }
                        scores.push(dot * scale);
                    }
                    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut denominator = 0.0f32;
                    for score in &mut scores {
                        *score = (*score - maximum).exp();
                        denominator += *score;
                    }
                    for (key, score) in scores.iter().enumerate() {
                        let weight = *score / denominator;
                        for dim in 0..head_dim {
                            attention[query * width + head * head_dim + dim] +=
                                weight * v[key * width + head * head_dim + dim];
                        }
                    }
                }
            }
            for token in 0..tokens {
                let projected = block
                    .out
                    .apply(&attention[token * width..(token + 1) * width], Some(&block.out_bias));
                for column in 0..width {
                    hidden[token * width + column] += block.ls_1[column] * projected[column];
                }
            }
            // x = x + ls_2 * mlp(ln_2(x))
            for token in 0..tokens {
                let normed = block
                    .ln_2
                    .norm(&hidden[token * width..(token + 1) * width], epsilon);
                let mut up = block.fc.apply(&normed, Some(&block.fc_bias));
                for value in &mut up {
                    *value = gelu(*value);
                }
                let down = block.proj.apply(&up, Some(&block.proj_bias));
                for column in 0..width {
                    hidden[token * width + column] += block.ls_2[column] * down[column];
                }
            }
        }

        // The 2x2 merger drops the class token and concatenates each block's four patches in
        // row-major order: (0,0), (0,1), (1,0), (1,1).
        let blocks_w = grid_w / merge;
        let part = width;
        let mut out = vec![0f32; merged * config.out_hidden_size];
        let mut row = vec![0f32; width * merge * merge];
        for token in 0..merged {
            let block_row = token / blocks_w;
            let block_column = token % blocks_w;
            for inner_row in 0..merge {
                for inner_column in 0..merge {
                    let patch = (block_row * merge + inner_row) * grid_w + block_column * merge + inner_column;
                    let slot = (inner_row * merge + inner_column) * part;
                    row[slot..slot + part]
                        .copy_from_slice(&hidden[(patch + 1) * width..(patch + 2) * width]);
                }
            }
            let mut projected = self.proj_in.apply(&row, None);
            for value in &mut projected {
                *value = gelu(*value);
            }
            let projected = self.proj_out.apply(&projected, None);
            out[token * config.out_hidden_size..(token + 1) * config.out_hidden_size]
                .copy_from_slice(&projected);
        }
        Ok(out)
    }
}

/// The axial two-dimensional rope for one head width.
struct AxialRope {
    tokens: usize,
    head_dim: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl AxialRope {
    /// Tables for a `grid_h` by `grid_w` patch grid, with `leading` tokens before the patches
    /// (the class token) that the rotation leaves alone.
    ///
    /// A head's first half turns with the patch's row and its second half with its column, so a
    /// head can address a position in the image rather than an index in a flattened sequence.
    fn new(grid_h: usize, grid_w: usize, head_dim: usize, leading: usize) -> Self {
        let half = head_dim / 2;
        let quarter = half / 2;
        let tokens = grid_h * grid_w + leading;
        let mut cos = vec![1f32; tokens * head_dim];
        let mut sin = vec![0f32; tokens * head_dim];
        for row in 0..grid_h {
            for column in 0..grid_w {
                let token = leading + row * grid_w + column;
                for index in 0..quarter {
                    let frequency = AXIAL_ROPE_THETA.powf(-(2.0 * index as f64) / half as f64);
                    let (row_angle, column_angle) = (
                        row as f64 * frequency,
                        column as f64 * frequency,
                    );
                    for (base, angle) in [(0usize, row_angle), (half, column_angle)] {
                        let (sin_value, cos_value) = angle.sin_cos();
                        for offset in [base + index, base + quarter + index] {
                            cos[token * head_dim + offset] = cos_value as f32;
                            sin[token * head_dim + offset] = sin_value as f32;
                        }
                    }
                }
            }
        }
        Self {
            tokens,
            head_dim,
            cos,
            sin,
        }
    }

    /// Rotates each half of each head in place.
    fn apply(&self, values: &mut [f32], tokens: usize, heads: usize, width: usize) {
        let head_dim = self.head_dim;
        let half = head_dim / 2;
        let quarter = half / 2;
        debug_assert_eq!(tokens, self.tokens);
        for token in 0..tokens {
            for head in 0..heads {
                let base = token * width + head * head_dim;
                for part in 0..2 {
                    let part_base = base + part * half;
                    for index in 0..quarter {
                        let left = values[part_base + index];
                        let right = values[part_base + quarter + index];
                        let cos = self.cos[token * head_dim + part * half + index];
                        let sin = self.sin[token * head_dim + part * half + index];
                        values[part_base + index] = left * cos - right * sin;
                        values[part_base + quarter + index] = right * cos + left * sin;
                    }
                }
            }
        }
    }
}

/// Bilinear taps for one axis position mapped onto a `side`-wide learned grid.
fn axis_taps(position: usize, extent: usize, side: usize) -> (Vec<usize>, Vec<f32>) {
    if extent == side {
        return (vec![position], vec![1.0]);
    }
    let scale = (side as f32) / (extent as f32);
    let centre = (position as f32 + 0.5) * scale - 0.5;
    let low = centre.floor().max(0.0);
    let high = (low + 1.0).min((side - 1) as f32);
    let weight = (centre - low).clamp(0.0, 1.0);
    let (low, high) = (low as usize, high as usize);
    if low == high {
        (vec![low], vec![1.0])
    } else {
        (vec![low, high], vec![1.0 - weight, weight])
    }
}

fn layer_norm(input: &[f32], weight: &[f32], bias: Option<&[f32]>, epsilon: f64) -> Vec<f32> {
    let mean = input.iter().map(|value| f64::from(*value)).sum::<f64>() / input.len() as f64;
    let variance = input
        .iter()
        .map(|value| (f64::from(*value) - mean).powi(2))
        .sum::<f64>()
        / input.len() as f64;
    let scale = 1.0 / (variance + epsilon).sqrt();
    input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let normalised = ((f64::from(*value) - mean) * scale) as f32;
            normalised * weight[index] + bias.map_or(0.0, |bias| bias[index])
        })
        .collect()
}

/// GELU with the tanh approximation the ViT family uses.
fn gelu(value: f32) -> f32 {
    let inner = (2.0 / std::f32::consts::PI).sqrt() * (value + 0.044_715 * value.powi(3));
    0.5 * value * (1.0 + inner.tanh())
}

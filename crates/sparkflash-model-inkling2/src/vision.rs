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
//
// The arithmetic below follows Hugging Face Transformers' Inkling-2 implementation
// (src/transformers/models/inkling2/modeling_inkling2.py and image_processing_inkling2.py, and
// src/transformers/vision_utils.py; Copyright 2026 the HuggingFace Team, Apache License 2.0),
// which was read for it. No Transformers source is copied; see NOTICE.

//! The Inkling-2-Small vision path: the image front end, the prompt span, and an f32 host
//! reference of the tower.
//!
//! What the released implementation does, and so what this module does:
//!
//! - The processor does not resize. It pads the right and bottom edges to a multiple of the
//!   40-pixel cell (the 20-pixel patch times the 2x2 merge) with the sample -1 on the 0-255
//!   scale, normalises every sample to `(x - 127.5) / 127.5`, and cuts one cell per
//!   language-model token, in raster order. An image with transparency is first composited onto
//!   an 8-pixel checkerboard ([AlphaPolicy::Checkerboard]).
//! - The embedding flattens a patch as (row, column, channel) and keeps the patches in merge-block
//!   order - cell row, cell column, row in the cell, column in the cell ([patch_position]). It adds
//!   the 32x32 learned position grid resampled bilinearly to the patch grid (half-pixel centres,
//!   borders clamped; [interpolate_positions]) and puts one class token in front of the image.
//! - 23 pre-norm blocks: LayerNorm, attention with a 2D axial RoPE, layer scale; LayerNorm, an erf
//!   GELU MLP, layer scale. The rope's interleaved pairs turn with the patch's column in the first
//!   half of a head and with its row in the second; positions count from one, and the class token
//!   sits at (0, 0), where the rotation is the identity ([axial_rope_tables]).
//! - The merger drops the class token, concatenates each cell's four patches and projects them
//!   through two bias-free linears with a GELU between to the text width. The text model takes
//!   those rows in place of the placeholders' embeddings after its embedding norm.
//!
//! [InklingVision] runs that in f32 on the host. It is the oracle the device tower
//! ([crate::device] runs it on rank 0) is held to, small enough to run on a test image; it is not
//! a serving path.

use std::sync::Arc;

use sha2::{Digest, Sha256};
use sparkflash_model::vision::{
    ImageGrid, ImageSpan, PreparedImage, VisionCodec, VisionError, VisionImage, VisionOptions,
};
use sparkflash_vision::AlphaPolicy;
use sparkflash_weights::WeightSource;
use sparkflash_weights::linalg::decode_by_name;
use sparkflash_weights::quant::bf16_round;

use crate::config::InklingVisionConfig;

/// The most image tokens one image may make.
///
/// The processor pads rather than resizes, so the count is the image's 40-pixel cells: 8,192 is a
/// 12-megapixel photograph (4032 x 3024 makes 101 x 76 = 7,676). The tower's attention is
/// quadratic in the image's 4 x 8,192 patches, which is what the cap bounds; an image over it is
/// refused with the way to send it, never shrunk behind the client's back.
pub const MAX_MERGED_TOKENS: usize = 8_192;
/// The rope base. The release's vision config names no rope parameters, so this is the library
/// default the released code then uses.
pub const AXIAL_ROPE_THETA: f32 = 10_000.0;
/// The sample the processor pads the right and bottom edges with, on the 0-255 scale.
pub const PAD_SAMPLE: f32 = -1.0;
/// The image placeholder (`<|unused_200054|>`, the config's `image_token_id`).
pub const IMAGE_TOKEN_ID: u32 = 200_054;

/// The pixels one language-model token covers along each side: the patch times the merge.
#[must_use]
pub fn cell_side(config: &InklingVisionConfig) -> usize {
    config.patch_size * config.adapter_merge_size
}

/// The token grid an image of `height` by `width` pixels makes: `(rows, columns)` of cells, the
/// last row and column padded out.
///
/// # Errors
/// An empty image, or one over [MAX_MERGED_TOKENS].
pub fn plan_cells(
    height: usize,
    width: usize,
    config: &InklingVisionConfig,
) -> Result<(usize, usize), String> {
    if height == 0 || width == 0 {
        return Err(format!("a {width}x{height} image has no pixels"));
    }
    let cell = cell_side(config);
    let (rows, columns) = (height.div_ceil(cell), width.div_ceil(cell));
    let tokens = rows * columns;
    if tokens > MAX_MERGED_TOKENS {
        return Err(format!(
            "a {width}x{height} image makes {tokens} image tokens ({columns}x{rows} cells of \
             {cell} pixels), over the {MAX_MERGED_TOKENS}-token limit for one image; send it \
             smaller, or with \"detail\": \"low\""
        ));
    }
    Ok((rows, columns))
}

/// The patch grid an image makes, `(grid_h, grid_w)`: [plan_cells] times the merge.
///
/// # Errors
/// As [plan_cells].
pub fn plan_grid(
    height: usize,
    width: usize,
    config: &InklingVisionConfig,
) -> Result<(usize, usize), String> {
    let (rows, columns) = plan_cells(height, width, config)?;
    let merge = config.adapter_merge_size;
    Ok((rows * merge, columns * merge))
}

/// One patch's flattened samples: three channels of `patch_size` squared.
#[must_use]
pub fn patch_values(config: &InklingVisionConfig) -> usize {
    3 * config.patch_size * config.patch_size
}

/// Where patch `index` of a merge-block-ordered grid `grid_w` patches wide sits, as
/// `(row, column)` of the patch grid.
#[must_use]
pub fn patch_position(index: usize, grid_w: usize, merge: usize) -> (usize, usize) {
    let per_cell = merge * merge;
    let cells_w = grid_w / merge;
    let (cell, inner) = (index / per_cell, index % per_cell);
    (
        (cell / cells_w) * merge + inner / merge,
        (cell % cells_w) * merge + inner % merge,
    )
}

/// The processor's normalisation of one sample, in f32: rescale by 1/255, then mean and standard
/// deviation 0.5, which the processor fuses into `(x - 127.5) / 127.5`.
#[must_use]
pub fn normalize_sample(sample: f32) -> f32 {
    (sample - 127.5) / 127.5
}

/// The patches of an RGB canvas (`height * width * 3` bytes, rows top to bottom), as the
/// processor and the embedding cut them: `[patches][3 p^2]` in merge-block order, each patch's
/// samples in (row, column, channel) order, the padding [PAD_SAMPLE], and every value rounded to
/// bf16 as the model's cast of `pixel_values` rounds it. Returns the patches and the grid.
///
/// # Errors
/// A canvas of the wrong length, or one [plan_grid] refuses.
pub fn patchify(
    rgb: &[u8],
    height: usize,
    width: usize,
    config: &InklingVisionConfig,
) -> Result<(Vec<f32>, usize, usize), String> {
    if rgb.len() != height * width * 3 {
        return Err(format!(
            "a {width}x{height} canvas is {} bytes, not {}",
            rgb.len(),
            height * width * 3
        ));
    }
    let (grid_h, grid_w) = plan_grid(height, width, config)?;
    let patch = config.patch_size;
    let merge = config.adapter_merge_size;
    let features = patch_values(config);
    let pad = bf16_round(normalize_sample(PAD_SAMPLE));
    let mut out = vec![0f32; grid_h * grid_w * features];
    for (index, values) in out.chunks_exact_mut(features).enumerate() {
        let (row, column) = patch_position(index, grid_w, merge);
        for py in 0..patch {
            for px in 0..patch {
                let (y, x) = (row * patch + py, column * patch + px);
                for channel in 0..3 {
                    values[(py * patch + px) * 3 + channel] = if y < height && x < width {
                        bf16_round(normalize_sample(f32::from(rgb[(y * width + x) * 3 + channel])))
                    } else {
                        pad
                    };
                }
            }
        }
    }
    Ok((out, grid_h, grid_w))
}

/// The two bilinear taps of position `index` on an axis of `size` patches, into a `side`-long
/// learned table: half-pixel centres, each tap clamped to the table, weighted `1 - distance`.
/// Both taps are kept when the clamp folds them onto one entry, as the released gather does.
fn axis_taps(index: usize, size: usize, side: usize) -> [(usize, f32); 2] {
    let source = ((index as f32 + 0.5) * side as f32) / size as f32 - 0.5;
    let floor = source.floor();
    let mut taps = [(0usize, 0f32); 2];
    for (offset, tap) in taps.iter_mut().enumerate() {
        let raw = floor as i64 + offset as i64;
        let clamped = raw.clamp(0, side as i64 - 1) as usize;
        let distance = (source - floor - offset as f32).abs();
        *tap = (clamped, (1.0 - distance).max(0.0));
    }
    taps
}

/// The learned `side` x `side` position grid (`table`, `[side * side][width]`) resampled to a
/// `grid_h` x `grid_w` patch grid: `[patches][width]` in merge-block order, each row the f32 sum
/// of its four taps in (row tap, column tap) order.
#[must_use]
pub fn interpolate_positions(
    table: &[f32],
    side: usize,
    width: usize,
    grid_h: usize,
    grid_w: usize,
    merge: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; grid_h * grid_w * width];
    for (index, slot) in out.chunks_exact_mut(width).enumerate() {
        let (row, column) = patch_position(index, grid_w, merge);
        let rows = axis_taps(row, grid_h, side);
        let columns = axis_taps(column, grid_w, side);
        for (grid_row, row_weight) in rows {
            for (grid_column, column_weight) in columns {
                let weight = row_weight * column_weight;
                let base = (grid_row * side + grid_column) * width;
                for (value, entry) in slot.iter_mut().zip(&table[base..base + width]) {
                    *value += entry * weight;
                }
            }
        }
    }
    out
}

/// The axial rope's angles along one axis: `cos` and `sin` of `position * f_j` for positions
/// `0..positions` and `j < head_dim / 4`, `[positions][head_dim / 4]`, with
/// `f_j = 1 / theta^(2j / (head_dim / 2))` computed in f32 as the released code computes it.
#[must_use]
pub fn axial_rope_tables(positions: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let spatial = head_dim / 2;
    let count = spatial / 2;
    let inverse: Vec<f32> = (0..count)
        .map(|j| 1.0 / AXIAL_ROPE_THETA.powf((2 * j) as f32 / spatial as f32))
        .collect();
    let mut cos = Vec::with_capacity(positions * count);
    let mut sin = Vec::with_capacity(positions * count);
    for position in 0..positions {
        for frequency in &inverse {
            let angle = position as f32 * frequency;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

/// A row-major matrix read from the checkpoint.
struct Matrix {
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
        Ok(Self { columns, values })
    }

    fn apply(&self, input: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
        self.values
            .chunks_exact(self.columns)
            .enumerate()
            .map(|(row, weights)| {
                let dot: f32 = weights.iter().zip(input).map(|(w, x)| w * x).sum();
                dot + bias.map_or(0.0, |bias| bias[row])
            })
            .collect()
    }
}

/// A LayerNorm's weight and bias.
struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Norm {
    fn load(source: &dyn WeightSource, prefix: &str, width: usize) -> Result<Self, String> {
        let weight = vector(source, &format!("{prefix}.weight"), width)?;
        let bias = vector(source, &format!("{prefix}.bias"), width)?;
        Ok(Self { weight, bias })
    }

    fn apply(&self, input: &[f32], epsilon: f64) -> Vec<f32> {
        layer_norm(input, &self.weight, &self.bias, epsilon)
    }
}

/// One vector of `width` values.
fn vector(source: &dyn WeightSource, name: &str, width: usize) -> Result<Vec<f32>, String> {
    let values = decode_by_name(source, name, 1 << 24).map_err(|error| format!("{name}: {error}"))?;
    if values.len() != width {
        return Err(format!("{name} holds {} values, expected {width}", values.len()));
    }
    Ok(values)
}

/// One transformer block.
struct Block {
    ln_1: Norm,
    ln_2: Norm,
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

/// The tower's weights, in f32.
pub struct InklingVision {
    config: InklingVisionConfig,
    patch: Matrix,
    position: Vec<f32>,
    class_embedding: Vec<f32>,
    cls_position: Vec<f32>,
    ln_pre: Norm,
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

/// The width a block's MLP expands to: the released `mlp_ratio` of 4.
#[must_use]
pub fn mlp_width(width: usize) -> usize {
    width * 4
}

impl InklingVision {
    /// Reads the tower's weights from a checkpoint source.
    ///
    /// # Errors
    /// A missing or mis-shaped tensor.
    pub fn load(config: &InklingVisionConfig, source: Arc<dyn WeightSource>) -> Result<Self, String> {
        let source = source.as_ref();
        let prefix = "model.visual";
        let width = config.hidden_size;
        let intermediate = mlp_width(width);
        let embeddings = format!("{prefix}.backbone.embeddings");
        let patch = Matrix::load(
            source,
            &format!("{embeddings}.patch_embedding.weight"),
            width,
            patch_values(config),
        )?;
        let side = config.posemb_grid_size;
        let position = decode_by_name(source, &format!("{embeddings}.position_embedding"), 1 << 30)
            .map_err(|error| format!("position_embedding: {error}"))?;
        if position.len() != side * side * width {
            return Err(format!(
                "position_embedding holds {} values, expected a {side}x{side} grid of {width}",
                position.len()
            ));
        }
        let class_embedding = vector(source, &format!("{embeddings}.class_embedding"), width)?;
        let cls_position = vector(source, &format!("{embeddings}.cls_position_embedding"), width)?;
        let ln_pre = Norm::load(source, &format!("{prefix}.backbone.ln_pre"), width)?;
        let mut blocks = Vec::with_capacity(config.depth);
        for layer in 0..config.depth {
            let p = format!("{prefix}.backbone.layers.{layer}");
            let attention = format!("{p}.self_attn");
            let projection = |name: &str| {
                Matrix::load(source, &format!("{attention}.{name}.weight"), width, width)
            };
            let bias = |name: &str| vector(source, &format!("{attention}.{name}.bias"), width);
            blocks.push(Block {
                ln_1: Norm::load(source, &format!("{p}.ln_1"), width)?,
                ln_2: Norm::load(source, &format!("{p}.ln_2"), width)?,
                ls_1: vector(source, &format!("{p}.ls_1.gamma"), width)?,
                ls_2: vector(source, &format!("{p}.ls_2.gamma"), width)?,
                q: projection("q_proj")?,
                q_bias: bias("q_proj")?,
                k: projection("k_proj")?,
                k_bias: bias("k_proj")?,
                v: projection("v_proj")?,
                v_bias: bias("v_proj")?,
                out: projection("out_proj")?,
                out_bias: bias("out_proj")?,
                fc: Matrix::load(source, &format!("{p}.mlp.c_fc.weight"), intermediate, width)?,
                fc_bias: vector(source, &format!("{p}.mlp.c_fc.bias"), intermediate)?,
                proj: Matrix::load(source, &format!("{p}.mlp.c_proj.weight"), width, intermediate)?,
                proj_bias: vector(source, &format!("{p}.mlp.c_proj.bias"), width)?,
            });
        }
        let merge = config.adapter_merge_size;
        let proj_in = Matrix::load(
            source,
            &format!("{prefix}.proj_in.weight"),
            config.projection_hidden_dim,
            width * merge * merge,
        )?;
        let proj_out = Matrix::load(
            source,
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

    /// Encodes an RGB canvas (`height * width * 3` bytes) into image tokens through [patchify]
    /// and [Self::encode].
    ///
    /// # Errors
    /// As [patchify] and [Self::encode].
    pub fn encode_canvas(&self, rgb: &[u8], height: usize, width: usize) -> Result<Vec<f32>, String> {
        let (pixels, grid_h, grid_w) = patchify(rgb, height, width, &self.config)?;
        self.encode(&pixels, grid_h, grid_w)
    }

    /// Encodes patches into image tokens, `[grid_h * grid_w / merge^2][out_hidden]`, one per cell
    /// in raster order.
    ///
    /// `pixels` is `[grid_h * grid_w][3 p^2]` as [patchify] lays it out: merge-block order, each
    /// patch's samples in (row, column, channel) order.
    ///
    /// # Errors
    /// A grid that does not divide into merge blocks, patches of the wrong length, or more image
    /// tokens than [MAX_MERGED_TOKENS].
    pub fn encode(&self, pixels: &[f32], grid_h: usize, grid_w: usize) -> Result<Vec<f32>, String> {
        let config = &self.config;
        let width = config.hidden_size;
        let merge = config.adapter_merge_size;
        let epsilon = config.layer_norm_eps;
        if grid_h == 0 || grid_w == 0 || !grid_h.is_multiple_of(merge) || !grid_w.is_multiple_of(merge) {
            return Err(format!(
                "a {grid_h}x{grid_w} patch grid does not divide into {merge}x{merge} merge blocks"
            ));
        }
        let patches = grid_h * grid_w;
        let merged = patches / (merge * merge);
        if merged > MAX_MERGED_TOKENS {
            return Err(format!(
                "a {grid_h}x{grid_w} patch grid makes {merged} image tokens, over the \
                 {MAX_MERGED_TOKENS}-token cap"
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

        // The class token, then each patch's embedding plus its resampled position.
        let positions = interpolate_positions(
            &self.position,
            config.posemb_grid_size,
            width,
            grid_h,
            grid_w,
            merge,
        );
        let mut hidden = vec![0f32; tokens * width];
        for ((value, class), position) in hidden[..width]
            .iter_mut()
            .zip(&self.class_embedding)
            .zip(&self.cls_position)
        {
            *value = class + position;
        }
        for (patch, (values, position)) in pixels
            .chunks_exact(patch_width)
            .zip(positions.chunks_exact(width))
            .enumerate()
        {
            let encoded = self.patch.apply(values, None);
            let row = &mut hidden[(patch + 1) * width..(patch + 2) * width];
            for ((slot, value), position) in row.iter_mut().zip(encoded).zip(position) {
                *slot = value + position;
            }
        }
        for token in hidden.chunks_exact_mut(width) {
            let normed = self.ln_pre.apply(token, epsilon);
            token.copy_from_slice(&normed);
        }

        // Patch (row, column) counts from one; the class token is at (0, 0).
        let heads = config.heads;
        let head_dim = width / heads;
        let mut grid_positions = Vec::with_capacity(tokens);
        grid_positions.push((0usize, 0usize));
        for patch in 0..patches {
            let (row, column) = patch_position(patch, grid_w, merge);
            grid_positions.push((row + 1, column + 1));
        }
        let (cos, sin) = axial_rope_tables(grid_h.max(grid_w) + 1, head_dim);

        for block in &self.blocks {
            // x = x + ls_1 * attention(ln_1(x))
            let mut q = Vec::with_capacity(tokens * width);
            let mut k = Vec::with_capacity(tokens * width);
            let mut v = Vec::with_capacity(tokens * width);
            for token in hidden.chunks_exact(width) {
                let normed = block.ln_1.apply(token, epsilon);
                q.extend(block.q.apply(&normed, Some(&block.q_bias)));
                k.extend(block.k.apply(&normed, Some(&block.k_bias)));
                v.extend(block.v.apply(&normed, Some(&block.v_bias)));
            }
            rotate(&mut q, &grid_positions, &cos, &sin, heads, head_dim);
            rotate(&mut k, &grid_positions, &cos, &sin, heads, head_dim);
            let attention = attend(&q, &k, &v, tokens, heads, head_dim);
            for (token, row) in hidden.chunks_exact_mut(width).enumerate() {
                let projected = block.out.apply(
                    &attention[token * width..(token + 1) * width],
                    Some(&block.out_bias),
                );
                for ((value, scale), add) in row.iter_mut().zip(&block.ls_1).zip(projected) {
                    *value += scale * add;
                }
            }
            // x = x + ls_2 * mlp(ln_2(x))
            for row in hidden.chunks_exact_mut(width) {
                let normed = block.ln_2.apply(row, epsilon);
                let mut up = block.fc.apply(&normed, Some(&block.fc_bias));
                for value in &mut up {
                    *value = gelu(*value);
                }
                let down = block.proj.apply(&up, Some(&block.proj_bias));
                for ((value, scale), add) in row.iter_mut().zip(&block.ls_2).zip(down) {
                    *value += scale * add;
                }
            }
        }

        // The merger: the patches are already in merge-block order, so each cell's four rows are
        // consecutive and the class token is the only row left out.
        let cell = width * merge * merge;
        let mut out = Vec::with_capacity(merged * config.out_hidden_size);
        for row in hidden[width..].chunks_exact(cell) {
            let mut projected = self.proj_in.apply(row, None);
            for value in &mut projected {
                *value = gelu(*value);
            }
            out.extend(self.proj_out.apply(&projected, None));
        }
        Ok(out)
    }
}

/// Turns every head of every row by its position's angles, in place: pair `(2i, 2i + 1)` of a head
/// takes the column's angle `i` for `i < head_dim / 4` and the row's angle `i - head_dim / 4`
/// after, as `(x0 cos - x1 sin, x1 cos + x0 sin)`.
fn rotate(
    values: &mut [f32],
    positions: &[(usize, usize)],
    cos: &[f32],
    sin: &[f32],
    heads: usize,
    head_dim: usize,
) {
    let count = head_dim / 4;
    let width = heads * head_dim;
    for (row, &(grid_row, grid_column)) in values.chunks_exact_mut(width).zip(positions) {
        for head in row.chunks_exact_mut(head_dim) {
            for (pair, values) in head.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                let (position, frequency) = if pair < count {
                    (grid_column, pair)
                } else {
                    (grid_row, pair - count)
                };
                let (c, s) = (cos[position * count + frequency], sin[position * count + frequency]);
                let (x0, x1) = (values[0], values[1]);
                values[0] = x0 * c - x1 * s;
                values[1] = x1 * c + x0 * s;
            }
        }
    }
}

/// Bidirectional softmax attention over all `tokens`, each head on its own, scale `head_dim^-1/2`.
fn attend(q: &[f32], k: &[f32], v: &[f32], tokens: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let width = heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0f32; tokens * width];
    let mut scores = vec![0f32; tokens];
    for head in 0..heads {
        let at = |token: usize| token * width + head * head_dim;
        for query in 0..tokens {
            let q_row = &q[at(query)..at(query) + head_dim];
            for (key, score) in scores.iter_mut().enumerate() {
                let k_row = &k[at(key)..at(key) + head_dim];
                *score = q_row.iter().zip(k_row).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denominator = 0.0f32;
            for score in &mut scores {
                *score = (*score - maximum).exp();
                denominator += *score;
            }
            let out_row = at(query);
            for (key, score) in scores.iter().enumerate() {
                let weight = score / denominator;
                for (slot, value) in out[out_row..out_row + head_dim]
                    .iter_mut()
                    .zip(&v[at(key)..at(key) + head_dim])
                {
                    *slot += weight * value;
                }
            }
        }
    }
    out
}

fn layer_norm(input: &[f32], weight: &[f32], bias: &[f32], epsilon: f64) -> Vec<f32> {
    let mean = input.iter().map(|value| f64::from(*value)).sum::<f64>() / input.len() as f64;
    let variance = input
        .iter()
        .map(|value| (f64::from(*value) - mean).powi(2))
        .sum::<f64>()
        / input.len() as f64;
    let scale = 1.0 / (variance + epsilon).sqrt();
    input
        .iter()
        .zip(weight.iter().zip(bias))
        .map(|(value, (weight, bias))| ((f64::from(*value) - mean) * scale) as f32 * weight + bias)
        .collect()
}

/// The exact (erf) GELU the released config's `"gelu"` names, as the Qwen tower computes it.
fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + libm::erf(f64::from(value) / std::f64::consts::SQRT_2) as f32)
}

/// Inkling's image front end: decode, plan, and the prompt span.
///
/// The released template writes an image as `<|content_image|>`, one placeholder and
/// `<|end_message|>`; the processor repeats the placeholder once per cell. A span here runs from
/// that `<|content_image|>` to that `<|end_message|>`, so [ImageSpan::row] names the delimiters
/// [sparkflash_model::vision::SpanRow::Start] and [sparkflash_model::vision::SpanRow::End] and
/// the cells between them [sparkflash_model::vision::SpanRow::Image] in raster order. The
/// delimiters are ordinary text tokens; only the cells take the tower's rows.
#[derive(Debug, Clone)]
pub struct InklingVisionCodec {
    config: InklingVisionConfig,
    image_token: u32,
    open: u32,
    close: u32,
}

impl InklingVisionCodec {
    /// A codec for `config`'s tower whose template writes `image_token` between `open`
    /// (`<|content_image|>`) and `close` (`<|end_message|>`).
    #[must_use]
    pub fn new(config: InklingVisionConfig, image_token: u32, open: u32, close: u32) -> Self {
        Self {
            config,
            image_token,
            open,
            close,
        }
    }

    /// The tower geometry the codec plans for.
    #[must_use]
    pub fn config(&self) -> &InklingVisionConfig {
        &self.config
    }
}

/// The key names the family, the canvas geometry and the pixels, so two canvases with the same
/// bytes and a different shape never collide, and no other family's key is in this namespace.
fn image_key(pixels: &[u8], width: usize, height: usize, rows: usize, columns: usize) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"inkling2-small/vision/v1\0");
    for value in [width, height, rows, columns] {
        hasher.update(u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes());
    }
    hasher.update(pixels);
    hasher.finalize().into()
}

impl VisionCodec for InklingVisionCodec {
    fn image_token_id(&self) -> u32 {
        self.image_token
    }

    fn prepare(&self, bytes: &[u8], options: &VisionOptions) -> Result<PreparedImage, VisionError> {
        let limits = sparkflash_vision::DecodeLimits {
            max_bytes: options.max_bytes,
            max_dimension: options.max_dimension,
        };
        let mut image = sparkflash_vision::decode_with_alpha(bytes, &limits, AlphaPolicy::Checkerboard)
            .map_err(|error| VisionError::new(error.to_string()))?;
        // Low detail is this server's option, not the processor's: the processor never resizes,
        // so a full-detail image is its own pixels and a low-detail one is shrunk first.
        if let Some(long_side) = options.max_long_side {
            image = sparkflash_vision::shrink_long_side(&image, long_side as usize);
        }
        let (rows, columns) =
            plan_cells(image.height, image.width, &self.config).map_err(VisionError::new)?;
        let key = image_key(&image.data, image.width, image.height, rows, columns);
        Ok(PreparedImage {
            grid: ImageGrid {
                n_llm_h: rows,
                n_llm_w: columns,
                best_height: image.height,
                best_width: image.width,
                line_stride: columns,
            },
            pixels: Arc::from(image.data),
            key,
        })
    }

    fn expand(
        &self,
        prompt: &[u32],
        images: &[PreparedImage],
    ) -> Result<(Vec<u32>, Vec<ImageSpan>), VisionError> {
        let placeholders = prompt.iter().filter(|id| **id == self.image_token).count();
        if placeholders != images.len() {
            return Err(VisionError::new(format!(
                "the prompt has {placeholders} image placeholders but the request has {} images",
                images.len()
            )));
        }
        let mut ids = Vec::with_capacity(prompt.len());
        let mut spans = Vec::with_capacity(images.len());
        for (index, &token) in prompt.iter().enumerate() {
            if token != self.image_token {
                ids.push(token);
                continue;
            }
            let image = &images[spans.len()];
            if ids.last() != Some(&self.open) || prompt.get(index + 1) != Some(&self.close) {
                return Err(VisionError::new(format!(
                    "image {} is not written as the template writes it, an image placeholder \
                     between <|content_image|> and <|end_message|>",
                    spans.len() + 1
                )));
            }
            let start = ids.len() - 1;
            ids.extend(std::iter::repeat_n(self.image_token, image.grid.image_rows()));
            spans.push(ImageSpan {
                start,
                grid: image.grid,
                key: image.key,
            });
        }
        Ok((ids, spans))
    }
}

/// What a rank checks of a prompt's image spans before it prefills them, and what rank 0 checks
/// before it broadcasts them: every rank of a group runs the same check on the same frames, so a
/// span one rank would refuse is refused before any rank has started the request.
///
/// A span must be the codec's: its grid the canvas's [plan_cells], its rows packed without
/// separators, inside the prompt and after the span before it, and every position between its
/// delimiters the placeholder. A placeholder outside every span is refused too, since it would be
/// read as an unused text token. The rank that encodes also needs each image's canvas.
#[derive(Debug, Clone)]
pub struct SpanCheck {
    vision: Option<InklingVisionConfig>,
    canvases: bool,
}

impl SpanCheck {
    /// The check for a checkpoint whose tower is `vision` (`None`: no tower, so no image), on a
    /// rank that encodes the canvases when `canvases` is set.
    #[must_use]
    pub fn new(vision: Option<InklingVisionConfig>, canvases: bool) -> Self {
        Self { vision, canvases }
    }

    /// Checks `images`' spans against `tokens`.
    ///
    /// # Errors
    /// The first span, or stray placeholder, that breaks one of the rules above.
    pub fn check(&self, tokens: &[u32], images: &[VisionImage]) -> Result<(), String> {
        let Some(config) = &self.vision else {
            if images.is_empty() {
                return Ok(());
            }
            return Err(format!(
                "the request carries {} images and this checkpoint has no vision tower",
                images.len()
            ));
        };
        let stray = |from: usize, to: usize| match tokens[from..to]
            .iter()
            .position(|token| *token == IMAGE_TOKEN_ID)
        {
            Some(offset) => Err(format!(
                "position {} holds the image placeholder {IMAGE_TOKEN_ID} outside every image",
                from + offset
            )),
            None => Ok(()),
        };
        let mut covered = 0usize;
        for (index, image) in images.iter().enumerate() {
            let span = &image.span;
            let grid = &span.grid;
            let number = index + 1;
            if span.start < covered {
                return Err(format!(
                    "image {number} starts at {} inside the span before it, which runs to {}",
                    span.start,
                    covered - 1
                ));
            }
            if grid.line_stride != grid.n_llm_w {
                return Err(format!(
                    "image {number}'s rows are {} positions apart; Inkling packs an image's {} \
                     cells a row with no separator",
                    grid.line_stride, grid.n_llm_w
                ));
            }
            let planned = plan_cells(grid.best_height, grid.best_width, config)
                .map_err(|error| format!("image {number}: {error}"))?;
            if planned != (grid.n_llm_h, grid.n_llm_w) {
                return Err(format!(
                    "image {number} is {}x{} cells, and its {}x{} canvas makes {}x{}",
                    grid.n_llm_w, grid.n_llm_h, grid.best_width, grid.best_height, planned.1,
                    planned.0
                ));
            }
            if span.end() > tokens.len() {
                return Err(format!(
                    "image {number}'s span ends at {}, past the {}-position prompt",
                    span.end(),
                    tokens.len()
                ));
            }
            stray(covered, span.start + 1)?;
            if let Some(offset) = tokens[span.start + 1..span.end() - 1]
                .iter()
                .position(|token| *token != IMAGE_TOKEN_ID)
            {
                return Err(format!(
                    "image {number}'s cell {offset} is token {}, not the placeholder",
                    tokens[span.start + 1 + offset]
                ));
            }
            let canvas = grid.best_height * grid.best_width * 3;
            if self.canvases && image.pixels.len() != canvas {
                return Err(format!(
                    "image {number}'s canvas is {} bytes, and a {}x{} RGB canvas is {canvas}",
                    image.pixels.len(),
                    grid.best_width,
                    grid.best_height
                ));
            }
            covered = span.end();
        }
        stray(covered, tokens.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn released() -> InklingVisionConfig {
        InklingVisionConfig {
            depth: 23,
            hidden_size: 1024,
            heads: 16,
            patch_size: 20,
            temporal_patch_size: 2,
            adapter_merge_size: 2,
            projection_hidden_dim: 4096,
            posemb_grid_size: 32,
            layer_norm_eps: 1e-5,
            out_hidden_size: 4608,
        }
    }

    /// The processor pads rather than resizes: a 41 x 81 image is 2 x 3 cells.
    #[test]
    fn the_plan_pads_each_side_up_to_the_cell() {
        let config = released();
        assert_eq!(plan_cells(41, 81, &config), Ok((2, 3)));
        assert_eq!(plan_cells(40, 40, &config), Ok((1, 1)));
        assert_eq!(plan_cells(1, 1, &config), Ok((1, 1)));
        assert_eq!(plan_grid(41, 81, &config), Ok((4, 6)));
        assert!(plan_cells(0, 40, &config).is_err());
        // 4032 x 3024 is 101 x 76 = 7,676 tokens; 4096 x 4096 is 103^2 = 10,609.
        assert_eq!(plan_cells(3024, 4032, &config), Ok((76, 101)));
        let refused = plan_cells(4096, 4096, &config).expect_err("over the cap");
        assert!(refused.contains("detail"), "{refused}");
    }

    /// Merge-block order walks a cell's four patches before moving to the next cell.
    #[test]
    fn patches_are_in_merge_block_order() {
        // A 4 x 6 patch grid is 2 x 3 cells.
        let order: Vec<(usize, usize)> = (0..8).map(|index| patch_position(index, 6, 2)).collect();
        assert_eq!(
            order,
            vec![(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
        );
        assert_eq!(patch_position(12, 6, 2), (2, 0));
    }

    /// Each patch is (row, column, channel) samples, and the padding is the -1 sample.
    #[test]
    fn patchify_lays_samples_out_as_the_processor_does() {
        let config = InklingVisionConfig {
            patch_size: 2,
            ..released()
        };
        // A 3 x 5 image (cell 4): one cell row, two cell columns, padded to 4 x 8.
        let (height, width) = (3, 5);
        let rgb: Vec<u8> = (0..height * width * 3).map(|index| index as u8).collect();
        let (pixels, grid_h, grid_w) = patchify(&rgb, height, width, &config).expect("patchify");
        assert_eq!((grid_h, grid_w), (2, 4));
        let features = 12;
        let sample = |y: usize, x: usize, c: usize| {
            bf16_round(normalize_sample(f32::from(rgb[(y * width + x) * 3 + c])))
        };
        let pad = bf16_round(normalize_sample(PAD_SAMPLE));
        // Patch 0 is rows 0-1, columns 0-1; its sample 5 is (row 0, column 1, channel 2).
        assert_eq!(pixels[5], sample(0, 1, 2));
        // Patch 2 is rows 2-3, columns 0-1: row 3 is padding.
        let patch2 = &pixels[2 * features..3 * features];
        assert_eq!(patch2[0], sample(2, 0, 0));
        assert_eq!(patch2[6], pad);
        // Patch 5 is rows 0-1, columns 6-7 - wholly right of the image.
        assert!(pixels[5 * features..6 * features].iter().all(|value| *value == pad));
        assert_eq!(pad, bf16_round(-128.5 / 127.5));
    }

    /// At the table's own size the resampling is the identity; off it, the taps are clamped.
    #[test]
    fn the_position_grid_resamples_with_clamped_half_pixel_taps() {
        assert_eq!(axis_taps(3, 32, 32), [(3, 1.0), (4, 0.0)]);
        assert_eq!(axis_taps(31, 32, 32), [(31, 1.0), (31, 0.0)]);
        // 64 patches on a 32 table: patch 0's centre is -0.25, which folds onto entry 0.
        let [(low, low_weight), (high, high_weight)] = axis_taps(0, 64, 32);
        assert_eq!((low, high), (0, 0));
        assert!((low_weight + high_weight - 1.0).abs() < 1e-6);
        // Patch 1's centre is 0.25: three quarters of entry 0, a quarter of entry 1.
        assert_eq!(axis_taps(1, 64, 32), [(0, 0.75), (1, 0.25)]);
    }

    /// Position zero is the identity, and the first and last frequencies are 1 and theta^-15/16.
    #[test]
    fn the_rope_tables_start_at_the_identity() {
        let (cos, sin) = axial_rope_tables(3, 64);
        assert_eq!(cos.len(), 3 * 16);
        assert!(cos[..16].iter().all(|value| *value == 1.0));
        assert!(sin[..16].iter().all(|value| *value == 0.0));
        assert!((sin[16] - 1f32.sin()).abs() < 1e-7);
        let last = 1.0 / 10_000f32.powf(30.0 / 32.0);
        assert!((sin[31] - last.sin()).abs() < 1e-7);
    }

    /// The span runs from `<|content_image|>` to `<|end_message|>` with one placeholder a cell.
    #[test]
    fn expand_repeats_the_placeholder_once_per_cell() {
        let codec = InklingVisionCodec::new(released(), 9, 7, 8);
        let image = PreparedImage {
            grid: ImageGrid {
                n_llm_h: 2,
                n_llm_w: 3,
                best_height: 60,
                best_width: 100,
                line_stride: 3,
            },
            pixels: Arc::from(vec![0u8; 60 * 100 * 3]),
            key: [1; 32],
        };
        let (ids, spans) = codec
            .expand(&[1, 7, 9, 8, 2], std::slice::from_ref(&image))
            .expect("expand");
        assert_eq!(ids, vec![1, 7, 9, 9, 9, 9, 9, 9, 8, 2]);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].start, 1);
        assert_eq!(spans[0].end(), 9);
        assert_eq!(spans[0].row(2), sparkflash_model::vision::SpanRow::Image(0));
        assert!(codec.expand(&[1, 9, 8], std::slice::from_ref(&image)).is_err());
        assert!(codec.expand(&[1, 7, 9, 2], std::slice::from_ref(&image)).is_err());
        assert!(codec.expand(&[1, 7, 9, 8], &[]).is_err());
    }
}

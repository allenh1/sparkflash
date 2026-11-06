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

//! The Qwen3.8-Flash-Next vision path.
//!
//! The checkpoint carries a 27-layer ViT (width 1152, 16 heads, 4304-wide MLP) followed by a
//! 2x2 patch merger that projects each merged patch to the 2560-wide text embedding. The released
//! processor does not pad: it resizes the image with "smart_resize" to a multiple of 32, normalises
//! with mean and standard deviation 0.5, and emits the patches in 2x2-merge-block order, with the
//! two temporal frames of a still image being the same frame.
//!
//! This module owns the host side of that path: the image preparation, the prompt span, and a
//! reference vision tower. The device arm can use [Qwen4Vision::encode] to turn a canvas into the
//! rows it writes over the image-placeholder embeddings. The tower runs in f32 on the host, so it
//! is a correctness path first; the device kernels are the throughput path.

use std::sync::Arc;

use half::bf16;
use sha2::{Digest, Sha256};
use sparkflash_model::vision::{
    ImageGrid, ImageSpan, PreparedImage, VisionCodec, VisionError, VisionOptions,
};
use sparkflash_vision::decode::AlphaPolicy;
use sparkflash_weights::WeightSource;
use sparkflash_weights::linalg::layer_norm;
use sparkflash_weights::safetensors::SafeTensorsSource;

use crate::config::Qwen4ExpConfig;
use crate::forward::Mat;

/// The most merged image tokens the host reference tower will plan.
///
/// The tower materialises a `patches x patches` score matrix, so memory grows quadratically with
/// the patch grid. 576 merged tokens is the released learned-position grid (48 x 48 patches, 2 x 2
/// merge), or 768 x 768 pixels at a 32-pixel cell; it keeps the score matrix at 5.3 Mi pixels
/// (21 MiB). The released processor's 16 Mi-pixel ceiling would need 65,536 patches and a 17 GiB
/// matrix, which is not a viable host path. The device kernels will relax this; until then a larger
/// image is downscaled by `smart_resize` and `encode` refuses a larger grid outright.
pub const MAX_MERGED_TOKENS: usize = 576;

/// The epsilon of every LayerNorm in the tower.
const VISION_NORM_EPS: f32 = 1e-6;

/// The vision tower's geometry, derived from the released config.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisionTowerConfig {
    /// ViT depth (27).
    pub depth: usize,
    /// ViT width (1152).
    pub hidden_size: usize,
    /// Attention heads (16).
    pub num_heads: usize,
    /// MLP intermediate width (4304).
    pub intermediate_size: usize,
    /// Merger output width (2560).
    pub out_hidden_size: usize,
    /// Patch side in pixels (16).
    pub patch_size: usize,
    /// Temporal patch depth (2; a still duplicates its frame).
    pub temporal_patch_size: usize,
    /// Spatial merge per side (2).
    pub spatial_merge_size: usize,
    /// Learned position grid entries (2304 = 48 x 48).
    pub num_position_embeddings: usize,
    /// Axial RoPE base (10,000).
    pub rope_theta: f64,
    /// Smallest planned image in pixels (65,536).
    pub min_pixels: usize,
    /// Largest planned image in pixels, capped by [`MAX_MERGED_TOKENS`].
    pub max_pixels: usize,
}

impl VisionTowerConfig {
    /// This tower as the shared geometry the device kernels read
    /// ([sparkflash_model::VisionGeometry]).
    ///
    /// The two fields the released Qwen tower does not name are the merger's MLP width, which its
    /// two-layer patch merge does not have, and the image channels, which every release here fixes
    /// at 3.
    #[must_use]
    pub fn geometry(&self) -> sparkflash_model::VisionGeometry {
        sparkflash_model::VisionGeometry {
            depth: self.depth,
            hidden_size: self.hidden_size,
            num_heads: self.num_heads,
            intermediate_size: self.intermediate_size,
            out_hidden_size: self.out_hidden_size,
            projection_intermediate_size: 0,
            in_channels: 3,
            patch_size: self.patch_size,
            temporal_patch_size: self.temporal_patch_size,
            spatial_merge_size: self.spatial_merge_size,
            num_position_embeddings: self.num_position_embeddings,
            rope_theta: self.rope_theta,
            min_pixels: self.min_pixels,
            max_pixels: self.max_pixels,
        }
    }

    /// The shared architecture's structural tower geometry, or None without a tower.
    ///
    /// This reads the checkpoint rather than the carried configuration: a fine-tune that changes
    /// the patch grid or the merge factor keeps the same codec arithmetic.
    #[must_use]
    pub fn from_architecture(architecture: &sparkflash_model::Architecture) -> Option<Self> {
        let geometry = architecture.vision_geometry()?;
        // A geometry the kernels cannot run is not a tower: refuse it here rather than build a
        // codec with a zero merge cell or a zero-width merger.
        geometry.validate().ok()?;
        Some(Self::from_geometry(&geometry))
    }

    /// The arm's defaults over one validated structural geometry.
    #[must_use]
    fn from_geometry(geometry: &sparkflash_model::VisionGeometry) -> Self {
        let cell = geometry.patch_size * geometry.spatial_merge_size;
        Self {
            depth: geometry.depth,
            hidden_size: geometry.hidden_size,
            num_heads: geometry.num_heads,
            intermediate_size: geometry.intermediate_size,
            out_hidden_size: geometry.out_hidden_size,
            patch_size: geometry.patch_size,
            temporal_patch_size: geometry.temporal_patch_size,
            spatial_merge_size: geometry.spatial_merge_size,
            num_position_embeddings: geometry.num_position_embeddings,
            rope_theta: geometry.rope_theta,
            min_pixels: if geometry.min_pixels > 0 {
                geometry.min_pixels
            } else {
                65_536
            },
            max_pixels: if geometry.max_pixels > 0 {
                geometry.max_pixels
            } else {
                MAX_MERGED_TOKENS.saturating_mul(cell).saturating_mul(cell)
            },
        }
    }

    /// The released configuration, or None for a checkpoint without a vision tower.
    #[must_use]
    pub fn from_config(config: &Qwen4ExpConfig) -> Option<Self> {
        let depth = config.vision_depth?;
        let hidden_size = config.vision_hidden_size?;
        let patch_size = config.vision_patch_size.unwrap_or(16);
        let spatial_merge_size = config.vision_spatial_merge_size.unwrap_or(2);
        let cell = patch_size * spatial_merge_size;
        let tower = Self {
            depth,
            hidden_size,
            num_heads: config.vision_num_heads.unwrap_or(16),
            intermediate_size: config.vision_intermediate_size.unwrap_or(4304),
            out_hidden_size: config.vision_out_hidden_size.unwrap_or(2560),
            patch_size,
            temporal_patch_size: config.vision_temporal_patch_size.unwrap_or(2),
            spatial_merge_size,
            num_position_embeddings: config.vision_num_position_embeddings.unwrap_or(2304),
            rope_theta: config.vision_rope_theta.unwrap_or(10_000.0),
            min_pixels: 65_536,
            max_pixels: MAX_MERGED_TOKENS.saturating_mul(cell).saturating_mul(cell),
        };
        tower.geometry().validate().ok()?;
        Some(tower)
    }

    /// Pixels per side of one merged token (patch * merge).
    #[must_use]
    pub const fn cell(&self) -> usize {
        self.patch_size * self.spatial_merge_size
    }

    /// The learned position grid's side.
    #[must_use]
    pub fn position_side(&self) -> usize {
        (self.num_position_embeddings as f64).sqrt() as usize
    }
}

/// A small deterministic source for the tower's shape tests.
///
/// The released weights are loaded by [Qwen4Vision::load]. This constructor exists so the tower's
/// arithmetic can be exercised without a 132 GiB checkpoint.
#[must_use]
pub fn synthetic_tower(config: &VisionTowerConfig, seed: u64) -> Qwen4Vision {
    let hidden = config.hidden_size;
    let patch_dim = 3 * config.temporal_patch_size * config.patch_size * config.patch_size;
    let intermediate = config.intermediate_size;
    let merged = hidden * config.spatial_merge_size * config.spatial_merge_size;
    let mut state = seed | 1;
    let mut blocks = Vec::with_capacity(config.depth);
    for _ in 0..config.depth {
        blocks.push(synthetic_block(&mut state, hidden, intermediate));
    }
    Qwen4Vision {
        config: *config,
        patch: synthetic_matrix(&mut state, hidden, patch_dim),
        patch_bias: synthetic_vector(&mut state, hidden),
        pos: synthetic_matrix(&mut state, config.num_position_embeddings, hidden),
        blocks,
        merger_norm: synthetic_vector(&mut state, hidden),
        merger_norm_bias: synthetic_vector(&mut state, hidden),
        merger_fc1: synthetic_matrix(&mut state, merged, merged),
        merger_fc1_bias: synthetic_vector(&mut state, merged),
        merger_fc2: synthetic_matrix(&mut state, config.out_hidden_size, merged),
        merger_fc2_bias: synthetic_vector(&mut state, config.out_hidden_size),
    }
}

fn synthetic_draw(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 40) as f32 / 16_777_216.0 - 0.5) * 0.05
}

fn synthetic_matrix(state: &mut u64, rows: usize, columns: usize) -> Mat {
    let mut data = Vec::with_capacity(rows * columns);
    for _ in 0..rows * columns {
        data.push(synthetic_draw(state));
    }
    Mat::new(rows, columns, data)
}

fn synthetic_vector(state: &mut u64, len: usize) -> Vec<f32> {
    (0..len).map(|_| synthetic_draw(state)).collect()
}

fn synthetic_block(state: &mut u64, hidden: usize, intermediate: usize) -> VisionBlock {
    VisionBlock {
        norm1: synthetic_vector(state, hidden),
        norm1_bias: synthetic_vector(state, hidden),
        norm2: synthetic_vector(state, hidden),
        norm2_bias: synthetic_vector(state, hidden),
        qkv: synthetic_matrix(state, 3 * hidden, hidden),
        qkv_bias: synthetic_vector(state, 3 * hidden),
        proj: synthetic_matrix(state, hidden, hidden),
        proj_bias: synthetic_vector(state, hidden),
        fc1: synthetic_matrix(state, intermediate, hidden),
        fc1_bias: synthetic_vector(state, intermediate),
        fc2: synthetic_matrix(state, hidden, intermediate),
        fc2_bias: synthetic_vector(state, hidden),
    }
}

/// One ViT block's weights.
#[derive(Debug, Clone)]
struct VisionBlock {
    norm1: Vec<f32>,
    norm1_bias: Vec<f32>,
    norm2: Vec<f32>,
    norm2_bias: Vec<f32>,
    qkv: Mat,
    qkv_bias: Vec<f32>,
    proj: Mat,
    proj_bias: Vec<f32>,
    fc1: Mat,
    fc1_bias: Vec<f32>,
    fc2: Mat,
    fc2_bias: Vec<f32>,
}

/// The Qwen vision tower.
#[derive(Debug, Clone)]
pub struct Qwen4Vision {
    config: VisionTowerConfig,
    patch: Mat,
    patch_bias: Vec<f32>,
    pos: Mat,
    blocks: Vec<VisionBlock>,
    merger_norm: Vec<f32>,
    merger_norm_bias: Vec<f32>,
    merger_fc1: Mat,
    merger_fc1_bias: Vec<f32>,
    merger_fc2: Mat,
    merger_fc2_bias: Vec<f32>,
}

impl Qwen4Vision {
    /// The tower's geometry.
    #[must_use]
    pub fn config(&self) -> &VisionTowerConfig {
        &self.config
    }

    /// Load the tower from a checkpoint directory.
    ///
    /// # Errors
    /// When the configuration has no vision tower, a tensor is missing, or its byte length does
    /// not match the planned BF16 shape.
    pub fn load(directory: &std::path::Path, config: &Qwen4ExpConfig) -> Result<Self, String> {
        let tower = VisionTowerConfig::from_config(config)
            .ok_or_else(|| "the configuration has no vision tower".to_owned())?;
        tower
            .geometry()
            .validate()
            .map_err(|error| error.to_string())?;
        let source =
            SafeTensorsSource::open(directory, &directory.join(crate::checkpoint::INDEX_NAME))
                .map_err(|error| error.to_string())?;
        let prefix = "model.visual";
        let hidden = tower.hidden_size;
        let patch_dim = 3 * tower.temporal_patch_size * tower.patch_size * tower.patch_size;
        let merged = hidden * tower.spatial_merge_size * tower.spatial_merge_size;
        let matrix =
            |name: String, rows: usize, columns: usize| load_bf16(&source, &name, rows, columns);
        let vector = |name: String, len: usize| load_bf16(&source, &name, 1, len).map(|m| m.data);
        let mut blocks = Vec::with_capacity(tower.depth);
        for block in 0..tower.depth {
            let p = format!("{prefix}.blocks.{block}");
            blocks.push(VisionBlock {
                norm1: vector(format!("{p}.norm1.weight"), hidden)?,
                norm1_bias: vector(format!("{p}.norm1.bias"), hidden)?,
                norm2: vector(format!("{p}.norm2.weight"), hidden)?,
                norm2_bias: vector(format!("{p}.norm2.bias"), hidden)?,
                qkv: matrix(format!("{p}.attn.qkv.weight"), 3 * hidden, hidden)?,
                qkv_bias: vector(format!("{p}.attn.qkv.bias"), 3 * hidden)?,
                proj: matrix(format!("{p}.attn.proj.weight"), hidden, hidden)?,
                proj_bias: vector(format!("{p}.attn.proj.bias"), hidden)?,
                fc1: matrix(
                    format!("{p}.mlp.linear_fc1.weight"),
                    tower.intermediate_size,
                    hidden,
                )?,
                fc1_bias: vector(format!("{p}.mlp.linear_fc1.bias"), tower.intermediate_size)?,
                fc2: matrix(
                    format!("{p}.mlp.linear_fc2.weight"),
                    hidden,
                    tower.intermediate_size,
                )?,
                fc2_bias: vector(format!("{p}.mlp.linear_fc2.bias"), hidden)?,
            });
        }
        Ok(Self {
            config: tower,
            patch: matrix(
                format!("{prefix}.patch_embed.proj.weight"),
                hidden,
                patch_dim,
            )?,
            patch_bias: vector(format!("{prefix}.patch_embed.proj.bias"), hidden)?,
            pos: matrix(
                format!("{prefix}.pos_embed.weight"),
                tower.num_position_embeddings,
                hidden,
            )?,
            blocks,
            merger_norm: vector(format!("{prefix}.merger.norm.weight"), hidden)?,
            merger_norm_bias: vector(format!("{prefix}.merger.norm.bias"), hidden)?,
            merger_fc1: matrix(format!("{prefix}.merger.linear_fc1.weight"), merged, merged)?,
            merger_fc1_bias: vector(format!("{prefix}.merger.linear_fc1.bias"), merged)?,
            merger_fc2: matrix(
                format!("{prefix}.merger.linear_fc2.weight"),
                tower.out_hidden_size,
                merged,
            )?,
            merger_fc2_bias: vector(
                format!("{prefix}.merger.linear_fc2.bias"),
                tower.out_hidden_size,
            )?,
        })
    }

    /// Encode one prepared canvas into merged image rows.
    ///
    /// "pixels" is [Qwen4Vision::patchify]'s output: one row of
    /// 3 * temporal * patch * patch normalised samples per ViT patch, in merge-block order.
    /// The result has one row of "out_hidden_size" per merged token, in the same order.
    ///
    /// # Errors
    /// When the canvas and grid disagree, or a shape does not divide into the merge grid.
    pub fn encode(&self, pixels: &[f32], grid_h: usize, grid_w: usize) -> Result<Vec<f32>, String> {
        let tower = &self.config;
        let hidden = tower.hidden_size;
        let patch_dim = 3 * tower.temporal_patch_size * tower.patch_size * tower.patch_size;
        if !grid_h.is_multiple_of(tower.spatial_merge_size)
            || !grid_w.is_multiple_of(tower.spatial_merge_size)
        {
            return Err(format!(
                "a {grid_h}x{grid_w} patch grid does not divide into {}x{} merge blocks",
                tower.spatial_merge_size, tower.spatial_merge_size
            ));
        }
        let patches = grid_h * grid_w;
        let merged = patches / (tower.spatial_merge_size * tower.spatial_merge_size);
        if merged > MAX_MERGED_TOKENS {
            return Err(format!(
                "a {grid_h}x{grid_w} patch grid makes {merged} merged image tokens, over the host tower's {MAX_MERGED_TOKENS}-token cap; shrink the image"
            ));
        }
        if pixels.len() != patches * patch_dim {
            return Err(format!(
                "the canvas has {} samples, expected {patches} patches of {patch_dim}",
                pixels.len()
            ));
        }
        let mut hidden_states = vec![0.0f32; patches * hidden];
        for patch in 0..patches {
            let input = &pixels[patch * patch_dim..(patch + 1) * patch_dim];
            let output = &mut hidden_states[patch * hidden..(patch + 1) * hidden];
            for (row, slot) in output.iter_mut().enumerate() {
                let base = row * patch_dim;
                let mut sum = self.patch_bias[row];
                for (value, activation) in self.patch.data[base..base + patch_dim].iter().zip(input)
                {
                    sum += value * activation;
                }
                *slot = sum;
            }
        }
        // Learned position embeddings, bilinearly resampled to this patch grid.
        let side = tower.position_side();
        let merge = tower.spatial_merge_size;
        let blocks_w = grid_w / merge;
        for patch in 0..patches {
            let in_col = patch % merge;
            let in_row = (patch / merge) % merge;
            let block_col = (patch / (merge * merge)) % blocks_w;
            let block_row = patch / (merge * merge * blocks_w);
            let row = block_row * merge + in_row;
            let col = block_col * merge + in_col;
            let (h_taps, h_weights) = axis_taps(row, grid_h, side);
            let (w_taps, w_weights) = axis_taps(col, grid_w, side);
            let output = &mut hidden_states[patch * hidden..(patch + 1) * hidden];
            for (h_index, h_weight) in h_taps.iter().zip(&h_weights) {
                for (w_index, w_weight) in w_taps.iter().zip(&w_weights) {
                    let weight = h_weight * w_weight;
                    let base = (h_index * side + w_index) * hidden;
                    for (column, slot) in output.iter_mut().enumerate() {
                        *slot += weight * self.pos.data[base + column];
                    }
                }
            }
        }
        let (cos, sin) = self.rope_tables(grid_h, grid_w);
        let head_dim = hidden / tower.num_heads;
        let scaling = 1.0 / (head_dim as f32).sqrt();
        let mut scores = vec![0.0f32; patches * patches];
        let mut qkv = vec![0.0f32; patches * 3 * hidden];
        for block in &self.blocks {
            let mut normed = hidden_states.clone();
            layer_norm(
                &mut normed,
                &block.norm1,
                &block.norm1_bias,
                hidden,
                VISION_NORM_EPS,
            )
            .map_err(|error| error.to_string())?;
            for patch in 0..patches {
                linear_into(
                    &block.qkv,
                    &block.qkv_bias,
                    &normed[patch * hidden..(patch + 1) * hidden],
                    &mut qkv[patch * 3 * hidden..(patch + 1) * 3 * hidden],
                );
            }
            for patch in 0..patches {
                for head in 0..tower.num_heads {
                    let q_base = patch * 3 * hidden + head * head_dim;
                    apply_rope(
                        &mut qkv[q_base..q_base + head_dim],
                        &cos[patch * head_dim..(patch + 1) * head_dim],
                        &sin[patch * head_dim..(patch + 1) * head_dim],
                    );
                    let k_base = patch * 3 * hidden + hidden + head * head_dim;
                    apply_rope(
                        &mut qkv[k_base..k_base + head_dim],
                        &cos[patch * head_dim..(patch + 1) * head_dim],
                        &sin[patch * head_dim..(patch + 1) * head_dim],
                    );
                }
            }
            let mut attended = vec![0.0f32; patches * hidden];
            for head in 0..tower.num_heads {
                for query in 0..patches {
                    let q_base = query * 3 * hidden + head * head_dim;
                    let q = &qkv[q_base..q_base + head_dim];
                    for key in 0..patches {
                        let k_base = key * 3 * hidden + hidden + head * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[d] * qkv[k_base + d];
                        }
                        scores[query * patches + key] = dot * scaling;
                    }
                    softmax_in_place(&mut scores[query * patches..(query + 1) * patches]);
                    let out_base = query * hidden + head * head_dim;
                    for key in 0..patches {
                        let weight = scores[query * patches + key];
                        let v_base = key * 3 * hidden + 2 * hidden + head * head_dim;
                        for d in 0..head_dim {
                            attended[out_base + d] += weight * qkv[v_base + d];
                        }
                    }
                }
            }
            let mut projected = vec![0.0f32; patches * hidden];
            for patch in 0..patches {
                linear_into(
                    &block.proj,
                    &block.proj_bias,
                    &attended[patch * hidden..(patch + 1) * hidden],
                    &mut projected[patch * hidden..(patch + 1) * hidden],
                );
                for column in 0..hidden {
                    hidden_states[patch * hidden + column] += projected[patch * hidden + column];
                }
            }
            normed.copy_from_slice(&hidden_states);
            layer_norm(
                &mut normed,
                &block.norm2,
                &block.norm2_bias,
                hidden,
                VISION_NORM_EPS,
            )
            .map_err(|error| error.to_string())?;
            let mut mlp = vec![0.0f32; patches * hidden];
            let mut intermediate = vec![0.0f32; tower.intermediate_size];
            for patch in 0..patches {
                linear_into(
                    &block.fc1,
                    &block.fc1_bias,
                    &normed[patch * hidden..(patch + 1) * hidden],
                    &mut intermediate,
                );
                for value in &mut intermediate {
                    *value = gelu_tanh(*value);
                }
                linear_into(
                    &block.fc2,
                    &block.fc2_bias,
                    &intermediate,
                    &mut mlp[patch * hidden..(patch + 1) * hidden],
                );
                for column in 0..hidden {
                    hidden_states[patch * hidden + column] += mlp[patch * hidden + column];
                }
            }
        }
        // The merger normalises each patch, concatenates each 2x2 block, then projects.
        let merged = hidden * merge * merge;
        let merged_rows = patches / (merge * merge);
        let mut normed = hidden_states.clone();
        layer_norm(
            &mut normed,
            &self.merger_norm,
            &self.merger_norm_bias,
            hidden,
            VISION_NORM_EPS,
        )
        .map_err(|error| error.to_string())?;
        let mut output = vec![0.0f32; merged_rows * tower.out_hidden_size];
        let mut first = vec![0.0f32; merged];
        let mut second = vec![0.0f32; merged];
        for row in 0..merged_rows {
            first.copy_from_slice(&normed[row * merged..(row + 1) * merged]);
            linear_into(&self.merger_fc1, &self.merger_fc1_bias, &first, &mut second);
            for value in &mut second {
                *value = gelu_erf(*value);
            }
            linear_into(
                &self.merger_fc2,
                &self.merger_fc2_bias,
                &second,
                &mut output[row * tower.out_hidden_size..(row + 1) * tower.out_hidden_size],
            );
        }
        Ok(output)
    }

    fn rope_tables(&self, grid_h: usize, grid_w: usize) -> (Vec<f32>, Vec<f32>) {
        let head_dim = self.config.hidden_size / self.config.num_heads;
        let spatial = head_dim / 2;
        let inv_freq: Vec<f32> = (0..spatial)
            .step_by(2)
            .map(|index| (1.0 / self.config.rope_theta.powf(index as f64 / spatial as f64)) as f32)
            .collect();
        let patches = grid_h * grid_w;
        let mut cos = vec![0.0f32; patches * head_dim];
        let mut sin = vec![0.0f32; patches * head_dim];
        let merge = self.config.spatial_merge_size;
        let blocks_w = grid_w / merge;
        for patch in 0..patches {
            let in_col = patch % merge;
            let in_row = (patch / merge) % merge;
            let block_col = (patch / (merge * merge)) % blocks_w;
            let block_row = patch / (merge * merge * blocks_w);
            let row = block_row * merge + in_row;
            let col = block_col * merge + in_col;
            let half = spatial / 2;
            for (index, freq) in inv_freq.iter().enumerate() {
                let h = row as f32 * freq;
                let w = col as f32 * freq;
                // [h(18), w(18), h(18), w(18)] after the reference recomposition.
                cos[patch * head_dim + index] = h.cos();
                cos[patch * head_dim + half + index] = w.cos();
                cos[patch * head_dim + 2 * half + index] = h.cos();
                cos[patch * head_dim + 3 * half + index] = w.cos();
                sin[patch * head_dim + index] = h.sin();
                sin[patch * head_dim + half + index] = w.sin();
                sin[patch * head_dim + 2 * half + index] = h.sin();
                sin[patch * head_dim + 3 * half + index] = w.sin();
            }
        }
        (cos, sin)
    }
}

fn load_bf16(
    source: &dyn WeightSource,
    name: &str,
    rows: usize,
    columns: usize,
) -> Result<Mat, String> {
    let bytes = source.read(name).map_err(|error| error.to_string())?;
    let expected = rows * columns * 2;
    if bytes.len() != expected {
        return Err(format!(
            "{name} is {} bytes, expected {expected} for BF16 [{rows}, {columns}]",
            bytes.len()
        ));
    }
    let data = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| bf16::from_le_bytes(*pair).to_f32())
        .collect();
    Ok(Mat::new(rows, columns, data))
}

fn linear_into(matrix: &Mat, bias: &[f32], input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), matrix.columns);
    debug_assert_eq!(output.len(), matrix.rows);
    for row in 0..matrix.rows {
        let base = row * matrix.columns;
        let mut sum = bias[row];
        for (value, activation) in matrix.data[base..base + matrix.columns].iter().zip(input) {
            sum += value * activation;
        }
        output[row] = sum;
    }
}

fn apply_rope(value: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = value.len() / 2;
    let mut rotated = vec![0.0f32; value.len()];
    for index in 0..half {
        rotated[index] = -value[half + index];
        rotated[half + index] = value[index];
    }
    for index in 0..value.len() {
        value[index] = value[index] * cos[index] + rotated[index] * sin[index];
    }
}

fn softmax_in_place(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in row.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum > 0.0 {
        for value in row.iter_mut() {
            *value /= sum;
        }
    }
}

fn gelu_tanh(value: f32) -> f32 {
    0.5 * value
        * (1.0
            + ((2.0 / std::f32::consts::PI).sqrt() * (value + 0.044_715 * value * value * value))
                .tanh())
}

fn gelu_erf(value: f32) -> f32 {
    0.5 * value * (1.0 + libm::erf(value as f64 / std::f64::consts::SQRT_2) as f32)
}

/// The two taps and weights for one axis of a bilinear resample with align_corners=true.
fn axis_taps(index: usize, size: usize, side: usize) -> ([usize; 2], [f32; 2]) {
    let src = if size <= 1 {
        0.0
    } else {
        index as f64 * (side - 1) as f64 / (size - 1) as f64
    };
    let floor = src.floor();
    let raw = [floor as isize, floor as isize + 1];
    let taps = [
        raw[0].clamp(0, side as isize - 1) as usize,
        raw[1].clamp(0, side as isize - 1) as usize,
    ];
    let mut weights = [0.0f32; 2];
    for (slot, offset) in raw.iter().enumerate() {
        weights[slot] = (1.0 - (src - *offset as f64).abs()).max(0.0) as f32;
    }
    (taps, weights)
}

/// "smart_resize" from the released Qwen2-VL image processor.
///
/// # Errors
/// When the image has a zero side or an aspect ratio over 200.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), String> {
    if height == 0 || width == 0 {
        return Err("the image has a zero width or height".to_owned());
    }
    let ratio = height.max(width) as f64 / height.min(width) as f64;
    if ratio > 200.0 {
        return Err(format!(
            "absolute aspect ratio must be smaller than 200, got {ratio}"
        ));
    }
    let mut h_bar = round_half_even(height as f64 / factor as f64) as usize * factor;
    let mut w_bar = round_half_even(width as f64 / factor as f64) as usize * factor;
    if h_bar * w_bar > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h_bar = factor.max((height as f64 / beta / factor as f64).floor() as usize * factor);
        w_bar = factor.max((width as f64 / beta / factor as f64).floor() as usize * factor);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h_bar = (height as f64 * beta / factor as f64).ceil() as usize * factor;
        w_bar = (width as f64 * beta / factor as f64).ceil() as usize * factor;
    }
    Ok((h_bar, w_bar))
}

fn round_half_even(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if (fraction - 0.5).abs() < f64::EPSILON {
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        value.round()
    }
}

/// Turn a prepared canvas into the normalised patch rows the tower reads.
#[must_use]
pub fn patchify(
    pixels: &[u8],
    width: usize,
    height: usize,
    config: &VisionTowerConfig,
) -> Vec<f32> {
    let patch = config.patch_size;
    let merge = config.spatial_merge_size;
    let temporal = config.temporal_patch_size;
    let grid_h = height / patch;
    let grid_w = width / patch;
    let blocks_w = grid_w / merge;
    let mut out = Vec::with_capacity(grid_h * grid_w * 3 * temporal * patch * patch);
    for block_row in 0..grid_h / merge {
        for block_col in 0..blocks_w {
            for in_row in 0..merge {
                for in_col in 0..merge {
                    let row = block_row * merge + in_row;
                    let col = block_col * merge + in_col;
                    for channel in 0..3 {
                        for _ in 0..temporal {
                            for py in 0..patch {
                                for px in 0..patch {
                                    let y = row * patch + py;
                                    let x = col * patch + px;
                                    let sample = pixels[(y * width + x) * 3 + channel];
                                    out.push(sample as f32 / 127.5 - 1.0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// The prompt span and image preparation for the Qwen vision path.
#[derive(Debug, Clone)]
pub struct QwenVisionCodec {
    config: VisionTowerConfig,
    image_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
}

impl QwenVisionCodec {
    /// A codec over the released geometry and token ids.
    #[must_use]
    pub fn new(
        config: VisionTowerConfig,
        image_token_id: u32,
        vision_start_token_id: u32,
        vision_end_token_id: u32,
    ) -> Self {
        Self {
            config,
            image_token_id,
            vision_start_token_id,
            vision_end_token_id,
        }
    }

    /// The codec a checkpoint's shared architecture names, or None without a vision tower.
    ///
    /// # Errors
    /// When the checkpoint carries a tower but omits any of the three token ids the template
    /// inserts; a defaulted id would decode to a different token than the one the model expects.
    pub fn from_architecture(
        architecture: &sparkflash_model::Architecture,
    ) -> Result<Option<Self>, String> {
        let Some(geometry) = architecture.vision_geometry() else {
            return Ok(None);
        };
        // A tower whose geometry the kernels cannot run is a refusal, not a text-only model.
        geometry.validate().map_err(|error| error.to_string())?;
        let tower = VisionTowerConfig::from_geometry(&geometry);
        let tokens = &architecture.tokens;
        let image_token_id = tokens
            .image_token_id
            .ok_or("the checkpoint carries a vision tower but names no image_token_id")?;
        let vision_start_token_id = tokens
            .vision_start_token_id
            .ok_or("the checkpoint carries a vision tower but names no vision_start_token_id")?;
        let vision_end_token_id = tokens
            .vision_end_token_id
            .ok_or("the checkpoint carries a vision tower but names no vision_end_token_id")?;
        Ok(Some(Self::new(
            tower,
            image_token_id,
            vision_start_token_id,
            vision_end_token_id,
        )))
    }

    /// The codec a checkpoint configuration names, or None without a vision tower.
    ///
    /// # Errors
    /// A vision geometry the kernels cannot run.
    pub fn from_config(config: &Qwen4ExpConfig) -> Result<Option<Self>, String> {
        let Some(tower) = VisionTowerConfig::from_config(config) else {
            return Ok(None);
        };
        tower
            .geometry()
            .validate()
            .map_err(|error| error.to_string())?;
        Ok(Some(Self::new(
            tower,
            config.image_token_id,
            config.vision_start_token_id,
            config.vision_end_token_id,
        )))
    }

    /// A codec for tests: the same span rules at a small pixel budget.
    #[must_use]
    pub fn tiny(patch_size: usize, merge: usize, min_pixels: usize, max_pixels: usize) -> Self {
        Self {
            config: VisionTowerConfig {
                depth: 1,
                hidden_size: 8,
                num_heads: 2,
                intermediate_size: 16,
                out_hidden_size: 8,
                patch_size,
                temporal_patch_size: 2,
                spatial_merge_size: merge,
                num_position_embeddings: 16,
                rope_theta: 10_000.0,
                min_pixels,
                max_pixels,
            },
            image_token_id: 248_056,
            vision_start_token_id: 248_053,
            vision_end_token_id: 248_054,
        }
    }

    /// The tower's geometry.
    #[must_use]
    pub fn config(&self) -> &VisionTowerConfig {
        &self.config
    }
}

/// The content key for one prepared canvas.
///
/// The key names the model family, the canvas geometry and the pixels. Hashing only the pixel
/// bytes lets two canvases with the same samples but different widths collide, and leaves Qwen and
/// V4.1 keys in one namespace. The prefix cache compares keys across requests, so both separations
/// matter.
fn image_key(
    pixels: &[u8],
    width: usize,
    height: usize,
    n_llm_h: usize,
    n_llm_w: usize,
    cell: usize,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"qwen3.8-flash-next-nvfp4/vision/v1\0");
    for value in [width, height, n_llm_h, n_llm_w, cell] {
        hasher.update(u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes());
    }
    hasher.update(pixels);
    hasher.finalize().into()
}

impl VisionCodec for QwenVisionCodec {
    fn image_token_id(&self) -> u32 {
        self.image_token_id
    }

    fn prepare(&self, bytes: &[u8], options: &VisionOptions) -> Result<PreparedImage, VisionError> {
        let limits = sparkflash_vision::DecodeLimits {
            max_bytes: options.max_bytes,
            max_dimension: options.max_dimension,
        };
        // Qwen's processor uses Pillow's `convert("RGB")`, which drops an alpha channel rather
        // than compositing it onto a background. The shared decoder's default is the serving
        // recipe's blend, so ask for the Qwen policy explicitly.
        let mut image =
            sparkflash_vision::decode::decode_with_alpha(bytes, &limits, AlphaPolicy::Drop)
                .map_err(|error| VisionError::new(error.to_string()))?;
        if let Some(long_side) = options.max_long_side {
            let longest = image.width.max(image.height);
            if longest > long_side as usize {
                let scale = long_side as f64 / longest as f64;
                let width = ((image.width as f64 * scale).round() as usize).max(1);
                let height = ((image.height as f64 * scale).round() as usize).max(1);
                image = sparkflash_vision::resize_bicubic(&image, width, height);
            }
        }
        let (height, width) = smart_resize(
            image.height,
            image.width,
            self.config.cell(),
            self.config.min_pixels,
            self.config.max_pixels,
        )
        .map_err(VisionError::new)?;
        let resized = sparkflash_vision::resize_bicubic(&image, width, height);
        let cell = self.config.cell();
        let key = image_key(
            &resized.data,
            width,
            height,
            height / cell,
            width / cell,
            cell,
        );
        let grid = ImageGrid {
            n_llm_h: height / cell,
            n_llm_w: width / cell,
            best_height: height,
            best_width: width,
            line_stride: width / cell,
        };
        Ok(PreparedImage {
            grid,
            pixels: Arc::from(resized.data),
            key,
        })
    }

    fn expand(
        &self,
        prompt: &[u32],
        images: &[PreparedImage],
    ) -> Result<(Vec<u32>, Vec<ImageSpan>), VisionError> {
        let placeholders = prompt
            .iter()
            .filter(|id| **id == self.image_token_id())
            .count();
        if placeholders != images.len() {
            return Err(VisionError::new(format!(
                "the prompt has {placeholders} image placeholders but the request has {} images",
                images.len()
            )));
        }
        let mut ids = Vec::with_capacity(prompt.len());
        let mut spans = Vec::with_capacity(images.len());
        let mut next = 0usize;
        for &token in prompt {
            if token != self.image_token_id() {
                ids.push(token);
                continue;
            }
            let image = &images[next];
            let grid = image.grid;
            let start = ids.len();
            ids.push(self.vision_start_token_id);
            ids.extend(std::iter::repeat_n(self.image_token_id, grid.image_rows()));
            ids.push(self.vision_end_token_id);
            spans.push(ImageSpan {
                start,
                grid,
                key: image.key,
            });
            next += 1;
        }
        Ok((ids, spans))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_the_released_processor() {
        assert_eq!(
            smart_resize(100, 100, 32, 65_536, 16_777_216).unwrap(),
            (256, 256)
        );
        assert_eq!(
            smart_resize(1000, 500, 32, 65_536, 16_777_216).unwrap(),
            (992, 512)
        );
        assert!(smart_resize(1, 1000, 32, 65_536, 16_777_216).is_err());
    }

    #[test]
    fn the_bilinear_taps_hit_the_endpoints() {
        let (taps, weights) = axis_taps(0, 48, 48);
        assert_eq!(taps, [0, 1]);
        assert_eq!(weights, [1.0, 0.0]);
        let (taps, weights) = axis_taps(47, 48, 48);
        assert_eq!(taps, [47, 47]);
        assert_eq!(weights, [1.0, 0.0]);
        let (taps, weights) = axis_taps(23, 48, 48);
        assert_eq!(taps, [23, 24]);
        assert_eq!(weights, [1.0, 0.0]);
    }

    #[test]
    fn the_tower_produces_one_row_per_merged_patch() {
        let config = VisionTowerConfig {
            depth: 1,
            hidden_size: 8,
            num_heads: 2,
            intermediate_size: 16,
            out_hidden_size: 8,
            patch_size: 2,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            num_position_embeddings: 16,
            rope_theta: 10_000.0,
            min_pixels: 64,
            max_pixels: 1024,
        };
        let vision = synthetic_tower(&config, 7);
        let grid_h = 4;
        let grid_w = 4;
        let patch_dim = 3 * 2 * 2 * 2;
        let pixels: Vec<f32> = (0..grid_h * grid_w * patch_dim)
            .map(|index| (index as f32 * 0.01).sin())
            .collect();
        let first = vision.encode(&pixels, grid_h, grid_w).expect("encodes");
        let second = vision.encode(&pixels, grid_h, grid_w).expect("encodes");
        assert_eq!(first, second);
        assert_eq!(first.len(), (grid_h * grid_w / 4) * 8);
        assert!(first.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn the_tower_refuses_a_grid_over_the_merged_token_cap() {
        let config = VisionTowerConfig {
            depth: 1,
            hidden_size: 8,
            num_heads: 2,
            intermediate_size: 16,
            out_hidden_size: 8,
            patch_size: 2,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            num_position_embeddings: 16,
            rope_theta: 10_000.0,
            min_pixels: 64,
            max_pixels: usize::MAX,
        };
        let vision = synthetic_tower(&config, 7);
        // 50 x 50 patches is 625 merged tokens, one past the cap; the cap check runs before the
        // pixel-length check, so an empty canvas is enough to prove it.
        let error = vision
            .encode(&[], 50, 50)
            .expect_err("the cap must refuse the grid");
        assert!(error.contains("cap"), "{error}");
        assert_eq!(MAX_MERGED_TOKENS, 576);
    }

    #[test]
    fn the_image_key_separates_geometry_and_family() {
        let pixels = b"same pixels";
        let first = image_key(pixels, 8, 4, 2, 4, 2);
        let second = image_key(pixels, 4, 8, 4, 2, 2);
        assert_ne!(first, second, "the same bytes at another geometry collide");
        let third = image_key(pixels, 8, 4, 2, 4, 4);
        assert_ne!(first, third, "the merge cell is part of the key");
    }

    #[test]
    fn a_span_has_no_newlines_and_counts_the_image_rows() {
        let codec = QwenVisionCodec::tiny(2, 2, 64, 1024);
        let image = PreparedImage {
            grid: ImageGrid {
                n_llm_h: 2,
                n_llm_w: 3,
                best_height: 12,
                best_width: 12,
                line_stride: 3,
            },
            pixels: vec![0; 12 * 12 * 3].into(),
            key: [0; 32],
        };
        let (ids, spans) = codec.expand(&[1, 248_056, 2], &[image]).unwrap();
        assert_eq!(spans[0].len(), 2 * 3 + 2);
        assert_eq!(ids.len(), 2 + 8);
        assert_eq!(ids[1], 248_053);
        assert_eq!(ids[ids.len() - 2], 248_054);
        assert_eq!(*ids.last().unwrap(), 2);
        assert_eq!(
            spans[0].row(spans[0].start + 1),
            sparkflash_model::vision::SpanRow::Image(0)
        );
        assert_eq!(
            spans[0].row(spans[0].start + 4),
            sparkflash_model::vision::SpanRow::Image(3)
        );
    }
}

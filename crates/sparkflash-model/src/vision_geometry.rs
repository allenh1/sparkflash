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

//! The vision tower's geometry, described once.
//!
//! Two arms carry a ViT in their checkpoint and each described it in its own crate: Qwen3.8-Flash-
//! Next's `VisionTowerConfig` (depth 27, width 1,152, heads 16, MLP 4,304, merger 2,560, patch 16,
//! temporal 2, merge 2, 2,304 positions, RoPE 10,000) and GLM-5.3-Flash's `VisionConfig` (24,
//! 1,024, 16, 4,096, 4,096, patch 14, temporal 2, merge 2). They are the same description with
//! different numbers, which is the shape [D69](../../../docs/DECISIONS.md) says belongs in this
//! crate: a part's geometry is data, and the arm that uses it maps its own configuration onto it.
//!
//! Nothing here runs a tower. The device kernels are one shared image
//! (`kernels/opt_vision_sm121.cu`) and the orchestration is [D84]'s next extraction; this type is
//! what both will read, so a tower's shape cannot differ between the arm that loads it and the
//! kernel sequence that runs it.
//!
//! [D84]: ../../../docs/DECISIONS.md

use crate::config::ConfigError;

/// One ViT's geometry, as both released checkpoints describe theirs.
///
/// A field an arm's configuration does not carry keeps the value here, which is why the two arms
/// can share the type without either losing information: GLM-5.3-Flash names no position grid or
/// RoPE base, and Qwen3.8-Flash-Next names no merger width, so each leaves the other's field at
/// its declared default.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionGeometry {
    /// Transformer blocks.
    pub depth: usize,
    /// Block width.
    pub hidden_size: usize,
    /// Attention heads.
    pub num_heads: usize,
    /// Block MLP width.
    pub intermediate_size: usize,
    /// The width the merger hands the language model.
    pub out_hidden_size: usize,
    /// The merger MLP's width.
    pub projection_intermediate_size: usize,
    /// Image channels (3 for every release here).
    pub in_channels: usize,
    /// Patch side in pixels.
    pub patch_size: usize,
    /// Frames one patch covers; a still image duplicates its frame.
    pub temporal_patch_size: usize,
    /// Spatial merge factor before the merger.
    pub spatial_merge_size: usize,
    /// Learned position-grid entries (a square grid), or 0 when the tower uses RoPE only.
    pub num_position_embeddings: usize,
    /// Axial RoPE base.
    pub rope_theta: f64,
    /// The smallest image the tower plans, in pixels.
    pub min_pixels: usize,
    /// The largest image the tower plans, in pixels.
    pub max_pixels: usize,
}

impl VisionGeometry {
    /// Parse a `vision_config` structurally, without matching the producer's `model_type`.
    ///
    /// A new tower composed of the same parts names the same structural keys; matching a literal
    /// `model_type` string forced a new arm for every release. Only the numbers are read here.
    ///
    /// # Errors
    /// A key that is present but not a non-negative number, or a missing `patch_size` (a
    /// merge cell of zero would plan every image as empty). `out_hidden_size` defaults to the
    /// language model's width (`language_hidden`), because a tower that names no merger output
    /// hands the language model its own width rather than the tower's block width.
    /// `spatial_merge_size` falls back to `downsample_ratio`, then to 1.
    pub fn from_value(
        config: &serde_json::Value,
        language_hidden: usize,
    ) -> Result<Self, ConfigError> {
        let number = |name: &str| -> Result<Option<usize>, ConfigError> {
            match config.get(name) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(value) => value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .map(Some)
                    .ok_or_else(|| {
                        ConfigError::Invalid(format!("vision_config.{name} is not an integer"))
                    }),
            }
        };
        let float = |name: &str| -> Result<Option<f64>, ConfigError> {
            match config.get(name) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(value) => value.as_f64().map(Some).ok_or_else(|| {
                    ConfigError::Invalid(format!("vision_config.{name} is not a number"))
                }),
            }
        };
        let depth = number("num_hidden_layers")?
            .or(number("depth")?)
            .ok_or_else(|| ConfigError::Invalid("vision_config names no depth".to_owned()))?;
        let hidden_size = number("hidden_size")?
            .ok_or_else(|| ConfigError::Invalid("vision_config names no hidden_size".to_owned()))?;
        let num_heads = number("num_attention_heads")?
            .or(number("num_heads")?)
            .ok_or_else(|| {
                ConfigError::Invalid("vision_config names no attention heads".to_owned())
            })?;
        let intermediate_size = number("intermediate_size")?.ok_or_else(|| {
            ConfigError::Invalid("vision_config names no intermediate_size".to_owned())
        })?;
        Ok(Self {
            depth,
            hidden_size,
            num_heads,
            intermediate_size,
            // The merger hands the language model its own width; a tower that does not name one
            // hands it the language width, not its own block width.
            out_hidden_size: number("out_hidden_size")?.unwrap_or(language_hidden),
            projection_intermediate_size: number("projection_intermediate_size")?.unwrap_or(0),
            in_channels: number("in_channels")?.unwrap_or(3),
            // A patch side is geometry, not a default: a missing value would make the merge cell
            // zero and every planned image empty. An arm that carries a default states it.
            patch_size: number("patch_size")?.ok_or_else(|| {
                ConfigError::Invalid("vision_config names no patch_size".to_owned())
            })?,
            temporal_patch_size: number("temporal_patch_size")?.unwrap_or(1),
            // DeepSeek names the same factor `downsample_ratio`.
            spatial_merge_size: number("spatial_merge_size")?
                .or(number("downsample_ratio")?)
                .unwrap_or(1),
            num_position_embeddings: number("num_position_embeddings")?.unwrap_or(0),
            rope_theta: float("rope_theta")?.unwrap_or(10_000.0),
            min_pixels: number("min_pixels")?.unwrap_or(0),
            max_pixels: number("max_pixels")?.unwrap_or(0),
        })
    }

    /// The same geometry from the architecture parser's `Vision` summary.
    #[must_use]
    pub fn from_vision(vision: &crate::architecture::Vision) -> Self {
        Self {
            depth: vision.depth,
            hidden_size: vision.hidden_size,
            num_heads: vision.heads,
            intermediate_size: vision.intermediate,
            out_hidden_size: vision.out_hidden_size,
            projection_intermediate_size: vision.projection_intermediate,
            in_channels: vision.in_channels,
            patch_size: vision.patch_size,
            temporal_patch_size: vision.temporal_patch_size,
            spatial_merge_size: vision.spatial_merge,
            num_position_embeddings: vision.num_position_embeddings,
            rope_theta: vision.rope_theta,
            min_pixels: vision.min_pixels,
            max_pixels: vision.max_pixels,
        }
    }

    /// The width of one attention head.
    ///
    /// # Panics
    /// Never: a geometry whose head count is zero is refused by [Self::validate] before any
    /// arithmetic reads it, and this method is only reached through one.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size.checked_div(self.num_heads).unwrap_or(0)
    }

    /// The elements one patch contributes before the first projection.
    #[must_use]
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    /// The side of the learned position grid, when the tower has one.
    ///
    /// # Panics
    /// Never: [Self::validate] refuses a grid that is not a perfect square.
    #[must_use]
    pub fn position_side(&self) -> usize {
        (self.num_position_embeddings as f64).sqrt() as usize
    }

    /// The merged rows one grid of `rows` by `cols` patches becomes.
    #[must_use]
    pub fn merged_rows(&self, rows: usize, cols: usize) -> usize {
        (rows / self.spatial_merge_size) * (cols / self.spatial_merge_size)
    }

    /// Refuses a geometry the kernels cannot run.
    ///
    /// # Errors
    /// A zero in any dimension the kernels divide by, a width that is not a multiple of its head
    /// count, a position grid that is not a perfect square, or a pixel window that is inverted.
    pub fn validate(&self) -> Result<(), ConfigError> {
        for (name, value) in [
            ("depth", self.depth),
            ("hidden_size", self.hidden_size),
            ("num_heads", self.num_heads),
            ("intermediate_size", self.intermediate_size),
            ("out_hidden_size", self.out_hidden_size),
            ("in_channels", self.in_channels),
            ("patch_size", self.patch_size),
            ("temporal_patch_size", self.temporal_patch_size),
            ("spatial_merge_size", self.spatial_merge_size),
        ] {
            if value == 0 {
                return Err(ConfigError::Invalid(format!(
                    "the vision tower's {name} is zero"
                )));
            }
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(ConfigError::Invalid(format!(
                "the vision tower's width {} is not a multiple of its {} heads",
                self.hidden_size, self.num_heads
            )));
        }
        if self.num_position_embeddings > 0
            && self.position_side().pow(2) != self.num_position_embeddings
        {
            return Err(ConfigError::Invalid(format!(
                "the vision tower's {} position embeddings are not a square grid",
                self.num_position_embeddings
            )));
        }
        // A zero maximum is "the arm's own cap", not an inverted window.
        if self.max_pixels > 0 && self.min_pixels > self.max_pixels {
            return Err(ConfigError::Invalid(format!(
                "the vision tower's pixel window is inverted: {} is above {}",
                self.min_pixels, self.max_pixels
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen3.8-Flash-Next's released tower.
    fn qwen() -> VisionGeometry {
        VisionGeometry {
            depth: 27,
            hidden_size: 1152,
            num_heads: 16,
            intermediate_size: 4304,
            out_hidden_size: 2560,
            projection_intermediate_size: 0,
            in_channels: 3,
            patch_size: 16,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            num_position_embeddings: 2304,
            rope_theta: 10_000.0,
            min_pixels: 65_536,
            max_pixels: 16_777_216,
        }
    }

    /// GLM-5.3-Flash's released tower, whose configuration carries no position grid.
    fn glm() -> VisionGeometry {
        VisionGeometry {
            depth: 24,
            hidden_size: 1024,
            num_heads: 16,
            intermediate_size: 4096,
            out_hidden_size: 4096,
            projection_intermediate_size: 10_240,
            in_channels: 3,
            patch_size: 14,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            num_position_embeddings: 0,
            rope_theta: 10_000.0,
            min_pixels: 0,
            max_pixels: 0,
        }
    }

    #[test]
    fn both_released_towers_validate() {
        qwen().validate().expect("the Qwen tower");
        glm().validate().expect("the GLM tower");
    }

    #[test]
    fn the_derived_dimensions_match_their_documented_values() {
        assert_eq!(qwen().head_dim(), 72);
        assert_eq!(glm().head_dim(), 64);
        assert_eq!(qwen().patch_dim(), 3 * 2 * 16 * 16);
        assert_eq!(glm().patch_dim(), 3 * 2 * 14 * 14);
        assert_eq!(qwen().position_side(), 48);
        assert_eq!(glm().position_side(), 0, "no learned grid");
        // A 28 x 28 patch grid merges to 14 x 14 at merge 2.
        assert_eq!(qwen().merged_rows(28, 28), 196);
        assert_eq!(glm().merged_rows(28, 28), 196);
    }

    #[test]
    fn a_geometry_the_kernels_cannot_run_is_refused_by_name() {
        let mut broken = qwen();
        broken.num_heads = 0;
        assert!(
            broken
                .validate()
                .unwrap_err()
                .to_string()
                .contains("num_heads")
        );
        let mut raggedly = qwen();
        raggedly.hidden_size = 1150;
        assert!(
            raggedly
                .validate()
                .unwrap_err()
                .to_string()
                .contains("multiple")
        );
        let mut grid = qwen();
        grid.num_position_embeddings = 2305;
        assert!(grid.validate().unwrap_err().to_string().contains("square"));
        let mut window = glm();
        window.min_pixels = 9;
        window.max_pixels = 8;
        assert!(
            window
                .validate()
                .unwrap_err()
                .to_string()
                .contains("inverted")
        );
    }
}

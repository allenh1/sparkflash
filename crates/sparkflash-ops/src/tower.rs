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

//! The vision tower's tensors and row arithmetic, described once.
//!
//! A tower is a patch embedding, `depth` pre-norm blocks and a merger. Every release in this tree
//! has that shape, and the device kernels ([D84](../../../docs/DECISIONS.md)) do not care which
//! checkpoint the weights came from: what differs is the geometry, which
//! [sparkflash_model::VisionGeometry] now carries, and the *names* under which a checkpoint stores
//! each piece, which stay the arm's business ([TowerTensor::role] is what the loader matches on).
//!
//! This module is the arithmetic between the two: given a geometry it says how many tensors a
//! tower needs and what each one's shape is, so a loader can check a checkpoint's index against the
//! geometry before a byte is uploaded, and the kernel sequence can size its scratch from the same
//! numbers. It touches no device and no checkpoint.

use sparkflash_model::VisionGeometry;

/// One tensor a tower needs: its role, the block it belongs to, and its shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TowerTensor {
    /// The role, stable across releases (`patch_embed.weight`, `block.qkv.weight`, ...).
    pub role: &'static str,
    /// The transformer block this tensor belongs to, or `None` for the embedding and the merger.
    pub block: Option<usize>,
    /// Row-major shape.
    pub shape: Vec<usize>,
}

impl TowerTensor {
    /// How many elements the tensor holds.
    #[must_use]
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Every tensor one tower needs, from its geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct TowerPlan {
    geometry: VisionGeometry,
    tensors: Vec<TowerTensor>,
}

impl TowerPlan {
    /// The plan for `geometry`.
    ///
    /// The merger is the two-layer form both releases use (a norm, a projection to the merged
    /// width and a projection to the language model's width); an arm whose merger differs passes
    /// its own tensors rather than bending this plan.
    #[must_use]
    pub fn new(geometry: &VisionGeometry) -> Self {
        let hidden = geometry.hidden_size;
        let intermediate = geometry.intermediate_size;
        let patch = geometry.patch_dim();
        let merged = geometry.spatial_merge_size * geometry.spatial_merge_size * hidden;
        let projection = if geometry.projection_intermediate_size == 0 {
            merged
        } else {
            geometry.projection_intermediate_size
        };
        let mut tensors = vec![
            TowerTensor {
                role: "patch_embed.weight",
                block: None,
                shape: vec![hidden, patch],
            },
            TowerTensor {
                role: "patch_embed.bias",
                block: None,
                shape: vec![hidden],
            },
        ];
        for block in 0..geometry.depth {
            for (role, shape) in [
                ("block.norm1.weight", vec![hidden]),
                ("block.norm1.bias", vec![hidden]),
                ("block.qkv.weight", vec![3 * hidden, hidden]),
                ("block.qkv.bias", vec![3 * hidden]),
                ("block.proj.weight", vec![hidden, hidden]),
                ("block.proj.bias", vec![hidden]),
                ("block.norm2.weight", vec![hidden]),
                ("block.norm2.bias", vec![hidden]),
                ("block.fc1.weight", vec![intermediate, hidden]),
                ("block.fc1.bias", vec![intermediate]),
                ("block.fc2.weight", vec![hidden, intermediate]),
                ("block.fc2.bias", vec![hidden]),
            ] {
                tensors.push(TowerTensor {
                    role,
                    block: Some(block),
                    shape,
                });
            }
        }
        for (role, shape) in [
            ("merger.norm.weight", vec![hidden]),
            ("merger.norm.bias", vec![hidden]),
            ("merger.fc1.weight", vec![projection, merged]),
            ("merger.fc1.bias", vec![projection]),
            (
                "merger.fc2.weight",
                vec![geometry.out_hidden_size, projection],
            ),
            ("merger.fc2.bias", vec![geometry.out_hidden_size]),
        ] {
            tensors.push(TowerTensor {
                role,
                block: None,
                shape,
            });
        }
        Self {
            geometry: geometry.clone(),
            tensors,
        }
    }

    /// The geometry the plan was built from.
    #[must_use]
    pub fn geometry(&self) -> &VisionGeometry {
        &self.geometry
    }

    /// Every tensor, in load order: the embedding, each block, the merger.
    #[must_use]
    pub fn tensors(&self) -> &[TowerTensor] {
        &self.tensors
    }

    /// How many elements the whole tower holds.
    #[must_use]
    pub fn elements(&self) -> usize {
        self.tensors.iter().map(TowerTensor::elements).sum()
    }

    /// The patches one `rows` by `cols` patch grid has.
    #[must_use]
    pub fn patches(rows: usize, cols: usize) -> usize {
        rows * cols
    }

    /// The rows the encoder runs over for a `rows` by `cols` patch grid: one a patch.
    #[must_use]
    pub fn encode_rows(rows: usize, cols: usize) -> usize {
        Self::patches(rows, cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn a_tower_is_the_embedding_twelve_a_block_and_the_merger() {
        let plan = TowerPlan::new(&qwen());
        // 2 embedding + 12 a block + 6 merger.
        assert_eq!(plan.tensors().len(), 2 + 12 * 27 + 6);
        assert_eq!(TowerPlan::new(&glm()).tensors().len(), 2 + 12 * 24 + 6);
        assert_eq!(plan.tensors()[0].role, "patch_embed.weight");
        assert_eq!(plan.tensors()[0].shape, vec![1152, 3 * 2 * 16 * 16]);
        let block = plan
            .tensors()
            .iter()
            .find(|tensor| tensor.role == "block.qkv.weight" && tensor.block == Some(26))
            .expect("the last block's qkv");
        assert_eq!(block.shape, vec![3 * 1152, 1152]);
    }

    #[test]
    fn a_merger_without_its_own_width_projects_from_the_merged_patch() {
        // Qwen's merger has no MLP of its own: 2 x 2 patches of 1152 is its input width.
        let plan = TowerPlan::new(&qwen());
        let fc1 = plan
            .tensors()
            .iter()
            .find(|tensor| tensor.role == "merger.fc1.weight")
            .expect("the merger");
        assert_eq!(fc1.shape, vec![4 * 1152, 4 * 1152]);
        let fc2 = plan
            .tensors()
            .iter()
            .find(|tensor| tensor.role == "merger.fc2.weight")
            .expect("the merger");
        assert_eq!(fc2.shape, vec![2560, 4 * 1152]);
        // GLM declares its merger MLP width, so the projection is that wide.
        let glm_plan = TowerPlan::new(&glm());
        let glm_fc1 = glm_plan
            .tensors()
            .iter()
            .find(|tensor| tensor.role == "merger.fc1.weight")
            .expect("the merger");
        assert_eq!(glm_fc1.shape, vec![10_240, 4 * 1024]);
    }

    #[test]
    fn the_element_count_is_the_sum_of_the_shapes() {
        let plan = TowerPlan::new(&glm());
        let sum: usize = plan.tensors().iter().map(TowerTensor::elements).sum();
        assert_eq!(plan.elements(), sum);
        // A 24-block 1,024-wide tower with 4,096-wide blocks is the release's ~0.3 B.
        assert!(
            (250_000_000..400_000_000).contains(&plan.elements()),
            "{}",
            plan.elements()
        );
    }

    #[test]
    fn the_encoder_runs_one_row_a_patch() {
        assert_eq!(TowerPlan::patches(28, 28), 784);
        assert_eq!(TowerPlan::encode_rows(28, 28), 784);
    }
}

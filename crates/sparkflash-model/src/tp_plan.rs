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

//! The tensor-parallel plan a model executes under.
//!
//! A [TpPlan] is the model-side answer to "what does rank `r` of `world` hold?".
//! It is deliberately independent of kernels and of a particular arm: the architecture says
//! which axes are sharded, the plan divides them, and an execution plan
//! (`sparkflash_ops::engine_plan`) carries the rank-local geometry to the kernel resolver.
//!
//! The split policy is conservative:
//!
//! - [TpAxis::Replicated] and [TpAxis::Mtp] are held whole on every rank;
//! - a sharded axis must divide evenly, except [TpAxis::KvHeads], which may be replicated
//!   when there are fewer key/value heads than ranks (the Qwen and GLM arms do this);
//! - padding is reported, never silently applied to the logical total. A kernel that needs
//!   whole tiles sees the padded per-rank width and the plan records how many columns are
//!   zero fill.

use std::fmt;

use crate::architecture::{Architecture, AttentionKind, Ffn};

/// An axis a model can split over the tensor-parallel world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpAxis {
    /// Held whole on every rank.
    Replicated,
    /// Query/attention heads.
    Heads,
    /// Key/value heads. May be replicated when there are fewer than the world size.
    KvHeads,
    /// Routed experts.
    Experts,
    /// An expert or MLP intermediate width.
    Intermediate,
    /// Vocabulary rows (embedding or LM head).
    Vocab,
    /// The residual width.
    Hidden,
    /// MTP/DSpark stages.
    Mtp,
    /// A stored tensor axis, split by the resident plan's per-tensor rule.
    Tensor,
}

impl TpAxis {
    /// The stable name used in diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Replicated => "replicated",
            Self::Heads => "heads",
            Self::KvHeads => "kv_heads",
            Self::Experts => "experts",
            Self::Intermediate => "intermediate",
            Self::Vocab => "vocab",
            Self::Hidden => "hidden",
            Self::Mtp => "mtp",
            Self::Tensor => "tensor",
        }
    }

    /// Whether this axis is actually divided over the world.
    #[must_use]
    pub const fn is_sharded(self) -> bool {
        !matches!(self, Self::Replicated | Self::Mtp)
    }
}

/// One rank's share of an axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Split {
    /// The logical total the axis represents.
    pub total: usize,
    /// This rank's logical share.
    pub per_rank: usize,
    /// This rank's first logical element.
    pub offset: usize,
    /// This rank's share rounded up to the kernel's alignment.
    pub padded: usize,
    /// Zero-fill columns added by [Self::padded].
    pub padding: usize,
    /// The rank this split describes.
    pub rank: usize,
    /// The world the split covers.
    pub world: usize,
}

impl Split {
    /// Whether this rank holds the whole axis.
    #[must_use]
    pub const fn is_replicated(&self) -> bool {
        self.world == 1 || self.total == self.per_rank
    }

    /// Whether the axis has fewer elements than ranks, so several ranks share one element.
    ///
    /// A key/value head count of 2 over a world of 4 is shared: ranks 0/1 read head 0 and
    /// ranks 2/3 read head 1. [Self::offset] still names the global head.
    #[must_use]
    pub const fn is_shared(&self) -> bool {
        self.total < self.world
    }

    /// The last logical element this rank holds, exclusive.
    #[must_use]
    pub const fn end(&self) -> usize {
        self.offset + self.per_rank
    }
}

/// A [TpPlan] error: an axis that cannot be divided as asked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpPlanError {
    /// What was being divided.
    pub axis: TpAxis,
    /// Why it failed.
    pub detail: String,
}

impl fmt::Display for TpPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "tensor-parallel {}: {}",
            self.axis.name(),
            self.detail
        )
    }
}

impl std::error::Error for TpPlanError {}

/// The tensor-parallel plan of one run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpPlan {
    world: usize,
    rank: usize,
}

impl TpPlan {
    /// A plan for `rank` of `world`.
    ///
    /// # Errors
    /// A zero world, or a rank outside it.
    pub fn new(world: usize, rank: usize) -> Result<Self, TpPlanError> {
        if world == 0 {
            return Err(TpPlanError {
                axis: TpAxis::Replicated,
                detail: "the world size is zero".to_owned(),
            });
        }
        if rank >= world {
            return Err(TpPlanError {
                axis: TpAxis::Replicated,
                detail: format!("rank {rank} is outside a world of {world}"),
            });
        }
        Ok(Self { world, rank })
    }

    /// The world size.
    #[must_use]
    pub const fn world(&self) -> usize {
        self.world
    }

    /// This process's rank.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// One rank's share of `total`, rounded up to `alignment`.
    ///
    /// # Errors
    /// A sharded axis that the world does not divide, or that is zero.
    pub fn split(
        &self,
        total: usize,
        axis: TpAxis,
        alignment: usize,
    ) -> Result<Split, TpPlanError> {
        if total == 0 {
            return Err(TpPlanError {
                axis,
                detail: "the axis is empty".to_owned(),
            });
        }
        if !axis.is_sharded() || self.world == 1 {
            let padded = total.max(1).next_multiple_of(alignment.max(1));
            return Ok(Split {
                total,
                per_rank: total,
                offset: 0,
                padded,
                padding: padded - total,
                rank: self.rank,
                world: self.world,
            });
        }
        // A key/value head count below the world is shared: every rank reads the same head
        // rather than inventing a fractional one. This is the Qwen/GLM rule.
        if axis == TpAxis::KvHeads && !total.is_multiple_of(self.world) {
            if self.world.is_multiple_of(total) {
                return Ok(Split {
                    total,
                    per_rank: 1,
                    offset: self.rank * total / self.world,
                    padded: 1,
                    padding: 0,
                    rank: self.rank,
                    world: self.world,
                });
            }
            return Err(TpPlanError {
                axis,
                detail: format!(
                    "{total} heads neither divide nor are divided by {}",
                    self.world
                ),
            });
        }
        if !total.is_multiple_of(self.world) {
            return Err(TpPlanError {
                axis,
                detail: format!("{total} does not divide over {}", self.world),
            });
        }
        let per_rank = total / self.world;
        let padded = per_rank.max(1).next_multiple_of(alignment.max(1));
        Ok(Split {
            total,
            per_rank,
            offset: self.rank * per_rank,
            padded,
            padding: padded - per_rank,
            rank: self.rank,
            world: self.world,
        })
    }

    /// One rank's share of `total` as a run of whole `tile`-wide blocks.
    ///
    /// This is the split an FP8 expert's weight takes: one BF16 inverse scale covers a whole
    /// 128-column tile, so a rank holds a whole number of tiles and the remainder goes to the
    /// first ranks (five tiles over four ranks are 2/1/1/1, not 1.25 each). [Split::padded]
    /// equals [Split::per_rank] because a tile run is already aligned, and [Split::offset] is in
    /// elements, not tiles.
    ///
    /// A single rank, or an axis that is not sharded, holds the whole total, a partial last tile
    /// included, since nothing is cut.
    ///
    /// # Errors
    /// A zero total or tile, a sharded total that is not whole tiles, or a rank that holds no
    /// tile.
    pub fn split_tiles(
        &self,
        total: usize,
        tile: usize,
        axis: TpAxis,
    ) -> Result<Split, TpPlanError> {
        if total == 0 {
            return Err(TpPlanError {
                axis,
                detail: "the axis is empty".to_owned(),
            });
        }
        if tile == 0 {
            return Err(TpPlanError {
                axis,
                detail: "the tile width is zero".to_owned(),
            });
        }
        // One rank holds the whole axis, a partial last tile included: nothing is cut.
        if !axis.is_sharded() || self.world == 1 {
            return Ok(Split {
                total,
                per_rank: total,
                offset: 0,
                padded: total,
                padding: 0,
                rank: self.rank,
                world: self.world,
            });
        }
        if !total.is_multiple_of(tile) {
            return Err(TpPlanError {
                axis,
                detail: format!("{total} is not whole {tile}-wide tiles"),
            });
        }
        let tiles = total / tile;
        let base = tiles / self.world;
        let extra = tiles % self.world;
        let count = base + usize::from(self.rank < extra);
        if count == 0 {
            return Err(TpPlanError {
                axis,
                detail: format!(
                    "{tiles} tiles over {} ranks leave rank {} none",
                    self.world, self.rank
                ),
            });
        }
        let first = self.rank * base + self.rank.min(extra);
        let per_rank = count * tile;
        Ok(Split {
            total,
            per_rank,
            offset: first * tile,
            padded: per_rank,
            padding: 0,
            rank: self.rank,
            world: self.world,
        })
    }

    /// Validate a plan against a released architecture and return it.
    ///
    /// The check is the model-side one: every axis the arm will shard has to divide under the
    /// plan before any weight is read. It is deliberately independent of the precision and the
    /// kernel catalogue; the execution plan and the resolver add those.
    ///
    /// # Errors
    /// A world/rank outside the plan, or an architecture axis the world does not divide.
    pub fn for_architecture(
        architecture: &Architecture,
        world: usize,
        rank: usize,
    ) -> Result<Self, TpPlanError> {
        let plan = Self::new(world, rank)?;
        for layer in &architecture.layers {
            let attention = &layer.attention;
            let heads = attention.heads;
            plan.split(heads, TpAxis::Heads, 1)
                .map_err(|error| TpPlanError {
                    axis: error.axis,
                    detail: format!("layer {}: {}", layer.index, error.detail),
                })?;
            match attention.kind {
                AttentionKind::Gqa { kv_heads, .. } => {
                    plan.split(kv_heads, TpAxis::KvHeads, 1)
                        .map_err(|error| TpPlanError {
                            axis: error.axis,
                            detail: format!("layer {}: {}", layer.index, error.detail),
                        })?;
                }
                AttentionKind::Gdn(ref gdn) => {
                    plan.split(gdn.key_heads, TpAxis::Heads, 1)
                        .map_err(|error| TpPlanError {
                            axis: error.axis,
                            detail: format!("layer {}: {}", layer.index, error.detail),
                        })?;
                    plan.split(gdn.value_heads, TpAxis::KvHeads, 1)
                        .map_err(|error| TpPlanError {
                            axis: error.axis,
                            detail: format!("layer {}: {}", layer.index, error.detail),
                        })?;
                }
                AttentionKind::Mla { .. } | AttentionKind::Kda(_) => {}
            }
            match layer.ffn {
                Ffn::Dense(ref dense) => {
                    plan.split(dense.intermediate, TpAxis::Intermediate, 1)
                        .map_err(|error| TpPlanError {
                            axis: error.axis,
                            detail: format!("layer {}: {}", layer.index, error.detail),
                        })?;
                }
                Ffn::Moe(ref moe) => {
                    plan.split(moe.intermediate, TpAxis::Intermediate, 1)
                        .map_err(|error| TpPlanError {
                            axis: error.axis,
                            detail: format!("layer {}: {}", layer.index, error.detail),
                        })?;
                }
            }
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEEPSEEK: &str =
        include_str!("../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json");
    const GLM: &str = include_str!("../../sparkflash-model-glm5-next/config/glm53_flash.hf.json");
    const QWEN: &str =
        include_str!("../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.hf.json");

    fn architecture(text: &str) -> Architecture {
        Architecture::from_json(text).expect("the released config parses")
    }

    #[test]
    fn a_replicated_axis_is_held_whole() {
        let plan = TpPlan::new(4, 2).expect("plan");
        let split = plan.split(5120, TpAxis::Replicated, 1).expect("split");
        assert_eq!(split.per_rank, 5120);
        assert_eq!(split.offset, 0);
        assert!(split.is_replicated());
    }

    #[test]
    fn a_sharded_axis_divides_and_pads() {
        let plan = TpPlan::new(4, 1).expect("plan");
        let split = plan.split(2304, TpAxis::Intermediate, 64).expect("split");
        assert_eq!(split.per_rank, 576);
        assert_eq!(split.offset, 576);
        assert_eq!(split.padded, 576);
        assert_eq!(split.padding, 0);

        let padded = plan.split(640, TpAxis::Intermediate, 64).expect("split");
        assert_eq!(padded.per_rank, 160);
        assert_eq!(padded.padded, 192);
        assert_eq!(padded.padding, 32);
    }

    #[test]
    fn a_shared_kv_head_names_the_global_head_this_rank_reads() {
        let offsets: Vec<usize> = (0..4)
            .map(|rank| {
                TpPlan::new(4, rank)
                    .expect("plan")
                    .split(2, TpAxis::KvHeads, 1)
                    .expect("split")
                    .offset
            })
            .collect();
        assert_eq!(offsets, [0, 0, 1, 1]);
        let split = TpPlan::new(4, 3)
            .expect("plan")
            .split(2, TpAxis::KvHeads, 1)
            .expect("split");
        assert_eq!(split.per_rank, 1);
        assert!(split.is_shared());
    }

    #[test]
    fn a_single_rank_still_pads_to_the_kernel_alignment() {
        let split = TpPlan::new(1, 0)
            .expect("plan")
            .split(640, TpAxis::Intermediate, 64)
            .expect("split");
        assert_eq!(split.per_rank, 640);
        assert_eq!(split.padded, 640);
        assert_eq!(split.padding, 0);
    }

    #[test]
    fn a_whole_tile_split_keeps_every_rank_on_tile_boundaries() {
        // Five 128-wide tiles over four ranks are 2/1/1/1, not 160 columns each rounded up to
        // the alignment: an FP8 expert's scale covers a whole tile, so a slice cannot split one.
        let shares: Vec<(usize, usize)> = (0..4)
            .map(|rank| {
                let split = TpPlan::new(4, rank)
                    .expect("plan")
                    .split_tiles(640, 128, TpAxis::Intermediate)
                    .expect("split");
                (split.offset, split.per_rank)
            })
            .collect();
        assert_eq!(shares, [(0, 256), (256, 128), (384, 128), (512, 128)]);
        let whole = TpPlan::new(1, 0)
            .expect("plan")
            .split_tiles(640, 128, TpAxis::Intermediate)
            .expect("split");
        assert_eq!(whole.per_rank, 640);
        assert_eq!(whole.padding, 0);
        // One rank holds a partial last tile whole: nothing is cut inside a scale tile.
        let partial = TpPlan::new(1, 0)
            .expect("plan")
            .split_tiles(8, 128, TpAxis::Intermediate)
            .expect("one rank holds the partial tile");
        assert_eq!((partial.offset, partial.per_rank), (0, 8));
        // A width that is not whole tiles is refused rather than split inside a scale tile.
        let error = TpPlan::new(4, 0)
            .expect("plan")
            .split_tiles(644, 128, TpAxis::Intermediate)
            .expect_err("644 is not whole 128-wide tiles");
        assert!(error.to_string().contains("whole"), "{error}");
    }

    #[test]
    fn an_uneven_axis_is_refused() {
        let plan = TpPlan::new(3, 0).expect("plan");
        let error = plan.split(4, TpAxis::Heads, 1).expect_err("uneven");
        assert!(error.to_string().contains("does not divide"));
    }

    #[test]
    fn all_three_released_architectures_plan_at_tp1_and_tp4() {
        for (name, text) in [("DeepSeek", DEEPSEEK), ("GLM", GLM), ("Qwen", QWEN)] {
            for world in [1, 2, 4] {
                TpPlan::for_architecture(&architecture(text), world, 0)
                    .unwrap_or_else(|error| panic!("{name} TP{world}: {error}"));
            }
        }
    }

    #[test]
    fn a_world_that_does_not_divide_the_architecture_is_refused() {
        let error = TpPlan::for_architecture(&architecture(GLM), 3, 0).expect_err("TP3");
        assert!(error.to_string().contains("layer"));
    }

    #[test]
    fn a_rank_outside_the_world_is_refused() {
        let error = TpPlan::new(4, 4).expect_err("rank 4 of 4");
        assert!(error.to_string().contains("outside"));
    }
}

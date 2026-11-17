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

//! The sparse indexer's launch geometry: pool, score and pick.
//!
//! Qwen's QSA and GLM's DSA indexer run the same three bodies over one pass of query rows. Every
//! `ratio` cached keys pool into one block; each row scores every pool it can see; the scores
//! pick the tokens the row attends over. The grids are one rule for both families, so the rank's
//! boot plan ([crate::launchers::SparseIndexerLauncher]) and the arms' walks read them here. The
//! bodies' arithmetic is their own; nothing here changes which body runs.

/// The tokens a sparse-attention query selects at most: every selection buffer's row stride.
///
/// Mirrors `Q4E_MAX_SELECTED` in the Qwen and GLM kernels. The released configurations select
/// at most 2,051 (Qwen3.8: 512 complete blocks of four plus the incomplete tail), so the margin is
/// small on purpose: a configuration that outgrows it is refused at open rather than writing past
/// the buffer.
pub const MAX_SELECTED_TOKENS: usize = 2304;

/// The pools the selection kernel folds into (`Q4E_MAX_BUDGET`).
pub const MAX_BUDGET_BLOCKS: usize = 1024;

/// The pools one score block covers (`Q4E_QSA_TILE`), for both families' tiled score body.
pub const SCORE_TILE: usize = 64;

/// The threads of a score, pick or GLM pool block.
pub const INDEX_BLOCK: u32 = 256;

/// The threads of a Qwen pool block (`q4e_qsa_pool` pools one block a CTA).
pub const QSA_POOL_BLOCK: u32 = 128;

/// `glm_prefill_index_scores`' geometry (`GPI_HEADS`, `GPI_DIM`, `GPI_POOLS`, `GPI_ROWS`,
/// `GPI_THREADS`): the indexer shape it is compiled for, and the pools and rows a block scores.
pub const RUN_SCORE_HEADS: usize = 32;
/// See [RUN_SCORE_HEADS].
pub const RUN_SCORE_DIM: usize = 128;
/// See [RUN_SCORE_HEADS].
pub const RUN_SCORE_POOLS: usize = 64;
/// See [RUN_SCORE_HEADS].
pub const RUN_SCORE_ROWS: usize = 32;
/// See [RUN_SCORE_HEADS].
pub const RUN_SCORE_THREADS: u32 = 128;

/// A kernel's grid and block.
pub type Launch = ((u32, u32, u32), (u32, u32, u32));

/// One indexer pass: `rows` query rows starting at absolute `position`, over pools of `ratio`
/// keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexerPass {
    /// The first row's absolute position.
    pub position: usize,
    /// The query rows the pass covers.
    pub rows: usize,
    /// The keys a pool holds.
    pub ratio: usize,
}

impl IndexerPass {
    /// A pass of `rows` rows at `position`, over pools of `ratio` keys (at least one).
    #[must_use]
    pub const fn new(position: usize, rows: usize, ratio: usize) -> Self {
        Self {
            position,
            rows,
            ratio: if ratio == 0 { 1 } else { ratio },
        }
    }

    /// The whole pools the pass's last row sees: a pool is visible to a row at or past its end.
    #[must_use]
    pub const fn pools(&self) -> usize {
        (self.position + self.rows) / self.ratio
    }

    /// Qwen's pool: a 128-thread block a pool, every pool the last row sees pooled once for all
    /// the rows. `None` when no pool is complete yet (the launch would be empty).
    #[must_use]
    pub const fn qsa_pool(&self) -> Option<Launch> {
        let pools = self.pools();
        if pools == 0 {
            None
        } else {
            Some(((pools as u32, 1, 1), (QSA_POOL_BLOCK, 1, 1)))
        }
    }

    /// The tiled score body (Qwen's `q4e_qsa_scores`, GLM's `q4e_glm_index_scores`): a block a
    /// [SCORE_TILE] of pools and a row, at least one tile (a block past a row's pools is a no-op).
    #[must_use]
    pub const fn scores(&self) -> Launch {
        let tiles = self.pools().div_ceil(SCORE_TILE);
        (
            (
                if tiles == 0 { 1 } else { tiles } as u32,
                self.rows as u32,
                1,
            ),
            (INDEX_BLOCK, 1, 1),
        )
    }

    /// The pick (Qwen's `q4e_qsa_pick`, GLM's `q4e_glm_pick`), or GLM's pool
    /// (`q4e_glm_pool`): a block a row.
    #[must_use]
    pub const fn per_row(&self) -> Launch {
        ((self.rows as u32, 1, 1), (INDEX_BLOCK, 1, 1))
    }

    /// GLM's prefill score body (`glm_prefill_index_scores`), which keeps a tile of pooled keys
    /// while it walks a run's rows and computes the tiled body's floats: for a run longer than
    /// `token_rows` (the arm's token-path limit) at the one indexer shape it is compiled for,
    /// else `None` (the tiled [Self::scores]).
    #[must_use]
    pub const fn run_scores(&self, heads: usize, dim: usize, token_rows: usize) -> Option<Launch> {
        if self.rows > token_rows && heads == RUN_SCORE_HEADS && dim == RUN_SCORE_DIM {
            let tiles = self.pools().div_ceil(RUN_SCORE_POOLS);
            Some((
                (
                    if tiles == 0 { 1 } else { tiles } as u32,
                    self.rows.div_ceil(RUN_SCORE_ROWS) as u32,
                    1,
                ),
                (RUN_SCORE_THREADS, 1, 1),
            ))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grids_are_the_arms_old_rules() {
        // Qwen's old pool, score and pick launches (blocks.rs) and GLM's (device.rs), verbatim.
        for (position, rows, ratio) in [
            (0, 1, 4),
            (3, 1, 4),
            (0, 128, 4),
            (5000, 37, 4),
            (9, 3, 128),
        ] {
            let pass = IndexerPass::new(position, rows, ratio);
            let pooled = (position + rows) / ratio;
            assert_eq!(
                pass.qsa_pool(),
                (pooled > 0).then_some(((pooled as u32, 1, 1), (128, 1, 1)))
            );
            assert_eq!(
                pass.scores(),
                (
                    (pooled.div_ceil(64).max(1) as u32, rows as u32, 1),
                    (256, 1, 1)
                )
            );
            assert_eq!(pass.per_row(), ((rows as u32, 1, 1), (256, 1, 1)));
            let run = pass.run_scores(32, 128, 8);
            if rows > 8 {
                assert_eq!(
                    run,
                    Some((
                        (
                            pooled.div_ceil(64).max(1) as u32,
                            rows.div_ceil(32) as u32,
                            1
                        ),
                        (128, 1, 1)
                    ))
                );
            } else {
                assert_eq!(run, None);
            }
            assert_eq!(pass.run_scores(16, 128, 8), None, "another shape is tiled");
        }
    }
}

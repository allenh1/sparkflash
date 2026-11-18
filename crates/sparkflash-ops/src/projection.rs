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

//! The dense projection dispatcher: which kernel a BF16 projection of `rows` token rows takes,
//! and the shared wide tile's launch.
//!
//! A projection is `out[rows][columns] = input[rows][inner] x weightᵀ` over a row-major
//! `columns x inner` weight. Its kernel follows from the shape, the row count and the arm's
//! [WidePolicy]:
//!
//! * up to [GEMV_MAX_BATCH] rows, the row GEMV ([Route::RowGemv]): the decode path, whose bits a
//!   verified row is defined by;
//! * more rows on a shape the wide tile covers, the wide tensor-core tile ([Route::Wide],
//!   `kernels/shared_prefill_mma_sm121.cuh`'s 128 x 256 body); an arm whose prompt walk takes the
//!   tile at every row count says so ([WidePolicy::min_rows] of 1);
//! * more rows on a shape it does not cover, the skinny kernel where the arm has one
//!   ([Route::Skinny]), the row GEMV a group of rows at a time ([Route::BatchedGemv]), or a
//!   refusal where falling back would change a chunk's arithmetic ([Route::Refuse]).
//!
//! The policy is the arm's, because its rule is what its digests were recorded under: GLM's bulk
//! prefill, Dory's prompt walk and Inkling's tensor walk launch the same tile on the same grid, and
//! differ only in the rows they start it at and in what an uncovered shape does. Each still
//! launches its own kernel symbol from its own image ([WideTile::call] gives the grid, block and
//! arguments every one of them passes).
//!
//! Two more tiles live here, each with the rule its digests were recorded under:
//!
//! * the split-K BF16 tile ([SplitTile], Qwen's chunk path): 128 x 128, its inner dimension cut
//!   over the grid's third axis by [split_k_slices], the cut that decides a row's sums;
//! * the block-scaled FP8 tile ([Fp8Tile]): 128 x 128 over E4M3 weights with a scale a 128-column
//!   block, the tile Qwen's and GLM's FP8 projections launch.
//!
//! Still the arm's own: DeepSeek's `mxfp8_gemv_launch` family, whose MX scales no other arm reads.

use crate::gemv::{E4M3_TILE, GEMV_MAX_BATCH};

/// Output columns a wide-tile block computes (`GLM_WIDE_BN`).
pub const WIDE_COLUMNS: usize = 256;

/// Token rows a wide-tile block computes (`GLM_BM`).
pub const WIDE_ROWS: usize = 128;

/// Threads a wide-tile block runs (`OM_THREADS`).
pub const WIDE_THREADS: u32 = 256;

/// The wide tile's dynamic shared memory (`GLM_WIDE_SHARED`): three cp.async stages of a 128-row
/// f32 activation tile and a 256-row BF16 weight tile, 32 K each. A kernel on it is loaded with
/// this much.
pub const WIDE_SHARED: u32 = 98_304;

/// The wide tile reads whole tiles of this many output columns and inner values.
pub const WIDE_MULTIPLE: usize = 32;

/// What a BF16 projection does where the wide tile does not cover its shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffShape {
    /// The row GEMV, a group of rows at a time.
    Gemv,
    /// The skinny kernel for at most [SKINNY_MAX_COLUMNS] output columns over an inner width of
    /// whole [SKINNY_INNER_MULTIPLE] steps, else the row GEMV.
    Skinny,
    /// Refuse: the row GEMV would change the chunk's arithmetic.
    Refuse,
}

/// The widest output the skinny kernels serve (four instantiations of eight columns).
pub const SKINNY_MAX_COLUMNS: usize = 32;

/// The inner step the skinny kernels read: eight floats a lane, 32 lanes.
pub const SKINNY_INNER_MULTIPLE: usize = 256;

/// When an arm's BF16 projections take the wide tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WidePolicy {
    /// The fewest rows the tile takes. Below it the row GEMV runs.
    pub min_rows: usize,
    /// What a shape the tile does not cover takes, at `min_rows` rows or more.
    pub off_shape: OffShape,
}

impl WidePolicy {
    /// GLM's bulk prefill: past the row GEMV's rows, the tile, else the skinny kernel or the row
    /// GEMV.
    pub const GLM: Self = Self {
        min_rows: GEMV_MAX_BATCH + 1,
        off_shape: OffShape::Skinny,
    };

    /// Dory's prompt walk: past the row GEMV's rows the tile or nothing, since the token GEMV
    /// would change the chunk's arithmetic.
    pub const DORY: Self = Self {
        min_rows: GEMV_MAX_BATCH + 1,
        off_shape: OffShape::Refuse,
    };

    /// Inkling's tensor walk: the tile at every row count (the prompt's bits do not depend on its
    /// chunk widths), else the row GEMV.
    pub const INKLING_TENSOR: Self = Self {
        min_rows: 1,
        off_shape: OffShape::Gemv,
    };

    /// The shared dense arm's prompt walk: past the row GEMV's rows the tile, else the row GEMV a
    /// group of rows at a time. Decode rows keep the row GEMV, as on GLM; the arm has no skinny
    /// kernel.
    pub const DENSE: Self = Self {
        min_rows: GEMV_MAX_BATCH + 1,
        off_shape: OffShape::Gemv,
    };
}

/// The kernel a projection takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// The row GEMV over at most [GEMV_MAX_BATCH] rows.
    RowGemv,
    /// The row GEMV a group of [GEMV_MAX_BATCH] rows at a time.
    BatchedGemv,
    /// The wide tensor-core tile ([WideTile]).
    Wide,
    /// The skinny kernel ([OffShape::Skinny]).
    Skinny,
    /// No kernel the policy allows.
    Refuse,
}

/// Whether the wide tile covers a `columns x inner` weight.
#[must_use]
pub fn wide_fits(columns: usize, inner: usize) -> bool {
    columns.is_multiple_of(WIDE_MULTIPLE) && inner.is_multiple_of(WIDE_MULTIPLE)
}

/// The kernel a BF16 projection of `rows` rows over a `columns x inner` weight takes under
/// `policy`; `None` is an arm (or a walk) that never takes the tile.
#[must_use]
pub fn route_bf16(policy: Option<WidePolicy>, columns: usize, inner: usize, rows: usize) -> Route {
    let Some(policy) = policy else {
        return gemv(rows);
    };
    if rows < policy.min_rows {
        return gemv(rows);
    }
    policy.bulk(columns, inner, rows)
}

impl WidePolicy {
    /// The kernel once the caller has taken the bulk path ([route_bf16] at [Self::min_rows] rows
    /// or more, or a bulk walk the arm runs at any rows, as GLM's sequence-parallel share does).
    #[must_use]
    pub fn bulk(self, columns: usize, inner: usize, rows: usize) -> Route {
        if wide_fits(columns, inner) {
            return Route::Wide;
        }
        match self.off_shape {
            OffShape::Gemv => gemv(rows),
            OffShape::Skinny
                if columns <= SKINNY_MAX_COLUMNS && inner.is_multiple_of(SKINNY_INNER_MULTIPLE) =>
            {
                Route::Skinny
            }
            OffShape::Skinny => gemv(rows),
            OffShape::Refuse => Route::Refuse,
        }
    }
}

fn gemv(rows: usize) -> Route {
    if rows <= GEMV_MAX_BATCH {
        Route::RowGemv
    } else {
        Route::BatchedGemv
    }
}

/// One launch of the wide tile: its grid, block and the arguments every kernel on the body takes
/// (weight, input, output, columns, inner, rows).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WideTile {
    /// Blocks: the output's columns by its token rows.
    pub grid: (u32, u32, u32),
    /// Threads a block.
    pub block: (u32, u32, u32),
    /// The kernel's arguments.
    pub args: [u64; 6],
}

impl WideTile {
    /// The launch of `out[rows][columns] = input[rows][inner] x weightᵀ` (device addresses).
    ///
    /// No rows is one block that writes nothing, as the kernel guards its rows.
    ///
    /// # Errors
    /// A shape the tile does not cover ([wide_fits]).
    pub fn call(
        weight: u64,
        input: u64,
        output: u64,
        columns: usize,
        inner: usize,
        rows: usize,
    ) -> Result<Self, String> {
        if !wide_fits(columns, inner) {
            return Err(format!(
                "the wide tile reads whole {WIDE_MULTIPLE}-column tiles, not a {columns} x {inner} \
                 projection"
            ));
        }
        let blocks = |count: usize, per: usize| {
            u32::try_from(count.div_ceil(per).max(1))
                .map_err(|_| format!("{count} is more than one launch's blocks hold"))
        };
        Ok(Self {
            grid: (blocks(columns, WIDE_COLUMNS)?, blocks(rows, WIDE_ROWS)?, 1),
            block: (WIDE_THREADS, 1, 1),
            args: [
                weight,
                input,
                output,
                columns as u64,
                inner as u64,
                rows as u64,
            ],
        })
    }
}

/// Output columns and token rows a split-K tile block computes (`OM_BM`, `OM_BN`).
pub const SPLIT_TILE: usize = 128;

/// The inner stage the split-K tile reads: a slice is whole stages of this many values.
pub const SPLIT_STAGE: usize = 32;

/// Blocks a split-K launch aims for: two per SM on GB10's forty-eight.
pub const SPLIT_TARGET_BLOCKS: usize = 96;

/// The thinnest inner slice a split takes, in [SPLIT_STAGE]s.
pub const SPLIT_MIN_SLICE_STAGES: usize = 8;

/// Whether the split-K tile covers a `columns x inner` weight (whole 32-wide tiles both ways; the
/// arm checks its operands' alignment).
#[must_use]
pub fn split_fits(columns: usize, inner: usize) -> bool {
    columns.is_multiple_of(SPLIT_STAGE) && inner.is_multiple_of(SPLIT_STAGE)
}

/// The inner split of a split-K launch over `rows` token rows of a weight with `columns` output
/// columns and `stages` [SPLIT_STAGE]s of inner values: enough slices to bring the grid near
/// [SPLIT_TARGET_BLOCKS], as many as `partial_floats` of partial sums hold, none thinner than
/// [SPLIT_MIN_SLICE_STAGES], and all the same width, so a divisor of the stages.
///
/// A row's sums depend on this cut and nothing else in the launch, so an arm that wants a row's
/// bits independent of the batch passes the same `rows` for every piece (Qwen passes its piece
/// width, `DENSE_ROWS`, with the decode steps on the tile).
#[must_use]
pub fn split_k_slices(columns: usize, stages: usize, rows: usize, partial_floats: usize) -> usize {
    let tiles = columns.div_ceil(SPLIT_TILE) * rows.div_ceil(SPLIT_TILE);
    let ceiling = SPLIT_TARGET_BLOCKS
        .div_ceil(tiles)
        .min(partial_floats / (rows * columns).max(1))
        .min(stages / SPLIT_MIN_SLICE_STAGES)
        .max(1);
    (1..=ceiling)
        .rev()
        .find(|slices| stages.is_multiple_of(*slices))
        .unwrap_or(1)
}

/// One launch of the split-K tile: a block a 128 x 128 output tile and inner slice, and the
/// arguments its body takes (weight, input, target, columns, inner, rows, slices). With more than
/// one slice the target is the arm's partials, which its reduce adds in slice order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitTile {
    /// Blocks: output columns by token rows by slices.
    pub grid: (u32, u32, u32),
    /// Threads a block.
    pub block: (u32, u32, u32),
    /// The kernel's arguments.
    pub args: [u64; 7],
}

impl SplitTile {
    /// The launch of `target[slices][rows][columns]`, the partial products of
    /// `input[rows][inner] x weightᵀ` (device addresses), `slices` ways ([split_k_slices]).
    #[must_use]
    pub fn call(
        weight: u64,
        input: u64,
        target: u64,
        columns: usize,
        inner: usize,
        rows: usize,
        slices: usize,
    ) -> Self {
        Self {
            grid: (
                columns.div_ceil(SPLIT_TILE) as u32,
                rows.div_ceil(SPLIT_TILE) as u32,
                slices as u32,
            ),
            block: (256, 1, 1),
            args: [
                weight,
                input,
                target,
                columns as u64,
                inner as u64,
                rows as u64,
                slices as u64,
            ],
        }
    }
}

/// Output columns and token rows a block-scaled FP8 tile block computes; its scales cover one
/// [E4M3_TILE] of inner values each.
pub const FP8_TILE: usize = 128;

const _: () = assert!(FP8_TILE == E4M3_TILE);

/// Whether the block-scaled FP8 tile covers a `columns x inner` weight: whole 32-column output
/// tiles, and whole scale blocks of inner values. An arm adds what its own operands need
/// (Qwen's 16-byte alignment, GLM's slice start on a scale block).
#[must_use]
pub fn fp8_tile_fits(columns: usize, inner: usize) -> bool {
    columns.is_multiple_of(SPLIT_STAGE) && inner.is_multiple_of(E4M3_TILE)
}

/// The kernel a block-scaled FP8 projection of `rows` rows over a `columns x inner` weight takes
/// under `policy`, by [route_bf16]'s rule: [Route::Wide] is the FP8 tile ([Fp8Tile]) where
/// [fp8_tile_fits] covers the shape, and the row GEMV (the block-scaled E4M3 GEMV) otherwise. An
/// arm adds its operands' alignment.
#[must_use]
pub fn route_fp8(policy: Option<WidePolicy>, columns: usize, inner: usize, rows: usize) -> Route {
    let Some(policy) = policy else {
        return gemv(rows);
    };
    if rows < policy.min_rows {
        return gemv(rows);
    }
    if fp8_tile_fits(columns, inner) {
        return Route::Wide;
    }
    match policy.off_shape {
        OffShape::Refuse => Route::Refuse,
        OffShape::Gemv | OffShape::Skinny => gemv(rows),
    }
}

/// One launch of the block-scaled FP8 tile: its grid and block, and the arguments the
/// BF16-activation bodies take (weight, scales, input, output, columns, inner, rows). GLM's
/// quantised-activation body runs on the same grid with its own arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp8Tile {
    /// Blocks: output columns by token rows.
    pub grid: (u32, u32, u32),
    /// Threads a block.
    pub block: (u32, u32, u32),
    /// The kernel's arguments.
    pub args: [u64; 7],
}

impl Fp8Tile {
    /// The launch of `out[rows][columns] = input[rows][inner] x (weight * scales)ᵀ`.
    ///
    /// # Errors
    /// A shape the tile does not cover ([fp8_tile_fits]).
    #[allow(clippy::too_many_arguments)]
    pub fn call(
        weight: u64,
        scales: u64,
        input: u64,
        output: u64,
        columns: usize,
        inner: usize,
        rows: usize,
    ) -> Result<Self, String> {
        if !fp8_tile_fits(columns, inner) {
            return Err(format!(
                "the FP8 tile reads whole {SPLIT_STAGE}-column tiles over {E4M3_TILE}-value scale \
                 blocks, not a {columns} x {inner} projection"
            ));
        }
        Ok(Self {
            grid: Self::grid(columns, rows),
            block: (256, 1, 1),
            args: [
                weight,
                scales,
                input,
                output,
                columns as u64,
                inner as u64,
                rows as u64,
            ],
        })
    }

    /// The grid over `columns` output columns and `rows` token rows.
    #[must_use]
    pub const fn grid(columns: usize, rows: usize) -> (u32, u32, u32) {
        (
            columns.div_ceil(FP8_TILE) as u32,
            rows.div_ceil(FP8_TILE) as u32,
            1,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_decode_step_keeps_the_row_gemv_under_every_policy() {
        for policy in [WidePolicy::GLM, WidePolicy::DORY] {
            for rows in 1..=GEMV_MAX_BATCH {
                assert_eq!(route_bf16(Some(policy), 4096, 4096, rows), Route::RowGemv);
            }
        }
        assert_eq!(route_bf16(None, 4096, 4096, 3), Route::RowGemv);
        assert_eq!(route_bf16(None, 4096, 4096, 300), Route::BatchedGemv);
    }

    #[test]
    fn the_tensor_walk_takes_the_tile_at_every_row_count() {
        for rows in [1, 7, 8, 9, 4096] {
            assert_eq!(
                route_bf16(Some(WidePolicy::INKLING_TENSOR), 2048, 1024, rows),
                Route::Wide
            );
        }
        assert_eq!(
            route_bf16(Some(WidePolicy::INKLING_TENSOR), 2047, 1024, 4),
            Route::RowGemv
        );
        assert_eq!(
            route_bf16(Some(WidePolicy::INKLING_TENSOR), 2047, 1024, 40),
            Route::BatchedGemv
        );
    }

    #[test]
    fn an_uncovered_shape_takes_what_the_arm_says() {
        // GLM: a 24-column mixer over 4096 takes the skinny kernel, a 40-column one the GEMV.
        assert_eq!(
            route_bf16(Some(WidePolicy::GLM), 24, 4096, 64),
            Route::Skinny
        );
        assert_eq!(
            route_bf16(Some(WidePolicy::GLM), 40, 4096, 64),
            Route::BatchedGemv
        );
        assert_eq!(
            route_bf16(Some(WidePolicy::GLM), 24, 4000, 64),
            Route::BatchedGemv
        );
        // Dory refuses rather than change the chunk's arithmetic.
        assert_eq!(
            route_bf16(Some(WidePolicy::DORY), 24, 4096, 64),
            Route::Refuse
        );
        assert_eq!(
            route_bf16(Some(WidePolicy::DORY), 2048, 4096, 64),
            Route::Wide
        );
    }

    #[test]
    fn the_wide_launch_covers_the_output_in_blocks_of_its_tile() {
        let tile = WideTile::call(1, 2, 3, 4096 + 32, 2048, 129).expect("a covered shape");
        assert_eq!(tile.grid, (17, 2, 1));
        assert_eq!(tile.block, (WIDE_THREADS, 1, 1));
        assert_eq!(tile.args, [1, 2, 3, 4128, 2048, 129]);
        assert!(WideTile::call(1, 2, 3, 4100, 2048, 8).is_err());
        assert_eq!(
            WideTile::call(1, 2, 3, 4096, 2048, 0)
                .expect("no rows")
                .grid,
            (16, 1, 1)
        );
    }
    /// Qwen's rule as it stood in its arm (prefill.rs `dense_segments`), kept here to pin the
    /// shared one to it.
    fn qwen_segments(rows: usize, k_blocks: usize, tokens: usize) -> usize {
        let tiles = rows.div_ceil(128) * tokens.div_ceil(128);
        let ceiling = 96usize
            .div_ceil(tiles)
            .min((4 << 20) / (tokens * rows).max(1))
            .min(k_blocks / 8)
            .max(1);
        (1..=ceiling)
            .rev()
            .find(|segments| k_blocks.is_multiple_of(*segments))
            .unwrap_or(1)
    }

    #[test]
    fn the_split_is_qwens_rule() {
        for columns in [
            32, 128, 512, 1024, 2048, 4096, 5120, 6144, 8192, 12288, 17408, 248_320,
        ] {
            for stages in [1, 8, 16, 32, 64, 128, 160, 256, 544] {
                for rows in [1, 8, 9, 63, 64, 128, 300, 512, 4096] {
                    assert_eq!(
                        split_k_slices(columns, stages, rows, 4 << 20),
                        qwen_segments(columns, stages, rows),
                        "{columns} x {stages} stages at {rows} rows"
                    );
                }
            }
        }
        // A narrow 1024-column projection over 4096 values at one 512-row piece: 32 tiles, three
        // slices would reach the target but 128 stages split evenly only by two.
        assert_eq!(split_k_slices(1024, 128, 512, 4 << 20), 2);
        let tile = SplitTile::call(1, 2, 3, 1024, 4096, 512, 2);
        assert_eq!(tile.grid, (8, 4, 2));
        assert_eq!(tile.args, [1, 2, 3, 1024, 4096, 512, 2]);
    }

    #[test]
    fn the_fp8_tile_takes_whole_scale_blocks() {
        assert!(fp8_tile_fits(4096, 2048));
        assert!(!fp8_tile_fits(4096, 2048 + 32));
        assert!(!fp8_tile_fits(4100, 2048));
        let tile = Fp8Tile::call(1, 2, 3, 4, 4096 + 32, 2048, 129).expect("a covered shape");
        assert_eq!(tile.grid, (33, 2, 1));
        assert_eq!(tile.block, (256, 1, 1));
        assert_eq!(tile.args, [1, 2, 3, 4, 4128, 2048, 129]);
        assert!(Fp8Tile::call(1, 2, 3, 4, 4096, 2080, 8).is_err());
    }

    #[test]
    fn an_fp8_projection_routes_by_the_bf16_rule_over_its_own_tile() {
        let dense = Some(WidePolicy::DENSE);
        for rows in 1..=GEMV_MAX_BATCH {
            assert_eq!(route_fp8(dense, 4096, 2048, rows), Route::RowGemv);
        }
        assert_eq!(route_fp8(dense, 4096, 2048, 256), Route::Wide);
        assert_eq!(route_fp8(None, 4096, 2048, 256), Route::BatchedGemv);
        // A width the tile does not take: the BF16 tile's 32-wide inner is not a scale block.
        assert_eq!(route_fp8(dense, 4096, 2048 + 32, 256), Route::BatchedGemv);
        assert_eq!(route_fp8(Some(WidePolicy::DORY), 4100, 2048, 256), Route::Refuse);
    }
}

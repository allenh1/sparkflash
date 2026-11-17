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
//! Not yet routed here: Qwen's BF16 tensor-core path (a 128 x 128 tile with split K whose cut
//! decides the sums), the FP8 projections (block, tile and MX scales differ by arm) and DeepSeek's
//! `mxfp8_gemv_launch` family. Each keeps the rule its digests were recorded under until a gate
//! shows a shared one gives the same bits.

use crate::gemv::GEMV_MAX_BATCH;

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
}

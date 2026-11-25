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

//! The projection of a release whose FP8 linears quantise their own input: E4M3 weights on
//! 128 x 128 tiles with an f32 inverse scale a tile, or BF16 rows, multiplied by f32 rows.
//!
//! The release's dynamic activation scheme (`quantization_config.activation_scheme = dynamic`)
//! quantises each input row per 128 columns at `amax / 448` before an FP8 linear multiplies it.
//! The bodies here read that activation, so they are the ones such a linear can run on:
//!
//! * up to [GEMV_GROUP_ROWS] rows, the tiled row GEMV (`q4e_gemv_tiled_e4m3_block`, which
//!   quantises a row as it reads it, and `q4e_gemv_tiled_bf16_parts`): the decode path, whose
//!   bits a verified row is defined by, a group of [GEMV_MAX_BATCH] rows a z-block;
//! * more rows, the bulk tiles: the rows quantised once (`glm_quantise_rows`) into an E4M3 x
//!   E4M3 MMA that folds each block's sum by its two scales (`glm_prefill_q8_mma`), and a BF16
//!   weight on the wide tile ([WideTile]) or the skinny kernels, else the row GEMV.
//!
//! The release's prefill activation ([TiledActivation]) picks the bulk FP8 body: the quantised
//! MMA, or the f32 input split into a BF16 head and remainder (`glm_prefill_fp8b128_mma`), or
//! the head alone. An `exact` projection keeps the row GEMV's bits past [GEMV_MAX_BATCH] rows,
//! a BF16 part `glm_prefill_exact_bf16` covers taking that body (the same floats).
//!
//! A `held` projection rounds its outputs to BF16 in place afterwards: the producer's module
//! returns them in BF16, and a reader that does not round on its own read takes them so.
//!
//! These are GLM-5.3-Flash's bodies, moved here from its arm with their arguments unchanged.

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_kernels::images::Image;
use sparkflash_model::backend::BackendError;

use crate::gemv::{
    GEMV_MAX_BATCH, TILED_GEMV_BLOCKS, TILED_GEMV_PARTIALS, TILED_GEMV_ROUND,
    tiled_gemv_e4m3_splits, tiled_gemv_splits,
};
use crate::plan::Precision;
use crate::projection::{Fp8Tile, Route, WIDE_SHARED, WidePolicy, WideTile, fp8_tile_fits};
use crate::roles::{Bound, KernelTable, Launcher};

/// The error code of the tiled projection's refusals.
const TILED: &str = "tiled projection";

/// The most rows one grouped GEMV launch takes.
pub const GEMV_GROUP_ROWS: usize = 512;
/// The groups of [GEMV_MAX_BATCH] rows a grouped GEMV launch holds partials and counters for.
pub const GEMV_GROUPS: usize = GEMV_GROUP_ROWS / GEMV_MAX_BATCH;
/// The rows of a tiled GEMV's tile.
const TILE_ROWS: usize = 32;
/// The side of a block-scaled FP8 tile, and of an activation's quantised block.
pub const FP8_BLOCK: usize = 128;
/// GLM_Q8_SHARED: the q8 GEMM's cp.async stages and a column tile's weight scales.
const Q8_SHARED: u32 = 68_608;
/// GLM_Q8_BLOCKS: the widest inner, in 128-column blocks, the q8 GEMM holds the scales of.
const Q8_MAX_INNER: usize = 128 * FP8_BLOCK;
/// The exact BF16 body's outputs, rows and threads a block.
const EXACT_OUTS: usize = 32;
const EXACT_ROWS: usize = 32;
const EXACT_THREADS: u32 = 128;
/// The threads of every other launch here.
const BLOCK: u32 = 256;

/// How a bulk projection reads its f32 input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiledActivation {
    /// The release's dynamic E4M3 codes into an E4M3 x E4M3 MMA (an FP8 weight's input), a BF16
    /// weight's as [Self::Split]: the default for a release whose linears quantise their input.
    Fp8,
    /// Each value as a BF16 head and remainder, two MMA chains: the default otherwise.
    Split,
    /// The BF16 head alone, one chain.
    Bf16,
}

/// One linear the tiled projection multiplies by: `rows x columns`, BF16 (`scales` absent) or
/// E4M3 with its f32 128 x 128 inverse scales, in up to four fused parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TiledMatrix {
    /// The values: two bytes each for BF16, one for E4M3, row-major.
    pub weight: Deviceptr,
    /// The E4M3 matrix's `[ceil(rows / 128)][columns / 128]` f32 inverse scales.
    pub scales: Option<Deviceptr>,
    /// Output rows.
    pub rows: usize,
    /// Inner columns.
    pub columns: usize,
    /// The first rows of the second, third and fourth parts (`rows` for a part that is absent).
    pub parts: [usize; 3],
    /// A packed copy of a BF16 matrix's rows ([crate::packed]): its four tiled planes.
    pub packed: Option<[u64; 4]>,
}

impl TiledMatrix {
    /// A one-part matrix.
    #[must_use]
    pub const fn whole(
        weight: Deviceptr,
        scales: Option<Deviceptr>,
        rows: usize,
        columns: usize,
    ) -> Self {
        Self {
            weight,
            scales,
            rows,
            columns,
            parts: [rows; 3],
            packed: None,
        }
    }

    const fn element(&self) -> usize {
        if self.scales.is_some() { 1 } else { 2 }
    }

    /// Part `first..end`'s rows as a one-part matrix.
    fn part(&self, first: usize, end: usize) -> Self {
        let scales = self.scales.map(|scales| {
            scales.offset((first / FP8_BLOCK) * (self.columns / FP8_BLOCK) * 4)
        });
        Self::whole(
            self.weight.offset(first * self.columns * self.element()),
            scales,
            end - first,
            self.columns,
        )
    }

    /// Each non-empty part's first and end rows.
    fn spans(&self) -> impl Iterator<Item = (usize, usize)> {
        let starts = [0, self.parts[0], self.parts[1], self.parts[2], self.rows];
        (0..4)
            .map(move |part| (starts[part], starts[part + 1]))
            .filter(|(first, end)| end > first)
    }
}

/// The device scratch one rank's tiled projections share.
#[derive(Debug, Clone, Copy)]
pub struct TiledScratch {
    /// The GEMV's split partials: [TiledScratch::gemv_bytes].
    pub partials: Deviceptr,
    /// The GEMV's tile counters, zeroed once: [TiledScratch::gemv_bytes].
    pub counters: Deviceptr,
    /// A bulk input's E4M3 codes, one byte a value.
    pub codes: Deviceptr,
    /// Their scales, one f32 a 128-column block.
    pub code_scales: Deviceptr,
    /// The most values the codes hold.
    pub code_capacity: usize,
}

impl TiledScratch {
    /// The bytes of the partials and of the counters.
    #[must_use]
    pub const fn gemv_bytes() -> (usize, usize) {
        (
            TILED_GEMV_PARTIALS * GEMV_GROUPS * 4,
            TILED_GEMV_BLOCKS * GEMV_GROUPS * 4,
        )
    }

    /// The bytes of the codes and of their scales for `values` values.
    #[must_use]
    pub const fn code_bytes(values: usize) -> (usize, usize) {
        (values, values / FP8_BLOCK * 4)
    }
}

/// The tiled projection's kernels, loaded at open.
#[derive(Debug, Clone, Copy)]
pub struct TiledProjection {
    /// `q4e_gemv_tiled_bf16_parts`, `q4e_gemv_tiled_e4m3_block`, `q4e_gemv_tiled_packed_parts`.
    gemv: [Bound; 3],
    /// `glm_quantise_rows`, `glm_prefill_fp8b128_mma(_bf16a)`, `glm_prefill_exact_bf16`.
    prefill: [Bound; 4],
    /// `glm_prefill_skinny_bf16_1..4`: eight output columns a step.
    skinny: [Bound; 4],
    /// `glm_prefill_q8_mma`, `glm_prefill_bf16_wide(_bf16a)`.
    shared: [Bound; 3],
    /// `inkling_draft_round`: values rounded to BF16 in place.
    hold: Bound,
    /// Whether the release quantises its FP8 linears' input.
    quantise: bool,
    /// The bulk input's reading.
    activation: TiledActivation,
}

impl TiledProjection {
    /// The row GEMVs.
    pub const GEMV: KernelTable<3> = KernelTable {
        image: Image::GlmDecode,
        symbols: [
            ("q4e_gemv_tiled_bf16_parts", Precision::Bf16),
            ("q4e_gemv_tiled_e4m3_block", Precision::Fp8E4M3),
            ("q4e_gemv_tiled_packed_parts", Precision::Bf16),
        ],
    };

    /// The bulk bodies without a shared-memory reservation.
    pub const PREFILL: KernelTable<8> = KernelTable {
        image: Image::GlmPrefill,
        symbols: [
            ("glm_quantise_rows", Precision::F32),
            ("glm_prefill_fp8b128_mma", Precision::Fp8E4M3),
            ("glm_prefill_fp8b128_mma_bf16a", Precision::Fp8E4M3),
            ("glm_prefill_exact_bf16", Precision::Bf16),
            ("glm_prefill_skinny_bf16_1", Precision::Bf16),
            ("glm_prefill_skinny_bf16_2", Precision::Bf16),
            ("glm_prefill_skinny_bf16_3", Precision::Bf16),
            ("glm_prefill_skinny_bf16_4", Precision::Bf16),
        ],
    };

    /// The rounding a held projection takes.
    pub const HOLD: KernelTable<1> = KernelTable {
        image: Image::Inkling,
        symbols: [("inkling_draft_round", Precision::F32)],
    };

    /// The bodies loaded with a shared-memory reservation: symbol, arithmetic, bytes.
    const SHARED: [(&'static str, Precision, u32); 3] = [
        ("glm_prefill_q8_mma", Precision::Fp8E4M3, Q8_SHARED),
        ("glm_prefill_bf16_wide", Precision::Bf16, WIDE_SHARED),
        ("glm_prefill_bf16_wide_bf16a", Precision::Bf16, WIDE_SHARED),
    ];

    /// Loads the kernels. `quantise` is the release's dynamic activation scheme; an
    /// [TiledActivation::Fp8] reading of a release without one is refused by name.
    ///
    /// # Errors
    /// That refusal, an image this build does not embed, or a symbol it does not export.
    pub fn open(cuda: &Cuda, quantise: bool, activation: TiledActivation) -> Result<Self, String> {
        if activation == TiledActivation::Fp8 && !quantise {
            return Err(
                "prefill.activation=fp8 quantises activations this release does not; take split \
                 or bf16"
                    .to_owned(),
            );
        }
        let gemv = Self::GEMV.load(cuda)?;
        let [quantiser, fp8, fp8_bf16a, exact, s1, s2, s3, s4] = Self::PREFILL.load(cuda)?;
        let image = Image::GlmPrefill;
        let mut shared = Vec::with_capacity(3);
        for (symbol, precision, bytes) in Self::SHARED {
            shared.push(Bound::shared(
                cuda,
                image,
                image.bytes(),
                symbol,
                precision,
                bytes,
            )?);
        }
        let [hold] = Self::HOLD.load(cuda)?;
        Ok(Self {
            gemv,
            prefill: [quantiser, fp8, fp8_bf16a, exact],
            skinny: [s1, s2, s3, s4],
            shared: [shared[0], shared[1], shared[2]],
            hold,
            quantise,
            activation,
        })
    }

    /// The default reading for a release: [TiledActivation::Fp8] when it quantises, else
    /// [TiledActivation::Split].
    #[must_use]
    pub const fn default_activation(quantise: bool) -> TiledActivation {
        if quantise {
            TiledActivation::Fp8
        } else {
            TiledActivation::Split
        }
    }

    /// `output[rows][matrix.rows] = input[rows][matrix.columns] x matrixᵀ`, both f32, through
    /// `scratch`: the row GEMV up to [GEMV_MAX_BATCH] rows, the bulk tiles past that (the row
    /// GEMV's bits when `exact`), then rounded to BF16 in place when `held`.
    ///
    /// # Errors
    /// An input that is not 16-byte aligned, a bulk input the scratch cannot quantise, or the
    /// driver's refusal.
    #[allow(clippy::too_many_arguments)]
    pub fn project<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        scratch: &TiledScratch,
        matrix: &TiledMatrix,
        (input, output): (Deviceptr, Deviceptr),
        rows: usize,
        (exact, held): (bool, bool),
    ) -> Result<(), BackendError> {
        if !input.0.is_multiple_of(16) {
            return Err(refusal(
                "a tiled projection's input is not 16-byte aligned; its reads would be torn",
            ));
        }
        if rows <= GEMV_MAX_BATCH {
            self.gemv(cx, scratch, matrix, (input, output), rows)?;
        } else if exact {
            self.exact(cx, scratch, matrix, (input, output), rows)?;
        } else {
            self.bulk(cx, scratch, matrix, (input, output), rows)?;
        }
        if held {
            self.hold(cx, output, rows * matrix.rows)?;
        }
        Ok(())
    }

    /// Rounds `count` f32 values at `values` to BF16 in place.
    ///
    /// # Errors
    /// The driver's refusal.
    pub fn hold<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        values: Deviceptr,
        count: usize,
    ) -> Result<(), BackendError> {
        crate::roles::round_bf16(cx, self.hold, values, count)
    }

    /// The loaded rounding kernel [Self::hold] launches, for a body that rounds its own values.
    #[must_use]
    pub const fn rounding(&self) -> Bound {
        self.hold
    }

    /// The tiled GEMV over up to [GEMV_GROUP_ROWS] rows a launch, a group of [GEMV_MAX_BATCH]
    /// rows a z-block, each the small launch's arithmetic.
    fn gemv<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        scratch: &TiledScratch,
        matrix: &TiledMatrix,
        (input, output): (Deviceptr, Deviceptr),
        rows: usize,
    ) -> Result<(), BackendError> {
        let mut done = 0;
        while done < rows {
            let count = GEMV_GROUP_ROWS.min(rows - done);
            self.gemv_launch(
                cx,
                scratch,
                matrix,
                (
                    input.offset(done * matrix.columns * 4),
                    output.offset(done * matrix.rows * 4),
                ),
                count,
            )?;
            done += count;
        }
        Ok(())
    }

    fn gemv_launch<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        scratch: &TiledScratch,
        matrix: &TiledMatrix,
        (input, output): (Deviceptr, Deviceptr),
        rows: usize,
    ) -> Result<(), BackendError> {
        let tiles = matrix.rows.div_ceil(TILE_ROWS);
        let (kernel, splits, mut args) = match (matrix.scales, matrix.packed) {
            (None, Some(planes)) => (
                self.gemv[2],
                tiled_gemv_splits(tiles, matrix.columns),
                planes[..3].to_vec(),
            ),
            (None, None) => (
                self.gemv[0],
                tiled_gemv_splits(tiles, matrix.columns),
                vec![matrix.weight.0],
            ),
            (Some(scales), _) => (
                self.gemv[1],
                tiled_gemv_e4m3_splits(tiles, matrix.columns),
                vec![matrix.weight.0, scales.0],
            ),
        };
        args.extend([
            input.0,
            output.0,
            matrix.rows as u64,
            matrix.columns as u64,
            rows as u64,
            scratch.partials.0,
            scratch.counters.0,
        ]);
        if matrix.scales.is_some() {
            args.push(u64::from(self.quantise));
        }
        args.extend(matrix.parts.map(|start| start as u64));
        if let (None, Some(planes)) = (matrix.scales, matrix.packed) {
            args.push(planes[3]);
        }
        let groups = rows.div_ceil(GEMV_MAX_BATCH) as u32;
        cx.launch(
            kernel,
            ((tiles as u32, splits as u32, groups), (BLOCK, 1, 1)),
            &args,
        )
    }

    /// The bulk tiles over each part: an FP8 part's input quantised once for all parts.
    fn bulk<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        scratch: &TiledScratch,
        matrix: &TiledMatrix,
        (input, output): (Deviceptr, Deviceptr),
        rows: usize,
    ) -> Result<(), BackendError> {
        let columns = matrix.columns;
        let mut quantised = false;
        for (first, end) in matrix.spans() {
            let part = matrix.part(first, end);
            let out = output.offset(rows * first * 4);
            let Some(scales) = part.scales else {
                match WidePolicy::GLM.bulk(part.rows, columns, rows) {
                    Route::Wide => {
                        let wide = if self.activation == TiledActivation::Bf16 {
                            self.shared[2]
                        } else {
                            self.shared[1]
                        };
                        let tile =
                            WideTile::call(part.weight.0, input.0, out.0, part.rows, columns, rows)
                                .map_err(refusal)?;
                        cx.launch(wide, (tile.grid, (tile.block.0, 1, 1)), &tile.args)?;
                    }
                    Route::Skinny => cx.launch(
                        self.skinny[part.rows.div_ceil(8) - 1],
                        ((rows.div_ceil(16) as u32, 1, 1), (BLOCK, 1, 1)),
                        &[
                            part.weight.0,
                            input.0,
                            out.0,
                            part.rows as u64,
                            columns as u64,
                            rows as u64,
                        ],
                    )?,
                    _ => self.gemv(cx, scratch, &part, (input, out), rows)?,
                }
                continue;
            };
            // The block-scaled kernel indexes its scales a 128-column block at a time, so a
            // scaled part needs whole blocks, not just whole 32-column stages.
            if !fp8_tile_fits(part.rows, columns) || !first.is_multiple_of(FP8_BLOCK) {
                self.gemv(cx, scratch, &part, (input, out), rows)?;
                continue;
            }
            let grid = Fp8Tile::grid(part.rows, rows);
            if self.activation == TiledActivation::Fp8 && columns.is_multiple_of(FP8_BLOCK) {
                if columns > Q8_MAX_INNER {
                    return Err(refusal(format!(
                        "an FP8-activation prefill GEMM over {columns} columns is wider than its \
                         {Q8_MAX_INNER}; prefill.activation=split takes it"
                    )));
                }
                if !quantised {
                    self.quantise(cx, scratch, input, rows, columns)?;
                    quantised = true;
                }
                cx.launch(
                    self.shared[0],
                    (grid, (BLOCK, 1, 1)),
                    &[
                        part.weight.0,
                        scales.0,
                        scratch.codes.0,
                        scratch.code_scales.0,
                        out.0,
                        part.rows as u64,
                        columns as u64,
                        rows as u64,
                    ],
                )?;
            } else {
                let fp8 = if self.activation == TiledActivation::Bf16 {
                    self.prefill[2]
                } else {
                    self.prefill[1]
                };
                cx.launch(
                    fp8,
                    (grid, (BLOCK, 1, 1)),
                    &[
                        part.weight.0,
                        scales.0,
                        input.0,
                        out.0,
                        part.rows as u64,
                        columns as u64,
                        rows as u64,
                    ],
                )?;
            }
        }
        Ok(())
    }

    /// `rows` rows of `columns` f32 at `input` as the release's dynamic activation, into the
    /// scratch's E4M3 codes and per-128-column scales.
    fn quantise<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        scratch: &TiledScratch,
        input: Deviceptr,
        rows: usize,
        columns: usize,
    ) -> Result<(), BackendError> {
        if !columns.is_multiple_of(FP8_BLOCK) || rows * columns > scratch.code_capacity {
            return Err(refusal(format!(
                "the prefill quantises whole 128-column blocks of at most {} values, not {rows} \
                 x {columns}",
                scratch.code_capacity
            )));
        }
        // One warp a 128-column block.
        let warps = rows * columns / FP8_BLOCK;
        cx.launch(
            self.prefill[0],
            (((warps * 32).div_ceil(BLOCK as usize) as u32, 1, 1), (BLOCK, 1, 1)),
            &[
                input.0,
                scratch.codes.0,
                scratch.code_scales.0,
                rows as u64,
                columns as u64,
            ],
        )
    }

    /// The row GEMV's bits over more rows than its batch: each part on its own, a BF16 part
    /// the tiled GEMV splits one 256-column round a block on `glm_prefill_exact_bf16`, which
    /// computes the same floats.
    fn exact<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        scratch: &TiledScratch,
        matrix: &TiledMatrix,
        (input, output): (Deviceptr, Deviceptr),
        rows: usize,
    ) -> Result<(), BackendError> {
        if rows <= GEMV_GROUP_ROWS {
            return self.gemv_launch(cx, scratch, matrix, (input, output), rows);
        }
        let columns = matrix.columns;
        for (first, end) in matrix.spans() {
            let part = matrix.part(first, end);
            let out = output.offset(rows * first * 4);
            let rounds = columns / TILED_GEMV_ROUND;
            if part.scales.is_none()
                && columns.is_multiple_of(TILED_GEMV_ROUND)
                && part.weight.0.is_multiple_of(16)
                && tiled_gemv_splits(part.rows.div_ceil(TILE_ROWS), columns) == rounds
            {
                cx.launch(
                    self.prefill[3],
                    (
                        (
                            part.rows.div_ceil(EXACT_OUTS) as u32,
                            rows.div_ceil(EXACT_ROWS) as u32,
                            1,
                        ),
                        (EXACT_THREADS, 1, 1),
                    ),
                    &[
                        part.weight.0,
                        input.0,
                        out.0,
                        part.rows as u64,
                        columns as u64,
                        rows as u64,
                    ],
                )?;
                continue;
            }
            self.gemv(cx, scratch, &part, (input, out), rows)?;
        }
        Ok(())
    }
}

fn refusal(detail: impl Into<String>) -> BackendError {
    BackendError::new(TILED, detail.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_table_is_carried() {
        TiledProjection::GEMV.check().expect("the row GEMVs");
        TiledProjection::PREFILL.check().expect("the bulk bodies");
        TiledProjection::HOLD.check().expect("the hold");
    }

    #[test]
    fn parts_split_at_their_starts() {
        let mut matrix = TiledMatrix::whole(Deviceptr(0x1000), None, 96, 256);
        assert_eq!(matrix.spans().collect::<Vec<_>>(), vec![(0, 96)]);
        matrix.parts = [32, 64, 96];
        assert_eq!(
            matrix.spans().collect::<Vec<_>>(),
            vec![(0, 32), (32, 64), (64, 96)]
        );
        assert_eq!(matrix.part(32, 64).weight, Deviceptr(0x1000 + 32 * 256 * 2));
    }
}

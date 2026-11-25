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

//! A dense projection whose stored class the released MXFP8 bodies do not read.
//!
//! The base's dense projections are MXFP8 (E4M3 with one power-of-two scale a 32 x 32 block,
//! B12x-tiled at load) and its fastest loops multiply a quantised row by them. A release stored
//! another way — FP8 on 128 x 128 tiles with f32 or BF16 inverse scales, or BF16 row-major — runs
//! on the shared bulk-prefill tile's dense mode (kernels/dense_format_sm121.cu), the body GLM's and
//! Qwen's projections already run on. The reference multiplies the unquantised rows by the
//! decoded weight, so the route reads the rows a site already holds before its quantiser: f32 rows
//! (as a BF16 head and remainder) or BF16 rows as stored, and writes f32 or BF16 rows as the next
//! reader takes them. One launch a projection, or one a group of same-shape projections on the
//! grid's z axis.
//!
//! Which entries a projection launches is the plan's choice, made at open from the catalogue
//! symbols that decode the projection's class ([DenseIo::prefix]); this module only launches them.

use crate::cuda_driver::{Cuda, Kernel};

/// The tile's rows and columns, and its threads.
const TILE_ROWS: usize = 128;
const TILE_COLS: usize = 128;
const TILE_THREADS: u32 = 256;

/// How a row is held on one side of a projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowForm {
    /// f32 values.
    F32,
    /// BF16 values.
    Bf16,
}

impl RowForm {
    const fn bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
}

/// A projection's input and output row forms: which of the class's four entries it launches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseIo {
    /// The rows read.
    pub input: RowForm,
    /// The rows written.
    pub output: RowForm,
}

impl DenseIo {
    /// Every row form pair, in [DenseFormat::entries] order.
    pub const ALL: [Self; 4] = [
        Self::new(RowForm::F32, RowForm::F32),
        Self::new(RowForm::F32, RowForm::Bf16),
        Self::new(RowForm::Bf16, RowForm::F32),
        Self::new(RowForm::Bf16, RowForm::Bf16),
    ];

    /// The pair `input` -> `output`.
    #[must_use]
    pub const fn new(input: RowForm, output: RowForm) -> Self {
        Self { input, output }
    }

    /// The catalogue prefix of this pair's entries (`dense_format_<io>_<class>_sm121`).
    #[must_use]
    pub const fn prefix(self) -> &'static str {
        match (self.input, self.output) {
            (RowForm::F32, RowForm::F32) => "dense_format_f32a_f32o_",
            (RowForm::F32, RowForm::Bf16) => "dense_format_f32a_bf16o_",
            (RowForm::Bf16, RowForm::F32) => "dense_format_bf16a_f32o_",
            (RowForm::Bf16, RowForm::Bf16) => "dense_format_bf16a_bf16o_",
        }
    }

    const fn index(self) -> usize {
        match (self.input, self.output) {
            (RowForm::F32, RowForm::F32) => 0,
            (RowForm::F32, RowForm::Bf16) => 1,
            (RowForm::Bf16, RowForm::F32) => 2,
            (RowForm::Bf16, RowForm::Bf16) => 3,
        }
    }
}

/// One rank-local weight in a dense class the format route reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseMatrix {
    /// `[cols][inner]` row-major weight bytes (group 0's, for a grouped projection).
    pub weight: u64,
    /// The scale plane (one inverse scale a 128 x 128 tile), or 0 for BF16.
    pub scales: u64,
    /// Output columns.
    pub cols: usize,
    /// Reduction length.
    pub inner: usize,
}

/// The rows one launch multiplies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseRows {
    /// `[tokens][a_stride]` input rows; a row's first `inner` values are read.
    pub a: u64,
    /// Values between consecutive input rows.
    pub a_stride: usize,
    /// `[tokens][out_stride]` output rows; a row's first `cols` values are written.
    pub out: u64,
    /// Values between consecutive output rows.
    pub out_stride: usize,
    /// Rows.
    pub tokens: usize,
}

/// A grouped projection: `count` same-shape matrices, group `g` reading `a + g * a` values,
/// writing `out + g * out` values, its weight `g * weight` bytes and its scales `g * scales`
/// scale elements on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseGroups {
    /// Groups.
    pub count: usize,
    /// Input values between groups.
    pub a: u64,
    /// Output values between groups.
    pub out: u64,
    /// Weight bytes between groups.
    pub weight: u64,
    /// Scale elements between groups.
    pub scales: u64,
}

impl DenseGroups {
    /// One matrix.
    pub const ONE: Self = Self {
        count: 1,
        a: 0,
        out: 0,
        weight: 0,
        scales: 0,
    };
}

/// The dense format route's entries for one stored class, one a row form pair.
#[derive(Debug, Clone, Copy)]
pub struct DenseFormat {
    /// `dense_format_<io>_<class>_sm121`, in [DenseIo::ALL] order.
    pub entries: [Kernel; 4],
}

impl DenseFormat {
    /// `out[t][c] = sum_k a[t][k] * W[c][k]` for every row, through the class's `io` entry.
    ///
    /// # Errors
    /// A geometry the tile cannot cut (columns or inner not a multiple of 32, a stride shorter
    /// than its row or not a whole 16-byte load), or the launch fails.
    pub fn project(
        &self,
        cuda: &Cuda,
        io: DenseIo,
        matrix: &DenseMatrix,
        rows: &DenseRows,
        groups: DenseGroups,
    ) -> Result<(), String> {
        if rows.tokens == 0 || groups.count == 0 {
            return Ok(());
        }
        if matrix.cols == 0 || matrix.cols % 32 != 0 || matrix.inner == 0 || matrix.inner % 32 != 0
        {
            return Err(format!(
                "dense format: a {}x{} matrix is not cut by the tile's 32-wide steps",
                matrix.cols, matrix.inner
            ));
        }
        if rows.a_stride < matrix.inner || rows.out_stride < matrix.cols {
            return Err(format!(
                "dense format: row strides {} in / {} out are shorter than the {}x{} matrix's rows",
                rows.a_stride, rows.out_stride, matrix.cols, matrix.inner
            ));
        }
        // The tile reads an input row four values a load and stores two outputs at a time.
        let a_unit = 16 / io.input.bytes();
        if rows.a_stride as u64 % a_unit != 0 || groups.a % a_unit != 0 {
            return Err(format!(
                "dense format: input stride {} / group stride {} is not a whole {a_unit}-value load",
                rows.a_stride, groups.a
            ));
        }
        if rows.out_stride % 2 != 0 || groups.out % 2 != 0 {
            return Err(format!(
                "dense format: output stride {} / group stride {} splits a stored pair",
                rows.out_stride, groups.out
            ));
        }
        let grid = (
            matrix.cols.div_ceil(TILE_COLS) as u32,
            rows.tokens.div_ceil(TILE_ROWS) as u32,
            groups.count as u32,
        );
        cuda.launch(
            self.entries[io.index()],
            grid,
            (TILE_THREADS, 1, 1),
            &[
                rows.a,
                matrix.weight,
                matrix.scales,
                rows.out,
                matrix.cols as u64,
                matrix.inner as u64,
                rows.tokens as u64,
                rows.a_stride as u64,
                rows.out_stride as u64,
                groups.a,
                groups.out,
                groups.weight,
                groups.scales,
            ],
        )
        .map_err(|error| {
            format!(
                "dense format: {}x{} projection ({:?} -> {:?}): {error}",
                matrix.cols, matrix.inner, io.input, io.output
            )
        })
    }
}

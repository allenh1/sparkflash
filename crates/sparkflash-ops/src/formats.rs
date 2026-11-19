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

//! Formats as types, and the scale layout each one needs.
//!
//! A weight is a [Mat]<F> for the format it was uploaded as. The format is a zero-sized type
//! ([Bf16], [Fp8E4m3], [MxFp8], [NvFp4], [F32]), so a kernel body that takes `&Mat<Bf16>` cannot
//! be handed an NVFP4 matrix by mistake: the two are different types and the compiler says so.
//!
//! # One arithmetic image per precision, and it is the tile's
//!
//! A part has one arithmetic image per precision, and that image is the tensor-core tile's. The
//! prompt body and the token body must give the same bits for the same row, or a continued turn
//! differs from a cold one. A few-row body may be a different kernel - the released arms have
//! one-row GEMVs beside their tiles - but it must reproduce the tile's bits. The format marker is
//! what pins the image: [Format::PRECISION] is the arithmetic, not the stored dtype, and a
//! widening is a record ([crate::widen]) rather than a quiet second path.
//!
//! # Scales are layouts, not an afterthought
//!
//! "Carries the scales its format needs" is the whole problem, because four formats carry four
//! different things:
//!
//! | Format | Values | Scales |
//! |---|---|---|
//! | [Bf16] | BF16, or the lossless packed form (two columns a 32-bit word) | none; the packed form's word layout is [PackedBf16] |
//! | [Fp8E4m3] | E4M3, one byte a value | one inverse scale a 128 x 128 tile |
//! | [MxFp8] | E4M3, one byte a value | one E8M0 (power-of-two) code a 32 x 32 block |
//! | [NvFp4] | E2M1, two nibbles a byte | one E4M3 scale a 16-value group, plus one global f32 |
//! | [F32] | f32 | none |
//!
//! The layouts are spelled out in [ScaleLayout], and [Mat] holds a [Scales] value whose variant
//! is the layout it was uploaded with. [upload] checks the container's stored dtype with
//! [sparkflash_weights::TensorSpec::is_stored_as] and checks the scale entry count before it
//! hands back a matrix; a mismatch names both formats in the error.
//!
//! The packed BF16 form has no numeric scale plane. It is included here because its row layout
//! is the one thing a `Mat<Bf16>` has to know that a bare BF16 pointer does not: a 32-bit word
//! carries two columns' BF16 bits, built from a mantissa plane, a 4-bit exponent-offset plane and
//! a per-step top-exponent plane, with escapes in a side list. See [PackedBf16]. [Bf16]'s
//! [Format::SCALES] names that word layout; an unpacked checkpoint matrix uploads with
//! [Scales::None], which [upload] accepts because the packed form is a lossless re-encoding of
//! the same arithmetic.
//!
//! Activations stay f32; that is the accumulation format, not a storage choice.

use std::marker::PhantomData;

use sparkflash_kernels::cuda_driver::Deviceptr;
use sparkflash_weights::{DType as StoredDType, TensorSpec};
use thiserror::Error;

use crate::plan::Precision;

mod sealed {
    /// Keeps [Format] closed: a new format is a marker in this module, not an implementation a
    /// model crate can add and a body can then silently accept.
    pub trait Sealed {}
}

/// A stored weight format.
///
/// The markers are zero-sized and sealed. Each names the dtype a checkpoint stores, the logical
/// dtype the arithmetic reads, the scale layout it needs, and the arithmetic precision a body
/// resolved for it runs at.
pub trait Format: sealed::Sealed + Copy + 'static {
    /// The marker's identity, for a ledger or an error that has to name a format.
    const KIND: FormatKind;
    /// The dtype the weight container reports for this format.
    const STORED: StoredDType;
    /// The logical element dtype the arithmetic sees. An NVFP4 tensor is stored as `U8` and
    /// read as `F4E2M1`; every other format's two dtypes agree.
    const LOGICAL: StoredDType;
    /// The scale layout the format needs.
    const SCALES: ScaleLayout;
    /// The arithmetic a body for this format accumulates in.
    const PRECISION: Precision;
}

/// The format markers by identity, for a ledger or an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatKind {
    /// Unquantized bf16, or its lossless packed form.
    Bf16,
    /// FP8 E4M3 with one inverse scale a 128 x 128 tile.
    Fp8E4m3,
    /// MXFP8: E4M3 codes with one E8M0 power-of-two scale a 32 x 32 block.
    MxFp8,
    /// NVFP4: E2M1 codes with one E4M3 scale a 16-value group and a global f32.
    NvFp4,
    /// IEEE binary32.
    F32,
}

impl FormatKind {
    /// The name a ledger and an error use.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8E4m3 => "fp8-e4m3",
            Self::MxFp8 => "mx-fp8",
            Self::NvFp4 => "nvfp4",
            Self::F32 => "f32",
        }
    }

    /// The marker's identity.
    #[must_use]
    pub const fn of<F: Format>() -> Self {
        F::KIND
    }

    /// The dtype a checkpoint stores this format's values as.
    #[must_use]
    pub const fn stored(self) -> StoredDType {
        match self {
            Self::Bf16 => StoredDType::BF16,
            Self::Fp8E4m3 | Self::MxFp8 => StoredDType::F8E4M3,
            Self::NvFp4 => StoredDType::U8,
            Self::F32 => StoredDType::F32,
        }
    }

    /// The logical element dtype the arithmetic sees.
    #[must_use]
    pub const fn logical(self) -> StoredDType {
        match self {
            Self::NvFp4 => StoredDType::F4E2M1,
            other => other.stored(),
        }
    }

    /// The scale layout this format needs.
    #[must_use]
    pub const fn scales(self) -> ScaleLayout {
        match self {
            Self::Bf16 => ScaleLayout::PackedBf16 {
                columns_per_word: 2,
            },
            Self::Fp8E4m3 => ScaleLayout::Fp8Tile {
                rows: 128,
                columns: 128,
            },
            Self::MxFp8 => ScaleLayout::MxFp8 {
                rows: 32,
                columns: 32,
            },
            Self::NvFp4 => ScaleLayout::NvFp4 { group: 16 },
            Self::F32 => ScaleLayout::None,
        }
    }

    /// The arithmetic precision a body for this format runs at.
    #[must_use]
    pub const fn precision(self) -> Precision {
        match self {
            Self::Bf16 => Precision::Bf16,
            Self::Fp8E4m3 => Precision::Fp8E4M3,
            Self::MxFp8 => Precision::MxFp8,
            Self::NvFp4 => Precision::NvFp4,
            Self::F32 => Precision::F32,
        }
    }
}

/// Unquantized bf16, or its lossless packed form.
///
/// The marker's [Format::SCALES] is the packed form's word layout ([ScaleLayout::PackedBf16]),
/// because that is the row geometry a reader needs when the form is used. An unpacked checkpoint
/// matrix carries [Scales::None]; [upload] accepts it because the packed form is lossless.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16;
impl sealed::Sealed for Bf16 {}
impl Format for Bf16 {
    const KIND: FormatKind = FormatKind::Bf16;
    const STORED: StoredDType = StoredDType::BF16;
    const LOGICAL: StoredDType = StoredDType::BF16;
    const SCALES: ScaleLayout = ScaleLayout::PackedBf16 {
        columns_per_word: 2,
    };
    const PRECISION: Precision = Precision::Bf16;
}

/// FP8 E4M3 with one inverse scale per 128 x 128 tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp8E4m3;
impl sealed::Sealed for Fp8E4m3 {}
impl Format for Fp8E4m3 {
    const KIND: FormatKind = FormatKind::Fp8E4m3;
    const STORED: StoredDType = StoredDType::F8E4M3;
    const LOGICAL: StoredDType = StoredDType::F8E4M3;
    const SCALES: ScaleLayout = ScaleLayout::Fp8Tile {
        rows: 128,
        columns: 128,
    };
    const PRECISION: Precision = Precision::Fp8E4M3;
}

/// MXFP8: E4M3 codes with one E8M0 power-of-two scale per 32 x 32 block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MxFp8;
impl sealed::Sealed for MxFp8 {}
impl Format for MxFp8 {
    const KIND: FormatKind = FormatKind::MxFp8;
    const STORED: StoredDType = StoredDType::F8E4M3;
    const LOGICAL: StoredDType = StoredDType::F8E4M3;
    const SCALES: ScaleLayout = ScaleLayout::MxFp8 {
        rows: 32,
        columns: 32,
    };
    const PRECISION: Precision = Precision::MxFp8;
}

/// NVFP4: E2M1 codes with one E4M3 scale per 16-value group and one global f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvFp4;
impl sealed::Sealed for NvFp4 {}
impl Format for NvFp4 {
    const KIND: FormatKind = FormatKind::NvFp4;
    const STORED: StoredDType = StoredDType::U8;
    const LOGICAL: StoredDType = StoredDType::F4E2M1;
    const SCALES: ScaleLayout = ScaleLayout::NvFp4 { group: 16 };
    const PRECISION: Precision = Precision::NvFp4;
}

/// IEEE binary32, with no scale plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F32;
impl sealed::Sealed for F32 {}
impl Format for F32 {
    const KIND: FormatKind = FormatKind::F32;
    const STORED: StoredDType = StoredDType::F32;
    const LOGICAL: StoredDType = StoredDType::F32;
    const SCALES: ScaleLayout = ScaleLayout::None;
    const PRECISION: Precision = Precision::F32;
}

/// The shape of a format's scale plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleLayout {
    /// No scale plane. The packed BF16 form is not a scale plane either: its word layout is
    /// [PackedBf16].
    None,
    /// NVFP4: `weight_scale` is `[rows, ceil(columns / group)]` E4M3 scales, one a group of
    /// **16** values along the row. The per-matrix global is `weight_scale_2`, one f32; the
    /// static `input_scale` also one f32 belongs to the activation, not to the matrix.
    NvFp4 {
        /// Values one block scale covers.
        group: usize,
    },
    /// MXFP8: `weight_scale` is `[ceil(rows / rows), ceil(columns / columns)]` E8M0 codes.
    /// One code is a power of two, `2^(code - 127)`, and covers a **32 x 32** block. This is
    /// DeepSeek's dense grid (`[ceil(out/32), ceil(in/32)]`); its routed experts' per-row
    /// MXFP4 grid is a different format.
    MxFp8 {
        /// Rows one code covers.
        rows: usize,
        /// Columns one code covers.
        columns: usize,
    },
    /// FP8 E4M3: `weight_scale_inv` is `[ceil(rows / rows), ceil(columns / columns)]`, one
    /// inverse scale a **128 x 128** tile. The scalar is f32 in GLM's release and bf16 in
    /// Qwen's; the [Scales] value carries which.
    Fp8Tile {
        /// Rows one scale covers.
        rows: usize,
        /// Columns one scale covers.
        columns: usize,
    },
    /// The lossless packed BF16 form. A 32-bit word carries **two columns**' BF16 bits; there
    /// is no numeric scale. A packed row is plane A (columns bytes of sign/mantissa), plane B
    /// (`columns / 2` bytes of 4-bit exponent offsets, two a byte), then plane C (32 top
    /// exponent bytes a step), padded to 16 bytes; escapes are `column << 16 | word` entries.
    PackedBf16 {
        /// Columns a 32-bit word holds.
        columns_per_word: usize,
    },
}

impl ScaleLayout {
    /// The number of block-scale entries the layout needs for a matrix.
    #[must_use]
    pub fn entries(self, rows: usize, columns: usize) -> usize {
        match self {
            Self::None | Self::PackedBf16 { .. } => 0,
            Self::NvFp4 { group } => rows.saturating_mul(columns.div_ceil(group.max(1))),
            Self::MxFp8 {
                rows: block_rows,
                columns: block_columns,
            } => rows
                .div_ceil(block_rows.max(1))
                .saturating_mul(columns.div_ceil(block_columns.max(1))),
            Self::Fp8Tile {
                rows: tile_rows,
                columns: tile_columns,
            } => rows
                .div_ceil(tile_rows.max(1))
                .saturating_mul(columns.div_ceil(tile_columns.max(1))),
        }
    }

    /// The dtype one block-scale entry is stored as, when the layout has one.
    #[must_use]
    pub const fn block_scale_dtype(self) -> Option<StoredDType> {
        match self {
            Self::NvFp4 { .. } => Some(StoredDType::F8E4M3),
            Self::MxFp8 { .. } => Some(StoredDType::F8E8M0),
            Self::Fp8Tile { .. } => None,
            Self::None | Self::PackedBf16 { .. } => None,
        }
    }

    /// The layout's name in a diagnostic.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::NvFp4 { .. } => "nvfp4-per-16",
            Self::MxFp8 { .. } => "mxfp8-per-32x32",
            Self::Fp8Tile { .. } => "e4m3-per-128x128",
            Self::PackedBf16 { .. } => "packed-bf16-words",
        }
    }
}

/// The scales a [Mat] carries.
///
/// A variant names the layout and holds the device address of the block plane. [Scales::None]
/// is every format without a plane; [Scales::PackedBf16] is the packed word layout, which is
/// not a numeric scale but is the thing a BF16 reader has to know.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scales {
    /// No scale plane: F32, or a BF16 checkpoint read in its unpacked form.
    None,
    /// NVFP4's per-16 E4M3 block scales, and the one f32 global that multiplies the matrix.
    NvFp4 {
        /// The device address of `weight_scale`.
        blocks: Deviceptr,
        /// The E4M3 scale entries in that plane.
        entries: usize,
        /// `weight_scale_2`: the matrix's global multiplier.
        global: f32,
    },
    /// MXFP8's per-32 x 32 E8M0 block codes.
    MxFp8 {
        /// The device address of `weight_scale`.
        blocks: Deviceptr,
        /// The E8M0 entries in that plane.
        entries: usize,
    },
    /// FP8 E4M3's per-128 x 128 inverse scales.
    Fp8E4m3 {
        /// The device address of `weight_scale_inv`.
        blocks: Deviceptr,
        /// The inverse-scale entries in that plane.
        entries: usize,
        /// The scalar dtype of one entry (f32 or bf16 in the released releases).
        dtype: StoredDType,
    },
    /// The lossless packed BF16 form's word layout.
    PackedBf16(PackedBf16),
}

impl Scales {
    /// No scale plane.
    #[must_use]
    pub const fn none() -> Self {
        Self::None
    }

    /// NVFP4's per-16 block plane and its global.
    #[must_use]
    pub const fn nvfp4(blocks: Deviceptr, entries: usize, global: f32) -> Self {
        Self::NvFp4 {
            blocks,
            entries,
            global,
        }
    }

    /// MXFP8's per-32 x 32 block plane.
    #[must_use]
    pub const fn mxfp8(blocks: Deviceptr, entries: usize) -> Self {
        Self::MxFp8 { blocks, entries }
    }

    /// FP8 E4M3's per-tile inverse-scale plane.
    #[must_use]
    pub const fn fp8_e4m3(blocks: Deviceptr, entries: usize, dtype: StoredDType) -> Self {
        Self::Fp8E4m3 {
            blocks,
            entries,
            dtype,
        }
    }

    /// The packed BF16 word layout.
    #[must_use]
    pub const fn packed_bf16(packed: PackedBf16) -> Self {
        Self::PackedBf16(packed)
    }

    /// The layout this value describes.
    #[must_use]
    pub fn layout(&self) -> ScaleLayout {
        match self {
            Self::None => ScaleLayout::None,
            Self::NvFp4 { .. } => ScaleLayout::NvFp4 { group: 16 },
            Self::MxFp8 { .. } => ScaleLayout::MxFp8 {
                rows: 32,
                columns: 32,
            },
            Self::Fp8E4m3 { .. } => ScaleLayout::Fp8Tile {
                rows: 128,
                columns: 128,
            },
            Self::PackedBf16(_) => ScaleLayout::PackedBf16 {
                columns_per_word: 2,
            },
        }
    }

    /// The number of block-scale entries stored.
    #[must_use]
    pub const fn entries(&self) -> usize {
        match self {
            Self::None | Self::PackedBf16(_) => 0,
            Self::NvFp4 { entries, .. }
            | Self::MxFp8 { entries, .. }
            | Self::Fp8E4m3 { entries, .. } => *entries,
        }
    }

    /// The block plane's device address, when the layout has one.
    #[must_use]
    pub const fn blocks(&self) -> Option<Deviceptr> {
        match self {
            Self::None | Self::PackedBf16(_) => None,
            Self::NvFp4 { blocks, .. }
            | Self::MxFp8 { blocks, .. }
            | Self::Fp8E4m3 { blocks, .. } => Some(*blocks),
        }
    }
}

/// The packed BF16 form's word layout (`kernels/bf16_packed_sm121.cuh`).
///
/// A [Mat]<[Bf16]> whose [Scales] is this variant points [Mat::data] at plane A of the packed
/// rows and reports the packed row stride. A row is `[A: columns][B: columns / 2][C: 32 a
/// step]`, padded to 16 bytes; a 32-bit word of the reconstructed row holds two columns. The
/// escape list holds the words whose exponent is 15 or more binades under their step's largest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedBf16 {
    /// The escape list: `column << 16 | word`, one u32 an escape.
    pub escapes: Deviceptr,
    /// The row offsets into [PackedBf16::escapes], `rows + 1` u32.
    pub offsets: Deviceptr,
    /// The bytes of one packed row.
    pub row_bytes: usize,
    /// Which order the plane bytes take; the two hold the same words and bits.
    pub order: PackedOrder,
}

/// The two orderings of the packed BF16 plane bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedOrder {
    /// The row walk's order: lane `lane`'s chunks together (the Inkling, Dory and dense arms).
    Walk,
    /// The tiled reader's order: a step's chunks by chunk (GLM's projections, Qwen's few-row
    /// tile).
    Tiled,
}

/// One stored tensor on its way to a [Mat].
///
/// [Tensor::stored] is the container's view - for an NVFP4 weight that is `U8` with the last
/// dimension halved - and [Tensor::shape] is the logical shape the arithmetic sees. [upload]
/// checks the two are the same tensor with
/// [sparkflash_weights::TensorSpec::is_stored_as], so a caller cannot relabel a BF16 matrix as
/// FP4 by passing a different shape.
#[derive(Debug, Clone, Copy)]
pub struct Tensor<'a> {
    /// The tensor as the weight container reports it.
    pub stored: &'a TensorSpec,
    /// The logical shape the arithmetic sees.
    pub shape: &'a [usize],
    /// The device address of the stored values. For the packed BF16 form this is plane A.
    pub data: Deviceptr,
    /// The byte stride from one row's start to the next.
    pub stride: usize,
    /// The scale planes the format needs.
    pub scales: Scales,
}

/// A typed matrix: one tensor's values, its shape, and the scales its format needs.
///
/// A body that takes `&Mat<F>` was resolved for `F` and runs at [Format::PRECISION]. Handing it
/// a matrix of another format is a compile error:
///
/// ```compile_fail,E0308
/// use sparkflash_ops::formats::{Bf16, Mat, NvFp4};
///
/// fn bf16_body(_: &Mat<Bf16>) {}
/// let weight: Mat<NvFp4> = todo!("the uploaded NVFP4 matrix");
/// bf16_body(&weight);
/// ```
#[derive(Debug, Clone)]
pub struct Mat<F: Format> {
    spec: TensorSpec,
    data: Deviceptr,
    rows: usize,
    columns: usize,
    stride: usize,
    scales: Scales,
    format: PhantomData<F>,
}

impl<F: Format> Mat<F> {
    /// The logical tensor spec: name, arithmetic dtype and shape.
    #[must_use]
    pub fn spec(&self) -> &TensorSpec {
        &self.spec
    }

    /// The device address of the stored values (plane A for the packed BF16 form).
    #[must_use]
    pub const fn data(&self) -> Deviceptr {
        self.data
    }

    /// The logical rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// The logical columns.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// The byte stride from one row's start to the next.
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// The scales this matrix carries.
    #[must_use]
    pub const fn scales(&self) -> &Scales {
        &self.scales
    }

    /// The format marker's identity.
    #[must_use]
    pub const fn format(&self) -> FormatKind {
        F::KIND
    }

    /// The dtype the checkpoint stores this matrix's values as.
    #[must_use]
    pub const fn stored_dtype(&self) -> StoredDType {
        F::STORED
    }

    /// The arithmetic precision a bound body runs.
    #[must_use]
    pub const fn precision(&self) -> Precision {
        F::PRECISION
    }

    /// The stored bytes a token reads from this matrix: every row once.
    ///
    /// This is the floor a widening must not inflate without the ledger saying so; see
    /// [crate::widen::bytes_per_token_floor].
    #[must_use]
    pub fn bytes_per_token(&self) -> u128 {
        (self.stride as u128) * (self.rows as u128)
    }
}

/// Upload one stored tensor as a typed matrix.
///
/// The checks are the two a bare device pointer cannot make:
///
/// 1. the container's dtype and shape are this format's stored form, its packing, or the
///    logical shape with the container's unit dimensions dropped
///    ([sparkflash_weights::TensorSpec::is_stored_as]);
/// 2. the scale plane has exactly the entry count the layout needs
///    ([ScaleLayout::entries]).
///
/// A mismatch names both the format the tensor is stored as and the format the `Mat` is.
///
/// # Errors
/// A non-matrix shape, a stored dtype or shape of another format, a scale layout or count of
/// another format, a scale dtype the FP8 tile does not take, or a packed BF16 row whose stride
/// is not the form's.
pub fn upload<F: Format>(tensor: Tensor<'_>) -> Result<Mat<F>, UploadError> {
    let name = tensor.stored.name.clone();
    let expected = F::KIND;
    if tensor.shape.len() != 2 {
        return Err(UploadError::NotAMatrix {
            name,
            expected: expected.name(),
            shape: tensor.shape.to_vec(),
        });
    }
    let rows = tensor.shape[0];
    let columns = tensor.shape[1];
    let logical = TensorSpec {
        name: name.clone(),
        dtype: F::LOGICAL,
        shape: tensor.shape.to_vec(),
    };
    let actual = tensor.scales.layout();
    let stored = stored_format_name(tensor.stored.dtype, actual);
    if !logical.is_stored_as(tensor.stored) {
        return Err(UploadError::StoredFormat {
            name,
            expected: expected.name(),
            stored,
            dtype: tensor.stored.dtype.name(),
            shape: tensor.stored.shape.clone(),
        });
    }
    // A BF16 checkpoint may be uploaded unpacked, with no word layout, because the packed form
    // is a lossless re-encoding of the same arithmetic.
    let unpacked_bf16 = F::KIND == FormatKind::Bf16 && actual == ScaleLayout::None;
    if actual != F::SCALES && !unpacked_bf16 {
        return Err(UploadError::ScaleLayout {
            name,
            expected: expected.name(),
            stored,
            actual: actual.name(),
            wanted: F::SCALES.name(),
        });
    }
    let wanted = F::SCALES.entries(rows, columns);
    if tensor.scales.entries() != wanted {
        return Err(UploadError::ScaleCount {
            name,
            expected: expected.name(),
            stored,
            actual: tensor.scales.entries(),
            actual_layout: actual.name(),
            wanted,
            wanted_layout: F::SCALES.name(),
        });
    }
    if let Scales::Fp8E4m3 { dtype, .. } = &tensor.scales
        && !matches!(dtype, StoredDType::F32 | StoredDType::BF16)
    {
        return Err(UploadError::ScaleDtype {
            name,
            expected: expected.name(),
            dtype: dtype.name(),
        });
    }
    if let Scales::PackedBf16(packed) = &tensor.scales {
        if !crate::packed::packable(rows, columns) {
            return Err(UploadError::Packed {
                name,
                expected: expected.name(),
                detail: format!(
                    "a packed {} x {} matrix, whose columns are a multiple of 256",
                    rows, columns
                ),
            });
        }
        let wanted_row = crate::packed::row_bytes(columns);
        if packed.row_bytes != wanted_row {
            return Err(UploadError::Packed {
                name,
                expected: expected.name(),
                detail: format!(
                    "a {} x {} row of {wanted_row} bytes, not {}",
                    rows, columns, packed.row_bytes
                ),
            });
        }
        if tensor.stride != wanted_row {
            return Err(UploadError::Packed {
                name,
                expected: expected.name(),
                detail: format!(
                    "a row stride of {wanted_row} packed bytes, not {}",
                    tensor.stride
                ),
            });
        }
    }
    Ok(Mat {
        spec: logical,
        data: tensor.data,
        rows,
        columns,
        stride: tensor.stride,
        scales: tensor.scales,
        format: PhantomData,
    })
}

/// A stored tensor the uploader refuses.
#[derive(Debug, Error)]
pub enum UploadError {
    /// The logical shape is not a matrix.
    #[error(
        "upload::<{expected}> refuses `{name}`: a Mat is a matrix, and the logical shape is          {shape:?}"
    )]
    NotAMatrix {
        /// The tensor's name.
        name: String,
        /// The format the `Mat` is.
        expected: &'static str,
        /// The logical shape supplied.
        shape: Vec<usize>,
    },
    /// The container stores another format.
    #[error(
        "upload::<{expected}> refuses `{name}`: the tensor is stored as {stored} ({dtype} {shape:?}), and the Mat is {expected}"
    )]
    StoredFormat {
        /// The tensor's name.
        name: String,
        /// The format the `Mat` is.
        expected: &'static str,
        /// The format the container's dtype and scales name.
        stored: String,
        /// The container's dtype name.
        dtype: &'static str,
        /// The container's shape.
        shape: Vec<usize>,
    },
    /// The scale planes are a different layout.
    #[error(
        "upload::<{expected}> refuses `{name}`: the tensor is stored as {stored} with {actual} scales, and {expected} needs {wanted}"
    )]
    ScaleLayout {
        /// The tensor's name.
        name: String,
        /// The format the `Mat` is.
        expected: &'static str,
        /// The format the container's dtype and scales name.
        stored: String,
        /// The layout supplied.
        actual: &'static str,
        /// The layout the format needs.
        wanted: &'static str,
    },
    /// The scale plane has the wrong entry count.
    #[error(
        "upload::<{expected}> refuses `{name}`: the tensor is stored as {stored}, whose {actual_layout} plane has {actual} entries, and {expected} needs {wanted} ({wanted_layout})"
    )]
    ScaleCount {
        /// The tensor's name.
        name: String,
        /// The format the `Mat` is.
        expected: &'static str,
        /// The format the container's dtype and scales name.
        stored: String,
        /// The entries supplied.
        actual: usize,
        /// The layout supplied.
        actual_layout: &'static str,
        /// The entries the format needs.
        wanted: usize,
        /// The layout the format needs.
        wanted_layout: &'static str,
    },
    /// The FP8 tile's inverse scales are not a scalar the tile takes.
    #[error("upload::<{expected}> refuses `{name}`: an FP8 tile scale is f32 or bf16, not {dtype}")]
    ScaleDtype {
        /// The tensor's name.
        name: String,
        /// The format the `Mat` is.
        expected: &'static str,
        /// The dtype supplied.
        dtype: &'static str,
    },
    /// The packed BF16 form does not hold this row.
    #[error("upload::<{expected}> refuses `{name}`: the packed BF16 form needs {detail}")]
    Packed {
        /// The tensor's name.
        name: String,
        /// The format the `Mat` is.
        expected: &'static str,
        /// What the form needs.
        detail: String,
    },
}

/// The name a container's dtype and scale layout describe, for a diagnostic that has to name
/// the format the caller actually supplied.
fn stored_format_name(dtype: StoredDType, layout: ScaleLayout) -> String {
    match (dtype, layout) {
        (StoredDType::U8, ScaleLayout::NvFp4 { .. }) => "nvfp4".to_owned(),
        (StoredDType::F8E4M3, ScaleLayout::MxFp8 { .. }) => "mx-fp8".to_owned(),
        (StoredDType::F8E4M3, ScaleLayout::Fp8Tile { .. }) => "fp8-e4m3".to_owned(),
        (StoredDType::BF16, ScaleLayout::PackedBf16 { .. }) => "packed-bf16".to_owned(),
        (StoredDType::BF16, ScaleLayout::None) => "bf16".to_owned(),
        (StoredDType::F32, ScaleLayout::None) => "f32".to_owned(),
        (dtype, layout) => format!("{} with {} scales", dtype.name(), layout.name()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str, dtype: StoredDType, shape: &[usize]) -> TensorSpec {
        TensorSpec {
            name: name.to_owned(),
            dtype,
            shape: shape.to_vec(),
        }
    }

    #[test]
    fn every_marker_names_its_stored_dtype_layout_and_arithmetic() {
        assert_eq!(
            (Bf16::STORED, Bf16::LOGICAL, Bf16::SCALES, Bf16::PRECISION),
            (
                StoredDType::BF16,
                StoredDType::BF16,
                ScaleLayout::PackedBf16 {
                    columns_per_word: 2
                },
                Precision::Bf16
            )
        );
        assert_eq!(
            (Fp8E4m3::STORED, Fp8E4m3::SCALES, Fp8E4m3::PRECISION),
            (
                StoredDType::F8E4M3,
                ScaleLayout::Fp8Tile {
                    rows: 128,
                    columns: 128
                },
                Precision::Fp8E4M3
            )
        );
        assert_eq!(
            (MxFp8::STORED, MxFp8::SCALES, MxFp8::PRECISION),
            (
                StoredDType::F8E4M3,
                ScaleLayout::MxFp8 {
                    rows: 32,
                    columns: 32
                },
                Precision::MxFp8
            )
        );
        assert_eq!(
            (
                NvFp4::STORED,
                NvFp4::LOGICAL,
                NvFp4::SCALES,
                NvFp4::PRECISION
            ),
            (
                StoredDType::U8,
                StoredDType::F4E2M1,
                ScaleLayout::NvFp4 { group: 16 },
                Precision::NvFp4
            )
        );
        assert_eq!(
            (F32::STORED, F32::SCALES, F32::PRECISION),
            (StoredDType::F32, ScaleLayout::None, Precision::F32)
        );
    }

    #[test]
    fn the_ledger_identity_agrees_with_every_marker() {
        let cases = [
            (
                FormatKind::of::<Bf16>(),
                Bf16::STORED,
                Bf16::SCALES,
                Bf16::PRECISION,
            ),
            (
                FormatKind::of::<Fp8E4m3>(),
                Fp8E4m3::STORED,
                Fp8E4m3::SCALES,
                Fp8E4m3::PRECISION,
            ),
            (
                FormatKind::of::<MxFp8>(),
                MxFp8::STORED,
                MxFp8::SCALES,
                MxFp8::PRECISION,
            ),
            (
                FormatKind::of::<NvFp4>(),
                NvFp4::STORED,
                NvFp4::SCALES,
                NvFp4::PRECISION,
            ),
            (
                FormatKind::of::<F32>(),
                F32::STORED,
                F32::SCALES,
                F32::PRECISION,
            ),
        ];
        for (kind, stored, scales, precision) in cases {
            assert_eq!(kind.stored(), stored);
            assert_eq!(kind.scales(), scales);
            assert_eq!(kind.precision(), precision);
        }
    }

    #[test]
    fn the_scale_layouts_count_their_blocks() {
        assert_eq!(ScaleLayout::None.entries(7, 9), 0);
        assert_eq!(
            ScaleLayout::NvFp4 { group: 16 }.entries(4, 64),
            4 * 4,
            "one E4M3 scale a 16-value group"
        );
        assert_eq!(
            ScaleLayout::MxFp8 {
                rows: 32,
                columns: 32
            }
            .entries(64, 96),
            2 * 3,
            "one E8M0 code a 32 x 32 block"
        );
        assert_eq!(
            ScaleLayout::Fp8Tile {
                rows: 128,
                columns: 128
            }
            .entries(129, 256),
            2 * 2,
            "one inverse scale a 128 x 128 tile"
        );
        assert_eq!(
            ScaleLayout::PackedBf16 {
                columns_per_word: 2
            }
            .entries(4, 512),
            0,
            "the packed form has no numeric scale plane"
        );
    }

    #[test]
    fn one_small_upload_per_format() {
        // bf16: no scales.
        let stored = spec("norm.weight", StoredDType::BF16, &[3, 8]);
        let bf16: Mat<Bf16> = upload(Tensor {
            stored: &stored,
            shape: &[3, 8],
            data: Deviceptr(0x1000),
            stride: 16,
            scales: Scales::none(),
        })
        .expect("a bf16 upload");
        assert_eq!((bf16.rows(), bf16.columns(), bf16.stride()), (3, 8, 16));
        assert_eq!(bf16.bytes_per_token(), 48);
        assert_eq!(bf16.format(), FormatKind::Bf16);

        // f32: no scales.
        let stored = spec("norm.bias", StoredDType::F32, &[3, 8]);
        let f32: Mat<F32> = upload(Tensor {
            stored: &stored,
            shape: &[3, 8],
            data: Deviceptr(0x2000),
            stride: 32,
            scales: Scales::none(),
        })
        .expect("an f32 upload");
        assert_eq!(f32.bytes_per_token(), 96);

        // FP8 E4M3: one f32 inverse scale a 128 x 128 tile.
        let stored = spec("layer.mlp.up.weight", StoredDType::F8E4M3, &[3, 256]);
        let fp8: Mat<Fp8E4m3> = upload(Tensor {
            stored: &stored,
            shape: &[3, 256],
            data: Deviceptr(0x3000),
            stride: 256,
            scales: Scales::fp8_e4m3(Deviceptr(0x4000), 2, StoredDType::F32),
        })
        .expect("an FP8 upload");
        assert_eq!(fp8.scales().entries(), 2);
        assert_eq!(fp8.bytes_per_token(), 768);

        // MXFP8: one E8M0 code a 32 x 32 block.
        let stored = spec("layer.mlp.down.weight", StoredDType::F8E4M3, &[3, 128]);
        let mx: Mat<MxFp8> = upload(Tensor {
            stored: &stored,
            shape: &[3, 128],
            data: Deviceptr(0x5000),
            stride: 128,
            scales: Scales::mxfp8(Deviceptr(0x6000), 4),
        })
        .expect("an MXFP8 upload");
        assert_eq!(mx.scales().entries(), 4);
        assert_eq!(mx.bytes_per_token(), 384);

        // NVFP4: one E4M3 scale a 16-value group plus the global.
        let stored = spec("layer.mlp.up.weight", StoredDType::U8, &[3, 32]);
        let nvfp4: Mat<NvFp4> = upload(Tensor {
            stored: &stored,
            shape: &[3, 64],
            data: Deviceptr(0x7000),
            stride: 32,
            scales: Scales::nvfp4(Deviceptr(0x8000), 12, 0.25),
        })
        .expect("an NVFP4 upload");
        assert_eq!(nvfp4.spec().dtype, StoredDType::F4E2M1);
        assert_eq!(nvfp4.bytes_per_token(), 96);
        assert!(matches!(nvfp4.scales(), Scales::NvFp4 { global: 0.25, .. }));
    }

    #[test]
    fn the_packed_bf16_word_layout_is_one_small_upload() {
        let columns = 512;
        let row_bytes = crate::packed::row_bytes(columns);
        let stored = spec("dense.weight", StoredDType::BF16, &[4, columns]);
        let packed = PackedBf16 {
            escapes: Deviceptr(0x9000),
            offsets: Deviceptr(0xa000),
            row_bytes,
            order: PackedOrder::Walk,
        };
        let mat: Mat<Bf16> = upload(Tensor {
            stored: &stored,
            shape: &[4, columns],
            data: Deviceptr(0xb000),
            stride: row_bytes,
            scales: Scales::packed_bf16(packed),
        })
        .expect("a packed BF16 upload");
        assert_eq!(mat.stride(), row_bytes);
        assert_eq!(mat.bytes_per_token(), (row_bytes * 4) as u128);
        assert_eq!(
            mat.scales().layout(),
            ScaleLayout::PackedBf16 {
                columns_per_word: 2
            }
        );
    }

    #[test]
    fn a_stored_format_mismatch_names_both_formats() {
        let stored = spec("layer.mlp.up.weight", StoredDType::U8, &[3, 32]);
        let error = upload::<Bf16>(Tensor {
            stored: &stored,
            shape: &[3, 64],
            data: Deviceptr(0x1000),
            stride: 32,
            scales: Scales::nvfp4(Deviceptr(0x2000), 12, 1.0),
        })
        .expect_err("an NVFP4 tensor is not a bf16 Mat");
        let message = error.to_string();
        assert!(message.contains("bf16"), "{message}");
        assert!(message.contains("nvfp4"), "{message}");
        assert!(message.contains("U8"), "{message}");
    }

    #[test]
    fn a_scale_count_mismatch_names_both_formats() {
        let stored = spec("layer.mlp.down.weight", StoredDType::F8E4M3, &[4, 64]);
        let error = upload::<MxFp8>(Tensor {
            stored: &stored,
            shape: &[4, 64],
            data: Deviceptr(0x1000),
            stride: 64,
            // A 4 x 64 matrix is 1 x 2 = 2 blocks of 32 x 32, not 1.
            scales: Scales::mxfp8(Deviceptr(0x2000), 1),
        })
        .expect_err("the scale plane is short");
        let message = error.to_string();
        assert!(message.contains("mx-fp8"), "{message}");
        assert!(message.contains("needs 2"), "{message}");
        assert!(message.contains("has 1"), "{message}");
    }

    #[test]
    fn an_fp8_tile_scale_has_a_scalar_dtype_the_tile_takes() {
        let stored = spec("layer.mlp.up.weight", StoredDType::F8E4M3, &[1, 128]);
        let error = upload::<Fp8E4m3>(Tensor {
            stored: &stored,
            shape: &[1, 128],
            data: Deviceptr(0x1000),
            stride: 128,
            scales: Scales::fp8_e4m3(Deviceptr(0x2000), 1, StoredDType::U8),
        })
        .expect_err("an inverse scale is f32 or bf16");
        assert!(error.to_string().contains("f32 or bf16"), "{error}");
    }

    #[test]
    fn the_inkling_fixture_short_convolution_uploads_as_bf16() {
        // `model.llm.layers.0.attn_sconv.weight` is the fixture's 3-D short-convolution weight,
        // `[hidden, 1, kernel]`; the boot flattens it to the `[hidden, kernel]` matrix the
        // arithmetic sees and still passes the container's 3-D spec (`device::type_slots`).
        // Upload dereferences nothing, so the address need not be a live device pointer.
        let stored = spec(
            "model.llm.layers.0.attn_sconv.weight",
            StoredDType::BF16,
            &[32, 1, 4],
        );
        let mat: Mat<Bf16> = upload(Tensor {
            stored: &stored,
            shape: &[32, 4],
            data: Deviceptr(0x1000),
            stride: 8,
            scales: Scales::none(),
        })
        .expect("the fixture's short-convolution weight uploads as bf16");
        assert_eq!((mat.rows(), mat.columns(), mat.stride()), (32, 4, 8));
        assert_eq!(mat.spec().dtype, StoredDType::BF16);
        assert_eq!(mat.spec().shape, vec![32, 4]);
        assert_eq!(mat.bytes_per_token(), 256);
    }
}

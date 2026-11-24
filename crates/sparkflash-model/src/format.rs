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

//! One stored weight format, carried from the layout to the body that reads it.
//!
//! A format is three things, and losing any of them makes a body run another arithmetic:
//!
//! * the **values** (bf16, f32, E4M3 codes, E2M1 codes);
//! * the **scale kind** and the **block** one scale covers. A power-of-two grid is an MX block
//!   (one `ue8m0` code a 32); a tile scale is an FP8 inverse scale a 128 x 128 tile; a row scale is
//!   one power-of-two code a row; and a tensor-wide scale is one factor for the whole matrix, on
//!   top of a block scale (NVFP4);
//! * the **layout** the body reads: row-major, the 32 x 32 tiling DeepSeek's dense bodies read, or
//!   the packed bf16 word form.
//!
//! The checkpoint's `quantization_config` names a format in [WeightFormat]; the layout reader
//! turns every tensor's declaration into one [StoredFormat], and that value is what the plan, the
//! catalogue key and the bind carry. Nothing downstream reads a producer's dtype name again.
//!
//! The scale plane's own values are part of the format ([ScaleValues]): an FP8 128-tile release
//! stores its inverse scales as f32 or as bf16, and the two are different bytes the body reads
//! with different instructions. A format that named only "a tile scale" would let a body be handed
//! the other one.

use crate::architecture::{ScaleDType, WeightFormat};

/// The values a stored format keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatValues {
    /// Unquantized bfloat16.
    Bf16,
    /// IEEE binary32.
    F32,
    /// E4M3 codes, one byte a value.
    Fp8E4M3,
    /// E2M1 codes, two values a byte.
    Fp4E2M1,
}

impl FormatValues {
    /// The name a ledger or a scan row prints.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::Fp8E4M3 => "e4m3",
            Self::Fp4E2M1 => "e2m1",
        }
    }

    /// The values one byte holds: two for the 4-bit codes, one otherwise.
    #[must_use]
    pub const fn values_per_byte(self) -> usize {
        match self {
            Self::Fp4E2M1 => 2,
            Self::Bf16 | Self::F32 | Self::Fp8E4M3 => 1,
        }
    }
}

/// The values one scale plane holds.
///
/// A block scale is one plane with one set of values: DeepSeek's power-of-two grid is a `ue8m0`
/// byte a block, NVFP4's group scale is an E4M3 byte a group, and an FP8 128-tile release stores
/// its inverse scale as f32 or as bf16. The tensor-wide factor is always f32
/// ([ScaleKind::TensorWide]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleValues {
    /// One unsigned power-of-two exponent byte a block (`ue8m0`, E8M0).
    Ue8m0,
    /// One E4M3 code a block (NVFP4's group scale).
    E4M3,
    /// One IEEE bfloat16 a block.
    Bf16,
    /// One IEEE binary32 a block.
    F32,
}

impl ScaleValues {
    /// The name a ledger or an evidence row prints.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Ue8m0 => "ue8m0",
            Self::E4M3 => "e4m3",
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
        }
    }

    /// The bytes one scale occupies.
    #[must_use]
    pub const fn bytes(self) -> usize {
        match self {
            Self::Ue8m0 | Self::E4M3 => 1,
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }

    /// The scale values a checkpoint declares, when it declares them.
    #[must_use]
    pub const fn of(scale: ScaleDType) -> Self {
        match scale {
            ScaleDType::F32 => Self::F32,
            ScaleDType::Bf16 => Self::Bf16,
            ScaleDType::E4M3 => Self::E4M3,
        }
    }
}

/// What a format's scales are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleKind {
    /// One power-of-two (`ue8m0`) code every [StoredFormat::block] values along the contraction.
    PowerOfTwoGrid,
    /// One inverse scale a tile of the matrix.
    Tile,
    /// One E4M3 group scale a row, every [StoredFormat::block] columns (NVFP4's per-16 plane).
    Group,
    /// One power-of-two scale a row, every [StoredFormat::block] columns.
    Row,
    /// One factor for the whole tensor, beside any block scale.
    TensorWide,
}

impl ScaleKind {
    /// The name a description uses.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::PowerOfTwoGrid => "power-of-two grid",
            Self::Tile => "tile scale",
            Self::Group => "per-group scale",
            Self::Row => "row scale",
            Self::TensorWide => "tensor-wide scale",
        }
    }
}

/// How a 4-bit format carries its tensor-wide factors.
///
/// The two releases that store NVFP4 multiply the same group dequantisation by a different
/// factor in a different place. Qwen stores a weight global (`weight_scale_2`) beside a static
/// activation global (`input_scale`); its body folds the weight global into the group scale
/// before accumulation. Inkling stores a per-expert `amax` and one outer denominator per expert
/// tensor; its body applies `amax / denominator` to the expert's output after accumulation.
/// The products are mathematically equal but round at different points, so the two are formats a
/// body cannot be handed for one another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NvFp4Wide {
    /// No tensor-wide factor.
    None,
    /// Qwen's weight global and static activation global: two f32 factors.
    Qwen,
    /// Inkling's per-expert amax and outer denominator: two f32 factors.
    Inkling,
}

impl NvFp4Wide {
    /// The number of f32 factors the convention stores.
    #[must_use]
    pub const fn factors(self) -> usize {
        match self {
            Self::None => 0,
            Self::Qwen | Self::Inkling => 2,
        }
    }

    /// The name a scan row prints.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Qwen => "qwen",
            Self::Inkling => "inkling",
        }
    }
}

/// How a body reads a tensor's values in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatLayout {
    /// Consecutive columns, a row at a time.
    RowMajor,
    /// Re-tiled into 32 x 32 tiles at load, the B12x layout DeepSeek's dense bodies read.
    Tile32,
    /// The packed bf16 word form (two columns a 32-bit word, with its escape list).
    Packed,
}

impl FormatLayout {
    /// The name a description uses.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::RowMajor => "row-major",
            Self::Tile32 => "32-tile",
            Self::Packed => "packed",
        }
    }
}

/// The scale kinds one format applies, in apply order.
const NONE: &[ScaleKind] = &[];
const GRID: &[ScaleKind] = &[ScaleKind::PowerOfTwoGrid];
const TILE: &[ScaleKind] = &[ScaleKind::Tile];
const ROW_AND_GRID: &[ScaleKind] = &[ScaleKind::Row, ScaleKind::PowerOfTwoGrid];
const GROUP_AND_TENSOR: &[ScaleKind] = &[ScaleKind::Group, ScaleKind::TensorWide];

/// One stored tensor format: the values, the block one scale covers, the scales the body applies
/// and the layout it reads.
///
/// A format is a value, not a model fact: two checkpoints that store the same arithmetic in the
/// same arrangement compare equal, and a body that takes one cannot be handed another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StoredFormat {
    values: FormatValues,
    block: usize,
    scales: &'static [ScaleKind],
    scale_values: ScaleValues,
    layout: FormatLayout,
    wide: NvFp4Wide,
}

/// Whether a scale list carries a tile scale. A const predicate, because the format's implied
/// scale values are computed in a const constructor.
const fn has_tile(scales: &[ScaleKind]) -> bool {
    has_kind(scales, ScaleKind::Tile)
}

/// Whether a scale list carries a per-group (NVFP4 per-16) scale.
const fn has_group(scales: &[ScaleKind]) -> bool {
    has_kind(scales, ScaleKind::Group)
}

/// Whether a scale list carries a tensor-wide factor.
const fn has_tensor(scales: &[ScaleKind]) -> bool {
    has_kind(scales, ScaleKind::TensorWide)
}

/// Whether a scale list carries one kind.
const fn has_kind(scales: &[ScaleKind], kind: ScaleKind) -> bool {
    let mut index = 0;
    while index < scales.len() {
        if scales[index] as u8 == kind as u8 {
            return true;
        }
        index += 1;
    }
    false
}

impl StoredFormat {
    /// A format from its parts, taking the scale values its class implies.
    ///
    /// The implication is the release each scale kind came from: a power-of-two grid is a
    /// `ue8m0` byte, NVFP4's group scale is an E4M3 byte, an unscaled format has no plane,
    /// and an FP8 tile scale is f32 unless the declaration says otherwise
    /// ([Self::with_scale_values]).
    #[must_use]
    pub const fn new(
        values: FormatValues,
        block: usize,
        scales: &'static [ScaleKind],
        layout: FormatLayout,
    ) -> Self {
        let scale_values = match values {
            // A 4-bit group scale is E4M3 (NVFP4); a 4-bit block or row scale is a power of two.
            FormatValues::Fp4E2M1 if has_group(scales) || has_tile(scales) => ScaleValues::E4M3,
            FormatValues::Fp8E4M3 if has_tile(scales) => ScaleValues::F32,
            _ => ScaleValues::Ue8m0,
        };
        let wide = match values {
            FormatValues::Fp4E2M1 if has_tensor(scales) => NvFp4Wide::Qwen,
            _ => NvFp4Wide::None,
        };
        Self {
            values,
            block,
            scales,
            scale_values,
            layout,
            wide,
        }
    }

    /// Unquantized bf16, row-major.
    pub const BF16: Self = Self::new(FormatValues::Bf16, 1, NONE, FormatLayout::RowMajor);
    /// IEEE binary32, row-major.
    pub const F32: Self = Self::new(FormatValues::F32, 1, NONE, FormatLayout::RowMajor);
    /// FP8 E4M3 with one inverse scale a 128 x 128 tile, row-major, stored f32.
    pub const FP8_E4M3_TILE_128: Self =
        Self::new(FormatValues::Fp8E4M3, 128, TILE, FormatLayout::RowMajor);
    /// The same arithmetic with the tile scale stored bf16 (GLM's and Qwen's releases).
    pub const FP8_E4M3_TILE_128_BF16: Self =
        Self::new(FormatValues::Fp8E4M3, 128, TILE, FormatLayout::RowMajor)
            .with_scale_values(ScaleValues::Bf16);
    /// MXFP8: E4M3 codes with one power-of-two scale a 32-value block, row-major.
    pub const MXFP8_32: Self = Self::new(FormatValues::Fp8E4M3, 32, GRID, FormatLayout::RowMajor);
    /// MXFP8 in the 32 x 32 tiling DeepSeek's dense and attention bodies read.
    pub const MXFP8_TILED_32: Self =
        Self::new(FormatValues::Fp8E4M3, 32, GRID, FormatLayout::Tile32);
    /// Row-scaled MXFP8: one power-of-two scale a row, every 32 columns, row-major.
    pub const MXFP8_ROW_32: Self = Self::new(
        FormatValues::Fp8E4M3,
        32,
        ROW_AND_GRID,
        FormatLayout::RowMajor,
    );
    /// NVFP4 with Qwen's two tensor-wide factors: E2M1 codes, one E4M3 scale a 16-value group,
    /// a weight global folded into the group scale and a static activation global.
    pub const NVFP4_QWEN_16: Self = Self::new(
        FormatValues::Fp4E2M1,
        16,
        GROUP_AND_TENSOR,
        FormatLayout::RowMajor,
    );
    /// NVFP4 with Inkling's two tensor-wide factors: E2M1 codes, one E4M3 scale a 16-value
    /// group, a per-expert amax and an outer denominator applied to the expert's output.
    pub const NVFP4_INKLING_16: Self = Self::NVFP4_QWEN_16.with_tensor_wide(NvFp4Wide::Inkling);
    /// The name the older code uses for Qwen's NVFP4 convention.
    pub const NVFP4_16: Self = Self::NVFP4_QWEN_16;
    /// MXFP4 in the expert arrangement: E2M1 codes with one power-of-two scale a row, every 32
    /// values along the contraction. This is the form every fp4 expert body in the tree reads, so
    /// it is the one a declaration's `mx-fp4` means.
    pub const MXFP4_ROW_32: Self = Self::new(
        FormatValues::Fp4E2M1,
        32,
        ROW_AND_GRID,
        FormatLayout::RowMajor,
    );
    /// MXFP4 on a two-dimensional 32-value block grid. No body in the tree reads this form; a
    /// release that stores one is refused by name rather than decoded as the expert arrangement.
    pub const MXFP4_32: Self = Self::new(FormatValues::Fp4E2M1, 32, GRID, FormatLayout::RowMajor);
    /// The packed bf16 form: the same arithmetic as [Self::BF16], in the packed word layout.
    pub const PACKED_BF16: Self = Self::new(FormatValues::Bf16, 1, NONE, FormatLayout::Packed);

    /// The values the format keeps.
    #[must_use]
    pub const fn values(self) -> FormatValues {
        self.values
    }

    /// The values one scale covers, or one for an unscaled format.
    #[must_use]
    pub const fn block(self) -> usize {
        self.block
    }

    /// The scale kinds the body applies, in order.
    #[must_use]
    pub const fn scales(self) -> &'static [ScaleKind] {
        self.scales
    }

    /// The values the block scale plane holds.
    #[must_use]
    pub const fn scale_values(self) -> ScaleValues {
        self.scale_values
    }

    /// The tensor-wide convention this format applies.
    #[must_use]
    pub const fn tensor_wide(self) -> NvFp4Wide {
        self.wide
    }

    /// This format with a tensor-wide convention.
    ///
    /// The factors a format stores are part of its arithmetic: Qwen's weight global is folded
    /// into the group scale, Inkling's per-expert factor is applied to the output.
    #[must_use]
    pub const fn with_tensor_wide(self, wide: NvFp4Wide) -> Self {
        Self { wide, ..self }
    }

    /// This format's scale plane in other values: the same arithmetic, stored another way.
    ///
    /// It is not the same arithmetic: a body that decodes f32 scales cannot read bf16 ones, so
    /// [Self::same_arithmetic] tells them apart.
    #[must_use]
    pub const fn with_scale_values(self, scale_values: ScaleValues) -> Self {
        Self {
            scale_values,
            ..self
        }
    }

    /// The bytes one scale of this format occupies.
    #[must_use]
    pub const fn scale_bytes(self) -> usize {
        self.scale_values.bytes()
    }

    /// The number of block scales a `rows` x `cols` tensor of this format stores.
    ///
    /// The three arrangements a stored tensor can have: a 2-D block grid, one scale a row (one
    /// plane the values and one the scale kind), and one factor for the whole matrix.
    #[must_use]
    pub const fn scale_count(self, rows: usize, cols: usize) -> usize {
        if self.is_unscaled() {
            return 0;
        }
        let block = if self.block == 0 { 1 } else { self.block };
        if self.has_row() || self.has_group() {
            return rows * cols.div_ceil(block);
        }
        rows.div_ceil(block) * cols.div_ceil(block)
    }

    /// Whether the format carries no scale plane at all (bf16 or f32 values).
    ///
    /// An unscaled body widens what it reads, so bf16 and f32 are one operand class for
    /// selection; the arithmetic a body accepts still comes from its precision mask.
    #[must_use]
    pub const fn is_unscaled(self) -> bool {
        self.block == 1 && self.scales.is_empty()
    }

    /// Whether the format applies this scale kind.
    #[must_use]
    pub const fn has_scale(self, kind: ScaleKind) -> bool {
        let mut index = 0;
        while index < self.scales.len() {
            if self.scales[index] as u8 == kind as u8 {
                return true;
            }
            index += 1;
        }
        false
    }

    /// The layout the body reads.
    #[must_use]
    pub const fn layout(self) -> FormatLayout {
        self.layout
    }

    /// This format read in another layout: the same arithmetic, arranged for one body.
    #[must_use]
    pub const fn with_layout(self, layout: FormatLayout) -> Self {
        Self { layout, ..self }
    }

    /// Whether two formats are the same arithmetic: the values, the block and the scales agree.
    ///
    /// The layout is the body's arrangement of the same stored tensor, so it is not part of the
    /// arithmetic. A body may therefore read a row-major checkpoint through its 32-tile layout
    /// without the format changing meaning.
    #[must_use]
    pub const fn same_arithmetic(self, other: Self) -> bool {
        self.values as u8 == other.values as u8
            && self.block == other.block
            && self.scale_values as u8 == other.scale_values as u8
            && self.wide as u8 == other.wide as u8
            && self.scales.len() == other.scales.len()
            && {
                let mut index = 0;
                let mut same = true;
                while index < self.scales.len() {
                    same = same && self.scales[index] as u8 == other.scales[index] as u8;
                    index += 1;
                }
                same
            }
    }

    /// The format's name, as a scan row or a ledger prints it.
    ///
    /// The names match [WeightFormat::name] for the formats a checkpoint declares, so the scan's
    /// format column does not move when the reader starts carrying the whole format.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match (self.values, self.layout) {
            (FormatValues::Bf16, FormatLayout::Packed) => "PACKED-BF16",
            (FormatValues::Bf16, _) => "BF16",
            (FormatValues::F32, _) => "F32",
            (FormatValues::Fp8E4M3, _) if self.has_row() => "MXFP8-ROW",
            (FormatValues::Fp8E4M3, _)
                if self.has_scale(ScaleKind::Tile)
                    && self.scale_values as u8 == ScaleValues::Bf16 as u8 =>
            {
                "FP8-BF16"
            }
            (FormatValues::Fp8E4M3, _) if self.has_scale(ScaleKind::Tile) => "FP8",
            (FormatValues::Fp8E4M3, _) => "MXFP8",
            (FormatValues::Fp4E2M1, _)
                if self.has_scale(ScaleKind::TensorWide)
                    && self.wide as u8 == NvFp4Wide::Inkling as u8 =>
            {
                "NVFP4-INKLING"
            }
            (FormatValues::Fp4E2M1, _) if self.has_scale(ScaleKind::TensorWide) => "NVFP4",
            (FormatValues::Fp4E2M1, _) if self.has_row() => "MXFP4",
            (FormatValues::Fp4E2M1, _) => "MXFP4-GRID",
        }
    }

    /// Whether the format carries one E4M3 scale a row, every 16 values.
    #[must_use]
    pub const fn has_group(self) -> bool {
        has_group(self.scales)
    }

    /// Whether the format carries one scale a row.
    const fn has_row(self) -> bool {
        let mut index = 0;
        while index < self.scales.len() {
            if matches!(self.scales[index], ScaleKind::Row) {
                return true;
            }
            index += 1;
        }
        false
    }

    /// A one-line description: the name, the block and the scales, and the layout.
    #[must_use]
    pub fn describe(self) -> String {
        let scales = if self.scales.is_empty() {
            "unscaled".to_owned()
        } else {
            self.scales
                .iter()
                .map(|scale| scale.name())
                .collect::<Vec<_>>()
                .join(" + ")
        };
        let block = if self.block > 1 {
            format!(", one scale per {} values", self.block)
        } else {
            String::new()
        };
        format!(
            "{} ({}{}, {})",
            self.name(),
            scales,
            block,
            self.layout.name()
        )
    }

    /// The format a checkpoint's declaration describes.
    ///
    /// Every block width the declaration carries travels into the format, so a release that
    /// declares an unusual group or tile keeps it rather than being read at a default.
    #[must_use]
    pub const fn of(format: WeightFormat) -> Self {
        match format {
            WeightFormat::Bf16 => Self::BF16,
            WeightFormat::F32 => Self::F32,
            WeightFormat::Fp8Tile {
                block_n,
                block_m,
                scale_dtype,
                ..
            } => {
                let block = if block_n >= block_m { block_n } else { block_m };
                Self::new(FormatValues::Fp8E4M3, block, TILE, FormatLayout::RowMajor)
                    .with_scale_values(ScaleValues::of(scale_dtype))
            }
            WeightFormat::NvFp4 { group_size, wide } => Self::new(
                FormatValues::Fp4E2M1,
                group_size,
                GROUP_AND_TENSOR,
                FormatLayout::RowMajor,
            )
            .with_tensor_wide(wide),
            WeightFormat::MxFp8 { block } => {
                Self::new(FormatValues::Fp8E4M3, block, GRID, FormatLayout::RowMajor)
            }
            WeightFormat::MxFp4 { block } => Self::new(
                FormatValues::Fp4E2M1,
                block,
                ROW_AND_GRID,
                FormatLayout::RowMajor,
            ),
        }
    }

    /// The format a name states, for a build list or a layout file.
    ///
    /// The names are the ones [Self::name] prints plus the explicit spellings a build file uses:
    /// `mx-fp8-tiled` is the 32-tile layout, `mx-fp8-row` the row-scaled form, and `packed-bf16`
    /// the packed word form. An unknown name is refused rather than guessed.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "bf16" | "bfloat16" => Some(Self::BF16),
            "f32" | "float32" => Some(Self::F32),
            "fp8" | "fp8-e4m3" | "fp8-tile" | "e4m3-tile" | "fp8-f32" => {
                Some(Self::FP8_E4M3_TILE_128)
            }
            "fp8-bf16" | "fp8-tile-bf16" => Some(Self::FP8_E4M3_TILE_128_BF16),
            "mxfp8" | "mx-fp8" => Some(Self::MXFP8_32),
            "mxfp8-tiled" | "mx-fp8-tiled" => Some(Self::MXFP8_TILED_32),
            "mxfp8-row" | "mx-fp8-row" | "mxfp8-rowscale" | "mx-fp8-rowscale" => {
                Some(Self::MXFP8_ROW_32)
            }
            "nvfp4" | "nvfp4-qwen" | "nvfp4-qwen-16" => Some(Self::NVFP4_QWEN_16),
            "nvfp4-inkling" | "nvfp4-inkling-16" => Some(Self::NVFP4_INKLING_16),
            "mxfp4" | "mx-fp4" | "mxfp4-row" | "mx-fp4-row" => Some(Self::MXFP4_ROW_32),
            "mx-fp4-grid" | "mxfp4-grid" => Some(Self::MXFP4_32),
            "packed" | "packed-bf16" => Some(Self::PACKED_BF16),
            _ => None,
        }
    }
}

/// A stored format stated for a range of blocks.
///
/// A checkpoint whose expert format changes part-way up the stack (Inkling stores NVFP4 in
/// blocks 0-39 and row-scaled MXFP8 above) states one entry a range. The name is the format's
/// own spelling, `@`, and the inclusive block range: `nvfp4-qwen@0-39`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockPartFormat {
    /// The format the range stores.
    pub format: StoredFormat,
    /// The first block the range covers.
    pub first: usize,
    /// The last block the range covers, inclusive.
    pub last: usize,
}

impl BlockPartFormat {
    /// A format and the inclusive block range it covers.
    #[must_use]
    pub const fn new(format: StoredFormat, first: usize, last: usize) -> Self {
        Self {
            format,
            first,
            last,
        }
    }

    /// Whether a block belongs to the range.
    #[must_use]
    pub const fn covers(&self, block: usize) -> bool {
        block >= self.first && block <= self.last
    }

    /// The name a build list or a ledger prints: `format@first-last`.
    #[must_use]
    pub fn name(&self) -> String {
        format!("{}@{}..={}", self.format.name(), self.first, self.last)
    }

    /// The format and range a name describes, or `None` for an unknown spelling.
    ///
    /// The accepted names are [StoredFormat::from_name]'s, followed by `@first-last` or
    /// `@first..=last`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        let (format, blocks) = name.rsplit_once('@')?;
        let blocks = blocks.trim();
        let (first, last) = blocks
            .split_once("..=")
            .or_else(|| blocks.split_once('-'))?;
        Some(Self::new(
            StoredFormat::from_name(format)?,
            first.parse().ok()?,
            last.parse().ok()?,
        ))
    }
}

impl From<WeightFormat> for StoredFormat {
    fn from(format: WeightFormat) -> Self {
        Self::of(format)
    }
}

impl WeightFormat {
    /// The format this declaration describes, with its block, scale kinds and layout.
    #[must_use]
    pub const fn stored_format(self) -> StoredFormat {
        StoredFormat::of(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_checkpoint_declaration_carries_its_block_scales_and_layout() {
        assert_eq!(StoredFormat::of(WeightFormat::Bf16), StoredFormat::BF16);
        assert_eq!(StoredFormat::of(WeightFormat::F32), StoredFormat::F32);
        assert_eq!(
            StoredFormat::of(WeightFormat::Fp8Tile {
                block_m: 128,
                block_n: 128,
                scale_dtype: crate::architecture::ScaleDType::F32,
                dynamic_activations: true,
            }),
            StoredFormat::FP8_E4M3_TILE_128
        );
        assert_eq!(
            StoredFormat::of(WeightFormat::MxFp8 { block: 32 }),
            StoredFormat::MXFP8_32
        );
        assert_eq!(
            StoredFormat::of(WeightFormat::MxFp4 { block: 32 }),
            StoredFormat::MXFP4_ROW_32
        );
        assert_eq!(
            StoredFormat::of(WeightFormat::Fp8Tile {
                block_m: 128,
                block_n: 128,
                scale_dtype: crate::architecture::ScaleDType::Bf16,
                dynamic_activations: true,
            }),
            StoredFormat::FP8_E4M3_TILE_128_BF16
        );
        assert!(
            !StoredFormat::FP8_E4M3_TILE_128.same_arithmetic(StoredFormat::FP8_E4M3_TILE_128_BF16),
            "an f32 tile scale and a bf16 one are different bytes a body reads"
        );
        assert_eq!(
            StoredFormat::of(WeightFormat::NvFp4 {
                group_size: 16,
                wide: NvFp4Wide::Qwen
            }),
            StoredFormat::NVFP4_16
        );
    }

    #[test]
    fn an_unusual_block_or_group_travels_rather_than_being_read_at_a_default() {
        let wide = StoredFormat::of(WeightFormat::MxFp4 { block: 64 });
        assert_eq!(wide.block(), 64);
        assert!(!wide.same_arithmetic(StoredFormat::MXFP4_ROW_32));
        let group = StoredFormat::of(WeightFormat::NvFp4 {
            group_size: 32,
            wide: NvFp4Wide::Qwen,
        });
        assert_eq!(group.block(), 32);
        assert!(group.has_scale(ScaleKind::TensorWide));
    }

    #[test]
    fn the_two_4_bit_formats_are_not_one_format() {
        assert!(!StoredFormat::MXFP4_32.same_arithmetic(StoredFormat::NVFP4_16));
        assert!(StoredFormat::MXFP4_32.has_scale(ScaleKind::PowerOfTwoGrid));
        assert!(!StoredFormat::MXFP4_32.has_scale(ScaleKind::TensorWide));
        assert!(StoredFormat::NVFP4_16.has_scale(ScaleKind::Group));
        assert!(StoredFormat::NVFP4_16.has_scale(ScaleKind::TensorWide));
        assert!(!StoredFormat::NVFP4_16.has_scale(ScaleKind::Tile));
    }

    #[test]
    fn a_layout_is_the_body_arrangement_of_one_arithmetic() {
        let tiled = StoredFormat::MXFP8_32.with_layout(FormatLayout::Tile32);
        assert!(tiled.same_arithmetic(StoredFormat::MXFP8_32));
        assert_ne!(tiled.layout(), StoredFormat::MXFP8_32.layout());
        assert_eq!(tiled.name(), "MXFP8");
        assert!(tiled.describe().contains("32-tile"));
        assert!(StoredFormat::PACKED_BF16.same_arithmetic(StoredFormat::BF16));
        assert_eq!(StoredFormat::PACKED_BF16.layout(), FormatLayout::Packed);
    }

    #[test]
    fn a_build_name_round_trips_or_is_refused() {
        for format in [
            StoredFormat::BF16,
            StoredFormat::F32,
            StoredFormat::FP8_E4M3_TILE_128,
            StoredFormat::MXFP8_32,
            StoredFormat::MXFP8_TILED_32,
            StoredFormat::MXFP8_ROW_32,
            StoredFormat::NVFP4_16,
            StoredFormat::MXFP4_32,
            StoredFormat::MXFP4_ROW_32,
            StoredFormat::FP8_E4M3_TILE_128_BF16,
            StoredFormat::PACKED_BF16,
        ] {
            // A name states the arithmetic, not the layout: the 32-tile and row-scaled forms
            // print the same family name and round-trip to the same arithmetic.
            let named = StoredFormat::from_name(&format.name().to_ascii_lowercase())
                .unwrap_or_else(|| panic!("{}", format.name()));
            assert!(named.same_arithmetic(format), "{}", format.name());
        }
        assert_eq!(StoredFormat::from_name("bf16"), Some(StoredFormat::BF16));
        assert_eq!(
            StoredFormat::from_name("mx-fp8-tiled"),
            Some(StoredFormat::MXFP8_TILED_32)
        );
        assert_eq!(
            StoredFormat::from_name("mx-fp8-row"),
            Some(StoredFormat::MXFP8_ROW_32)
        );
        assert_eq!(
            StoredFormat::from_name("nvfp4"),
            Some(StoredFormat::NVFP4_16)
        );
        assert_eq!(
            StoredFormat::from_name("mx-fp4"),
            Some(StoredFormat::MXFP4_ROW_32)
        );
        assert_eq!(
            StoredFormat::from_name("mx-fp4-grid"),
            Some(StoredFormat::MXFP4_32)
        );
        assert_eq!(
            StoredFormat::from_name("fp8-bf16"),
            Some(StoredFormat::FP8_E4M3_TILE_128_BF16)
        );
        assert_eq!(StoredFormat::from_name("nope"), None);
    }

    #[test]
    fn a_format_name_can_state_a_block_part_range() {
        let range = BlockPartFormat::from_name("nvfp4-qwen@0-39").expect("the range spelling");
        assert!(range.format.same_arithmetic(StoredFormat::NVFP4_QWEN_16));
        assert_eq!((range.first, range.last), (0, 39));
        assert!(range.covers(0) && range.covers(39) && !range.covers(40));
        assert_eq!(
            BlockPartFormat::from_name("nvfp4-inkling@0..=39")
                .expect("the long range spelling")
                .format,
            StoredFormat::NVFP4_INKLING_16
        );
        assert!(BlockPartFormat::from_name("nvfp4-qwen").is_none());
        assert!(BlockPartFormat::from_name("nope@0-1").is_none());
    }

    #[test]
    fn the_two_nvfp4_conventions_round_trip_their_names() {
        for format in [StoredFormat::NVFP4_QWEN_16, StoredFormat::NVFP4_INKLING_16] {
            let named = StoredFormat::from_name(&format.name().to_ascii_lowercase())
                .unwrap_or_else(|| panic!("{}", format.name()));
            assert!(named.same_arithmetic(format), "{}", format.name());
        }
        assert_eq!(
            StoredFormat::from_name("nvfp4-qwen"),
            Some(StoredFormat::NVFP4_QWEN_16)
        );
        assert_eq!(
            StoredFormat::from_name("nvfp4-inkling"),
            Some(StoredFormat::NVFP4_INKLING_16)
        );
    }

    #[test]
    fn an_unscaled_format_says_so() {
        assert!(StoredFormat::BF16.is_unscaled());
        assert!(StoredFormat::F32.is_unscaled());
        assert!(StoredFormat::PACKED_BF16.is_unscaled());
        assert!(!StoredFormat::MXFP8_32.is_unscaled());
        assert!(!StoredFormat::NVFP4_16.is_unscaled());
    }
}

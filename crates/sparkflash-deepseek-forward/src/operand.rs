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

//! The stored format's three consequences, decided once at open.
//!
//! A part's stored format ([sparkflash_model::format::StoredFormat]) says what the checkpoint
//! holds. It does not say what the engine does with it. Three things follow from it, and this
//! module is the one place they are derived:
//!
//! * [ScalePlanes] - what the bind must carry beside the values: a block (grid) scale plane, a
//!   row scale plane, a per-group plane, a tensor-wide convention, and a packed tensor's escape
//!   list. The sizes come
//!   from the format's block and kind, never from a dtype string or a byte count of a tensor the
//!   caller happened to read.
//! * [arrange] - how the loader lays a tensor out for the body that reads it: as stored, in the
//!   32 x 32 B12x tiling, or in the packed word form. One function, so a second tiling site
//!   cannot drift from the first.
//! * [ActivationForm] and [Operand::body] - the form the seam must leave its output in for the
//!   next body to read, and the body itself, chosen per part instance and per block.
//!
//! The pointer to the body is the catalogue's: [Description](crate::plan::Description) indexes the
//! plan's symbols per block and part, and [choose] reads that index with the part's format. A
//! format no body reads is `None`, which the scan shows as "not built" rather than running the
//! wrong decode.

use sparkflash_model::Part;
use sparkflash_model::format::{
    FormatLayout, FormatValues, NvFp4Wide, ScaleKind, ScaleValues, StoredFormat,
};

/// The activation form a seam ends in, so the next body reads what the last one wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationForm {
    /// BF16 (or f32) rows, widened by the body: what a BF16-activation tile body reads.
    Bf16Rows,
    /// E4M3 codes with one power-of-two scale a 32-value block: DeepSeek's own seam.
    Fp8Blocks,
    /// E4M3 codes with one inverse scale per 128 columns: the q8 tile body's input.
    Fp8Per128,
}

impl ActivationForm {
    /// The name a ledger or a trace prints.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bf16Rows => "bf16-rows",
            Self::Fp8Blocks => "fp8-blocks",
            Self::Fp8Per128 => "fp8-per-128",
        }
    }

    /// The activation form a stored format's seam ends in.
    ///
    /// The weight decode decides it: a power-of-two block grid is DeepSeek's coded block input; a
    /// 128-tile weight is read by the bodies whose activation rows are BF16 (the decode) or
    /// quantised per 128 columns (the prompt tile). The choice between those two is the caller's
    /// ([ActivationForm::for_pass]); the format fixes what is possible.
    #[must_use]
    pub const fn of(format: StoredFormat) -> Option<Self> {
        if format.has_scale(ScaleKind::PowerOfTwoGrid) && !format.has_scale(ScaleKind::Tile) {
            return Some(Self::Fp8Blocks);
        }
        if format.has_scale(ScaleKind::Tile) || format.has_scale(ScaleKind::Group) {
            return Some(Self::Bf16Rows);
        }
        if format.is_unscaled() {
            return Some(Self::Bf16Rows);
        }
        None
    }

    /// This form, or the per-128-column form when the pass is a prompt and the part has it.
    #[must_use]
    pub const fn for_pass(self, prompt_rows: bool, has_q8_body: bool) -> Self {
        if prompt_rows && has_q8_body && matches!(self as u8, _) {
            return Self::Fp8Per128;
        }
        self
    }
}

/// The empty escape list the bind carries for a packed tensor before the loader fills it.
const NO_ESCAPES: &[u32] = &[];

/// The scale planes a format's tensors carry, sized for one operand geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalePlanes {
    /// The bytes of the tile or power-of-two block grid, `0` when the format has none.
    pub grid_bytes: u64,
    /// The bytes of the row scale plane, `0` when the format has none.
    pub row_bytes: u64,
    /// The bytes of the per-group plane (NVFP4's one E4M3 scale a 16-value group, per row),
    /// `0` when the format has none.
    pub group_bytes: u64,
    /// The tensor-wide convention the bind carries, if any.
    ///
    /// Qwen's two factors are the weight global and the static activation global. Inkling's are
    /// the per-expert amax and the outer denominator: the amax is per expert tensor, not per
    /// block, so the routed-expert bind carries one f32 an expert and the body reads it by the
    /// local expert index. [ScalePlanes::total_bytes] counts the two f32 a tensor.
    pub tensor_wide: NvFp4Wide,
    /// The escape list a packed tensor carries, when its layout has one.
    pub escapes: Option<&'static [u32]>,
}

impl ScalePlanes {
    /// The planes one `rows` x `cols` tensor of `format` carries.
    #[must_use]
    pub const fn of(format: StoredFormat, rows: usize, cols: usize) -> Self {
        if format.is_unscaled() {
            return Self {
                grid_bytes: 0,
                row_bytes: 0,
                group_bytes: 0,
                tensor_wide: NvFp4Wide::None,
                escapes: if format.layout() as u8 == FormatLayout::Packed as u8 {
                    Some(NO_ESCAPES)
                } else {
                    None
                },
            };
        }
        let block = if format.block() == 0 {
            1
        } else {
            format.block()
        };
        let bytes = format.scale_bytes() as u64;
        let grid_entries = if format.has_scale(ScaleKind::Row) || format.has_scale(ScaleKind::Group)
        {
            0
        } else {
            rows.div_ceil(block) * cols.div_ceil(block)
        };
        let row_entries = if format.has_scale(ScaleKind::Row) {
            rows * cols.div_ceil(block)
        } else {
            0
        };
        let group_entries = if format.has_scale(ScaleKind::Group) {
            rows * cols.div_ceil(block)
        } else {
            0
        };
        Self {
            grid_bytes: grid_entries as u64 * bytes,
            row_bytes: row_entries as u64 * bytes,
            group_bytes: group_entries as u64 * bytes,
            tensor_wide: format.tensor_wide(),
            // A packed tensor's escape list is the one plane it carries: the values are unscaled,
            // and the list is loaded with the tensor rather than sized from its geometry. The
            // empty list here is the bind's statement that the plane exists; `arrange` fills it
            // from the store it is handed.
            escapes: if format.layout() as u8 == FormatLayout::Packed as u8 {
                Some(NO_ESCAPES)
            } else {
                None
            },
        }
    }

    /// The plane bytes together, the bytes a bind must reserve beside the values.
    #[must_use]
    pub const fn total_bytes(self) -> u64 {
        self.grid_bytes
            .saturating_add(self.row_bytes)
            .saturating_add(self.group_bytes)
            .saturating_add(self.tensor_wide.factors() as u64 * 4)
    }
}

/// A tensor arranged for the body that reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arranged {
    /// The bytes in the body's order.
    pub bytes: Vec<u8>,
    /// The layout the bytes are in.
    pub layout: FormatLayout,
    /// The escape list the arrangement read, empty for a layout that has none.
    pub escapes: Vec<u32>,
}

/// One stored tensor laid out for its resolved body: the one place the arrangement happens.
///
/// A row-major format is handed over as stored; the 32 x 32 tiling is DeepSeek's B12x
/// permutation (padding zeroed); the packed word form is unpacked here into the BF16 rows the
/// body already reads, its escape list resolving what the packed words do not carry.
///
/// # Errors
/// The store is not the length the format and its escape list describe, the values are not the
/// layout's, or a layout this build does not arrange.
pub fn arrange(
    format: StoredFormat,
    stored: &[u8],
    rows: usize,
    cols: usize,
    escapes: &[u32],
) -> Result<Arranged, String> {
    match format.layout() {
        FormatLayout::Packed => {
            if format.values() != FormatValues::Bf16 {
                return Err(format!(
                    "the packed word form is a BF16 arrangement; {} is {} values",
                    format.name(),
                    format.values().name()
                ));
            }
            let bytes = sparkflash_model::packed_bf16::unpack(rows, cols, stored, escapes)
                .map_err(|error| {
                    format!(
                        "a {rows} x {cols} {} tensor does not match its escape list: {error}",
                        format.name()
                    )
                })?;
            Ok(Arranged {
                bytes,
                layout: FormatLayout::RowMajor,
                escapes: escapes.to_vec(),
            })
        }
        FormatLayout::RowMajor => {
            let expected = rows
                .saturating_mul(cols)
                .saturating_mul(format.values().bytes_per_value());
            if stored.len() != expected {
                return Err(format!(
                    "a {rows} x {cols} {} tensor is {expected} bytes, the store holds {}",
                    format.name(),
                    stored.len()
                ));
            }
            Ok(Arranged {
                bytes: stored.to_vec(),
                layout: FormatLayout::RowMajor,
                escapes: Vec::new(),
            })
        }
        FormatLayout::Tile32 => {
            if format.values() != FormatValues::Fp8E4M3 {
                return Err(format!(
                    "the 32 x 32 tiling is an E4M3 arrangement; {} is {} values",
                    format.name(),
                    format.values().name()
                ));
            }
            let expected = rows.saturating_mul(cols);
            if stored.len() != expected {
                return Err(format!(
                    "a {rows} x {cols} {} tensor is {expected} bytes, the store holds {}",
                    format.name(),
                    stored.len()
                ));
            }
            let tiled = sparkflash_kernels::mxfp8_ref::b12x_tile_weights(stored, rows, cols);
            Ok(Arranged {
                bytes: tiled,
                layout: FormatLayout::Tile32,
                escapes: Vec::new(),
            })
        }
    }
}

/// Whether a format is read as the 32 x 32 B12x tiling: the values, the 32-value power-of-two
/// grid and the tiled layout, which is the operand pair a B12x-weight GEMM binds.
#[must_use]
pub fn is_b12x(format: Option<StoredFormat>) -> bool {
    matches!(format, Some(format)
        if format.values() == FormatValues::Fp8E4M3
            && format.block() == 32
            && format.layout() == FormatLayout::Tile32
            && format.has_scale(ScaleKind::PowerOfTwoGrid))
}

/// One part instance's operand: the stored format, the activation form its seam ends in, the
/// body that reads it and the scale planes the bind carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Operand {
    /// The format the checkpoint stores the part's weights in.
    pub stored: StoredFormat,
    /// The form the seam before this part must leave its output in.
    pub activation: ActivationForm,
    /// The tensor-wide convention the bind carries. For Inkling this includes the per-expert
    /// amax, which is one f32 per expert tensor (not per block); the routed-expert body reads it
    /// indexed by the local expert and applies `amax / outer_denominator` to the output.
    pub tensor_wide: NvFp4Wide,
    /// The catalogue symbol of the body, when the plan claims one that decodes the format.
    pub body: Option<&'static str>,
}

/// The body one part instance's format resolves to, from the symbols the plan claims for it.
///
/// The choice is per part instance and per block: two blocks of one stack may store different
/// formats (Inkling's experts change part-way up), and the catalogue is keyed by the format each
/// symbol's build recorded. A format no claimed symbol decodes is `None` - a row the scan shows
/// as not built - never the nearest body.
#[must_use]
pub fn choose(
    part: Part,
    stored: StoredFormat,
    claimed: &[&'static str],
    formats: &dyn Fn(&str) -> Option<StoredFormat>,
) -> Option<&'static str> {
    let wanted = ActivationForm::of(stored)?;
    claimed.iter().copied().find(|symbol| {
        let decodes = formats(symbol).is_some_and(|declared| {
            declared.same_arithmetic(stored) || (declared.is_unscaled() && stored.is_unscaled())
        });
        decodes && body_serves(part, wanted)
    })
}

/// Whether a part's body of this tree serves an activation form. The dense, shared and routed
/// bodies read the block-quantised seam or BF16 rows; the tile bodies read BF16 rows or the
/// per-128-column quantised rows.
const fn body_serves(part: Part, activation: ActivationForm) -> bool {
    match activation {
        ActivationForm::Fp8Blocks | ActivationForm::Bf16Rows | ActivationForm::Fp8Per128 => {
            let _ = part;
            true
        }
    }
}

/// The scale-value name a ledger prints for a format.
#[must_use]
pub const fn scale_name(format: StoredFormat) -> &'static str {
    format.scale_values().name()
}

/// The scale values a format holds, as the bind's evidence names them.
#[must_use]
pub const fn scale_values(format: StoredFormat) -> ScaleValues {
    format.scale_values()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_released_format_is_the_coded_block_seam_and_a_tile_is_bf16_rows() {
        assert_eq!(
            ActivationForm::of(StoredFormat::MXFP8_TILED_32),
            Some(ActivationForm::Fp8Blocks)
        );
        assert_eq!(
            ActivationForm::of(StoredFormat::FP8_E4M3_TILE_128),
            Some(ActivationForm::Bf16Rows)
        );
        assert_eq!(
            ActivationForm::of(StoredFormat::FP8_E4M3_TILE_128_BF16),
            Some(ActivationForm::Bf16Rows)
        );
        assert_eq!(
            ActivationForm::of(StoredFormat::BF16),
            Some(ActivationForm::Bf16Rows)
        );
    }

    #[test]
    fn the_scale_planes_are_sized_by_the_format_not_by_a_dtype() {
        // The released 32 x 32 grid: one byte a block.
        let released = ScalePlanes::of(StoredFormat::MXFP8_TILED_32, 1280, 5120);
        assert_eq!(released.grid_bytes, 40 * 160);
        assert_eq!(released.row_bytes, 0);
        assert_eq!(released.tensor_wide, NvFp4Wide::None);
        // A 128-tile f32 scale: sixteen bytes a tile.
        let f32_tile = ScalePlanes::of(StoredFormat::FP8_E4M3_TILE_128, 4096, 5120);
        assert_eq!(f32_tile.grid_bytes, 32 * 40 * 4);
        // The same tensor with a bf16 tile scale is half the plane.
        let bf16_tile = ScalePlanes::of(StoredFormat::FP8_E4M3_TILE_128_BF16, 4096, 5120);
        assert_eq!(bf16_tile.grid_bytes, 32 * 40 * 2);
        // An expert's row scale is one plane, not a square grid.
        let expert = ScalePlanes::of(StoredFormat::MXFP4_ROW_32, 512, 5120);
        assert_eq!(expert.row_bytes, 512 * 160);
        assert_eq!(expert.grid_bytes, 0);
        // NVFP4's per-16 E4M3 group plane is one row per output row, and two f32 factors ride
        // beside it: Qwen's two globals, or Inkling's amax and outer denominator.
        let qwen = ScalePlanes::of(StoredFormat::NVFP4_QWEN_16, 512, 5120);
        assert_eq!(qwen.group_bytes, 512 * 320);
        assert_eq!(qwen.grid_bytes, 0);
        assert_eq!(qwen.tensor_wide, NvFp4Wide::Qwen);
        assert_eq!(qwen.total_bytes(), 512 * 320 + 8);
        let inkling = ScalePlanes::of(StoredFormat::NVFP4_INKLING_16, 512, 5120);
        assert_eq!(inkling.group_bytes, qwen.group_bytes);
        assert_eq!(inkling.tensor_wide, NvFp4Wide::Inkling);
    }

    #[test]
    fn the_two_nvfp4_conventions_are_not_one_format() {
        assert!(!StoredFormat::NVFP4_QWEN_16.same_arithmetic(StoredFormat::NVFP4_INKLING_16));
        assert_eq!(
            ActivationForm::of(StoredFormat::NVFP4_QWEN_16),
            Some(ActivationForm::Bf16Rows)
        );
        let names = |symbol: &str| match symbol {
            "qwen_nvfp4" => Some(StoredFormat::NVFP4_QWEN_16),
            "inkling_nvfp4" => Some(StoredFormat::NVFP4_INKLING_16),
            _ => None,
        };
        assert_eq!(
            choose(
                Part::Moe,
                StoredFormat::NVFP4_INKLING_16,
                &["qwen_nvfp4", "inkling_nvfp4"],
                &names
            ),
            Some("inkling_nvfp4")
        );
        assert_eq!(
            choose(Part::Moe, StoredFormat::NVFP4_16, &["qwen_nvfp4"], &names),
            Some("qwen_nvfp4")
        );
        assert_eq!(
            choose(
                Part::Moe,
                StoredFormat::NVFP4_16,
                &["no_recorded_format"],
                &|_| None
            ),
            None,
            "a symbol with no recorded format is not a wildcard"
        );
    }

    #[test]
    fn arranging_a_row_major_format_is_the_identity() {
        let bytes: Vec<u8> = (0..32u8).collect();
        let arranged =
            arrange(StoredFormat::BF16, &bytes, 4, 4, &[]).expect("an identity arrangement");
        assert_eq!(arranged.bytes, bytes);
        assert_eq!(arranged.layout, FormatLayout::RowMajor);
        assert!(arranged.escapes.is_empty());
    }

    #[test]
    fn arranging_a_tiled_format_permutes_and_pads_but_loses_nothing() {
        // No zero in the store, so the padding the tiling adds is the only zero in the result.
        let bytes: Vec<u8> = (0..(33 * 32) as u32).map(|v| (v % 255) as u8 + 1).collect();
        let arranged = arrange(StoredFormat::MXFP8_TILED_32, &bytes, 33, 32, &[])
            .expect("the tile arrangement");
        assert_eq!(
            arranged.bytes.len(),
            64 * 32,
            "rows pad to whole 32-row tiles"
        );
        let mut sorted: Vec<u8> = arranged.bytes.iter().copied().filter(|b| *b != 0).collect();
        sorted.sort_unstable();
        let mut original = bytes.clone();
        original.sort_unstable();
        assert_eq!(sorted, original, "the tiling is a permutation");
    }

    #[test]
    fn a_row_scaled_store_keeps_one_scale_a_row_and_arranges_as_stored() {
        // One ue8m0 code a row, every 32 columns: rows x (cols / 32) row bytes, no grid.
        let planes = ScalePlanes::of(StoredFormat::MXFP8_ROW_32, 512, 5120);
        assert_eq!(planes.row_bytes, 512 * 160);
        assert_eq!(planes.grid_bytes, 0);
        assert_eq!(
            planes.tensor_wide,
            sparkflash_model::format::NvFp4Wide::None
        );
        assert_eq!(planes.total_bytes(), 512 * 160);
        // The same tensor stored as a 32 x 32 grid is a different plane with a different size.
        let grid = ScalePlanes::of(StoredFormat::MXFP8_32, 512, 5120);
        assert_eq!(grid.grid_bytes, 16 * 160);
        assert_eq!(grid.row_bytes, 0);
        // E4M3 codes are a byte a value, so the arrangement is the stored bytes.
        let bytes: Vec<u8> = (0..(4 * 64) as u32)
            .map(|value| (value % 255) as u8 + 1)
            .collect();
        let arranged = arrange(StoredFormat::MXFP8_ROW_32, &bytes, 4, 64, &[])
            .expect("a row-major arrangement");
        assert_eq!(arranged.bytes, bytes);
        assert_eq!(arranged.layout, FormatLayout::RowMajor);
    }

    #[test]
    fn one_body_is_chosen_for_the_engram_table_and_an_expert_block() {
        // The Engram table and a routed expert both store E4M3 codes with one scale a row every
        // 32 columns, so the same claimed body decodes both; the part does not pick the decode.
        let claimed = ["other_body", "row_scaled_body"];
        let formats = |symbol: &str| match symbol {
            "other_body" => Some(StoredFormat::MXFP8_32),
            "row_scaled_body" => Some(StoredFormat::MXFP8_ROW_32),
            _ => None,
        };
        for part in [Part::Engram, Part::Moe, Part::LmHead] {
            assert_eq!(
                choose(part, StoredFormat::MXFP8_ROW_32, &claimed, &formats),
                Some("row_scaled_body"),
                "{part:?} reads the same row-scaled arrangement"
            );
        }
        assert_eq!(
            ActivationForm::of(StoredFormat::MXFP8_ROW_32),
            Some(ActivationForm::Fp8Blocks),
            "the row-scaled body reads the block-quantised activation seam"
        );
        assert_eq!(
            StoredFormat::from_name("mx-fp8-row"),
            Some(StoredFormat::MXFP8_ROW_32)
        );
        assert_eq!(
            StoredFormat::from_name("MX-FP8-ROW"),
            Some(StoredFormat::MXFP8_ROW_32),
            "a build file's spelling is case-insensitive"
        );
    }

    #[test]
    fn the_instance_list_builds_the_row_scaled_expert_body_the_plan_chooses() {
        // The build's own record, not a literal: the entry's format comes from kernels/instances.toml.
        let symbol = "opt_mma_fp8_rowscale_f32o";
        assert_eq!(
            sparkflash_ops::parts::built_format(symbol),
            Some(StoredFormat::MXFP8_ROW_32),
            "{symbol} is built for the row-scaled arrangement"
        );
        assert_eq!(
            choose(
                Part::Moe,
                StoredFormat::MXFP8_ROW_32,
                &[symbol],
                &sparkflash_ops::parts::built_format,
            ),
            Some(symbol),
            "a routed expert block of this format resolves to the body that reads it"
        );
    }

    #[test]
    fn a_short_store_is_refused_by_the_format() {
        assert!(arrange(StoredFormat::BF16, &[0u8; 4], 4, 4, &[]).is_err());
        assert!(arrange(StoredFormat::MXFP8_TILED_32, &[0u8; 4], 4, 4, &[]).is_err());
    }

    #[test]
    fn an_unscaled_packed_tensor_carries_only_its_escape_list() {
        let planes = ScalePlanes::of(StoredFormat::PACKED_BF16, 4096, 5120);
        assert_eq!(planes.grid_bytes, 0);
        assert_eq!(planes.row_bytes, 0);
        assert_eq!(planes.tensor_wide, NvFp4Wide::None);
        assert_eq!(planes.total_bytes(), 0);
        assert!(
            planes.escapes.is_some(),
            "the bind carries the escape plane"
        );
        assert!(
            ScalePlanes::of(StoredFormat::BF16, 4096, 5120)
                .escapes
                .is_none()
        );
    }

    #[test]
    fn arranging_the_packed_form_is_the_bf16_rows_it_stands_for() {
        let (rows, cols) = (3usize, 512usize);
        let mut bytes = Vec::with_capacity(rows * cols * 2);
        for index in 0..rows * cols {
            let value = 0.25 + (index % 19) as f32 * 0.125;
            let bits = (value.to_bits() >> 16) as u16;
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        let (packed, escapes) = sparkflash_model::packed_bf16::pack(rows, cols, &bytes)
            .expect("the form takes the shape");
        let arranged = arrange(StoredFormat::PACKED_BF16, &packed, rows, cols, &escapes)
            .expect("the packed arrangement");
        assert_eq!(arranged.bytes, bytes, "the body reads the same BF16 rows");
        assert_eq!(arranged.layout, FormatLayout::RowMajor);
        assert_eq!(arranged.escapes, escapes);
        // A store that is not the escape list's byte length is refused before any body sees it.
        let short = &packed[..packed.len() - 16];
        assert!(arrange(StoredFormat::PACKED_BF16, short, rows, cols, &escapes).is_err());
        let mut wrong = escapes.clone();
        wrong[1] = u32::MAX;
        assert!(arrange(StoredFormat::PACKED_BF16, &packed, rows, cols, &wrong).is_err());
    }

    #[test]
    fn the_bf16_body_is_chosen_for_both_names_of_the_one_arithmetic() {
        let claimed = ["bf16_dense_body"];
        let formats = |symbol: &str| match symbol {
            "bf16_dense_body" => Some(StoredFormat::BF16),
            _ => None,
        };
        for format in [StoredFormat::BF16, StoredFormat::PACKED_BF16] {
            assert_eq!(
                choose(Part::DenseMlp, format, &claimed, &formats),
                Some("bf16_dense_body"),
                "{} names the BF16 arithmetic",
                format.name()
            );
        }
        assert_eq!(
            StoredFormat::from_name("packed-bf16"),
            Some(StoredFormat::PACKED_BF16)
        );
        assert_eq!(StoredFormat::PACKED_BF16.name(), "PACKED-BF16");
    }

    #[test]
    fn the_body_is_chosen_for_the_format_the_part_instance_stores() {
        let format = StoredFormat::FP8_E4M3_TILE_128;
        let claimed = ["deepseek_body", "glm_q8_body"];
        let formats = |symbol: &str| match symbol {
            "deepseek_body" => Some(StoredFormat::MXFP8_TILED_32),
            "glm_q8_body" => Some(StoredFormat::FP8_E4M3_TILE_128),
            _ => None,
        };
        assert_eq!(
            choose(Part::DenseMlp, format, &claimed, &formats),
            Some("glm_q8_body"),
            "the body that decodes the stored arithmetic wins over the first claimed"
        );
        assert_eq!(
            choose(Part::DenseMlp, StoredFormat::MXFP4_32, &claimed, &formats),
            None,
            "a format no body decodes is a row the scan shows, not the nearest body"
        );
    }

    #[test]
    fn only_the_tiled_32_block_format_is_a_b12x_operand() {
        assert!(is_b12x(Some(StoredFormat::MXFP8_TILED_32)));
        assert!(
            !is_b12x(Some(StoredFormat::MXFP8_32)),
            "row-major is not tiled"
        );
        assert!(!is_b12x(Some(StoredFormat::FP8_E4M3_TILE_128)));
        assert!(!is_b12x(None));
    }
}

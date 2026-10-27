//! dtype-aware weight views for the fused coded matmuls.
//!
//! [WeightView] keeps the one permanent F32 fallback and adds the release path: an F8-E4M3 or
//! F4-E2M1 weight is read as its raw codes plus its raw E8M0 scale bytes and handed to
//! [crate::linalg]'s fused kernels, so no f32 weight buffer is ever materialised. BF16 weights
//! (head, embed, norm, the attention/compressor/indexer BF16 tensors) are read as raw BF16 bytes.
//!
//! The choice is made once per weight read from the source's served [DType], never per element:
//! a source that serves F32 (every fixture) takes [WeightView::F32] and the exact old
//! decode + apply_scales + matmul path, so fixtures stay byte-identical.

use std::borrow::Cow;

use crate::dtype::DType;
use crate::linalg::{self, EngineError, ScaleGrid};
use crate::tensor::TensorSpec;
use crate::weights::WeightSource;

/// One weight, in the representation its source stores it in.
pub(crate) enum WeightView<'a> {
    /// A materialised f32 weight (the generic/fixture path).
    F32(Vec<f32>),
    /// Raw F8-E4M3 or F4-E2M1 codes and their raw E8M0 grid.
    Coded {
        /// Code dtype: [DType::F8E4M3] or [DType::F4E2M1].
        dtype: DType,
        /// The stored codes, in the dtype's packing.
        codes: Cow<'a, [u8]>,
        /// Raw E8M0 scale bytes for `grid`.
        scales: Cow<'a, [u8]>,
        /// The grid those scale bytes follow.
        grid: ScaleGrid,
    },
    /// Raw little-endian BF16 bytes, no scale grid.
    Bf16(Cow<'a, [u8]>),
    /// Raw little-endian BF16 bytes in a raw E8M0 grid.
    Bf16Scaled {
        /// The stored BF16 bytes.
        data: Cow<'a, [u8]>,
        /// Raw E8M0 scale bytes for `grid`.
        scales: Cow<'a, [u8]>,
        /// The grid those scale bytes follow.
        grid: ScaleGrid,
    },
}

/// The number of logical elements a weight must have for this geometry.
fn expected_elements(rows: usize, cols: usize) -> Result<usize, EngineError> {
    rows.checked_mul(cols).ok_or_else(|| EngineError::Shape {
        what: "weight view".to_owned(),
        detail: "rows times cols overflows".to_owned(),
    })
}

/// Checks one tensor's element limit and returns its spec.
fn checked_spec(
    source: &dyn WeightSource,
    name: &str,
    limit: usize,
) -> Result<TensorSpec, EngineError> {
    let spec = source.spec(name)?;
    let elements = spec.elements();
    if elements > limit as u128 {
        return Err(EngineError::TooLarge {
            name: name.to_owned(),
            elements: usize::try_from(elements).unwrap_or(usize::MAX),
            limit,
        });
    }
    Ok(spec)
}

/// The exact old path: decode the weight (and its scale, if present) to f32 and apply the grid.
///
/// This is what every F32 fixture serves, and what any dtype without a fused kernel falls back
/// to; it is deliberately the same calls in the same order as the old call sites.
fn materialised(
    source: &dyn WeightSource,
    weight_name: &str,
    scale_name: Option<&str>,
    rows: usize,
    cols: usize,
    grid: ScaleGrid,
    limit: usize,
) -> Result<Vec<f32>, EngineError> {
    let mut values = linalg::decode_by_name(source, weight_name, limit)?;
    if let Some(scale_name) = scale_name {
        let scales = linalg::decode_by_name(source, scale_name, limit)?;
        linalg::apply_scales(&mut values, rows, cols, &scales, grid)?;
    }
    Ok(values)
}

/// The stored bytes of one tensor: borrowed from a resident source, owned otherwise.
fn read_bytes<'a>(source: &'a dyn WeightSource, name: &str) -> Result<Cow<'a, [u8]>, EngineError> {
    match source.read_view(name)? {
        Some(bytes) => Ok(Cow::Borrowed(bytes)),
        None => Ok(Cow::Owned(source.read(name)?)),
    }
}

/// Reads one weight and its optional scale grid into a [WeightView].
///
/// The dtype branch happens here, once per weight read. Shape and limit checks are also once:
/// the kernels then only see slices.
fn read<'a>(
    source: &'a dyn WeightSource,
    weight_name: &str,
    scale_name: Option<&str>,
    rows: usize,
    cols: usize,
    grid: ScaleGrid,
    limit: usize,
) -> Result<WeightView<'a>, EngineError> {
    let spec = checked_spec(source, weight_name, limit)?;
    let elements = expected_elements(rows, cols)?;
    if spec.elements() != elements as u128 {
        return Err(EngineError::Shape {
            what: "weight view".to_owned(),
            detail: format!(
                "{weight_name} holds {} elements, but {rows}x{cols} needs {elements}",
                spec.elements()
            ),
        });
    }
    let scale_spec = match scale_name {
        Some(name) => Some(checked_spec(source, name, limit)?),
        None => None,
    };
    match (spec.dtype, scale_spec.map(|spec| spec.dtype)) {
        // The fixture/generic path: exactly the old decode + apply_scales + matmul arithmetic.
        (DType::F32, _) | (_, Some(DType::F32)) => Ok(WeightView::F32(materialised(
            source,
            weight_name,
            scale_name,
            rows,
            cols,
            grid,
            limit,
        )?)),
        (DType::F8E4M3, Some(DType::F8E8M0)) | (DType::F4E2M1, Some(DType::F8E8M0)) => {
            let codes = read_bytes(source, weight_name)?;
            let expected_bytes = if spec.dtype == DType::F4E2M1 {
                elements.div_ceil(2)
            } else {
                elements
            };
            if codes.len() < expected_bytes {
                return Err(EngineError::Shape {
                    what: "weight view".to_owned(),
                    detail: format!(
                        "{weight_name} needs {expected_bytes} code bytes, got {}",
                        codes.len()
                    ),
                });
            }
            let scales = read_bytes(source, scale_name.expect("the coded arm has a grid"))?;
            Ok(WeightView::Coded {
                dtype: spec.dtype,
                codes,
                scales,
                grid,
            })
        }
        (DType::BF16, None) => {
            let data = read_bytes(source, weight_name)?;
            let bytes = elements.checked_mul(2).ok_or_else(|| EngineError::Shape {
                what: "weight view".to_owned(),
                detail: "bf16 byte count overflows".to_owned(),
            })?;
            if data.len() < bytes {
                return Err(EngineError::Shape {
                    what: "weight view".to_owned(),
                    detail: format!("{weight_name} needs {bytes} bytes, got {}", data.len()),
                });
            }
            Ok(WeightView::Bf16(data))
        }
        (DType::BF16, Some(DType::F8E8M0)) => {
            let data = read_bytes(source, weight_name)?;
            let bytes = elements.checked_mul(2).ok_or_else(|| EngineError::Shape {
                what: "weight view".to_owned(),
                detail: "bf16 byte count overflows".to_owned(),
            })?;
            if data.len() < bytes {
                return Err(EngineError::Shape {
                    what: "weight view".to_owned(),
                    detail: format!("{weight_name} needs {bytes} bytes, got {}", data.len()),
                });
            }
            let scales = read_bytes(source, scale_name.expect("the scaled arm has a grid"))?;
            Ok(WeightView::Bf16Scaled { data, scales, grid })
        }
        // Every other dtype keeps the materialised path, so its arithmetic cannot move.
        _ => Ok(WeightView::F32(materialised(
            source,
            weight_name,
            scale_name,
            rows,
            cols,
            grid,
            limit,
        )?)),
    }
}

impl<'a> WeightView<'a> {
    /// The raw F4-E2M1 codes and E8M0 scales when this view is a packed F4 expert weight.
    pub(crate) fn f4_parts(&self) -> Option<(&[u8], &[u8])> {
        match self {
            WeightView::Coded {
                dtype: DType::F4E2M1,
                codes,
                scales,
                ..
            } => Some((codes.as_ref(), scales.as_ref())),
            _ => None,
        }
    }

    /// The scale grid this view decodes with, when it has one.
    pub(crate) fn grid(&self) -> Option<ScaleGrid> {
        match self {
            WeightView::Coded { grid, .. } | WeightView::Bf16Scaled { grid, .. } => Some(*grid),
            WeightView::F32(_) | WeightView::Bf16(_) => None,
        }
    }

    /// Reads `{name}.weight` and its required `{name}.scale` grid.
    pub(crate) fn read_dense(
        source: &'a dyn WeightSource,
        name: &str,
        rows: usize,
        cols: usize,
        grid: ScaleGrid,
        limit: usize,
    ) -> Result<WeightView<'a>, EngineError> {
        let weight_name = format!("{name}.weight");
        let scale_name = format!("{name}.scale");
        Self::read_impl(
            source,
            &weight_name,
            Some(&scale_name),
            rows,
            cols,
            grid,
            limit,
        )
    }

    /// Reads `{name}.weight` and applies `{name}.scale` only when the source has one.
    pub(crate) fn read_optional_scale(
        source: &'a dyn WeightSource,
        name: &str,
        rows: usize,
        cols: usize,
        grid: ScaleGrid,
        limit: usize,
    ) -> Result<WeightView<'a>, EngineError> {
        let weight_name = format!("{name}.weight");
        let scale_name = format!("{name}.scale");
        let scale = if source.contains(&scale_name) {
            Some(scale_name.as_str())
        } else {
            None
        };
        Self::read_impl(source, &weight_name, scale, rows, cols, grid, limit)
    }

    /// Reads a plain (unscaled) weight, e.g. a BF16 head or embedding matrix.
    pub(crate) fn read_plain(
        source: &'a dyn WeightSource,
        name: &str,
        rows: usize,
        cols: usize,
        limit: usize,
    ) -> Result<WeightView<'a>, EngineError> {
        Self::read_impl(source, name, None, rows, cols, ScaleGrid::Block32, limit)
    }

    fn read_impl(
        source: &'a dyn WeightSource,
        weight_name: &str,
        scale_name: Option<&str>,
        rows: usize,
        cols: usize,
        grid: ScaleGrid,
        limit: usize,
    ) -> Result<WeightView<'a>, EngineError> {
        read(source, weight_name, scale_name, rows, cols, grid, limit)
    }

    /// `out = x * W` for this weight, through the fused kernel when the source is coded.
    pub(crate) fn matmul(
        &self,
        out: &mut [f32],
        x: &[f32],
        rows: usize,
        cols: usize,
        batch: usize,
    ) -> Result<(), EngineError> {
        match self {
            WeightView::F32(values) => linalg::matmul(out, x, values, rows, cols, batch),
            WeightView::Coded {
                dtype,
                codes,
                scales,
                grid,
            } => match dtype {
                DType::F8E4M3 => linalg::matmul_f8_scaled(
                    out,
                    x,
                    codes.as_ref(),
                    scales.as_ref(),
                    *grid,
                    rows,
                    cols,
                    batch,
                ),
                DType::F4E2M1 => linalg::matmul_f4_scaled(
                    out,
                    x,
                    codes.as_ref(),
                    scales.as_ref(),
                    *grid,
                    rows,
                    cols,
                    batch,
                ),
                other => Err(EngineError::Shape {
                    what: "weight view matmul".to_owned(),
                    detail: format!("coded arm cannot hold {other:?}"),
                }),
            },
            WeightView::Bf16(data) => linalg::matmul_bf16(out, x, data.as_ref(), rows, cols, batch),
            WeightView::Bf16Scaled { data, scales, grid } => linalg::matmul_bf16_scaled(
                out,
                x,
                data.as_ref(),
                scales.as_ref(),
                *grid,
                rows,
                cols,
                batch,
            ),
        }
    }

    /// Whether this weight can run as one block-diagonal grouped matmul.
    ///
    /// The grouped kernel multiplies the whole wide row by the whole grouped weight in one call,
    /// selecting the group per output row; the layout is `wo_a`'s [groups][o_lora_rank, group_in].
    pub(crate) fn supports_grouped(&self) -> bool {
        matches!(
            self,
            WeightView::Coded {
                dtype: DType::F8E4M3,
                ..
            }
        )
    }

    /// `out = x * W` as one block-diagonal grouped matmul, for a weight laid out [groups][rows, cols].
    ///
    /// `out` is `groups * rows_per_group` values, `x` is `groups * cols_per_group`; each output row
    /// sees its own group's rows and its group's slice of `x`. Call only when [supports_grouped].
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] when the slices do not match the grouped geometry.
    pub(crate) fn matmul_grouped(
        &self,
        out: &mut [f32],
        x: &[f32],
        groups: usize,
        rows_per_group: usize,
        cols_per_group: usize,
    ) -> Result<(), EngineError> {
        match self {
            WeightView::Coded {
                dtype: DType::F8E4M3,
                codes,
                scales,
                grid,
            } => linalg::matmul_f8_grouped(
                out,
                x,
                codes.as_ref(),
                scales.as_ref(),
                *grid,
                groups,
                rows_per_group,
                cols_per_group,
            ),
            _ => Err(EngineError::Shape {
                what: "weight view grouped matmul".to_owned(),
                detail: "the grouped arm holds an F8-E4M3 coded weight".to_owned(),
            }),
        }
    }

    /// Whether [WeightView::matmul_rows] can slice blocks of this geometry from this weight.
    ///
    /// Block32 scales are per 32-row block, so a block start must be a multiple of 32; a packed F4
    /// block must also start on a byte boundary. A weight whose geometry fails this check keeps the
    /// old materialised path, which is exact for any block size.
    pub(crate) fn row_blocks_supported(&self, block_rows: usize, cols: usize) -> bool {
        match self {
            WeightView::F32(_) | WeightView::Bf16(_) => true,
            WeightView::Coded { dtype, grid, .. } => match (dtype, grid) {
                (DType::F8E4M3, ScaleGrid::Block32) => block_rows.is_multiple_of(32),
                (DType::F4E2M1, ScaleGrid::PerRow32) => {
                    block_rows.is_multiple_of(2) || cols.is_multiple_of(2)
                }
                _ => false,
            },
            WeightView::Bf16Scaled { grid, .. } => {
                *grid == ScaleGrid::Block32 && block_rows.is_multiple_of(32)
            }
        }
    }

    /// `out = x * W[row_start .. row_start + rows]`, for a weight used as row blocks.
    ///
    /// This is the grouped output projection: `wo_a` holds `groups` consecutive row blocks of
    /// `o_lora_rank` rows each, and the old path sliced the materialised f32 matrix before each
    /// matmul. Slicing the stored representation instead is exact when the block starts on a grid
    /// boundary (Block32 scales are per 32-row block) or, for a plain weight, always; a source whose
    /// grid cannot be sliced this way keeps the materialised path.
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] when the block does not fit the weight or the grid cannot be sliced.
    pub(crate) fn matmul_rows(
        &self,
        out: &mut [f32],
        x: &[f32],
        rows: usize,
        cols: usize,
        batch: usize,
        row_start: usize,
    ) -> Result<(), EngineError> {
        match self {
            WeightView::F32(values) => {
                let start = row_start
                    .checked_mul(cols)
                    .ok_or_else(|| EngineError::Shape {
                        what: "weight view matmul".to_owned(),
                        detail: "row offset overflows".to_owned(),
                    })?;
                let block =
                    values
                        .get(start..start + rows * cols)
                        .ok_or_else(|| EngineError::Shape {
                            what: "weight view matmul".to_owned(),
                            detail: format!(
                                "rows {row_start}..{} do not fit {} f32 values",
                                row_start + rows,
                                values.len()
                            ),
                        })?;
                linalg::matmul(out, x, block, rows, cols, batch)
            }
            WeightView::Coded {
                dtype,
                codes,
                scales,
                grid,
            } => {
                let scale_cols = cols.div_ceil(32);
                match (dtype, grid) {
                    (DType::F8E4M3, ScaleGrid::Block32) if row_start.is_multiple_of(32) => {
                        let code_start = row_start * cols;
                        let scale_start = (row_start / 32) * scale_cols;
                        let block = codes.get(code_start..).ok_or_else(|| EngineError::Shape {
                            what: "weight view matmul".to_owned(),
                            detail: format!("row block {row_start} does not fit the codes"),
                        })?;
                        let block_scales =
                            scales
                                .get(scale_start..)
                                .ok_or_else(|| EngineError::Shape {
                                    what: "weight view matmul".to_owned(),
                                    detail: format!(
                                        "row block {row_start} does not fit the scales"
                                    ),
                                })?;
                        linalg::matmul_f8_scaled(
                            out,
                            x,
                            block,
                            block_scales,
                            ScaleGrid::Block32,
                            rows,
                            cols,
                            batch,
                        )
                    }
                    (DType::F4E2M1, ScaleGrid::PerRow32) => {
                        let element_start = row_start * cols;
                        if !element_start.is_multiple_of(2) {
                            return Err(EngineError::Shape {
                                what: "weight view matmul".to_owned(),
                                detail: "a packed F4 row block that starts mid-byte needs the materialised path".to_owned(),
                            });
                        }
                        let block =
                            codes
                                .get(element_start / 2..)
                                .ok_or_else(|| EngineError::Shape {
                                    what: "weight view matmul".to_owned(),
                                    detail: format!(
                                        "row block {row_start} does not fit the packed codes"
                                    ),
                                })?;
                        let scale_start = row_start * scale_cols;
                        let block_scales =
                            scales
                                .get(scale_start..)
                                .ok_or_else(|| EngineError::Shape {
                                    what: "weight view matmul".to_owned(),
                                    detail: format!(
                                        "row block {row_start} does not fit the scales"
                                    ),
                                })?;
                        linalg::matmul_f4_scaled(
                            out,
                            x,
                            block,
                            block_scales,
                            ScaleGrid::PerRow32,
                            rows,
                            cols,
                            batch,
                        )
                    }
                    _ => Err(EngineError::Shape {
                        what: "weight view matmul".to_owned(),
                        detail: "this grid cannot be sliced by row block".to_owned(),
                    }),
                }
            }
            WeightView::Bf16(data) => {
                let start = row_start
                    .checked_mul(cols)
                    .and_then(|v| v.checked_mul(2))
                    .ok_or_else(|| EngineError::Shape {
                        what: "weight view matmul".to_owned(),
                        detail: "row offset overflows".to_owned(),
                    })?;
                let block = data.get(start..).ok_or_else(|| EngineError::Shape {
                    what: "weight view matmul".to_owned(),
                    detail: format!("row block {row_start} does not fit the bf16 bytes"),
                })?;
                linalg::matmul_bf16(out, x, block, rows, cols, batch)
            }
            WeightView::Bf16Scaled { data, scales, grid } => {
                if *grid != ScaleGrid::Block32 || !row_start.is_multiple_of(32) {
                    return Err(EngineError::Shape {
                        what: "weight view matmul".to_owned(),
                        detail: "this scaled bf16 grid cannot be sliced by row block".to_owned(),
                    });
                }
                let scale_cols = cols.div_ceil(32);
                let start = row_start
                    .checked_mul(cols)
                    .and_then(|v| v.checked_mul(2))
                    .ok_or_else(|| EngineError::Shape {
                        what: "weight view matmul".to_owned(),
                        detail: "row offset overflows".to_owned(),
                    })?;
                let block = data.get(start..).ok_or_else(|| EngineError::Shape {
                    what: "weight view matmul".to_owned(),
                    detail: format!("row block {row_start} does not fit the bf16 bytes"),
                })?;
                let block_scales =
                    scales.get((row_start / 32) * scale_cols..).ok_or_else(|| {
                        EngineError::Shape {
                            what: "weight view matmul".to_owned(),
                            detail: format!("row block {row_start} does not fit the scales"),
                        }
                    })?;
                linalg::matmul_bf16_scaled(out, x, block, block_scales, *grid, rows, cols, batch)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::parallel::with_reference_threads;
    use crate::quant;
    use crate::tensor::TensorSpec;
    use crate::weights::WeightError;

    /// A source that serves one weight in its checkpoint storage dtype and one raw E8M0 grid.
    struct StoredSource {
        weight: TensorSpec,
        bytes: Vec<u8>,
        scale: Option<(TensorSpec, Vec<u8>)>,
    }

    impl WeightSource for StoredSource {
        fn names(&self) -> Vec<String> {
            let mut names = vec![self.weight.name.clone()];
            if let Some((spec, _)) = &self.scale {
                names.push(spec.name.clone());
            }
            names
        }

        fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
            if name == self.weight.name {
                return Ok(self.weight.clone());
            }
            if let Some((spec, _)) = self.scale.as_ref().filter(|(spec, _)| spec.name == name) {
                return Ok(spec.clone());
            }
            Err(WeightError::Missing {
                name: name.to_owned(),
            })
        }

        fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
            if name == self.weight.name {
                return Ok(self.bytes.clone());
            }
            if let Some((_, bytes)) = self.scale.as_ref().filter(|(spec, _)| spec.name == name) {
                return Ok(bytes.clone());
            }
            Err(WeightError::Missing {
                name: name.to_owned(),
            })
        }
    }

    fn spec(name: &str, dtype: DType, shape: &[usize]) -> TensorSpec {
        TensorSpec {
            name: name.to_owned(),
            dtype,
            shape: shape.to_vec(),
        }
    }

    fn scale_bytes(grid: ScaleGrid, rows: usize, cols: usize) -> usize {
        let scale_cols = cols.div_ceil(32);
        match grid {
            ScaleGrid::Block32 => rows.div_ceil(32) * scale_cols,
            ScaleGrid::PerRow32 => rows * scale_cols,
        }
    }

    /// One deterministic finite value pattern shared by the coded patterns.
    fn pattern(index: usize) -> f32 {
        0.01 + 0.000_1 * ((index.wrapping_mul(37)) % 256) as f32
    }

    fn f8_source(name: &str, rows: usize, cols: usize, grid: ScaleGrid) -> StoredSource {
        let mut codes = Vec::with_capacity(rows * cols);
        for index in 0..rows * cols {
            codes.push(quant::e4m3_encode(pattern(index)));
        }
        let mut scales = Vec::with_capacity(scale_bytes(grid, rows, cols));
        for index in 0..scale_bytes(grid, rows, cols) {
            // E8M0 exponent bytes in a narrow finite range around 1.0.
            scales.push(127 - (index % 4) as u8);
        }
        StoredSource {
            weight: spec(&format!("{name}.weight"), DType::F8E4M3, &[rows, cols]),
            bytes: codes,
            scale: Some((
                spec(&format!("{name}.scale"), DType::F8E8M0, &[scales.len()]),
                scales,
            )),
        }
    }

    fn f4_source(name: &str, rows: usize, cols: usize) -> StoredSource {
        let elements = rows * cols;
        let mut codes = vec![0u8; elements.div_ceil(2)];
        for index in 0..elements {
            // Nibbles 0..8 are finite E2M1 values; keep both halves finite.
            let nibble = (index % 8) as u8;
            if index.is_multiple_of(2) {
                codes[index / 2] = nibble;
            } else {
                codes[index / 2] |= nibble << 4;
            }
        }
        let mut scales = Vec::with_capacity(rows * cols.div_ceil(32));
        for index in 0..rows * cols.div_ceil(32) {
            scales.push(127 - (index % 4) as u8);
        }
        StoredSource {
            weight: spec(&format!("{name}.weight"), DType::F4E2M1, &[rows, cols]),
            bytes: codes,
            scale: Some((
                spec(&format!("{name}.scale"), DType::F8E8M0, &[scales.len()]),
                scales,
            )),
        }
    }

    fn bf16_source(name: &str, rows: usize, cols: usize) -> StoredSource {
        let mut bytes = Vec::with_capacity(rows * cols * 2);
        for index in 0..rows * cols {
            let value = quant::bf16_round(pattern(index));
            bytes.extend_from_slice(&quant::f32_to_bf16_bits(value).to_le_bytes());
        }
        StoredSource {
            weight: spec(name, DType::BF16, &[rows, cols]),
            bytes,
            scale: None,
        }
    }

    fn old_weight(
        source: &StoredSource,
        weight_name: &str,
        scale_name: Option<&str>,
        rows: usize,
        cols: usize,
        grid: ScaleGrid,
    ) -> Vec<f32> {
        let mut values = linalg::decode_by_name(source, weight_name, usize::MAX).expect("decode");
        if let Some(scale_name) = scale_name {
            let scales = linalg::decode_by_name(source, scale_name, usize::MAX).expect("scales");
            linalg::apply_scales(&mut values, rows, cols, &scales, grid).expect("apply");
        }
        values
    }

    fn fnv(values: &[f32]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for value in values {
            for byte in value.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    fn check(
        source: &StoredSource,
        weight_name: &str,
        scale_name: Option<&str>,
        rows: usize,
        cols: usize,
        grid: ScaleGrid,
        batch: usize,
    ) {
        let old = old_weight(source, weight_name, scale_name, rows, cols, grid);
        let view = match scale_name {
            Some(_) => WeightView::read_dense(
                source,
                weight_name.trim_end_matches(".weight"),
                rows,
                cols,
                grid,
                usize::MAX,
            )
            .expect("fused view"),
            None => WeightView::read_plain(source, weight_name, rows, cols, usize::MAX)
                .expect("plain view"),
        };
        let x: Vec<f32> = (0..batch * cols)
            .map(|index| 0.001 * (index as f32) - 0.25)
            .collect();
        let mut expected = vec![0.0f32; batch * rows];
        linalg::matmul(&mut expected, &x, &old, rows, cols, batch).expect("old matmul");
        let mut actual = vec![0.0f32; batch * rows];
        view.matmul(&mut actual, &x, rows, cols, batch)
            .expect("fused matmul");
        assert_eq!(actual.len(), expected.len());
        let mismatches = actual
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            mismatches,
            0,
            "{weight_name} {rows}x{cols} batch {batch} differs in {mismatches} values; fnv old {:016x} new {:016x}",
            fnv(&expected),
            fnv(&actual)
        );
    }

    #[test]
    fn fused_views_are_bit_identical_to_the_materialised_path() {
        for threads in [1usize, 20] {
            with_reference_threads(threads, || {
                // The released dense Linear shapes: 8192x4096, 5120x8192, 32768x1280, 1280x5120,
                // 512x5120, plus a small serial shape.
                let shapes = [
                    (8192usize, 4096usize),
                    (5120, 8192),
                    (32768, 1280),
                    (1280, 5120),
                    (512, 5120),
                    (64, 128),
                ];
                for (rows, cols) in shapes {
                    let source = f8_source("dense", rows, cols, ScaleGrid::Block32);
                    check(
                        &source,
                        "dense.weight",
                        Some("dense.scale"),
                        rows,
                        cols,
                        ScaleGrid::Block32,
                        1,
                    );
                }
                // The released routed-expert shapes are packed F4 with a PerRow32 grid.
                for (rows, cols) in [(2304usize, 2560usize), (5120, 1152)] {
                    let source = f4_source("expert", rows, cols);
                    check(
                        &source,
                        "expert.weight",
                        Some("expert.scale"),
                        rows,
                        cols,
                        ScaleGrid::PerRow32,
                        1,
                    );
                }
                // BF16 head, plus a batch to exercise the batched arm.
                let (rows, cols) = (1024usize, 5120usize);
                let source = bf16_source("head.weight", rows, cols);
                check(
                    &source,
                    "head.weight",
                    None,
                    rows,
                    cols,
                    ScaleGrid::Block32,
                    2,
                );
            });
        }
    }

    #[test]
    fn grouped_f8_matmul_is_bit_identical_to_the_per_group_calls() {
        // wo_a layout: groups x o_lora_rank rows, each group's columns are its own wide-row slice.
        let groups = 8usize;
        let rows_per_group = 1024usize;
        let cols_per_group = 4096usize;
        let rows = groups * rows_per_group;
        let source = f8_source("wo_a", rows, cols_per_group, ScaleGrid::Block32);
        let view = WeightView::read_dense(
            &source,
            "wo_a",
            rows,
            cols_per_group,
            ScaleGrid::Block32,
            usize::MAX,
        )
        .expect("the fused view");
        assert!(view.supports_grouped());
        let x: Vec<f32> = (0..groups * cols_per_group)
            .map(|index| 0.001 * (index % 97) as f32 - 0.04)
            .collect();
        let mut grouped = vec![0.0f32; rows];
        view.matmul_grouped(&mut grouped, &x, groups, rows_per_group, cols_per_group)
            .expect("grouped matmul");
        let mut per_group = vec![0.0f32; rows];
        for group in 0..groups {
            let row_start = group * rows_per_group;
            let source_x = &x[group * cols_per_group..][..cols_per_group];
            view.matmul_rows(
                &mut per_group[row_start..][..rows_per_group],
                source_x,
                rows_per_group,
                cols_per_group,
                1,
                row_start,
            )
            .expect("per-group matmul");
        }
        let mismatches = grouped
            .iter()
            .zip(&per_group)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            mismatches, 0,
            "grouped and per-group differ in {mismatches} values"
        );
    }
}

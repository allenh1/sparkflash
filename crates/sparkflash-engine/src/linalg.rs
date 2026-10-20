//! The CPU reference arithmetic every forward pass here is assembled from.
//!
//! The forward is a *reference*: it exists so the engine's shape flow, masking, causality and
//! determinism can be tested on a machine with no GPU and no checkpoint. It therefore works on
//! `f32` slices, takes caller-owned output buffers so nothing here allocates per token, and reads
//! each weight through the public trait.
//!
//! Every function that reads a tensor takes an ELEMENT LIMIT. A reference pass that silently
//! materialised a 600 MB Engram table would be a claim this crate cannot support on a host without
//! weights, so the limit makes the refusal explicit and testable instead of implied by a comment
//! (the same rule the loader applies to the checkpoint it opens).
//!
//! Quantisation is not re-implemented here: [crate::quant] already decodes every storage format, and
//! [crate::dtype] already owns the block rules. This module only *applies* a decoded E8M0 grid.

use crate::dtype::DType;
use crate::parallel::{par_aligned_mut, par_chunks_mut};
use crate::quant::{self, QuantError};
use crate::weights::{TensorView, WeightError, WeightSource};

/// Everything that can stop a forward pass.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// The weight source refused a read.
    #[error(transparent)]
    Weights(#[from] WeightError),
    /// A stored value could not be decoded.
    #[error(transparent)]
    Quant(#[from] QuantError),
    /// The runtime configuration is not usable.
    #[error(transparent)]
    Config(#[from] crate::config::ConfigError),
    /// A tensor the pass needs is not in the source.
    #[error("the source has no tensor named {name}")]
    Missing {
        /// The tensor name that was asked for.
        name: String,
    },
    /// A tensor is larger than the reference pass will materialise.
    #[error(
        "{name} holds {elements} elements, above the {limit}-element limit of the reference pass"
    )]
    TooLarge {
        /// The tensor that was refused.
        name: String,
        /// Its element count.
        elements: usize,
        /// The limit in force.
        limit: usize,
    },
    /// A shape that does not fit the operation being applied.
    #[error("{what}: {detail}")]
    Shape {
        /// The operation.
        what: String,
        /// Why the shapes do not fit.
        detail: String,
    },
}

/// How an E8M0 scale grid maps onto a dense weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleGrid {
    /// One scale per 32x32 block: `scale[r / 32][c / 32]`. This is the released fp8 grid for the
    /// dense Linears (`[ceil(out/32), ceil(in/32)]`, root config `weight_block_size: [32, 32]`).
    Block32,
    /// One scale per 32 elements along K, per output row: `scale[r][c / 32]`. This is the released
    /// FP4 expert grid (`[out, ceil(in/32)]`, kernel.py:582).
    PerRow32,
}

/// The element limit a forward pass applies to a single tensor unless the caller says otherwise.
///
/// 32 Mi elements is 128 MiB of `f32`: comfortably above every tensor of a test configuration and
/// far below the released Engram tables (98 GB each), so a real checkpoint is refused rather than
/// silently paged in.
pub const REFERENCE_TENSOR_LIMIT: usize = 32 * 1024 * 1024;

/// The element limit for a real-checkpoint reference run.
///
/// The released plan's largest materialised tensor is `embed.weight` with 661,913,600 elements
/// (`head.weight` ties it exactly); the 98 GB Engram tables are paged through
/// [crate::loader::ReferenceWeights] and never materialised, so every other planned tensor is
/// smaller (the next largest non-Engram tensor is `mtp.0.main_proj.weight` with 78,643,200
/// elements). `1 << 30` elements is 4 GiB of `f32`, which admits all of them.
///
/// [REFERENCE_TENSOR_LIMIT] stays the small-shape guard: it deliberately refuses a real
/// checkpoint, and the synthetic configurations keep it. The one-token harness, the boundary dump
/// and the sequence fixture generator all take this constant from here, so no real-checkpoint
/// entry point can drift.
pub const CHECKPOINT_TENSOR_LIMIT: usize = 1 << 30;

/// Reads one tensor and decodes it to `f32`.
///
/// # Errors
///
/// [EngineError::Missing] when the source does not hold the name, [EngineError::TooLarge] when the
/// logical element count is above `limit`, and whatever [crate::quant] reports for the bytes.
pub fn decode_by_name<S: WeightSource + ?Sized>(
    source: &S,
    name: &str,
    limit: usize,
) -> Result<Vec<f32>, EngineError> {
    let spec = source.spec(name).map_err(|_| EngineError::Missing {
        name: name.to_owned(),
    })?;
    let elements = spec.elements();
    if elements > limit as u128 {
        return Err(EngineError::TooLarge {
            name: name.to_owned(),
            elements: usize::try_from(elements).unwrap_or(usize::MAX),
            limit,
        });
    }
    let bytes = source.read(name)?;
    let decoded = quant::decode(spec.dtype, &bytes, usize::try_from(elements).unwrap_or(0))?;
    if decoded.len() != usize::try_from(elements).unwrap_or(0) {
        return Err(EngineError::Shape {
            what: name.to_owned(),
            detail: format!("decoded {} values for {} elements", decoded.len(), elements),
        });
    }
    Ok(decoded)
}

/// Reads an optional tensor: `None` stays `None`, a present name is decoded.
///
/// # Errors
///
/// As [decode_by_name].
pub fn decode_optional<S: WeightSource + ?Sized>(
    source: &S,
    name: Option<&str>,
    limit: usize,
) -> Result<Option<Vec<f32>>, EngineError> {
    match name {
        None => Ok(None),
        Some(name) => decode_by_name(source, name, limit).map(Some),
    }
}

/// Decodes a raw view with no source involved, for tests and for callers holding bytes already.
///
/// # Errors
///
/// As [crate::quant::decode].
pub fn decode_view(view: &TensorView<'_>) -> Result<Vec<f32>, EngineError> {
    let elements: usize = view.shape.iter().product();
    Ok(quant::decode(view.dtype, view.data, elements)?)
}

/// Multiplies a decoded weight in place by its E8M0 scale grid.
///
/// The scale buffer is the *decoded* grid ([crate::quant::decode_f8e8m0]), not its bytes: the
/// reference multiplies in f32 (documented on that function), and doing it here keeps one place
/// responsible for the mapping from an element to its block.
///
/// Rows are independent, so large weights are scaled a row block at a time on the reference
/// thread pool. Each element still receives exactly one multiplication by its own grid entry, so
/// the result is identical for any thread count.
///
/// # Errors
///
/// [EngineError::Shape] when the buffer, the grid or the declared shape disagree.
pub fn apply_scales(
    weight: &mut [f32],
    rows: usize,
    cols: usize,
    scales: &[f32],
    grid: ScaleGrid,
) -> Result<(), EngineError> {
    if rows == 0 || cols == 0 || weight.len() != rows * cols {
        return Err(EngineError::Shape {
            what: "scaled weight".to_owned(),
            detail: format!(
                "a {rows}x{cols} weight needs {} values, got {}",
                rows * cols,
                weight.len()
            ),
        });
    }
    let expected = match grid {
        ScaleGrid::Block32 => rows.div_ceil(32) * cols.div_ceil(32),
        ScaleGrid::PerRow32 => rows * cols.div_ceil(32),
    };
    if scales.len() != expected {
        return Err(EngineError::Shape {
            what: "scale grid".to_owned(),
            detail: format!(
                "{rows}x{cols} on {grid:?} needs {expected} scales, got {}",
                scales.len()
            ),
        });
    }
    let min_rows = (65_536 / cols).max(1);
    par_aligned_mut(weight, cols, min_rows, |first_row, block| {
        for (local_row, row_values) in block.chunks_mut(cols).enumerate() {
            let row = first_row + local_row;
            for (col, value) in row_values.iter_mut().enumerate() {
                let scale = match grid {
                    ScaleGrid::Block32 => scales[(row / 32) * cols.div_ceil(32) + col / 32],
                    ScaleGrid::PerRow32 => scales[row * cols.div_ceil(32) + col / 32],
                };
                *value *= scale;
            }
        }
    });
    Ok(())
}

/// `out[b, row] = sum_col x[b, col] * w[row, col]` for every batch row, overwriting `out`.
///
/// Large multiplies are split over output rows on the reference thread pool; each output element
/// still runs the same ascending-`col` accumulation, so no output depends on the thread count.
///
/// # Errors
///
/// [EngineError::Shape] when the slice lengths do not match the declared geometry.
pub fn matmul(
    out: &mut [f32],
    x: &[f32],
    w: &[f32],
    rows: usize,
    cols: usize,
    batch: usize,
) -> Result<(), EngineError> {
    let geometry = format!("{batch}x{cols} by {rows}x{cols}");
    if x.len() < batch * cols || w.len() < rows * cols || out.len() < batch * rows {
        return Err(EngineError::Shape {
            what: "matmul".to_owned(),
            detail: format!(
                "{geometry} needs x>={}, w>={}, out>={}; got {}, {}, {}",
                batch * cols,
                rows * cols,
                batch * rows,
                x.len(),
                w.len(),
                out.len()
            ),
        });
    }
    let total = batch * rows;
    let multiplies = batch as u128 * rows as u128 * cols as u128;
    // A matvec with fewer than four million products is mostly a weight read; starting a scoped
    // thread for it costs more than the arithmetic saves. Above that gate the split is over output
    // rows, so every output keeps its serial reduction order at any row count and batch size. The
    // former extra "batch one and rows under 16_384" arm was the only reason the 5k-8k-row
    // attention and expert GEMVs ran serially; the row split is bit-identical, so it is retired.
    const PARALLEL_MATMUL_MIN_MULTIPLIES: u128 = 4_000_000;
    if multiplies < PARALLEL_MATMUL_MIN_MULTIPLIES {
        for b in 0..batch {
            let xb = &x[b * cols..b * cols + cols];
            let ob = &mut out[b * rows..b * rows + rows];
            for row in 0..rows {
                let wr = &w[row * cols..row * cols + cols];
                let mut acc = 0.0f32;
                for col in 0..cols {
                    acc += xb[col] * wr[col];
                }
                ob[row] = acc;
            }
        }
        return Ok(());
    }
    let min_chunk = (65_536 / cols.max(1)).max(1);
    par_chunks_mut(&mut out[..total], min_chunk, |offset, chunk| {
        for (index, slot) in chunk.iter_mut().enumerate() {
            let global = offset + index;
            let b = global / rows;
            let row = global % rows;
            let xb = &x[b * cols..b * cols + cols];
            let wr = &w[row * cols..row * cols + cols];
            let mut acc = 0.0f32;
            for col in 0..cols {
                acc += xb[col] * wr[col];
            }
            *slot = acc;
        }
    });
    Ok(())
}

/// `out = matmul(x, w) + bias`.
///
/// # Errors
///
/// As [matmul], plus [EngineError::Shape] when the bias is not `rows` long.
pub fn matmul_biased(
    out: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    rows: usize,
    cols: usize,
    batch: usize,
) -> Result<(), EngineError> {
    if bias.len() != rows {
        return Err(EngineError::Shape {
            what: "matmul bias".to_owned(),
            detail: format!("{rows} rows need {rows} biases, got {}", bias.len()),
        });
    }
    matmul(out, x, w, rows, cols, batch)?;
    for b in 0..batch {
        for row in 0..rows {
            out[b * rows + row] += bias[row];
        }
    }
    Ok(())
}

/// Accumulates a contiguous block of independent output rows.
///
/// Four rows are evaluated per column step: each lane owns one output row and one accumulator, so
/// the per-row sequence of f32 multiplies and adds is exactly the scalar order. No lane is ever
/// combined with another and no column sum is split or reassociated.
fn accumulate_row_group<F>(
    out: &mut [f32],
    xb: &[f32],
    first_row: usize,
    count: usize,
    _cols: usize,
    element: &F,
) where
    F: Fn(usize, usize) -> f32,
{
    let mut row = 0;
    while row + 4 <= count {
        let base = first_row + row;
        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;
        for (col, value) in xb.iter().copied().enumerate() {
            acc0 += value * element(base, col);
            acc1 += value * element(base + 1, col);
            acc2 += value * element(base + 2, col);
            acc3 += value * element(base + 3, col);
        }
        out[row] = acc0;
        out[row + 1] = acc1;
        out[row + 2] = acc2;
        out[row + 3] = acc3;
        row += 4;
    }
    while row < count {
        let mut acc = 0.0f32;
        for (col, value) in xb.iter().copied().enumerate() {
            acc += value * element(first_row + row, col);
        }
        out[row] = acc;
        row += 1;
    }
}

/// A fused coded-weight matmul: the exact decode and scale multiply happen inside the inner loop,
/// so no f32 weight buffer is materialised.
///
/// Every output element runs the same ascending-`col` accumulation as [matmul] over the same
/// decoded values: the closure must return `decode(code) * scale` for element (row, col), and the
/// scalar mul and add stay separate. Work is split over output rows exactly as [matmul] does.
fn coded_matmul<F>(
    out: &mut [f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    batch: usize,
    element: F,
) -> Result<(), EngineError>
where
    F: Fn(usize, usize) -> f32 + Sync,
{
    if x.len() < batch * cols || out.len() < batch * rows {
        return Err(EngineError::Shape {
            what: "coded matmul".to_owned(),
            detail: format!(
                "{batch}x{cols} by {rows}x{cols} needs x>={}, out>={}; got {}, {}",
                batch * cols,
                batch * rows,
                x.len(),
                out.len()
            ),
        });
    }
    let total = batch * rows;
    let multiplies = batch as u128 * rows as u128 * cols as u128;
    const PARALLEL_MATMUL_MIN_MULTIPLIES: u128 = 4_000_000;
    if multiplies < PARALLEL_MATMUL_MIN_MULTIPLIES {
        for b in 0..batch {
            let xb = &x[b * cols..b * cols + cols];
            let ob = &mut out[b * rows..b * rows + rows];
            accumulate_row_group(ob, xb, 0, rows, cols, &element);
        }
        return Ok(());
    }
    let min_chunk = (65_536 / cols.max(1)).max(1);
    if batch == 1 {
        // Split whole rows; inside a chunk four independent output rows share the broadcast
        // x[col] and each keeps its own ascending-col accumulator.
        par_chunks_mut(&mut out[..rows], min_chunk, |offset, chunk| {
            accumulate_row_group(chunk, x, offset, chunk.len(), cols, &element);
        });
        return Ok(());
    }
    par_chunks_mut(&mut out[..total], min_chunk, |offset, chunk| {
        for (index, slot) in chunk.iter_mut().enumerate() {
            let global = offset + index;
            let b = global / rows;
            let row = global % rows;
            let xb = &x[b * cols..b * cols + cols];
            let mut acc = 0.0f32;
            for (col, value) in xb.iter().copied().enumerate() {
                acc += value * element(row, col);
            }
            *slot = acc;
        }
    });
    Ok(())
}

/// The E8M0 scale byte for (row, col), using the same grid arithmetic as [apply_scales].
fn scale_offset(grid: ScaleGrid, row: usize, col: usize, cols: usize) -> usize {
    let scale_cols = cols.div_ceil(32);
    match grid {
        ScaleGrid::Block32 => (row / 32) * scale_cols + col / 32,
        ScaleGrid::PerRow32 => row * scale_cols + col / 32,
    }
}

/// The number of scale bytes a coded weight needs for this geometry.
fn scale_bytes(grid: ScaleGrid, rows: usize, cols: usize) -> usize {
    let scale_cols = cols.div_ceil(32);
    match grid {
        ScaleGrid::Block32 => rows.div_ceil(32) * scale_cols,
        ScaleGrid::PerRow32 => rows * scale_cols,
    }
}

/// Checks shared coded-matmul geometry and returns the number of codes needed.
fn coded_geometry(
    out: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    batch: usize,
    scales: &[u8],
    grid: ScaleGrid,
) -> Result<usize, EngineError> {
    if rows == 0 || cols == 0 {
        return Err(EngineError::Shape {
            what: "coded matmul".to_owned(),
            detail: format!("geometry must be non-empty, got {rows}x{cols}"),
        });
    }
    if x.len() < batch * cols || out.len() < batch * rows {
        return Err(EngineError::Shape {
            what: "coded matmul".to_owned(),
            detail: format!(
                "{batch}x{cols} by {rows}x{cols} needs x>={}, out>={}; got {}, {}",
                batch * cols,
                batch * rows,
                x.len(),
                out.len()
            ),
        });
    }
    let elements = rows.checked_mul(cols).ok_or_else(|| EngineError::Shape {
        what: "coded matmul".to_owned(),
        detail: "rows times cols overflows".to_owned(),
    })?;
    let needed_scales = scale_bytes(grid, rows, cols);
    if scales.len() < needed_scales {
        return Err(EngineError::Shape {
            what: "coded matmul".to_owned(),
            detail: format!(
                "{rows}x{cols} needs {needed_scales} scale bytes, got {}",
                scales.len()
            ),
        });
    }
    Ok(elements)
}

/// Multiplies an F8-E4M3 coded weight in its E8M0 grid without materialising f32 weights.
///
/// This is `decode + apply_scales + matmul` fused: each element is the exact
/// [quant::f8e4m3_to_f32] code times the exact [quant::f8e8m0_to_f32] scale in one f32 multiply,
/// then one f32 multiply-add into the ascending-`col` accumulator. The order and the separate mul
/// and add are those of [matmul], so the output is bit-identical to the materialised path.
///
/// # Errors
///
/// [EngineError::Shape] when the slices do not match the declared geometry or the grid needs more
/// scale bytes than `scales` holds.
#[allow(clippy::too_many_arguments)] // the coded weight, its grid and the matmul geometry are distinct inputs.
pub fn matmul_f8_scaled(
    out: &mut [f32],
    x: &[f32],
    codes: &[u8],
    scales: &[u8],
    grid: ScaleGrid,
    rows: usize,
    cols: usize,
    batch: usize,
) -> Result<(), EngineError> {
    coded_geometry(out, x, rows, cols, batch, scales, grid)?;
    if codes.len() < rows * cols {
        return Err(EngineError::Shape {
            what: "coded matmul".to_owned(),
            detail: format!(
                "{rows}x{cols} needs {} codes, got {}",
                rows * cols,
                codes.len()
            ),
        });
    }
    let code_table: [f32; 256] = std::array::from_fn(|index| quant::f8e4m3_to_f32(index as u8));
    let scale_table: [f32; 256] = std::array::from_fn(|index| quant::f8e8m0_to_f32(index as u8));
    coded_matmul(out, x, rows, cols, batch, |row, col| {
        code_table[codes[row * cols + col] as usize]
            * scale_table[scales[scale_offset(grid, row, col, cols)] as usize]
    })
}

/// Multiplies a block-diagonal stack of F8-E4M3 coded weights in one parallel pass.
///
/// Output row g*rows_per_group+r is the dot of input columns
/// g*cols_per_group .. + cols_per_group with the matching row of block g. Per-row arithmetic is
/// exactly [matmul_f8_scaled]; only output-row work is shared across threads.
///
/// # Errors
///
/// [EngineError::Shape] when buffers or grid do not match the block geometry.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn matmul_f8_grouped(
    out: &mut [f32],
    x: &[f32],
    codes: &[u8],
    scales: &[u8],
    grid: ScaleGrid,
    groups: usize,
    rows_per_group: usize,
    cols_per_group: usize,
) -> Result<(), EngineError> {
    if groups == 0 || rows_per_group == 0 || cols_per_group == 0 {
        return Err(EngineError::Shape {
            what: "grouped coded matmul".to_owned(),
            detail: "a zero block geometry has no product".to_owned(),
        });
    }
    let rows = groups
        .checked_mul(rows_per_group)
        .ok_or_else(|| EngineError::Shape {
            what: "grouped coded matmul".to_owned(),
            detail: "group times rows overflows".to_owned(),
        })?;
    let width = groups
        .checked_mul(cols_per_group)
        .ok_or_else(|| EngineError::Shape {
            what: "grouped coded matmul".to_owned(),
            detail: "group times columns overflows".to_owned(),
        })?;
    let elements = rows
        .checked_mul(cols_per_group)
        .ok_or_else(|| EngineError::Shape {
            what: "grouped coded matmul".to_owned(),
            detail: "rows times columns overflows".to_owned(),
        })?;
    if out.len() != rows || x.len() != width || codes.len() < elements {
        return Err(EngineError::Shape {
            what: "grouped coded matmul".to_owned(),
            detail: format!(
                "{groups}x{rows_per_group}x{cols_per_group} needs {rows} outputs, {width} inputs and {elements} codes; got {}, {} and {}",
                out.len(),
                x.len(),
                codes.len()
            ),
        });
    }
    let scale_cols = cols_per_group.div_ceil(32);
    let needed_scales = match grid {
        ScaleGrid::Block32 => rows.div_ceil(32) * scale_cols,
        ScaleGrid::PerRow32 => rows * scale_cols,
    };
    if scales.len() < needed_scales {
        return Err(EngineError::Shape {
            what: "grouped coded matmul".to_owned(),
            detail: format!(
                "{rows}x{cols_per_group} needs {needed_scales} scale bytes, got {}",
                scales.len()
            ),
        });
    }
    let code_table: [f32; 256] = std::array::from_fn(|index| quant::f8e4m3_to_f32(index as u8));
    let scale_table: [f32; 256] = std::array::from_fn(|index| quant::f8e8m0_to_f32(index as u8));
    let element = |global_row: usize, col: usize| {
        let code = codes[global_row * cols_per_group + col];
        let scale = scales[scale_offset(grid, global_row, col, cols_per_group)];
        code_table[code as usize] * scale_table[scale as usize]
    };
    let min_chunk = (65_536 / cols_per_group).max(1);
    let multiplies = rows as u128 * cols_per_group as u128;
    const PARALLEL_MATMUL_MIN_MULTIPLIES: u128 = 1_000_000;
    if multiplies < PARALLEL_MATMUL_MIN_MULTIPLIES {
        for global_row in 0..rows {
            let xbase = (global_row / rows_per_group) * cols_per_group;
            let mut acc = 0.0f32;
            for (col, value) in x[xbase..xbase + cols_per_group].iter().copied().enumerate() {
                acc += value * element(global_row, col);
            }
            out[global_row] = acc;
        }
        return Ok(());
    }
    par_chunks_mut(out, min_chunk, |offset, chunk| {
        let mut index = 0;
        while index + 4 <= chunk.len() {
            let base = offset + index;
            if base / rows_per_group != (base + 3) / rows_per_group {
                break;
            }
            let xbase = (base / rows_per_group) * cols_per_group;
            let mut acc0 = 0.0f32;
            let mut acc1 = 0.0f32;
            let mut acc2 = 0.0f32;
            let mut acc3 = 0.0f32;
            for (col, value) in x[xbase..xbase + cols_per_group].iter().copied().enumerate() {
                acc0 += value * element(base, col);
                acc1 += value * element(base + 1, col);
                acc2 += value * element(base + 2, col);
                acc3 += value * element(base + 3, col);
            }
            let slot = &mut chunk[index..index + 4];
            slot[0] = acc0;
            slot[1] = acc1;
            slot[2] = acc2;
            slot[3] = acc3;
            index += 4;
        }
        while index < chunk.len() {
            let global_row = offset + index;
            let xbase = (global_row / rows_per_group) * cols_per_group;
            let mut acc = 0.0f32;
            for (col, value) in x[xbase..xbase + cols_per_group].iter().copied().enumerate() {
                acc += value * element(global_row, col);
            }
            chunk[index] = acc;
            index += 1;
        }
    });
    Ok(())
}

/// The released Engram wkv projection as a parallel fp8 W8A8 matmul.
///
/// One output row is exactly the scalar kernel: for each ascending 32-column block, the 32 code
/// products are reduced in the device kernel's fixed shuffle-tree order (offsets 16, 8, 4, 2, 1),
/// multiplied by that block's activation and weight scales, and accumulated over blocks; the row is
/// then rounded to bf16. Output rows are independent, so parallelising across them changes no bits.
///
/// # Errors
///
/// [EngineError::Shape] when a slice does not match out_width x inner and its 32-column grid.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn matmul_f8_w8a8(
    out: &mut [f32],
    act_codes: &[u8],
    act_scales: &[u8],
    weight: &[u8],
    weight_scales: &[u8],
    out_width: usize,
    inner: usize,
) -> Result<(), EngineError> {
    if out_width == 0 || inner == 0 {
        return Err(EngineError::Shape {
            what: "engram wkv".to_owned(),
            detail: "a zero geometry has no product".to_owned(),
        });
    }
    let blocks = inner.div_ceil(32);
    if out.len() != out_width
        || act_codes.len() != inner
        || act_scales.len() != blocks
        || weight.len() != out_width * inner
        || weight_scales.len() != out_width.div_ceil(32) * blocks
    {
        return Err(EngineError::Shape {
            what: "engram wkv".to_owned(),
            detail: format!(
                "{out_width}x{inner} needs {out_width} outputs, {inner} activation codes, {blocks} activation scales, {} weight codes and {} weight scales; got {}, {}, {}, {} and {}",
                out_width * inner,
                out_width.div_ceil(32) * blocks,
                out.len(),
                act_codes.len(),
                act_scales.len(),
                weight.len(),
                weight_scales.len()
            ),
        });
    }
    let code_table: [f32; 256] = std::array::from_fn(|index| quant::f8e4m3_to_f32(index as u8));
    let scale_table: [f32; 256] = std::array::from_fn(|index| quant::f8e8m0_to_f32(index as u8));
    let row_value = |row: usize| {
        let weight_row = &weight[row * inner..(row + 1) * inner];
        let scale_row = &weight_scales[(row / 32) * blocks..(row / 32 + 1) * blocks];
        let mut accumulator = 0.0f32;
        for kb in 0..blocks {
            let base = kb * 32;
            let width = (inner - base).min(32);
            let mut lane = [0.0f32; 32];
            for column in 0..width {
                lane[column] = code_table[act_codes[base + column] as usize]
                    * code_table[weight_row[base + column] as usize];
            }
            let mut offset = 16;
            while offset > 0 {
                for partial in 0..offset {
                    lane[partial] += lane[partial + offset];
                }
                offset /= 2;
            }
            accumulator += (lane[0] * scale_table[act_scales[kb] as usize])
                * scale_table[scale_row[kb] as usize];
        }
        quant::bf16_round(accumulator)
    };
    let multiplies = out_width as u128 * inner as u128;
    const PARALLEL_MIN_MULTIPLIES: u128 = 1_000_000;
    if multiplies < PARALLEL_MIN_MULTIPLIES {
        for (row, slot) in out.iter_mut().enumerate() {
            *slot = row_value(row);
        }
        return Ok(());
    }
    // 64 rows per chunk keeps the per-chunk work above the scheduler cost while giving every
    // reference thread a contiguous run of output rows.
    par_chunks_mut(out, 64, |offset, chunk| {
        for (index, slot) in chunk.iter_mut().enumerate() {
            *slot = row_value(offset + index);
        }
    });
    Ok(())
}

/// Multiplies a packed F4-E2M1 coded weight in its E8M0 grid without materialising f32 weights.
///
/// The low nibble of each byte is the even element and the high nibble the odd one, exactly as
/// [quant::decode_f4e2m1] reads them. Each element is the exact [quant::f4e2m1_to_f32] nibble times
/// the exact [quant::f8e8m0_to_f32] scale in one f32 multiply, then the same ascending-`col`
/// multiply-add as [matmul]; the result is bit-identical to the materialised path.
///
/// # Errors
///
/// [EngineError::Shape] when the slices do not match the declared geometry or the grid needs more
/// scale bytes than `scales` holds.
#[allow(clippy::too_many_arguments)] // the coded weight, its grid and the matmul geometry are distinct inputs.
pub fn matmul_f4_scaled(
    out: &mut [f32],
    x: &[f32],
    codes: &[u8],
    scales: &[u8],
    grid: ScaleGrid,
    rows: usize,
    cols: usize,
    batch: usize,
) -> Result<(), EngineError> {
    coded_geometry(out, x, rows, cols, batch, scales, grid)?;
    let packed = (rows * cols).div_ceil(2);
    if codes.len() < packed {
        return Err(EngineError::Shape {
            what: "coded matmul".to_owned(),
            detail: format!(
                "{rows}x{cols} needs {packed} packed code bytes, got {}",
                codes.len()
            ),
        });
    }
    let code_table: [f32; 16] = std::array::from_fn(|index| quant::f4e2m1_to_f32(index as u8));
    let scale_table: [f32; 256] = std::array::from_fn(|index| quant::f8e8m0_to_f32(index as u8));
    coded_matmul(out, x, rows, cols, batch, |row, col| {
        let index = row * cols + col;
        let byte = codes[index / 2];
        let nibble = if index.is_multiple_of(2) {
            byte
        } else {
            byte >> 4
        };
        code_table[(nibble & 0x0f) as usize]
            * scale_table[scales[scale_offset(grid, row, col, cols)] as usize]
    })
}

/// Multiplies several selected F4-E2M1 routed experts in one parallel region.
///
/// xs[e] is the token vector for expert e, and output row e*rows_per_expert+r is that expert's row r.
/// Per row the arithmetic is exactly [matmul_f4_scaled] with the same ascending column order and four
/// independent row accumulators; only output rows are shared across threads, so one call replaces one
/// call per expert and keeps every thread busy on the combined row set.
///
/// # Errors
///
/// [EngineError::Shape] when the expert count, output length, input widths or coded grids disagree.
#[allow(clippy::too_many_arguments)]
pub fn matmul_f4_experts(
    out: &mut [f32],
    xs: &[&[f32]],
    codes: &[&[u8]],
    scales: &[&[u8]],
    grid: ScaleGrid,
    rows_per_expert: usize,
    cols_per_expert: usize,
) -> Result<(), EngineError> {
    let experts = xs.len();
    if experts == 0 || rows_per_expert == 0 || cols_per_expert == 0 {
        return Err(EngineError::Shape {
            what: "routed expert matmul".to_owned(),
            detail: "a zero expert geometry has no product".to_owned(),
        });
    }
    if codes.len() != experts || scales.len() != experts {
        return Err(EngineError::Shape {
            what: "routed expert matmul".to_owned(),
            detail: format!(
                "{experts} inputs need {experts} code slices and {experts} grids, got {} and {}",
                codes.len(),
                scales.len()
            ),
        });
    }
    let rows = experts
        .checked_mul(rows_per_expert)
        .ok_or_else(|| EngineError::Shape {
            what: "routed expert matmul".to_owned(),
            detail: "expert count times rows overflows".to_owned(),
        })?;
    if out.len() != rows {
        return Err(EngineError::Shape {
            what: "routed expert matmul".to_owned(),
            detail: format!(
                "{experts}x{rows_per_expert} needs {rows} outputs, got {}",
                out.len()
            ),
        });
    }
    let packed = rows_per_expert
        .checked_mul(cols_per_expert)
        .and_then(|elements| elements.checked_add(1))
        .map(|elements| elements / 2)
        .ok_or_else(|| EngineError::Shape {
            what: "routed expert matmul".to_owned(),
            detail: "rows times columns overflows".to_owned(),
        })?;
    let needed_scales = scale_bytes(grid, rows_per_expert, cols_per_expert);
    for expert in 0..experts {
        if xs[expert].len() != cols_per_expert {
            return Err(EngineError::Shape {
                what: "routed expert matmul".to_owned(),
                detail: format!(
                    "expert {expert} needs {} input values, got {}",
                    cols_per_expert,
                    xs[expert].len()
                ),
            });
        }
        if codes[expert].len() < packed || scales[expert].len() < needed_scales {
            return Err(EngineError::Shape {
                what: "routed expert matmul".to_owned(),
                detail: format!(
                    "expert {expert} needs {packed} code bytes and {needed_scales} scale bytes, got {} and {}",
                    codes[expert].len(),
                    scales[expert].len()
                ),
            });
        }
    }
    let code_table: [f32; 16] = std::array::from_fn(|index| quant::f4e2m1_to_f32(index as u8));
    let scale_table: [f32; 256] = std::array::from_fn(|index| quant::f8e8m0_to_f32(index as u8));
    let element = |expert: usize, row: usize, col: usize| {
        let index = row * cols_per_expert + col;
        let byte = codes[expert][index / 2];
        let nibble = if index.is_multiple_of(2) {
            byte
        } else {
            byte >> 4
        };
        code_table[(nibble & 0x0f) as usize]
            * scale_table[scales[expert][scale_offset(grid, row, col, cols_per_expert)] as usize]
    };
    let multiplies = rows as u128 * cols_per_expert as u128;
    const PARALLEL_MIN_MULTIPLIES: u128 = 1_000_000;
    if multiplies < PARALLEL_MIN_MULTIPLIES {
        for (global_row, slot) in out.iter_mut().enumerate() {
            let expert = global_row / rows_per_expert;
            let row = global_row % rows_per_expert;
            let mut acc = 0.0f32;
            for (col, value) in xs[expert].iter().copied().enumerate() {
                acc += value * element(expert, row, col);
            }
            *slot = acc;
        }
        return Ok(());
    }
    let min_chunk = (65_536 / cols_per_expert).max(1);
    par_chunks_mut(out, min_chunk, |offset, chunk| {
        let mut index = 0;
        while index + 4 <= chunk.len() {
            let global_row = offset + index;
            let expert = global_row / rows_per_expert;
            if expert != (global_row + 3) / rows_per_expert {
                break;
            }
            let row = global_row % rows_per_expert;
            let x = xs[expert];
            let codes = &codes[expert];
            let scales = &scales[expert];
            let mut acc0 = 0.0f32;
            let mut acc1 = 0.0f32;
            let mut acc2 = 0.0f32;
            let mut acc3 = 0.0f32;
            for base in (0..cols_per_expert).step_by(32) {
                let width = (cols_per_expert - base).min(32);
                let scale0 = scales[scale_offset(grid, row, base, cols_per_expert)] as usize;
                let scale1 = scales[scale_offset(grid, row + 1, base, cols_per_expert)] as usize;
                let scale2 = scales[scale_offset(grid, row + 2, base, cols_per_expert)] as usize;
                let scale3 = scales[scale_offset(grid, row + 3, base, cols_per_expert)] as usize;
                let s0 = scale_table[scale0];
                let s1 = scale_table[scale1];
                let s2 = scale_table[scale2];
                let s3 = scale_table[scale3];
                for j in 0..width {
                    let column = base + j;
                    let value = x[column];
                    let at0 = row * cols_per_expert + column;
                    let byte0 = codes[at0 / 2];
                    let byte1 = codes[(at0 + cols_per_expert) / 2];
                    let byte2 = codes[(at0 + 2 * cols_per_expert) / 2];
                    let byte3 = codes[(at0 + 3 * cols_per_expert) / 2];
                    let (nib0, nib1, nib2, nib3) = if at0.is_multiple_of(2) {
                        (byte0 & 0x0f, byte1 & 0x0f, byte2 & 0x0f, byte3 & 0x0f)
                    } else {
                        (byte0 >> 4, byte1 >> 4, byte2 >> 4, byte3 >> 4)
                    };
                    acc0 += value * (code_table[nib0 as usize] * s0);
                    acc1 += value * (code_table[nib1 as usize] * s1);
                    acc2 += value * (code_table[nib2 as usize] * s2);
                    acc3 += value * (code_table[nib3 as usize] * s3);
                }
            }
            let slot = &mut chunk[index..index + 4];
            slot[0] = acc0;
            slot[1] = acc1;
            slot[2] = acc2;
            slot[3] = acc3;
            index += 4;
        }
        while index < chunk.len() {
            let global_row = offset + index;
            let expert = global_row / rows_per_expert;
            let row = global_row % rows_per_expert;
            let mut acc = 0.0f32;
            for (col, value) in xs[expert].iter().copied().enumerate() {
                acc += value * element(expert, row, col);
            }
            chunk[index] = acc;
            index += 1;
        }
    });
    Ok(())
}

/// Multiplies a plain BF16 weight without materialising an f32 weight buffer.
///
/// The widening is exact (bf16 is the high half of f32), so this is bit-identical to
/// [quant::decode_bf16] followed by [matmul].
///
/// # Errors
///
/// [EngineError::Shape] when `w` is shorter than `rows * cols * 2` bytes or the vector slices do
/// not match the geometry.
pub fn matmul_bf16(
    out: &mut [f32],
    x: &[f32],
    w: &[u8],
    rows: usize,
    cols: usize,
    batch: usize,
) -> Result<(), EngineError> {
    let elements = rows.checked_mul(cols).ok_or_else(|| EngineError::Shape {
        what: "bf16 matmul".to_owned(),
        detail: "rows times cols overflows".to_owned(),
    })?;
    let bytes = elements.checked_mul(2).ok_or_else(|| EngineError::Shape {
        what: "bf16 matmul".to_owned(),
        detail: "element byte count overflows".to_owned(),
    })?;
    if w.len() < bytes {
        return Err(EngineError::Shape {
            what: "bf16 matmul".to_owned(),
            detail: format!("{rows}x{cols} needs {bytes} bytes, got {}", w.len()),
        });
    }
    coded_matmul(out, x, rows, cols, batch, |row, col| {
        let index = row * cols + col;
        quant::bf16_to_f32(u16::from_le_bytes([w[index * 2], w[index * 2 + 1]]))
    })
}

/// Multiplies a BF16 weight in an E8M0 grid without materialising an f32 weight buffer.
///
/// As [matmul_bf16], with each element multiplied by its exact E8M0 scale before the ascending-col
/// accumulation; bit-identical to decode + [apply_scales] + [matmul].
///
/// # Errors
///
/// [EngineError::Shape] when the slices do not match the declared geometry or the grid needs more
/// scale bytes than `scales` holds.
#[allow(clippy::too_many_arguments)] // the coded weight, its grid and the matmul geometry are distinct inputs.
pub fn matmul_bf16_scaled(
    out: &mut [f32],
    x: &[f32],
    w: &[u8],
    scales: &[u8],
    grid: ScaleGrid,
    rows: usize,
    cols: usize,
    batch: usize,
) -> Result<(), EngineError> {
    let elements = rows.checked_mul(cols).ok_or_else(|| EngineError::Shape {
        what: "bf16 matmul".to_owned(),
        detail: "rows times cols overflows".to_owned(),
    })?;
    let bytes = elements.checked_mul(2).ok_or_else(|| EngineError::Shape {
        what: "bf16 matmul".to_owned(),
        detail: "element byte count overflows".to_owned(),
    })?;
    if w.len() < bytes {
        return Err(EngineError::Shape {
            what: "bf16 matmul".to_owned(),
            detail: format!("{rows}x{cols} needs {bytes} bytes, got {}", w.len()),
        });
    }
    let needed_scales = scale_bytes(grid, rows, cols);
    if scales.len() < needed_scales {
        return Err(EngineError::Shape {
            what: "bf16 matmul".to_owned(),
            detail: format!(
                "{rows}x{cols} needs {needed_scales} scale bytes, got {}",
                scales.len()
            ),
        });
    }
    coded_matmul(out, x, rows, cols, batch, |row, col| {
        let index = row * cols + col;
        quant::bf16_to_f32(u16::from_le_bytes([w[index * 2], w[index * 2 + 1]]))
            * quant::f8e8m0_to_f32(scales[scale_offset(grid, row, col, cols)])
    })
}

/// Normalises each `cols`-long row of `x` in place: `x * rsqrt(mean(x^2) + eps) * weight`.
///
/// Whole rows are split over the reference thread pool; a row's mean and scaling are computed in
/// the same order as the serial pass, so the result is identical for any thread count.
///
/// # Errors
///
/// [EngineError::Shape] when `x` is not a whole number of `cols`-long rows or the weight is the
/// wrong length.
pub fn rms_norm(x: &mut [f32], weight: &[f32], cols: usize, eps: f32) -> Result<(), EngineError> {
    if cols == 0 || !x.len().is_multiple_of(cols) {
        return Err(EngineError::Shape {
            what: "rms norm".to_owned(),
            detail: format!("{} values are not whole rows of {cols}", x.len()),
        });
    }
    if weight.len() != cols {
        return Err(EngineError::Shape {
            what: "rms norm weight".to_owned(),
            detail: format!(
                "{cols} columns need a {cols}-long weight, got {}",
                weight.len()
            ),
        });
    }
    let min_rows = (65_536 / cols).max(1);
    par_aligned_mut(x, cols, min_rows, |_first_row, block| {
        for row in block.chunks_mut(cols) {
            let mean = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
            let inv = 1.0 / (mean + eps).sqrt();
            for (value, w) in row.iter_mut().zip(weight.iter()) {
                *value = *value * inv * w;
            }
        }
    });
    Ok(())
}

/// The numerically stable softmax of `x`, in place.
///
/// An all-`-inf` row is left at zero rather than becoming NaN: the sparse attention path relies on
/// masked rows contributing nothing.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        x.fill(0.0);
        return;
    }
    let mut total = 0.0f32;
    for value in x.iter_mut() {
        let exp = (*value - max).exp();
        *value = exp;
        total += exp;
    }
    if total > 0.0 {
        for value in x.iter_mut() {
            *value /= total;
        }
    }
}

/// `1 / (1 + exp(-x))`, saturating instead of overflowing.
#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp = x.exp();
        exp / (1.0 + exp)
    }
}

/// The SiLU (swish) activation: `x * sigmoid(x)`.
#[must_use]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Rotates the pairs of `x` using `cos`/`sin` tables: `(a, b) -> (a cos - b sin, a sin + b cos)`.
///
/// `x` is the rope slice of a head (the reference rotates the last `rope_head_dim` of each head,
/// model.py:772) and the tables are per position, `rope_head_dim / 2` pairs long.
///
/// # Errors
///
/// [EngineError::Shape] when `x` or the tables do not describe whole pairs.
pub fn rope_rotate(x: &mut [f32], cos: &[f32], sin: &[f32]) -> Result<(), EngineError> {
    rope_rotate_with(x, cos, sin, 1.0)
}

/// The inverse of [rope_rotate]: the rotation by `-theta`, which is what the attention output
/// undoes before the cache keeps one rotated form (model.py:781).
///
/// # Errors
///
/// As [rope_rotate].
pub fn rope_rotate_inverse(x: &mut [f32], cos: &[f32], sin: &[f32]) -> Result<(), EngineError> {
    rope_rotate_with(x, cos, sin, -1.0)
}

fn rope_rotate_with(x: &mut [f32], cos: &[f32], sin: &[f32], sign: f32) -> Result<(), EngineError> {
    if !x.len().is_multiple_of(2) || cos.len() != x.len() / 2 || sin.len() != x.len() / 2 {
        return Err(EngineError::Shape {
            what: "rope".to_owned(),
            detail: format!(
                "{} values need {} cos and {} sin, got {} and {}",
                x.len(),
                x.len() / 2,
                x.len() / 2,
                cos.len(),
                sin.len()
            ),
        });
    }
    for pair in 0..x.len() / 2 {
        let a = x[2 * pair];
        let b = x[2 * pair + 1];
        x[2 * pair] = a * cos[pair] - sign * b * sin[pair];
        x[2 * pair + 1] = sign * a * sin[pair] + b * cos[pair];
    }
    Ok(())
}

/// The index of the largest value, or `None` for an empty slice. Ties take the lower index.
#[must_use]
pub fn argmax(x: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (index, value) in x.iter().copied().enumerate() {
        if best.is_none_or(|(_, held)| value > held) {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index)
}

/// The arithmetic mean of a slice, or 0 for an empty one.
#[must_use]
pub fn mean(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f32>() / x.len() as f32
}

/// The element format a plan uses for a dense weight, for callers that assert rather than decode.
#[must_use]
pub fn is_dense_weight(dtype: DType) -> bool {
    matches!(dtype, DType::F8E4M3 | DType::BF16 | DType::F32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patterned_bytes(len: usize, seed: usize) -> Vec<u8> {
        (0..len)
            .map(|index| ((index * 37 + seed * 11) % 251) as u8)
            .collect()
    }

    fn fnv(values: &[f32]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for word in bits(values) {
            for byte in word.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    fn assert_same_bits(old: &[f32], new: &[f32]) {
        assert_eq!(bits(old), bits(new), "fused path changed bits");
        assert_eq!(fnv(old), fnv(new), "fused path changed the FNV dump");
    }

    #[test]
    fn fused_f8_scaled_is_bit_identical_to_decode_scale_and_matmul() {
        let (rows, cols, batch): (usize, usize, usize) = (37, 41, 2);
        let elements = rows * cols;
        let codes = patterned_bytes(elements, 3);
        let scales = patterned_bytes(rows.div_ceil(32) * cols.div_ceil(32), 5);
        let mut x = vec![0.0f32; batch * cols];
        for (index, value) in x.iter_mut().enumerate() {
            *value = ((index * 29 % 97) as f32 - 48.0) / 31.0;
        }
        let mut decoded = quant::decode(DType::F8E4M3, &codes, elements).expect("decode");
        apply_scales(
            &mut decoded,
            rows,
            cols,
            &quant::decode_f8e8m0(&scales),
            ScaleGrid::Block32,
        )
        .expect("scales");
        let mut old = vec![0.0f32; batch * rows];
        matmul(&mut old, &x, &decoded, rows, cols, batch).expect("matmul");
        let mut new = vec![0.0f32; batch * rows];
        matmul_f8_scaled(
            &mut new,
            &x,
            &codes,
            &scales,
            ScaleGrid::Block32,
            rows,
            cols,
            batch,
        )
        .expect("fused");
        assert_same_bits(&old, &new);
    }

    #[test]
    fn fused_f8_scaled_per_row_grid_is_bit_identical_too() {
        let (rows, cols): (usize, usize) = (35, 40);
        let elements = rows * cols;
        let codes = patterned_bytes(elements, 7);
        let scales = patterned_bytes(rows * cols.div_ceil(32), 9);
        let x: Vec<f32> = (0..cols).map(|index| (index as f32) / 17.0 - 1.0).collect();
        let mut decoded = quant::decode(DType::F8E4M3, &codes, elements).expect("decode");
        apply_scales(
            &mut decoded,
            rows,
            cols,
            &quant::decode_f8e8m0(&scales),
            ScaleGrid::PerRow32,
        )
        .expect("scales");
        let mut old = vec![0.0f32; rows];
        matmul(&mut old, &x, &decoded, rows, cols, 1).expect("matmul");
        let mut new = vec![0.0f32; rows];
        matmul_f8_scaled(
            &mut new,
            &x,
            &codes,
            &scales,
            ScaleGrid::PerRow32,
            rows,
            cols,
            1,
        )
        .expect("fused");
        assert_same_bits(&old, &new);
    }

    #[test]
    fn fused_f4_scaled_is_bit_identical_even_when_a_row_ends_mid_byte() {
        let (rows, cols): (usize, usize) = (37, 41);
        let elements = rows * cols;
        let codes = patterned_bytes(elements.div_ceil(2), 13);
        let scales = patterned_bytes(rows * cols.div_ceil(32), 17);
        let x: Vec<f32> = (0..cols).map(|index| (index as f32) / 23.0 - 0.5).collect();
        let mut decoded = quant::decode(DType::F4E2M1, &codes, elements).expect("decode packed f4");
        apply_scales(
            &mut decoded,
            rows,
            cols,
            &quant::decode_f8e8m0(&scales),
            ScaleGrid::PerRow32,
        )
        .expect("scales");
        let mut old = vec![0.0f32; rows];
        matmul(&mut old, &x, &decoded, rows, cols, 1).expect("matmul");
        let mut new = vec![0.0f32; rows];
        matmul_f4_scaled(
            &mut new,
            &x,
            &codes,
            &scales,
            ScaleGrid::PerRow32,
            rows,
            cols,
            1,
        )
        .expect("fused");
        assert_same_bits(&old, &new);
    }

    #[test]
    fn fused_bf16_is_bit_identical_to_decode_and_matmul() {
        let (rows, cols, batch): (usize, usize, usize) = (33, 47, 2);
        let elements = rows * cols;
        let words: Vec<u16> = (0..elements)
            .map(|index| 0x3f80 | ((index * 251 % 511) as u16))
            .collect();
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let x: Vec<f32> = (0..batch * cols)
            .map(|index| (index as f32) / 13.0 - 3.0)
            .collect();
        let decoded = quant::decode(DType::BF16, &bytes, elements).expect("decode bf16");
        let mut old = vec![0.0f32; batch * rows];
        matmul(&mut old, &x, &decoded, rows, cols, batch).expect("matmul");
        let mut new = vec![0.0f32; batch * rows];
        matmul_bf16(&mut new, &x, &bytes, rows, cols, batch).expect("fused");
        assert_same_bits(&old, &new);
    }

    #[test]
    fn fused_bf16_scaled_is_bit_identical_to_decode_scale_and_matmul() {
        let (rows, cols): (usize, usize) = (34, 40);
        let elements = rows * cols;
        let words: Vec<u16> = (0..elements)
            .map(|index| 0x3f00 | ((index * 197 % 255) as u16))
            .collect();
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let scales = patterned_bytes(rows.div_ceil(32) * cols.div_ceil(32), 19);
        let x: Vec<f32> = (0..cols).map(|index| (index as f32) / 11.0 - 1.0).collect();
        let mut decoded = quant::decode(DType::BF16, &bytes, elements).expect("decode bf16");
        apply_scales(
            &mut decoded,
            rows,
            cols,
            &quant::decode_f8e8m0(&scales),
            ScaleGrid::Block32,
        )
        .expect("scales");
        let mut old = vec![0.0f32; rows];
        matmul(&mut old, &x, &decoded, rows, cols, 1).expect("matmul");
        let mut new = vec![0.0f32; rows];
        matmul_bf16_scaled(
            &mut new,
            &x,
            &bytes,
            &scales,
            ScaleGrid::Block32,
            rows,
            cols,
            1,
        )
        .expect("fused");
        assert_same_bits(&old, &new);
    }

    #[test]
    fn fused_expert_batch_matches_separate_f4_calls_bitwise() {
        let (experts, rows_per_expert, cols_per_expert) = (3usize, 32usize, 64usize);
        let packed = (rows_per_expert * cols_per_expert).div_ceil(2);
        let scales_len = rows_per_expert * cols_per_expert.div_ceil(32);
        let x: Vec<f32> = (0..cols_per_expert)
            .map(|index| (index as f32) / 13.0 - 1.0)
            .collect();
        let xs: Vec<&[f32]> = (0..experts).map(|_| x.as_slice()).collect();
        let codes: Vec<Vec<u8>> = (0..experts)
            .map(|e| patterned_bytes(packed, 41 + e))
            .collect();
        let scales: Vec<Vec<u8>> = (0..experts)
            .map(|e| patterned_bytes(scales_len, 51 + e))
            .collect();
        let code_slices: Vec<&[u8]> = codes.iter().map(Vec::as_slice).collect();
        let scale_slices: Vec<&[u8]> = scales.iter().map(Vec::as_slice).collect();
        let mut want = vec![0.0f32; experts * rows_per_expert];
        for expert in 0..experts {
            matmul_f4_scaled(
                &mut want[expert * rows_per_expert..(expert + 1) * rows_per_expert],
                &x,
                &codes[expert],
                &scales[expert],
                ScaleGrid::PerRow32,
                rows_per_expert,
                cols_per_expert,
                1,
            )
            .expect("separate expert");
        }
        let mut got = vec![0.0f32; experts * rows_per_expert];
        matmul_f4_experts(
            &mut got,
            &xs,
            &code_slices,
            &scale_slices,
            ScaleGrid::PerRow32,
            rows_per_expert,
            cols_per_expert,
        )
        .expect("batched experts");
        assert_same_bits(&want, &got);
    }

    #[test]
    fn fused_w8a8_engram_wkv_matches_the_scalar_tree_bitwise() {
        let (out_width, inner) = (64usize, 96usize);
        let blocks = inner.div_ceil(32);
        let act_codes = patterned_bytes(inner, 31);
        let act_scales = patterned_bytes(blocks, 33);
        let weight = patterned_bytes(out_width * inner, 35);
        let weight_scales = patterned_bytes(out_width.div_ceil(32) * blocks, 37);
        let mut want = vec![0.0f32; out_width];
        for (row, slot) in want.iter_mut().enumerate() {
            let weight_row = &weight[row * inner..(row + 1) * inner];
            let scale_row = &weight_scales[(row / 32) * blocks..(row / 32 + 1) * blocks];
            let mut accumulator = 0.0f32;
            for kb in 0..blocks {
                let base = kb * 32;
                let width = (inner - base).min(32);
                let mut lane = [0.0f32; 32];
                for column in 0..width {
                    lane[column] = quant::f8e4m3_to_f32(act_codes[base + column])
                        * quant::f8e4m3_to_f32(weight_row[base + column]);
                }
                let mut offset = 16;
                while offset > 0 {
                    for partial in 0..offset {
                        lane[partial] += lane[partial + offset];
                    }
                    offset /= 2;
                }
                accumulator += (lane[0] * quant::f8e8m0_to_f32(act_scales[kb]))
                    * quant::f8e8m0_to_f32(scale_row[kb]);
            }
            *slot = quant::bf16_round(accumulator);
        }
        let mut got = vec![0.0f32; out_width];
        matmul_f8_w8a8(
            &mut got,
            &act_codes,
            &act_scales,
            &weight,
            &weight_scales,
            out_width,
            inner,
        )
        .expect("w8a8");
        assert_same_bits(&want, &got);
    }

    #[test]
    fn fused_grouped_f8_matches_the_group_loop_bitwise() {
        let (groups, rows_per_group, cols_per_group) = (4usize, 32usize, 128usize);
        let rows = groups * rows_per_group;
        let width = groups * cols_per_group;
        let codes = patterned_bytes(rows * cols_per_group, 21);
        let scales = patterned_bytes(rows.div_ceil(32) * cols_per_group.div_ceil(32), 23);
        let x: Vec<f32> = (0..width)
            .map(|index| (index as f32) / 19.0 - 2.0)
            .collect();
        let mut want = vec![0.0f32; rows];
        for group in 0..groups {
            let row = group * rows_per_group;
            let scale_row = row.div_ceil(32);
            let scale_cols = cols_per_group.div_ceil(32);
            matmul_f8_scaled(
                &mut want[row..row + rows_per_group],
                &x[group * cols_per_group..(group + 1) * cols_per_group],
                &codes[row * cols_per_group..(row + rows_per_group) * cols_per_group],
                &scales[scale_row * scale_cols
                    ..(scale_row + rows_per_group.div_ceil(32)) * scale_cols],
                ScaleGrid::Block32,
                rows_per_group,
                cols_per_group,
                1,
            )
            .expect("group");
        }
        let mut got = vec![0.0f32; rows];
        matmul_f8_grouped(
            &mut got,
            &x,
            &codes,
            &scales,
            ScaleGrid::Block32,
            groups,
            rows_per_group,
            cols_per_group,
        )
        .expect("grouped");
        assert_same_bits(&want, &got);
    }

    #[test]
    fn fused_coded_matmul_refuses_short_buffers() {
        let mut out = [0.0f32; 4];
        let x = [1.0f32; 4];
        let codes = [0u8; 8];
        let scales = [0u8; 1];
        assert!(matches!(
            matmul_f8_scaled(&mut out, &x, &codes, &scales, ScaleGrid::Block32, 4, 4, 1),
            Err(EngineError::Shape { .. })
        ));
        assert!(matches!(
            matmul_f8_scaled(&mut out, &x, &[0u8; 16], &[], ScaleGrid::Block32, 4, 4, 1),
            Err(EngineError::Shape { .. })
        ));
        assert!(matches!(
            matmul_bf16(&mut out, &x, &[0u8; 4], 4, 4, 1),
            Err(EngineError::Shape { .. })
        ));
    }

    #[test]
    fn matmul_matches_the_naive_product() {
        // 2 rows out, 3 columns in, one batch row: [1 2 3] * [[1 0 0],[0 1 0]] = [1 2].
        let mut out = [0.0f32; 2];
        matmul(
            &mut out,
            &[1.0, 2.0, 3.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            2,
            3,
            1,
        )
        .unwrap();
        assert_eq!(out, [1.0, 2.0]);
        // The same weights on a second batch row must not see the first row's values.
        let mut two = [0.0f32; 4];
        matmul(
            &mut two,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            2,
            3,
            2,
        )
        .unwrap();
        assert_eq!(two, [1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn matmul_refuses_geometry_that_does_not_fit() {
        let mut out = [0.0f32; 2];
        let error = matmul(&mut out, &[1.0, 2.0], &[1.0, 0.0, 0.0, 1.0], 2, 3, 1).unwrap_err();
        assert!(matches!(error, EngineError::Shape { .. }), "{error}");
        assert!(error.to_string().contains("3"), "{error}");
    }

    #[test]
    fn rms_norm_keeps_a_unit_row_at_one_and_scales_the_rest() {
        let mut row = [3.0f32, 4.0, 0.0, 0.0];
        rms_norm(&mut row, &[1.0; 4], 4, 0.0).unwrap();
        // rms = sqrt(25/4) = 2.5, so 3/2.5 = 1.2 and 4/2.5 = 1.6.
        assert!((row[0] - 1.2).abs() < 1e-6, "{row:?}");
        assert!((row[1] - 1.6).abs() < 1e-6, "{row:?}");
        let mut unit = [1.0f32, 1.0];
        rms_norm(&mut unit, &[2.0, 3.0], 2, 0.0).unwrap();
        assert!(
            (unit[0] - 2.0).abs() < 1e-6 && (unit[1] - 3.0).abs() < 1e-6,
            "{unit:?}"
        );
    }

    #[test]
    fn softmax_is_a_distribution_and_all_inf_is_zero() {
        let mut x = [1.0f32, 2.0, 3.0];
        softmax(&mut x);
        let total: f32 = x.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "{x:?}");
        assert!(x[2] > x[1] && x[1] > x[0]);
        let mut masked = [f32::NEG_INFINITY; 3];
        softmax(&mut masked);
        assert_eq!(masked, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn a_rope_rotation_is_undone_by_its_inverse() {
        let original = [0.5f32, -1.25, 2.0, 0.75];
        let cos = [0.6f32, 0.8];
        let sin = [0.8f32, 0.6];
        let mut x = original;
        rope_rotate(&mut x, &cos, &sin).unwrap();
        assert_ne!(x, original);
        rope_rotate_inverse(&mut x, &cos, &sin).unwrap();
        for (got, want) in x.iter().zip(original.iter()) {
            assert!((got - want).abs() < 1e-6, "{x:?} vs {original:?}");
        }
    }

    #[test]
    fn rope_refuses_a_slice_that_is_not_whole_pairs() {
        let mut x = [1.0f32, 2.0, 3.0];
        let error = rope_rotate(&mut x, &[1.0], &[0.0]).unwrap_err();
        assert!(matches!(error, EngineError::Shape { .. }), "{error}");
    }

    #[test]
    fn the_block_grid_multiplies_every_element_by_its_own_block() {
        // Four 32x32 blocks of a 64x64 weight, one scale each: 1, 2, 4, 8.
        let mut weight = vec![1.0f32; 64 * 64];
        let scales = [1.0f32, 2.0, 4.0, 8.0];
        apply_scales(&mut weight, 64, 64, &scales, ScaleGrid::Block32).unwrap();
        assert_eq!(weight[0], 1.0);
        assert_eq!(weight[32], 2.0);
        assert_eq!(weight[64 * 32], 4.0);
        assert_eq!(weight[64 * 32 + 32], 8.0);
    }

    #[test]
    fn the_per_row_grid_uses_the_row_index_not_the_block_column() {
        // 3 rows, 64 columns: 2 scales per row, so row 1 must not read row 0's scales.
        let mut weight = vec![1.0f32; 3 * 64];
        let scales = [1.0f32, 2.0, 4.0, 8.0, 16.0, 32.0];
        apply_scales(&mut weight, 3, 64, &scales, ScaleGrid::PerRow32).unwrap();
        assert_eq!(weight[0], 1.0);
        assert_eq!(weight[32], 2.0);
        assert_eq!(weight[64], 4.0);
        assert_eq!(weight[64 + 32], 8.0);
        assert_eq!(weight[128], 16.0);
        assert_eq!(weight[128 + 32], 32.0);
    }

    #[test]
    fn a_scale_grid_of_the_wrong_size_is_refused() {
        let mut weight = vec![1.0f32; 64 * 64];
        let error = apply_scales(&mut weight, 64, 64, &[1.0, 2.0], ScaleGrid::Block32).unwrap_err();
        assert!(matches!(error, EngineError::Shape { .. }), "{error}");
        assert!(error.to_string().contains("4 scales"), "{error}");
    }

    #[test]
    fn silu_and_sigmoid_agree_at_the_origin_and_saturate() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert_eq!(silu(0.0), 0.0);
        assert!(sigmoid(-100.0) >= 0.0 && sigmoid(100.0) <= 1.0);
        assert!(sigmoid(-100.0) < 1e-30, "{}", sigmoid(-100.0));
    }

    #[test]
    fn argmax_takes_the_lower_index_on_a_tie() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0]), Some(1));
        assert_eq!(argmax(&[]), None);
        assert_eq!(mean(&[]), 0.0);
        assert!((mean(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-6);
    }

    /// The exact bits of a slice, for comparing two runs that must agree bit for bit.
    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn the_parallel_matmul_is_bit_identical_to_the_serial_one() {
        // Batch > 1 and ten million multiplies, so the parallel arm is taken. Thread count 1 is the
        // serial arm; thread count 4 splits the output rows into four chunks and the assertion
        // proves those chunks compute the same f32 bits as the serial loop. The values are not
        // powers of two, so any reassociation would show up in the low bits. This proves the split
        // is value-transparent, not anything about a particular scheduling.
        let rows = 5_000;
        let cols = 1_000;
        let batch = 2;
        let x: Vec<f32> = (0..batch * cols)
            .map(|index| ((index as f32) * 0.001).sin())
            .collect();
        let w: Vec<f32> = (0..rows * cols)
            .map(|index| ((index as f32) * 0.002).cos())
            .collect();
        let run = |threads: usize| {
            crate::parallel::with_reference_threads(threads, || {
                let mut out = vec![0.0f32; batch * rows];
                matmul(&mut out, &x, &w, rows, cols, batch).expect("matmul");
                out
            })
        };
        assert_eq!(bits(&run(1)), bits(&run(4)));
    }

    /// The ascending-column dot the serial arm has always used, written here so the bitwise
    /// comparison does not depend on the arm under test.
    fn serial_matmul_reference(
        x: &[f32],
        w: &[f32],
        rows: usize,
        cols: usize,
        batch: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * rows];
        for b in 0..batch {
            let xb = &x[b * cols..b * cols + cols];
            for row in 0..rows {
                let wr = &w[row * cols..row * cols + cols];
                let mut acc = 0.0f32;
                for col in 0..cols {
                    acc += xb[col] * wr[col];
                }
                out[b * rows + row] = acc;
            }
        }
        out
    }

    #[test]
    fn the_gate_shapes_keep_the_serial_dot_order_at_one_and_twenty_threads() {
        // The shapes the retired batch-one/rows-under-16_384 arm used to send down the serial path:
        // the attention wo_a/wo_b GEMVs, the shared expert and an expert body. Every one is over the
        // four-million-product gate, so the parallel arm is taken. The comparison is against an
        // independently written ascending-column loop, not against the other arm, and it is on the
        // raw f32 bits.
        for (rows, cols, batch) in [
            (2_304usize, 5_120usize, 1usize),
            (5_121, 1_280, 1),
            (8_192, 4_096, 1),
        ] {
            let x: Vec<f32> = (0..batch * cols)
                .map(|index| ((index as f32) * 0.0017).sin())
                .collect();
            let w: Vec<f32> = (0..rows * cols)
                .map(|index| ((index as f32) * 0.0023).cos())
                .collect();
            let want = bits(&serial_matmul_reference(&x, &w, rows, cols, batch));
            for threads in [1usize, 3, 7, 20] {
                let got = crate::parallel::with_reference_threads(threads, || {
                    let mut out = vec![0.0f32; batch * rows];
                    matmul(&mut out, &x, &w, rows, cols, batch).expect("matmul");
                    out
                });
                assert_eq!(
                    bits(&got),
                    want,
                    "{rows}x{cols} batch {batch} at {threads} threads"
                );
            }
        }
    }

    #[test]
    fn the_gate_shapes_are_deterministic_across_repeats_at_twenty_threads() {
        let rows = 5_120usize;
        let cols = 2_304usize;
        let x: Vec<f32> = (0..cols)
            .map(|index| ((index as f32) * 0.0031).sin())
            .collect();
        let w: Vec<f32> = (0..rows * cols)
            .map(|index| ((index as f32) * 0.0043).cos())
            .collect();
        let run = |threads: usize| {
            crate::parallel::with_reference_threads(threads, || {
                let mut out = vec![0.0f32; rows];
                matmul(&mut out, &x, &w, rows, cols, 1).expect("matmul");
                out
            })
        };
        let serial = bits(&run(1));
        let first = bits(&run(20));
        let second = bits(&run(20));
        let odd = bits(&run(13));
        assert_eq!(first, serial);
        assert_eq!(second, serial);
        assert_eq!(odd, serial);
    }

    #[test]
    fn the_parallel_scale_grid_is_bit_identical_to_the_serial_one() {
        let rows = 2_048usize;
        let cols = 64usize;
        let weight: Vec<f32> = (0..rows * cols)
            .map(|index| ((index as f32) * 0.003).sin())
            .collect();
        // The two grids have different lengths: Block32 scales a 32x32 element block, PerRow32
        // scales one 32-element block per row. Test both, each with the size the shape requires.
        let block32: Vec<f32> = (0..rows.div_ceil(32) * cols.div_ceil(32))
            .map(|index| 1.0 + index as f32 * 1e-3)
            .collect();
        let per_row32: Vec<f32> = (0..rows * cols.div_ceil(32))
            .map(|index| 1.0 + index as f32 * 2e-3)
            .collect();
        for (grid, scales) in [
            (ScaleGrid::Block32, block32.as_slice()),
            (ScaleGrid::PerRow32, per_row32.as_slice()),
        ] {
            let run = |threads: usize| {
                crate::parallel::with_reference_threads(threads, || {
                    let mut copy = weight.clone();
                    apply_scales(&mut copy, rows, cols, scales, grid).expect("scales");
                    copy
                })
            };
            assert_eq!(bits(&run(1)), bits(&run(4)), "grid {grid:?}");
        }
    }

    #[test]
    fn the_parallel_rms_norm_is_bit_identical_to_the_serial_one() {
        let rows = 1_024;
        let cols = 128;
        let x: Vec<f32> = (0..rows * cols)
            .map(|index| ((index as f32) * 0.004).sin())
            .collect();
        let weight: Vec<f32> = (0..cols).map(|index| 1.0 + index as f32 * 1e-3).collect();
        let run = |threads: usize| {
            crate::parallel::with_reference_threads(threads, || {
                let mut copy = x.clone();
                rms_norm(&mut copy, &weight, cols, 1e-5).expect("rms norm");
                copy
            })
        };
        assert_eq!(bits(&run(1)), bits(&run(4)));
    }
}

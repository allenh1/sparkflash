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
    // A matvec with one batch row is mostly a weight read; starting a scoped thread for it costs
    // more than the arithmetic saves unless the row set is matrix-sized (a vocabulary head). Above
    // that gate the split is over output rows, so every output keeps its serial reduction order.
    if multiplies < 4_000_000 || (batch == 1 && rows < 16_384) {
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

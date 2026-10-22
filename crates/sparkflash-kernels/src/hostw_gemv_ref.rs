//! CPU twin of `kernels/b12x_hostw_gemv_m1_sm121.cu`: the host-weight decode GEMV the ref_seam
//! block's compressor and indexer projections run on the device.
//!
//! The kernel is templated on the weight's element type (bf16 widened exactly, or f32) and on the
//! store (bf16 rounding or the f32 accumulator). This module writes the same three instantiations
//! the way the kernel writes them - one accumulator per output column, K ascending, a rounded
//! multiply then a rounded add per element, one bf16 round at the store when the seam rounds - and
//! exists so the device chain can be compared with the engine's host rules on a CPU host, without a
//! GPU and without a second copy of the arithmetic in the test file.
//!
//! The tiling is the coded GEMV's B12x permutation scaled by the element size: a 32x32 micro-tile
//! of the weight's own values, `tile = (n / 32) * k_blocks + (k / 32)`, element offset
//! `(n % 32) * 32 + (k % 32)` inside it. A pure permutation - the host stages it once per layer and
//! no value changes - and the kernel reads it as one coalesced run per warp.

/// The B12x element index of the logical weight element at `(col, k)`.
///
/// `inner` is the reduction length; `cols` is implied by the buffer. This is the same permutation
/// [crate::mxfp8_ref::b12x_weight_index] applies to E4M3 byte payloads, scaled by the element size
/// of the weight's own type.
///
/// # Panics
/// In debug builds, when `col` or `k` is outside a whole number of 32-value micro-tiles.
#[must_use]
pub fn b12x_hostw_weight_index(col: usize, k: usize, inner: usize) -> usize {
    debug_assert!(
        inner.is_multiple_of(32),
        "the reduction is whole 32-value blocks"
    );
    let k_blocks = inner / 32;
    let tile = (col / 32) * k_blocks + (k / 32);
    tile * 1024 + (col % 32) * 32 + (k % 32)
}

/// Stages a row-major `[cols][inner]` f32 weight into the kernel's B12x element order.
#[must_use]
pub fn stage_f32_weight(weight: &[f32], cols: usize, inner: usize) -> Vec<f32> {
    let mut staged = vec![0.0f32; cols * inner];
    for col in 0..cols {
        for k in 0..inner {
            staged[b12x_hostw_weight_index(col, k, inner)] = weight[col * inner + k];
        }
    }
    staged
}

/// Stages a row-major `[cols][inner]` bf16 weight (raw 16-bit patterns) into the kernel's B12x
/// element order.
#[must_use]
pub fn stage_bf16_weight(weight: &[u16], cols: usize, inner: usize) -> Vec<u16> {
    let mut staged = vec![0u16; cols * inner];
    for col in 0..cols {
        for k in 0..inner {
            staged[b12x_hostw_weight_index(col, k, inner)] = weight[col * inner + k];
        }
    }
    staged
}

/// The exact widening the kernel's bf16 mode performs: a bf16 pattern is the top 16 bits of an f32.
#[must_use]
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// One output row of the device chain, written exactly as the kernel writes it.
///
/// `element(col, k)` is the weight value: for the f32 mode the staged f32 itself, for the bf16
/// modes the exact widening of the staged bf16. Ascending `k`, a rounded multiply then a rounded
/// add per element, one accumulator per output column - the engine matmul's chain.
#[must_use]
pub fn row<F>(a_row: &[f32], cols: usize, inner: usize, element: F) -> Vec<f32>
where
    F: Fn(usize, usize) -> f32,
{
    assert_eq!(a_row.len(), inner, "the activation row is `inner` values");
    let mut out = vec![0.0f32; cols];
    for (col, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (k, a) in a_row.iter().enumerate() {
            acc += *a * element(col, k);
        }
        *slot = acc;
    }
    out
}

/// The f32-weight mode: f32 activation x f32 weight, f32 accumulators.
///
/// # Panics
/// When the staged weight is not `cols * inner` values.
#[must_use]
pub fn f32_weight_row(a_row: &[f32], staged: &[f32], cols: usize, inner: usize) -> Vec<f32> {
    assert_eq!(
        staged.len(),
        cols * inner,
        "the staged f32 weight is `cols * inner` values"
    );
    row(a_row, cols, inner, |col, k| {
        staged[b12x_hostw_weight_index(col, k, inner)]
    })
}

/// The bf16-weight modes: f32 activation x bf16 weight widened exactly, f32 accumulators.
///
/// # Panics
/// When the staged weight is not `cols * inner` values.
#[must_use]
pub fn bf16_weight_row(a_row: &[f32], staged: &[u16], cols: usize, inner: usize) -> Vec<f32> {
    assert_eq!(
        staged.len(),
        cols * inner,
        "the staged bf16 weight is `cols * inner` values"
    );
    row(a_row, cols, inner, |col, k| {
        bf16_bits_to_f32(staged[b12x_hostw_weight_index(col, k, inner)])
    })
}

/// The bf16 store: the one rounding the bf16-output mode applies to each accumulator.
#[must_use]
pub fn bf16_store(accumulators: &[f32]) -> Vec<u16> {
    accumulators
        .iter()
        .map(|value| crate::m4_quant::bf16_bits(*value))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_staged_layout_is_a_permutation_of_the_row_major_weight() {
        for (cols, inner) in [
            (32usize, 32usize),
            (64, 128),
            (512, 5120),
            (128, 512),
            (32, 5120),
        ] {
            let weight: Vec<f32> = (0..cols * inner).map(|i| i as f32 * 0.5).collect();
            let staged = stage_f32_weight(&weight, cols, inner);
            let mut seen = vec![false; cols * inner];
            for col in 0..cols {
                for k in 0..inner {
                    let index = b12x_hostw_weight_index(col, k, inner);
                    assert!(!seen[index], "{cols}x{inner}: index {index} written twice");
                    seen[index] = true;
                    assert_eq!(staged[index], weight[col * inner + k]);
                }
            }
            assert!(seen.iter().all(|written| *written));
        }
    }

    #[test]
    fn the_row_is_the_rounded_multiply_then_rounded_add_chain() {
        let (cols, inner) = (4usize, 32usize);
        let a_row: Vec<f32> = (0..inner).map(|k| 1.0 + k as f32 * 0.125).collect();
        let weight: Vec<f32> = (0..cols * inner)
            .map(|i| (i % 7) as f32 * 0.25 - 0.75)
            .collect();
        let staged = stage_f32_weight(&weight, cols, inner);
        let out = f32_weight_row(&a_row, &staged, cols, inner);
        for col in 0..cols {
            let mut acc = 0.0f32;
            for k in 0..inner {
                acc += a_row[k] * weight[col * inner + k];
            }
            assert_eq!(out[col].to_bits(), acc.to_bits(), "col {col}");
        }
    }

    #[test]
    fn the_bf16_mode_widens_exactly_and_rounds_once_at_the_store() {
        let (cols, inner) = (32usize, 32usize);
        let a_row: Vec<f32> = (0..inner).map(|k| (k as f32 - 16.0) * 0.0625).collect();
        let weight: Vec<f32> = (0..cols * inner)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.125)
            .collect();
        let bf16: Vec<u16> = weight
            .iter()
            .map(|value| crate::m4_quant::bf16_bits(*value))
            .collect();
        let widened: Vec<f32> = bf16.iter().map(|bits| bf16_bits_to_f32(*bits)).collect();
        let staged = stage_bf16_weight(&bf16, cols, inner);
        let out = bf16_weight_row(&a_row, &staged, cols, inner);
        let f32_staged = stage_f32_weight(&widened, cols, inner);
        let reference = f32_weight_row(&a_row, &f32_staged, cols, inner);
        assert_eq!(out, reference, "a bf16 weight is its exact f32 widening");
        let stored = bf16_store(&out);
        for (col, bits) in stored.iter().enumerate() {
            assert_eq!(*bits, crate::m4_quant::bf16_bits(out[col]));
        }
    }
}

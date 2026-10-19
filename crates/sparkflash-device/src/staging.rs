//! M3-b layer-0 weight staging through WeightSource.
//!
//! This is the first data path into the device chain: the layer-0 mHC tensors and the declared
//! embedding rows are read verbatim through [WeightSource], checked against the released dtype and
//! shape before their bytes are accepted, and kept in the exact form the typed mHC launch consumes.
//! It is deliberately not an all-resident loader: only layer 0's attention mHC tensors, its
//! attn_norm.weight and the declared embed rows are read.
//!
//! Reads and checks, in order. Every tensor's [WeightSource::spec] is checked against the released
//! dtype and shape before [WeightSource::read] is called, so a type or shape mismatch costs no data
//! read and is named by tensor. The bytes read must be exactly the spec's storage length. The mHC
//! predictor is permuted from the released row order into the kernel B/A/C order with
//! [sparkflash_kernels::mega_mhc_ref::released_predictor_into_kernel_rows] - no averaging - and the
//! base is rearranged with [sparkflash_kernels::mega_mhc_ref::bias_into_kernel_rows]. The report
//! carries two clearly labelled digest sets: a source digest over the checkpoint bytes read, and a
//! buffer digest over the exact host vector later passed to copy_htod (permuted predictor, kernel
//! order base, widened norm, four-copy X row). There is no hidden conversion or re-read.
//!
//! Embed rows are paged, not materialised. embed.weight is 1.32 GB of BF16, so this path refuses a
//! source that cannot page rows with [WeightSource::read_ranges] rather than falling back to a
//! whole-tensor read; the source's own size limit is not the guard.
//!
//! The layer-0 boundary seed is host-built and declared: B = identity, C = 0, A = [1, 0, 0, 0],
//! Y_-1 = 0, because the checkpoint stores no layer -1 coefficients. It is not read from the
//! checkpoint and it is not a device result.

use sha2::{Digest, Sha256};
use sparkflash_engine::dtype::DType;
use sparkflash_engine::weights::{WeightError, WeightSource};
use sparkflash_kernels::m4_quant::{QUANT_BLOCK, bf16_value};
use sparkflash_kernels::mega_mhc_ref::{
    HC_MULT, HIDDEN, PREDICTOR_ROWS, bias_into_kernel_rows, released_predictor_into_kernel_rows,
};
use thiserror::Error;

/// The number of mHC device buffers one layer-0 launch needs.
pub const MHC_BUFFER_COUNT: usize = 16;

/// The grid statistics scratch the mHC kernel needs: STATS x (tiles + 1).
pub const MHC_GRID_STATS: usize = 130;

/// The layer-0 checkpoint prefix.
pub const LAYER0_PREFIX: &str = "layers.0";

/// The released layer-0 mHC predictor: [PREDICTOR_ROWS, HC_MULT * HIDDEN] F32.
pub const LAYER0_MHC_FN: &str = "layers.0.hc_attn_fn";

/// The released layer-0 mHC bias: [PREDICTOR_ROWS] F32.
pub const LAYER0_MHC_BASE: &str = "layers.0.hc_attn_base";

/// The released layer-0 mHC group scale: \[3\] F32.
pub const LAYER0_MHC_SCALE: &str = "layers.0.hc_attn_scale";

/// The released layer-0 attention norm: [HIDDEN] F32 or BF16.
pub const LAYER0_ATTN_NORM: &str = "layers.0.attn_norm.weight";

/// The released embedding table: [vocab, HIDDEN] BF16.
pub const EMBED_WEIGHT: &str = "embed.weight";

/// One tensor's provenance: its raw byte count and SHA-256 as read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedTensor {
    /// The tensor name, or a row name for an embed row.
    pub name: String,
    /// Raw bytes read.
    pub bytes: u64,
    /// SHA-256 of those bytes, lower-case hex.
    pub sha256: String,
}

/// One declared embedding row, kept as the checkpoint's BF16 bytes.
///
/// The fields are private and `EmbedRow::new` is crate-private: [LayerZeroStaging] is the only
/// constructor, so a row that reaches a launch is the row staging read, validated and digested.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedRow {
    position: usize,
    bf16: Vec<u8>,
}

impl EmbedRow {
    /// Builds and validates one embedding row from the checkpoint bytes.
    ///
    /// The row must be exactly one HIDDEN BF16 row, so `EmbedRow::x_bf16` is exactly
    /// HC_MULT copies (HC_MULT * HIDDEN * 2 bytes). The position must be inside the declared
    /// table; a zero-length row is rejected by the same exact-length check.
    ///
    /// # Errors
    /// [StagingError::Position] for an out-of-range position and [StagingError::Bytes] when the
    /// row is not exactly HIDDEN BF16 values.
    pub(crate) fn new(position: usize, bf16: Vec<u8>, rows: usize) -> Result<Self, StagingError> {
        if rows == 0 || position >= rows {
            return Err(StagingError::Position {
                name: EMBED_WEIGHT.to_string(),
                position,
                rows,
            });
        }
        // A row is one HIDDEN BF16 row, so its four-copy expansion must be exactly
        // HC_MULT * HIDDEN * 2 bytes. checked_mul also refuses a length that cannot expand.
        if bf16.len().checked_mul(HC_MULT) != Some(HC_MULT * HIDDEN * 2) {
            return Err(StagingError::Bytes {
                name: format!("{EMBED_WEIGHT}[{position}]"),
                expected: HIDDEN * 2,
                found: bf16.len(),
            });
        }
        Ok(Self { position, bf16 })
    }

    /// The layer-0 X_-1 seed: the row repeated HC_MULT times.
    ///
    /// Public so a window caller can compare the device X readback against the exact bytes the
    /// launcher copied; the row's fields remain private and its only constructor is the admitted
    /// staging path.
    #[must_use]
    pub fn x_bf16(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.bf16.len() * HC_MULT);
        for _ in 0..HC_MULT {
            bytes.extend_from_slice(&self.bf16);
        }
        bytes
    }

    /// Widens the row to f32 for the host layer-0 trace.
    ///
    /// The host pass is the reference arithmetic; it consumes the same released BF16 row the
    /// device X seed gets, widened exactly as the kernel's f32 capture would be. This is a host
    /// seam on the c1 path, not a device readback.
    #[must_use]
    pub fn values_f32(&self) -> Vec<f32> {
        self.bf16
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| bf16_value(u16::from_le_bytes(*chunk)))
            .collect()
    }

    /// Builds a row without validation, only so tests can exercise the launch-time length check.
    #[cfg(all(test, feature = "cuda"))]
    pub(crate) fn new_unchecked(position: usize, bf16: Vec<u8>) -> Self {
        Self { position, bf16 }
    }
}

/// The exact byte sizes and nominal offsets of the sixteen mHC device buffers.
///
/// Each buffer is a separate driver allocation, so the offsets document the logical layout and the
/// staging report, not one packed allocation. Every size is the released shape's byte size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MhcBufferPlan {
    /// X_l, four BF16 copies.
    pub x: usize,
    /// Y_-1, BF16.
    pub y_prev: usize,
    /// B, f32.
    pub b_prev: usize,
    /// C, f32.
    pub c_prev: usize,
    /// A, f32.
    pub a_prev: usize,
    /// The 24 x 4d predictor, f32.
    pub predictor: usize,
    /// Three group scales, f32.
    pub hc_scale: usize,
    /// Twenty-four kernel-order biases, f32.
    pub hc_base: usize,
    /// attn_norm.weight widened to f32.
    pub rms_weight: usize,
    /// E4M3 block input codes.
    pub xhat_fp8: usize,
    /// UE8M0 block scales.
    pub xhat_scales: usize,
    /// Grid statistics scratch.
    pub hc_stats: usize,
    /// Grid synchronisation scratch.
    pub hc_sync: usize,
    /// B/A/C coefficients.
    pub hc_coeffs: usize,
    /// 1/rms(X_hat).
    pub hc_rms_out: usize,
    /// The f32 block-input capture.
    pub xhat_normed: usize,
}

impl MhcBufferPlan {
    /// The layer-0 buffer plan, in the exact kernel parameter order.
    pub const LAYER_ZERO: Self = Self {
        x: HC_MULT * HIDDEN * 2,
        y_prev: HIDDEN * 2,
        b_prev: HC_MULT * HC_MULT * 4,
        c_prev: HC_MULT * 4,
        a_prev: HC_MULT * 4,
        predictor: PREDICTOR_ROWS * HC_MULT * HIDDEN * 4,
        hc_scale: 3 * 4,
        hc_base: PREDICTOR_ROWS * 4,
        rms_weight: HIDDEN * 4,
        xhat_fp8: HIDDEN,
        xhat_scales: HIDDEN / QUANT_BLOCK,
        hc_stats: MHC_GRID_STATS * 4,
        hc_sync: 2 * 4,
        hc_coeffs: PREDICTOR_ROWS * 4,
        hc_rms_out: 4,
        xhat_normed: HIDDEN * 4,
    };

    /// The sixteen (name, size) pairs in kernel parameter order.
    #[must_use]
    pub fn sizes(self) -> [(&'static str, usize); MHC_BUFFER_COUNT] {
        [
            ("x", self.x),
            ("y_prev", self.y_prev),
            ("b_prev", self.b_prev),
            ("c_prev", self.c_prev),
            ("a_prev", self.a_prev),
            ("predictor", self.predictor),
            ("hc_scale", self.hc_scale),
            ("hc_base", self.hc_base),
            ("rms_weight", self.rms_weight),
            ("xhat_fp8", self.xhat_fp8),
            ("xhat_scales", self.xhat_scales),
            ("hc_stats", self.hc_stats),
            ("hc_sync", self.hc_sync),
            ("hc_coeffs", self.hc_coeffs),
            ("hc_rms_out", self.hc_rms_out),
            ("xhat_normed", self.xhat_normed),
        ]
    }

    /// The total device bytes the sixteen allocations request.
    #[must_use]
    pub fn device_bytes(self) -> usize {
        self.sizes().iter().map(|(_, bytes)| bytes).sum()
    }

    /// The nominal offset of every buffer in the documented layout.
    #[must_use]
    pub fn offsets(self) -> [(&'static str, usize); MHC_BUFFER_COUNT] {
        let mut offsets = [("", 0usize); MHC_BUFFER_COUNT];
        let mut at = 0usize;
        for (index, (name, bytes)) in self.sizes().iter().enumerate() {
            offsets[index] = (*name, at);
            at += bytes;
        }
        offsets
    }
}

/// A typed staging failure. Every message names the tensor and keeps the source's detail.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum StagingError {
    /// The source refused the tensor or its spec.
    #[error("tensor {name}: {detail}")]
    Source {
        /// The tensor name.
        name: String,
        /// The source detail.
        detail: String,
    },
    /// The dtype or shape disagrees with the released tensor.
    #[error("tensor {name}: expected {expected}, the source reports {found}")]
    Spec {
        /// The tensor name.
        name: String,
        /// The released dtype/shape.
        expected: String,
        /// What the source reported.
        found: String,
    },
    /// The byte count is not exactly the spec's storage length.
    #[error("tensor {name}: read {found} bytes, the spec requires {expected}")]
    Bytes {
        /// The tensor name.
        name: String,
        /// The spec's storage length.
        expected: usize,
        /// The bytes returned.
        found: usize,
    },
    /// The source cannot page embed rows and a whole-tensor read is refused.
    #[error(
        "tensor {name}: the source cannot page a row (read_ranges returned None); refusing a {bytes}-byte whole-tensor read"
    )]
    EmbedUnpaged {
        /// The tensor name.
        name: String,
        /// The tensor's whole storage length.
        bytes: u128,
    },
    /// A declared position is outside the embedding table.
    #[error("tensor {name}: position {position} is outside the {rows}-row table")]
    Position {
        /// The tensor name.
        name: String,
        /// The declared position.
        position: usize,
        /// The table's row count.
        rows: usize,
    },
}

/// The staged layer-0 data: exactly what the mHC launch consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerZeroStaging {
    /// The released 24 x 4d predictor, permuted into kernel B/A/C row order (no averaging).
    pub predictor: Vec<f32>,
    /// The released hc_attn_base, in the checkpoint's row order.
    pub hc_base: [f32; PREDICTOR_ROWS],
    /// The released hc_attn_scale, in checkpoint order (pre, post, comb).
    pub hc_scale: [f32; 3],
    /// attn_norm.weight widened to f32.
    pub attn_norm: Vec<f32>,
    /// hc_attn_base rearranged into the kernel's row order.
    pub hc_base_kernel: [f32; PREDICTOR_ROWS],
    /// The declared embedding rows, in the requested order.
    pub embed_rows: Vec<EmbedRow>,
    /// One source-provenance digest per raw tensor and per embed row, as read.
    pub digests: Vec<StagedTensor>,
    /// One digest per exact host vector the staging path passes to copy_htod.
    ///
    /// These are not the source bytes: the predictor is permuted, the base is in kernel row order,
    /// the norm is widened to f32 and each X row is the four-copy BF16 seed.
    pub buffer_digests: Vec<StagedTensor>,
}

impl LayerZeroStaging {
    /// Reads the layer-0 mHC tensors and the declared embed rows through a WeightSource.
    ///
    /// # Errors
    /// [StagingError] on a missing tensor, a dtype/shape mismatch, a short read, an out-of-range
    /// position, or a source that cannot page a row.
    pub fn from_source(
        source: &dyn WeightSource,
        positions: &[usize],
    ) -> Result<Self, StagingError> {
        let fn_bytes = read_checked(
            source,
            LAYER0_MHC_FN,
            DType::F32,
            &[PREDICTOR_ROWS, HC_MULT * HIDDEN],
        )?;
        let released_fn = bytes_to_f32(LAYER0_MHC_FN, &fn_bytes)?;
        let mut predictor = vec![0.0f32; released_fn.len()];
        released_predictor_into_kernel_rows(&released_fn, &mut predictor);

        let base_bytes = read_checked(source, LAYER0_MHC_BASE, DType::F32, &[PREDICTOR_ROWS])?;
        let base_values = bytes_to_f32(LAYER0_MHC_BASE, &base_bytes)?;
        let hc_base: [f32; PREDICTOR_ROWS] =
            base_values
                .as_slice()
                .try_into()
                .map_err(|_| StagingError::Spec {
                    name: LAYER0_MHC_BASE.to_string(),
                    expected: format!("{PREDICTOR_ROWS} f32 values"),
                    found: format!("{} f32 values", base_values.len()),
                })?;

        let scale_bytes = read_checked(source, LAYER0_MHC_SCALE, DType::F32, &[3])?;
        let scale_values = bytes_to_f32(LAYER0_MHC_SCALE, &scale_bytes)?;
        let hc_scale: [f32; 3] =
            scale_values
                .as_slice()
                .try_into()
                .map_err(|_| StagingError::Spec {
                    name: LAYER0_MHC_SCALE.to_string(),
                    expected: "3 f32 values".to_string(),
                    found: format!("{} f32 values", scale_values.len()),
                })?;

        let norm_spec = source
            .spec(LAYER0_ATTN_NORM)
            .map_err(|error| source_error(LAYER0_ATTN_NORM, error))?;
        if norm_spec.shape != [HIDDEN] || !matches!(norm_spec.dtype, DType::F32 | DType::BF16) {
            return Err(StagingError::Spec {
                name: LAYER0_ATTN_NORM.to_string(),
                expected: format!("F32 or BF16 [{HIDDEN}]"),
                found: format!("{} {:?}", norm_spec.dtype.name(), norm_spec.shape),
            });
        }
        let norm_bytes = source
            .read(LAYER0_ATTN_NORM)
            .map_err(|error| source_error(LAYER0_ATTN_NORM, error))?;
        let norm_expected = norm_spec.dtype.storage_bytes(HIDDEN);
        if norm_bytes.len() != norm_expected {
            return Err(StagingError::Bytes {
                name: LAYER0_ATTN_NORM.to_string(),
                expected: norm_expected,
                found: norm_bytes.len(),
            });
        }
        let attn_norm = if norm_spec.dtype == DType::F32 {
            bytes_to_f32(LAYER0_ATTN_NORM, &norm_bytes)?
        } else {
            bytes_to_bf16_f32(LAYER0_ATTN_NORM, &norm_bytes)?
        };

        let embed_spec = source
            .spec(EMBED_WEIGHT)
            .map_err(|error| source_error(EMBED_WEIGHT, error))?;
        if embed_spec.dtype != DType::BF16
            || embed_spec.shape.len() != 2
            || embed_spec.shape[1] != HIDDEN
        {
            return Err(StagingError::Spec {
                name: EMBED_WEIGHT.to_string(),
                expected: format!("BF16 [vocab, {HIDDEN}]"),
                found: format!("{} {:?}", embed_spec.dtype.name(), embed_spec.shape),
            });
        }
        let rows = embed_spec.shape[0];
        let row_bytes = HIDDEN * 2;
        let mut embed_rows = Vec::with_capacity(positions.len());
        let mut digests = vec![
            digest(LAYER0_MHC_FN, &fn_bytes),
            digest(LAYER0_MHC_BASE, &base_bytes),
            digest(LAYER0_MHC_SCALE, &scale_bytes),
            digest(LAYER0_ATTN_NORM, &norm_bytes),
        ];
        for &position in positions {
            if position >= rows {
                return Err(StagingError::Position {
                    name: EMBED_WEIGHT.to_string(),
                    position,
                    rows,
                });
            }
            let offset = position
                .checked_mul(row_bytes)
                .ok_or_else(|| StagingError::Position {
                    name: EMBED_WEIGHT.to_string(),
                    position,
                    rows,
                })? as u64;
            let ranges = source
                .read_ranges(EMBED_WEIGHT, &[(offset, row_bytes)])
                .ok_or_else(|| StagingError::EmbedUnpaged {
                    name: EMBED_WEIGHT.to_string(),
                    bytes: embed_spec.storage_bytes(),
                })?;
            let row = ranges.map_err(|error| source_error(EMBED_WEIGHT, error))?;
            if row.len() != row_bytes {
                return Err(StagingError::Bytes {
                    name: format!("{EMBED_WEIGHT}[{position}]"),
                    expected: row_bytes,
                    found: row.len(),
                });
            }
            digests.push(digest(&format!("{EMBED_WEIGHT}[{position}]"), &row));
            embed_rows.push(EmbedRow::new(position, row, rows)?);
        }

        let hc_base_kernel = bias_into_kernel_rows(&hc_base);
        let mut buffer_digests = vec![
            digest(
                &format!("buffer predictor <- {LAYER0_MHC_FN}"),
                &f32_bytes(&predictor),
            ),
            digest(
                &format!("buffer hc_scale <- {LAYER0_MHC_SCALE}"),
                &f32_bytes(&hc_scale),
            ),
            digest(
                &format!("buffer hc_base <- {LAYER0_MHC_BASE}"),
                &f32_bytes(&hc_base_kernel),
            ),
            digest(
                &format!("buffer rms_weight <- {LAYER0_ATTN_NORM}"),
                &f32_bytes(&attn_norm),
            ),
        ];
        for row in &embed_rows {
            buffer_digests.push(digest(
                &format!("buffer x <- {EMBED_WEIGHT}[{}]", row.position),
                &row.x_bf16(),
            ));
        }
        Ok(Self {
            predictor,
            hc_base,
            hc_scale,
            attn_norm,
            hc_base_kernel,
            embed_rows,
            digests,
            buffer_digests,
        })
    }

    /// The retained host bytes of the staged tensors and rows.
    #[must_use]
    pub fn host_bytes(&self) -> u64 {
        let mut bytes = 0u64;
        bytes += (self.predictor.len() * 4) as u64;
        bytes += (self.hc_base.len() * 4) as u64;
        bytes += (self.hc_scale.len() * 4) as u64;
        bytes += (self.attn_norm.len() * 4) as u64;
        bytes += (self.hc_base_kernel.len() * 4) as u64;
        bytes += self
            .embed_rows
            .iter()
            .map(|row| row.bf16.len() as u64)
            .sum::<u64>();
        bytes
    }

    /// The exact device buffer bytes this stage requests.
    #[must_use]
    pub fn device_buffer_bytes(&self) -> u64 {
        MhcBufferPlan::LAYER_ZERO.device_bytes() as u64
    }

    /// The embedding row for one declared position.
    #[must_use]
    pub fn embed_row(&self, position: usize) -> Option<&EmbedRow> {
        self.embed_rows.iter().find(|row| row.position == position)
    }

    /// The layer-0 boundary seed: B = identity, C = 0, A = [1, 0, 0, 0], Y_-1 = 0.
    ///
    /// The checkpoint stores no layer -1 coefficients, so this is host-built and declared, exactly
    /// as the m11 harness declares it.
    #[must_use]
    pub fn layer_zero_seed() -> (
        [f32; HC_MULT * HC_MULT],
        [f32; HC_MULT],
        [f32; HC_MULT],
        Vec<f32>,
    ) {
        let mut b = [0.0f32; HC_MULT * HC_MULT];
        for (index, value) in b.iter_mut().enumerate() {
            if index / HC_MULT == index % HC_MULT {
                *value = 1.0;
            }
        }
        let c = [0.0f32; HC_MULT];
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let y = vec![0.0f32; HIDDEN];
        (b, c, a, y)
    }

    /// The provenance lines a stage report prints before any device call.
    #[must_use]
    pub fn provenance_lines(&self) -> Vec<String> {
        let mut lines = vec![format!(
            "staged layer-0 mHC: predictor {} f32 (released [24, 4*d] -> kernel B/A/C order, no averaging), hc_attn_base {} f32, hc_attn_scale {:?}, attn_norm {} f32, embed rows {}",
            self.predictor.len(),
            self.hc_base.len(),
            self.hc_scale,
            self.attn_norm.len(),
            self.embed_rows.len()
        )];
        for row in &self.embed_rows {
            lines.push(format!(
                "staged embed row {}: {} BF16 bytes",
                row.position,
                row.bf16.len()
            ));
        }
        for tensor in &self.digests {
            lines.push(format!(
                "source tensor {}: {} bytes, sha256 {}",
                tensor.name, tensor.bytes, tensor.sha256
            ));
        }
        for tensor in &self.buffer_digests {
            lines.push(format!(
                "buffer tensor {}: {} bytes, sha256 {}",
                tensor.name, tensor.bytes, tensor.sha256
            ));
        }
        lines
    }
}

fn source_error(name: &str, error: WeightError) -> StagingError {
    StagingError::Source {
        name: name.to_string(),
        detail: error.to_string(),
    }
}

fn read_checked(
    source: &dyn WeightSource,
    name: &str,
    dtype: DType,
    shape: &[usize],
) -> Result<Vec<u8>, StagingError> {
    let spec = source
        .spec(name)
        .map_err(|error| source_error(name, error))?;
    if spec.dtype != dtype || spec.shape != shape {
        return Err(StagingError::Spec {
            name: name.to_string(),
            expected: format!("{} {shape:?}", dtype.name()),
            found: format!("{} {:?}", spec.dtype.name(), spec.shape),
        });
    }
    let expected = spec.storage_bytes();
    let expected = usize::try_from(expected).map_err(|_| StagingError::Bytes {
        name: name.to_string(),
        expected: usize::MAX,
        found: 0,
    })?;
    let bytes = source
        .read(name)
        .map_err(|error| source_error(name, error))?;
    if bytes.len() != expected {
        return Err(StagingError::Bytes {
            name: name.to_string(),
            expected,
            found: bytes.len(),
        });
    }
    Ok(bytes)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bytes_to_f32(name: &str, bytes: &[u8]) -> Result<Vec<f32>, StagingError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(StagingError::Bytes {
            name: name.to_string(),
            expected: bytes.len().div_ceil(4) * 4,
            found: bytes.len(),
        });
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

fn bytes_to_bf16_f32(name: &str, bytes: &[u8]) -> Result<Vec<f32>, StagingError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(StagingError::Bytes {
            name: name.to_string(),
            expected: bytes.len().div_ceil(2) * 2,
            found: bytes.len(),
        });
    }
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| bf16_value(u16::from_le_bytes(*chunk)))
        .collect())
}

fn digest(name: &str, bytes: &[u8]) -> StagedTensor {
    StagedTensor {
        name: name.to_string(),
        bytes: bytes.len() as u64,
        sha256: crate::cubins::hex(&Sha256::digest(bytes)),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use sha2::{Digest, Sha256};
    use sparkflash_engine::tensor::TensorSpec;
    use sparkflash_kernels::m4_quant::bf16_bits;

    use super::*;

    #[test]
    fn embed_row_constructor_refuses_invalid_rows() {
        let rows = 8;
        for length in [0usize, HIDDEN * 2 - 2, HIDDEN * 2 + 2] {
            let error = EmbedRow::new(1, vec![0u8; length], rows)
                .expect_err("a row that is not exactly one HIDDEN BF16 row must be refused");
            assert!(
                matches!(error, StagingError::Bytes { .. }),
                "length {length} must be a byte-count refusal, got {error:?}"
            );
        }
        let error = EmbedRow::new(rows, vec![0u8; HIDDEN * 2], rows)
            .expect_err("a position outside the table must be refused");
        assert!(
            matches!(error, StagingError::Position { position, .. } if position == rows),
            "out-of-range position must be a Position refusal, got {error:?}"
        );
        let empty_table = EmbedRow::new(0, vec![0u8; HIDDEN * 2], 0)
            .expect_err("a zero-row table cannot admit a row");
        assert!(matches!(
            empty_table,
            StagingError::Position { rows: 0, .. }
        ));
        let row =
            EmbedRow::new(1, vec![0x3cu8; HIDDEN * 2], rows).expect("a valid row is admitted");
        assert_eq!(
            row.x_bf16().len(),
            HC_MULT * HIDDEN * 2,
            "the admitted row expands to exactly HC_MULT copies"
        );
    }

    #[derive(Default)]
    struct FakeSource {
        specs: BTreeMap<String, TensorSpec>,
        data: BTreeMap<String, Vec<u8>>,
        paged: bool,
        whole_reads: Mutex<Vec<String>>,
    }

    impl FakeSource {
        fn add(&mut self, name: &str, dtype: DType, shape: Vec<usize>, data: Vec<u8>) {
            self.specs.insert(
                name.to_string(),
                TensorSpec {
                    name: name.to_string(),
                    dtype,
                    shape,
                },
            );
            self.data.insert(name.to_string(), data);
        }

        fn layer_zero(&mut self, positions: usize) -> Self {
            self.paged = true;
            let mut fn_values = vec![0.0f32; PREDICTOR_ROWS * HC_MULT * HIDDEN];
            for (index, value) in fn_values.iter_mut().enumerate() {
                *value = (index % 97) as f32 * 0.001;
            }
            let fn_bytes = fn_values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            let base = (0..PREDICTOR_ROWS)
                .map(|value| value as f32 * 0.5)
                .collect::<Vec<_>>();
            let base_bytes = base.iter().flat_map(|value| value.to_le_bytes()).collect();
            let scale = [0.25f32, 0.5, 0.75];
            let scale_bytes = scale.iter().flat_map(|value| value.to_le_bytes()).collect();
            let norm = (0..HIDDEN)
                .map(|value| bf16_bits(value as f32 / 10.0))
                .collect::<Vec<_>>();
            let norm_bytes = norm.iter().flat_map(|value| value.to_le_bytes()).collect();
            let embed = (0..positions * HIDDEN)
                .map(|value| bf16_bits(value as f32 / 100.0))
                .collect::<Vec<_>>();
            let embed_bytes = embed.iter().flat_map(|value| value.to_le_bytes()).collect();
            self.add(
                LAYER0_MHC_FN,
                DType::F32,
                vec![PREDICTOR_ROWS, HC_MULT * HIDDEN],
                fn_bytes,
            );
            self.add(
                LAYER0_MHC_BASE,
                DType::F32,
                vec![PREDICTOR_ROWS],
                base_bytes,
            );
            self.add(LAYER0_MHC_SCALE, DType::F32, vec![3], scale_bytes);
            self.add(LAYER0_ATTN_NORM, DType::BF16, vec![HIDDEN], norm_bytes);
            self.add(
                EMBED_WEIGHT,
                DType::BF16,
                vec![positions, HIDDEN],
                embed_bytes,
            );
            Self {
                specs: std::mem::take(&mut self.specs),
                data: std::mem::take(&mut self.data),
                paged: true,
                whole_reads: Mutex::new(Vec::new()),
            }
        }
    }

    impl WeightSource for FakeSource {
        fn names(&self) -> Vec<String> {
            self.specs.keys().cloned().collect()
        }

        fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
            self.specs
                .get(name)
                .cloned()
                .ok_or_else(|| WeightError::Missing {
                    name: name.to_string(),
                })
        }

        fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
            self.whole_reads
                .lock()
                .expect("whole-read log")
                .push(name.to_string());
            self.data
                .get(name)
                .cloned()
                .ok_or_else(|| WeightError::Missing {
                    name: name.to_string(),
                })
        }

        fn read_ranges(
            &self,
            name: &str,
            ranges: &[(u64, usize)],
        ) -> Option<Result<Vec<u8>, WeightError>> {
            if !self.paged {
                return None;
            }
            let bytes = self.data.get(name)?;
            let mut out = Vec::new();
            for &(offset, len) in ranges {
                let start = usize::try_from(offset).ok()?;
                out.extend_from_slice(bytes.get(start..start + len)?);
            }
            Some(Ok(out))
        }
    }

    fn source_with_rows(rows: usize) -> FakeSource {
        FakeSource::default().layer_zero(rows)
    }

    #[test]
    fn staging_reads_layer_zero_and_the_declared_rows_verbatim() {
        let source = source_with_rows(4);
        let staging = LayerZeroStaging::from_source(&source, &[0, 3]).expect("stage");
        assert_eq!(staging.predictor.len(), PREDICTOR_ROWS * HC_MULT * HIDDEN);
        let released = bytes_to_f32(
            LAYER0_MHC_FN,
            source.data.get(LAYER0_MHC_FN).expect("fn bytes"),
        )
        .expect("decode fn");
        let mut expected = vec![0.0; released.len()];
        released_predictor_into_kernel_rows(&released, &mut expected);
        assert_eq!(staging.predictor, expected);
        assert_eq!(staging.hc_base.len(), PREDICTOR_ROWS);
        assert_eq!(staging.hc_scale, [0.25, 0.5, 0.75]);
        assert_eq!(staging.attn_norm.len(), HIDDEN);
        assert_eq!(
            staging.hc_base_kernel,
            bias_into_kernel_rows(&staging.hc_base)
        );
        assert_eq!(staging.embed_rows.len(), 2);
        assert_eq!(staging.embed_rows[0].position, 0);
        assert_eq!(staging.embed_rows[1].position, 3);
        assert_eq!(staging.embed_rows[0].bf16.len(), HIDDEN * 2);
        assert_eq!(
            staging.embed_rows[0].x_bf16(),
            staging.embed_rows[0].bf16.repeat(HC_MULT)
        );
        assert_eq!(staging.digests.len(), 4 + 2);
        assert_eq!(staging.buffer_digests.len(), 4 + 2);
        let source_digest = |name: &str| {
            staging
                .digests
                .iter()
                .find(|tensor| tensor.name == name)
                .unwrap_or_else(|| panic!("missing source digest {name}"))
                .sha256
                .clone()
        };
        let buffer_digest = |name: &str| {
            staging
                .buffer_digests
                .iter()
                .find(|tensor| tensor.name == name)
                .unwrap_or_else(|| panic!("missing buffer digest {name}"))
                .sha256
                .clone()
        };
        assert_ne!(
            source_digest(LAYER0_MHC_FN),
            buffer_digest(&format!("buffer predictor <- {LAYER0_MHC_FN}")),
            "the buffer digest must cover the permuted predictor, not the source bytes"
        );
        assert_ne!(
            source_digest(LAYER0_MHC_BASE),
            buffer_digest(&format!("buffer hc_base <- {LAYER0_MHC_BASE}")),
            "the buffer digest must cover the kernel-order base"
        );
        assert_ne!(
            source_digest(LAYER0_ATTN_NORM),
            buffer_digest(&format!("buffer rms_weight <- {LAYER0_ATTN_NORM}")),
            "the buffer digest must cover the f32-widened norm"
        );
        assert_eq!(
            source_digest(LAYER0_MHC_SCALE),
            buffer_digest(&format!("buffer hc_scale <- {LAYER0_MHC_SCALE}")),
            "hc_scale is copied without conversion"
        );
        assert_ne!(
            source_digest(&format!("{EMBED_WEIGHT}[0]")),
            buffer_digest(&format!("buffer x <- {EMBED_WEIGHT}[0]")),
            "the X digest covers the four copies, not one source row"
        );
        let digest_of = |bytes: &[u8]| crate::cubins::hex(&Sha256::digest(bytes));
        assert_eq!(
            buffer_digest(&format!("buffer predictor <- {LAYER0_MHC_FN}")),
            digest_of(&f32_bytes(&staging.predictor)),
            "the predictor buffer digest must be over the permuted host f32 bytes"
        );
        assert_eq!(
            buffer_digest(&format!("buffer hc_scale <- {LAYER0_MHC_SCALE}")),
            digest_of(&f32_bytes(&staging.hc_scale)),
            "the scale buffer digest must be over the host f32 bytes"
        );
        assert_eq!(
            buffer_digest(&format!("buffer hc_base <- {LAYER0_MHC_BASE}")),
            digest_of(&f32_bytes(&staging.hc_base_kernel)),
            "the base buffer digest must be over the kernel-order host f32 bytes"
        );
        assert_eq!(
            buffer_digest(&format!("buffer rms_weight <- {LAYER0_ATTN_NORM}")),
            digest_of(&f32_bytes(&staging.attn_norm)),
            "the norm buffer digest must be over the widened host f32 bytes"
        );
        for row in &staging.embed_rows {
            assert_eq!(
                buffer_digest(&format!("buffer x <- {EMBED_WEIGHT}[{}]", row.position)),
                digest_of(&row.x_bf16()),
                "the X buffer digest must be over the four-copy host bytes"
            );
        }
        assert_eq!(
            staging.host_bytes(),
            (PREDICTOR_ROWS * HC_MULT * HIDDEN * 4
                + PREDICTOR_ROWS * 4
                + 3 * 4
                + HIDDEN * 4
                + PREDICTOR_ROWS * 4
                + 2 * HIDDEN * 2) as u64
        );
        assert_eq!(staging.device_buffer_bytes(), 2_064_352);
        let text = staging.provenance_lines().join("\n");
        assert!(text.contains("no averaging"));
        assert!(text.contains("staged embed row 3"));
        assert!(text.contains("sha256"));
        assert!(text.contains("source tensor"));
        assert!(text.contains("buffer tensor"));
        assert!(
            !source
                .whole_reads
                .lock()
                .expect("whole-read log")
                .iter()
                .any(|name| name == EMBED_WEIGHT),
            "embed.weight must be paged, never read whole"
        );
    }

    #[test]
    fn the_buffer_plan_is_the_exact_sixteen_buffer_layout() {
        let plan = MhcBufferPlan::LAYER_ZERO;
        assert_eq!(plan.sizes().len(), MHC_BUFFER_COUNT);
        assert_eq!(plan.device_bytes(), 2_064_352);
        let offsets = plan.offsets();
        assert_eq!(offsets[0], ("x", 0));
        assert_eq!(offsets[offsets.len() - 1].0, "xhat_normed");
        let mut at = 0usize;
        for (name, offset) in offsets {
            assert_eq!(offset, at, "{name}");
            at += plan
                .sizes()
                .iter()
                .find(|(candidate, _)| *candidate == name)
                .expect("sized")
                .1;
        }
        assert_eq!(at, plan.device_bytes());
        // Every allocation size is a multiple of the driver's 4-byte-or-better alignment.
        assert!(
            plan.sizes()
                .iter()
                .all(|(_, bytes)| *bytes > 0 && bytes % 4 == 0)
        );
    }

    #[test]
    fn a_missing_tensor_is_named_before_a_read() {
        let source = source_with_rows(2);
        let mut source = source;
        source.specs.remove(LAYER0_MHC_SCALE);
        let error = LayerZeroStaging::from_source(&source, &[0]).expect_err("missing");
        assert!(error.to_string().contains("layers.0.hc_attn_scale"));
        assert!(error.to_string().contains("no tensor"));
    }

    #[test]
    fn a_wrong_dtype_or_short_read_is_refused() {
        let mut source = source_with_rows(2);
        let spec = source.specs.get_mut(LAYER0_MHC_BASE).expect("base");
        spec.dtype = DType::BF16;
        let error = LayerZeroStaging::from_source(&source, &[0]).expect_err("dtype");
        assert!(error.to_string().contains("hc_attn_base"));

        let mut source = source_with_rows(2);
        source.data.get_mut(LAYER0_MHC_SCALE).expect("scale").pop();
        let error = LayerZeroStaging::from_source(&source, &[0]).expect_err("short");
        assert!(matches!(error, StagingError::Bytes { .. }));
    }

    #[test]
    fn a_changed_staged_byte_changes_its_digest() {
        let source = source_with_rows(2);
        let first = LayerZeroStaging::from_source(&source, &[0]).expect("first stage");
        let digest = |staging: &LayerZeroStaging| {
            staging
                .digests
                .iter()
                .find(|tensor| tensor.name == LAYER0_MHC_SCALE)
                .expect("scale digest")
                .sha256
                .clone()
        };
        let mut changed = source_with_rows(2);
        changed.data.get_mut(LAYER0_MHC_SCALE).expect("scale")[0] ^= 1;
        let second = LayerZeroStaging::from_source(&changed, &[0]).expect("second stage");
        assert_eq!(digest(&first).len(), 64);
        assert_ne!(digest(&first), digest(&second));
    }

    #[test]
    fn an_out_of_range_row_and_an_unpaged_source_are_refused() {
        let source = source_with_rows(2);
        let error = LayerZeroStaging::from_source(&source, &[2]).expect_err("row range");
        assert!(matches!(
            error,
            StagingError::Position {
                position: 2,
                rows: 2,
                ..
            }
        ));

        let mut source = source_with_rows(2);
        source.paged = false;
        let error = LayerZeroStaging::from_source(&source, &[0]).expect_err("paging");
        assert!(matches!(error, StagingError::EmbedUnpaged { .. }));
        assert!(error.to_string().contains("refusing"));
    }
}

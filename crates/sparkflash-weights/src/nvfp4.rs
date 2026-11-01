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

//! The NVIDIA ModelOpt NVFP4 weight format, and the per-tensor FP8 rows it ships beside.
//!
//! # NVFP4
//!
//! A logical weight `[out, in]` is stored as four tensors under one prefix:
//!
//! * `weight`: `U8 [out, in / 2]`, two E2M1 nibbles per byte, the **low** nibble the even column;
//! * `weight_scale`: `F8_E4M3 [out, in / 16]`, one scale per 16 values along K;
//! * `weight_scale_2`: one `F32` for the whole tensor;
//! * `input_scale`: one `F32` static activation scale. The activation is quantised with it before
//!   the GEMM; a reference dot product applies it itself, which is why [dequantize_nvfp4] exposes
//!   the weight product only.
//!
//! The value is `e2m1(nibble) * e4m3(scale) * scale_2`. ModelOpt stores its E4M3 scales as
//! *multipliers*, not the power-of-two E8M0 form DeepSeek's FP4 uses, so this is a decoder of its
//! own rather than another use of [crate::quant::decode_f4e2m1]: DeepSeek's expert scale grid is
//! `[out, in / 32]` of E8M0, while this format is `[out, in / 16]` of E4M3 plus two per-tensor
//! factors. The nibble itself is the same E2M1 code, decoded by [crate::quant::f4e2m1_to_f32].
//!
//! Every NVFP4 checkpoint this tree serves (Qwen3.8-Flash-Next, GLM-5.3-Flash) uses this layout.
//!
//! # FP8 with one per-tensor scale
//!
//! ModelOpt's FP8 tensors are `F8_E4M3` with a single per-tensor scale, so a value decodes as
//! `e4m3(byte) * scale` - the Engram FP8 row format minus its per-32 E8M0 scales.

use crate::dtype::DType;
use crate::quant::{QuantError, decode_f8e4m3, f4e2m1_to_f32};

/// The number of values one NVFP4 scale covers (ModelOpt `group_size: 16`).
pub const NVFP4_GROUP: usize = 16;

/// The dtype a stored NVFP4 weight reports through the safetensors container.
pub const NVFP4_STORED_DTYPE: DType = DType::U8;

/// Dequantise an NVFP4 weight to f32, row-major `[rows, columns]`.
///
/// The two byte strings are the tensor's packed codes and block scales. `scale_2` is applied to
/// every value, and the block scale to its own 16-column group.
///
/// # Errors
/// [QuantError::Length] when a byte count does not match the shape the caller declared.
pub fn dequantize_nvfp4(
    weight: &[u8],
    scales: &[u8],
    scale_2: f32,
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, QuantError> {
    let packed_columns = columns.div_ceil(2);
    let scale_columns = columns.div_ceil(NVFP4_GROUP);
    // A row is padded to a whole byte, so the codes are sized as 2 * packed_columns per row.
    let codes = rows * packed_columns * 2;
    if weight.len() != rows * packed_columns {
        return Err(QuantError::Length {
            dtype: DType::F4E2M1,
            elements: codes,
            expected: DType::F4E2M1.storage_bytes(codes),
            actual: weight.len(),
        });
    }
    if scales.len() != rows * scale_columns {
        return Err(QuantError::Length {
            dtype: DType::F8E4M3,
            elements: rows * scale_columns,
            expected: rows * scale_columns,
            actual: scales.len(),
        });
    }
    let decoded_scales = decode_f8e4m3(scales);
    let mut out = vec![0f32; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let byte = weight[row * packed_columns + column / 2];
            let code = if column % 2 == 0 {
                byte & 0x0F
            } else {
                byte >> 4
            };
            let scale = decoded_scales[row * scale_columns + column / NVFP4_GROUP];
            out[row * columns + column] = f4e2m1_to_f32(code) * scale * scale_2;
        }
    }
    Ok(out)
}

/// Dequantise an FP8 E4M3 tensor with one per-tensor scale.
///
/// # Errors
/// [QuantError::Length] when the byte count does not match `elements`.
pub fn dequantize_fp8(bytes: &[u8], scale: f32, elements: usize) -> Result<Vec<f32>, QuantError> {
    if bytes.len() != elements {
        return Err(QuantError::Length {
            dtype: DType::F8E4M3,
            elements,
            expected: elements,
            actual: bytes.len(),
        });
    }
    let mut out = decode_f8e4m3(bytes);
    for value in &mut out {
        *value *= scale;
    }
    Ok(out)
}

/// A reference NVFP4 matrix-vector product: `y = W x`, with `W` dequantised on the fly.
///
/// This is the oracle for the device expert kernels, not a fast path.
///
/// # Errors
/// As [dequantize_nvfp4], and [QuantError::Width] when `input` is not `columns` wide.
pub fn matvec_nvfp4(
    weight: &[u8],
    scales: &[u8],
    scale_2: f32,
    rows: usize,
    columns: usize,
    input: &[f32],
) -> Result<Vec<f32>, QuantError> {
    if input.len() != columns {
        return Err(QuantError::Width {
            expected: columns,
            actual: input.len(),
        });
    }
    let dequantised = dequantize_nvfp4(weight, scales, scale_2, rows, columns)?;
    let mut out = vec![0f32; rows];
    for row in 0..rows {
        let mut accumulator = 0f32;
        for column in 0..columns {
            accumulator += dequantised[row * columns + column] * input[column];
        }
        out[row] = accumulator;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_two_by_four_nvfp4_weight_dequantises_by_hand() {
        // Row 0: columns 0..4 = [1.0, -1.0, 0.0, 2.0]; row 1: [0.5, 0.5, -6.0, 0.0].
        // Nibble order is low then high: E2M1 codes are 2 (1.0), 0xA (-1.0), 0 (0.0), 4 (2.0),
        // then 1 (0.5), 1 (0.5), 0xF (-6.0), 0 (0.0).
        let weight = [0xA2u8, 0x40, 0x11, 0x0F];
        // One E4M3 scale of 1.0 per row (0x38 is 1.0 in E4M3).
        let scales = [0x38u8, 0x38];
        let values = dequantize_nvfp4(&weight, &scales, 1.0, 2, 4).expect("the shapes match");
        assert_eq!(values, vec![1.0, -1.0, 0.0, 2.0, 0.5, 0.5, -6.0, 0.0]);
    }

    #[test]
    fn the_per_tensor_scale_multiplies_every_value() {
        let weight = [0x22u8];
        let scales = [0x38u8];
        let values = dequantize_nvfp4(&weight, &scales, 2.0, 1, 2).expect("the shapes match");
        assert_eq!(values, vec![2.0, 2.0]);
    }

    #[test]
    fn a_block_scale_covers_sixteen_columns() {
        // 17 columns: the last column is in the second group, and only it sees scale 0x40 (2.0).
        let mut weight = vec![0u8; 9];
        weight[8] = 0x02; // column 16, low nibble, value 1.0
        weight[0] = 0x02; // column 0, low nibble, value 1.0
        let mut scales = vec![0u8; 2];
        scales[0] = 0x38; // 1.0
        scales[1] = 0x40; // 2.0
        let values = dequantize_nvfp4(&weight, &scales, 1.0, 1, 17).expect("the shapes match");
        assert_eq!(values[0], 1.0);
        assert_eq!(values[16], 2.0);
    }

    #[test]
    fn a_shape_that_does_not_match_its_bytes_is_refused() {
        let error = dequantize_nvfp4(&[0u8; 3], &[0u8; 1], 1.0, 2, 4)
            .expect_err("three bytes cannot hold a 2x4 weight");
        assert!(error.to_string().contains("needs 4 bytes"), "{error}");
        let error = dequantize_nvfp4(&[0u8; 4], &[0u8; 1], 1.0, 2, 4)
            .expect_err("one scale cannot cover two rows");
        assert!(error.to_string().contains("needs 2 bytes"), "{error}");
    }

    #[test]
    fn the_reference_matvec_agrees_with_the_dequantised_product() {
        let weight = [0xA2u8, 0x40, 0x11, 0x0F];
        let scales = [0x38u8, 0x38];
        let out = matvec_nvfp4(&weight, &scales, 1.0, 2, 4, &[1.0, 2.0, 3.0, 4.0])
            .expect("the shapes match");
        assert_eq!(out, vec![1.0 - 2.0 + 0.0 + 8.0, 0.5 + 1.0 - 18.0 + 0.0]);
        assert_eq!(
            matvec_nvfp4(&weight, &scales, 1.0, 2, 4, &[1.0; 3]),
            Err(QuantError::Width {
                expected: 4,
                actual: 3
            })
        );
    }

    #[test]
    fn fp8_rows_apply_the_per_tensor_scale() {
        // 0x38 is 1.0 and 0x3C is 1.5 in E4M3.
        let values = dequantize_fp8(&[0x38u8, 0x3C], 2.0, 2).expect("the shapes match");
        assert_eq!(values, vec![2.0, 3.0]);
    }
}

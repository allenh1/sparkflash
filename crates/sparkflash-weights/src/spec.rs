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

//! The description of one stored tensor.
//!
//! A model plan names tensors; the weight container and every loader only need this small,
//! model-independent record. Keeping it here lets a source describe a tensor without depending on
//! any particular model's layer or parameter plan.

use crate::dtype::DType;

/// One tensor's entry in a plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// The name exactly as the checkpoint spells it.
    pub name: String,
    /// The element format.
    pub dtype: DType,
    /// The logical shape. For an FP4 weight this is the shape the arithmetic sees, twice the last
    /// dimension of the bytes on disk.
    pub shape: Vec<usize>,
}

impl TensorSpec {
    /// The number of logical elements.
    pub fn elements(&self) -> u128 {
        self.shape.iter().map(|&dim| dim as u128).product()
    }

    /// The number of bytes this tensor occupies, packing included.
    pub fn storage_bytes(&self) -> u128 {
        self.dtype.storage_bytes(self.elements() as usize) as u128
    }

    /// Whether `stored`, a tensor as its container reports it, holds this logical spec.
    ///
    /// That is the same dtype and shape; or the same dtype and elements as a contiguous collapse
    /// of the container's leading dimensions into the logical rows. A container reports a 1-D
    /// convolution's weight as `[out, in/groups, kernel]` and a stacked expert block as
    /// `[experts, out, in]`, while the arithmetic sees the `[out, kernel]` and
    /// `[experts * out, in]` matrices: the same values in the same row-major order. The last
    /// dimension must equal the logical columns exactly and the product of every leading
    /// dimension must equal the logical rows exactly, so a reshape that moves the last axis or
    /// changes the element count is a different tensor; and the dtype must agree, so a `U8`
    /// tensor never stands in for one of another format. A unit dimension is the collapse's
    /// degenerate case `[out, 1, kernel] = [out, kernel]`. Or, for an FP4 E2M1 spec, the NVFP4
    /// packing of it: ModelOpt stores the codes as `U8`
    /// ([crate::nvfp4::NVFP4_STORED_DTYPE]) with the same leading product and the last dimension
    /// halved, the same bytes. A container that already reports the logical FP4 view (the
    /// safetensors source does for `I8`) matches as itself. Anything else is a different tensor.
    pub fn is_stored_as(&self, stored: &TensorSpec) -> bool {
        if self.dtype == stored.dtype && self.shape == stored.shape {
            return true;
        }
        let (Some((logical_rows, logical_columns)), Some((stored_rows, stored_columns))) =
            (matrix_of(&self.shape), matrix_of(&stored.shape))
        else {
            return false;
        };
        if self.dtype == stored.dtype {
            return self.elements() == stored.elements()
                && logical_columns == stored_columns
                && logical_rows == stored_rows;
        }
        if self.dtype != DType::F4E2M1 || stored.dtype != crate::nvfp4::NVFP4_STORED_DTYPE {
            return false;
        }
        logical_rows == stored_rows && stored_columns.checked_mul(2) == Some(logical_columns)
    }
}

/// A shape's matrix view: the product of every dimension but the last, and the last.
///
/// A rank-1 shape is one row, and a rank-0 shape has no matrix view.
fn matrix_of(shape: &[usize]) -> Option<(u128, usize)> {
    let (&columns, leading) = shape.split_last()?;
    Some((leading.iter().map(|&dim| dim as u128).product(), columns))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(dtype: DType, shape: &[usize]) -> TensorSpec {
        TensorSpec {
            name: "experts.0.gate_proj.weight".to_owned(),
            dtype,
            shape: shape.to_vec(),
        }
    }

    #[test]
    fn an_fp4_spec_is_stored_as_itself_or_as_modelopt_packs_it() {
        let logical = spec(DType::F4E2M1, &[640, 2560]);
        assert!(logical.is_stored_as(&logical));
        // The released NVFP4 checkpoint: U8, two codes a byte along the last axis.
        assert!(logical.is_stored_as(&spec(DType::U8, &[640, 1280])));
        // Not the packing: the logical width as U8, a halved leading axis, another dtype.
        assert!(!logical.is_stored_as(&spec(DType::U8, &[640, 2560])));
        assert!(!logical.is_stored_as(&spec(DType::U8, &[320, 2560])));
        assert!(!logical.is_stored_as(&spec(DType::F8E4M3, &[640, 1280])));
        // Only FP4 has a packed form: a U8 spec is matched exactly, and so is every other dtype.
        let bytes = spec(DType::U8, &[640, 1280]);
        assert!(bytes.is_stored_as(&bytes));
        assert!(!bytes.is_stored_as(&spec(DType::U8, &[640, 640])));

        // A stacked expert block packs the same way: `[384, 4608, 2304]` U8 holds the
        // `[1769472, 4608]` F4E2M1 view, whose leading product the collapse matches.
        let stacked = spec(DType::F4E2M1, &[1_769_472, 4_608]);
        assert!(stacked.is_stored_as(&spec(DType::U8, &[384, 4_608, 2_304])));
        // The last dimension is exactly half, and the leading product exactly the rows.
        assert!(!stacked.is_stored_as(&spec(DType::U8, &[384, 4_608, 2_303])));
        assert!(!stacked.is_stored_as(&spec(DType::U8, &[383, 4_608, 2_304])));

        let bf16 = spec(DType::BF16, &[640, 2560]);
        assert!(!bf16.is_stored_as(&spec(DType::U8, &[640, 1280])));
    }

    #[test]
    fn a_stacked_expert_block_is_the_matrix_the_arithmetic_sees() {
        // The tiny fixture's routed expert block is `[experts, out, in]`; the boot flattens it
        // to the `[experts * out, in]` matrix the arithmetic sees: `[2, 32, 32]` is `[64, 32]`
        // in row-major order.
        let logical = spec(DType::BF16, &[64, 32]);
        assert!(logical.is_stored_as(&spec(DType::BF16, &[2, 32, 32])));
        // The same stacked form at the released widths: `[384, 2 * 2304, 4608]` is the
        // `[1769472, 4608]` matrix. (The released NVFP4 block is `[384, 4608, 2304]` U8; its
        // last axis is bytes, half the logical columns.)
        assert!(
            spec(DType::BF16, &[1_769_472, 4_608])
                .is_stored_as(&spec(DType::BF16, &[384, 4_608, 4_608]))
        );
        // A 1-D convolution's `[out, in/groups, kernel]` is the same collapse; a unit dimension
        // is its degenerate case, and a non-unit one is no more special.
        assert!(spec(DType::BF16, &[32, 4]).is_stored_as(&spec(DType::BF16, &[32, 1, 4])));
        assert!(spec(DType::BF16, &[8, 32]).is_stored_as(&spec(DType::BF16, &[4, 2, 32])));
        // A unit dimension may also survive in the stored block.
        assert!(logical.is_stored_as(&spec(DType::BF16, &[2, 1, 32, 32])));

        // A reshape that moves the last dimension is a different tensor, even with the same
        // element count: the same bytes would be read in another order.
        assert!(!spec(DType::BF16, &[128, 16]).is_stored_as(&spec(DType::BF16, &[2, 32, 32])));
        // A wrong column, and a wrong leading product, are different tensors.
        assert!(!logical.is_stored_as(&spec(DType::BF16, &[2, 32, 31])));
        assert!(!logical.is_stored_as(&spec(DType::BF16, &[2, 33, 32])));
        // A rank-1 stored shape is one row of 128, not the logical `[32, 4]`.
        assert!(!spec(DType::BF16, &[32, 4]).is_stored_as(&spec(DType::BF16, &[128])));

        // Only the same dtype may stand in, so a U8 or F32 tensor never passes as BF16.
        assert!(!logical.is_stored_as(&spec(DType::U8, &[2, 32, 32])));
        assert!(!logical.is_stored_as(&spec(DType::F32, &[2, 32, 32])));
    }
}

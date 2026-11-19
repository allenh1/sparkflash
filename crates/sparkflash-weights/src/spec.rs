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
    /// That is the same dtype and shape; or the same dtype, elements and logical shape with the
    /// container's unit dimensions dropped: a 1-D convolution's weight is reported as
    /// `[out, in/groups, kernel]`, and when `in/groups` is one the arithmetic sees the
    /// `[out, kernel]` matrix - the same values in the same order, since a unit dimension carries
    /// no stride. Only unit dimensions may vanish, so the squeezed rank and the element count are
    /// the logical ones; and only under the same dtype, so a `U8` tensor never stands in for one
    /// of another format. Or, for an FP4 E2M1 spec, the NVFP4 packing of it: ModelOpt stores the
    /// codes as `U8` ([crate::nvfp4::NVFP4_STORED_DTYPE]) with the last dimension halved, the same
    /// bytes. A container that already reports the logical FP4 view (the safetensors source does
    /// for `I8`) matches as itself. Anything else is a different tensor.
    pub fn is_stored_as(&self, stored: &TensorSpec) -> bool {
        if self.dtype == stored.dtype {
            if self.shape == stored.shape {
                return true;
            }
            // The container's shape without its unit dimensions: the matrix a 1-D convolution's
            // arithmetic sees when its input has one channel. `eq` compares the lengths too, so a
            // squeezed shape of another rank is a different tensor even when the element count
            // matches, and the explicit count keeps that refusal independent of the shape walk.
            let squeezed = stored.shape.iter().copied().filter(|&dim| dim != 1);
            if self.elements() == stored.elements() && self.shape.iter().copied().eq(squeezed) {
                return true;
            }
        }
        if self.dtype != DType::F4E2M1 || stored.dtype != crate::nvfp4::NVFP4_STORED_DTYPE {
            return false;
        }
        match (self.shape.split_last(), stored.shape.split_last()) {
            (Some((&logical, leading)), Some((&packed, stored_leading))) => {
                leading == stored_leading && packed.checked_mul(2) == Some(logical)
            }
            _ => false,
        }
    }
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
        let bf16 = spec(DType::BF16, &[640, 2560]);
        assert!(!bf16.is_stored_as(&spec(DType::U8, &[640, 1280])));
    }

    #[test]
    fn a_unit_dimension_is_the_same_tensor() {
        // The container's 1-D convolution weight is `[out, in/groups, kernel]`; with one input
        // channel the arithmetic sees `[out, kernel]` and the row-major bytes are the same.
        let logical = spec(DType::BF16, &[32, 4]);
        assert!(logical.is_stored_as(&spec(DType::BF16, &[32, 1, 4])));
        // A unit dimension may lead or trail; it carries no stride either way.
        assert!(logical.is_stored_as(&spec(DType::BF16, &[1, 32, 4])));
        assert!(logical.is_stored_as(&spec(DType::BF16, &[32, 4, 1])));

        // Another kernel: 160 elements are not 128.
        assert!(!logical.is_stored_as(&spec(DType::BF16, &[32, 1, 5])));
        // The same element count with a shape of another rank once the unit dimensions are gone.
        assert!(!logical.is_stored_as(&spec(DType::BF16, &[128])));
        // A genuine flattening is not a unit dimension: `[4, 2, 32]` is not the matrix `[8, 32]`.
        assert!(!spec(DType::BF16, &[8, 32]).is_stored_as(&spec(DType::BF16, &[4, 2, 32])));

        // Only the same dtype may stand in, so a U8 or F32 tensor never passes as BF16.
        assert!(!logical.is_stored_as(&spec(DType::U8, &[32, 1, 4])));
        assert!(!logical.is_stored_as(&spec(DType::F32, &[32, 1, 4])));
        // And the FP4 packing rule is unchanged: a U8 spec is not a BF16 one, packed or not.
        assert!(!logical.is_stored_as(&spec(DType::U8, &[32, 2])));
    }
}

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
}

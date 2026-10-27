//! The storage formats in the checkpoint, and the block rules that size their scales.
//!
//! Three facts drive this module and all three are properties of the released checkpoint rather
//! than choices made here (docs/architecture/03-moe.md, table "quantisation"):
//!
//! - Dense Linears and the shared expert are FP8 E4M3 with one UE8M0 scale per 32x32 block, so a
//!   weight of logical shape `[out, in]` carries `[ceil(out/32), ceil(in/32)]` scale bytes.
//! - Routed and MTP experts are FP4 E2M1 packed two values per byte with one scale per output row
//!   per 32 along K, so `[out, in]` carries `[out, ceil(in/32)]` scale bytes - no blocking along
//!   `out`. The stored tensor is `[out, in/2]` int8; this module keeps the LOGICAL shape and lets
//!   [`DType::F4E2M1`] imply the packing, because every consumer wants the logical one.
//! - `wo_a` is stored quantised like any other dense Linear - the index carries
//!   `layers.N.attn.wo_a.scale` for all 40 layers - and the reference dequantises it to BF16 at
//!   conversion time (`convert.py:157-173`, `.bfloat16()` at 173). The plan describes the checkpoint
//!   as stored; the conversion belongs to the loader, not to the plan. `convert.py:164` accepts either
//!   a 32x32 or a 128x128 scale grid for it; the plan uses the 32x32 grid, and that choice is not a
//!   guess - a 128x128 grid would leave the tensor map 1,228,800 bytes short of the released
//!   `total_size` (40 layers of 30,720 bytes), so the byte identity pins it. A header read remains the
//!   direct confirmation.
//!
//! A logical shape plus a [`DType`] is therefore enough to compute the exact byte length the
//! checkpoint stores, which is what lets the tensor plan cross-check itself against the real index.

/// The element format a tensor is stored in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DType {
    /// 32-bit float. Used by `attn_sink`, the `hc_*` coefficient tensors and the gate biases. Not the
    /// Engram gate weights, which are BF16 in storage.
    F32,
    /// 16-bit brain float. Embeddings, norms, the compressor, `wo_a` after conversion, and the
    /// indexer projections the index stores without a scale.
    BF16,
    /// 16-bit float. Not used by this checkpoint; kept so an unknown dtype can be named.
    F16,
    /// FP8 E4M3. Every dense Linear and the shared expert.
    F8E4M3,
    /// FP8 E8M0, the scale format: one unsigned byte holding a power of two exponent.
    F8E8M0,
    /// FP4 E2M1, packed two values per byte. Routed and MTP experts.
    F4E2M1,
}

impl DType {
    /// The number of bytes one element occupies when it is not packed.
    ///
    /// [`DType::F4E2M1`] returns 1 because a packed pair occupies one byte; use
    /// [`DType::storage_bytes`] rather than multiplying this by an element count.
    pub const fn bytes_per_element(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::BF16 | DType::F16 => 2,
            DType::F8E4M3 | DType::F8E8M0 | DType::F4E2M1 => 1,
        }
    }

    /// The number of values one byte holds. Two for FP4, one for everything else.
    pub const fn elements_per_byte(self) -> usize {
        match self {
            DType::F4E2M1 => 2,
            _ => 1,
        }
    }

    /// Whether this format packs more than one value into a byte.
    pub const fn is_packed(self) -> bool {
        matches!(self, DType::F4E2M1)
    }

    /// Whether this format is a scale, not a value.
    pub const fn is_scale(self) -> bool {
        matches!(self, DType::F8E8M0)
    }

    /// The number of bytes needed to store `elements` values of this format.
    ///
    /// Packed formats round up: one FP4 value still costs a whole byte. Multi-byte formats
    /// multiply: four F32 values cost sixteen bytes, not four. A `storage_bytes` that forgets the
    /// multiplication is wrong by 2x or 4x for every non-packed tensor, and the error is invisible
    /// until a byte total is compared against something else - the plan's cross-check against the
    /// released `total_size` is what caught exactly that.
    pub const fn storage_bytes(self, elements: usize) -> usize {
        let units = elements.div_ceil(self.elements_per_byte());
        units * self.bytes_per_element()
    }

    /// The canonical short name, as it appears in a safetensors header or a log line.
    pub const fn name(self) -> &'static str {
        match self {
            DType::F32 => "F32",
            DType::BF16 => "BF16",
            DType::F16 => "F16",
            DType::F8E4M3 => "F8_E4M3",
            DType::F8E8M0 => "F8_E8M0",
            DType::F4E2M1 => "F4_E2M1",
        }
    }
}

/// The K-block width of every scale grid in this checkpoint: one scale per 32 elements.
pub const SCALE_BLOCK: usize = 32;

/// The number of scale values a dense (FP8) weight of shape `[out, in]` carries.
///
/// This is the true 2-D grid `[ceil(out/32), ceil(in/32)]` asserted by the reference FP8 GEMM
/// (kernel.py:293-298).
pub fn dense_scale_shape(out: usize, r#in: usize) -> [usize; 2] {
    [out.div_ceil(SCALE_BLOCK), r#in.div_ceil(SCALE_BLOCK)]
}

/// The number of scale values an expert (FP4) weight of shape `[out, in]` carries.
///
/// This is the row grid `[out, ceil(in/32)]` - one scale per output row per 32 along K, with no
/// blocking along `out` (model.py:219-224, kernel.py:486-499).
pub fn expert_scale_shape(out: usize, r#in: usize) -> [usize; 2] {
    [out, r#in.div_ceil(SCALE_BLOCK)]
}

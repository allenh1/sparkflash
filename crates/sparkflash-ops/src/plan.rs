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

//! The typed vocabulary every op shares: shapes, dtypes, layouts, streams and the plan contract.
//!
//! A plan is built once, at load, from a model's configuration. It owns no device memory and
//! performs no synchronization: OpPlan::kernel and OpPlan::launch are pure functions of the
//! plan, and the caller binds the buffers it already holds. That is what lets a kernel
//! improvement land in every model without a graph runtime or a host round trip.

use crate::error::OpError;

/// A scalar type a kernel reads or writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE binary32.
    F32,
    /// bfloat16.
    Bf16,
    /// E4M3 FN, one byte a value.
    Fp8E4M3,
    /// MXFP8: E4M3 codes with one E8M0 scale each 32 values.
    MxFp8,
    /// NVFP4: E2M1 codes with one E4M3 scale each 16 values and a per-tensor scale.
    NvFp4,
    /// Signed 32-bit integer, for indices and counters.
    I32,
    /// Unsigned 32-bit integer, for token ids.
    U32,
}

impl DType {
    /// The element width in bytes.
    #[must_use]
    pub const fn element_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::Bf16 => 2,
            Self::Fp8E4M3 | Self::MxFp8 | Self::NvFp4 => 1,
        }
    }

    /// The width a row start must obey for the vector loads the stack emits.
    #[must_use]
    pub const fn row_alignment(self) -> usize {
        match self {
            Self::MxFp8 => 32,
            Self::NvFp4 => 16,
            _ => 16,
        }
    }

    /// The name used in diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
            Self::Fp8E4M3 => "e4m3",
            Self::MxFp8 => "mxfp8",
            Self::NvFp4 => "nvfp4",
            Self::I32 => "i32",
            Self::U32 => "u32",
        }
    }
}

/// The arithmetic a GEMM or attention kernel accumulates in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    /// Multiplies and accumulates in f32, as the reference forward does.
    F32,
    /// bf16 operands with an f32 accumulator.
    Bf16,
    /// E4M3 operands with an f32 accumulator.
    Fp8E4M3,
    /// MXFP8 operands with an f32 accumulator.
    MxFp8,
    /// NVFP4 operands with an f32 accumulator.
    NvFp4,
}

impl Precision {
    /// The operand dtype the precision implies.
    #[must_use]
    pub const fn operand(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Bf16 => DType::Bf16,
            Self::Fp8E4M3 => DType::Fp8E4M3,
            Self::MxFp8 => DType::MxFp8,
            Self::NvFp4 => DType::NvFp4,
        }
    }

    /// The name used in diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.operand().name()
    }
}

/// How a matrix's values are laid out in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Consecutive columns, a row at a time.
    RowMajor,
    /// Consecutive rows, a column at a time.
    ColumnMajor,
    /// A row-major matrix with a block scale group every block values.
    BlockScaled {
        /// Values a scale covers, 16 for NVFP4 and 32 for MXFP8.
        block: usize,
    },
}

impl Layout {
    /// The values one scale covers, or one for an unscaled matrix.
    #[must_use]
    pub const fn scale_block(self) -> usize {
        match self {
            Self::RowMajor | Self::ColumnMajor => 1,
            Self::BlockScaled { block } => block,
        }
    }

    /// Whether the layout carries a scale plane beside the values.
    #[must_use]
    pub const fn is_block_scaled(self) -> bool {
        matches!(self, Self::BlockScaled { .. })
    }
}

/// A two-dimensional shape, rows by columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Matrix {
    /// Output rows.
    pub rows: usize,
    /// Values in a row.
    pub columns: usize,
}

impl Matrix {
    /// A shape.
    #[must_use]
    pub const fn new(rows: usize, columns: usize) -> Self {
        Self { rows, columns }
    }

    /// The number of values in the matrix.
    #[must_use]
    pub const fn elements(self) -> usize {
        self.rows * self.columns
    }

    /// Refuse an empty matrix or a multiplication that cannot be addressed.
    ///
    /// # Errors
    /// An empty dimension, or an element count past isize::MAX.
    pub fn validate(self, what: &'static str) -> Result<(), OpError> {
        if self.rows == 0 || self.columns == 0 {
            return Err(OpError::shape(
                what,
                format!("a [{}, {}] matrix is empty", self.rows, self.columns),
            ));
        }
        if self.elements() > isize::MAX as usize {
            return Err(OpError::shape(
                what,
                format!("a [{}, {}] matrix overflows", self.rows, self.columns),
            ));
        }
        Ok(())
    }
}

/// One operand: its shape, dtype and layout, with no pointer attached.
///
/// The pointer is bound at launch, because the same plan serves every session that reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Operand {
    /// The operand's shape.
    pub shape: Matrix,
    /// The scalar type.
    pub dtype: DType,
    /// The memory layout.
    pub layout: Layout,
}

impl Operand {
    /// A row-major operand.
    #[must_use]
    pub const fn row_major(shape: Matrix, dtype: DType) -> Self {
        Self {
            shape,
            dtype,
            layout: Layout::RowMajor,
        }
    }

    /// An operand with an explicit layout.
    #[must_use]
    pub const fn new(shape: Matrix, dtype: DType, layout: Layout) -> Self {
        Self {
            shape,
            dtype,
            layout,
        }
    }

    /// The bytes the values alone occupy.
    #[must_use]
    pub const fn value_bytes(&self) -> usize {
        self.shape.elements() * self.dtype.element_bytes()
    }

    /// Refuse a shape, dtype or layout that cannot be addressed.
    ///
    /// # Errors
    /// As Matrix::validate, a block scale that is not a power of two, or a row that does not end
    /// on a block boundary.
    pub fn validate(&self, what: &'static str) -> Result<(), OpError> {
        self.shape.validate(what)?;
        if let Layout::BlockScaled { block } = self.layout {
            if block == 0 || !block.is_power_of_two() {
                return Err(OpError::shape(
                    what,
                    format!("a block scale of {block} is not a power of two"),
                ));
            }
            if !self.shape.columns.is_multiple_of(block) {
                return Err(OpError::shape(
                    what,
                    format!(
                        "{} columns do not divide into {block}-value scale groups",
                        self.shape.columns
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// The stream a plan launches on.
///
/// A session owns one stream for its whole life, so the number is explicit in the plan rather
/// than discovered from thread state. Stream 0 is the process's default stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct StreamId(pub u64);

impl StreamId {
    /// The process's default stream.
    pub const DEFAULT: Self = Self(0);

    /// Whether this is the default stream.
    #[must_use]
    pub const fn is_default(self) -> bool {
        self.0 == 0
    }
}

/// The kind of operation a plan describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// A dense matrix multiply.
    DenseGemm,
    /// A GEMM over a token batch, tiled once.
    TileBatchedGemm,
    /// A dense GEMM whose K axis is split and reduced.
    SplitKGemm,
    /// A grouped GEMM over routed experts.
    GroupedMoeGemm,
    /// Router logits and top-k selection.
    MoeRouting,
    /// The expert combine.
    ExpertCombine,
    /// Multi-head latent attention.
    MlaAttention,
    /// Grouped-query attention.
    GqaAttention,
    /// Sparse block attention.
    SparseAttention,
    /// CSA2 compressed attention.
    Csa2Attention,
    /// Split-KV decode attention.
    SplitKvDecode,
    /// The Gated DeltaNet scan.
    GdnScan,
    /// The Kimi Delta Attention scan.
    KdaScan,
    /// A normalization epilogue.
    Norm,
    /// A rotary embedding epilogue.
    Rope,
    /// An activation or gated activation epilogue.
    Activation,
    /// An on-the-fly quantization epilogue.
    Quantize,
    /// An embedding or PLE gather.
    EmbeddingGather,
    /// An Engram gather.
    EngramGather,
    /// A one-shot all-reduce.
    OneShotCollective,
    /// A two-shot all-reduce.
    TwoShotCollective,
    /// A reduce-scatter.
    ReduceScatter,
    /// An all-gather.
    AllGather,
    /// An overlapped-halves collective runner.
    OverlappedHalves,
}

impl OpKind {
    /// The stable name used by logs and parity reports.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DenseGemm => "dense_gemm",
            Self::TileBatchedGemm => "tile_batched_gemm",
            Self::SplitKGemm => "split_k_gemm",
            Self::GroupedMoeGemm => "grouped_moe_gemm",
            Self::MoeRouting => "moe_routing",
            Self::ExpertCombine => "expert_combine",
            Self::MlaAttention => "mla_attention",
            Self::GqaAttention => "gqa_attention",
            Self::SparseAttention => "sparse_attention",
            Self::Csa2Attention => "csa2_attention",
            Self::SplitKvDecode => "split_kv_decode",
            Self::GdnScan => "gdn_scan",
            Self::KdaScan => "kda_scan",
            Self::Norm => "norm",
            Self::Rope => "rope",
            Self::Activation => "activation",
            Self::Quantize => "quantize",
            Self::EmbeddingGather => "embedding_gather",
            Self::EngramGather => "engram_gather",
            Self::OneShotCollective => "one_shot_collective",
            Self::TwoShotCollective => "two_shot_collective",
            Self::ReduceScatter => "reduce_scatter",
            Self::AllGather => "all_gather",
            Self::OverlappedHalves => "overlapped_halves",
        }
    }
}

/// A kernel's identity in the one stack: the image it lives in and its exported symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelId {
    /// The image (a cubin or the shared PTX set).
    pub image: crate::images::Image,
    /// The symbol the driver loads by name.
    pub symbol: &'static str,
}

impl KernelId {
    /// A kernel in an explicit image.
    #[must_use]
    pub const fn new(image: crate::images::Image, symbol: &'static str) -> Self {
        Self { image, symbol }
    }
}

/// A launch, fully described: no allocation, no host synchronization, no graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Launch {
    /// The kernel to run.
    pub kernel: KernelId,
    /// The grid, in blocks.
    pub grid: (u32, u32, u32),
    /// The block, in threads.
    pub block: (u32, u32, u32),
    /// Dynamic shared memory in bytes.
    pub shared_bytes: u32,
    /// The stream the launch is ordered on.
    pub stream: StreamId,
}

/// What every op plan provides.
pub trait OpPlan {
    /// The plan's operation.
    fn kind(&self) -> OpKind;

    /// The stream every launch of this plan is ordered on.
    fn stream(&self) -> StreamId;

    /// The kernel the plan launches.
    fn kernel(&self) -> KernelId;

    /// The plan's launch geometry.
    fn launch(&self) -> Launch;

    /// Scratch floats the caller must reserve, now or once at load.
    fn workspace_floats(&self) -> usize;

    /// Refuse an invalid shape, dtype, layout or split before anything reaches the device.
    ///
    /// # Errors
    /// An OpError naming the first invalid part of the plan.
    fn validate(&self) -> Result<(), OpError>;
}

/// Whether count splits into equal, non-empty parts over world ranks.
#[must_use]
pub fn splits_evenly(count: usize, world: usize) -> bool {
    world > 0 && count > 0 && count.is_multiple_of(world)
}

/// A count split into world equal chunks, refusing an uneven split.
///
/// # Errors
/// A zero or non-dividing world.
pub fn even_chunk(count: usize, world: usize) -> Result<usize, OpError> {
    if !splits_evenly(count, world) {
        return Err(OpError::shape(
            "tensor-parallel split",
            format!("{count} does not split evenly over {world} ranks"),
        ));
    }
    Ok(count / world)
}

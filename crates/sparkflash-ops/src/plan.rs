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

/// The pass a kernel implementation belongs to.
///
/// A model can have a prefill body and a decode/token body in different images. The resolver
/// can select between them; a boot check that must prove both passes calls it once for each.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelPhase {
    /// Both the bulk prefill and the token/decode pass.
    Both,
    /// The bulk, many-row prefill pass.
    Prefill,
    /// The token/decode (few-row) pass.
    Decode,
}

/// The set of operand precisions a kernel accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrecisionMask(u8);

impl PrecisionMask {
    /// No precision.
    pub const NONE: Self = Self(0);
    /// Every precision this tree knows.
    pub const ALL: Self = Self(0b1_1111);

    /// The bit for one precision.
    #[must_use]
    pub const fn of(precision: Precision) -> Self {
        Self(match precision {
            Precision::F32 => 0b00001,
            Precision::Bf16 => 0b00010,
            Precision::Fp8E4M3 => 0b00100,
            Precision::MxFp8 => 0b01000,
            Precision::NvFp4 => 0b10000,
        })
    }

    /// This mask with `precision` added.
    #[must_use]
    pub const fn with(self, precision: Precision) -> Self {
        Self(self.0 | Self::of(precision).0)
    }

    /// Whether `precision` is in the set.
    #[must_use]
    pub const fn contains(self, precision: Precision) -> bool {
        self.0 & Self::of(precision).0 != 0
    }

    /// This mask without `precision`.
    #[must_use]
    pub const fn without(self, precision: Precision) -> Self {
        Self(self.0 & !Self::of(precision).0)
    }

    /// The number of precisions in the set.
    #[must_use]
    pub const fn count(self) -> usize {
        self.0.count_ones() as usize
    }
}

/// The geometry one part runs at on this rank, after the tensor-parallel split.
///
/// Zero means "not applicable to this part". The fields are deliberately a flat bag: a
/// [Part] uses only the dimensions its arithmetic has, and a new part can add a check
/// without teaching every other part about it. The execution plan fills this from the
/// released architecture and the [TpPlan](sparkflash_model::TpPlan); the resolver consumes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct PartGeometry {
    /// Residual width.
    pub hidden: usize,
    /// Rows the plan describes (1 for decode; a fixture may use more).
    pub rows: usize,
    /// One expert's or MLP's rank-local intermediate width, padded.
    pub intermediate: usize,
    /// Rank-local attention heads.
    pub heads: usize,
    /// Rank-local key/value heads.
    pub kv_heads: usize,
    /// The per-head width.
    pub head_dim: usize,
    /// The first global attention head this rank holds.
    pub head_offset: usize,
    /// The first global key/value head this rank holds.
    pub kv_head_offset: usize,
    /// Routed experts, replicated.
    pub experts: usize,
    /// Experts a token activates.
    pub top_k: usize,
    /// Context positions the plan may attend.
    pub positions: usize,
    /// Vocabulary rows.
    pub vocab: usize,
    /// A compression or pooling ratio.
    pub ratio: usize,
    /// Residual streams for a hyper-connection.
    pub streams: usize,
    /// Shared experts, when the MoE block has any.
    pub shared_experts: usize,
    /// The compressed vocabulary an Engram table keys on.
    pub compressed_vocab: usize,
    /// The padding token an Engram compressor skips.
    pub pad_id: usize,
    /// Which table of an Engram layer's list this is.
    pub table_index: usize,
    /// Tokens one MTP stage drafts.
    pub block_size: usize,
    /// Vision patch side.
    pub patch_size: usize,
    /// Vision spatial merge factor.
    pub spatial_merge: usize,
    /// The width a vision aligner hands the language model.
    pub out_hidden: usize,
    /// MLA query LoRA rank.
    pub q_lora: usize,
    /// MLA key/value latent width.
    pub kv_lora: usize,
    /// MLA value head width.
    pub v_head: usize,
    /// MLA output groups.
    pub groups: usize,
    /// Sliding window, when the layer has one.
    pub window: usize,
    /// Shared-expert intermediate width, rank-local and padded.
    pub shared: usize,
    /// The tensors parallel world.
    pub world: usize,
    /// This rank in that world.
    pub rank: usize,
}

impl PartGeometry {
    /// A geometry with every field at zero.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            hidden: 0,
            rows: 0,
            intermediate: 0,
            heads: 0,
            kv_heads: 0,
            head_dim: 0,
            head_offset: 0,
            kv_head_offset: 0,
            experts: 0,
            top_k: 0,
            positions: 0,
            vocab: 0,
            ratio: 0,
            streams: 0,
            shared_experts: 0,
            compressed_vocab: 0,
            pad_id: 0,
            table_index: 0,
            block_size: 0,
            patch_size: 0,
            spatial_merge: 0,
            out_hidden: 0,
            q_lora: 0,
            kv_lora: 0,
            v_head: 0,
            groups: 0,
            window: 0,
            shared: 0,
            world: 0,
            rank: 0,
        }
    }
}

/// What shapes a kernel implementation accepts.
///
/// The constraint is checked at boot by [crate::resolver::KernelResolver], before a weight is
/// read. A constraint is intentionally conservative: a field left at its "any" value is not
/// checked, and a kernel that has no proven shape rule says so by leaving [Self::ANY].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShapeConstraint {
    /// Hidden width must divide this. 1 means no constraint.
    pub hidden_multiple: usize,
    /// Vocabulary rows must divide this.
    pub vocab_multiple: usize,
    /// Head count must divide this.
    pub heads_multiple: usize,
    /// Head width must divide this.
    pub head_dim_multiple: usize,
    /// Intermediate width must divide this.
    pub intermediate_multiple: usize,
    /// Expert count must divide this.
    pub experts_multiple: usize,
    /// Residual stream count must divide this.
    pub streams_multiple: usize,
    /// Fewest context positions this kernel supports.
    pub min_positions: usize,
    /// Most context positions this kernel supports; zero means no upper bound.
    pub max_positions: usize,
    /// Smallest compression ratio this kernel supports.
    pub min_ratio: usize,
    /// The GQA group sizes this kernel implements; empty means any.
    pub gqa_groups: &'static [usize],
    /// The operand precisions this kernel accepts.
    pub precisions: PrecisionMask,
}

impl ShapeConstraint {
    /// No constraint; every shape and precision is accepted.
    pub const ANY: Self = Self {
        hidden_multiple: 1,
        vocab_multiple: 1,
        heads_multiple: 1,
        head_dim_multiple: 1,
        intermediate_multiple: 1,
        experts_multiple: 1,
        streams_multiple: 1,
        min_positions: 0,
        max_positions: 0,
        min_ratio: 0,
        gqa_groups: &[],
        precisions: PrecisionMask::ALL,
    };

    /// Require the hidden width to be a multiple of `multiple`.
    #[must_use]
    pub const fn hidden_multiple(mut self, multiple: usize) -> Self {
        self.hidden_multiple = multiple;
        self
    }

    /// Require the vocabulary to be a multiple of `multiple`.
    #[must_use]
    pub const fn vocab_multiple(mut self, multiple: usize) -> Self {
        self.vocab_multiple = multiple;
        self
    }

    /// Require the head count to be a multiple of `multiple`.
    #[must_use]
    pub const fn heads_multiple(mut self, multiple: usize) -> Self {
        self.heads_multiple = multiple;
        self
    }

    /// Require the head width to be a multiple of `multiple`.
    #[must_use]
    pub const fn head_dim_multiple(mut self, multiple: usize) -> Self {
        self.head_dim_multiple = multiple;
        self
    }

    /// Require the intermediate width to be a multiple of `multiple`.
    #[must_use]
    pub const fn intermediate_multiple(mut self, multiple: usize) -> Self {
        self.intermediate_multiple = multiple;
        self
    }

    /// Require the expert count to be a multiple of `multiple`.
    #[must_use]
    pub const fn experts_multiple(mut self, multiple: usize) -> Self {
        self.experts_multiple = multiple;
        self
    }

    /// Require the residual stream count to be a multiple of `multiple`.
    #[must_use]
    pub const fn streams_multiple(mut self, multiple: usize) -> Self {
        self.streams_multiple = multiple;
        self
    }

    /// Restrict the context range.
    #[must_use]
    pub const fn positions(mut self, min: usize, max: usize) -> Self {
        self.min_positions = min;
        self.max_positions = max;
        self
    }

    /// Require a nonzero compression ratio of at least `ratio`.
    #[must_use]
    pub const fn min_ratio(mut self, ratio: usize) -> Self {
        self.min_ratio = ratio;
        self
    }

    /// Restrict grouped-query attention to these group sizes.
    #[must_use]
    pub const fn gqa_groups(mut self, groups: &'static [usize]) -> Self {
        self.gqa_groups = groups;
        self
    }

    /// Restrict the operand precisions.
    #[must_use]
    pub const fn precisions(mut self, precisions: PrecisionMask) -> Self {
        self.precisions = precisions;
        self
    }

    /// Whether `geometry` at `precision` satisfies this constraint.
    ///
    /// # Errors
    /// A message naming the first field that does not match.
    pub fn accepts(&self, geometry: &PartGeometry, precision: Precision) -> Result<(), String> {
        fn multiple(name: &str, value: usize, divisor: usize) -> Result<(), String> {
            if divisor > 1 && value != 0 && !value.is_multiple_of(divisor) {
                return Err(format!("{name} {value} is not a multiple of {divisor}"));
            }
            Ok(())
        }
        multiple("hidden", geometry.hidden, self.hidden_multiple)?;
        multiple("vocab", geometry.vocab, self.vocab_multiple)?;
        multiple("heads", geometry.heads, self.heads_multiple)?;
        multiple("head width", geometry.head_dim, self.head_dim_multiple)?;
        multiple(
            "intermediate",
            geometry.intermediate,
            self.intermediate_multiple,
        )?;
        multiple("experts", geometry.experts, self.experts_multiple)?;
        multiple("streams", geometry.streams, self.streams_multiple)?;
        if self.min_positions > 0 && geometry.positions < self.min_positions {
            return Err(format!(
                "positions {} are below the kernel's {}",
                geometry.positions, self.min_positions
            ));
        }
        if self.max_positions > 0 && geometry.positions > self.max_positions {
            return Err(format!(
                "positions {} exceed the kernel's {}",
                geometry.positions, self.max_positions
            ));
        }
        if self.min_ratio > 0 && geometry.ratio < self.min_ratio {
            return Err(format!(
                "ratio {} is below the kernel's {}",
                geometry.ratio, self.min_ratio
            ));
        }
        if !self.gqa_groups.is_empty() {
            if geometry.kv_heads == 0 || geometry.heads == 0 {
                return Err("grouped attention has no heads".to_owned());
            }
            if !geometry.heads.is_multiple_of(geometry.kv_heads) {
                return Err(format!(
                    "{} query heads do not group over {} key/value heads",
                    geometry.heads, geometry.kv_heads
                ));
            }
            let group = geometry.heads / geometry.kv_heads;
            if !self.gqa_groups.contains(&group) {
                return Err(format!(
                    "group size {group} is not implemented (got {:?})",
                    self.gqa_groups
                ));
            }
        }
        if !self.precisions.contains(precision) {
            return Err(format!("precision {} is not accepted", precision.name()));
        }
        Ok(())
    }
}

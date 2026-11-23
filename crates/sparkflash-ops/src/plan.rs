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

use sparkflash_model::Target;

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

/// A kernel's identity in the one stack: the target, the image it lives in and its exported
/// symbol.
///
/// The target is part of the identity rather than implied by a plan constructor, so a phase-4
/// [EngineBuilder](crate::engine_plan::EngineBuilder) can prove that the kernel it is about to
/// load was resolved for the device it opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelId {
    /// The device target the kernel was compiled for.
    pub target: Target,
    /// The image (a cubin or the shared PTX set).
    pub image: crate::images::Image,
    /// The symbol the driver loads by name.
    pub symbol: &'static str,
}

impl KernelId {
    /// A kernel in an explicit image on an explicit target.
    #[must_use]
    pub const fn new(target: Target, image: crate::images::Image, symbol: &'static str) -> Self {
        Self {
            target,
            image,
            symbol,
        }
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
/// A model can have a prefill body and a decode/token body in different images. The phase is a
/// property of the symbol, not of the image: one image can export a prefill body and a decode
/// body under different names. A boot check that must prove both passes resolves each symbol
/// once for each.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelPhase {
    /// Both the bulk prefill and the token/decode pass.
    Both,
    /// The bulk, many-row prefill pass.
    Prefill,
    /// The token/decode (few-row) pass.
    Decode,
}

impl KernelPhase {
    /// The pass a launch of `rows` rows belongs to.
    ///
    /// A one-row (or empty) launch is the token/decode pass; a wider launch is bulk prefill.
    #[must_use]
    pub const fn for_rows(rows: usize) -> Self {
        if rows <= 1 {
            Self::Decode
        } else {
            Self::Prefill
        }
    }

    /// Whether a symbol of this phase may serve a request for `requested`.
    #[must_use]
    pub const fn serves(self, requested: Self) -> bool {
        matches!(requested, Self::Both)
            || matches!(self, Self::Both)
            || matches!(
                (self, requested),
                (Self::Prefill, Self::Prefill) | (Self::Decode, Self::Decode)
            )
    }

    /// The stable name used in diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Both => "both",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

/// Which compute unit a kernel body runs on.
///
/// The stack chooses the body, so the class is a property of the symbol and not of the arm that
/// happens to call it. [Self::TensorCore] is a body multiply-accumulating on the MMA tile;
/// [Self::Row] is a dot-product body over a few token rows (a row GEMV or matvec);
/// [Self::Elementwise] is a body that reads each value without a matrix product. A GEMM-class
/// part needs a [Self::TensorCore] body in every pass it accepts, or a [crate::parts::WAIVERS]
/// line that names the gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeClass {
    /// The body multiply-accumulates on the tensor-core tile.
    TensorCore,
    /// The body is a row dot product (a GEMV, a matvec, or a few-row tile that is not MMA).
    Row,
    /// The body is elementwise, a reduction, a gather or a quantiser.
    Elementwise,
}

impl ComputeClass {
    /// Whether this body is a matrix product the catalogue can route by row count.
    #[must_use]
    pub const fn is_gemm(self) -> bool {
        matches!(self, Self::TensorCore | Self::Row)
    }

    /// The stable name used in diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::TensorCore => "tensor_core",
            Self::Row => "row",
            Self::Elementwise => "elementwise",
        }
    }
}

/// The token-row band a kernel body is written for.
///
/// A row GEMV reads at most [crate::gemv::GEMV_MAX_BATCH] rows a launch, a bulk tile starts at
/// the arm's row threshold, and an elementwise body usually serves every row count. The band is
/// half of what lets the stack answer "is there a tensor-core body for this pass?" without
/// knowing which arm is asking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RowBand {
    /// The fewest token rows the body takes.
    pub first: usize,
    /// The most token rows the body takes, inclusive.
    pub last: usize,
}

impl RowBand {
    /// Every positive row count.
    pub const ALL: Self = Self {
        first: 1,
        last: usize::MAX,
    };

    /// A band from `first` to `last`, inclusive.
    #[must_use]
    pub const fn new(first: usize, last: usize) -> Self {
        Self { first, last }
    }

    /// The bands from `first` rows up.
    #[must_use]
    pub const fn from(first: usize) -> Self {
        Self {
            first,
            last: usize::MAX,
        }
    }

    /// The bands of at most `last` rows.
    #[must_use]
    pub const fn at_most(last: usize) -> Self {
        Self { first: 1, last }
    }

    /// Whether `rows` token rows fall in the band.
    #[must_use]
    pub const fn contains(self, rows: usize) -> bool {
        rows >= self.first && rows <= self.last
    }
}

/// One exported kernel symbol and the arithmetic it accepts.
///
/// Precision, phase, compute class and the row band are per symbol, not per image or per part. An
/// image can export a bf16 GEMM beside an NVFP4 quantiser, and a part can have a prefill body and
/// a decode body in different images. The resolver selects a symbol only when this symbol's own
/// mask contains the requested precision, this symbol's own phase serves the requested pass, and
/// this symbol's own row band contains the launch's rows; there is no fallback to "the image
/// exports something".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelSymbol {
    /// The name the driver loads.
    pub name: &'static str,
    /// The operand precisions this symbol's arithmetic accepts.
    pub precisions: PrecisionMask,
    /// The pass this symbol belongs to.
    pub phase: KernelPhase,
    /// The compute unit this body runs on.
    pub class: ComputeClass,
    /// The token-row band this body is written for.
    pub rows: RowBand,
    /// The shapes this body's own arithmetic accepts, beyond its entry's rule.
    pub shape: ShapeConstraint,
}

impl KernelSymbol {
    /// A symbol accepting exactly `precisions` in both passes.
    #[must_use]
    pub const fn both(
        name: &'static str,
        precisions: &[Precision],
        class: ComputeClass,
        rows: RowBand,
    ) -> Self {
        Self {
            name,
            precisions: PrecisionMask::from_slice(precisions),
            phase: KernelPhase::Both,
            class,
            rows,
            shape: ShapeConstraint::ANY,
        }
    }

    /// A bulk-prefill symbol accepting exactly `precisions`.
    #[must_use]
    pub const fn prefill(
        name: &'static str,
        precisions: &[Precision],
        class: ComputeClass,
        rows: RowBand,
    ) -> Self {
        Self {
            name,
            precisions: PrecisionMask::from_slice(precisions),
            phase: KernelPhase::Prefill,
            class,
            rows,
            shape: ShapeConstraint::ANY,
        }
    }

    /// A token/decode symbol accepting exactly `precisions`.
    #[must_use]
    pub const fn decode(
        name: &'static str,
        precisions: &[Precision],
        class: ComputeClass,
        rows: RowBand,
    ) -> Self {
        Self {
            name,
            precisions: PrecisionMask::from_slice(precisions),
            phase: KernelPhase::Decode,
            class,
            rows,
            shape: ShapeConstraint::ANY,
        }
    }

    /// Whether this symbol accepts `precision`.
    #[must_use]
    pub const fn accepts(&self, precision: Precision) -> bool {
        self.precisions.contains(precision)
    }

    /// Whether this symbol serves `precision` for `rows` token rows in the requested pass.
    ///
    /// All three properties are the symbol's own: a body does not inherit its image's precision,
    /// its caller's pass or another body's row band.
    #[must_use]
    pub const fn serves(&self, precision: Precision, phase: KernelPhase, rows: usize) -> bool {
        self.accepts(precision) && self.phase.serves(phase) && self.rows.contains(rows)
    }

    /// This symbol with its own shape rule.
    #[must_use]
    pub const fn with_shape(mut self, shape: ShapeConstraint) -> Self {
        self.shape = shape;
        self
    }
}

/// The set of operand precisions a kernel accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrecisionMask(u8);

impl PrecisionMask {
    /// No precision.
    pub const NONE: Self = Self(0);
    /// Every precision this tree knows.
    pub const ALL: Self = Self(0b1_1111);

    /// The mask naming exactly `precisions`.
    #[must_use]
    pub const fn from_slice(precisions: &[Precision]) -> Self {
        let mut mask = Self::NONE;
        let mut index = 0;
        while index < precisions.len() {
            mask = mask.with(precisions[index]);
            index += 1;
        }
        mask
    }

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

/// The grouped FP8 body an arm's host launches once a batch reaches its grouped threshold.
///
/// The bodies are not interchangeable: Qwen's grouped tile takes whole 128-column scale
/// tiles and its own slot layout, while GLM's block MMA serves three activation forms. Naming
/// the body, rather than inferring it from a family string or the hidden width, keeps the
/// plan's symbol priority the host's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupedFp8Body {
    /// Qwen's grouped 128-tile over E4M3 weights (`q4p_moe_*_fp8`), with its f32
    /// activations and slot layout.
    Tiled,
    /// GLM's dynamic-activation q8 MMA (`glm_moe_q8_mma`): activations quantised per row,
    /// with the split-activation and BF16-activation block MMAs as its fallbacks.
    QuantizedActivation,
    /// GLM's block MMA over an unquantised BF16 activation
    /// (`glm_moe_fp8b128_mma_bf16a`), selected by `prefill.activation = "bf16"`.
    Bf16Activation,
    /// GLM's block MMA over a split bf16 activation (`glm_moe_fp8b128_mma`), the
    /// `prefill.activation = "split"` body.
    SplitActivation,
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
    /// The width a projection writes, when the part is a matrix projection.
    pub output: usize,
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
    /// Tokens in a batch from which the FP8 host takes its grouped body instead of the token
    /// experts. Zero means the plan applies no token threshold and chooses the pass from the
    /// token count alone.
    ///
    /// The threshold is in tokens, not routed rows: Qwen's host compares its batch against
    /// `GEMM_MIN_BATCH` (64 tokens), while GLM's counts token rows against `MAX_ROWS` (8)
    /// and groups from 9 tokens, so a GLM plan sets this to nine.
    pub grouped_min_tokens: usize,
    /// Which grouped FP8 body the host launches at or above that threshold.
    ///
    /// `None` means the plan has not chosen one. An FP8 grouped GEMM refuses such a
    /// requirement rather than inheriting another arm's body; [EnginePlan](crate::EnginePlan)
    /// fills it from the released formats with the released dynamic-activation default. A
    /// serving profile that overrides the activation form (`bf16` or `split`) is not read by
    /// [EnginePlan](crate::EnginePlan): the caller constructing the grouped requirement must
    /// replace this with [GroupedFp8Body::Bf16Activation] or
    /// [GroupedFp8Body::SplitActivation] itself.
    pub grouped_fp8_body: Option<GroupedFp8Body>,
    /// Image channels a vision patch embeds.
    pub in_channels: usize,
    /// Frames one vision patch covers.
    pub temporal_patch: usize,
    /// A vision norm's epsilon, as f64 bits; zero means unknown.
    pub norm_epsilon_bits: u64,
    /// A vision norm's kind, encoded by VisionNorm::code.
    pub vision_norm: u8,
    /// A vision block's activation, encoded by VisionActivation::code.
    pub vision_activation: u8,
    /// Whether a vision attention's projections carry a bias.
    pub attention_bias: bool,
    /// Whether vision queries and keys take a per-head norm; None means unknown.
    pub qk_norm: Option<bool>,
    /// A vision tower's position scheme, encoded by VisionPosition::code.
    pub vision_position: u8,
    /// A vision tower's downsample, encoded by VisionDownsample::code.
    pub vision_downsample: u8,
    /// A vision merger MLP's width.
    pub merger_intermediate: usize,
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
            output: 0,
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
            grouped_min_tokens: 0,
            grouped_fp8_body: None,
            in_channels: 0,
            temporal_patch: 0,
            norm_epsilon_bits: 0,
            vision_norm: 0,
            vision_activation: 0,
            attention_bias: false,
            qk_norm: None,
            vision_position: 0,
            vision_downsample: 0,
            merger_intermediate: 0,
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
    /// Projection output width must divide this.
    pub output_multiple: usize,
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
    /// Hidden width that must equal this, when named and nonzero.
    pub hidden_equals: Option<usize>,
    /// Vocabulary rows that must equal this, when named and nonzero.
    pub vocab_equals: Option<usize>,
    /// Intermediate width that must equal this, when named and nonzero.
    pub intermediate_equals: Option<usize>,
    /// Expert count that must equal this, when named and nonzero.
    pub experts_equals: Option<usize>,
    /// Active-expert count that must equal this, when named and nonzero.
    pub top_k_equals: Option<usize>,
    /// Head width that must equal this, when named and nonzero.
    pub head_dim_equals: Option<usize>,
    /// Value-head width that must equal this, when named and nonzero.
    pub v_head_equals: Option<usize>,
    /// Most experts this implementation addresses.
    pub max_experts: Option<usize>,
    /// Most active experts this implementation addresses.
    pub max_top_k: Option<usize>,
    /// Most heads this implementation addresses.
    pub max_heads: Option<usize>,
    /// Fewest residual streams this implementation accepts.
    pub min_streams: Option<usize>,
    /// Most residual streams this implementation accepts.
    pub max_streams: Option<usize>,
    /// Fewest convolution/window taps this implementation accepts.
    pub min_window: Option<usize>,
    /// Most convolution/window taps this implementation accepts.
    pub max_window: Option<usize>,
    /// Fewest output groups this implementation accepts.
    pub groups_min: Option<usize>,
    /// Most output groups this implementation accepts.
    pub groups_max: Option<usize>,
    /// Vision patch sides this implementation accepts; empty means any.
    pub patch_sizes: &'static [usize],
    /// Image channels that must equal this, when named.
    pub in_channels_equals: Option<usize>,
    /// The exact vision norm kind code, when named.
    pub vision_norm: Option<u8>,
    /// The exact spatial merge factor, when named.
    pub spatial_merge_equals: Option<usize>,
    /// Frames per patch that must equal this, when named.
    pub temporal_patch_equals: Option<usize>,
    /// The exact norm epsilon (as f64 bits) this implementation hardcodes, when named.
    pub norm_epsilon_bits: Option<u64>,
    /// The vision activation code this implementation carries, when named.
    pub vision_activation: Option<u8>,
    /// Whether the vision attention must carry a bias, when named.
    pub attention_bias: Option<bool>,
    /// Whether the vision attention must carry q/k norms, when named.
    pub qk_norm: Option<bool>,
    /// The vision position code this implementation carries, when named.
    pub vision_position: Option<u8>,
    /// The vision downsample code this implementation carries, when named.
    pub vision_downsample: Option<u8>,
    /// The exact merger MLP width, when named.
    pub merger_intermediate_equals: Option<usize>,
    /// The exact width a vision merger hands the language model, when named.
    pub out_hidden_equals: Option<usize>,
    /// The `(key, value)` head-width template pairs this implementation accepts; empty means any.
    pub head_pairs: &'static [(usize, usize)],
    /// The routed-expert `(hidden, experts, top-k)` shapes this implementation is published at;
    /// empty means any.
    pub routed_shapes: &'static [(usize, usize, usize)],
    /// The tensor-parallel world this implementation requires, when named.
    pub world_equals: Option<usize>,
    /// The smallest tensor-parallel world this implementation addresses.
    pub world_min: Option<usize>,
}

impl ShapeConstraint {
    /// No constraint; every shape and precision is accepted.
    pub const ANY: Self = Self {
        hidden_multiple: 1,
        vocab_multiple: 1,
        heads_multiple: 1,
        head_dim_multiple: 1,
        intermediate_multiple: 1,
        output_multiple: 1,
        experts_multiple: 1,
        streams_multiple: 1,
        min_positions: 0,
        max_positions: 0,
        min_ratio: 0,
        gqa_groups: &[],
        hidden_equals: None,
        vocab_equals: None,
        intermediate_equals: None,
        experts_equals: None,
        top_k_equals: None,
        head_dim_equals: None,
        v_head_equals: None,
        max_experts: None,
        max_top_k: None,
        max_heads: None,
        min_streams: None,
        max_streams: None,
        min_window: None,
        max_window: None,
        groups_min: None,
        groups_max: None,
        patch_sizes: &[],
        in_channels_equals: None,
        vision_norm: None,
        spatial_merge_equals: None,
        temporal_patch_equals: None,
        norm_epsilon_bits: None,
        vision_activation: None,
        attention_bias: None,
        qk_norm: None,
        vision_position: None,
        vision_downsample: None,
        merger_intermediate_equals: None,
        out_hidden_equals: None,
        head_pairs: &[],
        routed_shapes: &[],
        world_equals: None,
        world_min: None,
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

    /// Require the projection output width to be a multiple of `multiple`.
    #[must_use]
    pub const fn output_multiple(mut self, multiple: usize) -> Self {
        self.output_multiple = multiple;
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

    /// Require an exact hidden width (ignored when the geometry leaves it at zero).
    #[must_use]
    pub const fn hidden_equals(mut self, hidden: usize) -> Self {
        self.hidden_equals = Some(hidden);
        self
    }

    /// Require the vocabulary to equal `vocab`.
    #[must_use]
    pub const fn vocab_equals(mut self, vocab: usize) -> Self {
        self.vocab_equals = Some(vocab);
        self
    }

    /// Require an exact intermediate width (ignored when the geometry leaves it at zero).
    #[must_use]
    pub const fn intermediate_equals(mut self, intermediate: usize) -> Self {
        self.intermediate_equals = Some(intermediate);
        self
    }

    /// Require an exact expert count (ignored when the geometry leaves it at zero).
    #[must_use]
    pub const fn experts_equals(mut self, experts: usize) -> Self {
        self.experts_equals = Some(experts);
        self
    }

    /// Require an exact active-expert count (ignored when the geometry leaves it at zero).
    #[must_use]
    pub const fn top_k_equals(mut self, top_k: usize) -> Self {
        self.top_k_equals = Some(top_k);
        self
    }

    /// Require an exact head width (ignored when the geometry leaves it at zero).
    #[must_use]
    pub const fn head_dim_equals(mut self, head_dim: usize) -> Self {
        self.head_dim_equals = Some(head_dim);
        self
    }

    /// Require an exact value-head width (ignored when the geometry leaves it at zero).
    #[must_use]
    pub const fn v_head_equals(mut self, v_head: usize) -> Self {
        self.v_head_equals = Some(v_head);
        self
    }

    /// Cap the expert count this implementation addresses.
    #[must_use]
    pub const fn max_experts(mut self, max: usize) -> Self {
        self.max_experts = Some(max);
        self
    }

    /// Cap the active-expert count this implementation addresses.
    #[must_use]
    pub const fn max_top_k(mut self, max: usize) -> Self {
        self.max_top_k = Some(max);
        self
    }

    /// Cap the head count this implementation addresses.
    #[must_use]
    pub const fn max_heads(mut self, max: usize) -> Self {
        self.max_heads = Some(max);
        self
    }

    /// Restrict the residual stream count to `min..=max`.
    #[must_use]
    pub const fn streams_between(mut self, min: usize, max: usize) -> Self {
        self.min_streams = Some(min);
        self.max_streams = Some(max);
        self
    }

    /// Restrict the convolution/window taps to `min..=max`.
    #[must_use]
    pub const fn window_between(mut self, min: usize, max: usize) -> Self {
        self.min_window = Some(min);
        self.max_window = Some(max);
        self
    }

    /// Restrict the output-group count to `min..=max`.
    #[must_use]
    pub const fn groups_between(mut self, min: usize, max: usize) -> Self {
        self.groups_min = Some(min);
        self.groups_max = Some(max);
        self
    }

    /// Restrict the vision patch side to these values.
    #[must_use]
    pub const fn patch_sizes(mut self, sides: &'static [usize]) -> Self {
        self.patch_sizes = sides;
        self
    }

    /// Restrict the scan to these `(key, value)` head-width pairs.
    #[must_use]
    pub const fn head_pairs(mut self, pairs: &'static [(usize, usize)]) -> Self {
        self.head_pairs = pairs;
        self
    }

    /// Restrict the routed experts to these `(hidden, experts, top-k)` shapes.
    #[must_use]
    pub const fn routed_shapes(mut self, shapes: &'static [(usize, usize, usize)]) -> Self {
        self.routed_shapes = shapes;
        self
    }

    /// Require an exact image channel count.
    #[must_use]
    pub const fn in_channels_equals(mut self, channels: usize) -> Self {
        self.in_channels_equals = Some(channels);
        self
    }

    /// Require an exact temporal patch size.
    #[must_use]
    pub const fn temporal_patch_equals(mut self, frames: usize) -> Self {
        self.temporal_patch_equals = Some(frames);
        self
    }

    /// Require an exact vision norm epsilon, compared as f64 bits.
    #[must_use]
    pub const fn norm_epsilon_bits(mut self, bits: u64) -> Self {
        self.norm_epsilon_bits = Some(bits);
        self
    }

    /// Require a vision norm kind code.
    #[must_use]
    pub const fn vision_norm(mut self, code: u8) -> Self {
        self.vision_norm = Some(code);
        self
    }

    /// Require an exact spatial merge factor.
    #[must_use]
    pub const fn spatial_merge_equals(mut self, merge: usize) -> Self {
        self.spatial_merge_equals = Some(merge);
        self
    }

    /// Require a vision activation code.
    #[must_use]
    pub const fn vision_activation(mut self, code: u8) -> Self {
        self.vision_activation = Some(code);
        self
    }

    /// Require a vision attention bias.
    #[must_use]
    pub const fn attention_bias(mut self, bias: bool) -> Self {
        self.attention_bias = Some(bias);
        self
    }

    /// Require a vision q/k norm.
    #[must_use]
    pub const fn qk_norm(mut self, norm: bool) -> Self {
        self.qk_norm = Some(norm);
        self
    }

    /// Require a vision position scheme code.
    #[must_use]
    pub const fn vision_position(mut self, code: u8) -> Self {
        self.vision_position = Some(code);
        self
    }

    /// Require a vision downsample code.
    #[must_use]
    pub const fn vision_downsample(mut self, code: u8) -> Self {
        self.vision_downsample = Some(code);
        self
    }

    /// Require an exact merger MLP width.
    #[must_use]
    pub const fn merger_intermediate_equals(mut self, width: usize) -> Self {
        self.merger_intermediate_equals = Some(width);
        self
    }

    /// Require an exact merger output width.
    #[must_use]
    pub const fn out_hidden_equals(mut self, width: usize) -> Self {
        self.out_hidden_equals = Some(width);
        self
    }

    /// Require an exact tensor-parallel world.
    #[must_use]
    pub const fn world_equals(mut self, world: usize) -> Self {
        self.world_equals = Some(world);
        self
    }

    /// Require a tensor-parallel world of at least `world`.
    #[must_use]
    pub const fn world_min(mut self, world: usize) -> Self {
        self.world_min = Some(world);
        self
    }

    /// Whether `geometry` satisfies this constraint.
    ///
    /// A constraint that names a field is a proof obligation, not a hint: if the geometry leaves
    /// that field at zero (the value a plan that does not know the dimension carries), the entry
    /// is refused. This is what makes a wrong image unrepresentable rather than merely
    /// improbable: a specific entry cannot win on a plan that did not populate the field the
    /// entry's kernel was compiled for.
    ///
    /// # Errors
    /// A message naming the first field that does not match, or is unknown.
    pub fn accepts(&self, geometry: &PartGeometry) -> Result<(), String> {
        fn multiple(name: &str, value: usize, divisor: usize) -> Result<(), String> {
            if divisor <= 1 {
                return Ok(());
            }
            if value == 0 {
                return Err(format!(
                    "{name} is unknown (zero) but the kernel needs a multiple of {divisor}"
                ));
            }
            if !value.is_multiple_of(divisor) {
                return Err(format!("{name} {value} is not a multiple of {divisor}"));
            }
            Ok(())
        }
        fn equals(name: &str, value: usize, expected: Option<usize>) -> Result<(), String> {
            if let Some(expected) = expected
                && value != expected
            {
                return Err(format!("{name} {value} does not equal {expected}"));
            }
            Ok(())
        }
        fn at_most(name: &str, value: usize, max: Option<usize>) -> Result<(), String> {
            if let Some(max) = max {
                if value == 0 {
                    return Err(format!(
                        "{name} is unknown (zero) but the kernel caps it at {max}"
                    ));
                }
                if value > max {
                    return Err(format!("{name} {value} exceeds the kernel's {max}"));
                }
            }
            Ok(())
        }
        fn at_least(name: &str, value: usize, min: Option<usize>) -> Result<(), String> {
            if let Some(min) = min {
                if min > 0 && value == 0 {
                    return Err(format!(
                        "{name} is unknown (zero) but the kernel needs at least {min}"
                    ));
                }
                if value < min {
                    return Err(format!("{name} {value} is below the kernel's {min}"));
                }
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
        multiple("output width", geometry.output, self.output_multiple)?;
        multiple("experts", geometry.experts, self.experts_multiple)?;
        multiple("streams", geometry.streams, self.streams_multiple)?;
        equals("hidden", geometry.hidden, self.hidden_equals)?;
        equals("vocab", geometry.vocab, self.vocab_equals)?;
        equals(
            "intermediate",
            geometry.intermediate,
            self.intermediate_equals,
        )?;
        equals("experts", geometry.experts, self.experts_equals)?;
        equals("top-k", geometry.top_k, self.top_k_equals)?;
        equals("head width", geometry.head_dim, self.head_dim_equals)?;
        equals("value head width", geometry.v_head, self.v_head_equals)?;
        at_most("experts", geometry.experts, self.max_experts)?;
        at_most("top-k", geometry.top_k, self.max_top_k)?;
        at_most("heads", geometry.heads, self.max_heads)?;
        at_least("streams", geometry.streams, self.min_streams)?;
        at_most("streams", geometry.streams, self.max_streams)?;
        at_least("window", geometry.window, self.min_window)?;
        at_most("window", geometry.window, self.max_window)?;
        at_least("output groups", geometry.groups, self.groups_min)?;
        if let Some(max) = self.groups_max
            && geometry.groups > max
        {
            return Err(format!(
                "output groups {} exceed the kernel's {max}",
                geometry.groups
            ));
        }
        if !self.patch_sizes.is_empty() {
            if geometry.patch_size == 0 {
                return Err(format!(
                    "vision patch is unknown (zero) but the kernel implements {:?}",
                    self.patch_sizes
                ));
            }
            if !self.patch_sizes.contains(&geometry.patch_size) {
                return Err(format!(
                    "vision patch {} is not one of {:?}",
                    geometry.patch_size, self.patch_sizes
                ));
            }
        }
        if let Some(channels) = self.in_channels_equals
            && geometry.in_channels != channels
        {
            return Err(format!(
                "image channels {} do not equal the kernel's {channels}",
                geometry.in_channels
            ));
        }
        if let Some(frames) = self.temporal_patch_equals
            && geometry.temporal_patch != frames
        {
            return Err(format!(
                "temporal patch {} does not equal the kernel's {frames}",
                geometry.temporal_patch
            ));
        }
        if let Some(bits) = self.norm_epsilon_bits
            && geometry.norm_epsilon_bits != bits
        {
            return Err(format!(
                "norm epsilon bits {:#x} do not equal the kernel's {bits:#x}",
                geometry.norm_epsilon_bits
            ));
        }
        if let Some(code) = self.vision_norm
            && geometry.vision_norm != code
        {
            return Err(format!(
                "vision norm {} does not equal the kernel's {code}",
                geometry.vision_norm
            ));
        }
        if let Some(merge) = self.spatial_merge_equals
            && geometry.spatial_merge != merge
        {
            return Err(format!(
                "spatial merge {} does not equal the kernel's {merge}",
                geometry.spatial_merge
            ));
        }
        if let Some(code) = self.vision_activation
            && geometry.vision_activation != code
        {
            return Err(format!(
                "vision activation {} does not equal the kernel's {code}",
                geometry.vision_activation
            ));
        }
        if let Some(bias) = self.attention_bias
            && geometry.attention_bias != bias
        {
            return Err(format!(
                "attention bias {} does not equal the kernel's {bias}",
                geometry.attention_bias
            ));
        }
        if let Some(norm) = self.qk_norm
            && geometry.qk_norm != Some(norm)
        {
            return Err(format!(
                "q/k norm {:?} does not equal the kernel's {norm}",
                geometry.qk_norm
            ));
        }
        if let Some(code) = self.vision_position
            && geometry.vision_position != code
        {
            return Err(format!(
                "vision position {} does not equal the kernel's {code}",
                geometry.vision_position
            ));
        }
        if let Some(code) = self.vision_downsample
            && geometry.vision_downsample != code
        {
            return Err(format!(
                "vision downsample {} does not equal the kernel's {code}",
                geometry.vision_downsample
            ));
        }
        if let Some(width) = self.merger_intermediate_equals
            && geometry.merger_intermediate != width
        {
            return Err(format!(
                "merger intermediate {} does not equal the kernel's {width}",
                geometry.merger_intermediate
            ));
        }
        if let Some(width) = self.out_hidden_equals
            && geometry.out_hidden != width
        {
            return Err(format!(
                "merger output {} does not equal the kernel's {width}",
                geometry.out_hidden
            ));
        }
        if !self.head_pairs.is_empty() {
            if geometry.head_dim == 0 || geometry.v_head == 0 {
                return Err(format!(
                    "head pair is unknown (zero) but the kernel implements {:?}",
                    self.head_pairs
                ));
            }
            if !self
                .head_pairs
                .iter()
                .any(|(key, value)| *key == geometry.head_dim && *value == geometry.v_head)
            {
                return Err(format!(
                    "head pair ({}, {}) is not one of {:?}",
                    geometry.head_dim, geometry.v_head, self.head_pairs
                ));
            }
        }
        if !self.routed_shapes.is_empty() {
            let shape = (geometry.hidden, geometry.experts, geometry.top_k);
            if shape.0 == 0 || shape.1 == 0 || shape.2 == 0 {
                return Err(format!(
                    "routed shape (hidden, experts, top-k) {shape:?} is unknown (zero) but the \
                     kernel is published at {:?}",
                    self.routed_shapes
                ));
            }
            if !self.routed_shapes.contains(&shape) {
                return Err(format!(
                    "routed shape (hidden, experts, top-k) {shape:?} is not one of {:?}",
                    self.routed_shapes
                ));
            }
        }
        if self.min_positions > 0 || self.max_positions > 0 {
            if geometry.positions == 0 {
                return Err("positions are unknown (zero) but the kernel bounds them".to_owned());
            }
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
        }
        if self.min_ratio > 0 {
            if geometry.ratio == 0 {
                return Err("ratio is unknown (zero) but the kernel requires one".to_owned());
            }
            if geometry.ratio < self.min_ratio {
                return Err(format!(
                    "ratio {} is below the kernel's {}",
                    geometry.ratio, self.min_ratio
                ));
            }
        }
        if let Some(world) = self.world_equals
            && geometry.world != world
        {
            return Err(format!(
                "world {} does not equal the kernel's {world}",
                geometry.world
            ));
        }
        if let Some(min) = self.world_min {
            if geometry.world == 0 {
                return Err(format!(
                    "world is unknown (zero) but the kernel needs at least {min}"
                ));
            }
            if geometry.world < min {
                return Err(format!(
                    "world {} is below the kernel's {min}",
                    geometry.world
                ));
            }
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
        Ok(())
    }
}

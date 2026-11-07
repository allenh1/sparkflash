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

//! Fused epilogues and gathers: norms, RoPE, activations, quantization, embeddings and Engram.

use sparkflash_model::Part;

use crate::engine_plan::PartRequirement;
use crate::error::OpError;
use crate::plan::{
    DType, KernelId, KernelPhase, Launch, Layout, Matrix, OpKind, OpPlan, Operand, Precision,
    StreamId,
};
use crate::resolver::PlanContext;

/// The norm a plan applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    /// Root-mean-square normalization.
    RmsNorm,
    /// Layer normalization.
    LayerNorm,
    /// The hyper-connection mix.
    HyperConnection,
    /// The manifold-constrained hyper-connection mix.
    ManifoldHyperConnection,
}

impl NormKind {
    /// The model-side part the norm belongs to.
    #[must_use]
    pub const fn part(self) -> Part {
        match self {
            Self::RmsNorm => Part::RmsNorm,
            Self::LayerNorm => Part::LayerNorm,
            Self::HyperConnection => Part::HyperConnection,
            Self::ManifoldHyperConnection => Part::ManifoldHyperConnection,
        }
    }

    /// The norm's stable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::RmsNorm => "RMSNorm",
            Self::LayerNorm => "LayerNorm",
            Self::HyperConnection => "hyper-connection",
            Self::ManifoldHyperConnection => "manifold hyper-connection",
        }
    }
}

/// A normalization epilogue over rows of a fixed width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormPlan {
    /// The norm.
    pub kind: NormKind,
    /// The rows the launch normalizes.
    pub rows: usize,
    /// The row width.
    pub width: usize,
    /// Input and output dtype.
    pub dtype: DType,
    /// The stream the plan launches on.
    pub stream: StreamId,
    /// The geometry the kernel was resolved at.
    pub geometry: crate::plan::PartGeometry,
    /// The catalogue-selected kernel.
    pub resolved: KernelId,
}

impl NormPlan {
    /// A norm plan built from the requirement's geometry.
    ///
    /// # Errors
    /// A zero dimension, a dtype the norm does not take, a requirement for a different part, or
    /// no catalogued implementation. LayerNorm never falls back to an RMSNorm body.
    pub fn new(
        context: PlanContext,
        requirement: &PartRequirement,
        kind: NormKind,
        rows: usize,
        width: usize,
        dtype: DType,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        if requirement.part != kind.part() {
            return Err(OpError::Binding(format!(
                "{} was built from a {} requirement",
                kind.name(),
                requirement.part
            )));
        }
        if rows == 0 || width == 0 {
            return Err(OpError::shape("norm", "a dimension is zero"));
        }
        if !matches!(dtype, DType::F32 | DType::Bf16) {
            return Err(OpError::Unsupported {
                op: "norm",
                dtype: dtype.name(),
            });
        }
        let precision = if dtype == DType::F32 {
            Precision::F32
        } else {
            Precision::Bf16
        };
        let mut geometry = requirement.geometry;
        geometry.rows = rows;
        geometry.hidden = width;
        let symbols = match kind {
            NormKind::RmsNorm => &["rmsnorm_sm121", "opt_rmsnorm_bf16", "q4e_rmsnorm"] as &[_],
            NormKind::LayerNorm => &["q4e_layer_norm"],
            NormKind::HyperConnection => &[
                "mega_mhc_fused_bf16_sm121",
                "opt_mhc_seam",
                "opt_mhc_post",
                "q4e_hyper_mix",
                "opt_bulk_mhc_tail",
            ],
            NormKind::ManifoldHyperConnection => &["q4e_mhc_in", "q4e_mhc_post"],
        };
        let resolved = context
            .resolve(
                &PartRequirement {
                    part: kind.part(),
                    geometry,
                    precision,
                },
                KernelPhase::for_rows(rows),
                symbols,
            )?
            .id();
        let plan = Self {
            kind,
            rows,
            width,
            dtype,
            stream,
            geometry,
            resolved,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Blocks the launch uses, one row at a time.
    fn blocks(&self) -> u32 {
        self.rows.div_ceil(4) as u32
    }
}

impl OpPlan for NormPlan {
    fn kind(&self) -> OpKind {
        OpKind::Norm
    }

    fn stream(&self) -> StreamId {
        self.stream
    }

    fn kernel(&self) -> KernelId {
        self.resolved
    }

    fn launch(&self) -> Launch {
        Launch {
            kernel: self.kernel(),
            grid: (self.blocks(), 1, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
            stream: self.stream,
        }
    }

    fn workspace_floats(&self) -> usize {
        0
    }

    fn validate(&self) -> Result<(), OpError> {
        if self.rows == 0 || self.width == 0 {
            return Err(OpError::shape("norm", "a dimension is zero"));
        }
        if !matches!(self.dtype, DType::F32 | DType::Bf16) {
            return Err(OpError::Unsupported {
                op: "norm",
                dtype: self.dtype.name(),
            });
        }
        Ok(())
    }
}

/// A rotary embedding epilogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RopePlan {
    /// The rows the launch rotates.
    pub rows: usize,
    /// Heads per row.
    pub heads: usize,
    /// The rotated width per head.
    pub head_dim: usize,
    /// The row stride in floats.
    pub row_stride: usize,
    /// The first position.
    pub position: usize,
    /// The stream the plan launches on.
    pub stream: StreamId,
    /// The geometry the kernel was resolved at.
    pub geometry: crate::plan::PartGeometry,
    /// The catalogue-selected kernel.
    pub resolved: KernelId,
}

impl RopePlan {
    /// A RoPE plan built from the requirement's part and hidden width.
    ///
    /// # Errors
    /// A zero dimension, a head width that is not a whole number of pairs or wider than the row
    /// stride, a requirement for a non-RoPE part, or no catalogue kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: PlanContext,
        requirement: &PartRequirement,
        rows: usize,
        heads: usize,
        head_dim: usize,
        row_stride: usize,
        position: usize,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        if requirement.part != Part::Rope {
            return Err(OpError::Binding(format!(
                "RoPE was built from a {} requirement",
                requirement.part
            )));
        }
        if rows == 0 || heads == 0 || head_dim == 0 {
            return Err(OpError::shape("RoPE", "a dimension is zero"));
        }
        if !head_dim.is_multiple_of(2) {
            return Err(OpError::shape(
                "RoPE",
                format!("a head width of {head_dim} is not a whole number of pairs"),
            ));
        }
        if head_dim > row_stride {
            return Err(OpError::shape(
                "RoPE",
                format!("a row stride of {row_stride} cannot hold {head_dim} values"),
            ));
        }
        let mut geometry = requirement.geometry;
        geometry.rows = rows;
        geometry.heads = heads;
        geometry.head_dim = head_dim;
        // Resolve the plan's own geometry; the schedule's head width is a kernel obligation.
        let requirement = PartRequirement {
            geometry,
            ..*requirement
        };
        let resolved = context
            .resolve(
                &requirement,
                KernelPhase::for_rows(rows),
                &[
                    "q4e_rope",
                    "opt_rope_heads_bf16",
                    "rope_rotate_rows_bf16",
                    "rope_rotate_bf16",
                    "kv_norm_rope_bf16",
                    "opt_rows_inv_rope",
                ],
            )?
            .id();
        let plan = Self {
            rows,
            heads,
            head_dim,
            row_stride,
            position,
            stream,
            geometry,
            resolved,
        };
        plan.validate()?;
        Ok(plan)
    }
}

impl OpPlan for RopePlan {
    fn kind(&self) -> OpKind {
        OpKind::Rope
    }

    fn stream(&self) -> StreamId {
        self.stream
    }

    fn kernel(&self) -> KernelId {
        self.resolved
    }

    fn launch(&self) -> Launch {
        Launch {
            kernel: self.kernel(),
            grid: (self.heads as u32, self.rows as u32, 1),
            block: (128, 1, 1),
            shared_bytes: 0,
            stream: self.stream,
        }
    }

    fn workspace_floats(&self) -> usize {
        0
    }

    fn validate(&self) -> Result<(), OpError> {
        if self.rows == 0 || self.heads == 0 || self.head_dim == 0 {
            return Err(OpError::shape("RoPE", "a dimension is zero"));
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(OpError::shape(
                "RoPE",
                format!(
                    "a head width of {} is not a whole number of pairs",
                    self.head_dim
                ),
            ));
        }
        if self.head_dim > self.row_stride {
            return Err(OpError::shape(
                "RoPE",
                format!(
                    "a row stride of {} cannot hold {} values",
                    self.row_stride, self.head_dim
                ),
            ));
        }
        Ok(())
    }
}

/// The activation a plan applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationKind {
    /// SiLU.
    Silu,
    /// SwiGLU: SiLU(gate) times up.
    SwiGlu,
    /// Sigmoid.
    Sigmoid,
    /// Tanh.
    Tanh,
}

impl ActivationKind {
    /// The activation's stable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Silu => "SiLU",
            Self::SwiGlu => "SwiGLU",
            Self::Sigmoid => "sigmoid",
            Self::Tanh => "tanh",
        }
    }
}

/// A fused activation epilogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationPlan {
    /// The activation.
    pub kind: ActivationKind,
    /// Elements before the gate halves are paired.
    pub rows: usize,
    /// The intermediate width.
    pub width: usize,
    /// The operand precision the kernel was resolved at.
    pub precision: Precision,
    /// The stream the plan launches on.
    pub stream: StreamId,
    /// The geometry the kernel was resolved at.
    pub geometry: crate::plan::PartGeometry,
    /// The catalogue-selected kernel.
    pub resolved: KernelId,
}

impl ActivationPlan {
    /// An activation plan built from the requirement's geometry.
    ///
    /// # Errors
    /// A zero dimension, a requirement for a non-activation part, or no catalogue kernel.
    pub fn new(
        context: PlanContext,
        requirement: &PartRequirement,
        kind: ActivationKind,
        rows: usize,
        width: usize,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        if requirement.part != Part::Activation {
            return Err(OpError::Binding(format!(
                "an activation was built from a {} requirement",
                requirement.part
            )));
        }
        if rows == 0 || width == 0 {
            return Err(OpError::shape("activation", "a dimension is zero"));
        }
        let mut geometry = requirement.geometry;
        geometry.rows = rows;
        geometry.intermediate = width;
        let symbols: &[&'static str] = match kind {
            ActivationKind::SwiGlu | ActivationKind::Silu => &[
                "swiglu_mxfp8_quant_sm121",
                "q4e_swiglu",
                "glm_moe_swiglu",
                "q4e_swiglu_limited",
                "glm_moe_swiglu_q8",
                // The MoE pipeline's fused FP8 gated activation: a Qwen FP8 chunk's gate/up
                // output is E4M3 at the tile, so its SwiGLU is this body, not a BF16 epilogue.
                "q4p_moe_swiglu_fp8",
                "shared_swiglu_f32",
                "opt_rows_shared_swiglu",
                "q4p_epilogue",
            ],
            ActivationKind::Sigmoid | ActivationKind::Tanh => {
                &["q4p_epilogue", "q4e_swiglu", "glm_moe_swiglu"]
            }
        };
        // Resolve the plan's own geometry (its row count and intermediate width), not the
        // requirement's original shape.
        let requirement = PartRequirement {
            geometry,
            ..*requirement
        };
        let resolved = context.resolve(&requirement, KernelPhase::for_rows(rows), symbols)?;
        let plan = Self {
            kind,
            rows,
            width,
            precision: resolved.precision,
            stream,
            geometry,
            resolved: resolved.id(),
        };
        plan.validate()?;
        Ok(plan)
    }
}

impl OpPlan for ActivationPlan {
    fn kind(&self) -> OpKind {
        OpKind::Activation
    }

    fn stream(&self) -> StreamId {
        self.stream
    }

    fn kernel(&self) -> KernelId {
        self.resolved
    }

    fn launch(&self) -> Launch {
        Launch {
            kernel: self.kernel(),
            grid: (self.rows.div_ceil(8) as u32, 1, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
            stream: self.stream,
        }
    }

    fn workspace_floats(&self) -> usize {
        match self.kind {
            ActivationKind::SwiGlu => self.rows * self.width,
            _ => 0,
        }
    }

    fn validate(&self) -> Result<(), OpError> {
        if self.rows == 0 || self.width == 0 {
            return Err(OpError::shape("activation", "a dimension is zero"));
        }
        Ok(())
    }
}

/// An on-the-fly quantization epilogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizePlan {
    /// The output matrix.
    pub output: Matrix,
    /// The input dtype.
    pub input: DType,
    /// The output dtype.
    pub output_dtype: DType,
    /// The output layout.
    pub layout: Layout,
    /// The precision of the operand being quantized.
    pub precision: Precision,
    /// The stream the plan launches on.
    pub stream: StreamId,
    /// The geometry the kernel was resolved at.
    pub geometry: crate::plan::PartGeometry,
    /// The catalogue-selected kernel.
    pub resolved: KernelId,
}

impl QuantizePlan {
    /// A quantization plan built from the requirement's geometry.
    ///
    /// The precision that selects the symbol is the input's, not the output's: a bf16 row is
    /// quantised by a body that takes bf16, and the float-only 128-block quantiser cannot stand
    /// in for it.
    ///
    /// # Errors
    /// A zero dimension, a block-scaled layout whose rows do not divide, an unsupported input or
    /// output dtype, a requirement for a non-quantization part, or no catalogue kernel.
    pub fn new(
        context: PlanContext,
        requirement: &PartRequirement,
        output: Matrix,
        input: DType,
        output_dtype: DType,
        layout: Layout,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        if requirement.part != Part::Quantize {
            return Err(OpError::Binding(format!(
                "a quantizer was built from a {} requirement",
                requirement.part
            )));
        }
        Operand::new(output, output_dtype, layout).validate("quantization output")?;
        if !matches!(output_dtype, DType::NvFp4 | DType::Fp8E4M3 | DType::MxFp8) {
            return Err(OpError::Unsupported {
                op: "quantization",
                dtype: output_dtype.name(),
            });
        }
        let precision = match input {
            DType::F32 => Precision::F32,
            DType::Bf16 => Precision::Bf16,
            DType::Fp8E4M3 => Precision::Fp8E4M3,
            DType::MxFp8 => Precision::MxFp8,
            DType::NvFp4 => Precision::NvFp4,
            DType::I32 | DType::U32 => {
                return Err(OpError::Unsupported {
                    op: "quantization",
                    dtype: input.name(),
                });
            }
        };
        let symbols: &[&'static str] = match output_dtype {
            DType::NvFp4 => &["opt_bulk_fp4_pack"],
            DType::Fp8E4M3 => match input {
                DType::F32 => &["glm_quantise_rows", "q4e_quantize_e4m3_rows"],
                _ => &["q4e_quantize_e4m3_rows", "glm_quantise_rows"],
            },
            DType::MxFp8 => &["mxfp8_quantize_activations_sm121"],
            _ => &[],
        };
        let mut geometry = requirement.geometry;
        geometry.rows = output.rows;
        geometry.intermediate = output.columns;
        let resolved = context
            .resolve(
                &PartRequirement {
                    part: Part::Quantize,
                    geometry,
                    precision,
                },
                KernelPhase::for_rows(output.rows),
                symbols,
            )?
            .id();
        let plan = Self {
            output,
            input,
            output_dtype,
            layout,
            precision,
            stream,
            geometry,
            resolved,
        };
        plan.validate()?;
        Ok(plan)
    }
}

impl OpPlan for QuantizePlan {
    fn kind(&self) -> OpKind {
        OpKind::Quantize
    }

    fn stream(&self) -> StreamId {
        self.stream
    }

    fn kernel(&self) -> KernelId {
        self.resolved
    }

    fn launch(&self) -> Launch {
        Launch {
            kernel: self.kernel(),
            grid: (self.output.rows.div_ceil(8) as u32, 1, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
            stream: self.stream,
        }
    }

    fn workspace_floats(&self) -> usize {
        0
    }

    fn validate(&self) -> Result<(), OpError> {
        Operand::new(self.output, self.output_dtype, self.layout)
            .validate("quantization output")?;
        if !matches!(
            self.output_dtype,
            DType::NvFp4 | DType::Fp8E4M3 | DType::MxFp8
        ) {
            return Err(OpError::Unsupported {
                op: "quantization",
                dtype: self.output_dtype.name(),
            });
        }
        Ok(())
    }
}

/// A gather from an embedding, PLE or Engram table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatherPlan {
    /// Whether the table is a PLE/Engram table rather than a plain embedding.
    pub engram: bool,
    /// Rows to gather.
    pub rows: usize,
    /// The table's width.
    pub width: usize,
    /// Ids per row.
    pub ids_per_row: usize,
    /// The stream the plan launches on.
    pub stream: StreamId,
    /// The geometry the kernel was resolved at.
    pub geometry: crate::plan::PartGeometry,
    /// The catalogue-selected kernel.
    pub resolved: KernelId,
}

impl GatherPlan {
    /// A gather plan built from the requirement's geometry.
    ///
    /// # Errors
    /// A zero dimension, a requirement for the wrong table part, or no catalogue kernel.
    pub fn new(
        context: PlanContext,
        requirement: &PartRequirement,
        engram: bool,
        rows: usize,
        width: usize,
        ids_per_row: usize,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        let part = if engram {
            Part::Engram
        } else {
            Part::Embedding
        };
        if requirement.part != part {
            return Err(OpError::Binding(format!(
                "a {} gather was built from a {} requirement",
                if engram { "Engram" } else { "embedding" },
                requirement.part
            )));
        }
        if rows == 0 || width == 0 || ids_per_row == 0 {
            return Err(OpError::shape("gather", "a dimension is zero"));
        }
        let mut geometry = requirement.geometry;
        geometry.rows = rows;
        // The table width is the embedding's hidden row for a plain table and the Engram head
        // width for an Engram table; each is the field its catalogue entry's rule names.
        if engram {
            geometry.head_dim = width;
        } else {
            geometry.hidden = width;
        }
        let symbols: &[&'static str] = if engram {
            &["engram_embed_sm121", "opt_bulk_engram_embed"]
        } else {
            &["q4e_embed", "opt_rows_embed_own"]
        };
        // Resolve the table width the gather launches, not the requirement's original hidden.
        let requirement = PartRequirement {
            geometry,
            ..*requirement
        };
        let resolved = context
            .resolve(&requirement, KernelPhase::for_rows(rows), symbols)?
            .id();
        let plan = Self {
            engram,
            rows,
            width,
            ids_per_row,
            stream,
            geometry,
            resolved,
        };
        plan.validate()?;
        Ok(plan)
    }
}

impl OpPlan for GatherPlan {
    fn kind(&self) -> OpKind {
        if self.engram {
            OpKind::EngramGather
        } else {
            OpKind::EmbeddingGather
        }
    }

    fn stream(&self) -> StreamId {
        self.stream
    }

    fn kernel(&self) -> KernelId {
        self.resolved
    }

    fn launch(&self) -> Launch {
        Launch {
            kernel: self.kernel(),
            grid: (self.rows.div_ceil(4) as u32, 1, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
            stream: self.stream,
        }
    }

    fn workspace_floats(&self) -> usize {
        0
    }

    fn validate(&self) -> Result<(), OpError> {
        if self.rows == 0 || self.width == 0 || self.ids_per_row == 0 {
            return Err(OpError::shape("gather", "a dimension is zero"));
        }
        Ok(())
    }
}

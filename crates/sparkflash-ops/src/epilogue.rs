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

use crate::error::OpError;
use crate::images::Image;
use crate::plan::{DType, KernelId, Launch, Layout, Matrix, OpKind, OpPlan, Operand, StreamId};

/// The norm a plan applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    /// Root-mean-square normalization.
    RmsNorm,
    /// Layer normalization.
    LayerNorm,
    /// The hyper-connection mix.
    HyperConnection,
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
}

impl NormPlan {
    /// A norm plan.
    ///
    /// # Errors
    /// A zero dimension, or a dtype the norm does not take.
    pub fn new(
        kind: NormKind,
        rows: usize,
        width: usize,
        dtype: DType,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        let plan = Self {
            kind,
            rows,
            width,
            dtype,
            stream,
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
        let symbol = match self.kind {
            NormKind::RmsNorm => "rmsnorm_sm121",
            NormKind::LayerNorm => "layer_seams_sm121",
            NormKind::HyperConnection => "mega_mhc_sm121",
        };
        KernelId::new(Image::Deepseek, symbol)
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
}

impl RopePlan {
    /// A RoPE plan.
    ///
    /// # Errors
    /// A zero dimension or an even head width that is wider than the row stride.
    pub fn new(
        rows: usize,
        heads: usize,
        head_dim: usize,
        row_stride: usize,
        position: usize,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        let plan = Self {
            rows,
            heads,
            head_dim,
            row_stride,
            position,
            stream,
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
        KernelId::new(Image::Qwen4Exp, "q4e_rope")
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

/// A fused activation epilogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationPlan {
    /// The activation.
    pub kind: ActivationKind,
    /// Elements before the gate halves are paired.
    pub rows: usize,
    /// The intermediate width.
    pub width: usize,
    /// The stream the plan launches on.
    pub stream: StreamId,
}

impl ActivationPlan {
    /// An activation plan.
    ///
    /// # Errors
    /// A zero dimension.
    pub fn new(
        kind: ActivationKind,
        rows: usize,
        width: usize,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        let plan = Self {
            kind,
            rows,
            width,
            stream,
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
        KernelId::new(Image::Deepseek, "swiglu_mxfp8_quant_sm121")
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
    /// The stream the plan launches on.
    pub stream: StreamId,
}

impl QuantizePlan {
    /// A quantization plan.
    ///
    /// # Errors
    /// A zero dimension, a block-scaled layout whose rows do not divide, or an unsupported pair.
    pub fn new(
        output: Matrix,
        input: DType,
        output_dtype: DType,
        layout: Layout,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        let plan = Self {
            output,
            input,
            output_dtype,
            layout,
            stream,
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
        let symbol = match self.output_dtype {
            DType::NvFp4 => "q4e_quantize_e2m1_rows",
            DType::Fp8E4M3 => "q4e_quantize_e4m3_rows",
            _ => "glm_quantise_rows",
        };
        KernelId::new(Image::Qwen4Exp, symbol)
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
            DType::NvFp4 | DType::Fp8E4M3 | DType::Bf16
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
}

impl GatherPlan {
    /// A gather plan.
    ///
    /// # Errors
    /// A zero dimension.
    pub fn new(
        engram: bool,
        rows: usize,
        width: usize,
        ids_per_row: usize,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        let plan = Self {
            engram,
            rows,
            width,
            ids_per_row,
            stream,
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
        if self.engram {
            KernelId::new(Image::Deepseek, "engram_embed_sm121")
        } else {
            KernelId::new(Image::Qwen4Exp, "q4e_embed")
        }
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

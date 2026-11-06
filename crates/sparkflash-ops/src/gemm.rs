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

//! Dense, tile-batched, split-K and grouped-MoE GEMM plans.

use crate::error::OpError;
use crate::images::Image;
use crate::plan::{
    DType, KernelId, Launch, Layout, Matrix, OpKind, OpPlan, Operand, Precision, StreamId,
};

/// A dense matrix multiply, C = A B, with every operand fixed at load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseGemmPlan {
    /// The output, M by N.
    pub output: Matrix,
    /// The reduction width K.
    pub k: usize,
    /// The operands' precision.
    pub precision: Precision,
    /// How the operands are laid out.
    pub layout: Layout,
    /// The stream the plan launches on.
    pub stream: StreamId,
    /// How many K slices the launch takes; 1 is an unsplit GEMM.
    pub split_k: usize,
    /// Whether this is the explicit tile-batched form.
    pub tile_batched: bool,
    /// The kernel image.
    pub image: Image,
    /// The symbol the image exports.
    pub symbol: &'static str,
}

impl DenseGemmPlan {
    /// A dense GEMM for an M by N output reduced over K.
    ///
    /// # Errors
    /// An empty dimension, a K that does not split evenly, or a layout the precision cannot take.
    pub fn new(
        output: Matrix,
        k: usize,
        precision: Precision,
        layout: Layout,
        stream: StreamId,
        split_k: usize,
    ) -> Result<Self, OpError> {
        let (image, symbol) = match (precision, split_k) {
            (Precision::MxFp8, 1) => (Image::Deepseek, "b12x_mxfp8_dense_sm121"),
            (Precision::MxFp8, _) => (Image::Deepseek, "b12x_mxfp8_dense_split_k_sm121"),
            (Precision::NvFp4, _) => (Image::Deepseek, "b12x_mxfp4_moe_sm121"),
            (Precision::Fp8E4M3, _) => (Image::GlmPrefill, "glm_prefill_fp8b128_mma"),
            (Precision::Bf16, _) => (Image::GlmPrefill, "glm_prefill_bf16_mma"),
            (Precision::F32, _) => (Image::Qwen4Exp, "q4e_gemm_bf16"),
        };
        let plan = Self {
            output,
            k,
            precision,
            layout,
            stream,
            split_k,
            tile_batched: false,
            image,
            symbol,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// The same GEMM as an explicit tile-batched plan.
    ///
    /// # Errors
    /// As [Self::new].
    pub fn tile_batched(
        output: Matrix,
        k: usize,
        precision: Precision,
        layout: Layout,
        stream: StreamId,
        split_k: usize,
    ) -> Result<Self, OpError> {
        let mut plan = Self::new(output, k, precision, layout, stream, split_k)?;
        plan.tile_batched = true;
        Ok(plan)
    }

    /// Whether the plan takes the split-K path.
    #[must_use]
    pub fn is_split_k(&self) -> bool {
        self.split_k > 1
    }

    /// The output operand description.
    #[must_use]
    pub fn output_operand(&self) -> Operand {
        Operand::new(self.output, self.precision.operand(), self.layout)
    }
}

impl OpPlan for DenseGemmPlan {
    fn kind(&self) -> OpKind {
        if self.is_split_k() {
            OpKind::SplitKGemm
        } else if self.tile_batched {
            OpKind::TileBatchedGemm
        } else {
            OpKind::DenseGemm
        }
    }

    fn stream(&self) -> StreamId {
        self.stream
    }

    fn kernel(&self) -> KernelId {
        KernelId::new(self.image, self.symbol)
    }

    fn launch(&self) -> Launch {
        let tile = if self.tile_batched { 64 } else { 128 };
        Launch {
            kernel: self.kernel(),
            grid: (
                self.output.columns.div_ceil(tile) as u32,
                self.output.rows.div_ceil(tile) as u32,
                self.split_k as u32,
            ),
            block: (256, 1, 1),
            shared_bytes: 0,
            stream: self.stream,
        }
    }

    fn workspace_floats(&self) -> usize {
        if self.is_split_k() {
            self.split_k * self.output.elements()
        } else {
            0
        }
    }

    fn validate(&self) -> Result<(), OpError> {
        self.output.validate("dense GEMM output")?;
        if self.k == 0 {
            return Err(OpError::shape("dense GEMM", "K is zero"));
        }
        if self.split_k == 0 || !self.k.is_multiple_of(self.split_k) {
            return Err(OpError::shape(
                "dense GEMM",
                format!("K {} does not split into {} slices", self.k, self.split_k),
            ));
        }
        self.output_operand().validate("dense GEMM output")?;
        Ok(())
    }
}

/// A grouped GEMM over the experts a router selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupedMoeGemmPlan {
    /// The routed rows, token-slot pairs.
    pub rows: usize,
    /// The hidden width each row reads.
    pub hidden: usize,
    /// The intermediate width one expert computes.
    pub intermediate: usize,
    /// The number of experts the table addresses.
    pub experts: usize,
    /// How many experts one token selects.
    pub top_k: usize,
    /// The experts' precision.
    pub precision: Precision,
    /// The stream the plan launches on.
    pub stream: StreamId,
}

impl GroupedMoeGemmPlan {
    /// A grouped expert GEMM.
    ///
    /// # Errors
    /// An empty dimension, a top-k larger than the expert count, or an unsupported precision.
    pub fn new(
        rows: usize,
        hidden: usize,
        intermediate: usize,
        experts: usize,
        top_k: usize,
        precision: Precision,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        let plan = Self {
            rows,
            hidden,
            intermediate,
            experts,
            top_k,
            precision,
            stream,
        };
        plan.validate()?;
        Ok(plan)
    }
}

impl OpPlan for GroupedMoeGemmPlan {
    fn kind(&self) -> OpKind {
        OpKind::GroupedMoeGemm
    }

    fn stream(&self) -> StreamId {
        self.stream
    }

    fn kernel(&self) -> KernelId {
        let (image, symbol) = match self.precision {
            Precision::NvFp4 => (Image::Deepseek, "b12x_mxfp4_moe_sm121"),
            // The grouped MXFP8 GEMM is the output-projection kernel: the dense twins
            // (b12x_mxfp8_dense_sm121 and b12x_bf16a_mxfp8w_dense_sm121) tile one
            // matrix, not a group axis.
            Precision::MxFp8 => (Image::Deepseek, "b12x_mxfp8_wo_a_sm121"),
            Precision::Fp8E4M3 => (Image::GlmMoe, "glm_moe_fp8b128_mma"),
            Precision::Bf16 => (Image::GlmMoe, "glm_moe_fp8b128_mma_bf16a"),
            Precision::F32 => (Image::Qwen4Moe, "q4e_expert_gate_up"),
        };
        KernelId::new(image, symbol)
    }

    fn launch(&self) -> Launch {
        Launch {
            kernel: self.kernel(),
            grid: (self.experts as u32, self.rows.div_ceil(32) as u32, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
            stream: self.stream,
        }
    }

    fn workspace_floats(&self) -> usize {
        self.rows * (self.hidden + 3 * self.intermediate)
    }

    fn validate(&self) -> Result<(), OpError> {
        for (what, value) in [
            ("rows", self.rows),
            ("hidden", self.hidden),
            ("intermediate", self.intermediate),
            ("experts", self.experts),
            ("top_k", self.top_k),
        ] {
            if value == 0 {
                return Err(OpError::shape(
                    "grouped MoE GEMM",
                    format!("{what} is zero"),
                ));
            }
        }
        if self.top_k > self.experts {
            return Err(OpError::shape(
                "grouped MoE GEMM",
                format!("top-k {} exceeds {} experts", self.top_k, self.experts),
            ));
        }
        if self.precision == Precision::MxFp8 {
            // The only grouped MXFP8 symbol in the tree is the MLA wo_a projection, which takes
            // attention operands, not expert ids. Refuse rather than name a kernel whose ABI does
            // not match the plan.
            return Err(OpError::Unsupported {
                op: "grouped MoE GEMM",
                dtype: "mxfp8 (no grouped expert kernel yet)",
            });
        }
        if !matches!(
            self.precision,
            Precision::NvFp4 | Precision::Fp8E4M3 | Precision::Bf16 | Precision::F32
        ) {
            return Err(OpError::Unsupported {
                op: "grouped MoE GEMM",
                dtype: self.precision.name(),
            });
        }
        Ok(())
    }
}

/// Router logits and top-k selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeRoutingPlan {
    /// Routed rows.
    pub rows: usize,
    /// The hidden width.
    pub hidden: usize,
    /// Experts to score.
    pub experts: usize,
    /// Experts to keep.
    pub top_k: usize,
    /// The stream the plan launches on.
    pub stream: StreamId,
}

impl MoeRoutingPlan {
    /// A routing plan.
    ///
    /// # Errors
    /// A zero or inverted dimension.
    pub fn new(
        rows: usize,
        hidden: usize,
        experts: usize,
        top_k: usize,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        let plan = Self {
            rows,
            hidden,
            experts,
            top_k,
            stream,
        };
        plan.validate()?;
        Ok(plan)
    }
}

impl OpPlan for MoeRoutingPlan {
    fn kind(&self) -> OpKind {
        OpKind::MoeRouting
    }

    fn stream(&self) -> StreamId {
        self.stream
    }

    fn kernel(&self) -> KernelId {
        KernelId::new(Image::Deepseek, "moe_router_sm121")
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
        self.rows * self.experts
    }

    fn validate(&self) -> Result<(), OpError> {
        if self.rows == 0 || self.hidden == 0 || self.experts == 0 {
            return Err(OpError::shape("MoE routing", "a dimension is zero"));
        }
        if self.top_k == 0 || self.top_k > self.experts {
            return Err(OpError::shape(
                "MoE routing",
                format!("top-k {} is not in 1..={}", self.top_k, self.experts),
            ));
        }
        Ok(())
    }
}

/// The expert combine: a weighted sum of routed rows back into the hidden stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertCombinePlan {
    /// Output rows.
    pub rows: usize,
    /// The hidden width.
    pub hidden: usize,
    /// Experts each row sums.
    pub top_k: usize,
    /// The accumulator's dtype.
    pub dtype: DType,
    /// The stream the plan launches on.
    pub stream: StreamId,
}

impl ExpertCombinePlan {
    /// A combine plan.
    ///
    /// # Errors
    /// A zero dimension or a dtype the combine does not take.
    pub fn new(
        rows: usize,
        hidden: usize,
        top_k: usize,
        dtype: DType,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        let plan = Self {
            rows,
            hidden,
            top_k,
            dtype,
            stream,
        };
        plan.validate()?;
        Ok(plan)
    }
}

impl OpPlan for ExpertCombinePlan {
    fn kind(&self) -> OpKind {
        OpKind::ExpertCombine
    }

    fn stream(&self) -> StreamId {
        self.stream
    }

    fn kernel(&self) -> KernelId {
        KernelId::new(Image::Deepseek, "moe_routed_combine_sm121")
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
        self.rows * self.hidden * self.top_k
    }

    fn validate(&self) -> Result<(), OpError> {
        if self.rows == 0 || self.hidden == 0 || self.top_k == 0 {
            return Err(OpError::shape("expert combine", "a dimension is zero"));
        }
        if !matches!(self.dtype, DType::F32 | DType::Bf16) {
            return Err(OpError::Unsupported {
                op: "expert combine",
                dtype: self.dtype.name(),
            });
        }
        Ok(())
    }
}

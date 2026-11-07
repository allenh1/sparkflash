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

use sparkflash_model::Part;

use crate::engine_plan::PartRequirement;
use crate::error::OpError;
use crate::plan::{
    DType, GroupedFp8Body, KernelId, KernelPhase, Launch, Layout, Matrix, OpKind, OpPlan, Operand,
    PartGeometry, Precision, StreamId,
};
use crate::resolver::PlanContext;

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
    /// The geometry the kernel was resolved at.
    pub geometry: PartGeometry,
    /// The kernel the catalogue selected. The image and symbol are not public: a caller cannot
    /// bypass the resolver by writing them.
    resolved: KernelId,
}

impl DenseGemmPlan {
    /// A dense GEMM for an M by N output reduced over K, from a DenseGemm requirement.
    ///
    /// # Errors
    /// An empty dimension, a K that does not split evenly, a precision with no dense kernel, a
    /// requirement for a different part, or no catalogue kernel.
    pub fn new(
        context: PlanContext,
        requirement: &PartRequirement,
        output: Matrix,
        k: usize,
        layout: Layout,
        stream: StreamId,
        split_k: usize,
    ) -> Result<Self, OpError> {
        if requirement.part != Part::DenseGemm {
            return Err(OpError::Binding(format!(
                "a dense GEMM was built from a {} requirement",
                requirement.part
            )));
        }
        output.validate("dense GEMM output")?;
        if k == 0 {
            return Err(OpError::shape("dense GEMM", "K is zero"));
        }
        if split_k == 0 || !k.is_multiple_of(split_k) {
            return Err(OpError::shape(
                "dense GEMM",
                format!("K {k} does not split into {split_k} slices"),
            ));
        }
        let precision = requirement.precision;
        let symbols = dense_symbols(precision, split_k)?;
        let mut geometry = requirement.geometry;
        geometry.hidden = k;
        geometry.rows = output.rows;
        geometry.intermediate = output.columns;
        let resolved = context
            .resolve(
                &PartRequirement {
                    geometry,
                    ..*requirement
                },
                KernelPhase::for_rows(output.rows),
                symbols,
            )?
            .id();
        let plan = Self {
            output,
            k,
            precision,
            layout,
            stream,
            split_k,
            tile_batched: false,
            geometry,
            resolved,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// The same GEMM as an explicit tile-batched plan.
    ///
    /// # Errors
    /// As [Self::new].
    pub fn tile_batched(
        context: PlanContext,
        requirement: &PartRequirement,
        output: Matrix,
        k: usize,
        layout: Layout,
        stream: StreamId,
        split_k: usize,
    ) -> Result<Self, OpError> {
        let mut plan = Self::new(context, requirement, output, k, layout, stream, split_k)?;
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

/// The dense GEMM symbols a precision and split need, most specific first.
fn dense_symbols(precision: Precision, split_k: usize) -> Result<&'static [&'static str], OpError> {
    match (precision, split_k > 1) {
        (Precision::MxFp8, true) => Ok(&["b12x_mxfp8_dense_split_k_sm121"]),
        (_, true) => Err(OpError::Unsupported {
            op: "split-K dense GEMM",
            dtype: precision.name(),
        }),
        (Precision::MxFp8, false) => Ok(&[
            "b12x_mxfp8_dense_sm121",
            "b12x_mxfp8_dense_m1_sm121",
            "b12x_bf16a_mxfp8w_dense_sm121",
        ]),
        (Precision::Bf16, false) => Ok(&[
            "q4e_gemm_bf16",
            "opt_mma_bf16a_bf16o",
            "glm_prefill_bf16_mma",
            "q4p_dense_mma",
        ]),
        (Precision::Fp8E4M3, false) => Ok(&[
            "glm_prefill_fp8b128_mma",
            "q4p_dense_mma",
            "q4e_gemv_tiled_e4m3_block",
            "q4e_gemv_tiled_e4m3",
            "opt_mma_fp8_bf16o",
        ]),
        (Precision::F32, false) => Ok(&[
            "q4e_gemm_bf16",
            "opt_mma_f32a_f32o",
            "b12x_hostw_gemv_m1_f32w_f32o_sm121",
        ]),
        (Precision::NvFp4, false) => Err(OpError::Unsupported {
            op: "dense GEMM",
            dtype: "nvfp4 (no dense NVFP4 body; use the grouped MoE kernel)",
        }),
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
        self.resolved
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

/// The side of one FP8 expert's scale tile: a rank's slice of an FP8 expert is a run of whole
/// 128-column tiles, so a grouped FP8 plan over anything else cannot launch.
const FP8_EXPERT_TILE: usize = 128;

/// The Qwen FP8 symbols, in the host's priority order: the grouped tile for a chunk, then the
/// token experts for a step below the host's threshold.
const FP8_TILED_SYMBOLS: &[&str] = &[
    "q4p_moe_gate_up_fp8",
    "q4p_moe_swiglu_fp8",
    "q4p_moe_down_fp8",
    "q4e_expert_gate_up_fp8",
    "q4e_expert_down_fp8",
    "q4e_expert_gemv_e4m3_block",
];

/// The GLM dynamic-activation symbols, in the host's priority order. Its released profile
/// quantises the chunk's activations (`PrefillActivation::Fp8`), so `glm_moe_q8_mma` is the
/// body a chunk launches; the split-activation block MMA is the first fallback and the
/// BF16-activation one the second. A step of at most `MAX_ROWS` (8) token rows takes the
/// decode GEMV, and the host groups from 9 tokens, which is why the GEMV precedes the token
/// FP8 experts here.
const FP8_QUANTIZED_SYMBOLS: &[&str] = &[
    "glm_moe_q8_mma",
    "glm_moe_fp8b128_mma",
    "glm_moe_fp8b128_mma_bf16a",
    "q4p_moe_gate_up_fp8",
    "q4e_expert_gemv_e4m3_block",
    "q4e_expert_gate_up_fp8",
    "q4e_expert_down_fp8",
];

/// The GLM BF16-activation symbols (`prefill.activation = "bf16"`): the `_bf16a` block MMA
/// first, then the split form and the dynamic q8 form. The decode GEMV and token FP8 experts
/// follow so a step below the host's grouped threshold still resolves.
const FP8_BF16_SYMBOLS: &[&str] = &[
    "glm_moe_fp8b128_mma_bf16a",
    "glm_moe_fp8b128_mma",
    "glm_moe_q8_mma",
    "q4p_moe_gate_up_fp8",
    "q4e_expert_gemv_e4m3_block",
    "q4e_expert_gate_up_fp8",
    "q4e_expert_down_fp8",
];

/// The GLM split-activation symbols (`prefill.activation = "split"`): the split block MMA
/// first, then the BF16-activation and dynamic q8 forms, then the token bodies.
const FP8_SPLIT_SYMBOLS: &[&str] = &[
    "glm_moe_fp8b128_mma",
    "glm_moe_fp8b128_mma_bf16a",
    "glm_moe_q8_mma",
    "q4p_moe_gate_up_fp8",
    "q4e_expert_gemv_e4m3_block",
    "q4e_expert_gate_up_fp8",
    "q4e_expert_down_fp8",
];

/// A grouped GEMM over the experts a router selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupedMoeGemmPlan {
    /// The tokens in the batch.
    pub tokens: usize,
    /// The routed rows the launch covers: `tokens * top_k` token-slot pairs.
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
    /// The geometry the kernel was resolved at.
    pub geometry: PartGeometry,
    /// The catalogue-selected kernel.
    pub resolved: KernelId,
}

impl GroupedMoeGemmPlan {
    /// A grouped expert GEMM over `tokens` tokens.
    ///
    /// `tokens` is the batch, not the routed row count. The plan derives the routed rows as
    /// `tokens * top_k` and records both, so no caller has to guess whether a host threshold
    /// counts tokens or token-slot pairs. The host's grouped threshold is named in tokens too
    /// ([PartGeometry::grouped_min_tokens]): Qwen compares its batch against 64 tokens and GLM
    /// counts a step's token rows against 8, grouping from 9 tokens.
    ///
    /// The pass falls back on `tokens`, never on the derived row count: a one-token batch is
    /// a decode step even when top-k makes its routed rows look like a chunk. A zero threshold
    /// therefore keeps [KernelPhase::for_rows]`(tokens)`, and an FP8 plan must name its
    /// grouped body ([PartGeometry::grouped_fp8_body]).
    ///
    /// # Errors
    /// An empty batch or dimension, a top-k larger than the expert count, an unsupported
    /// precision, a requirement for a different part, or no catalogue kernel.
    pub fn new(
        context: PlanContext,
        requirement: &PartRequirement,
        tokens: usize,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        if requirement.part != Part::GroupedMoeGemm {
            return Err(OpError::Binding(format!(
                "a grouped MoE GEMM was built from a {} requirement",
                requirement.part
            )));
        }
        let precision = requirement.precision;
        if tokens == 0 {
            return Err(OpError::shape("grouped MoE GEMM", "tokens is zero"));
        }
        let rows = tokens
            .checked_mul(requirement.geometry.top_k)
            .ok_or_else(|| {
                OpError::shape(
                    "grouped MoE GEMM",
                    format!(
                        "{tokens} tokens x top-k {} exceeds usize",
                        requirement.geometry.top_k
                    ),
                )
            })?;
        let geometry = PartGeometry {
            rows,
            ..requirement.geometry
        };
        for (what, value) in [
            ("rows", geometry.rows),
            ("hidden", geometry.hidden),
            ("intermediate", geometry.intermediate),
            ("experts", geometry.experts),
            ("top_k", geometry.top_k),
        ] {
            if value == 0 {
                return Err(OpError::shape(
                    "grouped MoE GEMM",
                    format!("{what} is zero"),
                ));
            }
        }
        if geometry.top_k > geometry.experts {
            return Err(OpError::shape(
                "grouped MoE GEMM",
                format!(
                    "top-k {} exceeds {} experts",
                    geometry.top_k, geometry.experts
                ),
            ));
        }
        if precision == Precision::MxFp8 {
            // The only grouped MXFP8 symbol in the tree is the MLA wo_a projection, which takes
            // attention operands, not expert ids. Refuse rather than name a kernel whose ABI does
            // not match the plan.
            return Err(OpError::Unsupported {
                op: "grouped MoE GEMM",
                dtype: "mxfp8 (no grouped expert kernel yet)",
            });
        }
        if precision == Precision::Fp8E4M3 && !geometry.intermediate.is_multiple_of(FP8_EXPERT_TILE)
        {
            // The host's `routed_experts_mma_fp8` guard: an FP8 expert is whole 128-column
            // scale tiles, so a grouped FP8 plan over a narrower or fractional width cannot
            // launch the body it would name.
            return Err(OpError::shape(
                "grouped MoE GEMM",
                format!(
                    "an FP8 expert {} columns is not whole {FP8_EXPERT_TILE}-column scale tiles",
                    geometry.intermediate
                ),
            ));
        }
        let symbols: &[&'static str] = match precision {
            Precision::NvFp4 => &[
                "b12x_mxfp4_moe_sm121",
                "q4e_expert_gate_up",
                "q4p_moe_gate_up",
            ],
            // The body the arm's host actually launches comes first: the grouped tile for a
            // Qwen chunk, the q8 MMA for GLM's released dynamic-activation profile. A step below
            // the host's threshold requests the decode pass and falls through the prefill-only
            // grouped symbols to that arm's token body.
            Precision::Fp8E4M3 => match geometry.grouped_fp8_body.ok_or_else(|| {
                OpError::Binding(
                    "an FP8 grouped MoE GEMM names no grouped body; set \
                     PartGeometry::grouped_fp8_body to the arm's Tiled, QuantizedActivation, \
                     Bf16Activation or SplitActivation body"
                        .to_owned(),
                )
            })? {
                GroupedFp8Body::Tiled => FP8_TILED_SYMBOLS,
                GroupedFp8Body::QuantizedActivation => FP8_QUANTIZED_SYMBOLS,
                GroupedFp8Body::Bf16Activation => FP8_BF16_SYMBOLS,
                GroupedFp8Body::SplitActivation => FP8_SPLIT_SYMBOLS,
            },
            // glm_moe_fp8b128_mma_bf16a is an FP8 weight body (E4M3 codes, BF16 tile scale)
            // whose `_bf16a` suffix names its activations; it is not a BF16 weight kernel. No
            // BF16 or F32 routed-expert body exists at all: `q4e_expert_gate_up`/`down` decode
            // packed E2M1 codes, `_fp8` decodes E4M3, and the GLM bodies read E4M3 with a BF16
            // tile scale. Refuse rather than launch a body that would misread the weights.
            Precision::Bf16 | Precision::F32 => {
                return Err(OpError::Unsupported {
                    op: "grouped MoE GEMM",
                    dtype: precision.name(),
                });
            }
            Precision::MxFp8 => &[],
        };
        // The pass is the host's threshold, not a bare row test, and the fallback counts
        // tokens rather than routed rows: one token with top-k 10 has ten rows but is still a
        // decode step. Otherwise a batch below the arm's token threshold runs the token bodies,
        // so a 32-token Qwen batch names that body and not the prefill-only grouped tile.
        let phase = if geometry.grouped_min_tokens != 0 && tokens < geometry.grouped_min_tokens {
            KernelPhase::Decode
        } else {
            KernelPhase::for_rows(tokens)
        };
        let requirement = PartRequirement {
            geometry,
            ..*requirement
        };
        let resolved = context.resolve(&requirement, phase, symbols)?.id();
        let plan = Self {
            tokens,
            rows,
            hidden: geometry.hidden,
            intermediate: geometry.intermediate,
            experts: geometry.experts,
            top_k: geometry.top_k,
            precision,
            stream,
            geometry,
            resolved,
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
        self.resolved
    }

    fn launch(&self) -> Launch {
        // The Qwen token expert GEMV stages its activation in dynamic shared memory: two
        // members of the wider projection, one padded float a column group.
        let pad = |columns: usize| columns.div_ceil(32) * 33;
        let shared = if self.precision == Precision::F32 {
            2 * pad(self.hidden.max(self.intermediate)) * std::mem::size_of::<f32>()
        } else {
            0
        };
        Launch {
            kernel: self.kernel(),
            grid: (self.experts as u32, self.rows.div_ceil(32) as u32, 1),
            block: (256, 1, 1),
            shared_bytes: shared as u32,
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
        // No grouped MXFP8 body exists, and no BF16 or F32 routed-expert body either: the
        // released experts are E2M1 (the `q4e_expert_*` and `q4p_moe_*` forms) or E4M3 (the
        // `_fp8` forms and the GLM block MMA). A plan built around one of them must be refused
        // by validate even when it did not come through [GroupedMoeGemmPlan::new].
        if matches!(
            self.precision,
            Precision::MxFp8 | Precision::Bf16 | Precision::F32
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
    /// The precision the router body was resolved at.
    pub precision: Precision,
    /// The stream the plan launches on.
    pub stream: StreamId,
    /// The geometry the kernel was resolved at.
    pub geometry: PartGeometry,
    /// The catalogue-selected kernel.
    pub resolved: KernelId,
}

impl MoeRoutingPlan {
    /// A routing plan built from the requirement's geometry.
    ///
    /// # Errors
    /// A zero or inverted dimension, a requirement for a different part, or no catalogue kernel.
    pub fn new(
        context: PlanContext,
        requirement: &PartRequirement,
        rows: usize,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        if requirement.part != Part::MoeRouter {
            return Err(OpError::Binding(format!(
                "a router was built from a {} requirement",
                requirement.part
            )));
        }
        let geometry = PartGeometry {
            rows,
            ..requirement.geometry
        };
        if geometry.rows == 0 || geometry.hidden == 0 || geometry.experts == 0 {
            return Err(OpError::shape("MoE routing", "a dimension is zero"));
        }
        if geometry.top_k == 0 || geometry.top_k > geometry.experts {
            return Err(OpError::shape(
                "MoE routing",
                format!(
                    "top-k {} is not in 1..={}",
                    geometry.top_k, geometry.experts
                ),
            ));
        }
        let symbols: &[&'static str] = if geometry.experts == 128 && geometry.top_k == 3 {
            &["opt_router_e128k3_sm121", "q4e_router", "q4e_noaux_router"]
        } else {
            &[
                "opt_router_sm121",
                "moe_router_sm121",
                "moe_router_topk_sm121",
                "opt_mma_router_dots",
                "opt_mma_router_topk",
                "q4e_router",
                "q4e_noaux_router",
            ]
        };
        // Resolve the plan's own row count with the requirement's other dimensions.
        let requirement = PartRequirement {
            geometry,
            ..*requirement
        };
        let resolved = context.resolve_any(&requirement, KernelPhase::for_rows(rows), symbols)?;
        let plan = Self {
            rows,
            hidden: geometry.hidden,
            experts: geometry.experts,
            top_k: geometry.top_k,
            precision: resolved.precision,
            stream,
            geometry,
            resolved: resolved.id(),
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
    /// The precision the combine body was resolved at.
    pub precision: Precision,
    /// The stream the plan launches on.
    pub stream: StreamId,
    /// The geometry the kernel was resolved at.
    pub geometry: PartGeometry,
    /// The catalogue-selected kernel.
    pub resolved: KernelId,
}

impl ExpertCombinePlan {
    /// A combine plan built from the requirement's geometry.
    ///
    /// # Errors
    /// A zero dimension, a dtype the combine does not take, a requirement for a different part,
    /// or no catalogue kernel.
    pub fn new(
        context: PlanContext,
        requirement: &PartRequirement,
        rows: usize,
        dtype: DType,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        if requirement.part != Part::MoeCombine {
            return Err(OpError::Binding(format!(
                "a combine was built from a {} requirement",
                requirement.part
            )));
        }
        let geometry = PartGeometry {
            rows,
            ..requirement.geometry
        };
        if geometry.rows == 0 || geometry.hidden == 0 || geometry.top_k == 0 {
            return Err(OpError::shape("expert combine", "a dimension is zero"));
        }
        if !matches!(dtype, DType::F32 | DType::Bf16) {
            return Err(OpError::Unsupported {
                op: "expert combine",
                dtype: dtype.name(),
            });
        }
        // Resolve the plan's own row count with the requirement's other dimensions.
        let requirement = PartRequirement {
            geometry,
            ..*requirement
        };
        let resolved = context.resolve_any(
            &requirement,
            KernelPhase::for_rows(rows),
            &[
                "moe_routed_combine_sm121",
                "q4e_moe_combine",
                "opt_mma_moe_combine",
            ],
        )?;
        let plan = Self {
            rows,
            hidden: geometry.hidden,
            top_k: geometry.top_k,
            dtype,
            precision: resolved.precision,
            stream,
            geometry,
            resolved: resolved.id(),
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

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

//! Attention plans: MLA, grouped GQA, sparse/CSA and split-KV decode.

use sparkflash_model::Part;

use crate::engine_plan::PartRequirement;
use crate::error::OpError;
use crate::plan::{
    KernelId, KernelPhase, Launch, OpKind, OpPlan, PartGeometry, Precision, StreamId,
};
use crate::resolver::PlanContext;

/// The dynamic shared memory q4e_attn_group reserves: logits and a reduction block for each
/// of its four query heads, then the token's selection.
const GQA_GROUP_SHARED_BYTES: u32 = 45_136;

/// The dynamic shared memory q4e_attn_group6 reserves: the same layout over six query heads.
const GQA_GROUP6_SHARED_BYTES: u32 = 63_600;

/// The attention arithmetic a plan runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    /// Multi-head latent attention over a compressed KV latent.
    Mla,
    /// Grouped-query attention, including six-head groups.
    Gqa {
        /// Query heads.
        query_heads: usize,
        /// Key/value heads; query_heads must divide by it.
        kv_heads: usize,
    },
    /// Sparse block attention over selected blocks.
    Sparse {
        /// Tokens selected per query.
        selected: usize,
        /// Tokens a selected block holds.
        block: usize,
    },
    /// CSA2 compressed attention over pooled keys.
    Csa2 {
        /// The compression ratio.
        ratio: usize,
    },
}

impl AttentionKind {
    /// The model-side part this arithmetic belongs to.
    #[must_use]
    pub const fn part(self) -> Part {
        match self {
            Self::Mla => Part::MlaAttention,
            Self::Gqa { .. } => Part::GqaAttention,
            Self::Sparse { .. } => Part::SparseIndexer,
            Self::Csa2 { .. } => Part::Csa2Compression,
        }
    }
}

/// One attention launch, with the shapes that decide its kernel and shared memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionPlan {
    /// The attention arithmetic.
    pub kind: AttentionKind,
    /// Query rows this launch computes.
    pub rows: usize,
    /// Context positions the queries read.
    pub context: usize,
    /// The per-head width.
    pub head_dim: usize,
    /// The stream the plan launches on.
    pub stream: StreamId,
    /// The operand precision.
    pub precision: Precision,
    /// The geometry the kernel was resolved at, including the requirement's hidden, latent,
    /// group, window and position fields.
    pub geometry: PartGeometry,
    /// The catalogue-selected kernel.
    pub resolved: KernelId,
}

impl AttentionPlan {
    /// Build a plan from the requirement's geometry and the launch's row counts.
    ///
    /// The requirement is the one canonical geometry source: the constructor does not invent a
    /// second, partial shape. It overrides only the fields the schedule owns (rows, context and
    /// head width), checks the kind's own invariants before it divides, and then resolves a
    /// symbol whose precision and pass accept the request.
    ///
    /// # Errors
    /// A zero dimension, a query/key-value head count that does not divide, an unsupported
    /// precision, a requirement for a different part, or no catalogue kernel.
    pub fn new(
        context: PlanContext,
        requirement: &PartRequirement,
        kind: AttentionKind,
        rows: usize,
        context_len: usize,
        head_dim: usize,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        if requirement.part != kind.part() {
            return Err(OpError::Binding(format!(
                "{} attention was built from a {} requirement",
                kind.name(),
                requirement.part
            )));
        }
        validate_inputs(kind, rows, context_len, head_dim)?;
        let precision = requirement.precision;
        if precision == Precision::NvFp4 {
            return Err(OpError::Unsupported {
                op: "attention",
                dtype: precision.name(),
            });
        }
        let mut geometry = requirement.geometry;
        geometry.rows = rows;
        geometry.positions = context_len;
        geometry.head_dim = head_dim;
        if let AttentionKind::Gqa {
            query_heads,
            kv_heads,
        } = kind
        {
            geometry.heads = query_heads;
            geometry.kv_heads = kv_heads;
        }
        if let AttentionKind::Sparse { selected, block } = kind {
            geometry.ratio = selected / block;
        }
        if let AttentionKind::Csa2 { ratio } = kind {
            geometry.ratio = ratio;
        }
        let symbols = match kind {
            AttentionKind::Gqa {
                query_heads,
                kv_heads,
            } => gqa_symbols(query_heads / kv_heads),
            _ => symbols_for(kind, precision),
        };
        let phase = KernelPhase::for_rows(rows);
        // Resolve the geometry the plan will launch: the schedule owns the row count, context
        // and head width, and a kernel must accept those, not the requirement's original values.
        let requirement = PartRequirement {
            geometry,
            ..*requirement
        };
        let resolved = context.resolve(&requirement, phase, symbols)?.id();
        let plan = Self {
            kind,
            rows,
            context: context_len,
            head_dim,
            stream,
            precision,
            geometry,
            resolved,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// The group size for grouped-query attention, or 1 for the other kinds.
    #[must_use]
    pub fn group_size(&self) -> usize {
        match self.kind {
            AttentionKind::Gqa {
                query_heads,
                kv_heads,
            } => query_heads / kv_heads,
            _ => 1,
        }
    }
}

impl AttentionKind {
    /// The kind's stable name for diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Mla => "MLA",
            Self::Gqa { .. } => "grouped-query",
            Self::Sparse { .. } => "sparse",
            Self::Csa2 { .. } => "CSA2",
        }
    }
}

/// Check the kind's invariants before a division or a catalogue lookup.
fn validate_inputs(
    kind: AttentionKind,
    rows: usize,
    context: usize,
    head_dim: usize,
) -> Result<(), OpError> {
    if rows == 0 || context == 0 || head_dim == 0 {
        return Err(OpError::shape("attention", "a dimension is zero"));
    }
    match kind {
        AttentionKind::Gqa {
            query_heads,
            kv_heads,
        } => {
            if query_heads == 0 || kv_heads == 0 || !query_heads.is_multiple_of(kv_heads) {
                return Err(OpError::shape(
                    "grouped-query attention",
                    format!(
                        "{query_heads} query heads do not group over {kv_heads} key/value heads"
                    ),
                ));
            }
        }
        AttentionKind::Sparse { selected, block } => {
            if selected == 0 || block == 0 {
                return Err(OpError::shape(
                    "sparse attention",
                    "selected or block is zero",
                ));
            }
            if !selected.is_multiple_of(block) {
                return Err(OpError::shape(
                    "sparse attention",
                    format!("{selected} selected tokens are not whole {block}-token blocks"),
                ));
            }
        }
        AttentionKind::Csa2 { ratio } => {
            if ratio == 0 {
                return Err(OpError::shape(
                    "CSA2 attention",
                    "the compression ratio is zero",
                ));
            }
        }
        AttentionKind::Mla => {}
    }
    Ok(())
}

/// The symbols the arithmetic needs, most specific first.
const fn symbols_for(kind: AttentionKind, precision: Precision) -> &'static [&'static str] {
    match kind {
        AttentionKind::Mla => match precision {
            Precision::MxFp8 => &["flash_mla_csa2_sm121", "opt_bulk_attn"],
            Precision::Bf16 => &[
                "flash_mla_csa2_sm121",
                "opt_bulk_attn",
                "glm_prefill_mla_attend",
            ],
            Precision::Fp8E4M3 => &[
                "glm_prefill_mla_attend",
                "glm_prefill_mla_attend_bf16a",
                "q4e_glm_mla_attend",
            ],
            Precision::F32 => &["opt_bulk_attn", "glm_prefill_mla_attend"],
            Precision::NvFp4 => &[],
        },
        AttentionKind::Gqa { .. } => &["q4e_attn"], // replaced below for 4/6-head groups
        AttentionKind::Sparse { .. } => {
            &["flash_mla_csa2_seq_sm121", "q4e_qsa_pick", "q4e_glm_pick"]
        }
        AttentionKind::Csa2 { .. } => &["flash_mla_csa2_ring_sm121"],
    }
}

impl OpPlan for AttentionPlan {
    fn kind(&self) -> OpKind {
        match self.kind {
            AttentionKind::Mla => OpKind::MlaAttention,
            AttentionKind::Gqa { .. } => OpKind::GqaAttention,
            AttentionKind::Sparse { .. } => OpKind::SparseAttention,
            AttentionKind::Csa2 { .. } => OpKind::Csa2Attention,
        }
    }

    fn stream(&self) -> StreamId {
        self.stream
    }

    fn kernel(&self) -> KernelId {
        self.resolved
    }

    fn launch(&self) -> Launch {
        let blocks = match self.kind {
            AttentionKind::Gqa {
                query_heads,
                kv_heads,
            } => self.rows * (query_heads / kv_heads).max(1),
            _ => self.rows,
        };
        Launch {
            kernel: self.kernel(),
            grid: (blocks.div_ceil(4) as u32, 1, 1),
            block: (256, 1, 1),
            shared_bytes: match self.kind {
                AttentionKind::Gqa { .. } => gqa_kernel(self.group_size()).1,
                _ => 0,
            },
            stream: self.stream,
        }
    }

    fn workspace_floats(&self) -> usize {
        let selected = match self.kind {
            AttentionKind::Sparse { selected, .. } => selected,
            AttentionKind::Csa2 { ratio } => self.context.div_ceil(ratio.max(1)),
            _ => self.context,
        };
        self.rows * selected
    }

    fn validate(&self) -> Result<(), OpError> {
        validate_inputs(self.kind, self.rows, self.context, self.head_dim)?;
        if self.precision == Precision::NvFp4 {
            return Err(OpError::Unsupported {
                op: "attention",
                dtype: self.precision.name(),
            });
        }
        Ok(())
    }
}

/// The image symbol and dynamic shared memory a GQA head group runs.
///
/// The six-head and four-head grouped kernels are the only specialised variants; every other
/// group (including the 12-head group a TP=1 Qwen arm holds) runs the per-head q4e_attn body,
/// which needs no dynamic shared reservation.
fn gqa_kernel(group: usize) -> (&'static str, u32) {
    match group {
        6 => ("q4e_attn_group6", GQA_GROUP6_SHARED_BYTES),
        4 => ("q4e_attn_group", GQA_GROUP_SHARED_BYTES),
        _ => ("q4e_attn", 0),
    }
}

/// The symbol list for a GQA group, most specialised first.
fn gqa_symbols(group: usize) -> &'static [&'static str] {
    match group {
        6 => &["q4e_attn_group6", "q4e_attn"],
        4 => &["q4e_attn_group", "q4e_attn"],
        _ => &["q4e_attn"],
    }
}

/// Split-KV decode attention: the context is cut over blocks and reduced in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitKvDecodePlan {
    /// The attention arithmetic the split reduces.
    pub attention: AttentionPlan,
    /// How many context slices the launch takes.
    pub splits: usize,
    /// The catalogue-selected kernel.
    pub resolved: KernelId,
}

impl SplitKvDecodePlan {
    /// A split decode plan over an MLA attention.
    ///
    /// # Errors
    /// A non-MLA attention, a context that does not split evenly, or no catalogue kernel.
    pub fn new(
        context: PlanContext,
        attention: AttentionPlan,
        splits: usize,
    ) -> Result<Self, OpError> {
        if attention.kind != AttentionKind::Mla {
            return Err(OpError::Unsupported {
                op: "split-KV decode",
                dtype: "non-MLA attention",
            });
        }
        if splits == 0 || !attention.context.is_multiple_of(splits) {
            return Err(OpError::shape(
                "split-KV decode",
                format!(
                    "a context of {} does not split into {}",
                    attention.context, splits
                ),
            ));
        }
        let requirement = PartRequirement {
            part: Part::MlaAttention,
            geometry: attention.geometry,
            precision: attention.precision,
        };
        let resolved = context
            .resolve(
                &requirement,
                KernelPhase::for_rows(attention.rows),
                &[
                    "opt_flash_mla_csa2_seq",
                    "glm_prefill_mla_attend",
                    "q4e_glm_mla_attend",
                ],
            )?
            .id();
        Ok(Self {
            attention,
            splits,
            resolved,
        })
    }
}

impl OpPlan for SplitKvDecodePlan {
    fn kind(&self) -> OpKind {
        OpKind::SplitKvDecode
    }

    fn stream(&self) -> StreamId {
        self.attention.stream
    }

    fn kernel(&self) -> KernelId {
        self.resolved
    }

    fn launch(&self) -> Launch {
        let mut launch = self.attention.launch();
        launch.kernel = self.kernel();
        launch.grid.2 = self.splits as u32;
        launch
    }

    fn workspace_floats(&self) -> usize {
        self.splits * self.attention.rows * self.attention.head_dim
    }

    fn validate(&self) -> Result<(), OpError> {
        self.attention.validate()?;
        if self.splits == 0 || !self.attention.context.is_multiple_of(self.splits) {
            return Err(OpError::shape(
                "split-KV decode",
                format!(
                    "a context of {} does not split into {}",
                    self.attention.context, self.splits
                ),
            ));
        }
        Ok(())
    }
}

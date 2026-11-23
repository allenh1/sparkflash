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

/// The selected tokens a grouped attention body stages in shared memory
/// (`Q4E_ATTN_GROUP_SELECTED`, kernels/qwen4_attn_sm121.cu); a configuration that selects more
/// attends through `q4e_attn`.
pub const GQA_GROUP_SELECTED: usize = 2052;

/// The grouped attention bodies by the query heads one block computes, in the order a
/// configuration takes them: the first whose head count divides its query groups. Six is a
/// four-rank group's share of Qwen3.8's twelve.
pub const GQA_GROUP_KERNELS: [(usize, &str); 2] = [(4, "q4e_attn_group"), (6, "q4e_attn_group6")];

/// A grouped attention body's dynamic shared memory over `heads` heads: a row of logits and a
/// block of reduction floats for each head, then the token's selection (45,136 bytes at four
/// heads, 63,600 at six).
#[must_use]
pub const fn gqa_group_shared_bytes(heads: usize) -> u32 {
    ((heads * (GQA_GROUP_SELECTED + 256) + GQA_GROUP_SELECTED) * 4) as u32
}

/// The body one grouped-query attention launch runs: how many query heads a block computes
/// together, its symbol and its dynamic shared memory. One head is `q4e_attn`, which reserves
/// none. Every body computes each head as `q4e_attn` would, so the choice moves no bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GqaBody {
    /// The query heads a block computes.
    pub heads: usize,
    /// The image symbol.
    pub symbol: &'static str,
    /// The dynamic shared memory a launch reserves.
    pub shared_bytes: u32,
}

impl GqaBody {
    /// The per-head body, which takes any configuration.
    pub const PER_HEAD: Self = Self {
        heads: 1,
        symbol: "q4e_attn",
        shared_bytes: 0,
    };

    /// The body for `query_groups` query heads a key/value head, attending over at most
    /// `selected` tokens a query: the first of [GQA_GROUP_KERNELS] whose head count divides the
    /// groups, whose selection fits its shared memory and that `granted` (the device granted its
    /// reservation when the image loaded), else [Self::PER_HEAD]. The one rule the boot plan and
    /// every arm that launches these bodies take.
    #[must_use]
    pub fn choose(query_groups: usize, selected: usize, granted: impl Fn(usize) -> bool) -> Self {
        GQA_GROUP_KERNELS
            .iter()
            .find(|&&(heads, _)| {
                query_groups.is_multiple_of(heads)
                    && selected <= GQA_GROUP_SELECTED
                    && granted(heads)
            })
            .map_or(Self::PER_HEAD, |&(heads, symbol)| Self {
                heads,
                symbol,
                shared_bytes: gqa_group_shared_bytes(heads),
            })
    }

    /// The launch grid over `query_heads` heads and `rows` query rows: a block a head group and
    /// a row, 256 threads a block.
    #[must_use]
    pub const fn grid(
        &self,
        query_heads: usize,
        rows: usize,
    ) -> ((u32, u32, u32), (u32, u32, u32)) {
        (
            ((query_heads / self.heads) as u32, rows as u32, 1),
            (256, 1, 1),
        )
    }
}

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
        let (requirement, geometry) = Self::shape(requirement, kind, rows, context_len, head_dim)?;
        let symbols = match kind {
            AttentionKind::Gqa {
                query_heads,
                kv_heads,
            } => match GqaBody::choose(
                query_heads / kv_heads,
                gqa_selected(&requirement.geometry, context_len),
                |_| true,
            ) {
                body if body == GqaBody::PER_HEAD => &["q4e_attn"][..],
                body => &[body.symbol, "q4e_attn"][..],
            },
            _ => symbols_for(kind, requirement.precision()),
        };
        let phase = KernelPhase::for_rows(rows);
        // Resolve the geometry the plan will launch: the schedule owns the row count, context
        // and head width, and a kernel must accept those, not the requirement's original values.
        let resolved = context.resolve(&requirement, phase, symbols)?.id();
        Self::finish(Self {
            kind,
            rows,
            context: context_len,
            head_dim,
            stream,
            precision: requirement.precision(),
            geometry,
            resolved,
        })
    }

    /// The same plan around an already-resolved kernel.
    ///
    /// A caller that has resolved the part instance once (the shared rank's launcher) must not
    /// re-resolve a second, possibly different symbol here: the descriptor's kernel and the
    /// body the arm launches have to be the same. The shape and validation are those of
    /// [Self::new]; only the catalogue lookup is skipped.
    ///
    /// # Errors
    /// A requirement for another part, a zero dimension, a head count that does not divide, an
    /// unsupported precision, or a shape the kind refuses.
    pub fn with_resolved(
        requirement: &PartRequirement,
        kind: AttentionKind,
        rows: usize,
        context_len: usize,
        head_dim: usize,
        stream: StreamId,
        resolved: KernelId,
    ) -> Result<Self, OpError> {
        let (_requirement, geometry) = Self::shape(requirement, kind, rows, context_len, head_dim)?;
        Self::finish(Self {
            kind,
            rows,
            context: context_len,
            head_dim,
            stream,
            precision: requirement.precision(),
            geometry,
            resolved,
        })
    }

    /// The schedule-adjusted requirement and geometry both constructors share.
    fn shape(
        requirement: &PartRequirement,
        kind: AttentionKind,
        rows: usize,
        context_len: usize,
        head_dim: usize,
    ) -> Result<(PartRequirement, PartGeometry), OpError> {
        if requirement.part != kind.part() {
            return Err(OpError::Binding(format!(
                "{} attention was built from a {} requirement",
                kind.name(),
                requirement.part
            )));
        }
        validate_inputs(kind, rows, context_len, head_dim)?;
        let precision = requirement.precision();
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
        Ok((
            PartRequirement {
                geometry,
                ..*requirement
            },
            geometry,
        ))
    }

    fn finish(plan: Self) -> Result<Self, OpError> {
        plan.validate()?;
        Ok(plan)
    }

    /// The grouped body a GQA plan launches ([GqaBody::choose] over its query groups and the
    /// tokens a query reads), or [GqaBody::PER_HEAD] for the other kinds.
    #[must_use]
    pub fn gqa_body(&self) -> GqaBody {
        match self.kind {
            AttentionKind::Gqa { .. } => GqaBody::choose(
                self.group_size(),
                gqa_selected(&self.geometry, self.context),
                |_| true,
            ),
            _ => GqaBody::PER_HEAD,
        }
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
            Precision::MxFp8 => &["flash_mla_csa2_seq_sm121", "opt_bulk_attn"],
            Precision::Bf16 => &[
                "flash_mla_csa2_seq_sm121",
                "opt_bulk_attn",
                "glm_prefill_mla_attend",
            ],
            Precision::Fp8E4M3 => &[
                "glm_prefill_mla_attend",
                "glm_prefill_mla_attend_bf16a",
                "q4e_glm_mla_attend",
            ],
            Precision::F32 => &["opt_bulk_attn", "glm_prefill_mla_attend"],
            // No attention body reads 4-bit weights: the MLA projections are MXFP8 and the cache
            // is E4M3/mxfp8. A 4-bit operand here is a plan error, not a missing build.
            Precision::NvFp4 | Precision::MxFp4 => &[],
        },
        AttentionKind::Gqa { .. } => &["q4e_attn"], // replaced below for 4/6-head groups
        AttentionKind::Sparse { .. } => {
            &["flash_mla_csa2_seq_sm121", "q4e_qsa_pick", "q4e_glm_pick"]
        }
        AttentionKind::Csa2 { .. } => &["opt_pool_rows", "compressed_append_sm121"],
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
        if let AttentionKind::Gqa { query_heads, .. } = self.kind {
            let body = self.gqa_body();
            let (grid, block) = body.grid(query_heads, self.rows);
            return Launch {
                kernel: self.kernel(),
                grid,
                block,
                shared_bytes: body.shared_bytes,
                stream: self.stream,
            };
        }
        Launch {
            kernel: self.kernel(),
            grid: (self.rows.div_ceil(4) as u32, 1, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
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

/// The tokens a GQA query reads: its indexer's selection when the layer has one, else the
/// context.
const fn gqa_selected(geometry: &PartGeometry, context: usize) -> usize {
    if geometry.top_k > 0 {
        geometry.top_k
    } else {
        context
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
            format: crate::engine_plan::format_of_precision(attention.precision),
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

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

use crate::error::OpError;
use crate::images::Image;
use crate::plan::{KernelId, Launch, OpKind, OpPlan, Precision, StreamId};

/// The dynamic shared memory `q4e_attn_group` reserves: logits and a reduction block for each
/// of its four query heads, then the token's selection.
const GQA_GROUP_SHARED_BYTES: u32 = 45_136;

/// The dynamic shared memory `q4e_attn_group6` reserves: the same layout over six query heads.
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
}

impl AttentionPlan {
    /// A plan of the given arithmetic and shape.
    ///
    /// # Errors
    /// A zero dimension, a query/key-value head count that does not divide, or an unsupported
    /// precision.
    pub fn new(
        kind: AttentionKind,
        rows: usize,
        context: usize,
        head_dim: usize,
        precision: Precision,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        let plan = Self {
            kind,
            rows,
            context,
            head_dim,
            stream,
            precision,
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
        let symbol = match self.kind {
            AttentionKind::Mla => "flash_mla_csa2_sm121",
            AttentionKind::Gqa { .. } => gqa_kernel(self.group_size()).0,
            AttentionKind::Sparse { .. } => "flash_mla_csa2_seq_sm121",
            AttentionKind::Csa2 { .. } => "flash_mla_csa2_ring_sm121",
        };
        let image = match self.kind {
            AttentionKind::Gqa { .. } => Image::Qwen4Attn,
            _ => Image::Deepseek,
        };
        KernelId::new(image, symbol)
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
        if self.rows == 0 || self.context == 0 || self.head_dim == 0 {
            return Err(OpError::shape("attention", "a dimension is zero"));
        }
        if let AttentionKind::Gqa {
            query_heads,
            kv_heads,
        } = self.kind
            && (query_heads == 0 || kv_heads == 0 || !query_heads.is_multiple_of(kv_heads))
        {
            return Err(OpError::shape(
                "grouped-query attention",
                format!("{query_heads} query heads do not group over {kv_heads} key/value heads"),
            ));
        }
        if let AttentionKind::Sparse { selected, block } = self.kind {
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
        if let AttentionKind::Csa2 { ratio } = self.kind
            && ratio == 0
        {
            return Err(OpError::shape(
                "CSA2 attention",
                "the compression ratio is zero",
            ));
        }
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
/// group (including the 12-head group a TP=2 Qwen arm holds) runs the per-head `q4e_attn` body,
/// which needs no dynamic shared reservation.
fn gqa_kernel(group: usize) -> (&'static str, u32) {
    match group {
        6 => ("q4e_attn_group6", GQA_GROUP6_SHARED_BYTES),
        4 => ("q4e_attn_group", GQA_GROUP_SHARED_BYTES),
        _ => ("q4e_attn", 0),
    }
}

/// Split-KV decode attention: the context is cut over blocks and reduced in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitKvDecodePlan {
    /// The attention arithmetic the split reduces.
    pub attention: AttentionPlan,
    /// How many context slices the launch takes.
    pub splits: usize,
}

impl SplitKvDecodePlan {
    /// A split decode plan.
    ///
    /// # Errors
    /// As [AttentionPlan::new], and a context that does not split evenly.
    pub fn new(attention: AttentionPlan, splits: usize) -> Result<Self, OpError> {
        let plan = Self { attention, splits };
        plan.validate()?;
        Ok(plan)
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
        KernelId::new(Image::Deepseek, "opt_attn_sm121")
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

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

//! Linear-attention scans: Gated DeltaNet and Kimi Delta Attention.

use sparkflash_model::Part;

use crate::engine_plan::PartRequirement;
use crate::error::OpError;
use crate::plan::{
    KernelId, KernelPhase, Launch, OpKind, OpPlan, PartGeometry, Precision, StreamId,
};
use crate::resolver::PlanContext;

/// The GDN scan symbols for one key/value head template, most specific first.
///
/// The scan image instantiates the shared body at the three pairs in the catalogue's
/// `SCAN_HEAD_PAIRS`; the host launches `q4p_gdn_scan_64_64` and `q4p_gdn_scan_128_256` for the
/// two that are not the released 128/128 body (D98). Selecting the symbol from the launch's own
/// head widths is what keeps the plan from always naming the 128/128 template on a 64/64 launch.
/// Every list keeps `q4p_gdn_scan` and `q4e_gdn_recur` behind the template so an uninstantiated
/// pair still has the catalogue's reason (it will be refused by the entry's head-pair rule).
fn gdn_symbols(key_dim: usize, value_dim: usize) -> &'static [&'static str] {
    match (key_dim, value_dim) {
        (64, 64) => &["q4p_gdn_scan_64_64", "q4p_gdn_scan", "q4e_gdn_recur"],
        (128, 256) => &["q4p_gdn_scan_128_256", "q4p_gdn_scan", "q4e_gdn_recur"],
        _ => &["q4p_gdn_scan", "q4e_gdn_recur"],
    }
}

/// The scan's recurrence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanKind {
    /// Gated DeltaNet.
    Gdn,
    /// Kimi Delta Attention.
    Kda,
}

impl ScanKind {
    /// The model-side part the recurrence belongs to.
    #[must_use]
    pub const fn part(self) -> Part {
        match self {
            Self::Gdn => Part::GdnScan,
            Self::Kda => Part::KdaScan,
        }
    }
}

/// One scan launch over a row batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanPlan {
    /// The recurrence.
    pub kind: ScanKind,
    /// The rows to scan.
    pub rows: usize,
    /// Key or query heads.
    pub heads: usize,
    /// The key head width.
    pub key_dim: usize,
    /// The value head width.
    pub value_dim: usize,
    /// How many column slices per head the launch takes.
    pub splits: usize,
    /// The stream the plan launches on.
    pub stream: StreamId,
    /// The recurrence's precision.
    pub precision: Precision,
    /// The geometry the kernel was resolved at, including the requirement's hidden width and
    /// convolution window.
    pub geometry: PartGeometry,
    /// The catalogue-selected kernel.
    pub resolved: KernelId,
}

impl ScanPlan {
    /// A scan plan built from the requirement's geometry.
    ///
    /// The parameters mirror the kernel's launch contract; every discriminating field the
    /// catalogue checks (hidden width, key/value head template, convolution window) comes from
    /// the requirement, so a plan cannot silently accept the wrong template.
    ///
    /// # Errors
    /// A zero dimension, a split that leaves no columns or is not a power of two, an unsupported
    /// precision, a requirement for the other recurrence, or no catalogue kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: PlanContext,
        requirement: &PartRequirement,
        kind: ScanKind,
        rows: usize,
        heads: usize,
        key_dim: usize,
        value_dim: usize,
        splits: usize,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        if requirement.part != kind.part() {
            return Err(OpError::Binding(format!(
                "{kind:?} scan was built from a {} requirement",
                requirement.part
            )));
        }
        let precision = requirement.precision;
        if rows == 0 || heads == 0 || key_dim == 0 || value_dim == 0 {
            return Err(OpError::shape(
                "linear-attention scan",
                "a dimension is zero",
            ));
        }
        if splits == 0 || !splits.is_power_of_two() || !value_dim.is_multiple_of(splits) {
            return Err(OpError::shape(
                "linear-attention scan",
                format!("{value_dim} value columns do not split into {splits} power-of-two slices"),
            ));
        }
        if !matches!(
            precision,
            Precision::F32 | Precision::Bf16 | Precision::Fp8E4M3
        ) {
            return Err(OpError::Unsupported {
                op: "linear-attention scan",
                dtype: precision.name(),
            });
        }
        let mut geometry = requirement.geometry;
        geometry.rows = rows;
        geometry.heads = heads;
        geometry.head_dim = key_dim;
        geometry.v_head = value_dim;
        let symbols: &[&'static str] = match kind {
            ScanKind::Gdn => gdn_symbols(key_dim, value_dim),
            ScanKind::Kda => &["q4p_kda_scan", "q4e_kda_decay"],
        };
        // Resolve the geometry this plan will launch, not the requirement's original one: the
        // schedule's head widths are the template the kernel must accept.
        let requirement = PartRequirement {
            geometry,
            ..*requirement
        };
        let resolved = context
            .resolve(&requirement, KernelPhase::for_rows(rows), symbols)?
            .id();
        let plan = Self {
            kind,
            rows,
            heads,
            key_dim,
            value_dim,
            splits,
            stream,
            precision,
            geometry,
            resolved,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// The scan grid and block width for the head count, as the kernels choose it.
    #[must_use]
    pub fn shape(&self) -> ((u32, u32, u32), u32) {
        (
            (self.heads as u32, self.splits as u32, 1),
            (4 * self.value_dim / self.splits.max(1)) as u32,
        )
    }
}

impl OpPlan for ScanPlan {
    fn kind(&self) -> OpKind {
        match self.kind {
            ScanKind::Gdn => OpKind::GdnScan,
            ScanKind::Kda => OpKind::KdaScan,
        }
    }

    fn stream(&self) -> StreamId {
        self.stream
    }

    fn kernel(&self) -> KernelId {
        self.resolved
    }

    fn launch(&self) -> Launch {
        let (grid, block) = self.shape();
        Launch {
            kernel: self.kernel(),
            grid,
            block: (block, 1, 1),
            shared_bytes: 0,
            stream: self.stream,
        }
    }

    fn workspace_floats(&self) -> usize {
        self.rows * self.heads * self.key_dim.max(self.value_dim)
    }

    fn validate(&self) -> Result<(), OpError> {
        if self.rows == 0 || self.heads == 0 || self.key_dim == 0 || self.value_dim == 0 {
            return Err(OpError::shape(
                "linear-attention scan",
                "a dimension is zero",
            ));
        }
        if self.splits == 0
            || !self.splits.is_power_of_two()
            || !self.value_dim.is_multiple_of(self.splits)
        {
            return Err(OpError::shape(
                "linear-attention scan",
                format!(
                    "{} value columns do not split into {} power-of-two slices",
                    self.value_dim, self.splits
                ),
            ));
        }
        if !matches!(
            self.precision,
            Precision::F32 | Precision::Bf16 | Precision::Fp8E4M3
        ) {
            return Err(OpError::Unsupported {
                op: "linear-attention scan",
                dtype: self.precision.name(),
            });
        }
        Ok(())
    }
}

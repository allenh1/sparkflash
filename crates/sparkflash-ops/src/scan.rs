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

use crate::error::OpError;
use crate::images::Image;
use crate::plan::{KernelId, Launch, OpKind, OpPlan, Precision, StreamId};

/// The scan's recurrence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanKind {
    /// Gated DeltaNet.
    Gdn,
    /// Kimi Delta Attention.
    Kda,
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
}

impl ScanPlan {
    /// A scan plan.
    ///
    /// The parameters mirror the kernel's shape one-for-one, so a builder struct would only hide
    /// the launch contract.
    ///
    /// # Errors
    /// A zero dimension, a split that leaves no columns or is not a power of two, or an
    /// unsupported precision.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: ScanKind,
        rows: usize,
        heads: usize,
        key_dim: usize,
        value_dim: usize,
        splits: usize,
        precision: Precision,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        let plan = Self {
            kind,
            rows,
            heads,
            key_dim,
            value_dim,
            splits,
            stream,
            precision,
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
        let symbol = match self.kind {
            ScanKind::Gdn => "q4p_gdn_scan",
            ScanKind::Kda => "q4p_kda_scan",
        };
        KernelId::new(Image::Qwen4Scan, symbol)
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
        if !matches!(self.precision, Precision::F32 | Precision::Bf16) {
            return Err(OpError::Unsupported {
                op: "linear-attention scan",
                dtype: self.precision.name(),
            });
        }
        Ok(())
    }
}

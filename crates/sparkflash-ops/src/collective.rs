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

//! Collectives: one-shot and two-shot all-reduce, reduce-scatter, all-gather and the
//! overlapped-halves runner.

use sparkflash_model::Part;

use crate::engine_plan::PartRequirement;
use crate::error::OpError;
use crate::plan::{
    KernelId, KernelPhase, Launch, OpKind, OpPlan, PartGeometry, Precision, StreamId,
};
use crate::resolver::PlanContext;

/// The collective a plan runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectiveKind {
    /// A one-shot all-reduce over the whole payload.
    OneShot,
    /// A two-shot all-reduce: reduce-scatter then all-gather.
    TwoShot,
    /// A reduce-scatter.
    ReduceScatter,
    /// An all-gather.
    AllGather,
    /// Two independent halves overlapped.
    OverlappedHalves,
}

impl CollectiveKind {
    /// The collective's stable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::OneShot => "one-shot",
            Self::TwoShot => "two-shot",
            Self::ReduceScatter => "reduce-scatter",
            Self::AllGather => "all-gather",
            Self::OverlappedHalves => "overlapped-halves",
        }
    }
}

/// A collective plan over a typed payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectivePlan {
    /// The collective.
    pub kind: CollectiveKind,
    /// Payload elements.
    pub elements: usize,
    /// Elements in one value, in bytes.
    pub element_bytes: usize,
    /// The ranks participating.
    pub world: usize,
    /// The stream the plan launches on.
    pub stream: StreamId,
    /// The precision the collective body was resolved at.
    pub precision: Precision,
    /// The geometry the kernel was resolved at.
    pub geometry: PartGeometry,
    /// The catalogue-selected kernel.
    pub resolved: KernelId,
}

impl CollectivePlan {
    /// A collective plan built from a Collective requirement.
    ///
    /// # Errors
    /// A zero dimension, a world of one, a payload that does not split for a two-shot form, a
    /// kind with no catalogued kernel, or no catalogue entry for the given world.
    pub fn new(
        context: PlanContext,
        requirement: &PartRequirement,
        kind: CollectiveKind,
        elements: usize,
        element_bytes: usize,
        world: usize,
        stream: StreamId,
    ) -> Result<Self, OpError> {
        if requirement.part != Part::Collective {
            return Err(OpError::Binding(format!(
                "a collective was built from a {} requirement",
                requirement.part
            )));
        }
        let plan_inputs = Self {
            kind,
            elements,
            element_bytes,
            world,
            stream,
            precision: requirement.precision,
            geometry: PartGeometry {
                world,
                ..requirement.geometry
            },
            resolved: KernelId::new(context.target(), crate::images::Image::Rocenante, ""),
        };
        plan_inputs.validate()?;
        let symbols: &[&'static str] = match kind {
            CollectiveKind::OneShot => &["rocenante_oneshot_allreduce_bf16_tp4"],
            CollectiveKind::TwoShot | CollectiveKind::OverlappedHalves => &["q4e_sum_slots"],
            CollectiveKind::ReduceScatter | CollectiveKind::AllGather => {
                // Neither form has a catalogued kernel: the names the old hardcoded plan used
                // were split-K GEMM symbols, not collectives. Refuse rather than misdescribe
                // them.
                return Err(OpError::Unsupported {
                    op: kind.name(),
                    dtype: "no catalogued collective kernel",
                });
            }
        };
        let resolved = context.resolve_any(
            &PartRequirement {
                part: Part::Collective,
                geometry: plan_inputs.geometry,
                precision: requirement.precision,
            },
            KernelPhase::Both,
            symbols,
        )?;
        Ok(Self {
            precision: resolved.precision,
            resolved: resolved.id(),
            ..plan_inputs
        })
    }

    /// The bytes the payload moves.
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.elements * self.element_bytes
    }

    /// Whether the payload splits into whole float4 chunks over the world, the two-shot
    /// collective's contract.
    #[must_use]
    pub fn two_shot_splits(&self) -> bool {
        self.world > 1 && self.elements.is_multiple_of(4 * self.world)
    }
}

impl OpPlan for CollectivePlan {
    fn kind(&self) -> OpKind {
        match self.kind {
            CollectiveKind::OneShot => OpKind::OneShotCollective,
            CollectiveKind::TwoShot => OpKind::TwoShotCollective,
            CollectiveKind::ReduceScatter => OpKind::ReduceScatter,
            CollectiveKind::AllGather => OpKind::AllGather,
            CollectiveKind::OverlappedHalves => OpKind::OverlappedHalves,
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
            grid: (self.world as u32, 1, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
            stream: self.stream,
        }
    }

    fn workspace_floats(&self) -> usize {
        match self.kind {
            CollectiveKind::TwoShot | CollectiveKind::OverlappedHalves => self.elements.div_ceil(4),
            _ => 0,
        }
    }

    fn validate(&self) -> Result<(), OpError> {
        if self.elements == 0 || self.element_bytes == 0 {
            return Err(OpError::shape("collective", "the payload is empty"));
        }
        if self.world < 2 {
            return Err(OpError::shape(
                "collective",
                format!("a world of {} has nothing to collect", self.world),
            ));
        }
        if matches!(
            self.kind,
            CollectiveKind::TwoShot | CollectiveKind::ReduceScatter | CollectiveKind::AllGather
        ) && !self.elements.is_multiple_of(self.world)
        {
            return Err(OpError::shape(
                "collective",
                format!(
                    "{} elements do not split over {} ranks",
                    self.elements, self.world
                ),
            ));
        }
        if self.kind == CollectiveKind::TwoShot && !self.two_shot_splits() {
            return Err(OpError::shape(
                "two-shot collective",
                format!(
                    "{} elements do not split into float4 chunks over {} ranks",
                    self.elements, self.world
                ),
            ));
        }
        Ok(())
    }
}

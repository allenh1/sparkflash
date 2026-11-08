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

//! The shared `EnginePlan`-driven rank and backend wrapper.
//!
//! [PartsDeviceRank] is the part of a device backend every arm owns identically: the
//! [EnginePlan], the rank-local [ArchitectureTensorPlan], the catalogue's resolved kernel for
//! every part instance, and the [LayerLauncher]s composed from them. An arm keeps its own
//! numeric forward (which is where its kernel-specific argument layouts live) and wraps it in
//! [PartsDeviceBackend], so a boot from `EngineBuilder` proves the shared plan and the arm's
//! execution describe the same checkpoint.

use sparkflash_model::{ModelBackend, ModelDescriptor, ModelSession, Part};
use sparkflash_weights::WeightSource;

use crate::engine_plan::{EnginePlan, PlanError};
use crate::launchers::{
    AttentionLauncher, LayerLauncher, LinearScanLauncher, MoeLauncher, ResidualLauncher,
    SparseIndexerLauncher,
};
use crate::plan::StreamId;
use crate::resolver::{PlanContext, ResolveError, ResolvedKernel};
use crate::tensor_plan::ArchitectureTensorPlan;

/// One decoder layer's launchers, in the shared walk's order.
#[derive(Debug, Clone)]
pub struct PartsLayer {
    /// The checkpoint layer index.
    pub index: usize,
    /// The layer's parts, in forward order.
    pub launchers: Vec<LayerLauncher>,
}

/// The rank-local execution plan every arm shares.
#[derive(Debug, Clone)]
pub struct PartsDeviceRank {
    plan: EnginePlan,
    tensors: ArchitectureTensorPlan,
    kernels: Vec<ResolvedKernel>,
    unresolved: Vec<ResolveError>,
    skipped: Vec<Part>,
    globals: Vec<LayerLauncher>,
    layers: Vec<PartsLayer>,
}

impl PartsDeviceRank {
    /// Open the shared rank view of `plan` over `weights`.
    ///
    /// The phase-aware serving boot check runs first, for both prefill and decode: an arm must
    /// not start uploading weights for a plan whose parts have no body in one of the passes. The
    /// weight source is then checked against the plan before any kernel is resolved: every
    /// logical tensor must be present with the planned dtype and shape, or stored as its NVFP4
    /// packing ([TensorSpec::is_stored_as]), so an arm that later slices it has the bytes it
    /// expects.
    ///
    /// [TensorSpec::is_stored_as]: sparkflash_weights::TensorSpec::is_stored_as
    ///
    /// # Errors
    /// Every served part instance with no catalogue kernel in either pass, aggregated; then a
    /// weight the plan names and the source lacks (or stores wrongly).
    pub fn open(
        plan: &EnginePlan,
        weights: &dyn WeightSource,
        context: PlanContext,
    ) -> Result<Self, PlanError> {
        let resolver = context.resolver();
        // The language-only serving set: every released document plans a vision tower, but no
        // arm in this tree stages one, so the boot gate must not demand a tower body.
        let serving = crate::parts::serving_language(context.target());
        if let Err(errors) = resolver.boot_check_serving(plan, &serving) {
            let reasons: Vec<String> = errors.iter().map(ToString::to_string).collect();
            return Err(PlanError::Invalid(format!(
                "the serving boot check refuses {} part instance(s) in the prefill and decode \
                 passes: {}",
                errors.len(),
                reasons.join("; ")
            )));
        }
        let tensors = ArchitectureTensorPlan::served_and_mtp(plan.architecture(), plan.tp())?;
        for spec in tensors.specs() {
            let stored = weights.spec(&spec.name).map_err(|error| {
                PlanError::Invalid(format!(
                    "{}: the execution plan names it, the weight source does not ({error})",
                    spec.name
                ))
            })?;
            if !spec.is_stored_as(&stored) {
                return Err(PlanError::Invalid(format!(
                    "{} is stored as {:?} {:?}, the plan expects {:?} {:?}",
                    spec.name, stored.dtype, stored.shape, spec.dtype, spec.shape
                )));
            }
        }
        let mut slots: Vec<Option<ResolvedKernel>> = Vec::new();
        let mut unresolved = Vec::new();
        let mut skipped = Vec::new();
        for requirement in plan.requirements() {
            if !serving.contains(requirement.part) {
                skipped.push(requirement.part);
                slots.push(None);
                continue;
            }
            match resolver.resolve(requirement) {
                Ok(kernel) => slots.push(Some(kernel)),
                Err(error) => {
                    unresolved.push(error);
                    slots.push(None);
                }
            }
        }
        let kernels: Vec<ResolvedKernel> = slots.iter().flatten().cloned().collect();
        let mut cursor = 0usize;
        let mut globals = Vec::new();
        for requirement in plan.globals() {
            match slots.get(cursor).and_then(Option::as_ref) {
                Some(kernel) => globals.push(LayerLauncher::Other(kernel.id())),
                None => globals.push(LayerLauncher::Unresolved(requirement.part)),
            }
            cursor += 1;
        }
        let mut layers = Vec::with_capacity(plan.layers().len());
        for layer in plan.layers() {
            let mut launchers = Vec::with_capacity(layer.requirements.len());
            for requirement in &layer.requirements {
                match slots.get(cursor).and_then(Option::as_ref) {
                    Some(resolved) => launchers.push(launcher_for(requirement, resolved)?),
                    None => launchers.push(LayerLauncher::Unresolved(requirement.part)),
                }
                cursor += 1;
            }
            layers.push(PartsLayer {
                index: layer.index,
                launchers,
            });
        }
        Ok(Self {
            plan: plan.clone(),
            tensors,
            kernels,
            unresolved,
            skipped,
            globals,
            layers,
        })
    }

    /// The execution plan.
    #[must_use]
    pub const fn plan(&self) -> &EnginePlan {
        &self.plan
    }

    /// The shared tensor plan at this rank.
    #[must_use]
    pub const fn tensors(&self) -> &ArchitectureTensorPlan {
        &self.tensors
    }

    /// Every resolved part instance, in plan order.
    #[must_use]
    pub fn kernels(&self) -> &[ResolvedKernel] {
        &self.kernels
    }

    /// The requirements no catalogue kernel serves at this geometry.
    #[must_use]
    pub fn unresolved(&self) -> &[ResolveError] {
        &self.unresolved
    }

    /// The plan parts this build does not serve (the vision tower today).
    #[must_use]
    pub fn skipped(&self) -> &[Part] {
        &self.skipped
    }

    /// Refuse a rank some part of which this build cannot execute.
    ///
    /// # Errors
    /// The first unresolved requirement, with every candidate's reason.
    pub fn boot_check(&self) -> Result<(), PlanError> {
        if self.unresolved.is_empty() {
            return Ok(());
        }
        let reasons: Vec<String> = self.unresolved.iter().map(ToString::to_string).collect();
        Err(PlanError::Invalid(format!(
            "{} part instance(s) have no kernel: {}",
            reasons.len(),
            reasons.join("; ")
        )))
    }

    /// The global parts' kernels.
    #[must_use]
    pub fn globals(&self) -> &[LayerLauncher] {
        &self.globals
    }

    /// The decoder layers' launchers.
    #[must_use]
    pub fn layers(&self) -> &[PartsLayer] {
        &self.layers
    }

    /// A one-line description of what this rank will execute.
    #[must_use]
    pub fn boot_summary(&self) -> String {
        format!(
            "{}; {} tensors, {} bytes at this rank, {} kernels, {} unresolved",
            self.plan.boot_summary(),
            self.tensors.len(),
            self.tensors.rank_bytes(),
            self.kernels.len(),
            self.unresolved.len()
        )
    }
}

fn launcher_for(
    requirement: &crate::engine_plan::PartRequirement,
    resolved: &ResolvedKernel,
) -> Result<LayerLauncher, PlanError> {
    Ok(match requirement.part {
        Part::GdnScan | Part::KdaScan => {
            LayerLauncher::LinearScan(LinearScanLauncher::new(requirement, resolved)?)
        }
        Part::GqaAttention | Part::MlaAttention => {
            LayerLauncher::Attention(AttentionLauncher::new(
                requirement,
                resolved,
                1,
                requirement.geometry.positions.max(1),
                StreamId(0),
            )?)
        }
        Part::SparseIndexer => LayerLauncher::Indexer(SparseIndexerLauncher::new(
            requirement,
            resolved,
            requirement.geometry.rows.max(1),
            requirement.geometry.positions.max(1),
        )?),
        Part::Moe => LayerLauncher::Moe(MoeLauncher::new(requirement, resolved)?),
        Part::HyperConnection | Part::ManifoldHyperConnection => {
            LayerLauncher::Residual(ResidualLauncher::new(requirement, resolved)?)
        }
        _ => LayerLauncher::Other(resolved.id()),
    })
}

/// The shared backend wrapper: the arm's numeric forward behind the common plan and launchers.
#[derive(Debug, Clone)]
pub struct PartsDeviceBackend<B> {
    rank: PartsDeviceRank,
    arm: B,
}

impl<B> PartsDeviceBackend<B> {
    /// Wrap an arm backend that has been opened for the same plan.
    #[must_use]
    pub const fn new(rank: PartsDeviceRank, arm: B) -> Self {
        Self { rank, arm }
    }

    /// The shared rank view.
    #[must_use]
    pub const fn rank(&self) -> &PartsDeviceRank {
        &self.rank
    }

    /// The arm's numeric backend.
    #[must_use]
    pub const fn arm(&self) -> &B {
        &self.arm
    }

    /// The arm's numeric backend, mutably.
    #[must_use]
    pub const fn arm_mut(&mut self) -> &mut B {
        &mut self.arm
    }

    /// Unwrap the arm, discarding the shared rank view.
    ///
    /// An arm whose serving driver already holds its own typed backend uses this after opening
    /// through [crate::engine_plan::EngineBuilder] so the boot proof runs without changing the
    /// driver's backend type.
    #[must_use]
    pub fn into_arm(self) -> B {
        self.arm
    }

    /// Unwrap both halves: the shared rank view and the arm's numeric backend.
    #[must_use]
    pub fn into_parts(self) -> (PartsDeviceRank, B) {
        (self.rank, self.arm)
    }
}

impl<B: ModelBackend> ModelBackend for PartsDeviceBackend<B> {
    fn descriptor(&self) -> &ModelDescriptor {
        self.arm.descriptor()
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, sparkflash_model::BackendError> {
        self.arm.open_session()
    }
}

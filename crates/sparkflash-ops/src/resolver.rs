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

//! The kernel resolver: [Part] + target + shape + precision + pass -> a loadable kernel.
//!
//! The catalogue answers "which symbols exist in this tree". The resolver answers the question
//! a boot path actually needs: "which of them can this process launch for this checkpoint, on
//! this target, at this rank's shapes, in this pass?" A part with no matching entry is refused
//! by name, with every candidate's reason, before a weight is read.
//!
//! Availability is an explicit policy:
//!
//! - [Availability::Catalogue] accepts every catalogued implementation (planning and tests);
//! - [Availability::Embedded] accepts only images this build embedded;
//! - [Availability::Runtime] adds the images an arm supplies or compiles at run time, and
//!   carries whether the run-time cubin directory is present.
//!
//! [Availability::loadable] delegates to [crate::parts::image_loadable], so the resolver and the
//! catalogue's serving answer cannot disagree about whether an image is loadable.
//!
//! Precision and phase are per symbol. An entry is eligible only when the symbol the plan will
//! load accepts the requested precision and serves the requested pass; a bf16 GEMM in the same
//! image as an NVFP4 quantiser does not make the image an NVFP4 GEMM.

use std::fmt;

use sparkflash_kernels::images::Image;
use sparkflash_model::parts::PartSet;
use sparkflash_model::{Part, Target};

use crate::engine_plan::{EnginePlan, PartRequirement};
use crate::error::OpError;
use crate::parts::{self, CATALOGUE};
use crate::plan::{KernelId, KernelPhase, KernelSymbol, PartGeometry, Precision};

/// Which implementations a resolver treats as loadable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Availability {
    /// Every catalogue entry, whether or not this build embeds its image.
    Catalogue,
    /// Only images embedded in this binary.
    Embedded,
    /// Embedded images plus those an arm supplies or compiles at run time.
    Runtime {
        /// Whether SPARKFLASH_CUBINS names a non-empty directory.
        cubin_dir: bool,
    },
}

impl Availability {
    /// The run-time policy with or without a cubin directory.
    #[must_use]
    pub const fn runtime(cubin_dir: bool) -> Self {
        Self::Runtime { cubin_dir }
    }

    /// The run-time policy for this process, reading SPARKFLASH_CUBINS.
    #[must_use]
    pub fn runtime_env() -> Self {
        let cubin_dir =
            std::env::var(crate::parts::CUBIN_DIR_ENV).is_ok_and(|value| !value.trim().is_empty());
        Self::runtime(cubin_dir)
    }

    /// Whether the run-time cubin directory is configured, if this policy consults it.
    #[must_use]
    pub const fn cubin_dir(self) -> Option<bool> {
        match self {
            Self::Runtime { cubin_dir } => Some(cubin_dir),
            Self::Catalogue | Self::Embedded => None,
        }
    }

    /// Whether image is loadable under this policy.
    ///
    /// [Availability::Catalogue] accepts everything; the other policies share
    /// [crate::parts::image_loadable] with the catalogue's own serving answer.
    #[must_use]
    pub fn loadable(self, image: Image) -> bool {
        match self {
            Self::Catalogue => true,
            Self::Embedded => image.available(),
            Self::Runtime { cubin_dir } => parts::image_loadable(image, cubin_dir),
        }
    }

    /// The policy's stable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Catalogue => "catalogue",
            Self::Embedded => "embedded",
            Self::Runtime { .. } => "runtime",
        }
    }
}

/// A part resolved to one catalogue implementation and one of its symbols.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedKernel {
    /// The model-side part.
    pub part: Part,
    /// The target the implementation runs on.
    pub target: Target,
    /// The image that carries the symbol.
    pub image: Image,
    /// The symbol that will be launched, with its own precision mask and pass.
    pub symbol: KernelSymbol,
    /// Whether this is the canonical implementation for the part and target.
    pub canonical: bool,
    /// The rank-local geometry that was resolved.
    pub geometry: PartGeometry,
    /// The precision that was resolved.
    pub precision: Precision,
}

impl ResolvedKernel {
    /// The pass the selected symbol belongs to.
    #[must_use]
    pub const fn phase(&self) -> KernelPhase {
        self.symbol.phase
    }

    /// The selected symbol's name.
    #[must_use]
    pub const fn symbol_name(&self) -> &'static str {
        self.symbol.name
    }

    /// The kernel identity a launch names.
    #[must_use]
    pub const fn id(&self) -> KernelId {
        KernelId::new(self.target, self.image, self.symbol.name)
    }
}

/// A part no implementation could serve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolveError {
    /// The part.
    pub part: Part,
    /// The target.
    pub target: Target,
    /// The requested precision.
    pub precision: Precision,
    /// The requested pass.
    pub phase: KernelPhase,
    /// Why each candidate was refused, or that there were no candidates.
    pub reasons: Vec<String>,
}

impl fmt::Display for ResolveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} on {} at {} in the {} pass: {}",
            self.part,
            self.target,
            self.precision.name(),
            self.phase.name(),
            if self.reasons.is_empty() {
                "no catalogue entry".to_owned()
            } else {
                self.reasons.join("; ")
            }
        )
    }
}

impl std::error::Error for ResolveError {}

/// The target and availability a typed plan is built against.
///
/// A plan constructor takes a [PlanContext] instead of hardcoding Sm121, and every kernel it
/// resolves carries the target in its [KernelId]. A later EngineBuilder can therefore prove that
/// the kernel it is about to load was resolved for the device it opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanContext {
    target: Target,
    availability: Availability,
}

impl PlanContext {
    /// A context for target under availability.
    #[must_use]
    pub const fn new(target: Target, availability: Availability) -> Self {
        Self {
            target,
            availability,
        }
    }

    /// The planning context for target: every catalogued entry is eligible.
    #[must_use]
    pub const fn catalogue(target: Target) -> Self {
        Self::new(target, Availability::Catalogue)
    }

    /// The planning context for the one target this tree builds today.
    pub const SM121_CATALOGUE: Self = Self::catalogue(Target::Sm121);

    /// The context's target.
    #[must_use]
    pub const fn target(&self) -> Target {
        self.target
    }

    /// The context's availability policy.
    #[must_use]
    pub const fn availability(&self) -> Availability {
        self.availability
    }

    /// The resolver this context describes.
    #[must_use]
    pub const fn resolver(&self) -> KernelResolver {
        KernelResolver::new(self.target, self.availability)
    }

    /// Resolve one requirement to a symbol whose own mask and pass accept the request.
    ///
    /// # Errors
    /// [OpError::Resolve] naming every candidate's reason.
    pub fn resolve(
        &self,
        requirement: &PartRequirement,
        phase: KernelPhase,
        symbols: &[&'static str],
    ) -> Result<ResolvedKernel, OpError> {
        self.resolver()
            .resolve_symbol(requirement, phase, symbols)
            .map_err(OpError::Resolve)
    }

    /// Resolve a requirement that has no stored weight format, trying each precision.
    ///
    /// The returned kernel names the precision that matched, so the caller can record it rather
    /// than silently reporting the first (f32) failure.
    ///
    /// # Errors
    /// [OpError::Resolve] when no precision the plan may use is served, with every reason
    /// aggregated.
    pub fn resolve_any(
        &self,
        requirement: &PartRequirement,
        phase: KernelPhase,
        symbols: &[&'static str],
    ) -> Result<ResolvedKernel, OpError> {
        let resolver = self.resolver();
        let mut reasons = Vec::new();
        for precision in [
            Precision::Bf16,
            Precision::F32,
            Precision::MxFp8,
            Precision::Fp8E4M3,
            Precision::NvFp4,
        ] {
            let candidate = PartRequirement {
                precision,
                ..*requirement
            };
            match resolver.resolve_symbol(&candidate, phase, symbols) {
                Ok(resolved) => return Ok(resolved),
                Err(error) => {
                    for reason in error.reasons {
                        reasons.push(format!("[{}] {reason}", precision.name()));
                    }
                }
            }
        }
        Err(OpError::Resolve(ResolveError {
            part: requirement.part,
            target: self.target,
            precision: requirement.precision,
            phase,
            reasons,
        }))
    }
}

/// Resolves execution-plan parts to loadable catalogue entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelResolver {
    target: Target,
    availability: Availability,
}

impl KernelResolver {
    /// A resolver for target with the given availability policy.
    #[must_use]
    pub const fn new(target: Target, availability: Availability) -> Self {
        Self {
            target,
            availability,
        }
    }

    /// The target.
    #[must_use]
    pub const fn target(&self) -> Target {
        self.target
    }

    /// The availability policy.
    #[must_use]
    pub const fn availability(&self) -> Availability {
        self.availability
    }

    /// Resolve one requirement without constraining the pass.
    ///
    /// # Errors
    /// No catalogued symbol for the part/target is loadable and accepts the shape and precision.
    /// The error lists every candidate's reason.
    pub fn resolve(&self, requirement: &PartRequirement) -> Result<ResolvedKernel, ResolveError> {
        self.resolve_for(requirement, KernelPhase::Both)
    }

    /// Resolve one requirement for a specific pass.
    ///
    /// KernelPhase::Both is the unconstrained form [Self::resolve] uses. A model with separate
    /// prefill and decode images asks for each phase, so a boot check proves both bodies exist.
    ///
    /// # Errors
    /// As [Self::resolve], with the phase added to the candidate reasons.
    pub fn resolve_for(
        &self,
        requirement: &PartRequirement,
        phase: KernelPhase,
    ) -> Result<ResolvedKernel, ResolveError> {
        self.resolve_filtered(requirement, phase, None)
    }

    /// Resolve one requirement to an entry that exports at least one of symbols.
    ///
    /// The symbols are a priority list: the resolver prefers the first requested symbol that some
    /// entry exports with a matching precision and pass, so a plan whose arithmetic needs a
    /// specific body names that body first and does not merely accept any symbol in the image.
    ///
    /// # Errors
    /// As [Self::resolve], with a reason for every image that exports none of symbols with the
    /// requested precision and pass.
    pub fn resolve_symbol(
        &self,
        requirement: &PartRequirement,
        phase: KernelPhase,
        symbols: &[&'static str],
    ) -> Result<ResolvedKernel, ResolveError> {
        self.resolve_filtered(requirement, phase, Some(symbols))
    }

    fn resolve_filtered(
        &self,
        requirement: &PartRequirement,
        phase: KernelPhase,
        symbols: Option<&[&'static str]>,
    ) -> Result<ResolvedKernel, ResolveError> {
        let mut candidates: Vec<&crate::parts::PartKernel> = CATALOGUE
            .iter()
            .filter(|entry| entry.part == requirement.part && entry.target == self.target)
            .collect();
        // Score the canonical entry ahead of the others; the stable sort keeps catalogue order
        // within each group. This is what makes the canonical flag more than documentation.
        candidates.sort_by_key(|entry| !entry.canonical);
        if candidates.is_empty() {
            return Err(ResolveError {
                part: requirement.part,
                target: self.target,
                precision: requirement.precision,
                phase,
                reasons: Vec::new(),
            });
        }
        let mut reasons = Vec::new();
        match symbols {
            Some(symbols) => {
                for name in symbols {
                    for entry in &candidates {
                        match self.try_entry(entry, requirement, phase, Some(name)) {
                            Ok(symbol) => return Ok(self.resolved(entry, symbol, requirement)),
                            Err(reason) => reasons.push(reason),
                        }
                    }
                }
            }
            None => {
                for entry in &candidates {
                    match self.try_entry(entry, requirement, phase, None) {
                        Ok(symbol) => return Ok(self.resolved(entry, symbol, requirement)),
                        Err(reason) => reasons.push(reason),
                    }
                }
            }
        }
        Err(ResolveError {
            part: requirement.part,
            target: self.target,
            precision: requirement.precision,
            phase,
            reasons,
        })
    }

    fn resolved(
        &self,
        entry: &crate::parts::PartKernel,
        symbol: KernelSymbol,
        requirement: &PartRequirement,
    ) -> ResolvedKernel {
        ResolvedKernel {
            part: entry.part,
            target: entry.target,
            image: entry.image,
            symbol,
            canonical: entry.canonical,
            geometry: requirement.geometry,
            precision: requirement.precision,
        }
    }

    fn try_entry(
        &self,
        entry: &crate::parts::PartKernel,
        requirement: &PartRequirement,
        phase: KernelPhase,
        name: Option<&'static str>,
    ) -> Result<KernelSymbol, String> {
        if !self.availability.loadable(entry.image) {
            return Err(format!(
                "{} is not loadable under {} availability",
                entry.image.name(),
                self.availability.name()
            ));
        }
        if let Err(reason) = entry.shape.accepts(&requirement.geometry) {
            return Err(format!("{}: {reason}", entry.image.name()));
        }
        let found = entry.symbols.iter().copied().find(|symbol| {
            name.is_none_or(|name| symbol.name == name)
                && symbol.accepts(requirement.precision)
                && symbol.phase.serves(phase)
        });
        match found {
            Some(symbol) => Ok(symbol),
            None => Err(match name {
                Some(name) => format!(
                    "{} exports no {name} accepting {} in the {} pass",
                    entry.image.name(),
                    requirement.precision.name(),
                    phase.name()
                ),
                None => format!(
                    "{} exports no symbol accepting {} in the {} pass",
                    entry.image.name(),
                    requirement.precision.name(),
                    phase.name()
                ),
            }),
        }
    }

    /// Resolve every part of a plan, in plan order, without constraining the pass.
    ///
    /// # Errors
    /// Every unresolved part, not merely the first, so a bring-up log names the whole gap.
    pub fn resolve_all(&self, plan: &EnginePlan) -> Result<Vec<ResolvedKernel>, Vec<ResolveError>> {
        self.resolve_all_for(plan, KernelPhase::Both)
    }

    /// Resolve every part of a plan for one pass, in plan order.
    ///
    /// # Errors
    /// Every unresolved part, not merely the first.
    pub fn resolve_all_for(
        &self,
        plan: &EnginePlan,
        phase: KernelPhase,
    ) -> Result<Vec<ResolvedKernel>, Vec<ResolveError>> {
        let mut resolved = Vec::new();
        let mut errors = Vec::new();
        for requirement in plan.requirements() {
            match self.resolve_for(requirement, phase) {
                Ok(kernel) => resolved.push(kernel),
                Err(error) => errors.push(error),
            }
        }
        if errors.is_empty() {
            Ok(resolved)
        } else {
            Err(errors)
        }
    }

    /// The boot check: prove that every requirement resolves in both passes.
    ///
    /// The result names every requirement instance once per pass: the prefill pass in plan order,
    /// then the decode pass in plan order. Nothing is deduplicated, because the point of the
    /// check is that every instance of every part has a body in both passes.
    ///
    /// # Errors
    /// Every unresolved part instance in either pass.
    pub fn boot_check(&self, plan: &EnginePlan) -> Result<Vec<ResolvedKernel>, Vec<ResolveError>> {
        self.boot_check_where(plan, |_| true)
    }

    /// The serving boot check: prove both passes for the parts the arm actually executes.
    ///
    /// A plan describes the checkpoint, not the process. Every released document names a vision
    /// tower (D84): DeepSeek loads and runs the 14-patch tower, while the Qwen and GLM packs do
    /// not load theirs, so gating a serve on the whole plan refuses a 16-patch document whose
    /// language parts all resolve. Restricting the check to served keeps [Self::boot_check]'s
    /// whole-plan answer - and the catalogue's honest 14-patch refusal with it - while the
    /// serving gate demands a body only for the parts it launches.
    /// [crate::parts::serving] is the full serving set and [crate::parts::serving_language]
    /// the language-only set the Qwen and GLM packs use.
    ///
    /// # Errors
    /// Every unresolved part instance of a served part in either pass, in plan order.
    pub fn boot_check_serving(
        &self,
        plan: &EnginePlan,
        served: &PartSet,
    ) -> Result<Vec<ResolvedKernel>, Vec<ResolveError>> {
        self.boot_check_where(plan, |requirement| served.contains(requirement.part))
    }

    fn boot_check_where(
        &self,
        plan: &EnginePlan,
        served: impl Fn(&PartRequirement) -> bool,
    ) -> Result<Vec<ResolvedKernel>, Vec<ResolveError>> {
        let mut resolved = Vec::new();
        let mut errors = Vec::new();
        for phase in [KernelPhase::Prefill, KernelPhase::Decode] {
            for requirement in plan
                .requirements()
                .filter(|requirement| served(requirement))
            {
                match self.resolve_for(requirement, phase) {
                    Ok(kernel) => resolved.push(kernel),
                    Err(error) => errors.push(error),
                }
            }
        }
        if errors.is_empty() {
            Ok(resolved)
        } else {
            Err(errors)
        }
    }

    /// The unresolved parts of a plan, in plan order, without constraining the pass.
    #[must_use]
    pub fn failures(&self, plan: &EnginePlan) -> Vec<ResolveError> {
        self.failures_for(plan, KernelPhase::Both)
    }

    /// The unresolved parts of a plan for one pass, in plan order.
    #[must_use]
    pub fn failures_for(&self, plan: &EnginePlan, phase: KernelPhase) -> Vec<ResolveError> {
        plan.requirements()
            .filter_map(|requirement| self.resolve_for(requirement, phase).err())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::ShapeConstraint;

    fn requirement(part: Part, geometry: PartGeometry, precision: Precision) -> PartRequirement {
        PartRequirement {
            part,
            geometry,
            precision,
        }
    }

    fn deepseek_moe() -> PartRequirement {
        requirement(
            Part::Moe,
            PartGeometry {
                hidden: 5120,
                rows: 1,
                intermediate: 576,
                heads: 16,
                kv_heads: 2,
                head_dim: 128,
                experts: 384,
                top_k: 6,
                positions: 131_072,
                vocab: 129_280,
                world: 4,
                rank: 0,
                ..PartGeometry::new()
            },
            Precision::NvFp4,
        )
    }

    #[test]
    fn a_constraint_checks_the_fields_it_names_and_refuses_unknowns() {
        let constraint = ShapeConstraint::ANY
            .hidden_multiple(128)
            .head_dim_multiple(16)
            .gqa_groups(&[1, 2, 4, 6, 8]);
        let okay = PartGeometry {
            hidden: 5120,
            heads: 8,
            kv_heads: 2,
            head_dim: 128,
            ..PartGeometry::new()
        };
        constraint.accepts(&okay).expect("valid");

        let bad_head = PartGeometry {
            head_dim: 12,
            ..okay
        };
        assert!(constraint.accepts(&bad_head).is_err());
        let bad_group = PartGeometry {
            heads: 10,
            kv_heads: 2,
            ..okay
        };
        assert!(constraint.accepts(&bad_group).is_err());
        // A specific entry may not win on a geometry that did not populate the field.
        let unknown = PartGeometry::new();
        assert!(constraint.accepts(&unknown).is_err());
        assert!(ShapeConstraint::ANY.accepts(&unknown).is_ok());
    }

    #[test]
    fn a_catalogue_entry_resolves_for_its_target() {
        let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
        let resolved = resolver
            .resolve(&deepseek_moe())
            .expect("the catalogue carries MoE");
        assert_eq!(resolved.part, Part::Moe);
        assert_eq!(resolved.image, Image::Deepseek);
        assert_eq!(resolved.precision, Precision::NvFp4);
        assert_eq!(resolved.phase(), KernelPhase::Both);
        assert!(resolved.canonical);
        assert_eq!(resolved.id().target, Target::Sm121);
    }

    #[test]
    fn a_symbol_must_accept_the_precision_itself() {
        let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
        // The Qwen token image exports q4e_gemm_bf16, but it is not an NVFP4 dense GEMM.
        let requirement = requirement(
            Part::DenseMlp,
            PartGeometry {
                hidden: 2560,
                rows: 1,
                intermediate: 640,
                positions: 1,
                world: 1,
                rank: 0,
                ..PartGeometry::new()
            },
            Precision::NvFp4,
        );
        let error = resolver
            .resolve_symbol(&requirement, KernelPhase::Both, &["q4e_gemm_bf16"])
            .expect_err("q4e_gemm_bf16 is not an NVFP4 body");
        assert!(
            error
                .reasons
                .iter()
                .any(|reason| reason.contains("no q4e_gemm_bf16 accepting nvfp4")),
            "{error}"
        );
    }

    #[test]
    fn the_phase_is_a_property_of_the_symbol() {
        let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
        let glm = requirement(
            Part::DenseMlp,
            PartGeometry {
                hidden: 4096,
                rows: 16,
                intermediate: 3072,
                positions: 4096,
                world: 1,
                rank: 0,
                ..PartGeometry::new()
            },
            Precision::Fp8E4M3,
        );
        assert_eq!(
            resolver
                .resolve_for(&glm, KernelPhase::Prefill)
                .expect("prefill")
                .image,
            Image::GlmPrefill
        );
        assert_eq!(
            resolver
                .resolve_for(&glm, KernelPhase::Decode)
                .expect("decode")
                .image,
            Image::GlmDecode
        );
        assert_eq!(
            resolver.resolve(&glm).expect("either").image,
            Image::GlmPrefill
        );
    }

    #[test]
    fn a_deepseek_moe_at_tp1_is_refused_by_its_intermediate_width() {
        let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
        let mut requirement = deepseek_moe();
        requirement.geometry.intermediate = 2304;
        let error = resolver
            .resolve(&requirement)
            .expect_err("the b12x kernel is compiled at 576");
        assert!(
            error
                .reasons
                .iter()
                .any(|reason| reason.contains("does not equal 576")),
            "{error}"
        );
    }

    #[test]
    fn an_embedded_policy_refuses_an_image_this_build_did_not_embed() {
        if Image::Qwen4Exp.available() {
            return;
        }
        let resolver = KernelResolver::new(Target::Sm121, Availability::Embedded);
        let error = resolver
            .resolve(&requirement(
                Part::GqaAttention,
                PartGeometry {
                    hidden: 5120,
                    heads: 8,
                    kv_heads: 2,
                    head_dim: 128,
                    world: 1,
                    rank: 0,
                    ..PartGeometry::new()
                },
                Precision::Bf16,
            ))
            .expect_err("no embedded image");
        assert!(!error.reasons.is_empty());
    }

    #[test]
    fn a_wrong_target_has_no_candidates() {
        let resolver = KernelResolver::new(Target::Sm120, Availability::Catalogue);
        let error = resolver
            .resolve(&requirement(
                Part::RmsNorm,
                PartGeometry {
                    hidden: 5120,
                    world: 1,
                    rank: 0,
                    ..PartGeometry::new()
                },
                Precision::Bf16,
            ))
            .expect_err("sm120 is empty");
        assert!(error.reasons.is_empty());
    }
}

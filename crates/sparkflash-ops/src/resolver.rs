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

//! The kernel resolver: [Part] + target + shape + precision -> a loadable [PartKernel].
//!
//! The catalogue answers "which symbols exist in this tree". The resolver answers the question
//! a boot path actually needs: "which of them can this process launch for this checkpoint, on
//! this target, at this rank's shapes?" A part with no matching entry is refused by name, with
//! every candidate's reason, before a weight is read.
//!
//! Availability is an explicit policy:
//!
//! - [Availability::Catalogue] accepts every catalogued implementation (planning and tests);
//! - [Availability::Embedded] accepts only images this build embedded;
//! - [Availability::Runtime] adds the images an arm supplies at run time (DeepSeek's cubin
//!   directory, the RoCEnante collective).

use std::fmt;

use sparkflash_kernels::images::Image;
use sparkflash_model::{Part, Target};

use crate::engine_plan::{EnginePlan, PartRequirement};
use crate::parts::{CATALOGUE, PartKernel};
use crate::plan::{KernelPhase, PartGeometry, Precision};

/// Which implementations a resolver treats as loadable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Availability {
    /// Every catalogue entry, whether or not this build embeds its image.
    Catalogue,
    /// Only images embedded in this binary.
    Embedded,
    /// Embedded images plus those an arm supplies at run time.
    Runtime,
}

impl Availability {
    /// Whether `image` is loadable under this policy.
    #[must_use]
    pub fn loadable(self, image: Image) -> bool {
        match self {
            Self::Catalogue => true,
            Self::Embedded => image.available(),
            Self::Runtime => {
                image.available() || matches!(image, Image::Deepseek | Image::Rocenante)
            }
        }
    }

    /// The policy's stable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Catalogue => "catalogue",
            Self::Embedded => "embedded",
            Self::Runtime => "runtime",
        }
    }
}

/// A part resolved to one catalogue implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedKernel {
    /// The model-side part.
    pub part: Part,
    /// The target the implementation runs on.
    pub target: Target,
    /// The image that carries the symbols.
    pub image: Image,
    /// The symbols the image exports for this part.
    pub symbols: &'static [&'static str],
    /// Whether this is the canonical implementation for the part and target.
    pub canonical: bool,
    /// The rank-local geometry that was resolved.
    pub geometry: PartGeometry,
    /// The precision that was resolved.
    pub precision: Precision,
    /// The pass the resolved implementation serves.
    pub phase: KernelPhase,
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
    /// Why each candidate was refused, or that there were no candidates.
    pub reasons: Vec<String>,
}

impl fmt::Display for ResolveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} on {} at {}: {}",
            self.part,
            self.target,
            self.precision.name(),
            if self.reasons.is_empty() {
                "no catalogue entry".to_owned()
            } else {
                self.reasons.join("; ")
            }
        )
    }
}

impl std::error::Error for ResolveError {}

/// The model family an image belongs to, if it carries one family's kernels.
///
/// A shared image (`Dflash2`, `Rocenante`) has no family and competes on the canonical flag
/// alone. The names are the released `model_type` values, the same string
/// [sparkflash_model::Architecture::family] holds.
#[must_use]
fn image_families(image: Image) -> &'static [&'static str] {
    match image {
        // The Qwen token/norm/scan images are shared with GLM; the first family owns the
        // image, the second is a served family. The resolver prefers an owned image over a
        // shared one.
        Image::Qwen4Exp
        | Image::Qwen4Attn
        | Image::Qwen4Moe
        | Image::Qwen4Prefill
        | Image::Qwen4Scan => &["qwen4_exp", "glm5_next"],
        Image::GlmPrefill | Image::GlmAttention | Image::GlmMoe | Image::GlmDecode => {
            &["glm5_next"]
        }
        Image::Deepseek => &["deepseek_v41"],
        // The planner serves every family, so it competes on the canonical flag alone.
        Image::MoePlan | Image::Dflash2 | Image::Rocenante => &[],
    }
}

/// How well `image` serves `family`: 2 for an owned image, 1 for a shared one, 0 otherwise.
fn family_rank(image: Image, family: &str) -> u8 {
    let families = image_families(image);
    if families.first() == Some(&family) {
        2
    } else if families.contains(&family) {
        1
    } else {
        0
    }
}

/// Resolves execution-plan parts to loadable catalogue entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelResolver {
    target: Target,
    availability: Availability,
    family: Option<String>,
}

impl KernelResolver {
    /// A resolver for `target` with the given availability policy.
    #[must_use]
    pub const fn new(target: Target, availability: Availability) -> Self {
        Self {
            target,
            availability,
            family: None,
        }
    }

    /// Prefer the implementations belonging to `family` over the global canonical entry.
    ///
    /// The catalogue's canonical flag says which implementation a *new target* brings up first;
    /// it is DeepSeek for every shared part. A Qwen or GLM plan must still open its own image,
    /// so the family preference decides between equally loadable candidates. A part with no
    /// family entry falls back to the canonical one.
    #[must_use]
    pub fn with_family(mut self, family: impl Into<String>) -> Self {
        self.family = Some(family.into());
        self
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

    /// Resolve one requirement.
    ///
    /// # Errors
    /// No catalogued implementation for the part/target is loadable and accepts the shape and
    /// precision. The error lists every candidate's reason.
    pub fn resolve(&self, requirement: &PartRequirement) -> Result<ResolvedKernel, ResolveError> {
        self.resolve_for(requirement, KernelPhase::Both)
    }

    /// Resolve one requirement for a specific pass.
    ///
    /// `KernelPhase::Both` is the unconstrained form [Self::resolve] uses. A model with separate
    /// prefill and decode images asks for each phase, so a boot check proves both bodies exist.
    ///
    /// # Errors
    /// As [Self::resolve], with the phase added to the candidate reasons.
    pub fn resolve_for(
        &self,
        requirement: &PartRequirement,
        phase: KernelPhase,
    ) -> Result<ResolvedKernel, ResolveError> {
        let mut reasons = Vec::new();
        let mut chosen: Option<&'static PartKernel> = None;
        let mut chosen_score = (0u8, false);
        // When the catalogue has a family-owned entry for this part, a cross-family image is
        // only a fallback if the family entry is *absent*, not merely unembeddable. Otherwise a
        // Qwen build without cubins could silently resolve its MoE to DeepSeek's image.
        let family_has_entry = self.family.as_deref().is_some_and(|family| {
            CATALOGUE.iter().any(|entry| {
                entry.part == requirement.part
                    && entry.target == self.target
                    && family_rank(entry.image, family) > 0
            })
        });
        let mut candidates = CATALOGUE
            .iter()
            .filter(|entry| entry.part == requirement.part && entry.target == self.target)
            .peekable();
        if candidates.peek().is_none() {
            return Err(ResolveError {
                part: requirement.part,
                target: self.target,
                precision: requirement.precision,
                reasons,
            });
        }
        for entry in candidates {
            let family_match = self
                .family
                .as_deref()
                .is_some_and(|family| family_rank(entry.image, family) > 0);
            if phase != KernelPhase::Both
                && entry.phase != KernelPhase::Both
                && entry.phase != phase
            {
                reasons.push(format!(
                    "{} is the {} pass",
                    entry.image.name(),
                    match entry.phase {
                        KernelPhase::Both => "both-pass",
                        KernelPhase::Prefill => "prefill",
                        KernelPhase::Decode => "decode",
                    }
                ));
                continue;
            }
            if family_has_entry && !family_match {
                reasons.push(format!("{} is another family's image", entry.image.name()));
                continue;
            }
            if !self.availability.loadable(entry.image) {
                reasons.push(format!(
                    "{} is not loadable under {} availability",
                    entry.image.name(),
                    self.availability.name()
                ));
                continue;
            }
            if let Err(reason) = entry
                .shape
                .accepts(&requirement.geometry, requirement.precision)
            {
                reasons.push(format!("{}: {reason}", entry.image.name()));
                continue;
            }
            if !entry.precisions.contains(requirement.precision) {
                reasons.push(format!(
                    "{}: precision {} is not accepted",
                    entry.image.name(),
                    requirement.precision.name()
                ));
                continue;
            }
            let score = (
                self.family
                    .as_deref()
                    .map_or(0, |family| family_rank(entry.image, family)),
                entry.canonical,
            );
            if chosen.is_none() || score > chosen_score {
                chosen = Some(entry);
                chosen_score = score;
            }
        }
        chosen
            .map(|entry| ResolvedKernel {
                part: entry.part,
                target: entry.target,
                image: entry.image,
                symbols: entry.symbols,
                canonical: entry.canonical,
                geometry: requirement.geometry,
                precision: requirement.precision,
                phase: entry.phase,
            })
            .ok_or(ResolveError {
                part: requirement.part,
                target: self.target,
                precision: requirement.precision,
                reasons,
            })
    }

    /// Resolve every part of a plan, in plan order.
    ///
    /// # Errors
    /// Every unresolved part, not merely the first, so a bring-up log names the whole gap.
    pub fn resolve_all(&self, plan: &EnginePlan) -> Result<Vec<ResolvedKernel>, Vec<ResolveError>> {
        let mut resolved = Vec::new();
        let mut errors = Vec::new();
        for requirement in plan.requirements() {
            match self.resolve(requirement) {
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

    /// The unresolved parts of a plan, in plan order.
    #[must_use]
    pub fn failures(&self, plan: &EnginePlan) -> Vec<ResolveError> {
        plan.requirements()
            .filter_map(|requirement| self.resolve(requirement).err())
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

    #[test]
    fn a_constraint_checks_the_fields_it_names() {
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
        constraint.accepts(&okay, Precision::Bf16).expect("valid");

        let bad_head = PartGeometry {
            head_dim: 12,
            ..okay
        };
        assert!(constraint.accepts(&bad_head, Precision::Bf16).is_err());
        let bad_group = PartGeometry {
            heads: 10,
            kv_heads: 2,
            ..okay
        };
        assert!(constraint.accepts(&bad_group, Precision::Bf16).is_err());
    }

    #[test]
    fn a_catalogue_entry_resolves_for_its_target() {
        let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
        let geometry = PartGeometry {
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
        };
        let resolved = resolver
            .resolve(&requirement(Part::Moe, geometry, Precision::NvFp4))
            .expect("the catalogue carries MoE");
        assert_eq!(resolved.part, Part::Moe);
        assert!(resolved.canonical);
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

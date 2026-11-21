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

//! The model scan: one row per part instance, with the reason it did or did not resolve.
//!
//! The scan is the one report the console, `onboard` and the load-time refusal print. It reads
//! a checkpoint's config (and, when the caller has one, its tensor index), plans the model with
//! the existing [EnginePlan], and resolves every requirement for both passes with the existing
//! [KernelResolver]. Step 0 of milestone 1 fills the resolved rows and leaves the two coverage
//! statuses empty; the tensor-claim and config-key passes land on top of the same report.
//!
//! Every refusal names the part, its shape, its stored format and the target, and carries what
//! would fix it. The statuses are the seven of the plan's section 4, no more.

use std::fmt;
use std::fmt::Write as _;

use sparkflash_model::architecture::{Architecture, WeightFormat};
use sparkflash_model::stack::{Shape, StackKind};
use sparkflash_model::{Part, Target};

use crate::engine_plan::{EnginePlan, PartRequirement};
use crate::plan::{KernelPhase, PartGeometry};
use crate::resolver::{Availability, KernelResolver, ResolveCause, ResolveError};

/// One tensor from a checkpoint's index: its name, stored shape and dtype.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    /// The checkpoint tensor name.
    pub name: String,
    /// The stored shape, outermost first.
    pub shape: Vec<usize>,
    /// The stored dtype, as the index spells it.
    pub dtype: String,
}

impl TensorEntry {
    /// An indexed tensor.
    #[must_use]
    pub fn new(name: impl Into<String>, shape: Vec<usize>, dtype: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype: dtype.into(),
        }
    }
}

/// A checkpoint's tensor index, when the caller has read one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TensorIndex {
    /// Every indexed tensor, in the index's order.
    pub tensors: Vec<TensorEntry>,
}

impl TensorIndex {
    /// An index over the given tensors.
    #[must_use]
    pub fn new(tensors: Vec<TensorEntry>) -> Self {
        Self { tensors }
    }
}

/// What the scan reads.
#[derive(Debug, Clone, Copy)]
pub struct ScanInputs<'a> {
    /// The checkpoint's `config.json` text.
    pub config: &'a str,
    /// The tensor index, when the caller has read the checkpoint's index files.
    pub tensors: Option<&'a TensorIndex>,
    /// The GPU target the scan resolves against.
    pub target: Target,
    /// The tensor-parallel world size the plan is built for.
    pub world: usize,
}

/// The status of one row of the scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScanStatus {
    /// A kernel was chosen for the pass.
    Resolved,
    /// A kernel covers the operation, but not this shape or stored format.
    NotBuilt {
        /// What would fix the gap.
        fix: String,
    },
    /// The part kind exists, but this target has no kernel body.
    NoKernelOnTarget {
        /// What would fix the gap.
        fix: String,
    },
    /// The description names an operation the shared op library lacks.
    NoPartKind {
        /// What would fix the gap.
        fix: String,
    },
    /// A config key nothing read.
    UnknownKey {
        /// The unconsumed config key.
        key: String,
        /// What would fix the gap.
        fix: String,
    },
    /// A tensor in the index that no planned part claimed.
    UnclaimedTensor {
        /// The unclaimed tensor name.
        tensor: String,
        /// What would fix the gap.
        fix: String,
    },
    /// The wiring could not be determined without a layout file.
    UnknownLayout {
        /// What the scan could not determine.
        detail: String,
        /// What would fix the gap.
        fix: String,
    },
}

impl ScanStatus {
    /// The stable status name used by the renderer.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Resolved => "resolved",
            Self::NotBuilt { .. } => "not built",
            Self::NoKernelOnTarget { .. } => "no kernel on this target",
            Self::NoPartKind { .. } => "no part kind",
            Self::UnknownKey { .. } => "unknown key",
            Self::UnclaimedTensor { .. } => "unclaimed tensor",
            Self::UnknownLayout { .. } => "unknown layout",
        }
    }

    /// What would fix this gap, or `None` when the row resolved.
    #[must_use]
    pub fn fix(&self) -> Option<&str> {
        match self {
            Self::Resolved => None,
            Self::NotBuilt { fix }
            | Self::NoKernelOnTarget { fix }
            | Self::NoPartKind { fix }
            | Self::UnknownKey { fix, .. }
            | Self::UnclaimedTensor { fix, .. }
            | Self::UnknownLayout { fix, .. } => Some(fix),
        }
    }

    /// Whether this row resolved.
    #[must_use]
    pub const fn is_resolved(&self) -> bool {
        matches!(self, Self::Resolved)
    }
}

impl fmt::Display for ScanStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// One row of the scan: a part instance and what happened to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanRow {
    /// The stack the instance belongs to.
    pub stack: StackKind,
    /// The block index, or `None` for a part outside the decoder blocks.
    pub block: Option<usize>,
    /// The part kind.
    pub part: Part,
    /// The rank-local shape the instance was resolved at.
    pub shape: Shape,
    /// The stored weight format.
    pub format: WeightFormat,
    /// The pass this row resolves.
    pub pass: KernelPhase,
    /// The target this row resolves against.
    pub target: Target,
    /// The row's status.
    pub status: ScanStatus,
}

/// The complete scan of one checkpoint at one target and world size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanReport {
    /// The target every row resolves against.
    pub target: Target,
    /// The tensor-parallel world size the plan was built for.
    pub world: usize,
    /// The rows, globals first and then the blocks in forward order.
    pub rows: Vec<ScanRow>,
    /// A failure that stopped the scan before it could produce rows.
    pub error: Option<String>,
}

impl ScanReport {
    /// The rows that did not resolve.
    pub fn unresolved(&self) -> impl Iterator<Item = &ScanRow> {
        self.rows.iter().filter(|row| !row.status.is_resolved())
    }

    /// Whether the scan ran and every row resolved.
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        self.error.is_none() && self.rows.iter().all(|row| row.status.is_resolved())
    }

    /// The plain-text report the console, `onboard` and the load refusal print.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "scan report: {} TP{}", self.target.name(), self.world);
        if let Some(error) = &self.error {
            let _ = writeln!(out, "scan failed: {error}");
        }
        let _ = writeln!(
            out,
            "{:<9} {:<6} {:<18} {:<16} {:<8} {:<8} status",
            "stack", "block", "part", "shape", "format", "pass"
        );
        for row in &self.rows {
            let block = row
                .block
                .map_or_else(|| "global".to_owned(), |index| index.to_string());
            let shape = row.shape.to_string();
            let _ = writeln!(
                out,
                "{:<9} {:<6} {:<18} {:<16} {:<8} {:<8} {}",
                row.stack.name(),
                block,
                row.part.name(),
                shape,
                row.format.name(),
                row.pass.name(),
                row.status.name()
            );
            if let Some(fix) = row.status.fix() {
                let _ = writeln!(out, "    fix: {fix}");
            }
        }
        let resolved = self
            .rows
            .iter()
            .filter(|row| row.status.is_resolved())
            .count();
        let _ = writeln!(
            out,
            "{} rows, {} resolved, {} unresolved",
            self.rows.len(),
            resolved,
            self.rows.len() - resolved
        );
        out
    }
}

impl fmt::Display for ScanReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.render())
    }
}

/// Scan a checkpoint's config against the kernel catalogue.
///
/// The plan is built at rank zero of `inputs.world`; a scan describes the checkpoint, not one
/// rank's slice, so rank zero is the representative the report prints. A config that does not
/// parse or plan is reported in [ScanReport::error] with no rows; it is never silently empty.
#[must_use]
pub fn scan(inputs: ScanInputs<'_>) -> ScanReport {
    let mut report = ScanReport {
        target: inputs.target,
        world: inputs.world,
        rows: Vec::new(),
        error: None,
    };
    let plan = match EnginePlan::from_json(inputs.config, inputs.target, inputs.world, 0) {
        Ok(plan) => plan,
        Err(error) => {
            report.error = Some(error.to_string());
            return report;
        }
    };
    // The coverage passes are deliberately not wired in yet: step 0 lands the rows the plan
    // already knows. When they land they read this index and append the two coverage statuses.
    let _ = inputs.tensors;
    let resolver = KernelResolver::new(inputs.target, Availability::Catalogue);
    let architecture = plan.architecture();
    for requirement in plan.globals() {
        push_rows(
            &mut report,
            architecture,
            &resolver,
            None,
            stack_of(requirement.part),
            requirement,
        );
    }
    for layer in plan.layers() {
        for requirement in &layer.requirements {
            push_rows(
                &mut report,
                architecture,
                &resolver,
                Some(layer.index),
                StackKind::Language,
                requirement,
            );
        }
    }
    report
}

fn push_rows(
    report: &mut ScanReport,
    architecture: &Architecture,
    resolver: &KernelResolver,
    block: Option<usize>,
    stack: StackKind,
    requirement: &PartRequirement,
) {
    let shape = part_shape(requirement.part, &requirement.geometry);
    let format = architecture.stored_format(requirement.part);
    for pass in [KernelPhase::Prefill, KernelPhase::Decode] {
        let status = match resolver.resolve_for(requirement, pass) {
            Ok(_) => ScanStatus::Resolved,
            Err(error) => refusal_status(&error, &shape, format),
        };
        report.rows.push(ScanRow {
            stack,
            block,
            part: requirement.part,
            shape: shape.clone(),
            format,
            pass,
            target: resolver.target(),
            status,
        });
    }
}

fn refusal_status(error: &ResolveError, shape: &Shape, format: WeightFormat) -> ScanStatus {
    match &error.cause {
        ResolveCause::NoCatalogueEntry => ScanStatus::NoPartKind {
            fix: format!(
                "add a `{}` part kind to the shared op library",
                error.part.name()
            ),
        },
        ResolveCause::ShapeOrFormatRefused => ScanStatus::NotBuilt {
            fix: format!(
                "add a build-list line for `{}` at {} stored as {} on {} and rebuild",
                error.part.name(),
                shape,
                format.name(),
                error.target.name()
            ),
        },
        ResolveCause::OtherTargetOnly { targets } => {
            let targets = targets
                .iter()
                .map(|target| target.name())
                .collect::<Vec<_>>()
                .join(", ");
            ScanStatus::NoKernelOnTarget {
                fix: format!(
                    "port `{}` to {} (the catalogue has it for {targets})",
                    error.part.name(),
                    error.target.name()
                ),
            }
        }
    }
}

fn stack_of(part: Part) -> StackKind {
    match part {
        Part::Vision => StackKind::Vision,
        Part::Mtp => StackKind::Draft,
        _ => StackKind::Language,
    }
}

fn part_shape(part: Part, geometry: &PartGeometry) -> Shape {
    let dims = match part {
        Part::Embedding | Part::LmHead => vec![geometry.vocab, geometry.hidden],
        Part::RmsNorm | Part::LayerNorm => vec![geometry.hidden],
        Part::MlaAttention | Part::GqaAttention | Part::GdnScan | Part::KdaScan => {
            vec![geometry.heads, geometry.head_dim]
        }
        Part::DenseMlp => vec![geometry.intermediate, geometry.hidden],
        Part::Moe => vec![geometry.experts, geometry.intermediate, geometry.hidden],
        Part::Vision => vec![geometry.patch_size, geometry.patch_size, geometry.hidden],
        Part::Mtp => vec![geometry.rows, geometry.hidden],
        _ => vec![geometry.hidden],
    };
    Shape::new(dims.into_iter().filter(|dim| *dim > 0).collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"{"model_type":"","hidden_size":5120,"intermediate_size":256,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":128,"vocab_size":128,"max_position_embeddings":16}"#;

    fn inputs(target: Target) -> ScanInputs<'static> {
        ScanInputs {
            config: CONFIG,
            tensors: None,
            target,
            world: 1,
        }
    }

    #[test]
    fn a_supported_plan_resolves_and_the_coverage_statuses_stay_empty() {
        let report = scan(inputs(Target::Sm121));
        assert!(report.error.is_none(), "{report:?}");
        assert!(!report.rows.is_empty());
        assert!(report.is_resolved(), "{}", report.render());
        assert!(!report.rows.iter().any(|row| matches!(
            row.status,
            ScanStatus::UnknownKey { .. }
                | ScanStatus::UnclaimedTensor { .. }
                | ScanStatus::UnknownLayout { .. }
        )));
        let text = report.render();
        assert!(text.contains("sm121"));
        assert!(text.contains("gqa_attention"));
        assert_eq!(text, report.to_string());
    }

    #[test]
    fn another_target_is_a_missing_kernel_not_a_missing_part() {
        let report = scan(inputs(Target::Sm100));
        assert!(report.error.is_none(), "{report:?}");
        assert!(!report.rows.is_empty());
        for row in &report.rows {
            assert!(
                matches!(row.status, ScanStatus::NoKernelOnTarget { .. }),
                "{:?}",
                row.status
            );
            assert!(row.status.fix().is_some());
        }
    }

    #[test]
    fn a_config_that_does_not_plan_is_reported_not_silently_empty() {
        let report = scan(ScanInputs {
            config: "{",
            tensors: None,
            target: Target::Sm121,
            world: 1,
        });
        assert!(report.rows.is_empty());
        assert!(report.error.is_some());
        assert!(report.render().contains("scan failed"));
    }
}

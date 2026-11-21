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

//! The model scan: one row per part instance, with the reason it resolved or did not.
//!
//! The scan is the one report the console, `onboard` and the load-time refusal print. It reads
//! a checkpoint's config, plans the model with the existing [EnginePlan], and resolves each part
//! instance for the prefill and decode passes with the existing [KernelResolver]. Each instance
//! is one row: it resolves only when a kernel is chosen for both passes, and a gap names the
//! pass that refused it.
//!
//! Step 0 lands the kernel rows and the vocabulary of the two coverage checks. The tensor-claim
//! and config-key passes themselves are not wired yet, so [ScanReport::coverage] records them as
//! [CoverageState::NotChecked] and [ScanReport::is_clean] stays `false` even when every row
//! resolved. A supplied tensor index is counted and printed, never treated as checked.
//!
//! The scan resolves against the catalogue, not against this build's images:
//! [ScanReport::availability] is [Availability::Catalogue] and the report prints it. A caller
//! that gates a load must consult the process's own availability policy; the scan never claims a
//! build embeds an image it did not check.
//!
//! Every refusal names the part, the rank-local shape the plan states, the stored format when the
//! config states one, the pass and the target, and carries what would fix it. The statuses are the seven of the plan's section 4,
//! no more.

use std::fmt;
use std::fmt::Write as _;

use sparkflash_model::architecture::{Architecture, WeightFormat};
use sparkflash_model::stack::{Shape, StackKind};
use sparkflash_model::{Part, Target};

use crate::engine_plan::{EnginePlan, PartRequirement, precision_of};
use crate::plan::{KernelPhase, PartGeometry, Precision};
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

/// Whether one of the two coverage checks of the plan's section 4 has run.
///
/// Step 0 defines the states and leaves both checks [Self::NotChecked]; the tensor-claim and
/// config-key passes land on the same report later. A check that has not run is never reported
/// as clean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverageState {
    /// The check ran and every item was claimed or consumed.
    Clean,
    /// The check has not been wired in yet.
    NotChecked,
}

impl CoverageState {
    /// The stable name used by the renderer.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Clean => "clean",
            Self::NotChecked => "not checked",
        }
    }

    /// Whether this check has run and found nothing to report.
    #[must_use]
    pub const fn is_clean(self) -> bool {
        matches!(self, Self::Clean)
    }
}

impl fmt::Display for CoverageState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// The two checks that make the report trustworthy: every tensor claimed, every key consumed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Coverage {
    /// Whether every indexed tensor was claimed by a planned part.
    pub tensors: CoverageState,
    /// Whether every config key was consumed or listed as ignorable with a reason.
    pub config_keys: CoverageState,
    /// How many tensors the caller supplied, when it supplied an index.
    pub indexed_tensors: Option<usize>,
}

impl Coverage {
    /// A scan whose coverage checks have not run.
    #[must_use]
    pub const fn not_checked(indexed_tensors: Option<usize>) -> Self {
        Self {
            tensors: CoverageState::NotChecked,
            config_keys: CoverageState::NotChecked,
            indexed_tensors,
        }
    }

    /// Whether both coverage checks ran and found nothing to report.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.tensors.is_clean() && self.config_keys.is_clean()
    }
}

/// What the scan reads.
#[derive(Debug, Clone, Copy)]
pub struct ScanInputs<'a> {
    /// The checkpoint's `config.json` text.
    pub config: &'a str,
    /// The tensor index, when the caller has read the checkpoint's index files.
    ///
    /// Step 0 records how many tensors it holds and prints that count; the claim check that
    /// consumes it lands with the layout reader. Passing an index therefore never turns an
    /// unclaimed tensor into a clean report.
    pub tensors: Option<&'a TensorIndex>,
    /// The GPU target the scan resolves against.
    pub target: Target,
    /// The tensor-parallel world size the plan is built for.
    pub world: usize,
}

/// The status of one row of the scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScanStatus {
    /// A kernel was chosen for both passes.
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
    ///
    /// Every [Part] variant has a catalogue entry today, so no current scan produces this row;
    /// a layout file that names an operation outside the library will. It is defined and tested
    /// so the classification is never dropped.
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

    /// The key, tensor or detail this status refuses, when it has one.
    ///
    /// The renderer prints it beside the status name: a refusal that did not name the config key
    /// or tensor would be the silent omission the plan's section 7 forbids.
    #[must_use]
    pub fn subject(&self) -> Option<&str> {
        match self {
            Self::Resolved
            | Self::NotBuilt { .. }
            | Self::NoKernelOnTarget { .. }
            | Self::NoPartKind { .. } => None,
            Self::UnknownKey { key, .. } => Some(key),
            Self::UnclaimedTensor { tensor, .. } => Some(tensor),
            Self::UnknownLayout { detail, .. } => Some(detail),
        }
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
    /// The rank-local principal matrix the instance was resolved at, or [Shape::unknown].
    ///
    /// The plan is built at rank zero of the report's world size, so a TP4 row states the heads
    /// and columns this rank holds, not the model's full width. A part whose operative shape
    /// needs the tensor names and the layout file is printed as `?` rather than guessed.
    pub shape: Shape,
    /// The stored weight format, when the checkpoint's typed fields state it.
    ///
    /// `None` is printed as `?`: a composite part's format is not in the config, and the scan
    /// never reports a format it did not resolve at.
    pub format: Option<WeightFormat>,
    /// The pass this row reports: both when the instance resolved in both passes (or failed in
    /// both), otherwise the single pass that refused it.
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
    /// The availability policy the rows were resolved under.
    pub availability: Availability,
    /// The two coverage checks the plan requires.
    pub coverage: Coverage,
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

    /// Whether the scan ran and every kernel row resolved.
    ///
    /// This is kernel coverage only. A checkpoint is clean when [Self::is_clean] is also true,
    /// which additionally requires both coverage checks to have run.
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        self.error.is_none() && self.rows.iter().all(|row| row.status.is_resolved())
    }

    /// Whether the checkpoint is clean: every part resolved and both coverage checks are clean.
    ///
    /// Step 0 never returns true because the coverage checks are not wired in yet. A caller must
    /// not gate a load on [Self::is_resolved] alone.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.is_resolved() && self.coverage.is_complete()
    }

    /// The plain-text report the console, `onboard` and the load refusal print.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "scan report: {} TP{}, availability {}",
            self.target.name(),
            self.world,
            self.availability.name()
        );
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
            let format = row
                .format
                .map_or_else(|| "?".to_owned(), |format| format.name().to_owned());
            let status = match row.status.subject() {
                Some(subject) => format!("{}: {subject}", row.status.name()),
                None => row.status.name().to_owned(),
            };
            let _ = writeln!(
                out,
                "{:<9} {:<6} {:<18} {:<16} {:<8} {:<8} {}",
                row.stack.name(),
                block,
                row.part.name(),
                shape,
                format,
                row.pass.name(),
                status
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
        let tensors = match (self.coverage.tensors, self.coverage.indexed_tensors) {
            (CoverageState::NotChecked, Some(count)) => {
                format!("{} ({count} indexed)", CoverageState::NotChecked.name())
            }
            (state, _) => state.name().to_owned(),
        };
        let _ = writeln!(
            out,
            "coverage: tensors {tensors}; config keys {}",
            self.coverage.config_keys.name()
        );
        let _ = writeln!(out, "clean: {}", if self.is_clean() { "yes" } else { "no" });
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
        availability: Availability::Catalogue,
        coverage: Coverage::not_checked(inputs.tensors.map(|index| index.tensors.len())),
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
    // Step 0 records the index's size; the claim check that consumes its names is the coverage
    // pass that lands on top of this report. Ignoring the index would let a checkpoint with an
    // unclaimed tower print a clean scan, so the report states that the check did not run.
    let resolver = KernelResolver::new(inputs.target, report.availability);
    let architecture = plan.architecture();
    for requirement in plan.globals() {
        push_row(
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
            push_row(
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

fn push_row(
    report: &mut ScanReport,
    architecture: &Architecture,
    resolver: &KernelResolver,
    block: Option<usize>,
    stack: StackKind,
    requirement: &PartRequirement,
) {
    let shape = part_shape(requirement.part, &requirement.geometry);
    let format = architecture.stored_format(requirement.part);
    let prefill = resolve_pass(resolver, requirement, KernelPhase::Prefill, format);
    let decode = resolve_pass(resolver, requirement, KernelPhase::Decode, format);
    let (pass, status) = match (prefill, decode) {
        (Ok(_), Ok(_)) => (KernelPhase::Both, ScanStatus::Resolved),
        (Err(error), Ok(_)) => (KernelPhase::Prefill, refusal_status(&error, &shape, format)),
        (Ok(_), Err(error)) => (KernelPhase::Decode, refusal_status(&error, &shape, format)),
        (Err(error), Err(_)) => (KernelPhase::Both, refusal_status(&error, &shape, format)),
    };
    report.rows.push(ScanRow {
        stack,
        block,
        part: requirement.part,
        shape,
        format,
        pass,
        target: resolver.target(),
        status,
    });
}

/// Resolve one instance for one pass at the format its row reports.
///
/// A stated format is resolved at exactly that precision, so the format column and the kernel
/// lookup cannot disagree. An unstated format is tried at the plan's precision and then at every
/// other: the row says a body exists without claiming which stored format it reads.
fn resolve_pass(
    resolver: &KernelResolver,
    requirement: &PartRequirement,
    phase: KernelPhase,
    format: Option<WeightFormat>,
) -> Result<(), ResolveError> {
    if let Some(format) = format {
        let requirement = PartRequirement {
            precision: precision_of(format),
            ..*requirement
        };
        return resolver.resolve_for(&requirement, phase).map(|_| ());
    }
    let mut plan_error = None;
    for precision in [
        requirement.precision,
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
        match resolver.resolve_for(&candidate, phase) {
            Ok(_) => return Ok(()),
            Err(error) if precision == requirement.precision => plan_error = Some(error),
            Err(_) => {}
        }
    }
    Err(plan_error.expect("the plan's precision was tried"))
}

fn refusal_status(error: &ResolveError, shape: &Shape, format: Option<WeightFormat>) -> ScanStatus {
    match &error.cause {
        ResolveCause::NoCatalogueEntry => ScanStatus::NoPartKind {
            fix: format!(
                "add a `{}` part kind to the shared op library",
                error.part.name()
            ),
        },
        ResolveCause::ShapeOrFormatRefused => {
            let at = if shape.is_unknown() {
                "at the checkpoint's shape (the config does not state it)".to_owned()
            } else {
                format!("at {shape}")
            };
            let stored = format.map_or_else(
                || "whose stored format the config does not state".to_owned(),
                |format| format!("stored as {}", format.name()),
            );
            ScanStatus::NotBuilt {
                fix: format!(
                    "add a build-list line for `{}` {at} {stored} on {} and rebuild",
                    error.part.name(),
                    error.target.name()
                ),
            }
        }
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

/// The rank-local principal matrix the plan sizes for one part, or [Shape::unknown].
///
/// Only the parts whose operative matrix the checkpoint's typed fields size are stated. A part
/// whose shape needs the tensor index and the layout file (an Engram table, a hyper-connection
/// seam, an indexer) is unknown: the report never substitutes the residual width for a shape it
/// does not know. A partially unset geometry is unknown rather than a lower-rank shape.
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
        _ => Vec::new(),
    };
    if dims.is_empty() || dims.contains(&0) {
        Shape::unknown()
    } else {
        Shape::new(dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::Precision;

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
        let plan = EnginePlan::from_json(CONFIG, Target::Sm121, 1, 0).expect("the plan");
        assert_eq!(report.rows.len(), plan.requirements().count());
        assert!(report.rows.iter().all(|row| row.pass == KernelPhase::Both));
        assert_eq!(report.availability, Availability::Catalogue);
        assert_eq!(report.coverage.tensors, CoverageState::NotChecked);
        assert_eq!(report.coverage.config_keys, CoverageState::NotChecked);
        assert!(!report.is_clean(), "step 0 cannot call a checkpoint clean");
        let text = report.render();
        assert!(text.contains("coverage: tensors not checked"), "{text}");
        assert!(text.contains("clean: no"), "{text}");
    }

    #[test]
    fn the_report_renders_the_exact_table_a_work_item_is_taken_from() {
        let expected = concat!(
            "scan report: sm121 TP1, availability catalogue\n",
            "stack     block  part               shape            format   pass     status\n",
            "language  global embedding          128x5120         BF16     both     resolved\n",
            "language  global rms_norm           5120             BF16     both     resolved\n",
            "language  global lm_head            128x5120         BF16     both     resolved\n",
            "language  0      gqa_attention      1x128            BF16     both     resolved\n",
            "language  0      dense_mlp          256x5120         BF16     both     resolved\n",
            "5 rows, 5 resolved, 0 unresolved\n",
            "coverage: tensors not checked; config keys not checked\n",
            "clean: no\n",
        );
        assert_eq!(scan(inputs(Target::Sm121)).render(), expected);
    }

    #[test]
    fn a_supplied_tensor_index_is_counted_and_never_turns_the_scan_clean() {
        let index = TensorIndex::new(vec![TensorEntry::new(
            "model.visual.nothing_claims_this",
            vec![16, 16],
            "BF16",
        )]);
        let report = scan(ScanInputs {
            tensors: Some(&index),
            ..inputs(Target::Sm121)
        });
        assert!(report.error.is_none(), "{report:?}");
        assert!(report.is_resolved());
        assert!(!report.is_clean());
        assert_eq!(report.coverage.indexed_tensors, Some(1));
        assert_eq!(report.coverage.tensors, CoverageState::NotChecked);
        let text = report.render();
        assert!(text.contains("tensors not checked (1 indexed)"), "{text}");
        assert!(text.contains("clean: no"), "{text}");
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
            assert_eq!(row.pass, KernelPhase::Both);
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

    #[test]
    fn a_tower_with_no_built_body_is_not_built_by_the_scan() {
        const VISION_CONFIG: &str = r#"{"model_type":"","hidden_size":5120,"intermediate_size":256,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":128,"vocab_size":128,"max_position_embeddings":16,"vision_config":{"depth":1,"hidden_size":64,"num_attention_heads":2,"intermediate_size":128,"patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2}}"#;
        let report = scan(ScanInputs {
            config: VISION_CONFIG,
            tensors: None,
            target: Target::Sm121,
            world: 1,
        });
        assert!(report.error.is_none(), "{report:?}");
        let vision = report
            .rows
            .iter()
            .find(|row| row.part == Part::Vision)
            .expect("the tower is a part instance");
        assert!(
            matches!(vision.status, ScanStatus::NotBuilt { .. }),
            "{:?}",
            vision.status
        );
        let fix = vision.status.fix().expect("a fix");
        assert!(fix.contains("build-list line"), "{fix}");
        assert!(!report.is_resolved(), "a 16-patch tower has no body");
    }

    #[test]
    fn a_part_the_config_does_not_size_is_unknown_not_a_guess() {
        let geometry = PartGeometry {
            hidden: 5120,
            intermediate: 256,
            streams: 4,
            compressed_vocab: 8192,
            ..PartGeometry::new()
        };
        for part in [
            Part::Engram,
            Part::HyperConnection,
            Part::SparseIndexer,
            Part::Vision,
        ] {
            assert!(
                part_shape(part, &geometry).is_unknown(),
                "{part} must not be reported at the residual width"
            );
        }
        assert_eq!(
            part_shape(Part::DenseMlp, &geometry),
            Shape::new([256, 5120])
        );
        assert_eq!(part_shape(Part::RmsNorm, &geometry), Shape::new([5120]));
    }

    #[test]
    fn not_built_names_the_shape_and_the_build_list_line() {
        let requirement = PartRequirement {
            part: Part::MlaAttention,
            geometry: PartGeometry {
                hidden: 1,
                heads: 1,
                head_dim: 1,
                ..PartGeometry::new()
            },
            precision: Precision::Bf16,
        };
        let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
        let error = resolver
            .resolve_for(&requirement, KernelPhase::Prefill)
            .expect_err("the catalogue has no one-wide attention");
        let shape = part_shape(requirement.part, &requirement.geometry);
        let status = refusal_status(&error, &shape, Some(WeightFormat::Bf16));
        let ScanStatus::NotBuilt { fix } = &status else {
            panic!("expected not built, got {status}");
        };
        assert_eq!(
            fix,
            "add a build-list line for `mla_attention` at 1x1 stored as BF16 on sm121 and rebuild"
        );
    }

    #[test]
    fn a_partially_unset_geometry_is_unknown_not_lower_rank() {
        let geometry = PartGeometry {
            hidden: 5120,
            ..PartGeometry::new()
        };
        assert!(part_shape(Part::DenseMlp, &geometry).is_unknown());
        assert!(part_shape(Part::GqaAttention, &geometry).is_unknown());
    }

    #[test]
    fn a_tower_shape_the_catalogue_covers_resolves_with_an_unstated_format() {
        const VISION_CONFIG: &str = r#"{"model_type":"","hidden_size":5120,"intermediate_size":256,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":128,"vocab_size":128,"max_position_embeddings":16,"vision_config":{"depth":1,"hidden_size":64,"num_attention_heads":2,"intermediate_size":128,"patch_size":14,"temporal_patch_size":2,"spatial_merge_size":2}}"#;
        let report = scan(ScanInputs {
            config: VISION_CONFIG,
            tensors: None,
            target: Target::Sm121,
            world: 1,
        });
        assert!(report.error.is_none(), "{report:?}");
        let vision = report
            .rows
            .iter()
            .find(|row| row.part == Part::Vision)
            .expect("the tower is a part instance");
        assert!(vision.status.is_resolved(), "{:?}", vision.status);
        assert_eq!(
            vision.format, None,
            "the config does not state the tower's format"
        );
        assert_eq!(vision.shape, Shape::new([14, 14, 64]));
        assert!(report.render().contains("14x14x64"), "{report:?}");
    }

    #[test]
    fn a_missing_part_kind_is_named_by_the_scan() {
        let error = ResolveError {
            part: Part::Moe,
            target: Target::Sm121,
            precision: Precision::Bf16,
            phase: KernelPhase::Prefill,
            reasons: Vec::new(),
            cause: ResolveCause::NoCatalogueEntry,
        };
        let status = refusal_status(&error, &Shape::unknown(), None);
        assert_eq!(status.name(), "no part kind");
        assert_eq!(
            status.fix(),
            Some("add a `moe` part kind to the shared op library")
        );
    }

    #[test]
    fn a_coverage_status_renders_the_key_tensor_or_detail_it_refuses() {
        let report = ScanReport {
            target: Target::Sm121,
            world: 1,
            availability: Availability::Catalogue,
            coverage: Coverage::not_checked(None),
            error: None,
            rows: vec![
                ScanRow {
                    stack: StackKind::Language,
                    block: None,
                    part: Part::Embedding,
                    shape: Shape::unknown(),
                    format: None,
                    pass: KernelPhase::Both,
                    target: Target::Sm121,
                    status: ScanStatus::UnknownKey {
                        key: "model.new_switch".to_owned(),
                        fix: "teach the reader".to_owned(),
                    },
                },
                ScanRow {
                    stack: StackKind::Language,
                    block: None,
                    part: Part::Embedding,
                    shape: Shape::unknown(),
                    format: None,
                    pass: KernelPhase::Both,
                    target: Target::Sm121,
                    status: ScanStatus::UnclaimedTensor {
                        tensor: "model.visual.patch.weight".to_owned(),
                        fix: "add a layout entry".to_owned(),
                    },
                },
                ScanRow {
                    stack: StackKind::Language,
                    block: None,
                    part: Part::Embedding,
                    shape: Shape::unknown(),
                    format: None,
                    pass: KernelPhase::Both,
                    target: Target::Sm121,
                    status: ScanStatus::UnknownLayout {
                        detail: "no layout for the tower".to_owned(),
                        fix: "ship a layout file".to_owned(),
                    },
                },
            ],
        };
        let expected = concat!(
            "scan report: sm121 TP1, availability catalogue\n",
            "stack     block  part               shape            format   pass     status\n",
            "language  global embedding          ?                ?        both     unknown key: model.new_switch\n",
            "    fix: teach the reader\n",
            "language  global embedding          ?                ?        both     unclaimed tensor: model.visual.patch.weight\n",
            "    fix: add a layout entry\n",
            "language  global embedding          ?                ?        both     unknown layout: no layout for the tower\n",
            "    fix: ship a layout file\n",
            "3 rows, 0 resolved, 3 unresolved\n",
            "coverage: tensors not checked; config keys not checked\n",
            "clean: no\n",
        );
        assert_eq!(report.render(), expected);
    }

    #[test]
    fn every_status_has_its_name_and_its_fix() {
        let statuses = [
            (ScanStatus::Resolved, "resolved", false),
            (
                ScanStatus::NotBuilt {
                    fix: "f".to_owned(),
                },
                "not built",
                true,
            ),
            (
                ScanStatus::NoKernelOnTarget {
                    fix: "f".to_owned(),
                },
                "no kernel on this target",
                true,
            ),
            (
                ScanStatus::NoPartKind {
                    fix: "f".to_owned(),
                },
                "no part kind",
                true,
            ),
            (
                ScanStatus::UnknownKey {
                    key: "k".to_owned(),
                    fix: "f".to_owned(),
                },
                "unknown key",
                true,
            ),
            (
                ScanStatus::UnclaimedTensor {
                    tensor: "t".to_owned(),
                    fix: "f".to_owned(),
                },
                "unclaimed tensor",
                true,
            ),
            (
                ScanStatus::UnknownLayout {
                    detail: "d".to_owned(),
                    fix: "f".to_owned(),
                },
                "unknown layout",
                true,
            ),
        ];
        assert_eq!(statuses.len(), 7);
        for (status, name, has_fix) in statuses {
            assert_eq!(status.name(), name);
            assert_eq!(status.to_string(), name);
            assert_eq!(status.fix().is_some(), has_fix);
        }
    }
}

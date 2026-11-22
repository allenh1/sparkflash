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
//! pass that refused it. Every row is resolved at exactly the plan's precision, the same request
//! the load path makes; there is no fallback to another precision.
//!
//! The kernel rows are step 0's. When [ScanInputs::tensors] supplies an index, the tensor-claim
//! pass compares it in both directions with [TensorClaims], the description's claim set, and
//! pushes one row per unclaimed group and one per planned tensor the index does not match. The
//! config-key pass always runs: a present key no read consumed is one [ScanStatus::UnknownKey]
//! row. A report is clean only when both checks ran and found nothing and every kernel row
//! resolved.
//!
//! The scan resolves against the catalogue, not against this build's images:
//! [ScanReport::availability] is [Availability::Catalogue] and the report prints it. A caller
//! that gates a load must consult the process's own availability policy; the scan never claims a
//! build embeds an image it did not check.
//!
//! Every refusal names the part, the rank-local shape the plan states, the stored format when the
//! config states one, the pass and the target, and carries what would fix it. The statuses are the
//! seven of the plan's section 4, no more.

use std::collections::BTreeSet;
use std::fmt;
use std::fmt::Write as _;
use std::path::Path;

use sparkflash_model::architecture::{Architecture, WeightFormat};
use sparkflash_model::stack::{Shape, StackKind};
use sparkflash_model::{Part, Target, TpPlan};

use crate::engine_plan::{EnginePlan, PartRequirement, precision_of};
use crate::plan::{KernelPhase, PartGeometry};
use crate::resolver::{Availability, KernelResolver, ResolveCause, ResolveError};
use crate::tensor_plan::{
    TensorClaims, TensorDisagreement, TensorGroup, compare_tensor_claims, group_unclaimed,
    is_tower_tensor, shape_text,
};

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

    /// The logical spec of this stored descriptor, for comparison with a planned tensor.
    ///
    /// The one transformation is safetensors' packed FP4 convention, which stores an E2M1
    /// weight as `I8` with the last dimension halved.
    ///
    /// # Errors
    /// A dtype word this reader does not map, named with the tensor.
    pub fn logical_spec(
        &self,
    ) -> Result<sparkflash_weights::TensorSpec, sparkflash_weights::tensor_index::IndexError> {
        sparkflash_weights::tensor_index::logical_spec(&self.name, &self.dtype, &self.shape)
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

    /// An index from a reader's stored descriptors, preserving the dtype words the headers spell.
    #[must_use]
    pub fn from_reader(index: &sparkflash_weights::tensor_index::TensorIndex) -> Self {
        Self::new(
            index
                .tensors
                .iter()
                .map(|tensor| {
                    TensorEntry::new(
                        tensor.name.clone(),
                        tensor.shape.clone(),
                        tensor.dtype.clone(),
                    )
                })
                .collect(),
        )
    }
}

/// Whether one of the two coverage checks of the plan's section 4 has run.
///
/// The config-key check always runs. The tensor-claim check runs whenever [ScanInputs::tensors]
/// supplies an index; without one it stays [Self::NotChecked]. A check that has not run is never
/// reported as clean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverageState {
    /// The check ran and every item was claimed or consumed.
    Clean,
    /// The check ran and found items to report.
    Dirty,
    /// The check has not been wired in yet.
    NotChecked,
}

impl CoverageState {
    /// The stable name used by the renderer.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Clean => "clean",
            Self::Dirty => "dirty",
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
    /// The tensor index, when the caller has read the checkpoint's safetensors headers.
    ///
    /// The scan compares it in both directions with [TensorClaims]: a planned tensor missing from
    /// the index or stored at another shape or dtype is one row naming it, and an indexed tensor
    /// no part claims is one row per prefix group. Passing no index leaves the check
    /// [CoverageState::NotChecked], never clean.
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
    /// A part of a stack the switch turned off.
    ///
    /// The row is not a gap: the stack is deliberately not loaded, its tensors are still
    /// claimed by the description, and the reason names the switch. It is never the default.
    Off {
        /// Why the stack is off.
        reason: String,
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
            Self::Off { .. } => "off",
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
            Self::Off { .. } => None,
        }
    }

    /// Whether this row is a gap.
    ///
    /// A deliberately off row is not a gap: the report still prints it as off, but a gate that
    /// refuses every row that is not `resolved` must not refuse a stack the operator turned off.
    #[must_use]
    pub const fn is_gap(&self) -> bool {
        !matches!(self, Self::Resolved | Self::Off { .. })
    }

    /// Whether this row resolved or is deliberately off.
    #[must_use]
    pub const fn is_resolved(&self) -> bool {
        !self.is_gap()
    }

    /// Whether the switch turned this row's stack off.
    #[must_use]
    pub const fn is_off(&self) -> bool {
        matches!(self, Self::Off { .. })
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
            Self::Off { reason } => Some(reason),
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
    /// The rows that are gaps: everything that did not resolve and is not deliberately off.
    pub fn unresolved(&self) -> impl Iterator<Item = &ScanRow> {
        self.rows.iter().filter(|row| row.status.is_gap())
    }

    /// The rows the vision switch turned off, if any.
    pub fn off_rows(&self) -> impl Iterator<Item = &ScanRow> {
        self.rows.iter().filter(|row| row.status.is_off())
    }

    /// Why the vision stack is off, when its rows are.
    #[must_use]
    pub fn off_reason(&self) -> Option<&str> {
        self.off_rows().find_map(|row| match &row.status {
            ScanStatus::Off { reason } => Some(reason.as_str()),
            _ => None,
        })
    }

    /// Whether the scan ran with no gap row.
    ///
    /// This is kernel coverage only, and deliberately-off rows are not gaps. A checkpoint is
    /// clean when [Self::is_clean] is also true, which additionally requires both coverage
    /// checks to have run.
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        self.error.is_none() && self.unresolved().next().is_none()
    }

    /// Whether the checkpoint is clean: every part resolved and both coverage checks are clean.
    ///
    /// A caller must not gate a load on [Self::is_resolved] alone: a kernel row can resolve while
    /// the tensor index was never compared, and the config-key check can report an unread key.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.is_resolved() && self.coverage.is_complete()
    }

    /// The rows that block a load.
    ///
    /// An unread config key is a description warning, not a load blocker: the read consumed the
    /// keys the arithmetic needs, and an arm's own typed parser may consume keys the shared
    /// description does not. Every other gap - a missing kernel, an unclaimed or mismatched
    /// tensor, an undetermined layout, an unread key that changes no arithmetic - remains a
    /// blocker. [Self::is_clean] and the rendered `clean:` verdict stay stricter: they still
    /// report the description warning.
    pub fn load_gaps(&self) -> impl Iterator<Item = &ScanRow> {
        self.unresolved()
            .filter(|row| !matches!(row.status, ScanStatus::UnknownKey { .. }))
    }

    /// Whether this checkpoint can be loaded: no load gap and the tensor claims were checked.
    ///
    /// This is [Self::is_clean] without the unread-config-key warning. A report that never
    /// compared its tensor index is not loadable, even when every kernel row resolved.
    #[must_use]
    pub fn load_is_clean(&self) -> bool {
        self.error.is_none()
            && self.load_gaps().next().is_none()
            && self.coverage.tensors.is_clean()
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
            let status = match &row.status {
                // A config key has no tensor shape; say so in the row rather than leaving a
                // bare "?" that reads like a missing shape the index could have supplied.
                ScanStatus::UnknownKey { key, .. } => {
                    format!("unknown key: {key} (a config key, not a tensor)")
                }
                status => match status.subject() {
                    Some(subject) => format!("{}: {subject}", status.name()),
                    None => status.name().to_owned(),
                },
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
            .filter(|row| matches!(row.status, ScanStatus::Resolved))
            .count();
        let off = self.off_rows().count();
        let unresolved = self.unresolved().count();
        let _ = writeln!(
            out,
            "{} rows, {} resolved, {off} off, {unresolved} unresolved",
            self.rows.len(),
            resolved,
        );
        if let Some(reason) = self.off_reason() {
            let _ = writeln!(out, "vision: {reason}");
        }
        let tensors = match (self.coverage.tensors, self.coverage.indexed_tensors) {
            (state, Some(count)) => format!("{} ({count} indexed)", state.name()),
            (state, None) => state.name().to_owned(),
        };
        let _ = writeln!(
            out,
            "coverage: tensors {tensors}; config keys {}",
            self.coverage.config_keys.name()
        );
        let _ = writeln!(out, "clean: {}", if self.is_clean() { "yes" } else { "no" });
        out
    }

    /// The one boot line for this report.
    ///
    /// It is the table's summary without the table: the counts, the vision switch's reason when
    /// the stack is off, the coverage states and the verdict. A serve prints this after a clean
    /// scan so the boot log says a tower is off and why.
    #[must_use]
    pub fn boot_line(&self) -> String {
        let resolved = self
            .rows
            .iter()
            .filter(|row| matches!(row.status, ScanStatus::Resolved))
            .count();
        let off = self.off_rows().count();
        let unresolved = self.unresolved().count();
        let tensors = match (self.coverage.tensors, self.coverage.indexed_tensors) {
            (state, Some(count)) => format!("{} ({count} indexed)", state.name()),
            (state, None) => state.name().to_owned(),
        };
        let mut line = format!(
            "scan {} TP{}: {} rows, {resolved} resolved, {off} off, {unresolved} unresolved; \
             tensors {tensors}; config keys {}; clean: {}; load: {}",
            self.target.name(),
            self.world,
            self.rows.len(),
            self.coverage.config_keys.name(),
            if self.is_clean() { "yes" } else { "no" },
            if self.load_is_clean() { "yes" } else { "no" }
        );
        if let Some(reason) = self.off_reason() {
            let _ = write!(line, "; {reason}");
        }
        if let Some(error) = &self.error {
            let _ = write!(line, "; error: {error}");
        }
        line
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
///
/// The vision switch is read once, here, through
/// [sparkflash_model::profile::vision_enabled]. A stack the switch turns off is printed as off,
/// never as resolved and never as an unstated gap.
#[must_use]
pub fn scan(inputs: ScanInputs<'_>) -> ScanReport {
    scan_impl(
        inputs,
        Availability::Catalogue,
        sparkflash_model::profile::vision_enabled(),
        None,
    )
}

/// [scan] under an explicit availability policy.
///
/// A serving gate passes the policy its process runs under: a device arm uses the run-time
/// images it can actually load, while a host reference that launches no kernel uses
/// [Availability::Catalogue]. The scan still reads the one vision switch.
#[must_use]
pub fn scan_with_availability(inputs: ScanInputs<'_>, availability: Availability) -> ScanReport {
    scan_impl(
        inputs,
        availability,
        sparkflash_model::profile::vision_enabled(),
        None,
    )
}

/// [scan] with the vision switch already resolved.
///
/// This is the seam a test or a caller that has already read the switch uses; it does not read
/// the environment itself, so the helper above remains the only reading.
#[must_use]
pub fn scan_with_switch(
    inputs: ScanInputs<'_>,
    availability: Availability,
    vision: bool,
) -> ScanReport {
    scan_impl(inputs, availability, Ok(vision), None)
}

/// Scan the plan a caller already built, against the checkpoint's documents.
///
/// A pack arm may correct the document's own arithmetic while parsing it (for example, a
/// weight-only quantization bridge). The gate must judge the plan that arm will execute, not a
/// second raw parse, so this entry point takes the plan and reads only the config keys and the
/// tensor index through [ScanInputs]. It reads the one vision switch.
#[must_use]
pub fn scan_plan(
    inputs: ScanInputs<'_>,
    plan: EnginePlan,
    availability: Availability,
) -> ScanReport {
    scan_impl(
        inputs,
        availability,
        sparkflash_model::profile::vision_enabled(),
        Some(plan),
    )
}

/// [scan_plan] with the vision switch already resolved, for a caller that read it once.
#[must_use]
pub fn scan_plan_with_switch(
    inputs: ScanInputs<'_>,
    plan: EnginePlan,
    availability: Availability,
    vision: bool,
) -> ScanReport {
    scan_impl(inputs, availability, Ok(vision), Some(plan))
}

/// The checkpoint's configuration file, the only document outside the safetensors headers a
/// gate reads.
const CHECKPOINT_CONFIG_NAME: &str = "config.json";

/// Scan a checkpoint directory from its `config.json` and safetensors headers only.
///
/// This is the one gate call a serve makes before it opens a weight: the configuration and the
/// headers name every part and tensor, and no weight byte is read. The report is returned even
/// when the documents cannot be read, with [ScanReport::error] set, so a caller prints the same
/// table and refusal for a malformed checkpoint as for an unresolved part.
#[must_use]
pub fn scan_checkpoint(
    directory: &Path,
    target: Target,
    world: usize,
    availability: Availability,
) -> ScanReport {
    let config_path = directory.join(CHECKPOINT_CONFIG_NAME);
    let config = match std::fs::read_to_string(&config_path) {
        Ok(config) => config,
        Err(error) => {
            return ScanReport {
                target,
                world,
                availability,
                coverage: Coverage::not_checked(None),
                rows: Vec::new(),
                error: Some(format!("{}: {error}", config_path.display())),
            };
        }
    };
    let index = match sparkflash_weights::tensor_index::read_local(directory) {
        Ok(index) => TensorIndex::from_reader(&index),
        Err(error) => {
            return ScanReport {
                target,
                world,
                availability,
                coverage: Coverage::not_checked(None),
                rows: Vec::new(),
                error: Some(format!(
                    "{}: {error}; the gate needs the safetensors headers, not the weights",
                    directory.display()
                )),
            };
        }
    };
    scan_with_availability(
        ScanInputs {
            config: &config,
            tensors: Some(&index),
            target,
            world,
        },
        availability,
    )
}

fn scan_impl(
    inputs: ScanInputs<'_>,
    availability: Availability,
    vision: Result<bool, sparkflash_model::profile::ProfileError>,
    plan: Option<EnginePlan>,
) -> ScanReport {
    let mut report = ScanReport {
        target: inputs.target,
        world: inputs.world,
        availability,
        coverage: Coverage::not_checked(inputs.tensors.map(|index| index.tensors.len())),
        rows: Vec::new(),
        error: None,
    };
    let vision = match vision {
        Ok(vision) => vision,
        Err(error) => {
            report.error = Some(format!(
                "{}: {error}; the vision stack cannot be turned on or off by an unknown value",
                sparkflash_model::profile::VISION_ENV
            ));
            return report;
        }
    };
    let names: Option<Vec<String>> = inputs.tensors.map(|index| {
        index
            .tensors
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect()
    });
    let (mut parsed_architecture, keys) =
        match Architecture::from_json_with_coverage_and_tensor_names(
            inputs.config,
            names.as_deref(),
        ) {
            Ok(parsed) => parsed,
            Err(error) => {
                report.error = Some(error.to_string());
                return report;
            }
        };
    if let Some(index) = inputs.tensors {
        apply_patch_temporal(&mut parsed_architecture, index);
    }
    let plan_architecture = plan.as_ref().map(|plan| plan.architecture().clone());
    let architecture = plan_architecture.as_ref().unwrap_or(&parsed_architecture);
    let unknown_keys = keys.unknown_keys();
    report.coverage.config_keys = match keys.state() {
        sparkflash_model::coverage::KeyCoverageState::Clean => CoverageState::Clean,
        // A switched-off stack's own keys are not read because its parser does not run. They are
        // part of the stated choice, like its tensors, not an unknown-key gap; a language key
        // nothing read still reports dirty.
        sparkflash_model::coverage::KeyCoverageState::Dirty
            if !vision
                && unknown_keys
                    .iter()
                    .all(|key| key.path.starts_with("vision_config.")) =>
        {
            CoverageState::Clean
        }
        sparkflash_model::coverage::KeyCoverageState::Dirty => CoverageState::Dirty,
        sparkflash_model::coverage::KeyCoverageState::NotChecked => CoverageState::NotChecked,
    };
    for key in unknown_keys {
        let vision = key.path.starts_with("vision_config.");
        report.rows.push(ScanRow {
            stack: if vision {
                StackKind::Vision
            } else {
                StackKind::Language
            },
            block: None,
            part: if vision {
                Part::Vision
            } else {
                Part::Embedding
            },
            shape: Shape::unknown(),
            format: None,
            pass: KernelPhase::Both,
            target: inputs.target,
            status: ScanStatus::UnknownKey {
                key: key.path.clone(),
                fix: key.fix.clone(),
            },
        });
    }
    let plan = match plan {
        Some(plan) => plan,
        None => {
            match EnginePlan::for_architecture(architecture.clone(), inputs.target, inputs.world, 0)
            {
                Ok(plan) => plan,
                Err(error) => {
                    report.error = Some(error.to_string());
                    return report;
                }
            }
        }
    };
    let resolver = KernelResolver::new(inputs.target, availability);
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
    if let Some(spec) = architecture.vision_spec() {
        use sparkflash_model::vision_geometry::VisionFact;
        let unstated = &spec.unstated;
        for row in &mut report.rows {
            if row.stack != StackKind::Vision {
                continue;
            }
            let facts: Vec<VisionFact> = unstated
                .iter()
                .copied()
                .filter(|fact| fact.part() == row.part)
                .collect();
            if facts.is_empty() {
                continue;
            }
            let missing = facts
                .iter()
                .map(|fact| fact.name())
                .collect::<Vec<_>>()
                .join(", ");
            row.status = ScanStatus::UnknownLayout {
                detail: format!("vision_config does not state {missing}"),
                fix: facts[0].fix().to_owned(),
            };
            if facts.contains(&VisionFact::Position) {
                row.shape = Shape::unknown();
            }
            row.format = None;
        }
        // A declared clamp is arithmetic the plan must carry. The vision MLP geometry has no
        // field for it yet, so a kernel that resolves at the right width is still not the right
        // body; say so rather than call the row resolved.
        if let Some(clamp) = spec.activation_clamp {
            for row in &mut report.rows {
                if row.stack == StackKind::Vision
                    && row.part == Part::VisionMlp
                    && row.status.is_resolved()
                {
                    row.status = ScanStatus::UnknownLayout {
                        detail: format!(
                            "vision_config states a {clamp} clamp on the vision MLP, which the \
                             vision MLP geometry does not carry"
                        ),
                        fix: "carry the clamp into the vision MLP geometry and gate a kernel that \
                              applies it"
                            .to_owned(),
                    };
                }
            }
        }
    }
    if let Some(index) = inputs.tensors {
        apply_tensor_coverage(&mut report, architecture, inputs.world, index, vision);
    }
    if !vision {
        mark_vision_off(&mut report);
    }
    report
}

/// Take the patch kernel's frame count from the stored patch projection.
///
/// The config states the processor's frames, but a tower may fold a temporal pair into the
/// stored weight. The projection's input width is the stored arithmetic: `in_channels * frames *
/// patch * patch`. An index that does not carry the projection, or whose width is not a whole
/// number of frames, leaves the config's statement in place.
fn apply_patch_temporal(architecture: &mut Architecture, index: &TensorIndex) {
    let Some(vision) = architecture.vision.as_mut() else {
        return;
    };
    let Some(entry) = index.tensors.iter().find(|tensor| {
        tensor.name.ends_with("patch_embed.proj.weight")
            || tensor.name.ends_with("patch_embedding.weight")
    }) else {
        return;
    };
    let Some(per_frame) = vision
        .in_channels
        .checked_mul(vision.patch_size)
        .and_then(|value| value.checked_mul(vision.patch_size))
    else {
        return;
    };
    // A 2-D projection stores [out, features]; a 5-D one stores [out, channels, frames, patch,
    // patch]. The input width is every axis after the output, for both spellings.
    let Some(features) = entry.shape.get(1..).and_then(|axes| {
        axes.iter()
            .try_fold(1usize, |width, &axis| width.checked_mul(axis))
    }) else {
        return;
    };
    if per_frame == 0 || !features.is_multiple_of(per_frame) {
        return;
    }
    let frames = features / per_frame;
    if frames > 0 {
        vision.patch_temporal = Some(frames);
    }
}

/// Print every row of the vision stack as deliberately off.
///
/// The rows stay in the report with their part and shape, so the table says exactly what is not
/// loaded; only the status changes. The reason is the one the switch documents.
fn mark_vision_off(report: &mut ScanReport) {
    for row in &mut report.rows {
        if row.stack == StackKind::Vision {
            row.status = ScanStatus::Off {
                reason: sparkflash_model::profile::VISION_OFF_REASON.to_owned(),
            };
        }
    }
}

/// The tensor-claim pass: compare the description's claim set with the supplied index.
fn apply_tensor_coverage(
    report: &mut ScanReport,
    architecture: &Architecture,
    world: usize,
    index: &TensorIndex,
    vision: bool,
) {
    report.coverage.indexed_tensors = Some(index.tensors.len());
    let tp = match TpPlan::for_architecture(architecture, world, 0) {
        Ok(tp) => tp,
        Err(error) => {
            report_tensor_plan_failure(
                report,
                index,
                format!("the tensor-parallel plan could not be built: {error}"),
            );
            return;
        }
    };
    let claims = match TensorClaims::from_checkpoint(architecture, tp) {
        Ok(claims) if !claims.is_empty() => claims,
        Ok(_) => {
            report_tensor_plan_failure(
                report,
                index,
                "the shared checkpoint plan claims no tensor".to_owned(),
            );
            return;
        }
        Err(error) => {
            report_tensor_plan_failure(
                report,
                index,
                format!("the shared checkpoint plan could not be built: {error}"),
            );
            return;
        }
    };
    // A plan whose names are not this checkpoint's spelling cannot be compared tensor by
    // tensor: every name would be a disagreement and every index tensor unclaimed. That is one
    // missing layout, not a hundred thousand findings, so it is one row naming the plan.
    let present = {
        let names: BTreeSet<&str> = index
            .tensors
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();
        claims
            .specs()
            .iter()
            .filter(|spec| names.contains(spec.name.as_str()))
            .count()
    };
    if present.saturating_mul(20) < claims.len() {
        report_tensor_plan_failure(
            report,
            index,
            format!(
                "the shared checkpoint plan names {} tensors, but only {present} of them are in \
                 the index; the checkpoint uses a spelling the shared plan does not describe",
                claims.len()
            ),
        );
        return;
    }
    let mut coverage = compare_tensor_claims(&claims, index);
    if !vision {
        // The tower is deliberately not loaded, so its bytes are claimed but not required: a
        // checkpoint whose tower is stored under another spelling, or not stored at all, must
        // not turn a text-only serve into a refusal. The language stack is still checked in
        // both directions.
        coverage
            .disagreements
            .retain(|entry| stack_of_name(&entry.tensor) != StackKind::Vision);
        coverage
            .unclaimed
            .retain(|group| stack_of_name(&group.prefix) != StackKind::Vision);
    }
    report.coverage.tensors = if coverage.is_clean() {
        CoverageState::Clean
    } else {
        CoverageState::Dirty
    };
    for disagreement in &coverage.disagreements {
        push_disagreement(report, disagreement);
    }
    for group in &coverage.unclaimed {
        push_unclaimed(report, group, index);
    }
}

/// A checkpoint whose claim set cannot be built: the tower tensors it does describe are still
/// unclaimed, and the plan failure is one row rather than one row per language tensor.
fn report_tensor_plan_failure(report: &mut ScanReport, index: &TensorIndex, detail: String) {
    report.coverage.tensors = CoverageState::Dirty;
    report.rows.push(ScanRow {
        stack: StackKind::Language,
        block: None,
        part: Part::Embedding,
        shape: Shape::unknown(),
        format: None,
        pass: KernelPhase::Both,
        target: report.target,
        status: ScanStatus::UnknownLayout {
            detail,
            fix: "add the checkpoint's stack or layout entries to the shared description so its \
                  tensors can be planned and claimed"
                .to_owned(),
        },
    });
    let tower: Vec<TensorEntry> = index
        .tensors
        .iter()
        .filter(|entry| is_tower_tensor(&entry.name))
        .cloned()
        .collect();
    for group in group_unclaimed(&tower) {
        push_unclaimed(report, &group, index);
    }
}

/// One planned tensor the checkpoint does not match.
fn push_disagreement(report: &mut ScanReport, disagreement: &TensorDisagreement) {
    let found = match (&disagreement.found_dtype, &disagreement.found_shape) {
        (Some(dtype), Some(shape)) => format!("{} {}", dtype, shape_text(shape)),
        _ => "absent from the index".to_owned(),
    };
    let expected = format!(
        "{} {}",
        disagreement.expected_dtype,
        shape_text(&disagreement.expected_shape)
    );
    let detail = format!(
        "planned tensor {} is {found}; the plan expects {expected} ({})",
        disagreement.tensor, disagreement.reason
    );
    let fix = format!(
        "fix the checkpoint or the shared tensor plan so {} is {expected}",
        disagreement.tensor
    );
    let stack = stack_of_name(&disagreement.tensor);
    report.rows.push(ScanRow {
        stack,
        block: block_of_name(&disagreement.tensor),
        part: part_of_stack(stack),
        shape: if disagreement.expected_shape.is_empty() {
            Shape::unknown()
        } else {
            Shape::new(disagreement.expected_shape.clone())
        },
        format: None,
        pass: KernelPhase::Both,
        target: report.target,
        status: ScanStatus::UnknownLayout { detail, fix },
    });
}

/// One unclaimed prefix group.
fn push_unclaimed(report: &mut ScanReport, group: &TensorGroup, index: &TensorIndex) {
    let stack = stack_of_name(&group.prefix);
    // The row names a tensor (or the first tensor of a prefix group), so print the shape the
    // checkpoint stores for it. The index is the authority; a name it does not hold is unknown.
    let shape = group
        .names
        .iter()
        .find_map(|name| index.tensors.iter().find(|tensor| &tensor.name == name))
        .map_or_else(Shape::unknown, |tensor| Shape::new(tensor.shape.clone()));
    let fix = if group.count() == 1 {
        format!(
            "the checkpoint tensor {} is not claimed by any planned part; add the stack or layout \
             entry that takes it",
            group.prefix
        )
    } else {
        format!(
            "add the stack or layout entry that takes {} ({} checkpoint tensors)",
            group.prefix,
            group.count()
        )
    };
    report.rows.push(ScanRow {
        stack,
        block: block_of_name(&group.prefix),
        part: part_of_stack(stack),
        shape,
        format: None,
        pass: KernelPhase::Both,
        target: report.target,
        status: ScanStatus::UnclaimedTensor {
            tensor: group.label(),
            fix,
        },
    });
}

/// The stack a checkpoint tensor's name belongs to, from the name's own path segments.
fn stack_of_name(name: &str) -> StackKind {
    let parts: Vec<&str> = name.split('.').collect();
    if parts
        .iter()
        .any(|part| matches!(*part, "vision" | "visual" | "aligner"))
    {
        StackKind::Vision
    } else if parts.contains(&"audio") {
        StackKind::Audio
    } else if parts.contains(&"mtp") {
        StackKind::Draft
    } else {
        StackKind::Language
    }
}

/// The representative part of a coverage row: the tower's own part, the draft stage's, or the
/// embedding for a language tensor no part took.
fn part_of_stack(stack: StackKind) -> Part {
    match stack {
        StackKind::Vision | StackKind::Audio => Part::Vision,
        StackKind::Draft => Part::Mtp,
        StackKind::Language => Part::Embedding,
    }
}

/// The first block index a tensor name spells, when it spells one.
fn block_of_name(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for pair in parts.windows(2) {
        if matches!(pair[0], "layers" | "blocks")
            && let Ok(index) = pair[1].parse()
        {
            return Some(index);
        }
    }
    None
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
    // Report a format only when it is the precision this row is actually resolved at. A config
    // that states a different format, or none, gets "?" rather than a claim the lookup did not
    // use.
    let format = architecture
        .stored_format(requirement.part)
        .filter(|format| precision_of(*format) == requirement.precision);
    // Resolve at the plan's precision, exactly as the load path does. Falling back to another
    // precision would report a part resolved that the plan still refuses.
    let prefill = resolver.resolve_for(requirement, KernelPhase::Prefill);
    let decode = resolver.resolve_for(requirement, KernelPhase::Decode);
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
                || {
                    format!(
                        "at the {} precision (the stored format is not in the config)",
                        error.precision.name()
                    )
                },
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
        Part::VisionPatchEmbed
        | Part::VisionPosition
        | Part::VisionNorm
        | Part::VisionAttention
        | Part::VisionMlp
        | Part::VisionDownsample
        | Part::VisionMerger => StackKind::Vision,
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
        Part::VisionPatchEmbed => geometry
            .in_channels
            .checked_mul(geometry.temporal_patch)
            .and_then(|features| features.checked_mul(geometry.patch_size))
            .and_then(|features| features.checked_mul(geometry.patch_size))
            .map_or_else(Vec::new, |features| vec![geometry.hidden, features]),
        Part::VisionPosition if geometry.positions > 0 => {
            vec![geometry.positions, geometry.hidden]
        }
        Part::VisionPosition => vec![geometry.head_dim / 2],
        Part::VisionNorm => vec![geometry.hidden],
        Part::VisionAttention => vec![geometry.heads, geometry.head_dim],
        Part::VisionMlp => vec![geometry.intermediate, geometry.hidden],
        Part::VisionDownsample => geometry
            .hidden
            .checked_mul(geometry.spatial_merge)
            .and_then(|merged| merged.checked_mul(geometry.spatial_merge))
            .map_or_else(Vec::new, |merged| vec![merged, geometry.hidden]),
        Part::VisionMerger => geometry
            .hidden
            .checked_mul(geometry.spatial_merge)
            .and_then(|merged| merged.checked_mul(geometry.spatial_merge))
            .map_or_else(Vec::new, |merged| vec![geometry.out_hidden, merged]),
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
        assert_eq!(report.coverage.config_keys, CoverageState::Clean);
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
            "5 rows, 5 resolved, 0 off, 0 unresolved\n",
            "coverage: tensors not checked; config keys clean\n",
            "clean: no\n",
        );
        assert_eq!(scan(inputs(Target::Sm121)).render(), expected);
    }

    #[test]
    fn an_unread_config_key_is_a_description_warning_not_a_load_blocker() {
        let mut config: serde_json::Value =
            serde_json::from_str(CONFIG).expect("the synthetic config");
        config["an_unread_switch"] = serde_json::json!(true);
        let config = config.to_string();
        let entries = planned_entries(&config, 1);
        let index = TensorIndex::new(entries);
        let report = scan(ScanInputs {
            config: &config,
            tensors: Some(&index),
            target: Target::Sm121,
            world: 1,
        });
        assert!(report.error.is_none(), "{report:?}");
        assert_eq!(report.coverage.tensors, CoverageState::Clean);
        assert_eq!(report.coverage.config_keys, CoverageState::Dirty);
        assert!(
            report.rows.iter().any(|row| matches!(
                row.status,
                ScanStatus::UnknownKey { ref key, .. } if key == "an_unread_switch"
            )),
            "{}",
            report.render()
        );
        assert!(!report.is_clean(), "the description warning still shows");
        assert!(report.load_is_clean(), "{}", report.render());
        assert_eq!(report.load_gaps().count(), 0, "{}", report.render());
        let text = report.render();
        assert!(text.contains("unknown key: an_unread_switch"), "{text}");
        assert!(
            text.contains("(a config key, not a tensor)"),
            "the unknown-key row says why its shape is '?': {text}"
        );
        assert!(text.contains("clean: no"), "{text}");
    }

    #[test]
    fn a_supplied_tensor_index_is_checked_and_never_turns_the_scan_clean() {
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
        assert!(!report.is_clean());
        assert!(
            !report.load_is_clean(),
            "an unclaimed tensor still blocks a load"
        );
        assert_eq!(report.coverage.indexed_tensors, Some(1));
        assert_eq!(report.coverage.tensors, CoverageState::Dirty);
        // The one indexed tensor is unclaimed, and the plan that shares no name with the index is
        // one layout row rather than one row per planned tensor.
        let unclaimed: Vec<&ScanRow> = report
            .rows
            .iter()
            .filter(|row| matches!(row.status, ScanStatus::UnclaimedTensor { .. }))
            .collect();
        assert_eq!(unclaimed.len(), 1, "{}", report.render());
        assert_eq!(
            unclaimed[0].status.subject(),
            Some("model.visual.nothing_claims_this")
        );
        assert_eq!(
            unclaimed[0].shape,
            Shape::new([16, 16]),
            "the row prints the stored shape, not '?'"
        );
        let layout: Vec<&ScanRow> = report
            .rows
            .iter()
            .filter(|row| matches!(row.status, ScanStatus::UnknownLayout { .. }))
            .collect();
        assert_eq!(layout.len(), 1, "{}", report.render());
        assert!(
            layout[0]
                .status
                .subject()
                .is_some_and(|detail| detail.contains("only 0 of them are in the index")),
            "{}",
            report.render()
        );
        let text = report.render();
        assert!(text.contains("tensors dirty (1 indexed)"), "{text}");
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
            .find(|row| row.part == Part::VisionPatchEmbed)
            .expect("the tower's patch projection is a part instance");
        assert!(
            matches!(vision.status, ScanStatus::NotBuilt { .. }),
            "{:?}",
            vision.status
        );
        let fix = vision.status.fix().expect("a fix");
        assert!(
            fix.contains("at the bf16 precision"),
            "the refusal must name the precision the plan asked for: {fix}"
        );
        assert!(!report.is_resolved(), "a 16-patch tower has no body");
        assert!(
            !report.load_is_clean(),
            "a missing kernel still blocks a load"
        );
    }

    #[test]
    fn a_switched_off_vision_stack_is_off_not_a_gap() {
        const VISION_CONFIG: &str = r#"{"model_type":"","hidden_size":5120,"intermediate_size":256,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":128,"vocab_size":128,"max_position_embeddings":16,"vision_config":{"depth":1,"hidden_size":64,"num_attention_heads":2,"intermediate_size":128,"patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2}}"#;
        let report = scan_with_switch(
            ScanInputs {
                config: VISION_CONFIG,
                tensors: None,
                target: Target::Sm121,
                world: 1,
            },
            Availability::Catalogue,
            false,
        );
        assert!(report.error.is_none(), "{report:?}");
        let vision: Vec<&ScanRow> = report
            .rows
            .iter()
            .filter(|row| row.stack == StackKind::Vision)
            .collect();
        assert!(!vision.is_empty());
        assert!(
            vision
                .iter()
                .all(|row| matches!(row.status, ScanStatus::Off { .. })),
            "{}",
            report.render()
        );
        assert!(report.is_resolved(), "{}", report.render());
        assert_eq!(
            report.off_reason(),
            Some(sparkflash_model::profile::VISION_OFF_REASON)
        );
        let text = report.render();
        assert!(text.contains("off: the vision stack is off"), "{text}");
        assert!(text.contains("vision: the vision stack is off"), "{text}");
        assert!(!report.is_clean(), "the tensor check did not run");
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
    fn a_coverage_check_that_found_items_is_dirty_not_not_checked() {
        assert_eq!(CoverageState::Clean.name(), "clean");
        assert_eq!(CoverageState::Dirty.name(), "dirty");
        assert_eq!(CoverageState::NotChecked.name(), "not checked");
        assert!(!CoverageState::Dirty.is_clean());
        let coverage = Coverage {
            tensors: CoverageState::Dirty,
            config_keys: CoverageState::Clean,
            indexed_tensors: None,
        };
        assert!(!coverage.is_complete());
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
        const VISION_CONFIG: &str = r#"{"model_type":"","hidden_size":5120,"intermediate_size":256,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":128,"vocab_size":128,"max_position_embeddings":16,"vision_config":{"depth":1,"hidden_size":1024,"num_attention_heads":16,"intermediate_size":2816,"patch_size":14,"temporal_patch_size":1,"out_hidden_size":5120,"projection_intermediate_size":5120,"rope_theta":10000,"downsample_ratio":3}}"#;
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
            .find(|row| row.part == Part::VisionPatchEmbed)
            .expect("the tower's patch projection is a part instance");
        assert!(vision.status.is_resolved(), "{:?}", vision.status);
        assert_eq!(
            vision.format, None,
            "the config does not state the tower's format"
        );
        assert_eq!(vision.shape, Shape::new([1024, 588]));
        assert!(report.render().contains("1024x588"), "{report:?}");
    }

    #[test]
    fn an_unstated_vision_epsilon_is_unknown_layout_not_resolved() {
        const VISION_CONFIG: &str = r#"{"model_type":"","hidden_size":5120,"intermediate_size":256,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":128,"vocab_size":128,"max_position_embeddings":16,"vision_config":{"depth":1,"hidden_size":1024,"num_attention_heads":16,"intermediate_size":2816,"patch_size":14,"temporal_patch_size":1,"out_hidden_size":5120,"projection_intermediate_size":5120,"rope_theta":10000,"downsample_ratio":3}}"#;
        let report = scan(ScanInputs {
            config: VISION_CONFIG,
            tensors: None,
            target: Target::Sm121,
            world: 1,
        });
        let norm = report
            .rows
            .iter()
            .find(|row| row.part == Part::VisionNorm)
            .expect("the norm row");
        let ScanStatus::UnknownLayout { detail, fix } = &norm.status else {
            panic!("an unstated epsilon must not resolve: {}", report.render());
        };
        assert!(detail.contains("norm epsilon"), "{detail}");
        assert!(!fix.is_empty());
        assert!(!report.is_resolved());
    }

    #[test]
    fn a_grid_position_row_uses_the_learned_grid_shape() {
        let geometry = PartGeometry {
            hidden: 1024,
            heads: 16,
            head_dim: 64,
            positions: 2304,
            ..PartGeometry::new()
        };
        assert_eq!(
            part_shape(Part::VisionPosition, &geometry),
            Shape::new([2304, 1024])
        );
        let rotary = PartGeometry {
            positions: 0,
            ..geometry
        };
        assert_eq!(part_shape(Part::VisionPosition, &rotary), Shape::new([32]));
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
            "language  global embedding          ?                ?        both     unknown key: model.new_switch (a config key, not a tensor)\n",
            "    fix: teach the reader\n",
            "language  global embedding          ?                ?        both     unclaimed tensor: model.visual.patch.weight\n",
            "    fix: add a layout entry\n",
            "language  global embedding          ?                ?        both     unknown layout: no layout for the tower\n",
            "    fix: ship a layout file\n",
            "3 rows, 0 resolved, 0 off, 3 unresolved\n",
            "coverage: tensors not checked; config keys not checked\n",
            "clean: no\n",
        );
        assert_eq!(report.render(), expected);
    }

    fn planned_entries(config: &str, world: usize) -> Vec<TensorEntry> {
        let architecture = Architecture::from_json(config).expect("the config parses");
        let tp = TpPlan::for_architecture(&architecture, world, 0).expect("the tp plan");
        let plan = crate::tensor_plan::ArchitectureTensorPlan::checkpoint(&architecture, tp)
            .expect("the checkpoint plan");
        plan.specs()
            .iter()
            .map(|spec| TensorEntry::new(spec.name.clone(), spec.shape.clone(), spec.dtype.name()))
            .collect()
    }

    #[test]
    fn a_missing_tensor_a_mismatched_shape_and_an_extra_tensor_are_all_named() {
        let mut entries = planned_entries(CONFIG, 1);
        entries.retain(|entry| entry.name != "model.embed_tokens.weight");
        for entry in &mut entries {
            if entry.name == "lm_head.weight" {
                entry.shape = vec![1, 1];
            }
        }
        entries.push(TensorEntry::new(
            "model.visual.blocks.0.patch.weight",
            vec![16, 16],
            "BF16",
        ));
        let index = TensorIndex::new(entries);
        let report = scan(ScanInputs {
            tensors: Some(&index),
            ..inputs(Target::Sm121)
        });
        assert!(report.error.is_none(), "{report:?}");
        assert_eq!(report.coverage.tensors, CoverageState::Dirty);

        let missing = report
            .rows
            .iter()
            .find(|row| {
                row.status
                    .subject()
                    .is_some_and(|subject| subject.contains("model.embed_tokens.weight"))
            })
            .expect("the missing tensor is named");
        let ScanStatus::UnknownLayout { detail, fix } = &missing.status else {
            panic!(
                "a missing tensor is a layout refusal, got {:?}",
                missing.status
            );
        };
        assert!(detail.contains("is absent from the index"), "{detail}");
        assert!(fix.contains("model.embed_tokens.weight"), "{fix}");

        let mismatch = report
            .rows
            .iter()
            .find(|row| {
                row.status
                    .subject()
                    .is_some_and(|subject| subject.contains("lm_head.weight"))
            })
            .expect("the shape mismatch is named");
        let ScanStatus::UnknownLayout { detail, .. } = &mismatch.status else {
            panic!(
                "a shape mismatch is a layout refusal, got {:?}",
                mismatch.status
            );
        };
        assert!(detail.contains("BF16 1x1"), "{detail}");
        assert!(detail.contains("expects BF16 128x5120"), "{detail}");

        let unclaimed = report
            .rows
            .iter()
            .filter(|row| matches!(row.status, ScanStatus::UnclaimedTensor { .. }))
            .collect::<Vec<_>>();
        assert_eq!(unclaimed.len(), 1, "{}", report.render());
        assert_eq!(
            unclaimed[0].status.subject(),
            Some("model.visual.blocks.0.patch.weight")
        );
        assert_eq!(unclaimed[0].stack, StackKind::Vision);
        let text = report.render();
        assert!(
            text.contains("unclaimed tensor: model.visual.blocks.0.patch.weight"),
            "{text}"
        );
    }

    #[test]
    fn a_large_unclaimed_tower_is_one_prefix_row_with_its_count() {
        let mut entries = planned_entries(CONFIG, 1);
        for index in 0..346 {
            entries.push(TensorEntry::new(
                format!("model.visual.blocks.{index}.attn.qkv.weight"),
                vec![3, 16, 16],
                "BF16",
            ));
        }
        // A different branch under the same tower: the exact common prefix is `model.visual`.
        entries.push(TensorEntry::new(
            "model.visual.merger.proj.weight",
            vec![16, 16],
            "BF16",
        ));
        let index = TensorIndex::new(entries);
        let report = scan(ScanInputs {
            tensors: Some(&index),
            ..inputs(Target::Sm121)
        });
        assert_eq!(report.coverage.tensors, CoverageState::Dirty);
        let unclaimed = report
            .rows
            .iter()
            .filter(|row| matches!(row.status, ScanStatus::UnclaimedTensor { .. }))
            .collect::<Vec<_>>();
        assert_eq!(unclaimed.len(), 1, "{}", report.render());
        assert_eq!(
            unclaimed[0].status.subject(),
            Some("model.visual.* (347 tensors)")
        );
        assert_eq!(unclaimed[0].stack, StackKind::Vision);
        assert!(
            report
                .render()
                .contains("unclaimed tensor: model.visual.* (347 tensors)"),
            "{}",
            report.render()
        );
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

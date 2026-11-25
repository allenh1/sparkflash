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
//! Every refusal names the part, the row's shape, the stored format the layout's claimed tensors
//! state, the pass and the target, and carries what would fix it. The statuses are the seven of
//! the plan's section 4, no more.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fmt::Write as _;

use sparkflash_model::architecture::{Architecture, WeightFormat};
use sparkflash_model::format::StoredFormat;
use sparkflash_model::stack::{Shape, StackKind};
use sparkflash_model::{Part, Target, TpPlan};

use crate::engine_plan::EnginePlan;
use crate::engine_plan::PartRequirement;
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
    /// The stored shape of the first tensor the part claims in the layout, or [Shape::unknown].
    ///
    /// The shape is the checkpoint's own, the one the index stores, so it is the same at every
    /// TP world. A part with no claimed tensor falls back to the rank-local plan geometry; a part
    /// with neither is printed as `?` rather than guessed.
    pub shape: Shape,
    /// The stored weight format the part's claimed tensors agree on.
    ///
    /// `None` is printed as `?`: the row then has no stored dtype evidence and the scan never
    /// substitutes a format it did not read. The value carries the block, scale kinds and layout,
    /// so the row states the whole format the requirement was resolved at, not only its name.
    pub format: Option<StoredFormat>,
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
    /// Layout facts the reader carried but no shared field applies, by fact name.
    ///
    /// These are reported for the next reader step; they never make a scan dirty.
    pub carried_facts: Vec<String>,
    /// Parts whose arithmetic has no tensor or row of its own, and the row that covers it.
    ///
    /// A fused epilogue is not a part instance the walk schedules, but the arm still launches
    /// it. The note says which scheduled row's arithmetic contains it, so its absence from the
    /// table is stated rather than silent.
    pub coverage_notes: Vec<String>,
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
    /// A caller must not gate a load on [Self::is_resolved] alone: a kernel row can resolve while
    /// the tensor index was never compared, and the config-key check can report an unread key.
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
            (state, Some(count)) => format!("{} ({count} indexed)", state.name()),
            (state, None) => state.name().to_owned(),
        };
        let _ = writeln!(
            out,
            "coverage: tensors {tensors}; config keys {}",
            self.coverage.config_keys.name()
        );
        if self.carried_facts.is_empty() {
            let _ = writeln!(out, "carried facts: 0");
        } else {
            let _ = writeln!(
                out,
                "carried facts: {} ({})",
                self.carried_facts.len(),
                self.carried_facts.join(", ")
            );
        }
        for note in &self.coverage_notes {
            let _ = writeln!(out, "coverage note: {note}");
        }
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
        carried_facts: Vec::new(),
        coverage_notes: Vec::new(),
    };
    let names: Option<Vec<String>> = inputs.tensors.map(|index| {
        index
            .tensors
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect()
    });
    let (mut architecture, keys) = match Architecture::from_json_with_coverage_and_tensor_names(
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
        apply_patch_temporal(&mut architecture, index);
    }
    report.coverage.config_keys = match keys.state() {
        sparkflash_model::coverage::KeyCoverageState::Clean => CoverageState::Clean,
        sparkflash_model::coverage::KeyCoverageState::Dirty => CoverageState::Dirty,
        sparkflash_model::coverage::KeyCoverageState::NotChecked => CoverageState::NotChecked,
    };
    report.carried_facts = keys.carried_facts().map(str::to_owned).collect();
    for key in keys.unknown_keys() {
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
    let plan = match EnginePlan::for_architecture(architecture, inputs.target, inputs.world, 0) {
        Ok(plan) => plan,
        Err(error) => {
            report.error = Some(error.to_string());
            return report;
        }
    };
    let resolver = KernelResolver::new(inputs.target, report.availability);
    let architecture = plan.architecture();
    // A softcapped SwiGLU is the MLP gate/up epilogue, not a tensor-bearing part: it is launched
    // inside the layer's FFN row. The short convolution has tensors of its own and is planned as
    // a row, so its absence needs no note.
    let parts = architecture.parts();
    if parts.contains(Part::SoftcappedActivation) {
        report.coverage_notes.push(
            "softcapped_activation has no tensor of its own: it is the MLP gate/up epilogue and \
             is covered by each layer's ffn row; short_conv is planned as its own row"
                .to_owned(),
        );
    }
    // A grouped-query block inside an MLA stack runs on the MLA stack's base, whose grouped-query
    // part has no body for interleaved prompt chunks yet: the base takes the chunks one at a time.
    if parts.contains(Part::GqaAttention) && parts.contains(Part::MlaAttention) {
        report.coverage_notes.push(
            "gqa_attention inside an MLA stack has no body for interleaved prompt chunks: the \
             base runs this stack's prompt chunks one at a time (one prefill way)"
                .to_owned(),
        );
    }
    // The layout's claimed tensors are the row's evidence: their stored dtypes decide the
    // format and precision, their shapes decide the shape. The plan already resolved the
    // format per part; the stored shape comes from the shared tensor plan's claim for the first
    // tensor the part declares.
    let shapes = claimed_shapes(architecture, &plan);
    for requirement in plan.globals() {
        let block = None;
        push_row(
            &mut report,
            &resolver,
            block,
            stack_of(requirement.part),
            requirement,
            plan.stored_format(requirement.part, block),
            shapes.get(&(stack_of(requirement.part), block, requirement.part)),
        );
    }
    for layer in plan.layers() {
        for requirement in &layer.requirements {
            let block = Some(layer.index);
            push_row(
                &mut report,
                &resolver,
                block,
                StackKind::Language,
                requirement,
                plan.stored_format(requirement.part, block),
                shapes.get(&(StackKind::Language, block, requirement.part)),
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
        apply_layout_vision(&mut report, architecture, index);
        apply_tensor_coverage(&mut report, architecture, inputs.world, index);
    }
    report
}

/// The stored shape of the first tensor each part claims in the layout.
///
/// A part with several tensors has no single shape; the first declaration is the one the layout
/// names first, which is the part's principal matrix (a query projection before its clip
/// scales, a fusion table before a bias). The shape is the checkpoint's stored shape, so it does
/// not depend on the report's TP world.
fn claimed_shapes(
    architecture: &Architecture,
    plan: &EnginePlan,
) -> BTreeMap<(StackKind, Option<usize>, Part), Shape> {
    let mut shapes = BTreeMap::new();
    let Ok(Some(layout)) = sparkflash_model::layout::matching(architecture) else {
        return shapes;
    };
    let Ok(tensors) = layout.tensors(architecture) else {
        return shapes;
    };
    let claims = TensorClaims::from_checkpoint(architecture, plan.tp()).ok();
    let by_name: BTreeMap<&str, &[usize]> = claims
        .as_ref()
        .map(|claims| {
            claims
                .specs()
                .iter()
                .map(|spec| (spec.name.as_str(), spec.shape.as_slice()))
                .collect()
        })
        .unwrap_or_default();
    for tensor in tensors {
        let block = match tensor.stack {
            StackKind::Draft | StackKind::Vision | StackKind::Audio => None,
            StackKind::Language => tensor.block,
        };
        let key = (tensor.stack, block, tensor.part);
        if shapes.contains_key(&key) {
            continue;
        }
        let dims = by_name
            .get(tensor.name.as_str())
            .map(|dims| (*dims).to_vec())
            .unwrap_or_else(|| tensor.shape.clone());
        if dims.is_empty() || dims.contains(&0) {
            continue;
        }
        shapes.insert(key, Shape::new(dims));
    }
    shapes
}

/// Fill a vision row's shape and stored format from the layout and the tensor index.
///
/// The config states the tower's components; the index is the only source for their stored
/// shape and dtype. A component with no tensor of its own (a rotary position scheme) takes the
/// tower's stored format from its sibling tensors, and its shape stays the geometry the spec
/// states.
fn apply_layout_vision(report: &mut ScanReport, architecture: &Architecture, index: &TensorIndex) {
    let Ok(Some(layout)) = sparkflash_model::layout::matching(architecture) else {
        return;
    };
    let Ok(tensors) = layout.tensors(architecture) else {
        return;
    };
    let by_name: std::collections::BTreeMap<&str, &TensorEntry> = index
        .tensors
        .iter()
        .map(|entry| (entry.name.as_str(), entry))
        .collect();
    let tower_format = tensors
        .iter()
        .filter(|tensor| tensor.stack == StackKind::Vision)
        .find_map(|tensor| by_name.get(tensor.name.as_str()))
        .and_then(|entry| stored_format(&entry.dtype, architecture));
    for row in &mut report.rows {
        if row.stack != StackKind::Vision {
            continue;
        }
        let tensor = tensors
            .iter()
            .find(|tensor| tensor.stack == StackKind::Vision && tensor.part == row.part)
            .and_then(|tensor| by_name.get(tensor.name.as_str()));
        match tensor {
            Some(entry) => {
                row.shape = Shape::new(entry.shape.clone());
                if let Some(format) = stored_format(&entry.dtype, architecture) {
                    row.format = Some(format);
                }
            }
            None => {
                row.format = tower_format;
            }
        }
    }
}

/// The stored weight format a dtype word names, from the checkpoint's quantization block.
fn stored_format(dtype: &str, architecture: &Architecture) -> Option<StoredFormat> {
    let format = match dtype.to_ascii_uppercase().as_str() {
        "BF16" | "BFLOAT16" => Some(WeightFormat::Bf16),
        "F32" | "FLOAT32" => Some(WeightFormat::F32),
        "F8_E4M3" | "F8E4M3" | "FP8" => match architecture.quantization.dense_format {
            format @ WeightFormat::Fp8Tile { .. } => Some(format),
            _ => Some(WeightFormat::Bf16),
        },
        "F4_E2M1" | "F4E2M1" | "I8" => match architecture.quantization.expert_format {
            format @ (WeightFormat::NvFp4 { .. } | WeightFormat::MxFp4 { .. }) => Some(format),
            _ => Some(WeightFormat::NvFp4 {
                group_size: 16,
                wide: sparkflash_model::format::NvFp4Wide::Qwen,
            }),
        },
        _ => None,
    };
    format.map(WeightFormat::stored_format)
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

/// The tensor-claim pass: compare the description's claim set with the supplied index.
fn apply_tensor_coverage(
    report: &mut ScanReport,
    architecture: &Architecture,
    world: usize,
    index: &TensorIndex,
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
    let coverage = compare_tensor_claims(&claims, index);
    report.coverage.tensors = if coverage.is_clean() {
        CoverageState::Clean
    } else {
        CoverageState::Dirty
    };
    for disagreement in &coverage.disagreements {
        push_disagreement(report, disagreement);
    }
    for group in &coverage.unclaimed {
        push_unclaimed(report, group);
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
        push_unclaimed(report, &group);
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
fn push_unclaimed(report: &mut ScanReport, group: &TensorGroup) {
    let stack = stack_of_name(&group.prefix);
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
        shape: Shape::unknown(),
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
    resolver: &KernelResolver,
    block: Option<usize>,
    stack: StackKind,
    requirement: &PartRequirement,
    stored: Option<StoredFormat>,
    claimed: Option<&Shape>,
) {
    let shape = claimed
        .cloned()
        .unwrap_or_else(|| part_shape(requirement.part, &requirement.geometry));
    // Report the format the claimed tensors are stored in. A part the layout does not claim falls
    // back to the config's stated format; neither is guessed from the part kind. The format is
    // shown even when it did not resolve: a row whose format has no body says "not built" and
    // names the format, rather than hiding it behind a `?`.
    let format = stored;
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

fn refusal_status(error: &ResolveError, shape: &Shape, format: Option<StoredFormat>) -> ScanStatus {
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
                        "reading {} (the stored format is not in the config)",
                        error.format.describe()
                    )
                },
                |format| format!("stored as {}", format.describe()),
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
            "5 rows, 5 resolved, 0 unresolved\n",
            "coverage: tensors not checked; config keys clean\n",
            "carried facts: 0\n",
            "clean: no\n",
        );
        assert_eq!(scan(inputs(Target::Sm121)).render(), expected);
    }

    #[test]
    fn carried_facts_are_printed_by_name_without_making_the_scan_dirty() {
        let mut report = ScanReport {
            target: Target::Sm121,
            world: 1,
            availability: Availability::Catalogue,
            coverage: Coverage {
                tensors: CoverageState::Clean,
                config_keys: CoverageState::Clean,
                indexed_tensors: Some(0),
            },
            rows: Vec::new(),
            error: None,
            carried_facts: Vec::new(),
            coverage_notes: Vec::new(),
        };
        assert!(report.is_clean(), "the baseline scan is clean");
        report.carried_facts = vec![
            "language.mlp_bias".to_owned(),
            "language.ngpt.alpha_init".to_owned(),
        ];
        let text = report.render();
        assert!(
            text.contains(
                "carried facts: 2 (language.mlp_bias, language.ngpt.alpha_init)
"
            ),
            "{text}"
        );
        assert!(report.is_clean(), "carried facts never fail the load");
        assert!(text.contains("clean: yes"), "{text}");
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
            fix.contains("reading BF16"),
            "the refusal must name the format the plan asked for: {fix}"
        );
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
            format: StoredFormat::BF16,
        };
        let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
        let error = resolver
            .resolve_for(&requirement, KernelPhase::Prefill)
            .expect_err("the catalogue has no one-wide attention");
        let shape = part_shape(requirement.part, &requirement.geometry);
        let status = refusal_status(&error, &shape, Some(StoredFormat::BF16));
        let ScanStatus::NotBuilt { fix } = &status else {
            panic!("expected not built, got {status}");
        };
        assert_eq!(
            fix,
            "add a build-list line for `mla_attention` at 1x1 stored as BF16 (unscaled, row-major) \
             on sm121 and rebuild"
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
            format: StoredFormat::BF16,
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
            carried_facts: Vec::new(),
            coverage_notes: Vec::new(),
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
            "carried facts: 0\n",
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

    #[test]
    fn a_format_with_no_body_is_not_built_and_names_the_format() {
        // MXFP4 is a real stored format (one release stores its routed experts that way) and
        // no head body decodes it. The row must say so and name the format, its block and its
        // scale kind: before the format reached the catalogue key, the requirement was folded
        // to NVFP4 and this lookup resolved against a body that would have misread the weights.
        let requirement = PartRequirement {
            part: Part::LmHead,
            geometry: PartGeometry {
                hidden: 5120,
                vocab: 129_280,
                rows: 1,
                world: 4,
                rank: 0,
                ..PartGeometry::new()
            },
            format: StoredFormat::MXFP4_32,
        };
        let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
        let error = resolver
            .resolve_for(&requirement, KernelPhase::Decode)
            .expect_err("no lm_head body decodes MXFP4");
        let status = refusal_status(
            &error,
            &Shape::new([129_280, 5120]),
            Some(StoredFormat::MXFP4_32),
        );
        let ScanStatus::NotBuilt { fix } = &status else {
            panic!("expected not built, got {status}");
        };
        assert!(fix.contains("stored as MXFP4"), "{fix}");
        assert!(fix.contains("power-of-two grid"), "{fix}");
        assert!(fix.contains("one scale per 32 values"), "{fix}");
    }
}
